#include "gpu/d3d12_xs_dump.h"

#include <cinttypes>
#include <cstdio>
#include <filesystem>
#include <system_error>

#include "gpu/d3d12_guest_resources.h"
#include "gpu/descriptor_heap.h"

namespace playgta6::gpu::d3d12_guest {

namespace {

std::string id_text(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof(b), "0x%016" PRIx64, v);
    return b;
}

std::string json_string(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (static_cast<unsigned char>(c) < 0x20) {
            char b[8];
            std::snprintf(b, sizeof(b), "\\u%04x", static_cast<unsigned char>(c));
            out += b;
        } else {
            out += c;
        }
    }
    return out + "\"";
}

std::string digest_text(const std::array<uint8_t, 16>& d) {
    static const std::array<uint8_t, 16> kZero{};
    if (d == kZero) {
        return "";
    }
    std::string s;
    char b[4];
    for (uint8_t x : d) {
        std::snprintf(b, sizeof(b), "%02x", x);
        s += b;
    }
    return s;
}

std::string num(uint64_t v) {
    return std::to_string(v);
}
std::string inum(int64_t v) {
    return std::to_string(v);
}
std::string fnum(float v) {
    char b[32];
    std::snprintf(b, sizeof(b), "%.9g", static_cast<double>(v));
    return b;
}
std::string boolean(bool v) {
    return v ? "true" : "false";
}
template <typename E>
std::string en(E v) {
    return std::to_string(static_cast<uint64_t>(v));
}

std::string stage_entry(uint64_t dxil, const std::array<uint8_t, 16>& digest, const Device::NativeStageRecord& n) {
    if (dxil == 0) {
        return "null";
    }
    return "{\"dxil_identity\": " + json_string(id_text(dxil)) + ", \"hash_digest\": " + json_string(digest_text(digest)) +
           ", \"native_code_identity\": " + json_string(n.identity != 0 ? id_text(n.code_identity) : "") +
           ", \"native_identity\": " + json_string(n.identity != 0 ? id_text(n.identity) : "") + "}";
}

const char* range_type(d3d::DescriptorRangeType t) {
    switch (t) {
        case d3d::DescriptorRangeType::Srv:
            return "srv";
        case d3d::DescriptorRangeType::Uav:
            return "uav";
        case d3d::DescriptorRangeType::Cbv:
            return "cbv";
        default:
            return "sampler";
    }
}

const char* parameter_type(d3d::RootParameterType t) {
    switch (t) {
        case d3d::RootParameterType::DescriptorTable:
            return "table";
        case d3d::RootParameterType::Constants32Bit:
            return "constants";
        case d3d::RootParameterType::Cbv:
            return "cbv";
        case d3d::RootParameterType::Srv:
            return "srv";
        default:
            return "uav";
    }
}

std::string root_entry(int index, const d3d::RootSignatureDesc& d) {
    std::string s = "{\"index\": " + inum(index) + ", \"version\": \"1." +
                    num(static_cast<uint32_t>(d.version) - 1) + "\", \"flags\": " + num(d.flags) +
                    ", \"static_samplers\": " + num(d.static_samplers.size()) + ", \"parameters\": [";
    for (size_t p = 0; p < d.parameters.size(); ++p) {
        const d3d::RootParameterDesc& r = d.parameters[p];
        s += std::string(p ? ", " : "") + "{\"type\": \"" + parameter_type(r.parameter_type) +
             "\", \"visibility\": " + en(r.shader_visibility) + ", \"register\": " + num(r.shader_register) +
             ", \"space\": " + num(r.register_space) + ", \"num_32bit_values\": " + num(r.num_32bit_values) +
             ", \"ranges\": [";
        for (size_t i = 0; i < r.ranges.size(); ++i) {
            const d3d::DescriptorRangeDesc& g = r.ranges[i];
            s += std::string(i ? ", " : "") + "{\"type\": \"" + range_type(g.range_type) + "\", \"count\": " +
                 (g.num_descriptors == d3d::kUnboundedDescriptorCount ? std::string("-1") : num(g.num_descriptors)) +
                 ", \"base_register\": " + num(g.base_shader_register) + ", \"space\": " + num(g.register_space) +
                 ", \"offset\": " + num(g.offset_in_descriptors_from_table_start) + "}";
        }
        s += "]}";
    }
    return s + "]}";
}

std::string op_entry(const pso::D3D12DepthStencilOpDesc& o) {
    return "[" + en(o.StencilFailOp) + ", " + en(o.StencilDepthFailOp) + ", " + en(o.StencilPassOp) + ", " +
           en(o.StencilFunc) + "]";
}

}  // namespace

PipelineDump::PipelineDump(std::string dir) : dir_(std::move(dir)) {
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(dir_) / "native", ec);
    std::filesystem::create_directories(std::filesystem::path(dir_) / "dxil", ec);
    ok_ = !ec;
}

bool PipelineDump::write_file(const std::string& path, const uint8_t* bytes, size_t size) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        return false;
    }
    const bool ok = size == 0 || std::fwrite(bytes, 1, size, f) == size;
    return std::fclose(f) == 0 && ok;
}

void PipelineDump::native_stage(uint64_t code_identity, uint64_t identity, bool pixel, bool layout_b,
                                const uint8_t* code, size_t code_size, const uint8_t* metadata, size_t metadata_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ok_ || !native_seen_.insert(identity).second) {
        return;
    }
    const std::string base = dir_ + "/native/" + id_text(code_identity);
    const std::string code_file = "native/" + id_text(code_identity) + ".bin";
    std::string meta_file = "native/" + id_text(code_identity) + ".meta";
    auto it = code_meta_.find(code_identity);
    if (it == code_meta_.end()) {
        write_file(base + ".bin", code, code_size);
        write_file(base + ".meta", metadata, metadata_size);
        code_meta_.emplace(code_identity, identity);
    } else {
        meta_file = "native/" + id_text(code_identity) + "-" + id_text(identity) + ".meta";
        write_file(dir_ + "/" + meta_file, metadata, metadata_size);
    }
    native_entries_.push_back("{\"code_identity\": " + json_string(id_text(code_identity)) + ", \"identity\": " +
                              json_string(id_text(identity)) + ", \"stage\": \"" + (pixel ? "pixel" : "vertex") +
                              "\", \"layout\": \"" + (layout_b ? "B" : "A") + "\", \"code_bytes\": " +
                              num(code_size) + ", \"metadata_bytes\": " + num(metadata_size) + ", \"code_file\": " +
                              json_string(code_file) + ", \"metadata_file\": " + json_string(meta_file) + "}");
}

void PipelineDump::dxil(uint64_t identity, const std::array<uint8_t, 16>* hash_digest, int kind, const uint8_t* bytes,
                        size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ok_ || identity == 0 || !dxil_seen_.insert(identity).second) {
        return;
    }
    const std::string file = "dxil/" + id_text(identity) + ".dxbc";
    write_file(dir_ + "/" + file, bytes, size);
    dxil_entries_.push_back("{\"identity\": " + json_string(id_text(identity)) + ", \"hash_digest\": " +
                            json_string(hash_digest != nullptr ? digest_text(*hash_digest) : "") + ", \"kind\": " +
                            inum(kind) + ", \"bytes\": " + num(size) + ", \"file\": " + json_string(file) + "}");
}

uint32_t PipelineDump::pipeline(const GraphicsPsoSource& src, const std::array<uint8_t, 16>& vs_digest,
                                const std::array<uint8_t, 16>& ps_digest, const void* root_signature,
                                const d3d::RootSignatureDesc* root_desc) {
    const pso::D3D12GraphicsPsoDesc& d = src.desc;
    std::lock_guard<std::mutex> lock(mutex_);
    int root = -1;
    if (root_signature != nullptr && root_desc != nullptr) {
        auto it = root_index_.find(root_signature);
        if (it == root_index_.end()) {
            root = static_cast<int>(root_entries_.size());
            root_index_.emplace(root_signature, root);
            root_entries_.push_back(root_entry(root, *root_desc));
        } else {
            root = it->second;
        }
    }
    std::string s = "\"vs\": " + stage_entry(d.vs, vs_digest, src.native_vs) +
                    ", \"ps\": " + stage_entry(d.ps, ps_digest, src.native_ps) + ", \"root_signature\": " +
                    inum(root) + ", \"topology_type\": " + en(d.PrimitiveTopologyType) +
                    ", \"rect_list\": " + boolean(src.rect_list) + ", \"sample_mask\": " + num(d.SampleMask) +
                    ", \"sample_count\": " + num(d.SampleDesc.Count) + ", \"rtv_formats\": [";
    for (uint32_t i = 0; i < d.NumRenderTargets && i < pso::kD3D12SimultaneousRenderTargetCount; ++i) {
        s += std::string(i ? ", " : "") + en(d.RTVFormats[i]);
    }
    s += "], \"dsv_format\": " + en(d.DSVFormat) + ", \"ib_strip_cut\": " + en(d.IBStripCutValue);
    s += ", \"blend\": {\"alpha_to_coverage\": " + boolean(d.BlendState.AlphaToCoverageEnable) +
         ", \"independent\": " + boolean(d.BlendState.IndependentBlendEnable) + ", \"targets\": [";
    for (uint32_t i = 0; i < d.NumRenderTargets && i < pso::kD3D12SimultaneousRenderTargetCount; ++i) {
        const pso::D3D12RenderTargetBlendDesc& t = d.BlendState.RenderTarget[i];
        s += std::string(i ? ", " : "") + "{\"enable\": " + boolean(t.BlendEnable) + ", \"logic_enable\": " +
             boolean(t.LogicOpEnable) + ", \"src\": " + en(t.SrcBlend) + ", \"dst\": " + en(t.DestBlend) +
             ", \"op\": " + en(t.BlendOp) + ", \"src_alpha\": " + en(t.SrcBlendAlpha) + ", \"dst_alpha\": " +
             en(t.DestBlendAlpha) + ", \"op_alpha\": " + en(t.BlendOpAlpha) + ", \"logic_op\": " + en(t.LogicOp) +
             ", \"write_mask\": " + num(t.RenderTargetWriteMask) + "}";
    }
    const pso::D3D12DepthStencilDesc& z = d.DepthStencilState;
    s += "]}, \"depth_stencil\": {\"depth_enable\": " + boolean(z.DepthEnable) + ", \"depth_write_mask\": " +
         en(z.DepthWriteMask) + ", \"depth_func\": " + en(z.DepthFunc) + ", \"stencil_enable\": " +
         boolean(z.StencilEnable) + ", \"read_mask\": " + num(z.StencilReadMask) + ", \"write_mask\": " +
         num(z.StencilWriteMask) + ", \"front\": " + op_entry(z.FrontFace) + ", \"back\": " + op_entry(z.BackFace) +
         ", \"depth_bounds\": " + boolean(z.DepthBoundsTestEnable) + "}";
    const pso::D3D12RasterizerDesc& r = d.RasterizerState;
    s += ", \"rasterizer\": {\"fill\": " + en(r.FillMode) + ", \"cull\": " + en(r.CullMode) + ", \"front_ccw\": " +
         boolean(r.FrontCounterClockwise) + ", \"depth_bias\": " + inum(r.DepthBias) + ", \"depth_bias_clamp\": " +
         fnum(r.DepthBiasClamp) + ", \"slope_scaled_depth_bias\": " + fnum(r.SlopeScaledDepthBias) +
         ", \"depth_clip\": " + boolean(r.DepthClipEnable) + ", \"multisample\": " + boolean(r.MultisampleEnable) +
         ", \"antialiased_line\": " + boolean(r.AntialiasedLineEnable) + ", \"forced_sample_count\": " +
         num(r.ForcedSampleCount) + ", \"conservative\": " + en(r.ConservativeRaster) + "}";
    s += ", \"input_layout\": [";
    for (uint32_t i = 0; i < d.NumInputElements && d.pInputElementDescs != nullptr; ++i) {
        const pso::D3D12InputElementDesc& e = d.pInputElementDescs[i];
        s += std::string(i ? ", " : "") + "{\"semantic\": " + json_string(e.SemanticName != nullptr ? e.SemanticName : "") +
             ", \"index\": " + num(e.SemanticIndex) + ", \"format\": " + en(e.Format) + ", \"slot\": " +
             num(e.InputSlot) + ", \"offset\": " + num(e.AlignedByteOffset) + ", \"per_instance\": " +
             boolean(e.InputSlotClass == pso::D3D12InputClassification::PerInstanceData) + ", \"step_rate\": " +
             num(e.InstanceDataStepRate) + "}";
    }
    s += "]";
    auto it = pipeline_index_.find(s);
    if (it != pipeline_index_.end()) {
        ++pipelines_[it->second].created;
        return it->second;
    }
    const uint32_t index = static_cast<uint32_t>(pipelines_.size());
    pipeline_index_.emplace(s, index);
    pipelines_.push_back(Pipeline{std::move(s), 1, 0});
    return index;
}

void PipelineDump::draw(uint32_t pipeline_index) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pipeline_index < pipelines_.size()) {
        ++pipelines_[pipeline_index].draws;
    }
}

void PipelineDump::write_manifest() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ok_) {
        return;
    }
    std::string s = "{\"schema\": \"playgta6-pipeline-dump\", \"schema_version\": 1,\n\"native_stages\": [";
    auto list = [&](const std::vector<std::string>& v) {
        for (size_t i = 0; i < v.size(); ++i) {
            s += std::string(i ? ",\n " : "\n ") + v[i];
        }
    };
    list(native_entries_);
    s += "],\n\"dxil\": [";
    list(dxil_entries_);
    s += "],\n\"root_signatures\": [";
    list(root_entries_);
    s += "],\n\"pipelines\": [";
    for (size_t i = 0; i < pipelines_.size(); ++i) {
        s += std::string(i ? ",\n " : "\n ") + "{\"index\": " + num(i) + ", \"created\": " +
             num(pipelines_[i].created) + ", \"draws\": " + num(pipelines_[i].draws) + ", " + pipelines_[i].body + "}";
    }
    s += "]}\n";
    const std::string tmp = dir_ + "/manifest.json.tmp";
    if (write_file(tmp, reinterpret_cast<const uint8_t*>(s.data()), s.size())) {
        std::error_code ec;
        std::filesystem::rename(tmp, dir_ + "/manifest.json", ec);
    }
}

}  // namespace playgta6::gpu::d3d12_guest
