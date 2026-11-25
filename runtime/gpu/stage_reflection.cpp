#include "gpu/stage_reflection.h"

#include "gpu/descriptor_words.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>

namespace playgta6::gpu {

namespace {

// ---------------------------------------------------------------------------------------------
// Minimal JSON reader (objects, arrays, strings, numbers, true/false/null).
// ---------------------------------------------------------------------------------------------

struct Json {
    enum class Type { Null, Bool, Number, String, Array, Object } type = Type::Null;
    bool boolean = false;
    double number = 0;
    std::string string;
    std::vector<Json> array;
    std::vector<std::pair<std::string, Json>> object;

    const Json* get(std::string_view key) const {
        if (type != Type::Object) {
            return nullptr;
        }
        for (const auto& [k, v] : object) {
            if (k == key) {
                return &v;
            }
        }
        return nullptr;
    }
};

class Reader {
public:
    explicit Reader(std::string_view s) : s_(s) {}

    bool parse(Json* out, std::string* error) {
        if (!value(out, 0)) {
            *error = "malformed JSON near byte " + std::to_string(at_);
            return false;
        }
        skip();
        if (at_ != s_.size()) {
            *error = "trailing data at byte " + std::to_string(at_);
            return false;
        }
        return true;
    }

private:
    void skip() {
        while (at_ < s_.size() && (s_[at_] == ' ' || s_[at_] == '\n' || s_[at_] == '\r' || s_[at_] == '\t')) {
            ++at_;
        }
    }
    bool literal(std::string_view word) {
        if (s_.substr(at_, word.size()) != word) {
            return false;
        }
        at_ += word.size();
        return true;
    }
    bool string(std::string* out) {
        if (at_ >= s_.size() || s_[at_] != '"') {
            return false;
        }
        ++at_;
        while (at_ < s_.size() && s_[at_] != '"') {
            char c = s_[at_++];
            if (c == '\\') {
                if (at_ >= s_.size()) {
                    return false;
                }
                const char e = s_[at_++];
                switch (e) {
                    case 'n': c = '\n'; break;
                    case 't': c = '\t'; break;
                    case 'r': c = '\r'; break;
                    case 'b': c = '\b'; break;
                    case 'f': c = '\f'; break;
                    case 'u':
                        if (at_ + 4 > s_.size()) {
                            return false;
                        }
                        at_ += 4;  // non-ASCII escapes are not needed by this format
                        c = '?';
                        break;
                    default: c = e; break;
                }
            }
            out->push_back(c);
        }
        if (at_ >= s_.size()) {
            return false;
        }
        ++at_;
        return true;
    }
    bool value(Json* out, int depth) {
        if (depth > 32) {
            return false;
        }
        skip();
        if (at_ >= s_.size()) {
            return false;
        }
        const char c = s_[at_];
        if (c == '{') {
            ++at_;
            out->type = Json::Type::Object;
            skip();
            if (at_ < s_.size() && s_[at_] == '}') {
                ++at_;
                return true;
            }
            for (;;) {
                skip();
                std::string key;
                if (!string(&key)) {
                    return false;
                }
                skip();
                if (at_ >= s_.size() || s_[at_] != ':') {
                    return false;
                }
                ++at_;
                Json v;
                if (!value(&v, depth + 1)) {
                    return false;
                }
                out->object.emplace_back(std::move(key), std::move(v));
                skip();
                if (at_ < s_.size() && s_[at_] == ',') {
                    ++at_;
                    continue;
                }
                if (at_ < s_.size() && s_[at_] == '}') {
                    ++at_;
                    return true;
                }
                return false;
            }
        }
        if (c == '[') {
            ++at_;
            out->type = Json::Type::Array;
            skip();
            if (at_ < s_.size() && s_[at_] == ']') {
                ++at_;
                return true;
            }
            for (;;) {
                Json v;
                if (!value(&v, depth + 1)) {
                    return false;
                }
                out->array.push_back(std::move(v));
                skip();
                if (at_ < s_.size() && s_[at_] == ',') {
                    ++at_;
                    continue;
                }
                if (at_ < s_.size() && s_[at_] == ']') {
                    ++at_;
                    return true;
                }
                return false;
            }
        }
        if (c == '"') {
            out->type = Json::Type::String;
            return string(&out->string);
        }
        if (literal("true")) {
            out->type = Json::Type::Bool;
            out->boolean = true;
            return true;
        }
        if (literal("false")) {
            out->type = Json::Type::Bool;
            return true;
        }
        if (literal("null")) {
            return true;
        }
        const size_t start = at_;
        while (at_ < s_.size() && (std::isdigit(static_cast<unsigned char>(s_[at_])) || s_[at_] == '-' ||
                                   s_[at_] == '+' || s_[at_] == '.' || s_[at_] == 'e' || s_[at_] == 'E')) {
            ++at_;
        }
        if (at_ == start) {
            return false;
        }
        out->type = Json::Type::Number;
        out->number = std::strtod(std::string(s_.substr(start, at_ - start)).c_str(), nullptr);
        return true;
    }

    std::string_view s_;
    size_t at_ = 0;
};

bool get_u32(const Json& o, std::string_view key, uint32_t* out, bool required = true) {
    const Json* v = o.get(key);
    if (v == nullptr) {
        return !required;
    }
    if (v->type != Json::Type::Number || v->number < 0 || v->number > 4294967295.0) {
        return false;
    }
    *out = static_cast<uint32_t>(v->number);
    return true;
}

std::string get_string(const Json& o, std::string_view key) {
    const Json* v = o.get(key);
    return v != nullptr && v->type == Json::Type::String ? v->string : std::string();
}

bool get_bool(const Json& o, std::string_view key) {
    const Json* v = o.get(key);
    return v != nullptr && v->type == Json::Type::Bool && v->boolean;
}

bool vertex_format_from_name(std::string_view name, pso::WgpuVertexFormat* out) {
    using F = pso::WgpuVertexFormat;
    static const std::pair<std::string_view, F> kNames[] = {
        {"uint8", F::Uint8},         {"uint8x2", F::Uint8x2},     {"uint8x4", F::Uint8x4},
        {"sint8", F::Sint8},         {"sint8x2", F::Sint8x2},     {"sint8x4", F::Sint8x4},
        {"unorm8", F::Unorm8},       {"unorm8x2", F::Unorm8x2},   {"unorm8x4", F::Unorm8x4},
        {"snorm8", F::Snorm8},       {"snorm8x2", F::Snorm8x2},   {"snorm8x4", F::Snorm8x4},
        {"uint16", F::Uint16},       {"uint16x2", F::Uint16x2},   {"uint16x4", F::Uint16x4},
        {"sint16", F::Sint16},       {"sint16x2", F::Sint16x2},   {"sint16x4", F::Sint16x4},
        {"unorm16", F::Unorm16},     {"unorm16x2", F::Unorm16x2}, {"unorm16x4", F::Unorm16x4},
        {"snorm16", F::Snorm16},     {"snorm16x2", F::Snorm16x2}, {"snorm16x4", F::Snorm16x4},
        {"float16", F::Float16},     {"float16x2", F::Float16x2}, {"float16x4", F::Float16x4},
        {"float32", F::Float32},     {"float32x2", F::Float32x2}, {"float32x3", F::Float32x3},
        {"float32x4", F::Float32x4}, {"uint32", F::Uint32},       {"uint32x2", F::Uint32x2},
        {"uint32x3", F::Uint32x3},   {"uint32x4", F::Uint32x4},   {"sint32", F::Sint32},
        {"sint32x2", F::Sint32x2},   {"sint32x3", F::Sint32x3},   {"sint32x4", F::Sint32x4},
        {"unorm10-10-10-2", F::Unorm10_10_10_2},
    };
    for (const auto& [n, f] : kNames) {
        if (n == name) {
            *out = f;
            return true;
        }
    }
    return false;
}

bool view_dimension_from_name(std::string_view name, GpuTextureViewDimension* out) {
    using D = GpuTextureViewDimension;
    static const std::pair<std::string_view, D> kNames[] = {
        {"1d", D::Dim1D}, {"2d", D::Dim2D}, {"2d-array", D::Dim2DArray}, {"2d_array", D::Dim2DArray},
        {"cube", D::Cube}, {"cube-array", D::CubeArray}, {"cube_array", D::CubeArray}, {"3d", D::Dim3D},
    };
    for (const auto& [n, d] : kNames) {
        if (n == name) {
            *out = d;
            return true;
        }
    }
    return false;
}

bool parse_stage(const Json& s, StageReflection* out, std::string* error) {
    const std::string stage = get_string(s, "stage");
    if (stage == "vertex") {
        out->stage = ReflectedStage::Vertex;
    } else if (stage == "fragment" || stage == "pixel") {
        out->stage = ReflectedStage::Fragment;
    } else if (stage == "compute") {
        out->stage = ReflectedStage::Compute;
    } else {
        *error = "unknown stage \"" + stage + "\"";
        return false;
    }
    if (const Json* inputs = s.get("vertex_inputs"); inputs != nullptr && inputs->type == Json::Type::Array) {
        for (const Json& in : inputs->array) {
            ReflectedVertexInput v;
            if (!get_u32(in, "location", &v.location) || !get_u32(in, "vb_slot", &v.vb_slot) ||
                !get_u32(in, "byte_offset", &v.byte_offset) || !get_u32(in, "components", &v.components, false)) {
                *error = "vertex input needs location, vb_slot and byte_offset";
                return false;
            }
            const std::string type = get_string(in, "component_type");
            v.component_type = type == "u32" ? ReflectedComponentType::U32
                               : type == "i32" ? ReflectedComponentType::I32
                                               : ReflectedComponentType::F32;
            const std::string fmt = get_string(in, "fetch_format");
            if (!fmt.empty() && !vertex_format_from_name(fmt, &v.fetch_format)) {
                *error = "unknown fetch_format \"" + fmt + "\"";
                return false;
            }
            out->vertex_inputs.push_back(v);
        }
    }
    if (const Json* bindings = s.get("bindings"); bindings != nullptr && bindings->type == Json::Type::Array) {
        for (const Json& b : bindings->array) {
            ReflectedBinding r;
            if (!get_u32(b, "group", &r.group) || !get_u32(b, "binding", &r.binding)) {
                *error = "binding needs group and binding";
                return false;
            }
            const std::string kind = get_string(b, "kind");
            static const std::pair<std::string_view, ReflectedBindingKind> kKinds[] = {
                {"uniform", ReflectedBindingKind::Uniform},
                {"storage", ReflectedBindingKind::Storage},
                {"read_only_storage", ReflectedBindingKind::ReadOnlyStorage},
                {"texture", ReflectedBindingKind::Texture},
                {"storage_texture", ReflectedBindingKind::StorageTexture},
                {"sampler", ReflectedBindingKind::Sampler},
                {"comparison_sampler", ReflectedBindingKind::ComparisonSampler},
            };
            bool known = false;
            for (const auto& [n, k] : kKinds) {
                if (n == kind) {
                    r.kind = k;
                    known = true;
                }
            }
            if (!known) {
                *error = "unknown binding kind \"" + kind + "\"";
                return false;
            }
            const std::string dim = get_string(b, "view_dimension");
            if (!dim.empty() && !view_dimension_from_name(dim, &r.view_dimension)) {
                *error = "unknown view_dimension \"" + dim + "\"";
                return false;
            }
            const std::string st = get_string(b, "sample_type");
            r.sample_type = st == "unfilterable_float" ? GpuTextureSampleType::UnfilterableFloat
                            : st == "depth"            ? GpuTextureSampleType::Depth
                            : st == "sint"             ? GpuTextureSampleType::Sint
                            : st == "uint"             ? GpuTextureSampleType::Uint
                                                       : GpuTextureSampleType::Float;
            r.multisampled = get_bool(b, "multisampled");
            uint32_t storage_format = 0;
            uint32_t min_size = 0;
            if (!get_u32(b, "storage_format", &storage_format, false) ||
                !get_u32(b, "min_binding_size_bytes", &min_size, false)) {
                *error = "bad storage_format or min_binding_size_bytes";
                return false;
            }
            r.storage_format = static_cast<uint8_t>(storage_format);
            r.min_binding_size = min_size;
            const Json* src = b.get("source");
            if (src == nullptr || src->type != Json::Type::Object) {
                *error = "binding " + std::to_string(r.group) + ":" + std::to_string(r.binding) + " has no source";
                return false;
            }
            const Json* heap = src->get("sampler_heap");
            const bool sampler_heap = heap != nullptr;
            if (heap == nullptr) {
                heap = src->get("resource_heap");
            }
            if (heap != nullptr) {
                r.source = sampler_heap ? ReflectedSource::SamplerHeap : ReflectedSource::ResourceHeap;
                const Json* index = heap->type == Json::Type::Object ? heap->get("index") : nullptr;
                if (index == nullptr || index->type != Json::Type::Object ||
                    !get_u32(*heap, "base_user_sgpr", &r.heap_base_sgpr) || !get_u32(*heap, "stride", &r.heap_stride) ||
                    !get_u32(*index, "user_sgpr", &r.heap_index_sgpr) || !get_u32(*index, "shift", &r.heap_shift) ||
                    r.heap_shift >= 32) {
                    *error = std::string(sampler_heap ? "sampler_heap" : "resource_heap") +
                             " needs base_user_sgpr, stride and index {user_sgpr, shift}";
                    return false;
                }
            } else if (const Json* inline_sampler = src->get("inline_sampler"); inline_sampler != nullptr) {
                r.source = ReflectedSource::InlineSampler;
                const Json* words =
                    inline_sampler->type == Json::Type::Object ? inline_sampler->get("sdwords") : nullptr;
                bool ok = words != nullptr && words->type == Json::Type::Array && words->array.size() == 4;
                for (size_t i = 0; ok && i < 4; ++i) {
                    const Json& w = words->array[i];
                    ok = w.type == Json::Type::Number && w.number >= 0 && w.number <= 4294967295.0 &&
                         w.number == static_cast<double>(static_cast<uint32_t>(w.number));
                    if (ok) {
                        r.sampler_words[i] = static_cast<uint32_t>(w.number);
                    }
                }
                if (!ok) {
                    *error = "inline_sampler needs sdwords: four 32-bit unsigned integers";
                    return false;
                }
            } else if (src->get("static_sampler") != nullptr) {
                r.source = ReflectedSource::StaticSampler;
                if (!get_u32(*src, "static_sampler", &r.static_sampler)) {
                    *error = "bad static_sampler";
                    return false;
                }
            } else {
                if (!get_u32(*src, "user_sgpr", &r.user_sgpr)) {
                    *error = "source needs user_sgpr";
                    return false;
                }
                if (get_bool(*src, "root_constants")) {
                    r.source = ReflectedSource::RootConstants;
                } else if (get_bool(*src, "root_descriptor")) {
                    r.source = ReflectedSource::RootDescriptor;
                } else if (!get_u32(*src, "table_byte_offset", &r.table_byte_offset)) {
                    *error = "source needs table_byte_offset, root_constants or root_descriptor";
                    return false;
                }
            }
            out->bindings.push_back(r);
        }
    }
    return true;
}

uint32_t parameter_dwords(const d3d::RootParameterDesc& p) {
    switch (p.parameter_type) {
        case d3d::RootParameterType::DescriptorTable: return 1;
        case d3d::RootParameterType::Constants32Bit: return p.num_32bit_values;
        default: return 2;
    }
}

const char* kind_name(ReflectedBindingKind k) {
    switch (k) {
        case ReflectedBindingKind::Uniform: return "uniform";
        case ReflectedBindingKind::Storage: return "storage";
        case ReflectedBindingKind::ReadOnlyStorage: return "read_only_storage";
        case ReflectedBindingKind::Texture: return "texture";
        case ReflectedBindingKind::StorageTexture: return "storage_texture";
        case ReflectedBindingKind::Sampler: return "sampler";
        case ReflectedBindingKind::ComparisonSampler: return "comparison_sampler";
    }
    return "?";
}

std::string binding_key(const ReflectedBinding& b) {
    std::string k = "@group(" + std::to_string(b.group) + ") @binding(" + std::to_string(b.binding) + ")";
    switch (b.source) {
        case ReflectedSource::Table:
            k += " <- table at s" + std::to_string(b.user_sgpr) + " + " + std::to_string(b.table_byte_offset);
            break;
        case ReflectedSource::RootConstants: k += " <- root constants at s" + std::to_string(b.user_sgpr); break;
        case ReflectedSource::RootDescriptor: k += " <- root descriptor at s" + std::to_string(b.user_sgpr); break;
        case ReflectedSource::StaticSampler: k += " <- static sampler " + std::to_string(b.static_sampler); break;
        case ReflectedSource::InlineSampler: {
            char w[64];
            std::snprintf(w, sizeof(w), " <- inline sampler %08x %08x %08x %08x", b.sampler_words[0], b.sampler_words[1],
                          b.sampler_words[2], b.sampler_words[3]);
            k += w;
            break;
        }
        case ReflectedSource::SamplerHeap:
        case ReflectedSource::ResourceHeap:
            k += std::string(" <- ") + (b.source == ReflectedSource::SamplerHeap ? "sampler" : "resource") +
                 " heap[s" + std::to_string(b.heap_index_sgpr) + " << " + std::to_string(b.heap_shift) + "]";
            break;
    }
    return k;
}

// The root-constants parameter holding user SGPR `user_sgpr` and the 32-bit value's index in it
// (-1 when no constants parameter covers it).
int root_constant_for_user_sgpr(const d3d::RootSignatureDesc& rs, bool compute, uint32_t user_sgpr, uint32_t* dword) {
    uint32_t sgpr = compute ? kComputeUserDataBase : kGraphicsUserDataBase;
    for (size_t i = 0; i < rs.parameters.size(); ++i) {
        const uint32_t n = parameter_dwords(rs.parameters[i]);
        if (user_sgpr >= sgpr && user_sgpr < sgpr + n) {
            if (rs.parameters[i].parameter_type != d3d::RootParameterType::Constants32Bit) {
                return -1;
            }
            *dword = user_sgpr - sgpr;
            return static_cast<int>(i);
        }
        sgpr += n;
    }
    return -1;
}

bool kind_accepts(ReflectedBindingKind kind, d3d::DescriptorRangeType cls) {
    switch (kind) {
        case ReflectedBindingKind::Uniform:
            return cls == d3d::DescriptorRangeType::Cbv || cls == d3d::DescriptorRangeType::Srv;
        case ReflectedBindingKind::ReadOnlyStorage:
            return cls == d3d::DescriptorRangeType::Srv || cls == d3d::DescriptorRangeType::Uav ||
                   cls == d3d::DescriptorRangeType::Cbv;
        case ReflectedBindingKind::Storage:
        case ReflectedBindingKind::StorageTexture:
            return cls == d3d::DescriptorRangeType::Uav;
        case ReflectedBindingKind::Texture:
            return cls == d3d::DescriptorRangeType::Srv;
        case ReflectedBindingKind::Sampler:
        case ReflectedBindingKind::ComparisonSampler:
            return cls == d3d::DescriptorRangeType::Sampler;
    }
    return false;
}

// The (class, register, space) of descriptor `slot` of a table (D3D12 numbering: ranges in order,
// appended unless they give an explicit offset). False when no range covers the slot.
bool table_slot_register(const d3d::RootParameterDesc& p, uint32_t slot, d3d::DescriptorRangeType* cls,
                         uint32_t* reg, uint32_t* space) {
    uint64_t next = 0;
    for (const d3d::DescriptorRangeDesc& r : p.ranges) {
        const uint64_t start =
            r.offset_in_descriptors_from_table_start == d3d::kDescriptorRangeOffsetAppend ? next
                                                                                          : r.offset_in_descriptors_from_table_start;
        const bool unbounded = r.num_descriptors == d3d::kUnboundedDescriptorCount;
        if (slot >= start && (unbounded || slot < start + r.num_descriptors)) {
            *cls = r.range_type;
            *reg = r.base_shader_register + static_cast<uint32_t>(slot - start);
            *space = r.register_space;
            return true;
        }
        next = unbounded ? UINT64_MAX / 2 : start + r.num_descriptors;
    }
    return false;
}

bool same_binding(const bridge::LayoutBinding& a, const bridge::LayoutBinding& b) {
    return a.entry.type == b.entry.type && a.entry.binding_type == b.entry.binding_type &&
           a.entry.view_dimension == b.entry.view_dimension && a.join.register_class == b.join.register_class &&
           a.join.shader_register == b.join.shader_register && a.join.register_space == b.join.register_space &&
           a.join.root_constants == b.join.root_constants && a.join.heap == b.join.heap &&
           a.join.heap_index_parameter == b.join.heap_index_parameter &&
           a.join.heap_index_dword == b.join.heap_index_dword && a.join.heap_shift == b.join.heap_shift &&
           a.join.table_parameter == b.join.table_parameter && a.join.table_slot == b.join.table_slot &&
           a.join.inline_sampler == b.join.inline_sampler &&
           std::memcmp(&a.join.sampler, &b.join.sampler, sizeof(a.join.sampler)) == 0;
}

pso::WgpuVertexFormat element_class(pso::WgpuVertexFormat f, ReflectedComponentType* cls) {
    using F = pso::WgpuVertexFormat;
    switch (f) {
        case F::Uint8: case F::Uint8x2: case F::Uint8x4: case F::Uint16: case F::Uint16x2: case F::Uint16x4:
        case F::Uint32: case F::Uint32x2: case F::Uint32x3: case F::Uint32x4:
            *cls = ReflectedComponentType::U32;
            break;
        case F::Sint8: case F::Sint8x2: case F::Sint8x4: case F::Sint16: case F::Sint16x2: case F::Sint16x4:
        case F::Sint32: case F::Sint32x2: case F::Sint32x3: case F::Sint32x4:
            *cls = ReflectedComponentType::I32;
            break;
        default:
            *cls = ReflectedComponentType::F32;
            break;
    }
    return f;
}

const char* class_name(ReflectedComponentType c) {
    return c == ReflectedComponentType::U32 ? "u32" : c == ReflectedComponentType::I32 ? "i32" : "f32";
}

}  // namespace

bool parse_stage_reflection_json(std::string_view json, StageReflectionIndex* out, std::string* error) {
    Json root;
    if (!Reader(json).parse(&root, error)) {
        return false;
    }
    if (root.type != Json::Type::Object || get_string(root, "schema") != "playgta6-stage-reflection") {
        *error = "not a playgta6-stage-reflection document";
        return false;
    }
    uint32_t version = 0;
    if (!get_u32(root, "version", &version) || version != 1) {
        *error = "unsupported reflection version";
        return false;
    }
    const Json* stages = root.get("stages");
    if (stages == nullptr || stages->type != Json::Type::Object) {
        *error = "missing \"stages\"";
        return false;
    }
    for (const auto& [key, s] : stages->object) {
        char* end = nullptr;
        const uint64_t k = std::strtoull(key.c_str(), &end, 16);
        if (key.size() < 3 || end == nullptr || *end != '\0' || k == 0) {
            *error = "bad stage key \"" + key + "\"";
            return false;
        }
        StageReflection r;
        std::string why;
        if (!parse_stage(s, &r, &why)) {
            *error = "stage " + key + ": " + why;
            return false;
        }
        (*out)[k] = std::move(r);
    }
    return true;
}

int root_parameter_for_user_sgpr(const d3d::RootSignatureDesc& rs, bool compute, uint32_t user_sgpr) {
    uint32_t sgpr = compute ? kComputeUserDataBase : kGraphicsUserDataBase;
    for (size_t i = 0; i < rs.parameters.size(); ++i) {
        if (sgpr == user_sgpr) {
            return static_cast<int>(i);
        }
        sgpr += parameter_dwords(rs.parameters[i]);
    }
    return -1;
}

bool join_pipeline_layout(const d3d::RootSignatureDesc& rs, bool compute, const std::vector<ReflectedStageRef>& stages,
                          bridge::PipelineLayoutDesc* out, std::string* report, JoinStats* stats) {
    std::map<std::pair<uint32_t, uint32_t>, bridge::LayoutBinding> joined;
    uint64_t outside_rs = 0;
    bool ok = true;
    auto fail = [&](const ReflectedBinding& b, const std::string& why) {
        *report += binding_key(b) + ": " + why + "\n";
        ok = false;
    };
    for (const ReflectedStageRef& st : stages) {
        if (st.reflection == nullptr) {
            continue;
        }
        for (const ReflectedBinding& b : st.reflection->bindings) {
            bridge::LayoutBinding lb;
            lb.entry.binding = b.binding;
            lb.entry.visibility = st.visibility;
            // Where the descriptor comes from, as a root-signature register.
            d3d::DescriptorRangeType cls = d3d::DescriptorRangeType::Srv;
            uint32_t reg = 0, space = 0;
            if (b.source == ReflectedSource::SamplerHeap || b.source == ReflectedSource::ResourceHeap) {
                // Heap-indexed: the index must be a root constant; the class follows the kind.
                uint32_t dword = 0;
                const int pi = root_constant_for_user_sgpr(rs, compute, b.heap_index_sgpr, &dword);
                if (pi < 0) {
                    fail(b, "s" + std::to_string(b.heap_index_sgpr) + " is not inside a root-constants parameter");
                    continue;
                }
                const bool sampler_kind =
                    b.kind == ReflectedBindingKind::Sampler || b.kind == ReflectedBindingKind::ComparisonSampler;
                if ((b.source == ReflectedSource::SamplerHeap) != sampler_kind) {
                    fail(b, std::string("a ") + kind_name(b.kind) + " binding cannot come from the " +
                                (b.source == ReflectedSource::SamplerHeap ? "sampler" : "resource") + " heap");
                    continue;
                }
                cls = b.kind == ReflectedBindingKind::Uniform ? d3d::DescriptorRangeType::Cbv
                      : sampler_kind                          ? d3d::DescriptorRangeType::Sampler
                      : (b.kind == ReflectedBindingKind::Storage || b.kind == ReflectedBindingKind::StorageTexture)
                          ? d3d::DescriptorRangeType::Uav
                          : d3d::DescriptorRangeType::Srv;
                lb.join.heap = sampler_kind ? d3d::HeapAccess::Sampler : d3d::HeapAccess::Resource;
                lb.join.heap_index_parameter = static_cast<uint32_t>(pi);
                lb.join.heap_index_dword = dword;
                lb.join.heap_shift = b.heap_shift;
            } else if (b.source == ReflectedSource::InlineSampler) {
                if (b.kind != ReflectedBindingKind::Sampler && b.kind != ReflectedBindingKind::ComparisonSampler) {
                    fail(b, std::string("a ") + kind_name(b.kind) + " binding cannot take an inline sampler");
                    continue;
                }
                d3d::SamplerDesc desc;
                const bridge::DecodeStatus st = bridge::decode_sampler_words(b.sampler_words, &desc);
                if (st != bridge::DecodeStatus::Ok) {
                    fail(b, std::string("the inline sampler words do not decode (") + bridge::decode_status_name(st) +
                                ")");
                    continue;
                }
                cls = d3d::DescriptorRangeType::Sampler;
                reg = space = d3d::kNoIndex;
                lb.join.inline_sampler = true;
                lb.join.sampler = desc;
            } else if (b.source == ReflectedSource::StaticSampler) {
                cls = d3d::DescriptorRangeType::Sampler;
                if (b.static_sampler < rs.static_samplers.size()) {
                    reg = rs.static_samplers[b.static_sampler].shader_register;
                    space = rs.static_samplers[b.static_sampler].register_space;
                } else {
                    // A sampler the console stage builds from constants in its code: the root
                    // signature has no static sampler for it. Bound as the default sampler (an
                    // unresolved join binds null, i.e. the worker's default), not a join failure.
                    reg = space = d3d::kNoIndex;
                }
            } else {
                const int pi = root_parameter_for_user_sgpr(rs, compute, b.user_sgpr);
                if (pi < 0) {
                    fail(b, "no root parameter starts at s" + std::to_string(b.user_sgpr));
                    continue;
                }
                const d3d::RootParameterDesc& p = rs.parameters[static_cast<size_t>(pi)];
                const std::string pname = "root parameter " + std::to_string(pi);
                switch (b.source) {
                    case ReflectedSource::Table: {
                        if (p.parameter_type != d3d::RootParameterType::DescriptorTable) {
                            fail(b, pname + " is not a descriptor table");
                            continue;
                        }
                        if (b.table_byte_offset % kTableSlotBytes != 0) {
                            fail(b, pname + ": byte offset " + std::to_string(b.table_byte_offset) +
                                        " is not a descriptor boundary");
                            continue;
                        }
                        const bool read_only_kind = b.kind != ReflectedBindingKind::Storage &&
                                                    b.kind != ReflectedBindingKind::StorageTexture;
                        const bool declared =
                            table_slot_register(p, b.table_byte_offset / kTableSlotBytes, &cls, &reg, &space);
                        if (!declared || (read_only_kind && !kind_accepts(b.kind, cls)) ||
                            (!read_only_kind && cls == d3d::DescriptorRangeType::Srv)) {
                            // A slot the root signature does not declare, or declares as another class:
                            // console stages read their tables as memory, past and regardless of the
                            // declared ranges. Join the slot itself, with the class the stage reads.
                            // That includes a writable binding over a declared SRV: the stage stores
                            // through the descriptor (image_store / buffer_store), so the slot holds a
                            // UAV-capable view on the console; it is bound as a storage view (a texture
                            // created without storage usage binds null, see the state bridge). A
                            // writable binding over a declared CBV stays a failure.
                            const bool sampler_kind = b.kind == ReflectedBindingKind::Sampler ||
                                                      b.kind == ReflectedBindingKind::ComparisonSampler;
                            bool sampler_table = false;
                            for (const d3d::DescriptorRangeDesc& r : p.ranges) {
                                sampler_table = sampler_table || r.range_type == d3d::DescriptorRangeType::Sampler;
                            }
                            if (sampler_kind != sampler_table) {
                                fail(b, pname + " has no descriptor at byte offset " +
                                            std::to_string(b.table_byte_offset) + " and is a " +
                                            (sampler_table ? "sampler" : "resource") + " table");
                                continue;
                            }
                            cls = b.kind == ReflectedBindingKind::Uniform ? d3d::DescriptorRangeType::Cbv
                                  : sampler_kind                         ? d3d::DescriptorRangeType::Sampler
                                  : (b.kind == ReflectedBindingKind::Storage ||
                                     b.kind == ReflectedBindingKind::StorageTexture)
                                      ? d3d::DescriptorRangeType::Uav
                                      : d3d::DescriptorRangeType::Srv;
                            reg = space = 0;
                            lb.join.table_parameter = static_cast<uint32_t>(pi);
                            lb.join.table_slot = b.table_byte_offset / kTableSlotBytes;
                            ++outside_rs;
                        }
                        break;
                    }
                    case ReflectedSource::RootConstants:
                        if (p.parameter_type != d3d::RootParameterType::Constants32Bit) {
                            fail(b, pname + " is not root constants");
                            continue;
                        }
                        cls = d3d::DescriptorRangeType::Cbv;
                        reg = p.shader_register;
                        space = p.register_space;
                        lb.join.root_constants = true;
                        break;
                    case ReflectedSource::RootDescriptor:
                        if (p.parameter_type == d3d::RootParameterType::Cbv) {
                            cls = d3d::DescriptorRangeType::Cbv;
                        } else if (p.parameter_type == d3d::RootParameterType::Srv) {
                            cls = d3d::DescriptorRangeType::Srv;
                        } else if (p.parameter_type == d3d::RootParameterType::Uav) {
                            cls = d3d::DescriptorRangeType::Uav;
                        } else {
                            fail(b, pname + " is not a root descriptor");
                            continue;
                        }
                        reg = p.shader_register;
                        space = p.register_space;
                        break;
                    case ReflectedSource::StaticSampler:
                    case ReflectedSource::SamplerHeap:
                    case ReflectedSource::ResourceHeap:
                    case ReflectedSource::InlineSampler:
                        break;
                }
            }
            if (!lb.join.root_constants && !kind_accepts(b.kind, cls)) {
                fail(b, std::string("a ") + kind_name(b.kind) + " binding cannot take a " +
                            (cls == d3d::DescriptorRangeType::Cbv   ? "CBV"
                             : cls == d3d::DescriptorRangeType::Srv ? "SRV"
                             : cls == d3d::DescriptorRangeType::Uav ? "UAV"
                                                                    : "sampler"));
                continue;
            }
            if (lb.join.root_constants && b.kind != ReflectedBindingKind::Uniform) {
                fail(b, "root constants bind as a uniform buffer only");
                continue;
            }
            lb.join.register_class = cls;
            lb.join.shader_register = reg;
            lb.join.register_space = space;
            // The WebGPU entry, from the reflection.
            switch (b.kind) {
                case ReflectedBindingKind::Uniform:
                    lb.entry.type = static_cast<uint8_t>(GpuBindingLayoutType::Buffer);
                    lb.entry.binding_type = static_cast<uint8_t>(GpuBufferBindingType::Uniform);
                    lb.entry.min_binding_size = b.min_binding_size;
                    break;
                case ReflectedBindingKind::Storage:
                    lb.entry.type = static_cast<uint8_t>(GpuBindingLayoutType::Buffer);
                    lb.entry.binding_type = static_cast<uint8_t>(GpuBufferBindingType::Storage);
                    lb.entry.min_binding_size = b.min_binding_size;
                    break;
                case ReflectedBindingKind::ReadOnlyStorage:
                    lb.entry.type = static_cast<uint8_t>(GpuBindingLayoutType::Buffer);
                    lb.entry.binding_type = static_cast<uint8_t>(GpuBufferBindingType::ReadOnlyStorage);
                    lb.entry.min_binding_size = b.min_binding_size;
                    break;
                case ReflectedBindingKind::Texture:
                    lb.entry.type = static_cast<uint8_t>(GpuBindingLayoutType::Texture);
                    lb.entry.binding_type = static_cast<uint8_t>(b.sample_type);
                    lb.entry.view_dimension = static_cast<uint8_t>(b.view_dimension);
                    if (b.multisampled) {
                        lb.entry.flags |= GpuBindGroupLayoutEntryFlags::Multisampled;
                    }
                    break;
                case ReflectedBindingKind::StorageTexture:
                    if (b.storage_format == 0) {
                        fail(b, "a storage texture needs its format");
                        continue;
                    }
                    lb.entry.type = static_cast<uint8_t>(GpuBindingLayoutType::StorageTexture);
                    // The translator declares storage textures write-only (texture_storage_*<fmt, write>);
                    // the layout must match, and read-write access is not even allowed for formats
                    // such as rgba8unorm (WebGPU rejects the layout, the pipeline and the frame).
                    lb.entry.binding_type = static_cast<uint8_t>(GpuStorageTextureAccess::WriteOnly);
                    lb.entry.view_dimension = static_cast<uint8_t>(b.view_dimension);
                    lb.entry.storage_format = b.storage_format;
                    break;
                case ReflectedBindingKind::Sampler:
                    lb.entry.type = static_cast<uint8_t>(GpuBindingLayoutType::Sampler);
                    lb.entry.binding_type = static_cast<uint8_t>(GpuSamplerBindingType::Filtering);
                    break;
                case ReflectedBindingKind::ComparisonSampler:
                    lb.entry.type = static_cast<uint8_t>(GpuBindingLayoutType::Sampler);
                    lb.entry.binding_type = static_cast<uint8_t>(GpuSamplerBindingType::Comparison);
                    break;
            }
            const std::pair<uint32_t, uint32_t> key{b.group, b.binding};
            auto it = joined.find(key);
            if (it == joined.end()) {
                joined.emplace(key, lb);
            } else if (!same_binding(it->second, lb)) {
                fail(b, "the stages disagree on this binding's kind or source");
            } else {
                it->second.entry.visibility |= lb.entry.visibility;
                it->second.entry.min_binding_size =
                    std::max(it->second.entry.min_binding_size, lb.entry.min_binding_size);
            }
        }
    }
    if (!ok) {
        return false;
    }
    bridge::PipelineLayoutDesc layout;
    for (const auto& [key, lb] : joined) {
        if (layout.groups.size() <= key.first) {
            layout.groups.resize(key.first + 1);
        }
        layout.groups[key.first].bindings.push_back(lb);
    }
    if (layout.groups.empty()) {
        layout.groups.resize(1);  // stages that bind nothing still get one (empty) group
    }
    *out = std::move(layout);
    if (stats != nullptr) {
        stats->table_slots_outside_rs += outside_rs;
    }
    return true;
}

bool join_vertex_inputs(const StageReflection& vs, const pso::D3D12InputElementDesc* elements, uint32_t count,
                        std::vector<pso::VertexInputFetch>* out, std::string* report) {
    struct Element {
        uint32_t slot, offset, size;
        pso::WgpuVertexFormat format;
        const char* name;
        uint32_t index;
    };
    std::vector<Element> placed;
    uint32_t slot_end[pso::kD3D12IaVertexInputSlotCount] = {};
    for (uint32_t i = 0; i < count && elements != nullptr; ++i) {
        const pso::D3D12InputElementDesc& e = elements[i];
        const pso::VertexFormatMapping vf = pso::map_vertex_format(e.Format);
        if (!vf.supported() || e.InputSlot >= pso::kD3D12IaVertexInputSlotCount) {
            continue;
        }
        uint32_t offset = e.AlignedByteOffset;
        if (offset == pso::kD3D12AppendAlignedElement) {
            const uint32_t a = std::min(4u, vf.byte_size);
            offset = (slot_end[e.InputSlot] + a - 1) / a * a;
        }
        slot_end[e.InputSlot] = std::max(slot_end[e.InputSlot], offset + vf.byte_size);
        placed.push_back(Element{e.InputSlot, offset, vf.byte_size, vf.format, e.SemanticName, e.SemanticIndex});
    }
    bool ok = true;
    std::vector<pso::VertexInputFetch> fetches;
    for (const ReflectedVertexInput& in : vs.vertex_inputs) {
        const std::string key = "@location(" + std::to_string(in.location) + ") <- slot " + std::to_string(in.vb_slot) +
                                " + " + std::to_string(in.byte_offset);
        const Element* hit = nullptr;
        for (const Element& el : placed) {
            if (el.slot == in.vb_slot && in.byte_offset >= el.offset && in.byte_offset < el.offset + el.size) {
                hit = &el;
                break;
            }
        }
        if (hit == nullptr) {
            *report += key + ": no input-layout element covers it\n";
            ok = false;
            continue;
        }
        pso::VertexInputFetch f;
        f.slot = in.vb_slot;
        f.offset = in.byte_offset;
        f.location = in.location;
        if (in.byte_offset == hit->offset && in.fetch_format == pso::WgpuVertexFormat::Undefined) {
            f.format = hit->format;
        } else if (in.fetch_format != pso::WgpuVertexFormat::Undefined) {
            f.format = in.fetch_format;  // the stage's own fetch (inside an element, or a reinterpretation)
        } else {
            *report += key + ": reads inside element " + std::string(hit->name ? hit->name : "?") +
                       std::to_string(hit->index) + " and has no fetch_format\n";
            ok = false;
            continue;
        }
        ReflectedComponentType cls;
        element_class(f.format, &cls);
        if (cls != in.component_type) {
            *report += key + ": the stage expects " + class_name(in.component_type) + ", the attribute gives " +
                       class_name(cls) + "\n";
            ok = false;
            continue;
        }
        fetches.push_back(f);
    }
    if (!ok) {
        return false;
    }
    *out = std::move(fetches);
    return true;
}

}  // namespace playgta6::gpu
