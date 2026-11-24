// Synthetic tests: stage reflection sidecar, user-data -> root parameter rule, and the joins that
// build vertex attributes and pipeline layouts for pack-resolved stages (G62, #651).
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "gpu/pso_cache.h"
#include "gpu/stage_reflection.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_stage_reflection.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6;
using namespace playgta6::gpu;
namespace d3d = playgta6::gpu::d3d12_state;

// Root signature: [0] table {SRV t2 space1, CBV b0 space1}, [1] 2 root constants b500,
// [2] root CBV b7; static sampler s3 (comparison).
d3d::RootSignatureDesc ui_root_signature() {
    d3d::RootSignatureDesc rs;
    d3d::RootParameterDesc table;
    table.parameter_type = d3d::RootParameterType::DescriptorTable;
    d3d::DescriptorRangeDesc srv;
    srv.range_type = d3d::DescriptorRangeType::Srv;
    srv.base_shader_register = 2;
    srv.register_space = 1;
    d3d::DescriptorRangeDesc cbv;
    cbv.range_type = d3d::DescriptorRangeType::Cbv;
    cbv.base_shader_register = 0;
    cbv.register_space = 1;
    table.ranges = {srv, cbv};
    d3d::RootParameterDesc constants;
    constants.parameter_type = d3d::RootParameterType::Constants32Bit;
    constants.shader_register = 500;
    constants.num_32bit_values = 2;
    d3d::RootParameterDesc root_cbv;
    root_cbv.parameter_type = d3d::RootParameterType::Cbv;
    root_cbv.shader_register = 7;
    rs.parameters = {table, constants, root_cbv};
    d3d::StaticSamplerDesc s;
    s.filter = 0x94;
    s.comparison_func = 4;
    s.shader_register = 3;
    rs.static_samplers = {s};
    return rs;
}

const char* kSidecar = R"({
  "schema": "playgta6-stage-reflection", "version": 1,
  "stages": {
    "0x00000000000000a1": {
      "stage": "vertex",
      "vertex_inputs": [
        {"location": 0, "vb_slot": 0, "byte_offset": 8, "components": 2, "component_type": "f32"},
        {"location": 1, "vb_slot": 0, "byte_offset": 0, "components": 4, "component_type": "f32"},
        {"location": 2, "vb_slot": 0, "byte_offset": 6, "components": 2, "component_type": "f32",
         "fetch_format": "unorm8x2"}
      ],
      "bindings": [
        {"group": 0, "binding": 0, "kind": "read_only_storage", "min_binding_size_bytes": 64,
         "source": {"user_sgpr": 8, "table_byte_offset": 0}}
      ]
    },
    "0x00000000000000b2": {
      "stage": "fragment",
      "bindings": [
        {"group": 0, "binding": 0, "kind": "read_only_storage", "min_binding_size_bytes": 128,
         "source": {"user_sgpr": 8, "table_byte_offset": 0}},
        {"group": 0, "binding": 1, "kind": "uniform", "min_binding_size_bytes": 16,
         "source": {"user_sgpr": 9, "root_constants": true}},
        {"group": 0, "binding": 2, "kind": "uniform", "min_binding_size_bytes": 48,
         "source": {"user_sgpr": 8, "table_byte_offset": 32}},
        {"group": 0, "binding": 3, "kind": "uniform", "source": {"user_sgpr": 11, "root_descriptor": true}},
        {"group": 1, "binding": 0, "kind": "comparison_sampler", "source": {"static_sampler": 0}},
        {"group": 1, "binding": 1, "kind": "texture", "view_dimension": "2d-array", "sample_type": "depth",
         "source": {"user_sgpr": 8, "table_byte_offset": 0}}
      ]
    }
  }
})";

void test_parse() {
    StageReflectionIndex index;
    std::string error;
    CHECK(parse_stage_reflection_json(kSidecar, &index, &error));
    CHECK(error.empty());
    CHECK(index.size() == 2);
    const StageReflection& vs = index.at(0xa1);
    CHECK(vs.stage == ReflectedStage::Vertex);
    CHECK(vs.vertex_inputs.size() == 3 && vs.vertex_inputs[2].fetch_format == pso::WgpuVertexFormat::Unorm8x2);
    CHECK(vs.bindings.size() == 1 && vs.bindings[0].kind == ReflectedBindingKind::ReadOnlyStorage &&
          vs.bindings[0].min_binding_size == 64 && vs.bindings[0].user_sgpr == 8);
    const StageReflection& ps = index.at(0xb2);
    CHECK(ps.stage == ReflectedStage::Fragment && ps.bindings.size() == 6);
    CHECK(ps.bindings[1].source == ReflectedSource::RootConstants);
    CHECK(ps.bindings[3].source == ReflectedSource::RootDescriptor && ps.bindings[3].user_sgpr == 11);
    CHECK(ps.bindings[4].source == ReflectedSource::StaticSampler);
    CHECK(ps.bindings[5].view_dimension == GpuTextureViewDimension::Dim2DArray &&
          ps.bindings[5].sample_type == GpuTextureSampleType::Depth);

    // Malformed documents are rejected with a message.
    StageReflectionIndex bad;
    CHECK(!parse_stage_reflection_json("{\"schema\": \"x\", \"version\": 1, \"stages\": {}}", &bad, &error));
    CHECK(!parse_stage_reflection_json(
        R"({"schema": "playgta6-stage-reflection", "version": 1, "stages": {"0x1": {"stage": "vertex",
           "bindings": [{"group": 0, "binding": 0, "kind": "uniform"}]}}})",
        &bad, &error));
    CHECK(error.find("no source") != std::string::npos);
    CHECK(!parse_stage_reflection_json("{\"schema\": ", &bad, &error));
}

void test_user_data_rule() {
    const d3d::RootSignatureDesc rs = ui_root_signature();
    // Graphics: table at s8, 2 constants at s9..s10, root CBV at s11..s12.
    CHECK(root_parameter_for_user_sgpr(rs, false, 8) == 0);
    CHECK(root_parameter_for_user_sgpr(rs, false, 9) == 1);
    CHECK(root_parameter_for_user_sgpr(rs, false, 10) == -1);
    CHECK(root_parameter_for_user_sgpr(rs, false, 11) == 2);
    CHECK(root_parameter_for_user_sgpr(rs, false, 4) == -1);
    // Compute: from s4.
    CHECK(root_parameter_for_user_sgpr(rs, true, 4) == 0);
    CHECK(root_parameter_for_user_sgpr(rs, true, 5) == 1);
    CHECK(root_parameter_for_user_sgpr(rs, true, 7) == 2);
}

std::vector<pso::D3D12InputElementDesc> ui_input_layout() {
    // COLOR0 unorm8x4 @0, COLOR1 unorm8x4 @4 (appended), POSITION float2 @8 (appended).
    std::vector<pso::D3D12InputElementDesc> e(3);
    e[0].SemanticName = "COLOR";
    e[0].Format = pso::DxgiFormat::R8G8B8A8_UNORM;
    e[0].AlignedByteOffset = 0;
    e[1].SemanticName = "COLOR";
    e[1].SemanticIndex = 1;
    e[1].Format = pso::DxgiFormat::R8G8B8A8_UNORM;
    e[2].SemanticName = "POSITION";
    e[2].Format = pso::DxgiFormat::R32G32_FLOAT;
    return e;
}

void test_vertex_join() {
    StageReflectionIndex index;
    std::string error;
    CHECK(parse_stage_reflection_json(kSidecar, &index, &error));
    const std::vector<pso::D3D12InputElementDesc> layout = ui_input_layout();
    std::vector<pso::VertexInputFetch> fetches;
    std::string report;
    CHECK(join_vertex_inputs(index.at(0xa1), layout.data(), 3, &fetches, &report));
    CHECK(report.empty());
    CHECK(fetches.size() == 3);
    if (fetches.size() == 3) {
        // POSITION feeds @location(0), COLOR0 @location(1); COLOR1 is read inside (bytes 6..7).
        CHECK(fetches[0].location == 0 && fetches[0].offset == 8 && fetches[0].format == pso::WgpuVertexFormat::Float32x2);
        CHECK(fetches[1].location == 1 && fetches[1].offset == 0 && fetches[1].format == pso::WgpuVertexFormat::Unorm8x4);
        CHECK(fetches[2].location == 2 && fetches[2].offset == 6 && fetches[2].format == pso::WgpuVertexFormat::Unorm8x2);
    }

    // Failures: an offset no element covers, a read inside an element without fetch format, and a
    // component class mismatch. Each is reported with its key.
    StageReflection bad;
    bad.vertex_inputs = {{0, 0, 40, 2, ReflectedComponentType::F32, pso::WgpuVertexFormat::Undefined},
                         {1, 0, 5, 1, ReflectedComponentType::F32, pso::WgpuVertexFormat::Undefined},
                         {2, 0, 0, 4, ReflectedComponentType::U32, pso::WgpuVertexFormat::Undefined},
                         {3, 1, 0, 4, ReflectedComponentType::F32, pso::WgpuVertexFormat::Undefined}};
    report.clear();
    CHECK(!join_vertex_inputs(bad, layout.data(), 3, &fetches, &report));
    CHECK(report.find("@location(0) <- slot 0 + 40: no input-layout element") != std::string::npos);
    CHECK(report.find("@location(1) <- slot 0 + 5: reads inside element COLOR1") != std::string::npos);
    CHECK(report.find("@location(2) <- slot 0 + 0: the stage expects u32, the attribute gives f32") !=
          std::string::npos);
    CHECK(report.find("@location(3) <- slot 1 + 0: no input-layout element") != std::string::npos);

    // The pipeline key takes the fetches: locations, offsets and formats from the stage; COLOR1 dropped.
    pso::D3D12GraphicsPsoDesc desc{};
    desc.vs = 1;
    desc.ps = 2;
    desc.pInputElementDescs = layout.data();
    desc.NumInputElements = 3;
    desc.NumRenderTargets = 1;
    desc.RTVFormats[0] = pso::DxgiFormat::R8G8B8A8_UNORM;
    desc.vs_input_fetches = fetches.data();
    desc.vs_input_fetch_count = 0;
    std::vector<pso::VertexInputFetch> good;
    report.clear();
    CHECK(join_vertex_inputs(index.at(0xa1), layout.data(), 3, &good, &report));
    desc.vs_input_fetches = good.data();
    desc.vs_input_fetch_count = static_cast<uint32_t>(good.size());
    const pso::GraphicsPsoTranslation t = pso::translate_graphics_pso(desc, pso::PipelineKeyOptions{});
    CHECK(t.base.vertex_attribute_count == 3 && t.base.vertex_buffer_count == 1);
    CHECK(t.base.vertex_buffers[0].array_stride == 16 && t.base.vertex_buffers[0].attribute_count == 3);
    bool saw_position = false, saw_inside = false;
    for (uint32_t i = 0; i < t.base.vertex_attribute_count; ++i) {
        const auto& a = t.base.vertex_attributes[i];
        saw_position |= a.shader_location == 0 && a.offset == 8 && a.format == pso::WgpuVertexFormat::Float32x2;
        saw_inside |= a.shader_location == 2 && a.offset == 6 && a.format == pso::WgpuVertexFormat::Unorm8x2;
    }
    CHECK(saw_position && saw_inside);
    CHECK((t.diagnostics & (pso::PipelineDiag::VertexLayoutInvalid | pso::PipelineDiag::MissingVertexInput)) == 0);
}

const bridge::LayoutBinding* find(const bridge::PipelineLayoutDesc& l, uint32_t group, uint32_t binding) {
    if (group >= l.groups.size()) {
        return nullptr;
    }
    for (const bridge::LayoutBinding& b : l.groups[group].bindings) {
        if (b.entry.binding == binding) {
            return &b;
        }
    }
    return nullptr;
}

void test_layout_join() {
    StageReflectionIndex index;
    std::string error;
    CHECK(parse_stage_reflection_json(kSidecar, &index, &error));
    const d3d::RootSignatureDesc rs = ui_root_signature();
    bridge::PipelineLayoutDesc layout;
    std::string report;
    // The fragment stage's texture at slot 0 conflicts with the vertex stage's storage buffer at
    // group 0 binding 0? No: different (group, binding). Join both stages.
    StageReflection ps = index.at(0xb2);
    ps.bindings.pop_back();  // texture <- slot 0 (an SRV): valid, but kept for the failure test below
    CHECK(join_pipeline_layout(rs, false, {{GpuShaderStageBits::Vertex, &index.at(0xa1)},
                                           {GpuShaderStageBits::Fragment, &ps}},
                               &layout, &report));
    CHECK(report.empty());
    CHECK(layout.groups.size() == 2);
    const bridge::LayoutBinding* b0 = find(layout, 0, 0);
    CHECK(b0 != nullptr);
    if (b0 != nullptr) {
        // Shared by both stages: visibility merged, the larger minimum wins, joined to SRV t2 space 1.
        CHECK(b0->entry.visibility == (GpuShaderStageBits::Vertex | GpuShaderStageBits::Fragment));
        CHECK(b0->entry.min_binding_size == 128);
        CHECK(b0->entry.type == static_cast<uint8_t>(GpuBindingLayoutType::Buffer) &&
              b0->entry.binding_type == static_cast<uint8_t>(GpuBufferBindingType::ReadOnlyStorage));
        CHECK(b0->join.register_class == d3d::DescriptorRangeType::Srv && b0->join.shader_register == 2 &&
              b0->join.register_space == 1 && !b0->join.root_constants);
    }
    const bridge::LayoutBinding* b1 = find(layout, 0, 1);
    CHECK(b1 != nullptr && b1->join.root_constants && b1->join.shader_register == 500 &&
          b1->entry.min_binding_size == 16 && b1->entry.visibility == GpuShaderStageBits::Fragment);
    const bridge::LayoutBinding* b2 = find(layout, 0, 2);
    CHECK(b2 != nullptr && b2->join.register_class == d3d::DescriptorRangeType::Cbv && b2->join.shader_register == 0 &&
          b2->join.register_space == 1 && b2->entry.min_binding_size == 48);
    const bridge::LayoutBinding* b3 = find(layout, 0, 3);
    CHECK(b3 != nullptr && b3->join.register_class == d3d::DescriptorRangeType::Cbv && b3->join.shader_register == 7);
    const bridge::LayoutBinding* s0 = find(layout, 1, 0);
    CHECK(s0 != nullptr && s0->join.register_class == d3d::DescriptorRangeType::Sampler && s0->join.shader_register == 3 &&
          s0->entry.binding_type == static_cast<uint8_t>(GpuSamplerBindingType::Comparison));
    // Unused root-signature entries are omitted: no binding for anything else.
    CHECK(layout.groups[0].bindings.size() == 4 && layout.groups[1].bindings.size() == 1);

    // Compute: the same table at s4.
    StageReflection cs;
    cs.stage = ReflectedStage::Compute;
    ReflectedBinding cb;
    cb.kind = ReflectedBindingKind::Uniform;
    cb.user_sgpr = 4;
    cb.table_byte_offset = 32;
    cs.bindings = {cb};
    report.clear();
    CHECK(join_pipeline_layout(rs, true, {{GpuShaderStageBits::Compute, &cs}}, &layout, &report));
    CHECK(layout.groups.size() == 1 && layout.groups[0].bindings.size() == 1 &&
          layout.groups[0].bindings[0].join.register_class == d3d::DescriptorRangeType::Cbv);

    // Failures, one line each with the key that failed; no layout is produced.
    StageReflection bad;
    ReflectedBinding nb;
    nb.user_sgpr = 12;  // inside the root CBV, not at a parameter start
    ReflectedBinding off;
    off.binding = 1;
    off.user_sgpr = 8;
    off.table_byte_offset = 64;  // slot 2: the table has 2 slots
    ReflectedBinding tex;
    tex.binding = 2;
    tex.kind = ReflectedBindingKind::Texture;
    tex.user_sgpr = 8;
    tex.table_byte_offset = 32;  // slot 1 is CBV b0
    ReflectedBinding odd;
    odd.binding = 3;
    odd.user_sgpr = 8;
    odd.table_byte_offset = 16;  // not a slot boundary
    ReflectedBinding sampler;
    sampler.binding = 4;
    sampler.kind = ReflectedBindingKind::Sampler;
    sampler.source = ReflectedSource::StaticSampler;
    sampler.static_sampler = 5;
    bad.bindings = {nb, off, tex, odd, sampler};
    report.clear();
    bridge::PipelineLayoutDesc untouched;
    CHECK(!join_pipeline_layout(rs, false, {{GpuShaderStageBits::Vertex, &bad}}, &untouched, &report));
    CHECK(untouched.groups.empty());
    CHECK(report.find("@group(0) @binding(0) <- table at s12 + 0: no root parameter starts at s12") != std::string::npos);
    CHECK(report.find("@group(0) @binding(1) <- table at s8 + 64: root parameter 0 has no descriptor at byte offset 64") !=
          std::string::npos);
    CHECK(report.find("@group(0) @binding(2) <- table at s8 + 32: a texture binding cannot take a CBV") !=
          std::string::npos);
    CHECK(report.find("byte offset 16") != std::string::npos);
    CHECK(report.find("no static sampler 5") != std::string::npos);

    // Two stages that disagree on one binding.
    StageReflection a, b;
    ReflectedBinding x;
    x.kind = ReflectedBindingKind::Uniform;
    x.user_sgpr = 8;
    x.table_byte_offset = 32;
    a.bindings = {x};
    x.kind = ReflectedBindingKind::ReadOnlyStorage;
    x.table_byte_offset = 0;
    b.bindings = {x};
    report.clear();
    CHECK(!join_pipeline_layout(rs, false, {{GpuShaderStageBits::Vertex, &a}, {GpuShaderStageBits::Fragment, &b}},
                                &untouched, &report));
    CHECK(report.find("the stages disagree") != std::string::npos);
}

}  // namespace

int main() {
    test_parse();
    test_user_data_rule();
    test_vertex_join();
    test_layout_join();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_stage_reflection: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_stage_reflection: all checks passed\n");
    return 0;
}
