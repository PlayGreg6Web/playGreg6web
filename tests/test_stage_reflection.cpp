// Synthetic tests: stage reflection sidecar, user-data -> root parameter rule, and the joins that
// build vertex attributes and pipeline layouts for pack-resolved stages (G62, #651).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
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
    // Containing lookup: s10 is dword 1 of the constants (parameter 1); s12 is the high half of the root CBV.
    uint32_t dw = 99;
    CHECK(root_parameter_containing_user_sgpr(rs, false, 10, &dw) == 1 && dw == 1);
    CHECK(root_parameter_containing_user_sgpr(rs, false, 12, &dw) == 2 && dw == 1);
    CHECK(root_parameter_containing_user_sgpr(rs, false, 13, &dw) == -1);
    // Mesh stage: the first root parameter's table is at s16.
    CHECK(root_parameter_for_user_sgpr(rs, false, 16, kMeshUserDataBase) == 0);
    CHECK(root_parameter_for_user_sgpr(rs, false, 17, kMeshUserDataBase) == 1);
    CHECK(root_parameter_for_user_sgpr(rs, false, 8, kMeshUserDataBase) == -1);
    CHECK(root_parameter_containing_user_sgpr(rs, false, 18, &dw, kMeshUserDataBase) == 1 && dw == 1);
}

void test_contained_sgpr_and_bitcast_vertex() {
    const d3d::RootSignatureDesc rs = ui_root_signature();
    StageReflection fs;
    ReflectedBinding b;
    b.group = 0;
    b.binding = 0;
    b.kind = ReflectedBindingKind::Uniform;
    b.source = ReflectedSource::RootConstants;
    b.user_sgpr = 10;  // second dword of the 2-constant parameter
    fs.bindings = {b};
    ReflectedStageRef ref;
    ref.reflection = &fs;
    bridge::PipelineLayoutDesc out;
    std::string report;
    CHECK(join_pipeline_layout(rs, false, {ref}, &out, &report, nullptr, JoinOptions{}));
    // A table read through the middle of a root CBV is refused with a precise reason.
    fs.bindings[0].source = ReflectedSource::Table;
    fs.bindings[0].user_sgpr = 12;
    fs.bindings[0].table_byte_offset = 1056;
    report.clear();
    CHECK(!join_pipeline_layout(rs, false, {ref}, &out, &report, nullptr, JoinOptions{}));
    CHECK(report.find("dword 1 of root parameter 2") != std::string::npos);

    // f32 attribute over a u32 stage input: reinterpreted, not refused.
    std::vector<pso::D3D12InputElementDesc> e(1);
    e[0].SemanticName = "POSITION";
    e[0].Format = pso::DxgiFormat::R32G32_FLOAT;
    StageReflection vs;
    vs.vertex_inputs = {{0, 0, 0, 2, ReflectedComponentType::U32, pso::WgpuVertexFormat::Undefined}};
    std::vector<pso::VertexInputFetch> fetches;
    report.clear();
    CHECK(join_vertex_inputs(vs, e.data(), 1, &fetches, &report));
    CHECK(fetches.size() == 1 && fetches[0].format == pso::WgpuVertexFormat::Uint32x2);
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

    // Failures: a read inside an element without fetch format and a component class mismatch. Each is
    // reported with its key. (Offsets no element covers are defaulted, see test_vertex_default_inputs.)
    StageReflection bad;
    bad.vertex_inputs = {{0, 0, 40, 2, ReflectedComponentType::F32, pso::WgpuVertexFormat::Undefined},
                         {1, 0, 5, 1, ReflectedComponentType::F32, pso::WgpuVertexFormat::Undefined},
                         {2, 0, 0, 4, ReflectedComponentType::U32, pso::WgpuVertexFormat::Undefined},
                         {3, 1, 0, 4, ReflectedComponentType::F32, pso::WgpuVertexFormat::Undefined}};
    report.clear();
    CHECK(!join_vertex_inputs(bad, layout.data(), 3, &fetches, &report));
    CHECK(report.find("@location(1) <- slot 0 + 5: reads inside element COLOR1") != std::string::npos);
    CHECK(report.find("@location(2) <- slot 0 + 0: the stage expects u32, the attribute gives f32") !=
          std::string::npos);
    CHECK(report.find("@location(3)") == std::string::npos);

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

// An input no layout element covers reads D3D12's (0, 0, 0, 1): a constant-default fetch placed at a
// spare instance slot with stride 0, counted in JoinStats.
void test_vertex_default_inputs() {
    const std::vector<pso::D3D12InputElementDesc> layout = ui_input_layout();
    StageReflection vs;
    vs.vertex_inputs = {{0, 0, 8, 2, ReflectedComponentType::F32, pso::WgpuVertexFormat::Undefined},
                        {1, 3, 0, 4, ReflectedComponentType::F32, pso::WgpuVertexFormat::Undefined},
                        {2, 3, 4, 4, ReflectedComponentType::U32, pso::WgpuVertexFormat::Undefined}};
    std::vector<pso::VertexInputFetch> fetches;
    std::string report;
    JoinStats stats;
    CHECK(join_vertex_inputs(vs, layout.data(), 3, &fetches, &report, &stats));
    CHECK(report.empty() && fetches.size() == 3 && stats.vertex_inputs_defaulted == 2);
    CHECK(fetches[1].constant_default && fetches[1].format == pso::WgpuVertexFormat::Float32x4 && fetches[1].offset == 0);
    CHECK(fetches[2].constant_default && fetches[2].format == pso::WgpuVertexFormat::Uint32x4 &&
          fetches[2].offset == pso::kDefaultVertexIntOffset);

    pso::D3D12GraphicsPsoDesc desc{};
    desc.vs = 1;
    desc.ps = 2;
    desc.pInputElementDescs = layout.data();
    desc.NumInputElements = 3;
    desc.NumRenderTargets = 1;
    desc.RTVFormats[0] = pso::DxgiFormat::R8G8B8A8_UNORM;
    desc.vs_input_fetches = fetches.data();
    desc.vs_input_fetch_count = static_cast<uint32_t>(fetches.size());
    const pso::GraphicsPsoTranslation t = pso::translate_graphics_pso(desc, pso::PipelineKeyOptions{});
    CHECK(t.default_vertex_slot_mask == 2u);  // slot 0 carries POSITION, slot 1 is the first spare
    CHECK(t.base.vertex_buffer_count == 2 && t.base.vertex_buffers[1].array_stride == 0 &&
          t.base.vertex_buffers[1].step_mode == pso::WgpuVertexStepMode::Instance &&
          t.base.vertex_buffers[1].attribute_count == 2);
    CHECK((t.diagnostics & (pso::PipelineDiag::VertexLayoutInvalid | pso::PipelineDiag::VertexSlotOutOfRange)) == 0);
}

// #691: menu glyph quads (COLOR unorm8x4 @0, TEXCOORD float2 @4, POSITION float2 @12, stride 20)
// whose stage reads UV and position as one 4-dword fetch at +4. The attribute must be float32x4,
// not TEXCOORD's float32x2 (position would read the (0, 1) defaults and every quad collapse).
void test_vertex_join_wide_fetch() {
    std::vector<pso::D3D12InputElementDesc> e(3);
    e[0].SemanticName = "COLOR";
    e[0].Format = pso::DxgiFormat::R8G8B8A8_UNORM;
    e[1].SemanticName = "TEXCOORD";
    e[1].Format = pso::DxgiFormat::R32G32_FLOAT;
    e[1].AlignedByteOffset = 4;
    e[2].SemanticName = "POSITION";
    e[2].Format = pso::DxgiFormat::R32G32_FLOAT;
    e[2].AlignedByteOffset = 12;
    StageReflection vs;
    vs.vertex_inputs = {{0, 0, 0, 4, ReflectedComponentType::F32, pso::WgpuVertexFormat::Undefined},
                        {1, 0, 4, 4, ReflectedComponentType::F32, pso::WgpuVertexFormat::Undefined}};
    std::vector<pso::VertexInputFetch> fetches;
    std::string report;
    CHECK(join_vertex_inputs(vs, e.data(), 3, &fetches, &report));
    CHECK(fetches.size() == 2);
    if (fetches.size() == 2) {
        CHECK(fetches[0].format == pso::WgpuVertexFormat::Unorm8x4);
        CHECK(fetches[1].offset == 4 && fetches[1].format == pso::WgpuVertexFormat::Float32x4);
    }
    // Without a following element the fetch keeps the element's own format.
    report.clear();
    CHECK(join_vertex_inputs(vs, e.data(), 2, &fetches, &report));
    CHECK(fetches.size() == 2 && fetches[1].format == pso::WgpuVertexFormat::Float32x2);
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
    JoinStats failed_stats;
    CHECK(!join_pipeline_layout(rs, false, {{GpuShaderStageBits::Vertex, &bad}}, &untouched, &report, &failed_stats));
    CHECK(untouched.groups.empty());
    CHECK(failed_stats.table_slots_outside_rs == 0);  // counted only when the join succeeds
    CHECK(report.find("@group(0) @binding(0) <- table at s12 + 0: s12 is dword 1 of root parameter 2") != std::string::npos);
    // A slot declared as another class joins the slot itself (no failure).
    CHECK(report.find("@binding(2)") == std::string::npos);
    CHECK(report.find("byte offset 16") != std::string::npos);
    // Undeclared slots and samplers the root signature lacks are not failures (below).
    CHECK(report.find("@binding(1)") == std::string::npos && report.find("@binding(4)") == std::string::npos);

    // G62: a slot past the table's declared ranges joins the slot itself (console stages read their
    // tables as memory); a static sampler the root signature lacks binds the default sampler.
    StageReflection extra;
    extra.bindings = {off, sampler};
    extra.bindings[0].kind = ReflectedBindingKind::Texture;
    report.clear();
    bridge::PipelineLayoutDesc slots;
    JoinStats stats;
    CHECK(join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &extra}}, &slots, &report, &stats));
    CHECK(report.empty() && slots.groups.size() == 1 && slots.groups[0].bindings.size() == 2);
    CHECK(stats.table_slots_outside_rs == 1);  // the slot past the ranges; the sampler is not a table slot
    if (slots.groups.size() == 1 && slots.groups[0].bindings.size() == 2) {
        const auto& t = slots.groups[0].bindings[0].join;
        CHECK(t.table_parameter == 0 && t.table_slot == 2 && t.register_class == d3d::DescriptorRangeType::Srv);
        const auto& s = slots.groups[0].bindings[1].join;
        CHECK(s.register_class == d3d::DescriptorRangeType::Sampler && s.shader_register == d3d::kNoIndex &&
              s.table_parameter == d3d::kNoIndex);
    }
    // A sampler cannot come from an undeclared slot of a resource table.
    StageReflection wrong;
    ReflectedBinding smp_slot = off;
    smp_slot.kind = ReflectedBindingKind::Sampler;
    wrong.bindings = {smp_slot};
    report.clear();
    CHECK(!join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &wrong}}, &slots, &report));
    CHECK(report.find("is a resource table") != std::string::npos);

    // A writable binding over a declared CBV joins the slot as a UAV (#482: shared root signatures
    // list a CBV where the stage's own space-1 layout has a UAV; the stage stores through it).
    StageReflection writable;
    ReflectedBinding st = tex;  // slot 1 = CBV b0 space 1
    st.kind = ReflectedBindingKind::StorageTexture;
    st.storage_format = 18;
    ReflectedBinding sb = tex;
    sb.binding = 5;
    sb.kind = ReflectedBindingKind::Storage;
    sb.min_binding_size = 16;
    writable.bindings = {st, sb};
    report.clear();
    JoinStats cbv_stats;
    CHECK(join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &writable}}, &slots, &report, &cbv_stats));
    CHECK(report.empty() && cbv_stats.table_slots_outside_rs == 2);
    if (slots.groups.size() == 1 && slots.groups[0].bindings.size() == 2) {
        const bridge::LayoutBinding& w = slots.groups[0].bindings[0];
        CHECK(w.entry.binding == 2 && w.join.table_parameter == 0 && w.join.table_slot == 1 &&
              w.join.register_class == d3d::DescriptorRangeType::Uav &&
              w.entry.type == static_cast<uint8_t>(GpuBindingLayoutType::StorageTexture) &&
              w.entry.storage_format == 18);
        const bridge::LayoutBinding& s = slots.groups[0].bindings[1];
        CHECK(s.entry.binding == 5 && s.join.table_parameter == 0 && s.join.table_slot == 1 &&
              s.join.register_class == d3d::DescriptorRangeType::Uav &&
              s.entry.type == static_cast<uint8_t>(GpuBindingLayoutType::Buffer) &&
              s.entry.binding_type == static_cast<uint8_t>(GpuBufferBindingType::Storage));
    } else {
        CHECK(false);
    }
    // Over a declared SRV it joins the slot as a UAV (the stage writes through the descriptor, #482:
    // compute stages image_store through s5 + 32 where the root signature declares an SRV).
    StageReflection over_srv;
    ReflectedBinding st_srv = st;
    st_srv.table_byte_offset = 0;  // slot 0 = SRV t2 space 1
    over_srv.bindings = {st_srv};
    report.clear();
    JoinStats srv_stats;
    CHECK(join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &over_srv}}, &slots, &report, &srv_stats));
    CHECK(report.empty() && srv_stats.table_slots_outside_rs == 1);
    if (slots.groups.size() == 1 && slots.groups[0].bindings.size() == 1) {
        const bridge::LayoutBinding& w = slots.groups[0].bindings[0];
        CHECK(w.join.table_parameter == 0 && w.join.table_slot == 0 &&
              w.join.register_class == d3d::DescriptorRangeType::Uav &&
              w.entry.type == static_cast<uint8_t>(GpuBindingLayoutType::StorageTexture) && w.entry.storage_format == 18 &&
              w.entry.binding_type == static_cast<uint8_t>(GpuStorageTextureAccess::WriteOnly));
    } else {
        CHECK(false);
    }

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

// S34 heap sources: parsed, and joined to a root-constants parameter for the index.
void test_heap_sources() {
    const char* doc = R"({
  "schema": "playgta6-stage-reflection", "version": 1,
  "stages": {
    "0x00000000000000c3": {
      "stage": "fragment",
      "bindings": [
        {"group": 0, "binding": 0, "kind": "texture", "view_dimension": "2d-array",
         "source": {"resource_heap": {"base_user_sgpr": 11, "stride": 32, "index": {"user_sgpr": 10, "shift": 5}}}},
        {"group": 0, "binding": 1, "kind": "sampler",
         "source": {"sampler_heap": {"base_user_sgpr": 11, "stride": 16, "index": {"user_sgpr": 9, "shift": 4}}}}
      ]
    }
  }
})";
    StageReflectionIndex index;
    std::string error;
    CHECK(parse_stage_reflection_json(doc, &index, &error));
    const StageReflection& ps = index.at(0xc3);
    CHECK(ps.bindings.size() == 2);
    CHECK(ps.bindings[0].source == ReflectedSource::ResourceHeap && ps.bindings[0].heap_base_sgpr == 11 &&
          ps.bindings[0].heap_stride == 32 && ps.bindings[0].heap_index_sgpr == 10 && ps.bindings[0].heap_shift == 5);
    CHECK(ps.bindings[1].source == ReflectedSource::SamplerHeap && ps.bindings[1].heap_stride == 16 &&
          ps.bindings[1].heap_index_sgpr == 9 && ps.bindings[1].heap_shift == 4);

    StageReflectionIndex bad;
    CHECK(!parse_stage_reflection_json(
        R"({"schema": "playgta6-stage-reflection", "version": 1, "stages": {"0x1": {"stage": "fragment",
           "bindings": [{"group": 0, "binding": 0, "kind": "sampler",
                         "source": {"sampler_heap": {"base_user_sgpr": 11, "stride": 16}}}]}}})",
        &bad, &error));
    CHECK(error.find("sampler_heap needs") != std::string::npos);

    // ui_root_signature: [1] = 2 root constants at s9..s10.
    const d3d::RootSignatureDesc rs = ui_root_signature();
    bridge::PipelineLayoutDesc layout;
    std::string report;
    CHECK(join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &ps}}, &layout, &report));
    CHECK(report.empty());
    const bridge::LayoutBinding* tex = find(layout, 0, 0);
    CHECK(tex != nullptr && tex->join.heap == d3d::HeapAccess::Resource &&
          tex->join.register_class == d3d::DescriptorRangeType::Srv && tex->join.heap_index_parameter == 1 &&
          tex->join.heap_index_dword == 1 && tex->join.heap_shift == 5 &&
          tex->entry.view_dimension == static_cast<uint8_t>(GpuTextureViewDimension::Dim2DArray));
    const bridge::LayoutBinding* smp = find(layout, 0, 1);
    CHECK(smp != nullptr && smp->join.heap == d3d::HeapAccess::Sampler &&
          smp->join.register_class == d3d::DescriptorRangeType::Sampler && smp->join.heap_index_parameter == 1 &&
          smp->join.heap_index_dword == 0 && smp->join.heap_shift == 4);

    // The index outside a constants parameter, and a texture from the sampler heap, are reported.
    StageReflection wrong = ps;
    wrong.bindings[0].heap_index_sgpr = 8;  // the table
    wrong.bindings[1].kind = ReflectedBindingKind::Texture;
    report.clear();
    bridge::PipelineLayoutDesc untouched;
    CHECK(!join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &wrong}}, &untouched, &report));
    CHECK(report.find("s8 is not inside a root-constants parameter") != std::string::npos);
    CHECK(report.find("cannot come from the sampler heap") != std::string::npos);
}

// G62: a direct table-slot join compiles to that table offset, past the declared ranges; a slot
// of a table of the other heap type stays unresolved.
void test_plan_direct_slot() {
    const d3d::RootSignatureDesc desc = ui_root_signature();
    d3d::RootSignature rs;
    CHECK(d3d::RootSignature::build(desc, &rs).ok());
    d3d::ShaderBinding tex;
    tex.binding = 1;
    tex.register_class = d3d::DescriptorRangeType::Srv;
    tex.table_parameter = 0;
    tex.table_slot = 5;
    d3d::ShaderBinding smp = tex;
    smp.binding = 2;
    smp.register_class = d3d::DescriptorRangeType::Sampler;
    const d3d::ShaderBinding bindings[] = {tex, smp};
    const d3d::BindGroupPlan plan = d3d::BindGroupPlan::build(rs, bindings, d3d::WebGpuStage::Fragment);
    CHECK(plan.entries().size() == 2 && plan.unresolved_count() == 1);
    if (plan.entries().size() == 2) {
        CHECK(plan.entries()[0].source == d3d::PlanSource::DescriptorTable && plan.entries()[0].root_parameter_index == 0 &&
              plan.entries()[0].table_offset == 5);
        CHECK(plan.entries()[1].source == d3d::PlanSource::Unresolved);
    }
    CHECK(plan.table_windows().size() == 1 && plan.table_windows()[0].first_offset == 5);
}

// #482 contract: a sampler the stage builds from constants in its code comes as its four S# dwords
// ("inline_sampler": {"sdwords": [...]}) and is bound as the decoded sampler, whatever the root
// signature declares. Hand-made GFX10 S# words: clamp-to-edge bilinear with linear mips (filter 0x15),
// and a LESS_EQUAL comparison sampler with point filtering (filter 0x80).
void test_inline_sampler() {
    const char* doc = R"({
  "schema": "playgta6-stage-reflection", "version": 1,
  "stages": {
    "0x00000000000000d4": {
      "stage": "fragment",
      "bindings": [
        {"group": 0, "binding": 34, "kind": "sampler",
         "source": {"inline_sampler": {"sdwords": [146, 15728640, 139460608, 0]}}},
        {"group": 0, "binding": 36, "kind": "comparison_sampler",
         "source": {"inline_sampler": {"sdwords": [12288, 15728640, 67108864, 0]}}}
      ]
    }
  }
})";
    StageReflectionIndex index;
    std::string error;
    CHECK(parse_stage_reflection_json(doc, &index, &error));
    const StageReflection& ps = index.at(0xd4);
    CHECK(ps.bindings.size() == 2 && ps.bindings[0].source == ReflectedSource::InlineSampler);
    CHECK(ps.bindings[0].sampler_words[0] == 0x92 && ps.bindings[0].sampler_words[1] == 0x00F00000 &&
          ps.bindings[0].sampler_words[2] == 0x08500000 && ps.bindings[0].sampler_words[3] == 0);

    // A root signature without static samplers: the inline samplers join anyway.
    d3d::RootSignatureDesc rs;
    d3d::RootParameterDesc constants;
    constants.parameter_type = d3d::RootParameterType::Constants32Bit;
    constants.num_32bit_values = 1;
    rs.parameters = {constants};
    bridge::PipelineLayoutDesc layout;
    std::string report;
    CHECK(join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &ps}}, &layout, &report));
    CHECK(report.empty());
    const bridge::LayoutBinding* lin = find(layout, 0, 34);
    CHECK(lin != nullptr && lin->join.inline_sampler && lin->join.register_class == d3d::DescriptorRangeType::Sampler);
    if (lin != nullptr) {
        CHECK(lin->join.sampler.filter == 0x15 && lin->join.sampler.address_u == 3 && lin->join.sampler.address_v == 3 &&
              lin->join.sampler.max_lod == 15.0f && lin->join.sampler.comparison_func == 0);
        CHECK(lin->entry.binding_type == static_cast<uint8_t>(GpuSamplerBindingType::Filtering));
    }
    const bridge::LayoutBinding* cmp = find(layout, 0, 36);
    CHECK(cmp != nullptr && cmp->join.inline_sampler);
    if (cmp != nullptr) {
        CHECK(cmp->join.sampler.filter == 0x80 && cmp->join.sampler.comparison_func == 4 &&
              cmp->join.sampler.address_u == 1);
        CHECK(cmp->entry.binding_type == static_cast<uint8_t>(GpuSamplerBindingType::Comparison));
    }

    // Words that are not a sampler (dword 3 bits 12..29 are zero in every S#) and a texture binding
    // with an inline sampler source fail the join, with the words in the key.
    StageReflection bad = ps;
    bad.bindings[0].sampler_words[3] = 0x1000;
    bad.bindings[1].kind = ReflectedBindingKind::Texture;
    report.clear();
    bridge::PipelineLayoutDesc untouched;
    CHECK(!join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &bad}}, &untouched, &report));
    CHECK(report.find("@group(0) @binding(34) <- inline sampler 00000092 00f00000 08500000 00001000: the inline "
                      "sampler words do not decode") != std::string::npos);
    CHECK(report.find("a texture binding cannot take an inline sampler") != std::string::npos);

    // Malformed sdwords are rejected by the parser.
    for (const char* words : {"[1, 2, 3]", "[1, 2, 3, -1]", "[1, 2, 3, 4294967296]", "[1, 2, 3, 0.5]", "7"}) {
        const std::string bad_doc = std::string(R"({"schema": "playgta6-stage-reflection", "version": 1, "stages": {
            "0x1": {"stage": "fragment", "bindings": [{"group": 0, "binding": 0, "kind": "sampler",
            "source": {"inline_sampler": {"sdwords": )") + words + "}}}]}}}";
        StageReflectionIndex rejected;
        CHECK(!parse_stage_reflection_json(bad_doc, &rejected, &error));
        CHECK(error.find("inline_sampler needs sdwords") != std::string::npos);
    }
}

}  // namespace

// G62 (#482): CB-indexed heap sources (SM 6.6 ResourceDescriptorHeap[cb.index]) parse and join to
// the root parameter holding the CB: a root CBV, a table slot's CBV or root constants.
void test_cbv_indexed_heap_sources() {
    const char* doc = R"({
  "schema": "playgta6-stage-reflection", "version": 1,
  "stages": {
    "0x00000000000000c4": {
      "stage": "fragment",
      "bindings": [
        {"group": 0, "binding": 0, "kind": "texture",
         "source": {"cbv_indexed_resource_heap": {"cbv": {"user_sgpr": 11, "root_descriptor": true},
                                                  "byte_offset": 16, "shift": 5, "stride": 32}}},
        {"group": 0, "binding": 1, "kind": "texture",
         "source": {"cbv_indexed_resource_heap": {"cbv": {"user_sgpr": 8, "table_byte_offset": 32},
                                                  "byte_offset": 4, "shift": 5, "bit_offset": 16, "mask": 65535}}},
        {"group": 0, "binding": 2, "kind": "sampler",
         "source": {"cbv_indexed_sampler_heap": {"cbv": {"user_sgpr": 9, "root_constants": true},
                                                 "byte_offset": 4, "shift": 4}}}
      ]
    }
  }
})";
    StageReflectionIndex index;
    std::string error;
    CHECK(parse_stage_reflection_json(doc, &index, &error));
    const StageReflection& ps = index.at(0xc4);
    CHECK(ps.bindings.size() == 3);
    const ReflectedBinding& b0 = ps.bindings[0];
    CHECK(b0.source == ReflectedSource::CbvIndexedResourceHeap && b0.cbv == ReflectedCbv::RootDescriptor &&
          b0.user_sgpr == 11 && b0.cb_byte_offset == 16 && b0.heap_shift == 5 && b0.heap_stride == 32 &&
          b0.cb_bit_offset == 0 && b0.cb_mask == 0xffffffffu);
    const ReflectedBinding& b1 = ps.bindings[1];
    CHECK(b1.source == ReflectedSource::CbvIndexedResourceHeap && b1.cbv == ReflectedCbv::Table &&
          b1.table_byte_offset == 32 && b1.cb_bit_offset == 16 && b1.cb_mask == 0xffff);
    const ReflectedBinding& b2 = ps.bindings[2];
    CHECK(b2.source == ReflectedSource::CbvIndexedSamplerHeap && b2.cbv == ReflectedCbv::RootConstants &&
          b2.user_sgpr == 9 && b2.cb_byte_offset == 4 && b2.heap_shift == 4);

    // Malformed: no cbv, unaligned byte offset, shift out of range.
    for (const char* src : {R"({"cbv_indexed_resource_heap": {"byte_offset": 0, "shift": 5}})",
                            R"({"cbv_indexed_resource_heap": {"cbv": {"user_sgpr": 11, "root_descriptor": true}, "byte_offset": 2, "shift": 5}})",
                            R"({"cbv_indexed_resource_heap": {"cbv": {"user_sgpr": 11, "root_descriptor": true}, "byte_offset": 0, "shift": 32}})",
                            R"({"cbv_indexed_resource_heap": {"cbv": {"user_sgpr": 8}, "byte_offset": 0, "shift": 5}})"}) {
        const std::string bad_doc = std::string(R"({"schema": "playgta6-stage-reflection", "version": 1, "stages": {"0x1": {"stage": "fragment", "bindings": [{"group": 0, "binding": 0, "kind": "texture", "source": )") +
                                    src + "}]}}}";
        StageReflectionIndex bad;
        error.clear();
        CHECK(!parse_stage_reflection_json(bad_doc, &bad, &error));
        CHECK(error.find("cbv_indexed_resource_heap needs") != std::string::npos);
    }

    // ui_root_signature: table at s8 (p0), 2 constants at s9..s10 (p1), root CBV at s11 (p2).
    const d3d::RootSignatureDesc rs = ui_root_signature();
    bridge::PipelineLayoutDesc layout;
    std::string report;
    CHECK(join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &ps}}, &layout, &report));
    CHECK(report.empty());
    const bridge::LayoutBinding* t0 = find(layout, 0, 0);
    CHECK(t0 != nullptr && t0->join.heap == d3d::HeapAccess::Resource &&
          t0->join.heap_index_source == d3d::HeapIndexSource::CbRootDescriptor &&
          t0->join.register_class == d3d::DescriptorRangeType::Srv && t0->join.heap_index_parameter == 2 &&
          t0->join.cb_byte_offset == 16 && t0->join.heap_shift == 5);
    const bridge::LayoutBinding* t1 = find(layout, 0, 1);
    CHECK(t1 != nullptr && t1->join.heap_index_source == d3d::HeapIndexSource::CbTable &&
          t1->join.heap_index_parameter == 0 && t1->join.cb_table_slot == 1 && t1->join.cb_byte_offset == 4 &&
          t1->join.cb_bit_offset == 16 && t1->join.cb_mask == 0xffff);
    const bridge::LayoutBinding* s2 = find(layout, 0, 2);
    CHECK(s2 != nullptr && s2->join.heap == d3d::HeapAccess::Sampler &&
          s2->join.heap_index_source == d3d::HeapIndexSource::CbRootConstants &&
          s2->join.register_class == d3d::DescriptorRangeType::Sampler && s2->join.heap_index_parameter == 1 &&
          s2->join.cb_byte_offset == 4 && s2->join.heap_shift == 4);

    // The plan the bridge builds from these joins resolves all three as heap-indexed entries.
    d3d::RootSignature built;
    CHECK(d3d::RootSignature::build(rs, &built).ok());
    const d3d::ShaderBinding joins[] = {t0->join, t1->join, s2->join};
    const d3d::BindGroupPlan plan = d3d::BindGroupPlan::build(built, joins, d3d::WebGpuStage::Fragment);
    CHECK(plan.unresolved_count() == 0 && plan.parameter_mask() == 7);

    // Wrong parameter types, a constants offset past the parameter, a sampler from the resource heap.
    StageReflection wrong = ps;
    wrong.bindings[0].user_sgpr = 9;   // constants, not a root CBV
    wrong.bindings[1].user_sgpr = 11;  // root CBV, not a table
    wrong.bindings[2].cb_byte_offset = 8;
    report.clear();
    bridge::PipelineLayoutDesc untouched;
    CHECK(!join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &wrong}}, &untouched, &report));
    CHECK(report.find("is not a root CBV") != std::string::npos);
    CHECK(report.find("is not a CBV_SRV_UAV descriptor table") != std::string::npos);
    CHECK(report.find("is not root constants covering byte 8") != std::string::npos);
    StageReflection wrong_heap = ps;
    wrong_heap.bindings[2].source = ReflectedSource::CbvIndexedResourceHeap;
    wrong_heap.bindings[1].user_sgpr = 12;  // no parameter starts there
    report.clear();
    CHECK(!join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &wrong_heap}}, &untouched, &report));
    CHECK(report.find("cannot come from the resource heap") != std::string::npos);
    CHECK(report.find("no root parameter starts at s12") != std::string::npos);
}

// G62 (#482): lane-indexed heap windows (VGPR / waterfall bindless) parse and validate, and the
// join refuses them with their own, counted reason until the bridge can bind descriptor arrays.
void test_lane_indexed_heap_sources() {
    const char* doc = R"({
  "schema": "playgta6-stage-reflection", "version": 1,
  "stages": {
    "0x00000000000000c5": {
      "stage": "fragment",
      "bindings": [
        {"group": 0, "binding": 0, "kind": "texture",
         "source": {"lane_indexed_resource_heap": {"max_count": 64, "shift": 5, "stride": 32}}},
        {"group": 0, "binding": 1, "kind": "sampler",
         "source": {"lane_indexed_sampler_heap": {"max_count": 16, "shift": 4,
                    "window_base": {"cbv": {"user_sgpr": 11, "root_descriptor": true}, "byte_offset": 8,
                                    "bit_offset": 8, "mask": 255}}}}
      ]
    }
  }
})";
    StageReflectionIndex index;
    std::string error;
    CHECK(parse_stage_reflection_json(doc, &index, &error));
    const StageReflection& ps = index.at(0xc5);
    CHECK(ps.bindings.size() == 2);
    const ReflectedBinding& b0 = ps.bindings[0];
    CHECK(b0.source == ReflectedSource::LaneIndexedResourceHeap && b0.lane_max_count == 64 && b0.heap_shift == 5 &&
          b0.heap_stride == 32 && !b0.lane_has_window_base);
    const ReflectedBinding& b1 = ps.bindings[1];
    CHECK(b1.source == ReflectedSource::LaneIndexedSamplerHeap && b1.lane_max_count == 16 && b1.heap_shift == 4 &&
          b1.lane_has_window_base && b1.cbv == ReflectedCbv::RootDescriptor && b1.user_sgpr == 11 &&
          b1.cb_byte_offset == 8 && b1.cb_bit_offset == 8 && b1.cb_mask == 255);

    // Malformed: no max_count, zero or oversized window, shift out of range, bad window base.
    for (const char* src :
         {R"({"lane_indexed_resource_heap": {"shift": 5}})",
          R"({"lane_indexed_resource_heap": {"max_count": 0, "shift": 5}})",
          R"({"lane_indexed_resource_heap": {"max_count": 257, "shift": 5}})",
          R"({"lane_indexed_resource_heap": {"max_count": 8, "shift": 32}})",
          R"({"lane_indexed_resource_heap": {"max_count": 8, "shift": 5, "window_base": {"byte_offset": 0}}})",
          R"({"lane_indexed_resource_heap": {"max_count": 8, "shift": 5, "window_base": {"cbv": {"user_sgpr": 11, "root_descriptor": true}, "byte_offset": 6}}})",
          R"({"lane_indexed_resource_heap": 3})"}) {
        const std::string bad_doc = std::string(R"({"schema": "playgta6-stage-reflection", "version": 1, "stages": {"0x1": {"stage": "fragment", "bindings": [{"group": 0, "binding": 0, "kind": "texture", "source": )") +
                                    src + "}]}}}";
        StageReflectionIndex bad;
        error.clear();
        CHECK(!parse_stage_reflection_json(bad_doc, &bad, &error));
        CHECK(error.find("lane_indexed_resource_heap needs") != std::string::npos);
    }

    // Well formed without binding_arrays: degrades to scalar heap bindings at the window base
    // (window_base + 0, or constant slot 0 without window_base), counted in lane_indexed_bindings.
    const d3d::RootSignatureDesc rs = ui_root_signature();
    bridge::PipelineLayoutDesc layout;
    std::string report;
    JoinStats stats;
    CHECK(join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &ps}}, &layout, &report, &stats));
    CHECK(report.empty());
    CHECK(stats.lane_indexed_bindings == 2 && stats.lane_indexed_arrays == 0 && stats.table_slots_outside_rs == 0);
    CHECK(layout.groups.size() == 1 && layout.groups[0].bindings.size() == 2);
    if (layout.groups.size() == 1 && layout.groups[0].bindings.size() == 2) {
        const bridge::LayoutBinding& tex = layout.groups[0].bindings[0];
        CHECK(tex.entry.binding_array_size == 0 && tex.join.array_size == 0 && tex.join.array_index == 0);
        CHECK(tex.join.heap == d3d::HeapAccess::Resource && tex.join.heap_shift == 5 &&
              tex.join.heap_index_source == d3d::HeapIndexSource::Constant &&
              tex.join.register_class == d3d::DescriptorRangeType::Srv);
        const bridge::LayoutBinding& smp = layout.groups[0].bindings[1];
        CHECK(smp.entry.binding_array_size == 0 && smp.join.array_size == 0 && smp.join.array_index == 0);
        CHECK(smp.join.heap == d3d::HeapAccess::Sampler && smp.join.heap_shift == 4 &&
              smp.join.heap_index_source == d3d::HeapIndexSource::CbRootDescriptor &&
              smp.join.heap_index_parameter == 2 && smp.join.cb_byte_offset == 8 && smp.join.cb_bit_offset == 8 &&
              smp.join.cb_mask == 255 && smp.join.register_class == d3d::DescriptorRangeType::Sampler);
        d3d::RootSignature built;
        CHECK(d3d::RootSignature::build(rs, &built).ok());
        d3d::ShaderBinding joins[2] = {tex.join, smp.join};
        joins[0].binding = 0;
        joins[1].binding = 1;
        const d3d::BindGroupPlan plan = d3d::BindGroupPlan::build(built, joins, d3d::WebGpuStage::Fragment);
        CHECK(plan.unresolved_count() == 0 && plan.entries().size() == 2);
    }

    // A consumer that binds binding arrays: both windows join as arrays of max_count heap descriptors.
    bridge::PipelineLayoutDesc arrays;
    report.clear();
    JoinStats array_stats;
    CHECK(join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &ps}}, &arrays, &report, &array_stats,
                               JoinOptions{true}));
    CHECK(report.empty());
    CHECK(array_stats.lane_indexed_bindings == 0 && array_stats.lane_indexed_arrays == 2);
    CHECK(arrays.groups.size() == 1 && arrays.groups[0].bindings.size() == 2);
    if (arrays.groups.size() == 1 && arrays.groups[0].bindings.size() == 2) {
        const bridge::LayoutBinding& tex = arrays.groups[0].bindings[0];
        CHECK(tex.entry.binding_array_size == 64 && tex.join.array_size == 64 && tex.join.array_index == 0);
        CHECK(tex.join.heap == d3d::HeapAccess::Resource && tex.join.heap_shift == 5 &&
              tex.join.heap_index_source == d3d::HeapIndexSource::Constant);
        CHECK(tex.join.register_class == d3d::DescriptorRangeType::Srv);
        const bridge::LayoutBinding& smp = arrays.groups[0].bindings[1];
        CHECK(smp.entry.binding_array_size == 16 && smp.join.array_size == 16);
        CHECK(smp.join.heap == d3d::HeapAccess::Sampler && smp.join.heap_shift == 4 &&
              smp.join.heap_index_source == d3d::HeapIndexSource::CbRootDescriptor && smp.join.cb_byte_offset == 8 &&
              smp.join.cb_bit_offset == 8 && smp.join.cb_mask == 255);
        CHECK(smp.join.register_class == d3d::DescriptorRangeType::Sampler);
    }

    // Malformed joins are not counted as lane-indexed: wrong heap for the kind, a window base that is
    // not a root CBV.
    StageReflection wrong = ps;
    wrong.bindings[0].source = ReflectedSource::LaneIndexedSamplerHeap;
    wrong.bindings[1].user_sgpr = 9;  // root constants, not a root CBV
    report.clear();
    JoinStats wrong_stats;
    CHECK(!join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &wrong}}, &layout, &report, &wrong_stats));
    CHECK(wrong_stats.lane_indexed_bindings == 0);
    CHECK(report.find("cannot come from the sampler heap") != std::string::npos);
    CHECK(report.find("window base: root parameter 1 is not a root CBV") != std::string::npos);

    // A stage without lane-indexed bindings joined next to one with them succeeds via scalar fallback.
    StageReflection vs;
    vs.stage = ReflectedStage::Vertex;
    report.clear();
    JoinStats mixed;
    CHECK(join_pipeline_layout(rs, false, {{GpuShaderStageBits::Vertex, &vs}, {GpuShaderStageBits::Fragment, &ps}},
                               &layout, &report, &mixed));
    CHECK(report.empty() && mixed.lane_indexed_bindings == 2 && mixed.lane_indexed_arrays == 0);
}

// S63 (#482, #1394): the translator's exact lane-indexed sidecar sources (rdna2_to_wgsl.rs tests)
// parse at every sidecar version, and join as binding arrays; a non-power-of-two scale (shift 0,
// stride 24 from s_mul_i32) joins with that stride, a power-of-two stride as the shift.
void test_s63_lane_indexed_sidecar() {
    for (int version : {1, 2, 3}) {
        const std::string doc =
            std::string(R"({"schema": "playgta6-stage-reflection", "version": )") + std::to_string(version) + R"(,
  "stages": {"0x00000000000000d1": {"stage": "fragment", "bindings": [
    {"group": 0, "binding": 0, "kind": "texture",
     "source": {"lane_indexed_resource_heap": {"max_count": 64, "shift": 5, "stride": 32}}},
    {"group": 0, "binding": 1, "kind": "sampler",
     "source": {"lane_indexed_sampler_heap": {"max_count": 16, "shift": 4, "stride": 16}}},
    {"group": 0, "binding": 2, "kind": "texture",
     "source": {"lane_indexed_resource_heap": {"max_count": 64, "shift": 0, "stride": 24}}}]},
  "0x00000000000000d2": {"stage": "fragment", "bindings": [
    {"group": 0, "binding": 0, "kind": "texture",
     "source": {"lane_indexed_resource_heap": {"max_count": 32, "shift": 5, "stride": 32, "window_base": {"cbv": {"user_sgpr": 10, "root_descriptor": true}, "byte_offset": 8}}}},
    {"group": 0, "binding": 1, "kind": "texture",
     "source": {"lane_indexed_resource_heap": {"max_count": 8, "shift": 5, "stride": 32, "window_base": {"cbv": {"user_sgpr": 8, "table_byte_offset": 64}, "byte_offset": 4, "bit_offset": 8, "mask": 255}}}}]}}})";
        StageReflectionIndex index;
        std::string error;
        CHECK(parse_stage_reflection_json(doc, &index, &error));
        if (index.size() != 2) {
            continue;
        }
        const StageReflection& ps = index.at(0xd1);
        CHECK(ps.bindings.size() == 3);
        CHECK(ps.bindings[2].source == ReflectedSource::LaneIndexedResourceHeap && ps.bindings[2].heap_shift == 0 &&
              ps.bindings[2].heap_stride == 24 && ps.bindings[2].lane_max_count == 64);
        const StageReflection& based = index.at(0xd2);
        CHECK(based.bindings.size() == 2);
        const ReflectedBinding& root = based.bindings[0];
        CHECK(root.lane_has_window_base && root.cbv == ReflectedCbv::RootDescriptor && root.user_sgpr == 10 &&
              root.cb_byte_offset == 8 && root.cb_bit_offset == 0 && root.cb_mask == 0xffffffffu &&
              root.lane_max_count == 32);
        const ReflectedBinding& table = based.bindings[1];
        CHECK(table.lane_has_window_base && table.cbv == ReflectedCbv::Table && table.user_sgpr == 8 &&
              table.table_byte_offset == 64 && table.cb_byte_offset == 4 && table.cb_bit_offset == 8 &&
              table.cb_mask == 255 && table.lane_max_count == 8);

        const d3d::RootSignatureDesc rs = ui_root_signature();
        bridge::PipelineLayoutDesc layout;
        std::string report;
        JoinStats stats;
        CHECK(join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &ps}}, &layout, &report, &stats,
                                   JoinOptions{true}));
        CHECK(report.empty() && stats.lane_indexed_arrays == 3 && stats.lane_indexed_bindings == 0);
        CHECK(layout.groups.size() == 1 && layout.groups[0].bindings.size() == 3);
        if (layout.groups.size() == 1 && layout.groups[0].bindings.size() == 3) {
            const auto& b = layout.groups[0].bindings;
            CHECK(b[0].entry.binding_array_size == 64 && b[0].join.heap_shift == 5 && b[0].join.heap_stride == 0);
            CHECK(b[1].entry.binding_array_size == 16 && b[1].join.heap == d3d::HeapAccess::Sampler &&
                  b[1].join.heap_shift == 4 && b[1].join.heap_stride == 0);
            CHECK(b[2].entry.binding_array_size == 64 && b[2].join.heap_shift == 0 && b[2].join.heap_stride == 24 &&
                  b[2].join.array_size == 64);
        }
        // Without binding arrays the same stage degrades to scalar heap bindings at slot 0,
        // preserving non-power-of-two stride and counting all three in lane_indexed_bindings.
        report.clear();
        JoinStats off;
        CHECK(join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &ps}}, &layout, &report, &off));
        CHECK(report.empty() && off.lane_indexed_bindings == 3 && off.lane_indexed_arrays == 0);
        if (layout.groups.size() == 1 && layout.groups[0].bindings.size() == 3) {
            const auto& b = layout.groups[0].bindings;
            CHECK(b[0].entry.binding_array_size == 0 && b[0].join.array_size == 0 &&
                  b[0].join.heap_index_source == d3d::HeapIndexSource::Constant && b[0].join.heap_shift == 5);
            CHECK(b[1].entry.binding_array_size == 0 && b[1].join.array_size == 0 &&
                  b[1].join.heap == d3d::HeapAccess::Sampler && b[1].join.heap_shift == 4);
            CHECK(b[2].entry.binding_array_size == 0 && b[2].join.array_size == 0 &&
                  b[2].join.heap_shift == 0 && b[2].join.heap_stride == 24);
        }
    }
}

// Root-descriptor sources over a descriptor table (G62, #482): the user SGPR starts a table, so
// the binding joins as a table slot: slot 0 by default, the sidecar's table_byte_offset from
// version 2 (ignored in version 1).
std::string root_table_sidecar(int version) {
    return std::string(R"({"schema": "playgta6-stage-reflection", "version": )") + std::to_string(version) +
           R"(, "stages": {"0xc3": {"stage": "fragment", "bindings": [
             {"group": 0, "binding": 0, "kind": "read_only_storage", "source": {"user_sgpr": 8, "root_descriptor": true}},
             {"group": 0, "binding": 1, "kind": "uniform",
              "source": {"user_sgpr": 8, "root_descriptor": true, "table_byte_offset": 32}},
             {"group": 0, "binding": 2, "kind": "uniform", "source": {"user_sgpr": 11, "root_descriptor": true}}]}}})";
}

void test_root_descriptor_over_table() {
    const d3d::RootSignatureDesc rs = ui_root_signature();
    std::string error;

    // Version 1: the offset is not part of the format; both table bindings join at slot 0 (SRV t2).
    StageReflectionIndex v1;
    CHECK(parse_stage_reflection_json(root_table_sidecar(1), &v1, &error));
    const StageReflection& ps1 = v1.at(0xc3);
    CHECK(ps1.bindings.size() == 3 && ps1.bindings[1].source == ReflectedSource::RootDescriptor &&
          !ps1.bindings[1].root_table_offset_given && ps1.bindings[1].table_byte_offset == 0);
    bridge::PipelineLayoutDesc layout;
    std::string report;
    JoinStats stats;
    StageReflection only_srv = ps1;
    only_srv.bindings.erase(only_srv.bindings.begin() + 1);
    CHECK(join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &only_srv}}, &layout, &report, &stats));
    CHECK(report.empty());
    CHECK(stats.root_descriptor_tables == 1 && stats.table_slots_outside_rs == 0);
    const bridge::LayoutBinding* b0 = find(layout, 0, 0);
    CHECK(b0 != nullptr && b0->join.register_class == d3d::DescriptorRangeType::Srv && b0->join.shader_register == 2 &&
          b0->join.register_space == 1 && b0->join.table_parameter == d3d::kNoIndex);
    // A real root CBV keeps its root-descriptor join and is not counted.
    const bridge::LayoutBinding* b2 = find(layout, 0, 2);
    CHECK(b2 != nullptr && b2->join.register_class == d3d::DescriptorRangeType::Cbv && b2->join.shader_register == 7);
    // v1 ignores the offset: the uniform binding also joins slot 0 (SRV t2, a buffer view).
    stats = JoinStats{};
    report.clear();
    CHECK(join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &ps1}}, &layout, &report, &stats));
    CHECK(stats.root_descriptor_tables == 2 && stats.table_slots_outside_rs == 0);
    const bridge::LayoutBinding* u1 = find(layout, 0, 1);
    CHECK(u1 != nullptr && u1->join.register_class == d3d::DescriptorRangeType::Srv && u1->join.shader_register == 2 &&
          u1->join.register_space == 1);
    // A slot past the declared ranges joins by slot.
    StageReflection past = ps1;
    past.bindings = {ps1.bindings[0]};
    past.bindings[0].root_table_offset_given = true;
    past.bindings[0].table_byte_offset = 64;
    stats = JoinStats{};
    CHECK(join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &past}}, &layout, &report, &stats));
    CHECK(stats.root_descriptor_tables == 1 && stats.table_slots_outside_rs == 1);
    const bridge::LayoutBinding* p0 = find(layout, 0, 0);
    CHECK(p0 != nullptr && p0->join.table_parameter == 0 && p0->join.table_slot == 2);

    // Version 2: the given table_byte_offset picks slot 1 (CBV b0 space 1).
    StageReflectionIndex v2;
    CHECK(parse_stage_reflection_json(root_table_sidecar(2), &v2, &error));
    const StageReflection& ps2 = v2.at(0xc3);
    CHECK(ps2.bindings[1].root_table_offset_given && ps2.bindings[1].table_byte_offset == 32);
    CHECK(!ps2.bindings[0].root_table_offset_given);
    stats = JoinStats{};
    report.clear();
    CHECK(join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &ps2}}, &layout, &report, &stats));
    CHECK(report.empty());
    CHECK(stats.root_descriptor_tables == 2 && stats.table_slots_outside_rs == 0);
    u1 = find(layout, 0, 1);
    CHECK(u1 != nullptr && u1->join.register_class == d3d::DescriptorRangeType::Cbv && u1->join.shader_register == 0 &&
          u1->join.register_space == 1 && u1->join.table_parameter == d3d::kNoIndex);

    // A misaligned offset fails the join like a table source; an unknown version is rejected.
    StageReflection bad = ps2;
    bad.bindings[1].table_byte_offset = 36;
    report.clear();
    CHECK(!join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &bad}}, &layout, &report));
    CHECK(report.find("not a descriptor boundary") != std::string::npos);
    StageReflectionIndex v4;
    CHECK(!parse_stage_reflection_json(root_table_sidecar(4), &v4, &error));
    CHECK(error.find("version") != std::string::npos);
    // Root constants are still not a root descriptor.
    StageReflection consts = ps1;
    consts.bindings = {ps1.bindings[0]};
    consts.bindings[0].user_sgpr = 9;
    report.clear();
    CHECK(!join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &consts}}, &layout, &report));
    CHECK(report.find("is not a root descriptor") != std::string::npos);
}

// G66: a mesh stage's compute translation carries its output record layout.
void test_mesh_output() {
    StageReflectionIndex index;
    std::string error;
    CHECK(parse_stage_reflection_json(
        R"({"schema": "playgta6-stage-reflection", "version": 1, "stages": {"0x4d535343000000d7": {"stage": "compute",
            "mesh_output": {"max_vertices": 64, "max_primitives": 126, "vertex_words": 12}}}})",
        &index, &error));
    const auto it = index.find(0x4d535343000000d7ull);
    CHECK(it != index.end() && it->second.has_mesh_output && it->second.mesh_output.max_vertices == 64 &&
          it->second.mesh_output.max_primitives == 126 && it->second.mesh_output.vertex_words == 12);
    StageReflectionIndex bad;
    CHECK(!parse_stage_reflection_json(
        R"({"schema": "playgta6-stage-reflection", "version": 1, "stages": {"0x1": {"stage": "compute",
            "mesh_output": {"max_vertices": 64, "max_primitives": 126, "vertex_words": 10}}}})",
        &bad, &error));
}

// G62 / #482 (sidecar version 3): a binding whose descriptor origin the translator could not trace
// comes with an "unresolved" source. The stage still joins: the binding is bound as the worker's
// default for its kind (no register), counted in JoinStats::unresolved_bindings.
void test_unresolved_source() {
    const char* doc = R"({
  "schema": "playgta6-stage-reflection", "version": 3,
  "stages": {
    "0x00000000000000e1": {
      "stage": "fragment",
      "bindings": [
        {"group": 0, "binding": 0, "kind": "uniform", "source": {"user_sgpr": 8, "root_constants": true}},
        {"group": 0, "binding": 1, "kind": "texture", "view_dimension": "2d", "source": {"unresolved": true}},
        {"group": 0, "binding": 2, "kind": "sampler", "source": {"unresolved": "cb_dynamic_offset"}}
      ]
    }
  }
})";
    StageReflectionIndex index;
    std::string error;
    CHECK(parse_stage_reflection_json(doc, &index, &error));
    const StageReflection& ps = index.at(0xe1);
    CHECK(ps.bindings.size() == 3 && ps.bindings[1].source == ReflectedSource::Unresolved &&
          ps.bindings[2].source == ReflectedSource::Unresolved);

    d3d::RootSignatureDesc desc;
    d3d::RootParameterDesc constants;
    constants.parameter_type = d3d::RootParameterType::Constants32Bit;
    constants.num_32bit_values = 4;
    desc.parameters = {constants};
    bridge::PipelineLayoutDesc layout;
    std::string report;
    JoinStats stats;
    CHECK(join_pipeline_layout(desc, false, {{GpuShaderStageBits::Fragment, &ps}}, &layout, &report, &stats));
    CHECK(report.empty() && stats.unresolved_bindings == 2);
    const bridge::LayoutBinding* tex = find(layout, 0, 1);
    CHECK(tex != nullptr && tex->join.register_class == d3d::DescriptorRangeType::Srv &&
          tex->join.shader_register == d3d::kNoIndex && tex->join.register_space == d3d::kNoIndex &&
          tex->join.heap == d3d::HeapAccess::None && tex->join.table_parameter == d3d::kNoIndex &&
          tex->entry.type == static_cast<uint8_t>(GpuBindingLayoutType::Texture));
    const bridge::LayoutBinding* samp = find(layout, 0, 2);
    CHECK(samp != nullptr && samp->join.register_class == d3d::DescriptorRangeType::Sampler &&
          samp->join.shader_register == d3d::kNoIndex);
    // The planner finds no slot for them (bound as the worker's default); the root constants resolve.
    d3d::RootSignature rs;
    CHECK(d3d::RootSignature::build(desc, &rs).ok());
    std::vector<d3d::ShaderBinding> bindings;
    for (const auto& g : layout.groups) {
        for (const auto& lb : g.bindings) {
            d3d::ShaderBinding sb = lb.join;
            sb.binding = lb.entry.binding;
            bindings.push_back(sb);
        }
    }
    const d3d::BindGroupPlan plan = d3d::BindGroupPlan::build(rs, bindings, d3d::WebGpuStage::Fragment);
    CHECK(plan.entries().size() == 3 && plan.unresolved_count() == 2);

    // Before version 3, not literally true, or on a writable kind: malformed.
    const char* bad_docs[] = {
        R"({"schema": "playgta6-stage-reflection", "version": 2, "stages": {"0x1": {"stage": "fragment",
            "bindings": [{"group": 0, "binding": 0, "kind": "texture", "source": {"unresolved": true}}]}}})",
        R"({"schema": "playgta6-stage-reflection", "version": 3, "stages": {"0x1": {"stage": "fragment",
            "bindings": [{"group": 0, "binding": 0, "kind": "texture", "source": {"unresolved": false}}]}}})",
        R"({"schema": "playgta6-stage-reflection", "version": 3, "stages": {"0x1": {"stage": "fragment",
            "bindings": [{"group": 0, "binding": 0, "kind": "texture", "source": {"unresolved": ""}}]}}})",
        R"({"schema": "playgta6-stage-reflection", "version": 3, "stages": {"0x1": {"stage": "compute",
            "bindings": [{"group": 0, "binding": 0, "kind": "storage", "source": {"unresolved": true}}]}}})",
        R"({"schema": "playgta6-stage-reflection", "version": 3, "stages": {"0x1": {"stage": "compute",
            "bindings": [{"group": 0, "binding": 0, "kind": "storage_texture", "storage_format": 18,
                          "source": {"unresolved": true}}]}}})",
        R"({"schema": "playgta6-stage-reflection", "version": 4, "stages": {}})",
    };
    for (const char* bad_doc : bad_docs) {
        StageReflectionIndex bad;
        CHECK(!parse_stage_reflection_json(bad_doc, &bad, &error));
    }
}

// docs/RAYTRACING_DESIGN.md C3: a stage that reflects the arena bindings (group 3, 16..21) gets the
// runtime-owned group appended; the arena bindings are not joined to the root signature.
void test_rt_arena_bindings() {
    const char* doc = R"({"schema": "playgta6-stage-reflection", "version": 3, "stages": {
      "0x7a": {"stage": "compute", "bindings": [
        {"group": 3, "binding": 16, "kind": "storage", "source": {"rt_arena": true}},
        {"group": 3, "binding": 20, "kind": "read_only_storage", "source": {"rt_arena": true}},
        {"group": 3, "binding": 21, "kind": "storage", "source": {"rt_arena": true}},
        {"group": 0, "binding": 0, "kind": "uniform", "source": {"user_sgpr": 4, "root_constants": true}}]},
      "0x7b": {"stage": "compute", "rt_arena": true, "bindings": [], "address_sources": [{"kind": "root_param", "index": 3, "extent": 4096}, {"kind": "root_constants", "index": 0, "dword": 2, "bvh_root": true}]},
      "0x7c": {"stage": "compute", "bindings": []}}})";
    StageReflectionIndex index;
    std::string error;
    CHECK(parse_stage_reflection_json(doc, &index, &error));
    CHECK(index.at(0x7a).rt_arena_mask == ((1u << 0) | (1u << 4) | (1u << 5)));
    CHECK(index.at(0x7a).bindings.size() == 1);  // the arena bindings are runtime-owned
    CHECK(index.at(0x7b).rt_arena_mask == 0x3f && index.at(0x7c).rt_arena_mask == 0);
    {
        const auto& a = index.at(0x7b).address_sources;
        CHECK(a.size() == 2 && a[0].index == 3 && a[0].extent == 4096 && !a[0].bvh_root);
        CHECK(a[1].dword == 2 && a[1].bvh_root && index.at(0x7a).address_sources.empty());
    }

    d3d::RootSignatureDesc rs;
    d3d::RootParameterDesc rc;
    rc.parameter_type = d3d::RootParameterType::Constants32Bit;
    rc.num_32bit_values = 4;
    rs.parameters.push_back(rc);
    bridge::PipelineLayoutDesc layout;
    std::string report;
    CHECK(join_pipeline_layout(rs, true, {{GpuShaderStageBits::Compute, &index.at(0x7a)}}, &layout, &report));
    CHECK(layout.rt_arena && layout.groups.size() == 1);  // the runtime appends group 3 itself
    bridge::PipelineLayoutDesc plain;
    CHECK(join_pipeline_layout(rs, true, {{GpuShaderStageBits::Compute, &index.at(0x7c)}}, &plain, &report));
    CHECK(!plain.rt_arena);

    // Only group 3, bindings 16..21 may be arena bindings.
    StageReflectionIndex bad;
    CHECK(!parse_stage_reflection_json(
        R"({"schema": "playgta6-stage-reflection", "version": 3, "stages": {"0x1": {"stage": "compute",
            "bindings": [{"group": 3, "binding": 22, "kind": "storage", "source": {"rt_arena": true}}]}}})",
        &bad, &error));
    CHECK(error.find("rt_arena binding") != std::string::npos);
    CHECK(!parse_stage_reflection_json(
        R"({"schema": "playgta6-stage-reflection", "version": 3, "stages": {"0x1": {"stage": "compute",
            "bindings": [{"group": 2, "binding": 16, "kind": "storage", "source": {"rt_arena": true}}]}}})",
        &bad, &error));
    CHECK(!parse_stage_reflection_json(
        R"({"schema": "playgta6-stage-reflection", "version": 3, "stages": {"0x1": {"stage": "compute",
            "bindings": [{"group": 3, "binding": 15, "kind": "storage", "source": {"rt_arena": true}}]}}})",
        &bad, &error));
}

// G83 (#1748 / #651): comprehensive v3 unresolved read-only vs writable rejection, v1/v2 rejection,
// and multi-stage sharing agreement (unresolved + unresolved, resolved + unresolved, mismatches).
void test_g83_unresolved_and_multistage() {
    const char* doc = R"({
  "schema": "playgta6-stage-reflection", "version": 3,
  "stages": {
    "0x0000000000000e10": {
      "stage": "fragment",
      "bindings": [
        {"group": 0, "binding": 0, "kind": "uniform", "min_binding_size_bytes": 64, "source": {"unresolved": true}},
        {"group": 0, "binding": 1, "kind": "read_only_storage", "source": {"unresolved": "dynamic_cb_index"}},
        {"group": 0, "binding": 2, "kind": "texture", "view_dimension": "cube", "source": {"unresolved": true}},
        {"group": 0, "binding": 3, "kind": "sampler", "source": {"unresolved": "inline_dynamic"}},
        {"group": 0, "binding": 4, "kind": "comparison_sampler", "source": {"unresolved": true}}
      ]
    }
  }
})";
    StageReflectionIndex idx;
    std::string error;
    CHECK(parse_stage_reflection_json(doc, &idx, &error));
    CHECK(error.empty() && idx.size() == 1);
    const StageReflection& ps = idx.at(0xe10);
    CHECK(ps.bindings.size() == 5);

    const d3d::RootSignatureDesc rs = ui_root_signature();
    bridge::PipelineLayoutDesc layout;
    std::string report;
    JoinStats stats;
    CHECK(join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &ps}}, &layout, &report, &stats));
    CHECK(report.empty() && stats.unresolved_bindings == 5);
    const bridge::LayoutBinding* u0 = find(layout, 0, 0);
    const bridge::LayoutBinding* ro1 = find(layout, 0, 1);
    const bridge::LayoutBinding* t2 = find(layout, 0, 2);
    const bridge::LayoutBinding* s3 = find(layout, 0, 3);
    const bridge::LayoutBinding* cs4 = find(layout, 0, 4);
    CHECK(u0 != nullptr && u0->join.register_class == d3d::DescriptorRangeType::Cbv &&
          u0->join.shader_register == d3d::kNoIndex && u0->entry.min_binding_size == 64);
    CHECK(ro1 != nullptr && ro1->join.register_class == d3d::DescriptorRangeType::Srv &&
          ro1->join.shader_register == d3d::kNoIndex);
    CHECK(t2 != nullptr && t2->join.register_class == d3d::DescriptorRangeType::Srv &&
          t2->entry.view_dimension == static_cast<uint8_t>(GpuTextureViewDimension::Cube));
    CHECK(s3 != nullptr && s3->join.register_class == d3d::DescriptorRangeType::Sampler &&
          s3->entry.binding_type == static_cast<uint8_t>(GpuSamplerBindingType::Filtering));
    CHECK(cs4 != nullptr && cs4->join.register_class == d3d::DescriptorRangeType::Sampler &&
          cs4->entry.binding_type == static_cast<uint8_t>(GpuSamplerBindingType::Comparison));

    // Reject v1, v2, mixed-key unresolved source, and writable kinds in JSON.
    const char* bad_json[] = {
        R"({"schema": "playgta6-stage-reflection", "version": 1, "stages": {"0x1": {"stage": "fragment",
            "bindings": [{"group": 0, "binding": 0, "kind": "uniform", "source": {"unresolved": true}}]}}})",
        R"({"schema": "playgta6-stage-reflection", "version": 2, "stages": {"0x1": {"stage": "fragment",
            "bindings": [{"group": 0, "binding": 0, "kind": "texture", "source": {"unresolved": "reason"}}]}}})",
        R"({"schema": "playgta6-stage-reflection", "version": 3, "stages": {"0x1": {"stage": "fragment",
            "bindings": [{"group": 0, "binding": 0, "kind": "texture",
                          "source": {"unresolved": true, "user_sgpr": 8}}]}}})",
    };
    for (const char* bdoc : bad_json) {
        StageReflectionIndex rejected;
        error.clear();
        CHECK(!parse_stage_reflection_json(bdoc, &rejected, &error));
        CHECK(!error.empty());
    }

    // Programmatic writable unresolved bindings are rejected cleanly in join_pipeline_layout too.
    for (const ReflectedBindingKind wk : {ReflectedBindingKind::Storage, ReflectedBindingKind::StorageTexture}) {
        StageReflection st;
        ReflectedBinding wb;
        wb.group = 0;
        wb.binding = 0;
        wb.kind = wk;
        wb.storage_format = 18;
        wb.source = ReflectedSource::Unresolved;
        st.bindings = {wb};
        bridge::PipelineLayoutDesc untouched;
        JoinStats wstats;
        report.clear();
        CHECK(!join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &st}}, &untouched, &report, &wstats));
        CHECK(wstats.unresolved_bindings == 0);
        CHECK(report.find("<- unresolved") != std::string::npos);
        CHECK(report.find("cannot have an unresolved source") != std::string::npos);
    }

    // Multi-stage: two stages sharing the same unresolved binding merge visibility and count ONCE.
    StageReflection vs_u, ps_u;
    vs_u.stage = ReflectedStage::Vertex;
    ps_u.stage = ReflectedStage::Fragment;
    ReflectedBinding ub;
    ub.group = 0;
    ub.binding = 0;
    ub.kind = ReflectedBindingKind::Uniform;
    ub.min_binding_size = 64;
    ub.source = ReflectedSource::Unresolved;
    vs_u.bindings = {ub};
    ub.min_binding_size = 128;
    ps_u.bindings = {ub};
    stats = JoinStats{};
    report.clear();
    CHECK(join_pipeline_layout(rs, false, {{GpuShaderStageBits::Vertex, &vs_u}, {GpuShaderStageBits::Fragment, &ps_u}},
                               &layout, &report, &stats));
    CHECK(report.empty() && stats.unresolved_bindings == 1);
    const bridge::LayoutBinding* shared_u = find(layout, 0, 0);
    CHECK(shared_u != nullptr &&
          shared_u->entry.visibility == (GpuShaderStageBits::Vertex | GpuShaderStageBits::Fragment) &&
          shared_u->entry.min_binding_size == 128 && shared_u->join.shader_register == d3d::kNoIndex);

    // Multi-stage: resolved + unresolved on the same read-only WebGPU entry adopts the resolved
    // source symmetrically (both orders) and does not count as unresolved.
    StageReflection vs_res;
    vs_res.stage = ReflectedStage::Vertex;
    ReflectedBinding rb;
    rb.group = 0;
    rb.binding = 0;
    rb.kind = ReflectedBindingKind::Uniform;
    rb.min_binding_size = 96;
    rb.source = ReflectedSource::Table;
    rb.user_sgpr = 8;
    rb.table_byte_offset = 32;  // CBV b0 space 1
    vs_res.bindings = {rb};
    for (int order = 0; order < 2; ++order) {
        std::vector<ReflectedStageRef> pair =
            order == 0
                ? std::vector<ReflectedStageRef>{{GpuShaderStageBits::Vertex, &vs_res},
                                                 {GpuShaderStageBits::Fragment, &ps_u}}
                : std::vector<ReflectedStageRef>{{GpuShaderStageBits::Fragment, &ps_u},
                                                 {GpuShaderStageBits::Vertex, &vs_res}};
        stats = JoinStats{};
        report.clear();
        CHECK(join_pipeline_layout(rs, false, pair, &layout, &report, &stats));
        CHECK(report.empty() && stats.unresolved_bindings == 0);
        const bridge::LayoutBinding* m = find(layout, 0, 0);
        CHECK(m != nullptr && m->entry.visibility == (GpuShaderStageBits::Vertex | GpuShaderStageBits::Fragment) &&
              m->entry.min_binding_size == 128 && m->join.register_class == d3d::DescriptorRangeType::Cbv &&
              m->join.shader_register == 0 && m->join.register_space == 1);
    }

    // Multi-stage: resolved + unresolved with mismatched entry kind fails cleanly.
    StageReflection ps_wrong_u;
    ps_wrong_u.stage = ReflectedStage::Fragment;
    ReflectedBinding wrong_u;
    wrong_u.group = 0;
    wrong_u.binding = 0;
    wrong_u.kind = ReflectedBindingKind::Texture;
    wrong_u.source = ReflectedSource::Unresolved;
    ps_wrong_u.bindings = {wrong_u};
    report.clear();
    CHECK(!join_pipeline_layout(rs, false, {{GpuShaderStageBits::Vertex, &vs_res}, {GpuShaderStageBits::Fragment, &ps_wrong_u}},
                                &layout, &report));
    CHECK(report.find("the stages disagree") != std::string::npos);

    // Multi-stage: differing multisampled flag or storage format on the same (group, binding) fails.
    StageReflection tex_a, tex_b;
    ReflectedBinding ta;
    ta.group = 0;
    ta.binding = 0;
    ta.kind = ReflectedBindingKind::Texture;
    ta.source = ReflectedSource::Table;
    ta.user_sgpr = 8;
    ta.table_byte_offset = 0;
    ta.multisampled = true;
    tex_a.bindings = {ta};
    ta.multisampled = false;
    tex_b.bindings = {ta};
    report.clear();
    CHECK(!join_pipeline_layout(rs, false, {{GpuShaderStageBits::Vertex, &tex_a}, {GpuShaderStageBits::Fragment, &tex_b}},
                                &layout, &report));
    CHECK(report.find("the stages disagree") != std::string::npos);

    StageReflection st_a, st_b;
    ReflectedBinding sfa;
    sfa.group = 0;
    sfa.binding = 0;
    sfa.kind = ReflectedBindingKind::StorageTexture;
    sfa.storage_format = 18;
    sfa.source = ReflectedSource::Table;
    sfa.user_sgpr = 8;
    sfa.table_byte_offset = 0;
    st_a.bindings = {sfa};
    sfa.storage_format = 24;
    st_b.bindings = {sfa};
    report.clear();
    CHECK(!join_pipeline_layout(rs, false, {{GpuShaderStageBits::Vertex, &st_a}, {GpuShaderStageBits::Fragment, &st_b}},
                                &layout, &report));
    CHECK(report.find("the stages disagree") != std::string::npos);
}

// G83 (#1748 / #651): cbv_indexed_*_heap + lane_indexed_*_heap stride propagation, multi-stage
// sharing agreement, window_base variants, and validation boundaries.
void test_g83_heap_strides_and_arrays() {
    const char* doc = R"({
  "schema": "playgta6-stage-reflection", "version": 3,
  "stages": {
    "0x0000000000000e20": {
      "stage": "vertex",
      "bindings": [
        {"group": 0, "binding": 0, "kind": "texture",
         "source": {"cbv_indexed_resource_heap": {"cbv": {"user_sgpr": 11, "root_descriptor": true},
                                                  "byte_offset": 12, "shift": 0, "stride": 24,
                                                  "bit_offset": 4, "mask": 4095}}},
        {"group": 0, "binding": 1, "kind": "sampler",
         "source": {"cbv_indexed_sampler_heap": {"cbv": {"user_sgpr": 9, "root_constants": true},
                                                 "byte_offset": 4, "shift": 0, "stride": 48}}},
        {"group": 0, "binding": 2, "kind": "texture",
         "source": {"lane_indexed_resource_heap": {"max_count": 256, "shift": 0, "stride": 24,
                    "window_base": {"cbv": {"user_sgpr": 8, "table_byte_offset": 32},
                                    "byte_offset": 8, "bit_offset": 8, "mask": 255}}}}
      ]
    },
    "0x0000000000000e21": {
      "stage": "fragment",
      "bindings": [
        {"group": 0, "binding": 2, "kind": "texture",
         "source": {"lane_indexed_resource_heap": {"max_count": 256, "shift": 0, "stride": 24,
                    "window_base": {"cbv": {"user_sgpr": 8, "table_byte_offset": 32},
                                    "byte_offset": 8, "bit_offset": 8, "mask": 255}}}},
        {"group": 0, "binding": 3, "kind": "sampler",
         "source": {"lane_indexed_sampler_heap": {"max_count": 1, "shift": 4, "stride": 16,
                    "window_base": {"cbv": {"user_sgpr": 9, "root_constants": true}, "byte_offset": 0}}}}
      ]
    }
  }
})";
    StageReflectionIndex idx;
    std::string error;
    CHECK(parse_stage_reflection_json(doc, &idx, &error));
    CHECK(error.empty() && idx.size() == 2);

    const d3d::RootSignatureDesc rs = ui_root_signature();
    bridge::PipelineLayoutDesc layout;
    std::string report;
    JoinStats stats;
    CHECK(join_pipeline_layout(rs, false,
                               {{GpuShaderStageBits::Vertex, &idx.at(0xe20)},
                                {GpuShaderStageBits::Fragment, &idx.at(0xe21)}},
                               &layout, &report, &stats, JoinOptions{true}));
    CHECK(report.empty() && stats.lane_indexed_arrays == 2 && stats.lane_indexed_bindings == 0);

    // Non-power-of-two stride propagates to ShaderBinding::heap_stride for cbv_indexed_*_heap too!
    const bridge::LayoutBinding* cb_res = find(layout, 0, 0);
    CHECK(cb_res != nullptr && cb_res->join.heap == d3d::HeapAccess::Resource &&
          cb_res->join.heap_index_source == d3d::HeapIndexSource::CbRootDescriptor &&
          cb_res->join.heap_shift == 0 && cb_res->join.heap_stride == 24 &&
          cb_res->join.cb_byte_offset == 12 && cb_res->join.cb_bit_offset == 4 && cb_res->join.cb_mask == 4095);
    const bridge::LayoutBinding* cb_smp = find(layout, 0, 1);
    CHECK(cb_smp != nullptr && cb_smp->join.heap == d3d::HeapAccess::Sampler &&
          cb_smp->join.heap_index_source == d3d::HeapIndexSource::CbRootConstants &&
          cb_smp->join.heap_shift == 0 && cb_smp->join.heap_stride == 48 && cb_smp->join.cb_byte_offset == 4);
    // Shared lane-indexed binding merges visibility and counts once in lane_indexed_arrays.
    const bridge::LayoutBinding* lane_res = find(layout, 0, 2);
    CHECK(lane_res != nullptr &&
          lane_res->entry.visibility == (GpuShaderStageBits::Vertex | GpuShaderStageBits::Fragment) &&
          lane_res->entry.binding_array_size == 256 && lane_res->join.array_size == 256 &&
          lane_res->join.heap_stride == 24 && lane_res->join.heap_index_source == d3d::HeapIndexSource::CbTable &&
          lane_res->join.cb_table_slot == 1 && lane_res->join.cb_byte_offset == 8 &&
          lane_res->join.cb_bit_offset == 8 && lane_res->join.cb_mask == 255);
    const bridge::LayoutBinding* lane_smp = find(layout, 0, 3);
    CHECK(lane_smp != nullptr && lane_smp->entry.binding_array_size == 1 && lane_smp->join.array_size == 1 &&
          lane_smp->join.heap_shift == 4 && lane_smp->join.heap_stride == 0 &&
          lane_smp->join.heap_index_source == d3d::HeapIndexSource::CbRootConstants);

    // BindGroupPlan preserves cbv_indexed and lane_indexed heap_stride cleanly.
    d3d::RootSignature built;
    CHECK(d3d::RootSignature::build(rs, &built).ok());
    std::vector<d3d::ShaderBinding> sbs = {cb_res->join, cb_smp->join, lane_res->join, lane_smp->join};
    for (uint32_t i = 0; i < sbs.size(); ++i) {
        sbs[i].binding = i;
    }
    const d3d::BindGroupPlan plan =
        d3d::BindGroupPlan::build(built, sbs, d3d::WebGpuStage::Vertex | d3d::WebGpuStage::Fragment);
    CHECK(plan.unresolved_count() == 0 && plan.entries().size() == 4);
    CHECK(plan.entries()[0].heap_stride == 24 && plan.entries()[1].heap_stride == 48 &&
          plan.entries()[2].heap_stride == 24 && plan.entries()[3].heap_stride == 0);

    // Multi-stage disagreement on stride fails cleanly.
    StageReflection bad_ps = idx.at(0xe21);
    bad_ps.bindings[0].heap_stride = 48;
    report.clear();
    CHECK(!join_pipeline_layout(rs, false,
                                {{GpuShaderStageBits::Vertex, &idx.at(0xe20)}, {GpuShaderStageBits::Fragment, &bad_ps}},
                                &layout, &report, nullptr, JoinOptions{true}));
    CHECK(report.find("the stages disagree") != std::string::npos);

    // Ambiguous multi-key cbv and fractional integers are rejected at parse time.
    StageReflectionIndex bad;
    CHECK(!parse_stage_reflection_json(
        R"({"schema": "playgta6-stage-reflection", "version": 3, "stages": {"0x1": {"stage": "fragment",
            "bindings": [{"group": 0, "binding": 0, "kind": "texture",
                          "source": {"cbv_indexed_resource_heap": {"cbv": {"user_sgpr": 11, "root_descriptor": true,
                                                                           "root_constants": true},
                                                                   "byte_offset": 0, "shift": 5}}}]}}})",
        &bad, &error));
    CHECK(!parse_stage_reflection_json(
        R"({"schema": "playgta6-stage-reflection", "version": 2, "stages": {"0x1": {"stage": "fragment",
            "bindings": [{"group": 0, "binding": 0, "kind": "uniform",
                          "source": {"user_sgpr": 8, "root_descriptor": true, "table_byte_offset": 32.5}}]}}})",
        &bad, &error));

    // Programmatic validation of out-of-range shift / lane_max_count / cb_mask / alignment.
    StageReflection prog;
    ReflectedBinding pb;
    pb.group = 0;
    pb.binding = 0;
    pb.kind = ReflectedBindingKind::Texture;
    pb.source = ReflectedSource::LaneIndexedResourceHeap;
    pb.lane_max_count = kMaxLaneIndexedWindow + 1;
    prog.bindings = {pb};
    report.clear();
    CHECK(!join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &prog}}, &layout, &report, nullptr,
                                JoinOptions{true}));
    CHECK(report.find("max_count") != std::string::npos);

    pb.source = ReflectedSource::CbvIndexedResourceHeap;
    pb.user_sgpr = 11;
    pb.cbv = ReflectedCbv::RootDescriptor;
    pb.heap_shift = 32;
    prog.bindings = {pb};
    report.clear();
    CHECK(!join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &prog}}, &layout, &report));
    CHECK(report.find("heap shift 32") != std::string::npos);
}

// G83 (#1748 / #651): mesh_output boundaries, kMeshUserDataBase (s16) full layout joins,
// and RT group-3 conflict + multi-stage merging.
void test_g83_mesh_and_rt_group3_joins() {
    // mesh_output boundary acceptance and rejection.
    StageReflectionIndex idx;
    std::string error;
    CHECK(parse_stage_reflection_json(
        R"({"schema": "playgta6-stage-reflection", "version": 3, "stages": {
            "0x1": {"stage": "compute", "mesh_output": {"max_vertices": 1, "max_primitives": 1, "vertex_words": 4}},
            "0x2": {"stage": "compute", "mesh_output": {"max_vertices": 256, "max_primitives": 256, "vertex_words": 64}}}})",
        &idx, &error));
    CHECK(idx.at(1).has_mesh_output && idx.at(2).has_mesh_output && idx.at(2).mesh_output.vertex_words == 64);
    for (const char* bad_mesh : {
             R"({"max_vertices": 0, "max_primitives": 1, "vertex_words": 4})",
             R"({"max_vertices": 257, "max_primitives": 1, "vertex_words": 4})",
             R"({"max_vertices": 1, "max_primitives": 0, "vertex_words": 4})",
             R"({"max_vertices": 1, "max_primitives": 257, "vertex_words": 4})",
             R"({"max_vertices": 1, "max_primitives": 1, "vertex_words": 0})",
             R"({"max_vertices": 1, "max_primitives": 1, "vertex_words": 68})",
         }) {
        StageReflectionIndex bad;
        CHECK(!parse_stage_reflection_json(
            std::string(R"({"schema": "playgta6-stage-reflection", "version": 3, "stages": {"0x1": {"stage": "compute", "mesh_output": )") +
                bad_mesh + "}}}",
            &bad, &error));
    }

    // Mesh stage lowered to compute with kMeshUserDataBase = s16:
    // ui_root_signature parameters sit at s16 (table), s17..s18 (2 root constants), s19..s20 (root CBV).
    const char* mesh_doc = R"({
  "schema": "playgta6-stage-reflection", "version": 3,
  "stages": {
    "0x4d53435300000010": {
      "stage": "compute",
      "mesh_output": {"max_vertices": 64, "max_primitives": 126, "vertex_words": 16},
      "bindings": [
        {"group": 0, "binding": 0, "kind": "read_only_storage", "source": {"user_sgpr": 16, "table_byte_offset": 0}},
        {"group": 0, "binding": 1, "kind": "uniform", "source": {"user_sgpr": 17, "root_constants": true}},
        {"group": 0, "binding": 2, "kind": "uniform",
         "source": {"user_sgpr": 16, "root_descriptor": true, "table_byte_offset": 32}},
        {"group": 0, "binding": 3, "kind": "texture",
         "source": {"cbv_indexed_resource_heap": {"cbv": {"user_sgpr": 19, "root_descriptor": true},
                                                  "byte_offset": 0, "shift": 5}}},
        {"group": 0, "binding": 4, "kind": "texture",
         "source": {"lane_indexed_resource_heap": {"max_count": 16, "shift": 5,
                    "window_base": {"cbv": {"user_sgpr": 17, "root_constants": true}, "byte_offset": 4}}}}
      ]
    }
  }
})";
    StageReflectionIndex mesh_idx;
    CHECK(parse_stage_reflection_json(mesh_doc, &mesh_idx, &error));
    const StageReflection& ms = mesh_idx.at(0x4d53435300000010ull);
    const d3d::RootSignatureDesc rs = ui_root_signature();
    bridge::PipelineLayoutDesc layout;
    std::string report;
    JoinStats stats;
    JoinOptions mesh_opts;
    mesh_opts.binding_arrays = true;
    mesh_opts.user_data_base = kMeshUserDataBase;
    CHECK(join_pipeline_layout(rs, false, {{GpuShaderStageBits::Compute, &ms}}, &layout, &report, &stats, mesh_opts));
    CHECK(report.empty() && stats.root_descriptor_tables == 1 && stats.lane_indexed_arrays == 1);
    CHECK(find(layout, 0, 0) != nullptr && find(layout, 0, 0)->join.shader_register == 2);
    CHECK(find(layout, 0, 1) != nullptr && find(layout, 0, 1)->join.root_constants);
    CHECK(find(layout, 0, 2) != nullptr && find(layout, 0, 2)->join.register_class == d3d::DescriptorRangeType::Cbv &&
          find(layout, 0, 2)->join.shader_register == 0);
    CHECK(find(layout, 0, 3) != nullptr && find(layout, 0, 3)->join.heap_index_parameter == 2);
    CHECK(find(layout, 0, 4) != nullptr && find(layout, 0, 4)->join.heap_index_parameter == 1 &&
          find(layout, 0, 4)->entry.binding_array_size == 16);

    // Without kMeshUserDataBase, s16..s19 fail cleanly.
    report.clear();
    CHECK(!join_pipeline_layout(rs, false, {{GpuShaderStageBits::Compute, &ms}}, &layout, &report, nullptr,
                                JoinOptions{true, 0}));
    CHECK(!report.empty());

    // RT group-3: multi-stage merge succeeds when user bindings stay in groups 0..2,
    // and fails when any stage joins a regular binding into group 3 or has invalid rt_arena_mask bits.
    const char* rt_doc = R"({"schema": "playgta6-stage-reflection", "version": 3, "stages": {
      "0x90": {"stage": "vertex", "bindings": [
        {"group": 0, "binding": 0, "kind": "uniform", "source": {"user_sgpr": 9, "root_constants": true}},
        {"group": 3, "binding": 16, "kind": "storage", "source": {"rt_arena": true}},
        {"group": 3, "binding": 17, "kind": "storage", "source": {"rt_arena": true}},
        {"group": 3, "binding": 18, "kind": "storage", "source": {"rt_arena": true}},
        {"group": 3, "binding": 19, "kind": "storage", "source": {"rt_arena": true}}]},
      "0x91": {"stage": "fragment", "bindings": [
        {"group": 2, "binding": 0, "kind": "uniform", "source": {"user_sgpr": 11, "root_descriptor": true}},
        {"group": 3, "binding": 20, "kind": "read_only_storage", "source": {"rt_arena": true}},
        {"group": 3, "binding": 21, "kind": "storage", "source": {"rt_arena": true}}]},
      "0x92": {"stage": "fragment", "rt_arena": true, "bindings": [
        {"group": 3, "binding": 0, "kind": "uniform", "source": {"user_sgpr": 11, "root_descriptor": true}}]}}})";
    StageReflectionIndex rt_idx;
    CHECK(parse_stage_reflection_json(rt_doc, &rt_idx, &error));
    CHECK(rt_idx.at(0x90).rt_arena_mask == 0x0fu && rt_idx.at(0x91).rt_arena_mask == 0x30u);
    report.clear();
    CHECK(join_pipeline_layout(rs, false,
                               {{GpuShaderStageBits::Vertex, &rt_idx.at(0x90)},
                                {GpuShaderStageBits::Fragment, &rt_idx.at(0x91)}},
                               &layout, &report));
    CHECK(report.empty() && layout.rt_arena && layout.groups.size() == 3);

    // Conflict: stage with rt_arena: true also has a non-RT binding in group 3.
    report.clear();
    CHECK(!join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &rt_idx.at(0x92)}}, &layout, &report));
    CHECK(report.find("the runtime owns group 3") != std::string::npos);

    // Programmatic invalid rt_arena_mask bits outside 16..21.
    StageReflection bad_rt;
    bad_rt.rt_arena_mask = 0x40u;
    report.clear();
    CHECK(!join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &bad_rt}}, &layout, &report));
    CHECK(report.find("rt_arena_mask") != std::string::npos);
}

// #663 / #482: sub-element vertex fetch inference (32-bit dword sub-reads like NORMAL0 + 24 and
// TANGENT0 + 40, adjacent element spanning, and 2-byte aligned 16-bit element sub-reads).
void test_vertex_sub_element_inference() {
    // 32-bit dword sub-reads: POSITION0 float3 @0 (12B), NORMAL0 float4 @16 (16B: 16..32),
    // TANGENT0 float4 @32 (16B: 32..48).
    std::vector<pso::D3D12InputElementDesc> layout(3);
    layout[0].SemanticName = "POSITION";
    layout[0].Format = pso::DxgiFormat::R32G32B32_FLOAT;
    layout[0].AlignedByteOffset = 0;
    layout[1].SemanticName = "NORMAL";
    layout[1].Format = pso::DxgiFormat::R32G32B32A32_FLOAT;
    layout[1].AlignedByteOffset = 16;
    layout[2].SemanticName = "TANGENT";
    layout[2].Format = pso::DxgiFormat::R32G32B32A32_FLOAT;
    layout[2].AlignedByteOffset = 32;

    StageReflection vs;
    vs.stage = ReflectedStage::Vertex;
    vs.vertex_inputs = {
        {0, 0, 0, 3, ReflectedComponentType::F32, pso::WgpuVertexFormat::Undefined},
        // Reads inside NORMAL0 at +24 (2 floats inside NORMAL0).
        {1, 0, 24, 2, ReflectedComponentType::F32, pso::WgpuVertexFormat::Undefined},
        // Reads inside TANGENT0 at +40 (2 floats left at tail, even if stage asks for 4 components).
        {2, 0, 40, 4, ReflectedComponentType::F32, pso::WgpuVertexFormat::Undefined},
    };
    std::vector<pso::VertexInputFetch> fetches;
    std::string report;
    CHECK(join_vertex_inputs(vs, layout.data(), static_cast<uint32_t>(layout.size()), &fetches, &report));
    CHECK(report.empty() && fetches.size() == 3);
    if (fetches.size() == 3) {
        CHECK(fetches[0].offset == 0 && fetches[0].format == pso::WgpuVertexFormat::Float32x3);
        CHECK(fetches[1].offset == 24 && fetches[1].format == pso::WgpuVertexFormat::Float32x2);
        CHECK(fetches[2].offset == 40 && fetches[2].format == pso::WgpuVertexFormat::Float32x2);
    }

    // Verify translate_graphics_pso accepts the inferred sub-element fetches with zero validation diags.
    pso::D3D12GraphicsPsoDesc pso_desc{};
    pso_desc.vs = 1;
    pso_desc.ps = 2;
    pso_desc.pInputElementDescs = layout.data();
    pso_desc.NumInputElements = static_cast<uint32_t>(layout.size());
    pso_desc.NumRenderTargets = 1;
    pso_desc.RTVFormats[0] = pso::DxgiFormat::R8G8B8A8_UNORM;
    pso_desc.vs_input_fetches = fetches.data();
    pso_desc.vs_input_fetch_count = static_cast<uint32_t>(fetches.size());
    const pso::GraphicsPsoTranslation t = pso::translate_graphics_pso(pso_desc, pso::PipelineKeyOptions{});
    CHECK((t.diagnostics & (pso::PipelineDiag::VertexLayoutInvalid | pso::PipelineDiag::MissingVertexInput)) == 0);

    // Sub-element dword read spanning NORMAL0 + 24 (2 floats) into TANGENT0 + 32 (2 floats) -> Float32x4.
    StageReflection span_vs;
    span_vs.vertex_inputs = {{1, 0, 24, 4, ReflectedComponentType::F32, pso::WgpuVertexFormat::Undefined}};
    fetches.clear();
    report.clear();
    CHECK(join_vertex_inputs(span_vs, layout.data(), static_cast<uint32_t>(layout.size()), &fetches, &report));
    CHECK(report.empty() && fetches.size() == 1 && fetches[0].format == pso::WgpuVertexFormat::Float32x4);

    // 32-bit bitcast sub-read: Uint32x4 read as F32 at +8 -> Float32x2.
    std::vector<pso::D3D12InputElementDesc> u32_el(1);
    u32_el[0].SemanticName = "TEXCOORD";
    u32_el[0].Format = pso::DxgiFormat::R32G32B32A32_UINT;
    u32_el[0].AlignedByteOffset = 0;
    StageReflection bitcast_vs;
    bitcast_vs.vertex_inputs = {{0, 0, 8, 2, ReflectedComponentType::F32, pso::WgpuVertexFormat::Undefined}};
    fetches.clear();
    report.clear();
    CHECK(join_vertex_inputs(bitcast_vs, u32_el.data(), 1, &fetches, &report));
    CHECK(report.empty() && fetches.size() == 1 && fetches[0].format == pso::WgpuVertexFormat::Float32x2);

    // 16-bit sub-element reads: Float16x4, Unorm16x4, Snorm16x4, Uint16x4, Sint16x2.
    std::vector<pso::D3D12InputElementDesc> half_layout(5);
    half_layout[0].SemanticName = "NORMAL";
    half_layout[0].Format = pso::DxgiFormat::R16G16B16A16_FLOAT;
    half_layout[0].AlignedByteOffset = 0;   // 0..8
    half_layout[1].SemanticName = "TANGENT";
    half_layout[1].Format = pso::DxgiFormat::R16G16B16A16_UNORM;
    half_layout[1].AlignedByteOffset = 8;   // 8..16
    half_layout[2].SemanticName = "BINORMAL";
    half_layout[2].Format = pso::DxgiFormat::R16G16B16A16_SNORM;
    half_layout[2].AlignedByteOffset = 16;  // 16..24
    half_layout[3].SemanticName = "BLENDINDICES";
    half_layout[3].Format = pso::DxgiFormat::R16G16B16A16_UINT;
    half_layout[3].AlignedByteOffset = 24;  // 24..32
    half_layout[4].SemanticName = "TEXCOORD";
    half_layout[4].Format = pso::DxgiFormat::R16G16_SINT;
    half_layout[4].AlignedByteOffset = 32;  // 32..36

    StageReflection half_vs;
    half_vs.vertex_inputs = {
        {0, 0, 4, 2, ReflectedComponentType::F32, pso::WgpuVertexFormat::Undefined},   // Float16x2
        {1, 0, 12, 2, ReflectedComponentType::F32, pso::WgpuVertexFormat::Undefined},  // Unorm16x2
        {2, 0, 22, 1, ReflectedComponentType::F32, pso::WgpuVertexFormat::Undefined},  // Snorm16 (2-byte aligned)
        {3, 0, 28, 2, ReflectedComponentType::U32, pso::WgpuVertexFormat::Undefined},  // Uint16x2
        {4, 0, 34, 1, ReflectedComponentType::I32, pso::WgpuVertexFormat::Undefined},  // Sint16 (2-byte aligned)
    };
    fetches.clear();
    report.clear();
    CHECK(join_vertex_inputs(half_vs, half_layout.data(), static_cast<uint32_t>(half_layout.size()), &fetches, &report));
    CHECK(report.empty() && fetches.size() == 5);
    if (fetches.size() == 5) {
        CHECK(fetches[0].format == pso::WgpuVertexFormat::Float16x2);
        CHECK(fetches[1].format == pso::WgpuVertexFormat::Unorm16x2);
        CHECK(fetches[2].format == pso::WgpuVertexFormat::Snorm16);
        CHECK(fetches[3].format == pso::WgpuVertexFormat::Uint16x2);
        CHECK(fetches[4].format == pso::WgpuVertexFormat::Sint16);
    }

    // Rejections: unaligned dword read (+2 in Float32x4), 2-comp half read at 2-byte-only offset (+2 in Float16x4),
    // odd offset in 16-bit element (+1 in Float16x4), and class mismatch on 16-bit sub-read (Uint16x4 as f32).
    StageReflection bad_dw;
    bad_dw.vertex_inputs = {{0, 0, 18, 2, ReflectedComponentType::F32, pso::WgpuVertexFormat::Undefined}};
    report.clear();
    CHECK(!join_vertex_inputs(bad_dw, layout.data(), static_cast<uint32_t>(layout.size()), &fetches, &report));
    CHECK(report.find("reads inside element NORMAL0 and has no fetch_format") != std::string::npos);

    StageReflection bad_half_align;
    bad_half_align.vertex_inputs = {
        {0, 0, 2, 2, ReflectedComponentType::F32, pso::WgpuVertexFormat::Undefined},
        {1, 0, 1, 1, ReflectedComponentType::F32, pso::WgpuVertexFormat::Undefined},
        {2, 0, 28, 2, ReflectedComponentType::F32, pso::WgpuVertexFormat::Undefined},
    };
    report.clear();
    CHECK(!join_vertex_inputs(bad_half_align, half_layout.data(), static_cast<uint32_t>(half_layout.size()), &fetches,
                              &report));
    CHECK(report.find("@location(0) <- slot 0 + 2: reads inside element NORMAL0") != std::string::npos);
    CHECK(report.find("@location(1) <- slot 0 + 1: reads inside element NORMAL0") != std::string::npos);
    CHECK(report.find("@location(2) <- slot 0 + 28: the stage expects f32, the attribute gives u32") !=
          std::string::npos);
}

// #663 / #482: per-stage user_data_base override on ReflectedStageRef and automatic GFX10.3
// shifted-base fallback (s16 NGG/merged VS, s12 4-SGPR-shifted VS, s14) paired with PS at s8.
void test_per_stage_user_data_base_and_shifted_fallback() {
    // Root signature (4 dwords total):
    //   [0] table {SRV t2 space1, CBV b0 space1} (offset +0)
    //   [1] 2 root constants b500                (offset +1..+2)
    //   [2] table {SRV t5 space0}                (offset +3)
    // At base 8 (PS):     param 0 = s8,  param 1 = s9..s10,  param 2 = s11
    // At base 16 (NGG):   param 0 = s16, param 1 = s17..s18, param 2 = s19
    // At base 12 (shift): param 0 = s12, param 1 = s13..s14, param 2 = s15
    d3d::RootSignatureDesc rs;
    d3d::RootParameterDesc t0;
    t0.parameter_type = d3d::RootParameterType::DescriptorTable;
    d3d::DescriptorRangeDesc srv0;
    srv0.range_type = d3d::DescriptorRangeType::Srv;
    srv0.base_shader_register = 2;
    srv0.register_space = 1;
    d3d::DescriptorRangeDesc cbv0;
    cbv0.range_type = d3d::DescriptorRangeType::Cbv;
    cbv0.base_shader_register = 0;
    cbv0.register_space = 1;
    t0.ranges = {srv0, cbv0};
    d3d::RootParameterDesc rc;
    rc.parameter_type = d3d::RootParameterType::Constants32Bit;
    rc.shader_register = 500;
    rc.num_32bit_values = 2;
    d3d::RootParameterDesc t2;
    t2.parameter_type = d3d::RootParameterType::DescriptorTable;
    d3d::DescriptorRangeDesc srv2;
    srv2.range_type = d3d::DescriptorRangeType::Srv;
    srv2.base_shader_register = 5;
    srv2.register_space = 0;
    t2.ranges = {srv2};
    rs.parameters = {t0, rc, t2};

    // VS at NGG base s16: table at s16 + 0 (shared with PS), table at s19 + 1312 (live log pattern!).
    StageReflection vs_ngg;
    vs_ngg.stage = ReflectedStage::Vertex;
    ReflectedBinding v0;
    v0.group = 0;
    v0.binding = 0;
    v0.kind = ReflectedBindingKind::ReadOnlyStorage;
    v0.source = ReflectedSource::Table;
    v0.user_sgpr = 16;
    v0.table_byte_offset = 0;
    ReflectedBinding v2;
    v2.group = 0;
    v2.binding = 2;
    v2.kind = ReflectedBindingKind::Texture;
    v2.source = ReflectedSource::Table;
    v2.user_sgpr = 19;
    v2.table_byte_offset = 1312;  // slot 41, past declared ranges -> outside_rs on param 2
    vs_ngg.bindings = {v0, v2};

    // PS at default base s8: shares @group(0) @binding(0) via s8 + 0, plus root constants at s9.
    StageReflection ps;
    ps.stage = ReflectedStage::Fragment;
    ReflectedBinding p0 = v0;
    p0.user_sgpr = 8;
    ReflectedBinding p1;
    p1.group = 0;
    p1.binding = 1;
    p1.kind = ReflectedBindingKind::Uniform;
    p1.source = ReflectedSource::RootConstants;
    p1.user_sgpr = 9;
    ps.bindings = {p0, p1};

    // 1. Auto GFX10.3 shifted-base fallback: VS resolves at s16 while PS stays at s8.
    bridge::PipelineLayoutDesc layout;
    std::string report;
    JoinStats stats;
    CHECK(join_pipeline_layout(rs, false,
                               {{GpuShaderStageBits::Vertex, &vs_ngg}, {GpuShaderStageBits::Fragment, &ps}},
                               &layout, &report, &stats));
    CHECK(report.empty());
    CHECK(stats.shifted_user_data_base_stages == 1 && stats.table_slots_outside_rs == 1);
    const bridge::LayoutBinding* b0 = find(layout, 0, 0);
    CHECK(b0 != nullptr && b0->entry.visibility == (GpuShaderStageBits::Vertex | GpuShaderStageBits::Fragment) &&
          b0->join.register_class == d3d::DescriptorRangeType::Srv && b0->join.shader_register == 2 &&
          b0->join.register_space == 1);
    const bridge::LayoutBinding* b1 = find(layout, 0, 1);
    CHECK(b1 != nullptr && b1->join.root_constants && b1->join.shader_register == 500);
    const bridge::LayoutBinding* b2 = find(layout, 0, 2);
    CHECK(b2 != nullptr && b2->join.table_parameter == 2 && b2->join.table_slot == 41 &&
          b2->join.register_class == d3d::DescriptorRangeType::Srv);

    // 2. Explicit per-stage ReflectedStageRef::user_data_base: does not count as auto-shifted.
    report.clear();
    JoinStats explicit_stats;
    CHECK(join_pipeline_layout(rs, false,
                               {{GpuShaderStageBits::Vertex, &vs_ngg, kMeshUserDataBase},
                                {GpuShaderStageBits::Fragment, &ps, 0}},
                               &layout, &report, &explicit_stats));
    CHECK(report.empty() && explicit_stats.shifted_user_data_base_stages == 0);

    // 3. Auto 4-SGPR-shifted VS (base s12: s12 table, s13 constants, s15 table) paired with PS at s8.
    StageReflection vs_s12;
    vs_s12.stage = ReflectedStage::Vertex;
    ReflectedBinding s12_t = v0;
    s12_t.user_sgpr = 12;
    ReflectedBinding s12_c = p1;
    s12_c.user_sgpr = 13;
    ReflectedBinding s12_t2 = v2;
    s12_t2.user_sgpr = 15;
    s12_t2.table_byte_offset = 0;  // SRV t5 space 0
    vs_s12.bindings = {s12_t, s12_c, s12_t2};
    report.clear();
    JoinStats s12_stats;
    CHECK(join_pipeline_layout(rs, false,
                               {{GpuShaderStageBits::Vertex, &vs_s12}, {GpuShaderStageBits::Fragment, &ps}},
                               &layout, &report, &s12_stats));
    CHECK(report.empty() && s12_stats.shifted_user_data_base_stages == 1);
    const bridge::LayoutBinding* s12_b2 = find(layout, 0, 2);
    CHECK(s12_b2 != nullptr && s12_b2->join.shader_register == 5 && s12_b2->join.register_space == 0);

    // 4. Derived shifted base (s14: table at s14, constants at s15, table at s17).
    StageReflection vs_s14 = vs_s12;
    vs_s14.bindings[0].user_sgpr = 14;
    vs_s14.bindings[1].user_sgpr = 15;
    vs_s14.bindings[2].user_sgpr = 17;
    report.clear();
    JoinStats s14_stats;
    CHECK(join_pipeline_layout(rs, false,
                               {{GpuShaderStageBits::Vertex, &vs_s14}, {GpuShaderStageBits::Fragment, &ps}},
                               &layout, &report, &s14_stats));
    CHECK(report.empty() && s14_stats.shifted_user_data_base_stages == 1);

    // 5. Negative: inconsistent stage mixing s8 and s19 fails (no single base covers both).
    StageReflection bad_mix;
    bad_mix.stage = ReflectedStage::Vertex;
    bad_mix.bindings = {p0, v2};  // s8 + s19 in one stage
    report.clear();
    JoinStats bad_stats;
    CHECK(!join_pipeline_layout(rs, false, {{GpuShaderStageBits::Vertex, &bad_mix}}, &layout, &report, &bad_stats));
    CHECK(bad_stats.shifted_user_data_base_stages == 0);
    CHECK(report.find("no root parameter covers s19") != std::string::npos);

    // 6. Negative: wrong parameter kind under every shifted base (two adjacent tables at s16 & s17,
    // while rs only has tables at offsets +0 and +3, never adjacent).
    StageReflection bad_kind;
    bad_kind.stage = ReflectedStage::Vertex;
    ReflectedBinding bk0 = v0;
    bk0.binding = 0;
    bk0.user_sgpr = 16;
    ReflectedBinding bk1 = v0;
    bk1.binding = 1;
    bk1.user_sgpr = 17;
    bad_kind.bindings = {bk0, bk1};
    report.clear();
    bad_stats = JoinStats{};
    CHECK(!join_pipeline_layout(rs, false, {{GpuShaderStageBits::Vertex, &bad_kind}}, &layout, &report, &bad_stats));
    CHECK(bad_stats.shifted_user_data_base_stages == 0);

    // 7. Negative: out-of-bounds base (> 24, s30 on 4-dword RS) fails cleanly.
    StageReflection bad_hi;
    bad_hi.stage = ReflectedStage::Vertex;
    ReflectedBinding hi = v0;
    hi.user_sgpr = 30;
    bad_hi.bindings = {hi};
    report.clear();
    bad_stats = JoinStats{};
    CHECK(!join_pipeline_layout(rs, false, {{GpuShaderStageBits::Vertex, &bad_hi}}, &layout, &report, &bad_stats));
    CHECK(report.find("no root parameter covers s30") != std::string::npos);

    // 8. Negative: compute pipeline does NOT auto-shift.
    StageReflection cs_shifted = vs_ngg;
    cs_shifted.stage = ReflectedStage::Compute;
    report.clear();
    bad_stats = JoinStats{};
    CHECK(!join_pipeline_layout(rs, true, {{GpuShaderStageBits::Compute, &cs_shifted}}, &layout, &report, &bad_stats));
    CHECK(bad_stats.shifted_user_data_base_stages == 0);
}

// PR #1868 regression (#663, #651): disjoint fragment @group(0) @binding(64..127) joining,
// ShaderBinding::binding invariant, BindGroupPlan + BindGroupCache null/resource emission, and
// StateBridgeDevice layout interning with high binding numbers.
void test_fragment_high_bindings_64_to_95() {
    // Root signature with CBV_SRV_UAV table at s8 (param 0), Sampler table at s9 (param 1),
    // 2 root constants at s10..s11 (param 2), root CBV at s12..s13 (param 3).
    d3d::RootSignatureDesc rs_desc;
    d3d::RootParameterDesc res_tbl;
    res_tbl.parameter_type = d3d::RootParameterType::DescriptorTable;
    d3d::DescriptorRangeDesc srv_r;
    srv_r.range_type = d3d::DescriptorRangeType::Srv;
    srv_r.num_descriptors = 4;
    srv_r.base_shader_register = 0;
    d3d::DescriptorRangeDesc cbv_r;
    cbv_r.range_type = d3d::DescriptorRangeType::Cbv;
    cbv_r.num_descriptors = 2;
    cbv_r.base_shader_register = 0;
    res_tbl.ranges = {srv_r, cbv_r};

    d3d::RootParameterDesc smp_tbl;
    smp_tbl.parameter_type = d3d::RootParameterType::DescriptorTable;
    d3d::DescriptorRangeDesc smp_r;
    smp_r.range_type = d3d::DescriptorRangeType::Sampler;
    smp_r.num_descriptors = 4;
    smp_r.base_shader_register = 0;
    smp_tbl.ranges = {smp_r};

    d3d::RootParameterDesc rc;
    rc.parameter_type = d3d::RootParameterType::Constants32Bit;
    rc.shader_register = 10;
    rc.num_32bit_values = 2;

    d3d::RootParameterDesc rcbv;
    rcbv.parameter_type = d3d::RootParameterType::Cbv;
    rcbv.shader_register = 11;

    rs_desc.parameters = {res_tbl, smp_tbl, rc, rcbv};

    const char* doc = R"({
  "schema": "playgta6-stage-reflection", "version": 3,
  "stages": {
    "0x0000000000006401": {
      "stage": "vertex",
      "bindings": [
        {"group": 0, "binding": 0, "kind": "uniform", "min_binding_size_bytes": 64,
         "source": {"user_sgpr": 8, "table_byte_offset": 128}},
        {"group": 0, "binding": 1, "kind": "uniform", "min_binding_size_bytes": 8,
         "source": {"user_sgpr": 10, "root_constants": true}}
      ]
    },
    "0x0000000000006402": {
      "stage": "fragment",
      "bindings": [
        {"group": 0, "binding": 64, "kind": "texture", "view_dimension": "2d",
         "source": {"user_sgpr": 8, "table_byte_offset": 0}},
        {"group": 0, "binding": 65, "kind": "sampler",
         "source": {"user_sgpr": 9, "table_byte_offset": 16}},
        {"group": 0, "binding": 72, "kind": "uniform", "min_binding_size_bytes": 32,
         "source": {"user_sgpr": 12, "root_descriptor": true}},
        {"group": 0, "binding": 80, "kind": "texture", "view_dimension": "2d",
         "source": {"cbv_indexed_resource_heap": {"cbv": {"user_sgpr": 12, "root_descriptor": true},
                                                  "byte_offset": 0, "shift": 5}}},
        {"group": 0, "binding": 88, "kind": "texture", "view_dimension": "2d",
         "source": {"lane_indexed_resource_heap": {"max_count": 16, "shift": 5}}},
        {"group": 0, "binding": 95, "kind": "sampler",
         "source": {"inline_sampler": {"sdwords": [146, 15728640, 139460608, 0]}}},
        {"group": 0, "binding": 127, "kind": "texture", "view_dimension": "2d",
         "source": {"unresolved": "cb_dynamic_offset"}}
      ]
    }
  }
})";
    StageReflectionIndex idx;
    std::string error;
    CHECK(parse_stage_reflection_json(doc, &idx, &error));
    CHECK(error.empty() && idx.size() == 2);

    bridge::PipelineLayoutDesc layout;
    std::string report;
    JoinStats stats;
    CHECK(join_pipeline_layout(rs_desc, false,
                               {{GpuShaderStageBits::Vertex, &idx.at(0x6401)},
                                {GpuShaderStageBits::Fragment, &idx.at(0x6402)}},
                               &layout, &report, &stats));
    CHECK(report.empty() && layout.groups.size() == 1 && layout.groups[0].bindings.size() == 9);
    CHECK(stats.unresolved_bindings == 1 && stats.lane_indexed_bindings == 1 && stats.lane_indexed_arrays == 0);

    // Every LayoutBinding preserves exact binding number in BOTH entry.binding and join.binding.
    for (const uint32_t bn : {0u, 1u, 64u, 65u, 72u, 80u, 88u, 95u, 127u}) {
        const bridge::LayoutBinding* lb = find(layout, 0, bn);
        CHECK(lb != nullptr);
        if (lb != nullptr) {
            CHECK(lb->entry.binding == bn && lb->join.binding == bn);
        }
    }
    const bridge::LayoutBinding* ps_smp = find(layout, 0, 65);
    CHECK(ps_smp != nullptr && ps_smp->join.register_class == d3d::DescriptorRangeType::Sampler &&
          ps_smp->join.shader_register == 1 && ps_smp->join.table_parameter == d3d::kNoIndex);
    const bridge::LayoutBinding* ps_unres = find(layout, 0, 127);
    CHECK(ps_unres != nullptr && ps_unres->join.shader_register == d3d::kNoIndex &&
          ps_unres->join.register_space == d3d::kNoIndex && ps_unres->join.table_parameter == d3d::kNoIndex);

    // Build a real BindGroupPlan directly from lb.join (without manually overwriting sb.binding!)
    // and verify BindGroupCache produces resource_id == 0 (Null) for @binding(127) and preserves
    // all binding numbers 64..127.
    d3d::RootSignature rs;
    CHECK(d3d::RootSignature::build(rs_desc, &rs).ok());
    std::vector<d3d::ShaderBinding> sbs;
    for (const bridge::LayoutBinding& lb : layout.groups[0].bindings) {
        sbs.push_back(lb.join);
    }
    const d3d::BindGroupPlan plan =
        d3d::BindGroupPlan::build(rs, sbs, d3d::WebGpuStage::Vertex | d3d::WebGpuStage::Fragment);
    CHECK(plan.entries().size() == 9 && plan.unresolved_count() == 1);
    bool saw_64 = false, saw_65 = false, saw_95 = false, saw_127_unres = false;
    for (const d3d::PlanEntry& e : plan.entries()) {
        if (e.binding == 64) {
            saw_64 = e.source == d3d::PlanSource::DescriptorTable && e.root_parameter_index == 0 && e.table_offset == 0;
        } else if (e.binding == 65) {
            saw_65 = e.source == d3d::PlanSource::DescriptorTable && e.root_parameter_index == 1 && e.table_offset == 1;
        } else if (e.binding == 95) {
            saw_95 = e.source == d3d::PlanSource::InlineSampler;
        } else if (e.binding == 127) {
            saw_127_unres = e.source == d3d::PlanSource::Unresolved;
        }
    }
    CHECK(saw_64 && saw_65 && saw_95 && saw_127_unres);

    // Also verify StateBridgeDevice accepts the joined @binding(64..127) layout & creates a usable graphics PSO.
    d3d::DescriptorHeapRegistry heaps;
    bridge::StateBridgeDevice dev(heaps);
    dev.register_shader_module(0x6401, 101);
    dev.register_shader_module(0x6402, 102);
    dev.register_pipeline_layout(0x6401, 0x6402, layout);
    pso::D3D12GraphicsPsoDesc pdesc{};
    pdesc.vs = 0x6401;
    pdesc.ps = 0x6402;
    pdesc.NumRenderTargets = 1;
    pdesc.RTVFormats[0] = pso::DxgiFormat::R8G8B8A8_UNORM;
    auto gpso = dev.create_graphics_pipeline_state(pdesc, &rs);
    CHECK(gpso != nullptr && !pso::is_fatal(gpso->pso().diagnostics()));
}

// Sampler descriptor tables use 16-byte S# slots (matching descriptor_handle_increment_size(Sampler) == 16),
// while CBV_SRV_UAV tables remain 32-byte slots.
void test_sampler_table_16_byte_slots() {
    d3d::RootSignatureDesc rs;
    d3d::RootParameterDesc smp_tbl;
    smp_tbl.parameter_type = d3d::RootParameterType::DescriptorTable;
    d3d::DescriptorRangeDesc smp_r;
    smp_r.range_type = d3d::DescriptorRangeType::Sampler;
    smp_r.num_descriptors = 6;
    smp_r.base_shader_register = 0;
    smp_r.register_space = 2;
    smp_tbl.ranges = {smp_r};

    d3d::RootParameterDesc res_tbl;
    res_tbl.parameter_type = d3d::RootParameterType::DescriptorTable;
    d3d::DescriptorRangeDesc srv_r;
    srv_r.range_type = d3d::DescriptorRangeType::Srv;
    srv_r.num_descriptors = 4;
    srv_r.base_shader_register = 0;
    res_tbl.ranges = {srv_r};
    rs.parameters = {smp_tbl, res_tbl};

    StageReflection ps;
    ps.stage = ReflectedStage::Fragment;
    auto make_smp = [](uint32_t binding, uint32_t byte_off) {
        ReflectedBinding b;
        b.group = 0;
        b.binding = binding;
        b.kind = ReflectedBindingKind::Sampler;
        b.source = ReflectedSource::Table;
        b.user_sgpr = 8;
        b.table_byte_offset = byte_off;
        return b;
    };
    ps.bindings = {
        make_smp(64, 0),    // slot 0 -> s0
        make_smp(65, 16),   // slot 1 -> s1 (odd 16-byte slot!)
        make_smp(66, 32),   // slot 2 -> s2 (NOT slot 1!)
        make_smp(67, 48),   // slot 3 -> s3
        make_smp(68, 80),   // slot 5 -> s5
        make_smp(69, 144),  // slot 9 -> past declared 6 slots -> outside_rs slot 9
    };

    bridge::PipelineLayoutDesc layout;
    std::string report;
    JoinStats stats;
    CHECK(join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &ps}}, &layout, &report, &stats));
    CHECK(report.empty() && stats.table_slots_outside_rs == 1);
    const std::pair<uint32_t, uint32_t> expected_regs[] = {{64, 0}, {65, 1}, {66, 2}, {67, 3}, {68, 5}};
    for (const auto& [bn, reg] : expected_regs) {
        const bridge::LayoutBinding* lb = find(layout, 0, bn);
        CHECK(lb != nullptr && lb->join.register_class == d3d::DescriptorRangeType::Sampler &&
              lb->join.shader_register == reg && lb->join.register_space == 2 &&
              lb->join.table_parameter == d3d::kNoIndex);
    }
    const bridge::LayoutBinding* oob = find(layout, 0, 69);
    CHECK(oob != nullptr && oob->join.register_class == d3d::DescriptorRangeType::Sampler &&
          oob->join.table_parameter == 0 && oob->join.table_slot == 9);

    // Non-16-byte sampler offset (8) and non-32-byte resource offset (16) still fail cleanly.
    StageReflection bad;
    bad.stage = ReflectedStage::Fragment;
    ReflectedBinding bad_smp = make_smp(64, 8);
    ReflectedBinding bad_res;
    bad_res.group = 0;
    bad_res.binding = 65;
    bad_res.kind = ReflectedBindingKind::Texture;
    bad_res.source = ReflectedSource::Table;
    bad_res.user_sgpr = 9;
    bad_res.table_byte_offset = 16;
    bad.bindings = {bad_smp, bad_res};
    report.clear();
    CHECK(!join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &bad}}, &layout, &report));
    CHECK(report.find("byte offset 8 is not a descriptor boundary") != std::string::npos);
    CHECK(report.find("byte offset 16 is not a descriptor boundary") != std::string::npos);
}

// Wide graphics root signature where an NGG/merged VS at s16 has all its user SGPRs still covered
// by the default s8 range (so had_uncovered_user_sgpr == false), failing on parameter type at s8.
void test_wide_rs_covered_shifted_user_data_base() {
    // 14 dwords total:
    //   [0] table {SRV t0}        (+0)   -> s8 at base 8,  s16 at base 16
    //   [1] 4 root constants b0   (+1..4)-> s9..s12,       s17..s20
    //   [2] root CBV b1           (+5..6)-> s13..s14,      s21..s22
    //   [3] 7 root constants b2   (+7..13)-> s15..s21,     s23..s29
    // At base 8, s16 and s17 both fall INSIDE param 3 (root constants)!
    d3d::RootSignatureDesc rs;
    d3d::RootParameterDesc t0;
    t0.parameter_type = d3d::RootParameterType::DescriptorTable;
    d3d::DescriptorRangeDesc srv0;
    srv0.range_type = d3d::DescriptorRangeType::Srv;
    srv0.base_shader_register = 0;
    t0.ranges = {srv0};

    d3d::RootParameterDesc rc1;
    rc1.parameter_type = d3d::RootParameterType::Constants32Bit;
    rc1.shader_register = 0;
    rc1.num_32bit_values = 4;

    d3d::RootParameterDesc cbv2;
    cbv2.parameter_type = d3d::RootParameterType::Cbv;
    cbv2.shader_register = 1;

    d3d::RootParameterDesc rc3;
    rc3.parameter_type = d3d::RootParameterType::Constants32Bit;
    rc3.shader_register = 2;
    rc3.num_32bit_values = 7;
    rs.parameters = {t0, rc1, cbv2, rc3};

    // NGG VS reads table at s16 + 0 and root constants at s17 (both covered by s8..s21!).
    StageReflection vs;
    vs.stage = ReflectedStage::Vertex;
    ReflectedBinding vt;
    vt.group = 0;
    vt.binding = 0;
    vt.kind = ReflectedBindingKind::ReadOnlyStorage;
    vt.source = ReflectedSource::Table;
    vt.user_sgpr = 16;
    vt.table_byte_offset = 0;
    ReflectedBinding vc;
    vc.group = 0;
    vc.binding = 1;
    vc.kind = ReflectedBindingKind::Uniform;
    vc.source = ReflectedSource::RootConstants;
    vc.user_sgpr = 17;
    vs.bindings = {vt, vc};

    // PS at s8 uses disjoint @binding(64).
    StageReflection ps;
    ps.stage = ReflectedStage::Fragment;
    ReflectedBinding pt = vt;
    pt.binding = 64;
    pt.user_sgpr = 8;
    ps.bindings = {pt};

    bridge::PipelineLayoutDesc layout;
    std::string report;
    JoinStats stats;
    CHECK(join_pipeline_layout(rs, false,
                               {{GpuShaderStageBits::Vertex, &vs}, {GpuShaderStageBits::Fragment, &ps}},
                               &layout, &report, &stats));
    CHECK(report.empty() && stats.shifted_user_data_base_stages == 1);
    const bridge::LayoutBinding* b0 = find(layout, 0, 0);
    const bridge::LayoutBinding* b1 = find(layout, 0, 1);
    const bridge::LayoutBinding* b64 = find(layout, 0, 64);
    CHECK(b0 != nullptr && b0->join.register_class == d3d::DescriptorRangeType::Srv && b0->join.shader_register == 0);
    CHECK(b1 != nullptr && b1->join.root_constants && b1->join.shader_register == 0);
    CHECK(b64 != nullptr && b64->join.register_class == d3d::DescriptorRangeType::Srv && b64->join.shader_register == 0);

    // Fragment stage with covered s16 table mismatch must NOT auto-shift.
    StageReflection bad_ps = vs;
    bad_ps.stage = ReflectedStage::Fragment;
    report.clear();
    JoinStats ps_stats;
    CHECK(!join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &bad_ps}}, &layout, &report, &ps_stats));
    CHECK(ps_stats.shifted_user_data_base_stages == 0);
}

// CB-indexed and lane-indexed window_base CBV completeness: contained root-constants user_sgpr
// folds dword offset into cb_byte_offset, and root_descriptor CBV over a descriptor table coerces
// to CbTable (slot 0 or v2/v3 table_byte_offset / 32).
void test_cbv_indexed_root_constants_and_table_coercion() {
    const char* doc = R"({
  "schema": "playgta6-stage-reflection", "version": 3,
  "stages": {
    "0x000000000000cb01": {
      "stage": "fragment",
      "bindings": [
        {"group": 0, "binding": 64, "kind": "texture",
         "source": {"cbv_indexed_resource_heap": {
             "cbv": {"user_sgpr": 10, "root_constants": true},
             "byte_offset": 0, "shift": 5}}},
        {"group": 0, "binding": 65, "kind": "texture",
         "source": {"cbv_indexed_resource_heap": {
             "cbv": {"user_sgpr": 8, "root_descriptor": true},
             "byte_offset": 12, "shift": 5}}},
        {"group": 0, "binding": 66, "kind": "sampler",
         "source": {"lane_indexed_sampler_heap": {
             "max_count": 8, "shift": 4,
             "window_base": {"cbv": {"user_sgpr": 8, "root_descriptor": true, "table_byte_offset": 32},
                             "byte_offset": 4}}}}
      ]
    }
  }
})";
    StageReflectionIndex idx;
    std::string error;
    CHECK(parse_stage_reflection_json(doc, &idx, &error));
    CHECK(error.empty() && idx.size() == 1);

    // ui_root_signature: [0] table at s8 (slot 0 SRV, slot 1 CBV), [1] 2 constants at s9..s10, [2] root CBV at s11.
    const d3d::RootSignatureDesc rs = ui_root_signature();
    bridge::PipelineLayoutDesc layout;
    std::string report;
    JoinStats stats;
    CHECK(join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &idx.at(0xcb01)}}, &layout, &report, &stats,
                               JoinOptions{true}));
    CHECK(report.empty() && stats.root_descriptor_tables == 2 && stats.lane_indexed_arrays == 1);

    const bridge::LayoutBinding* b64 = find(layout, 0, 64);
    CHECK(b64 != nullptr && b64->join.heap_index_source == d3d::HeapIndexSource::CbRootConstants &&
          b64->join.heap_index_parameter == 1 && b64->join.cb_byte_offset == 4);
    const bridge::LayoutBinding* b65 = find(layout, 0, 65);
    CHECK(b65 != nullptr && b65->join.heap_index_source == d3d::HeapIndexSource::CbTable &&
          b65->join.heap_index_parameter == 0 && b65->join.cb_table_slot == 0 && b65->join.cb_byte_offset == 12);
    const bridge::LayoutBinding* b66 = find(layout, 0, 66);
    CHECK(b66 != nullptr && b66->join.heap_index_source == d3d::HeapIndexSource::CbTable &&
          b66->join.heap_index_parameter == 0 && b66->join.cb_table_slot == 1 && b66->join.cb_byte_offset == 4 &&
          b66->entry.binding_array_size == 8);

    // Out-of-bounds contained root constant (s10 + byte_offset 4 -> effective byte 8 >= 2 * 4) fails cleanly.
    StageReflection oob = idx.at(0xcb01);
    oob.bindings = {oob.bindings[0]};
    oob.bindings[0].cb_byte_offset = 4;
    report.clear();
    CHECK(!join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &oob}}, &layout, &report));
    CHECK(report.find("covering byte 8") != std::string::npos);
}

// Vertex join + PSO completeness regressions: unorm8x4-bgra parse, exact-offset priority,
// 16-bit widening & single-tail sub-read, integer signedness reinterpret, APPEND_ALIGNED past
// unsupported formats, constant_default spare-slot collision avoidance, tight_strides from fetches,
// ReadOnlyDepth/Stencil variant propagation, AlphaToCoverage with null PS, 2x->4x SampleMask,
// Xbox formats 116/117, +/-Inf depth bias, and parser storage_format/group validation.
void test_vertex_join_and_pso_completeness() {
    // 1. unorm8x4-bgra parses in fetch_format; storage_format > 255 and group >= 16 rejected.
    StageReflectionIndex idx;
    std::string error;
    CHECK(parse_stage_reflection_json(
        R"({"schema": "playgta6-stage-reflection", "version": 3, "stages": {"0x1": {"stage": "vertex",
            "vertex_inputs": [{"location": 0, "vb_slot": 0, "byte_offset": 0, "components": 4,
                               "component_type": "f32", "fetch_format": "unorm8x4-bgra"}]}}})",
        &idx, &error));
    CHECK(idx.at(1).vertex_inputs[0].fetch_format == pso::WgpuVertexFormat::Unorm8x4Bgra);
    StageReflectionIndex bad;
    CHECK(!parse_stage_reflection_json(
        R"({"schema": "playgta6-stage-reflection", "version": 3, "stages": {"0x1": {"stage": "compute",
            "bindings": [{"group": 0, "binding": 0, "kind": "storage_texture", "storage_format": 256,
                          "source": {"user_sgpr": 4, "table_byte_offset": 0}}]}}})",
        &bad, &error));
    CHECK(!parse_stage_reflection_json(
        R"({"schema": "playgta6-stage-reflection", "version": 3, "stages": {"0x1": {"stage": "compute",
            "bindings": [{"group": 16, "binding": 0, "kind": "uniform",
                          "source": {"user_sgpr": 4, "table_byte_offset": 0}}]}}})",
        &bad, &error));

    // 2. APPEND_ALIGNED_ELEMENT stays accurate after unsupported R11G11B10_FLOAT (4B),
    //    exact-offset match wins over earlier overlapping element, 16-bit widening (Float16x2+Float16x2 -> Float16x4),
    //    16-bit single-tail sub-read (Float16x2 @+2 with default components=4 -> Float16),
    //    and uint16x2 <-> sint16x2 signedness reinterpretation.
    std::vector<pso::D3D12InputElementDesc> el(6);
    el[0].SemanticName = "TEXCOORD";
    el[0].Format = pso::DxgiFormat::R11G11B10_FLOAT;  // 4B unsupported vertex format @0..4
    el[0].AlignedByteOffset = 0;
    el[1].SemanticName = "POSITION";
    el[1].Format = pso::DxgiFormat::R32G32_FLOAT;     // appended -> MUST be @4..12, not @0!
    el[2].SemanticName = "NORMAL";
    el[2].Format = pso::DxgiFormat::R16G16_FLOAT;     // appended -> @12..16
    el[3].SemanticName = "TANGENT";
    el[3].Format = pso::DxgiFormat::R16G16_FLOAT;     // appended -> @16..20
    el[4].SemanticName = "BLENDINDICES";
    el[4].Format = pso::DxgiFormat::R16G16_UINT;      // appended -> @20..24
    el[5].SemanticName = "COLOR";
    el[5].Format = pso::DxgiFormat::R16G16_FLOAT;     // appended -> @24..28

    StageReflection vs;
    vs.stage = ReflectedStage::Vertex;
    vs.vertex_inputs = {
        {0, 0, 4, 2, ReflectedComponentType::F32, pso::WgpuVertexFormat::Undefined},   // POSITION @4
        {1, 0, 12, 4, ReflectedComponentType::F32, pso::WgpuVertexFormat::Undefined},  // widens NORMAL+TANGENT -> Float16x4
        {2, 0, 20, 2, ReflectedComponentType::I32, pso::WgpuVertexFormat::Undefined},  // Uint16x2 -> Sint16x2
        {3, 0, 26, 4, ReflectedComponentType::F32, pso::WgpuVertexFormat::Undefined},  // tail half of COLOR @26 -> Float16
    };
    std::vector<pso::VertexInputFetch> fetches;
    std::string report;
    CHECK(join_vertex_inputs(vs, el.data(), static_cast<uint32_t>(el.size()), &fetches, &report));
    CHECK(report.empty() && fetches.size() == 4);
    if (fetches.size() == 4) {
        CHECK(fetches[0].offset == 4 && fetches[0].format == pso::WgpuVertexFormat::Float32x2);
        CHECK(fetches[1].offset == 12 && fetches[1].format == pso::WgpuVertexFormat::Float16x4);
        CHECK(fetches[2].offset == 20 && fetches[2].format == pso::WgpuVertexFormat::Sint16x2);
        CHECK(fetches[3].offset == 26 && fetches[3].format == pso::WgpuVertexFormat::Float16);
    }

    // 3. constant_default avoids declared slot 0 when reflected VS only reads slot 1 + default,
    //    and tight_strides incorporates vs_input_fetches extent without spurious MissingVertexInput.
    pso::D3D12InputElementDesc two_slots[2]{};
    two_slots[0].SemanticName = "POSITION";
    two_slots[0].Format = pso::DxgiFormat::R32G32B32_FLOAT;
    two_slots[0].InputSlot = 0;
    two_slots[0].AlignedByteOffset = 0;
    two_slots[1].SemanticName = "TEXCOORD";
    two_slots[1].Format = pso::DxgiFormat::R32G32_FLOAT;
    two_slots[1].InputSlot = 1;
    two_slots[1].AlignedByteOffset = 0;

    pso::VertexInputFetch f_slot1[2]{};
    f_slot1[0] = {1, 0, 0, pso::WgpuVertexFormat::Float32x4, false};  // wider (16B) than slot 1 element (8B)
    f_slot1[1] = {0, 0, 1, pso::WgpuVertexFormat::Float32x4, true};   // constant default
    const pso::VertexInputSignatureElement dummy_sig = {"POSITION", 0, 0};

    pso::D3D12GraphicsPsoDesc pso_desc{};
    pso_desc.vs = 1;
    pso_desc.ps = 2;
    pso_desc.pInputElementDescs = two_slots;
    pso_desc.NumInputElements = 2;
    pso_desc.vs_input_signature = &dummy_sig;
    pso_desc.vs_input_signature_count = 1;
    pso_desc.vs_input_fetches = f_slot1;
    pso_desc.vs_input_fetch_count = 2;
    pso_desc.NumRenderTargets = 1;
    pso_desc.RTVFormats[0] = pso::DxgiFormat::R8G8B8A8_UNORM;
    const pso::GraphicsPsoTranslation tr = pso::translate_graphics_pso(pso_desc, pso::PipelineKeyOptions{});
    CHECK(tr.default_vertex_slot_mask == (1u << 2));  // avoids BOTH declared slot 0 and fetched slot 1!
    CHECK(tr.tight_strides[1] == 16u);                // reflects Float32x4 extent, not just 8B element!
    CHECK((tr.diagnostics & (pso::PipelineDiag::VertexLayoutInvalid | pso::PipelineDiag::MissingVertexInput)) == 0);

    // 4. ReadOnlyDepth & ReadOnlyStencil propagate into GraphicsPipelineKey and preserve pending clears.
    pso::D3D12GraphicsPsoDesc ds_pso{};
    ds_pso.vs = 1;
    ds_pso.ps = 2;
    ds_pso.NumRenderTargets = 1;
    ds_pso.RTVFormats[0] = pso::DxgiFormat::R8G8B8A8_UNORM;
    ds_pso.DSVFormat = pso::DxgiFormat::D24_UNORM_S8_UINT;
    ds_pso.DepthStencilState.DepthEnable = true;
    ds_pso.DepthStencilState.DepthWriteMask = pso::D3D12DepthWriteMask::All;
    ds_pso.DepthStencilState.StencilEnable = true;
    ds_pso.DepthStencilState.FrontFace.StencilPassOp = pso::D3D12StencilOp::Replace;
    pso::GraphicsPso gpu_pso(ds_pso);
    pso::PipelineCache cache;
    pso::CommandListStateTracker tracker;
    const pso::RenderTargetViewInfo rt_info{1, pso::DxgiFormat::R8G8B8A8_UNORM, 128, 128, 1};
    const pso::DepthStencilViewInfo dsv_rw{2, pso::DxgiFormat::D24_UNORM_S8_UINT, 128, 128, 1, 0};
    const pso::DepthStencilViewInfo dsv_ro{
        2, pso::DxgiFormat::D24_UNORM_S8_UINT, 128, 128, 1,
        pso::D3D12DsvFlags::ReadOnlyDepth | pso::D3D12DsvFlags::ReadOnlyStencil};
    tracker.ClearDepthStencilView(dsv_rw, pso::D3D12ClearFlags::Depth | pso::D3D12ClearFlags::Stencil, 0.5f, 3);
    tracker.set_render_targets(std::span<const pso::RenderTargetViewInfo>(&rt_info, 1), &dsv_ro);
    tracker.SetPipelineState(&gpu_pso);
    const pso::D3D12Viewport vp{0.0f, 0.0f, 128.0f, 128.0f, 0.0f, 1.0f};
    const pso::D3D12Rect sc{0, 0, 128, 128};
    tracker.RSSetViewports(1, &vp);
    tracker.RSSetScissorRects(1, &sc);
    CHECK(tracker.prepare_draw(&cache));
    const pso::GraphicsPipelineKey* ro_key = nullptr;
    for (const pso::StateEvent& ev : tracker.events()) {
        if (ev.type == pso::StateEventType::SetPipeline) {
            ro_key = ev.pipeline_key;
        }
    }
    CHECK(ro_key != nullptr && ro_key->depth_stencil.depth_write_enabled == 0 &&
          ro_key->depth_stencil.stencil_write_mask == 0 &&
          ro_key->depth_stencil.stencil_front.pass_op == pso::WgpuStencilOperation::Keep);
    // Pending clear was NOT silently discarded by the read-only pass: switching to writable DSV folds it!
    tracker.set_render_targets(std::span<const pso::RenderTargetViewInfo>(&rt_info, 1), &dsv_rw);
    CHECK(tracker.prepare_draw(&cache));
    CHECK(tracker.passes().size() == 2 &&
          tracker.passes()[1].depth_stencil.depth_load_op == pso::WgpuLoadOp::Clear &&
          tracker.passes()[1].depth_stencil.stencil_load_op == pso::WgpuLoadOp::Clear);

    // 5. AlphaToCoverage dropped with null PS, 2x->4x MSAA full mask expansion, Xbox 116/117 HDR formats, +/-Inf bias.
    pso::D3D12GraphicsPsoDesc edge{};
    edge.vs = 1;
    edge.ps = 0;
    edge.NumRenderTargets = 1;
    edge.RTVFormats[0] = pso::DxgiFormat::R10G10B10_7E3_A2_FLOAT;
    edge.SampleDesc.Count = 2;
    edge.SampleMask = 0x3u;
    edge.BlendState.AlphaToCoverageEnable = true;
    edge.RasterizerState.SlopeScaledDepthBias = std::numeric_limits<float>::infinity();
    edge.RasterizerState.DepthBiasClamp = -std::numeric_limits<float>::infinity();
    const pso::GraphicsKeyResult er = pso::make_graphics_pipeline_key(edge);
    CHECK(!pso::is_fatal(er.diagnostics));
    CHECK(er.key.color_targets[0].format == pso::WgpuTextureFormat::Rgba16Float);
    CHECK(er.key.multisample.count == 4u && er.key.multisample.mask == 0xFu);
    CHECK(er.key.multisample.alpha_to_coverage_enabled == 0 &&
          (er.diagnostics & pso::PipelineDiag::AlphaToCoverageDropped) != 0);
    CHECK(er.key.depth_stencil.depth_bias_slope_scale_bits == 0u &&
          er.key.depth_stencil.depth_bias_clamp_bits == 0u);
    CHECK(pso::map_texture_format(pso::DxgiFormat::R10G10B10_6E4_A2_FLOAT, pso::FormatUsage::RenderTarget, 0).format ==
          pso::WgpuTextureFormat::Rgba16Float);
}

// SM 6.6 directly-indexed descriptor heap implicit SGPRs:
// When CbvSrvUavHeapDirectlyIndexed (0x400) and/or SamplerHeapDirectlyIndexed (0x800) are set,
// trailing 1-SGPR heap pointers sit immediately after the explicit root parameters:
//   s_res = base + total_root_dwords(rs)
//   s_smp = base + total_root_dwords(rs) + (has_res ? 1 : 0)
void test_sm66_implicit_heap_sgprs() {
    class FakeCbMemory final : public d3d::DescriptorSlotSource {
    public:
        std::map<uint64_t, uint32_t> dwords;
        bool reads(const d3d::DescriptorHeap&) const override { return false; }
        bool snapshot(const d3d::DescriptorHeap&, uint32_t, uint32_t, std::vector<uint8_t>* out) const override {
            out->clear();
            return false;
        }
        d3d::ViewDescriptor view(const d3d::DescriptorHeap&, uint32_t, d3d::DescriptorRangeType) const override {
            return {};
        }
        bool sampler(const d3d::DescriptorHeap&, uint32_t, d3d::SamplerDesc*) const override { return false; }
        bool read_gpu_memory(uint64_t va, void* out, size_t bytes) const override {
            auto it = dwords.find(va);
            if (bytes != 4 || it == dwords.end()) {
                return false;
            }
            std::memcpy(out, &it->second, 4);
            return true;
        }
    };
    auto fake_va_resolver = [](void*, uint64_t va, d3d::GpuVaResolution* out) -> bool {
        if (va == 0x9000) {
            *out = {500, 0, 256};
            return true;
        }
        return false;
    };

    // 1. RS #9-like compute RS: flags=0x400, 2 root dwords at s4..s5 -> implicit s_res = s6.
    {
        d3d::RootSignatureDesc rs9;
        rs9.flags = d3d::RootSignatureFlags::CbvSrvUavHeapDirectlyIndexed;
        d3d::RootParameterDesc rc;
        rc.parameter_type = d3d::RootParameterType::Constants32Bit;
        rc.shader_register = 0;
        rc.num_32bit_values = 2;  // s4..s5
        rs9.parameters = {rc};

        const char* doc = R"({
          "schema": "playgta6-stage-reflection", "version": 3,
          "stages": {
            "0x0000000000009009": {
              "stage": "compute",
              "bindings": [
                {"group": 0, "binding": 0, "kind": "uniform",
                 "source": {"user_sgpr": 4, "root_constants": true}},
                {"group": 0, "binding": 1, "kind": "read_only_storage",
                 "source": {"user_sgpr": 6, "root_descriptor": true}},
                {"group": 0, "binding": 2, "kind": "uniform",
                 "source": {"user_sgpr": 6, "root_descriptor": true, "table_byte_offset": 96}},
                {"group": 0, "binding": 3, "kind": "texture",
                 "source": {"user_sgpr": 6, "table_byte_offset": 32}},
                {"group": 0, "binding": 4, "kind": "storage",
                 "source": {"user_sgpr": 6, "table_byte_offset": 128}},
                {"group": 0, "binding": 5, "kind": "storage_texture", "storage_format": 18,
                 "source": {"user_sgpr": 6, "table_byte_offset": 160}}
              ]
            }
          }
        })";
        StageReflectionIndex idx;
        std::string err;
        CHECK(parse_stage_reflection_json(doc, &idx, &err));
        CHECK(err.empty() && idx.size() == 1);

        bridge::PipelineLayoutDesc layout;
        std::string report;
        JoinStats stats;
        CHECK(join_pipeline_layout(rs9, true, {{GpuShaderStageBits::Compute, &idx.at(0x9009)}}, &layout, &report,
                                   &stats));
        CHECK(report.empty() && stats.implicit_heap_bindings == 5 && stats.root_descriptor_tables == 0);

        const bridge::LayoutBinding* b0 = find(layout, 0, 0);
        CHECK(b0 != nullptr && b0->join.root_constants && b0->join.heap == d3d::HeapAccess::None);
        const bridge::LayoutBinding* b1 = find(layout, 0, 1);
        CHECK(b1 != nullptr && b1->join.heap == d3d::HeapAccess::Resource &&
              b1->join.heap_index_source == d3d::HeapIndexSource::Constant &&
              b1->join.heap_index_parameter == d3d::kNoIndex && b1->join.heap_index_dword == 0 &&
              b1->join.heap_shift == 5 && b1->join.register_class == d3d::DescriptorRangeType::Srv);
        const bridge::LayoutBinding* b2 = find(layout, 0, 2);
        CHECK(b2 != nullptr && b2->join.heap == d3d::HeapAccess::Resource &&
              b2->join.heap_index_source == d3d::HeapIndexSource::Constant && b2->join.heap_index_dword == 3 &&
              b2->join.heap_shift == 5 && b2->join.register_class == d3d::DescriptorRangeType::Cbv);
        const bridge::LayoutBinding* b3 = find(layout, 0, 3);
        CHECK(b3 != nullptr && b3->join.heap == d3d::HeapAccess::Resource &&
              b3->join.heap_index_source == d3d::HeapIndexSource::Constant && b3->join.heap_index_dword == 1 &&
              b3->join.heap_shift == 5 && b3->join.register_class == d3d::DescriptorRangeType::Srv);
        const bridge::LayoutBinding* b4 = find(layout, 0, 4);
        CHECK(b4 != nullptr && b4->join.heap == d3d::HeapAccess::Resource &&
              b4->join.heap_index_source == d3d::HeapIndexSource::Constant && b4->join.heap_index_dword == 4 &&
              b4->join.heap_shift == 5 && b4->join.register_class == d3d::DescriptorRangeType::Uav);
        const bridge::LayoutBinding* b5 = find(layout, 0, 5);
        CHECK(b5 != nullptr && b5->join.heap == d3d::HeapAccess::Resource &&
              b5->join.heap_index_source == d3d::HeapIndexSource::Constant && b5->join.heap_index_dword == 5 &&
              b5->join.heap_shift == 5 && b5->join.register_class == d3d::DescriptorRangeType::Uav);

        // s7 is uncovered on RS #9 (SamplerHeapDirectlyIndexed is NOT set).
        StageReflection s7_stage;
        s7_stage.stage = ReflectedStage::Compute;
        ReflectedBinding s7_b;
        s7_b.group = 0;
        s7_b.binding = 0;
        s7_b.kind = ReflectedBindingKind::Sampler;
        s7_b.source = ReflectedSource::Table;
        s7_b.user_sgpr = 7;
        s7_b.table_byte_offset = 0;
        s7_stage.bindings = {s7_b};
        bridge::PipelineLayoutDesc untouched;
        report.clear();
        CHECK(!join_pipeline_layout(rs9, true, {{GpuShaderStageBits::Compute, &s7_stage}}, &untouched, &report));
        CHECK(untouched.groups.empty() && report.find("no root parameter covers s7") != std::string::npos);
    }

    // 2. RS #10-like compute RS: flags=0xc00, 6 root dwords at s4..s9 -> s_res=s10, s_smp=s11.
    // Includes end-to-end BindGroupPlan + BindGroupCache + StateBridgeDevice verification that
    // non-zero constant slots (e.g. sampler slots 64, 65, 67, 84 and resource slot 2) bind their
    // exact slots rather than slot 0.
    {
        d3d::RootSignatureDesc rs10;
        rs10.flags = d3d::RootSignatureFlags::CbvSrvUavHeapDirectlyIndexed |
                     d3d::RootSignatureFlags::SamplerHeapDirectlyIndexed;
        d3d::RootParameterDesc cbv0;
        cbv0.parameter_type = d3d::RootParameterType::Cbv;
        cbv0.shader_register = 0;  // s4..s5 (2 dw)
        d3d::RootParameterDesc cbv1;
        cbv1.parameter_type = d3d::RootParameterType::Cbv;
        cbv1.shader_register = 1;  // s6..s7 (2 dw)
        d3d::RootParameterDesc rc2;
        rc2.parameter_type = d3d::RootParameterType::Constants32Bit;
        rc2.shader_register = 2;
        rc2.num_32bit_values = 2;  // s8..s9 (2 dw)
        rs10.parameters = {cbv0, cbv1, rc2};

        StageReflection cs;
        cs.stage = ReflectedStage::Compute;
        auto mk_res = [](uint32_t binding, ReflectedBindingKind kind, ReflectedSource src, uint32_t sgpr,
                         uint32_t off) {
            ReflectedBinding b;
            b.group = 0;
            b.binding = binding;
            b.kind = kind;
            b.source = src;
            b.user_sgpr = sgpr;
            b.table_byte_offset = off;
            return b;
        };
        // Explicit root CBV at s4, implicit resource heap at s10 (RootDescriptor -> slot 0, Table +64 -> slot 2),
        // implicit sampler heap at s11 (+1024 -> slot 64, +1040 -> 65, +1072 -> 67, +1344 -> 84 comparison),
        // plus CBV-indexed resource heap reading its CBV from s10 + 96 (slot 3 of ResourceDescriptorHeap)
        // and ResourceHeap / SamplerHeap with base_user_sgpr = s10 / s11.
        ReflectedBinding cb_idx;
        cb_idx.group = 0;
        cb_idx.binding = 7;
        cb_idx.kind = ReflectedBindingKind::Texture;
        cb_idx.source = ReflectedSource::CbvIndexedResourceHeap;
        cb_idx.cbv = ReflectedCbv::Table;
        cb_idx.user_sgpr = 10;
        cb_idx.table_byte_offset = 96;  // CBV in ResourceDescriptorHeap[3]
        cb_idx.cb_byte_offset = 8;
        cb_idx.heap_shift = 5;

        ReflectedBinding lane_idx;
        lane_idx.group = 0;
        lane_idx.binding = 8;
        lane_idx.kind = ReflectedBindingKind::Sampler;
        lane_idx.source = ReflectedSource::LaneIndexedSamplerHeap;
        lane_idx.lane_max_count = 4;
        lane_idx.lane_has_window_base = true;
        lane_idx.cbv = ReflectedCbv::RootDescriptor;
        lane_idx.user_sgpr = 10;  // CBV in ResourceDescriptorHeap[0]
        lane_idx.cb_byte_offset = 4;
        lane_idx.heap_shift = 4;

        ReflectedBinding direct_res_heap;
        direct_res_heap.group = 0;
        direct_res_heap.binding = 9;
        direct_res_heap.kind = ReflectedBindingKind::Texture;
        direct_res_heap.source = ReflectedSource::ResourceHeap;
        direct_res_heap.heap_base_sgpr = 10;
        direct_res_heap.heap_stride = 32;
        direct_res_heap.heap_index_sgpr = 8;
        direct_res_heap.heap_shift = 5;

        cs.bindings = {
            mk_res(0, ReflectedBindingKind::Uniform, ReflectedSource::RootDescriptor, 4, 0),
            mk_res(1, ReflectedBindingKind::ReadOnlyStorage, ReflectedSource::RootDescriptor, 10, 0),
            mk_res(2, ReflectedBindingKind::Texture, ReflectedSource::Table, 10, 64),
            mk_res(3, ReflectedBindingKind::Sampler, ReflectedSource::Table, 11, 1024),
            mk_res(4, ReflectedBindingKind::Sampler, ReflectedSource::Table, 11, 1040),
            mk_res(5, ReflectedBindingKind::Sampler, ReflectedSource::Table, 11, 1072),
            mk_res(6, ReflectedBindingKind::ComparisonSampler, ReflectedSource::Table, 11, 1344),
            cb_idx,
            lane_idx,
            direct_res_heap,
        };

        bridge::PipelineLayoutDesc layout;
        std::string report;
        JoinStats stats;
        CHECK(join_pipeline_layout(rs10, true, {{GpuShaderStageBits::Compute, &cs}}, &layout, &report, &stats));
        CHECK(report.empty() && stats.implicit_heap_bindings == 9 && stats.lane_indexed_bindings == 1);

        const bridge::LayoutBinding* s64 = find(layout, 0, 3);
        const bridge::LayoutBinding* s65 = find(layout, 0, 4);
        const bridge::LayoutBinding* s67 = find(layout, 0, 5);
        const bridge::LayoutBinding* s84 = find(layout, 0, 6);
        const bridge::LayoutBinding* b7 = find(layout, 0, 7);
        const bridge::LayoutBinding* b8 = find(layout, 0, 8);
        CHECK(s64 != nullptr && s64->join.heap == d3d::HeapAccess::Sampler &&
              s64->join.heap_index_source == d3d::HeapIndexSource::Constant && s64->join.heap_index_dword == 64 &&
              s64->join.heap_shift == 4);
        CHECK(s65 != nullptr && s65->join.heap_index_dword == 65 && s65->join.heap_shift == 4);
        CHECK(s67 != nullptr && s67->join.heap_index_dword == 67 && s67->join.heap_shift == 4);
        CHECK(s84 != nullptr && s84->join.heap_index_dword == 84 && s84->join.heap_shift == 4 &&
              s84->entry.binding_type == static_cast<uint8_t>(GpuSamplerBindingType::Comparison));
        CHECK(b7 != nullptr && b7->join.heap == d3d::HeapAccess::Resource &&
              b7->join.heap_index_source == d3d::HeapIndexSource::CbTable &&
              b7->join.heap_index_parameter == d3d::kNoIndex && b7->join.cb_table_slot == 3 &&
              b7->join.cb_byte_offset == 8);
        CHECK(b8 != nullptr && b8->join.heap == d3d::HeapAccess::Sampler &&
              b8->join.heap_index_source == d3d::HeapIndexSource::CbTable &&
              b8->join.heap_index_parameter == d3d::kNoIndex && b8->join.cb_table_slot == 0 &&
              b8->join.cb_byte_offset == 4);

        // End-to-end BindGroupPlan + BindGroupCache verification against real heaps.
        d3d::RootSignature built_rs;
        CHECK(d3d::RootSignature::build(rs10, &built_rs).ok());
        std::vector<d3d::ShaderBinding> joins;
        for (const bridge::LayoutBinding& lb : layout.groups[0].bindings) {
            joins.push_back(lb.join);
        }
        const d3d::BindGroupPlan plan = d3d::BindGroupPlan::build(built_rs, joins, d3d::WebGpuStage::Compute);
        CHECK(plan.unresolved_count() == 0 && plan.has_heap_indexed());
        // Only explicit param 0 (root CBV) and param 2 (root constants for binding 9) are in parameter_mask!
        CHECK(plan.parameter_mask() == ((1ull << 0) | (1ull << 2)));

        d3d::DescriptorHeapRegistry reg;
        d3d::DescriptorHeap* res_heap = nullptr;
        d3d::DescriptorHeap* smp_heap = nullptr;
        CHECK(reg.create_heap({d3d::DescriptorHeapType::CbvSrvUav, 32, d3d::DescriptorHeapFlags::ShaderVisible, 0, 0},
                              &res_heap) == d3d::Status::Ok);
        CHECK(reg.create_heap({d3d::DescriptorHeapType::Sampler, 128, d3d::DescriptorHeapFlags::ShaderVisible, 0, 0},
                              &smp_heap) == d3d::Status::Ok);
        d3d::ViewDesc srv_desc;
        srv_desc.format = static_cast<uint32_t>(pso::DxgiFormat::R8G8B8A8_UNORM);
        srv_desc.dimension = d3d::ViewDimension::Texture2D;
        srv_desc.mip_levels = 1;
        // Resource slot 0: CBV at 0xA000 (also read as SRV if bound; keep distinct slots for clear checks).
        // Put SRVs at slots 0, 2, 5, 7 and CBV at slot 3 (0xB000).
        CHECK(reg.create_shader_resource_view(1000, &srv_desc, res_heap->cpu_handle(0)) == d3d::Status::Ok);
        CHECK(reg.create_shader_resource_view(1002, &srv_desc, res_heap->cpu_handle(2)) == d3d::Status::Ok);
        CHECK(reg.create_shader_resource_view(1005, &srv_desc, res_heap->cpu_handle(5)) == d3d::Status::Ok);
        CHECK(reg.create_shader_resource_view(1007, &srv_desc, res_heap->cpu_handle(7)) == d3d::Status::Ok);
        d3d::ConstantBufferViewDesc cb3{0xB000, 256};
        CHECK(reg.create_constant_buffer_view(&cb3, res_heap->cpu_handle(3)) == d3d::Status::Ok);
        // Distinct sampler anisotropy per slot so slot 0 can never pass for slots 64/65/67/84!
        auto write_smp = [&](uint32_t slot, uint32_t aniso) {
            d3d::SamplerDesc sd;
            sd.max_anisotropy = aniso;
            CHECK(reg.create_sampler(sd, smp_heap->cpu_handle(slot)) == d3d::Status::Ok);
        };
        write_smp(0, 1);
        write_smp(64, 4);
        write_smp(65, 6);
        write_smp(67, 8);
        write_smp(84, 12);

        FakeCbMemory mem;
        mem.dwords[0xB008] = 5;  // binding 7: CBV at ResourceDescriptorHeap[3] + 8 -> resource slot 5 (id 1005)

        d3d::RootArgumentState args;
        d3d::DescriptorHeap* heaps[2] = {res_heap, smp_heap};
        CHECK(args.set_descriptor_heaps(2, heaps) == d3d::Status::Ok);
        CHECK(args.set_root_signature(d3d::BindPoint::Compute, &built_rs) == d3d::Status::Ok);
        CHECK(args.set_root_view(d3d::BindPoint::Compute, 0, d3d::RootParameterType::Cbv, 0x9000) == d3d::Status::Ok);
        CHECK(args.set_root_constant(d3d::BindPoint::Compute, 2, 7, 0) == d3d::Status::Ok);  // binding 9 -> slot 7

        d3d::BindGroupCache cache;
        cache.set_slot_source(&mem);
        const d3d::BindGroupLookup bg =
            cache.get_or_build(args, d3d::BindPoint::Compute, plan, 1, fake_va_resolver, nullptr);
        CHECK(bg.status == d3d::Status::Ok && bg.bind_group != nullptr && bg.bind_group->entries.size() == 10);
        if (bg.bind_group != nullptr && bg.bind_group->entries.size() == 10) {
            const auto& e = bg.bind_group->entries;
            CHECK(e[0].kind == d3d::BindResourceKind::Buffer && e[0].resource_id == 500);
            CHECK(e[1].kind == d3d::BindResourceKind::TextureView && e[1].resource_id == 1000);
            CHECK(e[2].kind == d3d::BindResourceKind::TextureView && e[2].resource_id == 1002);
            CHECK(e[3].kind == d3d::BindResourceKind::Sampler && e[3].sampler.max_anisotropy == 4);
            CHECK(e[4].kind == d3d::BindResourceKind::Sampler && e[4].sampler.max_anisotropy == 6);
            CHECK(e[5].kind == d3d::BindResourceKind::Sampler && e[5].sampler.max_anisotropy == 8);
            CHECK(e[6].kind == d3d::BindResourceKind::Sampler && e[6].sampler.max_anisotropy == 12);
            CHECK(e[7].kind == d3d::BindResourceKind::TextureView && e[7].resource_id == 1005);
            CHECK(e[9].kind == d3d::BindResourceKind::TextureView && e[9].resource_id == 1007);
        }

        // Verify StateBridgeDevice compute pipeline creation preserves constant heap slots cleanly.
        bridge::StateBridgeDevice dev(reg);
        dev.register_shader_module(0x1010, 1);
        dev.register_pipeline_layout(0x1010, 0, layout);
        pso::D3D12ComputePsoDesc cpso_desc{};
        cpso_desc.cs = 0x1010;
        auto cpso = dev.create_compute_pipeline_state(cpso_desc, &built_rs);
        CHECK(cpso != nullptr && cpso->diagnostics() == 0);
    }

    // 3. Sampler-only directly indexed (flags=0x800 without 0x400): s_smp = base + total_root_dwords.
    {
        d3d::RootSignatureDesc rs_smp;
        rs_smp.flags = d3d::RootSignatureFlags::SamplerHeapDirectlyIndexed;
        d3d::RootParameterDesc rc;
        rc.parameter_type = d3d::RootParameterType::Constants32Bit;
        rc.num_32bit_values = 2;  // s4..s5
        rs_smp.parameters = {rc};

        StageReflection cs;
        cs.stage = ReflectedStage::Compute;
        ReflectedBinding b;
        b.group = 0;
        b.binding = 0;
        b.kind = ReflectedBindingKind::Sampler;
        b.source = ReflectedSource::Table;
        b.user_sgpr = 6;  // s4 + 2 + 0 (no s_res gap!)
        b.table_byte_offset = 32;
        cs.bindings = {b};

        bridge::PipelineLayoutDesc layout;
        std::string report;
        JoinStats stats;
        CHECK(join_pipeline_layout(rs_smp, true, {{GpuShaderStageBits::Compute, &cs}}, &layout, &report, &stats));
        CHECK(report.empty() && stats.implicit_heap_bindings == 1);
        const bridge::LayoutBinding* lb = find(layout, 0, 0);
        CHECK(lb != nullptr && lb->join.heap == d3d::HeapAccess::Sampler && lb->join.heap_index_dword == 2 &&
              lb->join.heap_shift == 4);
    }

    // 4. Negative tests:
    //    (a) flags=0x0 rejects uncovered s6/s10/s11;
    //    (b) kind mismatch: sampler on s_res, resource on s_smp;
    //    (c) misaligned byte offset: s_res+16, s_smp+8;
    //    (d) RootConstants on s_res/s_smp;
    //    (e) CBV-indexed CBV on s_smp; ResourceHeap base_user_sgpr == s_smp.
    {
        d3d::RootSignatureDesc rs_no_flags;
        rs_no_flags.flags = d3d::RootSignatureFlags::None;
        d3d::RootParameterDesc rc;
        rc.parameter_type = d3d::RootParameterType::Constants32Bit;
        rc.num_32bit_values = 2;  // s4..s5
        rs_no_flags.parameters = {rc};

        StageReflection unflagged;
        unflagged.stage = ReflectedStage::Compute;
        ReflectedBinding s6;
        s6.group = 0;
        s6.binding = 0;
        s6.kind = ReflectedBindingKind::ReadOnlyStorage;
        s6.source = ReflectedSource::RootDescriptor;
        s6.user_sgpr = 6;
        ReflectedBinding s7;
        s7.group = 0;
        s7.binding = 1;
        s7.kind = ReflectedBindingKind::Sampler;
        s7.source = ReflectedSource::Table;
        s7.user_sgpr = 7;
        s7.table_byte_offset = 1024;
        unflagged.bindings = {s6, s7};

        bridge::PipelineLayoutDesc untouched;
        std::string report;
        JoinStats stats;
        CHECK(!join_pipeline_layout(rs_no_flags, true, {{GpuShaderStageBits::Compute, &unflagged}}, &untouched,
                                    &report, &stats));
        CHECK(untouched.groups.empty() && stats.implicit_heap_bindings == 0);
        CHECK(report.find("no root parameter covers s6") != std::string::npos);
        CHECK(report.find("no root parameter covers s7") != std::string::npos);

        d3d::RootSignatureDesc rs_both = rs_no_flags;
        rs_both.flags = d3d::RootSignatureFlags::CbvSrvUavHeapDirectlyIndexed |
                        d3d::RootSignatureFlags::SamplerHeapDirectlyIndexed;  // s_res=s6, s_smp=s7

        StageReflection bad_kinds;
        bad_kinds.stage = ReflectedStage::Compute;
        ReflectedBinding smp_on_res = s6;
        smp_on_res.binding = 0;
        smp_on_res.kind = ReflectedBindingKind::Sampler;
        smp_on_res.source = ReflectedSource::Table;
        smp_on_res.user_sgpr = 6;
        ReflectedBinding res_on_smp = s7;
        res_on_smp.binding = 1;
        res_on_smp.kind = ReflectedBindingKind::Texture;
        res_on_smp.source = ReflectedSource::Table;
        res_on_smp.user_sgpr = 7;
        res_on_smp.table_byte_offset = 0;
        ReflectedBinding misalign_res = s6;
        misalign_res.binding = 2;
        misalign_res.source = ReflectedSource::Table;
        misalign_res.user_sgpr = 6;
        misalign_res.table_byte_offset = 16;
        ReflectedBinding misalign_smp = s7;
        misalign_smp.binding = 3;
        misalign_smp.source = ReflectedSource::Table;
        misalign_smp.user_sgpr = 7;
        misalign_smp.table_byte_offset = 8;
        ReflectedBinding rc_on_res = s6;
        rc_on_res.binding = 4;
        rc_on_res.kind = ReflectedBindingKind::Uniform;
        rc_on_res.source = ReflectedSource::RootConstants;
        rc_on_res.user_sgpr = 6;
        ReflectedBinding cb_on_smp;
        cb_on_smp.group = 0;
        cb_on_smp.binding = 5;
        cb_on_smp.kind = ReflectedBindingKind::Texture;
        cb_on_smp.source = ReflectedSource::CbvIndexedResourceHeap;
        cb_on_smp.cbv = ReflectedCbv::Table;
        cb_on_smp.user_sgpr = 7;
        cb_on_smp.table_byte_offset = 0;
        cb_on_smp.cb_byte_offset = 0;
        cb_on_smp.heap_shift = 5;
        ReflectedBinding wrong_heap_base;
        wrong_heap_base.group = 0;
        wrong_heap_base.binding = 6;
        wrong_heap_base.kind = ReflectedBindingKind::Texture;
        wrong_heap_base.source = ReflectedSource::ResourceHeap;
        wrong_heap_base.heap_base_sgpr = 7;  // s_smp, not s_res!
        wrong_heap_base.heap_stride = 32;
        wrong_heap_base.heap_index_sgpr = 4;
        wrong_heap_base.heap_shift = 5;
        bad_kinds.bindings = {smp_on_res, res_on_smp, misalign_res, misalign_smp, rc_on_res, cb_on_smp,
                              wrong_heap_base};

        report.clear();
        CHECK(!join_pipeline_layout(rs_both, true, {{GpuShaderStageBits::Compute, &bad_kinds}}, &untouched, &report,
                                    &stats));
        CHECK(untouched.groups.empty() && stats.implicit_heap_bindings == 0);
        CHECK(report.find("cannot come from the implicit resource heap at s6") != std::string::npos);
        CHECK(report.find("cannot come from the implicit sampler heap at s7") != std::string::npos);
        CHECK(report.find("byte offset 16 is not a descriptor boundary") != std::string::npos);
        CHECK(report.find("byte offset 8 is not a descriptor boundary") != std::string::npos);
        CHECK(report.find("not root constants") != std::string::npos);
        CHECK(report.find("implicit sampler heap, not a CBV_SRV_UAV descriptor heap") != std::string::npos);
        CHECK(report.find("s7 is the implicit sampler heap, not the resource heap") != std::string::npos);
    }

    // 5. Graphics VS/PS (default base s8 + shifted NGG VS at s16) and Mesh stage (explicit base 16)
    //    with flags=0xc00, plus a 0-parameter pure bindless RS.
    {
        d3d::RootSignatureDesc rs_gfx;
        rs_gfx.flags = d3d::RootSignatureFlags::CbvSrvUavHeapDirectlyIndexed |
                       d3d::RootSignatureFlags::SamplerHeapDirectlyIndexed;
        d3d::RootParameterDesc cbv0;
        cbv0.parameter_type = d3d::RootParameterType::Cbv;
        cbv0.shader_register = 0;  // +0..1
        d3d::RootParameterDesc rc1;
        rc1.parameter_type = d3d::RootParameterType::Constants32Bit;
        rc1.shader_register = 1;
        rc1.num_32bit_values = 2;  // +2..3 -> total 4 dw
        rs_gfx.parameters = {cbv0, rc1};

        // Default base s8: explicit s8..s11, s_res=s12, s_smp=s13.
        // Shifted NGG VS at s16: explicit s16..s19, s_res=s20, s_smp=s21.
        StageReflection vs;
        vs.stage = ReflectedStage::Vertex;
        ReflectedBinding vs_cbv;
        vs_cbv.group = 0;
        vs_cbv.binding = 0;
        vs_cbv.kind = ReflectedBindingKind::Uniform;
        vs_cbv.source = ReflectedSource::RootDescriptor;
        vs_cbv.user_sgpr = 16;  // forces shifted base s16
        ReflectedBinding vs_shared;
        vs_shared.group = 0;
        vs_shared.binding = 1;
        vs_shared.kind = ReflectedBindingKind::ReadOnlyStorage;
        vs_shared.source = ReflectedSource::Table;
        vs_shared.user_sgpr = 20;  // s_res at base 16
        vs_shared.table_byte_offset = 64;  // slot 2
        vs.bindings = {vs_cbv, vs_shared};

        StageReflection ps;
        ps.stage = ReflectedStage::Fragment;
        ReflectedBinding ps_shared = vs_shared;
        ps_shared.user_sgpr = 12;  // s_res at base 8 -> same slot 2, must merge cleanly!
        ReflectedBinding ps_smp;
        ps_smp.group = 0;
        ps_smp.binding = 2;
        ps_smp.kind = ReflectedBindingKind::Sampler;
        ps_smp.source = ReflectedSource::Table;
        ps_smp.user_sgpr = 13;  // s_smp at base 8
        ps_smp.table_byte_offset = 48;  // slot 3
        ps.bindings = {ps_shared, ps_smp};

        bridge::PipelineLayoutDesc layout;
        std::string report;
        JoinStats stats;
        CHECK(join_pipeline_layout(rs_gfx, false,
                                   {{GpuShaderStageBits::Vertex, &vs}, {GpuShaderStageBits::Fragment, &ps}}, &layout,
                                   &report, &stats));
        CHECK(report.empty() && stats.shifted_user_data_base_stages == 1 && stats.implicit_heap_bindings == 2);
        const bridge::LayoutBinding* shared = find(layout, 0, 1);
        CHECK(shared != nullptr &&
              shared->entry.visibility == (GpuShaderStageBits::Vertex | GpuShaderStageBits::Fragment) &&
              shared->join.heap == d3d::HeapAccess::Resource && shared->join.heap_index_dword == 2 &&
              shared->join.heap_shift == 5);
        const bridge::LayoutBinding* ps_s = find(layout, 0, 2);
        CHECK(ps_s != nullptr && ps_s->join.heap == d3d::HeapAccess::Sampler && ps_s->join.heap_index_dword == 3 &&
              ps_s->join.heap_shift == 4);

        // Mesh stage with explicit user_data_base = kMeshUserDataBase (16): s_res=s20, s_smp=s21.
        StageReflection mesh;
        mesh.stage = ReflectedStage::Compute;
        ReflectedBinding m_res = vs_shared;
        ReflectedBinding m_smp = ps_smp;
        m_smp.user_sgpr = 21;
        mesh.bindings = {m_res, m_smp};
        report.clear();
        JoinStats mesh_stats;
        CHECK(join_pipeline_layout(rs_gfx, false, {{GpuShaderStageBits::Compute, &mesh}}, &layout, &report,
                                   &mesh_stats, JoinOptions{false, kMeshUserDataBase}));
        CHECK(report.empty() && mesh_stats.shifted_user_data_base_stages == 0 &&
              mesh_stats.implicit_heap_bindings == 2);

        // Pure 0-parameter SM 6.6 bindless RS (parameters empty, flags=0xc00):
        // compute uses s_res=s4, s_smp=s5; shifted VS uses s_res=s16, s_smp=s17.
        d3d::RootSignatureDesc rs_empty;
        rs_empty.flags = d3d::RootSignatureFlags::CbvSrvUavHeapDirectlyIndexed |
                         d3d::RootSignatureFlags::SamplerHeapDirectlyIndexed;
        StageReflection empty_cs;
        empty_cs.stage = ReflectedStage::Compute;
        ReflectedBinding ecs_r = vs_shared;
        ecs_r.user_sgpr = 4;
        ecs_r.table_byte_offset = 0;
        ReflectedBinding ecs_s = ps_smp;
        ecs_s.user_sgpr = 5;
        ecs_s.table_byte_offset = 16;
        empty_cs.bindings = {ecs_r, ecs_s};
        report.clear();
        JoinStats empty_stats;
        CHECK(join_pipeline_layout(rs_empty, true, {{GpuShaderStageBits::Compute, &empty_cs}}, &layout, &report,
                                   &empty_stats));
        CHECK(report.empty() && empty_stats.implicit_heap_bindings == 2);

        StageReflection empty_vs;
        empty_vs.stage = ReflectedStage::Vertex;
        ReflectedBinding evs_r = ecs_r;
        evs_r.user_sgpr = 16;
        ReflectedBinding evs_s = ecs_s;
        evs_s.user_sgpr = 17;
        empty_vs.bindings = {evs_r, evs_s};
        report.clear();
        JoinStats empty_vs_stats;
        CHECK(join_pipeline_layout(rs_empty, false, {{GpuShaderStageBits::Vertex, &empty_vs}}, &layout, &report,
                                   &empty_vs_stats));
        CHECK(report.empty() && empty_vs_stats.shifted_user_data_base_stages == 1 &&
              empty_vs_stats.implicit_heap_bindings == 2);
    }
}

// #2085: Parse optional FSSM vs_outputs / ps_inputs semantic maps and compute non-identity
// SPI_PS_INPUT_CNTL @location remaps (Brightness Calibration PIPE 55065, swaps, identity, unused mask=0).
void test_interstage_location_remaps() {
    const char* doc = R"({
  "schema": "playgta6-stage-reflection", "version": 1,
  "stages": {
    "0x5fe7a88d9cb680b3": {
      "stage": "vertex",
      "vs_outputs": [
        {"semantic_reg": 1, "location": 0},
        {"semantic_reg": 2, "location": 1}
      ],
      "bindings": []
    },
    "0x56214b8d3b10a108": {
      "stage": "fragment",
      "ps_inputs": [
        {"semantic_reg": 1, "location": 0, "mask": 0},
        {"semantic_reg": 2, "location": 0, "mask": 3}
      ],
      "bindings": []
    },
    "0x0000000000007701": {
      "stage": "fragment",
      "ps_inputs": [
        {"semantic_reg": 1, "location": 0, "mask": 15},
        {"semantic_reg": 2, "location": 1, "mask": 3}
      ],
      "bindings": []
    },
    "0x0000000000007702": {
      "stage": "fragment",
      "ps_inputs": [
        {"semantic_reg": 2, "location": 0, "mask": 3},
        {"semantic_reg": 1, "location": 1, "mask": 15}
      ],
      "bindings": []
    }
  }
})";
    StageReflectionIndex index;
    std::string error;
    CHECK(parse_stage_reflection_json(doc, &index, &error));
    CHECK(error.empty());

    const StageReflection& vs = index.at(0x5fe7a88d9cb680b3ull);
    const StageReflection& ps_calib = index.at(0x56214b8d3b10a108ull);
    const StageReflection& ps_ident = index.at(0x7701ull);
    const StageReflection& ps_swap = index.at(0x7702ull);

    CHECK(vs.vs_outputs.size() == 2);
    CHECK(vs.vs_outputs[0].semantic_reg == 1 && vs.vs_outputs[0].location == 0 && !vs.vs_outputs[0].has_mask);
    CHECK(vs.vs_outputs[1].semantic_reg == 2 && vs.vs_outputs[1].location == 1 && !vs.vs_outputs[1].has_mask);
    CHECK(ps_calib.ps_inputs.size() == 2);
    CHECK(ps_calib.ps_inputs[0].has_mask && ps_calib.ps_inputs[0].mask == 0);
    CHECK(ps_calib.ps_inputs[1].semantic_reg == 2 && ps_calib.ps_inputs[1].location == 0 &&
          ps_calib.ps_inputs[1].has_mask && ps_calib.ps_inputs[1].mask == 3);

    // 1. Brightness Calibration: unused mask=0 entry skipped; TEXCOORD0 remaps @location(0) -> @location(1).
    const std::vector<InterstageLocationRemap> calib_remaps = compute_interstage_location_remaps(vs, ps_calib);
    CHECK(calib_remaps.size() == 1);
    if (calib_remaps.size() == 1) {
        CHECK(calib_remaps[0].semantic_reg == 2 && calib_remaps[0].ps_location == 0 &&
              calib_remaps[0].vs_location == 1);
    }

    // 2. Identity mapping produces zero remaps.
    CHECK(compute_interstage_location_remaps(vs, ps_ident).empty());

    // 3. Swap/cycle (0->1, 1->0) produces both remaps sorted by ascending ps_location.
    const std::vector<InterstageLocationRemap> swap_remaps = compute_interstage_location_remaps(vs, ps_swap);
    CHECK(swap_remaps.size() == 2);
    if (swap_remaps.size() == 2) {
        CHECK(swap_remaps[0].semantic_reg == 2 && swap_remaps[0].ps_location == 0 && swap_remaps[0].vs_location == 1);
        CHECK(swap_remaps[1].semantic_reg == 1 && swap_remaps[1].ps_location == 1 && swap_remaps[1].vs_location == 0);
    }

    // 4. Missing metadata on either stage produces zero remaps.
    StageReflection empty_stage;
    CHECK(compute_interstage_location_remaps(empty_stage, ps_calib).empty());
    CHECK(compute_interstage_location_remaps(vs, empty_stage).empty());

    // 5. Malformed vs_outputs / ps_inputs rejected cleanly.
    const char* bad_docs[] = {
        R"({"schema":"playgta6-stage-reflection","version":1,"stages":{"0x1":{"stage":"vertex","vs_outputs":"bad","bindings":[]}}})",
        R"({"schema":"playgta6-stage-reflection","version":1,"stages":{"0x1":{"stage":"vertex","vs_outputs":[{"semantic_reg":256,"location":0}],"bindings":[]}}})",
        R"({"schema":"playgta6-stage-reflection","version":1,"stages":{"0x1":{"stage":"fragment","vs_outputs":[{"semantic_reg":1,"location":0}],"bindings":[]}}})",
        R"({"schema":"playgta6-stage-reflection","version":1,"stages":{"0x1":{"stage":"fragment","ps_inputs":[{"semantic_reg":2,"location":0,"mask":16}],"bindings":[]}}})",
        R"({"schema":"playgta6-stage-reflection","version":1,"stages":{"0x1":{"stage":"vertex","ps_inputs":[{"semantic_reg":2,"location":0,"mask":3}],"bindings":[]}}})",
    };
    for (const char* bad_doc : bad_docs) {
        StageReflectionIndex rejected;
        error.clear();
        CHECK(!parse_stage_reflection_json(bad_doc, &rejected, &error));
        CHECK(!error.empty());
    }
}

int main() {
    test_rt_arena_bindings();
    test_mesh_output();
    test_parse();
    test_user_data_rule();
    test_contained_sgpr_and_bitcast_vertex();
    test_vertex_join();
    test_vertex_join_wide_fetch();
    test_vertex_default_inputs();
    test_vertex_sub_element_inference();
    test_layout_join();
    test_plan_direct_slot();
    test_heap_sources();
    test_cbv_indexed_heap_sources();
    test_lane_indexed_heap_sources();
    test_s63_lane_indexed_sidecar();
    test_inline_sampler();
    test_root_descriptor_over_table();
    test_unresolved_source();
    test_g83_unresolved_and_multistage();
    test_g83_heap_strides_and_arrays();
    test_g83_mesh_and_rt_group3_joins();
    test_per_stage_user_data_base_and_shifted_fallback();
    test_fragment_high_bindings_64_to_95();
    test_sampler_table_16_byte_slots();
    test_wide_rs_covered_shifted_user_data_base();
    test_cbv_indexed_root_constants_and_table_coercion();
    test_vertex_join_and_pso_completeness();
    test_sm66_implicit_heap_sgprs();
    test_interstage_location_remaps();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_stage_reflection: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_stage_reflection: all checks passed\n");
    return 0;
}
