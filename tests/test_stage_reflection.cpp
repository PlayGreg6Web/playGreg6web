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
    CHECK(report.find("@group(0) @binding(0) <- table at s12 + 0: no root parameter starts at s12") != std::string::npos);
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

    // Well formed: the join fails with the lane-indexed reason and counts both bindings.
    const d3d::RootSignatureDesc rs = ui_root_signature();
    bridge::PipelineLayoutDesc layout;
    std::string report;
    JoinStats stats;
    CHECK(!join_pipeline_layout(rs, false, {{GpuShaderStageBits::Fragment, &ps}}, &layout, &report, &stats));
    CHECK(stats.lane_indexed_bindings == 2 && stats.table_slots_outside_rs == 0);
    CHECK(report.find("lane-indexed heap window of 64 descriptors needs a binding array") != std::string::npos);
    CHECK(report.find("lane-indexed heap window of 16 descriptors") != std::string::npos);
    CHECK(report.find("resource heap[(0 + lane) << 5] x 64") != std::string::npos);
    CHECK(report.find("sampler heap[(cb(root CBV at s11)[8] + lane) << 4] x 16") != std::string::npos);

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

    // A stage without lane-indexed bindings joined next to one with them still fails the whole join.
    StageReflection vs;
    vs.stage = ReflectedStage::Vertex;
    report.clear();
    JoinStats mixed;
    CHECK(!join_pipeline_layout(rs, false, {{GpuShaderStageBits::Vertex, &vs}, {GpuShaderStageBits::Fragment, &ps}},
                                &layout, &report, &mixed));
    CHECK(mixed.lane_indexed_bindings == 2);
}

int main() {
    test_parse();
    test_user_data_rule();
    test_vertex_join();
    test_vertex_join_wide_fetch();
    test_layout_join();
    test_plan_direct_slot();
    test_heap_sources();
    test_cbv_indexed_heap_sources();
    test_lane_indexed_heap_sources();
    test_inline_sampler();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_stage_reflection: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_stage_reflection: all checks passed\n");
    return 0;
}
