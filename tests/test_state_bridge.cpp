#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <atomic>
#include <thread>
#include <vector>

#include "abi/playgta6_abi.h"
#include "gpu/cmd_encoder.h"
#include "gpu/d3d12_guest_objects.h"
#include "gpu/d3d12_guest_resources.h"
#include "gpu/descriptor_heap.h"
#include "gpu/descriptor_words.h"
#include "gpu/gfx10_detile.h"
#include "gpu/indirect_clamp_batch.h"
#include "gpu/pipeline_prewarm.h"
#include "gpu/prewarm_pipeline_desc.h"
#include "gpu/pso_cache.h"
#include "wasm_boot.h"
#include "gpu/state_bridge.h"
#include "guest/com_bridge.h"
#include "os/vmem.h"
#include "os/win32_kernel.h"

#ifndef PG6_TEST_FIXTURE_DIR
#define PG6_TEST_FIXTURE_DIR "tests/fixtures"
#endif

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_state_bridge.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6;
namespace bridge = playgta6::gpu::bridge;
namespace pso = playgta6::gpu::pso;
namespace d3d = playgta6::gpu::d3d12_state;

struct Packet {
    GpuOpcode opcode{};
    std::vector<uint8_t> payload;
};

std::vector<Packet> decode(const gpu::CommandEncoder& enc, size_t from = 0) {
    std::vector<Packet> out;
    size_t off = from;
    while (off + sizeof(GpuPacketHeader) <= enc.size()) {
        GpuPacketHeader h{};
        std::memcpy(&h, enc.data() + off, sizeof(h));
        off += sizeof(h);
        Packet p;
        p.opcode = static_cast<GpuOpcode>(h.opcode);
        p.payload.assign(enc.data() + off, enc.data() + off + h.payload_bytes);
        off += h.payload_bytes;
        out.push_back(std::move(p));
    }
    CHECK(off == enc.size());
    return out;
}

template <typename T>
T pod(const Packet& p, size_t offset = 0) {
    T v{};
    CHECK(p.payload.size() >= offset + sizeof(T));
    if (p.payload.size() >= offset + sizeof(T)) {
        std::memcpy(&v, p.payload.data() + offset, sizeof(T));
    }
    return v;
}

std::vector<GpuOpcode> opcodes(const std::vector<Packet>& packets) {
    std::vector<GpuOpcode> ops;
    for (const Packet& p : packets) {
        ops.push_back(p.opcode);
    }
    return ops;
}

bridge::BridgeOptions counting_options(uint32_t first_id) {
    bridge::BridgeOptions o;
    // Atomic: the concurrent-recording case calls the allocator from several threads.
    auto next = std::make_shared<std::atomic<uint32_t>>(first_id);
    o.id_allocator = [next] { return next->fetch_add(1); };
    return o;
}

d3d::DescriptorHeap* make_heap(d3d::DescriptorHeapRegistry& reg, d3d::DescriptorHeapType type, uint32_t n, bool visible) {
    d3d::DescriptorHeapDesc desc;
    desc.type = type;
    desc.num_descriptors = n;
    desc.flags = visible ? d3d::DescriptorHeapFlags::ShaderVisible : d3d::DescriptorHeapFlags::None;
    d3d::DescriptorHeap* heap = nullptr;
    CHECK(reg.create_heap(desc, &heap) == d3d::Status::Ok);
    return heap;
}

GpuBindGroupLayoutEntry layout_entry(uint32_t binding, uint32_t visibility, GpuBindingLayoutType type,
                                     uint8_t binding_type, uint8_t view_dimension = 0, uint8_t flags = 0) {
    GpuBindGroupLayoutEntry e{};
    e.binding = binding;
    e.visibility = visibility;
    e.type = static_cast<uint8_t>(type);
    e.binding_type = binding_type;
    e.view_dimension = view_dimension;
    e.flags = flags;
    return e;
}

d3d::ShaderBinding join(uint32_t binding, d3d::DescriptorRangeType cls, uint32_t reg, bool root_constants = false) {
    d3d::ShaderBinding b;
    b.binding = binding;
    b.register_class = cls;
    b.shader_register = reg;
    b.root_constants = root_constants;
    return b;
}

// Shared fixture: one swapchain RTV, one offscreen RTV, one DSV, an SRV table, root constants and a
// static sampler, and a graphics PSO with a registered S6-style layout.
struct Scene {
    static constexpr uint32_t kSwapchain = 100;
    static constexpr uint32_t kOffscreen = 101;
    static constexpr uint32_t kDepth = 102;
    static constexpr uint32_t kAlbedo = 103;
    static constexpr uint32_t kAlbedo2 = 104;
    static constexpr uint32_t kVertexBuffer = 200;
    static constexpr uint32_t kVsModule = 1;
    static constexpr uint32_t kFsModule = 2;
    static constexpr uint64_t kVs = 0xA1;
    static constexpr uint64_t kPs = 0xB2;
    static constexpr uint64_t kBglContent = 0x472f25790659490full;
    static constexpr uint64_t kPlContent = 0x1122334455667788ull;

    d3d::DescriptorHeapRegistry heaps;
    d3d::ResourceStateRegistry states;
    bridge::StateBridgeDevice device;
    d3d::DescriptorHeap* rtv_heap = nullptr;
    d3d::DescriptorHeap* dsv_heap = nullptr;
    d3d::DescriptorHeap* srv_heap = nullptr;
    d3d::RootSignature root_signature;
    std::unique_ptr<bridge::GraphicsPipelineState> pso;

    explicit Scene(uint32_t first_id = 1000) : device(heaps, counting_options(first_id), &states) {
        rtv_heap = make_heap(heaps, d3d::DescriptorHeapType::Rtv, 4, false);
        dsv_heap = make_heap(heaps, d3d::DescriptorHeapType::Dsv, 2, false);
        srv_heap = make_heap(heaps, d3d::DescriptorHeapType::CbvSrvUav, 8, true);

        device.register_texture(kSwapchain, {pso::DxgiFormat::B8G8R8A8_UNORM, 1280, 720, 1, 1, 1,
                                             GpuTextureDimension::Dim2D, true});
        device.register_texture(kOffscreen, {pso::DxgiFormat::B8G8R8A8_UNORM, 640, 360, 1, 1, 1,
                                             GpuTextureDimension::Dim2D, false});
        device.register_texture(kDepth, {pso::DxgiFormat::D32_FLOAT, 1280, 720, 1, 1, 1,
                                         GpuTextureDimension::Dim2D, false});
        device.register_texture(kAlbedo, {pso::DxgiFormat::R8G8B8A8_UNORM, 256, 256, 1, 9, 1,
                                          GpuTextureDimension::Dim2D, false});
        device.register_texture(kAlbedo2, {pso::DxgiFormat::R8G8B8A8_UNORM, 64, 64, 1, 1, 1,
                                           GpuTextureDimension::Dim2D, false});
        device.register_shader_module(kVs, kVsModule);
        device.register_shader_module(kPs, kFsModule);

        CHECK(heaps.create_render_target_view(kSwapchain, nullptr, rtv_heap->cpu_handle(0)) == d3d::Status::Ok);
        d3d::ViewDesc rtv1;
        rtv1.format = 87;  // B8G8R8A8_UNORM
        rtv1.dimension = d3d::ViewDimension::Texture2D;
        CHECK(heaps.create_render_target_view(kOffscreen, &rtv1, rtv_heap->cpu_handle(1)) == d3d::Status::Ok);
        d3d::ViewDesc dsv;
        dsv.format = 40;  // D32_FLOAT
        dsv.dimension = d3d::DsvDimension::Texture2D;
        CHECK(heaps.create_depth_stencil_view(kDepth, &dsv, dsv_heap->cpu_handle(0)) == d3d::Status::Ok);
        write_albedo_srv(kAlbedo, 0);

        d3d::RootSignatureDesc rs;
        d3d::RootParameterDesc table;
        table.parameter_type = d3d::RootParameterType::DescriptorTable;
        d3d::DescriptorRangeDesc srv_range;
        srv_range.range_type = d3d::DescriptorRangeType::Srv;
        srv_range.num_descriptors = 1;
        table.ranges.push_back(srv_range);
        rs.parameters.push_back(table);
        d3d::RootParameterDesc constants;
        constants.parameter_type = d3d::RootParameterType::Constants32Bit;
        constants.num_32bit_values = 4;
        rs.parameters.push_back(constants);
        rs.static_samplers.push_back(d3d::StaticSamplerDesc{});
        CHECK(d3d::RootSignature::build(rs, &root_signature).ok());

        bridge::PipelineLayoutDesc layout;
        layout.content_id = kPlContent;
        bridge::GroupLayoutDesc group;
        group.content_id = kBglContent;
        group.bindings.push_back({layout_entry(2, GpuShaderStageBits::Fragment, GpuBindingLayoutType::Sampler,
                                               static_cast<uint8_t>(GpuSamplerBindingType::Filtering)),
                                  join(2, d3d::DescriptorRangeType::Sampler, 0)});
        group.bindings.push_back({layout_entry(0, GpuShaderStageBits::Vertex | GpuShaderStageBits::Fragment,
                                               GpuBindingLayoutType::Buffer,
                                               static_cast<uint8_t>(GpuBufferBindingType::Uniform)),
                                  join(0, d3d::DescriptorRangeType::Cbv, 0, true)});
        group.bindings.push_back({layout_entry(1, GpuShaderStageBits::Fragment, GpuBindingLayoutType::Texture,
                                               static_cast<uint8_t>(GpuTextureSampleType::Float),
                                               static_cast<uint8_t>(GpuTextureViewDimension::Dim2D)),
                                  join(1, d3d::DescriptorRangeType::Srv, 0)});
        layout.groups.push_back(group);
        device.register_pipeline_layout(kVs, kPs, layout);

        pso = device.create_graphics_pipeline_state(pso_desc(), &root_signature);
        CHECK(pso->diagnostics() == 0);
    }

    void write_albedo_srv(uint32_t texture, uint32_t slot) {
        d3d::ViewDesc srv;
        srv.format = 28;  // R8G8B8A8_UNORM
        srv.dimension = d3d::ViewDimension::Texture2D;
        srv.first_mip = 0;
        srv.mip_levels = 0xffff;
        CHECK(heaps.create_shader_resource_view(texture, &srv, srv_heap->cpu_handle(slot)) == d3d::Status::Ok);
    }

    static pso::D3D12GraphicsPsoDesc pso_desc() {
        static const pso::D3D12InputElementDesc elements[] = {
            {"POSITION", 0, pso::DxgiFormat::R32G32B32_FLOAT, 0, 0, pso::D3D12InputClassification::PerVertexData, 0},
        };
        pso::D3D12GraphicsPsoDesc d{};
        d.root_signature_id = 7;
        d.vs = kVs;
        d.ps = kPs;
        d.pInputElementDescs = elements;
        d.NumInputElements = 1;
        d.NumRenderTargets = 1;
        d.RTVFormats[0] = pso::DxgiFormat::B8G8R8A8_UNORM;
        d.DSVFormat = pso::DxgiFormat::D32_FLOAT;
        return d;
    }

    d3d::CpuDescriptorHandle rtv(uint32_t i) const { return rtv_heap->cpu_handle(i); }
    d3d::CpuDescriptorHandle dsv() const { return dsv_heap->cpu_handle(0); }

    // Common prologue: heaps, root signature, PSO, root arguments, IA state.
    void bind_root(bridge::CommandListBridge& list) {
        d3d::DescriptorHeap* visible[] = {srv_heap};
        CHECK(list.SetDescriptorHeaps(1, visible) == d3d::Status::Ok);
        CHECK(list.SetGraphicsRootSignature(&root_signature) == d3d::Status::Ok);
        list.SetPipelineState(pso.get());
        CHECK(list.SetGraphicsRootDescriptorTable(0, srv_heap->gpu_handle(0)) == d3d::Status::Ok);
        const uint32_t tint[4] = {0x3f800000u, 0x3f000000u, 0x3e800000u, 0x3f800000u};
        CHECK(list.SetGraphicsRoot32BitConstants(1, 4, tint, 0) == d3d::Status::Ok);
        list.IASetPrimitiveTopology(pso::D3DPrimitiveTopology::TriangleList);
        const gpu::d3d12_xs::D3D12VertexBufferView vb{kVertexBuffer, 0, 36, 12};
        list.IASetVertexBuffers(0, 1, &vb);
    }
};

std::string to_hex(const uint8_t* data, size_t size) {
    static const char* digits = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < size; ++i) {
        s.push_back(digits[data[i] >> 4]);
        s.push_back(digits[data[i] & 0xF]);
        if (i % 32 == 31 || i + 1 == size) {
            s.push_back('\n');
        }
    }
    return s;
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    std::string text = ss.str();
    std::string compact;
    for (char c : text) {
        if (c != '\r') {
            compact.push_back(c);
        }
    }
    return compact;
}

// ------------------------------------------------------------------------------------------------

void test_seam_conversions() {
    CHECK(bridge::to_wgpu_index_format(GpuIndexFormat::Uint16) == pso::WgpuIndexFormat::Uint16);
    CHECK(bridge::to_wgpu_index_format(GpuIndexFormat::Uint32) == pso::WgpuIndexFormat::Uint32);

    bool ok = false;
    CHECK(bridge::to_view_dimension(d3d::ViewDimension::Texture2DArray, &ok) == GpuTextureViewDimension::Dim2DArray && ok);
    CHECK(bridge::to_view_dimension(d3d::ViewDimension::TextureCubeArray, &ok) == GpuTextureViewDimension::CubeArray && ok);
    CHECK(bridge::to_view_dimension(d3d::ViewDimension::Texture2DMs, &ok) == GpuTextureViewDimension::Dim2D && ok);
    CHECK(bridge::to_view_dimension(d3d::ViewDimension::Unknown, &ok) == GpuTextureViewDimension::Undefined && ok);
    CHECK(bridge::to_view_dimension(d3d::ViewDimension::Texture1DArray, &ok) == GpuTextureViewDimension::Undefined && !ok);

    // Default D3D12 sampler: trilinear wrap.
    bool exact = false;
    GpuCreateSampler2Payload s = bridge::to_sampler_payload(d3d::SamplerDesc{}, &exact);
    CHECK(exact);
    CHECK(s.address_u == static_cast<uint8_t>(GpuAddressMode::Repeat));
    CHECK(s.mag_filter == 1 && s.min_filter == 1 && s.mipmap_filter == 1);
    CHECK(s.compare == 0 && s.max_anisotropy == 1);
    CHECK(s.lod_min_clamp == 0.0f && s.lod_max_clamp == 32.0f);

    // Comparison point/linear filter (D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT = 0x94), clamp.
    d3d::SamplerDesc cmp;
    cmp.filter = 0x94;
    cmp.address_u = cmp.address_v = cmp.address_w = 3;
    cmp.comparison_func = static_cast<uint32_t>(pso::D3D12ComparisonFunc::LessEqual);
    cmp.max_lod = 4.0f;
    s = bridge::to_sampler_payload(cmp, &exact);
    CHECK(exact);
    CHECK(s.address_v == static_cast<uint8_t>(GpuAddressMode::ClampToEdge));
    CHECK(s.min_filter == 1 && s.mag_filter == 1 && s.mipmap_filter == 0);
    CHECK(s.compare == static_cast<uint8_t>(pso::WgpuCompareFunction::LessEqual));
    CHECK(s.lod_max_clamp == 4.0f);

    // A comparison sampler bound where the layout samples without comparison loses the comparison;
    // in a comparison entry, and for plain samplers, nothing changes.
    bool changed = false;
    const d3d::SamplerDesc plain_slot =
        bridge::sampler_for_binding(cmp, static_cast<uint8_t>(GpuSamplerBindingType::Filtering), &changed);
    CHECK(changed && plain_slot.filter == 0x14);
    s = bridge::to_sampler_payload(plain_slot, &exact);
    CHECK(s.compare == 0 && s.min_filter == 1 && s.mag_filter == 1 && s.mipmap_filter == 0);
    const d3d::SamplerDesc cmp_slot =
        bridge::sampler_for_binding(cmp, static_cast<uint8_t>(GpuSamplerBindingType::Comparison), &changed);
    CHECK(!changed && cmp_slot.filter == 0x94);
    const d3d::SamplerDesc plain = bridge::sampler_for_binding(
        d3d::SamplerDesc{}, static_cast<uint8_t>(GpuSamplerBindingType::Filtering), &changed);
    CHECK(!changed && plain.filter == d3d::SamplerDesc{}.filter);

    // Anisotropic + border address mode: approximated.
    d3d::SamplerDesc aniso;
    aniso.filter = 0x55;
    aniso.max_anisotropy = 32;
    aniso.address_u = 4;
    s = bridge::to_sampler_payload(aniso, &exact);
    CHECK(!exact);
    CHECK(s.max_anisotropy == 16 && s.min_filter == 1 && s.mipmap_filter == 1);
}

// The end-to-end contract: a representative command list and the exact packet stream it produces.
// The stream is pinned by tests/fixtures/state_bridge_frame.hex, which the JS worker test
// (tests/test_render_state_packets.mjs) and the disassembler test (tests/test_gpu_trace.py) replay.
void test_representative_command_list() {
    Scene scene;
    gpu::CommandEncoder enc;
    bridge::CommandListBridge list(scene.device, enc);

    scene.bind_root(list);
    const d3d::CpuDescriptorHandle rtv0 = scene.rtv(0);
    const d3d::CpuDescriptorHandle dsv = scene.dsv();
    list.OMSetRenderTargets(1, &rtv0, false, &dsv);
    const float clear[4] = {0.1f, 0.2f, 0.3f, 1.0f};
    list.ClearRenderTargetView(rtv0, clear);
    list.ClearDepthStencilView(dsv, pso::D3D12ClearFlags::Depth, 1.0f, 0);
    const pso::D3D12Viewport viewport{10.0f, 20.0f, 640.0f, 360.0f, 0.0f, 1.0f};
    list.RSSetViewports(1, &viewport);
    const pso::D3D12Rect scissor{8, 8, 1272, 712};
    list.RSSetScissorRects(1, &scissor);
    list.OMSetStencilRef(0x80);
    CHECK(enc.size() == 0);  // nothing is emitted before the first draw
    list.DrawInstanced(3, 1, 0, 0);

    // Switch to the offscreen target (no depth): new pass, new pipeline variant, new constants.
    const d3d::CpuDescriptorHandle rtv1 = scene.rtv(1);
    list.OMSetRenderTargets(1, &rtv1, false, nullptr);
    const float clear2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    list.ClearRenderTargetView(rtv1, clear2);
    const uint32_t tint2[4] = {0u, 0x3f800000u, 0u, 0x3f800000u};
    CHECK(list.SetGraphicsRoot32BitConstants(1, 4, tint2, 0) == d3d::Status::Ok);
    list.DrawInstanced(6, 2, 0, 0);
    list.Close();

    const std::vector<Packet> p = decode(enc);
    using O = GpuOpcode;
    const std::vector<GpuOpcode> expected = {
        O::BeginRenderPass, O::CreateBindGroupLayout, O::CreatePipelineLayout, O::CreateRenderPipeline2,
        O::SetPipeline, O::SetViewport, O::SetScissorRect, O::SetStencilReference,
        O::CreateBuffer, O::CreateSampler2, O::CreateBindGroup2, O::WriteBuffer, O::SetBindGroup2,
        O::SetVertexBuffer, O::Draw, O::EndRenderPass,
        O::BeginRenderPass, O::CreateRenderPipeline2, O::SetPipeline, O::SetViewport, O::SetScissorRect,
        O::SetStencilReference, O::WriteBuffer, O::SetBindGroup2, O::SetVertexBuffer, O::Draw, O::EndRenderPass,
    };
    CHECK(opcodes(p) == expected);
    if (opcodes(p) != expected) {
        for (const Packet& pk : p) {
            std::fprintf(stderr, "  opcode %u (%zu bytes)\n", static_cast<unsigned>(pk.opcode), pk.payload.size());
        }
        return;
    }

    // Pass 1: swapchain color cleared, depth cleared, folded into load ops.
    const auto pass1 = pod<GpuBeginRenderPassPayload>(p[0]);
    CHECK(pass1.color_count == 1);
    CHECK(pass1.color[0].texture_id == kGpuSwapchainTextureId);
    CHECK(pass1.color[0].load_op == static_cast<uint8_t>(GpuLoadOp::Clear));
    CHECK(pass1.color[0].store_op == static_cast<uint8_t>(GpuStoreOp::Store));
    CHECK(pass1.color[0].clear_value[0] == 0.1f && pass1.color[0].clear_value[3] == 1.0f);
    CHECK(pass1.color[0].view.format == static_cast<uint8_t>(pso::WgpuTextureFormat::Bgra8Unorm));
    CHECK(pass1.color[0].view.dimension == static_cast<uint8_t>(GpuTextureViewDimension::Dim2D));
    CHECK(pass1.color[0].view.mip_level_count == 1 && pass1.color[0].view.array_layer_count == 1);
    CHECK(pass1.depth_stencil.texture_id == Scene::kDepth);
    CHECK(pass1.depth_stencil.depth_load_op == static_cast<uint8_t>(GpuLoadOp::Clear));
    CHECK(pass1.depth_stencil.depth_clear_value == 1.0f);
    CHECK(pass1.depth_stencil.stencil_load_op == 0 && pass1.depth_stencil.stencil_store_op == 0);  // no stencil
    CHECK(pass1.depth_stencil.view.format == static_cast<uint8_t>(pso::WgpuTextureFormat::Depth32Float));

    // Layouts: ids 1000 (group) / 1001 (pipeline); root constants became a dynamic uniform.
    const auto bgl = pod<GpuCreateBindGroupLayoutPayload>(p[1]);
    CHECK(bgl.id == 1000 && bgl.entry_count == 3);
    CHECK(bgl.content_id == 0);  // dynamic offset added: no longer the manifest layout
    const auto e0 = pod<GpuBindGroupLayoutEntry>(p[1], sizeof(bgl));
    CHECK(e0.binding == 0 && (e0.flags & GpuBindGroupLayoutEntryFlags::HasDynamicOffset) != 0);
    const auto e2 = pod<GpuBindGroupLayoutEntry>(p[1], sizeof(bgl) + 2 * sizeof(GpuBindGroupLayoutEntry));
    CHECK(e2.binding == 2 && e2.type == static_cast<uint8_t>(GpuBindingLayoutType::Sampler));
    const auto pl = pod<GpuCreatePipelineLayoutPayload>(p[2]);
    CHECK(pl.id == 1001 && pl.bind_group_layout_count == 1 && pl.content_id == Scene::kPlContent);
    CHECK(pod<uint32_t>(p[2], sizeof(pl)) == 1000);

    // Pipeline 1002 carries the G7 key verbatim.
    const auto rp = pod<GpuCreateRenderPipeline2Payload>(p[3]);
    CHECK(rp.id == 1002 && rp.vs_module_id == Scene::kVsModule && rp.fs_module_id == Scene::kFsModule);
    CHECK(rp.pipeline_layout_id == 1001);
    CHECK(rp.key.vs_shader == Scene::kVs && rp.key.ps_shader == Scene::kPs && rp.key.root_signature_id == 7);
    CHECK(rp.key.color_target_count == 1);
    CHECK(rp.key.color_targets[0].format == static_cast<uint8_t>(pso::WgpuTextureFormat::Bgra8Unorm));
    CHECK(rp.key.depth_stencil.format == static_cast<uint8_t>(pso::WgpuTextureFormat::Depth32Float));
    CHECK(rp.key.vertex_buffers[0].array_stride == 12);
    CHECK(rp.key.vertex_attributes[0].format == static_cast<uint8_t>(pso::WgpuVertexFormat::Float32x3));
    CHECK(pod<GpuSetPipelinePayload>(p[4]).pipeline_id == 1002);

    const auto vp = pod<GpuSetViewportPayload>(p[5]);
    CHECK(vp.x == 10.0f && vp.y == 20.0f && vp.width == 640.0f && vp.height == 360.0f && vp.max_depth == 1.0f);
    const auto sc = pod<GpuSetScissorRectPayload>(p[6]);
    CHECK(sc.x == 8 && sc.y == 8 && sc.width == 1264 && sc.height == 704);
    CHECK(pod<GpuSetStencilReferencePayload>(p[7]).reference == 0x80);

    // Bind group entries in binding order: root constant ring (binding 0), static sampler (binding 2),
    // then the bind group itself and the constants upload.
    const auto ring = pod<GpuCreateBufferPayload>(p[8]);
    CHECK(ring.id == 1003 && ring.byte_size == (1u << 20));
    const auto smp = pod<GpuCreateSampler2Payload>(p[9]);
    CHECK(smp.id == 1004 && smp.address_u == static_cast<uint8_t>(GpuAddressMode::Repeat) && smp.min_filter == 1);
    CHECK(ring.usage_flags == (GpuBufferUsage::Uniform | GpuBufferUsage::CopyDst));
    const auto bg = pod<GpuCreateBindGroup2Payload>(p[10]);
    CHECK(bg.id == 1005 && bg.layout_id == 1000 && bg.entry_count == 3);
    const auto b0 = pod<GpuBindGroupEntry2>(p[10], sizeof(bg));
    const auto b1 = pod<GpuBindGroupEntry2>(p[10], sizeof(bg) + sizeof(GpuBindGroupEntry2));
    const auto b2 = pod<GpuBindGroupEntry2>(p[10], sizeof(bg) + 2 * sizeof(GpuBindGroupEntry2));
    CHECK(b0.binding == 0 && b0.resource_type == static_cast<uint32_t>(GpuBindResourceType::Buffer));
    CHECK(b0.resource_id == 1003 && b0.offset == 0 && b0.size == 256);  // the whole ring slot
    CHECK(b1.binding == 1 && b1.resource_type == static_cast<uint32_t>(GpuBindResourceType::TextureView));
    CHECK(b1.resource_id == Scene::kAlbedo);
    CHECK(b1.view.format == static_cast<uint8_t>(pso::WgpuTextureFormat::Rgba8Unorm));
    CHECK(b1.view.dimension == static_cast<uint8_t>(GpuTextureViewDimension::Dim2D));
    CHECK(b1.view.mip_level_count == 0 && b1.view.array_layer_count == 1);
    CHECK(b2.binding == 2 && b2.resource_type == static_cast<uint32_t>(GpuBindResourceType::Sampler));
    CHECK(b2.resource_id == 1004);
    const auto wb = pod<GpuWriteBufferPayload>(p[11]);
    CHECK(wb.id == 1003 && wb.dst_offset == 0 && wb.data_len == 256);  // zero past the values
    CHECK(pod<uint32_t>(p[11], sizeof(wb) + 4) == 0x3f000000u);
    const auto sbg = pod<GpuSetBindGroup2Payload>(p[12]);
    CHECK(sbg.group_index == 0 && sbg.bind_group_id == 1005 && sbg.dynamic_offset_count == 1);
    CHECK(pod<uint32_t>(p[12], sizeof(sbg)) == 0);
    const auto svb = pod<GpuSetVertexBufferPayload>(p[13]);
    CHECK(svb.slot == 0 && svb.buffer_id == Scene::kVertexBuffer);
    CHECK(pod<GpuDrawPayload>(p[14]).vertex_count == 3);

    // Pass 2: offscreen target, depth-less pipeline variant 1006, constants re-uploaded at 256.
    const auto pass2 = pod<GpuBeginRenderPassPayload>(p[16]);
    CHECK(pass2.color[0].texture_id == Scene::kOffscreen);
    CHECK(pass2.color[0].load_op == static_cast<uint8_t>(GpuLoadOp::Clear));
    CHECK(pass2.depth_stencil.texture_id == 0);
    const auto rp2 = pod<GpuCreateRenderPipeline2Payload>(p[17]);
    CHECK(rp2.id == 1006 && rp2.pipeline_layout_id == 1001 && rp2.key.depth_stencil.format == 0);
    CHECK(pod<GpuSetPipelinePayload>(p[18]).pipeline_id == 1006);
    const auto vp2 = pod<GpuSetViewportPayload>(p[19]);
    CHECK(vp2.width == 640.0f);
    const auto sc2 = pod<GpuSetScissorRectPayload>(p[20]);
    CHECK(sc2.width == 632 && sc2.height == 352);  // clipped to the 640x360 target
    CHECK(pod<GpuSetStencilReferencePayload>(p[21]).reference == 0x80);  // re-emitted after pass begin
    const auto wb2 = pod<GpuWriteBufferPayload>(p[22]);
    CHECK(wb2.dst_offset == 256);
    const auto sbg2 = pod<GpuSetBindGroup2Payload>(p[23]);
    CHECK(sbg2.bind_group_id == 1005 && pod<uint32_t>(p[23], sizeof(sbg2)) == 256);
    const auto draw2 = pod<GpuDrawPayload>(p[25]);
    CHECK(draw2.vertex_count == 6 && draw2.instance_count == 2);

    CHECK(list.stats().draws == 2 && list.stats().render_passes == 2);
    CHECK(list.stats().pipelines_created == 2 && list.stats().bind_groups_created == 1);
    CHECK(list.stats().root_constant_uploads == 2);
    CHECK((list.diagnostics() & bridge::BridgeDiag::UsageConflict) == 0);
    CHECK(list.usage_conflicts().empty());
    CHECK(!list.is_open());

    // Byte-identical fixture shared with the JS and Python tests.
    const std::string hex = to_hex(enc.data(), enc.size());
    const std::string path = std::string(PG6_TEST_FIXTURE_DIR) + "/state_bridge_frame.hex";
    if (const char* update = std::getenv("PG6_UPDATE_FIXTURES"); update != nullptr && update[0] == '1') {
        std::ofstream(path, std::ios::binary) << hex;
    }
    CHECK(read_file(path) == hex);
}

void test_bind_group_rebuild_and_retire() {
    Scene scene;
    gpu::CommandEncoder enc;
    bridge::CommandListBridge list(scene.device, enc);
    scene.bind_root(list);
    const d3d::CpuDescriptorHandle rtv0 = scene.rtv(0);
    list.OMSetRenderTargets(1, &rtv0, false, nullptr);
    list.DrawInstanced(3, 1, 0, 0);
    const size_t mark = enc.size();

    // Same table, same constants, same pass: nothing but the draw.
    list.DrawInstanced(3, 1, 0, 0);
    std::vector<Packet> p = decode(enc, mark);
    CHECK(opcodes(p) == std::vector<GpuOpcode>{GpuOpcode::Draw});

    // Rewriting the descriptor the table points at rebuilds the bind group and retires the old one.
    scene.write_albedo_srv(Scene::kAlbedo2, 0);
    const size_t mark2 = enc.size();
    list.DrawInstanced(3, 1, 0, 0);
    p = decode(enc, mark2);
    CHECK((opcodes(p) == std::vector<GpuOpcode>{GpuOpcode::CreateBindGroup2, GpuOpcode::DestroyResource,
                                                  GpuOpcode::SetBindGroup2, GpuOpcode::Draw}));
    if (p.size() == 4) {
        const auto bg = pod<GpuCreateBindGroup2Payload>(p[0]);
        CHECK(pod<GpuBindGroupEntry2>(p[0], sizeof(bg) + sizeof(GpuBindGroupEntry2)).resource_id == Scene::kAlbedo2);
        const uint32_t retired = pod<GpuDestroyResourcePayload>(p[1]).id;
        CHECK(retired != bg.id && retired != 0);
        CHECK(pod<GpuSetBindGroup2Payload>(p[2]).bind_group_id == bg.id);
    }
    CHECK(list.stats().bind_groups_retired == 1);

    // Moving the table to another heap slot with a null SRV binds a null resource.
    d3d::ViewDesc null_srv;
    null_srv.format = 28;
    null_srv.dimension = d3d::ViewDimension::Texture2D;
    CHECK(scene.heaps.create_shader_resource_view(0, &null_srv, scene.srv_heap->cpu_handle(3)) == d3d::Status::Ok);
    CHECK(list.SetGraphicsRootDescriptorTable(0, scene.srv_heap->gpu_handle(3)) == d3d::Status::Ok);
    const size_t mark3 = enc.size();
    list.DrawInstanced(3, 1, 0, 0);
    p = decode(enc, mark3);
    CHECK(!p.empty() && p[0].opcode == GpuOpcode::CreateBindGroup2);
    if (!p.empty()) {
        const auto bg = pod<GpuCreateBindGroup2Payload>(p[0]);
        const auto e1 = pod<GpuBindGroupEntry2>(p[0], sizeof(bg) + sizeof(GpuBindGroupEntry2));
        CHECK(e1.resource_id == 0 && e1.resource_type == static_cast<uint32_t>(GpuBindResourceType::TextureView));
    }
    CHECK((list.diagnostics() & bridge::BridgeDiag::NullBinding) != 0);
    list.Close();
}

// G66 (#691): the title releases a texture and leaves its descriptor in the heap. The descriptor
// words do not change, so the cached bind group stayed a hit and kept binding the destroyed texture:
// WebGPU then rejects the whole submit (show1 capture: black frame 2216). The hit is rebuilt with
// the destroyed texture bound null, and the old group is retired.
void test_cached_bind_group_drops_destroyed_texture() {
    Scene scene;
    gpu::CommandEncoder enc;
    bridge::CommandListBridge list(scene.device, enc);
    scene.bind_root(list);
    const d3d::CpuDescriptorHandle rtv0 = scene.rtv(0);
    list.OMSetRenderTargets(1, &rtv0, false, nullptr);
    list.DrawInstanced(3, 1, 0, 0);
    list.Close();

    // Another texture's destruction leaves the cached group alone.
    scene.device.unregister_resource(Scene::kAlbedo2);
    CHECK(scene.device.retired(Scene::kAlbedo2));
    const auto record = [&] {
        const size_t mark = enc.size();
        list.Reset(nullptr);
        scene.bind_root(list);
        list.OMSetRenderTargets(1, &rtv0, false, nullptr);
        list.DrawInstanced(3, 1, 0, 0);
        list.Close();
        return decode(enc, mark);
    };
    std::vector<Packet> p = record();
    CHECK(std::none_of(p.begin(), p.end(), [](const Packet& x) { return x.opcode == GpuOpcode::CreateBindGroup2; }));
    CHECK(list.stats().bind_groups_resource_retired == 0);

    // Destroying the bound texture: the next hit builds a group with it bound null.
    scene.device.unregister_resource(Scene::kAlbedo);
    p = record();
    const auto created = std::find_if(p.begin(), p.end(),
                                      [](const Packet& x) { return x.opcode == GpuOpcode::CreateBindGroup2; });
    CHECK(created != p.end());
    if (created != p.end()) {
        const auto bg = pod<GpuCreateBindGroup2Payload>(*created);
        const auto e1 = pod<GpuBindGroupEntry2>(*created, sizeof(bg) + sizeof(GpuBindGroupEntry2));
        CHECK(e1.resource_id == 0 && e1.resource_type == static_cast<uint32_t>(GpuBindResourceType::TextureView));
    }
    CHECK(std::any_of(p.begin(), p.end(), [](const Packet& x) { return x.opcode == GpuOpcode::DestroyResource; }));
    CHECK(list.stats().bind_groups_resource_retired == 1);
    CHECK(list.stats().retired_resources_nulled >= 1);

    // The rebuilt group is cached again; a texture registered under the id again is bound again.
    p = record();
    CHECK(std::none_of(p.begin(), p.end(), [](const Packet& x) { return x.opcode == GpuOpcode::CreateBindGroup2; }));
    scene.device.register_texture(Scene::kAlbedo, {pso::DxgiFormat::R8G8B8A8_UNORM, 256, 256, 1, 9, 1,
                                                   GpuTextureDimension::Dim2D, false});
    CHECK(!scene.device.retired(Scene::kAlbedo));
}

// G66 partly-held mip chains: the title streams only a texture's small levels. Sampled views start
// at the first held level (and at the view's min-LOD clamp), cached bind groups follow changes of
// the held levels, and each clamped binding is counted.
void test_partly_held_mip_chain_clamps_views() {
    // The clamp itself.
    uint16_t base = 0, count = 0;
    {
        // Host texture lacking the top 2 of 9 levels: guest level N is host level N - 2.
        uint16_t b = 0, c = 0;
        bridge::shift_view_to_host_mips(9, 2, &b, &c);
        CHECK(b == 0 && c == 0);  // whole chain stays whole
        b = 3; c = 0;
        bridge::shift_view_to_host_mips(9, 2, &b, &c);
        CHECK(b == 1 && c == 0);
        b = 2; c = 4;
        bridge::shift_view_to_host_mips(9, 2, &b, &c);
        CHECK(b == 0 && c == 4);
        b = 0; c = 4;  // levels 0..3: raised to 2..3
        bridge::shift_view_to_host_mips(9, 2, &b, &c);
        CHECK(b == 0 && c == 2);
        b = 0; c = 1;  // only dropped level: the first held level stays
        bridge::shift_view_to_host_mips(9, 2, &b, &c);
        CHECK(b == 0 && c == 1);
        b = 1; c = 0;
        bridge::shift_view_to_host_mips(9, 0, &b, &c);
        CHECK(b == 1 && c == 0);  // no offset: unchanged
    }
    CHECK(bridge::clamp_view_to_resident_mips(9, 3, 0.0f, &base, &count) && base == 3 && count == 0);
    base = 1, count = 4;  // levels 1..4
    CHECK(bridge::clamp_view_to_resident_mips(9, 3, 0.0f, &base, &count) && base == 3 && count == 2);
    base = 1, count = 2;  // levels 1..2, nothing held: the view's last level stays
    CHECK(bridge::clamp_view_to_resident_mips(9, 6, 0.0f, &base, &count) && base == 2 && count == 1);
    base = 4, count = 0;
    CHECK(!bridge::clamp_view_to_resident_mips(9, 3, 0.0f, &base, &count) && base == 4 && count == 0);
    base = 0, count = 0;
    CHECK(bridge::clamp_view_to_resident_mips(9, 0, 2.5f, &base, &count) && base == 2 && count == 0);
    base = 0, count = 0;
    CHECK(bridge::clamp_view_to_resident_mips(9, 1, 20.0f, &base, &count) && base == 8);

    // Held levels from per-subresource upload marks (mip + slice * mips).
    const uint64_t small_held[] = {0, 0, 0, 7, 7, 7};
    CHECK(bridge::held_mip_floor(small_held, 6, 1) == 3);
    const uint64_t all_held[] = {1, 1, 1};
    CHECK(bridge::held_mip_floor(all_held, 3, 1) == 0);
    const uint64_t top_only[] = {1, 0, 0};  // smaller levels filled on the GPU: not clamped
    CHECK(bridge::held_mip_floor(top_only, 3, 1) == 0);
    CHECK(bridge::held_mip_floor(small_held, 6, 1, /*streamable=*/false) == 0);  // black mips of a plain texture
    const uint64_t two_slices[] = {0, 0, 1, 0, 1, 0};  // slice 0 holds level 2, slice 1 level 1
    CHECK(bridge::held_mip_floor(two_slices, 3, 2) == 1);

    Scene scene;
    CHECK(!scene.device.set_resident_mip_floor(9999, 2));
    gpu::CommandEncoder enc;
    bridge::CommandListBridge list(scene.device, enc);
    const d3d::CpuDescriptorHandle rtv0 = scene.rtv(0);
    const auto draw = [&] {
        const size_t mark = enc.size();
        scene.bind_root(list);
        list.OMSetRenderTargets(1, &rtv0, false, nullptr);
        list.DrawInstanced(3, 1, 0, 0);
        return decode(enc, mark);
    };
    const auto texture_entry = [](const std::vector<Packet>& p, GpuBindGroupEntry2* out) {
        for (const Packet& x : p) {
            if (x.opcode == GpuOpcode::CreateBindGroup2) {
                *out = pod<GpuBindGroupEntry2>(x, sizeof(GpuCreateBindGroup2Payload) + sizeof(GpuBindGroupEntry2));
                return true;
            }
        }
        return false;
    };
    const auto destroys = [](const std::vector<Packet>& p) {
        return std::count_if(p.begin(), p.end(), [](const Packet& x) { return x.opcode == GpuOpcode::DestroyResource; });
    };

    // Levels 3.. held: the bound view starts at level 3.
    CHECK(scene.device.set_resident_mip_floor(Scene::kAlbedo, 3));
    std::vector<Packet> p = draw();
    GpuBindGroupEntry2 e{};
    CHECK(texture_entry(p, &e));
    CHECK(e.resource_id == Scene::kAlbedo && e.view.base_mip_level == 3 && e.view.mip_level_count == 0);
    CHECK(list.stats().mip_clamped_binds == 1);

    // Unchanged held levels: the cached group is bound again.
    p = draw();
    CHECK(!texture_entry(p, &e) && destroys(p) == 0);

    // Another texture's floor (limited to its single level) changes nothing.
    CHECK(scene.device.set_resident_mip_floor(Scene::kAlbedo2, 0));
    p = draw();
    CHECK(!texture_entry(p, &e));

    // More levels streamed in: the cached group is re-created over the new first held level.
    CHECK(scene.device.set_resident_mip_floor(Scene::kAlbedo, 1));
    p = draw();
    CHECK(texture_entry(p, &e) && e.view.base_mip_level == 1 && destroys(p) == 1);
    CHECK(list.stats().bind_groups_residency_changed == 1 && list.stats().mip_clamped_binds == 2);

    // Registering the same description again keeps the held levels.
    scene.device.register_texture(Scene::kAlbedo, {pso::DxgiFormat::R8G8B8A8_UNORM, 256, 256, 1, 9, 1,
                                                   GpuTextureDimension::Dim2D, false});
    bridge::TextureInfo info{};
    CHECK(scene.device.texture_info(Scene::kAlbedo, &info) && info.resident_mip_floor == 1);

    // The whole chain held: unclamped again.
    CHECK(scene.device.set_resident_mip_floor(Scene::kAlbedo, 0));
    p = draw();
    CHECK(texture_entry(p, &e) && e.view.base_mip_level == 0 && e.view.mip_level_count == 0);
    CHECK(list.stats().bind_groups_residency_changed == 2 && list.stats().mip_clamped_binds == 2);

    // The view's own min-LOD clamp (ResourceMinLODClamp) clamps it too.
    d3d::ViewDesc srv;
    srv.format = 28;
    srv.dimension = d3d::ViewDimension::Texture2D;
    srv.min_lod_clamp = 4.25f;
    CHECK(scene.heaps.create_shader_resource_view(Scene::kAlbedo, &srv, scene.srv_heap->cpu_handle(0)) ==
          d3d::Status::Ok);
    p = draw();
    CHECK(texture_entry(p, &e) && e.view.base_mip_level == 4 && e.view.mip_level_count == 0);
    CHECK(list.stats().mip_clamped_binds == 3);

    // Texture size cap: the host texture was re-created without its top 2 levels. The view (clamped
    // to guest level 4 by min-LOD) starts at host level 2; the cached group is re-created; writes
    // of dropped levels are refused and the others shift.
    CHECK(!scene.device.set_host_mip_offset(9999, 2));
    uint32_t m = 1;
    CHECK(scene.device.translate_host_mip(Scene::kAlbedo, &m) && m == 1);
    CHECK(scene.device.set_host_mip_offset(Scene::kAlbedo, 2));
    m = 1;
    CHECK(!scene.device.translate_host_mip(Scene::kAlbedo, &m));
    m = 5;
    CHECK(scene.device.translate_host_mip(Scene::kAlbedo, &m) && m == 3);
    p = draw();
    CHECK(texture_entry(p, &e) && e.view.base_mip_level == 2 && e.view.mip_level_count == 0);
    CHECK(scene.device.texture_info(Scene::kAlbedo, &info) && info.host_mip_offset == 2 && info.mip_levels == 9);
    // Registering the same description again keeps the offset.
    scene.device.register_texture(Scene::kAlbedo, {pso::DxgiFormat::R8G8B8A8_UNORM, 256, 256, 1, 9, 1,
                                                   GpuTextureDimension::Dim2D, false});
    CHECK(scene.device.texture_info(Scene::kAlbedo, &info) && info.host_mip_offset == 2);
    CHECK(scene.device.set_host_mip_offset(Scene::kAlbedo, 0));
    m = 1;
    CHECK(scene.device.translate_host_mip(Scene::kAlbedo, &m) && m == 1);
    list.Close();
}

void test_usage_conflict_is_diagnosed() {
    Scene scene;
    gpu::CommandEncoder enc;
    bridge::CommandListBridge list(scene.device, enc);
    scene.bind_root(list);
    // Sample the offscreen texture while rendering into it.
    scene.write_albedo_srv(Scene::kOffscreen, 0);
    const d3d::CpuDescriptorHandle rtv1 = scene.rtv(1);
    list.OMSetRenderTargets(1, &rtv1, false, nullptr);
    list.DrawInstanced(3, 1, 0, 0);
    CHECK(list.usage_conflicts().empty());  // checked when the pass (usage scope) ends
    list.Close();
    CHECK((list.diagnostics() & bridge::BridgeDiag::UsageConflict) != 0);
    CHECK(list.usage_conflicts().size() == 1);
    if (!list.usage_conflicts().empty()) {
        const d3d::UsageConflict& c = list.usage_conflicts()[0];
        CHECK(c.resource_id == Scene::kOffscreen);
        CHECK(c.first == d3d::ResourceUsage::RenderTarget && c.second == d3d::ResourceUsage::Sampled);
    }
    CHECK(list.stats().usage_conflicts == 1);
}

void test_barrier_flushes_pending_clears() {
    Scene scene;
    CHECK(scene.states.register_resource(Scene::kOffscreen, d3d::ResourceStateDesc{1, d3d::ResourceStates::RenderTarget}) ==
          d3d::Status::Ok);
    gpu::CommandEncoder enc;
    bridge::CommandListBridge list(scene.device, enc);
    const d3d::CpuDescriptorHandle rtv1 = scene.rtv(1);
    const float color[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    list.ClearRenderTargetView(rtv1, color);
    CHECK(enc.size() == 0);
    const d3d::ResourceBarrier barrier = d3d::ResourceBarrier::transition(
        Scene::kOffscreen, d3d::ResourceStates::RenderTarget, d3d::ResourceStates::PixelShaderResource);
    list.ResourceBarrier(1, &barrier);
    const std::vector<Packet> p = decode(enc);
    CHECK((opcodes(p) == std::vector<GpuOpcode>{GpuOpcode::BeginRenderPass, GpuOpcode::EndRenderPass}));
    if (!p.empty()) {
        const auto pass = pod<GpuBeginRenderPassPayload>(p[0]);
        CHECK(pass.color[0].texture_id == Scene::kOffscreen);
        CHECK(pass.color[0].load_op == static_cast<uint8_t>(GpuLoadOp::Clear));
        CHECK(pass.color[0].clear_value[0] == 1.0f);
    }
    CHECK(list.resource_states().state(Scene::kOffscreen, 0) == d3d::ResourceStates::PixelShaderResource);
    CHECK(list.resource_states().diagnostics().empty());
    list.Close();
    CHECK(decode(enc).size() == 2);
}

void test_partial_clear_is_reported() {
    Scene scene;
    gpu::CommandEncoder enc;
    bridge::CommandListBridge list(scene.device, enc);
    const d3d::CpuDescriptorHandle rtv1 = scene.rtv(1);
    const float color[4] = {0.0f, 1.0f, 0.0f, 1.0f};
    const pso::D3D12Rect rect{0, 0, 32, 32};
    list.ClearRenderTargetView(rtv1, color, 1, &rect);
    list.Close();
    CHECK((list.diagnostics() & bridge::BridgeDiag::ClearFallbackDropped) != 0);
    CHECK(list.stats().clear_fallbacks_dropped == 1);
    // The fallback still opened (and closed) a pass on the view.
    CHECK((opcodes(decode(enc)) == std::vector<GpuOpcode>{GpuOpcode::BeginRenderPass, GpuOpcode::EndRenderPass}));
}

void test_compute_dispatch_packets() {
    Scene scene(5000);
    constexpr uint64_t kCs = 0xC5;
    constexpr uint32_t kCsModule = 3;
    constexpr uint32_t kParams = 301;
    scene.device.register_shader_module(kCs, kCsModule);
    scene.device.register_buffer(kParams, 0x200000, 1024);

    d3d::RootSignatureDesc rs;
    d3d::RootParameterDesc cbv;
    cbv.parameter_type = d3d::RootParameterType::Cbv;
    rs.parameters.push_back(cbv);
    d3d::RootSignature signature;
    CHECK(d3d::RootSignature::build(rs, &signature).ok());
    bridge::PipelineLayoutDesc layout;
    bridge::GroupLayoutDesc group;
    group.bindings.push_back({layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer,
                                           static_cast<uint8_t>(GpuBufferBindingType::Uniform)),
                              join(0, d3d::DescriptorRangeType::Cbv, 0)});
    layout.groups.push_back(group);
    scene.device.register_pipeline_layout(kCs, 0, layout);
    pso::D3D12ComputePsoDesc desc;
    desc.cs = kCs;
    desc.root_signature_id = 9;
    auto cpso = scene.device.create_compute_pipeline_state(desc, &signature);

    gpu::CommandEncoder enc;
    bridge::CommandListBridge list(scene.device, enc);
    CHECK(list.SetComputeRootSignature(&signature) == d3d::Status::Ok);
    CHECK(list.SetComputeRootConstantBufferView(0, 0x200100) == d3d::Status::Ok);
    list.SetComputePipelineState(cpso.get());
    list.Dispatch(4, 2, 1);
    list.Dispatch(1, 1, 1);
    // A render pass in between invalidates the compute pass state.
    scene.bind_root(list);
    const d3d::CpuDescriptorHandle rtv1 = scene.rtv(1);
    list.OMSetRenderTargets(1, &rtv1, false, nullptr);
    list.DrawInstanced(3, 1, 0, 0);
    list.Dispatch(2, 2, 2);
    list.Close();

    const std::vector<Packet> p = decode(enc);
    using O = GpuOpcode;
    CHECK(p.size() >= 8);
    if (p.size() < 8) {
        return;
    }
    CHECK(p[0].opcode == O::CreateBindGroupLayout && p[1].opcode == O::CreatePipelineLayout);
    CHECK(p[2].opcode == O::CreateComputePipeline2);
    const auto cp = pod<GpuCreateComputePipeline2Payload>(p[2]);
    CHECK(cp.id == cpso->pipeline_id() && cp.cs_module_id == kCsModule);
    CHECK(cp.pipeline_layout_id == pod<GpuCreatePipelineLayoutPayload>(p[1]).id);
    CHECK(cp.key.cs_shader == kCs && cp.key.root_signature_id == 9);
    CHECK(p[3].opcode == O::SetComputePipeline && pod<GpuSetComputePipelinePayload>(p[3]).pipeline_id == cp.id);
    CHECK(p[4].opcode == O::CreateBindGroup2);
    const auto bg = pod<GpuCreateBindGroup2Payload>(p[4]);
    const auto entry = pod<GpuBindGroupEntry2>(p[4], sizeof(bg));
    CHECK(entry.resource_id == kParams && entry.offset == 0x100 && entry.size == 1024 - 0x100);
    CHECK(p[5].opcode == O::SetBindGroup2 && pod<GpuSetBindGroup2Payload>(p[5]).dynamic_offset_count == 0);
    CHECK(p[6].opcode == O::DispatchCompute && pod<GpuDispatchComputePayload>(p[6]).workgroup_x == 4);
    CHECK(p[7].opcode == O::DispatchCompute);  // same pipeline and bind group: just the dispatch

    // After the render pass: pipeline and bind group are set again before the third dispatch.
    size_t third = p.size();
    for (size_t i = 8; i < p.size(); ++i) {
        if (p[i].opcode == O::DispatchCompute) {
            third = i;
        }
    }
    CHECK(third < p.size() && third >= 2);
    if (third < p.size() && third >= 2) {
        CHECK(p[third - 2].opcode == O::SetComputePipeline);
        CHECK(p[third - 1].opcode == O::SetBindGroup2);
        CHECK(p[third - 3].opcode == O::EndRenderPass);
    }
    size_t compute_creates = 0;
    for (const Packet& pk : p) {
        compute_creates += pk.opcode == O::CreateComputePipeline2 ? 1 : 0;
    }
    CHECK(compute_creates == 1);
    CHECK(list.stats().dispatches == 3);

    {
        // Another list without the root CBV: the bind group cannot be built and the dispatch is
        // dropped (the pipeline was already created by the first list's stream).
        gpu::CommandEncoder scratch;
        bridge::CommandListBridge early(scene.device, scratch);
        CHECK(early.SetComputeRootSignature(&signature) == d3d::Status::Ok);
        early.SetComputePipelineState(cpso.get());
        early.Dispatch(1, 1, 1);
        CHECK((early.diagnostics() & bridge::BridgeDiag::RootArgumentNotSet) != 0);
        CHECK(early.stats().dispatches == 0);
        CHECK((opcodes(decode(scratch)) == std::vector<GpuOpcode>{GpuOpcode::SetComputePipeline}));
        early.SetComputePipelineState(nullptr);
        early.Dispatch(1, 1, 1);
        CHECK((early.diagnostics() & bridge::BridgeDiag::NoComputePipeline) != 0);
    }

}

// G66 ExecuteIndirect: a clamp compute pass (count buffer, out-of-range arguments) writes compacted
// records into the bridge's indirect ring, which one DrawIndirect / DrawIndexedIndirect /
// DispatchIndirect per command reads.
size_t find_op(const std::vector<Packet>& p, GpuOpcode op, size_t from = 0) {
    for (size_t i = from; i < p.size(); ++i) {
        if (p[i].opcode == op) {
            return i;
        }
    }
    return p.size();
}

size_t count_op(const std::vector<Packet>& p, GpuOpcode op) {
    return static_cast<size_t>(std::count_if(p.begin(), p.end(), [&](const Packet& k) { return k.opcode == op; }));
}

// The clamp pass's params (the last 32-byte WriteBuffer before `before`).
std::array<uint32_t, 8> indirect_params(const std::vector<Packet>& p, size_t before) {
    std::array<uint32_t, 8> v{};
    for (size_t i = std::min(before, p.size()); i-- > 0;) {
        if (p[i].opcode == GpuOpcode::WriteBuffer && pod<GpuWriteBufferPayload>(p[i]).data_len == 32) {
            std::memcpy(v.data(), p[i].payload.data() + sizeof(GpuWriteBufferPayload), sizeof(v));
            break;
        }
    }
    return v;
}

// Backlog 41: clamps of consecutive indirect draws are batched ahead of the draws.
void test_indirect_clamp_batch_planner() {
    using bridge::IndirectClampBatch;
    gpu::CommandEncoder enc;
    enc.nop();
    IndirectClampBatch b;
    CHECK(!b.open() && b.try_join(1, 2, 10) == IndirectClampBatch::kNoSlot);
    b.begin(enc.size(), 120, enc.size(), 0);
    CHECK(b.open());
    // Draw-ish packets and writes to other buffers keep the run joinable; slots go out in order.
    enc.write_buffer(900, 0, "abcd", 4);
    enc.set_compute_pipeline(5);
    enc.dispatch_compute(1, 1, 1);  // a dispatch may write any storage buffer: the run ends
    b.scan(enc.data(), enc.size());
    CHECK(!b.open());

    gpu::CommandEncoder e2;
    b.begin(0, 120, 0, 0);
    e2.write_buffer(900, 0, "abcd", 4);
    e2.draw(3, 1, 0, 0);
    b.scan(e2.data(), e2.size());
    CHECK(b.open());
    CHECK(b.try_join(1, 2, 256) == 0);
    CHECK(b.try_join(3, 0, 512) == 120);
    // A buffer written since the run started cannot be read earlier: no slot, run stays open.
    e2.write_buffer(7, 16, "wxyz", 4);
    b.scan(e2.data(), e2.size());
    CHECK(b.try_join(7, 0, 768) == IndirectClampBatch::kNoSlot);
    CHECK(b.try_join(8, 7, 768) == IndirectClampBatch::kNoSlot);  // count buffer too
    CHECK(b.open() && b.try_join(9, 0, 768) == 240);
    // A reused (wrapped) ring window ends the run.
    CHECK(b.try_join(10, 0, 0) == IndirectClampBatch::kNoSlot && !b.open());
    // Capacity: the run's inline clamp plus kIndirectClampBatchSize - 1 slots.
    b.begin(0, 120, 0, 0);
    uint32_t joined = 0;
    for (uint32_t i = 0; i < 20; ++i) {
        if (b.try_join(100 + i, 0, 256 * (i + 1)) != IndirectClampBatch::kNoSlot) {
            ++joined;
        }
    }
    CHECK(joined == bridge::kIndirectClampBatchSize - 1);
    // Unknown packets are ordering points.
    b.begin(0, 120, 0, 0);
    gpu::CommandEncoder e3;
    e3.begin_frame();
    b.scan(e3.data(), e3.size());
    CHECK(!b.open());
}

// The second execution's clamp lands in a slot reserved behind the first one, before the first
// draws; a write to its arguments in between keeps it in place.
void test_execute_indirect_clamp_batching() {
    Scene scene(7100);
    constexpr uint32_t kArgsA = 410;
    constexpr uint32_t kArgsB = 411;
    constexpr uint32_t kArgsC = 412;
    scene.device.register_buffer(kArgsA, 0x400000, 4096);
    scene.device.register_buffer(kArgsB, 0x410000, 4096);
    scene.device.register_buffer(kArgsC, 0x420000, 4096);
    bridge::IndirectExecution a;
    a.kind = bridge::IndirectKind::Draw;
    a.max_count = 2;
    a.stride = 16;
    a.args_buffer = kArgsA;
    bridge::IndirectExecution b = a;
    b.args_buffer = kArgsB;
    b.max_count = 3;
    bridge::IndirectExecution c = a;
    c.args_buffer = kArgsC;
    c.max_count = 4;
    const d3d::CpuDescriptorHandle rtv1 = scene.rtv(1);

    {  // Warm-up list: creates the clamp bind groups (these are not batched; the groups are new).
        gpu::CommandEncoder warm;
        bridge::CommandListBridge list(scene.device, warm);
        scene.bind_root(list);
        list.OMSetRenderTargets(1, &rtv1, false, nullptr);
        CHECK(list.ExecuteIndirect(a) && list.ExecuteIndirect(b) && list.ExecuteIndirect(c));
        CHECK(list.stats().indirect_clamps_batched == 0);
        CHECK(count_op(decode(warm), GpuOpcode::DrawIndirect) == 9);
        list.Close();
    }

    gpu::CommandEncoder enc;
    bridge::CommandListBridge list(scene.device, enc);
    scene.bind_root(list);
    list.OMSetRenderTargets(1, &rtv1, false, nullptr);
    CHECK(list.ExecuteIndirect(a));
    const size_t after_a = enc.size();
    list.DrawInstanced(3, 1, 0, 0);
    CHECK(list.ExecuteIndirect(b));  // joins: its clamp is patched in behind a's
    CHECK(list.stats().indirect_clamps_batched == 1);
    {
        const std::vector<Packet> p = decode(enc);
        // One compute pass worth of dispatches before any draw, for both executions.
        const size_t first_draw = find_op(p, GpuOpcode::DrawIndirect);
        CHECK(first_draw < p.size());
        size_t dispatches_before_draws = 0;
        std::vector<uint32_t> groups;
        for (size_t i = 0; i < first_draw; ++i) {
            if (p[i].opcode == GpuOpcode::DispatchCompute) {
                ++dispatches_before_draws;
                groups.push_back(pod<GpuDispatchComputePayload>(p[i]).workgroup_x);
            }
        }
        CHECK(dispatches_before_draws == 2);  // a's clamp and b's batched clamp
        CHECK(count_op(p, GpuOpcode::DispatchCompute) == 2);  // none between the draws
        CHECK(count_op(p, GpuOpcode::DrawIndirect) == 5);
        CHECK(count_op(p, GpuOpcode::BeginRenderPass) == 1);  // the render pass was not broken for b
        // b's parameters (3 commands) are written before a's first draw.
        bool found_b_params = false;
        for (size_t i = 0; i < first_draw; ++i) {
            if (p[i].opcode == GpuOpcode::WriteBuffer && pod<GpuWriteBufferPayload>(p[i]).data_len == 32) {
                std::array<uint32_t, 8> v{};
                std::memcpy(v.data(), p[i].payload.data() + sizeof(GpuWriteBufferPayload), sizeof(v));
                found_b_params = found_b_params || v[2] == 3;
            }
        }
        CHECK(found_b_params);
        // The bytes the draws were recorded after are untouched: a's clamp stayed at its place.
        CHECK(after_a > 0);
    }
    // An upload to c's arguments after the run started: c's clamp must run after it, inline.
    enc.write_buffer(kArgsC, 0, "abcd", 4);
    const size_t before_c = enc.size();
    CHECK(list.ExecuteIndirect(c));
    CHECK(list.stats().indirect_clamps_batched == 1);
    {
        const std::vector<Packet> p = decode(enc, before_c);
        CHECK(find_op(p, GpuOpcode::DispatchCompute) < find_op(p, GpuOpcode::DrawIndirect));
        CHECK(count_op(p, GpuOpcode::DispatchCompute) == 1);
    }
    list.Close();
}

void test_execute_indirect_packets() {
    Scene scene(7000);
    constexpr uint32_t kArgs = 400;
    constexpr uint32_t kCount = 401;
    constexpr uint32_t kIndex = 402;
    scene.device.register_buffer(kArgs, 0x300000, 4096);
    scene.device.register_buffer(kCount, 0x310000, 256);
    scene.device.register_buffer(kIndex, 0x320000, 1024);

    const std::string wgsl = bridge::indirect_clamp_wgsl();
    CHECK(wgsl.find("@compute @workgroup_size(64)") != std::string::npos);
    CHECK(wgsl.find("fn cs_main") != std::string::npos);
    CHECK(wgsl.find("arrayLength(&cnt)") != std::string::npos);

    gpu::CommandEncoder enc;
    bridge::CommandListBridge list(scene.device, enc);
    scene.bind_root(list);
    const d3d::CpuDescriptorHandle rtv1 = scene.rtv(1);
    list.OMSetRenderTargets(1, &rtv1, false, nullptr);
    list.DrawInstanced(3, 1, 0, 0);
    const size_t mark = enc.size();

    // Draws: 3 commands, 32-byte stride at offset 64, count buffer word 1.
    bridge::IndirectExecution draw;
    draw.kind = bridge::IndirectKind::Draw;
    draw.max_count = 3;
    draw.stride = 32;
    draw.args_buffer = kArgs;
    draw.args_offset = 64;
    draw.count_buffer = kCount;
    draw.count_offset = 4;
    CHECK(list.ExecuteIndirect(draw));
    {
        const std::vector<Packet> p = decode(enc, mark);
        using O = GpuOpcode;
        const size_t end_pass = find_op(p, O::EndRenderPass);
        const size_t module = find_op(p, O::CreateShaderModule);
        const size_t layout = find_op(p, O::CreateBindGroupLayout);
        const size_t pipeline = find_op(p, O::CreateComputePipeline2);
        const size_t ring_create = find_op(p, O::CreateBuffer);
        const size_t group = find_op(p, O::CreateBindGroup2);
        const size_t dispatch = find_op(p, O::DispatchCompute);
        const size_t begin = find_op(p, O::BeginRenderPass, dispatch);
        const size_t first_draw = find_op(p, O::DrawIndirect);
        CHECK(end_pass < module && module < pipeline && pipeline < dispatch && dispatch < begin &&
              begin < first_draw && first_draw < p.size());
        CHECK(layout < p.size() && ring_create < p.size() && group < dispatch);
        if (first_draw >= p.size() || group >= p.size() || ring_create >= p.size() || layout >= p.size()) {
            return;
        }
        const auto layout_head = pod<GpuCreateBindGroupLayoutPayload>(p[layout]);
        CHECK(layout_head.entry_count == 4);
        const auto params_entry = pod<GpuBindGroupLayoutEntry>(p[layout], sizeof(layout_head));
        const auto out_entry = pod<GpuBindGroupLayoutEntry>(p[layout], sizeof(layout_head) + 3 * sizeof(GpuBindGroupLayoutEntry));
        CHECK(params_entry.binding_type == static_cast<uint8_t>(GpuBufferBindingType::Uniform) &&
              params_entry.flags == GpuBindGroupLayoutEntryFlags::HasDynamicOffset);
        CHECK(out_entry.binding_type == static_cast<uint8_t>(GpuBufferBindingType::Storage) &&
              out_entry.flags == GpuBindGroupLayoutEntryFlags::HasDynamicOffset);
        const auto ring = pod<GpuCreateBufferPayload>(p[ring_create]);
        CHECK(ring.usage_flags == (GpuBufferUsage::Storage | GpuBufferUsage::Indirect));
        CHECK(ring.byte_size == bridge::kIndirectRingBytes + bridge::kIndirectWindowBytes);
        const auto bg = pod<GpuCreateBindGroup2Payload>(p[group]);
        CHECK(bg.entry_count == 4 && bg.layout_id == layout_head.id);
        const auto args_entry = pod<GpuBindGroupEntry2>(p[group], sizeof(bg) + sizeof(GpuBindGroupEntry2));
        const auto count_entry = pod<GpuBindGroupEntry2>(p[group], sizeof(bg) + 2 * sizeof(GpuBindGroupEntry2));
        const auto ring_entry = pod<GpuBindGroupEntry2>(p[group], sizeof(bg) + 3 * sizeof(GpuBindGroupEntry2));
        CHECK(args_entry.resource_id == kArgs && count_entry.resource_id == kCount && ring_entry.resource_id == ring.id);
        CHECK(ring_entry.size == bridge::kIndirectWindowBytes);
        const auto set = pod<GpuSetBindGroup2Payload>(p[dispatch - 1]);
        CHECK(p[dispatch - 1].opcode == O::SetBindGroup2 && set.bind_group_id == bg.id && set.dynamic_offset_count == 2);
        const uint32_t window = pod<uint32_t>(p[dispatch - 1], sizeof(set) + 4);
        CHECK(pod<GpuDispatchComputePayload>(p[dispatch]).workgroup_x == 1);
        const std::array<uint32_t, 8> params = indirect_params(p, dispatch);
        CHECK((params == std::array<uint32_t, 8>{16, 8, 3, 1, 1, 0, 0xFFFFFFFFu, bridge::kMaxIndirectWorkgroups}));
        CHECK(count_op(p, O::DrawIndirect) == 3);
        for (uint32_t i = 0; i < 3; ++i) {
            const auto d = pod<GpuIndirectPayload>(p[first_draw + i]);
            CHECK(p[first_draw + i].opcode == O::DrawIndirect && d.buffer_id == ring.id &&
                  d.offset == uint64_t{window} + 16u * i);
        }
    }
    CHECK(list.stats().indirect_executions == 1 && list.stats().indirect_commands == 3);

    // Indexed draws without a count buffer: the bind group (args, none) is new, the pipeline is not;
    // the index count is clamped to the bound index buffer (600 bytes of uint16 = 300 indices).
    const gpu::d3d12_xs::D3D12IndexBufferView ib{kIndex, 0, 600, GpuIndexFormat::Uint16};
    list.IASetIndexBuffer(&ib);
    const size_t mark2 = enc.size();
    bridge::IndirectExecution indexed;
    indexed.kind = bridge::IndirectKind::DrawIndexed;
    indexed.max_count = 2;
    indexed.stride = 20;
    indexed.args_buffer = kArgs;
    CHECK(list.ExecuteIndirect(indexed));
    {
        const std::vector<Packet> p = decode(enc, mark2);
        CHECK(count_op(p, GpuOpcode::CreateShaderModule) == 0 && count_op(p, GpuOpcode::CreateComputePipeline2) == 0);
        const size_t group = find_op(p, GpuOpcode::CreateBindGroup2);
        CHECK(group < p.size());
        if (group < p.size()) {
            const auto bg = pod<GpuCreateBindGroup2Payload>(p[group]);
            const auto count_entry = pod<GpuBindGroupEntry2>(p[group], sizeof(bg) + 2 * sizeof(GpuBindGroupEntry2));
            CHECK(count_entry.resource_id == kArgs);  // no count buffer: the args buffer stands in
        }
        const size_t dispatch = find_op(p, GpuOpcode::DispatchCompute);
        const std::array<uint32_t, 8> params = indirect_params(p, dispatch);
        CHECK(params[0] == 0 && params[1] == 5 && params[2] == 2 && params[4] == 0 && params[5] == 1 && params[6] == 300);
        CHECK(find_op(p, GpuOpcode::SetIndexBuffer) < find_op(p, GpuOpcode::DrawIndexedIndirect));
        const size_t first = find_op(p, GpuOpcode::DrawIndexedIndirect);
        CHECK(count_op(p, GpuOpcode::DrawIndexedIndirect) == 2 && first + 1 < p.size());
        if (first + 1 < p.size()) {
            CHECK(pod<GpuIndirectPayload>(p[first + 1]).offset - pod<GpuIndirectPayload>(p[first]).offset == 20);
        }
    }

    // The same buffers again: cached bind group, a new ring window.
    const size_t mark3 = enc.size();
    CHECK(list.ExecuteIndirect(draw));
    {
        const std::vector<Packet> p = decode(enc, mark3);
        CHECK(count_op(p, GpuOpcode::CreateBindGroup2) <= 1);  // only the re-begun pass's own groups
        CHECK(count_op(p, GpuOpcode::DrawIndirect) == 3);
    }

    // Over kMaxIndirectCommands: truncated and counted.
    bridge::IndirectExecution many = draw;
    many.max_count = bridge::kMaxIndirectCommands + 44;
    const size_t mark4 = enc.size();
    CHECK(list.ExecuteIndirect(many));
    {
        // The clamp may have been batched into a slot reserved earlier: look at the whole list.
        const std::vector<Packet> all = decode(enc);
        const std::vector<Packet> p = decode(enc, mark4);
        CHECK(count_op(p, GpuOpcode::DrawIndirect) == bridge::kMaxIndirectCommands);
        size_t clamp_at = all.size();
        for (size_t i = all.size(); i-- > 0;) {
            if (all[i].opcode == GpuOpcode::DispatchCompute) {
                clamp_at = i;
                break;
            }
        }
        CHECK(clamp_at < all.size());
        CHECK(clamp_at < all.size() &&
              pod<GpuDispatchComputePayload>(all[clamp_at]).workgroup_x == bridge::kMaxIndirectCommands / 64);
    }
    CHECK(list.stats().indirect_truncated == 44);

    // Invalid executions are skipped without packets.
    const size_t mark5 = enc.size();
    bridge::IndirectExecution bad = draw;
    bad.stride = 8;  // shorter than a draw record
    CHECK(!list.ExecuteIndirect(bad));
    bad = draw;
    bad.args_buffer = 0;
    CHECK(!list.ExecuteIndirect(bad));
    bad = draw;
    bad.args_offset = 6;  // unaligned
    CHECK(!list.ExecuteIndirect(bad));
    bad = draw;
    bad.kind = bridge::IndirectKind::Dispatch;  // no compute pipeline state
    bad.stride = 12;
    CHECK(!list.ExecuteIndirect(bad));
    CHECK(enc.size() == mark5);
    CHECK(list.stats().indirect_skipped == 4);
    bridge::IndirectExecution none = draw;
    none.max_count = 0;
    CHECK(list.ExecuteIndirect(none) && enc.size() == mark5);  // nothing to draw
    list.Close();

    {
        // Indexed draws without an index buffer are skipped.
        gpu::CommandEncoder scratch;
        bridge::CommandListBridge other(scene.device, scratch);
        scene.bind_root(other);
        other.OMSetRenderTargets(1, &rtv1, false, nullptr);
        CHECK(!other.ExecuteIndirect(indexed));
        CHECK(count_op(decode(scratch), GpuOpcode::DrawIndexedIndirect) == 0);
    }

    // Dispatches: clamped to the per-dimension limit, then the stage's pipeline and bind groups.
    constexpr uint64_t kCs = 0xC6;
    constexpr uint32_t kParams = 403;
    scene.device.register_shader_module(kCs, 4);
    scene.device.register_buffer(kParams, 0x330000, 1024);
    d3d::RootSignatureDesc rs;
    d3d::RootParameterDesc cbv;
    cbv.parameter_type = d3d::RootParameterType::Cbv;
    rs.parameters.push_back(cbv);
    d3d::RootSignature signature;
    CHECK(d3d::RootSignature::build(rs, &signature).ok());
    bridge::PipelineLayoutDesc layout;
    bridge::GroupLayoutDesc group;
    group.bindings.push_back({layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer,
                                           static_cast<uint8_t>(GpuBufferBindingType::Uniform)),
                              join(0, d3d::DescriptorRangeType::Cbv, 0)});
    layout.groups.push_back(group);
    scene.device.register_pipeline_layout(kCs, 0, layout);
    pso::D3D12ComputePsoDesc desc;
    desc.cs = kCs;
    desc.root_signature_id = 11;
    auto cpso = scene.device.create_compute_pipeline_state(desc, &signature);

    gpu::CommandEncoder cenc;
    bridge::CommandListBridge compute(scene.device, cenc);
    CHECK(compute.SetComputeRootSignature(&signature) == d3d::Status::Ok);
    CHECK(compute.SetComputeRootConstantBufferView(0, 0x330000) == d3d::Status::Ok);
    compute.SetComputePipelineState(cpso.get());
    bridge::IndirectExecution dispatch;
    dispatch.kind = bridge::IndirectKind::Dispatch;
    dispatch.max_count = 2;
    dispatch.stride = 12;
    dispatch.args_buffer = kArgs;
    dispatch.args_offset = 128;
    dispatch.count_buffer = kCount;
    CHECK(compute.ExecuteIndirect(dispatch));
    compute.Close();
    {
        const std::vector<Packet> p = decode(cenc);
        const size_t clamp = find_op(p, GpuOpcode::DispatchCompute);
        const size_t first = find_op(p, GpuOpcode::DispatchIndirect);
        CHECK(clamp < first && first + 1 < p.size() && count_op(p, GpuOpcode::DispatchIndirect) == 2);
        if (clamp < first && first < p.size()) {
            CHECK(p[first - 1].opcode == GpuOpcode::SetBindGroup2);
            bool stage_set = false;
            for (size_t i = clamp; i < first; ++i) {
                stage_set |= p[i].opcode == GpuOpcode::SetComputePipeline &&
                             pod<GpuSetComputePipelinePayload>(p[i]).pipeline_id == cpso->pipeline_id();
            }
            CHECK(stage_set);
            const std::array<uint32_t, 8> params = indirect_params(p, clamp);
            CHECK(params[0] == 32 && params[1] == 3 && params[2] == 2 && params[3] == 0 && params[4] == 1 &&
                  params[5] == 2 && params[7] == 65535);
            CHECK(pod<GpuIndirectPayload>(p[first + 1]).offset - pod<GpuIndirectPayload>(p[first]).offset == 12);
        }
    }
    CHECK(compute.stats().indirect_executions == 1 && compute.stats().indirect_commands == 2);

    // A destroyed argument buffer retires its clamp bind groups: the next use builds a new one.
    scene.device.unregister_resource(kCount);
    scene.device.register_buffer(kCount, 0x310000, 256);
    gpu::CommandEncoder renc;
    bridge::CommandListBridge again(scene.device, renc);
    CHECK(again.SetComputeRootSignature(&signature) == d3d::Status::Ok);
    CHECK(again.SetComputeRootConstantBufferView(0, 0x330000) == d3d::Status::Ok);
    again.SetComputePipelineState(cpso.get());
    CHECK(again.ExecuteIndirect(dispatch));
    again.Close();
    {
        const std::vector<Packet> p = decode(renc);
        bool rebuilt = false;
        for (const Packet& k : p) {
            if (k.opcode == GpuOpcode::CreateBindGroup2 && pod<GpuCreateBindGroup2Payload>(k).entry_count == 4) {
                const auto e = pod<GpuBindGroupEntry2>(k, sizeof(GpuCreateBindGroup2Payload) + 2 * sizeof(GpuBindGroupEntry2));
                rebuilt |= e.resource_id == kCount;
            }
        }
        CHECK(rebuilt);
    }
    CHECK(scene.device.deferred_destroy_count() >= 1);
}

// G66 mesh pipelines: the mesh stage runs as compute (groups 0..2 from the graphics root arguments,
// group 3 the runtime's output buffer), then a draw with the runtime vertex shader pulls the
// triangles. DispatchMesh directly and through ExecuteIndirect (clamp kind 3: dispatch + draw pair).
void test_mesh_pipeline_packets() {
    Scene scene(11000);
    constexpr uint32_t kArgs = 500;
    scene.device.register_buffer(kArgs, 0x400000, 4096);

    bridge::MeshOutputLayout out{64, 32, 12};
    CHECK(bridge::valid_mesh_output_layout(out));
    CHECK(!bridge::valid_mesh_output_layout({0, 32, 12}) && !bridge::valid_mesh_output_layout({64, 300, 12}) &&
          !bridge::valid_mesh_output_layout({64, 32, 10}) && !bridge::valid_mesh_output_layout({64, 32, 68}));
    const uint32_t record = 4 + 64 * 12 + 32 * 3;
    CHECK(bridge::mesh_group_record_words(out) == record);
    CHECK(bridge::mesh_max_groups(out) == std::min<uint64_t>(bridge::kMeshOutputBytes / (record * 4u), 65535u));
    CHECK(bridge::mesh_max_groups({256, 256, 64}) == bridge::kMeshOutputBytes / ((4u + 256u * 64u + 768u) * 4u));
    CHECK(bridge::indirect_output_bytes(bridge::IndirectKind::DispatchMesh) == 28);
    CHECK(bridge::mesh_compute_pack_key(bridge::mesh_compute_pack_key(0x1234)) == 0x1234);
    CHECK(bridge::mesh_draw_vertex_identity(out) != bridge::mesh_draw_vertex_identity({64, 32, 8}));
    const std::string vs = bridge::mesh_draw_vertex_wgsl(out);
    CHECK(vs.find("fn vs_main(@builtin(vertex_index)") != std::string::npos);
    CHECK(vs.find("@group(3) @binding(0) var<storage, read> ms_out: array<u32>;") != std::string::npos);
    CHECK(vs.find("@location(1) a1: vec4<f32>") != std::string::npos && vs.find("@location(2)") == std::string::npos);
    CHECK(vs.find("const RECORD_WORDS: u32 = " + std::to_string(record) + "u;") != std::string::npos);
    CHECK(vs.find("const INDEX_BASE: u32 = 772u;") != std::string::npos);
    const std::string clamp = bridge::indirect_clamp_wgsl();
    CHECK(clamp.find("p.kind == 3u") != std::string::npos && clamp.find("groups * p.index_limit") != std::string::npos);

    // The mesh stage: module 6, one group with the root constants (compute visibility).
    constexpr uint64_t kMs = 0xD7;
    const uint64_t stage_identity = bridge::mesh_compute_pack_key(kMs);
    scene.device.register_shader_module(stage_identity, 6);
    bridge::PipelineLayoutDesc stage_layout;
    bridge::GroupLayoutDesc stage_group;
    stage_group.bindings.push_back({layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer,
                                                 static_cast<uint8_t>(GpuBufferBindingType::Uniform)),
                                    join(0, d3d::DescriptorRangeType::Cbv, 0, true)});
    stage_layout.groups.push_back(stage_group);
    scene.device.register_pipeline_layout(stage_identity, 0, stage_layout);
    // The draw: the runtime vertex shader (module 7) and the scene's pixel shader and layout.
    const uint64_t draw_vs = bridge::mesh_draw_vertex_identity(out);
    scene.device.register_shader_module(draw_vs, 7);
    bridge::PipelineLayoutDesc draw_layout;
    bridge::GroupLayoutDesc draw_group;
    draw_group.bindings.push_back({layout_entry(0, GpuShaderStageBits::Vertex | GpuShaderStageBits::Fragment,
                                                GpuBindingLayoutType::Buffer,
                                                static_cast<uint8_t>(GpuBufferBindingType::Uniform)),
                                   join(0, d3d::DescriptorRangeType::Cbv, 0, true)});
    draw_layout.groups.push_back(draw_group);
    scene.device.register_pipeline_layout(draw_vs, Scene::kPs, draw_layout);
    pso::D3D12GraphicsPsoDesc desc = Scene::pso_desc();
    desc.vs = draw_vs;
    desc.pInputElementDescs = nullptr;
    desc.NumInputElements = 0;
    auto mesh = scene.device.create_mesh_pipeline_state(desc, stage_identity, out, &scene.root_signature);
    CHECK(mesh != nullptr && mesh->diagnostics() == 0);
    if (mesh == nullptr || mesh->diagnostics() != 0) {
        return;
    }

    gpu::CommandEncoder enc;
    bridge::CommandListBridge list(scene.device, enc);
    scene.bind_root(list);
    const d3d::CpuDescriptorHandle rtv1 = scene.rtv(1);
    list.OMSetRenderTargets(1, &rtv1, false, nullptr);
    list.IASetPrimitiveTopology(pso::D3DPrimitiveTopology::LineList);  // mesh draws are triangles regardless
    list.SetMeshPipelineState(mesh.get());
    const size_t mark = enc.size();
    CHECK(list.DispatchMesh(5, 1, 1));
    uint32_t output_buffer = 0;
    {
        using O = GpuOpcode;
        const std::vector<Packet> p = decode(enc, mark);
        const size_t create = find_op(p, O::CreateComputePipeline2);
        const size_t dispatch = find_op(p, O::DispatchCompute);
        const size_t begin = find_op(p, O::BeginRenderPass, dispatch);
        const size_t render = find_op(p, O::CreateRenderPipeline2, dispatch);
        const size_t draw = find_op(p, O::Draw, dispatch);
        CHECK(create < dispatch && dispatch < begin && begin < draw && render < draw && draw < p.size());
        if (draw >= p.size() || render >= p.size()) {
            return;
        }
        const auto cp = pod<GpuCreateComputePipeline2Payload>(p[create]);
        const auto rp = pod<GpuCreateRenderPipeline2Payload>(p[render]);
        CHECK(cp.cs_module_id == 6 && rp.vs_module_id == 7 && rp.fs_module_id == Scene::kFsModule);
        CHECK(pod<GpuCreateRenderPipeline2Payload>(p[render]).key.primitive.topology ==
              static_cast<uint8_t>(pso::WgpuPrimitiveTopology::TriangleList));
        // Both pipeline layouts: own group, two empty groups, the output group.
        size_t layouts = 0;
        for (const Packet& k : p) {
            if (k.opcode == O::CreatePipelineLayout) {
                const auto pl = pod<GpuCreatePipelineLayoutPayload>(k);
                if (pl.id == cp.pipeline_layout_id || pl.id == rp.pipeline_layout_id) {
                    CHECK(pl.bind_group_layout_count == 4);
                    ++layouts;
                }
            }
        }
        CHECK(layouts == 2);
        for (size_t i = 0; i < dispatch; ++i) {
            if (p[i].opcode == O::CreateBuffer && pod<GpuCreateBufferPayload>(p[i]).byte_size == bridge::kMeshOutputBytes) {
                CHECK(pod<GpuCreateBufferPayload>(p[i]).usage_flags == GpuBufferUsage::Storage);
                output_buffer = pod<GpuCreateBufferPayload>(p[i]).id;
            }
        }
        CHECK(output_buffer != 0);
        // Group 3 is set right before the dispatch and before the draw.
        const auto stage_set = pod<GpuSetBindGroup2Payload>(p[dispatch - 1]);
        CHECK(p[dispatch - 1].opcode == O::SetBindGroup2 && stage_set.group_index == bridge::kMeshOutputGroup);
        const auto draw_set = pod<GpuSetBindGroup2Payload>(p[draw - 1]);
        CHECK(p[draw - 1].opcode == O::SetBindGroup2 && draw_set.group_index == bridge::kMeshOutputGroup &&
              draw_set.bind_group_id != stage_set.bind_group_id);
        size_t output_groups = 0;
        for (const Packet& k : p) {
            if (k.opcode == O::CreateBindGroup2 && pod<GpuCreateBindGroup2Payload>(k).entry_count == 1) {
                output_groups += pod<GpuBindGroupEntry2>(k, sizeof(GpuCreateBindGroup2Payload)).resource_id ==
                                 output_buffer;
            }
        }
        CHECK(output_groups == 2);
        const auto d = pod<GpuDispatchComputePayload>(p[dispatch]);
        CHECK(d.workgroup_x == 5 && d.workgroup_y == 1 && d.workgroup_z == 1);
        const auto dr = pod<GpuDrawPayload>(p[draw]);
        CHECK(dr.vertex_count == 5 * 32 * 3 && dr.instance_count == 1);
    }
    CHECK(list.stats().mesh_dispatches == 1);

    // y/z groups and over-budget x are skipped without packets.
    const size_t mark2 = enc.size();
    CHECK(!list.DispatchMesh(5, 2, 1));
    CHECK(!list.DispatchMesh(bridge::mesh_max_groups(out) + 1, 1, 1));
    CHECK(enc.size() == mark2 && list.stats().mesh_skipped == 2);

    // Indirect: per command a dispatch record, then the draw record 12 bytes on (28-byte pairs).
    bridge::IndirectExecution e;
    e.kind = bridge::IndirectKind::DispatchMesh;
    e.max_count = 2;
    e.stride = 16;
    e.args_buffer = kArgs;
    e.args_offset = 32;
    CHECK(list.ExecuteIndirect(e));
    {
        using O = GpuOpcode;
        const std::vector<Packet> p = decode(enc, mark2);
        const size_t clamp_dispatch = find_op(p, O::DispatchCompute);
        const std::array<uint32_t, 8> params = indirect_params(p, clamp_dispatch);
        CHECK((params == std::array<uint32_t, 8>{8, 4, 2, 0, 0, 3, 32 * 3, bridge::mesh_max_groups(out)}));
        std::vector<GpuIndirectPayload> work;
        std::vector<O> ops;
        for (size_t i = clamp_dispatch; i < p.size(); ++i) {
            if (p[i].opcode == O::DispatchIndirect || p[i].opcode == O::DrawIndirect) {
                work.push_back(pod<GpuIndirectPayload>(p[i]));
                ops.push_back(p[i].opcode);
                const auto set = pod<GpuSetBindGroup2Payload>(p[i - 1]);
                CHECK(p[i - 1].opcode == O::SetBindGroup2 && set.group_index == bridge::kMeshOutputGroup);
            }
        }
        CHECK((ops == std::vector<O>{O::DispatchIndirect, O::DrawIndirect, O::DispatchIndirect, O::DrawIndirect}));
        if (work.size() == 4) {
            CHECK(work[1].offset == work[0].offset + 12 && work[2].offset == work[0].offset + 28 &&
                  work[3].offset == work[0].offset + 40);
            CHECK(work[0].buffer_id == work[3].buffer_id);
        }
    }
    CHECK(list.stats().mesh_dispatches == 3 && list.stats().indirect_commands == 2);

    // A regular pipeline state unbinds the mesh one: mesh work is skipped.
    list.SetPipelineState(scene.pso.get());
    const size_t mark3 = enc.size();
    CHECK(!list.ExecuteIndirect(e));
    CHECK(!list.DispatchMesh(1, 1, 1));
    CHECK(enc.size() == mark3 && list.stats().mesh_skipped == 4);
    list.Close();

    // A stage layout with more than three groups cannot take the output group.
    constexpr uint64_t kWide = 0xE8;
    bridge::PipelineLayoutDesc wide;
    for (int i = 0; i < 4; ++i) {
        wide.groups.push_back(bridge::GroupLayoutDesc{});
    }
    scene.device.register_shader_module(kWide, 8);
    scene.device.register_pipeline_layout(kWide, 0, wide);
    auto bad = scene.device.create_mesh_pipeline_state(desc, kWide, out, &scene.root_signature);
    CHECK((bad->diagnostics() & bridge::BridgeDiag::MissingPipelineLayout) != 0);
}

// Binding arrays (G62, lane-indexed heap windows): with the device capability on, the layout entry
// carries binding_array_size and the bind group a run of elements; off (default), such a layout is
// refused and nothing array-shaped reaches the stream.
void test_binding_array_layouts() {
    constexpr uint64_t kCs = 0xCA;
    constexpr uint32_t kCsModule = 9;
    constexpr uint32_t kFirstTexture = 400;

    d3d::RootSignatureDesc rs;
    rs.flags = 0x400;  // CBV_SRV_UAV heap directly indexed
    d3d::RootParameterDesc base;
    base.parameter_type = d3d::RootParameterType::Constants32Bit;
    base.num_32bit_values = 1;
    rs.parameters.push_back(base);
    d3d::RootSignature signature;
    CHECK(d3d::RootSignature::build(rs, &signature).ok());

    bridge::PipelineLayoutDesc layout;
    bridge::GroupLayoutDesc group;
    GpuBindGroupLayoutEntry tex = layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Texture,
                                               static_cast<uint8_t>(GpuTextureSampleType::Float),
                                               static_cast<uint8_t>(GpuTextureViewDimension::Dim2D));
    tex.binding_array_size = 3;
    d3d::ShaderBinding window = join(0, d3d::DescriptorRangeType::Srv, 0);
    window.heap = d3d::HeapAccess::Resource;
    window.heap_index_parameter = 0;
    window.heap_index_source = d3d::HeapIndexSource::RootConstantIndex;
    window.heap_shift = 5;
    group.bindings.push_back({tex, window});
    layout.groups.push_back(group);
    pso::D3D12ComputePsoDesc desc;
    desc.cs = kCs;
    desc.root_signature_id = 11;

    auto record = [&](bool binding_arrays, std::vector<Packet>* out) {
        d3d::DescriptorHeapRegistry heaps;
        bridge::BridgeOptions options = counting_options(7000);
        options.binding_arrays = binding_arrays;
        bridge::StateBridgeDevice device(heaps, options);
        d3d::DescriptorHeap* srv_heap = make_heap(heaps, d3d::DescriptorHeapType::CbvSrvUav, 8, true);
        d3d::ViewDesc srv;
        srv.format = 28;  // R8G8B8A8_UNORM
        srv.dimension = d3d::ViewDimension::Texture2D;
        srv.mip_levels = 0xffff;
        for (uint32_t i = 0; i < 8; ++i) {
            device.register_texture(kFirstTexture + i, {pso::DxgiFormat::R8G8B8A8_UNORM, 16, 16, 1, 1, 1,
                                                        GpuTextureDimension::Dim2D, false});
            CHECK(heaps.create_shader_resource_view(kFirstTexture + i, &srv, srv_heap->cpu_handle(i)) ==
                  d3d::Status::Ok);
        }
        device.register_shader_module(kCs, kCsModule);
        device.register_pipeline_layout(kCs, 0, layout);
        auto cpso = device.create_compute_pipeline_state(desc, &signature);
        gpu::CommandEncoder enc;
        bridge::CommandListBridge list(device, enc);
        d3d::DescriptorHeap* visible[] = {srv_heap};
        CHECK(list.SetDescriptorHeaps(1, visible) == d3d::Status::Ok);
        CHECK(list.SetComputeRootSignature(&signature) == d3d::Status::Ok);
        const uint32_t window_base = 2;
        CHECK(list.SetComputeRoot32BitConstants(0, 1, &window_base, 0) == d3d::Status::Ok);
        list.SetComputePipelineState(cpso.get());
        list.Dispatch(1, 1, 1);
        list.Close();
        *out = decode(enc);
    };

    std::vector<Packet> on;
    record(true, &on);
    const Packet* bgl = nullptr;
    const Packet* bg = nullptr;
    for (const Packet& pk : on) {
        bgl = pk.opcode == GpuOpcode::CreateBindGroupLayout ? &pk : bgl;
        bg = pk.opcode == GpuOpcode::CreateBindGroup2 ? &pk : bg;
    }
    CHECK(bgl != nullptr && bg != nullptr);
    if (bgl != nullptr && bg != nullptr) {
        const auto h = pod<GpuCreateBindGroupLayoutPayload>(*bgl);
        CHECK(h.flags == GpuLayoutFlags::BindingArrays && h.entry_count == 1);
        CHECK(bgl->payload.size() == sizeof(h) + sizeof(GpuBindGroupLayoutEntry));
        const auto e = pod<GpuBindGroupLayoutEntry>(*bgl, sizeof(h));
        CHECK(e.binding == 0 && e.binding_array_size == 3 && e.reserved == 0);
        const auto g = pod<GpuCreateBindGroup2Payload>(*bg);
        CHECK(g.flags == GpuBindGroupFlags::BindingArrays && g.entry_count == 3 && g.layout_id == h.id);
        CHECK(bg->payload.size() == sizeof(g) + 3 * sizeof(GpuBindGroupEntry2));
        for (uint32_t k = 0; k < 3 && bg->payload.size() == sizeof(g) + 3 * sizeof(GpuBindGroupEntry2); ++k) {
            const auto el = pod<GpuBindGroupEntry2>(*bg, sizeof(g) + k * sizeof(GpuBindGroupEntry2));
            CHECK(el.binding == 0 && el.array_index == k && el.reserved == 0);
            CHECK(el.resource_type == static_cast<uint32_t>(GpuBindResourceType::TextureView));
            // Heap descriptors window_base + k.
            CHECK(el.resource_id == kFirstTexture + 2 + k);
        }
    }

    // Capability off (default): the array layout is refused, no layout / bind group is emitted.
    std::vector<Packet> off;
    record(false, &off);
    for (const Packet& pk : off) {
        CHECK(pk.opcode != GpuOpcode::CreateBindGroupLayout && pk.opcode != GpuOpcode::CreateBindGroup2);
    }
    CHECK(!bridge::BridgeOptions{}.binding_arrays);
}

void test_diagnostics_and_closed_list() {
    Scene scene;
    gpu::CommandEncoder enc;
    bridge::CommandListBridge list(scene.device, enc);
    // Draw without render targets: skipped by G7.
    list.SetPipelineState(scene.pso.get());
    list.DrawInstanced(3, 1, 0, 0);
    CHECK(list.stats().draws_skipped == 1);
    CHECK((list.state_diagnostics() & pso::StateDiag::DrawWithoutAttachments) != 0);

    // Root arguments missing: the bind group cannot be built and the draw is skipped.
    d3d::DescriptorHeap* visible[] = {scene.srv_heap};
    CHECK(list.SetDescriptorHeaps(1, visible) == d3d::Status::Ok);
    CHECK(list.SetGraphicsRootSignature(&scene.root_signature) == d3d::Status::Ok);
    const d3d::CpuDescriptorHandle rtv0 = scene.rtv(0);
    list.OMSetRenderTargets(1, &rtv0, false, nullptr);
    list.DrawInstanced(3, 1, 0, 0);
    CHECK((list.diagnostics() & bridge::BridgeDiag::RootArgumentNotSet) != 0);
    CHECK(list.stats().draws == 0 && list.stats().draws_skipped == 2);

    // Indexed draw without an index buffer.
    list.DrawIndexedInstanced(3, 1, 0, 0, 0);
    CHECK(list.stats().draws_skipped == 3);

    list.Close();
    const size_t size = enc.size();
    list.DrawInstanced(3, 1, 0, 0);
    CHECK(enc.size() == size);
    CHECK((list.diagnostics() & bridge::BridgeDiag::CommandListClosed) != 0);

    // Reset reopens with clean state.
    list.Reset(scene.pso.get());
    CHECK(list.is_open() && list.diagnostics() == 0 && list.stats().draws == 0);

    // A PSO without a registered layout / shader modules is diagnosed at creation.
    pso::D3D12GraphicsPsoDesc desc = Scene::pso_desc();
    desc.vs = 0xDEAD;
    auto orphan = scene.device.create_graphics_pipeline_state(desc, &scene.root_signature);
    CHECK((orphan->diagnostics() & bridge::BridgeDiag::UnknownShaderModule) != 0);
    CHECK((orphan->diagnostics() & bridge::BridgeDiag::MissingPipelineLayout) != 0);
}

void test_indexed_draw_and_layout_dedup() {
    Scene scene;
    // Registering the same layout for another shader pair reuses the layout objects.
    bridge::PipelineLayoutDesc again;
    bridge::GroupLayoutDesc group;
    group.content_id = Scene::kBglContent;
    group.bindings.push_back({layout_entry(0, GpuShaderStageBits::Vertex | GpuShaderStageBits::Fragment,
                                           GpuBindingLayoutType::Buffer,
                                           static_cast<uint8_t>(GpuBufferBindingType::Uniform)),
                              join(0, d3d::DescriptorRangeType::Cbv, 0, true)});
    group.bindings.push_back({layout_entry(1, GpuShaderStageBits::Fragment, GpuBindingLayoutType::Texture,
                                           static_cast<uint8_t>(GpuTextureSampleType::Float),
                                           static_cast<uint8_t>(GpuTextureViewDimension::Dim2D)),
                              join(1, d3d::DescriptorRangeType::Srv, 0)});
    group.bindings.push_back({layout_entry(2, GpuShaderStageBits::Fragment, GpuBindingLayoutType::Sampler,
                                           static_cast<uint8_t>(GpuSamplerBindingType::Filtering)),
                              join(2, d3d::DescriptorRangeType::Sampler, 0)});
    again.content_id = Scene::kPlContent;
    again.groups.push_back(group);
    scene.device.register_pipeline_layout(0xA1, 0xB3, again);
    scene.device.register_shader_module(0xB3, 4);
    pso::D3D12GraphicsPsoDesc desc = Scene::pso_desc();
    desc.ps = 0xB3;
    auto second = scene.device.create_graphics_pipeline_state(desc, &scene.root_signature);

    gpu::CommandEncoder enc;
    bridge::CommandListBridge list(scene.device, enc);
    scene.bind_root(list);
    const d3d::CpuDescriptorHandle rtv0 = scene.rtv(0);
    list.OMSetRenderTargets(1, &rtv0, false, nullptr);
    const gpu::d3d12_xs::D3D12IndexBufferView ib{201, 64, 1024, GpuIndexFormat::Uint32};
    list.IASetIndexBuffer(&ib);
    list.DrawIndexedInstanced(36, 1, 0, -2, 0);
    list.SetPipelineState(second.get());
    list.DrawIndexedInstanced(36, 1, 0, 0, 0);
    list.Close();

    size_t layouts = 0;
    size_t pipelines = 0;
    size_t index_sets = 0;
    uint32_t layout_ids[2] = {0, 0};
    for (const Packet& pk : decode(enc)) {
        layouts += pk.opcode == GpuOpcode::CreatePipelineLayout ? 1 : 0;
        index_sets += pk.opcode == GpuOpcode::SetIndexBuffer ? 1 : 0;
        if (pk.opcode == GpuOpcode::CreateRenderPipeline2 && pipelines < 2) {
            layout_ids[pipelines++] = pod<GpuCreateRenderPipeline2Payload>(pk).pipeline_layout_id;
        }
        if (pk.opcode == GpuOpcode::SetIndexBuffer) {
            const auto ibp = pod<GpuSetIndexBufferPayload>(pk);
            CHECK(ibp.buffer_id == 201 && ibp.offset == 64 &&
                  ibp.index_format == static_cast<uint32_t>(GpuIndexFormat::Uint32));
        }
        if (pk.opcode == GpuOpcode::DrawIndexed) {
            CHECK(pod<GpuDrawIndexedPayload>(pk).index_count == 36);
        }
    }
    CHECK(layouts == 1);
    CHECK(pipelines == 2 && layout_ids[0] == layout_ids[1] && layout_ids[0] != 0);
    CHECK(index_sets == 1);  // same pass: the index buffer is not re-bound for the second PSO
    CHECK(list.stats().draws == 2);
}

// Backlog 28: a released command list takes its cached bind groups with it (destroy deferred, the
// process-wide live gauges drop), so create/release cycles plateau.
void test_released_list_retires_bind_groups() {
    Scene scene;
    const bridge::BridgeLiveCounts base = bridge::bridge_live_counts();
    const size_t base_deferred = scene.device.deferred_destroy_count();
    {
        gpu::CommandEncoder enc;
        bridge::CommandListBridge list(scene.device, enc);
        scene.bind_root(list);
        const d3d::CpuDescriptorHandle rtv0 = scene.rtv(0);
        list.OMSetRenderTargets(1, &rtv0, false, nullptr);
        list.DrawInstanced(3, 1, 0, 0);
        list.Close();
        CHECK(list.cached_bind_group_count() >= 1);
        const bridge::BridgeLiveCounts live = bridge::bridge_live_counts();
        CHECK(live.command_lists == base.command_lists + 1);
        CHECK(live.cached_bind_groups >= base.cached_bind_groups + list.cached_bind_group_count());
    }
    const bridge::BridgeLiveCounts after = bridge::bridge_live_counts();
    CHECK(after.command_lists == base.command_lists);
    CHECK(after.cached_bind_groups == base.cached_bind_groups);
    CHECK(scene.device.deferred_destroy_count() > base_deferred);
}

// G66: N identical draws create exactly the objects one draw does; every later draw is a cache hit.
void test_identical_draws_create_no_new_objects() {
    struct Result {
        size_t creates = 0;
        bridge::BridgeStats stats;
    };
    auto run = [](uint32_t n) {
        Scene scene;
        gpu::CommandEncoder enc;
        bridge::CommandListBridge list(scene.device, enc);
        scene.bind_root(list);
        const d3d::CpuDescriptorHandle rtv0 = scene.rtv(0);
        list.OMSetRenderTargets(1, &rtv0, false, nullptr);
        for (uint32_t i = 0; i < n; ++i) {
            list.DrawInstanced(3, 1, 0, 0);
        }
        list.Close();
        Result r;
        for (const Packet& pk : decode(enc)) {
            using O = GpuOpcode;
            r.creates += (pk.opcode == O::CreateBuffer || pk.opcode == O::CreateSampler2 ||
                          pk.opcode == O::CreateBindGroup2 || pk.opcode == O::CreateBindGroupLayout ||
                          pk.opcode == O::CreatePipelineLayout || pk.opcode == O::CreateRenderPipeline2)
                             ? 1
                             : 0;
        }
        r.stats = list.stats();
        return r;
    };
    const Result one = run(1);
    const Result many = run(64);
    CHECK(one.creates > 0);
    CHECK(many.creates == one.creates);
    CHECK(many.stats.draws == 64);
    CHECK(many.stats.pipelines_created == one.stats.pipelines_created);
    CHECK(many.stats.bind_groups_created == one.stats.bind_groups_created);
    CHECK(many.stats.bind_group_cache_misses == one.stats.bind_group_cache_misses);
    CHECK(many.stats.bind_group_cache_hits == one.stats.bind_group_cache_hits + 63);
}

// Several recording threads share one device: pipelines, layouts, samplers and the root constant ring
// are created exactly once across all streams.
void test_concurrent_command_lists() {
    Scene scene;
    constexpr size_t kThreads = 4;
    constexpr uint64_t kFrames = 25;
    std::vector<gpu::CommandEncoder> encoders(kThreads);
    std::vector<std::thread> threads;
    std::vector<uint64_t> draws(kThreads, 0);
    for (size_t t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            bridge::CommandListBridge list(scene.device, encoders[t]);
            for (uint64_t frame = 0; frame < kFrames; ++frame) {
                list.Reset();
                scene.bind_root(list);
                const d3d::CpuDescriptorHandle rtv = scene.rtv(static_cast<uint32_t>(frame & 1));
                list.OMSetRenderTargets(1, &rtv, false, nullptr);
                const float color[4] = {0.0f, 0.0f, 0.0f, 1.0f};
                list.ClearRenderTargetView(rtv, color);
                list.DrawInstanced(3, 1, 0, 0);
                list.Close();
                draws[t] += list.stats().draws;
            }
        });
    }
    for (std::thread& th : threads) {
        th.join();
    }
    size_t layouts = 0;
    size_t pipelines = 0;
    size_t samplers = 0;
    size_t rings = 0;
    for (size_t t = 0; t < kThreads; ++t) {
        CHECK(draws[t] == kFrames);
        for (const Packet& p : decode(encoders[t])) {
            layouts += p.opcode == GpuOpcode::CreatePipelineLayout ? 1 : 0;
            pipelines += p.opcode == GpuOpcode::CreateRenderPipeline2 ? 1 : 0;
            samplers += p.opcode == GpuOpcode::CreateSampler2 ? 1 : 0;
            rings += p.opcode == GpuOpcode::CreateBuffer ? 1 : 0;
        }
    }
    CHECK(layouts == 1);
    CHECK(pipelines == 1);  // both targets are bgra8unorm without depth: one variant
    CHECK(samplers == 1);
    CHECK(rings == 1);
}

}  // namespace

// A constant-buffer view at a 16-byte aligned address (Xbox) bound through a descriptor table:
// WebGPU cannot bind a uniform buffer at that offset, so the bridge asks the relocate hook for an
// aligned copy and binds that; without the hook the binding is diagnosed and bound null. G66
// (#691): a view of zero size (a console descriptor of zero records) is relocated too. Every
// uniform binding the bridge emits sits at a 256-byte aligned offset, within the 64 KiB limit.
void test_unaligned_uniform_relocation() {
    for (const int variant : {0, 1, 2}) {  // 0: hook, 1: no hook, 2: hook + zero-size view
        const bool with_hook = variant != 1;
        const bool zero_size = variant == 2;
        d3d::DescriptorHeapRegistry heaps;
        heaps.set_cbv_alignment(16);
        d3d::ResourceStateRegistry states;
        std::vector<std::pair<uint64_t, uint64_t>> calls;
        bridge::BridgeOptions options = counting_options(5000);
        if (with_hook) {
            options.relocate_uniform = [&calls](uint64_t va, uint64_t size, uint64_t, uint32_t* id, uint64_t* offset) {
                calls.emplace_back(va, size);
                *id = 777;
                *offset = 512;
                return true;
            };
        }
        bridge::StateBridgeDevice device(heaps, options, &states);
        constexpr uint32_t kTarget = 100, kCb = 300, kVsModule = 1, kFsModule = 2;
        constexpr uint64_t kVs = 0xC1, kPs = 0xC2, kCbVa = 0x10000;
        device.register_texture(kTarget, {pso::DxgiFormat::B8G8R8A8_UNORM, 64, 64, 1, 1, 1,
                                          GpuTextureDimension::Dim2D, false});
        device.register_buffer(kCb, kCbVa, 4096);
        device.register_shader_module(kVs, kVsModule);
        device.register_shader_module(kPs, kFsModule);
        d3d::DescriptorHeap* rtv_heap = make_heap(heaps, d3d::DescriptorHeapType::Rtv, 1, false);
        d3d::DescriptorHeap* cbv_heap = make_heap(heaps, d3d::DescriptorHeapType::CbvSrvUav, 4, true);
        CHECK(heaps.create_render_target_view(kTarget, nullptr, rtv_heap->cpu_handle(0)) == d3d::Status::Ok);
        d3d::ConstantBufferViewDesc cbv;
        cbv.buffer_location = kCbVa + 0x40;
        cbv.size_in_bytes = zero_size ? 0 : 96;
        CHECK(heaps.create_constant_buffer_view(&cbv, cbv_heap->cpu_handle(0)) == d3d::Status::Ok);

        d3d::RootSignatureDesc rs;
        d3d::RootParameterDesc table;
        table.parameter_type = d3d::RootParameterType::DescriptorTable;
        d3d::DescriptorRangeDesc range;
        range.range_type = d3d::DescriptorRangeType::Cbv;
        range.num_descriptors = 1;
        table.ranges.push_back(range);
        rs.parameters.push_back(table);
        d3d::RootSignature root;
        CHECK(d3d::RootSignature::build(rs, &root).ok());
        bridge::PipelineLayoutDesc layout;
        bridge::GroupLayoutDesc group;
        group.bindings.push_back({layout_entry(0, GpuShaderStageBits::Vertex | GpuShaderStageBits::Fragment,
                                               GpuBindingLayoutType::Buffer,
                                               static_cast<uint8_t>(GpuBufferBindingType::Uniform)),
                                  join(0, d3d::DescriptorRangeType::Cbv, 0)});
        if (zero_size) {
            group.bindings.back().entry.min_binding_size = 16;  // the stage's reflected block (as in the trace)
        }
        layout.groups.push_back(group);
        device.register_pipeline_layout(kVs, kPs, layout);
        pso::D3D12GraphicsPsoDesc d = Scene::pso_desc();
        d.vs = kVs;
        d.ps = kPs;
        d.DSVFormat = pso::DxgiFormat::UNKNOWN;
        auto state = device.create_graphics_pipeline_state(d, &root);
        CHECK(state->diagnostics() == 0);

        gpu::CommandEncoder enc;
        bridge::CommandListBridge list(device, enc);
        d3d::DescriptorHeap* visible[] = {cbv_heap};
        CHECK(list.SetDescriptorHeaps(1, visible) == d3d::Status::Ok);
        CHECK(list.SetGraphicsRootSignature(&root) == d3d::Status::Ok);
        list.SetPipelineState(state.get());
        CHECK(list.SetGraphicsRootDescriptorTable(0, cbv_heap->gpu_handle(0)) == d3d::Status::Ok);
        list.IASetPrimitiveTopology(pso::D3DPrimitiveTopology::TriangleList);
        const gpu::d3d12_xs::D3D12VertexBufferView vb{Scene::kVertexBuffer, 0, 36, 12};
        list.IASetVertexBuffers(0, 1, &vb);
        const d3d::CpuDescriptorHandle rtv0 = rtv_heap->cpu_handle(0);
        list.OMSetRenderTargets(1, &rtv0, false, nullptr);
        list.DrawInstanced(3, 1, 0, 0);
        list.Close();

        const auto p = decode(enc);
        const Packet* bind_group = nullptr;
        for (const Packet& packet : p) {
            if (packet.opcode == GpuOpcode::CreateBindGroup2 && packet.payload.size() > sizeof(GpuCreateBindGroup2Payload)) {
                bind_group = &packet;
                // Every buffer binding emitted: a 256-byte aligned offset, within the 64 KiB
                // uniform limit (or null).
                const auto g = pod<GpuCreateBindGroup2Payload>(packet, 0);
                for (uint32_t k = 0; k < g.entry_count; ++k) {
                    const auto be = pod<GpuBindGroupEntry2>(
                        packet, sizeof(GpuCreateBindGroup2Payload) + size_t{k} * sizeof(GpuBindGroupEntry2));
                    if (be.resource_type == static_cast<uint32_t>(GpuBindResourceType::Buffer)) {
                        CHECK(be.offset % bridge::kUniformOffsetAlignment == 0);
                        CHECK(be.resource_id == 0 || (be.size != 0 && be.size <= bridge::kMaxUniformBindingSize));
                    }
                }
            }
        }
        CHECK(bind_group != nullptr);
        if (bind_group != nullptr) {
            const auto entry = pod<GpuBindGroupEntry2>(*bind_group, sizeof(GpuCreateBindGroup2Payload));
            if (with_hook) {
                CHECK(entry.resource_id == 777 && entry.offset == 512 && entry.size == 256);  // whole relocated slot
                // A zero-size view is relocated at the stage's reflected block (16 bytes).
                CHECK(calls.size() == 1 && calls[0].first == kCbVa + 0x40 &&
                      calls[0].second == (zero_size ? 16u : 96u));
                CHECK((list.diagnostics() & bridge::BridgeDiag::UnalignedUniform) == 0);
            } else {
                // Bound null: an unaligned offset would invalidate the bind group and its draws.
                CHECK(entry.resource_id == 0 && entry.offset == 0);
                CHECK((list.diagnostics() & bridge::BridgeDiag::UnalignedUniform) != 0);
            }
        }
    }
}

// O36: a structured-buffer SRV starting at an element that is not 256-byte aligned, bound as a
// read-only storage buffer: WebGPU's minStorageBufferOffsetAlignment is 256 too, so the bridge binds
// a relocated copy (relocate_storage hook); without the hook the binding is diagnosed.
// #482: a storage-texture binding (a stage writing through a table slot). A texture created with
// STORAGE_BINDING usage is bound as a storage view in the stage's storage format (an sRGB view of
// an R8G8B8A8_UNORM_SRGB texture becomes rgba8unorm); one created without it is bound null and
// counted, instead of a view WebGPU rejects with the whole command buffer.
void test_storage_texture_needs_storage_usage() {
    for (const int variant : {0, 1, 2}) {  // 0: storage usage, 1: no storage usage, 2: other format
        const bool with_usage = variant != 1;
        d3d::DescriptorHeapRegistry heaps;
        d3d::ResourceStateRegistry states;
        bridge::StateBridgeDevice device(heaps, counting_options(7000), &states);
        constexpr uint32_t kTarget = 100, kTex = 302;
        constexpr uint64_t kVs = 0xE1, kPs = 0xE2;
        device.register_texture(kTarget, {pso::DxgiFormat::B8G8R8A8_UNORM, 64, 64, 1, 1, 1,
                                          GpuTextureDimension::Dim2D, false});
        bridge::TextureInfo tex{variant == 2 ? pso::DxgiFormat::R8G8B8A8_UINT : pso::DxgiFormat::R8G8B8A8_UNORM_SRGB,
                                32, 32, 1, 1, 1, GpuTextureDimension::Dim2D, false};
        tex.storage = with_usage;
        device.register_texture(kTex, tex);
        device.register_shader_module(kVs, 1);
        device.register_shader_module(kPs, 2);
        d3d::DescriptorHeap* rtv_heap = make_heap(heaps, d3d::DescriptorHeapType::Rtv, 1, false);
        d3d::DescriptorHeap* view_heap = make_heap(heaps, d3d::DescriptorHeapType::CbvSrvUav, 4, true);
        CHECK(heaps.create_render_target_view(kTarget, nullptr, rtv_heap->cpu_handle(0)) == d3d::Status::Ok);
        d3d::ViewDesc vd;
        vd.format = static_cast<uint32_t>(tex.format);
        vd.dimension = d3d::ViewDimension::Texture2D;
        CHECK(heaps.create_unordered_access_view(kTex, 0, &vd, view_heap->cpu_handle(0)) == d3d::Status::Ok);
        d3d::RootSignatureDesc rs;
        d3d::RootParameterDesc table;
        table.parameter_type = d3d::RootParameterType::DescriptorTable;
        d3d::DescriptorRangeDesc range;
        range.range_type = d3d::DescriptorRangeType::Uav;
        range.num_descriptors = 1;
        table.ranges.push_back(range);
        rs.parameters.push_back(table);
        d3d::RootSignature root;
        CHECK(d3d::RootSignature::build(rs, &root).ok());
        bridge::PipelineLayoutDesc layout;
        bridge::GroupLayoutDesc group;
        GpuBindGroupLayoutEntry entry =
            layout_entry(0, GpuShaderStageBits::Fragment, GpuBindingLayoutType::StorageTexture,
                         static_cast<uint8_t>(GpuStorageTextureAccess::WriteOnly),
                         static_cast<uint8_t>(GpuTextureViewDimension::Dim2D));
        entry.storage_format = static_cast<uint8_t>(pso::WgpuTextureFormat::Rgba8Unorm);
        group.bindings.push_back({entry, join(0, d3d::DescriptorRangeType::Uav, 0)});
        layout.groups.push_back(group);
        device.register_pipeline_layout(kVs, kPs, layout);
        pso::D3D12GraphicsPsoDesc d = Scene::pso_desc();
        d.vs = kVs;
        d.ps = kPs;
        d.DSVFormat = pso::DxgiFormat::UNKNOWN;
        auto state = device.create_graphics_pipeline_state(d, &root);
        gpu::CommandEncoder enc;
        bridge::CommandListBridge list(device, enc);
        d3d::DescriptorHeap* visible[] = {view_heap};
        CHECK(list.SetDescriptorHeaps(1, visible) == d3d::Status::Ok);
        CHECK(list.SetGraphicsRootSignature(&root) == d3d::Status::Ok);
        list.SetPipelineState(state.get());
        CHECK(list.SetGraphicsRootDescriptorTable(0, view_heap->gpu_handle(0)) == d3d::Status::Ok);
        list.IASetPrimitiveTopology(pso::D3DPrimitiveTopology::TriangleList);
        const gpu::d3d12_xs::D3D12VertexBufferView vb{Scene::kVertexBuffer, 0, 36, 12};
        list.IASetVertexBuffers(0, 1, &vb);
        const d3d::CpuDescriptorHandle rtv0 = rtv_heap->cpu_handle(0);
        list.OMSetRenderTargets(1, &rtv0, false, nullptr);
        list.DrawInstanced(3, 1, 0, 0);
        list.Close();
        GpuBindGroupEntry2 out{};
        bool found = false;
        for (const Packet& packet : decode(enc)) {
            if (packet.opcode == GpuOpcode::CreateBindGroup2 &&
                packet.payload.size() > sizeof(GpuCreateBindGroup2Payload)) {
                out = pod<GpuBindGroupEntry2>(packet, sizeof(GpuCreateBindGroup2Payload));
                found = true;
            }
        }
        CHECK(found);
        if (variant == 0) {
            CHECK(out.resource_id == kTex &&
                  out.view.format == static_cast<uint8_t>(pso::WgpuTextureFormat::Rgba8Unorm));
            CHECK(list.stats().storage_views_without_usage == 0 && list.stats().storage_views_format_mismatch == 0 &&
                  (list.diagnostics() & bridge::BridgeDiag::StorageViewNoUsage) == 0);
        } else {
            CHECK(out.resource_id == 0);
            CHECK(list.stats().storage_views_without_usage == (variant == 1 ? 1u : 0u));
            CHECK(list.stats().storage_views_format_mismatch == (variant == 2 ? 1u : 0u));
            CHECK((list.diagnostics() & bridge::BridgeDiag::StorageViewNoUsage) != 0);
        }
    }
}

void test_unaligned_storage_relocation() {
    for (const bool with_hook : {true, false}) {
        d3d::DescriptorHeapRegistry heaps;
        d3d::ResourceStateRegistry states;
        std::vector<std::pair<uint64_t, uint64_t>> calls;
        bridge::BridgeOptions options = counting_options(5000);
        if (with_hook) {
            options.relocate_storage = [&calls](uint64_t va, uint64_t size, uint32_t* id, uint64_t* offset) {
                calls.emplace_back(va, size);
                *id = 888;
                *offset = 1024;
                return true;
            };
        }
        bridge::StateBridgeDevice device(heaps, options, &states);
        constexpr uint32_t kTarget = 100, kBuf = 301, kVsModule = 1, kFsModule = 2;
        constexpr uint64_t kVs = 0xC1, kPs = 0xC2, kBufVa = 0x20000;
        device.register_texture(kTarget, {pso::DxgiFormat::B8G8R8A8_UNORM, 64, 64, 1, 1, 1,
                                          GpuTextureDimension::Dim2D, false});
        device.register_buffer(kBuf, kBufVa, 4096);
        device.register_shader_module(kVs, kVsModule);
        device.register_shader_module(kPs, kFsModule);
        d3d::DescriptorHeap* rtv_heap = make_heap(heaps, d3d::DescriptorHeapType::Rtv, 1, false);
        d3d::DescriptorHeap* srv_heap = make_heap(heaps, d3d::DescriptorHeapType::CbvSrvUav, 4, true);
        CHECK(heaps.create_render_target_view(kTarget, nullptr, rtv_heap->cpu_handle(0)) == d3d::Status::Ok);
        d3d::ViewDesc srv;
        srv.dimension = d3d::ViewDimension::Buffer;
        srv.first_element = 4;  // 4 x 16 bytes: offset 0x40
        srv.num_elements = 8;
        srv.structure_byte_stride = 16;
        CHECK(heaps.create_shader_resource_view(kBuf, &srv, srv_heap->cpu_handle(0)) == d3d::Status::Ok);

        d3d::RootSignatureDesc rs;
        d3d::RootParameterDesc table;
        table.parameter_type = d3d::RootParameterType::DescriptorTable;
        d3d::DescriptorRangeDesc range;
        range.range_type = d3d::DescriptorRangeType::Srv;
        range.num_descriptors = 1;
        table.ranges.push_back(range);
        rs.parameters.push_back(table);
        d3d::RootSignature root;
        CHECK(d3d::RootSignature::build(rs, &root).ok());
        bridge::PipelineLayoutDesc layout;
        bridge::GroupLayoutDesc group;
        group.bindings.push_back({layout_entry(0, GpuShaderStageBits::Vertex, GpuBindingLayoutType::Buffer,
                                               static_cast<uint8_t>(GpuBufferBindingType::ReadOnlyStorage)),
                                  join(0, d3d::DescriptorRangeType::Srv, 0)});
        layout.groups.push_back(group);
        device.register_pipeline_layout(kVs, kPs, layout);
        pso::D3D12GraphicsPsoDesc d = Scene::pso_desc();
        d.vs = kVs;
        d.ps = kPs;
        d.DSVFormat = pso::DxgiFormat::UNKNOWN;
        auto state = device.create_graphics_pipeline_state(d, &root);

        gpu::CommandEncoder enc;
        bridge::CommandListBridge list(device, enc);
        d3d::DescriptorHeap* visible[] = {srv_heap};
        CHECK(list.SetDescriptorHeaps(1, visible) == d3d::Status::Ok);
        CHECK(list.SetGraphicsRootSignature(&root) == d3d::Status::Ok);
        list.SetPipelineState(state.get());
        CHECK(list.SetGraphicsRootDescriptorTable(0, srv_heap->gpu_handle(0)) == d3d::Status::Ok);
        list.IASetPrimitiveTopology(pso::D3DPrimitiveTopology::TriangleList);
        const gpu::d3d12_xs::D3D12VertexBufferView vb{Scene::kVertexBuffer, 0, 36, 12};
        list.IASetVertexBuffers(0, 1, &vb);
        const d3d::CpuDescriptorHandle rtv0 = rtv_heap->cpu_handle(0);
        list.OMSetRenderTargets(1, &rtv0, false, nullptr);
        list.DrawInstanced(3, 1, 0, 0);
        list.Close();

        const Packet* bind_group = nullptr;
        const auto p = decode(enc);
        for (const Packet& packet : p) {
            if (packet.opcode == GpuOpcode::CreateBindGroup2 && packet.payload.size() > sizeof(GpuCreateBindGroup2Payload)) {
                bind_group = &packet;
            }
        }
        CHECK(bind_group != nullptr);
        if (bind_group != nullptr) {
            const auto entry = pod<GpuBindGroupEntry2>(*bind_group, sizeof(GpuCreateBindGroup2Payload));
            if (with_hook) {
                CHECK(entry.resource_id == 888 && entry.offset == 1024 && entry.size == 128);
                CHECK(calls.size() == 1 && calls[0].first == kBufVa + 0x40 && calls[0].second == 128);
                CHECK((list.diagnostics() & bridge::BridgeDiag::UnalignedStorage) == 0);
            } else {
                // G66 (#691): bound null, never at the unaligned offset (it would invalidate the
                // bind group and every draw using it).
                CHECK(entry.resource_id == 0 && entry.offset == 0);
                CHECK((list.diagnostics() & bridge::BridgeDiag::UnalignedStorage) != 0);
            }
        }
    }
}

// G66: the title's UI vertex stage (root signature: one root SRV) reads that root SRV as a uniform
// header (binding 0) and as read-only storage for vertex pulling (binding 3): the same GPU VA. At
// an unaligned VA both bindings are relocated copies of that memory; the storage one (no view size:
// a root view) copies up to the end of the buffer. It was bound null, which emptied the LEGAL
// NOTICES alpha mask.
void test_root_srv_read_as_uniform_and_storage() {
    d3d::DescriptorHeapRegistry heaps;
    d3d::ResourceStateRegistry states;
    std::vector<std::pair<uint64_t, uint64_t>> uniform_calls, storage_calls;
    bridge::BridgeOptions options = counting_options(5000);
    options.relocate_uniform = [&uniform_calls](uint64_t va, uint64_t size, uint64_t, uint32_t* id, uint64_t* offset) {
        uniform_calls.emplace_back(va, size);
        *id = 777;
        *offset = 512;
        return true;
    };
    options.relocate_storage = [&storage_calls](uint64_t va, uint64_t size, uint32_t* id, uint64_t* offset) {
        storage_calls.emplace_back(va, size);
        *id = 888;
        *offset = 1024;
        return true;
    };
    bridge::StateBridgeDevice device(heaps, options, &states);
    constexpr uint32_t kTarget = 100, kBuf = 301, kVsModule = 1, kFsModule = 2;
    constexpr uint64_t kVs = 0xC1, kPs = 0xC2, kBufVa = 0x20000, kViewOffset = 0x40;
    device.register_texture(kTarget, {pso::DxgiFormat::B8G8R8A8_UNORM, 64, 64, 1, 1, 1,
                                      GpuTextureDimension::Dim2D, false});
    device.register_buffer(kBuf, kBufVa, 4096);
    device.register_shader_module(kVs, kVsModule);
    device.register_shader_module(kPs, kFsModule);
    d3d::DescriptorHeap* rtv_heap = make_heap(heaps, d3d::DescriptorHeapType::Rtv, 1, false);
    CHECK(heaps.create_render_target_view(kTarget, nullptr, rtv_heap->cpu_handle(0)) == d3d::Status::Ok);

    d3d::RootSignatureDesc rs;
    d3d::RootParameterDesc srv;
    srv.parameter_type = d3d::RootParameterType::Srv;
    srv.shader_register = 0;
    rs.parameters.push_back(srv);
    d3d::RootSignature root;
    CHECK(d3d::RootSignature::build(rs, &root).ok());
    bridge::PipelineLayoutDesc layout;
    bridge::GroupLayoutDesc group;
    auto uniform = layout_entry(0, GpuShaderStageBits::Vertex, GpuBindingLayoutType::Buffer,
                                static_cast<uint8_t>(GpuBufferBindingType::Uniform));
    uniform.min_binding_size = 16;
    group.bindings.push_back({uniform, join(0, d3d::DescriptorRangeType::Srv, 0)});
    group.bindings.push_back({layout_entry(3, GpuShaderStageBits::Vertex, GpuBindingLayoutType::Buffer,
                                           static_cast<uint8_t>(GpuBufferBindingType::ReadOnlyStorage)),
                              join(3, d3d::DescriptorRangeType::Srv, 0)});
    layout.groups.push_back(group);
    device.register_pipeline_layout(kVs, kPs, layout);
    pso::D3D12GraphicsPsoDesc d = Scene::pso_desc();
    d.vs = kVs;
    d.ps = kPs;
    d.DSVFormat = pso::DxgiFormat::UNKNOWN;
    auto state = device.create_graphics_pipeline_state(d, &root);

    gpu::CommandEncoder enc;
    bridge::CommandListBridge list(device, enc);
    CHECK(list.SetGraphicsRootSignature(&root) == d3d::Status::Ok);
    list.SetPipelineState(state.get());
    CHECK(list.SetGraphicsRootShaderResourceView(0, kBufVa + kViewOffset) == d3d::Status::Ok);
    list.IASetPrimitiveTopology(pso::D3DPrimitiveTopology::TriangleList);
    const d3d::CpuDescriptorHandle rtv0 = rtv_heap->cpu_handle(0);
    list.OMSetRenderTargets(1, &rtv0, false, nullptr);
    list.DrawInstanced(3, 1, 0, 0);
    list.Close();

    const Packet* bind_group = nullptr;
    const auto packets = decode(enc);
    for (const Packet& packet : packets) {
        if (packet.opcode == GpuOpcode::CreateBindGroup2 &&
            packet.payload.size() >= sizeof(GpuCreateBindGroup2Payload) + 2 * sizeof(GpuBindGroupEntry2)) {
            bind_group = &packet;
        }
    }
    CHECK(bind_group != nullptr);
    if (bind_group != nullptr) {
        const auto e0 = pod<GpuBindGroupEntry2>(*bind_group, sizeof(GpuCreateBindGroup2Payload));
        const auto e3 =
            pod<GpuBindGroupEntry2>(*bind_group, sizeof(GpuCreateBindGroup2Payload) + sizeof(GpuBindGroupEntry2));
        CHECK(e0.binding == 0 && e0.resource_id == 777 && e0.offset == 512);
        CHECK(e3.binding == 3 && e3.resource_id == 888 && e3.offset == 1024 && e3.size == 4096 - kViewOffset);
        CHECK(uniform_calls.size() == 1 && uniform_calls[0].first == kBufVa + kViewOffset);
        CHECK(storage_calls.size() == 1 && storage_calls[0].first == kBufVa + kViewOffset &&
              storage_calls[0].second == 4096 - kViewOffset);
        CHECK((list.diagnostics() & (bridge::BridgeDiag::UnalignedStorage | bridge::BridgeDiag::NullBinding)) == 0);
    }
}

// G66 (#691): a cached bind group over a relocated storage copy must follow the copy when the
// device hands its home to another range (LRU eviction, relocation_epoch bumped). Before, the cache
// hit kept binding the old home, so the LEGAL NOTICES draws read another draw's vertex data (grey).
void test_cached_bind_group_follows_moved_relocation() {
    d3d::DescriptorHeapRegistry heaps;
    d3d::ResourceStateRegistry states;
    uint64_t home = 1024, epoch = 0;
    int calls = 0;
    bridge::BridgeOptions options = counting_options(5000);
    options.relocate_storage = [&](uint64_t, uint64_t, uint32_t* id, uint64_t* offset) {
        ++calls;
        *id = 888;
        *offset = home;
        return true;
    };
    options.relocation_epoch = [&epoch] { return epoch; };
    bridge::StateBridgeDevice device(heaps, options, &states);
    constexpr uint32_t kTarget = 100, kBuf = 301;
    constexpr uint64_t kVs = 0xC5, kPs = 0xC6, kBufVa = 0x20000;
    device.register_texture(kTarget, {pso::DxgiFormat::B8G8R8A8_UNORM, 64, 64, 1, 1, 1,
                                      GpuTextureDimension::Dim2D, false});
    device.register_buffer(kBuf, kBufVa, 4096);
    device.register_shader_module(kVs, 1);
    device.register_shader_module(kPs, 2);
    d3d::DescriptorHeap* rtv_heap = make_heap(heaps, d3d::DescriptorHeapType::Rtv, 1, false);
    d3d::DescriptorHeap* srv_heap = make_heap(heaps, d3d::DescriptorHeapType::CbvSrvUav, 4, true);
    CHECK(heaps.create_render_target_view(kTarget, nullptr, rtv_heap->cpu_handle(0)) == d3d::Status::Ok);
    d3d::ViewDesc srv;
    srv.dimension = d3d::ViewDimension::Buffer;
    srv.first_element = 4;  // offset 0x40: not 256-byte aligned, so relocated
    srv.num_elements = 8;
    srv.structure_byte_stride = 16;
    CHECK(heaps.create_shader_resource_view(kBuf, &srv, srv_heap->cpu_handle(0)) == d3d::Status::Ok);
    d3d::RootSignatureDesc rs;
    d3d::RootParameterDesc table;
    table.parameter_type = d3d::RootParameterType::DescriptorTable;
    d3d::DescriptorRangeDesc range;
    range.range_type = d3d::DescriptorRangeType::Srv;
    range.num_descriptors = 1;
    table.ranges.push_back(range);
    rs.parameters.push_back(table);
    d3d::RootSignature root;
    CHECK(d3d::RootSignature::build(rs, &root).ok());
    bridge::PipelineLayoutDesc layout;
    bridge::GroupLayoutDesc group;
    group.bindings.push_back({layout_entry(0, GpuShaderStageBits::Vertex, GpuBindingLayoutType::Buffer,
                                           static_cast<uint8_t>(GpuBufferBindingType::ReadOnlyStorage)),
                              join(0, d3d::DescriptorRangeType::Srv, 0)});
    layout.groups.push_back(group);
    device.register_pipeline_layout(kVs, kPs, layout);
    pso::D3D12GraphicsPsoDesc d = Scene::pso_desc();
    d.vs = kVs;
    d.ps = kPs;
    d.DSVFormat = pso::DxgiFormat::UNKNOWN;
    auto state = device.create_graphics_pipeline_state(d, &root);

    gpu::CommandEncoder enc;
    bridge::CommandListBridge list(device, enc);
    // One recording of the frame; returns the offsets of the bind groups created so far.
    const auto record = [&] {
        list.Reset(state.get());
        d3d::DescriptorHeap* visible[] = {srv_heap};
        CHECK(list.SetDescriptorHeaps(1, visible) == d3d::Status::Ok);
        CHECK(list.SetGraphicsRootSignature(&root) == d3d::Status::Ok);
        CHECK(list.SetGraphicsRootDescriptorTable(0, srv_heap->gpu_handle(0)) == d3d::Status::Ok);
        list.IASetPrimitiveTopology(pso::D3DPrimitiveTopology::TriangleList);
        const gpu::d3d12_xs::D3D12VertexBufferView vb{Scene::kVertexBuffer, 0, 36, 12};
        list.IASetVertexBuffers(0, 1, &vb);
        const d3d::CpuDescriptorHandle rtv0 = rtv_heap->cpu_handle(0);
        list.OMSetRenderTargets(1, &rtv0, false, nullptr);
        list.DrawInstanced(3, 1, 0, 0);
        list.Close();
        std::vector<uint64_t> offsets;
        for (const Packet& packet : decode(enc)) {
            if (packet.opcode == GpuOpcode::CreateBindGroup2 && packet.payload.size() > sizeof(GpuCreateBindGroup2Payload)) {
                offsets.push_back(pod<GpuBindGroupEntry2>(packet, sizeof(GpuCreateBindGroup2Payload)).offset);
            }
        }
        return offsets;
    };
    CHECK((record() == std::vector<uint64_t>{1024}));
    CHECK((record() == std::vector<uint64_t>{1024}));  // cache hit, nothing moved
    epoch = 1;                                          // another range evicted: same home, kept
    CHECK((record() == std::vector<uint64_t>{1024}));
    CHECK(list.stats().bind_groups_relocation_moved == 0);
    home = 2048;  // this range's home handed away: the next lookup relocates it elsewhere
    epoch = 2;
    CHECK((record() == std::vector<uint64_t>{1024, 2048}));
    CHECK(list.stats().bind_groups_relocation_moved == 1);
    CHECK((record() == std::vector<uint64_t>{1024, 2048}));  // the new group is cached again
    CHECK(calls >= 3);
}

// G62: a uniform binding with a reflected minimum size (layout min_binding_size) is bound with at
// least that many bytes, instead of the 256-byte PC minimum; without one the 256-byte rule stays.
GpuBindGroupEntry2 bind_uniform_view(uint64_t view_offset, uint32_t view_size, uint64_t min_size,
                                     std::vector<std::pair<uint64_t, uint64_t>>* calls, uint64_t buffer_size = 4096,
                                     uint64_t allocated_size = 0, bool texture_slot = false) {
    d3d::DescriptorHeapRegistry heaps;
    heaps.set_cbv_alignment(16);
    d3d::ResourceStateRegistry states;
    bridge::BridgeOptions options = counting_options(5000);
    options.relocate_uniform = [calls](uint64_t, uint64_t size, uint64_t binding_size, uint32_t* id, uint64_t* offset) {
        calls->emplace_back(size, binding_size);  // (view bytes copied, binding size)
        *id = 777;
        *offset = 512;
        return true;
    };
    bridge::StateBridgeDevice device(heaps, options, &states);
    constexpr uint32_t kTarget = 100, kCb = 300;
    constexpr uint64_t kVs = 0xC1, kPs = 0xC2, kCbVa = 0x10000;
    device.register_texture(kTarget, {pso::DxgiFormat::B8G8R8A8_UNORM, 64, 64, 1, 1, 1, GpuTextureDimension::Dim2D, false});
    device.register_buffer(kCb, kCbVa, buffer_size, allocated_size);
    device.register_shader_module(kVs, 1);
    device.register_shader_module(kPs, 2);
    d3d::DescriptorHeap* rtv_heap = make_heap(heaps, d3d::DescriptorHeapType::Rtv, 1, false);
    d3d::DescriptorHeap* cbv_heap = make_heap(heaps, d3d::DescriptorHeapType::CbvSrvUav, 4, true);
    CHECK(heaps.create_render_target_view(kTarget, nullptr, rtv_heap->cpu_handle(0)) == d3d::Status::Ok);
    d3d::ConstantBufferViewDesc cbv;
    cbv.buffer_location = kCbVa + view_offset;
    cbv.size_in_bytes = view_size;
    CHECK(heaps.create_constant_buffer_view(&cbv, cbv_heap->cpu_handle(0)) == d3d::Status::Ok);
    d3d::RootSignatureDesc rs;
    d3d::RootParameterDesc table;
    table.parameter_type = d3d::RootParameterType::DescriptorTable;
    d3d::DescriptorRangeDesc range;
    range.range_type = d3d::DescriptorRangeType::Cbv;
    range.num_descriptors = 1;
    table.ranges.push_back(range);
    rs.parameters.push_back(table);
    d3d::RootSignature root;
    CHECK(d3d::RootSignature::build(rs, &root).ok());
    bridge::PipelineLayoutDesc layout;
    bridge::GroupLayoutDesc group;
    GpuBindGroupLayoutEntry entry =
        texture_slot ? layout_entry(0, GpuShaderStageBits::Vertex | GpuShaderStageBits::Fragment,
                                    GpuBindingLayoutType::Texture, static_cast<uint8_t>(GpuTextureSampleType::Float))
                     : layout_entry(0, GpuShaderStageBits::Vertex | GpuShaderStageBits::Fragment,
                                    GpuBindingLayoutType::Buffer, static_cast<uint8_t>(GpuBufferBindingType::Uniform));
    entry.min_binding_size = min_size;
    group.bindings.push_back({entry, join(0, d3d::DescriptorRangeType::Cbv, 0)});
    layout.groups.push_back(group);
    device.register_pipeline_layout(kVs, kPs, layout);
    pso::D3D12GraphicsPsoDesc d = Scene::pso_desc();
    d.vs = kVs;
    d.ps = kPs;
    d.DSVFormat = pso::DxgiFormat::UNKNOWN;
    auto state = device.create_graphics_pipeline_state(d, &root);
    gpu::CommandEncoder enc;
    bridge::CommandListBridge list(device, enc);
    d3d::DescriptorHeap* visible[] = {cbv_heap};
    CHECK(list.SetDescriptorHeaps(1, visible) == d3d::Status::Ok);
    CHECK(list.SetGraphicsRootSignature(&root) == d3d::Status::Ok);
    list.SetPipelineState(state.get());
    CHECK(list.SetGraphicsRootDescriptorTable(0, cbv_heap->gpu_handle(0)) == d3d::Status::Ok);
    list.IASetPrimitiveTopology(pso::D3DPrimitiveTopology::TriangleList);
    const gpu::d3d12_xs::D3D12VertexBufferView vb{Scene::kVertexBuffer, 0, 36, 12};
    list.IASetVertexBuffers(0, 1, &vb);
    const d3d::CpuDescriptorHandle rtv0 = rtv_heap->cpu_handle(0);
    list.OMSetRenderTargets(1, &rtv0, false, nullptr);
    list.DrawInstanced(3, 1, 0, 0);
    list.Close();
    GpuBindGroupEntry2 out{};
    for (const Packet& packet : decode(enc)) {
        if (packet.opcode == GpuOpcode::CreateBindGroup2 && packet.payload.size() > sizeof(GpuCreateBindGroup2Payload)) {
            out = pod<GpuBindGroupEntry2>(packet, sizeof(GpuCreateBindGroup2Payload));
        }
    }
    return out;
}

void test_reflected_uniform_min_size() {
    std::vector<std::pair<uint64_t, uint64_t>> calls;
    // Aligned view of 96 bytes: no minimum -> the rest of the buffer (the block size is unknown;
    // 4096 - 0x100 bytes follow, as on the console); minimum 160 -> 160; minimum 64 -> 96.
    CHECK(bind_uniform_view(0x100, 96, 0, &calls).size == 4096 - 0x100);
    CHECK(bind_uniform_view(0x100, 96, 160, &calls).size == 160);
    const GpuBindGroupEntry2 within = bind_uniform_view(0x100, 96, 64, &calls);
    CHECK(within.size == 96 && within.offset == 0x100 && within.resource_id == 300);
    CHECK(calls.empty());
    // Unaligned view relocated: the copy holds the view's 96 bytes and covers the minimum (320
    // bytes -> a 512-byte home, zero past the view).
    const GpuBindGroupEntry2 moved = bind_uniform_view(0x40, 96, 320, &calls);
    CHECK(moved.resource_id == 777 && moved.size == 512);
    CHECK(calls.size() == 1 && calls[0].first == 96 && calls[0].second == 320);
    // G66: an aligned 16-byte view at the end of its 4 KiB buffer, for a 128-byte block: no room
    // to extend, so it is bound as a relocated copy of its 16 bytes, zero-extended to the block.
    calls.clear();
    const GpuBindGroupEntry2 tail = bind_uniform_view(0xF00, 16, 512, &calls);
    CHECK(tail.resource_id == 777 && tail.offset == 512 && tail.size == 512);
    CHECK(calls.size() == 1 && calls[0].first == 16 && calls[0].second == 512);
    // No reflected minimum: the 256-byte rule, extended in place when the buffer has the room
    // (here exactly 256 bytes are left) ...
    calls.clear();
    const GpuBindGroupEntry2 room = bind_uniform_view(0xF00, 16, 0, &calls);
    CHECK(room.resource_id == 300 && room.offset == 0xF00 && room.size == 256 && calls.empty());
    // ... and with less room than the block, relocated and zero-extended.
    const GpuBindGroupEntry2 short_room = bind_uniform_view(0xF00, 16, 384, &calls);
    CHECK(short_room.resource_id == 777 && short_room.size == 512);
    CHECK(calls.size() == 1 && calls[0].first == 16 && calls[0].second == 384);
    // G66: a whole 16-byte buffer whose WebGPU buffer is padded to 256 bytes (Resource::initialize)
    // binds in place for a 128-byte block (and for the 256-byte rule): the padding is zero.
    calls.clear();
    const GpuBindGroupEntry2 padded = bind_uniform_view(0, 16, 128, &calls, 16, 256);
    CHECK(padded.resource_id == 300 && padded.offset == 0 && padded.size == 128 && calls.empty());
    const GpuBindGroupEntry2 padded_pc = bind_uniform_view(0, 16, 0, &calls, 16, 256);
    CHECK(padded_pc.resource_id == 300 && padded_pc.size == 256 && calls.empty());
    // Without the padding the same view has no room: relocated and zero-extended.
    const GpuBindGroupEntry2 unpadded = bind_uniform_view(0, 16, 128, &calls, 16, 0);
    CHECK(unpadded.resource_id == 777 && unpadded.size == 256);
    CHECK(calls.size() == 1 && calls[0].first == 16 && calls[0].second == 128);
    // A CBV in a slot the conventional layout calls a texture (the translated stage reads it as a
    // uniform block of unknown size): 256 bytes, in place in a padded buffer, else relocated.
    calls.clear();
    const GpuBindGroupEntry2 cbv_in_place = bind_uniform_view(0, 16, 0, &calls, 16, 256, true);
    CHECK(cbv_in_place.resource_id == 300 && cbv_in_place.offset == 0 && cbv_in_place.size == 256 && calls.empty());
    const GpuBindGroupEntry2 cbv_moved = bind_uniform_view(0, 16, 0, &calls, 16, 0, true);
    CHECK(cbv_moved.resource_id == 777 && cbv_moved.size == 256);
    CHECK(calls.size() == 1 && calls[0].first == 16 && calls[0].second == 256);
}

// G48: a slot source over test-owned "descriptor memory" for one heap.
class TestSlotSource final : public d3d::DescriptorSlotSource {
public:
    TestSlotSource(const bridge::StateBridgeDevice& device, const d3d::DescriptorHeap* heap)
        : device_(device), heap_(heap), memory_(size_t{heap->capacity()} * heap->increment(), 0) {}
    bool reads(const d3d::DescriptorHeap& heap) const override { return &heap == heap_; }
    bool snapshot(const d3d::DescriptorHeap&, uint32_t first, uint32_t count, std::vector<uint8_t>* out) const override {
        const size_t inc = heap_->increment();
        out->assign(memory_.begin() + first * inc, memory_.begin() + (first + count) * inc);
        return true;
    }
    d3d::ViewDescriptor view(const d3d::DescriptorHeap&, uint32_t index, d3d::DescriptorRangeType range) const override {
        d3d::ViewDescriptor v;
        last_status = bridge::decode_view_words(device_, words(index), range, &v);
        return v;
    }
    bool sampler(const d3d::DescriptorHeap&, uint32_t index, d3d::SamplerDesc* out) const override {
        return bridge::decode_sampler_words(words(index), out) == bridge::DecodeStatus::Ok;
    }
    void write(uint32_t index, const uint32_t* w, size_t bytes) {
        std::memcpy(memory_.data() + size_t{index} * heap_->increment(), w, bytes);
    }
    const uint32_t* words(uint32_t index) const {
        return reinterpret_cast<const uint32_t*>(memory_.data() + size_t{index} * heap_->increment());
    }
    mutable bridge::DecodeStatus last_status = bridge::DecodeStatus::Ok;

private:
    const bridge::StateBridgeDevice& device_;
    const d3d::DescriptorHeap* heap_;
    std::vector<uint8_t> memory_;
};

GpuBindGroupEntry2 last_bind_group_entry(const gpu::CommandEncoder& enc, uint32_t* id = nullptr) {
    GpuBindGroupEntry2 entry{};
    for (const Packet& packet : decode(enc)) {
        if (packet.opcode == GpuOpcode::CreateBindGroup2 && packet.payload.size() > sizeof(GpuCreateBindGroup2Payload)) {
            entry = pod<GpuBindGroupEntry2>(packet, sizeof(GpuCreateBindGroup2Payload));
            if (id != nullptr) {
                *id = pod<GpuCreateBindGroup2Payload>(packet).id;
            }
        }
    }
    return entry;
}

// G48: root tables read from descriptor memory, revalidated by their words, refreshed at execution.
void test_memory_backed_tables() {
    d3d::DescriptorHeapRegistry heaps;
    heaps.set_cbv_alignment(16);
    d3d::ResourceStateRegistry states;
    bridge::BridgeOptions options = counting_options(6000);
    // The slot source needs the device; it is installed through the options before the device exists.
    struct Holder {
        std::unique_ptr<TestSlotSource> source;
    } holder;
    struct Forward final : d3d::DescriptorSlotSource {
        Holder* h = nullptr;
        bool reads(const d3d::DescriptorHeap& heap) const override { return h->source && h->source->reads(heap); }
        bool snapshot(const d3d::DescriptorHeap& heap, uint32_t f, uint32_t c, std::vector<uint8_t>* o) const override {
            return h->source->snapshot(heap, f, c, o);
        }
        d3d::ViewDescriptor view(const d3d::DescriptorHeap& heap, uint32_t i, d3d::DescriptorRangeType r) const override {
            return h->source->view(heap, i, r);
        }
        bool sampler(const d3d::DescriptorHeap& heap, uint32_t i, d3d::SamplerDesc* o) const override {
            return h->source->sampler(heap, i, o);
        }
    } forward;
    forward.h = &holder;
    options.slot_source = &forward;
    constexpr uint32_t kTarget = 100, kCb = 300, kTexture = 301, kVsModule = 1, kFsModule = 2;
    // #663: kCb stands in for a buffer of relocation homes bound straight from descriptor memory.
    uint64_t relocation_epoch = 0;
    options.relocation_epoch = [&relocation_epoch] { return relocation_epoch; };
    options.relocation_home = [](uint32_t buffer_id) { return buffer_id == kCb; };
    bridge::StateBridgeDevice device(heaps, options, &states);
    constexpr uint64_t kVs = 0xD1, kPs = 0xD2, kCbVa = 0x10000, kTextureVa = 0x7E00'0001'0000ull;
    device.register_texture(kTarget, {pso::DxgiFormat::B8G8R8A8_UNORM, 64, 64, 1, 1, 1, GpuTextureDimension::Dim2D, false});
    bridge::TextureInfo tex{pso::DxgiFormat::R8G8B8A8_UNORM, 128, 64, 1, 8, 1, GpuTextureDimension::Dim2D, false};
    tex.gpu_va = kTextureVa;
    device.register_texture(kTexture, tex);
    device.register_buffer(kCb, kCbVa, 4096);
    device.register_shader_module(kVs, kVsModule);
    device.register_shader_module(kPs, kFsModule);
    d3d::DescriptorHeap* rtv_heap = make_heap(heaps, d3d::DescriptorHeapType::Rtv, 1, false);
    d3d::DescriptorHeap* heap = make_heap(heaps, d3d::DescriptorHeapType::CbvSrvUav, 8, true);
    holder.source = std::make_unique<TestSlotSource>(device, heap);
    TestSlotSource& memory = *holder.source;
    CHECK(heaps.create_render_target_view(kTarget, nullptr, rtv_heap->cpu_handle(0)) == d3d::Status::Ok);

    // Decoding: words of API-created views read back as the same bindings.
    {
        uint32_t w[8];
        d3d::ViewDescriptor cbv{};
        cbv.kind = d3d::DescriptorKind::Cbv;
        cbv.first_element_or_address = kCbVa + 0x40;
        cbv.num_elements_or_size = 96;
        CHECK(bridge::view_words(device, cbv, w) == bridge::WordsStatus::Ok);
        d3d::ViewDescriptor back;
        CHECK(bridge::decode_view_words(device, w, d3d::DescriptorRangeType::Cbv, &back) == bridge::DecodeStatus::Ok);
        CHECK(back.kind == d3d::DescriptorKind::Cbv && back.first_element_or_address == kCbVa + 0x40 &&
              back.num_elements_or_size == 96);
        d3d::ViewDescriptor srv{};
        srv.kind = d3d::DescriptorKind::Srv;
        srv.resource_id = kCb;
        srv.dimension = d3d::ViewDimension::Buffer;
        srv.first_element_or_address = 4;
        srv.num_elements_or_size = 8;
        srv.structure_byte_stride = 16;
        CHECK(bridge::view_words(device, srv, w) == bridge::WordsStatus::Ok);
        CHECK(bridge::decode_view_words(device, w, d3d::DescriptorRangeType::Srv, &back) == bridge::DecodeStatus::Ok);
        CHECK(back.resource_id == kCb && back.structure_byte_stride == 1 && back.first_element_or_address == 64 &&
              back.num_elements_or_size == 128);
        d3d::ViewDescriptor img{};
        img.kind = d3d::DescriptorKind::Srv;
        img.resource_id = kTexture;
        img.dimension = d3d::ViewDimension::Texture2D;
        img.format = 29;  // sRGB view of an R8G8B8A8_UNORM texture
        img.component_mapping = d3d::kDefaultShader4ComponentMapping;
        img.first_mip = 2;
        img.mip_levels = 3;
        CHECK(bridge::view_words(device, img, w) == bridge::WordsStatus::Ok);
        CHECK(bridge::decode_view_words(device, w, d3d::DescriptorRangeType::Srv, &back) == bridge::DecodeStatus::Ok);
        CHECK(back.resource_id == kTexture && back.format == 29 && back.dimension == d3d::ViewDimension::Texture2D &&
              back.first_mip == 2 && back.mip_levels == 3 && back.component_mapping == d3d::kDefaultShader4ComponentMapping);
        CHECK(bridge::decode_view_words(device, w, d3d::DescriptorRangeType::Cbv, &back) == bridge::DecodeStatus::KindMismatch);
        const uint32_t link[8] = {7, 0, 0, 0, 0, 0, 0, 0};
        CHECK(bridge::decode_view_words(device, link, d3d::DescriptorRangeType::Srv, &back) ==
                  bridge::DecodeStatus::NotDescriptor &&
              back.resource_id == 0);
        uint32_t far[4];
        bridge::constant_buffer_words(0x9'0000'0000ull, 64, far);
        const uint32_t far8[8] = {far[0], far[1], far[2], far[3], 0, 0, 0, 0};
        CHECK(bridge::decode_view_words(device, far8, d3d::DescriptorRangeType::Srv, &back) ==
              bridge::DecodeStatus::UnresolvedAddress);
        uint32_t tw[8];
        bridge::AddressTexture at;
        at.width = 64;
        at.height = 64;
        at.format = 28;
        CHECK(bridge::image_words_at_address(0x9'0000'0000ull, at, img, tw) == bridge::WordsStatus::Ok);
        CHECK(bridge::decode_view_words(device, tw, d3d::DescriptorRangeType::Srv, &back) ==
              bridge::DecodeStatus::UnknownTexture);
        uint32_t sw[4];
        d3d::SamplerDesc s;
        s.filter = 0x15;
        s.address_u = 3;
        bridge::sampler_words(s, sw);
        d3d::SamplerDesc sb;
        CHECK(bridge::decode_sampler_words(sw, &sb) == bridge::DecodeStatus::Ok && sb.filter == 0x15 && sb.address_u == 3);
    }

    // A CBV table whose slot only exists in memory (the title copied it there).
    uint32_t cb[4];
    bridge::constant_buffer_words(kCbVa + 0x400, 96, cb);
    memory.write(3, cb, sizeof(cb));
    CHECK(heap->view(3)->kind == d3d::DescriptorKind::Empty);  // nothing in the registry
    d3d::RootSignatureDesc rs;
    d3d::RootParameterDesc table;
    table.parameter_type = d3d::RootParameterType::DescriptorTable;
    d3d::DescriptorRangeDesc range;
    range.range_type = d3d::DescriptorRangeType::Cbv;
    range.num_descriptors = 1;
    table.ranges.push_back(range);
    rs.parameters.push_back(table);
    d3d::RootSignature root;
    CHECK(d3d::RootSignature::build(rs, &root).ok());
    bridge::PipelineLayoutDesc layout;
    bridge::GroupLayoutDesc group;
    group.bindings.push_back({layout_entry(0, GpuShaderStageBits::Vertex | GpuShaderStageBits::Fragment,
                                           GpuBindingLayoutType::Buffer, static_cast<uint8_t>(GpuBufferBindingType::Uniform)),
                              join(0, d3d::DescriptorRangeType::Cbv, 0)});
    layout.groups.push_back(group);
    device.register_pipeline_layout(kVs, kPs, layout);
    pso::D3D12GraphicsPsoDesc d = Scene::pso_desc();
    d.vs = kVs;
    d.ps = kPs;
    d.DSVFormat = pso::DxgiFormat::UNKNOWN;
    auto state = device.create_graphics_pipeline_state(d, &root);

    gpu::CommandEncoder enc;
    bridge::CommandListBridge list(device, enc);
    d3d::DescriptorHeap* visible[] = {heap};
    CHECK(list.SetDescriptorHeaps(1, visible) == d3d::Status::Ok);
    CHECK(list.SetGraphicsRootSignature(&root) == d3d::Status::Ok);
    list.SetPipelineState(state.get());
    CHECK(list.SetGraphicsRootDescriptorTable(0, heap->gpu_handle(3)) == d3d::Status::Ok);
    list.IASetPrimitiveTopology(pso::D3DPrimitiveTopology::TriangleList);
    const gpu::d3d12_xs::D3D12VertexBufferView vb{Scene::kVertexBuffer, 0, 36, 12};
    list.IASetVertexBuffers(0, 1, &vb);
    const d3d::CpuDescriptorHandle rtv0 = rtv_heap->cpu_handle(0);
    list.OMSetRenderTargets(1, &rtv0, false, nullptr);
    list.DrawInstanced(3, 1, 0, 0);
    list.DrawInstanced(3, 1, 0, 0);  // same words: same bind group
    list.Close();
    uint32_t id = 0;
    GpuBindGroupEntry2 e = last_bind_group_entry(enc, &id);
    // (A 256-byte aligned view: bound in place, widened through the buffer as below; an unaligned
    // one without a relocate hook binds null, see test_unaligned_uniform_relocation.)
    CHECK(e.resource_id == kCb && e.offset == 0x400 && e.size == 4096 - 0x400);
    CHECK(list.stats().bind_groups_created == 1 && list.stats().memory_bind_groups == 1);

    // Unchanged at execution: nothing to do. Changed: the recorded entries are rewritten in place.
    gpu::CommandEncoder prelude;
    CHECK(list.refresh_memory_bind_groups(prelude) == 0);
    bridge::constant_buffer_words(kCbVa + 0x100, 32, cb);
    memory.write(3, cb, sizeof(cb));
    const size_t before = enc.size();
    CHECK(list.refresh_memory_bind_groups(prelude) == 1);
    CHECK(enc.size() == before && prelude.size() == 0);
    uint32_t id2 = 0;
    e = last_bind_group_entry(enc, &id2);
    // Widened through to the end of the 4 KiB buffer (no reflected block size: the bytes that follow).
    CHECK(id2 == id && e.resource_id == kCb && e.offset == 0x100 && e.size == 4096 - 0x100);
    CHECK(list.refresh_memory_bind_groups(prelude) == 0);
    // #663: same words, but a relocation home was handed to another range: resolved again once.
    ++relocation_epoch;
    CHECK(list.refresh_memory_bind_groups(prelude) == 1);
    CHECK(list.refresh_memory_bind_groups(prelude) == 0);

    // Words the runtime cannot read as a descriptor bind null.
    const uint32_t link[8] = {9, 0, 0, 0, 0, 0, 0, 0};
    memory.write(3, link, sizeof(link));
    CHECK(list.refresh_memory_bind_groups(prelude) == 1);
    e = last_bind_group_entry(enc);
    CHECK(e.resource_id == 0 && memory.last_status == bridge::DecodeStatus::NotDescriptor);

    // A new recording over the same table: the cached group is revalidated by its words.
    bridge::constant_buffer_words(kCbVa + 0x200, 16, cb);
    memory.write(3, cb, sizeof(cb));
    gpu::CommandEncoder enc2;
    bridge::CommandListBridge list2(device, enc2);
    CHECK(list2.SetDescriptorHeaps(1, visible) == d3d::Status::Ok);
    CHECK(list2.SetGraphicsRootSignature(&root) == d3d::Status::Ok);
    list2.SetPipelineState(state.get());
    CHECK(list2.SetGraphicsRootDescriptorTable(0, heap->gpu_handle(3)) == d3d::Status::Ok);
    list2.IASetPrimitiveTopology(pso::D3DPrimitiveTopology::TriangleList);
    list2.IASetVertexBuffers(0, 1, &vb);
    list2.OMSetRenderTargets(1, &rtv0, false, nullptr);
    list2.DrawInstanced(3, 1, 0, 0);
    bridge::constant_buffer_words(kCbVa + 0x300, 16, cb);
    memory.write(3, cb, sizeof(cb));
    list2.DrawInstanced(3, 1, 0, 0);  // words changed between draws: rebuilt under a new id
    list2.Close();
    CHECK(list2.stats().bind_groups_created == 2 && list2.stats().memory_bind_groups == 2);
    e = last_bind_group_entry(enc2);
    CHECK(e.offset == 0x300);
}

// ------------------------------------------------------------------------------------------------
// G49b: Dirty-tracked guest texture memory upload (WriteTexture) & detile hook
// ------------------------------------------------------------------------------------------------

std::vector<Packet> decode_bytes(const std::vector<uint8_t>& buf) {
    std::vector<Packet> out;
    size_t off = 0;
    while (off + sizeof(GpuPacketHeader) <= buf.size()) {
        GpuPacketHeader h{};
        std::memcpy(&h, buf.data() + off, sizeof(h));
        off += sizeof(h);
        Packet p;
        p.opcode = static_cast<GpuOpcode>(h.opcode);
        p.payload.assign(buf.data() + off, buf.data() + off + h.payload_bytes);
        off += h.payload_bytes;
        out.push_back(std::move(p));
    }
    CHECK(off == buf.size());
    return out;
}

void test_dirty_texture_memory_upload() {
    namespace com = playgta6::guest::com;
    namespace dg = playgta6::gpu::d3d12_guest;

    os::VirtualMemoryArena arena;
    com::reset_com_bridge();
    com::set_guest_memory(&arena);

    {
        std::vector<uint8_t> submitted;
        dg::DeviceOptions dev_opts;
        dev_opts.sink = [&submitted](const uint8_t* data, size_t size) {
            submitted.insert(submitted.end(), data, data + size);
        };
        dev_opts.defer_gpu_objects = false;  // the create packets are checked at creation
        auto device = com::create_object<dg::Device>(com::implements(dg::kID3D12Device), dev_opts);
        CHECK(device);
        device->initialize_guest_payload();

        auto make_tex_desc = [](uint32_t w, uint32_t h, uint32_t fmt, uint32_t layout, uint32_t flags = 0) {
            dg::GuestResourceDesc d{};
            d.Dimension = dg::ResourceDimension::Texture2D;
            d.Width = w;
            d.Height = h;
            d.DepthOrArraySize = 1;
            d.MipLevels = 1;
            d.Format = fmt;
            d.SampleCount = 1;
            d.Layout = layout;
            d.Flags = flags;
            return d;
        };

        // 1) Allocate guest memory for a 4x4 RGBA8 linear texture (layout 0) and create address-backed Resource.
        const uint64_t tex_mem =
            arena.virtual_alloc(0, 65536, os::kMemReserve | os::kMemCommit, os::PageProtect::ReadWrite);
        CHECK(tex_mem != 0);
        dg::Resource::Placement pl_addr{};
        pl_addr.address = tex_mem;
        auto tex = com::create_object<dg::Resource>(com::implements(dg::kID3D12Resource), dg::device_ref(device.get()),
                                                    make_tex_desc(4, 4, 28 /*R8G8B8A8_UNORM*/, 0), pl_addr, 0u);
        CHECK(tex && tex->initialize());

        // Verify CreateTexture2 emitted with TEXTURE_BINDING | COPY_DST usage.
        bool saw_create_tex2 = false;
        for (const Packet& pk : decode_bytes(submitted)) {
            if (pk.opcode == GpuOpcode::CreateTexture2) {
                const auto ct = pod<GpuCreateTexture2Payload>(pk);
                if (ct.id == tex->id()) {
                    saw_create_tex2 = true;
                    CHECK((ct.usage_flags & GpuTextureUsage::TextureBinding) != 0);
                    CHECK((ct.usage_flags & GpuTextureUsage::CopyDst) != 0);
                }
            }
        }
        CHECK(saw_create_tex2);

        // Set up a render target, SRV heap, root signature, pipeline layout, and PSO on the device's state bridge.
        dg::Resource::Placement pl_rt{};
        pl_rt.heap.Type = dg::HeapType::Default;
        auto rt = com::create_object<dg::Resource>(
            com::implements(dg::kID3D12Resource), dg::device_ref(device.get()),
            make_tex_desc(64, 64, 87 /*B8G8R8A8_UNORM*/, 0, dg::ResourceFlags::AllowRenderTarget), pl_rt, 0u);
        CHECK(rt && rt->initialize());

        d3d::DescriptorHeap* rtv_heap = make_heap(device->heaps(), d3d::DescriptorHeapType::Rtv, 2, false);
        d3d::DescriptorHeap* srv_heap = make_heap(device->heaps(), d3d::DescriptorHeapType::CbvSrvUav, 8, true);
        CHECK(device->heaps().create_render_target_view(rt->id(), nullptr, rtv_heap->cpu_handle(0)) == d3d::Status::Ok);

        d3d::ViewDesc srv{};
        srv.format = 28;
        srv.dimension = d3d::ViewDimension::Texture2D;
        srv.first_mip = 0;
        srv.mip_levels = 1;
        CHECK(device->heaps().create_shader_resource_view(tex->id(), &srv, srv_heap->cpu_handle(0)) == d3d::Status::Ok);

        constexpr uint64_t kVs = 0x49B1, kPs = 0x49B2, kCs = 0x49B3;
        device->register_shader_module(kVs, 11);
        device->register_shader_module(kPs, 12);
        device->register_shader_module(kCs, 13);

        d3d::RootSignatureDesc rs_desc;
        d3d::RootParameterDesc table;
        table.parameter_type = d3d::RootParameterType::DescriptorTable;
        d3d::DescriptorRangeDesc range;
        range.range_type = d3d::DescriptorRangeType::Srv;
        range.num_descriptors = 1;
        table.ranges.push_back(range);
        rs_desc.parameters.push_back(table);
        d3d::RootSignature root_sig;
        CHECK(d3d::RootSignature::build(rs_desc, &root_sig).ok());

        bridge::PipelineLayoutDesc gfx_layout;
        bridge::GroupLayoutDesc gfx_group;
        gfx_group.bindings.push_back({layout_entry(0, GpuShaderStageBits::Fragment, GpuBindingLayoutType::Texture,
                                                   static_cast<uint8_t>(GpuTextureSampleType::Float),
                                                   static_cast<uint8_t>(GpuTextureViewDimension::Dim2D)),
                                      join(0, d3d::DescriptorRangeType::Srv, 0)});
        gfx_layout.groups.push_back(gfx_group);
        device->state_bridge().register_pipeline_layout(kVs, kPs, gfx_layout);

        bridge::PipelineLayoutDesc cmp_layout;
        bridge::GroupLayoutDesc cmp_group;
        cmp_group.bindings.push_back({layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Texture,
                                                   static_cast<uint8_t>(GpuTextureSampleType::Float),
                                                   static_cast<uint8_t>(GpuTextureViewDimension::Dim2D)),
                                      join(0, d3d::DescriptorRangeType::Srv, 0)});
        cmp_layout.groups.push_back(cmp_group);
        device->state_bridge().register_pipeline_layout(kCs, 0, cmp_layout);

        pso::D3D12GraphicsPsoDesc pso_d = Scene::pso_desc();
        pso_d.vs = kVs;
        pso_d.ps = kPs;
        pso_d.DSVFormat = pso::DxgiFormat::UNKNOWN;
        auto gfx_pso = device->state_bridge().create_graphics_pipeline_state(pso_d, &root_sig);
        CHECK(gfx_pso && gfx_pso->diagnostics() == 0);

        pso::D3D12ComputePsoDesc cpso_d{};
        cpso_d.cs = kCs;
        cpso_d.root_signature_id = 77;
        auto cmp_pso = device->state_bridge().create_compute_pipeline_state(cpso_d, &root_sig);
        CHECK(cmp_pso && cmp_pso->diagnostics() == 0);

        gpu::CommandEncoder enc;
        bridge::CommandListBridge list(device->state_bridge(), enc);
        d3d::DescriptorHeap* visible[] = {srv_heap};
        CHECK(list.SetDescriptorHeaps(1, visible) == d3d::Status::Ok);
        CHECK(list.SetGraphicsRootSignature(&root_sig) == d3d::Status::Ok);
        list.SetPipelineState(gfx_pso.get());
        CHECK(list.SetGraphicsRootDescriptorTable(0, srv_heap->gpu_handle(0)) == d3d::Status::Ok);
        list.IASetPrimitiveTopology(pso::D3DPrimitiveTopology::TriangleList);
        const gpu::d3d12_xs::D3D12VertexBufferView vb{200, 0, 36, 12};
        list.IASetVertexBuffers(0, 1, &vb);
        const d3d::CpuDescriptorHandle rtv0 = rtv_heap->cpu_handle(0);
        list.OMSetRenderTargets(1, &rtv0, false, nullptr);

        auto count_write_textures = [&](size_t from) {
            std::vector<Packet> out;
            for (const Packet& pk : decode(enc, from)) {
                if (pk.opcode == GpuOpcode::WriteTexture) {
                    out.push_back(pk);
                }
            }
            return out;
        };

        // Case A1: All-zero guest texture memory that has never been uploaded -> 0 WriteTexture packets.
        size_t mark = enc.size();
        list.DrawInstanced(3, 1, 0, 0);
        CHECK(count_write_textures(mark).empty());
        CHECK(device->stats().textures_uploaded == 0);
        CHECK(device->stats().upload_bytes == 0);

        // Case A2: Write non-zero 4x4 RGBA8 texels -> 1 WriteTexture packet with 256-byte aligned row pitch.
        std::array<uint8_t, 64> texels{};
        for (size_t i = 0; i < texels.size(); ++i) {
            texels[i] = static_cast<uint8_t>(0x10u + i);
        }
        CHECK(com::write_guest(tex_mem, texels.data(), texels.size()));
        mark = enc.size();
        list.DrawInstanced(3, 1, 0, 0);
        auto wt1 = count_write_textures(mark);
        CHECK(wt1.size() == 1);
        if (wt1.size() == 1) {
            const auto wt = pod<GpuWriteTexturePayload>(wt1[0]);
            CHECK(wt.texture_id == tex->id());
            CHECK(wt.mip_level == 0 && wt.origin_x == 0 && wt.origin_y == 0 && wt.origin_z == 0);
            CHECK(wt.width == 4 && wt.height == 4 && wt.depth_or_layers == 1);
            CHECK(wt.bytes_per_row == 256 && (wt.bytes_per_row % 256) == 0);
            CHECK(wt.rows_per_image == 4 && wt.data_len == 1024);
            CHECK(wt1[0].payload.size() == sizeof(GpuWriteTexturePayload) + 1024);
            const uint8_t* rows = wt1[0].payload.data() + sizeof(GpuWriteTexturePayload);
            CHECK(std::memcmp(rows + 0 * 256, texels.data() + 0 * 16, 16) == 0);
            CHECK(std::memcmp(rows + 3 * 256, texels.data() + 3 * 16, 16) == 0);
        }
        CHECK(device->stats().textures_uploaded == 1);
        CHECK(device->stats().upload_bytes == 1024);
        dg::TrackedTexture info{};
        CHECK(device->tracked_texture_info(tex->id(), &info) && info.ever_uploaded && info.last_upload_hash != 0);

        // Case A3: Unchanged memory on subsequent draw -> skips re-upload.
        mark = enc.size();
        list.DrawInstanced(3, 1, 0, 0);
        CHECK(count_write_textures(mark).empty());
        CHECK(device->stats().textures_uploaded == 1);

        // Case A4: Modified guest memory -> triggers a 2nd WriteTexture upload.
        texels[0] = 0xFE;
        CHECK(com::write_guest(tex_mem, texels.data(), texels.size()));
        mark = enc.size();
        list.DrawInstanced(3, 1, 0, 0);
        auto wt2 = count_write_textures(mark);
        CHECK(wt2.size() == 1);
        if (wt2.size() == 1) {
            const uint8_t* rows = wt2[0].payload.data() + sizeof(GpuWriteTexturePayload);
            CHECK(rows[0] == 0xFE);
        }
        CHECK(device->stats().textures_uploaded == 2);
        CHECK(device->stats().upload_bytes == 2048);

        // Case B: Compute dispatch with modified texture triggers WriteTexture as well.
        texels[1] = 0xCC;
        CHECK(com::write_guest(tex_mem, texels.data(), texels.size()));
        CHECK(list.SetComputeRootSignature(&root_sig) == d3d::Status::Ok);
        list.SetComputePipelineState(cmp_pso.get());
        CHECK(list.SetComputeRootDescriptorTable(0, srv_heap->gpu_handle(0)) == d3d::Status::Ok);
        mark = enc.size();
        list.Dispatch(1, 1, 1);
        CHECK(count_write_textures(mark).size() == 1);
        CHECK(device->stats().textures_uploaded == 3);
        mark = enc.size();
        list.Dispatch(1, 1, 1);
        CHECK(count_write_textures(mark).empty());
        list.Close();

        // Case C: GPU-window alias translation + guard-page / unmapped safety.
        const uint64_t cpu_win =
            arena.virtual_alloc(0, 65536, os::kMemReserve | os::kMemCommit, os::PageProtect::ReadWrite);
        const uint64_t gpu_win = 0x0000'5000'0000'0000ull;
        device->register_gpu_alias(gpu_win, cpu_win, 65536);
        dg::Resource::Placement pl_alias{};
        pl_alias.address = gpu_win;
        auto alias_tex = com::create_object<dg::Resource>(
            com::implements(dg::kID3D12Resource), dg::device_ref(device.get()),
            make_tex_desc(2, 2, 28, 0), pl_alias, 0u);
        CHECK(alias_tex && alias_tex->initialize());
        const uint32_t px[4] = {0xFF0000FFu, 0xFF00FF00u, 0xFFFF0000u, 0xFFFFFFFFu};
        CHECK(com::write_guest(cpu_win, px, sizeof(px)));
        gpu::CommandEncoder alias_enc;
        CHECK(device->sync_bound_texture_memory(alias_tex->id(), alias_enc));
        CHECK(decode(alias_enc).size() == 1 && decode(alias_enc)[0].opcode == GpuOpcode::WriteTexture);
        // Once a recorded copy has written the texture, its memory is not uploaded (or de-tiled)
        // over the copied texels, even when that memory changes.
        const uint32_t px2[4] = {1u, 2u, 3u, 4u};
        CHECK(com::write_guest(cpu_win, px2, sizeof(px2)));
        device->note_texture_copy_destination(alias_tex->id());
        gpu::CommandEncoder alias_enc2;
        CHECK(!device->sync_bound_texture_memory(alias_tex->id(), alias_enc2));
        CHECK(alias_enc2.size() == 0);

        // Guard / PAGE_NOACCESS page: must safely skip without crashing.
        const uint64_t guard_mem =
            arena.virtual_alloc(0, 65536, os::kMemReserve | os::kMemCommit, os::PageProtect::NoAccess);
        CHECK(guard_mem != 0);
        dg::Resource::Placement pl_guard{};
        pl_guard.address = guard_mem;
        auto guard_tex = com::create_object<dg::Resource>(
            com::implements(dg::kID3D12Resource), dg::device_ref(device.get()),
            make_tex_desc(4, 4, 28, 0), pl_guard, 0u);
        CHECK(guard_tex && guard_tex->initialize());
        gpu::CommandEncoder guard_enc;
        CHECK(!device->sync_bound_texture_memory(guard_tex->id(), guard_enc));
        CHECK(guard_enc.size() == 0);

        // Case D: Tiled console layouts (layout >= 0x101) — default Device detiles via GFX10 detiler;
        // clearing DetileFn skips non-tiny tiled textures, and custom DetileFn overrides work.
        const uint64_t tiled_mem =
            arena.virtual_alloc(0, 65536, os::kMemReserve | os::kMemCommit, os::PageProtect::ReadWrite);
        CHECK(com::write_guest(tiled_mem, texels.data(), texels.size()));

        dg::Resource::Placement pl_tiled_tiny{};
        pl_tiled_tiny.address = tiled_mem;
        auto tiled_tiny = com::create_object<dg::Resource>(
            com::implements(dg::kID3D12Resource), dg::device_ref(device.get()),
            make_tex_desc(2, 2, 28, 0x102), pl_tiled_tiny, 0u);
        CHECK(tiled_tiny && tiled_tiny->initialize());
        gpu::CommandEncoder tiny_enc;
        CHECK(device->sync_bound_texture_memory(tiled_tiny->id(), tiny_enc));
        CHECK(decode(tiny_enc).size() == 1 && decode(tiny_enc)[0].opcode == GpuOpcode::WriteTexture);

        dg::Resource::Placement pl_tiled_4x4{};
        pl_tiled_4x4.address = tiled_mem;
        auto tiled_4x4 = com::create_object<dg::Resource>(
            com::implements(dg::kID3D12Resource), dg::device_ref(device.get()),
            make_tex_desc(4, 4, 28, 0x102), pl_tiled_4x4, 0u);
        CHECK(tiled_4x4 && tiled_4x4->initialize());
        // With detile hook explicitly cleared, non-tiny tiled texture is skipped.
        device->set_detile_fn(nullptr);
        gpu::CommandEncoder tiled_enc1;
        CHECK(!device->sync_bound_texture_memory(tiled_4x4->id(), tiled_enc1));
        CHECK(tiled_enc1.size() == 0);

        // Install pluggable DetileFn hook (G49a integration point).
        int detile_calls = 0;
        device->set_detile_fn([&](const dg::TextureDetileRequest& req, const uint8_t* src, size_t src_bytes,
                                  uint8_t* dst, size_t dst_bytes) {
            ++detile_calls;
            CHECK(req.layout == 0x102 && req.width == 4 && req.height == 4 && req.dst_row_pitch == 256);
            CHECK(src_bytes >= 64 && dst_bytes >= 1024);
            for (uint32_t y = 0; y < req.height; ++y) {
                for (uint32_t b = 0; b < 16; ++b) {
                    dst[y * req.dst_row_pitch + b] = static_cast<uint8_t>(src[y * 16 + b] ^ 0x5Au);
                }
            }
            return true;
        });
        gpu::CommandEncoder tiled_enc2;
        CHECK(device->sync_bound_texture_memory(tiled_4x4->id(), tiled_enc2));
        CHECK(detile_calls == 1);
        const auto tiled_pkts = decode(tiled_enc2);
        CHECK(tiled_pkts.size() == 1 && tiled_pkts[0].opcode == GpuOpcode::WriteTexture);
        if (tiled_pkts.size() == 1) {
            const uint8_t* rows = tiled_pkts[0].payload.data() + sizeof(GpuWriteTexturePayload);
            CHECK(rows[0] == static_cast<uint8_t>(texels[0] ^ 0x5Au));
        }

        // Case E: Committed CPU-visible texture gets guest memory backing and uploads on sync.
        dg::Resource::Placement pl_committed{};
        pl_committed.heap.Type = dg::HeapType::Custom;
        pl_committed.heap.CPUPageProperty = 2;  // WRITE_COMBINE
        auto comm_tex = com::create_object<dg::Resource>(
            com::implements(dg::kID3D12Resource), dg::device_ref(device.get()),
            make_tex_desc(4, 2, 28, 1), pl_committed, 0u);
        CHECK(comm_tex && comm_tex->initialize() && comm_tex->gpu_va() != 0);
        CHECK(com::write_guest(comm_tex->gpu_va(), texels.data(), 32));
        gpu::CommandEncoder comm_enc;
        CHECK(device->sync_bound_texture_memory(comm_tex->id(), comm_enc));
        CHECK(decode(comm_enc).size() == 1 && decode(comm_enc)[0].opcode == GpuOpcode::WriteTexture);

        // Destroy resources and verify untrack cleans up.
        const uint32_t tex_id = tex->id();
        tex.reset();
        CHECK(!device->tracked_texture_info(tex_id, nullptr));
    }

    com::reset_com_bridge();
    com::set_guest_memory(nullptr);
}

// G64: every mip and array slice of a console-tiled texture uploads from the whole GFX10 surface,
// with per-subresource dirty tracking (one changed mip -> one WriteTexture; steady state -> none).
void test_all_subresource_uploads() {
    namespace com = playgta6::guest::com;
    namespace dg = playgta6::gpu::d3d12_guest;
    namespace gfx10 = playgta6::gpu::gfx10;

    os::VirtualMemoryArena arena;
    com::reset_com_bridge();
    com::set_guest_memory(&arena);
    {
        dg::DeviceOptions dev_opts;
        dev_opts.sink = [](const uint8_t*, size_t) {};
        auto device = com::create_object<dg::Device>(com::implements(dg::kID3D12Device), dev_opts);
        CHECK(device);
        device->initialize_guest_payload();

        // 64x64 R8G8B8A8, 7 mips, 2 array slices, 4KB_S: mips 0..1 are macro blocks, 2..6 in the tail.
        constexpr uint32_t kW = 64, kMips = 7, kLayers = 2;
        gfx10::TextureLayoutDesc g{};
        g.dxgi_format = 28;
        g.width = kW;
        g.height = kW;
        g.mip_levels = kMips;
        g.d3d12_layout = 0x105;
        g.array_size = kLayers;
        const size_t surface_size = gfx10::surface_bytes(g);
        CHECK(surface_size != 0 && surface_size <= (1u << 20));
        CHECK(gfx10::mip_placement(g, 2).in_tail && !gfx10::mip_placement(g, 1).in_tail);

        std::vector<std::vector<uint8_t>> lin(kMips * kLayers);
        std::vector<uint8_t> surface(surface_size, 0);
        auto fill = [&](uint32_t m, uint32_t s, uint32_t seed) {
            const uint32_t w = std::max(1u, kW >> m);
            std::vector<uint8_t>& v = lin[m + s * kMips];
            v.resize(size_t{w} * w * 4);
            for (size_t i = 0; i < v.size(); ++i) {
                v[i] = static_cast<uint8_t>(1u + ((i * 7u + seed * 31u + m * 13u + s * 101u) % 251u));
            }
            CHECK(gfx10::tile_surface_subresource(g, m, s, v.data(), v.size(), surface.data(), surface.size()));
        };
        for (uint32_t s = 0; s < kLayers; ++s) {
            for (uint32_t m = 0; m < kMips; ++m) {
                fill(m, s, 1);
            }
        }
        const uint64_t mem = arena.virtual_alloc(0, 1u << 20, os::kMemReserve | os::kMemCommit, os::PageProtect::ReadWrite);
        CHECK(com::write_guest(mem, surface.data(), surface.size()));

        dg::GuestResourceDesc d{};
        d.Dimension = dg::ResourceDimension::Texture2D;
        d.Width = kW;
        d.Height = kW;
        d.DepthOrArraySize = kLayers;
        d.MipLevels = kMips;
        d.Format = 28;
        d.SampleCount = 1;
        d.Layout = 0x105;
        dg::Resource::Placement pl{};
        pl.address = mem;
        auto tex = com::create_object<dg::Resource>(com::implements(dg::kID3D12Resource), dg::device_ref(device.get()),
                                                    d, pl, 0u);
        CHECK(tex && tex->initialize());

        auto writes = [](const gpu::CommandEncoder& enc) {
            std::vector<Packet> out;
            for (const Packet& pk : decode(enc)) {
                if (pk.opcode == GpuOpcode::WriteTexture) {
                    out.push_back(pk);
                }
            }
            return out;
        };
        auto check_payload = [&](const Packet& pk) {
            const auto wt = pod<GpuWriteTexturePayload>(pk);
            CHECK(wt.texture_id == tex->id() && wt.mip_level < kMips && wt.origin_z < kLayers);
            if (wt.mip_level >= kMips || wt.origin_z >= kLayers) {
                return;
            }
            const uint32_t w = std::max(1u, kW >> wt.mip_level);
            CHECK(wt.width == w && wt.height == w && wt.depth_or_layers == 1 && wt.bytes_per_row == 256);
            const std::vector<uint8_t>& v = lin[wt.mip_level + wt.origin_z * kMips];
            const uint8_t* rows = pk.payload.data() + sizeof(GpuWriteTexturePayload);
            bool same = pk.payload.size() >= sizeof(GpuWriteTexturePayload) + size_t{256} * w;
            for (uint32_t y = 0; same && y < w; ++y) {
                same = std::memcmp(rows + y * 256u, v.data() + size_t{y} * w * 4, size_t{w} * 4) == 0;
            }
            CHECK(same);
        };

        // First sync: all 14 subresources, each with its own texels.
        gpu::CommandEncoder e1;
        CHECK(device->sync_bound_texture_memory(tex->id(), e1));
        const auto w1 = writes(e1);
        CHECK(w1.size() == kMips * kLayers);
        uint32_t seen = 0;
        for (const Packet& pk : w1) {
            check_payload(pk);
            const auto wt = pod<GpuWriteTexturePayload>(pk);
            seen |= 1u << (wt.mip_level + wt.origin_z * kMips);
        }
        CHECK(seen == (1u << (kMips * kLayers)) - 1u);
        const uint64_t uploads = device->stats().textures_uploaded;
        CHECK(uploads == kMips * kLayers);

        // Steady state: no re-upload.
        gpu::CommandEncoder e2;
        CHECK(!device->sync_bound_texture_memory(tex->id(), e2));
        CHECK(writes(e2).empty());

        // One changed macro-block mip (mip 1 of slice 1) -> one upload.
        fill(1, 1, 2);
        CHECK(com::write_guest(mem, surface.data(), surface.size()));
        gpu::CommandEncoder e3;
        CHECK(device->sync_bound_texture_memory(tex->id(), e3));
        const auto w3 = writes(e3);
        CHECK(w3.size() == 1);
        if (w3.size() == 1) {
            const auto wt = pod<GpuWriteTexturePayload>(w3[0]);
            CHECK(wt.mip_level == 1 && wt.origin_z == 1);
            check_payload(w3[0]);
        }

        // One changed tail mip (mip 5 of slice 0, inside the shared tail block) -> one upload.
        fill(5, 0, 3);
        CHECK(com::write_guest(mem, surface.data(), surface.size()));
        gpu::CommandEncoder e4;
        CHECK(device->sync_bound_texture_memory(tex->id(), e4));
        const auto w4 = writes(e4);
        CHECK(w4.size() == 1);
        if (w4.size() == 1) {
            const auto wt = pod<GpuWriteTexturePayload>(w4[0]);
            CHECK(wt.mip_level == 5 && wt.origin_z == 0);
            check_payload(w4[0]);
        }
        CHECK(device->stats().textures_uploaded == uploads + 2);

        // Inside one submission a texture is read once: a change made between two bind groups of
        // the same ExecuteCommandLists reaches the GPU with the next submission.
        device->begin_submission();
        fill(2, 0, 4);
        CHECK(com::write_guest(mem, surface.data(), surface.size()));
        gpu::CommandEncoder e5a;
        CHECK(device->sync_bound_texture_memory(tex->id(), e5a));
        fill(3, 0, 5);
        CHECK(com::write_guest(mem, surface.data(), surface.size()));
        gpu::CommandEncoder e5b;
        CHECK(!device->sync_bound_texture_memory(tex->id(), e5b));
        CHECK(writes(e5b).empty());
        device->end_submission();
        device->begin_submission();
        gpu::CommandEncoder e6;
        CHECK(device->sync_bound_texture_memory(tex->id(), e6));
        CHECK(writes(e6).size() == 1);
        device->end_submission();
        CHECK(device->stats().textures_uploaded == uploads + 4);

        // Console linear 0x100 with a mip chain (#678): every mip uploads from its addrlib
        // ADDR_SW_LINEAR offset (smallest mip first, 256 B pitch), inside the allocation size.
        {
            gfx10::TextureLayoutDesc lg{};
            lg.dxgi_format = 41;  // R32_FLOAT
            lg.width = 128;
            lg.height = 72;
            lg.mip_levels = 7;
            lg.d3d12_layout = 0x100;
            dg::GuestResourceDesc rd = d;
            rd.Width = 128;
            rd.Height = 72;
            rd.DepthOrArraySize = 1;
            rd.MipLevels = 7;
            rd.Format = 41;
            rd.Layout = 0x100;
            const size_t lsize = gfx10::surface_bytes(lg);
            CHECK(lsize == 55552u && dg::resource_size_estimate(rd) == lsize);
            std::vector<uint8_t> lsurf(lsize, 0);
            std::vector<std::vector<uint8_t>> mips(7);
            for (uint32_t m = 0; m < 7; ++m) {
                const uint32_t mw = std::max(1u, 128u >> m), mh = std::max(1u, 72u >> m);
                mips[m].resize(size_t{mw} * mh * 4);
                for (size_t i = 0; i < mips[m].size(); ++i) {
                    mips[m][i] = static_cast<uint8_t>(1u + (i * 5u + m * 41u) % 241u);
                }
                CHECK(gfx10::tile_surface_subresource(lg, m, 0, mips[m].data(), mips[m].size(), lsurf.data(), lsurf.size()));
            }
            const uint64_t lmem =
                arena.virtual_alloc(0, 65536, os::kMemReserve | os::kMemCommit, os::PageProtect::ReadWrite);
            CHECK(com::write_guest(lmem, lsurf.data(), lsurf.size()));
            dg::Resource::Placement lp{};
            lp.address = lmem;
            auto ctex = com::create_object<dg::Resource>(com::implements(dg::kID3D12Resource),
                                                         dg::device_ref(device.get()), rd, lp, 0u);
            CHECK(ctex && ctex->initialize());
            gpu::CommandEncoder ec;
            CHECK(device->sync_bound_texture_memory(ctex->id(), ec));
            const auto wc = writes(ec);
            CHECK(wc.size() == 7);
            for (const Packet& pk : wc) {
                const auto wt = pod<GpuWriteTexturePayload>(pk);
                CHECK(wt.mip_level < 7 && wt.origin_z == 0);
                if (wt.mip_level >= 7) {
                    continue;
                }
                const uint32_t mw = std::max(1u, 128u >> wt.mip_level), mh = std::max(1u, 72u >> wt.mip_level);
                const uint32_t pitch = (mw * 4u + 255u) / 256u * 256u;
                CHECK(wt.width == mw && wt.height == mh && wt.bytes_per_row == pitch);
                bool same = true;
                for (uint32_t y = 0; same && y < mh; ++y) {
                    same = std::memcmp(pk.payload.data() + sizeof(GpuWriteTexturePayload) + size_t{y} * pitch,
                                       mips[wt.mip_level].data() + size_t{y} * mw * 4, size_t{mw} * 4) == 0;
                }
                CHECK(same);
            }
        }

        // A linear 2D array whose rows are a multiple of 256 B uploads every slice (mip 0 only).
        const uint64_t lin_mem =
            arena.virtual_alloc(0, 65536, os::kMemReserve | os::kMemCommit, os::PageProtect::ReadWrite);
        std::vector<uint8_t> slices(64u * 2u * 4u * 3u);
        for (size_t i = 0; i < slices.size(); ++i) {
            slices[i] = static_cast<uint8_t>(1u + i % 199u);
        }
        CHECK(com::write_guest(lin_mem, slices.data(), slices.size()));
        dg::GuestResourceDesc ld = d;
        ld.Width = 64;
        ld.Height = 2;
        ld.DepthOrArraySize = 3;
        ld.MipLevels = 1;
        ld.Layout = 0x100;
        dg::Resource::Placement lpl{};
        lpl.address = lin_mem;
        auto ltex = com::create_object<dg::Resource>(com::implements(dg::kID3D12Resource),
                                                     dg::device_ref(device.get()), ld, lpl, 0u);
        CHECK(ltex && ltex->initialize());
        gpu::CommandEncoder e5;
        CHECK(device->sync_bound_texture_memory(ltex->id(), e5));
        const auto w5 = writes(e5);
        CHECK(w5.size() == 3);
        for (const Packet& pk : w5) {
            const auto wt = pod<GpuWriteTexturePayload>(pk);
            CHECK(wt.mip_level == 0 && wt.origin_z < 3 && wt.width == 64 && wt.height == 2);
            if (wt.origin_z < 3) {
                CHECK(std::memcmp(pk.payload.data() + sizeof(GpuWriteTexturePayload),
                                  slices.data() + size_t{wt.origin_z} * 512u, 512u) == 0);
            }
        }
    }
    com::reset_com_bridge();
    com::set_guest_memory(nullptr);
}

// G66: streamed textures arrive as BC (and 16-bit normalized) texels in console layouts. They are
// uploaded only when the device's WebGPU features include the format: the browser player's set
// (boot::kDefaultWasmWgpuFeatures, which the native harness now declares too) uploads them; a
// device with no features substitutes the format and skips the texels.
void test_bc_texture_upload_needs_feature() {
    namespace com = playgta6::guest::com;
    namespace dg = playgta6::gpu::d3d12_guest;
    namespace gfx10 = playgta6::gpu::gfx10;

    os::VirtualMemoryArena arena;
    com::reset_com_bridge();
    com::set_guest_memory(&arena);
    {
        // 64x64 BC3 (16 B blocks, 16x16 blocks), 4KB_S, one mip: one 4 KiB block.
        gfx10::TextureLayoutDesc g{};
        g.dxgi_format = 77;
        g.width = 64;
        g.height = 64;
        g.d3d12_layout = 0x105;
        const size_t size = gfx10::surface_bytes(g);
        CHECK(size == 4096u);
        std::vector<uint8_t> blocks(16u * 16u * 16u);
        for (size_t i = 0; i < blocks.size(); ++i) {
            blocks[i] = static_cast<uint8_t>(1u + i % 253u);
        }
        std::vector<uint8_t> surface(size, 0);
        CHECK(gfx10::tile_surface_subresource(g, 0, 0, blocks.data(), blocks.size(), surface.data(), surface.size()));
        const uint64_t mem = arena.virtual_alloc(0, 65536, os::kMemReserve | os::kMemCommit, os::PageProtect::ReadWrite);
        CHECK(com::write_guest(mem, surface.data(), surface.size()));

        for (const uint32_t features : {0u, static_cast<uint32_t>(playgta6::boot::kDefaultWasmWgpuFeatures)}) {
            dg::DeviceOptions dev_opts;
            dev_opts.sink = [](const uint8_t*, size_t) {};
            dev_opts.bridge.features = features;
            auto device = com::create_object<dg::Device>(com::implements(dg::kID3D12Device), dev_opts);
            CHECK(device);
            device->initialize_guest_payload();
            const uint32_t id = device->ensure_address_texture(mem, 77, 64, 64, 1, 1, 0x105,
                                                               GpuTextureDimension::Dim2D, false);
            CHECK(id != 0);
            gpu::CommandEncoder enc;
            const bool uploaded = device->sync_bound_texture_memory(id, enc);
            std::vector<Packet> writes;
            for (const Packet& pk : decode(enc)) {
                if (pk.opcode == GpuOpcode::WriteTexture) {
                    writes.push_back(pk);
                }
            }
            if (features == 0) {
                CHECK(!uploaded && writes.empty());  // bc3 unavailable: substitute format, no texels
            } else {
                CHECK(uploaded && writes.size() == 1);
                if (writes.size() == 1) {
                    const auto wt = pod<GpuWriteTexturePayload>(writes[0]);
                    CHECK(wt.width == 64 && wt.height == 64 && wt.bytes_per_row == 256 && wt.rows_per_image == 16);
                    bool same = true;
                    for (uint32_t y = 0; same && y < 16u; ++y) {
                        same = std::memcmp(writes[0].payload.data() + sizeof(GpuWriteTexturePayload) + y * 256u,
                                           blocks.data() + y * 256u, 256u) == 0;
                    }
                    CHECK(same);
                }
            }
        }
    }
    com::reset_com_bridge();
    com::set_guest_memory(nullptr);
}

// G66: a view-only R16_UNORM texture (sampled only as unfilterable-float in WebGPU) is created in a
// filterable float format and its texels are converted on upload: r32float (exact) with
// float32-filterable, r16float (rounded) without. Its SRV view names the same format.
void test_norm16_address_texture_upload() {
    namespace com = playgta6::guest::com;
    namespace dg = playgta6::gpu::d3d12_guest;
    namespace pso = playgta6::gpu::pso;

    os::VirtualMemoryArena arena;
    com::reset_com_bridge();
    com::set_guest_memory(&arena);
    {
        // 4x2 R16_UNORM, linear 0x100: 4 texels x 2 B = 8 B rows at a 256 B pitch.
        const uint16_t row0[4] = {0, 65535, 32768, 1};
        const uint16_t row1[4] = {100, 200, 300, 400};
        const uint64_t mem = arena.virtual_alloc(0, 65536, os::kMemReserve | os::kMemCommit, os::PageProtect::ReadWrite);
        CHECK(com::write_guest(mem, row0, sizeof(row0)) && com::write_guest(mem + 256, row1, sizeof(row1)));
        const uint32_t tier1 = pso::WgpuFeature::TextureFormatsTier1;
        for (const uint32_t features : {tier1 | pso::WgpuFeature::Float32Filterable, tier1}) {
            const bool f32 = (features & pso::WgpuFeature::Float32Filterable) != 0;
            std::vector<uint8_t> submitted;
            dg::DeviceOptions dev_opts;
            dev_opts.sink = [&submitted](const uint8_t* data, size_t size) {
                submitted.insert(submitted.end(), data, data + size);
            };
            dev_opts.bridge.features = features;
            auto device = com::create_object<dg::Device>(com::implements(dg::kID3D12Device), dev_opts);
            CHECK(device);
            device->initialize_guest_payload();
            const uint32_t id = device->ensure_address_texture(mem, 56, 4, 2, 1, 1, 0x100,
                                                               GpuTextureDimension::Dim2D, false);
            CHECK(id != 0);
            bool created = false;
            for (const Packet& pk : decode_bytes(submitted)) {
                if (pk.opcode == GpuOpcode::CreateTexture2) {
                    const auto ct = pod<GpuCreateTexture2Payload>(pk);
                    created = ct.id == id && ct.format == static_cast<uint8_t>(f32 ? pso::WgpuTextureFormat::R32Float
                                                                                    : pso::WgpuTextureFormat::R16Float);
                }
            }
            CHECK(created);
            gpu::CommandEncoder enc;
            CHECK(device->sync_bound_texture_memory(id, enc));
            std::vector<Packet> writes;
            for (const Packet& pk : decode(enc)) {
                if (pk.opcode == GpuOpcode::WriteTexture) {
                    writes.push_back(pk);
                }
            }
            CHECK(writes.size() == 1);
            if (writes.size() == 1) {
                const auto wt = pod<GpuWriteTexturePayload>(writes[0]);
                CHECK(wt.width == 4 && wt.height == 2 && wt.bytes_per_row == 256);
                const uint8_t* rows = writes[0].payload.data() + sizeof(GpuWriteTexturePayload);
                if (f32) {
                    float v[4] = {};
                    std::memcpy(v, rows, sizeof(v));
                    CHECK(v[0] == 0.0f && v[1] == 1.0f && v[2] == 32768.0f / 65535.0f && v[3] == 1.0f / 65535.0f);
                    std::memcpy(v, rows + 256, sizeof(v));
                    CHECK(v[0] == 100.0f / 65535.0f && v[3] == 400.0f / 65535.0f);
                } else {
                    uint16_t hv[4] = {};
                    std::memcpy(hv, rows, sizeof(hv));
                    CHECK(hv[0] == 0x0000 && hv[1] == 0x3C00 && hv[2] == 0x3800 && hv[3] == 0x0100);
                }
            }
        }
    }
    com::reset_com_bridge();
    com::set_guest_memory(nullptr);
}

// ------------------------------------------------------------------------------------------------
// G49b: GPU Fence completion audit & ExecuteCommandLists / PresentX com::Ref lifetime audit
// ------------------------------------------------------------------------------------------------

void test_fence_completion_and_ref_lifetime() {
    namespace com = playgta6::guest::com;
    namespace dg = playgta6::gpu::d3d12_guest;

    os::VirtualMemoryArena arena;
    com::reset_com_bridge();
    com::set_guest_memory(&arena);

    {
        dg::DeviceOptions dev_opts;
        dev_opts.sink = [](const uint8_t*, size_t) {};
        auto device = com::create_object<dg::Device>(com::implements(dg::kID3D12Device), dev_opts);
        CHECK(device);
        device->initialize_guest_payload();

        dg::GuestCommandQueueDesc qdesc{};
        qdesc.Type = 0;  // DIRECT
        auto queue = com::create_object<dg::CommandQueue>(com::implements(dg::kID3D12CommandQueue),
                                                          dg::device_ref(device.get()), qdesc);
        auto fence = com::create_object<dg::Fence>(com::implements(dg::kID3D12Fence), dg::device_ref(device.get()),
                                                   10ull, 0u);
        CHECK(queue && fence);
        const uint64_t fence_ptr = fence->interface_pointer(dg::kID3D12Fence);
        const uint64_t word_va = fence->CompletedValueAddress();
        CHECK(word_va != 0);
        uint64_t published = 0;
        CHECK(com::read_guest(word_va, published) && published == 10ull);

        // 1) Pre-signalled SetEventOnCompletion: immediately signals event & null handle returns immediately.
        const Pg6Handle ev_pre = pg6_win32_create_event_w(nullptr, 1, 0, nullptr);
        CHECK(ev_pre != 0);
        CHECK(fence->SetEventOnCompletion(10, ev_pre) == dg::kSOk);
        CHECK(pg6_win32_wait_for_single_object(ev_pre, 0) == os::win32::WAIT_OBJECT_0);
        CHECK(fence->SetEventOnCompletion(5, 0) == dg::kSOk);
        pg6_win32_close_handle(ev_pre);

        // 2) Post-signalled multi-waiter: <= value wake up, > value stay pending until later signal.
        const Pg6Handle ev_15a = pg6_win32_create_event_w(nullptr, 1, 0, nullptr);
        const Pg6Handle ev_15b = pg6_win32_create_event_w(nullptr, 1, 0, nullptr);
        const Pg6Handle ev_12 = pg6_win32_create_event_w(nullptr, 1, 0, nullptr);
        const Pg6Handle ev_20 = pg6_win32_create_event_w(nullptr, 1, 0, nullptr);
        CHECK(fence->SetEventOnCompletion(15, ev_15a) == dg::kSOk);
        CHECK(fence->SetEventOnCompletion(15, ev_15b) == dg::kSOk);
        CHECK(fence->SetEventOnCompletion(12, ev_12) == dg::kSOk);
        CHECK(fence->SetEventOnCompletion(20, ev_20) == dg::kSOk);
        CHECK(pg6_win32_wait_for_single_object(ev_12, 0) == os::win32::WAIT_TIMEOUT);

        auto fence_arg = com::InterfaceArg<dg::IFenceHost>::resolve(fence_ptr);
        CHECK(fence_arg && queue->Signal(fence_arg, 15) == dg::kSOk);
        CHECK(fence->GetCompletedValue() == 15);
        CHECK(com::read_guest(word_va, published) && published == 15ull);
        CHECK(pg6_win32_wait_for_single_object(ev_12, 0) == os::win32::WAIT_OBJECT_0);
        CHECK(pg6_win32_wait_for_single_object(ev_15a, 0) == os::win32::WAIT_OBJECT_0);
        CHECK(pg6_win32_wait_for_single_object(ev_15b, 0) == os::win32::WAIT_OBJECT_0);
        CHECK(pg6_win32_wait_for_single_object(ev_20, 0) == os::win32::WAIT_TIMEOUT);

        // Out-of-order lower CommandQueue::Signal must not regress completed value.
        CHECK(queue->Signal(fence_arg, 11) == dg::kSOk);
        CHECK(fence->GetCompletedValue() == 15);
        CHECK(queue->Signal(fence_arg, 20) == dg::kSOk);
        CHECK(fence->GetCompletedValue() == 20);
        CHECK(pg6_win32_wait_for_single_object(ev_20, 0) == os::win32::WAIT_OBJECT_0);
        pg6_win32_close_handle(ev_15a);
        pg6_win32_close_handle(ev_15b);
        pg6_win32_close_handle(ev_12);
        pg6_win32_close_handle(ev_20);

        // 3) 64-thread multi-waiter + concurrent CommandQueue::Signal stress test (event + null-handle waiters).
        constexpr uint64_t kBase = 100;
        constexpr uint32_t kEventWaiters = 32;
        constexpr uint32_t kSyncWaiters = 16;
        constexpr uint32_t kSignallers = 16;
        std::atomic<uint32_t> woken_events{0};
        std::atomic<uint32_t> woken_sync{0};
        std::vector<std::thread> threads;
        threads.reserve(kEventWaiters + kSyncWaiters + kSignallers);

        for (uint32_t i = 0; i < kEventWaiters; ++i) {
            threads.emplace_back([&, i] {
                const uint64_t target = kBase + (i % kSignallers) + 1;
                const Pg6Handle ev = pg6_win32_create_event_w(nullptr, 1, 0, nullptr);
                CHECK(ev != 0);
                CHECK(fence->SetEventOnCompletion(target, ev) == dg::kSOk);
                const uint32_t wait_rc = pg6_win32_wait_for_single_object(ev, 5000);
                CHECK(wait_rc == os::win32::WAIT_OBJECT_0);
                CHECK(fence->GetCompletedValue() >= target);
                pg6_win32_close_handle(ev);
                woken_events.fetch_add(1, std::memory_order_relaxed);
            });
        }
        for (uint32_t i = 0; i < kSyncWaiters; ++i) {
            threads.emplace_back([&, i] {
                const uint64_t target = kBase + i + 1;
                CHECK(fence->SetEventOnCompletion(target, 0) == dg::kSOk);
                CHECK(fence->GetCompletedValue() >= target);
                woken_sync.fetch_add(1, std::memory_order_relaxed);
            });
        }
        for (uint32_t i = 0; i < kSignallers; ++i) {
            threads.emplace_back([&, i] {
                // Deliberately signal in scrambled order to verify monotonic advance across threads.
                const uint64_t val = kBase + ((i * 7 + 3) % kSignallers) + 1;
                auto arg = com::InterfaceArg<dg::IFenceHost>::resolve(fence_ptr);
                CHECK(arg && queue->Signal(arg, val) == dg::kSOk);
            });
        }
        for (std::thread& th : threads) {
            th.join();
        }
        CHECK(woken_events.load() == kEventWaiters);
        CHECK(woken_sync.load() == kSyncWaiters);
        CHECK(fence->GetCompletedValue() == kBase + kSignallers);
        CHECK(com::read_guest(word_va, published) && published == kBase + kSignallers);

        // 4) Zero com::Ref leaks across repeated ExecuteCommandLists + PresentX cycles.
        {
            dg::GuestResourceDesc rt_desc{};
            rt_desc.Dimension = dg::ResourceDimension::Texture2D;
            rt_desc.Width = 64;
            rt_desc.Height = 64;
            rt_desc.DepthOrArraySize = 1;
            rt_desc.MipLevels = 1;
            rt_desc.Format = 87;  // B8G8R8A8_UNORM
            rt_desc.SampleCount = 1;
            rt_desc.Flags = dg::ResourceFlags::AllowRenderTarget;
            dg::Resource::Placement pl{};
            pl.heap.Type = dg::HeapType::Default;
            auto back_buf = com::create_object<dg::Resource>(com::implements(dg::kID3D12Resource),
                                                             dg::device_ref(device.get()), rt_desc, pl, 0u);
            CHECK(back_buf && back_buf->initialize());

            auto alloc = com::create_object<dg::CommandAllocator>(com::implements(dg::kID3D12CommandAllocator),
                                                                  dg::device_ref(device.get()), 0u);
            auto cmd = com::create_object<dg::CommandList>(com::implements(dg::kID3D12GraphicsCommandList),
                                                           dg::device_ref(device.get()), 0u);
            CHECK(alloc && cmd);
            cmd->initialize_guest_payload();
            cmd->reset(nullptr);

            const uint64_t bb_ptr = back_buf->interface_pointer(dg::kID3D12Resource);
            const uint64_t cmd_ptr = cmd->interface_pointer(dg::kID3D12GraphicsCommandList);
            const uint64_t alloc_ptr = alloc->interface_pointer(dg::kID3D12CommandAllocator);
            const uint64_t bb_slot = com::guest_alloc(sizeof(uint64_t));
            CHECK(bb_slot != 0 && com::write_guest(bb_slot, bb_ptr));

            dg::xbox::GuestPresentPlaneX plane{};
            plane.Token = 1;
            plane.ResourceCount = 1;
            plane.ppResources = bb_slot;
            const uint64_t plane_va = com::guest_alloc(sizeof(plane));
            CHECK(plane_va != 0 && com::write_guest(plane_va, plane));

            const uint32_t bb_refs_before = back_buf->ref_count();
            const uint32_t cmd_refs_before = cmd->ref_count();
            const uint32_t fence_refs_before = fence->ref_count();

            for (uint32_t iter = 0; iter < 16; ++iter) {
                CHECK(cmd->Close() == dg::kSOk);
                queue->execute({cmd_ptr});
                plane.Token = iter + 1;
                CHECK(com::write_guest(plane_va, plane));
                CHECK(queue->PresentX(1, plane_va, 0) == dg::kSOk);
                CHECK(queue->Signal(com::InterfaceArg<dg::IFenceHost>::resolve(fence_ptr), 200 + iter) == dg::kSOk);
                CHECK(cmd->Reset(com::InterfaceArg<dg::ICommandAllocatorHost>::resolve(alloc_ptr), {}) == dg::kSOk);
                CHECK(back_buf->ref_count() == bb_refs_before);
                CHECK(cmd->ref_count() == cmd_refs_before);
                CHECK(fence->ref_count() == fence_refs_before);
            }
            // Each present drew the back buffer into the canvas; the texture was never retargeted.
            CHECK(!back_buf->is_back_buffer());
            CHECK(device->counter(dg::xbox::Counter::PresentPlanesDrawn) == 16);
            com::guest_free(plane_va);
            com::guest_free(bb_slot);
        }
        CHECK(device->stats().resources == 0);
        CHECK(device->stats().command_lists == 0);
    }

    com::reset_com_bridge();
    com::set_guest_memory(nullptr);
}

// content_hash / all_zero_bytes: the change detection of mirrored guest memory sees every
// single-bit change at every position and length (lane boundaries, word tails, byte tails).
void test_content_hash_change_detection() {
    namespace dg = playgta6::gpu::d3d12_guest;
    std::vector<uint8_t> buf(203);
    for (size_t i = 0; i < buf.size(); ++i) {
        buf[i] = static_cast<uint8_t>(i * 37u + 11u);
    }
    int missed = 0;
    for (size_t len = 0; len <= buf.size(); ++len) {
        const uint64_t base = dg::content_hash(buf.data(), len);
        CHECK(dg::content_hash(buf.data(), len) == base);
        if (len > 0) {
            CHECK(dg::content_hash(buf.data(), len - 1) != base);
        }
        for (size_t pos = 0; pos < len; ++pos) {
            for (int bit = 0; bit < 8; bit += 3) {
                buf[pos] ^= static_cast<uint8_t>(1u << bit);
                missed += dg::content_hash(buf.data(), len) == base ? 1 : 0;
                buf[pos] ^= static_cast<uint8_t>(1u << bit);
            }
        }
    }
    CHECK(missed == 0);
    // Swapped words (same multiset of bytes) hash differently.
    std::vector<uint8_t> swapped(buf.begin(), buf.begin() + 64);
    std::swap_ranges(swapped.begin(), swapped.begin() + 8, swapped.begin() + 8);
    CHECK(dg::content_hash(swapped.data(), 64) != dg::content_hash(buf.data(), 64));

    std::vector<uint8_t> zero(300, 0);
    for (size_t len = 0; len <= zero.size(); len += 7) {
        CHECK(dg::all_zero_bytes(zero.data(), len));
    }
    for (size_t pos = 0; pos < zero.size(); ++pos) {
        zero[pos] = 1;
        CHECK(!dg::all_zero_bytes(zero.data(), zero.size()));
        CHECK(dg::all_zero_bytes(zero.data(), pos));
        zero[pos] = 0;
    }
}

// #482 / #954: the translator declares every storage texture rgba8unorm. A compute binding of a
// storage-capable texture of another float format (the rgba16float 3D LUT of stage
// 0x311b9e5fc666d0b5) is bound to an intermediate texture in the declared format, converted into the
// real texture after the dispatch (and from it before, for read-write access). Matching formats
// keep the exact binding and emit no conversion.
void test_storage_texture_format_conversion() {
    struct Case {
        pso::DxgiFormat format;
        bool volume;
        GpuStorageTextureAccess access;
    };
    const Case cases[] = {
        {pso::DxgiFormat::R16G16B16A16_FLOAT, true, GpuStorageTextureAccess::WriteOnly},   // mismatched 3D LUT
        {pso::DxgiFormat::R16G16B16A16_FLOAT, false, GpuStorageTextureAccess::ReadWrite},  // mismatched, read-write
        {pso::DxgiFormat::R8G8B8A8_UNORM, true, GpuStorageTextureAccess::WriteOnly},       // matched
    };
    for (const Case& c : cases) {
        const bool mismatched = c.format != pso::DxgiFormat::R8G8B8A8_UNORM;
        d3d::DescriptorHeapRegistry heaps;
        d3d::ResourceStateRegistry states;
        bridge::StateBridgeDevice device(heaps, counting_options(7000), &states);
        constexpr uint32_t kTex = 302;
        constexpr uint64_t kCs = 0xE3;
        bridge::TextureInfo tex{c.format, 32, 16, c.volume ? 8u : 1u, 1, 1,
                                c.volume ? GpuTextureDimension::Dim3D : GpuTextureDimension::Dim2D, false};
        tex.storage = true;
        device.register_texture(kTex, tex);
        device.register_shader_module(kCs, 3);
        d3d::DescriptorHeap* view_heap = make_heap(heaps, d3d::DescriptorHeapType::CbvSrvUav, 4, true);
        d3d::ViewDesc vd;
        vd.format = static_cast<uint32_t>(tex.format);
        vd.dimension = c.volume ? d3d::ViewDimension::Texture3D : d3d::ViewDimension::Texture2D;
        CHECK(heaps.create_unordered_access_view(kTex, 0, &vd, view_heap->cpu_handle(0)) == d3d::Status::Ok);
        d3d::RootSignatureDesc rs;
        d3d::RootParameterDesc table;
        table.parameter_type = d3d::RootParameterType::DescriptorTable;
        d3d::DescriptorRangeDesc range;
        range.range_type = d3d::DescriptorRangeType::Uav;
        range.num_descriptors = 1;
        table.ranges.push_back(range);
        rs.parameters.push_back(table);
        d3d::RootSignature root;
        CHECK(d3d::RootSignature::build(rs, &root).ok());
        const auto dim = c.volume ? GpuTextureViewDimension::Dim3D : GpuTextureViewDimension::Dim2D;
        bridge::PipelineLayoutDesc layout;
        bridge::GroupLayoutDesc group;
        GpuBindGroupLayoutEntry entry =
            layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::StorageTexture,
                         static_cast<uint8_t>(c.access), static_cast<uint8_t>(dim));
        entry.storage_format = static_cast<uint8_t>(pso::WgpuTextureFormat::Rgba8Unorm);
        group.bindings.push_back({entry, join(0, d3d::DescriptorRangeType::Uav, 0)});
        layout.groups.push_back(group);
        device.register_pipeline_layout(kCs, 0, layout);
        pso::D3D12ComputePsoDesc desc;
        desc.cs = kCs;
        auto cpso = device.create_compute_pipeline_state(desc, &root);
        gpu::CommandEncoder enc;
        bridge::CommandListBridge list(device, enc);
        d3d::DescriptorHeap* visible[] = {view_heap};
        CHECK(list.SetDescriptorHeaps(1, visible) == d3d::Status::Ok);
        CHECK(list.SetComputeRootSignature(&root) == d3d::Status::Ok);
        list.SetComputePipelineState(cpso.get());
        CHECK(list.SetComputeRootDescriptorTable(0, view_heap->gpu_handle(0)) == d3d::Status::Ok);
        list.Dispatch(2, 2, 2);
        list.Dispatch(1, 1, 1);
        list.Close();

        const std::vector<Packet> p = decode(enc);
        uint32_t intermediate = 0, game_group = 0, conversion_modules = 0;
        GpuCreateTexture2Payload created{};
        GpuBindGroupEntry2 bound{};
        std::vector<std::pair<uint32_t, uint32_t>> dispatches;  // (pipeline, x)
        uint32_t pipeline = 0;
        for (const Packet& pk : p) {
            if (pk.opcode == GpuOpcode::CreateTexture2) {
                created = pod<GpuCreateTexture2Payload>(pk);
                intermediate = created.id;
            } else if (pk.opcode == GpuOpcode::CreateShaderModule) {
                ++conversion_modules;
            } else if (pk.opcode == GpuOpcode::CreateBindGroup2 && game_group == 0) {
                game_group = pod<GpuCreateBindGroup2Payload>(pk).id;
                bound = pod<GpuBindGroupEntry2>(pk, sizeof(GpuCreateBindGroup2Payload));
            } else if (pk.opcode == GpuOpcode::SetComputePipeline) {
                pipeline = pod<GpuSetComputePipelinePayload>(pk).pipeline_id;
            } else if (pk.opcode == GpuOpcode::DispatchCompute) {
                dispatches.emplace_back(pipeline, pod<GpuDispatchComputePayload>(pk).workgroup_x);
            }
        }
        CHECK(list.stats().dispatches == 2);
        CHECK(list.stats().storage_views_format_mismatch == 0 && list.stats().storage_views_without_usage == 0);
        if (!mismatched) {
            CHECK(intermediate == 0 && conversion_modules == 0);
            CHECK(bound.resource_id == kTex);
            CHECK(list.stats().storage_views_converted == 0 && list.stats().storage_conversions == 0);
            CHECK(dispatches.size() == 2);
            continue;
        }
        // The intermediate: the texture's size, dimension and mips in the declared format.
        CHECK(intermediate != 0 && intermediate != kTex);
        CHECK(created.format == static_cast<uint8_t>(pso::WgpuTextureFormat::Rgba8Unorm));
        CHECK(created.width == 32 && created.height == 16 && created.depth_or_layers == tex.depth_or_array_size);
        CHECK(created.dimension == static_cast<uint8_t>(tex.dimension) && created.mip_levels == 1);
        CHECK((created.usage_flags & GpuTextureUsage::StorageBinding) != 0);
        CHECK(bound.resource_id == intermediate &&
              bound.view.format == static_cast<uint8_t>(pso::WgpuTextureFormat::Rgba8Unorm));
        CHECK(list.stats().storage_views_converted == 1);
        // One conversion out per dispatch, plus one in per dispatch for read-write access.
        const bool rw = c.access == GpuStorageTextureAccess::ReadWrite;
        const uint32_t per_dispatch = rw ? 2u : 1u;
        CHECK(list.stats().storage_conversions == 2 * per_dispatch);
        CHECK(conversion_modules == per_dispatch);  // one pipeline per (target format, dimension)
        CHECK(dispatches.size() == 2 * (1 + per_dispatch));
        if (dispatches.size() == 2 * (1 + per_dispatch)) {
            for (size_t d = 0; d < 2; ++d) {
                const size_t base = d * (1 + per_dispatch);
                const size_t game = base + (rw ? 1 : 0);
                CHECK(dispatches[game].first == cpso->pipeline_id());
                CHECK(dispatches[game + 1].first != cpso->pipeline_id());  // converted after it
                // 3D: 4x4x4 groups over 32x16x8; 2D: 8x8 groups over 32x16.
                CHECK(dispatches[game + 1].second == (c.volume ? 8u : 4u));
                if (rw) {
                    CHECK(dispatches[base].first != cpso->pipeline_id());  // copied in before it
                }
            }
        }
        // The conversion out binds the intermediate (sampled) and the real texture (storage, its format).
        bool out_group = false;
        for (const Packet& pk : p) {
            if (pk.opcode == GpuOpcode::CreateBindGroup2 && pk.payload.size() >= sizeof(GpuCreateBindGroup2Payload) +
                                                                                   2 * sizeof(GpuBindGroupEntry2)) {
                const auto src = pod<GpuBindGroupEntry2>(pk, sizeof(GpuCreateBindGroup2Payload));
                const auto dst =
                    pod<GpuBindGroupEntry2>(pk, sizeof(GpuCreateBindGroup2Payload) + sizeof(GpuBindGroupEntry2));
                if (src.resource_id == intermediate && dst.resource_id == kTex) {
                    out_group = dst.view.format == static_cast<uint8_t>(pso::WgpuTextureFormat::Rgba16Float) &&
                                dst.view.dimension == static_cast<uint8_t>(dim) && dst.view.mip_level_count == 1;
                }
            }
        }
        CHECK(out_group);
    }
}

// Backlog 39 / 35: a compute dispatch whose writable storage view is null is dropped and counted
// (once per pipeline), not recorded; a depth-format texture given a storage view is null too (WebGPU
// has no storage depth) and is counted in the census. variant 0: valid view (recorded), 1: empty
// descriptor slot, 2: depth texture, 3: empty slot with the drop switched off.
void test_null_storage_dispatch_dropped() {
    for (int variant = 0; variant < 4; ++variant) {
        d3d::DescriptorHeapRegistry heaps;
        d3d::ResourceStateRegistry states;
        bridge::BridgeOptions options = counting_options(7000);
        options.drop_null_storage_dispatch = variant != 3;
        bridge::StateBridgeDevice device(heaps, options, &states);
        constexpr uint32_t kTex = 302;
        constexpr uint64_t kCs = 0xE3;
        const pso::DxgiFormat fmt = variant == 2 ? pso::DxgiFormat::D32_FLOAT : pso::DxgiFormat::R8G8B8A8_UNORM;
        bridge::TextureInfo tex{fmt, 32, 16, 1, 1, 1, GpuTextureDimension::Dim2D, false};
        tex.storage = variant == 0;
        device.register_texture(kTex, tex);
        device.register_shader_module(kCs, 3);
        d3d::DescriptorHeap* view_heap = make_heap(heaps, d3d::DescriptorHeapType::CbvSrvUav, 4, true);
        if (variant == 0 || variant == 2) {
            d3d::ViewDesc vd;
            vd.format = static_cast<uint32_t>(fmt);
            vd.dimension = d3d::ViewDimension::Texture2D;
            CHECK(heaps.create_unordered_access_view(kTex, 0, &vd, view_heap->cpu_handle(0)) == d3d::Status::Ok);
        }
        d3d::RootSignatureDesc rs;
        d3d::RootParameterDesc table;
        table.parameter_type = d3d::RootParameterType::DescriptorTable;
        d3d::DescriptorRangeDesc range;
        range.range_type = d3d::DescriptorRangeType::Uav;
        range.num_descriptors = 1;
        table.ranges.push_back(range);
        rs.parameters.push_back(table);
        d3d::RootSignature root;
        CHECK(d3d::RootSignature::build(rs, &root).ok());
        bridge::PipelineLayoutDesc layout;
        bridge::GroupLayoutDesc group;
        group.content_id = 0x5EED;  // a manifest layout: lists only the bindings the shader uses
        GpuBindGroupLayoutEntry entry =
            layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::StorageTexture,
                         static_cast<uint8_t>(GpuStorageTextureAccess::WriteOnly),
                         static_cast<uint8_t>(GpuTextureViewDimension::Dim2D));
        entry.storage_format = static_cast<uint8_t>(pso::WgpuTextureFormat::Rgba8Unorm);
        group.bindings.push_back({entry, join(0, d3d::DescriptorRangeType::Uav, 0)});
        layout.groups.push_back(group);
        device.register_pipeline_layout(kCs, 0, layout);
        pso::D3D12ComputePsoDesc desc;
        desc.cs = kCs;
        auto cpso = device.create_compute_pipeline_state(desc, &root);
        gpu::CommandEncoder enc;
        bridge::CommandListBridge list(device, enc);
        d3d::DescriptorHeap* visible[] = {view_heap};
        CHECK(list.SetDescriptorHeaps(1, visible) == d3d::Status::Ok);
        CHECK(list.SetComputeRootSignature(&root) == d3d::Status::Ok);
        list.SetComputePipelineState(cpso.get());
        CHECK(list.SetComputeRootDescriptorTable(0, view_heap->gpu_handle(0)) == d3d::Status::Ok);
        list.Dispatch(2, 2, 2);
        list.Dispatch(1, 1, 1);
        list.Close();
        uint32_t recorded = 0;
        for (const Packet& pk : decode(enc)) {
            recorded += pk.opcode == GpuOpcode::DispatchCompute ? 1u : 0u;
        }
        const bool dropped = variant == 1 || variant == 2;
        CHECK(recorded == (dropped ? 0u : 2u));
        CHECK(list.stats().dispatches == (dropped ? 0u : 2u));
        CHECK(list.stats().dispatches_dropped_null_storage == (dropped ? 2u : 0u));
        CHECK(list.stats().null_storage_dispatch_pipelines == (dropped ? 1u : 0u));
        CHECK(((list.diagnostics() & bridge::BridgeDiag::NullStorageDispatchDropped) != 0) == dropped);
        CHECK(list.stats().storage_views_depth_untwinned == (variant == 2 ? 1u : 0u));
    }
}

// Backlog 35: a compute storage view of a depth texture is bound to a colour twin: the depth plane is
// copied into it once (while the twin may differ), and back after each dispatch that writes it
// (a fullscreen draw writing frag_depth); a read-only binding is never copied back.
// variant 0: write-only r32float, 1: read-only r32float, 2: write-only r32uint, 3: rgba8unorm (no twin).
void test_depth_storage_twin() {
    for (int variant = 0; variant < 4; ++variant) {
        d3d::DescriptorHeapRegistry heaps;
        d3d::ResourceStateRegistry states;
        bridge::BridgeOptions options = counting_options(9000);
        options.drop_null_storage_dispatch = true;
        bridge::StateBridgeDevice device(heaps, options, &states);
        constexpr uint32_t kTex = 312;
        constexpr uint64_t kCs = 0xE5;
        bridge::TextureInfo tex{pso::DxgiFormat::D32_FLOAT, 32, 16, 1, 1, 1, GpuTextureDimension::Dim2D, false};
        tex.storage = false;
        device.register_texture(kTex, tex);
        device.register_shader_module(kCs, 3);
        d3d::DescriptorHeap* view_heap = make_heap(heaps, d3d::DescriptorHeapType::CbvSrvUav, 4, true);
        d3d::ViewDesc vd;
        vd.format = static_cast<uint32_t>(pso::DxgiFormat::D32_FLOAT);
        vd.dimension = d3d::ViewDimension::Texture2D;
        CHECK(heaps.create_unordered_access_view(kTex, 0, &vd, view_heap->cpu_handle(0)) == d3d::Status::Ok);
        d3d::RootSignatureDesc rs;
        d3d::RootParameterDesc table;
        table.parameter_type = d3d::RootParameterType::DescriptorTable;
        d3d::DescriptorRangeDesc range;
        range.range_type = d3d::DescriptorRangeType::Uav;
        range.num_descriptors = 1;
        table.ranges.push_back(range);
        rs.parameters.push_back(table);
        d3d::RootSignature root;
        CHECK(d3d::RootSignature::build(rs, &root).ok());
        bridge::PipelineLayoutDesc layout;
        bridge::GroupLayoutDesc group;
        group.content_id = 0x5EEE;
        GpuBindGroupLayoutEntry entry = layout_entry(
            0, GpuShaderStageBits::Compute, GpuBindingLayoutType::StorageTexture,
            static_cast<uint8_t>(variant == 1 ? GpuStorageTextureAccess::ReadOnly : GpuStorageTextureAccess::WriteOnly),
            static_cast<uint8_t>(GpuTextureViewDimension::Dim2D));
        entry.storage_format = static_cast<uint8_t>(variant == 3   ? pso::WgpuTextureFormat::Rgba8Unorm
                                                    : variant == 2 ? pso::WgpuTextureFormat::R32Uint
                                                                   : pso::WgpuTextureFormat::R32Float);
        group.bindings.push_back({entry, join(0, d3d::DescriptorRangeType::Uav, 0)});
        layout.groups.push_back(group);
        device.register_pipeline_layout(kCs, 0, layout);
        pso::D3D12ComputePsoDesc desc;
        desc.cs = kCs;
        auto cpso = device.create_compute_pipeline_state(desc, &root);
        gpu::CommandEncoder enc;
        bridge::CommandListBridge list(device, enc);
        d3d::DescriptorHeap* visible[] = {view_heap};
        CHECK(list.SetDescriptorHeaps(1, visible) == d3d::Status::Ok);
        CHECK(list.SetComputeRootSignature(&root) == d3d::Status::Ok);
        list.SetComputePipelineState(cpso.get());
        CHECK(list.SetComputeRootDescriptorTable(0, view_heap->gpu_handle(0)) == d3d::Status::Ok);
        list.Dispatch(2, 2, 1);
        list.Dispatch(1, 1, 1);
        list.Close();

        uint32_t render_passes = 0, draws = 0, compute = 0, bad_view = 0;
        for (const Packet& pk : decode(enc)) {
            render_passes += pk.opcode == GpuOpcode::BeginRenderPass ? 1u : 0u;
            draws += pk.opcode == GpuOpcode::Draw ? 1u : 0u;
            compute += pk.opcode == GpuOpcode::DispatchCompute ? 1u : 0u;
            if (pk.opcode == GpuOpcode::BeginRenderPass) {
                const auto p = pod<GpuBeginRenderPassPayload>(pk);
                bad_view += (p.depth_stencil.texture_id != kTex || p.color_count != 0 ||
                             p.depth_stencil.view.aspect != static_cast<uint8_t>(GpuTextureAspect::All))
                                ? 1u
                                : 0u;
            }
        }
        const bool twinned = variant != 3;
        const bool writes = variant != 1;
        const bridge::BridgeStats& st = list.stats();
        CHECK(st.storage_views_depth_untwinned == (twinned ? 0u : 1u));
        CHECK(st.depth_twin_bindings == (twinned ? 1u : 0u));
        CHECK(st.depth_twin_copies_in == (twinned ? 1u : 0u));  // once: the twin stays equal afterwards
        CHECK(st.depth_twin_copies_out == (twinned && writes ? 2u : 0u));
        CHECK(render_passes == st.depth_twin_copies_out && draws == st.depth_twin_copies_out && bad_view == 0);
        CHECK(compute == (twinned ? 2u + st.depth_twin_copies_in : 0u));  // the stages' own + the copy in
        CHECK(st.dispatches == (twinned ? 2u : 0u));
    }
}

// Backlog 35 follow-up: a mipped, layered depth texture twins per subresource. A 2D-array storage
// view of mip 1, layers 1..2 copies exactly those two subresources in (once), writes them back one
// draw each into the matching mip / layer, and a render pass that writes the depth of one of them
// makes only that one stale again.
void test_depth_storage_twin_subresources() {
    d3d::DescriptorHeapRegistry heaps;
    d3d::ResourceStateRegistry states;
    bridge::BridgeOptions options = counting_options(9500);
    bridge::StateBridgeDevice device(heaps, options, &states);
    constexpr uint32_t kTex = 313;
    constexpr uint64_t kCs = 0xE6;
    bridge::TextureInfo tex{pso::DxgiFormat::D32_FLOAT, 32, 16, 4, 3, 1, GpuTextureDimension::Dim2D, false};
    device.register_texture(kTex, tex);
    device.register_shader_module(kCs, 3);
    d3d::DescriptorHeap* view_heap = make_heap(heaps, d3d::DescriptorHeapType::CbvSrvUav, 4, true);
    d3d::ViewDesc vd;
    vd.format = static_cast<uint32_t>(pso::DxgiFormat::D32_FLOAT);
    vd.dimension = d3d::ViewDimension::Texture2DArray;
    vd.first_mip = 1;
    vd.mip_levels = 1;
    vd.first_array_slice = 1;
    vd.array_size = 2;
    CHECK(heaps.create_unordered_access_view(kTex, 0, &vd, view_heap->cpu_handle(0)) == d3d::Status::Ok);
    d3d::RootSignatureDesc rs;
    d3d::RootParameterDesc table;
    table.parameter_type = d3d::RootParameterType::DescriptorTable;
    d3d::DescriptorRangeDesc range;
    range.range_type = d3d::DescriptorRangeType::Uav;
    range.num_descriptors = 1;
    table.ranges.push_back(range);
    rs.parameters.push_back(table);
    d3d::RootSignature root;
    CHECK(d3d::RootSignature::build(rs, &root).ok());
    bridge::PipelineLayoutDesc layout;
    bridge::GroupLayoutDesc group;
    group.content_id = 0x5EEF;
    GpuBindGroupLayoutEntry entry = layout_entry(
        0, GpuShaderStageBits::Compute, GpuBindingLayoutType::StorageTexture,
        static_cast<uint8_t>(GpuStorageTextureAccess::WriteOnly), static_cast<uint8_t>(GpuTextureViewDimension::Dim2DArray));
    entry.storage_format = static_cast<uint8_t>(pso::WgpuTextureFormat::R32Float);
    group.bindings.push_back({entry, join(0, d3d::DescriptorRangeType::Uav, 0)});
    layout.groups.push_back(group);
    device.register_pipeline_layout(kCs, 0, layout);
    pso::D3D12ComputePsoDesc desc;
    desc.cs = kCs;
    auto cpso = device.create_compute_pipeline_state(desc, &root);
    gpu::CommandEncoder enc;
    bridge::CommandListBridge list(device, enc);
    d3d::DescriptorHeap* visible[] = {view_heap};
    CHECK(list.SetDescriptorHeaps(1, visible) == d3d::Status::Ok);
    CHECK(list.SetComputeRootSignature(&root) == d3d::Status::Ok);
    list.SetComputePipelineState(cpso.get());
    CHECK(list.SetComputeRootDescriptorTable(0, view_heap->gpu_handle(0)) == d3d::Status::Ok);
    list.Dispatch(1, 1, 1);
    list.Dispatch(1, 1, 1);
    // A read-back of the depth texture finds nothing pending: both writes are already back in it.
    const size_t before_readback = decode(enc).size();
    GpuReadbackTexturePayload rb{};
    rb.texture_id = kTex;
    CHECK(list.Readback(GpuOpcode::ReadbackTexture, &rb, sizeof(rb), kTex) != SIZE_MAX);
    list.Close();

    uint32_t passes = 0, bad_view = 0;
    bool layer1 = false, layer2 = false;
    size_t readback_at = 0, i = 0;
    for (const Packet& pk : decode(enc)) {
        if (pk.opcode == GpuOpcode::BeginRenderPass) {
            const auto p = pod<GpuBeginRenderPassPayload>(pk);
            ++passes;
            bad_view += (p.depth_stencil.texture_id != kTex || p.depth_stencil.view.base_mip_level != 1 ||
                         p.depth_stencil.view.array_layer_count != 1)
                            ? 1u
                            : 0u;
            layer1 |= p.depth_stencil.view.base_array_layer == 1;
            layer2 |= p.depth_stencil.view.base_array_layer == 2;
        }
        if (pk.opcode == GpuOpcode::ReadbackTexture) {
            readback_at = i;
        }
        ++i;
    }
    const bridge::BridgeStats& st = list.stats();
    CHECK(st.storage_views_depth_untwinned == 0u);
    CHECK(st.depth_twin_bindings == 1u);
    CHECK(st.depth_twin_copies_in == 2u);   // layers 1 and 2 of mip 1, once
    CHECK(st.depth_twin_copies_out == 4u);  // two layers after each of the two dispatches
    CHECK(passes == 4u && bad_view == 0u && layer1 && layer2);
    CHECK(readback_at >= before_readback);
}

// #482: a fragment-stage storage binding of a mismatched float format goes through an intermediate
// too: copied in (read-write) before the draw in a pass of its own, converted out when the pass ends.
void test_fragment_storage_texture_format_conversion() {
    d3d::DescriptorHeapRegistry heaps;
    d3d::ResourceStateRegistry states;
    bridge::StateBridgeDevice device(heaps, counting_options(7000), &states);
    constexpr uint32_t kTarget = 100, kTex = 302;
    constexpr uint64_t kVs = 0xE1, kPs = 0xE2;
    device.register_texture(kTarget, {pso::DxgiFormat::B8G8R8A8_UNORM, 64, 64, 1, 1, 1,
                                      GpuTextureDimension::Dim2D, false});
    bridge::TextureInfo tex{pso::DxgiFormat::R16G16B16A16_FLOAT, 32, 32, 1, 1, 1, GpuTextureDimension::Dim2D, false};
    tex.storage = true;
    device.register_texture(kTex, tex);
    device.register_shader_module(kVs, 1);
    device.register_shader_module(kPs, 2);
    d3d::DescriptorHeap* rtv_heap = make_heap(heaps, d3d::DescriptorHeapType::Rtv, 1, false);
    d3d::DescriptorHeap* view_heap = make_heap(heaps, d3d::DescriptorHeapType::CbvSrvUav, 4, true);
    CHECK(heaps.create_render_target_view(kTarget, nullptr, rtv_heap->cpu_handle(0)) == d3d::Status::Ok);
    d3d::ViewDesc vd;
    vd.format = static_cast<uint32_t>(tex.format);
    vd.dimension = d3d::ViewDimension::Texture2D;
    CHECK(heaps.create_unordered_access_view(kTex, 0, &vd, view_heap->cpu_handle(0)) == d3d::Status::Ok);
    d3d::RootSignatureDesc rs;
    d3d::RootParameterDesc table;
    table.parameter_type = d3d::RootParameterType::DescriptorTable;
    d3d::DescriptorRangeDesc range;
    range.range_type = d3d::DescriptorRangeType::Uav;
    range.num_descriptors = 1;
    table.ranges.push_back(range);
    rs.parameters.push_back(table);
    d3d::RootSignature root;
    CHECK(d3d::RootSignature::build(rs, &root).ok());
    bridge::PipelineLayoutDesc layout;
    bridge::GroupLayoutDesc group;
    GpuBindGroupLayoutEntry entry =
        layout_entry(0, GpuShaderStageBits::Fragment, GpuBindingLayoutType::StorageTexture,
                     static_cast<uint8_t>(GpuStorageTextureAccess::ReadWrite),
                     static_cast<uint8_t>(GpuTextureViewDimension::Dim2D));
    entry.storage_format = static_cast<uint8_t>(pso::WgpuTextureFormat::Rgba8Unorm);
    group.bindings.push_back({entry, join(0, d3d::DescriptorRangeType::Uav, 0)});
    layout.groups.push_back(group);
    device.register_pipeline_layout(kVs, kPs, layout);
    pso::D3D12GraphicsPsoDesc d = Scene::pso_desc();
    d.vs = kVs;
    d.ps = kPs;
    d.DSVFormat = pso::DxgiFormat::UNKNOWN;
    auto state = device.create_graphics_pipeline_state(d, &root);
    gpu::CommandEncoder enc;
    bridge::CommandListBridge list(device, enc);
    d3d::DescriptorHeap* visible[] = {view_heap};
    CHECK(list.SetDescriptorHeaps(1, visible) == d3d::Status::Ok);
    CHECK(list.SetGraphicsRootSignature(&root) == d3d::Status::Ok);
    list.SetPipelineState(state.get());
    CHECK(list.SetGraphicsRootDescriptorTable(0, view_heap->gpu_handle(0)) == d3d::Status::Ok);
    list.IASetPrimitiveTopology(pso::D3DPrimitiveTopology::TriangleList);
    const gpu::d3d12_xs::D3D12VertexBufferView vb{Scene::kVertexBuffer, 0, 36, 12};
    list.IASetVertexBuffers(0, 1, &vb);
    const d3d::CpuDescriptorHandle rtv0 = rtv_heap->cpu_handle(0);
    list.OMSetRenderTargets(1, &rtv0, false, nullptr);
    list.DrawInstanced(3, 1, 0, 0);
    list.DrawInstanced(3, 1, 0, 0);
    list.Close();

    CHECK(list.stats().storage_views_format_mismatch == 0 && list.stats().storage_views_without_usage == 0);
    CHECK(list.stats().storage_views_converted == 1 && list.stats().draws == 2);
    CHECK(list.stats().storage_conversions == 2);  // one in before the first draw, one out at the pass end
    // Order: copy-in dispatch, then the render pass with both draws, then the copy-out dispatch.
    std::string order;
    for (const Packet& pk : decode(enc)) {
        switch (pk.opcode) {
        case GpuOpcode::DispatchCompute: order += 'C'; break;
        case GpuOpcode::BeginRenderPass: order += '['; break;
        case GpuOpcode::EndRenderPass: order += ']'; break;
        case GpuOpcode::Draw: order += 'D'; break;
        default: break;
        }
    }
    CHECK(order.find("C[DD]C") != std::string::npos);
}

// #482: a mixed vertex+fragment storage binding is converted too; when the texture is also the pass's
// render attachment it is counted as an alias and not converted.
void test_vertex_mixed_storage_and_attachment_alias() {
    d3d::DescriptorHeapRegistry heaps;
    d3d::ResourceStateRegistry states;
    bridge::StateBridgeDevice device(heaps, counting_options(7000), &states);
    constexpr uint32_t kTarget = 100, kTex = 302;
    constexpr uint64_t kVs = 0xE1, kPs = 0xE2;
    device.register_texture(kTarget, {pso::DxgiFormat::B8G8R8A8_UNORM, 64, 64, 1, 1, 1,
                                      GpuTextureDimension::Dim2D, false});
    bridge::TextureInfo tex{pso::DxgiFormat::R16G16B16A16_FLOAT, 32, 32, 1, 1, 1, GpuTextureDimension::Dim2D, false};
    tex.storage = true;
    device.register_texture(kTex, tex);
    device.register_shader_module(kVs, 1);
    device.register_shader_module(kPs, 2);
    d3d::DescriptorHeap* rtv_heap = make_heap(heaps, d3d::DescriptorHeapType::Rtv, 2, false);
    d3d::DescriptorHeap* view_heap = make_heap(heaps, d3d::DescriptorHeapType::CbvSrvUav, 4, true);
    CHECK(heaps.create_render_target_view(kTarget, nullptr, rtv_heap->cpu_handle(0)) == d3d::Status::Ok);
    d3d::ViewDesc vd;
    vd.format = static_cast<uint32_t>(tex.format);
    vd.dimension = d3d::ViewDimension::Texture2D;
    CHECK(heaps.create_unordered_access_view(kTex, 0, &vd, view_heap->cpu_handle(0)) == d3d::Status::Ok);
    d3d::RootSignatureDesc rs;
    d3d::RootParameterDesc table;
    table.parameter_type = d3d::RootParameterType::DescriptorTable;
    d3d::DescriptorRangeDesc range;
    range.range_type = d3d::DescriptorRangeType::Uav;
    range.num_descriptors = 1;
    table.ranges.push_back(range);
    rs.parameters.push_back(table);
    d3d::RootSignature root;
    CHECK(d3d::RootSignature::build(rs, &root).ok());
    bridge::PipelineLayoutDesc layout;
    bridge::GroupLayoutDesc group;
    GpuBindGroupLayoutEntry entry =
        layout_entry(0, GpuShaderStageBits::Vertex | GpuShaderStageBits::Fragment, GpuBindingLayoutType::StorageTexture,
                     static_cast<uint8_t>(GpuStorageTextureAccess::ReadWrite),
                     static_cast<uint8_t>(GpuTextureViewDimension::Dim2D));
    entry.storage_format = static_cast<uint8_t>(pso::WgpuTextureFormat::Rgba8Unorm);
    group.bindings.push_back({entry, join(0, d3d::DescriptorRangeType::Uav, 0)});
    layout.groups.push_back(group);
    device.register_pipeline_layout(kVs, kPs, layout);
    pso::D3D12GraphicsPsoDesc d = Scene::pso_desc();
    d.vs = kVs;
    d.ps = kPs;
    d.DSVFormat = pso::DxgiFormat::UNKNOWN;
    auto state = device.create_graphics_pipeline_state(d, &root);
    gpu::CommandEncoder enc;
    bridge::CommandListBridge list(device, enc);
    d3d::DescriptorHeap* visible[] = {view_heap};
    CHECK(list.SetDescriptorHeaps(1, visible) == d3d::Status::Ok);
    CHECK(list.SetGraphicsRootSignature(&root) == d3d::Status::Ok);
    list.SetPipelineState(state.get());
    CHECK(list.SetGraphicsRootDescriptorTable(0, view_heap->gpu_handle(0)) == d3d::Status::Ok);
    list.IASetPrimitiveTopology(pso::D3DPrimitiveTopology::TriangleList);
    const gpu::d3d12_xs::D3D12VertexBufferView vb{Scene::kVertexBuffer, 0, 36, 12};
    list.IASetVertexBuffers(0, 1, &vb);
    const d3d::CpuDescriptorHandle rtv0 = rtv_heap->cpu_handle(0);
    list.OMSetRenderTargets(1, &rtv0, false, nullptr);
    list.DrawInstanced(3, 1, 0, 0);
    list.Close();
    CHECK(list.stats().storage_conversions == 2 && list.stats().storage_views_attachment_alias == 0);

    gpu::CommandEncoder enc2;
    bridge::CommandListBridge list2(device, enc2);
    CHECK(list2.SetDescriptorHeaps(1, visible) == d3d::Status::Ok);
    CHECK(list2.SetGraphicsRootSignature(&root) == d3d::Status::Ok);
    list2.SetPipelineState(state.get());
    CHECK(list2.SetGraphicsRootDescriptorTable(0, view_heap->gpu_handle(0)) == d3d::Status::Ok);
    list2.IASetPrimitiveTopology(pso::D3DPrimitiveTopology::TriangleList);
    list2.IASetVertexBuffers(0, 1, &vb);
    CHECK(heaps.create_render_target_view(kTex, nullptr, rtv_heap->cpu_handle(1)) == d3d::Status::Ok);
    const d3d::CpuDescriptorHandle rtv1 = rtv_heap->cpu_handle(1);
    list2.OMSetRenderTargets(1, &rtv1, false, nullptr);
    list2.DrawInstanced(3, 1, 0, 0);
    list2.Close();
    CHECK(list2.stats().storage_views_attachment_alias >= 1 && list2.stats().storage_conversions == 0);
}

// #691: a glyph atlas (A8_UNORM, console linear, widened to rgba8) the title fills a few glyphs at a
// time uploads only the changed bands of kUploadBandRows rows, not the whole texture per glyph.
void test_banded_texture_upload() {
    namespace com = playgta6::guest::com;
    namespace dg = playgta6::gpu::d3d12_guest;
    namespace gfx10 = playgta6::gpu::gfx10;

    os::VirtualMemoryArena arena;
    com::reset_com_bridge();
    com::set_guest_memory(&arena);
    {
        dg::DeviceOptions dev_opts;
        dev_opts.sink = [](const uint8_t*, size_t) {};
        auto device = com::create_object<dg::Device>(com::implements(dg::kID3D12Device), dev_opts);
        CHECK(device);
        device->initialize_guest_payload();

        constexpr uint32_t kW = 256, kH = 256, kBand = dg::kUploadBandRows;
        static_assert(kH == 4 * kBand, "four bands");
        gfx10::TextureLayoutDesc g{};
        g.dxgi_format = 65;  // A8_UNORM
        g.width = kW;
        g.height = kH;
        g.mip_levels = 1;
        g.d3d12_layout = 0x100;
        const size_t surface_size = gfx10::surface_bytes(g);
        CHECK(surface_size != 0 && surface_size <= (1u << 20));
        std::vector<uint8_t> texels(size_t{kW} * kH, 0);
        std::vector<uint8_t> surface(surface_size, 0);
        const uint64_t mem = arena.virtual_alloc(0, 1u << 20, os::kMemReserve | os::kMemCommit, os::PageProtect::ReadWrite);
        auto glyph = [&](uint32_t y0, uint32_t rows, uint8_t v) {
            for (uint32_t y = y0; y < y0 + rows; ++y) {
                for (uint32_t x = 10; x < 20; ++x) {
                    texels[size_t{y} * kW + x] = v;
                }
            }
            CHECK(gfx10::tile_surface_subresource(g, 0, 0, texels.data(), texels.size(), surface.data(), surface.size()));
            CHECK(com::write_guest(mem, surface.data(), surface.size()));
        };

        dg::GuestResourceDesc d{};
        d.Dimension = dg::ResourceDimension::Texture2D;
        d.Width = kW;
        d.Height = kH;
        d.DepthOrArraySize = 1;
        d.MipLevels = 1;
        d.Format = 65;
        d.SampleCount = 1;
        d.Layout = 0x100;
        dg::Resource::Placement pl{};
        pl.address = mem;
        auto tex = com::create_object<dg::Resource>(com::implements(dg::kID3D12Resource), dg::device_ref(device.get()),
                                                    d, pl, 0u);
        CHECK(tex && tex->initialize());

        auto writes = [](const gpu::CommandEncoder& enc) {
            std::vector<Packet> out;
            for (const Packet& pk : decode(enc)) {
                if (pk.opcode == GpuOpcode::WriteTexture) {
                    out.push_back(pk);
                }
            }
            return out;
        };
        // Every row of a write holds the widened texels (0, 0, 0, a) of its rows.
        auto check_rows = [&](const Packet& pk, uint32_t y0, uint32_t rows) {
            const auto wt = pod<GpuWriteTexturePayload>(pk);
            CHECK(wt.texture_id == tex->id() && wt.origin_x == 0 && wt.origin_y == y0 && wt.width == kW &&
                  wt.height == rows && wt.bytes_per_row == kW * 4u && wt.rows_per_image == rows);
            bool same = pk.payload.size() >= sizeof(GpuWriteTexturePayload) + size_t{rows} * kW * 4u;
            for (uint32_t y = 0; same && y < rows; ++y) {
                const uint8_t* row = pk.payload.data() + sizeof(GpuWriteTexturePayload) + size_t{y} * kW * 4u;
                for (uint32_t x = 0; same && x < kW; ++x) {
                    same = row[x * 4u] == 0 && row[x * 4u + 1] == 0 && row[x * 4u + 2] == 0 &&
                           row[x * 4u + 3] == texels[size_t{y0 + y} * kW + x];
                }
            }
            CHECK(same);
        };

        // First glyph: the whole texture (never uploaded before).
        glyph(2, 12, 200);
        gpu::CommandEncoder e1;
        CHECK(device->sync_bound_texture_memory(tex->id(), e1));
        const auto w1 = writes(e1);
        CHECK(w1.size() == 1);
        if (w1.size() == 1) {
            check_rows(w1[0], 0, kH);
        }

        // A glyph inside the second band: that band only.
        glyph(kBand + 6, 12, 150);
        gpu::CommandEncoder e2;
        CHECK(device->sync_bound_texture_memory(tex->id(), e2));
        const auto w2 = writes(e2);
        CHECK(w2.size() == 1);
        if (w2.size() == 1) {
            check_rows(w2[0], kBand, kBand);
        }

        // One glyph across bands 2 and 3 plus one in band 0: two writes, the adjacent bands merged.
        glyph(3 * kBand - 4, 8, 90);
        glyph(30, 4, 60);
        gpu::CommandEncoder e3;
        CHECK(device->sync_bound_texture_memory(tex->id(), e3));
        const auto w3 = writes(e3);
        CHECK(w3.size() == 2);
        if (w3.size() == 2) {
            check_rows(w3[0], 0, kBand);
            check_rows(w3[1], 2 * kBand, 2 * kBand);
        }

        // Steady state: nothing.
        gpu::CommandEncoder e4;
        CHECK(!device->sync_bound_texture_memory(tex->id(), e4));
        CHECK(writes(e4).empty());
    }
    com::reset_com_bridge();
    com::set_guest_memory(nullptr);
}

// O36: views die with their texture: create/destroy N textures with views, the table stays small.
void test_view_table_bounded() {
    d3d::DescriptorHeapRegistry heaps;
    bridge::StateBridgeDevice device(heaps, bridge::BridgeOptions{});
    for (uint32_t n = 1; n <= 5000; ++n) {
        bridge::ViewRecord rec{};
        rec.texture_id = n;
        rec.resource_id = n;
        CHECK(device.test_intern_view(rec) != 0);
        device.unregister_resource(n);
        CHECK(device.live_view_count() == 0);
    }
    // A recycled id resolves to the new view, and the same texture re-interns to one id.
    bridge::ViewRecord rec{};
    rec.texture_id = 9000;
    const uint32_t a = device.test_intern_view(rec);
    CHECK(device.test_intern_view(rec) == a && a <= 2u && device.view(a)->texture_id == 9000u);
}

// G70 (#691): unregistering/evicting a texture immediately invalidates every cached bind group
// referencing it (both directly and via interned view ids), retires the evicted bind groups with
// zero GPU leak, and prevents stale bind groups or destroyed texture ids from being bound.
void test_unregister_texture_invalidates_cached_bind_groups() {
    // Part A: direct texture unregister invalidates cached bind groups in CommandListBridge.
    {
        Scene scene;
        gpu::CommandEncoder enc;
        bridge::CommandListBridge list(scene.device, enc);
        scene.bind_root(list);
        const d3d::CpuDescriptorHandle rtv0 = scene.rtv(0);
        list.OMSetRenderTargets(1, &rtv0, false, nullptr);

        // Draw 1: creates and caches bind group bg1 referencing kAlbedo.
        list.DrawInstanced(3, 1, 0, 0);
        uint32_t bg1 = 0;
        for (const Packet& pk : decode(enc)) {
            if (pk.opcode == GpuOpcode::CreateBindGroup2) {
                bg1 = pod<GpuCreateBindGroup2Payload>(pk).id;
            }
        }
        CHECK(bg1 != 0);
        CHECK(list.cached_bind_group_count() == 1);
        const size_t views_before = scene.device.live_view_count();
        CHECK(views_before >= 2u);  // swapchain RTV + kAlbedo SRV

        // Draw 2: cache hit (only Draw emitted).
        const size_t mark_hit = enc.size();
        list.DrawInstanced(3, 1, 0, 0);
        CHECK(opcodes(decode(enc, mark_hit)) == std::vector<GpuOpcode>{GpuOpcode::Draw});
        CHECK(list.stats().bind_group_cache_hits == 1);

        // Unregister kAlbedo: must immediately evict bg1 from BindGroupCache, retire its interned
        // view, and queue bg1 for DestroyResource (zero GPU leak).
        scene.device.unregister_resource(Scene::kAlbedo);
        CHECK(list.cached_bind_group_count() == 0);
        CHECK(list.stats().bind_groups_invalidated == 1);
        CHECK(scene.device.live_view_count() < views_before);
        CHECK(scene.device.deferred_destroy_count() == 1);

        // Draw 3: heap slot still names unregistered kAlbedo; cache must miss and emit a fresh bind
        // group with null resource_id (0), NEVER reusing bg1 or binding destroyed kAlbedo.
        const size_t mark_null = enc.size();
        list.DrawInstanced(3, 1, 0, 0);
        uint32_t bg2 = 0;
        GpuBindGroupEntry2 tex_entry{};
        bool found_bg2 = false;
        for (const Packet& pk : decode(enc, mark_null)) {
            if (pk.opcode == GpuOpcode::CreateBindGroup2) {
                bg2 = pod<GpuCreateBindGroup2Payload>(pk).id;
                tex_entry = pod<GpuBindGroupEntry2>(pk, sizeof(GpuCreateBindGroup2Payload) + sizeof(GpuBindGroupEntry2));
                found_bg2 = true;
            }
        }
        CHECK(found_bg2 && bg2 != 0 && bg2 != bg1);
        CHECK(tex_entry.resource_id == 0);
        CHECK((list.diagnostics() & bridge::BridgeDiag::UnknownTexture) != 0);

        // Point the heap slot at live kAlbedo2: rebuilds cleanly to kAlbedo2.
        scene.write_albedo_srv(Scene::kAlbedo2, 0);
        const size_t mark_live = enc.size();
        list.DrawInstanced(3, 1, 0, 0);
        uint32_t bg3 = 0;
        GpuBindGroupEntry2 tex_entry3{};
        for (const Packet& pk : decode(enc, mark_live)) {
            if (pk.opcode == GpuOpcode::CreateBindGroup2) {
                bg3 = pod<GpuCreateBindGroup2Payload>(pk).id;
                tex_entry3 = pod<GpuBindGroupEntry2>(pk, sizeof(GpuCreateBindGroup2Payload) + sizeof(GpuBindGroupEntry2));
            }
        }
        CHECK(bg3 != 0 && bg3 != bg1 && bg3 != bg2);
        CHECK(tex_entry3.resource_id == Scene::kAlbedo2);

        // Present flushes deferred DestroyResource(bg1) strictly after Present.
        list.Present();
        const std::vector<Packet> all = decode(enc);
        size_t present_idx = all.size();
        size_t destroy_bg1_idx = all.size();
        for (size_t i = 0; i < all.size(); ++i) {
            if (all[i].opcode == GpuOpcode::Present) {
                present_idx = i;
            } else if (all[i].opcode == GpuOpcode::DestroyResource &&
                       pod<GpuDestroyResourcePayload>(all[i]).id == bg1) {
                destroy_bg1_idx = i;
            }
        }
        CHECK(present_idx < all.size());
        CHECK(destroy_bg1_idx < all.size() && destroy_bg1_idx > present_idx);
        CHECK(scene.device.deferred_destroy_count() == 0);
    }

    // Part B: BindGroupCache invalidates entries matching an interned view id even when resource_ids is empty.
    {
        d3d::DescriptorHeapRegistry heaps;
        d3d::DescriptorHeap* srv_heap = make_heap(heaps, d3d::DescriptorHeapType::CbvSrvUav, 4, true);
        d3d::ViewDesc srv{};
        srv.format = 28;
        srv.dimension = d3d::ViewDimension::Texture2D;
        CHECK(heaps.create_shader_resource_view(555u, &srv, srv_heap->cpu_handle(0)) == d3d::Status::Ok);
        d3d::RootSignatureDesc rs_desc;
        d3d::RootParameterDesc table;
        table.parameter_type = d3d::RootParameterType::DescriptorTable;
        d3d::DescriptorRangeDesc range;
        range.range_type = d3d::DescriptorRangeType::Srv;
        range.num_descriptors = 1;
        table.ranges.push_back(range);
        rs_desc.parameters.push_back(table);
        d3d::RootSignature rs;
        CHECK(d3d::RootSignature::build(rs_desc, &rs).ok());
        const d3d::ShaderBinding join_srv = join(0, d3d::DescriptorRangeType::Srv, 0);
        const d3d::BindGroupPlan plan = d3d::BindGroupPlan::build(rs, std::span<const d3d::ShaderBinding>(&join_srv, 1),
                                                                  d3d::WebGpuStage::Fragment);
        d3d::RootArgumentState root_args;
        d3d::DescriptorHeap* visible[] = {srv_heap};
        CHECK(root_args.set_descriptor_heaps(1, visible) == d3d::Status::Ok);
        CHECK(root_args.set_root_signature(d3d::BindPoint::Graphics, &rs) == d3d::Status::Ok);
        CHECK(root_args.set_descriptor_table(d3d::BindPoint::Graphics, 0, srv_heap->gpu_handle(0)) == d3d::Status::Ok);

        d3d::BindGroupCache cache;
        d3d::BindGroupLookup lk = cache.get_or_build(root_args, d3d::BindPoint::Graphics, plan, 1u, nullptr, nullptr);
        CHECK(lk.status == d3d::Status::Ok && lk.bind_group != nullptr);
        lk.bind_group->gpu_id = 7777u;
        lk.bind_group->view_ids.push_back(42u);
        cache.note_group_ids(*lk.bind_group);  // ids added after the lookup go through the eviction filter
        CHECK(cache.size() == 1);

        const uint32_t retired_view = 42u;
        std::vector<uint32_t> retired_bgs;
        const size_t evicted = cache.evict_resources_and_views({}, std::span<const uint32_t>(&retired_view, 1),
                                                               &retired_bgs);
        CHECK(evicted == 1 && cache.size() == 0);
        CHECK(retired_bgs.size() == 1 && retired_bgs[0] == 7777u);
    }

    // Part C: memory-backed descriptor tables force-refresh when their texture id is unregistered
    // and replaced at the same GPU VA even though the raw descriptor memory bytes did not change.
    {
        d3d::DescriptorHeapRegistry heaps;
        d3d::ResourceStateRegistry states;
        bridge::BridgeOptions options = counting_options(8000);
        struct Holder {
            std::unique_ptr<TestSlotSource> source;
        } holder;
        struct Forward final : d3d::DescriptorSlotSource {
            Holder* h = nullptr;
            bool reads(const d3d::DescriptorHeap& heap) const override { return h->source && h->source->reads(heap); }
            bool snapshot(const d3d::DescriptorHeap& heap, uint32_t f, uint32_t c, std::vector<uint8_t>* o) const override {
                return h->source->snapshot(heap, f, c, o);
            }
            d3d::ViewDescriptor view(const d3d::DescriptorHeap& heap, uint32_t i, d3d::DescriptorRangeType r) const override {
                return h->source->view(heap, i, r);
            }
            bool sampler(const d3d::DescriptorHeap& heap, uint32_t i, d3d::SamplerDesc* o) const override {
                return h->source->sampler(heap, i, o);
            }
        } forward;
        forward.h = &holder;
        options.slot_source = &forward;

        bridge::StateBridgeDevice device(heaps, options, &states);
        constexpr uint32_t kTarget = 100, kTex1 = 301, kTex2 = 302;
        constexpr uint64_t kVs = 0xF1, kPs = 0xF2, kTexVa = 0x7E00'0005'0000ull;
        device.register_texture(kTarget, {pso::DxgiFormat::B8G8R8A8_UNORM, 64, 64, 1, 1, 1, GpuTextureDimension::Dim2D, false});
        bridge::TextureInfo tex{pso::DxgiFormat::R32G32B32A32_FLOAT, 256, 1, 1, 1, 1, GpuTextureDimension::Dim2D, false};
        tex.gpu_va = kTexVa;
        device.register_texture(kTex1, tex);
        device.register_shader_module(kVs, 1);
        device.register_shader_module(kPs, 2);

        d3d::DescriptorHeap* rtv_heap = make_heap(heaps, d3d::DescriptorHeapType::Rtv, 1, false);
        d3d::DescriptorHeap* heap = make_heap(heaps, d3d::DescriptorHeapType::CbvSrvUav, 4, true);
        holder.source = std::make_unique<TestSlotSource>(device, heap);
        CHECK(heaps.create_render_target_view(kTarget, nullptr, rtv_heap->cpu_handle(0)) == d3d::Status::Ok);

        d3d::ViewDescriptor img{};
        img.kind = d3d::DescriptorKind::Srv;
        img.resource_id = kTex1;
        img.dimension = d3d::ViewDimension::Texture2D;
        img.format = 2;  // R32G32B32A32_FLOAT
        img.component_mapping = d3d::kDefaultShader4ComponentMapping;
        img.mip_levels = 1;
        uint32_t words[8] = {};
        CHECK(bridge::view_words(device, img, words) == bridge::WordsStatus::Ok);
        holder.source->write(0, words, sizeof(words));

        d3d::RootSignatureDesc rs_desc;
        d3d::RootParameterDesc table;
        table.parameter_type = d3d::RootParameterType::DescriptorTable;
        d3d::DescriptorRangeDesc range;
        range.range_type = d3d::DescriptorRangeType::Srv;
        range.num_descriptors = 1;
        table.ranges.push_back(range);
        rs_desc.parameters.push_back(table);
        d3d::RootSignature rs;
        CHECK(d3d::RootSignature::build(rs_desc, &rs).ok());

        bridge::PipelineLayoutDesc layout;
        bridge::GroupLayoutDesc group;
        group.bindings.push_back({layout_entry(0, GpuShaderStageBits::Fragment, GpuBindingLayoutType::Texture,
                                               static_cast<uint8_t>(GpuTextureSampleType::UnfilterableFloat),
                                               static_cast<uint8_t>(GpuTextureViewDimension::Dim2D)),
                                  join(0, d3d::DescriptorRangeType::Srv, 0)});
        layout.groups.push_back(group);
        device.register_pipeline_layout(kVs, kPs, layout);
        pso::D3D12GraphicsPsoDesc pd = Scene::pso_desc();
        pd.vs = kVs;
        pd.ps = kPs;
        pd.DSVFormat = pso::DxgiFormat::UNKNOWN;
        auto state = device.create_graphics_pipeline_state(pd, &rs);

        gpu::CommandEncoder enc;
        bridge::CommandListBridge list(device, enc);
        d3d::DescriptorHeap* visible[] = {heap};
        CHECK(list.SetDescriptorHeaps(1, visible) == d3d::Status::Ok);
        CHECK(list.SetGraphicsRootSignature(&rs) == d3d::Status::Ok);
        list.SetPipelineState(state.get());
        CHECK(list.SetGraphicsRootDescriptorTable(0, heap->gpu_handle(0)) == d3d::Status::Ok);
        list.IASetPrimitiveTopology(pso::D3DPrimitiveTopology::TriangleList);
        const gpu::d3d12_xs::D3D12VertexBufferView vb{Scene::kVertexBuffer, 0, 36, 12};
        list.IASetVertexBuffers(0, 1, &vb);
        const d3d::CpuDescriptorHandle rtv0 = rtv_heap->cpu_handle(0);
        list.OMSetRenderTargets(1, &rtv0, false, nullptr);
        list.DrawInstanced(3, 1, 0, 0);
        list.Close();
        uint32_t bg1 = 0;
        CHECK(last_bind_group_entry(enc, &bg1).resource_id == kTex1);
        CHECK(list.cached_bind_group_count() == 1);

        // Evict kTex1 and re-create a replacement kTex2 at the same GPU VA (descriptor words unchanged).
        device.unregister_resource(kTex1);
        CHECK(list.cached_bind_group_count() == 0);
        device.register_texture(kTex2, tex);

        // refresh_memory_bind_groups must detect the invalidation and patch the recorded bind group to kTex2.
        gpu::CommandEncoder prelude;
        CHECK(list.refresh_memory_bind_groups(prelude) == 1);
        CHECK(last_bind_group_entry(enc).resource_id == kTex2);
    }
}

// G70 (#691): a texture bound in a draw and evicted before Present must have its DestroyResource
// packet emitted strictly AFTER Present (never before), both at the StateBridge layer and through
// d3d12_guest::Device::ensure_address_texture LRU eviction.
void test_evicted_texture_destroy_deferred_until_after_present() {
    // Part A: StateBridgeDevice + CommandListBridge deferral before Present, immediate destroy for unused.
    {
        Scene scene;
        gpu::CommandEncoder enc;
        bridge::CommandListBridge list(scene.device, enc);
        scene.bind_root(list);
        const d3d::CpuDescriptorHandle rtv0 = scene.rtv(0);
        list.OMSetRenderTargets(1, &rtv0, false, nullptr);

        // Unused texture can be destroyed immediately without bloating the deferred queue.
        constexpr uint32_t kUnused = 999u;
        CHECK(!scene.device.is_resource_used_in_frame(kUnused));
        CHECK(!scene.device.destroy_or_defer_resource(kUnused, enc));

        // Draw 1 binds kAlbedo via bg1.
        list.DrawInstanced(3, 1, 0, 0);
        uint32_t bg1 = 0;
        for (const Packet& pk : decode(enc)) {
            if (pk.opcode == GpuOpcode::CreateBindGroup2) {
                bg1 = pod<GpuCreateBindGroup2Payload>(pk).id;
            }
        }
        CHECK(bg1 != 0);
        CHECK(scene.device.is_resource_used_in_frame(Scene::kAlbedo));
        CHECK(scene.device.is_resource_used_in_frame(bg1));

        // Evict kAlbedo BEFORE Present: must defer DestroyResource for both kAlbedo and bg1!
        scene.device.unregister_resource(Scene::kAlbedo);
        CHECK(scene.device.destroy_or_defer_resource(Scene::kAlbedo, enc));
        for (const Packet& pk : decode(enc)) {
            if (pk.opcode == GpuOpcode::DestroyResource) {
                const uint32_t id = pod<GpuDestroyResourcePayload>(pk).id;
                CHECK(id != Scene::kAlbedo && id != bg1);
            }
        }
        CHECK(scene.device.deferred_destroy_count() == 2);

        // Emit Present: DestroyResource(kAlbedo) and DestroyResource(bg1) must appear strictly AFTER Present.
        list.Present();
        const std::vector<Packet> packets = decode(enc);
        size_t present_idx = packets.size();
        size_t destroy_tex_idx = packets.size();
        size_t destroy_bg_idx = packets.size();
        for (size_t i = 0; i < packets.size(); ++i) {
            if (packets[i].opcode == GpuOpcode::Present) {
                present_idx = i;
            } else if (packets[i].opcode == GpuOpcode::DestroyResource) {
                const uint32_t id = pod<GpuDestroyResourcePayload>(packets[i]).id;
                if (id == Scene::kAlbedo) {
                    destroy_tex_idx = i;
                } else if (id == bg1) {
                    destroy_bg_idx = i;
                }
            }
        }
        CHECK(present_idx < packets.size());
        CHECK(destroy_tex_idx < packets.size() && destroy_tex_idx > present_idx);
        CHECK(destroy_bg_idx < packets.size() && destroy_bg_idx > present_idx);
        CHECK(scene.device.deferred_destroy_count() == 0);
    }

    // Part B: multi-frame advance_present_frame preserves resources already recorded into an unexecuted
    // command list of Frame N+1 while flushing deferred destroys from Frame N.
    {
        Scene scene;
        gpu::CommandEncoder enc;
        bridge::CommandListBridge list(scene.device, enc);
        scene.bind_root(list);
        const d3d::CpuDescriptorHandle rtv0 = scene.rtv(0);
        list.OMSetRenderTargets(1, &rtv0, false, nullptr);

        // Frame 0 deferred resource (from a previous frame).
        constexpr uint32_t kOldTex = 888u;
        scene.device.defer_destroy_resource(kOldTex);

        // Frame 1 starts recording into `list` (not yet executed) and binds kAlbedo BEFORE
        // advance_present_frame(1) runs.
        list.DrawInstanced(3, 1, 0, 0);
        CHECK(scene.device.is_resource_used_in_frame(Scene::kAlbedo));

        gpu::CommandEncoder flush_enc;
        CHECK(scene.device.advance_present_frame(1, flush_enc));
        // kOldTex is flushed, while kAlbedo stays pinned in-frame for the unexecuted command list!
        bool saw_old = false;
        for (const Packet& pk : decode(flush_enc)) {
            if (pk.opcode == GpuOpcode::DestroyResource && pod<GpuDestroyResourcePayload>(pk).id == kOldTex) {
                saw_old = true;
            }
        }
        CHECK(saw_old);
        CHECK(scene.device.is_resource_used_in_frame(Scene::kAlbedo));
        CHECK(scene.device.destroy_or_defer_resource(Scene::kAlbedo, flush_enc));
        list.Close();
    }

    // Part C: full d3d12_guest::Device::ensure_address_texture LRU eviction of a 256x1 RGBA32Float
    // texture used in Frame 0: LRU prefers non-in-frame victims, and any forced in-frame eviction
    // emits DestroyResource strictly AFTER Present, never before.
    {
        namespace com = playgta6::guest::com;
        namespace dg = playgta6::gpu::d3d12_guest;

        os::VirtualMemoryArena arena;
        com::reset_com_bridge();
        com::set_guest_memory(&arena);
        {
            std::vector<uint8_t> stream;
            dg::DeviceOptions dev_opts;
            dev_opts.sink = [&stream](const uint8_t* data, size_t size) {
                stream.insert(stream.end(), data, data + size);
            };
            auto device = com::create_object<dg::Device>(com::implements(dg::kID3D12Device), dev_opts);
            CHECK(device);
            device->initialize_guest_payload();

            auto decode_stream = [](const std::vector<uint8_t>& buf) {
                std::vector<Packet> out;
                size_t off = 0;
                while (off + sizeof(GpuPacketHeader) <= buf.size()) {
                    GpuPacketHeader h{};
                    std::memcpy(&h, buf.data() + off, sizeof(h));
                    off += sizeof(h);
                    Packet p;
                    p.opcode = static_cast<GpuOpcode>(h.opcode);
                    p.payload.assign(buf.data() + off, buf.data() + off + h.payload_bytes);
                    off += h.payload_bytes;
                    out.push_back(std::move(p));
                }
                CHECK(off == buf.size());
                return out;
            };

            // Create a 256x1 RGBA32Float (DXGI format 2) address texture and mark it used in Frame 0.
            constexpr uint64_t kBaseVa = 0x7E00'1000'0000ull;
            const uint32_t lut_id = device->ensure_address_texture(kBaseVa, 2u, 256u, 1u, 1u, 1u, 0x100u,
                                                                   GpuTextureDimension::Dim2D, false);
            CHECK(lut_id != 0);
            device->state_bridge().note_resource_used(lut_id);

            // Fill the LRU with kMaxAddressTextures cold (not used in Frame 0) textures:
            // lut_id must NOT be evicted while cold non-in-frame candidates exist.
            std::vector<uint32_t> others;
            others.reserve(dg::Device::kMaxAddressTextures);
            for (size_t i = 1; i <= dg::Device::kMaxAddressTextures; ++i) {
                const uint32_t id = device->ensure_address_texture(kBaseVa + i * 0x10000ull, 2u, 256u, 1u, 1u, 1u,
                                                                   0x100u, GpuTextureDimension::Dim2D, false);
                CHECK(id != 0);
                others.push_back(id);
            }
            bridge::TextureInfo info{};
            CHECK(device->state_bridge().texture_info(lut_id, &info));

            // Now mark every live texture in Frame 0 with a newer tick than lut_id so the next
            // allocation MUST evict in-frame lut_id BEFORE Present!
            for (const uint32_t id : others) {
                if (device->state_bridge().texture_info(id, &info)) {
                    device->state_bridge().note_resource_used(id);
                }
            }
            const uint32_t overflow_id = device->ensure_address_texture(
                kBaseVa + (dg::Device::kMaxAddressTextures + 10u) * 0x10000ull, 2u, 256u, 1u, 1u, 1u, 0x100u,
                GpuTextureDimension::Dim2D, false);
            CHECK(overflow_id != 0);
            CHECK(!device->state_bridge().texture_info(lut_id, &info));  // lut_id was evicted

            // Before Present: ZERO DestroyResource(lut_id) packets may have been emitted!
            for (const Packet& pk : decode_stream(stream)) {
                if (pk.opcode == GpuOpcode::DestroyResource) {
                    CHECK(pod<GpuDestroyResourcePayload>(pk).id != lut_id);
                }
            }
            CHECK(device->state_bridge().deferred_destroy_count() >= 1);

            // Present Frame 0, then flush Frame 1 submission prelude: DestroyResource(lut_id) must
            // appear strictly AFTER Present!
            device->present();
            gpu::CommandEncoder frame1_enc;
            device->flush_cpu_buffers(frame1_enc);
            device->submit(frame1_enc);

            const std::vector<Packet> all = decode_stream(stream);
            size_t present_idx = all.size();
            size_t destroy_lut_idx = all.size();
            for (size_t i = 0; i < all.size(); ++i) {
                if (all[i].opcode == GpuOpcode::Present) {
                    present_idx = i;
                } else if (all[i].opcode == GpuOpcode::DestroyResource &&
                           pod<GpuDestroyResourcePayload>(all[i]).id == lut_id) {
                    destroy_lut_idx = i;
                }
            }
            CHECK(present_idx < all.size());
            CHECK(destroy_lut_idx < all.size() && destroy_lut_idx > present_idx);
        }
        com::reset_com_bridge();
        com::set_guest_memory(nullptr);
    }
}

// Native pipeline prewarm: a pipeline built from a recorded warm-list entry is the pipeline on-demand
// creation makes (same state key, modules, layout packets), the title then reuses it without emitting
// anything, and the pipeline is built exactly once whichever side gets there first.
struct PrewarmFixture {
    static uint64_t pack_key(uint32_t module) {
        return module == Scene::kVsModule ? 0x5101ull : module == Scene::kFsModule ? 0x5102ull : 0ull;
    }
    static void attach(Scene& scene, bool sync) {
        scene.device.set_module_pack_key_resolver(pack_key);
        scene.device.set_prewarm_sync(sync);
    }
    static std::vector<Packet> draw(Scene& scene) {
        gpu::CommandEncoder enc;
        bridge::CommandListBridge list(scene.device, enc);
        scene.bind_root(list);
        const d3d::CpuDescriptorHandle rtv0 = scene.rtv(0);
        list.OMSetRenderTargets(1, &rtv0, false, nullptr);
        list.DrawInstanced(3, 1, 0, 0);
        list.Close();
        return decode(enc);
    }
    static std::vector<Packet> only(const std::vector<Packet>& all, GpuOpcode a, GpuOpcode b, GpuOpcode c) {
        std::vector<Packet> out;
        for (const Packet& p : all) {
            if (p.opcode == a || p.opcode == b || p.opcode == c) out.push_back(p);
        }
        return out;
    }
    static size_t count(const std::vector<Packet>& all, GpuOpcode op) {
        return static_cast<size_t>(std::count_if(all.begin(), all.end(), [op](const Packet& p) { return p.opcode == op; }));
    }
};

void test_prewarm_pipeline_matches_on_demand() {
    using O = GpuOpcode;
    gpu::pipeline_prewarm().reset();
    std::vector<Packet> on_demand;
    {
        Scene scene;
        PrewarmFixture::attach(scene, false);
        on_demand = PrewarmFixture::draw(scene);
    }
    const std::vector<uint8_t> warm = gpu::pipeline_prewarm().save_warm_list();
    CHECK(warm.size() > gpu::kPrewarmListHeaderBytes && warm[4] == gpu::kPrewarmListVersionPipelines);
    const std::vector<Packet> creates =
        PrewarmFixture::only(on_demand, O::CreateBindGroupLayout, O::CreatePipelineLayout, O::CreateRenderPipeline2);
    CHECK(PrewarmFixture::count(on_demand, O::CreateRenderPipeline2) == 1);

    gpu::pipeline_prewarm().reset();
    CHECK(gpu::pipeline_prewarm().load_warm_list(warm.data(), warm.size(), nullptr));
    gpu::PrewarmTask task[2];
    CHECK(gpu::pipeline_prewarm().take(task, 2) == 1);
    const gpu::PrewarmDetail detail = gpu::pipeline_prewarm().detail_of(task[0].id);
    CHECK(detail != nullptr);
    gpu::PrewarmPipelineDesc desc;
    CHECK(gpu::decode_prewarm_pipeline_desc(detail->data(), detail->size(), &desc));
    CHECK(desc.kind == gpu::kPrewarmKindGraphics && desc.pack_key0 == 0x5101 && desc.pack_key1 == 0x5102);

    Scene scene;
    PrewarmFixture::attach(scene, true);
    std::vector<Packet> built;
    auto submit = [&](gpu::CommandEncoder& enc) { built = decode(enc); };
    CHECK(scene.device.prewarm_pipeline(desc, Scene::kVsModule, Scene::kFsModule, task[0].id, submit) ==
          bridge::PrewarmPipelineResult::Built);
    // Identical to what the title's own first draw creates (ids included: both devices allocate alike).
    const std::vector<Packet> pre =
        PrewarmFixture::only(built, O::CreateBindGroupLayout, O::CreatePipelineLayout, O::CreateRenderPipeline2);
    CHECK(built.size() == pre.size() && pre.size() == creates.size());
    for (size_t i = 0; i < pre.size() && i < creates.size(); ++i) {
        CHECK(pre[i].opcode == creates[i].opcode);
        if (pre[i].opcode == O::CreateRenderPipeline2) {
            // The pipeline id differs (the cache hands out ids in request order); everything else is equal.
            CHECK(pre[i].payload.size() == creates[i].payload.size());
            CHECK(std::memcmp(pre[i].payload.data() + 4, creates[i].payload.data() + 4, pre[i].payload.size() - 4) == 0);
        } else {
            CHECK(pre[i].payload == creates[i].payload);
        }
    }
    const uint32_t built_id = pre.empty() ? 0 : pod<uint32_t>(pre.back());
    CHECK(built_id != 0);
    CHECK(gpu::pipeline_prewarm().complete(task[0].id, true, 1));

    // The same pipeline again: lost to the earlier build, nothing emitted.
    built.clear();
    CHECK(scene.device.prewarm_pipeline(desc, Scene::kVsModule, Scene::kFsModule, task[0].id, submit) ==
          bridge::PrewarmPipelineResult::Lost);
    CHECK(built.empty());

    // The title's draw finds the pipeline and its layout ready: no creation packets, same pipeline id.
    const std::vector<Packet> list = PrewarmFixture::draw(scene);
    CHECK(PrewarmFixture::only(list, O::CreateBindGroupLayout, O::CreatePipelineLayout, O::CreateRenderPipeline2).empty());
    bool set = false;
    for (const Packet& p : list) {
        if (p.opcode == O::SetPipeline) {
            set = true;
            CHECK(pod<uint32_t>(p) == built_id);
        }
    }
    CHECK(set);
    const gpu::PrewarmUse use = gpu::pipeline_prewarm().use_stats();
    CHECK(use.pipelines_prewarmed == 1 && use.pipelines_hit == 1 && use.pipelines_wasted == 0);
    gpu::pipeline_prewarm().reset();
}

void test_prewarm_title_first_and_concurrent_single_build() {
    using O = GpuOpcode;
    gpu::pipeline_prewarm().reset();
    std::vector<Packet> first;
    {
        Scene scene;
        PrewarmFixture::attach(scene, false);
        first = PrewarmFixture::draw(scene);
    }
    const std::vector<uint8_t> warm = gpu::pipeline_prewarm().save_warm_list();
    gpu::PrewarmPipelineDesc desc;
    {
        gpu::PipelinePrewarm q;
        CHECK(q.load_warm_list(warm.data(), warm.size(), nullptr));
        gpu::PrewarmTask t[2];
        CHECK(q.take(t, 2) == 1);
        const gpu::PrewarmDetail d = q.detail_of(t[0].id);
        CHECK(d != nullptr && gpu::decode_prewarm_pipeline_desc(d->data(), d->size(), &desc));
    }
    gpu::pipeline_prewarm().reset();

    // The title creates the pipeline first: the prewarm finds it taken and emits nothing.
    {
        Scene scene;
        PrewarmFixture::attach(scene, true);
        const std::vector<Packet> list = PrewarmFixture::draw(scene);
        CHECK(PrewarmFixture::count(list, O::CreateRenderPipeline2) == 1);
        size_t submitted = 0;
        CHECK(scene.device.prewarm_pipeline(desc, Scene::kVsModule, Scene::kFsModule, 1,
                                            [&](gpu::CommandEncoder&) { ++submitted; }) ==
              bridge::PrewarmPipelineResult::Lost);
        CHECK(submitted == 0);
    }

    // Race: the prewarm thread and a recording title thread request the same pipeline. Exactly one creation
    // packet exists across both streams, and the title draws with that pipeline's id.
    size_t built_count = 0;
    for (int round = 0; round < 40; ++round) {
        Scene scene;
        PrewarmFixture::attach(scene, true);
        std::vector<Packet> pre;
        std::vector<Packet> list;
        bridge::PrewarmPipelineResult result = bridge::PrewarmPipelineResult::Failed;
        std::atomic<bool> go{false};
        std::thread prewarm([&] {
            while (!go.load()) std::this_thread::yield();
            result = scene.device.prewarm_pipeline(desc, Scene::kVsModule, Scene::kFsModule, 1,
                                                   [&](gpu::CommandEncoder& enc) { pre = decode(enc); });
        });
        std::thread title([&] {
            while (!go.load()) std::this_thread::yield();
            list = PrewarmFixture::draw(scene);
        });
        go = true;
        prewarm.join();
        title.join();
        const size_t creations =
            PrewarmFixture::count(pre, O::CreateRenderPipeline2) + PrewarmFixture::count(list, O::CreateRenderPipeline2);
        CHECK(creations == 1);
        if (result == bridge::PrewarmPipelineResult::Built) {
            ++built_count;
            CHECK(PrewarmFixture::count(list, O::CreateRenderPipeline2) == 0);
            uint32_t built_id = 0;
            for (const Packet& p : pre) {
                if (p.opcode == O::CreateRenderPipeline2) built_id = pod<uint32_t>(p);
            }
            for (const Packet& p : list) {
                if (p.opcode == O::SetPipeline) CHECK(pod<uint32_t>(p) == built_id);
            }
            // Layout packets exist exactly once as well.
            CHECK(PrewarmFixture::count(pre, O::CreatePipelineLayout) + PrewarmFixture::count(list, O::CreatePipelineLayout) == 1);
        } else {
            CHECK(result == bridge::PrewarmPipelineResult::Lost);
            CHECK(pre.empty());
        }
    }
    (void)built_count;  // either side may win a round
    gpu::pipeline_prewarm().reset();
}

// Design sections 3 and 5: an RT-class compute pipeline (layout with the arena group) creates the
// arena, directory and stats buffers on its first dispatch, uploads the resident pages and the
// directory, and binds group 3, in that order; RT off or an unsatisfiable range skips and counts.
struct RtRig {
    d3d::DescriptorHeapRegistry heaps;
    d3d::ResourceStateRegistry states;
    std::vector<uint64_t> reads;
    std::unique_ptr<bridge::StateBridgeDevice> device;
    d3d::RootSignature signature;
    std::unique_ptr<bridge::ComputePipelineState> pso;
    explicit RtRig(bool enabled, uint32_t first_id) {
        bridge::BridgeOptions o = counting_options(first_id);
        o.rt_enabled = enabled;
        o.rt_arenas = 2;
        o.rt_pages_per_arena = 4;  // 8 pages
        o.rt_page_reader = [this](uint64_t addr, void* dst, size_t n) {
            reads.push_back(addr);
            std::memset(dst, 0xAB, n);
            return true;
        };
        device = std::make_unique<bridge::StateBridgeDevice>(heaps, o, &states);
        constexpr uint64_t kCs = 0xC7;
        device->register_shader_module(kCs, 8);
        d3d::RootSignatureDesc rs;
        CHECK(d3d::RootSignature::build(rs, &signature).ok());
        bridge::PipelineLayoutDesc layout;
        layout.rt_arena = true;
        device->register_pipeline_layout(kCs, 0, layout);
        pso::D3D12ComputePsoDesc desc;
        desc.cs = kCs;
        desc.root_signature_id = 9;
        pso = device->create_compute_pipeline_state(desc, &signature);
    }
};

void test_rt_arena_dispatch() {
    using O = GpuOpcode;
    RtRig rig(true, 21000);
    CHECK(rig.pso != nullptr && rig.pso->diagnostics() == 0);
    gpu::CommandEncoder enc;
    bridge::CommandListBridge list(*rig.device, enc);
    CHECK(list.SetComputeRootSignature(&rig.signature) == d3d::Status::Ok);
    list.SetComputePipelineState(rig.pso.get());
    CHECK(rig.device->rt_report_json() == "{}");  // nothing before the first RT-class dispatch

    list.RtRequireRange(0x1000, 1);        // page 0
    list.RtRequireRange(0x20000, 0x20000);  // pages 2 and 3
    list.Dispatch(1, 1, 1);
    CHECK(list.stats().rt_dispatches == 1 && list.stats().rt_skipped == 0);
    {
        const std::vector<Packet> p = decode(enc);
        const size_t dispatch = find_op(p, O::DispatchCompute);
        CHECK(dispatch < p.size());
        // Four buffers (two arenas, directory, stats), then three page writes, one directory write, the
        // bind group, group 3 set, the dispatch.
        std::vector<uint32_t> created;
        size_t last_create = 0, first_write = p.size(), last_write = 0, pages = 0, dir_writes = 0;
        uint32_t dir_id = 0;
        for (size_t i = 0; i < dispatch; ++i) {
            if (p[i].opcode == O::CreateBuffer) {
                created.push_back(pod<GpuCreateBufferPayload>(p[i]).id);
                last_create = i;
                if (pod<GpuCreateBufferPayload>(p[i]).byte_size == (4096u + 56u * 4096u) * 4u) {
                    dir_id = pod<GpuCreateBufferPayload>(p[i]).id;
                }
            } else if (p[i].opcode == O::WriteBuffer) {
                first_write = std::min(first_write, i);
                last_write = i;
                const auto w = pod<GpuWriteBufferPayload>(p[i]);
                if (w.data_len == 65536) {
                    ++pages;
                    CHECK(w.id == created[0] || w.id == created[1]);
                } else {
                    ++dir_writes;
                    CHECK(w.id == dir_id && dir_id != 0);
                }
            }
        }
        CHECK(created.size() == 4 && pages == 3 && dir_writes == 1);
        CHECK(last_create < first_write);
        const size_t group = find_op(p, O::CreateBindGroup2, last_write);
        CHECK(group < dispatch);
        const auto g = pod<GpuCreateBindGroup2Payload>(p[group]);
        CHECK(g.entry_count == 6);
        for (uint32_t i = 0; i < 6; ++i) {
            const auto e = pod<GpuBindGroupEntry2>(p[group], sizeof(GpuCreateBindGroup2Payload) + i * sizeof(GpuBindGroupEntry2));
            // Arenas 2 and 3 do not exist with two arenas: null bindings; then directory and stats.
            const uint32_t want = i < 2 ? created[i] : i < 4 ? 0u : created[i - 2];
            CHECK(e.binding == 16 + i && e.resource_id == want);
        }
        const auto set = pod<GpuSetBindGroup2Payload>(p[dispatch - 1]);
        CHECK(p[dispatch - 1].opcode == O::SetBindGroup2 && set.group_index == 3 && set.bind_group_id == g.id);
        CHECK(group < dispatch - 1);
        // The pages were read from guest memory at their page addresses.
        CHECK((rig.reads == std::vector<uint64_t>{0x0, 0x20000, 0x30000}));
    }
    // A second dispatch over resident pages uploads nothing and only rebinds group 3.
    const size_t mark = enc.size();
    list.RtRequireRange(0x1000, 1);
    list.Dispatch(1, 1, 1);
    {
        const std::vector<Packet> p = decode(enc, mark);
        CHECK(find_op(p, O::CreateBuffer) == p.size() && find_op(p, O::WriteBuffer) == p.size() &&
              find_op(p, O::CreateBindGroup2) == p.size());
        const size_t dispatch = find_op(p, O::DispatchCompute);
        CHECK(dispatch < p.size() && p[dispatch - 1].opcode == O::SetBindGroup2 &&
              pod<GpuSetBindGroup2Payload>(p[dispatch - 1]).group_index == 3);
    }
    // 9 pages in one dispatch cannot fit 8 slots (the dispatch's own pages are not evictable): skipped.
    const size_t mark2 = enc.size();
    list.RtRequireRange(0x100000, 9 * 65536);
    list.Dispatch(1, 1, 1);
    CHECK(find_op(decode(enc, mark2), O::DispatchCompute) == decode(enc, mark2).size());
    CHECK(list.stats().rt_dispatches == 2 && list.stats().rt_skipped == 1);
    const std::string json = rig.device->rt_report_json();
    CHECK(json.find("\"dispatches\": 2") != std::string::npos && json.find("\"arena_full\": 1") != std::string::npos &&
          json.find("\"pages_resident\": ") != std::string::npos);
    CHECK(playgta6::gpu::rt_arena::report_json() == json);  // what the boot report prints
}

void test_rt_off_skips_and_counts() {
    using O = GpuOpcode;
    RtRig rig(false, 22000);
    gpu::CommandEncoder enc;
    bridge::CommandListBridge list(*rig.device, enc);
    CHECK(list.SetComputeRootSignature(&rig.signature) == d3d::Status::Ok);
    list.SetComputePipelineState(rig.pso.get());
    list.RtRequireRange(0x1000, 1);
    list.Dispatch(1, 1, 1);
    list.Dispatch(1, 1, 1);
    const std::vector<Packet> p = decode(enc);
    CHECK(find_op(p, O::DispatchCompute) == p.size());  // a skipped dispatch is not recorded
    CHECK(find_op(p, O::CreateBuffer) == p.size() && find_op(p, O::WriteBuffer) == p.size());  // no buffers made
    CHECK(list.stats().rt_skipped == 2 && list.stats().rt_dispatches == 0 && list.stats().dispatches == 0);
    CHECK(rig.device->rt_report_json().find("\"rt_off\": 2") != std::string::npos);
    CHECK(rig.reads.empty());
}

// C2 wiring (rt_residency): plan from the stage's address_sources -> upload -> bind -> dispatch -> fault
// readback; the logged pages are resident (uploaded) by the next dispatch; unwalkable structures skip.
struct RtSrcRig {
    d3d::DescriptorHeapRegistry heaps;
    d3d::ResourceStateRegistry states;
    std::vector<uint64_t> reads;
    uint64_t unreadable_addr = ~0ull;
    std::unique_ptr<bridge::StateBridgeDevice> device;
    d3d::RootSignature signature;
    std::unique_ptr<bridge::ComputePipelineState> pso;
    RtSrcRig(uint32_t first_id, bool with_bvh_root) {
        bridge::BridgeOptions o = counting_options(first_id);
        o.rt_enabled = true;
        o.rt_arenas = 2;
        o.rt_pages_per_arena = 4;
        o.rt_page_reader = [this](uint64_t addr, void* dst, size_t n) {
            if (addr == unreadable_addr) return false;
            reads.push_back(addr);
            std::memset(dst, 0xAB, n);
            return true;
        };
        device = std::make_unique<bridge::StateBridgeDevice>(heaps, o, &states);
        constexpr uint64_t kCs = 0xC8;
        device->register_shader_module(kCs, 8);
        d3d::RootSignatureDesc rs;
        for (int i = 0; i < 2; ++i) {
            d3d::RootParameterDesc p;
            p.parameter_type = d3d::RootParameterType::Cbv;
            p.shader_register = static_cast<uint32_t>(i);
            rs.parameters.push_back(p);
        }
        CHECK(d3d::RootSignature::build(rs, &signature).ok());
        bridge::PipelineLayoutDesc layout;
        layout.rt_arena = true;
        layout.rt_address_sources.push_back({gpu::rt_residency::SourceKind::RootParam, 0, 0, 0x20000, false});
        if (with_bvh_root) {
            layout.rt_address_sources.push_back({gpu::rt_residency::SourceKind::RootParam, 1, 0, 1, true});
        }
        device->register_pipeline_layout(kCs, 0, layout);
        pso::D3D12ComputePsoDesc desc;
        desc.cs = kCs;
        desc.root_signature_id = 9;
        pso = device->create_compute_pipeline_state(desc, &signature);
    }
};

void test_rt_residency_wiring() {
    using O = GpuOpcode;
    RtSrcRig rig(23000, false);
    gpu::CommandEncoder enc;
    bridge::CommandListBridge list(*rig.device, enc);
    CHECK(list.SetComputeRootSignature(&rig.signature) == d3d::Status::Ok);
    list.SetComputePipelineState(rig.pso.get());
    CHECK(list.SetComputeRootConstantBufferView(0, 0x10000) == d3d::Status::Ok);  // pages 1 and 2
    CHECK(list.SetComputeRootConstantBufferView(1, 0) == d3d::Status::Ok);
    list.Dispatch(1, 1, 1);
    CHECK(list.stats().rt_dispatches == 1 && list.stats().rt_skipped == 0);
    GpuReadbackBufferPayload rb{};
    {
        const std::vector<Packet> p = decode(enc);
        const size_t dispatch = find_op(p, O::DispatchCompute);
        const size_t last_write = [&] {
            size_t w = 0;
            for (size_t i = 0; i < dispatch; ++i) if (p[i].opcode == O::WriteBuffer) w = i;
            return w;
        }();
        const size_t group = find_op(p, O::CreateBindGroup2, last_write);
        const size_t readback = find_op(p, O::ReadbackBuffer);
        // plan -> upload (pages from the address source's extent) -> bind -> dispatch -> fault readback
        CHECK((rig.reads == std::vector<uint64_t>{0x10000, 0x20000}));
        CHECK(last_write < group && group < dispatch && dispatch < readback && readback + 1 == p.size());
        rb = pod<GpuReadbackBufferPayload>(p[readback]);
        CHECK(rb.size == 128 && rb.dst_addr != 0 && rb.done_addr != 0 && rb.seq == 1);
    }
    // A second dispatch while the readback is in flight issues no second one and does not block.
    list.Dispatch(1, 1, 1);
    CHECK(find_op(decode(enc), O::ReadbackBuffer, find_op(decode(enc), O::ReadbackBuffer) + 1) == decode(enc).size());
    // The consumer lands the stats: 2 faults, pages 7 and 6 (page + 1 encoded), then completes.
    auto* words = reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(rb.dst_addr));
    std::memset(words, 0, 128);
    words[0] = 2;
    words[1] = 2;
    words[2] = 8;
    words[3] = 7;
    reinterpret_cast<std::atomic<uint32_t>*>(static_cast<uintptr_t>(rb.done_addr))->fetch_add(1);
    rig.reads.clear();
    const size_t mark = enc.size();
    list.Dispatch(1, 1, 1);
    {
        const std::vector<Packet> p = decode(enc, mark);
        const size_t dispatch = find_op(p, O::DispatchCompute);
        CHECK(dispatch < p.size());
        // The faulted pages were made resident before this dispatch (uploaded), the fault words 1..31
        // were reset on the GPU, the arena group was bound, and a fresh readback follows.
        CHECK((rig.reads == std::vector<uint64_t>{0x70000, 0x60000}));
        bool zeroed = false;
        size_t uploads = 0;
        for (size_t i = 0; i < dispatch; ++i) {
            if (p[i].opcode != O::WriteBuffer) continue;
            const auto w = pod<GpuWriteBufferPayload>(p[i]);
            zeroed |= w.data_len == 124 && w.dst_offset == 4;
            uploads += w.data_len == 65536;
        }
        CHECK(zeroed && uploads == 2);
        CHECK(p[dispatch - 1].opcode == O::SetBindGroup2);
        CHECK(find_op(p, O::ReadbackBuffer) > dispatch);
    }
    const std::string json = rig.device->rt_report_json();
    CHECK(json.find("\"rt_fault\": 2") != std::string::npos && json.find("\"pages_serviced\": 2") != std::string::npos);
    CHECK(json.find("\"residency\": {\"plans\": 3") != std::string::npos);
    CHECK(json.find("\"fault_readbacks_issued\": 2") != std::string::npos);
}

void test_rt_residency_structures_unavailable_skips() {
    using O = GpuOpcode;
    RtSrcRig rig(24000, true);
    rig.unreadable_addr = 0x500000;  // the BVH header cannot be read: GPU-written or unmapped
    gpu::CommandEncoder enc;
    bridge::CommandListBridge list(*rig.device, enc);
    CHECK(list.SetComputeRootSignature(&rig.signature) == d3d::Status::Ok);
    list.SetComputePipelineState(rig.pso.get());
    CHECK(list.SetComputeRootConstantBufferView(0, 0x10000) == d3d::Status::Ok);
    CHECK(list.SetComputeRootConstantBufferView(1, 0x500000) == d3d::Status::Ok);
    list.Dispatch(1, 1, 1);
    const std::vector<Packet> p = decode(enc);
    CHECK(find_op(p, O::DispatchCompute) == p.size() && find_op(p, O::ReadbackBuffer) == p.size());
    CHECK(list.stats().rt_skipped == 1 && list.stats().rt_dispatches == 0);
    CHECK(rig.device->rt_report_json().find("\"structures_unavailable\": 1") != std::string::npos);
}

// =================================================================================================
// G81: Full 3D pipeline StateBridge tests
// =================================================================================================

void test_g81_depth_bias_and_shadow_map_parity() {
    // 1. Direct float / bits sanitization (NaN, +/-Inf, -0.0f -> +0.0f; finite preserved).
    CHECK(bridge::sanitize_depth_bias_float(std::numeric_limits<float>::quiet_NaN()) == 0.0f);
    CHECK(bridge::sanitize_depth_bias_float(std::numeric_limits<float>::infinity()) == 0.0f);
    CHECK(bridge::sanitize_depth_bias_float(-std::numeric_limits<float>::infinity()) == 0.0f);
    CHECK(bridge::sanitize_depth_bias_bits(-0.0f) == 0u);
    CHECK(bridge::sanitize_depth_bias_bits(std::numeric_limits<float>::infinity()) == 0u);
    CHECK(bridge::sanitize_depth_bias_bits(-std::numeric_limits<float>::infinity()) == 0u);
    CHECK(bridge::sanitize_depth_bias_bits(std::numeric_limits<float>::quiet_NaN()) == 0u);
    {
        uint32_t expected_bits = 0;
        const float slope = -1.75f;
        std::memcpy(&expected_bits, &slope, sizeof(expected_bits));
        CHECK(bridge::sanitize_depth_bias_bits(slope) == expected_bits);
    }

    // 2. Shadow-map depth-only render pass (NumRenderTargets = 0, ps = 0, D32_FLOAT) with
    //    DepthBias / SlopeScaledDepthBias / DepthBiasClamp forwarded into CreateRenderPipeline2.
    d3d::DescriptorHeapRegistry heaps;
    bridge::StateBridgeDevice device(heaps, counting_options(30000));
    d3d::DescriptorHeap* rtv_heap = make_heap(heaps, d3d::DescriptorHeapType::Rtv, 2, false);
    d3d::DescriptorHeap* dsv_heap = make_heap(heaps, d3d::DescriptorHeapType::Dsv, 2, false);

    constexpr uint32_t kShadowTex = 301;
    constexpr uint32_t kColorTex = 302;
    constexpr uint32_t kVb = 303;
    constexpr uint64_t kVs = 0x5A01;
    constexpr uint64_t kPs = 0x5A02;
    device.register_texture(kShadowTex, {pso::DxgiFormat::D32_FLOAT, 1024, 1024, 1, 1, 1,
                                         GpuTextureDimension::Dim2D, false});
    device.register_texture(kColorTex, {pso::DxgiFormat::R8G8B8A8_UNORM, 1024, 1024, 1, 1, 1,
                                        GpuTextureDimension::Dim2D, false});
    device.register_shader_module(kVs, 11);
    device.register_shader_module(kPs, 12);

    d3d::ViewDesc dsv_desc;
    dsv_desc.format = static_cast<uint32_t>(pso::DxgiFormat::D32_FLOAT);
    dsv_desc.dimension = d3d::DsvDimension::Texture2D;
    CHECK(heaps.create_depth_stencil_view(kShadowTex, &dsv_desc, dsv_heap->cpu_handle(0)) == d3d::Status::Ok);
    CHECK(heaps.create_render_target_view(kColorTex, nullptr, rtv_heap->cpu_handle(0)) == d3d::Status::Ok);

    d3d::RootSignature rs;
    CHECK(d3d::RootSignature::build(d3d::RootSignatureDesc{}, &rs).ok());

    auto make_shadow_desc = [&](int32_t bias, float slope, float clamp, bool with_ps) {
        pso::D3D12GraphicsPsoDesc d{};
        d.root_signature_id = 1;
        d.vs = kVs;
        d.ps = with_ps ? kPs : 0;
        d.NumRenderTargets = with_ps ? 1 : 0;
        if (with_ps) {
            d.RTVFormats[0] = pso::DxgiFormat::R8G8B8A8_UNORM;
        }
        d.DSVFormat = pso::DxgiFormat::D32_FLOAT;
        d.DepthStencilState.DepthEnable = true;
        d.DepthStencilState.DepthWriteMask = pso::D3D12DepthWriteMask::All;
        d.DepthStencilState.DepthFunc = pso::D3D12ComparisonFunc::GreaterEqual;  // reverse-Z shadow map
        d.RasterizerState.DepthBias = bias;
        d.RasterizerState.SlopeScaledDepthBias = slope;
        d.RasterizerState.DepthBiasClamp = clamp;
        return d;
    };

    auto shadow_pso = device.create_graphics_pipeline_state(make_shadow_desc(-4, -2.5f, 0.02f, false), &rs);
    auto inf_pso = device.create_graphics_pipeline_state(
        make_shadow_desc(3, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), true),
        &rs);
    auto nan_pso = device.create_graphics_pipeline_state(
        make_shadow_desc(3, std::numeric_limits<float>::quiet_NaN(), -0.0f, true), &rs);

    gpu::CommandEncoder enc;
    bridge::CommandListBridge list(device, enc);
    CHECK(list.SetGraphicsRootSignature(&rs) == d3d::Status::Ok);
    const gpu::d3d12_xs::D3D12VertexBufferView vb{kVb, 0, 48, 16};
    list.IASetVertexBuffers(0, 1, &vb);
    list.IASetPrimitiveTopology(pso::D3DPrimitiveTopology::TriangleList);

    const d3d::CpuDescriptorHandle dsv = dsv_heap->cpu_handle(0);
    const d3d::CpuDescriptorHandle rtv = rtv_heap->cpu_handle(0);
    list.OMSetRenderTargets(0, nullptr, false, &dsv);
    list.SetPipelineState(shadow_pso.get());
    list.DrawInstanced(3, 1, 0, 0);

    // Non-triangle topology (LineList) with depth bias -> WebGPU requires zeroed depth bias.
    const size_t line_mark = enc.size();
    list.IASetPrimitiveTopology(pso::D3DPrimitiveTopology::LineList);
    list.DrawInstanced(2, 1, 0, 0);

    // Inf and NaN slope/clamp sanitize to +0.0f and deduplicate to the SAME pipeline ID!
    const size_t inf_mark = enc.size();
    list.IASetPrimitiveTopology(pso::D3DPrimitiveTopology::TriangleList);
    list.OMSetRenderTargets(1, &rtv, false, &dsv);
    list.SetPipelineState(inf_pso.get());
    list.DrawInstanced(3, 1, 0, 0);
    const size_t nan_mark = enc.size();
    list.SetPipelineState(nan_pso.get());
    list.DrawInstanced(3, 1, 0, 0);

    // No depth attachment bound -> depth_stencil (including depth_bias) completely zeroed.
    const size_t nodepth_mark = enc.size();
    list.OMSetRenderTargets(1, &rtv, false, nullptr);
    list.SetPipelineState(inf_pso.get());
    list.DrawInstanced(3, 1, 0, 0);
    list.Close();

    auto find_rp2 = [](const std::vector<Packet>& pkts, GpuCreateRenderPipeline2Payload* out) {
        for (const Packet& pk : pkts) {
            if (pk.opcode == GpuOpcode::CreateRenderPipeline2) {
                *out = pod<GpuCreateRenderPipeline2Payload>(pk);
                return true;
            }
        }
        return false;
    };

    GpuCreateRenderPipeline2Payload rp{};
    CHECK(find_rp2(decode(enc, 0), &rp));
    CHECK(rp.fs_module_id == 0 && rp.key.color_target_count == 0);
    CHECK(rp.key.depth_stencil.format == static_cast<uint8_t>(pso::WgpuTextureFormat::Depth32Float));
    CHECK(rp.key.depth_stencil.depth_bias == -4);
    CHECK(rp.key.depth_stencil.depth_bias_slope_scale_bits == bridge::sanitize_depth_bias_bits(-2.5f));
    CHECK(rp.key.depth_stencil.depth_bias_clamp_bits == bridge::sanitize_depth_bias_bits(0.02f));

    // LineList: depth bias zeroed for non-triangle topology.
    CHECK(find_rp2(decode(enc, line_mark), &rp));
    CHECK(rp.key.depth_stencil.depth_bias == 0);
    CHECK(rp.key.depth_stencil.depth_bias_slope_scale_bits == 0u);
    CHECK(rp.key.depth_stencil.depth_bias_clamp_bits == 0u);

    // Inf PSO: depth_bias=3 preserved, non-finite slope & clamp sanitized to 0.
    const std::vector<Packet> inf_pkts = decode(enc, inf_mark);
    // Slice to nan_mark by decoding full and filtering.
    GpuCreateRenderPipeline2Payload inf_rp{};
    CHECK(find_rp2(inf_pkts, &inf_rp));
    CHECK(inf_rp.key.depth_stencil.depth_bias == 3);
    CHECK(inf_rp.key.depth_stencil.depth_bias_slope_scale_bits == 0u);
    CHECK(inf_rp.key.depth_stencil.depth_bias_clamp_bits == 0u);

    // NaN PSO: deduplicated with Inf PSO (no CreateRenderPipeline2 between nan_mark and nodepth_mark).
    const std::vector<Packet> nan_pkts = decode(enc, nan_mark);
    // The only CreateRenderPipeline2 after nan_mark must be the depth-less variant (at nodepth_mark).
    GpuCreateRenderPipeline2Payload nodepth_rp{};
    CHECK(find_rp2(decode(enc, nodepth_mark), &nodepth_rp));
    CHECK(nodepth_rp.key.depth_stencil.format == 0);
    CHECK(nodepth_rp.key.depth_stencil.depth_bias == 0);
    CHECK(nodepth_rp.key.depth_stencil.depth_bias_slope_scale_bits == 0u);
    CHECK(nodepth_rp.key.depth_stencil.depth_bias_clamp_bits == 0u);
    // Count total CreateRenderPipeline2 in [nan_mark, nodepth_mark): must be 0.
    {
        gpu::CommandEncoder sub;
        // Verify by checking total pipelines created == 4 (shadow tri, shadow line, inf/nan shared, depth-less).
        CHECK(list.stats().pipelines_created == 4);
    }

    // 3. Direct sanitize_render_pipeline_key catches raw infinity & missing-depth bias on prewarmed keys.
    GpuRenderPipelineKey dirty{};
    dirty.primitive.topology = static_cast<uint8_t>(GpuPrimitiveTopology::TriangleList);
    dirty.depth_stencil.format = static_cast<uint8_t>(pso::WgpuTextureFormat::Depth32Float);
    dirty.depth_stencil.depth_write_enabled = 1;
    dirty.depth_stencil.depth_compare = static_cast<uint8_t>(pso::WgpuCompareFunction::Less);
    dirty.depth_stencil.depth_bias = 5;
    const float pinf = std::numeric_limits<float>::infinity();
    std::memcpy(&dirty.depth_stencil.depth_bias_slope_scale_bits, &pinf, 4);
    dirty.depth_stencil.depth_bias_clamp_bits = 0x80000000u;  // -0.0f
    CHECK(bridge::sanitize_render_pipeline_key(&dirty, 0));
    CHECK(dirty.depth_stencil.depth_bias == 5);
    CHECK(dirty.depth_stencil.depth_bias_slope_scale_bits == 0u);
    CHECK(dirty.depth_stencil.depth_bias_clamp_bits == 0u);
    CHECK(!bridge::sanitize_render_pipeline_key(&dirty, 0));  // idempotent
}

void test_g81_eight_mrt_blend_and_integer_guards() {
    d3d::DescriptorHeapRegistry heaps;
    bridge::BridgeOptions opts = counting_options(31000);
    opts.features = pso::WgpuFeature::Rg11b10UfloatRenderable;  // Float32Blendable & DualSourceBlending OFF
    bridge::StateBridgeDevice device(heaps, opts);
    d3d::DescriptorHeap* rtv_heap = make_heap(heaps, d3d::DescriptorHeapType::Rtv, 8, false);
    d3d::DescriptorHeap* dsv_heap = make_heap(heaps, d3d::DescriptorHeapType::Dsv, 1, false);

    // 8 distinct MRTs covering HDR float, sRGB, UNORM, R32_FLOAT (non-blendable w/o feature), and integer formats.
    const std::array<pso::DxgiFormat, 8> mrt_formats = {
        pso::DxgiFormat::R16G16B16A16_FLOAT,  // RT0: HDR float16 (blendable)
        pso::DxgiFormat::R11G11B10_FLOAT,     // RT1: HDR rg11b10 (blendable w/ feature)
        pso::DxgiFormat::R10G10B10A2_UNORM,   // RT2: 10-bit HDR/UI (blendable)
        pso::DxgiFormat::R8G8B8A8_UNORM_SRGB, // RT3: sRGB (blendable)
        pso::DxgiFormat::R32_UINT,            // RT4: integer uint (NOT blendable!)
        pso::DxgiFormat::R16G16B16A16_SINT,   // RT5: integer sint (NOT blendable!)
        pso::DxgiFormat::R32_FLOAT,           // RT6: float32 (NOT blendable without Float32Blendable!)
        pso::DxgiFormat::R8G8_UNORM,          // RT7: RG8 (blendable)
    };
    for (uint32_t i = 0; i < 8; ++i) {
        device.register_texture(400 + i, {mrt_formats[i], 256, 256, 1, 1, 1, GpuTextureDimension::Dim2D, false});
        d3d::ViewDesc vd;
        vd.format = static_cast<uint32_t>(mrt_formats[i]);
        vd.dimension = d3d::ViewDimension::Texture2D;
        CHECK(heaps.create_render_target_view(400 + i, &vd, rtv_heap->cpu_handle(i)) == d3d::Status::Ok);
    }
    device.register_texture(410, {pso::DxgiFormat::D32_FLOAT, 256, 256, 1, 1, 1, GpuTextureDimension::Dim2D, false});
    d3d::ViewDesc dsv_desc;
    dsv_desc.format = static_cast<uint32_t>(pso::DxgiFormat::D32_FLOAT);
    dsv_desc.dimension = d3d::DsvDimension::Texture2D;
    CHECK(heaps.create_depth_stencil_view(410, &dsv_desc, dsv_heap->cpu_handle(0)) == d3d::Status::Ok);

    constexpr uint64_t kVs = 0x8B01;
    constexpr uint64_t kPs = 0x8B02;
    device.register_shader_module(kVs, 21);
    device.register_shader_module(kPs, 22);
    d3d::RootSignature rs;
    CHECK(d3d::RootSignature::build(d3d::RootSignatureDesc{}, &rs).ok());

    auto base_mrt_desc = [&] {
        pso::D3D12GraphicsPsoDesc d{};
        d.root_signature_id = 1;
        d.vs = kVs;
        d.ps = kPs;
        d.NumRenderTargets = 8;
        for (uint32_t i = 0; i < 8; ++i) {
            d.RTVFormats[i] = mrt_formats[i];
        }
        d.DSVFormat = pso::DxgiFormat::D32_FLOAT;
        return d;
    };

    // Case A: IndependentBlendEnable = false -> RT0 replicated to all blendable targets,
    // integer & non-blendable targets (RT4, RT5, RT6) have blend stripped cleanly!
    pso::D3D12GraphicsPsoDesc desc_shared = base_mrt_desc();
    desc_shared.BlendState.IndependentBlendEnable = false;
    desc_shared.BlendState.RenderTarget[0].BlendEnable = true;
    desc_shared.BlendState.RenderTarget[0].SrcBlend = pso::D3D12Blend::SrcAlpha;
    desc_shared.BlendState.RenderTarget[0].DestBlend = pso::D3D12Blend::InvSrcAlpha;
    desc_shared.BlendState.RenderTarget[0].BlendOp = pso::D3D12BlendOp::Add;
    desc_shared.BlendState.RenderTarget[0].SrcBlendAlpha = pso::D3D12Blend::One;
    desc_shared.BlendState.RenderTarget[0].DestBlendAlpha = pso::D3D12Blend::Zero;
    desc_shared.BlendState.RenderTarget[0].BlendOpAlpha = pso::D3D12BlendOp::Add;
    desc_shared.BlendState.RenderTarget[0].RenderTargetWriteMask = 0x7;  // RGB only
    // Poison RT1..RT7 with distinct values that must be ignored when IndependentBlendEnable == false.
    desc_shared.BlendState.RenderTarget[1].BlendEnable = false;
    desc_shared.BlendState.RenderTarget[1].RenderTargetWriteMask = 0xF;
    auto pso_shared = device.create_graphics_pipeline_state(desc_shared, &rs);

    // Case B: IndependentBlendEnable = true -> per-target blend factors, ops, constant factor, write masks.
    pso::D3D12GraphicsPsoDesc desc_indep = base_mrt_desc();
    desc_indep.BlendState.IndependentBlendEnable = true;
    for (uint32_t i = 0; i < 8; ++i) {
        desc_indep.BlendState.RenderTarget[i].RenderTargetWriteMask = static_cast<uint8_t>((i + 1) & 0xF);
    }
    // RT0: constant blend factor
    desc_indep.BlendState.RenderTarget[0].BlendEnable = true;
    desc_indep.BlendState.RenderTarget[0].SrcBlend = pso::D3D12Blend::BlendFactor;
    desc_indep.BlendState.RenderTarget[0].DestBlend = pso::D3D12Blend::InvBlendFactor;
    desc_indep.BlendState.RenderTarget[0].BlendOp = pso::D3D12BlendOp::Add;
    // RT1: Min blend op (WebGPU requires src/dst factors = One)
    desc_indep.BlendState.RenderTarget[1].BlendEnable = true;
    desc_indep.BlendState.RenderTarget[1].SrcBlend = pso::D3D12Blend::SrcAlpha;
    desc_indep.BlendState.RenderTarget[1].DestBlend = pso::D3D12Blend::DestAlpha;
    desc_indep.BlendState.RenderTarget[1].BlendOp = pso::D3D12BlendOp::Min;
    // RT2: blend disabled, custom write mask
    desc_indep.BlendState.RenderTarget[2].BlendEnable = false;
    // RT3: ReverseSubtract
    desc_indep.BlendState.RenderTarget[3].BlendEnable = true;
    desc_indep.BlendState.RenderTarget[3].SrcBlend = pso::D3D12Blend::One;
    desc_indep.BlendState.RenderTarget[3].DestBlend = pso::D3D12Blend::One;
    desc_indep.BlendState.RenderTarget[3].BlendOp = pso::D3D12BlendOp::RevSubtract;
    // RT4/RT5/RT6: blend requested on integer / non-blendable float32 -> must be stripped!
    desc_indep.BlendState.RenderTarget[4].BlendEnable = true;
    desc_indep.BlendState.RenderTarget[4].SrcBlend = pso::D3D12Blend::One;
    desc_indep.BlendState.RenderTarget[4].DestBlend = pso::D3D12Blend::One;
    desc_indep.BlendState.RenderTarget[5].BlendEnable = true;
    desc_indep.BlendState.RenderTarget[5].SrcBlend = pso::D3D12Blend::SrcAlpha;
    desc_indep.BlendState.RenderTarget[5].DestBlend = pso::D3D12Blend::InvSrcAlpha;
    desc_indep.BlendState.RenderTarget[6].BlendEnable = true;
    desc_indep.BlendState.RenderTarget[6].SrcBlend = pso::D3D12Blend::One;
    desc_indep.BlendState.RenderTarget[6].DestBlend = pso::D3D12Blend::One;
    // RT7: additive blend on R8G8_UNORM
    desc_indep.BlendState.RenderTarget[7].BlendEnable = true;
    desc_indep.BlendState.RenderTarget[7].SrcBlend = pso::D3D12Blend::One;
    desc_indep.BlendState.RenderTarget[7].DestBlend = pso::D3D12Blend::One;
    auto pso_indep = device.create_graphics_pipeline_state(desc_indep, &rs);

    gpu::CommandEncoder enc;
    bridge::CommandListBridge list(device, enc);
    CHECK(list.SetGraphicsRootSignature(&rs) == d3d::Status::Ok);
    list.IASetPrimitiveTopology(pso::D3DPrimitiveTopology::TriangleList);

    // Bind all 8 MRTs using rts_single_handle_to_descriptor_range = true (tests the G81 contiguous-range fix too!).
    const d3d::CpuDescriptorHandle rtv0 = rtv_heap->cpu_handle(0);
    const d3d::CpuDescriptorHandle dsv0 = dsv_heap->cpu_handle(0);
    list.OMSetRenderTargets(8, &rtv0, true, &dsv0);
    list.SetPipelineState(pso_shared.get());
    list.DrawInstanced(3, 1, 0, 0);

    const size_t indep_mark = enc.size();
    list.SetPipelineState(pso_indep.get());
    list.DrawInstanced(3, 1, 0, 0);
    list.Close();

    CHECK(list.stats().draws == 2 && list.stats().draws_skipped == 0);
    CHECK((list.pipeline_diagnostics() & pso::PipelineDiag::BlendOnNonBlendableFormat) != 0);

    const std::vector<Packet> p0 = decode(enc, 0);
    GpuCreateRenderPipeline2Payload rp_shared{};
    bool found_shared = false;
    for (const Packet& pk : p0) {
        if (pk.opcode == GpuOpcode::CreateRenderPipeline2) {
            rp_shared = pod<GpuCreateRenderPipeline2Payload>(pk);
            found_shared = true;
            break;
        }
    }
    CHECK(found_shared && rp_shared.key.color_target_count == 8);
    // Blendable targets (0, 1, 2, 3, 7) inherit RT0's blend & write_mask 0x7; non-blendable (4, 5, 6) drop blend.
    for (uint32_t i = 0; i < 8; ++i) {
        const auto& ct = rp_shared.key.color_targets[i];
        CHECK(ct.write_mask == 0x7);
        const bool expected_blend = (i != 4 && i != 5 && i != 6);
        CHECK(ct.blend_enabled == (expected_blend ? 1u : 0u));
        if (expected_blend) {
            CHECK(ct.color.src_factor == static_cast<uint8_t>(pso::WgpuBlendFactor::SrcAlpha));
            CHECK(ct.color.dst_factor == static_cast<uint8_t>(pso::WgpuBlendFactor::OneMinusSrcAlpha));
        }
    }

    const std::vector<Packet> p1 = decode(enc, indep_mark);
    GpuCreateRenderPipeline2Payload rp_indep{};
    bool found_indep = false;
    for (const Packet& pk : p1) {
        if (pk.opcode == GpuOpcode::CreateRenderPipeline2) {
            rp_indep = pod<GpuCreateRenderPipeline2Payload>(pk);
            found_indep = true;
            break;
        }
    }
    CHECK(found_indep && rp_indep.key.color_target_count == 8);
    CHECK(rp_indep.key.color_targets[0].blend_enabled == 1 &&
          rp_indep.key.color_targets[0].color.src_factor == static_cast<uint8_t>(pso::WgpuBlendFactor::Constant) &&
          rp_indep.key.color_targets[0].color.dst_factor ==
              static_cast<uint8_t>(pso::WgpuBlendFactor::OneMinusConstant) &&
          rp_indep.key.color_targets[0].write_mask == 1);
    CHECK(rp_indep.key.color_targets[1].blend_enabled == 1 &&
          rp_indep.key.color_targets[1].color.operation == static_cast<uint8_t>(pso::WgpuBlendOperation::Min) &&
          rp_indep.key.color_targets[1].color.src_factor == static_cast<uint8_t>(pso::WgpuBlendFactor::One) &&
          rp_indep.key.color_targets[1].color.dst_factor == static_cast<uint8_t>(pso::WgpuBlendFactor::One) &&
          rp_indep.key.color_targets[1].write_mask == 2);
    CHECK(rp_indep.key.color_targets[2].blend_enabled == 0 && rp_indep.key.color_targets[2].write_mask == 3);
    CHECK(rp_indep.key.color_targets[3].blend_enabled == 1 &&
          rp_indep.key.color_targets[3].color.operation ==
              static_cast<uint8_t>(pso::WgpuBlendOperation::ReverseSubtract) &&
          rp_indep.key.color_targets[3].write_mask == 4);
    CHECK(rp_indep.key.color_targets[4].blend_enabled == 0 && rp_indep.key.color_targets[4].write_mask == 5);
    CHECK(rp_indep.key.color_targets[5].blend_enabled == 0 && rp_indep.key.color_targets[5].write_mask == 6);
    CHECK(rp_indep.key.color_targets[6].blend_enabled == 0 && rp_indep.key.color_targets[6].write_mask == 7);
    CHECK(rp_indep.key.color_targets[7].blend_enabled == 1 && rp_indep.key.color_targets[7].write_mask == 8);

    // Case C: With DualSourceBlending + Float32Blendable enabled:
    //  - R32_FLOAT becomes blendable;
    //  - Dual-source works on 1 RT and is rejected (DualSourceMultipleTargets) on 2 RTs.
    {
        d3d::DescriptorHeapRegistry h2;
        bridge::BridgeOptions opts2 = counting_options(32000);
        opts2.features = pso::WgpuFeature::Float32Blendable | pso::WgpuFeature::DualSourceBlending;
        bridge::StateBridgeDevice dev2(h2, opts2);
        d3d::DescriptorHeap* rtv2 = make_heap(h2, d3d::DescriptorHeapType::Rtv, 2, false);
        dev2.register_texture(420, {pso::DxgiFormat::R32_FLOAT, 64, 64, 1, 1, 1, GpuTextureDimension::Dim2D, false});
        dev2.register_texture(421, {pso::DxgiFormat::R8G8B8A8_UNORM, 64, 64, 1, 1, 1, GpuTextureDimension::Dim2D, false});
        CHECK(h2.create_render_target_view(420, nullptr, rtv2->cpu_handle(0)) == d3d::Status::Ok);
        CHECK(h2.create_render_target_view(421, nullptr, rtv2->cpu_handle(1)) == d3d::Status::Ok);
        dev2.register_shader_module(kVs, 31);
        dev2.register_shader_module(kPs, 32);

        pso::D3D12GraphicsPsoDesc ds_desc{};
        ds_desc.vs = kVs;
        ds_desc.ps = kPs;
        ds_desc.NumRenderTargets = 1;
        ds_desc.RTVFormats[0] = pso::DxgiFormat::R32_FLOAT;
        ds_desc.BlendState.RenderTarget[0].BlendEnable = true;
        ds_desc.BlendState.RenderTarget[0].SrcBlend = pso::D3D12Blend::Src1Color;
        ds_desc.BlendState.RenderTarget[0].DestBlend = pso::D3D12Blend::InvSrc1Alpha;
        auto ds_pso = dev2.create_graphics_pipeline_state(ds_desc, &rs);

        gpu::CommandEncoder enc2;
        bridge::CommandListBridge list2(dev2, enc2);
        list2.IASetPrimitiveTopology(pso::D3DPrimitiveTopology::TriangleList);
        const d3d::CpuDescriptorHandle rt_one = rtv2->cpu_handle(0);
        list2.OMSetRenderTargets(1, &rt_one, false, nullptr);
        list2.SetPipelineState(ds_pso.get());
        list2.DrawInstanced(3, 1, 0, 0);
        CHECK(list2.stats().draws == 1 && list2.stats().draws_skipped == 0);
        GpuCreateRenderPipeline2Payload ds_rp{};
        for (const Packet& pk : decode(enc2)) {
            if (pk.opcode == GpuOpcode::CreateRenderPipeline2) {
                ds_rp = pod<GpuCreateRenderPipeline2Payload>(pk);
                break;
            }
        }
        CHECK(ds_rp.key.color_targets[0].format == static_cast<uint8_t>(pso::WgpuTextureFormat::R32Float));
        CHECK(ds_rp.key.color_targets[0].blend_enabled == 1);
        CHECK(ds_rp.key.color_targets[0].color.src_factor == static_cast<uint8_t>(pso::WgpuBlendFactor::Src1));
        CHECK(ds_rp.key.color_targets[0].color.dst_factor ==
              static_cast<uint8_t>(pso::WgpuBlendFactor::OneMinusSrc1Alpha));

        // Binding 2 targets with dual-source blending must skip the draw (WebGPU fatal rule).
        const d3d::CpuDescriptorHandle rt_two[2] = {rtv2->cpu_handle(0), rtv2->cpu_handle(1)};
        list2.OMSetRenderTargets(2, rt_two, false, nullptr);
        list2.DrawInstanced(3, 1, 0, 0);
        CHECK(list2.stats().draws == 1 && list2.stats().draws_skipped == 1);
        CHECK((list2.pipeline_diagnostics() & pso::PipelineDiag::DualSourceMultipleTargets) != 0);
        list2.Close();
    }
}

void test_g81_stencil_faces_masks_and_dynamic_reemission() {
    d3d::DescriptorHeapRegistry heaps;
    bridge::BridgeOptions opts = counting_options(33000);
    opts.features = pso::WgpuFeature::Depth32FloatStencil8;
    bridge::StateBridgeDevice device(heaps, opts);
    d3d::DescriptorHeap* rtv_heap = make_heap(heaps, d3d::DescriptorHeapType::Rtv, 2, false);
    d3d::DescriptorHeap* dsv_heap = make_heap(heaps, d3d::DescriptorHeapType::Dsv, 3, false);

    device.register_texture(500, {pso::DxgiFormat::R8G8B8A8_UNORM, 256, 256, 1, 1, 1, GpuTextureDimension::Dim2D, false});
    device.register_texture(501, {pso::DxgiFormat::R8G8B8A8_UNORM, 256, 256, 1, 1, 1, GpuTextureDimension::Dim2D, false});
    device.register_texture(502, {pso::DxgiFormat::D24_UNORM_S8_UINT, 256, 256, 1, 1, 1, GpuTextureDimension::Dim2D, false});
    device.register_texture(503, {pso::DxgiFormat::D32_FLOAT_S8X24_UINT, 256, 256, 1, 1, 1, GpuTextureDimension::Dim2D, false});
    device.register_texture(504, {pso::DxgiFormat::D32_FLOAT, 256, 256, 1, 1, 1, GpuTextureDimension::Dim2D, false});
    CHECK(heaps.create_render_target_view(500, nullptr, rtv_heap->cpu_handle(0)) == d3d::Status::Ok);
    CHECK(heaps.create_render_target_view(501, nullptr, rtv_heap->cpu_handle(1)) == d3d::Status::Ok);

    d3d::ViewDesc dsv24;
    dsv24.format = static_cast<uint32_t>(pso::DxgiFormat::D24_UNORM_S8_UINT);
    dsv24.dimension = d3d::DsvDimension::Texture2D;
    dsv24.flags = d3d::DsvFlags::ReadOnlyDepth;  // depth read-only, stencil WRITABLE!
    CHECK(heaps.create_depth_stencil_view(502, &dsv24, dsv_heap->cpu_handle(0)) == d3d::Status::Ok);

    d3d::ViewDesc dsv32s8;
    dsv32s8.format = static_cast<uint32_t>(pso::DxgiFormat::D32_FLOAT_S8X24_UINT);
    dsv32s8.dimension = d3d::DsvDimension::Texture2D;
    CHECK(heaps.create_depth_stencil_view(503, &dsv32s8, dsv_heap->cpu_handle(1)) == d3d::Status::Ok);

    d3d::ViewDesc dsv32;
    dsv32.format = static_cast<uint32_t>(pso::DxgiFormat::D32_FLOAT);
    dsv32.dimension = d3d::DsvDimension::Texture2D;
    dsv32.flags = d3d::DsvFlags::ReadOnlyDepth | d3d::DsvFlags::ReadOnlyStencil;  // depth-only must ignore stencil_ro!
    CHECK(heaps.create_depth_stencil_view(504, &dsv32, dsv_heap->cpu_handle(2)) == d3d::Status::Ok);

    constexpr uint64_t kVs = 0x5701;
    constexpr uint64_t kPs = 0x5702;
    device.register_shader_module(kVs, 41);
    device.register_shader_module(kPs, 42);
    d3d::RootSignature rs;
    CHECK(d3d::RootSignature::build(d3d::RootSignatureDesc{}, &rs).ok());

    pso::D3D12GraphicsPsoDesc desc{};
    desc.vs = kVs;
    desc.ps = kPs;
    desc.NumRenderTargets = 1;
    desc.RTVFormats[0] = pso::DxgiFormat::R8G8B8A8_UNORM;
    desc.DSVFormat = pso::DxgiFormat::D24_UNORM_S8_UINT;
    desc.DepthStencilState.DepthEnable = true;
    desc.DepthStencilState.DepthWriteMask = pso::D3D12DepthWriteMask::Zero;
    desc.DepthStencilState.DepthFunc = pso::D3D12ComparisonFunc::LessEqual;
    desc.DepthStencilState.StencilEnable = true;
    desc.DepthStencilState.StencilReadMask = 0x3F;
    desc.DepthStencilState.StencilWriteMask = 0xC0;
    desc.DepthStencilState.FrontFace = {pso::D3D12StencilOp::Replace, pso::D3D12StencilOp::IncrSat,
                                        pso::D3D12StencilOp::Incr, pso::D3D12ComparisonFunc::Equal};
    desc.DepthStencilState.BackFace = {pso::D3D12StencilOp::Zero, pso::D3D12StencilOp::DecrSat,
                                       pso::D3D12StencilOp::Invert, pso::D3D12ComparisonFunc::NotEqual};
    auto stencil_pso = device.create_graphics_pipeline_state(desc, &rs);

    // Check resolve_dsv usage: ReadOnlyDepth on D24S8 with writable stencil must remain DepthStencilWrite!
    pso::DepthStencilViewInfo dsv_info{};
    CHECK(device.resolve_dsv(dsv_heap->cpu_handle(0).ptr, &dsv_info));
    const bridge::ViewRecord* vr = device.view(dsv_info.view_id);
    CHECK(vr != nullptr && vr->usage == d3d::ResourceUsage::DepthStencilWrite);

    gpu::CommandEncoder enc;
    bridge::CommandListBridge list(device, enc);
    list.IASetPrimitiveTopology(pso::D3DPrimitiveTopology::TriangleList);
    list.SetPipelineState(stencil_pso.get());

    const d3d::CpuDescriptorHandle rtv0 = rtv_heap->cpu_handle(0);
    const d3d::CpuDescriptorHandle rtv1 = rtv_heap->cpu_handle(1);
    const d3d::CpuDescriptorHandle dsv0 = dsv_heap->cpu_handle(0);
    const d3d::CpuDescriptorHandle dsv1 = dsv_heap->cpu_handle(1);
    const d3d::CpuDescriptorHandle dsv2 = dsv_heap->cpu_handle(2);

    // Pass 1: D24S8 (depth_read_only=1, stencil_read_only=0), dynamic stencil ref & blend constant (with NaN sanitized).
    list.OMSetRenderTargets(1, &rtv0, false, &dsv0);
    list.OMSetStencilRef(0xAB);
    const float bf_dirty[4] = {0.25f, std::numeric_limits<float>::quiet_NaN(), 0.75f,
                               std::numeric_limits<float>::infinity()};
    list.OMSetBlendFactor(bf_dirty);
    list.DrawInstanced(3, 1, 0, 0);

    // Second draw in Pass 1 with identical dynamic state: no redundant SetStencilReference / SetBlendConstant.
    const size_t same_pass_mark = enc.size();
    list.DrawInstanced(3, 1, 0, 0);

    // Pass 2: switch to rtv1 + D32S8 -> dynamic stencil ref & blend constant MUST re-emit across pass boundary!
    const size_t pass2_mark = enc.size();
    list.OMSetRenderTargets(1, &rtv1, false, &dsv1);
    list.DrawInstanced(3, 1, 0, 0);

    // Pass 3: ResourceBarrier breaks pass + OMSetBlendFactor(nullptr) sets {1,1,1,1}, D32_FLOAT clears stencil state.
    const size_t pass3_mark = enc.size();
    list.ResourceBarrier(0, nullptr);
    list.OMSetBlendFactor(nullptr);
    list.OMSetStencilRef(0x55);
    list.OMSetRenderTargets(1, &rtv0, false, &dsv2);
    list.DrawInstanced(3, 1, 0, 0);
    list.Close();

    // Verify Pass 1 packets.
    const std::vector<Packet> p1 = decode(enc, 0);
    CHECK(p1[0].opcode == GpuOpcode::BeginRenderPass);
    const auto bp1 = pod<GpuBeginRenderPassPayload>(p1[0]);
    CHECK(bp1.depth_stencil.depth_read_only == 1 && bp1.depth_stencil.stencil_read_only == 0);
    CHECK(bp1.depth_stencil.depth_load_op == 0 && bp1.depth_stencil.stencil_load_op == static_cast<uint8_t>(GpuLoadOp::Load));
    GpuCreateRenderPipeline2Payload rp1{};
    GpuSetBlendConstantPayload bc1{};
    GpuSetStencilReferencePayload sr1{};
    bool got_rp1 = false, got_bc1 = false, got_sr1 = false;
    for (const Packet& pk : p1) {
        if (!got_rp1 && pk.opcode == GpuOpcode::CreateRenderPipeline2) {
            rp1 = pod<GpuCreateRenderPipeline2Payload>(pk);
            got_rp1 = true;
        } else if (!got_bc1 && pk.opcode == GpuOpcode::SetBlendConstant) {
            bc1 = pod<GpuSetBlendConstantPayload>(pk);
            got_bc1 = true;
        } else if (!got_sr1 && pk.opcode == GpuOpcode::SetStencilReference) {
            sr1 = pod<GpuSetStencilReferencePayload>(pk);
            got_sr1 = true;
        }
    }
    CHECK(got_rp1 && got_bc1 && got_sr1);
    CHECK(rp1.key.depth_stencil.format == static_cast<uint8_t>(pso::WgpuTextureFormat::Depth24PlusStencil8));
    CHECK(rp1.key.depth_stencil.stencil_read_mask == 0x3Fu && rp1.key.depth_stencil.stencil_write_mask == 0xC0u);
    CHECK(rp1.key.depth_stencil.stencil_front.compare == static_cast<uint8_t>(pso::WgpuCompareFunction::Equal) &&
          rp1.key.depth_stencil.stencil_front.fail_op == static_cast<uint8_t>(pso::WgpuStencilOperation::Replace) &&
          rp1.key.depth_stencil.stencil_front.depth_fail_op ==
              static_cast<uint8_t>(pso::WgpuStencilOperation::IncrementClamp) &&
          rp1.key.depth_stencil.stencil_front.pass_op ==
              static_cast<uint8_t>(pso::WgpuStencilOperation::IncrementWrap));
    CHECK(rp1.key.depth_stencil.stencil_back.compare == static_cast<uint8_t>(pso::WgpuCompareFunction::NotEqual) &&
          rp1.key.depth_stencil.stencil_back.fail_op == static_cast<uint8_t>(pso::WgpuStencilOperation::Zero) &&
          rp1.key.depth_stencil.stencil_back.depth_fail_op ==
              static_cast<uint8_t>(pso::WgpuStencilOperation::DecrementClamp) &&
          rp1.key.depth_stencil.stencil_back.pass_op == static_cast<uint8_t>(pso::WgpuStencilOperation::Invert));
    CHECK(bc1.r == 0.25f && bc1.g == 0.0f && bc1.b == 0.75f && bc1.a == 0.0f);
    CHECK(sr1.reference == 0xABu);

    // Same-pass draw emitted ONLY Draw (check first packet at same_pass_mark).
    const std::vector<Packet> same_pkts = decode(enc, same_pass_mark);
    CHECK(!same_pkts.empty() && same_pkts[0].opcode == GpuOpcode::Draw);

    // Pass 2 re-emits SetBlendConstant and SetStencilReference with the exact same values!
    const std::vector<Packet> p2 = decode(enc, pass2_mark);
    bool got_bc2 = false, got_sr2 = false, got_rp2 = false;
    GpuCreateRenderPipeline2Payload rp2{};
    for (const Packet& pk : p2) {
        if (pk.opcode == GpuOpcode::EndRenderPass && got_rp2) {
            break;
        }
        if (pk.opcode == GpuOpcode::CreateRenderPipeline2 && !got_rp2) {
            rp2 = pod<GpuCreateRenderPipeline2Payload>(pk);
            got_rp2 = true;
        } else if (pk.opcode == GpuOpcode::SetBlendConstant && !got_bc2) {
            const auto bc = pod<GpuSetBlendConstantPayload>(pk);
            CHECK(bc.r == 0.25f && bc.g == 0.0f && bc.b == 0.75f && bc.a == 0.0f);
            got_bc2 = true;
        } else if (pk.opcode == GpuOpcode::SetStencilReference && !got_sr2) {
            CHECK(pod<GpuSetStencilReferencePayload>(pk).reference == 0xABu);
            got_sr2 = true;
        }
    }
    CHECK(got_rp2 && got_bc2 && got_sr2);
    CHECK(rp2.key.depth_stencil.format == static_cast<uint8_t>(pso::WgpuTextureFormat::Depth32FloatStencil8));
    CHECK(rp2.key.depth_stencil.stencil_read_mask == 0x3Fu && rp2.key.depth_stencil.stencil_write_mask == 0xC0u);

    // Pass 3: D32_FLOAT (no stencil aspect) -> stencil_read_only forced 0, stencil ops/masks cleared,
    // OMSetBlendFactor(nullptr) emitted {1, 1, 1, 1}, stencil ref 0x55 re-emitted.
    const std::vector<Packet> p3 = decode(enc, pass3_mark);
    bool got_bp3 = false, got_rp3 = false, got_bc3 = false, got_sr3 = false;
    for (const Packet& pk : p3) {
        if (pk.opcode == GpuOpcode::BeginRenderPass) {
            const auto bp3 = pod<GpuBeginRenderPassPayload>(pk);
            CHECK(bp3.depth_stencil.depth_read_only == 1 && bp3.depth_stencil.stencil_read_only == 0);
            got_bp3 = true;
        } else if (pk.opcode == GpuOpcode::CreateRenderPipeline2) {
            const auto rp3 = pod<GpuCreateRenderPipeline2Payload>(pk);
            CHECK(rp3.key.depth_stencil.format == static_cast<uint8_t>(pso::WgpuTextureFormat::Depth32Float));
            CHECK(rp3.key.depth_stencil.stencil_read_mask == 0u && rp3.key.depth_stencil.stencil_write_mask == 0u);
            CHECK(rp3.key.depth_stencil.stencil_front.compare ==
                  static_cast<uint8_t>(pso::WgpuCompareFunction::Always));
            got_rp3 = true;
        } else if (pk.opcode == GpuOpcode::SetBlendConstant) {
            const auto bc = pod<GpuSetBlendConstantPayload>(pk);
            CHECK(bc.r == 1.0f && bc.g == 1.0f && bc.b == 1.0f && bc.a == 1.0f);
            got_bc3 = true;
        } else if (pk.opcode == GpuOpcode::SetStencilReference) {
            CHECK(pod<GpuSetStencilReferencePayload>(pk).reference == 0x55u);
            got_sr3 = true;
        }
    }
    CHECK(got_bp3 && got_rp3 && got_bc3 && got_sr3);
}

void test_g81_primitive_topology_and_strip_index_format() {
    // 1. Direct seam conversion & stripIndexFormat helpers.
    bool exact = false;
    CHECK(bridge::to_gpu_primitive_topology(pso::D3DPrimitiveTopology::PointList, &exact) ==
              GpuPrimitiveTopology::PointList &&
          exact);
    CHECK(bridge::to_gpu_primitive_topology(pso::D3DPrimitiveTopology::LineList, &exact) ==
              GpuPrimitiveTopology::LineList &&
          exact);
    CHECK(bridge::to_gpu_primitive_topology(pso::D3DPrimitiveTopology::LineStrip, &exact) ==
              GpuPrimitiveTopology::LineStrip &&
          exact);
    CHECK(bridge::to_gpu_primitive_topology(pso::D3DPrimitiveTopology::TriangleList, &exact) ==
              GpuPrimitiveTopology::TriangleList &&
          exact);
    CHECK(bridge::to_gpu_primitive_topology(pso::D3DPrimitiveTopology::TriangleStrip, &exact) ==
              GpuPrimitiveTopology::TriangleStrip &&
          exact);
    CHECK(bridge::to_gpu_primitive_topology(pso::D3DPrimitiveTopology::TriangleFan, &exact) ==
              GpuPrimitiveTopology::TriangleList &&
          !exact);
    CHECK(bridge::to_gpu_primitive_topology(pso::D3DPrimitiveTopology::TriangleStripAdj, &exact) ==
              GpuPrimitiveTopology::TriangleStrip &&
          !exact);

    CHECK(!bridge::is_strip_topology(GpuPrimitiveTopology::PointList));
    CHECK(!bridge::is_strip_topology(GpuPrimitiveTopology::LineList));
    CHECK(bridge::is_strip_topology(GpuPrimitiveTopology::LineStrip));
    CHECK(!bridge::is_strip_topology(GpuPrimitiveTopology::TriangleList));
    CHECK(bridge::is_strip_topology(GpuPrimitiveTopology::TriangleStrip));

    CHECK(bridge::resolve_strip_index_format(pso::WgpuPrimitiveTopology::TriangleList, pso::WgpuIndexFormat::Uint32,
                                             pso::D3D12IndexBufferStripCutValue::Value0xFFFFFFFF) ==
          pso::WgpuIndexFormat::Undefined);
    CHECK(bridge::resolve_strip_index_format(pso::WgpuPrimitiveTopology::PointList, pso::WgpuIndexFormat::Uint16) ==
          pso::WgpuIndexFormat::Undefined);
    CHECK(bridge::resolve_strip_index_format(pso::WgpuPrimitiveTopology::TriangleStrip, pso::WgpuIndexFormat::Uint16) ==
          pso::WgpuIndexFormat::Uint16);
    CHECK(bridge::resolve_strip_index_format(pso::WgpuPrimitiveTopology::LineStrip, pso::WgpuIndexFormat::Uint32) ==
          pso::WgpuIndexFormat::Uint32);
    CHECK(bridge::resolve_strip_index_format(pso::WgpuPrimitiveTopology::TriangleStrip, pso::WgpuIndexFormat::Undefined,
                                             pso::D3D12IndexBufferStripCutValue::Value0xFFFF) ==
          pso::WgpuIndexFormat::Uint16);
    CHECK(bridge::resolve_strip_index_format(pso::WgpuPrimitiveTopology::TriangleStrip, pso::WgpuIndexFormat::Undefined,
                                             pso::D3D12IndexBufferStripCutValue::Disabled) ==
          pso::WgpuIndexFormat::Undefined);

    // 2. End-to-end draws: re-keying on strip index format changes, Undefined on list topologies,
    //    and clearing stale index format on IASetIndexBuffer(nullptr).
    Scene scene(34000);
    gpu::CommandEncoder enc;
    bridge::CommandListBridge list(scene.device, enc);
    scene.bind_root(list);
    const d3d::CpuDescriptorHandle rtv0 = scene.rtv(0);
    list.OMSetRenderTargets(1, &rtv0, false, nullptr);

    const gpu::d3d12_xs::D3D12IndexBufferView ib16{601, 0, 128, GpuIndexFormat::Uint16};
    const gpu::d3d12_xs::D3D12IndexBufferView ib32{602, 0, 256, GpuIndexFormat::Uint32};

    // Draw 1: TriangleStrip + Uint16 -> strip_index_format == Uint16.
    list.IASetPrimitiveTopology(pso::D3DPrimitiveTopology::TriangleStrip);
    list.IASetIndexBuffer(&ib16);
    list.DrawIndexedInstanced(6, 1, 0, 0, 0);

    // Draw 2: TriangleStrip + Uint32 -> new pipeline with strip_index_format == Uint32.
    const size_t m_u32 = enc.size();
    list.IASetIndexBuffer(&ib32);
    list.DrawIndexedInstanced(6, 1, 0, 0, 0);

    // Draw 3: IASetIndexBuffer(nullptr) clears stale Uint32 -> non-indexed TriangleStrip uses Undefined (0)!
    const size_t m_null_ib = enc.size();
    list.IASetIndexBuffer(nullptr);
    list.DrawInstanced(4, 1, 0, 0);

    // Draw 4: TriangleList + Uint32 -> strip_index_format == Undefined (0).
    const size_t m_list = enc.size();
    list.IASetPrimitiveTopology(pso::D3DPrimitiveTopology::TriangleList);
    list.IASetIndexBuffer(&ib32);
    list.DrawIndexedInstanced(6, 1, 0, 0, 0);

    // Draw 5: TriangleList + Uint16 -> reuses the SAME list pipeline (no new CreateRenderPipeline2)!
    const size_t m_list_u16 = enc.size();
    list.IASetIndexBuffer(&ib16);
    list.DrawIndexedInstanced(6, 1, 0, 0, 0);

    // Draw 6: LineStrip + Uint16 -> strip_index_format == Uint16.
    const size_t m_line_strip = enc.size();
    list.IASetPrimitiveTopology(pso::D3DPrimitiveTopology::LineStrip);
    list.DrawIndexedInstanced(4, 1, 0, 0, 0);
    list.Close();

    auto first_rp = [](const std::vector<Packet>& pkts, GpuCreateRenderPipeline2Payload* out) {
        for (const Packet& pk : pkts) {
            if (pk.opcode == GpuOpcode::CreateRenderPipeline2) {
                *out = pod<GpuCreateRenderPipeline2Payload>(pk);
                return true;
            }
        }
        return false;
    };

    GpuCreateRenderPipeline2Payload rp{};
    CHECK(first_rp(decode(enc, 0), &rp));
    CHECK(rp.key.primitive.topology == static_cast<uint8_t>(GpuPrimitiveTopology::TriangleStrip));
    CHECK(rp.key.primitive.strip_index_format == static_cast<uint8_t>(pso::WgpuIndexFormat::Uint16));

    CHECK(first_rp(decode(enc, m_u32), &rp));
    CHECK(rp.key.primitive.topology == static_cast<uint8_t>(GpuPrimitiveTopology::TriangleStrip));
    CHECK(rp.key.primitive.strip_index_format == static_cast<uint8_t>(pso::WgpuIndexFormat::Uint32));

    CHECK(first_rp(decode(enc, m_null_ib), &rp));
    CHECK(rp.key.primitive.topology == static_cast<uint8_t>(GpuPrimitiveTopology::TriangleStrip));
    CHECK(rp.key.primitive.strip_index_format == static_cast<uint8_t>(pso::WgpuIndexFormat::Undefined));

    CHECK(first_rp(decode(enc, m_list), &rp));
    CHECK(rp.key.primitive.topology == static_cast<uint8_t>(GpuPrimitiveTopology::TriangleList));
    CHECK(rp.key.primitive.strip_index_format == static_cast<uint8_t>(pso::WgpuIndexFormat::Undefined));

    // Draw 5 reused Draw 4's pipeline -> first CreateRenderPipeline2 after m_list_u16 is Draw 6 (LineStrip).
    CHECK(first_rp(decode(enc, m_list_u16), &rp));
    CHECK(rp.key.primitive.topology == static_cast<uint8_t>(GpuPrimitiveTopology::LineStrip));
    CHECK(first_rp(decode(enc, m_line_strip), &rp));
    CHECK(rp.key.primitive.topology == static_cast<uint8_t>(GpuPrimitiveTopology::LineStrip));
    CHECK(rp.key.primitive.strip_index_format == static_cast<uint8_t>(pso::WgpuIndexFormat::Uint16));
    CHECK(list.stats().pipelines_created == 5);
}

void test_g81_hdr_depth_srgb_view_formats() {
    using D = pso::DxgiFormat;
    using F = pso::WgpuTextureFormat;
    using A = pso::WgpuTextureAspect;

    // 1. All 6 sRGB <-> linear sibling viewFormats pairs (both directions).
    const std::pair<F, F> srgb_pairs[] = {
        {F::Rgba8Unorm, F::Rgba8UnormSrgb},       {F::Bgra8Unorm, F::Bgra8UnormSrgb},
        {F::Bc1RgbaUnorm, F::Bc1RgbaUnormSrgb},   {F::Bc2RgbaUnorm, F::Bc2RgbaUnormSrgb},
        {F::Bc3RgbaUnorm, F::Bc3RgbaUnormSrgb},   {F::Bc7RgbaUnorm, F::Bc7RgbaUnormSrgb},
    };
    for (const auto& [lin, srgb] : srgb_pairs) {
        const auto c_lin = bridge::compatible_view_formats(lin);
        const auto c_srgb = bridge::compatible_view_formats(srgb);
        CHECK(c_lin.count == 1 && c_lin.formats[0] == srgb);
        CHECK(c_srgb.count == 1 && c_srgb.formats[0] == lin);
        CHECK(bridge::is_compatible_view_format(lin, srgb));
        CHECK(bridge::is_compatible_view_format(srgb, lin));
        CHECK(bridge::is_compatible_view_format(lin, lin));
        CHECK(bridge::is_compatible_view_format(lin, F::Undefined));
        CHECK(!bridge::is_srgb_format(lin) && bridge::is_srgb_format(srgb));
    }
    // Cross-family or integer reinterpretation is NOT a valid WebGPU viewFormat pair.
    CHECK(!bridge::is_compatible_view_format(F::Rgba8Unorm, F::Bgra8Unorm));
    CHECK(!bridge::is_compatible_view_format(F::Rgba8Unorm, F::Rgba8Uint));
    CHECK(!bridge::is_compatible_view_format(F::Rgba16Float, F::Rgba16Uint));
    CHECK(bridge::compatible_view_formats(F::Rgba16Float).count == 0);

    // DXGI viewFormats helper (with and without BC compression feature).
    const auto bc7_with = bridge::compatible_view_formats_for_dxgi(
        D::BC7_UNORM, pso::FormatUsage::Sampled, pso::WgpuFeature::TextureCompressionBc);
    CHECK(bc7_with.count == 1 && bc7_with.formats[0] == F::Bc7RgbaUnormSrgb);
    const auto bc7_without = bridge::compatible_view_formats_for_dxgi(D::BC7_UNORM, pso::FormatUsage::Sampled, 0);
    CHECK(bc7_without.count == 1 && bc7_without.formats[0] == F::Rgba8UnormSrgb);

    // 2. HDR format classification and mapping.
    CHECK(bridge::is_hdr_color_format(D::R16G16B16A16_FLOAT));
    CHECK(bridge::is_hdr_color_format(D::R11G11B10_FLOAT));
    CHECK(bridge::is_hdr_color_format(D::R10G10B10A2_UNORM));
    CHECK(bridge::is_hdr_color_format(D::R32G32B32A32_FLOAT));
    CHECK(bridge::is_hdr_color_format(D::R32_FLOAT));
    CHECK(bridge::is_hdr_color_format(D::R16_FLOAT));
    CHECK(bridge::is_hdr_color_format(D::R16G16_FLOAT));
    CHECK(!bridge::is_hdr_color_format(D::R8G8_UNORM));
    CHECK(!bridge::is_hdr_color_format(D::R8G8B8A8_UNORM));

    // 3. Depth & typeless-depth sibling SRV format resolution.
    CHECK(bridge::is_depth_stencil_dxgi_format(D::D32_FLOAT));
    CHECK(bridge::is_depth_stencil_dxgi_format(D::D24_UNORM_S8_UINT));
    CHECK(bridge::is_depth_stencil_dxgi_format(D::D32_FLOAT_S8X24_UINT));
    CHECK(bridge::is_depth_stencil_dxgi_format(D::D16_UNORM));
    CHECK(bridge::is_depth_stencil_dxgi_format(D::R24G8_TYPELESS));
    CHECK(bridge::is_depth_stencil_dxgi_format(D::R32G8X24_TYPELESS));

    const auto d32_srv = bridge::resolve_sampled_view_format(D::D32_FLOAT, D::R32_FLOAT, false, 0);
    CHECK(d32_srv.format == F::Depth32Float && d32_srv.aspect == A::DepthOnly && d32_srv.depth_sibling);

    const auto r32_typeless_depth =
        bridge::resolve_sampled_view_format(D::R32_TYPELESS, D::R32_FLOAT, false, 0, /*is_depth_resource=*/true);
    CHECK(r32_typeless_depth.format == F::Depth32Float && r32_typeless_depth.aspect == A::DepthOnly &&
          r32_typeless_depth.depth_sibling);
    const auto r32_typeless_color =
        bridge::resolve_sampled_view_format(D::R32_TYPELESS, D::R32_FLOAT, false, 0, /*is_depth_resource=*/false);
    CHECK(r32_typeless_color.format == F::R32Float && r32_typeless_color.aspect == A::All &&
          !r32_typeless_color.depth_sibling);

    const auto d16_srv =
        bridge::resolve_sampled_view_format(D::R16_TYPELESS, D::R16_UNORM, false, 0, /*is_depth_resource=*/true);
    CHECK(d16_srv.format == F::Depth16Unorm && d16_srv.aspect == A::DepthOnly && d16_srv.depth_sibling);

    const auto d24_depth_srv =
        bridge::resolve_sampled_view_format(D::R24G8_TYPELESS, D::R24_UNORM_X8_TYPELESS, false, 0);
    CHECK(d24_depth_srv.format == F::Depth24PlusStencil8 && d24_depth_srv.aspect == A::DepthOnly);
    const auto d24_stencil_srv =
        bridge::resolve_sampled_view_format(D::R24G8_TYPELESS, D::X24_TYPELESS_G8_UINT, false, 0);
    CHECK(d24_stencil_srv.format == F::Depth24PlusStencil8 && d24_stencil_srv.aspect == A::StencilOnly);

    const auto d32s8_depth = bridge::resolve_sampled_view_format(
        D::R32G8X24_TYPELESS, D::R32_FLOAT_X8X24_TYPELESS, false, pso::WgpuFeature::Depth32FloatStencil8);
    CHECK(d32s8_depth.format == F::Depth32FloatStencil8 && d32s8_depth.aspect == A::DepthOnly);
    const auto d32s8_stencil_fb =
        bridge::resolve_sampled_view_format(D::R32G8X24_TYPELESS, D::X32_TYPELESS_G8X24_UINT, false, 0);
    CHECK(d32s8_stencil_fb.format == F::Depth24PlusStencil8 && d32s8_stencil_fb.aspect == A::StencilOnly);

    // Valid sRGB view alias vs invalid integer alias fallback.
    const auto srgb_alias = bridge::resolve_sampled_view_format(D::R8G8B8A8_UNORM, D::R8G8B8A8_UNORM_SRGB, false, 0);
    CHECK(srgb_alias.format == F::Rgba8UnormSrgb && srgb_alias.srgb_aliased && !srgb_alias.fallback_to_base);
    const auto bad_alias = bridge::resolve_sampled_view_format(D::R8G8B8A8_UNORM, D::R8G8B8A8_UINT, false, 0);
    CHECK(bad_alias.format == F::Rgba8Unorm && !bad_alias.srgb_aliased && bad_alias.fallback_to_base);

    // 4. End-to-end CommandListBridge test: R32_TYPELESS depth texture written as D32_FLOAT DSV,
    //    then sampled via R32_FLOAT SRV -> CreateBindGroup2 carries Depth32Float + DepthOnly!
    //    And R8G8B8A8_UNORM sampled via R8G8B8A8_UNORM_SRGB -> Rgba8UnormSrgb; invalid UINT SRV -> Rgba8Unorm!
    Scene scene(35000);
    constexpr uint32_t kTypelessShadow = 701;
    scene.device.register_texture(kTypelessShadow, {D::R32_TYPELESS, 512, 512, 1, 1, 1, GpuTextureDimension::Dim2D, false});
    d3d::ViewDesc shadow_dsv;
    shadow_dsv.format = static_cast<uint32_t>(D::D32_FLOAT);
    shadow_dsv.dimension = d3d::DsvDimension::Texture2D;
    CHECK(scene.heaps.create_depth_stencil_view(kTypelessShadow, &shadow_dsv, scene.dsv_heap->cpu_handle(1)) ==
          d3d::Status::Ok);

    // Resolve the DSV (marks kTypelessShadow as a depth resource), then bind an R32_FLOAT SRV of it.
    pso::DepthStencilViewInfo shadow_info{};
    CHECK(scene.device.resolve_dsv(scene.dsv_heap->cpu_handle(1).ptr, &shadow_info));

    d3d::ViewDesc shadow_srv;
    shadow_srv.format = static_cast<uint32_t>(D::R32_FLOAT);
    shadow_srv.dimension = d3d::ViewDimension::Texture2D;
    CHECK(scene.heaps.create_shader_resource_view(kTypelessShadow, &shadow_srv, scene.srv_heap->cpu_handle(4)) ==
          d3d::Status::Ok);

    d3d::ViewDesc srgb_srv;
    srgb_srv.format = static_cast<uint32_t>(D::R8G8B8A8_UNORM_SRGB);
    srgb_srv.dimension = d3d::ViewDimension::Texture2D;
    CHECK(scene.heaps.create_shader_resource_view(Scene::kAlbedo, &srgb_srv, scene.srv_heap->cpu_handle(5)) ==
          d3d::Status::Ok);

    d3d::ViewDesc bad_uint_srv;
    bad_uint_srv.format = static_cast<uint32_t>(D::R8G8B8A8_UINT);
    bad_uint_srv.dimension = d3d::ViewDimension::Texture2D;
    CHECK(scene.heaps.create_shader_resource_view(Scene::kAlbedo, &bad_uint_srv, scene.srv_heap->cpu_handle(6)) ==
          d3d::Status::Ok);

    gpu::CommandEncoder enc;
    bridge::CommandListBridge list(scene.device, enc);
    scene.bind_root(list);
    const d3d::CpuDescriptorHandle rtv0 = scene.rtv(0);
    list.OMSetRenderTargets(1, &rtv0, false, nullptr);

    auto draw_with_srv_slot = [&](uint32_t slot) {
        const size_t mark = enc.size();
        CHECK(list.SetGraphicsRootDescriptorTable(0, scene.srv_heap->gpu_handle(slot)) == d3d::Status::Ok);
        list.DrawInstanced(3, 1, 0, 0);
        GpuBindGroupEntry2 tex_entry{};
        bool found = false;
        for (const Packet& pk : decode(enc, mark)) {
            if (pk.opcode == GpuOpcode::CreateBindGroup2) {
                tex_entry = pod<GpuBindGroupEntry2>(pk, sizeof(GpuCreateBindGroup2Payload) + sizeof(GpuBindGroupEntry2));
                found = true;
            }
        }
        CHECK(found);
        return tex_entry;
    };

    const GpuBindGroupEntry2 e_shadow = draw_with_srv_slot(4);
    CHECK(e_shadow.resource_id == kTypelessShadow);
    CHECK(e_shadow.view.format == static_cast<uint8_t>(F::Depth32Float));
    CHECK(e_shadow.view.aspect == static_cast<uint8_t>(GpuTextureAspect::DepthOnly));

    const GpuBindGroupEntry2 e_srgb = draw_with_srv_slot(5);
    CHECK(e_srgb.resource_id == Scene::kAlbedo);
    CHECK(e_srgb.view.format == static_cast<uint8_t>(F::Rgba8UnormSrgb));

    const GpuBindGroupEntry2 e_bad = draw_with_srv_slot(6);
    CHECK(e_bad.resource_id == Scene::kAlbedo);
    CHECK(e_bad.view.format == static_cast<uint8_t>(F::Rgba8Unorm));
    list.Close();
}

void test_g81_zero_alloc_deterministic_hashing() {
    // 1. Deterministic FNV-1a / SplitMix64 hashes: identical inputs produce identical hashes,
    //    single-bit changes produce distinct hashes.
    const uint32_t data_a[4] = {1u, 2u, 3u, 4u};
    const uint32_t data_b[4] = {1u, 2u, 3u, 5u};
    CHECK(bridge::fnv1a_hash64(data_a, sizeof(data_a)) == bridge::fnv1a_hash64(data_a, sizeof(data_a)));
    CHECK(bridge::fnv1a_hash64(data_a, sizeof(data_a)) != bridge::fnv1a_hash64(data_b, sizeof(data_b)));
    CHECK(bridge::splitmix64(0) != bridge::splitmix64(1));
    CHECK(bridge::hash_combine64(12345ull, 67890ull) == bridge::hash_combine64(12345ull, 67890ull));
    CHECK(bridge::hash_combine64(12345ull, 67890ull) != bridge::hash_combine64(67890ull, 12345ull));

    // 2. Sampler payload hashing ignores id/reserved and canonicalizes -0.0f / NaN / Inf LOD clamps.
    GpuCreateSampler2Payload s1{};
    s1.id = 10;
    s1.address_u = static_cast<uint8_t>(GpuAddressMode::Repeat);
    s1.lod_min_clamp = 0.0f;
    s1.lod_max_clamp = 16.0f;
    GpuCreateSampler2Payload s2 = s1;
    s2.id = 999;
    s2.reserved0 = 0xAA;
    s2.reserved1 = 0xBEEF;
    s2.lod_min_clamp = -0.0f;
    CHECK(bridge::hash_sampler_payload(s1) == bridge::hash_sampler_payload(s2));
    s2.compare = static_cast<uint8_t>(pso::WgpuCompareFunction::LessEqual);
    CHECK(bridge::hash_sampler_payload(s1) != bridge::hash_sampler_payload(s2));

    // 3. ViewRecord hashing ignores GpuTextureViewDesc::reserved and distinguishes aspect/mip/usage.
    bridge::ViewRecord v1{};
    v1.texture_id = 42;
    v1.resource_id = 42;
    v1.view.format = static_cast<uint8_t>(pso::WgpuTextureFormat::Rgba8Unorm);
    v1.view.dimension = static_cast<uint8_t>(GpuTextureViewDimension::Dim2D);
    v1.view.aspect = static_cast<uint8_t>(GpuTextureAspect::All);
    v1.usage = d3d::ResourceUsage::Sampled;
    bridge::ViewRecord v2 = v1;
    v2.view.reserved = 0x77;
    CHECK(bridge::hash_view_record(v1) == bridge::hash_view_record(v2));
    v2.view.base_mip_level = 1;
    CHECK(bridge::hash_view_record(v1) != bridge::hash_view_record(v2));

    // 4. RenderPipelineKey hashing distinguishes depth_bias & MRT state deterministically.
    GpuRenderPipelineKey k1{};
    k1.vs_shader = 0x1111;
    k1.depth_stencil.depth_bias = 2;
    GpuRenderPipelineKey k2 = k1;
    CHECK(bridge::hash_render_pipeline_key(k1) == bridge::hash_render_pipeline_key(k2));
    k2.depth_stencil.depth_bias = -2;
    CHECK(bridge::hash_render_pipeline_key(k1) != bridge::hash_render_pipeline_key(k2));

    // 5. Sampler interning deduplicates -0.0f vs +0.0f LOD clamps and reuses the same sampler ID.
    d3d::DescriptorHeapRegistry heaps;
    bridge::StateBridgeDevice device(heaps, counting_options(36000));
    d3d::DescriptorHeap* srv_heap = make_heap(heaps, d3d::DescriptorHeapType::CbvSrvUav, 2, true);
    d3d::DescriptorHeap* smp_heap = make_heap(heaps, d3d::DescriptorHeapType::Sampler, 4, true);
    d3d::DescriptorHeap* rtv_heap = make_heap(heaps, d3d::DescriptorHeapType::Rtv, 1, false);
    device.register_texture(800, {pso::DxgiFormat::R8G8B8A8_UNORM, 64, 64, 1, 1, 1, GpuTextureDimension::Dim2D, false});
    device.register_texture(801, {pso::DxgiFormat::R8G8B8A8_UNORM, 64, 64, 1, 1, 1, GpuTextureDimension::Dim2D, false});
    CHECK(heaps.create_render_target_view(800, nullptr, rtv_heap->cpu_handle(0)) == d3d::Status::Ok);
    CHECK(heaps.create_shader_resource_view(801, nullptr, srv_heap->cpu_handle(0)) == d3d::Status::Ok);

    d3d::SamplerDesc sd_pos{};
    sd_pos.min_lod = 0.0f;
    sd_pos.max_lod = 8.0f;
    d3d::SamplerDesc sd_neg = sd_pos;
    sd_neg.min_lod = -0.0f;
    CHECK(heaps.create_sampler(sd_pos, smp_heap->cpu_handle(0)) == d3d::Status::Ok);
    CHECK(heaps.create_sampler(sd_neg, smp_heap->cpu_handle(1)) == d3d::Status::Ok);

    // View interning deduplicates & recycles cleanly through ViewKey.
    const uint32_t vid1 = device.test_intern_view(v1);
    const uint32_t vid2 = device.test_intern_view(v2);
    CHECK(vid1 != 0 && vid2 != 0 && vid1 != vid2);
    CHECK(device.test_intern_view(v1) == vid1);
    CHECK(device.live_view_count() == 2);
    device.unregister_resource(42);
    CHECK(device.live_view_count() == 0);
    CHECK(device.view(vid1) == nullptr && device.view(vid2) == nullptr);
    const uint32_t vid3 = device.test_intern_view(v1);
    CHECK(vid3 == vid1 || vid3 == vid2);  // recycled from free list
    CHECK(device.live_view_count() == 1);
}

int main() {
    test_g81_depth_bias_and_shadow_map_parity();
    test_g81_eight_mrt_blend_and_integer_guards();
    test_g81_stencil_faces_masks_and_dynamic_reemission();
    test_g81_primitive_topology_and_strip_index_format();
    test_g81_hdr_depth_srgb_view_formats();
    test_g81_zero_alloc_deterministic_hashing();
    test_rt_arena_dispatch();
    test_rt_off_skips_and_counts();
    test_rt_residency_wiring();
    test_rt_residency_structures_unavailable_skips();
    test_partly_held_mip_chain_clamps_views();
    test_view_table_bounded();
    test_unregister_texture_invalidates_cached_bind_groups();
    test_evicted_texture_destroy_deferred_until_after_present();
    test_content_hash_change_detection();
    test_storage_texture_format_conversion();
    test_null_storage_dispatch_dropped();
    test_depth_storage_twin();
    test_depth_storage_twin_subresources();
    test_fragment_storage_texture_format_conversion();
    test_vertex_mixed_storage_and_attachment_alias();
    test_seam_conversions();
    test_memory_backed_tables();
    test_representative_command_list();
    test_unaligned_uniform_relocation();
    test_unaligned_storage_relocation();
    test_root_srv_read_as_uniform_and_storage();
    test_cached_bind_group_follows_moved_relocation();
    test_storage_texture_needs_storage_usage();
    test_reflected_uniform_min_size();
    test_bind_group_rebuild_and_retire();
    test_cached_bind_group_drops_destroyed_texture();
    test_usage_conflict_is_diagnosed();
    test_barrier_flushes_pending_clears();
    test_partial_clear_is_reported();
    test_compute_dispatch_packets();
    test_execute_indirect_packets();
    test_indirect_clamp_batch_planner();
    test_execute_indirect_clamp_batching();
    test_mesh_pipeline_packets();
    test_binding_array_layouts();
    test_diagnostics_and_closed_list();
    test_indexed_draw_and_layout_dedup();
    test_identical_draws_create_no_new_objects();
    test_released_list_retires_bind_groups();
    test_concurrent_command_lists();
    test_dirty_texture_memory_upload();
    test_all_subresource_uploads();
    test_banded_texture_upload();
    test_bc_texture_upload_needs_feature();
    test_norm16_address_texture_upload();
    test_fence_completion_and_ref_lifetime();
    test_prewarm_pipeline_matches_on_demand();
    test_prewarm_title_first_and_concurrent_single_build();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("test_state_bridge: all checks passed\n");
    return 0;
}
