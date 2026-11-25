#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
// aligned copy and binds that; without the hook the binding is diagnosed.
void test_unaligned_uniform_relocation() {
    for (const bool with_hook : {true, false}) {
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
        cbv.size_in_bytes = 96;
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
            }
        }
        CHECK(bind_group != nullptr);
        if (bind_group != nullptr) {
            const auto entry = pod<GpuBindGroupEntry2>(*bind_group, sizeof(GpuCreateBindGroup2Payload));
            if (with_hook) {
                CHECK(entry.resource_id == 777 && entry.offset == 512 && entry.size == 256);  // whole relocated slot
                CHECK(calls.size() == 1 && calls[0].first == kCbVa + 0x40 && calls[0].second == 96);
                CHECK((list.diagnostics() & bridge::BridgeDiag::UnalignedUniform) == 0);
            } else {
                CHECK(entry.resource_id == kCb && entry.offset == 0x40);
                CHECK((list.diagnostics() & bridge::BridgeDiag::UnalignedUniform) != 0);
            }
        }
    }
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
    bridge::StateBridgeDevice device(heaps, options, &states);
    constexpr uint32_t kTarget = 100, kCb = 300, kTexture = 301, kVsModule = 1, kFsModule = 2;
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
    bridge::constant_buffer_words(kCbVa + 0x40, 96, cb);
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
    CHECK(e.resource_id == kCb && e.offset == 0x40 && e.size == 96);
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
            CHECK(back_buf->is_back_buffer());
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

int main() {
    test_content_hash_change_detection();
    test_seam_conversions();
    test_memory_backed_tables();
    test_representative_command_list();
    test_unaligned_uniform_relocation();
    test_reflected_uniform_min_size();
    test_bind_group_rebuild_and_retire();
    test_usage_conflict_is_diagnosed();
    test_barrier_flushes_pending_clears();
    test_partial_clear_is_reported();
    test_compute_dispatch_packets();
    test_diagnostics_and_closed_list();
    test_indexed_draw_and_layout_dedup();
    test_concurrent_command_lists();
    test_dirty_texture_memory_upload();
    test_all_subresource_uploads();
    test_bc_texture_upload_needs_feature();
    test_norm16_address_texture_upload();
    test_fence_completion_and_ref_lifetime();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("test_state_bridge: all checks passed\n");
    return 0;
}
