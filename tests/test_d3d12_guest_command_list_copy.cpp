// Regression tests for GPU-to-GPU texture copies (CopyResource, CopyTextureRegion, and Xbox slot 85
// CopyTextureRegionX) in runtime/gpu/d3d12_guest_command_list.cpp.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <set>
#include <utility>
#include <vector>

#include "abi/playgta6_abi.h"
#include "gpu/d3d12_guest_objects.h"
#include "gpu/d3d12_guest_resources.h"
#include "gpu/d3d12_xs_exports.h"
#include "gpu/small_texture_pool.h"
#include "guest/com_bridge.h"
#include "guest/guest_runtime.h"
#include "os/vmem.h"

namespace {

using namespace playgta6;
using namespace playgta6::guest;
namespace com = playgta6::guest::com;
namespace dg = playgta6::gpu::d3d12_guest;
namespace dx = playgta6::gpu::d3d12_xs_exports;
namespace pso = playgta6::gpu::pso;

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_d3d12_guest_command_list_copy.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

struct Packet {
    GpuOpcode opcode{};
    std::vector<uint8_t> payload;
};

template <typename T>
T payload_as(const Packet& p) {
    T v{};
    if (p.payload.size() >= sizeof(T)) {
        std::memcpy(&v, p.payload.data(), sizeof(T));
    }
    return v;
}

std::vector<Packet> parse_packets(const std::vector<uint8_t>& buf, size_t from = 0) {
    std::vector<Packet> out;
    size_t pos = from;
    while (pos + sizeof(GpuPacketHeader) <= buf.size()) {
        GpuPacketHeader h{};
        std::memcpy(&h, buf.data() + pos, sizeof(h));
        pos += sizeof(h);
        if (pos + h.payload_bytes > buf.size()) {
            break;
        }
        Packet p;
        p.opcode = static_cast<GpuOpcode>(h.opcode);
        p.payload.assign(buf.data() + pos, buf.data() + pos + h.payload_bytes);
        out.push_back(std::move(p));
        pos += h.payload_bytes;
    }
    return out;
}

size_t count_op(const std::vector<Packet>& ps, GpuOpcode op) {
    size_t n = 0;
    for (const Packet& p : ps) {
        if (p.opcode == op) {
            ++n;
        }
    }
    return n;
}

constexpr uint64_t kVCall = 0x140010000ull;

struct VCallSpec {
    uint64_t target = 0;
    uint32_t slot = 0;
    std::vector<uint64_t> args;
};

thread_local VCallSpec t_spec;

void sub_vcall(GuestState* s) {
    static constexpr uint32_t kRegs[4] = {GuestReg::Rcx, GuestReg::Rdx, GuestReg::R8, GuestReg::R9};
    const size_t n = t_spec.args.size();
    const uint64_t extra = n > 4 ? (n - 4) * 8u : 0u;
    const uint64_t frame = ((0x20u + extra + 15u) & ~uint64_t{15u}) + 8u;
    s->gpr[GuestReg::Rsp] -= frame;
    const uint64_t rsp = s->gpr[GuestReg::Rsp];
    for (size_t i = 0; i < n; ++i) {
        if (i < 4) {
            s->gpr[kRegs[i]] = t_spec.args[i];
        } else {
            CHECK(com::write_guest(rsp + 0x20u + 8u * (i - 4u), t_spec.args[i]));
        }
    }
    uint64_t target = t_spec.target;
    if (target == 0) {
        uint64_t vtable = 0;
        CHECK(com::read_guest(s->gpr[GuestReg::Rcx], vtable));
        CHECK(com::read_guest(vtable + 8ull * t_spec.slot, target));
    }
    s->gpr[GuestReg::Rsp] -= 8u;
    CHECK(com::write_guest(s->gpr[GuestReg::Rsp], kVCall + 0x40u));
    __playgta6_indirect_call(s, target);
    CHECK(s->gpr[GuestReg::Rsp] == rsp);
    s->gpr[GuestReg::Rsp] += frame + 8u;
}

const LiftedFunctionEntry kFunctions[] = {{kVCall, sub_vcall}};

struct Harness {
    os::VirtualMemoryArena arena;
    GuestThread thread{};
    uint64_t scratch_base = 0;
    uint64_t scratch_cursor = 0;
    std::vector<uint8_t> captured;

    Harness()
        : arena(os::kDefaultArenaBase, os::kDefaultGuestSpaceSize, os::ArenaPlacement::Anywhere) {
        reset_guest_runtime();
        com::reset_com_bridge();
        dx::reset_for_tests();
        dg::reset_sret_stats();
        com::set_guest_memory(&arena);
        CHECK(register_lifted_functions(kFunctions));
        CHECK(create_guest_thread(arena, 512 * 1024, thread));
        CHECK(dx::register_d3d12_xs_exports() == 3);
        scratch_base = arena.virtual_alloc(0, 256u << 10, os::kMemReserve | os::kMemCommit, os::PageProtect::ReadWrite);
        scratch_cursor = scratch_base;
        CHECK(scratch_base != 0);
        dg::DeviceOptions opts{};
        opts.defer_gpu_objects = true;
        opts.bridge.features = pso::WgpuFeature::TextureCompressionBc | pso::WgpuFeature::Bgra8UnormStorage;
        opts.sink = [this](const uint8_t* bytes, size_t size) {
            captured.insert(captured.end(), bytes, bytes + size);
        };
        dx::set_device_options(opts);
    }

    ~Harness() {
        com::reset_com_bridge();
        dx::reset_for_tests();
        destroy_guest_thread(arena, thread);
        com::set_guest_memory(nullptr);
        reset_guest_runtime();
    }

    uint64_t alloc_scratch(size_t bytes, size_t align = 16) {
        scratch_cursor = (scratch_cursor + align - 1u) & ~(align - 1u);
        const uint64_t va = scratch_cursor;
        scratch_cursor += bytes;
        return va;
    }

    template <typename T>
    uint64_t put(const T& v) {
        const uint64_t va = alloc_scratch(sizeof(T), alignof(T));
        CHECK(com::write_guest(va, v));
        return va;
    }

    uint64_t put_bytes(const void* data, size_t bytes) {
        const uint64_t va = alloc_scratch(bytes, 16);
        CHECK(com::write_guest(va, data, bytes));
        return va;
    }

    uint64_t dcall(uint64_t target, std::initializer_list<uint64_t> args) {
        t_spec = VCallSpec{};
        t_spec.target = target;
        t_spec.args.assign(args.begin(), args.end());
        return call_guest(arena, thread.state, kVCall);
    }

    uint64_t vcall(uint64_t obj, uint32_t slot, std::initializer_list<uint64_t> args = {}) {
        t_spec = VCallSpec{};
        t_spec.slot = slot;
        t_spec.args.push_back(obj);
        t_spec.args.insert(t_spec.args.end(), args.begin(), args.end());
        return call_guest(arena, thread.state, kVCall);
    }

    uint64_t create_device() {
        uint8_t params[0x50] = {};
        const uint32_t version = 4;
        std::memcpy(params, &version, 4);
        const uint64_t out = put<uint64_t>(0);
        const uint64_t target = dx::export_address("D3D12XboxCreateDevice");
        CHECK(target != 0);
        const int32_t hr = static_cast<int32_t>(
            dcall(target, {0, put_bytes(params, sizeof(params)), put(dg::IID_ID3D12Device11), out}));
        CHECK(hr == 0);
        uint64_t dev = 0;
        CHECK(com::read_guest(out, dev));
        return dev;
    }

    uint64_t create_queue(uint64_t dev) {
        dg::GuestCommandQueueDesc qd{};
        const uint64_t out = put<uint64_t>(0);
        CHECK(static_cast<int32_t>(vcall(dev, 8, {put(qd), put(dg::IID_ID3D12CommandQueue), out})) == 0);
        uint64_t q = 0;
        CHECK(com::read_guest(out, q));
        return q;
    }

    uint64_t create_allocator(uint64_t dev) {
        const uint64_t out = put<uint64_t>(0);
        CHECK(static_cast<int32_t>(vcall(dev, 9, {0, put(dg::IID_ID3D12CommandAllocator), out})) == 0);
        uint64_t a = 0;
        CHECK(com::read_guest(out, a));
        return a;
    }

    uint64_t create_list(uint64_t dev, uint64_t alloc) {
        const uint64_t out = put<uint64_t>(0);
        CHECK(static_cast<int32_t>(vcall(dev, 12, {0, 0, alloc, 0, put(dg::IID_ID3D12GraphicsCommandList), out})) == 0);
        uint64_t l = 0;
        CHECK(com::read_guest(out, l));
        CHECK(static_cast<int32_t>(vcall(l, 9)) == 0);  // Close
        return l;
    }

    uint64_t create_committed(uint64_t dev, const dg::GuestResourceDesc& desc, uint32_t initial_state = 0x400) {
        dg::GuestHeapProperties hp{};
        hp.Type = 1;  // DEFAULT
        const uint64_t out = put<uint64_t>(0);
        CHECK(static_cast<int32_t>(
                  vcall(dev, 27, {put(hp), 0, put(desc), initial_state, 0, put(dg::IID_ID3D12Resource), out})) == 0);
        uint64_t r = 0;
        CHECK(com::read_guest(out, r));
        return r;
    }

    void execute(uint64_t queue, std::initializer_list<uint64_t> lists) {
        std::vector<uint64_t> arr(lists.begin(), lists.end());
        const uint64_t va = put_bytes(arr.data(), arr.size() * 8u);
        vcall(queue, 35 /* ExecuteCommandListsX */, {arr.size(), va, 0, 0});
    }

    void release(std::initializer_list<uint64_t> objs) {
        for (uint64_t o : objs) {
            if (o != 0) {
                vcall(o, 2 /* Release */);
            }
        }
    }
};

dg::Resource* host_resource(uint64_t va) {
    auto r = com::InterfaceArg<dg::IResourceHost>::resolve(va);
    return r ? static_cast<dg::Resource*>(r.get()) : nullptr;
}

dg::GuestResourceDesc tex2d_desc(uint32_t w, uint32_t h, uint32_t format, uint32_t flags, uint16_t mips = 1,
                                 uint16_t layers = 1) {
    dg::GuestResourceDesc d{};
    d.Dimension = dg::ResourceDimension::Texture2D;
    d.Width = w;
    d.Height = h;
    d.DepthOrArraySize = layers;
    d.MipLevels = mips;
    d.Format = format;
    d.SampleCount = 1;
    d.Flags = flags;
    return d;
}

dg::GuestResourceDesc tex3d_desc(uint32_t w, uint32_t h, uint32_t depth, uint32_t format, uint32_t flags,
                                 uint16_t mips = 1) {
    dg::GuestResourceDesc d{};
    d.Dimension = dg::ResourceDimension::Texture3D;
    d.Width = w;
    d.Height = h;
    d.DepthOrArraySize = static_cast<uint16_t>(depth);
    d.MipLevels = mips;
    d.Format = format;
    d.SampleCount = 1;
    d.Flags = flags;
    return d;
}

void test_copy_resource_color_srgb_depth_int() {
    Harness h;
    const uint64_t dev = h.create_device();
    const uint64_t q = h.create_queue(dev);
    const uint64_t alloc = h.create_allocator(dev);
    const uint64_t list = h.create_list(dev, alloc);
    com::Ref<dg::Device> hdev = dx::current_device();

    // 1. Multi-mip, 2-slice 2D-Array RGBA8_UNORM RT -> RT CopyResource.
    const uint64_t src_rt = h.create_committed(dev, tex2d_desc(64, 64, 28, dg::ResourceFlags::AllowRenderTarget, 2, 2));
    const uint64_t dst_rt = h.create_committed(dev, tex2d_desc(64, 64, 28, dg::ResourceFlags::AllowRenderTarget, 2, 2));
    const uint32_t src_rt_id = host_resource(src_rt)->id();
    const uint32_t dst_rt_id = host_resource(dst_rt)->id();

    // 2. sRGB -> sRGB CopyResource: must use linear Rgba8Unorm view format for bit-exact copy.
    const uint64_t src_srgb = h.create_committed(dev, tex2d_desc(32, 32, 29, dg::ResourceFlags::AllowRenderTarget));
    const uint64_t dst_srgb = h.create_committed(dev, tex2d_desc(32, 32, 29, dg::ResourceFlags::AllowRenderTarget));
    const uint32_t src_srgb_id = host_resource(src_srgb)->id();
    const uint32_t dst_srgb_id = host_resource(dst_srgb)->id();

    // 3. Depth -> Depth (D32_FLOAT) AND Depth -> R32_FLOAT RT CopyResource.
    const uint64_t src_depth = h.create_committed(dev, tex2d_desc(32, 32, 40, dg::ResourceFlags::AllowDepthStencil));
    const uint64_t dst_depth = h.create_committed(dev, tex2d_desc(32, 32, 40, dg::ResourceFlags::AllowDepthStencil));
    const uint64_t dst_r32f = h.create_committed(dev, tex2d_desc(32, 32, 41, dg::ResourceFlags::AllowRenderTarget));
    const uint32_t src_depth_id = host_resource(src_depth)->id();
    const uint32_t dst_depth_id = host_resource(dst_depth)->id();
    const uint32_t dst_r32f_id = host_resource(dst_r32f)->id();

    // 4. Integer RT -> Integer RT (R32_UINT and R16G16_SINT).
    const uint64_t src_u32 = h.create_committed(dev, tex2d_desc(16, 16, 42, dg::ResourceFlags::AllowRenderTarget));
    const uint64_t dst_u32 = h.create_committed(dev, tex2d_desc(16, 16, 42, dg::ResourceFlags::AllowRenderTarget));
    const uint64_t src_i16 = h.create_committed(dev, tex2d_desc(16, 16, 38, dg::ResourceFlags::AllowRenderTarget));
    const uint64_t dst_i16 = h.create_committed(dev, tex2d_desc(16, 16, 38, dg::ResourceFlags::AllowRenderTarget));

    CHECK(static_cast<int32_t>(h.vcall(list, 10 /* Reset */, {alloc, 0})) == 0);
    const size_t mark = h.captured.size();
    h.vcall(list, 17 /* CopyResource */, {dst_rt, src_rt});
    h.vcall(list, 17 /* CopyResource */, {dst_srgb, src_srgb});
    h.vcall(list, 17 /* CopyResource */, {dst_depth, src_depth});
    h.vcall(list, 17 /* CopyResource */, {dst_r32f, src_depth});
    h.vcall(list, 17 /* CopyResource */, {dst_u32, src_u32});
    h.vcall(list, 17 /* CopyResource */, {dst_i16, src_i16});
    CHECK(static_cast<int32_t>(h.vcall(list, 9 /* Close */)) == 0);
    h.execute(q, {list});

    CHECK(hdev->stats().unsupported_copies == 0);
    const auto ps = parse_packets(h.captured, mark);

    // 4 subresources (dst_rt) + 1 (srgb) + 1 (depth) + 1 (depth->r32f) + 1 (u32) + 1 (i16) = 9 passes.
    CHECK(count_op(ps, GpuOpcode::BeginRenderPass) == 9);
    CHECK(count_op(ps, GpuOpcode::Draw) == 9);
    CHECK(count_op(ps, GpuOpcode::EndRenderPass) == 9);

    std::set<uint32_t> created_textures;
    std::set<std::pair<uint32_t, uint32_t>> rt_subresources;
    bool saw_srgb_linear_rt = false;
    bool saw_srgb_linear_srv = false;
    bool saw_depth_copy_pass = false;
    bool saw_depth_to_r32f_pass = false;
    bool saw_depth_only_srv = false;

    for (const Packet& p : ps) {
        if (p.opcode == GpuOpcode::CreateTexture2) {
            created_textures.insert(payload_as<GpuCreateTexture2Payload>(p).id);
        } else if (p.opcode == GpuOpcode::BeginRenderPass) {
            const auto rp = payload_as<GpuBeginRenderPassPayload>(p);
            if (rp.color_count == 1 && rp.color[0].texture_id == dst_rt_id) {
                CHECK(rp.color[0].load_op == static_cast<uint8_t>(GpuLoadOp::Clear));
                CHECK(rp.color[0].store_op == static_cast<uint8_t>(GpuStoreOp::Store));
                rt_subresources.emplace(rp.color[0].view.base_mip_level, rp.color[0].view.base_array_layer);
            } else if (rp.color_count == 1 && rp.color[0].texture_id == dst_srgb_id) {
                saw_srgb_linear_rt = (rp.color[0].view.format == static_cast<uint8_t>(pso::WgpuTextureFormat::Rgba8Unorm));
            } else if (rp.color_count == 0 && rp.depth_stencil.texture_id == dst_depth_id) {
                saw_depth_copy_pass = (rp.depth_stencil.depth_store_op == static_cast<uint8_t>(GpuStoreOp::Store));
            } else if (rp.color_count == 1 && rp.color[0].texture_id == dst_r32f_id) {
                saw_depth_to_r32f_pass = (rp.color[0].view.format == static_cast<uint8_t>(pso::WgpuTextureFormat::R32Float));
            }
        } else if (p.opcode == GpuOpcode::CreateBindGroup2) {
            const auto bg = payload_as<GpuCreateBindGroup2Payload>(p);
            if (bg.entry_count >= 1 && p.payload.size() >= sizeof(bg) + sizeof(GpuBindGroupEntry2)) {
                GpuBindGroupEntry2 e{};
                std::memcpy(&e, p.payload.data() + sizeof(bg), sizeof(e));
                if (e.resource_id == src_srgb_id &&
                    e.view.format == static_cast<uint8_t>(pso::WgpuTextureFormat::Rgba8Unorm)) {
                    saw_srgb_linear_srv = true;
                }
                if (e.resource_id == src_depth_id && e.view.aspect == static_cast<uint8_t>(GpuTextureAspect::DepthOnly)) {
                    saw_depth_only_srv = true;
                }
            }
        }
    }

    CHECK(created_textures.count(src_rt_id) == 1 && created_textures.count(dst_rt_id) == 1);
    CHECK(rt_subresources.size() == 4);
    CHECK(rt_subresources.count({0, 0}) == 1 && rt_subresources.count({0, 1}) == 1 &&
          rt_subresources.count({1, 0}) == 1 && rt_subresources.count({1, 1}) == 1);
    CHECK(saw_srgb_linear_rt && saw_srgb_linear_srv);
    CHECK(saw_depth_copy_pass && saw_depth_to_r32f_pass && saw_depth_only_srv);

    h.release({list, alloc, q, src_rt, dst_rt, src_srgb, dst_srgb, src_depth, dst_depth, dst_r32f, src_u32, dst_u32,
               src_i16, dst_i16, dev});
}

void test_copy_texture_region_subbox_and_xbox_slot85() {
    Harness h;
    const uint64_t dev = h.create_device();
    const uint64_t q = h.create_queue(dev);
    const uint64_t alloc = h.create_allocator(dev);
    const uint64_t list = h.create_list(dev, alloc);
    com::Ref<dg::Device> hdev = dx::current_device();

    // src: 64x64, 2 mips, 2 layers (mip 1 is 32x32, subresource index for mip 1 layer 1 = 1 + 1*2 = 3).
    // dst: 64x64, 1 mip, 1 layer.
    const uint64_t src = h.create_committed(dev, tex2d_desc(64, 64, 28, dg::ResourceFlags::AllowRenderTarget, 2, 2));
    const uint64_t dst = h.create_committed(dev, tex2d_desc(64, 64, 28, dg::ResourceFlags::AllowRenderTarget, 1, 1));
    const uint32_t src_id = host_resource(src)->id();
    const uint32_t dst_id = host_resource(dst)->id();

    CHECK(static_cast<int32_t>(h.vcall(list, 10 /* Reset */, {alloc, 0})) == 0);
    const size_t mark = h.captured.size();

    dg::GuestTextureCopyLocation dst_loc{};
    dst_loc.pResource = dst;
    dst_loc.Type = 0;
    dst_loc.PlacedFootprint.Offset = 0;

    dg::GuestTextureCopyLocation src_loc{};
    src_loc.pResource = src;
    src_loc.Type = 0;
    src_loc.PlacedFootprint.Offset = 3;  // mip 1, layer 1

    // Copy 16x12 box from src (8, 4) -> dst (20, 10). Delta = (-12, -6).
    dg::GuestBox box{8, 4, 0, 24, 16, 1};
    h.vcall(list, 16 /* CopyTextureRegion */, {h.put(dst_loc), 20, 10, 0, h.put(src_loc), h.put(box)});

    // Also test Xbox slot 85 CopyTextureRegionX forwarding with flags = 0.
    dg::GuestBox box2{0, 0, 0, 32, 32, 1};
    h.vcall(list, 85 /* CopyTextureRegionX */, {h.put(dst_loc), 4, 8, 0, h.put(src_loc), h.put(box2), 0});

    CHECK(static_cast<int32_t>(h.vcall(list, 9 /* Close */)) == 0);
    h.execute(q, {list});

    CHECK(hdev->stats().unsupported_copies == 0);
    const auto ps = parse_packets(h.captured, mark);
    CHECK(count_op(ps, GpuOpcode::BeginRenderPass) == 2);
    CHECK(count_op(ps, GpuOpcode::SetViewport) == 2);
    CHECK(count_op(ps, GpuOpcode::SetScissorRect) == 2);
    CHECK(count_op(ps, GpuOpcode::Draw) == 2);
    CHECK(count_op(ps, GpuOpcode::EndRenderPass) == 2);

    bool saw_first_load_op = false;
    bool saw_first_vp = false;
    bool saw_first_sc = false;
    bool saw_first_delta = false;
    bool saw_src_sub3 = false;
    const uint32_t expected_packed_delta =
        static_cast<uint32_t>(static_cast<uint16_t>(-12)) | (static_cast<uint32_t>(static_cast<uint16_t>(-6)) << 16u);

    for (const Packet& p : ps) {
        if (p.opcode == GpuOpcode::BeginRenderPass) {
            const auto rp = payload_as<GpuBeginRenderPassPayload>(p);
            if (rp.color_count == 1 && rp.color[0].texture_id == dst_id &&
                rp.color[0].load_op == static_cast<uint8_t>(GpuLoadOp::Load)) {
                saw_first_load_op = true;
            }
        } else if (p.opcode == GpuOpcode::SetViewport) {
            const auto vp = payload_as<GpuSetViewportPayload>(p);
            if (vp.x == 20.0f && vp.y == 10.0f && vp.width == 16.0f && vp.height == 12.0f) {
                saw_first_vp = true;
            }
        } else if (p.opcode == GpuOpcode::SetScissorRect) {
            const auto sc = payload_as<GpuSetScissorRectPayload>(p);
            if (sc.x == 20 && sc.y == 10 && sc.width == 16 && sc.height == 12) {
                saw_first_sc = true;
            }
        } else if (p.opcode == GpuOpcode::Draw) {
            const auto dr = payload_as<GpuDrawPayload>(p);
            if (dr.vertex_count == 3 && dr.instance_count == 1 && dr.first_instance == expected_packed_delta) {
                saw_first_delta = true;
            }
        } else if (p.opcode == GpuOpcode::CreateBindGroup2) {
            const auto bg = payload_as<GpuCreateBindGroup2Payload>(p);
            if (bg.entry_count == 1 && p.payload.size() >= sizeof(bg) + sizeof(GpuBindGroupEntry2)) {
                GpuBindGroupEntry2 e{};
                std::memcpy(&e, p.payload.data() + sizeof(bg), sizeof(e));
                if (e.resource_id == src_id && e.view.base_mip_level == 1 && e.view.base_array_layer == 1) {
                    saw_src_sub3 = true;
                }
            }
        }
    }
    CHECK(saw_first_load_op && saw_first_vp && saw_first_sc && saw_first_delta && saw_src_sub3);

    h.release({list, alloc, q, src, dst, dev});
}

void test_compute_storage_2d_and_3d_texture_copies() {
    Harness h;
    const uint64_t dev = h.create_device();
    const uint64_t q = h.create_queue(dev);
    const uint64_t alloc = h.create_allocator(dev);
    const uint64_t list = h.create_list(dev, alloc);
    com::Ref<dg::Device> hdev = dx::current_device();

    // 1. 2D UAV-only destination (no AllowRenderTarget) -> compute-pass copy.
    const uint64_t src2d = h.create_committed(dev, tex2d_desc(32, 32, 28, 0));
    const uint64_t dst2d = h.create_committed(dev, tex2d_desc(32, 32, 28, dg::ResourceFlags::AllowUnorderedAccess));

    // 2. 3D LUT / volume texture (16x16x8 RGBA16_FLOAT, AllowUnorderedAccess) full CopyResource + shifted sub-box.
    const uint64_t src3d = h.create_committed(dev, tex3d_desc(16, 16, 8, 10, 0));
    const uint64_t dst3d = h.create_committed(dev, tex3d_desc(16, 16, 8, 10, dg::ResourceFlags::AllowUnorderedAccess));
    const uint32_t dst3d_id = host_resource(dst3d)->id();

    CHECK(static_cast<int32_t>(h.vcall(list, 10 /* Reset */, {alloc, 0})) == 0);
    const size_t mark = h.captured.size();
    h.vcall(list, 17 /* CopyResource */, {dst2d, src2d});
    h.vcall(list, 17 /* CopyResource */, {dst3d, src3d});

    dg::GuestTextureCopyLocation dst_loc{};
    dst_loc.pResource = dst3d;
    dst_loc.Type = 0;
    dst_loc.PlacedFootprint.Offset = 0;
    dg::GuestTextureCopyLocation src_loc{};
    src_loc.pResource = src3d;
    src_loc.Type = 0;
    src_loc.PlacedFootprint.Offset = 0;
    dg::GuestBox box3d{2, 4, 1, 10, 12, 5};  // 8x8x4 box -> workgroups (2, 2, 1)
    h.vcall(list, 16 /* CopyTextureRegion */, {h.put(dst_loc), 6, 2, 3, h.put(src_loc), h.put(box3d)});

    CHECK(static_cast<int32_t>(h.vcall(list, 9 /* Close */)) == 0);
    h.execute(q, {list});

    CHECK(hdev->stats().unsupported_copies == 0);
    const auto ps = parse_packets(h.captured, mark);
    CHECK(count_op(ps, GpuOpcode::DispatchCompute) == 3);
    CHECK(count_op(ps, GpuOpcode::CreateComputePipeline2) == 3);

    bool saw_2d_dispatch = false;
    bool saw_3d_full_dispatch = false;
    bool saw_3d_sub_dispatch = false;
    bool saw_3d_view_and_params = false;
    for (const Packet& p : ps) {
        if (p.opcode == GpuOpcode::DispatchCompute) {
            const auto d = payload_as<GpuDispatchComputePayload>(p);
            if (d.workgroup_x == 4 && d.workgroup_y == 4 && d.workgroup_z == 1) saw_2d_dispatch = true;
            if (d.workgroup_x == 4 && d.workgroup_y == 4 && d.workgroup_z == 2) saw_3d_full_dispatch = true;
            if (d.workgroup_x == 2 && d.workgroup_y == 2 && d.workgroup_z == 1) saw_3d_sub_dispatch = true;
        } else if (p.opcode == GpuOpcode::CreateBindGroup2) {
            const auto bg = payload_as<GpuCreateBindGroup2Payload>(p);
            if (bg.entry_count == 3 && p.payload.size() >= sizeof(bg) + 3 * sizeof(GpuBindGroupEntry2)) {
                GpuBindGroupEntry2 e1{};
                std::memcpy(&e1, p.payload.data() + sizeof(bg) + sizeof(GpuBindGroupEntry2), sizeof(e1));
                if (e1.resource_id == dst3d_id &&
                    e1.view.dimension == static_cast<uint8_t>(GpuTextureViewDimension::Dim3D)) {
                    saw_3d_view_and_params = true;
                }
            }
        }
    }
    CHECK(saw_2d_dispatch && saw_3d_full_dispatch && saw_3d_sub_dispatch && saw_3d_view_and_params);

    h.release({list, alloc, q, src2d, dst2d, src3d, dst3d, dev});
}

void test_copy_edge_cases_fallbacks_and_device_lifecycle() {
    // Part A: edge cases & safe fallbacks.
    {
        Harness h;
        const uint64_t dev = h.create_device();
        const uint64_t q = h.create_queue(dev);
        const uint64_t alloc = h.create_allocator(dev);
        const uint64_t list = h.create_list(dev, alloc);
        com::Ref<dg::Device> hdev = dx::current_device();

        const uint64_t rt = h.create_committed(dev, tex2d_desc(32, 32, 28, dg::ResourceFlags::AllowRenderTarget));
        const uint64_t rt_other_size =
            h.create_committed(dev, tex2d_desc(64, 64, 28, dg::ResourceFlags::AllowRenderTarget));
        const uint64_t srv_only_src = h.create_committed(dev, tex2d_desc(32, 32, 28, 0));
        const uint64_t srv_only_dst = h.create_committed(dev, tex2d_desc(32, 32, 28, 0));

        CHECK(static_cast<int32_t>(h.vcall(list, 10 /* Reset */, {alloc, 0})) == 0);
        const size_t mark = h.captured.size();

        // 1. Self-copy CopyResource -> clean no-op.
        h.vcall(list, 17 /* CopyResource */, {rt, rt});
        CHECK(hdev->stats().unsupported_copies == 0 && hdev->stats().bad_arguments == 0);

        // 2. Empty D3D12_BOX -> clean no-op.
        dg::GuestTextureCopyLocation loc{};
        loc.pResource = rt;
        dg::GuestBox empty_box{10, 10, 0, 10, 20, 1};
        h.vcall(list, 16 /* CopyTextureRegion */, {h.put(loc), 0, 0, 0, h.put(loc), h.put(empty_box)});
        CHECK(hdev->stats().unsupported_copies == 0 && hdev->stats().bad_arguments == 0);

        // 3. Same-subresource shifted overlap -> safe unsupported fallback, zero invalid WebGPU pass.
        dg::GuestBox box{0, 0, 0, 16, 16, 1};
        h.vcall(list, 16 /* CopyTextureRegion */, {h.put(loc), 4, 4, 0, h.put(loc), h.put(box)});
        CHECK(hdev->stats().unsupported_copies == 1);

        // 4. Non-RT, non-UAV SRV-only dst -> safe unsupported fallback, zero invalid pass.
        h.vcall(list, 17 /* CopyResource */, {srv_only_dst, srv_only_src});
        CHECK(hdev->stats().unsupported_copies == 2);

        // 5. Dimension mismatch -> bad_argument diagnostic, zero invalid pass.
        const uint64_t bad_before = hdev->stats().bad_arguments;
        h.vcall(list, 17 /* CopyResource */, {rt_other_size, rt});
        CHECK(hdev->stats().bad_arguments == bad_before + 1);

        CHECK(static_cast<int32_t>(h.vcall(list, 9 /* Close */)) == 0);
        h.execute(q, {list});
        const auto ps = parse_packets(h.captured, mark);
        CHECK(count_op(ps, GpuOpcode::BeginRenderPass) == 0);
        CHECK(count_op(ps, GpuOpcode::DispatchCompute) == 0);

        h.release({list, alloc, q, rt, rt_other_size, srv_only_src, srv_only_dst, dev});
    }

    // Part B: Second independent Device lifecycle must re-emit setup pipelines cleanly (no stale pointer-reuse cache).
    {
        Harness h;
        const uint64_t dev = h.create_device();
        const uint64_t q = h.create_queue(dev);
        const uint64_t alloc = h.create_allocator(dev);
        const uint64_t list = h.create_list(dev, alloc);
        const uint64_t src = h.create_committed(dev, tex2d_desc(16, 16, 28, dg::ResourceFlags::AllowRenderTarget));
        const uint64_t dst = h.create_committed(dev, tex2d_desc(16, 16, 28, dg::ResourceFlags::AllowRenderTarget));

        CHECK(static_cast<int32_t>(h.vcall(list, 10 /* Reset */, {alloc, 0})) == 0);
        const size_t mark = h.captured.size();
        h.vcall(list, 17 /* CopyResource */, {dst, src});
        CHECK(static_cast<int32_t>(h.vcall(list, 9 /* Close */)) == 0);
        h.execute(q, {list});

        const auto ps = parse_packets(h.captured, mark);
        CHECK(count_op(ps, GpuOpcode::CreateShaderModule) >= 2);
        CHECK(count_op(ps, GpuOpcode::CreateRenderPipeline2) == 1);
        CHECK(count_op(ps, GpuOpcode::BeginRenderPass) == 1);
        CHECK(count_op(ps, GpuOpcode::Draw) == 1);

        h.release({list, alloc, q, src, dst, dev});
    }
}

// With the consumer's CopyTexture packet (gpu_texture_copy), a copy into a destination that is neither a
// render target nor a UAV becomes a plain texture-to-texture copy instead of being skipped.
void test_native_copy_for_plain_destination() {
    gpu::set_gpu_texture_copy(true);
    Harness h;
    const uint64_t dev = h.create_device();
    const uint64_t q = h.create_queue(dev);
    const uint64_t alloc = h.create_allocator(dev);
    const uint64_t list = h.create_list(dev, alloc);
    com::Ref<dg::Device> hdev = dx::current_device();

    const uint64_t src = h.create_committed(dev, tex2d_desc(32, 32, 28, 0));
    const uint64_t dst = h.create_committed(dev, tex2d_desc(32, 32, 28, 0));
    const uint64_t other_fmt = h.create_committed(dev, tex2d_desc(32, 32, 10 /* RGBA16F */, 0));

    CHECK(static_cast<int32_t>(h.vcall(list, 10 /* Reset */, {alloc, 0})) == 0);
    const size_t mark = h.captured.size();

    // CopyResource: whole texture.
    h.vcall(list, 17 /* CopyResource */, {dst, src});
    // CopyTextureRegion: a sub box (4..12 x 8..16) to (2, 6).
    dg::GuestTextureCopyLocation dloc{}, sloc{};
    dloc.pResource = dst;
    sloc.pResource = src;
    dg::GuestBox box{4, 8, 0, 12, 16, 1};
    h.vcall(list, 16 /* CopyTextureRegion */, {h.put(dloc), 2, 6, 0, h.put(sloc), h.put(box)});
    // Formats that are not copy-compatible stay skipped (counted).
    h.vcall(list, 17 /* CopyResource */, {other_fmt, src});
    CHECK(hdev->stats().unsupported_copies == 1);
    CHECK(static_cast<int32_t>(h.vcall(list, 9 /* Close */)) == 0);
    h.execute(q, {list});

    const auto ps = parse_packets(h.captured, mark);
    CHECK(count_op(ps, GpuOpcode::BeginRenderPass) == 0);
    CHECK(count_op(ps, GpuOpcode::DispatchCompute) == 0);
    std::vector<GpuCopyTexturePayload> copies;
    for (const Packet& p : ps) {
        if (p.opcode == kGpuOpcodeCopyTexture) {
            copies.push_back(payload_as<GpuCopyTexturePayload>(p));
        }
    }
    CHECK(copies.size() == 2);
    if (copies.size() == 2) {
        CHECK(copies[0].dst_id != copies[0].src_id);
        CHECK(copies[0].width == 32 && copies[0].height == 32 && copies[0].depth_or_layers == 1);
        CHECK(copies[0].dst_x == 0 && copies[0].src_x == 0);
        CHECK(copies[1].src_x == 4 && copies[1].src_y == 8 && copies[1].dst_x == 2 && copies[1].dst_y == 6);
        CHECK(copies[1].width == 8 && copies[1].height == 8);
        CHECK(copies[1].dst_id == copies[0].dst_id && copies[1].src_id == copies[0].src_id);
    }
    h.release({list, alloc, q, src, dst, other_fmt, dev});
    gpu::set_gpu_texture_copy(false);
}

}  // namespace

int main() {
    test_copy_resource_color_srgb_depth_int();
    test_copy_texture_region_subbox_and_xbox_slot85();
    test_compute_storage_2d_and_3d_texture_copies();
    test_copy_edge_cases_fallbacks_and_device_lifecycle();
    test_native_copy_for_plain_destination();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d test failure(s) in test_d3d12_guest_command_list_copy\n", g_failures);
        return 1;
    }
    std::printf("test_d3d12_guest_command_list_copy: all checks passed\n");
    return 0;
}
