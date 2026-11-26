// The runtime-owned WGSL prelude pg6_bvh_intersect (gpu/rt_bvh.h) compiled by Dawn/Tint and run on the
// host's GPU: every shared vector (tests/rt_bvh_vectors.h) is traced in one compute dispatch against
// the arena/page-directory layout of gpu/rt_arena (contract C3 bindings, group 3, 16-21), and the
// result bits are compared with the CPU twin. Exits 77 (CTest: skipped) without a WebGPU adapter.
// Built with -DPLAYGTA6_NATIVE_EXECUTOR=ON.
#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "gpu/rt_arena.h"
#include "gpu/rt_bvh.h"
#include "rt_bvh_vectors.h"

namespace {
using namespace playgta6::gpu;
constexpr uint64_t kWait = UINT64_MAX;
constexpr uint32_t kArenaPages = 8;  // pages per arena in this test (kArenaCount arenas)
constexpr uint32_t kStride = 16;     // input dwords per vector

std::string sv(wgpu::StringView s) {
    if (!s.data) return std::string();
    return std::string(s.data, s.length == WGPU_STRLEN ? std::strlen(s.data) : s.length);
}

const char* kMain = R"WGSL(
@group(0) @binding(0) var<storage, read_write> pg6_test_io: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = arrayLength(&pg6_test_io) / 20u;
    let i = gid.x;
    if (i >= n) { return; }
    let b = i * 16u;
    let desc = vec4<u32>(pg6_test_io[b], pg6_test_io[b + 1u], pg6_test_io[b + 2u], pg6_test_io[b + 3u]);
    let node = vec2<u32>(pg6_test_io[b + 4u], pg6_test_io[b + 5u]);
    let extent = bitcast<f32>(pg6_test_io[b + 6u]);
    let o = vec3<f32>(bitcast<f32>(pg6_test_io[b + 7u]), bitcast<f32>(pg6_test_io[b + 8u]), bitcast<f32>(pg6_test_io[b + 9u]));
    let d = vec3<f32>(bitcast<f32>(pg6_test_io[b + 10u]), bitcast<f32>(pg6_test_io[b + 11u]), bitcast<f32>(pg6_test_io[b + 12u]));
    let inv = vec3<f32>(bitcast<f32>(pg6_test_io[b + 13u]), bitcast<f32>(pg6_test_io[b + 14u]), bitcast<f32>(pg6_test_io[b + 15u]));
    let r = pg6_bvh_intersect(desc, node, extent, o, d, inv);
    let ob = n * 16u + i * 4u;
    pg6_test_io[ob] = r.x;
    pg6_test_io[ob + 1u] = r.y;
    pg6_test_io[ob + 2u] = r.z;
    pg6_test_io[ob + 3u] = r.w;
}
)WGSL";

uint32_t fbits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
}  // namespace

int main() {
    wgpu::InstanceFeatureName feats[] = {wgpu::InstanceFeatureName::TimedWaitAny};
    wgpu::InstanceDescriptor idesc{};
    idesc.requiredFeatureCount = 1;
    idesc.requiredFeatures = feats;
    wgpu::Instance instance = wgpu::CreateInstance(&idesc);
    if (!instance) {
        std::fprintf(stderr, "no instance\n");
        return 77;
    }
    wgpu::Adapter adapter;
    instance.WaitAny(instance.RequestAdapter(
                         nullptr, wgpu::CallbackMode::WaitAnyOnly,
                         [](wgpu::RequestAdapterStatus s, wgpu::Adapter a, wgpu::StringView, wgpu::Adapter* out) {
                             if (s == wgpu::RequestAdapterStatus::Success) *out = std::move(a);
                         },
                         &adapter),
                     kWait);
    if (!adapter) {
        std::printf("test_rt_bvh_gpu: SKIP (no WebGPU adapter)\n");
        return 77;
    }
    std::vector<std::string> errors;
    wgpu::DeviceDescriptor ddesc{};
    ddesc.SetUncapturedErrorCallback(
        [](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView m, std::vector<std::string>* e) { e->push_back(sv(m)); },
        &errors);
    wgpu::Device device;
    instance.WaitAny(adapter.RequestDevice(
                         &ddesc, wgpu::CallbackMode::WaitAnyOnly,
                         [](wgpu::RequestDeviceStatus s, wgpu::Device d, wgpu::StringView, wgpu::Device* out) {
                             if (s == wgpu::RequestDeviceStatus::Success) *out = std::move(d);
                         },
                         &device),
                     kWait);
    if (!device) {
        std::printf("test_rt_bvh_gpu: SKIP (no device)\n");
        return 77;
    }
    wgpu::Queue queue = device.GetQueue();
    int failures = 0;
    auto fail = [&](const std::string& m) {
        std::fprintf(stderr, "FAIL: %s\n", m.c_str());
        ++failures;
    };

    // Arena + directory exactly as the runtime builds them; page contents come from the vectors.
    // Every vector uses the same node addresses (kBase), so vector i is relocated by i * 64 KiB (its own
    // page(s)): the descriptor base moves when it carries the address, the node pointer when the
    // address rides in a 64-bit pointer (descriptor base 0).
    auto vectors = rt_bvh_vectors::all();
    rt_arena::Arena arena(rt_arena::kArenaCount, kArenaPages);
    std::map<uint64_t, uint32_t> mem;  // all relocated memory, byte address -> dword
    std::vector<std::map<uint64_t, uint32_t>> vmem(vectors.size());
    for (size_t vi = 0; vi < vectors.size(); ++vi) {
        auto& v = vectors[vi];
        const uint64_t delta = (vi + 1) * rt_arena::kPageBytes;
        if (v.desc[0] != 0 || (v.desc[1] & 0xffu) != 0) v.desc[0] += static_cast<uint32_t>(delta >> 8);
        else v.node += (delta >> 6) << 3;
        for (const auto& m : v.memory)
            for (size_t i = 0; i < m.dwords.size(); ++i) {
                const uint64_t a = m.addr + delta + 4 * i;
                mem[a] = vmem[vi][a] = m.dwords[i];
                const auto r = arena.ensure_range(a, 4);
                if (r != rt_arena::Ensure::Resident && r != rt_arena::Ensure::Uploaded) fail("arena ensure_range failed");
            }
    }
    std::vector<uint32_t> arena_words[rt_arena::kArenaCount];
    const size_t page_words = rt_arena::kPageBytes / 4;
    for (auto& a : arena_words) a.assign(page_words * kArenaPages, 0);
    for (const auto& u : arena.take_uploads()) {
        for (const auto& [a, w] : mem) {
            if (rt_arena::page_of(a) == u.page)
                arena_words[u.arena][u.slot * page_words + ((a & (rt_arena::kPageBytes - 1)) >> 2)] = w;
        }
    }

    auto make_buf = [&](uint64_t size, wgpu::BufferUsage usage) {
        wgpu::BufferDescriptor d{};
        d.size = size;
        d.usage = usage;
        return device.CreateBuffer(&d);
    };
    const auto st = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
    wgpu::Buffer abuf[rt_arena::kArenaCount];
    for (uint32_t i = 0; i < rt_arena::kArenaCount; ++i) {
        abuf[i] = make_buf(arena_words[i].size() * 4, st);
        queue.WriteBuffer(abuf[i], 0, arena_words[i].data(), arena_words[i].size() * 4);
    }
    const auto& dir = arena.directory();
    wgpu::Buffer dbuf = make_buf(dir.size() * 4, st);
    queue.WriteBuffer(dbuf, 0, dir.data(), dir.size() * 4);
    wgpu::Buffer sbuf = make_buf(16, st);
    const uint32_t zero4[4] = {0, 0, 0, 0};
    queue.WriteBuffer(sbuf, 0, zero4, 16);

    const size_t n = vectors.size();
    std::vector<uint32_t> io(n * 20, 0);
    for (size_t i = 0; i < n; ++i) {
        const auto& v = vectors[i];
        uint32_t* p = &io[i * kStride];
        for (int k = 0; k < 4; ++k) p[k] = v.desc[k];
        p[4] = static_cast<uint32_t>(v.node);
        p[5] = static_cast<uint32_t>(v.node >> 32);
        p[6] = fbits(v.extent);
        for (int k = 0; k < 3; ++k) {
            p[7 + k] = fbits(v.origin[k]);
            p[10 + k] = fbits(v.dir[k]);
            p[13 + k] = fbits(rt_bvh_vectors::inv_of(v.dir[k]));
        }
    }
    wgpu::Buffer iobuf = make_buf(io.size() * 4, st);
    queue.WriteBuffer(iobuf, 0, io.data(), io.size() * 4);
    wgpu::Buffer rb = make_buf(io.size() * 4, wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst);

    const std::string src = std::string(rt_bvh::prelude_wgsl()) + kMain;
    wgpu::ShaderSourceWGSL wgsl{};
    wgsl.code = wgpu::StringView(src.data(), src.size());
    wgpu::ShaderModuleDescriptor smd{};
    smd.nextInChain = &wgsl;
    wgpu::ShaderModule sm = device.CreateShaderModule(&smd);
    instance.WaitAny(
        sm.GetCompilationInfo(
            wgpu::CallbackMode::WaitAnyOnly,
            [](wgpu::CompilationInfoRequestStatus, const wgpu::CompilationInfo* info, std::vector<std::string>* e) {
                for (size_t i = 0; i < info->messageCount; ++i)
                    if (info->messages[i].type == wgpu::CompilationMessageType::Error)
                        e->push_back("WGSL line " + std::to_string(info->messages[i].lineNum) + ": " +
                                     sv(info->messages[i].message));
            },
            &errors),
        kWait);
    if (!errors.empty()) {
        for (const auto& e : errors) std::fprintf(stderr, "%s\n", e.c_str());
        std::fprintf(stderr, "FAIL: prelude did not compile\n");
        return 1;
    }
    wgpu::ComputePipelineDescriptor cpd{};
    cpd.compute.module = sm;
    cpd.compute.entryPoint = "main";
    wgpu::ComputePipeline pipe = device.CreateComputePipeline(&cpd);

    wgpu::BindGroupEntry g0[1] = {};
    g0[0].binding = 0;
    g0[0].buffer = iobuf;
    g0[0].size = io.size() * 4;
    wgpu::BindGroupDescriptor bd0{};
    bd0.layout = pipe.GetBindGroupLayout(0);
    bd0.entryCount = 1;
    bd0.entries = g0;
    wgpu::BindGroup bg0 = device.CreateBindGroup(&bd0);
    wgpu::BindGroupEntry g3[6] = {};
    wgpu::Buffer* bufs[6] = {&abuf[0], &abuf[1], &abuf[2], &abuf[3], &dbuf, &sbuf};
    for (uint32_t i = 0; i < 6; ++i) {
        g3[i].binding = rt_arena::kBindArena0 + i;
        g3[i].buffer = *bufs[i];
        g3[i].size = bufs[i]->GetSize();
    }
    wgpu::BindGroupDescriptor bd3{};
    bd3.layout = pipe.GetBindGroupLayout(rt_arena::kBindGroup);
    bd3.entryCount = 6;
    bd3.entries = g3;
    wgpu::BindGroup bg3 = device.CreateBindGroup(&bd3);

    wgpu::CommandEncoder enc = device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = enc.BeginComputePass();
    pass.SetPipeline(pipe);
    pass.SetBindGroup(0, bg0);
    pass.SetBindGroup(rt_arena::kBindGroup, bg3);
    pass.DispatchWorkgroups(static_cast<uint32_t>(n));
    pass.End();
    enc.CopyBufferToBuffer(iobuf, 0, rb, 0, io.size() * 4);
    wgpu::CommandBuffer cb = enc.Finish();
    queue.Submit(1, &cb);
    wgpu::MapAsyncStatus ms = wgpu::MapAsyncStatus::Error;
    instance.WaitAny(rb.MapAsync(
                         wgpu::MapMode::Read, 0, io.size() * 4, wgpu::CallbackMode::WaitAnyOnly,
                         [](wgpu::MapAsyncStatus s, wgpu::StringView, wgpu::MapAsyncStatus* out) { *out = s; }, &ms),
                     kWait);
    if (ms != wgpu::MapAsyncStatus::Success) {
        fail("map failed");
        return 1;
    }
    const uint32_t* res = static_cast<const uint32_t*>(rb.GetConstMappedRange(0, io.size() * 4));

    for (size_t i = 0; i < n; ++i) {
        const auto& v = vectors[i];
        const float inv[3] = {rt_bvh_vectors::inv_of(v.dir[0]), rt_bvh_vectors::inv_of(v.dir[1]),
                              rt_bvh_vectors::inv_of(v.dir[2])};
        const auto load = [&](uint64_t a) -> uint32_t {
            const auto it = vmem[i].find(a);
            return it == vmem[i].end() ? 0u : it->second;
        };
        const auto cpu = rt_bvh::intersect(v.desc, v.node, v.extent, {v.origin[0], v.origin[1], v.origin[2]},
                                           {v.dir[0], v.dir[1], v.dir[2]}, {inv[0], inv[1], inv[2]}, load);
        const uint32_t* g = res + n * 16 + i * 4;
        if (std::memcmp(g, cpu.w, 16) != 0 || std::memcmp(g, v.expected, 16) != 0) {
            char b[256];
            std::snprintf(b, sizeof b, "%s: gpu %08x %08x %08x %08x cpu %08x %08x %08x %08x", v.name.c_str(), g[0],
                          g[1], g[2], g[3], cpu.w[0], cpu.w[1], cpu.w[2], cpu.w[3]);
            fail(b);
        }
    }
    rb.Unmap();
    for (const auto& e : errors) fail("device error: " + e);
    if (failures == 0) std::printf("test_rt_bvh_gpu: OK (%zu vectors bit-exact on GPU)\n", n);
    return failures == 0 ? 0 : 1;
}
