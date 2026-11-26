// CPU twin of the BVH node test against the shared vectors (contract C5), and the prelude contract.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>

#include "gpu/rt_bvh.h"
#include "rt_bvh_vectors.h"

namespace {
int g_failures = 0;
void check_impl(bool cond, const std::string& what, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_rt_bvh.cpp:%d]: %s\n", line, what.c_str());
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)
using namespace playgta6::gpu::rt_bvh;
}  // namespace

int main() {
    const auto vectors = rt_bvh_vectors::all();
    CHECK(vectors.size() >= 18);
    for (const auto& v : vectors) {
        std::map<uint64_t, uint32_t> mem;
        for (const auto& m : v.memory) {
            for (size_t i = 0; i < m.dwords.size(); ++i) mem[m.addr + 4 * i] = m.dwords[i];
        }
        const auto load = [&](uint64_t a) -> uint32_t {
            const auto it = mem.find(a);
            return it == mem.end() ? 0u : it->second;
        };
        const float inv[3] = {rt_bvh_vectors::inv_of(v.dir[0]), rt_bvh_vectors::inv_of(v.dir[1]),
                              rt_bvh_vectors::inv_of(v.dir[2])};
        const Result r = intersect(v.desc, v.node, v.extent, {v.origin[0], v.origin[1], v.origin[2]},
                                   {v.dir[0], v.dir[1], v.dir[2]}, {inv[0], inv[1], inv[2]}, load);
        char got[160];
        std::snprintf(got, sizeof got, " got %08x %08x %08x %08x want %08x %08x %08x %08x", r.w[0], r.w[1], r.w[2],
                      r.w[3], v.expected[0], v.expected[1], v.expected[2], v.expected[3]);
        check_impl(std::memcmp(r.w, v.expected, 16) == 0, v.name + got, __LINE__);
    }
    // The prelude exposes the C4 entry point and the C3 bindings, and carries the version.
    const std::string wgsl = prelude_wgsl();
    CHECK(wgsl.find("fn pg6_bvh_intersect(desc: vec4<u32>, node: vec2<u32>, extent: f32, origin: vec3<f32>, "
                    "dir: vec3<f32>,") != std::string::npos);
    CHECK(wgsl.find("-> vec4<u32>") != std::string::npos);
    for (int b = 16; b <= 21; ++b) {
        CHECK(wgsl.find("@group(3) @binding(" + std::to_string(b) + ")") != std::string::npos);
    }
    CHECK(wgsl.find("prelude, version " + std::to_string(kPreludeVersion)) != std::string::npos);
    if (g_failures == 0) std::printf("test_rt_bvh: OK (%zu vectors)\n", vectors.size());
    return g_failures == 0 ? 1 - 1 : 1;
}
