// Software image_bvh_intersect_ray / image_bvh64_intersect_ray (docs/RAYTRACING_DESIGN.md sections 2
// and 6, contract C4/C5): the per-node test on the console's native node format, as a CPU twin of
// the runtime-owned WGSL prelude function pg6_bvh_intersect. One implementation of the semantics,
// pinned by shared test vectors (tests/rt_bvh_vectors.h) that run against both.
//
// Choices the design leaves open ("unknown on the console, assumed") are made here and pinned by the
// vectors: box slab tests are inclusive; a NaN slab (0 * inf) constrains nothing on that axis; any
// NaN in the ray gives an all-miss; an empty box (min > max on an axis) is never entered; triangle
// edge functions use single precision with no double fallback; zero edge functions count as inside.
#pragma once

#include <cstdint>
#include <functional>

namespace playgta6::gpu::rt_bvh {

// Version of the WGSL prelude (prelude_wgsl). Bump on any change to its semantics or interface; the
// translator reports the version it was written against.
constexpr uint32_t kPreludeVersion = 1;

constexpr uint32_t kNoChild = 0xFFFFFFFFu;
constexpr uint64_t kAddressMask = (1ull << 40) - 1;

struct Vec3 {
    float x, y, z;
};

// Reads the dword at a 40-bit GPU address (what pg6_load32 does on the GPU).
using LoadU32 = std::function<uint32_t(uint64_t addr)>;

struct Result {
    uint32_t w[4];
    bool operator==(const Result& o) const {
        return w[0] == o.w[0] && w[1] == o.w[1] && w[2] == o.w[2] && w[3] == o.w[3];
    }
};

// `desc`: the 128-bit BVH descriptor (base in 256-byte units in dw0 and the low byte of dw1, sort in
// dw1 bit 31). `node`: the 32- or 64-bit node pointer (kind in the low 3 bits; the node is at
// base + ((ptr >> 3) << 6)). Box nodes (kind 4 f16, kind 5 f32) return the four child pointers the
// ray enters within [0, extent] (sorted nearest first when sort is set; 0xFFFFFFFF for a miss);
// triangle kinds 0-3 return t_num, t_den, i_num, j_num as f32 bits (a miss is +inf, 1, 0, 0);
// instances and procedural nodes (6, 7) answer a triangle miss.
Result intersect(const uint32_t desc[4], uint64_t node, float extent, Vec3 origin, Vec3 dir, Vec3 inv_dir,
                 const LoadU32& load);

// The WGSL prelude: arena bindings (group 3, 16-21), pg6_load32 (directory walk), and
// pg6_bvh_intersect(desc: vec4<u32>, node: vec2<u32>, extent: f32, origin: vec3<f32>,
// dir: vec3<f32>, inv_dir: vec3<f32>) -> vec4<u32>.
const char* prelude_wgsl();

}  // namespace playgta6::gpu::rt_bvh
