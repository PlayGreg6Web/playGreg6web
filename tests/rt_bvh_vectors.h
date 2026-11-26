// Shared test vectors (contract C5) for the BVH node test: the CPU twin (gpu/rt_bvh.h) runs them in
// tests/test_rt_bvh.cpp, and a headless-WebGPU run of the prelude (pg6_bvh_intersect) has to give
// the same bits. Everything is plain data (u32 bit patterns), so a GPU harness can consume it as is.
// Expected values are derived by hand in the comments, not by running the twin.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace rt_bvh_vectors {

inline uint32_t fb(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

struct Memory {
    uint64_t addr;
    std::vector<uint32_t> dwords;
};

struct Vector {
    std::string name;
    uint32_t desc[4] = {0, 0, 0, 0};
    uint64_t node = 0;  // node pointer (low 3 bits = kind)
    float extent = 1000.0f;
    float origin[3] = {0, 0, 0};
    float dir[3] = {0, 0, 1};
    uint32_t expected[4] = {0, 0, 0, 0};
    std::vector<Memory> memory;
};

constexpr uint32_t kNone = 0xFFFFFFFFu;
constexpr uint32_t kBase = 0x1000;  // node 0 address; descriptor base = kBase >> 8

// One 128-byte f32 box node (kind 5): four child pointers, then 4 x (min xyz, max xyz).
struct Box {
    uint32_t ptr;
    float mn[3], mx[3];
};
inline std::vector<uint32_t> box32(const Box (&c)[4]) {
    std::vector<uint32_t> n(32, 0);
    for (int i = 0; i < 4; ++i) {
        n[i] = c[i].ptr;
        for (int k = 0; k < 3; ++k) {
            n[4 + 6 * i + k] = fb(c[i].mn[k]);
            n[4 + 6 * i + 3 + k] = fb(c[i].mx[k]);
        }
    }
    return n;
}

// A triangle node: five vertices and the selector dword (dword 15).
inline std::vector<uint32_t> tri_node(const float (&v)[5][3], uint32_t sel) {
    std::vector<uint32_t> n(16, 0);
    for (int i = 0; i < 5; ++i) {
        for (int k = 0; k < 3; ++k) {
            n[i * 3 + k] = fb(v[i][k]);
        }
    }
    n[15] = sel;
    return n;
}

inline float inv_of(float d) { return d == 0.0f ? INFINITY : 1.0f / d; }

inline Vector make(const std::string& name, uint64_t node, std::vector<uint32_t> mem, const float o[3],
                   const float d[3], const uint32_t exp[4], bool sort = false, float extent = 1000.0f) {
    Vector v;
    v.name = name;
    v.desc[0] = kBase >> 8;
    v.desc[1] = sort ? 0x80000000u : 0u;
    v.desc[3] = 0x80000000u;  // type 8 in bits 31:28
    v.node = node;
    v.extent = extent;
    for (int i = 0; i < 3; ++i) {
        v.origin[i] = o[i];
        v.dir[i] = d[i];
    }
    for (int i = 0; i < 4; ++i) {
        v.expected[i] = exp[i];
    }
    v.memory.push_back({kBase, std::move(mem)});
    return v;
}

inline std::vector<Vector> all() {
    std::vector<Vector> out;
    const float inf = INFINITY;
    const float one[3] = {1, 1, 1};
    const float px[3] = {1, 0, 0};
    const float zero3[3] = {0, 0, 0};
    const float pz[3] = {0, 0, 1};
    const uint32_t P0 = 0x100 | 4, P1 = 0x200 | 4, P2 = 0x300 | 4, P3 = 0x400 | 4;

    // ---- f32 box node (kind 5) at ptr 5 (node 0). Ray along +x from inside child 0.
    // c0 [0,2]^3 (origin inside: enters at t=0), c1 x in [10,12] (entry 9), c2 inverted bounds (never
    // entered), c3 absent.
    {
        const Box c[4] = {{P0, {0, 0, 0}, {2, 2, 2}},
                          {P1, {10, 0, 0}, {12, 2, 2}},
                          {P2, {1, 1, 1}, {0, 0, 0}},
                          {kNone, {0, 0, 0}, {0, 0, 0}}};
        const uint32_t e[4] = {P0, P1, kNone, kNone};
        out.push_back(make("box32 inside + ahead + empty slot + absent", 5, box32(c), one, px, e));
        // Extent 5 clips child 1 (entry at 9).
        const uint32_t e2[4] = {P0, kNone, kNone, kNone};
        out.push_back(make("box32 extent clips far child", 5, box32(c), one, px, e2, false, 5.0f));
        // Ray pointing away from child 1: only child 0 (origin inside it).
        const float nx[3] = {-1, 0, 0};
        out.push_back(make("box32 ray away", 5, box32(c), one, nx, e2));
    }
    // ---- sort: nearest entry first, ties keep child order, misses last.
    {
        const Box c[4] = {{P0, {10, 0, 0}, {12, 2, 2}},   // entry 9
                          {P1, {3, 0, 0}, {4, 2, 2}},     // entry 2
                          {P2, {3, 0, 0}, {5, 2, 2}},     // entry 2 (tie with c1)
                          {P3, {-5, 0, 0}, {-4, 2, 2}}};  // behind: miss
        const uint32_t sorted[4] = {P1, P2, P0, kNone};
        out.push_back(make("box32 sorted nearest first, tie keeps child order", 5, box32(c), one, px, sorted, true));
        const uint32_t unsorted[4] = {P0, P1, P2, kNone};
        out.push_back(make("box32 unsorted keeps child order", 5, box32(c), one, px, unsorted, false));
    }
    // ---- on-face, axis-aligned: origin on the x = 2 face of [0,2]^3, ray along +y (parallel to the
    // face). The x slab is 0 * inf = NaN and constrains nothing: entered.
    {
        const Box c[4] = {{P0, {0, 0, 0}, {2, 2, 2}}, {kNone, {}, {}}, {kNone, {}, {}}, {kNone, {}, {}}};
        const float o[3] = {2, 1, 1};
        const float py[3] = {0, 1, 0};
        const uint32_t e[4] = {P0, kNone, kNone, kNone};
        out.push_back(make("box32 on-face parallel ray enters", 5, box32(c), o, py, e));
        // Touching a face from outside at exactly t = extent is inclusive.
        const float o2[3] = {-3, 1, 1};
        const uint32_t e2[4] = {P0, kNone, kNone, kNone};
        out.push_back(make("box32 entry at exactly extent is inclusive", 5, box32(c), o2, px, e2, false, 3.0f));
        // NaN origin: every child misses.
        const float on[3] = {NAN, 1, 1};
        const uint32_t none[4] = {kNone, kNone, kNone, kNone};
        out.push_back(make("box32 NaN ray misses all", 5, box32(c), on, px, none));
    }
    // ---- f16 box node (kind 4): c0 = [0,2]^3 (0x0000, 0x4000), c1 = [1,2]x[0,2]x[0,2] ... origin 1,1,1.
    {
        std::vector<uint32_t> n(16, 0);
        n[0] = P0;
        n[1] = P1;
        n[2] = kNone;
        n[3] = kNone;
        // c0: min (0,0,0), max (2,2,2): dw4 = min.x|min.y<<16, dw5 = min.z|max.x<<16, dw6 = max.y|max.z<<16
        n[4] = 0;
        n[5] = 0x40000000u;
        n[6] = 0x40004000u;
        // c1: min (4,0,0) (f16 0x4400), max (6,2,2) (f16 0x4600)
        n[7] = 0x4400u;
        n[8] = 0x46000000u;
        n[9] = 0x40004000u;
        const uint32_t e[4] = {P0, P1, kNone, kNone};
        out.push_back(make("box16 two children", 4, n, one, px, e));
    }
    // ---- Triangles, vertices at z = 1, ray from the origin along +z, left-handed.
    // Front face (clockwise seen from the origin): A(-1,-1) B(0,1) C(1,-1). Edge functions
    // U = 1, V = 2, W = 1, det = 4, T = 4 (t = 1). Selector byte 0x09: I weights vertex 1 (V = 2),
    // J weights vertex 2 (W = 1).
    {
        const float v[5][3] = {{-1, -1, 1}, {0, 1, 1}, {1, -1, 1}, {2, 1, 1}, {0, -3, 1}};
        const uint32_t e[4] = {fb(4), fb(4), fb(2), fb(1)};
        out.push_back(make("tri k0 front face", 0, tri_node(v, 0x09), zero3, pz, e));
        // Ray through the shared edge v1-v2 at (0.5, 0): triangle 0 (U = 0, V = 2, W = 2) and
        // triangle 1 = {v1, v3, v2} (U = 2, V = 0, W = 2) both hit at t = 1.
        const float o[3] = {0.5f, 0, 0};
        const uint32_t e0[4] = {fb(4), fb(4), fb(2), fb(2)};
        out.push_back(make("tri k0 shared edge hits", 0, tri_node(v, 0x0909), o, pz, e0));
        const uint32_t e1[4] = {fb(4), fb(4), fb(0), fb(2)};
        out.push_back(make("tri k1 shared edge hits too", 1, tri_node(v, 0x0909), o, pz, e1));
        // Outside the triangle: (+inf, 1, 0, 0).
        const float far[3] = {5, 0, 0};
        const uint32_t miss[4] = {fb(inf), fb(1.0f), 0, 0};
        out.push_back(make("tri k0 miss outside", 0, tri_node(v, 0x09), far, pz, miss));
        // Instance / procedural nodes answer a triangle miss.
        out.push_back(make("kind 6 answers miss", 6, tri_node(v, 0x09), zero3, pz, miss));
        out.push_back(make("kind 7 answers miss", 7, tri_node(v, 0x09), zero3, pz, miss));
        // NaN direction: miss.
        const float dn[3] = {NAN, 0, 1};
        out.push_back(make("tri k0 NaN direction misses", 0, tri_node(v, 0x09), zero3, dn, miss));
        // Zero direction: miss.
        const float d0[3] = {0, 0, 0};
        out.push_back(make("tri k0 zero direction misses", 0, tri_node(v, 0x09), zero3, d0, miss));
    }
    // Back face: B and C swapped: U = -1, V = -1, W = -2, det = -4, T = -4 (t = 1, den negative).
    {
        const float v[5][3] = {{-1, -1, 1}, {1, -1, 1}, {0, 1, 1}, {2, 1, 1}, {0, -3, 1}};
        const uint32_t e[4] = {fb(-4), fb(-4), fb(-1), fb(-2)};
        out.push_back(make("tri k0 back face has negative denominator", 0, tri_node(v, 0x09), zero3, pz, e));
    }
    // Edge-on: the triangle lies in a plane containing the ray (y = 0, ray along +x): det = 0, miss.
    {
        const float v[5][3] = {{1, 0, -1}, {1, 0, 1}, {3, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        const uint32_t miss[4] = {fb(inf), fb(1.0f), 0, 0};
        out.push_back(make("tri k0 edge-on misses", 0, tri_node(v, 0x09), zero3, px, miss));
    }
    // Negative dominant axis: the same triangle at z = -1 traced along -z is seen from behind, so
    // it is a back face. Hand check: kz = 2, d < 0 swaps kx/ky (kx = 1, ky = 0); Sx = Sy = 0,
    // Sz = -1. A = (-1,-1,-1) -> (x,y) = (-1,-1), Az = 1; B -> (1, 0); C -> (-1, 1).
    // U = -1, V = -2, W = -1, det = -4, T = -4; I = V, J = W.
    {
        const float v[5][3] = {{-1, -1, -1}, {0, 1, -1}, {1, -1, -1}, {2, 1, -1}, {0, -3, -1}};
        const float nz[3] = {0, 0, -1};
        const uint32_t back[4] = {fb(-4), fb(-4), fb(-2), fb(-1)};
        out.push_back(make("tri k0 along -z is a back face", 0, tri_node(v, 0x09), zero3, nz, back));
    }
    // ---- 64-bit pointer: node at address 0x80_0000_0000 (bit 39) with descriptor base 0.
    {
        const Box c[4] = {{P0, {0, 0, 0}, {2, 2, 2}}, {kNone, {}, {}}, {kNone, {}, {}}, {kNone, {}, {}}};
        const uint32_t e[4] = {P0, kNone, kNone, kNone};
        Vector v = make("bvh64 pointer reaches bit 39", 0, {}, one, px, e);
        v.memory.clear();
        v.desc[0] = 0;
        v.desc[1] = 0;
        v.node = (0x8000000000ull >> 3) | 5u;  // (addr >> 6) << 3 | kind
        v.memory.push_back({0x8000000000ull, box32(c)});
        out.push_back(std::move(v));
    }
    return out;
}

}  // namespace rt_bvh_vectors
