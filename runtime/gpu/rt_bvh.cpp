#include "gpu/rt_bvh.h"

#include <cmath>
#include <cstring>

namespace playgta6::gpu::rt_bvh {

namespace {

float bits_f(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
uint32_t f_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
bool isnan_(float f) { return f != f; }

// Exact binary16 -> binary32 (what unpack2x16float does).
float half_f(uint32_t h) {
    const uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    uint32_t out;
    if (e == 0) {
        if (m == 0) {
            out = s << 31;
        } else {  // subnormal: normalise
            uint32_t ee = 127 - 15 + 1;
            uint32_t mm = m;
            while ((mm & 0x400u) == 0) {
                mm <<= 1;
                --ee;
            }
            out = (s << 31) | (ee << 23) | ((mm & 0x3ffu) << 13);
        }
    } else if (e == 31) {
        out = (s << 31) | 0x7f800000u | (m << 13);
    } else {
        out = (s << 31) | ((e + 127 - 15) << 23) | (m << 13);
    }
    return bits_f(out);
}

// Slab test; true when entered. *near_t = entry distance (0 when the ray starts inside).
bool slab(const float mn[3], const float mx[3], const float o[3], const float inv[3], float extent, float* near_t) {
    float lo = 0.0f, hi = extent;
    for (int a = 0; a < 3; ++a) {
        if (mn[a] > mx[a]) {
            return false;  // empty slot
        }
        const float t0 = (mn[a] - o[a]) * inv[a];
        const float t1 = (mx[a] - o[a]) * inv[a];
        if (isnan_(t0) || isnan_(t1)) {
            continue;
        }
        const float a0 = t0 < t1 ? t0 : t1;
        const float a1 = t0 < t1 ? t1 : t0;
        if (a0 > lo) lo = a0;
        if (a1 < hi) hi = a1;
    }
    *near_t = lo;
    return lo <= hi;
}

Result miss_triangle() { return {{f_bits(INFINITY), f_bits(1.0f), 0u, 0u}}; }

Result triangle(const uint32_t n[16], uint32_t kind, const float o[3], const float d[3]) {
    static const uint32_t kVerts[4][3] = {{0, 1, 2}, {1, 3, 2}, {2, 3, 4}, {2, 4, 0}};
    float v[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int c = 0; c < 3; ++c) {
            v[i][c] = bits_f(n[kVerts[kind][i] * 3 + c]) - o[c];
        }
    }
    int kz = 0;
    float az = std::fabs(d[0]);
    if (std::fabs(d[1]) > az) {
        kz = 1;
        az = std::fabs(d[1]);
    }
    if (std::fabs(d[2]) > az) {
        kz = 2;
    }
    if (d[kz] == 0.0f) {
        return miss_triangle();
    }
    int kx = (kz + 1) % 3, ky = (kx + 1) % 3;
    if (d[kz] < 0.0f) {
        const int t = kx;
        kx = ky;
        ky = t;
    }
    const float sx = d[kx] / d[kz], sy = d[ky] / d[kz], sz = 1.0f / d[kz];
    float px[3], py[3], pz[3];
    for (int i = 0; i < 3; ++i) {
        px[i] = v[i][kx] - sx * v[i][kz];
        py[i] = v[i][ky] - sy * v[i][kz];
        pz[i] = sz * v[i][kz];
    }
    const float u = px[2] * py[1] - py[2] * px[1];
    const float vv = px[0] * py[2] - py[0] * px[2];
    const float w = px[1] * py[0] - py[1] * px[0];
    if ((u < 0.0f || vv < 0.0f || w < 0.0f) && (u > 0.0f || vv > 0.0f || w > 0.0f)) {
        return miss_triangle();
    }
    const float det = u + vv + w;
    const float t_num = u * pz[0] + vv * pz[1] + w * pz[2];
    if (det == 0.0f || isnan_(det) || isnan_(t_num) || isnan_(u) || isnan_(vv) || isnan_(w)) {
        return miss_triangle();
    }
    // dword 15: one byte per triangle; bits 1:0 the vertex weighted by I, bits 3:2 the one by J
    // (an index 3 selects nothing: weight 0).
    const uint32_t sel = (n[15] >> (8 * kind)) & 0xffu;
    const float wt[4] = {u, vv, w, 0.0f};
    return {{f_bits(t_num), f_bits(det), f_bits(wt[sel & 3u]), f_bits(wt[(sel >> 2) & 3u])}};
}

}  // namespace

Result intersect(const uint32_t desc[4], uint64_t node, float extent, Vec3 origin, Vec3 dir, Vec3 inv_dir,
                 const LoadU32& load) {
    const uint32_t kind = static_cast<uint32_t>(node & 7u);
    const float o[3] = {origin.x, origin.y, origin.z};
    const float d[3] = {dir.x, dir.y, dir.z};
    const float inv[3] = {inv_dir.x, inv_dir.y, inv_dir.z};
    const bool nan_ray = isnan_(extent) || isnan_(o[0]) || isnan_(o[1]) || isnan_(o[2]) || isnan_(d[0]) ||
                         isnan_(d[1]) || isnan_(d[2]) || isnan_(inv[0]) || isnan_(inv[1]) || isnan_(inv[2]);
    const uint64_t base = ((static_cast<uint64_t>(desc[1] & 0xffu) << 32) | desc[0]) << 8;
    const uint64_t addr = (base + ((node >> 3) << 6)) & kAddressMask;
    const bool sort = (desc[1] >> 31) != 0;

    if (kind >= 6) {
        return miss_triangle();
    }
    if (kind < 4) {
        if (nan_ray) {
            return miss_triangle();
        }
        uint32_t n[16];
        for (uint32_t i = 0; i < 16; ++i) {
            n[i] = load((addr + 4ull * i) & kAddressMask);
        }
        return triangle(n, kind, o, d);
    }
    Result out = {{kNoChild, kNoChild, kNoChild, kNoChild}};
    if (nan_ray) {
        return out;
    }
    uint32_t ptr[4];
    float key[4];
    bool hit[4];
    for (uint32_t i = 0; i < 4; ++i) {
        ptr[i] = load((addr + 4ull * i) & kAddressMask);
        float mn[3], mx[3];
        if (kind == 5) {
            for (uint32_t c = 0; c < 3; ++c) {
                mn[c] = bits_f(load((addr + 16 + 24ull * i + 4ull * c) & kAddressMask));
                mx[c] = bits_f(load((addr + 16 + 24ull * i + 12 + 4ull * c) & kAddressMask));
            }
        } else {
            const uint32_t a = load((addr + 16 + 12ull * i) & kAddressMask);
            const uint32_t b = load((addr + 16 + 12ull * i + 4) & kAddressMask);
            const uint32_t c = load((addr + 16 + 12ull * i + 8) & kAddressMask);
            mn[0] = half_f(a & 0xffffu);
            mn[1] = half_f(a >> 16);
            mn[2] = half_f(b & 0xffffu);
            mx[0] = half_f(b >> 16);
            mx[1] = half_f(c & 0xffffu);
            mx[2] = half_f(c >> 16);
        }
        key[i] = 0.0f;
        hit[i] = ptr[i] != kNoChild && slab(mn, mx, o, inv, extent, &key[i]);
    }
    for (uint32_t i = 0; i < 4; ++i) {
        if (!hit[i]) continue;
        uint32_t rank = i;
        if (sort) {
            rank = 0;
            for (uint32_t j = 0; j < 4; ++j) {
                if (hit[j] && (key[j] < key[i] || (key[j] == key[i] && j < i))) ++rank;
            }
        }
        out.w[rank] = ptr[i];
    }
    return out;
}

const char* prelude_wgsl() {
    return R"WGSL(// pg6 runtime prelude, version 1 (rt_bvh.h kPreludeVersion). Group 3, bindings 16-21: contract C3.
@group(3) @binding(16) var<storage, read_write> pg6_arena0: array<u32>;
@group(3) @binding(17) var<storage, read_write> pg6_arena1: array<u32>;
@group(3) @binding(18) var<storage, read_write> pg6_arena2: array<u32>;
@group(3) @binding(19) var<storage, read_write> pg6_arena3: array<u32>;
@group(3) @binding(20) var<storage, read> pg6_arena_dir: array<u32>;
@group(3) @binding(21) var<storage, read_write> pg6_rt_stats: array<atomic<u32>>;

// Dword at a 40-bit address (x = low 32 bits, y = bits 39:32). Unmapped: counts a fault, reads 0.
fn pg6_load32(addr: vec2<u32>) -> u32 {
    let page = (addr.x >> 16u) | ((addr.y & 0xffu) << 16u);
    let t = pg6_arena_dir[page >> 12u];
    if (t == 0u) {
        atomicAdd(&pg6_rt_stats[0], 1u);
        return 0u;
    }
    let e = pg6_arena_dir[4096u + (t - 1u) * 4096u + (page & 0xfffu)];
    if ((e & 0x80000000u) == 0u) {
        atomicAdd(&pg6_rt_stats[0], 1u);
        return 0u;
    }
    let w = (e & 0x1fffffffu) * 16384u + ((addr.x & 0xffffu) >> 2u);
    switch ((e >> 29u) & 3u) {
        case 0u: { return pg6_arena0[w]; }
        case 1u: { return pg6_arena1[w]; }
        case 2u: { return pg6_arena2[w]; }
        default: { return pg6_arena3[w]; }
    }
}

fn pg6_add40(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    var carry = 0u;
    if (lo < a.x) { carry = 1u; }
    return vec2<u32>(lo, (a.y + b.y + carry) & 0xffu);
}

fn pg6_addr_off(a: vec2<u32>, off: u32) -> vec2<u32> {
    return pg6_add40(a, vec2<u32>(off, 0u));
}

fn pg6_f(a: vec2<u32>, off: u32) -> f32 {
    return bitcast<f32>(pg6_load32(pg6_addr_off(a, off)));
}

// Slab test of one box: .x = 1.0 when entered, .y = the entry distance.
fn pg6_slab(mn: vec3<f32>, mx: vec3<f32>, o: vec3<f32>, inv: vec3<f32>, extent: f32) -> vec2<f32> {
    var lo = 0.0;
    var hi = extent;
    for (var a = 0u; a < 3u; a = a + 1u) {
        if (mn[a] > mx[a]) {
            return vec2<f32>(0.0, 0.0);
        }
        let t0 = (mn[a] - o[a]) * inv[a];
        let t1 = (mx[a] - o[a]) * inv[a];
        if (t0 != t0 || t1 != t1) {
            continue;
        }
        var a0 = t1;
        var a1 = t0;
        if (t0 < t1) {
            a0 = t0;
            a1 = t1;
        }
        if (a0 > lo) { lo = a0; }
        if (a1 < hi) { hi = a1; }
    }
    if (lo <= hi) {
        return vec2<f32>(1.0, lo);
    }
    return vec2<f32>(0.0, 0.0);
}

fn pg6_tri_miss() -> vec4<u32> {
    return vec4<u32>(0x7f800000u, 0x3f800000u, 0u, 0u);
}

fn pg6_vec_nan(v: vec3<f32>) -> bool {
    return v.x != v.x || v.y != v.y || v.z != v.z;
}

fn pg6_triangle(addr: vec2<u32>, kind: u32, o: vec3<f32>, d: vec3<f32>) -> vec4<u32> {
    var vi = vec3<u32>(0u, 1u, 2u);
    if (kind == 1u) { vi = vec3<u32>(1u, 3u, 2u); }
    if (kind == 2u) { vi = vec3<u32>(2u, 3u, 4u); }
    if (kind == 3u) { vi = vec3<u32>(2u, 4u, 0u); }
    var v: array<vec3<f32>, 3>;
    for (var i = 0u; i < 3u; i = i + 1u) {
        let b = vi[i] * 12u;
        v[i] = vec3<f32>(pg6_f(addr, b), pg6_f(addr, b + 4u), pg6_f(addr, b + 8u)) - o;
    }
    var kz = 0u;
    var az = abs(d.x);
    if (abs(d.y) > az) { kz = 1u; az = abs(d.y); }
    if (abs(d.z) > az) { kz = 2u; }
    let dz = d[kz];
    if (dz == 0.0) {
        return pg6_tri_miss();
    }
    var kx = (kz + 1u) % 3u;
    var ky = (kx + 1u) % 3u;
    if (dz < 0.0) {
        let t = kx;
        kx = ky;
        ky = t;
    }
    let sx = d[kx] / dz;
    let sy = d[ky] / dz;
    let sz = 1.0 / dz;
    var px: array<f32, 3>;
    var py: array<f32, 3>;
    var pz: array<f32, 3>;
    for (var i = 0u; i < 3u; i = i + 1u) {
        px[i] = v[i][kx] - sx * v[i][kz];
        py[i] = v[i][ky] - sy * v[i][kz];
        pz[i] = sz * v[i][kz];
    }
    let u = px[2] * py[1] - py[2] * px[1];
    let vv = px[0] * py[2] - py[0] * px[2];
    let w = px[1] * py[0] - py[1] * px[0];
    if ((u < 0.0 || vv < 0.0 || w < 0.0) && (u > 0.0 || vv > 0.0 || w > 0.0)) {
        return pg6_tri_miss();
    }
    let det = u + vv + w;
    let t_num = u * pz[0] + vv * pz[1] + w * pz[2];
    if (det == 0.0 || det != det || t_num != t_num || u != u || vv != vv || w != w) {
        return pg6_tri_miss();
    }
    let sel = (pg6_load32(pg6_addr_off(addr, 60u)) >> (8u * kind)) & 0xffu;
    var wt = array<f32, 4>(u, vv, w, 0.0);
    return vec4<u32>(bitcast<u32>(t_num), bitcast<u32>(det), bitcast<u32>(wt[sel & 3u]),
                     bitcast<u32>(wt[(sel >> 2u) & 3u]));
}

// C4: the console's image_bvh(64)_intersect_ray node test. desc = the 128-bit BVH descriptor,
// node = the node pointer (x low, y high; high 0 for the 32-bit form).
fn pg6_bvh_intersect(desc: vec4<u32>, node: vec2<u32>, extent: f32, origin: vec3<f32>, dir: vec3<f32>,
                     inv_dir: vec3<f32>) -> vec4<u32> {
    let kind = node.x & 7u;
    if (kind >= 6u) {
        return pg6_tri_miss();
    }
    let nan_ray = extent != extent || pg6_vec_nan(origin) || pg6_vec_nan(dir) || pg6_vec_nan(inv_dir);
    // address = (base256 << 8) + ((ptr >> 3) << 6), 40 bits
    let base = vec2<u32>(desc.x << 8u, ((desc.y & 0xffu) << 8u) | (desc.x >> 24u));
    let idx = vec2<u32>((node.x >> 3u) | (node.y << 29u), node.y >> 3u);
    let off = vec2<u32>(idx.x << 6u, (idx.y << 6u) | (idx.x >> 26u));
    let addr = pg6_add40(base, off);
    if (kind < 4u) {
        if (nan_ray) {
            return pg6_tri_miss();
        }
        return pg6_triangle(addr, kind, origin, dir);
    }
    if (nan_ray) {
        return vec4<u32>(0xffffffffu);
    }
    var ptr: array<u32, 4>;
    var key: array<f32, 4>;
    var hit: array<bool, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        ptr[i] = pg6_load32(pg6_addr_off(addr, 4u * i));
        var mn: vec3<f32>;
        var mx: vec3<f32>;
        if (kind == 5u) {
            let b = 16u + 24u * i;
            mn = vec3<f32>(pg6_f(addr, b), pg6_f(addr, b + 4u), pg6_f(addr, b + 8u));
            mx = vec3<f32>(pg6_f(addr, b + 12u), pg6_f(addr, b + 16u), pg6_f(addr, b + 20u));
        } else {
            let b = 16u + 12u * i;
            let a = unpack2x16float(pg6_load32(pg6_addr_off(addr, b)));
            let bb = unpack2x16float(pg6_load32(pg6_addr_off(addr, b + 4u)));
            let c = unpack2x16float(pg6_load32(pg6_addr_off(addr, b + 8u)));
            mn = vec3<f32>(a.x, a.y, bb.x);
            mx = vec3<f32>(bb.y, c.x, c.y);
        }
        key[i] = 0.0;
        hit[i] = false;
        if (ptr[i] != 0xffffffffu) {
            let s = pg6_slab(mn, mx, origin, inv_dir, extent);
            hit[i] = s.x > 0.5;
            key[i] = s.y;
        }
    }
    var out = vec4<u32>(0xffffffffu);
    let sort = (desc.y >> 31u) != 0u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        if (!hit[i]) {
            continue;
        }
        var rank = i;
        if (sort) {
            rank = 0u;
            for (var j = 0u; j < 4u; j = j + 1u) {
                if (hit[j] && (key[j] < key[i] || (key[j] == key[i] && j < i))) {
                    rank = rank + 1u;
                }
            }
        }
        out[rank] = ptr[i];
    }
    return out;
}
)WGSL";
}

}  // namespace playgta6::gpu::rt_bvh
