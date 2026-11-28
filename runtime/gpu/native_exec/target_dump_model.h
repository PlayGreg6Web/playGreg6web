// Native executor: --dump-targets-at (dump every render / storage target of a frame). Pure logic, no GPU:
// the output size, the half-float read-back conversion, per-target value statistics (to find the first stage
// of a frame that is black without opening any image), the auto-scaled PNG pixels and the index line.
// The executor draws each target with the --show-target display shader in raw mode (show_target_model.h,
// mode.y = 1: the texel as it is, no range mapping) into an rgba16float texture of target_dump_extent() and
// reads it back. Unit tested in tests/test_native_exec_model.cpp; the GPU path in tests/test_native_executor.cpp.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace playgta6::gpu::native_exec {

// The dump's size: the target's size scaled down (nearest, aspect kept) so the larger side is at most max_dim.
struct DumpExtent {
    uint32_t w = 1, h = 1;
};
inline DumpExtent target_dump_extent(uint32_t w, uint32_t h, uint32_t max_dim) {
    DumpExtent d{std::max<uint32_t>(1, w), std::max<uint32_t>(1, h)};
    if (max_dim == 0) return d;
    const uint32_t big = std::max(d.w, d.h);
    if (big <= max_dim) return d;
    d.w = std::max<uint32_t>(1, static_cast<uint32_t>(uint64_t{d.w} * max_dim / big));
    d.h = std::max<uint32_t>(1, static_cast<uint32_t>(uint64_t{d.h} * max_dim / big));
    return d;
}

// IEEE half to float, keeping NaN and infinity (shot_half_to_float maps them for display).
inline float dump_half_to_float(uint16_t h) {
    const uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) {
            bits = s;
        } else {  // subnormal
            const float v = std::ldexp(static_cast<float>(m), -24);
            return s ? -v : v;
        }
    } else if (e == 31) {
        bits = s | 0x7F800000u | (m << 13);
    } else {
        bits = s | ((e + 112u) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// rgba16float read-back (rows padded_row bytes apart) to tightly packed float rgba.
inline std::vector<float> dump_rgba16f_to_float(const uint8_t* src, size_t padded_row, uint32_t w, uint32_t h) {
    std::vector<float> out;
    if (!src || padded_row < size_t{w} * 8) return out;
    out.resize(size_t{w} * h * 4);
    for (uint32_t y = 0; y < h; ++y) {
        const uint8_t* s = src + padded_row * y;
        float* d = out.data() + size_t{w} * 4 * y;
        for (uint32_t i = 0; i < w * 4; ++i) {
            uint16_t hv;
            std::memcpy(&hv, s + 2 * i, 2);
            d[i] = dump_half_to_float(hv);
        }
    }
    return out;
}

// Value statistics of one dumped target (rgba). NaN / inf texels do not enter min / max / mean.
struct TargetDumpStats {
    float min[4] = {0, 0, 0, 0}, max[4] = {0, 0, 0, 0};
    double mean[4] = {0, 0, 0, 0};
    uint64_t texels = 0;
    uint64_t nonzero_rgb = 0;  // texels whose rgb is not (0, 0, 0)
    uint64_t nan = 0, inf = 0;  // texels with any NaN / infinite channel
    float max_rgb = 0;          // the largest finite rgb channel value (the PNG's scale)
    bool black() const { return nonzero_rgb == 0; }
    bool flat() const {  // every finite texel the same (a cleared target)
        for (int c = 0; c < 4; ++c)
            if (min[c] != max[c]) return false;
        return true;
    }
};

inline TargetDumpStats target_dump_stats(const float* rgba, size_t texels) {
    TargetDumpStats s;
    s.texels = texels;
    double sum[4] = {0, 0, 0, 0};
    uint64_t finite = 0;
    bool first = true;
    for (size_t i = 0; i < texels; ++i) {
        const float* p = rgba + i * 4;
        bool is_nan = false, is_inf = false;
        for (int c = 0; c < 4; ++c) {
            if (std::isnan(p[c])) is_nan = true;
            else if (std::isinf(p[c])) is_inf = true;
        }
        if (is_nan) ++s.nan;
        if (is_inf) ++s.inf;
        if (is_nan || is_inf) continue;
        ++finite;
        for (int c = 0; c < 4; ++c) {
            if (first || p[c] < s.min[c]) s.min[c] = p[c];
            if (first || p[c] > s.max[c]) s.max[c] = p[c];
            sum[c] += p[c];
        }
        first = false;
        if (p[0] != 0.0f || p[1] != 0.0f || p[2] != 0.0f) ++s.nonzero_rgb;
        s.max_rgb = std::max({s.max_rgb, p[0], p[1], p[2]});
    }
    for (int c = 0; c < 4; ++c) s.mean[c] = finite ? sum[c] / static_cast<double>(finite) : 0.0;
    return s;
}

// PNG pixels: rgb scaled by 1 / max_rgb (auto exposure, so a dim HDR or a tiny-valued target is still
// visible; the index line names the scale), negative / NaN -> 0. A black target stays black.
inline std::vector<uint8_t> target_dump_rgb8(const float* rgba, size_t texels, float max_rgb) {
    std::vector<uint8_t> out(texels * 3);
    const float k = max_rgb > 0.0f ? 1.0f / max_rgb : 0.0f;
    for (size_t i = 0; i < texels; ++i)
        for (int c = 0; c < 3; ++c) {
            const float v = rgba[i * 4 + c] * k;
            out[i * 3 + c] = !(v > 0.0f) ? 0 : v >= 1.0f ? 255 : static_cast<uint8_t>(v * 255.0f + 0.5f);
        }
    return out;
}

// One index line (index.txt and the log): what the target is, its values, the work of its pass.
inline std::string target_dump_stats_text(const TargetDumpStats& s) {
    char buf[320];
    std::snprintf(buf, sizeof buf,
                  "%s nonzero %.1f%% rgb max %.4g mean %.4g %.4g %.4g alpha %.4g..%.4g nan %llu inf %llu",
                  s.black() ? "BLACK" : s.flat() ? "FLAT" : "ok",
                  s.texels ? 100.0 * static_cast<double>(s.nonzero_rgb) / static_cast<double>(s.texels) : 0.0,
                  static_cast<double>(s.max_rgb), s.mean[0], s.mean[1], s.mean[2], static_cast<double>(s.min[3]),
                  static_cast<double>(s.max[3]), static_cast<unsigned long long>(s.nan),
                  static_cast<unsigned long long>(s.inf));
    return buf;
}

// "pass-007-tex-1234-depth" (no extension): the dump's file stem; kind is "", "depth" or "storage".
inline std::string target_dump_stem(uint32_t pass, uint32_t id, const char* kind, uint32_t slot) {
    char buf[96];
    std::snprintf(buf, sizeof buf, "pass-%03u-tex-%u%s%s", pass, id, kind && *kind ? "-" : "", kind ? kind : "");
    std::string s = buf;
    if (slot) s += "-c" + std::to_string(slot);
    return s;
}

}  // namespace playgta6::gpu::native_exec
