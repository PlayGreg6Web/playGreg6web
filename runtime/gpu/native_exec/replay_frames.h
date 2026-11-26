// Native executor: the replay side of tools/replay/replay.html, so a .pg6trace replayed natively
// (pg6_native_exec) and in Chrome (tools/replay_capture.py) is cut into the same frames and its
// swapchain reported with the same numbers: splitFrames, the full-resolution `digest frame` hash and
// the sampled `swapchain frame ... non-black ... distinct` line. No WebGPU types (unit-tested on
// every host build).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

#include "abi/playgta6_abi.h"

namespace playgta6::gpu::native_exec {

inline constexpr uint32_t kThinMarker = 0x6e696874u;  // 'thin' (gpu/frame_capture.h kThinUploadMarker)

// replay.html splitFrames: frame i = the packets after Present i-1 up to and including Present i;
// packets after the last Present form a final frame. Thinned uploads (WriteBuffer / WriteTexture
// with data_len 0 and reserved == 'thin') are dropped: the replay keeps the previous contents.
inline std::vector<std::vector<uint8_t>> split_replay_frames(const uint8_t* bytes, size_t len) {
    std::vector<std::vector<uint8_t>> frames;
    std::vector<uint8_t> parts;
    size_t start = 0, off = 0;
    auto u16 = [&](size_t at) {
        uint16_t v;
        std::memcpy(&v, bytes + at, 2);
        return v;
    };
    auto u32 = [&](size_t at) {
        uint32_t v;
        std::memcpy(&v, bytes + at, 4);
        return v;
    };
    auto push = [&](size_t from, size_t to) {
        if (to > len) to = len;  // replay.html's subarray clamps a truncated tail the same way
        if (to > from) parts.insert(parts.end(), bytes + from, bytes + to);
    };
    while (off + 8 <= len) {
        const size_t at = off;
        const uint16_t op = u16(off);
        const uint32_t ln = u32(off + 4);
        off += 8 + size_t{ln};
        uint32_t fixed = 0, data_len_at = 0;
        if (op == static_cast<uint16_t>(GpuOpcode::WriteBuffer)) fixed = 16, data_len_at = 8;
        if (op == static_cast<uint16_t>(GpuOpcode::WriteTexture)) fixed = 48, data_len_at = 40;
        if (fixed && ln == fixed && at + 8 + fixed <= len && u32(at + 8 + data_len_at) == 0 &&
            u32(at + 8 + fixed - 4) == kThinMarker) {
            if (at > start) push(start, at);
            start = off;
        }
        if (op == static_cast<uint16_t>(GpuOpcode::Present)) {
            if (off > start) push(start, off);
            frames.push_back(std::move(parts));
            parts.clear();
            start = off;
        }
    }
    if (start < len) push(start, len);
    if (!parts.empty()) frames.push_back(std::move(parts));
    return frames;
}

// replay.html `digest frame N:`: two 32-bit FNV-style lanes over every pixel as a little-endian u32
// (rows tightly packed, 4 bytes per pixel), printed as 16 hex digits.
inline std::string replay_digest(const uint8_t* px, uint32_t width, uint32_t height) {
    uint32_t h1 = 0x811c9dc5u, h2 = 0x01000193u;
    for (uint32_t y = 0; y < height; ++y) {
        const uint8_t* row = px + size_t{width} * 4 * y;
        for (uint32_t x = 0; x < width; ++x) {
            uint32_t v;
            std::memcpy(&v, row + size_t{x} * 4, 4);
            h1 = (h1 ^ v) * 0x01000193u;
            h2 = (h2 ^ ((v >> 16) | (v << 16))) * 0x85ebca6bu;
        }
    }
    char buf[17];
    std::snprintf(buf, sizeof buf, "%08x%08x", h1, h2);
    return buf;
}

struct SampledStats {
    uint64_t non_black = 0;
    uint64_t total = 0;
    uint64_t distinct = 0;
};

// replay.html: every 4th pixel of every 4th row; the key is the first three bytes (alpha ignored).
inline SampledStats replay_sampled_stats(const uint8_t* px, uint32_t width, uint32_t height) {
    SampledStats s;
    std::unordered_set<uint32_t> hist;
    for (uint32_t y = 0; y < height; y += 4) {
        for (uint32_t x = 0; x < width; x += 4) {
            const uint8_t* p = px + (size_t{y} * width + x) * 4;
            const uint32_t k = uint32_t{p[0]} | uint32_t{p[1]} << 8 | uint32_t{p[2]} << 16;
            ++s.total;
            if (k) ++s.non_black;
            hist.insert(k);
        }
    }
    s.distinct = hist.size();
    return s;
}

}  // namespace playgta6::gpu::native_exec
