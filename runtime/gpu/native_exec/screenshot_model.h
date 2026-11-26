// Native executor: screenshots (F12 / --screenshot-at). Pure logic, no GPU / Win32: a small
// self-contained PNG encoder (zlib "stored" blocks, no compression, no dependency), pixel conversion of
// the presented stream image to 8-bit RGB, the --screenshot-at list parser and the output path /
// file name rules. Unit tested in tests/test_screenshot_model.cpp.
//
// Colour handling matches the swapchain blit (blit_model.h, #1739): the stream image is saved as it
// is written to the unorm canvas, without re-encoding. An sRGB stream format is stored in its sRGB
// encoded bytes (what the canvas shows), so the PNG carries an sRGB chunk.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace playgta6::gpu::native_exec {

// ---- PNG encoder ----

inline uint32_t png_crc32(const uint8_t* d, size_t n, uint32_t crc = 0) {
    static const auto* table = [] {
        static uint32_t t[256];
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[i] = c;
        }
        return t;
    }();
    crc = ~crc;
    for (size_t i = 0; i < n; ++i) crc = table[(crc ^ d[i]) & 0xFFu] ^ (crc >> 8);
    return ~crc;
}

inline void png_put_be32(std::vector<uint8_t>& o, uint32_t v) {
    o.push_back(static_cast<uint8_t>(v >> 24));
    o.push_back(static_cast<uint8_t>(v >> 16));
    o.push_back(static_cast<uint8_t>(v >> 8));
    o.push_back(static_cast<uint8_t>(v));
}

inline void png_chunk(std::vector<uint8_t>& o, const char type[4], const std::vector<uint8_t>& data) {
    png_put_be32(o, static_cast<uint32_t>(data.size()));
    const size_t start = o.size();
    o.insert(o.end(), type, type + 4);
    o.insert(o.end(), data.begin(), data.end());
    png_put_be32(o, png_crc32(o.data() + start, o.size() - start));
}

// Encodes `rgb` (width * height * 3 bytes, top row first) as a PNG (8-bit truecolour, sRGB chunk).
// Empty on bad arguments. The IDAT payload is a zlib stream of stored (uncompressed) blocks.
inline std::vector<uint8_t> png_encode_rgb(const uint8_t* rgb, uint32_t width, uint32_t height) {
    if (!rgb || width == 0 || height == 0) return {};
    const size_t row = size_t{width} * 3;
    const size_t raw_size = (row + 1) * height;
    std::vector<uint8_t> raw;
    raw.reserve(raw_size);
    for (uint32_t y = 0; y < height; ++y) {
        raw.push_back(0);  // filter: none
        raw.insert(raw.end(), rgb + row * y, rgb + row * (y + 1));
    }
    std::vector<uint8_t> z;
    z.reserve(raw_size + raw_size / 65535 * 5 + 16);
    z.push_back(0x78);
    z.push_back(0x01);
    size_t pos = 0;
    do {
        const size_t n = std::min<size_t>(65535, raw_size - pos);
        const bool last = pos + n >= raw_size;
        z.push_back(last ? 1 : 0);
        z.push_back(static_cast<uint8_t>(n & 0xFF));
        z.push_back(static_cast<uint8_t>(n >> 8));
        z.push_back(static_cast<uint8_t>(~n & 0xFF));
        z.push_back(static_cast<uint8_t>((~n >> 8) & 0xFF));
        z.insert(z.end(), raw.begin() + static_cast<std::ptrdiff_t>(pos),
                 raw.begin() + static_cast<std::ptrdiff_t>(pos + n));
        pos += n;
    } while (pos < raw_size);
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < raw_size; ++i) {
        a = (a + raw[i]) % 65521u;
        b = (b + a) % 65521u;
    }
    png_put_be32(z, (b << 16) | a);

    std::vector<uint8_t> out = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    std::vector<uint8_t> ihdr;
    png_put_be32(ihdr, width);
    png_put_be32(ihdr, height);
    ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});  // 8 bit, truecolour, deflate, no filter method, no interlace
    png_chunk(out, "IHDR", ihdr);
    png_chunk(out, "sRGB", {0});  // rendering intent: perceptual
    png_chunk(out, "IDAT", z);
    png_chunk(out, "IEND", {});
    return out;
}

// ---- pixel conversion ----

// The stream formats a screenshot can convert (the 4-byte / 8-byte colour formats a swapchain uses).
enum class ShotFormat { Unsupported, Rgba8, Bgra8, Rgb10a2, Rgba16Float };

inline size_t shot_format_bytes(ShotFormat f) {
    switch (f) {
        case ShotFormat::Rgba8:
        case ShotFormat::Bgra8:
        case ShotFormat::Rgb10a2: return 4;
        case ShotFormat::Rgba16Float: return 8;
        default: return 0;
    }
}

inline float shot_half_to_float(uint16_t h) {
    const uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    float v;
    if (e == 0) v = std::ldexp(static_cast<float>(m), -24);
    else if (e == 31) v = m ? 0.0f : 65504.0f;  // NaN -> 0, inf -> clamped later
    else v = std::ldexp(static_cast<float>(m | 0x400u), static_cast<int>(e) - 25);
    return (h & 0x8000u) ? -v : v;
}

inline uint8_t shot_unorm8(float v) {
    if (!(v > 0.0f)) return 0;
    if (v >= 1.0f) return 255;
    return static_cast<uint8_t>(v * 255.0f + 0.5f);
}

// Converts a padded GPU read-back (`src`, rows `padded_row` bytes apart) to tightly packed RGB8.
// Alpha is dropped (the canvas is opaque). Empty for an unsupported format.
inline std::vector<uint8_t> shot_to_rgb8(const uint8_t* src, size_t padded_row, uint32_t w, uint32_t h, ShotFormat f) {
    const size_t bpp = shot_format_bytes(f);
    if (!src || bpp == 0 || padded_row < size_t{w} * bpp) return {};
    std::vector<uint8_t> out(size_t{w} * h * 3);
    for (uint32_t y = 0; y < h; ++y) {
        const uint8_t* s = src + padded_row * y;
        uint8_t* d = out.data() + size_t{w} * 3 * y;
        for (uint32_t x = 0; x < w; ++x, s += bpp, d += 3) {
            switch (f) {
                case ShotFormat::Rgba8: d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; break;
                case ShotFormat::Bgra8: d[0] = s[2]; d[1] = s[1]; d[2] = s[0]; break;
                case ShotFormat::Rgb10a2: {
                    uint32_t v;
                    std::memcpy(&v, s, 4);
                    d[0] = shot_unorm8(static_cast<float>(v & 0x3FFu) / 1023.0f);
                    d[1] = shot_unorm8(static_cast<float>((v >> 10) & 0x3FFu) / 1023.0f);
                    d[2] = shot_unorm8(static_cast<float>((v >> 20) & 0x3FFu) / 1023.0f);
                    break;
                }
                case ShotFormat::Rgba16Float:
                    for (int c = 0; c < 3; ++c) {
                        uint16_t hv;
                        std::memcpy(&hv, s + 2 * c, 2);
                        d[c] = shot_unorm8(shot_half_to_float(hv));
                    }
                    break;
                default: break;
            }
        }
    }
    return out;
}

// ---- --screenshot-at, paths ----

// "10,50,300" -> sorted unique present indices (0-based, counting presents like the digest). False on junk.
inline bool parse_screenshot_at(const std::string& text, std::vector<uint64_t>* out, std::string* error) {
    std::vector<uint64_t> res;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t end = text.find(',', pos);
        if (end == std::string::npos) end = text.size();
        const std::string item = text.substr(pos, end - pos);
        pos = end + 1;
        if (item.empty()) continue;
        char* tail = nullptr;
        const unsigned long long v = std::strtoull(item.c_str(), &tail, 10);
        if (item[0] == '-' || *tail != '\0') {
            if (error) *error = "--screenshot-at: bad present index '" + item + "'";
            return false;
        }
        res.push_back(v);
    }
    std::sort(res.begin(), res.end());
    res.erase(std::unique(res.begin(), res.end()), res.end());
    *out = std::move(res);
    return true;
}

// %LOCALAPPDATA%\playgta6\screenshots (Windows), $XDG_DATA_HOME / ~/.local/share elsewhere. Empty if unknown.
inline std::filesystem::path default_screenshot_dir() {
    namespace fs = std::filesystem;
    if (const char* la = std::getenv("LOCALAPPDATA"); la && *la) return fs::path(la) / "playgta6" / "screenshots";
    if (const char* x = std::getenv("XDG_DATA_HOME"); x && *x) return fs::path(x) / "playgta6" / "screenshots";
    if (const char* h = std::getenv("HOME"); h && *h) return fs::path(h) / ".local" / "share" / "playgta6" / "screenshots";
    return {};
}

// "20260102-153045-123" in local time.
inline std::string screenshot_timestamp(std::chrono::system_clock::time_point tp) {
    const std::time_t t = std::chrono::system_clock::to_time_t(tp);
    const int ms = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch()).count() % 1000);
    std::tm tmv{};
#ifdef _WIN32
    localtime_s(&tmv, &t);
#else
    localtime_r(&t, &tmv);
#endif
    char buf[40];
    std::snprintf(buf, sizeof buf, "%04d%02d%02d-%02d%02d%02d-%03d", tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
                  tmv.tm_hour, tmv.tm_min, tmv.tm_sec, ms < 0 ? 0 : ms);
    return buf;
}

// dir/<timestamp>.png, with -2, -3 ... appended while the name exists.
inline std::filesystem::path screenshot_unique_path(const std::filesystem::path& dir, const std::string& stamp) {
    namespace fs = std::filesystem;
    fs::path p = dir / (stamp + ".png");
    std::error_code ec;
    for (int n = 2; fs::exists(p, ec) && n < 1000; ++n) p = dir / (stamp + "-" + std::to_string(n) + ".png");
    return p;
}

}  // namespace playgta6::gpu::native_exec
