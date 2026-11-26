// Test helper: decodes the PNGs screenshot_model.h writes (8-bit truecolour, filter 0, stored zlib blocks).
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

#include "gpu/native_exec/screenshot_model.h"

inline uint32_t png_test_be32(const uint8_t* p) {
    return (uint32_t{p[0]} << 24) | (uint32_t{p[1]} << 16) | (uint32_t{p[2]} << 8) | p[3];
}

// Verifies signature, chunk CRCs, zlib header / stored blocks / adler32. False on any mismatch.
inline bool png_test_decode_rgb(const std::vector<uint8_t>& png, uint32_t* w, uint32_t* h, std::vector<uint8_t>* rgb) {
    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    if (png.size() < 8 || std::memcmp(png.data(), sig, 8) != 0) return false;
    size_t pos = 8;
    std::vector<uint8_t> idat;
    bool have_ihdr = false, have_iend = false;
    while (pos + 12 <= png.size()) {
        const uint32_t len = png_test_be32(&png[pos]);
        if (pos + 12 + len > png.size()) return false;
        const uint8_t* type = &png[pos + 4];
        const uint8_t* data = &png[pos + 8];
        if (png_test_be32(data + len) != playgta6::gpu::native_exec::png_crc32(type, len + 4)) return false;
        if (!std::memcmp(type, "IHDR", 4)) {
            if (len != 13 || data[8] != 8 || data[9] != 2) return false;
            *w = png_test_be32(data);
            *h = png_test_be32(data + 4);
            have_ihdr = true;
        } else if (!std::memcmp(type, "IDAT", 4)) {
            idat.insert(idat.end(), data, data + len);
        } else if (!std::memcmp(type, "IEND", 4)) {
            have_iend = true;
        }
        pos += 12 + len;
    }
    if (!have_ihdr || !have_iend || idat.size() < 6 || idat[0] != 0x78) return false;
    std::vector<uint8_t> raw;
    size_t i = 2;
    for (;;) {
        if (i + 5 > idat.size()) return false;
        const bool last = idat[i] & 1;
        if ((idat[i] >> 1) != 0) return false;  // stored blocks only
        const uint32_t n = idat[i + 1] | (uint32_t{idat[i + 2]} << 8);
        if (uint32_t{idat[i + 3]} + (uint32_t{idat[i + 4]} << 8) != (~n & 0xFFFFu)) return false;
        i += 5;
        if (i + n > idat.size()) return false;
        raw.insert(raw.end(), idat.begin() + static_cast<std::ptrdiff_t>(i),
                   idat.begin() + static_cast<std::ptrdiff_t>(i + n));
        i += n;
        if (last) break;
    }
    if (i + 4 != idat.size()) return false;
    uint32_t a = 1, b = 0;
    for (uint8_t c : raw) {
        a = (a + c) % 65521u;
        b = (b + a) % 65521u;
    }
    if (png_test_be32(&idat[i]) != ((b << 16) | a)) return false;
    if (raw.size() != (size_t{*w} * 3 + 1) * *h) return false;
    rgb->clear();
    for (uint32_t y = 0; y < *h; ++y) {
        const uint8_t* row = &raw[(size_t{*w} * 3 + 1) * y];
        if (row[0] != 0) return false;
        rgb->insert(rgb->end(), row + 1, row + 1 + size_t{*w} * 3);
    }
    return true;
}
