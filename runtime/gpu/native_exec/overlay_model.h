// Native executor: on-screen stats overlay (F3 by default, os/window_hotkeys.h Action::Overlay). Pure logic, no GPU:
// the 5x7 bitmap font (ASCII only, lower case is drawn as upper case), the frame history, the text lines and the CPU
// rasterizer that builds the overlay bitmap, plus a CPU twin of the composite the executor does on the GPU
// (premultiplied "over", nearest scaling). The executor draws the bitmap into the window canvas AFTER every copy of
// the guest's frame (digest, screenshot, probe), so the overlay never reaches --live-digest or screenshots.
// Docs: docs/NATIVE_EXECUTOR_DESIGN.md "On-screen overlay".
#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace playgta6::gpu::native_exec {

// Glyphs for ASCII 0x20..0x5F, 5 columns each, bit 0 = top row.
inline constexpr uint8_t kOverlayGlyphs[64][5] = {
    {0x00, 0x00, 0x00, 0x00, 0x00}, {0x00, 0x00, 0x5F, 0x00, 0x00}, {0x00, 0x07, 0x00, 0x07, 0x00},
    {0x14, 0x7F, 0x14, 0x7F, 0x14}, {0x24, 0x2A, 0x7F, 0x2A, 0x12}, {0x23, 0x13, 0x08, 0x64, 0x62},
    {0x36, 0x49, 0x55, 0x22, 0x50}, {0x00, 0x05, 0x03, 0x00, 0x00}, {0x00, 0x1C, 0x22, 0x41, 0x00},
    {0x00, 0x41, 0x22, 0x1C, 0x00}, {0x14, 0x08, 0x3E, 0x08, 0x14}, {0x08, 0x08, 0x3E, 0x08, 0x08},
    {0x00, 0x50, 0x30, 0x00, 0x00}, {0x08, 0x08, 0x08, 0x08, 0x08}, {0x00, 0x60, 0x60, 0x00, 0x00},
    {0x20, 0x10, 0x08, 0x04, 0x02}, {0x3E, 0x51, 0x49, 0x45, 0x3E}, {0x00, 0x42, 0x7F, 0x40, 0x00},
    {0x42, 0x61, 0x51, 0x49, 0x46}, {0x21, 0x41, 0x45, 0x4B, 0x31}, {0x18, 0x14, 0x12, 0x7F, 0x10},
    {0x27, 0x45, 0x45, 0x45, 0x39}, {0x3C, 0x4A, 0x49, 0x49, 0x30}, {0x01, 0x71, 0x09, 0x05, 0x03},
    {0x36, 0x49, 0x49, 0x49, 0x36}, {0x06, 0x49, 0x49, 0x29, 0x1E}, {0x00, 0x36, 0x36, 0x00, 0x00},
    {0x00, 0x56, 0x36, 0x00, 0x00}, {0x08, 0x14, 0x22, 0x41, 0x00}, {0x14, 0x14, 0x14, 0x14, 0x14},
    {0x00, 0x41, 0x22, 0x14, 0x08}, {0x02, 0x01, 0x51, 0x09, 0x06}, {0x32, 0x49, 0x79, 0x41, 0x3E},
    {0x7E, 0x11, 0x11, 0x11, 0x7E}, {0x7F, 0x49, 0x49, 0x49, 0x36}, {0x3E, 0x41, 0x41, 0x41, 0x22},
    {0x7F, 0x41, 0x41, 0x22, 0x1C}, {0x7F, 0x49, 0x49, 0x49, 0x41}, {0x7F, 0x09, 0x09, 0x09, 0x01},
    {0x3E, 0x41, 0x49, 0x49, 0x7A}, {0x7F, 0x08, 0x08, 0x08, 0x7F}, {0x00, 0x41, 0x7F, 0x41, 0x00},
    {0x20, 0x40, 0x41, 0x3F, 0x01}, {0x7F, 0x08, 0x14, 0x22, 0x41}, {0x7F, 0x40, 0x40, 0x40, 0x40},
    {0x7F, 0x02, 0x0C, 0x02, 0x7F}, {0x7F, 0x04, 0x08, 0x10, 0x7F}, {0x3E, 0x41, 0x41, 0x41, 0x3E},
    {0x7F, 0x09, 0x09, 0x09, 0x06}, {0x3E, 0x41, 0x51, 0x21, 0x5E}, {0x7F, 0x09, 0x19, 0x29, 0x46},
    {0x46, 0x49, 0x49, 0x49, 0x31}, {0x01, 0x01, 0x7F, 0x01, 0x01}, {0x3F, 0x40, 0x40, 0x40, 0x3F},
    {0x1F, 0x20, 0x40, 0x20, 0x1F}, {0x3F, 0x40, 0x38, 0x40, 0x3F}, {0x63, 0x14, 0x08, 0x14, 0x63},
    {0x07, 0x08, 0x70, 0x08, 0x07}, {0x61, 0x51, 0x49, 0x45, 0x43}, {0x00, 0x7F, 0x41, 0x41, 0x00},
    {0x02, 0x04, 0x08, 0x10, 0x20}, {0x00, 0x41, 0x41, 0x7F, 0x00}, {0x04, 0x02, 0x01, 0x02, 0x04},
    {0x40, 0x40, 0x40, 0x40, 0x40},
};

inline constexpr int kOverlayCols = 34;  // characters per line
inline constexpr int kOverlayPad = 4;
inline constexpr int kOverlayGlyphW = 6, kOverlayLineH = 9;
inline constexpr int kOverlayWidth = kOverlayPad * 2 + kOverlayCols * kOverlayGlyphW;  // 212
inline constexpr int kOverlayGraphH = 32;
inline constexpr size_t kOverlayHistory = 120;      // frames in the frame-time graph
inline constexpr double kOverlayGraphMaxMs = 50.0;  // graph top; slower frames are clipped

// The glyph columns of one character (lower case folds to upper, anything outside 0x20..0x5F is '?').
inline const uint8_t* overlay_glyph(char c) {
    unsigned u = static_cast<unsigned char>(c);
    if (u >= 'a' && u <= 'z') u -= 32;
    if (u < 0x20 || u > 0x5F) u = '?';
    return kOverlayGlyphs[u - 0x20];
}

// Last kOverlayHistory presented frames (oldest first through *_at()).
class OverlayHistory {
public:
    void push(double frame_ms, uint32_t draws) {
        ms_[head_] = static_cast<float>(frame_ms);
        draws_[head_] = draws;
        head_ = (head_ + 1) % kOverlayHistory;
        if (count_ < kOverlayHistory) ++count_;
    }
    size_t size() const { return count_; }
    float ms_at(size_t i) const { return ms_[(head_ + kOverlayHistory - count_ + i) % kOverlayHistory]; }
    uint32_t draws_at(size_t i) const { return draws_[(head_ + kOverlayHistory - count_ + i) % kOverlayHistory]; }
    // Mean of the newest n entries (n clamped to size()); 0 when empty.
    double avg_ms(size_t n) const {
        n = n < count_ ? n : count_;
        double s = 0;
        for (size_t i = count_ - n; i < count_; ++i) s += ms_at(i);
        return n ? s / static_cast<double>(n) : 0.0;
    }
    double avg_draws(size_t n) const {
        n = n < count_ ? n : count_;
        double s = 0;
        for (size_t i = count_ - n; i < count_; ++i) s += draws_at(i);
        return n ? s / static_cast<double>(n) : 0.0;
    }

private:
    std::array<float, kOverlayHistory> ms_{};
    std::array<uint32_t, kOverlayHistory> draws_{};
    size_t head_ = 0, count_ = 0;
};

struct OverlayStatus {
    double fps = 0;  // 0 = no recent frames
    double frame_ms = 0;
    double draws_per_frame = 0;
    double gpu_ms = -1;            // < 0: no GPU timestamps (adapter lacks timestamp-query / nothing measured yet)
    int64_t audio_underruns = -1;  // < 0: no live audio
    std::string present_mode;
    std::string fault;  // non-empty once the guest stopped: why (shown in red)
};

// One row of text per entry, each no longer than kOverlayCols.
inline std::vector<std::string> overlay_lines(const OverlayStatus& s) {
    char b[64];
    std::vector<std::string> l;
    std::snprintf(b, sizeof b, "%.0f FPS  %.1f MS", s.fps, s.frame_ms);
    l.emplace_back(b);
    l.emplace_back(("PRESENT " + (s.present_mode.empty() ? std::string("?") : s.present_mode)).substr(0, kOverlayCols));
    std::snprintf(b, sizeof b, "DRAWS %.0f/FRAME", s.draws_per_frame);
    l.emplace_back(b);
    if (s.gpu_ms >= 0) {
        std::snprintf(b, sizeof b, "GPU %.2f MS", s.gpu_ms);
        l.emplace_back(b);
    } else {
        l.emplace_back("GPU N/A");
    }
    if (s.audio_underruns >= 0)
        l.emplace_back("AUDIO UNDERRUNS " + std::to_string(s.audio_underruns));
    else
        l.emplace_back("AUDIO OFF");
    return l;
}

// Wraps the fault text to kOverlayCols columns (hard wrap, at most max_lines rows, "..." when cut).
inline std::vector<std::string> overlay_fault_lines(const std::string& fault, size_t max_lines = 6) {
    std::vector<std::string> out;
    if (fault.empty()) return out;
    const std::string text = "STOPPED: " + fault;
    size_t pos = 0;
    while (pos < text.size()) {
        if (out.size() == max_lines) {
            std::string& last = out.back();
            last.replace(last.size() - 3, 3, "...");
            break;
        }
        out.push_back(text.substr(pos, kOverlayCols));
        pos += kOverlayCols;
    }
    return out;
}

struct OverlayBitmap {
    uint32_t w = 0, h = 0;
    std::vector<uint8_t> rgba;  // premultiplied, row-major, w * h * 4
};

inline int overlay_height(const OverlayStatus& s) {
    const int fault_rows = static_cast<int>(overlay_fault_lines(s.fault).size());
    return kOverlayPad * 2 + static_cast<int>(overlay_lines(s).size()) * kOverlayLineH + 2 + kOverlayGraphH + 2 +
           (fault_rows ? 2 + fault_rows * kOverlayLineH : 0);
}

// Rasterizes the box: translucent dark background, text, the frame-time graph (green; orange above 20 ms, red above
// 33 ms; 50 ms at the top) and the fault text in red.
inline OverlayBitmap render_overlay(const OverlayStatus& s, const OverlayHistory& hist) {
    OverlayBitmap bm;
    bm.w = kOverlayWidth;
    bm.h = static_cast<uint32_t>(overlay_height(s));
    bm.rgba.assign(size_t{bm.w} * bm.h * 4, 0);
    auto put = [&](int x, int y, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {  // straight colour, stored premultiplied
        if (x < 0 || y < 0 || x >= static_cast<int>(bm.w) || y >= static_cast<int>(bm.h)) return;
        uint8_t* p = &bm.rgba[(size_t{static_cast<uint32_t>(y)} * bm.w + static_cast<uint32_t>(x)) * 4];
        p[0] = static_cast<uint8_t>(r * a / 255);
        p[1] = static_cast<uint8_t>(g * a / 255);
        p[2] = static_cast<uint8_t>(b * a / 255);
        p[3] = a;
    };
    for (uint32_t y = 0; y < bm.h; ++y)
        for (uint32_t x = 0; x < bm.w; ++x) put(static_cast<int>(x), static_cast<int>(y), 0, 0, 0, 166);  // ~65 % black
    auto text = [&](int x, int y, const std::string& t, uint8_t r, uint8_t g, uint8_t b) {
        for (size_t i = 0; i < t.size() && static_cast<int>(i) < kOverlayCols; ++i) {
            const uint8_t* gl = overlay_glyph(t[i]);
            for (int c = 0; c < 5; ++c)
                for (int row = 0; row < 7; ++row)
                    if (gl[c] & (1 << row)) put(x + static_cast<int>(i) * kOverlayGlyphW + c, y + row, r, g, b, 255);
        }
    };
    int y = kOverlayPad;
    const std::vector<std::string> lines = overlay_lines(s);
    for (size_t i = 0; i < lines.size(); ++i) {
        text(kOverlayPad, y, lines[i], 255, 255, 255);
        y += kOverlayLineH;
        if (i == 0) {  // the graph sits under the fps line
            y += 2;
            const int gx = kOverlayPad, gy = y;
            for (int x = 0; x < static_cast<int>(kOverlayHistory); ++x) put(gx + x, gy + kOverlayGraphH - 1, 90, 90, 90, 255);
            const size_t n = hist.size();
            for (size_t k = 0; k < n; ++k) {
                const double ms = hist.ms_at(k);
                int hh = static_cast<int>(ms / kOverlayGraphMaxMs * kOverlayGraphH + 0.5);
                if (hh < 1) hh = 1;
                if (hh > kOverlayGraphH) hh = kOverlayGraphH;
                const uint8_t r = ms > 20.0 ? 255 : 80;
                const uint8_t g = ms > 33.4 ? 60 : ms > 20.0 ? 170 : 220;
                const uint8_t b = ms > 33.4 ? 60 : ms > 20.0 ? 40 : 80;
                const int x = gx + static_cast<int>(kOverlayHistory - n + k);
                for (int row = 0; row < hh; ++row) put(x, gy + kOverlayGraphH - 1 - row, r, g, b, 255);
            }
            y += kOverlayGraphH + 2;
        }
    }
    const std::vector<std::string> fl = overlay_fault_lines(s.fault);
    if (!fl.empty()) {
        y += 2;
        for (const std::string& t : fl) {
            text(kOverlayPad, y, t, 255, 90, 90);
            y += kOverlayLineH;
        }
    }
    return bm;
}

// Integer pixel scale of the box for a canvas this tall: 1 below 900 px, 2 below 1800, else 3.
inline uint32_t overlay_scale(uint32_t canvas_h) { return canvas_h < 900 ? 1u : canvas_h < 1800 ? 2u : 3u; }

// CPU twin of the GPU composite: the box at the canvas' top-left, nearest-scaled, premultiplied "over" onto 4-byte
// pixels (dst_bgra: the destination stores blue first).
inline void overlay_composite(std::vector<uint8_t>& dst, uint32_t dw, uint32_t dh, const OverlayBitmap& bm,
                              uint32_t scale, bool dst_bgra) {
    for (uint32_t y = 0; y < bm.h * scale && y < dh; ++y)
        for (uint32_t x = 0; x < bm.w * scale && x < dw; ++x) {
            const uint8_t* s = &bm.rgba[(size_t{y / scale} * bm.w + x / scale) * 4];
            uint8_t* d = &dst[(size_t{y} * dw + x) * 4];
            const uint32_t inv = 255u - s[3];
            const int r = dst_bgra ? 2 : 0, b = dst_bgra ? 0 : 2;
            d[r] = static_cast<uint8_t>(s[0] + (d[r] * inv + 127) / 255);
            d[1] = static_cast<uint8_t>(s[1] + (d[1] * inv + 127) / 255);
            d[b] = static_cast<uint8_t>(s[2] + (d[b] * inv + 127) / 255);
            d[3] = static_cast<uint8_t>(s[3] + (d[3] * inv + 127) / 255);
        }
}

}  // namespace playgta6::gpu::native_exec
