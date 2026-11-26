// Native window: pure rect math and last-window-state persistence (no Win32 here, unit tested in
// tests/test_window_layout.cpp). The Win32 layer feeds it real monitor/DPI values.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace playgta6::gpu::native_exec {

struct PresentRect {
    int32_t x = 0, y = 0;
    uint32_t w = 0, h = 0;
    bool operator==(const PresentRect& o) const { return x == o.x && y == o.y && w == o.w && h == o.h; }
};

// Largest src-aspect rectangle centred in a dst_w x dst_h client area (letterbox / pillarbox).
// Degenerate input yields an empty rect. Integer math, rounds to nearest, never exceeds dst.
inline PresentRect letterbox_rect(uint32_t src_w, uint32_t src_h, uint32_t dst_w, uint32_t dst_h) {
    PresentRect r;
    if (src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0) return r;
    const uint64_t lhs = static_cast<uint64_t>(dst_w) * src_h;  // compare dst aspect with src aspect
    const uint64_t rhs = static_cast<uint64_t>(dst_h) * src_w;
    if (lhs >= rhs) {  // dst is wider: fit height, pillarbox
        r.h = dst_h;
        r.w = static_cast<uint32_t>(
            std::min<uint64_t>(dst_w, (static_cast<uint64_t>(dst_h) * src_w + src_h / 2) / src_h));
    } else {  // dst is taller: fit width, letterbox
        r.w = dst_w;
        r.h = static_cast<uint32_t>(
            std::min<uint64_t>(dst_h, (static_cast<uint64_t>(dst_w) * src_h + src_w / 2) / src_w));
    }
    r.w = std::max<uint32_t>(r.w, 1);
    r.h = std::max<uint32_t>(r.h, 1);
    r.x = static_cast<int32_t>((dst_w - r.w) / 2);
    r.y = static_cast<int32_t>((dst_h - r.h) / 2);
    return r;
}

// Maps a point in client coordinates into source (swapchain) coordinates; false when it falls in
// the letterbox bars.
inline bool client_to_source(const PresentRect& r, uint32_t src_w, uint32_t src_h, int32_t cx, int32_t cy,
                             int32_t* sx, int32_t* sy) {
    if (r.w == 0 || r.h == 0) return false;
    if (cx < r.x || cy < r.y || cx >= r.x + static_cast<int32_t>(r.w) || cy >= r.y + static_cast<int32_t>(r.h))
        return false;
    *sx = static_cast<int32_t>(static_cast<int64_t>(cx - r.x) * src_w / r.w);
    *sy = static_cast<int32_t>(static_cast<int64_t>(cy - r.y) * src_h / r.h);
    return true;
}

// Scales a 96-dpi pixel length to the given dpi (round to nearest). dpi 0 is treated as 96.
inline int32_t scale_for_dpi(int32_t px, uint32_t dpi) {
    if (dpi == 0) dpi = 96;
    return static_cast<int32_t>((static_cast<int64_t>(px) * dpi + (px >= 0 ? 48 : -48)) / 96);
}

struct SavedWindowState {
    int32_t x = 0, y = 0;   // outer (restored-position) top-left, virtual-desktop pixels
    uint32_t w = 0, h = 0;  // outer size
    bool maximized = false;
    bool operator==(const SavedWindowState& o) const {
        return x == o.x && y == o.y && w == o.w && h == o.h && maximized == o.maximized;
    }
};

struct MonitorRect {
    int32_t left = 0, top = 0, right = 0, bottom = 0;
};

inline std::string format_window_state(const SavedWindowState& s) {
    char buf[160];
    std::snprintf(buf, sizeof buf, "x=%d\ny=%d\nw=%u\nh=%u\nmaximized=%d\n", s.x, s.y, s.w, s.h,
                  s.maximized ? 1 : 0);
    return buf;
}

// Parses key=value lines. Unknown keys are ignored; the state is valid only with w and h present
// and sane (16..32768).
inline bool parse_window_state(const std::string& text, SavedWindowState* out) {
    SavedWindowState s;
    bool have_w = false, have_h = false;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t end = text.find('\n', pos);
        if (end == std::string::npos) end = text.size();
        const std::string line = text.substr(pos, end - pos);
        pos = end + 1;
        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = line.substr(0, eq);
        const std::string val = line.substr(eq + 1);
        char* tail = nullptr;
        const long v = std::strtol(val.c_str(), &tail, 10);
        if (tail == val.c_str()) continue;
        if (key == "x") {
            s.x = static_cast<int32_t>(v);
        } else if (key == "y") {
            s.y = static_cast<int32_t>(v);
        } else if (key == "w" && v >= 16 && v <= 32768) {
            s.w = static_cast<uint32_t>(v);
            have_w = true;
        } else if (key == "h" && v >= 16 && v <= 32768) {
            s.h = static_cast<uint32_t>(v);
            have_h = true;
        } else if (key == "maximized") {
            s.maximized = v != 0;
        }
    }
    if (!have_w || !have_h) return false;
    *out = s;
    return true;
}

// Makes a saved window usable on the current monitor layout: shrinks it to the chosen monitor if
// needed and, when it would be (almost) off-screen (a monitor was unplugged), re-centres it in the
// nearest monitor. False when there are no monitors.
inline bool fit_state_to_monitors(SavedWindowState* s, const std::vector<MonitorRect>& monitors) {
    if (monitors.empty()) return false;
    // Pick the monitor with the largest overlap, else the one nearest the window centre.
    const int64_t cx = static_cast<int64_t>(s->x) + s->w / 2, cy = static_cast<int64_t>(s->y) + s->h / 2;
    const MonitorRect* best = nullptr;
    int64_t best_score = -1;
    bool overlaps = false;
    for (const auto& m : monitors) {
        const int64_t ox = std::max<int64_t>(
            0, std::min<int64_t>(static_cast<int64_t>(s->x) + s->w, m.right) - std::max<int64_t>(s->x, m.left));
        const int64_t oy = std::max<int64_t>(
            0, std::min<int64_t>(static_cast<int64_t>(s->y) + s->h, m.bottom) - std::max<int64_t>(s->y, m.top));
        const int64_t area = ox * oy;
        if (area > 0) {
            if (!overlaps || area > best_score) {
                best = &m;
                best_score = area;
            }
            overlaps = true;
        } else if (!overlaps) {
            const int64_t dx = cx < m.left ? m.left - cx : (cx > m.right ? cx - m.right : 0);
            const int64_t dy = cy < m.top ? m.top - cy : (cy > m.bottom ? cy - m.bottom : 0);
            const int64_t score = -(dx * dx + dy * dy);  // closer to 0 is better
            if (!best || score > best_score) {
                best = &m;
                best_score = score;
            }
        }
    }
    const int64_t mw = best->right - best->left, mh = best->bottom - best->top;
    s->w = static_cast<uint32_t>(std::min<int64_t>(s->w, mw));
    s->h = static_cast<uint32_t>(std::min<int64_t>(s->h, mh));
    // Require at least 64x32 px on the chosen monitor, else re-centre fully in it.
    const int64_t vis_w = std::min<int64_t>(static_cast<int64_t>(s->x) + s->w, best->right) -
                          std::max<int64_t>(s->x, best->left);
    const int64_t vis_h = std::min<int64_t>(static_cast<int64_t>(s->y) + s->h, best->bottom) -
                          std::max<int64_t>(s->y, best->top);
    if (!overlaps || vis_w < 64 || vis_h < 32) {
        s->x = static_cast<int32_t>(best->left + (mw - s->w) / 2);
        s->y = static_cast<int32_t>(best->top + (mh - s->h) / 2);
    }
    return true;
}

}  // namespace playgta6::gpu::native_exec
