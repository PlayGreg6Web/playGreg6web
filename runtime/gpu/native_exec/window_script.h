// Native window: test-hook script (`--window-script`), swapchain reconfigure state machine and the
// letterbox read-back probe. Pure logic, no Win32 / Dawn: unit tested in tests/test_window_layout.cpp.
//
// Script syntax: comma separated `<ms>:<action>`, times counted from the first presented frame,
// non-decreasing. Actions: `resize=WxH` (client area in physical pixels), `minimize`, `restore`,
// `maximize`, `fullscreen`, `windowed` (leave fullscreen), `probe` (read the presented canvas back
// after the next present and check the letterbox bars).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "gpu/native_exec/window_layout.h"

namespace playgta6::gpu::native_exec {

enum class WindowActionKind { Resize, Minimize, Restore, Maximize, Fullscreen, Windowed, Probe };

struct WindowAction {
    WindowActionKind kind = WindowActionKind::Probe;
    uint32_t w = 0, h = 0;  // Resize: client area
};

struct TimedWindowAction {
    uint64_t at_ms = 0;
    WindowAction action;
};

inline bool parse_window_script(const std::string& text, std::vector<TimedWindowAction>* out, std::string* error) {
    std::vector<TimedWindowAction> res;
    size_t pos = 0;
    uint64_t last = 0;
    auto fail = [&](const std::string& m) {
        if (error) *error = m;
        return false;
    };
    while (pos < text.size()) {
        size_t end = text.find(',', pos);
        if (end == std::string::npos) end = text.size();
        const std::string item = text.substr(pos, end - pos);
        pos = end + 1;
        if (item.empty()) continue;
        const size_t colon = item.find(':');
        if (colon == std::string::npos || colon == 0)
            return fail("window script: expected <ms>:<action> in '" + item + "'");
        char* tail = nullptr;
        const unsigned long long ms = std::strtoull(item.c_str(), &tail, 10);
        if (tail != item.c_str() + colon) return fail("window script: bad time in '" + item + "'");
        if (ms < last) return fail("window script: times must not decrease ('" + item + "')");
        last = ms;
        const std::string name = item.substr(colon + 1);
        TimedWindowAction t;
        t.at_ms = ms;
        if (name == "minimize") t.action.kind = WindowActionKind::Minimize;
        else if (name == "restore") t.action.kind = WindowActionKind::Restore;
        else if (name == "maximize") t.action.kind = WindowActionKind::Maximize;
        else if (name == "fullscreen") t.action.kind = WindowActionKind::Fullscreen;
        else if (name == "windowed") t.action.kind = WindowActionKind::Windowed;
        else if (name == "probe") t.action.kind = WindowActionKind::Probe;
        else if (name.compare(0, 7, "resize=") == 0) {
            unsigned w = 0, h = 0;
            char extra = 0;
            if (std::sscanf(name.c_str() + 7, "%ux%u%c", &w, &h, &extra) != 2 || w < 16 || h < 16 || w > 16384 ||
                h > 16384)
                return fail("window script: bad size in '" + item + "' (16..16384)");
            t.action.kind = WindowActionKind::Resize;
            t.action.w = w;
            t.action.h = h;
        } else {
            return fail("window script: unknown action '" + name + "'");
        }
        res.push_back(t);
    }
    *out = std::move(res);
    return true;
}

// Hands out the script's actions as time passes (elapsed_ms since the first presented frame).
class WindowScriptRunner {
public:
    void set(std::vector<TimedWindowAction> s) {
        script_ = std::move(s);
        next_ = 0;
    }
    bool empty() const { return script_.empty(); }
    bool finished() const { return next_ >= script_.size(); }
    // The next action whose time has come, advancing; false when none is due.
    bool pop_due(uint64_t elapsed_ms, WindowAction* out) {
        if (next_ >= script_.size() || script_[next_].at_ms > elapsed_ms) return false;
        *out = script_[next_++].action;
        return true;
    }

private:
    std::vector<TimedWindowAction> script_;
    size_t next_ = 0;
};

// ---- swapchain reconfigure state machine ----

// wgpu::SurfaceGetCurrentTextureStatus, without the dependency.
enum class AcquireStatus { Optimal, Suboptimal, Timeout, Outdated, Lost, OutOfMemory, DeviceLost, Error };

enum class SurfacePlan {
    SkipMinimized,  // zero-sized client area: nothing can be presented, keep the configuration
    Configure,      // (re)configure to the client size, then acquire
    Acquire,        // configuration is current
};

enum class AcquireNext {
    Use,          // the texture is good
    Reconfigure,  // reconfigure to the client size and acquire again
    Drop,         // drop this frame (the next present tries again)
    Fatal,        // device-level failure: reconfiguring cannot help
};

class SurfaceState {
public:
    SurfacePlan plan(uint32_t client_w, uint32_t client_h) {
        if (client_w == 0 || client_h == 0) {
            ++minimized_skips_;
            return SurfacePlan::SkipMinimized;
        }
        if (!configured_ || dirty_ || client_w != w_ || client_h != h_) return SurfacePlan::Configure;
        return SurfacePlan::Acquire;
    }
    void on_configured(uint32_t w, uint32_t h) {
        configured_ = true;
        dirty_ = false;
        w_ = w;
        h_ = h;
        ++configures_;
    }
    // The surface configuration must be redone (format change, lost surface).
    void invalidate() { dirty_ = true; }
    // attempt counts the retries within one acquire (0 = first try).
    AcquireNext on_acquire(AcquireStatus s, int attempt) {
        switch (s) {
            case AcquireStatus::Optimal:
            case AcquireStatus::Suboptimal:  // still presentable; the size check catches a real resize
                return AcquireNext::Use;
            case AcquireStatus::Timeout:
                return AcquireNext::Drop;  // transient: no reconfigure, the next frame asks again
            case AcquireStatus::Outdated:
            case AcquireStatus::Lost:
            case AcquireStatus::Error:
                if (attempt == 0) {
                    ++retries_;
                    return AcquireNext::Reconfigure;
                }
                dirty_ = true;  // still failing: start from a clean configure next frame
                return AcquireNext::Drop;
            case AcquireStatus::OutOfMemory:
            case AcquireStatus::DeviceLost:
                return AcquireNext::Fatal;
        }
        return AcquireNext::Drop;
    }
    bool configured() const { return configured_; }
    uint32_t width() const { return w_; }
    uint32_t height() const { return h_; }
    uint64_t configures() const { return configures_; }
    uint64_t minimized_skips() const { return minimized_skips_; }
    uint64_t retries() const { return retries_; }

private:
    bool configured_ = false, dirty_ = false;
    uint32_t w_ = 0, h_ = 0;
    uint64_t configures_ = 0, minimized_skips_ = 0, retries_ = 0;
};

// ---- letterbox read-back probe ----

struct LetterboxProbe {
    uint64_t bar_pixels = 0, bar_nonblack = 0;      // outside the present rect
    uint64_t inner_pixels = 0, inner_nonblack = 0;  // inside it
    // The bars are clean (the picture itself may legitimately be black, e.g. a fade).
    bool ok() const { return bar_nonblack == 0; }
    bool blank() const { return inner_nonblack == 0; }
};

// Pixels are 4-byte, rows `row_bytes` apart; a pixel is black when its 3 colour bytes are 0 (works for
// rgba and bgra alike; the alpha byte is ignored).
inline LetterboxProbe probe_letterbox(const uint8_t* px, size_t row_bytes, uint32_t w, uint32_t h,
                                      const PresentRect& r) {
    LetterboxProbe p;
    for (uint32_t y = 0; y < h; ++y) {
        const uint8_t* row = px + row_bytes * y;
        const bool row_in = static_cast<int64_t>(y) >= r.y && static_cast<int64_t>(y) < static_cast<int64_t>(r.y) + r.h;
        for (uint32_t x = 0; x < w; ++x) {
            const bool in = row_in && static_cast<int64_t>(x) >= r.x && static_cast<int64_t>(x) < static_cast<int64_t>(r.x) + r.w;
            const bool nonblack = (row[x * 4] | row[x * 4 + 1] | row[x * 4 + 2]) != 0;
            if (in) {
                ++p.inner_pixels;
                p.inner_nonblack += nonblack;
            } else {
                ++p.bar_pixels;
                p.bar_nonblack += nonblack;
            }
        }
    }
    return p;
}

}  // namespace playgta6::gpu::native_exec
