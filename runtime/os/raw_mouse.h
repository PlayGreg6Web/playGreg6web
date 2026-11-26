#pragma once

// Hand-off between the native window (Win32 Raw Input, window thread) and the host input poller
// (--host-input thread) for mouse look. The window accumulates real relative mouse deltas while it has
// focus and has captured the cursor; the poller drains them every poll. Only real input passes through.

#include <atomic>
#include <cstdint>

namespace playgta6::os::raw_mouse {

struct State {
    std::atomic<bool> look_enabled{false};  // keymap.ini [mouse] look = on (set by the host input)
    std::atomic<bool> window_ready{false};  // a live window registered for raw input
    std::atomic<bool> capturing{false};     // cursor currently captured (focused, not released/minimised)
    std::atomic<int32_t> dx{0}, dy{0};      // accumulated since the last take()
};

inline State& state() {
    static State s;
    return s;
}

// Window thread: adds one WM_INPUT relative movement. Ignored unless capturing.
inline void add(int32_t dx, int32_t dy) {
    State& s = state();
    if (!s.capturing.load()) return;
    s.dx.fetch_add(dx);
    s.dy.fetch_add(dy);
}

// Poll thread: returns and clears the accumulated movement.
inline void take(int32_t* dx, int32_t* dy) {
    State& s = state();
    *dx = s.dx.exchange(0);
    *dy = s.dy.exchange(0);
}

}  // namespace playgta6::os::raw_mouse
