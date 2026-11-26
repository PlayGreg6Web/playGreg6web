#pragma once

// User keymap for the native --host-input keyboard/mouse (keymap.ini). Rebinds real host keys to pad
// buttons, stick directions and triggers, and optionally maps real mouse movement to the right stick.
// Only real host input is mapped; nothing is synthesized. The file is applied on top of the default
// table (docs/PLAY_LOCALLY.md), so it only needs to list changes ("Key = none" unbinds a key).
//
//   # comment (also ';')
//   [keys]
//   Enter = a              # key name = target
//   X     = right_trigger
//   [mouse]
//   look = on              # mouse movement drives the right stick (default off)
//   sensitivity = 0.05     # stick deflection per pixel per 8 ms poll, 0.001..10
//   invert_y = off

#include <cstdint>
#include <string>
#include <vector>

#include "abi/playgta6_abi.h"

namespace playgta6::guest::keymap {

enum class Axis : uint8_t { None, LeftX, LeftY, RightX, RightY, LeftTrigger, RightTrigger };

struct Target {
    uint32_t button = 0;     // InputState button bit, or 0
    Axis axis = Axis::None;  // stick axis / trigger
    float sign = 0.0f;       // +1 / -1 for sticks, +1 for triggers
    bool none() const { return button == 0 && axis == Axis::None; }
};

struct Binding {
    int vk = 0;  // Windows virtual-key code
    Target target;
};

struct Keymap {
    std::vector<Binding> keys;
    bool mouse_look = false;
    float mouse_sensitivity = 0.05f;
    bool mouse_invert_y = false;
};

// The built-in table (same as convert_keyboard).
Keymap default_keymap();

// Parses `text` on top of the defaults. Returns false and fills `errors` ("line N: ...") on any problem.
bool parse_keymap(const std::string& text, Keymap& out, std::vector<std::string>& errors);

// Text of the default file written on first run (documented, lists the whole default table).
std::string default_keymap_text();

// Loads the keymap for a run. `explicit_path` (--keymap) must exist; otherwise
// %LOCALAPPDATA%\playgta6\keymap.ini is used and written with the defaults when missing.
// Returns false with `errors` set when the file cannot be read or is invalid.
bool load_keymap(const std::string& explicit_path, Keymap& out, std::vector<std::string>& errors, std::string& used_path);

inline float clamp_axis(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Pure mapping: `is_down(vk)` reports a held key; mouse_dx/dy are the pixel movement since the last poll
// (screen coordinates, y down). Sticks clamp to [-1,1], opposite keys cancel.
template <typename DownFn>
InputState evaluate(const Keymap& map, DownFn&& is_down, float mouse_dx, float mouse_dy) {
    InputState s{};
    float lx = 0, ly = 0, rx = 0, ry = 0, lt = 0, rt = 0;
    for (const Binding& b : map.keys) {
        if (b.target.none() || !is_down(b.vk)) {
            continue;
        }
        s.buttons |= b.target.button;
        switch (b.target.axis) {
            case Axis::LeftX: lx += b.target.sign; break;
            case Axis::LeftY: ly += b.target.sign; break;
            case Axis::RightX: rx += b.target.sign; break;
            case Axis::RightY: ry += b.target.sign; break;
            case Axis::LeftTrigger: lt = 1.0f; break;
            case Axis::RightTrigger: rt = 1.0f; break;
            case Axis::None: break;
        }
    }
    if (map.mouse_look) {
        rx += mouse_dx * map.mouse_sensitivity;
        ry += (map.mouse_invert_y ? mouse_dy : -mouse_dy) * map.mouse_sensitivity;
    }
    s.left_stick_x = clamp_axis(lx, -1.0f, 1.0f);
    s.left_stick_y = clamp_axis(ly, -1.0f, 1.0f);
    s.right_stick_x = clamp_axis(rx, -1.0f, 1.0f);
    s.right_stick_y = clamp_axis(ry, -1.0f, 1.0f);
    s.left_trigger = lt;
    s.right_trigger = rt;
    return s;
}

}  // namespace playgta6::guest::keymap
