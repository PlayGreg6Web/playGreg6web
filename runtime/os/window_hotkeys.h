#pragma once

// The one table of native live-window hotkeys (window layer, no Win32 here so it is unit-testable).
// Every window-level key lives here with its default; keymap.ini [hotkeys] (runtime/guest/keymap.cpp)
// rebinds them and rejects duplicates at load. The window reads the active table via active().
// Esc stays the pad Menu button ([keys]), it is not a window hotkey.

#include <array>
#include <atomic>
#include <cstddef>
#include <string>

namespace playgta6::os::window_hotkeys {

enum class Action : int {
    Fullscreen = 0,  // toggle borderless fullscreen
    Screenshot,      // save the presented frame as PNG
    MouseRelease,    // release / re-capture the mouse-look cursor
    TargetMip,       // --show-target: next mip
    TargetLayer,     // --show-target: next array layer
    TargetPrev,      // --show-target: previous render target
    TargetNext,      // --show-target: next render target
    Overlay,         // toggle the on-screen stats overlay
    Count
};
inline constexpr size_t kCount = static_cast<size_t>(Action::Count);

struct Chord {
    int vk = 0;  // Windows virtual-key code
    bool ctrl = false;
    bool operator==(const Chord& o) const { return vk == o.vk && ctrl == o.ctrl; }
};

struct Info {
    Action action;
    const char* name;  // keymap.ini [hotkeys] key
    const char* help;
    Chord def;
};

inline constexpr Info kTable[kCount] = {
    {Action::Fullscreen, "fullscreen", "toggle borderless fullscreen", {0x7A, false}},                // F11
    {Action::Screenshot, "screenshot", "save a screenshot (PNG)", {0x7B, false}},                     // F12
    {Action::MouseRelease, "mouse_release", "release / re-capture the mouse-look cursor", {0x77, false}},  // F8
    {Action::TargetMip, "target_mip", "--show-target: next mip", {0x76, true}},                       // Ctrl+F7
    {Action::TargetLayer, "target_layer", "--show-target: next array layer", {0x77, true}},           // Ctrl+F8
    {Action::TargetPrev, "target_prev", "--show-target: previous render target", {0x78, true}},       // Ctrl+F9
    {Action::TargetNext, "target_next", "--show-target: next render target", {0x79, true}},           // Ctrl+F10
    {Action::Overlay, "overlay", "toggle the on-screen stats overlay (fps, frame graph, fault)", {0x72, false}},  // F3
};

using Table = std::array<Chord, kCount>;

inline Table defaults() {
    Table t;
    for (size_t i = 0; i < kCount; ++i) t[i] = kTable[i].def;
    return t;
}

inline const Info* find_by_name(const std::string& lower_name) {
    for (const Info& i : kTable)
        if (lower_name == i.name) return &i;
    return nullptr;
}

// Empty when no two actions share a chord, else a message naming the first clash.
inline std::string find_duplicate(const Table& t) {
    for (size_t a = 0; a < kCount; ++a)
        for (size_t b = a + 1; b < kCount; ++b)
            if (t[a] == t[b])
                return std::string("hotkeys '") + kTable[a].name + "' and '" + kTable[b].name +
                       "' are bound to the same key";
    return {};
}

inline std::string vk_name(int vk) {
    if (vk >= 0x70 && vk <= 0x87) return "F" + std::to_string(vk - 0x6F);
    if ((vk >= 'A' && vk <= 'Z') || (vk >= '0' && vk <= '9')) return std::string(1, static_cast<char>(vk));
    return "VK" + std::to_string(vk);
}
inline std::string chord_name(const Chord& c) { return (c.ctrl ? "Ctrl+" : "") + vk_name(c.vk); }

// Short title-bar hint for the --show-target keys.
inline std::string target_hint(const Table& t) {
    return chord_name(t[static_cast<size_t>(Action::TargetMip)]) + " mip " +
           chord_name(t[static_cast<size_t>(Action::TargetLayer)]) + " layer " +
           chord_name(t[static_cast<size_t>(Action::TargetPrev)]) + "/" +
           chord_name(t[static_cast<size_t>(Action::TargetNext)]) + " target";
}

// One line per action ("name = Chord  (help)"), for the default keymap.ini and docs.
inline std::string help_text(const Table& t) {
    std::string s;
    for (size_t i = 0; i < kCount; ++i)
        s += std::string(kTable[i].name) + " = " + chord_name(t[i]) + "  (" + kTable[i].help + ")\n";
    return s;
}

// Active table (set once at startup from the keymap, read by the window thread).
struct State {
    std::array<std::atomic<int>, kCount> packed;  // vk | (ctrl << 16)
    State() { set(defaults()); }
    void set(const Table& t) {
        for (size_t i = 0; i < kCount; ++i) packed[i].store(t[i].vk | (t[i].ctrl ? 0x10000 : 0));
    }
    Chord get(Action a) const {
        const int p = packed[static_cast<size_t>(a)].load();
        return Chord{p & 0xFFFF, (p & 0x10000) != 0};
    }
    Table table() const {
        Table t;
        for (size_t i = 0; i < kCount; ++i) t[i] = get(static_cast<Action>(i));
        return t;
    }
};
inline State& active() {
    static State s;
    return s;
}

// True when the pressed key matches the action's chord (ctrl_down = Ctrl held).
inline bool matches(Action a, int vk, bool ctrl_down) {
    const Chord c = active().get(a);
    return c.vk == vk && c.ctrl == ctrl_down;
}

}  // namespace playgta6::os::window_hotkeys
