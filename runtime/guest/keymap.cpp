#include "guest/keymap.h"

#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

#include "os/game_input.h"
#include "os/window_hotkeys.h"

namespace playgta6::guest::keymap {

namespace button = playgta6::os::input_button;

namespace {

std::string lower(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

struct NamedKey {
    const char* name;
    int vk;
};
// Matched case-insensitively. Virtual-key codes are the Windows values.
const NamedKey kNamedKeys[] = {
    {"backspace", 0x08}, {"tab", 0x09}, {"enter", 0x0D}, {"return", 0x0D}, {"shift", 0x10}, {"ctrl", 0x11},
    {"alt", 0x12}, {"escape", 0x1B}, {"esc", 0x1B}, {"space", 0x20}, {"pageup", 0x21}, {"pagedown", 0x22},
    {"end", 0x23}, {"home", 0x24}, {"left", 0x25}, {"up", 0x26}, {"right", 0x27}, {"down", 0x28},
    {"insert", 0x2D}, {"delete", 0x2E}, {"lshift", 0xA0}, {"rshift", 0xA1}, {"lctrl", 0xA2}, {"rctrl", 0xA3},
    {"lalt", 0xA4}, {"ralt", 0xA5}, {"semicolon", 0xBA}, {"equals", 0xBB}, {"comma", 0xBC}, {"minus", 0xBD},
    {"period", 0xBE}, {"slash", 0xBF}, {"backtick", 0xC0}, {"lbracket", 0xDB}, {"backslash", 0xDC},
    {"rbracket", 0xDD}, {"quote", 0xDE}, {"mouse_left", 0x01}, {"mouse_right", 0x02}, {"mouse_middle", 0x04},
    {"mouse_x1", 0x05}, {"mouse_x2", 0x06},
};

bool key_from_name(const std::string& raw, int& vk) {
    const std::string n = lower(raw);
    if (n.size() == 1 && n[0] >= 'a' && n[0] <= 'z') {
        vk = 'A' + (n[0] - 'a');
        return true;
    }
    if (n.size() == 1 && n[0] >= '0' && n[0] <= '9') {
        vk = n[0];
        return true;
    }
    if (n.size() >= 2 && n.size() <= 3 && n[0] == 'f' && std::isdigit(static_cast<unsigned char>(n[1]))) {
        const int f = std::atoi(n.c_str() + 1);
        if (f >= 1 && f <= 12 && std::to_string(f) == n.substr(1)) {
            vk = 0x70 + f - 1;
            return true;
        }
    }
    if (n.size() == 7 && n.rfind("numpad", 0) == 0 && n[6] >= '0' && n[6] <= '9') {
        vk = 0x60 + (n[6] - '0');
        return true;
    }
    for (const NamedKey& k : kNamedKeys) {
        if (n == k.name) {
            vk = k.vk;
            return true;
        }
    }
    return false;
}

struct NamedTarget {
    const char* name;
    Target t;
};
Target btn(uint32_t b) { return Target{b, Axis::None, 0.0f}; }
Target ax(Axis a, float sign) { return Target{0, a, sign}; }
const NamedTarget kTargets[] = {
    {"a", btn(button::kGamepadA)}, {"b", btn(button::kGamepadB)}, {"x", btn(button::kGamepadX)},
    {"y", btn(button::kGamepadY)}, {"lb", btn(button::kLeftBumper)}, {"rb", btn(button::kRightBumper)},
    {"menu", btn(button::kMenu)}, {"view", btn(button::kView)}, {"l3", btn(button::kLeftStickPress)},
    {"r3", btn(button::kRightStickPress)}, {"dpad_up", btn(button::kDPadUp)}, {"dpad_down", btn(button::kDPadDown)},
    {"dpad_left", btn(button::kDPadLeft)}, {"dpad_right", btn(button::kDPadRight)},
    {"left_stick_up", ax(Axis::LeftY, 1)}, {"left_stick_down", ax(Axis::LeftY, -1)},
    {"left_stick_left", ax(Axis::LeftX, -1)}, {"left_stick_right", ax(Axis::LeftX, 1)},
    {"right_stick_up", ax(Axis::RightY, 1)}, {"right_stick_down", ax(Axis::RightY, -1)},
    {"right_stick_left", ax(Axis::RightX, -1)}, {"right_stick_right", ax(Axis::RightX, 1)},
    {"left_trigger", ax(Axis::LeftTrigger, 1)}, {"right_trigger", ax(Axis::RightTrigger, 1)},
    {"none", Target{}},
};

bool target_from_name(const std::string& raw, Target& t) {
    const std::string n = lower(raw);
    for (const NamedTarget& k : kTargets) {
        if (n == k.name) {
            t = k.t;
            return true;
        }
    }
    return false;
}

struct DefaultKey {
    const char* key;
    const char* target;
};
const DefaultKey kDefaults[] = {
    {"W", "left_stick_up"}, {"S", "left_stick_down"}, {"A", "left_stick_left"}, {"D", "left_stick_right"},
    {"I", "right_stick_up"}, {"K", "right_stick_down"}, {"J", "right_stick_left"}, {"L", "right_stick_right"},
    {"Up", "dpad_up"}, {"Down", "dpad_down"}, {"Left", "dpad_left"}, {"Right", "dpad_right"},
    {"Space", "a"}, {"Enter", "a"}, {"Backspace", "b"}, {"R", "x"}, {"F", "y"}, {"Q", "lb"}, {"E", "rb"},
    {"Escape", "menu"}, {"Tab", "view"}, {"Shift", "l3"}, {"Ctrl", "r3"}, {"Z", "left_trigger"}, {"C", "right_trigger"},
};

bool parse_bool(const std::string& v, bool& out) {
    const std::string n = lower(v);
    if (n == "on" || n == "true" || n == "yes" || n == "1") {
        out = true;
        return true;
    }
    if (n == "off" || n == "false" || n == "no" || n == "0") {
        out = false;
        return true;
    }
    return false;
}

void set_binding(Keymap& m, int vk, const Target& t) {
    for (Binding& b : m.keys) {
        if (b.vk == vk) {
            b.target = t;
            return;
        }
    }
    m.keys.push_back(Binding{vk, t});
}

}  // namespace

Keymap default_keymap() {
    Keymap m;
    for (const DefaultKey& d : kDefaults) {
        int vk = 0;
        Target t;
        key_from_name(d.key, vk);
        target_from_name(d.target, t);
        m.keys.push_back(Binding{vk, t});
    }
    return m;
}

std::string default_keymap_text() {
    std::string s =
        "# playgta6 native keymap (used by --host-input). Edit and restart.\n"
        "# Lines are \"Key = target\". This file is applied on top of the built-in table, so list only what you\n"
        "# change; \"Key = none\" unbinds a key. Comments start with # or ;.\n"
        "#\n"
        "# Keys: A-Z, 0-9, F1-F12, Numpad0-9, Up Down Left Right, Space Enter Backspace Tab Escape, Shift Ctrl Alt\n"
        "#       (LShift RShift LCtrl RCtrl LAlt RAlt), Insert Delete Home End PageUp PageDown, Comma Period Slash\n"
        "#       Semicolon Quote Minus Equals Backtick LBracket RBracket Backslash, Mouse_Left Mouse_Right\n"
        "#       Mouse_Middle Mouse_X1 Mouse_X2.\n"
        "# Targets: a b x y lb rb menu view l3 r3 dpad_up dpad_down dpad_left dpad_right, left_stick_up/down/left/right,\n"
        "#          right_stick_up/down/left/right, left_trigger right_trigger, none.\n"
        "# Each key can be bound once per file. Keys and mouse only act while the game window has focus.\n"
        "# [hotkeys] rebinds the live window's own keys (value: Key or Ctrl+Key); two actions cannot share a key,\n"
        "# and a plain hotkey cannot reuse a key bound in [keys].\n"
        "\n[keys]\n";
    for (const DefaultKey& d : kDefaults) {
        s += std::string(d.key) + " = " + d.target + "\n";
    }
    s +=
        "\n[mouse]\n"
        "# Real mouse movement drives the right stick. In the live window the cursor is captured (Raw Input,\n"
        "# hidden and confined to the window) while it has focus; release_key toggles the capture, a click\n"
        "# in the window re-captures. Focus loss and minimising release it too.\n"
        "look = off\n"
        "# Stick deflection per pixel moved per 8 ms poll (0.001 .. 10).\n"
        "sensitivity = 0.05\n"
        "invert_y = off\n"
        "# release_key is kept for old files; it is the same as [hotkeys] mouse_release.\n"
        "release_key = F8\n"
        "\n[hotkeys]\n";
    for (const auto& h : playgta6::os::window_hotkeys::kTable) {
        s += std::string("# ") + h.help + "\n";
        s += std::string(h.name) + " = " + playgta6::os::window_hotkeys::chord_name(h.def) + "\n";
    }
    return s;
}

static std::string dup_key_hint(const playgta6::os::window_hotkeys::Table& t) {
    namespace wh = playgta6::os::window_hotkeys;
    for (size_t a = 0; a < wh::kCount; ++a)
        for (size_t b = a + 1; b < wh::kCount; ++b)
            if (t[a] == t[b]) return wh::chord_name(t[a]);
    return "";
}

bool parse_keymap(const std::string& text, Keymap& out, std::vector<std::string>& errors) {
    out = default_keymap();
    errors.clear();
    enum class Section { None, Keys, Mouse, Hotkeys } section = Section::None;
    std::map<int, int> seen;  // vk -> line first bound in this file
    int line_no = 0;
    size_t pos = 0;
    auto err = [&](const std::string& m) { errors.push_back("line " + std::to_string(line_no) + ": " + m); };
    while (pos <= text.size()) {
        size_t end = text.find('\n', pos);
        if (end == std::string::npos) end = text.size();
        std::string line = text.substr(pos, end - pos);
        pos = end + 1;
        ++line_no;
        const size_t c = line.find_first_of("#;");
        if (c != std::string::npos) line.erase(c);
        line = trim(line);
        if (line.empty()) continue;
        if (line.front() == '[') {
            const std::string name = line.back() == ']' ? lower(trim(line.substr(1, line.size() - 2))) : "";
            if (name == "keys") {
                section = Section::Keys;
            } else if (name == "mouse") {
                section = Section::Mouse;
            } else if (name == "hotkeys") {
                section = Section::Hotkeys;
            } else {
                err("unknown section '" + line + "' (expected [keys], [mouse] or [hotkeys])");
                section = Section::None;
            }
            continue;
        }
        const size_t eq = line.find('=');
        if (eq == std::string::npos) {
            err("expected 'name = value', got '" + line + "'");
            continue;
        }
        const std::string key = trim(line.substr(0, eq));
        const std::string value = trim(line.substr(eq + 1));
        if (key.empty() || value.empty()) {
            err("missing name or value in '" + line + "'");
            continue;
        }
        if (section == Section::None) {
            err("'" + line + "' is outside a [keys], [mouse] or [hotkeys] section");
            continue;
        }
        if (section == Section::Keys) {
            int vk = 0;
            Target t;
            if (!key_from_name(key, vk)) {
                err("unknown key '" + key + "'");
                continue;
            }
            if (!target_from_name(value, t)) {
                err("unknown target '" + value + "' for key " + key);
                continue;
            }
            const auto prev = seen.find(vk);
            if (prev != seen.end()) {
                err("key '" + key + "' is already bound on line " + std::to_string(prev->second));
                continue;
            }
            seen[vk] = line_no;
            set_binding(out, vk, t);
        } else if (section == Section::Hotkeys) {
            namespace wh = playgta6::os::window_hotkeys;
            const wh::Info* info = wh::find_by_name(lower(key));
            if (!info) {
                std::string names;
                for (const auto& h : wh::kTable) names += std::string(names.empty() ? "" : ", ") + h.name;
                err("unknown hotkey '" + key + "' (expected " + names + ")");
                continue;
            }
            std::string keyname = value;
            bool ctrl = false;
            if (lower(keyname).rfind("ctrl+", 0) == 0) {
                ctrl = true;
                keyname = trim(keyname.substr(5));
            }
            int vk = 0;
            if (!key_from_name(keyname, vk)) {
                err("unknown key '" + value + "' for hotkey " + key);
                continue;
            }
            out.hotkeys[static_cast<size_t>(info->action)] = wh::Chord{vk, ctrl};
        } else {
            const std::string k = lower(key);
            if (k == "look" || k == "invert_y") {
                bool b = false;
                if (!parse_bool(value, b)) {
                    err(k + " must be on or off, got '" + value + "'");
                    continue;
                }
                (k == "look" ? out.mouse_look : out.mouse_invert_y) = b;
            } else if (k == "sensitivity") {
                char* e = nullptr;
                const float f = std::strtof(value.c_str(), &e);
                if (e == value.c_str() || *e != '\0' || !(f >= 0.001f && f <= 10.0f)) {
                    err("sensitivity must be a number from 0.001 to 10, got '" + value + "'");
                    continue;
                }
                out.mouse_sensitivity = f;
            } else if (k == "release_key") {
                int vk = 0;
                if (!key_from_name(value, vk)) {
                    err("unknown key '" + value + "' for release_key");
                    continue;
                }
                out.hotkeys[static_cast<size_t>(playgta6::os::window_hotkeys::Action::MouseRelease)] =
                    playgta6::os::window_hotkeys::Chord{vk, false};
            } else {
                err("unknown mouse setting '" + key + "' (expected look, sensitivity, invert_y, release_key)");
            }
        }
    }
    if (errors.empty()) {
        namespace wh = playgta6::os::window_hotkeys;
        const std::string dup = wh::find_duplicate(out.hotkeys);
        if (!dup.empty()) errors.push_back("[hotkeys]: " + dup + " (" + dup_key_hint(out.hotkeys) + ")");
        for (size_t i = 0; i < wh::kCount; ++i) {
            const wh::Chord& c = out.hotkeys[i];
            if (c.ctrl) continue;
            for (const Binding& b : out.keys) {
                if (b.vk == c.vk && !b.target.none()) {
                    errors.push_back(std::string("[hotkeys]: ") + wh::kTable[i].name + " uses " + wh::chord_name(c) +
                                     ", which is also bound in [keys]");
                    break;
                }
            }
        }
    }
    return errors.empty();
}

bool load_keymap(const std::string& explicit_path, Keymap& out, std::vector<std::string>& errors, std::string& used_path) {
    namespace fs = std::filesystem;
    errors.clear();
    fs::path path;
    bool create_default = false;
    if (!explicit_path.empty()) {
        path = explicit_path;
    } else {
        const char* local = std::getenv("LOCALAPPDATA");
        if (local == nullptr || *local == '\0') {
            out = default_keymap();  // no per-user location on this host: built-in table
            used_path.clear();
            return true;
        }
        path = fs::path(local) / "playgta6" / "keymap.ini";
        create_default = true;
    }
    used_path = path.string();
    std::error_code ec;
    if (!fs::exists(path, ec)) {
        if (!create_default) {
            errors.push_back("keymap file not found: " + used_path);
            return false;
        }
        fs::create_directories(path.parent_path(), ec);
        std::ofstream f(path, std::ios::binary);
        if (f) {
            f << default_keymap_text();
        }  // best effort: a read-only profile just runs the defaults
        out = default_keymap();
        return true;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        errors.push_back("cannot read keymap file: " + used_path);
        return false;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    if (!parse_keymap(ss.str(), out, errors)) {
        for (std::string& e : errors) {
            e = used_path + ": " + e;
        }
        return false;
    }
    return true;
}

}  // namespace playgta6::guest::keymap
