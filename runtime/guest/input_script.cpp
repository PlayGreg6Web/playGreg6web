#include "guest/input_script.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <sstream>

#include "os/game_input.h"

namespace playgta6::guest {

namespace {

bool parse_number(const std::string& text, uint64_t& out) {
    if (text.empty()) {
        return false;
    }
    char* end = nullptr;
    out = std::strtoull(text.c_str(), &end, 0);
    return end != nullptr && *end == '\0';
}

bool button_bit(const std::string& name, uint32_t& bit) {
    namespace b = playgta6::os::input_button;
    struct Name {
        const char* text;
        uint32_t bit;
    };
    static constexpr Name kNames[] = {
        {"A", b::kGamepadA},       {"B", b::kGamepadB},     {"X", b::kGamepadX},   {"Y", b::kGamepadY},
        {"LB", b::kLeftBumper},    {"RB", b::kRightBumper}, {"Menu", b::kMenu},    {"View", b::kView},
        // Left stick pushed fully (the bridge's W/A/S/D bits, which the GameInput layer turns
        // into the left thumbstick): menu navigation.
        {"Up", b::kMoveForward},   {"Down", b::kMoveBackward}, {"Left", b::kMoveLeft}, {"Right", b::kMoveRight},
        // D-pad and stick clicks (gamepad buttons only).
        {"DUp", b::kDPadUp},       {"DDown", b::kDPadDown},    {"DLeft", b::kDPadLeft}, {"DRight", b::kDPadRight},
        {"LS", b::kLeftStickPress}, {"RS", b::kRightStickPress},
    };
    for (const Name& n : kNames) {
        if (name == n.text) {
            bit = n.bit;
            return true;
        }
    }
    return false;
}

// Analog names (right stick, triggers): full deflection while held. False when `name` is not one.
bool analog_name(const std::string& name, InputScriptEvent& e) {
    if (name == "RUp") e.right_stick_y = 1.0f;
    else if (name == "RDown") e.right_stick_y = -1.0f;
    else if (name == "RLeft") e.right_stick_x = -1.0f;
    else if (name == "RRight") e.right_stick_x = 1.0f;
    else if (name == "LT") e.left_trigger = 1.0f;
    else if (name == "RT") e.right_trigger = 1.0f;
    else return false;
    return true;
}

bool parse_buttons(const std::string& text, InputScriptEvent& e) {
    uint32_t& out = e.buttons;
    uint64_t number = 0;
    if (parse_number(text, number)) {
        if (number == 0 || number > 0xffffffffull) {
            return false;
        }
        out = static_cast<uint32_t>(number);
        return true;
    }
    out = 0;
    bool any_analog = false;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t plus = text.find('+', start);
        const std::string part = text.substr(start, plus == std::string::npos ? std::string::npos : plus - start);
        uint32_t bit = 0;
        if (analog_name(part, e)) {
            any_analog = true;
        } else if (button_bit(part, bit)) {
            out |= bit;
        } else {
            return false;
        }
        if (plus == std::string::npos) {
            break;
        }
        start = plus + 1;
    }
    return out != 0 || any_analog;
}

}  // namespace

bool parse_input_script(const std::string& text, std::vector<InputScriptEvent>& out, std::string& error) {
    out.clear();
    std::istringstream lines(text);
    std::string line;
    size_t number = 0;
    while (std::getline(lines, line)) {
        ++number;
        if (const size_t hash = line.find('#'); hash != std::string::npos) {
            line.resize(hash);
        }
        std::istringstream fields(line);
        std::string time_text;
        if (!(fields >> time_text)) {
            continue;  // blank or comment
        }
        std::string buttons_text;
        std::string hold_text;
        std::string extra;
        fields >> buttons_text >> hold_text >> extra;
        InputScriptEvent e;
        const std::string where = "line " + std::to_string(number) + ": ";
        if (!parse_number(time_text, e.time_ms)) {
            error = where + "bad time '" + time_text + "'";
            return false;
        }
        if (!parse_buttons(buttons_text, e)) {
            error = where + "bad buttons '" + buttons_text + "'";
            return false;
        }
        if (!hold_text.empty() && (!parse_number(hold_text, e.hold_ms) || e.hold_ms == 0)) {
            error = where + "bad hold time '" + hold_text + "'";
            return false;
        }
        if (!extra.empty()) {
            error = where + "unexpected '" + extra + "'";
            return false;
        }
        out.push_back(e);
    }
    std::stable_sort(out.begin(), out.end(),
                     [](const InputScriptEvent& a, const InputScriptEvent& b) { return a.time_ms < b.time_ms; });
    return true;
}

InputScriptPlayer::InputScriptPlayer(std::vector<InputScriptEvent> events, playgta6::InputState* state)
    : events_(std::move(events)), state_(state) {}

InputScriptPlayer::~InputScriptPlayer() { stop(); }

void InputScriptPlayer::start() {
    if (state_ == nullptr || thread_.joinable()) {
        return;
    }
    stop_.store(false);
    thread_ = std::thread([this] { run(); });
}

void InputScriptPlayer::stop() {
    stop_.store(true);
    if (thread_.joinable()) {
        thread_.join();
    }
}

void InputScriptPlayer::write(const InputScriptEvent* held) {
    playgta6::InputState values{};
    if (held != nullptr) {
        values.buttons = held->buttons;
        values.right_stick_x = held->right_stick_x;
        values.right_stick_y = held->right_stick_y;
        values.left_trigger = held->left_trigger;
        values.right_trigger = held->right_trigger;
    }
    playgta6::os::GameInputReader::write_monotonic(*state_, values);
    writes_.fetch_add(1);
}

void InputScriptPlayer::run() {
    using clock = std::chrono::steady_clock;
    const clock::time_point t0 = clock::now();
    // Sleeps until `at` in short steps so stop() takes effect promptly; false when stopped.
    auto wait_until = [&](clock::time_point at) {
        while (!stop_.load()) {
            const clock::time_point now = clock::now();
            if (now >= at) {
                return true;
            }
            std::this_thread::sleep_for(std::min<clock::duration>(at - now, std::chrono::milliseconds(20)));
        }
        return false;
    };
    for (const InputScriptEvent& e : events_) {
        const clock::time_point press = t0 + std::chrono::milliseconds(e.time_ms);
        if (!wait_until(press)) {
            return;
        }
        write(&e);
        if (!wait_until(press + std::chrono::milliseconds(e.hold_ms))) {
            write(nullptr);
            return;
        }
        write(nullptr);
    }
}

}  // namespace playgta6::guest
