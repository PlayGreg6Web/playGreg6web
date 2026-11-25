#pragma once

// Scripted controller input for the native boot harness (`playgta6_boot --input-script <file>`).
//
// A test tool, not an input source of the runtime: it writes timed button states into the same
// playgta6::InputState the browser's input bridge fills, so the title reads them through the
// unchanged GameInput adapter. Without the option nothing is pressed.
//
// Script format, one event per line (blank lines and `#` comments ignored):
//
//     <time_ms> <buttons> [hold_ms]
//
// `time_ms` counts from the start of the player; `buttons` is `+`-separated names (A, B, X, Y,
// LB, RB, Menu, View, LS, RS; DUp, DDown, DLeft, DRight the D-pad; Up, Down, Left, Right push the
// left stick; RUp, RDown, RLeft, RRight push the right stick fully; LT, RT pull a trigger fully)
// or a number (hex with 0x) of
// playgta6::os::input_button bits; `hold_ms`
// (default 100) is how long the buttons stay down before they are released.

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "abi/playgta6_abi.h"

namespace playgta6::guest {

struct InputScriptEvent {
    uint64_t time_ms = 0;
    uint32_t buttons = 0;
    uint64_t hold_ms = 100;
    // Analog axes held for the event (up = +y, like the shared InputState); 0 when not named.
    float right_stick_x = 0.0f;
    float right_stick_y = 0.0f;
    float left_trigger = 0.0f;
    float right_trigger = 0.0f;
};

// Parses a script. Returns false with `error` set ("line N: ...") on the first malformed line.
// Events come back sorted by time.
bool parse_input_script(const std::string& text, std::vector<InputScriptEvent>& out, std::string& error);

// Plays events into `state` on its own thread. Each event writes its buttons, then clears them
// after its hold time. stop() (or destruction) ends playback early.
class InputScriptPlayer {
public:
    InputScriptPlayer(std::vector<InputScriptEvent> events, playgta6::InputState* state);
    ~InputScriptPlayer();
    InputScriptPlayer(const InputScriptPlayer&) = delete;
    InputScriptPlayer& operator=(const InputScriptPlayer&) = delete;

    void start();
    void stop();
    // Button writes done so far (presses and releases).
    uint64_t writes() const { return writes_.load(); }

private:
    void run();
    void write(const InputScriptEvent* held);

    std::vector<InputScriptEvent> events_;
    playgta6::InputState* state_;
    std::thread thread_;
    std::atomic<bool> stop_{false};
    std::atomic<uint64_t> writes_{0};
};

}  // namespace playgta6::guest
