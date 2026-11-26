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
//
// Screen-synchronised steps (R40): a run that is slower than usual (a payload GPU capture, a busy
// machine) shifts every screen, so wall-clock presses land on the wrong one. These steps follow the
// live present counters instead (the newest `present_samples` entry, guest::present_status()):
//
//     wait <cond> [for <dur>] [timeout <dur>]
//         Blocks until <cond> has held continuously for <dur> (default 0: once), or <timeout>
//         passed (default none; the script then goes on and logs the timeout). The time of every
//         later press counts from the moment the wait ended.
//     repeat every <dur> until <cond> [for <dur>] [timeout <dur>]
//       <time_ms> <buttons> [hold_ms]   (times from the start of each round)
//       ...
//     end
//         Plays the body every <dur> until <cond> has held for its <dur> (or the timeout). No press
//         is sent while <cond> holds, so the body stops pressing once the target screen is up.
//     state <present> <buttons> <left_x> <left_y> <right_x> <right_y> <lt> <rt>
//         A recorded host reading (`playgta6_boot --record-input`, guest/input_record.h; the file starts
//         with `# playgta6-input-script v2`): written as the whole input state once the present counter
//         has reached <present>, and held until the next state line. <buttons> is a number (hex with
//         0x), axes and triggers are floats (sticks up = +y). Replays by present index, not wall time.
//     capture
//         Starts a deferred GPU capture here (playgta6_boot --gpu-capture-start-at-script).
//
// <cond> is one token `<metric><op><values>`: metric `draws_per_frame` (draws of the newest
// present) or `presents` (presents so far); op ==, !=, >=, <=, >, <; `==` and `!=` take
// alternatives separated by `|` (draws_per_frame==33|50: either). <dur> is `500ms`, `2s`, `1.5s`
// or plain milliseconds. Without wait/repeat steps the script plays exactly as before.

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "abi/playgta6_abi.h"

namespace playgta6::guest {

enum class InputStepKind : uint8_t { Press, Wait, RepeatBegin, RepeatEnd, Capture, State };

struct InputCondition {
    enum class Metric : uint8_t { DrawsPerFrame, Presents };
    enum class Op : uint8_t { Eq, Ne, Ge, Le, Gt, Lt };
    Metric metric = Metric::DrawsPerFrame;
    Op op = Op::Eq;
    std::vector<uint64_t> values;  // >= 1; several only for Eq / Ne
};

// The live state a condition reads (guest::present_status() in the harness).
struct InputStatus {
    uint64_t presents = 0;
    uint64_t draws_per_frame = 0;
};
using InputStatusProbe = std::function<InputStatus()>;

bool condition_holds(const InputCondition& cond, const InputStatus& status);
std::string condition_text(const InputCondition& cond);

// "Held continuously for `for_ms`": feed it the status at increasing times; met() once the
// condition has held from some sample through `now_ms` for at least `for_ms`.
class ConditionTracker {
public:
    ConditionTracker(const InputCondition& cond, uint64_t for_ms) : cond_(&cond), for_ms_(for_ms) {}
    // Returns met().
    bool update(uint64_t now_ms, const InputStatus& status);
    bool met() const { return met_; }
    bool holds_now() const { return holds_; }

private:
    const InputCondition* cond_;
    uint64_t for_ms_;
    bool holds_ = false;
    bool met_ = false;
    uint64_t since_ms_ = 0;
};

struct InputScriptEvent {
    uint64_t time_ms = 0;
    uint32_t buttons = 0;
    uint64_t hold_ms = 100;
    // Analog axes held for the event (up = +y, like the shared InputState); 0 when not named.
    float left_stick_x = 0.0f;  // State only (recorded readings)
    float left_stick_y = 0.0f;
    float right_stick_x = 0.0f;
    float right_stick_y = 0.0f;
    float left_trigger = 0.0f;
    float right_trigger = 0.0f;
    // Screen-synchronised steps (everything but Press).
    InputStepKind kind = InputStepKind::Press;
    InputCondition cond;      // Wait, RepeatBegin
    uint64_t for_ms = 0;      // Wait, RepeatBegin: the condition must hold this long
    uint64_t timeout_ms = 0;  // Wait, RepeatBegin: 0 = none
    uint64_t every_ms = 0;    // RepeatBegin: round period
    uint64_t present = 0;     // State: written once the present counter reaches this
    size_t end_index = 0;     // RepeatBegin: index of its RepeatEnd
};

// Parses a script. Returns false with `error` set ("line N: ...") on the first malformed line.
// Presses come back sorted by time within each run between two other steps (all of them when the
// script has none).
bool parse_input_script(const std::string& text, std::vector<InputScriptEvent>& out, std::string& error);

// Plays events into `state` on its own thread. Each event writes its buttons, then clears them
// after its hold time. stop() (or destruction) ends playback early.
class InputScriptPlayer {
public:
    InputScriptPlayer(std::vector<InputScriptEvent> events, playgta6::InputState* state);
    ~InputScriptPlayer();
    InputScriptPlayer(const InputScriptPlayer&) = delete;
    InputScriptPlayer& operator=(const InputScriptPlayer&) = delete;

    // Live counters for wait/repeat (without one they see a zero status); the capture step's action.
    void set_status_probe(InputStatusProbe probe) { probe_ = std::move(probe); }
    void set_on_capture(std::function<void()> on_capture) { on_capture_ = std::move(on_capture); }
    // Log presses and wait results to stderr ("[input] <ms> ms: ...").
    void set_log(bool log) { log_ = log; }

    void start();
    void stop();
    // Button writes done so far (presses and releases).
    uint64_t writes() const { return writes_.load(); }
    // Wait / repeat steps ended so far, and how many of them by timeout.
    uint64_t waits_done() const { return waits_done_.load(); }
    uint64_t waits_timed_out() const { return waits_timed_out_.load(); }
    // True once playback reached the end of the script.
    bool finished() const { return finished_.load(); }

private:
    void run();
    void write(const InputScriptEvent* held);
    InputStatus status() const { return probe_ ? probe_() : InputStatus{}; }

    std::vector<InputScriptEvent> events_;
    InputStatusProbe probe_;
    std::function<void()> on_capture_;
    bool log_ = false;
    std::atomic<uint64_t> waits_done_{0};
    std::atomic<uint64_t> waits_timed_out_{0};
    std::atomic<bool> finished_{false};
    playgta6::InputState* state_;
    std::thread thread_;
    std::atomic<bool> stop_{false};
    std::atomic<uint64_t> writes_{0};
};

}  // namespace playgta6::guest
