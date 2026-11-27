#pragma once

// Real host input for the native boot harness (`playgta6_boot --host-input`).
//
// Polls the host's XInput pads and the keyboard and writes the result into the same
// playgta6::InputState the browser's input bridge fills, so the title reads it through the
// unchanged GameInput adapter. Raw values are passed through: the adapter applies the deadzones.
// Keyboard table: docs/PLAY_LOCALLY.md. Not used together with --input-script.

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>
#include <utility>

#include "abi/playgta6_abi.h"
#include "guest/keymap.h"
#include "os/game_input.h"
#include "os/raw_mouse.h"

namespace playgta6::guest {
class InputRecorder;
}

namespace playgta6::guest::host_input {

// XInput wButtons bits (XINPUT_GAMEPAD_*).
inline constexpr uint16_t kXDpadUp = 0x0001, kXDpadDown = 0x0002, kXDpadLeft = 0x0004, kXDpadRight = 0x0008;
inline constexpr uint16_t kXStart = 0x0010, kXBack = 0x0020, kXLeftThumb = 0x0040, kXRightThumb = 0x0080;
inline constexpr uint16_t kXLeftShoulder = 0x0100, kXRightShoulder = 0x0200;
inline constexpr uint16_t kXA = 0x1000, kXB = 0x2000, kXX = 0x4000, kXY = 0x8000;

struct XPadSample {
    uint16_t buttons = 0;
    uint8_t left_trigger = 0;
    uint8_t right_trigger = 0;
    int16_t thumb_lx = 0, thumb_ly = 0, thumb_rx = 0, thumb_ry = 0;
};

// Pad -> InputState fields (buttons, sticks in [-1,1] with up = +y, triggers in [0,1]). No deadzone.
InputState convert_pad(const XPadSample& pad);

// Picks which of the four XInput slots drives guest user 0: the most recently active pad (any
// button, trigger or stick beyond the deadzone). A connected pad is polled every tick; slots that
// are empty are rescanned every `rescan_ms` (every tick while no pad is connected at all).
class PadSelector {
public:
    // Reads one slot; returns false when nothing is connected there.
    using ReadFn = std::function<bool(int slot, XPadSample* out)>;
    explicit PadSelector(ReadFn fn, uint64_t rescan_ms = 1000) : read_(std::move(fn)), rescan_ms_(rescan_ms) {}

    static bool is_active(const XPadSample& p);

    // Polls the slots; returns true with the active slot and its sample, false when no pad is connected.
    bool poll(uint64_t now_ms, int* slot, XPadSample* sample);
    int active_slot() const { return active_; }

private:
    ReadFn read_;
    uint64_t rescan_ms_;
    int active_ = -1;
    bool connected_[4] = {};
    uint64_t last_scan_ms_[4] = {};
    bool scanned_[4] = {};
};

// Keyboard keys of the table, as bits of a mask.
enum HostKey : uint32_t {
    kKeyW = 1u << 0, kKeyA = 1u << 1, kKeyS = 1u << 2, kKeyD = 1u << 3,       // left stick
    kKeyI = 1u << 4, kKeyJ = 1u << 5, kKeyK = 1u << 6, kKeyL = 1u << 7,       // right stick
    kKeyUp = 1u << 8, kKeyDown = 1u << 9, kKeyLeft = 1u << 10, kKeyRight = 1u << 11,  // D-pad
    kKeySpace = 1u << 12, kKeyEnter = 1u << 13,   // A
    kKeyBackspace = 1u << 14,                      // B
    kKeyF = 1u << 15,                              // Y
    kKeyR = 1u << 16,                              // X
    kKeyQ = 1u << 17, kKeyE = 1u << 18,            // LB, RB
    kKeyEscape = 1u << 19,                         // Menu
    kKeyTab = 1u << 20,                            // View
    kKeyShift = 1u << 21,                          // L3
    kKeyCtrl = 1u << 22,                           // R3
    kKeyZ = 1u << 23, kKeyC = 1u << 24,            // LT, RT
};

// Window hotkey chords (Alt+Enter, Alt+Tab, ...) must not reach the guest keymap: a key that goes
// down while Alt is held is hidden until it is released (so Enter-up does not fire A either). Keys
// already held before Alt went down keep passing, and Alt itself is never hidden, so nothing sticks.
class ChordKeyFilter {
public:
    static bool is_alt(int vk) { return vk == 0x12 || vk == 0xA4 || vk == 0xA5; }
    // raw_down = physical state of vk now, alt_down = Alt physically held now.
    bool down(int vk, bool raw_down, bool alt_down) {
        if (vk < 0 || vk > 255) return raw_down;
        if (!raw_down) {
            hidden_[vk] = false;
            passing_[vk] = false;
            return false;
        }
        if (hidden_[vk]) return false;
        if (!passing_[vk] && alt_down && !is_alt(vk)) {
            hidden_[vk] = true;
            return false;
        }
        passing_[vk] = true;
        return true;
    }
    void reset() {
        hidden_.fill(false);
        passing_.fill(false);
    }

private:
    std::array<bool, 256> hidden_{};
    std::array<bool, 256> passing_{};
};

// Keyboard -> InputState (sticks full deflection, opposite keys cancel, triggers 0 or 1).
InputState convert_keyboard(uint32_t keys);

// Pad wins per axis when it is deflected more than the keyboard; buttons are ORed.
InputState merge(const InputState& pad, const InputState& keyboard);

// What the title sees: pad merged with keyboard/mouse, but only while the window has focus (`focused`);
// unfocused it is all zero so nothing sticks and a pad does not drive the game from another window.
InputState compose_input(bool have_pad, const XPadSample& pad, bool focused, const InputState& keys);

// Forwards the title's GameInput rumble requests (os::RumbleTelemetry::last_*) to the host pad's
// motors through XInputSetState. XInput has no trigger motors: non-zero trigger requests are only
// counted. Rate limited; a stop (both motors 0) is always sent at once. Pure logic over an injectable
// set-state function so it is testable without a pad.
using XInputSetStateFn = uint32_t (*)(uint32_t pad, uint16_t low_motor, uint16_t high_motor);  // 0 = ok

struct RumbleForwardStats {
    uint64_t sends = 0;                // XInputSetState calls that succeeded or failed
    uint64_t send_failures = 0;
    uint64_t rate_limited = 0;         // updates held back by the rate limit
    uint64_t trigger_unsupported = 0;  // updates with a non-zero trigger request (no XInput motor)
    uint64_t stops = 0;                // zero sends
};

class RumbleForwarder {
public:
    explicit RumbleForwarder(XInputSetStateFn fn, uint64_t min_interval_ms = 16) : fn_(fn), min_interval_ms_(min_interval_ms) {}
    // Call regularly. `pad` < 0 = no pad; `allowed` false (focus lost) forces the motors off.
    void update(const playgta6::os::RumbleTelemetry& t, int pad, bool allowed, uint64_t now_ms);
    // Motors off if they may be running (exit, fault, shutdown).
    void stop();
    const RumbleForwardStats& stats() const { return stats_; }

private:
    void send(uint16_t low, uint16_t high, uint64_t now_ms);

    XInputSetStateFn fn_;
    uint64_t min_interval_ms_;
    int pad_ = -1;          // pad the motors were last set on
    bool running_ = false;  // non-zero motors sent and not yet stopped
    uint16_t low_ = 0, high_ = 0;
    uint64_t last_send_ms_ = 0;
    bool sent_any_ = false;
    RumbleForwardStats stats_;
};

// Best-effort motors-off from any thread (fault/terminate path). No-op when nothing was driven.
void silence_host_rumble();

// Polls the host (Windows: the most recently active XInput pad plus GetAsyncKeyState) on its own
// thread and publishes into `state`. A no-op on other hosts.
class HostInputSource {
public:
    explicit HostInputSource(playgta6::InputState* state) : state_(state) {}
    ~HostInputSource() { stop(); }
    HostInputSource(const HostInputSource&) = delete;
    HostInputSource& operator=(const HostInputSource&) = delete;

    void start();
    void stop();
    uint64_t writes() const { return writes_.load(); }
    // The keyboard is read globally (GetAsyncKeyState); with a live window the harness enables it only
    // while that window has focus, so typing elsewhere does not drive the game. The pad is gated the same way (compose_input).
    void set_keyboard_enabled(bool enabled) { keyboard_enabled_.store(enabled); }
    // Every reading (after the keyboard gate and pad merge) is also handed to `recorder` (--record-input).
    void set_recorder(InputRecorder* recorder) { recorder_ = recorder; }
    // Rebinding table (keymap.h); call before start(). Without it the built-in table is used.
    void set_keymap(const keymap::Keymap& map) {
        keymap_ = map;
        auto& rm = playgta6::os::raw_mouse::state();
        playgta6::os::window_hotkeys::active().set(map.hotkeys);
        rm.look_enabled.store(map.mouse_look);
    }
    bool keyboard_enabled() const { return keyboard_enabled_.load(); }

private:
    void run();

    playgta6::InputState* state_;
    std::thread thread_;
    InputRecorder* recorder_ = nullptr;
    keymap::Keymap keymap_ = keymap::default_keymap();
    float look_x_ = 0.0f, look_y_ = 0.0f;  // decaying mouse-look stick (poll thread only)
    std::atomic<bool> stop_{false};
    std::atomic<bool> keyboard_enabled_{true};
    std::atomic<uint64_t> writes_{0};
};

}  // namespace playgta6::guest::host_input
