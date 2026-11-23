#pragma once

#include <cstdint>

#include "abi/playgta6_abi.h"

namespace playgta6::os {

// Button bit definitions matching web/input/input_bridge.js (bits 0..5) and Xbox GameInput extensions.
namespace input_button {
inline constexpr uint32_t kMoveForward = 1u << 0;   // KeyW
inline constexpr uint32_t kMoveBackward = 1u << 1;  // KeyS
inline constexpr uint32_t kMoveLeft = 1u << 2;      // KeyA
inline constexpr uint32_t kMoveRight = 1u << 3;     // KeyD
inline constexpr uint32_t kJump = 1u << 4;          // Space / A
inline constexpr uint32_t kSprint = 1u << 5;        // ShiftLeft / L3
inline constexpr uint32_t kGamepadA = 1u << 8;
inline constexpr uint32_t kGamepadB = 1u << 9;
inline constexpr uint32_t kGamepadX = 1u << 10;
inline constexpr uint32_t kGamepadY = 1u << 11;
inline constexpr uint32_t kLeftBumper = 1u << 12;
inline constexpr uint32_t kRightBumper = 1u << 13;
inline constexpr uint32_t kMenu = 1u << 14;
inline constexpr uint32_t kView = 1u << 15;
}  // namespace input_button

enum class SeqlockMode : uint8_t {
    // Matches docs/wasm_abi.md and web/input/input_bridge.js: seq is incremented after each write.
    // Reader retries whenever seq changes between start and end of snapshot.
    MonotonicSeq = 0,
    // Strict even/odd seqlock: writer bumps seq to odd before writing and to even after writing.
    // Reader retries whenever seq is odd or changes during snapshot.
    EvenOddSeqlock = 1,
};

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

struct MouseDelta {
    int32_t dx = 0;
    int32_t dy = 0;
};

struct DeadzoneConfig {
    float stick_inner = 0.15f;
    float stick_outer = 0.98f;
    float trigger_inner = 0.05f;
    float trigger_outer = 0.98f;
};

struct GameInputFrame {
    uint32_t seq = 0;
    uint32_t buttons = 0;
    uint32_t buttons_pressed = 0;   // Rising edges since previous poll
    uint32_t buttons_released = 0;  // Falling edges since previous poll
    MouseDelta mouse_delta{};
    Vec2 left_stick{};
    Vec2 right_stick{};
    float left_trigger = 0.0f;
    float right_trigger = 0.0f;
    bool consistent = false;
};

// Button bitmask helpers.
inline constexpr bool is_button_down(uint32_t buttons, uint32_t mask) {
    return (buttons & mask) == mask && mask != 0;
}

inline constexpr bool is_any_button_down(uint32_t buttons, uint32_t mask) {
    return (buttons & mask) != 0;
}

inline constexpr uint32_t compute_buttons_pressed(uint32_t prev, uint32_t curr) {
    return (~prev) & curr;
}

inline constexpr uint32_t compute_buttons_released(uint32_t prev, uint32_t curr) {
    return prev & (~curr);
}

// Radial stick deadzone normalization: maps magnitude in [inner, outer] -> [0, 1] while preserving angle.
Vec2 normalize_stick(float x,
                     float y,
                     float inner_deadzone = 0.15f,
                     float outer_deadzone = 0.98f);

// Linear trigger deadzone normalization: maps [inner, outer] -> [0, 1].
float normalize_trigger(float value,
                        float inner_deadzone = 0.05f,
                        float outer_deadzone = 0.98f);

// Lock-free snapshot reader over the 40-byte playgta6::InputState SharedArrayBuffer.
class GameInputReader {
public:
    explicit GameInputReader(playgta6::InputState* shared_state = nullptr,
                             SeqlockMode mode = SeqlockMode::MonotonicSeq,
                             DeadzoneConfig deadzone = {});

    void attach(playgta6::InputState* shared_state,
                SeqlockMode mode = SeqlockMode::MonotonicSeq);
    bool is_attached() const { return state_ != nullptr; }

    void set_deadzone(const DeadzoneConfig& config) { deadzone_ = config; }
    const DeadzoneConfig& deadzone() const { return deadzone_; }

    void set_seqlock_mode(SeqlockMode mode) { mode_ = mode; }
    SeqlockMode seqlock_mode() const { return mode_; }

    // Lock-free consistent read of all 40 bytes of playgta6::InputState.
    // Retries up to max_retries times if a concurrent write tears the snapshot.
    // Returns true if a consistent (untorn) snapshot was captured; false on timeout/unattached.
    bool read_snapshot(playgta6::InputState& out,
                       uint32_t max_retries = 64,
                       uint32_t* retries_used = nullptr) const;

    // Atomically reads and zeroes accumulated mouse_dx / mouse_dy from the shared buffer
    // (pairing with Atomics.add in web/input/input_bridge.js).
    MouseDelta consume_mouse_delta();

    // Reads the current button bitmask atomically.
    uint32_t buttons() const;

    // Captures a consistent snapshot, computes pressed/released button transitions since the
    // previous poll, optionally consumes accumulated mouse deltas, and normalizes sticks/triggers.
    bool poll(GameInputFrame& out, bool consume_mouse = true, uint32_t max_retries = 64);

    // Atomic writer helpers matching both MonotonicSeq and EvenOddSeqlock protocols (useful for
    // host integration and multi-threaded unit tests without data races).
    static void write_monotonic(playgta6::InputState& dst,
                                const playgta6::InputState& values,
                                bool accumulate_mouse = false);
    static void write_seqlock(playgta6::InputState& dst,
                              const playgta6::InputState& values,
                              bool accumulate_mouse = false);

private:
    playgta6::InputState* state_ = nullptr;
    SeqlockMode mode_ = SeqlockMode::MonotonicSeq;
    DeadzoneConfig deadzone_{};
    uint32_t prev_buttons_ = 0;
};

}  // namespace playgta6::os
