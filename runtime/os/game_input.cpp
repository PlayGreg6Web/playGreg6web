#include "os/game_input.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>

namespace playgta6::os {

static_assert(sizeof(playgta6::InputState) == 40, "InputState must remain 40 bytes");
static_assert(alignof(playgta6::InputState) >= std::atomic_ref<uint32_t>::required_alignment,
              "InputState alignment must satisfy atomic_ref<uint32_t>");
static_assert(alignof(playgta6::InputState) >= std::atomic_ref<int32_t>::required_alignment,
              "InputState alignment must satisfy atomic_ref<int32_t>");
static_assert(alignof(playgta6::InputState) >= std::atomic_ref<float>::required_alignment,
              "InputState alignment must satisfy atomic_ref<float>");

Vec2 normalize_stick(float x, float y, float inner_deadzone, float outer_deadzone) {
    if (!std::isfinite(x) || !std::isfinite(y)) {
        return Vec2{0.0f, 0.0f};
    }

    const float inner = std::clamp(inner_deadzone, 0.0f, 0.99f);
    const float outer = std::clamp(outer_deadzone, inner + 1e-4f, 1.0f);

    const float mag = std::hypot(x, y);
    if (mag <= inner || mag <= 0.0f) {
        return Vec2{0.0f, 0.0f};
    }

    const float clamped_mag = std::min(mag, outer);
    const float normalized_mag = std::clamp((clamped_mag - inner) / (outer - inner), 0.0f, 1.0f);
    const float scale = normalized_mag / mag;
    return Vec2{x * scale, y * scale};
}

float normalize_trigger(float value, float inner_deadzone, float outer_deadzone) {
    if (!std::isfinite(value)) {
        return 0.0f;
    }

    const float inner = std::clamp(inner_deadzone, 0.0f, 0.99f);
    const float outer = std::clamp(outer_deadzone, inner + 1e-4f, 1.0f);

    if (value <= inner) {
        return 0.0f;
    }
    if (value >= outer) {
        return 1.0f;
    }
    return std::clamp((value - inner) / (outer - inner), 0.0f, 1.0f);
}

GameInputReader::GameInputReader(playgta6::InputState* shared_state,
                                 SeqlockMode mode,
                                 DeadzoneConfig deadzone)
    : mode_(mode), deadzone_(deadzone) {
    attach(shared_state, mode);
}

void GameInputReader::attach(playgta6::InputState* shared_state, SeqlockMode mode) {
    mode_ = mode;
    if (shared_state != nullptr &&
        (reinterpret_cast<uintptr_t>(shared_state) % alignof(playgta6::InputState)) == 0) {
        state_ = shared_state;
    } else {
        state_ = nullptr;
    }
    prev_buttons_ = 0;
}

bool GameInputReader::read_snapshot(playgta6::InputState& out,
                                    uint32_t max_retries,
                                    uint32_t* retries_used) const {
    if (state_ == nullptr) {
        if (retries_used != nullptr) {
            *retries_used = 0;
        }
        return false;
    }

    const uint32_t attempts = max_retries == 0 ? 1u : max_retries;
    std::atomic_ref<uint32_t> seq_ref(state_->seq);
    std::atomic_ref<uint32_t> buttons_ref(state_->buttons);
    std::atomic_ref<int32_t> mouse_dx_ref(state_->mouse_dx);
    std::atomic_ref<int32_t> mouse_dy_ref(state_->mouse_dy);
    std::atomic_ref<float> lx_ref(state_->left_stick_x);
    std::atomic_ref<float> ly_ref(state_->left_stick_y);
    std::atomic_ref<float> rx_ref(state_->right_stick_x);
    std::atomic_ref<float> ry_ref(state_->right_stick_y);
    std::atomic_ref<float> lt_ref(state_->left_trigger);
    std::atomic_ref<float> rt_ref(state_->right_trigger);

    for (uint32_t attempt = 0; attempt < attempts; ++attempt) {
        const uint32_t seq_before = seq_ref.load(std::memory_order_acquire);
        if (mode_ == SeqlockMode::EvenOddSeqlock && (seq_before & 1u) != 0u) {
            continue;
        }

        playgta6::InputState snap{};
        snap.seq = seq_before;
        snap.buttons = buttons_ref.load(std::memory_order_relaxed);
        snap.mouse_dx = mouse_dx_ref.load(std::memory_order_relaxed);
        snap.mouse_dy = mouse_dy_ref.load(std::memory_order_relaxed);
        snap.left_stick_x = lx_ref.load(std::memory_order_relaxed);
        snap.left_stick_y = ly_ref.load(std::memory_order_relaxed);
        snap.right_stick_x = rx_ref.load(std::memory_order_relaxed);
        snap.right_stick_y = ry_ref.load(std::memory_order_relaxed);
        snap.left_trigger = lt_ref.load(std::memory_order_relaxed);
        snap.right_trigger = rt_ref.load(std::memory_order_relaxed);

        std::atomic_thread_fence(std::memory_order_acquire);
        const uint32_t seq_after = seq_ref.load(std::memory_order_relaxed);

        if (seq_before == seq_after) {
            out = snap;
            if (retries_used != nullptr) {
                *retries_used = attempt;
            }
            return true;
        }
    }

    if (retries_used != nullptr) {
        *retries_used = attempts;
    }
    return false;
}

MouseDelta GameInputReader::consume_mouse_delta() {
    if (state_ == nullptr) {
        return MouseDelta{0, 0};
    }
    std::atomic_ref<int32_t> dx_ref(state_->mouse_dx);
    std::atomic_ref<int32_t> dy_ref(state_->mouse_dy);
    MouseDelta delta;
    delta.dx = dx_ref.exchange(0, std::memory_order_acq_rel);
    delta.dy = dy_ref.exchange(0, std::memory_order_acq_rel);
    return delta;
}

uint32_t GameInputReader::buttons() const {
    if (state_ == nullptr) {
        return 0;
    }
    return std::atomic_ref<uint32_t>(state_->buttons).load(std::memory_order_acquire);
}

bool GameInputReader::poll(GameInputFrame& out, bool consume_mouse, uint32_t max_retries) {
    playgta6::InputState snap{};
    if (!read_snapshot(snap, max_retries)) {
        out = GameInputFrame{};
        out.consistent = false;
        return false;
    }

    out.seq = snap.seq;
    out.buttons = snap.buttons;
    out.buttons_pressed = compute_buttons_pressed(prev_buttons_, snap.buttons);
    out.buttons_released = compute_buttons_released(prev_buttons_, snap.buttons);
    prev_buttons_ = snap.buttons;

    if (consume_mouse) {
        out.mouse_delta = consume_mouse_delta();
    } else {
        out.mouse_delta = MouseDelta{snap.mouse_dx, snap.mouse_dy};
    }

    out.left_stick = normalize_stick(
        snap.left_stick_x, snap.left_stick_y, deadzone_.stick_inner, deadzone_.stick_outer);
    out.right_stick = normalize_stick(
        snap.right_stick_x, snap.right_stick_y, deadzone_.stick_inner, deadzone_.stick_outer);
    out.left_trigger = normalize_trigger(
        snap.left_trigger, deadzone_.trigger_inner, deadzone_.trigger_outer);
    out.right_trigger = normalize_trigger(
        snap.right_trigger, deadzone_.trigger_inner, deadzone_.trigger_outer);
    out.consistent = true;
    return true;
}

void GameInputReader::write_monotonic(playgta6::InputState& dst,
                                      const playgta6::InputState& values,
                                      bool accumulate_mouse) {
    std::atomic_ref<uint32_t>(dst.buttons).store(values.buttons, std::memory_order_relaxed);
    if (accumulate_mouse) {
        std::atomic_ref<int32_t>(dst.mouse_dx).fetch_add(values.mouse_dx, std::memory_order_relaxed);
        std::atomic_ref<int32_t>(dst.mouse_dy).fetch_add(values.mouse_dy, std::memory_order_relaxed);
    } else {
        std::atomic_ref<int32_t>(dst.mouse_dx).store(values.mouse_dx, std::memory_order_relaxed);
        std::atomic_ref<int32_t>(dst.mouse_dy).store(values.mouse_dy, std::memory_order_relaxed);
    }
    std::atomic_ref<float>(dst.left_stick_x).store(values.left_stick_x, std::memory_order_relaxed);
    std::atomic_ref<float>(dst.left_stick_y).store(values.left_stick_y, std::memory_order_relaxed);
    std::atomic_ref<float>(dst.right_stick_x).store(values.right_stick_x, std::memory_order_relaxed);
    std::atomic_ref<float>(dst.right_stick_y).store(values.right_stick_y, std::memory_order_relaxed);
    std::atomic_ref<float>(dst.left_trigger).store(values.left_trigger, std::memory_order_relaxed);
    std::atomic_ref<float>(dst.right_trigger).store(values.right_trigger, std::memory_order_relaxed);

    std::atomic_ref<uint32_t>(dst.seq).fetch_add(1u, std::memory_order_release);
}

void GameInputReader::write_seqlock(playgta6::InputState& dst,
                                    const playgta6::InputState& values,
                                    bool accumulate_mouse) {
    std::atomic_ref<uint32_t> seq_ref(dst.seq);
    const uint32_t cur = seq_ref.load(std::memory_order_relaxed);
    const uint32_t odd_seq = (cur & 1u) != 0u ? cur : (cur + 1u);
    seq_ref.store(odd_seq, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);

    std::atomic_ref<uint32_t>(dst.buttons).store(values.buttons, std::memory_order_relaxed);
    if (accumulate_mouse) {
        std::atomic_ref<int32_t>(dst.mouse_dx).fetch_add(values.mouse_dx, std::memory_order_relaxed);
        std::atomic_ref<int32_t>(dst.mouse_dy).fetch_add(values.mouse_dy, std::memory_order_relaxed);
    } else {
        std::atomic_ref<int32_t>(dst.mouse_dx).store(values.mouse_dx, std::memory_order_relaxed);
        std::atomic_ref<int32_t>(dst.mouse_dy).store(values.mouse_dy, std::memory_order_relaxed);
    }
    std::atomic_ref<float>(dst.left_stick_x).store(values.left_stick_x, std::memory_order_relaxed);
    std::atomic_ref<float>(dst.left_stick_y).store(values.left_stick_y, std::memory_order_relaxed);
    std::atomic_ref<float>(dst.right_stick_x).store(values.right_stick_x, std::memory_order_relaxed);
    std::atomic_ref<float>(dst.right_stick_y).store(values.right_stick_y, std::memory_order_relaxed);
    std::atomic_ref<float>(dst.left_trigger).store(values.left_trigger, std::memory_order_relaxed);
    std::atomic_ref<float>(dst.right_trigger).store(values.right_trigger, std::memory_order_relaxed);

    seq_ref.store(odd_seq + 1u, std::memory_order_release);
}

}  // namespace playgta6::os
