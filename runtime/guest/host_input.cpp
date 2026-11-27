#include "guest/host_input.h"
#include "os/thread_name.h"
#include "guest/input_record.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>

#include "os/game_input.h"
#include "os/raw_mouse.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace playgta6::guest::host_input {

namespace button = playgta6::os::input_button;

namespace {

float stick(int16_t v) {
    const float f = static_cast<float>(v) / 32767.0f;
    return f < -1.0f ? -1.0f : f;
}

float axis(bool positive, bool negative) {
    return (positive ? 1.0f : 0.0f) - (negative ? 1.0f : 0.0f);
}

}  // namespace

InputState convert_pad(const XPadSample& p) {
    InputState s{};
    struct Map {
        uint16_t x;
        uint32_t bit;
    };
    static const Map map[] = {
        {kXA, button::kGamepadA}, {kXB, button::kGamepadB}, {kXX, button::kGamepadX}, {kXY, button::kGamepadY},
        {kXLeftShoulder, button::kLeftBumper}, {kXRightShoulder, button::kRightBumper},
        {kXStart, button::kMenu}, {kXBack, button::kView},
        {kXLeftThumb, button::kLeftStickPress}, {kXRightThumb, button::kRightStickPress},
        {kXDpadUp, button::kDPadUp}, {kXDpadDown, button::kDPadDown},
        {kXDpadLeft, button::kDPadLeft}, {kXDpadRight, button::kDPadRight},
    };
    for (const Map& m : map) {
        if ((p.buttons & m.x) != 0) {
            s.buttons |= m.bit;
        }
    }
    s.left_stick_x = stick(p.thumb_lx);
    s.left_stick_y = stick(p.thumb_ly);
    s.right_stick_x = stick(p.thumb_rx);
    s.right_stick_y = stick(p.thumb_ry);
    s.left_trigger = static_cast<float>(p.left_trigger) / 255.0f;
    s.right_trigger = static_cast<float>(p.right_trigger) / 255.0f;
    return s;
}

InputState convert_keyboard(uint32_t k) {
    InputState s{};
    auto has = [&](uint32_t key) { return (k & key) != 0; };
    struct Map {
        uint32_t key;
        uint32_t bit;
    };
    static const Map map[] = {
        {kKeySpace, button::kGamepadA}, {kKeyEnter, button::kGamepadA}, {kKeyBackspace, button::kGamepadB},
        {kKeyR, button::kGamepadX}, {kKeyF, button::kGamepadY},
        {kKeyQ, button::kLeftBumper}, {kKeyE, button::kRightBumper},
        {kKeyEscape, button::kMenu}, {kKeyTab, button::kView},
        {kKeyShift, button::kLeftStickPress}, {kKeyCtrl, button::kRightStickPress},
        {kKeyUp, button::kDPadUp}, {kKeyDown, button::kDPadDown},
        {kKeyLeft, button::kDPadLeft}, {kKeyRight, button::kDPadRight},
    };
    for (const Map& m : map) {
        if (has(m.key)) {
            s.buttons |= m.bit;
        }
    }
    s.left_stick_x = axis(has(kKeyD), has(kKeyA));
    s.left_stick_y = axis(has(kKeyW), has(kKeyS));
    s.right_stick_x = axis(has(kKeyL), has(kKeyJ));
    s.right_stick_y = axis(has(kKeyI), has(kKeyK));
    s.left_trigger = has(kKeyZ) ? 1.0f : 0.0f;
    s.right_trigger = has(kKeyC) ? 1.0f : 0.0f;
    return s;
}

InputState merge(const InputState& pad, const InputState& kb) {
    auto pick = [](float a, float b) { return std::fabs(b) > std::fabs(a) ? b : a; };
    auto pick_stick = [&](float px, float py, float kx, float ky, float& ox, float& oy) {
        // Prevent sub-deadzone pad resting drift (|pad| <= 0.15) from bleeding into an active
        // keyboard/mouse stick axis and defeating radial deadzone downstream.
        constexpr float kPadDriftEps = 0.15f;
        const float pad_mag = std::hypot(px, py);
        const float kb_mag = std::hypot(kx, ky);
        if (kb_mag > 0.0f && pad_mag <= kPadDriftEps) {
            ox = kx;
            oy = ky;
            return;
        }
        ox = pick(px, kx);
        oy = pick(py, ky);
    };
    InputState s{};
    s.buttons = pad.buttons | kb.buttons;
    pick_stick(pad.left_stick_x, pad.left_stick_y, kb.left_stick_x, kb.left_stick_y, s.left_stick_x, s.left_stick_y);
    pick_stick(pad.right_stick_x, pad.right_stick_y, kb.right_stick_x, kb.right_stick_y, s.right_stick_x,
               s.right_stick_y);
    s.left_trigger = pad.left_trigger > kb.left_trigger ? pad.left_trigger : kb.left_trigger;
    s.right_trigger = pad.right_trigger > kb.right_trigger ? pad.right_trigger : kb.right_trigger;
    return s;
}

InputState compose_input(bool have_pad, const XPadSample& pad, bool focused, const InputState& keys) {
    if (!focused) return InputState{};
    return merge(have_pad ? convert_pad(pad) : InputState{}, keys);
}

void RumbleForwarder::send(uint16_t low, uint16_t high, uint64_t now_ms) {
    ++stats_.sends;
    if (fn_(static_cast<uint32_t>(pad_), low, high) != 0) {
        ++stats_.send_failures;
    }
    if (low == 0 && high == 0) {
        ++stats_.stops;
    }
    low_ = low;
    high_ = high;
    last_send_ms_ = now_ms;
    sent_any_ = true;
    running_ = low != 0 || high != 0;
}

void RumbleForwarder::update(const playgta6::os::RumbleTelemetry& t, int pad, bool allowed, uint64_t now_ms) {
    if (pad < 0) {
        // Pad gone: nothing to drive (the motors stop with the device); resume cleanly on reconnect.
        running_ = false;
        low_ = high_ = 0;
        pad_ = -1;
        return;
    }
    if (pad != pad_) {
        if (pad_ >= 0 && running_) {
            fn_(static_cast<uint32_t>(pad_), 0, 0);  // the pad we leave must not keep buzzing
            ++stats_.sends;
            ++stats_.stops;
        }
        pad_ = pad;
        low_ = high_ = 0;
        running_ = false;
    }
    auto motor = [](float v) {
        const float c = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        return static_cast<uint16_t>(c * 65535.0f + 0.5f);
    };
    const uint16_t low = allowed ? motor(t.last_low_frequency) : 0;
    const uint16_t high = allowed ? motor(t.last_high_frequency) : 0;
    if (allowed && (t.last_left_trigger > 0.0f || t.last_right_trigger > 0.0f)) {
        ++stats_.trigger_unsupported;
    }
    if (low == low_ && high == high_) {
        return;
    }
    const bool stopping = low == 0 && high == 0;
    if (!stopping && sent_any_ && now_ms >= last_send_ms_ && (now_ms - last_send_ms_) < min_interval_ms_) {
        ++stats_.rate_limited;
        return;
    }
    send(low, high, now_ms);
}

void RumbleForwarder::stop() {
    if (running_ && pad_ >= 0) {
        send(0, 0, last_send_ms_);
    }
}

namespace {
std::atomic<XInputSetStateFn> g_silence_fn{nullptr};
std::atomic<int> g_silence_pad{-1};
}  // namespace

void silence_host_rumble() {
    const XInputSetStateFn fn = g_silence_fn.load();
    const int pad = g_silence_pad.load();
    if (fn != nullptr && pad >= 0) {
        fn(static_cast<uint32_t>(pad), 0, 0);
    }
}

void HostInputSource::start() {
#ifdef _WIN32
    if (state_ == nullptr || thread_.joinable()) {
        return;
    }
    stop_.store(false);
    thread_ = std::thread([this] { run(); });
#endif
}

bool PadSelector::is_active(const XPadSample& p) {
    constexpr int kLeftDead = 7849, kRightDead = 8689, kTrigDead = 30;
    auto mag2 = [](int x, int y) { return static_cast<int64_t>(x) * x + static_cast<int64_t>(y) * y; };
    return p.buttons != 0 || p.left_trigger > kTrigDead || p.right_trigger > kTrigDead ||
           mag2(p.thumb_lx, p.thumb_ly) > static_cast<int64_t>(kLeftDead) * kLeftDead ||
           mag2(p.thumb_rx, p.thumb_ry) > static_cast<int64_t>(kRightDead) * kRightDead;
}

bool PadSelector::poll(uint64_t now_ms, int* slot, XPadSample* sample) {
    XPadSample cur[4];
    bool any = false;
    for (int i = 0; i < 4; ++i) {
        const bool due = connected_[i] || !scanned_[i] || !(active_ >= 0 || any) ||
                         (now_ms >= last_scan_ms_[i] && now_ms - last_scan_ms_[i] >= rescan_ms_);
        if (!due) {
            continue;
        }
        scanned_[i] = true;
        last_scan_ms_[i] = now_ms;
        cur[i] = XPadSample{};
        connected_[i] = read_(i, &cur[i]);
        any = any || connected_[i];
    }
    if (active_ >= 0 && !connected_[active_]) {
        active_ = -1;
    }
    // Activity on another pad takes over; the current pad keeps the seat while it is also active.
    if (active_ < 0 || !is_active(cur[active_])) {
        int pick = active_;
        for (int i = 0; i < 4; ++i) {
            if (connected_[i] && i != active_ && is_active(cur[i])) {
                pick = i;
                break;
            }
        }
        if (pick < 0) {
            for (int i = 0; i < 4; ++i) {
                if (connected_[i]) {
                    pick = i;
                    break;
                }
            }
        }
        active_ = pick;
    }
    if (active_ < 0) {
        return false;
    }
    *slot = active_;
    *sample = cur[active_];
    return true;
}

void HostInputSource::stop() {
    stop_.store(true);
    if (thread_.joinable()) {
        thread_.join();
    }
}

#ifdef _WIN32
namespace {
struct RawPad {
    WORD buttons;
    BYTE lt, rt;
    SHORT lx, ly, rx, ry;
};
struct RawState {
    DWORD packet;
    RawPad pad;
};
using GetStateFn = DWORD(WINAPI*)(DWORD, RawState*);
struct RawVibration {
    WORD left, right;
};
using SetStateFn = DWORD(WINAPI*)(DWORD, RawVibration*);
std::atomic<SetStateFn> g_set_state{nullptr};
uint32_t xinput_set_state(uint32_t pad, uint16_t low, uint16_t high) {
    RawVibration v{low, high};
    const SetStateFn fn = g_set_state.load(std::memory_order_acquire);
    return fn != nullptr ? fn(pad, &v) : 1;
}

bool key_down(int vk) { return (GetAsyncKeyState(vk) & 0x8000) != 0; }

bool input_state_payload_equal(const InputState& a, const InputState& b) {
    return a.buttons == b.buttons && a.mouse_dx == b.mouse_dx && a.mouse_dy == b.mouse_dy &&
           std::memcmp(&a.left_stick_x, &b.left_stick_x, sizeof(float) * 6) == 0;
}

// Holds discrete button taps for a minimum window so rapid 16-30 ms keyboard/pad taps are never
// dropped between low-FPS guest polls. Movement bits (WASD) stay unlatched for crisp release.
class ButtonHoldLatch {
public:
    static constexpr uint32_t kMovementMask =
        button::kMoveForward | button::kMoveBackward | button::kMoveLeft | button::kMoveRight;
    static constexpr uint64_t kDefaultHoldMs = 64;

    explicit ButtonHoldLatch(uint64_t hold_ms = kDefaultHoldMs) : hold_ms_(hold_ms) {}

    void reset() {
        prev_raw_ = 0;
        pressed_at_.fill(0);
        latch_until_.fill(0);
    }

    uint32_t apply(uint32_t raw_buttons, uint64_t now_ms) {
        if (hold_ms_ == 0) {
            prev_raw_ = raw_buttons;
            return raw_buttons;
        }
        const uint32_t discrete_raw = raw_buttons & ~kMovementMask;
        const uint32_t discrete_prev = prev_raw_ & ~kMovementMask;
        uint32_t latched = 0;
        for (uint32_t bit = 0; bit < 32; ++bit) {
            const uint32_t mask = 1u << bit;
            const bool is_down = (discrete_raw & mask) != 0;
            const bool was_down = (discrete_prev & mask) != 0;
            if (is_down) {
                if (!was_down) {
                    pressed_at_[bit] = now_ms;
                }
                latch_until_[bit] = 0;
            } else {
                if (was_down && now_ms >= pressed_at_[bit] && (now_ms - pressed_at_[bit]) < hold_ms_) {
                    latch_until_[bit] = pressed_at_[bit] + hold_ms_;
                }
                if (latch_until_[bit] != 0) {
                    if (now_ms < latch_until_[bit]) {
                        latched |= mask;
                    } else {
                        latch_until_[bit] = 0;
                    }
                }
            }
        }
        prev_raw_ = raw_buttons;
        return raw_buttons | latched;
    }

private:
    uint64_t hold_ms_;
    uint32_t prev_raw_ = 0;
    std::array<uint64_t, 32> pressed_at_{};
    std::array<uint64_t, 32> latch_until_{};
};

void clamp_look_unit_disk(float& x, float& y) {
    if (!std::isfinite(x) || !std::isfinite(y)) {
        x = 0.0f;
        y = 0.0f;
        return;
    }
    const float mag = std::hypot(x, y);
    if (mag > 1.0f && std::isfinite(mag)) {
        x /= mag;
        y /= mag;
    }
}
}  // namespace
#endif

void HostInputSource::run() {
    playgta6::os::name_this_thread(L"pg6-host-input");
#ifdef _WIN32
    GetStateFn get_state = nullptr;
    for (const char* dll : {"xinput1_4.dll", "xinput1_3.dll", "xinput9_1_0.dll"}) {
        if (HMODULE m = LoadLibraryA(dll)) {
            get_state = reinterpret_cast<GetStateFn>(GetProcAddress(m, "XInputGetState"));
            if (get_state != nullptr) {
                g_set_state.store(reinterpret_cast<SetStateFn>(GetProcAddress(m, "XInputSetState")),
                                  std::memory_order_release);
                break;
            }
        }
    }
    RumbleForwarder rumble(&xinput_set_state);
    if (g_set_state.load(std::memory_order_acquire) != nullptr) {
        g_silence_fn.store(&xinput_set_state);
    }
    int pad_index = -1;
    PadSelector selector([&](int slot, XPadSample* out) {
        RawState raw{};
        if (get_state == nullptr || get_state(static_cast<DWORD>(slot), &raw) != ERROR_SUCCESS) {
            return false;
        }
        out->buttons = raw.pad.buttons;
        out->left_trigger = raw.pad.lt;
        out->right_trigger = raw.pad.rt;
        out->thumb_lx = raw.pad.lx;
        out->thumb_ly = raw.pad.ly;
        out->thumb_rx = raw.pad.rx;
        out->thumb_ry = raw.pad.ry;
        return true;
    });
    POINT last_cursor{};
    bool have_cursor = false;
    bool prev_kb_enabled = keyboard_enabled_.load();
    ButtonHoldLatch tap_latch;
    ChordKeyFilter chord_filter;
    bool has_published = false;
    InputState last_published{};
    while (!stop_.load()) {
        const uint64_t now_ms = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count());
        XPadSample sample;
        bool have_pad = false;
        if (get_state != nullptr) {
            int slot = -1;
            have_pad = selector.poll(now_ms, &slot, &sample);
            pad_index = have_pad ? slot : -1;
        }
        const bool kb = keyboard_enabled_.load();
        if (g_set_state.load(std::memory_order_acquire) != nullptr) {
            // Rumble only while the window has focus (the same gate as the keyboard).
            rumble.update(playgta6::os::game_input_rumble_telemetry(), have_pad ? pad_index : -1, kb, now_ms);
            g_silence_pad.store(have_pad ? pad_index : -1);
        }
        if (!kb && prev_kb_enabled) {
            tap_latch.reset();
        }
        prev_kb_enabled = kb;

        float look_x = 0.0f, look_y = 0.0f;
        if (keymap_.mouse_look) {
            namespace rm = playgta6::os::raw_mouse;
            const bool win_ready = rm::state().window_ready.load();
            const bool capturing = win_ready ? rm::state().capturing.load() : true;
            if (!kb || !capturing) {
                if (win_ready) {
                    int32_t dx = 0, dy = 0;
                    rm::take(&dx, &dy);
                }
                look_x_ = 0.0f;
                look_y_ = 0.0f;
                have_cursor = false;
            } else {
                float mdx = 0.0f, mdy = 0.0f;
                if (win_ready) {
                    // Live window: relative Raw Input deltas.
                    int32_t dx = 0, dy = 0;
                    rm::take(&dx, &dy);
                    mdx = static_cast<float>(dx);
                    mdy = static_cast<float>(dy);
                } else {
                    // No window to capture the cursor: real cursor movement since the last poll.
                    POINT p{};
                    if (GetCursorPos(&p)) {
                        if (have_cursor) {
                            mdx = static_cast<float>(p.x - last_cursor.x);
                            mdy = static_cast<float>(p.y - last_cursor.y);
                        }
                        last_cursor = p;
                        have_cursor = true;
                    }
                }
                look_x = keymap::mouse_stick_step(look_x_, mdx, keymap_.mouse_sensitivity);
                look_y = keymap::mouse_stick_step(look_y_, keymap_.mouse_invert_y ? mdy : -mdy, keymap_.mouse_sensitivity);
                clamp_look_unit_disk(look_x, look_y);
                look_x_ = look_x;
                look_y_ = look_y;
            }
        }
        if (!kb) chord_filter.reset();
        const bool alt_now = key_down(0x12);
        auto chord_down = [&](int vk) { return chord_filter.down(vk, key_down(vk), alt_now); };
        const InputState keys = kb ? keymap::evaluate_look(keymap_, chord_down, look_x, look_y) : InputState{};
        InputState values = compose_input(have_pad, sample, kb, keys);
        values.buttons = tap_latch.apply(values.buttons, now_ms);
        if (recorder_ != nullptr) {
            recorder_->record(values);
        }
        if (!has_published || !input_state_payload_equal(values, last_published)) {
            playgta6::os::GameInputReader::write_seqlock(*state_, values);
            last_published = values;
            has_published = true;
            writes_.fetch_add(1);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(8));
    }
    rumble.stop();
    g_silence_pad.store(-1);
    const InputState zero{};
    if (!has_published || !input_state_payload_equal(zero, last_published)) {
        playgta6::os::GameInputReader::write_seqlock(*state_, zero);
        writes_.fetch_add(1);
    }
#endif
}

}  // namespace playgta6::guest::host_input
