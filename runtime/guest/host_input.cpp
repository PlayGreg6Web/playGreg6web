#include "guest/host_input.h"
#include "guest/input_record.h"

#include <chrono>
#include <cmath>

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
    InputState s{};
    s.buttons = pad.buttons | kb.buttons;
    s.left_stick_x = pick(pad.left_stick_x, kb.left_stick_x);
    s.left_stick_y = pick(pad.left_stick_y, kb.left_stick_y);
    s.right_stick_x = pick(pad.right_stick_x, kb.right_stick_x);
    s.right_stick_y = pick(pad.right_stick_y, kb.right_stick_y);
    s.left_trigger = pad.left_trigger > kb.left_trigger ? pad.left_trigger : kb.left_trigger;
    s.right_trigger = pad.right_trigger > kb.right_trigger ? pad.right_trigger : kb.right_trigger;
    return s;
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
    if (!stopping && sent_any_ && now_ms - last_send_ms_ < min_interval_ms_) {
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
    stop_.store(false);
    thread_ = std::thread([this] { run(); });
#endif
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
SetStateFn g_set_state = nullptr;
uint32_t xinput_set_state(uint32_t pad, uint16_t low, uint16_t high) {
    RawVibration v{low, high};
    return g_set_state != nullptr ? g_set_state(pad, &v) : 1;
}

bool key_down(int vk) { return (GetAsyncKeyState(vk) & 0x8000) != 0; }
}  // namespace
#endif

void HostInputSource::run() {
#ifdef _WIN32
    GetStateFn get_state = nullptr;
    for (const char* dll : {"xinput1_4.dll", "xinput1_3.dll", "xinput9_1_0.dll"}) {
        if (HMODULE m = LoadLibraryA(dll)) {
            get_state = reinterpret_cast<GetStateFn>(GetProcAddress(m, "XInputGetState"));
            if (get_state != nullptr) {
                g_set_state = reinterpret_cast<SetStateFn>(GetProcAddress(m, "XInputSetState"));
                break;
            }
        }
    }
    RumbleForwarder rumble(&xinput_set_state);
    if (g_set_state != nullptr) {
        g_silence_fn.store(&xinput_set_state);
    }
    int pad_index = -1;
    POINT last_cursor{};
    bool have_cursor = false;
    while (!stop_.load()) {
        XPadSample sample;
        bool have_pad = false;
        if (get_state != nullptr) {
            // Stick to the connected pad; rescan the slots only while none is.
            for (int n = 0; n < 4 && !have_pad; ++n) {
                const int i = pad_index >= 0 ? pad_index : n;
                RawState raw{};
                if (get_state(static_cast<DWORD>(i), &raw) == ERROR_SUCCESS) {
                    pad_index = i;
                    have_pad = true;
                    sample.buttons = raw.pad.buttons;
                    sample.left_trigger = raw.pad.lt;
                    sample.right_trigger = raw.pad.rt;
                    sample.thumb_lx = raw.pad.lx;
                    sample.thumb_ly = raw.pad.ly;
                    sample.thumb_rx = raw.pad.rx;
                    sample.thumb_ry = raw.pad.ry;
                } else {
                    pad_index = -1;
                }
            }
        }
        if (g_set_state != nullptr) {
            // Rumble only while the window has focus (the same gate as the keyboard).
            const uint64_t now_ms = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
            rumble.update(playgta6::os::game_input_rumble_telemetry(), have_pad ? pad_index : -1, keyboard_enabled_.load(), now_ms);
            g_silence_pad.store(have_pad ? pad_index : -1);
        }
        float look_x = 0.0f, look_y = 0.0f;
        const bool kb = keyboard_enabled_.load();
        if (keymap_.mouse_look) {
            namespace rm = playgta6::os::raw_mouse;
            float mdx = 0.0f, mdy = 0.0f;
            if (kb && rm::state().window_ready.load()) {
                // Live window: relative Raw Input deltas (zero while the cursor is released).
                int32_t dx = 0, dy = 0;
                rm::take(&dx, &dy);
                mdx = static_cast<float>(dx);
                mdy = static_cast<float>(dy);
            } else if (kb) {
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
            } else {
                have_cursor = false;
            }
            look_x = keymap::mouse_stick_step(look_x_, mdx, keymap_.mouse_sensitivity);
            look_y = keymap::mouse_stick_step(look_y_, keymap_.mouse_invert_y ? mdy : -mdy, keymap_.mouse_sensitivity);
            look_x_ = look_x;
            look_y_ = look_y;
        }
        const InputState keys = kb ? keymap::evaluate_look(keymap_, key_down, look_x, look_y) : InputState{};
        const InputState values = merge(have_pad ? convert_pad(sample) : InputState{}, keys);
        if (recorder_ != nullptr) {
            recorder_->record(values);
        }
        playgta6::os::GameInputReader::write_monotonic(*state_, values);
        writes_.fetch_add(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(8));
    }
    rumble.stop();
    g_silence_pad.store(-1);
    const InputState zero{};
    playgta6::os::GameInputReader::write_monotonic(*state_, zero);
#endif
}

}  // namespace playgta6::guest::host_input
