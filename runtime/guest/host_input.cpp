#include "guest/host_input.h"

#include <chrono>
#include <cmath>

#include "os/game_input.h"

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

uint32_t read_keys() {
    struct K {
        int vk;
        uint32_t key;
    };
    static const K keys[] = {
        {'W', kKeyW}, {'A', kKeyA}, {'S', kKeyS}, {'D', kKeyD}, {'I', kKeyI}, {'J', kKeyJ}, {'K', kKeyK},
        {'L', kKeyL}, {VK_UP, kKeyUp}, {VK_DOWN, kKeyDown}, {VK_LEFT, kKeyLeft}, {VK_RIGHT, kKeyRight},
        {VK_SPACE, kKeySpace}, {VK_RETURN, kKeyEnter}, {VK_BACK, kKeyBackspace}, {'F', kKeyF}, {'R', kKeyR},
        {'Q', kKeyQ}, {'E', kKeyE}, {VK_ESCAPE, kKeyEscape}, {VK_TAB, kKeyTab}, {VK_SHIFT, kKeyShift},
        {VK_CONTROL, kKeyCtrl}, {'Z', kKeyZ}, {'C', kKeyC},
    };
    uint32_t mask = 0;
    for (const K& k : keys) {
        if ((GetAsyncKeyState(k.vk) & 0x8000) != 0) {
            mask |= k.key;
        }
    }
    return mask;
}
}  // namespace
#endif

void HostInputSource::run() {
#ifdef _WIN32
    GetStateFn get_state = nullptr;
    for (const char* dll : {"xinput1_4.dll", "xinput1_3.dll", "xinput9_1_0.dll"}) {
        if (HMODULE m = LoadLibraryA(dll)) {
            get_state = reinterpret_cast<GetStateFn>(GetProcAddress(m, "XInputGetState"));
            if (get_state != nullptr) {
                break;
            }
        }
    }
    int pad_index = -1;
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
        const InputState values = merge(have_pad ? convert_pad(sample) : InputState{}, convert_keyboard(keyboard_enabled_.load() ? read_keys() : 0));
        playgta6::os::GameInputReader::write_monotonic(*state_, values);
        writes_.fetch_add(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(8));
    }
    const InputState zero{};
    playgta6::os::GameInputReader::write_monotonic(*state_, zero);
#endif
}

}  // namespace playgta6::guest::host_input
