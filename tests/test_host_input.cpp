// Host input mapping (runtime/guest/host_input.h) and 3D gameplay GameInput tests (O53).
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <vector>

#include "abi/playgta6_abi.h"
#include "guest/com_bridge.h"
#include "guest/guest_callbacks.h"
#include "guest/guest_runtime.h"
#include "guest/host_input.h"
#include "os/game_input.h"
#include "os/vmem.h"
#include "os/window_hotkeys.h"
#include "os/win32_loader.h"
#include "os/xbox_scarlett.h"
#include "os/xbox_scarlett_guest.h"

namespace {
int g_failures = 0;
#define CHECK(cond)                                                                \
    do {                                                                           \
        if (!(cond)) {                                                             \
            std::fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                          \
        }                                                                          \
    } while (0)
bool near(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) < eps; }

using namespace playgta6;
using namespace playgta6::guest;
namespace hi = playgta6::guest::host_input;
namespace button = playgta6::os::input_button;
namespace com = playgta6::guest::com;
namespace cbk = playgta6::guest::callbacks;
namespace ga = playgta6::os::scarlett::guest_api;
namespace win32 = playgta6::os::win32;
using com::HResult;
using playgta6::os::VirtualMemoryArena;

// Exact GDK v0 GameInputMouseState layout (40 bytes) verified against xbox_scarlett_guest_input.cpp.
struct GuestMouseStateV0 {
    uint32_t buttons;
    int64_t position_x;
    int64_t position_y;
    int64_t wheel_x;
    int64_t wheel_y;
};
static_assert(sizeof(GuestMouseStateV0) == 40);

// ---------------------------------------------------------------------------
// Minimal guest-convention caller for COM GameInput tests
// ---------------------------------------------------------------------------

VirtualMemoryArena* g_arena = nullptr;

uint64_t load64(uint64_t va) {
    uint64_t value = 0;
    CHECK(g_arena->read(va, &value, sizeof(value)));
    return value;
}

void store64(uint64_t va, uint64_t value) {
    CHECK(g_arena->write(va, &value, sizeof(value)));
}

constexpr uint64_t kVCall = 0x140020000ULL;
constexpr uint64_t kGuestDeviceCb = 0x140020100ULL;

struct CallArg {
    uint64_t bits;
    bool is_float;
};

struct CallSpec {
    uint64_t target = 0;
    uint32_t slot = 0;
    std::vector<CallArg> args;
};

thread_local CallSpec t_call_spec;

void sub_vcall(GuestState* s) {
    static constexpr uint32_t kRegs[4] = {GuestReg::Rcx, GuestReg::Rdx, GuestReg::R8, GuestReg::R9};
    const size_t n = t_call_spec.args.size();
    const size_t stack_n = n > 4 ? n - 4 : 0;
    const uint64_t frame = ((0x20 + 8 * stack_n + 15) & ~uint64_t{15}) + 8;
    s->gpr[GuestReg::Rsp] -= frame;
    const uint64_t rsp = s->gpr[GuestReg::Rsp];
    for (size_t i = 0; i < n; ++i) {
        const CallArg& a = t_call_spec.args[i];
        if (i < 4) {
            if (a.is_float) {
                s->xmm[i][0] = a.bits;
            } else {
                s->gpr[kRegs[i]] = a.bits;
            }
        } else {
            store64(rsp + 0x20 + 8 * (i - 4), a.bits);
        }
    }
    uint64_t target = t_call_spec.target;
    if (target == 0) {
        target = load64(load64(s->gpr[GuestReg::Rcx]) + 8ull * t_call_spec.slot);
    }
    s->gpr[GuestReg::Rsp] -= 8;
    store64(s->gpr[GuestReg::Rsp], kVCall + 0x40);
    __playgta6_indirect_call(s, target);
    CHECK(s->gpr[GuestReg::Rsp] == rsp);
    s->gpr[GuestReg::Rsp] += frame + 8;
}

struct DeviceCbState {
    std::atomic<int> calls{0};
    uint64_t last_token = 0;
    uint64_t last_context = 0;
    uint64_t last_device = 0;
    uint32_t last_current = 0;
    std::vector<uint64_t> devices;
};
DeviceCbState g_cb;

void sub_device_cb(GuestState* s) {
    const uint64_t rsp = s->gpr[GuestReg::Rsp];
    g_cb.last_token = s->gpr[GuestReg::Rcx];
    g_cb.last_context = s->gpr[GuestReg::Rdx];
    g_cb.last_device = s->gpr[GuestReg::R8];
    g_cb.last_current = static_cast<uint32_t>(load64(rsp + kGuestStackArgOffset));
    g_cb.devices.push_back(g_cb.last_device);
    g_cb.calls.fetch_add(1);
    s->gpr[GuestReg::Rsp] += 8;
}

const LiftedFunctionEntry kTestFunctions[] = {
    {kVCall, sub_vcall},
    {kGuestDeviceCb, sub_device_cb},
};

struct TestFault {
    FaultInfo info;
};

void test_fault_handler(const FaultInfo* info) {
    throw TestFault{*info};
}

CallArg I(uint64_t v) { return CallArg{v, false}; }

HResult as_hr(uint64_t rax) { return static_cast<HResult>(static_cast<uint32_t>(rax)); }

struct GuestInputFixture {
    VirtualMemoryArena arena;
    GuestThread thread{};
    InputState shared{};

    GuestInputFixture() {
        reset_guest_runtime();
        com::reset_com_bridge();
        cbk::reset_callbacks();
        com::set_guest_memory(&arena);
        set_fault_handler(test_fault_handler);
        g_arena = &arena;
        CHECK(register_lifted_functions(kTestFunctions));
        CHECK(create_guest_thread(arena, 512 * 1024, thread));
        ga::register_xbox_scarlett_guest_exports();
        os::scarlett::set_game_input_state(&shared);
        os::reset_game_input_rumble_telemetry();
    }

    ~GuestInputFixture() {
        os::scarlett::set_game_input_state(nullptr);
        ga::reset_xbox_scarlett_guest();
        com::reset_com_bridge();
        cbk::reset_callbacks();
        destroy_guest_thread(arena, thread);
        com::set_guest_memory(nullptr);
        reset_guest_runtime();
        g_arena = nullptr;
    }

    uint64_t out_slot() { return com::guest_alloc(sizeof(uint64_t)); }

    uint64_t vcall(uint64_t obj, uint32_t slot, std::initializer_list<CallArg> args = {}) {
        t_call_spec = CallSpec{};
        t_call_spec.slot = slot;
        t_call_spec.args.push_back({obj, false});
        t_call_spec.args.insert(t_call_spec.args.end(), args.begin(), args.end());
        return call_guest(arena, thread.state, kVCall);
    }

    uint64_t dcall(uint64_t target, std::initializer_list<CallArg> args = {}) {
        t_call_spec = CallSpec{};
        t_call_spec.target = target;
        t_call_spec.args.assign(args.begin(), args.end());
        return call_guest(arena, thread.state, kVCall);
    }

    uint64_t create_game_input() {
        const uint64_t proc = pg6_loader_resolve_import("GameInput.dll", "GameInputCreate");
        CHECK(proc != 0);
        const uint64_t target = guest_callable_for_proc(proc);
        CHECK(target != 0);
        const uint64_t out = out_slot();
        CHECK(as_hr(dcall(target, {I(out)})) == 0);
        const uint64_t gi = load64(out);
        com::guest_free(out);
        return gi;
    }

    uint64_t current_reading(uint64_t gi, uint32_t kind, uint64_t dev = 0, HResult* hr_out = nullptr) {
        const uint64_t out = out_slot();
        const HResult hr = as_hr(vcall(gi, 4, {I(kind), I(dev), I(out)}));
        if (hr_out != nullptr) {
            *hr_out = hr;
        }
        const uint64_t r = load64(out);
        com::guest_free(out);
        return r;
    }

    uint64_t neighbour_reading(uint64_t gi, uint32_t slot, uint64_t ref, uint32_t kind, uint64_t dev = 0,
                               HResult* hr_out = nullptr) {
        const uint64_t out = out_slot();
        const HResult hr = as_hr(vcall(gi, slot, {I(ref), I(kind), I(dev), I(out)}));
        if (hr_out != nullptr) {
            *hr_out = hr;
        }
        const uint64_t r = load64(out);
        com::guest_free(out);
        return r;
    }

    ga::GameInputGamepadState read_gamepad(uint64_t r, bool* ok) {
        const uint64_t va = com::guest_alloc(sizeof(ga::GameInputGamepadState));
        *ok = (vcall(r, 22, {I(va)}) & 0xFF) != 0;
        ga::GameInputGamepadState s{};
        CHECK(com::read_guest(va, s));
        com::guest_free(va);
        return s;
    }

    GuestMouseStateV0 read_mouse(uint64_t r, bool* ok) {
        const uint64_t va = com::guest_alloc(sizeof(GuestMouseStateV0));
        *ok = (vcall(r, 16, {I(va)}) & 0xFF) != 0;
        GuestMouseStateV0 s{};
        CHECK(com::read_guest(va, s));
        com::guest_free(va);
        return s;
    }

    void write_state(const InputState& v, bool accumulate_mouse = false) {
        os::GameInputReader::write_monotonic(shared, v, accumulate_mouse);
    }
};

// ---------------------------------------------------------------------------
// 1. Original host_input mapping checks (preserved untouched)
// ---------------------------------------------------------------------------

void test_host_input_mapping() {
    hi::XPadSample p;
    p.buttons = hi::kXA | hi::kXStart | hi::kXDpadLeft | hi::kXRightThumb;
    p.left_trigger = 255;
    p.right_trigger = 0;
    p.thumb_lx = 32767;
    p.thumb_ly = -32768;
    p.thumb_rx = 1000;  // small value passes through: the adapter owns the deadzone
    p.thumb_ry = 0;
    const auto s = hi::convert_pad(p);
    CHECK(s.buttons == (button::kGamepadA | button::kMenu | button::kDPadLeft | button::kRightStickPress));
    CHECK(near(s.left_stick_x, 1.0f));
    CHECK(near(s.left_stick_y, -1.0f));
    CHECK(near(s.right_stick_x, 1000.0f / 32767.0f));
    CHECK(near(s.left_trigger, 1.0f) && near(s.right_trigger, 0.0f));

    const auto k = hi::convert_keyboard(hi::kKeyW | hi::kKeyD | hi::kKeySpace | hi::kKeyC | hi::kKeyTab);
    CHECK(near(k.left_stick_x, 1.0f) && near(k.left_stick_y, 1.0f));
    CHECK(k.buttons == (button::kGamepadA | button::kView));
    CHECK(near(k.right_trigger, 1.0f) && near(k.left_trigger, 0.0f));
    const auto c = hi::convert_keyboard(hi::kKeyA | hi::kKeyD);
    CHECK(near(c.left_stick_x, 0.0f));

    {
        // Focus loss releases everything (pad and keyboard); focused it is the merge.
        hi::XPadSample pad;
        pad.buttons = hi::kXA;
        pad.thumb_lx = 30000;
        const auto kk = hi::convert_keyboard(hi::kKeyW);
        const auto off = hi::compose_input(true, pad, false, kk);
        CHECK(off.buttons == 0 && near(off.left_stick_x, 0.0f) && near(off.left_stick_y, 0.0f));
        const auto on = hi::compose_input(true, pad, true, kk);
        CHECK(on.buttons == button::kGamepadA && on.left_stick_x > 0.5f && near(on.left_stick_y, 1.0f));
        CHECK(hi::compose_input(false, pad, true, InputState{}).buttons == 0);
        CHECK(playgta6::os::window_hotkeys::is_alt_enter(0x0D, true));
        CHECK(!playgta6::os::window_hotkeys::is_alt_enter(0x0D, false));
    }
    const auto m = hi::merge(s, k);
    CHECK(m.buttons == (s.buttons | k.buttons));
    CHECK(near(m.left_stick_x, 1.0f));
    CHECK(near(m.left_stick_y, -1.0f));
    CHECK(near(m.right_trigger, 1.0f) && near(m.left_trigger, 1.0f));

    playgta6::InputState state{};
    playgta6::os::GameInputReader::write_monotonic(state, m);
    CHECK(state.buttons == m.buttons);
}

// ---------------------------------------------------------------------------
// 2. Deadzone hardening, response curves, Pointer Lock mouse look & reader
// ---------------------------------------------------------------------------

void test_deadzone_curves_and_mouse_look() {
    using namespace playgta6::os;
    const float kNan = std::numeric_limits<float>::quiet_NaN();
    const float kInf = std::numeric_limits<float>::infinity();

    // NaN/Inf inputs and NaN/Inf/degenerate deadzone bounds (inner >= outer).
    {
        const Vec2 n1 = normalize_stick(kNan, 0.5f);
        const Vec2 n2 = normalize_stick(0.5f, kInf);
        const Vec2 n3 = normalize_stick(-kInf, 0.0f);
        CHECK(n1.x == 0.0f && n1.y == 0.0f);
        CHECK(n2.x == 0.0f && n2.y == 0.0f);
        CHECK(n3.x == 0.0f && n3.y == 0.0f);

        // NaN deadzone parameters fall back safely to defaults (0.15 / 0.98).
        const Vec2 dz_nan = normalize_stick(0.98f, 0.0f, kNan, kNan);
        CHECK(near(dz_nan.x, 1.0f) && near(dz_nan.y, 0.0f));

        // Degenerate inner >= outer clamps cleanly without div-by-zero or NaN.
        const Vec2 degen_lo = normalize_stick(0.75f, 0.0f, 0.8f, 0.5f);
        const Vec2 degen_hi = normalize_stick(0.95f, 0.0f, 0.8f, 0.5f);
        CHECK(degen_lo.x == 0.0f && degen_lo.y == 0.0f);
        CHECK(std::isfinite(degen_hi.x) && near(degen_hi.x, 1.0f));

        const Vec2 degen_eq = normalize_stick(1.0f, 0.0f, 1.0f, 1.0f);
        CHECK(std::isfinite(degen_eq.x) && near(degen_eq.x, 1.0f));

        // Magnitude > 1.0 and huge finite overflow clamp smoothly onto unit disk.
        const Vec2 diag_over = normalize_stick(3.0f, 4.0f, 0.15f, 0.98f);
        CHECK(near(diag_over.x, 0.6f) && near(diag_over.y, 0.8f));
        CHECK(std::hypot(diag_over.x, diag_over.y) <= 1.0f + 1e-5f);

        const Vec2 huge = normalize_stick(1e38f, 1e38f, 0.15f, 0.98f);
        CHECK(std::isfinite(huge.x) && std::isfinite(huge.y));
        CHECK(near(huge.x, 0.70710678f, 1e-3f) && near(huge.y, 0.70710678f, 1e-3f));

        // Trigger hardening against NaN, Inf, negative, >1, and degenerate bounds.
        CHECK(normalize_trigger(kNan) == 0.0f);
        CHECK(normalize_trigger(kInf) == 0.0f);
        CHECK(normalize_trigger(-0.5f) == 0.0f);
        CHECK(normalize_trigger(1.5f) == 1.0f);
        CHECK(normalize_trigger(0.7f, 0.8f, 0.4f) == 0.0f);
        CHECK(normalize_trigger(0.9f, 0.8f, 0.4f) == 1.0f);
        CHECK(near(normalize_trigger(0.98f, kNan, kNan), 1.0f));
    }

    // Response curves (Linear, Cubic, Expo) & sensitivity.
    {
        const Vec2 half{0.5f, 0.0f};
        const Vec2 lin = apply_stick_response_curve(half, StickResponseCurve::Linear);
        const Vec2 cub = apply_stick_response_curve(half, StickResponseCurve::Cubic);
        const Vec2 exp2 = apply_stick_response_curve(half, StickResponseCurve::Expo, 2.0f);
        CHECK(near(lin.x, 0.5f) && near(lin.y, 0.0f));
        CHECK(near(cub.x, 0.125f) && near(cub.y, 0.0f));
        CHECK(near(exp2.x, 0.25f) && near(exp2.y, 0.0f));

        // Diagonal preserves angle and sensitivity clamps onto unit disk.
        const Vec2 diag{0.6f, 0.8f};
        const Vec2 boosted = apply_stick_response_curve(diag, StickResponseCurve::Linear, 2.0f, 3.0f);
        CHECK(near(boosted.x, 0.6f) && near(boosted.y, 0.8f));
        const Vec2 nan_curve = apply_stick_response_curve(Vec2{kNan, 0.5f}, StickResponseCurve::Expo, kNan);
        CHECK(nan_curve.x == 0.0f && nan_curve.y == 0.0f);
    }

    // Pointer Lock raw mouse-delta-to-look conversion (matching W374).
    {
        MouseLookConfig cfg{};
        cfg.pixels_for_full_deflection = 40.0f;

        // Mouse right (+20) & mouse up (-20) -> right (+0.5) & look up (+0.5).
        const Vec2 look = mouse_delta_to_look(MouseDelta{20, -20}, cfg);
        CHECK(near(look.x, 0.5f) && near(look.y, 0.5f));

        // invert_y flips pitch sign (mouse up -> look down -0.5).
        cfg.invert_y = true;
        const Vec2 inv = mouse_delta_to_look(MouseDelta{20, -20}, cfg);
        CHECK(near(inv.x, 0.5f) && near(inv.y, -0.5f));

        // Micro-jitter deadzone suppresses tiny deltas.
        cfg.invert_y = false;
        cfg.deadzone_pixels = 3.0f;
        const Vec2 jitter = mouse_delta_to_look(MouseDelta{2, -2}, cfg);
        CHECK(jitter.x == 0.0f && jitter.y == 0.0f);

        // Large spike clamps to unit disk cleanly.
        cfg.deadzone_pixels = 0.0f;
        const Vec2 spike = mouse_delta_to_look(MouseDelta{2000, -2000}, cfg);
        CHECK(near(spike.x, 0.70710678f, 1e-3f) && near(spike.y, 0.70710678f, 1e-3f));

        // Expo curve softens small camera adjustments.
        cfg.curve = StickResponseCurve::Expo;
        cfg.expo_exponent = 2.0f;
        const Vec2 smooth = mouse_delta_to_look(MouseDelta{20, 0}, cfg);
        CHECK(near(smooth.x, 0.25f) && near(smooth.y, 0.0f));
    }

    // Keyboard scan-code decoding (zero heap allocations, deterministic ordering).
    {
        KeyboardScanEntry entries[8]{};
        CHECK(decode_keyboard_scan_entries(0u, entries, 8) == 0);
        const uint32_t all_kb = button::kMoveForward | button::kMoveLeft | button::kMoveBackward |
                                button::kMoveRight | button::kJump | button::kSprint;
        CHECK(decode_keyboard_scan_entries(all_kb, entries, 8) == 6);
        CHECK(entries[0].scan_code == 0x11 && entries[0].code_point == 'w' && entries[0].virtual_key == 0x57);
        CHECK(entries[1].scan_code == 0x1E && entries[1].code_point == 'a' && entries[1].virtual_key == 0x41);
        CHECK(entries[2].scan_code == 0x1F && entries[2].code_point == 's' && entries[2].virtual_key == 0x53);
        CHECK(entries[3].scan_code == 0x20 && entries[3].code_point == 'd' && entries[3].virtual_key == 0x44);
        CHECK(entries[4].scan_code == 0x39 && entries[4].code_point == ' ' && entries[4].virtual_key == 0x20);
        CHECK(entries[5].scan_code == 0x2A && entries[5].code_point == 0 && entries[5].virtual_key == 0xA0);
        CHECK(decode_keyboard_scan_entries(all_kb, entries, 2) == 2);
        CHECK(decode_keyboard_scan_entries(all_kb, nullptr, 8) == 0);
    }

    // GameInputReader::poll monotonic timestamps, button edges, and cumulative mouse_state().
    {
        InputState shared{};
        GameInputReader reader(&shared, SeqlockMode::EvenOddSeqlock);

        InputState s1{};
        s1.buttons = button::kMoveForward | button::kGamepadA;
        s1.mouse_dx = 15;
        s1.mouse_dy = -7;
        s1.right_stick_x = 0.565f;
        s1.right_stick_y = 0.0f;
        s1.right_trigger = 0.515f;
        GameInputReader::write_seqlock(shared, s1, true);

        GameInputFrame f1{};
        CHECK(reader.poll(f1, true));
        CHECK(f1.consistent && f1.timestamp_us > 0);
        CHECK(f1.buttons_pressed == (button::kMoveForward | button::kGamepadA));
        CHECK(f1.buttons_released == 0);
        CHECK(f1.mouse_delta.dx == 15 && f1.mouse_delta.dy == -7);
        CHECK(reader.mouse_state().position_x == 15 && reader.mouse_state().position_y == -7);
        CHECK(near(f1.right_stick.x, 0.5f, 1e-3f));
        CHECK(near(f1.right_trigger, 0.5f, 1e-3f));

        InputState s2{};
        s2.buttons = button::kSprint;
        s2.mouse_dx = -5;
        s2.mouse_dy = 12;
        GameInputReader::write_seqlock(shared, s2, true);

        GameInputFrame f2{};
        CHECK(reader.poll(f2, true));
        CHECK(f2.consistent && f2.timestamp_us > f1.timestamp_us);
        CHECK(f2.buttons_pressed == button::kSprint);
        CHECK(f2.buttons_released == (button::kMoveForward | button::kGamepadA));
        CHECK(f2.mouse_delta.dx == -5 && f2.mouse_delta.dy == 12);
        CHECK(reader.mouse_state().position_x == 10 && reader.mouse_state().position_y == 5);
        CHECK(reader.last_timestamp_us() == f2.timestamp_us);
    }
}

// ---------------------------------------------------------------------------
// 3. Guest COM IGameInput 3D gameplay scenarios (Gamepad, Mouse, Keyboard,
//    128-entry ring wrap-around, GetTemporalReading, Rumble & Haptics)
// ---------------------------------------------------------------------------

void test_guest_game_input_3d_gameplay() {
    GuestInputFixture fx;
    const uint64_t gi = fx.create_game_input();
    CHECK(gi != 0);

    // 3a. Gamepad analog sticks, triggers, and NaN/Inf/clamp sanitization.
    {
        InputState v{};
        v.buttons = button::kGamepadA | button::kLeftBumper | button::kDPadUp | button::kRightStickPress;
        v.left_stick_x = 0.0f;
        v.left_stick_y = 0.98f;
        v.right_stick_x = -0.565f;
        v.right_stick_y = 0.565f;
        v.left_trigger = 0.98f;
        v.right_trigger = 0.515f;
        fx.write_state(v);

        HResult hr = 0;
        const uint64_t r = fx.current_reading(gi, ga::kGameInputKindGamepad, 0, &hr);
        CHECK(hr == 0 && r != 0);
        bool ok = false;
        const ga::GameInputGamepadState gp = fx.read_gamepad(r, &ok);
        CHECK(ok);
        CHECK(gp.buttons == (os::scarlett::gamepad_button::kA | os::scarlett::gamepad_button::kLeftShoulder |
                             os::scarlett::gamepad_button::kDPadUp | os::scarlett::gamepad_button::kRightThumbstick));
        CHECK(near(gp.left_thumbstick_x, 0.0f) && near(gp.left_thumbstick_y, 1.0f));
        CHECK(gp.right_thumbstick_x < -0.4f && gp.right_thumbstick_y > 0.4f);
        CHECK(std::hypot(gp.right_thumbstick_x, gp.right_thumbstick_y) <= 1.0f + 1e-5f);
        CHECK(near(gp.left_trigger, 1.0f) && near(gp.right_trigger, 0.5f, 1e-3f));

        // GetMouseState on a gamepad reading must return false.
        bool mouse_ok = true;
        fx.read_mouse(r, &mouse_ok);
        CHECK(!mouse_ok);
        CHECK(fx.vcall(r, 2) == 0);

        // NaN/Inf in InputState axes/triggers are sanitized to 0.0f cleanly.
        InputState bad{};
        bad.left_stick_x = std::numeric_limits<float>::quiet_NaN();
        bad.left_stick_y = std::numeric_limits<float>::infinity();
        bad.right_stick_x = 5.0f;
        bad.right_stick_y = 5.0f;
        bad.left_trigger = std::numeric_limits<float>::quiet_NaN();
        bad.right_trigger = 3.0f;
        fx.write_state(bad);

        const uint64_t r_bad = fx.current_reading(gi, ga::kGameInputKindGamepad, 0, &hr);
        CHECK(hr == 0 && r_bad != 0);
        const ga::GameInputGamepadState gp_bad = fx.read_gamepad(r_bad, &ok);
        CHECK(ok);
        CHECK(gp_bad.left_thumbstick_x == 0.0f && gp_bad.left_thumbstick_y == 0.0f);
        CHECK(std::isfinite(gp_bad.right_thumbstick_x) && std::isfinite(gp_bad.right_thumbstick_y));
        CHECK(near(std::hypot(gp_bad.right_thumbstick_x, gp_bad.right_thumbstick_y), 1.0f, 1e-3f));
        CHECK(gp_bad.left_trigger == 0.0f && near(gp_bad.right_trigger, 1.0f));
        CHECK(fx.vcall(r_bad, 2) == 0);
    }

    // 3b. Mouse device enumeration (RegisterDeviceCallback + FindDeviceFromId) & GetMouseState (slot 16).
    {
        // Before mouse enumeration or movement, GetCurrentReading(Mouse, null) returns NotFound.
        HResult hr = 0;
        CHECK(fx.current_reading(gi, ga::kGameInputKindMouse, 0, &hr) == 0);
        CHECK(hr == ga::kGameInputReadingNotFound);

        // Enumerate Mouse via RegisterDeviceCallback (slot 9).
        g_cb.calls = 0;
        g_cb.devices.clear();
        const uint64_t token_va = fx.out_slot();
        CHECK(as_hr(fx.vcall(gi, 9, {I(0), I(ga::kGameInputKindMouse), I(1), I(2), I(0xCAFE), I(kGuestDeviceCb),
                                     I(token_va)})) == 0);
        CHECK(g_cb.calls.load() == 1);
        CHECK(g_cb.last_context == 0xCAFE);
        CHECK((g_cb.last_current & ga::kGameInputDeviceConnected) != 0);
        const uint64_t mouse_dev = g_cb.last_device;
        CHECK(mouse_dev != 0);

        // Verify GameInputDeviceInfo ("pg6 virtual mouse" + kGameInputKindMouse) & FindDeviceFromId.
        const uint64_t info_va = fx.vcall(mouse_dev, 3);
        CHECK(info_va != 0);
        ga::AppLocalDeviceId id{};
        CHECK(com::read_guest(info_va + ga::kDeviceInfoDeviceId, id));
        CHECK(std::strncmp(reinterpret_cast<const char*>(id.value), "pg6 virtual mouse", 17) == 0);
        uint32_t supported = 0;
        CHECK(com::read_guest(info_va + ga::kDeviceInfoSupportedInput, supported));
        CHECK(supported == ga::kGameInputKindMouse);

        const uint64_t id_va = com::guest_alloc(sizeof(id));
        const uint64_t found_va = fx.out_slot();
        CHECK(com::write_guest(id_va, id));
        CHECK(as_hr(fx.vcall(gi, 16, {I(id_va), I(found_va)})) == 0);
        CHECK(load64(found_va) == mouse_dev);
        CHECK(fx.vcall(mouse_dev, 2) >= 1);  // release FindDeviceFromId ref
        com::guest_free(id_va);
        com::guest_free(found_va);
        CHECK((fx.vcall(gi, 13, {I(load64(token_va)), I(5000)}) & 0xFF) == 1);
        com::guest_free(token_va);

        // Cumulative mouse look deltas across multiple frames + interleaved gamepad poll.
        InputState m1{};
        m1.mouse_dx = 24;
        m1.mouse_dy = -10;
        fx.write_state(m1, true);
        const uint64_t mr1 = fx.current_reading(gi, ga::kGameInputKindMouse, mouse_dev, &hr);
        CHECK(hr == 0 && mr1 != 0);
        CHECK(static_cast<uint32_t>(fx.vcall(mr1, 3)) == ga::kGameInputKindMouse);
        bool ok = false;
        GuestMouseStateV0 ms1 = fx.read_mouse(mr1, &ok);
        CHECK(ok && ms1.position_x == 24 && ms1.position_y == -10);
        bool gp_ok = true;
        fx.read_gamepad(mr1, &gp_ok);
        CHECK(!gp_ok);

        // Next frame: accumulate mouse delta AND poll gamepad first — mouse delta must NOT be lost!
        InputState m2{};
        m2.buttons = button::kGamepadB;
        m2.mouse_dx = -6;
        m2.mouse_dy = 18;
        fx.write_state(m2, true);
        const uint64_t gp_interleaved = fx.current_reading(gi, ga::kGameInputKindGamepad, 0, &hr);
        CHECK(hr == 0 && gp_interleaved != 0);
        CHECK(fx.vcall(gp_interleaved, 2) == 0);

        const uint64_t mr2 = fx.neighbour_reading(gi, 5, mr1, ga::kGameInputKindMouse, mouse_dev, &hr);
        CHECK(hr == 0 && mr2 != 0);
        GuestMouseStateV0 ms2 = fx.read_mouse(mr2, &ok);
        CHECK(ok && ms2.position_x == 18 && ms2.position_y == 8);
        CHECK(fx.vcall(mr2, 5) > fx.vcall(mr1, 5));  // strictly monotonic timestamp

        // Unrelated gamepad change does not emit a duplicate unchanged mouse reading.
        InputState pad_only{};
        pad_only.buttons = button::kGamepadY;
        fx.write_state(pad_only, false);
        CHECK(fx.neighbour_reading(gi, 5, mr2, ga::kGameInputKindMouse, mouse_dev, &hr) == 0);
        CHECK(hr == ga::kGameInputReadingNotFound);

        // GetPreviousReading walks back from mr2 -> mr1.
        const uint64_t mr_prev = fx.neighbour_reading(gi, 6, mr2, ga::kGameInputKindMouse, mouse_dev, &hr);
        CHECK(hr == 0 && mr_prev != 0);
        CHECK(fx.vcall(mr_prev, 4, {I(ga::kGameInputKindMouse)}) == fx.vcall(mr1, 4, {I(ga::kGameInputKindMouse)}));
        CHECK(fx.vcall(mr_prev, 2) == 0);
        CHECK(fx.vcall(mr2, 2) == 0);
        CHECK(fx.vcall(mr1, 2) == 0);
    }

    // 3c. Keyboard scan-code state (all 6 SAB keys + zero-key fast path).
    {
        InputState none{};
        fx.write_state(none);
        HResult hr = 0;
        const uint64_t kb0 = fx.current_reading(gi, ga::kGameInputKindKeyboard, 0, &hr);
        CHECK(hr == 0 && kb0 != 0);
        CHECK(fx.vcall(kb0, 14) == 0);
        CHECK(fx.vcall(kb0, 2) == 0);

        InputState all{};
        all.buttons = button::kMoveForward | button::kMoveLeft | button::kMoveBackward |
                      button::kMoveRight | button::kJump | button::kSprint;
        fx.write_state(all);
        const uint64_t kb1 = fx.current_reading(gi, ga::kGameInputKindKeyboard, 0, &hr);
        CHECK(hr == 0 && kb1 != 0);
        CHECK(fx.vcall(kb1, 14) == 6);
        const uint64_t buf_va = com::guest_alloc(8 * sizeof(ga::GameInputKeyState));
        CHECK(fx.vcall(kb1, 15, {I(8), I(buf_va)}) == 6);
        ga::GameInputKeyState ks[6]{};
        CHECK(com::read_guest(buf_va, ks, sizeof(ks)));
        CHECK(ks[0].scan_code == 0x11 && ks[0].code_point == 'w' && ks[0].virtual_key == 0x57);
        CHECK(ks[1].scan_code == 0x1E && ks[1].code_point == 'a' && ks[1].virtual_key == 0x41);
        CHECK(ks[2].scan_code == 0x1F && ks[2].code_point == 's' && ks[2].virtual_key == 0x53);
        CHECK(ks[3].scan_code == 0x20 && ks[3].code_point == 'd' && ks[3].virtual_key == 0x44);
        CHECK(ks[4].scan_code == 0x39 && ks[4].code_point == ' ' && ks[4].virtual_key == 0x20);
        CHECK(ks[5].scan_code == 0x2A && ks[5].code_point == 0 && ks[5].virtual_key == 0xA0);
        CHECK(fx.vcall(kb1, 15, {I(3), I(buf_va)}) == 3);
        com::guest_free(buf_va);
        CHECK(fx.vcall(kb1, 2) == 0);
    }

    // 3d. 128-entry fixed ring wrap-around, GetNext/GetPrevious, GetTemporalReading & zero-leak check.
    {
        // Get gamepad device handle.
        g_cb.calls = 0;
        g_cb.devices.clear();
        const uint64_t tok = fx.out_slot();
        CHECK(as_hr(fx.vcall(gi, 9, {I(0), I(ga::kGameInputKindGamepad), I(1), I(2), I(0), I(kGuestDeviceCb),
                                     I(tok)})) == 0);
        const uint64_t pad_dev = g_cb.last_device;
        CHECK(pad_dev != 0);
        CHECK((fx.vcall(gi, 13, {I(load64(tok)), I(5000)}) & 0xFF) == 1);
        com::guest_free(tok);

        const com::BridgeStats stats_before = com::bridge_stats();
        constexpr int kTotal = 160;  // exceeds 128-entry ring capacity to exercise wrap-around
        std::vector<uint64_t> seqs(kTotal);
        std::vector<uint64_t> times(kTotal);
        uint64_t last_reading = 0;
        for (int i = 0; i < kTotal; ++i) {
            InputState step{};
            step.buttons = (i & 1) ? button::kGamepadA : button::kGamepadX;
            step.left_trigger = static_cast<float>(i % 100) / 100.0f;
            fx.write_state(step);
            HResult hr = 0;
            const uint64_t r = fx.current_reading(gi, ga::kGameInputKindGamepad, pad_dev, &hr);
            CHECK(hr == 0 && r != 0);
            seqs[i] = fx.vcall(r, 4, {I(ga::kGameInputKindGamepad)});
            times[i] = fx.vcall(r, 5);
            if (i > 0) {
                CHECK(seqs[i] > seqs[i - 1]);
                CHECK(times[i] > times[i - 1]);  // strictly monotonic
            }
            if (last_reading != 0) {
                CHECK(fx.vcall(last_reading, 2) == 0);
            }
            last_reading = r;
        }

        // Walk backward through all 128 retained readings via GetPreviousReading.
        int walked = 1;
        uint64_t cur = last_reading;
        uint64_t oldest_retained = 0;
        for (;;) {
            HResult hr = 0;
            const uint64_t prev = fx.neighbour_reading(gi, 6, cur, ga::kGameInputKindGamepad, pad_dev, &hr);
            if (prev == 0) {
                CHECK(hr == ga::kGameInputReadingNotFound);
                oldest_retained = cur;
                break;
            }
            CHECK(hr == 0);
            CHECK(fx.vcall(prev, 4, {I(0)}) < fx.vcall(cur, 4, {I(0)}));
            CHECK(fx.vcall(cur, 2) == 0);
            cur = prev;
            ++walked;
        }
        CHECK(walked == 128);
        CHECK(fx.vcall(oldest_retained, 4, {I(0)}) == seqs[kTotal - 128]);

        // Walk forward 1 step from oldest_retained via GetNextReading.
        HResult hr = 0;
        const uint64_t second_oldest =
            fx.neighbour_reading(gi, 5, oldest_retained, ga::kGameInputKindGamepad, pad_dev, &hr);
        CHECK(hr == 0 && second_oldest != 0);
        CHECK(fx.vcall(second_oldest, 4, {I(0)}) == seqs[kTotal - 127]);
        CHECK(fx.vcall(second_oldest, 2) == 0);
        CHECK(fx.vcall(oldest_retained, 2) == 0);

        // Temporal reading lookup (exact, in-between, future, and before-oldest).
        uint64_t tr = 0;
        HResult thr = 0;
        const int target_idx = kTotal - 10;
        CHECK(os::game_input_guest_temporal_reading(gi, times[target_idx], pad_dev, &tr, &thr));
        CHECK(thr == 0 && tr != 0);
        CHECK(fx.vcall(tr, 4, {I(0)}) == seqs[target_idx]);
        CHECK(fx.vcall(tr, 2) == 0);

        // Timestamp strictly before oldest retained (index kTotal - 128) -> NotFound.
        CHECK(!os::game_input_guest_temporal_reading(gi, times[kTotal - 128] - 1, pad_dev, &tr, &thr));
        CHECK(thr == ga::kGameInputReadingNotFound && tr == 0);

        // Future timestamp -> returns newest reading.
        CHECK(os::game_input_guest_temporal_reading(gi, times[kTotal - 1] + 1000000ULL, pad_dev, &tr, &thr));
        CHECK(thr == 0 && tr != 0);
        CHECK(fx.vcall(tr, 4, {I(0)}) == seqs[kTotal - 1]);
        CHECK(fx.vcall(tr, 2) == 0);

        // Null device temporal lookup searches across all devices and finds the newest reading.
        CHECK(os::game_input_guest_temporal_reading(gi, times[kTotal - 1] + 1000000ULL, 0, &tr, &thr));
        CHECK(thr == 0 && tr != 0);
        CHECK(fx.vcall(tr, 4, {I(0)}) == seqs[kTotal - 1]);
        CHECK(fx.vcall(tr, 2) == 0);

        // Confirm zero guest object or guest heap leaks over 160+ readings.
        const com::BridgeStats stats_after = com::bridge_stats();
        CHECK(stats_after.live_objects == stats_before.live_objects);
        CHECK(stats_after.guest_bytes_in_use == stats_before.guest_bytes_in_use);

        // 3e. Rumble clamping, null-stop semantics, SetHapticMotorState (slot 9), and telemetry.
        const uint64_t rva = com::guest_alloc(sizeof(ga::GameInputRumbleParams));
        const ga::GameInputRumbleParams raw_rumble{
            -0.5f, 2.5f, std::numeric_limits<float>::quiet_NaN(), 0.75f};
        CHECK(com::write_guest(rva, raw_rumble));
        fx.vcall(pad_dev, 10, {I(rva)});

        const ga::GameInputRumbleParams got1 = ga::virtual_gamepad_rumble();
        CHECK(got1.low_frequency == 0.0f && got1.high_frequency == 1.0f &&
              got1.left_trigger == 0.0f && near(got1.right_trigger, 0.75f));

        // Second rumble with different peaks.
        const ga::GameInputRumbleParams rumble2{0.6f, 0.2f, 0.9f, 0.1f};
        CHECK(com::write_guest(rva, rumble2));
        fx.vcall(pad_dev, 10, {I(rva)});
        com::guest_free(rva);

        // Null pointer stops all 4 motors cleanly.
        fx.vcall(pad_dev, 10, {I(0)});
        const ga::GameInputRumbleParams stopped = ga::virtual_gamepad_rumble();
        CHECK(stopped.low_frequency == 0.0f && stopped.high_frequency == 0.0f &&
              stopped.left_trigger == 0.0f && stopped.right_trigger == 0.0f);

        // Slot 9 SetHapticMotorState returns S_OK cleanly (non-trapping) and records telemetry.
        CHECK(as_hr(fx.vcall(pad_dev, 9, {I(0), I(0)})) == 0);

        const os::RumbleTelemetry tel = os::game_input_rumble_telemetry();
        CHECK(tel.set_calls == 3);
        CHECK(tel.nonzero_updates == 2);
        CHECK(tel.stop_calls == 1);
        CHECK(tel.haptic_calls == 1);
        CHECK(tel.last_timestamp_us > 0);
        CHECK(tel.last_low_frequency == 0.0f && tel.last_high_frequency == 0.0f &&
              tel.last_left_trigger == 0.0f && tel.last_right_trigger == 0.0f);
        CHECK(near(tel.peak_low_frequency, 0.6f));
        CHECK(near(tel.peak_high_frequency, 1.0f));
        CHECK(near(tel.peak_left_trigger, 0.9f));
        CHECK(near(tel.peak_right_trigger, 0.75f));
    }

    CHECK(fx.vcall(gi, 2) >= 1);
}

struct FakeXInput {
    struct Call { uint32_t pad; uint16_t low, high; };
    std::vector<Call> calls;
};
FakeXInput g_fake;
uint32_t fake_set_state(uint32_t pad, uint16_t low, uint16_t high) {
    g_fake.calls.push_back({pad, low, high});
    return 0;
}

void test_rumble_forwarder() {
    os::RumbleTelemetry t{};
    hi::RumbleForwarder f(&fake_set_state, 16);

    // No pad: nothing is sent.
    t.last_low_frequency = 0.5f;
    f.update(t, -1, true, 0);
    CHECK(g_fake.calls.empty());

    // Pad present: motors follow the request (low -> left motor, high -> right motor).
    t.last_low_frequency = 1.0f;
    t.last_high_frequency = 0.5f;
    f.update(t, 0, true, 100);
    CHECK(g_fake.calls.size() == 1);
    CHECK(g_fake.calls[0].pad == 0 && g_fake.calls[0].low == 65535 && g_fake.calls[0].high == 32768);

    // Unchanged: no resend. Changed inside the interval: held back, sent once the interval passed.
    f.update(t, 0, true, 110);
    CHECK(g_fake.calls.size() == 1);
    t.last_low_frequency = 0.25f;
    f.update(t, 0, true, 105 + 0);
    CHECK(g_fake.calls.size() == 1 && f.stats().rate_limited == 1);
    f.update(t, 0, true, 120);
    CHECK(g_fake.calls.size() == 2 && g_fake.calls[1].low == 16384);

    // Triggers cannot be driven on XInput: counted only.
    t.last_left_trigger = 0.8f;
    f.update(t, 0, true, 200);
    CHECK(f.stats().trigger_unsupported == 1 && g_fake.calls.size() == 2);

    // A stop is never rate limited.
    t.last_low_frequency = 0.0f;
    t.last_high_frequency = 0.0f;
    f.update(t, 0, true, 201);
    CHECK(g_fake.calls.size() == 3 && g_fake.calls[2].low == 0 && g_fake.calls[2].high == 0);
    CHECK(f.stats().stops == 1);

    // Focus loss forces the motors off even though the title still asks for rumble; refocus resumes.
    t.last_low_frequency = 1.0f;
    f.update(t, 0, true, 300);
    CHECK(g_fake.calls.size() == 4);
    f.update(t, 0, false, 301);
    CHECK(g_fake.calls.size() == 5 && g_fake.calls[4].low == 0 && g_fake.calls[4].high == 0);
    f.update(t, 0, true, 400);
    CHECK(g_fake.calls.size() == 6 && g_fake.calls[5].low == 65535);

    // Exit/fault: stop() silences a running motor once.
    f.stop();
    CHECK(g_fake.calls.size() == 7 && g_fake.calls[6].low == 0);
    f.stop();
    CHECK(g_fake.calls.size() == 7);

    // Values are clamped.
    g_fake.calls.clear();
    hi::RumbleForwarder g(&fake_set_state, 0);
    t.last_low_frequency = 3.0f;
    t.last_high_frequency = -1.0f;
    g.update(t, 1, true, 0);
    CHECK(g_fake.calls.size() == 1 && g_fake.calls[0].pad == 1 && g_fake.calls[0].low == 65535 && g_fake.calls[0].high == 0);

    // Non-monotonic timestamp regression must not underflow rate limiter.
    g_fake.calls.clear();
    hi::RumbleForwarder mono(&fake_set_state, 16);
    t.last_low_frequency = 0.5f;
    t.last_high_frequency = 0.0f;
    mono.update(t, 0, true, 100);
    CHECK(g_fake.calls.size() == 1);
    t.last_low_frequency = 0.75f;
    mono.update(t, 0, true, 50);  // regressed below last_send_ms_ = 100 -> must not underflow
    CHECK(g_fake.calls.size() == 2 && mono.stats().rate_limited == 0);
}

void test_host_input_hardening() {
    // 1. merge() suppresses resting pad drift (|pad| <= 0.15) when keyboard stick is active
    //    so pressing pure W does not drift sideways, while preserving tie-break when pad is deflected.
    {
        InputState pad{};
        pad.left_stick_x = 0.06f;
        pad.left_stick_y = 0.04f;
        pad.right_stick_x = -0.05f;
        pad.right_stick_y = 0.07f;

        InputState kb{};
        kb.left_stick_x = 0.0f;
        kb.left_stick_y = 1.0f;
        kb.right_stick_x = 0.8f;
        kb.right_stick_y = 0.0f;

        const InputState merged = hi::merge(pad, kb);
        CHECK(near(merged.left_stick_x, 0.0f));
        CHECK(near(merged.left_stick_y, 1.0f));
        CHECK(near(merged.right_stick_x, 0.8f));
        CHECK(near(merged.right_stick_y, 0.0f));

        // When keyboard is idle, raw pad values still pass through to GameInputReader deadzone.
        const InputState pad_only = hi::merge(pad, InputState{});
        CHECK(near(pad_only.left_stick_x, 0.06f));
        CHECK(near(pad_only.left_stick_y, 0.04f));
    }

    // 2. HostInputSource null state safety, double-start safety, idle write dedup, and tear-free seqlock.
    {
        hi::HostInputSource null_src(nullptr);
        null_src.start();
        null_src.stop();
        CHECK(null_src.writes() == 0);

        InputState shared{};
        hi::HostInputSource src(&shared);
        src.set_keyboard_enabled(false);  // deterministic idle state regardless of host keyboard
        src.start();
        src.start();  // second start() must be a safe no-op (no std::terminate)
        std::this_thread::sleep_for(std::chrono::milliseconds(60));
        const uint64_t writes_idle = src.writes();
        src.stop();
        // Idle dedup: 60 ms at 8 ms/poll would have done ~7-8 redundant writes before; now stays <= 2.
        CHECK(writes_idle <= 2);
        // Tear-free write_seqlock always leaves seq even.
        CHECK((shared.seq & 1u) == 0u);
    }
}

}  // namespace

int main() {
    test_rumble_forwarder();
    test_host_input_mapping();
    test_host_input_hardening();
    test_deadzone_curves_and_mouse_look();
    test_guest_game_input_3d_gameplay();

    if (g_failures == 0) {
        std::puts("host input: ok");
    }
    return g_failures == 0 ? 0 : 1;
}
