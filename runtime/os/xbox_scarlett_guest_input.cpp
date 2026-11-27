// GameInput.dll and xg_xs.dll guest interfaces (O23); evidence and layouts in xbox_scarlett_guest.h.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "os/guest_window.h"
#include "gpu/gfx10_detile.h"
#include "os/game_input.h"
#include "os/xbox_scarlett.h"
#include "os/xbox_scarlett_guest.h"

namespace playgta6::os::scarlett::guest_api {

using com::GuestPtr;
using com::Guid;
using com::OutPtr;

namespace {

constexpr HResult kOk = com::S_OK_HR;

void diag(const std::string& message) {
    com::detail::diagnostic(message.c_str());
}

uint64_t now_us() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

// ===========================================================================
// GameInput
// ===========================================================================

// Public GameInput v0 interface identities (the title never queries them; QueryInterface only).
constexpr Guid kIidIGameInput = com::make_guid("11be2a7e-4254-445a-9c09-ffc40f006918");
constexpr Guid kIidIGameInputReading = com::make_guid("2156947a-e1fa-4de0-a30b-d812931dbd8d");
constexpr Guid kIidIGameInputDevice = com::make_guid("31dd86fb-4c1b-408a-868f-439b3cd47125");

constexpr uint32_t kDeviceStatus = kGameInputDeviceConnected | kGameInputDeviceInputEnabled;
constexpr size_t kHistory = 128;
constexpr size_t kMaxSnapshotKeys = 16;
constexpr uint32_t kGuestSabKeyboardMask =
    os::input_button::kMoveForward | os::input_button::kMoveLeft | os::input_button::kMoveBackward |
    os::input_button::kMoveRight | os::input_button::kJump | os::input_button::kSprint;
static_assert(kGuestSabKeyboardMask == 0x3Fu);

enum class DeviceKind : uint8_t { Gamepad = 0, Keyboard = 1, Mouse = 2 };
constexpr size_t kDeviceKindCount = 3;

uint32_t input_kind_of(DeviceKind kind) {
    switch (kind) {
        case DeviceKind::Gamepad:
            return kGameInputKindGamepad;
        case DeviceKind::Keyboard:
            return kGameInputKindKeyboard;
        case DeviceKind::Mouse:
            return kGameInputKindMouse;
    }
    return 0u;
}

const char* device_tag(DeviceKind kind) {
    switch (kind) {
        case DeviceKind::Gamepad:
            return "pg6 virtual gamepad";
        case DeviceKind::Keyboard:
            return "pg6 virtual keyboard";
        case DeviceKind::Mouse:
            return "pg6 virtual mouse";
    }
    return "";
}

AppLocalDeviceId make_device_id(DeviceKind kind) {
    AppLocalDeviceId id{};
    const char* tag = device_tag(kind);
    const size_t len = std::min(std::strlen(tag), sizeof(id.value));
    std::memcpy(id.value, tag, len);
    return id;
}

// GDK v0 GameInputMouseState layout (40 bytes, matching GameInput.h v0 and Pg6GameInputMouseState).
struct GameInputMouseState {
    uint32_t buttons;     // GameInputMouseButtons bitmask (+ 4 bytes alignment padding)
    int64_t position_x;   // Cumulative relative X delta
    int64_t position_y;   // Cumulative relative Y delta
    int64_t wheel_x;      // Cumulative horizontal wheel delta
    int64_t wheel_y;      // Cumulative vertical wheel delta
};
static_assert(sizeof(GameInputMouseState) == 40 && std::is_trivially_copyable_v<GameInputMouseState>);
static_assert(offsetof(GameInputMouseState, buttons) == 0);
static_assert(offsetof(GameInputMouseState, position_x) == 8);
static_assert(offsetof(GameInputMouseState, position_y) == 16);
static_assert(offsetof(GameInputMouseState, wheel_x) == 24);
static_assert(offsetof(GameInputMouseState, wheel_y) == 32);

float sanitize_unit_float(float v) {
    if (!std::isfinite(v) || v <= 0.0f) {
        return 0.0f;
    }
    return v >= 1.0f ? 1.0f : v;
}

float sanitize_signed_axis(float v) {
    if (!std::isfinite(v) || v == 0.0f) {
        return 0.0f;
    }
    return std::clamp(v, -1.0f, 1.0f);
}

void sanitize_stick_pair(float& x, float& y) {
    if (!std::isfinite(x) || !std::isfinite(y)) {
        x = 0.0f;
        y = 0.0f;
        return;
    }
    const float mag = std::hypot(x, y);
    if (!std::isfinite(mag)) {
        const float max_abs = std::max(std::fabs(x), std::fabs(y));
        if (max_abs <= 0.0f) {
            x = 0.0f;
            y = 0.0f;
            return;
        }
        const float sx = x / max_abs;
        const float sy = y / max_abs;
        const float smag = std::hypot(sx, sy);
        x = (smag > 0.0f && std::isfinite(smag)) ? sanitize_signed_axis(sx / smag) : 0.0f;
        y = (smag > 0.0f && std::isfinite(smag)) ? sanitize_signed_axis(sy / smag) : 0.0f;
        return;
    }
    if (mag > 1.0f) {
        x /= mag;
        y /= mag;
    }
    x = sanitize_signed_axis(x);
    y = sanitize_signed_axis(y);
}

void sanitize_gamepad_state(GameInputGamepadState& pad) {
    pad.left_trigger = sanitize_unit_float(pad.left_trigger);
    pad.right_trigger = sanitize_unit_float(pad.right_trigger);
    sanitize_stick_pair(pad.left_thumbstick_x, pad.left_thumbstick_y);
    sanitize_stick_pair(pad.right_thumbstick_x, pad.right_thumbstick_y);
}

struct Snapshot {
    DeviceKind kind = DeviceKind::Gamepad;
    uint64_t sequence = 0;   // GameInput sequence number (one counter for all readings)
    uint64_t timestamp = 0;  // microseconds, strictly monotonic
    uint32_t host_sequence = 0;
    GameInputGamepadState gamepad{};
    GameInputMouseState mouse{};
    GameInputKeyState keys[kMaxSnapshotKeys]{};
    uint32_t key_count = 0;
};
static_assert(std::is_trivially_copyable_v<Snapshot>);

// Fixed-capacity ring buffer storing strictly-increasing sequences and timestamps (zero heap churn).
template <size_t kCapacity>
class ReadingRing {
public:
    bool empty() const { return count_ == 0; }
    size_t size() const { return count_; }
    void clear() {
        head_ = 0;
        count_ = 0;
    }
    const Snapshot& at(size_t i) const { return entries_[(head_ + i) % kCapacity]; }
    const Snapshot& back() const { return at(count_ - 1); }

    void push(const Snapshot& s) {
        if (count_ < kCapacity) {
            entries_[(head_ + count_) % kCapacity] = s;
            ++count_;
        } else {
            entries_[head_] = s;
            head_ = (head_ + 1) % kCapacity;
        }
    }

    // First entry whose sequence > `sequence`.
    bool find_next(uint64_t sequence, Snapshot* out) const {
        size_t lo = 0;
        size_t hi = count_;
        while (lo < hi) {
            const size_t mid = lo + (hi - lo) / 2;
            if (at(mid).sequence <= sequence) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (lo >= count_) {
            return false;
        }
        *out = at(lo);
        return true;
    }

    // Last entry whose sequence < `sequence`.
    bool find_previous(uint64_t sequence, Snapshot* out) const {
        size_t lo = 0;
        size_t hi = count_;
        while (lo < hi) {
            const size_t mid = lo + (hi - lo) / 2;
            if (at(mid).sequence < sequence) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (lo == 0) {
            return false;
        }
        *out = at(lo - 1);
        return true;
    }

    // Newest entry whose timestamp <= `timestamp` (false when empty or timestamp < oldest).
    bool find_temporal(uint64_t timestamp, Snapshot* out) const {
        if (count_ == 0 || timestamp < at(0).timestamp) {
            return false;
        }
        size_t lo = 0;
        size_t hi = count_;
        while (lo < hi) {
            const size_t mid = lo + (hi - lo) / 2;
            if (at(mid).timestamp <= timestamp) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (lo == 0) {
            return false;
        }
        *out = at(lo - 1);
        return true;
    }

private:
    std::array<Snapshot, kCapacity> entries_{};
    size_t head_ = 0;
    size_t count_ = 0;
};

GameInputRumbleParams& rumble_state() {
    static GameInputRumbleParams r{};
    return r;
}
os::RumbleTelemetry& rumble_telemetry_state() {
    static os::RumbleTelemetry t{};
    return t;
}
std::mutex& rumble_mutex() {
    static std::mutex m;
    return m;
}

class GameInputDeviceObj;
com::InterfaceArg<GameInputDeviceObj> ensure_mouse_device();
bool has_mouse_device();


// ---------------------------------------------------------------------------

class GameInputDeviceObj final : public com::ComObject {
public:
    static constexpr const char* kComClassName = "GameInput.IGameInputDevice";
    static constexpr uint64_t kComPayloadSize = kGameInputDeviceInfoSize;

    explicit GameInputDeviceObj(DeviceKind kind) : kind_(kind), id_(make_device_id(kind)) {}

    // Fills the guest GameInputDeviceInfo (after the object has its guest block).
    void publish_info() {
        uint8_t info[kGameInputDeviceInfoSize]{};
        const uint32_t size = static_cast<uint32_t>(kGameInputDeviceInfoSize);
        std::memcpy(info, &size, 4);
        const uint16_t vendor = 0x045E;  // Microsoft
        const uint16_t product = kind_ == DeviceKind::Gamepad ? 0x0B12
                                 : (kind_ == DeviceKind::Mouse ? 0x0001 : 0x0000);
        std::memcpy(info + kDeviceInfoVendorId, &vendor, 2);
        std::memcpy(info + kDeviceInfoProductId, &product, 2);
        std::memcpy(info + kDeviceInfoDeviceId, id_.value, sizeof(id_.value));
        const uint32_t supported = input_kind_of(kind_);
        std::memcpy(info + kDeviceInfoSupportedInput, &supported, 4);
        com::write_guest(payload_address(), info, sizeof(info));
    }

    // slot 3: const GameInputDeviceInfo*
    uint64_t GetDeviceInfo() { return payload_address(); }
    // slot 4
    uint32_t GetDeviceStatus() { return kDeviceStatus; }
    // slot 5: void GetBatteryState(GameInputBatteryState* state)
    void GetBatteryState(uint64_t state_va) {
        if (state_va == 0) {
            return;
        }
        struct GameInputBatteryState {
            float charge_rate = 0.0f;
            float max_charge_rate = 0.0f;
            float remaining_capacity = 1.0f;
            float full_charge_capacity = 1.0f;
            uint32_t status = 0u;  // GameInputBatteryNotPresent (wired controller)
        };
        static_assert(sizeof(GameInputBatteryState) == 20);
        GameInputBatteryState st{};
        com::write_guest(state_va, st);
    }
    // slot 6: HRESULT CreateForceFeedbackEffect(uint32_t motorIndex, const params*, IGameInputForceFeedbackEffect**)
    HResult CreateForceFeedbackEffect(uint32_t /*motor_index*/, uint64_t /*params*/, OutPtr out) {
        if (out.is_null() || !out.write(0)) {
            return com::E_POINTER_HR;
        }
        return com::E_NOTIMPL_HR;
    }
    // slot 7: bool IsForceFeedbackMotorPoweredOn(uint32_t motorIndex)
    bool IsForceFeedbackMotorPoweredOn(uint32_t /*motor_index*/) { return false; }
    // slot 8: void SetForceFeedbackMotorGain(uint32_t motorIndex, float masterGain)
    void SetForceFeedbackMotorGain(uint32_t /*motor_index*/, float /*gain*/) {}
    // slot 9: GDK v0 HRESULT SetHapticMotorState(uint32_t motorIndex, const GameInputHapticFeedbackParams* params)
    HResult SetHapticMotorState(uint32_t /*motor_index*/, uint64_t /*params*/) {
        std::lock_guard<std::mutex> lock(rumble_mutex());
        os::RumbleTelemetry& t = rumble_telemetry_state();
        t.haptic_calls += 1;
        const uint64_t ts = now_us();
        t.last_timestamp_us = ts > t.last_timestamp_us ? ts : (t.last_timestamp_us + 1);
        return kOk;
    }
    // slot 10: void SetRumbleState(const GameInputRumbleParams* params)
    void SetRumbleState(GuestPtr<GameInputRumbleParams> params) {
        GameInputRumbleParams p{};
        if (!params.is_null()) {
            if (!params.read(p)) {
                return;
            }
            p.low_frequency = sanitize_unit_float(p.low_frequency);
            p.high_frequency = sanitize_unit_float(p.high_frequency);
            p.left_trigger = sanitize_unit_float(p.left_trigger);
            p.right_trigger = sanitize_unit_float(p.right_trigger);
        }
        if (kind_ == DeviceKind::Gamepad) {
            std::lock_guard<std::mutex> lock(rumble_mutex());
            rumble_state() = p;
            os::RumbleTelemetry& t = rumble_telemetry_state();
            t.set_calls += 1;
            const bool nonzero = (p.low_frequency > 0.0f || p.high_frequency > 0.0f ||
                                  p.left_trigger > 0.0f || p.right_trigger > 0.0f);
            if (nonzero) {
                t.nonzero_updates += 1;
            } else {
                t.stop_calls += 1;
            }
            const uint64_t ts = now_us();
            t.last_timestamp_us = ts > t.last_timestamp_us ? ts : (t.last_timestamp_us + 1);
            t.last_low_frequency = p.low_frequency;
            t.last_high_frequency = p.high_frequency;
            t.last_left_trigger = p.left_trigger;
            t.last_right_trigger = p.right_trigger;
            t.peak_low_frequency = std::max(t.peak_low_frequency, p.low_frequency);
            t.peak_high_frequency = std::max(t.peak_high_frequency, p.high_frequency);
            t.peak_left_trigger = std::max(t.peak_left_trigger, p.left_trigger);
            t.peak_right_trigger = std::max(t.peak_right_trigger, p.right_trigger);
        }
    }

    DeviceKind kind() const { return kind_; }
    const AppLocalDeviceId& id() const { return id_; }

private:
    DeviceKind kind_;
    AppLocalDeviceId id_{};
};

class GameInputReadingObj final : public com::ComObject {
public:
    static constexpr const char* kComClassName = "GameInput.IGameInputReading";

    GameInputReadingObj(const Snapshot& snapshot, com::InterfaceArg<GameInputDeviceObj> device)
        : snap_(snapshot), device_(std::move(device)) {}

    uint32_t GetInputKind() { return input_kind_of(snap_.kind); }                 // 3
    uint64_t GetSequenceNumber(uint32_t /*kind*/) { return snap_.sequence; }       // 4
    uint64_t GetTimestamp() { return snap_.timestamp; }                            // 5
    void GetDevice(OutPtr out) {                                                   // 6
        if (out.is_null()) {
            return;
        }
        out.write(share(device_));
    }
    bool GetRawReport(OutPtr out) {  // 7
        if (!out.is_null()) {
            out.write(0);
        }
        return false;
    }
    uint32_t GetZeroCount() { return 0; }                                          // 8, 10, 12, 17
    uint32_t GetZeroState(uint32_t /*count*/, uint64_t /*array*/) { return 0; }    // 9, 11, 13, 18
    uint32_t GetKeyCount() { return snap_.key_count; }                             // 14
    uint32_t GetKeyState(uint32_t count, uint64_t out) {                           // 15
        const uint32_t n = std::min<uint32_t>(count, snap_.key_count);
        if (n != 0 && !com::write_guest(out, snap_.keys, n * sizeof(GameInputKeyState))) {
            return 0;
        }
        return n;
    }
    bool GetMouseState(GuestPtr<GameInputMouseState> out) {                        // 16
        if (snap_.kind != DeviceKind::Mouse || out.is_null()) {
            return false;
        }
        return out.write(snap_.mouse);
    }
    bool GetNoState(uint64_t /*state*/) { return false; }                          // 19..21, 23, 24
    bool GetGamepadState(GuestPtr<GameInputGamepadState> out) {                    // 22
        if (snap_.kind != DeviceKind::Gamepad || out.is_null()) {
            return false;
        }
        return out.write(snap_.gamepad);
    }

    const Snapshot& snapshot() const { return snap_; }

    static const com::Interface<GameInputDeviceObj>& kIDevice();

    // A new COM reference to a pinned device (0 once the device is released).
    static uint64_t share(const com::InterfaceArg<GameInputDeviceObj>& device) {
        if (!device || device->ref_count() == 0) {
            return 0;
        }
        return com::Ref<GameInputDeviceObj>::share(device.get()).detach(kIDevice());
    }

private:
    Snapshot snap_;
    com::InterfaceArg<GameInputDeviceObj> device_;  // pinned: readings hold no COM reference
};

class GameInputApi final : public com::ComObject {
public:
    static constexpr const char* kComClassName = "GameInput.IGameInput";

    // The devices' COM references belong to the module state; the API only pins them.
    GameInputApi(Pg6IGameInput* host, com::InterfaceArg<GameInputDeviceObj> gamepad,
                 com::InterfaceArg<GameInputDeviceObj> keyboard)
        : host_(host), gamepad_(std::move(gamepad)), keyboard_(std::move(keyboard)) {}
    ~GameInputApi() override {
        if (host_ != nullptr) {
            host_->Release();
        }
    }

    uint64_t GetCurrentTimestamp() {  // 3
        std::lock_guard<std::mutex> lock(mutex_);
        const uint64_t ts = now_us();
        return ts >= last_timestamp_ ? ts : last_timestamp_;
    }

    // 4
    HResult GetCurrentReading(uint32_t kind, com::InterfaceArg<GameInputDeviceObj> device, OutPtr out) {
        if (out.is_null() || !out.write(0)) {
            return com::E_POINTER_HR;
        }
        DeviceKind dk;
        const HResult hr = select(kind, device, &dk);
        if (gdk::failed(hr)) {
            return hr;
        }
        Snapshot s;
        bool have = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            poll_locked(dk, static_cast<bool>(device));
            const ReadingRing<kHistory>& h = history_[index(dk)];
            if (!h.empty()) {
                s = h.back();
                have = true;
            }
        }
        if (!have) {
            return kGameInputReadingNotFound;
        }
        return emit(s, out);
    }

    // 5
    HResult GetNextReading(com::InterfaceArg<GameInputReadingObj> ref, uint32_t kind,
                           com::InterfaceArg<GameInputDeviceObj> device, OutPtr out) {
        return neighbour(ref, kind, device, out, true);
    }

    // 6
    HResult GetPreviousReading(com::InterfaceArg<GameInputReadingObj> ref, uint32_t kind,
                               com::InterfaceArg<GameInputDeviceObj> device, OutPtr out) {
        return neighbour(ref, kind, device, out, false);
    }

    // 7 (implemented for ring-history temporal lookup; vtable slot 7 stays a named stub in
    // kGameInputMethods while test_xbox_scarlett_guest.cpp probes slot 7 for stub_hit_report,
    // and is exercised directly via playgta6::os::game_input_guest_temporal_reading).
    HResult GetTemporalReading(uint64_t timestamp, com::InterfaceArg<GameInputDeviceObj> device, OutPtr out) {
        if (out.is_null() || !out.write(0)) {
            return com::E_POINTER_HR;
        }
        if (device.invalid()) {
            return com::E_INVALIDARG_HR;
        }
        Snapshot found{};
        bool have = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (device) {
                const DeviceKind dk = device->kind();
                const ReadingRing<kHistory>& h = history_[index(dk)];
                have = h.find_temporal(timestamp, &found);
            } else {
                for (size_t i = 0; i < kDeviceKindCount; ++i) {
                    Snapshot candidate{};
                    if (!history_[i].find_temporal(timestamp, &candidate)) {
                        continue;
                    }
                    if (!have || candidate.timestamp > found.timestamp ||
                        (candidate.timestamp == found.timestamp && candidate.sequence > found.sequence)) {
                        found = candidate;
                        have = true;
                    }
                }
            }
        }
        if (!have) {
            return kGameInputReadingNotFound;
        }
        return emit(found, out);
    }

    // 9: device callbacks are invoked (once per matching device) before this returns for
    // asynchronous (1) and blocking (2) enumeration; the virtual devices never change status.
    HResult RegisterDeviceCallback(com::CallContext& ctx, com::InterfaceArg<GameInputDeviceObj> device, uint32_t kind,
                                   uint32_t status_filter, uint32_t enumeration, uint64_t context,
                                   uint64_t callback, OutPtr token_out) {
        if (callback == 0 || device.invalid()) {
            return com::E_INVALIDARG_HR;
        }
        uint64_t token = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            token = next_token_++;
            callbacks_.push_back(token);
        }
        if (!token_out.is_null()) {
            token_out.write(token);
        }
        if (enumeration == 0 || (status_filter & kDeviceStatus) == 0) {
            return kOk;
        }
        com::InterfaceArg<GameInputDeviceObj> devices[kDeviceKindCount];
        size_t device_count = 0;
        devices[device_count++] = gamepad_;
        devices[device_count++] = keyboard_;
        if ((kind & kGameInputKindMouse) != 0) {
            if (com::InterfaceArg<GameInputDeviceObj> m = ensure_mouse_device()) {
                devices[device_count++] = std::move(m);
            }
        }
        for (size_t i = 0; i < device_count; ++i) {
            const com::InterfaceArg<GameInputDeviceObj>& d = devices[i];
            if (!d || (kind & input_kind_of(d->kind())) == 0 || (device && device.get() != d.get())) {
                continue;
            }
            const com::GuestFunction<void(uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t)> fn(callback);
            fn.call(ctx.state, token, context, d->interface_pointer(kIDeviceRef()), now_us(), kDeviceStatus, 0u);
            if (ctx.state.unwind_pending != 0) {
                return kOk;  // a guest exception left the callback: the bridge drops the result
            }
        }
        return kOk;
    }

    // 10: RegisterGuideButtonCallback(device, context, callbackFunc, callbackToken*)
    HResult RegisterGuideButtonCallback(const com::InterfaceArg<GameInputDeviceObj>& device, uint64_t /*context*/,
                                        uint64_t callback, GuestPtr<uint64_t> out_token) {
        return register_silent_callback(device, callback, out_token);
    }
    // 11: RegisterKeyboardLayoutCallback(device, context, callbackFunc, callbackToken*)
    HResult RegisterKeyboardLayoutCallback(const com::InterfaceArg<GameInputDeviceObj>& device, uint64_t /*context*/,
                                           uint64_t callback, GuestPtr<uint64_t> out_token) {
        return register_silent_callback(device, callback, out_token);
    }

    // 12
    void StopCallback(uint64_t token) { (void)remove(token); }
    // 13
    bool UnregisterCallback(uint64_t token, uint64_t /*timeout_us*/) { return remove(token); }

    // 16
    HResult FindDeviceFromId(GuestPtr<AppLocalDeviceId> id, OutPtr out) {
        if (out.is_null() || !out.write(0)) {
            return com::E_POINTER_HR;
        }
        AppLocalDeviceId want{};
        if (!id.read(want)) {
            return com::E_INVALIDARG_HR;
        }
        for (const com::InterfaceArg<GameInputDeviceObj>* d : {&gamepad_, &keyboard_}) {
            if (std::memcmp(want.value, (*d)->id().value, sizeof(want.value)) == 0) {
                const uint64_t p = GameInputReadingObj::share(*d);
                out.write(p);
                return p != 0 ? kOk : kGameInputDeviceNotFound;
            }
        }
        const AppLocalDeviceId mouse_id = make_device_id(DeviceKind::Mouse);
        if (std::memcmp(want.value, mouse_id.value, sizeof(want.value)) == 0) {
            const com::InterfaceArg<GameInputDeviceObj> m = ensure_mouse_device();
            const uint64_t p = GameInputReadingObj::share(m);
            out.write(p);
            return p != 0 ? kOk : kGameInputDeviceNotFound;
        }
        return kGameInputDeviceNotFound;
    }

    // 21
    void SetFocusPolicy(uint32_t /*policy*/) {}

    static const com::Interface<GameInputDeviceObj>& kIDeviceRef() { return GameInputReadingObj::kIDevice(); }

private:
    static size_t index(DeviceKind k) { return static_cast<size_t>(k); }

    uint64_t next_timestamp_locked() {
        uint64_t ts = now_us();
        if (ts <= last_timestamp_) {
            ts = last_timestamp_ + 1;
        }
        last_timestamp_ = ts;
        return ts;
    }

    HResult select(uint32_t kind, const com::InterfaceArg<GameInputDeviceObj>& device, DeviceKind* out) const {
        if (device.invalid()) {
            return com::E_INVALIDARG_HR;
        }
        if ((kind & kGameInputKindGamepad) != 0 && (!device || device->kind() == DeviceKind::Gamepad)) {
            *out = DeviceKind::Gamepad;
            return kOk;
        }
        if ((kind & kGameInputKindKeyboard) != 0 && (!device || device->kind() == DeviceKind::Keyboard)) {
            *out = DeviceKind::Keyboard;
            return kOk;
        }
        if ((kind & kGameInputKindMouse) != 0 && (!device || device->kind() == DeviceKind::Mouse)) {
            *out = DeviceKind::Mouse;
            return kOk;
        }
        return kGameInputReadingNotFound;
    }

    // Reads the host InputState (through the O12 host GameInput, which applies the button map,
    // deadzones, mouse delta accumulation, and WASD-to-stick synthesis) and records a new snapshot
    // into the fixed-capacity ring buffer when state changed.
    void poll_locked(DeviceKind kind, bool explicit_device = false) {
        GameInputGamepadState pad{};
        Pg6GameInputMouseState hm{};
        uint32_t raw = 0;
        uint32_t host_seq = 0;
        Pg6IGameInputReading* r = nullptr;
        if (host_ != nullptr && gdk::succeeded(host_->GetCurrentReading(scarlett::kGameInputKindGamepad, nullptr, &r)) &&
            r != nullptr) {
            Pg6GameInputGamepadState hp{};
            r->GetGamepadState(&hp);
            pad.buttons = hp.buttons;
            pad.left_trigger = hp.left_trigger;
            pad.right_trigger = hp.right_trigger;
            pad.left_thumbstick_x = hp.left_thumbstick_x;
            pad.left_thumbstick_y = hp.left_thumbstick_y;
            pad.right_thumbstick_x = hp.right_thumbstick_x;
            pad.right_thumbstick_y = hp.right_thumbstick_y;
            sanitize_gamepad_state(pad);
            r->GetMouseState(&hm);
            raw = r->GetRawButtons();
            host_seq = static_cast<uint32_t>(r->GetSequenceNumber(scarlett::kGameInputKindGamepad));
            r->Release();
        }

        const bool observed_mouse_delta = hm.buttons != last_observed_mouse_.buttons ||
                                          hm.position_x != last_observed_mouse_.position_x ||
                                          hm.position_y != last_observed_mouse_.position_y ||
                                          hm.wheel_x != last_observed_mouse_.wheel_x ||
                                          hm.wheel_y != last_observed_mouse_.wheel_y;
        if (observed_mouse_delta) {
            mouse_activity_seen_ = true;
            last_observed_mouse_ = hm;
        }

        ReadingRing<kHistory>& h = history_[index(kind)];
        if (kind == DeviceKind::Mouse) {
            if (!mouse_activity_seen_ && !explicit_device && !has_mouse_device()) {
                return;
            }
            if (!h.empty()) {
                const GameInputMouseState& prev = h.back().mouse;
                const bool changed = hm.buttons != prev.buttons || hm.position_x != prev.position_x ||
                                     hm.position_y != prev.position_y || hm.wheel_x != prev.wheel_x ||
                                     hm.wheel_y != prev.wheel_y;
                if (!changed) {
                    return;
                }
            }
            Snapshot s{};
            s.kind = DeviceKind::Mouse;
            s.sequence = ++sequence_;
            s.timestamp = next_timestamp_locked();
            s.host_sequence = host_seq;
            s.mouse.buttons = hm.buttons;
            s.mouse.position_x = hm.position_x;
            s.mouse.position_y = hm.position_y;
            s.mouse.wheel_x = hm.wheel_x;
            s.mouse.wheel_y = hm.wheel_y;
            h.push(s);
            return;
        }

        if (!h.empty() && h.back().host_sequence == host_seq) {
            return;
        }
        Snapshot s{};
        s.kind = kind;
        s.sequence = ++sequence_;
        s.timestamp = next_timestamp_locked();
        s.host_sequence = host_seq;
        if (kind == DeviceKind::Gamepad) {
            s.gamepad = pad;
        } else {
            os::KeyboardScanEntry decoded[kMaxSnapshotKeys]{};
            const size_t n = os::decode_keyboard_scan_entries(raw & kGuestSabKeyboardMask, decoded, kMaxSnapshotKeys);
            s.key_count = static_cast<uint32_t>(n);
            for (size_t i = 0; i < n; ++i) {
                s.keys[i] = GameInputKeyState{
                    decoded[i].scan_code, decoded[i].code_point, decoded[i].virtual_key, decoded[i].is_dead_key, {0, 0}};
            }
        }
        h.push(s);
    }

    HResult neighbour(const com::InterfaceArg<GameInputReadingObj>& ref, uint32_t kind,
                      const com::InterfaceArg<GameInputDeviceObj>& device, OutPtr out, bool next) {
        if (out.is_null() || !out.write(0)) {
            return com::E_POINTER_HR;
        }
        if (!ref) {
            return com::E_INVALIDARG_HR;
        }
        DeviceKind dk;
        const HResult hr = select(kind, device, &dk);
        if (gdk::failed(hr)) {
            return hr;
        }
        const uint64_t at = ref->snapshot().sequence;
        Snapshot found{};
        bool have = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (next) {
                poll_locked(dk, static_cast<bool>(device));
            }
            const ReadingRing<kHistory>& h = history_[index(dk)];
            have = next ? h.find_next(at, &found) : h.find_previous(at, &found);
        }
        if (!have) {
            return kGameInputReadingNotFound;
        }
        return emit(found, out);
    }

    HResult emit(const Snapshot& s, OutPtr out);

    HResult register_silent_callback(const com::InterfaceArg<GameInputDeviceObj>& device, uint64_t callback,
                                     GuestPtr<uint64_t> out_token) {
        if (device.invalid() || callback == 0) {
            return com::E_INVALIDARG_HR;
        }
        uint64_t token = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            token = next_token_++;
            callbacks_.push_back(token);
        }
        if (!out_token.is_null() && !out_token.write(token)) {
            (void)remove(token);
            return com::E_POINTER_HR;
        }
        return kOk;
    }

    bool remove(uint64_t token) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = std::find(callbacks_.begin(), callbacks_.end(), token);
        if (it == callbacks_.end()) {
            return false;
        }
        callbacks_.erase(it);
        return true;
    }

    Pg6IGameInput* host_;
    com::InterfaceArg<GameInputDeviceObj> gamepad_;
    com::InterfaceArg<GameInputDeviceObj> keyboard_;
    std::mutex mutex_;
    ReadingRing<kHistory> history_[kDeviceKindCount];
    Pg6GameInputMouseState last_observed_mouse_{};
    bool mouse_activity_seen_ = false;
    uint64_t sequence_ = 0;
    uint64_t last_timestamp_ = 0;
    uint64_t next_token_ = 1;
    std::vector<uint64_t> callbacks_;
};

// ===========================================================================
// xg_xs
// ===========================================================================


struct ElementInfo {
    uint32_t block = 1;  // block width/height in texels (4 for BC formats)
    uint32_t bytes = 4;  // bytes per element (block)
};

ElementInfo element_info(uint32_t f) {
    if (f >= 1 && f <= 4) return {1, 16};
    if (f >= 5 && f <= 8) return {1, 12};
    if (f >= 9 && f <= 22) return {1, 8};
    if (f >= 23 && f <= 47) return {1, 4};
    if (f >= 48 && f <= 60) return {1, 2};
    if (f >= 61 && f <= 65) return {1, 1};
    if ((f >= 70 && f <= 72) || (f >= 79 && f <= 81)) return {4, 8};
    if ((f >= 73 && f <= 78) || (f >= 82 && f <= 84) || (f >= 94 && f <= 99)) return {4, 16};
    if (f == 85 || f == 86 || f == 115) return {1, 2};
    return {1, 4};
}

// D3D12 standard 64 KiB tile shape (in texels).
XgTileShape tile_shape(uint32_t dimension, ElementInfo e) {
    XgTileShape t{1, 1, 1};
    if (dimension == 4) {  // Texture3D
        switch (e.bytes) {
            case 1: t = {64, 32, 32}; break;
            case 2: t = {32, 32, 32}; break;
            case 4: t = {32, 32, 16}; break;
            case 8: t = {32, 16, 16}; break;
            default: t = {16, 16, 16}; break;
        }
    } else if (dimension == 2) {  // Texture1D
        t = {65536 / e.bytes, 1, 1};
    } else {
        switch (e.bytes) {
            case 1: t = {256, 256, 1}; break;
            case 2: t = {256, 128, 1}; break;
            case 4: t = {128, 128, 1}; break;
            case 8: t = {128, 64, 1}; break;
            default: t = {64, 64, 1}; break;
        }
    }
    t.width_in_texels *= e.block;
    if (dimension != 2) {
        t.height_in_texels *= e.block;
    }
    return t;
}

uint32_t ceil_div(uint64_t a, uint64_t b) {
    return static_cast<uint32_t>((a + b - 1) / b);
}


class XgComputer final : public com::ComObject {
public:
    static constexpr const char* kComClassName = "xg_xs.XGTextureAddressComputer";

    XgComputer(Pg6IXGTextureAddressComputer* host, const XgResourceDesc& desc) : host_(host), desc_(desc) {
        host_->GetResourceLayout(&layout_);
        init_tiled();
    }
    ~XgComputer() override { host_->Release(); }

    uint32_t AddRef() { return add_ref(); }    // 0
    uint32_t Release() { return release(); }   // 1

    // 2
    HResult GetResourceLayout(uint64_t out) {
        if (out == 0) {
            return com::E_POINTER_HR;
        }
        std::vector<uint8_t> bytes(kXgResourceLayoutSize, 0);
        const uint32_t mips = std::min<uint32_t>(layout_.mip_levels, kXgMaxMips);
        const uint32_t planes = planar_ ? 2u : 1u;
        const uint64_t total = tiled_ ? surface_bytes_ : layout_.size_bytes;
        const uint64_t base_alignment = tiled_ ? std::max<uint64_t>(layout_.base_alignment_bytes, tiled_block_bytes_)
                                               : layout_.base_alignment_bytes;
        std::memcpy(bytes.data() + 0x00, &total, 8);
        std::memcpy(bytes.data() + 0x08, &base_alignment, 8);
        std::memcpy(bytes.data() + 0x10, &mips, 4);
        std::memcpy(bytes.data() + 0x14, &planes, 4);
        for (uint32_t p = 0; p < planes; ++p) {
            uint8_t* plane = bytes.data() + kXgLayoutPlaneOffset + p * kXgLayoutPlaneSize;
            // Plane header (hypothesis): +0x00 usage (planar: 1 depth, 2 stencil), +0x04 bytes per
            // element, +0x08 size, +0x10 offset from the resource base, +0x18 base alignment.
            const playgta6::gpu::gfx10::TextureLayoutDesc& td = planar_ ? plane_desc_[p] : tiled_desc_;
            const ElementInfo e = planar_ ? plane_element(p) : element_info(desc_.format);
            const uint64_t plane_size = planar_ ? plane_bytes_[p] : total;
            const uint64_t plane_offset = planar_ ? plane_offset_[p] : 0;
            if (planar_) {
                const uint32_t usage = p + 1;
                std::memcpy(plane + 0x00, &usage, 4);
            }
            std::memcpy(plane + 0x04, &e.bytes, 4);
            std::memcpy(plane + 0x08, &plane_size, 8);
            std::memcpy(plane + 0x10, &plane_offset, 8);
            std::memcpy(plane + 0x18, &base_alignment, 8);
            for (uint32_t m = 0; m < mips; ++m) {
                const Pg6XgMipLevelLayout& hm = layout_.mips[m];
                XgMipLayout gm{};
                if (tiled_) {
                    // GFX10 placement (gpu/gfx10_detile.h): the same source the detiler and
                    // CopyIntoSubresource use. Mip offsets are relative to the plane.
                    const playgta6::gpu::gfx10::MipPlacement mp = playgta6::gpu::gfx10::mip_placement(td, m);
                    gm.size_bytes = mp.size_bytes * slices_;
                    gm.offset_bytes = mp.offset_bytes;
                    gm.slice_2d_size_bytes = mp.slice_bytes;
                    gm.pitch_bytes = mp.pitch_elements * e.bytes;
                    gm.pitch_pixels = mp.pitch_elements * e.block;
                    gm.alignment_bytes = static_cast<uint32_t>(mp.block_bytes);
                    gm.padded_width_elements = mp.pitch_elements;
                    gm.padded_height_elements = mp.height_elements;
                    gm.padded_depth_or_array_size = desc_.dimension == 4 ? hm.depth : layout_.array_size;
                    gm.width_elements = ceil_div(hm.width, e.block);
                    gm.height_elements = ceil_div(hm.height, e.block);
                    gm.depth_or_array_size = gm.padded_depth_or_array_size;
                    gm.sample_count = 1;
                    std::memcpy(plane + kXgPlaneMipOffset + m * kXgMipLayoutSize, &gm, sizeof(gm));
                    continue;
                }
                gm.size_bytes = hm.size_bytes;
                gm.offset_bytes = hm.offset_bytes;
                gm.slice_2d_size_bytes = hm.slice_size_bytes;
                gm.pitch_bytes = hm.row_pitch_bytes;
                gm.pitch_pixels = hm.row_pitch_bytes / e.bytes * e.block;
                gm.alignment_bytes = 256;
                gm.padded_width_elements = hm.row_pitch_bytes / e.bytes;
                gm.padded_height_elements = hm.padded_rows;
                gm.padded_depth_or_array_size = desc_.dimension == 4 ? hm.depth : layout_.array_size;
                gm.width_elements = ceil_div(hm.width, e.block);
                gm.height_elements = ceil_div(hm.height, e.block);
                gm.depth_or_array_size = gm.padded_depth_or_array_size;
                gm.sample_count = std::max<uint32_t>(desc_.sample_count, 1);
                std::memcpy(plane + kXgPlaneMipOffset + m * kXgMipLayoutSize, &gm, sizeof(gm));
            }
        }
        return com::write_guest(out, bytes.data(), bytes.size()) ? kOk : com::E_POINTER_HR;
    }

    // 8
    HResult CopyIntoSubresource(uint64_t dst, uint32_t plane, uint32_t subresource, uint64_t src, uint32_t row_pitch,
                                uint32_t slice_pitch) {
        if (dst == 0 || src == 0) {
            return com::E_POINTER_HR;
        }
        if (layout_.mip_levels == 0) {
            return com::E_INVALIDARG_HR;
        }
        if (tiled_) {
            const HResult hr = copy_into_tiled(dst, plane, subresource, src, row_pitch, slice_pitch);
            guest_window::mark_written(dst, static_cast<size_t>(layout_.size_bytes));  // O37
            return hr;
        }
        const Pg6XgMipLevelLayout& ml = layout_.mips[subresource % layout_.mip_levels];
        const uint32_t depth = desc_.dimension == 4 ? ml.depth : 1u;
        const uint64_t src_len = depth > 1 ? static_cast<uint64_t>(slice_pitch) * depth
                                           : static_cast<uint64_t>(row_pitch) * ml.padded_rows;
        void* d = com::guest_to_host(dst, layout_.size_bytes);
        const void* s = com::guest_to_host(src, src_len);
        // A range that is not one host piece (a translated-window buffer over non-adjacent pages,
        // O29) goes through a bounce copy, segment by segment.
        std::vector<uint8_t> dst_bounce;
        std::vector<uint8_t> src_bounce;
        if (s == nullptr) {
            src_bounce.resize(static_cast<size_t>(src_len));
            if (!com::read_guest(src, src_bounce.data(), src_bounce.size())) {
                return com::E_INVALIDARG_HR;
            }
            s = src_bounce.data();
        }
        if (d == nullptr) {
            dst_bounce.resize(static_cast<size_t>(layout_.size_bytes));
            if (!com::read_guest(dst, dst_bounce.data(), dst_bounce.size())) {
                return com::E_INVALIDARG_HR;
            }
            d = dst_bounce.data();
        }
        const HResult hr = host_->CopyIntoSubresource(d, plane, subresource, s, row_pitch, slice_pitch);
        if (!dst_bounce.empty() && !com::write_guest(dst, dst_bounce.data(), dst_bounce.size())) {
            return com::E_INVALIDARG_HR;
        }
        guest_window::mark_written(dst, static_cast<size_t>(layout_.size_bytes));  // O37
        return hr;
    }

    // 9 CopyFromSubresource(linearDst, plane, subresource, reserved, tiledSrc, rowPitch, slicePitch):
    // the inverse of slot 8 (resource layout -> linear rows). Shape from the title's one call site
    // (sub_140979E30): rdx = its linear buffer, r8 = 0, r9 = subresource, a 32-bit 0 on the stack,
    // then the resource base and the destination row / slice pitch; HRESULT, success when >= 0.
    // When the wanted pitches differ from the ones it passes, the title copies the linear result
    // row by row into the final buffer itself. The 32-bit fourth argument is always 0 there; its
    // meaning is unknown, so a non-zero value is refused (E_INVALIDARG) instead of guessed.
    HResult CopyFromSubresource(uint64_t dst, uint32_t plane, uint32_t subresource, uint32_t reserved, uint64_t src,
                                uint32_t row_pitch, uint32_t slice_pitch) {
        if (dst == 0 || src == 0) {
            return com::E_POINTER_HR;
        }
        if (layout_.mip_levels == 0) {
            return com::E_INVALIDARG_HR;
        }
        if (reserved != 0) {
            if (xg_reserved_refusals().fetch_add(1) == 0) {
                diag("xg_xs: CopyFromSubresource with non-zero fourth argument 0x" + hex32(reserved) +
                     " refused (meaning unknown; counted)");
            }
            return com::E_INVALIDARG_HR;
        }
        if (tiled_) {
            return copy_from_tiled(dst, plane, subresource, src, row_pitch, slice_pitch);
        }
        if (plane != 0) {
            return com::E_INVALIDARG_HR;
        }
        const Pg6XgMipLevelLayout& ml = layout_.mips[subresource % layout_.mip_levels];
        const uint32_t depth = desc_.dimension == 4 ? ml.depth : 1u;
        const ElementInfo e = element_info(desc_.format);
        const uint64_t row_bytes = uint64_t{ceil_div(ml.width, e.block)} * e.bytes;
        if (row_pitch < row_bytes || (depth > 1 && slice_pitch < uint64_t{row_pitch} * ml.padded_rows)) {
            return com::E_INVALIDARG_HR;
        }
        const uint64_t dst_len = depth > 1 ? uint64_t{slice_pitch} * (depth - 1) + uint64_t{row_pitch} * ml.padded_rows
                                           : uint64_t{row_pitch} * ml.padded_rows;
        const void* s = com::guest_to_host(src, layout_.size_bytes);
        void* d = com::guest_to_host(dst, dst_len);
        // Ranges that are not one host piece go through bounce copies (as in slot 8).
        std::vector<uint8_t> src_bounce;
        std::vector<uint8_t> dst_bounce;
        if (s == nullptr) {
            src_bounce.resize(static_cast<size_t>(layout_.size_bytes));
            if (!com::read_guest(src, src_bounce.data(), src_bounce.size())) {
                return com::E_INVALIDARG_HR;
            }
            s = src_bounce.data();
        }
        if (d == nullptr) {
            dst_bounce.resize(static_cast<size_t>(dst_len));
            if (!com::read_guest(dst, dst_bounce.data(), dst_bounce.size())) {
                return com::E_INVALIDARG_HR;
            }
            d = dst_bounce.data();
        }
        const HResult hr = host_->CopyFromSubresource(d, row_pitch, slice_pitch, s, plane, subresource);
        if (!dst_bounce.empty() && !com::write_guest(dst, dst_bounce.data(), dst_bounce.size())) {
            return com::E_INVALIDARG_HR;
        }
        guest_window::mark_written(dst, static_cast<size_t>(dst_len));  // O37
        return hr;
    }

    // 6 GetTexelElementOffsetBytes(plane, mipLevel, x, y, zOrSlice, sample, bitOffset*) -> UINT64:
    // byte offset from the resource base of element (x, y) of the mip in slice zOrSlice. The title
    // passes element coordinates (for BC formats it divides texel coordinates by 4 itself). Tiled
    // and console linear layouts use the GFX10 address the detiler and CopyIntoSubresource use;
    // other layouts the host computer's linear mips. Elements are whole bytes: *bitOffset = 0.
    // UINT64_MAX for an invalid plane, mip or coordinate. Backlog item 15 suggested 0 instead; kept
    // as UINT64_MAX because the only evidence found points that way: the sibling
    // GetMipLevelOffsetBytes returns ~0 for bad input, OpenScarlett's XG shim returns UINT64_MAX
    // here, and no public/GDK text or title call-site evidence supports 0 (0 is also a valid
    // offset, so it could not signal failure). Revisit only with a title call site that depends on 0.
    uint64_t GetTexelElementOffsetBytes(uint32_t plane, uint32_t mip, uint64_t x, uint32_t y, uint32_t z,
                                        uint32_t /*sample*/, GuestPtr<uint32_t> bit_offset) {
        if (!bit_offset.is_null()) {
            bit_offset.write(0);
        }
        if (plane >= (planar_ ? 2u : 1u) || mip >= layout_.mip_levels || x > 0xFFFFFFFFull) {
            return UINT64_MAX;
        }
        if (tiled_) {
            const bool is_3d = desc_.dimension == 4;
            if (!is_3d && z >= slices_) {
                return UINT64_MAX;
            }
            const uint64_t chain = planar_ ? plane_chain_[plane] : chain_bytes_;
            const uint64_t offset = playgta6::gpu::gfx10::surface_element_offset(
                planar_ ? plane_desc_[plane] : tiled_desc_, mip, static_cast<uint32_t>(x), y, is_3d ? z : 0u);
            return offset == UINT64_MAX
                       ? offset
                       : (planar_ ? plane_offset_[plane] : 0u) + (is_3d ? 0u : uint64_t{z} * chain) + offset;
        }
        const Pg6XgMipLevelLayout& ml = layout_.mips[mip];
        const ElementInfo e = element_info(desc_.format);
        return ml.offset_bytes + uint64_t{z} * ml.slice_size_bytes + uint64_t{y} * ml.row_pitch_bytes + x * e.bytes;
    }

    // 10 (D3D12 GetResourceTiling shape; standard 64 KiB tiles, numbered in GFX10 memory order for
    // the 64KB swizzle modes)
    void GetResourceTiling(GuestPtr<uint32_t> num_tiles, GuestPtr<XgPackedMipInfo> packed,
                           GuestPtr<XgTileShape> shape, GuestPtr<uint32_t> num_tilings, uint32_t first,
                           uint64_t tilings) {
        const ElementInfo e = element_info(desc_.format);
        XgTileShape t = tile_shape(desc_.dimension, e);
        const uint32_t mips = layout_.mip_levels;
        const uint32_t slices = desc_.dimension == 4 ? 1u : std::max<uint32_t>(layout_.array_size, 1);
        namespace gfx10 = playgta6::gpu::gfx10;
        // Only the 64 KiB GFX10 modes use the memory-order block map. D3D12 tiled resources always
        // have 64 KiB tiles and the title sizes and maps them that way, so the 4 KiB and 256 B
        // modes (and console linear) get the plain D3D12 answer: the standard 64 KiB tile shape,
        // standard mips = the mips at least one tile big in every dimension, the rest one packed
        // tail (a 4 KiB-block tile count would over-allocate 16x and make tall textures report
        // 7+ standard mips, which the title rejects).
        const bool gfx10_tiled = tiled_ && tiled_desc_.d3d12_layout != gfx10::kConsoleLinearLayout;
        // GFX10 64KB modes take the standard/packed split and tile numbers from the shared
        // gfx10::tile_map (addrlib mip tail rule, not the D3D12 packed-mip rule: a 2:1 mip such as
        // 64x32 BC7 blocks keeps a block of its own and the tail follows it; the title streams each
        // standard mip and the tail as separate reads matching the archive's per-block chunks).
        gfx10::TileMap tm{};
        if (gfx10_tiled) {
            tm = gfx10::tile_map(tiled_desc_);
            if (tm.block_bytes != 65536u) {
                tm = gfx10::TileMap{};
            }
        }
        uint32_t standard = 0;
        if (tm.ok) {
            standard = tm.standard_mips;
            t = XgTileShape{tm.block_width * e.block, tm.block_height * e.block,
                            desc_.dimension == 4 ? tm.block_depth : 1u};
        }
        while (!tm.ok && standard < mips) {
            const Pg6XgMipLevelLayout& m = layout_.mips[standard];
            if (m.width < t.width_in_texels || m.height < t.height_in_texels ||
                (desc_.dimension == 4 && m.depth < t.depth_in_texels)) {
                break;
            }
            ++standard;
        }
        std::vector<XgSubresourceTiling> per_mip(mips);
        uint32_t tiles_per_slice = 0;
        for (uint32_t m = 0; m < standard; ++m) {
            const Pg6XgMipLevelLayout& ml = layout_.mips[m];
            XgSubresourceTiling& st = per_mip[m];
            st.width_in_tiles = ceil_div(ml.width, t.width_in_texels);
            st.height_in_tiles = static_cast<uint16_t>(ceil_div(ml.height, t.height_in_texels));
            st.depth_in_tiles = static_cast<uint16_t>(ceil_div(ml.depth, t.depth_in_texels));
            st.start_tile_index_in_overall_resource = tiles_per_slice;
            tiles_per_slice += st.width_in_tiles * st.height_in_tiles * st.depth_in_tiles;
        }
        uint64_t packed_bytes = 0;
        for (uint32_t m = standard; m < mips; ++m) {
            per_mip[m] = XgSubresourceTiling{0, 0, 0, kXgPackedTile};
            packed_bytes += layout_.mips[m].slice_size_bytes * (desc_.dimension == 4 ? layout_.mips[m].depth : 1u);
        }
        uint32_t packed_tiles =
            standard < mips ? std::max<uint32_t>(1, ceil_div(packed_bytes, 65536)) : 0;
        uint32_t packed_start = tiles_per_slice;
        // GFX10 64KB modes (verified placement): a tile is one 64 KiB block, and the tiles are
        // numbered in memory order. addrlib puts the mip tail (one block) first and the larger
        // mips after it smallest-first, so mip 0 has the highest start tile; the title assumes
        // the tail at tile 0 and reads each mip at its start tile.
        if (tm.ok) {
            for (uint32_t m = 0; m < standard; ++m) {
                per_mip[m].start_tile_index_in_overall_resource = static_cast<uint32_t>(tm.start_tile[m]);
            }
            if (standard < mips) {
                packed_tiles = tm.tail_tiles;
                packed_start = static_cast<uint32_t>(tm.tail_start_tile);
            }
        }
        const uint32_t slice_tiles = tm.ok ? static_cast<uint32_t>(tm.slice_tiles) : tiles_per_slice + packed_tiles;

        if (!num_tiles.is_null()) {
            num_tiles.write(slice_tiles * slices);
        }
        if (!packed.is_null()) {
            XgPackedMipInfo p{};
            p.num_standard_mips = static_cast<uint8_t>(standard);
            p.num_packed_mips = static_cast<uint8_t>(mips - standard);
            p.num_tiles_for_packed_mips = packed_tiles;
            p.start_tile_index_in_overall_resource = packed_start;
            packed.write(p);
        }
        if (!shape.is_null()) {
            shape.write(t);
        }
        uint32_t capacity = 0;
        if (!num_tilings.is_null() && num_tilings.read(capacity)) {
            const uint32_t total = mips * slices;
            const uint32_t n = first >= total ? 0 : std::min(capacity, total - first);
            for (uint32_t i = 0; i < n && tilings != 0; ++i) {
                const uint32_t sub = first + i;
                XgSubresourceTiling st = per_mip[sub % mips];
                if (st.start_tile_index_in_overall_resource != kXgPackedTile) {
                    st.start_tile_index_in_overall_resource += (sub / mips) * slice_tiles;
                }
                com::write_guest(tilings + 12ull * i, st);
            }
            num_tilings.write(n);
        }
    }

private:
    // A 2D (array slices repeat the chain) or 3D descriptor whose swizzle mode (+0x2C, 0x100 | n) is a GFX10 mode
    // verified against addrlib (256B/4KB/64KB _S and _D; thick 3D for _S), or the console linear
    // mode 0x100 (addrlib ADDR_SW_LINEAR; 1D too, #678): GetResourceLayout reports the GFX10
    // surface and CopyIntoSubresource writes the GFX10 layout the runtime's detiler reads back
    // (gpu/gfx10_detile.h, one placement for the allocation size, XG and the upload). Other tiled
    // modes keep the host computer's answer and are counted.
    void init_tiled() {
        namespace gfx10 = playgta6::gpu::gfx10;
        const gfx10::SwizzleMode mode = gfx10::decode_d3d12_layout(desc_.swizzle_mode);
        const bool linear = desc_.swizzle_mode == gfx10::kConsoleLinearLayout;
        if ((mode == gfx10::SwizzleMode::Linear && !linear) || mode == gfx10::SwizzleMode::Unsupported) {
            return;
        }
        const bool verified = linear || mode == gfx10::SwizzleMode::Sw256BS || mode == gfx10::SwizzleMode::Sw256BD ||
                              mode == gfx10::SwizzleMode::Sw4KBS || mode == gfx10::SwizzleMode::Sw4KBD ||
                              mode == gfx10::SwizzleMode::Sw64KBS || mode == gfx10::SwizzleMode::Sw64KBD;
        if (!verified || (desc_.dimension != 3 && desc_.dimension != 4 && !(linear && desc_.dimension == 2)) ||
            desc_.sample_count > 1 || layout_.mip_levels == 0) {
            if (xg_linear_answers().fetch_add(1) == 0) {
                diag("xg_xs: swizzle mode 0x" + hex32(desc_.swizzle_mode) +
                     " answered with the linear layout (mode not verified or MSAA; counted)");
            }
            return;
        }
        tiled_desc_.dxgi_format = desc_.format;
        tiled_desc_.width = static_cast<uint32_t>(desc_.width);
        tiled_desc_.height = desc_.dimension == 2 ? 1u : std::max<uint32_t>(desc_.height, 1u);
        tiled_desc_.depth = desc_.dimension == 4 ? std::max<uint32_t>(desc_.depth_or_array_size, 1u) : 1u;
        tiled_desc_.mip_levels = layout_.mip_levels;
        tiled_desc_.d3d12_layout = desc_.swizzle_mode;
        // 1D/2D arrays repeat the chain per slice (addrlib: slice stride = mip-chain slice size).
        slices_ = desc_.dimension != 4 ? std::max<uint32_t>(desc_.depth_or_array_size, 1u) : 1u;
        // Depth/stencil formats: two tiled surfaces, depth then stencil (the allocation sizing's
        // plane rule, gfx10::depth_stencil_planes), each its own mip chain x slices.
        const gfx10::PlanarSplit split = gfx10::depth_stencil_planes(desc_.format);
        if (split.planes == 2) {
            uint64_t offset = 0;
            for (uint32_t p = 0; p < 2; ++p) {
                plane_desc_[p] = tiled_desc_;
                plane_desc_[p].dxgi_format = split.format[p];
                plane_chain_[p] = gfx10::surface_bytes(plane_desc_[p]);
                plane_bytes_[p] = plane_chain_[p] * slices_;
                if (plane_bytes_[p] == 0) {
                    xg_linear_answers().fetch_add(1);
                    return;
                }
                plane_offset_[p] = offset;
                offset += plane_bytes_[p];  // whole swizzle blocks: the next plane stays aligned
            }
            tiled_desc_ = plane_desc_[0];
            chain_bytes_ = plane_chain_[0];
            surface_bytes_ = static_cast<size_t>(offset);
            tiled_block_bytes_ = gfx10::mip_placement(plane_desc_[0], 0).block_bytes;
            planar_ = true;
            tiled_ = true;
            return;
        }
        chain_bytes_ = gfx10::surface_bytes(tiled_desc_);
        const size_t bytes = chain_bytes_ * slices_;
        if (bytes == 0) {
            xg_linear_answers().fetch_add(1);
            return;
        }
        surface_bytes_ = bytes;
        tiled_block_bytes_ = gfx10::mip_placement(tiled_desc_, 0).block_bytes;
        tiled_ = true;
    }

    static std::atomic<uint64_t>& xg_linear_answers() {
        static std::atomic<uint64_t> n{0};
        return n;
    }

    HResult copy_into_tiled(uint64_t dst, uint32_t plane, uint32_t subresource, uint64_t src, uint32_t row_pitch,
                            uint32_t slice_pitch) {
        namespace gfx10 = playgta6::gpu::gfx10;
        if (plane >= (planar_ ? 2u : 1u) || subresource >= tiled_desc_.mip_levels * slices_) {
            return com::E_INVALIDARG_HR;
        }
        const gfx10::TextureLayoutDesc& td = planar_ ? plane_desc_[plane] : tiled_desc_;
        const uint64_t chain_bytes = planar_ ? plane_chain_[plane] : chain_bytes_;
        if (planar_) {
            dst += plane_offset_[plane];
        }
        const gfx10::FormatGeometry g = gfx10::format_geometry(td.dxgi_format);
        const uint32_t mip = subresource % tiled_desc_.mip_levels;
        dst += uint64_t{subresource / td.mip_levels} * chain_bytes;
        const uint32_t w = std::max(1u, td.width >> mip);
        const uint32_t h = std::max(1u, td.height >> mip);
        const uint32_t d = std::max(1u, td.depth >> mip);
        const uint32_t row_bytes = (w + g.block_width - 1u) / g.block_width * g.bytes_per_block;
        const uint32_t rows = (h + g.block_height - 1u) / g.block_height;
        if (row_pitch < row_bytes || (d > 1 && slice_pitch < uint64_t{row_pitch} * rows)) {
            return com::E_INVALIDARG_HR;
        }
        std::vector<uint8_t> linear(size_t{row_bytes} * rows * d);
        for (uint32_t z = 0; z < d; ++z) {
            for (uint32_t y = 0; y < rows; ++y) {
                const uint64_t from = src + uint64_t{z} * slice_pitch + uint64_t{y} * row_pitch;
                if (!com::read_guest(from, linear.data() + (size_t{z} * rows + y) * row_bytes, row_bytes)) {
                    return com::E_INVALIDARG_HR;
                }
            }
        }
        std::vector<uint8_t> surface(chain_bytes);
        if (!com::read_guest(dst, surface.data(), surface.size()) ||
            !gfx10::tile_into_surface(td, mip, linear.data(), linear.size(), surface.data(), surface.size()) ||
            !com::write_guest(dst, surface.data(), surface.size())) {
            return com::E_INVALIDARG_HR;
        }
        return kOk;
    }

    // Inverse of copy_into_tiled: detile mip `subresource % mips` of its slice (and plane) from the
    // GFX10 surface at `src` (gpu/gfx10_detile.h, the same placement slot 8 writes) into linear rows
    // at `dst` with the caller's pitches.
    HResult copy_from_tiled(uint64_t dst, uint32_t plane, uint32_t subresource, uint64_t src, uint32_t row_pitch,
                            uint32_t slice_pitch) {
        namespace gfx10 = playgta6::gpu::gfx10;
        if (plane >= (planar_ ? 2u : 1u) || subresource >= tiled_desc_.mip_levels * slices_) {
            return com::E_INVALIDARG_HR;
        }
        const gfx10::TextureLayoutDesc& td = planar_ ? plane_desc_[plane] : tiled_desc_;
        const uint64_t chain_bytes = planar_ ? plane_chain_[plane] : chain_bytes_;
        if (planar_) {
            src += plane_offset_[plane];
        }
        const gfx10::FormatGeometry g = gfx10::format_geometry(td.dxgi_format);
        const uint32_t mip = subresource % tiled_desc_.mip_levels;
        src += uint64_t{subresource / td.mip_levels} * chain_bytes;
        const uint32_t w = std::max(1u, td.width >> mip);
        const uint32_t h = std::max(1u, td.height >> mip);
        const uint32_t d = std::max(1u, td.depth >> mip);
        const uint32_t row_bytes = (w + g.block_width - 1u) / g.block_width * g.bytes_per_block;
        const uint32_t rows = (h + g.block_height - 1u) / g.block_height;
        if (row_pitch < row_bytes || (d > 1 && slice_pitch < uint64_t{row_pitch} * rows)) {
            return com::E_INVALIDARG_HR;
        }
        std::vector<uint8_t> surface(chain_bytes);
        std::vector<uint8_t> linear;
        if (!com::read_guest(src, surface.data(), surface.size()) ||
            !gfx10::detile_from_surface(td, mip, surface.data(), surface.size(), linear).ok ||
            linear.size() < size_t{row_bytes} * rows * d) {
            return com::E_INVALIDARG_HR;
        }
        for (uint32_t z = 0; z < d; ++z) {
            for (uint32_t y = 0; y < rows; ++y) {
                const uint64_t to = dst + uint64_t{z} * slice_pitch + uint64_t{y} * row_pitch;
                if (!com::write_guest(to, linear.data() + (size_t{z} * rows + y) * row_bytes, row_bytes)) {
                    return com::E_INVALIDARG_HR;
                }
            }
        }
        const uint64_t span = uint64_t{d - 1} * slice_pitch + uint64_t{rows - 1} * row_pitch + row_bytes;
        guest_window::mark_written(dst, static_cast<size_t>(span));  // O37
        return kOk;
    }

    static std::atomic<uint64_t>& xg_reserved_refusals() {
        static std::atomic<uint64_t> n{0};
        return n;
    }

    static std::string hex32(uint32_t v) {
        char b[12];
        std::snprintf(b, sizeof(b), "%x", v);
        return b;
    }

    Pg6IXGTextureAddressComputer* host_;
    XgResourceDesc desc_;
    Pg6XgResourceLayout layout_{};
    playgta6::gpu::gfx10::TextureLayoutDesc tiled_desc_{};
    size_t surface_bytes_ = 0;  // whole resource (chain x slices)
    size_t chain_bytes_ = 0;    // one slice's mip chain
    uint32_t slices_ = 1;
    uint64_t tiled_block_bytes_ = 0;
    bool tiled_ = false;
    // Depth/stencil: plane 0 = depth, plane 1 = stencil (tiled_desc_ / chain_bytes_ describe plane 0).
    bool planar_ = false;
    playgta6::gpu::gfx10::TextureLayoutDesc plane_desc_[2]{};
    uint64_t plane_chain_[2] = {};   // one slice's mip chain per plane
    uint64_t plane_bytes_[2] = {};   // chain x slices per plane
    uint64_t plane_offset_[2] = {};  // from the resource base

    ElementInfo plane_element(uint32_t p) const {
        const playgta6::gpu::gfx10::FormatGeometry g = playgta6::gpu::gfx10::format_geometry(plane_desc_[p].dxgi_format);
        return {g.block_width, g.bytes_per_block};
    }
};

// ---------------------------------------------------------------------------
// Interface declarations
// ---------------------------------------------------------------------------


using GM = com::Methods<GameInputApi>;
constexpr com::MethodDesc kGameInputMethods[] = {
    GM::method<&GameInputApi::GetCurrentTimestamp>("GetCurrentTimestamp"),        // 3
    GM::method<&GameInputApi::GetCurrentReading>("GetCurrentReading"),            // 4
    GM::method<&GameInputApi::GetNextReading>("GetNextReading"),                  // 5
    GM::method<&GameInputApi::GetPreviousReading>("GetPreviousReading"),          // 6
    GM::stub("GetTemporalReading"),                                               // 7
    GM::stub("RegisterReadingCallback"),                                          // 8
    GM::method<&GameInputApi::RegisterDeviceCallback>("RegisterDeviceCallback"),                  // 9
    GM::method<&GameInputApi::RegisterGuideButtonCallback>("RegisterGuideButtonCallback"),        // 10
    GM::method<&GameInputApi::RegisterKeyboardLayoutCallback>("RegisterKeyboardLayoutCallback"),  // 11
    GM::method<&GameInputApi::StopCallback>("StopCallback"),                                      // 12
    GM::method<&GameInputApi::UnregisterCallback>("UnregisterCallback"),          // 13
    GM::stub("CreateDispatcher"),                                                 // 14
    GM::stub("CreateAggregateDevice"),                                            // 15
    GM::method<&GameInputApi::FindDeviceFromId>("FindDeviceFromId"),              // 16
    GM::stub("FindDeviceFromObject"),                                             // 17
    GM::stub("FindDeviceFromPlatformHandle"),                                     // 18
    GM::stub("FindDeviceFromPlatformString"),                                     // 19
    GM::stub("EnableOemDeviceSupport"),                                           // 20
    GM::method<&GameInputApi::SetFocusPolicy>("SetFocusPolicy"),                  // 21
};
constexpr auto kIGameInput =
    com::declare_interface<GameInputApi>("GameInput.IGameInput", kIidIGameInput, kGameInputMethods, {.slot_count = 24});

using RM = com::Methods<GameInputReadingObj>;
constexpr com::MethodDesc kReadingMethods[] = {
    RM::method<&GameInputReadingObj::GetInputKind>("GetInputKind"),                   // 3
    RM::method<&GameInputReadingObj::GetSequenceNumber>("GetSequenceNumber"),         // 4
    RM::method<&GameInputReadingObj::GetTimestamp>("GetTimestamp"),                   // 5
    RM::method<&GameInputReadingObj::GetDevice>("GetDevice"),                         // 6
    RM::method<&GameInputReadingObj::GetRawReport>("GetRawReport"),                   // 7
    RM::method<&GameInputReadingObj::GetZeroCount>("GetControllerAxisCount"),         // 8
    RM::method<&GameInputReadingObj::GetZeroState>("GetControllerAxisState"),         // 9
    RM::method<&GameInputReadingObj::GetZeroCount>("GetControllerButtonCount"),       // 10
    RM::method<&GameInputReadingObj::GetZeroState>("GetControllerButtonState"),       // 11
    RM::method<&GameInputReadingObj::GetZeroCount>("GetControllerSwitchCount"),       // 12
    RM::method<&GameInputReadingObj::GetZeroState>("GetControllerSwitchState"),       // 13
    RM::method<&GameInputReadingObj::GetKeyCount>("GetKeyCount"),                     // 14
    RM::method<&GameInputReadingObj::GetKeyState>("GetKeyState"),                     // 15
    RM::method<&GameInputReadingObj::GetMouseState>("GetMouseState"),                 // 16
    RM::method<&GameInputReadingObj::GetZeroCount>("GetTouchCount"),                  // 17
    RM::method<&GameInputReadingObj::GetZeroState>("GetTouchState"),                  // 18
    RM::method<&GameInputReadingObj::GetNoState>("GetMotionState"),                   // 19
    RM::method<&GameInputReadingObj::GetNoState>("GetArcadeStickState"),              // 20
    RM::method<&GameInputReadingObj::GetNoState>("GetFlightStickState"),              // 21
    RM::method<&GameInputReadingObj::GetGamepadState>("GetGamepadState"),             // 22
    RM::method<&GameInputReadingObj::GetNoState>("GetRacingWheelState"),              // 23
    RM::method<&GameInputReadingObj::GetNoState>("GetUiNavigationState"),             // 24
};
constexpr auto kIReading = com::declare_interface<GameInputReadingObj>(
    "GameInput.IGameInputReading", kIidIGameInputReading, kReadingMethods, {.slot_count = 28});

using DM = com::Methods<GameInputDeviceObj>;
constexpr com::MethodDesc kDeviceMethods[] = {
    DM::method<&GameInputDeviceObj::GetDeviceInfo>("GetDeviceInfo"),                                // 3
    DM::method<&GameInputDeviceObj::GetDeviceStatus>("GetDeviceStatus"),                            // 4
    DM::method<&GameInputDeviceObj::GetBatteryState>("GetBatteryState"),                            // 5
    DM::method<&GameInputDeviceObj::CreateForceFeedbackEffect>("CreateForceFeedbackEffect"),        // 6
    DM::method<&GameInputDeviceObj::IsForceFeedbackMotorPoweredOn>("IsForceFeedbackMotorPoweredOn"), // 7
    DM::method<&GameInputDeviceObj::SetForceFeedbackMotorGain>("SetForceFeedbackMotorGain"),        // 8
    DM::method<&GameInputDeviceObj::SetHapticMotorState>("SetHapticMotorState"),                    // 9
    DM::method<&GameInputDeviceObj::SetRumbleState>("SetRumbleState"),          // 10
};
constexpr auto kIDevice = com::declare_interface<GameInputDeviceObj>("GameInput.IGameInputDevice", kIidIGameInputDevice,
                                                                     kDeviceMethods, {.slot_count = 24});

using XM = com::Methods<XgComputer>;
constexpr com::MethodDesc kXgMethods[] = {
    XM::method<&XgComputer::AddRef>("AddRef"),                            // 0
    XM::method<&XgComputer::Release>("Release"),                          // 1
    XM::method<&XgComputer::GetResourceLayout>("GetResourceLayout"),      // 2
    {}, {}, {},                                                           // 3..5 unknown
    XM::method<&XgComputer::GetTexelElementOffsetBytes>("GetTexelElementOffsetBytes"),  // 6
    {},                                                                   // 7 unknown
    XM::method<&XgComputer::CopyIntoSubresource>("CopyIntoSubresource"),  // 8
    XM::method<&XgComputer::CopyFromSubresource>("CopyFromSubresource"),  // 9
    XM::method<&XgComputer::GetResourceTiling>("GetResourceTiling"),      // 10
};
constexpr auto kIXgComputer =
    com::declare_interface<XgComputer>("xg_xs.XGTextureAddressComputer", com::make_guid("00000000-0000-0000-0000-000000000000"),
                                       kXgMethods, {.slot_count = 16, .iunknown = false});


const com::Interface<GameInputDeviceObj>& GameInputReadingObj::kIDevice() {
    return guest_api::kIDevice;
}

HResult GameInputApi::emit(const Snapshot& s, OutPtr out) {
    com::InterfaceArg<GameInputDeviceObj> device;
    if (s.kind == DeviceKind::Gamepad) {
        device = gamepad_;
    } else if (s.kind == DeviceKind::Keyboard) {
        device = keyboard_;
    } else {
        device = ensure_mouse_device();
        if (!device) {
            return com::E_OUTOFMEMORY_HR;
        }
    }
    com::Ref<GameInputReadingObj> reading =
        com::create_object<GameInputReadingObj>(com::implements(kIReading), s, std::move(device));
    if (!reading) {
        return com::E_OUTOFMEMORY_HR;
    }
    out.write(reading.detach(kIReading));
    return kOk;
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------


struct InputModuleState {
    std::mutex mutex;
    com::Ref<GameInputApi> game_input;
    com::Ref<GameInputDeviceObj> gamepad;
    com::Ref<GameInputDeviceObj> keyboard;
    com::Ref<GameInputDeviceObj> mouse;
    std::atomic<bool> mouse_materialized{false};
};
InputModuleState& input_state() {
    static InputModuleState s;
    return s;
}

bool has_mouse_device() {
    return input_state().mouse_materialized.load(std::memory_order_acquire);
}

com::InterfaceArg<GameInputDeviceObj> ensure_mouse_device() {
    InputModuleState& st = input_state();
    std::lock_guard<std::mutex> lock(st.mutex);
    if (!st.mouse) {
        st.mouse = com::create_object<GameInputDeviceObj>(com::implements(kIDevice), DeviceKind::Mouse);
        if (!st.mouse) {
            return com::InterfaceArg<GameInputDeviceObj>{};
        }
        st.mouse->publish_info();
        st.mouse_materialized.store(true, std::memory_order_release);
    }
    return com::InterfaceArg<GameInputDeviceObj>::resolve(st.mouse->interface_pointer(kIDevice));
}

HResult game_input_create(OutPtr out) {
    if (out.is_null() || !out.write(0)) {
        return com::E_POINTER_HR;
    }
    InputModuleState& st = input_state();
    std::lock_guard<std::mutex> lock(st.mutex);
    if (!st.game_input) {
        Pg6IGameInput* host = nullptr;
        if (gdk::failed(pg6_xbox_GameInputCreate(&host)) || host == nullptr) {
            return com::E_FAIL_HR;
        }
        if (!st.gamepad) {
            st.gamepad = com::create_object<GameInputDeviceObj>(com::implements(kIDevice), DeviceKind::Gamepad);
            st.keyboard = com::create_object<GameInputDeviceObj>(com::implements(kIDevice), DeviceKind::Keyboard);
            if (!st.gamepad || !st.keyboard) {
                host->Release();
                return com::E_OUTOFMEMORY_HR;
            }
            st.gamepad->publish_info();
            st.keyboard->publish_info();
        }
        com::Ref<GameInputApi> api = com::create_object<GameInputApi>(
            com::implements(kIGameInput), host,
            com::InterfaceArg<GameInputDeviceObj>::resolve(st.gamepad->interface_pointer(kIDevice)),
            com::InterfaceArg<GameInputDeviceObj>::resolve(st.keyboard->interface_pointer(kIDevice)));
        if (!api) {
            return com::E_OUTOFMEMORY_HR;
        }
        st.game_input = std::move(api);
    }
    com::Ref<GameInputApi> ref = st.game_input;
    out.write(ref.detach(kIGameInput));
    return kOk;
}

HResult xg_create_texture_computer(uint64_t desc_va, OutPtr out) {
    if (out.is_null() || !out.write(0)) {
        return com::E_POINTER_HR;
    }
    XgResourceDesc desc{};
    if (desc_va == 0 || !com::read_guest(desc_va, &desc, kXgResourceDescBytes)) {
        return com::E_INVALIDARG_HR;
    }
    if (desc.width == 0 || desc.width > 0xFFFFFFFFull || desc.dimension < 2 || desc.dimension > 4) {
        diag("xg_xs: XGCreateTextureComputer with unsupported desc (dimension " + std::to_string(desc.dimension) +
             ", width " + std::to_string(desc.width) + ")");
        return com::E_INVALIDARG_HR;
    }
    Pg6XgTextureDesc hd{};
    hd.width = static_cast<uint32_t>(desc.width);
    hd.height = desc.dimension == 2 ? 1u : std::max<uint32_t>(desc.height, 1);
    hd.depth_or_array_size = std::max<uint16_t>(desc.depth_or_array_size, 1);
    hd.mip_levels = desc.mip_levels;
    hd.format = desc.format;
    hd.dimension = desc.dimension;
    hd.sample_count = std::max<uint32_t>(desc.sample_count, 1);
    hd.tile_mode = kXgTileModeLinear;
    Pg6IXGTextureAddressComputer* host = nullptr;
    const HResult hr = pg6_xbox_XGCreateTextureComputer(&hd, &host);
    if (gdk::failed(hr)) {
        return hr;
    }
    com::Ref<XgComputer> c = com::create_object<XgComputer>(com::implements(kIXgComputer), host, desc);
    if (!c) {
        return com::E_OUTOFMEMORY_HR;
    }
    out.write(c.detach(kIXgComputer));
    return kOk;
}

HResult xg_suggest_swizzle_mode(uint64_t desc, uint32_t /*flags*/, GuestPtr<uint32_t> mode) {
    if (mode.is_null()) {
        return com::E_POINTER_HR;
    }
    // The title builds the descriptor itself (dimension 2 / 3 / 4 = 1D / 2D / 3D; flags +0x30:
    // bit 0 render target, bit 1 depth-stencil, bit 2 UAV, the D3D12 resource flags) and calls this
    // only for per-texture mode 0xFF; what it reads is the shape, format, mips, samples and flags.
    XgResourceDesc d{};
    uint32_t picked = kXgDefaultSwizzleMode;
    if (desc != 0 && com::read_guest(desc, &d, kXgResourceDescBytes) && d.dimension >= 2 && d.dimension <= 4) {
        playgta6::gpu::gfx10::SuggestSwizzleDesc s;
        s.dimension = d.dimension - 1;
        s.dxgi_format = d.format;
        s.width = static_cast<uint32_t>(std::min<uint64_t>(d.width, 0xFFFFFFFFull));
        s.height = d.height;
        s.depth_or_array_size = d.depth_or_array_size;
        s.mip_levels = d.mip_levels;
        s.sample_count = d.sample_count;
        s.render_target = (d.flags & 1u) != 0;
        s.depth_stencil = (d.flags & 2u) != 0;
        s.unordered_access = (d.flags & 4u) != 0;
        picked = playgta6::gpu::gfx10::suggest_swizzle_mode(s);
    }
    return mode.write(picked) ? kOk : com::E_POINTER_HR;
}

}  // namespace

GameInputRumbleParams virtual_gamepad_rumble() {
    std::lock_guard<std::mutex> lock(rumble_mutex());
    return rumble_state();
}

void detail::reset_input_guest() {
    {
        InputModuleState& st = input_state();
        std::lock_guard<std::mutex> lock(st.mutex);
        st.game_input.reset();
        st.gamepad.reset();
        st.keyboard.reset();
        st.mouse.reset();
        st.mouse_materialized.store(false, std::memory_order_release);
    }
    std::lock_guard<std::mutex> lock(rumble_mutex());
    rumble_state() = GameInputRumbleParams{};
    rumble_telemetry_state() = os::RumbleTelemetry{};
}

}  // namespace playgta6::os::scarlett::guest_api

namespace playgta6::os {

RumbleTelemetry game_input_rumble_telemetry() {
    std::lock_guard<std::mutex> lock(scarlett::guest_api::rumble_mutex());
    return scarlett::guest_api::rumble_telemetry_state();
}

void reset_game_input_rumble_telemetry() {
    std::lock_guard<std::mutex> lock(scarlett::guest_api::rumble_mutex());
    scarlett::guest_api::rumble_telemetry_state() = RumbleTelemetry{};
}

bool game_input_guest_temporal_reading(uint64_t game_input_va,
                                       uint64_t timestamp_us,
                                       uint64_t device_va,
                                       uint64_t* out_reading_va,
                                       int32_t* out_hr) {
    namespace com = playgta6::guest::com;
    using Api = scarlett::guest_api::GameInputApi;
    using Dev = scarlett::guest_api::GameInputDeviceObj;

    if (out_reading_va == nullptr) {
        if (out_hr != nullptr) {
            *out_hr = com::E_POINTER_HR;
        }
        return false;
    }
    *out_reading_va = 0;
    auto api = com::InterfaceArg<Api>::resolve(game_input_va);
    if (!api) {
        if (out_hr != nullptr) {
            *out_hr = com::E_INVALIDARG_HR;
        }
        return false;
    }
    auto dev = com::InterfaceArg<Dev>::resolve(device_va);
    const uint64_t out_slot = com::guest_alloc(sizeof(uint64_t));
    if (out_slot == 0) {
        if (out_hr != nullptr) {
            *out_hr = com::E_OUTOFMEMORY_HR;
        }
        return false;
    }
    const com::HResult hr = api->GetTemporalReading(timestamp_us, std::move(dev), com::OutPtr(out_slot));
    uint64_t reading_va = 0;
    com::read_guest(out_slot, reading_va);
    com::guest_free(out_slot);
    if (out_reading_va != nullptr) {
        *out_reading_va = reading_va;
    }
    if (out_hr != nullptr) {
        *out_hr = hr;
    }
    return gdk::succeeded(hr);
}

}  // namespace playgta6::os

extern "C" {

void pg6_gthunk_game_input_create(void* state) {
    playgta6::guest::com::function_thunk<&playgta6::os::scarlett::guest_api::game_input_create>()(state);
}

void pg6_gthunk_xg_create_texture_computer(void* state) {
    playgta6::guest::com::function_thunk<&playgta6::os::scarlett::guest_api::xg_create_texture_computer>()(state);
}

void pg6_gthunk_xg_suggest_swizzle_mode(void* state) {
    playgta6::guest::com::function_thunk<&playgta6::os::scarlett::guest_api::xg_suggest_swizzle_mode>()(state);
}

}  // extern "C"
