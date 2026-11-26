// GameInput.dll and xg_xs.dll guest interfaces (O23); evidence and layouts in xbox_scarlett_guest.h.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
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
constexpr size_t kHistory = 64;

enum class DeviceKind : uint8_t { Gamepad = 0, Keyboard = 1 };

uint32_t input_kind_of(DeviceKind kind) {
    return kind == DeviceKind::Gamepad ? kGameInputKindGamepad : kGameInputKindKeyboard;
}

struct Snapshot {
    DeviceKind kind = DeviceKind::Gamepad;
    uint64_t sequence = 0;   // GameInput sequence number (one counter for all readings)
    uint64_t timestamp = 0;  // microseconds, monotonic
    uint32_t host_sequence = 0;
    GameInputGamepadState gamepad{};
    std::vector<GameInputKeyState> keys;
};

// Keyboard keys the InputState SAB carries (web/input/input_bridge.js bits 0..5).
struct KeyMap {
    uint32_t bit;
    uint32_t scan_code;
    uint32_t code_point;
    uint8_t virtual_key;
};
constexpr KeyMap kKeyMap[] = {
    {os::input_button::kMoveForward, 0x11, 'w', 0x57},   // W
    {os::input_button::kMoveLeft, 0x1E, 'a', 0x41},      // A
    {os::input_button::kMoveBackward, 0x1F, 's', 0x53},  // S
    {os::input_button::kMoveRight, 0x20, 'd', 0x44},     // D
    {os::input_button::kJump, 0x39, ' ', 0x20},          // Space
    {os::input_button::kSprint, 0x2A, 0, 0xA0},          // Left Shift
};

GameInputRumbleParams& rumble_state() {
    static GameInputRumbleParams r{};
    return r;
}
std::mutex& rumble_mutex() {
    static std::mutex m;
    return m;
}


// ---------------------------------------------------------------------------

class GameInputDeviceObj final : public com::ComObject {
public:
    static constexpr const char* kComClassName = "GameInput.IGameInputDevice";
    static constexpr uint64_t kComPayloadSize = kGameInputDeviceInfoSize;

    explicit GameInputDeviceObj(DeviceKind kind) : kind_(kind) {
        const char* tag = kind == DeviceKind::Gamepad ? "pg6 virtual gamepad" : "pg6 virtual keyboard";
        std::memset(id_.value, 0, sizeof(id_.value));
        std::memcpy(id_.value, tag, std::strlen(tag));
    }

    // Fills the guest GameInputDeviceInfo (after the object has its guest block).
    void publish_info() {
        uint8_t info[kGameInputDeviceInfoSize]{};
        const uint32_t size = static_cast<uint32_t>(kGameInputDeviceInfoSize);
        std::memcpy(info, &size, 4);
        const uint16_t vendor = 0x045E;  // Microsoft
        const uint16_t product = kind_ == DeviceKind::Gamepad ? 0x0B12 : 0x0000;
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
    // slot 10
    void SetRumbleState(GuestPtr<GameInputRumbleParams> params) {
        GameInputRumbleParams p{};
        if (!params.is_null() && !params.read(p)) {
            return;
        }
        if (kind_ == DeviceKind::Gamepad) {
            std::lock_guard<std::mutex> lock(rumble_mutex());
            rumble_state() = p;
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

    GameInputReadingObj(Snapshot snapshot, com::InterfaceArg<GameInputDeviceObj> device)
        : snap_(std::move(snapshot)), device_(std::move(device)) {}

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
    uint32_t GetKeyCount() { return static_cast<uint32_t>(snap_.keys.size()); }   // 14
    uint32_t GetKeyState(uint32_t count, uint64_t out) {                           // 15
        const uint32_t n = std::min<uint32_t>(count, static_cast<uint32_t>(snap_.keys.size()));
        if (n != 0 && !com::write_guest(out, snap_.keys.data(), n * sizeof(GameInputKeyState))) {
            return 0;
        }
        return n;
    }
    bool GetNoState(uint64_t /*state*/) { return false; }                          // 16, 19..21, 23, 24
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

    uint64_t GetCurrentTimestamp() { return now_us(); }  // 3

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
        {
            std::lock_guard<std::mutex> lock(mutex_);
            poll_locked(dk);
            s = history_[index(dk)].back();
        }
        return emit(std::move(s), out);
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
        for (const com::InterfaceArg<GameInputDeviceObj>* dp : {&gamepad_, &keyboard_}) {
            const com::InterfaceArg<GameInputDeviceObj>& d = *dp;
            if ((kind & input_kind_of(d->kind())) == 0 || (device && device.get() != d.get())) {
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
        return kGameInputDeviceNotFound;
    }

    // 21
    void SetFocusPolicy(uint32_t /*policy*/) {}

    static const com::Interface<GameInputDeviceObj>& kIDeviceRef() { return GameInputReadingObj::kIDevice(); }

private:
    static size_t index(DeviceKind k) { return static_cast<size_t>(k); }

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
        return kGameInputReadingNotFound;
    }

    // Reads the host InputState (through the O12 host GameInput, which applies the button map,
    // deadzones and WASD-to-stick synthesis) and records a new snapshot when it changed.
    void poll_locked(DeviceKind kind) {
        GameInputGamepadState pad{};
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
            raw = r->GetRawButtons();
            host_seq = static_cast<uint32_t>(r->GetSequenceNumber(scarlett::kGameInputKindGamepad));
            r->Release();
        }
        std::deque<Snapshot>& h = history_[index(kind)];
        if (!h.empty() && h.back().host_sequence == host_seq) {
            return;
        }
        Snapshot s;
        s.kind = kind;
        s.sequence = ++sequence_;
        s.timestamp = now_us();
        s.host_sequence = host_seq;
        if (kind == DeviceKind::Gamepad) {
            s.gamepad = pad;
        } else {
            for (const KeyMap& k : kKeyMap) {
                if ((raw & k.bit) != 0) {
                    s.keys.push_back(GameInputKeyState{k.scan_code, k.code_point, k.virtual_key, 0, {0, 0}});
                }
            }
        }
        h.push_back(std::move(s));
        while (h.size() > kHistory) {
            h.pop_front();
        }
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
        Snapshot found;
        bool have = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (next) {
                poll_locked(dk);
            }
            const std::deque<Snapshot>& h = history_[index(dk)];
            if (next) {
                for (const Snapshot& s : h) {
                    if (s.sequence > at) {
                        found = s;
                        have = true;
                        break;
                    }
                }
            } else {
                for (auto it = h.rbegin(); it != h.rend(); ++it) {
                    if (it->sequence < at) {
                        found = *it;
                        have = true;
                        break;
                    }
                }
            }
        }
        if (!have) {
            return kGameInputReadingNotFound;
        }
        return emit(std::move(found), out);
    }

    HResult emit(Snapshot s, OutPtr out);

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
    std::deque<Snapshot> history_[2];
    uint64_t sequence_ = 0;
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

    // 6 GetTexelElementOffsetBytes(plane, mipLevel, x, y, zOrSlice, sample, bitOffset*) -> UINT64:
    // byte offset from the resource base of element (x, y) of the mip in slice zOrSlice. The title
    // passes element coordinates (for BC formats it divides texel coordinates by 4 itself). Tiled
    // and console linear layouts use the GFX10 address the detiler and CopyIntoSubresource use;
    // other layouts the host computer's linear mips. Elements are whole bytes: *bitOffset = 0.
    // UINT64_MAX for an invalid plane, mip or coordinate.
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
        // Every verified GFX10 swizzle mode (256 B / 4 KiB / 64 KiB), not only 64 KiB: a tile is
        // one swizzle block of that mode, not the D3D12 64 KiB standard tile.
        const bool gfx10_tiled = tiled_ && tiled_desc_.d3d12_layout != gfx10::kConsoleLinearLayout;
        // GFX10 64KB modes take the standard/packed split and tile numbers from the shared
        // gfx10::tile_map (addrlib mip tail rule, not the D3D12 packed-mip rule: a 2:1 mip such as
        // 64x32 BC7 blocks keeps a block of its own and the tail follows it; the title streams each
        // standard mip and the tail as separate reads matching the archive's per-block chunks).
        gfx10::TileMap tm{};
        if (gfx10_tiled) {
            tm = gfx10::tile_map(tiled_desc_);
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
            standard < mips ? std::max<uint32_t>(1, ceil_div(packed_bytes, tm.ok ? tm.block_bytes : 65536)) : 0;
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
    GM::method<&GameInputApi::RegisterDeviceCallback>("RegisterDeviceCallback"),  // 9
    GM::stub("RegisterGuideButtonCallback"),                                      // 10
    GM::stub("RegisterKeyboardLayoutCallback"),                                   // 11
    GM::method<&GameInputApi::StopCallback>("StopCallback"),                      // 12
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
    RM::method<&GameInputReadingObj::GetNoState>("GetMouseState"),                    // 16
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
    DM::method<&GameInputDeviceObj::GetDeviceInfo>("GetDeviceInfo"),      // 3
    DM::method<&GameInputDeviceObj::GetDeviceStatus>("GetDeviceStatus"),  // 4
    DM::stub("GetBatteryState"),                                          // 5
    DM::stub("CreateForceFeedbackEffect"),                                // 6
    DM::stub("IsForceFeedbackMotorPoweredOn"),                            // 7
    DM::stub("SetForceFeedbackMotorGain"),                                // 8
    DM::stub("SetHapticMotorState"),                                      // 9
    DM::method<&GameInputDeviceObj::SetRumbleState>("SetRumbleState"),    // 10
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
    {},                                                                   // 9 unknown
    XM::method<&XgComputer::GetResourceTiling>("GetResourceTiling"),      // 10
};
constexpr auto kIXgComputer =
    com::declare_interface<XgComputer>("xg_xs.XGTextureAddressComputer", com::make_guid("00000000-0000-0000-0000-000000000000"),
                                       kXgMethods, {.slot_count = 16, .iunknown = false});


const com::Interface<GameInputDeviceObj>& GameInputReadingObj::kIDevice() {
    return guest_api::kIDevice;
}

HResult GameInputApi::emit(Snapshot s, OutPtr out) {
    com::InterfaceArg<GameInputDeviceObj> device = s.kind == DeviceKind::Gamepad ? gamepad_ : keyboard_;
    com::Ref<GameInputReadingObj> reading =
        com::create_object<GameInputReadingObj>(com::implements(kIReading), std::move(s), std::move(device));
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
};
InputModuleState& input_state() {
    static InputModuleState s;
    return s;
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
    }
    std::lock_guard<std::mutex> lock(rumble_mutex());
    rumble_state() = GameInputRumbleParams{};
}

}  // namespace playgta6::os::scarlett::guest_api

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
