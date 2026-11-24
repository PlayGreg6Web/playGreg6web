// AcpHal.dll!AcpHalCreate: the audio control processor HAL object, guest-callable.
//
// The executable keeps the object in one global and calls it from its audio setup and shutdown
// only. Its vtable has no IUnknown prefix (slot 0 takes two integers). Slots, from the call
// shapes and the code around them:
//   0  (u32 0x1000, u32 0x1000) -> HRESULT, first call after creation, failure aborts audio
//      setup: connect, with command and message queue lengths.
//   1  () at shutdown, before the title wakes and waits for its own audio threads: disconnect.
//   2  (u32 command, u32 tag, u32 0xFFFFFFFF, const u32* payload, u32 0) -> HRESULT: submit one
//      command (setup sends command 2 with tag 0xF0F0F0F0 and payload 0xC0, the voice path
//      command 5 with tag 0xF6F6F6F6 and a voice index). Only the HRESULT is read back.
//   5  () at shutdown, after which the title clears its pointer: release.
//   3, 4 and the rest: never called by the executable; they trap.
//
// Model: there is no audio hardware. Commands are accepted while connected and discarded
// (counted per command id, the first of each logged); nothing is written back, because the
// title reads nothing back from these calls.

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>

#include "os/xbox_scarlett_guest.h"

namespace playgta6::os::scarlett::guest_api {

using com::GuestPtr;
using com::OutPtr;

namespace {

constexpr HResult kOk = com::S_OK_HR;
constexpr HResult kEUnexpected = static_cast<HResult>(0x8000FFFFu);

void diag(const std::string& message) {
    com::detail::diagnostic(message.c_str());
}

class AcpHalObj : public com::ComObject {
public:
    static constexpr const char* kComClassName = "AcpHal.IAcpHal";

    // 0
    HResult Connect(uint32_t command_queue_length, uint32_t message_queue_length) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (command_queue_length == 0 || message_queue_length == 0) {
            diag("AcpHal: connect with an empty queue length");
            return com::E_INVALIDARG_HR;
        }
        connected_ = true;
        command_queue_length_ = command_queue_length;
        message_queue_length_ = message_queue_length;
        return kOk;
    }

    // 1
    HResult Disconnect() {
        std::lock_guard<std::mutex> lock(mutex_);
        connected_ = false;
        return kOk;
    }

    // 2
    HResult SubmitCommand(uint32_t command, uint32_t tag, uint32_t arg, uint64_t payload, uint32_t flags) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!connected_) {
            diag("AcpHal: command " + std::to_string(command) + " submitted while not connected");
            return kEUnexpected;
        }
        if (++commands_[command] == 1) {
            char line[200];
            std::snprintf(line, sizeof(line),
                          "AcpHal: first command %u (tag 0x%08x, arg 0x%08x, payload 0x%" PRIx64
                          ", flags %u): accepted and discarded (no audio hardware)",
                          command, tag, arg, payload, flags);
            diag(line);
        }
        return kOk;
    }

    // 5
    uint32_t Release() { return release(); }

    bool connected() {
        std::lock_guard<std::mutex> lock(mutex_);
        return connected_;
    }
    uint64_t commands(uint32_t command) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = commands_.find(command);
        return it != commands_.end() ? it->second : 0;
    }

private:
    std::mutex mutex_;
    bool connected_ = false;
    uint32_t command_queue_length_ = 0;
    uint32_t message_queue_length_ = 0;
    std::map<uint32_t, uint64_t> commands_;
};

using AM = com::Methods<AcpHalObj>;
constexpr com::MethodDesc kAcpHalMethods[] = {
    AM::method<&AcpHalObj::Connect>("Connect"),              // 0
    AM::method<&AcpHalObj::Disconnect>("Disconnect"),        // 1
    AM::method<&AcpHalObj::SubmitCommand>("SubmitCommand"),  // 2
    {}, {},                                                  // 3, 4: never called
    AM::method<&AcpHalObj::Release>("Release"),              // 5
};
constexpr auto kIAcpHal = com::declare_interface<AcpHalObj>(
    "AcpHal.IAcpHal", com::make_guid("00000000-0000-0000-0000-000000000000"), kAcpHalMethods,
    {.slot_count = 16, .iunknown = false});

// AcpHalCreate(IAcpHal** hal) -> HRESULT
HResult acp_hal_create(OutPtr out) {
    if (out.is_null() || !out.write(0)) {
        return com::E_POINTER_HR;
    }
    com::Ref<AcpHalObj> hal = com::create_object<AcpHalObj>(com::implements(kIAcpHal));
    if (!hal) {
        return com::E_OUTOFMEMORY_HR;
    }
    out.write(hal.detach(kIAcpHal));
    return kOk;
}

// ---------------------------------------------------------------------------
// chad.dll: XDsp (audio DSP streams). The host shims' two- and three-argument forms do not match
// the executable, and their host-pointer handles cannot travel through guest code. Call shapes:
//   XDspConnectWithMaximumStreamLimit(void* memory, u32 size, u32 max_streams, u32 0x80, u32 1,
//       XDspHandle* out) -> HRESULT, success only when exactly 0; one per DSP block element, with
//       that element's work memory.
//   XDspActivate(handle, const params*, void* out_a, void* out_b) -> HRESULT, stored by the
//       caller; out_b is later passed to XDspDeactivate (only when the HRESULT is >= 0); on
//       failure the caller takes its cleanup path.
//   XDspDeactivate(stream) / XDspDisconnect(handle) (tail calls from title wrappers).
//   XDspSubmitCommand(handle, const void* command) -> HRESULT, success only when exactly 0.
// Model: no DSP. Connect hands out an opaque handle (a zeroed 64-byte guest block, never read
// by the runtime); activation fails with E_NOTIMPL and clears both outputs, because what the
// DSP writes into out_a is not established; commands are accepted and discarded.
// ---------------------------------------------------------------------------

struct XDspState {
    std::mutex mutex;
    std::map<uint64_t, uint64_t> handles;  // handle -> commands submitted
    bool activate_logged = false;
};
XDspState& xdsp() {
    static XDspState s;
    return s;
}

HResult xdsp_connect(uint64_t memory, uint32_t size, uint32_t max_streams, uint32_t /*a4*/, uint32_t /*a5*/,
                     OutPtr out) {
    if (out.is_null() || !out.write(0)) {
        return com::E_POINTER_HR;
    }
    if (memory == 0 || size == 0 || max_streams == 0) {
        diag("XDspConnectWithMaximumStreamLimit: invalid memory, size or stream limit");
        return com::E_INVALIDARG_HR;
    }
    const uint64_t handle = com::guest_alloc(64);
    if (handle == 0) {
        return com::E_OUTOFMEMORY_HR;
    }
    {
        std::lock_guard<std::mutex> lock(xdsp().mutex);
        xdsp().handles[handle] = 0;
    }
    out.write(handle);
    return kOk;
}

bool xdsp_known(uint64_t handle) {
    std::lock_guard<std::mutex> lock(xdsp().mutex);
    return xdsp().handles.count(handle) != 0;
}

HResult xdsp_activate(uint64_t handle, uint64_t /*params*/, OutPtr out_a, OutPtr out_b) {
    if (!out_a.is_null()) {
        out_a.write(0);
    }
    if (!out_b.is_null()) {
        out_b.write(0);
    }
    if (!xdsp_known(handle)) {
        return com::E_INVALIDARG_HR;
    }
    {
        std::lock_guard<std::mutex> lock(xdsp().mutex);
        if (!xdsp().activate_logged) {
            xdsp().activate_logged = true;
            diag("XDspActivate: no audio DSP; stream activation fails with E_NOTIMPL");
        }
    }
    return com::E_NOTIMPL_HR;
}

HResult xdsp_deactivate(uint64_t stream) {
    // Only reached for an activation that succeeded, which this model never reports.
    return stream != 0 ? com::E_INVALIDARG_HR : kOk;
}

HResult xdsp_disconnect(uint64_t handle) {
    {
        std::lock_guard<std::mutex> lock(xdsp().mutex);
        if (xdsp().handles.erase(handle) == 0) {
            return com::E_INVALIDARG_HR;
        }
    }
    com::guest_free(handle);
    return kOk;
}

HResult xdsp_submit_command(uint64_t handle, uint64_t command) {
    if (command == 0) {
        return com::E_INVALIDARG_HR;
    }
    std::lock_guard<std::mutex> lock(xdsp().mutex);
    auto it = xdsp().handles.find(handle);
    if (it == xdsp().handles.end()) {
        return com::E_INVALIDARG_HR;
    }
    ++it->second;
    return kOk;
}

// ---------------------------------------------------------------------------
// xapu.dll: the audio processing unit's command / result queues. Call shapes:
//   XApuConnect(const params*, p2*, XApuHandle* out) -> HRESULT (>= 0 is success)
//   XApuEnqueueCommand(handle, const command*, u32 0) -> HRESULT (ignored by every caller);
//       a command starts {u32 type, u32 voice index}; types 1, 2 and 5 are sent.
//   XApuGetResultCount(handle, u32 0) -> u32 count (returned directly, not an HRESULT)
//   XApuDequeueResult(handle, result* (0x38 bytes)) -> HRESULT, called count times
//   XApuDisconnect(handle) at shutdown
// The title's result handler (0x144EAA840) acknowledges its own commands: a result of type 1
// moves a voice from state 1 to 4, type 5 from state 2 (set when command 5 is sent) to 4, type
// 2 frees the voice (command 2 sets state 3); type 3 is a notification of its own.
// Model: no processing hardware, but every command of types 1, 2 and 5 is acknowledged with a
// result {type, voice index} (the rest zero) in submission order, so voices move through their
// states and are freed; other command types are accepted without a result (counted, the first
// logged). No type-3 notifications are produced.
// ---------------------------------------------------------------------------

constexpr size_t kXApuResultBytes = 0x38;

struct XApuState {
    std::mutex mutex;
    std::map<uint64_t, std::deque<std::array<uint32_t, 2>>> queues;  // handle -> pending results
    std::map<uint32_t, uint64_t> commands;                            // command type -> count
    std::map<uint64_t, uint64_t> memory;                              // handle -> shared memory
};
XApuState& xapu() {
    static XApuState s;
    return s;
}

// The memory XApuConnect hands back through its second argument (a 16-byte struct whose first
// qword the title reads as a base address): voice slot s uses base + s * 0x280 for the data it
// copies in (0x144F20AD0) and base + 0x14000 + s * 0x1E00 as its output area (0x144F20B64,
// 0x144E64B45); the voice free list ends at slot 0x80. So 128 * 0x280 = 0x14000 bytes of input
// slots followed by 128 * 0x1E00 bytes of output slots.
constexpr uint64_t kXApuVoiceSlots = 0x80;
constexpr uint64_t kXApuSharedBytes = kXApuVoiceSlots * 0x280 + kXApuVoiceSlots * 0x1E00;

uint64_t alloc_zeroed(uint64_t bytes) {
    const uint64_t va = com::guest_alloc(bytes);
    if (va != 0) {
        static const uint8_t zeros[4096] = {};
        for (uint64_t off = 0; off < bytes; off += sizeof(zeros)) {
            com::write_guest(va + off, zeros, static_cast<size_t>(std::min<uint64_t>(sizeof(zeros), bytes - off)));
        }
    }
    return va;
}

HResult xapu_connect(uint64_t params, GuestPtr<uint64_t> shared, OutPtr out) {
    if (out.is_null() || !out.write(0)) {
        return com::E_POINTER_HR;
    }
    if (params == 0 || shared.is_null()) {
        return com::E_INVALIDARG_HR;
    }
    const uint64_t handle = com::guest_alloc(64);
    const uint64_t memory = alloc_zeroed(kXApuSharedBytes);
    if (handle == 0 || memory == 0) {
        if (handle != 0) {
            com::guest_free(handle);
        }
        if (memory != 0) {
            com::guest_free(memory);
        }
        return com::E_OUTOFMEMORY_HR;
    }
    {
        std::lock_guard<std::mutex> lock(xapu().mutex);
        xapu().queues[handle];
        xapu().memory[handle] = memory;
    }
    shared.write(memory);
    out.write(handle);
    return kOk;
}

HResult xapu_disconnect(uint64_t handle) {
    uint64_t memory = 0;
    {
        std::lock_guard<std::mutex> lock(xapu().mutex);
        if (xapu().queues.erase(handle) == 0) {
            return com::E_INVALIDARG_HR;
        }
        memory = xapu().memory[handle];
        xapu().memory.erase(handle);
    }
    com::guest_free(handle);
    if (memory != 0) {
        com::guest_free(memory);
    }
    return kOk;
}

HResult xapu_enqueue_command(uint64_t handle, uint64_t command, uint32_t /*flags*/) {
    uint32_t head[2] = {};
    if (command == 0 || !com::read_guest(command, head, sizeof(head))) {
        return com::E_INVALIDARG_HR;
    }
    std::lock_guard<std::mutex> lock(xapu().mutex);
    auto it = xapu().queues.find(handle);
    if (it == xapu().queues.end()) {
        return com::E_INVALIDARG_HR;
    }
    const uint32_t type = head[0];
    const bool acknowledged = type == 1 || type == 2 || type == 5;
    if (++xapu().commands[type] == 1) {
        diag("XApu: first command of type " + std::to_string(type) +
             (acknowledged ? ": acknowledged with a result" : ": accepted without a result"));
    }
    if (acknowledged) {
        it->second.push_back({type, head[1]});
    }
    return kOk;
}

uint32_t xapu_get_result_count(uint64_t handle, uint32_t /*flags*/) {
    std::lock_guard<std::mutex> lock(xapu().mutex);
    auto it = xapu().queues.find(handle);
    return it != xapu().queues.end() ? static_cast<uint32_t>(it->second.size()) : 0u;
}

HResult xapu_dequeue_result(uint64_t handle, uint64_t result) {
    if (result == 0) {
        return com::E_POINTER_HR;
    }
    std::array<uint32_t, 2> head{};
    {
        std::lock_guard<std::mutex> lock(xapu().mutex);
        auto it = xapu().queues.find(handle);
        if (it == xapu().queues.end()) {
            return com::E_INVALIDARG_HR;
        }
        if (it->second.empty()) {
            return com::E_FAIL_HR;
        }
        head = it->second.front();
        it->second.pop_front();
    }
    uint8_t bytes[kXApuResultBytes] = {};
    std::memcpy(bytes, head.data(), sizeof(uint32_t) * 2);
    return com::write_guest(result, bytes, sizeof(bytes)) ? kOk : com::E_POINTER_HR;
}

}  // namespace

XApuStats xapu_stats() {
    XApuStats s;
    std::lock_guard<std::mutex> lock(xapu().mutex);
    s.connections = xapu().queues.size();
    for (const auto& [type, count] : xapu().commands) {
        s.commands += count;
    }
    for (const auto& [handle, queue] : xapu().queues) {
        s.pending_results += queue.size();
    }
    return s;
}

XDspStats xdsp_stats() {
    XDspStats s;
    std::lock_guard<std::mutex> lock(xdsp().mutex);
    s.connections = xdsp().handles.size();
    for (const auto& [handle, commands] : xdsp().handles) {
        s.commands += commands;
    }
    return s;
}

AcpHalState acp_hal_state(uint64_t hal) {
    AcpHalState s;
    auto arg = com::InterfaceArg<AcpHalObj>::resolve(hal);
    if (arg) {
        auto* obj = static_cast<AcpHalObj*>(arg.get());
        s.valid = true;
        s.connected = obj->connected();
        s.commands_2 = obj->commands(2);
        s.commands_5 = obj->commands(5);
    }
    return s;
}

}  // namespace playgta6::os::scarlett::guest_api

extern "C" {

void pg6_gthunk_acp_hal_create(void* state) {
    playgta6::guest::com::function_thunk<&playgta6::os::scarlett::guest_api::acp_hal_create>()(state);
}
void pg6_gthunk_xdsp_connect_with_maximum_stream_limit(void* state) {
    playgta6::guest::com::function_thunk<&playgta6::os::scarlett::guest_api::xdsp_connect>()(state);
}
void pg6_gthunk_xdsp_activate(void* state) {
    playgta6::guest::com::function_thunk<&playgta6::os::scarlett::guest_api::xdsp_activate>()(state);
}
void pg6_gthunk_xdsp_deactivate(void* state) {
    playgta6::guest::com::function_thunk<&playgta6::os::scarlett::guest_api::xdsp_deactivate>()(state);
}
void pg6_gthunk_xdsp_disconnect(void* state) {
    playgta6::guest::com::function_thunk<&playgta6::os::scarlett::guest_api::xdsp_disconnect>()(state);
}
void pg6_gthunk_xdsp_submit_command(void* state) {
    playgta6::guest::com::function_thunk<&playgta6::os::scarlett::guest_api::xdsp_submit_command>()(state);
}

void pg6_gthunk_xapu_connect(void* state) {
    playgta6::guest::com::function_thunk<&playgta6::os::scarlett::guest_api::xapu_connect>()(state);
}
void pg6_gthunk_xapu_disconnect(void* state) {
    playgta6::guest::com::function_thunk<&playgta6::os::scarlett::guest_api::xapu_disconnect>()(state);
}
void pg6_gthunk_xapu_enqueue_command(void* state) {
    playgta6::guest::com::function_thunk<&playgta6::os::scarlett::guest_api::xapu_enqueue_command>()(state);
}
void pg6_gthunk_xapu_get_result_count(void* state) {
    playgta6::guest::com::function_thunk<&playgta6::os::scarlett::guest_api::xapu_get_result_count>()(state);
}
void pg6_gthunk_xapu_dequeue_result(void* state) {
    playgta6::guest::com::function_thunk<&playgta6::os::scarlett::guest_api::xapu_dequeue_result>()(state);
}

}  // extern "C"
