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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "os/xapu_opus.h"
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
//       a command starts {u32 type, u32 voice index}; types 1, 2, 3 and 5 are sent.
//   XApuGetResultCount(handle, u32 0) -> u32 count (returned directly, not an HRESULT)
//   XApuDequeueResult(handle, result* (0x38 bytes)) -> HRESULT, called count times
//   XApuDisconnect(handle) at shutdown
// The title's result handler (0x144EAA840) acknowledges its own commands: a result of type 1
// moves a voice from state 1 to 4, type 5 from state 2 (set when command 5 is sent) to 4, type
// 2 frees the voice (command 2 sets state 3); type 3 is a notification of its own: for a voice in
// state 4 it stores the result's u32 at +0x34 (bytes of float32 output written to the voice's
// output slot) and moves it to state 5, after which the stream decoder (0x144E64B10) converts
// that many bytes to int16 PCM. Without these results the title's decoded streams stay silent.
//
// Command 3 is a decode request, built by 0x144F20AD0 (0x40 bytes):
//   +0x00 u32 3            +0x04 u32 voice index     +0x08..+0x1B zero
//   +0x1C f32 sample rate  +0x20 u32 frames to drop from the start of this packet's output (the
//                            stream's pending trim, cleared by the title once sent)
//   +0x24 u32 960 (frames per packet)
//   +0x28 u32 packet bytes +0x2C u64 packet address (the voice's input slot, base + s * 0x280)
//   +0x34 u32 output capacity in bytes (channels * 0xF00)
//   +0x38 u64 output address (the voice's output slot, base + 0x14000 + s * 0x1E00)
// The packet is one Opus packet: the title walks its streams as {i16 length, payload} records
// (0x144E64480), accepts payloads of at most 0x280 bytes (0x144E64710) and copies a record minus
// its length prefix into the input slot; 960 float32 frames per channel per packet is Opus's
// 20 ms frame at 48 kHz. The channel count is capacity / 0xF00.
// Model: no processing hardware, but every command of types 1, 2 and 5 is acknowledged with a
// result {type, voice index} (the rest zero) in submission order, so voices move through their
// states and are freed. Command 3 is decoded synchronously (os/xapu_opus.h; per-voice decoder
// state, reset by commands 1 and 2), the float32 PCM written to the output address, and a result
// {3, voice index, zeros, +0x34 bytes written} queued (0 bytes when the packet does not decode).
// Other command types are accepted without a result (counted, the first logged).
//
// Codec census (R40, menu-flow run of 420 s): the executable sends XApu only types 1, 2 and 3
// (5 is wired but was not seen); every decode command carries one Opus packet (287 decoded, 0
// failed, 287 type-3 results). No other codec reaches a hardware model: AcpHalAllocateShapeContexts
// is called with every count but kind 4 zeroed (xbox_middleware.cpp), so no SHAPE/XMA decoder
// contexts exist, XDspActivate is never reached, and the executable imports no XAudio2/XMA API;
// PCM and ADPCM are decoded by the title's own code. The boot report's audio section carries the
// per-type counts and `xapu_unhandled_commands` so a new codec path shows up there first.
// ---------------------------------------------------------------------------

constexpr size_t kXApuResultBytes = 0x38;
constexpr size_t kXApuDecodeCommandBytes = 0x40;
constexpr uint32_t kXApuChannelBytes = 0xF00;     // 960 float32 frames of one channel
constexpr uint32_t kXApuMaxPacketBytes = 0x280;   // one input slot
constexpr uint32_t kXApuMaxOutputBytes = 0x1E00;  // one output slot

using XApuResult = std::array<uint8_t, kXApuResultBytes>;

struct XApuState {
    std::mutex mutex;
    std::map<uint64_t, std::deque<XApuResult>> queues;  // handle -> pending results
    std::map<uint32_t, uint64_t> commands;              // command type -> count
    std::map<uint64_t, uint64_t> memory;                // handle -> shared memory
    std::map<std::pair<uint64_t, uint32_t>, std::unique_ptr<xapu_opus::VoiceDecoder>> decoders;  // (handle, voice)
    uint64_t decoded_packets = 0;
    uint64_t failed_packets = 0;
    uint64_t decoded_bytes = 0;
    uint64_t type3_results = 0;   // type-3 results queued (one per decode command)
    uint64_t results_posted = 0;  // results of every type queued
    uint64_t unhandled = 0;       // commands of a type the model does not know (no result)
    bool decode_logged = false;
    bool failure_logged = false;
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
        auto& decoders = xapu().decoders;
        decoders.erase(decoders.lower_bound({handle, 0u}), decoders.upper_bound({handle, ~0u}));
    }
    com::guest_free(handle);
    if (memory != 0) {
        com::guest_free(memory);
    }
    return kOk;
}

XApuResult make_result(uint32_t type, uint32_t voice, uint32_t bytes_written = 0) {
    XApuResult r{};
    std::memcpy(r.data(), &type, sizeof(type));
    std::memcpy(r.data() + 0x04, &voice, sizeof(voice));
    std::memcpy(r.data() + 0x34, &bytes_written, sizeof(bytes_written));
    return r;
}

template <typename T>
T field(const uint8_t* bytes, size_t offset) {
    T v;
    std::memcpy(&v, bytes + offset, sizeof(v));
    return v;
}

// Command 3 (the caller holds the lock): decodes the packet into the output area and returns the
// result to queue.
XApuResult decode_command(uint64_t handle, const uint8_t* cmd) {
    const uint32_t voice = field<uint32_t>(cmd, 0x04);
    const float rate = field<float>(cmd, 0x1C);
    const uint32_t skip = field<uint32_t>(cmd, 0x20);
    const uint32_t frames_per_packet = field<uint32_t>(cmd, 0x24);
    const uint32_t packet_bytes = field<uint32_t>(cmd, 0x28);
    const uint64_t packet_va = field<uint64_t>(cmd, 0x2C);
    const uint32_t capacity = std::min(field<uint32_t>(cmd, 0x34), kXApuMaxOutputBytes);
    const uint64_t output_va = field<uint64_t>(cmd, 0x38);
    const uint32_t channels = capacity / kXApuChannelBytes;
    auto& s = xapu();

    uint8_t packet[kXApuMaxPacketBytes];
    const bool readable = packet_bytes != 0 && packet_bytes <= sizeof(packet) && packet_va != 0 &&
                          com::read_guest(packet_va, packet, packet_bytes);
    if (readable && channels != 0 && output_va != 0) {
        auto& decoder = s.decoders[{handle, voice}];
        if (!decoder) {
            decoder = std::make_unique<xapu_opus::VoiceDecoder>();
        }
        const size_t max_frames = capacity / (channels * sizeof(float));
        std::vector<float> pcm(max_frames * channels);
        const uint32_t sample_rate = rate >= 1.0f && rate < 1.0e6f ? static_cast<uint32_t>(rate) : 48000u;
        const int frames = decoder->decode(packet, packet_bytes, sample_rate, channels, pcm.data(), max_frames);
        if (frames >= 0) {
            const uint32_t dropped = std::min<uint32_t>(skip, static_cast<uint32_t>(frames));
            const uint32_t bytes = (static_cast<uint32_t>(frames) - dropped) * channels * sizeof(float);
            if (bytes == 0 || com::write_guest(output_va, pcm.data() + size_t{dropped} * channels, bytes)) {
                ++s.decoded_packets;
                s.decoded_bytes += bytes;
                if (!s.decode_logged) {
                    s.decode_logged = true;
                    char line[256];
                    std::snprintf(line, sizeof(line),
                                  "XApu: first decode: voice %u, %u bytes (TOC 0x%02x), %u Hz, %u channel(s), "
                                  "%u frames/packet, skip %u -> %u bytes of float32%s",
                                  voice, packet_bytes, packet[0], sample_rate, channels, frames_per_packet, skip,
                                  bytes, xapu_opus::available() ? "" : " (silence: built without libopus)");
                    diag(line);
                }
                return make_result(3, voice, bytes);
            }
        }
    }
    ++s.failed_packets;
    if (!s.failure_logged) {
        s.failure_logged = true;
        char line[256];
        std::snprintf(line, sizeof(line),
                      "XApu: decode failed: voice %u, packet 0x%" PRIx64 " (%u bytes), capacity %u, output 0x%" PRIx64
                      "; result posted with 0 bytes",
                      voice, packet_va, packet_bytes, capacity, output_va);
        diag(line);
    }
    return make_result(3, voice, 0);
}

HResult xapu_enqueue_command(uint64_t handle, uint64_t command, uint32_t /*flags*/) {
    uint32_t head[2] = {};
    if (command == 0 || !com::read_guest(command, head, sizeof(head))) {
        return com::E_INVALIDARG_HR;
    }
    const uint32_t type = head[0];
    uint8_t decode[kXApuDecodeCommandBytes] = {};
    if (type == 3 && !com::read_guest(command, decode, sizeof(decode))) {
        return com::E_INVALIDARG_HR;
    }
    std::lock_guard<std::mutex> lock(xapu().mutex);
    auto it = xapu().queues.find(handle);
    if (it == xapu().queues.end()) {
        return com::E_INVALIDARG_HR;
    }
    const bool acknowledged = type == 1 || type == 2 || type == 3 || type == 5;
    if (++xapu().commands[type] == 1) {
        diag("XApu: first command of type " + std::to_string(type) +
             (acknowledged ? ": acknowledged with a result" : ": accepted without a result"));
    }
    if (type == 1 || type == 2) {
        // The voice starts or is freed: its next packet begins a new stream.
        auto d = xapu().decoders.find({handle, head[1]});
        if (d != xapu().decoders.end()) {
            d->second->reset();
        }
    }
    if (type == 3) {
        it->second.push_back(decode_command(handle, decode));
        ++xapu().type3_results;
    } else if (acknowledged) {
        it->second.push_back(make_result(type, head[1]));
    } else {
        ++xapu().unhandled;
    }
    if (acknowledged) {
        ++xapu().results_posted;
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
    XApuResult bytes{};
    {
        std::lock_guard<std::mutex> lock(xapu().mutex);
        auto it = xapu().queues.find(handle);
        if (it == xapu().queues.end()) {
            return com::E_INVALIDARG_HR;
        }
        if (it->second.empty()) {
            return com::E_FAIL_HR;
        }
        bytes = it->second.front();
        it->second.pop_front();
    }
    return com::write_guest(result, bytes.data(), bytes.size()) ? kOk : com::E_POINTER_HR;
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
    s.decoded_packets = xapu().decoded_packets;
    s.failed_packets = xapu().failed_packets;
    s.decoded_bytes = xapu().decoded_bytes;
    auto count = [](uint32_t type) {
        const auto it = xapu().commands.find(type);
        return it != xapu().commands.end() ? it->second : uint64_t{0};
    };
    s.type1_commands = count(1);
    s.type2_commands = count(2);
    s.type3_commands = count(3);
    s.type5_commands = count(5);
    s.unhandled_commands = xapu().unhandled;
    s.type3_results = xapu().type3_results;
    s.results_posted = xapu().results_posted;
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
