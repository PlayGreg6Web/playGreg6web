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
// Model: there is no audio hardware. Commands are accepted while connected (counted per command
// id, the first of each logged). The XMA context commands (behaviour of the SHAPE XMA block,
// reimplemented here) keep the per-context state the title can observe:
//   5 / 7   enable / disable one XMA context; the payload's first u32 is the context index.
//   6 / 8   enable / disable many; the payload is a bitmask of u32 words, bit i = context i,
//           as many words as the allocated contexts need.
//   18      (0x12, tag 0xF1F1F1F1, sent by the stream consumer 0x140078120) the title took
//           `blocks` 256-byte output blocks of context `index` (payload {u32 index, u32 blocks}):
//           its output read offset (dword 9 bits 0-4) moves forward modulo the output buffer's
//           block count (dword 0 bits 22-26, 0 = 32).
// Taking blocks also re-arms the context's "output writable" bit (dword 1 bit 31). Enabled contexts
// are decoded by a worker thread into their output rings (os/xma_decoder.h): it walks the XMA2
// packets of both input buffers, moves the write block and clears "writable" when the ring is full,
// handles loops and subframe skips, and sets the "input ran out" error once both inputs are used
// up. There is no WMA Pro frame decoder yet, so every frame is written as 512 samples of silence:
// voices run at their real pace and end, but stay silent. Other commands are discarded; the title
// reads nothing back but the HRESULT.

#include <algorithm>
#include "os/thread_name.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "os/audio_stream_log.h"
#include "os/xapu_connect.h"
#include "os/xapu_opus.h"
#include "os/xbox_middleware.h"
#include "os/xbox_scarlett_guest.h"
#include "os/win32_kernel.h"
#include "os/xma_decoder.h"

namespace playgta6::os::scarlett::guest_api {

using com::GuestPtr;
using com::OutPtr;
namespace xd = playgta6::os::xma;

namespace {

constexpr HResult kOk = com::S_OK_HR;
constexpr HResult kEUnexpected = static_cast<HResult>(0x8000FFFFu);

void diag(const std::string& message) {
    com::detail::diagnostic(message.c_str());
}

// The XMA side of the AcpHal commands (contexts are global, allocated by
// AcpHalAllocateShapeContexts in xbox_middleware.cpp). Each enabled context has a voice
// (os/xma_decoder.h) behind its own lock: commands arrive from the title's time-critical mixer
// thread and must never wait behind another voice's decode. A worker thread, started by the first
// enable, fills every enabled context's output ring every 2 ms or when a command wakes it.
struct XmaSlot {
    std::mutex mutex;
    std::optional<xd::Voice> voice;  // set while enabled
};

struct XmaState {
    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable pump_done;
    bool woken = false;
    bool worker_started = false;
    bool worker_allowed = true;  // tests pump by hand
    uint32_t active_pumps = 0;
    uint64_t contexts_generation = 0;
    std::vector<bool> enabled;
    std::vector<std::shared_ptr<XmaSlot>> slots;
    uint64_t contexts = 0;  // guest address of the live context array
    XmaCommandStats stats;
    uint64_t test_contexts = 0;  // set_xma_contexts_for_tests
    uint32_t test_count = 0;
    xd::ApuMap test_map;  // set_xma_apu_map_for_tests
    std::atomic<uint32_t> dumped{0};
    std::atomic<uint32_t> trace_lines{0};
    // Filled by the worker (atomics: read by xma_command_stats without the voice locks).
    std::atomic<uint64_t> frames{0}, frames_decoded{0}, blocks_written{0}, loops{0}, exhausted{0}, bad_frames{0};
};
void quiesce_xma_contexts();
XmaState& xma() {
    static XmaState s = [] {
        pg6_mw_set_xma_contexts_released_callback(&quiesce_xma_contexts);
        return XmaState{};
    }();
    return s;
}

void quiesce_xma_contexts() {
    auto& x = xma();
    std::unique_lock<std::mutex> lock(x.mutex);
    x.enabled.clear();
    x.slots.clear();
    x.contexts = 0;
    ++x.contexts_generation;
    x.stats.enabled_now = 0;
    x.pump_done.wait(lock, [&x] { return x.active_pumps == 0; });
}

constexpr uint32_t kAcpEnableXma = 5, kAcpEnableXmaMask = 6, kAcpDisableXma = 7, kAcpDisableXmaMask = 8,
                   kAcpXmaReadAdvance = 18;

uint32_t* xma_context_words(uint64_t contexts, uint32_t index) {
    return static_cast<uint32_t*>(com::guest_to_host(contexts + uint64_t{index} * xd::kContextBytes,
                                                     xd::kContextBytes));
}

// APU address range -> host memory, through the single-lock APU range mapper (xbox_middleware.cpp).
uint8_t* map_apu_range(uint32_t apu, size_t bytes) {
    return static_cast<uint8_t*>(pg6_mw_apu_map_apu_range(apu, bytes));
}

xd::ApuMap apu_map_locked() {
    if (xma().test_map) {
        return xma().test_map;
    }
    return [](uint32_t apu, size_t bytes) { return map_apu_range(apu, bytes); };
}

void xma_trace(const std::string& line) {
    if (audio_trace_enabled() && xma().trace_lines.fetch_add(1, std::memory_order_relaxed) < 400) {
        diag("audio-trace: " + line);
    }
}

// PG6_XMA_DUMP=<dir>: the context words (<n>-<index>.ctx) and input buffer 0 (<n>-<index>.xma)
// of the first 32 contexts enabled, to test the decoder on the title's streams offline.
void dump_enabled_context(uint32_t index, xd::Context ctx, const xd::ApuMap& map) {
    static const char* dir = std::getenv("PG6_XMA_DUMP");
    if (dir == nullptr || dir[0] == '\0') {
        return;
    }
    const uint32_t n = xma().dumped.fetch_add(1, std::memory_order_relaxed);
    if (n >= 32) {
        return;
    }
    char path[1024];
    std::snprintf(path, sizeof(path), "%s/%02u-%u.ctx", dir, n, index);
    if (FILE* f = std::fopen(path, "wb")) {
        uint32_t words[xd::kContextWords];
        for (unsigned i = 0; i < xd::kContextWords; ++i) {
            words[i] = ctx.word(i);
        }
        std::fwrite(words, sizeof(words), 1, f);
        std::fclose(f);
    }
    const uint32_t packets = ctx.input_packets(0);
    const uint8_t* data =
        packets != 0 ? map(ctx.input_address(0) & ~(xd::kPacketBytes - 1), size_t{packets} * xd::kPacketBytes)
                     : nullptr;
    std::snprintf(path, sizeof(path), "%s/%02u-%u.xma", dir, n, index);
    if (FILE* f = std::fopen(path, "wb")) {
        if (data != nullptr) {
            std::fwrite(data, xd::kPacketBytes, packets, f);
        }
        std::fclose(f);
    }
}

// Folds a voice's counters into the totals (the caller holds the voice lock).
void account(const xd::VoiceStats& before, const xd::VoiceStats& after) {
    auto& x = xma();
    x.frames += after.frames - before.frames;
    x.frames_decoded += after.frames_decoded - before.frames_decoded;
    x.blocks_written += after.blocks_written - before.blocks_written;
    x.loops += after.loops - before.loops;
    x.exhausted += after.exhausted - before.exhausted;
    x.bad_frames += after.bad_frames - before.bad_frames;
}

// One pass over the enabled contexts.
void pump_enabled_voices() {
    std::vector<std::pair<uint32_t, std::shared_ptr<XmaSlot>>> active;
    uint64_t contexts = 0;
    xd::ApuMap map;
    {
        auto& x = xma();
        std::lock_guard<std::mutex> lock(x.mutex);
        contexts = x.contexts;
        if (contexts == 0 || x.slots.empty()) {
            return;
        }
        map = apu_map_locked();
        for (uint32_t i = 0; i < x.slots.size(); ++i) {
            if (x.enabled[i]) {
                active.emplace_back(i, x.slots[i]);
            }
        }
        if (active.empty()) {
            return;
        }
        ++x.active_pumps;
    }
    struct PumpGuard {
        ~PumpGuard() {
            auto& x = xma();
            std::lock_guard<std::mutex> lock(x.mutex);
            if (--x.active_pumps == 0) {
                x.pump_done.notify_all();
            }
        }
    } pump_guard;
    for (auto& [index, slot] : active) {
        std::lock_guard<std::mutex> lock(slot->mutex);
        uint32_t* words = xma_context_words(contexts, index);
        if (!slot->voice || words == nullptr) {
            continue;
        }
        const xd::VoiceStats before = slot->voice->stats;
        xd::pump(xd::Context(words), *slot->voice, map);
        account(before, slot->voice->stats);
    }
}

void xma_worker() {
    playgta6::os::name_this_thread(L"pg6-xma");
    auto& x = xma();
    for (;;) {
        {
            std::unique_lock<std::mutex> lock(x.mutex);
            if (x.stats.enabled_now == 0) {
                x.wake.wait(lock, [&x] { return x.woken || x.stats.enabled_now != 0; });
            } else {
                // A 256-byte block of mono 16-bit PCM lasts 2.7 ms at 48 kHz.
                x.wake.wait_for(lock, std::chrono::milliseconds(2), [&x] { return x.woken; });
            }
            x.woken = false;
            if (x.stats.enabled_now == 0) {
                continue;
            }
        }
        pump_enabled_voices();
    }
}

// The caller holds xma().mutex.
void wake_worker_locked() {
    auto& x = xma();
    x.woken = true;
    if (!x.worker_started && x.worker_allowed) {
        x.worker_started = true;
        std::thread(xma_worker).detach();
    }
    x.wake.notify_one();
}

// Applies enable/disable to one slot WITHOUT holding xma().mutex (only slot->mutex).
void apply_voice_enabled_unlocked(uint32_t index, const std::shared_ptr<XmaSlot>& slot, uint64_t contexts,
                                  const xd::ApuMap& map, bool on) {
    uint32_t* words = xma_context_words(contexts, index);
    bool should_dump = false;
    {
        std::lock_guard<std::mutex> lock(slot->mutex);
        auto& voice = slot->voice;
        if (on) {
            voice.emplace();
            if (words != nullptr) {
                xd::Context ctx(words);
                // nullptr defaults to the built-in WMA Pro / XMA2 subframe decoder (os/xma_decoder.cpp).
                xd::enable(ctx, *voice, nullptr);
                if (audio_trace_enabled() && xma().trace_lines.load(std::memory_order_relaxed) < 400) {
                    char line[300];
                    std::snprintf(line, sizeof(line),
                                  "XMA context %u enabled: %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x", index,
                                  ctx.word(0), ctx.word(1), ctx.word(2), ctx.word(3), ctx.word(4), ctx.word(5),
                                  ctx.word(6), ctx.word(7), ctx.word(8), ctx.word(9));
                    xma_trace(line);
                }
                should_dump = true;
            }
        } else if (voice) {
            const xd::VoiceStats& s = voice->stats;
            char line[300];
            std::snprintf(line, sizeof(line),
                          "XMA context %u disabled: %" PRIu64 " frames, %" PRIu64 " blocks, %" PRIu64
                          " loops, %" PRIu64 " input switches, ran dry %" PRIu64 "x",
                          index, s.frames, s.blocks_written, s.loops, s.input_switches, s.exhausted);
            xma_trace(line);
            voice.reset();
        }
    }
    if (should_dump && words != nullptr) {
        dump_enabled_context(index, xd::Context(words), map);
    }
}

void handle_xma_command(uint32_t command, uint64_t payload) {
    if (command != kAcpEnableXma && command != kAcpEnableXmaMask && command != kAcpDisableXma &&
        command != kAcpDisableXmaMask && command != kAcpXmaReadAdvance) {
        return;
    }
    uint32_t count = 0;
    uint64_t contexts = pg6_mw_xma_contexts(&count);
    auto& x = xma();

    struct VoiceOp {
        uint32_t index = 0;
        std::shared_ptr<XmaSlot> slot;
        bool on = false;
    };
    std::vector<VoiceOp> ops;
    std::shared_ptr<XmaSlot> advance_slot;
    uint32_t advance_index = 0;
    uint32_t advance_blocks = 0;
    bool wake_worker = false;
    xd::ApuMap map;

    {
        std::lock_guard<std::mutex> lock(x.mutex);
        if (x.test_contexts != 0) {
            contexts = x.test_contexts;
            count = x.test_count;
        }
        if (x.enabled.size() != count || x.contexts != contexts) {
            x.enabled.assign(count, false);
            x.slots.clear();
            for (uint32_t i = 0; i < count; ++i) {
                x.slots.push_back(std::make_shared<XmaSlot>());
            }
            x.contexts = contexts;
            ++x.contexts_generation;
            x.stats.enabled_now = 0;
        }
        uint32_t word[2] = {};
        if (payload == 0 || count == 0 || !com::read_guest(payload, word, sizeof(uint32_t))) {
            ++x.stats.rejected;
            return;
        }
        map = apu_map_locked();
        auto queue_toggle_locked = [&](uint32_t idx, bool on) {
            if (x.enabled[idx] != on) {
                x.enabled[idx] = on;
                if (on) {
                    ++x.stats.enabled_now;
                } else {
                    --x.stats.enabled_now;
                }
            }
            if (on) {
                wake_worker = true;
            }
            ops.push_back({idx, x.slots[idx], on});
        };
        switch (command) {
            case kAcpEnableXma:
            case kAcpDisableXma:
                if (word[0] >= count) {
                    ++x.stats.rejected;
                    return;
                }
                queue_toggle_locked(word[0], command == kAcpEnableXma);
                ++(command == kAcpEnableXma ? x.stats.enables : x.stats.disables);
                break;
            case kAcpEnableXmaMask:
            case kAcpDisableXmaMask: {
                const size_t mask_words = static_cast<size_t>((uint64_t{count} + 31u) / 32u);
                std::vector<uint32_t> mask(mask_words);
                if (!com::read_guest(payload, mask.data(), mask.size() * sizeof(uint32_t))) {
                    ++x.stats.rejected;
                    return;
                }
                const bool on = command == kAcpEnableXmaMask;
                for (uint32_t i = 0; i < count; ++i) {
                    if (((mask[i / 32] >> (i % 32)) & 1u) != 0) {
                        queue_toggle_locked(i, on);
                    }
                }
                ++(on ? x.stats.mask_enables : x.stats.mask_disables);
                break;
            }
            default: {  // 18: the title consumed output blocks; advance the read offset
                if (!com::read_guest(payload, word, sizeof(word)) || word[0] >= count ||
                    xma_context_words(contexts, word[0]) == nullptr) {
                    ++x.stats.rejected;
                    return;
                }
                advance_index = word[0];
                advance_blocks = word[1];
                advance_slot = x.slots[word[0]];
                ++x.stats.read_advances;
                wake_worker = x.enabled[word[0]];
                break;
            }
        }
        ++x.active_pumps;  // pin contexts memory until per-slot updates finish
    }

    struct CmdGuard {
        ~CmdGuard() {
            auto& x = xma();
            std::lock_guard<std::mutex> lock(x.mutex);
            if (--x.active_pumps == 0) {
                x.pump_done.notify_all();
            }
        }
    } cmd_guard;

    for (const VoiceOp& op : ops) {
        apply_voice_enabled_unlocked(op.index, op.slot, contexts, map, op.on);
    }
    if (advance_slot) {
        uint32_t* words = xma_context_words(contexts, advance_index);
        if (words != nullptr) {
            std::lock_guard<std::mutex> voice_lock(advance_slot->mutex);
            xd::advance_read(xd::Context(words), advance_blocks);
        }
    }
    if (wake_worker) {
        std::lock_guard<std::mutex> lock(x.mutex);
        wake_worker_locked();
    }
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
        bool first = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!connected_) {
                diag("AcpHal: command " + std::to_string(command) + " submitted while not connected");
                return kEUnexpected;
            }
            first = (++commands_[command] == 1);
        }
        handle_xma_command(command, payload);
        if (command == kAcpEnableXma || command == kAcpEnableXmaMask) {
            audio_stream_log::log().record("xma_enable", command, payload, arg);
        }
        if (first) {
            char line[200];
            std::snprintf(line, sizeof(line),
                          "AcpHal: first command %u (tag 0x%08x, arg 0x%08x, payload 0x%" PRIx64
                          ", flags %u): accepted",
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
// Model: no processing hardware, but every command is acknowledged with a result {type, voice
// index} (the rest zero) in submission order, so voices move through their states and are freed.
// Command 3 is decoded synchronously (os/xapu_opus.h; per-voice decoder state, reset by commands 1
// and 2), the float32 PCM written to the output address, and a result {3, voice index, zeros, +0x34
// bytes written} queued (0 bytes when the packet does not decode).
// Evidence (R40 verification, static survey of the executable's five XApu import thunks and all
// their callers; no runtime trace): the title only ever sends types 1, 2, 3 and 5 (one sender
// each, 0x1400483d0 type 5, 0x144ee9930 type 1, 0x144f20900 type 2, 0x144f20ad0 type 3), and its
// result drain (0x144eaa840: GetResultCount(handle, 0), then that many DequeueResult calls into a
// zeroed 0x58-byte struct; result type at +0, voice at +4) acts only on result types 1, 2, 3 and
// 5 and ignores any other type, and skips a failing dequeue. So acknowledging types the title never
// sends (the "every type gets a result" rule) is harmless but unobservable. No caller checks the
// HRESULT of XApuEnqueueCommand (each discards it and advances its voice state), so the
// queue-full HRESULT below is a modelling choice, not something the title reacts to; a dropped
// command would strand that voice in its pending state.
//
// Codec census (R40, menu-flow run of 420 s): the executable sends XApu only types 1, 2 and 3
// (5 is wired but was not seen); every decode command carries one Opus packet (287 decoded, 0
// failed, 287 type-3 results). That Opus path carries only short one-shot sounds: a menu-flow run
// with PG6_AUDIO_TRACE=1 shows every voice on slot 0, started and freed after 7 or 23 packets
// (0.1 / 0.56 s), one at a time. Music and ambience are XMA, not Opus: AcpHalAllocateShapeContexts
// asks for 350 contexts of kind 4 (XMA, 64 bytes each, xbox_middleware.cpp), and the stream
// consumer at 0x140078120 reads the XMA context's output write / read offsets and only copies PCM
// out (then sends AcpHal command 0x12, tag 0xF1F1F1F1) once the hardware has advanced the write
// offset. The decode worker above moves that offset, but writes silence (no WMA Pro frame decoder
// yet). A menu-flow run (R40, 238 s) never set up or enabled one XMA context (xma_contexts_touched
// and xma_enable_commands 0), so the menus' music does not come this way; the boot report's
// `xma_*` fields show when a later screen does. The streamed banks themselves load
// fine: in that run all 1043 XCRDAPI requests on audio.rpf (zlib) and 557 ReadFile calls succeed
// within 1 ms. XDspActivate is never reached and the executable imports no XAudio2 API; PCM and
// ADPCM are decoded by the title's own code. The boot report's audio section carries the
// per-type counts and `xapu_unhandled_commands` so a new codec path shows up there first.
// ---------------------------------------------------------------------------

constexpr size_t kXApuResultBytes = 0x38;
constexpr size_t kXApuDecodeCommandBytes = 0x40;
constexpr uint32_t kXApuChannelBytes = 0xF00;     // 960 float32 frames of one channel
constexpr uint32_t kXApuMaxPacketBytes = 0x280;   // one input slot
constexpr uint32_t kXApuMaxOutputBytes = 0x1E00;  // one output slot

using XApuResult = std::array<uint8_t, kXApuResultBytes>;
// O36: a title that never dequeues must not grow a connection's result queue without limit: past
// this many pending results the oldest is dropped.
constexpr size_t kXApuMaxPendingResults = 4096;

void push_bounded(std::deque<XApuResult>& queue, const XApuResult& result) {
    queue.push_back(result);
    while (queue.size() > kXApuMaxPendingResults) {
        queue.pop_front();
    }
}

constexpr size_t kXApuMaxOutputFloats = kXApuMaxOutputBytes / sizeof(float);
static_assert(kXApuMaxOutputBytes % sizeof(float) == 0);

struct XApuVoiceSlot {
    std::mutex mutex;
    xapu_opus::VoiceDecoder decoder;
    std::array<float, kXApuMaxOutputFloats> pcm{};
};

struct XApuConn {
    std::mutex event_mutex;
    uint64_t event = 0;  // auto-reset result event (0 if none or disconnected)
};

struct XApuState {
    std::mutex mutex;
    std::map<uint64_t, std::deque<XApuResult>> queues;  // handle -> pending results
    std::map<uint32_t, uint64_t> commands;              // command type -> count
    std::map<uint64_t, uint64_t> memory;                // handle -> shared memory
    std::map<uint64_t, std::shared_ptr<XApuConn>> conns;
    std::map<uint64_t, size_t> limits;                  // handle -> max outstanding results (0 = unlimited)
    std::map<std::pair<uint64_t, uint32_t>, std::shared_ptr<XApuVoiceSlot>> voice_slots;  // (handle, voice)
    uint64_t decoded_packets = 0;
    uint64_t failed_packets = 0;
    uint64_t decoded_bytes = 0;
    uint64_t type3_results = 0;   // type-3 results queued (one per decode command)
    uint64_t results_posted = 0;  // results of every type queued
    uint64_t unhandled = 0;       // commands of a type the model does not know (no result)
    bool decode_logged = false;
    bool failure_logged = false;
    std::map<std::pair<uint64_t, uint32_t>, std::pair<uint64_t, int64_t>> trace_voices;  // packets, start ms
    uint32_t trace_lines = 0;
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

// Connect parameters: see os/xapu_connect.h (six u32 words, layout verified against the title's call).
// 0x8007006C (Win32 code 108) is a guess: no caller in the executable inspects the
// enqueue result, so any failing value behaves the same for the title.
constexpr HResult kXApuQueueFull = static_cast<HResult>(0x8007006Cu);

xapu_connect::Params parse_connect_params(uint64_t params) {
    uint32_t w[6] = {};
    if (!com::read_guest(params, w, sizeof(w))) {
        return {};
    }
    return xapu_connect::parse(w);
}

HResult xapu_connect(uint64_t params, GuestPtr<uint64_t> shared, OutPtr out) {
    if (out.is_null() || !out.write(0)) {
        return com::E_POINTER_HR;
    }
    if (params == 0 || shared.is_null()) {
        return com::E_INVALIDARG_HR;
    }
    // PG6_AUDIO_TRACE=1: the first connect logs the raw bytes of the parameter struct (the layout is
    // documented in os/xapu_connect.h).
    {
        static std::atomic<bool> logged{false};
        const char* trace = std::getenv("PG6_AUDIO_TRACE");
        if (trace != nullptr && trace[0] == '1' && !logged.exchange(true)) {
            uint8_t b[64] = {};
            if (com::read_guest(params, b, sizeof(b))) {
                std::string hex;
                char tmp[4];
                for (size_t i = 0; i < sizeof(b); ++i) {
                    std::snprintf(tmp, sizeof(tmp), "%02x", b[i]);
                    hex += tmp;
                    if ((i & 3) == 3) hex += ' ';
                }
                std::fprintf(stderr, "xapu: XApuConnect params 0x%" PRIx64 " len %zu raw: %s\n",
                             params, sizeof(b), hex.c_str());
            }
        }
    }
    const uint64_t handle = com::guest_alloc(64);
    const xapu_connect::Params cp = parse_connect_params(params);
    const uint64_t memory = alloc_zeroed(cp.memory_bytes);
    if (handle == 0 || memory == 0) {
        if (handle != 0) {
            com::guest_free(handle);
        }
        if (memory != 0) {
            com::guest_free(memory);
        }
        return com::E_OUTOFMEMORY_HR;
    }
    auto conn = std::make_shared<XApuConn>();
    uint64_t event = 0;
    if (cp.result_event) {
        event = static_cast<uint64_t>(win32::CreateEventA(nullptr, false, false, nullptr));
        conn->event = event;
    }
    if (!shared.write(memory) ||
        (cp.result_event && !com::write_guest(shared.va() + 8, &event, sizeof(event)))) {
        if (event != 0) {
            win32::CloseHandle(static_cast<win32::Handle>(event));
        }
        com::guest_free(handle);
        com::guest_free(memory);
        return com::E_POINTER_HR;
    }
    {
        std::lock_guard<std::mutex> lock(xapu().mutex);
        xapu().queues[handle];
        xapu().memory[handle] = memory;
        xapu().limits[handle] = cp.outstanding_limit;
        xapu().conns[handle] = std::move(conn);
    }
    out.write(handle);
    return kOk;
}

HResult xapu_disconnect(uint64_t handle) {
    uint64_t memory = 0;
    std::shared_ptr<XApuConn> conn;
    {
        std::lock_guard<std::mutex> lock(xapu().mutex);
        if (xapu().queues.erase(handle) == 0) {
            return com::E_INVALIDARG_HR;
        }
        memory = xapu().memory[handle];
        xapu().memory.erase(handle);
        xapu().limits.erase(handle);
        if (const auto c = xapu().conns.find(handle); c != xapu().conns.end()) {
            conn = std::move(c->second);
            xapu().conns.erase(c);
        }
        auto& slots = xapu().voice_slots;
        slots.erase(slots.lower_bound({handle, 0u}), slots.upper_bound({handle, ~0u}));
        auto& traces = xapu().trace_voices;
        traces.erase(traces.lower_bound({handle, 0u}), traces.upper_bound({handle, ~0u}));
    }
    if (conn) {
        // Close the event under conn->event_mutex so an in-flight SetEvent can never touch a recycled handle.
        std::lock_guard<std::mutex> elock(conn->event_mutex);
        if (conn->event != 0) {
            win32::CloseHandle(static_cast<win32::Handle>(conn->event));
            conn->event = 0;
        }
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

struct DecodeOutcome {
    XApuResult result{};
    bool ok = false;
    uint32_t bytes = 0;
    uint32_t voice = 0;
    uint32_t packet_bytes = 0;
    uint8_t toc = 0;
    uint32_t sample_rate = 48000u;
    uint32_t channels = 0;
    uint32_t frames_per_packet = 0;
    uint32_t skip = 0;
    uint32_t capacity = 0;
    uint64_t packet_va = 0;
    uint64_t output_va = 0;
};

// Command 3 (runs WITHOUT holding xapu().mutex — only the per-voice slot mutex): decodes the Opus
// packet into the guest output area so concurrent voices never serialize on a global decode lock.
DecodeOutcome decode_command_unlocked(const std::shared_ptr<XApuVoiceSlot>& slot, const uint8_t* cmd) {
    DecodeOutcome out{};
    out.voice = field<uint32_t>(cmd, 0x04);
    const float rate = field<float>(cmd, 0x1C);
    out.skip = field<uint32_t>(cmd, 0x20);
    out.frames_per_packet = field<uint32_t>(cmd, 0x24);
    out.packet_bytes = field<uint32_t>(cmd, 0x28);
    out.packet_va = field<uint64_t>(cmd, 0x2C);
    out.capacity = std::min(field<uint32_t>(cmd, 0x34), kXApuMaxOutputBytes);
    out.output_va = field<uint64_t>(cmd, 0x38);
    // Derive channel count from frames_per_packet when valid so 10 ms (480-frame) or 24 kHz packets
    // do not collapse to 0 or 1 channels; fall back to kXApuChannelBytes (960 frames) otherwise.
    const uint32_t channel_bytes =
        (out.frames_per_packet > 0u && out.frames_per_packet <= xapu_opus::kMaxOpusFrames)
            ? (out.frames_per_packet * static_cast<uint32_t>(sizeof(float)))
            : kXApuChannelBytes;
    const uint32_t raw_channels = channel_bytes != 0u ? (out.capacity / channel_bytes) : 0u;
    out.channels = std::min<uint32_t>(raw_channels, 2u);
    out.sample_rate = rate >= 1.0f && rate < 1.0e6f ? static_cast<uint32_t>(rate) : 48000u;

    uint8_t packet[kXApuMaxPacketBytes];
    const bool readable = out.packet_bytes != 0 && out.packet_bytes <= sizeof(packet) && out.packet_va != 0 &&
                          com::read_guest(out.packet_va, packet, out.packet_bytes);
    if (readable) {
        out.toc = packet[0];
    }
    if (readable && out.channels != 0 && out.output_va != 0 && slot) {
        std::lock_guard<std::mutex> vlock(slot->mutex);
        const size_t max_frames = out.capacity / (out.channels * sizeof(float));
        float* pcm = slot->pcm.data();
        const int frames = slot->decoder.decode(packet, out.packet_bytes, out.sample_rate, out.channels, pcm, max_frames);
        if (frames >= 0) {
            const uint32_t dropped = std::min<uint32_t>(out.skip, static_cast<uint32_t>(frames));
            const uint32_t bytes = (static_cast<uint32_t>(frames) - dropped) * out.channels * sizeof(float);
            if (bytes == 0 || com::write_guest(out.output_va, pcm + size_t{dropped} * out.channels, bytes)) {
                out.ok = true;
                out.bytes = bytes;
                out.result = make_result(3, out.voice, bytes);
                return out;
            }
        }
    }
    out.result = make_result(3, out.voice, 0);
    return out;
}

// PG6_AUDIO_TRACE=1 (the caller holds the lock): one line per voice start (type 1), free (type 2,
// with the packets decoded since its start) and other non-decode command; at most 400 lines.
void trace_command(uint64_t handle, uint32_t type, uint32_t voice) {
    auto& s = xapu();
    const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
    auto& v = s.trace_voices[{handle, voice}];
    if (type == 3) {
        ++v.first;
        return;
    }
    if (type == 1) {
        v = {0, now};
    }
    const uint64_t pkts = v.first;
    const int64_t start_ms = v.second;
    if (type == 2) {
        s.trace_voices.erase({handle, voice});
    }
    if (s.trace_lines++ >= 400) {
        return;
    }
    char line[200];
    std::snprintf(line, sizeof(line), "audio-trace: XApu command %u voice %u (t=%" PRId64 "): %" PRIu64
                  " packets over %" PRId64 " ms since its start",
                  type, voice, now, pkts, start_ms != 0 ? now - start_ms : -1);
    diag(line);
}

HResult xapu_enqueue_command(uint64_t handle, uint64_t command, uint32_t /*flags*/) {
    uint32_t head[2] = {};
    if (command == 0 || !com::read_guest(command, head, sizeof(head))) {
        return com::E_INVALIDARG_HR;
    }
    const uint32_t type = head[0];
    const uint32_t voice = head[1];
    uint8_t decode[kXApuDecodeCommandBytes] = {};
    if (type == 3 && !com::read_guest(command, decode, sizeof(decode))) {
        return com::E_INVALIDARG_HR;
    }

    std::shared_ptr<XApuVoiceSlot> slot;
    std::shared_ptr<XApuConn> conn;
    const bool needs_reset = (type == 1 || type == 2 || type == 5);
    {
        std::lock_guard<std::mutex> lock(xapu().mutex);
        auto it = xapu().queues.find(handle);
        if (it == xapu().queues.end()) {
            return com::E_INVALIDARG_HR;
        }
        if (const size_t limit = xapu().limits[handle]; limit != 0 && it->second.size() >= limit) {
            return kXApuQueueFull;
        }
        if (type == 3) {
            auto& entry = xapu().voice_slots[{handle, voice}];
            if (!entry) {
                entry = std::make_shared<XApuVoiceSlot>();
            }
            slot = entry;
        } else if (needs_reset) {
            if (auto d = xapu().voice_slots.find({handle, voice}); d != xapu().voice_slots.end()) {
                slot = d->second;
            }
        }
    }

    // Run Opus decode or decoder reset OUTSIDE xapu().mutex so concurrent voices and result drains
    // never block each other.
    DecodeOutcome decoded{};
    if (type == 3) {
        decoded = decode_command_unlocked(slot, decode);
    } else if (needs_reset && slot) {
        std::lock_guard<std::mutex> vlock(slot->mutex);
        slot->decoder.reset();
    }

    {
        std::lock_guard<std::mutex> lock(xapu().mutex);
        auto it = xapu().queues.find(handle);
        if (it == xapu().queues.end()) {
            return com::E_INVALIDARG_HR;
        }
        if (const size_t limit = xapu().limits[handle]; limit != 0 && it->second.size() >= limit) {
            return kXApuQueueFull;
        }
        const bool acknowledged = true;
        const uint64_t seen = ++xapu().commands[type];
        if (type != 3 || seen <= 8) {
            // Voice start (1), free (2), XMA-related (5); decodes (3) only the first few.
            static const char* const kNames[] = {"xapu_cmd_0", "xapu_voice_start", "xapu_voice_free",
                                                 "xapu_decode", "xapu_cmd_4", "xapu_cmd_5"};
            audio_stream_log::log().record(type < 6 ? kNames[type] : "xapu_cmd_other", handle, head[1], type);
        } else if (type == 3) {
            audio_stream_log::log().record("xapu_decode", handle, head[1], type);  // counted, not kept
        }
        if (seen == 1) {
            diag("XApu: first command of type " + std::to_string(type) +
                 (acknowledged ? ": acknowledged with a result" : ": accepted without a result"));
        }
        if (audio_trace_enabled()) {
            trace_command(handle, type, voice);
        }
        if (type == 3) {
            auto& s = xapu();
            if (decoded.ok) {
                ++s.decoded_packets;
                s.decoded_bytes += decoded.bytes;
                if (!s.decode_logged) {
                    s.decode_logged = true;
                    char line[256];
                    std::snprintf(line, sizeof(line),
                                  "XApu: first decode: voice %u, %u bytes (TOC 0x%02x), %u Hz, %u channel(s), "
                                  "%u frames/packet, skip %u -> %u bytes of float32%s",
                                  decoded.voice, decoded.packet_bytes, decoded.toc, decoded.sample_rate,
                                  decoded.channels, decoded.frames_per_packet, decoded.skip, decoded.bytes,
                                  xapu_opus::available() ? "" : " (silence: built without libopus)");
                    diag(line);
                }
            } else {
                ++s.failed_packets;
                if (!s.failure_logged) {
                    s.failure_logged = true;
                    char line[256];
                    std::snprintf(line, sizeof(line),
                                  "XApu: decode failed: voice %u, packet 0x%" PRIx64 " (%u bytes), capacity %u, "
                                  "output 0x%" PRIx64 "; result posted with 0 bytes",
                                  decoded.voice, decoded.packet_va, decoded.packet_bytes, decoded.capacity,
                                  decoded.output_va);
                    diag(line);
                }
            }
            push_bounded(it->second, decoded.result);
            ++xapu().type3_results;
        } else {
            push_bounded(it->second, make_result(type, voice));
        }
        if (type != 1 && type != 2 && type != 3 && type != 5) {
            ++xapu().unhandled;  // no behaviour modelled for this type (still acknowledged)
        }
        ++xapu().results_posted;
        if (const auto c = xapu().conns.find(handle); c != xapu().conns.end()) {
            conn = c->second;
        }
    }
    // Wake the title's audio thread after releasing xapu().mutex, guarded by conn->event_mutex
    // against concurrent xapu_disconnect() closing the Win32 event handle.
    if (conn) {
        std::lock_guard<std::mutex> elock(conn->event_mutex);
        if (conn->event != 0) {
            win32::SetEvent(static_cast<win32::Handle>(conn->event));
        }
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
    std::lock_guard<std::mutex> lock(xapu().mutex);
    auto it = xapu().queues.find(handle);
    if (it == xapu().queues.end()) {
        return com::E_INVALIDARG_HR;
    }
    if (it->second.empty()) {
        return com::E_FAIL_HR;
    }
    const XApuResult& bytes = it->second.front();
    if (!com::write_guest(result, bytes.data(), bytes.size())) {
        return com::E_POINTER_HR;
    }
    it->second.pop_front();
    return kOk;
}

}  // namespace

bool audio_trace_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("PG6_AUDIO_TRACE");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

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

XmaCommandStats xma_command_stats() {
    auto& x = xma();
    std::lock_guard<std::mutex> lock(x.mutex);
    XmaCommandStats s = x.stats;
    s.frames = x.frames;
    s.frames_decoded = x.frames_decoded;
    s.blocks_written = x.blocks_written;
    s.loops = x.loops;
    s.input_exhausted = x.exhausted;
    s.bad_frames = x.bad_frames;
    return s;
}

bool xma_context_enabled(uint32_t index) {
    std::lock_guard<std::mutex> lock(xma().mutex);
    return index < xma().enabled.size() && xma().enabled[index];
}

void reset_xma_commands() {
    auto& x = xma();
    std::unique_lock<std::mutex> lock(x.mutex);
    x.enabled.clear();
    x.slots.clear();
    x.contexts = 0;
    ++x.contexts_generation;
    x.stats = {};
    x.pump_done.wait(lock, [&x] { return x.active_pumps == 0; });
    x.frames = 0;
    x.frames_decoded = 0;
    x.blocks_written = 0;
    x.loops = 0;
    x.exhausted = 0;
    x.bad_frames = 0;
}

void set_xma_apu_map_for_tests(std::function<uint8_t*(uint32_t, size_t)> map) {
    std::lock_guard<std::mutex> lock(xma().mutex);
    xma().test_map = std::move(map);
}

void set_xma_worker_allowed_for_tests(bool allowed) {
    std::lock_guard<std::mutex> lock(xma().mutex);
    xma().worker_allowed = allowed;
}

void pump_xma_for_tests() {
    pump_enabled_voices();
}

void set_xma_contexts_for_tests(uint64_t va, uint32_t count) {
    std::lock_guard<std::mutex> lock(xma().mutex);
    xma().test_contexts = va;
    xma().test_count = va != 0 ? count : 0;
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
