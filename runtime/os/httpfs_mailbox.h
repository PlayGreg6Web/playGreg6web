#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string_view>
#include <type_traits>

namespace playgta6::os {

inline constexpr uint32_t kHttpfsMailboxVersion = 1;
inline constexpr size_t kHttpfsHeaderBytes = 40;
inline constexpr size_t kHttpfsPayloadBytes = 64 * 1024;  // 64 KiB fixed transfer buffer

// Mailbox state machine transitions for Atomics.wait / Atomics.notify:
//   Idle -> RequestReady (client) -> ResponseReady / Error (worker) -> Idle (client)
//   Any  -> Shutdown (teardown to wake blocked responder threads)
enum class HttpfsStatus : int32_t {
    Idle = 0,
    RequestReady = 1,
    ResponseReady = 2,
    Error = -1,
    Shutdown = -2,
};

enum class HttpfsOp : uint32_t {
    None = 0,
    Open = 1,
    Read = 2,
    Close = 3,
};

// Fixed-layout shared-memory mailbox protocol struct shared between Wasm64 pthread workers
// and web/workers/io_worker.js over a SharedArrayBuffer.
//
// Byte offsets (all naturally aligned, zero padding):
//   0..4   : status           (int32_t, Atomics.wait / Atomics.notify target)
//   4..8   : op               (uint32_t, HttpfsOp)
//   8..12  : handle           (int32_t, virtual file handle)
//   12..16 : payload_capacity (uint32_t, usable payload bytes)
//   16..24 : offset           (uint64_t, byte offset for Read)
//   24..32 : length           (uint64_t, path length for Open / byte count for Read)
//   32..40 : result           (int64_t, handle for Open / bytes read for Read / <0 on error)
//   40..   : payload[]        (uint8_t[kHttpfsPayloadBytes])
struct alignas(8) HttpfsMailbox {
    int32_t status = static_cast<int32_t>(HttpfsStatus::Idle);
    uint32_t op = static_cast<uint32_t>(HttpfsOp::None);
    int32_t handle = -1;
    uint32_t payload_capacity = static_cast<uint32_t>(kHttpfsPayloadBytes);
    uint64_t offset = 0;
    uint64_t length = 0;
    int64_t result = 0;
    uint8_t payload[kHttpfsPayloadBytes]{};
};

static_assert(std::is_standard_layout_v<HttpfsMailbox>,
              "HttpfsMailbox must be standard-layout for SAB interop");
static_assert(offsetof(HttpfsMailbox, status) == 0, "status offset must be 0");
static_assert(offsetof(HttpfsMailbox, op) == 4, "op offset must be 4");
static_assert(offsetof(HttpfsMailbox, handle) == 8, "handle offset must be 8");
static_assert(offsetof(HttpfsMailbox, payload_capacity) == 12, "payload_capacity offset must be 12");
static_assert(offsetof(HttpfsMailbox, offset) == 16, "offset offset must be 16");
static_assert(offsetof(HttpfsMailbox, length) == 24, "length offset must be 24");
static_assert(offsetof(HttpfsMailbox, result) == 32, "result offset must be 32");
static_assert(offsetof(HttpfsMailbox, payload) == kHttpfsHeaderBytes, "payload offset must be 40");
static_assert(sizeof(HttpfsMailbox) == kHttpfsHeaderBytes + kHttpfsPayloadBytes,
              "HttpfsMailbox total size must equal header + payload");

// Thread-safe C++ client helper that encodes Open/Read/Close requests into an HttpfsMailbox.
// Supports both synchronous injectable responder callbacks (for native unit tests) and
// asynchronous worker-to-worker Atomics.wait / Atomics.notify responders.
class HttpfsMailboxClient {
public:
    using ResponderCallback = std::function<void(HttpfsMailbox& mailbox)>;

    explicit HttpfsMailboxClient(HttpfsMailbox* mailbox,
                                 ResponderCallback responder = nullptr);

    void set_responder(ResponderCallback responder);
    size_t payload_capacity() const;

    // Opens a relative path (e.g. "update/x64/dlc.rpf"). Returns handle >= 0 or < 0 on error.
    int32_t open(std::string_view rel_path);

    // Reads up to `len` bytes from `handle` at `offset` into `dst`.
    // Automatically chunks reads exceeding payload_capacity().
    // Returns total bytes read (>= 0), or < 0 on error.
    int64_t read(int32_t handle, uint64_t offset, void* dst, size_t len);

    // Closes a virtual file handle. Returns true on success.
    bool close(int32_t handle);

    // Responder-side helpers (for native worker threads and unit tests).
    static void init_mailbox(HttpfsMailbox& mailbox,
                             uint32_t payload_capacity = static_cast<uint32_t>(kHttpfsPayloadBytes));
    static bool wait_for_request(HttpfsMailbox& mailbox);
    static void complete_response(HttpfsMailbox& mailbox,
                                  int64_t result,
                                  HttpfsStatus status = HttpfsStatus::ResponseReady);
    static void shutdown_waiter(HttpfsMailbox& mailbox);

private:
    HttpfsStatus execute_roundtrip_locked(int64_t& out_result);
    size_t effective_capacity_locked() const;

    HttpfsMailbox* mailbox_ = nullptr;
    ResponderCallback responder_;
    mutable std::mutex mutex_;
};

}  // namespace playgta6::os
