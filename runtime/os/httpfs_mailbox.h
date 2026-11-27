#pragma once

#include <atomic>
#include <condition_variable>
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
    // 4 is reserved for io_worker.js opt-in /pg6_dump/ write forwarding (HTTPFS_OP_WRITE).
    Stat = 5,
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
//   24..32 : length           (uint64_t, path length for Open/Stat / byte count for Read)
//   32..40 : result           (int64_t, handle for Open / bytes read for Read / file size for Stat / <0 on error)
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

// FIFO (ticket) lock: waiters get the lock in arrival order. A thread that releases and at once
// re-acquires it (a bulk reader looping over chunks) queues behind the threads already waiting,
// where a plain mutex usually lets it barge in again and starve them.
class HttpfsFairMutex {
public:
    void lock();
    void unlock();

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    uint64_t next_ticket_ = 0;
    uint64_t now_serving_ = 0;
};

// Host calls a client can forward requests to directly instead of staging them in a mailbox
// (the in-process browser bridge: the host reads straight into the caller's buffer).
struct HttpfsDirectBackend {
    std::function<int32_t(std::string_view rel_path)> open;
    std::function<int64_t(int32_t handle, uint64_t offset, void* dst, size_t len)> read;
    std::function<bool(int32_t handle)> close;
    // Optional fast 1-roundtrip file size query: returns exact byte size (>= 0), or < 0 on error
    // (-2 = ENOENT, -38 = unsupported). When unset, HttpfsMailboxClient::stat() returns -38.
    std::function<int64_t(std::string_view rel_path)> stat;
    // When true, the backend callbacks are thread-safe / multi-lane and HttpfsMailboxClient does
    // not hold its global HttpfsFairMutex across direct open/read/close/stat calls.
    bool concurrent = false;
    // Maximum bytes requested per direct read() call (0 falls back to kHttpfsPayloadBytes).
    size_t max_chunk_bytes = kHttpfsPayloadBytes;
};

// Thread-safe C++ client helper that encodes Open/Read/Close requests into an HttpfsMailbox.
// Supports both synchronous injectable responder callbacks (for native unit tests) and
// asynchronous worker-to-worker Atomics.wait / Atomics.notify responders.
//
// By default, one request is in flight at a time (the downstream mailbox / host channel is
// single-lane). Reads larger than the payload capacity are split into chunks, and the lock is
// taken per chunk in FIFO order: a short read from another thread waits for at most one chunk of
// a bulk read, not for the whole bulk read. Multi-lane direct backends may set
// HttpfsDirectBackend::concurrent = true to allow parallel open/read/close calls without holding
// the global lock.
class HttpfsMailboxClient {
public:
    using ResponderCallback = std::function<void(HttpfsMailbox& mailbox)>;

    explicit HttpfsMailboxClient(HttpfsMailbox* mailbox,
                                 ResponderCallback responder = nullptr);
    // Direct mode: requests go to `backend` (chunked to backend.max_chunk_bytes, defaulting to
    // kHttpfsPayloadBytes, with no payload copy).
    explicit HttpfsMailboxClient(HttpfsDirectBackend backend);

    void set_responder(ResponderCallback responder);

    // Bounded retry for transient host read failures (a failed range request). A failed chunk is
    // re-issued up to `max_retries` times with exponential backoff (initial_backoff_ms, doubling,
    // capped at 1 s); only then does read() return < 0 so the title sees a real error, never a
    // short read or zero-filled data. Not retried: a mailbox shutdown, or a malformed response.
    struct RetryPolicy {
        uint32_t max_retries = 3;
        uint32_t initial_backoff_ms = 20;
        std::function<void(uint32_t ms)> sleep;  // test hook; default sleeps the thread
    };
    void set_retry_policy(RetryPolicy policy);
    uint64_t retried_chunks() const { return retried_chunks_.load(std::memory_order_relaxed); }
    size_t payload_capacity() const;

    // Opens a relative path (e.g. "update/x64/dlc.rpf"). Returns handle >= 0 or < 0 on error.
    int32_t open(std::string_view rel_path);

    // Reads up to `len` bytes from `handle` at `offset` into `dst`.
    // Automatically chunks reads exceeding payload_capacity().
    // Returns total bytes read (>= 0), or < 0 on error.
    int64_t read(int32_t handle, uint64_t offset, void* dst, size_t len);

    // Closes a virtual file handle. Returns true on success.
    bool close(int32_t handle);

    // Queries the exact byte size of `rel_path` in 1 mailbox roundtrip (e.g. from io_worker's
    // in-memory manifest without network fetches). Returns byte size (>= 0), or < 0 on error
    // (-2 = ENOENT, -22 = EINVAL, -38 = ENOSYS when unsupported).
    int64_t stat(std::string_view rel_path);

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
    // One chunk (min(remaining, capacity) bytes, stored in *asked) under the lock: bytes read
    // (>= 0) or < 0 on error.
    int64_t read_chunk(int32_t handle, uint64_t offset, uint8_t* dst, size_t remaining, size_t* asked,
                       bool* retryable);

    HttpfsMailbox* mailbox_ = nullptr;
    ResponderCallback responder_;
    HttpfsDirectBackend direct_;
    bool direct_mode_ = false;
    RetryPolicy retry_;
    std::atomic<uint64_t> retried_chunks_{0};
    mutable HttpfsFairMutex mutex_;
};

}  // namespace playgta6::os
