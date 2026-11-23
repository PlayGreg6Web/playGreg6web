#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>

namespace playgta6::os {

// Win32 / Xbox Thread Local Storage slot table (TlsAlloc / TlsFree / TlsGetValue / TlsSetValue).
// Supports up to 1024 concurrent slots with per-thread isolation and generation-tracked slot reuse.
class TlsTable {
public:
    static constexpr uint32_t kMaxSlots = 1024;
    static constexpr uint32_t kInvalidIndex = 0xFFFFFFFFu;

    TlsTable();
    ~TlsTable() = default;

    TlsTable(const TlsTable&) = delete;
    TlsTable& operator=(const TlsTable&) = delete;

    // Allocates a free TLS slot and initializes its value to nullptr on all threads.
    // Returns kInvalidIndex if all kMaxSlots are in use.
    uint32_t tls_alloc();

    // Frees a previously allocated TLS slot. Returns false if index is invalid or unallocated.
    bool tls_free(uint32_t index);

    // Returns the calling thread's value for slot `index`, or nullptr if invalid/unallocated/unset.
    void* tls_get(uint32_t index) const;

    // Sets the calling thread's value for slot `index`. Returns false if invalid/unallocated.
    bool tls_set(uint32_t index, void* value);

    uint32_t allocated_count() const;

private:
    uint64_t table_id_ = 0;
    mutable std::mutex alloc_mutex_;
    // Encodes (generation << 1) | allocated_bit for lock-free tls_get/tls_set validation.
    std::array<std::atomic<uint64_t>, kMaxSlots> slot_state_{};
    std::atomic<uint32_t> allocated_count_{0};
};

// Win32 / Xbox Slim Reader/Writer Lock (SRWLOCK) wrapping std::shared_mutex.
// Also satisfies C++ Lockable and SharedLockable for std::unique_lock / std::shared_lock.
class SrwLock {
public:
    SrwLock() = default;
    ~SrwLock() = default;

    SrwLock(const SrwLock&) = delete;
    SrwLock& operator=(const SrwLock&) = delete;

    void lock_exclusive();
    bool try_lock_exclusive();
    void unlock_exclusive();

    void lock_shared();
    bool try_lock_shared();
    void unlock_shared();

    // Standard Lockable interface aliases.
    void lock() { lock_exclusive(); }
    bool try_lock() { return try_lock_exclusive(); }
    void unlock() { unlock_exclusive(); }

private:
    std::shared_mutex mutex_;
};

// Xbox GDK XTaskQueue shim with separate Work and Completion ports.
// Supports Manual dispatch, optional background ThreadPool worker draining, and Immediate execution.
class XTaskQueue {
public:
    enum class DispatchMode {
        Manual,
        ThreadPool,
        Immediate,
    };

    using SimpleCallback = std::function<void()>;
    using TaskCallback = std::function<void(bool canceled)>;

    explicit XTaskQueue(DispatchMode work_mode = DispatchMode::Manual,
                        DispatchMode completion_mode = DispatchMode::Manual,
                        uint32_t work_threads = 1);
    ~XTaskQueue();

    XTaskQueue(const XTaskQueue&) = delete;
    XTaskQueue& operator=(const XTaskQueue&) = delete;

    bool submit_work(SimpleCallback cb);
    bool submit_work_cancellable(TaskCallback cb);

    bool submit_completion(SimpleCallback cb);
    bool submit_completion_cancellable(TaskCallback cb);

    // Pops and executes at most one queued item from the respective port.
    // If timeout_ms > 0, waits up to timeout_ms milliseconds for an item to arrive.
    bool dispatch_work(uint32_t timeout_ms = 0);
    bool dispatch_completion(uint32_t timeout_ms = 0);

    // Non-blocking drain of all currently queued items on the calling thread.
    size_t drain_work();
    size_t drain_completion();

    // Stops accepting new callbacks. When cancel_pending is true, pending queued callbacks
    // are canceled (cancellable callbacks are invoked with canceled=true).
    // When wait is true, joins worker threads and waits for in-flight callbacks to finish.
    void terminate(bool wait = true, bool cancel_pending = false);

    bool is_terminated() const;
    size_t pending_work_count() const;
    size_t pending_completion_count() const;

private:
    struct TaskEntry {
        TaskCallback callback;
        bool invoke_on_cancel = false;
    };

    struct Port {
        DispatchMode mode = DispatchMode::Manual;
        mutable std::mutex mutex;
        std::condition_variable cv;
        std::condition_variable done_cv;
        std::deque<TaskEntry> queue;
        size_t active_callbacks = 0;
        bool terminated = false;
        std::mutex join_mutex;
        std::vector<std::thread> workers;
    };

    bool submit_to_port(Port& port, TaskEntry entry);
    bool dispatch_from_port(Port& port, uint32_t timeout_ms);
    size_t drain_port(Port& port);
    void start_workers(Port& port, uint32_t count);
    void worker_loop(Port& port);
    void terminate_port(Port& port, bool wait, bool cancel_pending);

    Port work_port_;
    Port completion_port_;
    std::atomic<bool> terminated_{false};
};

}  // namespace playgta6::os
