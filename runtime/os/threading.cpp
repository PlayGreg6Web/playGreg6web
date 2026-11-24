#include "os/threading.h"

#include <chrono>
#include <memory>
#include <unordered_map>

namespace playgta6::os {

namespace {

std::atomic<uint64_t> g_next_tls_table_id{1};

struct ThreadSlotEntry {
    uint64_t generation = 0;
    void* value = nullptr;
};

struct ThreadTableStorage {
    std::array<ThreadSlotEntry, TlsTable::kMaxSlots> slots{};
};

std::unordered_map<uint64_t, std::unique_ptr<ThreadTableStorage>>& thread_tls_map() {
    thread_local std::unordered_map<uint64_t, std::unique_ptr<ThreadTableStorage>> map;
    return map;
}

}  // namespace

// ---------------------------------------------------------------------------
// TlsTable
// ---------------------------------------------------------------------------

TlsTable::TlsTable()
    : table_id_(g_next_tls_table_id.fetch_add(1, std::memory_order_relaxed)) {
    for (auto& state : slot_state_) {
        state.store(0, std::memory_order_relaxed);
    }
}

uint32_t TlsTable::tls_alloc() {
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    for (uint32_t i = 0; i < kMaxSlots; ++i) {
        const uint64_t s = slot_state_[i].load(std::memory_order_relaxed);
        if ((s & 1ULL) == 0ULL) {
            const uint64_t next_gen = (s >> 1) + 1ULL;
            slot_state_[i].store((next_gen << 1) | 1ULL, std::memory_order_release);
            allocated_count_.fetch_add(1, std::memory_order_relaxed);
            return i;
        }
    }
    return kInvalidIndex;
}

bool TlsTable::tls_free(uint32_t index) {
    if (index >= kMaxSlots) {
        return false;
    }
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    const uint64_t s = slot_state_[index].load(std::memory_order_relaxed);
    if ((s & 1ULL) == 0ULL) {
        return false;
    }
    const uint64_t next_gen = (s >> 1) + 1ULL;
    slot_state_[index].store((next_gen << 1), std::memory_order_release);
    allocated_count_.fetch_sub(1, std::memory_order_relaxed);
    return true;
}

void* TlsTable::tls_get(uint32_t index) const {
    if (index >= kMaxSlots) {
        return nullptr;
    }
    const uint64_t s = slot_state_[index].load(std::memory_order_acquire);
    if ((s & 1ULL) == 0ULL) {
        return nullptr;
    }
    const uint64_t gen = s >> 1;
    auto& map = thread_tls_map();
    auto it = map.find(table_id_);
    if (it == map.end()) {
        return nullptr;
    }
    const ThreadSlotEntry& entry = it->second->slots[index];
    if (entry.generation != gen) {
        return nullptr;
    }
    return entry.value;
}

bool TlsTable::tls_set(uint32_t index, void* value) {
    if (index >= kMaxSlots) {
        return false;
    }
    const uint64_t s = slot_state_[index].load(std::memory_order_acquire);
    if ((s & 1ULL) == 0ULL) {
        return false;
    }
    const uint64_t gen = s >> 1;
    auto& map = thread_tls_map();
    auto& storage = map[table_id_];
    if (!storage) {
        storage = std::make_unique<ThreadTableStorage>();
    }
    storage->slots[index] = ThreadSlotEntry{gen, value};
    return true;
}

uint32_t TlsTable::allocated_count() const {
    return allocated_count_.load(std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// SrwLock
// ---------------------------------------------------------------------------

void SrwLock::lock_exclusive() {
    mutex_.lock();
}

bool SrwLock::try_lock_exclusive() {
    return mutex_.try_lock();
}

void SrwLock::unlock_exclusive() {
    mutex_.unlock();
}

void SrwLock::lock_shared() {
    mutex_.lock_shared();
}

bool SrwLock::try_lock_shared() {
    return mutex_.try_lock_shared();
}

void SrwLock::unlock_shared() {
    mutex_.unlock_shared();
}

// ---------------------------------------------------------------------------
// XTaskQueue
// ---------------------------------------------------------------------------

XTaskQueue::XTaskQueue(DispatchMode work_mode,
                       DispatchMode completion_mode,
                       uint32_t work_threads) {
    work_port_.mode = work_mode;
    completion_port_.mode = completion_mode;

    if (work_mode == DispatchMode::ThreadPool) {
        start_workers(work_port_, work_threads == 0 ? 1u : work_threads);
    }
    if (completion_mode == DispatchMode::ThreadPool) {
        start_workers(completion_port_, 1u);
    }
}

XTaskQueue::~XTaskQueue() {
    // Entries still queued own resources (their callbacks free them): cancel them so every
    // cancellable one runs once with canceled=true instead of being destroyed unrun.
    terminate(true, true);
}

namespace {

// Counts a callback as active for its scope, also when the callback exits by an exception, so a
// terminating queue never waits for a callback that is gone.
template <typename Port>
struct ActiveCallback {
    explicit ActiveCallback(Port& p) : port(p) {}
    ~ActiveCallback() {
        std::lock_guard<std::mutex> lock(port.mutex);
        --port.active_callbacks;
        if (port.active_callbacks == 0) {
            port.done_cv.notify_all();
        }
    }
    Port& port;
};

}  // namespace

void XTaskQueue::start_workers(Port& port, uint32_t count) {
    std::lock_guard<std::mutex> lock(port.join_mutex);
    port.workers.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        port.workers.emplace_back([this, &port]() { worker_loop(port); });
    }
}

void XTaskQueue::worker_loop(Port& port) {
    while (true) {
        TaskEntry entry;
        {
            std::unique_lock<std::mutex> lock(port.mutex);
            port.cv.wait(lock, [&]() {
                return !port.queue.empty() || port.terminated;
            });
            if (port.queue.empty()) {
                if (port.terminated) {
                    break;
                }
                continue;
            }
            entry = std::move(port.queue.front());
            port.queue.pop_front();
            ++port.active_callbacks;
        }

        {
            ActiveCallback<Port> active(port);
            if (entry.callback) {
                entry.callback(false);
            }
        }
    }
}

bool XTaskQueue::submit_to_port(Port& port, TaskEntry entry) {
    if (!entry.callback) {
        return false;
    }

    if (port.mode == DispatchMode::Immediate) {
        {
            std::lock_guard<std::mutex> lock(port.mutex);
            if (port.terminated) {
                return false;
            }
            ++port.active_callbacks;
        }
        {
            ActiveCallback<Port> active(port);
            entry.callback(false);
        }
        return true;
    }

    {
        std::lock_guard<std::mutex> lock(port.mutex);
        if (port.terminated) {
            return false;
        }
        port.queue.push_back(std::move(entry));
    }
    port.cv.notify_one();
    return true;
}

bool XTaskQueue::submit_work(SimpleCallback cb) {
    if (!cb) {
        return false;
    }
    TaskEntry entry;
    entry.callback = [fn = std::move(cb)](bool canceled) {
        if (!canceled) {
            fn();
        }
    };
    entry.invoke_on_cancel = false;
    return submit_to_port(work_port_, std::move(entry));
}

bool XTaskQueue::submit_work_cancellable(TaskCallback cb) {
    if (!cb) {
        return false;
    }
    TaskEntry entry;
    entry.callback = std::move(cb);
    entry.invoke_on_cancel = true;
    return submit_to_port(work_port_, std::move(entry));
}

bool XTaskQueue::submit_completion(SimpleCallback cb) {
    if (!cb) {
        return false;
    }
    TaskEntry entry;
    entry.callback = [fn = std::move(cb)](bool canceled) {
        if (!canceled) {
            fn();
        }
    };
    entry.invoke_on_cancel = false;
    return submit_to_port(completion_port_, std::move(entry));
}

bool XTaskQueue::submit_completion_cancellable(TaskCallback cb) {
    if (!cb) {
        return false;
    }
    TaskEntry entry;
    entry.callback = std::move(cb);
    entry.invoke_on_cancel = true;
    return submit_to_port(completion_port_, std::move(entry));
}

bool XTaskQueue::dispatch_from_port(Port& port, uint32_t timeout_ms) {
    TaskEntry entry;
    {
        std::unique_lock<std::mutex> lock(port.mutex);
        if (port.queue.empty()) {
            if (timeout_ms == 0 || port.terminated) {
                return false;
            }
            const bool ready = port.cv.wait_for(
                lock,
                std::chrono::milliseconds(timeout_ms),
                [&]() { return !port.queue.empty() || port.terminated; });
            if (!ready || port.queue.empty()) {
                return false;
            }
        }
        entry = std::move(port.queue.front());
        port.queue.pop_front();
        ++port.active_callbacks;
    }

    {
        ActiveCallback<Port> active(port);
        if (entry.callback) {
            entry.callback(false);
        }
    }
    return true;
}

bool XTaskQueue::dispatch_work(uint32_t timeout_ms) {
    return dispatch_from_port(work_port_, timeout_ms);
}

bool XTaskQueue::dispatch_completion(uint32_t timeout_ms) {
    return dispatch_from_port(completion_port_, timeout_ms);
}

size_t XTaskQueue::drain_port(Port& port) {
    size_t count = 0;
    while (dispatch_from_port(port, 0)) {
        ++count;
    }
    return count;
}

size_t XTaskQueue::drain_work() {
    return drain_port(work_port_);
}

size_t XTaskQueue::drain_completion() {
    return drain_port(completion_port_);
}

void XTaskQueue::terminate_port(Port& port, bool wait, bool cancel_pending) {
    std::deque<TaskEntry> canceled_entries;
    {
        std::lock_guard<std::mutex> lock(port.mutex);
        port.terminated = true;
        if (cancel_pending) {
            canceled_entries.swap(port.queue);
        }
    }
    port.cv.notify_all();

    for (auto& entry : canceled_entries) {
        if (entry.callback && entry.invoke_on_cancel) {
            entry.callback(true);
        }
    }

    if (!wait) {
        return;
    }

    std::vector<std::thread> to_join;
    bool called_from_worker = false;
    const auto self_id = std::this_thread::get_id();
    {
        std::lock_guard<std::mutex> lock(port.join_mutex);
        std::vector<std::thread> remaining;
        for (auto& worker : port.workers) {
            if (!worker.joinable()) {
                continue;
            }
            if (worker.get_id() == self_id) {
                called_from_worker = true;
                remaining.push_back(std::move(worker));
            } else {
                to_join.push_back(std::move(worker));
            }
        }
        port.workers = std::move(remaining);
    }

    for (auto& worker : to_join) {
        worker.join();
    }

    if (!called_from_worker) {
        std::unique_lock<std::mutex> lock(port.mutex);
        port.done_cv.wait(lock, [&]() { return port.active_callbacks == 0; });
    }
}

void XTaskQueue::terminate(bool wait, bool cancel_pending) {
    terminated_.store(true, std::memory_order_release);
    terminate_port(work_port_, wait, cancel_pending);
    terminate_port(completion_port_, wait, cancel_pending);
}

bool XTaskQueue::is_terminated() const {
    return terminated_.load(std::memory_order_acquire);
}

size_t XTaskQueue::pending_work_count() const {
    std::lock_guard<std::mutex> lock(work_port_.mutex);
    return work_port_.queue.size();
}

size_t XTaskQueue::pending_completion_count() const {
    std::lock_guard<std::mutex> lock(completion_port_.mutex);
    return completion_port_.queue.size();
}

}  // namespace playgta6::os
