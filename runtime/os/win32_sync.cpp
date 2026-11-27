#include "os/win32_sync.h"
#include "os/thread_name.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

#include "os/thread_spans.h"
#include "os/threading.h"

static_assert(sizeof(Pg6CriticalSection) <= 40, "CRITICAL_SECTION is 40 bytes on x64");

// ---------------------------------------------------------------------------
// Thread pool objects (global namespace, matching the opaque declarations in the header)
// ---------------------------------------------------------------------------

struct Pg6TpCallbackInstance {
    uint8_t reserved = 0;
};

struct Pg6TpWork {
    Pg6TpWorkCallback callback = nullptr;
    void* context = nullptr;

    std::mutex mutex;
    std::condition_variable idle_cv;
    uint64_t epoch = 0;    // bumped by WaitForThreadpoolWorkCallbacks(cancel = TRUE)
    uint32_t queued = 0;   // submitted in the current epoch but not started yet
    uint32_t running = 0;  // callbacks currently executing

    // The guest's reference. CloseThreadpoolWork drops it; queued callbacks keep their own
    // references, so the object outlives Close until every outstanding callback has finished.
    std::shared_ptr<Pg6TpWork> self;
};

namespace playgta6::os::win32 {

namespace {

using Clock = std::chrono::steady_clock;

// Locking model: every dispatcher object has its own lock guarding its state and waiter list.
// A wait-all locks all of its objects in address order; nothing else ever holds two object locks.
// A WaitBlock's mutex is innermost: it is taken under an object lock (to wake the waiter) and is
// never held while acquiring an object lock.

// One per WaitFor* call; registered on every object it waits for.
//
// Wait-any blocks are satisfied by direct handoff: a signaler that finds the object signaled for
// a registered waiter claims the block (CAS Waiting -> Claimed), applies the acquire side effect
// on the waiter's behalf (consumes the auto-reset event / semaphore unit, takes the mutex) and
// publishes the result. Waiters are queued FIFO per object, so auto-reset events and semaphores
// are handed out in arrival order. A timed-out waiter must win CAS Waiting -> TimedOut, so a
// wait never both times out and consumes a signal.
//
// Wait-all blocks are only notified (wake_seq bump); the waiter re-checks with all its objects
// locked and acquires them atomically or not at all.
struct WaitBlock {
    enum : uint32_t { kWaiting, kClaimed, kSatisfied, kTimedOut };

    WaitBlock() = default;
    WaitBlock(uint32_t waiter_thread, bool all) : thread_id(waiter_thread), wait_all(all) {}

    void reset(uint32_t waiter_thread, bool all) {
        thread_id = waiter_thread;
        wait_all = all;
        state.store(kWaiting, std::memory_order_relaxed);
        wake_seq.store(0, std::memory_order_relaxed);
        index = 0;
        abandoned = false;
        newly_owned = false;
    }

    // Called with the object lock held; the block outlives that lock because the waiter must take
    // it to deregister before returning.
    void publish_satisfied(uint32_t object_index, bool was_abandoned, bool took_mutex) {
        index = object_index;
        abandoned = was_abandoned;
        newly_owned = took_mutex;
        std::lock_guard<std::mutex> guard(mutex);
        state.store(kSatisfied, std::memory_order_release);
        cv.notify_one();
    }

    void notify_wait_all() {
        std::lock_guard<std::mutex> guard(mutex);
        wake_seq.fetch_add(1, std::memory_order_release);
        cv.notify_one();
    }

    uint32_t thread_id = 0;
    bool wait_all = false;
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<uint32_t> state{kWaiting};
    std::atomic<uint32_t> wake_seq{0};
    // Written by the claimer before state becomes kSatisfied.
    uint32_t index = 0;
    bool abandoned = false;
    bool newly_owned = false;
};

struct ReusableWaitBlockSlot {
    WaitBlock block;
    bool in_use = false;
};

thread_local ReusableWaitBlockSlot t_wait_block;

// Acquires the calling thread's reusable WaitBlock (avoiding per-wait mutex/cv init & destroy),
// falling back to a stack-allocated WaitBlock if a wait is ever re-entered on the same thread.
class ScopedWaitBlock {
public:
    ScopedWaitBlock(uint32_t thread_id, bool wait_all) {
        if (!t_wait_block.in_use) {
            t_wait_block.in_use = true;
            block_ = &t_wait_block.block;
            owns_tls_ = true;
        } else {
            fallback_.emplace();
            block_ = &*fallback_;
            owns_tls_ = false;
        }
        block_->reset(thread_id, wait_all);
    }

    ~ScopedWaitBlock() {
        if (owns_tls_) {
            t_wait_block.in_use = false;
        }
    }

    ScopedWaitBlock(const ScopedWaitBlock&) = delete;
    ScopedWaitBlock& operator=(const ScopedWaitBlock&) = delete;

    WaitBlock& get() { return *block_; }

private:
    WaitBlock* block_ = nullptr;
    std::optional<WaitBlock> fallback_;
    bool owns_tls_ = false;
};

inline void cpu_pause() noexcept {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(_M_ARM64)
#if defined(_MSC_VER)
    __yield();
#else
    __asm__ __volatile__("yield");
#endif
#else
    // No-op on Wasm and other targets.
#endif
}

struct WaiterLink {
    WaitBlock* block;
    uint32_t index;  // position of the object in the waiter's handle array
};

// Result of a satisfied wait on one object.
struct AcquireOutcome {
    bool abandoned = false;    // an abandoned mutex was acquired
    bool newly_owned = false;  // a mutex changed owner to the waiter (track it for abandonment)
};

struct ThreadInfo {
    ThreadInfo(uint32_t thread_id, uint64_t affinity) : id(thread_id), affinity_mask(affinity) {}

    const uint32_t id;
    std::atomic<uint64_t> affinity_mask;
};

std::atomic<uint32_t> g_next_thread_index{1};
std::atomic<uint64_t> g_process_affinity_mask{kDefaultProcessAffinityMask};

uint32_t allocate_thread_id() {
    // Win32 thread ids are non-zero multiples of 4.
    return g_next_thread_index.fetch_add(1, std::memory_order_relaxed) * 4u;
}

class MutexObject;

// Per-OS-thread state. owned_mutexes is only touched by its own thread.
struct ThreadState {
    std::shared_ptr<ThreadInfo> info;
    std::vector<std::shared_ptr<MutexObject>> owned_mutexes;

    ThreadState() = default;
    ThreadState(const ThreadState&) = delete;
    ThreadState& operator=(const ThreadState&) = delete;
    ~ThreadState();
};

thread_local ThreadState t_thread;

ThreadInfo& current_thread_info() {
    if (!t_thread.info) {
        t_thread.info = std::make_shared<ThreadInfo>(
            allocate_thread_id(), g_process_affinity_mask.load(std::memory_order_relaxed));
    }
    return *t_thread.info;
}

// PG6_EVENT_TRACE=1: SetEvent / ResetEvent / ReleaseSemaphore / waits on chosen handles, with thread, guest caller
// and time, to find who signals an event a thread is blocked on and how late.
// PG6_EVENT_TRACE_HANDLES=0x15f,0x160 picks the handles (default 0x15f). Timed-out polls are
// counted, not logged: the satisfied wait reports how many polls and how long it took.
struct EventTrace {
    bool enabled = false;
    std::vector<Handle> handles;
    Clock::time_point start = Clock::now();
};

std::atomic<uint64_t (*)()> g_event_trace_caller{nullptr};

const EventTrace& event_trace_config() {
    static const EventTrace config = [] {
        EventTrace t;
        const char* on = std::getenv("PG6_EVENT_TRACE");
        t.enabled = on != nullptr && on[0] != '\0' && on[0] != '0';
        if (!t.enabled) {
            return t;
        }
        const char* list = std::getenv("PG6_EVENT_TRACE_HANDLES");
        std::string text = list != nullptr ? list : "0x15f";
        size_t pos = 0;
        while (pos < text.size()) {
            size_t end = text.find(',', pos);
            if (end == std::string::npos) {
                end = text.size();
            }
            const std::string item = text.substr(pos, end - pos);
            if (!item.empty()) {
                t.handles.push_back(static_cast<Handle>(std::strtoull(item.c_str(), nullptr, 0)));
            }
            pos = end + 1;
        }
        return t;
    }();
    return config;
}

bool event_traced(Handle handle) {
    const EventTrace& t = event_trace_config();
    return t.enabled && std::find(t.handles.begin(), t.handles.end(), handle) != t.handles.end();
}

// Timed-out polls of a traced handle by this thread since its last satisfied wait.
struct PollRun {
    Handle handle = 0;
    uint32_t polls = 0;
    Clock::time_point first{};
};
thread_local PollRun t_poll_run;

void event_trace_line(const char* op, Handle handle, const char* detail) {
    const auto caller_fn = g_event_trace_caller.load(std::memory_order_relaxed);
    const uint64_t caller = caller_fn != nullptr ? caller_fn() : 0;
    const double ms =
        std::chrono::duration<double, std::milli>(Clock::now() - event_trace_config().start).count();
    std::fprintf(stderr, "[event-trace] %10.3f ms tid=%u %-6s h=0x%" PRIx64 " caller=0x%" PRIx64 "%s%s\n",
                 ms, current_thread_info().id, op, static_cast<uint64_t>(handle), caller,
                 detail[0] != '\0' ? " " : "", detail);
}

void event_trace_wait(Handle handle, uint32_t timeout_ms, uint32_t result) {
    if (!event_traced(handle)) {
        return;
    }
    PollRun& run = t_poll_run;
    if (run.handle != handle) {
        run = PollRun{handle, 0, Clock::now()};
    }
    if (result == WAIT_TIMEOUT && timeout_ms == 0) {
        if (run.polls++ == 0) {
            run.first = Clock::now();
        }
        return;
    }
    char detail[128];
    const double waited =
        run.polls != 0 ? std::chrono::duration<double, std::milli>(Clock::now() - run.first).count() : 0.0;
    std::snprintf(detail, sizeof(detail), "timeout=%d result=0x%x after %u polls / %.3f ms",
                  static_cast<int32_t>(timeout_ms), result, run.polls, waited);
    event_trace_line("wait", handle, detail);
    run.polls = 0;
}

}  // namespace

void set_event_trace_caller(uint64_t (*caller)()) {
    g_event_trace_caller.store(caller, std::memory_order_relaxed);
}

bool event_trace_enabled(Handle handle) {
    return event_traced(handle);
}

void event_trace_note(const char* op, Handle handle, const char* detail) {
    if (event_traced(handle)) {
        event_trace_line(op, handle, detail != nullptr ? detail : "");
    }
}

// ---------------------------------------------------------------------------
// Dispatcher objects
// ---------------------------------------------------------------------------

class DispatcherObject {
public:
    enum class Type : uint8_t {
        Event,
        Semaphore,
        Mutex,
        Thread,
    };

    explicit DispatcherObject(Type object_type) : type(object_type) {}
    virtual ~DispatcherObject() = default;

    DispatcherObject(const DispatcherObject&) = delete;
    DispatcherObject& operator=(const DispatcherObject&) = delete;

    // Both are called with this object's lock held (by the waiter or by a signaler handing off).
    virtual bool is_signaled_for(uint32_t thread_id) const = 0;
    // Applies the side effect of a satisfied wait for `thread_id`.
    virtual AcquireOutcome acquire(uint32_t thread_id) = 0;

    // Called with `lock` held after the state may have become signaled: hands the object to
    // queued wait-any waiters in FIFO order while it stays signaled and wakes wait-all waiters.
    void wake_waiters_locked() {
        // Lets a spinning waiter (see wait_for_objects) notice a state change without the lock.
        signal_seq.fetch_add(1, std::memory_order_release);
        for (size_t i = 0; i < waiters.size();) {
            WaitBlock* block = waiters[i].block;
            if (!is_signaled_for(block->thread_id)) {
                break;  // nothing left to hand out (a mutex is signaled for every non-owner alike)
            }
            if (block->wait_all) {
                block->notify_wait_all();
                ++i;
                continue;
            }
            uint32_t expected = WaitBlock::kWaiting;
            if (!block->state.compare_exchange_strong(expected, WaitBlock::kClaimed,
                                                      std::memory_order_acq_rel)) {
                ++i;  // satisfied through another object, or timing out
                continue;
            }
            const uint32_t index = waiters[i].index;
            waiters.erase(waiters.begin() + static_cast<std::ptrdiff_t>(i));
            const AcquireOutcome outcome = acquire(block->thread_id);
            block->publish_satisfied(index, outcome.abandoned, outcome.newly_owned);
        }
    }

    const Type type;
    std::mutex lock;
    std::vector<WaiterLink> waiters;  // FIFO
    std::atomic<uint32_t> signal_seq{0};
};

namespace {

using ObjectType = DispatcherObject::Type;

class EventObject final : public DispatcherObject {
public:
    EventObject(bool manual, bool initial)
        : DispatcherObject(ObjectType::Event), manual_reset(manual), signaled(initial) {}

    bool is_signaled_for(uint32_t) const override { return signaled; }
    AcquireOutcome acquire(uint32_t) override {
        if (!manual_reset) {
            signaled = false;
        }
        return {};
    }

    const bool manual_reset;
    bool signaled;
};

class SemaphoreObject final : public DispatcherObject {
public:
    SemaphoreObject(int32_t initial, int32_t maximum_count)
        : DispatcherObject(ObjectType::Semaphore), count(initial), maximum(maximum_count) {}

    bool is_signaled_for(uint32_t) const override { return count > 0; }
    AcquireOutcome acquire(uint32_t) override {
        --count;
        return {};
    }

    int32_t count;
    const int32_t maximum;
};

class MutexObject final : public DispatcherObject,
                          public std::enable_shared_from_this<MutexObject> {
public:
    MutexObject() : DispatcherObject(ObjectType::Mutex) {}

    bool is_signaled_for(uint32_t thread_id) const override {
        return owner == 0 || owner == thread_id;
    }

    // May run on a signaler thread (handoff); the waiter records ownership in its own
    // t_thread.owned_mutexes when it sees newly_owned. It cannot exit before doing so.
    AcquireOutcome acquire(uint32_t thread_id) override {
        if (owner == thread_id) {
            ++recursion;
            return {};
        }
        owner = thread_id;
        recursion = 1;
        const bool was_abandoned = abandoned;
        abandoned = false;
        return {was_abandoned, true};
    }

    uint32_t owner = 0;  // owning thread id, 0 when free
    uint32_t recursion = 0;
    bool abandoned = false;
};

class ThreadObject final : public DispatcherObject {
public:
    ThreadObject(std::shared_ptr<ThreadInfo> thread_info,
                 Pg6ThreadStartRoutine start_routine,
                 void* start_parameter,
                 bool suspended)
        : DispatcherObject(ObjectType::Thread),
          info(std::move(thread_info)),
          start(start_routine),
          parameter(start_parameter),
          suspend_count(suspended ? 1u : 0u) {}

    bool is_signaled_for(uint32_t) const override { return exited; }
    AcquireOutcome acquire(uint32_t) override { return {}; }

    const std::shared_ptr<ThreadInfo> info;
    const Pg6ThreadStartRoutine start;
    void* const parameter;
    bool exited = false;
    uint32_t exit_code = STILL_ACTIVE;
    uint32_t suspend_count;
    std::condition_variable resume_cv;  // waited on with this object's lock
};

void abandon_owned_mutexes(ThreadState& state) {
    for (auto& mutex : state.owned_mutexes) {
        std::lock_guard<std::mutex> guard(mutex->lock);
        mutex->owner = 0;
        mutex->recursion = 0;
        mutex->abandoned = true;
        mutex->wake_waiters_locked();
    }
    state.owned_mutexes.clear();
}

// Threads not created through CreateThread (host threads, pool workers) abandon their mutexes here.
ThreadState::~ThreadState() {
    abandon_owned_mutexes(*this);
}

// Records a mutex the current thread just took ownership of, so its exit abandons it.
void note_acquired(DispatcherObject& object, const AcquireOutcome& outcome) {
    if (outcome.newly_owned) {
        t_thread.owned_mutexes.push_back(static_cast<MutexObject&>(object).shared_from_this());
    }
}

template <typename T>
std::shared_ptr<T> lookup_as(Handle handle, ObjectType type) {
    auto object = lookup_dispatcher_object(handle);
    if (!object || object->type != type) {
        return nullptr;
    }
    return std::static_pointer_cast<T>(std::move(object));
}

// ---------------------------------------------------------------------------
// Named objects (one namespace shared by events, semaphores and mutexes)
// ---------------------------------------------------------------------------

struct NamedObjects {
    std::mutex mutex;
    std::unordered_map<std::string, std::weak_ptr<DispatcherObject>> objects;
};

NamedObjects& named_objects() {
    static auto* names = new NamedObjects();
    return *names;
}

// Lock order: named-object mutex -> handle table mutex / object lock. Never the reverse.
template <typename Factory>
Handle create_dispatcher_handle(std::string_view name, ObjectType type, Factory make_object) {
    if (name.empty()) {
        const Handle handle = insert_dispatcher_object(make_object());
        SetLastError(ERROR_SUCCESS);
        return handle;
    }

    auto& names = named_objects();
    std::lock_guard<std::mutex> lock(names.mutex);
    const std::string key(name);
    auto it = names.objects.find(key);
    if (it != names.objects.end()) {
        if (auto existing = it->second.lock()) {
            if (existing->type != type) {
                SetLastError(ERROR_INVALID_HANDLE);
                return kNullHandle;
            }
            const Handle handle = insert_dispatcher_object(std::move(existing));
            SetLastError(ERROR_ALREADY_EXISTS);
            return handle;
        }
        names.objects.erase(it);
    }

    std::shared_ptr<DispatcherObject> object = make_object();
    names.objects.emplace(key, object);
    const Handle handle = insert_dispatcher_object(std::move(object));
    SetLastError(ERROR_SUCCESS);
    return handle;
}

std::string narrow_name(const char* name) {
    return name != nullptr ? std::string(name) : std::string{};
}

std::string widen_name(const char16_t* name) {
    return name != nullptr ? utf16_to_utf8(name) : std::string{};
}

Handle create_event(bool manual_reset, bool initial_state, std::string_view name) {
    return create_dispatcher_handle(name, ObjectType::Event, [&]() {
        return std::make_shared<EventObject>(manual_reset, initial_state);
    });
}

Handle create_semaphore(int32_t initial_count, int32_t maximum_count, std::string_view name) {
    if (maximum_count <= 0 || initial_count < 0 || initial_count > maximum_count) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return kNullHandle;
    }
    return create_dispatcher_handle(name, ObjectType::Semaphore, [&]() {
        return std::make_shared<SemaphoreObject>(initial_count, maximum_count);
    });
}

Handle create_mutex(bool initial_owner, std::string_view name) {
    // initial_owner is ignored when opening an existing named mutex (factory is not invoked).
    return create_dispatcher_handle(name, ObjectType::Mutex, [&]() {
        auto mutex = std::make_shared<MutexObject>();
        if (initial_owner) {
            const uint32_t thread_id = current_thread_info().id;
            std::lock_guard<std::mutex> lock(mutex->lock);
            note_acquired(*mutex, mutex->acquire(thread_id));
        }
        return mutex;
    });
}

// ---------------------------------------------------------------------------
// Waiting
// ---------------------------------------------------------------------------

using ObjectArray = std::array<std::shared_ptr<DispatcherObject>, MAXIMUM_WAIT_OBJECTS>;

// Upper bound of the spin phase before a blocking wait parks the thread.
constexpr auto kWaitSpinWindow = std::chrono::microseconds(20);

// Short bounded spin before parking: producer/consumer pairs (render thread, Bink workers)
// usually signal within microseconds, and a futex sleep + wake costs far more. Only a
// wake_waiters_locked() on one of the objects ends the spin early; the caller then re-checks the
// state under the object locks, so no wait is ever skipped.
void spin_for_single_signal(const DispatcherObject& object,
                            uint32_t seen,
                            bool infinite,
                            Clock::time_point deadline) {
    if (object.signal_seq.load(std::memory_order_acquire) != seen) {
        return;
    }
    const Clock::time_point now = Clock::now();
    if (!infinite && now >= deadline) {
        return;
    }
    const Clock::time_point window_end = now + kWaitSpinWindow;
    const Clock::time_point spin_end =
        (!infinite && deadline < window_end) ? deadline : window_end;
    for (uint32_t iter = 1;; ++iter) {
        if (object.signal_seq.load(std::memory_order_acquire) != seen) {
            return;
        }
        cpu_pause();
        if ((iter & 63u) == 0 && Clock::now() >= spin_end) {
            return;
        }
        if ((iter & 31u) == 0) {
            std::this_thread::yield();
        }
    }
}

void spin_for_signal(const ObjectArray& objects,
                     uint32_t count,
                     bool infinite,
                     Clock::time_point deadline) {
    std::array<uint32_t, MAXIMUM_WAIT_OBJECTS> seen{};
    for (uint32_t i = 0; i < count; ++i) {
        seen[i] = objects[i]->signal_seq.load(std::memory_order_acquire);
    }
    const Clock::time_point now = Clock::now();
    if (!infinite && now >= deadline) {
        return;
    }
    const Clock::time_point window_end = now + kWaitSpinWindow;
    const Clock::time_point spin_end =
        (!infinite && deadline < window_end) ? deadline : window_end;
    for (uint32_t iter = 1;; ++iter) {
        for (uint32_t i = 0; i < count; ++i) {
            if (objects[i]->signal_seq.load(std::memory_order_acquire) != seen[i]) {
                return;
            }
        }
        cpu_pause();
        if ((iter & 63u) == 0 && Clock::now() >= spin_end) {
            return;
        }
        if ((iter & 31u) == 0) {
            std::this_thread::yield();
        }
    }
}

// Wait-any, one object lock at a time in index order (so the lowest signaled index wins among
// those seen). With `enqueue`, the block is queued on every object found unsignaled, atomically
// with that check, so a later signal hands the object off instead of being lost. Returns true
// once the block is claimed (by this call or, while enqueuing, by a concurrent handoff).
bool try_wait_any(const ObjectArray& objects, uint32_t count, WaitBlock& block, bool enqueue,
                  uint32_t& registered) {
    for (uint32_t i = 0; i < count; ++i) {
        DispatcherObject& object = *objects[i];
        std::lock_guard<std::mutex> guard(object.lock);
        if (object.is_signaled_for(block.thread_id)) {
            uint32_t expected = WaitBlock::kWaiting;
            if (block.state.compare_exchange_strong(expected, WaitBlock::kClaimed,
                                                    std::memory_order_acq_rel)) {
                const AcquireOutcome outcome = object.acquire(block.thread_id);
                block.index = i;
                block.abandoned = outcome.abandoned;
                block.newly_owned = outcome.newly_owned;
                block.state.store(WaitBlock::kSatisfied, std::memory_order_release);
            }
            return true;
        }
        if (enqueue) {
            if (block.state.load(std::memory_order_acquire) != WaitBlock::kWaiting) {
                return true;  // handed off through an object queued earlier in this pass
            }
            object.waiters.push_back({&block, i});
            registered = i + 1;
        }
    }
    return false;
}

void unregister_wait_block(const ObjectArray& objects, uint32_t registered, WaitBlock& block) {
    for (uint32_t i = 0; i < registered; ++i) {
        DispatcherObject& object = *objects[i];
        std::lock_guard<std::mutex> guard(object.lock);
        std::erase_if(object.waiters, [&](const WaiterLink& link) { return link.block == &block; });
    }
}

// Wait-all: all object locks taken in address order, then everything or nothing is acquired.
// With `enqueue`, the block is queued on every object while they are all still locked.
bool try_wait_all(const ObjectArray& objects,
                  const std::array<uint32_t, MAXIMUM_WAIT_OBJECTS>& lock_order,
                  uint32_t count,
                  WaitBlock& block,
                  bool enqueue,
                  uint32_t& result) {
    for (uint32_t k = 0; k < count; ++k) {
        objects[lock_order[k]]->lock.lock();
    }
    bool satisfied = true;
    for (uint32_t i = 0; i < count && satisfied; ++i) {
        satisfied = objects[i]->is_signaled_for(block.thread_id);
    }
    if (satisfied) {
        uint32_t first_abandoned = count;
        for (uint32_t i = 0; i < count; ++i) {
            const AcquireOutcome outcome = objects[i]->acquire(block.thread_id);
            note_acquired(*objects[i], outcome);
            if (outcome.abandoned && first_abandoned == count) {
                first_abandoned = i;
            }
        }
        result = first_abandoned < count ? WAIT_ABANDONED_0 + first_abandoned : WAIT_OBJECT_0;
    } else if (enqueue) {
        for (uint32_t i = 0; i < count; ++i) {
            objects[i]->waiters.push_back({&block, i});
        }
    }
    for (uint32_t k = count; k-- > 0;) {
        objects[lock_order[k]]->lock.unlock();
    }
    return satisfied;
}

// Dedicated single-object wait fast path: avoids constructing the 64-element ObjectArray, avoids
// Clock::now() and WaitBlock setup on immediate satisfaction or zero-timeout polls, and reuses the
// thread-local WaitBlock when blocking.
uint32_t wait_for_single_object(Handle handle, uint32_t timeout_ms) {
    auto object = lookup_dispatcher_object(handle);
    if (!object) {
        SetLastError(ERROR_INVALID_HANDLE);
        return WAIT_FAILED;
    }

    const uint32_t thread_id = current_thread_info().id;

    uint32_t initial_seq = 0;
    {
        AcquireOutcome outcome{};
        bool acquired = false;
        {
            std::lock_guard<std::mutex> guard(object->lock);
            if (object->is_signaled_for(thread_id)) {
                outcome = object->acquire(thread_id);
                acquired = true;
            } else if (timeout_ms == 0) {
                return WAIT_TIMEOUT;
            } else {
                initial_seq = object->signal_seq.load(std::memory_order_relaxed);
            }
        }
        if (acquired) {
            note_acquired(*object, outcome);
            return outcome.abandoned ? WAIT_ABANDONED_0 : WAIT_OBJECT_0;
        }
    }

    const bool infinite = timeout_ms == INFINITE_TIMEOUT;
    const Clock::time_point deadline =
        infinite ? Clock::time_point::max() : Clock::now() + std::chrono::milliseconds(timeout_ms);

    spin_for_single_signal(*object, initial_seq, infinite, deadline);

    ScopedWaitBlock scoped_block(thread_id, false);
    WaitBlock& block = scoped_block.get();
    bool enqueued = false;
    {
        AcquireOutcome outcome{};
        bool acquired = false;
        {
            std::lock_guard<std::mutex> guard(object->lock);
            if (object->is_signaled_for(thread_id)) {
                outcome = object->acquire(thread_id);
                acquired = true;
            } else if (infinite || Clock::now() < deadline) {
                object->waiters.push_back({&block, 0});
                enqueued = true;
            }
        }
        if (acquired) {
            note_acquired(*object, outcome);
            return outcome.abandoned ? WAIT_ABANDONED_0 : WAIT_OBJECT_0;
        }
    }

    if (!enqueued) {
        return WAIT_TIMEOUT;
    }

    {
        std::unique_lock<std::mutex> guard(block.mutex);
        const auto done = [&]() {
            return block.state.load(std::memory_order_acquire) == WaitBlock::kSatisfied;
        };
        if (infinite) {
            block.cv.wait(guard, done);
        } else if (!block.cv.wait_until(guard, deadline, done)) {
            uint32_t expected = WaitBlock::kWaiting;
            if (!block.state.compare_exchange_strong(expected, WaitBlock::kTimedOut,
                                                     std::memory_order_acq_rel)) {
                block.cv.wait(guard, done);  // a handoff won the race; it completes shortly
            }
        }
    }

    if (block.state.load(std::memory_order_acquire) != WaitBlock::kSatisfied) {
        std::lock_guard<std::mutex> guard(object->lock);
        std::erase_if(object->waiters, [&](const WaiterLink& link) { return link.block == &block; });
        return WAIT_TIMEOUT;
    }
    note_acquired(*object, {block.abandoned, block.newly_owned});
    return block.abandoned ? WAIT_ABANDONED_0 : WAIT_OBJECT_0;
}

uint32_t wait_any(const ObjectArray& objects, uint32_t count, uint32_t thread_id,
                  uint32_t timeout_ms, bool infinite, Clock::time_point deadline) {
    ScopedWaitBlock scoped_block(thread_id, false);
    WaitBlock& block = scoped_block.get();
    uint32_t registered = 0;
    bool claimed = try_wait_any(objects, count, block, false, registered);
    if (claimed) {
        note_acquired(*objects[block.index], {block.abandoned, block.newly_owned});
        return (block.abandoned ? WAIT_ABANDONED_0 : WAIT_OBJECT_0) + block.index;
    }
    if (timeout_ms == 0) {
        return WAIT_TIMEOUT;
    }

    spin_for_signal(objects, count, infinite, deadline);
    claimed = try_wait_any(objects, count, block, false, registered);
    if (claimed) {
        note_acquired(*objects[block.index], {block.abandoned, block.newly_owned});
        return (block.abandoned ? WAIT_ABANDONED_0 : WAIT_OBJECT_0) + block.index;
    }
    if (!infinite && Clock::now() >= deadline) {
        return WAIT_TIMEOUT;
    }

    claimed = try_wait_any(objects, count, block, true, registered);
    if (claimed && registered == 0) {
        note_acquired(*objects[block.index], {block.abandoned, block.newly_owned});
        return (block.abandoned ? WAIT_ABANDONED_0 : WAIT_OBJECT_0) + block.index;
    }

    if (claimed || registered != 0) {
        std::unique_lock<std::mutex> guard(block.mutex);
        const auto done = [&]() {
            return block.state.load(std::memory_order_acquire) == WaitBlock::kSatisfied;
        };
        if (infinite) {
            block.cv.wait(guard, done);
        } else if (!block.cv.wait_until(guard, deadline, done)) {
            uint32_t expected = WaitBlock::kWaiting;
            if (!block.state.compare_exchange_strong(expected, WaitBlock::kTimedOut,
                                                     std::memory_order_acq_rel)) {
                block.cv.wait(guard, done);  // a handoff won the race; it completes shortly
            }
        }
    }
    unregister_wait_block(objects, registered, block);

    if (block.state.load(std::memory_order_acquire) != WaitBlock::kSatisfied) {
        return WAIT_TIMEOUT;
    }
    note_acquired(*objects[block.index], {block.abandoned, block.newly_owned});
    return (block.abandoned ? WAIT_ABANDONED_0 : WAIT_OBJECT_0) + block.index;
}

uint32_t wait_all(const ObjectArray& objects, uint32_t count, uint32_t thread_id,
                  uint32_t timeout_ms, bool infinite, Clock::time_point deadline) {
    std::array<uint32_t, MAXIMUM_WAIT_OBJECTS> lock_order{};
    for (uint32_t i = 0; i < count; ++i) {
        lock_order[i] = i;
    }
    std::sort(lock_order.begin(), lock_order.begin() + count, [&](uint32_t a, uint32_t b) {
        return std::less<const DispatcherObject*>()(objects[a].get(), objects[b].get());
    });

    ScopedWaitBlock scoped_block(thread_id, true);
    WaitBlock& block = scoped_block.get();
    bool registered = false;
    bool spun = timeout_ms == 0;  // a poll never spins
    uint32_t result = WAIT_TIMEOUT;
    for (;;) {
        const uint32_t seen = block.wake_seq.load(std::memory_order_acquire);
        const bool enqueue = spun && !registered && timeout_ms != 0;
        if (try_wait_all(objects, lock_order, count, block, enqueue, result)) {
            break;
        }
        if (timeout_ms == 0) {
            result = WAIT_TIMEOUT;
            break;
        }
        registered = registered || enqueue;
        if (!infinite && Clock::now() >= deadline) {
            result = WAIT_TIMEOUT;
            break;
        }
        if (!spun) {
            spun = true;
            spin_for_signal(objects, count, infinite, deadline);
            continue;
        }
        std::unique_lock<std::mutex> guard(block.mutex);
        const auto woken = [&]() { return block.wake_seq.load(std::memory_order_acquire) != seen; };
        if (infinite) {
            block.cv.wait(guard, woken);
        } else {
            block.cv.wait_until(guard, deadline, woken);
        }
    }
    if (registered) {
        unregister_wait_block(objects, count, block);
    }
    return result;
}

uint32_t wait_for_objects(uint32_t count,
                          const Handle* handles,
                          bool wait_all_objects,
                          uint32_t timeout_ms) {
    if (count == 0 || count > MAXIMUM_WAIT_OBJECTS || handles == nullptr) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return WAIT_FAILED;
    }
    if (count == 1) {
        return wait_for_single_object(handles[0], timeout_ms);
    }

    // Handle lookups take the handle-table lock, so resolve them before any object lock.
    ObjectArray objects;
    for (uint32_t i = 0; i < count; ++i) {
        objects[i] = lookup_dispatcher_object(handles[i]);
        if (!objects[i]) {
            SetLastError(ERROR_INVALID_HANDLE);
            return WAIT_FAILED;
        }
    }
    if (wait_all_objects) {
        // Win32 rejects the same object appearing twice in a wait-all set.
        for (uint32_t i = 0; i < count; ++i) {
            for (uint32_t j = i + 1; j < count; ++j) {
                if (objects[i] == objects[j]) {
                    SetLastError(ERROR_INVALID_PARAMETER);
                    return WAIT_FAILED;
                }
            }
        }
    }

    const uint32_t thread_id = current_thread_info().id;
    const bool infinite = timeout_ms == INFINITE_TIMEOUT;
    const Clock::time_point deadline =
        (infinite || timeout_ms == 0) ? Clock::time_point::min()
                                      : Clock::now() + std::chrono::milliseconds(timeout_ms);
    return wait_all_objects ? wait_all(objects, count, thread_id, timeout_ms, infinite, deadline)
                            : wait_any(objects, count, thread_id, timeout_ms, infinite, deadline);
}

// ---------------------------------------------------------------------------
// Condition variables: waiters parked in a process-wide table keyed by the guest address
// ---------------------------------------------------------------------------

struct CvWaiter {
    const void* key = nullptr;
    std::condition_variable cv;
    bool signaled = false;
    CvWaiter* prev = nullptr;
    CvWaiter* next = nullptr;

    void reset(const void* wait_key) {
        key = wait_key;
        signaled = false;
        prev = nullptr;
        next = nullptr;
    }
};

struct ReusableCvWaiterSlot {
    CvWaiter waiter;
    bool in_use = false;
};

thread_local ReusableCvWaiterSlot t_cv_waiter;

class ScopedCvWaiter {
public:
    explicit ScopedCvWaiter(const void* key) {
        if (!t_cv_waiter.in_use) {
            t_cv_waiter.in_use = true;
            waiter_ = &t_cv_waiter.waiter;
            owns_tls_ = true;
        } else {
            fallback_.emplace();
            waiter_ = &*fallback_;
            owns_tls_ = false;
        }
        waiter_->reset(key);
    }

    ~ScopedCvWaiter() {
        if (owns_tls_) {
            t_cv_waiter.in_use = false;
        }
    }

    ScopedCvWaiter(const ScopedCvWaiter&) = delete;
    ScopedCvWaiter& operator=(const ScopedCvWaiter&) = delete;

    CvWaiter& get() { return *waiter_; }

private:
    CvWaiter* waiter_ = nullptr;
    std::optional<CvWaiter> fallback_;
    bool owns_tls_ = false;
};

struct CvBucket {
    std::mutex mutex;
    std::atomic<uint32_t> waiter_count{0};
    CvWaiter* head = nullptr;
    CvWaiter* tail = nullptr;

    void push_back(CvWaiter* waiter) {
        waiter->prev = tail;
        waiter->next = nullptr;
        if (tail != nullptr) {
            tail->next = waiter;
        } else {
            head = waiter;
        }
        tail = waiter;
        waiter_count.fetch_add(1, std::memory_order_seq_cst);
    }

    void unlink(CvWaiter* waiter) {
        if (waiter->prev != nullptr) {
            waiter->prev->next = waiter->next;
        } else {
            head = waiter->next;
        }
        if (waiter->next != nullptr) {
            waiter->next->prev = waiter->prev;
        } else {
            tail = waiter->prev;
        }
        waiter->prev = nullptr;
        waiter->next = nullptr;
        waiter_count.fetch_sub(1, std::memory_order_release);
    }
};

constexpr size_t kCvBucketCount = 64;

CvBucket& cv_bucket(const void* key) {
    static auto* buckets = new std::array<CvBucket, kCvBucketCount>();
    uint64_t h = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(key));
    h ^= h >> 17;
    h *= 0x9E3779B97F4A7C15ull;
    return (*buckets)[static_cast<size_t>(h >> 58) % kCvBucketCount];
}

// The waiter is queued before the caller's lock is released, so a wake issued by a thread that
// acquired that lock afterwards always finds it (no lost wakeups).
template <typename Release, typename Reacquire>
bool sleep_on_condition_variable(const void* key,
                                 uint32_t timeout_ms,
                                 Release release_lock,
                                 Reacquire reacquire_lock) {
    const bool infinite = timeout_ms == INFINITE_TIMEOUT;
    const Clock::time_point deadline =
        (infinite || timeout_ms == 0) ? Clock::time_point::min()
                                      : Clock::now() + std::chrono::milliseconds(timeout_ms);

    CvBucket& bucket = cv_bucket(key);
    ScopedCvWaiter scoped_waiter(key);
    CvWaiter& waiter = scoped_waiter.get();
    {
        std::lock_guard<std::mutex> lock(bucket.mutex);
        bucket.push_back(&waiter);
    }

    release_lock();

    bool signaled = false;
    {
        std::unique_lock<std::mutex> lock(bucket.mutex);
        if (infinite) {
            waiter.cv.wait(lock, [&]() { return waiter.signaled; });
        } else if (timeout_ms != 0) {
            waiter.cv.wait_until(lock, deadline, [&]() { return waiter.signaled; });
        }
        signaled = waiter.signaled;
        if (!signaled) {
            bucket.unlink(&waiter);
        }
    }

    reacquire_lock();
    if (!signaled) {
        SetLastError(ERROR_TIMEOUT);
    }
    return signaled;
}

void wake_condition_variable(const void* key, bool wake_all) {
    CvBucket& bucket = cv_bucket(key);
    // Pairs with push_back's seq_cst increment so a wake issued immediately after updating the
    // predicate and releasing the user lock never misses a waiter queued before lock release.
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (bucket.waiter_count.load(std::memory_order_acquire) == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(bucket.mutex);
    CvWaiter* waiter = bucket.head;
    while (waiter != nullptr) {
        CvWaiter* next = waiter->next;
        if (waiter->key == key) {
            bucket.unlink(waiter);
            waiter->signaled = true;
            // Notify under the bucket lock: the waiter owns this node and may return once unlocked.
            waiter->cv.notify_one();
            if (!wake_all) {
                break;
            }
        }
        waiter = next;
    }
}

// ---------------------------------------------------------------------------
// Threads
// ---------------------------------------------------------------------------

void thread_trampoline(std::shared_ptr<ThreadObject> self) {
    playgta6::os::name_this_thread(L"pg6-guest-thread");
    t_thread.info = self->info;
    {
        std::unique_lock<std::mutex> lock(self->lock);
        self->resume_cv.wait(lock, [&]() { return self->suspend_count == 0; });
    }

    const uint32_t exit_code = self->start(self->parameter);

    // Abandon owned mutexes before the thread handle becomes signaled, so a thread that waited
    // on the handle observes WAIT_ABANDONED on those mutexes.
    abandon_owned_mutexes(t_thread);
    std::lock_guard<std::mutex> lock(self->lock);
    self->exit_code = exit_code;
    self->exited = true;
    self->wake_waiters_locked();
}

// ---------------------------------------------------------------------------
// Thread pool: one bounded process pool on top of the O3 XTaskQueue worker port
// ---------------------------------------------------------------------------

struct ThreadPool {
    std::mutex mutex;
    std::unique_ptr<XTaskQueue> queue;
    uint32_t max_threads = kDefaultThreadPoolThreads;
};

ThreadPool& thread_pool() {
    static ThreadPool pool;
    return pool;
}

std::atomic<uint32_t> g_forced_submit_failures{0};

bool submit_to_pool(std::function<void()> callback) {
    for (uint32_t forced = g_forced_submit_failures.load(std::memory_order_relaxed); forced != 0;) {
        if (g_forced_submit_failures.compare_exchange_weak(forced, forced - 1, std::memory_order_relaxed)) {
            return false;  // test seam: stands in for a pool that refuses the work
        }
    }
    auto& pool = thread_pool();
    std::lock_guard<std::mutex> lock(pool.mutex);
    if (!pool.queue) {
        pool.queue = std::make_unique<XTaskQueue>(XTaskQueue::DispatchMode::ThreadPool,
                                                  XTaskQueue::DispatchMode::Manual,
                                                  pool.max_threads,
                                                  /*work_serialized=*/true);  // fixed, bounded pool
    }
    return pool.queue->submit_work(std::move(callback));
}

void run_work_callback(const std::shared_ptr<Pg6TpWork>& work, uint64_t epoch) {
    {
        std::lock_guard<std::mutex> lock(work->mutex);
        if (epoch != work->epoch) {
            return;  // canceled by WaitForThreadpoolWorkCallbacks(cancel = TRUE)
        }
        --work->queued;
        ++work->running;
    }

    Pg6TpCallbackInstance instance;
    work->callback(&instance, work->context, work.get());

    {
        std::lock_guard<std::mutex> lock(work->mutex);
        --work->running;
    }
    work->idle_cv.notify_all();
}

}  // namespace

// ---------------------------------------------------------------------------
// Runtime configuration
// ---------------------------------------------------------------------------

bool set_process_affinity_mask(uint64_t mask) {
    if (mask == 0) {
        return false;
    }
    g_process_affinity_mask.store(mask, std::memory_order_relaxed);
    return true;
}

uint64_t get_process_affinity_mask() {
    return g_process_affinity_mask.load(std::memory_order_relaxed);
}

bool set_thread_pool_max_threads(uint32_t count) {
    if (count == 0 || count > kMaxThreadPoolThreads) {
        return false;
    }
    auto& pool = thread_pool();
    std::lock_guard<std::mutex> lock(pool.mutex);
    pool.max_threads = count;
    return true;
}

uint32_t thread_pool_max_threads() {
    auto& pool = thread_pool();
    std::lock_guard<std::mutex> lock(pool.mutex);
    return pool.max_threads;
}

void shutdown_thread_pool() {
    std::unique_ptr<XTaskQueue> queue;
    {
        auto& pool = thread_pool();
        std::lock_guard<std::mutex> lock(pool.mutex);
        queue = std::move(pool.queue);
    }
    if (queue) {
        // Workers drain the remaining queue before exiting; terminate joins them.
        queue->terminate(/*wait=*/true, /*cancel_pending=*/false);
    }
}

// ---------------------------------------------------------------------------
// Events (declared in os/win32_kernel.h)
// ---------------------------------------------------------------------------

Handle CreateEventA(void* security_attributes,
                    bool manual_reset,
                    bool initial_state,
                    const char* name) {
    (void)security_attributes;
    return create_event(manual_reset, initial_state, narrow_name(name));
}

Handle CreateEventW(void* security_attributes,
                    bool manual_reset,
                    bool initial_state,
                    const char16_t* name) {
    (void)security_attributes;
    return create_event(manual_reset, initial_state, widen_name(name));
}

bool SetEvent(Handle event_handle) {
    auto event = lookup_as<EventObject>(event_handle, ObjectType::Event);
    if (!event) {
        SetLastError(ERROR_INVALID_HANDLE);
        return false;
    }
    if (event_traced(event_handle)) {
        event_trace_line("set", event_handle, "");
    }
    std::lock_guard<std::mutex> lock(event->lock);
    event->signaled = true;
    event->wake_waiters_locked();
    return true;
}

bool ResetEvent(Handle event_handle) {
    auto event = lookup_as<EventObject>(event_handle, ObjectType::Event);
    if (!event) {
        SetLastError(ERROR_INVALID_HANDLE);
        return false;
    }
    if (event_traced(event_handle)) {
        event_trace_line("reset", event_handle, "");
    }
    std::lock_guard<std::mutex> lock(event->lock);
    event->signaled = false;
    return true;
}

// ---------------------------------------------------------------------------
// Semaphores & mutexes
// ---------------------------------------------------------------------------

Handle CreateSemaphoreA(void* security_attributes,
                        int32_t initial_count,
                        int32_t maximum_count,
                        const char* name) {
    (void)security_attributes;
    return create_semaphore(initial_count, maximum_count, narrow_name(name));
}

Handle CreateSemaphoreW(void* security_attributes,
                        int32_t initial_count,
                        int32_t maximum_count,
                        const char16_t* name) {
    (void)security_attributes;
    return create_semaphore(initial_count, maximum_count, widen_name(name));
}

bool ReleaseSemaphore(Handle semaphore, int32_t release_count, int32_t* previous_count) {
    auto sem = lookup_as<SemaphoreObject>(semaphore, ObjectType::Semaphore);
    if (!sem) {
        SetLastError(ERROR_INVALID_HANDLE);
        return false;
    }
    if (release_count <= 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    std::lock_guard<std::mutex> lock(sem->lock);
    if (release_count > sem->maximum - sem->count) {
        SetLastError(ERROR_TOO_MANY_POSTS);
        return false;
    }
    if (previous_count != nullptr) {
        *previous_count = sem->count;
    }
    sem->count += release_count;
    sem->wake_waiters_locked();
    if (event_traced(semaphore)) {
        char detail[48];
        std::snprintf(detail, sizeof(detail), "+%d -> %d", release_count, sem->count);
        event_trace_line("release", semaphore, detail);
    }
    return true;
}

Handle CreateMutexA(void* security_attributes, bool initial_owner, const char* name) {
    (void)security_attributes;
    return create_mutex(initial_owner, narrow_name(name));
}

Handle CreateMutexW(void* security_attributes, bool initial_owner, const char16_t* name) {
    (void)security_attributes;
    return create_mutex(initial_owner, widen_name(name));
}

bool ReleaseMutex(Handle mutex) {
    auto object = lookup_as<MutexObject>(mutex, ObjectType::Mutex);
    if (!object) {
        SetLastError(ERROR_INVALID_HANDLE);
        return false;
    }
    const uint32_t thread_id = current_thread_info().id;
    std::lock_guard<std::mutex> lock(object->lock);
    if (object->owner != thread_id) {
        SetLastError(ERROR_NOT_OWNER);
        return false;
    }
    if (--object->recursion == 0) {
        object->owner = 0;
        std::erase(t_thread.owned_mutexes, object);
        object->wake_waiters_locked();
    }
    return true;
}

// ---------------------------------------------------------------------------
// Waiting
// ---------------------------------------------------------------------------

uint32_t WaitForSingleObject(Handle handle, uint32_t timeout_ms) {
    const uint32_t result = wait_for_single_object(handle, timeout_ms);
    event_trace_wait(handle, timeout_ms, result);
    return result;
}

uint32_t WaitForSingleObjectEx(Handle handle, uint32_t timeout_ms, bool alertable) {
    (void)alertable;
    const uint32_t result = wait_for_single_object(handle, timeout_ms);
    event_trace_wait(handle, timeout_ms, result);
    return result;
}

uint32_t WaitForMultipleObjects(uint32_t count,
                                const Handle* handles,
                                bool wait_all,
                                uint32_t timeout_ms) {
    return wait_for_objects(count, handles, wait_all, timeout_ms);
}

uint32_t WaitForMultipleObjectsEx(uint32_t count,
                                  const Handle* handles,
                                  bool wait_all,
                                  uint32_t timeout_ms,
                                  bool alertable) {
    (void)alertable;
    return wait_for_objects(count, handles, wait_all, timeout_ms);
}

// ---------------------------------------------------------------------------
// Condition variables
// ---------------------------------------------------------------------------

void InitializeConditionVariable(ConditionVariable* condition_variable) {
    if (condition_variable != nullptr) {
        condition_variable->reserved = 0;
    }
}

bool SleepConditionVariableSRW(ConditionVariable* condition_variable,
                               SrwLock* srw_lock,
                               uint32_t timeout_ms,
                               uint32_t flags) {
    if (condition_variable == nullptr || srw_lock == nullptr ||
        (flags & ~CONDITION_VARIABLE_LOCKMODE_SHARED) != 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    if ((flags & CONDITION_VARIABLE_LOCKMODE_SHARED) != 0) {
        return sleep_on_condition_variable(
            condition_variable, timeout_ms,
            [&]() { ReleaseSRWLockShared(srw_lock); },
            [&]() { AcquireSRWLockShared(srw_lock); });
    }
    return sleep_on_condition_variable(
        condition_variable, timeout_ms,
        [&]() { ReleaseSRWLockExclusive(srw_lock); },
        [&]() { AcquireSRWLockExclusive(srw_lock); });
}

bool SleepConditionVariableCS(ConditionVariable* condition_variable,
                              CriticalSection* critical_section,
                              uint32_t timeout_ms) {
    // The caller must own the section (Windows: undefined otherwise; here an error).
    if (condition_variable == nullptr || critical_section == nullptr ||
        critical_section->recursion_count <= 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    return sleep_on_condition_variable(
        condition_variable, timeout_ms,
        [&]() { LeaveCriticalSection(critical_section); },
        [&]() { EnterCriticalSection(critical_section); });
}

void WakeConditionVariable(ConditionVariable* condition_variable) {
    if (condition_variable != nullptr) {
        wake_condition_variable(condition_variable, false);
    }
}

void WakeAllConditionVariable(ConditionVariable* condition_variable) {
    if (condition_variable != nullptr) {
        wake_condition_variable(condition_variable, true);
    }
}

// ---------------------------------------------------------------------------
// Threads
// ---------------------------------------------------------------------------

Handle CreateThread(void* security_attributes,
                    uint64_t stack_size,
                    Pg6ThreadStartRoutine start_address,
                    void* parameter,
                    uint32_t creation_flags,
                    uint32_t* thread_id) {
    (void)security_attributes;
    (void)stack_size;
    if (start_address == nullptr) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return kNullHandle;
    }

    auto info = std::make_shared<ThreadInfo>(
        allocate_thread_id(), g_process_affinity_mask.load(std::memory_order_relaxed));
    auto thread = std::make_shared<ThreadObject>(
        info, start_address, parameter, (creation_flags & CREATE_SUSPENDED) != 0);
    const Handle handle = insert_dispatcher_object(thread);

    // Published before the thread starts, so the new thread may read it immediately.
    if (thread_id != nullptr) {
        *thread_id = info->id;
    }

    try {
        std::thread(thread_trampoline, std::move(thread)).detach();
    } catch (...) {
        CloseHandle(handle);
        SetLastError(ERROR_NOT_ENOUGH_MEMORY);
        return kNullHandle;
    }
    return handle;
}

uint32_t ResumeThread(Handle thread) {
    auto object = lookup_as<ThreadObject>(thread, ObjectType::Thread);
    if (!object) {
        SetLastError(ERROR_INVALID_HANDLE);
        return RESUME_THREAD_FAILED;
    }
    std::lock_guard<std::mutex> lock(object->lock);
    const uint32_t previous = object->suspend_count;
    if (previous > 0 && --object->suspend_count == 0) {
        object->resume_cv.notify_all();
    }
    return previous;
}

bool GetExitCodeThread(Handle thread, uint32_t* exit_code) {
    auto object = lookup_as<ThreadObject>(thread, ObjectType::Thread);
    if (!object) {
        SetLastError(ERROR_INVALID_HANDLE);
        return false;
    }
    if (exit_code == nullptr) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    std::lock_guard<std::mutex> lock(object->lock);
    *exit_code = object->exited ? object->exit_code : STILL_ACTIVE;
    return true;
}

Handle GetCurrentThread() {
    return kCurrentThreadPseudoHandle;
}

uint32_t GetCurrentThreadId() {
    return current_thread_info().id;
}

uint64_t SetThreadAffinityMask(Handle thread, uint64_t affinity_mask) {
    ThreadInfo* info = nullptr;
    std::shared_ptr<ThreadObject> object;
    if (thread == kCurrentThreadPseudoHandle) {
        info = &current_thread_info();
    } else {
        object = lookup_as<ThreadObject>(thread, ObjectType::Thread);
        if (!object) {
            SetLastError(ERROR_INVALID_HANDLE);
            return 0;
        }
        info = object->info.get();
    }
    if (affinity_mask == 0 || (affinity_mask & ~get_process_affinity_mask()) != 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return 0;
    }
    return info->affinity_mask.exchange(affinity_mask, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Thread pool
// ---------------------------------------------------------------------------

Pg6TpWork* CreateThreadpoolWork(Pg6TpWorkCallback callback, void* context, void* callback_environ) {
    (void)callback_environ;
    if (callback == nullptr) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }
    auto work = std::make_shared<Pg6TpWork>();
    work->callback = callback;
    work->context = context;
    work->self = work;
    return work.get();
}

bool SubmitThreadpoolWork(Pg6TpWork* work) {
    if (work == nullptr) {
        return false;
    }
    std::shared_ptr<Pg6TpWork> keep;
    uint64_t epoch = 0;
    {
        std::lock_guard<std::mutex> lock(work->mutex);
        keep = work->self;
        if (!keep) {
            return false;  // already closed
        }
        ++work->queued;
        epoch = work->epoch;
    }

    if (!submit_to_pool([keep, epoch]() { run_work_callback(keep, epoch); })) {
        {
            std::lock_guard<std::mutex> lock(work->mutex);
            if (epoch == work->epoch) {
                --work->queued;
            }
        }
        work->idle_cv.notify_all();
        return false;
    }
    return true;
}

void fail_next_thread_pool_submissions_for_testing(uint32_t count) {
    g_forced_submit_failures.store(count, std::memory_order_relaxed);
}

void WaitForThreadpoolWorkCallbacks(Pg6TpWork* work, bool cancel_pending_callbacks) {
    if (work == nullptr) {
        return;
    }
    std::unique_lock<std::mutex> lock(work->mutex);
    if (cancel_pending_callbacks) {
        ++work->epoch;  // queued callbacks from older epochs become no-ops
        work->queued = 0;
    }
    work->idle_cv.wait(lock, [&]() { return work->queued == 0 && work->running == 0; });
}

void CloseThreadpoolWork(Pg6TpWork* work) {
    if (work == nullptr) {
        return;
    }
    std::shared_ptr<Pg6TpWork> released;
    {
        std::lock_guard<std::mutex> lock(work->mutex);
        released = std::move(work->self);
    }
    // `released` may be the last reference; it is dropped here, outside the work mutex.
}

bool TrySubmitThreadpoolCallback(Pg6TpSimpleCallback callback,
                                 void* context,
                                 void* callback_environ) {
    (void)callback_environ;
    if (callback == nullptr) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    const bool submitted = submit_to_pool([callback, context]() {
        Pg6TpCallbackInstance instance;
        callback(&instance, context);
    });
    if (!submitted) {
        SetLastError(ERROR_NOT_ENOUGH_MEMORY);
    }
    return submitted;
}

}  // namespace playgta6::os::win32

// ---------------------------------------------------------------------------
// Prefixed C-ABI entrypoints
// ---------------------------------------------------------------------------

extern "C" {

Pg6Handle pg6_win32_create_semaphore_a(void* security_attributes,
                                       int32_t initial_count,
                                       int32_t maximum_count,
                                       const char* name) {
    return playgta6::os::win32::CreateSemaphoreA(
        security_attributes, initial_count, maximum_count, name);
}

Pg6Handle pg6_win32_create_semaphore_w(void* security_attributes,
                                       int32_t initial_count,
                                       int32_t maximum_count,
                                       const char16_t* name) {
    return playgta6::os::win32::CreateSemaphoreW(
        security_attributes, initial_count, maximum_count, name);
}

int32_t pg6_win32_release_semaphore(Pg6Handle semaphore,
                                    int32_t release_count,
                                    int32_t* previous_count) {
    return playgta6::os::win32::ReleaseSemaphore(semaphore, release_count, previous_count) ? 1
                                                                                         : 0;
}

Pg6Handle pg6_win32_create_mutex_a(void* security_attributes,
                                   int32_t initial_owner,
                                   const char* name) {
    return playgta6::os::win32::CreateMutexA(security_attributes, initial_owner != 0, name);
}

Pg6Handle pg6_win32_create_mutex_w(void* security_attributes,
                                   int32_t initial_owner,
                                   const char16_t* name) {
    return playgta6::os::win32::CreateMutexW(security_attributes, initial_owner != 0, name);
}

int32_t pg6_win32_release_mutex(Pg6Handle mutex) {
    return playgta6::os::win32::ReleaseMutex(mutex) ? 1 : 0;
}

uint32_t pg6_win32_wait_for_single_object_ex(Pg6Handle handle,
                                             uint32_t timeout_ms,
                                             int32_t alertable) {
    const playgta6::os::ThreadSpan frame_span(timeout_ms != 0 ? playgta6::os::SpanKind::Wait : playgta6::os::SpanKind::Count);
    return playgta6::os::win32::WaitForSingleObjectEx(handle, timeout_ms, alertable != 0);
}

uint32_t pg6_win32_wait_for_multiple_objects(uint32_t count,
                                             const Pg6Handle* handles,
                                             int32_t wait_all,
                                             uint32_t timeout_ms) {
    const playgta6::os::ThreadSpan frame_span(timeout_ms != 0 ? playgta6::os::SpanKind::Wait : playgta6::os::SpanKind::Count);
    return playgta6::os::win32::WaitForMultipleObjects(count, handles, wait_all != 0, timeout_ms);
}

uint32_t pg6_win32_wait_for_multiple_objects_ex(uint32_t count,
                                                const Pg6Handle* handles,
                                                int32_t wait_all,
                                                uint32_t timeout_ms,
                                                int32_t alertable) {
    const playgta6::os::ThreadSpan frame_span(timeout_ms != 0 ? playgta6::os::SpanKind::Wait : playgta6::os::SpanKind::Count);
    return playgta6::os::win32::WaitForMultipleObjectsEx(
        count, handles, wait_all != 0, timeout_ms, alertable != 0);
}

void pg6_win32_initialize_condition_variable(Pg6ConditionVariable* condition_variable) {
    playgta6::os::win32::InitializeConditionVariable(condition_variable);
}

int32_t pg6_win32_sleep_condition_variable_srw(Pg6ConditionVariable* condition_variable,
                                               Pg6SrwLock* srw_lock,
                                               uint32_t timeout_ms,
                                               uint32_t flags) {
    const playgta6::os::ThreadSpan frame_span(playgta6::os::SpanKind::CondVar);
    return playgta6::os::win32::SleepConditionVariableSRW(
               condition_variable, srw_lock, timeout_ms, flags)
               ? 1
               : 0;
}

int32_t pg6_win32_sleep_condition_variable_cs(Pg6ConditionVariable* condition_variable,
                                              Pg6CriticalSection* critical_section,
                                              uint32_t timeout_ms) {
    const playgta6::os::ThreadSpan frame_span(playgta6::os::SpanKind::CondVar);
    return playgta6::os::win32::SleepConditionVariableCS(
               condition_variable, critical_section, timeout_ms)
               ? 1
               : 0;
}

void pg6_win32_wake_condition_variable(Pg6ConditionVariable* condition_variable) {
    playgta6::os::win32::WakeConditionVariable(condition_variable);
}

void pg6_win32_wake_all_condition_variable(Pg6ConditionVariable* condition_variable) {
    playgta6::os::win32::WakeAllConditionVariable(condition_variable);
}

Pg6Handle pg6_win32_create_thread(void* security_attributes,
                                  uint64_t stack_size,
                                  Pg6ThreadStartRoutine start_address,
                                  void* parameter,
                                  uint32_t creation_flags,
                                  uint32_t* thread_id) {
    return playgta6::os::win32::CreateThread(
        security_attributes, stack_size, start_address, parameter, creation_flags, thread_id);
}

uint32_t pg6_win32_resume_thread(Pg6Handle thread) {
    return playgta6::os::win32::ResumeThread(thread);
}

int32_t pg6_win32_get_exit_code_thread(Pg6Handle thread, uint32_t* exit_code) {
    return playgta6::os::win32::GetExitCodeThread(thread, exit_code) ? 1 : 0;
}

Pg6Handle pg6_win32_get_current_thread(void) {
    return playgta6::os::win32::GetCurrentThread();
}

uint32_t pg6_win32_get_current_thread_id(void) {
    return playgta6::os::win32::GetCurrentThreadId();
}

uint64_t pg6_win32_set_thread_affinity_mask(Pg6Handle thread, uint64_t affinity_mask) {
    return playgta6::os::win32::SetThreadAffinityMask(thread, affinity_mask);
}

Pg6TpWork* pg6_win32_create_threadpool_work(Pg6TpWorkCallback callback,
                                            void* context,
                                            void* callback_environ) {
    return playgta6::os::win32::CreateThreadpoolWork(callback, context, callback_environ);
}

void pg6_win32_submit_threadpool_work(Pg6TpWork* work) {
    playgta6::os::win32::SubmitThreadpoolWork(work);
}

void pg6_win32_wait_for_threadpool_work_callbacks(Pg6TpWork* work,
                                                  int32_t cancel_pending_callbacks) {
    playgta6::os::win32::WaitForThreadpoolWorkCallbacks(work, cancel_pending_callbacks != 0);
}

void pg6_win32_close_threadpool_work(Pg6TpWork* work) {
    playgta6::os::win32::CloseThreadpoolWork(work);
}

int32_t pg6_win32_try_submit_threadpool_callback(Pg6TpSimpleCallback callback,
                                                 void* context,
                                                 void* callback_environ) {
    return playgta6::os::win32::TrySubmitThreadpoolCallback(callback, context, callback_environ)
               ? 1
               : 0;
}

}  // extern "C"
