#include "os/win32_sync.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

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

// Process-wide dispatcher lock guarding the state of every dispatcher object and its waiter
// list. Leaked on purpose so thread-exit paths that run during static destruction can use it.
std::mutex& dispatcher_mutex() {
    static auto* mutex = new std::mutex();
    return *mutex;
}

// One per blocked WaitFor* call; registered on every object it waits for.
struct WaitBlock {
    std::condition_variable cv;
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

// Per-OS-thread state. owned_mutexes is only touched by its own thread, under the dispatcher lock.
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

}  // namespace

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

    // Both are called with the dispatcher lock held, on the waiting thread.
    virtual bool is_signaled_for(uint32_t thread_id) const = 0;
    // Applies the side effect of a satisfied wait. Returns true if an abandoned mutex was acquired.
    virtual bool acquire(uint32_t thread_id) = 0;

    void wake_waiters_locked() {
        for (WaitBlock* block : waiters) {
            block->cv.notify_one();
        }
    }

    const Type type;
    std::vector<WaitBlock*> waiters;
};

namespace {

using ObjectType = DispatcherObject::Type;

class EventObject final : public DispatcherObject {
public:
    EventObject(bool manual, bool initial)
        : DispatcherObject(ObjectType::Event), manual_reset(manual), signaled(initial) {}

    bool is_signaled_for(uint32_t) const override { return signaled; }
    bool acquire(uint32_t) override {
        if (!manual_reset) {
            signaled = false;
        }
        return false;
    }

    const bool manual_reset;
    bool signaled;
};

class SemaphoreObject final : public DispatcherObject {
public:
    SemaphoreObject(int32_t initial, int32_t maximum_count)
        : DispatcherObject(ObjectType::Semaphore), count(initial), maximum(maximum_count) {}

    bool is_signaled_for(uint32_t) const override { return count > 0; }
    bool acquire(uint32_t) override {
        --count;
        return false;
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

    bool acquire(uint32_t thread_id) override {
        if (owner == thread_id) {
            ++recursion;
            return false;
        }
        owner = thread_id;
        recursion = 1;
        t_thread.owned_mutexes.push_back(shared_from_this());
        const bool was_abandoned = abandoned;
        abandoned = false;
        return was_abandoned;
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
    bool acquire(uint32_t) override { return false; }

    const std::shared_ptr<ThreadInfo> info;
    const Pg6ThreadStartRoutine start;
    void* const parameter;
    bool exited = false;
    uint32_t exit_code = STILL_ACTIVE;
    uint32_t suspend_count;
    std::condition_variable resume_cv;  // waited on with the dispatcher lock
};

void abandon_owned_mutexes_locked(ThreadState& state) {
    for (auto& mutex : state.owned_mutexes) {
        mutex->owner = 0;
        mutex->recursion = 0;
        mutex->abandoned = true;
        mutex->wake_waiters_locked();
    }
    state.owned_mutexes.clear();
}

// Threads not created through CreateThread (host threads, pool workers) abandon their mutexes here.
ThreadState::~ThreadState() {
    if (owned_mutexes.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(dispatcher_mutex());
    abandon_owned_mutexes_locked(*this);
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

// Lock order: named-object mutex -> handle table mutex / dispatcher lock. Never the reverse.
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
            std::lock_guard<std::mutex> lock(dispatcher_mutex());
            mutex->acquire(thread_id);
        }
        return mutex;
    });
}

// ---------------------------------------------------------------------------
// Waiting
// ---------------------------------------------------------------------------

using ObjectArray = std::array<std::shared_ptr<DispatcherObject>, MAXIMUM_WAIT_OBJECTS>;

bool try_satisfy_locked(ObjectArray& objects,
                        uint32_t count,
                        bool wait_all,
                        uint32_t thread_id,
                        uint32_t& result) {
    if (!wait_all) {
        // Wait-any reports the lowest satisfied index.
        for (uint32_t i = 0; i < count; ++i) {
            if (objects[i]->is_signaled_for(thread_id)) {
                const bool abandoned = objects[i]->acquire(thread_id);
                result = (abandoned ? WAIT_ABANDONED_0 : WAIT_OBJECT_0) + i;
                return true;
            }
        }
        return false;
    }

    // Wait-all: acquire nothing unless every object can be acquired right now.
    for (uint32_t i = 0; i < count; ++i) {
        if (!objects[i]->is_signaled_for(thread_id)) {
            return false;
        }
    }
    uint32_t first_abandoned = count;
    for (uint32_t i = 0; i < count; ++i) {
        if (objects[i]->acquire(thread_id) && first_abandoned == count) {
            first_abandoned = i;
        }
    }
    result = first_abandoned < count ? WAIT_ABANDONED_0 + first_abandoned : WAIT_OBJECT_0;
    return true;
}

uint32_t wait_for_objects(uint32_t count,
                          const Handle* handles,
                          bool wait_all,
                          uint32_t timeout_ms) {
    if (count == 0 || count > MAXIMUM_WAIT_OBJECTS || handles == nullptr) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return WAIT_FAILED;
    }

    // Handle lookups take the handle-table lock, so resolve them before the dispatcher lock.
    ObjectArray objects;
    for (uint32_t i = 0; i < count; ++i) {
        objects[i] = lookup_dispatcher_object(handles[i]);
        if (!objects[i]) {
            SetLastError(ERROR_INVALID_HANDLE);
            return WAIT_FAILED;
        }
    }
    if (wait_all) {
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
        infinite ? Clock::time_point::max() : Clock::now() + std::chrono::milliseconds(timeout_ms);

    WaitBlock block;
    bool registered = false;
    uint32_t result = WAIT_TIMEOUT;
    {
        std::unique_lock<std::mutex> lock(dispatcher_mutex());
        while (!try_satisfy_locked(objects, count, wait_all, thread_id, result)) {
            if (!infinite && Clock::now() >= deadline) {
                result = WAIT_TIMEOUT;
                break;
            }
            if (!registered) {
                for (uint32_t i = 0; i < count; ++i) {
                    objects[i]->waiters.push_back(&block);
                }
                registered = true;
            }
            if (infinite) {
                block.cv.wait(lock);
            } else {
                block.cv.wait_until(lock, deadline);
            }
        }
        if (registered) {
            for (uint32_t i = 0; i < count; ++i) {
                std::erase(objects[i]->waiters, &block);
            }
        }
    }
    return result;
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
};

struct CvBucket {
    std::mutex mutex;
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
        infinite ? Clock::time_point::max() : Clock::now() + std::chrono::milliseconds(timeout_ms);

    CvBucket& bucket = cv_bucket(key);
    CvWaiter waiter;
    waiter.key = key;
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
        } else {
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
    t_thread.info = self->info;
    {
        std::unique_lock<std::mutex> lock(dispatcher_mutex());
        self->resume_cv.wait(lock, [&]() { return self->suspend_count == 0; });
    }

    const uint32_t exit_code = self->start(self->parameter);

    // Abandon owned mutexes before the thread handle becomes signaled, so a thread that waited
    // on the handle observes WAIT_ABANDONED on those mutexes.
    std::lock_guard<std::mutex> lock(dispatcher_mutex());
    abandon_owned_mutexes_locked(t_thread);
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

bool submit_to_pool(std::function<void()> callback) {
    auto& pool = thread_pool();
    std::lock_guard<std::mutex> lock(pool.mutex);
    if (!pool.queue) {
        pool.queue = std::make_unique<XTaskQueue>(XTaskQueue::DispatchMode::ThreadPool,
                                                  XTaskQueue::DispatchMode::Manual,
                                                  pool.max_threads);
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
    std::lock_guard<std::mutex> lock(dispatcher_mutex());
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
    std::lock_guard<std::mutex> lock(dispatcher_mutex());
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
    std::lock_guard<std::mutex> lock(dispatcher_mutex());
    if (release_count > sem->maximum - sem->count) {
        SetLastError(ERROR_TOO_MANY_POSTS);
        return false;
    }
    if (previous_count != nullptr) {
        *previous_count = sem->count;
    }
    sem->count += release_count;
    sem->wake_waiters_locked();
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
    std::lock_guard<std::mutex> lock(dispatcher_mutex());
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
    return wait_for_objects(1, &handle, false, timeout_ms);
}

uint32_t WaitForSingleObjectEx(Handle handle, uint32_t timeout_ms, bool alertable) {
    (void)alertable;
    return wait_for_objects(1, &handle, false, timeout_ms);
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
    if (condition_variable == nullptr || critical_section == nullptr ||
        !critical_section->mutex) {
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
    std::lock_guard<std::mutex> lock(dispatcher_mutex());
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
    std::lock_guard<std::mutex> lock(dispatcher_mutex());
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

void SubmitThreadpoolWork(Pg6TpWork* work) {
    if (work == nullptr) {
        return;
    }
    std::shared_ptr<Pg6TpWork> keep;
    uint64_t epoch = 0;
    {
        std::lock_guard<std::mutex> lock(work->mutex);
        keep = work->self;
        if (!keep) {
            return;  // already closed
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
    }
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
    return playgta6::os::win32::WaitForSingleObjectEx(handle, timeout_ms, alertable != 0);
}

uint32_t pg6_win32_wait_for_multiple_objects(uint32_t count,
                                             const Pg6Handle* handles,
                                             int32_t wait_all,
                                             uint32_t timeout_ms) {
    return playgta6::os::win32::WaitForMultipleObjects(count, handles, wait_all != 0, timeout_ms);
}

uint32_t pg6_win32_wait_for_multiple_objects_ex(uint32_t count,
                                                const Pg6Handle* handles,
                                                int32_t wait_all,
                                                uint32_t timeout_ms,
                                                int32_t alertable) {
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
    return playgta6::os::win32::SleepConditionVariableSRW(
               condition_variable, srw_lock, timeout_ms, flags)
               ? 1
               : 0;
}

int32_t pg6_win32_sleep_condition_variable_cs(Pg6ConditionVariable* condition_variable,
                                              Pg6CriticalSection* critical_section,
                                              uint32_t timeout_ms) {
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
