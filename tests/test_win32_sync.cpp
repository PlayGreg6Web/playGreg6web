#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "os/win32_compat.h"
#include "os/win32_kernel.h"
#include "os/win32_sync.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_win32_sync.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6::os::win32;

// Upper bound for waits that must succeed; only reached when something is broken.
constexpr uint32_t kLongWait = 5000;
// Short window used to assert that something does NOT happen.
constexpr uint32_t kNegativeWait = 25;

static_assert(sizeof(Pg6ConditionVariable) == 8, "CONDITION_VARIABLE must stay pointer-sized");
static_assert(sizeof(Pg6CriticalSection) <= 40, "CRITICAL_SECTION guest storage is 40 bytes");

bool valid(Handle h) {
    return h != kNullHandle && h != INVALID_HANDLE_VALUE;
}

// ---------------------------------------------------------------------------
// 1. Semaphores
// ---------------------------------------------------------------------------

void test_semaphores() {
    // Parameter validation.
    CHECK(CreateSemaphoreW(nullptr, 2, 1, nullptr) == kNullHandle);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    CHECK(CreateSemaphoreW(nullptr, 0, 0, nullptr) == kNullHandle);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_create_semaphore_a(nullptr, -1, 4, nullptr) == kNullHandle);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);

    // Counting, maximum and previous-count reporting.
    {
        const Handle sem = CreateSemaphoreW(nullptr, 1, 3, nullptr);
        CHECK(valid(sem));
        CHECK(GetLastError() == ERROR_SUCCESS);
        CHECK(WaitForSingleObject(sem, 0) == WAIT_OBJECT_0);
        CHECK(WaitForSingleObject(sem, 0) == WAIT_TIMEOUT);

        int32_t prev = -7;
        CHECK(ReleaseSemaphore(sem, 2, &prev));
        CHECK(prev == 0);
        prev = -7;
        CHECK(!ReleaseSemaphore(sem, 2, &prev));
        CHECK(GetLastError() == ERROR_TOO_MANY_POSTS);
        CHECK(prev == -7);
        CHECK(pg6_win32_release_semaphore(sem, 1, &prev) == 1);
        CHECK(prev == 2);
        CHECK(!ReleaseSemaphore(sem, 1, nullptr));
        CHECK(GetLastError() == ERROR_TOO_MANY_POSTS);
        CHECK(!ReleaseSemaphore(sem, 0, nullptr));
        CHECK(GetLastError() == ERROR_INVALID_PARAMETER);

        for (int i = 0; i < 3; ++i) {
            CHECK(WaitForSingleObject(sem, 0) == WAIT_OBJECT_0);
        }
        CHECK(pg6_win32_wait_for_single_object_ex(sem, 0, 1) == WAIT_TIMEOUT);
        CHECK(CloseHandle(sem));
    }

    // Producer / consumers: every released count is consumed exactly once.
    {
        constexpr int kConsumers = 4;
        constexpr int kItems = 1000;
        const Handle sem = CreateSemaphoreA(nullptr, 0, kItems, nullptr);
        CHECK(valid(sem));

        std::atomic<int> consumed{0};
        std::atomic<int> wait_errors{0};
        std::vector<std::thread> consumers;
        for (int c = 0; c < kConsumers; ++c) {
            consumers.emplace_back([&]() {
                for (int i = 0; i < kItems / kConsumers; ++i) {
                    if (WaitForSingleObject(sem, kLongWait) != WAIT_OBJECT_0) {
                        wait_errors.fetch_add(1);
                        return;
                    }
                    consumed.fetch_add(1);
                }
            });
        }
        for (int i = 0; i < kItems; ++i) {
            CHECK(ReleaseSemaphore(sem, 1, nullptr));
        }
        for (auto& t : consumers) {
            t.join();
        }
        CHECK(wait_errors.load() == 0);
        CHECK(consumed.load() == kItems);
        CHECK(WaitForSingleObject(sem, 0) == WAIT_TIMEOUT);
        CHECK(CloseHandle(sem));
    }

    // Named semaphores share one object; the name namespace is shared with events.
    {
        const Handle a = CreateSemaphoreA(nullptr, 0, 5, "pg6_sync_test_sem");
        CHECK(valid(a));
        CHECK(GetLastError() == ERROR_SUCCESS);
        const Handle b = pg6_win32_create_semaphore_w(nullptr, 3, 5, u"pg6_sync_test_sem");
        CHECK(valid(b) && b != a);
        CHECK(GetLastError() == ERROR_ALREADY_EXISTS);
        CHECK(WaitForSingleObject(a, 0) == WAIT_TIMEOUT);  // initial count of the alias is ignored
        CHECK(ReleaseSemaphore(b, 1, nullptr));
        CHECK(WaitForSingleObject(a, 0) == WAIT_OBJECT_0);

        CHECK(CreateEventA(nullptr, false, false, "pg6_sync_test_sem") == kNullHandle);
        CHECK(GetLastError() == ERROR_INVALID_HANDLE);

        // Wrong-type operations are rejected.
        CHECK(!SetEvent(a));
        CHECK(GetLastError() == ERROR_INVALID_HANDLE);
        CHECK(!ReleaseMutex(a));
        CHECK(GetLastError() == ERROR_INVALID_HANDLE);

        CHECK(CloseHandle(a));
        CHECK(CloseHandle(b));

        // Once every handle is closed the name can be reused by another object type.
        const Handle ev = CreateEventA(nullptr, false, false, "pg6_sync_test_sem");
        CHECK(valid(ev));
        CHECK(GetLastError() == ERROR_SUCCESS);
        CHECK(!ReleaseSemaphore(ev, 1, nullptr));
        CHECK(GetLastError() == ERROR_INVALID_HANDLE);
        CHECK(CloseHandle(ev));
    }
}

// ---------------------------------------------------------------------------
// 2. Mutexes: recursion, ownership, contention, abandonment
// ---------------------------------------------------------------------------

uint32_t lock_and_exit_routine(void* param) {
    const Handle mutex = *static_cast<const Handle*>(param);
    return WaitForSingleObject(mutex, kLongWait) == WAIT_OBJECT_0 ? 7u : 99u;
}

void test_mutexes() {
    // Recursion and ownership.
    {
        const Handle m = CreateMutexW(nullptr, true, nullptr);
        CHECK(valid(m));
        CHECK(WaitForSingleObject(m, 0) == WAIT_OBJECT_0);
        CHECK(WaitForSingleObject(m, 0) == WAIT_OBJECT_0);  // recursion depth 3

        std::atomic<uint32_t> other_wait{0};
        std::atomic<bool> other_release{true};
        std::atomic<uint32_t> other_error{0};
        std::thread probe([&]() {
            other_wait.store(WaitForSingleObject(m, 0));
            other_release.store(ReleaseMutex(m));
            other_error.store(GetLastError());
        });
        probe.join();
        CHECK(other_wait.load() == WAIT_TIMEOUT);
        CHECK(!other_release.load());
        CHECK(other_error.load() == ERROR_NOT_OWNER);

        CHECK(ReleaseMutex(m));
        CHECK(ReleaseMutex(m));
        CHECK(pg6_win32_release_mutex(m) == 1);
        CHECK(!ReleaseMutex(m));
        CHECK(GetLastError() == ERROR_NOT_OWNER);

        std::thread taker([&]() {
            other_wait.store(WaitForSingleObject(m, 0));
            other_release.store(ReleaseMutex(m));
        });
        taker.join();
        CHECK(other_wait.load() == WAIT_OBJECT_0);
        CHECK(other_release.load());
        CHECK(CloseHandle(m));
    }

    // Contention on a shared counter.
    {
        const Handle m = pg6_win32_create_mutex_a(nullptr, 0, nullptr);
        CHECK(valid(m));
        constexpr int kThreads = 4;
        constexpr int kIterations = 2000;
        int counter = 0;
        std::atomic<int> errors{0};
        std::vector<std::thread> threads;
        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&]() {
                for (int i = 0; i < kIterations; ++i) {
                    if (WaitForSingleObject(m, kLongWait) != WAIT_OBJECT_0) {
                        errors.fetch_add(1);
                        return;
                    }
                    ++counter;
                    if (!ReleaseMutex(m)) {
                        errors.fetch_add(1);
                    }
                }
            });
        }
        for (auto& t : threads) {
            t.join();
        }
        CHECK(errors.load() == 0);
        CHECK(counter == kThreads * kIterations);
        CHECK(CloseHandle(m));
    }

    // Abandonment by a CreateThread thread: observable as soon as its handle is signaled.
    {
        Handle m = CreateMutexA(nullptr, false, nullptr);
        CHECK(valid(m));
        const Handle th = CreateThread(nullptr, 0, lock_and_exit_routine, &m, 0, nullptr);
        CHECK(valid(th));
        CHECK(WaitForSingleObject(th, kLongWait) == WAIT_OBJECT_0);
        uint32_t code = 0;
        CHECK(GetExitCodeThread(th, &code));
        CHECK(code == 7u);

        CHECK(WaitForSingleObject(m, 0) == WAIT_ABANDONED_0);
        CHECK(ReleaseMutex(m));
        CHECK(WaitForSingleObject(m, 0) == WAIT_OBJECT_0);  // abandonment is reported once
        CHECK(ReleaseMutex(m));
        CHECK(CloseHandle(th));
        CHECK(CloseHandle(m));
    }

    // Abandonment by a host thread (thread-exit path), reported through wait-any.
    {
        const Handle ev = CreateEventA(nullptr, true, false, nullptr);
        const Handle m = CreateMutexA(nullptr, false, nullptr);
        std::atomic<uint32_t> got{0};
        std::thread owner([&]() { got.store(WaitForSingleObject(m, 0)); });
        owner.join();
        CHECK(got.load() == WAIT_OBJECT_0);

        const Handle set[2] = {ev, m};
        CHECK(WaitForMultipleObjects(2, set, false, 0) == WAIT_ABANDONED_0 + 1);
        CHECK(ReleaseMutex(m));
        CHECK(CloseHandle(ev));
        CHECK(CloseHandle(m));
    }

    // Opening an existing named mutex ignores bInitialOwner.
    {
        const Handle m = CreateMutexA(nullptr, false, "pg6_sync_test_mutex");
        CHECK(valid(m));
        std::atomic<uint32_t> open_error{0};
        std::atomic<bool> open_ok{false};
        std::thread opener([&]() {
            const Handle alias = CreateMutexW(nullptr, true, u"pg6_sync_test_mutex");
            open_error.store(GetLastError());
            open_ok.store(valid(alias) && CloseHandle(alias));
        });
        opener.join();
        CHECK(open_ok.load());
        CHECK(open_error.load() == ERROR_ALREADY_EXISTS);
        CHECK(WaitForSingleObject(m, 0) == WAIT_OBJECT_0);  // neither owned nor abandoned
        CHECK(ReleaseMutex(m));
        CHECK(CloseHandle(m));
    }
}

// ---------------------------------------------------------------------------
// 3. Events on the shared dispatcher & timeouts
// ---------------------------------------------------------------------------

void test_events_and_timeouts() {
    // Auto-reset: each SetEvent releases exactly one waiter.
    {
        constexpr int kWaiters = 4;
        const Handle ev = CreateEventW(nullptr, false, false, nullptr);
        const Handle done = CreateSemaphoreA(nullptr, 0, kWaiters, nullptr);
        std::atomic<int> woken{0};
        std::atomic<int> errors{0};
        std::vector<std::thread> waiters;
        for (int i = 0; i < kWaiters; ++i) {
            waiters.emplace_back([&]() {
                if (WaitForSingleObject(ev, kLongWait) != WAIT_OBJECT_0) {
                    errors.fetch_add(1);
                }
                woken.fetch_add(1);
                ReleaseSemaphore(done, 1, nullptr);
            });
        }
        for (int round = 1; round <= kWaiters; ++round) {
            CHECK(SetEvent(ev));
            CHECK(WaitForSingleObject(done, kLongWait) == WAIT_OBJECT_0);
            if (round < kWaiters) {
                CHECK(WaitForSingleObject(done, kNegativeWait) == WAIT_TIMEOUT);
            }
            CHECK(woken.load() == round);
        }
        for (auto& t : waiters) {
            t.join();
        }
        CHECK(errors.load() == 0);
        CHECK(WaitForSingleObject(ev, 0) == WAIT_TIMEOUT);
        CHECK(CloseHandle(ev));
        CHECK(CloseHandle(done));
    }

    // Manual-reset: wakes every waiter, stays signaled until ResetEvent.
    {
        constexpr int kWaiters = 3;
        const Handle ev = CreateEventA(nullptr, true, false, nullptr);
        std::atomic<int> woken{0};
        std::vector<std::thread> waiters;
        for (int i = 0; i < kWaiters; ++i) {
            waiters.emplace_back([&]() {
                if (WaitForSingleObject(ev, kLongWait) == WAIT_OBJECT_0) {
                    woken.fetch_add(1);
                }
            });
        }
        CHECK(SetEvent(ev));
        for (auto& t : waiters) {
            t.join();
        }
        CHECK(woken.load() == kWaiters);
        CHECK(WaitForSingleObject(ev, 0) == WAIT_OBJECT_0);
        CHECK(ResetEvent(ev));
        CHECK(WaitForSingleObject(ev, 0) == WAIT_TIMEOUT);
        CHECK(CloseHandle(ev));
    }

    // Finite timeouts use the monotonic clock and never return early.
    {
        const Handle ev = CreateEventA(nullptr, false, false, nullptr);
        const auto start = std::chrono::steady_clock::now();
        CHECK(WaitForSingleObject(ev, 50) == WAIT_TIMEOUT);
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - start)
                                 .count();
        CHECK(elapsed >= 50);
        CHECK(elapsed < 3000);

        const Handle set[1] = {ev};
        const auto start2 = std::chrono::steady_clock::now();
        CHECK(WaitForMultipleObjectsEx(1, set, true, 30, true) == WAIT_TIMEOUT);
        CHECK(std::chrono::steady_clock::now() - start2 >= std::chrono::milliseconds(30));
        CHECK(CloseHandle(ev));
    }
}

// ---------------------------------------------------------------------------
// 4. WaitForMultipleObjects
// ---------------------------------------------------------------------------

uint32_t count_and_return_routine(void* param) {
    auto* counter = static_cast<std::atomic<uint32_t>*>(param);
    return counter->fetch_add(1) + 100u;
}

void test_wait_multiple() {
    const Handle e0 = CreateEventA(nullptr, true, false, nullptr);
    const Handle e1 = CreateEventA(nullptr, true, false, nullptr);
    const Handle sem = CreateSemaphoreA(nullptr, 0, 10, nullptr);
    CHECK(valid(e0) && valid(e1) && valid(sem));

    // Parameter and handle validation.
    {
        Handle set[MAXIMUM_WAIT_OBJECTS + 1];
        for (auto& h : set) {
            h = e0;
        }
        CHECK(WaitForMultipleObjects(0, set, false, 0) == WAIT_FAILED);
        CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
        CHECK(WaitForMultipleObjects(MAXIMUM_WAIT_OBJECTS + 1, set, false, 0) == WAIT_FAILED);
        CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
        CHECK(WaitForMultipleObjects(1, nullptr, false, 0) == WAIT_FAILED);
        CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
        // Duplicates are fine for wait-any, rejected for wait-all.
        CHECK(WaitForMultipleObjects(MAXIMUM_WAIT_OBJECTS, set, false, 0) == WAIT_TIMEOUT);
        CHECK(WaitForMultipleObjects(2, set, true, 0) == WAIT_FAILED);
        CHECK(GetLastError() == ERROR_INVALID_PARAMETER);

        const Handle closed = CreateEventA(nullptr, false, false, nullptr);
        CHECK(CloseHandle(closed));
        const Handle with_bad[3] = {e0, closed, e1};
        CHECK(WaitForMultipleObjects(3, with_bad, false, 0) == WAIT_FAILED);
        CHECK(GetLastError() == ERROR_INVALID_HANDLE);
    }

    // Wait-any: lowest signaled index wins; consumes only that object.
    {
        const Handle set[3] = {e0, e1, sem};
        CHECK(WaitForMultipleObjects(3, set, false, 0) == WAIT_TIMEOUT);
        CHECK(SetEvent(e1));
        CHECK(WaitForMultipleObjects(3, set, false, 0) == WAIT_OBJECT_0 + 1);
        CHECK(ResetEvent(e1));
        CHECK(ReleaseSemaphore(sem, 1, nullptr));
        CHECK(pg6_win32_wait_for_multiple_objects(3, set, 0, 0) == WAIT_OBJECT_0 + 2);
        CHECK(WaitForMultipleObjects(3, set, false, 0) == WAIT_TIMEOUT);
        CHECK(SetEvent(e0));
        CHECK(SetEvent(e1));
        CHECK(WaitForMultipleObjects(3, set, false, 0) == WAIT_OBJECT_0);
        CHECK(ResetEvent(e0));
        CHECK(ResetEvent(e1));
    }

    // Wait-all never acquires a subset.
    {
        const Handle a = CreateEventA(nullptr, false, false, nullptr);
        const Handle b = CreateEventA(nullptr, false, false, nullptr);
        const Handle set[2] = {a, b};
        CHECK(SetEvent(a));
        CHECK(WaitForMultipleObjects(2, set, true, 0) == WAIT_TIMEOUT);
        CHECK(WaitForMultipleObjects(2, set, true, kNegativeWait) == WAIT_TIMEOUT);
        CHECK(WaitForSingleObject(a, 0) == WAIT_OBJECT_0);  // still signaled: not consumed
        CHECK(SetEvent(a));
        CHECK(SetEvent(b));
        CHECK(pg6_win32_wait_for_multiple_objects_ex(2, set, 1, 0, 0) == WAIT_OBJECT_0);
        CHECK(WaitForMultipleObjects(2, set, false, 0) == WAIT_TIMEOUT);  // both consumed
        CHECK(CloseHandle(a));
        CHECK(CloseHandle(b));
    }

    // Wait-all across a semaphore and a mutex held by another thread.
    {
        const Handle s1 = CreateSemaphoreA(nullptr, 1, 1, nullptr);
        const Handle m = CreateMutexA(nullptr, false, nullptr);
        const Handle held = CreateEventA(nullptr, true, false, nullptr);
        const Handle release = CreateEventA(nullptr, true, false, nullptr);
        std::atomic<bool> owner_ok{true};
        std::thread owner([&]() {
            if (WaitForSingleObject(m, kLongWait) != WAIT_OBJECT_0) {
                owner_ok.store(false);
            }
            SetEvent(held);
            if (WaitForSingleObject(release, kLongWait) != WAIT_OBJECT_0 || !ReleaseMutex(m)) {
                owner_ok.store(false);
            }
        });
        CHECK(WaitForSingleObject(held, kLongWait) == WAIT_OBJECT_0);

        const Handle set[2] = {s1, m};
        CHECK(WaitForMultipleObjects(2, set, true, kNegativeWait) == WAIT_TIMEOUT);
        // Semaphore was not decremented by the failed wait-all: it is still at its maximum.
        CHECK(!ReleaseSemaphore(s1, 1, nullptr));
        CHECK(GetLastError() == ERROR_TOO_MANY_POSTS);

        CHECK(SetEvent(release));
        CHECK(WaitForMultipleObjects(2, set, true, kLongWait) == WAIT_OBJECT_0);
        owner.join();
        CHECK(owner_ok.load());
        CHECK(WaitForSingleObject(s1, 0) == WAIT_TIMEOUT);
        CHECK(ReleaseMutex(m));
        for (Handle h : {s1, m, held, release}) {
            CHECK(CloseHandle(h));
        }
    }

    // A blocked wait-all is released only once every object is signaled.
    {
        const Handle x = CreateEventA(nullptr, false, false, nullptr);
        const Handle y = CreateEventA(nullptr, false, false, nullptr);
        const Handle done = CreateEventA(nullptr, true, false, nullptr);
        std::atomic<uint32_t> result{WAIT_FAILED};
        std::thread waiter([&]() {
            const Handle set[2] = {x, y};
            result.store(WaitForMultipleObjects(2, set, true, kLongWait));
            SetEvent(done);
        });
        CHECK(SetEvent(x));
        CHECK(WaitForSingleObject(done, kNegativeWait) == WAIT_TIMEOUT);
        CHECK(SetEvent(y));
        CHECK(WaitForSingleObject(done, kLongWait) == WAIT_OBJECT_0);
        waiter.join();
        CHECK(result.load() == WAIT_OBJECT_0);
        const Handle set[2] = {x, y};
        CHECK(WaitForMultipleObjects(2, set, false, 0) == WAIT_TIMEOUT);
        for (Handle h : {x, y, done}) {
            CHECK(CloseHandle(h));
        }
    }

    // Thread handles in a wait-all set.
    {
        constexpr int kThreads = 6;
        std::atomic<uint32_t> counter{0};
        Handle threads[kThreads];
        for (auto& h : threads) {
            h = CreateThread(nullptr, 0, count_and_return_routine, &counter, 0, nullptr);
            CHECK(valid(h));
        }
        CHECK(WaitForMultipleObjects(kThreads, threads, true, kLongWait) == WAIT_OBJECT_0);
        CHECK(counter.load() == kThreads);
        std::set<uint32_t> codes;
        for (Handle h : threads) {
            uint32_t code = 0;
            CHECK(GetExitCodeThread(h, &code));
            codes.insert(code);
            CHECK(CloseHandle(h));
        }
        CHECK(codes.size() == static_cast<size_t>(kThreads));
        CHECK(*codes.begin() == 100u);
    }

    for (Handle h : {e0, e1, sem}) {
        CHECK(CloseHandle(h));
    }
}

// ---------------------------------------------------------------------------
// 5. Condition variables
// ---------------------------------------------------------------------------

void test_condition_variables() {
    // Timeout: returns FALSE / ERROR_TIMEOUT with the lock re-acquired.
    {
        ConditionVariable cv{};
        InitializeConditionVariable(&cv);
        SrwLock srw{};
        InitializeSRWLock(&srw);

        AcquireSRWLockExclusive(&srw);
        CHECK(!SleepConditionVariableSRW(&cv, &srw, 20, 0));
        CHECK(GetLastError() == ERROR_TIMEOUT);
        std::atomic<bool> other_got_lock{true};
        std::thread probe([&]() { other_got_lock.store(TryAcquireSRWLockExclusive(&srw)); });
        probe.join();
        CHECK(!other_got_lock.load());
        CHECK(pg6_win32_sleep_condition_variable_srw(&cv, &srw, 0, 0) == 0);
        CHECK(GetLastError() == ERROR_TIMEOUT);
        ReleaseSRWLockExclusive(&srw);

        CHECK(!SleepConditionVariableSRW(nullptr, &srw, 0, 0));
        CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
        CHECK(!SleepConditionVariableSRW(&cv, &srw, 0, 0x2u));
        CHECK(GetLastError() == ERROR_INVALID_PARAMETER);

        // Waking with no waiters is a no-op and does not leave a pending signal.
        WakeConditionVariable(&cv);
        pg6_win32_wake_all_condition_variable(&cv);
        AcquireSRWLockExclusive(&srw);
        CHECK(!SleepConditionVariableSRW(&cv, &srw, 0, 0));
        ReleaseSRWLockExclusive(&srw);
        CHECK(cv.reserved == 0);  // guest storage is never written
    }

    // WakeConditionVariable releases exactly one waiter; WakeAll releases the rest.
    {
        constexpr int kWaiters = 4;
        ConditionVariable cv{};
        SrwLock srw{};
        int waiting = 0;  // guarded by srw
        std::atomic<int> returned{0};
        std::atomic<int> errors{0};
        const Handle done = CreateSemaphoreA(nullptr, 0, kWaiters, nullptr);

        std::vector<std::thread> waiters;
        for (int i = 0; i < kWaiters; ++i) {
            waiters.emplace_back([&]() {
                AcquireSRWLockExclusive(&srw);
                ++waiting;
                if (!SleepConditionVariableSRW(&cv, &srw, INFINITE_TIMEOUT, 0)) {
                    errors.fetch_add(1);
                }
                returned.fetch_add(1);
                ReleaseSRWLockExclusive(&srw);
                ReleaseSemaphore(done, 1, nullptr);
            });
        }

        // Every waiter increments `waiting` and queues itself before releasing the lock.
        const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (true) {
            AcquireSRWLockExclusive(&srw);
            const int n = waiting;
            ReleaseSRWLockExclusive(&srw);
            if (n == kWaiters || std::chrono::steady_clock::now() > give_up) {
                break;
            }
            std::this_thread::yield();
        }

        pg6_win32_wake_condition_variable(&cv);
        CHECK(WaitForSingleObject(done, kLongWait) == WAIT_OBJECT_0);
        CHECK(WaitForSingleObject(done, kNegativeWait) == WAIT_TIMEOUT);
        CHECK(returned.load() == 1);

        WakeAllConditionVariable(&cv);
        for (int i = 1; i < kWaiters; ++i) {
            CHECK(WaitForSingleObject(done, kLongWait) == WAIT_OBJECT_0);
        }
        for (auto& t : waiters) {
            t.join();
        }
        CHECK(errors.load() == 0);
        CHECK(returned.load() == kWaiters);
        CHECK(CloseHandle(done));
    }

    // Shared-mode sleepers re-acquire the SRW lock in shared mode.
    {
        constexpr int kReaders = 3;
        ConditionVariable cv{};
        SrwLock srw{};
        bool go = false;  // guarded by srw
        std::atomic<int> resumed{0};
        std::atomic<int> errors{0};
        std::vector<std::thread> readers;
        for (int i = 0; i < kReaders; ++i) {
            readers.emplace_back([&]() {
                AcquireSRWLockShared(&srw);
                while (!go) {
                    if (!SleepConditionVariableSRW(&cv, &srw, kLongWait,
                                                   CONDITION_VARIABLE_LOCKMODE_SHARED)) {
                        errors.fetch_add(1);
                        break;
                    }
                }
                // Holding the lock shared: another shared acquire succeeds, exclusive fails.
                if (TryAcquireSRWLockExclusive(&srw)) {
                    errors.fetch_add(1);
                }
                resumed.fetch_add(1);
                ReleaseSRWLockShared(&srw);
            });
        }
        AcquireSRWLockExclusive(&srw);
        go = true;
        ReleaseSRWLockExclusive(&srw);
        WakeAllConditionVariable(&cv);
        for (auto& t : readers) {
            t.join();
        }
        CHECK(errors.load() == 0);
        CHECK(resumed.load() == kReaders);
    }

    // Bounded producer / consumer queue on an SRW lock with two condition variables.
    {
        constexpr int kProducers = 2;
        constexpr int kConsumers = 2;
        constexpr int kPerProducer = 2000;
        constexpr size_t kCapacity = 8;
        ConditionVariable not_full{};
        ConditionVariable not_empty{};
        SrwLock srw{};
        std::deque<int> queue;
        int produced_done = 0;
        long long consumed_sum = 0;
        int consumed_count = 0;

        std::vector<std::thread> threads;
        for (int p = 0; p < kProducers; ++p) {
            threads.emplace_back([&]() {
                for (int i = 1; i <= kPerProducer; ++i) {
                    AcquireSRWLockExclusive(&srw);
                    while (queue.size() >= kCapacity) {
                        SleepConditionVariableSRW(&not_full, &srw, INFINITE_TIMEOUT, 0);
                    }
                    queue.push_back(i);
                    ReleaseSRWLockExclusive(&srw);
                    WakeConditionVariable(&not_empty);
                }
                AcquireSRWLockExclusive(&srw);
                ++produced_done;
                ReleaseSRWLockExclusive(&srw);
                WakeAllConditionVariable(&not_empty);
            });
        }
        for (int c = 0; c < kConsumers; ++c) {
            threads.emplace_back([&]() {
                while (true) {
                    AcquireSRWLockExclusive(&srw);
                    while (queue.empty() && produced_done < kProducers) {
                        SleepConditionVariableSRW(&not_empty, &srw, INFINITE_TIMEOUT, 0);
                    }
                    if (queue.empty()) {
                        ReleaseSRWLockExclusive(&srw);
                        return;
                    }
                    consumed_sum += queue.front();
                    ++consumed_count;
                    queue.pop_front();
                    ReleaseSRWLockExclusive(&srw);
                    WakeConditionVariable(&not_full);
                }
            });
        }
        for (auto& t : threads) {
            t.join();
        }
        constexpr long long kExpectedSum =
            static_cast<long long>(kProducers) * kPerProducer * (kPerProducer + 1) / 2;
        CHECK(consumed_count == kProducers * kPerProducer);
        CHECK(consumed_sum == kExpectedSum);
    }

    // Critical-section variant.
    {
        ConditionVariable cv{};
        CriticalSection cs{};
        InitializeCriticalSection(&cs);
        std::deque<int> queue;
        constexpr int kItems = 2000;
        long long sum = 0;

        std::thread consumer([&]() {
            for (int received = 0; received < kItems; ++received) {
                EnterCriticalSection(&cs);
                while (queue.empty()) {
                    SleepConditionVariableCS(&cv, &cs, INFINITE_TIMEOUT);
                }
                sum += queue.front();
                queue.pop_front();
                LeaveCriticalSection(&cs);
            }
        });
        for (int i = 1; i <= kItems; ++i) {
            EnterCriticalSection(&cs);
            queue.push_back(i);
            WakeConditionVariable(&cv);  // waking while holding the lock is also valid
            LeaveCriticalSection(&cs);
        }
        consumer.join();
        CHECK(sum == static_cast<long long>(kItems) * (kItems + 1) / 2);

        EnterCriticalSection(&cs);
        CHECK(pg6_win32_sleep_condition_variable_cs(&cv, &cs, 10) == 0);
        CHECK(GetLastError() == ERROR_TIMEOUT);
        CHECK(TryEnterCriticalSection(&cs));  // re-entered by the sleeper
        LeaveCriticalSection(&cs);
        LeaveCriticalSection(&cs);
        DeleteCriticalSection(&cs);

        CHECK(!SleepConditionVariableCS(&cv, &cs, 0));  // section not entered
        CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    }
}

// ---------------------------------------------------------------------------
// 6. Threads
// ---------------------------------------------------------------------------

struct ThreadProbe {
    std::atomic<uint32_t> observed_id{0};
    std::atomic<uint64_t> affinity_seen{0};
    std::atomic<int> ran{0};
    Handle gate = kNullHandle;
    Handle done = kNullHandle;
};

uint32_t probe_routine(void* param) {
    auto* probe = static_cast<ThreadProbe*>(param);
    probe->observed_id.store(GetCurrentThreadId());
    probe->affinity_seen.store(SetThreadAffinityMask(GetCurrentThread(), 0x4));
    probe->ran.fetch_add(1);
    return 42;
}

uint32_t gated_routine(void* param) {
    auto* probe = static_cast<ThreadProbe*>(param);
    const uint32_t r = WaitForSingleObject(probe->gate, kLongWait);
    probe->ran.fetch_add(1);
    SetEvent(probe->done);
    return r;
}

void test_threads() {
    CHECK(CreateThread(nullptr, 0, nullptr, nullptr, 0, nullptr) == kNullHandle);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);

    const uint32_t main_id = GetCurrentThreadId();
    CHECK(main_id != 0 && (main_id % 4) == 0);
    CHECK(pg6_win32_get_current_thread_id() == main_id);
    CHECK(pg6_win32_get_current_thread() == kCurrentThreadPseudoHandle);
    CHECK(GetCurrentThread() == kCurrentThreadPseudoHandle);

    // Basic thread: id reporting, exit code, waitable handle.
    {
        ThreadProbe probe;
        uint32_t tid = 0;
        const Handle th = pg6_win32_create_thread(nullptr, 64 * 1024, probe_routine, &probe, 0, &tid);
        CHECK(valid(th));
        CHECK(tid != 0 && (tid % 4) == 0 && tid != main_id);
        CHECK(WaitForSingleObject(th, kLongWait) == WAIT_OBJECT_0);
        CHECK(probe.observed_id.load() == tid);
        CHECK(probe.affinity_seen.load() == get_process_affinity_mask());
        uint32_t code = 0;
        CHECK(pg6_win32_get_exit_code_thread(th, &code) == 1);
        CHECK(code == 42u);
        CHECK(!GetExitCodeThread(th, nullptr));
        CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
        CHECK(CloseHandle(th));
    }

    // CREATE_SUSPENDED: nothing runs until ResumeThread; affinity set from outside is visible inside.
    {
        ThreadProbe probe;
        const Handle th = CreateThread(nullptr, 0, probe_routine, &probe, CREATE_SUSPENDED, nullptr);
        CHECK(valid(th));
        CHECK(WaitForSingleObject(th, kNegativeWait) == WAIT_TIMEOUT);
        uint32_t code = 0;
        CHECK(GetExitCodeThread(th, &code));
        CHECK(code == STILL_ACTIVE);
        CHECK(probe.ran.load() == 0);

        CHECK(SetThreadAffinityMask(th, 0x2) == get_process_affinity_mask());
        CHECK(pg6_win32_resume_thread(th) == 1u);
        CHECK(WaitForSingleObject(th, kLongWait) == WAIT_OBJECT_0);
        CHECK(probe.ran.load() == 1);
        CHECK(probe.affinity_seen.load() == 0x2u);
        CHECK(SetThreadAffinityMask(th, 0x1) == 0x4u);  // value the thread set for itself
        CHECK(ResumeThread(th) == 0u);
        CHECK(CloseHandle(th));
    }

    // Closing the handle does not stop the thread.
    {
        ThreadProbe probe;
        probe.gate = CreateEventA(nullptr, true, false, nullptr);
        probe.done = CreateEventA(nullptr, true, false, nullptr);
        const Handle th = CreateThread(nullptr, 0, gated_routine, &probe, 0, nullptr);
        CHECK(valid(th));
        CHECK(CloseHandle(th));
        CHECK(WaitForSingleObject(th, 0) == WAIT_FAILED);
        CHECK(SetEvent(probe.gate));
        CHECK(WaitForSingleObject(probe.done, kLongWait) == WAIT_OBJECT_0);
        CHECK(probe.ran.load() == 1);
        CHECK(CloseHandle(probe.gate));
        CHECK(CloseHandle(probe.done));
    }

    // Affinity validation for the current thread.
    {
        const uint64_t process_mask = get_process_affinity_mask();
        CHECK(process_mask == kDefaultProcessAffinityMask);
        CHECK(SetThreadAffinityMask(GetCurrentThread(), 0x1) == process_mask);
        CHECK(pg6_win32_set_thread_affinity_mask(GetCurrentThread(), 0x3) == 0x1u);
        CHECK(SetThreadAffinityMask(GetCurrentThread(), 0) == 0);
        CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
        CHECK(SetThreadAffinityMask(GetCurrentThread(), uint64_t{1} << 40) == 0);
        CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
        CHECK(SetThreadAffinityMask(0x7777777, 0x1) == 0);
        CHECK(GetLastError() == ERROR_INVALID_HANDLE);

        CHECK(!set_process_affinity_mask(0));
        CHECK(set_process_affinity_mask(0xFF));
        CHECK(SetThreadAffinityMask(GetCurrentThread(), 0x100) == 0);
        CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
        CHECK(set_process_affinity_mask(kDefaultProcessAffinityMask));
        CHECK(SetThreadAffinityMask(GetCurrentThread(), process_mask) == 0x3u);

        // Per-thread state: other threads still see the default.
        std::atomic<uint64_t> other_prev{0};
        std::thread other([&]() {
            other_prev.store(SetThreadAffinityMask(GetCurrentThread(), 0x8));
        });
        other.join();
        CHECK(other_prev.load() == process_mask);
    }

    // Thread ids are unique and stable per thread.
    {
        constexpr int kThreads = 8;
        std::vector<std::atomic<uint32_t>> ids(kThreads);
        std::atomic<int> unstable{0};
        std::vector<std::thread> threads;
        for (int i = 0; i < kThreads; ++i) {
            threads.emplace_back([&, i]() {
                const uint32_t id = GetCurrentThreadId();
                ids[static_cast<size_t>(i)].store(id);
                if (GetCurrentThreadId() != id) {
                    unstable.fetch_add(1);
                }
            });
        }
        for (auto& t : threads) {
            t.join();
        }
        std::set<uint32_t> unique{main_id};
        for (auto& id : ids) {
            CHECK(id.load() != 0);
            unique.insert(id.load());
        }
        CHECK(unique.size() == static_cast<size_t>(kThreads + 1));
        CHECK(unstable.load() == 0);
        CHECK(GetCurrentThreadId() == main_id);
    }

    // Invalid handles and wrong object types.
    {
        CHECK(ResumeThread(0x7777777) == RESUME_THREAD_FAILED);
        CHECK(GetLastError() == ERROR_INVALID_HANDLE);
        const Handle ev = CreateEventA(nullptr, false, false, nullptr);
        uint32_t code = 0;
        CHECK(!GetExitCodeThread(ev, &code));
        CHECK(GetLastError() == ERROR_INVALID_HANDLE);
        CHECK(SetThreadAffinityMask(ev, 0x1) == 0);
        CHECK(GetLastError() == ERROR_INVALID_HANDLE);
        CHECK(CloseHandle(ev));
    }
}

// ---------------------------------------------------------------------------
// 7. Thread pool
// ---------------------------------------------------------------------------

struct PoolProbe {
    Pg6TpWork* expected_work = nullptr;
    std::atomic<int> count{0};
    std::atomic<int> concurrent{0};
    std::atomic<int> max_concurrent{0};
    std::atomic<int> wrong_args{0};
    std::mutex ids_mutex;
    std::set<uint32_t> thread_ids;
    Handle gate = kNullHandle;
    Handle started = kNullHandle;
    Handle done = kNullHandle;
};

void counting_work(Pg6TpCallbackInstance* instance, void* context, Pg6TpWork* work) {
    auto* probe = static_cast<PoolProbe*>(context);
    if (instance == nullptr || work != probe->expected_work) {
        probe->wrong_args.fetch_add(1);
    }
    const int now = probe->concurrent.fetch_add(1) + 1;
    int seen = probe->max_concurrent.load();
    while (now > seen && !probe->max_concurrent.compare_exchange_weak(seen, now)) {
    }
    {
        std::lock_guard<std::mutex> lock(probe->ids_mutex);
        probe->thread_ids.insert(GetCurrentThreadId());
    }
    std::this_thread::yield();
    probe->concurrent.fetch_sub(1);
    probe->count.fetch_add(1);
}

void blocking_work(Pg6TpCallbackInstance*, void* context, Pg6TpWork*) {
    auto* probe = static_cast<PoolProbe*>(context);
    SetEvent(probe->started);
    WaitForSingleObject(probe->gate, kLongWait);
}

void signaling_work(Pg6TpCallbackInstance*, void* context, Pg6TpWork*) {
    auto* probe = static_cast<PoolProbe*>(context);
    probe->count.fetch_add(1);
    ReleaseSemaphore(probe->done, 1, nullptr);
}

void simple_callback(Pg6TpCallbackInstance* instance, void* context) {
    auto* probe = static_cast<PoolProbe*>(context);
    if (instance == nullptr) {
        probe->wrong_args.fetch_add(1);
    }
    probe->count.fetch_add(1);
    if (probe->done != kNullHandle) {
        ReleaseSemaphore(probe->done, 1, nullptr);
    }
}

void test_thread_pool() {
    CHECK(thread_pool_max_threads() == kDefaultThreadPoolThreads);
    CHECK(!set_thread_pool_max_threads(0));
    CHECK(!set_thread_pool_max_threads(kMaxThreadPoolThreads + 1));
    shutdown_thread_pool();
    CHECK(set_thread_pool_max_threads(3));
    CHECK(thread_pool_max_threads() == 3);

    CHECK(CreateThreadpoolWork(nullptr, nullptr, nullptr) == nullptr);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    CHECK(!TrySubmitThreadpoolCallback(nullptr, nullptr, nullptr));
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);

    // N submissions of one work object run N times on at most 3 workers.
    {
        PoolProbe probe;
        Pg6TpWork* work = pg6_win32_create_threadpool_work(counting_work, &probe, nullptr);
        CHECK(work != nullptr);
        probe.expected_work = work;
        constexpr int kSubmissions = 200;
        for (int i = 0; i < kSubmissions; ++i) {
            pg6_win32_submit_threadpool_work(work);
        }
        pg6_win32_wait_for_threadpool_work_callbacks(work, 0);
        CHECK(probe.count.load() == kSubmissions);
        CHECK(probe.wrong_args.load() == 0);
        CHECK(probe.max_concurrent.load() >= 1 && probe.max_concurrent.load() <= 3);
        CHECK(!probe.thread_ids.empty() && probe.thread_ids.size() <= 3);
        CHECK(probe.thread_ids.count(GetCurrentThreadId()) == 0);

        // Waiting on an idle work object returns immediately.
        WaitForThreadpoolWorkCallbacks(work, false);
        pg6_win32_close_threadpool_work(work);
    }

    // Cancel pending callbacks queued behind a busy single worker.
    shutdown_thread_pool();
    CHECK(set_thread_pool_max_threads(1));
    {
        PoolProbe blocker_probe;
        blocker_probe.gate = CreateEventA(nullptr, true, false, nullptr);
        blocker_probe.started = CreateEventA(nullptr, true, false, nullptr);
        PoolProbe counted;

        Pg6TpWork* blocker = CreateThreadpoolWork(blocking_work, &blocker_probe, nullptr);
        Pg6TpWork* work = CreateThreadpoolWork(counting_work, &counted, nullptr);
        counted.expected_work = work;
        SubmitThreadpoolWork(blocker);
        CHECK(WaitForSingleObject(blocker_probe.started, kLongWait) == WAIT_OBJECT_0);
        for (int i = 0; i < 5; ++i) {
            SubmitThreadpoolWork(work);
        }
        WaitForThreadpoolWorkCallbacks(work, true);  // nothing of `work` is running yet
        CHECK(counted.count.load() == 0);

        CHECK(SetEvent(blocker_probe.gate));
        WaitForThreadpoolWorkCallbacks(blocker, false);
        SubmitThreadpoolWork(work);
        WaitForThreadpoolWorkCallbacks(work, false);
        CHECK(counted.count.load() == 1);  // canceled callbacks never ran

        CloseThreadpoolWork(blocker);
        CloseThreadpoolWork(work);
        CHECK(CloseHandle(blocker_probe.gate));
        CHECK(CloseHandle(blocker_probe.started));
    }
    shutdown_thread_pool();
    CHECK(set_thread_pool_max_threads(kDefaultThreadPoolThreads));

    // Closing a work object with outstanding callbacks lets them finish.
    {
        constexpr int kSubmissions = 10;
        PoolProbe probe;
        probe.done = CreateSemaphoreA(nullptr, 0, kSubmissions, nullptr);
        Pg6TpWork* work = CreateThreadpoolWork(signaling_work, &probe, nullptr);
        for (int i = 0; i < kSubmissions; ++i) {
            SubmitThreadpoolWork(work);
        }
        CloseThreadpoolWork(work);
        for (int i = 0; i < kSubmissions; ++i) {
            CHECK(WaitForSingleObject(probe.done, kLongWait) == WAIT_OBJECT_0);
        }
        CHECK(probe.count.load() == kSubmissions);
        CHECK(CloseHandle(probe.done));
    }

    // TrySubmitThreadpoolCallback.
    {
        constexpr int kCallbacks = 50;
        PoolProbe probe;
        probe.done = CreateSemaphoreA(nullptr, 0, kCallbacks, nullptr);
        for (int i = 0; i < kCallbacks; ++i) {
            CHECK(pg6_win32_try_submit_threadpool_callback(simple_callback, &probe, nullptr) == 1);
        }
        for (int i = 0; i < kCallbacks; ++i) {
            CHECK(WaitForSingleObject(probe.done, kLongWait) == WAIT_OBJECT_0);
        }
        CHECK(probe.count.load() == kCallbacks);
        CHECK(probe.wrong_args.load() == 0);
        CHECK(CloseHandle(probe.done));
    }

    // Shutdown drains queued callbacks before joining the workers.
    {
        PoolProbe probe;
        for (int i = 0; i < 20; ++i) {
            CHECK(TrySubmitThreadpoolCallback(simple_callback, &probe, nullptr));
        }
        shutdown_thread_pool();
        CHECK(probe.count.load() == 20);
    }
}

// ---------------------------------------------------------------------------
// 8. Handle error paths shared with os/win32_kernel
// ---------------------------------------------------------------------------

void test_handle_errors() {
    CHECK(WaitForSingleObject(kNullHandle, 0) == WAIT_FAILED);
    CHECK(GetLastError() == ERROR_INVALID_HANDLE);
    CHECK(WaitForSingleObjectEx(INVALID_HANDLE_VALUE, 0, false) == WAIT_FAILED);
    CHECK(GetLastError() == ERROR_INVALID_HANDLE);

    const Handle sem = CreateSemaphoreA(nullptr, 0, 1, nullptr);
    const Handle m = CreateMutexA(nullptr, false, nullptr);
    CHECK(CloseHandle(sem));
    CHECK(!CloseHandle(sem));
    CHECK(GetLastError() == ERROR_INVALID_HANDLE);
    CHECK(!ReleaseSemaphore(sem, 1, nullptr));
    CHECK(GetLastError() == ERROR_INVALID_HANDLE);
    CHECK(WaitForSingleObject(sem, 0) == WAIT_FAILED);
    CHECK(!ResetEvent(m));
    CHECK(GetLastError() == ERROR_INVALID_HANDLE);
    CHECK(pg6_win32_close_handle(m) == 1);
    CHECK(!ReleaseMutex(m));
    CHECK(GetLastError() == ERROR_INVALID_HANDLE);
}

}  // namespace

// Critical sections live in guest memory the runtime must not interpret: uninitialized bytes,
// re-initialization after Delete, and zero-initialized sections entered without an init call.
void test_critical_section_guest_storage() {
    alignas(8) unsigned char storage[40];
    auto* cs = reinterpret_cast<CriticalSection*>(storage);

    // Garbage that looks like a pointer must not be dereferenced or freed.
    std::memset(storage, 0xCD, sizeof(storage));
    InitializeCriticalSection(cs);
    CHECK(cs->lock_count == -1 && cs->recursion_count == 0 && cs->owning_thread == 0);
    CHECK(cs->debug_info == ~uint64_t{0});
    EnterCriticalSection(cs);
    EnterCriticalSection(cs);
    CHECK(cs->recursion_count == 2 && cs->owning_thread != 0);
    LeaveCriticalSection(cs);
    CHECK(cs->recursion_count == 1 && cs->owning_thread != 0);
    LeaveCriticalSection(cs);
    CHECK(cs->recursion_count == 0 && cs->owning_thread == 0);
    DeleteCriticalSection(cs);

    // Delete, then Initialize again (also over garbage), via the spin-count variant.
    std::memset(storage, 0xAB, sizeof(storage));
    CHECK(pg6_win32_initialize_critical_section_and_spin_count(cs, 4000) == 1);
    CHECK(cs->spin_count == 4000 && cs->recursion_count == 0);
    CHECK(TryEnterCriticalSection(cs));
    LeaveCriticalSection(cs);
    DeleteCriticalSection(cs);
    InitializeCriticalSection(cs);
    EnterCriticalSection(cs);
    LeaveCriticalSection(cs);
    DeleteCriticalSection(cs);

    // A statically zero-initialized section entered without Initialize works and excludes.
    std::memset(storage, 0, sizeof(storage));
    std::atomic<int> inside{0};
    std::atomic<int> overlaps{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < 500; ++i) {
                EnterCriticalSection(cs);
                if (inside.fetch_add(1) != 0) {
                    overlaps.fetch_add(1);
                }
                inside.fetch_sub(1);
                LeaveCriticalSection(cs);
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    CHECK(overlaps.load() == 0);
    CHECK(cs->recursion_count == 0 && cs->owning_thread == 0);
    DeleteCriticalSection(cs);

    // Leaving a section that was never entered or initialized reports an error, no crash.
    std::memset(storage, 0x11, sizeof(storage));
    SetLastError(0);
    LeaveCriticalSection(cs);
    CHECK(GetLastError() == ERROR_INVALID_HANDLE);
}

int main() {
    test_critical_section_guest_storage();
    test_semaphores();
    test_mutexes();
    test_events_and_timeouts();
    test_wait_multiple();
    test_condition_variables();
    test_threads();
    test_thread_pool();
    test_handle_errors();
    shutdown_thread_pool();

    if (g_failures != 0) {
        std::fprintf(stderr, "test_win32_sync failed with %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("All playgta6_win32_sync_tests passed.\n");
    return 0;
}
