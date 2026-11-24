#pragma once

// Win32 synchronization & thread pool shims (task O6).
//
// Every waitable kernel object (events, semaphores, mutexes, threads) is a "dispatcher object"
// implemented here and stored in the single HANDLE table owned by os/win32_kernel.cpp, so
// CloseHandle / WaitForSingleObject / WaitForMultipleObjects work on any of them. Object state is guarded by one process-wide dispatcher lock, which makes
// WaitForMultipleObjects(bWaitAll = TRUE) acquire all objects atomically.
//
// Blocking: all waits block the calling pthread. The runtime is built with
// -sENVIRONMENT=worker, so guest code never runs on the browser main thread; these shims must
// not be called from the page's main thread. CreateThread and the thread pool consume
// Emscripten pthread workers, so the pool size (-sPTHREAD_POOL_SIZE) must cover the guest's
// threads plus thread_pool_max_threads() pool workers.
//
// Guest-layout structs: CONDITION_VARIABLE is pointer-sized in the guest. The shim never
// stores state inside it; waiters are parked in a process-wide table keyed by its address.

#include <cstddef>
#include <cstdint>

#include "os/win32_kernel.h"

// CONDITION_VARIABLE (zero-initialized == CONDITION_VARIABLE_INIT). Must match guest storage.
struct Pg6ConditionVariable {
    uint64_t reserved = 0;
};
static_assert(sizeof(Pg6ConditionVariable) == 8, "CONDITION_VARIABLE is pointer-sized on x64");

// Opaque thread pool objects (PTP_WORK / PTP_CALLBACK_INSTANCE); allocated by the runtime.
struct Pg6TpWork;
struct Pg6TpCallbackInstance;

// LPTHREAD_START_ROUTINE: DWORD (*)(LPVOID).
using Pg6ThreadStartRoutine = uint32_t (*)(void* parameter);
// PTP_WORK_CALLBACK: VOID (*)(PTP_CALLBACK_INSTANCE, PVOID, PTP_WORK).
using Pg6TpWorkCallback = void (*)(Pg6TpCallbackInstance* instance, void* context, Pg6TpWork* work);
// PTP_SIMPLE_CALLBACK: VOID (*)(PTP_CALLBACK_INSTANCE, PVOID).
using Pg6TpSimpleCallback = void (*)(Pg6TpCallbackInstance* instance, void* context);

namespace playgta6::os::win32 {

using ConditionVariable = Pg6ConditionVariable;

// Wait results & limits.
inline constexpr uint32_t WAIT_ABANDONED_0 = 0x00000080u;
inline constexpr uint32_t MAXIMUM_WAIT_OBJECTS = 64u;

// Error codes specific to this module.
inline constexpr uint32_t ERROR_NOT_OWNER = 288u;
inline constexpr uint32_t ERROR_TOO_MANY_POSTS = 298u;
inline constexpr uint32_t ERROR_TIMEOUT = 1460u;

// Threads.
inline constexpr uint32_t CREATE_SUSPENDED = 0x00000004u;
inline constexpr uint32_t STILL_ACTIVE = 259u;
inline constexpr uint32_t RESUME_THREAD_FAILED = 0xFFFFFFFFu;
// GetCurrentThread() pseudo-handle ((HANDLE)-2); never stored in the handle table.
inline constexpr Handle kCurrentThreadPseudoHandle = ~uint64_t{1};

// SleepConditionVariableSRW flags.
inline constexpr uint32_t CONDITION_VARIABLE_LOCKMODE_SHARED = 0x1u;

// ---------------------------------------------------------------------------
// Runtime configuration (not Win32 APIs)
// ---------------------------------------------------------------------------

inline constexpr uint64_t kDefaultProcessAffinityMask = 0xFFFFull;  // 16 logical processors
inline constexpr uint32_t kDefaultThreadPoolThreads = 2;
inline constexpr uint32_t kMaxThreadPoolThreads = 64;

// Process affinity mask that SetThreadAffinityMask validates against (mask must be non-zero).
// Browsers cannot pin threads, so affinity masks are recorded but have no scheduling effect.
bool set_process_affinity_mask(uint64_t mask);
uint64_t get_process_affinity_mask();

// Bounded worker count for the Vista thread pool (1..kMaxThreadPoolThreads). Applies the next
// time the pool starts (lazily on first submission, or after shutdown_thread_pool()).
bool set_thread_pool_max_threads(uint32_t count);
uint32_t thread_pool_max_threads();

// Runs every callback already queued, then joins the pool workers. Later submissions start a
// fresh pool. Must not be called from a pool callback.
void shutdown_thread_pool();

// ---------------------------------------------------------------------------
// Semaphores & mutexes (dispatcher objects in the shared HANDLE table)
// ---------------------------------------------------------------------------

Handle CreateSemaphoreA(void* security_attributes,
                        int32_t initial_count,
                        int32_t maximum_count,
                        const char* name);
Handle CreateSemaphoreW(void* security_attributes,
                        int32_t initial_count,
                        int32_t maximum_count,
                        const char16_t* name);
bool ReleaseSemaphore(Handle semaphore, int32_t release_count, int32_t* previous_count);

Handle CreateMutexA(void* security_attributes, bool initial_owner, const char* name);
Handle CreateMutexW(void* security_attributes, bool initial_owner, const char16_t* name);
bool ReleaseMutex(Handle mutex);

// ---------------------------------------------------------------------------
// Waiting (WaitForSingleObject is declared in os/win32_kernel.h and implemented here)
// ---------------------------------------------------------------------------

// Alertable waits are accepted but APCs are not supported, so they never return
// WAIT_IO_COMPLETION.
uint32_t WaitForSingleObjectEx(Handle handle, uint32_t timeout_ms, bool alertable);
uint32_t WaitForMultipleObjects(uint32_t count,
                                const Handle* handles,
                                bool wait_all,
                                uint32_t timeout_ms);
uint32_t WaitForMultipleObjectsEx(uint32_t count,
                                  const Handle* handles,
                                  bool wait_all,
                                  uint32_t timeout_ms,
                                  bool alertable);

// ---------------------------------------------------------------------------
// Condition variables (SRW locks / critical sections from os/win32_kernel)
// ---------------------------------------------------------------------------

void InitializeConditionVariable(ConditionVariable* condition_variable);
// Returns false with ERROR_TIMEOUT when the timeout elapses; the lock is re-acquired either way.
bool SleepConditionVariableSRW(ConditionVariable* condition_variable,
                               SrwLock* srw_lock,
                               uint32_t timeout_ms,
                               uint32_t flags);
// The critical section must be entered exactly once by the caller.
bool SleepConditionVariableCS(ConditionVariable* condition_variable,
                              CriticalSection* critical_section,
                              uint32_t timeout_ms);
void WakeConditionVariable(ConditionVariable* condition_variable);
void WakeAllConditionVariable(ConditionVariable* condition_variable);

// ---------------------------------------------------------------------------
// Threads
// ---------------------------------------------------------------------------

// stack_size is ignored (pthread stacks are sized by the Emscripten build). The returned handle
// is signaled when the thread exits; mutexes it still owns become abandoned at that point.
Handle CreateThread(void* security_attributes,
                    uint64_t stack_size,
                    Pg6ThreadStartRoutine start_address,
                    void* parameter,
                    uint32_t creation_flags,
                    uint32_t* thread_id);
uint32_t ResumeThread(Handle thread);
bool GetExitCodeThread(Handle thread, uint32_t* exit_code);
Handle GetCurrentThread();
uint32_t GetCurrentThreadId();
uint64_t SetThreadAffinityMask(Handle thread, uint64_t affinity_mask);

// ---------------------------------------------------------------------------
// Vista thread pool (work objects + simple callbacks). Callback environments are accepted and
// ignored; all callbacks run on the single bounded process pool.
// ---------------------------------------------------------------------------

Pg6TpWork* CreateThreadpoolWork(Pg6TpWorkCallback callback, void* context, void* callback_environ);
// False when the work was not queued (null, closed, or the pool refused it): no callback will run.
bool SubmitThreadpoolWork(Pg6TpWork* work);
// Tests: the next `count` pool submissions (work and simple callbacks) fail as if the pool refused them.
void fail_next_thread_pool_submissions_for_testing(uint32_t count);
void WaitForThreadpoolWorkCallbacks(Pg6TpWork* work, bool cancel_pending_callbacks);
void CloseThreadpoolWork(Pg6TpWork* work);
bool TrySubmitThreadpoolCallback(Pg6TpSimpleCallback callback,
                                 void* context,
                                 void* callback_environ);

}  // namespace playgta6::os::win32

// ---------------------------------------------------------------------------
// Prefixed C-ABI entrypoints
// ---------------------------------------------------------------------------

extern "C" {

Pg6Handle pg6_win32_create_semaphore_a(void* security_attributes,
                                       int32_t initial_count,
                                       int32_t maximum_count,
                                       const char* name);
Pg6Handle pg6_win32_create_semaphore_w(void* security_attributes,
                                       int32_t initial_count,
                                       int32_t maximum_count,
                                       const char16_t* name);
int32_t pg6_win32_release_semaphore(Pg6Handle semaphore,
                                    int32_t release_count,
                                    int32_t* previous_count);

Pg6Handle pg6_win32_create_mutex_a(void* security_attributes,
                                   int32_t initial_owner,
                                   const char* name);
Pg6Handle pg6_win32_create_mutex_w(void* security_attributes,
                                   int32_t initial_owner,
                                   const char16_t* name);
int32_t pg6_win32_release_mutex(Pg6Handle mutex);

uint32_t pg6_win32_wait_for_single_object_ex(Pg6Handle handle,
                                             uint32_t timeout_ms,
                                             int32_t alertable);
uint32_t pg6_win32_wait_for_multiple_objects(uint32_t count,
                                             const Pg6Handle* handles,
                                             int32_t wait_all,
                                             uint32_t timeout_ms);
uint32_t pg6_win32_wait_for_multiple_objects_ex(uint32_t count,
                                                const Pg6Handle* handles,
                                                int32_t wait_all,
                                                uint32_t timeout_ms,
                                                int32_t alertable);

void pg6_win32_initialize_condition_variable(Pg6ConditionVariable* condition_variable);
int32_t pg6_win32_sleep_condition_variable_srw(Pg6ConditionVariable* condition_variable,
                                               Pg6SrwLock* srw_lock,
                                               uint32_t timeout_ms,
                                               uint32_t flags);
int32_t pg6_win32_sleep_condition_variable_cs(Pg6ConditionVariable* condition_variable,
                                              Pg6CriticalSection* critical_section,
                                              uint32_t timeout_ms);
void pg6_win32_wake_condition_variable(Pg6ConditionVariable* condition_variable);
void pg6_win32_wake_all_condition_variable(Pg6ConditionVariable* condition_variable);

Pg6Handle pg6_win32_create_thread(void* security_attributes,
                                  uint64_t stack_size,
                                  Pg6ThreadStartRoutine start_address,
                                  void* parameter,
                                  uint32_t creation_flags,
                                  uint32_t* thread_id);
uint32_t pg6_win32_resume_thread(Pg6Handle thread);
int32_t pg6_win32_get_exit_code_thread(Pg6Handle thread, uint32_t* exit_code);
Pg6Handle pg6_win32_get_current_thread(void);
uint32_t pg6_win32_get_current_thread_id(void);
uint64_t pg6_win32_set_thread_affinity_mask(Pg6Handle thread, uint64_t affinity_mask);

Pg6TpWork* pg6_win32_create_threadpool_work(Pg6TpWorkCallback callback,
                                            void* context,
                                            void* callback_environ);
void pg6_win32_submit_threadpool_work(Pg6TpWork* work);
void pg6_win32_wait_for_threadpool_work_callbacks(Pg6TpWork* work,
                                                  int32_t cancel_pending_callbacks);
void pg6_win32_close_threadpool_work(Pg6TpWork* work);
int32_t pg6_win32_try_submit_threadpool_callback(Pg6TpSimpleCallback callback,
                                                 void* context,
                                                 void* callback_environ);

}  // extern "C"
