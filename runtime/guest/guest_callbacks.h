#pragma once

#include <cstddef>
#include <cstdint>

#include "guest/com_bridge.h"
#include "os/win32_sync.h"
#include "os/xgameruntime.h"

// Guest callbacks for host APIs (O16).
//
// Several runtime shims take a host function pointer plus a `void*` context (XTaskQueue
// callbacks, CreateThread start routines) or call function pointers stored in guest memory
// (XAsyncBlock::callback, _initterm tables, qsort comparators). Guest code passes *guest*
// addresses there, which host code cannot call. This module bridges them:
//
// Trampolines
// -----------
// A Trampoline packages {guest function, guest context}. adapt_*() return the host function
// pointer the shim expects (a static entry per callback type) and the trampoline as its host
// context; when the shim calls back, the entry calls the guest function through
// com::GuestFunction (O9 call_guest, Win64 convention) on the current thread's guest state
// (com::current_guest_state: the state of the guest call that is running on this thread, or a
// guest thread created lazily for pool/worker threads). One-shot trampolines (every adapter
// below: the shims call each of these exactly once, also when cancelled) free themselves before
// running the guest function; if the shim call fails, the caller destroys the trampoline.
// live_trampolines() counts the ones not yet consumed (leak accounting).
//
// XAsyncBlock shadows
// -------------------
// A guest XAsyncBlock ({queue, context, callback, internal[32]}, 56 bytes, same layout as
// Pg6XAsyncBlock) cannot be handed to the shims: they call `callback` as a host function and keep
// their state in `internal`. async_shadow_begin() creates a host-side shadow block keyed by the
// guest block's address (queue copied, callback = a host entry that calls the guest callback
// with the *guest* block address); async_shadow_find() returns it for the XAsyncGetStatus /
// XxxResult calls that follow; async_shadow_end() drops it once the result was consumed. A begin
// on an address whose previous operation is still pending keeps the old shadow alive (the shim
// still references it) and reports a diagnostic. The operation state of a shadow lives in the
// XAsync provider framework (os/xasync_provider.h), keyed by the shadow; "pending" means the
// operation runs or its completion callback has not returned, and a dropped shadow's finished
// record is forgotten with it (O22). The guest block's `internal` bytes get a copy of the
// shadow's when the completion callback runs (opaque to the guest).
//
// Guest-state thunks for imports with function-pointer parameters
// ----------------------------------------------------------------
// R5's generated import thunks cannot bind shims with host function-pointer parameters. The
// pg6_gthunk_<name> functions below are complete import thunks (`void(void* state)`, entered
// with the return address pushed, pop it like `ret`) for such imports; R5 binds the IAT slot of
// import <Name> directly to pg6_gthunk_<name> whenever one exists (same name_key rule as for
// shims: ASCII-lowercase, underscores removed), instead of generating a marshaling thunk.
//
//   CreateThread                 start routine + parameter -> one-shot trampoline; the new host
//                                thread gets a lazily created guest thread (stack >= dwStackSize)
//   XTaskQueueSubmitCallback     callback + context -> one-shot trampoline
//   XTaskQueueTerminate          terminated callback + context -> one-shot trampoline
//   qsort, bsearch               implemented on guest memory, comparator called per comparison
//                                (merge sort on an index array, so an inconsistent guest
//                                comparator cannot make the host read out of bounds)
//   _initterm, _initterm_e       call the guest function-pointer table [first, last)
//   GetProcAddress               reads the name (or MAKEINTRESOURCE ordinal: HIWORD == 0) from
//                                guest memory, resolves it in the O7 module registry and returns
//                                the export's guest-callable synthetic address
//                                (guest_callable_for_proc). Exports registered only as host
//                                functions (no guest thunk) are not callable by guest code: they
//                                return NULL with ERROR_PROC_NOT_FOUND and a diagnostic instead of
//                                a host function pointer.
//   _crt_atexit, _crt_at_quick_exit, _register_onexit_function, _initialize_onexit_table,
//   _execute_onexit_table, _register_thread_local_exe_atexit_callback, exit, quick_exit, _exit,
//   _cexit, _c_exit              guest exit registration and the exit funnel
//                                (guest/guest_crt_exit.{h,cpp}, O22). `exit`/`_exit` and
//                                `_cexit`/`_c_exit` share a name key; the import bridge picks the
//                                thunk whose name after the prefix equals the import name.
//   HCCleanupAsync, HCHttpCallPerformAsync
//                                libHttpClient imports taking a guest XAsyncBlock: the shim
//                                completes a shadow block (O22)

namespace playgta6::guest::callbacks {

struct Trampoline {
    uint64_t function = 0;    // guest code address
    uint64_t context = 0;     // guest context / parameter passed back to it
    uint64_t stack_size = 0;  // thread start routines: minimum guest stack of the new thread
};

Trampoline* create_trampoline(uint64_t function, uint64_t context, uint64_t stack_size = 0);
void destroy_trampoline(Trampoline* trampoline);
uint64_t live_trampolines();

template <typename Fn>
struct HostCallback {
    Fn function = nullptr;               // pass to the shim
    void* context = nullptr;             // pass to the shim (the trampoline)
    Trampoline* trampoline = nullptr;    // destroy it if the shim call fails
};

// Pg6XTaskQueueCallback(context, canceled) -> guest void callback(void* context, bool canceled).
HostCallback<Pg6XTaskQueueCallback> adapt_task_queue_callback(uint64_t guest_function, uint64_t guest_context);
// Pg6XTaskQueueTerminatedCallback(context) -> guest void callback(void* context).
HostCallback<Pg6XTaskQueueTerminatedCallback> adapt_task_queue_terminated_callback(uint64_t guest_function,
                                                                                   uint64_t guest_context);
// Pg6ThreadStartRoutine(parameter) -> guest DWORD start(void* parameter) on the new thread.
HostCallback<Pg6ThreadStartRoutine> adapt_thread_start(uint64_t guest_function,
                                                       uint64_t guest_parameter,
                                                       uint64_t stack_size);

// Guest XAsyncBlock layout (Win64 GDK XAsyncBlock).
struct GuestAsyncBlock {
    uint64_t queue;
    uint64_t context;
    uint64_t callback;
    uint8_t internal[32];
};
static_assert(sizeof(GuestAsyncBlock) == 56, "XAsyncBlock is 56 bytes on x64");
static_assert(sizeof(Pg6XAsyncBlock) == sizeof(GuestAsyncBlock), "shadow and guest block sizes");

// nullptr if the guest block is not readable guest memory.
Pg6XAsyncBlock* async_shadow_begin(uint64_t guest_block);
Pg6XAsyncBlock* async_shadow_find(uint64_t guest_block);
void async_shadow_end(uint64_t guest_block);
size_t live_async_shadows();

// Tests only: forget every trampoline and shadow (not concurrently with callbacks).
void reset_callbacks();

}  // namespace playgta6::guest::callbacks

extern "C" {

void pg6_gthunk_create_thread(void* state);
void pg6_gthunk_xtask_queue_submit_callback(void* state);
void pg6_gthunk_xtask_queue_terminate(void* state);
void pg6_gthunk_qsort(void* state);
void pg6_gthunk_bsearch(void* state);
void pg6_gthunk_initterm(void* state);
void pg6_gthunk_initterm_e(void* state);
void pg6_gthunk_get_proc_address(void* state);
void pg6_gthunk_hc_cleanup_async(void* state);
void pg6_gthunk_hc_http_call_perform_async(void* state);

// guest/guest_rtl.cpp (O26): imports with guest callbacks or that need the guest state. The
// thread-pool work family takes opaque PTP_WORK values as uint64_t (see guest/guest_rtl.h).
void pg6_gthunk_fls_free(void* state);
void pg6_gthunk_init_once_execute_once(void* state);
void pg6_gthunk_rtl_capture_stack_back_trace(void* state);
void pg6_gthunk_create_threadpool_work(void* state);
void pg6_gthunk_submit_threadpool_work(void* state);
void pg6_gthunk_wait_for_threadpool_work_callbacks(void* state);
void pg6_gthunk_close_threadpool_work(void* state);
void pg6_gthunk_try_submit_threadpool_callback(void* state);

// guest/guest_crt_exit.cpp
void pg6_gthunk_crt_atexit(void* state);
void pg6_gthunk_crt_at_quick_exit(void* state);
void pg6_gthunk_register_onexit_function(void* state);
void pg6_gthunk_initialize_onexit_table(void* state);
void pg6_gthunk_execute_onexit_table(void* state);
void pg6_gthunk_register_thread_local_exe_atexit_callback(void* state);
void pg6_gthunk_exit(void* state);
void pg6_gthunk_quick_exit(void* state);
void pg6_gthunk__exit(void* state);
void pg6_gthunk__cexit(void* state);
void pg6_gthunk__c_exit(void* state);

}  // extern "C"
