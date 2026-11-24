#pragma once

#include <cstddef>
#include <cstdint>

#include "guest/guest_runtime.h"

// Guest side of the O26 Win32 shims (os/win32_rtl.h): the pieces that need the guest state.
//
// Import thunks (pg6_gthunk_*, declared in guest/guest_callbacks.h so the import bridge finds
// them): FlsFree (destructors of every thread's value run on the calling guest thread),
// InitOnceExecuteOnce (guest callback BOOL(PINIT_ONCE, PVOID, PVOID*); an exception escaping it
// propagates to the caller and leaves the once uninitialized), RtlCaptureStackBackTrace (walks
// the guest stack with R11's unwinder from the caller of the import) and the Vista thread-pool
// work family (CreateThreadpoolWork, SubmitThreadpoolWork, WaitForThreadpoolWorkCallbacks,
// CloseThreadpoolWork, TrySubmitThreadpoolCallback).
//
// Thread-pool handles. The PTP_WORK the guest holds is the runtime's work object address as an
// opaque 64-bit value (never dereferenced by the guest); every call validates it against the live
// set, so a stale or forged handle is a diagnostic and a no-op. Work callbacks run on pool
// threads as top-level callbacks (guest threads created lazily per pool thread) with
// (PTP_CALLBACK_INSTANCE = opaque, the context passed to CreateThreadpoolWork, the PTP_WORK).
// A callback environment is accepted and ignored. A cancelled submission (WaitFor...(TRUE))
// never calls the callback. The small per-work record is freed by CloseThreadpoolWork once the
// outstanding callbacks ran; submissions cancelled by a wait with cancel = TRUE are forgotten
// when that wait returns.
//
// FLS thread exit: run_fls_thread_exit(state) runs the calling thread's FLS destructors through
// call_guest on `state` (top-level callbacks) and releases its FLS storage. It is called by the
// lazily created guest thread's teardown (CreateThread threads, pool threads, any host thread
// that ran guest code through the COM bridge) and by the CRT exit funnel
// (guest_crt_exit run_exit_cleanup, ExitMode::Full) for the exiting thread.

namespace playgta6::guest::rtl {

void run_fls_thread_exit(GuestState& state);

// Number of thread-pool work records and one-shot callback records not yet released (leak
// accounting for tests).
uint64_t live_pool_records();

// Work-object records allocated and not yet deleted, including closed ones whose callbacks are
// still outstanding (tests: each record is deleted exactly once, so this never drops below zero).
int64_t live_work_records();

// Tests only: forget every pool record (not concurrently with callbacks).
void reset_rtl();

}  // namespace playgta6::guest::rtl
