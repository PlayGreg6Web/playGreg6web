#pragma once

#include <cstdint>

#include "abi/playgta6_abi.h"

// Guest CRT exit registration and the guest process exit funnel (O22).
//
// The executable registers its exit-time work with the UCRT through imports that take guest
// function addresses (`_crt_atexit`, `_register_onexit_function`,
// `_register_thread_local_exe_atexit_callback`). The host CRT bridge (os/ucrt_bridge.h) keeps
// host function pointers for host callers; for guest callers this module keeps the guest
// addresses and calls them through O9's call_guest. The pg6_gthunk_* import thunks that bind
// these imports are declared in guest/guest_callbacks.h.
//
// onexit tables
// -------------
// `_onexit_table_t` ({first, last, end}, three guest pointers, 24 bytes) lives in guest memory and
// is owned by the guest; the guest's statically linked startup code may also write the "use the
// process table" sentinel (all three = ~0) into it itself. Entries are stored faithfully in that
// table: `first..last` is an array of plain 8-byte guest function addresses in a buffer this
// module allocates from the COM bridge's guest memory (com::guest_alloc). Real UCRT encodes the
// pointers with its private cookie; guest code never reads the entries (only the UCRT functions
// do), so plain addresses are equivalent and keep guest and host views identical. The process-
// wide `_crt_atexit` / `_crt_at_quick_exit` lists are two such tables allocated on first use, so
// one code path serves both.
//
//   initialize_table   UCRT semantics: a table with first != end is left alone; otherwise all
//                      three pointers become null (this also resets the ~0 sentinel).
//   register_function  appends; grows the buffer (32 entries first, then +min(count, 512)); a
//                      table holding the ~0 sentinel is treated as empty. A null function is
//                      accepted and ignored. -1 for an unreadable table or no guest memory.
//   execute_table      calls the entries in reverse order (LIFO), each exactly once: an entry is
//                      cleared before it is called, and after every call the table is re-read so
//                      that functions registered by a running handler run next (as UCRT does).
//                      Then frees the buffer and nulls the table.
// All table operations take one recursive process "exit lock" (UCRT's __acrt_exit_lock): a
// handler may register further handlers or call exit itself on the same thread, other threads
// wait.
//
// Exit funnel
// -----------
// exit_process(state, code, mode) is the one guest process exit path:
//   ExitMode::Full   (exit)        the thread-local exe callback with (nullptr, DLL_PROCESS_DETACH,
//                                  nullptr) - once per process -, then the atexit table
//   ExitMode::Quick  (quick_exit)  only the at_quick_exit table
//   ExitMode::None   (_exit)       nothing
// on the exiting guest thread (`state`), then ends the process through the CRT exit hook
// (pg6_crt_exit for Full, pg6_crt__exit otherwise): the guest boot (guest/boot.h) installs that
// hook and stops the run with the exit code. If no hook stops execution the host process ends
// (std::_Exit): exit never returns to guest code. A handler that calls exit again re-enters the
// funnel, which continues with the handlers not yet run and then terminates; every handler runs
// at most once. `_cexit` runs the Full cleanup and returns; `_c_exit` runs nothing and returns.
// abort/terminate (ucrt_bridge shims) and TerminateProcess run no handlers. When the entry point
// returns instead (the boot's other exit path), the boot runs the Full cleanup first.
//
// The thread-local exe callback is the single `__dyn_tls_dtor` MSVC's startup code registers
// (`void (*)(void* dll, unsigned long reason, void* reserved)`); a second registration is ignored
// with a diagnostic. Per-thread destruction on thread exit is the loader's TLS-callback job
// (DLL_THREAD_DETACH), not this callback's.

namespace playgta6::guest::crt_exit {

struct GuestOnexitTable {
    uint64_t first;
    uint64_t last;
    uint64_t end;
};
static_assert(sizeof(GuestOnexitTable) == 24, "_onexit_table_t is three pointers on x64");

inline constexpr uint64_t kTableSentinel = ~uint64_t{0};
inline constexpr uint32_t kDllProcessDetach = 0;

enum class ExitMode : uint8_t {
    Full = 0,   // exit
    Quick = 1,  // quick_exit
    None = 2,   // _exit
};

int32_t initialize_table(uint64_t table_va);
int32_t register_function(uint64_t table_va, uint64_t function);
int32_t execute_table(GuestState& state, uint64_t table_va);

// Guest addresses of the process-wide tables (allocated on first use; 0 without guest memory).
uint64_t atexit_table();
uint64_t at_quick_exit_table();

// `_register_thread_local_exe_atexit_callback`.
void set_thread_local_exit_callback(uint64_t function);
uint64_t thread_local_exit_callback();

// Runs the cleanup of `mode` on `state` and returns.
void run_exit_cleanup(GuestState& state, ExitMode mode);

// The guest process exit funnel (see above). Does not return.
[[noreturn]] void exit_process(GuestState& state, int32_t code, ExitMode mode);

// Tests / reset_boot: forgets the process tables and the thread-local callback (buffers are not
// freed: the guest memory they live in may already be gone).
void reset_crt_exit();

}  // namespace playgta6::guest::crt_exit
