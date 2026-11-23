#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include "abi/playgta6_abi.h"
#include "os/vmem.h"

// Guest runtime: runs statically recompiled guest code (see docs/wasm_abi.md, "Recompiled Guest
// Code") and implements the four helpers lifted code calls.
//
// Dispatch
// --------
// `__playgta6_indirect_call(state, target)` resolves `target` in this order:
//   1. a synthetic address (see below): the guest-return sentinel returns immediately, any other
//      bound synthetic address runs its host thunk `void(void* state)` (R5's thunk shape; the
//      thunk pops the return address itself, like `ret`);
//   2. a registered lifted function entry: calls it;
//   3. anything else is a fault (FaultKind::UnknownTarget). This includes addresses inside lifted
//      code that are not function entries (e.g. the default arm of a lifted jump-table `switch`).
// The dispatcher writes `target` to GuestState::rip before running it (diagnostics only; lifted
// code never reads rip).
//
// Lifted functions are registered as static tables `{va, fn}` sorted by va (generated code
// passes its table in one call, nothing is copied). Lookups are a binary search per table and
// are lock-free; registration takes a mutex and publishes with release stores, so it may run
// concurrently with dispatch. Tables must stay alive for the life of the process.
//
// Synthetic addresses
// -------------------
// Imports (IAT slots) and host callables get synthetic guest addresses
//     kSyntheticBase + index * kSyntheticStride   (index 1 .. kMaxSyntheticTargets - 1)
// kSyntheticBase = 0x5047'0000'0000'0000 is a *non-canonical* x86-64 address (bits 63..47 are
// not all equal): no real guest code, data or stack can live there, so it never collides with
// the guest image, VirtualMemoryArena reservations (which start at 0x1'0000'0000 and grow
// upwards) or O7's virtual module handles (kVirtualModuleBase = 0x7FF8'0000'0000). It is also
// far above any wasm64 linear memory size, so it can never alias a real address in the browser.
// Index 0 is kGuestReturnSentinel, the return address call_guest() pushes for the entry
// function.
//
// Imports: bind_import_slots() gives every IAT slot its own synthetic address, writes it into
// the slot in guest memory and records slot -> thunk. Host callables: register_host_callable()
// gives a host thunk a synthetic address; passing the O7 Pg6ProcAddress of the export also
// records proc -> synthetic, so a guest-facing GetProcAddress can hand out dispatchable
// addresses via guest_callable_for_proc() (O7 itself returns raw host function pointers /
// Wasm64 table indices, which guest code cannot call).
//
// Faults
// ------
// Every FaultKind goes through one replaceable fault handler. The default handler logs the fault
// (kind, guest VA, register dump) through playgta6::os::log and calls std::abort(); in the
// Emscripten build that is a wasm trap (`RuntimeError: Aborted`) that reaches the JS loader's
// abort handler. A custom handler must not return (throw, longjmp or terminate); if it does
// return, the default handler runs. Unwinding out of the handler skips the rest of the faulting
// guest call (call_guest does not restore rsp in that case); the runtime keeps no destructible
// objects alive across guest code, so a C++ exception thrown by the handler unwinds cleanly
// through the extern "C" helper frames (tested with MSVC /EHsc x64, GCC, and Emscripten wasm64
// with -fexceptions; Emscripten builds without exception support abort on throw instead).
//
// Guest memory
// ------------
// The runtime touches guest memory only through VirtualMemoryArena (stack setup, call_guest
// arguments, IAT slots), like the O7 heap and the O10 image loader. Lifted code instead maps
// guest addresses 1:1 onto linear memory; in the browser build the two only agree once the
// arena hands out identity-mapped backing (not the case yet).

using Pg6HostThunkFn = void (*)(void* state);

// One entry of a lifted-function table (sorted by va, no duplicates).
struct Pg6LiftedFunctionEntry {
    uint64_t va;
    playgta6::LiftedFn fn;
};

// One IAT slot to bind (same shape as R5's Pg6ImportThunk {slot_rva, ..., thunk}).
struct Pg6ImportSlotBinding {
    uint32_t slot_rva;
    Pg6HostThunkFn thunk;
};

struct Pg6GuestFaultInfo {
    uint32_t kind;    // playgta6::guest::FaultKind
    uint32_t reserved;
    uint64_t va;      // faulting guest address / unknown target / entry of call_guest
    uint64_t detail;  // kind specific, see FaultKind
    playgta6::GuestState* state;
};

using Pg6GuestFaultHandler = void (*)(const Pg6GuestFaultInfo* info);

// A guest thread: CPU state plus its stack in the arena.
struct Pg6GuestThread {
    playgta6::GuestState state;
    uint64_t stack_reservation;  // arena reservation base (guard page included)
    uint64_t stack_limit;        // lowest usable stack address (above the guard page)
    uint64_t stack_top;          // one past the highest stack byte; initial rsp
};

namespace playgta6::guest {

using HostThunkFn = Pg6HostThunkFn;
using LiftedFunctionEntry = Pg6LiftedFunctionEntry;
using ImportSlotBinding = Pg6ImportSlotBinding;
using FaultInfo = Pg6GuestFaultInfo;
using FaultHandler = Pg6GuestFaultHandler;
using GuestThread = Pg6GuestThread;

inline constexpr uint64_t kSyntheticBase = 0x5047000000000000ULL;
inline constexpr uint64_t kSyntheticStride = 16;
inline constexpr uint64_t kMaxSyntheticTargets = uint64_t{1} << 20;
inline constexpr uint64_t kGuestReturnSentinel = kSyntheticBase;
inline constexpr size_t kMaxLiftedTables = 64;
inline constexpr uint64_t kDefaultGuestStackSize = uint64_t{1} << 20;

enum class FaultKind : uint32_t {
    Trap = 1,            // __playgta6_trap
    Unimplemented = 2,   // __playgta6_unimplemented
    DivideError = 3,     // __playgta6_divide_error
    UnknownTarget = 4,   // va = target; detail = nearest lifted entry below it (0 if none)
    StackImbalance = 5,  // va = call_guest entry; detail = rsp after return (expected rsp + 8)
    BadStack = 6,        // va = call_guest entry; detail = stack address that is not writable
    GeneralProtection = 7,  // #GP from a CPU helper (xgetbv): va = rip; detail = bad XCR index
};

const char* fault_kind_name(FaultKind kind);

inline constexpr bool is_synthetic_address(uint64_t va) {
    return va >= kSyntheticBase && va < kSyntheticBase + kMaxSyntheticTargets * kSyntheticStride;
}

// ---------------------------------------------------------------------------
// Lifted functions
// ---------------------------------------------------------------------------

// Registers a static table. Fails (false) on null/unsorted/duplicate entries, a null fn, a
// synthetic va, a range overlapping an already registered table, or more than kMaxLiftedTables.
bool register_lifted_functions(std::span<const LiftedFunctionEntry> table);
LiftedFn find_lifted_function(uint64_t va);
size_t lifted_function_count();

// ---------------------------------------------------------------------------
// Imports and host callables
// ---------------------------------------------------------------------------

// For each slot writes a fresh synthetic address into the 8-byte IAT slot at
// image_base + slot_rva (temporarily lifting a read-only page protection) and records the
// slot's thunk. Returns the number of slots bound; slots with a null thunk or unwritable memory
// are skipped and logged.
size_t bind_import_slots(os::VirtualMemoryArena& arena,
                         uint64_t image_base,
                         std::span<const ImportSlotBinding> slots);

// Gives `thunk` a synthetic guest address (0 if the table is full or thunk is null). `name` is
// copied and used in diagnostics. A non-zero `proc_address` (O7 Pg6ProcAddress) is mapped to the
// new address for guest_callable_for_proc().
uint64_t register_host_callable(HostThunkFn thunk, const char* name = nullptr, uint64_t proc_address = 0);
// Synthetic address registered for an O7 proc address, or 0 (takes the registry lock; meant for
// GetProcAddress, not the dispatch path).
uint64_t guest_callable_for_proc(uint64_t proc_address);

struct SyntheticTarget {
    HostThunkFn thunk = nullptr;
    uint64_t image_base = 0;  // imports only
    uint32_t slot_rva = 0;    // imports only
    bool is_import = false;
    const char* name = nullptr;  // host callables only, may be null
};

// Looks up a bound synthetic address (false for the sentinel and unbound addresses).
bool describe_synthetic(uint64_t va, SyntheticTarget* out);

// ---------------------------------------------------------------------------
// Faults
// ---------------------------------------------------------------------------

// Installs a fault handler (nullptr restores the default). Returns the previous custom handler.
FaultHandler set_fault_handler(FaultHandler handler);

// Formats the diagnostic the default handler logs ("guest fault: <kind> at 0x...", registers).
// Returns the full length (like snprintf).
size_t format_fault(const FaultInfo& info, char* buffer, size_t size);

[[noreturn]] void raise_fault(FaultKind kind, GuestState* state, uint64_t va, uint64_t detail = 0);

// Optional observer run by __playgta6_indirect_call before it resolves `target` (call counting
// and run bounds for diagnostics, see guest/boot.h). It may leave like a fault handler (throw,
// longjmp) to stop the guest. nullptr removes it. Returns the previous observer.
using DispatchObserver = void (*)(GuestState* state, uint64_t target);
DispatchObserver set_dispatch_observer(DispatchObserver observer);

// ---------------------------------------------------------------------------
// Guest threads and entry calls
// ---------------------------------------------------------------------------

// Reserves stack_size (rounded up to pages) plus one uncommitted guard page below it in the
// arena, commits the stack read/write, zeroes the state and sets rsp = stack_top (16-byte
// aligned). The TEB is not set up here: point state.gs_base at it with set_teb().
bool create_guest_thread(os::VirtualMemoryArena& arena, uint64_t stack_size, GuestThread& thread);
void destroy_guest_thread(os::VirtualMemoryArena& arena, GuestThread& thread);

inline void set_teb(GuestState& state, uint64_t teb_va) {
    state.gs_base = teb_va;
}

// Calls guest code at `va` (lifted function, import or host callable) as a Win64 caller would:
// below the current rsp it reserves the stack arguments and the 32-byte shadow space, pushes
// kGuestReturnSentinel so that (rsp + 8) % 16 == 0 at entry, passes args[0..3] in rcx rdx r8 r9
// and args[4..] at [rsp + 40 + 8 * i]. Integer/pointer arguments only. Returns rax and restores
// rsp. Works re-entrantly from host thunks (guest callbacks), since it only uses stack below
// the current rsp. A wrong rsp after return raises StackImbalance, an unwritable stack BadStack.
uint64_t call_guest(os::VirtualMemoryArena& arena,
                    GuestState& state,
                    uint64_t va,
                    std::span<const uint64_t> args = {});

template <typename... Args>
uint64_t call_guest_args(os::VirtualMemoryArena& arena, GuestState& state, uint64_t va, Args... args) {
    static_assert((std::is_integral_v<Args> && ...), "call_guest passes integer arguments only");
    const uint64_t values[sizeof...(Args) + 1] = {static_cast<uint64_t>(args)..., 0};
    return call_guest(arena, state, va, std::span<const uint64_t>(values, sizeof...(Args)));
}

// Drops every registration and restores the default fault handler. Only for tests: must not run
// concurrently with dispatch.
void reset_guest_runtime();

}  // namespace playgta6::guest

// ---------------------------------------------------------------------------
// C-ABI entry points (guest memory = playgta6::os::win32::get_vmem_arena())
// ---------------------------------------------------------------------------

extern "C" {

int32_t pg6_guest_register_lifted_functions(const Pg6LiftedFunctionEntry* entries, size_t count);
playgta6::LiftedFn pg6_guest_find_lifted_function(uint64_t va);
size_t pg6_guest_bind_import_slots(uint64_t image_base, const Pg6ImportSlotBinding* slots, size_t count);
uint64_t pg6_guest_register_host_callable(Pg6HostThunkFn thunk, const char* name, uint64_t proc_address);
uint64_t pg6_guest_callable_for_proc(uint64_t proc_address);
Pg6GuestFaultHandler pg6_guest_set_fault_handler(Pg6GuestFaultHandler handler);
int32_t pg6_guest_thread_create(Pg6GuestThread* thread, uint64_t stack_size);
void pg6_guest_thread_destroy(Pg6GuestThread* thread);
uint64_t pg6_guest_call(playgta6::GuestState* state, uint64_t va, const uint64_t* args, size_t count);

}  // extern "C"
