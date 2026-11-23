#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "abi/playgta6_abi.h"
#include "guest/guest_runtime.h"
#include "os/vmem.h"

// Guest exception dispatch and unwinding (Win64 x64 semantics over guest memory and GuestState).
//
// Split of responsibilities
// -------------------------
// Language semantics stay in guest code: VCRUNTIME140.dll / VCRUNTIME140_1.dll ship next to the
// game and are recompiled like the game itself, so `_CxxThrowException`, `__CxxFrameHandler3/4`,
// `__C_specific_handler`, `__intrinsic_setjmp` and `longjmp` run as lifted guest code. What they
// (and the game) need from the OS is implemented here, exactly as ntdll/kernel32 do it on x64:
// function-table lookup, RtlVirtualUnwind, RtlCaptureContext, RaiseException with the
// RtlDispatchException loop, RtlUnwindEx/RtlUnwind, RtlRestoreContext (incl. STATUS_LONGJUMP and
// STATUS_UNWIND_CONSOLIDATE), RtlPcToFileHeader, RtlAdd/DeleteFunctionTable and the top-level
// unhandled-exception filter. The guest stack and the registers saved by prologs are faithful
// (lifted calls push real return addresses, prologs store real nonvolatile registers), so the
// unmodified Windows algorithm walks lifted frames using the image's own .pdata. Language
// handlers are guest functions called through call_guest with EXCEPTION_RECORD, CONTEXT and
// DISPATCHER_CONTEXT laid out in guest memory as on Windows x64 (structs below).
//
// Transferring control (the resume protocol)
// ------------------------------------------
// A context restore must continue at an arbitrary rip inside some function while the host (wasm)
// stack still holds one native frame per guest activation below the target. Lifted code cannot
// be entered mid-function, so restore_context writes the restored registers into the state, sets
// GuestState::resume_rip and unwind_pending and returns (docs/wasm_abi.md). Every lifted
// activation that sees the flag after a call returns at once without writing its registers back;
// the innermost activation whose entry rsp lies above the restored rsp clears the flag, reloads
// its registers and branches to the block at resume_rip (the lifter's resume targets: catch
// continuations, __except bodies, setjmp return sites). Runtime frames take part the same way:
// call_guest returns immediately with the flag set, and every thunk here that ran guest code
// returns without touching the state. A restore whose target is the instruction right after the
// import call that requested it (RaiseException continued by a handler, RtlRestoreContext to its
// own return address) is applied in place instead: the thunk just returns into its caller.
//
// Frames across call_guest
// ------------------------
// Host code sits between guest activations whenever the runtime calls back into guest code
// (language handlers called by the dispatcher, consolidation callbacks, host callbacks). Each
// call_guest activation is a GuestCallBoundary; the frame walk crosses it when it unwinds to the
// kGuestReturnSentinel return address. What lies above depends on the boundary's tag, mirroring the
// system frames Windows has there:
//   - untagged (plain call_guest from a host thunk entered by a guest `call`): the frame above
//     is that guest caller (rip = [caller_rsp], rsp = caller_rsp + 8), like a leaf function;
//   - dispatch (a handler called by the dispatch loop; RtlpExecuteHandlerForException): the walk
//     continues at the context of the original raise; a new exception dispatched across it is a
//     nested exception (EXCEPTION_NESTED_CALL until the frame being dispatched is reached again);
//   - unwind (a handler called by RtlUnwindEx; RtlpExecuteHandlerForUnwind): dispatch continues
//     at RtlUnwindEx's caller, and a second unwind reaching it is a collided unwind (it takes over
//     the outer unwind's dispatcher context and continues from the frame being unwound there);
//   - consolidate (the STATUS_UNWIND_CONSOLIDATE callback, e.g. a C++ catch block run by
//     __CxxCallCatchBlock on top of the already unwound frames): the walk continues at the target
//     frame's context.
// The outermost call_guest of a thread ends the walk (an exception reaching it is unhandled).
//
// Unwinding details
// -----------------
// RtlVirtualUnwind implements every UNWIND_CODE (push/alloc small, large and far/set frame
// register/save nonvolatile and XMM, near and far/push machine frame, version-2 epilog markers
// are skipped), chained unwind info and indirect RUNTIME_FUNCTION entries. The establisher frame
// is rsp, or the frame register minus FrameOffset * 16 once UWOP_SET_FPREG has executed; SAVE_*
// offsets are relative to it (as on Windows). Inside the prolog only the codes whose prolog
// offset has been reached are applied and no handler is returned. Outside the prolog the
// epilog rule applies: if the bytes at the control PC are an epilog (optional `add rsp, imm` /
// `lea rsp, [reg+disp]`, pops of nonvolatile registers, then `ret`/`ret imm16`/`rep ret`, a `jmp`
// within the function that continues the epilog, or a tail `jmp` out of it / `jmp [rip+disp]`),
// it is emulated and no handler is returned.
// Lifted code is never *inside* an epilog when the unwinder looks at it: frames are only
// observed at call boundaries (a return address, or the context a thunk captured), never
// part-way through the epilog instructions. A return address can however be the *first*
// instruction of an epilog (`call f; add rsp, 28h; ret`). Windows applies the epilog rule there
// as well, and so does this unwinder, reading the original code bytes from guest memory; the
// unwound registers are identical to applying the unwind codes, and like on Windows no handler
// is reported for such a frame (MSVC inserts a `nop` after a call that would end a try block
// right before an epilog for that reason). If the image was loaded with executable bytes
// omitted (PGIM kFlagExecutableBytesOmitted), the epilog check sees zeros, never matches, and
// the frame is unwound with its unwind codes instead; only the "no handler for such a frame"
// detail differs.
//
// Not implemented: UNWIND_HISTORY_TABLE caching (the pointer is accepted and ignored), stack
// limit checks against the TEB (establisher frames are checked for 8-byte alignment only),
// raising STATUS_NONCONTINUABLE_EXCEPTION / STATUS_INVALID_DISPOSITION as nested guest exceptions
// (both are reported through the fault handler as unhandled exceptions with that code), and the
// non-unwinding longjmp form (Frame == 0 in the jump buffer jumps through the dispatcher, which
// reports an unknown target; MSVC x64 always passes a frame).
//
// Binding
// -------
// The APIs are host thunks of O9's shape (`void(void* state)`; arguments from the Win64 registers
// and stack slots of the state, result in rax, the thunk pops the return address). They cannot
// go through R5's marshaling bridge, because they operate on the caller's registers. The loader
// binds the IAT slots of these imports to exception_thunk_for(module, name) (it resolves API
// sets, e.g. api-ms-win-core-rtlsupport-l1-1-0.dll), before falling back to generated thunks.
// register_exception_exports() also publishes them in O7's module registry under KERNEL32.dll
// (and ntdll.dll for the Rtl* functions), each with a synthetic guest address, so a guest
// GetProcAddress can hand out a dispatchable address (guest_callable_for_proc).
//
// Guest memory is only touched through the VirtualMemoryArena (read/write); the C-ABI entry
// points and the thunks use playgta6::os::win32::get_vmem_arena(). Thread-safety: function-table
// registration takes a lock; lookups take it shared. Dispatch state is per call (and the
// call_guest boundary chain is thread-local), so threads unwind independently.

namespace playgta6::guest {

// ---------------------------------------------------------------------------
// Win64 structures in guest memory (layouts as on Windows x64)
// ---------------------------------------------------------------------------

struct GuestM128 {
    uint64_t low;
    uint64_t high;
};

// CONTEXT (x64), 1232 bytes, 16-byte aligned.
struct alignas(16) GuestContext {
    uint64_t p_home[6];
    uint32_t context_flags;
    uint32_t mx_csr;
    uint16_t seg_cs;
    uint16_t seg_ds;
    uint16_t seg_es;
    uint16_t seg_fs;
    uint16_t seg_gs;
    uint16_t seg_ss;
    uint32_t eflags;
    uint64_t dr[6];     // Dr0 Dr1 Dr2 Dr3 Dr6 Dr7
    uint64_t gpr[16];   // Rax Rcx Rdx Rbx Rsp Rbp Rsi Rdi R8..R15 (same order as GuestState::gpr)
    uint64_t rip;
    // XMM_SAVE_AREA32 (FltSave)
    uint16_t control_word;
    uint16_t status_word;
    uint8_t tag_word;
    uint8_t reserved1;
    uint16_t error_opcode;
    uint32_t error_offset;
    uint16_t error_selector;
    uint16_t reserved2;
    uint32_t data_offset;
    uint16_t data_selector;
    uint16_t reserved3;
    uint32_t flt_mx_csr;
    uint32_t flt_mx_csr_mask;
    GuestM128 float_registers[8];
    GuestM128 xmm[16];
    uint8_t reserved4[96];
    // Vector registers, control and debug state
    GuestM128 vector_register[26];
    uint64_t vector_control;
    uint64_t debug_control;
    uint64_t last_branch_to_rip;
    uint64_t last_branch_from_rip;
    uint64_t last_exception_to_rip;
    uint64_t last_exception_from_rip;
};

static_assert(sizeof(GuestContext) == 1232);
static_assert(alignof(GuestContext) == 16);
static_assert(offsetof(GuestContext, context_flags) == 0x30);
static_assert(offsetof(GuestContext, mx_csr) == 0x34);
static_assert(offsetof(GuestContext, seg_cs) == 0x38);
static_assert(offsetof(GuestContext, seg_ss) == 0x42);
static_assert(offsetof(GuestContext, eflags) == 0x44);
static_assert(offsetof(GuestContext, dr) == 0x48);
static_assert(offsetof(GuestContext, gpr) == 0x78);
static_assert(offsetof(GuestContext, rip) == 0xF8);
static_assert(offsetof(GuestContext, control_word) == 0x100);
static_assert(offsetof(GuestContext, flt_mx_csr) == 0x118);
static_assert(offsetof(GuestContext, xmm) == 0x1A0);
static_assert(offsetof(GuestContext, vector_register) == 0x300);
static_assert(offsetof(GuestContext, vector_control) == 0x4A0);
static_assert(offsetof(GuestContext, last_exception_from_rip) == 0x4C8);

// EXCEPTION_RECORD (x64), 152 bytes.
inline constexpr uint32_t kExceptionMaximumParameters = 15;
struct GuestExceptionRecord {
    uint32_t code;
    uint32_t flags;
    uint64_t record;  // chained EXCEPTION_RECORD (guest VA) or 0
    uint64_t address;
    uint32_t number_parameters;
    uint32_t reserved;
    uint64_t information[kExceptionMaximumParameters];
};
static_assert(sizeof(GuestExceptionRecord) == 152);
static_assert(offsetof(GuestExceptionRecord, address) == 16);
static_assert(offsetof(GuestExceptionRecord, number_parameters) == 24);
static_assert(offsetof(GuestExceptionRecord, information) == 32);

// DISPATCHER_CONTEXT (x64), 80 bytes. Pointers are guest VAs.
struct GuestDispatcherContext {
    uint64_t control_pc;
    uint64_t image_base;
    uint64_t function_entry;  // RUNTIME_FUNCTION*
    uint64_t establisher_frame;
    uint64_t target_ip;
    uint64_t context_record;  // CONTEXT*
    uint64_t language_handler;
    uint64_t handler_data;
    uint64_t history_table;
    uint32_t scope_index;
    uint32_t fill0;
};
static_assert(sizeof(GuestDispatcherContext) == 80);
static_assert(offsetof(GuestDispatcherContext, context_record) == 40);
static_assert(offsetof(GuestDispatcherContext, scope_index) == 72);

// RUNTIME_FUNCTION, 12 bytes (RVAs).
struct GuestRuntimeFunction {
    uint32_t begin;
    uint32_t end;
    uint32_t unwind;  // UNWIND_INFO RVA; bit 0 set: RVA + 1 of another RUNTIME_FUNCTION
};
static_assert(sizeof(GuestRuntimeFunction) == 12);

// KNONVOLATILE_CONTEXT_POINTERS (x64), 256 bytes: where RtlVirtualUnwind found each register.
struct GuestContextPointers {
    uint64_t xmm[16];  // M128A* of xmm0..15
    uint64_t gpr[16];  // DWORD64* of rax..r15
};
static_assert(sizeof(GuestContextPointers) == 256);

// _JUMP_BUFFER (x64), 256 bytes (setjmp / STATUS_LONGJUMP).
struct GuestJumpBuffer {
    uint64_t frame;
    uint64_t rbx;
    uint64_t rsp;
    uint64_t rbp;
    uint64_t rsi;
    uint64_t rdi;
    uint64_t r12;
    uint64_t r13;
    uint64_t r14;
    uint64_t r15;
    uint64_t rip;
    uint32_t mx_csr;
    uint16_t fp_csr;
    uint16_t spare;
    GuestM128 xmm[10];  // xmm6..xmm15
};
static_assert(sizeof(GuestJumpBuffer) == 256);
static_assert(offsetof(GuestJumpBuffer, rip) == 80);
static_assert(offsetof(GuestJumpBuffer, xmm) == 96);

// EXCEPTION_POINTERS, 16 bytes.
struct GuestExceptionPointers {
    uint64_t exception_record;
    uint64_t context_record;
};
static_assert(sizeof(GuestExceptionPointers) == 16);

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

inline constexpr uint32_t kExceptionNoncontinuable = 0x01;
inline constexpr uint32_t kExceptionUnwinding = 0x02;
inline constexpr uint32_t kExceptionExitUnwind = 0x04;
inline constexpr uint32_t kExceptionStackInvalid = 0x08;
inline constexpr uint32_t kExceptionNestedCall = 0x10;
inline constexpr uint32_t kExceptionTargetUnwind = 0x20;
inline constexpr uint32_t kExceptionCollidedUnwind = 0x40;

// EXCEPTION_DISPOSITION
inline constexpr uint32_t kExceptionContinueExecution = 0;
inline constexpr uint32_t kExceptionContinueSearch = 1;
inline constexpr uint32_t kExceptionNestedException = 2;
inline constexpr uint32_t kExceptionCollidedUnwindDisposition = 3;

// Filter results (UnhandledExceptionFilter, __except filters)
inline constexpr int32_t kExceptionExecuteHandler = 1;
inline constexpr int32_t kExceptionContinueSearchFilter = 0;
inline constexpr int32_t kExceptionContinueExecutionFilter = -1;

inline constexpr uint32_t kStatusLongjump = 0x80000026;
inline constexpr uint32_t kStatusUnwindConsolidate = 0x80000029;
inline constexpr uint32_t kStatusNoncontinuableException = 0xC0000025;
inline constexpr uint32_t kStatusInvalidDisposition = 0xC0000026;
inline constexpr uint32_t kStatusUnwind = 0xC0000027;
inline constexpr uint32_t kStatusBadStack = 0xC0000028;
inline constexpr uint32_t kStatusInvalidUnwindTarget = 0xC0000029;

// RtlVirtualUnwind HandlerType
inline constexpr uint32_t kUnwFlagNHandler = 0;
inline constexpr uint32_t kUnwFlagEHandler = 1;
inline constexpr uint32_t kUnwFlagUHandler = 2;
inline constexpr uint32_t kUnwFlagChainInfo = 4;

inline constexpr uint32_t kContextAmd64 = 0x00100000;
inline constexpr uint32_t kContextControl = kContextAmd64 | 0x1;
inline constexpr uint32_t kContextInteger = kContextAmd64 | 0x2;
inline constexpr uint32_t kContextSegments = kContextAmd64 | 0x4;
inline constexpr uint32_t kContextFloatingPoint = kContextAmd64 | 0x8;
inline constexpr uint32_t kContextFull = kContextControl | kContextInteger | kContextFloatingPoint;

// ---------------------------------------------------------------------------
// Function tables
// ---------------------------------------------------------------------------

// Registers the static function table of an image mapped at `image_base` (the RUNTIME_FUNCTION
// array of `count` entries at guest VA `table_va`, e.g. the .pdata directory). Entries are copied
// for the lookup; their guest addresses are what RtlLookupFunctionEntry returns. Fails on a null
// or overlapping image range, an unreadable table or more than 1M entries.
bool register_image_function_table(os::VirtualMemoryArena& arena,
                                   uint64_t image_base,
                                   uint64_t image_size,
                                   uint64_t table_va,
                                   uint32_t count);

// Same, finding SizeOfImage and the exception directory in the PE headers mapped at image_base
// (PGIM images keep their headers mapped).
bool register_image_from_headers(os::VirtualMemoryArena& arena, uint64_t image_base);

// RtlAddFunctionTable / RtlDeleteFunctionTable (dynamic tables; no image for RtlPcToFileHeader).
bool add_function_table(os::VirtualMemoryArena& arena, uint64_t table_va, uint32_t count, uint64_t base);
bool delete_function_table(uint64_t table_va);

// RtlLookupFunctionEntry: guest VA of the RUNTIME_FUNCTION covering `pc` (indirect entries
// followed), or 0 for leaf code / unknown addresses. `image_base` receives the table's base.
uint64_t lookup_function_entry(os::VirtualMemoryArena& arena, uint64_t pc, uint64_t* image_base);

// RtlPcToFileHeader: base of the registered image containing pc, or 0.
uint64_t pc_to_image_base(uint64_t pc);

// ---------------------------------------------------------------------------
// Contexts and virtual unwinding
// ---------------------------------------------------------------------------

// Context of the guest code that just called a thunk (rsp points at its return address): rip =
// the return address, rsp past it, the other registers, flags, MXCSR and xmm from the state.
bool capture_caller_context(os::VirtualMemoryArena& arena, const GuestState& state, GuestContext& out);

// Writes every register of `context` except rip into the state (gpr, flags from EFlags, MXCSR,
// xmm0..15).
void apply_context(const GuestContext& context, GuestState& state);

// RtlVirtualUnwind over a host copy of the context. Returns the language handler (guest VA) of
// the frame for `handler_type` (kUnwFlagEHandler / kUnwFlagUHandler), or 0. `ok` (optional)
// reports whether the unwind data and stack could be read; on failure the context is left as far
// as it got.
uint64_t virtual_unwind(os::VirtualMemoryArena& arena,
                        uint32_t handler_type,
                        uint64_t image_base,
                        uint64_t control_pc,
                        uint64_t function_entry,
                        GuestContext& context,
                        uint64_t* handler_data,
                        uint64_t* establisher_frame,
                        GuestContextPointers* pointers = nullptr,
                        bool* ok = nullptr);

// ---------------------------------------------------------------------------
// Exceptions
// ---------------------------------------------------------------------------

// What a restore did.
enum class RestoreResult {
    InPlace,  // the target is the instruction after the calling import: the state now holds the
              // restored registers and the thunk just returns (rsp already restored)
    Pending,  // GuestState::unwind_pending is set; every frame returns until the owner resumes
};

// RaiseException from the guest code that called the thunk whose entry rsp is the state's rsp:
// dispatches `record` (flags, parameters; address/record fields are filled in) starting at the
// caller's frame. Returns InPlace when a handler continued execution at the raise site, Pending
// when a handler unwound elsewhere. An unhandled exception goes to the top-level filter
// (SetUnhandledExceptionFilter), then to the fault handler (FaultKind::UnhandledException).
RestoreResult raise_exception(os::VirtualMemoryArena& arena, GuestState& state, const GuestExceptionRecord& record);

// RtlUnwindEx called by guest code (state rsp at the thunk's return address).
RestoreResult unwind(os::VirtualMemoryArena& arena,
                     GuestState& state,
                     uint64_t target_frame,
                     uint64_t target_ip,
                     uint64_t exception_record,
                     uint64_t return_value,
                     uint64_t original_context,
                     uint64_t history_table);

// RtlRestoreContext called by guest code (state rsp at the thunk's return address). Handles
// STATUS_LONGJUMP (registers from the jump buffer in information[0]) and
// STATUS_UNWIND_CONSOLIDATE (calls the callback in information[0] with the record; it returns
// the rip to continue at).
RestoreResult restore_context(os::VirtualMemoryArena& arena,
                              GuestState& state,
                              GuestContext context,
                              uint64_t exception_record);

// SetUnhandledExceptionFilter (guest function VA, 0 = none); returns the previous filter.
uint64_t set_unhandled_exception_filter(uint64_t filter);

// ---------------------------------------------------------------------------
// Thunks and binding
// ---------------------------------------------------------------------------

struct ExceptionExport {
    const char* module;  // real exporting DLL
    const char* name;
    HostThunkFn thunk;
};

// Every exported thunk (KERNEL32.dll and ntdll.dll entries).
std::span<const ExceptionExport> exception_exports();

// Thunk for an import, API sets resolved through O7's registry once register_exception_exports
// ran (direct module names work without it). nullptr if the import is not one of these APIs.
HostThunkFn exception_thunk_for(std::string_view module, std::string_view name);

// Publishes the thunks in O7's module registry and gives each a synthetic guest address
// (idempotent; returns the number of exports).
size_t register_exception_exports();

// Drops all function tables and the unhandled-exception filter. Tests only.
void reset_guest_exceptions();

}  // namespace playgta6::guest

// ---------------------------------------------------------------------------
// C-ABI entry points (guest memory = playgta6::os::win32::get_vmem_arena())
// ---------------------------------------------------------------------------

extern "C" {

int32_t pg6_guest_register_image_function_table(uint64_t image_base,
                                                uint64_t image_size,
                                                uint64_t table_va,
                                                uint32_t count);
int32_t pg6_guest_register_image_from_headers(uint64_t image_base);
size_t pg6_guest_register_exception_exports(void);
Pg6HostThunkFn pg6_guest_exception_thunk_for(const char* module, const char* name);

}  // extern "C"
