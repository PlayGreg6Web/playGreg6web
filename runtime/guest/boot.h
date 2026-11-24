#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "abi/playgta6_abi.h"
#include "guest/export_thunks.h"
#include "guest/guest_runtime.h"
#include "guest/memory_model.h"
#include "os/image_loader.h"
#include "os/vmem.h"

// Guest boot sequence: starts a recompiled guest executable the way the Windows loader starts a
// statically linked x64 EXE, then runs it until it exits or hits the first thing the runtime does
// not support, and describes that point in one structured report.
//
// A boot takes a list of guest modules (BootConfig::modules): modules[0] is the executable, any
// further entries are guest DLLs loaded with it (guest/program_loader.h builds the list from a
// recompiled program). Each module brings its PGIM image, lifted-function table(s) and import
// table.
//
// Sequence (run_boot)
// -------------------
//  1. Load every PGIM image (os/image_loader.h) at its preferred base; modules[0] is registered as
//     the main module (O7 module registry). Guest DLLs are registered with O7 by the program
//     loader through BootConfig::after_load; imports of one module bound to another module's
//     exports are pre-filled in the images at build time.
//  2. Register the lifted-function tables and bind every IAT slot (guest_runtime.h): slots whose
//     import resolved to a runtime shim get the generated thunk; unresolved slots get the boot's
//     own thunk, which reports `DLL!symbol` as an "unresolved_import" fault (or, in survey mode,
//     logs it and returns 0, see below). The startup summary counts functions and imports (bound
//     and unresolved, per DLL).
//  3. Create the process block (PEB + RTL_USER_PROCESS_PARAMETERS) and the main guest thread:
//     guest stack, TEB, implicit TLS (layouts below). gs_base = TEB, mxcsr = 0x1F80.
//  4. Implicit TLS from each module's PE TLS directory (data directory 9 of the mapped PE headers,
//     or BootModule::tls_directory_rva): every module with a TLS directory gets the next TLS index
//     in module order (the executable gets 0, like on Windows), written to its AddressOfIndex. Per
//     thread each such module gets a block = template [StartAddressOfRawData, EndAddressOfRawData)
//     + SizeOfZeroFill zero bytes, aligned as the directory's Characteristics (IMAGE_SCN_ALIGN_*,
//     at least 16) ask; TEB.ThreadLocalStoragePointer points at the vector of block pointers, so
//     the inline MSVC sequence `mov rax, gs:[0x58]; mov rax, [rax + idx*8]` reaches the block.
//  5. Run the TLS callbacks (AddressOfCallBacks, null-terminated) with
//     (image_base, DLL_PROCESS_ATTACH, nullptr) - the DLLs' first (modules[1..] in order, each
//     followed by its DllMain when BootConfig::call_dll_entry_points is set), the executable's
//     last - then call the executable's entry point with
//     rcx = PEB (what RtlUserThreadStart passes) through call_guest.
//  6. The run ends when the entry point returns (exit code = eax; the guest exit handlers run
//     first), when the guest calls the CRT exit/_exit (guest/guest_crt_exit.h runs the guest
//     exit handlers, then the ucrt_bridge exit hook the boot installs stops the run;
//     pg6_guest_boot_request_exit is the same path for other shims, and TerminateProcess of the
//     current process takes it through win32::set_process_exit_handler), on the first fault, or
//     at a run bound. Every outcome produces one JSON report plus a human-readable summary and a distinct
//     BootStatus; the host process is never aborted by a guest fault.
//
// Progress is logged through os::log in the line format of W40's boot tracer
// (web/telemetry/boot_tracer.js): `[boot] PGIM_LOAD ...`, `[boot] TLS_INIT ...`,
// `[boot] ENTRY_CALL ...` for milestones, the stopping fault as `[trap] <TRAP_KIND> at 0x<va> ...`
// and survey hits as `[import] unresolved import DLL!symbol ...`.
//
// Faults and the report
// ---------------------
// While a boot runs it owns the guest runtime's fault handler and dispatch observer. Faults
// (FaultKind of guest_runtime.h, unresolved imports, CRT abort/terminate, run bounds, and host
// faults reported by the harness through report_host_fault) stop the run by unwinding out of the
// guest with a C++ exception (builds without exception support abort after recording the
// report). The report records: fault kind, guest VA, the containing lifted function (nearest
// registered entry at or below the VA + offset), DLL!symbol for imports, the 16 GPRs plus rip,
// gs_base and mxcsr, a guest backtrace, counters (indirect dispatches, import calls by name with
// counts, the last `recent_import_count` imports in order) and the startup summary.
//
// Guest backtrace: each loaded image's .pdata is registered with R11's unwinder
// (guest_exceptions.h, register_image_from_headers), and the backtrace walks the guest stack with
// RtlVirtualUnwind from the fault ("rtl-virtual-unwind", or "rtl-virtual-unwind-partial" when the
// walk stops before call_guest's return sentinel). For code without .pdata the boot scans the
// guest stack from rsp to the stack top and reports every 8-byte word that is the address right
// after a `call` instruction inside a registered lifted function ("stack-scan-call-verified": the
// bytes before it decode as `call rel32` or `call r/m64`; "stack-scan-unverified" when the
// image's executable bytes are omitted). A scan can report stale return addresses left in dead
// stack slots; it never misses a live frame below the stack top.
//
// Runtime services wired by the boot: register_all_shim_exports (O7/O11, incl. the O16
// xgameruntime.dll provider), R11's exception exports (imports the bridge left unresolved but R11
// implements as runtime thunks - RaiseException, RtlVirtualUnwind, ... - are bound to them), the
// guest command line (GetCommandLineA/W), GetCurrentThreadStackLimits = the guest stack, O15's
// TerminateProcess exit handler, and O16's guest memory (= the boot's arena).
// Optionally (BootConfig::capture_host_stack) host return addresses are mapped back to lifted
// functions too ("host_frames"): with native lifted code every guest call is a host call, so
// this chain is exact up to inlining.
//
// Registers in a report are those in GuestState at the fault. Helper calls write every live
// register back first, so they are exact for faults raised by the runtime helpers; for host
// faults inside lifted code they are as of the last call boundary.
//
// Survey mode (diagnostic, UNSAFE)
// --------------------------------
// BootConfig::survey makes calls to unresolved imports log once, return
// BootConfig::survey_return (default 0) in rax, pop the return address and continue, so one run lists many missing imports in the order they are first called
// (report "survey.unresolved_called"). The guest then runs on made-up results: everything after
// the first unresolved call is suspect. Off by default. Bound the run with max_dispatches and/or
// max_wall_ms (checked on every indirect dispatch; a loop without calls is not interrupted - the
// native harness adds a watchdog for that).
//
// JSON report (schema "playgta6-boot-report", schema_version 1)
// -------------------------------------------------------------
// Top-level keys in order: schema, schema_version, status (BootStatus name), status_code,
// exit_code (number or null), phase (setup step or "tls_callback"/"entry"), message,
// milestones ([{milestone, elapsed_ms}]: the W40 BOOT_MILESTONES reached - PGIM_LOAD, TLS_INIT,
// ENTRY_CALL), fault (null or {kind, trap_kind (W40 TRAP_KIND name or null), va, detail,
// function {entry, offset} | null, import {dll, symbol, slot_rva, delay_load, module} | null,
// host_address, host_pc}), registers ({rax..r15, rip, gs_base,
// fs_base, mxcsr} or null), backtrace {method, frames [{stack_address, return_address, function,
// offset}]}, host_frames ([{host_pc, function, offset}] or null), counters {dispatches,
// import_calls, imports [{dll, symbol, resolved, count}] (sorted by count, descending), recent
// [{seq, dll, symbol}]}, survey {enabled, unresolved_called [{dll, symbol, first_seq, count}]},
// startup {image_base, entry_point (of modules[0]), modules [{name, image_base, image_size,
// entry_point, rebased, lifted_tables, imports {total, bound, unresolved}, tls {present, index,
// template_start, template_size, zero_fill, alignment, index_va, callbacks [va], callbacks_run}}],
// lifted_tables, functions_registered, imports {total, bound, unresolved, by_dll [{dll, bound,
// unresolved}]}, process {peb, process_parameters}, main_thread {teb, stack_base, stack_limit,
// tls_vector, tls_block}}, memory {model, cpu_window {maps, neighbour_pairs,
// neighbour_pairs_contiguous, neighbour_pairs_reversed, run_kept, run_moves, run_breaks,
// run_relocations, run_bytes_moved, regions [{base, size, translated, mapped_slots, aliased_slots, runs,
// longest_run, adjacent_pairs, aliased_pairs, contiguous_pairs}]} (os/xbox_scarlett.h
// XMemWindowContiguity), window {enabled, mapped_slots,
// contiguous_pairs, split_pairs, map_calls} (os/guest_window.h)}, elapsed_ms. Addresses are "0x..." strings (null when absent); counts
// are numbers. Fault kinds: trap, unimplemented, unimplemented_slot, divide_error, unknown_target,
// stack_imbalance, bad_stack, unresolved_import, crt_abort, crt_terminate, crt_invalid_parameter,
// crt_purecall, cxx_throw, longjmp, dispatch_limit, wall_time_limit, and the harness's host_*
// kinds.
//
// Guest TEB / PEB layout (x64 Windows, written by the boot; offsets checked below)
// --------------------------------------------------------------------------------
// TEB (0x2000 bytes reserved, zero except):
//   0x000 NtTib.ExceptionList (0)            0x008 NtTib.StackBase (stack top)
//   0x010 NtTib.StackLimit (lowest committed) 0x020 NtTib.FiberData (0)
//   0x030 NtTib.Self (TEB VA)                0x040 ClientId.UniqueProcess
//   0x048 ClientId.UniqueThread              0x058 ThreadLocalStoragePointer (TLS vector or 0)
//   0x060 ProcessEnvironmentBlock            0x068 LastErrorValue (shared with GetLastError)
//   0x1250 LastStatusValue (0)               0x1478 DeallocationStack (stack reservation base)
//   0x1480 TlsSlots[64] (0; TlsAlloc/TlsGetValue keep their own table, O3)
//   0x1780 TlsExpansionSlots (0)
// PEB (one page, zero except):
//   0x002 BeingDebugged (0)                  0x010 ImageBaseAddress
//   0x018 Ldr (null: no PEB_LDR_DATA module list yet; code that walks it sees no modules)
//   0x020 ProcessParameters                  0x030 ProcessHeap (O7 GetProcessHeap handle)
//   0x0B8 NumberOfProcessors                 0x0BC NtGlobalFlag (0)
//   0x118 OSMajorVersion 10, 0x11C OSMinorVersion 0, 0x120 OSBuildNumber, 0x124 OSPlatformId 2
//   0x128 ImageSubsystem (from the PE header when mapped)
// RTL_USER_PROCESS_PARAMETERS (minimal): MaximumLength/Length, Flags = NORMALIZED,
//   0x38 CurrentDirectory.DosPath, 0x60 ImagePathName, 0x70 CommandLine (UTF-16 copies of the
//   config strings), 0x80 Environment (empty block: two NUL characters). No console handles.
// Not modelled: KUSER_SHARED_DATA (0x7FFE0000), PEB_LDR_DATA, activation contexts, fibers.

namespace playgta6::guest::win64 {

inline constexpr uint64_t kTebSize = 0x2000;
inline constexpr uint64_t kPebSize = 0x1000;
inline constexpr uint64_t kTebTlsSlotCount = 64;
inline constexpr uint32_t kDllProcessAttach = 1;
inline constexpr uint32_t kDllThreadAttach = 2;
inline constexpr uint32_t kDllThreadDetach = 3;
inline constexpr uint32_t kDllProcessDetach = 0;

struct NtTib64 {
    uint64_t exception_list;
    uint64_t stack_base;
    uint64_t stack_limit;
    uint64_t sub_system_tib;
    uint64_t fiber_data;
    uint64_t arbitrary_user_pointer;
    uint64_t self;
};

struct Teb64 {
    NtTib64 nt_tib;                         // 0x000
    uint64_t environment_pointer;           // 0x038
    uint64_t unique_process;                // 0x040 ClientId
    uint64_t unique_thread;                 // 0x048
    uint64_t active_rpc_handle;             // 0x050
    uint64_t thread_local_storage_pointer;  // 0x058
    uint64_t process_environment_block;     // 0x060
    uint32_t last_error_value;              // 0x068
    uint32_t count_of_owned_critical_sections;  // 0x06C
    uint8_t reserved_070[0x1250 - 0x070];
    uint32_t last_status_value;             // 0x1250
    uint8_t reserved_1254[0x1478 - 0x1254];
    uint64_t deallocation_stack;            // 0x1478
    uint64_t tls_slots[kTebTlsSlotCount];   // 0x1480
    uint64_t tls_links[2];                  // 0x1680
    uint8_t reserved_1690[0x1780 - 0x1690];
    uint64_t tls_expansion_slots;           // 0x1780
    uint8_t reserved_1788[kTebSize - 0x1788];
};

struct UnicodeString64 {
    uint16_t length;          // bytes, without the terminator
    uint16_t maximum_length;  // bytes, with the terminator
    uint32_t pad;
    uint64_t buffer;
};

struct ProcessParameters64 {
    uint32_t maximum_length;            // 0x00
    uint32_t length;                    // 0x04
    uint32_t flags;                     // 0x08
    uint32_t debug_flags;               // 0x0C
    uint64_t console_handle;            // 0x10
    uint32_t console_flags;             // 0x18
    uint32_t pad_1c;
    uint64_t standard_input;            // 0x20
    uint64_t standard_output;           // 0x28
    uint64_t standard_error;            // 0x30
    UnicodeString64 current_directory;  // 0x38 CurrentDirectory.DosPath
    uint64_t current_directory_handle;  // 0x48
    UnicodeString64 dll_path;           // 0x50
    UnicodeString64 image_path_name;    // 0x60
    UnicodeString64 command_line;       // 0x70
    uint64_t environment;               // 0x80
    uint8_t reserved_088[0x400 - 0x088];
};

inline constexpr uint32_t kProcessParametersNormalized = 1;

struct Peb64 {
    uint8_t inherited_address_space;       // 0x000
    uint8_t read_image_file_exec_options;  // 0x001
    uint8_t being_debugged;                // 0x002
    uint8_t bit_field;                     // 0x003
    uint32_t pad_004;
    uint64_t mutant;                       // 0x008
    uint64_t image_base_address;           // 0x010
    uint64_t ldr;                          // 0x018
    uint64_t process_parameters;           // 0x020
    uint64_t sub_system_data;              // 0x028
    uint64_t process_heap;                 // 0x030
    uint8_t reserved_038[0x0B8 - 0x038];
    uint32_t number_of_processors;         // 0x0B8
    uint32_t nt_global_flag;               // 0x0BC
    uint8_t reserved_0c0[0x118 - 0x0C0];
    uint32_t os_major_version;             // 0x118
    uint32_t os_minor_version;             // 0x11C
    uint16_t os_build_number;              // 0x120
    uint16_t os_csd_version;               // 0x122
    uint32_t os_platform_id;               // 0x124
    uint32_t image_subsystem;              // 0x128
    uint32_t image_subsystem_major_version;  // 0x12C
    uint32_t image_subsystem_minor_version;  // 0x130
    uint8_t reserved_134[kPebSize - 0x134];
};

// IMAGE_TLS_DIRECTORY64 (VAs, relocated with the image).
struct TlsDirectory64 {
    uint64_t start_address_of_raw_data;
    uint64_t end_address_of_raw_data;
    uint64_t address_of_index;
    uint64_t address_of_callbacks;
    uint32_t size_of_zero_fill;
    uint32_t characteristics;
};

static_assert(sizeof(NtTib64) == 0x38);
static_assert(offsetof(Teb64, nt_tib) + offsetof(NtTib64, stack_base) == 0x08);
static_assert(offsetof(Teb64, nt_tib) + offsetof(NtTib64, stack_limit) == 0x10);
static_assert(offsetof(Teb64, nt_tib) + offsetof(NtTib64, fiber_data) == 0x20);
static_assert(offsetof(Teb64, nt_tib) + offsetof(NtTib64, self) == 0x30);
static_assert(offsetof(Teb64, unique_process) == 0x40);
static_assert(offsetof(Teb64, unique_thread) == 0x48);
static_assert(offsetof(Teb64, thread_local_storage_pointer) == 0x58);
static_assert(offsetof(Teb64, process_environment_block) == 0x60);
static_assert(offsetof(Teb64, last_error_value) == 0x68);
static_assert(offsetof(Teb64, last_status_value) == 0x1250);
static_assert(offsetof(Teb64, deallocation_stack) == 0x1478);
static_assert(offsetof(Teb64, tls_slots) == 0x1480);
static_assert(offsetof(Teb64, tls_links) == 0x1680);
static_assert(offsetof(Teb64, tls_expansion_slots) == 0x1780);
static_assert(sizeof(Teb64) == kTebSize);
static_assert(sizeof(UnicodeString64) == 16);
static_assert(offsetof(ProcessParameters64, current_directory) == 0x38);
static_assert(offsetof(ProcessParameters64, dll_path) == 0x50);
static_assert(offsetof(ProcessParameters64, image_path_name) == 0x60);
static_assert(offsetof(ProcessParameters64, command_line) == 0x70);
static_assert(offsetof(ProcessParameters64, environment) == 0x80);
static_assert(offsetof(Peb64, being_debugged) == 0x002);
static_assert(offsetof(Peb64, image_base_address) == 0x010);
static_assert(offsetof(Peb64, ldr) == 0x018);
static_assert(offsetof(Peb64, process_parameters) == 0x020);
static_assert(offsetof(Peb64, process_heap) == 0x030);
static_assert(offsetof(Peb64, number_of_processors) == 0x0B8);
static_assert(offsetof(Peb64, nt_global_flag) == 0x0BC);
static_assert(offsetof(Peb64, os_major_version) == 0x118);
static_assert(offsetof(Peb64, os_build_number) == 0x120);
static_assert(offsetof(Peb64, os_platform_id) == 0x124);
static_assert(offsetof(Peb64, image_subsystem) == 0x128);
static_assert(sizeof(Peb64) == kPebSize);
static_assert(sizeof(TlsDirectory64) == 40);

}  // namespace playgta6::guest::win64

// One import slot of the guest image. Same layout as the import bridge's generated
// `Pg6ImportThunk` (recomp/src/stubs.rs, pg6_import_thunks.h), so its table can be passed as is.
struct Pg6BootImport {
    uint32_t slot_rva;
    uint32_t flags;    // kPg6BootImport* bits
    uint32_t ordinal;  // set when kPg6BootImportByOrdinal
    const char* dll;
    const char* symbol;  // null for ordinal imports
    const char* shim;    // null when unresolved
    Pg6HostThunkFn thunk;
};

inline constexpr uint32_t kPg6BootImportBound = 1u;      // thunk calls a runtime shim
inline constexpr uint32_t kPg6BootImportDelayLoad = 2u;  // slot belongs to a delay-load IAT
inline constexpr uint32_t kPg6BootImportByOrdinal = 4u;

// BootStatus values (also the C-ABI result codes).
enum Pg6GuestBootStatus : int32_t {
    PG6_BOOT_EXITED = 0,         // entry returned or the guest exited; exit_code is valid
    PG6_BOOT_BAD_CONFIG = 1,
    PG6_BOOT_LOAD_FAILED = 2,    // PGIM image could not be loaded
    PG6_BOOT_REGISTER_FAILED = 3,  // lifted tables rejected or import binding failed
    PG6_BOOT_PROCESS_SETUP_FAILED = 4,  // PEB / TLS directory / main thread setup
    PG6_BOOT_FAULT = 5,          // the guest faulted (see the report)
    PG6_BOOT_GUEST_ABORT = 6,    // abort/terminate/purecall/... through the CRT
    PG6_BOOT_LIMIT_REACHED = 7,  // max_dispatches / max_wall_ms
};

// Records of a boot run that the C ABI and the harness read.
struct Pg6GuestBootStats {
    int32_t status;     // Pg6GuestBootStatus
    int32_t exit_code;  // valid when status == PG6_BOOT_EXITED
    uint64_t image_base;
    uint64_t entry_point;
    uint64_t functions_registered;
    uint64_t imports_total;
    uint64_t imports_bound;
    uint64_t imports_unresolved;
    uint64_t tls_callbacks;      // in the image
    uint64_t tls_callbacks_run;  // returned normally
    uint64_t dispatches;         // __playgta6_indirect_call invocations
    uint64_t import_calls;
    uint64_t unresolved_called;  // distinct unresolved imports called (survey mode)
    uint32_t entry_started;      // 1 once the entry point was called
    uint32_t reserved;
};

// A prepared guest thread: CPU state + stack (guest_runtime.h) plus TEB and implicit TLS.
struct Pg6GuestThreadContext {
    Pg6GuestThread thread;
    uint64_t teb;
    uint64_t tls_vector;   // TEB.ThreadLocalStoragePointer (0 if no module has TLS)
    uint64_t tls_block;    // this thread's block of the first module with TLS (0 if none)
    uint64_t block_reservation;  // arena reservation holding TEB, TLS vector and block
    uint32_t thread_id;
    uint32_t reserved;
};

namespace playgta6::guest {

using BootImport = Pg6BootImport;
using GuestThreadContext = Pg6GuestThreadContext;
using BootStats = Pg6GuestBootStats;

enum class BootStatus : int32_t {
    Exited = PG6_BOOT_EXITED,
    BadConfig = PG6_BOOT_BAD_CONFIG,
    LoadFailed = PG6_BOOT_LOAD_FAILED,
    RegisterFailed = PG6_BOOT_REGISTER_FAILED,
    ProcessSetupFailed = PG6_BOOT_PROCESS_SETUP_FAILED,
    Fault = PG6_BOOT_FAULT,
    GuestAbort = PG6_BOOT_GUEST_ABORT,
    LimitReached = PG6_BOOT_LIMIT_REACHED,
};

const char* boot_status_name(BootStatus status);

// Captures host return addresses of the calling thread (innermost first). Returns the count.
using CaptureHostStackFn = size_t (*)(void** frames, size_t max_frames);
using HostFunctionStartFn = uint64_t (*)(uint64_t host_pc);
using OffThreadFaultFn = void (*)();

struct BootModule {
    os::pgim::ImageReader* image = nullptr;  // required
    std::string name;                        // e.g. "game.exe" (empty: "game.exe" / "module<i>.dll")
    std::string path;                        // may be empty
    // This module's lifted functions (the generated `__playgta6_lifted_functions_<module>` table).
    std::vector<std::span<const LiftedFunctionEntry>> lifted_tables;
    std::span<const BootImport> imports;  // this image's IAT slots
    // TLS directory RVA override when the PE headers are not mapped (0 = read the headers).
    uint32_t tls_directory_rva = 0;
    uint32_t tls_directory_size = 0;
};

// Console-mode values (BootConfig::console_mode).
inline constexpr uint32_t kConsoleModeSeriesX = 0x60;
inline constexpr uint32_t kConsoleModeDevKit = 0x64;

struct BootConfig {
    // modules[0] is the executable (entry point, main module, TLS index 0); at least one.
    std::vector<BootModule> modules;
    // Guest memory; nullptr = os::win32::get_vmem_arena() (the arena the shims use).
    os::VirtualMemoryArena* arena = nullptr;
    std::string command_line;       // UTF-8
    std::string working_directory;  // UTF-8
    uint64_t stack_size = 0;  // 0: the executable's SizeOfStackReserve (1 MiB if unavailable)
    bool run_tls_callbacks = true;
    bool call_entry = true;
    bool survey = false;            // see "Survey mode" above
    uint64_t survey_return = 0;     // rax returned by unresolved imports in survey mode
    uint64_t max_dispatches = 0;    // 0 = unbounded
    uint64_t max_wall_ms = 0;       // 0 = unbounded
    size_t recent_import_count = 32;
    size_t max_backtrace_frames = 32;
    CaptureHostStackFn capture_host_stack = nullptr;
    // Record up to this many import calls (name, thread, first five integer arguments) for
    // import_trace_text(); 0 = off.
    size_t import_trace_limit = 0;
    // Address-space events kept in the report's address_space.log (all are counted).
    size_t arena_log_limit = 4096;
    // Emulated console type: the console-mode value the title reads at startup (REG_DWORD
    // "ConsoleMode" under HKLM\Software\Microsoft\Windows NT\CurrentVersion\OEM), seeded into the
    // runtime registry before the entry point runs. 0 leaves the registry untouched.
    uint32_t console_mode = kConsoleModeSeriesX;
    // Emulated console's Xbox Live sandbox (environment configuration next to the console mode):
    // what XSystemGetXboxLiveSandboxId reports. A retail console reports "RETAIL"; empty = no
    // sandbox (E_NOTIMPL). The runtime stays offline either way: no user, network or store.
    std::string sandbox_id = "RETAIL";
    // Start address of the host function containing a code address (0 if unknown), e.g. from the
    // host's unwind tables; makes host-frame -> lifted-function mapping exact.
    HostFunctionStartFn host_function_start = nullptr;
    // A guest fault on a thread other than the boot thread (a CreateThread worker, a pool
    // callback) cannot unwind into the boot. With this hook the boot records it like a boot-thread
    // fault (status Fault, fault record with that thread's registers, finished report) and then
    // calls the hook, which must not return (the harness writes its outputs and terminates).
    // nullptr: such faults go to the default fault handler (log and abort).
    OffThreadFaultFn off_thread_fault = nullptr;
    bool log_progress = true;       // phase lines through os::log
    // The generated export table (`pg6_export_thunks`, guest/export_thunks.h): guest-callable
    // entries for every shim export, registered after the IAT slots are bound so that
    // GetProcAddress finds them. Empty = none (exports without a static import then have no
    // guest-callable entry).
    std::span<const Pg6BootExport> export_thunks;
    // Runs once every image is mapped, its lifted tables are registered and its IAT slots are
    // bound, before the process block is created (R12's program loader registers the guest DLLs,
    // their exports and function tables here). Returning false ends the boot with
    // RegisterFailed and `error` as the message.
    std::function<bool(os::VirtualMemoryArena& arena, std::string& error)> after_load;
    // Call every DLL's entry point (modules[1..], in order, each right after its own TLS
    // callbacks) as DllMain(image_base, DLL_PROCESS_ATTACH, nullptr) before the executable's
    // entry point. A DllMain returning FALSE ends the boot with ProcessSetupFailed.
    bool call_dll_entry_points = false;
    // How the lifted code translates guest addresses (guest/memory_model.h; the program
    // descriptor's Pg6Program::memory). Zero = flat, the default. The boot sets up what the model
    // needs before loading any image and ends with BadConfig if this runtime cannot serve it.
    Pg6GuestMemoryModel memory{};
};

struct BootResult {
    BootStatus status = BootStatus::BadConfig;
    BootStats stats{};
    std::string report_json;  // always set
    std::string summary;      // human-readable, always set
};

// Runs the boot sequence on the calling host thread. Only one boot may run at a time per process;
// a previous boot's state is released first (see reset_boot).
BootResult run_boot(const BootConfig& config);

// Releases everything the last boot created (image, PEB, threads), restores the fault handler,
// the dispatch observer and the CRT exit hook, and resets the guest runtime registries
// (reset_guest_runtime). Not safe while guest code runs.
void reset_boot();

// Allocates a TEB + implicit TLS block (template copied, zero-fill zeroed, vector in
// TEB.ThreadLocalStoragePointer) and a guest stack for a new guest thread of the booted process.
// Guest threads that O16 (com_bridge.h) creates lazily for host threads - CreateThread start
// routines, task-queue and pool callbacks - get the same TEB/TLS setup automatically: from the
// main thread's setup until reset_boot the boot owns com::set_guest_thread_init_hook, which also
// binds the thread's last-error slot and runs the TLS callbacks with DLL_THREAD_ATTACH. `thread_id` 0 takes os::win32::GetCurrentThreadId(). Requires a boot whose
// process setup succeeded (the process block stays alive after run_boot returns).
bool prepare_guest_thread(uint64_t stack_size, uint32_t thread_id, GuestThreadContext& out);
void release_guest_thread(GuestThreadContext& context);

// Binds `context` to the calling host thread: GetLastError/SetLastError use its TEB slot and
// fault reports use its stack. nullptr unbinds. Returns the previous context.
GuestThreadContext* activate_guest_thread(GuestThreadContext* context);

// Runs every module's TLS callbacks (DLLs first, the executable last) for `reason` on `context`
// (e.g. kDllThreadAttach for a new thread). Returns the number that ran.
size_t run_tls_callbacks(GuestThreadContext& context, uint32_t reason);

// Ends the running boot as if the guest exited with `exit_code` (never returns while a boot runs
// on this thread; returns false otherwise).
bool request_exit(int32_t exit_code);

// Host-level fault inside guest code (access violation, stack overflow), reported by the native
// harness from its exception/signal handler. Builds the report of the boot running on this
// thread ("host_access_violation" etc.; `host_pc` is mapped to a lifted function), stores it as
// the last report and returns false if no boot is active. The harness then writes the report and
// terminates the process: execution cannot resume after a host fault.
bool report_host_fault(const char* kind, uint64_t host_address, uint64_t host_pc);

// Report / summary of the last boot (also after report_host_fault).
std::string last_report_json();
std::string last_summary();

// The import trace of the last boot (BootConfig::import_trace_limit): one line per import call,
// "seq thread dll!symbol arg0 arg1 arg2 arg3 arg4" (hex arguments: rcx rdx r8 r9 [rsp+0x28]).
std::string import_trace_text();

// Dispatched calls and import calls of the running (or last) boot so far; safe from any thread.
struct LiveCounters {
    uint64_t dispatches = 0;
    uint64_t import_calls = 0;
    uint64_t arena_events = 0;  // reserve / commit / release events of the boot arena
};
LiveCounters live_counters();

// Lifted function containing a host code address (exact only for natively compiled lifted code:
// nearest lifted function pointer at or below `host_pc`). 0 if unknown.
uint64_t guest_function_for_host_pc(uint64_t host_pc, uint64_t* offset = nullptr);

}  // namespace playgta6::guest

// ---------------------------------------------------------------------------
// C ABI (guest memory = os::win32::get_vmem_arena())
// ---------------------------------------------------------------------------

extern "C" {

typedef int32_t (*Pg6BootReadFn)(void* context, uint64_t offset, void* dst, uint64_t len);
typedef size_t (*Pg6BootCaptureHostStackFn)(void** frames, size_t max_frames);

struct Pg6GuestBootModule {
    // Image: either an in-memory container (image_bytes/image_size) or a reader callback.
    const uint8_t* image_bytes;
    uint64_t image_size;
    Pg6BootReadFn image_read;  // used when image_bytes is null
    void* image_read_context;
    const char* name;  // may be null
    const char* path;  // may be null
    const Pg6LiftedFunctionEntry* const* lifted_tables;
    const size_t* lifted_table_counts;
    size_t lifted_table_count;
    const Pg6BootImport* imports;
    size_t import_count;
};

struct Pg6GuestBootConfig {
    const struct Pg6GuestBootModule* modules;  // modules[0] = the executable
    size_t module_count;
    const char* command_line;       // may be null
    const char* working_directory;  // may be null
    uint64_t stack_size;
    uint64_t max_dispatches;
    uint64_t max_wall_ms;
    uint32_t survey;            // nonzero = survey mode
    uint32_t skip_tls_callbacks;
    uint32_t skip_entry;
    uint32_t quiet;
    Pg6BootCaptureHostStackFn capture_host_stack;
};

// Runs the boot; fills `stats` (may be null). Returns the Pg6GuestBootStatus.
int32_t pg6_guest_boot_run(const struct Pg6GuestBootConfig* config, struct Pg6GuestBootStats* stats);
// Copies the last report / summary (NUL-terminated, truncated to size); returns the full length.
size_t pg6_guest_boot_report_json(char* buffer, size_t size);
size_t pg6_guest_boot_summary(char* buffer, size_t size);
void pg6_guest_boot_reset(void);
int32_t pg6_guest_boot_prepare_thread(struct Pg6GuestThreadContext* context, uint64_t stack_size);
void pg6_guest_boot_release_thread(struct Pg6GuestThreadContext* context);
struct Pg6GuestThreadContext* pg6_guest_boot_activate_thread(struct Pg6GuestThreadContext* context);
int32_t pg6_guest_boot_request_exit(int32_t exit_code);
int32_t pg6_guest_boot_report_host_fault(const char* kind, uint64_t host_address, uint64_t host_pc);

}  // extern "C"
