#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "guest/guest_runtime.h"

// Guest-callable entries for every shim export (R18).
//
// Recompiled code reaches a host shim through a marshaling thunk. The import bridge generates one
// per IAT slot, so an API the guest resolves dynamically (GetModuleHandle + GetProcAddress) used
// to have a host proc address in the module registry but no guest-callable entry. The generated
// export table (`playgta6-recomp export-thunks`, `pg6_export_thunks.cpp`; `build-program` links
// it once per program and lists it in `Pg6Program::exports`) holds a thunk for every export that
// is bindable under the static-import rules, and the reason for each one that is not. This file
// is the runtime side:
//
//   register_export_thunks(table)   for every row, looks up the proc address the module registry
//                                   holds for (dll, name) and gives its thunk a synthetic guest
//                                   address with register_host_callable(thunk, name, proc), so
//                                   guest_callable_for_proc(proc) - and with it the guest
//                                   GetProcAddress (guest_callbacks.h) - finds it. A proc that
//                                   already has a guest-callable entry (xgameruntime provider,
//                                   exception thunks, d3d12_xs, ...) keeps it: one guest address
//                                   per shim however many DLL names reach it. A row without a
//                                   thunk records its reason for the proc, so GetProcAddress can
//                                   say why it returns NULL (unavailable_reason_for_proc).
//
// Addresses: a function pointer from GetProcAddress and the IAT slot value of a static import of
// the same shim are interchangeable for the guest (both dispatch to a thunk that marshals the
// same shim call) but not equal: bind_import_slots gives each slot its own synthetic address and
// slots map back to their import for diagnostics. The table needs at most one address per
// distinct shim (about 650 of the 2^20 synthetic addresses for the whole catalogue).

// Layout of the generated `Pg6ExportThunk` (pg6_export_thunks.h); the program descriptor
// static_asserts the two match.
struct Pg6BootExport {
    const char* dll;
    const char* name;
    uint32_t ordinal;  // 0 = none
    uint32_t flags;    // kPg6BootExport*
    const char* shim;  // shim or guest-state thunk symbol, null when there is none
    Pg6HostThunkFn thunk;  // null: no guest-callable entry
    const char* reason;    // set when thunk is null: "<code>: <detail>"
};

inline constexpr uint32_t kPg6BootExportGenerated = 1u;   // generated marshaling thunk
inline constexpr uint32_t kPg6BootExportGuestState = 2u;  // pg6_gthunk_* guest-state thunk

namespace playgta6::guest {

using BootExport = Pg6BootExport;

struct ExportThunkStats {
    size_t rows = 0;
    size_t registered = 0;   // rows that got a new synthetic address
    size_t shared = 0;       // rows whose proc already had a guest-callable entry
    size_t unavailable = 0;  // rows without a thunk (reason recorded for the proc)
    size_t unregistered = 0; // rows the module registry has no export for (skipped)
    size_t failed = 0;       // rows left out because the synthetic address table is full
};

// Registers the table (see above). Idempotent: running it again registers nothing new. The
// module registry must already hold the shim exports (os::win32::register_all_shim_exports()).
// Preserves the calling thread's last error.
ExportThunkStats register_export_thunks(std::span<const BootExport> table);

// The reason an export's proc has no guest-callable entry ("denied: pg6_crt_remove: denied by the
// shim audit: ..."), or nullptr when no row recorded one (the table does not list the export, or
// it has an entry). The pointer stays valid until the next register_export_thunks /
// reset_export_thunks; copy it if needed.
const char* unavailable_reason_for_proc(uint64_t proc_address);

// Forgets recorded reasons. Only for tests (pairs with reset_guest_runtime()).
void reset_export_thunks();

}  // namespace playgta6::guest

extern "C" {

// C-ABI wrapper of register_export_thunks; returns the number of rows that got a new address.
size_t pg6_guest_register_export_thunks(const Pg6BootExport* table, size_t count);

}  // extern "C"
