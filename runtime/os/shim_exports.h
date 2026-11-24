#pragma once

#include <cstddef>
#include <cstdint>

#include "os/win32_loader.h"

// Runtime shim export registration table (O11).
//
// Publishes every C-ABI shim of the runtime (pg6_win32_* and pg6_gdk_*) in the module registry of
// win32_loader.h, under the system DLL that exports the API on a real Xbox/Windows install and
// under the real exported name, so a guest GetProcAddress / LoadLibrary / import bind can find it:
//
//   KERNEL32.dll                                   win32_kernel.h + win32_loader.h +
//                                                  win32_sync.h shims
//   ADVAPI32.dll                                   ETW Event* (reached via api-ms-win-eventing-*),
//                                                  in-memory registry reads (win32_compat.h)
//   bcrypt.dll, USER32.dll, combase.dll            RNG-only bcrypt, window fake (ext-ms-win-rtcore-
//                                                  ntuser-*), COM helpers (win32_compat.h)
//   WS2_32.dll, WINHTTP.dll                        xbox_offline.h
//   xgameruntime.dll                               XGameRuntime, XTaskQueue, XAsync, XUser,
//                                                  XSystem, XNetworking, XStore
//   Microsoft.Xbox.Services.141.GDK.C.Thunks.dll   Xbl* (XSAPI C API); also reachable as
//   (and Microsoft.Xbox.Services.GDK.C.Thunks.dll) the unversioned alias
//   d3d12_xs.dll                                   D3D12XboxCreateDevice, D3D12SerializeVersioned-
//                                                  RootSignature, D3D12ConfigureVirtualMemoryX
//                                                  (pg6_d3d12_*, gpu/d3d12_xs_exports.h)
//
// The name <-> symbol mapping lives in exactly one place, the PG6_SHIM_EXPORTS table in
// shim_exports.cpp; tests/test_shim_exports.py checks it against the headers (every declared shim
// is registered or explicitly excluded) and against the R5 import-bridge name rule.
//
// What a caller gets
// ------------------
// GetProcAddress returns the Pg6ProcAddress produced by proc_address(fn): on the host build the
// address of the C-ABI shim itself, on Wasm64 its function-table index. That is a *host* function
// with the host calling convention and host-typed arguments (host pointers, not guest VAs, for
// pointer parameters such as `uint32_t*`). Guest (recompiled x86-64) code cannot call it directly:
// the dispatcher / import thunks (O9, runtime/guest) must marshal guest registers and stack into
// the shim's C signature, translate guest VAs to host pointers, and put the result back in RAX.
// Until then the address is a lookup key for that dispatcher and callable from host code only.
//
// register_all_shim_exports() also registers the xgameruntime.dll provider exports
// (xgameruntime_provider.h: QueryApiImpl, InitializeApiImpl*, XErrorReport, UninitializeApiImpl),
// which are guest-callable thunks rather than host C-ABI shims and so are not table entries.
//
// Uncertain placements: XGameRuntimeIsInitialized is not a documented GDK export (the shim backs
// runtime-internal state checks) and is published under xgameruntime.dll only so no shim is
// hidden. The XSAPI module name carries a toolset version in the GDK (141 here); the unversioned
// alias is registered as well. XStore/XNetworking are placed in xgameruntime.dll as in the GDK
// import library. Winsock exports are lowercase (socket, bind, ...) like the real ws2_32.

struct Pg6ShimExport {
    const char* module;  // canonical-cased virtual module name, e.g. "KERNEL32.dll"
    const char* name;    // exported API name, e.g. "CreateFileW"
    const char* symbol;  // C-ABI shim symbol, e.g. "pg6_win32_create_file_w"
    Pg6ProcAddress address;
};

namespace playgta6::os::win32 {

// Registers all shim exports plus the extra API-set rules they need. Idempotent and thread-safe
// (runs once; later calls return the same count immediately). Returns the number of table entries
// registered (alias module excluded).
size_t register_all_shim_exports();

// The registration table (pointer to static storage, *count entries).
const Pg6ShimExport* shim_export_table(size_t* count);

}  // namespace playgta6::os::win32

extern "C" {

size_t pg6_loader_register_all_shim_exports(void);
const Pg6ShimExport* pg6_loader_shim_export_table(size_t* count);

}  // extern "C"
