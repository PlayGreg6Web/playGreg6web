#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>

#include "os/win32_kernel.h"

// Win32 heap, module registry and GetProcAddress resolver.
//
// Heap memory model
// -----------------
// Heap blocks are carved out of VirtualMemoryArena reservations (the same arena that backs
// VirtualAlloc, see win32_kernel.h). Like VirtualAlloc, every heap pointer handed out or
// accepted here is a 64-bit *guest virtual address* (uint64_t), not a host pointer: guest code
// only ever sees addresses inside the arena's guest VA layout, and the runtime reaches the bytes
// via VirtualMemoryArena::host_ptr/read/write. A heap HANDLE is the guest VA of the heap's first
// segment (as on Windows, where a heap handle is the heap's base address), so it can never collide
// with kernel object handles from win32_kernel.cpp.
//
// Each heap owns arena segments (committed in one piece) and keeps all bookkeeping host-side:
// guest buffer overruns cannot corrupt allocator
// metadata, and HeapFree/HeapSize/HeapReAlloc on foreign or already-freed addresses are detected
// exactly and fail with a Win32 error instead of being undefined behaviour. Blocks are 16-byte
// aligned (Win64 MEMORY_ALLOCATION_ALIGNMENT). A heap binds to the arena returned by
// get_vmem_arena() when it is created; the process heap is created on the first GetProcessHeap().
//
// Module registry
// ---------------
// HMODULE values are guest VAs as well: the main image's handle is its image base (registered by
// the loader), virtual modules standing in for system DLLs (KERNEL32.dll, xgameruntime.dll,
// d3d12_xs.dll, ...) get synthetic, never-mapped addresses starting at kVirtualModuleBase. Module
// names match like Windows: ASCII case-insensitive, directory prefix stripped, ".dll" appended when
// the name has no extension (a trailing '.' means "no extension"). API-set contract names
// (api-ms-win-*, ext-ms-win-*) resolve to their host module through a longest-prefix rule table.
//
// Export addresses are uint64_t "proc addresses": on the host build a function pointer value, on
// Wasm64 the function-table index Emscripten uses for function pointers.

using Pg6ProcAddress = uint64_t;

// One export to publish: name may be nullptr/"" for ordinal-only exports, ordinal 0 = none.
struct Pg6ExportEntry {
    const char* name = nullptr;
    uint32_t ordinal = 0;
    Pg6ProcAddress address = 0;
};

namespace playgta6::os::win32 {

using ProcAddress = Pg6ProcAddress;
using ExportEntry = Pg6ExportEntry;

// HeapCreate / HeapAlloc / HeapReAlloc / HeapFree flags.
inline constexpr uint32_t HEAP_NO_SERIALIZE = 0x00000001u;  // Accepted; heaps always lock.
inline constexpr uint32_t HEAP_GROWABLE = 0x00000002u;
inline constexpr uint32_t HEAP_GENERATE_EXCEPTIONS = 0x00000004u;  // Accepted; never raises.
inline constexpr uint32_t HEAP_ZERO_MEMORY = 0x00000008u;
inline constexpr uint32_t HEAP_REALLOC_IN_PLACE_ONLY = 0x00000010u;
inline constexpr uint32_t HEAP_CREATE_ENABLE_EXECUTE = 0x00040000u;

// HeapSize failure value ((SIZE_T)-1).
inline constexpr uint64_t kHeapSizeError = ~uint64_t{0};
// Largest single heap request accepted (above any Wasm64 memory Emscripten can grow to).
inline constexpr uint64_t kMaxHeapRequest = uint64_t{1} << 36;
// Growable heaps grow in segments of at least this size.
inline constexpr uint64_t kHeapSegmentSize = uint64_t{1} << 20;

// Synthetic HMODULE range for virtual (runtime-implemented) modules.
inline constexpr uint64_t kVirtualModuleBase = 0x00007FF800000000ULL;
inline constexpr uint64_t kVirtualModuleStride = 0x0000000000100000ULL;

// Loader-specific Win32 error codes (the common ones live in win32_kernel.h).
inline constexpr uint32_t ERROR_INSUFFICIENT_BUFFER = 122u;
inline constexpr uint32_t ERROR_MOD_NOT_FOUND = 126u;
inline constexpr uint32_t ERROR_PROC_NOT_FOUND = 127u;

// ---------------------------------------------------------------------------
// Heap shims
// ---------------------------------------------------------------------------

Handle GetProcessHeap();
Handle HeapCreate(uint32_t options, uint64_t initial_size, uint64_t maximum_size);
bool HeapDestroy(Handle heap);
uint64_t HeapAlloc(Handle heap, uint32_t flags, uint64_t bytes);
uint64_t HeapReAlloc(Handle heap, uint32_t flags, uint64_t mem, uint64_t bytes);
bool HeapFree(Handle heap, uint32_t flags, uint64_t mem);
uint64_t HeapSize(Handle heap, uint32_t flags, uint64_t mem);
bool HeapValidate(Handle heap, uint32_t flags, uint64_t mem);

// ---------------------------------------------------------------------------
// Module shims
// ---------------------------------------------------------------------------

Handle GetModuleHandleA(const char* module_name);
Handle GetModuleHandleW(const char16_t* module_name);
Handle LoadLibraryA(const char* file_name);
Handle LoadLibraryW(const char16_t* file_name);
Handle LoadLibraryExA(const char* file_name, Handle file, uint32_t flags);
Handle LoadLibraryExW(const char16_t* file_name, Handle file, uint32_t flags);
bool FreeLibrary(Handle module);
uint32_t GetModuleFileNameA(Handle module, char* file_name, uint32_t size);
uint32_t GetModuleFileNameW(Handle module, char16_t* file_name, uint32_t size);

// Win32 semantics: a proc_name whose value has HIWORD == 0 is an ordinal (MAKEINTRESOURCE).
// On Wasm64, Emscripten places static data (string literals) below 64 KiB by default, so such a
// host pointer would be read as an ordinal: runtime-internal callers and import thunks (which can
// test HIWORD on the *guest* value, guest VAs are never below 64 KiB) should use
// GetProcAddressByName / GetProcAddressByOrdinal (pg6_loader_get_proc_address_by_*) instead.
ProcAddress GetProcAddress(Handle module, const char* proc_name);
ProcAddress GetProcAddressByName(Handle module, std::string_view name);
ProcAddress GetProcAddressByOrdinal(Handle module, uint32_t ordinal);

// ---------------------------------------------------------------------------
// Registration API (runtime startup, PE loader, R5 import thunks)
// ---------------------------------------------------------------------------

// Windows-style canonical module name: basename, ASCII-lowercased, ".dll" if no extension.
// Returns "" for names with an empty basename.
std::string canonical_module_name(std::string_view name);

// Registers the main guest image (GetModuleHandle(NULL), GetModuleFileName(NULL)).
// Returns image_base, or 0 (ERROR_ALREADY_EXISTS / ERROR_INVALID_PARAMETER) on conflict.
Handle register_main_module(std::string_view name, std::string_view path, uint64_t image_base);

// Registers (or returns the existing) virtual module. Pinned modules behave like statically
// imported DLLs: always visible, never unloaded by FreeLibrary. Unpinned modules are only visible
// while their LoadLibrary reference count is > 0.
Handle register_virtual_module(std::string_view name, bool pinned = true);

// Publishes an export (name and/or ordinal 1..65535). Creates a pinned virtual module on demand.
// Re-registering the same name/ordinal replaces the address.
bool register_export(std::string_view module,
                     std::string_view name,
                     uint32_t ordinal,
                     ProcAddress address);
size_t register_exports(std::string_view module, const ExportEntry* entries, size_t count);
size_t register_exports(std::string_view module, std::initializer_list<ExportEntry> entries);

// Maps API-set contracts starting with contract_prefix (e.g. "api-ms-win-core-") to host_module.
// The longest matching prefix wins; an exactly registered module name always wins over rules.
bool register_api_set_rule(std::string_view contract_prefix, std::string_view host_module);

// Import binding by module name (API sets resolved, visibility ignored). 0 if unresolved.
ProcAddress resolve_import(std::string_view module, std::string_view name);
ProcAddress resolve_import_ordinal(std::string_view module, uint32_t ordinal);

template <typename R, typename... Args>
inline ProcAddress proc_address(R (*fn)(Args...)) {
    return static_cast<ProcAddress>(reinterpret_cast<uintptr_t>(fn));
}

// export_entry("HeapAlloc", &pg6_win32_heap_alloc) -> ExportEntry for register_exports.
template <typename R, typename... Args>
inline ExportEntry export_entry(const char* name, R (*fn)(Args...), uint32_t ordinal = 0) {
    return ExportEntry{name, ordinal, proc_address(fn)};
}

}  // namespace playgta6::os::win32

// ---------------------------------------------------------------------------
// Prefixed C-ABI entrypoints
// ---------------------------------------------------------------------------

extern "C" {

Pg6Handle pg6_win32_get_process_heap(void);
Pg6Handle pg6_win32_heap_create(uint32_t options, uint64_t initial_size, uint64_t maximum_size);
int32_t pg6_win32_heap_destroy(Pg6Handle heap);
uint64_t pg6_win32_heap_alloc(Pg6Handle heap, uint32_t flags, uint64_t bytes);
uint64_t pg6_win32_heap_realloc(Pg6Handle heap, uint32_t flags, uint64_t mem, uint64_t bytes);
int32_t pg6_win32_heap_free(Pg6Handle heap, uint32_t flags, uint64_t mem);
uint64_t pg6_win32_heap_size(Pg6Handle heap, uint32_t flags, uint64_t mem);
int32_t pg6_win32_heap_validate(Pg6Handle heap, uint32_t flags, uint64_t mem);

Pg6Handle pg6_win32_get_module_handle_a(const char* module_name);
Pg6Handle pg6_win32_get_module_handle_w(const char16_t* module_name);
Pg6Handle pg6_win32_load_library_a(const char* file_name);
Pg6Handle pg6_win32_load_library_w(const char16_t* file_name);
Pg6Handle pg6_win32_load_library_ex_a(const char* file_name, Pg6Handle file, uint32_t flags);
Pg6Handle pg6_win32_load_library_ex_w(const char16_t* file_name, Pg6Handle file, uint32_t flags);
int32_t pg6_win32_free_library(Pg6Handle module);
uint32_t pg6_win32_get_module_file_name_a(Pg6Handle module, char* file_name, uint32_t size);
uint32_t pg6_win32_get_module_file_name_w(Pg6Handle module, char16_t* file_name, uint32_t size);
Pg6ProcAddress pg6_win32_get_proc_address(Pg6Handle module, const char* proc_name);

Pg6ProcAddress pg6_loader_get_proc_address_by_name(Pg6Handle module, const char* name);
Pg6ProcAddress pg6_loader_get_proc_address_by_ordinal(Pg6Handle module, uint32_t ordinal);

Pg6Handle pg6_loader_register_main_module(const char* name, const char* path, uint64_t image_base);
Pg6Handle pg6_loader_register_virtual_module(const char* name, int32_t pinned);
int32_t pg6_loader_register_export(const char* module,
                                   const char* name,
                                   uint32_t ordinal,
                                   Pg6ProcAddress address);
size_t pg6_loader_register_exports(const char* module, const Pg6ExportEntry* entries, size_t count);
int32_t pg6_loader_register_api_set_rule(const char* contract_prefix, const char* host_module);
Pg6ProcAddress pg6_loader_resolve_import(const char* module, const char* name);
Pg6ProcAddress pg6_loader_resolve_import_ordinal(const char* module, uint32_t ordinal);

}  // extern "C"
