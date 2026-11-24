#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "abi/playgta6_abi.h"
#include "guest/boot.h"
#include "guest/guest_runtime.h"
#include "os/image_loader.h"
#include "os/vmem.h"

// Program loader: boots a multi-module guest program built by `playgta6-recomp build-program`
// (recomp/src/pipeline/program.rs, R12) through the boot sequence (guest/boot.h, R14).
//
// A program is the executable plus the guest DLLs it imports (msvcp140.dll, vcruntime140.dll,
// ...), each recompiled as its own guest module. The recompiler gives every module a distinct
// image range, rebases moved modules at build time and pre-fills every IAT slot that imports
// another guest module's export with that export's guest VA, so the images load at their
// preferred bases without relocation and no guest-to-guest binding happens at run time. It also
// emits `pg6_program.cpp`, which defines `extern "C" const Pg6Program pg6_program`: the modules in
// initialisation order (dependencies first, executable last) with their lifted-function tables
// and import bridge tables.
//
// Booting a program (what a host harness calls):
//   os::win32::set_vmem_arena(&arena);
//   ProgramBoot boot;
//   ProgramError e = prepare_program_boot(pg6_program, source, options, boot);
//   boot.config.command_line = ...;                     // other BootConfig fields as needed
//   BootResult result = run_boot(boot.config);
// prepare_program_boot fills BootConfig with modules[0] = the executable and modules[1..] = the
// DLLs in initialisation order (the boot then gives TLS indices in that order, the executable 0,
// matching the recompiler's manifest), each module's PGIM reader, lifted table and import table
// (every import of an exception API - RaiseException, RtlCaptureContext, RtlLookupFunctionEntry,
// RtlVirtualUnwind, RtlUnwind(Ex), RtlRestoreContext, RtlPcToFileHeader,
// RtlAdd/DeleteFunctionTable, (Set)UnhandledExceptionFilter, RtlRaiseException, also through API
// sets - is rebound to the guest-state thunks of guest_exceptions.h: exception_thunk_for, after
// register_exception_exports), and sets
//   - after_load: register_program_modules() - every DLL becomes a guest module of the Win32
//     module registry under its file name (GetModuleHandle -> its base, GetProcAddress -> guest
//     VAs of its exports, forwarders followed), the executable's exports are published too, and
//     every image is registered with the unwinder (guest_exceptions.h,
//     register_image_from_headers) so RtlLookupFunctionEntry/RtlVirtualUnwind/RtlPcToFileHeader
//     work for PCs in any module (the unwinder cannot unregister images, so a second boot in one
//     process cannot re-register a module that has .pdata);
//   - call_dll_entry_points: each DLL's DllMain(DLL_PROCESS_ATTACH) runs right after its TLS
//     callbacks, in initialisation order, before the executable's entry point.
// Images must keep their PE headers mapped (the recompiler's default): exports, the exception
// directory and TLS are read from them.

inline constexpr uint32_t kPg6ProgramVersion = 1u;

inline constexpr uint32_t kPg6ProgramModuleMain = 1u << 0;  // the executable (exactly one, last)
inline constexpr uint32_t kPg6ProgramModuleDll = 1u << 1;   // a DLL: DllMain runs at boot
inline constexpr uint32_t kPg6ProgramModuleTls = 1u << 2;   // has a TLS directory

extern "C" {

// One module of the generated descriptor.
struct Pg6ProgramModule {
    const char* file_name;  // e.g. "msvcp140.dll" (module registry name)
    const char* module;     // build module name, e.g. "msvcp140" (symbol suffix)
    uint64_t base;          // assigned image base == the PGIM's preferred base
    uint64_t size_of_image;
    uint64_t entry_point;   // VA (DllMain for a DLL), 0 = none
    uint32_t flags;         // kPg6ProgramModule*
    uint32_t tls_index;     // implicit TLS index (with kPg6ProgramModuleTls)
    // `__playgta6_lifted_functions_<module>` and its count (may be null / 0).
    const Pg6LiftedFunctionEntry* lifted_functions;
    const uint64_t* lifted_function_count;
    // The import bridge's `pg6_import_thunks_<module>` / `pg6_import_thunk_count_<module>`
    // (Pg6ImportThunk has Pg6BootImport's layout). May be null.
    const Pg6BootImport* const* imports;
    const size_t* import_count;
};

struct Pg6Program {
    uint32_t version;  // kPg6ProgramVersion
    uint32_t module_count;
    const Pg6ProgramModule* modules;  // initialisation order, executable last
    // The program's export table: `pg6_export_thunks` / `pg6_export_thunk_count` of
    // `exports/pg6_export_thunks.cpp` (guest/export_thunks.h, Pg6BootExport has the generated
    // Pg6ExportThunk's layout). Added after version 1 shipped: a descriptor that stops at
    // `modules` is still valid and just has no export thunks (both fields null).
    const Pg6BootExport* const* exports;
    const size_t* export_count;
    // How the program's lifted code translates guest addresses (guest/memory_model.h). Added
    // after `exports` (O29): null - also for a descriptor that stops earlier - is the flat
    // default; the generator emits it only for another model. prepare_program_boot copies it into
    // BootConfig::memory; check_program refuses a model this runtime cannot serve. (The default
    // member initializer keeps -Wmissing-field-initializers quiet for descriptors without it.)
    const Pg6GuestMemoryModel* memory = nullptr;
};

}  // extern "C"

namespace playgta6::guest {

enum class ProgramErrorCode : uint32_t {
    Ok = 0,
    InvalidProgram,        // descriptor malformed (detail says why)
    MissingImage,          // the image source returned no reader
    BaseMismatch,          // a module is not mapped at its descriptor base / size
    RegisterModuleFailed,  // the Win32 module registry refused a module or an export
    BadHeaders,            // mapped PE headers or export directory unreadable or malformed
    FunctionTableFailed,   // the unwinder refused the module's .pdata
    UnsupportedMemoryModel,  // the lifted code's memory model is one this runtime cannot serve
};

const char* program_error_name(ProgramErrorCode code);

struct ProgramError {
    ProgramErrorCode code = ProgramErrorCode::Ok;
    size_t module = SIZE_MAX;  // descriptor index of the failing module
    std::string detail;

    bool ok() const { return code == ProgramErrorCode::Ok; }
};

// Supplies the PGIM container of a module (a file, an HTTP range reader, memory). Returning null
// fails with MissingImage.
using ImageSource =
    std::function<std::unique_ptr<os::pgim::ImageReader>(const Pg6ProgramModule& module)>;

// An ImageSource over in-memory containers, matched by build module name
// (`Pg6ProgramModule::module`). The bytes must outlive the boot.
ImageSource memory_image_source(std::vector<std::pair<std::string, std::span<const uint8_t>>> images);

struct ProgramLoadOptions {
    // Directory prefix for module paths (GetModuleFileName), e.g. "D:\\"; empty = file name only.
    std::string module_directory;
    // Register every module's .pdata with the unwinder (guest_exceptions.h).
    bool register_function_tables = true;
};

// Validates a descriptor: version, names, bases, exactly one executable and it comes last,
// disjoint image ranges.
ProgramError check_program(const Pg6Program& program);

// A boot configuration for a program; keeps the image readers and a copy of the descriptor header
// alive (the module array it points to must outlive the boot; the generated one is static). Not
// copyable or movable once prepared (the after_load hook refers to it).
struct ProgramBoot {
    BootConfig config;
    std::vector<std::unique_ptr<os::pgim::ImageReader>> readers;
    // Per boot module: the descriptor's import table, with the exception APIs rebound.
    std::vector<std::vector<BootImport>> imports;
    size_t exception_imports = 0;  // imports rebound to guest_exceptions.h thunks
    Pg6Program program{};
    ProgramLoadOptions options;
    ProgramError after_load_error;  // why after_load failed, if it did

    ProgramBoot() = default;
    ProgramBoot(const ProgramBoot&) = delete;
    ProgramBoot& operator=(const ProgramBoot&) = delete;
};

ProgramError prepare_program_boot(const Pg6Program& program,
                                  const ImageSource& images,
                                  const ProgramLoadOptions& options,
                                  ProgramBoot& out);

// What register_program_modules found for one module.
struct ProgramModuleInfo {
    size_t exports = 0;     // registered exports (forwarders included)
    size_t forwarders = 0;
    uint64_t pdata_va = 0;  // exception directory, 0 when absent
    uint32_t pdata_size = 0;
};

// The after_load step on its own: checks that every module is mapped at its base, registers the
// DLLs as guest modules and every module's exports (from the mapped export directories) with
// the Win32 module registry, and (options.register_function_tables) every module's .pdata with
// the unwinder. The executable must already be registered as the main module (run_boot does it).
ProgramError register_program_modules(os::VirtualMemoryArena& arena,
                                      const Pg6Program& program,
                                      const ProgramLoadOptions& options,
                                      std::vector<ProgramModuleInfo>* info = nullptr);

}  // namespace playgta6::guest
