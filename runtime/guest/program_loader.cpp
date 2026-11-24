#include "guest/program_loader.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <utility>

#include "guest/guest_exceptions.h"
#include "os/win32_loader.h"

namespace playgta6::guest {

namespace {

namespace pgim = os::pgim;
namespace win32 = os::win32;

constexpr uint32_t kMaxExports = 65536;
constexpr size_t kMaxNameBytes = 4096;
constexpr size_t kDirExport = 0;
constexpr size_t kDirException = 3;

ProgramError fail(ProgramErrorCode code, size_t module, std::string detail) {
    ProgramError error;
    error.code = code;
    error.module = module;
    error.detail = std::move(detail);
    return error;
}

std::string hex(uint64_t v) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "0x%" PRIx64, v);
    return buffer;
}

// Bounds-checked reads of a mapped image [base, base + size).
struct ImageView {
    os::VirtualMemoryArena& arena;
    uint64_t base;
    uint64_t size;

    template <typename T>
    bool read(uint64_t rva, T& out) const {
        if (rva > size || sizeof(T) > size - rva) {
            return false;
        }
        return arena.read(base + rva, &out, sizeof(T));
    }

    bool read_string(uint64_t rva, std::string& out) const {
        out.clear();
        for (size_t i = 0; i < kMaxNameBytes; ++i) {
            char c = 0;
            if (!read(rva + i, c)) {
                return false;
            }
            if (c == '\0') {
                return true;
            }
            out.push_back(c);
        }
        return false;
    }
};

struct DataDirectory {
    uint32_t rva = 0;
    uint32_t size = 0;
};

struct Headers {
    uint64_t image_base = 0;
    uint32_t size_of_image = 0;
    std::vector<DataDirectory> dirs;

    DataDirectory dir(size_t i) const { return i < dirs.size() ? dirs[i] : DataDirectory{}; }
};

// PE32+ headers of the mapped image.
bool read_headers(const ImageView& view, Headers& out) {
    uint16_t mz = 0;
    uint32_t e_lfanew = 0;
    uint32_t signature = 0;
    uint16_t magic = 0;
    uint32_t count = 0;
    if (!view.read(0, mz) || mz != 0x5A4D || !view.read(0x3C, e_lfanew) ||
        !view.read(e_lfanew, signature) || signature != 0x00004550) {
        return false;
    }
    const uint64_t optional = uint64_t{e_lfanew} + 24;
    if (!view.read(optional, magic) || magic != 0x020B || !view.read(optional + 24, out.image_base) ||
        !view.read(optional + 56, out.size_of_image) || !view.read(optional + 108, count)) {
        return false;
    }
    out.dirs.assign(std::min<uint32_t>(count, 16), DataDirectory{});
    for (size_t i = 0; i < out.dirs.size(); ++i) {
        if (!view.read(optional + 112 + 8 * i, out.dirs[i].rva) ||
            !view.read(optional + 116 + 8 * i, out.dirs[i].size)) {
            return false;
        }
    }
    return true;
}

// Registers every export of the module from its export directory: addresses as guest VAs,
// forwarder strings as forwarders.
ProgramError register_exports(const ImageView& view,
                              const DataDirectory& dir,
                              const std::string& module_name,
                              size_t index,
                              ProgramModuleInfo& info) {
    uint32_t base_ordinal = 0;
    uint32_t function_count = 0;
    uint32_t name_count = 0;
    uint32_t functions = 0;
    uint32_t names = 0;
    uint32_t ordinals = 0;
    if (!view.read(uint64_t{dir.rva} + 16, base_ordinal) ||
        !view.read(uint64_t{dir.rva} + 20, function_count) ||
        !view.read(uint64_t{dir.rva} + 24, name_count) || !view.read(uint64_t{dir.rva} + 28, functions) ||
        !view.read(uint64_t{dir.rva} + 32, names) || !view.read(uint64_t{dir.rva} + 36, ordinals)) {
        return fail(ProgramErrorCode::BadHeaders, index, "export directory unreadable");
    }
    if (function_count > kMaxExports || name_count > kMaxExports) {
        return fail(ProgramErrorCode::BadHeaders, index, "export directory too large");
    }
    std::vector<std::vector<std::string>> names_by_index(function_count);
    for (uint32_t i = 0; i < name_count; ++i) {
        uint32_t name_rva = 0;
        uint16_t slot = 0;
        std::string name;
        if (!view.read(uint64_t{names} + 4 * i, name_rva) ||
            !view.read(uint64_t{ordinals} + 2 * i, slot) || !view.read_string(name_rva, name) ||
            slot >= function_count) {
            return fail(ProgramErrorCode::BadHeaders, index, "export name table malformed");
        }
        names_by_index[slot].push_back(std::move(name));
    }
    const uint64_t dir_end = uint64_t{dir.rva} + dir.size;
    for (uint32_t slot = 0; slot < function_count; ++slot) {
        uint32_t rva = 0;
        if (!view.read(uint64_t{functions} + 4 * slot, rva)) {
            return fail(ProgramErrorCode::BadHeaders, index, "export address table unreadable");
        }
        if (rva == 0) {
            continue;
        }
        const uint32_t ordinal = base_ordinal + slot;
        const uint32_t registry_ordinal = ordinal <= 0xFFFFu ? ordinal : 0;
        const bool forwarder = rva >= dir.rva && rva < dir_end;
        std::string target;
        if (forwarder && !view.read_string(rva, target)) {
            return fail(ProgramErrorCode::BadHeaders, index, "export forwarder unreadable");
        }
        auto add = [&](std::string_view name) {
            if (name.empty() && registry_ordinal == 0) {
                return true;
            }
            const bool ok =
                forwarder
                    ? win32::register_export_forwarder(module_name, name, registry_ordinal, target)
                    : win32::register_guest_export(module_name, name, registry_ordinal,
                                                   view.base + rva);
            if (ok) {
                ++info.exports;
                info.forwarders += forwarder ? 1 : 0;
            }
            return ok;
        };
        bool ok = true;
        if (names_by_index[slot].empty()) {
            ok = add({});
        }
        for (const std::string& name : names_by_index[slot]) {
            ok = add(name) && ok;
        }
        if (!ok) {
            return fail(ProgramErrorCode::RegisterModuleFailed, index,
                        "module registry refused an export of " + module_name);
        }
    }
    return {};
}

}  // namespace

const char* program_error_name(ProgramErrorCode code) {
    switch (code) {
        case ProgramErrorCode::Ok:
            return "ok";
        case ProgramErrorCode::InvalidProgram:
            return "invalid program descriptor";
        case ProgramErrorCode::MissingImage:
            return "missing image";
        case ProgramErrorCode::BaseMismatch:
            return "image base mismatch";
        case ProgramErrorCode::RegisterModuleFailed:
            return "module registration failed";
        case ProgramErrorCode::BadHeaders:
            return "bad PE headers";
        case ProgramErrorCode::FunctionTableFailed:
            return "function table registration failed";
        case ProgramErrorCode::UnsupportedMemoryModel:
            return "unsupported memory model";
    }
    return "unknown";
}

ImageSource memory_image_source(std::vector<std::pair<std::string, std::span<const uint8_t>>> images) {
    return [images = std::move(images)](
               const Pg6ProgramModule& module) -> std::unique_ptr<pgim::ImageReader> {
        for (const auto& [name, bytes] : images) {
            if (name == module.module) {
                return std::make_unique<pgim::MemoryImageReader>(bytes);
            }
        }
        return nullptr;
    };
}

ProgramError check_program(const Pg6Program& program) {
    if (program.version != kPg6ProgramVersion) {
        return fail(ProgramErrorCode::InvalidProgram, SIZE_MAX,
                    "unsupported descriptor version " + std::to_string(program.version));
    }
    if (program.module_count == 0 || program.modules == nullptr) {
        return fail(ProgramErrorCode::InvalidProgram, SIZE_MAX, "no modules");
    }
    for (size_t i = 0; i < program.module_count; ++i) {
        const Pg6ProgramModule& m = program.modules[i];
        const uint32_t kind = m.flags & (kPg6ProgramModuleMain | kPg6ProgramModuleDll);
        const bool last = i + 1 == program.module_count;
        if (m.file_name == nullptr || m.module == nullptr || m.file_name[0] == '\0' || m.base == 0 ||
            m.size_of_image == 0) {
            return fail(ProgramErrorCode::InvalidProgram, i, "missing name, base or size");
        }
        if (kind != (last ? kPg6ProgramModuleMain : kPg6ProgramModuleDll)) {
            return fail(ProgramErrorCode::InvalidProgram, i,
                        "the executable must be the last module and every other module a DLL");
        }
        for (size_t j = 0; j < i; ++j) {
            const Pg6ProgramModule& o = program.modules[j];
            if (m.base < o.base + o.size_of_image && o.base < m.base + m.size_of_image) {
                return fail(ProgramErrorCode::InvalidProgram, i,
                            std::string("image range overlaps ") + o.file_name);
            }
        }
    }
    if (program.memory != nullptr) {
        if (std::string why = memory_model_unsupported(*program.memory); !why.empty()) {
            return fail(ProgramErrorCode::UnsupportedMemoryModel, SIZE_MAX,
                        "the program was lifted with the memory model " +
                            describe_memory_model(*program.memory) + ", which this runtime cannot serve: " +
                            why);
        }
    }
    return {};
}

ProgramError prepare_program_boot(const Pg6Program& program,
                                  const ImageSource& images,
                                  const ProgramLoadOptions& options,
                                  ProgramBoot& out) {
    out.config = BootConfig{};
    out.readers.clear();
    out.program = program;
    out.options = options;
    out.after_load_error = {};
    if (ProgramError error = check_program(program); !error.ok()) {
        return error;
    }
    // The exception APIs (RaiseException, RtlUnwindEx, ...) work on the caller's registers and are
    // bound to R11's guest-state thunks, also when imported through an API set (resolved through
    // the module registry once the thunks are published there).
    register_exception_exports();
    out.imports.clear();
    out.imports.reserve(program.module_count);
    out.exception_imports = 0;
    // modules[0] = the executable (last in the descriptor), then the DLLs in initialisation order.
    std::vector<size_t> order{program.module_count - 1};
    for (size_t i = 0; i + 1 < program.module_count; ++i) {
        order.push_back(i);
    }
    for (const size_t i : order) {
        const Pg6ProgramModule& m = program.modules[i];
        std::unique_ptr<pgim::ImageReader> reader = images ? images(m) : nullptr;
        if (reader == nullptr) {
            return fail(ProgramErrorCode::MissingImage, i, std::string("no image for ") + m.module);
        }
        BootModule module;
        module.image = reader.get();
        module.name = m.file_name;
        module.path = options.module_directory + m.file_name;
        const uint64_t count = m.lifted_function_count != nullptr ? *m.lifted_function_count : 0;
        if (m.lifted_functions != nullptr && count != 0) {
            module.lifted_tables.emplace_back(m.lifted_functions, static_cast<size_t>(count));
        }
        const Pg6BootImport* imports = m.imports != nullptr ? *m.imports : nullptr;
        const size_t import_count = m.import_count != nullptr ? *m.import_count : 0;
        std::vector<BootImport>& table = out.imports.emplace_back();
        if (imports != nullptr && import_count != 0) {
            table.assign(imports, imports + import_count);
            for (BootImport& import : table) {
                if (import.symbol == nullptr || import.dll == nullptr) {
                    continue;
                }
                if (HostThunkFn thunk = exception_thunk_for(import.dll, import.symbol)) {
                    import.thunk = thunk;
                    import.flags |= kPg6BootImportBound;
                    import.shim = "guest_exceptions";
                    ++out.exception_imports;
                }
            }
            module.imports = std::span<const BootImport>(table);
        }
        out.readers.push_back(std::move(reader));
        out.config.modules.push_back(std::move(module));
    }
    out.config.call_dll_entry_points = true;
    if (program.memory != nullptr) {
        out.config.memory = *program.memory;
    }
    if (program.exports != nullptr && *program.exports != nullptr && program.export_count != nullptr) {
        out.config.export_thunks = std::span<const Pg6BootExport>(*program.exports, *program.export_count);
    }
    ProgramBoot* self = &out;
    out.config.after_load = [self](os::VirtualMemoryArena& arena, std::string& message) {
        self->after_load_error = register_program_modules(arena, self->program, self->options);
        if (self->after_load_error.ok()) {
            return true;
        }
        message = std::string(program_error_name(self->after_load_error.code)) + ": " +
                  self->after_load_error.detail;
        return false;
    };
    return {};
}

ProgramError register_program_modules(os::VirtualMemoryArena& arena,
                                      const Pg6Program& program,
                                      const ProgramLoadOptions& options,
                                      std::vector<ProgramModuleInfo>* info) {
    if (ProgramError error = check_program(program); !error.ok()) {
        return error;
    }
    if (info != nullptr) {
        info->assign(program.module_count, ProgramModuleInfo{});
    }
    for (size_t i = 0; i < program.module_count; ++i) {
        const Pg6ProgramModule& m = program.modules[i];
        ProgramModuleInfo module_info;
        const ImageView view{arena, m.base, m.size_of_image};
        Headers headers;
        if (!read_headers(view, headers)) {
            return fail(ProgramErrorCode::BadHeaders, i,
                        std::string("no PE headers mapped at ") + hex(m.base) + " for " + m.file_name);
        }
        if (headers.image_base != m.base || headers.size_of_image != m.size_of_image) {
            return fail(ProgramErrorCode::BaseMismatch, i,
                        std::string(m.file_name) + " is mapped for " + hex(headers.image_base) +
                            " size " + hex(headers.size_of_image) + ", the descriptor says " +
                            hex(m.base) + " size " + hex(m.size_of_image));
        }
        if ((m.flags & kPg6ProgramModuleDll) != 0 &&
            win32::register_guest_module(m.file_name, options.module_directory + m.file_name, m.base,
                                         m.size_of_image) != m.base) {
            return fail(ProgramErrorCode::RegisterModuleFailed, i,
                        std::string("module registry refused ") + m.file_name);
        }
        if (headers.dir(kDirExport).rva != 0) {
            if (ProgramError error =
                    register_exports(view, headers.dir(kDirExport), m.file_name, i, module_info);
                !error.ok()) {
                return error;
            }
        }
        const DataDirectory pdata = headers.dir(kDirException);
        const bool has_pdata = pdata.rva != 0 && pdata.size != 0;
        if (has_pdata) {
            module_info.pdata_va = m.base + pdata.rva;
            module_info.pdata_size = pdata.size;
        }
        // Every image, so RtlPcToFileHeader knows it; only a module with .pdata must succeed.
        if (options.register_function_tables && !register_image_from_headers(arena, m.base) &&
            has_pdata) {
            return fail(ProgramErrorCode::FunctionTableFailed, i,
                        std::string("unwinder refused the .pdata of ") + m.file_name);
        }
        if (info != nullptr) {
            (*info)[i] = module_info;
        }
    }
    return {};
}

}  // namespace playgta6::guest
