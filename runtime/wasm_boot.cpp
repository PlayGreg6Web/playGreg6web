#include "wasm_boot.h"

#include <algorithm>
#include <cstring>
#include <utility>

#include "guest/boot.h"
#include "guest/program_loader.h"
#include "os/image_loader.h"
#include "os/os_shim.h"

#if defined(PG6_BOOT_LINKED_GUEST) && defined(PG6_BOOT_PROGRAM)
extern "C" const Pg6Program pg6_program;
#elif defined(PG6_BOOT_LINKED_GUEST)
#ifndef PG6_BOOT_MODULE
#define PG6_BOOT_MODULE main
#endif
#define PG6_CAT2(a, b) a##b
#define PG6_CAT(a, b) PG6_CAT2(a, b)
extern "C" const Pg6LiftedFunctionEntry PG6_CAT(__playgta6_lifted_functions_, PG6_BOOT_MODULE)[];
extern "C" const int64_t PG6_CAT(__playgta6_lifted_function_count_, PG6_BOOT_MODULE);
extern "C" const Pg6BootImport pg6_import_thunks[];
extern "C" const size_t pg6_import_thunk_count;
#endif

namespace playgta6::boot {
namespace {

int32_t open_path(const WasmBootConfig& cfg, std::string_view rel_path) {
    if (cfg.open_fn) {
        return cfg.open_fn(rel_path);
    }
    return os::httpfs_open(rel_path);
}

int64_t read_path(const WasmBootConfig& cfg,
                  int32_t handle,
                  uint64_t offset,
                  void* dst,
                  size_t len) {
    if (cfg.read_fn) {
        return cfg.read_fn(handle, offset, dst, len);
    }
    return os::httpfs_read(handle, offset, dst, len);
}

void close_path(const WasmBootConfig& cfg, int32_t handle) {
    if (cfg.close_fn) {
        cfg.close_fn(handle);
        return;
    }
    os::httpfs_close(handle);
}

void write_log(const WasmBootConfig& cfg, std::string_view message) {
    if (cfg.log_fn) {
        cfg.log_fn(message);
        return;
    }
    os::log(message);
}

struct HandleCloser {
    const WasmBootConfig* cfg = nullptr;
    int32_t handle = -1;

    HandleCloser(const WasmBootConfig& config, int32_t fd) : cfg(&config), handle(fd) {}
    HandleCloser(const HandleCloser&) = delete;
    HandleCloser& operator=(const HandleCloser&) = delete;
    HandleCloser(HandleCloser&& other) noexcept : cfg(other.cfg), handle(other.handle) {
        other.handle = -1;
    }
    HandleCloser& operator=(HandleCloser&& other) noexcept {
        if (this != &other) {
            reset();
            cfg = other.cfg;
            handle = other.handle;
            other.handle = -1;
        }
        return *this;
    }
    ~HandleCloser() { reset(); }

    void reset() {
        if (cfg != nullptr && handle >= 0) {
            close_path(*cfg, handle);
            handle = -1;
        }
    }
};

int hex_nibble(char c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return 10 + (c - 'a');
    }
    if (c >= 'A' && c <= 'F') {
        return 10 + (c - 'A');
    }
    return -1;
}

// Decode one JSON string starting at the opening quote. `at` is the index of `"`.
bool parse_json_string(std::string_view json, size_t& at, std::string& out) {
    if (at >= json.size() || json[at] != '"') {
        return false;
    }
    ++at;
    out.clear();
    while (at < json.size()) {
        const char c = json[at++];
        if (c == '"') {
            return true;
        }
        if (c == '\\') {
            if (at >= json.size()) {
                return false;
            }
            const char e = json[at++];
            switch (e) {
                case '"':
                case '\\':
                case '/':
                    out.push_back(e);
                    break;
                case 'b':
                    out.push_back('\b');
                    break;
                case 'f':
                    out.push_back('\f');
                    break;
                case 'n':
                    out.push_back('\n');
                    break;
                case 'r':
                    out.push_back('\r');
                    break;
                case 't':
                    out.push_back('\t');
                    break;
                case 'u': {
                    if (at + 4 > json.size()) {
                        return false;
                    }
                    unsigned code = 0;
                    for (int i = 0; i < 4; ++i) {
                        const int n = hex_nibble(json[at++]);
                        if (n < 0) {
                            return false;
                        }
                        code = (code << 4) | static_cast<unsigned>(n);
                    }
                    if (code <= 0x7F) {
                        out.push_back(static_cast<char>(code));
                    } else if (code <= 0x7FF) {
                        out.push_back(static_cast<char>(0xC0 | (code >> 6)));
                        out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                    } else {
                        out.push_back(static_cast<char>(0xE0 | (code >> 12)));
                        out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                        out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                    }
                    break;
                }
                default:
                    return false;
            }
            continue;
        }
        out.push_back(c);
    }
    return false;
}

bool ends_with_pgim(std::string_view path) {
    constexpr std::string_view kExt = ".pgim";
    if (path.size() < kExt.size()) {
        return false;
    }
    const std::string_view tail = path.substr(path.size() - kExt.size());
    for (size_t i = 0; i < kExt.size(); ++i) {
        const char a = tail[i];
        const char al = (a >= 'A' && a <= 'Z') ? static_cast<char>(a - 'A' + 'a') : a;
        if (al != kExt[i]) {
            return false;
        }
    }
    return true;
}

void normalize_rel_path(std::string& path) {
    std::replace(path.begin(), path.end(), '\\', '/');
    while (path.size() >= 2 && path[0] == '.' && path[1] == '/') {
        path.erase(0, 2);
    }
}

std::string to_lower_ascii(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c >= 'A' && c <= 'Z') {
            out.push_back(static_cast<char>(c - 'A' + 'a'));
        } else {
            out.push_back(c);
        }
    }
    return out;
}

std::string normalize_match_path(std::string_view raw) {
    std::string out(raw);
    normalize_rel_path(out);
    while (!out.empty() && out.front() == '/') {
        out.erase(0, 1);
    }
    return to_lower_ascii(out);
}

bool starts_with_boot_prefix(std::string_view path) {
    constexpr std::string_view kBoot = "boot/";
    if (path.size() <= kBoot.size()) {
        return false;
    }
    for (size_t i = 0; i < kBoot.size(); ++i) {
        const char c = path[i];
        const char cl = (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
        if (cl != kBoot[i]) {
            return false;
        }
    }
    return true;
}

bool load_module_pgim(const WasmBootConfig& cfg,
                      const std::string& resolved_path,
                      std::vector<uint8_t>& out,
                      std::string& loaded_path_out,
                      std::string& error_out) {
    if (load_file_via_httpfs(cfg, resolved_path, out, error_out)) {
        loaded_path_out = resolved_path;
        return true;
    }

    const std::string alt = starts_with_boot_prefix(resolved_path)
                                ? resolved_path.substr(5)
                                : ("boot/" + resolved_path);
    if (!alt.empty() && alt != resolved_path) {
        std::string alt_error;
        if (load_file_via_httpfs(cfg, alt, out, alt_error)) {
            loaded_path_out = alt;
            error_out.clear();
            return true;
        }
        if (error_out.find("httpfs_open failed") != std::string::npos &&
            alt_error.find("httpfs_open failed") == std::string::npos) {
            error_out = std::move(alt_error);
        }
    }
    return false;
}

WasmBootResult fail_result(std::string stage, std::string error) {
    WasmBootResult r;
    r.ok = false;
    r.linked_guest_available = false;
    r.stage = std::move(stage);
    r.error = std::move(error);
    return r;
}

}  // namespace

bool has_pgim_magic(std::span<const uint8_t> bytes) {
    if (bytes.size() < kPgimMagicBytes) {
        return false;
    }
    return std::memcmp(bytes.data(), kPgimMagic, kPgimMagicBytes) == 0;
}

std::string resolve_module_pgim_path(std::string_view module_name,
                                     std::span<const std::string> manifest_pgim_paths) {
    if (module_name.empty()) {
        return {};
    }

    const std::string target = to_lower_ascii(module_name);
    const std::string canonical_plain = target + "/image/guest.pgim";
    const std::string canonical_boot = "boot/" + canonical_plain;
    const std::string canonical_suffix = "/" + canonical_plain;

    for (const std::string& path : manifest_pgim_paths) {
        const std::string norm = normalize_match_path(path);
        if (norm == canonical_boot || norm == canonical_plain ||
            (norm.size() >= canonical_suffix.size() &&
             norm.compare(norm.size() - canonical_suffix.size(), canonical_suffix.size(),
                          canonical_suffix) == 0)) {
            return path;
        }
    }

    const std::string prefix_plain = target + "/";
    const std::string prefix_boot = "boot/" + prefix_plain;
    for (const std::string& path : manifest_pgim_paths) {
        const std::string norm = normalize_match_path(path);
        if (norm.rfind(prefix_boot, 0) == 0 || norm.rfind(prefix_plain, 0) == 0) {
            return path;
        }
    }

    const std::string segment = "/" + target + "/";
    for (const std::string& path : manifest_pgim_paths) {
        const std::string norm = normalize_match_path(path);
        if (norm.find(segment) != std::string::npos) {
            return path;
        }
    }

    return "boot/" + std::string(module_name) + "/image/guest.pgim";
}

std::vector<std::string> extract_pgim_paths_from_manifest(std::string_view json) {
    std::vector<std::string> paths;
    size_t i = 0;
    while (i < json.size()) {
        if (json[i] != '"') {
            ++i;
            continue;
        }
        std::string value;
        if (!parse_json_string(json, i, value)) {
            break;
        }
        if (!ends_with_pgim(value)) {
            continue;
        }
        normalize_rel_path(value);
        if (value.empty()) {
            continue;
        }
        if (std::find(paths.begin(), paths.end(), value) == paths.end()) {
            paths.push_back(std::move(value));
        }
    }
    return paths;
}

void emit_boot_milestone(const WasmBootConfig& cfg, const char* milestone, const std::string& detail) {
    if (milestone == nullptr || milestone[0] == '\0') {
        return;
    }
    std::string line = "[boot] ";
    line += milestone;
    if (!detail.empty()) {
        line += ": ";
        line += detail;
    }
    write_log(cfg, line);
}

bool load_file_via_httpfs(const WasmBootConfig& cfg,
                          const std::string& rel_path,
                          std::vector<uint8_t>& out,
                          std::string& error_out) {
    out.clear();
    error_out.clear();

    const size_t chunk = cfg.read_chunk_bytes == 0 ? 65536 : cfg.read_chunk_bytes;
    const int32_t handle = open_path(cfg, rel_path);
    if (handle < 0) {
        error_out = "httpfs_open failed for " + rel_path;
        return false;
    }

    HandleCloser closer(cfg, handle);
    uint64_t offset = 0;
    std::vector<uint8_t> buf(chunk);
    for (;;) {
        const size_t want = buf.size();
        const int64_t n = read_path(cfg, handle, offset, buf.data(), want);
        if (n < 0) {
            error_out = "httpfs_read failed for " + rel_path;
            return false;
        }
        if (n == 0) {
            return true;
        }
        const size_t got = static_cast<size_t>(n);
        if (got > want) {
            error_out = "httpfs_read returned more bytes than requested for " + rel_path;
            return false;
        }
        if (cfg.max_file_bytes != 0 && got > cfg.max_file_bytes - out.size()) {
            error_out = "file exceeds max_file_bytes: " + rel_path;
            return false;
        }
        out.insert(out.end(), buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(got));
        offset += static_cast<uint64_t>(got);
        if (got < want) {
            return true;
        }
    }
}

WasmBootResult run_wasm_program_boot(const WasmBootConfig& cfg, const Pg6Program& program) {
    WasmBootResult result;
    result.linked_guest_available = true;
    result.stage = "PGIM_LOAD";

    if (const guest::ProgramError check_err = guest::check_program(program); !check_err.ok()) {
        result.ok = false;
        result.stage = "program_prepare";
        result.error =
            check_err.detail.empty() ? guest::program_error_name(check_err.code) : check_err.detail;
        return result;
    }
    for (uint32_t i = 0; i < program.module_count; ++i) {
        if (program.modules[i].module[0] == '\0') {
            result.ok = false;
            result.stage = "program_prepare";
            result.error = "empty module build name";
            return result;
        }
    }

    std::vector<std::string> manifest_pgim_paths;
    if (!cfg.manifest_path.empty()) {
        std::vector<uint8_t> manifest;
        std::string manifest_error;
        if (load_file_via_httpfs(cfg, cfg.manifest_path, manifest, manifest_error)) {
            manifest_pgim_paths = extract_pgim_paths_from_manifest(
                std::string_view(reinterpret_cast<const char*>(manifest.data()), manifest.size()));
        }
    }

    std::vector<std::vector<uint8_t>> images;
    images.reserve(program.module_count);
    for (uint32_t i = 0; i < program.module_count; ++i) {
        const Pg6ProgramModule& mod = program.modules[i];
        const std::string resolved_path =
            resolve_module_pgim_path(mod.module, manifest_pgim_paths);
        std::vector<uint8_t> bytes;
        std::string loaded_path;
        std::string error;
        if (!load_module_pgim(cfg, resolved_path, bytes, loaded_path, error)) {
            result.ok = false;
            result.stage = "httpfs";
            result.error = std::move(error);
            return result;
        }
        if (!has_pgim_magic(bytes)) {
            result.ok = false;
            result.stage = "pgim_magic";
            result.error = "invalid PGIM magic for " + loaded_path;
            return result;
        }
        result.total_bytes_loaded += bytes.size();
        result.modules_loaded += 1;
        emit_boot_milestone(cfg, "PGIM_LOAD",
                            loaded_path + " loaded (" + std::to_string(bytes.size()) + " bytes)");
        images.push_back(std::move(bytes));
    }

    std::vector<std::pair<std::string, std::span<const uint8_t>>> mem_images;
    mem_images.reserve(program.module_count);
    for (uint32_t i = 0; i < program.module_count; ++i) {
        mem_images.emplace_back(program.modules[i].module, std::span<const uint8_t>(images[i]));
    }

    guest::ProgramLoadOptions load_opts;
    load_opts.module_directory = "G:\\";
    guest::ProgramBoot program_boot;
    const guest::ProgramError err = guest::prepare_program_boot(
        program, guest::memory_image_source(std::move(mem_images)), load_opts, program_boot);
    if (!err.ok()) {
        result.ok = false;
        result.stage = "program_prepare";
        result.error = err.detail.empty() ? guest::program_error_name(err.code) : err.detail;
        return result;
    }

    program_boot.config.log_progress = true;
    program_boot.config.console_mode = guest::kConsoleModeSeriesX;
    program_boot.config.working_directory = "G:\\";
    program_boot.config.command_line = "game_scarlett_master.exe";

    const guest::BootResult boot = guest::run_boot(program_boot.config);
    result.ok = boot.status == guest::BootStatus::Exited;
    result.exit_code = boot.stats.exit_code;
    result.stage = result.ok ? "ENTRY_CALL" : guest::boot_status_name(boot.status);
    if (!result.ok) {
        result.error = boot.summary;
    }
    return result;
}

WasmBootResult run_wasm_guest_boot(const WasmBootConfig& cfg) {
#if defined(PG6_BOOT_LINKED_GUEST) && defined(PG6_BOOT_PROGRAM)
    return run_wasm_program_boot(cfg, pg6_program);
#else
#ifdef PG6_BOOT_LINKED_GUEST
    constexpr bool kLinked = true;
#else
    constexpr bool kLinked = false;
#endif

    std::vector<std::string> pgim_paths;
    std::vector<uint8_t> manifest;
    std::string error;
    const bool have_manifest =
        !cfg.manifest_path.empty() && load_file_via_httpfs(cfg, cfg.manifest_path, manifest, error);
    if (have_manifest) {
        pgim_paths = extract_pgim_paths_from_manifest(
            std::string_view(reinterpret_cast<const char*>(manifest.data()), manifest.size()));
    }

    if (pgim_paths.empty()) {
        if (cfg.fallback_pgim_path.empty()) {
            auto r = fail_result("manifest",
                                 have_manifest ? "program-manifest.json has no .pgim paths"
                                               : "missing program-manifest.json and no fallback .pgim");
            r.linked_guest_available = kLinked;
            return r;
        }
        pgim_paths.push_back(cfg.fallback_pgim_path);
        emit_boot_milestone(cfg, "PGIM_LOAD",
                            have_manifest ? "no .pgim paths in manifest; using fallback"
                                          : "no manifest; using fallback " + cfg.fallback_pgim_path);
    }

    WasmBootResult result;
    result.linked_guest_available = kLinked;
    result.stage = "PGIM_LOAD";

    std::vector<std::vector<uint8_t>> images;
    images.reserve(pgim_paths.size());
    for (const std::string& path : pgim_paths) {
        std::vector<uint8_t> bytes;
        if (!load_file_via_httpfs(cfg, path, bytes, error)) {
            result.ok = false;
            result.error = error;
            result.stage = "httpfs";
            return result;
        }
        if (!has_pgim_magic(bytes)) {
            result.ok = false;
            result.error = "invalid PGIM magic for " + path;
            result.stage = "pgim_magic";
            return result;
        }
        result.total_bytes_loaded += bytes.size();
        result.modules_loaded += 1;
        emit_boot_milestone(cfg, "PGIM_LOAD",
                            path + " loaded (" + std::to_string(bytes.size()) + " bytes)");
        images.push_back(std::move(bytes));
    }

#ifdef PG6_BOOT_LINKED_GUEST
    if (images.empty()) {
        result.ok = false;
        result.error = "no PGIM modules loaded";
        result.stage = "pgim_magic";
        return result;
    }
    os::pgim::MemoryImageReader reader(images.front().data(), images.front().size());
    guest::BootConfig boot_cfg;
    guest::BootModule exe;
    exe.image = &reader;
    exe.name = "game.exe";
    exe.path = pgim_paths.front();
    const auto* table = PG6_CAT(__playgta6_lifted_functions_, PG6_BOOT_MODULE);
    const auto count = static_cast<size_t>(PG6_CAT(__playgta6_lifted_function_count_, PG6_BOOT_MODULE));
    exe.lifted_tables.emplace_back(table, count);
    exe.imports = std::span<const guest::BootImport>(
        reinterpret_cast<const guest::BootImport*>(pg6_import_thunks), pg6_import_thunk_count);
    boot_cfg.modules.push_back(std::move(exe));
    boot_cfg.log_progress = true;
    const guest::BootResult boot = guest::run_boot(boot_cfg);
    result.ok = boot.status == guest::BootStatus::Exited;
    result.exit_code = boot.stats.exit_code;
    result.stage = result.ok ? "ENTRY_CALL" : guest::boot_status_name(boot.status);
    if (!result.ok) {
        result.error = boot.summary;
    }
    return result;
#else
    emit_boot_milestone(cfg, "PGIM_LOAD",
                        "smoke shell: no linked guest shards (PG6_BOOT_LINKED_GUEST unset)");
    result.ok = true;
    result.exit_code = 0;
    result.stage = "PGIM_LOAD";
    return result;
#endif
#endif
}

}  // namespace playgta6::boot
