#include "wasm_boot.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>

#include "abi/playgta6_abi.h"
#include "gpu/d3d12_guest_objects.h"
#include "gpu/d3d12_xs_exports.h"
#include "gpu/pipeline_prewarm.h"
#include "gpu/shader_pack_index.h"
#include "gpu/stage_reflection.h"
#include "guest/boot.h"
#include "guest/program_loader.h"
#include "os/httpfs_mailbox.h"
#include "os/image_loader.h"
#include "os/os_shim.h"
#include "os/win32_compat.h"
#include "os/win32_kernel.h"
#include "os/xgamesave_local.h"
#include "os/xbox_scarlett.h"
#include "os/xbox_scarlett_guest.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define PG6_WASM_BOOT_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define PG6_WASM_BOOT_EXPORT
#endif

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

std::atomic<const WasmBootConfig*> g_active_boot_cfg{nullptr};
struct alignas(64) AlignedWasmInputState {
    playgta6::InputState state{};
};
AlignedWasmInputState g_wasm_input_state{};
static_assert(sizeof(playgta6::InputState) == 40, "InputState layout must stay 40 bytes");
static_assert(alignof(AlignedWasmInputState) == 64,
              "g_wasm_input_state must be 64-byte aligned");
static_assert(offsetof(AlignedWasmInputState, state) == 0,
              "InputState must sit at offset 0 of AlignedWasmInputState");

void wasm_off_thread_fault() {
    const WasmBootConfig* active = g_active_boot_cfg.load(std::memory_order_acquire);
    if (active != nullptr) {
        write_log(*active, "[boot] off-thread guest fault captured");
    } else {
        os::log("[boot] off-thread guest fault captured");
    }
    std::abort();
}

struct ScopedActiveBootConfig {
    const WasmBootConfig* prev = nullptr;

    explicit ScopedActiveBootConfig(const WasmBootConfig* cfg)
        : prev(g_active_boot_cfg.exchange(cfg, std::memory_order_acq_rel)) {}
    ~ScopedActiveBootConfig() {
        g_active_boot_cfg.store(prev, std::memory_order_release);
    }
    ScopedActiveBootConfig(const ScopedActiveBootConfig&) = delete;
    ScopedActiveBootConfig& operator=(const ScopedActiveBootConfig&) = delete;
};

// Bridges Win32 CreateFile/GetFileAttributes and Xbox DirectStorage (XCRDAPI) onto
// open_path/read_path/close_path whenever no Win32 HTTPFS client is already installed.
// Direct mode: reads go straight into the caller's buffer (no mailbox staging copy), one chunk
// per FIFO lock turn, so a bulk DirectStorage read no longer holds every other reader (Bink,
// ReadFile on the title's threads) off for its whole length.
class ScopedWin32HttpfsBridge {
public:
    explicit ScopedWin32HttpfsBridge(const WasmBootConfig& cfg) : cfg_(&cfg) {
        previous_ = os::win32::get_httpfs_client();
        if (previous_ != nullptr) {
            return;
        }
        os::HttpfsDirectBackend backend;
        backend.open = [this](std::string_view rel) { return open_rel(rel); };
        backend.read = [this](int32_t handle, uint64_t offset, void* dst, size_t len) -> int64_t {
            return handle < 0 ? -9 : read_path(*cfg_, handle, offset, dst, len);
        };
        backend.close = [this](int32_t handle) {
            if (handle < 0) {
                return false;
            }
            close_path(*cfg_, handle);
            return true;
        };
        client_ = std::make_unique<os::HttpfsMailboxClient>(std::move(backend));
        os::win32::set_httpfs_client(client_.get());
        installed_ = true;
    }

    ~ScopedWin32HttpfsBridge() {
        if (installed_) {
            os::win32::set_httpfs_client(previous_);
        }
    }

    ScopedWin32HttpfsBridge(const ScopedWin32HttpfsBridge&) = delete;
    ScopedWin32HttpfsBridge& operator=(const ScopedWin32HttpfsBridge&) = delete;

private:
    int32_t open_rel(std::string_view raw) {
        std::string rel(raw);
        std::replace(rel.begin(), rel.end(), '\\', '/');
        for (;;) {
            if (rel.size() >= 2 && rel[0] == '.' && rel[1] == '/') {
                rel.erase(0, 2);
            } else if (!rel.empty() && rel.front() == '/') {
                rel.erase(0, 1);
            } else {
                break;
            }
        }
        if (rel.empty()) {
            return -2;
        }
        const std::string lower = to_lower_ascii(rel);
        const bool has_root_prefix = lower.rfind("game/", 0) == 0 || lower.rfind("boot/", 0) == 0;
        int32_t handle = -1;
        if (!has_root_prefix) {
            handle = open_path(*cfg_, "game/" + rel);
        }
        if (handle < 0) {
            handle = open_path(*cfg_, rel);
        }
        return handle;
    }

    const WasmBootConfig* cfg_ = nullptr;
    os::HttpfsMailboxClient* previous_ = nullptr;
    bool installed_ = false;
    std::unique_ptr<os::HttpfsMailboxClient> client_;
};

struct ScopedDeviceOptions {
    bool installed = false;

    void install(gpu::d3d12_guest::DeviceOptions options) {
        gpu::d3d12_xs_exports::set_device_options(std::move(options));
        installed = true;
    }

    ~ScopedDeviceOptions() {
        if (installed) {
            gpu::d3d12_xs_exports::set_device_options(gpu::d3d12_guest::DeviceOptions{});
        }
    }

    ScopedDeviceOptions() = default;
    ScopedDeviceOptions(const ScopedDeviceOptions&) = delete;
    ScopedDeviceOptions& operator=(const ScopedDeviceOptions&) = delete;
};

struct ScopedDStorageWorkerThreads {
    explicit ScopedDStorageWorkerThreads(uint32_t count) {
        os::scarlett::guest_api::set_dstorage_worker_threads(count);
    }

    ~ScopedDStorageWorkerThreads() {
        os::scarlett::guest_api::set_dstorage_worker_threads(1);
    }

    ScopedDStorageWorkerThreads(const ScopedDStorageWorkerThreads&) = delete;
    ScopedDStorageWorkerThreads& operator=(const ScopedDStorageWorkerThreads&) = delete;
};

constexpr uint64_t kMaxShaderPackTocBytes = 64ULL * 1024 * 1024;

enum class ExactReadStatus {
    Ok,
    IoError,
    UnexpectedEof,
};

ExactReadStatus read_exact_from_handle(const WasmBootConfig& cfg,
                                       int32_t handle,
                                       uint64_t offset,
                                       uint8_t* dst,
                                       size_t want_bytes,
                                       size_t* bytes_read_out = nullptr) {
    const size_t chunk = cfg.read_chunk_bytes == 0 ? 65536 : cfg.read_chunk_bytes;
    size_t got_total = 0;
    while (got_total < want_bytes) {
        const size_t step = std::min(chunk, want_bytes - got_total);
        const int64_t n = read_path(cfg, handle, offset + got_total, dst + got_total, step);
        if (n < 0 || static_cast<uint64_t>(n) > step) {
            if (bytes_read_out != nullptr) {
                *bytes_read_out = got_total;
            }
            return ExactReadStatus::IoError;
        }
        if (n == 0) {
            if (bytes_read_out != nullptr) {
                *bytes_read_out = got_total;
            }
            return ExactReadStatus::UnexpectedEof;
        }
        got_total += static_cast<size_t>(n);
    }
    if (bytes_read_out != nullptr) {
        *bytes_read_out = got_total;
    }
    return ExactReadStatus::Ok;
}

bool try_load_shader_pack(const WasmBootConfig& cfg,
                          const std::string& path,
                          bool is_fallback,
                          std::shared_ptr<gpu::ShaderPackIndex>& out_index) {
    if (path.empty()) {
        return false;
    }

    const int32_t handle = open_path(cfg, path);
    if (handle < 0) {
        return false;
    }
    HandleCloser closer(cfg, handle);

    uint8_t header[gpu::kShaderPackHeaderBytes]{};
    size_t header_got = 0;
    const ExactReadStatus header_status = read_exact_from_handle(
        cfg, handle, 0, header, gpu::kShaderPackHeaderBytes, &header_got);
    if (header_status == ExactReadStatus::IoError) {
        return false;
    }

    uint32_t entry_count = 0;
    uint32_t pack_version = 1;
    std::string parse_error;
    if (!gpu::parse_shader_pack_header(header, header_got, &entry_count, &parse_error, &pack_version)) {
        emit_boot_milestone(cfg, "SHADER_PACK", path + " invalid: " + parse_error);
        return false;
    }

    const uint64_t entries_bytes =
        uint64_t{entry_count} * uint64_t{gpu::shader_pack_entry_bytes(pack_version)};
    const uint64_t toc_bytes = uint64_t{gpu::kShaderPackHeaderBytes} + entries_bytes;
    const uint64_t toc_cap =
        (cfg.max_file_bytes != 0 && cfg.max_file_bytes < kMaxShaderPackTocBytes)
            ? static_cast<uint64_t>(cfg.max_file_bytes)
            : kMaxShaderPackTocBytes;
    if (toc_bytes > toc_cap) {
        emit_boot_milestone(cfg, "SHADER_PACK",
                            path + " invalid: WGPK TOC exceeds " + std::to_string(toc_cap) +
                                " byte limit (" + std::to_string(toc_bytes) + " bytes)");
        return false;
    }

    std::vector<uint8_t> toc(static_cast<size_t>(toc_bytes));
    std::memcpy(toc.data(), header, gpu::kShaderPackHeaderBytes);
    if (entries_bytes > 0) {
        size_t entries_got = 0;
        const ExactReadStatus toc_status = read_exact_from_handle(
            cfg,
            handle,
            gpu::kShaderPackHeaderBytes,
            toc.data() + gpu::kShaderPackHeaderBytes,
            static_cast<size_t>(entries_bytes),
            &entries_got);
        if (toc_status == ExactReadStatus::IoError) {
            return false;
        }
        if (toc_status == ExactReadStatus::UnexpectedEof) {
            toc.resize(gpu::kShaderPackHeaderBytes + entries_got);
        }
    }

    // Close the HTTPFS handle immediately; WGSL payloads are never read into Wasm64 memory.
    closer.reset();

    auto index = std::make_shared<gpu::ShaderPackIndex>();
    if (!gpu::parse_shader_pack_index_toc(
            toc.data(), toc.size(), /*total_file_size=*/0, index.get(), &parse_error)) {
        emit_boot_milestone(cfg, "SHADER_PACK", path + " invalid: " + parse_error);
        return false;
    }
    const char* mode = is_fallback ? "fallback" : "primary";
    emit_boot_milestone(cfg, "SHADER_PACK",
                        path + " loaded (" + std::to_string(index->size()) + " key(s), " + mode +
                            ")");
    out_index = std::move(index);
    return true;
}

std::string resolve_pipeline_dump_dir(const WasmBootConfig& cfg) {
    if (!cfg.dump_pipelines_dir.empty()) {
        return cfg.dump_pipelines_dir;
    }
    if (const char* env_dir = std::getenv("PG6_DUMP_PIPELINES");
        env_dir != nullptr && env_dir[0] != '\0') {
        return std::string(env_dir);
    }
    const int32_t flag_handle = open_path(cfg, "shaders/dump_pipelines.flag");
    if (flag_handle >= 0) {
        close_path(cfg, flag_handle);
        return "/pg6_dump";
    }
    return {};
}

}  // namespace

bool try_load_stage_reflection(const WasmBootConfig& cfg,
                               const std::string& pack_path,
                               std::shared_ptr<gpu::StageReflectionIndex>& out_reflection) {
    if (pack_path.empty()) {
        return false;
    }
    const std::string sidecar_path = pack_path + ".reflect.json";
    std::vector<uint8_t> bytes;
    std::string read_error;
    if (!load_file_via_httpfs(cfg, sidecar_path, bytes, read_error)) {
        return false;  // The sidecar is optional.
    }
    const std::string json_str(bytes.begin(), bytes.end());
    auto reflection = std::make_shared<gpu::StageReflectionIndex>();
    std::string parse_error;
    if (!gpu::parse_stage_reflection_json(json_str, reflection.get(), &parse_error)) {
        emit_boot_milestone(cfg, "SHADER_PACK", sidecar_path + " invalid: " + parse_error);
        return false;
    }
    emit_boot_milestone(cfg, "SHADER_PACK",
                        sidecar_path + " loaded (" + std::to_string(reflection->size()) +
                            " stage(s))");
    out_reflection = std::move(reflection);
    return true;
}

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

namespace {
bool g_relaunch_pending = false;
std::string g_relaunch_args;  // last relaunch request, read by the browser loader through exports

std::string with_title_args(std::string base, const std::string& title_args) {
    if (!title_args.empty()) {
        base += ' ';
        base += title_args;
    }
    return base;
}

void finish_relaunch(WasmBootResult& result) {
    std::string args;
    result.relaunch_requested = os::win32::relaunch_requested(&args) &&
        static_cast<uint32_t>(result.exit_code) == os::win32::kRelaunchSelfExitCode;
    result.relaunch_args = result.relaunch_requested ? args : std::string();
    g_relaunch_args = result.relaunch_args;
    g_relaunch_pending = result.relaunch_requested;
}
}  // namespace

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
    program_boot.config.sandbox_id = "RETAIL";
    program_boot.config.working_directory = "G:\\";
    program_boot.config.command_line = with_title_args("game_scarlett_master.exe", cfg.title_args);
    program_boot.config.off_thread_fault = &wasm_off_thread_fault;

    ScopedActiveBootConfig active_boot_cfg(&cfg);
    ScopedWin32HttpfsBridge httpfs_bridge(cfg);
    ScopedDStorageWorkerThreads dstorage_threads_guard(cfg.dstorage_worker_threads);
    ScopedDeviceOptions device_options_guard;

    // O36 / #796: attach linear-memory InputState so GameInput reads browser inputSab updates.
    if (playgta6::InputState* current_input = os::scarlett::get_game_input_state();
        current_input == nullptr || current_input == &g_wasm_input_state.state) {
        attach_default_input_state();
    }

    std::shared_ptr<gpu::ShaderPackIndex> shader_pack;
    std::string loaded_pack_path;
    if (try_load_shader_pack(cfg, cfg.shader_pack_path, /*is_fallback=*/false, shader_pack)) {
        loaded_pack_path = cfg.shader_pack_path;
    } else {
        if (!cfg.fallback_shader_pack_path.empty() &&
            cfg.fallback_shader_pack_path != cfg.shader_pack_path) {
            if (!cfg.shader_pack_path.empty()) {
                emit_boot_milestone(cfg, "SHADER_PACK",
                                    "primary " + cfg.shader_pack_path +
                                        " unavailable; falling back to " +
                                        cfg.fallback_shader_pack_path);
            }
            if (try_load_shader_pack(
                    cfg, cfg.fallback_shader_pack_path, /*is_fallback=*/true, shader_pack)) {
                loaded_pack_path = cfg.fallback_shader_pack_path;
            }
        }
        if (!shader_pack &&
            (!cfg.shader_pack_path.empty() || !cfg.fallback_shader_pack_path.empty())) {
            emit_boot_milestone(cfg, "SHADER_PACK", "none loaded (draws will use unresolved PSOs)");
        }
    }

    // Stage reflection sidecar (G62/G63): `<pack>.reflect.json` next to the pack that loaded.
    std::shared_ptr<gpu::StageReflectionIndex> stage_reflection;
    if (shader_pack && !loaded_pack_path.empty()) {
        (void)try_load_stage_reflection(cfg, loaded_pack_path, stage_reflection);
    }

    const std::string dump_dir = resolve_pipeline_dump_dir(cfg);
    if (!dump_dir.empty()) {
        emit_boot_milestone(cfg, "PIPELINE_DUMP", dump_dir);
    }

    gpu::d3d12_guest::DeviceOptions device_options;
    gpu::pipeline_prewarm().set_pack_index(shader_pack);
    device_options.shader_pack = std::move(shader_pack);
    device_options.stage_reflection = std::move(stage_reflection);
    device_options.dump_dir = dump_dir;
    device_options.bridge.features = cfg.wgpu_features;
    device_options_guard.install(std::move(device_options));

    // O36: optional run options (wall-time bound, report to the log) from boot/boot-options.json.
    guest::load_boot_options(program_boot.config, [&cfg](const std::string& path, std::vector<uint8_t>& out) {
        std::string error;
        return load_file_via_httpfs(cfg, path, out, error);
    });
    os::win32::reset_relaunch_request();
    const guest::BootResult boot = guest::run_boot(program_boot.config);
    if (auto dev = gpu::d3d12_xs_exports::current_device()) {
        if (!dump_dir.empty()) {
            dev->flush_pipeline_dump();
        }
        const gpu::d3d12_guest::DeviceStats s = dev->stats();
        emit_boot_milestone(cfg, "GPU_STATS",
                            "psos=" + std::to_string(s.pipeline_states) +
                                " submissions=" + std::to_string(s.submissions) +
                                " presents=" + std::to_string(s.presents) +
                                " skipped_draws=" + std::to_string(s.skipped_draws) +
                                " shadow_bytes=" + std::to_string(s.shadow_bytes) +
                                " shadow_resident_bytes=" + std::to_string(s.shadow_resident_bytes));
    }
    result.ok = boot.status == guest::BootStatus::Exited;
    result.exit_code = boot.stats.exit_code;
    finish_relaunch(result);
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
    if (playgta6::InputState* current_input = os::scarlett::get_game_input_state();
        current_input == nullptr || current_input == &g_wasm_input_state.state) {
        attach_default_input_state();
    }
    boot_cfg.command_line = with_title_args("game.exe", cfg.title_args);
    os::win32::reset_relaunch_request();
    const guest::BootResult boot = guest::run_boot(boot_cfg);
    result.ok = boot.status == guest::BootStatus::Exited;
    result.exit_code = boot.stats.exit_code;
    finish_relaunch(result);
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

playgta6::InputState* input_state_ptr() noexcept {
    return &g_wasm_input_state.state;
}

void attach_default_input_state() {
    os::scarlett::set_game_input_state(&g_wasm_input_state.state, os::SeqlockMode::MonotonicSeq);
#ifdef __EMSCRIPTEN__
    // O36: a key tap shorter than the title's frame must still reach it (press latch).
    os::scarlett::set_game_input_press_sampler(true);
#endif
}

}  // namespace playgta6::boot

// Pipeline prewarm queue (backlog 27). The host stores the warm list outside git, loads it with
// playgta6_prewarm_load_list, pulls tasks, builds shader modules / pipelines, and reports back.
extern "C" PG6_WASM_BOOT_EXPORT int32_t playgta6_prewarm_load_list(const uint8_t* data, uint64_t size) {
    uint32_t queued = 0;
    std::string error;
    if (!playgta6::gpu::pipeline_prewarm().load_warm_list(data, static_cast<size_t>(size), &error, &queued)) {
        return -1;
    }
    return static_cast<int32_t>(queued);
}

// Writes up to `max` 24-byte PrewarmTask records to `out`; returns how many.
extern "C" PG6_WASM_BOOT_EXPORT int32_t playgta6_prewarm_take(playgta6::gpu::PrewarmTask* out, uint32_t max) {
    return static_cast<int32_t>(playgta6::gpu::pipeline_prewarm().take(out, max));
}

extern "C" PG6_WASM_BOOT_EXPORT int32_t playgta6_prewarm_complete(uint32_t id, int32_t ok, uint64_t build_ns) {
    return playgta6::gpu::pipeline_prewarm().complete(id, ok != 0, build_ns) ? 1 : 0;
}

// Writes the 48-byte PrewarmProgress to `out`.
extern "C" PG6_WASM_BOOT_EXPORT void playgta6_prewarm_progress(playgta6::gpu::PrewarmProgress* out) {
    *out = playgta6::gpu::pipeline_prewarm().progress();
}

// Returns the byte size of the warm list (pipelines this run created); copies it to `out` when cap fits.
extern "C" PG6_WASM_BOOT_EXPORT uint64_t playgta6_prewarm_save_list(uint8_t* out, uint64_t cap) {
    const std::vector<uint8_t> bytes = playgta6::gpu::pipeline_prewarm().save_warm_list();
    if (out != nullptr && cap >= bytes.size()) {
        std::memcpy(out, bytes.data(), bytes.size());
    }
    return bytes.size();
}

extern "C" PG6_WASM_BOOT_EXPORT uintptr_t playgta6_get_input_state_ptr(void) {
    return reinterpret_cast<uintptr_t>(playgta6::boot::input_state_ptr());
}

// Browser relaunch contract (XLaunchNewGame): after run_wasm_guest_boot returns, the loader reads
// playgta6_relaunch_requested(); when 1 it tears the worker down and boots again with the same
// options plus the arguments at playgta6_relaunch_args_ptr() (NUL-terminated UTF-8, valid until the
// next boot), passed as WasmBootConfig::title_args.
extern "C" PG6_WASM_BOOT_EXPORT int32_t playgta6_relaunch_requested(void) {
    return playgta6::boot::g_relaunch_pending ? 1 : 0;
}

extern "C" PG6_WASM_BOOT_EXPORT const char* playgta6_relaunch_args_ptr(void) {
    return playgta6::boot::g_relaunch_args.c_str();
}

// Browser storage contract (saves). The runtime writes saves below the save dir of the in-memory
// filesystem; the host (web/) mirrors them to IndexedDB/OPFS. After any completed save write
// playgta6_storage_generation() increases. The host polls it (or on visibilitychange/pagehide) and,
// when it changed, calls playgta6_storage_take_changed() which returns the container directories
// changed since the last take as newline-separated paths (clears the list; pointer valid until the next
// call), then persists those trees; removed paths show up as paths that no longer exist.
extern "C" PG6_WASM_BOOT_EXPORT uint64_t playgta6_storage_generation(void) {
    return playgta6::os::gdk::gamesave::storage_generation();
}

extern "C" PG6_WASM_BOOT_EXPORT const char* playgta6_storage_take_changed(void) {
    static std::string joined;
    joined.clear();
    for (const std::string& p : playgta6::os::gdk::gamesave::take_changed_paths()) {
        joined += p;
        joined += '\n';
    }
    return joined.c_str();
}
