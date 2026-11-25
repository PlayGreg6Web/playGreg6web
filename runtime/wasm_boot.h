#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "gpu/pso_cache.h"
#include "gpu/stage_reflection.h"

struct Pg6Program;

namespace playgta6::boot {

inline constexpr char kPgimMagic[4] = {'P', 'G', 'I', 'M'};
inline constexpr size_t kPgimMagicBytes = 4;

// Default pso::WgpuFeature bitmask matching the optional WebGPU adapter features negotiated by
// web/workers/wgpu_worker.js (W92) that the C++ GPU bridge gates on.
inline constexpr uint32_t kDefaultWasmWgpuFeatures =
    gpu::pso::WgpuFeature::TextureCompressionBc |
    gpu::pso::WgpuFeature::Depth32FloatStencil8 |
    gpu::pso::WgpuFeature::Rg11b10UfloatRenderable |
    gpu::pso::WgpuFeature::TextureFormatsTier1 |
    gpu::pso::WgpuFeature::Bgra8UnormStorage |
    gpu::pso::WgpuFeature::Float32Filterable;
inline constexpr uint32_t kDefaultDStorageWorkerThreads = 4;

using HttpfsOpenFn = std::function<int32_t(std::string_view rel_path)>;
using HttpfsReadFn = std::function<int64_t(int32_t handle, uint64_t offset, void* dst, size_t len)>;
using HttpfsCloseFn = std::function<void(int32_t handle)>;
using LogFn = std::function<void(std::string_view message)>;

// Browser HTTPFS guest program loader. Streams `program-manifest.json` and `.pgim` images
// through `playgta6::os::httpfs_*` (or injected callbacks) and emits W40 BootTracer lines.
struct WasmBootConfig {
    std::string manifest_path = "boot/program-manifest.json";
    std::string fallback_pgim_path = "boot/guest.pgim";
    std::string shader_pack_path = "shaders/splash.wgsl.pack";
    std::string fallback_shader_pack_path = "shaders/fallback.wgsl.pack";
    size_t read_chunk_bytes = 65536;
    size_t max_file_bytes = 512 * 1024 * 1024;
    HttpfsOpenFn open_fn;   // empty: playgta6::os::httpfs_open
    HttpfsReadFn read_fn;   // empty: playgta6::os::httpfs_read
    HttpfsCloseFn close_fn; // empty: playgta6::os::httpfs_close
    LogFn log_fn;           // empty: playgta6::os::log
    // Opt-in pipeline dump directory (empty: disabled unless PG6_DUMP_PIPELINES env
    // or HTTPFS `shaders/dump_pipelines.flag` is present). Kept trailing for ABI safety.
    std::string dump_pipelines_dir;
    // Optional WebGPU device features (gpu::pso::WgpuFeature bits) forwarded to
    // DeviceOptions::bridge.features. Kept trailing for ABI safety.
    uint32_t wgpu_features = kDefaultWasmWgpuFeatures;
    // DirectStorage worker thread count (clamped to [1, 16] by set_dstorage_worker_threads).
    uint32_t dstorage_worker_threads = kDefaultDStorageWorkerThreads;
};

struct WasmBootResult {
    bool ok = false;
    bool linked_guest_available = false;
    std::string stage;
    std::string error;
    size_t modules_loaded = 0;
    uint64_t total_bytes_loaded = 0;
    int32_t exit_code = 0;
};

// Reads `rel_path` in bounded chunks. Always closes a successfully opened handle.
bool load_file_via_httpfs(const WasmBootConfig& cfg,
                          const std::string& rel_path,
                          std::vector<uint8_t>& out,
                          std::string& error_out);

// W40 BootTracer-compatible line: `[boot] <MILESTONE>` or `[boot] <MILESTONE>: <detail>`.
void emit_boot_milestone(const WasmBootConfig& cfg,
                         const char* milestone,
                         const std::string& detail = "");

// Optional stage-reflection sidecar (G62/G63): reads `<pack_path>.reflect.json` over HTTPFS and
// parses it into a gpu::StageReflectionIndex. Missing/unreadable sidecar: false, silently.
// Malformed sidecar: false plus a `SHADER_PACK` warning milestone (boot continues with the
// layer's conventions). Valid: true, a `SHADER_PACK` loaded milestone, and `out_reflection` set.
bool try_load_stage_reflection(const WasmBootConfig& cfg,
                               const std::string& pack_path,
                               std::shared_ptr<gpu::StageReflectionIndex>& out_reflection);

// Zero-dependency scan of a program-manifest.json buffer for `.pgim` path strings
// (not `.pgim.json`). Order is first-seen; duplicates are dropped.
std::vector<std::string> extract_pgim_paths_from_manifest(std::string_view json);

bool has_pgim_magic(std::span<const uint8_t> bytes);

// Matches a Pg6ProgramModule::module build name (e.g. "main", "msvcp140", "vcruntime140_1")
// against manifest `.pgim` paths like "boot/main/image/guest.pgim" or "main/image/guest.pgim",
// falling back to "boot/<module_name>/image/guest.pgim".
std::string resolve_module_pgim_path(std::string_view module_name,
                                     std::span<const std::string> manifest_pgim_paths);

// Multi-module `Pg6Program` HTTPFS boot: validates `program`, streams every module's `.pgim`
// over HTTPFS, prepares `guest::ProgramBoot`, and invokes `guest::run_boot`.
WasmBootResult run_wasm_program_boot(const WasmBootConfig& cfg, const Pg6Program& program);

// Streams the manifest / fallback `.pgim` over HTTPFS, validates PGIM magic, emits
// `PGIM_LOAD`, and either hooks the linked guest boot (`PG6_BOOT_LINKED_GUEST` /
// `PG6_BOOT_PROGRAM`) or returns a smoke-shell diagnostic when shards are not linked.
WasmBootResult run_wasm_guest_boot(const WasmBootConfig& cfg);

}  // namespace playgta6::boot
