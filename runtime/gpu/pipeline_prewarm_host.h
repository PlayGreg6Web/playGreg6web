#pragma once

// Native side of the pipeline prewarm (backlog 27): the warm list file and the background builder.
//
// The warm list lives outside git in a per-title cache directory beside the object / lift caches
// (%LOCALAPPDATA%/playgta6/pipeline-warm/<title>.pg6w; ~/.cache/playgta6/pipeline-warm elsewhere;
// `--pipeline-warm-list <path>` overrides, "off" disables). It holds shader pack keys only, never game bytes.
//
// Boot: prewarm_host_configure() loads the file into pipeline_prewarm() (a missing or corrupt file is
// ignored); prewarm_host_start() runs a background thread that builds the queued pipelines through the
// device's normal creation path before the title first uses them. prewarm_host_save() writes the
// pipelines the title created this run, in first-use order (boot end, relaunch, suspend).

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "gpu/pipeline_prewarm.h"

namespace playgta6::gpu {

// "<cache root>/playgta6/pipeline-warm/<title>.pg6w" ("" when no cache root can be found).
std::string default_warm_list_path(const std::string& title);

// Reads a whole file; false when it cannot be opened.
bool read_warm_list_file(const std::string& path, std::vector<uint8_t>* out);
// Writes through a temporary file and a rename, so a crash never leaves a torn list. Creates the
// directory. A list without entries (no pipelines observed) is not written, so a short run never
// erases a good list.
bool write_warm_list_file(const std::string& path, const std::vector<uint8_t>& bytes);

// Sets the path ("" or "off" = disabled) and loads the file into pipeline_prewarm(). Returns how many
// entries were queued; `message` says why nothing was (missing, corrupt, ...).
uint32_t prewarm_host_configure(const std::string& path, std::string* message);

enum class PrewarmBuild : uint8_t {
    Built,   // created ahead of the title
    Lost,    // the title created it first (or while it was being built): nothing to count
    Failed,  // could not be built
};

struct PrewarmBuilder {
    std::function<bool()> ready;                                             // the device exists
    std::function<bool(uint32_t kind, uint64_t key0, uint64_t key1)> build;  // same path as on-demand
    // Full pipelines (warm-list entries with a detail description): builds the host pipeline through the
    // same creation path and ids the title would use, once, under the same lock a concurrent title
    // request takes. Optional: without it, a detailed entry is built like a module-only one (build).
    std::function<PrewarmBuild(uint32_t task_id, uint32_t kind, uint64_t key0, uint64_t key1,
                               const std::vector<uint8_t>& detail)>
        build_pipeline;
};
void prewarm_host_start(PrewarmBuilder builder);
void prewarm_host_wait();  // blocks until the queue is drained (then the thread has ended)
void prewarm_host_stop();  // asks the thread to end and joins it

// Saves this run's list to the configured path. No-op when not configured. True when a file was written.
bool prewarm_host_save();

// JSON object for the boot report (queued_total, prewarmed, hit_before_use, wasted, failed, ...).
std::string prewarm_host_report_json();

}  // namespace playgta6::gpu
