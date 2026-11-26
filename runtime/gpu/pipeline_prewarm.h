#pragma once

// Loading-phase prewarm queue (backlog 27).
//
// The title creates thousands of pipelines while it loads; a big compute program can take a driver
// tens of seconds to build. This queue lets the host build shader modules / pipelines ahead of the
// title's first use, on a background path, in the order earlier runs first used them.
//
// Sources of work:
//   * a recorded "warm list" (pipelines the title created in earlier runs, in first-use order), stored
//     by the host outside git and handed in with load_warm_list();
//   * the shader pack index (entries whose stage keys the pack cannot resolve are dropped).
// The title's own creations are reported with note_title_pipeline(): that records first-use order for
// the next run's warm list and retires a queued entry the title reached first.
//
// The queue is bookkeeping only (no GPU packets). The host pulls tasks with take(), builds the
// shader modules / pipelines, and reports back with complete(). Thread-safe.
//
// Warm list format (little endian): header {char magic[4] "PG6W", u32 version, u32 count, u32 0}.
//   version 1 (module-only): count entries of 24 bytes {u8 kind (0 graphics, 1 compute), u8 reserved[7],
//     u64 key0, u64 key1}. Graphics: key0 = vertex pack key, key1 = fragment pack key (0 = none).
//     Compute: key0 = compute pack key, key1 = 0.
//   version 2 (full pipelines): count entries of {u8 kind, u8 reserved[3], u32 detail_len, u64 key0,
//     u64 key1} (24 bytes) followed by detail_len bytes. detail_len 0 = a module-only entry as in v1;
//     otherwise the detail is an opaque pipeline description (prewarm_pipeline_desc.h: pipeline state key,
//     bind-group / pipeline layout, ...) that the host replays to create the pipeline ahead of the title.
//   A list without any detail is written as version 1, so older builds keep reading it; version 1 lists
//   always load.

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "gpu/shader_pack_index.h"

namespace playgta6::gpu {

inline constexpr uint32_t kPrewarmKindGraphics = 0;
inline constexpr uint32_t kPrewarmKindCompute = 1;
inline constexpr size_t kPrewarmListHeaderBytes = 16;
inline constexpr size_t kPrewarmListEntryBytes = 24;
inline constexpr uint32_t kPrewarmMaxEntries = 1u << 20;
inline constexpr uint32_t kPrewarmMaxDetailBytes = 1u << 20;  // per entry
inline constexpr uint32_t kPrewarmListVersionModules = 1;
inline constexpr uint32_t kPrewarmListVersionPipelines = 2;
inline constexpr uint32_t kPrewarmDefaultMaxInFlight = 4;

// A unit of work handed to the host. Plain data: it is copied across the wasm boundary (24 bytes).
struct PrewarmTask {
    uint32_t id = 0;    // pass back to complete()
    uint32_t kind = 0;  // kPrewarmKind*
    uint64_t key0 = 0;
    uint64_t key1 = 0;
};
static_assert(sizeof(PrewarmTask) == 24, "PrewarmTask is part of the host ABI");

struct PrewarmProgress {
    uint32_t total = 0;      // entries ever queued
    uint32_t queued = 0;     // waiting
    uint32_t in_flight = 0;  // taken by the host, not completed
    uint32_t done = 0;       // built ahead of the title
    uint32_t failed = 0;     // host reported failure
    uint32_t skipped = 0;    // the title created them first (or the pack cannot resolve them)
    uint32_t observed = 0;   // distinct pipelines the title created this run
    uint32_t reserved = 0;
    uint64_t longest_ns = 0;  // longest single build the host reported
    uint64_t total_ns = 0;    // summed build time
};
static_assert(sizeof(PrewarmProgress) == 48, "PrewarmProgress is part of the host ABI");

// Prewarm payoff counters (boot report): `prewarmed` built ahead of the title, `hit` of those the title
// then used (the pipeline was ready before first use), `wasted` built but never used (so far).
// The same counters for full pipelines (entries with a detail description) are kept apart.
struct PrewarmUse {
    uint32_t prewarmed = 0;
    uint32_t hit = 0;
    uint32_t wasted = 0;
    uint32_t pipelines_prewarmed = 0;
    uint32_t pipelines_hit = 0;
    uint32_t pipelines_wasted = 0;
};

using PrewarmDetail = std::shared_ptr<const std::vector<uint8_t>>;

class PipelinePrewarm {
public:
    // Pack used to drop entries it cannot resolve. May be null (then nothing is dropped).
    void set_pack_index(std::shared_ptr<const ShaderPackIndex> index);

    void set_max_in_flight(uint32_t n);

    // Parses a warm list and queues its entries in list order (= earlier first-use order). Entries
    // already queued, taken or observed are not queued again. False + message when malformed (the
    // queue is unchanged). `queued` receives how many were added.
    bool load_warm_list(const uint8_t* data, size_t size, std::string* error, uint32_t* queued = nullptr);

    // Serialises the pipelines the title created this run, in first-use order.
    std::vector<uint8_t> save_warm_list() const;

    // The title created a pipeline. True when it is new this run. A queued entry for it is retired.
    bool note_title_pipeline(uint32_t kind, uint64_t key0, uint64_t key1);
    // The same for a full pipeline: `detail` is the opaque description the host needs to recreate it
    // (empty = treated as note_title_pipeline). Distinct pipelines of one shader pair (different state
    // keys) are distinct entries.
    bool note_title_pipeline_full(uint32_t kind, uint64_t key0, uint64_t key1, std::vector<uint8_t> detail);
    // The title used a pipeline the host built ahead (task id from take()): counts as a hit and enters
    // the observed list (for the next run) like a title creation.
    bool note_task_used(uint32_t task_id);

    // Detail of a taken task (null for module-only entries or an unknown id).
    PrewarmDetail detail_of(uint32_t task_id) const;

    // Hands up to `max` tasks to the host in queue order, never exceeding the in-flight cap.
    size_t take(PrewarmTask* out, size_t max);

    // The host finished (or failed) a taken task. False for an unknown id.
    bool complete(uint32_t id, bool ok, uint64_t build_ns);
    // The host found the title had already created the pipeline when it came to build it: nothing was
    // built, the entry counts as skipped.
    bool complete_lost(uint32_t id);

    PrewarmProgress progress() const;
    PrewarmUse use_stats() const;

    void reset();

private:
    struct Identity {
        uint32_t kind;
        uint64_t k0;
        uint64_t k1;
        uint64_t k2;  // hash of the detail (0 = module-only)
        bool operator==(const Identity& o) const noexcept {
            return kind == o.kind && k0 == o.k0 && k1 == o.k1 && k2 == o.k2;
        }
    };
    struct IdentityHash {
        size_t operator()(const Identity& i) const noexcept {
            uint64_t h = i.k0 * 0x9E3779B97F4A7C15ull;
            h ^= (i.k1 + 0x7F4A7C15ull + (h << 6) + (h >> 2));
            h ^= (i.k2 + 0x7F4A7C15ull + (h << 6) + (h >> 2));
            return static_cast<size_t>(h ^ i.kind);
        }
    };
    enum class State : uint8_t { Queued, InFlight, Done, Used, Failed, Skipped };  // Used: Done, then the title used it
    struct Slot {
        Identity id;
        State state;
        PrewarmDetail detail;
    };

    bool resolvable(const Identity& i) const;
    bool note_locked(const Identity& id, PrewarmDetail detail);

    mutable std::mutex mu_;
    std::shared_ptr<const ShaderPackIndex> pack_;
    std::vector<Slot> slots_;                                           // task id - 1 -> slot
    std::unordered_map<Identity, uint32_t, IdentityHash> by_identity_;  // -> task id
    std::deque<uint32_t> queue_;                                        // task ids, queue order
    std::vector<Identity> observed_;                                    // title creations in first-use order
    std::vector<PrewarmDetail> observed_detail_;                        // parallel to observed_
    std::unordered_map<Identity, uint32_t, IdentityHash> observed_set_;
    uint32_t max_in_flight_ = kPrewarmDefaultMaxInFlight;
    uint32_t in_flight_ = 0;
    uint64_t longest_ns_ = 0;
    uint64_t total_ns_ = 0;
};

// Process-wide instance used by the boot layer and the title-side creation hooks.
PipelinePrewarm& pipeline_prewarm();

}  // namespace playgta6::gpu
