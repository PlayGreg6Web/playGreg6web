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
// Warm list format (little endian): header {char magic[4] "PG6W", u32 version (1), u32 count, u32 0},
// then count entries of 24 bytes {u8 kind (0 graphics, 1 compute), u8 reserved[7], u64 key0, u64 key1}.
// Graphics: key0 = vertex pack key, key1 = fragment pack key (0 = none). Compute: key0 = compute pack
// key, key1 = 0.

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

    // Hands up to `max` tasks to the host in queue order, never exceeding the in-flight cap.
    size_t take(PrewarmTask* out, size_t max);

    // The host finished (or failed) a taken task. False for an unknown id.
    bool complete(uint32_t id, bool ok, uint64_t build_ns);

    PrewarmProgress progress() const;

    void reset();

private:
    struct Identity {
        uint32_t kind;
        uint64_t k0;
        uint64_t k1;
        bool operator==(const Identity& o) const noexcept { return kind == o.kind && k0 == o.k0 && k1 == o.k1; }
    };
    struct IdentityHash {
        size_t operator()(const Identity& i) const noexcept {
            uint64_t h = i.k0 * 0x9E3779B97F4A7C15ull;
            h ^= (i.k1 + 0x7F4A7C15ull + (h << 6) + (h >> 2));
            return static_cast<size_t>(h ^ i.kind);
        }
    };
    enum class State : uint8_t { Queued, InFlight, Done, Failed, Skipped };
    struct Slot {
        Identity id;
        State state;
    };

    bool resolvable(const Identity& i) const;

    mutable std::mutex mu_;
    std::shared_ptr<const ShaderPackIndex> pack_;
    std::vector<Slot> slots_;                                           // task id - 1 -> slot
    std::unordered_map<Identity, uint32_t, IdentityHash> by_identity_;  // -> task id
    std::deque<uint32_t> queue_;                                        // task ids, queue order
    std::vector<Identity> observed_;                                    // title creations in first-use order
    std::unordered_map<Identity, uint32_t, IdentityHash> observed_set_;
    uint32_t max_in_flight_ = kPrewarmDefaultMaxInFlight;
    uint32_t in_flight_ = 0;
    uint64_t longest_ns_ = 0;
    uint64_t total_ns_ = 0;
};

// Process-wide instance used by the boot layer and the title-side creation hooks.
PipelinePrewarm& pipeline_prewarm();

}  // namespace playgta6::gpu
