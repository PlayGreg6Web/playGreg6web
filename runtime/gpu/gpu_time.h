// Per-pipeline GPU time (backlog 40), fed from WebGPU timestamp queries by whatever executes the
// command stream (the browser worker; the native harness has no GPU executor and reports
// "unavailable"). Two parts, both pure logic:
//  - GpuTimeRing: a ring of query sets, one per frame in flight. A frame hands out begin/end query
//    index pairs per pass, the executor resolves the whole set once per frame and calls complete().
//    A frame whose set is still unresolved when its slot comes round again is skipped (counted), so
//    the measurement never stalls the frame.
//  - GpuTimeTable: aggregates nanoseconds per (shader hash, pass kind); bounded like DropTable.
// Numbers only ever come from complete() / record(); with no support nothing is invented.
#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace playgta6::gpu::d3d12_guest {

enum class GpuPassKind : uint8_t { Render, Compute, Copy, Count };
const char* gpu_pass_kind_name(GpuPassKind k);

class GpuTimeTable {
public:
    static constexpr size_t kMaxKeys = 1024;
    enum class Support : uint8_t { Unknown, Unavailable, Available };

    struct Stats {
        uint64_t passes = 0;
        uint64_t total_ns = 0;
        uint64_t max_ns = 0;
    };
    struct Entry {
        uint64_t hash;
        GpuPassKind kind;
        Stats stats;
    };

    void set_support(Support s, const std::string& reason = std::string());
    Support support() const;
    // Adds one measured pass. Implies Available.
    void record(uint64_t hash, GpuPassKind kind, uint64_t ns);
    void note_skipped_frame();
    void note_skipped_passes(uint64_t n);
    void note_measured_frame(uint64_t gpu_ns);
    // Frames measured so far and their summed GPU time (the overlay averages the difference of two samples).
    void measured_totals(uint64_t* frames, uint64_t* gpu_ns) const;
    void reset();
    size_t tracked() const;
    // Top n keys by total time.
    std::vector<Entry> top(size_t n) const;
    // {"available","support","reason"?,"frames_measured","frames_skipped","passes_skipped",
    //  "frame_gpu_ns","total_ns","tracked","other":{..},"top":[{hash,kind,passes,total_ns,max_ns,avg_ns}]}
    std::string json(size_t top_n = 32) const;

private:
    mutable std::mutex mu_;
    Support support_ = Support::Unknown;
    std::string reason_;
    std::unordered_map<uint64_t, Stats> map_;
    std::unordered_map<uint64_t, std::pair<uint64_t, GpuPassKind>> keys_;
    Stats other_;
    uint64_t frames_measured_ = 0, frames_skipped_ = 0, passes_skipped_ = 0, frame_ns_ = 0;
};

class GpuTimeRing {
public:
    static constexpr int kNoSlot = -1;
    GpuTimeRing(size_t slots, size_t max_passes_per_frame);

    // Opens a frame in the next slot; returns kNoSlot when that slot still awaits its resolve (the
    // frame is then unmeasured and add_pass returns false for it).
    int begin_frame();
    // Reserves two query indices (begin, end) in the open frame's query set.
    bool add_pass(uint64_t hash, GpuPassKind kind, uint32_t* begin_query, uint32_t* end_query);
    // Closes the frame; its set is in flight until complete(). Returns the passes recorded.
    size_t end_frame(int slot);
    // Resolved ticks of `slot` (2 per pass, in add_pass order); `period_ns` is the tick length. A
    // zero or non-increasing pair counts as unmeasured, never as time.
    void complete(int slot, const uint64_t* ticks, size_t count, double period_ns, GpuTimeTable& table);
    // Drops an in-flight slot (lost device / failed map).
    void abandon(int slot, GpuTimeTable& table);
    size_t slots() const { return slots_.size(); }
    size_t query_count() const { return max_passes_ * 2; }
    uint64_t overflow_passes() const { return overflow_; }

private:
    struct Slot {
        bool in_flight = false;
        std::vector<std::pair<uint64_t, GpuPassKind>> passes;
    };
    std::vector<Slot> slots_;
    size_t max_passes_;
    size_t next_ = 0;
    int open_ = kNoSlot;
    uint64_t overflow_ = 0;
};

GpuTimeTable& gpu_time_table();

}  // namespace playgta6::gpu::d3d12_guest
