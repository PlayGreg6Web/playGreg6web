// Native executor: GPU / process memory over time and the lifetime of objects the executor's own
// caches hold past the stream's DestroyResource. Pure logic, no GPU.
//
// A std-pack boot lost the device at ~150 s (ID3D12Device::CreateDescriptorHeap E_OUTOFMEMORY, and every
// new device failed the same way). The boot report had no numbers to say what ran out, so the executor
// samples DXGI QueryVideoMemoryInfo, the process / system commit and the live object counts per kind
// every few seconds into `memory_timeline` (MemoryTimeline below keeps a bounded, evenly thinned series).
//
// The caches that look objects up by handle (view_meta: view -> subresource range, bg_infos: bind group ->
// its buffer / texture bindings) hold a reference to every texture, buffer and bind group they describe,
// so a destroyed texture stayed alive in Dawn until the cache wrapped (8192 views / 16384 groups).
// RetiredSet collects what DestroyResource dropped; the executor purges those caches of it once the
// frame that may still record them is encoded.
#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace playgta6::gpu::native_exec {

// Bytes of a texture's storage: every mip of every layer (3D: depth halves with the mips), `bits` per
// texel (block-compressed formats: bits per texel of the block, 4 or 8), times the sample count.
inline uint64_t texture_storage_bytes(uint32_t width, uint32_t height, uint32_t depth_or_layers, uint32_t mips,
                                      uint32_t samples, uint32_t bits_per_texel, bool is_3d) {
    if (width == 0 || height == 0) return 0;
    if (mips == 0) mips = 1;
    if (samples == 0) samples = 1;
    uint64_t bits = 0;
    for (uint32_t m = 0; m < mips && m < 32; ++m) {
        const uint64_t w = width >> m ? width >> m : 1;
        const uint64_t h = height >> m ? height >> m : 1;
        const uint64_t d = is_3d ? (depth_or_layers >> m ? depth_or_layers >> m : 1) : (depth_or_layers ? depth_or_layers : 1);
        bits += w * h * d * bits_per_texel;
    }
    return (bits + 7) / 8 * samples;
}

// One sample of the memory timeline. Sizes in bytes; counts are the executor's live handles per kind
// (what its tables and caches hold: Dawn frees an object once nothing references it).
struct MemorySample {
    uint64_t t_ms = 0;           // since the executor started
    uint64_t presents = 0;
    // DXGI QueryVideoMemoryInfo for this process (0 when unavailable).
    uint64_t gpu_local_usage = 0, gpu_local_budget = 0;
    uint64_t gpu_nonlocal_usage = 0, gpu_nonlocal_budget = 0;
    // Process private commit / working set and the system-wide commit charge and limit.
    uint64_t process_private = 0, process_working_set = 0;
    uint64_t system_commit = 0, system_commit_limit = 0;
    // Live handles per kind.
    uint64_t buffers = 0, buffer_bytes = 0;
    uint64_t textures = 0, texture_bytes = 0, texture_views = 0;
    uint64_t twin_textures = 0, twin_bytes = 0;          // depth / norm16 twins of live textures
    uint64_t alias_textures = 0, alias_texture_bytes = 0;  // storage-alias twin pool
    uint64_t alias_buffers = 0, alias_buffer_bytes = 0;    // storage-alias scratch pool
    uint64_t snap_textures = 0, snap_texture_bytes = 0;    // usage-scope snapshot pool
    uint64_t snap_buffers = 0, snap_buffer_bytes = 0;
    uint64_t dummy_objects = 0;                            // dummy buffers / views / samplers / textures
    uint64_t bind_groups = 0, bind_group_layouts = 0, pipeline_layouts = 0;
    uint64_t bind_groups_evicted = 0;  // ids whose Dawn group the bind group budget dropped (rebuilt on bind)
    uint64_t render_pipelines = 0, compute_pipelines = 0, shader_modules = 0, samplers = 0, query_sets = 0;
    uint64_t bg_infos = 0, bg_descs = 0, view_meta = 0, remap_cache = 0;
    // Textures the caches still reference although the stream destroyed them (should stay ~0).
    uint64_t dead_textures_pinned = 0, dead_texture_bytes_pinned = 0;
    uint64_t retired_pending = 0;  // destroyed handles waiting for the next purge
    uint64_t journal_bytes = 0;
    uint64_t pending_maps = 0;  // read-back / timer maps in flight
    const char* tag = "";       // "" periodic, "device_lost", "final", ...
};

// A bounded series: at most `capacity` samples spanning the whole run. When full, every second sample is
// dropped and the sampling interval doubles, so the series stays evenly spaced. Tagged samples (device
// lost) are kept apart and never thinned.
class MemoryTimeline {
public:
    explicit MemoryTimeline(size_t capacity = 128, uint64_t interval_ms = 5000)
        : capacity_(capacity < 4 ? 4 : capacity), interval_ms_(interval_ms ? interval_ms : 1) {}

    // True when a periodic sample is due at `t_ms`.
    bool due(uint64_t t_ms) const { return samples_.empty() || t_ms >= last_ms_ + interval_ms_; }

    void push(const MemorySample& s) {
        if (s.tag && *s.tag) {
            if (tagged_.size() < 32) tagged_.push_back(s);
            return;
        }
        last_ms_ = s.t_ms;
        samples_.push_back(s);
        if (samples_.size() >= capacity_) {
            std::vector<MemorySample> kept;
            kept.reserve(capacity_ / 2 + 1);
            for (size_t i = 0; i < samples_.size(); i += 2) kept.push_back(samples_[i]);
            samples_.swap(kept);
            interval_ms_ *= 2;
        }
        note_peak(s);
    }

    const std::vector<MemorySample>& samples() const { return samples_; }
    const std::vector<MemorySample>& tagged() const { return tagged_; }
    uint64_t interval_ms() const { return interval_ms_; }
    const MemorySample& peak() const { return peak_; }  // the sample with the highest local GPU usage

private:
    void note_peak(const MemorySample& s) {
        if (!have_peak_ || s.gpu_local_usage > peak_.gpu_local_usage) {
            peak_ = s;
            have_peak_ = true;
        }
    }
    size_t capacity_;
    uint64_t interval_ms_;
    uint64_t last_ms_ = 0;
    std::vector<MemorySample> samples_;
    std::vector<MemorySample> tagged_;
    MemorySample peak_;
    bool have_peak_ = false;
};

// Handles DestroyResource dropped from the id tables, kept until the caches are purged of them. `H` is
// any handle type with Get() (the wgpu C++ wrappers); the set is keyed by the raw pointer and the handle
// is held so the address cannot be reused by a new object before the purge.
template <class H>
class RetiredSet {
public:
    void add(const H& h) {
        if (!h) return;
        if (keys_.insert(h.Get()).second) held_.push_back(h);
    }
    bool contains(const void* key) const { return key != nullptr && keys_.count(key) != 0; }
    bool empty() const { return keys_.empty(); }
    size_t size() const { return keys_.size(); }
    void clear() {
        keys_.clear();
        held_.clear();
    }

private:
    std::unordered_set<const void*> keys_;
    std::vector<H> held_;
};

// Erases from `cache` (a map keyed by a handle pointer) every entry `dead(entry)` says references a retired
// object. Returns the number erased.
template <class Map, class Dead>
size_t purge_cache(Map& cache, Dead dead) {
    size_t n = 0;
    for (auto it = cache.begin(); it != cache.end();) {
        if (dead(*it)) {
            it = cache.erase(it);
            ++n;
        } else {
            ++it;
        }
    }
    return n;
}

}  // namespace playgta6::gpu::native_exec
