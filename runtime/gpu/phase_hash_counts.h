// Per-phase, per-shader-hash counters shared by the drop table (drop_table.h) and the binding
// fallback table (binding_fallbacks.h). The phase is the draws/frame regime (the same split as the
// boot report's frame_times): the present path publishes the regime of the frame that just ended with
// set_current_gpu_phase(), and records made during the next frame are attributed to it (so a record
// lands in the regime of the PREVIOUS frame; at phase boundaries this is off by one frame, which does
// not matter for "which stage of the boot"). Bounded: at most kMaxHashes distinct hashes per phase;
// the rest accumulate in that phase's "other".
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace playgta6::gpu::d3d12_guest {

inline constexpr size_t kGpuPhases = 5;
inline const char* gpu_phase_name(size_t p) {
    static const char* const n[kGpuPhases] = {"draws_0", "draws_1_9", "draws_10_99", "draws_100_999",
                                              "draws_1000_plus"};
    return p < kGpuPhases ? n[p] : "?";
}
inline size_t gpu_phase_for_draws(uint64_t draws) {
    return draws == 0 ? 0 : draws < 10 ? 1 : draws < 100 ? 2 : draws < 1000 ? 3 : 4;
}
inline std::atomic<uint32_t>& gpu_phase_slot() {
    static std::atomic<uint32_t> s{0};
    return s;
}
inline size_t current_gpu_phase() { return gpu_phase_slot().load(std::memory_order_relaxed); }
inline void set_current_gpu_phase(size_t p) {
    gpu_phase_slot().store(static_cast<uint32_t>(p < kGpuPhases ? p : kGpuPhases - 1), std::memory_order_relaxed);
}

template <size_t N>
struct HashCounts {
    uint64_t draws[N] = {};
    uint64_t dispatches[N] = {};
    uint64_t total() const { return sum_draws() + sum_dispatches(); }
    uint64_t sum_draws() const {
        uint64_t t = 0;
        for (size_t i = 0; i < N; ++i) t += draws[i];
        return t;
    }
    uint64_t sum_dispatches() const {
        uint64_t t = 0;
        for (size_t i = 0; i < N; ++i) t += dispatches[i];
        return t;
    }
};

// Not thread safe: the owner holds its own mutex.
template <size_t N>
class PhaseHashMaps {
public:
    static constexpr size_t kMaxHashes = 512;
    static constexpr size_t kTopPerPhase = 32;
    using Counts = HashCounts<N>;

    void add(size_t phase, uint64_t hash, size_t reason, bool dispatch, uint64_t n) {
        if (phase >= kGpuPhases) phase = kGpuPhases - 1;
        auto& m = maps_[phase];
        Counts* c = &other_[phase];
        auto it = m.find(hash);
        if (it != m.end()) {
            c = &it->second;
        } else if (m.size() < kMaxHashes) {
            c = &m[hash];
        }
        (dispatch ? c->dispatches : c->draws)[reason] += n;
    }
    void clear() {
        for (auto& m : maps_) m.clear();
        other_ = {};
    }
    size_t tracked(size_t phase) const { return maps_[phase].size(); }

    // [{"name","tracked","draws","dispatches","other":{draws,dispatches},"entries":[top N by total]}, ...]
    // for the non-empty phases. `reason_name(i)` names reason i.
    template <class NameFn>
    std::string json(NameFn reason_name) const {
        std::string o = "[";
        bool firstp = true;
        for (size_t p = 0; p < kGpuPhases; ++p) {
            if (maps_[p].empty() && other_[p].total() == 0) continue;
            std::vector<std::pair<uint64_t, Counts>> v(maps_[p].begin(), maps_[p].end());
            std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
                const uint64_t ta = a.second.total(), tb = b.second.total();
                return ta != tb ? ta > tb : a.first < b.first;
            });
            uint64_t draws = other_[p].sum_draws(), disp = other_[p].sum_dispatches();
            for (const auto& e : v) {
                draws += e.second.sum_draws();
                disp += e.second.sum_dispatches();
            }
            o += std::string(firstp ? "" : ",") + "\n    {\"name\": \"" + gpu_phase_name(p) +
                 "\", \"tracked\": " + std::to_string(maps_[p].size()) + ", \"draws\": " + std::to_string(draws) +
                 ", \"dispatches\": " + std::to_string(disp) + ", \"other\": {\"draws\": " +
                 std::to_string(other_[p].sum_draws()) + ", \"dispatches\": " +
                 std::to_string(other_[p].sum_dispatches()) + "}, \"entries\": [";
            firstp = false;
            bool first = true;
            for (size_t i = 0; i < v.size() && i < kTopPerPhase; ++i) {
                char h[24];
                std::snprintf(h, sizeof h, "0x%016" PRIx64, v[i].first);
                o += std::string(first ? "" : ", ") + "{\"hash\": \"" + h + "\", \"draws\": " +
                     std::to_string(v[i].second.sum_draws()) + ", \"dispatches\": " +
                     std::to_string(v[i].second.sum_dispatches()) + ", \"reasons\": {";
                first = false;
                bool fr = true;
                for (size_t r = 0; r < N; ++r) {
                    const uint64_t c = v[i].second.draws[r] + v[i].second.dispatches[r];
                    if (c == 0) continue;
                    o += std::string(fr ? "" : ",") + "\"" + reason_name(r) + "\":" + std::to_string(c);
                    fr = false;
                }
                o += "}}";
            }
            o += "]}";
        }
        return o + (firstp ? "]" : "\n  ]");
    }

private:
    std::array<std::unordered_map<uint64_t, Counts>, kGpuPhases> maps_{};
    std::array<Counts, kGpuPhases> other_{};
};

}  // namespace playgta6::gpu::d3d12_guest
