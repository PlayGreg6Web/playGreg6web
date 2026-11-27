// Per shader-hash table of the draws / dispatches that were recorded with at least one binding bound
// to a null/dummy resource because the sidecar entry was unresolved or the descriptor slot was empty
// (descriptor_heap BindResourceKind::Null). No draw is dropped by this; the picture may be wrong.
// Whole-run and per draws/frame phase (phase_hash_counts.h). Only the fallback path touches it (one
// mutex + hash-map update per draw that bound a null entry), never a clean draw.
#pragma once

#include <mutex>
#include <string>
#include <unordered_map>

#include "gpu/phase_hash_counts.h"

namespace playgta6::gpu::d3d12_guest {

class BindingFallbackTable {
public:
    static constexpr size_t kMaxHashes = 512;
    enum class Reason : uint8_t { Unresolved, EmptySlot, Count };
    static constexpr size_t kReasons = static_cast<size_t>(Reason::Count);
    using Counts = HashCounts<kReasons>;

    static const char* reason_name(size_t r) { return r == 0 ? "unresolved_entry" : r == 1 ? "empty_slot" : "?"; }

    void record(uint64_t hash, Reason reason, bool dispatch, uint64_t n = 1) {
        std::lock_guard<std::mutex> lock(mu_);
        const size_t r = static_cast<size_t>(reason);
        Counts* c = &other_;
        auto it = map_.find(hash);
        if (it != map_.end()) {
            c = &it->second;
        } else if (map_.size() < kMaxHashes) {
            c = &map_[hash];
        }
        (dispatch ? c->dispatches : c->draws)[r] += n;
        phases_.add(current_gpu_phase(), hash, r, dispatch, n);
    }
    void reset() {
        std::lock_guard<std::mutex> lock(mu_);
        map_.clear();
        other_ = Counts{};
        phases_.clear();
    }
    size_t tracked() const {
        std::lock_guard<std::mutex> lock(mu_);
        return map_.size();
    }
    // {"tracked","max_hashes","other":{...},"entries":[...],"phases":[...]}
    std::string detail_json() const;

private:
    mutable std::mutex mu_;
    std::unordered_map<uint64_t, Counts> map_;
    Counts other_;
    PhaseHashMaps<kReasons> phases_;
};

BindingFallbackTable& binding_fallback_table();

}  // namespace playgta6::gpu::d3d12_guest
