// Per shader-hash table of the draws / dispatches the D3D12 layer dropped, by reason, for the boot
// report (tools/program_census.py weights unrunnable programs with it). Bounded: at most
// kMaxHashes distinct hashes are tracked (first come); the rest accumulate in one "other" bucket.
// Only the drop path touches it (one mutex + hash-map update), never a draw that was emitted.
#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace playgta6::gpu::d3d12_guest {

class DropTable {
public:
    static constexpr size_t kMaxHashes = 512;
    enum class Reason : uint8_t { NoModule, MeshRefused, Bridge, RectList, Count };
    static constexpr size_t kReasons = static_cast<size_t>(Reason::Count);

    struct Counts {
        uint64_t draws[kReasons] = {};
        uint64_t dispatches[kReasons] = {};
        uint64_t total() const {
            uint64_t t = 0;
            for (size_t i = 0; i < kReasons; ++i) {
                t += draws[i] + dispatches[i];
            }
            return t;
        }
    };

    static const char* reason_name(Reason r) {
        switch (r) {
        case Reason::NoModule: return "no_shader_module";
        case Reason::MeshRefused: return "mesh_refused";
        case Reason::Bridge: return "dropped_by_bridge";
        case Reason::RectList: return "rect_list";
        case Reason::Count: break;
        }
        return "?";
    }

    // Records `n` dropped draws (or dispatches) against one shader identity.
    void record(uint64_t hash, Reason reason, bool dispatch, uint64_t n = 1) {
        std::lock_guard<std::mutex> lock(mu_);
        Counts* c = &other_;
        auto it = map_.find(hash);
        if (it != map_.end()) {
            c = &it->second;
        } else if (map_.size() < kMaxHashes) {
            c = &map_[hash];
        }
        (dispatch ? c->dispatches : c->draws)[static_cast<size_t>(reason)] += n;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mu_);
        map_.clear();
        other_ = Counts{};
    }

    size_t tracked() const {
        std::lock_guard<std::mutex> lock(mu_);
        return map_.size();
    }

    // {"0x<16 hex>": total, ...} (hashes only, the shape program_census --drop-counts reads), sorted
    // by total descending.
    std::string by_program_json() const;
    // {"tracked","max_hashes","other":{draws,dispatches,reasons},"entries":[{hash,draws,dispatches,reasons}]}
    std::string detail_json() const;

private:
    std::vector<std::pair<uint64_t, Counts>> sorted() const;
    mutable std::mutex mu_;
    std::unordered_map<uint64_t, Counts> map_;
    Counts other_;
};

DropTable& drop_table();

}  // namespace playgta6::gpu::d3d12_guest
