// Backlog 35: depth planes written by compute shaders.
//
// WebGPU has no storage access on depth/stencil formats, so a compute stage that writes a depth
// plane (the title's material classifier does, through an unordered-access view of the depth
// resource) is bound to a colour-format "twin" texture instead. This tracker records, per depth
// texture subresource (mip level x array layer) and per command list, whether the twin holds the depth plane's current contents, so the
// depth -> twin copy runs only when the depth side may have changed since the twin was last made
// equal to it, and the twin -> depth copy only after a dispatch that wrote the twin.
//
// Side states of one subresource:
//   Stale     the depth plane may differ from the twin (initial; a render pass, clear or copy wrote the
//             depth, or the twin was never filled in this list): copy depth -> twin before the next use.
//   Synced    twin and depth plane hold the same values: no copy needed.
//   TwinNewer a dispatch wrote the twin: the depth plane must be refreshed (twin -> depth).
// State is per command list: lists can be recorded in another order than they run, so no list may
// assume what an earlier one left in the depth plane. An untracked subresource is Stale.
// Copies: a depth write of any kind (render pass, clear, texture write) stales the covered
// subresources; a copy out of the depth texture (read-back) first writes pending twin data back.
// Stencil: the stencil plane has its own twin (an r32uint / r8uint texture holding the 8-bit value) and
// its own instance of this tracker (CommandListBridge::stencil_twin_): the stencil aspect is copied in by
// a compute pass, and written back after a dispatch by a pass that clears the plane to 0 and runs eight
// masked Replace draws (write mask and reference 1<<k, fragment discards where bit k of the twin is 0).
// The single-subresource overloads (mip 0, layer 0) are the plain 2D case.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace playgta6::gpu::bridge {

class DepthTwinTracker {
public:
    enum class Side : uint8_t { Stale = 0, Synced, TwinNewer };
    enum class SrvState : uint8_t { Stale = 0, Synced };
    static constexpr uint32_t kClearFlagDepth = 0x1u;
    static constexpr uint32_t kClearFlagStencil = 0x2u;

    struct Sub {
        uint32_t mip = 0;
        uint32_t layer = 0;

        bool operator==(const Sub& o) const noexcept { return mip == o.mip && layer == o.layer; }
        bool operator<(const Sub& o) const noexcept {
            return layer != o.layer ? layer < o.layer : mip < o.mip;
        }
    };

    // Synchronization plan before a depth texture is sampled as an SRV (SSAO/SSR/decals/shadows):
    //   flush_twin_to_depth  subresources where a compute stage wrote the twin (Side::TwinNewer)
    //                        and must be written back to the depth plane before sampling.
    //   refresh_depth_to_srv subresources where the depth plane was written/cleared/copied since
    //                        the SRV twin was last synced (and not already flushed above).
    struct SrvSyncPlan {
        std::vector<Sub> flush_twin_to_depth;
        std::vector<Sub> refresh_depth_to_srv;

        bool any() const noexcept {
            return !flush_twin_to_depth.empty() || !refresh_depth_to_srv.empty();
        }
    };

    Side side(uint32_t texture, uint32_t mip = 0, uint32_t layer = 0) const {
        const auto t = textures_.find(texture);
        if (t == textures_.end()) {
            return Side::Stale;
        }
        const auto it = t->second.find(key(mip, layer));
        return it == t->second.end() ? Side::Stale : it->second;
    }
    // The depth plane's contents must be copied into the twin before a dispatch uses it.
    bool needs_copy_in(uint32_t texture, uint32_t mip = 0, uint32_t layer = 0) const {
        return side(texture, mip, layer) == Side::Stale;
    }
    // The twin was written and the depth plane not yet refreshed.
    bool needs_copy_out(uint32_t texture, uint32_t mip = 0, uint32_t layer = 0) const {
        return side(texture, mip, layer) == Side::TwinNewer;
    }

    SrvState srv_state(uint32_t texture, uint32_t mip = 0, uint32_t layer = 0) const {
        if (side(texture, mip, layer) == Side::TwinNewer) {
            return SrvState::Stale;
        }
        const auto t = srv_state_.find(texture);
        if (t == srv_state_.end()) {
            return SrvState::Stale;
        }
        const auto it = t->second.find(key(mip, layer));
        return it == t->second.end() ? SrvState::Stale : it->second;
    }
    bool needs_srv_refresh(uint32_t texture, uint32_t mip = 0, uint32_t layer = 0) const {
        return srv_state(texture, mip, layer) != SrvState::Synced;
    }

    void note_copied_in(uint32_t texture, uint32_t mip = 0, uint32_t layer = 0) {
        const uint64_t k = key(mip, layer);
        textures_[texture][k] = Side::Synced;
        srv_state_[texture][k] = SrvState::Synced;
    }
    void note_copied_out(uint32_t texture, uint32_t mip = 0, uint32_t layer = 0) {
        const uint64_t k = key(mip, layer);
        textures_[texture][k] = Side::Synced;
        srv_state_[texture][k] = SrvState::Synced;
    }
    // Marks the depth-as-SRV twin for (mip, layer) up to date with the depth plane.
    void note_srv_synced(uint32_t texture, uint32_t mip = 0, uint32_t layer = 0) {
        const uint64_t k = key(mip, layer);
        if (side(texture, mip, layer) == Side::TwinNewer) {
            return;  // pending twin writes must be flushed (note_copied_out) before SRV is synced
        }
        srv_state_[texture][k] = SrvState::Synced;
    }
    // A dispatch wrote the twin (writable access).
    void note_twin_written(uint32_t texture, uint32_t mip = 0, uint32_t layer = 0) {
        const uint64_t k = key(mip, layer);
        textures_[texture][k] = Side::TwinNewer;
        srv_state_[texture][k] = SrvState::Stale;
        generation_[texture][k] = next_gen_++;
    }
    // Something wrote the depth plane of the given subresources (a render pass with depth writes, a
    // clear, a copy or an upload): the twin is behind. A count of 0 means "all remaining". A twin
    // with pending writes keeps them (they are flushed first by the caller).
    void note_depth_written(uint32_t texture, uint32_t base_mip, uint32_t mips, uint32_t base_layer,
                            uint32_t layers) {
        const uint64_t gen = next_gen_++;
        if (const auto t = textures_.find(texture); t != textures_.end()) {
            for (auto& [k, s] : t->second) {
                const Sub sub = unpack(k);
                if (in_range(sub, base_mip, mips, base_layer, layers)) {
                    if (s == Side::Synced) {
                        s = Side::Stale;
                    }
                    generation_[texture][k] = gen;
                }
            }
        }
        if (const auto st = srv_state_.find(texture); st != srv_state_.end()) {
            for (auto& [k, s] : st->second) {
                const Sub sub = unpack(k);
                if (in_range(sub, base_mip, mips, base_layer, layers)) {
                    s = SrvState::Stale;
                    generation_[texture][k] = gen;
                }
            }
        }
    }
    // Mip 0 / layer 0 only.
    void note_depth_written(uint32_t texture) { note_depth_written(texture, 0, 1, 0, 1); }
    // Every subresource of the texture.
    void note_depth_written_all(uint32_t texture) { note_depth_written(texture, 0, 0, 0, 0); }

    // ClearDepthStencilView synchronization: only clears with kClearFlagDepth (D3D12_CLEAR_FLAG_DEPTH)
    // modify the depth plane; stencil-only clears leave depth twin state untouched. A full-subresource
    // depth clear replaces every texel (discarding stale/pending twin state to Stale); a partial-rect
    // clear preserves pending TwinNewer writes so the caller flushes them before clearing.
    void note_depth_cleared(uint32_t texture, uint32_t clear_flags, uint32_t base_mip, uint32_t mips,
                            uint32_t base_layer, uint32_t layers, bool full_subresource = true) {
        if ((clear_flags & kClearFlagDepth) == 0u) {
            return;
        }
        if (!full_subresource) {
            note_depth_written(texture, base_mip, mips, base_layer, layers);
            return;
        }
        const uint64_t gen = next_gen_++;
        if (const auto t = textures_.find(texture); t != textures_.end()) {
            for (auto& [k, s] : t->second) {
                const Sub sub = unpack(k);
                if (in_range(sub, base_mip, mips, base_layer, layers)) {
                    s = Side::Stale;
                    generation_[texture][k] = gen;
                }
            }
        }
        if (const auto st = srv_state_.find(texture); st != srv_state_.end()) {
            for (auto& [k, s] : st->second) {
                const Sub sub = unpack(k);
                if (in_range(sub, base_mip, mips, base_layer, layers)) {
                    s = SrvState::Stale;
                    generation_[texture][k] = gen;
                }
            }
        }
    }

    // CopyTextureRegion / CopyResource / ResolveSubresource targeting a depth texture.
    void note_depth_copied(uint32_t texture, uint32_t base_mip, uint32_t mips, uint32_t base_layer,
                           uint32_t layers) {
        note_depth_written(texture, base_mip, mips, base_layer, layers);
    }

    // Subresources whose twin holds writes the depth plane has not received yet (a copy out of the
    // depth texture must flush these first). Sorted deterministically by (layer, mip).
    std::vector<Sub> pending_out(uint32_t texture) const {
        return pending_out(texture, 0, 0, 0, 0);
    }
    std::vector<Sub> pending_out(uint32_t texture, uint32_t base_mip, uint32_t mips, uint32_t base_layer,
                                 uint32_t layers) const {
        std::vector<Sub> out;
        const auto t = textures_.find(texture);
        if (t != textures_.end()) {
            for (const auto& [k, s] : t->second) {
                if (s == Side::TwinNewer) {
                    const Sub sub = unpack(k);
                    if (in_range(sub, base_mip, mips, base_layer, layers)) {
                        out.push_back(sub);
                    }
                }
            }
        }
        std::sort(out.begin(), out.end());
        return out;
    }
    bool has_pending_out(uint32_t texture) const { return has_pending_out(texture, 0, 0, 0, 0); }
    bool has_pending_out(uint32_t texture, uint32_t base_mip, uint32_t mips, uint32_t base_layer,
                         uint32_t layers) const {
        const auto t = textures_.find(texture);
        if (t == textures_.end()) {
            return false;
        }
        for (const auto& [k, s] : t->second) {
            if (s == Side::TwinNewer && in_range(unpack(k), base_mip, mips, base_layer, layers)) {
                return true;
            }
        }
        return false;
    }

    // Computes which subresources of `texture` in [base_mip, +mips) x [base_layer, +layers) need
    // synchronization before an SRV samples the depth target. Counts of 0 mean "all tracked
    // remaining subresources".
    SrvSyncPlan plan_srv_read(uint32_t texture, uint32_t base_mip, uint32_t mips, uint32_t base_layer,
                              uint32_t layers) const {
        SrvSyncPlan plan;
        plan.flush_twin_to_depth = pending_out(texture, base_mip, mips, base_layer, layers);
        std::unordered_set<uint64_t> flushed;
        flushed.reserve(plan.flush_twin_to_depth.size());
        for (const Sub& s : plan.flush_twin_to_depth) {
            flushed.insert(key(s.mip, s.layer));
        }

        // Bounded explicit subresource range: enumerate every (mip, layer) in the SRV view.
        if (mips > 0 && layers > 0 && mips <= 64u && layers <= 2048u) {
            for (uint32_t l = 0; l < layers; ++l) {
                for (uint32_t m = 0; m < mips; ++m) {
                    const uint32_t mip = base_mip + m;
                    const uint32_t layer = base_layer + l;
                    const uint64_t k = key(mip, layer);
                    if (flushed.count(k) != 0) {
                        continue;  // flushing twin -> depth leaves both depth and SRV twin equal
                    }
                    if (srv_state(texture, mip, layer) != SrvState::Synced) {
                        plan.refresh_depth_to_srv.push_back(Sub{mip, layer});
                    }
                }
            }
            return plan;
        }

        // Open-ended range (0 or unbounded sentinel): inspect all tracked subresources in range.
        std::unordered_set<uint64_t> candidates;
        if (const auto t = textures_.find(texture); t != textures_.end()) {
            for (const auto& [k, s] : t->second) {
                const Sub sub = unpack(k);
                if (in_range(sub, base_mip, mips, base_layer, layers) &&
                    srv_state(texture, sub.mip, sub.layer) != SrvState::Synced) {
                    candidates.insert(k);
                }
            }
        }
        if (const auto st = srv_state_.find(texture); st != srv_state_.end()) {
            for (const auto& [k, s] : st->second) {
                const Sub sub = unpack(k);
                if (in_range(sub, base_mip, mips, base_layer, layers) && s != SrvState::Synced) {
                    candidates.insert(k);
                }
            }
        }
        for (uint64_t k : candidates) {
            if (flushed.count(k) == 0) {
                plan.refresh_depth_to_srv.push_back(unpack(k));
            }
        }
        std::sort(plan.refresh_depth_to_srv.begin(), plan.refresh_depth_to_srv.end());
        return plan;
    }

    // Applies a previously computed SrvSyncPlan once its flush/refresh passes have been emitted.
    void apply_srv_sync(uint32_t texture, const SrvSyncPlan& plan) {
        for (const Sub& s : plan.flush_twin_to_depth) {
            note_copied_out(texture, s.mip, s.layer);
        }
        for (const Sub& s : plan.refresh_depth_to_srv) {
            note_srv_synced(texture, s.mip, s.layer);
        }
    }

    uint64_t write_generation(uint32_t texture, uint32_t mip = 0, uint32_t layer = 0) const {
        const auto t = generation_.find(texture);
        if (t == generation_.end()) {
            return 0;
        }
        const auto it = t->second.find(key(mip, layer));
        return it == t->second.end() ? 0 : it->second;
    }

    void forget(uint32_t texture) {
        textures_.erase(texture);
        srv_state_.erase(texture);
        generation_.erase(texture);
    }
    void reset() {
        textures_.clear();
        srv_state_.clear();
        generation_.clear();
        next_gen_ = 1;
    }
    // Tracked subresources over all textures (compute twin map).
    size_t tracked() const {
        size_t n = 0;
        for (const auto& t : textures_) {
            n += t.second.size();
        }
        return n;
    }
    // Tracked depth-as-SRV subresources over all textures.
    size_t tracked_srv() const {
        size_t n = 0;
        for (const auto& t : srv_state_) {
            n += t.second.size();
        }
        return n;
    }

private:
    static uint64_t key(uint32_t mip, uint32_t layer) {
        return (uint64_t{layer} << 32) | uint64_t{mip};
    }
    static Sub unpack(uint64_t k) {
        return Sub{static_cast<uint32_t>(k & 0xFFFFFFFFu), static_cast<uint32_t>(k >> 32)};
    }
    static bool in_range(const Sub& sub, uint32_t base_mip, uint32_t mips, uint32_t base_layer,
                         uint32_t layers) {
        return sub.mip >= base_mip && (mips == 0 || sub.mip - base_mip < mips) &&
               sub.layer >= base_layer && (layers == 0 || sub.layer - base_layer < layers);
    }
    std::unordered_map<uint32_t, std::unordered_map<uint64_t, Side>> textures_;
    std::unordered_map<uint32_t, std::unordered_map<uint64_t, SrvState>> srv_state_;
    std::unordered_map<uint32_t, std::unordered_map<uint64_t, uint64_t>> generation_;
    uint64_t next_gen_ = 1;
};

}  // namespace playgta6::gpu::bridge
