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
// Stencil: not twinned. The stencil plane is still counted as storage_views_depth_untwinned. A
// write-back is possible in WebGPU only as 8 bit-masked stencil passes (replace op, write mask 1<<k,
// reference 1<<k for set bits and 0 for clear bits, two shader variants discarding on the other
// polarity), which needs its own twin texture, a stencil-aspect axis here and 16 extra pipelines; it
// has no consumer yet (the title's classifier writes depth) and no GPU is available to validate it.
// The single-subresource overloads (mip 0, layer 0) are the plain 2D case.
#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace playgta6::gpu::bridge {

class DepthTwinTracker {
public:
    enum class Side : uint8_t { Stale = 0, Synced, TwinNewer };
    struct Sub {
        uint32_t mip = 0;
        uint32_t layer = 0;
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

    void note_copied_in(uint32_t texture, uint32_t mip = 0, uint32_t layer = 0) {
        textures_[texture][key(mip, layer)] = Side::Synced;
    }
    void note_copied_out(uint32_t texture, uint32_t mip = 0, uint32_t layer = 0) {
        textures_[texture][key(mip, layer)] = Side::Synced;
    }
    // A dispatch wrote the twin (writable access).
    void note_twin_written(uint32_t texture, uint32_t mip = 0, uint32_t layer = 0) {
        textures_[texture][key(mip, layer)] = Side::TwinNewer;
    }
    // Something wrote the depth plane of the given subresources (a render pass with depth writes, a
    // clear, a copy or an upload): the twin is behind. A count of 0 means "all remaining". A twin
    // with pending writes keeps them (they are flushed first by the caller).
    void note_depth_written(uint32_t texture, uint32_t base_mip, uint32_t mips, uint32_t base_layer,
                            uint32_t layers) {
        const auto t = textures_.find(texture);
        if (t == textures_.end()) {
            return;
        }
        for (auto& [k, side] : t->second) {
            const uint32_t mip = k & 0xFFu;
            const uint32_t layer = k >> 8;
            if (side == Side::Synced && mip >= base_mip && (mips == 0 || mip - base_mip < mips) &&
                layer >= base_layer && (layers == 0 || layer - base_layer < layers)) {
                side = Side::Stale;
            }
        }
    }
    // Mip 0 / layer 0 only.
    void note_depth_written(uint32_t texture) { note_depth_written(texture, 0, 1, 0, 1); }
    // Every subresource of the texture.
    void note_depth_written_all(uint32_t texture) { note_depth_written(texture, 0, 0, 0, 0); }
    // Subresources whose twin holds writes the depth plane has not received yet (a copy out of the
    // depth texture must flush these first).
    std::vector<Sub> pending_out(uint32_t texture) const {
        std::vector<Sub> out;
        const auto t = textures_.find(texture);
        if (t != textures_.end()) {
            for (const auto& [k, side] : t->second) {
                if (side == Side::TwinNewer) {
                    out.push_back({k & 0xFFu, k >> 8});
                }
            }
        }
        return out;
    }
    void forget(uint32_t texture) { textures_.erase(texture); }
    void reset() { textures_.clear(); }
    // Tracked subresources over all textures.
    size_t tracked() const {
        size_t n = 0;
        for (const auto& t : textures_) {
            n += t.second.size();
        }
        return n;
    }

private:
    static uint32_t key(uint32_t mip, uint32_t layer) { return (layer << 8) | (mip & 0xFFu); }
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, Side>> textures_;
};

}  // namespace playgta6::gpu::bridge
