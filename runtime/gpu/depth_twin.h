// Backlog 35: depth planes written by compute shaders.
//
// WebGPU has no storage access on depth/stencil formats, so a compute stage that writes a depth
// plane (the title's material classifier does, through an unordered-access view of the depth
// resource) is bound to a colour-format "twin" texture instead. This tracker records, per depth
// texture and per command list, whether the twin holds the depth plane's current contents, so the
// depth -> twin copy runs only when the depth side may have changed since the twin was last made
// equal to it, and the twin -> depth copy only after a dispatch that wrote the twin.
//
// Side states of one texture:
//   Stale     the depth plane may differ from the twin (initial; a render pass wrote the depth, or the
//             twin was never filled in this list): copy depth -> twin before the next use.
//   Synced    twin and depth plane hold the same values: no copy needed.
//   TwinNewer a dispatch wrote the twin: the depth plane must be refreshed (twin -> depth).
// State is per command list: lists can be recorded in another order than they run, so no list may
// assume what an earlier one left in the depth plane.
#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace playgta6::gpu::bridge {

class DepthTwinTracker {
public:
    enum class Side : uint8_t { Stale = 0, Synced, TwinNewer };

    Side side(uint32_t texture) const {
        const auto it = sides_.find(texture);
        return it == sides_.end() ? Side::Stale : it->second;
    }
    // The depth plane's contents must be copied into the twin before a dispatch uses it.
    bool needs_copy_in(uint32_t texture) const { return side(texture) == Side::Stale; }
    // The twin was written and the depth plane not yet refreshed.
    bool needs_copy_out(uint32_t texture) const { return side(texture) == Side::TwinNewer; }

    void note_copied_in(uint32_t texture) { sides_[texture] = Side::Synced; }
    void note_copied_out(uint32_t texture) { sides_[texture] = Side::Synced; }
    // A dispatch wrote the twin (writable access).
    void note_twin_written(uint32_t texture) { sides_[texture] = Side::TwinNewer; }
    // Something wrote the depth plane (a render pass with depth writes, a clear or copy): the twin is
    // behind. A twin with pending writes keeps them (they are flushed first by the caller).
    void note_depth_written(uint32_t texture) {
        const auto it = sides_.find(texture);
        if (it != sides_.end() && it->second == Side::Synced) {
            it->second = Side::Stale;
        }
    }
    void forget(uint32_t texture) { sides_.erase(texture); }
    void reset() { sides_.clear(); }
    size_t tracked() const { return sides_.size(); }

private:
    std::unordered_map<uint32_t, Side> sides_;
};

}  // namespace playgta6::gpu::bridge
