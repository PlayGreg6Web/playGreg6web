// WebGPU usage scopes (the "usage scope" validation Dawn runs when a pass is finished): inside one scope a
// buffer or texture subresource with a writable usage (storage read-write, write-only / read-write storage
// texture) may have no other usage. A render pass is ONE scope (every bind group set in it, whether a draw
// uses it or not); in a compute pass each dispatch is its own scope (the groups bound at the dispatch).
// D3D12 has no such rule: the title binds a buffer as a CBV and a UAV in the same draw (often disjoint
// ranges of one big buffer), or writes a UAV in one draw and reads it as a CBV / SRV in the next one
// after a barrier. Dawn rejects the pass ("usage (Uniform|Storage(read-write)) includes writable usage
// and another usage in the same synchronization scope") and every draw in it is lost.
//
// The executor binds bind groups lazily (at the draw / dispatch) and checks the groups bound there:
//  - a read-only use (uniform, read-only storage, sampled, read-only storage texture) of a resource the
//    same draw / dispatch also writes is bound from a snapshot: a copy of the read range taken right
//    before it (the pass is split there, the copy goes between the halves). That is what the read sees
//    on D3D12 too (the draw's own writes are a race there); when the read range overlaps a written range
//    the snapshot is retaken before every later draw / dispatch with the same groups ("volatile").
//  - a render draw whose uses conflict with uses of earlier draws in the pass splits the render pass
//    (end, restart with load ops): the writes of the earlier draws land and are visible to the later ones,
//    exactly as across a D3D12 barrier.
// Two writable uses of one resource (writable storage aliasing) are a different rule and not handled here.
// Pure; no GPU involved.
#pragma once

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace playgta6::gpu::native_exec {

enum class UseKind : uint8_t { Uniform, StorageRead, StorageWrite, Sampled, StorageTexRead, StorageTexWrite };

inline bool use_writable(UseKind k) { return k == UseKind::StorageWrite || k == UseKind::StorageTexWrite; }

// One bound resource of a bind group that can take part in a conflict: storage-capable buffers and
// textures only (a resource without storage usage is never written through a binding).
struct ResUse {
    const void* res = nullptr;  // the buffer / texture identity (its native handle)
    bool texture = false;
    UseKind kind = UseKind::Uniform;
    uint32_t entry = 0;  // index into the bind group's entries
    // Buffer: the bound range (static offset; dyn_index >= 0: plus that dynamic offset of the SetBindGroup).
    uint64_t offset = 0, size = 0;
    int32_t dyn_index = -1;
    // Texture: the view's subresources (resolved counts) and aspects (bit 0 colour / depth, bit 1 stencil).
    uint32_t mip_base = 0, mip_count = 1, layer_base = 0, layer_count = 1;
    uint8_t aspects = 1;
};

struct BgUses {
    std::vector<ResUse> uses;
    bool any_writable = false;
    void add(const ResUse& u) {
        uses.push_back(u);
        any_writable = any_writable || use_writable(u.kind);
    }
};

// The resolved subresource range of a texture view (WebGPU defaults for the counts left 0).
// tex_dim: 0 1d, 1 2d, 2 3d (texture); view_dim: 0 = undefined, else the view dimension code
// (1 1d, 2 2d, 3 2d-array, 4 cube, 5 cube-array, 6 3d).
struct ViewRange {
    uint32_t mip_base = 0, mip_count = 1, layer_base = 0, layer_count = 1;
    uint8_t view_dim = 2;  // resolved view dimension code
};
inline ViewRange resolve_view_range(uint8_t tex_dim, uint32_t tex_mips, uint32_t tex_layers, uint8_t view_dim,
                                    uint32_t base_mip, uint32_t mip_count, uint32_t base_layer, uint32_t layer_count) {
    ViewRange r;
    const uint32_t layers = tex_dim == 2 ? 1u : std::max<uint32_t>(1, tex_layers);
    if (view_dim == 0) view_dim = tex_dim == 2 ? 6 : tex_dim == 0 ? 1 : layers > 1 ? 3 : 2;
    r.view_dim = view_dim;
    r.mip_base = std::min(base_mip, tex_mips ? tex_mips - 1 : 0);
    r.mip_count = mip_count ? mip_count : (tex_mips > r.mip_base ? tex_mips - r.mip_base : 1);
    r.layer_base = tex_dim == 2 ? 0 : base_layer;
    if (layer_count) r.layer_count = layer_count;
    else if (view_dim == 3 || view_dim == 5) r.layer_count = layers > r.layer_base ? layers - r.layer_base : 1;
    else if (view_dim == 4) r.layer_count = 6;
    else r.layer_count = 1;
    return r;
}

inline bool spans_overlap(uint64_t a, uint64_t an, uint64_t b, uint64_t bn) { return a < b + bn && b < a + an; }

inline uint64_t use_offset(const ResUse& u, const std::vector<uint32_t>* dyn) {
    if (u.dyn_index < 0 || !dyn || static_cast<size_t>(u.dyn_index) >= dyn->size()) return u.offset;
    return u.offset + (*dyn)[static_cast<size_t>(u.dyn_index)];
}

// Same resource, and (textures) at least one subresource in common.
inline bool uses_touch(const ResUse& a, const ResUse& b) {
    if (a.res != b.res || a.texture != b.texture || !a.res) return false;
    if (!a.texture) return true;  // a buffer's usage is tracked for the whole buffer
    return (a.aspects & b.aspects) && spans_overlap(a.mip_base, a.mip_count, b.mip_base, b.mip_count) &&
           spans_overlap(a.layer_base, a.layer_count, b.layer_base, b.layer_count);
}

// A read-only use and a writable use that Dawn rejects in one scope.
inline bool uses_conflict(const ResUse& a, const ResUse& b) {
    return use_writable(a.kind) != use_writable(b.kind) && uses_touch(a, b);
}

// A bind group as bound at a draw / dispatch: its group index, uses and dynamic offsets.
struct BoundUses {
    uint32_t group = 0;
    const BgUses* uses = nullptr;
    const std::vector<uint32_t>* dyn = nullptr;
};

// A read-only use bound from a snapshot.
struct SnapshotUse {
    uint32_t group = 0;     // index into the BoundUses list
    uint32_t use = 0;       // index into that group's uses
    bool volatile_ = false;  // the read overlaps bytes / subresources the same draw writes: retake every draw
};

// The read-only uses of one draw / dispatch that conflict with a writable use of the same draw / dispatch.
inline std::vector<SnapshotUse> plan_snapshots(const std::vector<BoundUses>& bound) {
    std::vector<SnapshotUse> out;
    bool any_writable = false;
    for (const BoundUses& b : bound) any_writable = any_writable || (b.uses && b.uses->any_writable);
    if (!any_writable) return out;
    for (uint32_t gi = 0; gi < bound.size(); ++gi) {
        const BoundUses& rb = bound[gi];
        if (!rb.uses) continue;
        for (uint32_t ui = 0; ui < rb.uses->uses.size(); ++ui) {
            const ResUse& r = rb.uses->uses[ui];
            if (use_writable(r.kind)) continue;
            bool hit = false, vol = false;
            for (const BoundUses& wb : bound) {
                if (!wb.uses || !wb.uses->any_writable) continue;
                for (const ResUse& w : wb.uses->uses) {
                    if (!use_writable(w.kind) || !uses_conflict(r, w)) continue;
                    hit = true;
                    if (r.texture) vol = true;
                    else {
                        const uint64_t ro = use_offset(r, rb.dyn), wo = use_offset(w, wb.dyn);
                        // size 0 = to the end of the buffer
                        const uint64_t rs = r.size ? r.size : UINT64_MAX - ro, ws = w.size ? w.size : UINT64_MAX - wo;
                        if (spans_overlap(ro, rs, wo, ws)) vol = true;
                    }
                }
            }
            if (hit) out.push_back(SnapshotUse{gi, ui, vol});
        }
    }
    return out;
}

// The uses a render pass accumulated so far (all draws since the pass, or its last split, began).
class UsageScope {
public:
    void clear() {
        by_res_.clear();
        writable_ = 0;
    }
    bool empty() const { return by_res_.empty(); }
    // True when `u` cannot join the scope (a read against a write already in it, or the reverse).
    bool conflicts(const ResUse& u) const {
        if (!writable_ && !use_writable(u.kind)) return false;
        auto it = by_res_.find(u.res);
        if (it == by_res_.end()) return false;
        for (const ResUse& o : it->second)
            if (uses_conflict(u, o)) return true;
        return false;
    }
    void add(const ResUse& u) {
        std::vector<ResUse>& v = by_res_[u.res];
        for (const ResUse& o : v)  // already covered: same kind class and the same subresources
            if (use_writable(o.kind) == use_writable(u.kind) && o.texture == u.texture &&
                (!u.texture || (o.aspects == u.aspects && o.mip_base == u.mip_base && o.mip_count == u.mip_count &&
                                o.layer_base == u.layer_base && o.layer_count == u.layer_count)))
                return;
        v.push_back(u);
        if (use_writable(u.kind)) ++writable_;
    }

private:
    std::unordered_map<const void*, std::vector<ResUse>> by_res_;
    uint64_t writable_ = 0;
};

// What a draw / dispatch needs before it is encoded.
struct BindPlan {
    std::vector<SnapshotUse> snapshots;  // read-only uses to bind from a snapshot (needs a split: copies go outside passes)
    bool split = false;                  // end the pass here and continue in a new one
    bool scope_conflict = false;         // render: a use conflicts with an earlier draw of the pass
};

// render: the pass scope applies (compute: each dispatch is a scope of its own). `scope` holds the uses of
// the earlier draws of this render pass.
inline BindPlan plan_binds(const std::vector<BoundUses>& bound, const UsageScope& scope, bool render) {
    BindPlan p;
    p.snapshots = plan_snapshots(bound);
    if (!p.snapshots.empty()) p.split = true;
    if (render && !scope.empty()) {
        for (uint32_t gi = 0; gi < bound.size() && !p.scope_conflict; ++gi) {
            const BoundUses& b = bound[gi];
            if (!b.uses) continue;
            for (uint32_t ui = 0; ui < b.uses->uses.size(); ++ui) {
                bool snap = false;
                for (const SnapshotUse& s : p.snapshots) snap = snap || (s.group == gi && s.use == ui);
                if (!snap && scope.conflicts(b.uses->uses[ui])) {
                    p.scope_conflict = true;
                    break;
                }
            }
        }
        if (p.scope_conflict) p.split = true;
    }
    return p;
}

}  // namespace playgta6::gpu::native_exec
