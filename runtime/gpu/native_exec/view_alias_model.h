// Two binding-validity rules Dawn enforces on a bind group / dispatch, as pure models (no GPU):
//  1. A texture view bound to a layout entry must have the entry's viewDimension (2d / 2d-array / cube /
//     cube-array); the title's view of a cube-array BC texture meets 2d entries.
//  2. A writable storage buffer binding must not overlap any other binding of the same buffer in one
//     dispatch (same or different bind group); the read-only side is bound through a copy instead.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace playgta6::gpu::native_exec {

// View dimension codes: 0 default, 1 1d, 2 2d, 3 2d-array, 4 cube, 5 cube-array, 6 3d.
enum class ViewFit : uint8_t {
    Same,       // the view already has the entry's dimension (or the entry does not constrain it)
    Coerced,    // dim / layer range rewritten to the entry's dimension
    Unfixable,  // the texture cannot give that dimension (a cube entry over fewer than 6 layers, ...)
};

struct EntryViewFit {
    ViewFit fit = ViewFit::Same;
    uint8_t dim = 0;
    uint32_t base_layer = 0;
    uint32_t layer_count = 0;  // 0 = all remaining
};

// tex_dim: 0 1d, 1 2d, 2 3d. tex_layers: array layers of a 2d texture. view_dim / base_layer / layer_count:
// the title's view (view_dim 0 = the texture's default: 2d-array when it has layers). entry_dim: the layout's.
inline EntryViewFit fit_view_to_entry(uint8_t tex_dim, uint32_t tex_layers, uint8_t view_dim, uint32_t base_layer,
                                      uint32_t layer_count, uint8_t entry_dim) {
    EntryViewFit r;
    r.dim = view_dim;
    r.base_layer = base_layer;
    r.layer_count = layer_count;
    if (entry_dim < 2 || entry_dim > 5 || tex_dim != 1) return r;  // 3d / 1d entries and textures: untouched
    const uint8_t eff = view_dim ? view_dim : (tex_layers > 1 ? 3 : 2);
    if (eff == entry_dim) return r;
    const uint32_t avail = tex_layers > base_layer ? tex_layers - base_layer : 0;
    const uint32_t want = layer_count ? std::min(layer_count, avail) : avail;
    r.fit = ViewFit::Coerced;
    r.dim = entry_dim;
    switch (entry_dim) {
        case 2:
            if (avail < 1) { r.fit = ViewFit::Unfixable; return r; }
            r.layer_count = 1;
            break;
        case 3:
            if (avail < 1) { r.fit = ViewFit::Unfixable; return r; }
            r.layer_count = want;
            break;
        case 4:
            if (avail < 6) { r.fit = ViewFit::Unfixable; return r; }
            r.layer_count = 6;
            break;
        case 5:
            if (want < 6) { r.fit = ViewFit::Unfixable; return r; }
            r.layer_count = want / 6 * 6;
            break;
    }
    return r;
}

// One buffer binding of a dispatch (offset already includes any dynamic offset; size is explicit).
struct AliasBinding {
    uint64_t buffer = 0;  // identity of the buffer
    uint32_t group = 0, binding = 0;
    uint64_t offset = 0, size = 0;
    bool writable = false;
};

struct AliasPlan {
    std::vector<size_t> copy;  // indexes of the read-only bindings to bind through a copy
    std::vector<size_t> twin;  // indexes of writable bindings bound through a scratch copy that is merged back
    uint32_t unfixable = 0;    // overlapping pairs the plan cannot separate
};

inline bool ranges_overlap(const AliasBinding& a, const AliasBinding& b) {
    return a.offset < b.offset + b.size && b.offset < a.offset + a.size;
}

// Dawn rejects a dispatch whose writable storage binding overlaps any other binding of the same buffer.
//  - a read-only binding overlapping a writable one is bound through a copy of its range;
//  - of several overlapping writable bindings the first stays on the buffer and every later one that overlaps
//    a binding staying on the buffer is bound through a scratch copy whose changed words are merged back
//    after the dispatch (`twin`).
// A small writable range overlapping (or sharing a buffer with) a much larger read-only one: the writer is the
// cheap side to run on a scratch copy (a few bytes in, the changed words back) instead of copying the reader.
inline bool prefer_writer_twin(uint64_t ro_size, uint64_t w_size) {
    return ro_size >= 65536 && w_size != 0 && w_size * 8 <= ro_size;
}

inline AliasPlan plan_alias_copies(const std::vector<AliasBinding>& v) {
    AliasPlan p;
    std::vector<bool> forced(v.size(), false);
    for (size_t i = 0; i < v.size(); ++i) {
        for (size_t j = i + 1; j < v.size(); ++j) {
            const AliasBinding &a = v[i], &b = v[j];
            if (a.buffer != b.buffer || (!a.writable && !b.writable) || !ranges_overlap(a, b)) continue;
            if (a.writable && b.writable) continue;
            const size_t ro = a.writable ? j : i, w = a.writable ? i : j;
            if (prefer_writer_twin(v[ro].size, v[w].size)) {
                forced[w] = true;
                continue;
            }
            if (std::find(p.copy.begin(), p.copy.end(), ro) == p.copy.end()) p.copy.push_back(ro);
        }
    }
    std::vector<size_t> direct;
    for (size_t j = 0; j < v.size(); ++j) {
        if (!v[j].writable) continue;
        bool hit = forced[j];
        for (size_t d : direct)
            if (v[d].buffer == v[j].buffer && ranges_overlap(v[d], v[j])) hit = true;
        if (hit) p.twin.push_back(j);
        else direct.push_back(j);
    }
    std::sort(p.copy.begin(), p.copy.end());
    return p;
}

// One texture binding of a dispatch (a subresource range of a texture, used as sampled / read-only storage
// or as a writable storage texture).
struct AliasTexBinding {
    uint64_t texture = 0;
    uint32_t group = 0, binding = 0;
    uint32_t base_mip = 0, mip_count = 1, base_layer = 0, layer_count = 1;
    bool writable = false;
};

struct TexAliasPlan {
    std::vector<size_t> copy;  // indexes of the non-writable bindings to bind through a twin texture
    uint32_t unfixable = 0;    // overlapping writable pairs
};

inline bool tex_ranges_overlap(const AliasTexBinding& a, const AliasTexBinding& b) {
    return a.texture == b.texture && a.base_mip < b.base_mip + b.mip_count && b.base_mip < a.base_mip + a.mip_count &&
           a.base_layer < b.base_layer + b.layer_count && b.base_layer < a.base_layer + a.layer_count;
}

inline TexAliasPlan plan_texture_alias(const std::vector<AliasTexBinding>& v) {
    TexAliasPlan p;
    for (size_t i = 0; i < v.size(); ++i) {
        for (size_t j = i + 1; j < v.size(); ++j) {
            const AliasTexBinding &a = v[i], &b = v[j];
            if ((!a.writable && !b.writable) || !tex_ranges_overlap(a, b)) continue;
            if (a.writable && b.writable) {
                ++p.unfixable;
                continue;
            }
            const size_t ro = a.writable ? j : i;
            if (std::find(p.copy.begin(), p.copy.end(), ro) == p.copy.end()) p.copy.push_back(ro);
        }
    }
    std::sort(p.copy.begin(), p.copy.end());
    return p;
}

}  // namespace playgta6::gpu::native_exec
