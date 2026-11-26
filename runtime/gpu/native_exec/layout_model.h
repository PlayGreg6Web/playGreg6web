// Native executor: the worker's `layout:'auto'` bind-group remap, GPU-independent part. Ports
// web/workers/render_state_packets.js `parseWgslGroup0Bindings`, `mergeWgslGroup0Bindings`,
// `isExplicitLayoutCompatibleWithWgsl`, `synthLayoutEntryForWgslBinding` and the resource matching of
// `_getOrCreateRemappedBindGroup` (including the dynamic-offset folding). No WebGPU types, so
// tests/test_native_exec_model.cpp runs it on every host build. Design: docs/NATIVE_EXECUTOR_DESIGN.md.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gpu/native_exec/m1_model.h"

namespace playgta6::gpu::native_exec {

// ---- WGSL @group(0) binding parsing ----

enum class WgslBindingKind : uint8_t { Uniform, ReadOnlyStorage, Storage, Texture, StorageTexture, Sampler };
enum class WgslSampleType : uint8_t { Float, UnfilterableFloat, Depth, Sint, Uint };

struct WgslBinding {
    uint32_t binding = 0;
    WgslBindingKind kind = WgslBindingKind::Uniform;
    bool comparison = false;       // sampler_comparison
    uint8_t view_dim = 2;          // the layout view dimension codes: 1 1d, 2 2d, 3 2d-array, 4 cube, 5 cube-array, 6 3d
    bool multisampled = false;
    WgslSampleType sample_type = WgslSampleType::Float;  // texture (default 'float')
    std::string storage_format;    // storage-texture format name as written ("" = none)
    bool operator==(const WgslBinding& o) const {
        return binding == o.binding && kind == o.kind && comparison == o.comparison && view_dim == o.view_dim &&
               multisampled == o.multisampled && sample_type == o.sample_type && storage_format == o.storage_format;
    }
};

namespace layout_detail {

inline bool lw_is_ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }
inline bool lw_is_ident_start(char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_'; }
inline bool lw_is_ident_char(char c) { return lw_is_ident_start(c) || (c >= '0' && c <= '9'); }
inline bool lw_is_digit(char c) { return c >= '0' && c <= '9'; }

inline bool lw_starts_with(std::string_view s, size_t i, std::string_view p) { return s.substr(i, p.size()) == p; }
inline size_t lw_skip_ws(std::string_view s, size_t i) {
    while (i < s.size() && lw_is_ws(s[i])) ++i;
    return i;
}

// `\(\s*(\d+)[ui]?\s*\)` after an attribute name at `i` (pointing just past the name). `want_zero`
// restricts the number to the literal 0 (the group attribute: `0[ui]?`).
inline bool match_attr_args(std::string_view s, size_t& i, bool want_zero, uint32_t* value) {
    size_t j = lw_skip_ws(s, i);
    if (j >= s.size() || s[j] != '(') return false;
    j = lw_skip_ws(s, j + 1);
    if (j >= s.size() || !lw_is_digit(s[j])) return false;
    uint64_t v = 0;
    const size_t d0 = j;
    while (j < s.size() && lw_is_digit(s[j])) {
        v = v * 10 + static_cast<uint64_t>(s[j] - '0');
        if (v > 0xffffffffull) v = 0xffffffffull;
        ++j;
    }
    if (want_zero && !(j - d0 == 1 && s[d0] == '0')) return false;
    if (j < s.size() && (s[j] == 'u' || s[j] == 'i')) ++j;
    j = lw_skip_ws(s, j);
    if (j >= s.size() || s[j] != ')') return false;
    i = j + 1;
    if (value) *value = static_cast<uint32_t>(v);
    return true;
}

struct VarMatch {
    uint32_t binding = 0;
    std::string addr_raw, var_name, type_raw;
    size_t end = 0;
};

// WGSL_GROUP0_VAR_RE at position `i` (which holds '@'): either attribute order, then
// `var(<addr>)? name : type ;`.
inline bool match_group0_var(std::string_view s, size_t i, VarMatch* m) {
    size_t p = i;
    uint32_t binding = 0;
    auto group_attr = [&](size_t& q) {
        if (!lw_starts_with(s, q, "@group")) return false;
        q += 6;
        return match_attr_args(s, q, true, nullptr);
    };
    auto binding_attr = [&](size_t& q) {
        if (!lw_starts_with(s, q, "@binding")) return false;
        q += 8;
        return match_attr_args(s, q, false, &binding);
    };
    bool ok = false;
    {
        size_t q = p;
        if (group_attr(q)) {
            q = lw_skip_ws(s, q);
            if (binding_attr(q)) { p = q; ok = true; }
        }
    }
    if (!ok) {
        size_t q = p;
        if (binding_attr(q)) {
            q = lw_skip_ws(s, q);
            if (group_attr(q)) { p = q; ok = true; }
        }
    }
    if (!ok) return false;
    p = lw_skip_ws(s, p);
    if (!lw_starts_with(s, p, "var")) return false;
    p += 3;
    std::string addr;
    {
        size_t q = lw_skip_ws(s, p);
        if (q < s.size() && s[q] == '<') {
            size_t r = lw_skip_ws(s, q + 1);
            const size_t a0 = r;
            while (r < s.size() && s[r] != '>') ++r;
            if (r < s.size() && r > a0) {  // `([^>]+)`, then `>`
                size_t a1 = r;
                while (a1 > a0 && lw_is_ws(s[a1 - 1])) --a1;
                addr.assign(s.substr(a0, a1 - a0));
                p = r + 1;
            }
        }
    }
    const size_t ws0 = p;
    p = lw_skip_ws(s, p);
    if (p == ws0) return false;  // `\s+`
    if (p >= s.size() || !lw_is_ident_start(s[p])) return false;
    const size_t n0 = p;
    while (p < s.size() && lw_is_ident_char(s[p])) ++p;
    const std::string name(s.substr(n0, p - n0));
    p = lw_skip_ws(s, p);
    if (p >= s.size() || s[p] != ':') return false;
    p = lw_skip_ws(s, p + 1);
    const size_t t0 = p;
    while (p < s.size() && s[p] != ';') ++p;
    if (p >= s.size() || p == t0) return false;
    m->binding = binding;
    m->addr_raw = addr;
    m->var_name = name;
    m->type_raw.assign(s.substr(t0, p - t0));
    m->end = p + 1;
    return true;
}

inline std::string lw_trim(std::string s) {
    size_t a = 0, b = s.size();
    while (a < b && lw_is_ws(s[a])) ++a;
    while (b > a && lw_is_ws(s[b - 1])) --b;
    return s.substr(a, b - a);
}
inline std::string lw_lower(std::string s) {
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return s;
}

inline uint8_t classify_texture_dim(std::string_view prefix) {
    auto ends = [&](std::string_view e) { return prefix.size() >= e.size() && prefix.substr(prefix.size() - e.size()) == e; };
    if (ends("_2d_array")) return 3;
    if (ends("_cube_array")) return 5;
    if (ends("_cube")) return 4;
    if (ends("_3d")) return 6;
    if (ends("_1d")) return 1;
    return 2;  // _2d, texture_external, anything else
}

// extractWgslFunctionBodies: the text of every `fn name(...) { ... }` body, newline separated.
inline bool extract_function_bodies(std::string_view c, std::string* bodies) {
    bool found = false;
    size_t pos = 0;
    while (pos < c.size()) {
        // /\bfn\s+[A-Za-z_][A-Za-z0-9_]*\s*\(/g
        size_t f = c.find("fn", pos);
        if (f == std::string_view::npos) break;
        const bool boundary_before = f == 0 || !lw_is_ident_char(c[f - 1]);
        size_t q = f + 2;
        size_t q0 = q;
        q = lw_skip_ws(c, q);
        if (!boundary_before || q == q0 || q >= c.size() || !lw_is_ident_start(c[q])) { pos = f + 1; continue; }
        while (q < c.size() && lw_is_ident_char(c[q])) ++q;
        q = lw_skip_ws(c, q);
        if (q >= c.size() || c[q] != '(') { pos = f + 1; continue; }
        size_t i = q + 1;
        int paren = 1;
        while (i < c.size() && paren > 0) {
            if (c[i] == '(') ++paren;
            else if (c[i] == ')') --paren;
            ++i;
        }
        while (i < c.size() && c[i] != '{') {
            if (c[i] == ';') break;
            ++i;
        }
        if (i >= c.size() || c[i] != '{') { pos = q + 1; continue; }
        found = true;
        const size_t body_start = i + 1;
        int brace = 1;
        ++i;
        while (i < c.size() && brace > 0) {
            if (c[i] == '{') ++brace;
            else if (c[i] == '}') --brace;
            ++i;
        }
        const size_t body_end = std::max(body_start, i - 1);
        bodies->append(c.substr(body_start, body_end - body_start));
        *bodies += '\n';
        pos = i;
    }
    return found;
}

// /(?:^|[^A-Za-z0-9_.])NAME(?![A-Za-z0-9_])/.test(text)
inline bool references_name(std::string_view text, std::string_view name) {
    size_t p = 0;
    while (true) {
        p = text.find(name, p);
        if (p == std::string_view::npos) return false;
        const bool before_ok = p == 0 || !(lw_is_ident_char(text[p - 1]) || text[p - 1] == '.');
        const size_t e = p + name.size();
        const bool after_ok = e >= text.size() || !lw_is_ident_char(text[e]);
        if (before_ok && after_ok) return true;
        ++p;
    }
}

}  // namespace layout_detail

// parseWgslGroup0Bindings: the statically referenced @group(0) bindings of a WGSL source, sorted by
// binding and deduplicated (first declaration of a binding wins).
inline std::vector<WgslBinding> parse_wgsl_group0_bindings(std::string_view wgsl) {
    using namespace layout_detail;
    std::vector<WgslBinding> result;
    if (wgsl.empty()) return result;
    const std::string cleaned = strip_wgsl_comments(wgsl);
    if (cleaned.empty()) return result;

    std::vector<VarMatch> matches;
    for (size_t i = 0; i < cleaned.size();) {
        VarMatch m;
        if (cleaned[i] == '@' && match_group0_var(cleaned, i, &m)) {
            matches.push_back(m);
            i = m.end;
        } else {
            ++i;
        }
    }
    if (matches.empty()) return result;

    std::string usage;
    const bool has_function = extract_function_bodies(cleaned, &usage);

    std::vector<WgslBinding> by_binding;
    auto seen = [&](uint32_t b) {
        for (const WgslBinding& x : by_binding)
            if (x.binding == b) return true;
        return false;
    };
    for (const VarMatch& m : matches) {
        if (seen(m.binding)) continue;
        if (has_function && !references_name(usage, m.var_name)) continue;

        std::optional<WgslBinding> entry;
        const std::string type = m.type_raw;  // already trimmed at the start; ends may carry whitespace
        const std::string type_trim = lw_trim(type);
        if (!m.addr_raw.empty()) {
            // split(',') + lw_trim + lowercase
            std::vector<std::string> parts;
            size_t a = 0;
            while (true) {
                const size_t c = m.addr_raw.find(',', a);
                parts.push_back(lw_lower(lw_trim(m.addr_raw.substr(a, c == std::string::npos ? std::string::npos : c - a))));
                if (c == std::string::npos) break;
                a = c + 1;
            }
            WgslBinding b;
            b.binding = m.binding;
            if (parts[0] == "uniform") {
                b.kind = WgslBindingKind::Uniform;
                entry = b;
            } else if (parts[0] == "storage") {
                b.kind = parts.size() > 1 && parts[1] == "read" ? WgslBindingKind::ReadOnlyStorage : WgslBindingKind::Storage;
                entry = b;
            }
        } else if (type_trim.rfind("sampler_comparison", 0) == 0 &&
                   (type_trim.size() == 18 || !lw_is_ident_char(type_trim[18]))) {
            WgslBinding b;
            b.binding = m.binding;
            b.kind = WgslBindingKind::Sampler;
            b.comparison = true;
            entry = b;
        } else if (type_trim.rfind("sampler", 0) == 0 && (type_trim.size() == 7 || !lw_is_ident_char(type_trim[7]))) {
            WgslBinding b;
            b.binding = m.binding;
            b.kind = WgslBindingKind::Sampler;
            entry = b;
        } else if (type_trim.rfind("texture_storage_", 0) == 0) {
            size_t e = 0;
            while (e < type_trim.size() && lw_is_ident_char(type_trim[e])) ++e;
            const std::string prefix = type_trim.substr(0, e);
            WgslBinding b;
            b.binding = m.binding;
            b.kind = WgslBindingKind::StorageTexture;
            b.view_dim = classify_texture_dim(prefix);
            // /^texture_storage_[A-Za-z0-9_]+\s*<\s*([A-Za-z0-9_-]+)/
            size_t q = lw_skip_ws(type_trim, e);
            if (q < type_trim.size() && type_trim[q] == '<') {
                q = lw_skip_ws(type_trim, q + 1);
                size_t f = q;
                while (f < type_trim.size() && (lw_is_ident_char(type_trim[f]) || type_trim[f] == '-')) ++f;
                if (f > q) b.storage_format = type_trim.substr(q, f - q);
            }
            entry = b;
        } else if (type_trim.rfind("texture_", 0) == 0) {
            size_t e = 0;
            while (e < type_trim.size() && lw_is_ident_char(type_trim[e])) ++e;
            const std::string prefix = type_trim.substr(0, e);
            WgslBinding b;
            b.binding = m.binding;
            b.kind = WgslBindingKind::Texture;
            b.view_dim = classify_texture_dim(prefix);
            if (prefix.find("_multisampled_") != std::string::npos) b.multisampled = true;
            if (prefix.rfind("texture_depth_", 0) == 0) {
                b.sample_type = WgslSampleType::Depth;
            } else {
                // /<\s*(u32|i32|f32)\s*>/ anywhere in the type
                for (size_t k = 0; k < type_trim.size(); ++k) {
                    if (type_trim[k] != '<') continue;
                    size_t q = lw_skip_ws(type_trim, k + 1);
                    const std::string_view rest = std::string_view(type_trim).substr(q);
                    for (const char* t : {"u32", "i32", "f32"}) {
                        if (rest.substr(0, 3) == t) {
                            const size_t r = lw_skip_ws(type_trim, q + 3);
                            if (r < type_trim.size() && type_trim[r] == '>') {
                                if (t[0] == 'u') b.sample_type = WgslSampleType::Uint;
                                else if (t[0] == 'i') b.sample_type = WgslSampleType::Sint;
                                k = type_trim.size();
                                break;
                            }
                        }
                    }
                }
            }
            entry = b;
        }
        if (entry) by_binding.push_back(*entry);
    }
    result = std::move(by_binding);
    std::sort(result.begin(), result.end(), [](const WgslBinding& a, const WgslBinding& b) { return a.binding < b.binding; });
    return result;
}

// mergeWgslGroup0Bindings: vertex then fragment; a fragment `storage` upgrades a vertex `read-only-storage`.
inline std::vector<WgslBinding> merge_wgsl_group0_bindings(const std::vector<WgslBinding>& vs,
                                                           const std::vector<WgslBinding>& fs) {
    std::vector<WgslBinding> out;
    auto find = [&](uint32_t b) -> WgslBinding* {
        for (WgslBinding& x : out)
            if (x.binding == b) return &x;
        return nullptr;
    };
    for (const WgslBinding& b : vs) {
        if (WgslBinding* e = find(b.binding)) *e = b;
        else out.push_back(b);
    }
    for (const WgslBinding& fb : fs) {
        WgslBinding* e = find(fb.binding);
        if (!e) out.push_back(fb);
        else if (e->kind == WgslBindingKind::ReadOnlyStorage && fb.kind == WgslBindingKind::Storage) *e = fb;
    }
    std::sort(out.begin(), out.end(), [](const WgslBinding& a, const WgslBinding& b) { return a.binding < b.binding; });
    return out;
}

// The part of an explicit @group(0) layout entry the compatibility check reads.
struct Group0LayoutEntry {
    uint32_t binding = 0;
    uint8_t kind = 0;     // 0 buffer, 1 sampler, 2 texture, 3 storage texture (LayoutEntry::kind)
    uint8_t subtype = 0;  // buffer: 0 uniform, 1 storage, 2 read-only-storage
};

// isExplicitLayoutCompatibleWithWgsl. `entries == nullptr` (manifest layout, no group 0) is compatible.
inline bool explicit_layout_compatible(const std::vector<Group0LayoutEntry>* entries,
                                       const std::vector<WgslBinding>& wgsl) {
    if (wgsl.empty() || !entries) return true;
    for (const WgslBinding& b : wgsl) {
        const Group0LayoutEntry* e = nullptr;
        for (const Group0LayoutEntry& x : *entries)
            if (x.binding == b.binding) { e = &x; break; }
        if (!e) return false;
        switch (b.kind) {
            case WgslBindingKind::Uniform:
                if (!(e->kind == 0 && e->subtype == 0)) return false;
                break;
            case WgslBindingKind::ReadOnlyStorage:
                if (!(e->kind == 0 && (e->subtype == 2 || e->subtype == 1))) return false;
                break;
            case WgslBindingKind::Storage:
                if (!(e->kind == 0 && e->subtype == 1)) return false;
                break;
            case WgslBindingKind::Texture:
                if (e->kind != 2) return false;
                break;
            case WgslBindingKind::StorageTexture:
                if (e->kind != 3) return false;
                break;
            case WgslBindingKind::Sampler:
                if (e->kind != 1) return false;
                break;
        }
    }
    return true;
}

// ---- resource matching of _getOrCreateRemappedBindGroup ----

constexpr uint32_t kRemapBufferUsageUniform = 0x0040;
constexpr uint32_t kRemapBufferUsageStorage = 0x0080;
constexpr uint32_t kRemapTextureUsageBinding = 0x0004;
constexpr uint32_t kRemapTextureUsageStorage = 0x0008;

// One entry of a recorded bind group descriptor (`bindGroupDescriptors.get(id).resources`).
struct RemapResource {
    uint32_t binding = 0;
    WgslBindingKind kind = WgslBindingKind::Uniform;  // the descriptor kind: uniform / storage / read-only-storage /
                                                      // texture / storage-texture / sampler
    bool is_dummy = false;
    std::optional<uint32_t> buffer_usage;
    std::optional<uint32_t> texture_usage;
    std::optional<uint8_t> format;       // texture format code of the view
    std::optional<uint8_t> view_dim;     // layout view dimension code
    std::optional<bool> multisampled;
    std::optional<bool> comparison;
    uint64_t buffer_offset = 0, buffer_size = 0;  // buffer entry: the bound range (size 0 = whole)
    uint64_t buffer_total = 0;                    // the buffer's size (0 = unknown)
    bool has_dynamic_offset = false;
    uint32_t dynamic_offset_index = 0;
};

// resolveTextureSampleType for a format code (`float32_filterable`: the device feature).
inline WgslSampleType texture_sample_type(uint8_t format, bool float32_filterable) {
    if (!format_valid(format)) return WgslSampleType::Float;
    if (format_is_depth(format)) return WgslSampleType::Depth;
    const std::string_view n = format_info(format).name;
    if (format == 44 || (n.size() >= 4 && n.substr(n.size() - 4) == "uint")) return WgslSampleType::Uint;
    if (n.size() >= 4 && n.substr(n.size() - 4) == "sint") return WgslSampleType::Sint;
    if (format_is_float32(format)) return float32_filterable ? WgslSampleType::Float : WgslSampleType::UnfilterableFloat;
    return WgslSampleType::Float;
}

// A depth-format view bound to a layout entry whose sampleType cannot take it (float/sint/uint), or
// a non-depth view bound to a 'depth' entry: Dawn rejects the bind group. Mirrors the worker's
// other sample-type fallbacks (substitute the entry's dummy resource) instead of a validation error.
// layout_sample_type: the entry's code (0 float, 1 unfilterable-float, 2 depth, 3 sint, 4 uint).
inline bool depth_sample_type_mismatch(uint8_t layout_sample_type, uint8_t format) {
    const bool is_depth = format_valid(format) && format_is_depth(format);
    if (layout_sample_type == 2) return !is_depth;
    return is_depth && layout_sample_type != 1;
}

// How a depth-format texture is read through a layout entry (`pick_depth_read`).
//   Direct         the depth view binds as is (depth entry, or unfilterable-float entry)
//   StencilDirect  a stencil-aspect view binds as is to a uint entry (WebGPU: the stencil aspect's
//                  sample type is uint, so no copy is needed; D3D12 reads the stencil plane the same way)
//   Twin*          a color copy of the depth plane (see NativeExecutor::depth_twin)
//   Dummy          no real read exists; the entry's dummy is bound (counted)
enum class DepthRead : uint8_t {
    Dummy = 0,
    Direct,
    StencilDirect,
    TwinF32,  // r32float: float entry on a float32-filterable device
    TwinF16,  // r16float: float entry without float32-filterable (the only filterable float format; the
              // depth value is rounded to half precision, 11 bit mantissa)
    TwinU32,  // r32uint: uint entry
    TwinS32,  // r32sint: sint entry
};
constexpr size_t kDepthReadCount = 7;

// layout_sample_type: the entry's code (0 float, 1 unfilterable-float, 2 depth, 3 sint, 4 uint).
// aspect: the view's GpuTextureAspect (0 all, 1 depth, 2 stencil). A float entry that is multisampled
// is a WebGPU-invalid combination (Dawn rejects the layout); the executor creates such an entry as
// unfilterable-float, so it is treated as code 1 here.
inline DepthRead pick_depth_read(uint8_t layout_sample_type, bool layout_multisampled, uint8_t format,
                                 uint8_t aspect, uint32_t sample_count, uint8_t dimension,
                                 bool float32_filterable) {
    if (!format_valid(format) || !format_is_depth(format)) return DepthRead::Dummy;
    const bool has_stencil = format == 47 || format == 49;
    if (layout_sample_type == 0 && layout_multisampled) layout_sample_type = 1;
    // Stencil-aspect view (or an all-aspect view of a stencil-bearing format on a uint entry: D3D12
    // X24_TYPELESS_G8_UINT): only a uint entry takes the stencil plane, directly.
    const bool stencil_plane = aspect == 2 || (aspect == 0 && has_stencil && layout_sample_type == 4);
    if (stencil_plane) return has_stencil && layout_sample_type == 4 ? DepthRead::StencilDirect : DepthRead::Dummy;
    if (layout_sample_type == 4)  // depth plane through an integer view: raw stored bits
        return sample_count == 1 && dimension == 1 ? DepthRead::TwinU32 : DepthRead::Dummy;
    switch (layout_sample_type) {
        case 1:
        case 2: return DepthRead::Direct;
        case 0:
            if (sample_count != 1 || dimension != 1) return DepthRead::Dummy;
            return float32_filterable ? DepthRead::TwinF32 : DepthRead::TwinF16;
        default: break;  // 3: sint
    }
    // sint: the depth plane's raw value through an integer view (not defined for multisampled r32 targets)
    if (layout_sample_type == 3 && sample_count == 1 && dimension == 1) return DepthRead::TwinS32;
    return DepthRead::Dummy;
}

// The twin copy shader: one fullscreen triangle, @fragment fn fs writes the twin texel from the depth
// texel at the same coordinate (textureLoad, mip 0 of the single-mip/layer source view).
//   float kinds   the depth value (f16: rounded to half precision by the r16float target)
//   int kinds     the raw stored value, as D3D12 reads a depth resource through an R32_UINT / R32_SINT
//                 view of its typeless family: depth32float -> the f32 bit pattern; depth16unorm ->
//                 round(d * 65535); depth24plus(-stencil8) -> round(d * 16777215) (24-bit unorm).
//                 r32sint is the same 32 bits reinterpreted as i32 (a bit-cast, never a value conversion).
inline std::string depth_twin_wgsl(DepthRead kind, uint8_t src_format) {
    std::string raw;
    if (src_format == 45) raw = "u32(round(d * 65535.0))";
    else if (src_format == 46 || src_format == 47) raw = "u32(round(d * 16777215.0))";
    else raw = "bitcast<u32>(d)";
    std::string out_ty = "f32", expr = "d";
    if (kind == DepthRead::TwinU32) { out_ty = "u32"; expr = raw; }
    else if (kind == DepthRead::TwinS32) { out_ty = "i32"; expr = "bitcast<i32>(" + raw + ")"; }
    return std::string("@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {\n"
                       "  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));\n"
                       "  return vec4f(p[i], 0.0, 1.0);\n}\n"
                       "@group(0) @binding(0) var src: texture_depth_2d;\n"
                       "@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) ") +
           out_ty + " {\n  let d = textureLoad(src, vec2i(pos.xy), 0);\n  return " + expr + ";\n}\n";
}

struct PickResult {
    enum class Kind : uint8_t { Resource, FallbackSampler, Dummy } kind = Kind::Dummy;
    size_t index = 0;          // Resource: index into the available list
    uint64_t offset = 0;       // buffer: bound offset (dynamic offset folded in)
    uint64_t size = 0;         // buffer: bound size, 0 = whole (WebGPU "undefined")
    bool folded = false;       // a dynamic offset was folded in
    bool dyn_clamped = false;  // the size was clamped to the buffer remainder
    bool dyn_oob = false;      // the folded offset was out of bounds: dummy buffer
};

// pickResourceForBinding with its `usedIndices`. One picker per bind group build; call pick() for
// each WGSL binding in order.
class RemapPicker {
public:
    RemapPicker(const std::vector<RemapResource>& available, const std::vector<uint32_t>& dynamic_offsets,
                bool float32_filterable)
        : av_(available), dyn_(dynamic_offsets), f32f_(float32_filterable), used_(available.size(), false) {}

    PickResult pick(const WgslBinding& b) {
        for (int pass = 0; pass < 2; ++pass) {
            for (size_t i = 0; i < av_.size(); ++i) {
                if (used_[i]) continue;
                const RemapResource& r = av_[i];
                if (r.is_dummy) continue;
                const bool r_buf = r.kind == WgslBindingKind::Uniform || r.kind == WgslBindingKind::Storage ||
                                   r.kind == WgslBindingKind::ReadOnlyStorage;
                if (b.kind == WgslBindingKind::Uniform) {
                    if (!r_buf) continue;
                    if (pass == 0 && r.kind != WgslBindingKind::Uniform) continue;
                    if (r.buffer_usage) {
                        if ((*r.buffer_usage & kRemapBufferUsageUniform) == 0) continue;
                    } else if (r.kind != WgslBindingKind::Uniform) {
                        continue;
                    }
                    used_[i] = true;
                    return buffer_result(i, r);
                }
                if (b.kind == WgslBindingKind::ReadOnlyStorage || b.kind == WgslBindingKind::Storage) {
                    if (!r_buf) continue;
                    if (pass == 0 && !(r.kind == b.kind || (b.kind == WgslBindingKind::ReadOnlyStorage &&
                                                           r.kind == WgslBindingKind::Storage)))
                        continue;
                    if (r.buffer_usage) {
                        if ((*r.buffer_usage & kRemapBufferUsageStorage) == 0) continue;
                    } else if (!(r.kind == WgslBindingKind::Storage ||
                                 (b.kind == WgslBindingKind::ReadOnlyStorage && r.kind == WgslBindingKind::ReadOnlyStorage))) {
                        continue;
                    }
                    used_[i] = true;
                    return buffer_result(i, r);
                }
                if (pass > 0) continue;
                if (b.kind == WgslBindingKind::Texture) {
                    if (r.kind != WgslBindingKind::Texture) continue;
                    if (r.texture_usage && (*r.texture_usage & kRemapTextureUsageBinding) == 0) continue;
                    if (r.view_dim && *r.view_dim != b.view_dim) continue;
                    if (r.multisampled && *r.multisampled != b.multisampled) continue;
                    if (r.format) {
                        const WgslSampleType actual = texture_sample_type(*r.format, f32f_);
                        if (b.sample_type == WgslSampleType::UnfilterableFloat) {
                            if (actual != WgslSampleType::Float && actual != WgslSampleType::UnfilterableFloat) continue;
                        } else if (actual != b.sample_type) {
                            continue;
                        }
                    }
                    used_[i] = true;
                    return PickResult{PickResult::Kind::Resource, i};
                }
                if (b.kind == WgslBindingKind::StorageTexture) {
                    if (r.kind != WgslBindingKind::StorageTexture) continue;
                    if (r.texture_usage && (*r.texture_usage & kRemapTextureUsageStorage) == 0) continue;
                    if (r.view_dim && *r.view_dim != b.view_dim) continue;
                    if (r.format && !b.storage_format.empty() && format_info(*r.format).name != b.storage_format) continue;
                    used_[i] = true;
                    return PickResult{PickResult::Kind::Resource, i};
                }
                if (b.kind == WgslBindingKind::Sampler) {
                    if (r.kind != WgslBindingKind::Sampler) continue;
                    if (r.comparison && *r.comparison != b.comparison) continue;
                    used_[i] = true;
                    return PickResult{PickResult::Kind::Resource, i};
                }
            }
        }
        if (b.kind == WgslBindingKind::Sampler) return PickResult{PickResult::Kind::FallbackSampler};
        return PickResult{PickResult::Kind::Dummy};
    }

private:
    PickResult buffer_result(size_t i, const RemapResource& r) {
        PickResult out{PickResult::Kind::Resource, i};
        out.offset = r.buffer_offset;
        out.size = r.buffer_size;
        const uint64_t dyn = dynamic_offset_for(r);
        if (dyn == 0) return out;
        const uint64_t eff = r.buffer_offset + dyn;
        uint64_t size = r.buffer_size;
        const uint64_t total = r.buffer_total;
        if (total > 0) {
            if (eff >= total) {
                PickResult d{PickResult::Kind::Dummy};
                d.dyn_oob = true;
                return d;
            }
            const uint64_t remaining = total - eff;
            if (size > 0 && size > remaining) {
                size = remaining;
                out.dyn_clamped = true;
            }
        }
        out.folded = true;
        out.offset = eff;
        out.size = size;
        return out;
    }

    uint64_t dynamic_offset_for(const RemapResource& r) const {
        return r.has_dynamic_offset && r.dynamic_offset_index < dyn_.size() ? dyn_[r.dynamic_offset_index] : 0;
    }

    const std::vector<RemapResource>& av_;
    const std::vector<uint32_t>& dyn_;
    bool f32f_;
    std::vector<bool> used_;
};

// synthLayoutEntryForWgslBinding -> the executor's LayoutEntry fields (kind / subtype / format / dim)
// so the existing dummy resource builder can serve it. Buffers use minBindingSize 4096.
struct SynthLayoutEntry {
    uint32_t binding = 0;
    uint8_t kind = 0, subtype = 0, view_dim = 2, storage_format = 0;  // storage_format: code (0 = none)
    bool multisampled = false;
    uint64_t min_binding_size = 0;
};

inline uint8_t format_code_by_name(std::string_view name) {
    for (uint8_t c = 1; c < kFormatCodeCount; ++c)
        if (format_info(c).name == name) return c;
    return 0;
}

inline SynthLayoutEntry synth_layout_entry(const WgslBinding& b) {
    SynthLayoutEntry e;
    e.binding = b.binding;
    switch (b.kind) {
        case WgslBindingKind::Uniform: e.kind = 0; e.subtype = 0; e.min_binding_size = 4096; break;
        case WgslBindingKind::Storage: e.kind = 0; e.subtype = 1; e.min_binding_size = 4096; break;
        case WgslBindingKind::ReadOnlyStorage: e.kind = 0; e.subtype = 2; e.min_binding_size = 4096; break;
        case WgslBindingKind::Sampler: e.kind = 1; e.subtype = b.comparison ? 2 : 0; break;
        case WgslBindingKind::StorageTexture: {
            e.kind = 3;
            e.subtype = 0;  // write-only
            const uint8_t f = b.storage_format.empty() ? 0 : format_code_by_name(b.storage_format);
            e.storage_format = f ? f : 22;  // 'rgba8unorm' default
            e.view_dim = b.view_dim;
            break;
        }
        case WgslBindingKind::Texture:
            e.kind = 2;
            e.subtype = b.sample_type == WgslSampleType::Float ? 0
                        : b.sample_type == WgslSampleType::UnfilterableFloat ? 1
                        : b.sample_type == WgslSampleType::Depth ? 2
                        : b.sample_type == WgslSampleType::Sint ? 3 : 4;
            e.view_dim = b.view_dim;
            e.multisampled = b.multisampled;
            break;
    }
    return e;
}

}  // namespace playgta6::gpu::native_exec
