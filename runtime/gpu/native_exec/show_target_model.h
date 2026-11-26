// Native executor: "present any target" diagnostic model (--show-target). Pure logic, no GPU: the
// selector grammar, which render target a selector resolves to in a frame, hotkey cycling, which
// textures can be displayed, the display shader (and its CPU twin) and the title text. The executor
// blits the chosen texture to the window / offscreen canvas instead of the swapchain image; the guest's
// frame is not touched (read-only). Docs: docs/NATIVE_EXECUTOR_DESIGN.md "Present any target".
//
// Selector (the --show-target argument):
//   <id> | tex:<id>        the texture with this stream id, shown as colour
//   depth:<id>             a depth texture, shown through its r32float depth twin (grey from the depth value)
//   pass:<N>               the colour target of the N-th render pass of the presented frame (0-based, in
//                          encode order; the first colour attachment that is a texture, else its depth)
//   pass:<N>:depth         the depth target of that pass
//   last                   the colour target of the last pass that rendered to a texture
//   cycle                  nothing yet: the hotkeys pick (Ctrl+F10 next / Ctrl+F9 previous, Ctrl+F8 layer, Ctrl+F7 mip)
// Suffixes, any order: @layer=L  @mip=M  @range=LO:HI (value -> 0..1 as (v-LO)/(HI-LO), default 0:1)
// A selector that names a pass without an explicit @layer/@mip uses the attachment's own layer and mip.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include "gpu/native_exec/m1_model.h"
#include "os/window_hotkeys.h"

namespace playgta6::gpu::native_exec {

struct ShowTargetSpec {
    enum class Kind : uint8_t { Cycle, Tex, Depth, Pass, PassDepth, Last };
    Kind kind = Kind::Cycle;
    uint32_t id = 0;    // Tex / Depth
    uint32_t pass = 0;  // Pass / PassDepth
    int64_t layer = -1;  // -1 = the attachment's own (0 for a plain texture)
    int64_t mip = -1;
    float lo = 0.0f, hi = 1.0f;
};

inline bool parse_u32_strict(const std::string& s, uint32_t* out) {
    if (s.empty() || s.size() > 10) return false;
    uint64_t v = 0;
    for (char c : s) {
        if (c < '0' || c > '9') return false;
        v = v * 10 + static_cast<uint64_t>(c - '0');
    }
    if (v > 0xFFFFFFFFull) return false;
    *out = static_cast<uint32_t>(v);
    return true;
}

inline bool parse_float_strict(const std::string& s, float* out) {
    if (s.empty()) return false;
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (!end || *end != 0 || !std::isfinite(v)) return false;
    *out = static_cast<float>(v);
    return true;
}

inline bool parse_show_target(const std::string& text, ShowTargetSpec* out, std::string* error) {
    auto fail = [&](const char* msg) {
        if (error) *error = std::string("--show-target ") + msg + " (selector '" + text + "')";
        return false;
    };
    ShowTargetSpec s;
    size_t at = text.find('@');
    const std::string base = text.substr(0, at);
    // modifiers
    while (at != std::string::npos) {
        const size_t next = text.find('@', at + 1);
        const std::string mod = text.substr(at + 1, next == std::string::npos ? std::string::npos : next - at - 1);
        const size_t eq = mod.find('=');
        if (eq == std::string::npos) return fail("modifier needs name=value");
        const std::string name = mod.substr(0, eq), val = mod.substr(eq + 1);
        uint32_t u = 0;
        if (name == "layer" || name == "mip") {
            if (!parse_u32_strict(val, &u)) return fail("layer/mip must be a non-negative integer");
            (name == "layer" ? s.layer : s.mip) = u;
        } else if (name == "range") {
            const size_t colon = val.find(':');
            if (colon == std::string::npos || !parse_float_strict(val.substr(0, colon), &s.lo) ||
                !parse_float_strict(val.substr(colon + 1), &s.hi) || s.hi == s.lo)
                return fail("range must be LO:HI with HI != LO");
        } else {
            return fail("unknown modifier (layer, mip, range)");
        }
        at = next;
    }
    uint32_t n = 0;
    if (base == "cycle") {
        s.kind = ShowTargetSpec::Kind::Cycle;
    } else if (base == "last") {
        s.kind = ShowTargetSpec::Kind::Last;
    } else if (base.rfind("tex:", 0) == 0 || (!base.empty() && base[0] >= '0' && base[0] <= '9')) {
        if (!parse_u32_strict(base.rfind("tex:", 0) == 0 ? base.substr(4) : base, &n) || n == 0)
            return fail("texture id must be a positive integer");
        s.kind = ShowTargetSpec::Kind::Tex;
        s.id = n;
    } else if (base.rfind("depth:", 0) == 0) {
        if (!parse_u32_strict(base.substr(6), &n) || n == 0) return fail("texture id must be a positive integer");
        s.kind = ShowTargetSpec::Kind::Depth;
        s.id = n;
    } else if (base.rfind("pass:", 0) == 0) {
        std::string rest = base.substr(5);
        s.kind = ShowTargetSpec::Kind::Pass;
        const size_t colon = rest.find(':');
        if (colon != std::string::npos) {
            if (rest.substr(colon + 1) != "depth") return fail("pass:N may only be followed by :depth");
            s.kind = ShowTargetSpec::Kind::PassDepth;
            rest = rest.substr(0, colon);
        }
        if (!parse_u32_strict(rest, &n)) return fail("pass index must be a non-negative integer");
        s.pass = n;
    } else {
        return fail("unknown selector (expected an id, tex:ID, depth:ID, pass:N[:depth], last or cycle)");
    }
    *out = s;
    return true;
}

// One render pass of the frame as the executor recorded it.
struct PassTarget {
    uint32_t pass = 0;
    uint32_t color_id = 0;  // first colour attachment that is a texture (not the swapchain); 0 = none
    uint32_t color_mip = 0, color_layer = 0;
    uint32_t depth_id = 0;  // depth-stencil attachment, 0 = none
    uint32_t depth_mip = 0, depth_layer = 0;
};

struct TargetRef {
    uint32_t id = 0;
    bool depth = false;
    // Defaults for layer / mip when the selector does not name one (a pass attachment's own subresource).
    uint32_t mip = 0, layer = 0;
    bool operator==(const TargetRef& o) const { return id == o.id && depth == o.depth; }
};

struct TargetResolution {
    std::optional<TargetRef> target;
    std::string reason;  // set when target is empty
};

inline TargetResolution resolve_show_target(const ShowTargetSpec& s, const std::vector<PassTarget>& passes) {
    TargetResolution r;
    switch (s.kind) {
        case ShowTargetSpec::Kind::Cycle:
            r.reason = "cycle: nothing chosen yet (Ctrl+F10 / Ctrl+F9)";
            break;
        case ShowTargetSpec::Kind::Tex:
            r.target = TargetRef{s.id, false, 0, 0};
            break;
        case ShowTargetSpec::Kind::Depth:
            r.target = TargetRef{s.id, true, 0, 0};
            break;
        case ShowTargetSpec::Kind::Pass:
        case ShowTargetSpec::Kind::PassDepth: {
            const PassTarget* p = nullptr;
            for (const PassTarget& c : passes)
                if (c.pass == s.pass) p = &c;
            if (!p) {
                r.reason = "pass " + std::to_string(s.pass) + " does not exist (the frame has " +
                           std::to_string(passes.size()) + " passes)";
            } else if (s.kind == ShowTargetSpec::Kind::PassDepth) {
                if (p->depth_id) r.target = TargetRef{p->depth_id, true, p->depth_mip, p->depth_layer};
                else r.reason = "pass " + std::to_string(s.pass) + " has no depth attachment";
            } else if (p->color_id) {
                r.target = TargetRef{p->color_id, false, p->color_mip, p->color_layer};
            } else if (p->depth_id) {
                r.target = TargetRef{p->depth_id, true, p->depth_mip, p->depth_layer};
            } else {
                r.reason = "pass " + std::to_string(s.pass) + " renders only to the swapchain";
            }
            break;
        }
        case ShowTargetSpec::Kind::Last: {
            for (auto it = passes.rbegin(); it != passes.rend(); ++it)
                if (it->color_id) {
                    r.target = TargetRef{it->color_id, false, it->color_mip, it->color_layer};
                    return r;
                }
            r.reason = "no pass of the frame rendered to a texture";
            break;
        }
    }
    return r;
}

// The hotkey cycle list: every distinct (texture, colour/depth) the frame rendered to, sorted by id,
// colour before depth.
inline std::vector<TargetRef> show_target_candidates(const std::vector<PassTarget>& passes) {
    std::vector<TargetRef> v;
    auto add = [&](const TargetRef& t) {
        for (const TargetRef& e : v)
            if (e == t) return;
        v.push_back(t);
    };
    for (const PassTarget& p : passes) {
        if (p.color_id) add(TargetRef{p.color_id, false, p.color_mip, p.color_layer});
        if (p.depth_id) add(TargetRef{p.depth_id, true, p.depth_mip, p.depth_layer});
    }
    std::sort(v.begin(), v.end(), [](const TargetRef& a, const TargetRef& b) {
        return a.id != b.id ? a.id < b.id : a.depth < b.depth;
    });
    return v;
}

// Next / previous entry of the cycle: ... -> candidates -> off (nullopt = show the swapchain) -> wraps.
// From a target that is not in the list (or from off) next goes to the first larger id and previous to the
// last smaller one.
inline std::optional<TargetRef> show_target_cycle(const std::vector<TargetRef>& list,
                                                  const std::optional<TargetRef>& cur, bool forward) {
    if (list.empty()) return std::nullopt;
    auto key = [](const TargetRef& t) { return (uint64_t{t.id} << 1) | (t.depth ? 1u : 0u); };
    if (!cur) return forward ? list.front() : list.back();
    const uint64_t k = key(*cur);
    if (forward) {
        for (const TargetRef& t : list)
            if (key(t) > k) return t;
        return std::nullopt;  // past the end: back to the swapchain
    }
    for (auto it = list.rbegin(); it != list.rend(); ++it)
        if (key(*it) < k) return *it;
    return std::nullopt;
}

// target_mip / target_layer: step a mip or layer index through count entries, wrapping.
inline uint32_t show_target_step(uint32_t cur, uint32_t count) { return count ? (cur + 1) % count : 0; }

// Why a texture cannot be shown, "" when it can. usage is the WebGPU usage mask (TextureBinding = 4).
struct ShowTexture {
    uint8_t format = 0;
    uint8_t dimension = 1;  // TextureGeometry::dimension: 0 1d, 1 2d, 2 3d
    uint32_t usage = 0;
    uint32_t sample_count = 1;
    uint32_t width = 1, height = 1, layers = 1, mips = 1;
};

inline std::string show_target_unsupported(const ShowTexture& t, bool as_depth) {
    if (!format_valid(t.format)) return "unknown format";
    if (t.dimension == 2) return "3d texture";
    if ((t.usage & 0x4u) == 0) return "no TextureBinding usage (cannot be sampled)";
    if (as_depth) {
        if (!format_is_depth(t.format)) return "not a depth format";
        if (t.sample_count > 1) return "multisampled depth";
        return "";
    }
    if (format_is_depth_or_stencil(t.format)) return "depth/stencil texture (use depth:ID)";
    if (format_is_integer(t.format)) return "integer format";
    return "";
}

struct ShowSubresource {
    uint32_t mip = 0, layer = 0;
    bool clamped = false;
};

// Requested layer / mip (-1 = the target's default) clamped into the texture; clamped says it was out of range.
inline ShowSubresource show_target_subresource(const ShowTargetSpec& s, const TargetRef& t, const ShowTexture& tex) {
    ShowSubresource r;
    const int64_t want_mip = s.mip >= 0 ? s.mip : t.mip;
    const int64_t want_layer = s.layer >= 0 ? s.layer : t.layer;
    const int64_t mip_max = std::max<int64_t>(0, int64_t{tex.mips} - 1);
    const int64_t layer_max = std::max<int64_t>(0, int64_t{tex.layers} - 1);
    r.mip = static_cast<uint32_t>(std::min(want_mip, mip_max));
    r.layer = static_cast<uint32_t>(std::min(want_layer, layer_max));
    r.clamped = want_mip > mip_max || want_layer > layer_max;
    return r;
}

inline uint32_t show_target_mip_extent(uint32_t base, uint32_t mip) {
    return std::max<uint32_t>(1, mip >= 32 ? 1 : base >> mip);
}

// The display transform, shared by the WGSL below and the CPU twin the tests compare against: a value
// maps to (v - lo) / (hi - lo), clamped to 0..1 by the unorm target. Colour shows rgb with alpha forced
// to 1 (an alpha-0 render target would otherwise vanish); a single-channel source (depth) shows grey.
inline float show_target_map(float v, float lo, float hi) { return std::min(1.0f, std::max(0.0f, (v - lo) / (hi - lo))); }

// The display shader. The fragment reads the texel under its own framebuffer position
// (textureLoad, no filtering: exact values, any float format, multisampled sample 0).
inline std::string show_target_wgsl(bool multisampled) {
    std::string s =
        "struct U {\n"
        "  rect: vec4<f32>,  // x, y, w, h of the destination rect in framebuffer pixels\n"
        "  src: vec4<f32>,   // source w, h, lo, hi\n"
        "  mode: vec4<u32>,  // x: 1 = single channel (depth), 0 = rgb\n"
        "};\n"
        "@group(0) @binding(0) var tex: ";
    s += multisampled ? "texture_multisampled_2d<f32>;\n" : "texture_2d<f32>;\n";
    s +=
        "@group(0) @binding(1) var<uniform> u: U;\n"
        "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {\n"
        "  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));\n"
        "  return vec4<f32>(p[i], 0.0, 1.0);\n"
        "}\n"
        "@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {\n"
        "  let uv = (pos.xy - u.rect.xy) / u.rect.zw;\n"
        "  let xy = vec2<i32>(clamp(floor(uv * u.src.xy), vec2<f32>(0.0), u.src.xy - vec2<f32>(1.0)));\n"
        "  let c = textureLoad(tex, xy, 0);\n"
        "  var v = c.rgb;\n"
        "  if (u.mode.x == 1u) { v = vec3<f32>(c.r); }\n"
        "  return vec4<f32>(clamp((v - vec3<f32>(u.src.z)) / (u.src.w - u.src.z), vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);\n"
        "}\n";
    return s;
}

inline std::string show_target_label(const TargetRef& t, const ShowTexture& tex, const ShowSubresource& sub,
                                     const ShowTargetSpec& s) {
    char buf[200];
    std::snprintf(buf, sizeof buf, "tex %u%s %ux%u %s%s mip %u/%u layer %u/%u%s", t.id, t.depth ? " depth" : "",
                  show_target_mip_extent(tex.width, sub.mip), show_target_mip_extent(tex.height, sub.mip),
                  format_info(tex.format).name, tex.sample_count > 1 ? " msaa" : "", sub.mip, tex.mips, sub.layer,
                  tex.layers, (s.lo != 0.0f || s.hi != 1.0f) ? " (range)" : "");
    return buf;
}

// The title bar text while a target is shown (or why it cannot be).
inline std::string show_target_title(const std::string& base, const std::string& label) {
    return label.empty() ? base : base + " | SHOWING " + label + " | " + playgta6::os::window_hotkeys::target_hint(playgta6::os::window_hotkeys::active().table());
}

// Hotkey bits (NativeWindow::take_hotkeys).
inline constexpr uint32_t kHotkeyMip = 1, kHotkeyLayer = 2, kHotkeyPrev = 4, kHotkeyNext = 8;

}  // namespace playgta6::gpu::native_exec
