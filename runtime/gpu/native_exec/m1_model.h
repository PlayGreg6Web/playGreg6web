// Native executor milestone 1, GPU-independent part: the rules of web/workers/wgsl_pack_loader.js,
// render_state_packets.js and texture_packets.js that decide WHAT is sent to the GPU (WGSL pack
// lookup, entry point resolution, texture format tables, upload layout, viewport / scissor clamps).
// No WebGPU types, so it is unit-tested on every host build (tests/test_native_exec_model.cpp).
// Every function names the worker function it ports. Design: docs/NATIVE_EXECUTOR_DESIGN.md.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace playgta6::gpu::native_exec {

// ---- WGSL pack (.wgsl.pack, WGPK v1; shaders/src/pack.rs, wgsl_pack_loader.js) ----

// Index of a pack file. Only the entry table is read at load time; WGSL text is read on demand, so
// a 1 GiB pack costs a few MiB of memory. A later entry with the same hash replaces an earlier one
// (InlineWgslPackCache.loadPackBuffer sets `_entries` in order).
class WgslPack {
public:
    struct Entry {
        uint32_t stage = 0;  // WGPK stage code: 1 vertex, 2 fragment, 3 compute
        uint64_t offset = 0;
        uint32_t length = 0;
    };

    // false (and *error) for a bad magic / version / truncated or out-of-range entry table.
    bool load(const std::string& path, std::string* error) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) {
            if (error) *error = "cannot open the WGSL pack";
            return false;
        }
        auto fail = [&](const char* msg) {
            std::fclose(f);
            if (error) *error = msg;
            return false;
        };
        uint8_t head[16];
        if (std::fread(head, 1, 16, f) != 16) return fail("WGPK pack truncated (header)");
        if (std::memcmp(head, "WGPK", 4) != 0) return fail("invalid WGPK pack magic");
        uint32_t version, count;
        std::memcpy(&version, head + 4, 4);
        std::memcpy(&count, head + 8, 4);
        if (version != 1 && version != 2) return fail("unsupported WGPK pack version");
        // v1: 20-byte entries {hash u64, stage u8, pad[3], offset u32, length u32}; v2 (merged packs,
        // > 4 GiB): 24-byte entries {hash u64, stage u8, pad[3], length u32, offset u64}.
        const size_t entry_bytes = version == 2 ? 24 : 20;
        std::fseek(f, 0, SEEK_END);
#ifdef _WIN32
        const uint64_t size = static_cast<uint64_t>(_ftelli64(f));
#else
        const uint64_t size = static_cast<uint64_t>(std::ftell(f));
#endif
        const uint64_t data_start = 16 + uint64_t{count} * entry_bytes;
        if (size < data_start) return fail("WGPK pack truncated (entry table)");
        std::fseek(f, 16, SEEK_SET);
        std::vector<uint8_t> table(size_t{count} * entry_bytes);
        if (!table.empty() && std::fread(table.data(), 1, table.size(), f) != table.size())
            return fail("WGPK pack truncated (entry table)");
        entries_.clear();
        entries_.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            const uint8_t* e = table.data() + size_t{i} * entry_bytes;
            uint64_t hash, offset;
            uint32_t length;
            std::memcpy(&hash, e, 8);
            if (version == 2) {
                std::memcpy(&length, e + 12, 4);
                std::memcpy(&offset, e + 16, 8);
            } else {
                uint32_t o32, l32;
                std::memcpy(&o32, e + 12, 4);
                std::memcpy(&l32, e + 16, 4);
                offset = o32;
                length = l32;
            }
            const uint8_t stage = e[8];
            if (stage < 1 || stage > 3) return fail("invalid WGPK shader stage");
            if (offset < data_start || offset > size || offset + length > size) return fail("WGPK entry out of bounds");
            entries_[hash] = Entry{stage, offset, length};
        }
        path_ = path;
        file_ = f;
        return true;
    }

    ~WgslPack() {
        if (file_) std::fclose(file_);
    }
    WgslPack() = default;
    WgslPack(const WgslPack&) = delete;
    WgslPack& operator=(const WgslPack&) = delete;

    bool loaded() const { return file_ != nullptr; }
    size_t size() const { return entries_.size(); }
    bool has(uint64_t hash) const { return entries_.count(hash) != 0; }

    // Thread-safe (the shader workers read concurrently): one seek + read at a time.
    std::optional<std::string> get(uint64_t hash) {
        auto it = entries_.find(hash);
        if (it == entries_.end() || !file_) return std::nullopt;
        std::lock_guard<std::mutex> lock(read_mu_);
#ifdef _WIN32
        _fseeki64(file_, static_cast<long long>(it->second.offset), SEEK_SET);
#else
        std::fseek(file_, static_cast<long>(it->second.offset), SEEK_SET);
#endif
        std::string out(it->second.length, '\0');
        if (it->second.length && std::fread(out.data(), 1, out.size(), file_) != out.size()) return std::nullopt;
        return out;
    }

private:
    std::string path_;
    FILE* file_ = nullptr;
    std::mutex read_mu_;
    std::unordered_map<uint64_t, Entry> entries_;
};

inline bool is_hex(char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'); }
inline bool is_space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }
inline bool is_ident_start(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }
inline bool is_ident_char(char c) { return is_ident_start(c) || (c >= '0' && c <= '9'); }

inline std::string_view trim_view(std::string_view s) {
    size_t a = 0, b = s.size();
    while (a < b && is_space(s[a])) ++a;
    while (b > a && is_space(s[b - 1])) --b;
    return s.substr(a, b - a);
}

inline std::optional<uint64_t> parse_hex_u64(std::string_view h) {
    if (h.empty() || h.size() > 16) return std::nullopt;
    uint64_t v = 0;
    for (char c : h) {
        if (!is_hex(c)) return std::nullopt;
        v = (v << 4) | static_cast<uint64_t>(c <= '9' ? c - '0' : (c | 0x20) - 'a' + 10);
    }
    return v;
}

// wgpu_worker.js parseHashOnlyShaderPayload (the argument is already trimmed): `hash:` / `wgpk:`
// prefixed hex (case-insensitive prefix, optional 0x), a bare 0x-hex, or exactly 16 hex digits.
inline std::optional<uint64_t> parse_hash_only_shader_payload(std::string_view t) {
    if (t.empty()) return std::nullopt;
    auto ieq = [](std::string_view s, const char* lit) {
        const size_t n = std::strlen(lit);
        if (s.size() < n) return false;
        for (size_t i = 0; i < n; ++i)
            if ((s[i] | 0x20) != lit[i]) return false;
        return true;
    };
    std::string_view rest;
    if (ieq(t, "hash") || ieq(t, "wgpk")) {
        rest = t.substr(4);
        while (!rest.empty() && is_space(rest.front())) rest.remove_prefix(1);
        if (rest.empty() || rest.front() != ':') return std::nullopt;
        rest.remove_prefix(1);
        while (!rest.empty() && is_space(rest.front())) rest.remove_prefix(1);
        if (rest.size() >= 2 && rest[0] == '0' && (rest[1] == 'x' || rest[1] == 'X')) {
            // `(?:0[xX])?([0-9a-fA-F]{1,16})$`: a leading "0x" is the prefix unless the digits
            // that follow it are not hex (then the "0" is a digit and the "x" fails the match).
            rest.remove_prefix(2);
        }
        return parse_hex_u64(rest);
    }
    if (t.size() >= 3 && t[0] == '0' && (t[1] == 'x' || t[1] == 'X')) return parse_hex_u64(t.substr(2));
    if (t.size() == 16) return parse_hex_u64(t);
    return std::nullopt;
}

// A pack shader whose @group(0) bindings are renumbered (gpu::pack_key_code(key, shift), for
// fragment stages of packs translated before the disjoint fragment numbering):
// `wgpk:0x<1..16 hex>;group0_binding_shift=<decimal 1..999>`, exactly. Not a hash-only payload.
struct ShiftedPackShader {
    uint64_t hash = 0;
    uint32_t group0_binding_shift = 0;
};
inline std::optional<ShiftedPackShader> parse_shifted_pack_shader_payload(std::string_view t) {
    static constexpr std::string_view kSep = ";group0_binding_shift=";
    const size_t sep = t.find(kSep);
    if (sep == std::string_view::npos || t.substr(0, 7) != "wgpk:0x") return std::nullopt;
    const std::optional<uint64_t> hash = parse_hex_u64(t.substr(7, sep - 7));
    const std::string_view digits = t.substr(sep + kSep.size());
    if (!hash || digits.empty() || digits.size() > 3) return std::nullopt;
    uint32_t shift = 0;
    for (char c : digits) {
        if (c < '0' || c > '9') return std::nullopt;
        shift = shift * 10 + static_cast<uint32_t>(c - '0');
    }
    if (shift == 0) return std::nullopt;
    return ShiftedPackShader{*hash, shift};
}

// Raises N of every `@group(0) @binding(N)` (either attribute order, whitespace between them
// allowed) by `shift`; other groups and all other text are kept byte for byte.
inline std::string shift_wgsl_group0_bindings(std::string_view wgsl, uint32_t shift) {
    auto ws = [&](size_t p) {
        while (p < wgsl.size() && is_space(wgsl[p])) ++p;
        return p;
    };
    // `@name ( digits )` at p: the digits' [begin, end) and the position after `)`.
    auto attr = [&](size_t p, std::string_view name, size_t* d0, size_t* d1, size_t* after) {
        if (wgsl.substr(p, name.size()) != name) return false;
        p = ws(p + name.size());
        if (p >= wgsl.size() || wgsl[p] != '(') return false;
        p = ws(p + 1);
        const size_t b = p;
        while (p < wgsl.size() && wgsl[p] >= '0' && wgsl[p] <= '9') ++p;
        if (p == b || p - b > 9) return false;
        const size_t e = p;
        p = ws(p);
        if (p >= wgsl.size() || wgsl[p] != ')') return false;
        *d0 = b;
        *d1 = e;
        *after = p + 1;
        return true;
    };
    auto is_zero = [&](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i)
            if (wgsl[i] != '0') return false;
        return true;
    };
    std::string out;
    out.reserve(wgsl.size() + 64);
    size_t copied = 0;
    for (size_t i = wgsl.find('@'); i != std::string_view::npos; i = wgsl.find('@', i)) {
        size_t g0, g1, b0, b1, a1, a2;
        bool hit = false;
        if (attr(i, "@group", &g0, &g1, &a1) && attr(ws(a1), "@binding", &b0, &b1, &a2)) {
            hit = is_zero(g0, g1);
        } else if (attr(i, "@binding", &b0, &b1, &a1) && attr(ws(a1), "@group", &g0, &g1, &a2)) {
            hit = is_zero(g0, g1);
        }
        if (!hit) {
            ++i;
            continue;
        }
        uint64_t n = 0;
        for (size_t k = b0; k < b1; ++k) n = n * 10 + static_cast<uint64_t>(wgsl[k] - '0');
        out.append(wgsl.substr(copied, b0 - copied));
        out += std::to_string(n + shift);
        copied = b1;
        i = a2;
    }
    out.append(wgsl.substr(copied));
    return out;
}

// ---- WGSL scanning (render_state_packets.js) ----

// stripWgslComments: `//` line and nestable `/* */` block comments are removed (newlines kept).
inline std::string strip_wgsl_comments(std::string_view code) {
    std::string out;
    out.reserve(code.size());
    int depth = 0;
    for (size_t i = 0; i < code.size(); ++i) {
        const char ch = code[i];
        const char next = i + 1 < code.size() ? code[i + 1] : '\0';
        if (depth > 0) {
            if (ch == '/' && next == '*') {
                ++depth;
                ++i;
            } else if (ch == '*' && next == '/') {
                --depth;
                ++i;
                out += ' ';
            } else if (ch == '\n') {
                out += '\n';
            }
            continue;
        }
        if (ch == '/' && next == '/') {
            i += 2;
            while (i < code.size() && code[i] != '\n') ++i;
            if (i < code.size() && code[i] == '\n') out += '\n';
            continue;
        }
        if (ch == '/' && next == '*') {
            depth = 1;
            ++i;
            continue;
        }
        out += ch;
    }
    return out;
}

// Skips `\s*`.
inline size_t skip_ws(std::string_view s, size_t i) {
    while (i < s.size() && is_space(s[i])) ++i;
    return i;
}

// Matches the keyword `kw` at i with a word boundary after it; returns the index after it or npos.
inline size_t match_word(std::string_view s, size_t i, std::string_view kw) {
    if (s.compare(i, kw.size(), kw) != 0) return std::string_view::npos;
    const size_t e = i + kw.size();
    if (e < s.size() && is_ident_char(s[e])) return std::string_view::npos;
    return e;
}

// findWgslStageEntryPoints: every `@<stage> (@attr(...))* fn name`, in order, without duplicates.
// stage: "vertex" | "fragment" | "compute".
inline std::vector<std::string> find_wgsl_stage_entry_points_cleaned(std::string_view s, std::string_view stage) {
    std::vector<std::string> found;
    if (s.empty() || (stage != "vertex" && stage != "fragment" && stage != "compute")) return found;
    for (size_t at = s.find('@'); at != std::string_view::npos; at = s.find('@', at + 1)) {
        size_t i = match_word(s, at + 1, stage);
        if (i == std::string_view::npos) continue;
        for (;;) {  // (?:\s*@ident(?:\s*\([^)]*\))?)*
            size_t j = skip_ws(s, i);
            if (j >= s.size() || s[j] != '@' || j + 1 >= s.size() || !is_ident_start(s[j + 1])) break;
            ++j;
            while (j < s.size() && is_ident_char(s[j])) ++j;
            size_t k = skip_ws(s, j);
            if (k < s.size() && s[k] == '(') {
                const size_t close = s.find(')', k);
                if (close == std::string_view::npos) {
                    k = j;
                } else {
                    k = close + 1;
                }
                j = k;
            }
            i = j;
        }
        size_t j = skip_ws(s, i);
        j = match_word(s, j, "fn");
        if (j == std::string_view::npos) continue;
        const size_t name_at = skip_ws(s, j);
        if (name_at == j || name_at >= s.size() || !is_ident_start(s[name_at])) continue;
        size_t e = name_at;
        while (e < s.size() && is_ident_char(s[e])) ++e;
        std::string name(s.substr(name_at, e - name_at));
        if (std::find(found.begin(), found.end(), name) == found.end()) found.push_back(std::move(name));
    }
    return found;
}

inline std::vector<std::string> find_wgsl_stage_entry_points(std::string_view code, std::string_view stage) {
    return find_wgsl_stage_entry_points_cleaned(strip_wgsl_comments(code), stage);
}

// wgpu_worker.js extractWgslEntryPoint: the first `@<stage>` followed (anywhere, lazily) by
// `fn name`, else the first `fn name` of the source. Comments are NOT stripped here.
// stage: "vertex" | "fragment" | "compute" | "" (none).
inline std::optional<std::string> extract_wgsl_entry_point(std::string_view code, std::string_view stage) {
    if (code.empty()) return std::nullopt;
    auto first_fn_after = [&](size_t from) -> std::optional<std::string> {
        for (size_t i = from; i + 2 < code.size() + 0; ++i) {
            if (code.compare(i, 2, "fn") != 0) continue;
            if (i > 0 && is_ident_char(code[i - 1])) continue;
            const size_t j = i + 2;
            if (j >= code.size() || !is_space(code[j])) continue;
            const size_t n = skip_ws(code, j);
            if (n >= code.size() || !is_ident_start(code[n])) continue;
            size_t e = n;
            while (e < code.size() && is_ident_char(code[e])) ++e;
            return std::string(code.substr(n, e - n));
        }
        return std::nullopt;
    };
    if (!stage.empty()) {
        for (size_t at = code.find('@'); at != std::string_view::npos; at = code.find('@', at + 1)) {
            const size_t e = match_word(code, at + 1, stage);
            if (e == std::string_view::npos) continue;
            if (auto r = first_fn_after(e)) return r;
            break;  // the lazy match found no `fn` after the first `@stage`: the regex fails
        }
    }
    return first_fn_after(0);
}

// resolveShaderEntryPoint(shaderEntry, stage): `explicit_ep` is the entry's recorded entryPoint
// (extract_wgsl_entry_point of the shader's own stage), `stage` the stage it is used as.
inline std::string resolve_shader_entry_point_cleaned(std::string_view cleaned, const std::optional<std::string>& explicit_ep,
                                                      std::string_view stage) {
    const char* def = stage == "fragment" ? "fs_main" : stage == "compute" ? "cs_main" : "vs_main";
    const auto declared = find_wgsl_stage_entry_points_cleaned(cleaned, stage);
    auto valid_ident = [](const std::string& s) {
        if (s.empty() || !is_ident_start(s[0])) return false;
        return std::all_of(s.begin(), s.end(), [](char c) { return is_ident_char(c); });
    };
    std::string ex;
    if (explicit_ep && valid_ident(*explicit_ep)) ex = *explicit_ep;
    if (!declared.empty()) {
        if (!ex.empty() && std::find(declared.begin(), declared.end(), ex) != declared.end()) return ex;
        if (std::find(declared.begin(), declared.end(), std::string(def)) != declared.end()) return def;
        return declared.front();
    }
    return ex.empty() ? std::string(def) : ex;
}

inline std::string resolve_shader_entry_point(std::string_view code, const std::optional<std::string>& explicit_ep,
                                              std::string_view stage) {
    return resolve_shader_entry_point_cleaned(strip_wgsl_comments(code), explicit_ep, stage);
}

// Preserve alternate-stage decisions with one comment-stripping pass before dropping pack text.
inline std::array<std::string, 3> resolve_all_shader_entry_points(
    std::string_view code, const std::optional<std::string>& explicit_ep) {
    const std::string cleaned = strip_wgsl_comments(code);
    return {resolve_shader_entry_point_cleaned(cleaned, explicit_ep, "vertex"),
            resolve_shader_entry_point_cleaned(cleaned, explicit_ep, "fragment"),
            resolve_shader_entry_point_cleaned(cleaned, explicit_ep, "compute")};
}

// WGSL_GROUP0_RE: /@group\(\s*0\s*\)/
inline bool wgsl_declares_group0(std::string_view code) {
    for (size_t at = code.find("@group("); at != std::string_view::npos; at = code.find("@group(", at + 1)) {
        size_t i = skip_ws(code, at + 7);
        if (i >= code.size() || code[i] != '0') continue;
        i = skip_ws(code, i + 1);
        if (i < code.size() && code[i] == ')') return true;
    }
    return false;
}

// ---- Texture formats (docs/gpu_commands.md "WebGPU enum codes"; texture_packets.js) ----

struct FormatInfo {
    const char* name;
    uint8_t block_w, block_h, bytes;  // bytes per block
    bool compressed;
};

inline constexpr uint8_t kFormatCodeCount = 64;

inline const FormatInfo& format_info(uint8_t code) {
    static const FormatInfo kTable[kFormatCodeCount] = {
        {"undefined", 1, 1, 0, false},
        {"r8unorm", 1, 1, 1, false}, {"r8snorm", 1, 1, 1, false}, {"r8uint", 1, 1, 1, false},
        {"r8sint", 1, 1, 1, false}, {"r16unorm", 1, 1, 2, false}, {"r16snorm", 1, 1, 2, false},
        {"r16uint", 1, 1, 2, false}, {"r16sint", 1, 1, 2, false}, {"r16float", 1, 1, 2, false},
        {"rg8unorm", 1, 1, 2, false}, {"rg8snorm", 1, 1, 2, false}, {"rg8uint", 1, 1, 2, false},
        {"rg8sint", 1, 1, 2, false}, {"r32uint", 1, 1, 4, false}, {"r32sint", 1, 1, 4, false},
        {"r32float", 1, 1, 4, false}, {"rg16unorm", 1, 1, 4, false}, {"rg16snorm", 1, 1, 4, false},
        {"rg16uint", 1, 1, 4, false}, {"rg16sint", 1, 1, 4, false}, {"rg16float", 1, 1, 4, false},
        {"rgba8unorm", 1, 1, 4, false}, {"rgba8unorm-srgb", 1, 1, 4, false}, {"rgba8snorm", 1, 1, 4, false},
        {"rgba8uint", 1, 1, 4, false}, {"rgba8sint", 1, 1, 4, false}, {"bgra8unorm", 1, 1, 4, false},
        {"bgra8unorm-srgb", 1, 1, 4, false}, {"rgb9e5ufloat", 1, 1, 4, false}, {"rgb10a2uint", 1, 1, 4, false},
        {"rgb10a2unorm", 1, 1, 4, false}, {"rg11b10ufloat", 1, 1, 4, false}, {"rg32uint", 1, 1, 8, false},
        {"rg32sint", 1, 1, 8, false}, {"rg32float", 1, 1, 8, false}, {"rgba16unorm", 1, 1, 8, false},
        {"rgba16snorm", 1, 1, 8, false}, {"rgba16uint", 1, 1, 8, false}, {"rgba16sint", 1, 1, 8, false},
        {"rgba16float", 1, 1, 8, false}, {"rgba32uint", 1, 1, 16, false}, {"rgba32sint", 1, 1, 16, false},
        {"rgba32float", 1, 1, 16, false}, {"stencil8", 1, 1, 1, false}, {"depth16unorm", 1, 1, 2, false},
        {"depth24plus", 1, 1, 4, false}, {"depth24plus-stencil8", 1, 1, 4, false}, {"depth32float", 1, 1, 4, false},
        {"depth32float-stencil8", 1, 1, 8, false}, {"bc1-rgba-unorm", 4, 4, 8, true},
        {"bc1-rgba-unorm-srgb", 4, 4, 8, true}, {"bc2-rgba-unorm", 4, 4, 16, true},
        {"bc2-rgba-unorm-srgb", 4, 4, 16, true}, {"bc3-rgba-unorm", 4, 4, 16, true},
        {"bc3-rgba-unorm-srgb", 4, 4, 16, true}, {"bc4-r-unorm", 4, 4, 8, true}, {"bc4-r-snorm", 4, 4, 8, true},
        {"bc5-rg-unorm", 4, 4, 16, true}, {"bc5-rg-snorm", 4, 4, 16, true}, {"bc6h-rgb-ufloat", 4, 4, 16, true},
        {"bc6h-rgb-float", 4, 4, 16, true}, {"bc7-rgba-unorm", 4, 4, 16, true}, {"bc7-rgba-unorm-srgb", 4, 4, 16, true},
    };
    return kTable[code < kFormatCodeCount ? code : 0];
}

inline bool format_valid(uint8_t code) { return code != 0 && code < kFormatCodeCount; }
inline bool format_is_bc(uint8_t c) { return c >= 50 && c < kFormatCodeCount; }
inline bool format_is_srgb(uint8_t c) {
    return c == 23 || c == 28 || c == 51 || c == 53 || c == 55 || c == 63;
}
// DEPTH_TEXTURE_FORMATS (the ones with a depth aspect).
inline bool format_is_depth(uint8_t c) { return c >= 45 && c <= 49; }
inline bool format_is_depth_or_stencil(uint8_t c) { return c == 44 || format_is_depth(c); }
inline bool format_is_float32(uint8_t c) { return c == 16 || c == 35 || c == 43; }
// isIntegerTextureFormat: the name ends in uint / sint.
inline bool format_is_integer(uint8_t c) {
    const std::string_view n = format_info(c).name;
    return n.size() >= 4 && (n.substr(n.size() - 4) == "uint" || n.substr(n.size() - 4) == "sint");
}

// getSrgbSiblingFormat: rgba8 / bgra8 / bc1,2,3,7 linear <-> sRGB; 0 = none.
inline uint8_t srgb_sibling(uint8_t c) {
    switch (c) {
        case 22: return 23; case 23: return 22; case 27: return 28; case 28: return 27;
        case 50: return 51; case 51: return 50; case 52: return 53; case 53: return 52;
        case 54: return 55; case 55: return 54; case 62: return 63; case 63: return 62;
        default: return 0;
    }
}

// v1 GpuTextureFormat -> v2 format code (V1_TEXTURE_FORMAT_NAMES; unknown -> rgba8unorm).
inline uint8_t v1_format_code(uint32_t v1) {
    switch (v1) {
        case 0: return 22;  // rgba8unorm
        case 1: return 27;  // bgra8unorm
        case 2: return 40;  // rgba16float
        case 3: return 46;  // depth24plus
        case 4: return 48;  // depth32float
        case 5: return 62;  // bc7-rgba-unorm
        default: return 22;
    }
}

// resolveCompatibleViewFormats(format, requested): requested formats compatible with `format`
// (itself or its sRGB / linear sibling), plus the sibling when the format has one.
inline std::vector<uint8_t> resolve_compatible_view_formats(uint8_t format, const std::vector<uint8_t>& requested,
                                                            bool* sibling_added = nullptr) {
    const uint8_t sib = srgb_sibling(format);
    std::vector<uint8_t> out;
    bool added = false;
    if (requested.empty()) {
        if (sib) {
            out.push_back(sib);
            added = true;
        }
    } else {
        for (uint8_t vf : requested) {
            if (vf == 0 || std::find(out.begin(), out.end(), vf) != out.end()) continue;
            if (vf == format || (sib && vf == sib)) out.push_back(vf);
        }
        if (sib && std::find(out.begin(), out.end(), sib) == out.end()) {
            out.push_back(sib);
            added = true;
        }
    }
    if (sibling_added) *sibling_added = added;
    return out;
}

// computeMaxMipLevels / clampMipLevels / clampSampleCount (texture_packets.js).
inline uint32_t max_mip_levels(uint32_t w, uint32_t h, uint32_t d, uint8_t dimension /*0 1d,1 2d,2 3d*/) {
    if (dimension == 0) return 1;
    const uint32_t m = dimension == 2 ? std::max({w, h, d}) : std::max(w, h);
    uint32_t n = 0;
    for (uint32_t v = m ? m : 1; v; v >>= 1) ++n;
    return n;
}
inline uint32_t clamp_mip_levels(uint32_t requested, uint32_t w, uint32_t h, uint32_t d, uint8_t dimension) {
    if (requested < 1) return 1;
    return std::min(max_mip_levels(w, h, d, dimension), requested);
}
inline uint32_t clamp_sample_count(uint32_t requested, uint32_t mips, uint8_t dimension, uint32_t layers) {
    if (requested <= 1) return 1;
    return requested >= 4 && dimension == 1 && mips <= 1 && layers == 1 ? 4 : 1;
}

// calcMipDimension.
inline uint32_t mip_dimension(uint32_t base, uint32_t mip) {
    if (base <= 1) return 1;
    if (mip == 0) return base;
    if (mip >= 32) return 1;
    return std::max<uint32_t>(1, base >> mip);
}

// ---- WriteTexture planning (texture_packets.js handleWriteTexture + clampSubRegionToMipBoundsInto) ----

struct TextureGeometry {  // the tracked meta of a texture (effective descriptor)
    uint32_t width = 1, height = 1, depth_or_layers = 1, mips = 1;
    uint8_t dimension = 1;  // 0 1d, 1 2d, 2 3d (effective: a promoted 1d is 2d)
    bool requested_1d = false;
    uint8_t format = 22;
};

struct WriteTexturePlan {
    enum class Status : uint8_t {
        Write,
        Skip,         // nothing to write (empty / out of bounds / truncated): silent in the worker
        BadMip,       // mip level out of bounds: the worker records an error
        BadRowBytes,  // compressed bytesPerRow below the minimum: error
    } status = Status::Skip;
    uint32_t mip = 0, ox = 0, oy = 0, oz = 0, width = 0, height = 0, depth = 0;
    uint32_t bytes_per_row = 0;   // 0 = unset (WebGPU infers it for a single row)
    uint32_t rows_per_image = 0;  // 0 = unset
    bool clamped = false;
};

// The fields of a WriteTexture payload (opcode 13).
struct WriteTextureArgs {
    uint32_t mip = 0, ox = 0, oy = 0, oz = 0, width = 0, height = 0, depth = 1;
    uint32_t bytes_per_row = 0, rows_per_image = 0, data_len = 0;
};

inline WriteTexturePlan plan_write_texture(const WriteTextureArgs& a, const TextureGeometry& g) {
    WriteTexturePlan out;
    const FormatInfo& fi = format_info(g.format);
    const bool is_compressed = fi.compressed;
    const bool one_d = g.dimension == 0 || g.requested_1d;
    const uint32_t depth = std::max<uint32_t>(1, a.depth);
    const uint32_t height = (a.height == 0 && one_d && a.width > 0) ? 1 : a.height;
    const uint32_t bw = is_compressed ? fi.block_w : 1, bh = is_compressed ? fi.block_h : 1;
    const uint32_t orig_bx = a.width > 0 ? (a.width + bw - 1) / bw : 0;
    const uint32_t orig_by = height > 0 ? (height + bh - 1) / bh : 0;
    const uint64_t orig_min_row = is_compressed ? uint64_t{orig_bx} * fi.bytes : 0;

    uint32_t eff_rpi;
    if (a.rows_per_image > 0) {
        if (is_compressed && a.rows_per_image == height && a.rows_per_image > orig_by) {
            const uint64_t cand = a.bytes_per_row > 0 ? a.bytes_per_row : orig_min_row;
            if (cand > 0 && a.data_len < cand * a.rows_per_image * depth && a.data_len >= cand * orig_by * depth)
                eff_rpi = orig_by;
            else
                eff_rpi = a.rows_per_image;
        } else {
            eff_rpi = is_compressed ? std::max(orig_by, a.rows_per_image) : a.rows_per_image;
        }
    } else {
        eff_rpi = is_compressed ? orig_by : height;
    }
    const uint64_t orig_total_rows = uint64_t{eff_rpi} * depth;
    uint64_t eff_bpr = 0;
    if (a.bytes_per_row > 0) {
        eff_bpr = a.bytes_per_row;
    } else if (orig_total_rows > 0 && a.data_len > 0) {
        if (is_compressed) {
            if (a.data_len % orig_total_rows == 0 && a.data_len / orig_total_rows >= orig_min_row)
                eff_bpr = a.data_len / orig_total_rows;
            else if (a.data_len >= orig_min_row * orig_total_rows)
                eff_bpr = orig_min_row;
        } else if (a.data_len % orig_total_rows == 0) {
            eff_bpr = a.data_len / orig_total_rows;
        }
    }

    // clampSubRegionToMipBoundsInto
    uint32_t ox = a.ox, oy = a.oy, oz = a.oz, w = a.width, h = height, d = depth;
    bool clamped = false;
    if (is_compressed) {
        const uint32_t ax = ox / bw * bw, ay = oy / bh * bh;
        if (ax != ox) { ox = ax; clamped = true; }
        if (ay != oy) { oy = ay; clamped = true; }
    }
    if (w == 0 || h == 0 || d == 0) return out;  // invalid -> Skip
    if (is_compressed) {
        w = (w + bw - 1) / bw * bw;
        h = (h + bh - 1) / bh * bh;
    }
    if (a.mip >= g.mips) {
        out.status = WriteTexturePlan::Status::BadMip;
        return out;
    }
    const uint32_t base_w = std::max<uint32_t>(1, g.width);
    const uint32_t base_h = g.dimension == 0 ? 1 : std::max<uint32_t>(1, g.height);
    const uint32_t base_d = g.dimension == 0 ? 1 : std::max<uint32_t>(1, g.depth_or_layers);
    const uint32_t lw = mip_dimension(base_w, a.mip);
    const uint32_t lh = g.dimension == 0 ? 1 : mip_dimension(base_h, a.mip);
    const uint32_t md = g.dimension == 0 ? 1 : g.dimension == 2 ? mip_dimension(base_d, a.mip) : base_d;
    const uint32_t mw = is_compressed ? (lw + bw - 1) / bw * bw : lw;
    const uint32_t mh = is_compressed ? (lh + bh - 1) / bh * bh : lh;
    if (ox >= mw || oy >= mh || oz >= md) return out;
    const uint32_t max_w = mw - ox, max_h = mh - oy, max_d = md - oz;
    if (w > max_w) { w = is_compressed ? max_w / bw * bw : max_w; clamped = true; }
    if (h > max_h) { h = is_compressed ? max_h / bh * bh : max_h; clamped = true; }
    if (d > max_d) { d = max_d; clamped = true; }
    if (w == 0 || h == 0 || d == 0) return out;

    const uint32_t cbx = (w + bw - 1) / bw, cby = (h + bh - 1) / bh;
    const uint64_t min_row = is_compressed ? uint64_t{cbx} * fi.bytes : 0;
    if (d == 1 && eff_rpi > cby) eff_rpi = cby;
    else if (eff_rpi < cby) eff_rpi = cby;
    if (is_compressed && eff_bpr > 0 && min_row > 0 && eff_bpr < min_row) {
        out.status = WriteTexturePlan::Status::BadRowBytes;
        return out;
    }
    out.status = WriteTexturePlan::Status::Write;
    out.mip = a.mip;
    out.ox = ox; out.oy = oy; out.oz = oz;
    out.width = w; out.height = h; out.depth = d;
    out.clamped = clamped;
    // Direct queue.writeTexture layout (the worker's staging pool is absent in the replay page and
    // only changes how bytes reach the same texels): the row pitch as sent / inferred, rows per image
    // only when the packet or a multi-slice upload needs it.
    const uint64_t total_rows = uint64_t{eff_rpi} * d;
    if (a.bytes_per_row > 0)
        out.bytes_per_row = static_cast<uint32_t>(eff_bpr);
    else if (eff_bpr > 0 && (total_rows > 1 || is_compressed))
        out.bytes_per_row = static_cast<uint32_t>(eff_bpr);
    if (a.rows_per_image > 0)
        out.rows_per_image = eff_rpi;
    else if (d > 1 && eff_rpi > 0)
        out.rows_per_image = eff_rpi;
    return out;
}

// ---- Viewport / scissor clamps (render_state_packets.js) ----

struct Viewport {
    double x = 0, y = 0, width = 0, height = 0, min_depth = 0, max_depth = 1;
};
struct Rect {
    double x = 0, y = 0, width = 0, height = 0;
};

inline double finite_or(double v, double fb) { return std::isfinite(v) ? v : fb; }
inline double fround(double v) { return static_cast<double>(static_cast<float>(v)); }
inline double or_zero(double v) { return v == 0.0 || std::isnan(v) ? 0.0 : v; }  // JS `v || 0`

// clampViewportToTarget; target 0x0 = no target.
inline Viewport clamp_viewport_to_target(const Viewport& vp, double tw, double th) {
    const double rx = finite_or(vp.x, 0), ry = finite_or(vp.y, 0), rw = finite_or(vp.width, 0), rh = finite_or(vp.height, 0);
    double mn = std::min(1.0, std::max(0.0, finite_or(vp.min_depth, 0)));
    double mx = std::min(1.0, std::max(0.0, finite_or(vp.max_depth, 1)));
    if (mn > mx) std::swap(mn, mx);
    mn = or_zero(mn);
    mx = or_zero(mx);
    Viewport out;
    out.min_depth = mn;
    out.max_depth = mx;
    if (!(tw > 0 && th > 0 && std::isfinite(tw) && std::isfinite(th))) {
        out.x = std::isfinite(fround(rx)) ? or_zero(std::max(0.0, rx)) : 0;
        out.y = std::isfinite(fround(ry)) ? or_zero(std::max(0.0, ry)) : 0;
        out.width = std::isfinite(fround(rw)) ? or_zero(std::max(0.0, rw)) : 0;
        out.height = std::isfinite(fround(rh)) ? or_zero(std::max(0.0, rh)) : 0;
        return out;
    }
    if (rx >= 0 && ry >= 0 && rw >= 0 && rh >= 0 && rx + rw <= tw && ry + rh <= th &&
        fround(rx) + fround(rw) <= tw && fround(ry) + fround(rh) <= th) {
        out.x = or_zero(rx);
        out.y = or_zero(ry);
        out.width = or_zero(rw);
        out.height = or_zero(rh);
        return out;
    }
    double x = or_zero(std::min(std::max(0.0, rx), tw));
    if (fround(x) > tw) x = or_zero(std::max(0.0, std::floor(tw)));
    double y = or_zero(std::min(std::max(0.0, ry), th));
    if (fround(y) > th) y = or_zero(std::max(0.0, std::floor(th)));
    double w = or_zero(std::min(std::max(0.0, rw), tw - x));
    double h = or_zero(std::min(std::max(0.0, rh), th - y));
    if (x + w > tw || fround(x) + fround(w) > tw) w = or_zero(std::max(0.0, std::floor(tw - fround(x))));
    if (y + h > th || fround(y) + fround(h) > th) h = or_zero(std::max(0.0, std::floor(th - fround(y))));
    out.x = x; out.y = y; out.width = w; out.height = h;
    return out;
}

// clampScissorToTarget (integers; target 0x0 = no target).
inline Rect clamp_scissor_to_target(uint32_t rx, uint32_t ry, uint32_t rw, uint32_t rh, double tw, double th) {
    Rect out{double(rx), double(ry), double(rw), double(rh)};
    if (!(tw > 0 && th > 0 && std::isfinite(tw) && std::isfinite(th))) return out;
    const double w = std::floor(tw), h = std::floor(th);
    if (w <= 0 || h <= 0) return out;
    if (out.x + out.width <= w && out.y + out.height <= h) return out;
    const double x = std::min(out.x, w), y = std::min(out.y, h);
    return Rect{x, y, std::min(out.width, w - x), std::min(out.height, h - y)};
}

}  // namespace playgta6::gpu::native_exec
