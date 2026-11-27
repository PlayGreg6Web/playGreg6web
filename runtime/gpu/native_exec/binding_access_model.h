// Native executor: which resource bindings a WGSL module can actually reach, and which of them it can write.
// Feeds the usage-scope check (usage_scope_model.h): D3D12 root signatures declare UAV / CBV / SRV ranges a
// given shader never touches, and the title binds one buffer through several of them in the same draw /
// dispatch. Dawn sees a writable binding plus another use of the buffer and rejects the scope, so the
// executor used to bind the read side from a snapshot copy (#2130, #2147) even when the shader never touches
// either side.
//
// Per declared `@group(G) @binding(B) var...` of a module:
//  - Unreferenced: no function body names the variable (also every binding the module does not declare).
//  - Read:         referenced, and nothing can store through it.
//  - Written:      `var<storage, read_write>` (or a storage texture / unknown kind) the module can write: an
//                  assignment (`=`, compound, `++` / `--`) to the variable or an element / member of it, or its
//                  address taken (`&v...`) anywhere but `arrayLength(&v)` / `atomicLoad(&v...)` (atomics,
//                  pointers passed on: assumed written).
// The scan is textual and conservative: anything it cannot account for is Written, and a module whose
// declarations it cannot all match (an `@binding` it did not parse, no function bodies) is not known at all
// (every binding counts as Written). Byte ranges of the writes are not derived: the translated shaders index
// storage arrays with runtime values only (a pack census found no literal-index writes).
// Pure; no GPU involved.
#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "gpu/native_exec/layout_model.h"

namespace playgta6::gpu::native_exec {

enum class BindingAccess : uint8_t { Unreferenced = 0, Read = 1, Written = 2 };

struct BindingAccessEntry {
    uint32_t group = 0, binding = 0;
    BindingAccess access = BindingAccess::Written;
    bool declared_writable = false;  // `read_write` / `write` storage buffer, or a storage texture
};

struct ShaderBindingAccess {
    bool known = false;                     // false: every binding counts as Written
    std::vector<BindingAccessEntry> entries;  // the declared bindings, sorted by (group, binding)

    BindingAccess of(uint32_t group, uint32_t binding) const {
        if (!known) return BindingAccess::Written;
        auto it = std::lower_bound(entries.begin(), entries.end(), std::make_pair(group, binding),
                                   [](const BindingAccessEntry& e, const std::pair<uint32_t, uint32_t>& k) {
                                       return e.group != k.first ? e.group < k.first : e.binding < k.second;
                                   });
        if (it == entries.end() || it->group != group || it->binding != binding) return BindingAccess::Unreferenced;
        return it->access;
    }
    // Declared writable but never written by the module.
    uint32_t writable_never_written() const {
        uint32_t n = 0;
        for (const BindingAccessEntry& e : entries)
            n += e.declared_writable && e.access != BindingAccess::Written;
        return n;
    }
};

namespace access_detail {

using namespace layout_detail;

inline size_t skip_ws_back(std::string_view s, size_t i) {  // index of the last non-ws char before i, or npos
    while (i > 0) {
        --i;
        if (!lw_is_ws(s[i])) return i;
    }
    return std::string_view::npos;
}

// The identifier that ends at (and includes) s[i], or "".
inline std::string_view ident_ending_at(std::string_view s, size_t i) {
    if (i == std::string_view::npos || !lw_is_ident_char(s[i])) return {};
    size_t a = i + 1;
    while (a > 0 && lw_is_ident_char(s[a - 1])) --a;
    return s.substr(a, i + 1 - a);
}

// Past a `[ ... ]` group starting at s[i] == '[' (balanced), or npos.
inline size_t skip_brackets(std::string_view s, size_t i) {
    int depth = 0;
    for (; i < s.size(); ++i) {
        if (s[i] == '[') ++depth;
        else if (s[i] == ']' && --depth == 0) return i + 1;
    }
    return std::string_view::npos;
}

// True when the occurrence of a variable name at [p, e) of the body text can store through it.
inline bool occurrence_writes(std::string_view s, size_t p, size_t e) {
    // Backwards: opening parens, then a unary `&`.
    size_t open = 0;
    size_t b = skip_ws_back(s, p);
    while (b != std::string_view::npos && s[b] == '(') {
        ++open;
        b = skip_ws_back(s, b);
    }
    if (b != std::string_view::npos && s[b] == '&') {
        const size_t before = skip_ws_back(s, b);
        const bool binary = before != std::string_view::npos &&
                            (lw_is_ident_char(s[before]) || s[before] == ')' || s[before] == ']' || s[before] == '&');
        if (!binary) {
            // `&v` as the argument of a read-only builtin: arrayLength((&v)), atomicLoad(&v[i])
            size_t c = skip_ws_back(s, b);
            size_t parens = 0;
            while (c != std::string_view::npos && s[c] == '(') {
                ++parens;
                c = skip_ws_back(s, c);
            }
            const std::string_view fn = ident_ending_at(s, c);
            if (parens > 0 && (fn == "arrayLength" || fn == "atomicLoad")) return false;
            return true;  // a pointer escapes (atomics, function arguments, let p = &v)
        }
    }
    // Forwards: the accessor chain (`[...]`, `.member`, closing parens of `(v)[i]`), then an assignment.
    size_t i = e;
    while (true) {
        i = lw_skip_ws(s, i);
        if (i >= s.size()) return false;
        if (s[i] == '[') {
            i = skip_brackets(s, i);
            if (i == std::string_view::npos) return true;  // unbalanced: assume the worst
        } else if (s[i] == '.') {
            i = lw_skip_ws(s, i + 1);
            while (i < s.size() && lw_is_ident_char(s[i])) ++i;
        } else if (s[i] == ')' && open > 0) {
            --open;
            ++i;
        } else {
            break;
        }
    }
    const char c0 = s[i], c1 = i + 1 < s.size() ? s[i + 1] : '\0', c2 = i + 2 < s.size() ? s[i + 2] : '\0';
    if (c0 == '=') return c1 != '=';
    if ((c0 == '+' && (c1 == '=' || c1 == '+')) || (c0 == '-' && (c1 == '=' || c1 == '-'))) return true;
    if ((c0 == '*' || c0 == '/' || c0 == '%' || c0 == '&' || c0 == '|' || c0 == '^') && c1 == '=') return true;
    if ((c0 == '<' && c1 == '<' && c2 == '=') || (c0 == '>' && c1 == '>' && c2 == '=')) return true;
    return false;
}

inline size_t count_token(std::string_view s, std::string_view tok) {
    size_t n = 0;
    for (size_t p = s.find(tok); p != std::string_view::npos; p = s.find(tok, p + tok.size()))
        if (p + tok.size() >= s.size() || !lw_is_ident_char(s[p + tok.size()])) ++n;
    return n;
}

}  // namespace access_detail

inline ShaderBindingAccess scan_binding_access(std::string_view wgsl) {
    using namespace access_detail;
    ShaderBindingAccess out;
    const std::string cleaned = strip_wgsl_comments(wgsl);
    if (cleaned.empty()) return out;

    struct Var {
        uint32_t group, binding;
        std::string name;
        bool writable_decl = false;  // can be written at all (read_write storage, storage texture, unknown kind)
        bool referenced = false, written = false;
        bool declared_writable = false;
        bool written_if_referenced = false;  // storage texture (textureStore), or a kind the scan does not know
    };
    std::vector<Var> vars;
    for (size_t i = 0; i < cleaned.size();) {
        VarMatch m;
        if (cleaned[i] == '@' && match_group0_var(cleaned, i, &m, true)) {
            Var v{m.group, m.binding, m.var_name};
            const std::string type = lw_trim(m.type_raw);
            if (!m.addr_raw.empty()) {
                const std::string a = lw_lower(m.addr_raw);
                const size_t comma = a.find(',');
                const std::string space = lw_trim(a.substr(0, comma));
                const std::string mode = comma == std::string::npos ? "" : lw_trim(a.substr(comma + 1));
                if (space == "uniform" || (space == "storage" && (mode.empty() || mode == "read"))) v.writable_decl = false;
                else {
                    v.writable_decl = true;  // read_write / write storage, or an address space the scan does not know
                    v.declared_writable = space == "storage";
                    v.written_if_referenced = space != "storage";
                }
            } else if (type.rfind("texture_storage_", 0) == 0) {
                v.writable_decl = type.find(",read>") == std::string::npos && type.find(", read>") == std::string::npos;
                v.declared_writable = v.writable_decl;
                v.written_if_referenced = v.writable_decl;
            } else if (type.rfind("texture_", 0) == 0 || type.rfind("sampler", 0) == 0) {
                v.writable_decl = false;
            } else {
                v.writable_decl = true;  // an alias or a type the scan does not know
                v.written_if_referenced = true;
            }
            vars.push_back(std::move(v));
            i = m.end;
        } else {
            ++i;
        }
    }
    // Every `@binding` must belong to a matched declaration, or bindings would look unreferenced.
    if (count_token(cleaned, "@binding") != vars.size()) return out;
    if (vars.empty()) {
        out.known = true;
        return out;
    }
    std::string body;
    if (!extract_function_bodies(cleaned, &body)) return out;

    std::unordered_map<std::string_view, std::vector<size_t>> by_name;
    for (size_t k = 0; k < vars.size(); ++k) by_name[vars[k].name].push_back(k);
    const std::string_view s = body;
    for (size_t i = 0; i < s.size();) {
        if (!lw_is_ident_start(s[i]) || (i > 0 && lw_is_ident_char(s[i - 1]))) {
            ++i;
            continue;
        }
        size_t e = i;
        while (e < s.size() && lw_is_ident_char(s[e])) ++e;
        auto it = by_name.find(s.substr(i, e - i));
        const size_t before = skip_ws_back(s, i);
        const bool member = before != std::string_view::npos && s[before] == '.';
        if (it != by_name.end() && !member) {
            bool w = false, w_known = false;
            for (size_t k : it->second) {
                Var& v = vars[k];
                v.referenced = true;
                if (!v.writable_decl || v.written) continue;
                if (v.written_if_referenced) {
                    v.written = true;
                    continue;
                }
                if (!w_known) {
                    w = occurrence_writes(s, i, e);
                    w_known = true;
                }
                if (w) v.written = true;
            }
        }
        i = e;
    }

    for (const Var& v : vars) {
        BindingAccessEntry en;
        en.group = v.group;
        en.binding = v.binding;
        en.declared_writable = v.declared_writable;
        en.access = !v.referenced ? BindingAccess::Unreferenced : v.written ? BindingAccess::Written : BindingAccess::Read;
        // One binding declared twice (different entry points): the strongest access wins.
        bool merged = false;
        for (BindingAccessEntry& o : out.entries)
            if (o.group == en.group && o.binding == en.binding) {
                o.access = std::max(o.access, en.access);
                o.declared_writable = o.declared_writable || en.declared_writable;
                merged = true;
            }
        if (!merged) out.entries.push_back(en);
    }
    std::sort(out.entries.begin(), out.entries.end(), [](const BindingAccessEntry& a, const BindingAccessEntry& b) {
        return a.group != b.group ? a.group < b.group : a.binding < b.binding;
    });
    out.known = true;
    return out;
}

// A pipeline's view: the strongest access over its stages (any stage unknown: unknown).
inline ShaderBindingAccess merge_binding_access(const ShaderBindingAccess& a, const ShaderBindingAccess* b) {
    if (!b) return a;
    ShaderBindingAccess out;
    if (!a.known || !b->known) return out;
    out.known = true;
    out.entries = a.entries;
    for (const BindingAccessEntry& e : b->entries) {
        bool merged = false;
        for (BindingAccessEntry& o : out.entries)
            if (o.group == e.group && o.binding == e.binding) {
                o.access = std::max(o.access, e.access);
                o.declared_writable = o.declared_writable || e.declared_writable;
                merged = true;
            }
        if (!merged) out.entries.push_back(e);
    }
    std::sort(out.entries.begin(), out.entries.end(), [](const BindingAccessEntry& x, const BindingAccessEntry& y) {
        return x.group != y.group ? x.group < y.group : x.binding < y.binding;
    });
    return out;
}

}  // namespace playgta6::gpu::native_exec
