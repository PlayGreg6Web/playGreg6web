// Which @location outputs a WGSL fragment entry point writes. Dawn rejects a render pipeline whose
// color target has a non-zero writeMask but no matching fragment output ("color target N has no
// corresponding fragment output"), so targets the stage does not write get writeMask 0.
// Conservative text scan: nullopt means "could not tell" and the caller keeps the key's writeMask.
#pragma once

#include <cctype>
#include <cstdint>
#include <optional>
#include <string>

namespace playgta6::gpu::native_exec {

namespace fragout_detail {

inline bool is_id(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; }

// Index just past the `}` matching the `{` at `open`, or npos.
inline size_t match_brace(const std::string& s, size_t open) {
    int depth = 0;
    for (size_t i = open; i < s.size(); ++i) {
        if (s[i] == '{') ++depth;
        else if (s[i] == '}' && --depth == 0) return i + 1;
    }
    return std::string::npos;
}

// OR of 1<<N for every `@location(N)` in [from, to).
inline uint32_t locations_in(const std::string& s, size_t from, size_t to) {
    uint32_t mask = 0;
    static const std::string kTag = "@location";
    for (size_t p = s.find(kTag, from); p != std::string::npos && p < to; p = s.find(kTag, p + 1)) {
        size_t i = p + kTag.size();
        while (i < to && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
        if (i >= to || s[i] != '(') continue;
        ++i;
        while (i < to && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
        uint32_t n = 0;
        bool any = false;
        while (i < to && std::isdigit(static_cast<unsigned char>(s[i]))) { n = n * 10 + (s[i] - '0'); ++i; any = true; }
        if (any && n < 32) mask |= 1u << n;
    }
    return mask;
}

}  // namespace fragout_detail

// Bit N set: the fragment entry point `ep` (empty: the first @fragment) writes @location(N).
inline std::optional<uint32_t> wgsl_fragment_output_mask(const std::string& code, const std::string& ep) {
    using namespace fragout_detail;
    size_t at = code.find("@fragment");
    while (at != std::string::npos) {
        size_t fn = code.find("fn", at);
        while (fn != std::string::npos && !((fn == 0 || !is_id(code[fn - 1])) && fn + 2 < code.size() && !is_id(code[fn + 2])))
            fn = code.find("fn", fn + 1);
        if (fn == std::string::npos) return std::nullopt;
        size_t n = fn + 2;
        while (n < code.size() && std::isspace(static_cast<unsigned char>(code[n]))) ++n;
        size_t ne = n;
        while (ne < code.size() && is_id(code[ne])) ++ne;
        if (!ep.empty() && code.substr(n, ne - n) != ep) { at = code.find("@fragment", ne); continue; }
        const size_t lp = code.find('(', ne);
        if (lp == std::string::npos) return std::nullopt;
        int depth = 0;
        size_t rp = lp;
        for (; rp < code.size(); ++rp) {
            if (code[rp] == '(') ++depth;
            else if (code[rp] == ')' && --depth == 0) break;
        }
        const size_t body = code.find('{', rp);
        if (rp >= code.size() || body == std::string::npos) return std::nullopt;
        const size_t arrow = code.find("->", rp);
        if (arrow == std::string::npos || arrow > body) return 0u;  // no return value: writes nothing
        const uint32_t direct = locations_in(code, arrow, body);
        if (direct) return direct;
        // Return type: a struct name; its members carry the locations.
        size_t t = arrow + 2;
        while (t < body && !is_id(code[t])) ++t;
        size_t te = t;
        while (te < body && is_id(code[te])) ++te;
        const std::string type = code.substr(t, te - t);
        if (type.empty()) return std::nullopt;
        for (size_t s = code.find("struct"); s != std::string::npos; s = code.find("struct", s + 1)) {
            if (s > 0 && is_id(code[s - 1])) continue;
            size_t a = s + 6;
            while (a < code.size() && std::isspace(static_cast<unsigned char>(code[a]))) ++a;
            size_t ae = a;
            while (ae < code.size() && is_id(code[ae])) ++ae;
            if (code.substr(a, ae - a) != type) continue;
            const size_t ob = code.find('{', ae);
            if (ob == std::string::npos) return std::nullopt;
            const size_t cb = match_brace(code, ob);
            if (cb == std::string::npos) return std::nullopt;
            return locations_in(code, ob, cb);
        }
        return std::nullopt;
    }
    return std::nullopt;
}

}  // namespace playgta6::gpu::native_exec
