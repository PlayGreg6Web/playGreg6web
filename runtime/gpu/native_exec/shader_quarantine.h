// Diagnostic shader quarantine for the native executor: a set of shader hashes whose draws and
// dispatches (indirect ones too) are skipped, so a program that hangs or crashes the GPU can be
// bisected or sidestepped without a rebuild. Sources: --skip-shader <hash>[,<hash>], --skip-shader-file
// <path>, and the repo list runtime/gpu/native_exec/shader_quarantine.txt (empty by default).
// Every skip is counted in the drop table as "quarantined" (gpu/drop_table.h); nothing is ever
// substituted for the skipped work.
//
// File format (one entry per line, '#' starts a comment, blank lines ignored):
//     <hash> <reason> <issue link>
//     0x00000000deadbeef  hangs the Dawn/D3D12 driver in the hair pass  https://github.com/PlayGta6Web/playgta6web/issues/123
// <hash> is 1..16 hex digits (optional 0x). The repo list (require_reason) also needs a reason and an
// issue link (a token starting with http or '#'); per-run files may be bare hashes.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace playgta6::gpu::native_exec {

class ShaderQuarantine {
public:
    struct Entry {
        uint64_t hash = 0;
        std::string note;
    };

    static bool parse_hash(const std::string& s, uint64_t* out) {
        size_t i = 0;
        if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) i = 2;
        const size_t digits = s.size() - i;
        if (digits == 0 || digits > 16) return false;
        uint64_t v = 0;
        for (; i < s.size(); ++i) {
            const char c = s[i];
            int d;
            if (c >= '0' && c <= '9') d = c - '0';
            else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
            else return false;
            v = (v << 4) | static_cast<uint64_t>(d);
        }
        *out = v;
        return true;
    }

    // "h1,h2,..." (the --skip-shader value). False + *error on a malformed hash; nothing is added then.
    bool add_list(const std::string& csv, std::string* error) {
        std::vector<uint64_t> got;
        size_t pos = 0;
        while (pos <= csv.size()) {
            size_t end = csv.find(',', pos);
            if (end == std::string::npos) end = csv.size();
            const std::string tok = trim(csv.substr(pos, end - pos));
            if (!tok.empty()) {
                uint64_t h = 0;
                if (!parse_hash(tok, &h)) {
                    if (error) *error = "bad shader hash '" + tok + "'";
                    return false;
                }
                got.push_back(h);
            }
            pos = end + 1;
        }
        for (uint64_t h : got) add(h, "--skip-shader");
        return true;
    }

    // File text. False + *error ("line N: ...") on the first bad line; nothing is added then.
    bool add_text(const std::string& text, bool require_reason, const std::string& origin, std::string* error) {
        std::vector<Entry> got;
        size_t pos = 0;
        int line_no = 0;
        while (pos <= text.size()) {
            size_t nl = text.find('\n', pos);
            if (nl == std::string::npos) nl = text.size();
            std::string line = text.substr(pos, nl - pos);
            pos = nl + 1;
            ++line_no;
            // Only a line-leading '#' is a comment; '#123' later in a line is an issue reference.
            const std::string t0 = trim(line);
            if (t0.empty() || t0[0] == '#') continue;
            size_t sp = t0.find_first_of(" \t");
            const std::string tok = t0.substr(0, sp);
            const std::string rest = sp == std::string::npos ? std::string() : trim(t0.substr(sp));
            Entry e;
            if (!parse_hash(tok, &e.hash)) {
                if (error) *error = origin + " line " + std::to_string(line_no) + ": bad shader hash '" + tok + "'";
                return false;
            }
            if (require_reason && (rest.find(' ') == std::string::npos ||
                                   (rest.find("http") == std::string::npos && rest.find('#') == std::string::npos))) {
                if (error)
                    *error = origin + " line " + std::to_string(line_no) + ": needs '<hash> <reason> <issue link>'";
                return false;
            }
            e.note = rest.empty() ? origin : rest;
            got.push_back(std::move(e));
        }
        for (auto& e : got) add(e.hash, e.note);
        return true;
    }

    void add(uint64_t hash, const std::string& note) {
        if (hash == 0) return;  // 0 = "unknown shader", never quarantined
        notes_.emplace(hash, note);
    }

    bool contains(uint64_t hash) const { return hash != 0 && notes_.count(hash) != 0; }
    bool empty() const { return notes_.empty(); }
    size_t size() const { return notes_.size(); }
    const std::unordered_map<uint64_t, std::string>& entries() const { return notes_; }

private:
    static std::string trim(const std::string& s) {
        size_t a = 0, b = s.size();
        while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r')) ++a;
        while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r')) --b;
        return s.substr(a, b - a);
    }
    std::unordered_map<uint64_t, std::string> notes_;
};

}  // namespace playgta6::gpu::native_exec
