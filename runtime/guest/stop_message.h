// The one line a person sees when the guest stops: "Stopped: <slot> - needs an owner decision (see
// docs/HARD_STOPS.md §7). Not a crash." for an unimplemented slot listed in docs/HARD_STOPS.md, otherwise
// "Crashed: <kind> at <site> - report: <run dir> (see docs/WHEN_IT_CRASHES.md)". Pure logic, no GPU, no I/O.
// The launcher (tools/stop_message.py, same texts, same tests) passes the HARD_STOPS.md slot list in
// $PLAYGTA6_HARD_STOP_SLOTS as "iface/slot/section;..." (iface lower case without the "xgameruntime." prefix,
// section the "## N." of the detail text or "-"). The harness shows the text in the window (overlay banner).
#pragma once

#include <cstdlib>
#include <string>
#include <vector>

namespace playgta6::guest::stop_message {

struct PendingSlot {
    std::string iface;  // lower case, last dotted segment
    int slot = -1;
    std::string section;  // "7", or empty when no detail section is known
};

struct Facts {
    std::string kind;  // fault_record.kind, "" when unknown
    std::string site;  // "0x..." faulting address, "" when unknown
    std::string iface, slot_name;  // the unimplemented COM slot, when the fault is one
    int slot = -1;
    std::string run_dir;
};

inline std::string lower(std::string s) {
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
    return s;
}

inline std::string short_iface(const std::string& iface) {
    const size_t dot = iface.rfind('.');
    return lower(dot == std::string::npos ? iface : iface.substr(dot + 1));
}

inline std::vector<PendingSlot> parse_pending(const std::string& arg) {
    std::vector<PendingSlot> out;
    size_t pos = 0;
    while (pos < arg.size()) {
        size_t end = arg.find(';', pos);
        if (end == std::string::npos) end = arg.size();
        const std::string item = arg.substr(pos, end - pos);
        pos = end + 1;
        const size_t a = item.find('/');
        const size_t b = a == std::string::npos ? a : item.find('/', a + 1);
        if (b == std::string::npos) continue;
        PendingSlot p;
        p.iface = lower(item.substr(0, a));
        p.slot = std::atoi(item.substr(a + 1, b - a - 1).c_str());
        p.section = item.substr(b + 1);
        if (p.section == "-") p.section.clear();
        out.push_back(p);
    }
    return out;
}

// Decodes a JSON string value starting at json[pos] == '"' (\uXXXX below 0x80 and the simple escapes).
inline std::string read_json_string(const std::string& json, size_t pos) {
    std::string out;
    for (size_t i = pos + 1; i < json.size() && json[i] != '"'; ++i) {
        if (json[i] != '\\' || i + 1 >= json.size()) {
            out += json[i];
            continue;
        }
        const char e = json[++i];
        if (e == 'u' && i + 4 < json.size()) {
            out += static_cast<char>(std::strtol(json.substr(i + 1, 4).c_str(), nullptr, 16));
            i += 4;
        } else {
            out += e == 'n' ? '\n' : e;
        }
    }
    return out;
}

// json[from..]: the value after the first `"key":` (a string, or a bare token like a number or null).
inline std::string json_value(const std::string& json, const std::string& key, size_t from = 0) {
    const std::string needle = "\"" + key + "\":";
    const size_t at = json.find(needle, from);
    if (at == std::string::npos) return {};
    size_t v = at + needle.size();
    while (v < json.size() && json[v] == ' ') ++v;
    if (v >= json.size()) return {};
    if (json[v] == '"') return read_json_string(json, v);
    size_t e = v;
    while (e < json.size() && json[e] != ',' && json[e] != '}' && json[e] != ' ') ++e;
    const std::string tok = json.substr(v, e - v);
    return tok == "null" || tok == "{" ? std::string() : tok;
}

// Facts from the boot report's fault_record (kind, va, unimplemented_slot {interface, slot, name}).
inline Facts facts_from_report(const std::string& report_json, const std::string& run_dir) {
    Facts f;
    f.run_dir = run_dir;
    const size_t rec = report_json.find("\"fault_record\":");
    if (rec == std::string::npos) return f;
    f.kind = json_value(report_json, "kind", rec);
    f.site = json_value(report_json, "va", rec);
    const size_t slot = report_json.find("\"unimplemented_slot\": {", rec);
    if (slot != std::string::npos) {
        f.iface = json_value(report_json, "interface", slot);
        f.slot = std::atoi(json_value(report_json, "slot", slot).c_str());
        f.slot_name = json_value(report_json, "name", slot);
    }
    return f;
}

// True and the section when the slot is one docs/HARD_STOPS.md lists.
inline bool is_hard_stop(const Facts& f, const std::vector<PendingSlot>& pending, std::string* section) {
    if (f.iface.empty() || f.slot < 0) return false;
    const std::string want = short_iface(f.iface);
    for (const PendingSlot& p : pending)
        if (p.slot == f.slot && p.iface == want) {
            *section = p.section;
            return true;
        }
    return false;
}

// "Stopped: ..." for a listed hard stop, "Crashed: ..." for any other fault.
inline std::string select(const Facts& f, const std::vector<PendingSlot>& pending) {
    std::string section;
    if (is_hard_stop(f, pending, &section)) {
        const std::string name =
            !f.slot_name.empty() ? f.slot_name : short_iface(f.iface) + " slot " + std::to_string(f.slot);
        return "Stopped: " + name + " \xE2\x80\x94 this needs an owner decision (see docs/HARD_STOPS.md" +
               (section.empty() ? std::string() : std::string(" \xC2\xA7") + section) + "). Not a crash.";
    }
    std::string kind = f.kind.empty() ? "fault" : f.kind;
    return "Crashed: " + kind + " at " + (f.site.empty() ? std::string("unknown site") : f.site) +
           " \xE2\x80\x94 report: " + (f.run_dir.empty() ? std::string("(see the console)") : f.run_dir) +
           " (see docs/WHEN_IT_CRASHES.md)";
}

}  // namespace playgta6::guest::stop_message
