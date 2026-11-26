#pragma once
// Fault minidumps for the native harness (host faults, and guest faults with --dump-on-fault).
// The dump goes next to the boot report (the run directory, outside git), holds thread info and
// indirectly referenced memory only (never full memory, so guest memory pages stay out), and its
// path is recorded in the report's fault_record ("minidump"). The writer loads dbghelp at run
// time and does nothing when it is missing. tools/minidump_stack.py prints the host stack.

#include <cstdint>
#include <string>

namespace playgta6::guest::minidump {

// "<report dir>/<scope>-fault-<kind>-<pid>.dmp"; kind is reduced to [A-Za-z0-9_-].
inline std::string select_path(const std::string& report_path, const char* scope, const char* kind,
                               uint32_t pid) {
    std::string dir;
    const size_t slash = report_path.find_last_of("/\\");
    if (slash != std::string::npos) dir = report_path.substr(0, slash + 1);
    std::string k;
    for (const char* p = kind != nullptr ? kind : ""; *p != 0; ++p) {
        const char c = *p;
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
        k += ok ? c : '_';
    }
    if (k.empty()) k = "unknown";
    return dir + (scope != nullptr ? scope : "host") + "-fault-" + k + "-" + std::to_string(pid) + ".dmp";
}

// Adds "minidump": "<path>" as the first member of the report's fault_record object. Returns the
// report unchanged when it has no fault_record object.
inline std::string annotate_report(const std::string& report, const std::string& path) {
    static const char kKey[] = "\"fault_record\": {";
    const size_t at = report.find(kKey);
    if (at == std::string::npos || path.empty()) return report;
    std::string esc;
    for (char c : path) {
        if (c == '\\' || c == '"') esc += '\\';
        esc += c;
    }
    const size_t after = at + sizeof(kKey) - 1;
    return report.substr(0, after) + "\"minidump\": \"" + esc + "\", " + report.substr(after);
}

}  // namespace playgta6::guest::minidump
