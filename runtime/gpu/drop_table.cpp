#include "gpu/drop_table.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>

namespace playgta6::gpu::d3d12_guest {

DropTable& drop_table() {
    static DropTable t;
    return t;
}

namespace {
std::string hex16(uint64_t h) {
    char b[24];
    std::snprintf(b, sizeof b, "0x%016" PRIx64, h);
    return b;
}
uint64_t sum(const uint64_t* a) {
    uint64_t t = 0;
    for (size_t i = 0; i < DropTable::kReasons; ++i) {
        t += a[i];
    }
    return t;
}
const char* reason_name_idx(size_t i) { return DropTable::reason_name(static_cast<DropTable::Reason>(i)); }
std::string reasons_json(const DropTable::Counts& c) {
    std::string o = "{";
    bool first = true;
    for (size_t i = 0; i < DropTable::kReasons; ++i) {
        const uint64_t n = c.draws[i] + c.dispatches[i];
        if (n == 0) {
            continue;
        }
        o += std::string(first ? "" : ",") + "\"" + DropTable::reason_name(static_cast<DropTable::Reason>(i)) +
             "\":" + std::to_string(n);
        first = false;
    }
    return o + "}";
}
}  // namespace

std::vector<std::pair<uint64_t, DropTable::Counts>> DropTable::sorted() const {
    std::vector<std::pair<uint64_t, Counts>> v;
    {
        std::lock_guard<std::mutex> lock(mu_);
        v.assign(map_.begin(), map_.end());
    }
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
        const uint64_t ta = a.second.total(), tb = b.second.total();
        return ta != tb ? ta > tb : a.first < b.first;
    });
    return v;
}

std::string DropTable::by_program_json() const {
    std::string o = "{";
    bool first = true;
    for (const auto& e : sorted()) {
        o += std::string(first ? "" : ", ") + "\"" + hex16(e.first) + "\": " + std::to_string(e.second.total());
        first = false;
    }
    return o + "}";
}

std::string DropTable::detail_json() const {
    Counts other;
    std::string phases;
    {
        std::lock_guard<std::mutex> lock(mu_);
        other = other_;
        phases = phases_.json(&reason_name_idx);
    }
    std::string o = "{\"tracked\": " + std::to_string(tracked()) + ", \"max_hashes\": " + std::to_string(kMaxHashes) +
                    ", \"other\": {\"draws\": " + std::to_string(sum(other.draws)) +
                    ", \"dispatches\": " + std::to_string(sum(other.dispatches)) +
                    ", \"reasons\": " + reasons_json(other) + "}, \"entries\": [";
    bool first = true;
    for (const auto& e : sorted()) {
        o += std::string(first ? "" : ", ") + "{\"hash\": \"" + hex16(e.first) + "\", \"draws\": " +
             std::to_string(sum(e.second.draws)) + ", \"dispatches\": " + std::to_string(sum(e.second.dispatches)) +
             ", \"reasons\": " + reasons_json(e.second) + "}";
        first = false;
    }
    return o + "], \"phases\": " + phases + "}";
}

}  // namespace playgta6::gpu::d3d12_guest
