#include "gpu/binding_fallbacks.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>

namespace playgta6::gpu::d3d12_guest {

BindingFallbackTable& binding_fallback_table() {
    static BindingFallbackTable t;
    return t;
}

namespace {
std::string reasons_json(const BindingFallbackTable::Counts& c) {
    std::string o = "{";
    bool first = true;
    for (size_t i = 0; i < BindingFallbackTable::kReasons; ++i) {
        const uint64_t n = c.draws[i] + c.dispatches[i];
        if (n == 0) continue;
        o += std::string(first ? "" : ",") + "\"" + BindingFallbackTable::reason_name(i) + "\":" + std::to_string(n);
        first = false;
    }
    return o + "}";
}
}  // namespace

std::string BindingFallbackTable::detail_json() const {
    std::vector<std::pair<uint64_t, Counts>> v;
    Counts other;
    std::string phases;
    {
        std::lock_guard<std::mutex> lock(mu_);
        v.assign(map_.begin(), map_.end());
        other = other_;
        phases = phases_.json(&BindingFallbackTable::reason_name);
    }
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
        const uint64_t ta = a.second.total(), tb = b.second.total();
        return ta != tb ? ta > tb : a.first < b.first;
    });
    std::string o = "{\"tracked\": " + std::to_string(v.size()) + ", \"max_hashes\": " + std::to_string(kMaxHashes) +
                    ", \"other\": {\"draws\": " + std::to_string(other.sum_draws()) +
                    ", \"dispatches\": " + std::to_string(other.sum_dispatches()) +
                    ", \"reasons\": " + reasons_json(other) + "}, \"entries\": [";
    bool first = true;
    for (const auto& e : v) {
        char h[24];
        std::snprintf(h, sizeof h, "0x%016" PRIx64, e.first);
        o += std::string(first ? "" : ", ") + "{\"hash\": \"" + h + "\", \"draws\": " +
             std::to_string(e.second.sum_draws()) + ", \"dispatches\": " + std::to_string(e.second.sum_dispatches()) +
             ", \"reasons\": " + reasons_json(e.second) + "}";
        first = false;
    }
    return o + "], \"phases\": " + phases + "}";
}

}  // namespace playgta6::gpu::d3d12_guest
