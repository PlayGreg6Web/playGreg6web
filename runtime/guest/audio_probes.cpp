#include "guest/audio_probes.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>

namespace playgta6::guest {

void AudioProbes::configure(std::vector<AudioProbe> probes) {
    probes.erase(std::remove_if(probes.begin(), probes.end(), [](const AudioProbe& p) { return p.address == 0; }),
                 probes.end());
    std::stable_sort(probes.begin(), probes.end(),
                     [](const AudioProbe& a, const AudioProbe& b) { return a.address < b.address; });
    probes.erase(std::unique(probes.begin(), probes.end(),
                             [](const AudioProbe& a, const AudioProbe& b) { return a.address == b.address; }),
                 probes.end());
    probes_ = std::move(probes);
    count_ = probes_.size();
    calls_.reset(count_ == 0 ? nullptr : new std::atomic<uint64_t>[count_]());
    small_.reset(count_ == 0 ? nullptr : new std::atomic<uint64_t>[count_ * kArgs * kSmall]());
    min_ = count_ == 0 ? 0 : probes_.front().address;
    max_ = count_ == 0 ? 0 : probes_.back().address;
}

void AudioProbes::clear() {
    count_ = 0;
    min_ = max_ = 0;
    probes_.clear();
    calls_.reset();
    small_.reset();
}

long AudioProbes::hit(uint64_t target, const uint64_t args[kArgs], bool* first) noexcept {
    if (count_ == 0 || target < min_ || target > max_) {
        return -1;
    }
    const auto it = std::lower_bound(probes_.begin(), probes_.end(), target,
                                     [](const AudioProbe& p, uint64_t t) { return p.address < t; });
    if (it == probes_.end() || it->address != target) {
        return -1;
    }
    const auto i = static_cast<size_t>(it - probes_.begin());
    const uint64_t before = calls_[i].fetch_add(1, std::memory_order_relaxed);
    if (first != nullptr) {
        *first = before == 0;
    }
    for (size_t a = 0; a < kArgs; ++a) {
        if (args[a] < kSmall) {
            small_[(i * kArgs + a) * kSmall + static_cast<size_t>(args[a])].fetch_add(1, std::memory_order_relaxed);
        }
    }
    return static_cast<long>(i);
}

long AudioProbes::find(const std::string& name) const {
    for (size_t i = 0; i < probes_.size(); ++i) {
        if (probes_[i].name == name) {
            return static_cast<long>(i);
        }
    }
    return -1;
}

std::string AudioProbes::to_json() const {
    static const char* const kArgNames[kArgs] = {"rcx", "rdx", "r8", "r9"};
    std::string o = "{\"indirect_calls_only\": true, \"probes\": [";
    for (size_t i = 0; i < count_; ++i) {
        char head[160];
        std::snprintf(head, sizeof head, "%s{\"name\": \"%s\", \"address\": \"0x%" PRIx64 "\", \"calls\": %" PRIu64
                      ", \"small_args\": {", i == 0 ? "" : ", ", probes_[i].name.c_str(), probes_[i].address, calls(i));
        o += head;
        bool first_arg = true;
        for (size_t a = 0; a < kArgs; ++a) {
            std::string vals;
            for (size_t v = 0; v < kSmall; ++v) {
                const uint64_t n = small_value(i, a, v);
                if (n != 0) {
                    vals += (vals.empty() ? "" : ", ") + std::string("\"") + std::to_string(v) + "\": " + std::to_string(n);
                }
            }
            if (!vals.empty()) {
                o += std::string(first_arg ? "" : ", ") + "\"" + kArgNames[a] + "\": {" + vals + "}";
                first_arg = false;
            }
        }
        o += "}}";
    }
    return o + "]}";
}

}  // namespace playgta6::guest
