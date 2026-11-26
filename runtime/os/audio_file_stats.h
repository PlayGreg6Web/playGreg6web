#pragma once
// Backlog A4: counts the title's file opens whose path names audio (the audio archives that hold
// the music metadata and streams), so the boot report separates "the archive never opened" from
// "it opened and the music controller still never started". Counts only; a path is never stored,
// only a short FNV-1a hash of the first few failed audio paths.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>

namespace playgta6::os::audio_files {

inline bool path_names_audio(std::string_view path) {
    // ASCII case-insensitive search for "audio".
    constexpr std::string_view kKey = "audio";
    if (path.size() < kKey.size()) return false;
    for (size_t i = 0; i + kKey.size() <= path.size(); ++i) {
        size_t k = 0;
        while (k < kKey.size()) {
            char c = path[i + k];
            if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
            if (c != kKey[k]) break;
            ++k;
        }
        if (k == kKey.size()) return true;
    }
    return false;
}

inline uint64_t fnv1a(std::string_view s) {
    uint64_t h = 1469598103934665603ull;
    for (char c : s) {
        h ^= static_cast<unsigned char>(c);
        h *= 1099511628211ull;
    }
    return h;
}

struct Stats {
    static constexpr int kFailedHashes = 8;
    std::atomic<uint64_t> opens_total{0};
    std::atomic<uint64_t> audio_opens{0};
    std::atomic<uint64_t> audio_open_ok{0};
    std::atomic<uint64_t> audio_open_failed{0};
    std::atomic<uint64_t> failed_hash[kFailedHashes]{};
    std::atomic<uint32_t> failed_hash_count{0};

    void record(std::string_view path, bool ok) {
        opens_total.fetch_add(1, std::memory_order_relaxed);
        if (!path_names_audio(path)) return;
        audio_opens.fetch_add(1, std::memory_order_relaxed);
        if (ok) {
            audio_open_ok.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        audio_open_failed.fetch_add(1, std::memory_order_relaxed);
        const uint32_t slot = failed_hash_count.fetch_add(1, std::memory_order_relaxed);
        if (slot < kFailedHashes) failed_hash[slot].store(fnv1a(path), std::memory_order_relaxed);
    }

    void reset() {
        opens_total = 0;
        audio_opens = 0;
        audio_open_ok = 0;
        audio_open_failed = 0;
        failed_hash_count = 0;
        for (auto& h : failed_hash) h = 0;
    }

    std::string to_json() const {
        std::string o = "{\"opens_total\": " + std::to_string(opens_total.load()) +
                        ", \"audio_opens\": " + std::to_string(audio_opens.load()) +
                        ", \"audio_open_ok\": " + std::to_string(audio_open_ok.load()) +
                        ", \"audio_open_failed\": " + std::to_string(audio_open_failed.load()) +
                        ", \"failed_path_hashes\": [";
        const uint32_t n = failed_hash_count.load();
        for (uint32_t i = 0; i < n && i < static_cast<uint32_t>(kFailedHashes); ++i) {
            char b[24];
            std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(failed_hash[i].load()));
            o += (i == 0 ? "\"" : ", \"") + std::string(b) + "\"";
        }
        return o + "]}";
    }
};

inline Stats& stats() {
    static Stats s;
    return s;
}

}  // namespace playgta6::os::audio_files
