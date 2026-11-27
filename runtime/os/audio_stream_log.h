#pragma once
// Music-start diagnostics (backlog A4): a bounded, timestamped log of audio stream events (XApu voice
// start/free/XMA enable commands, decoder-factory and stream-create probe entries, music metadata
// lookups) with the present index at the time of each, plus per-kind counters. Reported as the boot
// report's `audio_stream_log`. Values are small integers or hashes, never game text.

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace playgta6::os::audio_stream_log {

constexpr size_t kMaxEvents = 400;
// Events kept per kind (the counters keep counting past it) so one chatty kind cannot fill the log.
constexpr size_t kMaxPerKind = 40;

struct Event {
    uint64_t t_ms = 0;
    uint64_t present = 0;
    std::string kind;
    uint64_t a = 0, b = 0, c = 0;
};

class Log {
public:
    using Source = uint64_t (*)();
    // Wall-clock ms since boot start and the present index; unset: ms since the first event, present 0.
    void set_sources(Source elapsed_ms, Source present_index) {
        elapsed_ = elapsed_ms;
        present_ = present_index;
    }
    void record(const char* kind, uint64_t a = 0, uint64_t b = 0, uint64_t c = 0) {
        const uint64_t t = now_ms();
        const uint64_t p = present_ != nullptr ? present_() : 0;
        std::lock_guard<std::mutex> lock(mutex_);
        Kind& k = kinds_[kind];
        if (k.count++ == 0) {
            k.first_ms = t;
            k.first_present = p;
        }
        k.last_ms = t;
        k.last_present = p;
        if (k.kept < kMaxPerKind && events_.size() < kMaxEvents) {
            ++k.kept;
            events_.push_back(Event{t, p, kind, a, b, c});
        }
    }
    uint64_t count(const std::string& kind) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = kinds_.find(kind);
        return it == kinds_.end() ? 0 : it->second.count;
    }
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.clear();
        kinds_.clear();
        start_ms_ = -1;
    }
    // {"kinds": {kind: {count, first_ms, first_present, last_ms, last_present}}, "events": [...]}
    std::string to_json() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::string o = "{\"kinds\": {";
        char buf[320];
        bool first = true;
        for (const auto& [name, k] : kinds_) {
            std::snprintf(buf, sizeof buf, "%s\"%s\": {\"count\": %" PRIu64 ", \"first_ms\": %" PRIu64
                          ", \"first_present\": %" PRIu64 ", \"last_ms\": %" PRIu64 ", \"last_present\": %" PRIu64 "}",
                          first ? "" : ", ", name.c_str(), k.count, k.first_ms, k.first_present, k.last_ms,
                          k.last_present);
            o += buf;
            first = false;
        }
        o += "}, \"events\": [";
        for (size_t i = 0; i < events_.size(); ++i) {
            const Event& e = events_[i];
            std::snprintf(buf, sizeof buf, "%s{\"t_ms\": %" PRIu64 ", \"present\": %" PRIu64
                          ", \"kind\": \"%s\", \"a\": \"0x%" PRIx64 "\", \"b\": \"0x%" PRIx64 "\", \"c\": \"0x%" PRIx64
                          "\"}", i == 0 ? "" : ", ", e.t_ms, e.present, e.kind.c_str(), e.a, e.b, e.c);
            o += buf;
        }
        return o + "]}";
    }

private:
    struct Kind {
        uint64_t count = 0, first_ms = 0, first_present = 0, last_ms = 0, last_present = 0;
        size_t kept = 0;
    };
    uint64_t now_ms() {
        if (elapsed_ != nullptr) {
            return elapsed_();
        }
        const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now().time_since_epoch()).count();
        int64_t expected = -1;
        start_ms_.compare_exchange_strong(expected, now);
        return static_cast<uint64_t>(now - start_ms_.load());
    }

    mutable std::mutex mutex_;
    std::vector<Event> events_;
    std::map<std::string, Kind> kinds_;
    Source elapsed_ = nullptr;
    Source present_ = nullptr;
    std::atomic<int64_t> start_ms_{-1};
};

inline Log& log() {
    static Log l;
    return l;
}

}  // namespace playgta6::os::audio_stream_log
