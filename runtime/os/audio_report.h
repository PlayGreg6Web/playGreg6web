#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// R40: the boot report's `audio` section. Totals come from the existing counters
// (audio_endpoint_stats, xapu_stats); this header adds the per-window bucketing of the frames the
// title released and the JSON writer. Windows are kWindowMs of wall time since the first render
// release; at most kMaxWindows are kept (later audio lands in the last window, `windows_clamped`).
namespace playgta6::os::audio_report {

constexpr uint64_t kWindowMs = 10000;
constexpr size_t kMaxWindows = 64;
// Frames of one spatial pass (10 ms at 48 kHz): passes count as frames in the fraction.
constexpr uint64_t kSpatialPassFrames = 480;

// Window index of an elapsed time, clamped to the last window.
inline size_t window_index(uint64_t elapsed_ms, bool* clamped = nullptr) {
    const uint64_t idx = elapsed_ms / kWindowMs;
    if (idx >= kMaxWindows) {
        if (clamped != nullptr) *clamped = true;
        return kMaxWindows - 1;
    }
    return static_cast<size_t>(idx);
}

struct Windows {
    std::array<std::atomic<uint64_t>, kMaxWindows> frames{};
    std::array<std::atomic<uint64_t>, kMaxWindows> nonsilent{};
    std::atomic<uint64_t> clamped{0};
    std::atomic<int64_t> start_ms{-1};

    void record_at(uint64_t elapsed_ms, uint64_t n_frames, bool non_silent) {
        bool c = false;
        const size_t i = window_index(elapsed_ms, &c);
        if (c) clamped.fetch_add(1);
        frames[i].fetch_add(n_frames);
        if (non_silent) nonsilent[i].fetch_add(n_frames);
    }
    // Wall-clock entry: the first call starts window 0.
    void record(uint64_t n_frames, bool non_silent) {
        const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now().time_since_epoch()).count();
        int64_t expected = -1;
        start_ms.compare_exchange_strong(expected, now);
        record_at(static_cast<uint64_t>(now - start_ms.load()), n_frames, non_silent);
    }
    void reset() {
        for (auto& a : frames) a = 0;
        for (auto& a : nonsilent) a = 0;
        clamped = 0;
        start_ms = -1;
    }
    // Non-silent frames per window up to the last window holding any released frame.
    std::vector<uint64_t> nonsilent_windows() const {
        size_t n = 0;
        for (size_t i = 0; i < kMaxWindows; ++i) {
            if (frames[i].load() != 0) n = i + 1;
        }
        std::vector<uint64_t> out(n);
        for (size_t i = 0; i < n; ++i) out[i] = nonsilent[i].load();
        return out;
    }
};

inline Windows& windows() {
    static Windows w;
    return w;
}

struct Totals {
    uint64_t render_frames = 0, render_nonzero_frames = 0, render_silent_releases = 0;
    uint64_t spatial_passes = 0, spatial_object_buffers = 0, spatial_nonzero_buffers = 0, spatial_nonzero_passes = 0;
    uint64_t xapu_decoded_packets = 0, xapu_failed_packets = 0, xapu_results_posted = 0;
    uint64_t xapu_commands = 0, xapu_unhandled_commands = 0;
    // Commands per type the model knows: 1 (start), 2 (free), 3 (Opus decode), 5.
    uint64_t xapu_type1 = 0, xapu_type2 = 0, xapu_type3 = 0, xapu_type5 = 0;
};

// Non-silent share of everything the title released: render-client frames plus spatial passes
// (kSpatialPassFrames each, non-silent when any object buffer holds a non-zero sample).
inline double non_silent_fraction(const Totals& t) {
    const uint64_t frames = t.render_frames + t.spatial_passes * kSpatialPassFrames;
    const uint64_t nonzero = t.render_nonzero_frames + t.spatial_nonzero_passes * kSpatialPassFrames;
    return frames == 0 ? 0.0 : static_cast<double>(nonzero) / static_cast<double>(frames);
}

// The `audio` value: totals, "non_silent_fraction" (0 with no frames), "window_ms" and
// "non_silent_frames_per_window".
inline std::string to_json(const Totals& t, const std::vector<uint64_t>& windows_nonsilent, uint64_t clamped) {
    const double fraction = non_silent_fraction(t);
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.6f", fraction > 1.0 ? 1.0 : fraction);
    std::string o = "{";
    auto add = [&o](const char* k, uint64_t v) { o += "\"" + std::string(k) + "\": " + std::to_string(v) + ", "; };
    add("render_frames", t.render_frames);
    add("render_nonzero_frames", t.render_nonzero_frames);
    add("render_silent_releases", t.render_silent_releases);
    o += "\"non_silent_fraction\": " + std::string(buf) + ", ";
    add("spatial_passes", t.spatial_passes);
    add("spatial_object_buffers", t.spatial_object_buffers);
    add("spatial_nonzero_buffers", t.spatial_nonzero_buffers);
    add("spatial_nonzero_passes", t.spatial_nonzero_passes);
    add("xapu_decoded_packets", t.xapu_decoded_packets);
    add("xapu_failed_packets", t.xapu_failed_packets);
    add("xapu_results_posted", t.xapu_results_posted);
    add("xapu_commands", t.xapu_commands);
    add("xapu_unhandled_commands", t.xapu_unhandled_commands);
    o += "\"xapu_commands_by_type\": {\"1\": " + std::to_string(t.xapu_type1) + ", \"2\": " +
         std::to_string(t.xapu_type2) + ", \"3\": " + std::to_string(t.xapu_type3) + ", \"5\": " +
         std::to_string(t.xapu_type5) + "}, ";
    add("window_ms", kWindowMs);
    add("windows_clamped", clamped);
    o += "\"non_silent_frames_per_window\": [";
    for (size_t i = 0; i < windows_nonsilent.size(); ++i) {
        o += (i == 0 ? "" : ", ") + std::to_string(windows_nonsilent[i]);
    }
    o += "]}";
    return o;
}

}  // namespace playgta6::os::audio_report
