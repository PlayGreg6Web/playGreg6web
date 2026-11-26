#pragma once

// Whole-run frame-time statistics with O(1) memory: the boot report's `frame_times`. The present
// samples ring keeps only the last 20000 presents and the `[live]` lines are 5 s averages, so
// neither shows stutter. Every present-to-present gap lands in a log-spaced histogram (1..512 ms,
// 4 buckets per octave) of the phase it belongs to (the draws/frame regime of that frame: loading
// and menus draw little, the world a lot), with the vblank intervals the frame overran relative to
// the title's requested sync interval, and 1% / 0.1% lows derived from the histogram.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

namespace playgta6::gpu {

class FrameTimeStats {
public:
    static constexpr int kBuckets = 38;  // [0]: < 1 ms, [1..36]: 1 ms * 2^((i-1)/4) .. 2^(i/4), [37]: >= 512 ms
    static constexpr double kVblankUs = 1e6 / 60.0;
    enum Phase : int { kAll = 0, kNoDraws, kLight, kMedium, kHeavy, kVeryHeavy, kPhaseCount };

    static const char* phase_name(int p) {
        static const char* const names[kPhaseCount] = {"all", "draws_0", "draws_1_9", "draws_10_99", "draws_100_999",
                                                       "draws_1000_plus"};
        return names[p];
    }
    static int phase_for_draws(uint64_t draws) {
        return draws == 0 ? kNoDraws : draws < 10 ? kLight : draws < 100 ? kMedium : draws < 1000 ? kHeavy : kVeryHeavy;
    }
    // Upper edge of bucket i in microseconds (the last bucket is open ended).
    static double bucket_upper_us(int i) {
        return i <= 0 ? 1000.0 : 1000.0 * std::exp2(static_cast<double>(i) / 4.0);
    }
    static int bucket_for_us(double us) {
        if (us < 1000.0) {
            return 0;
        }
        for (int i = 1; i < kBuckets - 1; ++i) {
            if (us < bucket_upper_us(i)) {
                return i;
            }
        }
        return kBuckets - 1;
    }

    struct Phase_ {
        uint64_t frames = 0;
        uint64_t sum_us = 0;
        uint64_t max_us = 0;
        uint64_t missed_vblanks = 0;  // vblank intervals over the requested one, paced frames only
        uint64_t late_frames = 0;     // frames that overran it by at least one
        uint64_t unpaced_frames = 0;  // interval 0 / unknown: no missed accounting
        uint64_t first_ms = 0;
        uint64_t last_ms = 0;
        std::array<uint64_t, kBuckets> buckets{};

        // The frame time at quantile q (0..1) from the histogram: the upper edge of the bucket the
        // quantile falls in, clamped to the observed max.
        double quantile_us(double q) const {
            if (frames == 0) {
                return 0;
            }
            const uint64_t rank = std::max<uint64_t>(1, static_cast<uint64_t>(std::ceil(q * static_cast<double>(frames))));
            uint64_t acc = 0;
            for (int i = 0; i < kBuckets; ++i) {
                acc += buckets[i];
                if (acc >= rank) {
                    return std::min(bucket_upper_us(i), static_cast<double>(max_us));
                }
            }
            return static_cast<double>(max_us);
        }
    };

    // One present-to-present gap of `dt_us` ending at `wall_ms`; `draws` issued during it; `gap_vblanks`
    // the title's requested minimum gap in vblanks (0: immediate/unknown).
    void add(uint64_t dt_us, uint64_t draws, uint32_t gap_vblanks, uint64_t wall_ms) {
        accumulate(phases_[kAll], dt_us, gap_vblanks, wall_ms);
        accumulate(phases_[phase_for_draws(draws)], dt_us, gap_vblanks, wall_ms);
    }
    const Phase_& phase(int p) const { return phases_[p]; }
    void clear() { phases_ = {}; }

    std::string to_json() const {
        std::string o = "{\"vblank_ms\": 16.667, \"bucket_edges_ms\": [";
        for (int i = 0; i < kBuckets - 1; ++i) {
            o += (i ? "," : "") + num(bucket_upper_us(i) / 1000.0);
        }
        o += "], \"phases\": [";
        bool first = true;
        for (int p = 0; p < kPhaseCount; ++p) {
            const Phase_& s = phases_[p];
            if (s.frames == 0) {
                continue;
            }
            const double p99 = s.quantile_us(0.99);
            const double p999 = s.quantile_us(0.999);
            o += first ? "" : ",";
            first = false;
            o += std::string("\n    {\"name\": \"") + phase_name(p) + "\", \"frames\": " + std::to_string(s.frames) +
                 ", \"first_ms\": " + std::to_string(s.first_ms) + ", \"last_ms\": " + std::to_string(s.last_ms) +
                 ", \"mean_ms\": " + num(static_cast<double>(s.sum_us) / 1000.0 / static_cast<double>(s.frames)) +
                 ", \"p50_ms\": " + num(s.quantile_us(0.5) / 1000.0) + ", \"p99_ms\": " + num(p99 / 1000.0) +
                 ", \"p999_ms\": " + num(p999 / 1000.0) + ", \"max_ms\": " + num(static_cast<double>(s.max_us) / 1000.0) +
                 ", \"low1_fps\": " + num(p99 > 0 ? 1e6 / p99 : 0) + ", \"low01_fps\": " + num(p999 > 0 ? 1e6 / p999 : 0) +
                 ", \"missed_vblanks\": " + std::to_string(s.missed_vblanks) +
                 ", \"late_frames\": " + std::to_string(s.late_frames) +
                 ", \"unpaced_frames\": " + std::to_string(s.unpaced_frames) + ", \"buckets\": [";
            for (int i = 0; i < kBuckets; ++i) {
                o += (i ? "," : "") + std::to_string(s.buckets[i]);
            }
            o += "]}";
        }
        o += first ? "]}" : "\n  ]}";
        return o;
    }

private:
    static std::string num(double v) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%.3f", v);
        return buf;
    }
    static void accumulate(Phase_& s, uint64_t dt_us, uint32_t gap_vblanks, uint64_t wall_ms) {
        if (s.frames == 0) {
            s.first_ms = wall_ms;
        }
        s.last_ms = wall_ms;
        ++s.frames;
        s.sum_us += dt_us;
        s.max_us = std::max(s.max_us, dt_us);
        ++s.buckets[bucket_for_us(static_cast<double>(dt_us))];
        if (gap_vblanks == 0) {
            ++s.unpaced_frames;
            return;
        }
        const double vblanks = std::floor(static_cast<double>(dt_us) / kVblankUs + 0.5);
        if (vblanks > gap_vblanks) {
            const uint64_t missed = static_cast<uint64_t>(vblanks) - gap_vblanks;
            s.missed_vblanks += missed;
            ++s.late_frames;
        }
    }

    std::array<Phase_, kPhaseCount> phases_{};
};

}  // namespace playgta6::gpu
