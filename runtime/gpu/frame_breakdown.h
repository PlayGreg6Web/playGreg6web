#pragma once

// Where a frame's wall time goes, per frame-time phase (the boot report's `frame_breakdown`, next to
// `frame_times`): at every present the present thread hands in cumulative counters (its CPU time,
// its ThreadSpan totals, the process CPU time, the live executor's busy time and the bytes
// submitted); the deltas since the previous present are summed into that frame's phase (draws per
// frame, FrameTimeStats::phase_for_draws) and reported as per-frame means. A present from a
// different thread than the previous one only re-bases (its deltas would mix two threads' clocks).
//
// Read it as: wall = present-thread CPU (guest code + our D3D12 layer's CPU) + blocked spans + the
// off-CPU part of the D3D12 spans (`d3d_off_cpu_ms`: blocking inside our layer no span names) + the
// rest (`off_cpu_other_ms`: preempted, or blocked somewhere no span covers, such as file reads). The
// process CPU and the executor time are whole-process / executor-thread totals over the same gap.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "gpu/frame_time_hist.h"
#include "os/thread_spans.h"

namespace playgta6::gpu {

struct FrameSample {
    uint64_t thread_key = 0;       // the presenting host thread
    uint64_t thread_cpu_ns = 0;    // that thread's CPU time
    os::ThreadSpanTotals spans;    // that thread's span totals
    uint64_t process_cpu_ns = 0;   // all threads of the process
    uint64_t executor_ns = 0;      // live executor: time inside execute() on its thread
    uint64_t submitted_bytes = 0;  // GPU packet bytes handed to the executor
    uint64_t texture_uploads = 0;  // D3D12 layer texture uploads
    uint64_t resources_created = 0;
    uint64_t shadow_compared_bytes = 0;  // CPU-visible buffer flush: bytes compared with the shadow
    uint64_t shadow_full_bytes = 0;      // ... of them in whole-buffer comparisons (no write tracking)
};

class FrameBreakdown {
public:
    static constexpr int kSpans = os::kSpanKinds;

    struct Phase {
        uint64_t frames = 0;
        uint64_t wall_ns = 0;
        uint64_t thread_cpu_ns = 0;
        std::array<uint64_t, kSpans> span_ns{};
        std::array<uint64_t, kSpans> span_cpu_ns{};  // CPU kinds
        uint64_t process_cpu_ns = 0;
        uint64_t executor_ns = 0;
        uint64_t submitted_bytes = 0;
        uint64_t texture_uploads = 0;
        uint64_t resources_created = 0;
        uint64_t shadow_compared_bytes = 0;
        uint64_t shadow_full_bytes = 0;
    };

    // One present ending a gap of `wall_ns` with `draws` draws; `s` holds the cumulative counters now.
    // `end_ms` (the boot clock at this present) also files the frame into its 5 s window.
    void add(uint64_t wall_ns, uint64_t draws, const FrameSample& s, uint64_t end_ms = 0) {
        if (have_prev_ && prev_.thread_key == s.thread_key) {
            Phase d;
            d.frames = 1;
            d.wall_ns = wall_ns;
            d.thread_cpu_ns = sub(s.thread_cpu_ns, prev_.thread_cpu_ns);
            for (int i = 0; i < kSpans; ++i) {
                d.span_ns[i] = sub(s.spans.ns[i], prev_.spans.ns[i]);
                d.span_cpu_ns[i] = sub(s.spans.cpu_ns[i], prev_.spans.cpu_ns[i]);
            }
            d.process_cpu_ns = sub(s.process_cpu_ns, prev_.process_cpu_ns);
            d.executor_ns = sub(s.executor_ns, prev_.executor_ns);
            d.submitted_bytes = sub(s.submitted_bytes, prev_.submitted_bytes);
            d.texture_uploads = sub(s.texture_uploads, prev_.texture_uploads);
            d.resources_created = sub(s.resources_created, prev_.resources_created);
            d.shadow_compared_bytes = sub(s.shadow_compared_bytes, prev_.shadow_compared_bytes);
            d.shadow_full_bytes = sub(s.shadow_full_bytes, prev_.shadow_full_bytes);
            accumulate(phases_[FrameTimeStats::kAll], d);
            accumulate(phases_[FrameTimeStats::phase_for_draws(draws)], d);
            add_window(end_ms, d);
        } else if (have_prev_) {
            ++thread_switches_;
        }
        prev_ = s;
        have_prev_ = true;
    }
    const Phase& phase(int p) const { return phases_[p]; }
    uint64_t thread_switches() const { return thread_switches_; }
    void clear() { *this = FrameBreakdown(); }

    // Per-frame means in ms (bytes in KiB), phases with frames only, then the 5 s windows (`all` only).
    std::string to_json() const {
        std::string o = "{\"unit\": \"mean per frame\", \"present_thread_switches\": " +
                        std::to_string(thread_switches_) + ", \"phases\": [";
        bool first = true;
        for (int p = 0; p < FrameTimeStats::kPhaseCount; ++p) {
            if (phases_[p].frames == 0) continue;
            o += (first ? "\n    " : ",\n    ") + phase_json(FrameTimeStats::phase_name(p), phases_[p]);
            first = false;
        }
        o += first ? "]" : "\n  ]";
        o += ", \"window_ms\": " + std::to_string(kWindowMs) + ", \"windows_dropped\": " + std::to_string(windows_dropped_) +
             ", \"windows\": [";
        for (size_t i = 0; i < windows_.size(); ++i) {
            o += (i ? ",\n    " : "\n    ") + phase_json("t" + std::to_string(windows_[i].start_ms), windows_[i].all);
        }
        o += windows_.empty() ? "]}" : "\n  ]}";
        return o;
    }

    static std::string phase_json(const std::string& name, const Phase& s) {
        const double n = static_cast<double>(s.frames);
        auto ms = [&](uint64_t ns) { return num(static_cast<double>(ns) / 1e6 / n); };
        uint64_t blocked = 0, d3d_cpu = 0, d3d_off = 0;
        for (int i = 0; i < kSpans; ++i) {
            if (os::span_kind_is_cpu(static_cast<os::SpanKind>(i))) {
                d3d_cpu += s.span_cpu_ns[i];
                d3d_off += sub(s.span_ns[i], s.span_cpu_ns[i]);
            } else {
                blocked += s.span_ns[i];
            }
        }
        const uint64_t accounted = s.thread_cpu_ns + blocked + d3d_off;
        std::string o = std::string("{\"name\": \"") + name +
             "\", \"frames\": " + std::to_string(s.frames) + ", \"wall_ms\": " + ms(s.wall_ns) +
             ", \"present_thread_cpu_ms\": " + ms(s.thread_cpu_ns) +
             ", \"guest_code_cpu_ms\": " + ms(sub(s.thread_cpu_ns, d3d_cpu)) + ", \"d3d_cpu_ms\": " + ms(d3d_cpu) +
             ", \"blocked_ms\": " + ms(blocked) + ", \"d3d_off_cpu_ms\": " + ms(d3d_off) +
             ", \"off_cpu_other_ms\": " + ms(sub(s.wall_ns, accounted)) + ", \"spans_ms\": {";
        for (int i = 0; i < kSpans; ++i) {
            o += (i ? ", \"" : "\"") + std::string(os::span_kind_name(static_cast<os::SpanKind>(i))) + "\": " +
                 ms(s.span_ns[i]);
        }
        o += "}, \"spans_cpu_ms\": {";
        bool first_cpu = true;
        for (int i = 0; i < kSpans; ++i) {
            if (!os::span_kind_is_cpu(static_cast<os::SpanKind>(i))) continue;
            o += (first_cpu ? "\"" : ", \"") + std::string(os::span_kind_name(static_cast<os::SpanKind>(i))) +
                 "\": " + ms(s.span_cpu_ns[i]);
            first_cpu = false;
        }
        o += "}, \"process_cpu_ms\": " + ms(s.process_cpu_ns) + ", \"executor_ms\": " + ms(s.executor_ns) +
             ", \"submitted_kib\": " + num(static_cast<double>(s.submitted_bytes) / 1024.0 / n) +
             ", \"texture_uploads\": " + num(static_cast<double>(s.texture_uploads) / n) +
             ", \"resources_created\": " + num(static_cast<double>(s.resources_created) / n) +
             ", \"shadow_compared_mib\": " + num(static_cast<double>(s.shadow_compared_bytes) / 1048576.0 / n) +
             ", \"shadow_full_mib\": " + num(static_cast<double>(s.shadow_full_bytes) / 1048576.0 / n) + "}";
        return o;
    }

private:
    static uint64_t sub(uint64_t a, uint64_t b) { return a > b ? a - b : 0; }
    static std::string num(double v) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.3f", v);
        return buf;
    }
    static void accumulate(Phase& p, const Phase& d) {
        p.frames += d.frames;
        p.wall_ns += d.wall_ns;
        p.thread_cpu_ns += d.thread_cpu_ns;
        for (int i = 0; i < kSpans; ++i) {
            p.span_ns[i] += d.span_ns[i];
            p.span_cpu_ns[i] += d.span_cpu_ns[i];
        }
        p.process_cpu_ns += d.process_cpu_ns;
        p.executor_ns += d.executor_ns;
        p.submitted_bytes += d.submitted_bytes;
        p.texture_uploads += d.texture_uploads;
        p.resources_created += d.resources_created;
        p.shadow_compared_bytes += d.shadow_compared_bytes;
        p.shadow_full_bytes += d.shadow_full_bytes;
    }

    void add_window(uint64_t end_ms, const Phase& d) {
        const uint64_t start = end_ms / kWindowMs * kWindowMs;
        if (windows_.empty() || windows_.back().start_ms != start) {
            if (windows_.size() >= kMaxWindows) {
                ++windows_dropped_;
                return;
            }
            windows_.push_back({start, {}});
        }
        accumulate(windows_.back().all, d);
    }

public:
    static constexpr uint64_t kWindowMs = 5000;
    static constexpr size_t kMaxWindows = 120;  // 10 minutes
    struct Window {
        uint64_t start_ms = 0;
        Phase all;
    };
    const std::vector<Window>& windows() const { return windows_; }

private:
    std::array<Phase, FrameTimeStats::kPhaseCount> phases_{};
    std::vector<Window> windows_;
    uint64_t windows_dropped_ = 0;
    FrameSample prev_;
    bool have_prev_ = false;
    uint64_t thread_switches_ = 0;
};

}  // namespace playgta6::gpu
