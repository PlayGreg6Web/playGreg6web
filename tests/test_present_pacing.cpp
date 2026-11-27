// Present pacing against a 60 Hz vblank clock (O36, gpu/present_pacing.h), on a fake clock.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

#include "gpu/frame_time_hist.h"
#include "gpu/present_pacing.h"
#include "gpu/present_samples.h"
#include "os/host_sleep.h"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

using playgta6::gpu::PresentPacer;
using Clock = PresentPacer::Clock;
constexpr auto kV = std::chrono::nanoseconds(PresentPacer::kVblankNs);

struct FakeClock {
    Clock::time_point origin = Clock::time_point{} + std::chrono::hours(1);
    Clock::time_point now = origin;
    std::vector<Clock::time_point> sleeps;
    void install(PresentPacer& p) {
        p.set_clock_for_testing([this] { return now; },
                                [this](Clock::time_point t) {
                                    sleeps.push_back(t);
                                    if (t > now) now = t;  // the wait ends at the vblank
                                },
                                origin);
    }
};

void test_vblank_clock() {
    PresentPacer p;
    FakeClock c;
    c.install(p);
    CHECK(p.vblank_at(c.origin) == 0);
    CHECK(p.vblank_at(c.origin + kV - std::chrono::nanoseconds(1)) == 0);
    CHECK(p.vblank_at(c.origin + 3 * kV) == 3);
    CHECK(p.vblank_start(60) == c.origin + 60 * kV);  // 60 vblanks: one second
    CHECK(p.vblank_at(c.origin + std::chrono::seconds(1)) == 59 || p.vblank_at(c.origin + std::chrono::seconds(1)) == 60);
}

void test_interval_one_paces_to_60hz() {
    PresentPacer p;
    FakeClock c;
    c.install(p);
    c.now = c.origin + kV / 2;  // mid vblank 0
    CHECK(p.present(1) == 1);   // the first present flips at the next vblank
    CHECK(c.now == c.origin + kV);
    // A title that presents as fast as it can is held to one present per vblank.
    for (uint64_t v = 2; v <= 61; ++v) {
        CHECK(p.present(1) == v);
    }
    CHECK(c.now == c.origin + 61 * kV);
    const auto s = p.stats();
    CHECK(s.presents == 61 && s.waits == 61 && s.missed_vblanks == 0 && s.immediate == 0);
}

void test_interval_n_and_missed_vblanks() {
    PresentPacer p;
    FakeClock c;
    c.install(p);
    CHECK(p.present(2) == 1);  // nothing earlier: the next vblank
    CHECK(p.present(2) == 3);  // two vblanks after the previous flip
    // The title is late: the vblank it aimed at (5) has passed; it flips at the next one and the
    // vblanks it was late by count as missed.
    c.now = c.origin + 7 * kV + kV / 3;
    CHECK(p.present(2) == 8);
    CHECK(p.stats().missed_vblanks == 3);  // aimed at 5, flipped at 8
}

void test_immediate_and_off() {
    PresentPacer p;
    FakeClock c;
    c.install(p);
    c.now = c.origin + 4 * kV;
    CHECK(p.present(0) == 4);  // interval 0: no wait
    CHECK(c.sleeps.empty() && p.stats().immediate == 1);
    CHECK(p.present(1) == 5);  // the next one counts from that flip
    p.set_enabled(false);
    const auto before = c.sleeps.size();
    for (int i = 0; i < 10; ++i) {
        p.present(1);
    }
    CHECK(c.sleeps.size() == before && p.stats().presents == 2);  // off: no waits, not counted
    CHECK(!p.enabled());
}

void test_title_driven_no_phase_beat_at_or_below_60hz() {
    PresentPacer p;
    FakeClock c;
    c.install(p);
    p.set_title_driven(true);
    CHECK(p.title_driven());

    // Start off-phase (+5 ms past vblank 0): title paces itself at 30 Hz (2 vblanks) then 60 Hz
    // (1 vblank). Zero waits and zero global vblank-grid quantization / phase beat.
    c.now = c.origin + std::chrono::milliseconds(5);
    CHECK(p.present(2) == 0);
    for (int i = 0; i < 5; ++i) {
        c.now += 2 * kV;  // 30 Hz, staying +5 ms off-phase
        (void)p.present(2);
    }
    for (int i = 0; i < 5; ++i) {
        c.now += kV;  // 60 Hz, still +5 ms off-phase
        (void)p.present(1);
    }
    const auto s = p.stats();
    CHECK(c.sleeps.empty());
    CHECK(s.presents == 11 && s.immediate == 11 && s.waits == 0 && s.wait_us == 0);
}

void test_title_driven_caps_unpaced_bursts_at_60hz() {
    PresentPacer p;
    FakeClock c;
    c.install(p);
    p.set_title_driven(true);

    const auto t0 = c.origin + std::chrono::milliseconds(5);
    c.now = t0;
    // First present is immediate.
    CHECK(p.present(1) == p.vblank_at(t0));
    CHECK(c.sleeps.empty() && p.stats().immediate == 1 && p.stats().waits == 0);

    // Back-to-back 0 ms burst: held to 1-vblank minimum gap from t0 (not quantized to vblank grid).
    CHECK(p.present(1) == p.vblank_at(t0 + kV));
    CHECK(c.now == t0 + kV);
    CHECK(c.sleeps.size() == 1 && c.sleeps[0] == t0 + kV);

    // Next present arrives 2 ms later with interval == 2: still capped at 1-vblank minimum gap (60 Hz).
    c.now += std::chrono::milliseconds(2);
    CHECK(p.present(2) == p.vblank_at(t0 + 2 * kV));
    CHECK(c.now == t0 + 2 * kV);
    CHECK(c.sleeps.size() == 2 && c.sleeps[1] == t0 + 2 * kV);

    // Rapid burst of 10 presents at 0 ms spacing advances clock by 10 vblanks (<= 60 Hz).
    const auto burst_start = c.now;
    for (int i = 0; i < 10; ++i) {
        (void)p.present(1);
    }
    CHECK(c.now == burst_start + 10 * kV);
    const auto s = p.stats();
    CHECK(s.presents == 13 && s.immediate == 1 && s.waits == 12 && s.wait_us > 0);
}

void test_title_driven_honor_interval() {
    CHECK(PresentPacer::title_gap_vblanks(2, false) == 1);
    CHECK(PresentPacer::title_gap_vblanks(2, true) == 2);
    CHECK(PresentPacer::title_gap_vblanks(1, true) == 1);
    CHECK(PresentPacer::title_gap_vblanks(0, true) == 1);
    PresentPacer p;
    FakeClock c;
    c.install(p);
    p.set_title_driven(true);
    CHECK(!p.honor_interval());
    p.set_honor_interval(true);
    CHECK(p.honor_interval());
    const auto t0 = c.origin + std::chrono::milliseconds(5);
    c.now = t0;
    (void)p.present(2);  // first present is immediate
    CHECK(c.sleeps.empty());
    // A burst at interval 2 is held to 2 vblanks apart (30 fps).
    (void)p.present(2);
    CHECK(c.now == t0 + 2 * kV);
    (void)p.present(2);
    CHECK(c.now == t0 + 4 * kV);
    // Interval 1 in honor mode still paces at 1 vblank.
    (void)p.present(1);
    CHECK(c.now == t0 + 5 * kV);
    // Interval 0 stays immediate.
    const auto before = c.sleeps.size();
    (void)p.present(0);
    CHECK(c.sleeps.size() == before);
    // Default (min1) at interval 2 is 1 vblank apart.
    PresentPacer q;
    FakeClock d;
    d.install(q);
    q.set_title_driven(true);
    d.now = d.origin + std::chrono::milliseconds(5);
    const auto s0 = d.now;
    (void)q.present(2);
    (void)q.present(2);
    CHECK(d.now == s0 + kV);
}

void test_title_driven_interval_zero_remains_immediate() {
    PresentPacer p;
    FakeClock c;
    c.install(p);
    p.set_title_driven(true);

    const auto t0 = c.origin + std::chrono::milliseconds(3);
    c.now = t0;
    CHECK(p.present(1) == p.vblank_at(t0));
    // Back-to-back interval 0 presents must never sleep, even within 1 vblank of a prior present.
    for (int i = 0; i < 5; ++i) {
        CHECK(p.present(0) == p.vblank_at(t0));
    }
    CHECK(c.now == t0);
    CHECK(c.sleeps.empty());
    const auto s = p.stats();
    CHECK(s.presents == 6 && s.immediate == 6 && s.waits == 0 && s.wait_us == 0);
}

void test_interval_clamp_and_overflow_safety() {
    CHECK(PresentPacer::title_gap_vblanks(30, true) == PresentPacer::kMaxInterval);
    CHECK(PresentPacer::title_gap_vblanks(UINT32_MAX, true) == PresentPacer::kMaxInterval);
    CHECK(PresentPacer::title_gap_vblanks(UINT32_MAX, false) == 1u);

    // 1. Vblank-grid mode: huge / malformed intervals clamp to kMaxInterval (100 ms) and recover
    // cleanly to 60 Hz on subsequent interval=1 presents.
    {
        PresentPacer p;
        FakeClock c;
        c.install(p);
        CHECK(p.vblank_start(std::numeric_limits<uint64_t>::max()) <= Clock::time_point::max());
        CHECK(p.vblank_start(std::numeric_limits<uint64_t>::max()) >= c.origin);

        c.now = c.origin + 5 * kV;
        CHECK(p.present(1) == 6);
        const auto before_huge = c.now;
        CHECK(p.present(30) == 6 + PresentPacer::kMaxInterval);
        CHECK(c.now - before_huge == PresentPacer::kMaxInterval * kV);
        CHECK(p.stats().last_interval == 30);

        const auto before_max = c.now;
        CHECK(p.present(std::numeric_limits<uint32_t>::max()) == 6 + 2 * PresentPacer::kMaxInterval);
        CHECK(c.now - before_max == PresentPacer::kMaxInterval * kV);

        // Next normal frame paces immediately at 1 vblank (60 Hz), with zero leftover sleep debt.
        const auto after_max = c.now;
        CHECK(p.present(1) == 6 + 2 * PresentPacer::kMaxInterval + 1);
        CHECK(c.now - after_max == kV);
    }

    // 2. Title-driven honor-interval mode: huge intervals clamp to kMaxInterval without overflow.
    {
        PresentPacer p;
        FakeClock c;
        c.install(p);
        p.set_title_driven(true);
        p.set_honor_interval(true);
        c.now = c.origin + std::chrono::milliseconds(5);
        (void)p.present(1);
        const auto t0 = c.now;
        (void)p.present(std::numeric_limits<uint32_t>::max());
        CHECK(c.now - t0 == PresentPacer::kMaxInterval * kV);
        const auto t1 = c.now;
        (void)p.present(1);
        CHECK(c.now - t1 == kV);
    }
}

void test_long_stall_and_oversleep_recovery() {
    // Title-driven mode: 500 ms and 250 s stalls + oversleep produce zero catch-up debt and
    // record actual oversleep duration in stats().wait_us.
    {
        PresentPacer p;
        Clock::time_point origin = Clock::time_point{} + std::chrono::hours(1);
        Clock::time_point now = origin;
        int64_t extra_oversleep_ms = 0;
        p.set_clock_for_testing([&] { return now; },
                                [&](Clock::time_point t) {
                                    if (t > now) now = t;
                                    now += std::chrono::milliseconds(extra_oversleep_ms);
                                },
                                origin);
        p.set_title_driven(true);
        (void)p.present(1);

        // Simulate a 250-second long-session stall (e.g. tab pause / shader compile): immediate!
        now += std::chrono::seconds(250);
        const auto s_before = p.stats();
        (void)p.present(1);
        CHECK(p.stats().immediate == s_before.immediate + 1);
        CHECK(p.stats().waits == s_before.waits);

        // Next frame arrives 1 ms later and oversleeps by 40 ms: wait_us reflects actual sleep
        // (~55.6 ms), and the following frame after the oversleep presents immediately (no debt).
        now += std::chrono::milliseconds(1);
        extra_oversleep_ms = 40;
        const auto wait_before = p.stats().wait_us;
        (void)p.present(1);
        CHECK(p.stats().wait_us - wait_before >= 50'000);
        extra_oversleep_ms = 0;
        const auto imm_before = p.stats().immediate;
        (void)p.present(1);
        CHECK(p.stats().immediate == imm_before + 1);
    }

    // Vblank-grid mode: 250 s stall jumps directly to current + 1 without multi-vblank sleep.
    {
        PresentPacer p;
        FakeClock c;
        c.install(p);
        CHECK(p.present(1) == 1);
        c.now += std::chrono::seconds(250) + kV / 2;
        const uint64_t cur = p.vblank_at(c.now);
        const auto t_before = c.now;
        CHECK(p.present(1) == cur + 1);
        CHECK(c.now - t_before <= kV);
    }
}

void test_early_wakeup_and_backward_clock_step() {
    // Emscripten Atomics.wait truncation can wake slightly early, or a clock can step backward.
    // Neither may stack future deadlines or wrap wait_us negative->UINT64_MAX.
    PresentPacer p;
    Clock::time_point origin = Clock::time_point{} + std::chrono::hours(1);
    Clock::time_point now = origin + std::chrono::seconds(10);
    std::vector<Clock::time_point> sleeps;
    bool wake_early = false;
    p.set_clock_for_testing([&] { return now; },
                            [&](Clock::time_point t) {
                                sleeps.push_back(t);
                                if (wake_early) {
                                    // Wake 2 ms before the deadline.
                                    now = t - std::chrono::milliseconds(2);
                                } else if (t > now) {
                                    now = t;
                                }
                            },
                            origin);
    p.set_title_driven(true);
    (void)p.present(1);

    // 10 consecutive early wakeups: every sleep lead stays strictly <= 1 vblank.
    wake_early = true;
    for (int i = 0; i < 10; ++i) {
        const auto before = now;
        (void)p.present(1);
        CHECK(sleeps.back() - before <= kV);
    }

    // Backward clock jump of 5 seconds: sleep lead still <= 1 vblank and wait_us never wraps.
    wake_early = false;
    now -= std::chrono::seconds(5);
    const auto before_jump = now;
    const auto wait_before = p.stats().wait_us;
    (void)p.present(1);
    CHECK(sleeps.back() - before_jump <= kV);
    CHECK(p.stats().wait_us >= wait_before && p.stats().wait_us - wait_before <= 20'000);
}

void test_concurrent_callers_and_mid_sleep_reset() {
    // Two concurrent threads calling present(1) must not stack deadlines to 2 * kV, and
    // reset_for_testing() mid-sleep must prevent waking threads from corrupting fresh stats.
    PresentPacer p;
    const Clock::time_point origin = Clock::time_point{} + std::chrono::hours(1);
    const Clock::time_point fixed_now = origin + std::chrono::milliseconds(5);
    std::mutex m;
    std::condition_variable cv;
    int sleeping = 0;
    bool release = false;
    std::vector<Clock::time_point> observed_deadlines;

    p.set_clock_for_testing([&] { return fixed_now; },
                            [&](Clock::time_point t) {
                                std::unique_lock<std::mutex> lk(m);
                                observed_deadlines.push_back(t);
                                ++sleeping;
                                cv.notify_all();
                                cv.wait(lk, [&] { return release; });
                            },
                            origin);
    p.set_title_driven(true);
    (void)p.present(1);  // anchor at fixed_now

    std::thread t1([&] { (void)p.present(1); });
    std::thread t2([&] { (void)p.present(1); });
    {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return sleeping == 2; });
        CHECK(observed_deadlines.size() == 2);
        for (const auto d : observed_deadlines) {
            CHECK(d - fixed_now <= kV);
        }
    }
    // Reset pacer while both threads are blocked in sleep_until.
    p.reset_for_testing();
    {
        std::lock_guard<std::mutex> lk(m);
        release = true;
    }
    cv.notify_all();
    t1.join();
    t2.join();

    // Waking threads belonged to the previous generation: fresh stats must stay clean.
    const auto s = p.stats();
    CHECK(s.presents == 0 && s.waits == 0 && s.wait_us == 0);
}

void test_mode_transitions_and_reset_hygiene() {
    PresentPacer p;
    FakeClock c;
    c.install(p);
    p.set_title_driven(true);
    p.set_honor_interval(true);

    const auto t0 = c.origin + std::chrono::milliseconds(5);
    c.now = t0;
    (void)p.present(1);
    // Repeated set_title_driven(true) (as Device::SetFrameIntervalX does) must NOT clear the
    // burst-capping anchor!
    p.set_title_driven(true);
    (void)p.present(1);
    CHECK(c.now == t0 + kV);

    // Leaving title_driven after 100 s must reset last_flip_ so grid mode has no bogus missed_vblanks.
    c.now += std::chrono::seconds(100);
    p.set_title_driven(false);
    const auto missed_before = p.stats().missed_vblanks;
    (void)p.present(1);
    CHECK(p.stats().missed_vblanks == missed_before);

    // reset_for_testing() must also clear honor_interval_.
    p.reset_for_testing();
    CHECK(!p.honor_interval());
    CHECK(!p.title_driven());
    CHECK(p.enabled());
}

}  // namespace

void test_frame_time_hist() {
    using playgta6::gpu::FrameTimeStats;
    CHECK(FrameTimeStats::bucket_for_us(500) == 0);
    CHECK(FrameTimeStats::bucket_for_us(1000) == 1);
    CHECK(FrameTimeStats::bucket_for_us(16667) > FrameTimeStats::bucket_for_us(8000));
    CHECK(FrameTimeStats::bucket_for_us(600000) == FrameTimeStats::kBuckets - 1);
    CHECK(FrameTimeStats::phase_for_draws(0) == FrameTimeStats::kNoDraws);
    CHECK(FrameTimeStats::phase_for_draws(5000) == FrameTimeStats::kVeryHeavy);
    FrameTimeStats h;
    for (int i = 0; i < 98; ++i) {
        h.add(16667, 200, 1, 1000 + i);  // on time at 60 fps
    }
    h.add(33333, 200, 1, 2000);   // one missed vblank
    h.add(100000, 200, 1, 2100);  // a hitch: 6 vblanks -> 5 missed
    h.add(16667, 0, 0, 2200);     // unpaced, other phase
    const auto& all = h.phase(FrameTimeStats::kAll);
    CHECK(all.frames == 101 && all.max_us == 100000 && all.first_ms == 1000 && all.last_ms == 2200);
    CHECK(all.late_frames == 2 && all.missed_vblanks == 6 && all.unpaced_frames == 1);
    CHECK(h.phase(FrameTimeStats::kHeavy).frames == 100 && h.phase(FrameTimeStats::kNoDraws).frames == 1);
    CHECK(all.quantile_us(0.5) < 20000 && all.quantile_us(0.5) >= 16667 * 0.8);
    CHECK(all.quantile_us(0.999) == 100000);  // clamped to the observed max
    CHECK(all.quantile_us(0.99) <= 40000);
    const std::string j = h.to_json();
    CHECK(j.find("\"name\": \"all\"") != std::string::npos && j.find("\"missed_vblanks\": 6") != std::string::npos);
    h.clear();
    CHECK(h.phase(FrameTimeStats::kAll).frames == 0 && h.to_json().find("\"phases\": []") != std::string::npos);
}

void test_present_samples_ring() {
    playgta6::gpu::PresentSampleRing r(3);
    CHECK(r.last().index == 0 && r.last().draws == 0);
    r.record(10, 5, 2);
    CHECK(r.last().draws == 5 && r.last().wall_ms == 10);
    r.record(20, 12, 2);
    auto v = r.snapshot();
    CHECK(v.size() == 2 && v[0].draws == 5 && v[0].resources == 2);
    CHECK(v[1].index == 1 && v[1].draws == 7 && v[1].resources == 0 && v[1].wall_ms == 20);
    r.record(30, 12, 4);
    r.record(40, 20, 4);
    r.record(50, 20, 4);  // bounded: the oldest two are gone
    v = r.snapshot();
    CHECK(r.total() == 5 && r.size() == 3 && v.size() == 3);
    CHECK(v[0].index == 2 && v[1].index == 3 && v[2].index == 4);
    CHECK(v[1].draws == 8 && v[2].draws == 0);
    CHECK(r.last().index == 4 && r.last().wall_ms == 50);  // the newest, also after wrapping
    r.record(60, 23, 4);
    CHECK(r.last().index == 5 && r.last().draws == 3);
    r.clear();
    CHECK(r.total() == 0 && r.snapshot().empty());
}

// R40: host sleeps wake near their deadline, not on the next system timer tick (15.6 ms by
// default on Windows), so a guest Sleep(1) poll and a paced present's vblank wait are not
// stretched by up to a tick. Median overshoot of 2 ms sleeps, generous for loaded runners.
void test_host_sleep_is_precise() {
    std::vector<int64_t> overshoot_us;
    for (int i = 0; i < 15; ++i) {
        const auto deadline = Clock::now() + std::chrono::milliseconds(2);
        playgta6::os::host_sleep::sleep_until(deadline);
        const auto late = Clock::now() - deadline;
        CHECK(late >= Clock::duration::zero());
        overshoot_us.push_back(std::chrono::duration_cast<std::chrono::microseconds>(late).count());
    }
    std::sort(overshoot_us.begin(), overshoot_us.end());
    const int64_t median = overshoot_us[overshoot_us.size() / 2];
    if (median >= 6000) {
        std::fprintf(stderr, "host sleep median overshoot %lld us\n", static_cast<long long>(median));
    }
    CHECK(median < 6000);
}

// A timed condition-variable wait (what the guest's WaitForSingleObject timeouts use) wakes near its
// timeout once the timer resolution is raised, not on the 15.6 ms tick. Median of 15 waits of 4 ms.
void test_timer_resolution_raised() {
    playgta6::os::host_sleep::raise_timer_resolution();
    std::mutex m;
    std::condition_variable cv;
    std::vector<int64_t> us;
    for (int i = 0; i < 15; ++i) {
        std::unique_lock<std::mutex> lock(m);
        const auto t0 = Clock::now();
        cv.wait_until(lock, t0 + std::chrono::milliseconds(4));
        us.push_back(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count());
    }
    std::sort(us.begin(), us.end());
    CHECK(us[us.size() / 2] < 10000);
}

// Guest Sleep latency stats: overshoot histogram buckets, counts and the --timer-resolution switch.
void test_guest_sleep_stats() {
    namespace hs = playgta6::os::host_sleep;
    using ms = std::chrono::microseconds;
    hs::reset_guest_sleep_stats();
    hs::record_guest_sleep(ms(1000), ms(1200));    // +0.2 ms -> lt0.5
    hs::record_guest_sleep(ms(1000), ms(15600));   // +14.6 ms -> lt16
    hs::record_guest_sleep(ms(1000), ms(900));     // early -> lt0.5
    const std::string j = hs::guest_sleep_stats_json();
    CHECK(j.find("\"count\": 3") != std::string::npos);
    CHECK(j.find("\"lt0.5\": 2") != std::string::npos);
    CHECK(j.find("\"lt16\": 1") != std::string::npos);
    CHECK(j.find("\"max_overshoot_ms\": 14.600") != std::string::npos);
    hs::set_mode(hs::Mode::Default);
    CHECK(hs::mode() == hs::Mode::Default);
    const auto t0 = Clock::now();
    hs::sleep_for(std::chrono::milliseconds(2));  // plain sleep path still waits
    CHECK(Clock::now() - t0 >= std::chrono::milliseconds(1));
    CHECK(hs::guest_sleep_stats_json().find("\"default\"") != std::string::npos);
    hs::set_mode(hs::Mode::OneMs);
    hs::reset_guest_sleep_stats();
}

int main() {
    std::printf("test_present_pacing\n");
    test_vblank_clock();
    test_interval_one_paces_to_60hz();
    test_interval_n_and_missed_vblanks();
    test_immediate_and_off();
    test_title_driven_no_phase_beat_at_or_below_60hz();
    test_title_driven_caps_unpaced_bursts_at_60hz();
    test_title_driven_honor_interval();
    test_title_driven_interval_zero_remains_immediate();
    test_interval_clamp_and_overflow_safety();
    test_long_stall_and_oversleep_recovery();
    test_early_wakeup_and_backward_clock_step();
    test_concurrent_callers_and_mid_sleep_reset();
    test_mode_transitions_and_reset_hygiene();
    test_present_samples_ring();
    test_frame_time_hist();
    test_host_sleep_is_precise();
    test_timer_resolution_raised();
    test_guest_sleep_stats();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_present_pacing: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_present_pacing: all tests passed\n");
    return 0;
}
