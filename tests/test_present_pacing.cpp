// Present pacing against a 60 Hz vblank clock (O36, gpu/present_pacing.h), on a fake clock.
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <cstdio>
#include <vector>

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

}  // namespace

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
    test_present_samples_ring();
    test_host_sleep_is_precise();
    test_timer_resolution_raised();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_present_pacing: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_present_pacing: all tests passed\n");
    return 0;
}
