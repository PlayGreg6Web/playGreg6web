// Present pacing against a 60 Hz vblank clock (O36, gpu/present_pacing.h), on a fake clock.
#include <chrono>
#include <cstdio>
#include <vector>

#include "gpu/present_pacing.h"
#include "gpu/present_samples.h"

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

}  // namespace

void test_present_samples_ring() {
    playgta6::gpu::PresentSampleRing r(3);
    r.record(10, 5, 2);
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
    r.clear();
    CHECK(r.total() == 0 && r.snapshot().empty());
}

int main() {
    std::printf("test_present_pacing\n");
    test_vblank_clock();
    test_interval_one_paces_to_60hz();
    test_interval_n_and_missed_vblanks();
    test_immediate_and_off();
    test_present_samples_ring();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_present_pacing: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_present_pacing: all tests passed\n");
    return 0;
}
