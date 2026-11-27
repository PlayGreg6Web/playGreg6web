// Frame-time breakdown (gpu/frame_breakdown.h) and per-thread spans (os/thread_spans.h).
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

#include "gpu/frame_breakdown.h"
#include "os/thread_spans.h"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                                \
    do {                                                                           \
        if (!(cond)) {                                                             \
            std::fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                          \
        }                                                                          \
    } while (0)

using playgta6::gpu::FrameBreakdown;
using playgta6::gpu::FrameSample;
using playgta6::gpu::FrameTimeStats;
using playgta6::os::SpanKind;
using playgta6::os::ThreadSpan;

uint64_t span_ns(SpanKind k) { return playgta6::os::this_thread_spans().ns[static_cast<size_t>(k)]; }

// Synthetic clocks: wall and thread CPU only move when a test advances them.
uint64_t g_wall_ns = 0;
uint64_t g_cpu_ns = 0;
std::chrono::steady_clock::time_point fake_now() {
    return std::chrono::steady_clock::time_point(std::chrono::nanoseconds(g_wall_ns));
}
uint64_t fake_cpu() { return g_cpu_ns; }
void install_fake_clock() {
    g_wall_ns = 1000000000u;
    g_cpu_ns = 0;
    playgta6::os::thread_spans_detail::hooks().now = &fake_now;
    playgta6::os::thread_spans_detail::hooks().cpu = &fake_cpu;
}
void busy_ms(uint64_t ms) {  // CPU work: both clocks advance
    g_wall_ns += ms * 1000000u;
    g_cpu_ns += ms * 1000000u;
}
void sleep_ms(uint64_t ms) { g_wall_ns += ms * 1000000u; }  // blocked: only wall advances

// An inner scope pauses the outer one: each kind gets only its own (self) time.
void test_nested_spans_charge_self_time() {
    {
        const ThreadSpan outer(SpanKind::D3dExecute);
        busy_ms(5);
        {
            const ThreadSpan inner(SpanKind::Wait);
            sleep_ms(20);
        }
        busy_ms(5);
    }
    CHECK(span_ns(SpanKind::Wait) == 20000000u);
    CHECK(span_ns(SpanKind::D3dExecute) == 10000000u);  // the 20 ms wait is not also charged to the outer scope
}

void test_inactive_span_and_open_scope() {
    const uint64_t before = span_ns(SpanKind::Wait);
    { const ThreadSpan none(SpanKind::Count); sleep_ms(3); }
    CHECK(span_ns(SpanKind::Wait) == before);
    const ThreadSpan open(SpanKind::Sleep);
    const uint64_t s0 = span_ns(SpanKind::Sleep);
    sleep_ms(2);
    CHECK(span_ns(SpanKind::Sleep) - s0 == 2000000u);  // a still-open scope's elapsed time is visible
}

// A CPU kind records the thread's CPU time inside it: busy work is CPU, a sleep inside it is not.
void test_cpu_kind_records_thread_cpu() {
    const auto before = playgta6::os::this_thread_spans();
    {
        const ThreadSpan s(SpanKind::D3dAppend);
        busy_ms(10);
        sleep_ms(20);
    }
    const auto after = playgta6::os::this_thread_spans();
    const size_t k = static_cast<size_t>(SpanKind::D3dAppend);
    CHECK(after.ns[k] - before.ns[k] == 30000000u);
    CHECK(after.cpu_ns[k] - before.cpu_ns[k] == 10000000u);
}

void test_spans_are_per_thread() {
    const uint64_t here = span_ns(SpanKind::Lock);
    std::thread([] {
        const ThreadSpan s(SpanKind::Lock);
        sleep_ms(2);
    }).join();
    CHECK(span_ns(SpanKind::Lock) == here);
}

FrameSample sample(uint64_t thread, uint64_t cpu_ms, uint64_t wait_ms, uint64_t d3d_ms) {
    FrameSample s;
    s.thread_key = thread;
    s.thread_cpu_ns = cpu_ms * 1000000u;
    s.spans.ns[static_cast<size_t>(SpanKind::Wait)] = wait_ms * 1000000u;
    s.spans.ns[static_cast<size_t>(SpanKind::D3dExecute)] = d3d_ms * 1000000u;
    s.spans.cpu_ns[static_cast<size_t>(SpanKind::D3dExecute)] = d3d_ms * 1000000u;
    s.process_cpu_ns = cpu_ms * 3000000u;
    return s;
}

void test_breakdown_deltas_by_phase() {
    FrameBreakdown b;
    b.add(33000000u, 5, sample(1, 0, 0, 0));    // baseline only
    b.add(33000000u, 5, sample(1, 10, 20, 4));  // draws_1_9: 10 cpu (4 d3d), 20 wait, 3 other
    b.add(17000000u, 50, sample(1, 25, 20, 6)); // draws_10_99: 15 cpu (2 d3d), 0 wait, 2 other
    const auto& light = b.phase(FrameTimeStats::kLight);
    CHECK(light.frames == 1);
    CHECK(light.thread_cpu_ns == 10000000u);
    CHECK(light.span_ns[static_cast<size_t>(SpanKind::Wait)] == 20000000u);
    CHECK(light.process_cpu_ns == 30000000u);
    const auto& medium = b.phase(FrameTimeStats::kMedium);
    CHECK(medium.frames == 1);
    CHECK(medium.thread_cpu_ns == 15000000u);
    CHECK(b.phase(FrameTimeStats::kAll).frames == 2);
    CHECK(b.phase(FrameTimeStats::kAll).wall_ns == 50000000u);
    const std::string j = b.to_json();
    CHECK(j.find("\"name\": \"draws_1_9\", \"frames\": 1, \"wall_ms\": 33.000, \"present_thread_cpu_ms\": 10.000, "
                 "\"guest_code_cpu_ms\": 6.000, \"d3d_cpu_ms\": 4.000, \"blocked_ms\": 20.000, "
                 "\"d3d_off_cpu_ms\": 0.000, \"off_cpu_other_ms\": 3.000") !=
          std::string::npos);
    CHECK(j.find("\"process_cpu_ms\": 30.000") != std::string::npos);
}

void test_thread_switch_rebases() {
    FrameBreakdown b;
    b.add(1000000u, 0, sample(1, 100, 0, 0));
    b.add(1000000u, 0, sample(2, 5, 0, 0));  // another thread: no delta across clocks
    b.add(1000000u, 0, sample(2, 6, 0, 0));
    CHECK(b.phase(FrameTimeStats::kNoDraws).frames == 1);
    CHECK(b.phase(FrameTimeStats::kNoDraws).thread_cpu_ns == 1000000u);
    CHECK(b.thread_switches() == 1);
}

void test_windows() {
    FrameBreakdown b;
    b.add(1000000u, 5, sample(1, 0, 0, 0), 100);
    b.add(1000000u, 5, sample(1, 1, 0, 0), 4900);
    b.add(1000000u, 50, sample(1, 3, 0, 0), 5100);
    b.add(1000000u, 50, sample(1, 6, 0, 0), 12000);
    CHECK(b.windows().size() == 3);
    CHECK(b.windows()[0].start_ms == 0 && b.windows()[0].all.frames == 1);
    CHECK(b.windows()[1].start_ms == 5000 && b.windows()[1].all.thread_cpu_ns == 2000000u);
    CHECK(b.windows()[2].start_ms == 10000 && b.windows()[2].all.thread_cpu_ns == 3000000u);
    CHECK(b.to_json().find("{\"name\": \"t5000\", \"frames\": 1") != std::string::npos);
}

}  // namespace

int main() {
    install_fake_clock();
    test_windows();
    test_nested_spans_charge_self_time();
    test_inactive_span_and_open_scope();
    test_cpu_kind_records_thread_cpu();
    test_spans_are_per_thread();
    test_breakdown_deltas_by_phase();
    test_thread_switch_rebases();
    if (g_failures == 0) std::printf("test_frame_breakdown: ok\n");
    return g_failures == 0 ? 0 : 1;
}
