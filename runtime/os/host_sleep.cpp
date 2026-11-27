// Precise host sleeps (R40). Design: host_sleep.h.

#include "os/host_sleep.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <timeapi.h>
#ifdef _MSC_VER
#pragma comment(lib, "winmm.lib")
#endif
#endif

namespace playgta6::os::host_sleep {

namespace {

std::atomic<Mode> g_mode{Mode::OneMs};

constexpr int kBuckets = 8;  // overshoot < 0.5, 1, 2, 4, 8, 16, 32 ms, >= 32 ms
constexpr double kBucketMs[kBuckets - 1] = {0.5, 1, 2, 4, 8, 16, 32};
std::atomic<uint64_t> g_count{0}, g_req_us{0}, g_act_us{0}, g_max_act_us{0}, g_max_over_us{0};
std::atomic<uint64_t> g_hist[kBuckets];

#ifdef _WIN32
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

// One timer per thread (a waitable timer is waited on by the thread that set it); null when the
// OS has no high-resolution timers.
HANDLE thread_timer() {
    struct Timer {
        HANDLE handle = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                               TIMER_ALL_ACCESS);
        ~Timer() {
            if (handle != nullptr) {
                CloseHandle(handle);
            }
        }
    };
    thread_local Timer timer;
    return timer.handle;
}
#endif

}  // namespace

void set_mode(Mode m) { g_mode.store(m); }
Mode mode() { return g_mode.load(); }

void record_guest_sleep(std::chrono::steady_clock::duration requested, std::chrono::steady_clock::duration actual) {
    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    const int64_t req = duration_cast<microseconds>(requested).count();
    const int64_t act = duration_cast<microseconds>(actual).count();
    const int64_t over = act > req ? act - req : 0;
    int b = 0;
    while (b < kBuckets - 1 && over >= static_cast<int64_t>(kBucketMs[b] * 1000.0)) ++b;
    g_hist[b].fetch_add(1, std::memory_order_relaxed);
    g_count.fetch_add(1, std::memory_order_relaxed);
    g_req_us.fetch_add(static_cast<uint64_t>(req), std::memory_order_relaxed);
    g_act_us.fetch_add(static_cast<uint64_t>(act), std::memory_order_relaxed);
    for (auto* m : {&g_max_act_us, &g_max_over_us}) {
        const uint64_t v = static_cast<uint64_t>(m == &g_max_act_us ? act : over);
        uint64_t cur = m->load(std::memory_order_relaxed);
        while (v > cur && !m->compare_exchange_weak(cur, v, std::memory_order_relaxed)) {
        }
    }
}

void reset_guest_sleep_stats() {
    g_count = 0;
    g_req_us = 0;
    g_act_us = 0;
    g_max_act_us = 0;
    g_max_over_us = 0;
    for (auto& h : g_hist) h = 0;
}

std::string guest_sleep_stats_json() {
    const uint64_t n = g_count.load();
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "{\"timer_resolution\": \"%s\", \"count\": %llu, \"mean_requested_ms\": %.3f, "
                  "\"mean_actual_ms\": %.3f, \"max_actual_ms\": %.3f, \"max_overshoot_ms\": %.3f, "
                  "\"overshoot_ms_buckets\": {\"lt0.5\": %llu, \"lt1\": %llu, \"lt2\": %llu, \"lt4\": %llu, "
                  "\"lt8\": %llu, \"lt16\": %llu, \"lt32\": %llu, \"ge32\": %llu}}",
                  mode() == Mode::OneMs ? "1ms" : "default", static_cast<unsigned long long>(n),
                  n ? g_req_us.load() / 1000.0 / static_cast<double>(n) : 0.0,
                  n ? g_act_us.load() / 1000.0 / static_cast<double>(n) : 0.0, g_max_act_us.load() / 1000.0,
                  g_max_over_us.load() / 1000.0, static_cast<unsigned long long>(g_hist[0].load()),
                  static_cast<unsigned long long>(g_hist[1].load()), static_cast<unsigned long long>(g_hist[2].load()),
                  static_cast<unsigned long long>(g_hist[3].load()), static_cast<unsigned long long>(g_hist[4].load()),
                  static_cast<unsigned long long>(g_hist[5].load()), static_cast<unsigned long long>(g_hist[6].load()),
                  static_cast<unsigned long long>(g_hist[7].load()));
    return buf;
}

void raise_timer_resolution() {
    if (mode() == Mode::Default) return;
#ifdef _WIN32
    static const bool done = [] {
        timeBeginPeriod(1);  // 1 ms tick; the OS restores it at process exit
        return true;
    }();
    (void)done;
    // Windows 11: do not let a windowless / occluded process have its timer resolution ignored.
    PROCESS_POWER_THROTTLING_STATE st{};
    st.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    st.ControlMask = PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION;
    st.StateMask = 0;
    SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &st, sizeof(st));
#endif
}

void sleep_until(std::chrono::steady_clock::time_point deadline) {
#ifdef _WIN32
    if (mode() == Mode::OneMs)
    if (HANDLE timer = thread_timer()) {
        for (;;) {
            const auto left = deadline - std::chrono::steady_clock::now();
            if (left <= std::chrono::steady_clock::duration::zero()) {
                return;
            }
            LARGE_INTEGER due;
            // Relative due time in 100 ns units (negative), rounded up.
            due.QuadPart = -static_cast<LONGLONG>(
                (std::chrono::duration_cast<std::chrono::nanoseconds>(left).count() + 99) / 100);
            if (!SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE) ||
                WaitForSingleObject(timer, INFINITE) != WAIT_OBJECT_0) {
                break;  // fall back to the standard sleep
            }
        }
    }
#endif
    std::this_thread::sleep_until(deadline);
}

}  // namespace playgta6::os::host_sleep
