#include "os/thread_name.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <intrin.h>
#endif

#include <chrono>

namespace playgta6::os {

void name_this_thread(const wchar_t* name) {
#ifdef _WIN32
    SetThreadDescription(GetCurrentThread(), name);
#else
    (void)name;
#endif
}

#ifdef _WIN32
namespace {
// QueryThreadCycleTime counts TSC ticks: calibrated once against the steady clock (2 ms spin).
double tsc_ticks_per_ns() {
    static const double ratio = [] {
        const auto t0 = std::chrono::steady_clock::now();
        const uint64_t c0 = __rdtsc();
        auto t1 = t0;
        while (t1 - t0 < std::chrono::milliseconds(2)) t1 = std::chrono::steady_clock::now();
        const uint64_t c1 = __rdtsc();
        const double ns = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        return ns > 0 ? static_cast<double>(c1 - c0) / ns : 1.0;
    }();
    return ratio;
}
}  // namespace
#endif

uint64_t this_thread_cpu_ns() {
#ifdef _WIN32
    ULONG64 cycles = 0;
    if (!QueryThreadCycleTime(GetCurrentThread(), &cycles)) return 0;
    return static_cast<uint64_t>(static_cast<double>(cycles) / tsc_ticks_per_ns());
#else
    return 0;
#endif
}

uint64_t process_cpu_ns() {
#ifdef _WIN32
    FILETIME created{}, exited{}, kernel{}, user{};
    if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) return 0;
    auto v = [](const FILETIME& f) { return (static_cast<uint64_t>(f.dwHighDateTime) << 32) | f.dwLowDateTime; };
    return (v(kernel) + v(user)) * 100u;
#else
    return 0;
#endif
}

}  // namespace playgta6::os
