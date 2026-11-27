// Names the calling host thread (visible in debuggers, ETW and per-thread CPU tools). Diagnostic
// only: nothing the title can observe depends on it.
#pragma once

#include <cstdint>

namespace playgta6::os {

void name_this_thread(const wchar_t* name);
// CPU time of the calling thread (cycle-accurate, QueryThreadCycleTime) and of the whole process
// (GetProcessTimes), in ns; 0 where the host has no such counter.
uint64_t this_thread_cpu_ns();
uint64_t process_cpu_ns();

}  // namespace playgta6::os
