// Process, thread, system information, time, debug and lifecycle part of the remaining Win32 core
// surface (task O15). See os/win32_compat.h for the overview.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "os/host_sleep.h"
#include "os/os_shim.h"
#include "os/win32_compat.h"
#include "os/win32_compat_internal.h"
#include "os/win32_loader.h"
#include "os/win32_sync.h"

namespace playgta6::os::win32 {

namespace {

// ---- process exit / configuration state ---------------------------------------------------

struct ProcessState {
    std::mutex mutex;
    ProcessExitHandler exit_handler = nullptr;
    void* exit_user = nullptr;
    bool exit_requested = false;
    uint32_t exit_code = 0;
    std::string command_line = "game.exe";
    uint64_t total_physical = 12ULL << 30;
    bool process_priority_boost_disabled = false;
};

ProcessState& process_state() {
    static ProcessState state;
    return state;
}

std::atomic<uint64_t> g_unhandled_filter{0};

thread_local uint64_t t_stack_low = 0;
thread_local uint64_t t_stack_high = 0;
thread_local int32_t t_current_priority = 0;

struct PriorityTable {
    std::mutex mutex;
    std::unordered_map<Handle, int32_t> by_handle;
};

PriorityTable& priorities() {
    static PriorityTable table;
    return table;
}

bool is_thread_handle(Handle h) {
    return h == kCurrentThreadPseudoHandle || lookup_dispatcher_object(h) != nullptr;
}

uint32_t processor_count() {
    const unsigned hw = std::thread::hardware_concurrency();
    uint64_t mask = get_process_affinity_mask();
    uint32_t bits = 0;
    while (mask != 0) {
        bits += static_cast<uint32_t>(mask & 1u);
        mask >>= 1;
    }
    uint32_t n = std::min<uint32_t>(hw == 0 ? 1u : hw, bits == 0 ? 1u : bits);
    return std::max<uint32_t>(1u, std::min<uint32_t>(n, 64u));
}

uint64_t active_mask(uint32_t n) {
    return n >= 64 ? ~uint64_t{0} : ((uint64_t{1} << n) - 1);
}

void fill_system_info(Pg6SystemInfo* info) {
    std::memset(info, 0, sizeof(*info));
    const uint32_t n = processor_count();
    info->processor_architecture = 9;  // PROCESSOR_ARCHITECTURE_AMD64
    info->page_size = 4096;
    info->minimum_application_address = 0x10000;
    info->maximum_application_address = 0x00007FFFFFFEFFFFULL;
    info->active_processor_mask = active_mask(n);
    info->number_of_processors = n;
    info->processor_type = 8664;  // PROCESSOR_AMD_X8664
    info->allocation_granularity = 0x10000;
    info->processor_level = 23;         // Zen 2 family (0x17), as in the console SoC
    info->processor_revision = 0x3100;  // model 0x31, stepping 0
}

// ---- app lifecycle notifications ---------------------------------------------------------

struct AppRegistration {
    AppNotificationKind kind;
    uint64_t routine;
    uint64_t context;
};

struct AppRegistry {
    std::mutex mutex;
    std::map<uint64_t, AppRegistration> entries;  // token order = registration order
    uint64_t next = 1;
};

AppRegistry& app_registry() {
    static AppRegistry r;
    return r;
}

uint32_t register_app_notification(AppNotificationKind kind,
                                   uint64_t routine,
                                   uint64_t context,
                                   uint64_t* registration) {
    if (registration != nullptr) {
        *registration = 0;
    }
    if (routine == 0 || registration == nullptr) {
        return ERROR_INVALID_PARAMETER;
    }
    auto& r = app_registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    const uint64_t token = 0x4150500000000000ULL + (r.next++ << 4);
    r.entries.emplace(token, AppRegistration{kind, routine, context});
    *registration = token;
    return ERROR_SUCCESS;
}

void unregister_app_notification(AppNotificationKind kind, uint64_t registration) {
    auto& r = app_registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    const auto it = r.entries.find(registration);
    if (it != r.entries.end() && it->second.kind == kind) {
        r.entries.erase(it);
    }
}

// Calls every live registration of `kind` (registration order) with `flag`. The list is copied
// first so a callback may unregister itself or others without deadlocking.
size_t invoke_app_notifications(AppNotificationKind kind,
                                uint8_t flag,
                                AppNotificationInvoker invoker,
                                void* user) {
    std::vector<AppRegistration> targets;
    {
        auto& r = app_registry();
        std::lock_guard<std::mutex> lock(r.mutex);
        for (const auto& [token, entry] : r.entries) {
            if (entry.kind == kind) {
                targets.push_back(entry);
            }
        }
    }
    for (const AppRegistration& t : targets) {
        invoker(AppNotification{t.kind, t.routine, flag, t.context}, user);
    }
    return targets.size();
}

// ---- civil date helpers -------------------------------------------------------------------

constexpr int64_t kDaysFrom1601To1970 = 134774;

int64_t days_from_civil(int64_t y, int64_t m, int64_t d) {
    y -= m <= 2;
    const int64_t era = (y >= 0 ? y : y - 399) / 400;
    const int64_t yoe = y - era * 400;
    const int64_t doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const int64_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + doe - 719468;  // days since 1970-01-01
}

void civil_from_days(int64_t z, int64_t* y, int64_t* m, int64_t* d) {
    z += 719468;
    const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const int64_t doe = z - era * 146097;
    const int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const int64_t mp = (5 * doy + 2) / 153;
    *d = doy - (153 * mp + 2) / 5 + 1;
    *m = mp < 10 ? mp + 3 : mp - 9;
    *y = yoe + era * 400 + (*m <= 2);
}

bool is_leap(int y) { return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0; }

}  // namespace

// Configuration / hooks ------------------------------------------------------------------------

void set_process_exit_handler(ProcessExitHandler handler, void* user) {
    auto& s = process_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.exit_handler = handler;
    s.exit_user = user;
}

bool process_exit_requested(uint32_t* exit_code) {
    auto& s = process_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (exit_code != nullptr) {
        *exit_code = s.exit_code;
    }
    return s.exit_requested;
}

void reset_process_exit_state() {
    auto& s = process_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.exit_requested = false;
    s.exit_code = 0;
}

void set_command_line(std::string_view utf8) {
    auto& s = process_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.command_line.assign(utf8);
}

void set_current_thread_stack_limits(uint64_t low, uint64_t high) {
    t_stack_low = low;
    t_stack_high = high;
}

void set_total_physical_memory(uint64_t bytes) {
    auto& s = process_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.total_physical = bytes;
}

uint64_t get_unhandled_exception_filter() {
    return g_unhandled_filter.load();
}

size_t app_notification_registration_count() {
    auto& r = app_registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    return r.entries.size();
}

size_t app_notification_registration_count(AppNotificationKind kind) {
    auto& r = app_registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    return static_cast<size_t>(std::count_if(r.entries.begin(), r.entries.end(),
                                             [kind](const auto& e) { return e.second.kind == kind; }));
}

// GDK PLM order. Suspend: the constrained-change routines run first (constrained = TRUE), then the
// app-state routines (quiesced = TRUE), whose handler suspends and must find the title already
// constrained. Resume mirrors it: app-state (quiesced = FALSE), then constrained (FALSE).
size_t dispatch_app_suspend(AppNotificationInvoker invoker, void* user) {
    if (invoker == nullptr) {
        return 0;
    }
    size_t n = invoke_app_notifications(AppNotificationKind::Constrained, 1, invoker, user);
    n += invoke_app_notifications(AppNotificationKind::State, 1, invoker, user);
    return n;
}

size_t dispatch_app_resume(AppNotificationInvoker invoker, void* user) {
    if (invoker == nullptr) {
        return 0;
    }
    size_t n = invoke_app_notifications(AppNotificationKind::State, 0, invoker, user);
    n += invoke_app_notifications(AppNotificationKind::Constrained, 0, invoker, user);
    return n;
}

namespace {
struct RelaunchState {
    std::mutex mutex;
    RelaunchPrepareHook hook = nullptr;
    void* hook_user = nullptr;
    bool requested = false;
    std::string args;
};
RelaunchState& relaunch_state() {
    static RelaunchState s;
    return s;
}
}  // namespace

void set_relaunch_prepare_hook(RelaunchPrepareHook hook, void* user) {
    auto& s = relaunch_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.hook = hook;
    s.hook_user = user;
}

void begin_guest_relaunch(std::string_view args) {
    auto& s = relaunch_state();
    RelaunchPrepareHook hook = nullptr;
    void* user = nullptr;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        s.requested = true;
        s.args.assign(args);
        hook = s.hook;
        user = s.hook_user;
    }
    if (hook != nullptr) {
        hook(user);
    }
}

bool relaunch_requested(std::string* args_out) {
    auto& s = relaunch_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.requested && args_out != nullptr) {
        *args_out = s.args;
    }
    return s.requested;
}

void reset_relaunch_request() {
    auto& s = relaunch_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.requested = false;
    s.args.clear();
}

size_t run_relaunch_suspend_sequence(AppNotificationInvoker invoker, void* invoker_user,
                                     RelaunchGraceFn grace, void* grace_user, uint32_t grace_ms) {
    const size_t n = dispatch_app_suspend(invoker, invoker_user);
    if (grace != nullptr) {
        grace(grace_ms, grace_user);
    }
    return n;
}

namespace compat_detail {

bool ticks_to_system_time(uint64_t ticks, Pg6SystemTime* out) {
    if (out == nullptr || ticks > 0x7FFFFFFFFFFFFFFFULL) {
        return false;
    }
    constexpr uint64_t kTicksPerMs = 10000ULL;
    constexpr uint64_t kMsPerDay = 86400000ULL;
    const uint64_t total_ms = ticks / kTicksPerMs;
    const int64_t days = static_cast<int64_t>(total_ms / kMsPerDay);
    const uint64_t ms_of_day = total_ms % kMsPerDay;
    int64_t y = 0, m = 0, d = 0;
    civil_from_days(days - kDaysFrom1601To1970, &y, &m, &d);
    if (y > 30827) {
        return false;
    }
    out->year = static_cast<uint16_t>(y);
    out->month = static_cast<uint16_t>(m);
    out->day = static_cast<uint16_t>(d);
    out->day_of_week = static_cast<uint16_t>((days + 1) % 7);  // 1601-01-01 was a Monday
    out->hour = static_cast<uint16_t>(ms_of_day / 3600000ULL);
    out->minute = static_cast<uint16_t>((ms_of_day / 60000ULL) % 60);
    out->second = static_cast<uint16_t>((ms_of_day / 1000ULL) % 60);
    out->milliseconds = static_cast<uint16_t>(ms_of_day % 1000ULL);
    return true;
}

bool system_time_to_ticks(const Pg6SystemTime& st, uint64_t* out) {
    static const int kDays[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (st.year < 1601 || st.year > 30827 || st.month < 1 || st.month > 12 || st.day < 1 ||
        st.hour > 23 || st.minute > 59 || st.second > 59 || st.milliseconds > 999) {
        return false;
    }
    const int dim = kDays[st.month - 1] + (st.month == 2 && is_leap(st.year) ? 1 : 0);
    if (st.day > dim) {
        return false;
    }
    const int64_t days = days_from_civil(st.year, st.month, st.day) + kDaysFrom1601To1970;
    const uint64_t secs = static_cast<uint64_t>(days) * 86400ULL + st.hour * 3600ULL +
                          st.minute * 60ULL + st.second;
    const uint64_t ticks = (secs * 1000ULL + st.milliseconds) * 10000ULL;
    if (ticks > 0x7FFFFFFFFFFFFFFFULL) {
        return false;
    }
    if (out != nullptr) {
        *out = ticks;
    }
    return true;
}

}  // namespace compat_detail

}  // namespace playgta6::os::win32

namespace win32 = playgta6::os::win32;
namespace cd = playgta6::os::win32::compat_detail;

extern "C" {

// ----- process / thread -------------------------------------------------------------------

Pg6Handle pg6_win32_get_current_process(void) {
    return win32::kCurrentProcessPseudoHandle;
}

uint32_t pg6_win32_get_current_process_id(void) {
    return win32::kCurrentProcessId;
}

uint32_t pg6_win32_get_process_id(Pg6Handle process) {
    if (process != win32::kCurrentProcessPseudoHandle) {
        win32::SetLastError(win32::ERROR_INVALID_HANDLE);
        return 0;
    }
    return win32::kCurrentProcessId;
}

void pg6_win32_get_startup_info_w(Pg6StartupInfoW* startup_info) {
    if (startup_info == nullptr) {
        return;
    }
    std::memset(startup_info, 0, sizeof(*startup_info));
    startup_info->cb = sizeof(Pg6StartupInfoW);
}

int32_t pg6_win32_get_thread_priority(Pg6Handle thread) {
    if (!win32::is_thread_handle(thread)) {
        win32::SetLastError(win32::ERROR_INVALID_HANDLE);
        return static_cast<int32_t>(win32::THREAD_PRIORITY_ERROR_RETURN);
    }
    win32::SetLastError(win32::ERROR_SUCCESS);
    if (thread == win32::kCurrentThreadPseudoHandle) {
        return win32::t_current_priority;
    }
    auto& t = win32::priorities();
    std::lock_guard<std::mutex> lock(t.mutex);
    auto it = t.by_handle.find(thread);
    return it == t.by_handle.end() ? 0 : it->second;
}

int32_t pg6_win32_set_thread_priority(Pg6Handle thread, int32_t priority) {
    if (!win32::is_thread_handle(thread)) {
        win32::SetLastError(win32::ERROR_INVALID_HANDLE);
        return 0;
    }
    if (priority < -15 || priority > 15) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    // Browsers cannot change thread priority: the value is recorded and has no scheduling effect.
    if (thread == win32::kCurrentThreadPseudoHandle) {
        win32::t_current_priority = priority;
    } else {
        auto& t = win32::priorities();
        std::lock_guard<std::mutex> lock(t.mutex);
        t.by_handle[thread] = priority;
    }
    win32::SetLastError(win32::ERROR_SUCCESS);
    return 1;
}

Pg6Handle pg6_win32_open_thread(uint32_t, int32_t, uint32_t thread_id) {
    // Only the calling thread can be named: other threads have no id -> handle index. The
    // pseudo-handle is returned (CloseHandle on it fails like on any pseudo-handle).
    if (thread_id != 0 && thread_id == win32::GetCurrentThreadId()) {
        win32::SetLastError(win32::ERROR_SUCCESS);
        return win32::kCurrentThreadPseudoHandle;
    }
    win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
    return win32::kNullHandle;
}

uint32_t pg6_win32_suspend_thread(Pg6Handle thread) {
    // A browser pthread cannot be suspended from outside: fail honestly.
    win32::SetLastError(win32::is_thread_handle(thread) ? win32::ERROR_NOT_SUPPORTED
                                                        : win32::ERROR_INVALID_HANDLE);
    return 0xFFFFFFFFu;
}

int32_t pg6_win32_terminate_process(Pg6Handle process, uint32_t exit_code) {
    if (process != win32::kCurrentProcessPseudoHandle) {
        win32::SetLastError(win32::ERROR_INVALID_HANDLE);
        return 0;
    }
    win32::ProcessExitHandler handler = nullptr;
    void* user = nullptr;
    {
        auto& s = win32::process_state();
        std::lock_guard<std::mutex> lock(s.mutex);
        s.exit_requested = true;
        s.exit_code = exit_code;
        handler = s.exit_handler;
        user = s.exit_user;
    }
    playgta6::os::log("TerminateProcess(current) requested, exit code " + std::to_string(exit_code));
    if (handler != nullptr) {
        handler(exit_code, user);
    }
    win32::SetLastError(win32::ERROR_SUCCESS);
    return 1;
}

int32_t pg6_win32_set_process_priority_boost(Pg6Handle process, int32_t disable) {
    if (process != win32::kCurrentProcessPseudoHandle) {
        win32::SetLastError(win32::ERROR_INVALID_HANDLE);
        return 0;
    }
    auto& s = win32::process_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.process_priority_boost_disabled = disable != 0;
    return 1;
}

void pg6_win32_get_current_thread_stack_limits(uint64_t* low_limit, uint64_t* high_limit) {
    uint64_t low = win32::t_stack_low;
    uint64_t high = win32::t_stack_high;
    if (low == 0 && high == 0) {
        low = 0x10000;
        high = 0x0000FFFFFFFFF000ULL;
    }
    if (low_limit != nullptr) {
        *low_limit = low;
    }
    if (high_limit != nullptr) {
        *high_limit = high;
    }
}

int32_t pg6_win32_is_processor_feature_present(uint32_t feature) {
    // Zen 2 (console SoC): SSE..SSE4.2, AVX, AVX2, RDRAND, no AVX-512, no ARM features.
    switch (feature) {
        case 2:   // PF_COMPARE_EXCHANGE_DOUBLE
        case 3:   // MMX
        case 6:   // SSE
        case 8:   // RDTSC
        case 9:   // PAE
        case 10:  // SSE2
        case 12:  // NX
        case 13:  // SSE3
        case 14:  // CMPXCHG16B
        case 17:  // XSAVE
        case 20:  // second level address translation
        case 21:  // virtualization firmware enabled
        case 22:  // RDFSBASE etc.
        case 23:  // __fastfail
        case 28:  // RDRAND
        case 36:  // SSSE3
        case 37:  // SSE4.1
        case 38:  // SSE4.2
        case 39:  // AVX
        case 40:  // AVX2
            return 1;
        default:
            return 0;
    }
}

int32_t pg6_win32_get_thread_group_affinity(Pg6Handle thread, Pg6GroupAffinity* out) {
    if (out == nullptr) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    std::memset(out, 0, sizeof(*out));
    if (!win32::is_thread_handle(thread)) {
        win32::SetLastError(win32::ERROR_INVALID_HANDLE);
        return 0;
    }
    out->mask = win32::active_mask(win32::processor_count()) & win32::get_process_affinity_mask();
    out->group = 0;
    return 1;
}

int32_t pg6_win32_set_thread_group_affinity(Pg6Handle thread,
                                            const Pg6GroupAffinity* group_affinity,
                                            Pg6GroupAffinity* previous) {
    if (previous != nullptr) {
        std::memset(previous, 0, sizeof(*previous));
    }
    if (group_affinity == nullptr) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    if (!win32::is_thread_handle(thread)) {
        win32::SetLastError(win32::ERROR_INVALID_HANDLE);
        return 0;
    }
    if (group_affinity->group != 0 || group_affinity->mask == 0) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    const uint64_t old = win32::SetThreadAffinityMask(thread, group_affinity->mask);
    if (old == 0) {
        return 0;  // SetThreadAffinityMask set the error (mask outside the process mask)
    }
    if (previous != nullptr) {
        previous->mask = old;
    }
    return 1;
}

int32_t pg6_win32_set_thread_ideal_processor_ex(Pg6Handle thread,
                                                const Pg6ProcessorNumber* ideal,
                                                Pg6ProcessorNumber* previous) {
    if (!win32::is_thread_handle(thread)) {
        win32::SetLastError(win32::ERROR_INVALID_HANDLE);
        return 0;
    }
    if (ideal == nullptr || ideal->group != 0 || ideal->number >= win32::processor_count()) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    static std::mutex mutex;
    // Keyed by handle; the current-thread pseudo handle by the calling thread's id.
    static std::map<uint64_t, uint8_t> ideals;
    const uint64_t raw = static_cast<uint64_t>(thread);
    const uint64_t id = raw == static_cast<uint64_t>(-2) ? (uint64_t{1} << 63) | pg6_win32_get_current_thread_id() : raw;
    std::lock_guard<std::mutex> lock(mutex);
    uint8_t& slot = ideals[id];
    if (previous != nullptr) {
        *previous = Pg6ProcessorNumber{0, slot, 0};
    }
    slot = ideal->number;
    return 1;
}

char* pg6_win32_get_command_line_a(void) {
    // Stable storage that outlives later set_command_line calls (previous strings are kept).
    static std::mutex mutex;
    static std::map<std::string, std::string> cache;
    std::string current;
    {
        auto& s = win32::process_state();
        std::lock_guard<std::mutex> lock(s.mutex);
        current = s.command_line;
    }
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(current);
    if (it == cache.end()) {
        it = cache.emplace(current, current).first;
    }
    return it->second.data();
}

void pg6_win32_sleep(uint32_t milliseconds) {
    if (milliseconds == 0) {
        std::this_thread::yield();
        return;
    }
    if (milliseconds == 0xFFFFFFFFu) {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::hours(1));
        }
    }
    // Precise (os/host_sleep.h): titles poll with Sleep(1), which the host timer tick stretched to
    // ~16 ms per poll.
    playgta6::os::host_sleep::sleep_for(std::chrono::milliseconds(milliseconds));
}

int32_t pg6_win32_initialize_critical_section_and_spin_count(Pg6CriticalSection* critical_section,
                                                             uint32_t spin_count) {
    if (critical_section == nullptr) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    win32::InitializeCriticalSectionWithSpinCount(critical_section, spin_count);
    return 1;
}

Pg6Handle pg6_win32_create_semaphore_ex_w(void* security_attributes,
                                          int32_t initial_count,
                                          int32_t maximum_count,
                                          const char16_t* name,
                                          uint32_t flags,
                                          uint32_t) {
    if (flags != 0) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);  // dwFlags is reserved
        return win32::kNullHandle;
    }
    return win32::CreateSemaphoreW(security_attributes, initial_count, maximum_count, name);
}

uint64_t pg6_win32_virtual_query_ex(Pg6Handle process,
                                    uint64_t address,
                                    Pg6MemoryBasicInformation* buffer,
                                    uint64_t length) {
    if (process != win32::kCurrentProcessPseudoHandle) {
        win32::SetLastError(win32::ERROR_INVALID_HANDLE);
        return 0;
    }
    return pg6_win32_virtual_query(address, buffer, static_cast<size_t>(length));
}

uint64_t pg6_win32_local_free(uint64_t memory) {
    if (memory == 0) {
        return 0;
    }
    // LocalAlloc memory is process-heap memory, as on Windows.
    if (win32::HeapFree(win32::GetProcessHeap(), 0, memory)) {
        return 0;
    }
    return memory;
}

void pg6_win32_initialize_s_list_head(void* list_head) {
    if (list_head != nullptr) {
        std::memset(list_head, 0, 16);  // SLIST_HEADER is 16 bytes
    }
}

// ----- sysinfo / time ---------------------------------------------------------------------

void pg6_win32_get_system_info(Pg6SystemInfo* system_info) {
    if (system_info != nullptr) {
        win32::fill_system_info(system_info);
    }
}

void pg6_win32_get_native_system_info(Pg6SystemInfo* system_info) {
    pg6_win32_get_system_info(system_info);
}

void pg6_win32_get_system_time(Pg6SystemTime* system_time) {
    if (system_time == nullptr) {
        return;
    }
    if (!cd::ticks_to_system_time(cd::now_ticks(), system_time)) {
        std::memset(system_time, 0, sizeof(*system_time));
    }
}

void pg6_win32_get_local_time(Pg6SystemTime* system_time) {
    pg6_win32_get_system_time(system_time);  // the time zone is UTC
}

void pg6_win32_get_system_time_precise_as_file_time(Pg6FileTime* out) {
    win32::GetSystemTimeAsFileTime(out);
}

int32_t pg6_win32_global_memory_status_ex(Pg6MemoryStatusEx* buffer) {
    if (buffer == nullptr || buffer->length != sizeof(Pg6MemoryStatusEx)) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    uint64_t total = 0;
    {
        auto& s = win32::process_state();
        std::lock_guard<std::mutex> lock(s.mutex);
        total = s.total_physical;
    }
    uint64_t used = 0;
    if (playgta6::os::VirtualMemoryArena* arena = win32::get_vmem_arena()) {
        used = arena->committed_bytes();
    }
    const uint64_t avail = used >= total ? 0 : total - used;
    constexpr uint64_t kVirtualTotal = 0x00007FFFFFFE0000ULL;
    buffer->memory_load = total == 0 ? 0 : static_cast<uint32_t>(std::min<uint64_t>(100, used * 100 / total));
    buffer->total_phys = total;
    buffer->avail_phys = avail;
    buffer->total_page_file = total;
    buffer->avail_page_file = avail;
    buffer->total_virtual = kVirtualTotal;
    buffer->avail_virtual = kVirtualTotal - std::min<uint64_t>(used, kVirtualTotal);
    buffer->avail_extended_virtual = 0;
    win32::SetLastError(win32::ERROR_SUCCESS);
    return 1;
}

int32_t pg6_win32_get_logical_processor_information(void* buffer, uint32_t* returned_length) {
    if (returned_length == nullptr) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    const uint32_t n = win32::processor_count();
    const uint32_t smt = n >= 2 ? 1u : 0u;
    const uint32_t cores = smt ? (n + 1) / 2 : 1;
    const uint32_t l3_groups = (cores + 3) / 4;
    const uint32_t entries = cores * 4 + l3_groups + 2;
    const uint32_t needed = entries * static_cast<uint32_t>(sizeof(Pg6LogicalProcessorInformation));
    if (buffer == nullptr || *returned_length < needed) {
        *returned_length = needed;
        win32::SetLastError(win32::ERROR_INSUFFICIENT_BUFFER);
        return 0;
    }
    auto* out = static_cast<Pg6LogicalProcessorInformation*>(buffer);
    uint32_t k = 0;
    auto cache = [](uint32_t level, uint32_t size_bytes, uint32_t type) {
        return (static_cast<uint64_t>(level)) | (uint64_t{8} << 8) | (uint64_t{64} << 16) |
               (static_cast<uint64_t>(size_bytes) << 32) | 0 * type;
    };
    const uint64_t all = win32::active_mask(n);
    for (uint32_t c = 0; c < cores; ++c) {
        const uint64_t core_mask = (smt ? uint64_t{3} << (2 * c) : uint64_t{1} << c) & all;
        auto put = [&](uint32_t relationship, uint64_t d0, uint64_t d1) {
            Pg6LogicalProcessorInformation& e = out[k++];
            std::memset(&e, 0, sizeof(e));
            e.processor_mask = core_mask;
            e.relationship = relationship;
            e.data[0] = d0;
            e.data[1] = d1;
        };
        put(0, smt, 0);                           // RelationProcessorCore, SMT flag
        put(2, cache(1, 32 * 1024, 2), 2);        // L1 data
        put(2, cache(1, 32 * 1024, 1), 1);        // L1 instruction
        put(2, cache(2, 512 * 1024, 0), 0);       // L2 unified
    }
    for (uint32_t g = 0; g < l3_groups; ++g) {
        uint64_t m = 0;
        for (uint32_t c = g * 4; c < std::min(cores, g * 4 + 4); ++c) {
            m |= (smt ? uint64_t{3} << (2 * c) : uint64_t{1} << c) & all;
        }
        Pg6LogicalProcessorInformation& e = out[k++];
        std::memset(&e, 0, sizeof(e));
        e.processor_mask = m;
        e.relationship = 2;
        e.data[0] = cache(3, 4 * 1024 * 1024, 0);
        e.data[1] = 0;
    }
    for (uint32_t relationship : {1u, 3u}) {  // NUMA node 0, processor package
        Pg6LogicalProcessorInformation& e = out[k++];
        std::memset(&e, 0, sizeof(e));
        e.processor_mask = all;
        e.relationship = relationship;
    }
    *returned_length = needed;
    win32::SetLastError(win32::ERROR_SUCCESS);
    return 1;
}

int32_t pg6_win32_file_time_to_system_time(const Pg6FileTime* file_time, Pg6SystemTime* system_time) {
    if (system_time != nullptr) {
        std::memset(system_time, 0, sizeof(*system_time));
    }
    if (file_time == nullptr || system_time == nullptr ||
        !cd::ticks_to_system_time(win32::file_time_ticks(*file_time), system_time)) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    return 1;
}

int32_t pg6_win32_system_time_to_file_time(const Pg6SystemTime* system_time, Pg6FileTime* file_time) {
    if (file_time != nullptr) {
        *file_time = Pg6FileTime{};
    }
    uint64_t ticks = 0;
    if (system_time == nullptr || file_time == nullptr ||
        !cd::system_time_to_ticks(*system_time, &ticks)) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    *file_time = win32::file_time_from_ticks(ticks);
    return 1;
}

uint32_t pg6_win32_get_time_zone_information(Pg6TimeZoneInformation* tz) {
    if (tz == nullptr) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0xFFFFFFFFu;  // TIME_ZONE_ID_INVALID
    }
    std::memset(tz, 0, sizeof(*tz));
    const char16_t name[] = u"Coordinated Universal Time";
    std::memcpy(tz->standard_name, name, sizeof(name));
    std::memcpy(tz->daylight_name, name, sizeof(name));
    return 0;  // TIME_ZONE_ID_UNKNOWN: UTC, no daylight saving
}

// ----- debug / error handling / lifecycle ---------------------------------------------------

int32_t pg6_win32_is_debugger_present(void) {
    return 0;
}

void pg6_win32_output_debug_string_a(const char* output_string) {
    if (output_string == nullptr) {
        return;
    }
    std::string_view text(output_string);
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
        text.remove_suffix(1);
    }
    playgta6::os::log(std::string("OutputDebugString: ") + std::string(text));
}

uint64_t pg6_win32_set_unhandled_exception_filter(uint64_t filter) {
    return win32::g_unhandled_filter.exchange(filter);
}

int32_t pg6_win32_unhandled_exception_filter(void*) {
    // A registered guest filter cannot be called from here (task R11 dispatches exceptions); the
    // documented result without a usable filter is EXCEPTION_EXECUTE_HANDLER.
    return 1;
}

uint32_t pg6_win32_register_app_state_change_notification(uint64_t routine,
                                                          uint64_t context,
                                                          uint64_t* registration) {
    return win32::register_app_notification(win32::AppNotificationKind::State, routine, context, registration);
}

void pg6_win32_unregister_app_state_change_notification(uint64_t registration) {
    win32::unregister_app_notification(win32::AppNotificationKind::State, registration);
}

uint32_t pg6_win32_register_app_constrained_change_notification(uint64_t routine,
                                                                uint64_t context,
                                                                uint64_t* registration) {
    return win32::register_app_notification(win32::AppNotificationKind::Constrained, routine, context,
                                            registration);
}

void pg6_win32_unregister_app_constrained_change_notification(uint64_t registration) {
    win32::unregister_app_notification(win32::AppNotificationKind::Constrained, registration);
}

int32_t pg6_win32_wer_register_file(const char16_t* file, uint32_t file_type, uint32_t flags) {
    constexpr int32_t kEInvalidArg = static_cast<int32_t>(0x80070057u);
    if (file == nullptr || file[0] == 0 || file_type < 1 || file_type > 2 || (flags & ~3u) != 0) {
        return kEInvalidArg;
    }
    return 0;  // accepted; no crash report is ever produced
}

int32_t pg6_win32_set_named_pipe_handle_state(Pg6Handle, uint32_t*, uint32_t*, uint32_t*) {
    win32::SetLastError(win32::ERROR_INVALID_HANDLE);  // named pipes cannot exist here
    return 0;
}

int32_t pg6_win32_wait_named_pipe_w(const char16_t* named_pipe_name, uint32_t) {
    win32::SetLastError(named_pipe_name == nullptr ? win32::ERROR_INVALID_PARAMETER
                                                   : win32::ERROR_FILE_NOT_FOUND);
    return 0;
}

}  // extern "C"
