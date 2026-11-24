// Tests for the O26 Win32 shims (os/win32_rtl*.cpp, guest/guest_rtl.cpp): critical section init
// with flags, fiber-local storage (incl. destructors through the guest-call path), InitOnce,
// pointer encoding, SLIST, GetModuleHandleExW, locale/string shims, file shims, thread-pool work
// with guest callbacks and RtlCaptureStackBackTrace.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "abi/playgta6_abi.h"
#include "guest/com_bridge.h"
#include "guest/guest_callbacks.h"
#include "guest/guest_crt_exit.h"
#include "guest/guest_exceptions.h"
#include "guest/guest_rtl.h"
#include "guest/guest_runtime.h"
#include "os/vmem.h"
#include "os/win32_compat.h"
#include "os/win32_kernel.h"
#include "os/win32_loader.h"
#include "os/win32_rtl.h"
#include "os/win32_sync.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_win32_rtl.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6;
using namespace playgta6::guest;
namespace win32 = playgta6::os::win32;
namespace com = playgta6::guest::com;
using playgta6::os::PageProtect;
using playgta6::os::VirtualMemoryArena;

template <typename Pred>
bool wait_for(Pred pred, int timeout_ms = 10000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (!pred()) {
        if (std::chrono::steady_clock::now() > deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

uint32_t last_error() {
    return win32::GetLastError();
}

// ---------------------------------------------------------------------------
// Guest functions ("lifted" test code)
// ---------------------------------------------------------------------------

VirtualMemoryArena* g_arena = nullptr;

constexpr uint64_t kFlsDtor = 0x140020000ULL;
constexpr uint64_t kOnceCb = 0x140020100ULL;
constexpr uint64_t kWorkCb = 0x140020200ULL;
constexpr uint64_t kSimpleCb = 0x140020300ULL;
constexpr uint64_t kThreadStart = 0x140020400ULL;
constexpr uint64_t kFlsDtorSet = 0x140020500ULL;

uint64_t& gpr(GuestState* s, uint32_t reg) {
    return s->gpr[reg];
}

void ret(GuestState* s) {
    gpr(s, GuestReg::Rsp) += 8;
}

struct DtorRecord {
    std::thread::id thread;
    uint64_t value;
};
std::mutex g_dtor_mutex;
std::vector<DtorRecord> g_dtors;
std::atomic<uint32_t> g_fls_slot{0};

void record_dtor(uint64_t value) {
    std::lock_guard<std::mutex> lock(g_dtor_mutex);
    g_dtors.push_back({std::this_thread::get_id(), value});
}

size_t dtor_count() {
    std::lock_guard<std::mutex> lock(g_dtor_mutex);
    return g_dtors.size();
}

std::vector<DtorRecord> dtor_records() {
    std::lock_guard<std::mutex> lock(g_dtor_mutex);
    return g_dtors;
}

void clear_dtors() {
    std::lock_guard<std::mutex> lock(g_dtor_mutex);
    g_dtors.clear();
}

// void NTAPI dtor(void* value)
void sub_fls_dtor(GuestState* s) {
    record_dtor(gpr(s, GuestReg::Rcx));
    ret(s);
}

// Destructor that stores a new value in the same slot once (exercises the repeat passes).
std::atomic<int> g_dtor_set_budget{0};
void sub_fls_dtor_set(GuestState* s) {
    record_dtor(gpr(s, GuestReg::Rcx));
    if (g_dtor_set_budget.fetch_sub(1) > 0) {
        pg6_win32_fls_set_value(g_fls_slot.load(), gpr(s, GuestReg::Rcx) + 1);
    }
    ret(s);
}

// BOOL CALLBACK once(PINIT_ONCE once, PVOID param, PVOID* context)
std::atomic<int> g_once_calls{0};
std::atomic<int> g_once_mode{0};  // 0 success (context = param), 1 failure, 2 abandon, 3 bad context
void sub_once_cb(GuestState* s) {
    g_once_calls.fetch_add(1);
    const uint64_t param = gpr(s, GuestReg::Rdx);
    const uint64_t slot = gpr(s, GuestReg::R8);
    const int mode = g_once_mode.load();
    uint64_t result = 1;
    if (mode == 1) {
        result = 0;
        win32::SetLastError(1234);
    } else if (mode == 2) {
        s->unwind_pending = 1;
    } else {
        const uint64_t context = mode == 3 ? (param | 1) : param;
        g_arena->write(slot, &context, sizeof(context));
    }
    gpr(s, GuestReg::Rax) = result;
    ret(s);
}

struct WorkRecordT {
    uint64_t instance;
    uint64_t context;
    uint64_t work;
    std::thread::id thread;
};
std::mutex g_work_mutex;
std::vector<WorkRecordT> g_works;

// void CALLBACK work(PTP_CALLBACK_INSTANCE, PVOID context, PTP_WORK work)
void sub_work_cb(GuestState* s) {
    std::lock_guard<std::mutex> lock(g_work_mutex);
    g_works.push_back({gpr(s, GuestReg::Rcx), gpr(s, GuestReg::Rdx), gpr(s, GuestReg::R8),
                       std::this_thread::get_id()});
    ret(s);
}

std::atomic<int> g_simple_calls{0};
std::atomic<uint64_t> g_simple_ctx{0};
void sub_simple_cb(GuestState* s) {
    g_simple_ctx = gpr(s, GuestReg::Rdx);
    g_simple_calls.fetch_add(1);
    ret(s);
}

// DWORD WINAPI start(void* p): FlsSetValue(g_fls_slot, p)
void sub_thread_start(GuestState* s) {
    pg6_win32_fls_set_value(g_fls_slot.load(), gpr(s, GuestReg::Rcx));
    gpr(s, GuestReg::Rax) = 0;
    ret(s);
}

// Guest code that calls an import thunk (g_caller_target) and treats a pending unwind as caught:
// the enclosing guest activation a callback exception needs to propagate to.
constexpr uint64_t kOnceCaller = 0x140020600ULL;
uint64_t g_caller_target = 0;
uint64_t g_caller_args[4] = {};
std::atomic<int> g_caller_unwound{0};
uint64_t g_caller_result = 0;
void sub_once_caller(GuestState* s) {
    const uint64_t saved = gpr(s, GuestReg::Rsp);
    gpr(s, GuestReg::Rsp) = saved - 0x28 - 8;
    const uint64_t return_address = kOnceCaller + 0x20;
    g_arena->write(gpr(s, GuestReg::Rsp), &return_address, sizeof(return_address));
    gpr(s, GuestReg::Rcx) = g_caller_args[0];
    gpr(s, GuestReg::Rdx) = g_caller_args[1];
    gpr(s, GuestReg::R8) = g_caller_args[2];
    gpr(s, GuestReg::R9) = g_caller_args[3];
    __playgta6_indirect_call(s, g_caller_target);
    if (s->unwind_pending != 0) {
        s->unwind_pending = 0;
        g_caller_unwound = 1;
    } else {
        g_caller_result = gpr(s, GuestReg::Rax);
    }
    gpr(s, GuestReg::Rsp) = saved;
    ret(s);
}

// O32: a work callback that only reports that it started (the closer races its return).
constexpr uint64_t kRaceCb = 0x140020700ULL;
std::atomic<int> g_race_started{0};
void sub_race_cb(GuestState* s) {
    g_race_started.store(1, std::memory_order_release);
    ret(s);
}

const LiftedFunctionEntry kFunctions[] = {
    {kFlsDtor, sub_fls_dtor},
    {kOnceCb, sub_once_cb},
    {kWorkCb, sub_work_cb},
    {kSimpleCb, sub_simple_cb},
    {kThreadStart, sub_thread_start},
    {kFlsDtorSet, sub_fls_dtor_set},
    {kOnceCaller, sub_once_caller},
    {kRaceCb, sub_race_cb},
};

struct GuestFault {
    FaultInfo info;
};

void throwing_handler(const FaultInfo* info) {
    throw GuestFault{*info};
}

struct Fixture {
    VirtualMemoryArena arena;
    GuestThread thread{};

    Fixture() {
        reset_guest_runtime();
        com::reset_com_bridge();
        callbacks::reset_callbacks();
        rtl::reset_rtl();
        win32::fls_reset_for_tests();
        com::set_guest_memory(&arena);
        win32::set_vmem_arena(&arena);
        set_fault_handler(throwing_handler);
        g_arena = &arena;
        CHECK(register_lifted_functions(kFunctions));
        CHECK(create_guest_thread(arena, 256 * 1024, thread));
        clear_dtors();
    }

    ~Fixture() {
        com::reset_com_bridge();
        callbacks::reset_callbacks();
        rtl::reset_rtl();
        win32::fls_reset_for_tests();
        destroy_guest_thread(arena, thread);
        com::set_guest_memory(nullptr);
        win32::set_vmem_arena(nullptr);
        reset_guest_runtime();
        g_arena = nullptr;
    }

    GuestState& state() { return thread.state; }

    // Runs an import thunk the way the import bridge does: arguments in rcx/rdx/r8/r9 and on the
    // stack above the shadow space, the return address on top of the stack.
    uint64_t call(void (*thunk)(void*), std::initializer_list<uint64_t> args) {
        GuestState& s = thread.state;
        const uint64_t entry_rsp = thread.stack_top - 0x200;
        gpr(&s, GuestReg::Rsp) = entry_rsp;
        const uint64_t return_address = 0x140020FF0ULL;
        CHECK(arena.write(entry_rsp, &return_address, sizeof(return_address)));
        static constexpr uint32_t kRegs[4] = {GuestReg::Rcx, GuestReg::Rdx, GuestReg::R8, GuestReg::R9};
        size_t i = 0;
        for (const uint64_t a : args) {
            if (i < 4) {
                gpr(&s, kRegs[i]) = a;
            } else {
                const uint64_t at = entry_rsp + 0x28 + 8 * (i - 4);
                CHECK(arena.write(at, &a, sizeof(a)));
            }
            ++i;
        }
        thunk(&s);
        return gpr(&s, GuestReg::Rax);
    }

    uint64_t alloc(uint64_t size) {
        const uint64_t va = arena.virtual_alloc(0, size, os::kMemReserve | os::kMemCommit, PageProtect::ReadWrite);
        CHECK(va != 0);
        return va;
    }

    template <typename T>
    T read(uint64_t va) {
        T value{};
        CHECK(arena.read(va, &value, sizeof(value)));
        return value;
    }

    template <typename T>
    void write(uint64_t va, const T& value) {
        CHECK(arena.write(va, &value, sizeof(value)));
    }
};

// ---------------------------------------------------------------------------
// InitializeCriticalSectionEx, pointer encoding, SwitchToThread
// ---------------------------------------------------------------------------

void test_critical_section_ex() {
    alignas(16) unsigned char storage[40];
    std::memset(storage, 0xCD, sizeof(storage));  // uninitialized guest memory is not zero
    auto* cs = reinterpret_cast<Pg6CriticalSection*>(storage);
    CHECK(pg6_win32_initialize_critical_section_ex(cs, 4000, 0) == 1);
    pg6_win32_enter_critical_section(cs);
    pg6_win32_enter_critical_section(cs);  // recursive
    pg6_win32_leave_critical_section(cs);
    pg6_win32_leave_critical_section(cs);
    CHECK(pg6_win32_try_enter_critical_section(cs) == 1);
    pg6_win32_leave_critical_section(cs);
    pg6_win32_delete_critical_section(cs);
    // The flags the CRT passes (spin count flavours, NO_DEBUG_INFO) work too.
    std::memset(storage, 0xCD, sizeof(storage));
    CHECK(pg6_win32_initialize_critical_section_ex(cs, 0, 0x01000000u) == 1);
    pg6_win32_delete_critical_section(cs);
    // Mutual exclusion across threads.
    std::memset(storage, 0, sizeof(storage));
    CHECK(pg6_win32_initialize_critical_section_ex(cs, 0, 0) == 1);
    int counter = 0;
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&] {
            for (int i = 0; i < 1000; ++i) {
                pg6_win32_enter_critical_section(cs);
                ++counter;
                pg6_win32_leave_critical_section(cs);
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    CHECK(counter == 4000);
    pg6_win32_delete_critical_section(cs);
    // Error paths.
    CHECK(pg6_win32_initialize_critical_section_ex(nullptr, 0, 0) == 0);
    CHECK(last_error() == win32::ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_initialize_critical_section_ex(cs, 0, 0x1u) == 0);
    CHECK(last_error() == win32::ERROR_INVALID_PARAMETER);
}

void test_pointer_encoding() {
    const uint64_t samples[] = {0, 1, 0x1000, 0x140001234ULL, 0x00007FF812345678ULL,
                                0xFFFFFFFFFFFFFFFFULL, 0x8000000000000000ULL, 0xDEADBEEFCAFEBABEULL};
    std::set<uint64_t> encoded;
    for (const uint64_t p : samples) {
        const uint64_t e = pg6_win32_encode_pointer(p);
        CHECK(pg6_win32_decode_pointer(e) == p);
        CHECK(e == pg6_win32_encode_pointer(p));  // stable
        encoded.insert(e);
        if (p != 0) {
            CHECK(e != p);
        }
    }
    CHECK(encoded.size() == sizeof(samples) / sizeof(samples[0]));  // injective on the samples
    CHECK(pg6_win32_encode_pointer(0) != 0);  // like Windows, NULL does not stay NULL
    // Stable across threads.
    uint64_t other = 0;
    std::thread([&] { other = pg6_win32_encode_pointer(0x1234); }).join();
    CHECK(other == pg6_win32_encode_pointer(0x1234));
    // Decode is the exact inverse for arbitrary values.
    uint64_t x = 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 1000; ++i) {
        x = x * 6364136223846793005ULL + 1442695040888963407ULL;
        CHECK(pg6_win32_decode_pointer(pg6_win32_encode_pointer(x)) == x);
    }
}

void test_switch_to_thread() {
    CHECK(pg6_win32_switch_to_thread() == 1);
}

// ---------------------------------------------------------------------------
// FLS
// ---------------------------------------------------------------------------

struct HostInvoker {
    std::vector<std::pair<uint64_t, uint64_t>> calls;  // (callback, value)
    static bool run(void* ctx, uint64_t callback, uint64_t value) {
        static_cast<HostInvoker*>(ctx)->calls.emplace_back(callback, value);
        return true;
    }
};

void test_fls_basic() {
    win32::fls_reset_for_tests();
    const uint32_t a = pg6_win32_fls_alloc(0);
    const uint32_t b = pg6_win32_fls_alloc(0);
    CHECK(a != win32::FLS_OUT_OF_INDEXES && b != win32::FLS_OUT_OF_INDEXES);
    CHECK(a != 0 && b != 0 && a != b);  // slot 0 is reserved
    CHECK(pg6_win32_fls_get_value(a) == 0);
    CHECK(last_error() == win32::ERROR_SUCCESS);
    CHECK(pg6_win32_fls_set_value(a, 0xAAAA) == 1);
    CHECK(pg6_win32_fls_set_value(b, 0xBBBB'0000'0000'0001ULL) == 1);
    CHECK(pg6_win32_fls_get_value(a) == 0xAAAA);
    CHECK(pg6_win32_fls_get_value(b) == 0xBBBB'0000'0000'0001ULL);
    // Values are per thread, a new thread starts with 0 in every slot.
    uint64_t seen_a = 1, seen_b = 1, own_after = 0;
    std::thread([&] {
        seen_a = pg6_win32_fls_get_value(a);
        seen_b = pg6_win32_fls_get_value(b);
        pg6_win32_fls_set_value(a, 0x7777);
        own_after = pg6_win32_fls_get_value(a);
    }).join();
    CHECK(seen_a == 0 && seen_b == 0 && own_after == 0x7777);
    CHECK(pg6_win32_fls_get_value(a) == 0xAAAA);  // unchanged by the other thread
    // FLS and TLS are separate index spaces.
    const uint32_t tls = pg6_win32_tls_alloc();
    CHECK(tls != win32::TLS_OUT_OF_INDEXES);
    CHECK(pg6_win32_tls_set_value(tls, reinterpret_cast<void*>(0x55)) == 1);
    CHECK(pg6_win32_fls_get_value(a) == 0xAAAA);
    CHECK(pg6_win32_tls_get_value(tls) == reinterpret_cast<void*>(0x55));
    pg6_win32_tls_free(tls);
    // Errors: index 0, unallocated, out of range.
    CHECK(pg6_win32_fls_get_value(0) == 0 && last_error() == win32::ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_fls_set_value(0, 1) == 0 && last_error() == win32::ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_fls_get_value(win32::kFlsMaxSlots) == 0 && last_error() == win32::ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_fls_set_value(0xFFFFFFFFu, 1) == 0);
    CHECK(pg6_win32_fls_free(0xFFFFFFFFu) == 0 && last_error() == win32::ERROR_INVALID_PARAMETER);
    // Free + reuse: the lowest free slot comes back, with a clean value in every thread.
    CHECK(pg6_win32_fls_free(a) == 1);
    CHECK(pg6_win32_fls_get_value(a) == 0 && last_error() == win32::ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_fls_free(a) == 0);  // double free
    const uint32_t c = pg6_win32_fls_alloc(0);
    CHECK(c == a);
    CHECK(pg6_win32_fls_get_value(c) == 0);
    uint64_t reused_other = 1;
    std::thread([&] { reused_other = pg6_win32_fls_get_value(c); }).join();
    CHECK(reused_other == 0);
    CHECK(win32::fls_allocated_count() == 2);
    // Exhaustion: FLS_OUT_OF_INDEXES with ERROR_NOT_ENOUGH_MEMORY, recovers after a free.
    std::vector<uint32_t> all;
    for (;;) {
        const uint32_t i = pg6_win32_fls_alloc(0);
        if (i == win32::FLS_OUT_OF_INDEXES) {
            break;
        }
        all.push_back(i);
    }
    CHECK(last_error() == win32::ERROR_NOT_ENOUGH_MEMORY);
    CHECK(all.size() + 2 == win32::kFlsMaxSlots - 1);
    CHECK(pg6_win32_fls_free(all[10]) == 1);
    CHECK(pg6_win32_fls_alloc(0) == all[10]);
    win32::fls_reset_for_tests();
    CHECK(win32::fls_allocated_count() == 0);
}

void test_fls_destructors_host() {
    win32::fls_reset_for_tests();
    const uint32_t with_dtor = pg6_win32_fls_alloc(0xD1);
    const uint32_t other_dtor = pg6_win32_fls_alloc(0xD2);
    const uint32_t no_dtor = pg6_win32_fls_alloc(0);
    const uint32_t never_set = pg6_win32_fls_alloc(0xD3);
    // A thread sets values and exits through the exit hook: only non-null slots with a destructor
    // run, each with its value.
    HostInvoker exit_calls;
    std::thread([&] {
        pg6_win32_fls_set_value(with_dtor, 0x11);
        pg6_win32_fls_set_value(other_dtor, 0x22);
        pg6_win32_fls_set_value(no_dtor, 0x33);
        pg6_win32_fls_set_value(other_dtor, 0);  // explicitly cleared: no call
        win32::fls_run_thread_exit(&HostInvoker::run, &exit_calls);
        // storage was released: a second exit is a no-op and values read as 0 afterwards
        win32::fls_run_thread_exit(&HostInvoker::run, &exit_calls);
        CHECK(pg6_win32_fls_get_value(with_dtor) == 0);
    }).join();
    CHECK(exit_calls.calls.size() == 1);
    CHECK(exit_calls.calls[0].first == 0xD1 && exit_calls.calls[0].second == 0x11);
    (void)never_set;

    // FlsFree runs the destructor for every thread's non-null value, on the calling thread.
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> workers;
    for (uint64_t t = 1; t <= 3; ++t) {
        workers.emplace_back([&, t] {
            pg6_win32_fls_set_value(with_dtor, 0x100 + t);
            ready.fetch_add(1);
            while (!go.load()) {
                std::this_thread::yield();
            }
            // After the free the slot is gone for this thread too.
            CHECK(pg6_win32_fls_get_value(with_dtor) == 0);
            CHECK(last_error() == win32::ERROR_INVALID_PARAMETER);
        });
    }
    CHECK(wait_for([&] { return ready.load() == 3; }));
    pg6_win32_fls_set_value(with_dtor, 0x100);
    HostInvoker free_calls;
    CHECK(win32::FlsFree(with_dtor, &HostInvoker::run, &free_calls));
    go = true;
    for (auto& w : workers) {
        w.join();
    }
    std::set<uint64_t> values;
    for (const auto& c : free_calls.calls) {
        CHECK(c.first == 0xD1);
        values.insert(c.second);
    }
    CHECK(free_calls.calls.size() == 4);
    CHECK((values == std::set<uint64_t>{0x100, 0x101, 0x102, 0x103}));
    // A slot without values or without a destructor: nothing runs.
    HostInvoker none;
    CHECK(win32::FlsFree(never_set, &HostInvoker::run, &none));
    CHECK(win32::FlsFree(no_dtor, &HostInvoker::run, &none));
    CHECK(none.calls.empty());
    // Reused slot does not inherit the old destructor or values.
    const uint32_t reused = pg6_win32_fls_alloc(0);
    CHECK(reused == with_dtor);
    CHECK(pg6_win32_fls_get_value(reused) == 0);
    HostInvoker reuse_calls;
    CHECK(win32::FlsFree(reused, &HostInvoker::run, &reuse_calls));
    CHECK(reuse_calls.calls.empty());
    win32::fls_reset_for_tests();
}

void test_fls_exit_passes() {
    win32::fls_reset_for_tests();
    // A destructor that stores a new value is called again (bounded number of passes).
    struct Ctx {
        uint32_t slot;
        int calls = 0;
        static bool run(void* c, uint64_t, uint64_t value) {
            auto* ctx = static_cast<Ctx*>(c);
            ++ctx->calls;
            win32::FlsSetValue(ctx->slot, value + 1);  // always sets again: must still terminate
            return true;
        }
    } ctx{};
    ctx.slot = pg6_win32_fls_alloc(0x1);
    std::thread([&] {
        pg6_win32_fls_set_value(ctx.slot, 1);
        win32::fls_run_thread_exit(&Ctx::run, &ctx);
    }).join();
    CHECK(ctx.calls >= 2 && ctx.calls <= 8);
    // An invoker that asks to stop (a guest unwind is pending) ends the run.
    struct Stop {
        int calls = 0;
        static bool run(void* c, uint64_t, uint64_t) {
            ++static_cast<Stop*>(c)->calls;
            return false;
        }
    } stop;
    const uint32_t s1 = pg6_win32_fls_alloc(0x2);
    const uint32_t s2 = pg6_win32_fls_alloc(0x3);
    std::thread([&] {
        pg6_win32_fls_set_value(s1, 5);
        pg6_win32_fls_set_value(s2, 6);
        win32::fls_run_thread_exit(&Stop::run, &stop);
    }).join();
    CHECK(stop.calls == 1);
    win32::fls_reset_for_tests();
}

void test_fls_guest_destructors() {
    Fixture fx;
    const uint32_t slot = pg6_win32_fls_alloc(kFlsDtor);
    g_fls_slot = slot;

    // FlsFree through the import thunk: the guest destructor runs with this thread's value.
    CHECK(pg6_win32_fls_set_value(slot, 0xABCD) == 1);
    CHECK(fx.call(&pg6_gthunk_fls_free, {slot}) == 1);
    {
        const auto records = dtor_records();
        CHECK(records.size() == 1);
        CHECK(records[0].value == 0xABCD);
        CHECK(records[0].thread == std::this_thread::get_id());
    }
    CHECK(com::bound_guest_state() == nullptr);
    CHECK(fx.call(&pg6_gthunk_fls_free, {slot}) == 0);  // already freed
    CHECK(last_error() == win32::ERROR_INVALID_PARAMETER);
    clear_dtors();

    // Exit of a host thread that ran guest code (lazily created guest thread): its destructors
    // run on that thread through call_guest, only for non-null slots.
    const uint32_t s2 = pg6_win32_fls_alloc(kFlsDtor);
    const uint32_t s3 = pg6_win32_fls_alloc(kFlsDtor);
    std::thread::id worker_id;
    std::thread worker([&] {
        worker_id = std::this_thread::get_id();
        (void)com::current_guest_state();  // creates the guest thread for this host thread
        pg6_win32_fls_set_value(s2, 0x2222);
        pg6_win32_fls_set_value(s3, 0x3333);
        pg6_win32_fls_set_value(s3, 0);
    });
    worker.join();
    CHECK(dtor_count() == 1);
    if (dtor_count() == 1) {
        const auto records = dtor_records();
        CHECK(records[0].value == 0x2222);
        CHECK(records[0].thread == worker_id);
    }
    clear_dtors();

    // CreateThread threads (O16 trampoline): the start routine sets an FLS value; the destructor
    // runs when the thread ends.
    g_fls_slot = s2;
    const uint64_t handle = fx.call(&pg6_gthunk_create_thread, {0, 0, kThreadStart, 0x5151, 0, 0});
    CHECK(handle != 0);
    CHECK(pg6_win32_wait_for_single_object(handle, 0xFFFFFFFFu) == 0);
    CHECK(wait_for([] { return dtor_count() == 1; }));
    if (dtor_count() == 1) {
        CHECK(dtor_records()[0].value == 0x5151);
        CHECK(dtor_records()[0].thread != std::this_thread::get_id());
    }
    pg6_win32_close_handle(handle);
    clear_dtors();

    // The CRT exit funnel runs the exiting thread's destructors, after the atexit handlers.
    pg6_win32_fls_set_value(s3, 0x9999);
    crt_exit::run_exit_cleanup(fx.state(), crt_exit::ExitMode::Full);
    CHECK(dtor_count() == 1);
    if (dtor_count() == 1) {
        CHECK(dtor_records()[0].value == 0x9999);
    }
    CHECK(pg6_win32_fls_get_value(s3) == 0);
    clear_dtors();
    // quick_exit and _exit paths do not run them.
    pg6_win32_fls_set_value(s3, 0x1);
    crt_exit::run_exit_cleanup(fx.state(), crt_exit::ExitMode::Quick);
    crt_exit::run_exit_cleanup(fx.state(), crt_exit::ExitMode::None);
    CHECK(dtor_count() == 0);
    CHECK(pg6_win32_fls_get_value(s3) == 0x1);
    crt_exit::reset_crt_exit();

    // A destructor that stores a value again is run again (and the loop terminates).
    g_dtor_set_budget = 2;
    const uint32_t s4 = pg6_win32_fls_alloc(kFlsDtorSet);
    g_fls_slot = s4;
    pg6_win32_fls_set_value(s4, 10);
    pg6_win32_fls_free(s3);
    pg6_win32_fls_free(s2);
    crt_exit::run_exit_cleanup(fx.state(), crt_exit::ExitMode::Full);
    CHECK(dtor_count() == 3);  // 10, 11, 12
    crt_exit::reset_crt_exit();
}

// ---------------------------------------------------------------------------
// InitOnce
// ---------------------------------------------------------------------------

std::atomic<int> g_host_once_calls{0};
std::atomic<int> g_host_once_fail{0};
int32_t host_once_fn(Pg6InitOnce*, void* parameter, void** context) {
    g_host_once_calls.fetch_add(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));  // widen the race window
    if (g_host_once_fail.load() != 0) {
        return 0;
    }
    *context = parameter;
    return 1;
}

void test_init_once_host() {
    alignas(8) Pg6InitOnce once{};
    g_host_once_calls = 0;
    g_host_once_fail = 0;
    std::atomic<int> ok_count{0};
    std::atomic<int> ctx_ok{0};
    std::atomic<int> ready{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&] {
            ready.fetch_add(1);
            while (ready.load() < 8) {
                std::this_thread::yield();
            }
            void* context = nullptr;
            if (pg6_win32_init_once_execute_once(&once, &host_once_fn, reinterpret_cast<void*>(0x4000), &context) == 1) {
                ok_count.fetch_add(1);
                if (context == reinterpret_cast<void*>(0x4000)) {
                    ctx_ok.fetch_add(1);
                }
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    CHECK(g_host_once_calls == 1);
    CHECK(ok_count == 8 && ctx_ok == 8);
    CHECK(once.ptr == (0x4000 | 2));  // completed state with the context in the 8 bytes
    // Completed: never runs again, context is returned.
    void* context = nullptr;
    CHECK(pg6_win32_init_once_execute_once(&once, &host_once_fn, nullptr, &context) == 1);
    CHECK(context == reinterpret_cast<void*>(0x4000));
    CHECK(g_host_once_calls == 1);
    CHECK(pg6_win32_init_once_execute_once(&once, &host_once_fn, nullptr, nullptr) == 1);

    // Failure leaves the once uninitialized; the next caller retries; waiters of a failed
    // initialization are not stuck.
    alignas(8) Pg6InitOnce retry{};
    g_host_once_calls = 0;
    g_host_once_fail = 1;
    void* ctx2 = reinterpret_cast<void*>(0x1);
    CHECK(pg6_win32_init_once_execute_once(&retry, &host_once_fn, nullptr, &ctx2) == 0);
    CHECK(ctx2 == nullptr);
    CHECK(retry.ptr == 0);
    CHECK(g_host_once_calls == 1);
    g_host_once_fail = 0;
    CHECK(pg6_win32_init_once_execute_once(&retry, &host_once_fn, reinterpret_cast<void*>(0x80), &ctx2) == 1);
    CHECK(ctx2 == reinterpret_cast<void*>(0x80));
    CHECK(g_host_once_calls == 2);

    // Abandoned (unwound) initialization resets the state too.
    alignas(8) Pg6InitOnce abandoned{};
    uint64_t result = 0;
    CHECK(!win32::InitOnceRun(&abandoned, [](uint64_t*) { return win32::InitOnceOutcome::Abandoned; }, &result));
    CHECK(abandoned.ptr == 0);
    // A context with reserved bits fails the call and leaves the once uninitialized.
    CHECK(!win32::InitOnceRun(&abandoned,
                              [](uint64_t* c) {
                                  *c = 0x1001;
                                  return win32::InitOnceOutcome::Completed;
                              },
                              &result));
    CHECK(abandoned.ptr == 0 && last_error() == win32::ERROR_INVALID_PARAMETER);
    // Error paths.
    CHECK(pg6_win32_init_once_execute_once(nullptr, &host_once_fn, nullptr, nullptr) == 0);
    CHECK(pg6_win32_init_once_execute_once(&once, nullptr, nullptr, nullptr) == 0);
    alignas(8) unsigned char raw[16] = {};
    CHECK(pg6_win32_init_once_execute_once(reinterpret_cast<Pg6InitOnce*>(raw + 1), &host_once_fn, nullptr, nullptr) == 0);
    alignas(8) Pg6InitOnce garbage{0x3};  // not a state this module writes
    CHECK(pg6_win32_init_once_execute_once(&garbage, &host_once_fn, nullptr, nullptr) == 0);
}

void test_init_once_guest() {
    Fixture fx;
    const uint64_t once = fx.alloc(0x1000);
    const uint64_t out = once + 0x100;
    g_once_calls = 0;
    g_once_mode = 0;
    CHECK(fx.call(&pg6_gthunk_init_once_execute_once, {once, kOnceCb, 0x7000, out}) == 1);
    CHECK(g_once_calls == 1);
    CHECK(fx.read<uint64_t>(out) == 0x7000);
    CHECK(fx.read<uint64_t>(once) == (0x7000 | 2));
    // Already complete: callback not called again, context delivered again.
    fx.write<uint64_t>(out, 0);
    CHECK(fx.call(&pg6_gthunk_init_once_execute_once, {once, kOnceCb, 0x9999, out}) == 1);
    CHECK(g_once_calls == 1);
    CHECK(fx.read<uint64_t>(out) == 0x7000);
    // NULL context pointer is allowed.
    CHECK(fx.call(&pg6_gthunk_init_once_execute_once, {once, kOnceCb, 0, 0}) == 1);

    // Failure then retry.
    const uint64_t once2 = once + 0x200;
    g_once_mode = 1;
    CHECK(fx.call(&pg6_gthunk_init_once_execute_once, {once2, kOnceCb, 0x10, out}) == 0);
    CHECK(last_error() == 1234);  // the callback's last error survives
    CHECK(fx.read<uint64_t>(once2) == 0);
    g_once_mode = 0;
    CHECK(fx.call(&pg6_gthunk_init_once_execute_once, {once2, kOnceCb, 0x20, out}) == 1);
    CHECK(fx.read<uint64_t>(once2) == (0x20 | 2));
    CHECK(g_once_calls == 3);
    // Context with reserved low bits: failure, once stays uninitialized.
    const uint64_t once3 = once + 0x300;
    g_once_mode = 3;
    CHECK(fx.call(&pg6_gthunk_init_once_execute_once, {once3, kOnceCb, 0x40, out}) == 0);
    CHECK(fx.read<uint64_t>(once3) == 0);
    g_once_mode = 0;
    // Error paths: null once, null callback, unmapped once.
    CHECK(fx.call(&pg6_gthunk_init_once_execute_once, {0, kOnceCb, 0, 0}) == 0);
    CHECK(last_error() == win32::ERROR_INVALID_PARAMETER);
    CHECK(fx.call(&pg6_gthunk_init_once_execute_once, {once + 0x400, 0, 0, 0}) == 0);
    CHECK(fx.call(&pg6_gthunk_init_once_execute_once, {0x10, kOnceCb, 0, 0}) == 0);

    // Eight host threads racing through the guest thunk: the callback runs once. The racers use
    // their own guest threads.
    const uint64_t once4 = once + 0x500;
    g_once_calls = 0;
    std::atomic<int> ok{0};
    std::atomic<int> ready{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&] {
            GuestThread gt{};
            CHECK(create_guest_thread(fx.arena, 128 * 1024, gt));
            ready.fetch_add(1);
            while (ready.load() < 8) {
                std::this_thread::yield();
            }
            const uint64_t entry_rsp = gt.stack_top - 0x200;
            gt.state.gpr[GuestReg::Rsp] = entry_rsp;
            const uint64_t ra = 0x140020FF0ULL;
            fx.arena.write(entry_rsp, &ra, sizeof(ra));
            gt.state.gpr[GuestReg::Rcx] = once4;
            gt.state.gpr[GuestReg::Rdx] = kOnceCb;
            gt.state.gpr[GuestReg::R8] = 0x6000;
            gt.state.gpr[GuestReg::R9] = 0;
            pg6_gthunk_init_once_execute_once(&gt.state);
            if (gt.state.gpr[GuestReg::Rax] == 1) {
                ok.fetch_add(1);
            }
            destroy_guest_thread(fx.arena, gt);
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    CHECK(ok == 8);
    CHECK(g_once_calls == 1);
    CHECK(fx.read<uint64_t>(once4) == (0x6000 | 2));

    // An unwind escaping the callback propagates through the import (guest caller sees it) and
    // leaves the once uninitialized, so the next caller runs the initializer.
    const uint64_t once5 = once + 0x600;
    g_once_mode = 2;
    g_caller_target = register_host_callable(&pg6_gthunk_init_once_execute_once, "InitOnceExecuteOnce");
    g_caller_args[0] = once5;
    g_caller_args[1] = kOnceCb;
    g_caller_args[2] = 0x30;
    g_caller_args[3] = out;
    g_caller_unwound = 0;
    g_once_calls = 0;
    fx.state().gpr[GuestReg::Rsp] = fx.thread.stack_top;
    call_guest(fx.arena, fx.state(), kOnceCaller);
    CHECK(g_caller_unwound == 1);
    CHECK(g_once_calls == 1);
    CHECK(fx.state().unwind_pending == 0);
    CHECK(fx.state().gpr[GuestReg::Rsp] == fx.thread.stack_top);
    CHECK(fx.read<uint64_t>(once5) == 0);
    g_once_mode = 0;
    g_caller_unwound = 0;
    call_guest(fx.arena, fx.state(), kOnceCaller);  // retried: succeeds
    CHECK(g_caller_unwound == 0 && g_caller_result == 1);
    CHECK(g_once_calls == 2);
    CHECK(fx.read<uint64_t>(once5) == (0x30 | 2));
}

// ---------------------------------------------------------------------------
// SLIST
// ---------------------------------------------------------------------------

struct SListFixture {
    Fixture fx;
    uint64_t base = 0;
    Pg6SListHeader* head = nullptr;

    explicit SListFixture(uint64_t entries) {
        base = fx.alloc(0x1000 + entries * 32);
        head = reinterpret_cast<Pg6SListHeader*>(fx.arena.host_ptr(base, 16));
        CHECK(head != nullptr);
        pg6_win32_initialize_s_list_head(head);
    }

    uint64_t entry(uint64_t i) const { return base + 0x1000 + i * 32; }
    uint64_t index_of(uint64_t va) const { return (va - base - 0x1000) / 32; }
    void set_payload(uint64_t i, uint64_t v) { fx.write<uint64_t>(entry(i) + 16, v); }
    uint64_t payload(uint64_t va) { return fx.read<uint64_t>(va + 16); }
};

void test_slist_basic() {
    SListFixture f(16);
    CHECK(sizeof(Pg6SListHeader) == 16 && alignof(Pg6SListHeader) == 16);
    CHECK(pg6_win32_query_depth_slist(f.head) == 0);
    CHECK(pg6_win32_interlocked_pop_entry_slist(f.head) == 0);
    CHECK(pg6_win32_interlocked_flush_slist(f.head) == 0);
    // Push returns the previous first entry (0 for an empty list); pop is LIFO.
    CHECK(pg6_win32_interlocked_push_entry_slist(f.head, f.entry(0)) == 0);
    CHECK(pg6_win32_interlocked_push_entry_slist(f.head, f.entry(1)) == f.entry(0));
    CHECK(pg6_win32_interlocked_push_entry_slist(f.head, f.entry(2)) == f.entry(1));
    CHECK(pg6_win32_query_depth_slist(f.head) == 3);
    CHECK(f.head->first_entry == f.entry(2));
    CHECK(f.fx.read<uint64_t>(f.entry(2)) == f.entry(1));  // SLIST_ENTRY.Next in guest memory
    CHECK(f.fx.read<uint64_t>(f.entry(0)) == 0);
    CHECK(pg6_win32_interlocked_pop_entry_slist(f.head) == f.entry(2));
    CHECK(pg6_win32_query_depth_slist(f.head) == 2);
    // Flush returns the whole chain (newest first) and empties the list.
    CHECK(pg6_win32_interlocked_push_entry_slist(f.head, f.entry(3)) == f.entry(1));
    const uint64_t chain = pg6_win32_interlocked_flush_slist(f.head);
    CHECK(chain == f.entry(3));
    CHECK(f.fx.read<uint64_t>(chain) == f.entry(1));
    CHECK(f.fx.read<uint64_t>(f.entry(1)) == f.entry(0));
    CHECK(f.fx.read<uint64_t>(f.entry(0)) == 0);
    CHECK(pg6_win32_query_depth_slist(f.head) == 0);
    CHECK(f.head->first_entry == 0);
    CHECK(pg6_win32_interlocked_pop_entry_slist(f.head) == 0);
    // Reuse after flush.
    CHECK(pg6_win32_interlocked_push_entry_slist(f.head, f.entry(5)) == 0);
    CHECK(pg6_win32_interlocked_pop_entry_slist(f.head) == f.entry(5));
    CHECK(pg6_win32_query_depth_slist(f.head) == 0);
    // The lock bit is never left set.
    CHECK((f.head->depth_sequence >> 63) == 0);
    // Hostile arguments: nothing is pushed, nothing is written outside guest memory.
    CHECK(pg6_win32_interlocked_push_entry_slist(f.head, 0) == 0);
    CHECK(last_error() == win32::ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_interlocked_push_entry_slist(f.head, f.entry(0) + 8) == 0);  // not 16-aligned
    CHECK(pg6_win32_interlocked_push_entry_slist(f.head, 0x10) == 0);            // unmapped
    CHECK(pg6_win32_query_depth_slist(f.head) == 0);
    CHECK(pg6_win32_interlocked_push_entry_slist(nullptr, f.entry(0)) == 0);
    CHECK(pg6_win32_interlocked_pop_entry_slist(nullptr) == 0);
    CHECK(pg6_win32_interlocked_flush_slist(nullptr) == 0);
    CHECK(pg6_win32_query_depth_slist(nullptr) == 0);
    // A corrupted list (first entry unmapped) is reported, not dereferenced.
    f.head->first_entry = 0x20;
    CHECK(pg6_win32_interlocked_pop_entry_slist(f.head) == 0);
    CHECK(last_error() == win32::ERROR_INVALID_PARAMETER);
    f.head->first_entry = 0;
}

void test_slist_stress() {
    constexpr uint64_t kThreads = 4;
    constexpr uint64_t kPerThread = 2000;
    constexpr uint64_t kEntries = kThreads * kPerThread;
    SListFixture f(kEntries);
    for (uint64_t i = 0; i < kEntries; ++i) {
        f.set_payload(i, 1000 + i);
    }
    std::vector<std::atomic<int>> seen(kEntries);
    for (auto& s : seen) {
        s = 0;
    }
    std::atomic<int> bad_payload{0};
    std::atomic<int> out_of_range{0};
    auto account = [&](uint64_t va) {
        if (va < f.entry(0) || va >= f.entry(kEntries) || ((va - f.entry(0)) % 32) != 0) {
            out_of_range.fetch_add(1);
            return;
        }
        const uint64_t i = f.index_of(va);
        if (f.payload(va) != 1000 + i) {
            bad_payload.fetch_add(1);
        }
        seen[i].fetch_add(1);
    };
    std::atomic<int> ready{0};
    std::vector<std::thread> threads;
    for (uint64_t t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            ready.fetch_add(1);
            while (ready.load() < static_cast<int>(kThreads)) {
                std::this_thread::yield();
            }
            for (uint64_t i = 0; i < kPerThread; ++i) {
                pg6_win32_interlocked_push_entry_slist(f.head, f.entry(t * kPerThread + i));
                if ((i & 1) != 0) {
                    const uint64_t popped = pg6_win32_interlocked_pop_entry_slist(f.head);
                    if (popped != 0) {
                        account(popped);
                        // The popped entry is ours again: push it back some of the time.
                        if ((i & 3) == 1) {
                            seen[f.index_of(popped)].fetch_sub(1);
                            pg6_win32_interlocked_push_entry_slist(f.head, popped);
                        }
                    }
                }
                if (i % 512 == 511) {
                    // Flush part of the list and give it back entry by entry.
                    uint64_t chain = pg6_win32_interlocked_flush_slist(f.head);
                    while (chain != 0) {
                        const uint64_t next = f.fx.read<uint64_t>(chain);
                        account(chain);
                        seen[f.index_of(chain)].fetch_sub(1);
                        pg6_win32_interlocked_push_entry_slist(f.head, chain);
                        chain = next;
                    }
                }
                pg6_win32_query_depth_slist(f.head);
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    // Drain: every entry comes out exactly once overall.
    const uint16_t depth = pg6_win32_query_depth_slist(f.head);
    uint64_t drained = 0;
    uint64_t chain = pg6_win32_interlocked_flush_slist(f.head);
    while (chain != 0) {
        const uint64_t next = f.fx.read<uint64_t>(chain);
        account(chain);
        ++drained;
        chain = next;
    }
    CHECK(out_of_range == 0 && bad_payload == 0);
    CHECK(depth == static_cast<uint16_t>(drained));
    CHECK(pg6_win32_query_depth_slist(f.head) == 0);
    int wrong = 0;
    for (uint64_t i = 0; i < kEntries; ++i) {
        if (seen[i].load() != 1) {
            ++wrong;
        }
    }
    CHECK(wrong == 0);
    CHECK(f.head->first_entry == 0);
}

// ---------------------------------------------------------------------------
// GetModuleHandleExW
// ---------------------------------------------------------------------------

void test_get_module_handle_ex() {
    Fixture fx;
    constexpr uint64_t kBase = 0x1A0000000ULL;
    constexpr uint64_t kSize = 0x30000;
    CHECK(win32::register_guest_module("rtlmod.dll", "G:\\rtlmod.dll", kBase, kSize) == kBase);
    Pg6Handle module = 123;
    // FROM_ADDRESS: anywhere inside the image maps to the module.
    CHECK(pg6_win32_get_module_handle_ex_w(win32::GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | win32::GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                           kBase + 0x1234, &module) == 1);
    CHECK(module == kBase);
    CHECK(pg6_win32_get_module_handle_ex_w(win32::GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, kBase, &module) == 1 &&
          module == kBase);
    CHECK(pg6_win32_get_module_handle_ex_w(win32::GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, kBase + kSize - 1, &module) == 1 &&
          module == kBase);
    // Just outside, null, wild addresses: not found, *module cleared.
    module = 77;
    CHECK(pg6_win32_get_module_handle_ex_w(win32::GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, kBase + kSize, &module) == 0);
    CHECK(module == 0 && last_error() == win32::ERROR_MOD_NOT_FOUND);
    CHECK(pg6_win32_get_module_handle_ex_w(win32::GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, kBase - 1, &module) == 0);
    CHECK(pg6_win32_get_module_handle_ex_w(win32::GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, 0, &module) == 0);
    CHECK(pg6_win32_get_module_handle_ex_w(win32::GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, ~0ULL, &module) == 0);
    // A second guest module is told apart.
    constexpr uint64_t kBase2 = 0x1B0000000ULL;
    CHECK(win32::register_guest_module("rtlmod2.dll", "", kBase2, 0x1000) == kBase2);
    CHECK(pg6_win32_get_module_handle_ex_w(win32::GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, kBase2 + 0x800, &module) == 1 &&
          module == kBase2);
    // The main module with a known size resolves by address too.
    constexpr uint64_t kMain = 0x140000000ULL;
    if (win32::register_main_module("rtlmain.exe", "G:\\rtlmain.exe", kMain, 0x80000) == kMain) {
        CHECK(pg6_win32_get_module_handle_ex_w(win32::GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, kMain + 0x7FFFF, &module) == 1 &&
              module == kMain);
        CHECK(pg6_win32_get_module_handle_ex_w(0, 0, &module) == 1 && module == kMain);  // NULL name: the exe
    }
    // By name (guest UTF-16 string), case-insensitive, ".dll" optional.
    const uint64_t name = fx.alloc(0x1000);
    const char16_t text[] = u"RTLMOD";
    CHECK(fx.arena.write(name, text, sizeof(text)));
    CHECK(pg6_win32_get_module_handle_ex_w(0, name, &module) == 1 && module == kBase);
    const char16_t missing[] = u"nosuchmodule.dll";
    CHECK(fx.arena.write(name, missing, sizeof(missing)));
    CHECK(pg6_win32_get_module_handle_ex_w(0, name, &module) == 0 && module == 0);
    CHECK(last_error() == win32::ERROR_MOD_NOT_FOUND);
    CHECK(pg6_win32_get_module_handle_ex_w(0, 0x10, &module) == 0);  // unreadable name
    CHECK(last_error() == win32::ERROR_INVALID_PARAMETER);

    // Reference counting on an unpinned virtual module.
    Pg6Handle virt = win32::register_virtual_module("rtlvirt.dll", false);
    CHECK(virt != 0);
    CHECK(fx.arena.write(name, u"rtlvirt.dll", sizeof(u"rtlvirt.dll")));
    CHECK(pg6_win32_get_module_handle_ex_w(0, name, &module) == 0);  // not loaded: invisible
    CHECK(pg6_win32_load_library_w(u"rtlvirt.dll") == virt);          // count 1
    CHECK(pg6_win32_get_module_handle_ex_w(0, name, &module) == 1 && module == virt);  // count 2
    CHECK(pg6_win32_free_library(virt) == 1);                          // count 1: still loaded
    CHECK(pg6_win32_get_module_handle_ex_w(win32::GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, name, &module) == 1);
    CHECK(pg6_win32_free_library(virt) == 1);                          // count 0
    CHECK(pg6_win32_get_module_handle_ex_w(0, name, &module) == 0);   // gone
    // PIN keeps a module loaded forever.
    CHECK(pg6_win32_load_library_w(u"rtlvirt.dll") == virt);
    CHECK(pg6_win32_get_module_handle_ex_w(win32::GET_MODULE_HANDLE_EX_FLAG_PIN, name, &module) == 1);
    CHECK(pg6_win32_free_library(virt) == 1);
    CHECK(pg6_win32_get_module_handle_ex_w(win32::GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, name, &module) == 1);
    // Virtual modules also own their synthetic address range.
    CHECK(pg6_win32_get_module_handle_ex_w(win32::GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, virt + 0x40, &module) == 1 &&
          module == virt);

    // Flag validation and null output.
    CHECK(pg6_win32_get_module_handle_ex_w(win32::GET_MODULE_HANDLE_EX_FLAG_PIN | win32::GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                           name, &module) == 0);
    CHECK(last_error() == win32::ERROR_INVALID_PARAMETER && module == 0);
    CHECK(pg6_win32_get_module_handle_ex_w(0x80, name, &module) == 0);
    CHECK(pg6_win32_get_module_handle_ex_w(0, name, nullptr) == 0);
    CHECK(last_error() == win32::ERROR_INVALID_PARAMETER);
}

// ---------------------------------------------------------------------------
// Locale and strings
// ---------------------------------------------------------------------------

std::u16string map_string(uint32_t flags, const std::u16string& in, bool terminated, int32_t* needed = nullptr) {
    const int32_t cch_src = terminated ? -1 : static_cast<int32_t>(in.size());
    const int32_t n = pg6_win32_lc_map_string_ex(nullptr, flags, in.c_str(), cch_src, nullptr, 0, nullptr, nullptr, 0);
    if (needed != nullptr) {
        *needed = n;
    }
    std::u16string out(static_cast<size_t>(n > 0 ? n : 0), u'\x7777');
    if (n > 0) {
        CHECK(pg6_win32_lc_map_string_ex(u"en-US", flags, in.c_str(), cch_src, out.data(), n, nullptr, nullptr, 0) == n);
        if (terminated) {
            CHECK(out.back() == 0);
            out.pop_back();
        }
    }
    return out;
}

void test_lc_map_string() {
    int32_t needed = 0;
    CHECK(map_string(win32::LCMAP_UPPERCASE, u"Hello, w\u00F6rld \u00FF \u00E9", true, &needed) ==
          u"HELLO, W\u00D6RLD \u0178 \u00C9");
    CHECK(needed == 17);  // size query counts the terminator with cchSrc == -1
    CHECK(map_string(win32::LCMAP_UPPERCASE, u"abc", false, &needed) == u"ABC");
    CHECK(needed == 3);  // and does not with an explicit length
    CHECK(map_string(win32::LCMAP_LOWERCASE, u"\u00C0\u00C9\u00CE\u00D5\u00DC \u0391\u0392\u0393 \u0414\u0416\u042F \u0401 ABC xyz", true) ==
          u"\u00E0\u00E9\u00EE\u00F5\u00FC \u03B1\u03B2\u03B3 \u0434\u0436\u044F \u0451 abc xyz");
    CHECK(map_string(win32::LCMAP_UPPERCASE, u"\u03B1\u03C2\u0434\u0451 \u0105\u0119 \u1E01", true) ==
          u"\u0391\u03A3\u0414\u0401 \u0104\u0118 \u1E00");
    // No 1:1 uppercase for sharp s; multiplication/division signs are not letters.
    CHECK(map_string(win32::LCMAP_UPPERCASE, u"\u00DF\u00D7\u00F7", true) == u"\u00DF\u00D7\u00F7");
    // Explicit length may contain NULs and is not terminated.
    const std::u16string with_nul(u"ab\0cd", 5);
    CHECK(map_string(win32::LCMAP_UPPERCASE, with_nul, false) == std::u16string(u"AB\0CD", 5));
    // Surrogates and unrelated BMP characters pass through.
    CHECK(map_string(win32::LCMAP_UPPERCASE, u"\U0001F600\u4E2D", true) == u"\U0001F600\u4E2D");
    // In place.
    char16_t buffer[8] = u"mixed";
    CHECK(pg6_win32_lc_map_string_ex(nullptr, win32::LCMAP_UPPERCASE, buffer, -1, buffer, 8, nullptr, nullptr, 0) == 6);
    CHECK(std::u16string(buffer) == u"MIXED");
    // Buffer too small.
    char16_t small[3];
    CHECK(pg6_win32_lc_map_string_ex(nullptr, win32::LCMAP_LOWERCASE, u"ABCDE", 5, small, 3, nullptr, nullptr, 0) == 0);
    CHECK(last_error() == 122);
    // Flags and argument validation.
    CHECK(pg6_win32_lc_map_string_ex(nullptr, 0, u"a", 1, small, 3, nullptr, nullptr, 0) == 0 && last_error() == 1004);
    CHECK(pg6_win32_lc_map_string_ex(nullptr, win32::LCMAP_UPPERCASE | win32::LCMAP_LOWERCASE, u"a", 1, small, 3, nullptr, nullptr, 0) == 0 &&
          last_error() == 1004);
    CHECK(pg6_win32_lc_map_string_ex(nullptr, win32::LCMAP_BYTEREV | win32::LCMAP_UPPERCASE, u"a", 1, small, 3, nullptr, nullptr, 0) == 0 &&
          last_error() == 1004);
    CHECK(pg6_win32_lc_map_string_ex(nullptr, win32::LCMAP_UPPERCASE, nullptr, 1, small, 3, nullptr, nullptr, 0) == 0 &&
          last_error() == win32::ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_lc_map_string_ex(nullptr, win32::LCMAP_UPPERCASE, u"a", 0, small, 3, nullptr, nullptr, 0) == 0);
    CHECK(pg6_win32_lc_map_string_ex(nullptr, win32::LCMAP_UPPERCASE, u"a", 1, nullptr, 3, nullptr, nullptr, 0) == 0);
    CHECK(pg6_win32_lc_map_string_ex(u"xx-nonexistent", win32::LCMAP_UPPERCASE, u"a", 1, small, 3, nullptr, nullptr, 0) == 0 &&
          last_error() == win32::ERROR_INVALID_PARAMETER);
    int reserved_dummy = 0;
    CHECK(pg6_win32_lc_map_string_ex(nullptr, win32::LCMAP_UPPERCASE, u"a", 1, small, 3, nullptr, &reserved_dummy, 0) == 0);
    CHECK(pg6_win32_lc_map_string_ex(u"de-DE", win32::LCMAP_UPPERCASE, u"a", 1, small, 3, nullptr, nullptr, 0) == 1);  // known name
}

std::vector<uint8_t> sort_key(const std::u16string& s, uint32_t extra = 0) {
    const int32_t bytes = pg6_win32_lc_map_string_ex(u"en-US", win32::LCMAP_SORTKEY | extra, s.c_str(), static_cast<int32_t>(s.size()),
                                                     nullptr, 0, nullptr, nullptr, 0);
    CHECK(bytes > 0);
    std::vector<char16_t> buffer((static_cast<size_t>(bytes) + 1) / 2);
    CHECK(pg6_win32_lc_map_string_ex(u"en-US", win32::LCMAP_SORTKEY | extra, s.c_str(), static_cast<int32_t>(s.size()),
                                     buffer.data(), static_cast<int32_t>(buffer.size()), nullptr, nullptr, 0) == bytes);
    std::vector<uint8_t> key(static_cast<size_t>(bytes));
    std::memcpy(key.data(), buffer.data(), key.size());
    return key;
}

int compare_ex(uint32_t flags, const std::u16string& a, const std::u16string& b) {
    return pg6_win32_compare_string_ex(u"en-US", flags, a.c_str(), static_cast<int32_t>(a.size()), b.c_str(),
                                       static_cast<int32_t>(b.size()), nullptr, nullptr, 0);
}

void test_sort_key_and_compare() {
    const std::u16string words[] = {u"a", u"A", u"ab", u"Ab", u"aB", u"abc", u"apple", u"Apple", u"APPLE",
                                    u"apples", u"banana", u"\u00E4pfel", u"\u00C4pfel", u"zebra", u"Zebra", u"1", u"10", u"2"};
    int checked = 0;
    for (const auto& a : words) {
        for (const auto& b : words) {
            const auto ka = sort_key(a);
            const auto kb = sort_key(b);
            const int key_order = ka == kb ? 0 : (std::lexicographical_compare(ka.begin(), ka.end(), kb.begin(), kb.end()) ? -1 : 1);
            const int cmp = compare_ex(0, a, b) - 2;  // CSTR_* -> -1/0/1
            CHECK(key_order == cmp);
            const auto ia = sort_key(a, 0x1);
            const auto ib = sort_key(b, 0x1);
            const int ikey_order = ia == ib ? 0 : (std::lexicographical_compare(ia.begin(), ia.end(), ib.begin(), ib.end()) ? -1 : 1);
            CHECK(ikey_order == compare_ex(win32::NORM_IGNORECASE, a, b) - 2);
            ++checked;
        }
    }
    CHECK(checked == 18 * 18);
    // Size-query protocol of the sort key: bytes; a buffer that is too small fails.
    char16_t tiny[1];
    CHECK(pg6_win32_lc_map_string_ex(nullptr, win32::LCMAP_SORTKEY, u"abc", 3, tiny, 1, nullptr, nullptr, 0) == 0);
    CHECK(last_error() == 122);
    // SORTKEY is not combinable with the case mappings.
    CHECK(pg6_win32_lc_map_string_ex(nullptr, win32::LCMAP_SORTKEY | win32::LCMAP_UPPERCASE, u"abc", 3, tiny, 1, nullptr, nullptr, 0) == 0);
    CHECK(last_error() == 1004);

    // CompareStringEx itself.
    CHECK(compare_ex(0, u"abc", u"abc") == win32::CSTR_EQUAL);
    CHECK(compare_ex(0, u"abc", u"abd") == win32::CSTR_LESS_THAN);
    CHECK(compare_ex(0, u"b", u"abc") == win32::CSTR_GREATER_THAN);
    CHECK(compare_ex(win32::NORM_IGNORECASE, u"HeLLo", u"hello") == win32::CSTR_EQUAL);
    CHECK(compare_ex(0, u"HeLLo", u"hello") != win32::CSTR_EQUAL);
    CHECK(compare_ex(win32::NORM_IGNORECASE, u"\u00C9COLE", u"\u00E9cole") == win32::CSTR_EQUAL);
    // Agrees with CompareStringW and handles NUL-terminated (-1) counts.
    CHECK(pg6_win32_compare_string_ex(nullptr, 0, u"abc", -1, u"abc", 3, nullptr, nullptr, 0) == win32::CSTR_EQUAL);
    CHECK(pg6_win32_compare_string_ex(u"", 0, u"abc", -1, u"abd", -1, nullptr, nullptr, 0) ==
          pg6_win32_compare_string_w(0x409, 0, u"abc", -1, u"abd", -1));
    // Errors.
    CHECK(pg6_win32_compare_string_ex(nullptr, 0, nullptr, 1, u"a", 1, nullptr, nullptr, 0) == 0);
    CHECK(last_error() == win32::ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_compare_string_ex(u"not-a-locale", 0, u"a", 1, u"a", 1, nullptr, nullptr, 0) == 0);
    CHECK(last_error() == win32::ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_compare_string_ex(nullptr, 0x40000000u, u"a", 1, u"a", 1, nullptr, nullptr, 0) == 0);
    CHECK(last_error() == 1004);
    int dummy = 0;
    CHECK(pg6_win32_compare_string_ex(nullptr, 0, u"a", 1, u"a", 1, nullptr, &dummy, 0) == 0);
    CHECK(pg6_win32_compare_string_ex(nullptr, 0, u"a", 1, u"a", 1, nullptr, nullptr, 5) == 0);
}

std::u16string locale_info(const char16_t* name, uint32_t type, int32_t* returned = nullptr) {
    const int32_t n = pg6_win32_get_locale_info_ex(name, type, nullptr, 0);
    if (returned != nullptr) {
        *returned = n;
    }
    std::u16string out(static_cast<size_t>(n > 0 ? n : 0), u'\x7777');
    if (n > 0) {
        CHECK(pg6_win32_get_locale_info_ex(name, type, out.data(), n) == n);
        CHECK(out.back() == 0);
        out.pop_back();
    }
    return out;
}

void test_get_locale_info_ex() {
    int32_t n = 0;
    CHECK(locale_info(u"en-US", win32::LOCALE_SNAME, &n) == u"en-US");
    CHECK(n == 6);
    // Consistent with GetUserDefaultLocaleName.
    char16_t user_default[16];
    CHECK(pg6_win32_get_user_default_locale_name(user_default, 16) > 0);
    CHECK(locale_info(user_default, win32::LOCALE_SNAME) == user_default);
    CHECK(locale_info(nullptr, win32::LOCALE_SNAME) == u"en-US");
    CHECK(locale_info(u"", 0x0E) == u".");
    CHECK(locale_info(u"EN-us", 0x0F) == u",");
    CHECK(locale_info(u"en-US", 0x10) == u"3;0");
    CHECK(locale_info(u"en-US", 0x1F) == u"M/d/yyyy");
    CHECK(locale_info(u"en-US", 0x20) == u"dddd, MMMM d, yyyy");
    CHECK(locale_info(u"en-US", 0x1003) == u"h:mm:ss tt");
    CHECK(locale_info(u"en-US", 0x28) == u"AM");
    CHECK(locale_info(u"en-US", 0x59) == u"en");
    CHECK(locale_info(u"en-US", 0x5A) == u"US");
    CHECK(locale_info(u"en-US", 0x1001) == u"English");
    CHECK(locale_info(u"en-US", 0x1002) == u"United States");
    CHECK(locale_info(u"en-US", 0x14) == u"$");
    CHECK(locale_info(u"en-US", 0x38) == u"January");
    CHECK(locale_info(u"en-US", 0x3A) == u"March");
    CHECK(locale_info(u"en-US", 0x43) == u"December");
    CHECK(locale_info(u"en-US", 0x44) == u"Jan");
    CHECK(locale_info(u"en-US", 0x2A) == u"Monday");  // day names start at Monday
    CHECK(locale_info(u"en-US", 0x30) == u"Sunday");
    CHECK(locale_info(u"en-US", 0x31) == u"Mon");
    CHECK(locale_info(u"en-US", 0x37) == u"Sun");
    CHECK(locale_info(u"en-US", 0x50).empty());  // empty string: size 1, success
    CHECK(pg6_win32_get_locale_info_ex(u"en-US", 0x50, nullptr, 0) == 1);
    // LOCALE_RETURN_NUMBER.
    uint32_t number = 0;
    CHECK(pg6_win32_get_locale_info_ex(u"en-US", 0x1004 | win32::LOCALE_RETURN_NUMBER, reinterpret_cast<char16_t*>(&number), 2) == 2);
    CHECK(number == 1252);
    CHECK(pg6_win32_get_locale_info_ex(u"en-US", 0x1004 | win32::LOCALE_RETURN_NUMBER, nullptr, 0) == 2);
    CHECK(pg6_win32_get_locale_info_ex(u"en-US", 0x1 | win32::LOCALE_RETURN_NUMBER, reinterpret_cast<char16_t*>(&number), 2) == 2);
    CHECK(number == 0x409);
    CHECK(pg6_win32_get_locale_info_ex(u"en-US", 0x0E | win32::LOCALE_RETURN_NUMBER, reinterpret_cast<char16_t*>(&number), 2) == 0);
    CHECK(last_error() == 1004);  // not a numeric item
    // Buffer protocol.
    char16_t small[3];
    CHECK(pg6_win32_get_locale_info_ex(u"en-US", 0x5C, small, 3) == 0 && last_error() == 122);
    CHECK(pg6_win32_get_locale_info_ex(u"en-US", 0x5C, nullptr, 3) == 0 && last_error() == win32::ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_get_locale_info_ex(u"en-US", 0x5C, small, -1) == 0);
    // Unsupported locale / unknown item.
    CHECK(pg6_win32_get_locale_info_ex(u"de-DE", 0x5C, nullptr, 0) == 0 && last_error() == win32::ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_get_locale_info_ex(u"en-US", 0x7777, nullptr, 0) == 0 && last_error() == 1004);
    // The date/time patterns agree with the O15 formatting shims.
    char16_t formatted[64];
    Pg6SystemTime st{};
    st.year = 2024;
    st.month = 3;
    st.day = 5;
    CHECK(pg6_win32_get_date_format_ex(u"en-US", 0, &st, nullptr, formatted, 64, nullptr) > 0);
    CHECK(std::u16string(formatted) == u"3/5/2024");
}

void test_get_string_type() {
    const std::u16string text = u"aA1 ,\t\u001F\u00E9\u00C9\u0416";
    std::vector<uint16_t> t1(text.size());
    CHECK(pg6_win32_get_string_type_w(win32::CT_CTYPE1, text.c_str(), static_cast<int32_t>(text.size()), t1.data()) == 1);
    CHECK(t1[0] == 0x382);  // lower alpha xdigit defined
    CHECK(t1[1] == 0x381);
    CHECK(t1[2] == 0x284);
    CHECK(t1[3] == 0x248);
    CHECK(t1[4] == 0x210);
    CHECK(t1[5] == 0x268);
    CHECK(t1[6] == 0x220);
    CHECK(t1[7] == 0x302);  // e acute: lower alpha
    CHECK(t1[8] == 0x301);  // E acute: upper alpha
    CHECK(t1[9] == 0x301);  // Zhe
    // -1 counts the terminator as an element.
    std::vector<uint16_t> with_nul(4, 0xFFFF);
    CHECK(pg6_win32_get_string_type_w(win32::CT_CTYPE1, u"ab", -1, with_nul.data()) == 1);
    CHECK(with_nul[2] == 0x220 && with_nul[3] == 0xFFFF);
    // Every Latin-1 code unit: alpha iff the classic letters, upper/lower exclusive.
    for (char16_t c = 0; c < 0x100; ++c) {
        uint16_t type = 0;
        CHECK(pg6_win32_get_string_type_w(win32::CT_CTYPE1, &c, 1, &type) == 1);
        const bool alpha = (type & 0x100) != 0;
        const bool latin_letter = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == 0xAA || c == 0xB5 || c == 0xBA ||
                                  (c >= 0xC0 && c != 0xD7 && c != 0xF7);
        CHECK(alpha == latin_letter);
        CHECK(((type & 1) != 0 && (type & 2) != 0) == false);
        CHECK((type & 0x200) != 0);
        if (c < 0x80) {
            CHECK(((type & 0x04) != 0) == (c >= '0' && c <= '9'));
            CHECK(((type & 0x08) != 0) == ((c >= 9 && c <= 13) || c == ' '));
        }
    }
    // CTYPE2 and CTYPE3.
    uint16_t t2[5];
    CHECK(pg6_win32_get_string_type_w(win32::CT_CTYPE2, u"a1 ,#", 5, t2) == 1);
    CHECK(t2[0] == 1 && t2[1] == 3 && t2[2] == 10 && t2[3] == 7 && t2[4] == 5);
    uint16_t t3[3];
    CHECK(pg6_win32_get_string_type_w(win32::CT_CTYPE3, u"a\u0301$", 3, t3) == 1);
    CHECK(t3[0] == 0x8400);
    CHECK((t3[1] & 0x3) == 0x3);
    CHECK((t3[2] & 0x8) == 0x8);
    // Arguments.
    uint16_t out = 0;
    CHECK(pg6_win32_get_string_type_w(7, u"a", 1, &out) == 0 && last_error() == win32::ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_get_string_type_w(0, u"a", 1, &out) == 0);
    CHECK(pg6_win32_get_string_type_w(win32::CT_CTYPE1, nullptr, 1, &out) == 0);
    CHECK(pg6_win32_get_string_type_w(win32::CT_CTYPE1, u"a", 1, nullptr) == 0);
    CHECK(pg6_win32_get_string_type_w(win32::CT_CTYPE1, u"a", -2, &out) == 0);
}

void test_get_cp_info() {
    Pg6CpInfo info;
    std::memset(&info, 0xEE, sizeof(info));
    CHECK(pg6_win32_get_cp_info(1252, &info) == 1);
    CHECK(info.max_char_size == 1 && info.default_char[0] == '?' && info.default_char[1] == 0);
    for (const uint8_t b : info.lead_byte) {
        CHECK(b == 0);
    }
    CHECK(pg6_win32_get_cp_info(win32::CP_ACP, &info) == 1 && info.max_char_size == 1);
    CHECK(pg6_win32_get_cp_info(win32::CP_THREAD_ACP, &info) == 1 && info.max_char_size == 1);
    CHECK(pg6_win32_get_cp_info(20127, &info) == 1 && info.max_char_size == 1);
    CHECK(pg6_win32_get_cp_info(28591, &info) == 1 && info.max_char_size == 1);
    CHECK(pg6_win32_get_cp_info(65001, &info) == 1);
    CHECK(info.max_char_size == 4 && info.default_char[0] == '?');
    std::memset(&info, 0xEE, sizeof(info));
    CHECK(pg6_win32_get_cp_info(932, &info) == 0 && last_error() == win32::ERROR_INVALID_PARAMETER);
    CHECK(info.max_char_size == 0 && info.default_char[0] == 0);  // cleared on failure
    CHECK(pg6_win32_get_cp_info(win32::CP_OEMCP, &info) == 0);
    CHECK(pg6_win32_get_cp_info(1252, nullptr) == 0 && last_error() == win32::ERROR_INVALID_PARAMETER);
    // Consistent with MultiByteToWideChar: every code page GetCPInfo accepts converts.
    for (const uint32_t cp : {0u, 3u, 1252u, 20127u, 28591u, 65001u}) {
        CHECK(pg6_win32_multi_byte_to_wide_char(cp, 0, "a", 1, nullptr, 0) == 1);
    }
}

// ---------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------

bool write_file(const char16_t* path, const void* data, size_t size) {
    const Pg6Handle h = pg6_win32_create_file_w(path, win32::GENERIC_WRITE, 0, nullptr, win32::CREATE_ALWAYS,
                                                win32::FILE_ATTRIBUTE_NORMAL, 0);
    if (h == win32::INVALID_HANDLE_VALUE) {
        return false;
    }
    uint32_t written = 0;
    const bool ok = pg6_win32_write_file(h, data, static_cast<uint32_t>(size), &written, nullptr) == 1 && written == size;
    pg6_win32_close_handle(h);
    return ok;
}

std::vector<uint8_t> read_file(const char16_t* path) {
    std::vector<uint8_t> out;
    const Pg6Handle h = pg6_win32_create_file_w(path, win32::GENERIC_READ, win32::FILE_SHARE_READ, nullptr, win32::OPEN_EXISTING,
                                                win32::FILE_ATTRIBUTE_NORMAL, 0);
    if (h == win32::INVALID_HANDLE_VALUE) {
        return out;
    }
    int64_t size = 0;
    pg6_win32_get_file_size_ex(h, &size);
    out.resize(static_cast<size_t>(size));
    uint32_t got = 0;
    if (size > 0) {
        pg6_win32_read_file(h, out.data(), static_cast<uint32_t>(size), &got, nullptr);
    }
    pg6_win32_close_handle(h);
    out.resize(got);
    return out;
}

void test_find_first_file_ex() {
    win32::clear_userdata_overlay();
    CHECK(write_file(u"T:\\userdata\\rtl\\a.txt", "aaaa", 4));
    CHECK(write_file(u"T:\\userdata\\rtl\\b.txt", "bb", 2));
    CHECK(write_file(u"T:\\userdata\\rtl\\c.dat", "c", 1));
    CHECK(pg6_win32_create_directory_w(u"T:\\userdata\\rtl\\sub", nullptr) == 1);

    for (const int32_t level : {win32::FindExInfoStandard, win32::FindExInfoBasic}) {
        Pg6Win32FindDataW data;
        std::memset(&data, 0xEE, sizeof(data));
        const Pg6Handle h = pg6_win32_find_first_file_ex_w(u"T:\\userdata\\rtl\\*.txt", level, &data,
                                                           win32::FindExSearchNameMatch, nullptr, 0);
        CHECK(h != win32::INVALID_HANDLE_VALUE);
        std::set<std::u16string> names;
        int guard = 0;
        do {
            names.insert(data.file_name);
            CHECK(data.alternate_file_name[0] == 0);
        } while (pg6_win32_find_next_file_w(h, &data) == 1 && ++guard < 10);
        CHECK(last_error() == win32::ERROR_NO_MORE_FILES);
        CHECK((names == std::set<std::u16string>{u"a.txt", u"b.txt"}));
        CHECK(pg6_win32_find_close(h) == 1);
    }
    // LimitToDirectories is accepted (a hint) and the directory is among the results.
    Pg6Win32FindDataW data;
    Pg6Handle h = pg6_win32_find_first_file_ex_w(u"T:\\userdata\\rtl\\su*", win32::FindExInfoBasic, &data,
                                                 win32::FindExSearchLimitToDirectories, nullptr, 0);
    CHECK(h != win32::INVALID_HANDLE_VALUE);
    CHECK(std::u16string(data.file_name) == u"sub" && (data.file_attributes & 0x10) != 0);
    pg6_win32_find_close(h);
    // Exact name, size reported, additional flags accepted.
    h = pg6_win32_find_first_file_ex_w(u"T:\\userdata\\rtl\\a.txt", win32::FindExInfoStandard, &data, 0, nullptr, 0x2 | 0x1);
    CHECK(h != win32::INVALID_HANDLE_VALUE && data.file_size_low == 4);
    pg6_win32_find_close(h);
    // No match.
    CHECK(pg6_win32_find_first_file_ex_w(u"T:\\userdata\\rtl\\*.zzz", 0, &data, 0, nullptr, 0) == win32::INVALID_HANDLE_VALUE);
    CHECK(last_error() == win32::ERROR_FILE_NOT_FOUND);
    // Validation: info level, search op, filter, flags, null args; find data is cleared on failure.
    std::memset(&data, 0xEE, sizeof(data));
    CHECK(pg6_win32_find_first_file_ex_w(u"T:\\userdata\\rtl\\*", 2, &data, 0, nullptr, 0) == win32::INVALID_HANDLE_VALUE);
    CHECK(last_error() == win32::ERROR_INVALID_PARAMETER);
    CHECK(data.file_name[0] == 0 && data.file_attributes == 0);
    CHECK(pg6_win32_find_first_file_ex_w(u"T:\\userdata\\rtl\\*", -1, &data, 0, nullptr, 0) == win32::INVALID_HANDLE_VALUE);
    CHECK(pg6_win32_find_first_file_ex_w(u"T:\\userdata\\rtl\\*", 0, &data, 2, nullptr, 0) == win32::INVALID_HANDLE_VALUE);
    CHECK(last_error() == win32::ERROR_NOT_SUPPORTED);
    CHECK(pg6_win32_find_first_file_ex_w(u"T:\\userdata\\rtl\\*", 0, &data, 3, nullptr, 0) == win32::INVALID_HANDLE_VALUE);
    CHECK(last_error() == win32::ERROR_INVALID_PARAMETER);
    int filter = 0;
    CHECK(pg6_win32_find_first_file_ex_w(u"T:\\userdata\\rtl\\*", 0, &data, 0, &filter, 0) == win32::INVALID_HANDLE_VALUE);
    CHECK(pg6_win32_find_first_file_ex_w(u"T:\\userdata\\rtl\\*", 0, &data, 0, nullptr, 0x100) == win32::INVALID_HANDLE_VALUE);
    CHECK(pg6_win32_find_first_file_ex_w(nullptr, 0, &data, 0, nullptr, 0) == win32::INVALID_HANDLE_VALUE);
    CHECK(pg6_win32_find_first_file_ex_w(u"T:\\userdata\\rtl\\*", 0, nullptr, 0, nullptr, 0) == win32::INVALID_HANDLE_VALUE);
    CHECK(last_error() == win32::ERROR_INVALID_PARAMETER);
    win32::clear_userdata_overlay();
}

void test_get_file_information_by_handle() {
    win32::clear_userdata_overlay();
    std::vector<uint8_t> payload(70000, 0x5A);
    CHECK(write_file(u"T:\\userdata\\info\\one.bin", payload.data(), payload.size()));
    CHECK(write_file(u"T:\\userdata\\info\\two.bin", "xy", 2));
    const Pg6Handle one = pg6_win32_create_file_w(u"T:\\userdata\\info\\one.bin", win32::GENERIC_READ, 1, nullptr, win32::OPEN_EXISTING, 0x80, 0);
    const Pg6Handle two = pg6_win32_create_file_w(u"T:\\userdata\\info\\two.bin", win32::GENERIC_READ, 1, nullptr, win32::OPEN_EXISTING, 0x80, 0);
    CHECK(one != win32::INVALID_HANDLE_VALUE && two != win32::INVALID_HANDLE_VALUE);
    Pg6ByHandleFileInformation a, b, again;
    std::memset(&a, 0xEE, sizeof(a));
    CHECK(pg6_win32_get_file_information_by_handle(one, &a) == 1);
    CHECK(pg6_win32_get_file_information_by_handle(two, &b) == 1);
    CHECK(pg6_win32_get_file_information_by_handle(one, &again) == 1);
    CHECK(a.file_size_high == 0 && a.file_size_low == 70000);
    CHECK(b.file_size_low == 2);
    CHECK((a.file_attributes & 0x10) == 0 && a.file_attributes != 0xEEEEEEEEu);
    CHECK(a.number_of_links == 1);
    CHECK(win32::file_time_ticks(a.creation_time) != 0 && win32::file_time_ticks(a.last_write_time) != 0);
    CHECK(a.volume_serial_number == b.volume_serial_number && a.volume_serial_number != 0);
    CHECK(a.file_index_high == again.file_index_high && a.file_index_low == again.file_index_low);
    CHECK(a.file_index_high != b.file_index_high || a.file_index_low != b.file_index_low);
    // Consistent with GetFileAttributesW and GetFileSizeEx.
    CHECK(a.file_attributes == pg6_win32_get_file_attributes_w(u"T:\\userdata\\info\\one.bin"));
    int64_t size = 0;
    CHECK(pg6_win32_get_file_size_ex(one, &size) == 1 && size == 70000);
    // Invalid handles and null pointers.
    CHECK(pg6_win32_get_file_information_by_handle(0x12345, &b) == 0);
    CHECK(last_error() == win32::ERROR_INVALID_HANDLE);
    CHECK(b.file_size_low == 0 && b.file_attributes == 0);  // zeroed on failure
    CHECK(pg6_win32_get_file_information_by_handle(win32::INVALID_HANDLE_VALUE, &b) == 0);
    CHECK(pg6_win32_get_file_information_by_handle(one, nullptr) == 0);
    CHECK(last_error() == win32::ERROR_INVALID_PARAMETER);
    pg6_win32_close_handle(one);
    pg6_win32_close_handle(two);
    CHECK(pg6_win32_get_file_information_by_handle(one, &b) == 0);  // closed
    win32::clear_userdata_overlay();
}

void test_get_temp_path() {
    win32::clear_userdata_overlay();
    const uint32_t needed = pg6_win32_get_temp_path_w(0, nullptr);
    CHECK(needed > 1);
    std::u16string path(needed, u'\x7777');
    const uint32_t written = pg6_win32_get_temp_path_w(needed, path.data());
    CHECK(written == needed - 1);  // length without the NUL when the buffer is large enough
    path.resize(written);
    CHECK(path.back() == u'\\');
    CHECK(path == u"T:\\userdata\\temp\\");
    // Too small a buffer returns the required size including the NUL and writes nothing.
    char16_t small[4] = {u'x', u'x', u'x', u'x'};
    CHECK(pg6_win32_get_temp_path_w(4, small) == needed);
    CHECK(small[0] == u'x');
    // The directory is writable: create, find, copy, delete a file there.
    const std::u16string file = path + u"scratch.tmp";
    CHECK(write_file(file.c_str(), "temp", 4));
    CHECK(read_file(file.c_str()) == std::vector<uint8_t>({'t', 'e', 'm', 'p'}));
    CHECK(pg6_win32_get_file_attributes_w(path.c_str()) != win32::INVALID_FILE_ATTRIBUTES);
    CHECK((pg6_win32_get_file_attributes_w(path.c_str()) & 0x10) != 0);
    CHECK(pg6_win32_delete_file_w(file.c_str()) == 1);
    win32::clear_userdata_overlay();
}

void test_copy_file2() {
    win32::clear_userdata_overlay();
    std::vector<uint8_t> big(200 * 1024 + 123);
    for (size_t i = 0; i < big.size(); ++i) {
        big[i] = static_cast<uint8_t>(i * 31 + (i >> 8));
    }
    CHECK(write_file(u"T:\\userdata\\copy\\src.bin", big.data(), big.size()));
    CHECK(pg6_win32_copy_file2(u"T:\\userdata\\copy\\src.bin", u"T:\\userdata\\copy\\dst.bin", nullptr) == 0);
    CHECK(read_file(u"T:\\userdata\\copy\\dst.bin") == big);
    CHECK(read_file(u"T:\\userdata\\copy\\src.bin") == big);
    // Overwrites by default, fails with FAIL_IF_EXISTS.
    CHECK(write_file(u"T:\\userdata\\copy\\small.bin", "z", 1));
    CHECK(pg6_win32_copy_file2(u"T:\\userdata\\copy\\small.bin", u"T:\\userdata\\copy\\dst.bin", nullptr) == 0);
    CHECK(read_file(u"T:\\userdata\\copy\\dst.bin") == std::vector<uint8_t>({'z'}));
    uint32_t params[8] = {32, win32::COPY_FILE_FAIL_IF_EXISTS, 0, 0, 0, 0, 0, 0};
    CHECK(pg6_win32_copy_file2(u"T:\\userdata\\copy\\src.bin", u"T:\\userdata\\copy\\dst.bin", params) ==
          win32::hresult_from_win32(win32::ERROR_FILE_EXISTS));
    CHECK(read_file(u"T:\\userdata\\copy\\dst.bin") == std::vector<uint8_t>({'z'}));  // untouched
    CHECK(pg6_win32_copy_file2(u"T:\\userdata\\copy\\src.bin", u"T:\\userdata\\copy\\new.bin", params) == 0);
    CHECK(read_file(u"T:\\userdata\\copy\\new.bin") == big);
    // Empty file.
    CHECK(write_file(u"T:\\userdata\\copy\\empty.bin", "", 0));
    CHECK(pg6_win32_copy_file2(u"T:\\userdata\\copy\\empty.bin", u"T:\\userdata\\copy\\empty2.bin", nullptr) == 0);
    CHECK(pg6_win32_get_file_attributes_w(u"T:\\userdata\\copy\\empty2.bin") != win32::INVALID_FILE_ATTRIBUTES);
    // Errors: missing source, same file, bad parameter block, null names, read-only destination.
    CHECK(pg6_win32_copy_file2(u"T:\\userdata\\copy\\missing.bin", u"T:\\userdata\\copy\\x.bin", nullptr) ==
          win32::hresult_from_win32(win32::ERROR_FILE_NOT_FOUND));
    CHECK(pg6_win32_get_file_attributes_w(u"T:\\userdata\\copy\\x.bin") == win32::INVALID_FILE_ATTRIBUTES);
    CHECK(pg6_win32_copy_file2(u"T:\\userdata\\copy\\src.bin", u"t:\\USERDATA\\copy\\SRC.bin", nullptr) ==
          win32::hresult_from_win32(32));
    CHECK(read_file(u"T:\\userdata\\copy\\src.bin") == big);  // not truncated
    uint32_t bad_size[8] = {16, 0, 0, 0, 0, 0, 0, 0};
    CHECK(pg6_win32_copy_file2(u"T:\\userdata\\copy\\src.bin", u"T:\\userdata\\copy\\y.bin", bad_size) ==
          win32::hresult_from_win32(win32::ERROR_INVALID_PARAMETER));
    uint32_t bad_flags[8] = {32, 0x80000000u, 0, 0, 0, 0, 0, 0};
    CHECK(pg6_win32_copy_file2(u"T:\\userdata\\copy\\src.bin", u"T:\\userdata\\copy\\y.bin", bad_flags) ==
          win32::hresult_from_win32(win32::ERROR_INVALID_PARAMETER));
    CHECK(pg6_win32_copy_file2(nullptr, u"T:\\userdata\\copy\\y.bin", nullptr) < 0);
    CHECK(pg6_win32_copy_file2(u"T:\\userdata\\copy\\src.bin", nullptr, nullptr) < 0);
    CHECK(pg6_win32_copy_file2(u"T:\\userdata\\copy\\src.bin", u"G:\\gamedata\\readonly.bin", nullptr) < 0);
    win32::clear_userdata_overlay();
}

void test_links() {
    win32::clear_userdata_overlay();
    CHECK(write_file(u"T:\\userdata\\link\\target.txt", "t", 1));
    // Hard links: not supported; a missing target reports its own error first.
    CHECK(pg6_win32_create_hard_link_w(u"T:\\userdata\\link\\h.txt", u"T:\\userdata\\link\\target.txt", nullptr) == 0);
    CHECK(last_error() == win32::ERROR_NOT_SUPPORTED);
    CHECK(pg6_win32_create_hard_link_w(u"T:\\userdata\\link\\h.txt", u"T:\\userdata\\link\\none.txt", nullptr) == 0);
    CHECK(last_error() == win32::ERROR_FILE_NOT_FOUND);
    CHECK(pg6_win32_create_hard_link_w(nullptr, u"T:\\userdata\\link\\target.txt", nullptr) == 0);
    CHECK(pg6_win32_create_hard_link_w(u"", u"T:\\userdata\\link\\target.txt", nullptr) == 0);
    CHECK(pg6_win32_get_file_attributes_w(u"T:\\userdata\\link\\h.txt") == win32::INVALID_FILE_ATTRIBUTES);
    // Symbolic links: privilege error like an unprivileged Windows process.
    CHECK(pg6_win32_create_symbolic_link_w(u"T:\\userdata\\link\\s.txt", u"target.txt", 0) == 0);
    CHECK(last_error() == win32::ERROR_PRIVILEGE_NOT_HELD);
    CHECK(pg6_win32_create_symbolic_link_w(u"T:\\userdata\\link\\s", u"dir", win32::SYMBOLIC_LINK_FLAG_DIRECTORY) == 0);
    CHECK(last_error() == win32::ERROR_PRIVILEGE_NOT_HELD);
    CHECK(pg6_win32_create_symbolic_link_w(u"T:\\userdata\\link\\s.txt", u"target.txt",
                                           win32::SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE) == 0);
    CHECK(last_error() == win32::ERROR_NOT_SUPPORTED);
    CHECK(pg6_win32_create_symbolic_link_w(u"T:\\userdata\\link\\s.txt", u"target.txt", 0x80) == 0);
    CHECK(last_error() == win32::ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_create_symbolic_link_w(nullptr, u"x", 0) == 0);
    CHECK(pg6_win32_create_symbolic_link_w(u"x", nullptr, 0) == 0);
    CHECK(pg6_win32_get_file_attributes_w(u"T:\\userdata\\link\\s.txt") == win32::INVALID_FILE_ATTRIBUTES);
    win32::clear_userdata_overlay();
}

// ---------------------------------------------------------------------------
// Thread pool with guest callbacks
// ---------------------------------------------------------------------------

void test_thread_pool_guest() {
    Fixture fx;
    {
        std::lock_guard<std::mutex> lock(g_work_mutex);
        g_works.clear();
    }
    // Missing callback fails the way the host shim does.
    CHECK(fx.call(&pg6_gthunk_create_threadpool_work, {0, 0x42, 0}) == 0);
    CHECK(last_error() == win32::ERROR_INVALID_PARAMETER);
    CHECK(rtl::live_pool_records() == 0);

    const uint64_t work = fx.call(&pg6_gthunk_create_threadpool_work, {kWorkCb, 0xC0DE, 0});
    CHECK(work != 0);
    CHECK(rtl::live_pool_records() == 1);
    for (int i = 0; i < 3; ++i) {
        fx.call(&pg6_gthunk_submit_threadpool_work, {work});
    }
    fx.call(&pg6_gthunk_wait_for_threadpool_work_callbacks, {work, 0});
    {
        std::lock_guard<std::mutex> lock(g_work_mutex);
        CHECK(g_works.size() == 3);
        for (const auto& r : g_works) {
            CHECK(r.context == 0xC0DE);
            CHECK(r.work == work);
            CHECK(r.instance != 0);
            CHECK(r.thread != std::this_thread::get_id());
        }
    }
    CHECK(com::bound_guest_state() == nullptr);
    // Close: record released, later use of the stale handle is diagnosed and ignored.
    fx.call(&pg6_gthunk_close_threadpool_work, {work});
    CHECK(rtl::live_pool_records() == 0);
    com::clear_diagnostics();
    fx.call(&pg6_gthunk_submit_threadpool_work, {work});
    fx.call(&pg6_gthunk_wait_for_threadpool_work_callbacks, {work, 1});
    fx.call(&pg6_gthunk_close_threadpool_work, {work});
    fx.call(&pg6_gthunk_submit_threadpool_work, {0x1234});  // forged
    CHECK(com::diagnostics().size() == 4);
    {
        std::lock_guard<std::mutex> lock(g_work_mutex);
        CHECK(g_works.size() == 3);  // nothing ran
    }

    // Cancel: queued-but-not-started submissions are dropped by a waiting cancel (best effort:
    // everything that did run was counted once), the record is still released by Close.
    {
        std::lock_guard<std::mutex> lock(g_work_mutex);
        g_works.clear();
    }
    const uint64_t cancel_work = fx.call(&pg6_gthunk_create_threadpool_work, {kWorkCb, 0x77, 0});
    for (int i = 0; i < 50; ++i) {
        fx.call(&pg6_gthunk_submit_threadpool_work, {cancel_work});
    }
    fx.call(&pg6_gthunk_wait_for_threadpool_work_callbacks, {cancel_work, 1});
    fx.call(&pg6_gthunk_close_threadpool_work, {cancel_work});
    CHECK(rtl::live_pool_records() == 0);
    {
        std::lock_guard<std::mutex> lock(g_work_mutex);
        CHECK(g_works.size() <= 50);
    }

    // Close while callbacks are still pending: they still run, then the record goes away.
    {
        std::lock_guard<std::mutex> lock(g_work_mutex);
        g_works.clear();
    }
    const uint64_t closing = fx.call(&pg6_gthunk_create_threadpool_work, {kWorkCb, 0x99, 0});
    for (int i = 0; i < 20; ++i) {
        fx.call(&pg6_gthunk_submit_threadpool_work, {closing});
    }
    fx.call(&pg6_gthunk_close_threadpool_work, {closing});
    CHECK(wait_for([] {
        std::lock_guard<std::mutex> lock(g_work_mutex);
        return g_works.size() == 20;
    }));
    CHECK(rtl::live_pool_records() == 0);

    // TrySubmitThreadpoolCallback.
    g_simple_calls = 0;
    CHECK(fx.call(&pg6_gthunk_try_submit_threadpool_callback, {kSimpleCb, 0x5150, 0}) == 1);
    CHECK(wait_for([] { return g_simple_calls.load() == 1; }));
    CHECK(g_simple_ctx == 0x5150);
    CHECK(wait_for([] { return rtl::live_pool_records() == 0; }));
    CHECK(fx.call(&pg6_gthunk_try_submit_threadpool_callback, {0, 0, 0}) == 0);
    CHECK(last_error() == win32::ERROR_INVALID_PARAMETER);
    CHECK(rtl::live_pool_records() == 0);

    // FreeLibraryWhenCallbackReturns only records the request.
    const uint64_t before = win32::free_library_when_callback_returns_count();
    pg6_win32_free_library_when_callback_returns(0x1, 0x2);
    CHECK(win32::free_library_when_callback_returns_count() == before + 1);
    win32::shutdown_thread_pool();
}

// O32: CloseThreadpoolWork while the last outstanding callback returns. Both sides used to decide
// "closed and nothing pending" separately, and when the callback's decrement fell between Close
// marking the record closed and Close checking it, both deleted the record: heap corruption,
// about one run in twenty of this suite. Each record must be deleted exactly once.
void test_thread_pool_close_races_last_callback() {
    Fixture fx;
    const int64_t before = rtl::live_work_records();
    for (int i = 0; i < 5000; ++i) {
        g_race_started.store(0, std::memory_order_relaxed);
        const uint64_t work = fx.call(&pg6_gthunk_create_threadpool_work, {kRaceCb, 0, 0});
        fx.call(&pg6_gthunk_submit_threadpool_work, {work});
        while (g_race_started.load(std::memory_order_acquire) == 0) {
            std::this_thread::yield();
        }
        fx.call(&pg6_gthunk_close_threadpool_work, {work});
        CHECK(rtl::live_work_records() >= before);  // never deleted twice
    }
    CHECK(wait_for([&] { return rtl::live_work_records() == before; }));
    CHECK(rtl::live_pool_records() == 0);
    win32::shutdown_thread_pool();
}

// ---------------------------------------------------------------------------
// RtlCaptureStackBackTrace
// ---------------------------------------------------------------------------

void test_capture_stack_back_trace() {
    Fixture fx;
    constexpr uint64_t kImageSize = 0x10000;
    const uint64_t kImage = fx.arena.reserve(0, kImageSize, PageProtect::ReadWrite);
    CHECK(kImage != 0);
    CHECK(fx.arena.commit(kImage, kImageSize, PageProtect::ReadWrite));
    // One RUNTIME_FUNCTION (a function far away from the frames below) and its empty UNWIND_INFO.
    const uint32_t function[3] = {0x8000, 0x8100, 0x9000};
    CHECK(fx.arena.write(kImage + 0xA000, function, sizeof(function)));
    const uint8_t unwind_info[4] = {1, 0, 0, 0};
    CHECK(fx.arena.write(kImage + 0x9000, unwind_info, sizeof(unwind_info)));
    CHECK(register_image_function_table(fx.arena, kImage, kImageSize, kImage + 0xA000, 1));

    const uint64_t r1 = kImage + 0x1111;
    const uint64_t r2 = kImage + 0x2222;
    const uint64_t r3 = kImage + 0x3333;
    const uint64_t outside = 0x12345;
    const uint64_t out_buffer = fx.alloc(0x1000);
    const uint64_t out_hash = out_buffer + 0x800;

    // The stack the thunk is entered with: [rsp] = return address into the caller (frame 0),
    // above it the return addresses of the callers (leaf frames: no unwind data).
    auto call = [&](uint32_t skip, uint32_t count, uint64_t buffer, uint64_t hash) {
        GuestState& s = fx.state();
        const uint64_t entry_rsp = fx.thread.stack_top - 0x200;
        gpr(&s, GuestReg::Rsp) = entry_rsp;
        const uint64_t stack[] = {r1, r2, r3, outside, 0};
        CHECK(fx.arena.write(entry_rsp, stack, sizeof(stack)));
        gpr(&s, GuestReg::Rcx) = skip;
        gpr(&s, GuestReg::Rdx) = count;
        gpr(&s, GuestReg::R8) = buffer;
        gpr(&s, GuestReg::R9) = hash;
        pg6_gthunk_rtl_capture_stack_back_trace(&s);
        CHECK(gpr(&s, GuestReg::Rsp) == entry_rsp + 8);  // popped like `ret`
        return gpr(&s, GuestReg::Rax) & 0xFFFF;
    };
    CHECK(call(0, 10, out_buffer, out_hash) == 3);  // r1, r2, r3; the walk ends outside the image
    CHECK(fx.read<uint64_t>(out_buffer) == r1);
    CHECK(fx.read<uint64_t>(out_buffer + 8) == r2);
    CHECK(fx.read<uint64_t>(out_buffer + 16) == r3);
    CHECK(fx.read<uint32_t>(out_hash) == static_cast<uint32_t>(r1 + r2 + r3));
    CHECK(call(1, 10, out_buffer, out_hash) == 2);
    CHECK(fx.read<uint64_t>(out_buffer) == r2 && fx.read<uint64_t>(out_buffer + 8) == r3);
    CHECK(call(0, 2, out_buffer, 0) == 2);  // capped, null hash pointer accepted
    CHECK(call(3, 10, out_buffer, out_hash) == 0);  // everything skipped
    CHECK(fx.read<uint32_t>(out_hash) == 0);
    CHECK(call(0, 0, out_buffer, out_hash) == 0);
    CHECK(call(0, 5, 0, out_hash) == 0);  // null buffer
    CHECK(last_error() == win32::ERROR_INVALID_PARAMETER);
    CHECK(call(0, 5, 0x10, out_hash) == 0);  // unmapped buffer
    // The host form (no guest stack) captures nothing and clears the hash.
    uint32_t hash = 0xFFFF;
    void* frames[4];
    CHECK(pg6_win32_rtl_capture_stack_back_trace(0, 4, frames, &hash) == 0 && hash == 0);
    // Frames outside registered images end the walk before the first frame.
    {
        GuestState& s = fx.state();
        const uint64_t entry_rsp = fx.thread.stack_top - 0x200;
        gpr(&s, GuestReg::Rsp) = entry_rsp;
        const uint64_t stack[] = {outside, r1, 0};
        CHECK(fx.arena.write(entry_rsp, stack, sizeof(stack)));
        gpr(&s, GuestReg::Rcx) = 0;
        gpr(&s, GuestReg::Rdx) = 4;
        gpr(&s, GuestReg::R8) = out_buffer;
        gpr(&s, GuestReg::R9) = 0;
        pg6_gthunk_rtl_capture_stack_back_trace(&s);
        CHECK((gpr(&s, GuestReg::Rax) & 0xFFFF) == 0);
    }
}

}  // namespace

#define RUN(fn)                                \
    do {                                       \
        std::fprintf(stderr, "[run] " #fn "\n"); \
        fn();                                  \
    } while (false)

int main() {
    RUN(test_critical_section_ex);
    RUN(test_pointer_encoding);
    RUN(test_switch_to_thread);
    RUN(test_fls_basic);
    RUN(test_fls_destructors_host);
    RUN(test_fls_exit_passes);
    RUN(test_fls_guest_destructors);
    RUN(test_init_once_host);
    RUN(test_init_once_guest);
    RUN(test_slist_basic);
    RUN(test_slist_stress);
    RUN(test_get_module_handle_ex);
    RUN(test_lc_map_string);
    RUN(test_sort_key_and_compare);
    RUN(test_get_locale_info_ex);
    RUN(test_get_string_type);
    RUN(test_get_cp_info);
    RUN(test_find_first_file_ex);
    RUN(test_get_file_information_by_handle);
    RUN(test_get_temp_path);
    RUN(test_copy_file2);
    RUN(test_links);
    RUN(test_thread_pool_guest);
    RUN(test_thread_pool_close_races_last_callback);
    RUN(test_capture_stack_back_trace);
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::puts("test_win32_rtl: all checks passed");
    return 0;
}
