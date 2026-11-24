#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "abi/playgta6_abi.h"
#include "guest/com_bridge.h"
#include "guest/guest_callbacks.h"
#include "guest/guest_crt_exit.h"
#include "guest/guest_runtime.h"
#include "os/ucrt_bridge.h"
#include "os/vmem.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_guest_crt_exit.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6;
using namespace playgta6::guest;
namespace com = playgta6::guest::com;
namespace ce = playgta6::guest::crt_exit;
using playgta6::os::VirtualMemoryArena;

VirtualMemoryArena* g_arena = nullptr;

uint64_t load64(uint64_t va) {
    uint64_t value = 0;
    CHECK(g_arena->read(va, &value, sizeof(value)));
    return value;
}

void store64(uint64_t va, uint64_t value) {
    CHECK(g_arena->write(va, &value, sizeof(value)));
}

uint64_t& gpr(GuestState* s, uint32_t reg) {
    return s->gpr[reg];
}

void ret(GuestState* s) {
    gpr(s, GuestReg::Rsp) += 8;
}

// What compiled code does for `call [import]` with up to two integer arguments.
void guest_call(GuestState* s, uint64_t target, uint64_t return_address, uint64_t a0 = 0, uint64_t a1 = 0) {
    gpr(s, GuestReg::Rsp) -= 0x28;
    gpr(s, GuestReg::Rcx) = a0;
    gpr(s, GuestReg::Rdx) = a1;
    gpr(s, GuestReg::Rsp) -= 8;
    store64(gpr(s, GuestReg::Rsp), return_address);
    __playgta6_indirect_call(s, target);
    gpr(s, GuestReg::Rsp) += 0x28;
}

// ---------------------------------------------------------------------------
// Guest functions (hand-written "lifted" code)
// ---------------------------------------------------------------------------

constexpr uint64_t kHandlerA = 0x140030000ULL;
constexpr uint64_t kHandlerB = 0x140030100ULL;
constexpr uint64_t kHandlerC = 0x140030200ULL;
constexpr uint64_t kHandlerD = 0x140030300ULL;
constexpr uint64_t kRegistersD = 0x140030400ULL;  // _crt_atexit(D)
constexpr uint64_t kCallsExit = 0x140030500ULL;   // exit(g_nested_code)
constexpr uint64_t kCallsCexit = 0x140030600ULL;  // _cexit()
constexpr uint64_t kTlsCallback = 0x140030700ULL;
constexpr uint64_t kTlsCallback2 = 0x140030800ULL;

std::mutex g_order_mutex;
std::vector<char> g_order;
std::atomic<int> g_count{0};
uint64_t g_crt_atexit = 0;
uint64_t g_exit = 0;
uint64_t g_cexit = 0;
int32_t g_nested_code = 0;
uint64_t g_tls_args[3] = {~0ull, ~0ull, ~0ull};

void record(char c) {
    std::lock_guard<std::mutex> lock(g_order_mutex);
    g_order.push_back(c);
}

std::string order() {
    std::lock_guard<std::mutex> lock(g_order_mutex);
    return std::string(g_order.begin(), g_order.end());
}

void sub_a(GuestState* s) {
    record('A');
    g_count.fetch_add(1);
    ret(s);
}
void sub_b(GuestState* s) {
    record('B');
    g_count.fetch_add(1);
    ret(s);
}
void sub_c(GuestState* s) {
    record('C');
    ret(s);
}
void sub_d(GuestState* s) {
    record('D');
    ret(s);
}
void sub_registers_d(GuestState* s) {
    record('R');
    guest_call(s, g_crt_atexit, kRegistersD + 0x10, kHandlerD);
    ret(s);
}
void sub_calls_exit(GuestState* s) {
    record('X');
    guest_call(s, g_exit, kCallsExit + 0x10, static_cast<uint64_t>(g_nested_code));
    record('!');  // never reached: exit does not return
    ret(s);
}
void sub_calls_cexit(GuestState* s) {
    record('K');
    guest_call(s, g_cexit, kCallsCexit + 0x10);
    ret(s);
}
void sub_tls(GuestState* s) {
    record('T');
    g_tls_args[0] = gpr(s, GuestReg::Rcx);
    g_tls_args[1] = gpr(s, GuestReg::Rdx);
    g_tls_args[2] = gpr(s, GuestReg::R8);
    ret(s);
}
void sub_tls2(GuestState* s) {
    record('U');
    ret(s);
}

const LiftedFunctionEntry kFunctions[] = {
    {kHandlerA, sub_a},          {kHandlerB, sub_b},       {kHandlerC, sub_c},
    {kHandlerD, sub_d},          {kRegistersD, sub_registers_d}, {kCallsExit, sub_calls_exit},
    {kCallsCexit, sub_calls_cexit}, {kTlsCallback, sub_tls}, {kTlsCallback2, sub_tls2},
};

// ---------------------------------------------------------------------------
// Faults and process exit leave the guest by throwing
// ---------------------------------------------------------------------------

struct GuestFault {
    FaultInfo info;
};

void throwing_handler(const FaultInfo* info) {
    throw GuestFault{*info};
}

struct ProcessExit {
    int32_t code;
    int32_t kind;
};

std::atomic<int> g_exit_hook_calls{0};

void throwing_exit_hook(int32_t code, int32_t kind) {
    g_exit_hook_calls.fetch_add(1);
    throw ProcessExit{code, kind};
}

bool diag_contains(const char* needle) {
    for (const std::string& d : com::diagnostics()) {
        if (d.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

struct Thunks {
    uint64_t crt_atexit, crt_at_quick_exit, register_onexit, initialize_onexit, execute_onexit, register_tls,
        exit, quick_exit, underscore_exit, cexit, c_exit;
};

struct Fixture {
    VirtualMemoryArena arena;
    GuestThread thread{};
    Thunks t{};

    Fixture() {
        reset_guest_runtime();
        com::reset_com_bridge();
        ce::reset_crt_exit();
        pg6_crt_reset_runtime_state();
        com::set_guest_memory(&arena);
        set_fault_handler(throwing_handler);
        pg6_crt_set_exit_hook(&throwing_exit_hook);
        g_arena = &arena;
        CHECK(register_lifted_functions(kFunctions));
        CHECK(create_guest_thread(arena, 256 * 1024, thread));
        t.crt_atexit = register_host_callable(&pg6_gthunk_crt_atexit, "_crt_atexit");
        t.crt_at_quick_exit = register_host_callable(&pg6_gthunk_crt_at_quick_exit, "_crt_at_quick_exit");
        t.register_onexit = register_host_callable(&pg6_gthunk_register_onexit_function, "_register_onexit_function");
        t.initialize_onexit = register_host_callable(&pg6_gthunk_initialize_onexit_table, "_initialize_onexit_table");
        t.execute_onexit = register_host_callable(&pg6_gthunk_execute_onexit_table, "_execute_onexit_table");
        t.register_tls = register_host_callable(&pg6_gthunk_register_thread_local_exe_atexit_callback,
                                                "_register_thread_local_exe_atexit_callback");
        t.exit = register_host_callable(&pg6_gthunk_exit, "exit");
        t.quick_exit = register_host_callable(&pg6_gthunk_quick_exit, "quick_exit");
        t.underscore_exit = register_host_callable(&pg6_gthunk__exit, "_exit");
        t.cexit = register_host_callable(&pg6_gthunk__cexit, "_cexit");
        t.c_exit = register_host_callable(&pg6_gthunk__c_exit, "_c_exit");
        g_crt_atexit = t.crt_atexit;
        g_exit = t.exit;
        g_cexit = t.cexit;
        {
            std::lock_guard<std::mutex> lock(g_order_mutex);
            g_order.clear();
        }
        g_count = 0;
        g_exit_hook_calls = 0;
        g_tls_args[0] = g_tls_args[1] = g_tls_args[2] = ~0ull;
    }

    ~Fixture() {
        ce::reset_crt_exit();
        pg6_crt_reset_runtime_state();
        com::reset_com_bridge();
        destroy_guest_thread(arena, thread);
        com::set_guest_memory(nullptr);
        reset_guest_runtime();
        g_arena = nullptr;
    }

    GuestState& state() { return thread.state; }

    uint64_t call(uint64_t target, std::initializer_list<uint64_t> args = {}) {
        return call_guest(arena, thread.state, target, std::span<const uint64_t>(args.begin(), args.size()));
    }

    int32_t call32(uint64_t target, std::initializer_list<uint64_t> args = {}) {
        return static_cast<int32_t>(static_cast<uint32_t>(call(target, args)));
    }

    // Calls a guest exit function; returns the process exit the hook saw (code -1000 if none).
    ProcessExit call_exit(uint64_t target, std::initializer_list<uint64_t> args) {
        ProcessExit seen{-1000, -1};
        try {
            call(target, args);
        } catch (const ProcessExit& e) {
            seen = e;
        }
        thread.state.gpr[GuestReg::Rsp] = thread.stack_top;
        return seen;
    }

    ce::GuestOnexitTable table(uint64_t va) {
        ce::GuestOnexitTable t{};
        CHECK(com::read_guest(va, t));
        return t;
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void test_atexit_lifo_exactly_once() {
    Fixture fx;
    for (uint64_t fn : {kHandlerA, kHandlerB, kHandlerC}) {
        CHECK(fx.call32(fx.t.crt_atexit, {fn}) == 0);
    }
    CHECK(fx.call32(fx.t.crt_atexit, {0}) == 0);  // null: accepted, ignored
    const uint64_t process_table = ce::atexit_table();
    CHECK(process_table != 0);
    const ce::GuestOnexitTable before = fx.table(process_table);
    CHECK(before.last - before.first == 3 * 8);
    CHECK(load64(before.first) == kHandlerA && load64(before.first + 16) == kHandlerC);  // plain addresses

    const ProcessExit e = fx.call_exit(fx.t.exit, {7});
    CHECK(e.code == 7 && e.kind == PG6_CRT_EXIT_NORMAL);
    CHECK(order() == "CBA");
    CHECK(pg6_crt_get_last_exit_code() == 7);
    const ce::GuestOnexitTable after = fx.table(process_table);
    CHECK(after.first == 0 && after.last == 0 && after.end == 0);

    // Nothing runs twice: _cexit and a second exit find an empty table.
    fx.call(fx.t.cexit);
    const ProcessExit again = fx.call_exit(fx.t.exit, {8});
    CHECK(again.code == 8);
    CHECK(order() == "CBA");
    CHECK(g_exit_hook_calls == 2);
}

void test_reentrant_registration() {
    {
        Fixture fx;
        for (uint64_t fn : {kHandlerA, kRegistersD, kHandlerB}) {
            CHECK(fx.call32(fx.t.crt_atexit, {fn}) == 0);
        }
        fx.call_exit(fx.t.exit, {0});
        CHECK(order() == "BRDA");  // D, registered while R ran, runs right after it
    }
    {
        // Growth while executing: a handler registering past the buffer capacity.
        Fixture fx;
        for (int i = 0; i < 31; ++i) {
            CHECK(fx.call32(fx.t.crt_atexit, {kHandlerA}) == 0);
        }
        CHECK(fx.call32(fx.t.crt_atexit, {kRegistersD}) == 0);  // entry 32: the buffer is full
        fx.call_exit(fx.t.exit, {0});
        CHECK(order() == "RD" + std::string(31, 'A'));
    }
}

void test_exit_inside_handler() {
    {
        Fixture fx;
        g_nested_code = 9;
        for (uint64_t fn : {kHandlerA, kCallsExit, kHandlerB}) {
            CHECK(fx.call32(fx.t.crt_atexit, {fn}) == 0);
        }
        const ProcessExit e = fx.call_exit(fx.t.exit, {1});
        // The nested exit continues with the handlers not yet run, then terminates with its
        // code; the outer exit never resumes.
        CHECK(e.code == 9);
        CHECK(order() == "BXA");
        CHECK(g_exit_hook_calls == 1);
    }
    {
        // _cexit inside a handler: runs the remaining handlers and returns; the outer execution
        // sees the emptied table and stops.
        Fixture fx;
        for (uint64_t fn : {kHandlerA, kCallsCexit, kHandlerB}) {
            CHECK(fx.call32(fx.t.crt_atexit, {fn}) == 0);
        }
        const ProcessExit e = fx.call_exit(fx.t.exit, {2});
        CHECK(e.code == 2);
        CHECK(order() == "BKA");
    }
}

void test_onexit_tables() {
    Fixture fx;
    const uint64_t table = com::guest_alloc(sizeof(ce::GuestOnexitTable));
    // The guest's startup code marks tables it routes to the process list with ~0.
    const ce::GuestOnexitTable sentinel{ce::kTableSentinel, ce::kTableSentinel, ce::kTableSentinel};
    CHECK(com::write_guest(table, sentinel));
    CHECK(fx.call32(fx.t.initialize_onexit, {table}) == 0);
    ce::GuestOnexitTable t = fx.table(table);
    CHECK(t.first == 0 && t.last == 0 && t.end == 0);
    CHECK(fx.call32(fx.t.register_onexit, {table, kHandlerA}) == 0);
    CHECK(fx.call32(fx.t.register_onexit, {table, kHandlerB}) == 0);
    t = fx.table(table);
    CHECK(t.last - t.first == 16 && t.end - t.first == 32 * 8);
    CHECK(load64(t.first) == kHandlerA && load64(t.first + 8) == kHandlerB);
    // An initialized table is left alone.
    CHECK(fx.call32(fx.t.initialize_onexit, {table}) == 0);
    CHECK(fx.table(table).last == t.last);
    // Grows past the first buffer, keeping the entries.
    for (int i = 0; i < 38; ++i) {
        CHECK(fx.call32(fx.t.register_onexit, {table, (i % 2) ? kHandlerA : kHandlerB}) == 0);
    }
    t = fx.table(table);
    CHECK(t.last - t.first == 40 * 8 && t.end - t.first == 64 * 8);
    CHECK(load64(t.first) == kHandlerA && load64(t.first + 8) == kHandlerB);
    CHECK(fx.call32(fx.t.execute_onexit, {table}) == 0);
    std::string expected;
    for (int i = 37; i >= 0; --i) {
        expected += (i % 2) ? 'A' : 'B';
    }
    expected += "BA";
    CHECK(order() == expected);
    t = fx.table(table);
    CHECK(t.first == 0 && t.last == 0 && t.end == 0);
    CHECK(fx.call32(fx.t.execute_onexit, {table}) == 0);  // empty: nothing
    CHECK(order().size() == 40);
    // A sentinel table registers like an empty one.
    CHECK(com::write_guest(table, sentinel));
    CHECK(fx.call32(fx.t.execute_onexit, {table}) == 0);
    CHECK(fx.call32(fx.t.register_onexit, {table, kHandlerC}) == 0);
    CHECK(fx.call32(fx.t.execute_onexit, {table}) == 0);
    CHECK(order().back() == 'C');
    // Bad tables.
    com::clear_diagnostics();
    CHECK(fx.call32(fx.t.register_onexit, {0, kHandlerA}) == -1);
    CHECK(fx.call32(fx.t.initialize_onexit, {0x7000'0000'0000ULL}) == -1);
    CHECK(diag_contains("invalid _onexit_table_t"));
    const ce::GuestOnexitTable corrupt{0x1000, 0x800, 0x2000};
    CHECK(com::write_guest(table, corrupt));
    CHECK(fx.call32(fx.t.register_onexit, {table, kHandlerA}) == -1);
    CHECK(diag_contains("corrupt _onexit_table_t"));
    // Module tables are not run by exit (only the process list is).
    const ce::GuestOnexitTable empty{0, 0, 0};
    CHECK(com::write_guest(table, empty));
    CHECK(fx.call32(fx.t.register_onexit, {table, kHandlerD}) == 0);
    const size_t before = order().size();
    fx.call_exit(fx.t.exit, {0});
    CHECK(order().size() == before);
}

void test_thread_local_callback() {
    Fixture fx;
    CHECK(fx.call32(fx.t.crt_atexit, {kHandlerA}) == 0);
    fx.call(fx.t.register_tls, {kTlsCallback});
    com::clear_diagnostics();
    fx.call(fx.t.register_tls, {kTlsCallback2});
    CHECK(diag_contains("second thread-local exit callback"));
    CHECK(ce::thread_local_exit_callback() == kTlsCallback);
    // _cexit: thread-local destructors first (DLL_PROCESS_DETACH = 0), then atexit; returns.
    fx.call(fx.t.cexit);
    CHECK(order() == "TA");
    CHECK(g_tls_args[0] == 0 && g_tls_args[1] == 0 && g_tls_args[2] == 0);
    // The callback runs once per process.
    CHECK(fx.call32(fx.t.crt_atexit, {kHandlerB}) == 0);
    fx.call_exit(fx.t.exit, {0});
    CHECK(order() == "TAB");
}

void test_quick_exit_and_no_cleanup_exits() {
    Fixture fx;
    CHECK(fx.call32(fx.t.crt_atexit, {kHandlerA}) == 0);
    CHECK(fx.call32(fx.t.crt_at_quick_exit, {kHandlerC}) == 0);
    CHECK(fx.call32(fx.t.crt_at_quick_exit, {kHandlerD}) == 0);
    fx.call(fx.t.register_tls, {kTlsCallback});
    fx.call(fx.t.c_exit);  // no cleanup, returns
    CHECK(order().empty());
    ProcessExit e = fx.call_exit(fx.t.underscore_exit, {4});
    CHECK(e.code == 4 && e.kind == PG6_CRT_EXIT_QUICK);
    CHECK(order().empty());
    e = fx.call_exit(fx.t.quick_exit, {3});
    CHECK(e.code == 3 && e.kind == PG6_CRT_EXIT_QUICK);
    CHECK(order() == "DC");
    e = fx.call_exit(fx.t.exit, {5});
    CHECK(e.code == 5 && e.kind == PG6_CRT_EXIT_NORMAL);
    CHECK(order() == "DCTA");
}

void test_concurrent_registration() {
    Fixture fx;
    constexpr int kThreads = 8;
    constexpr int kPerThread = 200;
    std::vector<std::thread> threads;
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([i] {
            for (int k = 0; k < kPerThread; ++k) {
                CHECK(ce::register_function(ce::atexit_table(), (i + k) % 2 ? kHandlerA : kHandlerB) == 0);
            }
        });
    }
    for (std::thread& t : threads) {
        t.join();
    }
    const ce::GuestOnexitTable t = fx.table(ce::atexit_table());
    CHECK(t.last - t.first == uint64_t{kThreads} * kPerThread * 8);
    fx.call_exit(fx.t.exit, {0});
    CHECK(g_count == kThreads * kPerThread);
    CHECK(order().size() == static_cast<size_t>(kThreads * kPerThread));
}

}  // namespace

int main() {
    test_atexit_lifo_exactly_once();
    test_reentrant_registration();
    test_exit_inside_handler();
    test_onexit_tables();
    test_thread_local_callback();
    test_quick_exit_and_no_cleanup_exits();
    test_concurrent_registration();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_guest_crt_exit: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_guest_crt_exit: all tests passed\n");
    return 0;
}
