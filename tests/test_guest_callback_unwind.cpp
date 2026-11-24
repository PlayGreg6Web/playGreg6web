#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>

#include "abi/playgta6_abi.h"
#include "guest/com_bridge.h"
#include "guest/guest_callbacks.h"
#include "guest/guest_crt_exit.h"
#include "guest/guest_exceptions.h"
#include "guest/guest_runtime.h"
#include "os/ucrt_bridge.h"
#include "os/vmem.h"
#include "os/win32_kernel.h"
#include "os/xgameruntime.h"

// Guest exceptions raised inside guest callbacks that the runtime calls (O16 trampolines and
// guest-state thunks, O22 exit handlers). The "lifted" functions are hand-written in the shape the
// recompiler emits (real return addresses and prologs matching the synthetic .pdata, the resume
// protocol after every call), like tests/test_guest_exceptions.cpp, with a miniature language
// handler that catches one exception code by unwinding to a continuation of its function.
//
//   synchronous callbacks (qsort/bsearch comparators, _initterm entries): the exception reaches
//   the guest frame that called the import, which catches it; the import abandons its work.
//   top-level callbacks (XTaskQueue callbacks, exit handlers): the exception is unhandled even
//   when a guest frame with a matching handler made the runtime run the callback.

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_guest_callback_unwind.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6;
using namespace playgta6::guest;
namespace com = playgta6::guest::com;
namespace cbk = playgta6::guest::callbacks;
namespace ce = playgta6::guest::crt_exit;
using playgta6::os::PageProtect;
using playgta6::os::VirtualMemoryArena;

VirtualMemoryArena& arena() {
    return *os::win32::get_vmem_arena();
}

constexpr uint64_t kBase = 0x140000000ULL;
constexpr uint64_t kImageSize = 0x10000;
constexpr uint64_t va(uint64_t rva) {
    return kBase + rva;
}

// Function layout (RVAs).
constexpr uint64_t kTry = 0x1000;           // try { g_target(g_args...) } catch (kCodeCaught) -> kCont
constexpr uint64_t kTryRet = kTry + 0x15;   // return address of the import call
constexpr uint64_t kCont = kTry + 0x50;     // catch continuation
constexpr uint64_t kHCatch = 0x1100;        // language handler of kTry
constexpr uint64_t kHCatchRet = kHCatch + 0x40;
constexpr uint64_t kCompare = 0x1200;       // int compare(int*, int*): raises on call g_throw_at
constexpr uint64_t kCompareRet = kCompare + 0x20;
constexpr uint64_t kRaise = 0x1300;         // void raise(...): raises kCodeCaught
constexpr uint64_t kRaiseRet = kRaise + 0x10;
constexpr uint64_t kRecord = 0x1400;        // void record(): leaf, appends 'R'
constexpr uint64_t kFilter = 0x1500;        // top-level filter: leaf, logs, continue search

constexpr uint64_t kPdata = 0x8000;
constexpr uint64_t kUnwind = 0x9000;
constexpr uint64_t kData = 0xA000;

constexpr uint32_t kCodeCaught = 0xE0000002;

uint64_t g_raise_va = 0;
uint64_t g_unwind_ex_va = 0;
uint64_t g_target = 0;
uint64_t g_args[5] = {};
int g_compare_calls = 0;
int g_throw_at = 0;
std::string g_log;

uint64_t load64(uint64_t at) {
    uint64_t value = 0;
    CHECK(arena().read(at, &value, sizeof(value)));
    return value;
}

void store64(uint64_t at, uint64_t value) {
    CHECK(arena().write(at, &value, sizeof(value)));
}

uint64_t& r(GuestState* s, uint32_t reg) {
    return s->gpr[reg];
}

void push(GuestState* s, uint64_t value) {
    r(s, GuestReg::Rsp) -= 8;
    store64(r(s, GuestReg::Rsp), value);
}

uint64_t pop(GuestState* s) {
    const uint64_t value = load64(r(s, GuestReg::Rsp));
    r(s, GuestReg::Rsp) += 8;
    return value;
}

void ret(GuestState* s) {
    r(s, GuestReg::Rsp) += 8;
}

void call_import(GuestState* s, uint64_t target, uint64_t return_rva) {
    push(s, va(return_rva));
    __playgta6_indirect_call(s, target);
}

enum class After { Continue, Return, Resume };

After after_call(GuestState* s, uint64_t entry_rsp, std::initializer_list<uint64_t> targets) {
    if (s->unwind_pending == 0) {
        return After::Continue;
    }
    if (targets.size() == 0 || r(s, GuestReg::Rsp) >= entry_rsp) {
        return After::Return;
    }
    for (uint64_t t : targets) {
        if (va(t) == s->resume_rip) {
            s->unwind_pending = 0;
            return After::Resume;
        }
    }
    __playgta6_trap(s, s->resume_rip);
    return After::Return;
}

// try: push rbx; sub rsp, 30h | g_target(rcx..r9, [rsp+20h]) | normal: rax = 1 | kCont: rax += 200h
void sub_try(GuestState* s) {
    const uint64_t entry = r(s, GuestReg::Rsp);
    push(s, r(s, GuestReg::Rbx));
    r(s, GuestReg::Rsp) -= 0x30;
    r(s, GuestReg::Rbx) = 0xB0B0;
    r(s, GuestReg::Rcx) = g_args[0];
    r(s, GuestReg::Rdx) = g_args[1];
    r(s, GuestReg::R8) = g_args[2];
    r(s, GuestReg::R9) = g_args[3];
    store64(r(s, GuestReg::Rsp) + 0x20, g_args[4]);
    call_import(s, g_target, kTryRet);
    switch (after_call(s, entry, {kCont})) {
        case After::Return:
            return;
        case After::Continue:
            r(s, GuestReg::Rax) = 1;
            break;
        case After::Resume:
            g_log += 'K';
            r(s, GuestReg::Rax) += 0x200;
            break;
    }
    r(s, GuestReg::Rsp) += 0x30;
    r(s, GuestReg::Rbx) = pop(s);
    ret(s);
}

// Language handler of try: sub rsp, 0C8h | dispatch of kCodeCaught: RtlUnwindEx(establisher,
// kCont, record, 0x77) | unwind: log 'u'/'T'.
void sub_h_catch(GuestState* s) {
    const uint64_t entry = r(s, GuestReg::Rsp);
    r(s, GuestReg::Rsp) -= 0xC8;
    const uint64_t rsp = r(s, GuestReg::Rsp);
    const uint64_t record_va = r(s, GuestReg::Rcx);
    const uint64_t establisher = r(s, GuestReg::Rdx);
    GuestDispatcherContext dc{};
    CHECK(arena().read(r(s, GuestReg::R9), &dc, sizeof(dc)));
    GuestExceptionRecord rec{};
    CHECK(arena().read(record_va, &rec, sizeof(rec)));
    if ((rec.flags & kExceptionUnwinding) != 0) {
        g_log += (rec.flags & kExceptionTargetUnwind) != 0 ? 'T' : 'u';
    } else if (rec.code == kCodeCaught) {
        g_log += 'H';
        r(s, GuestReg::Rcx) = establisher;
        r(s, GuestReg::Rdx) = va(kCont);
        r(s, GuestReg::R8) = record_va;
        r(s, GuestReg::R9) = 0x77;
        store64(rsp + 0x20, dc.context_record);
        store64(rsp + 0x28, dc.history_table);
        call_import(s, g_unwind_ex_va, kHCatchRet);
        if (after_call(s, entry, {}) == After::Return) {
            return;
        }
        g_log += '!';
    }
    r(s, GuestReg::Rax) = kExceptionContinueSearch;
    r(s, GuestReg::Rsp) += 0xC8;
    ret(s);
}

// RaiseException(kCodeCaught, 0, 0, nullptr) from a `sub rsp, 28h` frame.
bool raise_caught(GuestState* s, uint64_t entry, uint64_t return_rva) {
    r(s, GuestReg::Rcx) = kCodeCaught;
    r(s, GuestReg::Rdx) = 0;
    r(s, GuestReg::R8) = 0;
    r(s, GuestReg::R9) = 0;
    call_import(s, g_raise_va, return_rva);
    return after_call(s, entry, {}) == After::Return;
}

// compare: sub rsp, 28h | ++calls; on call g_throw_at raise | rax = sign(*a - *b)
void sub_compare(GuestState* s) {
    const uint64_t entry = r(s, GuestReg::Rsp);
    r(s, GuestReg::Rsp) -= 0x28;
    const uint64_t a = r(s, GuestReg::Rcx);
    const uint64_t b = r(s, GuestReg::Rdx);
    if (++g_compare_calls == g_throw_at) {
        g_log += 'C';
        if (raise_caught(s, entry, kCompareRet)) {
            return;
        }
        g_log += '!';  // the raise never returns here
    }
    int32_t x = 0;
    int32_t y = 0;
    CHECK(arena().read(a, &x, 4));
    CHECK(arena().read(b, &y, 4));
    r(s, GuestReg::Rax) = static_cast<uint64_t>(static_cast<int64_t>(x < y ? -1 : (x > y ? 1 : 0)));
    r(s, GuestReg::Rsp) += 0x28;
    ret(s);
}

// raise: sub rsp, 28h | log 'X' | RaiseException(kCodeCaught)
void sub_raise(GuestState* s) {
    const uint64_t entry = r(s, GuestReg::Rsp);
    r(s, GuestReg::Rsp) -= 0x28;
    g_log += 'X';
    if (raise_caught(s, entry, kRaiseRet)) {
        return;
    }
    g_log += '!';
    r(s, GuestReg::Rsp) += 0x28;
    ret(s);
}

void sub_record(GuestState* s) {
    g_log += 'R';
    ret(s);
}

// Top-level filter: logs 'F', EXCEPTION_CONTINUE_SEARCH (the exception stays unhandled).
void sub_filter(GuestState* s) {
    g_log += 'F';
    r(s, GuestReg::Rax) = 0;
    ret(s);
}

const LiftedFunctionEntry kFunctions[] = {
    {va(kTry), sub_try},         {va(kHCatch), sub_h_catch}, {va(kCompare), sub_compare},
    {va(kRaise), sub_raise},     {va(kRecord), sub_record},  {va(kFilter), sub_filter},
};

// ---------------------------------------------------------------------------
// Synthetic .pdata and PE headers
// ---------------------------------------------------------------------------

constexpr uint8_t kPush = 0;
constexpr uint8_t kAllocLarge = 1;
constexpr uint8_t kAllocSmall = 2;

uint16_t code(uint8_t prolog_offset, uint8_t op, uint8_t info) {
    return static_cast<uint16_t>(prolog_offset | (op << 8) | (info << 12));
}

std::vector<uint8_t> unwind_info(uint8_t flags, uint8_t prolog, std::vector<uint16_t> codes, uint32_t handler) {
    std::vector<uint8_t> out = {static_cast<uint8_t>(1 | (flags << 3)), prolog, static_cast<uint8_t>(codes.size()), 0};
    for (uint16_t c : codes) {
        out.push_back(static_cast<uint8_t>(c));
        out.push_back(static_cast<uint8_t>(c >> 8));
    }
    if (codes.size() % 2 != 0) {
        out.push_back(0);
        out.push_back(0);
    }
    if (flags != 0) {
        for (int i = 0; i < 4; ++i) {
            out.push_back(static_cast<uint8_t>(handler >> (8 * i)));
        }
        out.resize(out.size() + 8, 0);  // handler data
    }
    return out;
}

void build_image() {
    std::vector<GuestRuntimeFunction> entries;
    uint64_t next = kUnwind;
    auto add = [&](uint64_t begin, uint64_t size, const std::vector<uint8_t>& info) {
        CHECK(arena().write(va(next), info.data(), info.size()));
        entries.push_back({static_cast<uint32_t>(begin), static_cast<uint32_t>(begin + size), static_cast<uint32_t>(next)});
        next = (next + info.size() + 7) & ~uint64_t{7};
    };
    const uint8_t eu = kUnwFlagEHandler | kUnwFlagUHandler;
    add(kTry, 0x80, unwind_info(eu, 5, {code(5, kAllocSmall, 5), code(1, kPush, GuestReg::Rbx)}, kHCatch));
    add(kHCatch, 0x80, unwind_info(0, 7, {code(7, kAllocLarge, 0), static_cast<uint16_t>(0xC8 / 8)}, 0));
    add(kCompare, 0x40, unwind_info(0, 4, {code(4, kAllocSmall, 4)}, 0));
    add(kRaise, 0x40, unwind_info(0, 4, {code(4, kAllocSmall, 4)}, 0));
    // kRecord and kFilter are leaves without .pdata.
    CHECK(arena().write(va(kPdata), entries.data(), entries.size() * sizeof(GuestRuntimeFunction)));

    const uint32_t e_lfanew = 0x80;
    CHECK(arena().write(va(0x3C), &e_lfanew, 4));
    const uint32_t signature = 0x00004550;
    CHECK(arena().write(va(0x80), &signature, 4));
    const uint64_t optional = 0x80 + 4 + 20;
    const uint16_t magic = 0x20B;
    const uint32_t size_of_image = static_cast<uint32_t>(kImageSize);
    const uint32_t directories = 16;
    const uint32_t pdata[2] = {static_cast<uint32_t>(kPdata),
                               static_cast<uint32_t>(entries.size() * sizeof(GuestRuntimeFunction))};
    CHECK(arena().write(va(optional), &magic, 2));
    CHECK(arena().write(va(optional + 56), &size_of_image, 4));
    CHECK(arena().write(va(optional + 108), &directories, 4));
    CHECK(arena().write(va(optional + 112 + 3 * 8), pdata, sizeof(pdata)));
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

struct GuestFault {
    FaultInfo info;
};

void throwing_handler(const FaultInfo* info) {
    throw GuestFault{*info};
}

struct ProcessExit {
    int32_t code;
};

void throwing_exit_hook(int32_t code, int32_t /*kind*/) {
    throw ProcessExit{code};
}

uint64_t exception_thunk(const char* name) {
    return register_host_callable(exception_thunk_for("KERNEL32.dll", name), name);
}

struct Fixture {
    GuestThread thread{};
    FaultInfo last_fault{};

    Fixture() {
        reset_guest_runtime();
        reset_guest_exceptions();
        com::reset_com_bridge();
        cbk::reset_callbacks();
        ce::reset_crt_exit();
        com::set_guest_memory(nullptr);  // the process arena, where the image lives
        set_fault_handler(throwing_handler);
        CHECK(register_lifted_functions(kFunctions));
        static bool mapped = false;
        if (!mapped) {
            CHECK(arena().reserve(kBase, kImageSize, PageProtect::ReadWrite) == kBase);
            CHECK(arena().commit(kBase, kImageSize, PageProtect::ReadWrite));
            mapped = true;
        }
        build_image();
        CHECK(register_image_from_headers(arena(), kBase));
        CHECK(create_guest_thread(arena(), 256 * 1024, thread));
        thread.state.mxcsr = kGuestMxcsrDefault;
        g_raise_va = exception_thunk("RaiseException");
        g_unwind_ex_va = exception_thunk("RtlUnwindEx");
        g_compare_calls = 0;
        g_throw_at = 0;
        g_log.clear();
    }

    ~Fixture() {
        destroy_guest_thread(arena(), thread);
        set_unhandled_exception_filter(0);
        ce::reset_crt_exit();
        cbk::reset_callbacks();
        com::reset_com_bridge();
        reset_guest_exceptions();
        reset_guest_runtime();
    }

    // Calls kTry with g_target / g_args; returns the fault kind (0 = none).
    uint32_t run_try(uint64_t target, std::initializer_list<uint64_t> args, uint64_t* rax) {
        g_target = target;
        std::fill(std::begin(g_args), std::end(g_args), 0);
        std::copy(args.begin(), args.end(), std::begin(g_args));
        try {
            *rax = call_guest(arena(), thread.state, va(kTry));
            return 0;
        } catch (const GuestFault& fault) {
            last_fault = fault.info;
            thread.state.gpr[GuestReg::Rsp] = thread.stack_top;
            thread.state.unwind_pending = 0;
            return fault.info.kind;
        }
    }

    bool balanced() const { return thread.state.gpr[GuestReg::Rsp] == thread.stack_top; }
};

std::vector<int32_t> read_ints(uint64_t at, size_t n) {
    std::vector<int32_t> v(n);
    CHECK(arena().read(at, v.data(), n * 4));
    return v;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void test_qsort_comparator_exception_caught_by_caller() {
    Fixture fx;
    const uint64_t qsort = register_host_callable(&pg6_gthunk_qsort, "qsort");
    const std::vector<int32_t> values = {5, 3, 9, 1, 7, 2, 8, 6};
    const uint64_t array = va(kData);
    CHECK(arena().write(array, values.data(), values.size() * 4));

    // Without an exception the comparator sorts normally.
    uint64_t rax = 0;
    CHECK(fx.run_try(qsort, {array, values.size(), 4, va(kCompare)}, &rax) == 0);
    CHECK(rax == 1);
    CHECK(read_ints(array, values.size()) == (std::vector<int32_t>{1, 2, 3, 5, 6, 7, 8, 9}));

    // The fourth comparison raises: the caller's handler catches it, qsort stops calling the
    // comparator and leaves the array alone, and the caller continues at its catch block.
    CHECK(arena().write(array, values.data(), values.size() * 4));
    g_compare_calls = 0;
    g_throw_at = 4;
    g_log.clear();
    CHECK(fx.run_try(qsort, {array, values.size(), 4, va(kCompare)}, &rax) == 0);
    CHECK(rax == 0x77 + 0x200);  // RtlUnwindEx return value + the catch continuation
    CHECK(g_log == "CHTK");      // raise, handler, target unwind of the catching frame, continue
    CHECK(g_compare_calls == 4);
    CHECK(read_ints(array, values.size()) == values);
    CHECK(fx.balanced());
    CHECK(fx.thread.state.unwind_pending == 0);
}

void test_bsearch_and_initterm_abandon() {
    Fixture fx;
    const uint64_t bsearch = register_host_callable(&pg6_gthunk_bsearch, "bsearch");
    const std::vector<int32_t> sorted = {1, 3, 5, 7, 9, 11};
    const uint64_t array = va(kData);
    const uint64_t key = va(kData + 0x100);
    CHECK(arena().write(array, sorted.data(), sorted.size() * 4));
    const int32_t nine = 9;  // second comparison (with 11) raises
    CHECK(arena().write(key, &nine, 4));
    g_throw_at = 2;
    uint64_t rax = 0;
    CHECK(fx.run_try(bsearch, {key, array, sorted.size(), 4, va(kCompare)}, &rax) == 0);
    CHECK(rax == 0x77 + 0x200);
    CHECK(g_compare_calls == 2);
    CHECK(fx.balanced());

    // _initterm: the raising initializer is caught by the caller; the next one never runs.
    const uint64_t initterm = register_host_callable(&pg6_gthunk_initterm, "_initterm");
    const uint64_t table = va(kData + 0x200);
    store64(table, va(kRecord));
    store64(table + 8, va(kRaise));
    store64(table + 16, va(kRecord));
    g_log.clear();
    CHECK(fx.run_try(initterm, {table, table + 24}, &rax) == 0);
    CHECK(rax == 0x77 + 0x200);
    CHECK(g_log == "RXHTK");
    CHECK(fx.balanced());
    const uint64_t initterm_e = register_host_callable(&pg6_gthunk_initterm_e, "_initterm_e");
    g_log.clear();
    CHECK(fx.run_try(initterm_e, {table + 8, table + 24}, &rax) == 0);
    CHECK(rax == 0x77 + 0x200);
    CHECK(g_log == "XHTK");
}

void test_task_queue_callback_exception_is_unhandled() {
    Fixture fx;
    // An Immediate queue runs the callback inside XTaskQueueSubmitCallback, i.e. below the guest
    // frame of kTry, whose handler would catch the code. The callback is a top-level callback, so
    // the exception is unhandled: the top-level filter runs, then the fault handler.
    Pg6XTaskQueueHandle queue = nullptr;
    CHECK(pg6_gdk_xtask_queue_create(3, 3, &queue) == 0);
    const uint64_t submit = register_host_callable(&pg6_gthunk_xtask_queue_submit_callback, "XTaskQueueSubmitCallback");
    set_unhandled_exception_filter(va(kFilter));
    uint64_t rax = 0;
    const uint32_t kind = fx.run_try(submit, {reinterpret_cast<uintptr_t>(queue), 0, 0x1234, va(kRaise)}, &rax);
    CHECK(kind == static_cast<uint32_t>(FaultKind::UnhandledException));
    CHECK(fx.last_fault.detail == kCodeCaught);
    CHECK(g_log == "XF");  // kTry's handler is never consulted
    // The exception left the callback by unwinding, which still released its in-flight count, so
    // the queue closes without waiting (and without leaking).
    pg6_gdk_xtask_queue_close_handle(queue);

    // Same for a guest exit handler raising: unhandled, not caught by the guest calling exit.
    pg6_crt_set_exit_hook(&throwing_exit_hook);
    const uint64_t crt_atexit = register_host_callable(&pg6_gthunk_crt_atexit, "_crt_atexit");
    const uint64_t exit_fn = register_host_callable(&pg6_gthunk_exit, "exit");
    const uint64_t handler[1] = {va(kRaise)};
    CHECK(call_guest(arena(), fx.thread.state, crt_atexit, handler) == 0);
    g_log.clear();
    const uint32_t exit_kind = fx.run_try(exit_fn, {3}, &rax);
    CHECK(exit_kind == static_cast<uint32_t>(FaultKind::UnhandledException));
    CHECK(g_log == "XF");
    pg6_crt_set_exit_hook(nullptr);
}

}  // namespace

int main() {
    test_qsort_comparator_exception_caught_by_caller();
    test_bsearch_and_initterm_abandon();
    test_task_queue_callback_exception_is_unhandled();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_guest_callback_unwind: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_guest_callback_unwind: all tests passed\n");
    return 0;
}
