#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "abi/playgta6_abi.h"
#include "guest/guest_runtime.h"
#include "os/vmem.h"
#include "os/win32_kernel.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_guest_runtime.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6;
using namespace playgta6::guest;
using playgta6::os::PageProtect;
using playgta6::os::VirtualMemoryArena;

constexpr uint64_t kImageBase = 0x140000000ULL;
constexpr uint64_t kAdd6 = 0x140001000ULL;
constexpr uint64_t kCaller = 0x140001100ULL;
constexpr uint64_t kViaIat = 0x140001200ULL;
constexpr uint64_t kUnknownCaller = 0x140001300ULL;
constexpr uint64_t kTrapFn = 0x140001400ULL;
constexpr uint64_t kUnimplementedFn = 0x140001500ULL;
constexpr uint64_t kDivideFn = 0x140001600ULL;
constexpr uint64_t kJmpReturn = 0x140001700ULL;
constexpr uint64_t kLeaky = 0x140001800ULL;
constexpr uint64_t kMul = 0x140001900ULL;
constexpr uint64_t kProbe = 0x140001A00ULL;
constexpr uint32_t kIatRva = 0x3000;

// Guest memory used by the hand-written "lifted" functions below.
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

// What lifted code does for `push`, `pop` and `ret` (the popped return address is not checked).
void push(GuestState* s, uint64_t value) {
    gpr(s, GuestReg::Rsp) -= 8;
    store64(gpr(s, GuestReg::Rsp), value);
}

uint64_t pop(GuestState* s) {
    const uint64_t value = load64(gpr(s, GuestReg::Rsp));
    gpr(s, GuestReg::Rsp) += 8;
    return value;
}

void ret(GuestState* s) {
    gpr(s, GuestReg::Rsp) += 8;
}

bool g_entry_aligned = true;

void note_entry(GuestState* s) {
    if ((gpr(s, GuestReg::Rsp) + 8) % 16 != 0) {
        g_entry_aligned = false;
    }
}

// ---------------------------------------------------------------------------
// Hand-written lifted functions (Win64 convention, guest stack in the arena)
// ---------------------------------------------------------------------------

// sub_140001000: rax = rcx + rdx + r8 + r9 + [rsp+40] + [rsp+48]; ret
void sub_add6(GuestState* s) {
    note_entry(s);
    const uint64_t rsp = gpr(s, GuestReg::Rsp);
    gpr(s, GuestReg::Rax) = gpr(s, GuestReg::Rcx) + gpr(s, GuestReg::Rdx) + gpr(s, GuestReg::R8) +
                            gpr(s, GuestReg::R9) + load64(rsp + kGuestStackArgOffset) +
                            load64(rsp + kGuestStackArgOffset + 8);
    ret(s);
}

// sub_140001100(a, b): add6(1..6) via a direct call, then add6(a, b, 10, 20, 30, 40) through the
// dispatcher (`call rax`); returns the sum of both.
void sub_caller(GuestState* s) {
    note_entry(s);
    gpr(s, GuestReg::R12) = gpr(s, GuestReg::Rcx);
    gpr(s, GuestReg::R13) = gpr(s, GuestReg::Rdx);
    gpr(s, GuestReg::Rsp) -= 0x38;  // shadow space + 2 stack args, keeps 16-byte alignment
    const uint64_t rsp = gpr(s, GuestReg::Rsp);

    store64(rsp + 0x20, 5);
    store64(rsp + 0x28, 6);
    gpr(s, GuestReg::Rcx) = 1;
    gpr(s, GuestReg::Rdx) = 2;
    gpr(s, GuestReg::R8) = 3;
    gpr(s, GuestReg::R9) = 4;
    push(s, kCaller + 0x30);  // direct `call sub_140001000`
    sub_add6(s);
    gpr(s, GuestReg::Rbx) = gpr(s, GuestReg::Rax);

    store64(rsp + 0x20, 30);
    store64(rsp + 0x28, 40);
    gpr(s, GuestReg::Rcx) = gpr(s, GuestReg::R12);
    gpr(s, GuestReg::Rdx) = gpr(s, GuestReg::R13);
    gpr(s, GuestReg::R8) = 10;
    gpr(s, GuestReg::R9) = 20;
    gpr(s, GuestReg::Rax) = kAdd6;
    push(s, kCaller + 0x60);  // `call rax`
    __playgta6_indirect_call(s, gpr(s, GuestReg::Rax));

    gpr(s, GuestReg::Rax) += gpr(s, GuestReg::Rbx);
    gpr(s, GuestReg::Rsp) += 0x38;
    ret(s);
}

// sub_140001200: `call [rip+iat0]` then `jmp [rip+iat1]` with rcx = rax, rdx = 1.
void sub_via_iat(GuestState* s) {
    note_entry(s);
    push(s, kViaIat + 0x10);
    __playgta6_indirect_call(s, load64(kImageBase + kIatRva));
    gpr(s, GuestReg::Rcx) = gpr(s, GuestReg::Rax);
    gpr(s, GuestReg::Rdx) = 1;
    __playgta6_indirect_call(s, load64(kImageBase + kIatRva + 8));  // tail call: no push, no ret
}

// sub_140001300: `call` into the middle of sub_140001200 (not a function entry).
void sub_unknown_caller(GuestState* s) {
    push(s, kUnknownCaller + 0x10);
    __playgta6_indirect_call(s, kViaIat + 0x34);
    ret(s);
}

void sub_trap(GuestState* s) {
    __playgta6_trap(s, kTrapFn + 4);
}

void sub_unimplemented(GuestState* s) {
    __playgta6_unimplemented(s, kUnimplementedFn + 8);
}

void sub_divide(GuestState* s) {
    __playgta6_divide_error(s, kDivideFn + 0xC);
}

uint64_t g_seen_return_address = 0;

// sub_140001700: mov eax, 99; pop rcx; jmp rcx
void sub_jmp_return(GuestState* s) {
    gpr(s, GuestReg::Rax) = 99;
    gpr(s, GuestReg::Rcx) = pop(s);
    g_seen_return_address = gpr(s, GuestReg::Rcx);
    __playgta6_indirect_call(s, gpr(s, GuestReg::Rcx));
}

// sub_140001800: forgets its `ret`.
void sub_leaky(GuestState* s) {
    gpr(s, GuestReg::Rax) = 1;
}

// sub_140001900: rax = rcx * rdx; ret
void sub_mul(GuestState* s) {
    note_entry(s);
    gpr(s, GuestReg::Rax) = gpr(s, GuestReg::Rcx) * gpr(s, GuestReg::Rdx);
    ret(s);
}

// sub_140001A00: records the first g_probe_count Win64 arguments it receives.
size_t g_probe_count = 0;
std::vector<uint64_t> g_probe_args;

void sub_probe(GuestState* s) {
    note_entry(s);
    static constexpr uint32_t kRegs[4] = {GuestReg::Rcx, GuestReg::Rdx, GuestReg::R8, GuestReg::R9};
    g_probe_args.clear();
    for (size_t i = 0; i < g_probe_count; ++i) {
        g_probe_args.push_back(i < 4 ? gpr(s, kRegs[i])
                                     : load64(gpr(s, GuestReg::Rsp) + kGuestStackArgOffset + 8 * (i - 4)));
    }
    gpr(s, GuestReg::Rax) = g_probe_count;
    ret(s);
}

// Registered table of the fake image (sorted by va).
const LiftedFunctionEntry kImageFunctions[] = {
    {kAdd6, sub_add6},
    {kCaller, sub_caller},
    {kViaIat, sub_via_iat},
    {kUnknownCaller, sub_unknown_caller},
    {kTrapFn, sub_trap},
    {kUnimplementedFn, sub_unimplemented},
    {kDivideFn, sub_divide},
    {kJmpReturn, sub_jmp_return},
    {kLeaky, sub_leaky},
    {kMul, sub_mul},
    {kProbe, sub_probe},
};

// ---------------------------------------------------------------------------
// Host thunks (R5 shape: entered after the guest pushed its return address, pop it like `ret`)
// ---------------------------------------------------------------------------

void thunk_mul_add(void* state) {
    auto* s = static_cast<GuestState*>(state);
    gpr(s, GuestReg::Rax) = gpr(s, GuestReg::Rcx) * 10 + gpr(s, GuestReg::Rdx);
    ret(s);
}

void thunk_add(void* state) {
    auto* s = static_cast<GuestState*>(state);
    gpr(s, GuestReg::Rax) = gpr(s, GuestReg::Rcx) + gpr(s, GuestReg::Rdx);
    ret(s);
}

// RunCallback(fn, value): calls guest fn(value, 3) and returns its result + 1.
void thunk_run_callback(void* state) {
    auto* s = static_cast<GuestState*>(state);
    const uint64_t result =
        call_guest_args(*g_arena, *s, gpr(s, GuestReg::Rcx), gpr(s, GuestReg::Rdx), uint64_t{3});
    gpr(s, GuestReg::Rax) = result + 1;
    ret(s);
}

// ---------------------------------------------------------------------------
// Fault handling for tests: throw out of the guest call
// ---------------------------------------------------------------------------

struct GuestFault {
    FaultInfo info;
    std::string message;
};

void throwing_handler(const FaultInfo* info) {
    char message[1024];
    format_fault(*info, message, sizeof(message));
    throw GuestFault{*info, message};
}

template <typename F>
bool expect_fault(F&& body, GuestFault* out) {
    try {
        body();
    } catch (const GuestFault& fault) {
        *out = fault;
        return true;
    }
    return false;
}

struct Fixture {
    VirtualMemoryArena arena;
    GuestThread thread{};

    Fixture() {
        reset_guest_runtime();
        set_fault_handler(throwing_handler);
        g_arena = &arena;
        CHECK(register_lifted_functions(kImageFunctions));
        CHECK(arena.reserve(kImageBase, 0x10000, PageProtect::ReadWrite) == kImageBase);
        CHECK(arena.commit(kImageBase, 0x4000, PageProtect::ReadWrite));
        CHECK(create_guest_thread(arena, 64 * 1024, thread));
        g_entry_aligned = true;
    }

    ~Fixture() {
        destroy_guest_thread(arena, thread);
        g_arena = nullptr;
        reset_guest_runtime();
    }

    GuestState& state() {
        return thread.state;
    }

    uint64_t call(uint64_t va, std::initializer_list<uint64_t> args = {}) {
        return call_guest(arena, thread.state, va, std::span<const uint64_t>(args.begin(), args.size()));
    }

    // After a fault unwound a guest call, rsp is left where the fault happened.
    void reset_stack() {
        thread.state.gpr[GuestReg::Rsp] = thread.stack_top;
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void test_registry_lookup() {
    Fixture fx;
    CHECK(lifted_function_count() == sizeof(kImageFunctions) / sizeof(kImageFunctions[0]));
    CHECK(find_lifted_function(kAdd6) == sub_add6);
    CHECK(find_lifted_function(kProbe) == sub_probe);
    CHECK(find_lifted_function(kAdd6 + 1) == nullptr);
    CHECK(find_lifted_function(kAdd6 - 0x10) == nullptr);
    CHECK(find_lifted_function(0) == nullptr);
    CHECK(pg6_guest_find_lifted_function(kMul) == sub_mul);

    // Validation.
    static const LiftedFunctionEntry unsorted[] = {{0x150000020, sub_mul}, {0x150000010, sub_mul}};
    static const LiftedFunctionEntry duplicate[] = {{0x150000010, sub_mul}, {0x150000010, sub_mul}};
    static const LiftedFunctionEntry null_fn[] = {{0x150000010, nullptr}};
    static const LiftedFunctionEntry overlapping[] = {{0x140000800, sub_mul}, {0x140001050, sub_mul}};
    static const LiftedFunctionEntry synthetic[] = {{kSyntheticBase + 0x100, sub_mul}};
    static const LiftedFunctionEntry second_module[] = {{0x150000000, sub_mul}, {0x150000040, sub_add6}};
    CHECK(!register_lifted_functions(unsorted));
    CHECK(!register_lifted_functions(duplicate));
    CHECK(!register_lifted_functions(null_fn));
    CHECK(!register_lifted_functions(overlapping));
    CHECK(!register_lifted_functions(synthetic));
    CHECK(register_lifted_functions(std::span<const LiftedFunctionEntry>()));
    CHECK(pg6_guest_register_lifted_functions(nullptr, 1) == 0);
    CHECK(pg6_guest_register_lifted_functions(second_module, 2) == 1);
    CHECK(find_lifted_function(0x150000040) == sub_add6);
    CHECK(find_lifted_function(kAdd6) == sub_add6);
    CHECK(lifted_function_count() == sizeof(kImageFunctions) / sizeof(kImageFunctions[0]) + 2);
}

void test_guest_thread_stack() {
    Fixture fx;
    const GuestThread& t = fx.thread;
    CHECK(t.stack_top - t.stack_limit == 64 * 1024);
    CHECK(t.stack_limit - t.stack_reservation == os::kPageSize);
    CHECK(t.stack_top % 16 == 0);
    CHECK(t.state.gpr[GuestReg::Rsp] == t.stack_top);
    CHECK(t.state.gs_base == 0);

    uint64_t value = 0x1234;
    CHECK(fx.arena.write(t.stack_top - 8, &value, 8));
    CHECK(fx.arena.write(t.stack_limit, &value, 8));
    CHECK(!fx.arena.write(t.stack_limit - 8, &value, 8));  // guard page
    CHECK(!fx.arena.read(t.stack_limit - 8, &value, 8));

    set_teb(fx.thread.state, 0x7FFE0000ULL);
    CHECK(fx.thread.state.gs_base == 0x7FFE0000ULL);

    GuestThread other{};
    CHECK(create_guest_thread(fx.arena, 100, other));  // rounded up to one page
    CHECK(other.stack_top - other.stack_limit == os::kPageSize);
    const uint64_t reservation = other.stack_reservation;
    destroy_guest_thread(fx.arena, other);
    CHECK(other.stack_reservation == 0);
    CHECK(fx.arena.query(reservation).state == os::MemState::Free);
}

void test_call_chain_with_win64_args() {
    Fixture fx;
    CHECK(fx.call(kAdd6, {1, 2, 3, 4, 5, 6}) == 21);
    CHECK(fx.state().gpr[GuestReg::Rsp] == fx.thread.stack_top);

    // direct call (1+..+6 = 21) + indirect call (7 + 8 + 10 + 20 + 30 + 40 = 115)
    CHECK(call_guest_args(fx.arena, fx.state(), kCaller, 7, 8) == 136);
    CHECK(fx.state().gpr[GuestReg::Rsp] == fx.thread.stack_top);
    CHECK(fx.state().rip == kAdd6);  // last dispatched target
    CHECK(g_entry_aligned);

    // Every argument count lands in the right register / stack slot with entry alignment.
    for (size_t n = 0; n <= 9; ++n) {
        std::vector<uint64_t> args;
        for (size_t i = 0; i < n; ++i) {
            args.push_back(0x1000 + i);
        }
        g_probe_count = n;
        g_entry_aligned = true;
        CHECK(call_guest(fx.arena, fx.state(), kProbe, args) == n);
        CHECK(g_probe_args == args);
        CHECK(g_entry_aligned);
        CHECK(fx.state().gpr[GuestReg::Rsp] == fx.thread.stack_top);
    }

    // Also from an rsp that is only 8-byte aligned (e.g. a callback from inside a thunk).
    fx.state().gpr[GuestReg::Rsp] = fx.thread.stack_top - 0x48;
    g_probe_count = 5;
    g_entry_aligned = true;
    CHECK(fx.call(kProbe, {1, 2, 3, 4, 5}) == 5);
    CHECK(g_entry_aligned);
    CHECK(fx.state().gpr[GuestReg::Rsp] == fx.thread.stack_top - 0x48);
}

void test_import_binding_and_dispatch() {
    Fixture fx;
    // A read-only IAT page, as after the image loader applied section protections.
    CHECK(fx.arena.protect(kImageBase + kIatRva, os::kPageSize, PageProtect::ReadOnly));

    const ImportSlotBinding slots[] = {
        {kIatRva, thunk_mul_add},
        {kIatRva + 8, thunk_add},
        {kIatRva + 16, nullptr},  // unresolved without a thunk: left alone
    };
    CHECK(bind_import_slots(fx.arena, kImageBase, slots) == 2);
    CHECK(fx.arena.query(kImageBase + kIatRva).protect == PageProtect::ReadOnly);

    const uint64_t slot0 = load64(kImageBase + kIatRva);
    const uint64_t slot1 = load64(kImageBase + kIatRva + 8);
    CHECK(load64(kImageBase + kIatRva + 16) == 0);
    CHECK(slot0 != slot1);
    CHECK(is_synthetic_address(slot0) && is_synthetic_address(slot1));
    CHECK(slot0 != kGuestReturnSentinel && slot1 != kGuestReturnSentinel);
    CHECK(slot0 % kSyntheticStride == 0);
    // Never a real (canonical) x86-64 address, never an O7 virtual module handle.
    CHECK((slot0 >> 47) != 0 && (slot0 >> 47) != 0x1FFFF);

    SyntheticTarget info{};
    CHECK(describe_synthetic(slot0, &info));
    CHECK(info.is_import && info.slot_rva == kIatRva && info.image_base == kImageBase);
    CHECK(info.thunk == thunk_mul_add);
    CHECK(!describe_synthetic(kGuestReturnSentinel, &info));
    CHECK(!describe_synthetic(slot1 + 1, &info));

    // call [iat0] -> 4 * 10 + 2 = 42, then jmp [iat1] -> 42 + 1; the thunk's pop balances the stack.
    CHECK(fx.call(kViaIat, {4, 2}) == 43);
    CHECK(fx.state().gpr[GuestReg::Rsp] == fx.thread.stack_top);

    // A bound import is also a valid call_guest target.
    CHECK(fx.call(slot1, {5, 6}) == 11);

    // Binding through the C ABI uses the process arena.
    os::win32::set_vmem_arena(&fx.arena);
    const Pg6ImportSlotBinding more[] = {{kIatRva + 24, thunk_add}};
    CHECK(pg6_guest_bind_import_slots(kImageBase, more, 1) == 1);
    CHECK(is_synthetic_address(load64(kImageBase + kIatRva + 24)));
    CHECK(pg6_guest_bind_import_slots(kImageBase + 0x100000, more, 1) == 0);  // unmapped
    os::win32::set_vmem_arena(nullptr);
}

void test_host_callable_and_nested_callback() {
    Fixture fx;
    constexpr uint64_t kProc = 0xDEADBEEF00ULL;
    const uint64_t run_callback = register_host_callable(thunk_run_callback, "RunCallback", kProc);
    CHECK(is_synthetic_address(run_callback));
    CHECK(guest_callable_for_proc(kProc) == run_callback);
    CHECK(pg6_guest_callable_for_proc(kProc) == run_callback);
    CHECK(guest_callable_for_proc(kProc + 1) == 0);
    CHECK(register_host_callable(nullptr, "x", 0) == 0);

    SyntheticTarget info{};
    CHECK(describe_synthetic(run_callback, &info));
    CHECK(!info.is_import && info.thunk == thunk_run_callback);
    CHECK(info.name != nullptr && std::strcmp(info.name, "RunCallback") == 0);

    // RunCallback(sub_mul, 7): guest -> host thunk -> guest callback (7 * 3) -> + 1.
    g_entry_aligned = true;
    CHECK(fx.call(run_callback, {kMul, 7}) == 22);
    CHECK(g_entry_aligned);
    CHECK(fx.state().gpr[GuestReg::Rsp] == fx.thread.stack_top);
}

void test_unknown_targets_fault() {
    Fixture fx;
    GuestFault fault{};

    // A target inside lifted code that is not a function entry (jump-table default arm, ...).
    CHECK(expect_fault([&] { fx.call(kUnknownCaller); }, &fault));
    CHECK(static_cast<FaultKind>(fault.info.kind) == FaultKind::UnknownTarget);
    CHECK(fault.info.va == kViaIat + 0x34);
    CHECK(fault.info.detail == kViaIat);
    CHECK(fault.info.state == &fx.state());
    CHECK(fault.message.find("unknown target at 0x0000000140001234") != std::string::npos);
    CHECK(fault.message.find("sub_140001200 + 0x34") != std::string::npos);
    CHECK(fault.message.find("rsp=") != std::string::npos);
    fx.reset_stack();

    // Nothing registered anywhere near.
    CHECK(expect_fault([&] { fx.call(0x7000000000ULL); }, &fault));
    CHECK(static_cast<FaultKind>(fault.info.kind) == FaultKind::UnknownTarget);
    CHECK(fault.info.va == 0x7000000000ULL);
    fx.reset_stack();

    CHECK(expect_fault([&] { fx.call(0x10); }, &fault));
    CHECK(fault.info.va == 0x10 && fault.info.detail == 0);
    fx.reset_stack();

    // Unbound and misaligned synthetic addresses.
    CHECK(expect_fault([&] { fx.call(kSyntheticBase + 5000 * kSyntheticStride); }, &fault));
    CHECK(static_cast<FaultKind>(fault.info.kind) == FaultKind::UnknownTarget);
    CHECK(fault.message.find("synthetic address not bound") != std::string::npos);
    fx.reset_stack();
    const uint64_t bound = register_host_callable(thunk_add);
    CHECK(expect_fault([&] { fx.call(bound + 4); }, &fault));
    CHECK(fault.info.va == bound + 4);
    fx.reset_stack();
    CHECK(fx.call(bound, {1, 2}) == 3);
}

void test_non_returning_helpers_fault() {
    Fixture fx;
    struct Case {
        uint64_t fn;
        FaultKind kind;
        uint64_t va;
        const char* text;
    };
    const Case cases[] = {
        {kTrapFn, FaultKind::Trap, kTrapFn + 4, "guest fault: trap at 0x0000000140001404"},
        {kUnimplementedFn, FaultKind::Unimplemented, kUnimplementedFn + 8,
         "guest fault: unimplemented instruction at 0x0000000140001508"},
        {kDivideFn, FaultKind::DivideError, kDivideFn + 0xC,
         "guest fault: divide error at 0x000000014000160c"},
    };
    for (const Case& c : cases) {
        GuestFault fault{};
        fx.state().gpr[GuestReg::Rbx] = 0xB0B0B0B0ULL;
        CHECK(expect_fault([&] { fx.call(c.fn); }, &fault));
        CHECK(static_cast<FaultKind>(fault.info.kind) == c.kind);
        CHECK(fault.info.va == c.va);
        CHECK(fault.info.state == &fx.state());
        CHECK(fx.state().rip == c.va);
        CHECK(fault.message.rfind(c.text, 0) == 0);
        CHECK(fault.message.find("rbx=00000000b0b0b0b0") != std::string::npos);
        fx.reset_stack();
    }
}

void test_sentinel_and_stack_checks() {
    Fixture fx;
    // `pop rcx; jmp rcx` in the entry function jumps to the sentinel: the call ends cleanly.
    CHECK(fx.call(kJmpReturn) == 99);
    CHECK(g_seen_return_address == kGuestReturnSentinel);
    CHECK(fx.state().gpr[GuestReg::Rsp] == fx.thread.stack_top);

    GuestFault fault{};
    CHECK(expect_fault([&] { fx.call(kLeaky); }, &fault));
    CHECK(static_cast<FaultKind>(fault.info.kind) == FaultKind::StackImbalance);
    CHECK(fault.info.va == kLeaky);
    CHECK(fault.info.detail == fx.thread.stack_top - 40);  // entry rsp: nothing popped
    fx.reset_stack();

    fx.state().gpr[GuestReg::Rsp] = 0x700000000000ULL;  // not mapped
    CHECK(expect_fault([&] { fx.call(kAdd6); }, &fault));
    CHECK(static_cast<FaultKind>(fault.info.kind) == FaultKind::BadStack);
    CHECK(fault.info.detail == 0x700000000000ULL - 0x28);
    fx.state().gpr[GuestReg::Rsp] = 0x20;
    CHECK(expect_fault([&] { fx.call(kAdd6); }, &fault));
    CHECK(static_cast<FaultKind>(fault.info.kind) == FaultKind::BadStack);
    fx.reset_stack();
}

void test_fault_handler_install_and_format() {
    Fixture fx;
    CHECK(set_fault_handler(nullptr) == throwing_handler);
    CHECK(pg6_guest_set_fault_handler(throwing_handler) == nullptr);
    CHECK(pg6_guest_set_fault_handler(throwing_handler) == throwing_handler);

    // snprintf-style length reporting with a small buffer.
    const FaultInfo info{static_cast<uint32_t>(FaultKind::Trap), 0, 0x140001000ULL, 0, nullptr};
    char small[16];
    const size_t full = format_fault(info, small, sizeof(small));
    CHECK(full == std::strlen("guest fault: trap at 0x0000000140001000"));
    CHECK(std::strlen(small) == sizeof(small) - 1);
    CHECK(std::strcmp(fault_kind_name(FaultKind::BadStack), "guest stack not writable") == 0);
}

void test_c_abi_thread_and_call() {
    Fixture fx;
    os::win32::set_vmem_arena(&fx.arena);
    Pg6GuestThread thread{};
    CHECK(pg6_guest_thread_create(&thread, 0) == 1);
    CHECK(thread.stack_top - thread.stack_limit == kDefaultGuestStackSize);
    const uint64_t args[6] = {10, 20, 30, 40, 50, 60};
    CHECK(pg6_guest_call(&thread.state, kAdd6, args, 6) == 210);
    CHECK(thread.state.gpr[GuestReg::Rsp] == thread.stack_top);
    pg6_guest_thread_destroy(&thread);
    CHECK(thread.stack_top == 0);
    CHECK(pg6_guest_thread_create(nullptr, 0) == 0);
    os::win32::set_vmem_arena(nullptr);
}

void bulk_fn(GuestState* s) {
    gpr(s, GuestReg::Rax) = s->rip;  // the dispatcher stores the target in rip
    ret(s);
}

constexpr uint64_t kBulkBase = 0x200000000ULL;
constexpr size_t kBulkCount = 200000;

std::vector<LiftedFunctionEntry>& bulk_table() {
    static std::vector<LiftedFunctionEntry> table = [] {
        std::vector<LiftedFunctionEntry> t;
        t.reserve(kBulkCount);
        for (size_t i = 0; i < kBulkCount; ++i) {
            t.push_back({kBulkBase + i * 16, bulk_fn});
        }
        return t;
    }();
    return table;
}

void test_bulk_registration() {
    Fixture fx;
    const auto& table = bulk_table();
    const auto t0 = std::chrono::steady_clock::now();
    CHECK(register_lifted_functions(table));
    const auto t1 = std::chrono::steady_clock::now();
    CHECK(lifted_function_count() == kBulkCount + sizeof(kImageFunctions) / sizeof(kImageFunctions[0]));

    size_t hits = 0;
    size_t misses = 0;
    for (size_t i = 0; i < kBulkCount; ++i) {
        hits += find_lifted_function(kBulkBase + i * 16) == bulk_fn ? 1 : 0;
        misses += find_lifted_function(kBulkBase + i * 16 + 8) == nullptr ? 1 : 0;
    }
    const auto t2 = std::chrono::steady_clock::now();
    CHECK(hits == kBulkCount);
    CHECK(misses == kBulkCount);

    const double register_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double lookup_ns =
        std::chrono::duration<double, std::nano>(t2 - t1).count() / (2.0 * kBulkCount);
    std::printf("bulk: registered %zu functions in %.3f ms, %.1f ns per lookup\n", kBulkCount,
                register_ms, lookup_ns);
    CHECK(register_ms < 1000.0);  // one validation pass, no per-entry allocation
    CHECK(lookup_ns < 20000.0);   // O(log n); generous for unoptimized/sanitizer builds

    const uint64_t last = kBulkBase + (kBulkCount - 1) * 16;
    CHECK(fx.call(last) == last);
}

void test_concurrent_dispatch() {
    Fixture fx;
    CHECK(register_lifted_functions(bulk_table()));
    const ImportSlotBinding slots[] = {{kIatRva, thunk_mul_add}, {kIatRva + 8, thunk_add}};
    CHECK(bind_import_slots(fx.arena, kImageBase, slots) == 2);

    constexpr int kThreads = 4;
    constexpr int kIterations = 5000;
    std::atomic<bool> start{false};
    std::atomic<int> errors{0};
    std::vector<std::thread> workers;
    for (int w = 0; w < kThreads; ++w) {
        workers.emplace_back([&, w] {
            GuestThread thread{};
            if (!create_guest_thread(fx.arena, 32 * 1024, thread)) {
                errors.fetch_add(1);
                return;
            }
            while (!start.load()) {
                std::this_thread::yield();
            }
            uint64_t x = 0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(w + 1);
            for (int i = 0; i < kIterations; ++i) {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                const uint64_t va = kBulkBase + (x % kBulkCount) * 16;
                if (find_lifted_function(va) != bulk_fn) {
                    errors.fetch_add(1);
                }
                const uint64_t a = static_cast<uint64_t>(i);
                if (call_guest_args(fx.arena, thread.state, kViaIat, a, uint64_t{0}) != a * 10 + 1) {
                    errors.fetch_add(1);
                }
                // Host callables being registered concurrently are either invisible or complete.
                SyntheticTarget info{};
                const uint64_t probe = kSyntheticBase + (1 + x % 6000) * kSyntheticStride;
                if (describe_synthetic(probe, &info) && info.thunk == nullptr) {
                    errors.fetch_add(1);
                }
            }
            destroy_guest_thread(fx.arena, thread);
        });
    }
    start.store(true);
    uint64_t last = 0;
    for (int i = 0; i < 5000; ++i) {  // crosses a synthetic table chunk boundary
        last = register_host_callable(thunk_add);
    }
    for (auto& worker : workers) {
        worker.join();
    }
    CHECK(errors.load() == 0);
    CHECK(last == kSyntheticBase + 5002 * kSyntheticStride);
}

}  // namespace

int main() {
    test_registry_lookup();
    test_guest_thread_stack();
    test_call_chain_with_win64_args();
    test_import_binding_and_dispatch();
    test_host_callable_and_nested_callback();
    test_unknown_targets_fault();
    test_non_returning_helpers_fault();
    test_sentinel_and_stack_checks();
    test_fault_handler_install_and_format();
    test_c_abi_thread_and_call();
    test_bulk_registration();
    test_concurrent_dispatch();

    if (g_failures != 0) {
        std::fprintf(stderr, "test_guest_runtime failed with %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("All playgta6_guest_runtime_tests passed.\n");
    return 0;
}
