// R18: guest-callable thunks for catalogued shim exports (guest/export_thunks.h).
//
// tests/fixtures/export_thunks/ is the real output of `playgta6-recomp export-thunks --only ...`
// for a handful of exports (a Rust test checks it against the generator, see
// recomp/src/stubs/exports/tests.rs): generated marshaling thunks for real shims, not
// hand-written ones. This test registers that table the way the boot does and drives the guest
// GetProcAddress thunk plus the returned addresses with guest-convention calls
// (__playgta6_indirect_call with arguments in rcx rdx r8 r9 / xmm0-3 / the stack).
//
// On wasm64 a host pointer to a string literal can look like an ordinal (HIWORD == 0), so every
// guest-visible string lives in the guest arena and host-side lookups use the by-name entry points.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>

#include "abi/playgta6_abi.h"
#include "guest/com_bridge.h"
#include "guest/export_thunks.h"
#include "guest/guest_callbacks.h"
#include "guest/guest_runtime.h"
#include "os/shim_exports.h"
#include "os/vmem.h"
#include "os/win32_kernel.h"
#include "os/win32_loader.h"
#include "pg6_export_thunks.h"

static_assert(sizeof(Pg6ExportThunk) == sizeof(Pg6BootExport) &&
                  alignof(Pg6ExportThunk) == alignof(Pg6BootExport),
              "the generated table is read as Pg6BootExport");
static_assert(offsetof(Pg6ExportThunk, thunk) == offsetof(Pg6BootExport, thunk) &&
                  offsetof(Pg6ExportThunk, reason) == offsetof(Pg6BootExport, reason) &&
                  offsetof(Pg6ExportThunk, flags) == offsetof(Pg6BootExport, flags),
              "field layout");

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_export_thunks.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6;
using namespace playgta6::guest;
namespace com = playgta6::guest::com;
namespace cbk = playgta6::guest::callbacks;
namespace win32 = playgta6::os::win32;
using playgta6::os::VirtualMemoryArena;

constexpr uint64_t kReturnAddress = 0x140001000ULL;
constexpr uint32_t kRegs[4] = {GuestReg::Rcx, GuestReg::Rdx, GuestReg::R8, GuestReg::R9};

struct Arg {
    uint64_t bits = 0;
    bool is_float = false;
};

Arg I(uint64_t v) {
    return Arg{v, false};
}
Arg D(double v) {
    uint64_t bits;
    std::memcpy(&bits, &v, 8);
    return Arg{bits, true};
}
Arg F(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    return Arg{0xCCCCCCCC00000000ULL | bits, true};
}

bool diag_contains(const char* needle) {
    for (const std::string& d : com::diagnostics()) {
        if (d.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::span<const BootExport> table() {
    return std::span<const BootExport>(reinterpret_cast<const BootExport*>(pg6_export_thunks),
                                       pg6_export_thunk_count);
}

struct Fixture {
    VirtualMemoryArena arena;
    GuestThread thread{};
    uint64_t get_proc = 0;
    ExportThunkStats stats{};

    explicit Fixture(bool register_table = true) {
        reset_guest_runtime();
        reset_export_thunks();
        com::reset_com_bridge();
        cbk::reset_callbacks();
        win32::set_vmem_arena(&arena);
        com::set_guest_memory(&arena);
        com::clear_diagnostics();
        win32::register_all_shim_exports();
        CHECK(create_guest_thread(arena, 256 * 1024, thread));
        // What the IAT slot of an imported GetProcAddress holds at run time.
        get_proc = register_host_callable(&pg6_gthunk_get_proc_address, "GetProcAddress");
        if (register_table) {
            stats = register_export_thunks(table());
        }
    }

    ~Fixture() {
        com::reset_com_bridge();
        cbk::reset_callbacks();
        destroy_guest_thread(arena, thread);
        com::set_guest_memory(nullptr);
        win32::set_vmem_arena(nullptr);
        reset_export_thunks();
        reset_guest_runtime();
    }

    GuestState& state() { return thread.state; }

    uint64_t load64(uint64_t va) {
        uint64_t v = 0;
        CHECK(arena.read(va, &v, sizeof(v)));
        return v;
    }

    // The caller's side of a Win64 call: arguments in rcx rdx r8 r9 (or xmm0-3 for floats),
    // the rest on the stack after the shadow space, return address pushed; the callee is entered
    // through the dispatcher. Returns rax; xmm0 stays in state().
    uint64_t call(uint64_t target, std::initializer_list<Arg> arg_list) {
        const std::vector<Arg> args(arg_list);
        GuestState& s = state();
        const uint64_t saved_rsp = s.gpr[GuestReg::Rsp];
        const size_t stack_n = args.size() > 4 ? args.size() - 4 : 0;
        const uint64_t frame = ((0x20 + 8 * stack_n + 15) & ~uint64_t{15}) + 8;
        s.gpr[GuestReg::Rsp] -= frame;
        const uint64_t rsp = s.gpr[GuestReg::Rsp];
        for (size_t i = 0; i < args.size(); ++i) {
            if (i < 4) {
                if (args[i].is_float) {
                    s.xmm[i][0] = args[i].bits;
                    s.gpr[kRegs[i]] = 0xBAD0BAD0BAD0BAD0ULL;  // must not be read
                } else {
                    s.gpr[kRegs[i]] = args[i].bits;
                    s.xmm[i][0] = 0xBAD1BAD1BAD1BAD1ULL;
                }
            } else {
                const uint64_t value = args[i].bits;
                CHECK(arena.write(rsp + 0x20 + 8 * (i - 4), &value, sizeof(value)));
            }
        }
        s.gpr[GuestReg::Rsp] -= 8;
        const uint64_t ret = kReturnAddress;
        CHECK(arena.write(s.gpr[GuestReg::Rsp], &ret, sizeof(ret)));
        __playgta6_indirect_call(&s, target);
        CHECK(s.gpr[GuestReg::Rsp] == rsp);  // the thunk popped the return address
        s.gpr[GuestReg::Rsp] = saved_rsp;
        return s.gpr[GuestReg::Rax];
    }

    uint64_t cstring(const char* text) {
        const uint64_t va = com::guest_alloc(std::strlen(text) + 1);
        CHECK(com::write_guest(va, text, std::strlen(text) + 1));
        return va;
    }

    uint64_t module(const char* name) {
        const uint64_t handle = win32::GetModuleHandleA(name);
        CHECK(handle != 0);
        return handle;
    }

    // GetProcAddress as a guest caller invokes it.
    uint64_t proc(uint64_t module_handle, const char* name) {
        return call(get_proc, {I(module_handle), I(cstring(name))});
    }
    uint64_t proc(const char* dll, const char* name) { return proc(module(dll), name); }
    uint64_t proc_ordinal(const char* dll, uint32_t ordinal) {
        return call(get_proc, {I(module(dll)), I(ordinal)});
    }
};

double bits_f64(uint64_t bits) {
    double v;
    std::memcpy(&v, &bits, 8);
    return v;
}

float bits_f32(uint64_t bits) {
    const uint32_t low = static_cast<uint32_t>(bits);
    float v;
    std::memcpy(&v, &low, 4);
    return v;
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void test_registration_counts_and_budget() {
    Fixture fx;
    size_t no_thunk = 0;
    for (const BootExport& row : table()) {
        no_thunk += row.thunk == nullptr ? 1 : 0;
    }
    CHECK(no_thunk == 1);  // CRT remove: denied by the audit
    CHECK(fx.stats.rows == pg6_export_thunk_count);
    CHECK(fx.stats.unregistered == 0);  // every row is a registered export
    CHECK(fx.stats.unavailable == no_thunk);
    CHECK(fx.stats.failed == 0);
    CHECK(fx.stats.registered + fx.stats.shared + fx.stats.unavailable == fx.stats.rows);
    CHECK(fx.stats.registered >= 14);
    CHECK(fx.stats.registered < kMaxSyntheticTargets / 100);

    // Idempotent: nothing new on a second run, every proc already has its entry.
    const ExportThunkStats again = register_export_thunks(table());
    CHECK(again.registered == 0);
    CHECK(again.shared + again.unavailable == again.rows);
    CHECK(again.failed == 0);
}

void test_existing_entry_is_kept() {
    // A proc that already has a guest-callable entry (provider exports, exception thunks, ...)
    // keeps it: one guest address per shim.
    Fixture fx(false);
    const uint64_t tick = win32::resolve_import("KERNEL32.dll", "GetTickCount64");
    CHECK(tick != 0);
    const uint64_t custom = register_host_callable(table()[5].thunk, "custom", tick);
    CHECK(custom != 0);
    const ExportThunkStats stats = register_export_thunks(table());
    CHECK(stats.shared == 1);
    CHECK(fx.proc("KERNEL32.dll", "GetTickCount64") == custom);
    CHECK(guest_callable_for_proc(tick) == custom);
}

// ---------------------------------------------------------------------------
// GetProcAddress + calling the result
// ---------------------------------------------------------------------------

void test_get_tick_count_by_name() {
    Fixture fx;
    const uint64_t address = fx.proc("KERNEL32.dll", "GetTickCount64");
    CHECK(is_synthetic_address(address));
    const uint64_t before = pg6_win32_get_tick_count64();
    const uint64_t value = fx.call(address, {});
    const uint64_t after = pg6_win32_get_tick_count64();
    CHECK(value >= before && value <= after);
    // Resolving it again gives the same address.
    CHECK(fx.proc("KERNEL32.dll", "GetTickCount64") == address);
    CHECK(fx.state().gpr[GuestReg::Rsp] == fx.thread.stack_top);
}

void test_last_error_round_trip() {
    Fixture fx;
    const uint64_t set_error = fx.proc("KERNEL32.dll", "SetLastError");
    const uint64_t get_error = fx.proc("KERNEL32.dll", "GetLastError");
    CHECK(is_synthetic_address(set_error) && is_synthetic_address(get_error));
    CHECK(set_error != get_error);
    fx.state().gpr[GuestReg::Rax] = 0x77;
    CHECK(fx.call(set_error, {I(0xFFFF0000BEEFULL)}) == 0x77);  // void: rax untouched; DWORD arg
    CHECK(pg6_win32_get_last_error() == 0xBEEFu);
    CHECK(fx.call(get_error, {}) == 0xBEEFu);
    fx.state().gpr[GuestReg::Rax] = 0x78;
    CHECK(fx.call(set_error, {I(0)}) == 0x78);
    CHECK(fx.call(get_error, {}) == 0);
}

void test_string_function_with_pointer_arguments() {
    Fixture fx;
    const uint64_t strlen_address = fx.proc("api-ms-win-crt-string-l1-1-0.dll", "strlen");
    CHECK(is_synthetic_address(strlen_address));
    CHECK(fx.call(strlen_address, {I(fx.cstring("playgta6"))}) == 8);
    CHECK(fx.call(strlen_address, {I(fx.cstring(""))}) == 0);
}

void test_more_than_four_arguments() {
    Fixture fx;
    const uint64_t convert = fx.proc("KERNEL32.dll", "MultiByteToWideChar");
    CHECK(is_synthetic_address(convert));
    const char text[] = "hello world";
    const uint64_t src = fx.cstring(text);
    const uint64_t dst = com::guest_alloc(64 * sizeof(char16_t));
    constexpr uint64_t kCpUtf8 = 65001;
    // Arguments 5 and 6 come from the stack. cb = -1: convert up to and including the NUL.
    const uint64_t written = fx.call(
        convert, {I(kCpUtf8), I(0), I(src), I(0xFFFFFFFFULL), I(dst), I(64)});
    CHECK(static_cast<int32_t>(written) == 12);
    char16_t wide[12] = {};
    CHECK(com::read_guest(dst, wide, sizeof(wide)));
    bool same = true;
    for (size_t i = 0; i < 12; ++i) {
        same = same && wide[i] == static_cast<char16_t>(static_cast<unsigned char>(text[i]));
    }
    CHECK(same);
    CHECK(fx.state().gpr[GuestReg::Rsp] == fx.thread.stack_top);
}

void test_floating_point_crt_functions() {
    Fixture fx;
    const uint64_t pow_address = fx.proc("api-ms-win-crt-math-l1-1-0.dll", "pow");
    const uint64_t sinf_address = fx.proc("api-ms-win-crt-math-l1-1-0.dll", "sinf");
    CHECK(is_synthetic_address(pow_address) && is_synthetic_address(sinf_address));
    fx.call(pow_address, {D(2.0), D(10.0)});
    CHECK(bits_f64(fx.state().xmm[0][0]) == 1024.0);
    fx.call(sinf_address, {F(0.5f)});
    CHECK(std::fabs(bits_f32(fx.state().xmm[0][0]) - std::sin(0.5f)) < 1e-6f);
}

// SleepConditionVariableCS and WakeAllConditionVariable: the two APIs the real boot found
// without a guest-callable entry, resolved through an API-set name like the CRT does.
void test_condition_variable_apis_resolve_and_run() {
    Fixture fx;
    const uint64_t synch = win32::GetModuleHandleA("api-ms-win-core-synch-l1-2-0.dll");
    CHECK(synch != 0);
    const uint64_t sleep_cs = fx.proc(synch, "SleepConditionVariableCS");
    const uint64_t wake_all = fx.proc(synch, "WakeAllConditionVariable");
    const uint64_t init_cv = fx.proc(synch, "InitializeConditionVariable");
    CHECK(is_synthetic_address(sleep_cs) && is_synthetic_address(wake_all) && is_synthetic_address(init_cv));
    // The same addresses through KERNEL32 itself.
    CHECK(fx.proc("KERNEL32.dll", "SleepConditionVariableCS") == sleep_cs);
    CHECK(fx.proc("KERNEL32.dll", "WakeAllConditionVariable") == wake_all);

    const uint64_t cv = com::guest_alloc(64);
    const uint64_t cs = com::guest_alloc(128);
    fx.call(init_cv, {I(cv)});
    fx.call(fx.proc("KERNEL32.dll", "InitializeCriticalSection"), {I(cs)});
    fx.call(fx.proc("KERNEL32.dll", "EnterCriticalSection"), {I(cs)});
    // Nobody wakes it: times out after 20 ms (FALSE, ERROR_TIMEOUT) with the lock held again.
    const uint64_t slept = fx.call(sleep_cs, {I(cv), I(cs), I(20)});
    CHECK(static_cast<uint32_t>(slept) == 0);
    CHECK(pg6_win32_get_last_error() == 1460u);
    fx.call(wake_all, {I(cv)});
    fx.call(fx.proc("KERNEL32.dll", "LeaveCriticalSection"), {I(cs)});
}

void test_ordinal_lookup() {
    Fixture fx;
    const uint64_t by_name = fx.proc("WS2_32.dll", "closesocket");
    CHECK(is_synthetic_address(by_name));
    CHECK(fx.proc_ordinal("WS2_32.dll", 3) == by_name);  // MAKEINTRESOURCE(3)
    com::clear_diagnostics();
    CHECK(fx.proc_ordinal("WS2_32.dll", 60000) == 0);
    CHECK(pg6_win32_get_last_error() == win32::ERROR_PROC_NOT_FOUND);
}

void test_api_set_module_names() {
    Fixture fx;
    const uint64_t direct = fx.proc("KERNEL32.dll", "GetTickCount64");
    for (const char* api_set : {"api-ms-win-core-sysinfo-l1-1-0.dll", "API-MS-WIN-CORE-SYSINFO-L1-2-0"}) {
        const uint64_t module = win32::GetModuleHandleA(api_set);
        CHECK(module != 0);
        CHECK(fx.proc(module, "GetTickCount64") == direct);
    }
}

void test_denied_and_unlisted_exports_fail_cleanly() {
    Fixture fx;
    // Denied by the shim audit: NULL, ERROR_PROC_NOT_FOUND and the audit's reason.
    com::clear_diagnostics();
    win32::SetLastError(0);
    CHECK(fx.proc("api-ms-win-crt-filesystem-l1-1-0.dll", "remove") == 0);
    CHECK(pg6_win32_get_last_error() == win32::ERROR_PROC_NOT_FOUND);
    CHECK(diag_contains("host-only export"));
    CHECK(diag_contains("denied: pg6_crt_remove"));
    CHECK(diag_contains("deletes host files"));
    const uint64_t remove_proc = win32::resolve_import("api-ms-win-crt-filesystem-l1-1-0.dll", "remove");
    CHECK(remove_proc != 0);
    const char* reason = unavailable_reason_for_proc(remove_proc);
    CHECK(reason != nullptr && std::strncmp(reason, "denied: ", 8) == 0);
    CHECK(unavailable_reason_for_proc(0x1234) == nullptr);

    // A registered export the table does not list (it was generated for a handful of rows only).
    com::clear_diagnostics();
    CHECK(fx.proc("KERNEL32.dll", "CreateFileA") == 0);
    CHECK(pg6_win32_get_last_error() == win32::ERROR_PROC_NOT_FOUND);
    CHECK(diag_contains("not in the export thunk table"));

    // Not an export at all, and not a module.
    win32::SetLastError(0);
    CHECK(fx.proc("KERNEL32.dll", "NoSuchExport") == 0);
    CHECK(pg6_win32_get_last_error() == win32::ERROR_PROC_NOT_FOUND);
    CHECK(fx.call(fx.get_proc, {I(0x5555), I(fx.cstring("GetTickCount64"))}) == 0);
    CHECK(pg6_win32_get_last_error() == win32::ERROR_MOD_NOT_FOUND);
    CHECK(fx.state().gpr[GuestReg::Rsp] == fx.thread.stack_top);
}

// ---------------------------------------------------------------------------
// IAT thunk and GetProcAddress address
// ---------------------------------------------------------------------------

void test_iat_address_and_get_proc_address_are_interchangeable() {
    Fixture fx;
    const uint64_t from_gpa = fx.proc("KERNEL32.dll", "GetTickCount64");
    // A static import of the same shim: its IAT slot holds a per-slot synthetic address whose
    // thunk is the generated one.
    const Pg6ExportThunk* row = nullptr;
    for (size_t i = 0; i < pg6_export_thunk_count; ++i) {
        if (std::strcmp(pg6_export_thunks[i].name, "GetTickCount64") == 0) {
            row = &pg6_export_thunks[i];
        }
    }
    CHECK(row != nullptr && row->thunk != nullptr);
    const uint64_t image = com::guest_alloc(64);
    const ImportSlotBinding slot{0, row->thunk};
    CHECK(bind_import_slots(fx.arena, image, std::span<const ImportSlotBinding>(&slot, 1)) == 1);
    const uint64_t from_iat = fx.load64(image);
    CHECK(is_synthetic_address(from_iat));
    // Both dispatch to the same thunk and give the same kind of result; the addresses differ
    // (documented in guest/export_thunks.h): equality is not part of the contract.
    SyntheticTarget a;
    SyntheticTarget b;
    CHECK(describe_synthetic(from_gpa, &a) && describe_synthetic(from_iat, &b));
    CHECK(a.thunk == b.thunk && a.thunk == row->thunk);
    CHECK(!a.is_import && b.is_import);
    const uint64_t t1 = fx.call(from_iat, {});
    const uint64_t t2 = fx.call(from_gpa, {});
    CHECK(t2 >= t1 && t2 - t1 < 5000);
}

// ---------------------------------------------------------------------------
// Guest modules and forwarders
// ---------------------------------------------------------------------------

void test_guest_module_exports_are_real_guest_addresses() {
    Fixture fx;
    constexpr uint64_t kBase = 0x180000000ULL;
    CHECK(win32::register_guest_module("r18guest.dll", "C:\\r18guest.dll", kBase, 0x10000) == kBase);
    CHECK(win32::register_guest_export("r18guest.dll", "GuestFn", 5, kBase + 0x1230));
    CHECK(win32::register_export_forwarder("r18guest.dll", "FwdTick", 0, "KERNEL32.GetTickCount64"));
    CHECK(win32::register_export_forwarder("r18guest.dll", "FwdGuest", 0, "r18guest.GuestFn"));
    CHECK(win32::register_export_forwarder("r18guest.dll", "FwdDenied", 0,
                                           "api-ms-win-crt-filesystem-l1-1-0.remove"));
    const uint64_t module = fx.module("r18guest.dll");
    CHECK(fx.proc(module, "GuestFn") == kBase + 0x1230);  // a guest VA, not a synthetic address
    CHECK(fx.proc_ordinal("r18guest.dll", 5) == kBase + 0x1230);
    CHECK(fx.proc(module, "FwdGuest") == kBase + 0x1230);
    // A forwarder into a host DLL follows to the host shim's guest-callable entry.
    CHECK(fx.proc(module, "FwdTick") == fx.proc("KERNEL32.dll", "GetTickCount64"));
    CHECK(is_synthetic_address(fx.proc(module, "FwdTick")));
    // ... and fails like the target does.
    CHECK(fx.proc(module, "FwdDenied") == 0);
    CHECK(pg6_win32_get_last_error() == win32::ERROR_PROC_NOT_FOUND);
    // The main image's exports are guest VAs too.
    CHECK(win32::register_main_module("r18main.exe", "C:\\r18main.exe", 0x140000000ULL, 0x10000) != 0);
    CHECK(win32::register_guest_export("r18main.exe", "MainExport", 0, 0x140000000ULL + 0x40));
    CHECK(fx.proc(win32::GetModuleHandleA(nullptr), "MainExport") == 0x140000000ULL + 0x40);
}

}  // namespace

int main() {
    test_registration_counts_and_budget();
    test_existing_entry_is_kept();
    test_get_tick_count_by_name();
    test_last_error_round_trip();
    test_string_function_with_pointer_arguments();
    test_more_than_four_arguments();
    test_floating_point_crt_functions();
    test_condition_variable_apis_resolve_and_run();
    test_ordinal_lookup();
    test_api_set_module_names();
    test_denied_and_unlisted_exports_fail_cleanly();
    test_iat_address_and_get_proc_address_are_interchangeable();
    test_guest_module_exports_are_real_guest_addresses();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_export_thunks: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_export_thunks: all tests passed\n");
    return 0;
}
