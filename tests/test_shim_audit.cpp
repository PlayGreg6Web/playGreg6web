// Checks the shim audit table (runtime/os/shim_audit.h) against the shim headers and pins the ABI
// facts the `ok` rows rely on (MSVC layouts, Win64 `long` and `wchar_t` widths, errno numbering).

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>

#include "abi/playgta6_abi.h"
#include "os/crt_guest.h"
#include "os/shim_audit.h"
#include "os/ucrt_bridge.h"
#include "os/ucrt_ext.h"
#include "os/winsock_ordinals.h"
#include "os/xbox_middleware.h"
#include "os/xbox_offline.h"
#include "os/xbox_scarlett.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_shim_audit.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

struct AuditRow {
    const char* symbol;
    const void* address;  // taking the address fails to compile for a misspelled symbol
    const char* verdict;
    const char* note;
};

template <typename Fn>
const void* address_of(Fn* fn) {
    return reinterpret_cast<const void*>(fn);
}

#define PG6_AUDIT_ROW(sym, verdict, note) AuditRow{#sym, address_of(&sym), #verdict, note},
const AuditRow kRows[] = {PG6_SHIM_AUDIT(PG6_AUDIT_ROW)};
#undef PG6_AUDIT_ROW

struct GapRow {
    const char* dll;
    const char* name;
    const char* reason;
};

#define PG6_GAP_ROW(dll, name, reason) GapRow{dll, name, reason},
const GapRow kGaps[] = {PG6_IMPORT_GAPS(PG6_GAP_ROW) GapRow{nullptr, nullptr, nullptr}};
#undef PG6_GAP_ROW

bool starts_with(const char* s, const char* prefix) {
    return std::strncmp(s, prefix, std::strlen(prefix)) == 0;
}

void test_rows_are_well_formed() {
    std::set<std::string> seen;
    size_t ok = 0;
    for (const AuditRow& row : kRows) {
        CHECK(row.address != nullptr);
        CHECK(seen.insert(row.symbol).second);  // one row per shim
        CHECK(starts_with(row.symbol, "pg6_crt_") || starts_with(row.symbol, "pg6_mw_") ||
              starts_with(row.symbol, "pg6_xbox_") || starts_with(row.symbol, "pg6_ws2_"));
        const std::string verdict = row.verdict;
        CHECK(verdict == "ok" || verdict == "deny" || verdict == "pending");
        if (verdict == "ok") {
            ++ok;
        } else {
            CHECK(row.note != nullptr && row.note[0] != '\0');  // deny/pending say why
        }
    }
    // Every shim of the four audited families has a row (ucrt 186 + ucrt_ext 37, middleware 98,
    // xbox 16, winsock 25 registered + WSASetLastError, declared but not registered); stubs.rs
    // checks the same against the export tables.
    CHECK(seen.size() == 186u + 37u + 98u + 16u + 26u);
    CHECK(ok > 0);
    for (const GapRow& gap : kGaps) {
        if (gap.dll == nullptr) {
            break;  // sentinel
        }
        CHECK(gap.name != nullptr && gap.reason != nullptr);
        CHECK(gap.reason[0] != '\0');
    }
}

// MSVC x64 layouts the `ok` rows rely on.
void test_msvc_layouts() {
    CHECK(sizeof(Pg6CrtTm) == 36);
    CHECK(sizeof(Pg6CrtTimeb64) == 16);
    CHECK(sizeof(Pg6CrtStat64) == 56);
    CHECK(offsetof(Pg6CrtStat64, st_rdev) == 16);
    CHECK(offsetof(Pg6CrtStat64, st_size) == 24);
    CHECK(sizeof(Pg6CrtStdExceptionData) == 16);
    CHECK(sizeof(Pg6WsaData) == 408);
    CHECK(sizeof(Pg6FdSet) == 520);
    CHECK(sizeof(Pg6D3D12RaytracingPrebuildInfo) == 24);
}

// Win64 `long` is 32-bit and `wchar_t` 16-bit whatever the host (LP64 Linux / wasm64: 64 and
// 32): the shims use int32_t/uint32_t and uint16_t.
void test_long_and_wchar_widths() {
    char* end = nullptr;
    char big[] = "4294967296";
    pg6_crt_set_errno(0);
    CHECK(pg6_crt_strtol(big, &end, 10) == INT32_MAX);
    CHECK(*pg6_crt_errno() == ERANGE);
    char neg[] = "-4294967296";
    CHECK(pg6_crt_strtol(neg, &end, 10) == INT32_MIN);
    char ul[] = "4294967295";
    pg6_crt_set_errno(0);
    CHECK(pg6_crt_strtoul(ul, &end, 10) == UINT32_MAX);
    CHECK(*pg6_crt_errno() == 0);
    CHECK(pg6_crt_lroundf(2.5f) == 3);
    CHECK(pg6_crt_ftell(nullptr) == -1);  // int32 result, EINVAL
    const uint16_t wide[] = {'a', 0x20AC, 'b', 0};
    CHECK(pg6_crt_wcslen(wide) == 3);
    CHECK(pg6_crt_wcsrchr(wide, 0x20AC) == wide + 1);
}

// _errno() returns the address the guest reads and writes directly; the values used by the
// audited shims share the MSVC numbering.
void test_errno_pointer_semantics() {
    int32_t* slot = pg6_crt_errno();
    CHECK(slot != nullptr);
    CHECK(pg6_crt_errno() == slot);  // stable per thread
    *slot = 7;
    int32_t value = 0;
    CHECK(pg6_crt_get_errno(&value) == 0 && value == 7);
    pg6_crt_set_errno(0);
    CHECK(*slot == 0);
    CHECK(EINVAL == 22);   // MSVC EINVAL
    CHECK(ERANGE == 34);   // MSVC ERANGE
    CHECK(ENOENT == 2);    // MSVC ENOENT
    CHECK(ENOMEM == 12);   // MSVC ENOMEM
}

// MSVC _dtest/_fdtest codes and struct tm conversion.
void test_math_and_time_codes() {
    const double inf = INFINITY;
    const double nan = NAN;
    const double one = 1.0;
    const double zero = 0.0;
    CHECK(pg6_crt_dtest(&inf) == 1);
    CHECK(pg6_crt_dtest(&nan) == 2);
    CHECK(pg6_crt_dtest(&one) == -1);
    CHECK(pg6_crt_dtest(&zero) == 0);
    Pg6CrtTm tm{};
    const int64_t t = 86400 * 365;  // 1971-01-01T00:00:00Z
    CHECK(pg6_crt_gmtime64_s(&tm, &t) == 0);
    CHECK(tm.tm_year == 71 && tm.tm_mon == 0 && tm.tm_mday == 1 && tm.tm_wday == 5);
    CHECK(pg6_crt_mkgmtime64(&tm) == t);
}

// The callback-taking startup imports are guest-state thunks that keep guest addresses.
void test_crt_guest_handlers() {
    namespace cg = playgta6::os::crt_guest;
    using playgta6::GuestState;
    namespace reg = playgta6::GuestReg;
    cg::reset_handlers();
    GuestState state{};
    state.gpr[reg::Rsp] = 0x10000;
    state.gpr[reg::Rcx] = 0x140001000;
    state.gpr[reg::Rax] = 0xDEAD;
    pg6_gthunk_set_invalid_parameter_handler(&state);
    CHECK(state.gpr[reg::Rax] == 0);  // no previous handler
    CHECK(state.gpr[reg::Rsp] == 0x10008);  // popped the return address like ret
    CHECK(cg::invalid_parameter_handler() == 0x140001000);
    state.gpr[reg::Rcx] = 0x140002000;
    pg6_gthunk_set_invalid_parameter_handler(&state);
    CHECK(state.gpr[reg::Rax] == 0x140001000);
    CHECK(cg::invalid_parameter_handler() == 0x140002000);

    state.gpr[reg::Rcx] = 0x140003000;
    pg6_gthunk_setusermatherr(&state);
    CHECK(cg::user_matherr_handler() == 0x140003000);
    CHECK(state.gpr[reg::Rsp] == 0x10018);

    // _set_new_handler swaps; _callnewh without a handler reports "no handler" (0).
    state.gpr[reg::Rcx] = 0x140004000;
    pg6_gthunk_set_new_handler(&state);
    CHECK(state.gpr[reg::Rax] == 0);
    CHECK(cg::new_handler() == 0x140004000);
    cg::reset_handlers();
    state.gpr[reg::Rcx] = 64;
    state.gpr[reg::Rax] = 0xDEAD;
    pg6_gthunk_callnewh(&state);
    CHECK(state.gpr[reg::Rax] == 0);
    CHECK(state.gpr[reg::Rsp] == 0x10028);
}

// os/ucrt_ext.h: the UCRT internals msvcp140 depends on, "C" locale.
void test_ucrt_ext() {
    const uint16_t* ct = pg6_crt_pctype_func();
    CHECK(ct[-1] == 0);       // EOF
    CHECK(ct['A'] == 0x181);  // _UPPER | _HEX | alpha
    CHECK(ct['g'] == 0x102);  // _LOWER | alpha
    CHECK(ct['7'] == 0x84);   // _DIGIT | _HEX
    CHECK(ct[' '] == 0x48);   // _SPACE | _BLANK
    CHECK(ct['\t'] == 0x68);  // _SPACE | _CONTROL | _BLANK
    CHECK(ct['\n'] == 0x28);  // _SPACE | _CONTROL
    CHECK(ct['!'] == 0x10);   // _PUNCT
    CHECK(ct[0x7F] == 0x20);  // _CONTROL
    CHECK(ct[0xE9] == 0);     // nothing above 0x7F in the "C" locale
    CHECK(pg6_crt_mb_cur_max_func() == 1);
    CHECK(pg6_crt_lc_codepage_func() == 0);
    uint16_t** names = pg6_crt_lc_locale_name_func();
    for (int i = 0; i < 6; ++i) {
        CHECK(names[i] == nullptr);
    }
    CHECK(std::strcmp(pg6_crt_setlocale(0, nullptr), "C") == 0);
    CHECK(std::strcmp(pg6_crt_setlocale(0, ""), "C") == 0);
    CHECK(pg6_crt_setlocale(0, "de-DE") == nullptr);
    pg6_crt_lock_locales();
    pg6_crt_lock_locales();  // recursive
    pg6_crt_unlock_locales();
    pg6_crt_unlock_locales();

    CHECK(pg6_crt_btowc('A') == 'A');
    CHECK(pg6_crt_btowc(-1) == PG6_CRT_WEOF);
    CHECK(pg6_crt_atol("-99999999999") == INT32_MIN);
    CHECK(pg6_crt_iswspace(0xA0) != 0 && pg6_crt_iswspace('x') == 0);
    CHECK(pg6_crt_iswalnum(0xE9) != 0 && pg6_crt_iswalnum(0x20AC) == 0);
    int32_t exp = 0;
    CHECK(pg6_crt_frexp(8.0, &exp) == 0.5 && exp == 4);
    CHECK(pg6_crt_ldexp(0.5, 4) == 8.0);

    uint16_t small[3];
    const uint16_t abc[] = {'a', 'b', 'c', 0};
    CHECK(pg6_crt_wcscpy_s(small, 3, abc) == ERANGE && small[0] == 0);
    uint16_t big[8];
    CHECK(pg6_crt_wcscpy_s(big, 8, abc) == 0 && big[2] == 'c' && big[3] == 0);
    CHECK(pg6_crt_wcsnlen(abc, 2) == 2);
    CHECK(pg6_crt_strncnt("hello", 3) == 3);

    char* days = pg6_crt_getdays();
    CHECK(days != nullptr && std::strncmp(days, ":Sun:Sunday:Mon:Monday:", 23) == 0);
    pg6_crt_free(days);
    uint16_t* wmonths = pg6_crt_w_getmonths();
    CHECK(wmonths != nullptr && wmonths[0] == ':' && wmonths[1] == 'J' && wmonths[4] == ':');
    pg6_crt_free(wmonths);
    void* names_block = pg6_crt_gettnames();
    CHECK(names_block != nullptr);
    pg6_crt_free(names_block);

    Pg6CrtTm tm{};
    tm.tm_year = 124;  // 2024-03-05 (Tuesday)
    tm.tm_mon = 2;
    tm.tm_mday = 5;
    tm.tm_wday = 2;
    char narrow[64];
    CHECK(pg6_crt_strftime_lc(narrow, sizeof(narrow), "%a %b %d", &tm, nullptr) == 10);
    CHECK(std::strcmp(narrow, "Tue Mar 05") == 0);
    const uint16_t wfmt[] = {0x20AC, ' ', '%', 'Y', 0};
    uint16_t wide[16];
    CHECK(pg6_crt_wcsftime_lc(wide, 16, wfmt, &tm, nullptr) == 6);
    CHECK(wide[0] == 0x20AC && wide[2] == '2' && wide[5] == '4' && wide[6] == 0);
    CHECK(pg6_crt_wcsftime_lc(wide, 4, wfmt, &tm, nullptr) == 0 && wide[0] == 0);

    void* aligned = pg6_crt_aligned_malloc(100, 256);
    CHECK(aligned != nullptr && (reinterpret_cast<uintptr_t>(aligned) % 256) == 0);
    pg6_crt_aligned_free(aligned);
    uint32_t r = 0;
    CHECK(pg6_crt_rand_s(&r) == 0);
    CHECK(pg6_crt_rand_s(nullptr) == EINVAL);
}

}  // namespace

int main() {
    test_rows_are_well_formed();
    test_msvc_layouts();
    test_long_and_wchar_widths();
    test_errno_pointer_semantics();
    test_math_and_time_codes();
    test_crt_guest_handlers();
    test_ucrt_ext();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_shim_audit failed with %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("All playgta6_shim_audit_tests passed.\n");
    return 0;
}
