#include <cerrno>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "os/guest_heap.h"
#include "os/ucrt_bridge.h"
#include "os/win32_loader.h"

namespace {

bool default_guest_heap_owns_host(const void* host) {
    playgta6::os::GuestHeap& heap = playgta6::os::default_guest_heap();
    return heap.to_guest(host) != 0 && heap.arena().query(heap.to_guest(host)).state ==
                                           playgta6::os::MemState::Commit;
}

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_ucrt_bridge.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6::os::win32;

// Heap-allocated copy of an export name so GetProcAddress never mistakes a low
// static-data address on Wasm64 for an ordinal (MAKEINTRESOURCE).
struct HeapName {
    std::unique_ptr<char[]> chars;
    operator const char*() const { return chars.get(); }
};

HeapName heap_name(const char* literal) {
    const size_t len = std::strlen(literal) + 1;
    HeapName out{std::make_unique<char[]>(len)};
    std::memcpy(out.chars.get(), literal, len);
    return out;
}

int32_t call_vsprintf(uint64_t opts, char* buf, size_t cap, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    const int32_t rc = pg6_crt_stdio_common_vsprintf(opts, buf, cap, fmt, nullptr, ap);
    va_end(ap);
    return rc;
}

int32_t call_vsprintf_s(uint64_t opts, char* buf, size_t cap, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    const int32_t rc = pg6_crt_stdio_common_vsprintf_s(opts, buf, cap, fmt, nullptr, ap);
    va_end(ap);
    return rc;
}

int32_t call_vswprintf(uint64_t opts, uint16_t* buf, size_t cap, const uint16_t* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    const int32_t rc = pg6_crt_stdio_common_vswprintf(opts, buf, cap, fmt, nullptr, ap);
    va_end(ap);
    return rc;
}

int32_t call_vsnwprintf_s(uint64_t opts,
                          uint16_t* buf,
                          size_t cap,
                          size_t max_count,
                          const uint16_t* fmt,
                          ...) {
    va_list ap;
    va_start(ap, fmt);
    const int32_t rc =
        pg6_crt_stdio_common_vsnwprintf_s(opts, buf, cap, max_count, fmt, nullptr, ap);
    va_end(ap);
    return rc;
}

int32_t call_vsscanf(const char* input, size_t len, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    const int32_t rc = pg6_crt_stdio_common_vsscanf(0, input, len, fmt, nullptr, ap);
    va_end(ap);
    return rc;
}

int32_t call_vswscanf(const uint16_t* input, size_t len, const uint16_t* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    const int32_t rc = pg6_crt_stdio_common_vswscanf(0, input, len, fmt, nullptr, ap);
    va_end(ap);
    return rc;
}

std::vector<uint16_t> u16_lit(const char* ascii) {
    std::vector<uint16_t> v;
    while (*ascii != '\0') {
        v.push_back(static_cast<uint16_t>(static_cast<uint8_t>(*ascii++)));
    }
    v.push_back(0);
    return v;
}

// ---------------------------------------------------------------------------
// 1. Registration & Win32ModuleRegistry GetProcAddress verification (186 symbols / 12 DLLs)
// ---------------------------------------------------------------------------

void test_registration_and_lookup() {
    const uint32_t count1 = pg6_ucrt_register_all();
    CHECK(count1 == 186u);
    CHECK(pg6_ucrt_register_all() == 186u);  // idempotent

    size_t table_count = 0;
    const Pg6UcrtExport* table = pg6_ucrt_export_table(&table_count);
    CHECK(table != nullptr);
    CHECK(table_count == 186u);

    std::map<std::string, size_t> per_dll;
    std::set<std::string> seen;

    for (size_t i = 0; i < table_count; ++i) {
        const Pg6UcrtExport& e = table[i];
        CHECK(e.module != nullptr && e.name != nullptr && e.symbol != nullptr);
        CHECK(e.address != 0);
        CHECK(std::strncmp(e.symbol, "pg6_crt_", 8) == 0);
        CHECK(seen.insert(std::string(e.module) + "!" + e.name).second);
        ++per_dll[e.module];

        const Handle mod = GetModuleHandleA(e.module);
        CHECK(mod != kNullHandle);
        CHECK(GetProcAddress(mod, heap_name(e.name)) == e.address);
        CHECK(GetProcAddressByName(mod, e.name) == e.address);
        CHECK(resolve_import(e.module, e.name) == e.address);
    }

    CHECK(per_dll.size() == 12u);
    CHECK(per_dll["api-ms-win-crt-convert-l1-1-0.dll"] == 12u);
    CHECK(per_dll["api-ms-win-crt-filesystem-l1-1-0.dll"] == 4u);
    CHECK(per_dll["api-ms-win-crt-heap-l1-1-0.dll"] == 5u);
    CHECK(per_dll["api-ms-win-crt-locale-l1-1-0.dll"] == 2u);
    CHECK(per_dll["api-ms-win-crt-math-l1-1-0.dll"] == 41u);
    CHECK(per_dll["api-ms-win-crt-runtime-l1-1-0.dll"] == 25u);
    CHECK(per_dll["api-ms-win-crt-stdio-l1-1-0.dll"] == 29u);
    CHECK(per_dll["api-ms-win-crt-string-l1-1-0.dll"] == 31u);
    CHECK(per_dll["api-ms-win-crt-time-l1-1-0.dll"] == 9u);
    CHECK(per_dll["api-ms-win-crt-utility-l1-1-0.dll"] == 4u);
    CHECK(per_dll["VCRUNTIME140.dll"] == 23u);
    CHECK(per_dll["VCRUNTIME140_1.dll"] == 1u);

    // Fallback via ucrtbase.dll works as well.
    const Handle ucrt = GetModuleHandleA("ucrtbase.dll");
    CHECK(ucrt != kNullHandle);
    CHECK(GetProcAddress(ucrt, heap_name("malloc")) == proc_address(&pg6_crt_malloc));
    CHECK(GetProcAddress(ucrt, heap_name("_initterm")) == proc_address(&pg6_crt_initterm));
}

// ---------------------------------------------------------------------------
// 2. Runtime startup: _initterm, _initterm_e, onexit table, errno, exit hooks
// ---------------------------------------------------------------------------

std::vector<int> g_call_log;

void init_step_1() { g_call_log.push_back(1); }
void init_step_2() { g_call_log.push_back(2); }
int32_t init_e_ok() {
    g_call_log.push_back(10);
    return 0;
}
int32_t init_e_fail() {
    g_call_log.push_back(20);
    return 77;
}
int32_t init_e_unreached() {
    g_call_log.push_back(30);
    return 0;
}
int32_t onexit_a() {
    g_call_log.push_back(101);
    return 0;
}
int32_t onexit_b() {
    g_call_log.push_back(102);
    return 0;
}
void atexit_cb_1() { g_call_log.push_back(201); }
void atexit_cb_2() { g_call_log.push_back(202); }

int32_t g_hook_code = -1;
int32_t g_hook_kind = -1;
int32_t g_hook_hits = 0;

void test_exit_hook(int32_t code, int32_t kind) {
    g_hook_code = code;
    g_hook_kind = kind;
    ++g_hook_hits;
}

int32_t g_purecall_hits = 0;
void test_purecall_handler() { ++g_purecall_hits; }

void test_runtime_lifecycle() {
    pg6_crt_reset_runtime_state();
    g_call_log.clear();

    Pg6CrtPvfv init_list[] = {nullptr, &init_step_1, nullptr, &init_step_2};
    pg6_crt_initterm(init_list, init_list + 4);
    CHECK(g_call_log.size() == 2u && g_call_log[0] == 1 && g_call_log[1] == 2);

    g_call_log.clear();
    Pg6CrtPifv init_e_list[] = {&init_e_ok, nullptr, &init_e_fail, &init_e_unreached};
    CHECK(pg6_crt_initterm_e(init_e_list, init_e_list + 4) == 77);
    CHECK(g_call_log.size() == 2u && g_call_log[0] == 10 && g_call_log[1] == 20);

    // Onexit table initialization, registration (>32 growth test), LIFO execution.
    Pg6CrtOnexitTable tbl{};
    CHECK(pg6_crt_initialize_onexit_table(&tbl) == 0);
    CHECK(tbl.first != nullptr && tbl.last == tbl.first && tbl.end > tbl.first);
    CHECK(pg6_crt_register_onexit_function(&tbl, &onexit_a) == 0);
    CHECK(pg6_crt_register_onexit_function(&tbl, &onexit_b) == 0);
    // Grow beyond initial capacity of 32.
    for (int i = 0; i < 40; ++i) {
        CHECK(pg6_crt_register_onexit_function(&tbl, nullptr) == 0);
    }
    g_call_log.clear();
    CHECK(pg6_crt_execute_onexit_table(&tbl) == 0);
    CHECK(g_call_log.size() == 2u && g_call_log[0] == 102 && g_call_log[1] == 101);

    // errno helpers.
    CHECK(pg6_crt_set_errno(42) == 0);
    CHECK(*pg6_crt_errno() == 42);
    int32_t err_val = 0;
    CHECK(pg6_crt_get_errno(&err_val) == 0 && err_val == 42);
    CHECK(pg6_crt_get_errno(nullptr) == EINVAL);

    // argc/argv/command line.
    CHECK(*pg6_crt_p___argc() == 1);
    {
        // argv is an array of guest VAs in guest memory.
        playgta6::os::GuestHeap& heap = playgta6::os::default_guest_heap();
        char** argv_va = *pg6_crt_p___argv();
        CHECK(argv_va != nullptr);
        const auto* argv_host = reinterpret_cast<const uint64_t*>(
            heap.to_host(reinterpret_cast<uintptr_t>(argv_va)));
        CHECK(argv_host != nullptr && argv_host[0] != 0 && argv_host[1] == 0);
        CHECK(heap.owns(argv_host[0]));
        CHECK(std::strcmp(reinterpret_cast<const char*>(heap.to_host(argv_host[0])),
                          "game_scarlett_master.exe") == 0);
    }
    CHECK(pg6_crt_get_narrow_winmain_command_line() != nullptr);
    CHECK(default_guest_heap_owns_host(pg6_crt_get_narrow_winmain_command_line()));
    CHECK(pg6_crt_configure_narrow_argv(1) == 0);
    CHECK(pg6_crt_initialize_narrow_environment() == 0);

    // Safe non-crashing exit / abort / terminate / purecall hooks.
    g_hook_hits = 0;
    pg6_crt_set_exit_hook(&test_exit_hook);
    g_call_log.clear();
    CHECK(pg6_crt_crt_atexit(&atexit_cb_1) == 0);
    CHECK(pg6_crt_crt_atexit(&atexit_cb_2) == 0);
    pg6_crt_exit(5);
    CHECK(pg6_crt_get_last_exit_code() == 5);
    CHECK(g_hook_hits == 1 && g_hook_code == 5 && g_hook_kind == PG6_CRT_EXIT_NORMAL);
    CHECK(g_call_log.size() == 2u && g_call_log[0] == 202 && g_call_log[1] == 201);

    pg6_crt__exit(9);
    CHECK(g_hook_hits == 2 && g_hook_code == 9 && g_hook_kind == PG6_CRT_EXIT_QUICK);
    pg6_crt_abort();
    CHECK(g_hook_hits == 3 && g_hook_kind == PG6_CRT_EXIT_ABORT);
    pg6_crt_terminate();
    CHECK(g_hook_hits == 4 && g_hook_kind == PG6_CRT_EXIT_TERMINATE);
    pg6_crt_std_terminate();
    CHECK(g_hook_hits == 5 && g_hook_kind == PG6_CRT_EXIT_TERMINATE);
    pg6_crt_invalid_parameter_noinfo();
    CHECK(g_hook_hits == 6 && g_hook_kind == PG6_CRT_EXIT_INVALID_PARAM);

    g_purecall_hits = 0;
    CHECK(pg6_crt_set_purecall_handler(&test_purecall_handler) == nullptr);
    pg6_crt_purecall();
    CHECK(g_purecall_hits == 1);
    pg6_crt_reset_runtime_state();
}

// ---------------------------------------------------------------------------
// 3. Heap, Convert, Locale
// ---------------------------------------------------------------------------

void test_heap_convert_locale() {
    void* p = pg6_crt_malloc(64);
    CHECK(p != nullptr);
    CHECK(default_guest_heap_owns_host(p));  // the CRT heap lives in the guest address space
    pg6_crt_memset(p, 0x5A, 64);
    p = pg6_crt_realloc(p, 128);
    CHECK(p != nullptr);
    CHECK(static_cast<uint8_t*>(p)[0] == 0x5A && static_cast<uint8_t*>(p)[63] == 0x5A);
    pg6_crt_free(p);

    auto* z = static_cast<uint32_t*>(pg6_crt_calloc(16, sizeof(uint32_t)));
    CHECK(z != nullptr);
    for (int i = 0; i < 16; ++i) {
        CHECK(z[i] == 0u);
    }
    pg6_crt_free(z);

    CHECK(pg6_crt_set_new_mode(1) == 0);
    CHECK(pg6_crt_set_new_mode(0) == 1);

    // Convert.
    char buf[64];
    CHECK(std::strcmp(pg6_crt_itoa(-12345, buf, 10), "-12345") == 0);
    CHECK(std::strcmp(pg6_crt_itoa(255, buf, 16), "ff") == 0);
    CHECK(std::strcmp(pg6_crt_itoa(0, buf, 10), "0") == 0);
    CHECK(pg6_crt_atoi("  -42") == -42);
    CHECK(pg6_crt_atoll("9000000000000") == 9000000000000LL);
    CHECK(std::fabs(pg6_crt_atof("3.125") - 3.125) < 1e-12);

    char* end = nullptr;
    CHECK(pg6_crt_strtol("0x1A!", &end, 16) == 26 && end != nullptr && *end == '!');
    CHECK(pg6_crt_strtoul("4294967295", &end, 10) == 4294967295u);
    CHECK(pg6_crt_strtoll("-9223372036854775807", &end, 10) == -9223372036854775807LL);
    CHECK(pg6_crt_strtoull("18446744073709551615", &end, 10) == 18446744073709551615ULL);
    CHECK(pg6_crt_strtoi64("-10000000000", &end, 10) == -10000000000LL);
    CHECK(pg6_crt_strtoui64("10000000000", &end, 10) == 10000000000ULL);
    CHECK(std::fabs(pg6_crt_strtod("2.5e2", &end) - 250.0) < 1e-9);
    CHECK(std::fabs(pg6_crt_strtof("1.25f", &end) - 1.25f) < 1e-6f);

    // Locale.
    CHECK(pg6_crt_configthreadlocale(0) == 2);
    CHECK(pg6_crt_configthreadlocale(1) == 2);
    CHECK(pg6_crt_configthreadlocale(2) == 1);
    Pg6CrtLconv* lc = pg6_crt_localeconv();
    CHECK(lc != nullptr && lc->decimal_point != nullptr && lc->decimal_point[0] == '.');
}

// ---------------------------------------------------------------------------
// 4. Math (trig, exp/log/pow, _dtest, _fdtest, _fdsign)
// ---------------------------------------------------------------------------

void test_math() {
    CHECK(std::fabs(pg6_crt_sin(0.0)) < 1e-12);
    CHECK(std::fabs(pg6_crt_cos(0.0) - 1.0) < 1e-12);
    CHECK(std::fabs(pg6_crt_tan(0.0)) < 1e-12);
    CHECK(std::fabs(pg6_crt_pow(2.0, 10.0) - 1024.0) < 1e-9);
    CHECK(std::fabs(pg6_crt_powf(3.0f, 3.0f) - 27.0f) < 1e-5f);
    CHECK(std::fabs(pg6_crt_exp2(8.0) - 256.0) < 1e-9);
    CHECK(std::fabs(pg6_crt_log2(256.0) - 8.0) < 1e-9);
    CHECK(std::fabs(pg6_crt_cbrtf(27.0f) - 3.0f) < 1e-5f);
    CHECK(pg6_crt_ceilf(2.1f) == 3.0f);
    CHECK(pg6_crt_lroundf(2.6f) == 3);
    CHECK(std::fabs(pg6_crt_fmod(5.5, 2.0) - 1.5) < 1e-12);
    CHECK(std::fabs(pg6_crt_remainderf(5.5f, 2.0f) - (-0.5f)) < 1e-5f);

    double ipart = 0.0;
    CHECK(std::fabs(pg6_crt_modf(3.75, &ipart) - 0.75) < 1e-12 && ipart == 3.0);

    const double d_zero = 0.0;
    const double d_norm = 42.0;
    const double d_inf = std::numeric_limits<double>::infinity();
    const double d_nan = std::numeric_limits<double>::quiet_NaN();
    const double d_sub = std::numeric_limits<double>::denorm_min();
    CHECK(pg6_crt_dtest(&d_zero) == PG6_CRT_FP_ZERO);
    CHECK(pg6_crt_dtest(&d_norm) == PG6_CRT_FP_FINITE);
    CHECK(pg6_crt_dtest(&d_inf) == PG6_CRT_FP_INF);
    CHECK(pg6_crt_dtest(&d_nan) == PG6_CRT_FP_NAN);
    CHECK(pg6_crt_dtest(&d_sub) == PG6_CRT_FP_DENORM);

    const float f_neg_zero = -0.0f;
    const float f_pos = 1.0f;
    const float f_inf = -std::numeric_limits<float>::infinity();
    CHECK(pg6_crt_fdsign(f_neg_zero) != 0);
    CHECK(pg6_crt_fdsign(f_pos) == 0);
    CHECK(pg6_crt_fdtest(&f_inf) == PG6_CRT_FP_INF);
    CHECK(pg6_crt_nextafterf(1.0f, 2.0f) > 1.0f);
    CHECK(pg6_crt_nexttowardf(1.0f, 2.0) > 1.0f);
}

// ---------------------------------------------------------------------------
// 5. Strings (narrow, secure *_s, UTF-16 uint16_t*) & VCRUNTIME mem* ops
// ---------------------------------------------------------------------------

void test_strings_and_memory() {
    CHECK(pg6_crt_stricmp("ViceCity", "vicecity") == 0);
    CHECK(pg6_crt_strnicmp("ViceCity", "VICE_99", 4) == 0);
    CHECK(pg6_crt_stricmp("abc", "abd") < 0);

    char sbuf[16];
    CHECK(pg6_crt_strcpy_s(sbuf, sizeof(sbuf), "hello") == 0);
    CHECK(pg6_crt_strcat_s(sbuf, sizeof(sbuf), " world") == 0);
    CHECK(pg6_crt_strcmp(sbuf, "hello world") == 0);
    CHECK(pg6_crt_strcpy_s(sbuf, 4, "overflow") == ERANGE);
    CHECK(sbuf[0] == '\0');

    CHECK(pg6_crt_strncpy_s(sbuf, 5, "abcdefgh", PG6_CRT_TRUNCATE) == PG6_CRT_STRUNCATE);
    CHECK(pg6_crt_strcmp(sbuf, "abcd") == 0);

    char tok_buf[] = "a,b;c";
    char* ctx = nullptr;
    CHECK(pg6_crt_strcmp(pg6_crt_strtok_s(tok_buf, ",;", &ctx), "a") == 0);
    CHECK(pg6_crt_strcmp(pg6_crt_strtok_s(nullptr, ",;", &ctx), "b") == 0);
    CHECK(pg6_crt_strcmp(pg6_crt_strtok_s(nullptr, ",;", &ctx), "c") == 0);
    CHECK(pg6_crt_strtok_s(nullptr, ",;", &ctx) == nullptr);

    // UTF-16 wide strings.
    const auto w_hello = u16_lit("Hello");
    const auto w_HELLO = u16_lit("HELLO");
    const auto w_world = u16_lit(" World");
    CHECK(pg6_crt_wcslen(w_hello.data()) == 5u);
    CHECK(pg6_crt_wcscmp(w_hello.data(), w_hello.data()) == 0);
    CHECK(pg6_crt_wcscmp(w_hello.data(), w_HELLO.data()) > 0);
    CHECK(pg6_crt_wcsicoll(w_hello.data(), w_HELLO.data()) == 0);
    CHECK(pg6_crt_wcsnicmp(w_hello.data(), w_HELLO.data(), 5) == 0);

    uint16_t wdst[32] = {};
    pg6_crt_wcsncpy(wdst, w_hello.data(), 32);
    CHECK(pg6_crt_wcscat_s(wdst, 32, w_world.data()) == 0);
    CHECK(pg6_crt_wcslen(wdst) == 11u);
    uint16_t* last_l = pg6_crt_wcsrchr(wdst, static_cast<uint16_t>('l'));
    CHECK(last_l != nullptr && (last_l - wdst) == 9);

    // VCRUNTIME memory primitives (including overlapping memmove).
    uint8_t mem[16];
    for (uint8_t i = 0; i < 16; ++i) {
        mem[i] = i;
    }
    pg6_crt_memmove(mem + 4, mem, 8);
    CHECK(mem[4] == 0 && mem[11] == 7);
    uint8_t copy[8] = {};
    pg6_crt_memcpy(copy, mem + 4, 8);
    CHECK(pg6_crt_memcmp(copy, mem + 4, 8) == 0);
    CHECK(pg6_crt_memchr(copy, 5, 8) == copy + 5);
    CHECK(pg6_crt_strstr("playgta6web", "gta6") != nullptr);
    CHECK(*pg6_crt_strchr("playgta6web", '6') == '6');
    CHECK(*pg6_crt_strrchr("abca", 'a') == 'a');

    // VCRUNTIME C++ exception & type_info helpers.
    // `what` holds guest VAs (the struct lives in guest memory), so the source string is a guest
    // heap block and the copy is checked through the guest translation.
    playgta6::os::GuestHeap& gheap = playgta6::os::default_guest_heap();
    char* boom = playgta6::os::guest_crt::strdup("boom");
    CHECK(boom != nullptr);
    Pg6CrtStdExceptionData ex_src{
        reinterpret_cast<const char*>(static_cast<uintptr_t>(gheap.to_guest(boom))), 1};
    Pg6CrtStdExceptionData ex_dst{nullptr, 0};
    pg6_crt_std_exception_copy(&ex_src, &ex_dst);
    CHECK(ex_dst.do_free == 1 && ex_dst.what != nullptr && ex_dst.what != ex_src.what);
    CHECK(std::strcmp(reinterpret_cast<const char*>(
                          gheap.to_host(reinterpret_cast<uintptr_t>(ex_dst.what))),
                      "boom") == 0);
    pg6_crt_std_exception_destroy(&ex_dst);
    playgta6::os::guest_crt::free(boom);
    CHECK(ex_dst.what == nullptr && ex_dst.do_free == 0);

    struct FakeTypeInfo {
        const char* undec;
        char dec[16];
    };
    FakeTypeInfo ti_a{nullptr, ".?AVFoo@@"};
    FakeTypeInfo ti_b{nullptr, ".?AVFoo@@"};
    FakeTypeInfo ti_c{nullptr, ".?AVBar@@"};
    CHECK(pg6_crt_std_type_info_compare(reinterpret_cast<const Pg6CrtStdTypeInfoData*>(&ti_a),
                                        reinterpret_cast<const Pg6CrtStdTypeInfoData*>(&ti_b)) ==
          0);
    CHECK(pg6_crt_std_type_info_compare(reinterpret_cast<const Pg6CrtStdTypeInfoData*>(&ti_a),
                                        reinterpret_cast<const Pg6CrtStdTypeInfoData*>(&ti_c)) > 0);
}

// ---------------------------------------------------------------------------
// 6. Stdio formatting, scanning, and file I/O
// ---------------------------------------------------------------------------

void test_stdio() {
    CHECK(pg6_crt_acrt_iob_func(0) == stdin);
    CHECK(pg6_crt_acrt_iob_func(1) == stdout);
    CHECK(pg6_crt_acrt_iob_func(2) == stderr);
    CHECK(pg6_crt_acrt_iob_func(3) == nullptr);

    char out[64] = {};
    CHECK(call_vsprintf(0x2, out, sizeof(out), "val=%d hex=%x", 42, 0xAB) == 13);
    CHECK(std::strcmp(out, "val=42 hex=ab") == 0);
    CHECK(call_vsprintf_s(0, out, 4, "too long") == -1);
    CHECK(out[0] == '\0');

    int a = 0;
    int b = 0;
    CHECK(call_vsscanf("12 34", 5, "%d %d", &a, &b) == 2 && a == 12 && b == 34);

    // Wide printf / scanf via UTF-16.
    const auto wfmt = u16_lit("n=%d s=%s");
    const auto warg = u16_lit("GTA6");
    uint16_t wout[64] = {};
    const int32_t wn = call_vswprintf(0x2, wout, 64, wfmt.data(), 2026, warg.data());
    CHECK(wn == 13);
    CHECK(pg6_crt_wcscmp(wout, u16_lit("n=2026 s=GTA6").data()) == 0);

    uint16_t wtrunc[6] = {};
    CHECK(call_vsnwprintf_s(0, wtrunc, 6, PG6_CRT_TRUNCATE, wfmt.data(), 2026, warg.data()) == -1);
    CHECK(pg6_crt_wcscmp(wtrunc, u16_lit("n=202").data()) == 0);

    int wx = 0;
    int wy = 0;
    const auto win = u16_lit("77 88");
    const auto wsfmt = u16_lit("%d %d");
    CHECK(call_vswscanf(win.data(), 5, wsfmt.data(), &wx, &wy) == 2 && wx == 77 && wy == 88);

    // File round-trip with fopen/_wfopen/_wstat64/_fseeki64/fgetpos/fsetpos/remove.
    const char* tmp_name = "pg6_ucrt_test_tmp.bin";
    FILE* f = pg6_crt_fopen(tmp_name, "wb");
    CHECK(f != nullptr);
    if (f != nullptr) {
        pg6_crt_lock_file(f);
        const char payload[] = "PlayGTA6Web";
        CHECK(pg6_crt_fwrite(payload, 1, 11, f) == 11u);
        pg6_crt_unlock_file(f);
        CHECK(pg6_crt_fclose(f) == 0);
    }

    const auto wtmp = u16_lit(tmp_name);
    Pg6CrtStat64 st{};
    CHECK(pg6_crt_wstat64(wtmp.data(), &st) == 0);
    CHECK(st.st_size == 11);

    FILE* rf = pg6_crt_wfopen(wtmp.data(), u16_lit("rb").data());
    CHECK(rf != nullptr);
    if (rf != nullptr) {
        CHECK(pg6_crt_fseeki64(rf, 4, SEEK_SET) == 0);
        int64_t pos = 0;
        CHECK(pg6_crt_fgetpos(rf, &pos) == 0 && pos == 4);
        char part[5] = {};
        CHECK(pg6_crt_fread(part, 1, 4, rf) == 4u);
        CHECK(std::strcmp(part, "GTA6") == 0);
        CHECK(pg6_crt_fsetpos(rf, &pos) == 0);
        CHECK(pg6_crt_fgetc(rf) == 'G');
        CHECK(pg6_crt_fclose(rf) == 0);
    }
    CHECK(pg6_crt_remove(tmp_name) == 0);
}

// ---------------------------------------------------------------------------
// 7. Time (_time64, _gmtime64_s, _mkgmtime64, _difftime64, strftime) & Utility (qsort/bsearch)
// ---------------------------------------------------------------------------

int32_t cmp_i32(const void* a, const void* b) {
    const int32_t va = *static_cast<const int32_t*>(a);
    const int32_t vb = *static_cast<const int32_t*>(b);
    return (va > vb) - (va < vb);
}

void test_time_and_utility() {
    int64_t now_arg = 0;
    const int64_t now = pg6_crt_time64(&now_arg);
    CHECK(now > 1700000000LL && now == now_arg);
    CHECK(pg6_crt_difftime64(2000, 1500) == 500.0);

    Pg6CrtTimeb64 tb{};
    pg6_crt_ftime64(&tb);
    CHECK(tb.time >= now - 2 && tb.millitm < 1000u);

    // Known UTC timestamp: 2026-11-19 00:00:00 UTC -> 1795046400
    const int64_t ts = 1795046400LL;
    Pg6CrtTm utc{};
    CHECK(pg6_crt_gmtime64_s(&utc, &ts) == 0);
    CHECK(utc.tm_year == 126 && utc.tm_mon == 10 && utc.tm_mday == 19);
    CHECK(utc.tm_hour == 0 && utc.tm_min == 0 && utc.tm_sec == 0);
    CHECK(utc.tm_wday == 4);  // Thursday
    CHECK(pg6_crt_mkgmtime64(&utc) == ts);

    char tbuf[32] = {};
    CHECK(pg6_crt_strftime(tbuf, sizeof(tbuf), "%Y-%m-%d", &utc) == 10u);
    CHECK(std::strcmp(tbuf, "2026-11-19") == 0);

    // Localtime round-trip.
    Pg6CrtTm loc{};
    CHECK(pg6_crt_localtime64_s(&loc, &ts) == 0);

    // qsort & bsearch & deterministic srand/rand.
    int32_t arr[] = {42, -5, 17, 0, 99, 3};
    pg6_crt_qsort(arr, 6, sizeof(int32_t), &cmp_i32);
    CHECK(arr[0] == -5 && arr[1] == 0 && arr[2] == 3 && arr[3] == 17 && arr[4] == 42 &&
          arr[5] == 99);
    const int32_t key = 17;
    auto* found = static_cast<int32_t*>(pg6_crt_bsearch(&key, arr, 6, sizeof(int32_t), &cmp_i32));
    CHECK(found != nullptr && *found == 17);
    const int32_t missing = 18;
    CHECK(pg6_crt_bsearch(&missing, arr, 6, sizeof(int32_t), &cmp_i32) == nullptr);

    pg6_crt_srand(12345u);
    const int32_t r1 = pg6_crt_rand();
    const int32_t r2 = pg6_crt_rand();
    pg6_crt_srand(12345u);
    CHECK(pg6_crt_rand() == r1 && pg6_crt_rand() == r2);
    CHECK(r1 >= 0 && r1 <= 0x7FFF);
}

}  // namespace

int main() {
    test_registration_and_lookup();
    test_runtime_lifecycle();
    test_heap_convert_locale();
    test_math();
    test_strings_and_memory();
    test_stdio();
    test_time_and_utility();

    if (g_failures != 0) {
        std::fprintf(stderr, "test_ucrt_bridge failed with %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("All playgta6_ucrt_bridge_tests passed.\n");
    return 0;
}
