#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS 1
#endif
#if !defined(_MSC_VER) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include "os/ucrt_bridge.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <sys/stat.h>

#include "os/guest_heap.h"
#include "os/win32_loader.h"

namespace {

// ---------------------------------------------------------------------------
// Thread-local & process-global CRT state
// ---------------------------------------------------------------------------

thread_local int32_t g_crt_errno = 0;
thread_local int32_t g_thread_locale_mode = 2;  // _DISABLE_PER_THREAD_LOCALE
thread_local char* g_strtok_state = nullptr;
thread_local Pg6CrtTm g_gmtime_tls{};
thread_local uint32_t g_rand_state = 1u;
thread_local void* g_current_exception = nullptr;
thread_local void* g_current_exception_context = nullptr;

std::mutex g_runtime_mutex;
std::vector<Pg6CrtPvfv> g_atexit_callbacks;
Pg6CrtExitHook g_exit_hook = nullptr;
Pg6CrtInvalidParamHandler g_invalid_param_handler = nullptr;
Pg6CrtPurecallHandler g_purecall_handler = nullptr;
Pg6CrtUserMathErrHandler g_user_math_err_handler = nullptr;
Pg6CrtTlsAtexitCallback g_tls_atexit_callback = nullptr;
std::atomic<int32_t> g_new_mode{0};
std::atomic<int32_t> g_app_type{2};  // _crt_gui_app
std::atomic<int32_t> g_fmode{0x4000};
std::atomic<int32_t> g_last_exit_code{0};

int32_t g_commode = 0;
int32_t g_argc = 1;
const char kProgName[] = "game_scarlett_master.exe";

// argv and the command line live in guest memory (guest heap, O24). g_argv holds the guest VA of
// the argv array (an array of guest VAs); g_cmdline_va the guest VA of the command line string.
// __p___argv returns the address of g_argv itself, which the import bridge round-trips.
std::mutex g_guest_state_mutex;
playgta6::os::GuestHeap* g_guest_state_heap = nullptr;
char** g_argv = nullptr;
uint64_t g_cmdline_va = 0;

void ensure_guest_crt_state() {
    playgta6::os::GuestHeap& heap = playgta6::os::default_guest_heap();
    std::lock_guard<std::mutex> lock(g_guest_state_mutex);
    if (g_guest_state_heap == &heap) {
        return;
    }
    g_guest_state_heap = &heap;
    using playgta6::os::HeapFamily;
    const uint64_t name_va = heap.alloc(sizeof(kProgName), 16, HeapFamily::Crt);
    const uint64_t argv_va = heap.alloc(2 * sizeof(uint64_t), 16, HeapFamily::Crt);
    g_cmdline_va = heap.alloc(1, 16, HeapFamily::Crt, playgta6::os::kHeapZeroMemory);
    if (name_va == 0 || argv_va == 0 || g_cmdline_va == 0) {
        g_argv = nullptr;
        return;
    }
    std::memcpy(heap.to_host(name_va), kProgName, sizeof(kProgName));
    const uint64_t table[2] = {name_va, 0};
    std::memcpy(heap.to_host(argv_va), table, sizeof(table));
    g_argv = reinterpret_cast<char**>(static_cast<uintptr_t>(argv_va));
}

// Static "C" locale struct matching Win64 UCRT `struct lconv`.
char g_lconv_dot[] = ".";
char g_lconv_empty[] = "";
uint16_t g_lconv_wdot[] = {static_cast<uint16_t>('.'), 0};
uint16_t g_lconv_wempty[] = {0};

Pg6CrtLconv g_c_lconv = {
    g_lconv_dot,     // decimal_point
    g_lconv_empty,   // thousands_sep
    g_lconv_empty,   // grouping
    g_lconv_empty,   // int_curr_symbol
    g_lconv_empty,   // currency_symbol
    g_lconv_empty,   // mon_decimal_point
    g_lconv_empty,   // mon_thousands_sep
    g_lconv_empty,   // mon_grouping
    g_lconv_empty,   // positive_sign
    g_lconv_empty,   // negative_sign
    CHAR_MAX,        // int_frac_digits
    CHAR_MAX,        // frac_digits
    CHAR_MAX,        // p_cs_precedes
    CHAR_MAX,        // p_sep_by_space
    CHAR_MAX,        // n_cs_precedes
    CHAR_MAX,        // n_sep_by_space
    CHAR_MAX,        // p_sign_posn
    CHAR_MAX,        // n_sign_posn
    g_lconv_wdot,    // _W_decimal_point
    g_lconv_wempty,  // _W_thousands_sep
    g_lconv_wempty,  // _W_int_curr_symbol
    g_lconv_wempty,  // _W_currency_symbol
    g_lconv_wempty,  // _W_mon_decimal_point
    g_lconv_wempty,  // _W_mon_thousands_sep
    g_lconv_wempty,  // _W_positive_sign
    g_lconv_wempty,  // _W_negative_sign
};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

void invoke_exit_hook(int32_t code, int32_t kind) {
    Pg6CrtExitHook hook = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_runtime_mutex);
        hook = g_exit_hook;
    }
    if (hook != nullptr) {
        hook(code, kind);
    }
}

void run_atexit_callbacks() {
    std::vector<Pg6CrtPvfv> callbacks;
    {
        std::lock_guard<std::mutex> lock(g_runtime_mutex);
        callbacks.swap(g_atexit_callbacks);
    }
    for (auto it = callbacks.rbegin(); it != callbacks.rend(); ++it) {
        if (*it != nullptr) {
            (*it)();
        }
    }
}

inline uint8_t ascii_to_lower_u8(uint8_t c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<uint8_t>(c - 'A' + 'a') : c;
}

inline uint16_t ascii_to_lower_u16(uint16_t c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<uint16_t>(c - 'A' + 'a') : c;
}

std::string utf16_to_utf8_u16(const uint16_t* src, size_t max_units = static_cast<size_t>(-1)) {
    std::string out;
    if (src == nullptr) {
        return out;
    }
    size_t i = 0;
    while (i < max_units && src[i] != 0) {
        uint32_t cp = src[i++];
        if (cp >= 0xD800u && cp <= 0xDBFFu && i < max_units && src[i] >= 0xDC00u &&
            src[i] <= 0xDFFFu) {
            const uint32_t low = src[i++];
            cp = 0x10000u + (((cp - 0xD800u) << 10) | (low - 0xDC00u));
        }
        if (cp < 0x80u) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800u) {
            out.push_back(static_cast<char>(0xC0u | (cp >> 6)));
            out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
        } else if (cp < 0x10000u) {
            out.push_back(static_cast<char>(0xE0u | (cp >> 12)));
            out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
            out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
        } else {
            out.push_back(static_cast<char>(0xF0u | (cp >> 18)));
            out.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
            out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
            out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
        }
    }
    return out;
}

void append_cp_utf16(std::vector<uint16_t>& out, uint32_t cp) {
    if (cp >= 0x10000u) {
        cp -= 0x10000u;
        out.push_back(static_cast<uint16_t>(0xD800u + (cp >> 10)));
        out.push_back(static_cast<uint16_t>(0xDC00u + (cp & 0x3FFu)));
    } else {
        out.push_back(static_cast<uint16_t>(cp));
    }
}

std::vector<uint16_t> utf8_to_utf16_vec(std::string_view s) {
    std::vector<uint16_t> out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        const auto b0 = static_cast<uint8_t>(s[i]);
        uint32_t cp = 0xFFFDu;
        size_t len = 1;
        if (b0 < 0x80u) {
            cp = b0;
        } else if ((b0 & 0xE0u) == 0xC0u) {
            len = 2;
            cp = b0 & 0x1Fu;
        } else if ((b0 & 0xF0u) == 0xE0u) {
            len = 3;
            cp = b0 & 0x0Fu;
        } else if ((b0 & 0xF8u) == 0xF0u) {
            len = 4;
            cp = b0 & 0x07u;
        } else {
            append_cp_utf16(out, 0xFFFDu);
            ++i;
            continue;
        }
        bool valid = (i + len <= s.size());
        for (size_t k = 1; valid && k < len; ++k) {
            const auto b = static_cast<uint8_t>(s[i + k]);
            if ((b & 0xC0u) != 0x80u) {
                valid = false;
            } else {
                cp = (cp << 6) | (b & 0x3Fu);
            }
        }
        if (!valid || cp > 0x10FFFFu || (cp >= 0xD800u && cp <= 0xDFFFu)) {
            append_cp_utf16(out, 0xFFFDu);
            ++i;
            continue;
        }
        append_cp_utf16(out, cp);
        i += len;
    }
    return out;
}

// Formats a Win64 UTF-16 printf format string into a UTF-16 vector.
// Works identically on Windows, Linux, and Emscripten regardless of host `wchar_t` width.
std::vector<uint16_t> format_utf16_va(const uint16_t* fmt, va_list ap_in) {
    std::vector<uint16_t> out;
    if (fmt == nullptr) {
        return out;
    }

    va_list ap;
    va_copy(ap, ap_in);

    auto append_ascii = [&out](std::string_view sv) {
        for (char c : sv) {
            out.push_back(static_cast<uint16_t>(static_cast<uint8_t>(c)));
        }
    };

    const uint16_t* p = fmt;
    while (*p != 0) {
        if (*p != '%') {
            out.push_back(*p++);
            continue;
        }
        ++p;
        if (*p == '%') {
            out.push_back('%');
            ++p;
            continue;
        }

        // Capture specifier string in narrow ASCII for delegating numeric formats to snprintf.
        std::string spec = "%";
        bool left_align = false;
        while (*p == '-' || *p == '+' || *p == ' ' || *p == '#' || *p == '0') {
            if (*p == '-') {
                left_align = true;
            }
            spec.push_back(static_cast<char>(*p++));
        }

        // Field width.
        int width = -1;
        if (*p == '*') {
            width = va_arg(ap, int);
            if (width < 0) {
                left_align = true;
                width = -width;
                if (spec.find('-') == std::string::npos) {
                    spec.push_back('-');
                }
            }
            spec += std::to_string(width);
            ++p;
        } else if (*p >= '0' && *p <= '9') {
            width = 0;
            while (*p >= '0' && *p <= '9') {
                width = width * 10 + static_cast<int>(*p - '0');
                spec.push_back(static_cast<char>(*p++));
            }
        }

        // Precision.
        int precision = -1;
        if (*p == '.') {
            spec.push_back('.');
            ++p;
            if (*p == '*') {
                precision = va_arg(ap, int);
                if (precision < 0) {
                    spec.pop_back();
                } else {
                    spec += std::to_string(precision);
                }
                ++p;
            } else {
                precision = 0;
                while (*p >= '0' && *p <= '9') {
                    precision = precision * 10 + static_cast<int>(*p - '0');
                    spec.push_back(static_cast<char>(*p++));
                }
            }
        }

        // Length modifier.
        enum class LenMod { None, Short, Long, LongLong, SizeT };
        LenMod len_mod = LenMod::None;
        if (*p == 'h') {
            ++p;
            len_mod = LenMod::Short;
            if (*p == 'h') {
                ++p;
            }
        } else if (*p == 'l') {
            ++p;
            if (*p == 'l') {
                ++p;
                len_mod = LenMod::LongLong;
            } else {
                len_mod = LenMod::Long;
            }
        } else if (*p == 'z' || *p == 't') {
            ++p;
            len_mod = LenMod::SizeT;
        } else if (*p == 'j') {
            ++p;
            len_mod = LenMod::LongLong;
        } else if (*p == 'I') {
            ++p;
            if (p[0] == '6' && p[1] == '4') {
                p += 2;
                len_mod = LenMod::LongLong;
            } else if (p[0] == '3' && p[1] == '2') {
                p += 2;
                len_mod = LenMod::None;
            } else {
                len_mod = LenMod::SizeT;
            }
        }

        const uint16_t conv = *p;
        if (conv == 0) {
            break;
        }
        ++p;

        char num_buf[256];
        auto format_padded_u16 = [&](const std::vector<uint16_t>& text) {
            const size_t n = (precision >= 0 && static_cast<size_t>(precision) < text.size())
                                 ? static_cast<size_t>(precision)
                                 : text.size();
            const size_t pad = (width > 0 && static_cast<size_t>(width) > n)
                                   ? static_cast<size_t>(width) - n
                                   : 0;
            if (!left_align) {
                for (size_t i = 0; i < pad; ++i) {
                    out.push_back(static_cast<uint16_t>(' '));
                }
            }
            out.insert(out.end(), text.begin(), text.begin() + static_cast<ptrdiff_t>(n));
            if (left_align) {
                for (size_t i = 0; i < pad; ++i) {
                    out.push_back(static_cast<uint16_t>(' '));
                }
            }
        };

        switch (conv) {
            case 's': {
                if (len_mod == LenMod::Short) {
                    const char* s = va_arg(ap, const char*);
                    if (s == nullptr) {
                        s = "(null)";
                    }
                    format_padded_u16(utf8_to_utf16_vec(s));
                } else {
                    const uint16_t* ws = va_arg(ap, const uint16_t*);
                    std::vector<uint16_t> tmp;
                    if (ws == nullptr) {
                        tmp = utf8_to_utf16_vec("(null)");
                    } else {
                        while (*ws != 0) {
                            tmp.push_back(*ws++);
                        }
                    }
                    format_padded_u16(tmp);
                }
                break;
            }
            case 'S': {
                // MSVC wide-printf `%S` prints a narrow `const char*`.
                const char* s = va_arg(ap, const char*);
                if (s == nullptr) {
                    s = "(null)";
                }
                format_padded_u16(utf8_to_utf16_vec(s));
                break;
            }
            case 'c':
            case 'C': {
                const int val = va_arg(ap, int);
                std::vector<uint16_t> tmp{static_cast<uint16_t>(val & 0xFFFF)};
                precision = -1;
                format_padded_u16(tmp);
                break;
            }
            case 'd':
            case 'i': {
                int written = 0;
                if (len_mod == LenMod::LongLong) {
                    spec += "lld";
                    written = std::snprintf(num_buf, sizeof(num_buf), spec.c_str(),
                                            va_arg(ap, long long));
                } else if (len_mod == LenMod::SizeT) {
                    spec += "td";
                    written = std::snprintf(num_buf, sizeof(num_buf), spec.c_str(),
                                            va_arg(ap, ptrdiff_t));
                } else {
                    // Win64 `long` is 32-bit; both `int` and `long` pass as 32-bit signed.
                    spec += "d";
                    written = std::snprintf(num_buf, sizeof(num_buf), spec.c_str(),
                                            va_arg(ap, int));
                }
                if (written > 0) {
                    append_ascii(
                        std::string_view(num_buf, std::min(static_cast<size_t>(written),
                                                           sizeof(num_buf) - 1)));
                }
                break;
            }
            case 'u':
            case 'x':
            case 'X':
            case 'o': {
                int written = 0;
                if (len_mod == LenMod::LongLong) {
                    spec += "ll";
                    spec.push_back(static_cast<char>(conv));
                    written = std::snprintf(num_buf, sizeof(num_buf), spec.c_str(),
                                            va_arg(ap, unsigned long long));
                } else if (len_mod == LenMod::SizeT) {
                    spec += "z";
                    spec.push_back(static_cast<char>(conv));
                    written = std::snprintf(num_buf, sizeof(num_buf), spec.c_str(),
                                            va_arg(ap, size_t));
                } else {
                    spec.push_back(static_cast<char>(conv));
                    written = std::snprintf(num_buf, sizeof(num_buf), spec.c_str(),
                                            va_arg(ap, unsigned int));
                }
                if (written > 0) {
                    append_ascii(
                        std::string_view(num_buf, std::min(static_cast<size_t>(written),
                                                           sizeof(num_buf) - 1)));
                }
                break;
            }
            case 'f':
            case 'F':
            case 'e':
            case 'E':
            case 'g':
            case 'G':
            case 'a':
            case 'A': {
                spec.push_back(static_cast<char>(conv));
                const int written =
                    std::snprintf(num_buf, sizeof(num_buf), spec.c_str(), va_arg(ap, double));
                if (written > 0) {
                    append_ascii(
                        std::string_view(num_buf, std::min(static_cast<size_t>(written),
                                                           sizeof(num_buf) - 1)));
                }
                break;
            }
            case 'p': {
                spec.push_back('p');
                const int written =
                    std::snprintf(num_buf, sizeof(num_buf), spec.c_str(), va_arg(ap, void*));
                if (written > 0) {
                    append_ascii(
                        std::string_view(num_buf, std::min(static_cast<size_t>(written),
                                                           sizeof(num_buf) - 1)));
                }
                break;
            }
            default:
                out.push_back(conv);
                break;
        }
    }

    va_end(ap);
    return out;
}

// Howard Hinnant's civil-from-days / days-from-civil algorithms (public domain).
// Exact 64-bit UTC calendar math, identical across Windows, Linux, and Emscripten Wasm64.
void civil_from_days(int64_t z, int64_t& year, int32_t& month, int32_t& day) {
    z += 719468;
    const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const uint64_t doe = static_cast<uint64_t>(z - era * 146097);
    const uint64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const int64_t y = static_cast<int64_t>(yoe) + era * 400;
    const uint64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const uint64_t mp = (5 * doy + 2) / 153;
    const uint64_t d = doy - (153 * mp + 2) / 5 + 1;
    const uint64_t m = mp < 10 ? mp + 3 : mp - 9;
    year = y + (m <= 2 ? 1 : 0);
    month = static_cast<int32_t>(m);
    day = static_cast<int32_t>(d);
}

int64_t days_from_civil(int64_t y, int32_t m, int32_t d) {
    y -= (m <= 2) ? 1 : 0;
    const int64_t era = (y >= 0 ? y : y - 399) / 400;
    const uint64_t yoe = static_cast<uint64_t>(y - era * 400);
    const uint64_t doy =
        (153u * static_cast<uint64_t>(m > 2 ? m - 3 : m + 9) + 2u) / 5u + static_cast<uint64_t>(d) -
        1u;
    const uint64_t doe = yoe * 365u + yoe / 4u - yoe / 100u + doy;
    return era * 146097 + static_cast<int64_t>(doe) - 719468;
}

bool is_leap_year(int64_t y) {
    return (y % 4 == 0) && ((y % 100 != 0) || (y % 400 == 0));
}

int32_t day_of_year(int64_t y, int32_t month, int32_t day) {
    static constexpr int32_t kDaysBeforeMonth[12] = {
        0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};
    int32_t yday = kDaysBeforeMonth[month - 1] + (day - 1);
    if (month > 2 && is_leap_year(y)) {
        ++yday;
    }
    return yday;
}

std::tm to_std_tm(const Pg6CrtTm& src) {
    std::tm out{};
    out.tm_sec = src.tm_sec;
    out.tm_min = src.tm_min;
    out.tm_hour = src.tm_hour;
    out.tm_mday = src.tm_mday;
    out.tm_mon = src.tm_mon;
    out.tm_year = src.tm_year;
    out.tm_wday = src.tm_wday;
    out.tm_yday = src.tm_yday;
    out.tm_isdst = src.tm_isdst;
    return out;
}

void from_std_tm(const std::tm& src, Pg6CrtTm& dst) {
    dst.tm_sec = static_cast<int32_t>(src.tm_sec);
    dst.tm_min = static_cast<int32_t>(src.tm_min);
    dst.tm_hour = static_cast<int32_t>(src.tm_hour);
    dst.tm_mday = static_cast<int32_t>(src.tm_mday);
    dst.tm_mon = static_cast<int32_t>(src.tm_mon);
    dst.tm_year = static_cast<int32_t>(src.tm_year);
    dst.tm_wday = static_cast<int32_t>(src.tm_wday);
    dst.tm_yday = static_cast<int32_t>(src.tm_yday);
    dst.tm_isdst = static_cast<int32_t>(src.tm_isdst);
}

// ---------------------------------------------------------------------------
// Module names matching docs/imports_required.md verbatim
// ---------------------------------------------------------------------------

#define PG6_UCRT_MOD_CONVERT "api-ms-win-crt-convert-l1-1-0.dll"
#define PG6_UCRT_MOD_FS "api-ms-win-crt-filesystem-l1-1-0.dll"
#define PG6_UCRT_MOD_HEAP "api-ms-win-crt-heap-l1-1-0.dll"
#define PG6_UCRT_MOD_LOCALE "api-ms-win-crt-locale-l1-1-0.dll"
#define PG6_UCRT_MOD_MATH "api-ms-win-crt-math-l1-1-0.dll"
#define PG6_UCRT_MOD_RUNTIME "api-ms-win-crt-runtime-l1-1-0.dll"
#define PG6_UCRT_MOD_STDIO "api-ms-win-crt-stdio-l1-1-0.dll"
#define PG6_UCRT_MOD_STRING "api-ms-win-crt-string-l1-1-0.dll"
#define PG6_UCRT_MOD_TIME "api-ms-win-crt-time-l1-1-0.dll"
#define PG6_UCRT_MOD_UTILITY "api-ms-win-crt-utility-l1-1-0.dll"
#define PG6_UCRT_MOD_VCR140 "VCRUNTIME140.dll"
#define PG6_UCRT_MOD_VCR140_1 "VCRUNTIME140_1.dll"

#define PG6_UCRT_EXPORTS(E)                                                                    \
    /* api-ms-win-crt-convert-l1-1-0.dll (12) */                                               \
    E(CONVERT, "_itoa", pg6_crt_itoa)                                                          \
    E(CONVERT, "_strtoi64", pg6_crt_strtoi64)                                                  \
    E(CONVERT, "_strtoui64", pg6_crt_strtoui64)                                                \
    E(CONVERT, "atof", pg6_crt_atof)                                                           \
    E(CONVERT, "atoi", pg6_crt_atoi)                                                           \
    E(CONVERT, "atoll", pg6_crt_atoll)                                                         \
    E(CONVERT, "strtod", pg6_crt_strtod)                                                       \
    E(CONVERT, "strtof", pg6_crt_strtof)                                                       \
    E(CONVERT, "strtol", pg6_crt_strtol)                                                       \
    E(CONVERT, "strtoll", pg6_crt_strtoll)                                                     \
    E(CONVERT, "strtoul", pg6_crt_strtoul)                                                     \
    E(CONVERT, "strtoull", pg6_crt_strtoull)                                                   \
    /* api-ms-win-crt-filesystem-l1-1-0.dll (4) */                                             \
    E(FS, "_lock_file", pg6_crt_lock_file)                                                     \
    E(FS, "_unlock_file", pg6_crt_unlock_file)                                                 \
    E(FS, "_wstat64", pg6_crt_wstat64)                                                         \
    E(FS, "remove", pg6_crt_remove)                                                            \
    /* api-ms-win-crt-heap-l1-1-0.dll (5) */                                                   \
    E(HEAP, "_set_new_mode", pg6_crt_set_new_mode)                                             \
    E(HEAP, "calloc", pg6_crt_calloc)                                                          \
    E(HEAP, "free", pg6_crt_free)                                                              \
    E(HEAP, "malloc", pg6_crt_malloc)                                                          \
    E(HEAP, "realloc", pg6_crt_realloc)                                                        \
    /* api-ms-win-crt-locale-l1-1-0.dll (2) */                                                 \
    E(LOCALE, "_configthreadlocale", pg6_crt_configthreadlocale)                               \
    E(LOCALE, "localeconv", pg6_crt_localeconv)                                                \
    /* api-ms-win-crt-math-l1-1-0.dll (41) */                                                  \
    E(MATH, "__setusermatherr", pg6_crt_setusermatherr)                                        \
    E(MATH, "_dtest", pg6_crt_dtest)                                                           \
    E(MATH, "_fdsign", pg6_crt_fdsign)                                                         \
    E(MATH, "_fdtest", pg6_crt_fdtest)                                                         \
    E(MATH, "acos", pg6_crt_acos)                                                              \
    E(MATH, "acosf", pg6_crt_acosf)                                                            \
    E(MATH, "asin", pg6_crt_asin)                                                              \
    E(MATH, "asinf", pg6_crt_asinf)                                                            \
    E(MATH, "atan", pg6_crt_atan)                                                              \
    E(MATH, "atan2", pg6_crt_atan2)                                                            \
    E(MATH, "atan2f", pg6_crt_atan2f)                                                          \
    E(MATH, "atanf", pg6_crt_atanf)                                                            \
    E(MATH, "cbrtf", pg6_crt_cbrtf)                                                            \
    E(MATH, "ceilf", pg6_crt_ceilf)                                                            \
    E(MATH, "cos", pg6_crt_cos)                                                                \
    E(MATH, "cosf", pg6_crt_cosf)                                                              \
    E(MATH, "erff", pg6_crt_erff)                                                              \
    E(MATH, "exp", pg6_crt_exp)                                                                \
    E(MATH, "exp2", pg6_crt_exp2)                                                              \
    E(MATH, "exp2f", pg6_crt_exp2f)                                                            \
    E(MATH, "expf", pg6_crt_expf)                                                              \
    E(MATH, "fmod", pg6_crt_fmod)                                                              \
    E(MATH, "fmodf", pg6_crt_fmodf)                                                            \
    E(MATH, "log", pg6_crt_log)                                                                \
    E(MATH, "log10f", pg6_crt_log10f)                                                          \
    E(MATH, "log2", pg6_crt_log2)                                                              \
    E(MATH, "log2f", pg6_crt_log2f)                                                            \
    E(MATH, "logf", pg6_crt_logf)                                                              \
    E(MATH, "lroundf", pg6_crt_lroundf)                                                        \
    E(MATH, "modf", pg6_crt_modf)                                                              \
    E(MATH, "modff", pg6_crt_modff)                                                            \
    E(MATH, "nextafterf", pg6_crt_nextafterf)                                                  \
    E(MATH, "nexttowardf", pg6_crt_nexttowardf)                                                \
    E(MATH, "pow", pg6_crt_pow)                                                                \
    E(MATH, "powf", pg6_crt_powf)                                                              \
    E(MATH, "remainderf", pg6_crt_remainderf)                                                  \
    E(MATH, "sin", pg6_crt_sin)                                                                \
    E(MATH, "sinf", pg6_crt_sinf)                                                              \
    E(MATH, "sinhf", pg6_crt_sinhf)                                                            \
    E(MATH, "tan", pg6_crt_tan)                                                                \
    E(MATH, "tanf", pg6_crt_tanf)                                                              \
    /* api-ms-win-crt-runtime-l1-1-0.dll (25) */                                               \
    E(RUNTIME, "__p___argc", pg6_crt_p___argc)                                                 \
    E(RUNTIME, "__p___argv", pg6_crt_p___argv)                                                 \
    E(RUNTIME, "_c_exit", pg6_crt_c_exit)                                                      \
    E(RUNTIME, "_cexit", pg6_crt_cexit)                                                        \
    E(RUNTIME, "_configure_narrow_argv", pg6_crt_configure_narrow_argv)                        \
    E(RUNTIME, "_crt_atexit", pg6_crt_crt_atexit)                                              \
    E(RUNTIME, "_errno", pg6_crt_errno)                                                        \
    E(RUNTIME, "_exit", pg6_crt__exit)                                                         \
    E(RUNTIME, "_get_errno", pg6_crt_get_errno)                                                \
    E(RUNTIME, "_get_narrow_winmain_command_line", pg6_crt_get_narrow_winmain_command_line)    \
    E(RUNTIME, "_initialize_narrow_environment", pg6_crt_initialize_narrow_environment)        \
    E(RUNTIME, "_initialize_onexit_table", pg6_crt_initialize_onexit_table)                    \
    E(RUNTIME, "_initterm", pg6_crt_initterm)                                                  \
    E(RUNTIME, "_initterm_e", pg6_crt_initterm_e)                                              \
    E(RUNTIME, "_invalid_parameter_noinfo", pg6_crt_invalid_parameter_noinfo)                  \
    E(RUNTIME, "_invalid_parameter_noinfo_noreturn", pg6_crt_invalid_parameter_noinfo_noreturn)\
    E(RUNTIME, "_register_onexit_function", pg6_crt_register_onexit_function)                  \
    E(RUNTIME, "_register_thread_local_exe_atexit_callback",                                   \
      pg6_crt_register_thread_local_exe_atexit_callback)                                       \
    E(RUNTIME, "_seh_filter_exe", pg6_crt_seh_filter_exe)                                      \
    E(RUNTIME, "_set_app_type", pg6_crt_set_app_type)                                          \
    E(RUNTIME, "_set_errno", pg6_crt_set_errno)                                                \
    E(RUNTIME, "_set_invalid_parameter_handler", pg6_crt_set_invalid_parameter_handler)        \
    E(RUNTIME, "abort", pg6_crt_abort)                                                         \
    E(RUNTIME, "exit", pg6_crt_exit)                                                           \
    E(RUNTIME, "terminate", pg6_crt_terminate)                                                 \
    /* api-ms-win-crt-stdio-l1-1-0.dll (29) */                                                 \
    E(STDIO, "__acrt_iob_func", pg6_crt_acrt_iob_func)                                         \
    E(STDIO, "__p__commode", pg6_crt_p__commode)                                               \
    E(STDIO, "__stdio_common_vfprintf", pg6_crt_stdio_common_vfprintf)                         \
    E(STDIO, "__stdio_common_vfwprintf", pg6_crt_stdio_common_vfwprintf)                       \
    E(STDIO, "__stdio_common_vsnwprintf_s", pg6_crt_stdio_common_vsnwprintf_s)                 \
    E(STDIO, "__stdio_common_vsprintf", pg6_crt_stdio_common_vsprintf)                         \
    E(STDIO, "__stdio_common_vsprintf_s", pg6_crt_stdio_common_vsprintf_s)                     \
    E(STDIO, "__stdio_common_vsscanf", pg6_crt_stdio_common_vsscanf)                           \
    E(STDIO, "__stdio_common_vswprintf", pg6_crt_stdio_common_vswprintf)                       \
    E(STDIO, "__stdio_common_vswprintf_s", pg6_crt_stdio_common_vswprintf_s)                   \
    E(STDIO, "__stdio_common_vswscanf", pg6_crt_stdio_common_vswscanf)                         \
    E(STDIO, "_fseeki64", pg6_crt_fseeki64)                                                    \
    E(STDIO, "_get_stream_buffer_pointers", pg6_crt_get_stream_buffer_pointers)                \
    E(STDIO, "_set_fmode", pg6_crt_set_fmode)                                                  \
    E(STDIO, "_wfopen", pg6_crt_wfopen)                                                        \
    E(STDIO, "fclose", pg6_crt_fclose)                                                         \
    E(STDIO, "fflush", pg6_crt_fflush)                                                         \
    E(STDIO, "fgetc", pg6_crt_fgetc)                                                           \
    E(STDIO, "fgetpos", pg6_crt_fgetpos)                                                       \
    E(STDIO, "fopen", pg6_crt_fopen)                                                           \
    E(STDIO, "fputc", pg6_crt_fputc)                                                           \
    E(STDIO, "fread", pg6_crt_fread)                                                           \
    E(STDIO, "fseek", pg6_crt_fseek)                                                           \
    E(STDIO, "fsetpos", pg6_crt_fsetpos)                                                       \
    E(STDIO, "ftell", pg6_crt_ftell)                                                           \
    E(STDIO, "fwrite", pg6_crt_fwrite)                                                         \
    E(STDIO, "rewind", pg6_crt_rewind)                                                         \
    E(STDIO, "setvbuf", pg6_crt_setvbuf)                                                       \
    E(STDIO, "ungetc", pg6_crt_ungetc)                                                         \
    /* api-ms-win-crt-string-l1-1-0.dll (31) */                                                \
    E(STRING, "_stricmp", pg6_crt_stricmp)                                                     \
    E(STRING, "_strnicmp", pg6_crt_strnicmp)                                                   \
    E(STRING, "_wcsicoll", pg6_crt_wcsicoll)                                                   \
    E(STRING, "_wcsnicmp", pg6_crt_wcsnicmp)                                                   \
    E(STRING, "isalnum", pg6_crt_isalnum)                                                      \
    E(STRING, "isalpha", pg6_crt_isalpha)                                                      \
    E(STRING, "isdigit", pg6_crt_isdigit)                                                      \
    E(STRING, "islower", pg6_crt_islower)                                                      \
    E(STRING, "isspace", pg6_crt_isspace)                                                      \
    E(STRING, "isxdigit", pg6_crt_isxdigit)                                                    \
    E(STRING, "strcat", pg6_crt_strcat)                                                        \
    E(STRING, "strcat_s", pg6_crt_strcat_s)                                                    \
    E(STRING, "strcmp", pg6_crt_strcmp)                                                        \
    E(STRING, "strcpy", pg6_crt_strcpy)                                                        \
    E(STRING, "strcpy_s", pg6_crt_strcpy_s)                                                    \
    E(STRING, "strcspn", pg6_crt_strcspn)                                                      \
    E(STRING, "strlen", pg6_crt_strlen)                                                        \
    E(STRING, "strncat", pg6_crt_strncat)                                                      \
    E(STRING, "strncmp", pg6_crt_strncmp)                                                      \
    E(STRING, "strncpy", pg6_crt_strncpy)                                                      \
    E(STRING, "strncpy_s", pg6_crt_strncpy_s)                                                  \
    E(STRING, "strpbrk", pg6_crt_strpbrk)                                                      \
    E(STRING, "strtok", pg6_crt_strtok)                                                        \
    E(STRING, "strtok_s", pg6_crt_strtok_s)                                                    \
    E(STRING, "tolower", pg6_crt_tolower)                                                      \
    E(STRING, "toupper", pg6_crt_toupper)                                                      \
    E(STRING, "wcscat_s", pg6_crt_wcscat_s)                                                    \
    E(STRING, "wcscmp", pg6_crt_wcscmp)                                                        \
    E(STRING, "wcscoll", pg6_crt_wcscoll)                                                      \
    E(STRING, "wcslen", pg6_crt_wcslen)                                                        \
    E(STRING, "wcsncpy", pg6_crt_wcsncpy)                                                      \
    /* api-ms-win-crt-time-l1-1-0.dll (9) */                                                   \
    E(TIME, "_difftime64", pg6_crt_difftime64)                                                 \
    E(TIME, "_ftime64", pg6_crt_ftime64)                                                       \
    E(TIME, "_gmtime64", pg6_crt_gmtime64)                                                     \
    E(TIME, "_gmtime64_s", pg6_crt_gmtime64_s)                                                 \
    E(TIME, "_localtime64_s", pg6_crt_localtime64_s)                                           \
    E(TIME, "_mkgmtime64", pg6_crt_mkgmtime64)                                                 \
    E(TIME, "_mktime64", pg6_crt_mktime64)                                                     \
    E(TIME, "_time64", pg6_crt_time64)                                                         \
    E(TIME, "strftime", pg6_crt_strftime)                                                      \
    /* api-ms-win-crt-utility-l1-1-0.dll (4) */                                                \
    E(UTILITY, "bsearch", pg6_crt_bsearch)                                                     \
    E(UTILITY, "qsort", pg6_crt_qsort)                                                         \
    E(UTILITY, "rand", pg6_crt_rand)                                                           \
    E(UTILITY, "srand", pg6_crt_srand)                                                         \
    /* VCRUNTIME140.dll (23) */                                                                \
    E(VCR140, "_CxxThrowException", pg6_crt_cxx_throw_exception)                               \
    E(VCR140, "__C_specific_handler", pg6_crt_c_specific_handler)                              \
    E(VCR140, "__CxxFrameHandler3", pg6_crt_cxx_frame_handler3)                                \
    E(VCR140, "__RTDynamicCast", pg6_crt_rt_dynamic_cast)                                      \
    E(VCR140, "__current_exception", pg6_crt_current_exception)                                \
    E(VCR140, "__current_exception_context", pg6_crt_current_exception_context)                \
    E(VCR140, "__intrinsic_setjmp", pg6_crt_intrinsic_setjmp)                                  \
    E(VCR140, "__std_exception_copy", pg6_crt_std_exception_copy)                              \
    E(VCR140, "__std_exception_destroy", pg6_crt_std_exception_destroy)                        \
    E(VCR140, "__std_terminate", pg6_crt_std_terminate)                                        \
    E(VCR140, "__std_type_info_compare", pg6_crt_std_type_info_compare)                        \
    E(VCR140, "_purecall", pg6_crt_purecall)                                                   \
    E(VCR140, "_set_purecall_handler", pg6_crt_set_purecall_handler)                           \
    E(VCR140, "longjmp", pg6_crt_longjmp)                                                      \
    E(VCR140, "memchr", pg6_crt_memchr)                                                        \
    E(VCR140, "memcmp", pg6_crt_memcmp)                                                        \
    E(VCR140, "memcpy", pg6_crt_memcpy)                                                        \
    E(VCR140, "memmove", pg6_crt_memmove)                                                      \
    E(VCR140, "memset", pg6_crt_memset)                                                        \
    E(VCR140, "strchr", pg6_crt_strchr)                                                        \
    E(VCR140, "strrchr", pg6_crt_strrchr)                                                      \
    E(VCR140, "strstr", pg6_crt_strstr)                                                        \
    E(VCR140, "wcsrchr", pg6_crt_wcsrchr)                                                      \
    /* VCRUNTIME140_1.dll (1) */                                                               \
    E(VCR140_1, "__CxxFrameHandler4", pg6_crt_cxx_frame_handler4)

}  // namespace

// ---------------------------------------------------------------------------
// 1. api-ms-win-crt-convert-l1-1-0.dll (12 symbols)
// ---------------------------------------------------------------------------

extern "C" {

char* pg6_crt_itoa(int32_t value, char* buffer, int32_t radix) {
    if (buffer == nullptr || radix < 2 || radix > 36) {
        g_crt_errno = EINVAL;
        return buffer;
    }
    static constexpr char kDigits[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    uint32_t mag = 0;
    bool negative = false;
    if (value < 0 && radix == 10) {
        negative = true;
        mag = static_cast<uint32_t>(0u - static_cast<uint32_t>(value));
    } else {
        mag = static_cast<uint32_t>(value);
    }

    char tmp[36];
    size_t len = 0;
    do {
        tmp[len++] = kDigits[mag % static_cast<uint32_t>(radix)];
        mag /= static_cast<uint32_t>(radix);
    } while (mag != 0);

    size_t pos = 0;
    if (negative) {
        buffer[pos++] = '-';
    }
    while (len > 0) {
        buffer[pos++] = tmp[--len];
    }
    buffer[pos] = '\0';
    return buffer;
}

int64_t pg6_crt_strtoi64(const char* str, char** endptr, int32_t base) {
    return pg6_crt_strtoll(str, endptr, base);
}

uint64_t pg6_crt_strtoui64(const char* str, char** endptr, int32_t base) {
    return pg6_crt_strtoull(str, endptr, base);
}

double pg6_crt_atof(const char* str) {
    return str != nullptr ? std::strtod(str, nullptr) : 0.0;
}

int32_t pg6_crt_atoi(const char* str) {
    return str != nullptr ? static_cast<int32_t>(std::strtoll(str, nullptr, 10)) : 0;
}

int64_t pg6_crt_atoll(const char* str) {
    return str != nullptr ? static_cast<int64_t>(std::strtoll(str, nullptr, 10)) : 0;
}

double pg6_crt_strtod(const char* str, char** endptr) {
    if (str == nullptr) {
        g_crt_errno = EINVAL;
        if (endptr != nullptr) {
            *endptr = nullptr;
        }
        return 0.0;
    }
    errno = 0;
    const double val = std::strtod(str, endptr);
    if (errno != 0) {
        g_crt_errno = errno;
    }
    return val;
}

float pg6_crt_strtof(const char* str, char** endptr) {
    if (str == nullptr) {
        g_crt_errno = EINVAL;
        if (endptr != nullptr) {
            *endptr = nullptr;
        }
        return 0.0f;
    }
    errno = 0;
    const float val = std::strtof(str, endptr);
    if (errno != 0) {
        g_crt_errno = errno;
    }
    return val;
}

int32_t pg6_crt_strtol(const char* str, char** endptr, int32_t base) {
    if (str == nullptr) {
        g_crt_errno = EINVAL;
        if (endptr != nullptr) {
            *endptr = nullptr;
        }
        return 0;
    }
    errno = 0;
    const long long val = std::strtoll(str, endptr, base);
    if (errno == ERANGE || val > INT32_MAX) {
        g_crt_errno = ERANGE;
        return val < 0 ? INT32_MIN : INT32_MAX;
    }
    if (val < INT32_MIN) {
        g_crt_errno = ERANGE;
        return INT32_MIN;
    }
    if (errno != 0) {
        g_crt_errno = errno;
    }
    return static_cast<int32_t>(val);
}

int64_t pg6_crt_strtoll(const char* str, char** endptr, int32_t base) {
    if (str == nullptr) {
        g_crt_errno = EINVAL;
        if (endptr != nullptr) {
            *endptr = nullptr;
        }
        return 0;
    }
    errno = 0;
    const long long val = std::strtoll(str, endptr, base);
    if (errno != 0) {
        g_crt_errno = errno;
    }
    return static_cast<int64_t>(val);
}

uint32_t pg6_crt_strtoul(const char* str, char** endptr, int32_t base) {
    if (str == nullptr) {
        g_crt_errno = EINVAL;
        if (endptr != nullptr) {
            *endptr = nullptr;
        }
        return 0;
    }
    const char* s = str;
    while (std::isspace(static_cast<unsigned char>(*s))) {
        ++s;
    }
    const bool neg = (*s == '-');
    errno = 0;
    const unsigned long long val = std::strtoull(str, endptr, base);
    if (errno == ERANGE) {
        g_crt_errno = ERANGE;
        return UINT32_MAX;
    }
    if (neg) {
        const unsigned long long abs_val = 0ull - val;
        if (abs_val > UINT32_MAX) {
            g_crt_errno = ERANGE;
            return UINT32_MAX;
        }
        return static_cast<uint32_t>(0u - static_cast<uint32_t>(abs_val));
    }
    if (val > UINT32_MAX) {
        g_crt_errno = ERANGE;
        return UINT32_MAX;
    }
    if (errno != 0) {
        g_crt_errno = errno;
    }
    return static_cast<uint32_t>(val);
}

uint64_t pg6_crt_strtoull(const char* str, char** endptr, int32_t base) {
    if (str == nullptr) {
        g_crt_errno = EINVAL;
        if (endptr != nullptr) {
            *endptr = nullptr;
        }
        return 0;
    }
    errno = 0;
    const unsigned long long val = std::strtoull(str, endptr, base);
    if (errno != 0) {
        g_crt_errno = errno;
    }
    return static_cast<uint64_t>(val);
}

// ---------------------------------------------------------------------------
// 2. api-ms-win-crt-filesystem-l1-1-0.dll (4 symbols)
// ---------------------------------------------------------------------------

void pg6_crt_lock_file(FILE* stream) {
    if (stream == nullptr) {
        return;
    }
#if defined(_MSC_VER)
    _lock_file(stream);
#else
    flockfile(stream);
#endif
}

void pg6_crt_unlock_file(FILE* stream) {
    if (stream == nullptr) {
        return;
    }
#if defined(_MSC_VER)
    _unlock_file(stream);
#else
    funlockfile(stream);
#endif
}

int32_t pg6_crt_wstat64(const uint16_t* path, Pg6CrtStat64* buffer) {
    if (path == nullptr || buffer == nullptr) {
        g_crt_errno = EINVAL;
        return -1;
    }
    std::memset(buffer, 0, sizeof(*buffer));
    const std::string utf8 = utf16_to_utf8_u16(path);
    errno = 0;
#if defined(_MSC_VER)
    struct __stat64 st{};
    const int rc = _stat64(utf8.c_str(), &st);
#else
    struct stat st{};
    const int rc = stat(utf8.c_str(), &st);
#endif
    if (rc != 0) {
        g_crt_errno = errno != 0 ? errno : ENOENT;
        return -1;
    }
    buffer->st_dev = static_cast<uint32_t>(st.st_dev);
    buffer->st_ino = static_cast<uint16_t>(st.st_ino);
    buffer->st_mode = static_cast<uint16_t>(st.st_mode);
    buffer->st_nlink = static_cast<int16_t>(st.st_nlink);
    buffer->st_uid = static_cast<int16_t>(st.st_uid);
    buffer->st_gid = static_cast<int16_t>(st.st_gid);
    buffer->st_rdev = static_cast<uint32_t>(st.st_rdev);
    buffer->st_size = static_cast<int64_t>(st.st_size);
    buffer->atime_sec = static_cast<int64_t>(st.st_atime);
    buffer->mtime_sec = static_cast<int64_t>(st.st_mtime);
    buffer->ctime_sec = static_cast<int64_t>(st.st_ctime);
    return 0;
}

int32_t pg6_crt_remove(const char* path) {
    if (path == nullptr) {
        g_crt_errno = EINVAL;
        return -1;
    }
    errno = 0;
    const int rc = std::remove(path);
    if (rc != 0 && errno != 0) {
        g_crt_errno = errno;
    }
    return rc;
}

// ---------------------------------------------------------------------------
// 3. api-ms-win-crt-heap-l1-1-0.dll (5 symbols)
// ---------------------------------------------------------------------------

int32_t pg6_crt_set_new_mode(int32_t new_mode) {
    if (new_mode != 0 && new_mode != 1) {
        g_crt_errno = EINVAL;
        return -1;
    }
    return g_new_mode.exchange(new_mode, std::memory_order_acq_rel);
}

// The malloc family lives in the guest address space (os/guest_heap.h).
void* pg6_crt_calloc(size_t count, size_t size) {
    return playgta6::os::guest_crt::calloc(count, size, &g_crt_errno);
}

void pg6_crt_free(void* ptr) {
    playgta6::os::guest_crt::free(ptr);
}

void* pg6_crt_malloc(size_t size) {
    return playgta6::os::guest_crt::malloc(size, &g_crt_errno);
}

void* pg6_crt_realloc(void* ptr, size_t size) {
    return playgta6::os::guest_crt::realloc(ptr, size, &g_crt_errno);
}

// ---------------------------------------------------------------------------
// 4. api-ms-win-crt-locale-l1-1-0.dll (2 symbols)
// ---------------------------------------------------------------------------

int32_t pg6_crt_configthreadlocale(int32_t flag) {
    const int32_t prev = g_thread_locale_mode;
    if (flag == 0) {
        return prev;
    }
    if (flag == 1 || flag == 2) {
        g_thread_locale_mode = flag;
        return prev;
    }
    g_crt_errno = EINVAL;
    return -1;
}

Pg6CrtLconv* pg6_crt_localeconv(void) {
    return &g_c_lconv;
}

// ---------------------------------------------------------------------------
// 5. api-ms-win-crt-math-l1-1-0.dll (41 symbols)
// ---------------------------------------------------------------------------

void pg6_crt_setusermatherr(Pg6CrtUserMathErrHandler handler) {
    std::lock_guard<std::mutex> lock(g_runtime_mutex);
    g_user_math_err_handler = handler;
}

int16_t pg6_crt_dtest(const double* px) {
    if (px == nullptr) {
        return PG6_CRT_FP_ZERO;
    }
    switch (std::fpclassify(*px)) {
        case FP_NAN:
            return PG6_CRT_FP_NAN;
        case FP_INFINITE:
            return PG6_CRT_FP_INF;
        case FP_ZERO:
            return PG6_CRT_FP_ZERO;
        case FP_SUBNORMAL:
            return PG6_CRT_FP_DENORM;
        default:
            return PG6_CRT_FP_FINITE;
    }
}

int32_t pg6_crt_fdsign(float x) {
    uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    return (bits & 0x80000000u) != 0 ? 0x8000 : 0;
}

int16_t pg6_crt_fdtest(const float* px) {
    if (px == nullptr) {
        return PG6_CRT_FP_ZERO;
    }
    switch (std::fpclassify(*px)) {
        case FP_NAN:
            return PG6_CRT_FP_NAN;
        case FP_INFINITE:
            return PG6_CRT_FP_INF;
        case FP_ZERO:
            return PG6_CRT_FP_ZERO;
        case FP_SUBNORMAL:
            return PG6_CRT_FP_DENORM;
        default:
            return PG6_CRT_FP_FINITE;
    }
}

double pg6_crt_acos(double x) { return std::acos(x); }
float pg6_crt_acosf(float x) { return std::acos(x); }
double pg6_crt_asin(double x) { return std::asin(x); }
float pg6_crt_asinf(float x) { return std::asin(x); }
double pg6_crt_atan(double x) { return std::atan(x); }
double pg6_crt_atan2(double y, double x) { return std::atan2(y, x); }
float pg6_crt_atan2f(float y, float x) { return std::atan2(y, x); }
float pg6_crt_atanf(float x) { return std::atan(x); }
float pg6_crt_cbrtf(float x) { return std::cbrt(x); }
float pg6_crt_ceilf(float x) { return std::ceil(x); }
double pg6_crt_cos(double x) { return std::cos(x); }
float pg6_crt_cosf(float x) { return std::cos(x); }
float pg6_crt_erff(float x) { return std::erf(x); }
double pg6_crt_exp(double x) { return std::exp(x); }
double pg6_crt_exp2(double x) { return std::exp2(x); }
float pg6_crt_exp2f(float x) { return std::exp2(x); }
float pg6_crt_expf(float x) { return std::exp(x); }
double pg6_crt_fmod(double x, double y) { return std::fmod(x, y); }
float pg6_crt_fmodf(float x, float y) { return std::fmod(x, y); }
double pg6_crt_log(double x) { return std::log(x); }
float pg6_crt_log10f(float x) { return std::log10(x); }
double pg6_crt_log2(double x) { return std::log2(x); }
float pg6_crt_log2f(float x) { return std::log2(x); }
float pg6_crt_logf(float x) { return std::log(x); }
int32_t pg6_crt_lroundf(float x) { return static_cast<int32_t>(std::lround(x)); }
double pg6_crt_modf(double x, double* iptr) { return std::modf(x, iptr); }
float pg6_crt_modff(float x, float* iptr) { return std::modf(x, iptr); }
float pg6_crt_nextafterf(float x, float y) { return std::nextafter(x, y); }
float pg6_crt_nexttowardf(float x, double y) {
    return std::nexttoward(x, static_cast<long double>(y));
}
double pg6_crt_pow(double x, double y) { return std::pow(x, y); }
float pg6_crt_powf(float x, float y) { return std::pow(x, y); }
float pg6_crt_remainderf(float x, float y) { return std::remainder(x, y); }
double pg6_crt_sin(double x) { return std::sin(x); }
float pg6_crt_sinf(float x) { return std::sin(x); }
float pg6_crt_sinhf(float x) { return std::sinh(x); }
double pg6_crt_tan(double x) { return std::tan(x); }
float pg6_crt_tanf(float x) { return std::tan(x); }

// ---------------------------------------------------------------------------
// 6. api-ms-win-crt-runtime-l1-1-0.dll (25 symbols)
// ---------------------------------------------------------------------------

int32_t* pg6_crt_p___argc(void) {
    return &g_argc;
}

char*** pg6_crt_p___argv(void) {
    ensure_guest_crt_state();
    return &g_argv;
}

void pg6_crt_c_exit(void) {
    std::lock_guard<std::mutex> lock(g_runtime_mutex);
    g_atexit_callbacks.clear();
}

void pg6_crt_cexit(void) {
    run_atexit_callbacks();
}

int32_t pg6_crt_configure_narrow_argv(int32_t mode) {
    (void)mode;
    return 0;
}

int32_t pg6_crt_crt_atexit(Pg6CrtPvfv fn) {
    if (fn == nullptr) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(g_runtime_mutex);
    g_atexit_callbacks.push_back(fn);
    return 0;
}

int32_t* pg6_crt_errno(void) {
    return &g_crt_errno;
}

void pg6_crt__exit(int32_t status) {
    g_last_exit_code.store(status, std::memory_order_release);
    invoke_exit_hook(status, PG6_CRT_EXIT_QUICK);
}

int32_t pg6_crt_get_errno(int32_t* value) {
    if (value == nullptr) {
        g_crt_errno = EINVAL;
        return EINVAL;
    }
    *value = g_crt_errno;
    return 0;
}

char* pg6_crt_get_narrow_winmain_command_line(void) {
    ensure_guest_crt_state();
    return reinterpret_cast<char*>(playgta6::os::default_guest_heap().to_host(g_cmdline_va));
}

int32_t pg6_crt_initialize_narrow_environment(void) {
    return 0;
}

int32_t pg6_crt_initialize_onexit_table(Pg6CrtOnexitTable* table) {
    if (table == nullptr) {
        return -1;
    }
    // Match MSVC UCRT semantics: an already-initialized table is left untouched.
    if (table->first != nullptr && table->first != reinterpret_cast<Pg6CrtOnexitFn*>(static_cast<uintptr_t>(-1))) {
        return 0;
    }
    constexpr size_t kInitialCap = 32;
    auto* buf = static_cast<Pg6CrtOnexitFn*>(playgta6::os::guest_crt::calloc(kInitialCap, sizeof(Pg6CrtOnexitFn)));
    if (buf == nullptr) {
        return -1;
    }
    table->first = buf;
    table->last = buf;
    table->end = buf + kInitialCap;
    return 0;
}

void pg6_crt_initterm(Pg6CrtPvfv* first, Pg6CrtPvfv* last) {
    if (first == nullptr || last == nullptr) {
        return;
    }
    for (Pg6CrtPvfv* it = first; it < last; ++it) {
        if (*it != nullptr) {
            (*it)();
        }
    }
}

int32_t pg6_crt_initterm_e(Pg6CrtPifv* first, Pg6CrtPifv* last) {
    if (first == nullptr || last == nullptr) {
        return 0;
    }
    for (Pg6CrtPifv* it = first; it < last; ++it) {
        if (*it != nullptr) {
            const int32_t rc = (*it)();
            if (rc != 0) {
                return rc;
            }
        }
    }
    return 0;
}

void pg6_crt_invalid_parameter_noinfo(void) {
    Pg6CrtInvalidParamHandler handler = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_runtime_mutex);
        handler = g_invalid_param_handler;
    }
    if (handler != nullptr) {
        handler(nullptr, nullptr, nullptr, 0, 0);
        return;
    }
    invoke_exit_hook(EINVAL, PG6_CRT_EXIT_INVALID_PARAM);
}

void pg6_crt_invalid_parameter_noinfo_noreturn(void) {
    pg6_crt_invalid_parameter_noinfo();
}

int32_t pg6_crt_register_onexit_function(Pg6CrtOnexitTable* table, Pg6CrtOnexitFn fn) {
    if (table == nullptr) {
        return -1;
    }
    if (fn == nullptr) {
        return 0;
    }
    if (table->first == nullptr ||
        table->first == reinterpret_cast<Pg6CrtOnexitFn*>(static_cast<uintptr_t>(-1))) {
        if (pg6_crt_initialize_onexit_table(table) != 0) {
            return -1;
        }
    }
    if (table->last >= table->end) {
        const size_t count = static_cast<size_t>(table->last - table->first);
        const size_t old_cap = static_cast<size_t>(table->end - table->first);
        const size_t new_cap = old_cap == 0 ? 32 : old_cap * 2;
        auto* grown = static_cast<Pg6CrtOnexitFn*>(
            playgta6::os::guest_crt::realloc(table->first, new_cap * sizeof(Pg6CrtOnexitFn)));
        if (grown == nullptr) {
            return -1;
        }
        table->first = grown;
        table->last = grown + count;
        table->end = grown + new_cap;
    }
    *(table->last)++ = fn;
    return 0;
}

int32_t pg6_crt_execute_onexit_table(Pg6CrtOnexitTable* table) {
    if (table == nullptr) {
        return -1;
    }
    if (table->first == nullptr ||
        table->first == reinterpret_cast<Pg6CrtOnexitFn*>(static_cast<uintptr_t>(-1))) {
        return 0;
    }
    Pg6CrtOnexitFn* first = table->first;
    Pg6CrtOnexitFn* last = table->last;
    table->first = nullptr;
    table->last = nullptr;
    table->end = nullptr;
    for (Pg6CrtOnexitFn* it = last; it > first; --it) {
        Pg6CrtOnexitFn fn = *(it - 1);
        if (fn != nullptr) {
            fn();
        }
    }
    playgta6::os::guest_crt::free(first);
    return 0;
}

void pg6_crt_register_thread_local_exe_atexit_callback(Pg6CrtTlsAtexitCallback callback) {
    std::lock_guard<std::mutex> lock(g_runtime_mutex);
    g_tls_atexit_callback = callback;
}

int32_t pg6_crt_seh_filter_exe(uint32_t exception_num, void* exception_ptrs) {
    (void)exception_num;
    (void)exception_ptrs;
    return 0;  // EXCEPTION_CONTINUE_SEARCH
}

void pg6_crt_set_app_type(int32_t app_type) {
    g_app_type.store(app_type, std::memory_order_release);
}

int32_t pg6_crt_set_errno(int32_t value) {
    g_crt_errno = value;
    return 0;
}

Pg6CrtInvalidParamHandler pg6_crt_set_invalid_parameter_handler(Pg6CrtInvalidParamHandler handler) {
    std::lock_guard<std::mutex> lock(g_runtime_mutex);
    Pg6CrtInvalidParamHandler prev = g_invalid_param_handler;
    g_invalid_param_handler = handler;
    return prev;
}

void pg6_crt_abort(void) {
    g_last_exit_code.store(3, std::memory_order_release);
    invoke_exit_hook(3, PG6_CRT_EXIT_ABORT);
}

void pg6_crt_exit(int32_t status) {
    g_last_exit_code.store(status, std::memory_order_release);
    run_atexit_callbacks();
    invoke_exit_hook(status, PG6_CRT_EXIT_NORMAL);
}

void pg6_crt_terminate(void) {
    g_last_exit_code.store(3, std::memory_order_release);
    invoke_exit_hook(3, PG6_CRT_EXIT_TERMINATE);
}

// ---------------------------------------------------------------------------
// 7. api-ms-win-crt-stdio-l1-1-0.dll (29 symbols)
// ---------------------------------------------------------------------------

FILE* pg6_crt_acrt_iob_func(uint32_t index) {
    switch (index) {
        case 0:
            return stdin;
        case 1:
            return stdout;
        case 2:
            return stderr;
        default:
            return nullptr;
    }
}

int32_t* pg6_crt_p__commode(void) {
    return &g_commode;
}

int32_t pg6_crt_stdio_common_vfprintf(uint64_t options,
                                      FILE* stream,
                                      const char* format,
                                      void* locale,
                                      va_list arglist) {
    (void)options;
    (void)locale;
    if (stream == nullptr || format == nullptr) {
        g_crt_errno = EINVAL;
        return -1;
    }
    return static_cast<int32_t>(std::vfprintf(stream, format, arglist));
}

int32_t pg6_crt_stdio_common_vfwprintf(uint64_t options,
                                       FILE* stream,
                                       const uint16_t* format,
                                       void* locale,
                                       va_list arglist) {
    (void)options;
    (void)locale;
    if (stream == nullptr || format == nullptr) {
        g_crt_errno = EINVAL;
        return -1;
    }
    std::vector<uint16_t> formatted = format_utf16_va(format, arglist);
    formatted.push_back(0);
    const std::string utf8 = utf16_to_utf8_u16(formatted.data());
    const size_t written = std::fwrite(utf8.data(), 1, utf8.size(), stream);
    if (written != utf8.size()) {
        return -1;
    }
    return static_cast<int32_t>(formatted.size() - 1);
}

int32_t pg6_crt_stdio_common_vsnwprintf_s(uint64_t options,
                                          uint16_t* buffer,
                                          size_t buffer_count,
                                          size_t max_count,
                                          const uint16_t* format,
                                          void* locale,
                                          va_list arglist) {
    (void)options;
    (void)locale;
    if (format == nullptr) {
        g_crt_errno = EINVAL;
        return -1;
    }
    const std::vector<uint16_t> formatted = format_utf16_va(format, arglist);
    if (buffer == nullptr && buffer_count == 0) {
        return static_cast<int32_t>(formatted.size());
    }
    if (buffer == nullptr || buffer_count == 0) {
        g_crt_errno = EINVAL;
        return -1;
    }
    const bool truncate = (max_count == PG6_CRT_TRUNCATE);
    const size_t limit = truncate ? (buffer_count - 1) : std::min(max_count, buffer_count - 1);
    if (formatted.size() <= limit) {
        if (!formatted.empty()) {
            std::memcpy(buffer, formatted.data(), formatted.size() * sizeof(uint16_t));
        }
        buffer[formatted.size()] = 0;
        return static_cast<int32_t>(formatted.size());
    }
    if (truncate) {
        if (limit > 0) {
            std::memcpy(buffer, formatted.data(), limit * sizeof(uint16_t));
        }
        buffer[limit] = 0;
        return -1;
    }
    buffer[0] = 0;
    g_crt_errno = ERANGE;
    return -1;
}

int32_t pg6_crt_stdio_common_vsprintf(uint64_t options,
                                      char* buffer,
                                      size_t buffer_count,
                                      const char* format,
                                      void* locale,
                                      va_list arglist) {
    (void)locale;
    if (format == nullptr) {
        g_crt_errno = EINVAL;
        return -1;
    }
    // _CRT_INTERNAL_PRINTF_STANDARD_SNPRINTF_BEHAVIOR (0x2) returns full needed count.
    const bool standard_snprintf = (options & 0x2ull) != 0;
    const int needed = std::vsnprintf(buffer, buffer_count, format, arglist);
    if (needed < 0) {
        return -1;
    }
    if (buffer == nullptr && buffer_count == 0) {
        return needed;
    }
    if (!standard_snprintf && static_cast<size_t>(needed) >= buffer_count) {
        return -1;
    }
    return needed;
}

int32_t pg6_crt_stdio_common_vsprintf_s(uint64_t options,
                                        char* buffer,
                                        size_t buffer_count,
                                        const char* format,
                                        void* locale,
                                        va_list arglist) {
    (void)options;
    (void)locale;
    if (buffer == nullptr || buffer_count == 0 || format == nullptr) {
        g_crt_errno = EINVAL;
        return -1;
    }
    const int needed = std::vsnprintf(buffer, buffer_count, format, arglist);
    if (needed < 0 || static_cast<size_t>(needed) >= buffer_count) {
        buffer[0] = '\0';
        g_crt_errno = ERANGE;
        return -1;
    }
    return needed;
}

int32_t pg6_crt_stdio_common_vsscanf(uint64_t options,
                                     const char* buffer,
                                     size_t buffer_count,
                                     const char* format,
                                     void* locale,
                                     va_list arglist) {
    (void)options;
    (void)locale;
    if (buffer == nullptr || format == nullptr) {
        g_crt_errno = EINVAL;
        return -1;
    }
    if (buffer_count == static_cast<size_t>(-1)) {
        return static_cast<int32_t>(std::vsscanf(buffer, format, arglist));
    }
    size_t len = 0;
    while (len < buffer_count && buffer[len] != '\0') {
        ++len;
    }
    std::string bounded(buffer, len);
    return static_cast<int32_t>(std::vsscanf(bounded.c_str(), format, arglist));
}

int32_t pg6_crt_stdio_common_vswprintf(uint64_t options,
                                       uint16_t* buffer,
                                       size_t buffer_count,
                                       const uint16_t* format,
                                       void* locale,
                                       va_list arglist) {
    (void)locale;
    if (format == nullptr) {
        g_crt_errno = EINVAL;
        return -1;
    }
    const std::vector<uint16_t> formatted = format_utf16_va(format, arglist);
    if (buffer == nullptr && buffer_count == 0) {
        return static_cast<int32_t>(formatted.size());
    }
    if (buffer == nullptr || buffer_count == 0) {
        g_crt_errno = EINVAL;
        return -1;
    }
    const size_t copy_len = std::min(formatted.size(), buffer_count - 1);
    if (copy_len > 0) {
        std::memcpy(buffer, formatted.data(), copy_len * sizeof(uint16_t));
    }
    buffer[copy_len] = 0;
    const bool standard_snprintf = (options & 0x2ull) != 0;
    if (formatted.size() >= buffer_count) {
        return standard_snprintf ? static_cast<int32_t>(formatted.size()) : -1;
    }
    return static_cast<int32_t>(formatted.size());
}

int32_t pg6_crt_stdio_common_vswprintf_s(uint64_t options,
                                         uint16_t* buffer,
                                         size_t buffer_count,
                                         const uint16_t* format,
                                         void* locale,
                                         va_list arglist) {
    if (buffer == nullptr || buffer_count == 0 || format == nullptr) {
        g_crt_errno = EINVAL;
        return -1;
    }
    return pg6_crt_stdio_common_vsnwprintf_s(options, buffer, buffer_count, buffer_count, format,
                                             locale, arglist);
}

int32_t pg6_crt_stdio_common_vswscanf(uint64_t options,
                                      const uint16_t* buffer,
                                      size_t buffer_count,
                                      const uint16_t* format,
                                      void* locale,
                                      va_list arglist) {
    (void)options;
    (void)locale;
    if (buffer == nullptr || format == nullptr) {
        g_crt_errno = EINVAL;
        return -1;
    }
    const std::string input_utf8 = utf16_to_utf8_u16(buffer, buffer_count);
    const std::string fmt_utf8 = utf16_to_utf8_u16(format);
    return static_cast<int32_t>(std::vsscanf(input_utf8.c_str(), fmt_utf8.c_str(), arglist));
}

int32_t pg6_crt_fseeki64(FILE* stream, int64_t offset, int32_t origin) {
    if (stream == nullptr) {
        g_crt_errno = EINVAL;
        return -1;
    }
    errno = 0;
#if defined(_MSC_VER)
    const int rc = _fseeki64(stream, offset, origin);
#else
    const int rc = fseeko(stream, static_cast<off_t>(offset), origin);
#endif
    if (rc != 0 && errno != 0) {
        g_crt_errno = errno;
    }
    return rc;
}

int32_t pg6_crt_get_stream_buffer_pointers(FILE* stream,
                                           char*** base,
                                           char*** ptr,
                                           int32_t** cnt) {
    if (stream == nullptr) {
        g_crt_errno = EINVAL;
        return EINVAL;
    }
    // msvcp140's basic_filebuf keeps these three pointers and dereferences them for every get/put
    // area access. A host FILE has no MSVC buffer layout, so each stream gets an empty buffer
    // record {base, ptr, cnt} = {0, 0, 0} in guest memory: the stream buffer never sees buffered
    // characters and does every transfer through the stdio functions (fgetc/fputc/fwrite). The
    // out-pointers are guest addresses (stored into guest memory).
    static std::mutex records_mutex;
    static std::vector<std::pair<FILE*, uint64_t>> records;
    uint64_t record = 0;
    {
        std::lock_guard<std::mutex> lock(records_mutex);
        playgta6::os::GuestHeap& heap = playgta6::os::default_guest_heap();
        for (const auto& r : records) {
            if (r.first == stream && heap.owns(r.second)) {
                record = r.second;
            }
        }
        if (record == 0) {
            record = heap.alloc(24, 16, playgta6::os::HeapFamily::Crt, playgta6::os::kHeapZeroMemory);
            if (record == 0) {
                g_crt_errno = ENOMEM;
                return ENOMEM;
            }
            records.emplace_back(stream, record);
        }
    }
    if (base != nullptr) {
        *base = reinterpret_cast<char**>(static_cast<uintptr_t>(record));
    }
    if (ptr != nullptr) {
        *ptr = reinterpret_cast<char**>(static_cast<uintptr_t>(record + 8));
    }
    if (cnt != nullptr) {
        *cnt = reinterpret_cast<int32_t*>(static_cast<uintptr_t>(record + 16));
    }
    return 0;
}

int32_t pg6_crt_set_fmode(int32_t mode) {
    g_fmode.store(mode, std::memory_order_release);
    return 0;
}

FILE* pg6_crt_wfopen(const uint16_t* filename, const uint16_t* mode) {
    if (filename == nullptr || mode == nullptr) {
        g_crt_errno = EINVAL;
        return nullptr;
    }
    const std::string fname_utf8 = utf16_to_utf8_u16(filename);
    const std::string mode_utf8 = utf16_to_utf8_u16(mode);
    return pg6_crt_fopen(fname_utf8.c_str(), mode_utf8.c_str());
}

int32_t pg6_crt_fclose(FILE* stream) {
    if (stream == nullptr) {
        g_crt_errno = EINVAL;
        return EOF;
    }
    return std::fclose(stream);
}

int32_t pg6_crt_fflush(FILE* stream) {
    return std::fflush(stream);
}

int32_t pg6_crt_fgetc(FILE* stream) {
    if (stream == nullptr) {
        g_crt_errno = EINVAL;
        return EOF;
    }
    return std::fgetc(stream);
}

int32_t pg6_crt_fgetpos(FILE* stream, int64_t* pos) {
    if (stream == nullptr || pos == nullptr) {
        g_crt_errno = EINVAL;
        return -1;
    }
    errno = 0;
#if defined(_MSC_VER)
    const int64_t off = _ftelli64(stream);
#else
    const int64_t off = static_cast<int64_t>(ftello(stream));
#endif
    if (off < 0) {
        if (errno != 0) {
            g_crt_errno = errno;
        }
        return -1;
    }
    *pos = off;
    return 0;
}

FILE* pg6_crt_fopen(const char* filename, const char* mode) {
    if (filename == nullptr || mode == nullptr) {
        g_crt_errno = EINVAL;
        return nullptr;
    }
    errno = 0;
    FILE* f = std::fopen(filename, mode);
    if (f == nullptr && errno != 0) {
        g_crt_errno = errno;
    }
    return f;
}

int32_t pg6_crt_fputc(int32_t ch, FILE* stream) {
    if (stream == nullptr) {
        g_crt_errno = EINVAL;
        return EOF;
    }
    return std::fputc(ch, stream);
}

size_t pg6_crt_fread(void* buffer, size_t size, size_t count, FILE* stream) {
    if (buffer == nullptr || stream == nullptr) {
        g_crt_errno = EINVAL;
        return 0;
    }
    return std::fread(buffer, size, count, stream);
}

int32_t pg6_crt_fseek(FILE* stream, int32_t offset, int32_t origin) {
    if (stream == nullptr) {
        g_crt_errno = EINVAL;
        return -1;
    }
    return std::fseek(stream, static_cast<long>(offset), origin);
}

int32_t pg6_crt_fsetpos(FILE* stream, const int64_t* pos) {
    if (stream == nullptr || pos == nullptr) {
        g_crt_errno = EINVAL;
        return -1;
    }
    return pg6_crt_fseeki64(stream, *pos, SEEK_SET);
}

int32_t pg6_crt_ftell(FILE* stream) {
    if (stream == nullptr) {
        g_crt_errno = EINVAL;
        return -1;
    }
    return static_cast<int32_t>(std::ftell(stream));
}

size_t pg6_crt_fwrite(const void* buffer, size_t size, size_t count, FILE* stream) {
    if (buffer == nullptr || stream == nullptr) {
        g_crt_errno = EINVAL;
        return 0;
    }
    return std::fwrite(buffer, size, count, stream);
}

void pg6_crt_rewind(FILE* stream) {
    if (stream != nullptr) {
        std::rewind(stream);
    }
}

int32_t pg6_crt_setvbuf(FILE* stream, char* buffer, int32_t mode, size_t size) {
    if (stream == nullptr) {
        g_crt_errno = EINVAL;
        return -1;
    }
    return std::setvbuf(stream, buffer, mode, size);
}

int32_t pg6_crt_ungetc(int32_t ch, FILE* stream) {
    if (stream == nullptr) {
        g_crt_errno = EINVAL;
        return EOF;
    }
    return std::ungetc(ch, stream);
}

// ---------------------------------------------------------------------------
// 8. api-ms-win-crt-string-l1-1-0.dll (31 symbols)
// ---------------------------------------------------------------------------

int32_t pg6_crt_stricmp(const char* a, const char* b) {
    if (a == nullptr || b == nullptr) {
        g_crt_errno = EINVAL;
        return INT32_MAX;
    }
    while (*a != '\0' && *b != '\0') {
        const uint8_t ca = ascii_to_lower_u8(static_cast<uint8_t>(*a));
        const uint8_t cb = ascii_to_lower_u8(static_cast<uint8_t>(*b));
        if (ca != cb) {
            return static_cast<int32_t>(ca) - static_cast<int32_t>(cb);
        }
        ++a;
        ++b;
    }
    return static_cast<int32_t>(ascii_to_lower_u8(static_cast<uint8_t>(*a))) -
           static_cast<int32_t>(ascii_to_lower_u8(static_cast<uint8_t>(*b)));
}

int32_t pg6_crt_strnicmp(const char* a, const char* b, size_t max_count) {
    if (max_count == 0) {
        return 0;
    }
    if (a == nullptr || b == nullptr) {
        g_crt_errno = EINVAL;
        return INT32_MAX;
    }
    for (size_t i = 0; i < max_count; ++i) {
        const uint8_t ca = ascii_to_lower_u8(static_cast<uint8_t>(a[i]));
        const uint8_t cb = ascii_to_lower_u8(static_cast<uint8_t>(b[i]));
        if (ca != cb || ca == 0) {
            return static_cast<int32_t>(ca) - static_cast<int32_t>(cb);
        }
    }
    return 0;
}

int32_t pg6_crt_wcsicoll(const uint16_t* a, const uint16_t* b) {
    if (a == nullptr || b == nullptr) {
        g_crt_errno = EINVAL;
        return INT32_MAX;
    }
    while (*a != 0 && *b != 0) {
        const uint16_t ca = ascii_to_lower_u16(*a);
        const uint16_t cb = ascii_to_lower_u16(*b);
        if (ca != cb) {
            return ca < cb ? -1 : 1;
        }
        ++a;
        ++b;
    }
    const uint16_t ca = ascii_to_lower_u16(*a);
    const uint16_t cb = ascii_to_lower_u16(*b);
    return (ca > cb) - (ca < cb);
}

int32_t pg6_crt_wcsnicmp(const uint16_t* a, const uint16_t* b, size_t max_count) {
    if (max_count == 0) {
        return 0;
    }
    if (a == nullptr || b == nullptr) {
        g_crt_errno = EINVAL;
        return INT32_MAX;
    }
    for (size_t i = 0; i < max_count; ++i) {
        const uint16_t ca = ascii_to_lower_u16(a[i]);
        const uint16_t cb = ascii_to_lower_u16(b[i]);
        if (ca != cb || ca == 0) {
            return static_cast<int32_t>(ca) - static_cast<int32_t>(cb);
        }
    }
    return 0;
}

int32_t pg6_crt_isalnum(int32_t c) {
    return (c >= 0 && c <= 255) ? std::isalnum(c) : 0;
}
int32_t pg6_crt_isalpha(int32_t c) {
    return (c >= 0 && c <= 255) ? std::isalpha(c) : 0;
}
int32_t pg6_crt_isdigit(int32_t c) {
    return (c >= 0 && c <= 255) ? std::isdigit(c) : 0;
}
int32_t pg6_crt_islower(int32_t c) {
    return (c >= 0 && c <= 255) ? std::islower(c) : 0;
}
int32_t pg6_crt_isspace(int32_t c) {
    return (c >= 0 && c <= 255) ? std::isspace(c) : 0;
}
int32_t pg6_crt_isxdigit(int32_t c) {
    return (c >= 0 && c <= 255) ? std::isxdigit(c) : 0;
}

char* pg6_crt_strcat(char* dst, const char* src) {
    return std::strcat(dst, src);
}

int32_t pg6_crt_strcat_s(char* dst, size_t dst_size, const char* src) {
    if (dst == nullptr || dst_size == 0) {
        g_crt_errno = EINVAL;
        return EINVAL;
    }
    if (src == nullptr) {
        dst[0] = '\0';
        g_crt_errno = EINVAL;
        return EINVAL;
    }
    size_t dst_len = 0;
    while (dst_len < dst_size && dst[dst_len] != '\0') {
        ++dst_len;
    }
    if (dst_len == dst_size) {
        dst[0] = '\0';
        g_crt_errno = EINVAL;
        return EINVAL;
    }
    const size_t src_len = std::strlen(src);
    if (dst_len + src_len + 1 > dst_size) {
        dst[0] = '\0';
        g_crt_errno = ERANGE;
        return ERANGE;
    }
    std::memcpy(dst + dst_len, src, src_len + 1);
    return 0;
}

int32_t pg6_crt_strcmp(const char* a, const char* b) {
    return std::strcmp(a, b);
}

char* pg6_crt_strcpy(char* dst, const char* src) {
    return std::strcpy(dst, src);
}

int32_t pg6_crt_strcpy_s(char* dst, size_t dst_size, const char* src) {
    if (dst == nullptr || dst_size == 0) {
        g_crt_errno = EINVAL;
        return EINVAL;
    }
    if (src == nullptr) {
        dst[0] = '\0';
        g_crt_errno = EINVAL;
        return EINVAL;
    }
    const size_t src_len = std::strlen(src);
    if (src_len + 1 > dst_size) {
        dst[0] = '\0';
        g_crt_errno = ERANGE;
        return ERANGE;
    }
    std::memcpy(dst, src, src_len + 1);
    return 0;
}

size_t pg6_crt_strcspn(const char* str, const char* control) {
    return std::strcspn(str, control);
}

size_t pg6_crt_strlen(const char* str) {
    return std::strlen(str);
}

char* pg6_crt_strncat(char* dst, const char* src, size_t max_count) {
    return std::strncat(dst, src, max_count);
}

int32_t pg6_crt_strncmp(const char* a, const char* b, size_t max_count) {
    return std::strncmp(a, b, max_count);
}

char* pg6_crt_strncpy(char* dst, const char* src, size_t count) {
    return std::strncpy(dst, src, count);
}

int32_t pg6_crt_strncpy_s(char* dst, size_t dst_size, const char* src, size_t count) {
    if (dst == nullptr || dst_size == 0) {
        g_crt_errno = EINVAL;
        return EINVAL;
    }
    if (count == 0) {
        dst[0] = '\0';
        return 0;
    }
    if (src == nullptr) {
        dst[0] = '\0';
        g_crt_errno = EINVAL;
        return EINVAL;
    }
    const bool truncate = (count == PG6_CRT_TRUNCATE);
    const size_t limit = truncate ? (dst_size - 1) : count;
    size_t i = 0;
    while (i < limit && i < dst_size - 1 && src[i] != '\0') {
        dst[i] = src[i];
        ++i;
    }
    if (i < limit && src[i] != '\0') {
        // Ran out of destination space before copying `limit` characters or NUL.
        dst[0] = '\0';
        g_crt_errno = ERANGE;
        return ERANGE;
    }
    dst[i] = '\0';
    if (truncate && src[i] != '\0') {
        return PG6_CRT_STRUNCATE;
    }
    return 0;
}

char* pg6_crt_strpbrk(const char* str, const char* charset) {
    return const_cast<char*>(std::strpbrk(str, charset));
}

char* pg6_crt_strtok_s(char* str, const char* delim, char** context) {
    if (delim == nullptr || context == nullptr) {
        g_crt_errno = EINVAL;
        return nullptr;
    }
    char* cur = (str != nullptr) ? str : *context;
    if (cur == nullptr) {
        return nullptr;
    }
    cur += std::strspn(cur, delim);
    if (*cur == '\0') {
        *context = cur;
        return nullptr;
    }
    char* end = cur + std::strcspn(cur, delim);
    if (*end != '\0') {
        *end = '\0';
        *context = end + 1;
    } else {
        *context = end;
    }
    return cur;
}

char* pg6_crt_strtok(char* str, const char* delim) {
    return pg6_crt_strtok_s(str, delim, &g_strtok_state);
}

int32_t pg6_crt_tolower(int32_t c) {
    return (c >= 0 && c <= 255) ? std::tolower(c) : c;
}

int32_t pg6_crt_toupper(int32_t c) {
    return (c >= 0 && c <= 255) ? std::toupper(c) : c;
}

int32_t pg6_crt_wcscat_s(uint16_t* dst, size_t dst_size, const uint16_t* src) {
    if (dst == nullptr || dst_size == 0) {
        g_crt_errno = EINVAL;
        return EINVAL;
    }
    if (src == nullptr) {
        dst[0] = 0;
        g_crt_errno = EINVAL;
        return EINVAL;
    }
    size_t dst_len = 0;
    while (dst_len < dst_size && dst[dst_len] != 0) {
        ++dst_len;
    }
    if (dst_len == dst_size) {
        dst[0] = 0;
        g_crt_errno = EINVAL;
        return EINVAL;
    }
    const size_t src_len = pg6_crt_wcslen(src);
    if (dst_len + src_len + 1 > dst_size) {
        dst[0] = 0;
        g_crt_errno = ERANGE;
        return ERANGE;
    }
    std::memcpy(dst + dst_len, src, (src_len + 1) * sizeof(uint16_t));
    return 0;
}

int32_t pg6_crt_wcscmp(const uint16_t* a, const uint16_t* b) {
    while (*a != 0 && *a == *b) {
        ++a;
        ++b;
    }
    return (*a > *b) - (*a < *b);
}

int32_t pg6_crt_wcscoll(const uint16_t* a, const uint16_t* b) {
    return pg6_crt_wcscmp(a, b);
}

size_t pg6_crt_wcslen(const uint16_t* str) {
    if (str == nullptr) {
        return 0;
    }
    size_t len = 0;
    while (str[len] != 0) {
        ++len;
    }
    return len;
}

uint16_t* pg6_crt_wcsncpy(uint16_t* dst, const uint16_t* src, size_t count) {
    size_t i = 0;
    for (; i < count && src[i] != 0; ++i) {
        dst[i] = src[i];
    }
    for (; i < count; ++i) {
        dst[i] = 0;
    }
    return dst;
}

// ---------------------------------------------------------------------------
// 9. api-ms-win-crt-time-l1-1-0.dll (9 symbols)
// ---------------------------------------------------------------------------

double pg6_crt_difftime64(int64_t time1, int64_t time0) {
    return static_cast<double>(time1 - time0);
}

void pg6_crt_ftime64(Pg6CrtTimeb64* tp) {
    if (tp == nullptr) {
        return;
    }
    const auto now = std::chrono::system_clock::now();
    const auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    tp->time = ms / 1000;
    tp->millitm = static_cast<uint16_t>(ms >= 0 ? (ms % 1000) : 0);
    tp->timezone = 0;
    tp->dstflag = 0;
    tp->_pad = 0;
}

int32_t pg6_crt_gmtime64_s(Pg6CrtTm* result, const int64_t* timer) {
    if (result == nullptr) {
        g_crt_errno = EINVAL;
        return EINVAL;
    }
    if (timer == nullptr || *timer < 0) {
        std::memset(result, -1, sizeof(*result));
        g_crt_errno = EINVAL;
        return EINVAL;
    }
    const int64_t t = *timer;
    const int64_t days = t / 86400;
    const int64_t rem = t % 86400;
    int64_t year = 0;
    int32_t month = 0;
    int32_t day = 0;
    civil_from_days(days, year, month, day);

    result->tm_sec = static_cast<int32_t>(rem % 60);
    result->tm_min = static_cast<int32_t>((rem / 60) % 60);
    result->tm_hour = static_cast<int32_t>(rem / 3600);
    result->tm_mday = day;
    result->tm_mon = month - 1;
    result->tm_year = static_cast<int32_t>(year - 1900);
    // 1970-01-01 was a Thursday (4).
    result->tm_wday = static_cast<int32_t>((days + 4) % 7);
    result->tm_yday = day_of_year(year, month, day);
    result->tm_isdst = 0;
    return 0;
}

Pg6CrtTm* pg6_crt_gmtime64(const int64_t* timer) {
    if (pg6_crt_gmtime64_s(&g_gmtime_tls, timer) != 0) {
        return nullptr;
    }
    return &g_gmtime_tls;
}

int32_t pg6_crt_localtime64_s(Pg6CrtTm* result, const int64_t* timer) {
    if (result == nullptr) {
        g_crt_errno = EINVAL;
        return EINVAL;
    }
    if (timer == nullptr || *timer < 0) {
        std::memset(result, -1, sizeof(*result));
        g_crt_errno = EINVAL;
        return EINVAL;
    }
    std::tm local_tm{};
#if defined(_MSC_VER)
    const __time64_t t64 = static_cast<__time64_t>(*timer);
    const errno_t err = _localtime64_s(&local_tm, &t64);
    if (err != 0) {
        g_crt_errno = err;
        return err;
    }
#else
    const std::time_t t = static_cast<std::time_t>(*timer);
    if (localtime_r(&t, &local_tm) == nullptr) {
        g_crt_errno = EINVAL;
        return EINVAL;
    }
#endif
    from_std_tm(local_tm, *result);
    return 0;
}

int64_t pg6_crt_mkgmtime64(Pg6CrtTm* tm_ptr) {
    if (tm_ptr == nullptr) {
        g_crt_errno = EINVAL;
        return -1;
    }
    // Normalize month overflow/underflow into years.
    int64_t year = static_cast<int64_t>(tm_ptr->tm_year) + 1900;
    int64_t mon = tm_ptr->tm_mon;
    if (mon < 0 || mon >= 12) {
        const int64_t dy = (mon >= 0) ? (mon / 12) : ((mon - 11) / 12);
        year += dy;
        mon -= dy * 12;
    }
    const int64_t days = days_from_civil(year, static_cast<int32_t>(mon) + 1, 1) +
                         (static_cast<int64_t>(tm_ptr->tm_mday) - 1);
    const int64_t secs = days * 86400 + static_cast<int64_t>(tm_ptr->tm_hour) * 3600 +
                         static_cast<int64_t>(tm_ptr->tm_min) * 60 +
                         static_cast<int64_t>(tm_ptr->tm_sec);
    if (secs < 0) {
        return -1;
    }
    pg6_crt_gmtime64_s(tm_ptr, &secs);
    return secs;
}

int64_t pg6_crt_mktime64(Pg6CrtTm* tm_ptr) {
    if (tm_ptr == nullptr) {
        g_crt_errno = EINVAL;
        return -1;
    }
    std::tm host_tm = to_std_tm(*tm_ptr);
#if defined(_MSC_VER)
    const int64_t result = static_cast<int64_t>(_mktime64(&host_tm));
#else
    const int64_t result = static_cast<int64_t>(std::mktime(&host_tm));
#endif
    if (result != -1) {
        from_std_tm(host_tm, *tm_ptr);
    }
    return result;
}

int64_t pg6_crt_time64(int64_t* timer) {
    const auto now = std::chrono::system_clock::now();
    const int64_t secs =
        std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    if (timer != nullptr) {
        *timer = secs;
    }
    return secs;
}

size_t pg6_crt_strftime(char* buffer,
                        size_t max_size,
                        const char* format,
                        const Pg6CrtTm* tm_ptr) {
    if (buffer == nullptr || max_size == 0 || format == nullptr || tm_ptr == nullptr) {
        g_crt_errno = EINVAL;
        return 0;
    }
    const std::tm host_tm = to_std_tm(*tm_ptr);
    return std::strftime(buffer, max_size, format, &host_tm);
}

// ---------------------------------------------------------------------------
// 10. api-ms-win-crt-utility-l1-1-0.dll (4 symbols)
// ---------------------------------------------------------------------------

void* pg6_crt_bsearch(const void* key,
                      const void* base,
                      size_t num,
                      size_t size,
                      Pg6CrtCompareFn compare) {
    if (compare == nullptr || (num > 0 && (key == nullptr || base == nullptr || size == 0))) {
        g_crt_errno = EINVAL;
        return nullptr;
    }
    return std::bsearch(key, base, num, size, compare);
}

void pg6_crt_qsort(void* base, size_t num, size_t size, Pg6CrtCompareFn compare) {
    if (compare == nullptr || (num > 0 && (base == nullptr || size == 0))) {
        g_crt_errno = EINVAL;
        return;
    }
    std::qsort(base, num, size, compare);
}

int32_t pg6_crt_rand(void) {
    // Matches MSVC UCRT LCG (`RAND_MAX == 0x7FFF`), deterministic across all platforms.
    g_rand_state = g_rand_state * 214013u + 2531011u;
    return static_cast<int32_t>((g_rand_state >> 16) & 0x7FFFu);
}

void pg6_crt_srand(uint32_t seed) {
    g_rand_state = seed;
}

// ---------------------------------------------------------------------------
// 11. VCRUNTIME140.dll (23 symbols) & 12. VCRUNTIME140_1.dll (1 symbol)
// ---------------------------------------------------------------------------

void pg6_crt_cxx_throw_exception(void* exception_object, const void* throw_info) {
    g_current_exception = exception_object;
    g_current_exception_context = const_cast<void*>(throw_info);
    invoke_exit_hook(1, PG6_CRT_EXIT_CXX_THROW);
}

int32_t pg6_crt_c_specific_handler(void* exception_record,
                                   void* establisher_frame,
                                   void* context_record,
                                   void* dispatcher_context) {
    (void)exception_record;
    (void)establisher_frame;
    (void)context_record;
    (void)dispatcher_context;
    return 1;  // ExceptionContinueSearch
}

int32_t pg6_crt_cxx_frame_handler3(void* exception_record,
                                   void* establisher_frame,
                                   void* context_record,
                                   void* dispatcher_context) {
    (void)exception_record;
    (void)establisher_frame;
    (void)context_record;
    (void)dispatcher_context;
    return 1;  // ExceptionContinueSearch
}

int32_t pg6_crt_cxx_frame_handler4(void* exception_record,
                                   void* establisher_frame,
                                   void* context_record,
                                   void* dispatcher_context) {
    (void)exception_record;
    (void)establisher_frame;
    (void)context_record;
    (void)dispatcher_context;
    return 1;  // ExceptionContinueSearch
}

void* pg6_crt_rt_dynamic_cast(void* inptr,
                              int32_t vf_delta,
                              void* src_type,
                              void* target_type,
                              int32_t is_reference) {
    (void)vf_delta;
    (void)is_reference;
    if (inptr == nullptr) {
        return nullptr;
    }
    if (src_type != nullptr && src_type == target_type) {
        return inptr;
    }
    return nullptr;
}

void** pg6_crt_current_exception(void) {
    return &g_current_exception;
}

void** pg6_crt_current_exception_context(void) {
    return &g_current_exception_context;
}

int32_t pg6_crt_intrinsic_setjmp(void* jmp_buf, void* frame) {
    if (jmp_buf != nullptr) {
        std::memcpy(jmp_buf, &frame, sizeof(frame));
    }
    return 0;
}

void pg6_crt_std_exception_copy(const Pg6CrtStdExceptionData* from, Pg6CrtStdExceptionData* to) {
    if (from == nullptr || to == nullptr) {
        return;
    }
    if (from->do_free == 0 || from->what == nullptr) {
        to->what = from->what;
        to->do_free = 0;
        return;
    }
    // `what` is a guest VA (the struct lives in guest memory); the copy is a guest heap block.
    playgta6::os::GuestHeap& heap = playgta6::os::default_guest_heap();
    const char* from_host =
        reinterpret_cast<const char*>(heap.to_host(reinterpret_cast<uintptr_t>(from->what)));
    const size_t len = std::strlen(from_host) + 1;
    char* copy = static_cast<char*>(playgta6::os::guest_crt::malloc(len, &g_crt_errno));
    if (copy != nullptr) {
        std::memcpy(copy, from_host, len);
        to->what = reinterpret_cast<const char*>(static_cast<uintptr_t>(heap.to_guest(copy)));
        to->do_free = 1;
    } else {
        to->what = nullptr;
        to->do_free = 0;
    }
}

void pg6_crt_std_exception_destroy(Pg6CrtStdExceptionData* data) {
    if (data == nullptr) {
        return;
    }
    if (data->do_free != 0 && data->what != nullptr) {
        playgta6::os::GuestHeap& heap = playgta6::os::default_guest_heap();
        heap.free(reinterpret_cast<uintptr_t>(data->what));
    }
    data->what = nullptr;
    data->do_free = 0;
}

void pg6_crt_std_terminate(void) {
    pg6_crt_terminate();
}

int32_t pg6_crt_std_type_info_compare(const Pg6CrtStdTypeInfoData* lhs,
                                      const Pg6CrtStdTypeInfoData* rhs) {
    if (lhs == rhs) {
        return 0;
    }
    if (lhs == nullptr) {
        return -1;
    }
    if (rhs == nullptr) {
        return 1;
    }
    const char* ln = lhs->decorated_name[0] != '\0' ? lhs->decorated_name + 1 : lhs->decorated_name;
    const char* rn = rhs->decorated_name[0] != '\0' ? rhs->decorated_name + 1 : rhs->decorated_name;
    return std::strcmp(ln, rn);
}

int32_t pg6_crt_purecall(void) {
    Pg6CrtPurecallHandler handler = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_runtime_mutex);
        handler = g_purecall_handler;
    }
    if (handler != nullptr) {
        handler();
        return 0;
    }
    invoke_exit_hook(1, PG6_CRT_EXIT_PURECALL);
    return 0;
}

Pg6CrtPurecallHandler pg6_crt_set_purecall_handler(Pg6CrtPurecallHandler handler) {
    std::lock_guard<std::mutex> lock(g_runtime_mutex);
    Pg6CrtPurecallHandler prev = g_purecall_handler;
    g_purecall_handler = handler;
    return prev;
}

void pg6_crt_longjmp(void* jmp_buf, int32_t value) {
    (void)jmp_buf;
    invoke_exit_hook(value == 0 ? 1 : value, PG6_CRT_EXIT_LONGJMP);
}

void* pg6_crt_memchr(const void* buf, int32_t val, size_t max_count) {
    return const_cast<void*>(std::memchr(buf, val, max_count));
}

int32_t pg6_crt_memcmp(const void* lhs, const void* rhs, size_t size) {
    return std::memcmp(lhs, rhs, size);
}

void* pg6_crt_memcpy(void* dst, const void* src, size_t size) {
    return std::memcpy(dst, src, size);
}

void* pg6_crt_memmove(void* dst, const void* src, size_t size) {
    return std::memmove(dst, src, size);
}

void* pg6_crt_memset(void* dst, int32_t val, size_t size) {
    return std::memset(dst, val, size);
}

char* pg6_crt_strchr(const char* str, int32_t ch) {
    return const_cast<char*>(std::strchr(str, ch));
}

char* pg6_crt_strrchr(const char* str, int32_t ch) {
    return const_cast<char*>(std::strrchr(str, ch));
}

char* pg6_crt_strstr(const char* str, const char* substr) {
    return const_cast<char*>(std::strstr(str, substr));
}

uint16_t* pg6_crt_wcsrchr(const uint16_t* str, uint16_t ch) {
    if (str == nullptr) {
        return nullptr;
    }
    const uint16_t* found = nullptr;
    for (;; ++str) {
        if (*str == ch) {
            found = str;
        }
        if (*str == 0) {
            break;
        }
    }
    return const_cast<uint16_t*>(found);
}

// ---------------------------------------------------------------------------
// Registry table & runtime hook management
// ---------------------------------------------------------------------------

#define PG6_UCRT_ENTRY(mod, name, fn) \
    Pg6UcrtExport{PG6_UCRT_MOD_##mod, name, #fn, playgta6::os::win32::proc_address(&fn)},

static const Pg6UcrtExport kUcrtTable[] = {PG6_UCRT_EXPORTS(PG6_UCRT_ENTRY)};
static constexpr size_t kUcrtTableCount = sizeof(kUcrtTable) / sizeof(kUcrtTable[0]);
static_assert(kUcrtTableCount == 186, "Expected exactly 186 UCRT + VCRUNTIME140 symbols");

uint32_t pg6_ucrt_register_all(void) {
    static std::once_flag once;
    static uint32_t registered = 0;
    std::call_once(once, [] {
        using namespace playgta6::os::win32;
        uint32_t count = 0;
        for (const Pg6UcrtExport& e : kUcrtTable) {
            if (register_export(e.module, e.name, 0, e.address)) {
                ++count;
            }
            // Mirror api-ms-win-crt-* exports into ucrtbase.dll so fallback API-set queries work too.
            if (std::string_view(e.module).rfind("api-ms-win-crt-", 0) == 0) {
                register_export("ucrtbase.dll", e.name, 0, e.address);
            }
        }
        registered = count;
    });
    return registered;
}

const Pg6UcrtExport* pg6_ucrt_export_table(size_t* count) {
    if (count != nullptr) {
        *count = kUcrtTableCount;
    }
    return kUcrtTable;
}

Pg6CrtExitHook pg6_crt_set_exit_hook(Pg6CrtExitHook hook) {
    std::lock_guard<std::mutex> lock(g_runtime_mutex);
    Pg6CrtExitHook prev = g_exit_hook;
    g_exit_hook = hook;
    return prev;
}

void pg6_crt_reset_runtime_state(void) {
    std::lock_guard<std::mutex> lock(g_runtime_mutex);
    g_atexit_callbacks.clear();
    g_exit_hook = nullptr;
    g_invalid_param_handler = nullptr;
    g_purecall_handler = nullptr;
    g_user_math_err_handler = nullptr;
    g_tls_atexit_callback = nullptr;
    g_last_exit_code.store(0, std::memory_order_release);
    g_crt_errno = 0;
    g_current_exception = nullptr;
    g_current_exception_context = nullptr;
}

int32_t pg6_crt_get_last_exit_code(void) {
    return g_last_exit_code.load(std::memory_order_acquire);
}

}  // extern "C"
