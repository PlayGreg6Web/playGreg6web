#include "os/ucrt_ext.h"

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <string_view>

#include "os/guest_heap.h"
#include "os/win32_loader.h"

namespace {

std::recursive_mutex g_locale_lock;

// MSVC "C" locale ctype table; entry 0 is EOF (-1), __pctype_func() returns &table[1].
struct CtypeTable {
    uint16_t entries[257]{};
    CtypeTable() {
        for (int c = 0; c < 256; ++c) {
            uint16_t bits = 0;
            if (c < 0x20 || c == 0x7F) {
                bits |= PG6_CRT_CT_CONTROL;
            }
            if (c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r' || c == ' ') {
                bits |= PG6_CRT_CT_SPACE;
            }
            if (c == '\t' || c == ' ') {
                bits |= PG6_CRT_CT_BLANK;
            }
            if (c >= '0' && c <= '9') {
                bits |= PG6_CRT_CT_DIGIT | PG6_CRT_CT_HEX;
            }
            if (c >= 'A' && c <= 'Z') {
                bits |= PG6_CRT_CT_UPPER | 0x0100;
            }
            if (c >= 'a' && c <= 'z') {
                bits |= PG6_CRT_CT_LOWER | 0x0100;
            }
            if ((c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f')) {
                bits |= PG6_CRT_CT_HEX;
            }
            if (c > 0x20 && c < 0x7F && bits == 0) {
                bits = PG6_CRT_CT_PUNCT;
            }
            entries[c + 1] = bits;
        }
    }
};

const CtypeTable& ctype_table() {
    static const CtypeTable table;
    return table;
}

// ___lc_locale_name_func(): LC_ALL .. LC_TIME names, all null in the "C" locale.
uint16_t* g_locale_names[6] = {};
char g_c_locale_name[] = "C";

constexpr std::string_view kDays =
    ":Sun:Sunday:Mon:Monday:Tue:Tuesday:Wed:Wednesday:Thu:Thursday:Fri:Friday:Sat:Saturday";
constexpr std::string_view kMonths =
    ":Jan:January:Feb:February:Mar:March:Apr:April:May:May:Jun:June:Jul:July:Aug:August"
    ":Sep:September:Oct:October:Nov:November:Dec:December";
constexpr size_t kTimeNamesBlock = 1024;

char* guest_narrow(std::string_view text) {
    auto* out = static_cast<char*>(pg6_crt_with_host_errno(
        [&](int32_t* e) { return playgta6::os::guest_crt::malloc(text.size() + 1, e); }));
    if (out != nullptr) {
        std::memcpy(out, text.data(), text.size());
        out[text.size()] = '\0';
    }
    return out;
}

uint16_t* guest_wide(std::string_view ascii) {
    auto* out = static_cast<uint16_t*>(
        pg6_crt_with_host_errno([&](int32_t* e) {
            return playgta6::os::guest_crt::malloc((ascii.size() + 1) * sizeof(uint16_t), e);
        }));
    if (out != nullptr) {
        for (size_t i = 0; i < ascii.size(); ++i) {
            out[i] = static_cast<uint8_t>(ascii[i]);
        }
        out[ascii.size()] = 0;
    }
    return out;
}

bool latin1_alpha(uint16_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == 0xAA || c == 0xB5 ||
           c == 0xBA || (c >= 0xC0 && c <= 0xFF && c != 0xD7 && c != 0xF7);
}

// Win64 `long` result of strtol without errno side effects beyond ERANGE clamping (atol).
int32_t clamp_long(long long value) {
    if (value > INT32_MAX) {
        return INT32_MAX;
    }
    if (value < INT32_MIN) {
        return INT32_MIN;
    }
    return static_cast<int32_t>(value);
}

}  // namespace

// ---------------------------------------------------------------------------
// Registration table (one line per export)
// ---------------------------------------------------------------------------

#define PG6_UCRT_EXT_MOD_CONVERT "api-ms-win-crt-convert-l1-1-0.dll"
#define PG6_UCRT_EXT_MOD_HEAP "api-ms-win-crt-heap-l1-1-0.dll"
#define PG6_UCRT_EXT_MOD_LOCALE "api-ms-win-crt-locale-l1-1-0.dll"
#define PG6_UCRT_EXT_MOD_MATH "api-ms-win-crt-math-l1-1-0.dll"
#define PG6_UCRT_EXT_MOD_RUNTIME "api-ms-win-crt-runtime-l1-1-0.dll"
#define PG6_UCRT_EXT_MOD_STDIO "api-ms-win-crt-stdio-l1-1-0.dll"
#define PG6_UCRT_EXT_MOD_STRING "api-ms-win-crt-string-l1-1-0.dll"
#define PG6_UCRT_EXT_MOD_TIME "api-ms-win-crt-time-l1-1-0.dll"
#define PG6_UCRT_EXT_MOD_UTILITY "api-ms-win-crt-utility-l1-1-0.dll"

#define PG6_UCRT_EXT_EXPORTS(E) \
    E(LOCALE, "_lock_locales", pg6_crt_lock_locales) \
    E(LOCALE, "_unlock_locales", pg6_crt_unlock_locales) \
    E(LOCALE, "___lc_codepage_func", pg6_crt_lc_codepage_func) \
    E(LOCALE, "___lc_collate_cp_func", pg6_crt_lc_collate_cp_func) \
    E(LOCALE, "___lc_locale_name_func", pg6_crt_lc_locale_name_func) \
    E(LOCALE, "___mb_cur_max_func", pg6_crt_mb_cur_max_func) \
    E(LOCALE, "__pctype_func", pg6_crt_pctype_func) \
    E(LOCALE, "setlocale", pg6_crt_setlocale) \
    E(CONVERT, "atol", pg6_crt_atol) \
    E(CONVERT, "btowc", pg6_crt_btowc) \
    E(MATH, "frexp", pg6_crt_frexp) \
    E(MATH, "ldexp", pg6_crt_ldexp) \
    E(STRING, "isupper", pg6_crt_isupper) \
    E(STRING, "iswalnum", pg6_crt_iswalnum) \
    E(STRING, "iswdigit", pg6_crt_iswdigit) \
    E(STRING, "iswspace", pg6_crt_iswspace) \
    E(STRING, "iswxdigit", pg6_crt_iswxdigit) \
    E(STRING, "wcscpy_s", pg6_crt_wcscpy_s) \
    E(STRING, "wcsnlen", pg6_crt_wcsnlen) \
    E(STRING, "_wcsdup", pg6_crt_wcsdup) \
    E(STRING, "__strncnt", pg6_crt_strncnt) \
    E(HEAP, "_aligned_malloc", pg6_crt_aligned_malloc) \
    E(HEAP, "_aligned_free", pg6_crt_aligned_free) \
    E(UTILITY, "rand_s", pg6_crt_rand_s) \
    E(RUNTIME, "_seh_filter_dll", pg6_crt_seh_filter_dll) \
    E(STDIO, "fputs", pg6_crt_fputs) \
    E(STDIO, "fputwc", pg6_crt_fputwc) \
    E(STDIO, "fgetwc", pg6_crt_fgetwc) \
    E(STDIO, "ungetwc", pg6_crt_ungetwc) \
    E(TIME, "_Getdays", pg6_crt_getdays) \
    E(TIME, "_Getmonths", pg6_crt_getmonths) \
    E(TIME, "_Gettnames", pg6_crt_gettnames) \
    E(TIME, "_W_Getdays", pg6_crt_w_getdays) \
    E(TIME, "_W_Getmonths", pg6_crt_w_getmonths) \
    E(TIME, "_W_Gettnames", pg6_crt_w_gettnames) \
    E(TIME, "_Strftime", pg6_crt_strftime_lc) \
    E(TIME, "_Wcsftime", pg6_crt_wcsftime_lc) \
    E(MATH, "ceil", pg6_crt_ceil) \
    E(MATH, "floor", pg6_crt_floor) \
    E(MATH, "sqrt", pg6_crt_sqrt) \
    E(MATH, "sqrtf", pg6_crt_sqrtf) \
    E(MATH, "cosh", pg6_crt_cosh) \
    E(MATH, "sinh", pg6_crt_sinh) \
    E(MATH, "expm1", pg6_crt_expm1) \
    E(MATH, "log1p", pg6_crt_log1p) \
    E(STRING, "wcsncpy_s", pg6_crt_wcsncpy_s) \
    E(STRING, "wcsncat_s", pg6_crt_wcsncat_s) \
    E(CONVERT, "wcstoul", pg6_crt_wcstoul) \
    E(CONVERT, "_itow_s", pg6_crt_itow_s) \
    E(RUNTIME, "__p___wargv", pg6_crt_p___wargv) \
    E(RUNTIME, "_configure_wide_argv", pg6_crt_configure_wide_argv) \
    E(RUNTIME, "_invoke_watson", pg6_crt_invoke_watson)

extern "C" {

uint32_t pg6_ucrt_ext_register_all(void) {
    struct Entry {
        const char* module;
        const char* name;
        Pg6ProcAddress address;
    };
#define PG6_UCRT_EXT_ENTRY(mod, name, fn) \
    Entry{PG6_UCRT_EXT_MOD_##mod, name, playgta6::os::win32::proc_address(&fn)},
    static const Entry kTable[] = {PG6_UCRT_EXT_EXPORTS(PG6_UCRT_EXT_ENTRY)};
#undef PG6_UCRT_EXT_ENTRY
    static std::once_flag once;
    std::call_once(once, [] {
        for (const Entry& e : kTable) {
            playgta6::os::win32::register_export(e.module, e.name, 0, e.address);
            playgta6::os::win32::register_export("ucrtbase.dll", e.name, 0, e.address);
        }
    });
    return static_cast<uint32_t>(sizeof(kTable) / sizeof(kTable[0]));
}

// --- locale ----------------------------------------------------------------

void pg6_crt_lock_locales(void) {
    g_locale_lock.lock();
}

void pg6_crt_unlock_locales(void) {
    g_locale_lock.unlock();
}

uint32_t pg6_crt_lc_codepage_func(void) {
    return 0;  // CP_ACP: the "C" locale has no code page (_CLOCALECP)
}

uint32_t pg6_crt_lc_collate_cp_func(void) {
    return 0;
}

uint16_t** pg6_crt_lc_locale_name_func(void) {
    return g_locale_names;
}

int32_t pg6_crt_mb_cur_max_func(void) {
    return 1;
}

const uint16_t* pg6_crt_pctype_func(void) {
    return ctype_table().entries + 1;
}

char* pg6_crt_setlocale(int32_t category, const char* locale) {
    if (category < 0 || category > 5) {  // LC_ALL .. LC_TIME
        *pg6_crt_errno() = PG6_CRT_EINVAL;
        return nullptr;
    }
    if (locale == nullptr || locale[0] == '\0' || std::strcmp(locale, "C") == 0) {
        return g_c_locale_name;
    }
    return nullptr;  // only the "C" locale exists
}

// --- convert ---------------------------------------------------------------

int32_t pg6_crt_atol(const char* str) {
    return str != nullptr ? clamp_long(std::strtoll(str, nullptr, 10)) : 0;
}

uint16_t pg6_crt_btowc(int32_t c) {
    if (c == EOF || c < 0 || c > 0xFF) {
        return PG6_CRT_WEOF;
    }
    return static_cast<uint16_t>(c);
}

// --- math ------------------------------------------------------------------

double pg6_crt_frexp(double x, int32_t* exponent) {
    int e = 0;
    const double m = std::frexp(x, &e);
    if (exponent != nullptr) {
        *exponent = e;
    }
    return m;
}

double pg6_crt_ldexp(double x, int32_t exponent) {
    return std::ldexp(x, exponent);
}

// --- string ----------------------------------------------------------------

int32_t pg6_crt_isupper(int32_t c) {
    return (c >= 'A' && c <= 'Z') ? PG6_CRT_CT_UPPER : 0;
}

int32_t pg6_crt_iswalnum(uint16_t c) {
    return (latin1_alpha(c) || (c >= '0' && c <= '9')) ? 1 : 0;
}

int32_t pg6_crt_iswdigit(uint16_t c) {
    return (c >= '0' && c <= '9') ? PG6_CRT_CT_DIGIT : 0;
}

int32_t pg6_crt_iswspace(uint16_t c) {
    return ((c >= 0x09 && c <= 0x0D) || c == 0x20 || c == 0x85 || c == 0xA0) ? PG6_CRT_CT_SPACE
                                                                            : 0;
}

int32_t pg6_crt_iswxdigit(uint16_t c) {
    return ((c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f'))
               ? PG6_CRT_CT_HEX
               : 0;
}

int32_t pg6_crt_wcscpy_s(uint16_t* dst, size_t dst_size, const uint16_t* src) {
    if (dst == nullptr || dst_size == 0) {
        *pg6_crt_errno() = PG6_CRT_EINVAL;
        return PG6_CRT_EINVAL;
    }
    if (src == nullptr) {
        dst[0] = 0;
        *pg6_crt_errno() = PG6_CRT_EINVAL;
        return PG6_CRT_EINVAL;
    }
    size_t len = 0;
    while (src[len] != 0) {
        ++len;
    }
    if (len + 1 > dst_size) {
        dst[0] = 0;
        *pg6_crt_errno() = PG6_CRT_ERANGE;
        return PG6_CRT_ERANGE;
    }
    std::memcpy(dst, src, (len + 1) * sizeof(uint16_t));
    return 0;
}

size_t pg6_crt_wcsnlen(const uint16_t* str, size_t max_count) {
    if (str == nullptr) {
        return 0;
    }
    size_t len = 0;
    while (len < max_count && str[len] != 0) {
        ++len;
    }
    return len;
}

uint16_t* pg6_crt_wcsdup(const uint16_t* str) {
    return pg6_crt_with_host_errno(
        [&](int32_t* e) { return playgta6::os::guest_crt::wcsdup(str, e); });
}

size_t pg6_crt_strncnt(const char* str, size_t max_count) {
    size_t len = 0;
    while (len < max_count && str[len] != '\0') {
        ++len;
    }
    return len;
}

// --- heap ------------------------------------------------------------------

void* pg6_crt_aligned_malloc(size_t size, size_t alignment) {
    return pg6_crt_with_host_errno(
        [&](int32_t* e) { return playgta6::os::guest_crt::aligned_malloc(size, alignment, e); });
}

void pg6_crt_aligned_free(void* ptr) {
    playgta6::os::guest_crt::aligned_free(ptr);
}

// --- utility / runtime -----------------------------------------------------

int32_t pg6_crt_rand_s(uint32_t* value) {
    if (value == nullptr) {
        *pg6_crt_errno() = PG6_CRT_EINVAL;
        return PG6_CRT_EINVAL;
    }
    thread_local std::mt19937_64 engine{std::random_device{}()};
    *value = static_cast<uint32_t>(engine());
    return 0;
}

int32_t pg6_crt_seh_filter_dll(uint32_t exception_num, void* exception_ptrs) {
    return pg6_crt_seh_filter_exe(exception_num, exception_ptrs);
}

// --- stdio -----------------------------------------------------------------

int32_t pg6_crt_fputs(const char* str, FILE* stream) {
    if (str == nullptr || stream == nullptr) {
        *pg6_crt_errno() = PG6_CRT_EINVAL;
        return EOF;
    }
    return std::fputs(str, stream) < 0 ? EOF : 0;
}

uint16_t pg6_crt_fputwc(uint16_t ch, FILE* stream) {
    if (stream == nullptr) {
        *pg6_crt_errno() = PG6_CRT_EINVAL;
        return PG6_CRT_WEOF;
    }
    // Text output in the "C" locale: units above 0xFF have no single-byte form.
    if (ch > 0xFF) {
        *pg6_crt_errno() = PG6_CRT_EILSEQ;
        return PG6_CRT_WEOF;
    }
    return std::fputc(ch, stream) == EOF ? PG6_CRT_WEOF : ch;
}

uint16_t pg6_crt_fgetwc(FILE* stream) {
    if (stream == nullptr) {
        *pg6_crt_errno() = PG6_CRT_EINVAL;
        return PG6_CRT_WEOF;
    }
    const int c = std::fgetc(stream);
    return c == EOF ? PG6_CRT_WEOF : static_cast<uint16_t>(c);
}

uint16_t pg6_crt_ungetwc(uint16_t ch, FILE* stream) {
    if (stream == nullptr || ch == PG6_CRT_WEOF || ch > 0xFF) {
        return PG6_CRT_WEOF;
    }
    return std::ungetc(ch, stream) == EOF ? PG6_CRT_WEOF : ch;
}

// --- time ------------------------------------------------------------------

char* pg6_crt_getdays(void) {
    return guest_narrow(kDays);
}

char* pg6_crt_getmonths(void) {
    return guest_narrow(kMonths);
}

void* pg6_crt_gettnames(void) {
    return pg6_crt_with_host_errno(
        [&](int32_t* e) { return playgta6::os::guest_crt::calloc(1, kTimeNamesBlock, e); });
}

uint16_t* pg6_crt_w_getdays(void) {
    return guest_wide(kDays);
}

uint16_t* pg6_crt_w_getmonths(void) {
    return guest_wide(kMonths);
}

void* pg6_crt_w_gettnames(void) {
    return pg6_crt_gettnames();
}

size_t pg6_crt_strftime_lc(char* buffer,
                           size_t max_size,
                           const char* format,
                           const Pg6CrtTm* tm_ptr,
                           void* lc_time) {
    (void)lc_time;  // "C" locale names
    return pg6_crt_strftime(buffer, max_size, format, tm_ptr);
}

size_t pg6_crt_wcsftime_lc(uint16_t* buffer,
                           size_t max_size,
                           const uint16_t* format,
                           const Pg6CrtTm* tm_ptr,
                           void* lc_time) {
    (void)lc_time;
    if (buffer == nullptr || max_size == 0 || format == nullptr || tm_ptr == nullptr) {
        *pg6_crt_errno() = PG6_CRT_EINVAL;
        return 0;
    }
    // Literal units are copied as they are; each %-specification is formatted narrow (its
    // output is ASCII in the "C" locale) and widened.
    std::u16string out;
    for (size_t i = 0; format[i] != 0;) {
        if (format[i] != '%') {
            out.push_back(static_cast<char16_t>(format[i++]));
            continue;
        }
        std::string spec = "%";
        ++i;
        if (format[i] == '#') {
            spec.push_back('#');
            ++i;
        }
        if (format[i] == 0 || format[i] > 0x7F) {
            *pg6_crt_errno() = PG6_CRT_EINVAL;
            buffer[0] = 0;
            return 0;
        }
        spec.push_back(static_cast<char>(format[i++]));
        char piece[256];
        const size_t n = pg6_crt_strftime(piece, sizeof(piece), spec.c_str(), tm_ptr);
        for (size_t k = 0; k < n; ++k) {
            out.push_back(static_cast<char16_t>(static_cast<uint8_t>(piece[k])));
        }
    }
    if (out.size() + 1 > max_size) {
        buffer[0] = 0;
        *pg6_crt_errno() = PG6_CRT_ERANGE;
        return 0;
    }
    for (size_t k = 0; k < out.size(); ++k) {
        buffer[k] = static_cast<uint16_t>(out[k]);
    }
    buffer[out.size()] = 0;
    return out.size();
}

// --- the other redistributables --------------------------------------------

double pg6_crt_ceil(double x) { return std::ceil(x); }
double pg6_crt_floor(double x) { return std::floor(x); }
double pg6_crt_sqrt(double x) { return std::sqrt(x); }
float pg6_crt_sqrtf(float x) { return std::sqrt(x); }
double pg6_crt_cosh(double x) { return std::cosh(x); }
double pg6_crt_sinh(double x) { return std::sinh(x); }
double pg6_crt_expm1(double x) { return std::expm1(x); }
double pg6_crt_log1p(double x) { return std::log1p(x); }

int32_t pg6_crt_wcsncpy_s(uint16_t* dst, size_t dst_size, const uint16_t* src, size_t count) {
    if (dst == nullptr || dst_size == 0) {
        *pg6_crt_errno() = PG6_CRT_EINVAL;
        return PG6_CRT_EINVAL;
    }
    if (count == 0) {
        dst[0] = 0;
        return 0;
    }
    if (src == nullptr) {
        dst[0] = 0;
        *pg6_crt_errno() = PG6_CRT_EINVAL;
        return PG6_CRT_EINVAL;
    }
    const bool truncate = count == PG6_CRT_TRUNCATE;
    const size_t limit = truncate ? dst_size - 1 : count;
    size_t i = 0;
    while (i < limit && i < dst_size - 1 && src[i] != 0) {
        dst[i] = src[i];
        ++i;
    }
    if (i < limit && src[i] != 0) {  // destination full before `limit` units or the terminator
        dst[0] = 0;
        *pg6_crt_errno() = PG6_CRT_ERANGE;
        return PG6_CRT_ERANGE;
    }
    dst[i] = 0;
    return (truncate && src[i] != 0) ? PG6_CRT_STRUNCATE : 0;
}

int32_t pg6_crt_wcsncat_s(uint16_t* dst, size_t dst_size, const uint16_t* src, size_t count) {
    if (dst == nullptr || dst_size == 0) {
        *pg6_crt_errno() = PG6_CRT_EINVAL;
        return PG6_CRT_EINVAL;
    }
    if (src == nullptr && count != 0) {
        dst[0] = 0;
        *pg6_crt_errno() = PG6_CRT_EINVAL;
        return PG6_CRT_EINVAL;
    }
    size_t len = 0;
    while (len < dst_size && dst[len] != 0) {
        ++len;
    }
    if (len == dst_size) {  // destination not terminated
        dst[0] = 0;
        *pg6_crt_errno() = PG6_CRT_EINVAL;
        return PG6_CRT_EINVAL;
    }
    const bool truncate = count == PG6_CRT_TRUNCATE;
    size_t i = 0;
    while ((truncate || i < count) && src[i] != 0) {
        if (len + i + 1 >= dst_size) {
            if (truncate) {
                dst[len + i] = 0;
                return PG6_CRT_STRUNCATE;
            }
            dst[0] = 0;
            *pg6_crt_errno() = PG6_CRT_ERANGE;
            return PG6_CRT_ERANGE;
        }
        dst[len + i] = src[i];
        ++i;
    }
    dst[len + i] = 0;
    return 0;
}

uint32_t pg6_crt_wcstoul(const uint16_t* str, uint16_t** end, int32_t base) {
    if (str == nullptr) {
        *pg6_crt_errno() = PG6_CRT_EINVAL;
        if (end != nullptr) {
            *end = nullptr;
        }
        return 0;
    }
    // Digits, signs, prefixes and white space are ASCII; any other unit ends the number.
    std::string narrow;
    for (size_t i = 0; str[i] != 0; ++i) {
        narrow.push_back(str[i] < 0x80 ? static_cast<char>(str[i]) : '\x01');
    }
    char* narrow_end = nullptr;
    const uint32_t value = pg6_crt_strtoul(narrow.c_str(), &narrow_end, base);
    if (end != nullptr) {
        *end = const_cast<uint16_t*>(str) + (narrow_end - narrow.c_str());
    }
    return value;
}

int32_t pg6_crt_itow_s(int32_t value, uint16_t* buffer, size_t size, int32_t radix) {
    if (buffer == nullptr || size == 0 || radix < 2 || radix > 36) {
        if (buffer != nullptr && size > 0) {
            buffer[0] = 0;
        }
        *pg6_crt_errno() = PG6_CRT_EINVAL;
        return PG6_CRT_EINVAL;
    }
    char digits[40];
    pg6_crt_itoa(value, digits, radix);
    const size_t len = std::strlen(digits);
    if (len + 1 > size) {
        buffer[0] = 0;
        *pg6_crt_errno() = PG6_CRT_ERANGE;
        return PG6_CRT_ERANGE;
    }
    for (size_t i = 0; i <= len; ++i) {
        buffer[i] = static_cast<uint8_t>(digits[i]);
    }
    return 0;
}

uint16_t*** pg6_crt_p___wargv(void) {
    // Wide argv in guest memory ({program name, null}, guest addresses), built once; the
    // returned address of the variable is round-tripped by the import thunk.
    static uint16_t** wargv = nullptr;
    static std::once_flag once;
    std::call_once(once, [] {
        uint16_t* name = guest_wide("game_scarlett_master.exe");
        if (name == nullptr) {
            return;
        }
        auto* table = static_cast<uint64_t*>(pg6_crt_with_host_errno(
            [](int32_t* e) { return playgta6::os::guest_crt::calloc(2, sizeof(uint64_t), e); }));
        if (table == nullptr) {
            return;
        }
        playgta6::os::GuestHeap& heap = playgta6::os::default_guest_heap();
        table[0] = heap.to_guest(name);
        wargv = reinterpret_cast<uint16_t**>(static_cast<uintptr_t>(heap.to_guest(table)));
    });
    return &wargv;
}

int32_t pg6_crt_configure_wide_argv(int32_t mode) {
    (void)mode;
    return 0;
}

void pg6_crt_invoke_watson(const uint16_t*, const uint16_t*, const uint16_t*, uint32_t, uintptr_t) {
    // Fatal by contract: report like abort(); the call site is a no-return import, so lifted
    // code traps right after.
    pg6_crt_abort();
}

}  // extern "C"
