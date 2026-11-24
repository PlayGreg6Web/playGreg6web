#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

// MSVC printf for recompiled code (Track 2, Task O17).
//
// The UCRT exports its whole printf family as `__stdio_common_v*printf*(options, ..., format,
// locale, va_list)`. A Win64 `va_list` is a pointer to consecutive 8-byte argument slots (every
// variadic argument, doubles included, occupies one slot), and for a recompiled caller that
// pointer, the slots and every string they point to live in guest memory. A host `va_list` can
// therefore never be built from it; the shims of os/ucrt_bridge.h that take one stay unbound.
// This module formats from guest slots itself, with MSVC semantics, and implements the imports
// as guest-state thunks (pg6_gthunk_*, bound by the import bridge ahead of the pg6_crt_ shims).
//
// Format language (UCRT, "C" locale)
// ----------------------------------
//   flags      - + space # 0
//   width      digits or * (an int slot; negative = left-justify)
//   precision  .digits or .* (an int slot; negative = none)
//   size       hh h l ll L j z t I I32 I64 w; Win64 `long` (l) is 32-bit, L long double = double
//   conversion d i u o x X c C s S p f F e E g G a A %
// - %s/%c take the function's own character width; %S/%C the other one; h forces narrow, l/w wide.
//   In the wide functions %s/%c are wide only with _CRT_INTERNAL_PRINTF_LEGACY_WIDE_SPECIFIERS
//   (option 0x4, the MSVC default); without it they follow ISO C (narrow).
// - Characters convert between narrow and wide as the "C" locale does: a byte is the code unit
//   of the same value; a wide unit above 0xFF cannot be written narrow (encoding error, EILSEQ).
// - %p: 16 upper-case hex digits; %a: 13 hex digits unless a precision is given
//   (0x1.0000000000000p+0); infinities and NaNs as inf, nan, nan(snan), -nan(ind) (upper case for
//   F E G A); exponents have two digits, three with _CRT_INTERNAL_PRINTF_LEGACY_THREE_DIGIT_EXPONENTS.
// - A null %s prints (null). %n is disabled (MSVC default) and, like an unknown conversion, is an
//   invalid format: the call fails with -1 / EINVAL.
//
// Buffer semantics (the return value is in characters, without the terminator)
// -----------------------------------------------------------------------------
//   vsprintf / vswprintf    option 0x2 (_CRT_INTERNAL_PRINTF_STANDARD_SNPRINTF_BEHAVIOR, C99
//                           vsnprintf): writes min(n, count - 1) + NUL, returns n. Otherwise
//                           (_vsnprintf): n < count -> text + NUL, n; n == count -> text without
//                           NUL, n; n > count -> count characters, no NUL, -1.
//                           buffer == null with count == 0 only measures (returns n).
//   vsprintf_s / vswprintf_s  null buffer or count 0: -1, EINVAL; n >= count: buffer[0] = 0,
//                           -1, ERANGE; else text + NUL, n.
//   vsnwprintf_s            max_count == _TRUNCATE: as much as fits + NUL, -1 when truncated;
//                           n <= max_count: like vswprintf_s; max_count < n and
//                           max_count < count: max_count characters + NUL, -1; else like _s.
//   vfprintf / vfwprintf    the stream must be stdout or stderr (__acrt_iob_func's handles; the
//                           runtime never hands out another FILE*): writes the text, returns n
//                           (wide text written as UTF-8).
// Invalid formats return -1 with EINVAL and, when a buffer exists, buffer[0] = 0.

namespace playgta6::os::crt_format {

inline constexpr uint64_t kLegacyVsprintfNullTermination = 0x1;
inline constexpr uint64_t kStandardSnprintfBehavior = 0x2;
inline constexpr uint64_t kLegacyWideSpecifiers = 0x4;
inline constexpr uint64_t kLegacyMsvcrtCompatibility = 0x8;
inline constexpr uint64_t kLegacyThreeDigitExponents = 0x10;

inline constexpr size_t kTruncate = static_cast<size_t>(-1);

// Where the variadic arguments, and the strings they point to, come from.
class ArgSource {
public:
    virtual ~ArgSource() = default;
    // The next 8-byte argument slot; false when it cannot be read.
    virtual bool next(uint64_t& slot) = 0;
    // The string at `address`: up to the terminator, or `max` units when no terminator comes
    // first (the precision of %.Ns, which need not be terminated). False when unreadable.
    virtual bool narrow(uint64_t address, size_t max, std::string& out) = 0;
    virtual bool wide(uint64_t address, size_t max, std::u16string& out) = 0;
};

// A Win64 va_list in guest memory (strings at guest addresses).
class GuestArgSource final : public ArgSource {
public:
    explicit GuestArgSource(uint64_t va_list_va) : next_(va_list_va) {}
    bool next(uint64_t& slot) override;
    bool narrow(uint64_t address, size_t max, std::string& out) override;
    bool wide(uint64_t address, size_t max, std::u16string& out) override;

private:
    uint64_t next_;
};

enum class Status : uint8_t {
    Ok = 0,
    InvalidFormat,  // unknown conversion, %n, truncated specification (EINVAL)
    Unreadable,     // an argument slot or string could not be read (EINVAL)
    Encoding,       // a character the "C" locale cannot convert (EILSEQ)
};

Status format_narrow(std::string_view format, uint64_t options, ArgSource& args, std::string& out);
Status format_wide(std::u16string_view format, uint64_t options, ArgSource& args, std::u16string& out);

// What a buffer function stores for a formatted text of `length` characters: `units` characters
// of it, then a terminator if `terminate`; `clear_first` stores only a terminator at [0].
struct Store {
    int32_t result = 0;
    size_t units = 0;
    bool terminate = false;
    bool clear_first = false;
    int32_t error = 0;  // errno value to set (0 = none)
};

Store sprintf_store(size_t length, bool has_buffer, size_t count, uint64_t options);
Store sprintf_s_store(size_t length, bool has_buffer, size_t count);
Store snprintf_s_store(size_t length, bool has_buffer, size_t count, size_t max_count);

}  // namespace playgta6::os::crt_format

extern "C" {

// Guest-state import thunks for api-ms-win-crt-stdio (the import bridge binds them by name key).
void pg6_gthunk_stdio_common_vsprintf(void* state);
void pg6_gthunk_stdio_common_vsprintf_s(void* state);
void pg6_gthunk_stdio_common_vswprintf(void* state);
void pg6_gthunk_stdio_common_vswprintf_s(void* state);
void pg6_gthunk_stdio_common_vsnwprintf_s(void* state);
void pg6_gthunk_stdio_common_vfprintf(void* state);
void pg6_gthunk_stdio_common_vfwprintf(void* state);

}  // extern "C"
