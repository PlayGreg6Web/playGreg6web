#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

// MSVC sscanf for recompiled code (the scanf counterpart of os/crt_format.h).
//
// The UCRT exports `__stdio_common_vsscanf` / `__stdio_common_vswscanf(options, buffer,
// buffer_count, format, locale, va_list)`. As for printf, the Win64 `va_list` is a pointer to
// consecutive 8-byte slots in guest memory; here every slot holds a guest pointer to the object a
// conversion stores into (plus, for %s %c %[ under _CRT_INTERNAL_SCANF_SECURECRT, an unsigned
// size slot after it). The scanner reads the input and format on the host and stores through the
// guest pointers; the imports are guest-state thunks (pg6_gthunk_*) bound by the import bridge.
//
// Format language (UCRT, "C" locale)
// ----------------------------------
//   white space in the format skips any amount of input white space (including none); any other
//   ordinary character must match the next input character.
//   %[*][width][size]conversion
//     *        scan without storing (not counted)
//     size     hh h l ll L j z t I I32 I64 w; Win64 `long` (l) is 32-bit, L long double = double
//     d i u o x X p   integers (i: 0x / 0 prefixes pick the base; values wrap to the target size)
//     e E f F g G a A floats (float by default, double with l or L), incl. inf / nan / hex floats
//     s c [set] [^set] strings: %s stops at white space and stores a terminator, %c reads width
//              (default 1) characters without one, %[ reads characters in the set and stores a
//              terminator. The narrow function stores narrow characters unless l / w (or %S %C);
//              the wide function stores wide characters only with option 0x2 (legacy wide
//              specifiers) or l / w, otherwise narrow ("C" locale: a unit above 0xFF cannot be
//              stored narrow and ends the scan as a matching failure).
//     n        stores the number of characters consumed so far (not counted)
//     %        matches a '%'
//   Every conversion except %c %[ %n first skips input white space.
//   Secure (option 0x1): %s %c %[ take a size slot (in characters); a field that does not fit
//   stores a terminator at [0] (when the size is not 0) and ends the scan.
// Result: the number of stored fields; -1 (EOF) when the input ends before the first conversion;
// -1 with EINVAL for a null buffer / format, an unknown conversion or an unreadable slot or
// unwritable target.
// buffer_count: (size_t)-1 = up to the terminator; otherwise at most that many characters.

namespace playgta6::os::crt_scan {

inline constexpr uint64_t kSecureCrt = 0x1;
inline constexpr uint64_t kLegacyWideSpecifiers = 0x2;
inline constexpr uint64_t kLegacyMsvcrtCompatibility = 0x4;

// Where scanned values go.
class Target {
public:
    virtual ~Target() = default;
    // The next 8-byte argument slot; false when it cannot be read.
    virtual bool next(uint64_t& slot) = 0;
    // Stores `size` bytes at `address`; false when the memory is not writable.
    virtual bool store(uint64_t address, const void* data, size_t size) = 0;
};

// A Win64 va_list in guest memory, storing into guest memory.
class GuestTarget final : public Target {
public:
    explicit GuestTarget(uint64_t va_list_va) : next_(va_list_va) {}
    bool next(uint64_t& slot) override;
    bool store(uint64_t address, const void* data, size_t size) override;

private:
    uint64_t next_;
};

enum class Status : uint8_t {
    Ok = 0,
    InvalidFormat,  // unknown conversion or truncated specification (EINVAL)
    Unwritable,     // a slot could not be read or a target not written (EINVAL)
};

struct Result {
    int32_t value = 0;  // assigned fields, or -1
    Status status = Status::Ok;
};

Result scan_narrow(std::string_view input, std::string_view format, uint64_t options, Target& target);
Result scan_wide(std::u16string_view input, std::u16string_view format, uint64_t options, Target& target);

}  // namespace playgta6::os::crt_scan

extern "C" {

// Guest-state import thunks for api-ms-win-crt-stdio (the import bridge binds them by name key).
void pg6_gthunk_stdio_common_vsscanf(void* state);
void pg6_gthunk_stdio_common_vswscanf(void* state);

}  // extern "C"
