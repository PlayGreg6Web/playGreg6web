#pragma once

// Private helpers shared by the win32_compat_*.cpp files (task O15). Not part of the runtime API.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "os/win32_compat.h"
#include "os/win32_kernel.h"

namespace playgta6::os::win32::compat_detail {

// Handle-table tags for insert_opaque_object.
inline constexpr uint32_t kTagFind = 0x444E4946u;      // 'FIND'
inline constexpr uint32_t kTagRegKey = 0x5947454Bu;    // 'KEGY' (registry key)

// Lenient UTF-8 -> UTF-16 (ill-formed sequences become U+FFFD). Used for file names.
std::u16string utf8_to_utf16(std::string_view utf8);

// A path resolved against the virtual roots of os/win32_kernel.cpp.
struct ResolvedPath {
    std::string key;            // normalized: "/userdata/..." or a relative game-data path
    bool is_userdata = false;   // lives in the writable /userdata overlay
    bool is_root = false;       // the (empty) game-data root, e.g. "G:\\"
};

// ERROR_SUCCESS, or ERROR_INVALID_PARAMETER (null) / ERROR_PATH_NOT_FOUND (empty).
uint32_t resolve_path(const char16_t* path, ResolvedPath* out);
uint32_t resolve_path_utf8(std::string_view path, ResolvedPath* out);

// Simple case folding for name comparison: ASCII, Latin-1, basic Greek and Cyrillic.
inline char16_t fold_char(char16_t c) {
    if (c >= u'A' && c <= u'Z') {
        return static_cast<char16_t>(c + 0x20);
    }
    if (c < 0x80) {
        return c;
    }
    if ((c >= 0xC0 && c <= 0xDE && c != 0xD7) || (c >= 0x391 && c <= 0x3A9) ||
        (c >= 0x410 && c <= 0x42F)) {
        return static_cast<char16_t>(c + 0x20);
    }
    if (c >= 0x400 && c <= 0x40F) {
        return static_cast<char16_t>(c + 0x50);
    }
    return c;
}

// Current time as FILETIME ticks.
uint64_t now_ticks();

// FILETIME ticks <-> SYSTEMTIME (UTC). The first fails for ticks beyond the SYSTEMTIME range, the
// second for out-of-range fields (year 1601..30827, valid month/day/time, milliseconds < 1000).
bool ticks_to_system_time(uint64_t ticks, Pg6SystemTime* out);
bool system_time_to_ticks(const Pg6SystemTime& st, uint64_t* out);

// Fills `size` bytes from the platform CSPRNG (std::random_device). False if unavailable.
bool secure_random(void* dst, size_t size);

}  // namespace playgta6::os::win32::compat_detail
