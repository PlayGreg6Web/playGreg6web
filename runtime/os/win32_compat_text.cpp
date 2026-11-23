// Strings, locale and message formatting of the remaining Win32 core surface (task O15):
// MultiByteToWideChar / WideCharToMultiByte (CP_ACP = Windows-1252, CP_UTF8, Latin-1, US-ASCII),
// lstr* / CompareString*, locale queries (the user locale is en-US), FormatMessageA and the
// GetDateFormatEx / GetTimeFormatEx pattern formatter. See os/win32_compat.h.

#include <cstring>
#include <string>

#include "os/win32_compat.h"
#include "os/win32_compat_internal.h"
#include "os/win32_loader.h"

namespace win32 = playgta6::os::win32;
namespace cd = playgta6::os::win32::compat_detail;

namespace {

constexpr uint32_t kBadCp = 0;
enum class Codec { Win1252, Latin1, Ascii, Utf8, Unsupported };

Codec codec_for(uint32_t code_page) {
    switch (code_page) {
        case win32::CP_ACP:
        case win32::CP_THREAD_ACP:
        case win32::CP_WINDOWS_1252:
            return Codec::Win1252;
        case win32::CP_LATIN1:
            return Codec::Latin1;
        case win32::CP_USASCII:
            return Codec::Ascii;
        case win32::CP_UTF8:
            return Codec::Utf8;
        default:
            return Codec::Unsupported;  // CP_OEMCP / CP_MACCP and legacy DBCS pages
    }
}

// Windows-1252 bytes 0x80..0x9F (the undefined ones keep their C1 control code point).
constexpr char16_t kCp1252High[32] = {
    0x20AC, 0x0081, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021, 0x02C6, 0x2030, 0x0160,
    0x2039, 0x0152, 0x008D, 0x017D, 0x008F, 0x0090, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022,
    0x2013, 0x2014, 0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0x009D, 0x017E, 0x0178};

// One decoded UTF-8 scalar. Ill-formed input yields valid=false and consumes the "maximal
// subpart" (Unicode best practice): a lead byte plus the continuation bytes that were acceptable.
struct Utf8Scalar {
    uint32_t cp;
    size_t length;
    bool valid;
};

Utf8Scalar decode_utf8_at(const uint8_t* s, size_t n, size_t i) {
    const uint8_t b0 = s[i];
    if (b0 < 0x80) {
        return {b0, 1, true};
    }
    size_t need = 0;
    uint8_t lo = 0x80, hi = 0xBF;
    uint32_t cp = 0;
    if (b0 >= 0xC2 && b0 <= 0xDF) {
        need = 1;
        cp = b0 & 0x1Fu;
    } else if (b0 >= 0xE0 && b0 <= 0xEF) {
        need = 2;
        cp = b0 & 0x0Fu;
        if (b0 == 0xE0) {
            lo = 0xA0;
        } else if (b0 == 0xED) {
            hi = 0x9F;
        }
    } else if (b0 >= 0xF0 && b0 <= 0xF4) {
        need = 3;
        cp = b0 & 0x07u;
        if (b0 == 0xF0) {
            lo = 0x90;
        } else if (b0 == 0xF4) {
            hi = 0x8F;
        }
    } else {
        return {0, 1, false};  // continuation byte, C0, C1, F5..FF
    }
    size_t used = 1;
    for (size_t k = 0; k < need; ++k) {
        if (i + used >= n) {
            return {0, used, false};
        }
        const uint8_t b = s[i + used];
        const uint8_t min = k == 0 ? lo : 0x80;
        const uint8_t max = k == 0 ? hi : 0xBF;
        if (b < min || b > max) {
            return {0, used, false};
        }
        cp = (cp << 6) | (b & 0x3Fu);
        ++used;
    }
    return {cp, used, true};
}

// Output sink: counts always, writes while there is room.
struct WideSink {
    char16_t* dst;
    int32_t capacity;  // 0 = count only
    int64_t count = 0;
    bool overflow = false;

    void put(char16_t c) {
        if (capacity > 0) {
            if (count < capacity) {
                dst[count] = c;
            } else {
                overflow = true;
            }
        }
        ++count;
    }
};

struct ByteSink {
    char* dst;
    int32_t capacity;
    int64_t count = 0;
    bool overflow = false;

    void put(uint8_t b) {
        if (capacity > 0) {
            if (count < capacity) {
                dst[count] = static_cast<char>(b);
            } else {
                overflow = true;
            }
        }
        ++count;
    }
};

size_t u16_length(const char16_t* s) {
    size_t n = 0;
    while (s[n] != 0) {
        ++n;
    }
    return n;
}

bool is_high_surrogate(char16_t c) { return c >= 0xD800 && c <= 0xDBFF; }
bool is_low_surrogate(char16_t c) { return c >= 0xDC00 && c <= 0xDFFF; }

// ---- date/time pattern formatting (en-US) ---------------------------------------------------

const char16_t* const kMonthNames[12] = {u"January", u"February", u"March",     u"April",
                                         u"May",     u"June",     u"July",      u"August",
                                         u"September", u"October", u"November", u"December"};
const char16_t* const kDayNames[7] = {u"Sunday",   u"Monday", u"Tuesday", u"Wednesday",
                                      u"Thursday", u"Friday", u"Saturday"};

void append_number(std::u16string& out, uint32_t value, int min_digits) {
    std::string digits = std::to_string(value);
    while (static_cast<int>(digits.size()) < min_digits) {
        digits.insert(digits.begin(), '0');
    }
    for (char c : digits) {
        out.push_back(static_cast<char16_t>(c));
    }
}

uint16_t weekday_of(const Pg6SystemTime& st) {
    // Sakamoto's algorithm; 0 = Sunday.
    static const int t[] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
    int y = st.year;
    if (st.month < 3) {
        --y;
    }
    return static_cast<uint16_t>((y + y / 4 - y / 100 + y / 400 + t[st.month - 1] + st.day) % 7);
}

// Formats `st` with a date or time pattern. Quoted text ('...') is literal, '' is an apostrophe.
std::u16string format_pattern(const Pg6SystemTime& st, std::u16string_view pattern) {
    std::u16string out;
    size_t i = 0;
    while (i < pattern.size()) {
        const char16_t c = pattern[i];
        if (c == u'\'') {
            ++i;
            while (i < pattern.size()) {
                if (pattern[i] == u'\'') {
                    if (i + 1 < pattern.size() && pattern[i + 1] == u'\'') {
                        out.push_back(u'\'');
                        i += 2;
                        continue;
                    }
                    ++i;
                    break;
                }
                out.push_back(pattern[i++]);
            }
            continue;
        }
        size_t run = 1;
        while (i + run < pattern.size() && pattern[i + run] == c) {
            ++run;
        }
        i += run;
        switch (c) {
            case u'd':
                if (run <= 2) {
                    append_number(out, st.day, static_cast<int>(run));
                } else {
                    const char16_t* name = kDayNames[weekday_of(st)];
                    out.append(run == 3 ? std::u16string(name, 3) : std::u16string(name));
                }
                break;
            case u'M':
                if (run <= 2) {
                    append_number(out, st.month, static_cast<int>(run));
                } else {
                    const char16_t* name = kMonthNames[st.month - 1];
                    out.append(run == 3 ? std::u16string(name, 3) : std::u16string(name));
                }
                break;
            case u'y':
                if (run == 1) {
                    append_number(out, st.year % 100, 1);
                } else if (run == 2) {
                    append_number(out, st.year % 100, 2);
                } else {
                    append_number(out, st.year, 4);
                }
                break;
            case u'h': {
                const uint32_t h12 = st.hour % 12 == 0 ? 12u : st.hour % 12u;
                append_number(out, h12, run >= 2 ? 2 : 1);
                break;
            }
            case u'H':
                append_number(out, st.hour, run >= 2 ? 2 : 1);
                break;
            case u'm':
                append_number(out, st.minute, run >= 2 ? 2 : 1);
                break;
            case u's':
                append_number(out, st.second, run >= 2 ? 2 : 1);
                break;
            case u't':
                if (run == 1) {
                    out.push_back(st.hour < 12 ? u'A' : u'P');
                } else {
                    out.append(st.hour < 12 ? u"AM" : u"PM");
                }
                break;
            default:
                out.append(run, c);
                break;
        }
    }
    return out;
}

bool locale_is_default(const char16_t* name) {
    if (name == nullptr || name[0] == 0) {
        return true;  // user default / invariant: both render as en-US here
    }
    const char16_t* want = u"en-us";
    size_t i = 0;
    for (; want[i] != 0; ++i) {
        if (cd::fold_char(name[i]) != want[i]) {
            return false;
        }
    }
    return name[i] == 0;
}

// The shared "write a string result" protocol of the date/time/locale APIs: cch == 0 queries the
// size (including the NUL); too small a buffer is ERROR_INSUFFICIENT_BUFFER.
int32_t return_u16(const std::u16string& text, char16_t* dst, int32_t cch) {
    if (cch < 0 || (cch > 0 && dst == nullptr)) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    const int32_t needed = static_cast<int32_t>(text.size() + 1);
    if (cch == 0) {
        win32::SetLastError(win32::ERROR_SUCCESS);
        return needed;
    }
    if (cch < needed) {
        win32::SetLastError(win32::ERROR_INSUFFICIENT_BUFFER);
        return 0;
    }
    std::memcpy(dst, text.data(), text.size() * sizeof(char16_t));
    dst[text.size()] = 0;
    win32::SetLastError(win32::ERROR_SUCCESS);
    return needed;
}

// ---- comparison ---------------------------------------------------------------------------

// Linguistic-ish comparison: case-insensitive first, then lowercase sorts before uppercase.
int compare_text(const char16_t* a, size_t na, const char16_t* b, size_t nb, bool ignore_case) {
    const size_t common = na < nb ? na : nb;
    for (size_t i = 0; i < common; ++i) {
        const char16_t fa = cd::fold_char(a[i]);
        const char16_t fb = cd::fold_char(b[i]);
        if (fa != fb) {
            return fa < fb ? -1 : 1;
        }
    }
    if (na != nb) {
        return na < nb ? -1 : 1;
    }
    if (!ignore_case) {
        for (size_t i = 0; i < common; ++i) {
            if (a[i] != b[i]) {
                return a[i] == cd::fold_char(a[i]) ? -1 : 1;  // lowercase first
            }
        }
    }
    return 0;
}

int ordinal_compare(const char16_t* a, size_t na, const char16_t* b, size_t nb, bool ignore_case) {
    const size_t common = na < nb ? na : nb;
    for (size_t i = 0; i < common; ++i) {
        const char16_t ca = ignore_case ? cd::fold_char(a[i]) : a[i];
        const char16_t cb = ignore_case ? cd::fold_char(b[i]) : b[i];
        if (ca != cb) {
            return ca < cb ? -1 : 1;
        }
    }
    return na == nb ? 0 : (na < nb ? -1 : 1);
}

// Known CompareString flag bits; only the ignore-case bits change the result here.
constexpr uint32_t kCompareKnownFlags = 0x1u | 0x2u | 0x4u | 0x10u | 0x20u | 0x1000u | 0x10000u |
                                        0x20000u | 0x8000000u | 0x20000000u;

struct LocaleEntry {
    const char16_t* name;
    uint32_t lcid;
    bool neutral;
};
constexpr LocaleEntry kLocales[] = {
    {u"en-us", 0x0409, false}, {u"en-gb", 0x0809, false}, {u"en-au", 0x0C09, false},
    {u"en-ca", 0x1009, false}, {u"de-de", 0x0407, false}, {u"fr-fr", 0x040C, false},
    {u"es-es", 0x0C0A, false}, {u"it-it", 0x0410, false}, {u"ja-jp", 0x0411, false},
    {u"ko-kr", 0x0412, false}, {u"pt-br", 0x0416, false}, {u"pt-pt", 0x0816, false},
    {u"ru-ru", 0x0419, false}, {u"zh-cn", 0x0804, false}, {u"zh-tw", 0x0404, false},
    {u"pl-pl", 0x0415, false}, {u"nl-nl", 0x0413, false}, {u"sv-se", 0x041D, false},
    {u"tr-tr", 0x041F, false}, {u"cs-cz", 0x0405, false}, {u"en", 0x0009, true},
    {u"de", 0x0007, true},     {u"fr", 0x000C, true},     {u"es", 0x000A, true},
    {u"it", 0x0010, true},     {u"ja", 0x0011, true},     {u"ko", 0x0012, true},
    {u"pt", 0x0016, true},     {u"ru", 0x0019, true},     {u"zh", 0x0004, true},
};

struct SystemMessage {
    uint32_t id;
    const char* text;
};
constexpr SystemMessage kSystemMessages[] = {
    {0, "The operation completed successfully."},
    {2, "The system cannot find the file specified."},
    {3, "The system cannot find the path specified."},
    {5, "Access is denied."},
    {6, "The handle is invalid."},
    {8, "Not enough memory resources are available to process this command."},
    {18, "There are no more files."},
    {19, "The media is write protected."},
    {38, "Reached the end of the file."},
    {50, "The request is not supported."},
    {80, "The file exists."},
    {87, "The parameter is incorrect."},
    {122, "The data area passed to a system call is too small."},
    {123, "The filename, directory name, or volume label syntax is incorrect."},
    {145, "The directory is not empty."},
    {183, "Cannot create a file when that file already exists."},
    {234, "More data is available."},
    {259, "No more data is available."},
    {267, "The directory name is invalid."},
    {997, "Overlapped I/O operation is in progress."},
    {1400, "Invalid window handle."},
    {1407, "Cannot find window class."},
    {1410, "Class already exists."},
    {1460, "This operation returned because the timeout period expired."},
};

}  // namespace

namespace playgta6::os::win32::compat_detail {

std::u16string utf8_to_utf16(std::string_view utf8) {
    std::u16string out;
    out.reserve(utf8.size());
    const auto* s = reinterpret_cast<const uint8_t*>(utf8.data());
    for (size_t i = 0; i < utf8.size();) {
        const Utf8Scalar d = decode_utf8_at(s, utf8.size(), i);
        i += d.length;
        const uint32_t cp = d.valid ? d.cp : 0xFFFDu;
        if (cp >= 0x10000) {
            const uint32_t v = cp - 0x10000;
            out.push_back(static_cast<char16_t>(0xD800 + (v >> 10)));
            out.push_back(static_cast<char16_t>(0xDC00 + (v & 0x3FF)));
        } else {
            out.push_back(static_cast<char16_t>(cp));
        }
    }
    return out;
}

}  // namespace playgta6::os::win32::compat_detail

extern "C" {

int32_t pg6_win32_multi_byte_to_wide_char(uint32_t code_page,
                                          uint32_t flags,
                                          const char* multi_byte_str,
                                          int32_t cb_multi_byte,
                                          char16_t* wide_char_str,
                                          int32_t cch_wide_char) {
    if (multi_byte_str == nullptr || cb_multi_byte == 0 || cb_multi_byte < -1 ||
        cch_wide_char < 0 || (cch_wide_char > 0 && wide_char_str == nullptr)) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    const Codec codec = codec_for(code_page);
    if (codec == Codec::Unsupported) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    const uint32_t all_flags = win32::MB_PRECOMPOSED | win32::MB_COMPOSITE |
                               win32::MB_USEGLYPHCHARS | win32::MB_ERR_INVALID_CHARS;
    const bool flags_ok =
        codec == Codec::Utf8
            ? (flags & ~win32::MB_ERR_INVALID_CHARS) == 0
            : (flags & ~all_flags) == 0 &&
                  (flags & (win32::MB_PRECOMPOSED | win32::MB_COMPOSITE)) !=
                      (win32::MB_PRECOMPOSED | win32::MB_COMPOSITE);
    if (!flags_ok) {
        win32::SetLastError(win32::ERROR_INVALID_FLAGS);
        return 0;
    }
    const bool strict = (flags & win32::MB_ERR_INVALID_CHARS) != 0;
    const size_t n = cb_multi_byte == -1 ? std::strlen(multi_byte_str) + 1
                                         : static_cast<size_t>(cb_multi_byte);
    const auto* s = reinterpret_cast<const uint8_t*>(multi_byte_str);

    WideSink sink{wide_char_str, cch_wide_char};
    for (size_t i = 0; i < n;) {
        if (codec == Codec::Utf8) {
            const Utf8Scalar d = decode_utf8_at(s, n, i);
            i += d.length;
            if (!d.valid) {
                if (strict) {
                    win32::SetLastError(win32::ERROR_NO_UNICODE_TRANSLATION);
                    return 0;
                }
                sink.put(0xFFFD);
            } else if (d.cp >= 0x10000) {
                const uint32_t v = d.cp - 0x10000;
                sink.put(static_cast<char16_t>(0xD800 + (v >> 10)));
                sink.put(static_cast<char16_t>(0xDC00 + (v & 0x3FF)));
            } else {
                sink.put(static_cast<char16_t>(d.cp));
            }
            continue;
        }
        const uint8_t b = s[i++];
        if (b < 0x80) {
            sink.put(b);
        } else if (codec == Codec::Ascii) {
            if (strict) {
                win32::SetLastError(win32::ERROR_NO_UNICODE_TRANSLATION);
                return 0;
            }
            sink.put(0xFFFD);
        } else if (codec == Codec::Win1252 && b < 0xA0) {
            sink.put(kCp1252High[b - 0x80]);
        } else {
            sink.put(b);  // Latin-1 range: byte value == code point
        }
    }
    if (sink.overflow) {
        win32::SetLastError(win32::ERROR_INSUFFICIENT_BUFFER);
        return 0;
    }
    win32::SetLastError(win32::ERROR_SUCCESS);
    return static_cast<int32_t>(sink.count);
}

int32_t pg6_win32_wide_char_to_multi_byte(uint32_t code_page,
                                          uint32_t flags,
                                          const char16_t* wide_char_str,
                                          int32_t cch_wide_char,
                                          char* multi_byte_str,
                                          int32_t cb_multi_byte,
                                          const char* default_char,
                                          int32_t* used_default_char) {
    if (used_default_char != nullptr) {
        *used_default_char = 0;
    }
    if (wide_char_str == nullptr || cch_wide_char == 0 || cch_wide_char < -1 ||
        cb_multi_byte < 0 || (cb_multi_byte > 0 && multi_byte_str == nullptr)) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    const Codec codec = codec_for(code_page);
    if (codec == Codec::Unsupported) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    const uint32_t single_byte_flags = win32::WC_COMPOSITECHECK | win32::WC_DISCARDNS |
                                       win32::WC_SEPCHARS | win32::WC_DEFAULTCHAR |
                                       win32::WC_NO_BEST_FIT_CHARS;
    if (codec == Codec::Utf8) {
        if ((flags & ~win32::WC_ERR_INVALID_CHARS) != 0) {
            win32::SetLastError(win32::ERROR_INVALID_FLAGS);
            return 0;
        }
        if (default_char != nullptr || used_default_char != nullptr) {
            win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
            return 0;
        }
    } else if ((flags & ~single_byte_flags) != 0) {
        win32::SetLastError(win32::ERROR_INVALID_FLAGS);
        return 0;
    }
    const bool strict = (flags & win32::WC_ERR_INVALID_CHARS) != 0;
    const size_t n = cch_wide_char == -1 ? u16_length(wide_char_str) + 1
                                         : static_cast<size_t>(cch_wide_char);
    const uint8_t fallback = default_char != nullptr ? static_cast<uint8_t>(*default_char) : '?';

    ByteSink sink{multi_byte_str, cb_multi_byte};
    bool used_default = false;
    for (size_t i = 0; i < n; ++i) {
        const char16_t c = wide_char_str[i];
        uint32_t cp = c;
        bool paired = false;
        bool lone = false;
        if (is_high_surrogate(c)) {
            if (i + 1 < n && is_low_surrogate(wide_char_str[i + 1])) {
                cp = 0x10000u + ((static_cast<uint32_t>(c) - 0xD800u) << 10) +
                     (static_cast<uint32_t>(wide_char_str[i + 1]) - 0xDC00u);
                paired = true;
            } else {
                lone = true;
            }
        } else if (is_low_surrogate(c)) {
            lone = true;
        }
        if (codec == Codec::Utf8) {
            if (lone) {
                if (strict) {
                    win32::SetLastError(win32::ERROR_NO_UNICODE_TRANSLATION);
                    return 0;
                }
                cp = 0xFFFD;
            }
            if (cp < 0x80) {
                sink.put(static_cast<uint8_t>(cp));
            } else if (cp < 0x800) {
                sink.put(static_cast<uint8_t>(0xC0 | (cp >> 6)));
                sink.put(static_cast<uint8_t>(0x80 | (cp & 0x3F)));
            } else if (cp < 0x10000) {
                sink.put(static_cast<uint8_t>(0xE0 | (cp >> 12)));
                sink.put(static_cast<uint8_t>(0x80 | ((cp >> 6) & 0x3F)));
                sink.put(static_cast<uint8_t>(0x80 | (cp & 0x3F)));
            } else {
                sink.put(static_cast<uint8_t>(0xF0 | (cp >> 18)));
                sink.put(static_cast<uint8_t>(0x80 | ((cp >> 12) & 0x3F)));
                sink.put(static_cast<uint8_t>(0x80 | ((cp >> 6) & 0x3F)));
                sink.put(static_cast<uint8_t>(0x80 | (cp & 0x3F)));
            }
            if (paired) {
                ++i;
            }
            continue;
        }

        // Single-byte code pages.
        int byte = -1;
        if (!paired && !lone) {
            if (cp < 0x80) {
                byte = static_cast<int>(cp);
            } else if (codec == Codec::Latin1 && cp < 0x100) {
                byte = static_cast<int>(cp);
            } else if (codec == Codec::Win1252) {
                if (cp >= 0xA0 && cp <= 0xFF) {
                    byte = static_cast<int>(cp);
                } else {
                    for (int k = 0; k < 32; ++k) {
                        if (kCp1252High[k] == cp) {
                            byte = 0x80 + k;
                            break;
                        }
                    }
                }
            }
        }
        if (byte >= 0) {
            sink.put(static_cast<uint8_t>(byte));
        } else {
            sink.put(fallback);  // no best-fit mapping: every unmappable character is the default
            used_default = true;
        }
        if (paired) {
            ++i;
        }
    }
    if (sink.overflow) {
        win32::SetLastError(win32::ERROR_INSUFFICIENT_BUFFER);
        return 0;
    }
    if (used_default_char != nullptr) {
        *used_default_char = used_default ? 1 : 0;
    }
    win32::SetLastError(win32::ERROR_SUCCESS);
    return static_cast<int32_t>(sink.count);
}

int32_t pg6_win32_lstrcmp_w(const char16_t* string1, const char16_t* string2) {
    if (string1 == nullptr || string2 == nullptr) {
        return string1 == string2 ? 0 : (string1 == nullptr ? -1 : 1);
    }
    return compare_text(string1, u16_length(string1), string2, u16_length(string2), false);
}

int32_t pg6_win32_lstrcmpi_w(const char16_t* string1, const char16_t* string2) {
    if (string1 == nullptr || string2 == nullptr) {
        return string1 == string2 ? 0 : (string1 == nullptr ? -1 : 1);
    }
    return compare_text(string1, u16_length(string1), string2, u16_length(string2), true);
}

int32_t pg6_win32_lstrlen_w(const char16_t* string) {
    return string == nullptr ? 0 : static_cast<int32_t>(u16_length(string));
}

int32_t pg6_win32_compare_string_w(uint32_t,
                                   uint32_t flags,
                                   const char16_t* string1,
                                   int32_t count1,
                                   const char16_t* string2,
                                   int32_t count2) {
    if (string1 == nullptr || string2 == nullptr || count1 < -1 || count2 < -1) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    if ((flags & ~kCompareKnownFlags) != 0) {
        win32::SetLastError(win32::ERROR_INVALID_FLAGS);
        return 0;
    }
    const size_t n1 = count1 == -1 ? u16_length(string1) : static_cast<size_t>(count1);
    const size_t n2 = count2 == -1 ? u16_length(string2) : static_cast<size_t>(count2);
    const bool ignore_case = (flags & (win32::NORM_IGNORECASE | 0x10u)) != 0;
    win32::SetLastError(win32::ERROR_SUCCESS);
    return compare_text(string1, n1, string2, n2, ignore_case) + 2;  // -1/0/1 -> CSTR_*
}

int32_t pg6_win32_compare_string_ordinal(const char16_t* string1,
                                         int32_t count1,
                                         const char16_t* string2,
                                         int32_t count2,
                                         int32_t ignore_case) {
    if (string1 == nullptr || string2 == nullptr || count1 < -1 || count2 < -1) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    const size_t n1 = count1 == -1 ? u16_length(string1) : static_cast<size_t>(count1);
    const size_t n2 = count2 == -1 ? u16_length(string2) : static_cast<size_t>(count2);
    win32::SetLastError(win32::ERROR_SUCCESS);
    return ordinal_compare(string1, n1, string2, n2, ignore_case != 0) + 2;
}

int32_t pg6_win32_get_user_default_locale_name(char16_t* locale_name, int32_t cch_locale_name) {
    return return_u16(u"en-US", locale_name, cch_locale_name);
}

int32_t pg6_win32_get_user_default_geo_name(char16_t* geo_name, int32_t count) {
    return return_u16(u"US", geo_name, count);
}

uint32_t pg6_win32_locale_name_to_lcid(const char16_t* name, uint32_t flags) {
    constexpr uint32_t kAllowNeutral = 0x08000000u;
    if ((flags & ~kAllowNeutral) != 0) {
        win32::SetLastError(win32::ERROR_INVALID_FLAGS);
        return 0;
    }
    if (name == nullptr) {
        win32::SetLastError(win32::ERROR_SUCCESS);
        return 0x0409;  // LOCALE_NAME_USER_DEFAULT
    }
    if (name[0] == 0) {
        win32::SetLastError(win32::ERROR_SUCCESS);
        return 0x007F;  // LOCALE_INVARIANT
    }
    const size_t n = u16_length(name);
    for (const LocaleEntry& e : kLocales) {
        if (n != u16_length(e.name)) {
            continue;
        }
        bool same = true;
        for (size_t i = 0; i < n; ++i) {
            if (cd::fold_char(name[i]) != e.name[i]) {
                same = false;
                break;
            }
        }
        if (same && (!e.neutral || (flags & kAllowNeutral) != 0)) {
            win32::SetLastError(win32::ERROR_SUCCESS);
            return e.lcid;
        }
    }
    win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
    return 0;
}

uint32_t pg6_win32_format_message_a(uint32_t flags,
                                    const void* source,
                                    uint32_t message_id,
                                    uint32_t,
                                    char* buffer,
                                    uint32_t size,
                                    void*) {
    const bool allocate = (flags & win32::FORMAT_MESSAGE_ALLOCATE_BUFFER) != 0;
    if (buffer == nullptr || ((flags & win32::FORMAT_MESSAGE_FROM_STRING) != 0 &&
                              (flags & win32::FORMAT_MESSAGE_FROM_SYSTEM) != 0)) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    std::string text;
    if ((flags & win32::FORMAT_MESSAGE_FROM_STRING) != 0) {
        if (source == nullptr) {
            win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
            return 0;
        }
        const char* p = static_cast<const char*>(source);
        const bool keep_inserts = (flags & win32::FORMAT_MESSAGE_IGNORE_INSERTS) != 0;
        while (*p != 0) {
            if (*p != '%') {
                text.push_back(*p++);
                continue;
            }
            const char next = p[1];
            if (next == '%') {
                text.push_back('%');
                p += 2;
            } else if (next == 'n') {
                text.append("\r\n");
                p += 2;
            } else if (next == 'r') {
                text.push_back('\r');
                p += 2;
            } else if (next == 't') {
                text.push_back('\t');
                p += 2;
            } else if (next == '0') {
                break;  // %0 ends the message without a trailing newline
            } else if (next >= '1' && next <= '9') {
                const char* start = p;
                p += 2;
                while (*p >= '0' && *p <= '9') {
                    ++p;
                }
                if (*p == '!') {  // %1!d! style format: skipped
                    const char* q = p + 1;
                    while (*q != 0 && *q != '!') {
                        ++q;
                    }
                    if (*q == '!') {
                        p = q + 1;
                    }
                }
                if (keep_inserts) {
                    text.append(start, static_cast<size_t>(p - start));
                }  // else: no argument array is modelled, the insert expands to nothing
            } else {
                text.push_back(*p++);
            }
        }
    } else if ((flags & win32::FORMAT_MESSAGE_FROM_SYSTEM) != 0) {
        // HRESULT_FROM_WIN32 values carry the Win32 code in the low word.
        const uint32_t id = (message_id >> 16) == 0x8007u ? (message_id & 0xFFFFu) : message_id;
        const char* found = nullptr;
        for (const SystemMessage& m : kSystemMessages) {
            if (m.id == id) {
                found = m.text;
                break;
            }
        }
        if (found == nullptr) {
            win32::SetLastError(win32::ERROR_MR_MID_NOT_FOUND);
            return 0;
        }
        text = found;
        if ((flags & win32::FORMAT_MESSAGE_MAX_WIDTH_MASK) == 0) {
            text.append("\r\n");
        }
    } else {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);  // message tables are not modelled
        return 0;
    }

    if (!allocate) {
        if (static_cast<uint64_t>(text.size()) + 1 > size) {
            win32::SetLastError(win32::ERROR_INSUFFICIENT_BUFFER);
            if (size > 0) {
                buffer[0] = 0;
            }
            return 0;
        }
        std::memcpy(buffer, text.c_str(), text.size() + 1);
        win32::SetLastError(win32::ERROR_SUCCESS);
        return static_cast<uint32_t>(text.size());
    }

    // FORMAT_MESSAGE_ALLOCATE_BUFFER: `buffer` is the address of the caller's LPSTR slot. The
    // block comes from the process heap (== LocalAlloc), so LocalFree releases it.
    *reinterpret_cast<uint64_t*>(buffer) = 0;
    const uint64_t block = win32::HeapAlloc(win32::GetProcessHeap(), 0, text.size() + 1);
    playgta6::os::VirtualMemoryArena* arena = win32::get_vmem_arena();
    if (block == 0 || arena == nullptr || !arena->write(block, text.c_str(), text.size() + 1)) {
        if (block != 0) {
            win32::HeapFree(win32::GetProcessHeap(), 0, block);
        }
        win32::SetLastError(win32::ERROR_NOT_ENOUGH_MEMORY);
        return 0;
    }
    *reinterpret_cast<uint64_t*>(buffer) = block;
    win32::SetLastError(win32::ERROR_SUCCESS);
    return static_cast<uint32_t>(text.size());
}

int32_t pg6_win32_get_date_format_ex(const char16_t* locale_name,
                                     uint32_t flags,
                                     const Pg6SystemTime* date,
                                     const char16_t* format,
                                     char16_t* date_str,
                                     int32_t cch_date,
                                     const char16_t* calendar) {
    constexpr uint32_t kShort = 0x1, kLong = 0x2, kYearMonth = 0x8, kMonthDay = 0x80;
    if (!locale_is_default(locale_name) || calendar != nullptr) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    Pg6SystemTime st{};
    if (date != nullptr) {
        uint64_t ticks = 0;
        if (!cd::system_time_to_ticks(*date, &ticks)) {
            win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
            return 0;
        }
        st = *date;
    } else if (!cd::ticks_to_system_time(cd::now_ticks(), &st)) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    std::u16string_view pattern;
    if (format != nullptr) {
        if (flags != 0) {
            win32::SetLastError(win32::ERROR_INVALID_FLAGS);
            return 0;
        }
        pattern = format;
    } else if (flags == 0 || flags == kShort) {
        pattern = u"M/d/yyyy";
    } else if (flags == kLong) {
        pattern = u"dddd, MMMM d, yyyy";
    } else if (flags == kYearMonth) {
        pattern = u"MMMM yyyy";
    } else if (flags == kMonthDay) {
        pattern = u"MMMM d";
    } else {
        win32::SetLastError(win32::ERROR_INVALID_FLAGS);
        return 0;
    }
    return return_u16(format_pattern(st, pattern), date_str, cch_date);
}

int32_t pg6_win32_get_time_format_ex(const char16_t* locale_name,
                                     uint32_t flags,
                                     const Pg6SystemTime* time,
                                     const char16_t* format,
                                     char16_t* time_str,
                                     int32_t cch_time) {
    constexpr uint32_t kNoMinutesOrSeconds = 0x1, kNoSeconds = 0x2, kNoTimeMarker = 0x4,
                       kForce24Hour = 0x8;
    if (!locale_is_default(locale_name)) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    if ((flags & ~(kNoMinutesOrSeconds | kNoSeconds | kNoTimeMarker | kForce24Hour)) != 0) {
        win32::SetLastError(win32::ERROR_INVALID_FLAGS);
        return 0;
    }
    Pg6SystemTime st{};
    if (time != nullptr) {
        // Only the time fields matter; validate them against a fixed valid date.
        Pg6SystemTime probe = *time;
        probe.year = 2000;
        probe.month = 1;
        probe.day = 1;
        uint64_t ticks = 0;
        if (!cd::system_time_to_ticks(probe, &ticks)) {
            win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
            return 0;
        }
        st = *time;
    } else if (!cd::ticks_to_system_time(cd::now_ticks(), &st)) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    std::u16string pattern;
    if (format != nullptr) {
        pattern = format;  // flags are ignored when an explicit pattern is given
    } else {
        const bool h24 = (flags & kForce24Hour) != 0;
        if ((flags & kNoMinutesOrSeconds) != 0) {
            pattern = h24 ? u"HH" : u"h";
        } else if ((flags & kNoSeconds) != 0) {
            pattern = h24 ? u"HH:mm" : u"h:mm";
        } else {
            pattern = h24 ? u"HH:mm:ss" : u"h:mm:ss";
        }
        if (!h24 && (flags & kNoTimeMarker) == 0) {
            pattern += u" tt";
        }
    }
    return return_u16(format_pattern(st, pattern), time_str, cch_time);
}

}  // extern "C"
