// Locale and string shims of task O26 (LCMapStringEx, CompareStringEx, GetLocaleInfoEx,
// GetStringTypeW, GetCPInfo). Locale data is en-US only, consistent with the O15 shims
// (GetUserDefaultLocaleName = "en-US", GetDateFormatEx patterns). CompareStringEx reuses the O15
// comparison through pg6_win32_compare_string_w so both entry points always agree.
//
// Locale names: NULL (user default), "" (invariant, rendered as en-US), "en-US", "en" and
// "!x-sys-default-locale" are served; any other name is rejected with ERROR_INVALID_PARAMETER by
// GetLocaleInfoEx (no data) and, if it is not a name of the O15 locale table either, by
// LCMapStringEx / CompareStringEx. Case mapping is the simple (1:1) Unicode mapping of the BMP
// scripts below, locale independent: Basic Latin, Latin-1, Latin Extended-A/B subsets, Greek,
// Cyrillic, Armenian, Latin Extended Additional, circled and fullwidth Latin letters.

#include <cstring>
#include <string>
#include <vector>

#include "os/win32_compat.h"
#include "os/win32_compat_internal.h"
#include "os/win32_rtl.h"

namespace win32 = playgta6::os::win32;
namespace cd = playgta6::os::win32::compat_detail;

namespace {

constexpr uint32_t kErrorInvalidFlags = 1004u;
constexpr uint32_t kErrorInsufficientBuffer = 122u;

size_t u16_len(const char16_t* s) {
    size_t n = 0;
    while (s[n] != 0) {
        ++n;
    }
    return n;
}

bool in(char16_t c, char16_t lo, char16_t hi) {
    return c >= lo && c <= hi;
}

// ---- simple case mapping ------------------------------------------------------------------

char16_t to_upper(char16_t c) {
    if (c < 0x80) {
        return in(c, u'a', u'z') ? static_cast<char16_t>(c - 0x20) : c;
    }
    if (c < 0x100) {
        if (c == 0xB5) {
            return 0x39C;
        }
        if (c == 0xFF) {
            return 0x178;
        }
        return (c >= 0xE0 && c != 0xF7) ? static_cast<char16_t>(c - 0x20) : c;
    }
    if (c <= 0x17F) {
        if (c == 0x131) {
            return u'I';
        }
        if (c == 0x17F) {
            return u'S';
        }
        if (in(c, 0x100, 0x137) || in(c, 0x14A, 0x177)) {
            return (c & 1) ? static_cast<char16_t>(c - 1) : c;
        }
        if (in(c, 0x139, 0x148) || in(c, 0x179, 0x17E)) {
            return (c & 1) ? c : static_cast<char16_t>(c - 1);
        }
        return c;
    }
    if (in(c, 0x1CD, 0x1DC)) {  // Latin Extended-B: caron letters
        return (c & 1) ? c : static_cast<char16_t>(c - 1);
    }
    if (in(c, 0x1DE, 0x1EF) || in(c, 0x1F8, 0x21F) || in(c, 0x222, 0x233)) {
        return (c & 1) ? static_cast<char16_t>(c - 1) : c;
    }
    if (in(c, 0x3B1, 0x3C9)) {
        return c == 0x3C2 ? char16_t{0x3A3} : static_cast<char16_t>(c - 0x20);
    }
    if (c == 0x3AC) return 0x386;
    if (in(c, 0x3AD, 0x3AF)) return static_cast<char16_t>(c - 0x25);
    if (c == 0x3CA || c == 0x3CB) return static_cast<char16_t>(c - 0x20);
    if (c == 0x3CC) return 0x38C;
    if (c == 0x3CD || c == 0x3CE) return static_cast<char16_t>(c - 0x3F);
    if (in(c, 0x430, 0x44F)) return static_cast<char16_t>(c - 0x20);
    if (in(c, 0x450, 0x45F)) return static_cast<char16_t>(c - 0x50);
    if (in(c, 0x460, 0x481) || in(c, 0x48A, 0x4BF) || in(c, 0x4D0, 0x52F)) {
        return (c & 1) ? static_cast<char16_t>(c - 1) : c;
    }
    if (c == 0x4CF) return 0x4C0;
    if (in(c, 0x4C1, 0x4CE)) return (c & 1) ? c : static_cast<char16_t>(c - 1);
    if (in(c, 0x561, 0x586)) return static_cast<char16_t>(c - 0x30);
    if (in(c, 0x1E00, 0x1E95) || in(c, 0x1EA0, 0x1EFF)) {
        return (c & 1) ? static_cast<char16_t>(c - 1) : c;
    }
    if (in(c, 0x24D0, 0x24E9)) return static_cast<char16_t>(c - 26);
    if (in(c, 0xFF41, 0xFF5A)) return static_cast<char16_t>(c - 0x20);
    return c;
}

char16_t to_lower(char16_t c) {
    if (c < 0x80) {
        return in(c, u'A', u'Z') ? static_cast<char16_t>(c + 0x20) : c;
    }
    if (c < 0x100) {
        return (in(c, 0xC0, 0xDE) && c != 0xD7) ? static_cast<char16_t>(c + 0x20) : c;
    }
    if (c <= 0x17F) {
        if (c == 0x178) {
            return 0xFF;
        }
        if (in(c, 0x100, 0x137) || in(c, 0x14A, 0x177)) {
            return (c & 1) ? c : static_cast<char16_t>(c + 1);
        }
        if (in(c, 0x139, 0x148) || in(c, 0x179, 0x17E)) {
            return (c & 1) ? static_cast<char16_t>(c + 1) : c;
        }
        return c;
    }
    if (in(c, 0x1CD, 0x1DC)) {
        return (c & 1) ? static_cast<char16_t>(c + 1) : c;
    }
    if (in(c, 0x1DE, 0x1EF) || in(c, 0x1F8, 0x21F) || in(c, 0x222, 0x233)) {
        return (c & 1) ? c : static_cast<char16_t>(c + 1);
    }
    if (in(c, 0x391, 0x3A9) && c != 0x3A2) return static_cast<char16_t>(c + 0x20);
    if (c == 0x386) return 0x3AC;
    if (in(c, 0x388, 0x38A)) return static_cast<char16_t>(c + 0x25);
    if (c == 0x3AA || c == 0x3AB) return static_cast<char16_t>(c + 0x20);
    if (c == 0x38C) return 0x3CC;
    if (c == 0x38E || c == 0x38F) return static_cast<char16_t>(c + 0x3F);
    if (in(c, 0x410, 0x42F)) return static_cast<char16_t>(c + 0x20);
    if (in(c, 0x400, 0x40F)) return static_cast<char16_t>(c + 0x50);
    if (in(c, 0x460, 0x481) || in(c, 0x48A, 0x4BF) || in(c, 0x4D0, 0x52F)) {
        return (c & 1) ? c : static_cast<char16_t>(c + 1);
    }
    if (c == 0x4C0) return 0x4CF;
    if (in(c, 0x4C1, 0x4CE)) return (c & 1) ? static_cast<char16_t>(c + 1) : c;
    if (in(c, 0x531, 0x556)) return static_cast<char16_t>(c + 0x30);
    if (in(c, 0x1E00, 0x1E95) || in(c, 0x1EA0, 0x1EFF)) {
        return (c & 1) ? c : static_cast<char16_t>(c + 1);
    }
    if (in(c, 0x24B6, 0x24CF)) return static_cast<char16_t>(c + 26);
    if (in(c, 0xFF21, 0xFF3A)) return static_cast<char16_t>(c + 0x20);
    return c;
}

// ---- character classes --------------------------------------------------------------------

bool is_alpha(char16_t c) {
    if (c < 0x80) {
        return in(c, u'a', u'z') || in(c, u'A', u'Z');
    }
    return c == 0xAA || c == 0xB5 || c == 0xBA || (in(c, 0xC0, 0xFF) && c != 0xD7 && c != 0xF7) ||
           in(c, 0x100, 0x2AF) || (in(c, 0x370, 0x3FF) && c != 0x375 && c != 0x37E && c != 0x384 &&
                                   c != 0x385 && c != 0x387) ||
           in(c, 0x400, 0x481) || in(c, 0x48A, 0x52F) || in(c, 0x531, 0x556) || in(c, 0x561, 0x587) ||
           in(c, 0x5D0, 0x5EA) || in(c, 0x621, 0x64A) || in(c, 0x671, 0x6D3) || in(c, 0x904, 0x939) ||
           in(c, 0xE01, 0xE30) || in(c, 0x1100, 0x11FF) || in(c, 0x1E00, 0x1FFF) ||
           in(c, 0x3041, 0x3096) || in(c, 0x30A1, 0x30FA) || in(c, 0x3400, 0x4DBF) ||
           in(c, 0x4E00, 0x9FFF) || in(c, 0xAC00, 0xD7A3) || in(c, 0xFF21, 0xFF3A) ||
           in(c, 0xFF41, 0xFF5A) || in(c, 0x24B6, 0x24E9);
}

bool is_space(char16_t c) {
    return (c >= 0x09 && c <= 0x0D) || c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           in(c, 0x2000, 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F ||
           c == 0x3000;
}

bool is_blank(char16_t c) {
    return c == 0x09 || c == 0x20 || c == 0xA0 || c == 0x1680 || in(c, 0x2000, 0x200A) ||
           c == 0x202F || c == 0x205F || c == 0x3000;
}

bool is_ascii_punct(char16_t c) {
    return in(c, 0x21, 0x2F) || in(c, 0x3A, 0x40) || in(c, 0x5B, 0x60) || in(c, 0x7B, 0x7E);
}

bool is_punct(char16_t c) {
    if (c < 0x80) {
        return is_ascii_punct(c);
    }
    return (in(c, 0xA1, 0xBF) && c != 0xAA && c != 0xB5 && c != 0xBA && c != 0xB2 && c != 0xB3 &&
            c != 0xB9) ||
           c == 0xD7 || c == 0xF7 || in(c, 0x2010, 0x2027) || in(c, 0x2030, 0x205E) ||
           in(c, 0x3001, 0x3003) || in(c, 0x3008, 0x3011) || in(c, 0xFF01, 0xFF0F) ||
           in(c, 0xFF1A, 0xFF20) || in(c, 0xFF3B, 0xFF40) || in(c, 0xFF5B, 0xFF65);
}

bool is_symbol(char16_t c) {
    return c == u'$' || c == u'+' || c == u'<' || c == u'=' || c == u'>' || c == u'^' || c == u'`' ||
           c == u'|' || c == u'~' || in(c, 0xA2, 0xA9) || c == 0xAC || in(c, 0xAE, 0xB1) ||
           c == 0xB4 || c == 0xB8 || c == 0xD7 || c == 0xF7 || in(c, 0x20A0, 0x20CF) ||
           in(c, 0x2190, 0x23FF);
}

bool is_combining(char16_t c) {
    return in(c, 0x300, 0x36F) || in(c, 0x483, 0x489) || in(c, 0x591, 0x5BD) || in(c, 0x64B, 0x65F) ||
           in(c, 0x1AB0, 0x1AFF) || in(c, 0x1DC0, 0x1DFF) || in(c, 0x20D0, 0x20FF) ||
           in(c, 0xFE20, 0xFE2F);
}

bool is_ascii_digit(char16_t c) {
    return in(c, u'0', u'9');
}

bool is_xdigit(char16_t c) {
    return is_ascii_digit(c) || in(c, u'a', u'f') || in(c, u'A', u'F') || in(c, 0xFF10, 0xFF19) ||
           in(c, 0xFF21, 0xFF26) || in(c, 0xFF41, 0xFF46);
}

bool is_digit(char16_t c) {
    return is_ascii_digit(c) || in(c, 0xFF10, 0xFF19) || c == 0xB2 || c == 0xB3 || c == 0xB9 ||
           in(c, 0x660, 0x669) || in(c, 0x6F0, 0x6F9);
}

bool is_surrogate(char16_t c) {
    return in(c, 0xD800, 0xDFFF);
}

bool is_rtl(char16_t c) {
    return in(c, 0x590, 0x5FF) || in(c, 0x600, 0x6FF) || in(c, 0x700, 0x7FF);
}

uint16_t ctype1(char16_t c) {
    uint16_t r = 0;
    if (is_surrogate(c)) {
        return 0;
    }
    r |= 0x200;  // C1_DEFINED
    if (c < 0x20 || c == 0x7F || in(c, 0x80, 0x9F)) {
        r |= 0x20;  // C1_CNTRL
    }
    if (is_space(c)) {
        r |= 0x08;
    }
    if (is_blank(c)) {
        r |= 0x40;
    }
    if (is_digit(c)) {
        r |= 0x04;
    }
    if (is_xdigit(c)) {
        r |= 0x80;
    }
    if (is_punct(c)) {
        r |= 0x10;
    }
    if (is_alpha(c)) {
        r |= 0x100;
        if (to_lower(c) != c) {
            r |= 0x01;
        } else if (to_upper(c) != c) {
            r |= 0x02;
        }
    }
    return r;
}

uint16_t ctype2(char16_t c) {
    if (c < 0x20 && c != 0x09 && c != 0x0A && c != 0x0B && c != 0x0C && c != 0x0D) {
        return (c >= 0x1C && c <= 0x1E) ? 8 : (c == 0x1F ? 9 : 0);
    }
    if (c == 0x0A || c == 0x0D || c == 0x85 || c == 0x2029) return 8;   // block separator
    if (c == 0x09 || c == 0x0B) return 9;                                 // segment separator
    if (c == 0x0C || c == 0x20 || c == 0x1680 || in(c, 0x2000, 0x200A) || c == 0x2028 ||
        c == 0x205F || c == 0x3000) {
        return 10;  // whitespace
    }
    if (in(c, 0x660, 0x669)) return 6;                                             // AN
    if (is_digit(c)) return 3;                                                     // EN
    if (c == u'+' || c == u'-') return 4;                                          // ES
    if (c == u'#' || c == u'$' || c == u'%' || in(c, 0xA2, 0xA5) || c == 0xB0 || c == 0xB1) return 5;  // ET
    if (c == u',' || c == u'.' || c == u'/' || c == u':' || c == 0xA0) return 7;   // CS
    if (is_alpha(c)) return is_rtl(c) ? 2 : 1;
    if (c < 0x80 || is_punct(c) || is_symbol(c)) return 11;                        // ON
    if (is_rtl(c)) return 2;
    return 0;
}

uint16_t ctype3(char16_t c) {
    uint16_t r = 0;
    if (is_combining(c)) {
        r |= 0x1 | 0x2;  // C3_NONSPACING | C3_DIACRITIC
    }
    if (c == u'^' || c == u'`' || c == 0xA8 || c == 0xAF || c == 0xB4 || c == 0xB8) {
        r |= 0x2;
    }
    if (is_symbol(c)) {
        r |= 0x8;
    }
    if (in(c, 0x30A1, 0x30FA)) r |= 0x10;
    if (in(c, 0x3041, 0x3096)) r |= 0x20;
    if (in(c, 0xFF61, 0xFFDC)) r |= 0x40;
    if (in(c, 0xFF01, 0xFF5E)) r |= 0x80;
    if (in(c, 0x3400, 0x4DBF) || in(c, 0x4E00, 0x9FFF)) r |= 0x100;
    if (is_alpha(c)) r |= 0x8000;
    if (is_alpha(c) || is_digit(c)) r |= 0x400;
    return r;
}

// ---- locale names -------------------------------------------------------------------------

bool name_equals_ci(const char16_t* name, const char16_t* lower_ascii) {
    size_t i = 0;
    for (; lower_ascii[i] != 0; ++i) {
        if (cd::fold_char(name[i]) != lower_ascii[i]) {
            return false;
        }
        if (name[i] == 0) {
            return false;
        }
    }
    return name[i] == 0;
}

// The locales GetLocaleInfoEx has data for.
bool locale_has_data(const char16_t* name) {
    if (name == nullptr || name[0] == 0) {
        return true;
    }
    return name_equals_ci(name, u"en-us") || name_equals_ci(name, u"en") ||
           name_equals_ci(name, u"!x-sys-default-locale");
}

// A locale name LCMapStringEx / CompareStringEx accept: the served ones plus the names of the
// O15 locale table (their text handling is locale independent here).
bool locale_known(const char16_t* name) {
    if (locale_has_data(name)) {
        return true;
    }
    return pg6_win32_locale_name_to_lcid(name, 0x08000000u) != 0;
}

// The shared "write a string result" protocol: cch == 0 queries the size (including the NUL).
int32_t return_text(const std::u16string& text, char16_t* dst, int32_t cch) {
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
        win32::SetLastError(kErrorInsufficientBuffer);
        return 0;
    }
    std::memcpy(dst, text.data(), text.size() * sizeof(char16_t));
    dst[text.size()] = 0;
    win32::SetLastError(win32::ERROR_SUCCESS);
    return needed;
}

struct LocaleItem {
    uint32_t type;
    const char16_t* text;
    int32_t number;  // value for LOCALE_RETURN_NUMBER, -1 if the item is not numeric
};

constexpr const char16_t* kDayNames[7] = {u"Monday", u"Tuesday", u"Wednesday", u"Thursday",
                                          u"Friday", u"Saturday", u"Sunday"};
constexpr const char16_t* kDayAbbr[7] = {u"Mon", u"Tue", u"Wed", u"Thu", u"Fri", u"Sat", u"Sun"};
constexpr const char16_t* kMonthNames[12] = {u"January", u"February", u"March",     u"April",
                                             u"May",     u"June",     u"July",      u"August",
                                             u"September", u"October", u"November", u"December"};
constexpr const char16_t* kMonthAbbr[12] = {u"Jan", u"Feb", u"Mar", u"Apr", u"May", u"Jun",
                                            u"Jul", u"Aug", u"Sep", u"Oct", u"Nov", u"Dec"};

constexpr LocaleItem kItems[] = {
    {0x0001, u"0409", 0x409},                       // LOCALE_ILANGUAGE
    {0x0002, u"English (United States)", -1},       // LOCALE_SLANGUAGE / SLOCALIZEDDISPLAYNAME
    {0x0003, u"ENU", -1},                           // SABBREVLANGNAME
    {0x0004, u"English", -1},                       // SNATIVELANGNAME
    {0x0005, u"1", 1},                              // ICOUNTRY
    {0x0006, u"United States", -1},                 // SCOUNTRY / SLOCALIZEDCOUNTRYNAME
    {0x0007, u"USA", -1},                           // SABBREVCTRYNAME
    {0x0008, u"United States", -1},                 // SNATIVECTRYNAME
    {0x0009, u"0409", 0x409},                       // IDEFAULTLANGUAGE
    {0x000A, u"1", 1},                              // IDEFAULTCOUNTRY
    {0x000B, u"437", 437},                          // IDEFAULTCODEPAGE (OEM)
    {0x000C, u",", -1},                             // SLIST
    {0x000D, u"1", 1},                              // IMEASURE (US)
    {0x000E, u".", -1},                             // SDECIMAL
    {0x000F, u",", -1},                             // STHOUSAND
    {0x0010, u"3;0", -1},                           // SGROUPING
    {0x0011, u"2", 2},                              // IDIGITS
    {0x0012, u"1", 1},                              // ILZERO
    {0x0013, u"0123456789", -1},                    // SNATIVEDIGITS
    {0x0014, u"$", -1},                             // SCURRENCY
    {0x0015, u"USD", -1},                           // SINTLSYMBOL
    {0x0016, u".", -1},                             // SMONDECIMALSEP
    {0x0017, u",", -1},                             // SMONTHOUSANDSEP
    {0x0018, u"3;0", -1},                           // SMONGROUPING
    {0x0019, u"2", 2},                              // ICURRDIGITS
    {0x001B, u"0", 0},                              // ICURRENCY
    {0x001C, u"0", 0},                              // INEGCURR
    {0x001D, u"/", -1},                             // SDATE
    {0x001E, u":", -1},                             // STIME
    {0x001F, u"M/d/yyyy", -1},                      // SSHORTDATE
    {0x0020, u"dddd, MMMM d, yyyy", -1},            // SLONGDATE
    {0x0021, u"0", 0},                              // IDATE (month-day-year)
    {0x0022, u"2", 2},                              // ILDATE
    {0x0023, u"0", 0},                              // ITIME (12 hour)
    {0x0024, u"1", 1},                              // ICENTURY
    {0x0025, u"0", 0},                              // ITLZERO
    {0x0026, u"0", 0},                              // IDAYLZERO
    {0x0027, u"0", 0},                              // IMONLZERO
    {0x0028, u"AM", -1},                            // S1159
    {0x0029, u"PM", -1},                            // S2359
    {0x0050, u"", -1},                              // SPOSITIVESIGN
    {0x0051, u"-", -1},                             // SNEGATIVESIGN
    {0x0052, u"3", 3},                              // IPOSSIGNPOSN
    {0x0053, u"0", 0},                              // INEGSIGNPOSN
    {0x0054, u"1", 1},                              // IPOSSYMPRECEDES
    {0x0055, u"0", 0},                              // IPOSSEPBYSPACE
    {0x0056, u"1", 1},                              // INEGSYMPRECEDES
    {0x0057, u"0", 0},                              // INEGSEPBYSPACE
    {0x0059, u"en", -1},                            // SISO639LANGNAME
    {0x005A, u"US", -1},                            // SISO3166CTRYNAME
    {0x005B, u"244", 244},                          // IGEOID
    {0x005C, u"en-US", -1},                         // SNAME
    {0x005D, u"h:mm:ss", -1},                       // SDURATION
    {0x0067, u"eng", -1},                           // SISO639LANGNAME2
    {0x0068, u"USA", -1},                           // SISO3166CTRYNAME2
    {0x0069, u"NaN", -1},                           // SNAN
    {0x006A, u"Infinity", -1},                      // SPOSINFINITY
    {0x006B, u"-Infinity", -1},                     // SNEGINFINITY
    {0x006D, u"en", -1},                            // SPARENT
    {0x006E, u"Courier New", -1},                   // SCONSOLEFALLBACKNAME
    {0x006F, u"English", -1},                       // SLOCALIZEDLANGUAGENAME
    {0x0072, u"English (United States)", -1},       // SENGLISHDISPLAYNAME
    {0x0073, u"English (United States)", -1},       // SNATIVEDISPLAYNAME
    {0x0079, u"h:mm tt", -1},                       // SSHORTTIME
    {0x1001, u"English", -1},                       // SENGLISHLANGUAGENAME
    {0x1002, u"United States", -1},                 // SENGLISHCOUNTRYNAME
    {0x1003, u"h:mm:ss tt", -1},                    // STIMEFORMAT
    {0x1004, u"1252", 1252},                        // IDEFAULTANSICODEPAGE
    {0x1005, u"0", 0},                              // ITIMEMARKPOSN
    {0x1006, u"MMMM yyyy", -1},                     // SYEARMONTH
    {0x1009, u"1", 1},                              // ICALENDARTYPE
    {0x100A, u"1", 1},                              // IPAPERSIZE (Letter)
    {0x100B, u"0", 0},                              // IOPTIONALCALENDAR
    {0x100C, u"6", 6},                              // IFIRSTDAYOFWEEK (Sunday)
    {0x100D, u"0", 0},                              // IFIRSTWEEKOFYEAR
    {0x100E, u"", -1},                              // SMONTHNAME13
    {0x1010, u"1", 1},                              // INEGNUMBER
    {0x1013, u"Default", -1},                       // SSORTNAME
};

bool locale_item(uint32_t type, std::u16string* text, int32_t* number) {
    if (type >= 0x2A && type <= 0x30) {
        *text = kDayNames[type - 0x2A];
        *number = -1;
        return true;
    }
    if (type >= 0x31 && type <= 0x37) {
        *text = kDayAbbr[type - 0x31];
        *number = -1;
        return true;
    }
    if (type >= 0x38 && type <= 0x43) {
        *text = kMonthNames[type - 0x38];
        *number = -1;
        return true;
    }
    if (type >= 0x44 && type <= 0x4F) {
        *text = kMonthAbbr[type - 0x44];
        *number = -1;
        return true;
    }
    for (const LocaleItem& item : kItems) {
        if (item.type == type) {
            *text = item.text;
            *number = item.number;
            return true;
        }
    }
    return false;
}

// Case-insensitive sort key bytes consistent with compare_string_w: big-endian (fold + 1) per
// character, a 0x0000 terminator, then (unless ignoring case) one case byte per character
// (1 = lowercase/caseless, 2 = uppercase) and a 0x00 terminator.
void append_sort_key(const char16_t* s, size_t n, bool ignore_case, std::vector<uint8_t>* out) {
    for (size_t i = 0; i < n; ++i) {
        const char16_t folded = cd::fold_char(s[i]);
        const uint16_t weight = folded == 0xFFFF ? uint16_t{0xFFFF} : static_cast<uint16_t>(folded + 1);
        out->push_back(static_cast<uint8_t>(weight >> 8));
        out->push_back(static_cast<uint8_t>(weight & 0xFF));
    }
    out->push_back(0);
    out->push_back(0);
    if (!ignore_case) {
        for (size_t i = 0; i < n; ++i) {
            out->push_back(s[i] == cd::fold_char(s[i]) ? uint8_t{1} : uint8_t{2});
        }
    }
    out->push_back(0);
}

}  // namespace

extern "C" {

int32_t pg6_win32_lc_map_string_ex(const char16_t* locale_name,
                                   uint32_t map_flags,
                                   const char16_t* src_str,
                                   int32_t cch_src,
                                   char16_t* dest_str,
                                   int32_t cch_dest,
                                   void* version_information,
                                   void* reserved,
                                   uint64_t sort_handle) {
    (void)version_information;
    (void)sort_handle;
    if (src_str == nullptr || cch_src == 0 || cch_src < -1 || cch_dest < 0 ||
        (cch_dest > 0 && dest_str == nullptr) || reserved != nullptr) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    if (!locale_known(locale_name)) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    constexpr uint32_t kNormMask = 0x1u | 0x2u | 0x4u | 0x8u | 0x10u | 0x10000u | 0x20000u;  // NORM_* / SORT_*
    constexpr uint32_t kLinguisticCasing = 0x01000000u;
    const uint32_t op = map_flags & (win32::LCMAP_LOWERCASE | win32::LCMAP_UPPERCASE | win32::LCMAP_SORTKEY);
    const uint32_t known = kNormMask | win32::LCMAP_LOWERCASE | win32::LCMAP_UPPERCASE |
                           win32::LCMAP_SORTKEY | kLinguisticCasing;
    // BYTEREV, kana/width mappings and Chinese conversions are not implemented.
    if ((map_flags & ~known) != 0 || op == 0 ||
        ((op & win32::LCMAP_LOWERCASE) != 0 && (op & win32::LCMAP_UPPERCASE) != 0) ||
        ((op & win32::LCMAP_SORTKEY) != 0 && op != win32::LCMAP_SORTKEY)) {
        win32::SetLastError(kErrorInvalidFlags);
        return 0;
    }
    const bool null_terminated = cch_src == -1;
    const size_t n = null_terminated ? u16_len(src_str) : static_cast<size_t>(cch_src);

    if (op == win32::LCMAP_SORTKEY) {
        std::vector<uint8_t> key;
        // The 1-byte-per-char case weights are skipped by the NORM_IGNORECASE flag.
        append_sort_key(src_str, n, (map_flags & (0x1u | 0x10u)) != 0, &key);
        if (cch_dest == 0) {
            win32::SetLastError(win32::ERROR_SUCCESS);
            return static_cast<int32_t>(key.size());  // bytes
        }
        if (static_cast<size_t>(cch_dest) * sizeof(char16_t) < key.size()) {
            win32::SetLastError(kErrorInsufficientBuffer);
            return 0;
        }
        std::memcpy(dest_str, key.data(), key.size());
        win32::SetLastError(win32::ERROR_SUCCESS);
        return static_cast<int32_t>(key.size());
    }

    const int32_t needed = static_cast<int32_t>(null_terminated ? n + 1 : n);
    if (cch_dest == 0) {
        win32::SetLastError(win32::ERROR_SUCCESS);
        return needed;
    }
    if (cch_dest < needed) {
        win32::SetLastError(kErrorInsufficientBuffer);
        return 0;
    }
    const bool upper = (op & win32::LCMAP_UPPERCASE) != 0;
    for (size_t i = 0; i < static_cast<size_t>(needed); ++i) {
        // Read before write: source and destination may be the same buffer.
        const char16_t c = src_str[i];
        dest_str[i] = upper ? to_upper(c) : to_lower(c);
    }
    win32::SetLastError(win32::ERROR_SUCCESS);
    return needed;
}

int32_t pg6_win32_compare_string_ex(const char16_t* locale_name,
                                    uint32_t cmp_flags,
                                    const char16_t* string1,
                                    int32_t cch_count1,
                                    const char16_t* string2,
                                    int32_t cch_count2,
                                    void* version_information,
                                    void* reserved,
                                    int64_t lparam) {
    (void)version_information;
    if (reserved != nullptr || lparam != 0) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    if (!locale_known(locale_name)) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    return pg6_win32_compare_string_w(0x0409, cmp_flags, string1, cch_count1, string2, cch_count2);
}

int32_t pg6_win32_get_locale_info_ex(const char16_t* locale_name,
                                     uint32_t lc_type,
                                     char16_t* lc_data,
                                     int32_t cch_data) {
    if (!locale_has_data(locale_name)) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    constexpr uint32_t kNoUserOverride = 0x80000000u;
    constexpr uint32_t kUseCpAcp = 0x40000000u;
    constexpr uint32_t kReturnGenitive = 0x10000000u;
    constexpr uint32_t kAllowNeutral = 0x08000000u;
    const bool return_number = (lc_type & win32::LOCALE_RETURN_NUMBER) != 0;
    const uint32_t type = lc_type & ~(kNoUserOverride | kUseCpAcp | win32::LOCALE_RETURN_NUMBER |
                                      kReturnGenitive | kAllowNeutral);
    std::u16string text;
    int32_t number = -1;
    if (!locale_item(type, &text, &number)) {
        win32::SetLastError(kErrorInvalidFlags);
        return 0;
    }
    if (return_number) {
        if (number < 0) {
            win32::SetLastError(kErrorInvalidFlags);  // not a numeric item
            return 0;
        }
        if (cch_data < 0 || (cch_data > 0 && (lc_data == nullptr || cch_data < 2))) {
            win32::SetLastError(cch_data < 0 || lc_data == nullptr ? win32::ERROR_INVALID_PARAMETER
                                                                   : kErrorInsufficientBuffer);
            return 0;
        }
        if (cch_data == 0) {
            win32::SetLastError(win32::ERROR_SUCCESS);
            return 2;  // size of a DWORD in characters
        }
        const uint32_t value = static_cast<uint32_t>(number);
        std::memcpy(lc_data, &value, sizeof(value));
        win32::SetLastError(win32::ERROR_SUCCESS);
        return 2;
    }
    return return_text(text, lc_data, cch_data);
}

int32_t pg6_win32_get_string_type_w(uint32_t info_type,
                                    const char16_t* src_str,
                                    int32_t cch_src,
                                    uint16_t* char_type) {
    if (src_str == nullptr || cch_src < -1 || char_type == nullptr) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    if (info_type != win32::CT_CTYPE1 && info_type != win32::CT_CTYPE2 && info_type != win32::CT_CTYPE3) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    const size_t n = cch_src == -1 ? u16_len(src_str) + 1 : static_cast<size_t>(cch_src);
    for (size_t i = 0; i < n; ++i) {
        const char16_t c = src_str[i];
        char_type[i] = info_type == win32::CT_CTYPE1   ? ctype1(c)
                       : info_type == win32::CT_CTYPE2 ? ctype2(c)
                                                       : ctype3(c);
    }
    win32::SetLastError(win32::ERROR_SUCCESS);
    return 1;
}

int32_t pg6_win32_get_cp_info(uint32_t code_page, Pg6CpInfo* cp_info) {
    if (cp_info == nullptr) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    *cp_info = Pg6CpInfo{};
    switch (code_page) {
        case win32::CP_ACP:
        case win32::CP_THREAD_ACP:
        case win32::CP_WINDOWS_1252:
        case win32::CP_LATIN1:
        case win32::CP_USASCII:
            cp_info->max_char_size = 1;
            break;
        case win32::CP_UTF8:
            cp_info->max_char_size = 4;
            break;
        default:
            win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
            return 0;
    }
    cp_info->default_char[0] = '?';
    win32::SetLastError(win32::ERROR_SUCCESS);
    return 1;
}

}  // extern "C"
