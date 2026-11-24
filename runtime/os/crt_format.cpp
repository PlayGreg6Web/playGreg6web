#include "os/crt_format.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "abi/playgta6_abi.h"
#include "guest/com_bridge.h"
#include "os/ucrt_bridge.h"

namespace playgta6::os::crt_format {

namespace {

// Longest string read from guest memory for one %s (bounds a missing terminator).
constexpr size_t kMaxGuestString = size_t{1} << 24;

// ---------------------------------------------------------------------------
// Specification
// ---------------------------------------------------------------------------

enum class Size : uint8_t { None, Char, Short, Long, LongLong, Int64, Ptr, LongDouble, Wide };

struct Spec {
    bool left = false;
    bool plus = false;
    bool space = false;
    bool alt = false;
    bool zero = false;
    int width = 0;
    int precision = -1;  // -1 = none
    Size size = Size::None;
    char conv = 0;
};

template <typename CharT>
class Formatter {
public:
    Formatter(std::basic_string_view<CharT> format, uint64_t options, ArgSource& args)
        : fmt_(format), options_(options), args_(args) {}

    Status run(std::basic_string<CharT>& out) {
        out_ = &out;
        size_t i = 0;
        while (i < fmt_.size()) {
            const CharT c = fmt_[i];
            if (c != CharT('%')) {
                out.push_back(c);
                ++i;
                continue;
            }
            ++i;
            Spec spec;
            const Status parsed = parse(i, spec);
            if (parsed != Status::Ok) {
                return parsed;
            }
            const Status done = convert(spec);
            if (done != Status::Ok) {
                return done;
            }
        }
        return Status::Ok;
    }

private:
    static constexpr bool kWideOutput = sizeof(CharT) == 2;

    bool next_int(int& value) {
        uint64_t slot = 0;
        if (!args_.next(slot)) {
            return false;
        }
        value = static_cast<int32_t>(static_cast<uint32_t>(slot));
        return true;
    }

    Status parse(size_t& i, Spec& spec) {
        auto at = [&](size_t k) -> uint32_t {
            return k < fmt_.size() ? static_cast<uint32_t>(fmt_[k]) : 0u;
        };
        for (;; ++i) {
            const uint32_t c = at(i);
            if (c == '-') {
                spec.left = true;
            } else if (c == '+') {
                spec.plus = true;
            } else if (c == ' ') {
                spec.space = true;
            } else if (c == '#') {
                spec.alt = true;
            } else if (c == '0') {
                spec.zero = true;
            } else {
                break;
            }
        }
        if (at(i) == '*') {
            ++i;
            int w = 0;
            if (!next_int(w)) {
                return Status::Unreadable;
            }
            if (w < 0) {
                spec.left = true;
                w = w == INT_MIN ? INT_MAX : -w;
            }
            spec.width = w;
        } else {
            while (at(i) >= '0' && at(i) <= '9') {
                if (spec.width > (INT_MAX - 9) / 10) {
                    return Status::InvalidFormat;
                }
                spec.width = spec.width * 10 + static_cast<int>(at(i) - '0');
                ++i;
            }
        }
        if (at(i) == '.') {
            ++i;
            if (at(i) == '*') {
                ++i;
                int p = 0;
                if (!next_int(p)) {
                    return Status::Unreadable;
                }
                spec.precision = p < 0 ? -1 : p;
            } else {
                int p = 0;
                while (at(i) >= '0' && at(i) <= '9') {
                    if (p > (INT_MAX - 9) / 10) {
                        return Status::InvalidFormat;
                    }
                    p = p * 10 + static_cast<int>(at(i) - '0');
                    ++i;
                }
                spec.precision = p;
            }
        }
        switch (at(i)) {
            case 'h':
                ++i;
                spec.size = Size::Short;
                if (at(i) == 'h') {
                    ++i;
                    spec.size = Size::Char;
                }
                break;
            case 'l':
                ++i;
                spec.size = Size::Long;
                if (at(i) == 'l') {
                    ++i;
                    spec.size = Size::LongLong;
                }
                break;
            case 'L':
                ++i;
                spec.size = Size::LongDouble;
                break;
            case 'w':
                ++i;
                spec.size = Size::Wide;
                break;
            case 'j':
                ++i;
                spec.size = Size::Int64;
                break;
            case 'z':
            case 't':
                ++i;
                spec.size = Size::Ptr;
                break;
            case 'I':
                ++i;
                if (at(i) == '6' && at(i + 1) == '4') {
                    i += 2;
                    spec.size = Size::Int64;
                } else if (at(i) == '3' && at(i + 1) == '2') {
                    i += 2;
                    spec.size = Size::Long;  // 32-bit
                } else {
                    spec.size = Size::Ptr;
                }
                break;
            default:
                break;
        }
        const uint32_t conv = at(i);
        if (conv == 0 || conv > 0x7F) {
            return Status::InvalidFormat;
        }
        spec.conv = static_cast<char>(conv);
        ++i;
        return Status::Ok;
    }

    // --- output helpers ---------------------------------------------------

    void put_ascii(std::string_view text) {
        for (char c : text) {
            out_->push_back(static_cast<CharT>(static_cast<unsigned char>(c)));
        }
    }

    void pad(int count, char fill) {
        for (int k = 0; k < count; ++k) {
            out_->push_back(static_cast<CharT>(fill));
        }
    }

    // `prefix` (sign, 0x) + zero padding + `body`, justified in the field.
    void emit_number(const Spec& spec, std::string_view prefix, std::string_view body, bool zero_pad) {
        const int len = static_cast<int>(prefix.size() + body.size());
        const int fill = spec.width > len ? spec.width - len : 0;
        if (spec.left) {
            put_ascii(prefix);
            put_ascii(body);
            pad(fill, ' ');
        } else if (zero_pad) {
            put_ascii(prefix);
            pad(fill, '0');
            put_ascii(body);
        } else {
            pad(fill, ' ');
            put_ascii(prefix);
            put_ascii(body);
        }
    }

    template <typename Text>
    void emit_text(const Spec& spec, const Text& text) {
        const int len = static_cast<int>(text.size());
        const int fill = spec.width > len ? spec.width - len : 0;
        if (!spec.left) {
            pad(fill, ' ');
        }
        for (auto u : text) {
            out_->push_back(static_cast<CharT>(u));
        }
        if (spec.left) {
            pad(fill, ' ');
        }
    }

    // --- conversions ------------------------------------------------------

    // Whether %s / %c of this specification takes a wide argument.
    bool wide_argument(const Spec& spec) const {
        if (spec.size == Size::Short) {
            return false;
        }
        if (spec.size == Size::Long || spec.size == Size::Wide) {
            return true;
        }
        const bool upper = spec.conv == 'S' || spec.conv == 'C';
        bool natural = false;  // the function's own width for %s / %c
        if constexpr (kWideOutput) {
            natural = (options_ & kLegacyWideSpecifiers) != 0;
        }
        return upper ? !natural : natural;
    }

    Status convert(const Spec& spec) {
        switch (spec.conv) {
            case '%':
                out_->push_back(CharT('%'));
                return Status::Ok;
            case 'd':
            case 'i':
                return integer(spec, true, 10, false);
            case 'u':
                return integer(spec, false, 10, false);
            case 'o':
                return integer(spec, false, 8, false);
            case 'x':
                return integer(spec, false, 16, false);
            case 'X':
                return integer(spec, false, 16, true);
            case 'p':
                return pointer(spec);
            case 'c':
            case 'C':
                return character(spec);
            case 's':
            case 'S':
                return string(spec);
            case 'f':
            case 'F':
            case 'e':
            case 'E':
            case 'g':
            case 'G':
            case 'a':
            case 'A':
                return floating(spec);
            default:  // %n (disabled in the UCRT), %Z and unknown conversions
                return Status::InvalidFormat;
        }
    }

    Status integer(const Spec& spec, bool is_signed, unsigned base, bool upper) {
        uint64_t raw = 0;
        if (!args_.next(raw)) {
            return Status::Unreadable;
        }
        bool negative = false;
        uint64_t magnitude = 0;
        switch (spec.size) {
            case Size::Char:
                magnitude = is_signed ? static_cast<uint64_t>(static_cast<int8_t>(raw))
                                      : static_cast<uint8_t>(raw);
                break;
            case Size::Short:
                magnitude = is_signed ? static_cast<uint64_t>(static_cast<int16_t>(raw))
                                      : static_cast<uint16_t>(raw);
                break;
            case Size::LongLong:
            case Size::Int64:
            case Size::Ptr:
                magnitude = raw;
                break;
            default:  // int and Win64 long are 32-bit
                magnitude = is_signed ? static_cast<uint64_t>(static_cast<int32_t>(raw))
                                      : static_cast<uint32_t>(raw);
                break;
        }
        if (is_signed && static_cast<int64_t>(magnitude) < 0) {
            negative = true;
            magnitude = 0 - magnitude;
        }
        std::string digits;
        const char* alphabet = upper ? "0123456789ABCDEF" : "0123456789abcdef";
        for (uint64_t v = magnitude; v != 0; v /= base) {
            digits.insert(digits.begin(), alphabet[v % base]);
        }
        if (spec.precision >= 0) {
            if (static_cast<int>(digits.size()) < spec.precision) {
                digits.insert(0, static_cast<size_t>(spec.precision) - digits.size(), '0');
            }
        } else if (digits.empty()) {
            digits = "0";
        }
        std::string prefix;
        if (is_signed) {
            if (negative) {
                prefix = "-";
            } else if (spec.plus) {
                prefix = "+";
            } else if (spec.space) {
                prefix = " ";
            }
        }
        if (spec.alt && magnitude != 0) {
            if (base == 16) {
                prefix += upper ? "0X" : "0x";
            } else if (base == 8 && (digits.empty() || digits[0] != '0')) {
                digits.insert(digits.begin(), '0');
            }
        }
        emit_number(spec, prefix, digits, spec.zero && !spec.left && spec.precision < 0);
        return Status::Ok;
    }

    Status pointer(const Spec& spec) {
        uint64_t raw = 0;
        if (!args_.next(raw)) {
            return Status::Unreadable;
        }
        char digits[17];
        std::snprintf(digits, sizeof(digits), "%016llX", static_cast<unsigned long long>(raw));
        emit_number(spec, spec.alt ? "0X" : "", digits, false);
        return Status::Ok;
    }

    Status character(const Spec& spec) {
        uint64_t raw = 0;
        if (!args_.next(raw)) {
            return Status::Unreadable;
        }
        std::u16string unit;
        if (wide_argument(spec)) {
            unit.push_back(static_cast<char16_t>(raw & 0xFFFF));
        } else {
            unit.push_back(static_cast<char16_t>(raw & 0xFF));
        }
        if constexpr (!kWideOutput) {
            if (unit[0] > 0xFF) {
                return Status::Encoding;
            }
        }
        Spec text = spec;
        text.precision = -1;
        emit_text(text, unit);
        return Status::Ok;
    }

    Status string(const Spec& spec) {
        uint64_t address = 0;
        if (!args_.next(address)) {
            return Status::Unreadable;
        }
        const size_t max = spec.precision >= 0 ? static_cast<size_t>(spec.precision) : kMaxGuestString;
        std::u16string units;
        if (wide_argument(spec)) {
            if (address == 0) {
                units = u"(null)";
            } else {
                std::u16string ws;
                if (!args_.wide(address, max, ws)) {
                    return Status::Unreadable;
                }
                units = std::move(ws);
            }
            if constexpr (!kWideOutput) {
                for (char16_t u : units) {
                    if (u > 0xFF) {
                        return Status::Encoding;
                    }
                }
            }
        } else {
            std::string s;
            if (address == 0) {
                s = "(null)";
            } else if (!args_.narrow(address, max, s)) {
                return Status::Unreadable;
            }
            for (char c : s) {
                units.push_back(static_cast<char16_t>(static_cast<unsigned char>(c)));
            }
        }
        if (spec.precision >= 0 && units.size() > static_cast<size_t>(spec.precision)) {
            units.resize(static_cast<size_t>(spec.precision));
        }
        emit_text(spec, units);
        return Status::Ok;
    }

    Status floating(const Spec& spec) {
        uint64_t raw = 0;
        if (!args_.next(raw)) {
            return Status::Unreadable;
        }
        double value;
        std::memcpy(&value, &raw, sizeof(value));
        const bool upper = spec.conv == 'F' || spec.conv == 'E' || spec.conv == 'G' || spec.conv == 'A';
        const bool negative = (raw >> 63) != 0;
        std::string prefix;
        if (negative) {
            prefix = "-";
        } else if (spec.plus) {
            prefix = "+";
        } else if (spec.space) {
            prefix = " ";
        }
        if (std::isnan(value) || std::isinf(value)) {
            std::string body;
            if (std::isinf(value)) {
                body = "inf";
            } else {
                const uint64_t payload = raw & 0x000FFFFFFFFFFFFFull;
                const bool quiet = (payload & 0x0008000000000000ull) != 0;
                if (!quiet) {
                    body = "nan(snan)";
                } else if (negative && payload == 0x0008000000000000ull) {
                    body = "nan(ind)";  // the default NaN of invalid operations
                } else {
                    body = "nan";
                }
            }
            if (upper) {
                for (char& c : body) {
                    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                }
            }
            emit_number(spec, prefix, body, false);
            return Status::Ok;
        }
        std::string body;
        if (spec.conv == 'a' || spec.conv == 'A') {
            body = hex_float(std::fabs(value), spec, upper);
        } else {
            // Finite values: C99 formatting of the magnitude (MSVC's UCRT is correctly rounded
            // like the host libc); sign and padding are applied here.
            std::string fmt = "%";
            if (spec.alt) {
                fmt += '#';
            }
            fmt += '.';
            fmt += std::to_string(spec.precision >= 0 ? spec.precision : 6);
            fmt += spec.conv;
            const double magnitude = std::fabs(value);
            const int n = std::snprintf(nullptr, 0, fmt.c_str(), magnitude);
            if (n < 0) {
                return Status::InvalidFormat;
            }
            body.resize(static_cast<size_t>(n) + 1);
            std::snprintf(body.data(), body.size(), fmt.c_str(), magnitude);
            body.resize(static_cast<size_t>(n));
            if ((options_ & kLegacyThreeDigitExponents) != 0) {
                widen_exponent(body);
            }
        }
        if (spec.conv == 'a' || spec.conv == 'A') {
            prefix += upper ? "0X" : "0x";
        }
        emit_number(spec, prefix, body, spec.zero && !spec.left);
        return Status::Ok;
    }

    static void widen_exponent(std::string& body) {
        const size_t e = body.find_first_of("eE");
        if (e == std::string::npos || e + 2 > body.size()) {
            return;
        }
        const size_t digits = body.size() - (e + 2);
        if (digits < 3) {
            body.insert(e + 2, 3 - digits, '0');
        }
    }

    // %a of a non-negative finite value, without the 0x prefix: MSVC prints 13 hex digits
    // when no precision is given and keeps subnormals as 0x0.<fraction>p-1022.
    static std::string hex_float(double value, const Spec& spec, bool upper) {
        uint64_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        const int biased = static_cast<int>((bits >> 52) & 0x7FF);
        uint64_t mantissa = bits & 0x000FFFFFFFFFFFFFull;
        int lead = biased == 0 ? 0 : 1;
        int exponent = biased == 0 ? (mantissa == 0 ? 0 : -1022) : biased - 1023;
        int precision = spec.precision >= 0 ? spec.precision : 13;
        if (precision < 13) {
            const int drop = (13 - precision) * 4;
            const uint64_t half = uint64_t{1} << (drop - 1);
            const uint64_t rest = mantissa & ((uint64_t{1} << drop) - 1);
            mantissa >>= drop;
            // Ties go to even; with no fraction digits left the leading digit is the last one.
            const bool odd = precision == 0 ? (lead & 1) != 0 : (mantissa & 1) != 0;
            if (rest > half || (rest == half && odd)) {
                ++mantissa;
                if (mantissa >> (precision * 4) != 0) {  // carried into the leading digit
                    mantissa &= (precision == 0) ? 0 : ((uint64_t{1} << (precision * 4)) - 1);
                    ++lead;
                }
            }
        }
        const char* alphabet = upper ? "0123456789ABCDEF" : "0123456789abcdef";
        std::string out(1, alphabet[lead]);
        if (precision > 0 || spec.alt) {
            out += '.';
        }
        const int shown = precision < 13 ? precision : 13;
        for (int d = shown - 1; d >= 0; --d) {
            out += alphabet[(mantissa >> (d * 4)) & 0xF];
        }
        for (int d = 13; d < precision; ++d) {
            out += '0';
        }
        out += upper ? 'P' : 'p';
        out += exponent < 0 ? '-' : '+';
        out += std::to_string(exponent < 0 ? -exponent : exponent);
        return out;
    }

    std::basic_string_view<CharT> fmt_;
    uint64_t options_;
    ArgSource& args_;
    std::basic_string<CharT>* out_ = nullptr;
};

// ---------------------------------------------------------------------------
// Guest memory
// ---------------------------------------------------------------------------

template <typename Unit, typename Out>
bool read_guest_units(uint64_t address, size_t max, Out& out) {
    namespace gcom = playgta6::guest::com;
    out.clear();
    uint64_t va = address;
    while (out.size() < max) {
        // Read up to the end of the current 4 KiB page at a time.
        const uint64_t page_left = 0x1000 - (va & 0xFFF);
        size_t units = static_cast<size_t>(page_left / sizeof(Unit));
        if (units == 0) {
            units = 1;
        }
        units = std::min(units, max - out.size());
        const auto* host = static_cast<const Unit*>(gcom::guest_to_host(va, units * sizeof(Unit)));
        if (host == nullptr) {
            return false;
        }
        for (size_t k = 0; k < units; ++k) {
            if (host[k] == 0) {
                return true;
            }
            out.push_back(static_cast<typename Out::value_type>(host[k]));
        }
        va += units * sizeof(Unit);
    }
    return true;
}

}  // namespace

bool GuestArgSource::next(uint64_t& slot) {
    if (!playgta6::guest::com::read_guest(next_, slot)) {
        return false;
    }
    next_ += 8;
    return true;
}

bool GuestArgSource::narrow(uint64_t address, size_t max, std::string& out) {
    return read_guest_units<uint8_t>(address, max, out);
}

bool GuestArgSource::wide(uint64_t address, size_t max, std::u16string& out) {
    return read_guest_units<uint16_t>(address, max, out);
}

Status format_narrow(std::string_view format, uint64_t options, ArgSource& args, std::string& out) {
    out.clear();
    return Formatter<char>(format, options, args).run(out);
}

Status format_wide(std::u16string_view format, uint64_t options, ArgSource& args, std::u16string& out) {
    out.clear();
    return Formatter<char16_t>(format, options, args).run(out);
}

Store sprintf_store(size_t length, bool has_buffer, size_t count, uint64_t options) {
    Store s;
    if (length > static_cast<size_t>(INT32_MAX)) {
        s.result = -1;
        s.error = ERANGE;
        return s;
    }
    const auto n = static_cast<int32_t>(length);
    if (!has_buffer) {
        s.result = count == 0 ? n : -1;  // measuring; a null buffer with a count is invalid
        s.error = count == 0 ? 0 : EINVAL;
        return s;
    }
    if ((options & kStandardSnprintfBehavior) != 0) {
        if (count == 0) {
            s.result = n;  // nothing written
            return s;
        }
        s.units = std::min(length, count - 1);
        s.terminate = true;
        s.result = n;
        return s;
    }
    if (length < count) {
        s.units = length;
        s.terminate = true;
        s.result = n;
    } else if (length == count) {
        s.units = length;
        s.result = n;
    } else {
        s.units = count;
        s.result = -1;
    }
    return s;
}

Store sprintf_s_store(size_t length, bool has_buffer, size_t count) {
    Store s;
    if (!has_buffer || count == 0) {
        s.result = -1;
        s.error = EINVAL;
        return s;
    }
    if (length >= count || length > static_cast<size_t>(INT32_MAX)) {
        s.result = -1;
        s.clear_first = true;
        s.error = ERANGE;
        return s;
    }
    s.units = length;
    s.terminate = true;
    s.result = static_cast<int32_t>(length);
    return s;
}

Store snprintf_s_store(size_t length, bool has_buffer, size_t count, size_t max_count) {
    Store s;
    if (!has_buffer || count == 0) {
        s.result = -1;
        s.error = EINVAL;
        return s;
    }
    if (max_count == kTruncate) {
        if (length < count) {
            s.units = length;
            s.terminate = true;
            s.result = static_cast<int32_t>(length);
        } else {
            s.units = count - 1;
            s.terminate = true;
            s.result = -1;
        }
        return s;
    }
    if (length <= max_count && length < count) {
        s.units = length;
        s.terminate = true;
        s.result = static_cast<int32_t>(length);
        return s;
    }
    if (max_count < length && max_count < count) {
        s.units = max_count;
        s.terminate = true;
        s.result = -1;
        return s;
    }
    s.result = -1;
    s.clear_first = true;
    s.error = ERANGE;
    return s;
}

}  // namespace playgta6::os::crt_format

// ---------------------------------------------------------------------------
// Guest-state thunks
// ---------------------------------------------------------------------------

namespace {

namespace cf = playgta6::os::crt_format;
namespace gcom = playgta6::guest::com;

void set_errno(int32_t value) {
    *pg6_crt_errno() = value;
}

template <typename Unit>
bool read_format(uint64_t va, std::basic_string<Unit>& out) {
    if (va == 0) {
        return false;
    }
    cf::GuestArgSource reader(0);
    if constexpr (sizeof(Unit) == 1) {
        return reader.narrow(va, size_t{1} << 24, out);
    } else {
        return reader.wide(va, size_t{1} << 24, out);
    }
}

// Formats into `text`; on failure sets errno and returns false.
template <typename Unit>
bool format_from_guest(uint64_t options, uint64_t format_va, uint64_t va_list, std::basic_string<Unit>& text) {
    std::basic_string<Unit> format;
    if (!read_format(format_va, format)) {
        set_errno(EINVAL);
        return false;
    }
    cf::GuestArgSource args(va_list);
    cf::Status status;
    if constexpr (sizeof(Unit) == 1) {
        status = cf::format_narrow(format, options, args, text);
    } else {
        status = cf::format_wide(format, options, args, text);
    }
    if (status != cf::Status::Ok) {
        set_errno(status == cf::Status::Encoding ? EILSEQ : EINVAL);
        return false;
    }
    return true;
}

template <typename Unit>
void clear_buffer(uint64_t buffer, size_t count) {
    if (buffer != 0 && count > 0) {
        const Unit zero = 0;
        gcom::write_guest(buffer, &zero, sizeof(zero));
    }
}

// Applies `store` for `text` to the guest buffer; returns the result (or -1 if the buffer is not
// writable guest memory).
template <typename Unit>
int32_t apply_store(const cf::Store& store, uint64_t buffer, const std::basic_string<Unit>& text) {
    if (store.error != 0) {
        set_errno(store.error);
    }
    if (store.clear_first) {
        const Unit zero = 0;
        gcom::write_guest(buffer, &zero, sizeof(zero));
        return store.result;
    }
    if (store.units > 0 && !gcom::write_guest(buffer, text.data(), store.units * sizeof(Unit))) {
        set_errno(EINVAL);
        return -1;
    }
    if (store.terminate) {
        const Unit zero = 0;
        if (!gcom::write_guest(buffer + store.units * sizeof(Unit), &zero, sizeof(zero))) {
            set_errno(EINVAL);
            return -1;
        }
    }
    return store.result;
}

template <typename Unit>
int32_t vsprintf_impl(uint64_t options, uint64_t buffer, uint64_t count, uint64_t format, uint64_t /*locale*/,
                      uint64_t va_list) {
    std::basic_string<Unit> text;
    if (!format_from_guest<Unit>(options, format, va_list, text)) {
        clear_buffer<Unit>(buffer, count);
        return -1;
    }
    return apply_store(cf::sprintf_store(text.size(), buffer != 0, count, options), buffer, text);
}

template <typename Unit>
int32_t vsprintf_s_impl(uint64_t options, uint64_t buffer, uint64_t count, uint64_t format, uint64_t /*locale*/,
                        uint64_t va_list) {
    if (buffer == 0 || count == 0) {
        set_errno(EINVAL);
        return -1;
    }
    std::basic_string<Unit> text;
    if (!format_from_guest<Unit>(options, format, va_list, text)) {
        clear_buffer<Unit>(buffer, count);
        return -1;
    }
    return apply_store(cf::sprintf_s_store(text.size(), true, count), buffer, text);
}

int32_t vsprintf_narrow(uint64_t o, uint64_t b, uint64_t c, uint64_t f, uint64_t l, uint64_t v) {
    return vsprintf_impl<char>(o, b, c, f, l, v);
}
int32_t vsprintf_s_narrow(uint64_t o, uint64_t b, uint64_t c, uint64_t f, uint64_t l, uint64_t v) {
    return vsprintf_s_impl<char>(o, b, c, f, l, v);
}
int32_t vswprintf_wide(uint64_t o, uint64_t b, uint64_t c, uint64_t f, uint64_t l, uint64_t v) {
    return vsprintf_impl<char16_t>(o, b, c, f, l, v);
}
int32_t vswprintf_s_wide(uint64_t o, uint64_t b, uint64_t c, uint64_t f, uint64_t l, uint64_t v) {
    return vsprintf_s_impl<char16_t>(o, b, c, f, l, v);
}

int32_t vsnwprintf_s_wide(uint64_t options, uint64_t buffer, uint64_t count, uint64_t max_count,
                          uint64_t format, uint64_t /*locale*/, uint64_t va_list) {
    if (buffer == 0 || count == 0) {
        set_errno(EINVAL);
        return -1;
    }
    std::u16string text;
    if (!format_from_guest<char16_t>(options, format, va_list, text)) {
        clear_buffer<char16_t>(buffer, count);
        return -1;
    }
    return apply_store(cf::snprintf_s_store(text.size(), true, count, max_count), buffer, text);
}

// The FILE* the guest holds is the guest-address form of a host stream handed out by
// __acrt_iob_func: only stdout and stderr are accepted (nothing else is ever handed out for writing).
FILE* output_stream(uint64_t stream_va) {
    if (stream_va == 0) {
        return nullptr;
    }
    auto* stream = reinterpret_cast<FILE*>(static_cast<uintptr_t>(__playgta6_guest_base + stream_va));
    return (stream == stdout || stream == stderr) ? stream : nullptr;
}

int32_t vfprintf_narrow(uint64_t options, uint64_t stream_va, uint64_t format, uint64_t /*locale*/,
                        uint64_t va_list) {
    FILE* stream = output_stream(stream_va);
    if (stream == nullptr) {
        set_errno(EINVAL);
        return -1;
    }
    std::string text;
    if (!format_from_guest<char>(options, format, va_list, text)) {
        return -1;
    }
    if (text.size() > static_cast<size_t>(INT32_MAX) ||
        std::fwrite(text.data(), 1, text.size(), stream) != text.size()) {
        return -1;
    }
    return static_cast<int32_t>(text.size());
}

int32_t vfwprintf_wide(uint64_t options, uint64_t stream_va, uint64_t format, uint64_t /*locale*/,
                       uint64_t va_list) {
    FILE* stream = output_stream(stream_va);
    if (stream == nullptr) {
        set_errno(EINVAL);
        return -1;
    }
    std::u16string text;
    if (!format_from_guest<char16_t>(options, format, va_list, text)) {
        return -1;
    }
    // Written as UTF-8 (the host console's encoding).
    std::string utf8;
    for (size_t i = 0; i < text.size(); ++i) {
        uint32_t cp = text[i];
        if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < text.size() && text[i + 1] >= 0xDC00 &&
            text[i + 1] <= 0xDFFF) {
            cp = 0x10000 + ((cp - 0xD800) << 10) + (text[i + 1] - 0xDC00);
            ++i;
        }
        if (cp < 0x80) {
            utf8 += static_cast<char>(cp);
        } else if (cp < 0x800) {
            utf8 += static_cast<char>(0xC0 | (cp >> 6));
            utf8 += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            utf8 += static_cast<char>(0xE0 | (cp >> 12));
            utf8 += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            utf8 += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            utf8 += static_cast<char>(0xF0 | (cp >> 18));
            utf8 += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            utf8 += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            utf8 += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }
    if (text.size() > static_cast<size_t>(INT32_MAX) ||
        std::fwrite(utf8.data(), 1, utf8.size(), stream) != utf8.size()) {
        return -1;
    }
    return static_cast<int32_t>(text.size());
}

}  // namespace

extern "C" {

void pg6_gthunk_stdio_common_vsprintf(void* state) {
    gcom::function_thunk<&vsprintf_narrow>()(state);
}

void pg6_gthunk_stdio_common_vsprintf_s(void* state) {
    gcom::function_thunk<&vsprintf_s_narrow>()(state);
}

void pg6_gthunk_stdio_common_vswprintf(void* state) {
    gcom::function_thunk<&vswprintf_wide>()(state);
}

void pg6_gthunk_stdio_common_vswprintf_s(void* state) {
    gcom::function_thunk<&vswprintf_s_wide>()(state);
}

void pg6_gthunk_stdio_common_vsnwprintf_s(void* state) {
    gcom::function_thunk<&vsnwprintf_s_wide>()(state);
}

void pg6_gthunk_stdio_common_vfprintf(void* state) {
    gcom::function_thunk<&vfprintf_narrow>()(state);
}

void pg6_gthunk_stdio_common_vfwprintf(void* state) {
    gcom::function_thunk<&vfwprintf_wide>()(state);
}

}  // extern "C"
