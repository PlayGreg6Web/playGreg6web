#include "os/crt_scan.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "abi/playgta6_abi.h"
#include "guest/com_bridge.h"
#include "os/crt_format.h"
#include "os/ucrt_bridge.h"

namespace playgta6::os::crt_scan {

namespace {

enum class Size : uint8_t { None, Char, Short, Long, LongLong, Ptr, LongDouble, Wide, Int32 };

template <typename C>
bool is_space(C c) {
    return c == C(' ') || c == C('\t') || c == C('\n') || c == C('\v') || c == C('\f') || c == C('\r');
}

template <typename C>
int digit_value(C c) {
    if (c >= C('0') && c <= C('9')) {
        return static_cast<int>(c - C('0'));
    }
    if (c >= C('a') && c <= C('z')) {
        return static_cast<int>(c - C('a')) + 10;
    }
    if (c >= C('A') && c <= C('Z')) {
        return static_cast<int>(c - C('A')) + 10;
    }
    return 99;
}

template <typename C>
C lower(C c) {
    return (c >= C('A') && c <= C('Z')) ? C(c - C('A') + C('a')) : c;
}

template <typename C>
class Scanner {
public:
    Scanner(std::basic_string_view<C> input, std::basic_string_view<C> format, uint64_t options, Target& target)
        : in_(input), fmt_(format), options_(options), target_(target) {}

    Result run() {
        size_t i = 0;
        while (i < fmt_.size()) {
            const C c = fmt_[i];
            if (is_space(c)) {
                while (i < fmt_.size() && is_space(fmt_[i])) {
                    ++i;
                }
                skip_space();
                continue;
            }
            if (c != C('%')) {
                if (pos_ >= in_.size()) {
                    return input_failure();
                }
                if (in_[pos_] != c) {
                    return done();
                }
                ++pos_;
                ++i;
                continue;
            }
            ++i;
            if (i >= fmt_.size()) {
                return invalid();
            }
            if (fmt_[i] == C('%')) {
                ++i;
                skip_space();
                if (pos_ >= in_.size()) {
                    return input_failure();
                }
                if (in_[pos_] != C('%')) {
                    return done();
                }
                ++pos_;
                continue;
            }
            bool suppress = false;
            if (fmt_[i] == C('*')) {
                suppress = true;
                ++i;
            }
            size_t width = 0;
            while (i < fmt_.size() && fmt_[i] >= C('0') && fmt_[i] <= C('9')) {
                width = width * 10 + static_cast<size_t>(fmt_[i] - C('0'));
                if (width > (size_t{1} << 30)) {
                    return invalid();
                }
                ++i;
            }
            Size size = Size::None;
            if (!parse_size(i, size) || i >= fmt_.size()) {
                return invalid();
            }
            const C conv = fmt_[i++];
            Step step = Step::Continue;
            switch (static_cast<char32_t>(conv)) {
                case U'd':
                case U'i':
                case U'u':
                case U'o':
                case U'x':
                case U'X':
                case U'p':
                    step = scan_integer(static_cast<char>(conv), width, size, suppress);
                    break;
                case U'e':
                case U'E':
                case U'f':
                case U'F':
                case U'g':
                case U'G':
                case U'a':
                case U'A':
                    step = scan_float(width, size, suppress);
                    break;
                case U's':
                case U'S':
                case U'c':
                case U'C':
                    step = scan_chars(static_cast<char>(conv), width, size, suppress);
                    break;
                case U'[': {
                    std::vector<bool> set;
                    if (!parse_set(i, set)) {
                        return invalid();
                    }
                    step = scan_set(set, width, size, suppress);
                    break;
                }
                case U'n':
                    if (!suppress && !store_integer(pos_, size == Size::None ? Size::Int32 : size)) {
                        return unwritable();
                    }
                    break;
                default:
                    return invalid();
            }
            switch (step) {
                case Step::Continue:
                    break;
                case Step::InputFailure:
                    return input_failure();
                case Step::MatchingFailure:
                    return done();
                case Step::Unwritable:
                    return unwritable();
                case Step::Stop:
                    return done();
            }
        }
        return done();
    }

private:
    enum class Step : uint8_t { Continue, InputFailure, MatchingFailure, Unwritable, Stop };

    Result done() const { return Result{assigned_, Status::Ok}; }
    // The input ended: EOF unless a conversion already completed.
    Result input_failure() const { return Result{conversions_ == 0 ? -1 : assigned_, Status::Ok}; }
    static Result invalid() { return Result{-1, Status::InvalidFormat}; }
    static Result unwritable() { return Result{-1, Status::Unwritable}; }

    void skip_space() {
        while (pos_ < in_.size() && is_space(in_[pos_])) {
            ++pos_;
        }
    }

    bool parse_size(size_t& i, Size& size) const {
        auto at = [&](size_t k) { return k < fmt_.size() ? fmt_[k] : C(0); };
        const C c = at(i);
        if (c == C('h')) {
            if (at(i + 1) == C('h')) {
                size = Size::Char;
                i += 2;
            } else {
                size = Size::Short;
                ++i;
            }
        } else if (c == C('l')) {
            if (at(i + 1) == C('l')) {
                size = Size::LongLong;
                i += 2;
            } else {
                size = Size::Long;
                ++i;
            }
        } else if (c == C('L')) {
            size = Size::LongDouble;
            ++i;
        } else if (c == C('j') || c == C('z') || c == C('t')) {
            size = Size::Ptr;
            ++i;
        } else if (c == C('w')) {
            size = Size::Wide;
            ++i;
        } else if (c == C('I')) {
            if (at(i + 1) == C('6') && at(i + 2) == C('4')) {
                size = Size::LongLong;
                i += 3;
            } else if (at(i + 1) == C('3') && at(i + 2) == C('2')) {
                size = Size::Int32;
                i += 3;
            } else {
                size = Size::Ptr;
                ++i;
            }
        }
        return true;
    }

    // %[...]: `i` is just past '['; on success just past the closing ']'.
    bool parse_set(size_t& i, std::vector<bool>& set) const {
        set.assign(0x10000, false);
        bool negate = false;
        if (i < fmt_.size() && fmt_[i] == C('^')) {
            negate = true;
            ++i;
        }
        const size_t start = i;
        bool closed = false;
        while (i < fmt_.size()) {
            const C c = fmt_[i];
            if (c == C(']') && i != start) {
                closed = true;
                ++i;
                break;
            }
            if (i + 2 < fmt_.size() && fmt_[i + 1] == C('-') && fmt_[i + 2] != C(']')) {
                auto lo = static_cast<uint32_t>(static_cast<std::make_unsigned_t<C>>(c));
                auto hi = static_cast<uint32_t>(static_cast<std::make_unsigned_t<C>>(fmt_[i + 2]));
                if (lo > hi) {
                    std::swap(lo, hi);
                }
                for (uint32_t k = lo; k <= hi; ++k) {
                    set[k] = true;
                }
                i += 3;
                continue;
            }
            set[static_cast<std::make_unsigned_t<C>>(c)] = true;
            ++i;
        }
        if (!closed) {
            return false;
        }
        if (negate) {
            set.flip();
        }
        return true;
    }

    bool next_slot(uint64_t& slot) { return target_.next(slot); }

    bool store_integer(uint64_t value, Size size) {
        uint64_t address = 0;
        if (!next_slot(address) || address == 0) {
            return false;
        }
        switch (size) {
            case Size::Char: {
                const auto v = static_cast<uint8_t>(value);
                return target_.store(address, &v, 1);
            }
            case Size::Short: {
                const auto v = static_cast<uint16_t>(value);
                return target_.store(address, &v, 2);
            }
            case Size::LongLong:
            case Size::Ptr: {
                return target_.store(address, &value, 8);
            }
            default: {
                const auto v = static_cast<uint32_t>(value);
                return target_.store(address, &v, 4);
            }
        }
    }

    Step scan_integer(char conv, size_t width, Size size, bool suppress) {
        skip_space();
        if (pos_ >= in_.size()) {
            return Step::InputFailure;
        }
        const size_t limit = width == 0 ? in_.size() : std::min(in_.size(), pos_ + width);
        size_t p = pos_;
        bool negative = false;
        if (p < limit && (in_[p] == C('+') || in_[p] == C('-'))) {
            negative = in_[p] == C('-');
            ++p;
        }
        int base = 10;
        switch (conv) {
            case 'o':
                base = 8;
                break;
            case 'x':
            case 'X':
            case 'p':
                base = 16;
                break;
            case 'i':
                base = 0;
                break;
            default:
                break;
        }
        bool digits = false;
        if ((base == 0 || base == 16) && p < limit && in_[p] == C('0')) {
            if (p + 1 < limit && lower(in_[p + 1]) == C('x')) {
                p += 2;
                base = 16;
                digits = true;  // "0x" alone reads as 0
            } else if (base == 0) {
                base = 8;
            }
        }
        if (base == 0) {
            base = 10;
        }
        uint64_t value = 0;
        while (p < limit && digit_value(in_[p]) < base) {
            value = value * static_cast<uint64_t>(base) + static_cast<uint64_t>(digit_value(in_[p]));
            digits = true;
            ++p;
        }
        if (!digits) {
            return Step::MatchingFailure;
        }
        pos_ = p;
        ++conversions_;
        if (negative) {
            value = 0 - value;
        }
        if (!suppress) {
            if (!store_integer(value, conv == 'p' ? Size::Ptr : size)) {
                return Step::Unwritable;
            }
            ++assigned_;
        }
        return Step::Continue;
    }

    // Longest prefix of the input (within `limit`) that is a float, by the strtod grammar.
    size_t float_extent(size_t p, size_t limit) const {
        auto at = [&](size_t k) { return k < limit ? lower(in_[k]) : C(0); };
        auto match = [&](size_t k, const char* word) {
            for (size_t n = 0; word[n] != 0; ++n) {
                if (at(k + n) != C(word[n])) {
                    return false;
                }
            }
            return true;
        };
        const size_t start = p;
        if (at(p) == C('+') || at(p) == C('-')) {
            ++p;
        }
        if (match(p, "infinity")) {
            return p + 8 - start;
        }
        if (match(p, "inf")) {
            return p + 3 - start;
        }
        if (match(p, "nan")) {
            p += 3;
            if (at(p) == C('(')) {
                size_t q = p + 1;
                while (q < limit && (digit_value(in_[q]) < 36 || in_[q] == C('_'))) {
                    ++q;
                }
                if (at(q) == C(')')) {
                    p = q + 1;
                }
            }
            return p - start;
        }
        int base = 10;
        const size_t number = p;
        if (at(p) == C('0') && at(p + 1) == C('x')) {
            base = 16;
            p += 2;
        }
        bool digits = false;
        while (digit_value(at(p)) < base) {
            ++p;
            digits = true;
        }
        if (at(p) == C('.')) {
            ++p;
            while (digit_value(at(p)) < base) {
                ++p;
                digits = true;
            }
        }
        if (!digits) {
            // "0x" without hex digits reads as the "0" (as strtod does); nothing else is a number.
            return base == 16 ? number + 1 - start : 0;
        }
        const C exp = base == 16 ? C('p') : C('e');
        if (at(p) == exp) {
            size_t q = p + 1;
            if (at(q) == C('+') || at(q) == C('-')) {
                ++q;
            }
            if (digit_value(at(q)) < 10) {
                while (digit_value(at(q)) < 10) {
                    ++q;
                }
                p = q;
            }
        }
        return p - start;
    }

    Step scan_float(size_t width, Size size, bool suppress) {
        skip_space();
        if (pos_ >= in_.size()) {
            return Step::InputFailure;
        }
        const size_t limit = width == 0 ? in_.size() : std::min(in_.size(), pos_ + width);
        const size_t len = float_extent(pos_, limit);
        if (len == 0) {
            return Step::MatchingFailure;
        }
        std::string text;
        text.reserve(len);
        for (size_t k = 0; k < len; ++k) {
            text.push_back(static_cast<char>(in_[pos_ + k]));
        }
        const double value = std::strtod(text.c_str(), nullptr);
        pos_ += len;
        ++conversions_;
        if (!suppress) {
            uint64_t address = 0;
            if (!next_slot(address) || address == 0) {
                return Step::Unwritable;
            }
            bool ok = false;
            if (size == Size::Long || size == Size::LongDouble || size == Size::LongLong) {
                ok = target_.store(address, &value, sizeof(value));
            } else {
                const auto f = static_cast<float>(value);
                ok = target_.store(address, &f, sizeof(f));
            }
            if (!ok) {
                return Step::Unwritable;
            }
            ++assigned_;
        }
        return Step::Continue;
    }

    // Whether a string conversion stores wide characters.
    bool wide_target(char conv, Size size) const {
        const bool upper = conv == 'S' || conv == 'C';
        if (size == Size::Short || size == Size::Char) {
            return false;
        }
        if (size == Size::Long || size == Size::Wide) {
            return true;
        }
        if constexpr (sizeof(C) == 1) {
            return upper;
        } else {
            const bool lower_is_wide = (options_ & kLegacyWideSpecifiers) != 0;
            return upper ? !lower_is_wide : lower_is_wide;
        }
    }

    // Stores `units` (plus a terminator if `terminate`) at the next pointer slot.
    Step store_string(const std::basic_string<C>& units, bool wide, bool terminate) {
        uint64_t address = 0;
        if (!next_slot(address) || address == 0) {
            return Step::Unwritable;
        }
        const size_t needed = units.size() + (terminate ? 1 : 0);
        if ((options_ & kSecureCrt) != 0) {
            uint64_t capacity = 0;
            if (!next_slot(capacity)) {
                return Step::Unwritable;
            }
            capacity &= 0xFFFFFFFFu;
            if (needed > capacity) {
                if (capacity > 0) {
                    const uint16_t zero = 0;
                    target_.store(address, &zero, wide ? 2 : 1);
                }
                return Step::Stop;
            }
        }
        if (wide) {
            std::u16string out;
            out.reserve(needed);
            for (const C c : units) {
                out.push_back(static_cast<char16_t>(static_cast<std::make_unsigned_t<C>>(c)));
            }
            if (terminate) {
                out.push_back(0);
            }
            return target_.store(address, out.data(), out.size() * 2) ? Step::Continue : Step::Unwritable;
        }
        std::string out;
        out.reserve(needed);
        for (const C c : units) {
            const auto unit = static_cast<uint32_t>(static_cast<std::make_unsigned_t<C>>(c));
            if (unit > 0xFF) {
                return Step::MatchingFailure;  // not representable in the "C" locale
            }
            out.push_back(static_cast<char>(unit));
        }
        if (terminate) {
            out.push_back(0);
        }
        return target_.store(address, out.data(), out.size()) ? Step::Continue : Step::Unwritable;
    }

    Step finish_string(const std::basic_string<C>& units, bool wide, bool terminate, bool suppress) {
        ++conversions_;
        if (suppress) {
            return Step::Continue;
        }
        const Step step = store_string(units, wide, terminate);
        if (step == Step::Continue) {
            ++assigned_;
        }
        return step;
    }

    Step scan_chars(char conv, size_t width, Size size, bool suppress) {
        const bool is_c = conv == 'c' || conv == 'C';
        if (!is_c) {
            skip_space();
        }
        if (pos_ >= in_.size()) {
            return Step::InputFailure;
        }
        std::basic_string<C> units;
        if (is_c) {
            const size_t n = width == 0 ? 1 : width;
            if (in_.size() - pos_ < n) {
                return Step::InputFailure;
            }
            units.assign(in_.substr(pos_, n));
        } else {
            const size_t limit = width == 0 ? in_.size() : std::min(in_.size(), pos_ + width);
            size_t p = pos_;
            while (p < limit && !is_space(in_[p])) {
                ++p;
            }
            units.assign(in_.substr(pos_, p - pos_));
        }
        pos_ += units.size();
        return finish_string(units, wide_target(conv, size), !is_c, suppress);
    }

    Step scan_set(const std::vector<bool>& set, size_t width, Size size, bool suppress) {
        if (pos_ >= in_.size()) {
            return Step::InputFailure;
        }
        const size_t limit = width == 0 ? in_.size() : std::min(in_.size(), pos_ + width);
        size_t p = pos_;
        while (p < limit && set[static_cast<std::make_unsigned_t<C>>(in_[p])]) {
            ++p;
        }
        if (p == pos_) {
            return Step::MatchingFailure;
        }
        std::basic_string<C> units(in_.substr(pos_, p - pos_));
        pos_ = p;
        return finish_string(units, wide_target('[', size), true, suppress);
    }

    std::basic_string_view<C> in_;
    std::basic_string_view<C> fmt_;
    uint64_t options_;
    Target& target_;
    size_t pos_ = 0;
    int32_t assigned_ = 0;
    int32_t conversions_ = 0;
};

}  // namespace

bool GuestTarget::next(uint64_t& slot) {
    if (!playgta6::guest::com::read_guest(next_, slot)) {
        return false;
    }
    next_ += 8;
    return true;
}

bool GuestTarget::store(uint64_t address, const void* data, size_t size) {
    return playgta6::guest::com::write_guest(address, data, size);
}

Result scan_narrow(std::string_view input, std::string_view format, uint64_t options, Target& target) {
    return Scanner<char>(input, format, options, target).run();
}

Result scan_wide(std::u16string_view input, std::u16string_view format, uint64_t options, Target& target) {
    return Scanner<char16_t>(input, format, options, target).run();
}

}  // namespace playgta6::os::crt_scan

namespace {

namespace cs = playgta6::os::crt_scan;
namespace gcom = playgta6::guest::com;

constexpr size_t kMaxGuestText = size_t{1} << 24;

template <typename Unit>
bool read_text(uint64_t va, size_t max, std::basic_string<Unit>& out) {
    playgta6::os::crt_format::GuestArgSource reader(0);
    if constexpr (sizeof(Unit) == 1) {
        return reader.narrow(va, max, out);
    } else {
        return reader.wide(va, max, out);
    }
}

template <typename Unit>
int32_t vsscanf_impl(uint64_t options, uint64_t buffer, uint64_t count, uint64_t format, uint64_t /*locale*/,
                     uint64_t va_list) {
    std::basic_string<Unit> input;
    std::basic_string<Unit> fmt;
    const size_t max = count == UINT64_MAX ? kMaxGuestText : static_cast<size_t>(std::min<uint64_t>(count, kMaxGuestText));
    if (buffer == 0 || format == 0 || !read_text<Unit>(buffer, max, input) ||
        !read_text<Unit>(format, kMaxGuestText, fmt)) {
        *pg6_crt_errno() = PG6_CRT_EINVAL;
        return -1;
    }
    cs::GuestTarget target(va_list);
    cs::Result result;
    if constexpr (sizeof(Unit) == 1) {
        result = cs::scan_narrow(input, fmt, options, target);
    } else {
        result = cs::scan_wide(input, fmt, options, target);
    }
    if (result.status != cs::Status::Ok) {
        *pg6_crt_errno() = PG6_CRT_EINVAL;
        return -1;
    }
    return result.value;
}

int32_t vsscanf_narrow(uint64_t o, uint64_t b, uint64_t c, uint64_t f, uint64_t l, uint64_t v) {
    return vsscanf_impl<char>(o, b, c, f, l, v);
}
int32_t vswscanf_wide(uint64_t o, uint64_t b, uint64_t c, uint64_t f, uint64_t l, uint64_t v) {
    return vsscanf_impl<char16_t>(o, b, c, f, l, v);
}

}  // namespace

extern "C" {

void pg6_gthunk_stdio_common_vsscanf(void* state) {
    gcom::function_thunk<&vsscanf_narrow>()(state);
}

void pg6_gthunk_stdio_common_vswscanf(void* state) {
    gcom::function_thunk<&vswscanf_wide>()(state);
}

}  // extern "C"
