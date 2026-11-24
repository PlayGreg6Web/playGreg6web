// MSVC printf conformance (os/crt_format.h): flags, widths, precision, Win64 sizes, %I64, %ls,
// %p, %a, inf/nan spellings, truncation and return values of the __stdio_common_* buffer
// functions, and the guest-state thunks end to end on guest memory.

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "abi/playgta6_abi.h"
#include "os/crt_format.h"
#include "os/guest_heap.h"
#include "os/ucrt_bridge.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_crt_format.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

namespace cf = playgta6::os::crt_format;

uint64_t bits(double v) {
    uint64_t b;
    std::memcpy(&b, &v, sizeof(b));
    return b;
}

uint64_t addr(const void* p) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p));
}

// Argument slots in host memory, strings at host addresses.
class HostArgs final : public cf::ArgSource {
public:
    explicit HostArgs(std::vector<uint64_t> slots) : slots_(std::move(slots)) {}
    bool next(uint64_t& slot) override {
        if (pos_ >= slots_.size()) {
            return false;
        }
        slot = slots_[pos_++];
        return true;
    }
    bool narrow(uint64_t address, size_t max, std::string& out) override {
        const char* s = reinterpret_cast<const char*>(static_cast<uintptr_t>(address));
        out.clear();
        for (size_t i = 0; i < max && s[i] != 0; ++i) {
            out.push_back(s[i]);
        }
        return true;
    }
    bool wide(uint64_t address, size_t max, std::u16string& out) override {
        const char16_t* s = reinterpret_cast<const char16_t*>(static_cast<uintptr_t>(address));
        out.clear();
        for (size_t i = 0; i < max && s[i] != 0; ++i) {
            out.push_back(s[i]);
        }
        return true;
    }
    size_t used() const { return pos_; }

private:
    std::vector<uint64_t> slots_;
    size_t pos_ = 0;
};

constexpr uint64_t kLegacy = cf::kLegacyWideSpecifiers;

std::string fmt(const char* format, std::vector<uint64_t> slots, uint64_t options = kLegacy,
                cf::Status* status = nullptr) {
    HostArgs args(std::move(slots));
    std::string out;
    const cf::Status s = cf::format_narrow(format, options, args, out);
    if (status != nullptr) {
        *status = s;
    }
    return s == cf::Status::Ok ? out : std::string("<error>");
}

std::u16string wfmt(const char16_t* format, std::vector<uint64_t> slots, uint64_t options = kLegacy) {
    HostArgs args(std::move(slots));
    std::u16string out;
    return cf::format_wide(format, options, args, out) == cf::Status::Ok ? out : u"<error>";
}

void test_integers() {
    CHECK(fmt("%d|%5d|%-5d|%05d|%+d|% d", {42, 42, 42, 42, 42, 42}) == "42|   42|42   |00042|+42| 42");
    CHECK(fmt("%d", {static_cast<uint64_t>(-7)}) == "-7");
    CHECK(fmt("%i", {0xFFFFFFFF00000005ull}) == "5");  // only the low 32 bits are an int
    CHECK(fmt("%ld", {0x0000000080000000ull}) == "-2147483648");  // Win64 long is 32-bit
    CHECK(fmt("%lu", {0x1FFFFFFFFull}) == "4294967295");
    CHECK(fmt("%I64d|%lld|%jd", {~0ull, ~0ull, ~0ull}) == "-1|-1|-1");
    CHECK(fmt("%zu|%Iu", {~0ull, ~0ull}) == "18446744073709551615|18446744073709551615");
    CHECK(fmt("%I32u", {0x100000001ull}) == "1");
    CHECK(fmt("%hd|%hhd|%hu", {0x18000, 0x80, 0x1FFFF}) == "-32768|-128|65535");
    CHECK(fmt("%x %X %#x %#X %#o %o", {255, 255, 255, 255, 8, 8}) == "ff FF 0xff 0XFF 010 10");
    CHECK(fmt("%#x|%#o", {0, 0}) == "0|0");
    CHECK(fmt("%.3d|%.0d|%5.2d|%-6.3x|", {7, 0, 3, 10}) == "007||   03|00a   |");
    CHECK(fmt("%08.3d", {5}) == "     005");  // the 0 flag is ignored with a precision
    CHECK(fmt("%-05d|", {5}) == "5    |");
    CHECK(fmt("%*d|%-*d|%*d", {4, 1, 3, 2, static_cast<uint64_t>(-3), 3}) == "   1|2  |3  ");
    CHECK(fmt("%.*d", {static_cast<uint64_t>(-1), 4}) == "4");  // negative precision = none
    CHECK(fmt("%%|", {}) == "%|");
}

void test_pointers_and_characters() {
    CHECK(fmt("%p", {0x1234}) == "0000000000001234");
    CHECK(fmt("%p", {0x7FF6ABCD12345678ull}) == "7FF6ABCD12345678");
    CHECK(fmt("[%20p]", {1}) == "[    0000000000000001]");
    CHECK(fmt("%c%c%3c%-3c|", {'A', 0x142, 'x', 'y'}) == "AB  xy  |");
    CHECK(fmt("%lc%C", {0xE9, 0x41}) == "\xE9" "A");
    cf::Status st = cf::Status::Ok;
    CHECK(fmt("%lc", {0x20AC}, kLegacy, &st) == "<error>");
    CHECK(st == cf::Status::Encoding);
}

void test_strings() {
    const char hello[] = "hello";
    const char16_t wide[] = u"w\u00e9";
    const char16_t euro[] = u"\u20ac";
    CHECK(fmt("%s|%.2s|%7s|%-7s|", {addr(hello), addr(hello), addr(hello), addr(hello)}) ==
          "hello|he|  hello|hello  |");
    CHECK(fmt("%s|%5.1s|", {0, 0}) == "(null)|    (|");
    CHECK(fmt("%ls|%S|%ws|%hs", {addr(wide), addr(wide), addr(wide), addr(hello)}) ==
          "w\xE9|w\xE9|w\xE9|hello");
    cf::Status st = cf::Status::Ok;
    fmt("%ls", {addr(euro)}, kLegacy, &st);
    CHECK(st == cf::Status::Encoding);
    // A precision limits how far the string is read (it need not be terminated).
    const char unterminated[3] = {'a', 'b', 'c'};
    CHECK(fmt("%.3s", {addr(unterminated)}) == "abc");
}

void test_floating() {
    CHECK(fmt("%f|%.2f|%.0f|%#.0f", {bits(1.5), bits(3.14159), bits(2.5), bits(3.0)}) ==
          "1.500000|3.14|2|3.");
    CHECK(fmt("%e|%E|%.2e", {bits(12345.678), bits(0.00012), bits(1.0)}) ==
          "1.234568e+04|1.200000E-04|1.00e+00");
    CHECK(fmt("%g|%g|%G|%g|%#g", {bits(0.0001), bits(1e-5), bits(1e20), bits(100000.0), bits(1.0)}) ==
          "0.0001|1e-05|1E+20|100000|1.00000");
    CHECK(fmt("%10.3f|%-10.3f|%010.3f|%+.1f|% .1f", {bits(-1.5), bits(1.5), bits(-1.5), bits(2.0),
                                                     bits(2.0)}) ==
          "    -1.500|1.500     |-00001.500|+2.0| 2.0");
    CHECK(fmt("%.*f", {1, bits(0.25)}) == "0.2");  // correctly rounded (ties to even)
    CHECK(fmt("%f|%F|%e|%5f|%-6f|", {bits(INFINITY), bits(-INFINITY), bits(INFINITY), bits(INFINITY),
                                     bits(-INFINITY)}) == "inf|-INF|inf|  inf|-inf  |");
    const uint64_t ind = 0xFFF8000000000000ull;  // the default NaN of 0.0 / 0.0 on x86
    const uint64_t qnan = 0x7FF8000000000000ull;
    const uint64_t snan = 0x7FF0000000000001ull;
    CHECK(fmt("%f|%f|%F|%f|%E", {ind, qnan, qnan, snan, ind}) ==
          "-nan(ind)|nan|NAN|nan(snan)|-NAN(IND)");
    CHECK(fmt("%e", {bits(1.0)}, kLegacy | cf::kLegacyThreeDigitExponents) == "1.000000e+000");
    CHECK(fmt("%e", {bits(1e100)}, kLegacy | cf::kLegacyThreeDigitExponents) == "1.000000e+100");
    CHECK(fmt("%a|%A|%a|%a", {bits(1.0), bits(0.5), bits(-2.0), bits(0.0)}) ==
          "0x1.0000000000000p+0|0X1.0000000000000P-1|-0x1.0000000000000p+1|0x0.0000000000000p+0");
    CHECK(fmt("%.1a|%.0a|%.3a", {bits(1.0), bits(1.5), bits(1.0 + 1.0 / 4096)}) ==
          "0x1.0p+0|0x2p+0|0x1.001p+0");
    CHECK(fmt("%a", {bits(std::numeric_limits<double>::denorm_min())}) == "0x0.0000000000001p-1022");
    CHECK(fmt("%Lf", {bits(0.5)}) == "0.500000");  // MSVC long double = double
}

void test_invalid_formats() {
    cf::Status st = cf::Status::Ok;
    int sink = 0;
    CHECK(fmt("%n", {addr(&sink)}, kLegacy, &st) == "<error>");
    CHECK(st == cf::Status::InvalidFormat);
    CHECK(fmt("%y", {1}, kLegacy, &st) == "<error>" && st == cf::Status::InvalidFormat);
    CHECK(fmt("abc%", {}, kLegacy, &st) == "<error>" && st == cf::Status::InvalidFormat);
    CHECK(fmt("%d", {}, kLegacy, &st) == "<error>" && st == cf::Status::Unreadable);
    HostArgs args({1, 2, 3});
    std::string out;
    CHECK(cf::format_narrow("%d %d", kLegacy, args, out) == cf::Status::Ok && args.used() == 2);
}

void test_wide_functions() {
    const char narrow[] = "ab";
    const char16_t wide[] = u"cd";
    // Legacy (MSVC default): %s/%c are wide in the wide functions, %S/%C narrow.
    CHECK(wfmt(u"%s|%S|%c|%C|%hs|%ls", {addr(wide), addr(narrow), 0x20AC, 'x', addr(narrow), addr(wide)}) ==
          u"cd|ab|\u20ac|x|ab|cd");
    // ISO mode: %s/%c follow the C standard (narrow) unless l/w is given.
    CHECK(wfmt(u"%s|%ls|%c", {addr(narrow), addr(wide), 'z'}, 0) == u"ab|cd|z");
    CHECK(wfmt(u"\u00e9%5d|%.1f|%p", {12, bits(0.25), 255}) == u"\u00e9   12|0.2|00000000000000FF");
}

void test_store_semantics() {
    // vsnprintf (standard): always terminated, returns the full length.
    cf::Store s = cf::sprintf_store(10, true, 4, cf::kStandardSnprintfBehavior);
    CHECK(s.units == 3 && s.terminate && s.result == 10);
    s = cf::sprintf_store(10, true, 0, cf::kStandardSnprintfBehavior);
    CHECK(s.units == 0 && !s.terminate && s.result == 10);
    // _vsnprintf (legacy): fits -> terminated; exactly full -> not terminated; longer -> -1.
    s = cf::sprintf_store(3, true, 4, 0);
    CHECK(s.units == 3 && s.terminate && s.result == 3);
    s = cf::sprintf_store(4, true, 4, 0);
    CHECK(s.units == 4 && !s.terminate && s.result == 4);
    s = cf::sprintf_store(5, true, 4, 0);
    CHECK(s.units == 4 && !s.terminate && s.result == -1);
    s = cf::sprintf_store(0, true, 0, 0);
    CHECK(s.units == 0 && s.result == 0);
    s = cf::sprintf_store(7, false, 0, 0);  // _vscprintf-style measuring
    CHECK(s.result == 7 && s.units == 0);
    // vsprintf_s: too small -> buffer[0] = 0, -1, ERANGE.
    s = cf::sprintf_s_store(4, true, 4);
    CHECK(s.clear_first && s.result == -1 && s.error == PG6_CRT_ERANGE);
    s = cf::sprintf_s_store(3, true, 4);
    CHECK(s.units == 3 && s.terminate && s.result == 3);
    s = cf::sprintf_s_store(3, false, 4);
    CHECK(s.result == -1 && s.error == PG6_CRT_EINVAL);
    // vsnwprintf_s.
    s = cf::snprintf_s_store(10, true, 4, cf::kTruncate);
    CHECK(s.units == 3 && s.terminate && s.result == -1);
    s = cf::snprintf_s_store(2, true, 4, cf::kTruncate);
    CHECK(s.units == 2 && s.terminate && s.result == 2);
    s = cf::snprintf_s_store(10, true, 8, 5);
    CHECK(s.units == 5 && s.terminate && s.result == -1);
    s = cf::snprintf_s_store(10, true, 4, 9);
    CHECK(s.clear_first && s.result == -1 && s.error == PG6_CRT_ERANGE);
}

// The thunks read their arguments from guest registers and the guest stack and the format, the
// va_list slots and the strings from guest memory, and write into a guest buffer.
void test_thunks_on_guest_memory() {
    using playgta6::GuestState;
    namespace reg = playgta6::GuestReg;
    playgta6::os::GuestHeap& heap = playgta6::os::default_guest_heap();
    auto put = [&](const void* data, size_t size) {
        const uint64_t va = heap.alloc(size + 16);
        std::memcpy(heap.to_host(va), data, size);
        return va;
    };
    const char format_text[] = "%s=%d (%.2f) %ls";
    const char name_text[] = "frames";
    const char16_t wide_text[] = u"ok";
    const uint64_t format = put(format_text, sizeof(format_text));
    const uint64_t name = put(name_text, sizeof(name_text));
    const uint64_t wide = put(wide_text, sizeof(wide_text));
    const uint64_t slots[4] = {name, 60, bits(16.666), wide};
    const uint64_t va_list = put(slots, sizeof(slots));
    const uint64_t buffer = heap.alloc(64);
    const uint64_t stack = heap.alloc(256);

    GuestState state{};
    auto call = [&](void (*thunk)(void*), uint64_t options, uint64_t buf, uint64_t count) {
        state.gpr[reg::Rcx] = options;
        state.gpr[reg::Rdx] = buf;
        state.gpr[reg::R8] = count;
        state.gpr[reg::R9] = format;
        state.gpr[reg::Rsp] = stack;
        const uint64_t stack_args[2] = {0 /*locale*/, va_list};
        std::memcpy(heap.to_host(stack + 40), stack_args, sizeof(stack_args));
        thunk(&state);
        CHECK(state.gpr[reg::Rsp] == stack + 8);
        return static_cast<int32_t>(state.gpr[reg::Rax]);
    };
    const char* out = reinterpret_cast<const char*>(heap.to_host(buffer));
    CHECK(call(&pg6_gthunk_stdio_common_vsprintf, kLegacy, buffer, 64) == 20);
    CHECK(std::strcmp(out, "frames=60 (16.67) ok") == 0);
    // Truncation: legacy _vsnprintf returns -1 and leaves the buffer unterminated.
    std::memset(heap.to_host(buffer), 'Z', 64);
    CHECK(call(&pg6_gthunk_stdio_common_vsprintf, kLegacy, buffer, 6) == -1);
    CHECK(std::memcmp(out, "framesZ", 7) == 0);
    // vsnprintf (standard): truncated, terminated, full length.
    CHECK(call(&pg6_gthunk_stdio_common_vsprintf, kLegacy | cf::kStandardSnprintfBehavior, buffer, 6) == 20);
    CHECK(std::strcmp(out, "frame") == 0);
    // vsprintf_s: too small -> empty string, -1, ERANGE.
    *pg6_crt_errno() = 0;
    CHECK(call(&pg6_gthunk_stdio_common_vsprintf_s, kLegacy, buffer, 6) == -1);
    CHECK(out[0] == 0 && *pg6_crt_errno() == PG6_CRT_ERANGE);
    // Measuring.
    CHECK(call(&pg6_gthunk_stdio_common_vsprintf, kLegacy | cf::kStandardSnprintfBehavior, 0, 0) == 20);
}

// _get_stream_buffer_pointers hands msvcp140's basic_filebuf an empty buffer record in guest
// memory (guest addresses), stable per stream.
void test_stream_buffer_pointers() {
    playgta6::os::GuestHeap& heap = playgta6::os::default_guest_heap();
    char** base = nullptr;
    char** ptr = nullptr;
    int32_t* cnt = nullptr;
    FILE* out = pg6_crt_acrt_iob_func(1);
    CHECK(pg6_crt_get_stream_buffer_pointers(out, &base, &ptr, &cnt) == 0);
    const uint64_t record = addr(base);
    CHECK(heap.owns(record));
    CHECK(addr(ptr) == record + 8 && addr(cnt) == record + 16);
    uint64_t fields[3] = {1, 1, 1};
    std::memcpy(fields, heap.to_host(record), sizeof(fields));
    CHECK(fields[0] == 0 && fields[1] == 0 && static_cast<uint32_t>(fields[2]) == 0);
    char** base2 = nullptr;
    CHECK(pg6_crt_get_stream_buffer_pointers(out, &base2, nullptr, nullptr) == 0 && base2 == base);
    char** err_base = nullptr;
    CHECK(pg6_crt_get_stream_buffer_pointers(pg6_crt_acrt_iob_func(2), &err_base, nullptr, nullptr) == 0);
    CHECK(err_base != base);
    CHECK(pg6_crt_get_stream_buffer_pointers(nullptr, &base, &ptr, &cnt) == PG6_CRT_EINVAL);
}

}  // namespace

int main() {
    test_integers();
    test_pointers_and_characters();
    test_strings();
    test_floating();
    test_invalid_formats();
    test_wide_functions();
    test_store_semantics();
    test_thunks_on_guest_memory();
    test_stream_buffer_pointers();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_crt_format failed with %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("All playgta6_crt_format_tests passed.\n");
    return 0;
}
