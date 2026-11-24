// MSVC sscanf conformance (os/crt_scan.h): integers and Win64 sizes, floats, strings, %c, scan
// sets, %n, suppression, literal and white-space matching, EOF vs. matching failure, the secure
// size slots, the wide function, and the guest-state thunk end to end on guest memory.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "abi/playgta6_abi.h"
#include "os/crt_scan.h"
#include "os/guest_heap.h"
#include "os/ucrt_bridge.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_crt_scan.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

namespace cs = playgta6::os::crt_scan;

uint64_t addr(const void* p) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p));
}

// Slots in host memory, storing to host addresses.
class HostTarget final : public cs::Target {
public:
    explicit HostTarget(std::vector<uint64_t> slots) : slots_(std::move(slots)) {}
    bool next(uint64_t& slot) override {
        if (pos_ >= slots_.size()) {
            return false;
        }
        slot = slots_[pos_++];
        return true;
    }
    bool store(uint64_t address, const void* data, size_t size) override {
        std::memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(address)), data, size);
        return true;
    }

private:
    std::vector<uint64_t> slots_;
    size_t pos_ = 0;
};

int32_t scan(const char* input, const char* format, std::vector<uint64_t> slots, uint64_t options = 0) {
    HostTarget target(std::move(slots));
    const cs::Result r = cs::scan_narrow(input, format, options, target);
    return r.status == cs::Status::Ok ? r.value : -100;
}

void test_integers() {
    int32_t a = 0, b = 0;
    uint32_t u = 0;
    CHECK(scan("  42 -17", "%d %d", {addr(&a), addr(&b)}) == 2);
    CHECK(a == 42 && b == -17);
    CHECK(scan("0x1F 017 9", "%i %i %u", {addr(&a), addr(&b), addr(&u)}) == 3);
    CHECK(a == 31 && b == 15 && u == 9);
    CHECK(scan("ff", "%x", {addr(&u)}) == 1 && u == 0xFF);
    CHECK(scan("0Xff", "%X", {addr(&u)}) == 1 && u == 0xFF);
    CHECK(scan("777", "%o", {addr(&u)}) == 1 && u == 0777);
    // Width limits the field.
    CHECK(scan("12345", "%2d%d", {addr(&a), addr(&b)}) == 2);
    CHECK(a == 12 && b == 345);
    // Sizes: hh / h store 1 / 2 bytes, l is 32-bit, ll / I64 / z 64-bit.
    uint8_t c8[2] = {0xEE, 0xEE};
    uint16_t s16[2] = {0xEEEE, 0xEEEE};
    CHECK(scan("300 70000", "%hhu %hu", {addr(&c8[0]), addr(&s16[0])}) == 2);
    CHECK(c8[0] == static_cast<uint8_t>(300) && c8[1] == 0xEE);
    CHECK(s16[0] == static_cast<uint16_t>(70000) && s16[1] == 0xEEEE);
    uint64_t q[2] = {~0ull, ~0ull};
    CHECK(scan("-1", "%ld", {addr(&q[0])}) == 1 && q[0] == 0xFFFFFFFF00000000ull + 0xFFFFFFFFull);
    CHECK(scan("123456789012", "%lld", {addr(&q[1])}) == 1 && q[1] == 123456789012ull);
    CHECK(scan("123456789012", "%I64u", {addr(&q[1])}) == 1 && q[1] == 123456789012ull);
    CHECK(scan("5", "%zu", {addr(&q[1])}) == 1 && q[1] == 5);
    uint64_t p = 0;
    CHECK(scan("00000001400010A0", "%p", {addr(&p)}) == 1 && p == 0x1400010A0ull);
}

void test_floats() {
    float f = 0;
    double d = 0;
    CHECK(scan("3.5 -2.25e2", "%f %lf", {addr(&f), addr(&d)}) == 2);
    CHECK(f == 3.5f && d == -225.0);
    CHECK(scan("1e", "%lf", {addr(&d)}) == 1 && d == 1.0);
    CHECK(scan("inf", "%lf", {addr(&d)}) == 1 && std::isinf(d));
    CHECK(scan("-nan", "%lf", {addr(&d)}) == 1 && std::isnan(d));
    CHECK(scan("0x1.8p1", "%la", {addr(&d)}) == 1 && d == 3.0);
    CHECK(scan("2.5", "%Lg", {addr(&d)}) == 1 && d == 2.5);
    CHECK(scan("12.75", "%3f", {addr(&f)}) == 1 && f == 12.0f);
    CHECK(scan("x", "%f", {addr(&f)}) == 0);
}

void test_strings() {
    char s[16] = {};
    char t[16] = {};
    CHECK(scan("  hello world", "%s %s", {addr(s), addr(t)}) == 2);
    CHECK(std::strcmp(s, "hello") == 0 && std::strcmp(t, "world") == 0);
    CHECK(scan("abcdef", "%3s", {addr(s)}) == 1 && std::strcmp(s, "abc") == 0);
    char c[4] = {'.', '.', '.', '.'};
    CHECK(scan(" xy", "%c", {addr(c)}) == 1 && c[0] == ' ' && c[1] == '.');
    CHECK(scan("xyz", "%2c", {addr(c)}) == 1 && c[0] == 'x' && c[1] == 'y' && c[2] == '.');
    CHECK(scan("key=value;", "%[^=]=%[a-z]", {addr(s), addr(t)}) == 2);
    CHECK(std::strcmp(s, "key") == 0 && std::strcmp(t, "value") == 0);
    CHECK(scan("]]x", "%[]]", {addr(s)}) == 1 && std::strcmp(s, "]]") == 0);
    CHECK(scan("123", "%[a-z]", {addr(s)}) == 0);
    // %ls stores wide characters from the narrow function.
    char16_t w[8] = {};
    CHECK(scan("wide", "%ls", {addr(w)}) == 1 && w[0] == u'w' && w[3] == u'e' && w[4] == 0);
}

void test_control() {
    int32_t a = 0, b = 0, n = -1;
    // %n counts consumed characters and is not a field; * suppresses.
    CHECK(scan("10 20 30", "%*d %d%n", {addr(&a), addr(&n)}) == 1);
    CHECK(a == 20 && n == 5);
    // Literals and %%.
    CHECK(scan("v=7%", "v=%d%%", {addr(&a)}) == 1 && a == 7);
    CHECK(scan("w=7", "v=%d", {addr(&a)}) == 0);
    // EOF: the input ends before the first conversion; afterwards the count so far.
    CHECK(scan("", "%d", {addr(&a)}) == -1);
    CHECK(scan("   ", "%d", {addr(&a)}) == -1);
    CHECK(scan("5", "%d %d", {addr(&a), addr(&b)}) == 1);
    CHECK(scan("x", "%d", {addr(&a)}) == 0);
    // Format errors.
    CHECK(scan("1", "%q", {addr(&a)}) == -100);
    CHECK(scan("1", "%[abc", {addr(&a)}) == -100);
}

void test_secure() {
    char s[4] = {'.', '.', '.', '.'};
    // The size slot follows the pointer: "abc" + NUL fits 4.
    CHECK(scan("abc", "%s", {addr(s), 4}, cs::kSecureCrt) == 1 && std::strcmp(s, "abc") == 0);
    // Too small: [0] = 0, the scan stops.
    int32_t a = 0;
    CHECK(scan("abcd 5", "%s %d", {addr(s), 4, addr(&a)}, cs::kSecureCrt) == 0 && s[0] == 0 && a == 0);
    char c[2] = {};
    CHECK(scan("xy", "%2c", {addr(c), 2}, cs::kSecureCrt) == 1 && c[0] == 'x' && c[1] == 'y');
}

void test_wide() {
    int32_t a = 0;
    char16_t w[8] = {};
    char n[8] = {};
    HostTarget target({addr(&a), addr(w), addr(n)});
    // Legacy wide specifiers: %s stores wide; %hs narrow.
    const cs::Result r = cs::scan_wide(u"12 ab cd", u"%d %s %hs", cs::kLegacyWideSpecifiers, target);
    CHECK(r.status == cs::Status::Ok && r.value == 3);
    CHECK(a == 12 && w[0] == u'a' && w[1] == u'b' && w[2] == 0 && std::strcmp(n, "cd") == 0);
}

// The thunk reads its arguments from guest registers and the guest stack and the input, format
// and va_list slots from guest memory, and stores through the guest pointers.
void test_thunk_on_guest_memory() {
    using playgta6::GuestState;
    namespace reg = playgta6::GuestReg;
    playgta6::os::GuestHeap& heap = playgta6::os::default_guest_heap();
    auto put = [&](const void* data, size_t size) {
        const uint64_t va = heap.alloc(size + 16);
        std::memcpy(heap.to_host(va), data, size);
        return va;
    };
    const char input_text[] = "[NAME_TAG] 1.25 build 7";
    const char format_text[] = "[%[^]]] %f build %d";
    const uint64_t input = put(input_text, sizeof(input_text));
    const uint64_t format = put(format_text, sizeof(format_text));
    const uint64_t name = heap.alloc(32);
    const uint64_t number = heap.alloc(8);
    const uint64_t build = heap.alloc(8);
    const uint64_t slots[3] = {name, number, build};
    const uint64_t va_list = put(slots, sizeof(slots));
    const uint64_t stack = heap.alloc(256);

    GuestState state{};
    auto call = [&](uint64_t buffer, uint64_t count) {
        state.gpr[reg::Rcx] = 0;
        state.gpr[reg::Rdx] = buffer;
        state.gpr[reg::R8] = count;
        state.gpr[reg::R9] = format;
        state.gpr[reg::Rsp] = stack;
        const uint64_t stack_args[2] = {0 /*locale*/, va_list};
        std::memcpy(heap.to_host(stack + 40), stack_args, sizeof(stack_args));
        pg6_gthunk_stdio_common_vsscanf(&state);
        CHECK(state.gpr[reg::Rsp] == stack + 8);
        return static_cast<int32_t>(state.gpr[reg::Rax]);
    };
    CHECK(call(input, UINT64_MAX) == 3);
    CHECK(std::strcmp(reinterpret_cast<const char*>(heap.to_host(name)), "NAME_TAG") == 0);
    float f = 0;
    std::memcpy(&f, heap.to_host(number), sizeof(f));
    int32_t b = 0;
    std::memcpy(&b, heap.to_host(build), sizeof(b));
    CHECK(f == 1.25f && b == 7);
    // A count shorter than the text ends the input early: "[NA" stores "NA", then the input ends.
    CHECK(call(input, 3) == 1);
    CHECK(std::strcmp(reinterpret_cast<const char*>(heap.to_host(name)), "NA") == 0);
    CHECK(call(input, 1) == -1);
    // Null buffer: -1 with EINVAL.
    *pg6_crt_errno() = 0;
    CHECK(call(0, UINT64_MAX) == -1 && *pg6_crt_errno() == PG6_CRT_EINVAL);
}

}  // namespace

int main() {
    test_integers();
    test_floats();
    test_strings();
    test_control();
    test_secure();
    test_wide();
    test_thunk_on_guest_memory();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_crt_scan failed with %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_crt_scan: all checks passed\n");
    return 0;
}
