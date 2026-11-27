// Hand-written guest-state thunks (os/host_marshallers.h): inet_ntoa (WS2_32 ordinal 11) and
// HCTraceImplMessage, on guest memory, including malformed inputs.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "abi/playgta6_abi.h"
#include "os/guest_heap.h"
#include "os/host_marshallers.h"
#include "os/os_shim.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_host_marshallers.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

namespace reg = playgta6::GuestReg;
using playgta6::GuestState;
using playgta6::os::GuestHeap;

std::mutex g_log_mutex;
std::vector<std::string> g_log;

void tap(std::string_view line) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_log.emplace_back(line);
}

uint64_t put(GuestHeap& heap, const void* data, size_t size) {
    const uint64_t va = heap.alloc(size + 16);
    std::memcpy(heap.to_host(va), data, size);
    return va;
}

uint64_t put_str(GuestHeap& heap, const char* s) {
    return put(heap, s, std::strlen(s) + 1);
}

uint64_t inet_ntoa_call(GuestHeap& heap, uint64_t ecx_value) {
    GuestState state{};
    const uint64_t stack = heap.alloc(64);
    state.gpr[reg::Rcx] = ecx_value;
    state.gpr[reg::Rsp] = stack;
    pg6_gthunk_inet_ntoa(&state);
    CHECK(state.gpr[reg::Rsp] == stack + 8);
    return state.gpr[reg::Rax];
}

void test_inet_ntoa() {
    GuestHeap& heap = playgta6::os::default_guest_heap();
    // s_addr is in network order: the first octet is the low byte.
    uint64_t p = inet_ntoa_call(heap, 0x04030201u);
    CHECK(p != 0 && std::strcmp(reinterpret_cast<const char*>(heap.to_host(p)), "1.2.3.4") == 0);
    // The buffer is per-thread and reused: the second call overwrites the first result.
    const uint64_t q = inet_ntoa_call(heap, 0xFFFFFFFFu);
    CHECK(q == p);
    CHECK(std::strcmp(reinterpret_cast<const char*>(heap.to_host(q)), "255.255.255.255") == 0);
    CHECK(std::strcmp(reinterpret_cast<const char*>(heap.to_host(inet_ntoa_call(heap, 0))), "0.0.0.0") == 0);
    // Only ecx is the argument: garbage in the upper half of rcx is ignored.
    p = inet_ntoa_call(heap, 0xDEADBEEF0100007Full);
    CHECK(std::strcmp(reinterpret_cast<const char*>(heap.to_host(p)), "127.0.0.1") == 0);
}

// HCTraceImplMessage(area, level, format, ...) with variadic slots 0 = r9, then the stack.
struct TraceCall {
    uint64_t area;
    uint32_t level;
    uint64_t format;
    std::vector<uint64_t> varargs;
};

std::string trace(GuestHeap& heap, const TraceCall& c) {
    GuestState state{};
    const uint64_t stack = heap.alloc(512);
    state.gpr[reg::Rcx] = c.area;
    state.gpr[reg::Rdx] = c.level;
    state.gpr[reg::R8] = c.format;
    state.gpr[reg::R9] = c.varargs.empty() ? 0 : c.varargs[0];
    state.gpr[reg::Rsp] = stack;
    for (size_t i = 1; i < c.varargs.size(); ++i) {
        std::memcpy(heap.to_host(stack + playgta6::kGuestStackArgOffset + 8 * i), &c.varargs[i], 8);
    }
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        g_log.clear();
    }
    pg6_gthunk_hc_trace_impl_message(&state);
    CHECK(state.gpr[reg::Rsp] == stack + 8);
    std::lock_guard<std::mutex> lock(g_log_mutex);
    return g_log.size() == 1 ? g_log[0] : std::string("<log count ") + std::to_string(g_log.size()) + ">";
}

void test_trace() {
    GuestHeap& heap = playgta6::os::default_guest_heap();
    const uint64_t name = put_str(heap, "HTTPCLIENT");
    const uint64_t area = put(heap, &name, sizeof(name));  // HCTraceImplArea { Name, Verbosity }
    const uint64_t text = put_str(heap, "hello");

    CHECK(trace(heap, {area, 1, put_str(heap, "plain"), {}}) == "libHttpClient [HTTPCLIENT] Error: plain");
    // r9 and four stack slots: ints, a string, a double.
    const double d = 2.5;
    uint64_t dbits;
    std::memcpy(&dbits, &d, 8);
    CHECK(trace(heap, {area, 4, put_str(heap, "%d/%s/%x/%.1f/%u\n"), {7, text, 255, dbits, 9}}) ==
          "libHttpClient [HTTPCLIENT] Information: 7/hello/ff/2.5/9");
    CHECK(trace(heap, {area, 5, put_str(heap, "v"), {}}) == "libHttpClient [HTTPCLIENT] Verbose: v");
    CHECK(trace(heap, {area, 77, put_str(heap, "x"), {}}) == "libHttpClient [HTTPCLIENT] Level?: x");

    // Malformed inputs never fault and always log one line.
    CHECK(trace(heap, {0, 2, put_str(heap, "no area"), {}}) == "libHttpClient [?] Warning: no area");
    CHECK(trace(heap, {0x7777000000ull, 2, put_str(heap, "bad area"), {}}) ==
          "libHttpClient [?] Warning: bad area");
    CHECK(trace(heap, {area, 1, 0, {}}) == "libHttpClient [HTTPCLIENT] Error: <unreadable format>");
    CHECK(trace(heap, {area, 1, 0x7777000000ull, {}}) == "libHttpClient [HTTPCLIENT] Error: <unreadable format>");
    // %n and unknown conversions are invalid: the raw format is logged.
    CHECK(trace(heap, {area, 1, put_str(heap, "a%nb"), {text}}) ==
          "libHttpClient [HTTPCLIENT] Error: a%nb (format error, arguments not applied)");
    CHECK(trace(heap, {area, 1, put_str(heap, "100%"), {}}).find("format error") != std::string::npos);
    // A %s pointing at unmapped memory: the format cannot be applied, no fault.
    CHECK(trace(heap, {area, 1, put_str(heap, "%s"), {0x7777000000ull}}).find("format error") !=
          std::string::npos);
    // A null %s prints (null) as MSVC does.
    CHECK(trace(heap, {area, 1, put_str(heap, "%s"), {0}}) == "libHttpClient [HTTPCLIENT] Error: (null)");
    // More conversions than variadic slots: bounded, not an endless read.
    std::string many;
    for (uint32_t i = 0; i < kHcTraceMaxArgs + 8; ++i) {
        many += "%d ";
    }
    CHECK(trace(heap, {area, 1, put_str(heap, many.c_str()), {1, 2, 3}}).find("format error") !=
          std::string::npos);
    // Control characters are neutralised; long output is cut.
    CHECK(trace(heap, {area, 1, put_str(heap, "a\x1b[31mb\tc"), {}}) ==
          "libHttpClient [HTTPCLIENT] Error: a?[31mb?c");
    const std::string longtext(kHcTraceMaxFormat + 100, 'x');
    const std::string cut = trace(heap, {area, 1, put_str(heap, longtext.c_str()), {}});
    CHECK(cut.size() < kHcTraceMaxMessage + 64 && cut.find("...") != std::string::npos);
}

}  // namespace

int main() {
    playgta6::os::set_log_tap(&tap);
    test_inet_ntoa();
    test_trace();
    playgta6::os::set_log_tap(nullptr);
    if (g_failures != 0) {
        std::fprintf(stderr, "test_host_marshallers failed with %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("All playgta6_host_marshallers_tests passed.\n");
    return 0;
}
