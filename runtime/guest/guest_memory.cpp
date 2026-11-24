// Slow-path helpers of the R19 table memory models (recomp/src/lift/memory.rs): lifted code calls
// them for accesses that leave a 64 KiB slot and for `rep movs` / `rep stos` over translated
// ranges. They translate through os/guest_window.h, segment by segment; an unmapped window slot
// raises a guest fault (FaultKind::Trap at the faulting guest address).

#include <cstdint>

#include "guest/guest_runtime.h"
#include "os/guest_window.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define PG6_GUEST_HELPER EMSCRIPTEN_KEEPALIVE
#else
#define PG6_GUEST_HELPER
#endif

namespace {

// Largest single lifted access is 32 bytes (AVX); one buffer per thread.
alignas(32) thread_local uint8_t g_scratch[64];

[[noreturn]] void unmapped(uint64_t va) {
    playgta6::guest::raise_fault(playgta6::guest::FaultKind::Trap, nullptr, va);
}

}  // namespace

namespace window = playgta6::os::guest_window;

extern "C" {

PG6_GUEST_HELPER void* __playgta6_guest_read(uint64_t va, uint64_t n) {
    if (n > sizeof(g_scratch) || !window::copy_from_guest(g_scratch, va, static_cast<size_t>(n))) {
        unmapped(va);
    }
    return g_scratch;
}

PG6_GUEST_HELPER void* __playgta6_guest_scratch() {
    return g_scratch;
}

PG6_GUEST_HELPER void __playgta6_guest_write(uint64_t va, uint64_t n) {
    if (n > sizeof(g_scratch) || !window::copy_to_guest(va, g_scratch, static_cast<size_t>(n))) {
        unmapped(va);
    }
}

PG6_GUEST_HELPER void __playgta6_guest_memmove(uint64_t dst, uint64_t src, uint64_t n) {
    if (!window::move_guest(dst, src, static_cast<size_t>(n))) {
        unmapped(dst);
    }
}

PG6_GUEST_HELPER void __playgta6_guest_memset(uint64_t dst, uint8_t value, uint64_t n) {
    if (!window::fill_guest(dst, value, static_cast<size_t>(n))) {
        unmapped(dst);
    }
}

}  // extern "C"
