// Guest memory model of a lifted program (O29). Design: guest/memory_model.h.

#include "guest/memory_model.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <mutex>

#include "abi/playgta6_abi.h"
#include "guest/guest_runtime.h"
#include "os/guest_window.h"
#include "os/os_shim.h"
#include "os/xmem_windows.h"

namespace playgta6::guest {

namespace {

namespace window = os::guest_window;
namespace xw = os::xmem_windows;

// Whether this runtime keeps every two adjacent mapped window slots host-contiguous, which code
// lifted with kPg6MemoryContiguousRuns relies on (it has no straddle check): XMem's window-centric
// placement moves a page's bytes next to its window neighbour's (os/xbox_scarlett.cpp,
// keep_window_run_locked), and where it cannot, the guest stops with a fault at that slot
// (runs_violation) instead of lifted code reading the wrong bytes.
constexpr bool kServesContiguousRuns = true;

[[noreturn]] void runs_violation(uint64_t window_va) {
    char line[200];
    std::snprintf(line, sizeof(line),
                  "translated window: the slots around 0x%" PRIx64
                  " cannot be kept host-contiguous (the program was lifted with contiguous runs)",
                  window_va);
    os::log(line);
    raise_fault(FaultKind::Trap, nullptr, window_va);
}

std::string hex(uint64_t v) {
    char buf[24];
    std::snprintf(buf, sizeof(buf), "0x%" PRIx64, v);
    return buf;
}

const char* check_name(uint32_t check) {
    switch (check) {
        case kPg6MemoryCheckBranch:
            return "branch";
        case kPg6MemoryCheckSelect:
            return "select";
        case kPg6MemoryCheckOutline:
            return "outline";
        default:
            return "unknown";
    }
}

struct Applied {
    std::mutex mutex;
    bool active = false;
    os::VirtualMemoryArena* arena = nullptr;
    uint64_t reservation = 0;  // the arena's no-access reservation over the window, 0 = none
    xw::PlacementPolicy previous_policy = xw::PlacementPolicy::Arena;
    bool enabled_window = false;  // the window was off before
};

Applied& applied() {
    static Applied a;
    return a;
}

}  // namespace

std::string describe_memory_model(const Pg6GuestMemoryModel& m) {
    std::string out;
    switch (m.model) {
        case kPg6MemoryModelFlat:
            out = "flat";
            break;
        case kPg6MemoryModelWindowed:
            out = "windowed [" + hex(m.window_base) + ", " + hex(m.window_base + m.window_size) +
                  ") check=" + check_name(m.check);
            break;
        case kPg6MemoryModelPageTable:
            out = "page-table (" + std::to_string(m.window_size) + " slots)";
            break;
        default:
            out = "model " + std::to_string(m.model);
            break;
    }
    if ((m.flags & kPg6MemoryStackFlat) != 0) {
        out += " stack-flat";
    }
    if ((m.flags & kPg6MemoryContiguousRuns) != 0) {
        out += " contiguous-runs";
    }
    out += (m.flags & kPg6MemoryFlatConstant) != 0 ? " flat-base=" + hex(m.flat_base)
                                                   : " flat-base=__playgta6_guest_base";
    return out;
}

std::string memory_model_unsupported(const Pg6GuestMemoryModel& m, const os::VirtualMemoryArena* arena) {
    if ((m.flags & ~kPg6MemoryKnownFlags) != 0 || m.reserved != 0) {
        return "unknown memory model flags " + hex(m.flags) + " (lifted by a newer playgta6-recomp?)";
    }
    if ((m.flags & kPg6MemoryFlatConstant) != 0 && arena != nullptr && arena->guest_base() != m.flat_base) {
        return "the lifted code translates guest addresses with the constant base " + hex(m.flat_base) +
               " but the guest address space is at " + hex(arena->guest_base());
    }
    switch (m.model) {
        case kPg6MemoryModelFlat:
            if ((m.flags & (kPg6MemoryStackFlat | kPg6MemoryContiguousRuns)) != 0) {
                return "flags " + hex(m.flags) + " are not valid for the flat memory model";
            }
            return {};
        case kPg6MemoryModelWindowed:
            break;
        case kPg6MemoryModelPageTable:
            return "the page-table memory model is a lifter prototype; this runtime has no page table";
        default:
            return "unknown memory model " + std::to_string(m.model) + " (lifted by a newer playgta6-recomp?)";
    }
    if (m.window_base != kGuestWindowBase || m.window_size != kGuestWindowSize) {
        return "the lifted code's translated window [" + hex(m.window_base) + ", " +
               hex(m.window_base + m.window_size) + ") is not this runtime's [" + hex(kGuestWindowBase) + ", " +
               hex(kGuestWindowBase + kGuestWindowSize) + ") (docs/wasm_abi.md)";
    }
    if (m.check > kPg6MemoryCheckOutline) {
        return "unknown window check " + std::to_string(m.check);
    }
    if ((m.flags & kPg6MemoryContiguousRuns) != 0 && !kServesContiguousRuns) {
        return "the lifted code assumes contiguous window runs (no straddle checks), but this runtime does "
               "not keep adjacent window slots host-contiguous; lift with --memory-model window";
    }
    return {};
}

bool apply_memory_model(const Pg6GuestMemoryModel& m, os::VirtualMemoryArena& arena, std::string& error) {
    release_memory_model();
    if (std::string why = memory_model_unsupported(m, &arena); !why.empty()) {
        error = "memory model " + describe_memory_model(m) + ": " + why;
        return false;
    }
    if (m.model != kPg6MemoryModelWindowed) {
        return true;
    }
    Applied& a = applied();
    std::lock_guard<std::mutex> lock(a.mutex);
    // Keep the arena out of the window: what it already holds there would be unreachable for
    // lifted code (window VAs translate through the table), and later allocations must not land
    // there either.
    const uint64_t lo = std::max(m.window_base, arena.space_base());
    const uint64_t hi = std::min(m.window_base + m.window_size, arena.space_base() + arena.space_size());
    if (lo < hi) {
        const uint64_t got = arena.reserve(lo, hi - lo, os::PageProtect::NoAccess);
        if (got != lo) {
            if (got != 0) {
                arena.release(got);
            }
            error = "memory model " + describe_memory_model(m) + ": the guest address space already uses [" +
                    hex(lo) + ", " + hex(hi) + "), which the translated window needs";
            return false;
        }
        a.reservation = lo;
    }
    a.arena = &arena;
    a.enabled_window = !window::enabled();
    window::set_enabled(true);
    a.previous_policy = xw::policy();
    xw::set_policy(xw::PlacementPolicy::PhantomGpuTranslatedCpu);
    window::set_contiguous_runs_required((m.flags & kPg6MemoryContiguousRuns) != 0, &runs_violation);
    a.active = true;
    return true;
}

void release_memory_model() {
    Applied& a = applied();
    std::lock_guard<std::mutex> lock(a.mutex);
    if (!a.active) {
        return;
    }
    xw::set_policy(a.previous_policy);
    window::set_contiguous_runs_required(false);
    // Slots still mapped belong to an XMem CPU window the guest did not release; they stay valid
    // (and the window on) until XMem unmaps them.
    if (a.enabled_window && window::stats().mapped_slots == 0) {
        window::set_enabled(false);
    }
    if (a.reservation != 0 && a.arena != nullptr) {
        a.arena->release(a.reservation);
    }
    a.active = false;
    a.arena = nullptr;
    a.reservation = 0;
    a.enabled_window = false;
}

}  // namespace playgta6::guest
