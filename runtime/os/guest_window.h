#pragma once

// Translated CPU window (task R19, prototype; off unless a test or the boot configuration turns
// it on). Design, alternatives and measurements: recomp/src/lift/memory.rs (lifter side) and the
// section "R19: a translated window" in os/vmem.h.
//
// The title reserves a 16 GiB CPU window (24 GiB on the other console profile) and maps 64 KiB
// pool pages at arbitrary window slots (os/xmem_windows.h). The browser's identity-mapped guest
// pool is 12 GiB, so the window lives *outside* linear memory, at guest VAs
// [kWindowBase, kWindowBase + kWindowMaxSize) (above the 16 GiB wasm64 maximum), and every access
// to it translates through a per-64 KiB-slot delta table:
//
//     host = va + __playgta6_window_table[(va - kWindowBase) >> 16]        (window VAs)
//     host = va + __playgta6_guest_base                                    (everything else)
//
// A mapped slot's delta points the whole 64 KiB window page at the host bytes of the pool page
// mapped there; an unmapped slot holds kTrapDelta, which moves the address to >= 2^63: beyond any
// wasm64 memory (the access traps "out of bounds") and non-canonical natively (it faults). The
// table is 3 MiB of runtime memory (393,216 slots). Lifted code built with the lifter's
// `MemoryModel::Windowed` reads the same table; translate() below is the one runtime-side
// implementation every host consumer must use for guest pointers that may point into the window.
//
// Host code and the window: a guest buffer inside the window is contiguous in guest VA space but
// its slots may be backed by non-adjacent host pages. Code that takes (pointer, length) must
// either copy segment by segment (copy_from_guest / copy_to_guest) or ask for a contiguous view
// (contiguous(), which succeeds when every slot of the range is backed by host-adjacent pages, the
// common case when the title maps consecutive pool pages at consecutive window slots).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "abi/playgta6_abi.h"

namespace playgta6::os::guest_window {

inline constexpr uint64_t kSlotBytes = 64ull << 10;
inline constexpr uint32_t kSlotShift = 16;
inline constexpr uint64_t kWindowBase = 0x4'0000'0000ull;     // 16 GiB: above wasm64 memory
inline constexpr uint64_t kWindowMaxSize = 0x6'0000'0000ull;  // 24 GiB (larger console profile)
inline constexpr uint64_t kWindowSlots = kWindowMaxSize / kSlotBytes;
inline constexpr uint64_t kTrapDelta = 1ull << 63;

// Turns the translated window on (window VAs translate through the table) or off (default; every
// slot unmapped). Off -> on keeps nothing: all slots start unmapped.
void set_enabled(bool enabled);
bool enabled();

inline constexpr bool in_window(uint64_t va) {
    return va - kWindowBase < kWindowMaxSize;
}

// Maps the window slot containing `window_va` onto the 64 KiB host page `host_page` (nullptr
// unmaps). Returns false outside the window.
bool map_slot(uint64_t window_va, void* host_page);
bool unmap_slot(uint64_t window_va);
// Host page backing the slot of `window_va`, or nullptr if unmapped / outside.
void* slot_host(uint64_t window_va);

// Whether `va` translates through the window table right now (in the window, window on). Host
// consumers that validate guest pointers against an arena take the window path when this holds.
inline bool translated(uint64_t va) {
    return in_window(va) && enabled();
}

// The window part of translate(): `va` lies in the window.
void* translate_window(uint64_t va);

// The shared translation: host address of guest `va` (window slots through the table, everything
// else through __playgta6_guest_base). nullptr for an unmapped window slot. Outside the window
// (always, with the window off) the cost is one compare and a not-taken branch.
inline void* translate(uint64_t va) {
    if (in_window(va)) [[unlikely]] {
        return translate_window(va);
    }
    return reinterpret_cast<void*>(static_cast<uintptr_t>(__playgta6_guest_base + va));
}

// What a host consumer that must not see null for a non-null guest pointer uses (the generated
// import thunks): like translate(), but 0 stays null and an unmapped window slot gives the
// address the lifted code would access (va + kTrapDelta), so dereferencing it faults like the
// guest's own access would.
inline void* host_or_trap(uint64_t va) {
    if (in_window(va)) [[unlikely]] {
        void* host = translate_window(va);
        return host != nullptr ? host : reinterpret_cast<void*>(static_cast<uintptr_t>(va + kTrapDelta));
    }
    return va == 0 ? nullptr : reinterpret_cast<void*>(static_cast<uintptr_t>(__playgta6_guest_base + va));
}

// The reverse of translate(): the guest VA of host address `host`. A host page that backs a
// mapped window slot answers with the window VA (the address guest code knows the bytes by;
// a page mapped at several slots answers with one of them), anything else with
// host - __playgta6_guest_base. nullptr stays 0.
uint64_t guest_address(const void* host);

// Host pointer for [va, va + len) when the range is host-contiguous (outside the window, or every
// window slot it touches mapped onto adjacent host pages), else nullptr.
void* contiguous(uint64_t va, size_t len);

// Segmented copies through translate(); false (nothing or a prefix copied) on an unmapped slot.
bool copy_from_guest(void* dst, uint64_t va, size_t len);
bool copy_to_guest(uint64_t va, const void* src, size_t len);
// memmove / memset between guest ranges (window or not), overlap-safe.
bool move_guest(uint64_t dst, uint64_t src, size_t len);
bool fill_guest(uint64_t dst, uint8_t value, size_t len);

// Write marks (O37; docs/wasm_abi.md "__playgta6_window_dirty"): the browser has no OS write
// watch, so window slots carry their own. Lifted code marks a slot after storing into it; host code
// that writes window memory marks the range after the write (mark_written; the segmented copies,
// fill, move and Buffers do it themselves) and remapping a slot marks it. A reader that keeps a
// copy of window memory takes the marks of its range (take_written: appends the offsets of the
// marked slots within the range, relative to `va` and clamped to 0, and clears them) before it
// compares, so a write racing with the comparison is reported again. take_written is false for a
// range outside the window or with the window off: such memory has no marks.
void mark_written(uint64_t va, size_t len);
bool take_written(uint64_t va, size_t len, std::vector<uint64_t>* slot_offsets);
// Verification (tests, and real-program runs): a reader that uses the marks also compares the
// unmarked slots and reports changes found there (note_verify_miss): a miss is a writer that did
// not mark.
void set_write_marks_verified(bool on);
bool write_marks_verified();
void note_verify_miss(uint64_t va);
uint64_t verify_misses();

// (pointer, length) marshaling for host code called with guest buffers (O29; the generated
// import thunks of code lifted with the windowed memory model use one per call). Outside the
// window a buffer is its flat host pointer. A window buffer whose slots are mapped onto adjacent
// host pages is passed as that contiguous view ("pinned": the host reads and writes the guest's
// bytes in place). Any other window buffer is passed as a bounce copy of the guest bytes, copied
// back (writable buffers) by finish(), which the destructor calls: a segmented copy. NUL-terminated
// strings are measured through the window and handled the same way. guest_address() maps a host
// pointer into a bounce copy back to the guest address it stands for (pointer results).
// A bounce copy is a different address: host code that keeps the pointer beyond the call, or
// relies on its identity or on atomicity, only sees the guest's bytes in place for contiguous
// buffers (always, once the runtime keeps window runs contiguous; see guest/memory_model.h).
class Buffers {
public:
    Buffers() = default;
    Buffers(const Buffers&) = delete;
    Buffers& operator=(const Buffers&) = delete;
    ~Buffers() { finish(); }

    // Host pointer for the guest buffer [va, va + bytes).
    void* range(uint64_t va, uint64_t bytes, bool writable) {
        if (!in_window(va)) [[likely]] {
            return host_or_trap(va);
        }
        return range_window(va, bytes, writable);
    }
    // Host pointer for the NUL-terminated guest string at va (`char_size` bytes per character).
    void* string(uint64_t va, size_t char_size) {
        if (!in_window(va)) [[likely]] {
            return host_or_trap(va);
        }
        return string_window(va, char_size);
    }
    // Host pointer for a writable NUL-terminated string at dst that the callee extends with the
    // string at src (strcpy / strcat shapes: no length parameter).
    void* string_out(uint64_t dst, uint64_t src, size_t char_size) {
        if (!in_window(dst)) [[likely]] {
            return host_or_trap(dst);
        }
        return string_out_window(dst, src, char_size);
    }
    // A writable buffer the host writes in place (a contiguous view, or the COM bridge's direct
    // pointer arguments): its window slots are marked written by finish(), after the host call.
    void note_written(uint64_t va, uint64_t bytes) {
        if (in_window(va) && bytes != 0) [[unlikely]] {
            written_.push_back({va, bytes});
        }
    }
    // Copies writable bounce copies back to the guest and marks written ranges (once).
    void finish();
    // guest_window::guest_address(), plus pointers into this call's bounce copies.
    uint64_t guest_address(const void* host) const;
    size_t bounce_count() const { return bounces_.size(); }

private:
    struct Bounce {
        uint64_t va = 0;
        size_t bytes = 0;
        bool writable = false;
        std::unique_ptr<uint8_t[]> data;
    };
    void* range_window(uint64_t va, uint64_t bytes, bool writable);
    void* string_window(uint64_t va, size_t char_size);
    void* string_out_window(uint64_t dst, uint64_t src, size_t char_size);
    std::vector<Bounce> bounces_;  // empty (no allocation) unless a window buffer is split
    struct Written {
        uint64_t va = 0;
        uint64_t bytes = 0;
    };
    std::vector<Written> written_;  // window ranges written in place (marked by finish())
};

// Contiguous runs (O29): code lifted with the windowed model's `contiguous_runs` has no straddle
// check, so every two adjacent mapped slots must sit on adjacent host pages. The XMem layer keeps
// them so where it can (it moves a page's bytes to the frame next to its neighbour's); when it
// cannot, and runs are required, it reports the slot through the violation handler (the boot's
// stops the guest with a fault) instead of letting lifted code read the wrong bytes.
using ContiguityViolationFn = void (*)(uint64_t window_va);
void set_contiguous_runs_required(bool required, ContiguityViolationFn on_violation = nullptr);
bool contiguous_runs_required();
// Reports a slot whose neighbours could not be kept host-contiguous (no-op unless required).
void contiguity_violation(uint64_t window_va);

struct Stats {
    uint64_t mapped_slots = 0;
    // Adjacent pairs of mapped slots whose host pages are adjacent / not adjacent: how often a
    // range spanning two slots is host-contiguous.
    uint64_t contiguous_pairs = 0;
    uint64_t split_pairs = 0;
    uint64_t map_calls = 0;
    uint64_t unmap_calls = 0;
};
Stats stats();

void reset_for_testing();

}  // namespace playgta6::os::guest_window

extern "C" {
// The window's delta table read by lifted code (`MemoryModel::Windowed`), kWindowSlots entries.
extern int64_t __playgta6_window_table[];
// Write marks, kWindowSlots + 1 entries (the last is the spare branch-free checks store to).
extern uint8_t __playgta6_window_dirty[];
}
