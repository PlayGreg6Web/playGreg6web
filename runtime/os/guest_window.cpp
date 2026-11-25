// Translated CPU window (R19 prototype). Design: guest_window.h.

#include "os/guest_window.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

#include "abi/playgta6_abi.h"

extern "C" {
alignas(64) int64_t __playgta6_window_table[playgta6::os::guest_window::kWindowSlots];
// O37: write marks, one per slot plus the spare entry branch-free window checks store to.
alignas(64) uint8_t __playgta6_window_dirty[playgta6::os::guest_window::kWindowSlots + 1];
}

namespace playgta6::os::guest_window {

static_assert(kWindowBase == kGuestWindowBase && kWindowMaxSize == kGuestWindowSize);
static_assert(kSlotBytes == kGuestWindowSlotBytes && kTrapDelta == kGuestWindowTrapDelta);

namespace {

struct State {
    std::mutex mutex;  // serialises writers; readers (lifted code, translate) read entries atomically
    std::atomic<bool> enabled{false};
    uint64_t map_calls = 0;
    uint64_t unmap_calls = 0;
    // guest_address(): host page -> the (last) slot mapped onto it; reverse_entries mirrors its
    // size so the common case (nothing mapped) needs no lock.
    std::map<uint64_t, uint64_t> reverse;
    std::atomic<size_t> reverse_entries{0};
};

State& state() {
    static State s;
    return s;
}

std::atomic_ref<int64_t> entry(uint64_t slot) {
    return std::atomic_ref<int64_t>(__playgta6_window_table[slot]);
}

uint64_t slot_of(uint64_t va) {
    return (va - kWindowBase) >> kSlotShift;
}

uint64_t slot_va(uint64_t slot) {
    return kWindowBase + (slot << kSlotShift);
}

void fill_unmapped_locked() {
    for (uint64_t i = 0; i < kWindowSlots; ++i) {
        entry(i).store(static_cast<int64_t>(kTrapDelta), std::memory_order_relaxed);
    }
    state().reverse.clear();
    state().reverse_entries.store(0, std::memory_order_release);
}

// Forgets the reverse entry of `slot`'s current host page (if it still names this slot).
void forget_reverse_locked(State& s, uint64_t slot) {
    const uint64_t delta = static_cast<uint64_t>(entry(slot).load(std::memory_order_relaxed));
    if (delta == kTrapDelta) {
        return;
    }
    const auto it = s.reverse.find(slot_va(slot) + delta);
    if (it != s.reverse.end() && it->second == slot) {
        s.reverse.erase(it);
        s.reverse_entries.store(s.reverse.size(), std::memory_order_release);
    }
}

// Host address of `va` when [va, va + run) stays inside one translation unit (a window slot, or
// the flat space), else of the first byte; `run` is set to the bytes left in that unit.
uint8_t* host_and_run(uint64_t va, size_t want, size_t* run) {
    if (!in_window(va) || !state().enabled.load(std::memory_order_acquire)) {
        // Flat space: contiguous up to the window start (if it is ahead).
        uint64_t limit = want;
        if (va < kWindowBase && kWindowBase - va < limit) {
            limit = kWindowBase - va;
        }
        *run = static_cast<size_t>(limit);
        return reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(__playgta6_guest_base + va));
    }
    const uint64_t delta = static_cast<uint64_t>(entry(slot_of(va)).load(std::memory_order_acquire));
    const uint64_t left = kSlotBytes - (va & (kSlotBytes - 1));
    *run = static_cast<size_t>(std::min<uint64_t>(left, want));
    if (delta == kTrapDelta) {
        return nullptr;
    }
    return reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(va + delta));
}

}  // namespace

void set_enabled(bool enabled) {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (enabled && !s.enabled.load(std::memory_order_relaxed)) {
        fill_unmapped_locked();
    }
    s.enabled.store(enabled, std::memory_order_release);
}

bool enabled() {
    return state().enabled.load(std::memory_order_acquire);
}

namespace {

std::atomic_ref<uint8_t> mark(uint64_t slot) {
    return std::atomic_ref<uint8_t>(__playgta6_window_dirty[slot]);
}

std::atomic<bool> g_verify{false};
std::atomic<uint64_t> g_verify_misses{0};

}  // namespace

void mark_written(uint64_t va, size_t len) {
    if (!in_window(va) || len == 0) {
        return;
    }
    const uint64_t last_va = std::min<uint64_t>(va + (len - 1), kWindowBase + kWindowMaxSize - 1);
    for (uint64_t slot = slot_of(va); slot <= slot_of(last_va); ++slot) {
        mark(slot).store(1, std::memory_order_release);
    }
}

bool take_written(uint64_t va, size_t len, std::vector<uint64_t>* slot_offsets) {
    if (!translated(va) || len == 0) {
        return false;
    }
    const uint64_t last_va = std::min<uint64_t>(va + (len - 1), kWindowBase + kWindowMaxSize - 1);
    for (uint64_t slot = slot_of(va); slot <= slot_of(last_va); ++slot) {
        if (mark(slot).exchange(0, std::memory_order_acq_rel) != 0 && slot_offsets != nullptr) {
            const uint64_t start = slot_va(slot);
            slot_offsets->push_back(start > va ? start - va : 0);
        }
    }
    return true;
}

void set_write_marks_verified(bool on) {
    g_verify.store(on, std::memory_order_relaxed);
}

bool write_marks_verified() {
    return g_verify.load(std::memory_order_relaxed);
}

void note_verify_miss(uint64_t) {
    g_verify_misses.fetch_add(1, std::memory_order_relaxed);
}

uint64_t verify_misses() {
    return g_verify_misses.load(std::memory_order_relaxed);
}

bool map_slot(uint64_t window_va, void* host_page) {
    if (!in_window(window_va)) {
        return false;
    }
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const uint64_t slot = slot_of(window_va);
    // Other bytes appear at the slot (or none): a reader's copy of it is stale.
    mark(slot).store(1, std::memory_order_release);
    forget_reverse_locked(s, slot);
    if (host_page == nullptr) {
        ++s.unmap_calls;
        entry(slot).store(static_cast<int64_t>(kTrapDelta), std::memory_order_release);
        return true;
    }
    ++s.map_calls;
    const uint64_t host = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(host_page));
    entry(slot).store(static_cast<int64_t>(host - slot_va(slot)), std::memory_order_release);
    s.reverse[host] = slot;
    s.reverse_entries.store(s.reverse.size(), std::memory_order_release);
    return true;
}

bool unmap_slot(uint64_t window_va) {
    return map_slot(window_va, nullptr);
}

void* slot_host(uint64_t window_va) {
    if (!in_window(window_va)) {
        return nullptr;
    }
    const uint64_t slot = slot_of(window_va);
    const uint64_t delta = static_cast<uint64_t>(entry(slot).load(std::memory_order_acquire));
    if (delta == kTrapDelta) {
        return nullptr;
    }
    return reinterpret_cast<void*>(static_cast<uintptr_t>(slot_va(slot) + delta));
}

void* translate_window(uint64_t va) {
    size_t run = 0;
    return host_and_run(va, 1, &run);
}

uint64_t guest_address(const void* host) {
    if (host == nullptr) {
        return 0;
    }
    const uint64_t h = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(host));
    State& s = state();
    if (s.reverse_entries.load(std::memory_order_acquire) != 0) {
        std::lock_guard<std::mutex> lock(s.mutex);
        auto it = s.reverse.upper_bound(h);
        if (it != s.reverse.begin()) {
            --it;
            if (h - it->first < kSlotBytes) {
                return slot_va(it->second) + (h - it->first);
            }
        }
    }
    return h - __playgta6_guest_base;
}

void* contiguous(uint64_t va, size_t len) {
    if (len == 0) {
        return translate(va);
    }
    size_t run = 0;
    uint8_t* first = host_and_run(va, len, &run);
    if (first == nullptr) {
        return nullptr;
    }
    uint64_t done = run;
    while (done < len) {
        uint8_t* next = host_and_run(va + done, len - done, &run);
        if (next != first + done) {
            return nullptr;
        }
        done += run;
    }
    return first;
}

bool copy_from_guest(void* dst, uint64_t va, size_t len) {
    auto* out = static_cast<uint8_t*>(dst);
    size_t done = 0;
    while (done < len) {
        size_t run = 0;
        const uint8_t* src = host_and_run(va + done, len - done, &run);
        if (src == nullptr) {
            return false;
        }
        std::memcpy(out + done, src, run);
        done += run;
    }
    return true;
}

bool copy_to_guest(uint64_t va, const void* src, size_t len) {
    const auto* in = static_cast<const uint8_t*>(src);
    size_t done = 0;
    while (done < len) {
        size_t run = 0;
        uint8_t* dst = host_and_run(va + done, len - done, &run);
        if (dst == nullptr) {
            return false;
        }
        std::memcpy(dst, in + done, run);
        mark_written(va + done, run);
        done += run;
    }
    return true;
}

bool move_guest(uint64_t dst, uint64_t src, size_t len) {
    if (len == 0 || dst == src) {
        return true;
    }
    if (void* d = contiguous(dst, len)) {
        if (void* s = contiguous(src, len)) {
            std::memmove(d, s, len);
            mark_written(dst, len);
            return true;
        }
    }
    // Segments of the two ranges do not line up and may overlap in host memory: bounce.
    std::vector<uint8_t> bounce(len);
    return copy_from_guest(bounce.data(), src, len) && copy_to_guest(dst, bounce.data(), len);
}

bool fill_guest(uint64_t dst, uint8_t value, size_t len) {
    size_t done = 0;
    while (done < len) {
        size_t run = 0;
        uint8_t* p = host_and_run(dst + done, len - done, &run);
        if (p == nullptr) {
            return false;
        }
        std::memset(p, value, run);
        mark_written(dst + done, run);
        done += run;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------
// Buffers: (pointer, length) marshaling
// ---------------------------------------------------------------------------------------------

namespace {

// Larger buffers are passed as they are (the host sees the first slot's page): a bounce copy of
// more than this is not worth it, and no caller passes window buffers that large.
constexpr uint64_t kMaxBounceBytes = uint64_t{256} << 20;

}  // namespace

void* Buffers::range_window(uint64_t va, uint64_t bytes, bool writable) {
    if (!enabled() || bytes == 0 || bytes > kMaxBounceBytes) {
        if (writable) {
            note_written(va, bytes);
        }
        return host_or_trap(va);
    }
    if (void* view = contiguous(va, static_cast<size_t>(bytes))) {
        if (writable) {
            note_written(va, bytes);  // pinned: the host writes the guest's bytes in place
        }
        return view;
    }
    Bounce b;
    b.va = va;
    b.bytes = static_cast<size_t>(bytes);
    b.writable = writable;
    b.data = std::make_unique<uint8_t[]>(b.bytes);
    if (!copy_from_guest(b.data.get(), va, b.bytes)) {
        return host_or_trap(va);  // a slot of the range is unmapped: fault like the guest would
    }
    void* host = b.data.get();
    bounces_.push_back(std::move(b));
    return host;
}

namespace {

// Bytes of the NUL-terminated string at guest `va` (terminator included), measured slot by slot
// through the shared translation; 0 if it runs into an unmapped slot or past kMaxBounceBytes.
uint64_t measure_string(uint64_t va, size_t char_size) {
    uint64_t bytes = 0;
    for (;;) {
        const uint64_t at = va + bytes;
        const auto* p = static_cast<const uint8_t*>(translate(at));
        if (p == nullptr) {
            return 0;
        }
        const uint64_t left = kSlotBytes - (at & (kSlotBytes - 1));
        uint64_t i = 0;
        for (; i + char_size <= left; i += char_size) {
            bool zero = true;
            for (size_t k = 0; k < char_size; ++k) {
                zero = zero && p[i + k] == 0;
            }
            if (zero) {
                return bytes + i + char_size;
            }
        }
        if (i != left) {
            // A character straddles the slot end (odd alignment): measure it through a copy.
            uint8_t ch[4] = {};
            if (!copy_from_guest(ch, at + i, char_size)) {
                return 0;
            }
            bool zero = true;
            for (size_t k = 0; k < char_size; ++k) {
                zero = zero && ch[k] == 0;
            }
            if (zero) {
                return bytes + i + char_size;
            }
            i += char_size;
        }
        bytes += i;
        if (bytes > kMaxBounceBytes) {
            return 0;
        }
    }
}

}  // namespace

void* Buffers::string_window(uint64_t va, size_t char_size) {
    if (!enabled() || char_size == 0 || char_size > 4) {
        return host_or_trap(va);
    }
    const uint64_t bytes = measure_string(va, char_size);
    return bytes == 0 ? host_or_trap(va) : range_window(va, bytes, false);
}

void* Buffers::string_out_window(uint64_t dst, uint64_t src, size_t char_size) {
    if (!enabled() || char_size == 0 || char_size > 4) {
        return host_or_trap(dst);
    }
    // Room for the destination's current string followed by the source's (strcat; an upper
    // bound for strcpy): a bounce copy may be larger than what the callee writes, never smaller.
    const uint64_t have = measure_string(dst, char_size);
    const uint64_t add = src != 0 ? measure_string(src, char_size) : 0;
    if (add == 0) {
        return host_or_trap(dst);
    }
    return range_window(dst, (have != 0 ? have - char_size : 0) + add, true);
}

void Buffers::finish() {
    for (Bounce& b : bounces_) {
        if (b.writable) {
            copy_to_guest(b.va, b.data.get(), b.bytes);  // marks the range
            b.writable = false;
        }
    }
    for (const Written& w : written_) {
        mark_written(w.va, static_cast<size_t>(w.bytes));
    }
    written_.clear();
}

uint64_t Buffers::guest_address(const void* host) const {
    const auto h = reinterpret_cast<uintptr_t>(host);
    for (const Bounce& b : bounces_) {
        const auto start = reinterpret_cast<uintptr_t>(b.data.get());
        if (h >= start && h <= start + b.bytes) {
            return b.va + (h - start);
        }
    }
    return guest_window::guest_address(host);
}

namespace {

std::atomic<bool> g_runs_required{false};
std::atomic<ContiguityViolationFn> g_on_violation{nullptr};

}  // namespace

void set_contiguous_runs_required(bool required, ContiguityViolationFn on_violation) {
    g_on_violation.store(on_violation, std::memory_order_release);
    g_runs_required.store(required, std::memory_order_release);
}

bool contiguous_runs_required() {
    return g_runs_required.load(std::memory_order_acquire);
}

void contiguity_violation(uint64_t window_va) {
    if (!contiguous_runs_required()) {
        return;
    }
    if (ContiguityViolationFn fn = g_on_violation.load(std::memory_order_acquire)) {
        fn(window_va);
    }
}

Stats stats() {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    Stats out;
    out.map_calls = s.map_calls;
    out.unmap_calls = s.unmap_calls;
    if (!s.enabled.load(std::memory_order_relaxed)) {
        return out;
    }
    int64_t prev = static_cast<int64_t>(kTrapDelta);
    for (uint64_t i = 0; i < kWindowSlots; ++i) {
        const int64_t d = entry(i).load(std::memory_order_relaxed);
        if (d != static_cast<int64_t>(kTrapDelta)) {
            ++out.mapped_slots;
            if (prev != static_cast<int64_t>(kTrapDelta)) {
                // Same delta == the next slot's page directly follows the previous one.
                (d == prev ? out.contiguous_pairs : out.split_pairs) += 1;
            }
        }
        prev = d;
    }
    return out;
}

void reset_for_testing() {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    fill_unmapped_locked();
    s.enabled.store(false, std::memory_order_release);
    s.map_calls = 0;
    s.unmap_calls = 0;
    for (uint64_t i = 0; i <= kWindowSlots; ++i) {
        mark(i).store(0, std::memory_order_relaxed);
    }
    g_verify.store(false, std::memory_order_relaxed);
    g_verify_misses.store(0, std::memory_order_relaxed);
}

}  // namespace playgta6::os::guest_window
