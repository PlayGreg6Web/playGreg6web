#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "abi/playgta6_abi.h"

// Guest address space: VirtualMemoryArena
// =======================================
//
// One contiguous guest address space per arena, one translation for everybody
// ---------------------------------------------------------------------------
// An arena manages the guest virtual addresses [space_base(), space_base() + space_size()) and
// backs them with ONE contiguous host region, so a guest VA translates with a single add:
//
//     host address = guest_base() + guest VA        (64-bit wrapping add)
//
// This is the translation lifted code (recomp/src/lift, `guest_ptr`), the import bridge
// (pg6_guest_seam.h) and every runtime component (image loader, guest stacks and IAT slots,
// Win32 heaps, VirtualAlloc shims) use; see docs/wasm_abi.md "Guest Address Space". The process
// arena (win32::get_vmem_arena()) publishes its base in `__playgta6_guest_base`
// (publish_guest_address_space()). host_ptr() is O(1) and valid across page and allocation
// boundaries: host_ptr(va + n) == host_ptr(va) + n. read()/write() are protection-checked
// memcpy wrappers.
//
// On top of the backing the arena keeps the Win32/Xbox bookkeeping: reservations at 64 KiB
// allocation granularity, 4 KiB pages, commit/decommit/protect/release with VirtualAlloc,
// VirtualFree, VirtualProtect and VirtualQuery semantics, zero-filled pages on every (re)commit,
// committed/reserved byte accounting. Reservations, commits and protection changes never move
// memory; decommitted pages are given back to the OS where the platform can.
//
// Platform backends (vmem_backing.cpp, the one place with OS calls in runtime/os/)
// ---------------------------------------------------------------------------
// * Windows: VirtualAlloc(MEM_RESERVE) of the whole space, MEM_COMMIT / MEM_DECOMMIT per page run,
//   VirtualProtect. POSIX: mmap(PROT_NONE, MAP_NORESERVE), mprotect to commit, decommit by mapping
//   fresh PROT_NONE pages over the run (contents discarded, zero on the next commit).
//   Page protections are ENFORCED on these hosts: NoAccess -> no access, ReadOnly/ExecuteRead ->
//   read-only, ReadWrite/ExecuteReadWrite -> read/write (execute is never needed by the host).
//   Placement: an arena constructed with ArenaPlacement::PreferIdentity first tries to reserve
//   its space at host address == guest VA (guest_base() == 0, possible on 64-bit Windows/Linux/
//   macOS unless a sanitizer owns that range); otherwise, and for ArenaPlacement::Anywhere, the
//   OS picks the host address and the reservation also covers guest VAs [0, space_base()), which
//   are never committed: null and small guest pointers fault instead of aliasing host memory.
//   Reserving is virtual only (no commit charge); the default 64 GiB space costs nothing until
//   pages are committed.
// * Emscripten (wasm64, -sMEMORY64 -pthread): there is one linear memory starting at 0. Linear
//   memory [0, kGuestAddressSpaceBase) belongs to the runtime (static data, stacks, malloc);
//   [kGuestAddressSpaceBase, maximum memory) is the guest pool. The process arena takes the pool
//   at guest VA == linear address (identity, __playgta6_guest_base == 0); further arenas (tests)
//   get pool regions from the top with a positive base. The runtime heap is fenced: the runtime
//   links with -Wl,--wrap=sbrk and __wrap_sbrk refuses to move the break past
//   kGuestAddressSpaceBase (malloc then returns null), so Emscripten's allocator never hands out
//   guest memory. The module declares -sMAXIMUM_MEMORY=16GB (the browser limit for memory64;
//   shared memories must declare a maximum) and starts small: committing guest pages grows the
//   memory (emscripten_resize_heap) to cover them, so the memory size tracks the highest
//   committed guest page, never the reservations. Linear memory cannot shrink and wasm has no
//   page protection: decommit only updates the bookkeeping and the page is zeroed when it is
//   committed again (a pool-wide "ever committed" bitmap; never-committed pages are still zero
//   from the memory grow), and protections are tracked, not enforced.
//
// Why identity in the browser (and the rejected alternatives)
// -----------------------------------------------------------
// (a) Identity (chosen): guest VA == linear address. Lifted code needs no translation at all when
//     lifted against the constant base 0; runtime/shim pointers and guest pointers are the same
//     values, so the existing shims that store host pointers into guest structures (UCRT
//     argv/environ/onexit tables, XMem, ...) or read pointers embedded in guest structures keep
//     working. Costs: the memory must cover the highest committed guest VA (the image at
//     0x1'4000'0000 makes it >= 5 GiB once loaded; on Windows browsers commit the grown range,
//     physical pages are only touched on use), the guest gets at most 16 - 4 = 12 GiB of address
//     space, and null / small guest pointers read runtime memory instead of trapping.
// (b) Offset with a positive base (guest region above the runtime heap, base folded into the
//     load offset): same capacity cost as (a) or worse, and embedded pointers break. Rejected.
// (c) Negative-biased offset (guest VA 4 GiB at a low linear address): saves up to 4 GiB of
//     linear memory and makes null/small pointers trap (they wrap past the end of memory), but
//     every guest access needs an extra add that cannot be folded into the unsigned wasm offset,
//     and every pointer the runtime and guest exchange outside the bridge's top-level arguments
//     (pointers inside structures, varargs, tables the shims fill) would need translating.
//     Rejected for the browser; the same formula with a non-zero base is what native builds use
//     when identity placement is not available, where it is only needed for tests/tools.
// guest_base is a mutable global (not a link-time constant) because native placement is chosen
// at run time; lifted code reads it with an invariant load (hoisted per function), or can be
// lifted against a constant when the runtime configuration is fixed (identity in the browser).
//
// Sizes (build parameters, bytes): PLAYGTA6_PROCESS_GUEST_SPACE_SIZE (process arena;
// default 64 GiB native, 12 GiB wasm = the whole pool) and PLAYGTA6_GUEST_SPACE_SIZE (other
// arenas; default 64 GiB native, 2 GiB wasm). The constructor takes any size at run time.

namespace playgta6::os {

inline constexpr uint64_t kPageSize = 4096;                  // 4 KiB
inline constexpr uint64_t kAllocationGranularity = 65536;    // 64 KiB
inline constexpr uint64_t kDefaultArenaBase = kGuestAddressSpaceBase;  // 0x1'0000'0000

#ifdef PLAYGTA6_GUEST_SPACE_SIZE
inline constexpr uint64_t kDefaultGuestSpaceSize = PLAYGTA6_GUEST_SPACE_SIZE;
#elif defined(__EMSCRIPTEN__)
inline constexpr uint64_t kDefaultGuestSpaceSize = uint64_t{2} << 30;
#else
inline constexpr uint64_t kDefaultGuestSpaceSize = uint64_t{64} << 30;
#endif

#ifdef PLAYGTA6_PROCESS_GUEST_SPACE_SIZE
inline constexpr uint64_t kProcessGuestSpaceSize = PLAYGTA6_PROCESS_GUEST_SPACE_SIZE;
#elif defined(__EMSCRIPTEN__)
inline constexpr uint64_t kProcessGuestSpaceSize = uint64_t{12} << 30;
#else
inline constexpr uint64_t kProcessGuestSpaceSize = uint64_t{64} << 30;
#endif

// Win32 / Xbox VirtualAlloc & VirtualFree flags.
inline constexpr uint32_t kMemCommit = 0x00001000u;
inline constexpr uint32_t kMemReserve = 0x00002000u;
inline constexpr uint32_t kMemDecommit = 0x00004000u;
inline constexpr uint32_t kMemRelease = 0x00008000u;
inline constexpr uint32_t kMemFree = 0x00010000u;

enum class MemState : uint32_t {
    Free = kMemFree,
    Reserve = kMemReserve,
    Commit = kMemCommit,
};

// Win32 / Xbox PAGE_* protection constants.
enum class PageProtect : uint32_t {
    NoAccess = 0x01u,          // PAGE_NOACCESS
    ReadOnly = 0x02u,          // PAGE_READONLY
    ReadWrite = 0x04u,         // PAGE_READWRITE
    ExecuteRead = 0x20u,       // PAGE_EXECUTE_READ
    ExecuteReadWrite = 0x40u,  // PAGE_EXECUTE_READWRITE
};

inline constexpr bool is_readable(PageProtect protect) {
    return protect == PageProtect::ReadOnly ||
           protect == PageProtect::ReadWrite ||
           protect == PageProtect::ExecuteRead ||
           protect == PageProtect::ExecuteReadWrite;
}

inline constexpr bool is_writable(PageProtect protect) {
    return protect == PageProtect::ReadWrite ||
           protect == PageProtect::ExecuteReadWrite;
}

inline constexpr bool is_executable(PageProtect protect) {
    return protect == PageProtect::ExecuteRead ||
           protect == PageProtect::ExecuteReadWrite;
}

inline constexpr uint64_t align_down(uint64_t value, uint64_t alignment) {
    return value & ~(alignment - 1u);
}

inline constexpr bool align_up_checked(uint64_t value, uint64_t alignment, uint64_t& out) {
    if (alignment == 0 || (alignment & (alignment - 1u)) != 0) {
        return false;
    }
    const uint64_t mask = alignment - 1u;
    if (value > UINT64_MAX - mask) {
        return false;
    }
    out = (value + mask) & ~mask;
    return true;
}

struct MemoryBasicInfo {
    uint64_t base_address = 0;
    uint64_t allocation_base = 0;
    uint64_t region_size = 0;
    MemState state = MemState::Free;
    PageProtect protect = PageProtect::NoAccess;
    PageProtect allocation_protect = PageProtect::NoAccess;
};

enum class ArenaPlacement : uint8_t {
    Anywhere,        // any host address (guest_base() != 0 in general)
    PreferIdentity,  // host address == guest VA if possible, else like Anywhere
};

// Software virtual memory arena matching Win32/Xbox VirtualAlloc, VirtualProtect,
// VirtualFree, and VirtualQuery semantics for Wasm64 and native host execution.
class VirtualMemoryArena {
public:
    // Manages guest VAs [default_base, default_base + space_size) (default_base rounded up to
    // 64 KiB, space_size to 64 KiB). If no host backing can be reserved the arena is !valid() and
    // every reserve fails cleanly.
    explicit VirtualMemoryArena(uint64_t default_base = kDefaultArenaBase,
                                uint64_t space_size = kDefaultGuestSpaceSize,
                                ArenaPlacement placement = ArenaPlacement::Anywhere);
    ~VirtualMemoryArena();

    VirtualMemoryArena(const VirtualMemoryArena&) = delete;
    VirtualMemoryArena& operator=(const VirtualMemoryArena&) = delete;

    // Reserve a 64 KiB-aligned virtual address range (size rounded up to 4 KiB pages).
    // A non-zero hint is honoured exactly (rounded down to 64 KiB) or the call fails.
    // Returns the reserved base VA on success, or 0 on failure.
    uint64_t reserve(uint64_t hint_addr,
                     uint64_t size,
                     PageProtect allocation_protect = PageProtect::NoAccess);

    // Commit a 4 KiB page-aligned sub-range inside an existing reservation.
    // Newly committed pages are zero-initialized; already committed pages retain data
    // and receive the updated protection.
    bool commit(uint64_t addr,
                uint64_t size,
                PageProtect protect = PageProtect::ReadWrite);

    // Decommit a 4 KiB page-aligned sub-range, releasing its host backing storage.
    // If size == 0 and addr == reservation base, decommits the entire reservation.
    bool decommit(uint64_t addr, uint64_t size);

    // Change protection on a committed 4 KiB page-aligned sub-range.
    // Fails if any page in the range is uncommitted.
    bool protect(uint64_t addr,
                 uint64_t size,
                 PageProtect new_protect,
                 PageProtect* old_protect = nullptr);

    // Release an entire reservation by its exact base VA.
    bool release(uint64_t base_addr);

    // Unified Win32/Xbox VirtualAlloc wrapper (kMemReserve, kMemCommit, or both).
    uint64_t virtual_alloc(uint64_t addr,
                           uint64_t size,
                           uint32_t alloc_type,
                           PageProtect protect);

    // Unified Win32/Xbox VirtualFree wrapper (kMemDecommit or kMemRelease).
    bool virtual_free(uint64_t addr, uint64_t size, uint32_t free_type);

    // Query state and protection of the region starting at addr (Win32 VirtualQuery).
    MemoryBasicInfo query(uint64_t addr) const;

    // Bounds- and protection-checked copies (thin wrappers over memcpy on the backing).
    bool read(uint64_t addr, void* dst, size_t len) const;
    bool write(uint64_t addr, const void* src, size_t len);

    // Host pointer for [addr, addr + len) if every page in it is committed and writable
    // (host_ptr) / readable (host_ptr_const), else nullptr. The range may span pages,
    // commits and reservations: the backing is contiguous.
    uint8_t* host_ptr(uint64_t addr, size_t len = 1);
    const uint8_t* host_ptr_const(uint64_t addr, size_t len = 1) const;

    // Unchecked translation (what lifted code does): guest_base() + va, and back.
    uint8_t* translate(uint64_t va) const {
        return reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(guest_base_ + va));
    }
    uint64_t guest_address(const void* host) const {
        return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(host)) - guest_base_;
    }

    bool valid() const { return backing_size_ != 0; }
    uint64_t guest_base() const { return guest_base_; }
    bool is_identity() const { return valid() && guest_base_ == 0; }
    uint64_t space_base() const { return space_base_; }
    uint64_t space_size() const { return space_size_; }
    // True when page protections are enforced by the host (native builds), false on wasm.
    static bool protections_enforced();

    uint64_t reserved_bytes() const;
    uint64_t committed_bytes() const;
    size_t reservation_count() const;

private:
    struct PageMeta {
        MemState state = MemState::Reserve;
        PageProtect protect = PageProtect::NoAccess;
    };

    struct Reservation {
        uint64_t base = 0;
        uint64_t size = 0;              // Page-aligned reserved size
        uint64_t granularity_size = 0;  // 64 KiB-aligned VA slot occupancy
        PageProtect allocation_protect = PageProtect::NoAccess;
        std::vector<PageMeta> pages;
    };

    uint64_t reserve_locked(uint64_t hint_addr,
                            uint64_t size,
                            PageProtect allocation_protect);
    bool commit_locked(uint64_t addr, uint64_t size, PageProtect protect);
    bool decommit_locked(uint64_t addr, uint64_t size);
    bool release_locked(uint64_t base_addr);

    Reservation* find_reservation_locked(uint64_t addr);
    const Reservation* find_reservation_locked(uint64_t addr) const;
    bool pages_allow_locked(uint64_t addr, size_t len, bool need_write) const;
    bool range_overlaps_locked(uint64_t start, uint64_t end) const;
    uint64_t find_free_va_locked(uint64_t granularity_size) const;

    static bool is_valid_protect(PageProtect protect);

    uint64_t default_base_ = kDefaultArenaBase;  // == space_base_
    uint64_t space_base_ = kDefaultArenaBase;
    uint64_t space_size_ = 0;
    uint64_t guest_base_ = 0;     // host = guest_base_ + va
    uint8_t* backing_ = nullptr;  // host start of the backing reservation
    uint64_t backing_size_ = 0;
    mutable std::mutex mutex_;
    std::map<uint64_t, Reservation> reservations_;
    uint64_t reserved_bytes_ = 0;
    uint64_t committed_bytes_ = 0;
};

// Publishes `arena` as the guest address space lifted code and the import bridge use:
// sets __playgta6_guest_base = arena.guest_base(). Must not run while guest code runs.
void publish_guest_address_space(const VirtualMemoryArena& arena);

// Optional diagnostics observer of reserve / commit / decommit / release requests on any arena
// (successful or not), e.g. the boot report's address-space log. It runs under the arena's lock:
// it must not call back into the arena. nullptr removes it. Returns the previous observer.
enum class ArenaEvent : uint8_t { Reserve, Commit, Decommit, Release };
using ArenaObserver = void (*)(const VirtualMemoryArena& arena,
                               ArenaEvent event,
                               uint64_t va,     // result for a successful reserve, else the request
                               uint64_t size,   // requested bytes (0 for release)
                               PageProtect protect,
                               bool ok);
ArenaObserver set_arena_observer(ArenaObserver observer);

}  // namespace playgta6::os
