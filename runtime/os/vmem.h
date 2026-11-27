#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
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
//
// R19: a translated window (prototype, off by default)
// ----------------------------------------------------
// The title's 16 GiB CPU window does not fit the 12 GiB identity pool (os/xmem_windows.h). With
// os/guest_window.h turned on and xmem_windows::PlacementPolicy::PhantomGpuTranslatedCpu, the CPU
// window is placed above the wasm64 memory at [16 GiB, 40 GiB) and takes no arena address space;
// each mapped 64 KiB slot translates through a delta table onto the pool page's bytes. Lifted
// code must then be built with the lifter's MemoryModel::Windowed (recomp/src/lift/memory.rs,
// which also records the alternatives and the measurements), and every host consumer that takes
// a guest pointer that may point into the window must use guest_window::translate / the segmented
// copies instead of translate() above: this arena only translates arena addresses.

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
// O36: the process arena's shared zone (native hosts): room at the top of the space for the XMem
// physical-page reservations, whose pages the title maps at several addresses (its CPU window and
// pool views, os/xmem_windows.h: 16 GiB plus about 6 GiB of views on the largest profile; GPU
// windows stay in the arena: D3D12 reads their pages through the CPU window). With the default
// space that is guest VAs [40 GiB, 68 GiB), above the translated window's range. Spaces smaller
// than 64 GiB get none.
inline constexpr uint64_t kProcessSharedZoneSize = kProcessGuestSpaceSize >= (uint64_t{64} << 30) ? uint64_t{28} << 30 : 0;

// Win32 / Xbox VirtualAlloc & VirtualFree flags.
inline constexpr uint32_t kMemCommit = 0x00001000u;
inline constexpr uint32_t kMemReserve = 0x00002000u;
inline constexpr uint32_t kMemDecommit = 0x00004000u;
inline constexpr uint32_t kMemRelease = 0x00008000u;
inline constexpr uint32_t kMemFree = 0x00010000u;
inline constexpr uint32_t kMemReset = 0x00080000u;
inline constexpr uint32_t kMemResetUndo = 0x01000000u;
// Placeholders (VirtualAlloc2 / VirtualFree, Windows 10 1803+; the Xbox OS takes the same flags):
// a placeholder is a no-access reservation that can only be split, coalesced, replaced by a
// real reservation of exactly its range, or released.
inline constexpr uint32_t kMemCoalescePlaceholders = 0x00000001u;  // VirtualFree, with MEM_RELEASE
inline constexpr uint32_t kMemPreservePlaceholder = 0x00000002u;   // VirtualFree, with MEM_RELEASE
inline constexpr uint32_t kMemReplacePlaceholder = 0x00004000u;    // VirtualAlloc, with MEM_RESERVE
inline constexpr uint32_t kMemReservePlaceholder = 0x00040000u;    // VirtualAlloc, with MEM_RESERVE

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
    // O36: `shared_zone` bytes at the top of the space (rounded to 64 KiB) are kept for shared
    // page views (reserve_shared) where the host supports them (native); 0: none.
    explicit VirtualMemoryArena(uint64_t default_base = kDefaultArenaBase,
                                uint64_t space_size = kDefaultGuestSpaceSize,
                                ArenaPlacement placement = ArenaPlacement::Anywhere,
                                uint64_t shared_zone = 0);
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
    // and receive the updated protection. `newly_committed` (optional) receives how many bytes
    // of the range were not committed before (and so read as zero now): a caller that would
    // zero the range itself can skip that when it equals the range size (O36: a memset of a
    // fresh page touches memory the backing otherwise leaves untouched).
    bool commit(uint64_t addr,
                uint64_t size,
                PageProtect protect = PageProtect::ReadWrite,
                uint64_t* newly_committed = nullptr);

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

    // O36: shared page views - one memory object (a physical page) shown at several guest VAs.
    // reserve_shared reserves in the shared zone (0 when there is none or no room); its pages are
    // never committed: map_shared shows a shared page (backing::create_shared_page, `size` a
    // multiple of 64 KiB) read/write at a 64 KiB aligned VA in it, unmap_shared takes it away
    // (the pages are reserved again). Releasing the reservation unmaps what is still mapped.
    bool has_shared_zone() const { return shared_zone_size_ != 0; }
    uint64_t reserve_shared(uint64_t size);
    bool map_shared(uint64_t va, void* page, uint64_t size);
    bool unmap_shared(uint64_t va, uint64_t size);
    bool is_shared(uint64_t va) const;

    // Unified Win32/Xbox VirtualAlloc wrapper (kMemReserve, kMemCommit, or both).
    // Placeholders: kMemReserve | kMemReservePlaceholder with NoAccess reserves one (64 KiB
    // aligned, never committable); kMemReserve | kMemReplacePlaceholder (optionally | kMemCommit)
    // at the exact base and size of a placeholder turns it into a normal reservation.
    uint64_t virtual_alloc(uint64_t addr,
                           uint64_t size,
                           uint32_t alloc_type,
                           PageProtect protect);

    // Unified Win32/Xbox VirtualFree wrapper (kMemDecommit or kMemRelease).
    // kMemRelease | kMemPreservePlaceholder: splits [addr, addr + size) out of a placeholder as a
    // placeholder of its own (64 KiB boundaries), or turns a reservation that replaced a
    // placeholder back into one (its whole range; committed pages are discarded).
    // kMemRelease | kMemCoalescePlaceholders: merges the adjacent placeholders that exactly cover
    // [addr, addr + size) into one. kMemRelease of a placeholder (size 0) releases it.
    bool virtual_free(uint64_t addr, uint64_t size, uint32_t free_type);

    // Query state and protection of the region starting at addr (Win32 VirtualQuery).
    MemoryBasicInfo query(uint64_t addr) const;

    // Bounds- and protection-checked copies (thin wrappers over memcpy on the backing).
    bool read(uint64_t addr, void* dst, size_t len) const;

    // Fault reporting reads guest memory (backtraces, the report) on the faulting thread, and that
    // thread may hold this arena's lock: a host fault inside a GuestRangeView / host_ptr_const_held
    // hold (content_hash in flush_cpu_buffers). A blocking lock there self-deadlocks (MSVC throws
    // resource_deadlock_would_occur, the process terminates with exit 9 and no report). While a
    // DiagnosticReads scope is alive on a thread, read() there only tries the lock, for a bounded
    // time, and reads as "not readable" when it stays busy.
    class DiagnosticReads {
    public:
        DiagnosticReads();
        ~DiagnosticReads();
        DiagnosticReads(const DiagnosticReads&) = delete;
        DiagnosticReads& operator=(const DiagnosticReads&) = delete;
        static bool active();
    };
    bool write(uint64_t addr, const void* src, size_t len);
    // The runtime's own refill of pages that stay committed (XMem's in-place remap of a mirrored
    // window slot, docs/XMEM_REMAP.md): under the lock, copies `src` (zeroes when null) into
    // [addr, addr + len). The pages must be committed and readable (protection is the title's,
    // not checked for writing, as XMem's other fills); false (nothing written) otherwise. Holding
    // the lock over the copy keeps it whole for GuestRangeView / read_guest, which re-check the
    // window under the same lock. No arena event (it is not a guest write).
    bool fill_committed(uint64_t addr, const void* src, size_t len);

    // Host pointer for [addr, addr + len) if every page in it is committed and writable
    // (host_ptr) / readable (host_ptr_const), else nullptr. The range may span pages,
    // commits and reservations: the backing is contiguous.
    uint8_t* host_ptr(uint64_t addr, size_t len = 1);
    const uint8_t* host_ptr_const(uint64_t addr, size_t len = 1) const;
    // As host_ptr_const, but on success the arena's lock stays held in `hold` (the pointer stays
    // valid, i.e. no other thread can decommit / release / remap its pages, until `hold` is
    // released). On failure `hold` is empty. Use it when the bytes are read in place rather than
    // copied (a bare host_ptr_const pointer can be unmapped by another thread right after it returns).
    const uint8_t* host_ptr_const_held(uint64_t addr, size_t len, std::unique_lock<std::mutex>& hold) const;

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
    uint64_t peak_committed_bytes() const;
    uint64_t decommitted_bytes_total() const;
    uint64_t reclaimed_pages_reused() const;
    size_t reservation_count() const;

private:
    static constexpr size_t kLookupCacheSize = 64;
    static constexpr uint64_t kDecommitChunkShift = 24;  // 16 MiB = 4096 pages = 64 * u64
    static constexpr size_t kDecommitChunkWords = 64;

    struct DecommitChunk {
        std::array<uint64_t, kDecommitChunkWords> bits{};
        uint16_t count = 0;
    };

    struct PageMeta {
        uint8_t state_raw = 0;  // 0 = Reserve, 1 = Commit
        uint8_t protect_raw = static_cast<uint8_t>(PageProtect::NoAccess);

        PageMeta() = default;
        PageMeta(MemState s, PageProtect p)
            : state_raw(s == MemState::Commit ? 1 : 0),
              protect_raw(static_cast<uint8_t>(p)) {}

        MemState state() const { return state_raw != 0 ? MemState::Commit : MemState::Reserve; }
        void set_state(MemState s) { state_raw = s == MemState::Commit ? 1 : 0; }
        PageProtect protect() const { return static_cast<PageProtect>(protect_raw); }
        void set_protect(PageProtect p) { protect_raw = static_cast<uint8_t>(p); }
    };

    struct Reservation {
        uint64_t base = 0;
        uint64_t size = 0;              // Page-aligned reserved size
        uint64_t granularity_size = 0;  // 64 KiB-aligned VA slot occupancy
        PageProtect allocation_protect = PageProtect::NoAccess;
        std::vector<PageMeta> pages;
        bool placeholder = false;            // a placeholder (see virtual_alloc / virtual_free)
        bool replaced_placeholder = false;   // a reservation made by replacing a placeholder
        bool shared = false;                 // O36: in the shared zone (pages are views)
    };

    bool split_placeholder_locked(uint64_t addr, uint64_t size);
    bool coalesce_placeholders_locked(uint64_t addr, uint64_t size);

    uint64_t reserve_locked(uint64_t hint_addr,
                            uint64_t size,
                            PageProtect allocation_protect);
    bool commit_locked(uint64_t addr, uint64_t size, PageProtect protect, uint64_t* newly_committed = nullptr);
    bool decommit_locked(uint64_t addr, uint64_t size);
    bool release_locked(uint64_t base_addr);

    void invalidate_lookup_cache_locked() const;
    void mark_decommitted_page_locked(uint64_t va, uint64_t& cached_key, DecommitChunk*& cached_chunk);
    bool consume_reclaimed_page_locked(uint64_t va, uint64_t& cached_key, DecommitChunk*& cached_chunk);

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
    uint64_t shared_zone_base_ = 0;  // O36: guest VA of the shared zone (top of the space)
    uint64_t shared_zone_size_ = 0;
    mutable std::mutex mutex_;
    std::map<uint64_t, Reservation> reservations_;
    mutable const Reservation* last_res_ = nullptr;
    mutable std::array<const Reservation*, kLookupCacheSize> lookup_cache_{};
    std::unordered_map<uint64_t, DecommitChunk> decommitted_chunks_;
    uint64_t reserved_bytes_ = 0;
    uint64_t committed_bytes_ = 0;
    uint64_t peak_committed_bytes_ = 0;
    uint64_t decommitted_bytes_total_ = 0;
    uint64_t reclaimed_pages_reused_ = 0;
};

// O36: shared page objects for VirtualMemoryArena::map_shared (native hosts; nullptr elsewhere).
// view_shared_page gives the runtime its own view of the page's bytes.
void* create_shared_page(uint64_t size);
void release_shared_page(void* page);
uint8_t* view_shared_page(void* page, uint64_t size);
void unview_shared_page(uint8_t* view, uint64_t size);

// Publishes `arena` as the guest address space lifted code and the import bridge use:
// sets __playgta6_guest_base = arena.guest_base(). Must not run while guest code runs.
void publish_guest_address_space(const VirtualMemoryArena& arena);

// Optional diagnostics observer of reserve / commit / decommit / release requests on any arena
// (successful or not), e.g. the boot report's address-space log. It runs under the arena's lock:
// it must not call back into the arena. nullptr removes it. Returns the previous observer.
// Reset is a MEM_RESET / MEM_RESET_UNDO virtual_alloc (`protect` unused), Write a checked
// VirtualMemoryArena::write (both only reported when they succeed or fail on a valid request).
enum class ArenaEvent : uint8_t { Reserve, Commit, Decommit, Release, Reset, Write };
using ArenaObserver = void (*)(const VirtualMemoryArena& arena,
                               ArenaEvent event,
                               uint64_t va,     // result for a successful reserve, else the request
                               uint64_t size,   // requested bytes (0 for release)
                               PageProtect protect,
                               bool ok);
ArenaObserver set_arena_observer(ArenaObserver observer);

// Write tracking of arena memory. Appends to `offsets` the offsets, relative to `host` rounded
// down to kWrittenPageSize, of every kWrittenPageSize page in [host, host + len) written since the
// previous call covering that page, and starts tracking them afresh. Pages committed (zero-filled)
// after `*commit_seq` count as written too, since a decommit and recommit changes their contents
// without a write; `*commit_seq` is advanced. Returns false where the platform cannot tell (no
// write tracking, a range outside a tracked reservation, or more commits since `*commit_seq` than
// are remembered): the caller must then treat the whole range as possibly written.
inline constexpr uint64_t kWrittenPageSize = 4096;
// Write tracking is opt-in: reservations made after set_write_tracking(true) carry it (Windows).
// Tracking a large reservation costs the memory manager a bitmap and time on reserve/release, so
// processes that create many arenas (tests) leave it off unless they need it.
// R40: shared page views (the shared zone, Windows) are always tracked, without MEM_WRITE_WATCH: a
// call covering a view makes every view of its page read-only until the next write through any of
// them, which faults once, stamps them all written and makes them writable again. A caller's
// cursor remembers the stamps it has seen, so callers whose ranges overlap all see a write (the
// `commit_seq` form has no such position: it reports every view in the range written).
void set_write_tracking(bool on);
struct WriteCursor {
    uint64_t commit_seq = 0;
    uint64_t view_clock = 0;
    // Position in the MEM_WRITE_WATCH fan-out (O36): the OS write bits are consumed by one owner and
    // stamped; each cursor is told every page stamped after its epoch, so several readers of the
    // same arena memory (buffer flush, texture sync) each see every write.
    uint64_t watch_epoch = 0;
    // Peek mode (O36): the take does not reset OS write bits itself, it counts a bit not yet collected as
    // written and collects (the whole reservation, at most every few ms) only to learn it was
    // written again. Cheap for readers that mostly find nothing (texture sync); never reports less.
    bool peek_watch = false;
};
bool take_written_pages(const void* host, uint64_t len, WriteCursor* cursor, std::vector<uint64_t>* offsets);
bool take_written_pages(const void* host, uint64_t len, uint64_t* commit_seq, std::vector<uint64_t>* offsets);
// Batch of takes (a submission's buffer flush): collects the OS write bits of the (merged) host
// ranges once, up front, so each take inside them skips its own OS call; a write after the collect
// is reported by the next batch. Takes outside the ranges, and other threads, are unaffected.
class WriteBatch {
public:
    explicit WriteBatch(std::vector<std::pair<uintptr_t, uintptr_t>>* ranges);
    ~WriteBatch();
    WriteBatch(const WriteBatch&) = delete;
    WriteBatch& operator=(const WriteBatch&) = delete;
};
// Like the cursor form, but non-destructive for other readers: only for a range wholly covered by
// tracked shared page views (never reads MEM_WRITE_WATCH). False when the range is not (or the
// commit log was overrun): the caller must then treat everything as written. The cursor's commit
// position only advances on true.
bool take_written_view_pages(const void* host, uint64_t len, WriteCursor* cursor, std::vector<uint64_t>* offsets);
// R40: writes to shared page views caught so far (one per page and take interval).
uint64_t shared_view_write_faults();
// R40: a host or kernel call about to write [p, p + len) itself (ReadFile/fread into a guest
// buffer, DirectStorage destinations). A kernel write to an armed (read-only) view fails instead
// of faulting into the tracker, so while the scope lives the views it covers stay writable and are
// reported written, and they are stamped written again when it ends.
class HostWriteScope {
public:
    HostWriteScope(const void* p, uint64_t len);
    ~HostWriteScope();
    HostWriteScope(const HostWriteScope&) = delete;
    HostWriteScope& operator=(const HostWriteScope&) = delete;

private:
    uintptr_t lo_ = 0;
    uintptr_t hi_ = 0;
};

}  // namespace playgta6::os
