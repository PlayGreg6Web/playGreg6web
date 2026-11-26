#include "os/xbox_scarlett.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <condition_variable>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "os/guest_heap.h"
#include "os/guest_window.h"
#include "os/httpfs_mailbox.h"
#include "os/vmem.h"
#include "os/win32_kernel.h"
#include "os/win32_loader.h"
#include "os/surface_blocks.h"
#include "os/xmem_windows.h"
#include "guest/com_bridge.h"

namespace playgta6::os::scarlett {

namespace {

using gdk::E_FAIL_HR;
using gdk::E_INVALIDARG_HR;
using gdk::E_OUTOFMEMORY_HR;
using gdk::E_POINTER_HR;
using gdk::S_OK_HR;

constexpr size_t kMinAlignment = 16;
constexpr size_t kMaxHeapAlignment = 2 * 1024 * 1024;  // 2 MiB
constexpr uint32_t kDefaultStagingBufferSize = 32u * 1024u * 1024u;
constexpr uint32_t kMinStagingBufferSize = 64u * 1024u;
constexpr uint32_t kMaxStagingBufferSize = 256u * 1024u * 1024u;

uint64_t steady_microseconds() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(now).count());
}

// All XMem memory comes from the guest address space (os/guest_heap.h, os/vmem.h): heap blocks and
// physical-page backing from the guest heap (HeapFamily::XMem), XMemVirtualAlloc regions straight
// from the arena. The shims below hand out host pointers (the import bridge translates top-level
// results to guest VAs) and convert back with guest_of() when looking blocks up.
VirtualMemoryArena& xmem_arena() {
    return *win32::get_vmem_arena();
}

uint64_t guest_of(const void* host) {
    return host == nullptr ? 0 : xmem_arena().guest_address(host);
}

void* host_of(uint64_t va) {
    return va == 0 ? nullptr : xmem_arena().translate(va);
}

// O29: the guest VA of a pointer argument as the import thunks translated it. With the
// translated CPU window on, a window VA arrives as the host page backing its slot (or, for an
// unmapped slot, as va + kTrapDelta); both decode to the window VA. A page that also has an arena
// name (a pool-view page mapped into the window) decodes to the window VA: callers that key
// objects by host pointer look the raw pointer up first. Everything else: guest_of().
uint64_t guest_of_arg(const void* host) {
    if (host != nullptr && os::guest_window::enabled()) [[unlikely]] {
        const uint64_t h = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(host));
        if (os::guest_window::in_window(h - os::guest_window::kTrapDelta)) {
            return h - os::guest_window::kTrapDelta;
        }
        const uint64_t va = os::guest_window::guest_address(host);
        if (os::guest_window::in_window(va)) {
            return va;
        }
    }
    return guest_of(host);
}

void* aligned_alloc_bytes(size_t size, size_t alignment, bool zero = false) {
    GuestHeap& heap = default_guest_heap();
    const uint64_t va = heap.alloc(size,
                                   std::max(alignment, kMinAlignment),
                                   HeapFamily::XMem,
                                   zero ? kHeapZeroMemory : 0u);
    return heap.to_host(va);
}

void aligned_free_bytes(void* ptr, size_t alignment) {
    (void)alignment;
    if (ptr == nullptr) {
        return;
    }
    GuestHeap& heap = default_guest_heap();
    heap.free(heap.to_guest(ptr));
}

// Win32 PAGE_* value -> arena protection; console modifier bits (PAGE_GUARD, NOCACHE,
// WRITECOMBINE, GPU hints) in bits 8 and up are accepted and ignored.
bool xmem_protect(uint32_t protect, PageProtect& out) {
    switch (protect & 0xFFu) {
        case win32::PAGE_NOACCESS: out = PageProtect::NoAccess; return true;
        case win32::PAGE_READONLY: out = PageProtect::ReadOnly; return true;
        case win32::PAGE_READWRITE:
        case 0x08u: out = PageProtect::ReadWrite; return true;  // PAGE_WRITECOPY
        case win32::PAGE_EXECUTE_READ: out = PageProtect::ExecuteRead; return true;
        case win32::PAGE_EXECUTE_READWRITE:
        case 0x80u: out = PageProtect::ExecuteReadWrite; return true;  // PAGE_EXECUTE_WRITECOPY
        case 0x00u:
            // Only PAGE_GRAPHICS_* bits: the CPU view of a GPU-only mapping.
            if ((protect & 0x800u) != 0) {  // PAGE_GRAPHICS_NOACCESS
                out = PageProtect::NoAccess;
                return true;
            }
            if ((protect & 0x1000u) != 0) {  // PAGE_GRAPHICS_READONLY
                out = PageProtect::ReadOnly;
                return true;
            }
            if ((protect & 0x2000u) != 0) {  // PAGE_GRAPHICS_READWRITE
                out = PageProtect::ReadWrite;
                return true;
            }
            return false;
        default: return false;
    }
}

size_t decode_xmem_alignment(uint64_t attributes) {
    const uint64_t code = (attributes >> kXMemAttrAlignmentShift) & kXMemAttrAlignmentMask;
    if (code == 0) {
        return kMinAlignment;
    }
    if (code <= 20) {
        const size_t requested = size_t{1} << (code + 1u);
        return std::clamp(requested, kMinAlignment, kMaxHeapAlignment);
    }
    return static_cast<size_t>(kXMemPageSize64K);
}

uint64_t decode_physical_page_size(uint32_t flags) {
    if ((flags & kXMemFlagPage2M) != 0 || (flags & kXMemFlagLargePages) != 0) {
        return kXMemPageSize2M;
    }
    if ((flags & kXMemFlagPage4K) != 0) {
        return kXMemPageSize4K;
    }
    return kXMemPageSize64K;
}

uint64_t align_up_u64(uint64_t value, uint64_t alignment) {
    if (alignment == 0) {
        return value;
    }
    const uint64_t rem = value % alignment;
    return rem == 0 ? value : (value + (alignment - rem));
}

// ===========================================================================
// 1. xmem.dll internal state
// ===========================================================================

struct HeapAllocation {
    size_t size = 0;
    size_t alignment = kMinAlignment;
    uint64_t attributes = 0;
};

struct PhysicalPage {
    uint64_t handle = 0;
    uint64_t page_size = kXMemPageSize64K;
    uint64_t attributes = 0;
    // Guest heap block holding the page contents while the page has no home; 0 means "all zero,
    // or currently at its home" (see the physical page model below).
    uint64_t store_va = 0;
    // Guest VA where the page's bytes currently live (a real mapping), 0 if none.
    uint64_t home_va = 0;
    uint32_t mapped_slots = 0;  // real and phantom mappings
    // R19: translated-window slots this page is mapped at (they alias the page's bytes).
    std::vector<uint64_t> window_slots;
    // O36: native hosts: the page's shared memory object (one per page once it is mapped into a
    // shared reservation) and the runtime's own view of it. While it exists the page's bytes live
    // there, seen at every shared slot it is mapped at; home_va and store_va are unused.
    void* shared = nullptr;
    uint8_t* view = nullptr;
    // O29: the page's first real mapping (the title maps every pool page into its pool view
    // first), 0 if none. A CPU-window slot of a page whose pool_va lies outside that window
    // aliases those bytes under the translated window, so two such window slots are
    // host-contiguous exactly when their pages' pool VAs are adjacent.
    uint64_t pool_va = 0;
};

struct VirtualRegion {
    void* base = nullptr;
    size_t size = 0;
    size_t alignment = static_cast<size_t>(kXMemPageSize64K);
    uint32_t allocation_type = 0;
    uint32_t protect = win32::PAGE_READWRITE;
    bool committed = false;
    bool phantom = false;     // O27: placed in the phantom space (no guest memory)
    bool cpu_window = false;  // O27: a CPU window (xmem_windows.h)
    bool translated = false;  // R19: the CPU window at os/guest_window.h's translated range
    bool shared = false;      // O36: in the arena's shared zone (slots are views of shared pages)
};

// A guest heap block that XMemMapPhysicalPages(nullptr, ...) created for its pages.
struct OwnedBlock {
    uint64_t size = 0;
    size_t alignment = static_cast<size_t>(kXMemPageSize64K);
};

struct XMemState {
    std::mutex mutex;
    XMemAllocHookFn alloc_hook = nullptr;
    XMemFreeHookFn free_hook = nullptr;
    uint64_t guest_hook_table = 0;   // exported __XMemAllocationHooks__ ({alloc, free, ...})
    uint64_t guest_alloc_routine = 0;  // XMemSetAllocationHooks
    uint64_t guest_free_routine = 0;

    std::unordered_map<void*, HeapAllocation> heap_allocs;
    std::unordered_map<uint64_t, std::shared_ptr<PhysicalPage>> physical_pages;
    std::unordered_map<void*, VirtualRegion> virtual_regions;
    std::map<uint64_t, void*> region_index;     // guest base -> virtual_regions key
    std::map<uint64_t, OwnedBlock> owned_blocks;  // guest base -> block
    std::map<uint64_t, uint64_t> page_slots;      // mapped page VA (guest) -> page handle

    uint64_t next_page_handle = 0x1000ULL;
    uint64_t heap_allocated_bytes = 0;
    uint64_t physical_pages_bytes = 0;
    uint64_t virtual_committed_bytes = 0;
    uint64_t virtual_reserved_bytes = 0;
    uint64_t mapped_physical_bytes = 0;
    uint64_t peak_working_set_bytes = 0;
    uint64_t alias_conflicts = 0;  // a page mapped at a second real address while it had a home
    uint64_t zero_fills = 0;       // O36: slots zeroed by memset (fresh commits skip it)
    uint64_t alias_slots = 0;      // O36: ... mapped as an alias of its bytes instead (one memory)
    bool shared_exhausted_logged = false;
    bool translated_window_live = false;  // R19: the one translated window is in use
    // O29: CPU-window mappings of pages that live elsewhere (their pool view), and how often such
    // a mapping lands next to a mapped slot whose page is its pool-view neighbour
    // (host-contiguous under the translated window's aliasing).
    uint64_t cpu_window_maps = 0;
    uint64_t cpu_window_neighbour_pairs = 0;
    uint64_t cpu_window_neighbour_pairs_contiguous = 0;
    uint64_t cpu_window_neighbour_pairs_reversed = 0;  // pool-view neighbours in reverse order
    // O29: the translated window's run keeper (keep_window_run_locked).
    uint64_t run_kept = 0;      // a new slot already sat next to its neighbour's bytes
    uint64_t run_moves = 0;     // ... after moving the page's bytes (a frame swap)
    uint64_t run_breaks = 0;    // ... could not be kept contiguous
    uint64_t run_relocations = 0;  // whole runs moved to fresh frames (merges)
    bool keeping_runs = false;     // inside keep_window_run_locked (intermediate states are fine)
    uint64_t run_bytes_moved = 0;
    // Guest VA -> handle of the page whose bytes live there after a frame swap (validated on use:
    // the page must still exist and have that home).
    std::unordered_map<uint64_t, uint64_t> moved_homes;
    // O37: physical-page reservations handed to the title at a window range (guest_window
    // runtime ranges) under the window memory models: arena (backing) VA -> {window VA, bytes}.
    // XMem keeps working on the arena VAs; the title only sees the window VAs, so its stores into
    // the memory carry write marks. XMem's own writes there mark through mark_mirror_locked.
    struct Mirror {
        uint64_t window_va = 0;
        uint64_t bytes = 0;
    };
    std::map<uint64_t, Mirror> mirrors;
    bool mirror_exhausted_logged = false;
    bool pool_view_placed = false;  // O36: the first CPU window past the translated one was placed

    // R40: map/unmap data work outside the lock (map_slots). A plain map or unmap does its
    // bookkeeping under the lock, marks its slot (and the page whose bytes move) in flight, drops
    // the lock for the commit / copy / zeroing / shared-view work, and retakes it to finish. An
    // operation on an in-flight slot or page waits for it; anything else that touches page bytes
    // (the other map/unmap shapes, VirtualFree, XMemFreePhysicalPages, ...) waits until nothing is
    // in flight (wait_idle_locked), and new in-flight work waits while such an operation does.
    std::condition_variable idle_cv;
    std::unordered_set<uint64_t> busy_slots;  // guest VAs
    std::unordered_set<uint64_t> busy_pages;  // page handles
    uint32_t idle_waiters = 0;

    void update_peak_locked() {
        const uint64_t current =
            heap_allocated_bytes + physical_pages_bytes + virtual_committed_bytes;
        if (current > peak_working_set_bytes) {
            peak_working_set_bytes = current;
        }
    }

    VirtualRegion* find_virtual_region_locked(void* ptr, size_t needed_bytes) {
        if (ptr == nullptr) {
            return nullptr;
        }
        auto it = virtual_regions.find(ptr);
        if (it == virtual_regions.end()) {
            return nullptr;
        }
        if (it->second.size < needed_bytes) {
            return nullptr;
        }
        return &it->second;
    }

    // Region containing the guest range [va, va + len), if any.
    VirtualRegion* region_containing_locked(uint64_t va, uint64_t len);
    // Owned block containing [va, va + len): its base, or 0.
    uint64_t block_containing_locked(uint64_t va, uint64_t len) const {
        auto it = owned_blocks.upper_bound(va);
        if (it == owned_blocks.begin()) {
            return 0;
        }
        --it;
        return va - it->first < it->second.size && len <= it->second.size - (va - it->first) ? it->first : 0;
    }

    // Guest memory belongs to the arena; nothing to give back at process exit (the arena may
    // already be gone during static destruction).
};

XMemState& xmem_state() {
    static XMemState state;
    return state;
}

// Physical page model (O24, reworked in O27 for the title's call shapes)
// ----------------------------------------------------------------------
// XMemAllocatePhysicalPages(attributes, &count, pfns) returns page frame numbers for
// runtime-tracked pages; the pages own no memory until they hold data. Frame numbers count 4 KiB
// frames, as on the console (O30): a 64 KiB page's number is a multiple of 16 and the next page's
// is 16 higher. The title relies on it: its streaming code keeps pfn >> 4 per page and passes
// (pfn >> 4) << 4 to XMemMapPhysicalPagesScatter, so numbers 1 apart would collapse 16 pages
// into one. Pages are mapped one 64 KiB
// slot at a time: XMemMapPhysicalPages(va, count, pfns) maps consecutive slots (a null pfn array
// unmaps them), XMemMapPhysicalPagesScatter(vas, count, pfns) maps one slot per VA (a null array
// or a 0 pfn unmaps that slot), both into XMemVirtualAlloc reservations (only the mapped pages are
// committed; unmapping decommits them). A page's bytes live in exactly one place: its home (the
// real slot it was last mapped at) or, without a home, a store block in the guest heap. Mapping a
// page at a new real slot moves the bytes there; unmapping its home moves them back to the store.
// Slots in phantom reservations (xmem_windows.h: GPU windows) never become homes and hold no
// bytes, so mapping a page into the CPU window and the GPU window keeps one copy. Mapping a page
// at two real slots at once (the title does: the same page at several addresses of a CPU window)
// must alias, as on the console: a write through one address is seen through the others. O36:
// under the window memory models a slot of a reservation handed out at a window range (mirrors,
// O37) shows the page's bytes through the window table (an alias slot, guest_window::
// map_alias_slot) instead of a copy; native arenas back such reservations with shared page
// views (see shared_page below). Anything else (an arena slot of a page that already has a home)
// still gets a copy and the newer slot becomes the home (counted in alias_conflicts).
VirtualRegion* XMemState::region_containing_locked(uint64_t va, uint64_t len) {
    auto it = region_index.upper_bound(va);
    if (it == region_index.begin()) {
        return nullptr;
    }
    --it;
    auto rit = virtual_regions.find(it->second);
    if (rit == virtual_regions.end()) {
        return nullptr;
    }
    VirtualRegion& r = rit->second;
    if (va - it->first >= r.size || len > r.size - (va - it->first)) {
        return nullptr;
    }
    return &r;
}

// A store block holding a copy of `n` bytes (0 when they are all zero: no store needed). Needs no
// XMem lock.
uint64_t copy_to_store(const uint8_t* bytes, uint64_t n) {
    const size_t len = static_cast<size_t>(n);
    if (std::none_of(bytes, bytes + len, [](uint8_t b) { return b != 0; })) {
        return 0;
    }
    GuestHeap& heap = default_guest_heap();
    const uint64_t va = heap.alloc(len, kPageSize, HeapFamily::XMem);
    if (va == 0) {
        std::fprintf(stderr, "[xmem] out of guest memory saving a physical page; contents lost\n");
        return 0;
    }
    std::memcpy(heap.to_host(va), bytes, len);
    return va;
}

void save_page_contents(PhysicalPage& page, const uint8_t* bytes) {
    if (page.store_va != 0) {
        default_guest_heap().free(page.store_va);
        page.store_va = 0;
    }
    page.store_va = copy_to_store(bytes, page.page_size);
}

void release_virtual_region(const VirtualRegion& region) {
    if (region.phantom || region.translated) {
        return;  // no arena reservation behind it
    }
    xmem_arena().release(guest_of(region.base));
}

// R19: host bytes of `page` (its home, else its store; a zeroed store is created for a page that
// has neither, so a translated window slot always has backing).
uint8_t* page_bytes_locked(PhysicalPage& page) {
    if (page.view != nullptr) {
        return page.view;
    }
    if (page.home_va != 0) {
        return static_cast<uint8_t*>(host_of(page.home_va));
    }
    if (page.store_va == 0) {
        GuestHeap& heap = default_guest_heap();
        const uint64_t va = heap.alloc(static_cast<size_t>(page.page_size), kXMemPageSize64K, HeapFamily::XMem);
        if (va == 0) {
            return nullptr;
        }
        std::memset(heap.to_host(va), 0, static_cast<size_t>(page.page_size));
        page.store_va = va;
    }
    return static_cast<uint8_t*>(host_of(page.store_va));
}

// O36: the page's shared memory object (native hosts), created on first use with the page's
// current bytes (its home's or store's), which from then on live only there. An arena slot that
// was its home keeps a copy (as any second real mapping does). Null when the host has none.
uint8_t* shared_bytes_locked(XMemState& st, PhysicalPage& page) {
    if (page.view != nullptr) {
        return page.view;
    }
    const uint64_t ps = page.page_size;
    void* object = create_shared_page(ps);
    uint8_t* view = object != nullptr ? view_shared_page(object, ps) : nullptr;
    if (view == nullptr) {
        release_shared_page(object);
        return nullptr;
    }
    if (page.home_va != 0) {
        std::memcpy(view, host_of(page.home_va), static_cast<size_t>(ps));
        page.home_va = 0;
        ++st.alias_conflicts;  // the old home keeps a copy
    } else if (page.store_va != 0) {
        std::memcpy(view, host_of(page.store_va), static_cast<size_t>(ps));
        default_guest_heap().free(page.store_va);
        page.store_va = 0;
    }
    page.shared = object;
    page.view = view;
    return view;
}

void release_shared_locked(PhysicalPage& page) {
    if (page.view != nullptr) {
        unview_shared_page(page.view, page.page_size);
        page.view = nullptr;
    }
    if (page.shared != nullptr) {
        release_shared_page(page.shared);
        page.shared = nullptr;
    }
}

// R19: points every translated-window slot of `page` at the page's current bytes (after its home
// or store moved).
void retarget_window_slots_locked(PhysicalPage& page) {
    if (page.window_slots.empty()) {
        return;
    }
    uint8_t* bytes = page_bytes_locked(page);
    for (uint64_t slot : page.window_slots) {
        if (slot >= guest_window::kRuntimeRangeBase) {
            // O36: an alias slot of a mirrored reservation: it shows the page wherever it lives.
            guest_window::map_alias_slot(slot, bytes);
            continue;
        }
        guest_window::map_slot(slot, bytes);
        // O29: the bytes moved (their home was unmapped or the page was mapped elsewhere): with
        // contiguous runs required, the slot's neighbours must still be host-adjacent.
        if (guest_window::contiguous_runs_required() && !xmem_state().keeping_runs) {
            const uint64_t ps = page.page_size;
            const bool left_ok = guest_window::slot_host(slot - ps) == nullptr ||
                                 static_cast<uint8_t*>(guest_window::slot_host(slot - ps)) + ps == bytes;
            const bool right_ok = guest_window::slot_host(slot + ps) == nullptr ||
                                  static_cast<uint8_t*>(guest_window::slot_host(slot + ps)) == bytes + ps;
            if (!left_ok || !right_ok) {
                guest_window::contiguity_violation(slot);
            }
        }
    }
}

// O29: the page whose bytes live at guest VA `va` (a real mapping's home), or null.
PhysicalPage* home_owner_locked(XMemState& st, uint64_t va) {
    if (auto it = st.moved_homes.find(va); it != st.moved_homes.end()) {
        auto pit = st.physical_pages.find(it->second);
        if (pit != st.physical_pages.end() && pit->second->home_va == va) {
            return pit->second.get();
        }
        st.moved_homes.erase(it);
    }
    if (auto sit = st.page_slots.find(va); sit != st.page_slots.end()) {
        auto pit = st.physical_pages.find(sit->second);
        if (pit != st.physical_pages.end() && pit->second->home_va == va) {
            return pit->second.get();
        }
    }
    return nullptr;
}

// O37: the window VA the title knows arena VA `va` of a mirrored reservation by (0: none).
uint64_t mirror_window_va_locked(const XMemState& st, uint64_t va) {
    auto it = st.mirrors.upper_bound(va);
    if (it == st.mirrors.begin()) {
        return 0;
    }
    --it;
    return va - it->first < it->second.bytes ? it->second.window_va + (va - it->first) : 0;
}

// O37: XMem changed the bytes at arena VA [va, va + len) itself (a page mapped, unmapped or moved
// there): marks the window slots the title sees them at.
void mark_mirror_locked(const XMemState& st, uint64_t va, uint64_t len) {
    if (st.mirrors.empty()) {
        return;
    }
    if (const uint64_t w = mirror_window_va_locked(st, va)) {
        guest_window::mark_written(w, static_cast<size_t>(len));
    }
}

// O37: the arena VA behind a guest VA the title passed: a window VA of a mirrored reservation
// becomes its arena VA; anything else is returned as it is.
uint64_t unmirror(uint64_t va) {
    if (void* h = guest_window::runtime_range_host(va)) {
        return guest_of(h);
    }
    return va;
}

// Removes the page mapped at slot `va` (guest VA); the caller holds the lock.
void unmap_slot_locked(XMemState& st, uint64_t va) {
    auto sit = st.page_slots.find(va);
    if (sit == st.page_slots.end()) {
        return;
    }
    auto pit = st.physical_pages.find(sit->second);
    st.page_slots.erase(sit);
    if (pit == st.physical_pages.end()) {
        return;
    }
    PhysicalPage& page = *pit->second;
    const uint64_t ps = page.page_size;
    if (page.mapped_slots > 0) {
        --page.mapped_slots;
    }
    st.mapped_physical_bytes -= ps;
    // O36: a view of the page in a shared reservation (native): it goes, the page stays.
    if (VirtualRegion* sr = st.region_containing_locked(va, ps); sr != nullptr && sr->shared) {
        xmem_arena().unmap_shared(va, ps);
        return;
    }
    // O36: an alias slot of a mirrored reservation: it shows its own (uncommitted) range memory
    // again; the page's bytes stay where they are.
    if (const uint64_t w = mirror_window_va_locked(st, va);
        w != 0 && std::find(page.window_slots.begin(), page.window_slots.end(), w) != page.window_slots.end()) {
        page.window_slots.erase(std::remove(page.window_slots.begin(), page.window_slots.end(), w),
                                page.window_slots.end());
        guest_window::map_slot(w, host_of(va));
        return;
    }
    if (VirtualRegion* tr = st.region_containing_locked(va, ps); tr != nullptr && tr->translated) {
        // R19: an alias of the page's bytes; nothing moves.
        page.window_slots.erase(std::remove(page.window_slots.begin(), page.window_slots.end(), va),
                                page.window_slots.end());
        guest_window::unmap_slot(va);
        xmem_windows::note_cpu_window_mapping(-static_cast<int64_t>(ps));
        return;
    }
    // The bytes living here (the page's own, or after a frame swap another page's, O29) go to a
    // store before the slot loses its memory.
    if (PhysicalPage* owner = page.home_va == va ? &page : home_owner_locked(st, va)) {
        // O30: a page that is still mapped at its first real slot (the title's pool view), whose
        // memory holds nobody else's bytes, goes back there instead of into a new store block:
        // one copy, and no store per page the title streams through its CPU window.
        const uint64_t back = owner->pool_va;
        const auto still = back != 0 && back != va ? st.page_slots.find(back) : st.page_slots.end();
        if (still != st.page_slots.end() && still->second == owner->handle &&
            home_owner_locked(st, back) == nullptr) {
            std::memcpy(host_of(back), host_of(va), static_cast<size_t>(owner->page_size));
            mark_mirror_locked(st, back, owner->page_size);
            owner->home_va = back;
        } else {
            save_page_contents(*owner, static_cast<const uint8_t*>(host_of(va)));
            owner->home_va = 0;
        }
        retarget_window_slots_locked(*owner);
    }
    mark_mirror_locked(st, va, ps);  // O37: the title's view of the slot loses its bytes
    VirtualRegion* region = st.region_containing_locked(va, ps);
    if (region != nullptr && !region->phantom) {
        if (!region->committed) {
            xmem_arena().decommit(va, ps);
            st.virtual_committed_bytes -= ps;
        }
        if (region->cpu_window) {
            xmem_windows::note_cpu_window_mapping(-static_cast<int64_t>(ps));
        }
    }
}

uint64_t region_guest_base(const VirtualRegion& region) {
    return region.translated ? guest_window::kWindowBase : guest_of(region.base);
}

// Whether `page` lives outside `region` (its first mapping is elsewhere): its slots in `region`
// alias those bytes under the translated window.
bool aliases_into(const PhysicalPage& page, uint64_t base, const VirtualRegion& region) {
    return page.pool_va != 0 && page.pool_va - base >= region.size;
}

// O29: a page was just mapped at CPU-window slot `va` of `region`: count the mapping (if the page
// lives elsewhere) and its mapped neighbours that do too, and whether each neighbour's page is the
// page's pool-view neighbour.
void note_cpu_window_map_locked(XMemState& st, uint64_t va, const PhysicalPage& page, const VirtualRegion& region) {
    const uint64_t base = region_guest_base(region);
    if (!aliases_into(page, base, region)) {
        return;
    }
    ++st.cpu_window_maps;
    const uint64_t ps = page.page_size;
    for (const int64_t step : {-1, 1}) {
        const uint64_t nva = va + static_cast<uint64_t>(step * static_cast<int64_t>(ps));
        if (nva - base >= region.size) {
            continue;
        }
        const auto sit = st.page_slots.find(nva);
        if (sit == st.page_slots.end()) {
            continue;
        }
        const auto pit = st.physical_pages.find(sit->second);
        if (pit == st.physical_pages.end() || !aliases_into(*pit->second, base, region)) {
            continue;
        }
        ++st.cpu_window_neighbour_pairs;
        if (pit->second->pool_va - page.pool_va == nva - va) {
            ++st.cpu_window_neighbour_pairs_contiguous;
        } else if (page.pool_va - pit->second->pool_va == nva - va) {
            ++st.cpu_window_neighbour_pairs_reversed;
        }
    }
}

// O29: the guest VA of `page`'s bytes (its home, else its store), 0 if it has neither yet.
uint64_t page_position(const PhysicalPage& page) {
    return page.home_va != 0 ? page.home_va : page.store_va;
}

// O29: the page mapped at translated-window slot `va`, or null.
PhysicalPage* window_page_locked(XMemState& st, uint64_t va) {
    if (!guest_window::in_window(va)) {
        return nullptr;
    }
    auto sit = st.page_slots.find(va);
    if (sit == st.page_slots.end()) {
        return nullptr;
    }
    auto pit = st.physical_pages.find(sit->second);
    return pit != st.physical_pages.end() ? pit->second.get() : nullptr;
}

// O29: moves `page`'s bytes to guest VA `want`, the home of `other` (a page in no window run),
// whose bytes go where `page`'s were (its home or its store): two 64 KiB copies.
void swap_frames_locked(XMemState& st, PhysicalPage& page, PhysicalPage& other, uint64_t want) {
    const uint64_t ps = page.page_size;
    const uint64_t from = page_position(page);
    std::vector<uint8_t> tmp(static_cast<size_t>(ps));
    std::memcpy(tmp.data(), host_of(want), tmp.size());
    std::memcpy(host_of(want), host_of(from), tmp.size());
    std::memcpy(host_of(from), tmp.data(), tmp.size());
    mark_mirror_locked(st, want, ps);
    mark_mirror_locked(st, from, ps);
    other.home_va = page.home_va;  // 0 when `page` lived in a store
    if (page.home_va == 0) {
        other.store_va = page.store_va;  // the store block now holds `other`'s bytes
        page.store_va = 0;
    }
    page.home_va = want;
    st.moved_homes[want] = page.handle;
    if (other.home_va != 0) {
        st.moved_homes[other.home_va] = other.handle;
    }
    ++st.run_moves;
    st.run_bytes_moved += 2 * ps;
    retarget_window_slots_locked(page);
    retarget_window_slots_locked(other);
}

// O31: the reservation whose slots the keeper may move `page`'s bytes between: the one holding the
// page's first real mapping (pool_va), the title's pool view, which it never reads
// (os/xmem_windows.h). Null when the page has none. Bytes at any other real mapping (the title's
// XMem heaps, mapped right after the pool view) stay where they are: the title reads them there.
const VirtualRegion* frame_region_locked(XMemState& st, const PhysicalPage& page) {
    if (page.pool_va == 0) {
        return nullptr;
    }
    const VirtualRegion* r = st.region_containing_locked(page.pool_va, 1);
    return r != nullptr && !r->translated && !r->phantom ? r : nullptr;
}

// O31: whether guest VA `va` is a frame `page`'s bytes may move to or from.
bool is_frame_locked(XMemState& st, uint64_t va, const PhysicalPage& page) {
    const VirtualRegion* r = frame_region_locked(st, page);
    return r != nullptr && st.region_containing_locked(va, 1) == r;
}

// O29: puts `page`'s bytes at `want` (swapping with the page living there) unless that page is in
// a window run or `page`'s bytes also serve another window slot. False if not possible.
bool place_at_locked(XMemState& st, PhysicalPage& page, uint64_t want) {
    if (page_position(page) == want) {
        return true;
    }
    PhysicalPage* other = home_owner_locked(st, want);
    if (page.window_slots.size() > 1 || other == nullptr || other == &page || !other->window_slots.empty() ||
        other->page_size != page.page_size || !is_frame_locked(st, want, page) ||
        (page.home_va != 0 && !is_frame_locked(st, page.home_va, page))) {
        return false;
    }
    swap_frames_locked(st, page, *other, want);
    return true;
}

// O29: merging runs. Moves the pages of the window run [first, first + n slots) into n consecutive
// frames (homes of pages that are in no window run) of a real mapping outside the window, in slot
// order. False if a page of the run serves another window slot or no such frames exist.
// O31: all within the members' pool view (frame_region_locked).
bool relocate_run_locked(XMemState& st, uint64_t first, uint64_t n, uint64_t ps) {
    std::vector<PhysicalPage*> members;
    members.reserve(static_cast<size_t>(n));
    const VirtualRegion* frames = nullptr;
    for (uint64_t i = 0; i < n; ++i) {
        PhysicalPage* m = window_page_locked(st, first + i * ps);
        if (m == nullptr || m->window_slots.size() != 1 || m->page_size != ps) {
            return false;
        }
        const VirtualRegion* r = frame_region_locked(st, *m);
        if (r == nullptr || (frames != nullptr && r != frames) ||
            (m->home_va != 0 && st.region_containing_locked(m->home_va, 1) != r)) {
            return false;
        }
        frames = r;
        members.push_back(m);
    }
    // First fit: n consecutive guest VAs of real mappings outside the window whose bytes belong
    // to pages in no window run.
    uint64_t start = 0;
    uint64_t found = 0;
    uint64_t prev = 0;
    for (const auto& [va, handle] : st.page_slots) {
        if (guest_window::in_window(va)) {
            continue;
        }
        PhysicalPage* owner = home_owner_locked(st, va);
        const bool free = owner != nullptr && owner->window_slots.empty() && owner->page_size == ps &&
                          st.region_containing_locked(va, 1) == frames;
        if (!free) {
            found = 0;
            continue;
        }
        if (found == 0 || va != prev + ps) {
            start = va;
            found = 0;
        }
        prev = va;
        if (++found == n) {
            break;
        }
    }
    if (found != n) {
        return false;
    }
    for (uint64_t i = 0; i < n; ++i) {
        const uint64_t want = start + i * ps;
        PhysicalPage* other = home_owner_locked(st, want);
        if (other == nullptr) {
            return false;  // cannot happen: checked above, and members never own these frames
        }
        swap_frames_locked(st, *members[static_cast<size_t>(i)], *other, want);
    }
    ++st.run_relocations;
    return true;
}

// O29: window-centric placement. Under the translated window a slot aliases its page's bytes, so
// two adjacent slots are host-contiguous only when their pages' bytes are. `page` was just mapped
// at window slot `va` (already listed in its window_slots). Runs on either side are contiguous
// already (this keeps them so), so `page`'s bytes must sit right after the left run's last page
// and right before the right run's first: moved there by a frame swap with the page living at that
// position when needed. When the two sides do not line up, or that page is itself in a run, the
// whole merged run moves to consecutive free frames (merging runs copies). What cannot be kept
// contiguous - a page whose bytes serve two window slots, no free frames - is a break: counted,
// and reported to guest_window (whose handler stops the guest). Active only while contiguous runs
// are required.
void keep_window_run_locked(XMemState& st, uint64_t va, PhysicalPage& page) {
    // Only code lifted with contiguous runs needs it: moving pages leaves the pool view's VAs
    // naming other pages' bytes (the title never reads its pool view, os/xmem_windows.h), which
    // the plain windowed model has no reason to accept.
    if (!guest_window::contiguous_runs_required()) {
        return;
    }
    struct Keeping {
        XMemState& st;
        explicit Keeping(XMemState& s) : st(s) { st.keeping_runs = true; }
        ~Keeping() { st.keeping_runs = false; }
    } keeping(st);
    const uint64_t ps = page.page_size;
    PhysicalPage* left = window_page_locked(st, va - ps);
    PhysicalPage* right = window_page_locked(st, va + ps);
    if (left == nullptr && right == nullptr) {
        return;
    }
    if (left != nullptr) {
        page_bytes_locked(*left);
    }
    if (right != nullptr) {
        page_bytes_locked(*right);
    }
    page_bytes_locked(page);
    const uint64_t want = left != nullptr ? page_position(*left) + ps : page_position(*right) - ps;
    const bool sides_line_up = left == nullptr || right == nullptr || page_position(*right) == page_position(*left) + 2 * ps;
    if (sides_line_up && page_position(page) == want) {
        ++st.run_kept;
        return;
    }
    if (sides_line_up && ps == kXMemPageSize64K && place_at_locked(st, page, want)) {
        return;
    }
    // Merge: move the whole run around `va` to fresh frames.
    uint64_t first = va;
    while (window_page_locked(st, first - ps) != nullptr) {
        first -= ps;
    }
    uint64_t last = va;
    while (window_page_locked(st, last + ps) != nullptr) {
        last += ps;
    }
    if (ps == kXMemPageSize64K && relocate_run_locked(st, first, (last - first) / ps + 1, ps)) {
        return;
    }
    ++st.run_breaks;
    guest_window::contiguity_violation(va);
}

// Maps `page` at slot `va` (guest VA); the caller validated the page and holds the lock.
bool map_slot_locked(XMemState& st, uint64_t va, PhysicalPage& page) {
    const uint64_t ps = page.page_size;
    if ((va % std::min<uint64_t>(ps, kXMemPageSize64K)) != 0) {
        return false;
    }
    VirtualRegion* region = st.region_containing_locked(va, ps);
    const bool in_block = region == nullptr && st.block_containing_locked(va, ps) != 0;
    if (region == nullptr && !in_block) {
        return false;
    }
    unmap_slot_locked(st, va);
    st.page_slots[va] = page.handle;
    ++page.mapped_slots;
    st.mapped_physical_bytes += ps;
    if (region != nullptr && region->phantom) {
        return true;  // address only: no bytes move
    }
    if (region != nullptr && region->translated) {
        // R19: the window slot aliases the page's bytes (its home or store) through the table.
        uint8_t* bytes = page_bytes_locked(page);
        if (bytes == nullptr) {
            st.page_slots.erase(va);
            --page.mapped_slots;
            st.mapped_physical_bytes -= ps;
            return false;
        }
        page.window_slots.push_back(va);
        keep_window_run_locked(st, va, page);  // O29: may move the page's bytes next to a neighbour's
        bytes = page_bytes_locked(page);
        retarget_window_slots_locked(page);
        guest_window::map_slot(va, bytes);
        xmem_windows::note_cpu_window_mapping(static_cast<int64_t>(ps));
        note_cpu_window_map_locked(st, va, page, *region);
        return true;
    }
    // O36: the page already lives at another real slot and this slot belongs to a mirrored
    // reservation: its window VA shows those bytes (one copy, aliased as on the console).
    if (page.home_va != 0 && page.home_va != va) {
        if (const uint64_t w = mirror_window_va_locked(st, va)) {
            uint8_t* bytes = page_bytes_locked(page);
            page.window_slots.push_back(w);
            guest_window::map_alias_slot(w, bytes);
            ++st.alias_slots;
            return true;
        }
    }
    // O36: a shared reservation (native): the slot is a view of the page's shared memory object,
    // one memory for every address the page is mapped at.
    if (region != nullptr && region->shared) {
        uint8_t* bytes = (ps % kXMemPageSize64K) == 0 ? shared_bytes_locked(st, page) : nullptr;
        if (bytes == nullptr || !xmem_arena().map_shared(va, page.shared, ps)) {
            st.page_slots.erase(va);
            --page.mapped_slots;
            st.mapped_physical_bytes -= ps;
            return false;
        }
        if (page.mapped_slots > 1) {
            ++st.alias_slots;
        }
        if (page.pool_va == 0) {
            page.pool_va = va;
        }
        if (region->cpu_window) {
            xmem_windows::note_cpu_window_mapping(static_cast<int64_t>(ps));
            note_cpu_window_map_locked(st, va, page, *region);
        }
        retarget_window_slots_locked(page);
        return true;
    }
    // O36: a slot committed just now reads as zero already (the arena zero-fills fresh commits;
    // in the browser untouched linear memory is zero): zeroing it again would make every mapped
    // page resident, e.g. the whole multi-GiB pool view the title maps once at startup.
    bool fresh_zero = false;
    if (region != nullptr && !region->committed) {
        PageProtect pp = PageProtect::ReadWrite;
        xmem_protect(region->protect, pp);
        if (pp == PageProtect::NoAccess) {
            pp = PageProtect::ReadWrite;
        }
        uint64_t newly = 0;
        if (!xmem_arena().commit(va, ps, pp, &newly)) {
            st.page_slots.erase(va);
            --page.mapped_slots;
            st.mapped_physical_bytes -= ps;
            return false;
        }
        st.virtual_committed_bytes += ps;
        fresh_zero = newly == ps;
    }
    auto* dst = static_cast<uint8_t*>(host_of(va));
    if (page.view != nullptr) {
        // O36: the page lives in its shared memory object: this arena slot gets a copy.
        std::memcpy(dst, page.view, static_cast<size_t>(ps));
        ++st.alias_conflicts;
        if (page.pool_va == 0) {
            page.pool_va = va;
        }
        return true;
    }
    if (page.home_va != 0 && page.home_va != va) {
        std::memcpy(dst, host_of(page.home_va), static_cast<size_t>(ps));
        ++st.alias_conflicts;
    } else if (page.store_va != 0) {
        std::memcpy(dst, host_of(page.store_va), static_cast<size_t>(ps));
        default_guest_heap().free(page.store_va);
        page.store_va = 0;
    } else if (page.home_va == 0 && !fresh_zero) {
        std::memset(dst, 0, static_cast<size_t>(ps));
        ++st.zero_fills;
    }
    if (page.pool_va == 0) {
        page.pool_va = va;
    }
    page.home_va = va;
    mark_mirror_locked(st, va, ps);  // O37: the page's bytes appeared here
    if (region != nullptr && region->cpu_window) {
        xmem_windows::note_cpu_window_mapping(static_cast<int64_t>(ps));
        note_cpu_window_map_locked(st, va, page, *region);
    }
    retarget_window_slots_locked(page);
    return true;
}

// R40: waits (dropping the lock meanwhile) until no map/unmap has data work in flight; new
// in-flight work waits while this does. For anything that touches page bytes or slots other
// than through the in-flight paths below.
void wait_idle_locked(XMemState& st, std::unique_lock<std::mutex>& lock) {
    if (st.busy_slots.empty()) {
        return;
    }
    ++st.idle_waiters;
    st.idle_cv.wait(lock, [&] { return st.busy_slots.empty(); });
    if (--st.idle_waiters == 0) {
        st.idle_cv.notify_all();  // in-flight work held back by this wait may start
    }
}

void finish_in_flight_locked(XMemState& st, uint64_t va, uint64_t handle) {
    st.busy_slots.erase(va);
    if (handle != 0) {
        st.busy_pages.erase(handle);
    }
    st.idle_cv.notify_all();
}

enum class FastResult { NotTaken, Ok, Failed };

// R40: maps `page` at slot `va` with the data work outside the lock, for the plain shapes: a
// free slot of an arena reservation (commit + the page's bytes from its store, or zeroes) or of a
// shared reservation (the page's shared view, created on first use, mapped at the slot), the page
// living in no real slot (no home) and shown in no window slot. Anything else is NotTaken
// (map_slot_locked handles it). The caller waited until neither `va` nor the page is in flight.
FastResult map_slot_fast(XMemState& st, std::unique_lock<std::mutex>& lock, uint64_t va,
                         const std::shared_ptr<PhysicalPage>& page_ref) {
    PhysicalPage& page = *page_ref;
    const uint64_t ps = page.page_size;
    if ((va % std::min<uint64_t>(ps, kXMemPageSize64K)) != 0) {
        return FastResult::NotTaken;
    }
    VirtualRegion* region = st.region_containing_locked(va, ps);
    if (region == nullptr || region->phantom || region->translated || st.page_slots.count(va) != 0 ||
        page.home_va != 0 || !page.window_slots.empty()) {
        return FastResult::NotTaken;
    }
    const bool shared = region->shared;
    if (shared ? (ps % kXMemPageSize64K) != 0 : page.view != nullptr) {
        return FastResult::NotTaken;
    }
    const bool cpu_window = region->cpu_window;
    const uint64_t handle = page.handle;
    st.page_slots[va] = handle;
    ++page.mapped_slots;
    st.mapped_physical_bytes += ps;
    const uint64_t old_pool = page.pool_va;
    // Shared: the page's view, or the store its new view starts from. Arena: the store the slot's
    // bytes come from (zeroes without one).
    void* object = page.shared;
    const bool create = shared && page.view == nullptr;
    const uint64_t store = shared && !create ? 0 : page.store_va;
    page.store_va = shared && !create ? page.store_va : 0;
    const bool commit = !shared && !region->committed;
    PageProtect pp = PageProtect::ReadWrite;
    if (commit) {
        xmem_protect(region->protect, pp);
        if (pp == PageProtect::NoAccess) {
            pp = PageProtect::ReadWrite;
        }
        st.virtual_committed_bytes += ps;
    }
    if (!shared) {
        if (page.pool_va == 0) {
            page.pool_va = va;
        }
        page.home_va = va;
    }
    st.busy_slots.insert(va);
    st.busy_pages.insert(handle);
    lock.unlock();

    bool ok = true;
    bool fresh_zero = false;  // O36: slot committed just now, already zero
    bool zero_filled = false;
    uint8_t* view = nullptr;
    if (shared) {
        if (create) {
            object = create_shared_page(ps);
            view = object != nullptr ? view_shared_page(object, ps) : nullptr;
            if (view == nullptr) {
                release_shared_page(object);
                object = nullptr;
                ok = false;
            } else if (store != 0) {
                std::memcpy(view, host_of(store), static_cast<size_t>(ps));
                default_guest_heap().free(store);
            }
        }
        ok = ok && xmem_arena().map_shared(va, object, ps);
    } else {
        uint64_t newly = 0;
        ok = !commit || xmem_arena().commit(va, ps, pp, &newly);
        fresh_zero = commit && ok && newly == ps;
        if (ok) {
            auto* dst = static_cast<uint8_t*>(host_of(va));
            if (store != 0) {
                std::memcpy(dst, host_of(store), static_cast<size_t>(ps));
                default_guest_heap().free(store);
            } else if (!fresh_zero) {
                std::memset(dst, 0, static_cast<size_t>(ps));
                zero_filled = true;
            }
        }
    }

    lock.lock();
    if (zero_filled) {
        ++st.zero_fills;
    }
    if (create && view != nullptr) {
        page.shared = object;  // kept even if the slot could not show it (as shared_bytes_locked)
        page.view = view;
    } else if (create || (!shared && !ok)) {
        page.store_va = store;  // nothing moved
    }
    if (!ok) {
        st.page_slots.erase(va);
        --page.mapped_slots;
        st.mapped_physical_bytes -= ps;
        if (commit) {
            st.virtual_committed_bytes -= ps;
        }
        if (!shared) {
            page.home_va = 0;
            page.pool_va = old_pool;
        }
        finish_in_flight_locked(st, va, handle);
        return FastResult::Failed;
    }
    if (shared) {
        if (page.mapped_slots > 1) {
            ++st.alias_slots;
        }
        if (page.pool_va == 0) {
            page.pool_va = va;
        }
    } else {
        mark_mirror_locked(st, va, ps);  // O37: the page's bytes appeared here
    }
    if (cpu_window) {
        xmem_windows::note_cpu_window_mapping(static_cast<int64_t>(ps));
        if (const VirtualRegion* r = st.region_containing_locked(va, ps)) {
            note_cpu_window_map_locked(st, va, page, *r);
        }
    }
    finish_in_flight_locked(st, va, handle);
    return FastResult::Ok;
}

// R40: unmaps slot `va` with the data work outside the lock, for the plain shapes: a slot of a
// shared reservation (its view goes), or an arena slot that is its page's home (the bytes go to a
// store, the slot is decommitted) or a copy of a page living elsewhere. Anything else (alias and
// window slots, frame swaps, a pool-view page going back to its pool slot) is NotTaken.
FastResult unmap_slot_fast(XMemState& st, std::unique_lock<std::mutex>& lock, uint64_t va) {
    auto sit = st.page_slots.find(va);
    if (sit == st.page_slots.end()) {
        return FastResult::Ok;
    }
    auto pit = st.physical_pages.find(sit->second);
    if (pit == st.physical_pages.end()) {
        return FastResult::NotTaken;
    }
    PhysicalPage& page = *pit->second;
    const uint64_t ps = page.page_size;
    VirtualRegion* region = st.region_containing_locked(va, ps);
    if (region == nullptr || region->phantom || region->translated) {
        return FastResult::NotTaken;
    }
    const bool shared = region->shared;
    bool owner = false;
    if (!shared) {
        if (const uint64_t w = mirror_window_va_locked(st, va);
            w != 0 && std::find(page.window_slots.begin(), page.window_slots.end(), w) != page.window_slots.end()) {
            return FastResult::NotTaken;  // an alias slot
        }
        if (page.home_va == va) {
            const uint64_t back = page.pool_va;
            const auto still = back != 0 && back != va ? st.page_slots.find(back) : st.page_slots.end();
            if (!page.window_slots.empty() ||
                (still != st.page_slots.end() && still->second == page.handle && home_owner_locked(st, back) == nullptr)) {
                return FastResult::NotTaken;  // window slots to retarget, or back to the pool slot (O30)
            }
            owner = true;
        } else if (home_owner_locked(st, va) != nullptr) {
            return FastResult::NotTaken;  // another page's bytes live here (O29)
        }
    }
    const uint64_t handle = owner ? page.handle : 0;
    st.page_slots.erase(sit);
    if (page.mapped_slots > 0) {
        --page.mapped_slots;
    }
    st.mapped_physical_bytes -= ps;
    uint64_t old_store = 0;
    if (owner) {
        page.home_va = 0;
        old_store = page.store_va;
        page.store_va = 0;
    }
    const bool decommit = !shared && !region->committed;
    if (decommit) {
        st.virtual_committed_bytes -= ps;
    }
    if (!shared && region->cpu_window) {
        xmem_windows::note_cpu_window_mapping(-static_cast<int64_t>(ps));
    }
    st.busy_slots.insert(va);
    if (owner) {
        st.busy_pages.insert(handle);
    }
    lock.unlock();

    uint64_t new_store = 0;
    if (shared) {
        xmem_arena().unmap_shared(va, ps);
    } else {
        if (owner) {
            if (old_store != 0) {
                default_guest_heap().free(old_store);
            }
            new_store = copy_to_store(static_cast<const uint8_t*>(host_of(va)), ps);
        }
        if (decommit) {
            xmem_arena().decommit(va, ps);
        }
    }

    lock.lock();
    if (owner) {
        page.store_va = new_store;  // the page is in flight: it still exists
    }
    if (!shared) {
        mark_mirror_locked(st, va, ps);  // O37: the title's view of the slot loses its bytes
    }
    finish_in_flight_locked(st, va, handle);
    return FastResult::Ok;
}

// Maps (pfn != 0) or unmaps (pfn == 0 / null array) one slot per entry of `vas`; all-or-nothing
// validation of the page handles first. R40: plain entries do their data work outside the lock
// (map_slot_fast / unmap_slot_fast); the rest runs under it with nothing in flight.
bool map_slots(const uint64_t* vas, size_t count, const uint64_t* pfns) {
    auto& st = xmem_state();
    std::unique_lock<std::mutex> lock(st.mutex);
    if (pfns != nullptr) {
        for (size_t i = 0; i < count; ++i) {
            if (pfns[i] != 0 && st.physical_pages.find(pfns[i]) == st.physical_pages.end()) {
                win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
                return false;
            }
        }
    }
    bool ok = true;
    for (size_t i = 0; i < count; ++i) {
        const uint64_t va = unmirror(vas[i]);  // O37: a mirrored reservation's window VA
        const uint64_t handle = pfns == nullptr ? 0 : pfns[i];
        // Wait until this slot, the page going there and the page leaving it are out of flight,
        // and no operation is waiting for everything to be.
        st.idle_cv.wait(lock, [&] {
            if (st.idle_waiters != 0) {
                return false;  // they wake once nothing is in flight, and notify when done
            }
            if (st.busy_slots.empty()) {
                return true;
            }
            if (st.busy_slots.count(va) != 0 ||
                (handle != 0 && st.busy_pages.count(handle) != 0)) {
                return false;
            }
            const auto sit = st.page_slots.find(va);
            return sit == st.page_slots.end() || st.busy_pages.count(sit->second) == 0;
        });
        if (handle == 0) {
            if (unmap_slot_fast(st, lock, va) == FastResult::NotTaken) {
                wait_idle_locked(st, lock);
                unmap_slot_locked(st, va);
            }
            continue;
        }
        auto pit = st.physical_pages.find(handle);
        if (pit == st.physical_pages.end()) {
            ok = false;  // freed meanwhile
            continue;
        }
        const std::shared_ptr<PhysicalPage> page = pit->second;
        const FastResult fast = map_slot_fast(st, lock, va, page);
        if (fast == FastResult::NotTaken) {
            wait_idle_locked(st, lock);
            pit = st.physical_pages.find(handle);
            if (pit == st.physical_pages.end() || !map_slot_locked(st, va, *pit->second)) {
                ok = false;
            }
        } else if (fast == FastResult::Failed) {
            ok = false;
        }
    }
    st.update_peak_locked();
    if (!ok) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
    }
    return ok;
}

// Unmaps every slot in [base, base + size).
void unmap_range_locked(XMemState& st, uint64_t base, uint64_t size) {
    std::vector<uint64_t> slots;
    for (auto it = st.page_slots.lower_bound(base); it != st.page_slots.end() && it->first < base + size; ++it) {
        slots.push_back(it->first);
    }
    for (uint64_t va : slots) {
        unmap_slot_locked(st, va);
    }
}

// ===========================================================================
// 2. XCRDAPI.dll (DirectStorage) internal state & COM implementations
// ===========================================================================

struct DStorageGlobalState {
    std::mutex mutex;
    std::unordered_map<std::string, std::vector<uint8_t>> virtual_files;
    DStorageHostOpenCallback host_open_cb;
    DStorageHostReadCallback host_read_cb;
    uint32_t staging_buffer_size = kDefaultStagingBufferSize;
};

DStorageGlobalState& dstorage_state() {
    static DStorageGlobalState state;
    return state;
}

enum class DStorageFileBackend {
    VirtualTable,
    HostCallback,
    Httpfs,
    UserdataOverlay,
};

class DStorageFileImpl final : public Pg6IDStorageFile {
public:
    DStorageFileImpl(std::string normalized_path,
                     DStorageFileBackend backend,
                     uint64_t file_size,
                     int32_t httpfs_handle = -1)
        : path_(std::move(normalized_path)),
          backend_(backend),
          file_size_(file_size),
          httpfs_handle_(httpfs_handle) {}

    ~DStorageFileImpl() override { Close(); }

    Pg6HResult QueryInterface(const Pg6Guid& riid, void** ppv_object) override {
        if (ppv_object == nullptr) {
            return E_POINTER_HR;
        }
        if (riid == kIidIUnknown || riid == kIidIDStorageFile) {
            AddRef();
            *ppv_object = static_cast<Pg6IDStorageFile*>(this);
            return S_OK_HR;
        }
        *ppv_object = nullptr;
        return E_NOINTERFACE_HR;
    }

    uint32_t AddRef() override { return ref_count_.fetch_add(1, std::memory_order_relaxed) + 1; }

    uint32_t Release() override {
        const uint32_t next = ref_count_.fetch_sub(1, std::memory_order_acq_rel) - 1;
        if (next == 0) {
            delete this;
        }
        return next;
    }

    void Close() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return;
        }
        closed_ = true;
        if (backend_ == DStorageFileBackend::Httpfs && httpfs_handle_ >= 0) {
            if (auto* client = win32::get_httpfs_client(); client != nullptr) {
                client->close(httpfs_handle_);
            }
            httpfs_handle_ = -1;
        }
    }

    Pg6HResult GetFileInformation(Pg6DStorageFileInfo* info) override {
        if (info == nullptr) {
            return E_POINTER_HR;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return gdk::E_HANDLE_HR;
        }
        info->file_size = file_size_;
        return S_OK_HR;
    }

    Pg6HResult read_at(uint64_t offset, void* dst, uint32_t size) {
        if (size == 0) {
            return S_OK_HR;
        }
        if (dst == nullptr) {
            return E_POINTER_HR;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return gdk::E_HANDLE_HR;
        }
        // R40: the host callback / httpfs backends may write the destination from the kernel.
        const HostWriteScope host_write(dst, size);

        switch (backend_) {
            case DStorageFileBackend::VirtualTable: {
                auto& ds = dstorage_state();
                std::lock_guard<std::mutex> glock(ds.mutex);
                auto it = ds.virtual_files.find(path_);
                if (it == ds.virtual_files.end()) {
                    return E_FILE_NOT_FOUND_HR;
                }
                const auto& bytes = it->second;
                if (offset > bytes.size() || size > bytes.size() - offset) {
                    return E_EOF_HR;
                }
                std::memcpy(dst, bytes.data() + offset, size);
                return S_OK_HR;
            }
            case DStorageFileBackend::HostCallback: {
                DStorageHostReadCallback read_cb;
                {
                    auto& ds = dstorage_state();
                    std::lock_guard<std::mutex> glock(ds.mutex);
                    read_cb = ds.host_read_cb;
                }
                if (!read_cb) {
                    return E_FAIL_HR;
                }
                const int64_t n = read_cb(path_, offset, dst, size);
                if (n < 0) {
                    return E_FAIL_HR;
                }
                if (static_cast<uint64_t>(n) < size) {
                    return E_EOF_HR;
                }
                return S_OK_HR;
            }
            case DStorageFileBackend::Httpfs: {
                auto* client = win32::get_httpfs_client();
                if (client == nullptr || httpfs_handle_ < 0) {
                    return E_FAIL_HR;
                }
                const int64_t n = client->read(httpfs_handle_, offset, dst, size);
                if (n < 0) {
                    return E_FAIL_HR;
                }
                if (static_cast<uint64_t>(n) < size) {
                    return E_EOF_HR;
                }
                return S_OK_HR;
            }
            case DStorageFileBackend::UserdataOverlay: {
                std::vector<uint8_t> data;
                if (!win32::read_userdata_file(path_, data)) {
                    return E_FILE_NOT_FOUND_HR;
                }
                if (offset > data.size() || size > data.size() - offset) {
                    return E_EOF_HR;
                }
                std::memcpy(dst, data.data() + offset, size);
                return S_OK_HR;
            }
        }
        return E_FAIL_HR;
    }

private:
    std::atomic<uint32_t> ref_count_{1};
    std::mutex mutex_;
    std::string path_;
    DStorageFileBackend backend_;
    uint64_t file_size_ = 0;
    int32_t httpfs_handle_ = -1;
    bool closed_ = false;
};

class DStorageStatusArrayImpl final : public Pg6IDStorageStatusArray {
public:
    explicit DStorageStatusArrayImpl(uint32_t capacity)
        : complete_(capacity, 0u), results_(capacity, S_OK_HR) {}

    Pg6HResult QueryInterface(const Pg6Guid& riid, void** ppv_object) override {
        if (ppv_object == nullptr) {
            return E_POINTER_HR;
        }
        if (riid == kIidIUnknown || riid == kIidIDStorageStatusArray) {
            AddRef();
            *ppv_object = static_cast<Pg6IDStorageStatusArray*>(this);
            return S_OK_HR;
        }
        *ppv_object = nullptr;
        return E_NOINTERFACE_HR;
    }

    uint32_t AddRef() override { return ref_count_.fetch_add(1, std::memory_order_relaxed) + 1; }

    uint32_t Release() override {
        const uint32_t next = ref_count_.fetch_sub(1, std::memory_order_acq_rel) - 1;
        if (next == 0) {
            delete this;
        }
        return next;
    }

    bool IsComplete(uint32_t index) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index >= complete_.size()) {
            return false;
        }
        return complete_[index] != 0u;
    }

    Pg6HResult GetHResult(uint32_t index) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index >= results_.size()) {
            return E_INVALIDARG_HR;
        }
        return results_[index];
    }

    void mark_pending(uint32_t index) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index < complete_.size()) {
            complete_[index] = 0u;
            results_[index] = gdk::E_PENDING_HR;
        }
    }

    void mark_complete(uint32_t index, Pg6HResult hr) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index < complete_.size()) {
            results_[index] = hr;
            complete_[index] = 1u;
        }
    }

private:
    std::atomic<uint32_t> ref_count_{1};
    std::mutex mutex_;
    std::vector<uint8_t> complete_;
    std::vector<Pg6HResult> results_;
};

class DStorageQueueImpl final : public Pg6IDStorageQueue {
public:
    explicit DStorageQueueImpl(const Pg6DStorageQueueDesc& desc)
        : capacity_(desc.capacity == 0 ? 256u : desc.capacity) {}

    ~DStorageQueueImpl() override { Close(); }

    Pg6HResult QueryInterface(const Pg6Guid& riid, void** ppv_object) override {
        if (ppv_object == nullptr) {
            return E_POINTER_HR;
        }
        if (riid == kIidIUnknown || riid == kIidIDStorageQueue) {
            AddRef();
            *ppv_object = static_cast<Pg6IDStorageQueue*>(this);
            return S_OK_HR;
        }
        *ppv_object = nullptr;
        return E_NOINTERFACE_HR;
    }

    uint32_t AddRef() override { return ref_count_.fetch_add(1, std::memory_order_relaxed) + 1; }

    uint32_t Release() override {
        const uint32_t next = ref_count_.fetch_sub(1, std::memory_order_acq_rel) - 1;
        if (next == 0) {
            delete this;
        }
        return next;
    }

    void EnqueueRequest(const Pg6DStorageRequest* request) override {
        if (request == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return;
        }
        QueueItem item;
        item.kind = ItemKind::Read;
        item.file = static_cast<DStorageFileImpl*>(request->file);
        if (item.file != nullptr) {
            item.file->AddRef();
        }
        item.offset = request->offset;
        item.size = request->size;
        item.destination = request->destination;
        item.destination_size =
            request->destination_size == 0 ? request->size : request->destination_size;
        item.cancellation_tag = request->cancellation_tag;
        pending_.push_back(std::move(item));
    }

    void EnqueueRead(Pg6IDStorageFile* file,
                     uint64_t offset,
                     void* destination,
                     uint32_t size) override {
        Pg6DStorageRequest req{};
        req.file = file;
        req.offset = offset;
        req.size = size;
        req.destination_size = size;
        req.destination = destination;
        EnqueueRequest(&req);
    }

    void EnqueueStatus(Pg6IDStorageStatusArray* status_array, uint32_t index) override {
        if (status_array == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return;
        }
        auto* impl = static_cast<DStorageStatusArrayImpl*>(status_array);
        impl->AddRef();
        impl->mark_pending(index);
        QueueItem item;
        item.kind = ItemKind::Status;
        item.status_array = impl;
        item.status_index = index;
        pending_.push_back(std::move(item));
    }

    void EnqueueSignal(Pg6Handle event_handle, uint64_t value) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return;
        }
        QueueItem item;
        item.kind = ItemKind::Signal;
        item.signal_handle = event_handle;
        item.signal_value = value;
        pending_.push_back(std::move(item));
    }

    void Submit() override {
        std::vector<QueueItem> batch;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) {
                return;
            }
            batch.swap(pending_);
        }

        Pg6HResult batch_hr = S_OK_HR;
        for (auto& item : batch) {
            switch (item.kind) {
                case ItemKind::Read: {
                    if (item.cancelled) {
                        if (item.file != nullptr) {
                            item.file->Release();
                        }
                        break;
                    }
                    Pg6HResult hr = S_OK_HR;
                    if (item.file == nullptr || (item.size > 0 && item.destination == nullptr)) {
                        hr = E_INVALIDARG_HR;
                    } else if (item.destination_size < item.size) {
                        hr = gdk::E_NOT_SUFFICIENT_BUFFER_HR;
                    } else {
                        hr = item.file->read_at(item.offset, item.destination, item.size);
                    }
                    if (gdk::failed(hr)) {
                        record_error(hr, item.offset, item.size);
                        if (gdk::succeeded(batch_hr)) {
                            batch_hr = hr;
                        }
                    }
                    if (item.file != nullptr) {
                        item.file->Release();
                    }
                    break;
                }
                case ItemKind::Status: {
                    if (item.status_array != nullptr) {
                        item.status_array->mark_complete(item.status_index, batch_hr);
                        item.status_array->Release();
                    }
                    batch_hr = S_OK_HR;
                    break;
                }
                case ItemKind::Signal: {
                    if (item.signal_handle != win32::kNullHandle) {
                        win32::SetEvent(item.signal_handle);
                    }
                    break;
                }
            }
        }
    }

    void CancelRequestsWithTag(uint64_t mask, uint64_t value) override {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& item : pending_) {
            if (item.kind == ItemKind::Read && (item.cancellation_tag & mask) == value) {
                item.cancelled = true;
            }
        }
    }

    void Close() override {
        std::vector<QueueItem> leftover;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) {
                return;
            }
            closed_ = true;
            leftover.swap(pending_);
        }
        for (auto& item : leftover) {
            if (item.file != nullptr) {
                item.file->Release();
            }
            if (item.status_array != nullptr) {
                item.status_array->mark_complete(item.status_index, gdk::E_ABORT_HR);
                item.status_array->Release();
            }
        }
    }

    uint32_t RetrieveErrorRecord(Pg6DStorageErrorRecord* record) override {
        std::lock_guard<std::mutex> lock(mutex_);
        const uint32_t count = error_record_.failure_count;
        if (record != nullptr) {
            *record = error_record_;
        }
        error_record_ = Pg6DStorageErrorRecord{};
        return count;
    }

    uint32_t QueryPendingCount() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<uint32_t>(pending_.size());
    }

private:
    enum class ItemKind {
        Read,
        Status,
        Signal,
    };

    struct QueueItem {
        ItemKind kind = ItemKind::Read;
        DStorageFileImpl* file = nullptr;
        uint64_t offset = 0;
        uint32_t size = 0;
        uint32_t destination_size = 0;
        void* destination = nullptr;
        uint64_t cancellation_tag = 0;
        bool cancelled = false;
        DStorageStatusArrayImpl* status_array = nullptr;
        uint32_t status_index = 0;
        Pg6Handle signal_handle = win32::kNullHandle;
        uint64_t signal_value = 0;
    };

    void record_error(Pg6HResult hr, uint64_t offset, uint32_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (error_record_.failure_count == 0) {
            error_record_.first_failure_hresult = hr;
            error_record_.first_failure_offset = offset;
            error_record_.first_failure_size = size;
        }
        ++error_record_.failure_count;
    }

    std::atomic<uint32_t> ref_count_{1};
    mutable std::mutex mutex_;
    uint32_t capacity_ = 256;
    bool closed_ = false;
    std::vector<QueueItem> pending_;
    Pg6DStorageErrorRecord error_record_{};
};

class DStorageFactoryImpl final : public Pg6IDStorageFactory {
public:
    Pg6HResult QueryInterface(const Pg6Guid& riid, void** ppv_object) override {
        if (ppv_object == nullptr) {
            return E_POINTER_HR;
        }
        if (riid == kIidIUnknown || riid == kIidIDStorageFactory) {
            AddRef();
            *ppv_object = static_cast<Pg6IDStorageFactory*>(this);
            return S_OK_HR;
        }
        *ppv_object = nullptr;
        return E_NOINTERFACE_HR;
    }

    uint32_t AddRef() override { return ref_count_.fetch_add(1, std::memory_order_relaxed) + 1; }

    uint32_t Release() override {
        const uint32_t next = ref_count_.fetch_sub(1, std::memory_order_acq_rel) - 1;
        if (next == 0) {
            delete this;
        }
        return next;
    }

    Pg6HResult CreateQueue(const Pg6DStorageQueueDesc* desc,
                           const Pg6Guid& riid,
                           void** ppv) override {
        if (ppv == nullptr) {
            return E_POINTER_HR;
        }
        *ppv = nullptr;
        if (desc == nullptr) {
            return E_INVALIDARG_HR;
        }
        if (riid != kIidIUnknown && riid != kIidIDStorageQueue) {
            return E_NOINTERFACE_HR;
        }
        auto* queue = new (std::nothrow) DStorageQueueImpl(*desc);
        if (queue == nullptr) {
            return E_OUTOFMEMORY_HR;
        }
        *ppv = static_cast<Pg6IDStorageQueue*>(queue);
        return S_OK_HR;
    }

    Pg6HResult OpenFile(const char16_t* path, const Pg6Guid& riid, void** ppv) override {
        if (ppv == nullptr) {
            return E_POINTER_HR;
        }
        *ppv = nullptr;
        if (path == nullptr) {
            return E_INVALIDARG_HR;
        }
        const std::string utf8 = win32::utf16_to_utf8(path);
        return OpenFileA(utf8.c_str(), riid, ppv);
    }

    Pg6HResult OpenFileA(const char* path, const Pg6Guid& riid, void** ppv) override {
        if (ppv == nullptr) {
            return E_POINTER_HR;
        }
        *ppv = nullptr;
        if (path == nullptr || path[0] == '\0') {
            return E_INVALIDARG_HR;
        }
        if (riid != kIidIUnknown && riid != kIidIDStorageFile) {
            return E_NOINTERFACE_HR;
        }

        const std::string normalized = win32::normalize_path(path);
        if (normalized.empty()) {
            return E_INVALIDARG_HR;
        }

        // 1. In-memory virtual file table (highest priority).
        {
            auto& ds = dstorage_state();
            std::lock_guard<std::mutex> lock(ds.mutex);
            auto it = ds.virtual_files.find(normalized);
            if (it != ds.virtual_files.end()) {
                auto* file = new (std::nothrow) DStorageFileImpl(
                    normalized, DStorageFileBackend::VirtualTable, it->second.size());
                if (file == nullptr) {
                    return E_OUTOFMEMORY_HR;
                }
                *ppv = static_cast<Pg6IDStorageFile*>(file);
                return S_OK_HR;
            }
        }

        // 2. Host read callback hook.
        {
            DStorageHostOpenCallback open_cb;
            DStorageHostReadCallback read_cb;
            {
                auto& ds = dstorage_state();
                std::lock_guard<std::mutex> lock(ds.mutex);
                open_cb = ds.host_open_cb;
                read_cb = ds.host_read_cb;
            }
            if (read_cb) {
                const int64_t size = open_cb ? open_cb(normalized) : 0;
                if (size >= 0) {
                    auto* file = new (std::nothrow) DStorageFileImpl(
                        normalized, DStorageFileBackend::HostCallback, static_cast<uint64_t>(size));
                    if (file == nullptr) {
                        return E_OUTOFMEMORY_HR;
                    }
                    *ppv = static_cast<Pg6IDStorageFile*>(file);
                    return S_OK_HR;
                }
            }
        }

        // 3. HTTPFS mailbox client.
        if (auto* client = win32::get_httpfs_client(); client != nullptr) {
            const int32_t handle = client->open(normalized);
            if (handle >= 0) {
                auto* file = new (std::nothrow)
                    DStorageFileImpl(normalized, DStorageFileBackend::Httpfs, 0, handle);
                if (file == nullptr) {
                    client->close(handle);
                    return E_OUTOFMEMORY_HR;
                }
                *ppv = static_cast<Pg6IDStorageFile*>(file);
                return S_OK_HR;
            }
        }

        // 4. /userdata overlay fallback.
        if (win32::userdata_file_exists(normalized)) {
            std::vector<uint8_t> data;
            if (win32::read_userdata_file(normalized, data)) {
                auto* file = new (std::nothrow) DStorageFileImpl(
                    normalized, DStorageFileBackend::UserdataOverlay, data.size());
                if (file == nullptr) {
                    return E_OUTOFMEMORY_HR;
                }
                *ppv = static_cast<Pg6IDStorageFile*>(file);
                return S_OK_HR;
            }
        }

        return E_FILE_NOT_FOUND_HR;
    }

    Pg6HResult CreateStatusArray(uint32_t capacity,
                                 const char* name,
                                 const Pg6Guid& riid,
                                 void** ppv) override {
        (void)name;
        if (ppv == nullptr) {
            return E_POINTER_HR;
        }
        *ppv = nullptr;
        if (capacity == 0) {
            return E_INVALIDARG_HR;
        }
        if (riid != kIidIUnknown && riid != kIidIDStorageStatusArray) {
            return E_NOINTERFACE_HR;
        }
        auto* arr = new (std::nothrow) DStorageStatusArrayImpl(capacity);
        if (arr == nullptr) {
            return E_OUTOFMEMORY_HR;
        }
        *ppv = static_cast<Pg6IDStorageStatusArray*>(arr);
        return S_OK_HR;
    }

    Pg6HResult SetStagingBufferSize(uint32_t size) override {
        if (size < kMinStagingBufferSize || size > kMaxStagingBufferSize) {
            return E_INVALIDARG_HR;
        }
        auto& ds = dstorage_state();
        std::lock_guard<std::mutex> lock(ds.mutex);
        ds.staging_buffer_size = size;
        return S_OK_HR;
    }

    uint32_t GetStagingBufferSize() const override {
        auto& ds = dstorage_state();
        std::lock_guard<std::mutex> lock(ds.mutex);
        return ds.staging_buffer_size;
    }

private:
    std::atomic<uint32_t> ref_count_{1};
};

// ===========================================================================
// 3. xg_xs.dll (Texture Address Computer) implementation
// ===========================================================================

struct FormatMetrics {
    uint32_t block_width = 1;
    uint32_t block_height = 1;
    uint32_t bytes_per_block = 4;
};

FormatMetrics format_metrics(uint32_t dxgi_format) {
    switch (dxgi_format) {
        case kDxgiFormatR8Unorm:
            return {1, 1, 1};
        case kDxgiFormatR8G8Unorm:
        case kDxgiFormatR16Float:
            return {1, 1, 2};
        case kDxgiFormatR8G8B8A8Unorm:
        case kDxgiFormatR8G8B8A8UnormSrgb:
        case kDxgiFormatB8G8R8A8Unorm:
        case kDxgiFormatR32Float:
        case kDxgiFormatD32Float:
        case kDxgiFormatD24UnormS8Uint:
            return {1, 1, 4};
        case kDxgiFormatR16G16B16A16Float:
            return {1, 1, 8};
        case kDxgiFormatR32G32B32A32Float:
            return {1, 1, 16};
        case kDxgiFormatBc1Unorm:
        case kDxgiFormatBc1UnormSrgb:
        case kDxgiFormatBc4Unorm:
            return {4, 4, 8};
        case kDxgiFormatBc2Unorm:
        case kDxgiFormatBc3Unorm:
        case kDxgiFormatBc3UnormSrgb:
        case kDxgiFormatBc5Unorm:
        case kDxgiFormatBc6hUf16:
        case kDxgiFormatBc7Unorm:
        case kDxgiFormatBc7UnormSrgb:
            return {4, 4, 16};
        default:
            return {1, 1, 4};
    }
}

uint32_t compute_full_mip_count(uint32_t width, uint32_t height, uint32_t depth) {
    uint32_t dim = std::max({width, height, depth, 1u});
    uint32_t mips = 1;
    while (dim > 1) {
        dim >>= 1u;
        ++mips;
    }
    return std::min(mips, 16u);
}

class XGTextureAddressComputerImpl final : public Pg6IXGTextureAddressComputer {
public:
    explicit XGTextureAddressComputerImpl(const Pg6XgTextureDesc& desc) : desc_(desc) {
        metrics_ = format_metrics(desc.format);
        const bool is_3d = desc.dimension == kXgDimensionTexture3D;
        const bool is_1d = desc.dimension == kXgDimensionTexture1D;
        const uint32_t base_height = is_1d ? 1u : std::max(desc.height, 1u);
        const uint32_t base_depth = is_3d ? std::max(desc.depth_or_array_size, 1u) : 1u;
        const uint32_t array_size = is_3d ? 1u : std::max(desc.depth_or_array_size, 1u);

        const uint32_t max_mips = compute_full_mip_count(desc.width, base_height, base_depth);
        const uint32_t mips =
            desc.mip_levels == 0 ? max_mips : std::min({desc.mip_levels, max_mips, 16u});

        const uint32_t pitch_align = desc.pitch_alignment == 0 ? 256u : desc.pitch_alignment;
        const uint64_t mip_placement_align =
            desc.tile_mode == kXgTileModeStandard64KB ? kXMemPageSize64K : 512ULL;
        const uint64_t base_align =
            desc.tile_mode == kXgTileModeLinear ? 256ULL : kXMemPageSize64K;

        layout_.mip_levels = mips;
        layout_.array_size = array_size;
        layout_.tile_mode = desc.tile_mode;
        layout_.base_alignment_bytes = base_align;

        uint64_t cursor = 0;
        for (uint32_t m = 0; m < mips; ++m) {
            const uint32_t mip_w = std::max(1u, desc.width >> m);
            const uint32_t mip_h = is_1d ? 1u : std::max(1u, base_height >> m);
            const uint32_t mip_d = is_3d ? std::max(1u, base_depth >> m) : 1u;

            const uint32_t blocks_x =
                (mip_w + metrics_.block_width - 1u) / metrics_.block_width;
            const uint32_t blocks_y =
                (mip_h + metrics_.block_height - 1u) / metrics_.block_height;

            const uint32_t raw_row_bytes = blocks_x * metrics_.bytes_per_block;
            const uint32_t row_pitch =
                static_cast<uint32_t>(align_up_u64(raw_row_bytes, pitch_align));
            const uint64_t slice_size = static_cast<uint64_t>(row_pitch) * blocks_y;
            const uint64_t mip_total =
                slice_size * static_cast<uint64_t>(is_3d ? mip_d : array_size);

            cursor = align_up_u64(cursor, mip_placement_align);

            Pg6XgMipLevelLayout& ml = layout_.mips[m];
            ml.offset_bytes = cursor;
            ml.size_bytes = mip_total;
            ml.slice_size_bytes = slice_size;
            ml.width = mip_w;
            ml.height = mip_h;
            ml.depth = mip_d;
            ml.row_pitch_bytes = row_pitch;
            ml.padded_rows = blocks_y;
            ml.tile_mode = desc.tile_mode;

            cursor += mip_total;
        }

        layout_.size_bytes = align_up_u64(cursor, base_align);

        // The 64 KiB-block mode for the surfaces os/surface_blocks.h covers (single-sample 2D,
        // one mip, one slice, 1x1-element formats, at least one block in each dimension): padded
        // to whole blocks. Everything else keeps the linear layout above. Only sizes, pitches
        // and padded extents follow the block rule; element order inside blocks is not modelled.
        surface::Block64kLayout block{};
        if (desc.tile_mode == kXgTileModeStandard64KB && desc.dimension == 3 && mips == 1 && array_size == 1 &&
            std::max(desc.sample_count, 1u) == 1 && metrics_.block_width == 1 && metrics_.block_height == 1 &&
            surface::block64k_layout_2d(desc.width, base_height, metrics_.bytes_per_block, &block)) {
            Pg6XgMipLevelLayout& ml = layout_.mips[0];
            ml.row_pitch_bytes = static_cast<uint32_t>(block.row_pitch_bytes);
            ml.padded_rows = block.padded_height;
            ml.slice_size_bytes = block.size_bytes;
            ml.size_bytes = block.size_bytes;
            layout_.size_bytes = block.size_bytes;
            layout_.base_alignment_bytes = block.alignment;
        }
    }

    Pg6HResult QueryInterface(const Pg6Guid& riid, void** ppv_object) override {
        if (ppv_object == nullptr) {
            return E_POINTER_HR;
        }
        if (riid == kIidIUnknown || riid == kIidIXGTextureAddressComputer) {
            AddRef();
            *ppv_object = static_cast<Pg6IXGTextureAddressComputer*>(this);
            return S_OK_HR;
        }
        *ppv_object = nullptr;
        return E_NOINTERFACE_HR;
    }

    uint32_t AddRef() override { return ref_count_.fetch_add(1, std::memory_order_relaxed) + 1; }

    uint32_t Release() override {
        const uint32_t next = ref_count_.fetch_sub(1, std::memory_order_acq_rel) - 1;
        if (next == 0) {
            delete this;
        }
        return next;
    }

    Pg6HResult GetResourceLayout(Pg6XgResourceLayout* out_layout) override {
        if (out_layout == nullptr) {
            return E_POINTER_HR;
        }
        *out_layout = layout_;
        return S_OK_HR;
    }

    uint64_t GetResourceSizeBytes() override { return layout_.size_bytes; }

    uint64_t GetResourceBaseAlignmentBytes() override { return layout_.base_alignment_bytes; }

    uint64_t GetMipLevelOffsetBytes(uint32_t plane, uint32_t mip_level) override {
        if (plane != 0 || mip_level >= layout_.mip_levels) {
            return ~uint64_t{0};
        }
        return layout_.mips[mip_level].offset_bytes;
    }

    uint64_t GetTexelElementOffsetBytes(uint32_t plane,
                                        uint32_t mip_level,
                                        uint64_t x,
                                        uint32_t y,
                                        uint32_t z_or_slice,
                                        uint32_t sample) override {
        (void)sample;
        if (plane != 0 || mip_level >= layout_.mip_levels) {
            return ~uint64_t{0};
        }
        const Pg6XgMipLevelLayout& ml = layout_.mips[mip_level];
        const uint32_t max_slices =
            desc_.dimension == kXgDimensionTexture3D ? ml.depth : layout_.array_size;
        if (x >= ml.width || y >= ml.height || z_or_slice >= max_slices) {
            return ~uint64_t{0};
        }
        const uint64_t bx = x / metrics_.block_width;
        const uint64_t by = y / metrics_.block_height;
        return ml.offset_bytes + static_cast<uint64_t>(z_or_slice) * ml.slice_size_bytes +
               by * ml.row_pitch_bytes + bx * metrics_.bytes_per_block;
    }

    uint32_t GetMipLevelRowPitchBytes(uint32_t mip_level) override {
        if (mip_level >= layout_.mip_levels) {
            return 0;
        }
        return layout_.mips[mip_level].row_pitch_bytes;
    }

    uint64_t GetMipLevelSliceSizeBytes(uint32_t mip_level) override {
        if (mip_level >= layout_.mip_levels) {
            return 0;
        }
        return layout_.mips[mip_level].slice_size_bytes;
    }

    Pg6HResult CopyIntoSubresource(void* dst_base,
                                   uint32_t plane,
                                   uint32_t subresource,
                                   const void* src_data,
                                   uint32_t src_row_pitch,
                                   uint32_t src_slice_pitch) override {
        if (dst_base == nullptr || src_data == nullptr) {
            return E_POINTER_HR;
        }
        if (plane != 0 || layout_.mip_levels == 0) {
            return E_INVALIDARG_HR;
        }
        const uint32_t mip = subresource % layout_.mip_levels;
        const uint32_t slice = subresource / layout_.mip_levels;
        const bool is_3d = desc_.dimension == kXgDimensionTexture3D;
        if (!is_3d && slice >= layout_.array_size) {
            return E_INVALIDARG_HR;
        }
        if (is_3d && slice != 0) {
            return E_INVALIDARG_HR;
        }

        const Pg6XgMipLevelLayout& ml = layout_.mips[mip];
        const uint32_t blocks_x = (ml.width + metrics_.block_width - 1u) / metrics_.block_width;
        const uint32_t row_bytes = blocks_x * metrics_.bytes_per_block;
        const uint32_t depth_count = is_3d ? ml.depth : 1u;

        auto* dst_root = static_cast<uint8_t*>(dst_base) + ml.offset_bytes +
                         static_cast<uint64_t>(slice) * ml.slice_size_bytes;
        const auto* src_root = static_cast<const uint8_t*>(src_data);

        for (uint32_t z = 0; z < depth_count; ++z) {
            auto* dst_slice = dst_root + static_cast<uint64_t>(z) * ml.slice_size_bytes;
            const auto* src_slice = src_root + static_cast<uint64_t>(z) * src_slice_pitch;
            for (uint32_t r = 0; r < ml.padded_rows; ++r) {
                std::memcpy(dst_slice + static_cast<uint64_t>(r) * ml.row_pitch_bytes,
                            src_slice + static_cast<uint64_t>(r) * src_row_pitch,
                            row_bytes);
            }
        }
        return S_OK_HR;
    }

    Pg6HResult CopyFromSubresource(void* dst_data,
                                   uint32_t dst_row_pitch,
                                   uint32_t dst_slice_pitch,
                                   const void* src_base,
                                   uint32_t plane,
                                   uint32_t subresource) override {
        if (dst_data == nullptr || src_base == nullptr) {
            return E_POINTER_HR;
        }
        if (plane != 0 || layout_.mip_levels == 0) {
            return E_INVALIDARG_HR;
        }
        const uint32_t mip = subresource % layout_.mip_levels;
        const uint32_t slice = subresource / layout_.mip_levels;
        const bool is_3d = desc_.dimension == kXgDimensionTexture3D;
        if (!is_3d && slice >= layout_.array_size) {
            return E_INVALIDARG_HR;
        }
        if (is_3d && slice != 0) {
            return E_INVALIDARG_HR;
        }

        const Pg6XgMipLevelLayout& ml = layout_.mips[mip];
        const uint32_t blocks_x = (ml.width + metrics_.block_width - 1u) / metrics_.block_width;
        const uint32_t row_bytes = blocks_x * metrics_.bytes_per_block;
        const uint32_t depth_count = is_3d ? ml.depth : 1u;

        const auto* src_root = static_cast<const uint8_t*>(src_base) + ml.offset_bytes +
                               static_cast<uint64_t>(slice) * ml.slice_size_bytes;
        auto* dst_root = static_cast<uint8_t*>(dst_data);

        for (uint32_t z = 0; z < depth_count; ++z) {
            const auto* src_slice = src_root + static_cast<uint64_t>(z) * ml.slice_size_bytes;
            auto* dst_slice = dst_root + static_cast<uint64_t>(z) * dst_slice_pitch;
            for (uint32_t r = 0; r < ml.padded_rows; ++r) {
                std::memcpy(dst_slice + static_cast<uint64_t>(r) * dst_row_pitch,
                            src_slice + static_cast<uint64_t>(r) * ml.row_pitch_bytes,
                            row_bytes);
            }
        }
        return S_OK_HR;
    }

private:
    std::atomic<uint32_t> ref_count_{1};
    Pg6XgTextureDesc desc_{};
    FormatMetrics metrics_{};
    Pg6XgResourceLayout layout_{};
};

// ===========================================================================
// 4. GameInput.dll internal state & COM implementations
// ===========================================================================

struct GameInputGlobalState {
    std::mutex mutex;
    playgta6::InputState* shared_state = nullptr;
    GameInputReader reader;
    int64_t accum_mouse_x = 0;
    int64_t accum_mouse_y = 0;
    uint32_t focus_policy = 0;
    // Press latch (O36): the title samples GameInput once per frame, so a key tap that goes down
    // and up between two of its readings was lost. The sampler records press edges between
    // readings; the next reading reports them as down once.
    uint32_t latch_last_buttons = 0;  // newest buttons seen by the sampler or a reading
    uint32_t latch_pending = 0;       // press edges not reported by a reading yet
    uint64_t seq_bias = 0;            // keeps reading sequence numbers distinct around latched readings
    bool latch_emitted = false;       // the previous reading carried latched buttons
};

GameInputGlobalState& game_input_state() {
    static GameInputGlobalState state;
    return state;
}

void sample_game_input_presses_locked(GameInputGlobalState& gi) {
    if (!gi.reader.is_attached()) {
        return;
    }
    playgta6::InputState snap{};
    if (!gi.reader.read_snapshot(snap, 4)) {
        return;
    }
    gi.latch_pending |= snap.buttons & ~gi.latch_last_buttons;
    gi.latch_last_buttons = snap.buttons;
}

std::atomic<bool>& game_input_sampler_enabled() {
    static std::atomic<bool> enabled{false};
    return enabled;
}

constexpr auto kGameInputSamplePeriod = std::chrono::milliseconds(4);

uint32_t map_sab_buttons_to_gamepad(uint32_t raw_buttons) {
    uint32_t out = gamepad_button::kNone;
    if ((raw_buttons & (input_button::kGamepadA | input_button::kJump)) != 0) {
        out |= gamepad_button::kA;
    }
    if ((raw_buttons & input_button::kGamepadB) != 0) {
        out |= gamepad_button::kB;
    }
    if ((raw_buttons & input_button::kGamepadX) != 0) {
        out |= gamepad_button::kX;
    }
    if ((raw_buttons & input_button::kGamepadY) != 0) {
        out |= gamepad_button::kY;
    }
    if ((raw_buttons & input_button::kLeftBumper) != 0) {
        out |= gamepad_button::kLeftShoulder;
    }
    if ((raw_buttons & input_button::kRightBumper) != 0) {
        out |= gamepad_button::kRightShoulder;
    }
    if ((raw_buttons & input_button::kMenu) != 0) {
        out |= gamepad_button::kMenu;
    }
    if ((raw_buttons & input_button::kView) != 0) {
        out |= gamepad_button::kView;
    }
    if ((raw_buttons & (input_button::kSprint | input_button::kLeftStickPress)) != 0) {
        out |= gamepad_button::kLeftThumbstick;
    }
    if ((raw_buttons & input_button::kRightStickPress) != 0) {
        out |= gamepad_button::kRightThumbstick;
    }
    if ((raw_buttons & input_button::kDPadUp) != 0) {
        out |= gamepad_button::kDPadUp;
    }
    if ((raw_buttons & input_button::kDPadDown) != 0) {
        out |= gamepad_button::kDPadDown;
    }
    if ((raw_buttons & input_button::kDPadLeft) != 0) {
        out |= gamepad_button::kDPadLeft;
    }
    if ((raw_buttons & input_button::kDPadRight) != 0) {
        out |= gamepad_button::kDPadRight;
    }
    return out;
}

class GameInputReadingImpl final : public Pg6IGameInputReading {
public:
    GameInputReadingImpl(uint32_t input_kind,
                         uint64_t sequence,
                         uint64_t timestamp,
                         uint32_t raw_buttons,
                         const Pg6GameInputGamepadState& gamepad,
                         const Pg6GameInputMouseState& mouse)
        : input_kind_(input_kind),
          sequence_(sequence),
          timestamp_(timestamp),
          raw_buttons_(raw_buttons),
          gamepad_(gamepad),
          mouse_(mouse) {}

    Pg6HResult QueryInterface(const Pg6Guid& riid, void** ppv_object) override {
        if (ppv_object == nullptr) {
            return E_POINTER_HR;
        }
        if (riid == kIidIUnknown || riid == kIidIGameInputReading) {
            AddRef();
            *ppv_object = static_cast<Pg6IGameInputReading*>(this);
            return S_OK_HR;
        }
        *ppv_object = nullptr;
        return E_NOINTERFACE_HR;
    }

    uint32_t AddRef() override { return ref_count_.fetch_add(1, std::memory_order_relaxed) + 1; }

    uint32_t Release() override {
        const uint32_t next = ref_count_.fetch_sub(1, std::memory_order_acq_rel) - 1;
        if (next == 0) {
            delete this;
        }
        return next;
    }

    uint32_t GetInputKind() override { return input_kind_; }

    uint64_t GetSequenceNumber(uint32_t input_kind) override {
        (void)input_kind;
        return sequence_;
    }

    uint64_t GetTimestamp() override { return timestamp_; }

    bool GetGamepadState(Pg6GameInputGamepadState* state) override {
        if (state == nullptr) {
            return false;
        }
        *state = gamepad_;
        return true;
    }

    bool GetMouseState(Pg6GameInputMouseState* state) override {
        if (state == nullptr) {
            return false;
        }
        *state = mouse_;
        return true;
    }

    uint32_t GetRawButtons() override { return raw_buttons_; }

private:
    std::atomic<uint32_t> ref_count_{1};
    uint32_t input_kind_ = kGameInputKindGamepad;
    uint64_t sequence_ = 0;
    uint64_t timestamp_ = 0;
    uint32_t raw_buttons_ = 0;
    Pg6GameInputGamepadState gamepad_{};
    Pg6GameInputMouseState mouse_{};
};

class GameInputImpl final : public Pg6IGameInput {
public:
    Pg6HResult QueryInterface(const Pg6Guid& riid, void** ppv_object) override {
        if (ppv_object == nullptr) {
            return E_POINTER_HR;
        }
        if (riid == kIidIUnknown || riid == kIidIGameInput) {
            AddRef();
            *ppv_object = static_cast<Pg6IGameInput*>(this);
            return S_OK_HR;
        }
        *ppv_object = nullptr;
        return E_NOINTERFACE_HR;
    }

    uint32_t AddRef() override { return ref_count_.fetch_add(1, std::memory_order_relaxed) + 1; }

    uint32_t Release() override {
        const uint32_t next = ref_count_.fetch_sub(1, std::memory_order_acq_rel) - 1;
        if (next == 0) {
            delete this;
        }
        return next;
    }

    uint64_t GetCurrentTimestamp() override { return steady_microseconds(); }

    Pg6HResult GetCurrentReading(uint32_t input_kind,
                                 void* device,
                                 Pg6IGameInputReading** reading) override {
        (void)device;
        if (reading == nullptr) {
            return E_POINTER_HR;
        }
        *reading = nullptr;
        if (input_kind == kGameInputKindUnknown) {
            return E_INVALIDARG_HR;
        }

        Pg6GameInputGamepadState gp{};
        Pg6GameInputMouseState mouse{};
        uint64_t seq = 0;
        uint32_t raw_buttons = 0;

        {
            auto& gi = game_input_state();
            std::lock_guard<std::mutex> lock(gi.mutex);
            GameInputFrame frame{};
            if (gi.reader.is_attached() && gi.reader.poll(frame, true)) {
                // Presses that went down and up since the previous reading are reported once.
                const uint32_t latched = gi.latch_pending & ~frame.buttons;
                gi.latch_pending = 0;
                gi.latch_last_buttons = frame.buttons;
                if (latched != 0 || gi.latch_emitted) {
                    gi.seq_bias += 1;
                }
                gi.latch_emitted = latched != 0;
                frame.buttons |= latched;
                seq = frame.seq + gi.seq_bias;
                raw_buttons = frame.buttons;
                gp.buttons = map_sab_buttons_to_gamepad(frame.buttons);
                gp.left_trigger = frame.left_trigger;
                gp.right_trigger = frame.right_trigger;
                gp.left_thumbstick_x = frame.left_stick.x;
                gp.left_thumbstick_y = frame.left_stick.y;
                gp.right_thumbstick_x = frame.right_stick.x;
                gp.right_thumbstick_y = frame.right_stick.y;

                // Synthesize left stick from WASD when analog stick is neutral.
                if (gp.left_thumbstick_x == 0.0f && gp.left_thumbstick_y == 0.0f) {
                    float sx = 0.0f;
                    float sy = 0.0f;
                    if ((frame.buttons & input_button::kMoveRight) != 0) sx += 1.0f;
                    if ((frame.buttons & input_button::kMoveLeft) != 0) sx -= 1.0f;
                    if ((frame.buttons & input_button::kMoveForward) != 0) sy += 1.0f;
                    if ((frame.buttons & input_button::kMoveBackward) != 0) sy -= 1.0f;
                    const float mag = std::hypot(sx, sy);
                    if (mag > 1.0f) {
                        sx /= mag;
                        sy /= mag;
                    }
                    gp.left_thumbstick_x = sx;
                    gp.left_thumbstick_y = sy;
                }

                gi.accum_mouse_x += frame.mouse_delta.dx;
                gi.accum_mouse_y += frame.mouse_delta.dy;
                mouse.position_x = gi.accum_mouse_x;
                mouse.position_y = gi.accum_mouse_y;
            }
        }

        const uint32_t resolved_kind =
            (input_kind & kGameInputKindGamepad) != 0 ? kGameInputKindGamepad : input_kind;
        auto* impl = new (std::nothrow) GameInputReadingImpl(
            resolved_kind, seq, steady_microseconds(), raw_buttons, gp, mouse);
        if (impl == nullptr) {
            return E_OUTOFMEMORY_HR;
        }
        *reading = static_cast<Pg6IGameInputReading*>(impl);
        return S_OK_HR;
    }

    Pg6HResult GetNextReading(Pg6IGameInputReading* reference_reading,
                              uint32_t input_kind,
                              void* device,
                              Pg6IGameInputReading** reading) override {
        if (reading == nullptr) {
            return E_POINTER_HR;
        }
        *reading = nullptr;
        if (reference_reading == nullptr) {
            return GetCurrentReading(input_kind, device, reading);
        }
        Pg6IGameInputReading* latest = nullptr;
        const Pg6HResult hr = GetCurrentReading(input_kind, device, &latest);
        if (gdk::failed(hr)) {
            return hr;
        }
        if (latest->GetSequenceNumber(input_kind) <=
            reference_reading->GetSequenceNumber(input_kind)) {
            latest->Release();
            return GAMEINPUT_E_READING_NOT_FOUND;
        }
        *reading = latest;
        return S_OK_HR;
    }

    Pg6HResult GetPreviousReading(Pg6IGameInputReading* reference_reading,
                                  uint32_t input_kind,
                                  void* device,
                                  Pg6IGameInputReading** reading) override {
        (void)reference_reading;
        (void)input_kind;
        (void)device;
        if (reading == nullptr) {
            return E_POINTER_HR;
        }
        *reading = nullptr;
        return GAMEINPUT_E_READING_NOT_FOUND;
    }

    void SetFocusPolicy(uint32_t policy) override {
        auto& gi = game_input_state();
        std::lock_guard<std::mutex> lock(gi.mutex);
        gi.focus_policy = policy;
    }

private:
    std::atomic<uint32_t> ref_count_{1};
};

}  // namespace

// ---------------------------------------------------------------------------
// Public C++ helpers in playgta6::os::scarlett
// ---------------------------------------------------------------------------

void xmem_set_allocation_hooks(XMemAllocHookFn alloc_hook, XMemFreeHookFn free_hook) {
    auto& st = xmem_state();
    std::lock_guard<std::mutex> lock(st.mutex);
    st.alloc_hook = alloc_hook;
    st.free_hook = free_hook;
}

void xmem_set_guest_hook_table(uint64_t table_va) {
    auto& st = xmem_state();
    std::lock_guard<std::mutex> lock(st.mutex);
    st.guest_hook_table = table_va;
}

uint64_t xmem_guest_hook_table() {
    auto& st = xmem_state();
    std::lock_guard<std::mutex> lock(st.mutex);
    return st.guest_hook_table;
}

namespace {

// The guest routine for XMemAlloc (index 0) or XMemFree (index 1), 0 if none.
uint64_t guest_hook_routine(size_t index) {
    uint64_t table = 0;
    uint64_t explicit_routine = 0;
    {
        auto& st = xmem_state();
        std::lock_guard<std::mutex> lock(st.mutex);
        if (st.guest_alloc_routine != 0 || st.guest_free_routine != 0) {
            explicit_routine = index == 0 ? st.guest_alloc_routine : st.guest_free_routine;
            return explicit_routine;
        }
        table = st.guest_hook_table;
    }
    uint64_t routine = 0;
    if (table != 0 && !xmem_arena().read(table + 8 * index, &routine, sizeof(routine))) {
        routine = 0;
    }
    return routine;
}

// Calls a guest routine with two integer arguments on this thread's guest state. The state is
// restored afterwards: the caller may be a shim inside another guest call (D3D12 resource
// creation allocating through XMemAlloc).
uint64_t call_guest_routine(uint64_t routine, uint64_t a0, uint64_t a1) {
    namespace gcom = playgta6::guest::com;
    GuestState& state = gcom::current_guest_state();
    const GuestState saved = state;
    const uint64_t result = gcom::GuestFunction<uint64_t(uint64_t, uint64_t)>(routine).call(state, a0, a1);
    state = saved;
    return result;
}

}  // namespace

bool xmem_get_allocation_info(const void* address, XMemAllocInfo* out_info) {
    if (address == nullptr) {
        return false;
    }
    auto& st = xmem_state();
    std::lock_guard<std::mutex> lock(st.mutex);
    auto it = st.heap_allocs.find(const_cast<void*>(address));
    if (it == st.heap_allocs.end()) {
        return false;
    }
    if (out_info != nullptr) {
        out_info->size = it->second.size;
        out_info->alignment = it->second.alignment;
        out_info->attributes = it->second.attributes;
    }
    return true;
}

bool xmem_virtual_free(void* address) {
    if (address == nullptr) {
        return false;
    }
    auto& st = xmem_state();
    std::unique_lock<std::mutex> lock(st.mutex);
    wait_idle_locked(st, lock);  // R40
    auto it = st.virtual_regions.find(address);
    if (it == st.virtual_regions.end()) {
        // The translated window, as the import thunk handed its base over (O29), or a mirrored
        // reservation's window VA (O37).
        address = host_of(unmirror(guest_of_arg(address)));
        it = st.virtual_regions.find(address);
    }
    if (it == st.virtual_regions.end()) {
        return false;
    }
    // Unmap every physical page mapped inside this reservation (their bytes go back to the store).
    const uint64_t base_va = guest_of(address);
    unmap_range_locked(st, base_va, it->second.size);

    if (it->second.committed) {
        st.virtual_committed_bytes -= it->second.size;
    }
    if (!it->second.phantom && !it->second.translated) {
        st.virtual_reserved_bytes -= it->second.size;
    }
    if (it->second.translated) {
        st.translated_window_live = false;
    }
    xmem_windows::forget_window(base_va);
    const auto mit = st.mirrors.find(base_va);
    if (mit != st.mirrors.end() && mit->second.bytes == align_up_u64(it->second.size, kXMemPageSize64K)) {
        guest_window::unmap_runtime_range(mit->second.window_va, mit->second.bytes);  // O37
        st.mirrors.erase(mit);
    }
    release_virtual_region(it->second);
    st.region_index.erase(base_va);
    st.virtual_regions.erase(it);
    return true;
}

bool xmem_write_physical_pages(const std::function<bool(size_t index, uint64_t* entry)>& entry_at,
                               const uint8_t* data,
                               size_t len,
                               uint64_t first_offset) {
    auto& st = xmem_state();
    std::unique_lock<std::mutex> lock(st.mutex);
    wait_idle_locked(st, lock);  // no page bytes in flight while they are written
    auto find_page = [&](uint64_t entry) -> PhysicalPage* {
        if ((entry & (kXMemPageSize4K - 1)) == 0) {
            auto it = st.physical_pages.find(entry >> 12);  // a physical address
            if (it != st.physical_pages.end()) {
                return it->second.get();
            }
        }
        auto it = st.physical_pages.find(entry);  // a frame number
        return it != st.physical_pages.end() ? it->second.get() : nullptr;
    };
    std::vector<PhysicalPage*> pages;
    for (size_t done = 0; done < len;) {
        uint64_t entry = 0;
        if (!entry_at(pages.size(), &entry)) {
            return false;
        }
        PhysicalPage* page = find_page(entry);
        if (page == nullptr) {
            return false;
        }
        const uint64_t skip = pages.empty() ? first_offset : 0;
        if (skip >= page->page_size) {
            return false;
        }
        pages.push_back(page);
        done += static_cast<size_t>(std::min<uint64_t>(page->page_size - skip, len - done));
    }
    size_t off = 0;
    for (size_t i = 0; i < pages.size(); ++i) {
        PhysicalPage* page = pages[i];
        const uint64_t skip = i == 0 ? first_offset : 0;
        const size_t n = static_cast<size_t>(std::min<uint64_t>(page->page_size - skip, len - off));
        uint8_t* bytes = page_bytes_locked(*page);
        if (bytes == nullptr) {
            return false;
        }
        std::memcpy(bytes + skip, data + off, n);
        if (page->home_va != 0) {
            guest_window::mark_written(page->home_va + skip, n);
            mark_mirror_locked(st, page->home_va + skip, n);
        }
        for (const uint64_t w : page->window_slots) {
            guest_window::mark_written(w + skip, n);
        }
        off += n;
    }
    return true;
}

bool xmem_unmap_physical_pages(void* virtual_address) {
    if (virtual_address == nullptr) {
        return false;
    }
    auto& st = xmem_state();
    std::unique_lock<std::mutex> lock(st.mutex);
    wait_idle_locked(st, lock);  // R40
    const uint64_t base = guest_of(virtual_address);
    auto bit = st.owned_blocks.find(base);
    if (bit != st.owned_blocks.end()) {
        unmap_range_locked(st, base, bit->second.size);
        aligned_free_bytes(virtual_address, bit->second.alignment);
        st.owned_blocks.erase(bit);
        return true;
    }
    VirtualRegion* region = st.find_virtual_region_locked(virtual_address, 0);
    if (region == nullptr) {
        return false;
    }
    auto first = st.page_slots.lower_bound(base);
    if (first == st.page_slots.end() || first->first >= base + region->size) {
        return false;
    }
    unmap_range_locked(st, base, region->size);
    return true;
}

XMemWindowContiguity xmem_window_contiguity() {
    auto& st = xmem_state();
    std::lock_guard<std::mutex> lock(st.mutex);
    XMemWindowContiguity c;
    c.maps = st.cpu_window_maps;
    c.neighbour_pairs = st.cpu_window_neighbour_pairs;
    c.neighbour_pairs_contiguous = st.cpu_window_neighbour_pairs_contiguous;
    c.neighbour_pairs_reversed = st.cpu_window_neighbour_pairs_reversed;
    c.run_kept = st.run_kept;
    c.run_moves = st.run_moves;
    c.run_breaks = st.run_breaks;
    c.run_relocations = st.run_relocations;
    c.run_bytes_moved = st.run_bytes_moved;
    c.alias_slots = st.alias_slots;
    c.alias_conflicts = st.alias_conflicts;
    c.zero_fills = st.zero_fills;
    for (const auto& [key, region] : st.virtual_regions) {
        if (!region.cpu_window || region.phantom) {
            continue;
        }
        XMemWindowContiguity::Region r;
        r.base = region_guest_base(region);
        r.size = region.size;
        r.translated = region.translated;
        uint64_t prev_va = 0;
        const PhysicalPage* prev = nullptr;
        uint64_t run = 0;
        for (auto it = st.page_slots.lower_bound(r.base); it != st.page_slots.end() && it->first - r.base < r.size;
             ++it) {
            const auto pit = st.physical_pages.find(it->second);
            const PhysicalPage* page = pit != st.physical_pages.end() ? pit->second.get() : nullptr;
            const uint64_t ps = page != nullptr ? page->page_size : kXMemPageSize64K;
            const bool aliased = page != nullptr && aliases_into(*page, r.base, region);
            ++r.mapped_slots;
            r.aliased_slots += aliased ? 1 : 0;
            if (run != 0 && it->first == prev_va + ps) {
                ++run;
                ++r.adjacent_pairs;
                if (aliased && prev != nullptr && aliases_into(*prev, r.base, region)) {
                    ++r.aliased_pairs;
                    if (page->pool_va == prev->pool_va + ps) {
                        ++r.contiguous_pairs;
                    }
                }
            } else {
                ++r.runs;
                run = 1;
            }
            r.longest_run = std::max(r.longest_run, run);
            prev_va = it->first;
            prev = page;
        }
        c.regions.push_back(r);
    }
    std::sort(c.regions.begin(), c.regions.end(),
              [](const XMemWindowContiguity::Region& a, const XMemWindowContiguity::Region& b) { return a.base < b.base; });
    return c;
}

void xmem_reset_for_testing() {
    auto& st = xmem_state();
    std::unique_lock<std::mutex> lock(st.mutex);
    wait_idle_locked(st, lock);  // R40
    st.alloc_hook = nullptr;
    st.free_hook = nullptr;
    st.guest_hook_table = 0;
    st.guest_alloc_routine = 0;
    st.guest_free_routine = 0;
    for (auto& [ptr, rec] : st.heap_allocs) {
        aligned_free_bytes(ptr, rec.alignment);
    }
    st.heap_allocs.clear();
    st.page_slots.clear();
    for (auto& [base, block] : st.owned_blocks) {
        aligned_free_bytes(host_of(base), block.alignment);
    }
    st.owned_blocks.clear();
    for (auto& [ptr, region] : st.virtual_regions) {
        release_virtual_region(region);
    }
    st.virtual_regions.clear();
    st.region_index.clear();
    st.alias_conflicts = 0;
    st.zero_fills = 0;
    st.alias_slots = 0;
    st.shared_exhausted_logged = false;
    st.cpu_window_maps = 0;
    st.cpu_window_neighbour_pairs = 0;
    st.cpu_window_neighbour_pairs_contiguous = 0;
    st.cpu_window_neighbour_pairs_reversed = 0;
    st.run_kept = 0;
    st.run_moves = 0;
    st.run_breaks = 0;
    st.run_relocations = 0;
    st.run_bytes_moved = 0;
    st.moved_homes.clear();
    st.mirrors.clear();
    st.mirror_exhausted_logged = false;
    st.pool_view_placed = false;
    st.translated_window_live = false;
    xmem_windows::reset_for_testing();
    for (auto& [handle, page] : st.physical_pages) {
        if (page->store_va != 0) {
            default_guest_heap().free(page->store_va);
        }
        release_shared_locked(*page);
    }
    st.physical_pages.clear();
    st.next_page_handle = 0x1000ULL;
    st.heap_allocated_bytes = 0;
    st.physical_pages_bytes = 0;
    st.virtual_committed_bytes = 0;
    st.virtual_reserved_bytes = 0;
    st.mapped_physical_bytes = 0;
    st.peak_working_set_bytes = 0;
}

void register_dstorage_virtual_file(std::string_view path, std::vector<uint8_t> data) {
    const std::string normalized = win32::normalize_path(path);
    if (normalized.empty()) {
        return;
    }
    auto& ds = dstorage_state();
    std::lock_guard<std::mutex> lock(ds.mutex);
    ds.virtual_files[normalized] = std::move(data);
}

void unregister_dstorage_virtual_file(std::string_view path) {
    const std::string normalized = win32::normalize_path(path);
    auto& ds = dstorage_state();
    std::lock_guard<std::mutex> lock(ds.mutex);
    ds.virtual_files.erase(normalized);
}

void clear_dstorage_virtual_files() {
    auto& ds = dstorage_state();
    std::lock_guard<std::mutex> lock(ds.mutex);
    ds.virtual_files.clear();
}

void set_dstorage_host_callbacks(DStorageHostOpenCallback open_cb,
                                 DStorageHostReadCallback read_cb) {
    auto& ds = dstorage_state();
    std::lock_guard<std::mutex> lock(ds.mutex);
    ds.host_open_cb = std::move(open_cb);
    ds.host_read_cb = std::move(read_cb);
}

void reset_dstorage_for_testing() {
    auto& ds = dstorage_state();
    std::lock_guard<std::mutex> lock(ds.mutex);
    ds.virtual_files.clear();
    ds.host_open_cb = nullptr;
    ds.host_read_cb = nullptr;
    ds.staging_buffer_size = kDefaultStagingBufferSize;
}

void set_game_input_state(playgta6::InputState* shared_state, SeqlockMode mode) {
    auto& gi = game_input_state();
    std::lock_guard<std::mutex> lock(gi.mutex);
    gi.shared_state = shared_state;
    gi.reader.attach(shared_state, mode);
    gi.accum_mouse_x = 0;
    gi.accum_mouse_y = 0;
    gi.latch_last_buttons = 0;
    gi.latch_pending = 0;
    gi.seq_bias = 0;
    gi.latch_emitted = false;
}

void sample_game_input_presses() {
    auto& gi = game_input_state();
    std::lock_guard<std::mutex> lock(gi.mutex);
    sample_game_input_presses_locked(gi);
}

void set_game_input_press_sampler(bool enabled) {
    game_input_sampler_enabled().store(enabled, std::memory_order_release);
    if (!enabled) {
        return;
    }
    static std::once_flag started;
    std::call_once(started, [] {
        std::thread([] {
            for (;;) {
                if (game_input_sampler_enabled().load(std::memory_order_acquire)) {
                    sample_game_input_presses();
                }
                std::this_thread::sleep_for(kGameInputSamplePeriod);
            }
        }).detach();
    });
}

playgta6::InputState* get_game_input_state() {
    auto& gi = game_input_state();
    std::lock_guard<std::mutex> lock(gi.mutex);
    return gi.shared_state;
}

void set_game_input_deadzone(const DeadzoneConfig& config) {
    auto& gi = game_input_state();
    std::lock_guard<std::mutex> lock(gi.mutex);
    gi.reader.set_deadzone(config);
}

size_t register_all_scarlett_exports() {
    using win32::export_entry;
    using win32::register_exports;

    static std::mutex reg_mutex;
    std::lock_guard<std::mutex> lock(reg_mutex);

    size_t total = 0;
    total += register_exports(
        "xmem.dll",
        {
            export_entry("XMemAlloc", &pg6_xbox_XMemAlloc),
            export_entry("XMemSetAllocationHooks", &pg6_xbox_XMemSetAllocationHooks),
            export_entry("XMemAllocDefault", &pg6_xbox_XMemAllocDefault),
            export_entry("XMemAllocatePhysicalPages", &pg6_xbox_XMemAllocatePhysicalPages),
            export_entry("XMemFree", &pg6_xbox_XMemFree),
            export_entry("XMemFreeDefault", &pg6_xbox_XMemFreeDefault),
            export_entry("XMemFreePhysicalPages", &pg6_xbox_XMemFreePhysicalPages),
            export_entry("XMemGetWorkingSetStatistics", &pg6_xbox_XMemGetWorkingSetStatistics),
            export_entry("XMemIsToolsMemAvailable", &pg6_xbox_XMemIsToolsMemAvailable),
            export_entry("XMemMapPhysicalPages", &pg6_xbox_XMemMapPhysicalPages),
            export_entry("XMemMapPhysicalPagesScatter", &pg6_xbox_XMemMapPhysicalPagesScatter),
            export_entry("XMemTransferMemory", &pg6_xbox_XMemTransferMemory),
            export_entry("XMemVirtualAlloc", &pg6_xbox_XMemVirtualAlloc),
        });

    total += register_exports(
        "XCRDAPI.dll",
        {
            export_entry("DStorageGetFactory", &pg6_xbox_DStorageGetFactory),
        });

    total += register_exports(
        "xg_xs.dll",
        {
            export_entry("XGCreateTextureComputer", &pg6_xbox_XGCreateTextureComputer),
            export_entry("XGSuggestSwizzleMode", &pg6_xbox_XGSuggestSwizzleMode),
        });

    total += register_exports(
        "GameInput.dll",
        {
            export_entry("GameInputCreate", &pg6_xbox_GameInputCreate, 1u),
        });

    return total;
}

}  // namespace playgta6::os::scarlett

// ---------------------------------------------------------------------------
// extern "C" entrypoints
// ---------------------------------------------------------------------------

extern "C" {

void* pg6_xbox_XMemAlloc(size_t size, uint64_t attributes) {
    playgta6::os::scarlett::XMemAllocHookFn hook = nullptr;
    {
        auto& st = playgta6::os::scarlett::xmem_state();
        std::lock_guard<std::mutex> lock(st.mutex);
        hook = st.alloc_hook;
    }
    if (hook != nullptr) {
        return hook(size, attributes);
    }
    if (const uint64_t routine = playgta6::os::scarlett::guest_hook_routine(0); routine != 0) {
        const uint64_t va = playgta6::os::scarlett::call_guest_routine(routine, size, attributes);
        return playgta6::os::scarlett::host_of(va);
    }
    return pg6_xbox_XMemAllocDefault(size, attributes);
}

void pg6_xbox_XMemSetAllocationHooks(uint64_t alloc_routine, uint64_t free_routine) {
    auto& st = playgta6::os::scarlett::xmem_state();
    std::lock_guard<std::mutex> lock(st.mutex);
    st.guest_alloc_routine = alloc_routine;
    st.guest_free_routine = free_routine;
}

void* pg6_xbox_XMemAllocDefault(size_t size, uint64_t attributes) {
    using namespace playgta6::os::scarlett;
    const size_t align = decode_xmem_alignment(attributes);
    void* ptr = aligned_alloc_bytes(size, align);
    if (ptr == nullptr) {
        playgta6::os::win32::SetLastError(playgta6::os::win32::ERROR_NOT_ENOUGH_MEMORY);
        return nullptr;
    }
    if ((attributes & kXMemAttrZeroInitBit) != 0 && size > 0) {
        std::memset(ptr, 0, size);
    }

    auto& st = xmem_state();
    std::lock_guard<std::mutex> lock(st.mutex);
    st.heap_allocs.emplace(ptr, HeapAllocation{size, align, attributes});
    st.heap_allocated_bytes += size;
    st.update_peak_locked();
    return ptr;
}

void pg6_xbox_XMemFree(void* address, uint64_t attributes) {
    if (address == nullptr) {
        return;
    }
    playgta6::os::scarlett::XMemFreeHookFn hook = nullptr;
    {
        auto& st = playgta6::os::scarlett::xmem_state();
        std::lock_guard<std::mutex> lock(st.mutex);
        hook = st.free_hook;
    }
    if (hook != nullptr) {
        hook(address, attributes);
        return;
    }
    if (const uint64_t routine = playgta6::os::scarlett::guest_hook_routine(1); routine != 0) {
        // The title's allocator may hand out translated-window addresses: the guest routine gets
        // the address it returned, in whatever form the import thunk passed it (O31).
        playgta6::os::scarlett::call_guest_routine(routine, playgta6::os::scarlett::guest_of_arg(address), attributes);
        return;
    }
    pg6_xbox_XMemFreeDefault(address, attributes);
}

void pg6_xbox_XMemFreeDefault(void* address, uint64_t attributes) {
    (void)attributes;
    if (address == nullptr) {
        return;
    }
    using namespace playgta6::os::scarlett;
    size_t align = kMinAlignment;
    {
        auto& st = xmem_state();
        std::lock_guard<std::mutex> lock(st.mutex);
        auto it = st.heap_allocs.find(address);
        if (it == st.heap_allocs.end()) {
            playgta6::os::win32::SetLastError(playgta6::os::win32::ERROR_INVALID_PARAMETER);
            return;
        }
        align = it->second.alignment;
        st.heap_allocated_bytes -= it->second.size;
        st.heap_allocs.erase(it);
    }
    aligned_free_bytes(address, align);
}

int32_t pg6_xbox_XMemAllocatePhysicalPages(uint64_t attributes, size_t* num_pages, uint64_t* page_array) {
    using namespace playgta6::os::scarlett;
    if (num_pages == nullptr || page_array == nullptr || *num_pages == 0) {
        playgta6::os::win32::SetLastError(playgta6::os::win32::ERROR_INVALID_PARAMETER);
        return 0;
    }

    // The title sizes every page set in 64 KiB units (attributes 0x180 / 0x280 / 0x380); the
    // page-size bits of the runtime's own flag encoding are honoured for host callers.
    const uint64_t page_size = decode_physical_page_size(static_cast<uint32_t>(attributes));
    const size_t count = *num_pages;

    std::vector<std::shared_ptr<PhysicalPage>> created;
    created.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        auto page = std::make_shared<PhysicalPage>();
        page->page_size = page_size;
        page->attributes = attributes;
        created.push_back(std::move(page));
    }

    auto& st = xmem_state();
    std::lock_guard<std::mutex> lock(st.mutex);
    // Frame numbers in 4 KiB units: a page of page_size bytes takes page_size / 4 KiB numbers,
    // aligned to that (O30; see the physical page model above).
    const uint64_t frames = page_size / kXMemPageSize4K;
    for (size_t i = 0; i < count; ++i) {
        const uint64_t handle = align_up_u64(st.next_page_handle, frames);
        st.next_page_handle = handle + frames;
        created[i]->handle = handle;
        page_array[i] = handle;
        st.physical_pages.emplace(handle, std::move(created[i]));
        st.physical_pages_bytes += page_size;
    }
    st.update_peak_locked();
    return 1;
}

int32_t pg6_xbox_XMemFreePhysicalPages(size_t num_pages, const uint64_t* page_array) {
    using namespace playgta6::os::scarlett;
    if (num_pages == 0 || page_array == nullptr) {
        playgta6::os::win32::SetLastError(playgta6::os::win32::ERROR_INVALID_PARAMETER);
        return 0;
    }

    auto& st = xmem_state();
    std::unique_lock<std::mutex> lock(st.mutex);
    wait_idle_locked(st, lock);  // R40
    for (size_t i = 0; i < num_pages; ++i) {
        if (st.physical_pages.find(page_array[i]) == st.physical_pages.end()) {
            playgta6::os::win32::SetLastError(playgta6::os::win32::ERROR_INVALID_PARAMETER);
            return 0;
        }
    }
    for (size_t i = 0; i < num_pages; ++i) {
        auto it = st.physical_pages.find(page_array[i]);
        if (it != st.physical_pages.end()) {
            // Freed while mapped: the slots stop referring to it (their memory stays as is). R40:
            // only then: a walk over every mapped slot per freed page, under the lock, stalled
            // every other thread's map/unmap.
            for (auto sit = st.page_slots.begin(); it->second->mapped_slots > 0 && sit != st.page_slots.end();) {
                sit = sit->second == page_array[i] ? st.page_slots.erase(sit) : std::next(sit);
            }
            st.mapped_physical_bytes -= it->second->mapped_slots * it->second->page_size;
            st.physical_pages_bytes -= it->second->page_size;
            if (it->second->store_va != 0) {
                playgta6::os::default_guest_heap().free(it->second->store_va);
            }
            release_shared_locked(*it->second);  // views still mapped keep the memory alive
            st.physical_pages.erase(it);
        }
    }
    return 1;
}

void* pg6_xbox_XMemMapPhysicalPages(void* virtual_address, size_t num_pages, const uint64_t* page_array) {
    using namespace playgta6::os::scarlett;
    if (num_pages == 0 || (virtual_address == nullptr && page_array == nullptr)) {
        playgta6::os::win32::SetLastError(playgta6::os::win32::ERROR_INVALID_PARAMETER);
        return nullptr;
    }
    uint64_t page_size = kXMemPageSize64K;
    if (page_array != nullptr) {
        auto& st = xmem_state();
        std::lock_guard<std::mutex> lock(st.mutex);
        auto it = st.physical_pages.find(page_array[0]);
        if (it == st.physical_pages.end()) {
            playgta6::os::win32::SetLastError(playgta6::os::win32::ERROR_INVALID_PARAMETER);
            return nullptr;
        }
        page_size = it->second->page_size;
    }
    void* base = virtual_address;
    if (base == nullptr) {
        // No address: a fresh guest heap block holds the pages (released by
        // xmem_unmap_physical_pages).
        const size_t align = static_cast<size_t>(std::min<uint64_t>(page_size, kXMemPageSize64K));
        base = aligned_alloc_bytes(static_cast<size_t>(num_pages * page_size), align, true);
        if (base == nullptr) {
            playgta6::os::win32::SetLastError(playgta6::os::win32::ERROR_NOT_ENOUGH_MEMORY);
            return nullptr;
        }
        auto& st = xmem_state();
        std::lock_guard<std::mutex> lock(st.mutex);
        st.owned_blocks[guest_of(base)] = OwnedBlock{num_pages * page_size, align};
    }
    // A caller's address may be a translated-window VA (O29: guest_of_arg decodes the thunk's form).
    const uint64_t base_va = virtual_address != nullptr ? guest_of_arg(virtual_address) : guest_of(base);
    // (map_slots turns a mirrored reservation's window VAs into its arena VAs, O37.)
    std::vector<uint64_t> vas(num_pages);
    for (size_t i = 0; i < num_pages; ++i) {
        vas[i] = base_va + i * page_size;
    }
    if (!map_slots(vas.data(), num_pages, page_array)) {
        if (virtual_address == nullptr) {
            xmem_unmap_physical_pages(base);
        }
        return nullptr;
    }
    // The caller's address in the form the thunk translates back to the same VA.
    return virtual_address != nullptr ? host_of(base_va) : base;
}

int32_t pg6_xbox_XMemMapPhysicalPagesScatter(const uint64_t* virtual_addresses,
                                             size_t num_pages,
                                             const uint64_t* page_array) {
    using namespace playgta6::os::scarlett;
    if (num_pages == 0 || virtual_addresses == nullptr) {
        playgta6::os::win32::SetLastError(playgta6::os::win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    return map_slots(virtual_addresses, num_pages, page_array) ? 1 : 0;
}

void* pg6_xbox_XMemVirtualAlloc(void* address,
                                size_t size,
                                uint32_t allocation_type,
                                uint32_t xmem_flags,
                                uint32_t protect) {
    (void)xmem_flags;  // memory category for the title's budgets; one guest pool here
    using namespace playgta6::os::scarlett;
    namespace os = playgta6::os;
    if (size == 0) {
        os::win32::SetLastError(os::win32::ERROR_INVALID_PARAMETER);
        return nullptr;
    }
    // Console-only flags (MEM_LARGE_PAGES, MEM_TOP_DOWN, MEM_WRITE_WATCH, ...) are accepted; only
    // MEM_RESERVE / MEM_COMMIT select the operation, large pages are normal 64 KiB pages.
    const bool want_reserve = (allocation_type & os::kMemReserve) != 0;
    const bool want_commit = (allocation_type & os::kMemCommit) != 0;
    os::PageProtect pp = os::PageProtect::NoAccess;
    if ((!want_reserve && !want_commit) || !xmem_protect(protect, pp)) {
        os::win32::SetLastError(os::win32::ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    const uint64_t page_align =
        (allocation_type & kXMemFlagLargePages) != 0 ? kXMemPageSize64K : os::kPageSize;
    const uint64_t aligned_size = align_up_u64(size, page_align);
    if (aligned_size < size) {
        os::win32::SetLastError(os::win32::ERROR_NOT_ENOUGH_MEMORY);
        return nullptr;
    }

    auto& st = xmem_state();
    std::unique_lock<std::mutex> lock(st.mutex);
    wait_idle_locked(st, lock);  // R40: commits in place, replaces placeholders under mapped slots
    os::VirtualMemoryArena& arena = xmem_arena();
    // O37: an address inside a reservation handed out at a window range means its arena VA (the
    // import thunk passes the window VA's host page; a direct caller may pass the window VA).
    if (address != nullptr && !st.mirrors.empty()) {
        const uint64_t g = unmirror(guest_of_arg(address));
        if (mirror_window_va_locked(st, g) != 0) {
            address = host_of(g);
        }
    }

    // Committing an existing XMemVirtualAlloc reservation in place.
    if (address != nullptr && !want_reserve && want_commit) {
        VirtualRegion* existing =
            st.find_virtual_region_locked(address, static_cast<size_t>(aligned_size));
        if (existing != nullptr) {
            if (!existing->committed) {
                if (!arena.commit(guest_of(existing->base), existing->size, pp)) {
                    os::win32::SetLastError(os::win32::ERROR_NOT_ENOUGH_MEMORY);
                    return nullptr;
                }
                existing->committed = true;
                existing->protect = protect;
                st.virtual_committed_bytes += existing->size;
                st.update_peak_locked();
            }
            if (const uint64_t w = mirror_window_va_locked(st, guest_of(existing->base))) {
                return host_of(w);  // O37: a mirrored region, in the title's form
            }
            return existing->base;
        }
        // Not one of ours: plain VirtualAlloc semantics on whatever the arena has reserved there.
        const uint64_t committed =
            arena.virtual_alloc(guest_of(address), aligned_size, os::kMemCommit, pp);
        if (committed == 0) {
            os::win32::SetLastError(os::win32::ERROR_INVALID_PARAMETER);
            return nullptr;
        }
        if (const uint64_t w = mirror_window_va_locked(st, committed)) {
            return host_of(w);
        }
        return host_of(committed);
    }

    // O27: CPU / GPU windows (xmem_windows.h). Under the phantom policy a GPU window gets an
    // address outside the guest pool and no arena reservation.
    playgta6::os::xmem_windows::WindowKind window_kind{};
    const bool window =
        address == nullptr && playgta6::os::xmem_windows::classify(allocation_type, protect, aligned_size, &window_kind);
    const bool phantom = window && playgta6::os::xmem_windows::place_phantom(window_kind);
    // R19: the (first) CPU window goes to the translated range when that policy and the window
    // table are on; it needs no arena reservation.
    const bool translated = window && !phantom && !st.translated_window_live &&
                            playgta6::os::xmem_windows::place_translated(window_kind) &&
                            os::guest_window::enabled() && aligned_size <= os::guest_window::kWindowMaxSize &&
                            !want_commit;

    // A non-null address is a placement request that must be honoured exactly.
    // Placeholders (MEM_RESERVE_PLACEHOLDER / MEM_REPLACE_PLACEHOLDER) are the arena's.
    const uint32_t placeholder_bits =
        allocation_type & (os::kMemReservePlaceholder | os::kMemReplacePlaceholder);
    const uint32_t arena_type =
        (want_reserve ? os::kMemReserve : 0u) | (want_commit ? os::kMemCommit : 0u) | placeholder_bits;
    uint64_t va = 0;
    bool shared = false;
    if (phantom) {
        va = playgta6::os::xmem_windows::reserve_phantom(aligned_size);
    } else if (translated) {
        va = os::guest_window::kWindowBase;
        st.translated_window_live = true;
    } else {
        // O36: native hosts: physical-page reservations (CPU windows, pool views and smaller ones:
        // the title maps its physical pages into them, the same page at several addresses) go to
        // the arena's shared zone, where every slot is a view of the page's own memory. GPU
        // windows stay in the arena (D3D12 reads their pages through the CPU window); placeholders
        // are the arena's.
        const bool physical_reserve = !want_commit && placeholder_bits == 0 &&
                                      (allocation_type & playgta6::os::xmem_windows::kMemPhysical) != 0;
        const bool gpu_window = window && window_kind == playgta6::os::xmem_windows::WindowKind::Gpu;
        if (physical_reserve && !gpu_window && address == nullptr && !os::guest_window::enabled() &&
            arena.has_shared_zone() && (aligned_size % kXMemPageSize64K) == 0) {
            va = arena.reserve_shared(aligned_size);
            if (va != 0) {
                shared = true;
            } else if (!st.shared_exhausted_logged) {
                st.shared_exhausted_logged = true;
                std::fprintf(stderr,
                             "[xmem] no shared-zone room for a %llu-byte window: it is placed in the arena "
                             "(its pages mapped at several addresses get copies)\n",
                             static_cast<unsigned long long>(aligned_size));
            }
        }
        if (va == 0) {
            try {
                va = arena.virtual_alloc(guest_of(address), aligned_size, arena_type, pp);
            } catch (const std::bad_alloc&) {
                va = 0;
            }
        }
    }
    if (va == 0) {
        os::win32::SetLastError(os::win32::ERROR_NOT_ENOUGH_MEMORY);
        return nullptr;
    }
    // O37: under the window memory models the title's XMemVirtualAlloc regions are handed out at
    // a window range above the CPU window, backed by this arena reservation slot for slot: stores
    // into them carry write marks. That covers its physical-page reservations, the placeholders it
    // splits into them, and (O36) its committed regions and the CPU windows past the translated one,
    // which it sub-allocates GPU buffers from. The first of those CPU windows is the page pool's
    // view (xmem_windows.h): the title never reads it and the keeper's frames live in it, so it
    // stays in the arena, as do plain reservations. XMem keeps using the arena VA; its result is
    // the window VA's form. Without room a region stays where it is (logged once).
    uint64_t mirror = 0;
    bool mirrored_window = false;
    if (window && window_kind == playgta6::os::xmem_windows::WindowKind::Cpu && !translated && !phantom &&
        st.translated_window_live) {
        mirrored_window = st.pool_view_placed;
        st.pool_view_placed = true;
    }
    const bool mirrored_kind =
        mirrored_window ||
        (!window && (want_commit || (allocation_type & os::kMemReservePlaceholder) != 0 ||
                     (allocation_type & playgta6::os::xmem_windows::kMemPhysical) != 0));
    if (address == nullptr && mirrored_kind && st.translated_window_live &&
        os::guest_window::enabled() && (va & (kXMemPageSize64K - 1)) == 0) {
        const uint64_t bytes = align_up_u64(aligned_size, kXMemPageSize64K);
        mirror = os::guest_window::map_runtime_range(host_of(va), bytes);
        if (mirror != 0) {
            st.mirrors[va] = {mirror, bytes};
        } else if (!st.mirror_exhausted_logged) {
            st.mirror_exhausted_logged = true;
            std::fprintf(stderr,
                         "[xmem] no window range left for a %llu-byte region: it stays in the arena at "
                         "0x%llx (stores into it carry no write marks)\n",
                         static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(va));
        }
    }
    if ((allocation_type & os::kMemReservePlaceholder) != 0) {
        // A placeholder is not usable memory: it is split with VirtualFree and its pieces are
        // replaced by XMemVirtualAlloc(MEM_REPLACE_PLACEHOLDER), which become regions here.
        return mirror != 0 ? host_of(mirror) : host_of(va);
    }

    VirtualRegion region;
    region.base = host_of(va);
    region.size = static_cast<size_t>(aligned_size);
    region.alignment = static_cast<size_t>(kXMemPageSize64K);
    region.allocation_type = allocation_type;
    region.protect = protect;
    region.committed = want_commit;
    region.phantom = phantom;
    region.cpu_window = window && window_kind == playgta6::os::xmem_windows::WindowKind::Cpu;
    region.translated = translated;
    region.shared = shared;

    if (!phantom && !translated) {
        st.virtual_reserved_bytes += aligned_size;
    }
    if (want_commit) {
        st.virtual_committed_bytes += aligned_size;
    }
    st.virtual_regions.emplace(region.base, region);
    st.region_index[va] = region.base;
    st.update_peak_locked();
    if (window) {
        playgta6::os::xmem_windows::Window w;
        w.base = va;
        w.size = aligned_size;
        w.kind = window_kind;
        w.protect = protect;
        w.phantom = phantom;
        playgta6::os::xmem_windows::note_window(w);  // may call alias listeners (they must not call back into xmem)
    }
    if (mirror != 0) {
        return host_of(mirror);  // the window VA (the thunk hands the title this address)
    }
    if (const uint64_t w = mirror_window_va_locked(st, va)) {
        return host_of(w);  // a piece replacing part of a mirrored placeholder
    }
    return region.base;
}

Pg6HResult pg6_xbox_XMemTransferMemory(void* dst, const void* src, size_t size, uint32_t flags) {
    (void)flags;
    if (size == 0) {
        return playgta6::os::gdk::S_OK_HR;
    }
    if (dst == nullptr || src == nullptr) {
        playgta6::os::win32::SetLastError(playgta6::os::win32::ERROR_INVALID_PARAMETER);
        return playgta6::os::gdk::E_INVALIDARG_HR;
    }
    std::memmove(dst, src, size);
    return playgta6::os::gdk::S_OK_HR;
}

Pg6HResult pg6_xbox_XMemGetWorkingSetStatistics(Pg6XMemWorkingSetStats* stats) {
    using namespace playgta6::os::scarlett;
    if (stats == nullptr) {
        playgta6::os::win32::SetLastError(playgta6::os::win32::ERROR_INVALID_PARAMETER);
        return playgta6::os::gdk::E_INVALIDARG_HR;
    }

    auto& st = xmem_state();
    std::lock_guard<std::mutex> lock(st.mutex);
    const uint64_t used =
        st.heap_allocated_bytes + st.physical_pages_bytes + st.virtual_committed_bytes;

    stats->total_physical_bytes = kConsoleTotalRamBytes;
    stats->title_budget_bytes = kConsoleTitleBudgetBytes;
    stats->heap_allocated_bytes = st.heap_allocated_bytes;
    stats->physical_pages_bytes = st.physical_pages_bytes;
    stats->virtual_committed_bytes = st.virtual_committed_bytes;
    stats->virtual_reserved_bytes = st.virtual_reserved_bytes;
    stats->mapped_physical_bytes = st.mapped_physical_bytes;
    stats->peak_working_set_bytes = st.peak_working_set_bytes;
    stats->available_title_bytes =
        used >= kConsoleTitleBudgetBytes ? 0ULL : (kConsoleTitleBudgetBytes - used);
    stats->tools_mem_bytes = 0ULL;
    stats->live_heap_allocations = static_cast<uint32_t>(st.heap_allocs.size());
    stats->live_physical_pages = static_cast<uint32_t>(st.physical_pages.size());
    stats->live_virtual_mappings = static_cast<uint32_t>(st.page_slots.size());  // mapped page slots
    stats->reserved = 0u;
    return playgta6::os::gdk::S_OK_HR;
}

int32_t pg6_xbox_XMemIsToolsMemAvailable(void) {
    return 0;
}

Pg6HResult pg6_xbox_DStorageGetFactory(const Pg6Guid* riid, void** ppv) {
    using namespace playgta6::os::scarlett;
    if (ppv == nullptr) {
        return playgta6::os::gdk::E_POINTER_HR;
    }
    *ppv = nullptr;
    if (riid != nullptr && *riid != kIidIUnknown && *riid != kIidIDStorageFactory) {
        return E_NOINTERFACE_HR;
    }
    auto* factory = new (std::nothrow) DStorageFactoryImpl();
    if (factory == nullptr) {
        return playgta6::os::gdk::E_OUTOFMEMORY_HR;
    }
    *ppv = static_cast<Pg6IDStorageFactory*>(factory);
    return playgta6::os::gdk::S_OK_HR;
}

Pg6HResult pg6_xbox_XGCreateTextureComputer(const Pg6XgTextureDesc* desc,
                                            Pg6IXGTextureAddressComputer** out_computer) {
    using namespace playgta6::os::scarlett;
    if (out_computer == nullptr) {
        return playgta6::os::gdk::E_POINTER_HR;
    }
    *out_computer = nullptr;
    if (desc == nullptr || desc->width == 0 || desc->depth_or_array_size == 0) {
        return playgta6::os::gdk::E_INVALIDARG_HR;
    }
    if (desc->dimension != kXgDimensionTexture1D && desc->height == 0) {
        return playgta6::os::gdk::E_INVALIDARG_HR;
    }
    auto* comp = new (std::nothrow) XGTextureAddressComputerImpl(*desc);
    if (comp == nullptr) {
        return playgta6::os::gdk::E_OUTOFMEMORY_HR;
    }
    *out_computer = static_cast<Pg6IXGTextureAddressComputer*>(comp);
    return playgta6::os::gdk::S_OK_HR;
}

uint32_t pg6_xbox_XGSuggestSwizzleMode(const Pg6XgTextureDesc* desc) {
    using namespace playgta6::os::scarlett;
    if (desc == nullptr || desc->width == 0) {
        return kXgTileModeInvalid;
    }
    // WebGPU textures consume linear row-major upload layouts cleanly without GPU unswizzling.
    return kXgTileModeLinear;
}

Pg6HResult pg6_xbox_GameInputCreate(Pg6IGameInput** game_input) {
    using namespace playgta6::os::scarlett;
    if (game_input == nullptr) {
        return playgta6::os::gdk::E_POINTER_HR;
    }
    *game_input = nullptr;
    auto* impl = new (std::nothrow) GameInputImpl();
    if (impl == nullptr) {
        return playgta6::os::gdk::E_OUTOFMEMORY_HR;
    }
    *game_input = static_cast<Pg6IGameInput*>(impl);
    return playgta6::os::gdk::S_OK_HR;
}

size_t pg6_xbox_scarlett_register_all(void) {
    return playgta6::os::scarlett::register_all_scarlett_exports();
}

}  // extern "C"
