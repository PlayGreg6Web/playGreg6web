// Host backing of VirtualMemoryArena (see vmem.h): the only file in runtime/os/ that talks to the
// OS memory manager. One section per platform; nothing here leaks into headers.
#include "os/vmem_backing.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <utility>
#elif defined(__EMSCRIPTEN__)
#include <emscripten/heap.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstring>
#include <iterator>
#include <map>
#include <mutex>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#endif

namespace playgta6::os::backing {

#if defined(_WIN32)

namespace {

std::atomic<bool> g_write_watch{false};

// O33: the reservations made with MEM_WRITE_WATCH (base -> end). A take asks GetWriteWatch only
// inside one of them: anywhere else (zone placeholders, untracked arenas) the call fails anyway,
// and a failing call costs as much as a successful one. A leaf lock.
std::shared_mutex g_watched_mutex;
std::map<uintptr_t, uintptr_t> g_watched;

void note_watched(const void* base, uint64_t size) {
    std::unique_lock<std::shared_mutex> lock(g_watched_mutex);
    g_watched[reinterpret_cast<uintptr_t>(base)] = reinterpret_cast<uintptr_t>(base) + static_cast<uintptr_t>(size);
}

bool watched_covers(uintptr_t lo, uintptr_t hi, uintptr_t* res_lo = nullptr, uintptr_t* res_hi = nullptr) {
    std::shared_lock<std::shared_mutex> lock(g_watched_mutex);
    auto it = g_watched.upper_bound(lo);
    if (it == g_watched.begin() || hi > std::prev(it)->second) {
        return false;
    }
    if (res_lo != nullptr) {
        *res_lo = std::prev(it)->first;
        *res_hi = std::prev(it)->second;
    }
    return true;
}

DWORD win_protect(PageProtect protect) {
    if (is_writable(protect)) {
        return PAGE_READWRITE;
    }
    return is_readable(protect) ? PAGE_READONLY : PAGE_NOACCESS;
}

}  // namespace

uint8_t* reserve(uint64_t size, uint64_t preferred) {
    // MEM_WRITE_WATCH: the memory manager records written pages (take_written_pages), which lets
    // the GPU layer find changed CPU-visible buffer pages without comparing whole buffers.
    const DWORD type = MEM_RESERVE | (g_write_watch.load(std::memory_order_relaxed) ? MEM_WRITE_WATCH : 0);
    void* p = VirtualAlloc(reinterpret_cast<void*>(static_cast<uintptr_t>(preferred)),
                           static_cast<SIZE_T>(size), type, PAGE_NOACCESS);
    if (p != nullptr && preferred != 0 && reinterpret_cast<uintptr_t>(p) != preferred) {
        VirtualFree(p, 0, MEM_RELEASE);
        return nullptr;
    }
    if (p != nullptr && (type & MEM_WRITE_WATCH) != 0) {
        note_watched(p, size);
    }
    return static_cast<uint8_t*>(p);
}

// O36: shared zones (placeholders for shared page views). VirtualAlloc2, MapViewOfFile3 and
// UnmapViewOfFile2 are looked up at run time (Windows 10 1803+), so older hosts keep working
// without shared pages.
namespace {

using VirtualAlloc2Fn = PVOID(WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, void*, ULONG);
using MapViewOfFile3Fn = PVOID(WINAPI*)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T, ULONG, ULONG, void*, ULONG);
using UnmapViewOfFile2Fn = BOOL(WINAPI*)(HANDLE, PVOID, ULONG);

struct PlaceholderApi {
    VirtualAlloc2Fn virtual_alloc2 = nullptr;
    MapViewOfFile3Fn map_view3 = nullptr;
    UnmapViewOfFile2Fn unmap_view2 = nullptr;
    bool ok() const { return virtual_alloc2 != nullptr && map_view3 != nullptr && unmap_view2 != nullptr; }
};

const PlaceholderApi& placeholder_api() {
    static const PlaceholderApi api = [] {
        PlaceholderApi a;
        if (HMODULE k = GetModuleHandleW(L"kernelbase.dll")) {
            a.virtual_alloc2 = reinterpret_cast<VirtualAlloc2Fn>(GetProcAddress(k, "VirtualAlloc2"));
            a.map_view3 = reinterpret_cast<MapViewOfFile3Fn>(GetProcAddress(k, "MapViewOfFile3"));
            a.unmap_view2 = reinterpret_cast<UnmapViewOfFile2Fn>(GetProcAddress(k, "UnmapViewOfFile2"));
        }
        return a;
    }();
    return api;
}

#ifndef MEM_RESERVE_PLACEHOLDER
#define MEM_RESERVE_PLACEHOLDER 0x00040000
#endif
#ifndef MEM_REPLACE_PLACEHOLDER
#define MEM_REPLACE_PLACEHOLDER 0x00004000
#endif
#ifndef MEM_PRESERVE_PLACEHOLDER
#define MEM_PRESERVE_PLACEHOLDER 0x00000002
#endif

// O33: the zone table is read on every fault that is not a tracked view: a reader/writer lock.
std::shared_mutex g_zone_mutex;
std::map<uintptr_t, std::pair<uintptr_t, uintptr_t>> g_zones;  // reservation base -> [zone start, end)

bool in_zone(uintptr_t addr) {
    std::shared_lock<std::shared_mutex> lock(g_zone_mutex);
    for (const auto& [base, zone] : g_zones) {
        if (addr >= zone.first && addr < zone.second) {
            return true;
        }
    }
    return false;
}

// R40: write tracking of shared page views. A view of a file mapping cannot carry
// MEM_WRITE_WATCH, so take_written_pages tracks them itself: every view of a page (one memory
// object) is made read-only ("armed") when a caller takes its written pages, and the first write
// through any of them faults into track_fault, which marks every view of the page written and
// makes them writable again (one fault per page and taking interval, none for pages no caller
// tracks). A write is stamped on every view of the page (a write through one changes what the
// others show) with a clock each caller compares with its own (WriteCursor::view_clock), so callers
// whose ranges overlap all see it.
//
// O33: two lock levels instead of one global mutex, so the write-fault handler, takes (flush and
// re-arm), host-write pins and maps of other pages no longer serialise on each other:
//   g_track_mutex (reader/writer): the view and page tables. Exclusive only to add or remove a
//     view (map/unmap); faults, takes, pins and guest protections hold it shared.
//   TrackedPage::mutex: one page's state (armed, pins, untracked) and its views' stamps.
// Lock order: the arena's mutex, then g_track_mutex, then one TrackedPage::mutex at a time (leaves:
// nothing here writes guest memory while holding them). The clock is atomic; see
// take_written_pages for why a take reports against the clock read before it starts.
struct TrackedPage;
struct TrackedView {
    uint64_t size = 0;
    void* page = nullptr;
    TrackedPage* tracked = nullptr;  // g_tracked_pages[page] (map nodes are stable)
    uint64_t stamp = 0;  // g_track_clock at the last write seen (or mapping: new contents); page mutex
};
struct TrackedPage {
    std::mutex mutex;
    std::vector<uintptr_t> views;  // (changed with g_track_mutex held exclusively)
    bool armed = false;      // its views are read-only: the next write faults
    bool untracked = false;  // a view got a guest protection: always reported written, never armed
    uint32_t pins = 0;       // host writes in flight (pin_views): never armed, reported written
};
std::shared_mutex g_track_mutex;
std::map<uintptr_t, TrackedView> g_views;  // view base -> view, guarded by g_track_mutex
std::map<void*, TrackedPage> g_tracked_pages;
std::atomic<uint64_t> g_track_clock{1};
std::once_flag g_track_handler_once;
std::atomic<uint64_t> g_track_faults{0};

// The `*_page_locked` helpers: g_track_mutex held (either way) and the page's mutex held.
void protect_views_page_locked(const TrackedPage& page, DWORD protect) {
    for (uintptr_t v : page.views) {
        DWORD old = 0;
        VirtualProtect(reinterpret_cast<void*>(v), static_cast<SIZE_T>(g_views.find(v)->second.size), protect, &old);
    }
}

// The page's views become writable, each stamped written.
void disarm_page_locked(TrackedPage& page) {
    const uint64_t stamp = g_track_clock.fetch_add(1) + 1;
    for (uintptr_t v : page.views) {
        g_views.find(v)->second.stamp = stamp;
    }
    if (page.armed) {
        page.armed = false;
        protect_views_page_locked(page, PAGE_READWRITE);
    }
}

void arm_page_locked(TrackedPage& page) {
    if (page.pins != 0) {
        return;  // a host write is in flight: the kernel cannot fault into track_fault
    }
    for (uintptr_t v : page.views) {
        DWORD old = 0;
        if (!VirtualProtect(reinterpret_cast<void*>(v), static_cast<SIZE_T>(g_views.find(v)->second.size),
                            PAGE_READONLY, &old)) {
            page.armed = true;  // (so disarm restores the views armed so far)
            disarm_page_locked(page);
            page.untracked = true;
            return;
        }
    }
    page.armed = true;
}

LONG CALLBACK track_fault(EXCEPTION_POINTERS* info) {
    const EXCEPTION_RECORD* rec = info->ExceptionRecord;
    if (rec->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || rec->NumberParameters < 2 ||
        rec->ExceptionInformation[0] != 1) {  // a write
        return EXCEPTION_CONTINUE_SEARCH;
    }
    const uintptr_t addr = static_cast<uintptr_t>(rec->ExceptionInformation[1]);
    bool view = false;
    {
        std::shared_lock<std::shared_mutex> lock(g_track_mutex);
        auto it = g_views.upper_bound(addr);
        if (it != g_views.begin() && addr - std::prev(it)->first < std::prev(it)->second.size) {
            view = true;
            TrackedPage& page = *std::prev(it)->second.tracked;
            std::lock_guard<std::mutex> page_lock(page.mutex);
            if (page.armed) {
                disarm_page_locked(page);
                g_track_faults.fetch_add(1, std::memory_order_relaxed);
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            if (!page.untracked) {
                // O33 fast path: another thread disarmed the page first, so its views are writable
                // (a take re-arming them meanwhile only faults the write in again).
                return EXCEPTION_CONTINUE_EXECUTION;
            }
        }
    }
    // A view with a guest protection, or a view unmapped after it was disarmed: the write goes
    // through when the page is writable now. Anything else is a real fault.
    MEMORY_BASIC_INFORMATION mbi{};
    if ((view || in_zone(addr)) && VirtualQuery(reinterpret_cast<void*>(addr), &mbi, sizeof(mbi)) != 0 &&
        mbi.State == MEM_COMMIT && mbi.Protect == PAGE_READWRITE) {
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

// g_track_mutex held exclusively.
void forget_view_locked(uintptr_t at) {
    auto it = g_views.find(at);
    if (it == g_views.end()) {
        return;
    }
    if (auto pit = g_tracked_pages.find(it->second.page); pit != g_tracked_pages.end()) {
        TrackedPage& page = pit->second;
        bool empty = false;
        {
            std::lock_guard<std::mutex> page_lock(page.mutex);
            disarm_page_locked(page);  // a write racing with the unmap reaches the page's other views
            page.views.erase(std::remove(page.views.begin(), page.views.end(), at), page.views.end());
            empty = page.views.empty();
        }
        if (empty) {
            g_tracked_pages.erase(pit);
        }
    }
    g_views.erase(it);
}

void track_view(uint8_t* at, void* page, uint64_t size) {
    const uintptr_t v = reinterpret_cast<uintptr_t>(at);
    std::unique_lock<std::shared_mutex> lock(g_track_mutex);
    forget_view_locked(v);
    TrackedPage& tp = g_tracked_pages[page];
    std::lock_guard<std::mutex> page_lock(tp.mutex);
    tp.views.push_back(v);
    g_views[v] = TrackedView{size, page, &tp, 0};
    disarm_page_locked(tp);  // the new view is writable: so are the others until the next take
}

void untrack_view(uint8_t* at) {
    std::unique_lock<std::shared_mutex> lock(g_track_mutex);
    forget_view_locked(reinterpret_cast<uintptr_t>(at));
}

void untrack_range(uintptr_t lo, uintptr_t hi) {
    std::unique_lock<std::shared_mutex> lock(g_track_mutex);
    for (auto it = g_views.lower_bound(lo); it != g_views.end() && it->first < hi;) {
        const uintptr_t v = it->first;
        ++it;
        forget_view_locked(v);
    }
}

// A guest protection of [p, p + len): views it touches become writable and their pages
// untracked unless it is plain read/write.
void track_protect(uint8_t* p, uint64_t len, bool read_write) {
    const uintptr_t lo = reinterpret_cast<uintptr_t>(p);
    const uintptr_t hi = lo + static_cast<uintptr_t>(len);
    std::shared_lock<std::shared_mutex> lock(g_track_mutex);
    auto it = g_views.upper_bound(lo);
    if (it != g_views.begin()) {
        --it;
    }
    for (; it != g_views.end() && it->first < hi; ++it) {
        if (it->first + it->second.size <= lo) {
            continue;
        }
        TrackedPage& page = *it->second.tracked;
        std::lock_guard<std::mutex> page_lock(page.mutex);
        disarm_page_locked(page);
        page.untracked = page.untracked || !read_write;
    }
}

void all_written(uintptr_t lo, uintptr_t hi, uintptr_t origin, std::vector<uint64_t>* offsets) {
    const uintptr_t first = lo & ~(uintptr_t{kWrittenPageSize} - 1);
    if (first >= hi) {
        return;
    }
    const size_t at = offsets->size();
    offsets->resize(at + static_cast<size_t>((hi - first + kWrittenPageSize - 1) / kWrittenPageSize));
    uint64_t* out = offsets->data() + at;
    for (uintptr_t page = first; page < hi; page += kWrittenPageSize) {
        *out++ = static_cast<uint64_t>(page - origin);
    }
}

}  // namespace

uint64_t tracked_view_faults() {
    return g_track_faults.load(std::memory_order_relaxed);
}

void pin_views(uintptr_t lo, uintptr_t hi, bool pin) {
    std::shared_lock<std::shared_mutex> lock(g_track_mutex);
    if (g_views.empty()) {
        return;
    }
    auto it = g_views.upper_bound(lo);
    if (it != g_views.begin()) {
        --it;
    }
    void* last = nullptr;  // (consecutive views of one page: pinned once)
    for (; it != g_views.end() && it->first < hi; ++it) {
        if (it->first + it->second.size <= lo || it->second.page == last) {
            continue;
        }
        last = it->second.page;
        TrackedPage& page = *it->second.tracked;
        std::lock_guard<std::mutex> page_lock(page.mutex);
        if (pin) {
            ++page.pins;
        } else if (page.pins != 0) {
            --page.pins;
        }
        // Writable for the kernel before the write, stamped written again after it (a take
        // between the two saw the page pinned, i.e. written, and left it unarmed).
        disarm_page_locked(page);
    }
}

namespace {
void drop_stamps(uintptr_t lo, uintptr_t hi);  // O36 (below)
}  // namespace

void release(uint8_t* base, uint64_t /*size*/) {
    {
        uintptr_t watched_end = 0;
        {
            std::unique_lock<std::shared_mutex> lock(g_watched_mutex);
            if (auto it = g_watched.find(reinterpret_cast<uintptr_t>(base)); it != g_watched.end()) {
                watched_end = it->second;
                g_watched.erase(it);
            }
        }
        if (watched_end > reinterpret_cast<uintptr_t>(base)) {
            drop_stamps(reinterpret_cast<uintptr_t>(base), watched_end);
        }
    }
    std::pair<uintptr_t, uintptr_t> zone{0, 0};
    {
        std::unique_lock<std::shared_mutex> lock(g_zone_mutex);
        if (auto it = g_zones.find(reinterpret_cast<uintptr_t>(base)); it != g_zones.end()) {
            zone = it->second;
            g_zones.erase(it);
        }
    }
    if (zone.second != 0) {
        untrack_range(zone.first, zone.second);
    }
    if (zone.first != 0 && zone.first == reinterpret_cast<uintptr_t>(base)) {
        zone.first = 0;  // (no lower part)
    }
    if (zone.second != 0) {
        // The zone: views (unmapped, then their placeholders) and placeholders, piece by piece.
        uintptr_t at = zone.first != 0 ? zone.first : reinterpret_cast<uintptr_t>(base);
        while (at < zone.second) {
            MEMORY_BASIC_INFORMATION mbi{};
            if (VirtualQuery(reinterpret_cast<void*>(at), &mbi, sizeof(mbi)) == 0 || mbi.RegionSize == 0) {
                break;
            }
            if (mbi.Type == MEM_MAPPED) {
                UnmapViewOfFile(mbi.BaseAddress);
            } else if (mbi.State != MEM_FREE) {
                VirtualFree(mbi.AllocationBase, 0, MEM_RELEASE);
            }
            at = reinterpret_cast<uintptr_t>(mbi.BaseAddress) + mbi.RegionSize;
        }
        if (zone.first == 0) {
            return;
        }
    }
    VirtualFree(base, 0, MEM_RELEASE);
}

bool shared_pages_supported() {
    return placeholder_api().ok();
}

uint8_t* reserve_with_shared_zone(uint64_t size, uint64_t preferred, uint64_t zone_offset) {
    const PlaceholderApi& api = placeholder_api();
    if (!api.ok() || zone_offset >= size || zone_offset % (64u << 10) != 0) {
        return reserve(size, preferred);
    }
    void* p = api.virtual_alloc2(GetCurrentProcess(), reinterpret_cast<void*>(static_cast<uintptr_t>(preferred)),
                                 static_cast<SIZE_T>(size), MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS,
                                 nullptr, 0);
    if (p == nullptr) {
        return nullptr;
    }
    if (preferred != 0 && reinterpret_cast<uintptr_t>(p) != preferred) {
        VirtualFree(p, 0, MEM_RELEASE);
        return nullptr;
    }
    uint8_t* base = static_cast<uint8_t*>(p);
    if (zone_offset != 0) {
        // Split off the lower part and make it an ordinary reservation (with write tracking when
        // asked for and the host allows it on a replaced placeholder).
        if (!VirtualFree(base, static_cast<SIZE_T>(zone_offset), MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
            VirtualFree(base, 0, MEM_RELEASE);
            return nullptr;
        }
        const ULONG ww = g_write_watch.load(std::memory_order_relaxed) ? MEM_WRITE_WATCH : 0;
        void* lower = api.virtual_alloc2(GetCurrentProcess(), base, static_cast<SIZE_T>(zone_offset),
                                         MEM_RESERVE | MEM_REPLACE_PLACEHOLDER | ww, PAGE_NOACCESS, nullptr, 0);
        if (lower != nullptr && ww != 0) {
            note_watched(base, zone_offset);
        } else if (lower == nullptr && ww != 0) {
            lower = api.virtual_alloc2(GetCurrentProcess(), base, static_cast<SIZE_T>(zone_offset),
                                       MEM_RESERVE | MEM_REPLACE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0);
        }
        if (lower == nullptr) {
            VirtualFree(base, 0, MEM_RELEASE);
            VirtualFree(base + zone_offset, 0, MEM_RELEASE);
            return nullptr;
        }
    }
    std::unique_lock<std::shared_mutex> lock(g_zone_mutex);
    g_zones[reinterpret_cast<uintptr_t>(base)] = {reinterpret_cast<uintptr_t>(base + zone_offset),
                                                  reinterpret_cast<uintptr_t>(base + size)};
    return base;
}

void* create_shared_page(uint64_t size) {
    HANDLE h = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, static_cast<DWORD>(size >> 32),
                                  static_cast<DWORD>(size & 0xFFFFFFFFu), nullptr);
    return h;
}

void release_shared_page(void* page) {
    if (page != nullptr) {
        CloseHandle(page);
    }
}

bool map_shared_page(uint8_t* at, void* page, uint64_t size) {
    const PlaceholderApi& api = placeholder_api();
    if (!api.ok() || page == nullptr) {
        return false;
    }
    // [at, at + size) must be a placeholder of its own: split it out of a larger one. Other
    // threads split the same placeholder concurrently (R40: no lock around this), so a split that
    // fails is retried against what the range has become.
    for (int attempt = 0;; ++attempt) {
        MEMORY_BASIC_INFORMATION mbi{};
        if (VirtualQuery(at, &mbi, sizeof(mbi)) == 0 || mbi.State != MEM_RESERVE) {
            return false;
        }
        if (mbi.AllocationBase == at && mbi.RegionSize == size) {  // (BaseAddress is always `at`)
            break;
        }
        if (VirtualFree(at, static_cast<SIZE_T>(size), MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
            break;
        }
        if (attempt == 16) {
            return false;
        }
    }
    if (api.map_view3(page, GetCurrentProcess(), at, 0, static_cast<SIZE_T>(size), MEM_REPLACE_PLACEHOLDER,
                      PAGE_READWRITE, nullptr, 0) != at) {
        return false;
    }
    track_view(at, page, size);
    return true;
}

bool unmap_shared_page(uint8_t* at, uint64_t /*size*/) {
    const PlaceholderApi& api = placeholder_api();
    if (!api.ok()) {
        return false;
    }
    untrack_view(at);
    return api.unmap_view2(GetCurrentProcess(), at, MEM_PRESERVE_PLACEHOLDER) != 0;
}

uint8_t* view_shared_page(void* page, uint64_t size) {
    return static_cast<uint8_t*>(MapViewOfFile(page, FILE_MAP_ALL_ACCESS, 0, 0, static_cast<SIZE_T>(size)));
}

void unview_shared_page(uint8_t* view, uint64_t /*size*/) {
    if (view != nullptr) {
        UnmapViewOfFile(view);
    }
}

bool commit(uint8_t* p, uint64_t len, PageProtect protect) {
    // MEM_COMMIT of reserved (or decommitted) pages yields zero-filled pages.
    return VirtualAlloc(p, static_cast<SIZE_T>(len), MEM_COMMIT, win_protect(protect)) != nullptr;
}

bool decommit(uint8_t* p, uint64_t len) {
    return VirtualFree(p, static_cast<SIZE_T>(len), MEM_DECOMMIT) != 0;
}

#ifndef MEM_RESET
#define MEM_RESET 0x00080000
#endif
#ifndef MEM_RESET_UNDO
#define MEM_RESET_UNDO 0x01000000
#endif

bool reset_pages(uint8_t* p, uint64_t len) {
    if (p == nullptr || len == 0) {
        return false;
    }
    return VirtualAlloc(p, static_cast<SIZE_T>(len), MEM_RESET, PAGE_NOACCESS) != nullptr;
}

bool reset_undo_pages(uint8_t* p, uint64_t len) {
    if (p == nullptr || len == 0) {
        return false;
    }
    return VirtualAlloc(p, static_cast<SIZE_T>(len), MEM_RESET_UNDO, PAGE_NOACCESS) != nullptr;
}

bool protect(uint8_t* p, uint64_t len, PageProtect protect) {
    track_protect(p, len, win_protect(protect) == PAGE_READWRITE);
    DWORD old = 0;
    return VirtualProtect(p, static_cast<SIZE_T>(len), win_protect(protect), &old) != 0;
}

bool enforces_protection() {
    return true;
}

void set_write_watch(bool on) {
    g_write_watch.store(on, std::memory_order_relaxed);
}

namespace {

// O36: GetWriteWatch(RESET) is destructive, so the pages it reports are fanned out here: every
// take stamps the pages it consumed with a new epoch, and each reader keeps the epoch it has seen
// (WriteCursor::watch_epoch) and is told the pages stamped after it. Readers of the same pages
// (buffer flush, texture sync) therefore never steal a write from each other. Stamps live in
// 2 MiB chunks allocated on the first write (512 x u32 plus the newest stamp of the chunk); a
// released reservation drops its chunks. Stamping, the GetWriteWatch call itself and the reader's
// scan run under one mutex, so a reader's new epoch covers exactly what it was shown.
constexpr uintptr_t kStampChunkShift = 21;
constexpr size_t kStampChunkPages = size_t{1} << (kStampChunkShift - 12);
struct StampChunk {
    uint32_t stamp[kStampChunkPages] = {};
    uint32_t newest = 0;
};
std::mutex g_stamp_mutex;
std::map<uintptr_t, std::unique_ptr<StampChunk>> g_stamp_chunks;  // address >> kStampChunkShift
uint64_t g_watch_epoch = 0;                                       // newest epoch handed out

void drop_stamps(uintptr_t lo, uintptr_t hi) {
    std::lock_guard<std::mutex> lock(g_stamp_mutex);
    g_stamp_chunks.erase(g_stamp_chunks.lower_bound(lo >> kStampChunkShift),
                         g_stamp_chunks.lower_bound(((hi - 1) >> kStampChunkShift) + 1));
}

bool written_pages_by_watch(uintptr_t start, uintptr_t end, uintptr_t origin, uint64_t* epoch, bool peek,
                            std::vector<uint64_t>* offsets);

// Ranges this thread collected in one go (precollect_begin): a stamped take inside one needs no
// GetWriteWatch call of its own (a write after the collect is reported by the next one).
thread_local std::vector<std::pair<uintptr_t, uintptr_t>> t_precollected;

bool precollected_covers(uintptr_t lo, uintptr_t hi) {
    for (const auto& r : t_precollected) {
        if (lo >= r.first && hi <= r.second) {
            return true;
        }
    }
    return false;
}

// Peek mode: GetWriteWatch without RESET over the range (no page protection change, no TLB
// flush) reports the bits nobody has collected yet; those count as written. When there are some and
// the window around the range was last collected more than kPeekCollectMs ago, the window is
// collected (one reset for all readers' pages, stamped like any take) and the range peeked again.
// Result: the pages stamped after *epoch plus the pages still uncollected; never fewer than written.
// The reader's new epoch is the one read before it looked at anything: a collect running
// meanwhile stamps later values, which the next call reports again (twice, never missed).
constexpr int64_t kPeekCollectMs = 8;
std::map<uintptr_t, int64_t> g_last_collect_ms;  // window start -> last collect, under g_stamp_mutex

int64_t now_ms() {
    return static_cast<int64_t>(GetTickCount64());
}

bool written_pages_peek(uintptr_t start, uintptr_t end, uintptr_t origin, uintptr_t res_lo, uintptr_t res_hi,
                        uint64_t* epoch, std::vector<uint64_t>* offsets) {
    thread_local std::vector<PVOID> bits;
    const uint64_t pages = (end - start + kWrittenPageSize - 1) / kWrittenPageSize;
    const size_t cap = static_cast<size_t>(std::min<uint64_t>(pages, 4096));
    if (bits.size() < cap) {
        bits.resize(cap);
    }
    uint64_t e0 = 0;
    {
        std::lock_guard<std::mutex> lock(g_stamp_mutex);
        e0 = g_watch_epoch;
    }
    auto peek_bits = [&](size_t* n) {
        ULONG_PTR count = cap;
        DWORD granularity = 0;
        if (GetWriteWatch(0, reinterpret_cast<PVOID>(start), static_cast<SIZE_T>(end - start), bits.data(), &count,
                          &granularity) != 0 ||
            granularity != kWrittenPageSize) {
            return false;
        }
        *n = static_cast<size_t>(count);
        return !(*n == cap && cap < pages);  // more bits than the buffer: unknown (caller: all written)
    };
    size_t n = 0;
    if (!peek_bits(&n)) {
        return false;
    }
    if (n > 0) {
        // Collect a 64 MiB aligned window around the range (clipped to the reservation), at most
        // every kPeekCollectMs per window: one reset for every reader's pages in it.
        constexpr uintptr_t kWindow = uintptr_t{64} << 20;
        const uintptr_t wlo = std::max<uintptr_t>(res_lo, start & ~(kWindow - 1));
        const uintptr_t whi = std::min<uintptr_t>(res_hi, (end + kWindow - 1) & ~(kWindow - 1));
        bool due = false;
        {
            std::lock_guard<std::mutex> lock(g_stamp_mutex);
            int64_t& last = g_last_collect_ms[wlo];
            if (now_ms() - last >= kPeekCollectMs) {
                last = now_ms();
                due = true;
            }
        }
        if (due) {
            std::vector<uint64_t> none;
            if (written_pages_by_watch(wlo, whi, wlo, nullptr, false, &none)) {
                if (!peek_bits(&n)) {
                    return false;
                }
            }
        }
    }
    const uint64_t seen = *epoch;
    for (size_t i = 0; i < n; ++i) {
        offsets->push_back(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(bits[i]) - origin));
    }
    std::lock_guard<std::mutex> lock(g_stamp_mutex);
    for (auto it = g_stamp_chunks.lower_bound(start >> kStampChunkShift);
         it != g_stamp_chunks.end() && (it->first << kStampChunkShift) < end; ++it) {
        const StampChunk& chunk = *it->second;
        if (chunk.newest <= seen) {
            continue;
        }
        const uintptr_t chunk_base = it->first << kStampChunkShift;
        const uintptr_t lo = std::max<uintptr_t>(chunk_base, start);
        const uintptr_t hi = std::min<uintptr_t>(chunk_base + (uintptr_t{1} << kStampChunkShift), end);
        for (uintptr_t a = lo; a < hi; a += kWrittenPageSize) {
            if (chunk.stamp[(a >> 12) & (kStampChunkPages - 1)] > seen) {
                offsets->push_back(static_cast<uint64_t>(a - origin));
            }
        }
    }
    *epoch = e0;
    return true;
}

// MEM_WRITE_WATCH pages of [start, end) (one reservation), as offsets from `origin`.
// O33: GetWriteWatch's output buffer is kept per thread (it was up to 512 KiB allocated and zeroed
// on every take), and the pages it reports are appended in one go.
// `epoch` null: the pages this call consumed. Otherwise: the pages stamped after *epoch, by this
// call or by any other reader's, and *epoch moves to the newest stamp. False: not watched, or the
// call failed (the caller then treats the range as written; stamps already made stay valid).
bool written_pages_by_watch(uintptr_t start, uintptr_t end, uintptr_t origin, uint64_t* epoch, bool peek,
                            std::vector<uint64_t>* offsets) {
    uintptr_t res_lo = 0;
    uintptr_t res_hi = 0;
    if (!watched_covers(start, end, &res_lo, &res_hi)) {
        return false;
    }
    if (peek && epoch != nullptr) {
        return written_pages_peek(start, end, origin, res_lo, res_hi, epoch, offsets);
    }
    thread_local std::vector<PVOID> written;
    const uint64_t pages = (end - start + kWrittenPageSize - 1) / kWrittenPageSize;
    const size_t cap = static_cast<size_t>(std::min<uint64_t>(pages, 65536));
    if (written.size() < cap) {
        written.resize(cap);
    }
    const size_t before = offsets->size();
    std::lock_guard<std::mutex> lock(g_stamp_mutex);
    const uint64_t seen = epoch != nullptr ? *epoch : 0;
    // Stamps are 32-bit; once the epochs run out every take reports "unknown" (conservative).
    if (g_watch_epoch >= 0xFFFFFFF0u) {
        return false;
    }
    const uint32_t mine = static_cast<uint32_t>(g_watch_epoch + 1);
    bool consumed_any = false;
    bool ok = true;
    const bool collected_already = epoch != nullptr && precollected_covers(start, end);
    for (; !collected_already;) {
        ULONG_PTR count = cap;
        DWORD granularity = 0;
        // Each call resets what it reports (WRITE_WATCH_FLAG_RESET, no separate ResetWriteWatch);
        // a full buffer may leave more pages, so call again.
        if (GetWriteWatch(WRITE_WATCH_FLAG_RESET, reinterpret_cast<PVOID>(start), static_cast<SIZE_T>(end - start),
                          written.data(), &count, &granularity) != 0 ||
            granularity != kWrittenPageSize) {
            ok = false;
            break;
        }
        if (count > 0) {
            consumed_any = true;
        }
        const size_t at = offsets->size();
        if (epoch == nullptr) {
            offsets->resize(at + static_cast<size_t>(count));
        }
        uint64_t* out = epoch == nullptr ? offsets->data() + at : nullptr;
        StampChunk* chunk = nullptr;
        uintptr_t chunk_key = ~uintptr_t{0};
        for (ULONG_PTR i = 0; i < count; ++i) {
            const uintptr_t a = reinterpret_cast<uintptr_t>(written[i]);
            if (out != nullptr) {
                out[i] = static_cast<uint64_t>(a - origin);
            }
            if ((a >> kStampChunkShift) != chunk_key) {
                chunk_key = a >> kStampChunkShift;
                auto& slot = g_stamp_chunks[chunk_key];
                if (!slot) {
                    slot = std::make_unique<StampChunk>();
                }
                chunk = slot.get();
            }
            chunk->stamp[(a >> 12) & (kStampChunkPages - 1)] = mine;
            chunk->newest = mine;
        }
        if (count < cap) {
            break;
        }
    }
    if (consumed_any) {
        g_watch_epoch = mine;  // published with the stamps, still under the lock
    }
    if (!ok) {
        offsets->resize(before);
        return false;
    }
    if (epoch == nullptr) {
        return true;
    }
    // Everything in the range stamped after what this reader has seen (a chunk with no newer
    // stamp is skipped whole).
    for (auto it = g_stamp_chunks.lower_bound(start >> kStampChunkShift);
         it != g_stamp_chunks.end() && (it->first << kStampChunkShift) < end; ++it) {
        const StampChunk& chunk = *it->second;
        if (chunk.newest <= seen) {
            continue;
        }
        const uintptr_t chunk_base = it->first << kStampChunkShift;
        const uintptr_t lo = std::max<uintptr_t>(chunk_base, start);
        const uintptr_t hi = std::min<uintptr_t>(chunk_base + (uintptr_t{1} << kStampChunkShift), end);
        for (uintptr_t a = lo; a < hi; a += kWrittenPageSize) {
            if (chunk.stamp[(a >> 12) & (kStampChunkPages - 1)] > seen) {
                offsets->push_back(static_cast<uint64_t>(a - origin));
            }
        }
    }
    *epoch = g_watch_epoch;
    return true;
}

}  // namespace

void precollect_begin(std::vector<std::pair<uintptr_t, uintptr_t>>* ranges) {
    t_precollected.clear();
    if (ranges == nullptr || ranges->empty()) {
        return;
    }
    constexpr uintptr_t kMergeGap = uintptr_t{8} << 20;
    std::sort(ranges->begin(), ranges->end());
    std::vector<std::pair<uintptr_t, uintptr_t>> hulls;
    for (const auto& r : *ranges) {
        const uintptr_t lo = r.first & ~(uintptr_t{kWrittenPageSize} - 1);
        const uintptr_t hi = (r.second + kWrittenPageSize - 1) & ~(uintptr_t{kWrittenPageSize} - 1);
        if (!hulls.empty() && lo <= hulls.back().second + kMergeGap) {
            hulls.back().second = std::max(hulls.back().second, hi);
        } else {
            hulls.emplace_back(lo, hi);
        }
    }
    std::vector<uint64_t> none;
    for (const auto& h : hulls) {
        if (h.second <= h.first || !watched_covers(h.first, h.second)) {
            continue;
        }
        none.clear();
        if (written_pages_by_watch(h.first, h.second, h.first, nullptr, false, &none)) {
            t_precollected.push_back(h);
        }
    }
}

void precollect_end() {
    t_precollected.clear();
}

namespace {

// views_only: report only through shared page views (never MEM_WRITE_WATCH, whose take is
// destructive for other readers of the same pages); false when any part of the range is not
// covered by tracked views. On false nothing is consumed (the caller's clock stays).
bool take_pages_impl(const uint8_t* p, uint64_t len, uint64_t* view_clock, uint64_t* watch_epoch, bool peek,
                     std::vector<uint64_t>* offsets, bool views_only) {
    const uintptr_t start = reinterpret_cast<uintptr_t>(p) & ~(uintptr_t{kWrittenPageSize} - 1);
    const uintptr_t end = reinterpret_cast<uintptr_t>(p) + static_cast<uintptr_t>(len);
    {
        // R40: shared page views in the range (see TrackedView): those stamped after the caller's
        // previous call, and the pages they show armed for the next interval. Pieces of the range
        // between views (unmapped zone slots, or arena memory) are taken from MEM_WRITE_WATCH
        // where it covers them, else reported written.
        // O33: the tables shared (faults and other takes run meanwhile), each page under its own
        // lock. The caller's new position is the clock read before any page is looked at: a write
        // that disarms a page after this take armed it stamps a later value, so the next take
        // reports it (a write stamped meanwhile may be reported twice, never missed).
        const uint64_t clock = g_track_clock.load();
        std::shared_lock<std::shared_mutex> lock(g_track_mutex);
        auto it = g_views.upper_bound(start);
        if (it != g_views.begin() && std::prev(it)->first + std::prev(it)->second.size > start) {
            --it;
        }
        if (it != g_views.end() && it->first < end) {
            // Installed before anything is armed, and called first, ahead of handlers installed
            // earlier (the harness' fault reporter).
            std::call_once(g_track_handler_once, [] { AddVectoredExceptionHandler(1, &track_fault); });
            uintptr_t at = start;
            for (; it != g_views.end() && it->first < end; ++it) {
                const uintptr_t lo = std::max<uintptr_t>(it->first, start);
                const uintptr_t hi = std::min<uintptr_t>(it->first + static_cast<uintptr_t>(it->second.size), end);
                if (at < lo) {
                    if (views_only) {
                        return false;
                    }
                    if (!written_pages_by_watch(at, lo, start, watch_epoch, peek, offsets)) {
                        all_written(at, lo, start, offsets);
                    }
                }
                TrackedPage& page = *it->second.tracked;
                std::lock_guard<std::mutex> page_lock(page.mutex);
                const bool untracked = page.untracked || page.pins != 0;
                if (view_clock == nullptr || it->second.stamp > *view_clock || untracked) {
                    all_written(lo, hi, start, offsets);
                }
                if (!untracked && !page.armed) {
                    arm_page_locked(page);
                }
                at = hi;
            }
            if (at < end) {
                if (views_only) {
                    return false;
                }
                if (!written_pages_by_watch(at, end, start, watch_epoch, peek, offsets)) {
                    all_written(at, end, start, offsets);
                }
            }
            if (view_clock != nullptr) {
                *view_clock = clock;
            }
            return true;
        }
    }
    if (views_only) {
        return false;
    }
    return written_pages_by_watch(start, end, start, watch_epoch, peek, offsets);
}

}  // namespace

bool take_written_pages(const uint8_t* p, uint64_t len, uint64_t* view_clock, uint64_t* watch_epoch, bool peek,
                        std::vector<uint64_t>* offsets) {
    return take_pages_impl(p, len, view_clock, watch_epoch, peek, offsets, false);
}

bool take_written_view_pages(const uint8_t* p, uint64_t len, uint64_t* view_clock, std::vector<uint64_t>* offsets) {
    return view_clock != nullptr && take_pages_impl(p, len, view_clock, nullptr, false, offsets, true);
}

void write_positions(uint64_t* view_clock, uint64_t* watch_epoch) {
    *view_clock = g_track_clock.load();
    std::lock_guard<std::mutex> lock(g_stamp_mutex);
    *watch_epoch = g_watch_epoch;
}

#elif defined(__EMSCRIPTEN__)

// Linear memory [0, kGuestAddressSpaceBase) is the runtime's; [kGuestAddressSpaceBase, max) is the
// guest pool handed out to arenas. Uncommitted pool pages are zero (fresh from memory.grow, or
// zeroed when an "ever committed" page is committed again), so commits are zero-filled without
// touching pages that were never used.

namespace {

constexpr uint64_t kPoolStart = kGuestAddressSpaceBase;
constexpr uint64_t kPoolLimit = uint64_t{16} << 30;  // browser maximum for memory64
constexpr uint64_t kPoolPages = (kPoolLimit - kPoolStart) / kPageSize;

std::atomic<uint64_t> g_dirty[kPoolPages / 64];  // bit set: page was committed at least once
std::mutex g_pool_mutex;
std::map<uint64_t, uint64_t> g_regions;  // start -> size, guarded by g_pool_mutex
std::atomic_flag g_sbrk_lock = ATOMIC_FLAG_INIT;
std::atomic<bool> g_sbrk_fenced{false};  // set once __wrap_sbrk has run

uint64_t pool_end() {
    const uint64_t max = static_cast<uint64_t>(emscripten_get_heap_max());
    return align_down(max < kPoolLimit ? max : kPoolLimit, kAllocationGranularity);
}

}  // namespace

}  // namespace playgta6::os::backing

// The runtime heap fence: linked with -Wl,--wrap=sbrk, every sbrk (dlmalloc, emmalloc, mmap
// emulation) lands here and may not move the break into the guest pool.
extern "C" void* __real_sbrk(intptr_t increment);
extern "C" void* __wrap_sbrk(intptr_t increment) {
    using playgta6::os::backing::g_sbrk_lock;
    playgta6::os::backing::g_sbrk_fenced.store(true, std::memory_order_relaxed);
    while (g_sbrk_lock.test_and_set(std::memory_order_acquire)) {
    }
    void* result = reinterpret_cast<void*>(intptr_t{-1});
    const uintptr_t brk = reinterpret_cast<uintptr_t>(__real_sbrk(0));
    if (increment <= 0 ||
        static_cast<uint64_t>(increment) + 16 <= playgta6::kGuestAddressSpaceBase - brk) {
        result = __real_sbrk(increment);
    } else {
        errno = ENOMEM;
    }
    g_sbrk_lock.clear(std::memory_order_release);
    return result;
}

namespace playgta6::os::backing {

uint8_t* reserve(uint64_t size, uint64_t preferred) {
    // Without the sbrk fence (-Wl,--wrap=sbrk) the runtime heap could grow into the pool: refuse
    // to hand it out. sbrk(0) reaches __wrap_sbrk exactly when the fence is linked in.
    sbrk(0);
    if (!g_sbrk_fenced.load(std::memory_order_relaxed)) {
        return nullptr;
    }
    const uint64_t end_of_pool = pool_end();
    if (size == 0 || end_of_pool <= kPoolStart || size > end_of_pool - kPoolStart) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    uint64_t start = 0;
    if (preferred != 0) {
        if (preferred < kPoolStart || preferred > end_of_pool - size) {
            return nullptr;
        }
        auto next = g_regions.lower_bound(preferred);
        if (next != g_regions.end() && next->first < preferred + size) {
            return nullptr;
        }
        if (next != g_regions.begin() && std::prev(next)->first + std::prev(next)->second > preferred) {
            return nullptr;
        }
        start = preferred;
    } else {
        // Highest gap that fits, so the low end of the pool stays free for the identity arena.
        uint64_t gap_end = end_of_pool;
        for (auto it = g_regions.rbegin(); it != g_regions.rend(); ++it) {
            const uint64_t used_end = it->first + it->second;
            if (gap_end - used_end >= size) {
                start = gap_end - size;
                break;
            }
            gap_end = it->first;
        }
        if (start == 0 && gap_end - kPoolStart >= size) {
            start = gap_end - size;
        }
        if (start == 0) {
            return nullptr;
        }
    }
    g_regions.emplace(start, size);
    return reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(start));
}

void release(uint8_t* base, uint64_t /*size*/) {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    g_regions.erase(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(base)));
}

bool commit(uint8_t* p, uint64_t len, PageProtect /*protect*/) {
    const uint64_t start = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p));
    if (start < kPoolStart || len > kPoolLimit - start) {
        return false;
    }
    const uint64_t end = start + len;
    // Grow linear memory to cover the run (it never shrinks). With threads another thread may
    // grow it concurrently; emscripten_resize_heap then reports failure, so re-check the size.
    while (static_cast<uint64_t>(emscripten_get_heap_size()) < end) {
        if (!emscripten_resize_heap(static_cast<size_t>(end)) &&
            static_cast<uint64_t>(emscripten_get_heap_size()) < end) {
            return false;
        }
    }
    // O36 zero-skip: never-committed pages stay zero from memory.grow and are NOT touched.
    // Previously-committed (dirty) pages are coalesced into contiguous runs and zeroed on recommit.
    const uint64_t first_page = (start - kPoolStart) / kPageSize;
    const uint64_t last_page = (end - kPoolStart) / kPageSize;
    uint64_t zero_run_start = last_page;
    auto flush_zero_run = [&](uint64_t run_end) {
        if (zero_run_start < run_end) {
            std::memset(reinterpret_cast<void*>(
                            static_cast<uintptr_t>(kPoolStart + zero_run_start * kPageSize)),
                        0, static_cast<size_t>((run_end - zero_run_start) * kPageSize));
            zero_run_start = last_page;
        }
    };
    for (uint64_t page = first_page; page < last_page; ++page) {
        const uint64_t bit = uint64_t{1} << (page % 64);
        if ((g_dirty[page / 64].fetch_or(bit, std::memory_order_relaxed) & bit) != 0) {
            if (zero_run_start == last_page) {
                zero_run_start = page;
            }
        } else {
            flush_zero_run(page);
        }
    }
    flush_zero_run(last_page);
    return true;
}

bool decommit(uint8_t* /*p*/, uint64_t /*len*/) {
    return true;  // linear memory cannot shrink; the pages are zeroed when committed again
}

bool reset_pages(uint8_t* p, uint64_t len) {
    const uint64_t start = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p));
    if (len == 0 || start < kPoolStart || len > kPoolLimit - start) {
        return false;
    }
    // Linear memory cannot discard physical pages; committed state and g_dirty stay intact.
    return start + len <= static_cast<uint64_t>(emscripten_get_heap_size());
}

bool reset_undo_pages(uint8_t* p, uint64_t len) {
    return reset_pages(p, len);
}

bool protect(uint8_t* /*p*/, uint64_t /*len*/, PageProtect /*protect*/) {
    return true;  // wasm has no page protection: tracked by the arena, not enforced
}

bool enforces_protection() {
    return false;
}

void precollect_begin(std::vector<std::pair<uintptr_t, uintptr_t>>*) {}
void precollect_end() {}

bool take_written_pages(const uint8_t*, uint64_t, uint64_t*, uint64_t*, bool, std::vector<uint64_t>*) {
    return false;  // no write tracking in linear memory
}

bool take_written_view_pages(const uint8_t*, uint64_t, uint64_t*, std::vector<uint64_t>*) {
    return false;
}

void write_positions(uint64_t*, uint64_t*) {}

uint64_t tracked_view_faults() {
    return 0;
}

void pin_views(uintptr_t, uintptr_t, bool) {}

void set_write_watch(bool) {}

// One linear memory: no second view of anything (the translated window aliases instead).
bool shared_pages_supported() {
    return false;
}
uint8_t* reserve_with_shared_zone(uint64_t size, uint64_t preferred, uint64_t) {
    return reserve(size, preferred);
}
void* create_shared_page(uint64_t) {
    return nullptr;
}
void release_shared_page(void*) {}
bool map_shared_page(uint8_t*, void*, uint64_t) {
    return false;
}
bool unmap_shared_page(uint8_t*, uint64_t) {
    return false;
}
uint8_t* view_shared_page(void*, uint64_t) {
    return nullptr;
}
void unview_shared_page(uint8_t*, uint64_t) {}

#else  // POSIX

namespace {

constexpr int kReserveFlags = MAP_PRIVATE | MAP_ANONYMOUS
#ifdef MAP_NORESERVE
                              | MAP_NORESERVE
#endif
    ;

int posix_protect(PageProtect protect) {
    if (is_writable(protect)) {
        return PROT_READ | PROT_WRITE;
    }
    return is_readable(protect) ? PROT_READ : PROT_NONE;
}

}  // namespace

uint8_t* reserve(uint64_t size, uint64_t preferred) {
    void* p = mmap(reinterpret_cast<void*>(static_cast<uintptr_t>(preferred)),
                   static_cast<size_t>(size), PROT_NONE, kReserveFlags, -1, 0);
    if (p == MAP_FAILED) {
        return nullptr;
    }
    if (preferred != 0 && reinterpret_cast<uintptr_t>(p) != preferred) {
        munmap(p, static_cast<size_t>(size));  // the hint was not honoured
        return nullptr;
    }
    return static_cast<uint8_t*>(p);
}

void release(uint8_t* base, uint64_t size) {
    munmap(base, static_cast<size_t>(size));
}

bool commit(uint8_t* p, uint64_t len, PageProtect protect) {
    // Reserved and decommitted pages are fresh anonymous mappings: zero on first touch.
    return mprotect(p, static_cast<size_t>(len), posix_protect(protect)) == 0;
}

bool decommit(uint8_t* p, uint64_t len) {
    // Replace the run with fresh inaccessible pages: frees the memory, contents read as zero
    // after the next commit (portable, unlike MADV_DONTNEED / MADV_FREE).
    return mmap(p, static_cast<size_t>(len), PROT_NONE, kReserveFlags | MAP_FIXED, -1, 0) == p;
}

bool reset_pages(uint8_t* p, uint64_t len) {
    if (p == nullptr || len == 0) {
        return false;
    }
    return madvise(p, static_cast<size_t>(len), MADV_DONTNEED) == 0;
}

bool reset_undo_pages(uint8_t* p, uint64_t len) {
    if (p == nullptr || len == 0) {
        return false;
    }
#ifdef MADV_WILLNEED
    return madvise(p, static_cast<size_t>(len), MADV_WILLNEED) == 0;
#else
    return true;
#endif
}

bool protect(uint8_t* p, uint64_t len, PageProtect protect) {
    return mprotect(p, static_cast<size_t>(len), posix_protect(protect)) == 0;
}

bool enforces_protection() {
    return true;
}

void precollect_begin(std::vector<std::pair<uintptr_t, uintptr_t>>*) {}
void precollect_end() {}

bool take_written_pages(const uint8_t*, uint64_t, uint64_t*, uint64_t*, bool, std::vector<uint64_t>*) {
    return false;  // not tracked on POSIX hosts
}

bool take_written_view_pages(const uint8_t*, uint64_t, uint64_t*, std::vector<uint64_t>*) {
    return false;
}

void write_positions(uint64_t*, uint64_t*) {}

uint64_t tracked_view_faults() {
    return 0;
}

void pin_views(uintptr_t, uintptr_t, bool) {}

void set_write_watch(bool) {}

// O36: shared pages are memfd / POSIX shared memory objects mapped MAP_SHARED | MAP_FIXED over
// reserved pages; any reserved range can hold them.
bool shared_pages_supported() {
    return true;
}

uint8_t* reserve_with_shared_zone(uint64_t size, uint64_t preferred, uint64_t) {
    return reserve(size, preferred);
}

void* create_shared_page(uint64_t size) {
    int fd = -1;
#if defined(__linux__)
    fd = static_cast<int>(syscall(SYS_memfd_create, "pg6-page", 0u));
#endif
    if (fd < 0) {
        char name[64];
        static std::atomic<unsigned> counter{0};
        std::snprintf(name, sizeof(name), "/pg6-page-%d-%u", static_cast<int>(getpid()), counter.fetch_add(1));
        fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
        if (fd >= 0) {
            shm_unlink(name);
        }
    }
    if (fd < 0) {
        return nullptr;
    }
    if (ftruncate(fd, static_cast<off_t>(size)) != 0) {
        close(fd);
        return nullptr;
    }
    return reinterpret_cast<void*>(static_cast<intptr_t>(fd) + 1);
}

void release_shared_page(void* page) {
    if (page != nullptr) {
        close(static_cast<int>(reinterpret_cast<intptr_t>(page) - 1));
    }
}

bool map_shared_page(uint8_t* at, void* page, uint64_t size) {
    if (page == nullptr) {
        return false;
    }
    const int fd = static_cast<int>(reinterpret_cast<intptr_t>(page) - 1);
    return mmap(at, static_cast<size_t>(size), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0) == at;
}

bool unmap_shared_page(uint8_t* at, uint64_t size) {
    return mmap(at, static_cast<size_t>(size), PROT_NONE, kReserveFlags | MAP_FIXED, -1, 0) == at;
}

uint8_t* view_shared_page(void* page, uint64_t size) {
    if (page == nullptr) {
        return nullptr;
    }
    const int fd = static_cast<int>(reinterpret_cast<intptr_t>(page) - 1);
    void* p = mmap(nullptr, static_cast<size_t>(size), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    return p == MAP_FAILED ? nullptr : static_cast<uint8_t*>(p);
}

void unview_shared_page(uint8_t* view, uint64_t size) {
    if (view != nullptr) {
        munmap(view, static_cast<size_t>(size));
    }
}

#endif

}  // namespace playgta6::os::backing
