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
#include <mutex>
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

std::mutex g_zone_mutex;
std::map<uintptr_t, std::pair<uintptr_t, uintptr_t>> g_zones;  // reservation base -> [zone start, end)

}  // namespace

void release(uint8_t* base, uint64_t /*size*/) {
    std::pair<uintptr_t, uintptr_t> zone{0, 0};
    {
        std::lock_guard<std::mutex> lock(g_zone_mutex);
        if (auto it = g_zones.find(reinterpret_cast<uintptr_t>(base)); it != g_zones.end()) {
            zone = it->second;
            g_zones.erase(it);
        }
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
        if (lower == nullptr && ww != 0) {
            lower = api.virtual_alloc2(GetCurrentProcess(), base, static_cast<SIZE_T>(zone_offset),
                                       MEM_RESERVE | MEM_REPLACE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0);
        }
        if (lower == nullptr) {
            VirtualFree(base, 0, MEM_RELEASE);
            VirtualFree(base + zone_offset, 0, MEM_RELEASE);
            return nullptr;
        }
    }
    std::lock_guard<std::mutex> lock(g_zone_mutex);
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
    // [at, at + size) must be a placeholder of its own: split it out of a larger one.
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery(at, &mbi, sizeof(mbi)) == 0 || mbi.State != MEM_RESERVE) {
        return false;
    }
    if (mbi.AllocationBase != at || mbi.RegionSize != size) {  // (BaseAddress is always `at`)
        if (!VirtualFree(at, static_cast<SIZE_T>(size), MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
            return false;
        }
    }
    return api.map_view3(page, GetCurrentProcess(), at, 0, static_cast<SIZE_T>(size), MEM_REPLACE_PLACEHOLDER,
                         PAGE_READWRITE, nullptr, 0) == at;
}

bool unmap_shared_page(uint8_t* at, uint64_t /*size*/) {
    const PlaceholderApi& api = placeholder_api();
    return api.ok() && api.unmap_view2(GetCurrentProcess(), at, MEM_PRESERVE_PLACEHOLDER) != 0;
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

bool protect(uint8_t* p, uint64_t len, PageProtect protect) {
    DWORD old = 0;
    return VirtualProtect(p, static_cast<SIZE_T>(len), win_protect(protect), &old) != 0;
}

bool enforces_protection() {
    return true;
}

void set_write_watch(bool on) {
    g_write_watch.store(on, std::memory_order_relaxed);
}

bool take_written_pages(const uint8_t* p, uint64_t len, std::vector<uint64_t>* offsets) {
    const uintptr_t start = reinterpret_cast<uintptr_t>(p) & ~(uintptr_t{kWrittenPageSize} - 1);
    const uintptr_t end = reinterpret_cast<uintptr_t>(p) + static_cast<uintptr_t>(len);
    const uint64_t pages = (end - start + kWrittenPageSize - 1) / kWrittenPageSize;
    std::vector<PVOID> written(static_cast<size_t>(std::min<uint64_t>(pages, 65536)));
    for (;;) {
        ULONG_PTR count = written.size();
        DWORD granularity = 0;
        // Each call resets what it reports; a full buffer may leave more pages, so call again.
        if (GetWriteWatch(WRITE_WATCH_FLAG_RESET, reinterpret_cast<PVOID>(start), static_cast<SIZE_T>(end - start),
                          written.data(), &count, &granularity) != 0 ||
            granularity != kWrittenPageSize) {
            return false;
        }
        for (ULONG_PTR i = 0; i < count; ++i) {
            offsets->push_back(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(written[i]) - start));
        }
        if (count < written.size()) {
            return true;
        }
    }
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
    const uint64_t end = start + len;
    if (start < kPoolStart || end > kPoolLimit) {
        return false;
    }
    // Grow linear memory to cover the run (it never shrinks). With threads another thread may
    // grow it concurrently; emscripten_resize_heap then reports failure, so re-check the size.
    while (static_cast<uint64_t>(emscripten_get_heap_size()) < end) {
        if (!emscripten_resize_heap(static_cast<size_t>(end)) &&
            static_cast<uint64_t>(emscripten_get_heap_size()) < end) {
            return false;
        }
    }
    for (uint64_t page = (start - kPoolStart) / kPageSize; page < (end - kPoolStart) / kPageSize;
         ++page) {
        const uint64_t bit = uint64_t{1} << (page % 64);
        if ((g_dirty[page / 64].fetch_or(bit, std::memory_order_relaxed) & bit) != 0) {
            std::memset(reinterpret_cast<void*>(static_cast<uintptr_t>(kPoolStart + page * kPageSize)),
                        0, kPageSize);
        }
    }
    return true;
}

bool decommit(uint8_t* /*p*/, uint64_t /*len*/) {
    return true;  // linear memory cannot shrink; the pages are zeroed when committed again
}

bool protect(uint8_t* /*p*/, uint64_t /*len*/, PageProtect /*protect*/) {
    return true;  // wasm has no page protection: tracked by the arena, not enforced
}

bool enforces_protection() {
    return false;
}

bool take_written_pages(const uint8_t*, uint64_t, std::vector<uint64_t>*) {
    return false;  // no write tracking in linear memory
}

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

bool protect(uint8_t* p, uint64_t len, PageProtect protect) {
    return mprotect(p, static_cast<size_t>(len), posix_protect(protect)) == 0;
}

bool enforces_protection() {
    return true;
}

bool take_written_pages(const uint8_t*, uint64_t, std::vector<uint64_t>*) {
    return false;  // not tracked on POSIX hosts
}

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
