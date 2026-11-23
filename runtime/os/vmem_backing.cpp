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
#include <sys/mman.h>
#endif

namespace playgta6::os::backing {

#if defined(_WIN32)

namespace {

DWORD win_protect(PageProtect protect) {
    if (is_writable(protect)) {
        return PAGE_READWRITE;
    }
    return is_readable(protect) ? PAGE_READONLY : PAGE_NOACCESS;
}

}  // namespace

uint8_t* reserve(uint64_t size, uint64_t preferred) {
    void* p = VirtualAlloc(reinterpret_cast<void*>(static_cast<uintptr_t>(preferred)),
                           static_cast<SIZE_T>(size), MEM_RESERVE, PAGE_NOACCESS);
    if (p != nullptr && preferred != 0 && reinterpret_cast<uintptr_t>(p) != preferred) {
        VirtualFree(p, 0, MEM_RELEASE);
        return nullptr;
    }
    return static_cast<uint8_t*>(p);
}

void release(uint8_t* base, uint64_t /*size*/) {
    VirtualFree(base, 0, MEM_RELEASE);
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

#endif

}  // namespace playgta6::os::backing
