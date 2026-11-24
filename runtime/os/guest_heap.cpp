#include "os/guest_heap.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <new>
#include <vector>

#include "os/win32_kernel.h"

namespace playgta6::os {

namespace {

constexpr uint64_t kSmallAlign = GuestHeap::kMinAlignment;

uint64_t round_up(uint64_t value, uint64_t alignment) {
    return (value + (alignment - 1u)) & ~(alignment - 1u);
}

bool is_pow2(uint64_t v) {
    return v != 0 && (v & (v - 1u)) == 0;
}

uint64_t small_capacity(uint64_t size) {
    return round_up(size == 0 ? 1 : size, kSmallAlign);
}

uint64_t arena_alloc(VirtualMemoryArena& arena, uint64_t addr, uint64_t size) {
    try {
        return arena.virtual_alloc(addr, size, kMemReserve | kMemCommit, PageProtect::ReadWrite);
    } catch (const std::bad_alloc&) {
        return 0;
    }
}

}  // namespace

GuestHeap::GuestHeap(VirtualMemoryArena& arena) : arena_(arena) {}

GuestHeap::~GuestHeap() {
    if (!abandoned_) {
        release_all();
    }
}

// ---------------------------------------------------------------------------
// Accounting
// ---------------------------------------------------------------------------

bool GuestHeap::charge_locked(uint64_t size, HeapFamily family) {
    if (soft_limit_ != 0 && (size > soft_limit_ || stats_.live_bytes > soft_limit_ - size)) {
        return false;
    }
    GuestHeapFamilyStats& f = stats_.family[static_cast<size_t>(family)];
    stats_.live_bytes += size;
    stats_.peak_bytes = std::max(stats_.peak_bytes, stats_.live_bytes);
    stats_.live_blocks += 1;
    stats_.total_allocs += 1;
    f.live_bytes += size;
    f.peak_bytes = std::max(f.peak_bytes, f.live_bytes);
    f.live_blocks += 1;
    f.total_allocs += 1;
    return true;
}

void GuestHeap::uncharge_locked(uint64_t size, HeapFamily family) {
    GuestHeapFamilyStats& f = stats_.family[static_cast<size_t>(family)];
    stats_.live_bytes -= size;
    stats_.live_blocks -= 1;
    f.live_bytes -= size;
    f.live_blocks -= 1;
}

void GuestHeap::resize_charge_locked(const Block& block, uint64_t new_size) {
    GuestHeapFamilyStats& f = stats_.family[static_cast<size_t>(block.family)];
    stats_.live_bytes = stats_.live_bytes - block.size + new_size;
    stats_.peak_bytes = std::max(stats_.peak_bytes, stats_.live_bytes);
    f.live_bytes = f.live_bytes - block.size + new_size;
    f.peak_bytes = std::max(f.peak_bytes, f.live_bytes);
    if (block.large) {
        stats_.large_live_bytes = stats_.large_live_bytes - block.size + new_size;
        stats_.large_peak_bytes = std::max(stats_.large_peak_bytes, stats_.large_live_bytes);
    }
}

void GuestHeap::fail_locked(HeapStatus* status, HeapStatus why) {
    if (why == HeapStatus::OutOfMemory || why == HeapStatus::InvalidParameter) {
        stats_.failed_allocs += 1;
    }
    if (status != nullptr) {
        *status = why;
    }
}

void GuestHeap::note_invalid_free(uint64_t va) {
    stats_.invalid_frees += 1;
    if (invalid_logged_ < 8) {
        ++invalid_logged_;
        std::fprintf(stderr,
                     "[guest_heap] ignoring free/realloc of unknown block 0x%llx "
                     "(foreign pointer or double free)\n",
                     static_cast<unsigned long long>(va));
    }
}

void GuestHeap::zero(uint64_t va, uint64_t len) const {
    if (len != 0) {
        std::memset(arena_.translate(va), 0, static_cast<size_t>(len));
    }
}

// ---------------------------------------------------------------------------
// Small / medium backend
// ---------------------------------------------------------------------------

void GuestHeap::insert_free_locked(uint64_t addr, uint64_t size) {
    free_by_addr_.emplace(addr, size);
    free_by_size_.emplace(size, addr);
}

void GuestHeap::erase_free_locked(std::map<uint64_t, uint64_t>::iterator it) {
    free_by_size_.erase({it->second, it->first});
    free_by_addr_.erase(it);
}

uint64_t GuestHeap::segment_base_locked(uint64_t addr) const {
    auto it = segments_.upper_bound(addr);
    if (it == segments_.begin()) {
        return 0;
    }
    --it;
    return addr < it->first + it->second ? it->first : 0;
}

void GuestHeap::add_free_locked(uint64_t addr, uint64_t size) {
    const uint64_t seg = segment_base_locked(addr);
    auto next = free_by_addr_.find(addr + size);
    if (next != free_by_addr_.end() && segment_base_locked(next->first) == seg) {
        size += next->second;
        erase_free_locked(next);
    }
    auto it = free_by_addr_.lower_bound(addr);
    if (it != free_by_addr_.begin()) {
        auto prev = std::prev(it);
        if (prev->first + prev->second == addr && segment_base_locked(prev->first) == seg) {
            addr = prev->first;
            size += prev->second;
            erase_free_locked(prev);
        }
    }
    insert_free_locked(addr, size);
}

uint64_t GuestHeap::take_free_locked(uint64_t capacity, uint64_t alignment) {
    // Free ranges start 16-byte aligned, so any range of capacity + alignment - 16 bytes fits an
    // aligned block wherever it lies; smaller ranges only fit by luck, so look at a few of those
    // first and then jump straight to the guaranteed fits.
    const uint64_t guaranteed = capacity + (alignment > kSmallAlign ? alignment - kSmallAlign : 0);
    auto it = free_by_size_.lower_bound({capacity, 0});
    for (int tries = 0; it != free_by_size_.end(); ++it, ++tries) {
        if (tries == 16 && it->first < guaranteed) {
            it = free_by_size_.lower_bound({guaranteed, 0});
            if (it == free_by_size_.end()) {
                break;
            }
        }
        const uint64_t size = it->first;
        const uint64_t addr = it->second;
        const uint64_t start = round_up(addr, alignment);
        if (start - addr + capacity > size) {
            continue;
        }
        free_by_size_.erase(it);
        free_by_addr_.erase(addr);
        if (start > addr) {
            insert_free_locked(addr, start - addr);
        }
        const uint64_t tail = addr + size - (start + capacity);
        if (tail != 0) {
            insert_free_locked(start + capacity, tail);
        }
        return start;
    }
    return 0;
}

bool GuestHeap::grow_locked(uint64_t capacity, uint64_t alignment) {
    const uint64_t need = capacity + (alignment > kSmallAlign ? alignment : 0);
    const uint64_t seg = round_up(std::max(need, kSegmentSize), kAllocationGranularity);
    const uint64_t base = arena_alloc(arena_, 0, seg);
    if (base == 0) {
        return false;
    }
    segments_.emplace(base, seg);
    stats_.committed_bytes += seg;
    insert_free_locked(base, seg);
    return true;
}

uint64_t GuestHeap::alloc_small_locked(uint64_t size, uint64_t alignment, HeapFamily family) {
    const uint64_t capacity = small_capacity(size);
    uint64_t addr = take_free_locked(capacity, alignment);
    if (addr == 0 && grow_locked(capacity, alignment)) {
        addr = take_free_locked(capacity, alignment);
    }
    if (addr == 0) {
        return 0;
    }
    Block block;
    block.size = size;
    block.extent = capacity;
    block.committed = capacity;
    block.alignment = alignment;
    block.family = family;
    block.large = false;
    live_[addr] = block;
    return addr;
}

// ---------------------------------------------------------------------------
// Large backend
// ---------------------------------------------------------------------------

uint64_t GuestHeap::reserve_large(uint64_t size, uint64_t alignment) {
    const uint64_t rounded = round_up(size == 0 ? 1 : size, kPageSize);
    if (alignment <= kAllocationGranularity) {
        return arena_alloc(arena_, 0, rounded);
    }
    // Alignment above the reservation granularity: find an aligned hole and take it exactly.
    for (int attempt = 0; attempt < 8; ++attempt) {
        const uint64_t probe = arena_.reserve(0, rounded + alignment, PageProtect::NoAccess);
        if (probe == 0) {
            return 0;
        }
        const uint64_t aligned = round_up(probe, alignment);
        arena_.release(probe);
        const uint64_t got = arena_alloc(arena_, aligned, rounded);
        if (got == aligned) {
            return got;
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

uint64_t GuestHeap::alloc(uint64_t size,
                          uint64_t alignment,
                          HeapFamily family,
                          uint32_t flags,
                          HeapStatus* status) {
    if (status != nullptr) {
        *status = HeapStatus::Ok;
    }
    if (alignment < kMinAlignment) {
        alignment = kMinAlignment;
    }
    if (!is_pow2(alignment)) {
        std::lock_guard<std::mutex> lock(mutex_);
        fail_locked(status, HeapStatus::InvalidParameter);
        return 0;
    }
    if (size > kMaxRequest || alignment > kMaxRequest) {
        std::lock_guard<std::mutex> lock(mutex_);
        fail_locked(status, HeapStatus::OutOfMemory);
        return 0;
    }
    const bool large = size > kLargeThreshold || alignment > kAllocationGranularity;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!charge_locked(size, family)) {
            fail_locked(status, HeapStatus::OutOfMemory);
            return 0;
        }
        if (!large) {
            const uint64_t va = alloc_small_locked(size, alignment, family);
            if (va == 0) {
                uncharge_locked(size, family);
                stats_.total_allocs -= 1;
                stats_.family[static_cast<size_t>(family)].total_allocs -= 1;
                fail_locked(status, HeapStatus::OutOfMemory);
                return 0;
            }
            if ((flags & kHeapZeroMemory) != 0) {
                zero(va, size);
            }
            return va;
        }
    }

    const uint64_t va = reserve_large(size, alignment);
    std::lock_guard<std::mutex> lock(mutex_);
    if (va == 0) {
        uncharge_locked(size, family);
        stats_.total_allocs -= 1;
        stats_.family[static_cast<size_t>(family)].total_allocs -= 1;
        fail_locked(status, HeapStatus::OutOfMemory);
        return 0;
    }
    Block block;
    block.size = size;
    block.extent = round_up(size == 0 ? 1 : size, kPageSize);
    block.committed = block.extent;
    block.alignment = alignment;
    block.family = family;
    block.large = true;
    live_[va] = block;
    stats_.large_live_bytes += size;
    stats_.large_peak_bytes = std::max(stats_.large_peak_bytes, stats_.large_live_bytes);
    stats_.large_live_blocks += 1;
    stats_.committed_bytes += block.committed;
    // Fresh arena commits are zero filled: calloc of a large block needs no memset.
    return va;
}

uint64_t GuestHeap::calloc(uint64_t count,
                           uint64_t size,
                           uint64_t alignment,
                           HeapFamily family,
                           HeapStatus* status) {
    if (count != 0 && size > kMaxRequest / count) {
        std::lock_guard<std::mutex> lock(mutex_);
        fail_locked(status, HeapStatus::OutOfMemory);
        return 0;
    }
    return alloc(count * size, alignment, family, kHeapZeroMemory, status);
}

bool GuestHeap::free(uint64_t va, HeapStatus* status) {
    if (status != nullptr) {
        *status = HeapStatus::Ok;
    }
    if (va == 0) {
        return true;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    auto it = live_.find(va);
    if (it == live_.end()) {
        note_invalid_free(va);
        if (status != nullptr) {
            *status = HeapStatus::InvalidPointer;
        }
        return false;
    }
    const Block block = it->second;
    live_.erase(it);
    uncharge_locked(block.size, block.family);
    stats_.total_frees += 1;
    if (block.large) {
        stats_.large_live_bytes -= block.size;
        stats_.large_live_blocks -= 1;
        stats_.committed_bytes -= block.committed;
        lock.unlock();
        arena_.release(va);
    } else {
        add_free_locked(va, block.extent);
    }
    return true;
}

// 0 = resized in place, 1 = must move, 2 = soft limit.
int GuestHeap::resize_in_place_locked(uint64_t va, Block& block, uint64_t size, uint32_t flags) {
    if (size > block.size && soft_limit_ != 0 &&
        stats_.live_bytes + (size - block.size) > soft_limit_) {
        return 2;
    }
    const bool new_large = size > kLargeThreshold;
    const uint64_t old_size = block.size;
    if (block.large) {
        if (!new_large) {
            return 1;
        }
        const uint64_t want = round_up(size, kPageSize);
        if (want <= block.committed) {
            if (block.committed - want >= kPageSize) {
                if (!arena_.decommit(va + want, block.committed - want)) {
                    return 1;
                }
                stats_.committed_bytes -= block.committed - want;
                block.committed = want;
            }
        } else if (want <= block.extent) {
            if (!arena_.commit(va + block.committed, want - block.committed,
                               PageProtect::ReadWrite)) {
                return 1;
            }
            stats_.committed_bytes += want - block.committed;
            block.committed = want;
        } else {
            return 1;
        }
    } else {
        if (new_large) {
            return 1;
        }
        const uint64_t cap = small_capacity(size);
        if (cap <= block.extent) {
            if (block.extent - cap >= kSmallAlign) {
                add_free_locked(va + cap, block.extent - cap);
                block.extent = cap;
                block.committed = cap;
            }
        } else {
            auto nf = free_by_addr_.find(va + block.extent);
            if (nf == free_by_addr_.end() ||
                segment_base_locked(nf->first) != segment_base_locked(va) ||
                block.extent + nf->second < cap) {
                return 1;
            }
            const uint64_t remainder = block.extent + nf->second - cap;
            erase_free_locked(nf);
            if (remainder != 0) {
                insert_free_locked(va + cap, remainder);
            }
            block.extent = cap;
            block.committed = cap;
        }
    }
    resize_charge_locked(block, size);
    block.size = size;
    if ((flags & kHeapZeroMemory) != 0 && size > old_size) {
        zero(va + old_size, size - old_size);
    }
    return 0;
}

uint64_t GuestHeap::realloc(uint64_t va,
                            uint64_t size,
                            uint64_t alignment,
                            HeapFamily family,
                            uint32_t flags,
                            HeapStatus* status) {
    if (status != nullptr) {
        *status = HeapStatus::Ok;
    }
    if (va == 0) {
        return alloc(size, alignment == 0 ? kMinAlignment : alignment, family, flags, status);
    }
    if (size == 0) {
        free(va, status);
        return 0;
    }
    if (alignment != 0 && !is_pow2(alignment)) {
        std::lock_guard<std::mutex> lock(mutex_);
        fail_locked(status, HeapStatus::InvalidParameter);
        return 0;
    }
    if (size > kMaxRequest) {
        std::lock_guard<std::mutex> lock(mutex_);
        fail_locked(status, HeapStatus::OutOfMemory);
        return 0;
    }

    Block old;
    uint64_t new_align = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = live_.find(va);
        if (it == live_.end()) {
            note_invalid_free(va);
            if (status != nullptr) {
                *status = HeapStatus::InvalidPointer;
            }
            return 0;
        }
        old = it->second;
        new_align = std::max(alignment == 0 ? old.alignment : alignment, kMinAlignment);
        if ((va & (new_align - 1u)) == 0) {
            const int r = resize_in_place_locked(va, it->second, size, flags);
            if (r == 0) {
                it->second.alignment = std::max(old.alignment, new_align);
                return va;
            }
            if (r == 2) {
                fail_locked(status, HeapStatus::OutOfMemory);
                return 0;
            }
        }
    }

    // Move: allocate, copy outside the lock, free the old block.
    const uint64_t moved = this->alloc(size, new_align, old.family, 0, status);
    if (moved == 0) {
        return 0;
    }
    const uint64_t keep = std::min(old.size, size);
    std::memcpy(arena_.translate(moved), arena_.translate(va), static_cast<size_t>(keep));
    if ((flags & kHeapZeroMemory) != 0 && size > old.size) {
        zero(moved + old.size, size - old.size);
    }
    free(va);
    return moved;
}

uint64_t GuestHeap::expand(uint64_t va, uint64_t size, HeapStatus* status) {
    if (status != nullptr) {
        *status = HeapStatus::Ok;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = live_.find(va);
    if (va == 0 || it == live_.end()) {
        if (va != 0) {
            note_invalid_free(va);
        }
        if (status != nullptr) {
            *status = HeapStatus::InvalidPointer;
        }
        return 0;
    }
    if (size == 0 || size > kMaxRequest) {
        fail_locked(status, size == 0 ? HeapStatus::InvalidParameter : HeapStatus::OutOfMemory);
        return 0;
    }
    if (resize_in_place_locked(va, it->second, size, 0) != 0) {
        fail_locked(status, HeapStatus::OutOfMemory);
        return 0;
    }
    return va;
}

uint64_t GuestHeap::usable_size(uint64_t va) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = live_.find(va);
    return it == live_.end() ? kInvalidSize : it->second.size;
}

bool GuestHeap::block_info(uint64_t va, HeapBlockInfo* out) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = live_.find(va);
    if (va == 0 || it == live_.end()) {
        return false;
    }
    if (out != nullptr) {
        out->size = it->second.size;
        out->alignment = it->second.alignment;
        out->family = it->second.family;
        out->large = it->second.large;
    }
    return true;
}

GuestHeapStats GuestHeap::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    GuestHeapStats copy = stats_;
    copy.soft_limit_bytes = soft_limit_;
    return copy;
}

void GuestHeap::set_soft_limit(uint64_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    soft_limit_ = bytes;
}

uint64_t GuestHeap::soft_limit() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return soft_limit_;
}

void GuestHeap::abandon() {
    std::lock_guard<std::mutex> lock(mutex_);
    abandoned_ = true;
    live_.clear();
    segments_.clear();
    free_by_addr_.clear();
    free_by_size_.clear();
    stats_ = GuestHeapStats{};
}

void GuestHeap::release_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& [va, block] : live_) {
        if (block.large) {
            arena_.release(va);
        }
    }
    for (const auto& [base, size] : segments_) {
        (void)size;
        arena_.release(base);
    }
    live_.clear();
    segments_.clear();
    free_by_addr_.clear();
    free_by_size_.clear();
    stats_ = GuestHeapStats{};
}

// ---------------------------------------------------------------------------
// Default (process) heap
// ---------------------------------------------------------------------------

namespace {

struct DefaultHeapState {
    std::mutex mutex;
    std::atomic<GuestHeap*> current{nullptr};
    std::vector<GuestHeap*> retired;  // kept reachable and alive on purpose
};

DefaultHeapState& default_state() {
    static DefaultHeapState* state = new DefaultHeapState();
    return *state;
}

}  // namespace

GuestHeap& default_guest_heap() {
    VirtualMemoryArena* arena = win32::get_vmem_arena();
    DefaultHeapState& st = default_state();
    GuestHeap* h = st.current.load(std::memory_order_acquire);
    if (h != nullptr && &h->arena() == arena) {
        return *h;
    }
    std::lock_guard<std::mutex> lock(st.mutex);
    h = st.current.load(std::memory_order_acquire);
    if (h != nullptr && &h->arena() == arena) {
        return *h;
    }
    uint64_t limit = 0;
    if (h != nullptr) {
        limit = h->soft_limit();
        h->abandon();
        st.retired.push_back(h);
    }
    GuestHeap* fresh = new GuestHeap(*arena);
    fresh->set_soft_limit(limit);
    st.current.store(fresh, std::memory_order_release);
    return *fresh;
}

// ---------------------------------------------------------------------------
// CRT family
// ---------------------------------------------------------------------------

namespace guest_crt {

namespace {

void set_err(int32_t* out, int32_t value) {
    if (out != nullptr) {
        *out = value;
    }
}

int32_t errno_for(HeapStatus status) {
    return (status == HeapStatus::InvalidParameter || status == HeapStatus::InvalidPointer)
               ? EINVAL
               : ENOMEM;
}

}  // namespace

void* malloc(size_t size, int32_t* errno_out) {
    GuestHeap& h = default_guest_heap();
    HeapStatus st = HeapStatus::Ok;
    const uint64_t va = h.alloc(size, GuestHeap::kMinAlignment, HeapFamily::Crt, 0, &st);
    if (va == 0) {
        set_err(errno_out, ENOMEM);
        return nullptr;
    }
    return h.to_host(va);
}

void* calloc(size_t count, size_t size, int32_t* errno_out) {
    GuestHeap& h = default_guest_heap();
    HeapStatus st = HeapStatus::Ok;
    const uint64_t va = h.calloc(count, size, GuestHeap::kMinAlignment, HeapFamily::Crt, &st);
    if (va == 0) {
        set_err(errno_out, ENOMEM);
        return nullptr;
    }
    return h.to_host(va);
}

void* realloc(void* ptr, size_t size, int32_t* errno_out) {
    GuestHeap& h = default_guest_heap();
    HeapStatus st = HeapStatus::Ok;
    const uint64_t va = h.realloc(h.to_guest(ptr), size, 0, HeapFamily::Crt, 0, &st);
    if (va == 0 && (size != 0 || st != HeapStatus::Ok)) {
        set_err(errno_out, errno_for(st));
    }
    return h.to_host(va);
}

void* recalloc(void* ptr, size_t count, size_t size, int32_t* errno_out) {
    GuestHeap& h = default_guest_heap();
    if (count != 0 && size > GuestHeap::kMaxRequest / count) {
        set_err(errno_out, ENOMEM);
        return nullptr;
    }
    HeapStatus st = HeapStatus::Ok;
    const uint64_t total = static_cast<uint64_t>(count) * size;
    const uint64_t va =
        h.realloc(h.to_guest(ptr), total, 0, HeapFamily::Crt, kHeapZeroMemory, &st);
    if (va == 0 && (total != 0 || st != HeapStatus::Ok)) {
        set_err(errno_out, errno_for(st));
    }
    return h.to_host(va);
}

void* expand(void* ptr, size_t size, int32_t* errno_out) {
    GuestHeap& h = default_guest_heap();
    HeapStatus st = HeapStatus::Ok;
    const uint64_t va = h.expand(h.to_guest(ptr), size, &st);
    if (va == 0) {
        set_err(errno_out, errno_for(st));
        return nullptr;
    }
    return h.to_host(va);
}

void free(void* ptr) {
    GuestHeap& h = default_guest_heap();
    h.free(h.to_guest(ptr));
}

void* aligned_malloc(size_t size, size_t alignment, int32_t* errno_out) {
    if (!is_pow2(alignment)) {
        set_err(errno_out, EINVAL);
        return nullptr;
    }
    GuestHeap& h = default_guest_heap();
    HeapStatus st = HeapStatus::Ok;
    const uint64_t va = h.alloc(size, alignment, HeapFamily::Crt, 0, &st);
    if (va == 0) {
        set_err(errno_out, errno_for(st));
        return nullptr;
    }
    return h.to_host(va);
}

void* aligned_realloc(void* ptr, size_t size, size_t alignment, int32_t* errno_out) {
    if (!is_pow2(alignment)) {
        set_err(errno_out, EINVAL);
        return nullptr;
    }
    if (ptr == nullptr) {
        return aligned_malloc(size, alignment, errno_out);
    }
    GuestHeap& h = default_guest_heap();
    if (size == 0) {
        h.free(h.to_guest(ptr));
        return nullptr;
    }
    HeapStatus st = HeapStatus::Ok;
    const uint64_t va = h.realloc(h.to_guest(ptr), size, alignment, HeapFamily::Crt, 0, &st);
    if (va == 0) {
        set_err(errno_out, errno_for(st));
        return nullptr;
    }
    return h.to_host(va);
}

void aligned_free(void* ptr) {
    free(ptr);
}

size_t msize(void* ptr, int32_t* errno_out) {
    GuestHeap& h = default_guest_heap();
    const uint64_t size = h.usable_size(h.to_guest(ptr));
    if (size == GuestHeap::kInvalidSize) {
        set_err(errno_out, EINVAL);
        return static_cast<size_t>(-1);
    }
    return static_cast<size_t>(size);
}

char* strdup(const char* str, int32_t* errno_out) {
    if (str == nullptr) {
        set_err(errno_out, EINVAL);
        return nullptr;
    }
    const size_t len = std::strlen(str) + 1;
    char* copy = static_cast<char*>(guest_crt::malloc(len, errno_out));
    if (copy != nullptr) {
        std::memcpy(copy, str, len);
    }
    return copy;
}

uint16_t* wcsdup(const uint16_t* str, int32_t* errno_out) {
    if (str == nullptr) {
        set_err(errno_out, EINVAL);
        return nullptr;
    }
    size_t n = 0;
    while (str[n] != 0) {
        ++n;
    }
    auto* copy = static_cast<uint16_t*>(guest_crt::malloc((n + 1) * sizeof(uint16_t), errno_out));
    if (copy != nullptr) {
        std::memcpy(copy, str, (n + 1) * sizeof(uint16_t));
    }
    return copy;
}

}  // namespace guest_crt

}  // namespace playgta6::os

extern "C" {

int32_t pg6_guest_heap_get_stats(Pg6GuestHeapStats* out) {
    if (out == nullptr) {
        return 0;
    }
    const playgta6::os::GuestHeapStats s = playgta6::os::default_guest_heap().stats();
    auto fam = [](const playgta6::os::GuestHeapFamilyStats& f) {
        return Pg6GuestHeapFamilyStats{f.live_bytes, f.peak_bytes, f.live_blocks, f.total_allocs};
    };
    out->live_bytes = s.live_bytes;
    out->peak_bytes = s.peak_bytes;
    out->live_blocks = s.live_blocks;
    out->total_allocs = s.total_allocs;
    out->total_frees = s.total_frees;
    out->failed_allocs = s.failed_allocs;
    out->invalid_frees = s.invalid_frees;
    out->soft_limit_bytes = s.soft_limit_bytes;
    out->large_live_bytes = s.large_live_bytes;
    out->large_peak_bytes = s.large_peak_bytes;
    out->large_live_blocks = s.large_live_blocks;
    out->committed_bytes = s.committed_bytes;
    out->crt = fam(s.family[0]);
    out->xmem = fam(s.family[1]);
    out->other = fam(s.family[2]);
    return 1;
}

void pg6_guest_heap_set_soft_limit(uint64_t bytes) {
    playgta6::os::default_guest_heap().set_soft_limit(bytes);
}

}  // extern "C"
