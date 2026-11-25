#include "os/vmem.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>

#include "os/vmem_backing.h"

// The translation base lifted code and the import bridge add to every guest address
// (docs/wasm_abi.md "Guest Address Space"). Identity until an arena is published.
extern "C" {
uint64_t __playgta6_guest_base = 0;
}

namespace playgta6::os {

namespace {

#ifdef __EMSCRIPTEN__
// The wasm pool cannot spare address space for an inaccessible [0, space_base) prefix.
constexpr bool kGuardLowAddresses = false;
#else
constexpr bool kGuardLowAddresses = true;
#endif

std::atomic<ArenaObserver> g_arena_observer{nullptr};

void notify(const VirtualMemoryArena& arena,
            ArenaEvent event,
            uint64_t va,
            uint64_t size,
            PageProtect protect,
            bool ok) {
    if (ArenaObserver observer = g_arena_observer.load(std::memory_order_acquire)) {
        observer(arena, event, va, size, protect, ok);
    }
}

}  // namespace

namespace {

// Committed runs, newest last (see take_written_pages): a bounded log, so a caller that falls
// further behind than kCommitLogSize commits gets `false`.
struct CommitRecord {
    uintptr_t start = 0;
    uint64_t len = 0;
};
constexpr uint64_t kCommitLogSize = 16384;
std::mutex g_commit_log_mutex;
std::vector<CommitRecord> g_commit_log(kCommitLogSize);
uint64_t g_commit_seq = 0;  // number of runs ever logged

void log_commit(const uint8_t* p, uint64_t len) {
    std::lock_guard<std::mutex> lock(g_commit_log_mutex);
    g_commit_log[static_cast<size_t>(g_commit_seq % kCommitLogSize)] = {reinterpret_cast<uintptr_t>(p), len};
    ++g_commit_seq;
}

}  // namespace

void set_write_tracking(bool on) {
    backing::set_write_watch(on);
}

bool take_written_pages(const void* host, uint64_t len, uint64_t* commit_seq, std::vector<uint64_t>* offsets) {
    if (host == nullptr || len == 0 || commit_seq == nullptr || offsets == nullptr) {
        return false;
    }
    const uintptr_t start = reinterpret_cast<uintptr_t>(host) & ~(uintptr_t{kWrittenPageSize} - 1);
    const uintptr_t end = reinterpret_cast<uintptr_t>(host) + static_cast<uintptr_t>(len);
    {
        std::lock_guard<std::mutex> lock(g_commit_log_mutex);
        if (*commit_seq > g_commit_seq || g_commit_seq - *commit_seq > kCommitLogSize) {
            *commit_seq = g_commit_seq;
            return false;
        }
        for (uint64_t s = *commit_seq; s < g_commit_seq; ++s) {
            const CommitRecord& r = g_commit_log[static_cast<size_t>(s % kCommitLogSize)];
            const uintptr_t lo = std::max<uintptr_t>(r.start, start);
            const uintptr_t hi = std::min<uintptr_t>(r.start + static_cast<uintptr_t>(r.len), end);
            for (uintptr_t page = (lo - start) / kWrittenPageSize * kWrittenPageSize + start; lo < hi && page < hi;
                 page += kWrittenPageSize) {
                offsets->push_back(static_cast<uint64_t>(page - start));
            }
        }
        *commit_seq = g_commit_seq;
    }
    return backing::take_written_pages(static_cast<const uint8_t*>(host), len, offsets);
}

ArenaObserver set_arena_observer(ArenaObserver observer) {
    return g_arena_observer.exchange(observer, std::memory_order_acq_rel);
}

VirtualMemoryArena::VirtualMemoryArena(uint64_t default_base,
                                       uint64_t space_size,
                                       ArenaPlacement placement,
                                       uint64_t shared_zone) {
    uint64_t aligned = kDefaultArenaBase;
    if (default_base >= kAllocationGranularity &&
        align_up_checked(default_base, kAllocationGranularity, aligned) &&
        aligned >= kAllocationGranularity) {
        default_base_ = aligned;
    } else {
        default_base_ = kDefaultArenaBase;
    }
    space_base_ = default_base_;

    uint64_t size = 0;
    if (space_size == 0 || !align_up_checked(space_size, kAllocationGranularity, size) ||
        size > UINT64_MAX - space_base_) {
        return;
    }
    space_size_ = size;

    // O36: the shared zone is the top of the space; the backing keeps it able to hold views.
    uint64_t zone = 0;
    if (shared_zone != 0 && backing::shared_pages_supported() &&
        align_up_checked(shared_zone, kAllocationGranularity, zone) && zone < size) {
        shared_zone_size_ = zone;
        shared_zone_base_ = space_base_ + size - zone;
    }
    auto reserve_backing = [&](uint64_t bytes, uint64_t preferred, uint64_t zone_start_offset) -> uint8_t* {
        return shared_zone_size_ != 0 ? backing::reserve_with_shared_zone(bytes, preferred, zone_start_offset)
                                      : backing::reserve(bytes, preferred);
    };

    if (placement == ArenaPlacement::PreferIdentity) {
        if (uint8_t* host = reserve_backing(size, space_base_, size - shared_zone_size_)) {
            backing_ = host;
            backing_size_ = size;
            guest_base_ = 0;
            return;
        }
    }
    if (kGuardLowAddresses) {
        // Back guest VAs [0, space_end): the part below space_base_ is never committed, so null
        // and small guest pointers fault.
        // One extra granule so that the base can be 64 KiB aligned (mmap only aligns to 4 KiB):
        // guest 64 KiB alignment then carries over to host pointers.
        const uint64_t total = space_base_ + size + kAllocationGranularity;
        // (Hosts with shared zones return 64 KiB aligned reservations: guest_base_ == host.)
        if (uint8_t* host = reserve_backing(total, 0, space_base_ + size - shared_zone_size_)) {
            backing_ = host;
            backing_size_ = total;
            uint64_t aligned = 0;
            align_up_checked(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(host)),
                             kAllocationGranularity, aligned);
            guest_base_ = aligned;
        }
    } else if (uint8_t* host = reserve_backing(size, 0, size - shared_zone_size_)) {
        backing_ = host;
        backing_size_ = size;
        guest_base_ = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(host)) - space_base_;
    }
    if (backing_ == nullptr) {
        shared_zone_size_ = 0;
        shared_zone_base_ = 0;
    }
}

VirtualMemoryArena::~VirtualMemoryArena() {
    if (backing_ != nullptr) {
        backing::release(backing_, backing_size_);
    }
}

bool VirtualMemoryArena::protections_enforced() {
    return backing::enforces_protection();
}

bool VirtualMemoryArena::is_valid_protect(PageProtect protect) {
    switch (protect) {
        case PageProtect::NoAccess:
        case PageProtect::ReadOnly:
        case PageProtect::ReadWrite:
        case PageProtect::ExecuteRead:
        case PageProtect::ExecuteReadWrite:
            return true;
        default:
            return false;
    }
}

VirtualMemoryArena::Reservation* VirtualMemoryArena::find_reservation_locked(uint64_t addr) {
    auto it = reservations_.upper_bound(addr);
    if (it == reservations_.begin()) {
        return nullptr;
    }
    --it;
    Reservation& res = it->second;
    if (addr >= res.base && addr < res.base + res.size) {
        return &res;
    }
    return nullptr;
}

const VirtualMemoryArena::Reservation* VirtualMemoryArena::find_reservation_locked(uint64_t addr) const {
    auto it = reservations_.upper_bound(addr);
    if (it == reservations_.begin()) {
        return nullptr;
    }
    --it;
    const Reservation& res = it->second;
    if (addr >= res.base && addr < res.base + res.size) {
        return &res;
    }
    return nullptr;
}

// Every page of [addr, addr + len) is committed with a readable (or writable) protection.
// The range may cross reservations.
bool VirtualMemoryArena::pages_allow_locked(uint64_t addr, size_t len, bool need_write) const {
    if (len == 0 || addr > UINT64_MAX - len) {
        return false;
    }
    const uint64_t end = addr + len;
    uint64_t cursor = addr;
    while (cursor < end) {
        const Reservation* res = find_reservation_locked(cursor);
        if (res == nullptr) {
            return false;
        }
        const uint64_t res_end = res->base + res->size;
        const uint64_t stop = std::min(end, res_end);
        const size_t first_page = static_cast<size_t>((cursor - res->base) / kPageSize);
        const size_t last_page = static_cast<size_t>((stop - 1u - res->base) / kPageSize);
        for (size_t p = first_page; p <= last_page; ++p) {
            const PageMeta& page = res->pages[p];
            if (page.state != MemState::Commit ||
                !(need_write ? is_writable(page.protect) : is_readable(page.protect))) {
                return false;
            }
        }
        cursor = stop;
    }
    return true;
}

bool VirtualMemoryArena::range_overlaps_locked(uint64_t start, uint64_t end) const {
    auto it = reservations_.lower_bound(start);
    if (it != reservations_.end()) {
        if (it->second.base < end) {
            return true;
        }
    }
    if (it != reservations_.begin()) {
        --it;
        const uint64_t prev_end = it->second.base + it->second.granularity_size;
        if (prev_end > start) {
            return true;
        }
    }
    return false;
}

uint64_t VirtualMemoryArena::find_free_va_locked(uint64_t granularity_size) const {
    const uint64_t space_end = shared_zone_size_ != 0 ? shared_zone_base_ : space_base_ + space_size_;
    uint64_t candidate = default_base_;
    for (const auto& [base, res] : reservations_) {
        const uint64_t res_end = res.base + res.granularity_size;
        if (res_end <= candidate) {
            continue;
        }
        if (candidate <= UINT64_MAX - granularity_size &&
            candidate + granularity_size <= res.base) {
            break;
        }
        if (res_end > candidate) {
            candidate = res_end;
        }
    }
    if (candidate > space_end || granularity_size > space_end - candidate) {
        return 0;  // address space exhausted
    }
    return candidate;
}

uint64_t VirtualMemoryArena::reserve_locked(uint64_t hint_addr,
                                            uint64_t size,
                                            PageProtect allocation_protect) {
    if (size == 0 || !is_valid_protect(allocation_protect) || !valid()) {
        return 0;
    }

    uint64_t base = 0;
    uint64_t res_size = 0;
    uint64_t gran_size = 0;

    if (hint_addr != 0) {
        base = align_down(hint_addr, kAllocationGranularity);
        if (base < kAllocationGranularity) {
            return 0;
        }
        if (hint_addr > UINT64_MAX - size) {
            return 0;
        }
        uint64_t aligned_end = 0;
        if (!align_up_checked(hint_addr + size, kPageSize, aligned_end)) {
            return 0;
        }
        res_size = aligned_end - base;
        if (!align_up_checked(res_size, kAllocationGranularity, gran_size)) {
            return 0;
        }
        if (base > UINT64_MAX - gran_size) {
            return 0;
        }
        // Only addresses inside the backed space can be handed out (and, O36, not in the shared
        // zone: its pages can only be views).
        if (base < space_base_ || gran_size > space_size_ ||
            base - space_base_ > space_size_ - gran_size) {
            return 0;
        }
        if (shared_zone_size_ != 0 && base + gran_size > shared_zone_base_) {
            return 0;
        }
        if (range_overlaps_locked(base, base + gran_size)) {
            return 0;
        }
    } else {
        if (!align_up_checked(size, kPageSize, res_size)) {
            return 0;
        }
        if (!align_up_checked(res_size, kAllocationGranularity, gran_size)) {
            return 0;
        }
        base = find_free_va_locked(gran_size);
        if (base == 0) {
            return 0;
        }
    }

    const size_t page_count = static_cast<size_t>(res_size / kPageSize);
    Reservation res;
    res.base = base;
    res.size = res_size;
    res.granularity_size = gran_size;
    res.allocation_protect = allocation_protect;
    res.pages.assign(page_count, PageMeta{MemState::Reserve, PageProtect::NoAccess});

    reservations_.emplace(base, std::move(res));
    reserved_bytes_ += res_size;
    return base;
}

bool VirtualMemoryArena::commit_locked(uint64_t addr, uint64_t size, PageProtect protect) {
    if (size == 0 || !is_valid_protect(protect)) {
        return false;
    }
    if (addr > UINT64_MAX - size) {
        return false;
    }

    Reservation* res = find_reservation_locked(addr);
    if (!res || res->placeholder || res->shared) {
        return false;  // placeholders and shared-zone reservations are never committed
    }

    const uint64_t start = align_down(addr, kPageSize);
    uint64_t end = 0;
    if (!align_up_checked(addr + size, kPageSize, end)) {
        return false;
    }
    if (start < res->base || end > res->base + res->size) {
        return false;
    }

    const size_t first_page = static_cast<size_t>((start - res->base) / kPageSize);
    const size_t last_page = static_cast<size_t>((end - res->base) / kPageSize);
    auto run_ptr = [&](size_t page) { return translate(res->base + page * kPageSize); };

    // Back maximal runs of uncommitted pages first (zero-filled); on failure undo those runs so
    // the bookkeeping never disagrees with the backing.
    uint64_t newly_committed_bytes = 0;
    for (size_t i = first_page; i < last_page;) {
        if (res->pages[i].state == MemState::Commit) {
            ++i;
            continue;
        }
        const size_t run_start = i;
        while (i < last_page && res->pages[i].state != MemState::Commit) {
            ++i;
        }
        const uint64_t run_bytes = static_cast<uint64_t>(i - run_start) * kPageSize;
        if (!backing::commit(run_ptr(run_start), run_bytes, protect)) {
            for (size_t p = first_page; p < run_start; ++p) {
                if (res->pages[p].state != MemState::Commit) {
                    backing::decommit(run_ptr(p), kPageSize);
                }
            }
            return false;
        }
        newly_committed_bytes += run_bytes;
        log_commit(run_ptr(run_start), run_bytes);
    }

    // Already committed pages keep their data and take the new protection.
    for (size_t i = first_page; i < last_page;) {
        if (res->pages[i].state != MemState::Commit || res->pages[i].protect == protect) {
            ++i;
            continue;
        }
        const size_t run_start = i;
        while (i < last_page && res->pages[i].state == MemState::Commit &&
               res->pages[i].protect != protect) {
            ++i;
        }
        backing::protect(run_ptr(run_start), static_cast<uint64_t>(i - run_start) * kPageSize,
                         protect);
    }

    for (size_t p = first_page; p < last_page; ++p) {
        res->pages[p].state = MemState::Commit;
        res->pages[p].protect = protect;
    }
    committed_bytes_ += newly_committed_bytes;
    return true;
}

bool VirtualMemoryArena::decommit_locked(uint64_t addr, uint64_t size) {
    Reservation* res = find_reservation_locked(addr);
    if (!res || res->shared) {
        return false;  // shared-zone pages are views: unmap_shared
    }

    uint64_t start = 0;
    uint64_t end = 0;
    if (size == 0) {
        if (addr != res->base) {
            return false;
        }
        start = res->base;
        end = res->base + res->size;
    } else {
        if (addr > UINT64_MAX - size) {
            return false;
        }
        start = align_down(addr, kPageSize);
        if (!align_up_checked(addr + size, kPageSize, end)) {
            return false;
        }
        if (start < res->base || end > res->base + res->size) {
            return false;
        }
    }

    const size_t first_page = static_cast<size_t>((start - res->base) / kPageSize);
    const size_t last_page = static_cast<size_t>((end - res->base) / kPageSize);

    uint64_t freed_bytes = 0;
    for (size_t p = first_page; p < last_page; ++p) {
        if (res->pages[p].state == MemState::Commit) {
            freed_bytes += kPageSize;
        }
        res->pages[p].state = MemState::Reserve;
        res->pages[p].protect = PageProtect::NoAccess;
    }
    if (freed_bytes != 0) {
        backing::decommit(translate(start), end - start);
    }
    committed_bytes_ -= freed_bytes;
    return true;
}

bool VirtualMemoryArena::release_locked(uint64_t base_addr) {
    auto it = reservations_.find(base_addr);
    if (it == reservations_.end()) {
        return false;
    }
    const Reservation& res = it->second;
    uint64_t res_committed = 0;
    for (const auto& page : res.pages) {
        if (page.state == MemState::Commit) {
            res_committed += kPageSize;
        }
    }
    if (res.shared) {
        // O36: views still mapped go away (64 KiB at a time, as they were mapped).
        constexpr size_t kPagesPerView = static_cast<size_t>(kAllocationGranularity / kPageSize);
        for (size_t p = 0; p < res.pages.size(); p += kPagesPerView) {
            if (res.pages[p].state == MemState::Commit) {
                backing::unmap_shared_page(translate(res.base + p * kPageSize), kAllocationGranularity);
            }
        }
    } else if (res_committed != 0) {
        backing::decommit(translate(res.base), res.size);
    }
    reserved_bytes_ -= res.size;
    committed_bytes_ -= res_committed;
    reservations_.erase(it);
    return true;
}

void* create_shared_page(uint64_t size) {
    return backing::shared_pages_supported() ? backing::create_shared_page(size) : nullptr;
}

void release_shared_page(void* page) {
    backing::release_shared_page(page);
}

uint8_t* view_shared_page(void* page, uint64_t size) {
    return backing::view_shared_page(page, size);
}

void unview_shared_page(uint8_t* view, uint64_t size) {
    backing::unview_shared_page(view, size);
}

uint64_t VirtualMemoryArena::reserve_shared(uint64_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t gran_size = 0;
    if (shared_zone_size_ == 0 || size == 0 || !align_up_checked(size, kAllocationGranularity, gran_size)) {
        notify(*this, ArenaEvent::Reserve, 0, size, PageProtect::NoAccess, false);
        return 0;
    }
    // First fit in the zone.
    const uint64_t zone_end = shared_zone_base_ + shared_zone_size_;
    uint64_t candidate = shared_zone_base_;
    for (auto it = reservations_.lower_bound(shared_zone_base_); it != reservations_.end(); ++it) {
        if (candidate + gran_size <= it->second.base) {
            break;
        }
        candidate = std::max(candidate, it->second.base + it->second.granularity_size);
    }
    if (candidate > zone_end || gran_size > zone_end - candidate) {
        notify(*this, ArenaEvent::Reserve, 0, size, PageProtect::NoAccess, false);
        return 0;
    }
    Reservation res;
    res.base = candidate;
    res.size = gran_size;
    res.granularity_size = gran_size;
    res.allocation_protect = PageProtect::NoAccess;
    res.pages.assign(static_cast<size_t>(gran_size / kPageSize), PageMeta{MemState::Reserve, PageProtect::NoAccess});
    res.shared = true;
    reservations_.emplace(candidate, std::move(res));
    reserved_bytes_ += gran_size;
    notify(*this, ArenaEvent::Reserve, candidate, size, PageProtect::NoAccess, true);
    return candidate;
}

bool VirtualMemoryArena::map_shared(uint64_t va, void* page, uint64_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    Reservation* res = find_reservation_locked(va);
    if (res == nullptr || !res->shared || page == nullptr || size == 0 || (va % kAllocationGranularity) != 0 ||
        (size % kAllocationGranularity) != 0 || va - res->base > res->size - size) {
        return false;
    }
    const size_t first = static_cast<size_t>((va - res->base) / kPageSize);
    const size_t count = static_cast<size_t>(size / kPageSize);
    if (res->pages[first].state == MemState::Commit) {
        backing::unmap_shared_page(translate(va), size);  // another page was shown here
        committed_bytes_ -= size;
    }
    if (!backing::map_shared_page(translate(va), page, size)) {
        for (size_t p = first; p < first + count; ++p) {
            res->pages[p] = PageMeta{MemState::Reserve, PageProtect::NoAccess};
        }
        return false;
    }
    for (size_t p = first; p < first + count; ++p) {
        res->pages[p] = PageMeta{MemState::Commit, PageProtect::ReadWrite};
    }
    committed_bytes_ += size;
    notify(*this, ArenaEvent::Commit, va, size, PageProtect::ReadWrite, true);
    return true;
}

bool VirtualMemoryArena::unmap_shared(uint64_t va, uint64_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    Reservation* res = find_reservation_locked(va);
    if (res == nullptr || !res->shared || size == 0 || va - res->base > res->size - size) {
        return false;
    }
    const size_t first = static_cast<size_t>((va - res->base) / kPageSize);
    const size_t count = static_cast<size_t>(size / kPageSize);
    if (res->pages[first].state != MemState::Commit) {
        return true;
    }
    backing::unmap_shared_page(translate(va), size);
    for (size_t p = first; p < first + count; ++p) {
        res->pages[p] = PageMeta{MemState::Reserve, PageProtect::NoAccess};
    }
    committed_bytes_ -= size;
    notify(*this, ArenaEvent::Decommit, va, size, PageProtect::NoAccess, true);
    return true;
}

bool VirtualMemoryArena::is_shared(uint64_t va) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const Reservation* res = find_reservation_locked(va);
    return res != nullptr && res->shared;
}

uint64_t VirtualMemoryArena::reserve(uint64_t hint_addr,
                                     uint64_t size,
                                     PageProtect allocation_protect) {
    std::lock_guard<std::mutex> lock(mutex_);
    const uint64_t base = reserve_locked(hint_addr, size, allocation_protect);
    notify(*this, ArenaEvent::Reserve, base != 0 ? base : hint_addr, size, allocation_protect, base != 0);
    return base;
}

bool VirtualMemoryArena::commit(uint64_t addr, uint64_t size, PageProtect protect) {
    std::lock_guard<std::mutex> lock(mutex_);
    const bool ok = commit_locked(addr, size, protect);
    notify(*this, ArenaEvent::Commit, addr, size, protect, ok);
    return ok;
}

bool VirtualMemoryArena::decommit(uint64_t addr, uint64_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    const bool ok = decommit_locked(addr, size);
    notify(*this, ArenaEvent::Decommit, addr, size, PageProtect::NoAccess, ok);
    return ok;
}

bool VirtualMemoryArena::protect(uint64_t addr,
                                 uint64_t size,
                                 PageProtect new_protect,
                                 PageProtect* old_protect) {
    if (size == 0 || !is_valid_protect(new_protect)) {
        return false;
    }
    if (addr > UINT64_MAX - size) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    Reservation* res = find_reservation_locked(addr);
    if (!res) {
        return false;
    }

    const uint64_t start = align_down(addr, kPageSize);
    uint64_t end = 0;
    if (!align_up_checked(addr + size, kPageSize, end)) {
        return false;
    }
    if (start < res->base || end > res->base + res->size) {
        return false;
    }

    const size_t first_page = static_cast<size_t>((start - res->base) / kPageSize);
    const size_t last_page = static_cast<size_t>((end - res->base) / kPageSize);

    for (size_t p = first_page; p < last_page; ++p) {
        if (res->pages[p].state != MemState::Commit) {
            return false;
        }
    }

    if (!backing::protect(translate(start), end - start, new_protect)) {
        return false;
    }
    if (old_protect != nullptr) {
        *old_protect = res->pages[first_page].protect;
    }
    for (size_t p = first_page; p < last_page; ++p) {
        res->pages[p].protect = new_protect;
    }
    return true;
}

bool VirtualMemoryArena::release(uint64_t base_addr) {
    std::lock_guard<std::mutex> lock(mutex_);
    const bool ok = release_locked(base_addr);
    notify(*this, ArenaEvent::Release, base_addr, 0, PageProtect::NoAccess, ok);
    return ok;
}

uint64_t VirtualMemoryArena::virtual_alloc(uint64_t addr,
                                           uint64_t size,
                                           uint32_t alloc_type,
                                           PageProtect protect) {
    constexpr uint32_t kAllowed = kMemReserve | kMemCommit | kMemReservePlaceholder | kMemReplacePlaceholder;
    if (size == 0 || alloc_type == 0 || (alloc_type & ~kAllowed) != 0) {
        return 0;
    }
    if (!is_valid_protect(protect)) {
        return 0;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if ((alloc_type & kMemReservePlaceholder) != 0) {
        // A new placeholder: reserve only, no access, allocation-granularity aligned.
        if (alloc_type != (kMemReserve | kMemReservePlaceholder) || protect != PageProtect::NoAccess ||
            (addr % kAllocationGranularity) != 0 || (size % kAllocationGranularity) != 0) {
            return 0;
        }
        const uint64_t base = reserve_locked(addr, size, protect);
        notify(*this, ArenaEvent::Reserve, base != 0 ? base : addr, size, protect, base != 0);
        if (base != 0) {
            reservations_.at(base).placeholder = true;
        }
        return base;
    }
    if ((alloc_type & kMemReplacePlaceholder) != 0) {
        // Exactly one placeholder becomes a normal reservation (address space is unchanged).
        if ((alloc_type & kMemReserve) == 0 || addr == 0) {
            return 0;
        }
        auto it = reservations_.find(addr);
        if (it == reservations_.end() || !it->second.placeholder || it->second.size != size) {
            return 0;
        }
        Reservation& res = it->second;
        res.placeholder = false;
        res.replaced_placeholder = true;
        res.allocation_protect = protect;
        if ((alloc_type & kMemCommit) != 0) {
            const bool ok = commit_locked(addr, size, protect);
            notify(*this, ArenaEvent::Commit, addr, size, protect, ok);
            if (!ok) {
                res.placeholder = true;  // back to a placeholder: the call fails as a whole
                res.replaced_placeholder = false;
                res.allocation_protect = PageProtect::NoAccess;
                return 0;
            }
        }
        return addr;
    }

    const bool do_reserve = (alloc_type & kMemReserve) != 0 || addr == 0;
    const bool do_commit = (alloc_type & kMemCommit) != 0;

    if (do_reserve) {
        const uint64_t base = reserve_locked(addr, size, protect);
        notify(*this, ArenaEvent::Reserve, base != 0 ? base : addr, size, protect, base != 0);
        if (base == 0) {
            return 0;
        }
        if (do_commit) {
            const Reservation* res = find_reservation_locked(base);
            const bool ok = res != nullptr && commit_locked(base, res->size, protect);
            notify(*this, ArenaEvent::Commit, base, size, protect, ok);
            if (!ok) {
                release_locked(base);
                notify(*this, ArenaEvent::Release, base, 0, PageProtect::NoAccess, true);
                return 0;
            }
        }
        return base;
    }

    // Commit only on an existing reservation.
    const bool ok = commit_locked(addr, size, protect);
    notify(*this, ArenaEvent::Commit, addr, size, protect, ok);
    if (!ok) {
        return 0;
    }
    return align_down(addr, kPageSize);
}

bool VirtualMemoryArena::split_placeholder_locked(uint64_t addr, uint64_t size) {
    if (size == 0 || (addr % kAllocationGranularity) != 0 || (size % kAllocationGranularity) != 0 ||
        addr > UINT64_MAX - size) {
        return false;
    }
    Reservation* found = find_reservation_locked(addr);
    if (found == nullptr) {
        return false;
    }
    const uint64_t base = found->base;
    const uint64_t end = base + found->size;
    if (addr + size > end) {
        return false;
    }
    if (!found->placeholder) {
        // A reservation that replaced a placeholder goes back to being one (its whole range).
        if (!found->replaced_placeholder || addr != base || size != found->size) {
            return false;
        }
        decommit_locked(base, found->size);
        found->placeholder = true;
        found->replaced_placeholder = false;
        found->allocation_protect = PageProtect::NoAccess;
        return true;
    }
    if (addr == base && size == found->size) {
        return true;  // already exactly this placeholder
    }
    // [base, addr) / [addr, addr + size) / [addr + size, end): each non-empty piece is a placeholder.
    Reservation whole = std::move(*found);
    reservations_.erase(base);
    auto piece = [&](uint64_t from, uint64_t to) {
        if (from == to) {
            return;
        }
        Reservation r;
        r.base = from;
        r.size = to - from;
        r.granularity_size = to - from;
        r.allocation_protect = PageProtect::NoAccess;
        r.placeholder = true;
        const auto first = whole.pages.begin() + static_cast<std::ptrdiff_t>((from - base) / kPageSize);
        const auto last = whole.pages.begin() + static_cast<std::ptrdiff_t>((to - base) / kPageSize);
        r.pages.assign(first, last);
        reservations_.emplace(from, std::move(r));
    };
    piece(base, addr);
    piece(addr, addr + size);
    piece(addr + size, end);
    return true;
}

bool VirtualMemoryArena::coalesce_placeholders_locked(uint64_t addr, uint64_t size) {
    if (size == 0 || addr > UINT64_MAX - size) {
        return false;
    }
    const uint64_t end = addr + size;
    // The placeholders starting at addr must tile [addr, end) exactly.
    uint64_t cursor = addr;
    std::vector<uint64_t> bases;
    while (cursor < end) {
        auto it = reservations_.find(cursor);
        if (it == reservations_.end() || !it->second.placeholder || it->second.size > end - cursor) {
            return false;
        }
        bases.push_back(cursor);
        cursor += it->second.size;
    }
    if (bases.size() < 2) {
        return bases.size() == 1;  // a single placeholder is already coalesced
    }
    Reservation merged;
    merged.base = addr;
    merged.size = size;
    merged.granularity_size = size;
    merged.allocation_protect = PageProtect::NoAccess;
    merged.placeholder = true;
    for (const uint64_t b : bases) {
        auto it = reservations_.find(b);
        merged.pages.insert(merged.pages.end(), it->second.pages.begin(), it->second.pages.end());
        reservations_.erase(it);
    }
    reservations_.emplace(addr, std::move(merged));
    return true;
}

bool VirtualMemoryArena::virtual_free(uint64_t addr, uint64_t size, uint32_t free_type) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (free_type == (kMemRelease | kMemPreservePlaceholder)) {
        return split_placeholder_locked(addr, size);
    }
    if (free_type == (kMemRelease | kMemCoalescePlaceholders)) {
        return coalesce_placeholders_locked(addr, size);
    }
    if (free_type == kMemDecommit) {
        const Reservation* res = find_reservation_locked(addr);
        if (res != nullptr && res->placeholder) {
            return false;
        }
    }
    if (free_type == kMemRelease) {
        if (size != 0) {
            return false;
        }
        const bool ok = release_locked(addr);
        notify(*this, ArenaEvent::Release, addr, 0, PageProtect::NoAccess, ok);
        return ok;
    }
    if (free_type == kMemDecommit) {
        const bool ok = decommit_locked(addr, size);
        notify(*this, ArenaEvent::Decommit, addr, size, PageProtect::NoAccess, ok);
        return ok;
    }
    return false;
}

MemoryBasicInfo VirtualMemoryArena::query(uint64_t addr) const {
    const uint64_t page_addr = align_down(addr, kPageSize);

    std::lock_guard<std::mutex> lock(mutex_);
    const Reservation* res = find_reservation_locked(page_addr);
    if (res != nullptr) {
        const size_t idx = static_cast<size_t>((page_addr - res->base) / kPageSize);
        const PageMeta& first = res->pages[idx];
        size_t end_idx = idx + 1;
        while (end_idx < res->pages.size() &&
               res->pages[end_idx].state == first.state &&
               res->pages[end_idx].protect == first.protect) {
            ++end_idx;
        }
        MemoryBasicInfo info;
        info.base_address = page_addr;
        info.allocation_base = res->base;
        info.region_size = static_cast<uint64_t>(end_idx - idx) * kPageSize;
        info.state = first.state;
        info.protect = first.protect;
        info.allocation_protect = res->allocation_protect;
        return info;
    }

    MemoryBasicInfo free_info;
    free_info.base_address = page_addr;
    free_info.allocation_base = 0;
    free_info.state = MemState::Free;
    free_info.protect = PageProtect::NoAccess;
    free_info.allocation_protect = PageProtect::NoAccess;

    // A free region extends to the next reservation, or to the end of the guest address space.
    auto next_it = reservations_.upper_bound(page_addr);
    const uint64_t space_end = space_base_ + space_size_;
    if (next_it != reservations_.end()) {
        free_info.region_size = next_it->second.base - page_addr;
    } else if (page_addr < space_end) {
        free_info.region_size = space_end - page_addr;
    } else {
        free_info.region_size = 0;
    }
    return free_info;
}

bool VirtualMemoryArena::read(uint64_t addr, void* dst, size_t len) const {
    if (len == 0) {
        return true;
    }
    if (dst == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pages_allow_locked(addr, len, false)) {
        return false;
    }
    std::memcpy(dst, translate(addr), len);
    return true;
}

bool VirtualMemoryArena::write(uint64_t addr, const void* src, size_t len) {
    if (len == 0) {
        return true;
    }
    if (src == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pages_allow_locked(addr, len, true)) {
        return false;
    }
    std::memcpy(translate(addr), src, len);
    return true;
}

uint8_t* VirtualMemoryArena::host_ptr(uint64_t addr, size_t len) {
    std::lock_guard<std::mutex> lock(mutex_);
    return pages_allow_locked(addr, len, true) ? translate(addr) : nullptr;
}

const uint8_t* VirtualMemoryArena::host_ptr_const(uint64_t addr, size_t len) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pages_allow_locked(addr, len, false) ? translate(addr) : nullptr;
}

uint64_t VirtualMemoryArena::reserved_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return reserved_bytes_;
}

uint64_t VirtualMemoryArena::committed_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return committed_bytes_;
}

size_t VirtualMemoryArena::reservation_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return reservations_.size();
}

void publish_guest_address_space(const VirtualMemoryArena& arena) {
    __playgta6_guest_base = arena.guest_base();
}

}  // namespace playgta6::os
