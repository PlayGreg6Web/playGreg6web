#include "os/vmem.h"

#include <algorithm>
#include <cstring>

namespace playgta6::os {

VirtualMemoryArena::VirtualMemoryArena(uint64_t default_base) {
    uint64_t aligned = kDefaultArenaBase;
    if (default_base >= kAllocationGranularity &&
        align_up_checked(default_base, kAllocationGranularity, aligned) &&
        aligned >= kAllocationGranularity) {
        default_base_ = aligned;
    } else {
        default_base_ = kDefaultArenaBase;
    }
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
    uint64_t candidate = default_base_;
    for (const auto& [base, res] : reservations_) {
        const uint64_t res_end = res.base + res.granularity_size;
        if (res_end <= candidate) {
            continue;
        }
        if (candidate <= UINT64_MAX - granularity_size &&
            candidate + granularity_size <= res.base) {
            return candidate;
        }
        if (res_end > candidate) {
            candidate = res_end;
        }
    }
    if (candidate > UINT64_MAX - granularity_size) {
        return 0;
    }
    return candidate;
}

uint64_t VirtualMemoryArena::reserve_locked(uint64_t hint_addr,
                                            uint64_t size,
                                            PageProtect allocation_protect) {
    if (size == 0 || !is_valid_protect(allocation_protect)) {
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

    // Identify maximal uncommitted runs and allocate zero-initialized backing buffers first.
    std::vector<CommittedExtent> new_extents;
    uint64_t newly_committed_bytes = 0;

    size_t i = first_page;
    while (i < last_page) {
        if (res->pages[i].state == MemState::Commit) {
            ++i;
            continue;
        }
        size_t run_start = i;
        while (i < last_page && res->pages[i].state != MemState::Commit) {
            ++i;
        }
        const uint64_t run_offset = static_cast<uint64_t>(run_start) * kPageSize;
        const uint64_t run_bytes = static_cast<uint64_t>(i - run_start) * kPageSize;

        CommittedExtent ext;
        ext.offset = run_offset;
        ext.size = run_bytes;
        ext.data = std::make_unique<uint8_t[]>(static_cast<size_t>(run_bytes));
        new_extents.push_back(std::move(ext));
        newly_committed_bytes += run_bytes;
    }

    if (!new_extents.empty()) {
        for (auto& ext : new_extents) {
            auto pos = std::lower_bound(
                res->extents.begin(),
                res->extents.end(),
                ext.offset,
                [](const CommittedExtent& existing, uint64_t off) {
                    return existing.offset < off;
                });
            res->extents.insert(pos, std::move(ext));
        }
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
    if (!res) {
        return false;
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

    const uint64_t rel_start = start - res->base;
    const uint64_t rel_end = end - res->base;
    const size_t first_page = static_cast<size_t>(rel_start / kPageSize);
    const size_t last_page = static_cast<size_t>(rel_end / kPageSize);

    // Rebuild extents list, trimming or splitting any extent partially overlapping [rel_start, rel_end).
    std::vector<CommittedExtent> updated_extents;
    updated_extents.reserve(res->extents.size() + 1);

    for (auto& ext : res->extents) {
        const uint64_t ext_start = ext.offset;
        const uint64_t ext_end = ext.offset + ext.size;

        if (ext_end <= rel_start || ext_start >= rel_end) {
            // Completely disjoint.
            updated_extents.push_back(std::move(ext));
            continue;
        }

        // Left surviving slice.
        if (ext_start < rel_start) {
            const uint64_t left_size = rel_start - ext_start;
            CommittedExtent left;
            left.offset = ext_start;
            left.size = left_size;
            left.data = std::make_unique<uint8_t[]>(static_cast<size_t>(left_size));
            std::memcpy(left.data.get(), ext.data.get(), static_cast<size_t>(left_size));
            updated_extents.push_back(std::move(left));
        }

        // Right surviving slice.
        if (ext_end > rel_end) {
            const uint64_t right_size = ext_end - rel_end;
            const uint64_t src_off = rel_end - ext_start;
            CommittedExtent right;
            right.offset = rel_end;
            right.size = right_size;
            right.data = std::make_unique<uint8_t[]>(static_cast<size_t>(right_size));
            std::memcpy(right.data.get(),
                        ext.data.get() + static_cast<size_t>(src_off),
                        static_cast<size_t>(right_size));
            updated_extents.push_back(std::move(right));
        }
    }

    res->extents = std::move(updated_extents);

    uint64_t freed_bytes = 0;
    for (size_t p = first_page; p < last_page; ++p) {
        if (res->pages[p].state == MemState::Commit) {
            freed_bytes += kPageSize;
        }
        res->pages[p].state = MemState::Reserve;
        res->pages[p].protect = PageProtect::NoAccess;
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
    reserved_bytes_ -= res.size;
    committed_bytes_ -= res_committed;
    reservations_.erase(it);
    return true;
}

uint64_t VirtualMemoryArena::reserve(uint64_t hint_addr,
                                     uint64_t size,
                                     PageProtect allocation_protect) {
    std::lock_guard<std::mutex> lock(mutex_);
    return reserve_locked(hint_addr, size, allocation_protect);
}

bool VirtualMemoryArena::commit(uint64_t addr, uint64_t size, PageProtect protect) {
    std::lock_guard<std::mutex> lock(mutex_);
    return commit_locked(addr, size, protect);
}

bool VirtualMemoryArena::decommit(uint64_t addr, uint64_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    return decommit_locked(addr, size);
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
    return release_locked(base_addr);
}

uint64_t VirtualMemoryArena::virtual_alloc(uint64_t addr,
                                           uint64_t size,
                                           uint32_t alloc_type,
                                           PageProtect protect) {
    constexpr uint32_t kAllowed = kMemReserve | kMemCommit;
    if (size == 0 || alloc_type == 0 || (alloc_type & ~kAllowed) != 0) {
        return 0;
    }
    if (!is_valid_protect(protect)) {
        return 0;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    const bool do_reserve = (alloc_type & kMemReserve) != 0 || addr == 0;
    const bool do_commit = (alloc_type & kMemCommit) != 0;

    if (do_reserve) {
        const uint64_t base = reserve_locked(addr, size, protect);
        if (base == 0) {
            return 0;
        }
        if (do_commit) {
            const Reservation* res = find_reservation_locked(base);
            if (!res || !commit_locked(base, res->size, protect)) {
                release_locked(base);
                return 0;
            }
        }
        return base;
    }

    // Commit only on an existing reservation.
    if (!commit_locked(addr, size, protect)) {
        return 0;
    }
    return align_down(addr, kPageSize);
}

bool VirtualMemoryArena::virtual_free(uint64_t addr, uint64_t size, uint32_t free_type) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (free_type == kMemRelease) {
        if (size != 0) {
            return false;
        }
        return release_locked(addr);
    }
    if (free_type == kMemDecommit) {
        return decommit_locked(addr, size);
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

    auto next_it = reservations_.upper_bound(page_addr);
    if (next_it != reservations_.end()) {
        free_info.region_size = next_it->second.base - page_addr;
    } else {
        free_info.region_size = 0;
    }
    return free_info;
}

bool VirtualMemoryArena::read(uint64_t addr, void* dst, size_t len) const {
    if (len == 0) {
        return true;
    }
    if (dst == nullptr || addr > UINT64_MAX - len) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const Reservation* res = find_reservation_locked(addr);
    if (!res) {
        return false;
    }
    const uint64_t end = addr + len;
    if (end > res->base + res->size) {
        return false;
    }

    const size_t first_page = static_cast<size_t>((addr - res->base) / kPageSize);
    const size_t last_page = static_cast<size_t>((end - 1u - res->base) / kPageSize);
    for (size_t p = first_page; p <= last_page; ++p) {
        if (res->pages[p].state != MemState::Commit || !is_readable(res->pages[p].protect)) {
            return false;
        }
    }

    uint8_t* out = static_cast<uint8_t*>(dst);
    uint64_t cursor = addr - res->base;
    size_t remaining = len;

    for (const auto& ext : res->extents) {
        if (remaining == 0) {
            break;
        }
        const uint64_t ext_end = ext.offset + ext.size;
        if (ext_end <= cursor) {
            continue;
        }
        if (ext.offset > cursor) {
            return false;
        }
        const uint64_t in_ext_off = cursor - ext.offset;
        const size_t chunk = static_cast<size_t>(
            std::min<uint64_t>(remaining, ext_end - cursor));
        std::memcpy(out, ext.data.get() + static_cast<size_t>(in_ext_off), chunk);
        out += chunk;
        cursor += chunk;
        remaining -= chunk;
    }
    return remaining == 0;
}

bool VirtualMemoryArena::write(uint64_t addr, const void* src, size_t len) {
    if (len == 0) {
        return true;
    }
    if (src == nullptr || addr > UINT64_MAX - len) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    Reservation* res = find_reservation_locked(addr);
    if (!res) {
        return false;
    }
    const uint64_t end = addr + len;
    if (end > res->base + res->size) {
        return false;
    }

    const size_t first_page = static_cast<size_t>((addr - res->base) / kPageSize);
    const size_t last_page = static_cast<size_t>((end - 1u - res->base) / kPageSize);
    for (size_t p = first_page; p <= last_page; ++p) {
        if (res->pages[p].state != MemState::Commit || !is_writable(res->pages[p].protect)) {
            return false;
        }
    }

    const uint8_t* in = static_cast<const uint8_t*>(src);
    uint64_t cursor = addr - res->base;
    size_t remaining = len;

    for (auto& ext : res->extents) {
        if (remaining == 0) {
            break;
        }
        const uint64_t ext_end = ext.offset + ext.size;
        if (ext_end <= cursor) {
            continue;
        }
        if (ext.offset > cursor) {
            return false;
        }
        const uint64_t in_ext_off = cursor - ext.offset;
        const size_t chunk = static_cast<size_t>(
            std::min<uint64_t>(remaining, ext_end - cursor));
        std::memcpy(ext.data.get() + static_cast<size_t>(in_ext_off), in, chunk);
        in += chunk;
        cursor += chunk;
        remaining -= chunk;
    }
    return remaining == 0;
}

uint8_t* VirtualMemoryArena::host_ptr(uint64_t addr, size_t len) {
    if (len == 0 || addr > UINT64_MAX - len) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    Reservation* res = find_reservation_locked(addr);
    if (!res) {
        return nullptr;
    }
    const uint64_t end = addr + len;
    if (end > res->base + res->size) {
        return nullptr;
    }

    const size_t first_page = static_cast<size_t>((addr - res->base) / kPageSize);
    const size_t last_page = static_cast<size_t>((end - 1u - res->base) / kPageSize);
    for (size_t p = first_page; p <= last_page; ++p) {
        if (res->pages[p].state != MemState::Commit || !is_writable(res->pages[p].protect)) {
            return nullptr;
        }
    }

    const uint64_t rel_start = addr - res->base;
    const uint64_t rel_end = end - res->base;
    for (auto& ext : res->extents) {
        if (rel_start >= ext.offset && rel_end <= ext.offset + ext.size) {
            return ext.data.get() + static_cast<size_t>(rel_start - ext.offset);
        }
    }
    return nullptr;
}

const uint8_t* VirtualMemoryArena::host_ptr_const(uint64_t addr, size_t len) const {
    if (len == 0 || addr > UINT64_MAX - len) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const Reservation* res = find_reservation_locked(addr);
    if (!res) {
        return nullptr;
    }
    const uint64_t end = addr + len;
    if (end > res->base + res->size) {
        return nullptr;
    }

    const size_t first_page = static_cast<size_t>((addr - res->base) / kPageSize);
    const size_t last_page = static_cast<size_t>((end - 1u - res->base) / kPageSize);
    for (size_t p = first_page; p <= last_page; ++p) {
        if (res->pages[p].state != MemState::Commit || !is_readable(res->pages[p].protect)) {
            return nullptr;
        }
    }

    const uint64_t rel_start = addr - res->base;
    const uint64_t rel_end = end - res->base;
    for (const auto& ext : res->extents) {
        if (rel_start >= ext.offset && rel_end <= ext.offset + ext.size) {
            return ext.data.get() + static_cast<size_t>(rel_start - ext.offset);
        }
    }
    return nullptr;
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

}  // namespace playgta6::os
