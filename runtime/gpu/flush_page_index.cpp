#include "gpu/flush_page_index.h"

#include <algorithm>
#include <unordered_set>

namespace playgta6::gpu::d3d12_guest {

void HostPageIndex::unlink(const void* key, const Range& r) {
    const uintptr_t g_lo = r.first >> kGranuleShift;
    const uintptr_t g_hi = (r.second - 1) >> kGranuleShift;
    for (auto it = by_granule_.lower_bound(g_lo); it != by_granule_.end() && it->first <= g_hi;) {
        auto& keys = it->second;
        keys.erase(std::remove(keys.begin(), keys.end(), key), keys.end());
        if (keys.empty()) {
            it = by_granule_.erase(it);
        } else {
            ++it;
        }
    }
}

void HostPageIndex::add(const void* key, uintptr_t lo, uintptr_t hi) {
    if (hi <= lo) {
        remove(key);
        return;
    }
    auto it = entries_.find(key);
    if (it != entries_.end()) {
        if (it->second.first == lo && it->second.second == hi) {
            return;
        }
        unlink(key, it->second);
        it->second = {lo, hi};
    } else {
        entries_.emplace(key, Range{lo, hi});
    }
    hulls_dirty_ = true;
    for (uintptr_t g = lo >> kGranuleShift, last = (hi - 1) >> kGranuleShift;; ++g) {
        by_granule_[g].push_back(key);
        if (g == last) {
            break;
        }
    }
}

void HostPageIndex::remove(const void* key) {
    auto it = entries_.find(key);
    if (it == entries_.end()) {
        return;
    }
    unlink(key, it->second);
    entries_.erase(it);
    hulls_dirty_ = true;
}

bool HostPageIndex::range_of(const void* key, uintptr_t* lo, uintptr_t* hi) const {
    auto it = entries_.find(key);
    if (it == entries_.end()) {
        return false;
    }
    *lo = it->second.first;
    *hi = it->second.second;
    return true;
}

void HostPageIndex::clear() {
    entries_.clear();
    by_granule_.clear();
    hulls_.clear();
    hulls_dirty_ = false;
}

void HostPageIndex::collect_pages(const std::vector<uintptr_t>& pages, uintptr_t page_bytes,
                                  std::vector<const void*>* out) const {
    std::unordered_set<const void*> seen;
    size_t i = 0;
    while (i < pages.size()) {
        const uintptr_t g = pages[i] >> kGranuleShift;
        size_t j = i + 1;
        while (j < pages.size() && (pages[j] >> kGranuleShift) == g) {
            ++j;
        }
        // A page may also reach into the next granule (a page larger than the alignment): look the
        // page's last granule up as well.
        const uintptr_t g_last = (pages[j - 1] + page_bytes - 1) >> kGranuleShift;
        for (auto it = by_granule_.lower_bound(g); it != by_granule_.end() && it->first <= g_last; ++it) {
            for (const void* key : it->second) {
                if (seen.count(key) != 0) {
                    continue;
                }
                const Range& r = entries_.at(key);
                // First page of the run that ends after r.first: does it start before r.second?
                auto p = std::upper_bound(pages.begin() + static_cast<std::ptrdiff_t>(i),
                                          pages.begin() + static_cast<std::ptrdiff_t>(j), r.first,
                                          [page_bytes](uintptr_t v, uintptr_t page) { return v < page + page_bytes; });
                if (p != pages.begin() + static_cast<std::ptrdiff_t>(j) && *p < r.second) {
                    seen.insert(key);
                    out->push_back(key);
                }
            }
        }
        i = j;
    }
}

void HostPageIndex::collect_range(uintptr_t lo, uintptr_t hi, std::vector<const void*>* out) const {
    if (hi <= lo) {
        return;
    }
    std::unordered_set<const void*> seen;
    for (auto it = by_granule_.lower_bound(lo >> kGranuleShift);
         it != by_granule_.end() && it->first <= ((hi - 1) >> kGranuleShift); ++it) {
        for (const void* key : it->second) {
            const Range& r = entries_.at(key);
            if (r.first < hi && lo < r.second && seen.insert(key).second) {
                out->push_back(key);
            }
        }
    }
}

const std::vector<HostPageIndex::Range>& HostPageIndex::hulls() const {
    if (hulls_dirty_) {
        hulls_.clear();
        hulls_.reserve(entries_.size());
        for (const auto& [key, r] : entries_) {
            hulls_.emplace_back(r.first & ~(kHullPage - 1), (r.second + kHullPage - 1) & ~(kHullPage - 1));
        }
        std::sort(hulls_.begin(), hulls_.end());
        size_t n = 0;
        for (const auto& r : hulls_) {
            if (n != 0 && r.first <= hulls_[n - 1].second + kHullGap) {
                hulls_[n - 1].second = std::max(hulls_[n - 1].second, r.second);
            } else {
                hulls_[n++] = r;
            }
        }
        hulls_.resize(n);
        hulls_dirty_ = false;
    }
    return hulls_;
}

bool ranges_cover(const std::vector<HostPageIndex::Range>& sorted, uintptr_t lo, uintptr_t hi) {
    if (hi <= lo) {
        return false;
    }
    // The last range starting at or before lo.
    auto it = std::upper_bound(sorted.begin(), sorted.end(), lo,
                               [](uintptr_t v, const HostPageIndex::Range& r) { return v < r.first; });
    if (it == sorted.begin()) {
        return false;
    }
    --it;
    return it->first <= lo && hi <= it->second;
}

void merge_ranges(std::vector<HostPageIndex::Range>* ranges) {
    std::sort(ranges->begin(), ranges->end());
    size_t n = 0;
    for (const auto& r : *ranges) {
        if (r.second <= r.first) {
            continue;
        }
        if (n != 0 && r.first <= (*ranges)[n - 1].second) {
            (*ranges)[n - 1].second = std::max((*ranges)[n - 1].second, r.second);
        } else {
            (*ranges)[n++] = r;
        }
    }
    ranges->resize(n);
}

}  // namespace playgta6::gpu::d3d12_guest
