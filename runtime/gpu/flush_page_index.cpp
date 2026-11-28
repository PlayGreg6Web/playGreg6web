#include "gpu/flush_page_index.h"

#include <algorithm>
#include <limits>

namespace playgta6::gpu::d3d12_guest {

uint64_t HostPageIndex::next_visit_gen() const {
    if (++visit_gen_ == 0) {
        for (const Entry& e : pool_) {
            e.last_seen_gen = 0;
        }
        visit_gen_ = 1;
    }
    return visit_gen_;
}

void HostPageIndex::unlink(uint32_t id, uintptr_t lo, uintptr_t hi) {
    const uintptr_t g_lo = lo >> kGranuleShift;
    const uintptr_t g_hi = (hi - 1) >> kGranuleShift;
    for (uintptr_t g = g_lo;; ++g) {
        const auto it = by_granule_.find(g);
        if (it != by_granule_.end()) {
            auto& bucket = it->second;
            for (size_t idx = 0; idx < bucket.size(); ++idx) {
                if (bucket[idx].id == id) {
                    bucket[idx] = bucket.back();
                    bucket.pop_back();
                    break;
                }
            }
            if (bucket.empty()) {
                by_granule_.erase(it);
            }
        }
        if (g == g_hi) {
            break;
        }
    }
}

void HostPageIndex::add(const void* key, uintptr_t lo, uintptr_t hi) {
    if (hi <= lo) {
        remove(key);
        return;
    }
    const uintptr_t new_g_lo = lo >> kGranuleShift;
    const uintptr_t new_g_hi = (hi - 1) >> kGranuleShift;
    const auto it = key_to_id_.find(key);
    uint32_t id = 0;
    if (it != key_to_id_.end()) {
        id = it->second;
        Entry& e = pool_[id];
        if (e.lo == lo && e.hi == hi) {
            return;
        }
        const uintptr_t old_lo = e.lo;
        const uintptr_t old_hi = e.hi;
        const uintptr_t old_g_lo = old_lo >> kGranuleShift;
        const uintptr_t old_g_hi = (old_hi - 1) >> kGranuleShift;
        e.lo = lo;
        e.hi = hi;
        hulls_dirty_ = true;
        if (old_g_lo == new_g_lo && old_g_hi == new_g_hi) {
            for (uintptr_t g = new_g_lo;; ++g) {
                const auto bit = by_granule_.find(g);
                if (bit != by_granule_.end()) {
                    for (BucketItem& item : bit->second) {
                        if (item.id == id) {
                            item.lo = lo;
                            item.hi = hi;
                            break;
                        }
                    }
                }
                if (g == new_g_hi) {
                    break;
                }
            }
            return;
        }
        unlink(id, old_lo, old_hi);
    } else {
        if (!free_ids_.empty()) {
            id = free_ids_.back();
            free_ids_.pop_back();
            Entry& e = pool_[id];
            e.key = key;
            e.lo = lo;
            e.hi = hi;
            e.last_seen_gen = 0;
        } else {
            id = static_cast<uint32_t>(pool_.size());
            pool_.push_back(Entry{key, lo, hi, 0});
        }
        key_to_id_.emplace(key, id);
        hulls_dirty_ = true;
    }
    const BucketItem item{lo, hi, id};
    for (uintptr_t g = new_g_lo;; ++g) {
        by_granule_[g].push_back(item);
        if (g == new_g_hi) {
            break;
        }
    }
}

void HostPageIndex::remove(const void* key) {
    const auto it = key_to_id_.find(key);
    if (it == key_to_id_.end()) {
        return;
    }
    const uint32_t id = it->second;
    Entry& e = pool_[id];
    const uintptr_t old_lo = e.lo;
    const uintptr_t old_hi = e.hi;
    e.key = nullptr;
    e.lo = 0;
    e.hi = 0;
    e.last_seen_gen = 0;
    key_to_id_.erase(it);
    unlink(id, old_lo, old_hi);
    if (key_to_id_.empty()) {
        pool_.clear();
        free_ids_.clear();
    } else {
        free_ids_.push_back(id);
    }
    hulls_dirty_ = true;
}

bool HostPageIndex::range_of(const void* key, uintptr_t* lo, uintptr_t* hi) const {
    const auto it = key_to_id_.find(key);
    if (it == key_to_id_.end()) {
        return false;
    }
    const Entry& e = pool_[it->second];
    *lo = e.lo;
    *hi = e.hi;
    return true;
}

void HostPageIndex::clear() {
    key_to_id_.clear();
    pool_.clear();
    free_ids_.clear();
    by_granule_.clear();
    hulls_.clear();
    hulls_dirty_ = false;
    visit_gen_ = 0;
}

void HostPageIndex::collect_pages(const std::vector<uintptr_t>& pages, uintptr_t page_bytes,
                                  std::vector<const void*>* out) const {
    if (pages.empty() || page_bytes == 0 || key_to_id_.empty()) {
        return;
    }
    constexpr uintptr_t kMaxAddr = std::numeric_limits<uintptr_t>::max();
    const auto less_page_end = [page_bytes](uintptr_t v, uintptr_t page) {
        return v < page || (v - page) < page_bytes;
    };
    if (page_bytes > kDirectProbeGranules * kGranuleBytes) {
        for (const Entry& e : pool_) {
            if (e.lo >= e.hi) {
                continue;
            }
            const auto p = std::upper_bound(pages.begin(), pages.end(), e.lo, less_page_end);
            if (p != pages.end() && *p < e.hi) {
                out->push_back(e.key);
            }
        }
        return;
    }

    const uint64_t gen = next_visit_gen();
    size_t i = 0;
    while (i < pages.size()) {
        const uintptr_t g = pages[i] >> kGranuleShift;
        size_t j = i + 1;
        while (j < pages.size() && (pages[j] >> kGranuleShift) == g) {
            ++j;
        }
        const uintptr_t run_lo = pages[i];
        const uintptr_t last_p = pages[j - 1];
        const bool run_hi_wraps = (last_p > kMaxAddr - page_bytes);
        const uintptr_t run_hi = run_hi_wraps ? kMaxAddr : (last_p + page_bytes);
        // A page may also reach into subsequent granules (e.g. page_bytes > kGranuleBytes): probe
        // through the last page's final granule as well.
        const uintptr_t last_end_incl =
            (last_p > kMaxAddr - (page_bytes - 1)) ? kMaxAddr : (last_p + (page_bytes - 1));
        const uintptr_t g_last = last_end_incl >> kGranuleShift;
        const bool single_page = (j == i + 1);
        const auto run_begin = pages.begin() + static_cast<std::ptrdiff_t>(i);
        const auto run_end = pages.begin() + static_cast<std::ptrdiff_t>(j);

        for (uintptr_t gk = g;; ++gk) {
            const auto it = by_granule_.find(gk);
            if (it != by_granule_.end()) {
                for (const BucketItem& item : it->second) {
                    if (item.hi <= run_lo || (!run_hi_wraps && item.lo >= run_hi)) {
                        continue;
                    }
                    const Entry& e = pool_[item.id];
                    if (e.last_seen_gen == gen) {
                        continue;
                    }
                    if (single_page) {
                        e.last_seen_gen = gen;
                        out->push_back(e.key);
                    } else {
                        // First page of the run that ends after item.lo: does it start before item.hi?
                        const auto p = std::upper_bound(run_begin, run_end, item.lo, less_page_end);
                        if (p != run_end && *p < item.hi) {
                            e.last_seen_gen = gen;
                            out->push_back(e.key);
                        }
                    }
                }
            }
            if (gk == g_last) {
                break;
            }
        }
        i = j;
    }
}

void HostPageIndex::collect_range(uintptr_t lo, uintptr_t hi, std::vector<const void*>* out) const {
    if (hi <= lo || key_to_id_.empty()) {
        return;
    }
    const uintptr_t g_lo = lo >> kGranuleShift;
    const uintptr_t g_hi = (hi - 1) >> kGranuleShift;
    if (g_lo == g_hi) {
        const auto it = by_granule_.find(g_lo);
        if (it != by_granule_.end()) {
            for (const BucketItem& item : it->second) {
                if (item.lo < hi && lo < item.hi) {
                    out->push_back(pool_[item.id].key);
                }
            }
        }
        return;
    }
    if (g_hi - g_lo > kDirectProbeGranules) {
        for (const Entry& e : pool_) {
            if (e.lo < e.hi && e.lo < hi && lo < e.hi) {
                out->push_back(e.key);
            }
        }
        return;
    }
    const uint64_t gen = next_visit_gen();
    for (uintptr_t g = g_lo;; ++g) {
        const auto it = by_granule_.find(g);
        if (it != by_granule_.end()) {
            for (const BucketItem& item : it->second) {
                if (item.lo < hi && lo < item.hi) {
                    const Entry& e = pool_[item.id];
                    if (e.last_seen_gen != gen) {
                        e.last_seen_gen = gen;
                        out->push_back(e.key);
                    }
                }
            }
        }
        if (g == g_hi) {
            break;
        }
    }
}

const std::vector<HostPageIndex::Range>& HostPageIndex::hulls() const {
    if (hulls_dirty_) {
        hulls_.clear();
        if (!key_to_id_.empty()) {
            hulls_.reserve(key_to_id_.size());
            constexpr uintptr_t kMaxAddr = std::numeric_limits<uintptr_t>::max();
            for (const Entry& e : pool_) {
                if (e.lo < e.hi) {
                    const uintptr_t aligned_lo = e.lo & ~(kHullPage - 1);
                    const uintptr_t aligned_hi = (e.hi > kMaxAddr - (kHullPage - 1))
                                                     ? kMaxAddr
                                                     : ((e.hi + kHullPage - 1) & ~(kHullPage - 1));
                    hulls_.emplace_back(aligned_lo, aligned_hi);
                }
            }
            std::sort(hulls_.begin(), hulls_.end());
            size_t out_n = 0;
            for (const Range& r : hulls_) {
                if (out_n != 0) {
                    const uintptr_t prev_hi = hulls_[out_n - 1].second;
                    if (r.first <= prev_hi || r.first - prev_hi <= kHullGap) {
                        hulls_[out_n - 1].second = std::max(prev_hi, r.second);
                        continue;
                    }
                }
                hulls_[out_n++] = r;
            }
            hulls_.resize(out_n);
        }
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
