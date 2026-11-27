// Host page index for the CPU-visible buffer flush (#1963, #663).
//
// Device::flush_cpu_buffers used to visit every tracked buffer and memcmp every relocated uniform
// range at every ExecuteCommandLists. On native builds the OS write tracking (MEM_WRITE_WATCH plus
// the shared page views, os/vmem.h) already says which host pages were written; this index maps
// host address ranges (buffers, relocated ranges) to keys, so one take over the merged ranges
// (hulls()) names the few keys whose bytes can have changed (collect_pages).
//
// Keys are 64 KiB granules apart at most: a key is listed in every granule its range touches, and a
// query checks the exact range against the pages, so overlapping and page-straddling ranges and
// ranges sharing a granule are all handled. add() with a new range for a known key re-keys it (a
// remap or a resize); remove() drops it. Not thread-safe: the owner locks.
// A write landing after the flush's take is uploaded at the next submission, as on the old path
// (writes after a buffer's own take went next time too); nothing is lost. --flush-crosscheck counts
// these as gpu_work.flush_crosscheck_late (only flush_crosscheck_misses would be a bug).
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace playgta6::gpu::d3d12_guest {

class HostPageIndex {
public:
    using Range = std::pair<uintptr_t, uintptr_t>;  // [first, second)
    static constexpr unsigned kGranuleShift = 16;
    static constexpr uintptr_t kGranuleBytes = uintptr_t{1} << kGranuleShift;

    // Indexes [lo, hi) under `key`; an earlier range of the key is replaced. An empty range removes it.
    void add(const void* key, uintptr_t lo, uintptr_t hi);
    void remove(const void* key);
    bool contains(const void* key) const { return entries_.count(key) != 0; }
    // The indexed range of `key` (false when it is not indexed).
    bool range_of(const void* key, uintptr_t* lo, uintptr_t* hi) const;
    void clear();
    size_t size() const { return entries_.size(); }

    // Appends once each key whose range overlaps any page [p, p + page_bytes) of `pages` (sorted,
    // unique, page-aligned host addresses).
    void collect_pages(const std::vector<uintptr_t>& pages, uintptr_t page_bytes, std::vector<const void*>* out) const;
    // Appends once each key whose range overlaps [lo, hi).
    void collect_range(uintptr_t lo, uintptr_t hi, std::vector<const void*>* out) const;
    // Sorted, disjoint ranges covering every indexed range: each range widened to whole 4 KiB pages,
    // and ranges at most kHullGap apart merged (fewer takes; only small gaps are taken in addition).
    static constexpr uintptr_t kHullPage = 4096;
    static constexpr uintptr_t kHullGap = 4 * kHullPage;
    const std::vector<Range>& hulls() const;

private:
    std::unordered_map<const void*, Range> entries_;
    std::multiset<Range> sorted_;  // the ranges of entries_ in order, so hulls() needs no sort
    std::map<uintptr_t, std::vector<const void*>> by_granule_;  // granule number -> keys touching it
    mutable std::vector<Range> hulls_;
    mutable bool hulls_dirty_ = false;
    void unlink(const void* key, const Range& r);
};

// Sorted disjoint ranges (HostPageIndex::hulls, merge_ranges) cover [lo, hi) as a whole.
bool ranges_cover(const std::vector<HostPageIndex::Range>& sorted, uintptr_t lo, uintptr_t hi);
// Sorts `ranges` and merges overlapping or touching ones in place.
void merge_ranges(std::vector<HostPageIndex::Range>* ranges);

}  // namespace playgta6::gpu::d3d12_guest
