// Host page index of the CPU-visible buffer flush (#1963, gpu/flush_page_index.h): keys by host
// range, queried by written pages; synthetic addresses only.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "gpu/flush_page_index.h"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                                \
    do {                                                                           \
        if (!(cond)) {                                                             \
            std::fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                          \
        }                                                                          \
    } while (0)

using playgta6::gpu::d3d12_guest::HostPageIndex;
using playgta6::gpu::d3d12_guest::merge_ranges;
using playgta6::gpu::d3d12_guest::ranges_cover;
using Range = HostPageIndex::Range;

constexpr uintptr_t kPage = 4096;
constexpr uintptr_t kBase = uintptr_t{0x7000'0000'0000};
constexpr uintptr_t k64K = uintptr_t{64} << 10;

int key_storage[16];
const void* K(int i) { return &key_storage[i]; }

std::vector<const void*> pages_hit(const HostPageIndex& idx, std::vector<uintptr_t> pages) {
    std::sort(pages.begin(), pages.end());
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    std::vector<const void*> out;
    idx.collect_pages(pages, kPage, &out);
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<const void*> sorted(std::vector<const void*> v) {
    std::sort(v.begin(), v.end());
    return v;
}

bool has(const std::vector<const void*>& v, const void* k) { return std::find(v.begin(), v.end(), k) != v.end(); }

void test_basic_and_page_straddling() {
    HostPageIndex idx;
    // A buffer that starts mid-page and ends mid-page in the next one (straddles a page boundary).
    idx.add(K(0), kBase + 0xF00, kBase + kPage + 0x100);
    // One page later, wholly inside page 3.
    idx.add(K(1), kBase + 3 * kPage + 0x10, kBase + 3 * kPage + 0x20);
    CHECK(idx.size() == 2 && idx.contains(K(0)) && idx.contains(K(1)));
    CHECK(pages_hit(idx, {kBase}) == sorted({K(0)}));
    CHECK(pages_hit(idx, {kBase + kPage}) == sorted({K(0)}));
    CHECK(pages_hit(idx, {kBase + 2 * kPage}).empty());
    CHECK(pages_hit(idx, {kBase + 3 * kPage}) == sorted({K(1)}));
    CHECK(pages_hit(idx, {kBase + 4 * kPage}).empty());
    // Both, each once although two of K(0)'s pages are listed.
    CHECK(pages_hit(idx, {kBase, kBase + kPage, kBase + 3 * kPage}) == sorted({K(0), K(1)}));
    // A range ending exactly at a page boundary does not reach the next page.
    idx.add(K(2), kBase + 5 * kPage, kBase + 6 * kPage);
    CHECK(pages_hit(idx, {kBase + 6 * kPage}).empty());
    CHECK(pages_hit(idx, {kBase + 5 * kPage}) == sorted({K(2)}));
}

void test_overlapping_and_shared_granules() {
    HostPageIndex idx;
    // Overlapping ranges, and a large one spanning several 64 KiB granules around them.
    idx.add(K(0), kBase + 0x1000, kBase + 0x3000);
    idx.add(K(1), kBase + 0x2000, kBase + 0x5000);
    idx.add(K(2), kBase, kBase + 4 * k64K);
    CHECK(pages_hit(idx, {kBase + 0x2000}) == sorted({K(0), K(1), K(2)}));
    CHECK(pages_hit(idx, {kBase + 0x1000}) == sorted({K(0), K(2)}));
    CHECK(pages_hit(idx, {kBase + 0x4000}) == sorted({K(1), K(2)}));
    CHECK(pages_hit(idx, {kBase + 3 * k64K + 0x5000}) == sorted({K(2)}));
    CHECK(pages_hit(idx, {kBase + 4 * k64K}).empty());
    // collect_range: by overlap, each key once.
    std::vector<const void*> out;
    idx.collect_range(kBase + 0x2800, kBase + 0x2900, &out);
    CHECK(sorted(out) == sorted({K(0), K(1), K(2)}));
    out.clear();
    idx.collect_range(kBase + 0x3000, kBase + 0x3001, &out);  // K(0) ends at 0x3000
    CHECK(sorted(out) == sorted({K(1), K(2)}));
    out.clear();
    idx.collect_range(kBase + 2 * k64K, kBase + 3 * k64K, &out);
    CHECK(sorted(out) == sorted({K(2)}));
    // One hull: the ranges overlap.
    const auto& h = idx.hulls();
    CHECK(h.size() == 1 && h[0] == Range(kBase, kBase + 4 * k64K));
}

void test_map_unmap_remap_and_resize() {
    HostPageIndex idx;
    idx.add(K(0), kBase + 0x100, kBase + 0x200);
    idx.add(K(1), kBase + 10 * k64K, kBase + 10 * k64K + 0x100);
    CHECK(idx.hulls().size() == 2);
    // Remap (#1952-style: the same key's bytes show at another host address): the old pages no
    // longer name it, the new ones do.
    idx.add(K(0), kBase + 20 * k64K + 0x100, kBase + 20 * k64K + 0x200);
    CHECK(pages_hit(idx, {kBase}).empty());
    CHECK(pages_hit(idx, {kBase + 20 * k64K}) == sorted({K(0)}));
    uintptr_t lo = 0, hi = 0;
    CHECK(idx.range_of(K(0), &lo, &hi) && lo == kBase + 20 * k64K + 0x100 && hi == kBase + 20 * k64K + 0x200);
    {
        const auto& h = idx.hulls();
        CHECK(h.size() == 2 && h[0] == Range(kBase + 10 * k64K, kBase + 10 * k64K + kPage) &&
              h[1] == Range(kBase + 20 * k64K, kBase + 20 * k64K + kPage));
    }
    // Resize in place: grows into the next granule, then shrinks back.
    idx.add(K(0), kBase + 20 * k64K + 0x100, kBase + 21 * k64K + 0x10);
    CHECK(pages_hit(idx, {kBase + 21 * k64K}) == sorted({K(0)}));
    CHECK(idx.hulls().back() == Range(kBase + 20 * k64K, kBase + 21 * k64K + kPage));
    idx.add(K(0), kBase + 20 * k64K + 0x100, kBase + 20 * k64K + 0x200);
    CHECK(pages_hit(idx, {kBase + 21 * k64K}).empty());
    CHECK(idx.hulls().back() == Range(kBase + 20 * k64K, kBase + 20 * k64K + kPage));
    // Re-adding the same range changes nothing.
    idx.add(K(0), kBase + 20 * k64K + 0x100, kBase + 20 * k64K + 0x200);
    CHECK(idx.size() == 2);
    // Unmap: removed keys are never named; their granules leave the hulls.
    idx.remove(K(1));
    CHECK(!idx.contains(K(1)) && pages_hit(idx, {kBase + 10 * k64K}).empty());
    CHECK(idx.hulls().size() == 1);
    idx.remove(K(1));  // twice: no-op
    // An empty range removes too.
    idx.add(K(0), kBase, kBase);
    CHECK(idx.size() == 0 && idx.hulls().empty());
    // Two keys in one granule: removing one keeps the other (and the granule).
    idx.add(K(2), kBase + 0x1000, kBase + 0x1100);
    idx.add(K(3), kBase + 0x8000, kBase + 0x8100);
    idx.remove(K(2));
    CHECK(pages_hit(idx, {kBase + 0x1000}).empty());
    CHECK(pages_hit(idx, {kBase + 0x8000}) == sorted({K(3)}));
    CHECK(idx.hulls().size() == 1 && idx.hulls()[0] == Range(kBase + 0x8000, kBase + 0x9000));
    idx.clear();
    CHECK(idx.size() == 0 && idx.hulls().empty() && pages_hit(idx, {kBase + 0x8000}).empty());
}

void test_many_pages_one_query() {
    // A long dirty run over many granules and many small keys: each key named once, only the
    // keys on written pages.
    HostPageIndex idx;
    for (int i = 0; i < 16; ++i) {
        idx.add(K(i), kBase + static_cast<uintptr_t>(i) * 3 * kPage + 0x80, kBase + static_cast<uintptr_t>(i) * 3 * kPage + 0x180);
    }
    std::vector<uintptr_t> pages;
    for (int i = 0; i < 16; i += 2) {
        pages.push_back(kBase + static_cast<uintptr_t>(i) * 3 * kPage);
        pages.push_back(kBase + static_cast<uintptr_t>(i) * 3 * kPage + kPage);  // next page: no key
    }
    const auto hit = pages_hit(idx, pages);
    CHECK(hit.size() == 8);
    for (int i = 0; i < 16; ++i) {
        CHECK(has(hit, K(i)) == (i % 2 == 0));
    }
    // Page-aligned ranges 2 pages apart (within HostPageIndex::kHullGap): merged into one hull.
    CHECK(idx.hulls().size() == 1 && idx.hulls()[0] == Range(kBase, kBase + 46 * kPage));
    // A gap wider than kHullGap keeps two hulls.
    idx.add(K(0), kBase + 100 * kPage, kBase + 100 * kPage + 1);
    CHECK(idx.hulls().size() == 2 && idx.hulls()[1] == Range(kBase + 100 * kPage, kBase + 101 * kPage));
}

void test_cover_and_merge() {
    std::vector<Range> r = {{300, 400}, {100, 200}, {150, 250}, {250, 260}, {500, 500}, {600, 700}};
    merge_ranges(&r);
    CHECK((r == std::vector<Range>{{100, 260}, {300, 400}, {600, 700}}));
    CHECK(ranges_cover(r, 100, 260));
    CHECK(ranges_cover(r, 120, 130));
    CHECK(!ranges_cover(r, 90, 110));
    CHECK(!ranges_cover(r, 250, 310));  // spans a gap
    CHECK(ranges_cover(r, 650, 700));
    CHECK(!ranges_cover(r, 650, 701));
    CHECK(!ranges_cover(r, 50, 60));
    CHECK(!ranges_cover(r, 800, 900));
    CHECK(!ranges_cover(r, 120, 120));  // empty
    CHECK(!ranges_cover({}, 1, 2));
}

// Brute-force reference of collect_pages.
std::vector<const void*> ref_pages(const std::vector<std::pair<const void*, Range>>& m, const std::vector<uintptr_t>& pages,
                                   uintptr_t page_bytes) {
    std::vector<const void*> out;
    if (page_bytes == 0) {
        return out;
    }
    for (const auto& [k, r] : m) {
        for (uintptr_t p : pages) {
            if ((r.first < p || (r.first - p) < page_bytes) && p < r.second) {
                out.push_back(k);
                break;
            }
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<const void*> ref_range(const std::vector<std::pair<const void*, Range>>& m, uintptr_t lo, uintptr_t hi) {
    std::vector<const void*> out;
    if (hi <= lo) {
        return out;
    }
    for (const auto& [k, r] : m) {
        if (r.first < hi && lo < r.second) {
            out.push_back(k);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

struct Lcg {
    uint64_t s = 0x9E3779B97F4A7C15ull;
    uint64_t next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return s >> 33;
    }
    uint64_t below(uint64_t n) { return next() % n; }
};

// Random add / re-add (remap, resize) / remove / zero-length / same-address re-creation against a
// brute-force model, queried with 4 KiB pages and with pages larger than the 64 KiB granule.
void test_model_fuzz() {
    Lcg rng;
    static int keys[24];
    for (const uintptr_t page_bytes : {uintptr_t{4096}, uintptr_t{65536}, uintptr_t{2} << 20}) {
        HostPageIndex idx;
        std::vector<std::pair<const void*, Range>> model;
        auto model_remove = [&](const void* k) {
            model.erase(std::remove_if(model.begin(), model.end(), [&](const auto& e) { return e.first == k; }), model.end());
        };
        for (int step = 0; step < 4000; ++step) {
            const void* k = &keys[rng.below(24)];
            const uint64_t op = rng.below(10);
            if (op < 6) {
                // Boundaries are biased onto page / granule edges (+-1) to hit straddles.
                auto edge = [&] {
                    const uintptr_t unit = rng.below(3) == 0 ? uintptr_t{65536} : uintptr_t{4096};
                    const uintptr_t v = kBase + rng.below(40) * unit;
                    const uint64_t d = rng.below(4);
                    return d == 0 ? v - 1 : d == 1 ? v + 1 : v;
                };
                uintptr_t lo = edge(), hi = edge();
                if (rng.below(8) == 0) {
                    hi = lo;  // zero length
                }
                if (rng.below(10) == 0) {
                    hi = lo + 1 + rng.below(4 << 20);  // large
                }
                idx.add(k, lo, hi);
                model_remove(k);
                if (hi > lo) {
                    model.emplace_back(k, Range(lo, hi));
                }
            } else if (op < 8) {
                idx.remove(k);
                model_remove(k);
            } else {
                // Destroyed and re-created at the same host range under the same key address.
                uintptr_t lo = 0, hi = 0;
                if (idx.range_of(k, &lo, &hi)) {
                    idx.remove(k);
                    CHECK(!idx.contains(k));
                    idx.add(k, lo, hi);
                }
            }
            CHECK(idx.size() == model.size());
            if (step % 8 != 0) {
                continue;
            }
            std::vector<uintptr_t> pages;
            const uint64_t n = 1 + rng.below(6);
            for (uint64_t i = 0; i < n; ++i) {
                pages.push_back((kBase / page_bytes + rng.below(4 * (uintptr_t{4} << 20) / page_bytes + 2)) * page_bytes);
            }
            std::sort(pages.begin(), pages.end());
            pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
            std::vector<const void*> got;
            idx.collect_pages(pages, page_bytes, &got);
            std::sort(got.begin(), got.end());
            CHECK(std::adjacent_find(got.begin(), got.end()) == got.end());  // each key once
            CHECK(got == ref_pages(model, pages, page_bytes));
            const uintptr_t a = kBase + rng.below(5 << 20), b = a + rng.below(1 << 18);
            std::vector<const void*> rg;
            idx.collect_range(a, b, &rg);
            std::sort(rg.begin(), rg.end());
            std::vector<const void*> want;
            for (const auto& [mk, r] : model) {
                if (r.first < b && a < r.second) {
                    want.push_back(mk);
                }
            }
            std::sort(want.begin(), want.end());
            CHECK(rg == want);
            // hulls: sorted, disjoint, and each indexed range is covered whole (what `primed` relies on).
            const auto& h = idx.hulls();
            for (size_t i = 1; i < h.size(); ++i) {
                CHECK(h[i - 1].second < h[i].first);
            }
            for (const auto& [mk, r] : model) {
                CHECK(ranges_cover(h, r.first, r.second));
            }
        }
    }
}

// Exact boundaries: one-byte ranges at page edges, zero length, re-creation under a recycled key.
void test_page_edges_exact() {
    HostPageIndex idx;
    idx.add(K(0), kBase + kPage - 1, kBase + kPage);    // last byte of page 0
    idx.add(K(1), kBase + kPage, kBase + kPage + 1);    // first byte of page 1
    idx.add(K(2), kBase + k64K - 1, kBase + k64K + 1);  // straddles a granule and a page boundary
    idx.add(K(3), kBase + 2 * k64K, kBase + 2 * k64K);  // zero length: never indexed
    CHECK(!idx.contains(K(3)) && idx.size() == 3);
    CHECK(pages_hit(idx, {kBase}) == sorted({K(0)}));
    CHECK(pages_hit(idx, {kBase + kPage}) == sorted({K(1)}));
    CHECK(pages_hit(idx, {kBase + k64K - kPage}) == sorted({K(2)}));
    CHECK(pages_hit(idx, {kBase + k64K}) == sorted({K(2)}));
    CHECK(pages_hit(idx, {kBase + 2 * k64K}).empty());
    std::vector<const void*> out;
    idx.collect_range(kBase + 5, kBase + 5, &out);  // empty query range
    idx.collect_range(kBase + 9, kBase + 3, &out);  // inverted
    CHECK(out.empty());
    // A zero-length add of an existing key drops it (the buffer shrank to nothing).
    idx.add(K(0), kBase + 7, kBase + 7);
    CHECK(!idx.contains(K(0)) && pages_hit(idx, {kBase}).empty());
    // Re-created at the same address under a recycled key: no stale extent.
    idx.add(K(1), kBase + 30 * k64K, kBase + 31 * k64K);
    CHECK(pages_hit(idx, {kBase + kPage}).empty());
    CHECK(pages_hit(idx, {kBase + 30 * k64K + 5 * kPage}) == sorted({K(1)}));
    idx.remove(K(1));
    idx.add(K(1), kBase + 30 * k64K, kBase + 30 * k64K + kPage);
    CHECK(pages_hit(idx, {kBase + 30 * k64K + 5 * kPage}).empty());
    // Two keys over one identical range are both named; removing one leaves the other.
    idx.add(K(4), kBase + 30 * k64K, kBase + 30 * k64K + kPage);
    CHECK(pages_hit(idx, {kBase + 30 * k64K}) == sorted({K(1), K(4)}));
    idx.remove(K(1));
    CHECK(pages_hit(idx, {kBase + 30 * k64K}) == sorted({K(4)}));
    // A hull list taken before an add does not cover the new range (so it stays unprimed).
    const std::vector<Range> snap = idx.hulls();
    idx.add(K(5), kBase + 50 * k64K, kBase + 50 * k64K + 100);
    CHECK(!ranges_cover(snap, kBase + 50 * k64K, kBase + 50 * k64K + 100));
    CHECK(ranges_cover(idx.hulls(), kBase + 50 * k64K, kBase + 50 * k64K + 100));
    std::vector<Range> r = {{10, 20}, {20, 30}, {12, 14}, {40, 40}, {0, 5}};
    merge_ranges(&r);
    CHECK((r == std::vector<Range>{{0, 5}, {10, 30}}));
    CHECK(ranges_cover(r, 10, 30) && !ranges_cover(r, 4, 11));
}

// Repeated collect_pages / collect_range calls must reset generation-based deduplication cleanly
// across calls, never drop keys on subsequent queries, and preserve existing elements in `out`.
void test_repeated_queries_and_dedup_reset() {
    HostPageIndex idx;
    idx.add(K(0), kBase, kBase + 4 * k64K);                       // spans 4 granules
    idx.add(K(1), kBase + k64K - kPage, kBase + 2 * k64K + 128);  // straddles granules 0, 1, 2
    idx.add(K(2), kBase + 2 * k64K + 0x100, kBase + 2 * k64K + 0x200);
    idx.add(K(3), kBase + 100 * k64K, kBase + 103 * k64K);

    const std::vector<uintptr_t> multi_granule_pages = {
        kBase + 0x1000,
        kBase + k64K - kPage,
        kBase + k64K + 0x2000,
        kBase + 2 * k64K,
    };
    const auto want_pages = sorted({K(0), K(1), K(2)});
    const auto want_narrow = sorted({K(0), K(1), K(2)});
    const auto want_single = sorted({K(0), K(1)});
    const auto want_wide = sorted({K(0), K(1), K(2), K(3)});

    std::vector<const void*> out;
    for (int iter = 0; iter < 64; ++iter) {
        out.clear();
        idx.collect_pages(multi_granule_pages, kPage, &out);
        CHECK(out.size() == want_pages.size());
        CHECK(sorted(out) == want_pages);

        out.clear();
        idx.collect_range(kBase + k64K, kBase + k64K + kPage, &out);  // single granule
        CHECK(out.size() == want_single.size());
        CHECK(sorted(out) == want_single);

        out.clear();
        idx.collect_range(kBase + 0x2000, kBase + 3 * k64K, &out);  // narrow multi-granule
        CHECK(out.size() == want_narrow.size());
        CHECK(sorted(out) == want_narrow);

        out.clear();
        idx.collect_range(kBase, kBase + 105 * k64K, &out);  // wide (> 64 granules)
        CHECK(out.size() == want_wide.size());
        CHECK(sorted(out) == want_wide);
    }

    // Empty pages and zero page_bytes are no-ops and do not affect subsequent queries.
    out.clear();
    idx.collect_pages({}, kPage, &out);
    idx.collect_pages(multi_granule_pages, 0, &out);
    CHECK(out.empty());

    // Appending to a non-empty `out` preserves pre-existing entries.
    out.push_back(K(15));
    idx.collect_pages({kBase + 101 * k64K}, kPage, &out);
    CHECK(out.size() == 2 && out[0] == K(15) && out[1] == K(3));
}

// Large-range collect_range (> 64 granules) vs narrow-range collect_range (1..64 granules), plus
// multi-granule page sizes (both <= 64 granules and > 64 granules per page).
void test_large_vs_narrow_range_and_multi_granule_pages() {
    HostPageIndex idx;
    std::vector<std::pair<const void*, Range>> model;
    auto put = [&](const void* k, uintptr_t lo, uintptr_t hi) {
        idx.add(k, lo, hi);
        model.emplace_back(k, Range(lo, hi));
    };

    put(K(0), kBase + 0x100, kBase + 0x900);
    put(K(1), kBase + 1 * k64K - 16, kBase + 3 * k64K + 16);
    put(K(2), kBase + 32 * k64K + 0x2000, kBase + 33 * k64K + 0x4000);
    put(K(3), kBase + 64 * k64K, kBase + 65 * k64K);
    put(K(4), kBase + 65 * k64K + 0x100, kBase + 70 * k64K + 0x100);
    put(K(5), kBase + 130 * k64K + 0x5000, kBase + 131 * k64K);
    put(K(6), kBase + 195 * k64K, kBase + 200 * k64K);

    // Narrow single-granule, exact 64-granule boundary, 65-granule boundary, and 210-granule spans.
    const std::vector<Range> queries = {
        {kBase + 0x200, kBase + 0x300},
        {kBase + 2 * k64K, kBase + 2 * k64K + 1},
        {kBase, kBase + 64 * k64K},
        {kBase, kBase + 65 * k64K},
        {kBase + 64 * k64K, kBase + 135 * k64K},
        {kBase, kBase + 210 * k64K},
        {kBase + 70 * k64K + 0x100, kBase + 130 * k64K + 0x5000},  // gap between K(4) and K(5)
    };
    for (const Range& q : queries) {
        std::vector<const void*> got;
        idx.collect_range(q.first, q.second, &got);
        std::sort(got.begin(), got.end());
        CHECK(std::adjacent_find(got.begin(), got.end()) == got.end());
        CHECK(got == ref_range(model, q.first, q.second));
    }

    // Multi-granule page_bytes: 128 KiB (2 granules), 256 KiB (4 granules), and 8 MiB (128 granules > 64).
    for (const uintptr_t pg : {2 * k64K, 4 * k64K, 128 * k64K}) {
        const std::vector<uintptr_t> pages = {
            kBase,
            kBase + pg,
            kBase + 3 * pg,
        };
        std::vector<const void*> got;
        idx.collect_pages(pages, pg, &got);
        std::sort(got.begin(), got.end());
        CHECK(std::adjacent_find(got.begin(), got.end()) == got.end());
        CHECK(got == ref_pages(model, pages, pg));
    }
}

// Swap-and-pop removal integrity in a crowded granule, same-granule and cross-granule re-keying,
// clear(), and extreme address boundaries (0 and UINTPTR_MAX).
void test_swap_and_pop_rekey_clear_and_extremes() {
    HostPageIndex idx;
    uint8_t crowded[32] = {};
    std::vector<std::pair<const void*, Range>> model;

    // 32 entries packed inside a single 64 KiB granule (each 1 KiB apart).
    for (size_t i = 0; i < 32; ++i) {
        const uintptr_t lo = kBase + static_cast<uintptr_t>(i) * 2048 + 64;
        const uintptr_t hi = lo + 512;
        idx.add(&crowded[i], lo, hi);
        model.emplace_back(&crowded[i], Range(lo, hi));
    }
    CHECK(idx.size() == 32);

    // Re-key several entries in-place within the same granule (grow, shrink, shift).
    for (size_t i = 0; i < 32; i += 4) {
        const uintptr_t lo = kBase + static_cast<uintptr_t>(i) * 2048 + 128;
        const uintptr_t hi = lo + 1024;
        idx.add(&crowded[i], lo, hi);
        model[i].second = Range(lo, hi);
        uintptr_t got_lo = 0, got_hi = 0;
        CHECK(idx.range_of(&crowded[i], &got_lo, &got_hi) && got_lo == lo && got_hi == hi);
    }

    // Remove front, middle, back, and every even index; verify remaining entries after each step.
    const size_t remove_order[] = {0, 31, 15, 16, 2, 4, 6, 8, 10, 12, 14, 18, 20, 22, 24, 26, 28, 30};
    for (size_t rem : remove_order) {
        const void* k = &crowded[rem];
        idx.remove(k);
        model.erase(std::remove_if(model.begin(), model.end(), [&](const auto& e) { return e.first == k; }), model.end());
        CHECK(!idx.contains(k) && idx.size() == model.size());
        std::vector<const void*> got;
        idx.collect_range(kBase, kBase + k64K, &got);
        std::sort(got.begin(), got.end());
        CHECK(got == ref_range(model, kBase, kBase + k64K));
    }

    // Drain remaining entries to empty, then re-add to exercise pool reset and slot reuse.
    while (!model.empty()) {
        const void* k = model.back().first;
        idx.remove(k);
        model.pop_back();
    }
    CHECK(idx.size() == 0 && idx.hulls().empty());

    // Extreme addresses: lo == 0 and hi == UINTPTR_MAX (must not overflow align_up, kHullGap, or page_end).
    // K(1) ends at aligned 0x...C000 and K(2) starts at aligned 0x...F000 (gap 3 pages <= kHullGap,
    // while prev_hi + kHullGap == 0x...C000 + 0x4000 would wrap to 0 if computed naively).
    constexpr uintptr_t kMaxAddr = std::numeric_limits<uintptr_t>::max();
    const uintptr_t top_page = kMaxAddr - kPage + 1;
    idx.add(K(0), 0, 1);
    idx.add(K(1), kMaxAddr - 20000, kMaxAddr - 17000);
    idx.add(K(2), kMaxAddr - 100, kMaxAddr);
    CHECK(idx.size() == 3);

    const auto& h = idx.hulls();
    CHECK(h.size() == 2);
    CHECK(h[0] == Range(0, kPage));
    CHECK(ranges_cover(h, 0, 1));
    CHECK(ranges_cover(h, kMaxAddr - 20000, kMaxAddr - 17000));
    CHECK(ranges_cover(h, kMaxAddr - 100, kMaxAddr));
    CHECK(h[0].second < h[1].first);

    CHECK(pages_hit(idx, {0}) == sorted({K(0)}));
    CHECK(pages_hit(idx, {top_page}) == sorted({K(2)}));
    {
        std::vector<const void*> got;
        idx.collect_range(kMaxAddr - 200, kMaxAddr, &got);
        CHECK(sorted(got) == sorted({K(2)}));
    }

    idx.clear();
    CHECK(idx.size() == 0 && idx.hulls().empty());
    idx.add(K(0), kBase + 0x100, kBase + 0x200);
    CHECK(idx.size() == 1 && pages_hit(idx, {kBase}) == sorted({K(0)}));
}

// 5,000-entry stress parity test against a brute-force overlap oracle across adds, removals,
// re-keys, free-list reuse, narrow/wide collect_range, and multi-granule collect_pages.
void test_stress_5000_entries_parity() {
    constexpr size_t kNumKeys = 5000;
    std::vector<uint8_t> key_buf(kNumKeys, 0);
    HostPageIndex idx;
    std::vector<Range> model_ranges(kNumKeys, Range(0, 0));  // empty range means absent
    Lcg rng;
    rng.s = 0xD1B54A32D192ED03ull;

    auto make_range = [&]() -> Range {
        const uintptr_t gran = rng.below(512);
        const uintptr_t off = rng.below(k64K);
        const uintptr_t lo = kBase + gran * k64K + off;
        uintptr_t len = 0;
        const uint64_t kind = rng.below(10);
        if (kind < 5) {
            len = 1 + rng.below(kPage);  // small sub-page
        } else if (kind < 8) {
            len = kPage + rng.below(k64K);  // multi-page / granule-straddling
        } else {
            len = k64K + rng.below(6 * k64K);  // multi-granule
        }
        return Range(lo, lo + len);
    };

    // 1. Insert 5,000 entries.
    for (size_t i = 0; i < kNumKeys; ++i) {
        const Range r = make_range();
        idx.add(&key_buf[i], r.first, r.second);
        model_ranges[i] = r;
    }
    CHECK(idx.size() == kNumKeys);

    // 2. Remove 1,250 entries.
    for (size_t i = 0; i < kNumKeys; i += 4) {
        idx.remove(&key_buf[i]);
        model_ranges[i] = Range(0, 0);
    }
    CHECK(idx.size() == kNumKeys - kNumKeys / 4);

    // 3. Re-key 1,000 live entries (half within same granule, half to random granules).
    for (size_t i = 1; i < 4000; i += 4) {
        Range r;
        if ((i / 4) % 2 == 0) {
            const uintptr_t g_base = model_ranges[i].first & ~(k64K - 1);
            const uintptr_t lo = g_base + rng.below(k64K - 256);
            r = Range(lo, lo + 1 + rng.below(256));
        } else {
            r = make_range();
        }
        idx.add(&key_buf[i], r.first, r.second);
        model_ranges[i] = r;
    }

    // 4. Re-add 750 removed entries (exercises free-list slot reuse).
    for (size_t i = 0; i < 3000; i += 4) {
        const Range r = make_range();
        idx.add(&key_buf[i], r.first, r.second);
        model_ranges[i] = r;
    }

    // Build flat model vector and verify size() / contains() / range_of().
    std::vector<std::pair<const void*, Range>> model;
    model.reserve(kNumKeys);
    for (size_t i = 0; i < kNumKeys; ++i) {
        uintptr_t lo = 0, hi = 0;
        if (model_ranges[i].first < model_ranges[i].second) {
            model.emplace_back(&key_buf[i], model_ranges[i]);
            CHECK(idx.contains(&key_buf[i]));
            CHECK(idx.range_of(&key_buf[i], &lo, &hi) && lo == model_ranges[i].first && hi == model_ranges[i].second);
        } else {
            CHECK(!idx.contains(&key_buf[i]));
            CHECK(!idx.range_of(&key_buf[i], &lo, &hi));
        }
    }
    CHECK(idx.size() == model.size());

    // 5. Verify hulls invariants (sorted, strictly disjoint, covers every live entry).
    const auto& h = idx.hulls();
    for (size_t i = 1; i < h.size(); ++i) {
        CHECK(h[i - 1].second < h[i].first);
    }
    for (const auto& [k, r] : model) {
        (void)k;
        CHECK(ranges_cover(h, r.first, r.second));
    }

    // 6. Verify collect_pages parity across 4 KiB, 64 KiB, 2 MiB, and 8 MiB pages.
    for (const uintptr_t pg : {kPage, k64K, uintptr_t{2} << 20, uintptr_t{8} << 20}) {
        for (int q = 0; q < 4; ++q) {
            std::vector<uintptr_t> pages;
            const size_t n = 1 + static_cast<size_t>(rng.below(12));
            for (size_t p = 0; p < n; ++p) {
                const uintptr_t addr = ((kBase + rng.below(520 * k64K)) / pg) * pg;
                pages.push_back(addr);
            }
            std::sort(pages.begin(), pages.end());
            pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
            std::vector<const void*> got;
            idx.collect_pages(pages, pg, &got);
            std::sort(got.begin(), got.end());
            CHECK(std::adjacent_find(got.begin(), got.end()) == got.end());
            CHECK(got == ref_pages(model, pages, pg));
        }
    }

    // 7. Verify collect_range parity across single-granule, narrow, boundary, and wide spans.
    const std::vector<uintptr_t> spans = {kPage, 8 * k64K, 64 * k64K, 65 * k64K, 300 * k64K, 600 * k64K};
    for (const uintptr_t span : spans) {
        const uintptr_t lo = kBase + rng.below(200 * k64K);
        const uintptr_t hi = lo + span;
        std::vector<const void*> got;
        idx.collect_range(lo, hi, &got);
        std::sort(got.begin(), got.end());
        CHECK(std::adjacent_find(got.begin(), got.end()) == got.end());
        CHECK(got == ref_range(model, lo, hi));
    }
}

}  // namespace

int main() {
    test_model_fuzz();
    test_page_edges_exact();
    test_basic_and_page_straddling();
    test_overlapping_and_shared_granules();
    test_map_unmap_remap_and_resize();
    test_many_pages_one_query();
    test_cover_and_merge();
    test_repeated_queries_and_dedup_reset();
    test_large_vs_narrow_range_and_multi_granule_pages();
    test_swap_and_pop_rekey_clear_and_extremes();
    test_stress_5000_entries_parity();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("flush page index tests passed\n");
    return 0;
}
