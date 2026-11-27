// Host page index of the CPU-visible buffer flush (#1963, gpu/flush_page_index.h): keys by host
// range, queried by written pages; synthetic addresses only.
#include <algorithm>
#include <cstdint>
#include <cstdio>
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

}  // namespace

int main() {
    test_basic_and_page_straddling();
    test_overlapping_and_shared_granules();
    test_map_unmap_remap_and_resize();
    test_many_pages_one_query();
    test_cover_and_merge();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("flush page index tests passed\n");
    return 0;
}
