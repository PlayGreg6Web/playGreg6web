// XMem CPU/GPU windows (O27): recognition, alias pairs, phantom placement, physical pages mapped
// into the windows with one copy of the bytes, GPU page table, accounting and the browser budget
// planner replaying the title's reservation sequence. Synthetic data only.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "os/vmem.h"
#include "os/win32_kernel.h"
#include "os/xbox_scarlett.h"
#include "os/xmem_windows.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_xmem_windows.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6::os;
namespace xw = playgta6::os::xmem_windows;
using namespace playgta6::os::scarlett;

constexpr uint64_t GiB = 1ull << 30;
constexpr uint64_t MiB = 1ull << 20;
constexpr uint64_t k64K = 64ull << 10;

// The title's window reservation shapes (xmem_windows.h): MEM_RESERVE | MEM_64K_PAGES, no commit.
constexpr uint32_t kWindowType = 0x20402000u;
constexpr uint32_t kCpuProtect = 0x4u;    // PAGE_READWRITE
constexpr uint32_t kGpuProtect = 0x801u;  // PAGE_GRAPHICS_NOACCESS | PAGE_NOACCESS

VirtualMemoryArena& arena() {
    return *win32::get_vmem_arena();
}

uint64_t va_of(const void* p) {
    return arena().guest_address(p);
}

// Window size used in process-arena tests: the real 16 GiB where the arena can hold it (native,
// 64 GiB), 1 GiB in the 4 GiB wasm test arena.
uint64_t window_size() {
    return arena().space_size() >= 40 * GiB ? 16 * GiB : 1 * GiB;
}

void reset() {
    xmem_reset_for_testing();  // also resets xmem_windows
}

void test_classification() {
    xw::WindowKind kind{};
    CHECK(xw::classify(kWindowType, kCpuProtect, 16 * GiB, &kind) && kind == xw::WindowKind::Cpu);
    CHECK(xw::classify(kWindowType, kGpuProtect, 16 * GiB, &kind) && kind == xw::WindowKind::Gpu);
    CHECK(!xw::classify(kWindowType | 0x1000u, kCpuProtect, 16 * GiB, &kind));  // committed
    CHECK(!xw::classify(0x20002000u, kCpuProtect, 16 * GiB, &kind));            // no MEM_PHYSICAL bit
    CHECK(!xw::classify(kWindowType, kCpuProtect, 256 * MiB, &kind));           // a heap, not a window
}

void test_windows_alias_pairs_and_listeners() {
    reset();
    xw::set_policy(xw::PlacementPolicy::Arena);
    std::vector<xw::AliasPair> early, late;
    const uint64_t early_id = xw::add_alias_listener([&](const xw::AliasPair& p) { early.push_back(p); });
    const uint64_t S = window_size();
    const uint64_t reserved0 = arena().reserved_bytes();
    void* cpu = pg6_xbox_XMemVirtualAlloc(nullptr, S, kWindowType, 0x21u, kCpuProtect);
    void* gpu = pg6_xbox_XMemVirtualAlloc(nullptr, S, kWindowType, 0x3u, kGpuProtect);
    CHECK(cpu != nullptr && gpu != nullptr);
    CHECK(arena().reserved_bytes() == reserved0 + 2 * S);  // Arena policy: both in the pool
    const auto pairs = xw::alias_pairs();
    CHECK(pairs.size() == 1 && pairs[0].cpu_base == va_of(cpu) && pairs[0].gpu_base == va_of(gpu) &&
          pairs[0].size == S);
    CHECK(early.size() == 1 && early[0].gpu_base == va_of(gpu));
    // A listener subscribing later sees the existing pair at once.
    const uint64_t late_id = xw::add_alias_listener([&](const xw::AliasPair& p) { late.push_back(p); });
    CHECK(late.size() == 1 && late[0].cpu_base == va_of(cpu));
    // A second GPU window of the same size has no CPU partner left: no new pair.
    void* gpu2 = pg6_xbox_XMemVirtualAlloc(nullptr, S, kWindowType, 0x3u, kGpuProtect);
    CHECK(gpu2 != nullptr && xw::alias_pairs().size() == 1 && early.size() == 1);
    const xw::Accounting a = xw::accounting();
    CHECK(a.cpu_windows == 1 && a.gpu_windows == 2 && a.alias_pairs == 1 && a.cpu_window_reserved == S &&
          a.gpu_window_reserved == 2 * S && a.phantom_reserved == 0);
    // Releasing a window drops its pair.
    CHECK(xmem_virtual_free(gpu));
    CHECK(xw::alias_pairs().empty());
    CHECK(xmem_virtual_free(gpu2) && xmem_virtual_free(cpu));
    CHECK(arena().reserved_bytes() == reserved0);
    xw::remove_alias_listener(early_id);
    xw::remove_alias_listener(late_id);
    reset();
}

void test_phantom_gpu_window() {
    reset();
    xw::set_policy(xw::PlacementPolicy::PhantomGpuWindows);
    const uint64_t S = window_size();
    const uint64_t reserved0 = arena().reserved_bytes();
    void* cpu = pg6_xbox_XMemVirtualAlloc(nullptr, S, kWindowType, 0x21u, kCpuProtect);
    void* gpu = pg6_xbox_XMemVirtualAlloc(nullptr, S, kWindowType, 0x3u, kGpuProtect);
    CHECK(cpu != nullptr && gpu != nullptr);
    const uint64_t gpu_va = va_of(gpu);
    CHECK(gpu_va == xw::kPhantomBase && xw::is_phantom(gpu_va, S) && !xw::is_phantom(va_of(cpu)));
    CHECK(arena().reserved_bytes() == reserved0 + S);  // only the CPU window takes pool space
    // VirtualQuery reports the phantom window as a no-access reservation.
    win32::MemoryBasicInformation mbi{};
    CHECK(win32::VirtualQuery(gpu_va + 5 * k64K + 12, &mbi, sizeof(mbi)) == sizeof(mbi));
    CHECK(mbi.state == win32::MEM_RESERVE && mbi.allocation_base == gpu_va && mbi.protect == win32::PAGE_NOACCESS);
    // The runtime seam never translates a phantom address.
    CHECK(arena().host_ptr(gpu_va, 1) == nullptr);
    CHECK(xw::alias_pairs().size() == 1 && xw::accounting().phantom_reserved == S);
    CHECK(xmem_virtual_free(gpu) && !xw::is_phantom(gpu_va));
    CHECK(xmem_virtual_free(cpu));
    CHECK(arena().reserved_bytes() == reserved0);
    reset();
}

// The streaming pattern: a pool of pages, mapped once into a pool view, then mapped page by page
// into the CPU window (scatter) and the GPU window; one copy of the bytes, commits only for
// mapped CPU pages.
void test_pool_pages_in_windows() {
    reset();
    xw::set_policy(xw::PlacementPolicy::PhantomGpuWindows);
    const uint64_t S = window_size();
    void* cpu = pg6_xbox_XMemVirtualAlloc(nullptr, S, kWindowType, 0x21u, kCpuProtect);
    void* gpu = pg6_xbox_XMemVirtualAlloc(nullptr, S, kWindowType, 0x3u, kGpuProtect);
    CHECK(cpu != nullptr && gpu != nullptr);
    const uint64_t C = va_of(cpu), G = va_of(gpu);

    size_t count = 4;
    uint64_t pfn[4]{};
    CHECK(pg6_xbox_XMemAllocatePhysicalPages(0x380u, &count, pfn) == 1 && count == 4);
    // Pool view (a CPU-accessible physical reservation smaller than a window).
    void* view = pg6_xbox_XMemVirtualAlloc(nullptr, 4 * k64K, kWindowType, 0x23u, 0x2004u);
    CHECK(view != nullptr);
    CHECK(pg6_xbox_XMemMapPhysicalPages(view, 4, pfn) == view);
    auto* v = static_cast<uint8_t*>(view);
    v[0] = 0xA1;            // page 0
    v[2 * k64K + 7] = 0xC3;  // page 2

    // Map pool pages 2 and 0 at window pages 1000 and 3 of the CPU window: the bytes move there.
    const uint64_t committed0 = arena().committed_bytes();
    const uint64_t cpu_vas[2] = {C + 1000 * k64K, C + 3 * k64K};
    const uint64_t pages[2] = {pfn[2], pfn[0]};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(cpu_vas, 2, pages) == 1);
    CHECK(arena().committed_bytes() == committed0 + 2 * k64K);  // only the mapped pages
    const auto* w1000 = static_cast<const uint8_t*>(arena().translate(C + 1000 * k64K));
    const auto* w3 = static_cast<const uint8_t*>(arena().translate(C + 3 * k64K));
    CHECK(w1000[7] == 0xC3 && w3[0] == 0xA1);
    CHECK(xw::accounting().cpu_window_mapped == 2 * k64K);
    // The same pages in the phantom GPU window: no bytes move, nothing is committed.
    const uint64_t gpu_vas[2] = {G + 1000 * k64K, G + 3 * k64K};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(gpu_vas, 2, pages) == 1);
    CHECK(arena().committed_bytes() == committed0 + 2 * k64K);
    CHECK(w1000[7] == 0xC3);
    // The CPU writes through the window; unmapping the window page saves the bytes, a later
    // mapping (here back into the window at another index) shows them.
    const_cast<uint8_t*>(w3)[1] = 0x5A;
    const uint64_t unmap_vas[1] = {C + 3 * k64K};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(unmap_vas, 1, nullptr) == 1);
    CHECK(arena().query(C + 3 * k64K).state == MemState::Reserve);  // the window page is decommitted
    CHECK(arena().query(C + 1000 * k64K).state == MemState::Commit);
    const uint64_t again_vas[1] = {C + 7 * k64K};
    const uint64_t again_pages[1] = {pfn[0]};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(again_vas, 1, again_pages) == 1);
    const auto* w7 = static_cast<const uint8_t*>(arena().translate(C + 7 * k64K));
    CHECK(w7[0] == 0xA1 && w7[1] == 0x5A);
    // A pfn of 0 unmaps that slot only.
    const uint64_t mixed_vas[2] = {C + 7 * k64K, C + 1000 * k64K};
    const uint64_t mixed_pages[2] = {0, pfn[2]};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(mixed_vas, 2, mixed_pages) == 1);
    CHECK(w1000[7] == 0xC3);
    // Unknown frame numbers fail before anything changes.
    const uint64_t bad[1] = {0xDEAD};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(again_vas, 1, bad) == 0);
    // Addresses outside any reservation fail.
    const uint64_t stray[1] = {0x1234'0000};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(stray, 1, again_pages) == 0);

    Pg6XMemWorkingSetStats ws{};
    CHECK(pg6_xbox_XMemGetWorkingSetStatistics(&ws) == gdk::S_OK_HR);
    CHECK(ws.live_physical_pages == 4);
    CHECK(xmem_virtual_free(gpu) && xmem_virtual_free(cpu) && xmem_virtual_free(view));
    CHECK(pg6_xbox_XMemFreePhysicalPages(4, pfn) == 1);
    reset();
}

void test_gpu_page_table() {
    reset();
    xw::set_policy(xw::PlacementPolicy::PhantomGpuWindows);
    const uint64_t S = window_size();
    void* cpu = pg6_xbox_XMemVirtualAlloc(nullptr, S, kWindowType, 0x21u, kCpuProtect);
    void* gpu = pg6_xbox_XMemVirtualAlloc(nullptr, S, kWindowType, 0x3u, kGpuProtect);
    const uint64_t G = va_of(gpu);
    // The title's update shape: per range a GPU-window address and {kind, pool page, count}.
    std::vector<xw::PageMappingRange> ranges(2);
    ranges[0].gpu_va = G + 10 * k64K;
    ranges[0].entries = {{1, 5, 1}, {1, 9, 1}, {2, 0, 2}, {7, 0, 1}};
    ranges[1].gpu_va = 0x1000;  // not a GPU window
    ranges[1].entries = {{1, 1, 1}};
    const auto r = xw::apply_gpu_page_mappings(ranges, 0, 64);
    CHECK(r.mapped_pages == 2 && r.unmapped_pages == 2 && r.skipped_entries == 1 && r.outside_window == 1);
    CHECK(xw::gpu_page_pool_index(G + 10 * k64K + 100) == 5 && xw::gpu_page_pool_index(G + 11 * k64K) == 9);
    CHECK(xw::gpu_page_pool_index(G + 12 * k64K) == -1);
    CHECK(xw::accounting().gpu_window_mapped == 2 * k64K);
    std::vector<xw::PageMappingRange> unmap(1);
    unmap[0].gpu_va = G + 10 * k64K;
    unmap[0].entries = {{2, 0, 1}};
    xw::apply_gpu_page_mappings(unmap, 0, 64);
    CHECK(xw::gpu_page_pool_index(G + 10 * k64K) == -1 && xw::accounting().gpu_window_mapped == k64K);
    // Pool page beyond the pool is rejected.
    std::vector<xw::PageMappingRange> beyond(1);
    beyond[0].gpu_va = G;
    beyond[0].entries = {{1, 64, 1}};
    CHECK(xw::apply_gpu_page_mappings(beyond, 0, 64).outside_window == 1);
    CHECK(xmem_virtual_free(gpu) && xmem_virtual_free(cpu));
    CHECK(xw::accounting().gpu_window_mapped == 0);
    reset();
}

// O30: physical page frame numbers count 4 KiB frames. The title's streaming code keeps
// pfn >> 4 per 64 KiB pool page and maps window slots with (pfn >> 4) << 4, one scatter entry per
// 64 KiB slot; the same window pages reach the GPU window through queue slot 31 by pool page
// index. Synthetic sequence in that shape.
void test_title_shaped_streaming() {
    reset();
    xw::set_policy(xw::PlacementPolicy::PhantomGpuWindows);
    const uint64_t S = window_size();
    void* cpu = pg6_xbox_XMemVirtualAlloc(nullptr, S, kWindowType, 0x21u, kCpuProtect);
    void* gpu = pg6_xbox_XMemVirtualAlloc(nullptr, S, kWindowType, 0x3u, kGpuProtect);
    CHECK(cpu != nullptr && gpu != nullptr);
    const uint64_t C = va_of(cpu), G = va_of(gpu);

    constexpr size_t kPool = 64;
    size_t count = kPool;
    uint64_t pfn[kPool]{};
    CHECK(pg6_xbox_XMemAllocatePhysicalPages(0x380u, &count, pfn) == 1 && count == kPool);
    for (size_t i = 0; i < kPool; ++i) {
        CHECK(pfn[i] % 16 == 0);                          // a 64 KiB page is 16 frames
        CHECK(i == 0 || pfn[i] == pfn[i - 1] + 16);
    }
    void* view = pg6_xbox_XMemVirtualAlloc(nullptr, kPool * k64K, kWindowType, 0x23u, 0x2004u);
    CHECK(view != nullptr && pg6_xbox_XMemMapPhysicalPages(view, kPool, pfn) == view);
    auto* v = static_cast<uint8_t*>(view);
    for (size_t i = 0; i < kPool; ++i) {
        std::memset(v + i * k64K, static_cast<int>(0x80 + i), k64K);
    }

    // The title's packed table and scatter arrays: pool pages 8..39 at window slots 0..31.
    uint32_t packed[kPool];
    for (size_t i = 0; i < kPool; ++i) {
        packed[i] = static_cast<uint32_t>(pfn[i] >> 4);
    }
    constexpr size_t kFirst = 8, kSlots = 32;
    uint64_t vas[kSlots], pages[kSlots];
    for (size_t j = 0; j < kSlots; ++j) {
        vas[j] = C + j * k64K;
        pages[j] = static_cast<uint64_t>(packed[kFirst + j]) << 4;
    }
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(vas, kSlots, pages) == 1);
    for (size_t j = 0; j < kSlots; ++j) {  // every slot shows its own page, none repeats
        const auto* w = static_cast<const uint8_t*>(arena().translate(C + j * k64K));
        CHECK(w[0] == 0x80 + kFirst + j && w[k64K - 1] == 0x80 + kFirst + j);
    }
    // The same window pages in the GPU window (queue slot 31: pool page indices) agree.
    std::vector<xw::PageMappingRange> ranges(1);
    ranges[0].gpu_va = G;
    for (size_t j = 0; j < kSlots; ++j) {
        ranges[0].entries.push_back({1, static_cast<uint32_t>(kFirst + j), 1});
    }
    CHECK(xw::apply_gpu_page_mappings(ranges, va_of(view), kPool).mapped_pages == kSlots);
    for (size_t j = 0; j < kSlots; ++j) {
        const int64_t index = xw::gpu_page_pool_index(G + j * k64K);
        CHECK(index == static_cast<int64_t>(kFirst + j));
        const auto* w = static_cast<const uint8_t*>(arena().translate(C + j * k64K));
        CHECK(w[5] == 0x80 + index);
    }
    // Unmap four slots (null array), then remap one of their pages while it is still mapped at
    // another slot of the window.
    static_cast<uint8_t*>(arena().translate(C + 4 * k64K))[9] = 0x3C;  // written through the window
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(vas + 4, 4, nullptr) == 1);
    CHECK(arena().query(C + 4 * k64K).state == MemState::Reserve);
    // The unmapped pages' bytes went back to their pool-view slots (no store copies).
    CHECK(v[(kFirst + 4) * k64K + 9] == 0x3C && v[(kFirst + 4) * k64K + 10] == 0x80 + kFirst + 4);
    const uint64_t again_va[1] = {C + 100 * k64K};
    const uint64_t again_page[1] = {static_cast<uint64_t>(packed[kFirst + 1]) << 4};  // still at slot 1
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(again_va, 1, again_page) == 1);
    CHECK(static_cast<const uint8_t*>(arena().translate(C + 100 * k64K))[3] == 0x80 + kFirst + 1);
    // A frame number inside a page (not its first frame) is not a page.
    const uint64_t inside[1] = {pfn[3] + 1};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(again_va, 1, inside) == 0);

    // Other page sizes take their size in frames: 4 KiB pages 1, 2 MiB pages 512.
    size_t n4 = 2, n2m = 2;
    uint64_t p4[2]{}, p2m[2]{};
    CHECK(pg6_xbox_XMemAllocatePhysicalPages(kXMemFlagPage4K, &n4, p4) == 1 && p4[1] == p4[0] + 1);
    CHECK(pg6_xbox_XMemAllocatePhysicalPages(kXMemFlagPage2M, &n2m, p2m) == 1);
    CHECK(p2m[0] % 512 == 0 && p2m[1] == p2m[0] + 512);
    CHECK(pg6_xbox_XMemFreePhysicalPages(2, p4) == 1 && pg6_xbox_XMemFreePhysicalPages(2, p2m) == 1);

    CHECK(xmem_virtual_free(gpu) && xmem_virtual_free(cpu) && xmem_virtual_free(view));
    CHECK(pg6_xbox_XMemFreePhysicalPages(kPool, pfn) == 1);
    reset();
}

// Exhaustion fails cleanly: the planner reports the first reservation that does not fit, and a
// window bigger than the whole arena is refused by XMemVirtualAlloc without side effects.
void test_exhaustion() {
    reset();
    const xw::PlannedReservation seq[] = {{3 * MiB, 0x1000u | 0x2000u, kCpuProtect}, {2 * MiB, 0x3000u, kCpuProtect}};
    const auto fit = xw::plan_fit(seq, 4 * MiB, 0, xw::PlacementPolicy::Arena);
    CHECK(!fit.fits && fit.first_failure == 1 && fit.identity_bytes == 3 * MiB);
    xw::set_policy(xw::PlacementPolicy::PhantomGpuWindows);
    const uint64_t reserved0 = arena().reserved_bytes();
    const uint64_t too_big = arena().space_size() + 64 * GiB;
    CHECK(pg6_xbox_XMemVirtualAlloc(nullptr, too_big, kWindowType, 0x21u, kCpuProtect) == nullptr);
    CHECK(win32::GetLastError() == win32::ERROR_NOT_ENOUGH_MEMORY);
    CHECK(arena().reserved_bytes() == reserved0 && xw::windows().empty());
    // The GPU window of that size still has a (phantom) place.
    void* gpu = pg6_xbox_XMemVirtualAlloc(nullptr, too_big, kWindowType, 0x3u, kGpuProtect);
    CHECK(gpu != nullptr && xw::alias_pairs().empty());
    CHECK(xmem_virtual_free(gpu));
    reset();
}

// Replays the title's reservation sequence (sizes and flags from the executable, xmem_windows.h)
// against the browser's 12 GiB identity pool.
void test_browser_budget_replay() {
    const uint64_t pool = 12 * GiB;
    const uint64_t image = 0xB06D000;  // image mapping at 5 GiB (0xB06'D000 bytes)
    // Series X profile: windows of 16 GiB, 3.88 GiB pool view, fixed committed budgets.
    const xw::PlannedReservation series_x[] = {
        {16 * GiB, kWindowType, kCpuProtect},             // CPU window
        {16 * GiB, kWindowType, kGpuProtect},             // GPU window
        {0xF890'0000ull, kWindowType, 0x2004u},           // page pool view (mapped, committed)
        {0x4050'0000ull + (96 + 100 + 80 + 16 + 16 + 8 + 1) * MiB, 0x3000u, kCpuProtect},  // heap budgets
        {0xDC0'0000ull, kWindowType, kCpuProtect},        // physical-page heap
    };
    // Arena policy: the CPU window alone exceeds the pool.
    auto a = xw::plan_fit(series_x, pool, image, xw::PlacementPolicy::Arena);
    CHECK(!a.fits && a.first_failure == 0);
    // Phantom GPU windows: the GPU window costs nothing, the CPU window still does not fit.
    auto p = xw::plan_fit(series_x, pool, image, xw::PlacementPolicy::PhantomGpuWindows);
    CHECK(!p.fits && p.first_failure == 0 && p.phantom_bytes == 16 * GiB);
    // Everything except the CPU window fits with room to spare (5.4 GiB of 12).
    std::vector<xw::PlannedReservation> without_cpu(std::begin(series_x) + 1, std::end(series_x));
    auto w = xw::plan_fit(without_cpu, pool, image, xw::PlacementPolicy::PhantomGpuWindows);
    CHECK(w.fits && w.identity_bytes < 6 * GiB);
    // Even the largest possible pool (16 GiB linear memory, 1 GiB runtime) cannot hold the CPU window
    // next to the image.
    auto big = xw::plan_fit(series_x, 15 * GiB, image, xw::PlacementPolicy::PhantomGpuWindows);
    CHECK(!big.fits && big.first_failure == 0);
    // The other console profile is larger still (24 GiB windows, 6.09 GiB pool).
    const xw::PlannedReservation other[] = {
        {24 * GiB, kWindowType, kCpuProtect},
        {24 * GiB, kWindowType, kGpuProtect},
        {0x1'85C0'0000ull, kWindowType, 0x2004u},
    };
    auto o = xw::plan_fit(other, pool, image, xw::PlacementPolicy::PhantomGpuWindows);
    CHECK(!o.fits && o.phantom_bytes == 24 * GiB);
}

// On a native 64 GiB arena the real sequence is reserved as is (both policies).
void test_native_full_size_sequence() {
    if (arena().space_size() < 40 * GiB) {
        return;  // wasm test arena (4 GiB): covered by the planner
    }
    for (auto policy : {xw::PlacementPolicy::Arena, xw::PlacementPolicy::PhantomGpuWindows}) {
        reset();
        xw::set_policy(policy);
        void* cpu = pg6_xbox_XMemVirtualAlloc(nullptr, 16 * GiB, kWindowType, 0x21u, kCpuProtect);
        void* gpu = pg6_xbox_XMemVirtualAlloc(nullptr, 16 * GiB, kWindowType, 0x3u, kGpuProtect);
        CHECK(cpu != nullptr && gpu != nullptr && xw::alias_pairs().size() == 1);
        CHECK(xw::is_phantom(va_of(gpu)) == (policy == xw::PlacementPolicy::PhantomGpuWindows));
        CHECK(xmem_virtual_free(cpu) && xmem_virtual_free(gpu));
    }
    reset();
}

// O36: a physical page mapped at several addresses is ONE page (native shared views): a write
// through any address is seen through the others, at once, and unmapping one address leaves the
// others showing the page. Native process arena only (the browser aliases through the window
// table, test_guest_window).
void test_native_shared_page_aliases() {
    if (!arena().has_shared_zone()) {
        return;
    }
    reset();
    xw::set_policy(xw::PlacementPolicy::Arena);
    void* res = pg6_xbox_XMemVirtualAlloc(nullptr, 16 * k64K, kWindowType, 0x23u, kCpuProtect);
    void* other = pg6_xbox_XMemVirtualAlloc(nullptr, 8 * k64K, kWindowType, 0x23u, kCpuProtect);
    CHECK(res != nullptr && other != nullptr && arena().is_shared(va_of(res)) && arena().is_shared(va_of(other)));
    size_t count = 2;
    uint64_t pfn[2] = {};
    CHECK(pg6_xbox_XMemAllocatePhysicalPages(0x380u, &count, pfn) == 1 && count == 2);
    const uint64_t R = va_of(res), O = va_of(other);
    // One page at three addresses: slots 0 and 5 of one reservation, slot 2 of another.
    const uint64_t at[] = {R, R + 5 * k64K, O + 2 * k64K};
    const uint64_t same[] = {pfn[0], pfn[0], pfn[0]};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(at, 3, same) == 1);
    auto* a = arena().translate(R);
    auto* b = arena().translate(R + 5 * k64K);
    auto* c = arena().translate(O + 2 * k64K);
    a[100] = 0x11;
    CHECK(b[100] == 0x11 && c[100] == 0x11);
    c[k64K - 1] = 0x22;
    CHECK(a[k64K - 1] == 0x22 && b[k64K - 1] == 0x22);
    // Unmapping one address: the others keep the page; the address is reserved again.
    const uint64_t first[] = {R};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(first, 1, nullptr) == 1);
    CHECK(arena().query(R).state == MemState::Reserve);
    b[7] = 0x33;
    CHECK(c[7] == 0x33 && c[100] == 0x11);
    // Mapping it back: still the same page.
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(first, 1, &pfn[0]) == 1);
    CHECK(a[7] == 0x33 && a[100] == 0x11);
    // The last 64 KiB of a larger unmapped stretch (slots 1..4 are one piece after the maps above).
    const uint64_t tail[] = {R + 4 * k64K, O + 3 * k64K};
    const uint64_t second[] = {pfn[1], pfn[1]};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(tail, 2, second) == 1);
    arena().translate(R + 4 * k64K)[1] = 0x44;
    CHECK(arena().translate(O + 3 * k64K)[1] == 0x44);
    // Another page at an address replaces the view; the first page is unchanged elsewhere.
    const uint64_t replace[] = {R + 5 * k64K};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(replace, 1, &pfn[1]) == 1);
    CHECK(b[1] == 0x44 && a[100] == 0x11);
    CHECK(xmem_virtual_free(res) && xmem_virtual_free(other));
    CHECK(pg6_xbox_XMemFreePhysicalPages(2, pfn) == 1);
    reset();
}

}  // namespace

int main() {
    test_classification();
    test_windows_alias_pairs_and_listeners();
    test_phantom_gpu_window();
    test_pool_pages_in_windows();
    test_gpu_page_table();
    test_title_shaped_streaming();
    test_exhaustion();
    test_browser_budget_replay();
    test_native_full_size_sequence();
    test_native_shared_page_aliases();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_xmem_windows: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_xmem_windows: all tests passed\n");
    return 0;
}
