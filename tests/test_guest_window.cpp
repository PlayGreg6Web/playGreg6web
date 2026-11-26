// R19 translated CPU window (os/guest_window.h): the delta table, the shared translation,
// segmented copies, contiguity statistics, the slow-path helpers lifted code calls, and the XMem
// integration (PlacementPolicy::PhantomGpuTranslatedCpu): a 16 GiB CPU window that takes no pool
// address space, pool pages scattered across it, aliasing the pool view. Synthetic data only.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#include "guest/com_bridge.h"
#include "guest/guest_runtime.h"
#include "os/guest_window.h"
#include "os/vmem.h"
#include "os/win32_kernel.h"
#include "os/xbox_scarlett.h"
#include "os/xmem_windows.h"

extern "C" {
void* __playgta6_guest_read(uint64_t va, uint64_t n);
void* __playgta6_guest_scratch();
void __playgta6_guest_write(uint64_t va, uint64_t n);
void __playgta6_guest_memmove(uint64_t dst, uint64_t src, uint64_t n);
void __playgta6_guest_memset(uint64_t dst, uint8_t value, uint64_t n);
}

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_guest_window.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6::os;
namespace gw = playgta6::os::guest_window;
namespace xw = playgta6::os::xmem_windows;
using namespace playgta6::os::scarlett;

constexpr uint64_t GiB = 1ull << 30;
constexpr uint64_t k64K = 64ull << 10;
constexpr uint64_t W = gw::kWindowBase;
constexpr uint32_t kWindowType = 0x20402000u;  // MEM_RESERVE | MEM_64K_PAGES | MEM_PHYSICAL bit
constexpr uint32_t kCpuProtect = 0x4u;
constexpr uint32_t kGpuProtect = 0x801u;

VirtualMemoryArena& arena() {
    return *win32::get_vmem_arena();
}

struct GuestFault : std::runtime_error {
    explicit GuestFault(uint64_t va) : std::runtime_error("guest fault"), va(va) {}
    uint64_t va;
};

void throwing_handler(const Pg6GuestFaultInfo* info) {
    throw GuestFault(info->va);
}

// Eight host pages (a 512 KiB arena block) the tests map window slots onto.
struct Pages {
    uint64_t va = 0;
    uint8_t* host = nullptr;
    Pages() {
        va = arena().virtual_alloc(0, 8 * k64K, kMemReserve | kMemCommit, PageProtect::ReadWrite);
        host = va != 0 ? arena().translate(va) : nullptr;
    }
    ~Pages() {
        if (va != 0) {
            arena().release(va);
        }
    }
    uint8_t* page(int i) const { return host + i * k64K; }
};

void test_table_translation_and_copies() {
    gw::reset_for_testing();
    CHECK(gw::translate(W) == reinterpret_cast<void*>(static_cast<uintptr_t>(__playgta6_guest_base + W)));
    gw::set_enabled(true);
    CHECK(gw::enabled());
    CHECK(gw::translate(W) == nullptr);  // every slot starts unmapped
    CHECK(__playgta6_window_table[0] == static_cast<int64_t>(gw::kTrapDelta));

    Pages pages;
    CHECK(pages.host != nullptr);
    // Slots 0,1 on adjacent pages (a contiguous run), slot 2 on page 7, slot 5 on page 3.
    CHECK(gw::map_slot(W, pages.page(0)));
    CHECK(gw::map_slot(W + k64K, pages.page(1)));
    CHECK(gw::map_slot(W + 2 * k64K + 123, pages.page(7)));  // any address inside the slot
    CHECK(gw::map_slot(W + 5 * k64K, pages.page(3)));
    CHECK(!gw::map_slot(W - k64K, pages.page(0)));  // outside the window
    CHECK(gw::translate(W + 17) == pages.page(0) + 17);
    CHECK(gw::translate(W + 2 * k64K + 5) == pages.page(7) + 5);
    CHECK(gw::translate(W + 3 * k64K) == nullptr);
    CHECK(gw::slot_host(W + 5 * k64K + 9) == pages.page(3));
    // The table holds deltas: host = va + table[slot] (what lifted code computes).
    CHECK(W + 5 * k64K + 9 + static_cast<uint64_t>(__playgta6_window_table[5]) ==
          static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pages.page(3) + 9)));
    // Outside the window: the flat translation.
    CHECK(gw::translate(0x1'4000'0000ull) == arena().translate(0x1'4000'0000ull));

    // Contiguity: slots 0|1 are host-adjacent, 1|2 are not.
    CHECK(gw::contiguous(W + k64K - 8, 16) == pages.page(0) + k64K - 8);
    CHECK(gw::contiguous(2 * k64K + W - 8, 16) == nullptr);
    CHECK(gw::contiguous(W + 3 * k64K - 8, 16) == nullptr);  // runs into an unmapped slot

    // Segmented copies across non-adjacent slots.
    std::vector<uint8_t> data(3 * k64K);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(i * 13 + 7);
    }
    CHECK(gw::copy_to_guest(W, data.data(), data.size()));
    CHECK(std::memcmp(pages.page(1), data.data() + k64K, k64K) == 0);
    CHECK(std::memcmp(pages.page(7), data.data() + 2 * k64K, k64K) == 0);
    std::vector<uint8_t> back(data.size());
    CHECK(gw::copy_from_guest(back.data(), W, back.size()));
    CHECK(back == data);
    CHECK(!gw::copy_from_guest(back.data(), W + 2 * k64K, 2 * k64K));  // slot 3 unmapped
    // Overlapping move across slots (memmove semantics) and a fill.
    CHECK(gw::move_guest(W + k64K - 100, W + k64K - 300, 0x400));
    CHECK(gw::copy_from_guest(back.data(), W + k64K - 100, 0x400));
    CHECK(std::memcmp(back.data(), data.data() + k64K - 300, 0x400) == 0);
    CHECK(gw::fill_guest(W + 2 * k64K - 10, 0xEE, 20));
    CHECK(pages.page(1)[k64K - 1] == 0xEE && pages.page(7)[9] == 0xEE && pages.page(7)[10] != 0xEE);

    const gw::Stats s = gw::stats();
    CHECK(s.mapped_slots == 4);
    CHECK(s.contiguous_pairs == 1 && s.split_pairs == 1);  // 0|1 adjacent, 1|2 split (5 alone)

    // O29: the reverse translation answers with the window VA for bytes of a mapped slot's page,
    // flat for everything else; host_or_trap never turns a non-null guest pointer into null.
    CHECK(gw::guest_address(pages.page(7) + 5) == W + 2 * k64K + 5);
    CHECK(gw::guest_address(pages.page(0) + k64K - 1) == W + k64K - 1);
    CHECK(gw::guest_address(pages.page(1)) == W + k64K);
    CHECK(gw::guest_address(nullptr) == 0);
    const void* flat = arena().translate(0x1'4000'0000ull);
    CHECK(gw::guest_address(flat) == 0x1'4000'0000ull);
    CHECK(gw::guest_address(pages.page(4)) ==
          static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pages.page(4))) - __playgta6_guest_base);
    CHECK(gw::host_or_trap(0) == nullptr);
    CHECK(gw::host_or_trap(W + 2 * k64K + 5) == pages.page(7) + 5);
    CHECK(reinterpret_cast<uintptr_t>(gw::host_or_trap(W + 3 * k64K + 1)) == W + 3 * k64K + 1 + gw::kTrapDelta);
    CHECK(gw::host_or_trap(0x1'4000'0000ull) == flat);
    CHECK(gw::translated(W + 3 * k64K) && !gw::translated(W - 1));

    CHECK(gw::unmap_slot(W + 5 * k64K));
    CHECK(gw::translate(W + 5 * k64K) == nullptr);
    CHECK(gw::guest_address(pages.page(3)) != W + 5 * k64K);  // forgotten with the slot
    // Remapping a slot moves its reverse entry with it.
    CHECK(gw::map_slot(W + 2 * k64K, pages.page(6)));
    CHECK(gw::guest_address(pages.page(6) + 1) == W + 2 * k64K + 1);
    CHECK(gw::guest_address(pages.page(7) + 1) != W + 2 * k64K + 1);
    gw::reset_for_testing();
    CHECK(gw::guest_address(pages.page(6)) != W + 2 * k64K);
    CHECK(!gw::translated(W));
}

// O29: the COM bridge's guest pointer seam (guest_to_host / read_guest / write_guest /
// host_to_guest) serves window addresses through the shared translation.
void test_com_bridge_window() {
    namespace com = playgta6::guest::com;
    gw::reset_for_testing();
    gw::set_enabled(true);
    Pages pages;
    CHECK(gw::map_slot(W, pages.page(0)));
    CHECK(gw::map_slot(W + k64K, pages.page(2)));  // not host-adjacent to slot 0
    CHECK(com::guest_to_host(W + 10, 16) == pages.page(0) + 10);
    CHECK(com::guest_to_host(W + k64K - 8, 16) == nullptr);  // straddles non-adjacent pages
    CHECK(com::guest_to_host(W + 5 * k64K, 4) == nullptr);   // unmapped
    const uint8_t data[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    CHECK(com::write_guest(W + k64K - 8, data, sizeof(data)));
    CHECK(std::memcmp(pages.page(0) + k64K - 8, data, 8) == 0);
    CHECK(std::memcmp(pages.page(2), data + 8, 8) == 0);
    uint8_t back[16] = {};
    CHECK(com::read_guest(W + k64K - 8, back, sizeof(back)) && std::memcmp(back, data, 16) == 0);
    CHECK(!com::read_guest(W + 5 * k64K, back, 4));
    CHECK(com::host_to_guest(pages.page(2) + 4) == W + k64K + 4);
    CHECK(com::host_to_guest(pages.page(4)) == pages.va + 4 * k64K);
    gw::reset_for_testing();
    CHECK(com::host_to_guest(pages.page(2)) == pages.va + 2 * k64K);  // window off: the arena
}

void test_slow_path_helpers() {
    gw::reset_for_testing();
    gw::set_enabled(true);
    Pages pages;
    gw::map_slot(W, pages.page(4));
    gw::map_slot(W + k64K, pages.page(2));  // not adjacent: every straddle crosses pages
    const uint64_t at = W + k64K - 3;
    for (int i = 0; i < 8; ++i) {
        CHECK(gw::copy_to_guest(at + i, &i, 1));
    }
    const auto* read = static_cast<const uint8_t*>(__playgta6_guest_read(at, 8));
    bool same = true;
    for (int i = 0; i < 8; ++i) {
        same &= read[i] == i;
    }
    CHECK(same);
    auto* scratch = static_cast<uint8_t*>(__playgta6_guest_scratch());
    const uint64_t value = 0x1122334455667788ull;
    std::memcpy(scratch, &value, 8);
    __playgta6_guest_write(at, 8);
    uint64_t back = 0;
    CHECK(gw::copy_from_guest(&back, at, 8) && back == value);
    CHECK(pages.page(4)[k64K - 1] == 0x66 && pages.page(2)[0] == 0x55);
    __playgta6_guest_memset(W + 100, 0xAB, 2 * k64K - 200);
    CHECK(pages.page(4)[100] == 0xAB && pages.page(2)[k64K - 101] == 0xAB && pages.page(2)[k64K - 100] != 0xAB);
    __playgta6_guest_memmove(W + 10, W + k64K + 10, 64);
    CHECK(pages.page(4)[10] == 0xAB);

    // An unmapped slot raises a guest fault at the faulting address. The helpers are extern "C";
    // MSVC's /EHsc assumes such calls never throw, so they are called through pointers whose
    // targets the compiler cannot see (the runtime calls them from lifted code the same way).
    playgta6::guest::set_fault_handler(throwing_handler);
    void* (*volatile read_fn)(uint64_t, uint64_t) = __playgta6_guest_read;
    void (*volatile memset_fn)(uint64_t, uint8_t, uint64_t) = __playgta6_guest_memset;
    uint64_t faulted = 0;
    try {
        read_fn(W + 2 * k64K - 4, 8);
    } catch (const GuestFault& f) {
        faulted = f.va;
    }
    CHECK(faulted == W + 2 * k64K - 4);
    faulted = 0;
    try {
        memset_fn(W + 5 * k64K, 0, 16);
    } catch (const GuestFault& f) {
        faulted = f.va;
    }
    CHECK(faulted == W + 5 * k64K);
    playgta6::guest::set_fault_handler(nullptr);
    gw::reset_for_testing();
}

uint64_t va_of(const void* p) {
    return arena().guest_address(p);
}

void test_xmem_translated_cpu_window() {
    xmem_reset_for_testing();
    gw::reset_for_testing();
    gw::set_enabled(true);
    xw::set_policy(xw::PlacementPolicy::PhantomGpuTranslatedCpu);
    const uint64_t reserved0 = arena().reserved_bytes();

    // The title's sequence: CPU window, GPU window (16 GiB each), a pool of physical pages mapped
    // once into its own view.
    void* cpu = pg6_xbox_XMemVirtualAlloc(nullptr, 16 * GiB, kWindowType, 0x21u, kCpuProtect);
    void* gpu = pg6_xbox_XMemVirtualAlloc(nullptr, 16 * GiB, kWindowType, 0x3u, kGpuProtect);
    CHECK(cpu != nullptr && gpu != nullptr);
    CHECK(va_of(cpu) == W);
    CHECK(xw::is_phantom(va_of(gpu), 16 * GiB));
    CHECK(arena().reserved_bytes() == reserved0);  // neither window takes pool address space
    const auto pairs = xw::alias_pairs();
    CHECK(pairs.size() == 1 && pairs[0].cpu_base == W);

    constexpr size_t kPool = 16;
    size_t count = kPool;
    uint64_t pfn[kPool] = {};
    CHECK(pg6_xbox_XMemAllocatePhysicalPages(0x380u, &count, pfn) == 1 && count == kPool);
    void* view = pg6_xbox_XMemVirtualAlloc(nullptr, kPool * k64K, kWindowType & ~0x400000u, 0x23u, kCpuProtect);
    CHECK(view != nullptr);
    CHECK(pg6_xbox_XMemMapPhysicalPages(view, kPool, pfn) == view);
    auto* pool = static_cast<uint8_t*>(view);  // host pointer of the view (identity or offset)
    for (size_t p = 0; p < kPool; ++p) {
        std::memset(pool + p * k64K, static_cast<int>(0x10 + p), k64K);
    }

    // Scatter pool pages across the window, far apart and out of order, plus one run of three
    // consecutive pool pages at consecutive slots (the contiguous case).
    const uint64_t slots[] = {W + 3 * GiB, W + 15 * GiB + 7 * k64K, W, W + k64K, W + 2 * k64K, W + 9 * GiB};
    const uint64_t pages[] = {pfn[5], pfn[0], pfn[8], pfn[9], pfn[10], pfn[15]};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(slots, 6, pages) == 1);
    // Reads through the window see the pool page's bytes (one copy, aliased).
    uint8_t b = 0;
    CHECK(gw::copy_from_guest(&b, W + 3 * GiB + 77, 1) && b == 0x15);
    CHECK(gw::copy_from_guest(&b, W + 15 * GiB + 7 * k64K, 1) && b == 0x10);
    CHECK(gw::copy_from_guest(&b, W + 9 * GiB + k64K - 1, 1) && b == 0x1F);
    // Writes through the window land in the pool view, and the other way round.
    const uint32_t marker = 0xC0FFEE42u;
    CHECK(gw::copy_to_guest(W + 3 * GiB + 0x100, &marker, 4));
    uint32_t seen = 0;
    std::memcpy(&seen, pool + 5 * k64K + 0x100, 4);
    CHECK(seen == marker);
    pool[0x200] = 0x99;
    CHECK(gw::copy_from_guest(&b, W + 15 * GiB + 7 * k64K + 0x200, 1) && b == 0x99);
    // The consecutive run is host-contiguous; the scattered ones are not.
    CHECK(gw::contiguous(W + k64K - 16, 32) != nullptr);
    CHECK(gw::contiguous(W + 3 * k64K - 16, 32) == nullptr);
    const gw::Stats s = gw::stats();
    CHECK(s.mapped_slots == 6 && s.contiguous_pairs == 2 && s.split_pairs == 0);
    CHECK(xw::accounting().cpu_window_mapped == 6 * k64K);
    // O31: the Win32 virtual-memory shims see the translated window as XMem's reservation: mapped
    // slots committed (the run W .. W + 3 slots), others reserved; no arena operation touches it.
    {
        win32::MemoryBasicInformation mbi{};
        CHECK(win32::VirtualQuery(W + k64K + 5, &mbi, sizeof(mbi)) == sizeof(mbi));
        CHECK(mbi.state == kMemCommit && mbi.allocation_base == W && mbi.base_address == W + k64K);
        CHECK(mbi.region_size == 2 * k64K && mbi.protect == kCpuProtect);
        CHECK(win32::VirtualQuery(W + 4 * k64K, &mbi, sizeof(mbi)) == sizeof(mbi));
        CHECK(mbi.state == kMemReserve && mbi.protect == static_cast<uint32_t>(PageProtect::NoAccess));
        CHECK(!win32::VirtualFree(W, 0, win32::MEM_RELEASE));
        CHECK(!win32::VirtualProtect(W, k64K, kCpuProtect, nullptr));
        CHECK(win32::VirtualAlloc(W + 4 * k64K, k64K, win32::MEM_COMMIT, kCpuProtect) == 0);
    }
    // O29: XMem's view of the same mapping sequence (what the boot report records).
    {
        const XMemWindowContiguity c = xmem_window_contiguity();
        CHECK(c.maps == 6 && c.neighbour_pairs == 2 && c.neighbour_pairs_contiguous == 2);
        const XMemWindowContiguity::Region* cpu_region = nullptr;
        for (const auto& r : c.regions) {
            cpu_region = r.translated ? &r : cpu_region;
        }
        CHECK(cpu_region != nullptr);
        if (cpu_region != nullptr) {
            CHECK(cpu_region->base == W && cpu_region->mapped_slots == 6 && cpu_region->aliased_slots == 6);
            CHECK(cpu_region->runs == 4 && cpu_region->longest_run == 3);
            CHECK(cpu_region->adjacent_pairs == 2 && cpu_region->aliased_pairs == 2 &&
                  cpu_region->contiguous_pairs == 2);
        }
    }

    // Unmap one slot (it traps afterwards), remap another slot to a different page.
    const uint64_t unmap[] = {W + 9 * GiB};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(unmap, 1, nullptr) == 1);
    CHECK(gw::translate(W + 9 * GiB) == nullptr);
    const uint64_t remap[] = {W + 3 * GiB};
    const uint64_t remap_page[] = {pfn[1]};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(remap, 1, remap_page) == 1);
    CHECK(gw::copy_from_guest(&b, W + 3 * GiB + 5, 1) && b == 0x11);
    CHECK(xw::accounting().cpu_window_mapped == 5 * k64K);

    // A page mapped into the window without a pool-view home gets a zeroed store as backing.
    size_t one = 1;
    uint64_t lone[1] = {};
    CHECK(pg6_xbox_XMemAllocatePhysicalPages(0x380u, &one, lone) == 1);
    const uint64_t lone_slot[] = {W + 12 * GiB};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(lone_slot, 1, lone) == 1);
    CHECK(gw::copy_from_guest(&b, W + 12 * GiB + 9, 1) && b == 0);
    CHECK(gw::copy_to_guest(W + 12 * GiB + 9, &marker, 1));

    // O29: window addresses as the windowed import thunks hand them over (the slot's backing page,
    // or va + kTrapDelta for an unmapped slot) name the window VA.
    const uint64_t two_pages[] = {pfn[2], pfn[3]};
    void* mapped = pg6_xbox_XMemMapPhysicalPages(gw::host_or_trap(W + 7 * GiB), 2, two_pages);
    CHECK(mapped != nullptr && gw::guest_address(mapped) == W + 7 * GiB);
    CHECK(gw::translate(W + 7 * GiB + 3) == pool + 2 * k64K + 3);
    CHECK(gw::translate(W + 7 * GiB + k64K) == pool + 3 * k64K);
    CHECK(pg6_xbox_XMemMapPhysicalPages(gw::host_or_trap(W + 7 * GiB), 2, nullptr) != nullptr);  // unmap
    CHECK(gw::translate(W + 7 * GiB) == nullptr);

    CHECK(xmem_virtual_free(gw::host_or_trap(W)));  // slot 0 is mapped: its backing page arrives
    CHECK(gw::translate(W + 3 * GiB) == nullptr);
    CHECK(gw::stats().mapped_slots == 0);
    xmem_reset_for_testing();
    gw::reset_for_testing();
}

uint64_t g_violation = 0;
void record_violation(uint64_t window_va) {
    g_violation = window_va;
}

// O29: window-centric placement. A page mapped next to a mapped window slot whose bytes are not
// its neighbours' moves there (a frame swap with the page living at the wanted position); what
// cannot be kept contiguous is counted and, with contiguous runs required, reported.
void test_xmem_window_run_keeper() {
    xmem_reset_for_testing();
    gw::reset_for_testing();
    gw::set_enabled(true);
    xw::set_policy(xw::PlacementPolicy::PhantomGpuTranslatedCpu);
    void* cpu = pg6_xbox_XMemVirtualAlloc(nullptr, 16 * GiB, kWindowType, 0x21u, kCpuProtect);
    CHECK(cpu != nullptr && va_of(cpu) == W);
    constexpr size_t kPool = 16;
    size_t count = kPool;
    uint64_t pfn[kPool] = {};
    CHECK(pg6_xbox_XMemAllocatePhysicalPages(0x380u, &count, pfn) == 1 && count == kPool);
    void* view = pg6_xbox_XMemVirtualAlloc(nullptr, kPool * k64K, kWindowType & ~0x400000u, 0x23u, kCpuProtect);
    CHECK(view != nullptr && pg6_xbox_XMemMapPhysicalPages(view, kPool, pfn) == view);
    auto* pool = static_cast<uint8_t*>(view);
    for (size_t p = 0; p < kPool; ++p) {
        std::memset(pool + p * k64K, static_cast<int>(0x40 + p), k64K);
    }
    const auto byte_at = [](uint64_t va) {
        uint8_t b = 0;
        CHECK(gw::copy_from_guest(&b, va, 1));
        return b;
    };

    // The keeper works while contiguous runs are required (a violation handler that only records).
    gw::set_contiguous_runs_required(true, &record_violation);
    g_violation = 0;
    // Slot 0 <- page 5, slot 1 <- page 9: page 9's bytes swap frames with page 6 (in no window run).
    const uint64_t a[] = {W, W + k64K};
    const uint64_t pa[] = {pfn[5], pfn[9]};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(a, 2, pa) == 1);
    CHECK(gw::contiguous(W + k64K - 8, 16) != nullptr);
    CHECK(byte_at(W) == 0x45 && byte_at(W + k64K) == 0x49 && byte_at(W + 2 * k64K - 1) == 0x49);
    // Page 6 kept its bytes (now in page 9's old frame): visible when it is mapped somewhere.
    const uint64_t b[] = {W + 8 * k64K};
    const uint64_t pb[] = {pfn[6]};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(b, 1, pb) == 1);
    CHECK(byte_at(W + 8 * k64K + 77) == 0x46);
    // Mapping to the left of a run works too: slot 7 <- page 2 moves next to page 6's frame.
    const uint64_t c[] = {W + 7 * k64K};
    const uint64_t pc[] = {pfn[2]};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(c, 1, pc) == 1);
    CHECK(gw::contiguous(W + 8 * k64K - 8, 16) != nullptr && byte_at(W + 7 * k64K) == 0x42);
    XMemWindowContiguity s = xmem_window_contiguity();
    CHECK(s.run_moves == 2 && s.run_breaks == 0 && s.run_bytes_moved == 4 * k64K);

    CHECK(g_violation == 0);
    // The same page at two adjacent slots (one copy of its bytes cannot sit at both): a break,
    // reported through guest_window.
    const uint64_t d[] = {W + 2 * k64K};
    const uint64_t pd[] = {pfn[9]};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(d, 1, pd) == 1);
    CHECK(g_violation == W + 2 * k64K);
    s = xmem_window_contiguity();
    CHECK(s.run_breaks == 1);
    CHECK(gw::contiguous(W + 2 * k64K - 8, 16) == nullptr);
    gw::set_contiguous_runs_required(false);
    CHECK(!gw::contiguous_runs_required());

    // Unmapping the pool view slot whose frame now holds page 9's bytes keeps them (store).
    CHECK(pg6_xbox_XMemMapPhysicalPages(static_cast<uint8_t*>(view) + 6 * k64K, 1, nullptr) != nullptr);
    CHECK(byte_at(W + k64K + 5) == 0x49);

    CHECK(xmem_virtual_free(cpu));
    xmem_reset_for_testing();
    gw::reset_for_testing();
}

// O31: the keeper only moves bytes within the reservation where the page was first mapped (the
// pool view). The windowed boot of the title faulted when it swapped a page the title uses in an
// XMem heap mapped right after the pool view.
void test_xmem_run_keeper_stays_in_the_pool() {
    xmem_reset_for_testing();
    gw::reset_for_testing();
    gw::set_enabled(true);
    xw::set_policy(xw::PlacementPolicy::PhantomGpuTranslatedCpu);
    gw::set_contiguous_runs_required(true, &record_violation);
    g_violation = 0;
    void* cpu = pg6_xbox_XMemVirtualAlloc(nullptr, 16 * GiB, kWindowType, 0x21u, kCpuProtect);
    CHECK(cpu != nullptr && va_of(cpu) == W);
    size_t count = 5;
    uint64_t pfn[5] = {};
    CHECK(pg6_xbox_XMemAllocatePhysicalPages(0x380u, &count, pfn) == 1 && count == 5);
    // Pool view of four pages, and right after it a heap region holding the fifth page.
    void* view = pg6_xbox_XMemVirtualAlloc(nullptr, 5 * k64K, kWindowType & ~0x400000u, 0x23u, kCpuProtect);
    CHECK(view != nullptr);
    CHECK(xmem_virtual_free(view));  // only to learn a free range of five pages
    void* pool = pg6_xbox_XMemVirtualAlloc(view, 4 * k64K, kWindowType & ~0x400000u, 0x23u, kCpuProtect);
    void* heap = pg6_xbox_XMemVirtualAlloc(static_cast<uint8_t*>(view) + 4 * k64K, k64K,
                                           kWindowType & ~0x400000u, 0x23u, kCpuProtect);
    CHECK(pool != nullptr && heap != nullptr);
    CHECK(pg6_xbox_XMemMapPhysicalPages(pool, 4, pfn) == pool);
    CHECK(pg6_xbox_XMemMapPhysicalPages(heap, 1, pfn + 4) == heap);
    for (int i = 0; i < 4; ++i) {
        std::memset(static_cast<uint8_t*>(pool) + i * k64K, 0x10 + i, k64K);
    }
    std::memset(heap, 0xEE, k64K);
    // Slot 0 <- pool page 3 (the pool's last frame), slot 1 <- pool page 0: the frame after page 3
    // is the heap's page, which must not move; the run moves to two free pool frames instead.
    const uint64_t vas[] = {W, W + k64K};
    const uint64_t pages[] = {pfn[3], pfn[0]};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(vas, 2, pages) == 1);
    CHECK(static_cast<uint8_t*>(heap)[0] == 0xEE && static_cast<uint8_t*>(heap)[k64K - 1] == 0xEE);
    const XMemWindowContiguity c = xmem_window_contiguity();
    CHECK(c.run_breaks == 0 && c.run_relocations == 1 && g_violation == 0);
    const uint8_t* run = static_cast<const uint8_t*>(gw::contiguous(W, 2 * k64K));
    CHECK(run != nullptr && run[0] == 0x13 && run[k64K - 1] == 0x13 && run[k64K] == 0x10);
    gw::set_contiguous_runs_required(false);
    CHECK(xmem_virtual_free(cpu));
    xmem_reset_for_testing();
    gw::reset_for_testing();
}

// O37: write marks. Host writers mark the slots they write (after the write), remapping marks,
// take_written reports and clears, read-only uses never mark, and Buffers marks in-place writable
// views only when the host call is done.
std::vector<uint64_t> taken(uint64_t va, size_t len) {
    std::vector<uint64_t> out;
    CHECK(gw::take_written(va, len, &out));
    return out;
}

void test_write_marks() {
    gw::reset_for_testing();
    std::vector<uint64_t> none;
    CHECK(!gw::take_written(W, k64K, &none));  // window off: no marks
    gw::set_enabled(true);
    Pages pages;
    CHECK(gw::map_slot(W, pages.page(0)));
    CHECK(gw::map_slot(W + k64K, pages.page(1)));
    CHECK(gw::map_slot(W + 2 * k64K, pages.page(5)));
    // Mapping marks: the slot's bytes changed.
    CHECK((taken(W, 3 * k64K) == std::vector<uint64_t>{0, k64K, 2 * k64K}));
    CHECK(taken(W, 3 * k64K).empty());  // taking clears
    CHECK(!gw::take_written(W - k64K, 16, &none));  // outside the window: untracked

    // Segmented copy over a slot boundary marks both slots; offsets relative to the range.
    uint8_t data[64] = {1, 2, 3};
    CHECK(gw::copy_to_guest(W + k64K - 32, data, sizeof(data)));
    CHECK((taken(W + 100, 3 * k64K - 100) == std::vector<uint64_t>{0, k64K - 100}));
    // Reads mark nothing.
    uint8_t back[64] = {};
    CHECK(gw::copy_from_guest(back, W + 10, sizeof(back)));
    CHECK(gw::contiguous(W, 2 * k64K) != nullptr);
    CHECK(taken(W, 3 * k64K).empty());
    // fill and move mark their destination only.
    CHECK(gw::fill_guest(W + 2 * k64K + 5, 0x11, 10));
    CHECK((taken(W, 3 * k64K) == std::vector<uint64_t>{2 * k64K}));
    CHECK(gw::move_guest(W + 8, W + 2 * k64K, 16));
    CHECK((taken(W, 3 * k64K) == std::vector<uint64_t>{0}));
    gw::mark_written(W + 3 * k64K - 1, 2);  // past the mapped slots: still a slot mark
    CHECK((taken(W, 4 * k64K) == std::vector<uint64_t>{2 * k64K, 3 * k64K}));

    // Buffers: a writable in-place view is marked by finish(), not when it is handed out; a
    // read-only one is never marked; a writable bounce copy is marked when it is copied back.
    {
        gw::Buffers b;
        CHECK(b.range(W + 16, 64, true) == pages.page(0) + 16);
        CHECK(b.range(W + k64K + 16, 64, false) == pages.page(1) + 16);
        CHECK(taken(W, 3 * k64K).empty());
        b.finish();
        CHECK((taken(W, 3 * k64K) == std::vector<uint64_t>{0}));
    }
    {
        gw::Buffers b;
        void* bounce = b.range(W + 2 * k64K - 8, 16, true);  // slots 1 | 2 are not host-adjacent
        CHECK(bounce != nullptr && b.bounce_count() == 1);
        std::memset(bounce, 0x22, 16);
    }
    CHECK((taken(W, 3 * k64K) == std::vector<uint64_t>{k64K, 2 * k64K}));
    CHECK(pages.page(5)[7] == 0x22);

    // The slow-path store helper lifted code calls for straddling stores marks too.
    std::memset(__playgta6_guest_scratch(), 0x33, 8);
    __playgta6_guest_write(W + k64K - 4, 8);
    CHECK((taken(W, 3 * k64K) == std::vector<uint64_t>{0, k64K}));

    // write_marked peeks at a mark without taking it.
    gw::mark_written(W + 5, 1);
    CHECK(gw::write_marked(W + 0xFFFF) && !gw::write_marked(W + k64K));
    CHECK((taken(W, k64K) == std::vector<uint64_t>{0}));
    CHECK(!gw::write_marked(W));

    // O36: a mark pass takes a slot's mark once and reports it to every reader in the pass.
    {
        gw::MarkPass pass;
        pass.begin();
        gw::mark_written(W + k64K + 8, 4);
        std::vector<uint64_t> a, b, c;
        CHECK(pass.take(W, 2 * k64K, &a) && (a == std::vector<uint64_t>{k64K}));
        CHECK(pass.take(W + k64K + 0x100, 16, &b) && (b == std::vector<uint64_t>{0}));  // same slot, same pass
        CHECK(!gw::write_marked(W + k64K));  // taken once, from the table
        pass.begin();
        CHECK(pass.take(W, 2 * k64K, &c) && c.empty());  // a new pass: nothing new
        gw::mark_written(W + 5, 1);
        c.clear();
        CHECK(pass.take(W, 2 * k64K, &c) && c.empty());  // slot 0 was taken in this pass already
        pass.begin();
        c.clear();
        CHECK(pass.take(W, 2 * k64K, &c) && (c == std::vector<uint64_t>{0}));  // the next pass sees it
        CHECK(!pass.take(0x1000, 16, &c));  // not translated
    }

    // O36: one scan takes every dirty slot; a slot index visits only the buffers overlapping them.
    {
        gw::MarkPass pass;
        pass.begin();
        std::vector<uint64_t> drain;
        CHECK(pass.take_dirty(&drain));  // clear whatever earlier checks left marked
        pass.begin();
        gw::SlotIndex index;
        std::vector<int> keys(3000);
        for (int i = 0; i < 3000; ++i) {
            index.add(&keys[i], W + static_cast<uint64_t>(10 + i) * k64K, k64K);  // one slot each
        }
        index.add(&keys[0], W + 10 * k64K, 2 * k64K);  // keys[0] now spans slots 10 and 11 (replaces)
        CHECK(index.size() == 3000);
        gw::mark_written(W + 1500 * k64K + 5, 1);
        gw::mark_written(W + 11 * k64K, 1);  // shared slot: keys[0] and keys[1]
        std::vector<uint64_t> dirty;
        CHECK(pass.take_dirty(&dirty) && dirty.size() == 2);
        std::vector<const void*> visit;
        index.collect(dirty, &visit);
        CHECK(visit.size() == 3);
        CHECK(std::find(visit.begin(), visit.end(), &keys[0]) != visit.end());
        CHECK(std::find(visit.begin(), visit.end(), &keys[1]) != visit.end());
        CHECK(std::find(visit.begin(), visit.end(), &keys[1490]) != visit.end());
        // The pass answers later takes from the scan: marked slots only, none re-taken.
        std::vector<uint64_t> r;
        CHECK(pass.take(W + 1490 * k64K, 12 * k64K, &r) && (r == std::vector<uint64_t>{10 * k64K}));
        index.remove(&keys[0]);
        visit.clear();
        index.collect(dirty, &visit);
        CHECK(visit.size() == 2);
        pass.begin();
        dirty.clear();
        CHECK(pass.take_dirty(&dirty) && dirty.empty());
    }

    // Verification counters.
    CHECK(!gw::write_marks_verified() && gw::verify_misses() == 0);
    gw::set_write_marks_verified(true);
    gw::note_verify_miss(W);
    CHECK(gw::write_marks_verified() && gw::verify_misses() == 1);
    gw::reset_for_testing();
    CHECK(!gw::write_marks_verified() && gw::verify_misses() == 0);
}

// O37: under the window memory model the title's physical-page reservations (and placeholders) are
// handed out at window ranges above the CPU window, backed slot for slot by their arena
// reservation, so stores into them carry write marks; XMem itself keeps working on the arena VA.
void test_xmem_physical_reservations_in_the_window() {
    xmem_reset_for_testing();
    gw::reset_for_testing();
    gw::set_enabled(true);
    xw::set_policy(xw::PlacementPolicy::PhantomGpuTranslatedCpu);
    constexpr uint32_t kPhysicalReserve = 0x20402000u;  // MEM_RESERVE | MEM_PHYSICAL | large pages
    constexpr uint32_t kPlaceholder = 0x42000u;         // MEM_RESERVE | MEM_RESERVE_PLACEHOLDER
    // Before the CPU window exists (or with the window off) nothing moves.
    void* early = pg6_xbox_XMemVirtualAlloc(nullptr, 4 * k64K, kPhysicalReserve, 0x23u, kCpuProtect);
    CHECK(early != nullptr && !gw::in_window(va_of(early)));
    void* cpu = pg6_xbox_XMemVirtualAlloc(nullptr, 16 * GiB, kWindowType, 0x21u, kCpuProtect);
    CHECK(cpu != nullptr && va_of(cpu) == W);

    // A physical reservation: a window VA in the runtime-owned part of the window.
    void* heap = pg6_xbox_XMemVirtualAlloc(nullptr, 4 * k64K, kPhysicalReserve, 0x23u, kCpuProtect);
    const uint64_t H = va_of(heap);
    CHECK(H >= gw::kRuntimeRangeBase && gw::in_window(H));
    CHECK(gw::runtime_range_host(H + 5) != nullptr);
    std::vector<uint64_t> none;
    CHECK(gw::take_written(H, 4 * k64K, &none));  // clear the marks of the range's mapping
    // Mapping pages through the window VA (direct caller form and the thunk's host-page form).
    size_t count = 3;
    uint64_t pfn[3] = {};
    CHECK(pg6_xbox_XMemAllocatePhysicalPages(0x380u, &count, pfn) == 1 && count == 3);
    CHECK(pg6_xbox_XMemMapPhysicalPages(heap, 2, pfn) == heap);
    std::vector<uint64_t> marked;
    CHECK(gw::take_written(H, 4 * k64K, &marked));
    CHECK((marked == std::vector<uint64_t>{0, k64K}));  // mapping marks the slots
    // The title writes through the window (as lifted code does) and reads its bytes back; the
    // bytes are the region's arena memory, contiguous over the run.
    const uint32_t marker = 0x5EED1234u;
    CHECK(gw::copy_to_guest(H + k64K - 2, &marker, 4));  // straddles slots 0 | 1, both marked
    uint32_t back = 0;
    CHECK(gw::copy_from_guest(&back, H + k64K - 2, 4) && back == marker);
    CHECK(gw::contiguous(H, 2 * k64K) != nullptr);
    marked.clear();
    CHECK(gw::take_written(H, 4 * k64K, &marked) && (marked == std::vector<uint64_t>{0, k64K}));
    // Scatter with window VAs, unmap and remap: every change of a slot's bytes marks it.
    const uint64_t at[] = {H + 2 * k64K};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(at, 1, &pfn[2]) == 1);
    const uint64_t zero[] = {0};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(at, 1, zero) == 1);  // unmap
    marked.clear();
    CHECK(gw::take_written(H, 4 * k64K, &marked) && (marked == std::vector<uint64_t>{2 * k64K}));
    const uint64_t remap[] = {H};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(remap, 1, &pfn[2]) == 1);  // other bytes at slot 0
    marked.clear();
    CHECK(gw::take_written(H, 4 * k64K, &marked) && (marked == std::vector<uint64_t>{0}));
    // VirtualQuery reads the reservation at the title's addresses.
    playgta6::os::win32::MemoryBasicInformation mbi{};
    CHECK(playgta6::os::win32::VirtualQuery(H + 0x100, &mbi, sizeof(mbi)) == sizeof(mbi));
    CHECK(mbi.allocation_base == H);

    // A placeholder: split with VirtualFree at window VAs and replaced piece by piece, as the
    // title does; the pieces keep the window VAs.
    void* ph = pg6_xbox_XMemVirtualAlloc(nullptr, 8 * k64K, kPlaceholder, 0x7u, 0x1u);
    const uint64_t P = va_of(ph);
    CHECK(P >= gw::kRuntimeRangeBase && P >= H + 4 * k64K);
    CHECK(playgta6::os::win32::VirtualFree(P, 2 * k64K, 0x8002u));  // MEM_RELEASE | PRESERVE_PLACEHOLDER
    void* piece = pg6_xbox_XMemVirtualAlloc(ph, 2 * k64K, 0x20406000u, 0x27u, kCpuProtect);
    CHECK(piece != nullptr && va_of(piece) == P);
    CHECK(pg6_xbox_XMemMapPhysicalPages(piece, 1, &pfn[0]) == piece);
    CHECK(gw::copy_from_guest(&back, P + k64K - 2, 2));  // pfn[0] holds the marker's low half

    // A reservation large enough to be a window (>= 1 GiB) is not one of these: it is placed as a
    // window (here, with the CPU window taken, in the arena), as before.
    void* second_window = pg6_xbox_XMemVirtualAlloc(nullptr, 2 * GiB, kPhysicalReserve, 0x23u, kCpuProtect);
    CHECK(second_window != nullptr && !gw::in_window(va_of(second_window)));
    // O36: that first one is the page pool's view; a later CPU window (the title places GPU
    // buffers in it) is handed out at a window range, and mapping into it marks.
    void* third_window = pg6_xbox_XMemVirtualAlloc(nullptr, GiB, kPhysicalReserve, 0x23u, kCpuProtect);
    const uint64_t T = va_of(third_window);
    CHECK(third_window != nullptr && T >= gw::kRuntimeRangeBase && gw::in_window(T));
    CHECK(gw::take_written(T, 2 * k64K, &none));
    CHECK(pg6_xbox_XMemMapPhysicalPages(third_window, 1, &pfn[1]) == third_window);
    marked.clear();
    CHECK(gw::take_written(T, 2 * k64K, &marked) && (marked == std::vector<uint64_t>{0}));
    CHECK(xmem_virtual_free(third_window));
    CHECK(gw::runtime_range_host(T) == nullptr);
    // Exhaustion: what does not fit in the 8 GiB above the CPU window stays in the arena.
    void* big = pg6_xbox_XMemVirtualAlloc(nullptr, 7 * GiB, kPlaceholder, 0x7u, 0x1u);
    CHECK(big != nullptr && gw::in_window(va_of(big)));
    void* over = pg6_xbox_XMemVirtualAlloc(nullptr, 2 * GiB, kPlaceholder, 0x7u, 0x1u);
    CHECK(over != nullptr && !gw::in_window(va_of(over)));

    CHECK(xmem_virtual_free(second_window));
    CHECK(xmem_virtual_free(heap));
    CHECK(xmem_virtual_free(early));
    CHECK(xmem_virtual_free(cpu));
    xmem_reset_for_testing();
    gw::reset_for_testing();
}

// O36: the title's committed XMemVirtualAlloc regions (it sub-allocates GPU buffers from them) are
// handed out at window ranges too; reserve-then-commit keeps the window form, and freeing a region
// gives its window range back. Without the window model nothing moves.
void test_xmem_committed_regions_in_the_window() {
    constexpr uint32_t kCommitLarge = 0x20003000u;  // MEM_RESERVE | MEM_COMMIT | large pages
    constexpr uint32_t kReserve = 0x2000u;
    constexpr uint32_t kCommit = 0x1000u;
    for (const bool windowed : {false, true}) {
        xmem_reset_for_testing();
        gw::reset_for_testing();
        gw::set_enabled(windowed);
        xw::set_policy(xw::PlacementPolicy::PhantomGpuTranslatedCpu);
        void* cpu = pg6_xbox_XMemVirtualAlloc(nullptr, 16 * GiB, kWindowType, 0x21u, kCpuProtect);
        CHECK(cpu != nullptr && gw::in_window(va_of(cpu)) == windowed);
        void* heap = pg6_xbox_XMemVirtualAlloc(nullptr, 5 * k64K, kCommitLarge, 0x183u, kCpuProtect);
        const uint64_t H = va_of(heap);
        CHECK(heap != nullptr && gw::in_window(H) == windowed);
        if (!windowed) {
            CHECK(xmem_virtual_free(heap));
            CHECK(xmem_virtual_free(cpu));
            continue;
        }
        CHECK(H >= gw::kRuntimeRangeBase);
        // A store into the region (as lifted code makes it) lands in its memory and marks its slot.
        std::vector<uint64_t> marked;
        CHECK(gw::take_written(H, 5 * k64K, &marked));
        const uint64_t v = 0x0123456789ABCDEFull;
        CHECK(gw::copy_to_guest(H + 3 * k64K + 8, &v, sizeof(v)));
        marked.clear();
        CHECK(gw::take_written(H, 5 * k64K, &marked) && (marked == std::vector<uint64_t>{3 * k64K}));
        uint64_t back = 0;
        CHECK(gw::copy_from_guest(&back, H + 3 * k64K + 8, sizeof(back)) && back == v);
        CHECK(gw::contiguous(H, 5 * k64K) != nullptr);
        // A plain reservation (as a pool view is) stays in the arena.
        void* plain = pg6_xbox_XMemVirtualAlloc(nullptr, 2 * k64K, kReserve, 0x3u, kCpuProtect);
        CHECK(plain != nullptr && !gw::in_window(va_of(plain)));
        CHECK(xmem_virtual_free(plain));
        // A physical reservation committed in place at its window VA: the commit answers with it.
        void* res = pg6_xbox_XMemVirtualAlloc(nullptr, 2 * k64K, 0x20402000u, 0x3u, kCpuProtect);
        const uint64_t R = va_of(res);
        CHECK(res != nullptr && R >= H + 5 * k64K);
        void* committed = pg6_xbox_XMemVirtualAlloc(res, 2 * k64K, kCommit, 0x3u, kCpuProtect);
        CHECK(committed == res);
        CHECK(gw::copy_to_guest(R + k64K, &v, sizeof(v)));
        // A region whose size is not a whole number of slots still gives its range back when freed.
        void* odd = pg6_xbox_XMemVirtualAlloc(nullptr, 3 * 4096, kReserve | kCommit, 0x3u, kCpuProtect);
        const uint64_t O = va_of(odd);
        CHECK(odd != nullptr && gw::in_window(O));
        CHECK(xmem_virtual_free(odd));
        CHECK(gw::runtime_range_host(O) == nullptr);
        CHECK(xmem_virtual_free(res));
        CHECK(xmem_virtual_free(heap));
        CHECK(gw::runtime_range_host(H) == nullptr);
        void* again = pg6_xbox_XMemVirtualAlloc(nullptr, 5 * k64K, kCommitLarge, 0x183u, kCpuProtect);
        CHECK(va_of(again) == H);  // the freed range is handed out again
        CHECK(xmem_virtual_free(again));
        CHECK(xmem_virtual_free(cpu));
    }
    xmem_reset_for_testing();
    gw::reset_for_testing();
}

// O36: a physical page the title maps at two addresses of a mirrored reservation (and one it maps
// in the pool view and again in a mirrored window) is ONE page: a write through either address is
// seen through the other, write marks of either address reach readers of both, and unmapping the
// page's first address keeps the other showing its bytes.
void test_xmem_aliases_in_the_window() {
    xmem_reset_for_testing();
    gw::reset_for_testing();
    gw::set_enabled(true);
    xw::set_policy(xw::PlacementPolicy::PhantomGpuTranslatedCpu);
    void* cpu = pg6_xbox_XMemVirtualAlloc(nullptr, 16 * GiB, kWindowType, 0x21u, kCpuProtect);
    void* pool = pg6_xbox_XMemVirtualAlloc(nullptr, 2 * GiB, kWindowType, 0x23u, kCpuProtect);  // pool view
    void* second = pg6_xbox_XMemVirtualAlloc(nullptr, GiB, kWindowType, 0x23u, kCpuProtect);    // mirrored
    const uint64_t S = va_of(second);
    CHECK(cpu != nullptr && pool != nullptr && !gw::in_window(va_of(pool)) && S >= gw::kRuntimeRangeBase);
    size_t count = 2;
    uint64_t pfn[2] = {};
    CHECK(pg6_xbox_XMemAllocatePhysicalPages(0x380u, &count, pfn) == 1 && count == 2);

    // One page at slots 0 and 3 of the mirrored window.
    const uint64_t at[] = {S, S + 3 * k64K};
    const uint64_t same[] = {pfn[0], pfn[0]};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(at, 2, same) == 1);
    CHECK(gw::alias_slot_count() == 1);
    const uint32_t a = 0xA11A5ED1u;
    CHECK(gw::copy_to_guest(S + 3 * k64K + 0x40, &a, 4));
    uint32_t back = 0;
    CHECK(gw::copy_from_guest(&back, S + 0x40, 4) && back == a);  // written at slot 3, read at slot 0
    const uint32_t b = 0xB0B0B0B0u;
    CHECK(gw::copy_to_guest(S + 0x80, &b, 4));
    CHECK(gw::copy_from_guest(&back, S + 3 * k64K + 0x80, 4) && back == b);
    // A write through one address is a write mark for readers of the other.
    {
        gw::MarkPass pass;
        pass.begin();
        std::vector<uint64_t> none;
        CHECK(pass.take(S, 4 * k64K, &none));
        pass.begin();
        CHECK(gw::copy_to_guest(S + 3 * k64K + 0x10, &a, 4));
        std::vector<uint64_t> marked;
        CHECK(pass.take(S, k64K, &marked) && (marked == std::vector<uint64_t>{0}));
    }
    // Unmapping the page's first address: the other keeps showing its bytes.
    const uint64_t first[] = {S};
    const uint64_t zero[] = {0};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(first, 1, zero) == 1);
    CHECK(gw::copy_from_guest(&back, S + 3 * k64K + 0x40, 4) && back == a);
    CHECK(gw::copy_from_guest(&back, S + 3 * k64K + 0x80, 4) && back == b);
    // ... and mapping it back makes slot 0 its address again, still one page.
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(first, 1, &pfn[0]) == 1);
    CHECK(gw::copy_to_guest(S + 0x40, &b, 4));
    CHECK(gw::copy_from_guest(&back, S + 3 * k64K + 0x40, 4) && back == b);

    // A page in the pool view (arena memory) mapped again in the mirrored window: the same bytes.
    CHECK(pg6_xbox_XMemMapPhysicalPages(pool, 1, &pfn[1]) == pool);
    const uint64_t at2[] = {S + 8 * k64K};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(at2, 1, &pfn[1]) == 1);
    CHECK(gw::copy_to_guest(S + 8 * k64K + 4, &a, 4));
    uint32_t pooled = 0;
    std::memcpy(&pooled, static_cast<uint8_t*>(pool) + 4, 4);
    CHECK(pooled == a);
    // Unmapping the alias leaves the pool view page alone, and the slot no longer shows it.
    const uint64_t zero1[] = {0};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(at2, 1, zero1) == 1);
    std::memcpy(&pooled, static_cast<uint8_t*>(pool) + 4, 4);
    CHECK(pooled == a);

    CHECK(xmem_virtual_free(second));
    CHECK(xmem_virtual_free(pool));
    CHECK(xmem_virtual_free(cpu));
    CHECK(gw::alias_slot_count() == 0);
    xmem_reset_for_testing();
    gw::reset_for_testing();
}

// R40: under the window model, XMem map/unmap from many threads at once (plain slots do their
// data work outside the XMem lock): a mirrored physical reservation's disjoint slots keep every
// thread's bytes, while threads fighting over the same slots and pages - some in the translated
// CPU window, which take the locked path, next to the in-flight ones - leave every page's bytes
// intact and the accounting balanced.
void test_xmem_concurrent_map_unmap_in_the_window() {
    xmem_reset_for_testing();
    gw::reset_for_testing();
    gw::set_enabled(true);
    xw::set_policy(xw::PlacementPolicy::PhantomGpuTranslatedCpu);
    constexpr int kThreads = 8;
    constexpr uint64_t kOwn = 4;
    constexpr uint64_t kShared = 3;
    constexpr uint64_t kPool = 6;
    constexpr int kRounds = 30;
    void* cpu = pg6_xbox_XMemVirtualAlloc(nullptr, 16 * GiB, kWindowType, 0x21u, kCpuProtect);
    CHECK(cpu != nullptr && va_of(cpu) == W);
    const uint64_t slots = kThreads * kOwn + kShared + kPool;
    void* heap = pg6_xbox_XMemVirtualAlloc(nullptr, slots * k64K, 0x20402000u, 0x23u, kCpuProtect);
    const uint64_t H = va_of(heap);
    CHECK(heap != nullptr && H >= gw::kRuntimeRangeBase);
    auto host = [&](uint64_t slot) { return static_cast<uint8_t*>(gw::runtime_range_host(H + slot * k64K)); };
    auto map = [](uint64_t va, uint64_t page) {
        const uint64_t at[] = {va};
        const uint64_t pg[] = {page};
        return pg6_xbox_XMemMapPhysicalPagesScatter(at, 1, pg) == 1;
    };
    size_t n = kThreads * kOwn + kPool;
    std::vector<uint64_t> pfn(n);
    CHECK(pg6_xbox_XMemAllocatePhysicalPages(0x380u, &n, pfn.data()) == 1);
    const uint64_t* pool = pfn.data() + kThreads * kOwn;
    const uint64_t pool_slot = kThreads * kOwn + kShared;
    for (uint64_t p = 0; p < kPool; ++p) {
        CHECK(map(H + (pool_slot + p) * k64K, pool[p]));
        std::memset(host(pool_slot + p), static_cast<int>(0xA0 + p), k64K);
        CHECK(map(H + (pool_slot + p) * k64K, 0));
    }
    std::atomic<int> bad{0};
    std::vector<std::thread> workers;
    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&, t] {
            for (int r = 0; r < kRounds; ++r) {
                for (uint64_t k = 0; k < kOwn; ++k) {
                    const uint64_t s = t * kOwn + k;
                    const auto mark = static_cast<uint8_t>(t * 31 + r * 7 + k);
                    if (!map(H + s * k64K, pfn[s])) {
                        ++bad;
                        continue;
                    }
                    host(s)[r * 97 + k] = mark;
                    map(H + s * k64K, 0);
                    map(H + s * k64K, pfn[s]);
                    bad += host(s)[r * 97 + k] == mark ? 0 : 1;
                    map(H + s * k64K, 0);
                }
                // A shared slot of the reservation, or one of the translated CPU window.
                const uint64_t i = static_cast<uint64_t>(t + r) % kShared;
                const uint64_t va = (r % 3) == 0 ? W + i * k64K : H + (kThreads * kOwn + i) * k64K;
                bad += map(va, pool[static_cast<uint64_t>(t * 5 + r) % kPool]) ? 0 : 1;
                if ((r & 1) != 0) {
                    map(va, 0);
                }
            }
        });
    }
    for (auto& w : workers) {
        w.join();
    }
    CHECK(bad.load() == 0);
    for (uint64_t i = 0; i < kShared; ++i) {
        map(W + i * k64K, 0);
        map(H + (kThreads * kOwn + i) * k64K, 0);
    }
    Pg6XMemWorkingSetStats stats{};
    CHECK(pg6_xbox_XMemGetWorkingSetStatistics(&stats) == gdk::S_OK_HR);
    CHECK(stats.live_virtual_mappings == 0 && stats.mapped_physical_bytes == 0);
    for (int t = 0; t < kThreads; ++t) {
        for (uint64_t k = 0; k < kOwn; ++k) {
            const uint64_t s = t * kOwn + k;
            CHECK(map(H + s * k64K, pfn[s]));
            for (int r = 0; r < kRounds; ++r) {
                CHECK(host(s)[r * 97 + k] == static_cast<uint8_t>(t * 31 + r * 7 + k));
            }
        }
    }
    for (uint64_t p = 0; p < kPool; ++p) {
        CHECK(map(H + (pool_slot + p) * k64K, pool[p]));
        const uint8_t* b = host(pool_slot + p);
        CHECK(b[0] == 0xA0 + p && b[k64K / 2] == 0xA0 + p && b[k64K - 1] == 0xA0 + p);
    }
    CHECK(xmem_virtual_free(heap));
    CHECK(xmem_virtual_free(cpu));
    CHECK(pg6_xbox_XMemFreePhysicalPages(pfn.size(), pfn.data()) == 1);
    xmem_reset_for_testing();
    gw::reset_for_testing();
}

// Rank #2 regression: mirrored reservations and alias slots in [kRuntimeRangeBase, kWindowBase +
// kWindowMaxSize) must maintain contiguous runs (split_pairs == 0) under --contiguous-runs and
// report impossible adjacent aliases instead of silently violating the lifter contract.
void test_xmem_mirrored_alias_contiguity_zero_split_pairs() {
    xmem_reset_for_testing();
    gw::reset_for_testing();
    gw::set_enabled(true);
    xw::set_policy(xw::PlacementPolicy::PhantomGpuTranslatedCpu);
    gw::set_contiguous_runs_required(true, &record_violation);
    g_violation = 0;

    void* cpu = pg6_xbox_XMemVirtualAlloc(nullptr, 16 * GiB, kWindowType, 0x21u, kCpuProtect);
    void* pool = pg6_xbox_XMemVirtualAlloc(nullptr, 2 * GiB, kWindowType, 0x23u, kCpuProtect);
    void* committed = pg6_xbox_XMemVirtualAlloc(nullptr, 4 * k64K, 0x20003000u, 0x183u, kCpuProtect);
    void* phys_heap = pg6_xbox_XMemVirtualAlloc(nullptr, 8 * k64K, kWindowType, 0x23u, kCpuProtect);
    void* second = pg6_xbox_XMemVirtualAlloc(nullptr, GiB, kWindowType, 0x23u, kCpuProtect);
    CHECK(cpu != nullptr && pool != nullptr && committed != nullptr && phys_heap != nullptr && second != nullptr);
    const uint64_t H = va_of(phys_heap);
    const uint64_t S = va_of(second);
    CHECK(H >= gw::kRuntimeRangeBase && S >= gw::kRuntimeRangeBase);

    constexpr size_t kPool = 48;
    size_t count = kPool;
    uint64_t pfn[kPool] = {};
    CHECK(pg6_xbox_XMemAllocatePhysicalPages(0x380u, &count, pfn) == 1 && count == kPool);
    CHECK(pg6_xbox_XMemMapPhysicalPages(pool, kPool, pfn) == pool);
    auto* pool_bytes = static_cast<uint8_t*>(pool);
    for (size_t p = 0; p < kPool; ++p) {
        std::memset(pool_bytes + p * k64K, static_cast<int>(0x20 + p), k64K);
    }

    // Separate physical pages mapped into `phys_heap` (fresh homes, contiguous in arena).
    size_t hcount = 2;
    uint64_t hpfn[2] = {};
    CHECK(pg6_xbox_XMemAllocatePhysicalPages(0x380u, &hcount, hpfn) == 1);
    CHECK(pg6_xbox_XMemMapPhysicalPages(phys_heap, 2, hpfn) == phys_heap);

    // Map a live blocker right after pool page 5 in primary window W so mapping an adjacent alias
    // in `second` forces `relocate_run_locked` on mirrored alias slots too.
    const uint64_t block_va[] = {W + 100 * k64K};
    const uint64_t block_pg[] = {pfn[6]};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(block_va, 1, block_pg) == 1);

    // Map out-of-order pool pages into adjacent slots of the mirrored second CPU window `S`:
    // S + 10*64K <- pfn[5], S + 11*64K <- pfn[19], S + 12*64K <- pfn[2].
    // Also map pfn[30] at an isolated slot in W and then into an adjacent run in S.
    const uint64_t iso_va[] = {W + 200 * k64K};
    const uint64_t iso_pg[] = {pfn[30]};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(iso_va, 1, iso_pg) == 1);

    const uint64_t s_vas[] = {S + 10 * k64K, S + 11 * k64K, S + 12 * k64K, S + 13 * k64K};
    const uint64_t s_pgs[] = {pfn[5], pfn[19], pfn[2], pfn[30]};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(s_vas, 4, s_pgs) == 1);

    CHECK(g_violation == 0);
    const XMemWindowContiguity c = xmem_window_contiguity();
    const gw::Stats ws = gw::stats();
    CHECK(c.run_breaks == 0);
    CHECK(c.run_relocations >= 1);
    CHECK(ws.split_pairs == 0);

    // Contiguous direct host pointer spans all 4 alias slots with exact expected bytes.
    const auto* run = static_cast<const uint8_t*>(gw::contiguous(S + 10 * k64K, 4 * k64K));
    CHECK(run != nullptr);
    if (run != nullptr) {
        CHECK(run[0] == 0x25 && run[k64K - 1] == 0x25);
        CHECK(run[k64K] == 0x20 + 19 && run[2 * k64K - 1] == 0x20 + 19);
        CHECK(run[2 * k64K] == 0x22 && run[3 * k64K - 1] == 0x22);
        CHECK(run[3 * k64K] == 0x20 + 30 && run[4 * k64K - 1] == 0x20 + 30);
    }
    // Isolated slot in W still points to pfn[30]'s bytes after relocation.
    uint8_t b = 0;
    CHECK(gw::copy_from_guest(&b, W + 200 * k64K + 123, 1) && b == 0x20 + 30);

    // Impossible adjacent self-alias in a mirrored reservation (>= kRuntimeRangeBase) must record
    // a run_break without triggering a fatal contiguity_violation (real games mix primary and
    // alias/2 MiB pages in mirrored reservations).
    const uint64_t bad_va[] = {S + 14 * k64K};
    const uint64_t bad_pg[] = {pfn[30]};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(bad_va, 1, bad_pg) == 1);
    CHECK(g_violation == 0);
    CHECK(xmem_window_contiguity().run_breaks == 1);

    gw::set_contiguous_runs_required(false);
    CHECK(xmem_virtual_free(second));
    CHECK(xmem_virtual_free(phys_heap));
    CHECK(xmem_virtual_free(committed));
    CHECK(xmem_virtual_free(pool));
    CHECK(xmem_virtual_free(cpu));
    CHECK(pg6_xbox_XMemFreePhysicalPages(kPool, pfn) == 1);
    CHECK(pg6_xbox_XMemFreePhysicalPages(2, hpfn) == 1);
    xmem_reset_for_testing();
    gw::reset_for_testing();
}

// Rank #1 regression (readers): relocating a live mapped run under --contiguous-runs must never
// overwrite live mapped frames in place or expose torn/wrong-page bytes to concurrent readers
// scanning across 64 KiB boundaries with cached window-table deltas.
void test_xmem_concurrent_relocation_no_torn_reads() {
    xmem_reset_for_testing();
    gw::reset_for_testing();
    gw::set_enabled(true);
    xw::set_policy(xw::PlacementPolicy::PhantomGpuTranslatedCpu);
    gw::set_contiguous_runs_required(true, &record_violation);
    g_violation = 0;

    void* cpu = pg6_xbox_XMemVirtualAlloc(nullptr, 16 * GiB, kWindowType, 0x21u, kCpuProtect);
    void* pool = pg6_xbox_XMemVirtualAlloc(nullptr, 2 * GiB, kWindowType, 0x23u, kCpuProtect);
    CHECK(cpu != nullptr && pool != nullptr);

    constexpr size_t kPool = 128;
    size_t count = kPool;
    uint64_t pfn[kPool] = {};
    CHECK(pg6_xbox_XMemAllocatePhysicalPages(0x380u, &count, pfn) == 1 && count == kPool);
    CHECK(pg6_xbox_XMemMapPhysicalPages(pool, kPool, pfn) == pool);

    // Stamp each pool page with unique 64-bit words; page 0 starts with 'PSIN' like a PSO buffer.
    auto* pool_bytes = static_cast<uint8_t*>(pool);
    auto expected_word = [](size_t page_idx, size_t word_idx) -> uint64_t {
        if (page_idx == 0 && word_idx == 0) {
            return 0xA5A5A5A54E495350ull;  // "PSIN" + tag
        }
        return (static_cast<uint64_t>(page_idx + 1) << 32) | static_cast<uint64_t>(word_idx * 0x9E37u + 1u);
    };
    constexpr size_t kWordsPerPage = k64K / sizeof(uint64_t);
    for (size_t p = 0; p < kPool; ++p) {
        auto* words = reinterpret_cast<uint64_t*>(pool_bytes + p * k64K);
        for (size_t w = 0; w < kWordsPerPage; ++w) {
            words[w] = expected_word(p, w);
        }
    }

    // Live 2-page run [W + 16*k64K, W + 18*k64K) holding pages 0 and 1.
    const uint64_t run_va = W + 16 * k64K;
    const uint64_t init_vas[] = {run_va, run_va + k64K};
    const uint64_t init_pgs[] = {pfn[0], pfn[1]};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(init_vas, 2, init_pgs) == 1);

    std::atomic<bool> stop{false};
    std::atomic<int> readers_ready{0};
    std::atomic<uint64_t> torn_reads{0};
    std::atomic<uint64_t> total_scans{0};
    constexpr int kReaders = 4;
    std::vector<std::thread> readers;
    for (int t = 0; t < kReaders; ++t) {
        readers.emplace_back([&] {
            const uint64_t slot0 = (run_va - W) >> gw::kSlotShift;
            readers_ready.fetch_add(1, std::memory_order_release);
            while (!stop.load(std::memory_order_relaxed)) {
                // Simulate lifted --contiguous-runs code: load delta for slot0 once, then read
                // straight across the 64 KiB boundary into slot1 without a straddle check.
                const uint64_t delta = static_cast<uint64_t>(
                    std::atomic_ref<int64_t>(__playgta6_window_table[slot0]).load(std::memory_order_acquire));
                if (delta == gw::kTrapDelta) {
                    torn_reads.fetch_add(1, std::memory_order_relaxed);
                    break;
                }
                const auto* words = reinterpret_cast<const uint64_t*>(static_cast<uintptr_t>(run_va + delta));
                for (size_t i = 0; i < 2 * kWordsPerPage; i += 17) {
                    const size_t p = i / kWordsPerPage;
                    const size_t w = i % kWordsPerPage;
                    if (words[i] != expected_word(p, w)) {
                        torn_reads.fetch_add(1, std::memory_order_relaxed);
                        return;
                    }
                }
                // Also verify the exact 64 KiB straddle word.
                uint64_t straddle = 0;
                std::memcpy(&straddle,
                            reinterpret_cast<const uint8_t*>(words) + k64K - 4,
                            sizeof(straddle));
                uint64_t expect_straddle = 0;
                const uint64_t last_p0 = expected_word(0, kWordsPerPage - 1);
                const uint64_t first_p1 = expected_word(1, 0);
                std::memcpy(&expect_straddle, reinterpret_cast<const uint8_t*>(&last_p0) + 4, 4);
                std::memcpy(reinterpret_cast<uint8_t*>(&expect_straddle) + 4, &first_p1, 4);
                if (straddle != expect_straddle) {
                    torn_reads.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
                total_scans.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // Ensure all reader threads are actively scanning before starting the relocation storm.
    while (readers_ready.load(std::memory_order_acquire) < kReaders ||
           total_scans.load(std::memory_order_acquire) == 0) {
        std::this_thread::yield();
    }

    // Repeatedly force relocate_run_locked on the live run [run_va, run_va + 2*k64K):
    // pin the pool frame right after run_va + k64K by mapping whatever pool page lives there into
    // a separate live window slot, then map a 3rd page at run_va + 2*k64K (forcing full run
    // relocation), then unmap the 3rd page and the blocker.
    const uint64_t blocker_slot = W + 64 * k64K;
    const uint64_t third_slot = run_va + 2 * k64K;
    const uint64_t unmap_zero[] = {0};
    constexpr size_t kRelocationRounds = 32;
    for (size_t r = 0; r < kRelocationRounds; ++r) {
        // Find which pool page currently sits right after `run_va + k64K` and pin it in blocker_slot.
        const auto* h1 = static_cast<const uint8_t*>(gw::slot_host(run_va + k64K));
        CHECK(h1 != nullptr);
        const size_t next_pool_idx = static_cast<size_t>((h1 - pool_bytes) / k64K) + 1;
        if (next_pool_idx < kPool) {
            // Map all non-member pages around next_pool_idx? No, mapping the single page owning
            // next_pool_idx makes place_at_locked fail and forces relocate_run_locked.
            // Determine which pfn owns pool + next_pool_idx * k64K from its stamped word!
            uint64_t stamped = 0;
            std::memcpy(&stamped, pool_bytes + next_pool_idx * k64K + 8, sizeof(stamped));
            const size_t owner_idx = static_cast<size_t>((stamped >> 32) - 1);
            if (owner_idx >= 2 && owner_idx < kPool) {
                const uint64_t bpg = pfn[owner_idx];
                CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(&blocker_slot, 1, &bpg) == 1);
            }
        }
        const uint64_t third_pg = pfn[2 + (r % 20)];
        if (gw::slot_host(blocker_slot) != nullptr) {
            uint64_t b_word = 0;
            std::memcpy(&b_word, static_cast<const uint8_t*>(gw::slot_host(blocker_slot)) + 8, sizeof(b_word));
            const size_t b_idx = static_cast<size_t>((b_word >> 32) - 1);
            const uint64_t use_pg = (b_idx == 2 + (r % 20)) ? pfn[30 + (r % 20)] : third_pg;
            CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(&third_slot, 1, &use_pg) == 1);
        } else {
            CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(&third_slot, 1, &third_pg) == 1);
        }
        CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(&third_slot, 1, unmap_zero) == 1);
        CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(&blocker_slot, 1, unmap_zero) == 1);
    }

    stop.store(true, std::memory_order_release);
    for (auto& th : readers) {
        th.join();
    }

    const XMemWindowContiguity c = xmem_window_contiguity();
    const gw::Stats ws = gw::stats();
    CHECK(torn_reads.load() == 0);
    CHECK(total_scans.load() > 0);
    CHECK(c.run_relocations >= 16);
    CHECK(c.run_breaks == 0);
    CHECK(g_violation == 0);
    CHECK(ws.split_pairs == 0);

    gw::set_contiguous_runs_required(false);
    CHECK(xmem_virtual_free(pool));
    CHECK(xmem_virtual_free(cpu));
    CHECK(pg6_xbox_XMemFreePhysicalPages(kPool, pfn) == 1);
    xmem_reset_for_testing();
    gw::reset_for_testing();
}

// Rank #1 regression (concurrent host writers): relocating a live run while another thread holds a
// writable Buffers in-place view must wait for the write to finish so zero bytes are lost into a
// retired frame.
void test_xmem_concurrent_write_pin_and_relocation() {
    xmem_reset_for_testing();
    gw::reset_for_testing();
    gw::set_enabled(true);
    xw::set_policy(xw::PlacementPolicy::PhantomGpuTranslatedCpu);
    gw::set_contiguous_runs_required(true, &record_violation);
    g_violation = 0;

    void* cpu = pg6_xbox_XMemVirtualAlloc(nullptr, 16 * GiB, kWindowType, 0x21u, kCpuProtect);
    void* pool = pg6_xbox_XMemVirtualAlloc(nullptr, 2 * GiB, kWindowType, 0x23u, kCpuProtect);
    CHECK(cpu != nullptr && pool != nullptr);

    constexpr size_t kPool = 32;
    size_t count = kPool;
    uint64_t pfn[kPool] = {};
    CHECK(pg6_xbox_XMemAllocatePhysicalPages(0x380u, &count, pfn) == 1 && count == kPool);
    CHECK(pg6_xbox_XMemMapPhysicalPages(pool, kPool, pfn) == pool);

    const uint64_t run_va = W + 8 * k64K;
    const uint64_t init_vas[] = {run_va, run_va + k64K, W + 40 * k64K};
    const uint64_t init_pgs[] = {pfn[0], pfn[1], pfn[2]};  // pfn[2] blocks pool frame 2!
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(init_vas, 3, init_pgs) == 1);

    std::atomic<bool> pinned{false};
    std::atomic<bool> copied{false};
    std::vector<uint8_t> payload(2 * k64K);
    payload[0] = 'P';
    payload[1] = 'S';
    payload[2] = 'I';
    payload[3] = 'N';
    for (size_t i = 4; i < payload.size(); ++i) {
        payload[i] = static_cast<uint8_t>((i * 29u + 13u) & 0xFFu);
    }

    std::thread writer([&] {
        gw::Buffers bufs;
        void* dst = bufs.range(run_va, payload.size(), true);
        CHECK(dst != nullptr && bufs.bounce_count() == 0);
        pinned.store(true, std::memory_order_release);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (dst != nullptr) {
            std::memcpy(dst, payload.data(), payload.size());
        }
        // Set BEFORE the pin is released: the remap may only return after this point.
        copied.store(true, std::memory_order_release);
        bufs.finish();
    });

    while (!pinned.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    // Map a 3rd page adjacent to `run_va + k64K`: must relocate [run_va, run_va + 3*k64K), and
    // must wait for `writer`'s WritePin before copying & switching frames!
    const uint64_t third_va = run_va + 2 * k64K;
    const uint64_t third_pg = pfn[10];
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(&third_va, 1, &third_pg) == 1);
    // The remap returns only once the pin is released, which happens after the copy completed.
    // (Checking "writer thread fully finished" here would race with the thread's tail after finish().)
    CHECK(copied.load(std::memory_order_acquire));
    writer.join();

    // Every byte written by `writer` must be intact in the relocated run!
    std::vector<uint8_t> back(payload.size(), 0);
    CHECK(gw::copy_from_guest(back.data(), run_va, back.size()));
    CHECK(back == payload);
    const XMemWindowContiguity c = xmem_window_contiguity();
    CHECK(c.run_relocations >= 1 && c.run_breaks == 0 && g_violation == 0);
    CHECK(gw::stats().split_pairs == 0);

    gw::set_contiguous_runs_required(false);
    CHECK(xmem_virtual_free(pool));
    CHECK(xmem_virtual_free(cpu));
    CHECK(pg6_xbox_XMemFreePhysicalPages(kPool, pfn) == 1);
    xmem_reset_for_testing();
    gw::reset_for_testing();
}

// Regression: mirrored reservations (>= kRuntimeRangeBase) can legitimately mix primary mappings
// with adjacent alias mappings (e.g. a page whose home lives in another reservation) under
// --contiguous-runs. That must record run_breaks without triggering a fatal contiguity_violation.
void test_xmem_mirrored_mixed_primary_and_alias_no_fatal_trap() {
    xmem_reset_for_testing();
    gw::reset_for_testing();
    gw::set_enabled(true);
    xw::set_policy(xw::PlacementPolicy::PhantomGpuTranslatedCpu);
    gw::set_contiguous_runs_required(true, &record_violation);
    g_violation = 0;

    void* cpu = pg6_xbox_XMemVirtualAlloc(nullptr, 16 * GiB, kWindowType, 0x21u, kCpuProtect);
    void* pool = pg6_xbox_XMemVirtualAlloc(nullptr, 2 * GiB, kWindowType, 0x23u, kCpuProtect);
    void* mirror_a = pg6_xbox_XMemVirtualAlloc(nullptr, 64 * k64K, kWindowType, 0x23u, kCpuProtect);
    void* mirror_b = pg6_xbox_XMemVirtualAlloc(nullptr, 64 * k64K, kWindowType, 0x23u, kCpuProtect);
    CHECK(cpu != nullptr && pool != nullptr && mirror_a != nullptr && mirror_b != nullptr);
    const uint64_t wa = va_of(mirror_a);
    const uint64_t wb = va_of(mirror_b);
    CHECK(wa >= gw::kRuntimeRangeBase && wb >= gw::kRuntimeRangeBase);

    size_t count = 2;
    uint64_t pfn[2] = {};
    CHECK(pg6_xbox_XMemAllocatePhysicalPages(0x380u, &count, pfn) == 1 && count == 2);

    // Map pfn[0] as a primary page in mirror_a, then alias it at mirror_b[0], and map pfn[1] as a
    // primary page right beside it at mirror_b[1] (non-adjacent host frames in a mirrored region).
    const uint64_t a0 = wa;
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(&a0, 1, &pfn[0]) == 1);
    const uint64_t b0 = wb;
    const uint64_t b1 = wb + k64K;
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(&b0, 1, &pfn[0]) == 1);
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(&b1, 1, &pfn[1]) == 1);

    // Must not trigger a fatal contiguity_violation on mirrored slots (>= kRuntimeRangeBase).
    CHECK(g_violation == 0);
    CHECK(xmem_window_contiguity().run_breaks >= 1);

    const uint32_t tag0 = 0x11223344u;
    const uint32_t tag1 = 0x55667788u;
    CHECK(gw::copy_to_guest(b0, &tag0, sizeof(tag0)));
    CHECK(gw::copy_to_guest(b1, &tag1, sizeof(tag1)));
    uint32_t read_a0 = 0;
    uint32_t read_b1 = 0;
    CHECK(gw::copy_from_guest(&read_a0, a0, sizeof(read_a0)) && read_a0 == tag0);
    CHECK(gw::copy_from_guest(&read_b1, b1, sizeof(read_b1)) && read_b1 == tag1);

    gw::set_contiguous_runs_required(false);
    CHECK(xmem_virtual_free(mirror_b));
    CHECK(xmem_virtual_free(mirror_a));
    CHECK(xmem_virtual_free(pool));
    CHECK(xmem_virtual_free(cpu));
    CHECK(pg6_xbox_XMemFreePhysicalPages(2, pfn) == 1);
    xmem_reset_for_testing();
    gw::reset_for_testing();
}

// Regression: mapping a 2 MiB physical page into an uncommitted mirrored MEM_PHYSICAL reservation
// must populate ALL 32 x 64 KiB guest-window slots (not just slot 0), and unmapping must clear
// all 32 slots cleanly.
void test_xmem_mirrored_2mb_page_maps_all_32_slots() {
    xmem_reset_for_testing();
    gw::reset_for_testing();
    gw::set_enabled(true);
    xw::set_policy(xw::PlacementPolicy::PhantomGpuTranslatedCpu);
    gw::set_contiguous_runs_required(true, &record_violation);
    g_violation = 0;

    void* cpu = pg6_xbox_XMemVirtualAlloc(nullptr, 16 * GiB, kWindowType, 0x21u, kCpuProtect);
    void* pool = pg6_xbox_XMemVirtualAlloc(nullptr, 2 * GiB, kWindowType, 0x23u, kCpuProtect);
    constexpr uint64_t k2M = 2ull << 20;
    void* mirror_2m = pg6_xbox_XMemVirtualAlloc(nullptr, 4 * k2M, kWindowType, 0x23u, kCpuProtect);
    CHECK(cpu != nullptr && pool != nullptr && mirror_2m != nullptr);
    const uint64_t wm = va_of(mirror_2m);
    CHECK(wm >= gw::kRuntimeRangeBase);

    // 0x20000000 = MEM_LARGE_PAGES (2 MiB pages) in XMemAllocatePhysicalPages.
    size_t count = 1;
    uint64_t pfn[1] = {};
    CHECK(pg6_xbox_XMemAllocatePhysicalPages(0x20000380u, &count, pfn) == 1 && count == 1);

    // Uncommitted mirrored MEM_PHYSICAL reservation starts unmapped.
    CHECK(gw::slot_host(wm) == nullptr);
    CHECK(gw::slot_host(wm + 31 * k64K) == nullptr);

    // Map the 2 MiB page: ALL 32 64 KiB slots must become mapped and host-contiguous!
    CHECK(pg6_xbox_XMemMapPhysicalPages(mirror_2m, 1, pfn) == mirror_2m);
    CHECK(g_violation == 0);
    auto* base_host = static_cast<uint8_t*>(gw::slot_host(wm));
    CHECK(base_host != nullptr);
    for (uint64_t s = 0; s < 32; ++s) {
        CHECK(gw::slot_host(wm + s * k64K) == base_host + s * k64K);
    }
    // Full 2 MiB range must be directly host-contiguous (no kTrapDelta in slots 1..31).
    CHECK(gw::contiguous(wm, static_cast<size_t>(k2M)) == base_host);
    const uint32_t sentinel = 0xCAFEBABEu;
    CHECK(gw::copy_to_guest(wm + 31 * k64K + 4096, &sentinel, sizeof(sentinel)));
    uint32_t back = 0;
    CHECK(gw::copy_from_guest(&back, wm + 31 * k64K + 4096, sizeof(back)) && back == sentinel);

    // Unmap the 2 MiB page: ALL 32 slots must return to unmapped.
    CHECK(pg6_xbox_XMemMapPhysicalPages(mirror_2m, 1, nullptr) == mirror_2m);
    for (uint64_t s = 0; s < 32; ++s) {
        CHECK(gw::slot_host(wm + s * k64K) == nullptr);
    }

    gw::set_contiguous_runs_required(false);
    CHECK(xmem_virtual_free(mirror_2m));
    CHECK(xmem_virtual_free(pool));
    CHECK(xmem_virtual_free(cpu));
    CHECK(pg6_xbox_XMemFreePhysicalPages(1, pfn) == 1);
    xmem_reset_for_testing();
    gw::reset_for_testing();
}

}  // namespace

// A GuestRangeView over a window range holds the arena lock (a concurrent decommit of the arena
// pages behind the window must wait for the view, then the range reads as unmapped).
void test_range_view_holds_the_arena_lock_in_the_window() {
    gw::reset_for_testing();
    gw::set_enabled(true);
    Pages pages;
    CHECK(pages.host != nullptr);
    CHECK(gw::map_slot(W, pages.page(0)));
    std::atomic<bool> decommitted{false};
    std::thread decommitter;
    {
        const playgta6::guest::com::GuestRangeView view(W + 64, 4096);
        CHECK(view.data() == pages.page(0) + 64);
        CHECK(view.held());
        decommitter = std::thread([&] {
            CHECK(arena().decommit(pages.va, k64K));
            decommitted.store(true);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        CHECK(!decommitted.load());  // blocked while the view is alive
    }
    decommitter.join();
    CHECK(decommitted.load());
    const playgta6::guest::com::GuestRangeView after(W + 64, 4096);
    CHECK(after.data() == nullptr);  // decommitted: not mapped, no fault
    gw::reset_for_testing();
}

// A shared page view that is unmapped or re-pointed while readers hold the arena lock over it
// (host_ptr_const_held, what GuestRangeView does) must read as not mapped from the moment the
// pages leave the Commit state: the host unmap runs outside the lock, and a reader that saw Commit
// over a vanishing view faulted the host (flush_cpu_buffers hashing a window range).
void test_shared_view_unmap_never_faults_a_locked_reader() {
    if (!arena().has_shared_zone()) {
        return;
    }
    const uint64_t va = arena().reserve_shared(2 * k64K);
    void* page_a = create_shared_page(k64K);
    void* page_b = create_shared_page(k64K);
    CHECK(va != 0 && page_a != nullptr && page_b != nullptr);
    if (va == 0 || page_a == nullptr || page_b == nullptr) {
        return;
    }
    CHECK(arena().map_shared(va, page_a, k64K));
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> reads{0};
    std::vector<std::thread> readers;
    for (int r = 0; r < 3; ++r) {
        readers.emplace_back([&] {
            uint64_t sink = 0;
            while (!stop.load()) {
                std::unique_lock<std::mutex> hold;
                const uint8_t* p = arena().host_ptr_const_held(va, k64K, hold);
                if (p != nullptr) {
                    sink += *static_cast<const volatile uint8_t*>(p);  // the first byte: a read of an unmapped view faults
                    reads.fetch_add(1);
                }
            }
            (void)sink;
        });
    }
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(1000);
    while (std::chrono::steady_clock::now() < until) {
        CHECK(arena().map_shared(va, page_b, k64K));  // another page shown: unmap, then map
        CHECK(arena().unmap_shared(va, k64K));
        CHECK(arena().map_shared(va, page_a, k64K));
    }
    stop.store(true);
    for (std::thread& r : readers) {
        r.join();
    }
    CHECK(reads.load() > 0);
    CHECK(arena().unmap_shared(va, k64K));
    {
        std::unique_lock<std::mutex> hold;
        CHECK(arena().host_ptr_const_held(va, k64K, hold) == nullptr);
    }
    CHECK(arena().release(va));
    release_shared_page(page_a);
    release_shared_page(page_b);
}

int main() {
    std::printf("test_guest_window\n");
    test_table_translation_and_copies();
    test_slow_path_helpers();
    test_com_bridge_window();
    test_xmem_translated_cpu_window();
    test_xmem_window_run_keeper();
    test_xmem_run_keeper_stays_in_the_pool();
    test_write_marks();
    test_xmem_physical_reservations_in_the_window();
    test_xmem_committed_regions_in_the_window();
    test_xmem_aliases_in_the_window();
    test_xmem_concurrent_map_unmap_in_the_window();
    test_xmem_mirrored_alias_contiguity_zero_split_pairs();
    test_xmem_concurrent_relocation_no_torn_reads();
    test_xmem_concurrent_write_pin_and_relocation();
    test_xmem_mirrored_mixed_primary_and_alias_no_fatal_trap();
    test_range_view_holds_the_arena_lock_in_the_window();
    test_xmem_mirrored_2mb_page_maps_all_32_slots();
    test_shared_view_unmap_never_faults_a_locked_reader();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_guest_window: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_guest_window: all tests passed\n");
    return 0;
}
