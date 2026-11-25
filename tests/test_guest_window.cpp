// R19 translated CPU window (os/guest_window.h): the delta table, the shared translation,
// segmented copies, contiguity statistics, the slow-path helpers lifted code calls, and the XMem
// integration (PlacementPolicy::PhantomGpuTranslatedCpu): a 16 GiB CPU window that takes no pool
// address space, pool pages scattered across it, aliasing the pool view. Synthetic data only.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
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

}  // namespace

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
    if (g_failures != 0) {
        std::fprintf(stderr, "test_guest_window: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_guest_window: all tests passed\n");
    return 0;
}
