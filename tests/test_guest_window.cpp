// R19 translated CPU window (os/guest_window.h): the delta table, the shared translation,
// segmented copies, contiguity statistics, the slow-path helpers lifted code calls, and the XMem
// integration (PlacementPolicy::PhantomGpuTranslatedCpu): a 16 GiB CPU window that takes no pool
// address space, pool pages scattered across it, aliasing the pool view. Synthetic data only.

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

}  // namespace

int main() {
    std::printf("test_guest_window\n");
    test_table_translation_and_copies();
    test_slow_path_helpers();
    test_com_bridge_window();
    test_xmem_translated_cpu_window();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_guest_window: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_guest_window: all tests passed\n");
    return 0;
}
