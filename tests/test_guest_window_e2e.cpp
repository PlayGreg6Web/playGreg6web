// R19 end-to-end: the title's memory layout fits the browser's 12 GiB guest pool once the 16 GiB
// CPU window is translated (os/guest_window.h, lifter MemoryModel::Windowed), and lifted-style
// code reads and writes pool pages scattered across the window through window addresses.
//
// The test installs a 12 GiB identity process arena (the real wasm pool, [4 GiB, 16 GiB); natively
// the same size), replays the title's reservations (CPU and GPU windows of 16 GiB, the 3.88 GiB
// pool view, a 1.03 GiB heap budget, the image range), maps pool pages at scattered window slots
// and calls a guest function through the guest runtime:
//     rax = [rcx] + [rcx + 8];  [rdx] = rax;  [r8] = rax;  ret
// with rcx / rdx window addresses that straddle 64 KiB slots (one inside a contiguous run, one
// across two unrelated pages) and r8 outside the window. The function is a hand-written C++
// equivalent of what `MemoryModel::Windowed` lifted code does and, built with
// -DPLAYGTA6_E2E_LIFTED, the real lifted sub_140001000 from
// recomp/src/lift/tests/exec_memory.rs `writes_the_runtime_window_fixture`.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "abi/playgta6_abi.h"
#include "guest/guest_runtime.h"
#include "os/guest_window.h"
#include "os/vmem.h"
#include "os/win32_kernel.h"
#include "os/xbox_scarlett.h"
#include "os/xmem_windows.h"

extern "C" void* __playgta6_guest_read(uint64_t va, uint64_t n);
extern "C" void* __playgta6_guest_scratch();
extern "C" void __playgta6_guest_write(uint64_t va, uint64_t n);

#ifdef PLAYGTA6_E2E_LIFTED
extern "C" void sub_140001000(playgta6::GuestState* state);
#endif

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, const char* what, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_guest_window_e2e.cpp:%d] %s: %s\n", line, what, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, what, __LINE__)

using namespace playgta6;
using namespace playgta6::os;
using namespace playgta6::os::scarlett;
namespace gw = playgta6::os::guest_window;
namespace xw = playgta6::os::xmem_windows;

constexpr uint64_t GiB = 1ull << 30;
constexpr uint64_t k64K = 64ull << 10;
constexpr uint64_t W = gw::kWindowBase;
constexpr uint64_t S = gw::kWindowMaxSize;
constexpr uint64_t kPoolBytes = 12 * GiB;              // the browser's guest pool
constexpr uint64_t kTitleWindow = 16 * GiB;            // Series X window size
constexpr uint64_t kPoolView = 0xF890'0000ull;         // the title's page pool (3.88 GiB)
constexpr uint64_t kHeapBudget = 0x4200'0000ull;       // 1.03 GiB heap budget
constexpr uint64_t kImageBase = 0x1'4000'0000ull;
constexpr uint64_t kImageSize = 0x0B06'D000ull;
constexpr uint64_t kEntry = 0x1'4000'1000ull;
constexpr uint32_t kWindowType = 0x20402000u;
constexpr size_t kPages = 64;

// ---- what MemoryModel::Windowed lifted code does (flat translation: __playgta6_guest_base) ----

uint64_t guest_load64(uint64_t va) {
    uint64_t v = 0;
    if (va - W < S) {
        const uint64_t delta = static_cast<uint64_t>(__playgta6_window_table[(va - W) >> 16]);
        if ((va & 0xFFFF) <= 0x10000 - 8) {
            std::memcpy(&v, reinterpret_cast<const void*>(static_cast<uintptr_t>(va + delta)), 8);
        } else {
            std::memcpy(&v, __playgta6_guest_read(va, 8), 8);
        }
    } else {
        std::memcpy(&v, reinterpret_cast<const void*>(static_cast<uintptr_t>(__playgta6_guest_base + va)), 8);
    }
    return v;
}

void guest_store64(uint64_t va, uint64_t v) {
    if (va - W < S) {
        const uint64_t delta = static_cast<uint64_t>(__playgta6_window_table[(va - W) >> 16]);
        if ((va & 0xFFFF) <= 0x10000 - 8) {
            std::memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(va + delta)), &v, 8);
        } else {
            std::memcpy(__playgta6_guest_scratch(), &v, 8);
            __playgta6_guest_write(va, 8);
        }
    } else {
        std::memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(__playgta6_guest_base + va)), &v, 8);
    }
}

void sub_handwritten(GuestState* s) {
    uint64_t* r = s->gpr;
    r[GuestReg::Rax] = guest_load64(r[GuestReg::Rcx]) + guest_load64(r[GuestReg::Rcx] + 8);
    guest_store64(r[GuestReg::Rdx], r[GuestReg::Rax]);
    guest_store64(r[GuestReg::R8], r[GuestReg::Rax]);
    r[GuestReg::Rsp] += 8;  // ret
}

uint8_t pattern(size_t page, size_t offset) {
    return static_cast<uint8_t>(page * 31 + offset * 7 + 1);
}

uint64_t pattern64(size_t page, size_t offset) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) {
        v = (v << 8) | pattern(page, offset + static_cast<size_t>(i));
    }
    return v;
}

struct Title {
    void* cpu = nullptr;
    void* gpu = nullptr;
    uint8_t* view = nullptr;
    uint64_t pfn[kPages] = {};
    // Window slot -> pool page mapped there.
    std::vector<std::pair<uint64_t, size_t>> slots;
};

// The title's reservations and mappings; returns false if anything does not fit.
bool set_up(const char* what, VirtualMemoryArena& arena, Title& t) {
    // The image is mapped first (boot), the windows right after D3D12ConfigureVirtualMemoryX.
    const uint64_t image = arena.reserve(kImageBase, kImageSize, PageProtect::NoAccess);
    t.cpu = pg6_xbox_XMemVirtualAlloc(nullptr, kTitleWindow, kWindowType, 0x21u, 0x4u);
    t.gpu = pg6_xbox_XMemVirtualAlloc(nullptr, kTitleWindow, kWindowType, 0x3u, 0x801u);
    t.view = static_cast<uint8_t*>(
        pg6_xbox_XMemVirtualAlloc(nullptr, kPoolView, kWindowType & ~0x400000u, 0x23u, 0x4u));
    const uint64_t heap = arena.reserve(0, kHeapBudget, PageProtect::ReadWrite);
    CHECK(t.cpu != nullptr && t.gpu != nullptr && t.view != nullptr);
    CHECK(heap != 0 && image == kImageBase);
    CHECK(arena.guest_address(t.cpu) == W);
    if (t.cpu == nullptr || t.view == nullptr || heap == 0 || image != kImageBase) {
        return false;
    }
    std::printf("  %s: windows 2 x 16 GiB placed outside the pool; pool view %llu MiB, heap %llu MiB, image at 0x%llx; "
                "arena reserved %llu MiB of %llu GiB\n",
                what, static_cast<unsigned long long>(kPoolView >> 20),
                static_cast<unsigned long long>(kHeapBudget >> 20), static_cast<unsigned long long>(image),
                static_cast<unsigned long long>(arena.reserved_bytes() >> 20),
                static_cast<unsigned long long>(arena.space_size() >> 30));

    size_t count = kPages;
    CHECK(pg6_xbox_XMemAllocatePhysicalPages(0x380u, &count, t.pfn) == 1 && count == kPages);
    CHECK(pg6_xbox_XMemMapPhysicalPages(t.view, kPages, t.pfn) == t.view);
    for (size_t p = 0; p < kPages; ++p) {
        for (size_t i = 0; i < k64K; ++i) {
            t.view[p * k64K + i] = pattern(p, i);
        }
    }
    // A run of four consecutive pool pages at consecutive slots, then pages scattered across the
    // whole 16 GiB window (deterministic, distinct slots, never adjacent to another mapping).
    const uint64_t run = W + 5 * GiB + 3 * k64K;
    for (size_t i = 0; i < 4; ++i) {
        t.slots.emplace_back(run + i * k64K, 10 + i);
    }
    uint64_t x = 0x9E3779B97F4A7C15ull;
    for (size_t p = 20; p < 60; ++p) {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        const uint64_t slot = (x >> 20) % (kTitleWindow / k64K / 2) * 2;  // even slots only
        t.slots.emplace_back(W + slot * k64K, p);
    }
    // One more page right after the first scattered slot: adjacent slots, unrelated pages.
    t.slots.emplace_back(t.slots[4].first + k64K, 63);
    std::vector<uint64_t> vas, pages;
    for (const auto& [va, p] : t.slots) {
        vas.push_back(va);
        pages.push_back(t.pfn[p]);
    }
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(vas.data(), vas.size(), pages.data()) == 1);
    const gw::Stats st = gw::stats();
    std::printf("  %s: %llu window slots mapped, %llu host-contiguous neighbour pairs, %llu split pairs\n", what,
                static_cast<unsigned long long>(st.mapped_slots), static_cast<unsigned long long>(st.contiguous_pairs),
                static_cast<unsigned long long>(st.split_pairs));
    return true;
}

void run(const char* what, VirtualMemoryArena& arena, Title& t, LiftedFn fn) {
    guest::reset_guest_runtime();
    static Pg6LiftedFunctionEntry table[1];
    table[0] = {kEntry, fn};
    CHECK(guest::register_lifted_functions(table));
    guest::GuestThread thread;
    CHECK(guest::create_guest_thread(arena, 64 * 1024, thread));
    const uint64_t out = arena.virtual_alloc(0, k64K, kMemReserve | kMemCommit, PageProtect::ReadWrite);
    CHECK(out != 0);

    // rcx: 4 bytes before the end of run slot 1 (pages 11 | 12, host-adjacent); its second qword
    // continues on page 12. rdx: 3 bytes before the end of the scattered slot whose neighbour holds
    // page 63 (unrelated pages: the slow path).
    const uint64_t rcx = t.slots[1].first + k64K - 4;
    const uint64_t rdx = t.slots[4].first + k64K - 3;
    auto expect64 = [&](size_t page, size_t next, size_t offset) {
        uint64_t v = 0;
        for (int i = 7; i >= 0; --i) {
            const size_t o = offset + static_cast<size_t>(i);
            v = (v << 8) | (o < k64K ? pattern(page, o) : pattern(next, o - k64K));
        }
        return v;
    };
    const uint64_t a = expect64(11, 12, k64K - 4);
    const uint64_t b = pattern64(12, 4);
    const uint64_t rax = guest::call_guest_args(arena, thread.state, kEntry, rcx, rdx, out);
    CHECK(rax == a + b);
    // The straddling store landed on two unrelated pool pages, visible through the pool view.
    uint64_t back = 0;
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&back);
    for (int i = 0; i < 3; ++i) {
        bytes[i] = t.view[t.slots[4].second * k64K + k64K - 3 + static_cast<size_t>(i)];
    }
    for (int i = 3; i < 8; ++i) {
        bytes[i] = t.view[63 * k64K + static_cast<size_t>(i - 3)];
    }
    CHECK(back == a + b);
    uint64_t flat = 0;
    CHECK(arena.read(out, &flat, 8) && flat == a + b);
    std::printf("  %-34s rax=0x%016llx (window reads across a run, write across unrelated pages)\n", what,
                static_cast<unsigned long long>(rax));
    // Restore the bytes the store overwrote, so the next variant sees the same patterns.
    for (int i = 0; i < 3; ++i) {
        t.view[t.slots[4].second * k64K + k64K - 3 + static_cast<size_t>(i)] =
            pattern(t.slots[4].second, k64K - 3 + static_cast<size_t>(i));
    }
    for (int i = 3; i < 8; ++i) {
        t.view[63 * k64K + static_cast<size_t>(i - 3)] = pattern(63, static_cast<size_t>(i - 3));
    }
    guest::destroy_guest_thread(arena, thread);
    arena.release(out);
    guest::reset_guest_runtime();
}

}  // namespace

int main() {
    std::printf("test_guest_window_e2e\n");
    const char* what = "setup";
    // The real browser pool: 12 GiB at guest VA 4 GiB, identity where the platform allows it. Must
    // be installed before anything creates the default process arena.
    VirtualMemoryArena arena(kDefaultArenaBase, kPoolBytes, ArenaPlacement::PreferIdentity);
    CHECK(arena.valid());
#ifdef __EMSCRIPTEN__
    CHECK(arena.is_identity());
#endif
    win32::set_vmem_arena(&arena);
    gw::set_enabled(true);
    xw::set_policy(xw::PlacementPolicy::PhantomGpuTranslatedCpu);

    // Without the translated window the same sequence does not fit: the budget planner's verdict.
    const xw::PlannedReservation plan[] = {
        {kTitleWindow, kWindowType, 0x4u},
        {kTitleWindow, kWindowType, 0x801u},
        {kPoolView, kWindowType & ~0x400000u, 0x4u},
        {kHeapBudget, 0x2000u, 0x4u},
    };
    CHECK(!xw::plan_fit(plan, kPoolBytes, 2 * GiB, xw::PlacementPolicy::PhantomGpuWindows).fits);
    const xw::FitResult fit = xw::plan_fit(plan, kPoolBytes, 2 * GiB, xw::PlacementPolicy::PhantomGpuTranslatedCpu);
    CHECK(fit.fits && fit.translated_bytes == kTitleWindow);

    Title t;
    if (set_up("title", arena, t)) {
        run("hand-written windowed access", arena, t, sub_handwritten);
#ifdef PLAYGTA6_E2E_LIFTED
        run("lifted sub_140001000 (Windowed)", arena, t, sub_140001000);
#endif
    }
    xmem_reset_for_testing();
    gw::reset_for_testing();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_guest_window_e2e: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_guest_window_e2e: all tests passed\n");
    return 0;
}
