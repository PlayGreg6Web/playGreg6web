// Multi-threaded stress of the real VirtualMemoryArena / guest-window paths the GPU thread reads
// while guest threads change them (the host-AV family fixed by #1817, #1818, #1855, #1913): guest
// map_shared / unmap_shared, decommit / MEM_RESET / recommit (incl. the page-reclamation path of
// #1773), mirrored 2 MiB window slot remaps (#1857), against readers using GuestRangeView,
// com::read_guest and arena reads (what Device::flush_cpu_buffers does). Pass criteria: no host fault, and a reader
// sees either "not mapped" or one of the page contents that were ever mapped there, uniform per
// 4 KiB page (every writer fills a page in one locked write). Op schedules are seeded and
// deterministic; thread interleaving is made tight with start barriers. Synthetic data only.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "guest/com_bridge.h"
#include "os/guest_window.h"
#include "os/vmem.h"
#include "os/win32_kernel.h"
#include "os/xbox_scarlett.h"
#include "os/xmem_windows.h"

namespace {

using namespace playgta6::os;
namespace gw = playgta6::os::guest_window;
namespace xw = playgta6::os::xmem_windows;
using namespace playgta6::os::scarlett;
using playgta6::guest::com::GuestRangeView;

constexpr uint64_t k4K = 4096;
constexpr uint64_t k64K = 64ull << 10;
constexpr uint64_t k2M = 2ull << 20;
constexpr uint64_t W = gw::kWindowBase;
constexpr uint32_t kWindowType = 0x20402000u;
constexpr uint32_t kCpuProtect = 0x4u;

std::atomic<int> g_failures{0};
std::atomic<uint64_t> g_reads_ok{0};
std::atomic<uint64_t> g_reads_null{0};

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        if (g_failures.fetch_add(1) < 20) {
            std::fprintf(stderr, "FAIL [test_vmem_window_stress.cpp:%d]: %s\n", line, expr);
        }
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

VirtualMemoryArena& arena() {
    return *win32::get_vmem_arena();
}

// xorshift64*, one per thread, seeded: the op schedule is a function of the seed.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) {}
    uint64_t next() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return s * 0x2545F4914F6CDD1Dull;
    }
    uint32_t below(uint32_t n) { return static_cast<uint32_t>((next() >> 33) % n); }
};

// Spin barrier so every thread starts its loop together.
struct StartGate {
    std::atomic<int> arrived{0};
    int total;
    explicit StartGate(int n) : total(n) {}
    void wait() {
        arrived.fetch_add(1);
        while (arrived.load() < total) {
            std::this_thread::yield();
        }
    }
};

// Runs `mutate` (fixed op counts) on `mutators` threads while `read` loops on `readers` threads.
template <typename M, typename R>
void run(int mutators, int readers, M mutate, R read, uint64_t seed) {
    StartGate gate(mutators + readers);
    std::atomic<int> live{mutators};
    std::vector<std::thread> threads;
    for (int m = 0; m < mutators; ++m) {
        threads.emplace_back([&, m] {
            Rng rng(seed * 131 + static_cast<uint64_t>(m));
            gate.wait();
            mutate(m, rng);
            live.fetch_sub(1);
        });
    }
    for (int r = 0; r < readers; ++r) {
        threads.emplace_back([&, r] {
            Rng rng(seed * 977 + 1000 + static_cast<uint64_t>(r));
            gate.wait();
            while (live.load() > 0) {
                read(r, rng);
            }
        });
    }
    for (std::thread& t : threads) {
        t.join();
    }
}

// Every 4 KiB page must be uniform, and its byte 0 or one of `allowed`.
bool pages_ok(const uint8_t* p, size_t len, const uint8_t* allowed, size_t n_allowed) {
    for (size_t off = 0; off + k4K <= len; off += k4K) {
        const uint8_t b = p[off];
        bool known = b == 0;
        for (size_t i = 0; i < n_allowed && !known; ++i) {
            known = allowed[i] == b;
        }
        if (!known) {
            return false;
        }
        for (size_t i = 1; i < k4K; i += 61) {
            if (p[off + i] != b) {
                return false;
            }
        }
        if (p[off + k4K - 1] != b) {
            return false;
        }
    }
    return true;
}

constexpr uint8_t kFills[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};

// Phase 1: arena decommit / recommit / MEM_RESET / rewrite churn (many pages: exercises the #1773
// reclaimed-page bookkeeping) with readers over the flat range, the arena's checked read and,
// when `window`, a translated window range onto the same pages whose slots are being remapped.
void phase_arena_churn(uint64_t seed, bool window) {
    constexpr uint64_t kBlocks = 8;  // 8 x 64 KiB = 128 pages
    const uint64_t va = arena().virtual_alloc(0, kBlocks * k64K, kMemReserve | kMemCommit, PageProtect::ReadWrite);
    CHECK(va != 0);
    if (va == 0) {
        return;
    }
    if (window) {
        gw::reset_for_testing();
        gw::set_enabled(true);
        for (uint64_t b = 0; b < kBlocks; ++b) {
            CHECK(gw::map_slot(W + b * k64K, arena().translate(va + b * k64K)));
        }
    }
    run(
        2, 4,
        [&](int m, Rng& rng) {
            std::vector<uint8_t> buf(k4K);
            for (int i = 0; i < 4000; ++i) {
                // Ranges overlap between mutators (guest threads sharing a heap); every op is atomic
                // under the arena lock.
                const uint64_t page = rng.below(static_cast<uint32_t>(kBlocks * 16 - 8));
                const uint64_t count = 1 + rng.below(8);
                const uint64_t a = va + page * k4K;
                switch (rng.below(5)) {
                    case 0: arena().decommit(a, count * k4K); break;
                    case 1: arena().commit(a, count * k4K, PageProtect::ReadWrite); break;
                    case 2: arena().virtual_alloc(a, count * k4K, kMemReset, PageProtect::ReadWrite); break;
                    case 3: arena().virtual_free(a, count * k4K, kMemDecommit); break;
                    default: {
                        std::memset(buf.data(), kFills[rng.below(sizeof(kFills))], k4K);
                        for (uint64_t c = 0; c < count; ++c) {
                            arena().write(a + c * k4K, buf.data(), k4K);  // fails on a decommitted page: fine
                        }
                    }
                }
                if (window && (i % 16) == 0) {
                    const uint64_t b = rng.below(static_cast<uint32_t>(kBlocks));
                    if (m == (b & 1)) {  // one owner per slot
                        if (rng.below(2)) {
                            gw::unmap_slot(W + b * k64K);
                        }
                        gw::map_slot(W + b * k64K, arena().translate(va + b * k64K));
                    }
                }
            }
        },
        [&](int r, Rng& rng) {
            const uint64_t page = rng.below(static_cast<uint32_t>(kBlocks * 16 - 16));
            const size_t len = k4K * (1 + rng.below(16));
            const uint64_t a = va + page * k4K;
            if (r == 0 && window) {
                const GuestRangeView view(W + (a - va), len);
                if (view.data() == nullptr) {
                    ++g_reads_null;
                    return;
                }
                CHECK(pages_ok(view.data(), len, kFills, sizeof(kFills)));
                ++g_reads_ok;
            } else if (r == 1) {
                std::vector<uint8_t> copy(len);
                if (arena().read(a, copy.data(), len)) {
                    CHECK(pages_ok(copy.data(), len, kFills, sizeof(kFills)));
                    ++g_reads_ok;
                }
            } else {
                const GuestRangeView view(a, len);
                if (view.data() == nullptr) {
                    ++g_reads_null;
                    return;
                }
                CHECK(pages_ok(view.data(), len, kFills, sizeof(kFills)));
                ++g_reads_ok;
            }
        },
        seed);
    if (window) {
        gw::reset_for_testing();
    }
    CHECK(arena().release(va));
}

// Phase 2: shared page views mapped / unmapped / re-pointed by guest threads (each owns its slots,
// as the XMem lock serialises a slot's owner) with readers on the flat VA and, when `window`, on a
// translated window slot over the view (the GPU thread's route).
void phase_shared_views(uint64_t seed, bool window) {
    if (!arena().has_shared_zone()) {
        return;
    }
    constexpr uint64_t kSlots = 4;
    const uint64_t va = arena().reserve_shared(kSlots * k64K);
    void* page_a = create_shared_page(k64K);
    void* page_b = create_shared_page(k64K);
    CHECK(va != 0 && page_a != nullptr && page_b != nullptr);
    if (va == 0 || page_a == nullptr || page_b == nullptr) {
        return;
    }
    uint8_t* ha = view_shared_page(page_a, k64K);
    uint8_t* hb = view_shared_page(page_b, k64K);
    CHECK(ha != nullptr && hb != nullptr);
    if (ha == nullptr || hb == nullptr) {
        return;
    }
    std::memset(ha, 0xA5, k64K);
    std::memset(hb, 0x5A, k64K);
    unview_shared_page(ha, k64K);
    unview_shared_page(hb, k64K);
    if (window) {
        gw::reset_for_testing();
        gw::set_enabled(true);
        for (uint64_t s = 0; s < kSlots; ++s) {
            CHECK(gw::map_slot(W + s * k64K, arena().translate(va + s * k64K)));
        }
    }
    const uint8_t allowed[] = {0xA5, 0x5A};
    run(
        2, 4,
        [&](int m, Rng& rng) {
            for (int i = 0; i < 2500; ++i) {
                const uint64_t slot = static_cast<uint64_t>(m) * 2 + rng.below(2);
                const uint64_t a = va + slot * k64K;
                switch (rng.below(3)) {
                    case 0: CHECK(arena().map_shared(a, page_a, k64K)); break;
                    case 1: CHECK(arena().map_shared(a, page_b, k64K)); break;  // re-point or fresh
                    default: CHECK(arena().unmap_shared(a, k64K)); break;
                }
                if (window && rng.below(8) == 0) {
                    gw::map_slot(W + slot * k64K, arena().translate(a));  // remap the slot (same host)
                }
            }
        },
        [&](int r, Rng& rng) {
            const uint64_t slot = rng.below(kSlots);
            const uint64_t a = va + slot * k64K;
            const bool via_window = window && (r & 1);
            const GuestRangeView view(via_window ? W + slot * k64K : a, k64K);
            if (view.data() == nullptr) {
                ++g_reads_null;
                return;
            }
            const uint8_t first = view.data()[0];
            CHECK(first == 0xA5 || first == 0x5A);
            CHECK(pages_ok(view.data(), k64K, allowed, 2));
            // A held view keeps its pages mapped to the end: the last page matches the first.
            CHECK(view.data()[k64K - 1] == first);
            ++g_reads_ok;
        },
        seed);
    if (window) {
        gw::reset_for_testing();
    }
    for (uint64_t s = 0; s < kSlots; ++s) {
        CHECK(arena().unmap_shared(va + s * k64K, k64K));
    }
    CHECK(arena().release(va));
    release_shared_page(page_a);
    release_shared_page(page_b);
}

std::vector<uint8_t>& read_buf() {
    thread_local std::vector<uint8_t> buf(k2M);
    return buf;
}

// Phase 3: a mirrored 2 MiB MEM_PHYSICAL reservation whose window slots (all 32 per page, #1857)
// are mapped / unmapped / re-pointed between two physical pages while readers view the window.
void phase_mirrored_2m(uint64_t seed) {
    xmem_reset_for_testing();
    gw::reset_for_testing();
    gw::set_enabled(true);
    xw::set_policy(xw::PlacementPolicy::PhantomGpuTranslatedCpu);
    void* cpu = pg6_xbox_XMemVirtualAlloc(nullptr, 16ull << 30, kWindowType, 0x21u, kCpuProtect);
    void* pool = pg6_xbox_XMemVirtualAlloc(nullptr, 2ull << 30, kWindowType, 0x23u, kCpuProtect);
    void* mirror = pg6_xbox_XMemVirtualAlloc(nullptr, 4 * k2M, kWindowType, 0x23u, kCpuProtect);
    CHECK(cpu != nullptr && pool != nullptr && mirror != nullptr);
    size_t count = 2;
    uint64_t pfn[2] = {};
    const bool alloc_ok = pg6_xbox_XMemAllocatePhysicalPages(0x20000380u, &count, pfn) == 1 && count == 2;
    CHECK(alloc_ok);
    const uint64_t wm = mirror != nullptr ? arena().guest_address(mirror) : 0;
    if (cpu == nullptr || pool == nullptr || mirror == nullptr || !alloc_ok || wm < gw::kRuntimeRangeBase) {
        xmem_reset_for_testing();
        gw::reset_for_testing();
        return;
    }
    // Stamp both physical pages once (through the mirror), then only remap: content is constant.
    const uint8_t stamp[2] = {0x3C, 0xC3};
    std::vector<uint8_t> chunk(k2M);
    for (int p = 0; p < 2; ++p) {
        CHECK(pg6_xbox_XMemMapPhysicalPages(mirror, 1, &pfn[p]) == mirror);
        std::memset(chunk.data(), stamp[p], chunk.size());
        CHECK(gw::copy_to_guest(wm, chunk.data(), chunk.size()));
        CHECK(pg6_xbox_XMemMapPhysicalPages(mirror, 1, nullptr) == mirror);
    }
    // More readers than cores: a reader preempted mid-copy is what lets a remap overlap a copy.
    const int readers = static_cast<int>(std::clamp(2 * std::thread::hardware_concurrency(), 8u, 32u));
    run(
        1, readers,
        [&](int, Rng& rng) {
            for (int i = 0; i < 500; ++i) {
                switch (rng.below(3)) {
                    case 0: CHECK(pg6_xbox_XMemMapPhysicalPages(mirror, 1, &pfn[0]) == mirror); break;
                    case 1: CHECK(pg6_xbox_XMemMapPhysicalPages(mirror, 1, &pfn[1]) == mirror); break;
                    default: CHECK(pg6_xbox_XMemMapPhysicalPages(mirror, 1, nullptr) == mirror); break;
                }
            }
        },
        [&](int, Rng& rng) {
            const uint64_t slot = rng.below(32);
            if (rng.below(2) == 0) {
                // Host-side copies (read_guest: the GPU thread, audio, IO) re-check the slot under the
                // arena lock like a view does.
                // Half of them copy the whole 2 MiB page (one long memcpy that a remap would
                // otherwise overlap: zeros, or a fault on the decommitted page).
                const bool whole = rng.below(2) == 0;
                const size_t n = whole ? k2M : k64K;
                std::vector<uint8_t>& buf = read_buf();
                if (!playgta6::guest::com::read_guest(wm + (whole ? 0 : slot * k64K), buf.data(), n)) {
                    ++g_reads_null;
                    return;
                }
                // A copy is not atomic across slots (a remap between two slots' pieces is fine, as on
                // the console), but every 64 KiB slot it copied is one whole stamped page.
                bool exact = true;
                for (size_t base = 0; base < n && exact; base += k64K) {
                    exact = buf[base] == stamp[0] || buf[base] == stamp[1];
                    for (size_t i = base; i < base + k64K && exact; ++i) {
                        exact = buf[i] == buf[base];
                    }
                }
                CHECK(exact);
                ++g_reads_ok;
                return;
            }
            const GuestRangeView view(wm + slot * k64K, k64K);
            if (view.data() == nullptr) {
                ++g_reads_null;
                return;
            }
            // A remap commits the slot (zero) and copies the new page's bytes in before it shows
            // the slot again, and the view re-checks the slot under the arena lock, so a view
            // shows one whole page: every byte is the stamp of the page mapped there, never zero
            // and never a mix of the two pages.
            const uint8_t first = view.data()[0];
            bool exact = first == stamp[0] || first == stamp[1];
            for (size_t i = 0; i < k64K && exact; i += 7) {
                exact = view.data()[i] == first;
            }
            exact = exact && view.data()[k64K - 1] == first;
            CHECK(exact);
            ++g_reads_ok;
        },
        seed);
    CHECK(pg6_xbox_XMemMapPhysicalPages(mirror, 1, nullptr) == mirror);

    // Phase 4: the guest's own loads (what lifted code does: the slot's delta from the window
    // table, then a plain load at va + delta; no helper, no lock) while the mirrored page is only
    // ever re-pointed from one physical page to the other, never unmapped. A remap of a mapped
    // slot keeps the slot's memory committed and shown, and fills it in place, so every load sees
    // a stamp byte: never the trap delta (a guest fault), never zero (the page committed again but
    // not filled yet). Before the in-place remap both happened (docs/XMEM_REMAP.md).
    CHECK(pg6_xbox_XMemMapPhysicalPages(mirror, 1, &pfn[0]) == mirror);
    std::atomic<uint64_t> raw_traps{0};
    std::atomic<uint64_t> raw_zeros{0};
    std::atomic<uint64_t> raw_loads{0};
    run(
        1, readers,
        [&](int, Rng& rng) {
            for (int i = 0; i < 500; ++i) {
                CHECK(pg6_xbox_XMemMapPhysicalPages(mirror, 1, &pfn[rng.below(2)]) == mirror);
            }
        },
        [&](int, Rng& rng) {
            uint64_t traps = 0;
            uint64_t zeros = 0;
            for (int i = 0; i < 256; ++i) {
                const uint64_t va = wm + (rng.next() % k2M);
                const int64_t delta = std::atomic_ref<int64_t>(
                    __playgta6_window_table[(va - gw::kWindowBase) >> gw::kSlotShift]).load(std::memory_order_relaxed);
                if (static_cast<uint64_t>(delta) == gw::kTrapDelta) {
                    ++traps;  // the guest's load would fault here
                    continue;
                }
                const uint8_t b = *reinterpret_cast<const volatile uint8_t*>(static_cast<uintptr_t>(va + delta));
                if (b == 0) {
                    ++zeros;
                } else {
                    CHECK(b == stamp[0] || b == stamp[1]);
                }
            }
            raw_traps += traps;
            raw_zeros += zeros;
            raw_loads += 256;
        },
        seed);
    std::fprintf(stderr, "  raw guest loads: %llu, trapped %llu, zero %llu\n",
                 static_cast<unsigned long long>(raw_loads.load()), static_cast<unsigned long long>(raw_traps.load()),
                 static_cast<unsigned long long>(raw_zeros.load()));
    CHECK(raw_traps.load() == 0);
    CHECK(raw_zeros.load() == 0);
    g_reads_ok += raw_loads.load() - raw_traps.load() - raw_zeros.load();

    CHECK(pg6_xbox_XMemMapPhysicalPages(mirror, 1, nullptr) == mirror);
    CHECK(xmem_virtual_free(mirror));
    CHECK(xmem_virtual_free(pool));
    CHECK(xmem_virtual_free(cpu));
    CHECK(pg6_xbox_XMemFreePhysicalPages(1, &pfn[0]) == 1);
    CHECK(pg6_xbox_XMemFreePhysicalPages(1, &pfn[1]) == 1);
    xmem_reset_for_testing();
    gw::reset_for_testing();
}

}  // namespace

int main() {
    const auto t0 = std::chrono::steady_clock::now();
    for (uint64_t seed = 1; seed <= 3; ++seed) {
        std::fprintf(stderr, "seed %llu\n", static_cast<unsigned long long>(seed));
        phase_arena_churn(seed, false);
        phase_arena_churn(seed, true);
        phase_shared_views(seed, false);
        phase_shared_views(seed, true);
        phase_mirrored_2m(seed);
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CHECK(g_reads_ok.load() > 0);
    std::printf("vmem window stress: %llu reads ok, %llu unmapped, %.1fs\n",
                static_cast<unsigned long long>(g_reads_ok.load()),
                static_cast<unsigned long long>(g_reads_null.load()), secs);
    if (g_failures.load() != 0) {
        std::fprintf(stderr, "test_vmem_window_stress: %d failure(s)\n", g_failures.load());
        return 1;
    }
    std::printf("test_vmem_window_stress OK\n");
    return 0;
}
