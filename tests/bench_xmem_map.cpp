// Micro-benchmark for the XMem physical-page commit path (R40): the title's page-on-demand heap
// maps one 64 KiB physical page into its reservation when a block first touches it
// (XMemMapPhysicalPages(va, 1, &pfn)) and unmaps it again when the block is freed.
//
//   bench_xmem_map [--full] [--threads N]
//
// Per thread: maps `pages` fresh pages at their slots (bytes zero), unmaps them (bytes move to a
// store), maps them again (bytes move back), each step timed per call. Every remapped page is
// checked for the byte written before its unmap, so the benchmark doubles as a smoke test.
// Without --full (CTest) it runs a few pages per thread.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "os/win32_kernel.h"
#include "os/xbox_scarlett.h"

namespace {

using namespace playgta6::os;
using namespace playgta6::os::scarlett;
using Clock = std::chrono::steady_clock;

struct Times {
    double map_fresh_us = 0;
    double unmap_us = 0;
    double remap_us = 0;
    bool ok = true;
};

Times run_thread(uint64_t base_va, size_t pages, int seed) {
    Times t;
    std::vector<uint64_t> pfns(pages);
    size_t n = pages;
    if (pg6_xbox_XMemAllocatePhysicalPages(kXMemFlagPage64K, &n, pfns.data()) != 1 || n != pages) {
        t.ok = false;
        return t;
    }
    auto* arena = win32::get_vmem_arena();
    auto slot = [&](size_t i) { return arena->translate(base_va + i * kXMemPageSize64K); };
    auto time_each = [&](auto&& op) {
        const auto t0 = Clock::now();
        for (size_t i = 0; i < pages; ++i) {
            op(i);
        }
        return std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / static_cast<double>(pages);
    };
    t.map_fresh_us = time_each([&](size_t i) {
        t.ok = t.ok && pg6_xbox_XMemMapPhysicalPages(slot(i), 1, &pfns[i]) != nullptr;
    });
    for (size_t i = 0; i < pages && t.ok; ++i) {
        slot(i)[i % kXMemPageSize64K] = static_cast<uint8_t>(seed + i);
    }
    t.unmap_us = time_each([&](size_t i) { t.ok = t.ok && pg6_xbox_XMemMapPhysicalPages(slot(i), 1, nullptr) != nullptr; });
    t.remap_us = time_each([&](size_t i) {
        t.ok = t.ok && pg6_xbox_XMemMapPhysicalPages(slot(i), 1, &pfns[i]) != nullptr;
    });
    for (size_t i = 0; i < pages && t.ok; ++i) {
        t.ok = slot(i)[i % kXMemPageSize64K] == static_cast<uint8_t>(seed + i);
    }
    for (size_t i = 0; i < pages; ++i) {
        pg6_xbox_XMemMapPhysicalPages(slot(i), 1, nullptr);
    }
    pg6_xbox_XMemFreePhysicalPages(pages, pfns.data());
    return t;
}

}  // namespace

int main(int argc, char** argv) {
    bool full = false;
    int threads = 1;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--full") == 0) {
            full = true;
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = std::max(1, std::atoi(argv[++i]));
        } else {
            std::fprintf(stderr, "usage: bench_xmem_map [--full] [--threads N]\n");
            return 64;
        }
    }
    const size_t pages = full ? 2048 : 16;
    xmem_reset_for_testing();
    int failures = 0;
    for (int t : {1, threads}) {
        // The title's heap shape: one 64 KiB-page reservation, slots handed out per thread.
        void* reservation = pg6_xbox_XMemVirtualAlloc(nullptr, uint64_t{pages} * t * kXMemPageSize64K,
                                                      0x20400000u | win32::MEM_RESERVE, 0x21u, win32::PAGE_READWRITE);
        if (reservation == nullptr) {
            std::fprintf(stderr, "FAIL: reservation\n");
            return 1;
        }
        const uint64_t base = win32::get_vmem_arena()->guest_address(reservation);
        std::vector<Times> results(static_cast<size_t>(t));
        std::vector<std::thread> workers;
        for (int w = 0; w < t; ++w) {
            workers.emplace_back([&, w] {
                results[static_cast<size_t>(w)] =
                    run_thread(base + uint64_t{pages} * w * kXMemPageSize64K, pages, w * 17 + 3);
            });
        }
        for (auto& w : workers) {
            w.join();
        }
        Times avg;
        for (const Times& r : results) {
            avg.map_fresh_us += r.map_fresh_us / t;
            avg.unmap_us += r.unmap_us / t;
            avg.remap_us += r.remap_us / t;
            avg.ok = avg.ok && r.ok;
        }
        if (!avg.ok) {
            std::fprintf(stderr, "FAIL: mapping or contents (%d thread(s))\n", t);
            ++failures;
        }
        std::printf("%d thread(s), %zu x 64 KiB pages each: map fresh %.1f us, unmap %.1f us, remap %.1f us per call\n",
                    t, pages, avg.map_fresh_us, avg.unmap_us, avg.remap_us);
        // Through XMem, so the next pass's reservation (which may land at the same address) is not
        // shadowed by this one's stale region record.
        xmem_virtual_free(reservation);
        if (threads == 1) {
            break;
        }
    }
    return failures == 0 ? 0 : 1;
}
