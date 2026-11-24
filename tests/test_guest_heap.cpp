// Guest allocator service (runtime/os/guest_heap.h) and its users: the XMem shims
// (xbox_scarlett) and the CRT malloc family (ucrt_bridge). Every pointer must be a valid guest
// address of its arena, on the identity process arena and on a deliberately offset arena.
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "abi/playgta6_abi.h"
#include "os/guest_heap.h"
#include "os/ucrt_bridge.h"
#include "os/vmem.h"
#include "os/win32_kernel.h"
#include "os/xbox_scarlett.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_guest_heap.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6::os;
using namespace playgta6::os::scarlett;

constexpr uint64_t kKiB = 1024;
constexpr uint64_t kMiB = 1024 * kKiB;
constexpr uint64_t kGiB = 1024 * kMiB;

#if defined(__EMSCRIPTEN__)
constexpr uint64_t kBigBlock = 64 * kMiB;
constexpr uint64_t kOffsetArenaSize = 1 * kGiB;
#else
constexpr uint64_t kBigBlock = 512 * kMiB;
constexpr uint64_t kOffsetArenaSize = 4 * kGiB;
#endif

// va is a committed, writable guest address of `arena`, and host == guest_base + va.
bool valid_guest_range(VirtualMemoryArena& arena, uint64_t va, uint64_t len) {
    if (va < arena.space_base() || va + len > arena.space_base() + arena.space_size()) {
        return false;
    }
    if (arena.host_ptr(va, static_cast<size_t>(len)) != arena.translate(va)) {
        return false;
    }
    return arena.guest_address(arena.translate(va)) == va;
}

uint8_t pattern_byte(uint64_t seed, uint64_t i) {
    return static_cast<uint8_t>((seed * 131u + i * 7u + (i >> 8)) & 0xFFu);
}

void fill_pattern(GuestHeap& heap, uint64_t va, uint64_t len, uint64_t seed) {
    uint8_t* p = heap.to_host(va);
    // Large blocks: touch sparsely so the test stays fast.
    const uint64_t step = len > 4 * kMiB ? 4093 : 1;
    for (uint64_t i = 0; i < len; i += step) {
        p[i] = pattern_byte(seed, i);
    }
    if (len != 0) {
        p[len - 1] = pattern_byte(seed, len - 1);
    }
}

// Checks the first `len` bytes of a block that was filled with fill_pattern(fill_len).
bool check_pattern(GuestHeap& heap, uint64_t va, uint64_t len, uint64_t seed,
                   uint64_t fill_len = 0) {
    const uint8_t* p = heap.to_host(va);
    if (fill_len == 0) {
        fill_len = len;
    }
    const uint64_t step = fill_len > 4 * kMiB ? 4093 : 1;
    for (uint64_t i = 0; i < len; i += step) {
        if (p[i] != pattern_byte(seed, i)) {
            return false;
        }
    }
    return len != fill_len || len == 0 || p[len - 1] == pattern_byte(seed, len - 1);
}

// ---------------------------------------------------------------------------
// Core allocator behaviour, run on a heap bound to any arena.
// ---------------------------------------------------------------------------

void test_basic(GuestHeap& heap) {
    VirtualMemoryArena& arena = heap.arena();

    HeapStatus st = HeapStatus::Ok;
    const uint64_t a = heap.alloc(100, 16, HeapFamily::Crt, 0, &st);
    CHECK(a != 0 && st == HeapStatus::Ok);
    CHECK(valid_guest_range(arena, a, 100));
    CHECK(heap.owns(a));
    CHECK(heap.usable_size(a) == 100);
    CHECK(heap.to_host(a) == arena.translate(a));
    CHECK(heap.to_guest(heap.to_host(a)) == a);
    CHECK(heap.to_host(0) == nullptr && heap.to_guest(nullptr) == 0);

    // malloc(0): unique non-null pointers, size 0.
    const uint64_t z1 = heap.alloc(0);
    const uint64_t z2 = heap.alloc(0);
    CHECK(z1 != 0 && z2 != 0 && z1 != z2);
    CHECK(heap.usable_size(z1) == 0);

    // free(0) is a no-op, an unknown pointer is diagnosed, a double free is diagnosed.
    CHECK(heap.free(0));
    const uint64_t invalid_before = heap.stats().invalid_frees;
    CHECK(heap.free(a));
    CHECK(!heap.owns(a));
    CHECK(heap.usable_size(a) == GuestHeap::kInvalidSize);
    CHECK(!heap.free(a, &st) && st == HeapStatus::InvalidPointer);
    CHECK(!heap.free(a + 8, &st) && st == HeapStatus::InvalidPointer);
    CHECK(!heap.free(arena.space_base() + 12345 * kPageSize));
    CHECK(heap.stats().invalid_frees == invalid_before + 3);
    CHECK(heap.free(z1) && heap.free(z2));

    // Blocks never overlap and keep their contents.
    std::vector<uint64_t> blocks;
    std::vector<uint64_t> sizes;
    for (uint64_t i = 0; i < 200; ++i) {
        const uint64_t size = 1 + (i * 37) % 3000;
        const uint64_t va = heap.alloc(size);
        CHECK(va != 0);
        fill_pattern(heap, va, size, i);
        blocks.push_back(va);
        sizes.push_back(size);
    }
    for (size_t i = 0; i < blocks.size(); ++i) {
        CHECK(check_pattern(heap, blocks[i], sizes[i], i));
    }
    for (size_t i = 0; i < blocks.size(); i += 2) {
        CHECK(heap.free(blocks[i]));
    }
    for (size_t i = 1; i < blocks.size(); i += 2) {
        CHECK(check_pattern(heap, blocks[i], sizes[i], i));
        CHECK(heap.free(blocks[i]));
    }
    CHECK(heap.stats().live_blocks == 0);
    CHECK(heap.stats().live_bytes == 0);

    // calloc zero fills recycled memory and rejects overflow.
    const uint64_t dirty = heap.alloc(4096);
    std::memset(heap.to_host(dirty), 0xAB, 4096);
    CHECK(heap.free(dirty));
    const uint64_t zeroed = heap.calloc(64, 64);
    CHECK(zeroed != 0);
    const uint8_t* zp = heap.to_host(zeroed);
    bool all_zero = true;
    for (size_t i = 0; i < 4096; ++i) {
        all_zero = all_zero && zp[i] == 0;
    }
    CHECK(all_zero);
    CHECK(heap.free(zeroed));
    CHECK(heap.calloc(~uint64_t{0} / 2, 4, 16, HeapFamily::Crt, &st) == 0 &&
          st == HeapStatus::OutOfMemory);
    CHECK(heap.alloc(~uint64_t{0}) == 0);
    CHECK(heap.alloc(64, 24, HeapFamily::Crt, 0, &st) == 0 && st == HeapStatus::InvalidParameter);
}

void test_alignment_matrix(GuestHeap& heap) {
    VirtualMemoryArena& arena = heap.arena();
    const uint64_t sizes[] = {1, 15, 16, 17, 100, 4096, 70000, 130000, 3 * kMiB};
    const uint64_t aligns[] = {1, 16, 32, 64, 256, 4096, 65536, 131072, 2 * kMiB};
    std::vector<uint64_t> live;
    for (uint64_t size : sizes) {
        for (uint64_t align : aligns) {
            const uint64_t va = heap.alloc(size, align);
            CHECK(va != 0);
            if (va == 0) {
                continue;
            }
            CHECK((va & (std::max<uint64_t>(align, 16) - 1)) == 0);
            CHECK(valid_guest_range(arena, va, size));
            CHECK(heap.usable_size(va) == size);
            HeapBlockInfo info;
            CHECK(heap.block_info(va, &info) && info.alignment >= std::max<uint64_t>(align, 16));
            fill_pattern(heap, va, size, va);
            live.push_back(va);
        }
    }
    for (uint64_t va : live) {
        CHECK(check_pattern(heap, va, heap.usable_size(va), va));
        CHECK(heap.free(va));
    }
    CHECK(heap.stats().live_blocks == 0);
}

void test_realloc(GuestHeap& heap) {
    HeapStatus st = HeapStatus::Ok;
    // realloc(0, n) allocates; realloc(p, 0) frees and returns 0 without an error.
    uint64_t p = heap.realloc(0, 64);
    CHECK(p != 0 && heap.usable_size(p) == 64);
    fill_pattern(heap, p, 64, 1);
    CHECK(heap.realloc(p, 0, 0, HeapFamily::Crt, 0, &st) == 0 && st == HeapStatus::Ok);
    CHECK(!heap.owns(p));
    CHECK(heap.realloc(p, 10, 0, HeapFamily::Crt, 0, &st) == 0 && st == HeapStatus::InvalidPointer);

    // Growth and shrink across the small, large and back to small backends keep the contents.
    const uint64_t steps[] = {64,     200,    5000,  100000,   140000, 3 * kMiB,
                              kMiB,   300000, 70000, 100,      40,     1,
                              9 * kMiB, 20 * kMiB, 129 * kKiB, 16};
    p = heap.alloc(steps[0]);
    uint64_t current = steps[0];
    fill_pattern(heap, p, current, 77);
    for (size_t i = 1; i < sizeof(steps) / sizeof(steps[0]); ++i) {
        const uint64_t next = steps[i];
        const uint64_t q = heap.realloc(p, next);
        CHECK(q != 0);
        if (q == 0) {
            return;
        }
        p = q;
        CHECK(heap.usable_size(p) == next);
        CHECK(valid_guest_range(heap.arena(), p, next));
        CHECK(check_pattern(heap, p, std::min(current, next), 77, current));
        current = next;
        fill_pattern(heap, p, current, 77);
    }
    CHECK(heap.free(p));

    // realloc keeps a block's alignment; _aligned_realloc-style realloc can raise it.
    uint64_t al = heap.alloc(1000, 4096);
    fill_pattern(heap, al, 1000, 5);
    al = heap.realloc(al, 50000);
    CHECK(al != 0 && (al & 4095) == 0 && check_pattern(heap, al, 1000, 5));
    al = heap.realloc(al, 60000, 65536);
    CHECK(al != 0 && (al & 65535) == 0 && check_pattern(heap, al, 1000, 5));
    CHECK(heap.free(al));

    // zeroing growth.
    uint64_t g = heap.alloc(100);
    std::memset(heap.to_host(g), 0xCD, 100);
    g = heap.realloc(g, 5000, 0, HeapFamily::Crt, kHeapZeroMemory);
    const uint8_t* gp = heap.to_host(g);
    bool ok = true;
    for (size_t i = 0; i < 100; ++i) {
        ok = ok && gp[i] == 0xCD;
    }
    for (size_t i = 100; i < 5000; ++i) {
        ok = ok && gp[i] == 0;
    }
    CHECK(ok);

    // _expand: in place or nothing.
    CHECK(heap.expand(g, 5000) == g);
    CHECK(heap.expand(g, 4000) == g && heap.usable_size(g) == 4000);
    CHECK(heap.expand(g, 0) == 0);
    CHECK(heap.free(g));
    CHECK(heap.expand(g, 10) == 0);
}

void test_large_blocks_return_memory(GuestHeap& heap) {
    VirtualMemoryArena& arena = heap.arena();
    // Warm the small backend so segment growth is not mistaken for a leak.
    const uint64_t warm = heap.alloc(64);
    const uint64_t committed0 = arena.committed_bytes();
    const uint64_t reserved0 = arena.reserved_bytes();
    const uint64_t heap_committed0 = heap.stats().committed_bytes;

    const uint64_t big = heap.alloc(kBigBlock);
    CHECK(big != 0);
    CHECK(heap.stats().large_live_blocks == 1);
    CHECK(heap.stats().large_live_bytes == kBigBlock);
    CHECK(arena.committed_bytes() >= committed0 + kBigBlock);
    heap.to_host(big)[0] = 1;
    heap.to_host(big)[kBigBlock - 1] = 2;
    CHECK(heap.free(big));
    CHECK(arena.committed_bytes() == committed0);
    CHECK(arena.reserved_bytes() == reserved0);
    CHECK(heap.stats().committed_bytes == heap_committed0);
    CHECK(heap.stats().large_live_blocks == 0);
    CHECK(heap.stats().large_peak_bytes >= kBigBlock);

    // Shrinking a large block decommits the tail; growing again recommits (zeroed).
    const uint64_t blk = heap.alloc(8 * kMiB);
    std::memset(heap.to_host(blk), 0x5A, 8 * kMiB);
    const uint64_t committed_full = arena.committed_bytes();
    CHECK(heap.realloc(blk, 2 * kMiB) == blk);
    CHECK(arena.committed_bytes() == committed_full - 6 * kMiB);
    CHECK(heap.realloc(blk, 6 * kMiB) == blk);
    CHECK(arena.committed_bytes() == committed_full - 2 * kMiB);
    CHECK(check_pattern(heap, blk, 2 * kMiB, 0) || heap.to_host(blk)[0] == 0x5A);
    CHECK(heap.to_host(blk)[6 * kMiB - 1] == 0);
    CHECK(heap.free(blk));
    CHECK(arena.committed_bytes() == committed0);

    // An over-aligned large block lands on its alignment and is released with its reservation.
    const uint64_t huge_align = heap.alloc(kMiB, 4 * kMiB);
    CHECK(huge_align != 0 && (huge_align & (4 * kMiB - 1)) == 0);
    CHECK(heap.free(huge_align));
    CHECK(arena.committed_bytes() == committed0);
    CHECK(arena.reserved_bytes() == reserved0);
    CHECK(heap.free(warm));
}

void test_limits(GuestHeap& heap) {
    HeapStatus st = HeapStatus::Ok;
    const uint64_t base_live = heap.stats().live_bytes;
    heap.set_soft_limit(base_live + 4 * kMiB);
    CHECK(heap.soft_limit() == base_live + 4 * kMiB);
    std::vector<uint64_t> blocks;
    for (;;) {
        const uint64_t va = heap.alloc(300 * kKiB, 16, HeapFamily::XMem, 0, &st);
        if (va == 0) {
            break;
        }
        blocks.push_back(va);
        if (blocks.size() > 1000) {
            break;
        }
    }
    CHECK(st == HeapStatus::OutOfMemory);
    CHECK(blocks.size() == 13);  // 13 * 300 KiB <= 4 MiB < 14 * 300 KiB
    CHECK(heap.stats().failed_allocs >= 1);
    // Small allocations fail the same way, and the failure leaves nothing behind.
    const uint64_t live_at_limit = heap.stats().live_bytes;
    std::vector<uint64_t> filler;
    for (uint64_t va = heap.alloc(64 * kKiB); va != 0; va = heap.alloc(64 * kKiB)) {
        filler.push_back(va);
    }
    CHECK(heap.stats().live_bytes <= base_live + 4 * kMiB);
    CHECK(heap.alloc(base_live + 8 * kMiB) == 0);
    CHECK(heap.stats().live_bytes >= live_at_limit);
    // Realloc growth beyond the limit fails and keeps the old block.
    CHECK(heap.realloc(blocks[0], 5 * kMiB, 0, HeapFamily::XMem, 0, &st) == 0 &&
          st == HeapStatus::OutOfMemory);
    CHECK(heap.usable_size(blocks[0]) == 300 * kKiB);
    for (uint64_t va : blocks) {
        CHECK(heap.free(va));
    }
    for (uint64_t va : filler) {
        CHECK(heap.free(va));
    }
    heap.set_soft_limit(0);
    // Freed memory (and every 64 KiB block of the loop above) is usable again.
    const uint64_t again = heap.alloc(300 * kKiB);
    CHECK(again != 0);
    CHECK(heap.free(again));
    CHECK(heap.stats().family[static_cast<size_t>(HeapFamily::XMem)].peak_bytes >=
          13 * 300 * kKiB);
}

void test_family_accounting_and_cross_free(GuestHeap& heap) {
    const GuestHeapStats s0 = heap.stats();
    const uint64_t c = heap.alloc(1000, 16, HeapFamily::Crt);
    const uint64_t x = heap.alloc(2000, 64, HeapFamily::XMem);
    const uint64_t big = heap.alloc(1 * kMiB, 16, HeapFamily::XMem);
    GuestHeapStats s1 = heap.stats();
    CHECK(s1.family[0].live_bytes == s0.family[0].live_bytes + 1000);
    CHECK(s1.family[1].live_bytes == s0.family[1].live_bytes + 2000 + kMiB);
    CHECK(s1.live_blocks == s0.live_blocks + 3);
    CHECK(s1.large_live_blocks == s0.large_live_blocks + 1);
    HeapBlockInfo info;
    CHECK(heap.block_info(x, &info) && info.family == HeapFamily::XMem && info.alignment == 64);
    // One family: every block frees through the same entry point, whatever produced it.
    CHECK(heap.free(x) && heap.free(c) && heap.free(big));
    GuestHeapStats s2 = heap.stats();
    CHECK(s2.live_bytes == s0.live_bytes);
    CHECK(s2.total_frees == s0.total_frees + 3);
    CHECK(s2.peak_bytes >= s0.live_bytes + 3000 + kMiB);
}

void test_threads(GuestHeap& heap) {
    constexpr int kThreads = 8;
#if defined(__EMSCRIPTEN__)
    constexpr int kIterations = 400;
#else
    constexpr int kIterations = 3000;
#endif
    std::atomic<int> errors{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&heap, &errors, t] {
            uint64_t rng = 0x9E3779B97F4A7C15ull * static_cast<uint64_t>(t + 1);
            auto next = [&rng] {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                return rng;
            };
            struct Live {
                uint64_t va;
                uint64_t size;
                uint64_t seed;
            };
            std::vector<Live> live;
            for (int i = 0; i < kIterations; ++i) {
                const uint64_t r = next();
                if (live.size() < 48 && (r & 3) != 0) {
                    const uint64_t kind = (r >> 8) % 100;
                    const uint64_t size = kind < 80 ? 1 + (r >> 16) % 2000
                                          : kind < 97 ? 1 + (r >> 16) % 200000
                                                      : 1 + (r >> 16) % (3 * kMiB);
                    const uint64_t align = (r & 0x40) != 0 ? 16 : (uint64_t{16} << ((r >> 20) % 8));
                    const uint64_t va = heap.alloc(size, align);
                    if (va == 0 || (va & (align - 1)) != 0) {
                        { ++errors; std::fprintf(stderr, "thread error at line %d\n", __LINE__); }
                        continue;
                    }
                    const uint64_t seed = (static_cast<uint64_t>(t) << 32) | static_cast<uint64_t>(i);
                    fill_pattern(heap, va, size, seed);
                    live.push_back({va, size, seed});
                } else if (!live.empty()) {
                    const size_t idx = static_cast<size_t>((r >> 24) % live.size());
                    Live& l = live[idx];
                    if (!check_pattern(heap, l.va, l.size, l.seed)) {
                        { ++errors; std::fprintf(stderr, "thread error at line %d\n", __LINE__); }
                    }
                    if ((r & 0x100) != 0) {
                        const uint64_t new_size = 1 + (r >> 32) % 70000;
                        const uint64_t q = heap.realloc(l.va, new_size);
                        if (q == 0 || !check_pattern(heap, q, std::min(l.size, new_size), l.seed, l.size)) {
                            { ++errors; std::fprintf(stderr, "thread error at line %d\n", __LINE__); }
                            continue;
                        }
                        l.va = q;
                        l.size = new_size;
                        fill_pattern(heap, q, new_size, l.seed);
                    } else {
                        if (!heap.free(l.va)) {
                            { ++errors; std::fprintf(stderr, "thread error at line %d\n", __LINE__); }
                        }
                        live[idx] = live.back();
                        live.pop_back();
                    }
                }
            }
            for (const Live& l : live) {
                if (!check_pattern(heap, l.va, l.size, l.seed) || !heap.free(l.va)) {
                    { ++errors; std::fprintf(stderr, "thread error at line %d\n", __LINE__); }
                }
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    CHECK(errors.load() == 0);
    CHECK(heap.stats().live_blocks == 0);
    CHECK(heap.stats().live_bytes == 0);
}

void run_core_suite(GuestHeap& heap) {
    test_basic(heap);
    test_alignment_matrix(heap);
    test_realloc(heap);
    test_large_blocks_return_memory(heap);
    test_limits(heap);
    test_family_accounting_and_cross_free(heap);
    test_threads(heap);
}

void test_core_on_process_arena() {
    GuestHeap heap(*win32::get_vmem_arena());
    run_core_suite(heap);
    heap.release_all();
}

void test_core_on_offset_arena() {
    // Anywhere placement: the host address is picked by the OS, guest VAs stay >= 4 GiB
    // (guest_base != 0 natively); on wasm the arena is a pool region with a positive base.
    VirtualMemoryArena arena(kDefaultArenaBase, kOffsetArenaSize, ArenaPlacement::Anywhere);
    CHECK(arena.valid());
    if (!arena.valid()) {
        return;
    }
    std::printf("  offset arena: guest_base=0x%llx identity=%d\n",
                static_cast<unsigned long long>(arena.guest_base()), arena.is_identity() ? 1 : 0);
    GuestHeap heap(arena);
    run_core_suite(heap);

    // Exhausting a small arena fails cleanly and recovers.
    VirtualMemoryArena small(kDefaultArenaBase, 16 * kMiB, ArenaPlacement::Anywhere);
    if (small.valid()) {
        GuestHeap sheap(small);
        std::vector<uint64_t> blocks;
        for (int i = 0; i < 64; ++i) {
            HeapStatus st = HeapStatus::Ok;
            const uint64_t va = sheap.alloc(3 * kMiB, 16, HeapFamily::Crt, 0, &st);
            if (va == 0) {
                CHECK(st == HeapStatus::OutOfMemory);
                break;
            }
            blocks.push_back(va);
        }
        CHECK(!blocks.empty() && blocks.size() < 64);
        CHECK(sheap.alloc(3 * kMiB) == 0);
        const uint64_t failed = sheap.stats().failed_allocs;
        CHECK(failed >= 2);
        for (uint64_t va : blocks) {
            CHECK(sheap.free(va));
        }
        const uint64_t back = sheap.alloc(3 * kMiB);
        CHECK(back != 0);
        CHECK(sheap.free(back));
    }
}

// ---------------------------------------------------------------------------
// CRT family on the default heap (identity process arena and an installed offset arena)
// ---------------------------------------------------------------------------

bool crt_ptr_valid(const void* p, uint64_t len) {
    GuestHeap& heap = default_guest_heap();
    const uint64_t va = heap.to_guest(p);
    return va != 0 && heap.owns(va) && valid_guest_range(heap.arena(), va, len);
}

void test_crt_family() {
    namespace crt = playgta6::os::guest_crt;
    GuestHeap& heap = default_guest_heap();
    int32_t err = 0;

    void* m = crt::malloc(100, &err);
    CHECK(m != nullptr && crt_ptr_valid(m, 100) && crt::msize(m) == 100);
    std::memset(m, 0x42, 100);
    void* r = crt::realloc(m, 4000, &err);
    CHECK(r != nullptr && crt_ptr_valid(r, 4000) && crt::msize(r) == 4000);
    CHECK(static_cast<uint8_t*>(r)[99] == 0x42);
    CHECK(crt::realloc(r, 0, &err) == nullptr && err == 0);
    CHECK(crt::msize(r, &err) == static_cast<size_t>(-1) && err == EINVAL);

    auto* c = static_cast<uint32_t*>(crt::calloc(100, 4, &err));
    CHECK(c != nullptr && c[0] == 0 && c[99] == 0 && crt_ptr_valid(c, 400));
    c = static_cast<uint32_t*>(crt::recalloc(c, 1000, 4, &err));
    CHECK(c != nullptr && c[999] == 0 && crt::msize(c) == 4000);
    CHECK(crt::expand(c, 2000) == c && crt::msize(c) == 2000);
    crt::free(c);

    // _aligned_* and free() accept each other's blocks.
    for (size_t align : {size_t{16}, size_t{64}, size_t{4096}, size_t{65536}, size_t{1} << 20}) {
        void* a = crt::aligned_malloc(1234, align, &err);
        CHECK(a != nullptr && (heap.to_guest(a) & (align - 1)) == 0 && crt::msize(a) == 1234);
        void* a2 = crt::aligned_realloc(a, 300000, align, &err);
        CHECK(a2 != nullptr && (heap.to_guest(a2) & (align - 1)) == 0 && crt::msize(a2) == 300000);
        if ((align & 1) == 0 && align < 4096) {
            crt::free(a2);  // aligned block through free()
        } else {
            crt::aligned_free(a2);
        }
    }
    void* plain = crt::malloc(77);
    crt::aligned_free(plain);  // plain block through _aligned_free
    CHECK(crt::aligned_malloc(10, 24, &err) == nullptr && err == EINVAL);
    CHECK(crt::aligned_realloc(nullptr, 10, 32) != nullptr);

    char* s = crt::strdup("guest string");
    CHECK(s != nullptr && std::strcmp(s, "guest string") == 0 && crt_ptr_valid(s, 13));
    CHECK(crt::msize(s) == 13);
    crt::free(s);
    const uint16_t wide[] = {'h', 'i', 0};
    uint16_t* w = crt::wcsdup(wide);
    CHECK(w != nullptr && w[0] == 'h' && w[2] == 0 && crt::msize(w) == 6);
    crt::free(w);
    CHECK(crt::strdup(nullptr, &err) == nullptr && err == EINVAL);

    // Foreign pointer and double free are diagnosed, not UB.
    int on_stack = 0;
    const uint64_t invalid0 = heap.stats().invalid_frees;
    crt::free(&on_stack);
    void* d = crt::malloc(8);
    crt::free(d);
    crt::free(d);
    CHECK(heap.stats().invalid_frees == invalid0 + 2);
    CHECK(crt::msize(&on_stack, &err) == static_cast<size_t>(-1));

    // The exported shims are this family.
    void* sm = pg6_crt_malloc(32);
    CHECK(sm != nullptr && crt_ptr_valid(sm, 32));
    sm = pg6_crt_realloc(sm, 64);
    CHECK(sm != nullptr && crt_ptr_valid(sm, 64));
    CHECK(pg6_crt_realloc(sm, 0) == nullptr);
    void* sz = pg6_crt_malloc(0);
    CHECK(sz != nullptr && crt::msize(sz) == 0);
    pg6_crt_free(sz);
    CHECK(pg6_crt_calloc(1u << 20, 1u << 20) == nullptr);  // 1 TiB

    // Soft limit: nullptr + ENOMEM from the shims, stats through the C ABI.
    Pg6GuestHeapStats hud{};
    CHECK(pg6_guest_heap_get_stats(&hud) == 1 && pg6_guest_heap_get_stats(nullptr) == 0);
    pg6_guest_heap_set_soft_limit(hud.live_bytes + 1000);
    *pg6_crt_errno() = 0;
    CHECK(pg6_crt_malloc(5000) == nullptr && *pg6_crt_errno() == PG6_CRT_ENOMEM);
    void* fits = pg6_crt_malloc(500);
    CHECK(fits != nullptr);
    pg6_crt_free(fits);
    pg6_guest_heap_set_soft_limit(0);
    CHECK(pg6_guest_heap_get_stats(&hud) == 1 && hud.soft_limit_bytes == 0 &&
          hud.crt.total_allocs > 10 && hud.failed_allocs >= 1);

    // strings and tables the CRT hands to guest code live in the guest address space.
    char*** argv_slot = pg6_crt_p___argv();
    const uint64_t argv_va = reinterpret_cast<uintptr_t>(*argv_slot);
    CHECK(heap.owns(argv_va));
    char* cmd = pg6_crt_get_narrow_winmain_command_line();
    CHECK(crt_ptr_valid(cmd, 1));
}

void test_crt_on_offset_arena() {
    VirtualMemoryArena offset(kDefaultArenaBase, kOffsetArenaSize, ArenaPlacement::Anywhere);
    CHECK(offset.valid());
    if (!offset.valid()) {
        return;
    }
    win32::set_vmem_arena(&offset);
    CHECK(&default_guest_heap().arena() == &offset);
    test_crt_family();

    // Pointers returned by the shims convert to guest VAs of that arena with the shared formula.
    void* p = pg6_crt_malloc(256);
    CHECK(p != nullptr);
    const uint64_t va = offset.guest_address(p);
    CHECK(va >= offset.space_base() && va < offset.space_base() + offset.space_size());
    CHECK(offset.translate(va) == p && offset.host_ptr(va, 256) == p);
    pg6_crt_free(p);

    win32::set_vmem_arena(nullptr);
    CHECK(&default_guest_heap().arena() == win32::get_vmem_arena());
}

// ---------------------------------------------------------------------------
// XMem shims
// ---------------------------------------------------------------------------

bool xmem_ptr_valid(const void* p, uint64_t len) {
    VirtualMemoryArena& arena = *win32::get_vmem_arena();
    const uint64_t va = arena.guest_address(p);
    return p != nullptr && valid_guest_range(arena, va, len);
}

void test_xmem_attributes_and_family() {
    xmem_reset_for_testing();
    VirtualMemoryArena& arena = *win32::get_vmem_arena();
    GuestHeap& heap = default_guest_heap();
    const GuestHeapStats before = heap.stats();

    // Alignment codes 0 (16 B) .. through 64 KiB / 2 MiB and the zero-init bit.
    const struct {
        uint8_t code;
        size_t align;
    } cases[] = {{0, 16}, {3, 16}, {7, 256}, {11, 4096}, {15, 65536}, {20, 2u << 20}, {31, 65536}};
    for (const auto& cs : cases) {
        const uint64_t attrs = make_xmem_attributes(5, 1, cs.code, 2, true);
        void* p = pg6_xbox_XMemAllocDefault(1000, attrs);
        CHECK(p != nullptr && xmem_ptr_valid(p, 1000));
        CHECK((arena.guest_address(p) & (cs.align - 1)) == 0);
        XMemAllocInfo info{};
        CHECK(xmem_get_allocation_info(p, &info) && info.size == 1000 && info.alignment == cs.align);
        CHECK(std::all_of(static_cast<uint8_t*>(p), static_cast<uint8_t*>(p) + 1000,
                          [](uint8_t b) { return b == 0; }));
        // The block lives in the XMem family of the guest heap.
        HeapBlockInfo bi;
        CHECK(heap.block_info(arena.guest_address(p), &bi) && bi.family == HeapFamily::XMem);
        pg6_xbox_XMemFree(p, attrs);
    }

    // Large allocations come straight from the arena and return their memory on free.
    const uint64_t committed0 = arena.committed_bytes();
    void* big = pg6_xbox_XMemAlloc(kBigBlock, make_xmem_attributes(0, 0, 15, 0, false));
    CHECK(big != nullptr && xmem_ptr_valid(big, kBigBlock));
    CHECK(arena.committed_bytes() >= committed0 + kBigBlock);
    static_cast<uint8_t*>(big)[kBigBlock - 1] = 7;
    Pg6XMemWorkingSetStats ws{};
    CHECK(pg6_xbox_XMemGetWorkingSetStatistics(&ws) == gdk::S_OK_HR);
    CHECK(ws.heap_allocated_bytes == kBigBlock && ws.live_heap_allocations == 1);
    pg6_xbox_XMemFree(big, 0);
    CHECK(arena.committed_bytes() == committed0);
    // Double free and foreign free of the XMem entry points are diagnosed, not UB.
    pg6_xbox_XMemFree(big, 0);
    int local = 0;
    pg6_xbox_XMemFree(&local, 0);

    // Everything XMem allocated is gone from the heap's accounting.
    const GuestHeapStats after = heap.stats();
    CHECK(after.live_bytes == before.live_bytes);
    CHECK(after.family[static_cast<size_t>(HeapFamily::XMem)].peak_bytes >= kBigBlock);

    // Soft limit: nullptr + ERROR_NOT_ENOUGH_MEMORY.
    heap.set_soft_limit(heap.stats().live_bytes + 1 * kMiB);
    win32::SetLastError(0);
    CHECK(pg6_xbox_XMemAllocDefault(64 * kMiB, 0) == nullptr);
    CHECK(win32::GetLastError() == win32::ERROR_NOT_ENOUGH_MEMORY);
    heap.set_soft_limit(0);
}

void test_xmem_virtual_alloc() {
    xmem_reset_for_testing();
    VirtualMemoryArena& arena = *win32::get_vmem_arena();
    const uint64_t committed0 = arena.committed_bytes();
    const uint64_t reserved0 = arena.reserved_bytes();

    // Title-like: several GiB of reserved, sparsely committed address space.
#if defined(__EMSCRIPTEN__)
    const uint64_t kRegions = 3;
#else
    const uint64_t kRegions = 6;
#endif
    std::vector<void*> regions;
    const uint64_t region_count = std::min<uint64_t>(kRegions, arena.space_size() / (2 * kGiB));
    for (uint64_t i = 0; i < region_count; ++i) {
        void* r = pg6_xbox_XMemVirtualAlloc(nullptr, 1 * kGiB, win32::MEM_RESERVE,
                                            win32::PAGE_READWRITE);
        CHECK(r != nullptr);
        if (r == nullptr) {
            continue;
        }
        regions.push_back(r);
        CHECK(arena.guest_address(r) >= arena.space_base());
    }
    CHECK(arena.reserved_bytes() >= reserved0 + regions.size() * kGiB);
    CHECK(arena.committed_bytes() == committed0);  // reserving commits nothing
    Pg6XMemWorkingSetStats ws{};
    CHECK(pg6_xbox_XMemGetWorkingSetStatistics(&ws) == gdk::S_OK_HR);
    CHECK(ws.virtual_reserved_bytes == regions.size() * kGiB && ws.virtual_committed_bytes == 0);

    // Commit one page inside the first region through the arena (VirtualAlloc semantics), touch
    // it, and prove the commit is tiny.
    if (!regions.empty()) {
        const uint64_t base_va = arena.guest_address(regions[0]);
        CHECK(arena.commit(base_va + 512 * kMiB, kPageSize));
        static_cast<uint8_t*>(arena.translate(base_va + 512 * kMiB))[0] = 1;
        CHECK(arena.committed_bytes() == committed0 + kPageSize);
    }

    // In-place commit of a whole region, console flags accepted.
    void* part = pg6_xbox_XMemVirtualAlloc(nullptr, 3 * kMiB + 1,
                                           win32::MEM_RESERVE | 0x20000000u /*large pages*/,
                                           win32::PAGE_READWRITE | 0x400u /*write combine*/);
    CHECK(part != nullptr);
    const uint64_t committed1 = arena.committed_bytes();
    CHECK(pg6_xbox_XMemVirtualAlloc(part, 3 * kMiB, win32::MEM_COMMIT, win32::PAGE_READWRITE) ==
          part);
    CHECK(arena.committed_bytes() > committed1);
    std::memset(part, 0x3C, 3 * kMiB);
    // Placement request: honoured exactly or fails.
    CHECK(pg6_xbox_XMemVirtualAlloc(part, 64 * kKiB, win32::MEM_RESERVE, win32::PAGE_READWRITE) ==
          nullptr);
    // Bad arguments.
    CHECK(pg6_xbox_XMemVirtualAlloc(nullptr, 0, win32::MEM_RESERVE, win32::PAGE_READWRITE) ==
          nullptr);
    CHECK(pg6_xbox_XMemVirtualAlloc(nullptr, 4096, 0, win32::PAGE_READWRITE) == nullptr);
    CHECK(pg6_xbox_XMemVirtualAlloc(nullptr, 4096, win32::MEM_COMMIT, 0x3) == nullptr);

    CHECK(xmem_virtual_free(part));
    CHECK(!xmem_virtual_free(part));
    for (void* r : regions) {
        CHECK(xmem_virtual_free(r));
    }
    // arena.commit() above was outside the XMem record, release returned it with the region.
    CHECK(arena.committed_bytes() == committed0);
    CHECK(arena.reserved_bytes() == reserved0);
}

void test_xmem_physical_pages() {
    xmem_reset_for_testing();
    VirtualMemoryArena& arena = *win32::get_vmem_arena();
    const uint64_t committed0 = arena.committed_bytes();
    GuestHeap& heap = default_guest_heap();
    const uint64_t heap_live0 = heap.stats().live_bytes;

    // 2 MiB pages via the page-size flag; allocation commits nothing.
    size_t n = 4;
    uint64_t pfn[4]{};
    CHECK(pg6_xbox_XMemAllocatePhysicalPages(&n, pfn, kXMemFlagPage2M, win32::PAGE_READWRITE) == 1);
    CHECK(arena.committed_bytes() == committed0 && heap.stats().live_bytes == heap_live0);

    // Map contiguously: guest memory, zero filled; write; unmap persists into guest-backed pages.
    auto* m = static_cast<uint8_t*>(pg6_xbox_XMemMapPhysicalPages(
        nullptr, 4, win32::PAGE_READWRITE, pfn, kXMemFlagPage2M));
    CHECK(m != nullptr && xmem_ptr_valid(m, 4 * kXMemPageSize2M));
    CHECK(m[0] == 0 && m[4 * kXMemPageSize2M - 1] == 0);
    for (int i = 0; i < 4; ++i) {
        std::memset(m + i * kXMemPageSize2M, 0x10 + i, kXMemPageSize2M);
    }
    m[2 * kXMemPageSize2M + 5] = 0xEE;
    Pg6XMemWorkingSetStats ws{};
    CHECK(pg6_xbox_XMemGetWorkingSetStatistics(&ws) == gdk::S_OK_HR);
    CHECK(ws.physical_pages_bytes == 4 * kXMemPageSize2M &&
          ws.mapped_physical_bytes == 4 * kXMemPageSize2M && ws.live_virtual_mappings == 1);
    CHECK(xmem_unmap_physical_pages(m));
    CHECK(!xmem_unmap_physical_pages(m));
    // Stored in guest heap blocks while unmapped.
    CHECK(heap.stats().live_bytes == heap_live0 + 4 * kXMemPageSize2M);

    // Map a subset scattered into a reserved region (commits it), contents survive and move.
    void* region = pg6_xbox_XMemVirtualAlloc(nullptr, 6 * kXMemPageSize2M, win32::MEM_RESERVE,
                                             win32::PAGE_READWRITE);
    CHECK(region != nullptr);
    const uint64_t scatter_pfn[2] = {pfn[2], pfn[0]};
    const uint64_t offsets[2] = {kXMemPageSize2M, 3 * kXMemPageSize2M};
    auto* sc = static_cast<uint8_t*>(pg6_xbox_XMemMapPhysicalPagesScatter(
        region, 2, win32::PAGE_READWRITE, scatter_pfn, offsets, kXMemFlagPage2M));
    CHECK(sc == region);
    CHECK(sc[0] == 0);                                               // gap
    CHECK(sc[kXMemPageSize2M] == 0x12 && sc[kXMemPageSize2M + 5] == 0xEE);  // pfn[2] at +2 MiB
    CHECK(sc[3 * kXMemPageSize2M] == 0x10);                          // pfn[0] at +6 MiB
    CHECK(heap.stats().live_bytes == heap_live0 + 2 * kXMemPageSize2M);  // moved, not copied
    sc[3 * kXMemPageSize2M + 1] = 0x77;
    CHECK(xmem_virtual_free(region));  // saves the pages back
    CHECK(heap.stats().live_bytes == heap_live0 + 4 * kXMemPageSize2M);

    // Mapping again shows the saved bytes.
    auto* again = static_cast<uint8_t*>(pg6_xbox_XMemMapPhysicalPages(
        nullptr, 2, win32::PAGE_READWRITE, pfn, kXMemFlagPage2M));
    CHECK(again != nullptr && again[0] == 0x10 && again[1] == 0x77 &&
          again[kXMemPageSize2M] == 0x11);
    CHECK(xmem_unmap_physical_pages(again));

    // Bad frame numbers, zero pages, misaligned scatter offsets.
    const uint64_t bogus[1] = {0xDEADBEEF};
    CHECK(pg6_xbox_XMemMapPhysicalPages(nullptr, 1, win32::PAGE_READWRITE, bogus, 0) == nullptr);
    CHECK(pg6_xbox_XMemMapPhysicalPages(nullptr, 0, win32::PAGE_READWRITE, pfn, 0) == nullptr);
    const uint64_t odd[1] = {123};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(nullptr, 1, win32::PAGE_READWRITE, pfn, odd,
                                               kXMemFlagPage2M) == nullptr);

    // Freeing the pages returns every byte; zero pages never needed a store.
    CHECK(pg6_xbox_XMemFreePhysicalPages(4, pfn) == 1);
    CHECK(heap.stats().live_bytes == heap_live0);
    CHECK(pg6_xbox_XMemFreePhysicalPages(4, pfn) == 0);
    CHECK(arena.committed_bytes() == committed0 || arena.committed_bytes() >= committed0);

    // 4 KiB / default 64 KiB pages, all-zero pages need no backing at all.
    size_t k = 8;
    uint64_t small[8]{};
    CHECK(pg6_xbox_XMemAllocatePhysicalPages(&k, small, 0, win32::PAGE_READWRITE) == 1);
    auto* sm = static_cast<uint8_t*>(
        pg6_xbox_XMemMapPhysicalPages(nullptr, 8, win32::PAGE_READWRITE, small, 0));
    CHECK(sm != nullptr && xmem_ptr_valid(sm, 8 * kXMemPageSize64K));
    CHECK(xmem_unmap_physical_pages(sm));
    CHECK(heap.stats().live_bytes == heap_live0);  // nothing written, nothing stored
    CHECK(pg6_xbox_XMemFreePhysicalPages(8, small) == 1);
    xmem_reset_for_testing();
}

void test_xmem_on_offset_arena() {
    VirtualMemoryArena offset(kDefaultArenaBase, kOffsetArenaSize, ArenaPlacement::Anywhere);
    CHECK(offset.valid());
    if (!offset.valid()) {
        return;
    }
    win32::set_vmem_arena(&offset);
    test_xmem_attributes_and_family();
    test_xmem_virtual_alloc();
    test_xmem_physical_pages();
    xmem_reset_for_testing();
    win32::set_vmem_arena(nullptr);
}

}  // namespace

int main() {
    std::printf("test_guest_heap\n");
    test_core_on_process_arena();
    test_core_on_offset_arena();
    test_crt_family();
    test_crt_on_offset_arena();
    test_xmem_attributes_and_family();
    test_xmem_virtual_alloc();
    test_xmem_physical_pages();
    test_xmem_on_offset_arena();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_guest_heap: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_guest_heap: all tests passed\n");
    return 0;
}
