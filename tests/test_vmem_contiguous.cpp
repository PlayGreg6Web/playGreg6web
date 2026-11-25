// VirtualMemoryArena on one contiguous guest address space (runtime/os/vmem.h): translation,
// contiguity across page runs and reservations, zero-fill on (re)commit, protections, the image
// base hint, exhaustion, many reservations and concurrent use.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "abi/playgta6_abi.h"
#include "os/vmem.h"
#include "os/win32_kernel.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif !defined(__EMSCRIPTEN__)
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_vmem_contiguous.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6::os;

constexpr uint64_t kImageBase = 0x140000000ULL;

bool all_zero(const uint8_t* p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (p[i] != 0) {
            return false;
        }
    }
    return true;
}

void test_translation_and_contiguity() {
    VirtualMemoryArena arena;
    CHECK(arena.valid());
    CHECK(arena.space_base() == playgta6::kGuestAddressSpaceBase);
    CHECK(arena.space_size() == kDefaultGuestSpaceSize);

    // Two back-to-back reservations, committed in separate runs with a different protection.
    const uint64_t a = arena.reserve(0, 64 * 1024, PageProtect::ReadWrite);
    const uint64_t b = arena.reserve(0, 64 * 1024, PageProtect::ReadWrite);
    CHECK(a == arena.space_base());
    CHECK(b == a + 64 * 1024);
    CHECK(arena.commit(a, 2 * kPageSize, PageProtect::ReadWrite));
    CHECK(arena.commit(a + 2 * kPageSize, 14 * kPageSize, PageProtect::ExecuteReadWrite));
    CHECK(arena.commit(b, 64 * 1024, PageProtect::ReadWrite));

    uint8_t* p = arena.host_ptr(a);
    CHECK(p != nullptr);
    CHECK(p == arena.translate(a));
    CHECK(reinterpret_cast<uintptr_t>(p) == static_cast<uintptr_t>(arena.guest_base() + a));
    CHECK(arena.guest_address(p) == a);
    for (uint64_t off : {uint64_t{1}, kPageSize - 1, kPageSize, 2 * kPageSize + 17,
                         uint64_t{64 * 1024}, uint64_t{128 * 1024 - 8}}) {
        CHECK(arena.host_ptr(a + off) == p + off);
        CHECK(arena.guest_address(p + off) == a + off);
    }
    // One host pointer spans both commits and both reservations.
    uint8_t* span = arena.host_ptr(a + kPageSize, 128 * 1024 - kPageSize);
    CHECK(span == p + kPageSize);
    if (span != nullptr) {
        for (size_t i = 0; i < 128 * 1024 - kPageSize; ++i) {
            span[i] = static_cast<uint8_t>(i * 7);
        }
        std::vector<uint8_t> back(128 * 1024 - kPageSize);
        CHECK(arena.read(a + kPageSize, back.data(), back.size()));
        bool same = true;
        for (size_t i = 0; i < back.size(); ++i) {
            same &= back[i] == static_cast<uint8_t>(i * 7);
        }
        CHECK(same);
    }
    // Writes through read()/write() and through the raw translation see the same bytes.
    const uint64_t value = 0x1122334455667788ULL;
    CHECK(arena.write(b + 64 * 1024 - 8, &value, 8));
    uint64_t raw = 0;
    std::memcpy(&raw, arena.translate(b + 64 * 1024 - 8), 8);
    CHECK(raw == value);

    // A gap (uncommitted page) or a free range breaks the span.
    CHECK(arena.decommit(a + kPageSize, kPageSize));
    CHECK(arena.host_ptr(a, 3 * kPageSize) == nullptr);
    CHECK(arena.host_ptr(b + 60 * 1024, 8 * 1024) == nullptr);  // runs past the reservation
    CHECK(arena.release(a));
    CHECK(arena.release(b));
}

void test_zero_fill() {
    VirtualMemoryArena arena;
    const uint64_t a = arena.reserve(0, 256 * 1024);
    CHECK(a != 0);
    CHECK(arena.commit(a, 256 * 1024, PageProtect::ReadWrite));
    std::vector<uint8_t> buf(256 * 1024, 0xAB);
    CHECK(arena.read(a, buf.data(), buf.size()));
    CHECK(all_zero(buf.data(), buf.size()));

    std::vector<uint8_t> junk(256 * 1024, 0xCD);
    CHECK(arena.write(a, junk.data(), junk.size()));
    // Decommit / recommit a middle run: zero again, neighbours untouched.
    CHECK(arena.decommit(a + 16 * kPageSize, 8 * kPageSize));
    CHECK(arena.committed_bytes() == 256 * 1024 - 8 * kPageSize);
    // O36: commit reports how much of the range was fresh (zero-filled) versus kept.
    uint64_t newly = 123;
    CHECK(arena.commit(a + 16 * kPageSize, 8 * kPageSize, PageProtect::ReadWrite, &newly));
    CHECK(newly == 8 * kPageSize);
    CHECK(arena.read(a, buf.data(), buf.size()));
    CHECK(all_zero(buf.data() + 16 * kPageSize, 8 * kPageSize));
    CHECK(buf[16 * kPageSize - 1] == 0xCD && buf[24 * kPageSize] == 0xCD);
    CHECK(arena.commit(a + 12 * kPageSize, 8 * kPageSize, PageProtect::ReadWrite, &newly));
    CHECK(newly == 0);  // all of it was committed already: contents kept
    CHECK(arena.decommit(a + 20 * kPageSize, 2 * kPageSize));
    CHECK(arena.commit(a + 16 * kPageSize, 8 * kPageSize, PageProtect::ReadWrite, &newly));
    CHECK(newly == 2 * kPageSize);
    CHECK(!arena.commit(a + 16 * kPageSize, 512 * 1024, PageProtect::ReadWrite, &newly));
    CHECK(newly == 0);

    // Release + reserve the same VA again: fresh, zeroed pages.
    CHECK(arena.release(a));
    CHECK(arena.reserve(a, 256 * 1024) == a);
    CHECK(arena.commit(a, 256 * 1024, PageProtect::ReadWrite));
    CHECK(arena.read(a, buf.data(), buf.size()));
    CHECK(all_zero(buf.data(), buf.size()));
    CHECK(arena.release(a));

    // A second arena placed where the first one was gets zeroed pages too.
    {
        VirtualMemoryArena first;
        const uint64_t va = first.reserve(0, 64 * 1024);
        CHECK(first.commit(va, 64 * 1024, PageProtect::ReadWrite));
        CHECK(first.write(va, junk.data(), 64 * 1024));
    }
    VirtualMemoryArena second;
    const uint64_t va = second.reserve(0, 64 * 1024);
    CHECK(second.commit(va, 64 * 1024, PageProtect::ReadWrite));
    CHECK(second.read(va, buf.data(), 64 * 1024));
    CHECK(all_zero(buf.data(), 64 * 1024));
}

void test_protection_bookkeeping() {
    VirtualMemoryArena arena;
    const uint64_t a = arena.reserve(0, 64 * 1024, PageProtect::ReadWrite);
    CHECK(arena.commit(a, 4 * kPageSize, PageProtect::ReadWrite));
    const uint64_t v = 42;
    CHECK(arena.write(a + kPageSize, &v, 8));

    PageProtect old = PageProtect::NoAccess;
    CHECK(arena.protect(a + kPageSize, kPageSize, PageProtect::ReadOnly, &old));
    CHECK(old == PageProtect::ReadWrite);
    MemoryBasicInfo info = arena.query(a + kPageSize);
    CHECK(info.base_address == a + kPageSize && info.region_size == kPageSize);
    CHECK(info.state == MemState::Commit && info.protect == PageProtect::ReadOnly);
    CHECK(info.allocation_base == a && info.allocation_protect == PageProtect::ReadWrite);
    uint64_t r = 0;
    CHECK(arena.read(a + kPageSize, &r, 8) && r == 42);
    CHECK(!arena.write(a + kPageSize, &v, 8));
    CHECK(arena.host_ptr(a + kPageSize) == nullptr);
    CHECK(arena.host_ptr_const(a + kPageSize) == arena.translate(a + kPageSize));

    CHECK(arena.protect(a + kPageSize, kPageSize, PageProtect::NoAccess, &old));
    CHECK(old == PageProtect::ReadOnly);
    CHECK(!arena.read(a + kPageSize, &r, 8));
    CHECK(arena.protect(a + kPageSize, kPageSize, PageProtect::ReadWrite, &old));
    CHECK(arena.read(a + kPageSize, &r, 8) && r == 42);  // data survives protection changes

    // Free space reports up to the next reservation / the end of the space.
    const MemoryBasicInfo free_info = arena.query(a + 64 * 1024);
    CHECK(free_info.state == MemState::Free);
    CHECK(free_info.region_size == arena.space_base() + arena.space_size() - (a + 64 * 1024));
    CHECK(arena.release(a));

#if defined(__EMSCRIPTEN__)
    CHECK(!VirtualMemoryArena::protections_enforced());
#else
    CHECK(VirtualMemoryArena::protections_enforced());
#endif
}

// Native hosts: protections and the inaccessible low range really fault. Probed in a child
// process (POSIX) or under SEH (MSVC); nothing to probe on wasm.
#if defined(_MSC_VER)
bool faults_on_write(volatile uint8_t* p) {
    __try {
        *p = 1;
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ? EXCEPTION_EXECUTE_HANDLER
                                                                  : EXCEPTION_CONTINUE_SEARCH) {
        return true;
    }
    return false;
}
bool faults_on_read(volatile uint8_t* p) {
    __try {
        (void)*p;
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ? EXCEPTION_EXECUTE_HANDLER
                                                                  : EXCEPTION_CONTINUE_SEARCH) {
        return true;
    }
    return false;
}
#elif !defined(__EMSCRIPTEN__) && !defined(_WIN32)
bool child_faults(void (*probe)(volatile uint8_t*), volatile uint8_t* p) {
    std::fflush(nullptr);
    const pid_t pid = fork();
    if (pid == 0) {
        probe(p);
        _exit(0);
    }
    int status = 0;
    if (pid < 0 || waitpid(pid, &status, 0) != pid) {
        return false;
    }
    return !(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}
void write_probe(volatile uint8_t* p) {
    *p = 1;
}
void read_probe(volatile uint8_t* p) {
    (void)*p;
}
bool faults_on_write(volatile uint8_t* p) {
    return child_faults(write_probe, p);
}
bool faults_on_read(volatile uint8_t* p) {
    return child_faults(read_probe, p);
}
#endif

void test_protection_enforced() {
#if defined(_MSC_VER) || (!defined(__EMSCRIPTEN__) && !defined(_WIN32))
    VirtualMemoryArena arena;
    const uint64_t a = arena.reserve(0, 64 * 1024);
    CHECK(arena.commit(a, 3 * kPageSize, PageProtect::ReadWrite));
    CHECK(arena.protect(a + kPageSize, kPageSize, PageProtect::ReadOnly));
    CHECK(arena.protect(a + 2 * kPageSize, kPageSize, PageProtect::NoAccess));
    volatile uint8_t* p = arena.translate(a);
    CHECK(!faults_on_write(p));
    CHECK(!faults_on_read(p + kPageSize));
    CHECK(faults_on_write(p + kPageSize));
    CHECK(faults_on_read(p + 2 * kPageSize));
    CHECK(faults_on_read(p + 3 * kPageSize));  // reserved, never committed
    CHECK(arena.decommit(a, kPageSize));
    CHECK(faults_on_read(p));  // decommitted
    if (!arena.is_identity()) {
        CHECK(faults_on_read(arena.translate(0)));       // null guest pointer
        CHECK(faults_on_read(arena.translate(0x1000)));  // small guest pointer
    }
    CHECK(arena.release(a));
#endif
}

void test_image_base_hint() {
    VirtualMemoryArena arena;
    // Allocations without a hint do not prevent the preferred image base from being used later.
    const uint64_t heap = arena.virtual_alloc(0, 1 << 20, kMemReserve | kMemCommit,
                                              PageProtect::ReadWrite);
    CHECK(heap == arena.space_base());
    const uint64_t image = arena.reserve(kImageBase, 0x0B06D000ULL, PageProtect::NoAccess);
    CHECK(image == kImageBase);
    CHECK(arena.commit(kImageBase + 0x1000, 0x10000, PageProtect::ReadWrite));
    CHECK(arena.host_ptr(kImageBase + 0x1000) == arena.translate(kImageBase + 0x1000));
    CHECK(arena.reserve(kImageBase + 0x20000, 0x1000) == 0);  // inside the image
    CHECK(arena.release(image));
    CHECK(arena.release(heap));
}

// Placeholders: reserve, split (MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER), replace
// (MEM_REPLACE_PLACEHOLDER), back to a placeholder, coalesce, release.
void test_placeholders() {
    VirtualMemoryArena arena;
    constexpr uint64_t kG = 0x10000;
    // Only reserve + NoAccess + 64 KiB granularity.
    CHECK(arena.virtual_alloc(0, 8 * kG, kMemReserve | kMemCommit | kMemReservePlaceholder, PageProtect::NoAccess) == 0);
    CHECK(arena.virtual_alloc(0, 8 * kG, kMemReserve | kMemReservePlaceholder, PageProtect::ReadWrite) == 0);
    CHECK(arena.virtual_alloc(0, 8 * kG + 0x1000, kMemReserve | kMemReservePlaceholder, PageProtect::NoAccess) == 0);
    const uint64_t base = arena.virtual_alloc(0, 8 * kG, kMemReserve | kMemReservePlaceholder, PageProtect::NoAccess);
    CHECK(base != 0);
    const uint64_t reserved = arena.reserved_bytes();
    CHECK(!arena.commit(base, kG, PageProtect::ReadWrite));  // never committable
    CHECK(!arena.virtual_free(base, kG, kMemDecommit));

    // Split [base, +2G) | [+2G, +5G) | [+5G, +8G) out of the placeholder.
    CHECK(arena.virtual_free(base, 2 * kG, kMemRelease | kMemPreservePlaceholder));
    CHECK(arena.virtual_free(base + 2 * kG, 3 * kG, kMemRelease | kMemPreservePlaceholder));
    CHECK(arena.reservation_count() == 3);
    CHECK(arena.reserved_bytes() == reserved);
    CHECK(!arena.virtual_free(base + kG / 2, kG, kMemRelease | kMemPreservePlaceholder));  // not 64 KiB aligned
    CHECK(arena.query(base + 3 * kG).allocation_base == base + 2 * kG);

    // Replace the middle one exactly, committed: usable memory.
    CHECK(arena.virtual_alloc(base + 2 * kG, 2 * kG, kMemReserve | kMemReplacePlaceholder, PageProtect::ReadWrite) ==
          0);  // wrong size
    CHECK(arena.virtual_alloc(base + 2 * kG, 3 * kG, kMemReserve | kMemCommit | kMemReplacePlaceholder,
                              PageProtect::ReadWrite) == base + 2 * kG);
    const uint32_t value = 0xC0FFEE;
    CHECK(arena.write(base + 2 * kG + 8, &value, sizeof(value)));
    CHECK(arena.virtual_alloc(base + 2 * kG, 3 * kG, kMemReserve | kMemReplacePlaceholder, PageProtect::ReadWrite) ==
          0);  // no longer a placeholder
    CHECK(!arena.commit(base, kG, PageProtect::ReadWrite));  // the others still are
    // Coalescing needs placeholders only.
    CHECK(!arena.virtual_free(base, 8 * kG, kMemRelease | kMemCoalescePlaceholders));

    // Back to a placeholder (whole range only), then coalesce all three and release.
    CHECK(!arena.virtual_free(base + 2 * kG, kG, kMemRelease | kMemPreservePlaceholder));
    CHECK(arena.virtual_free(base + 2 * kG, 3 * kG, kMemRelease | kMemPreservePlaceholder));
    CHECK(arena.committed_bytes() == 0);
    CHECK(!arena.write(base + 2 * kG + 8, &value, sizeof(value)));
    CHECK(arena.virtual_free(base, 8 * kG, kMemRelease | kMemCoalescePlaceholders));
    CHECK(arena.reservation_count() == 1);
    CHECK(arena.virtual_free(base, 0, kMemRelease));
    CHECK(arena.reservation_count() == 0 && arena.reserved_bytes() == 0);
}

void test_exhaustion() {
    // A 1 MiB space: 16 granules, then clean failure.
    VirtualMemoryArena small(kDefaultArenaBase, 1 << 20);
    CHECK(small.valid());
    std::vector<uint64_t> got;
    for (int i = 0; i < 32; ++i) {
        const uint64_t va = small.reserve(0, 4096);
        if (va == 0) {
            break;
        }
        got.push_back(va);
    }
    CHECK(got.size() == 16);
    CHECK(small.reserve(0, 4096) == 0);
    CHECK(small.reserve(kDefaultArenaBase + (1 << 20), 4096) == 0);   // just past the end
    CHECK(small.reserve(kDefaultArenaBase - 0x10000, 4096) == 0);     // below the space
    CHECK(small.reserve(kDefaultArenaBase + 0xF0000, 0x20000) == 0);  // straddles the end
    CHECK(small.virtual_alloc(0, 2 << 20, kMemReserve, PageProtect::ReadWrite) == 0);
    CHECK(small.release(got[3]));
    CHECK(small.reserve(0, 4096) == got[3]);  // the hole is reused
    CHECK(small.query(kDefaultArenaBase + (1 << 20)).region_size == 0);

    // A space nobody can back: invalid, every reservation fails, nothing crashes.
    VirtualMemoryArena impossible(kDefaultArenaBase, uint64_t{1} << 62);
    CHECK(!impossible.valid());
    CHECK(impossible.reserve(0, 4096) == 0);
    CHECK(impossible.reserve(kImageBase, 4096) == 0);
    CHECK(impossible.virtual_alloc(0, 4096, kMemReserve | kMemCommit, PageProtect::ReadWrite) == 0);
    CHECK(impossible.query(kImageBase).state == MemState::Free);
    CHECK(impossible.host_ptr(kImageBase) == nullptr);
}

void test_many_reservations() {
    VirtualMemoryArena arena;
    constexpr int kCycles = 100000;
    const auto start = std::chrono::steady_clock::now();
    bool ok = true;
    for (int i = 0; i < kCycles; ++i) {
        const uint64_t va = arena.reserve(0, 8 * 1024, PageProtect::ReadWrite);
        ok &= va != 0;
        ok &= arena.commit(va, kPageSize, PageProtect::ReadWrite);
        const uint32_t v = static_cast<uint32_t>(i);
        ok &= arena.write(va + 16, &v, 4);
        ok &= arena.release(va);
        if (!ok) {
            break;
        }
    }
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start)
                          .count();
    CHECK(ok);
    CHECK(arena.reservation_count() == 0 && arena.committed_bytes() == 0);
    std::printf("  %d reserve/commit/write/release cycles: %.1f ms\n", kCycles, ms);

    // Many live reservations, then release in a different order.
    std::vector<uint64_t> live;
    for (int i = 0; i < 2000; ++i) {
        live.push_back(arena.reserve(0, 4096));
    }
    CHECK(live.back() == arena.space_base() + 1999 * kAllocationGranularity);
    for (size_t i = 0; i < live.size(); i += 2) {
        CHECK(arena.release(live[i]));
    }
    for (size_t i = 1; i < live.size(); i += 2) {
        CHECK(arena.release(live[i]));
    }
    CHECK(arena.reservation_count() == 0);
}

void test_concurrent_reservations() {
    VirtualMemoryArena arena;
    constexpr int kThreads = 8;
    constexpr int kIterations = 500;
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&arena, &failures, t] {
            for (int i = 0; i < kIterations; ++i) {
                const uint64_t va = arena.virtual_alloc(0, 3 * kPageSize, kMemReserve | kMemCommit,
                                                        PageProtect::ReadWrite);
                if (va == 0) {
                    failures.fetch_add(1);
                    continue;
                }
                uint64_t* p = reinterpret_cast<uint64_t*>(arena.host_ptr(va, 3 * kPageSize));
                if (p == nullptr || p[0] != 0) {
                    failures.fetch_add(1);
                }
                const uint64_t tag = (static_cast<uint64_t>(t) << 32) | static_cast<uint64_t>(i);
                if (p != nullptr) {
                    p[0] = tag;
                    p[(3 * kPageSize) / 8 - 1] = tag;
                }
                uint64_t back = 0;
                if (!arena.read(va + 3 * kPageSize - 8, &back, 8) || back != tag) {
                    failures.fetch_add(1);
                }
                if (!arena.virtual_free(va, 0, kMemRelease)) {
                    failures.fetch_add(1);
                }
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    CHECK(failures.load() == 0);
    CHECK(arena.reservation_count() == 0 && arena.committed_bytes() == 0);
}

void test_process_arena() {
    using namespace playgta6::os::win32;
    VirtualMemoryArena& process = default_vmem_arena();
    CHECK(process.valid());
    CHECK(process.space_base() == playgta6::kGuestAddressSpaceBase);
    CHECK(process.space_size() == kProcessGuestSpaceSize);
    CHECK(get_vmem_arena() == &process);
    CHECK(__playgta6_guest_base == process.guest_base());
#if defined(__EMSCRIPTEN__)
    CHECK(process.is_identity());  // guest VA == linear-memory address in the browser build
#endif
    std::printf("  process arena: guest_base=0x%llx (%s), %llu GiB\n",
                static_cast<unsigned long long>(process.guest_base()),
                process.is_identity() ? "identity" : "offset",
                static_cast<unsigned long long>(process.space_size() >> 30));

    // The preferred image base is available in the process space.
    const uint64_t image = process.reserve(kImageBase, 0x10000, PageProtect::ReadWrite);
    CHECK(image == kImageBase);
    CHECK(process.commit(image, 0x10000, PageProtect::ReadWrite));
    CHECK(process.host_ptr(image) ==
          reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(__playgta6_guest_base + image)));
    CHECK(process.release(image));

    // Installing another arena republishes the translation base, uninstalling restores it.
    VirtualMemoryArena other;
    set_vmem_arena(&other);
    CHECK(__playgta6_guest_base == other.guest_base());
    set_vmem_arena(nullptr);
    CHECK(__playgta6_guest_base == process.guest_base());
}

// take_written_pages: pages written (or recommitted, i.e. zero-filled again) since the previous
// call. Windows tracks writes (MEM_WRITE_WATCH); other hosts report "unknown" (false).
void test_written_pages() {
    set_write_tracking(true);
    VirtualMemoryArena arena;
    set_write_tracking(false);
    const uint64_t a = arena.reserve(0, 64 * 1024);
    CHECK(a != 0);
    CHECK(arena.commit(a, 64 * 1024, PageProtect::ReadWrite));
    uint8_t* p = arena.host_ptr(a, 64 * 1024);
    CHECK(p != nullptr);
    if (p == nullptr) {
        return;
    }
    // A fresh caller (seq 0) may be further behind than the commit log remembers: the first call
    // can answer "unknown", and it brings `seq` up to date either way.
    uint64_t seq = 0;
    std::vector<uint64_t> pages;
    take_written_pages(p, 64 * 1024, &seq, &pages);
    pages.clear();
    const bool tracked = take_written_pages(p, 64 * 1024, &seq, &pages);
#if defined(_WIN32)
    CHECK(tracked);
#endif
    if (!tracked) {
        return;
    }
    // Primed: nothing new until something is written.
    pages.clear();
    CHECK(take_written_pages(p, 64 * 1024, &seq, &pages) && pages.empty());
    p[5 * kWrittenPageSize + 3] = 1;  // a plain store (lifted code writes this way)
    const uint64_t v = 7;
    CHECK(arena.write(a + 9 * kWrittenPageSize, &v, sizeof(v)));
    CHECK(take_written_pages(p, 64 * 1024, &seq, &pages));
    CHECK(pages == (std::vector<uint64_t>{5 * kWrittenPageSize, 9 * kWrittenPageSize}));
    // Reported once.
    pages.clear();
    CHECK(take_written_pages(p, 64 * 1024, &seq, &pages) && pages.empty());
    // An unaligned range reports offsets from its page-aligned start; pages outside it are not
    // consumed for other callers.
    p[2 * kWrittenPageSize] = 2;
    p[12 * kWrittenPageSize] = 3;
    CHECK(take_written_pages(p + 2 * kWrittenPageSize + 100, 100, &seq, &pages));
    CHECK(pages == (std::vector<uint64_t>{0}));
    pages.clear();
    CHECK(take_written_pages(p, 64 * 1024, &seq, &pages));
    CHECK(pages == (std::vector<uint64_t>{12 * kWrittenPageSize}));
    // Decommit + recommit zero-fills without a write: still reported.
    pages.clear();
    CHECK(arena.decommit(a + 4 * kPageSize, kPageSize));
    CHECK(arena.commit(a + 4 * kPageSize, kPageSize, PageProtect::ReadWrite));
    CHECK(take_written_pages(p, 64 * 1024, &seq, &pages));
    bool recommitted = !pages.empty();
    for (uint64_t off : pages) {
        recommitted &= off >= 4 * kPageSize && off < 5 * kPageSize;
    }
    CHECK(recommitted);
    // A caller further behind than the remembered commits gets "unknown".
    uint64_t stale = seq + 1;
    pages.clear();
    CHECK(!take_written_pages(p, 64 * 1024, &stale, &pages));
    CHECK(arena.release(a));
}

}  // namespace

int main() {
    std::printf("test_vmem_contiguous\n");
    test_translation_and_contiguity();
    test_zero_fill();
    test_protection_bookkeeping();
    test_protection_enforced();
    test_image_base_hint();
    test_placeholders();
    test_exhaustion();
    test_many_reservations();
    test_concurrent_reservations();
    test_process_arena();
    test_written_pages();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_vmem_contiguous: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_vmem_contiguous: all tests passed\n");
    return 0;
}
