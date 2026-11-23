#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "os/vmem.h"
#include "os/win32_kernel.h"
#include "os/win32_loader.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_win32_loader.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6::os;
using namespace playgta6::os::win32;

uint8_t* host(uint64_t va, uint64_t len) {
    return get_vmem_arena()->host_ptr(va, static_cast<size_t>(len));
}

void fill(uint64_t va, uint64_t len, uint8_t value) {
    if (len == 0) {
        return;
    }
    uint8_t* p = host(va, len);
    if (p != nullptr) {
        std::memset(p, value, static_cast<size_t>(len));
    }
}

bool all_equal(uint64_t va, uint64_t len, uint8_t value) {
    if (len == 0) {
        return true;
    }
    const uint8_t* p = host(va, len);
    if (p == nullptr) {
        return false;
    }
    for (uint64_t i = 0; i < len; ++i) {
        if (p[i] != value) {
            return false;
        }
    }
    return true;
}

struct XorShift {
    uint64_t state;
    uint64_t next() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }
};

// GetProcAddress reads a name pointer below 64 KiB as an ordinal (MAKEINTRESOURCE). Emscripten
// places string literals there on Wasm64, so by-name GetProcAddress tests pass heap copies.
struct HeapName {
    std::unique_ptr<char[]> chars;
    operator const char*() const { return chars.get(); }
};

HeapName heap_name(const char* literal) {
    const size_t len = std::strlen(literal) + 1;
    HeapName out{std::make_unique<char[]>(len)};
    std::memcpy(out.chars.get(), literal, len);
    return out;
}

int32_t test_export_alpha() {
    return 11;
}

int32_t test_export_beta() {
    return 22;
}

// ---------------------------------------------------------------------------
// 1. Heap basics: process heap, alignment, exact HeapSize, distinct blocks
// ---------------------------------------------------------------------------

void test_heap_basics() {
    const Handle heap = GetProcessHeap();
    CHECK(heap != kNullHandle);
    CHECK(GetProcessHeap() == heap);
    CHECK(pg6_win32_get_process_heap() == heap);
    CHECK(HeapValidate(heap, 0, 0));

    const uint64_t sizes[] = {0, 1, 15, 16, 17, 100, 4096, 100000};
    std::vector<uint64_t> blocks;
    for (size_t i = 0; i < std::size(sizes); ++i) {
        const uint64_t p = HeapAlloc(heap, 0, sizes[i]);
        CHECK(p != 0);
        CHECK((p % 16) == 0);
        CHECK(HeapSize(heap, 0, p) == sizes[i]);
        CHECK(HeapValidate(heap, 0, p));
        fill(p, sizes[i], static_cast<uint8_t>(0x10 + i));
        blocks.push_back(p);
    }
    for (size_t i = 0; i < blocks.size(); ++i) {
        CHECK(all_equal(blocks[i], sizes[i], static_cast<uint8_t>(0x10 + i)));
        for (size_t j = i + 1; j < blocks.size(); ++j) {
            CHECK(blocks[i] != blocks[j]);
        }
    }

    // VirtualQuery sees heap memory as committed read/write arena memory.
    MemoryBasicInformation mbi{};
    CHECK(VirtualQuery(blocks[0], &mbi, sizeof(mbi)) == sizeof(mbi));
    CHECK(mbi.state == MEM_COMMIT);
    CHECK(mbi.protect == PAGE_READWRITE);

    CHECK(HeapValidate(heap, 0, 0));
    for (uint64_t p : blocks) {
        CHECK(HeapFree(heap, 0, p));
        CHECK(!HeapValidate(heap, 0, p));
    }
    CHECK(HeapValidate(heap, 0, 0));

    // C-ABI round trip.
    const uint64_t c = pg6_win32_heap_alloc(heap, HEAP_ZERO_MEMORY, 48);
    CHECK(c != 0);
    CHECK(pg6_win32_heap_size(heap, 0, c) == 48);
    CHECK(all_equal(c, 48, 0));
    CHECK(pg6_win32_heap_validate(heap, 0, c) == 1);
    const uint64_t c2 = pg6_win32_heap_realloc(heap, 0, c, 4096);
    CHECK(c2 != 0);
    CHECK(pg6_win32_heap_size(heap, 0, c2) == 4096);
    CHECK(pg6_win32_heap_free(heap, 0, c2) == 1);
}

// ---------------------------------------------------------------------------
// 2. HEAP_ZERO_MEMORY on alloc and on realloc growth (in place and moved)
// ---------------------------------------------------------------------------

void test_heap_zeroing() {
    const Handle heap = HeapCreate(0, 0, 0);
    CHECK(heap != kNullHandle);

    const uint64_t a = HeapAlloc(heap, 0, 256);
    CHECK(a != 0);
    fill(a, 256, 0xAB);
    CHECK(HeapFree(heap, 0, a));
    const uint64_t z = HeapAlloc(heap, HEAP_ZERO_MEMORY, 256);
    CHECK(z == a);  // Best fit reuses the freed block...
    CHECK(all_equal(z, 256, 0));  // ...and HEAP_ZERO_MEMORY clears the stale contents.

    // Shrink in place, then grow back in place: the stale tail must be zeroed.
    fill(z, 256, 0xEE);
    CHECK(HeapReAlloc(heap, 0, z, 16) == z);
    CHECK(HeapSize(heap, 0, z) == 16);
    CHECK(HeapReAlloc(heap, HEAP_ZERO_MEMORY | HEAP_REALLOC_IN_PLACE_ONLY, z, 1024) == z);
    CHECK(HeapSize(heap, 0, z) == 1024);
    CHECK(all_equal(z, 16, 0xEE));
    CHECK(all_equal(z + 16, 1024 - 16, 0));

    // Growth that must move: block it in with a neighbour first.
    const uint64_t neighbour = HeapAlloc(heap, 0, 64);
    CHECK(neighbour == z + 1024);
    const uint64_t moved = HeapReAlloc(heap, HEAP_ZERO_MEMORY, z, 8192);
    CHECK(moved != 0);
    CHECK(moved != z);
    CHECK(HeapSize(heap, 0, moved) == 8192);
    CHECK(all_equal(moved, 16, 0xEE));
    CHECK(all_equal(moved + 16, 1024 - 16, 0));
    CHECK(all_equal(moved + 1024, 8192 - 1024, 0));
    CHECK(HeapSize(heap, 0, z) == kHeapSizeError);  // Old block is gone.

    CHECK(HeapValidate(heap, 0, 0));
    CHECK(HeapDestroy(heap));
}

// ---------------------------------------------------------------------------
// 3. HeapReAlloc semantics: in-place-only failure keeps the block intact
// ---------------------------------------------------------------------------

void test_heap_realloc_semantics() {
    const Handle heap = HeapCreate(0, 0, 0);
    CHECK(heap != kNullHandle);

    const uint64_t a = HeapAlloc(heap, 0, 64);
    const uint64_t b = HeapAlloc(heap, 0, 64);
    CHECK(a != 0);
    CHECK(b == a + 64);
    fill(a, 64, 0x5A);

    SetLastError(ERROR_SUCCESS);
    CHECK(HeapReAlloc(heap, HEAP_REALLOC_IN_PLACE_ONLY, a, 128) == 0);
    CHECK(GetLastError() == ERROR_NOT_ENOUGH_MEMORY);
    CHECK(HeapSize(heap, 0, a) == 64);
    CHECK(all_equal(a, 64, 0x5A));

    CHECK(HeapFree(heap, 0, b));
    CHECK(HeapReAlloc(heap, HEAP_REALLOC_IN_PLACE_ONLY, a, 128) == a);
    CHECK(HeapSize(heap, 0, a) == 128);
    CHECK(all_equal(a, 64, 0x5A));

    // Same-capacity resize and in-place shrink keep the address.
    CHECK(HeapReAlloc(heap, 0, a, 120) == a);
    CHECK(HeapSize(heap, 0, a) == 120);
    CHECK(HeapReAlloc(heap, HEAP_REALLOC_IN_PLACE_ONLY, a, 32) == a);
    CHECK(HeapSize(heap, 0, a) == 32);
    CHECK(all_equal(a, 32, 0x5A));

    // Moving realloc preserves the content prefix.
    const uint64_t blocker = HeapAlloc(heap, 0, 16);
    CHECK(blocker == a + 32);
    const uint64_t big = HeapReAlloc(heap, 0, a, 3 * 1024 * 1024);  // Needs a new segment.
    CHECK(big != 0);
    CHECK(big != a);
    CHECK((big % 16) == 0);
    CHECK(all_equal(big, 32, 0x5A));
    CHECK(!HeapValidate(heap, 0, a));

    // Shrinking a moved block back.
    const uint64_t small = HeapReAlloc(heap, 0, big, 8);
    CHECK(small == big);
    CHECK(all_equal(small, 8, 0x5A));

    CHECK(HeapValidate(heap, 0, 0));
    CHECK(HeapDestroy(heap));
}

// ---------------------------------------------------------------------------
// 4. Private heaps: destroy frees everything, fixed-size heaps, segment release
// ---------------------------------------------------------------------------

void test_private_heaps() {
    (void)GetProcessHeap();
    VirtualMemoryArena* arena = get_vmem_arena();
    const size_t reservations_before = arena->reservation_count();
    const uint64_t committed_before = arena->committed_bytes();

    const Handle heap = pg6_win32_heap_create(0, 0, 0);
    CHECK(heap != kNullHandle);
    CHECK(heap != GetProcessHeap());
    CHECK(arena->reservation_count() == reservations_before + 1);

    std::vector<uint64_t> blocks;
    for (int i = 0; i < 64; ++i) {
        const uint64_t p = HeapAlloc(heap, 0, 1000 + static_cast<uint64_t>(i) * 37);
        CHECK(p != 0);
        blocks.push_back(p);
    }
    const uint64_t large = HeapAlloc(heap, HEAP_ZERO_MEMORY, 5 * 1024 * 1024);
    CHECK(large != 0);
    CHECK(all_equal(large, 5 * 1024 * 1024, 0));
    CHECK(arena->reservation_count() == reservations_before + 2);

    // Freeing the only block of an extra segment returns the segment to the arena.
    CHECK(HeapFree(heap, 0, large));
    CHECK(arena->reservation_count() == reservations_before + 1);

    const uint64_t large2 = HeapAlloc(heap, 0, 2 * 1024 * 1024);
    CHECK(large2 != 0);
    CHECK(arena->reservation_count() == reservations_before + 2);

    // HeapDestroy releases every segment, including outstanding blocks.
    CHECK(pg6_win32_heap_destroy(heap) == 1);
    CHECK(arena->reservation_count() == reservations_before);
    CHECK(arena->committed_bytes() == committed_before);

    SetLastError(ERROR_SUCCESS);
    CHECK(HeapAlloc(heap, 0, 16) == 0);
    CHECK(GetLastError() == ERROR_INVALID_HANDLE);
    CHECK(!HeapFree(heap, 0, blocks[0]));
    CHECK(HeapSize(heap, 0, blocks[0]) == kHeapSizeError);
    CHECK(!HeapValidate(heap, 0, 0));
    CHECK(!HeapDestroy(heap));

    // The process heap cannot be destroyed.
    SetLastError(ERROR_SUCCESS);
    CHECK(!HeapDestroy(GetProcessHeap()));
    CHECK(GetLastError() == ERROR_INVALID_HANDLE);
    CHECK(HeapValidate(GetProcessHeap(), 0, 0));

    // Fixed-size heap: capacity is bounded by maximum_size.
    const Handle fixed = HeapCreate(0, 4096, 64 * 1024);
    CHECK(fixed != kNullHandle);
    const uint64_t f1 = HeapAlloc(fixed, 0, 32 * 1024);
    const uint64_t f2 = HeapAlloc(fixed, 0, 32 * 1024);
    CHECK(f1 != 0);
    CHECK(f2 != 0);
    SetLastError(ERROR_SUCCESS);
    CHECK(HeapAlloc(fixed, 0, 16) == 0);
    CHECK(GetLastError() == ERROR_NOT_ENOUGH_MEMORY);
    CHECK(HeapFree(fixed, 0, f2));
    CHECK(HeapAlloc(fixed, 0, 16) != 0);
    CHECK(HeapDestroy(fixed));

    SetLastError(ERROR_SUCCESS);
    CHECK(HeapCreate(0, 128 * 1024, 64 * 1024) == kNullHandle);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);

    // Executable heaps are backed by PAGE_EXECUTE_READWRITE memory.
    const Handle exec = HeapCreate(HEAP_CREATE_ENABLE_EXECUTE, 0, 0);
    CHECK(exec != kNullHandle);
    const uint64_t e = HeapAlloc(exec, 0, 64);
    CHECK(e != 0);
    MemoryBasicInformation mbi{};
    CHECK(VirtualQuery(e, &mbi, sizeof(mbi)) == sizeof(mbi));
    CHECK(mbi.protect == PAGE_EXECUTE_READWRITE);
    CHECK(HeapDestroy(exec));

    CHECK(arena->reservation_count() == reservations_before);
    CHECK(arena->committed_bytes() == committed_before);
}

// ---------------------------------------------------------------------------
// 5. Invalid arguments: defined failures, never UB
// ---------------------------------------------------------------------------

void test_heap_invalid_arguments() {
    const Handle heap = GetProcessHeap();
    const Handle other = HeapCreate(0, 0, 0);
    CHECK(other != kNullHandle);

    SetLastError(ERROR_SUCCESS);
    CHECK(HeapAlloc(kNullHandle, 0, 16) == 0);
    CHECK(GetLastError() == ERROR_INVALID_HANDLE);
    SetLastError(ERROR_SUCCESS);
    CHECK(HeapAlloc(0x1234, 0, 16) == 0);
    CHECK(GetLastError() == ERROR_INVALID_HANDLE);

    SetLastError(ERROR_SUCCESS);
    CHECK(HeapAlloc(heap, 0, kMaxHeapRequest + 1) == 0);
    CHECK(GetLastError() == ERROR_NOT_ENOUGH_MEMORY);
    CHECK(HeapAlloc(heap, 0, ~uint64_t{0}) == 0);

    const uint64_t foreign = HeapAlloc(other, 0, 64);
    CHECK(foreign != 0);

    // Freeing another heap's block, an interior pointer, or a random address fails.
    SetLastError(ERROR_SUCCESS);
    CHECK(!HeapFree(heap, 0, foreign));
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    CHECK(!HeapFree(other, 0, foreign + 16));
    CHECK(!HeapFree(heap, 0, 0xDEADBEEF0ULL));
    CHECK(HeapSize(heap, 0, foreign) == kHeapSizeError);
    CHECK(!HeapValidate(heap, 0, foreign));
    CHECK(HeapReAlloc(heap, 0, foreign, 128) == 0);

    // Double free is detected.
    CHECK(HeapFree(other, 0, foreign));
    SetLastError(ERROR_SUCCESS);
    CHECK(!HeapFree(other, 0, foreign));
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    CHECK(HeapSize(other, 0, foreign) == kHeapSizeError);

    // NULL frees are no-ops; NULL reallocs fail.
    CHECK(HeapFree(heap, 0, 0));
    SetLastError(ERROR_SUCCESS);
    CHECK(HeapReAlloc(heap, 0, 0, 16) == 0);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);

    // Oversized realloc fails and keeps the block.
    const uint64_t keep = HeapAlloc(heap, 0, 32);
    CHECK(HeapReAlloc(heap, 0, keep, kMaxHeapRequest + 16) == 0);
    CHECK(HeapSize(heap, 0, keep) == 32);
    CHECK(HeapFree(heap, 0, keep));

    CHECK(pg6_win32_heap_free(heap, 0, 0x10) == 0);
    CHECK(pg6_win32_heap_destroy(kNullHandle) == 0);
    CHECK(pg6_win32_heap_size(kNullHandle, 0, 0) == kHeapSizeError);

    CHECK(HeapValidate(other, 0, 0));
    CHECK(HeapDestroy(other));
}

// ---------------------------------------------------------------------------
// 6. Multi-threaded alloc / realloc / free stress with content checks
// ---------------------------------------------------------------------------

void test_heap_stress() {
    const Handle shared_private = HeapCreate(0, 0, 0);
    CHECK(shared_private != kNullHandle);
    const Handle process = GetProcessHeap();

    constexpr int kThreads = 8;
    constexpr int kOps = 2500;
    std::atomic<int> errors{0};

    auto worker = [&](int tid) {
        struct Live {
            Handle heap;
            uint64_t addr;
            uint64_t size;
            uint8_t tag;
        };
        std::vector<Live> live;
        XorShift rng{0x9E3779B97F4A7C15ULL ^ (static_cast<uint64_t>(tid + 1) * 0x1000193ULL)};

        auto verify = [&](const Live& b) {
            if (HeapSize(b.heap, 0, b.addr) != b.size || !all_equal(b.addr, b.size, b.tag)) {
                errors.fetch_add(1);
            }
        };

        for (int op = 0; op < kOps; ++op) {
            const uint64_t r = rng.next();
            const int kind = static_cast<int>(r % 10);
            if (live.empty() || kind < 5) {
                const Handle h = (r & 0x100) ? shared_private : process;
                uint64_t size = (r >> 12) % 2048;
                if ((r >> 40) % 64 == 0) {
                    size = 64 * 1024 + (r >> 20) % (256 * 1024);
                }
                const uint8_t tag = static_cast<uint8_t>((tid * 31 + op) | 1);
                const uint64_t p = HeapAlloc(h, (r & 0x200) ? HEAP_ZERO_MEMORY : 0, size);
                if (p == 0 || (p % 16) != 0) {
                    errors.fetch_add(1);
                    continue;
                }
                if ((r & 0x200) && !all_equal(p, size, 0)) {
                    errors.fetch_add(1);
                }
                fill(p, size, tag);
                live.push_back(Live{h, p, size, tag});
            } else if (kind < 7) {
                const size_t idx = static_cast<size_t>((r >> 8) % live.size());
                Live& b = live[idx];
                verify(b);
                const uint64_t new_size = (r >> 24) % 4096;
                const uint64_t q = HeapReAlloc(b.heap, 0, b.addr, new_size);
                if (q == 0 || (q % 16) != 0) {
                    errors.fetch_add(1);
                    continue;
                }
                if (!all_equal(q, std::min(b.size, new_size), b.tag)) {
                    errors.fetch_add(1);
                }
                fill(q, new_size, b.tag);
                b.addr = q;
                b.size = new_size;
            } else {
                const size_t idx = static_cast<size_t>((r >> 8) % live.size());
                verify(live[idx]);
                if (!HeapFree(live[idx].heap, 0, live[idx].addr)) {
                    errors.fetch_add(1);
                }
                live[idx] = live.back();
                live.pop_back();
            }
        }
        for (const Live& b : live) {
            verify(b);
            if (!HeapFree(b.heap, 0, b.addr)) {
                errors.fetch_add(1);
            }
        }
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back(worker, t);
    }
    for (auto& t : threads) {
        t.join();
    }

    CHECK(errors.load() == 0);
    CHECK(HeapValidate(process, 0, 0));
    CHECK(HeapValidate(shared_private, 0, 0));
    CHECK(HeapDestroy(shared_private));
}

// ---------------------------------------------------------------------------
// 7. Module lookup: NULL, case, suffix, path, API sets, main image
// ---------------------------------------------------------------------------

void test_module_lookup() {
    // No main image registered yet.
    SetLastError(ERROR_SUCCESS);
    CHECK(GetModuleHandleA(nullptr) == kNullHandle);
    CHECK(GetLastError() == ERROR_MOD_NOT_FOUND);

    const Handle k32 = GetModuleHandleA("kernel32.dll");
    CHECK(k32 != kNullHandle);
    CHECK(k32 >= kVirtualModuleBase);
    CHECK(GetModuleHandleA("KERNEL32.DLL") == k32);
    CHECK(GetModuleHandleA("Kernel32") == k32);
    CHECK(GetModuleHandleA("C:\\Windows\\System32\\kernel32.dll") == k32);
    CHECK(GetModuleHandleA("c:/windows/system32/KERNEL32") == k32);
    CHECK(GetModuleHandleA("api-ms-win-core-heap-l1-1-0.dll") == k32);
    CHECK(GetModuleHandleA("API-MS-WIN-CORE-SYNCH-L1-2-0") == k32);
    CHECK(pg6_win32_get_module_handle_a("kernel32") == k32);

    CHECK(canonical_module_name("C:\\X\\Foo.DLL") == "foo.dll");
    CHECK(canonical_module_name("bar") == "bar.dll");
    CHECK(canonical_module_name("game.exe") == "game.exe");
    CHECK(canonical_module_name("noext.") == "noext");
    CHECK(canonical_module_name("C:\\dir\\").empty());

    SetLastError(ERROR_SUCCESS);
    CHECK(GetModuleHandleA("kernel32.") == kNullHandle);  // Trailing dot: no ".dll" appended.
    CHECK(GetLastError() == ERROR_MOD_NOT_FOUND);
    SetLastError(ERROR_SUCCESS);
    CHECK(GetModuleHandleA("nonexistent_module.dll") == kNullHandle);
    CHECK(GetLastError() == ERROR_MOD_NOT_FOUND);
    CHECK(GetModuleHandleA("") == kNullHandle);

    // API-set host not registered yet -> not found; registering the host makes it resolve.
    CHECK(GetModuleHandleA("api-ms-win-crt-runtime-l1-1-0.dll") == kNullHandle);
    const Handle ucrt = register_virtual_module("ucrtbase.dll");
    CHECK(ucrt != kNullHandle);
    CHECK(ucrt != k32);
    CHECK(GetModuleHandleA("api-ms-win-crt-runtime-l1-1-0.dll") == ucrt);

    // Longest prefix wins over the generic api-ms-win-core- rule.
    const Handle combase = register_virtual_module("combase.dll");
    CHECK(GetModuleHandleA("api-ms-win-core-com-l1-1-0.dll") == combase);
    CHECK(GetModuleHandleA("api-ms-win-core-winrt-string-l1-1-0.dll") == combase);

    // Custom rules and exact registrations.
    CHECK(!register_api_set_rule("not-an-api-set-", "kernel32.dll"));
    CHECK(register_api_set_rule("EXT-MS-WIN-PG6TEST-", "PG6Host"));
    CHECK(GetModuleHandleA("ext-ms-win-pg6test-l1-1-0.dll") == kNullHandle);
    const Handle pg6host = register_virtual_module("pg6host.dll");
    CHECK(GetModuleHandleA("ext-ms-win-pg6test-l1-1-0.dll") == pg6host);
    const Handle exact = register_virtual_module("api-ms-win-core-pg6exact-l1-1-0.dll");
    CHECK(exact != k32);
    CHECK(GetModuleHandleA("api-ms-win-core-pg6exact-l1-1-0.dll") == exact);
    CHECK(GetModuleHandleA("api-ms-win-core-pg6other-l1-1-0.dll") == k32);

    // register_virtual_module is idempotent and case-insensitive.
    CHECK(register_virtual_module("UCRTBASE") == ucrt);
    CHECK(register_virtual_module("") == kNullHandle);
    CHECK(pg6_loader_register_virtual_module("xgameruntime.dll", 1) != kNullHandle);
    CHECK(pg6_loader_register_virtual_module("XGameRuntime", 1) ==
          GetModuleHandleA("xgameruntime.dll"));

    // Main image.
    constexpr uint64_t kImageBase = 0x0000000140000000ULL;
    CHECK(register_main_module("GTA6.exe", "G:\\GTA6.exe", kImageBase) == kImageBase);
    CHECK(GetModuleHandleA(nullptr) == kImageBase);
    CHECK(GetModuleHandleW(nullptr) == kImageBase);
    CHECK(pg6_win32_get_module_handle_a(nullptr) == kImageBase);
    CHECK(GetModuleHandleA("gta6.EXE") == kImageBase);
    CHECK(GetModuleHandleA("D:\\somewhere\\GTA6.exe") == kImageBase);
    CHECK(GetModuleHandleA("gta6") == kNullHandle);  // Resolves to gta6.dll.
    CHECK(register_main_module("GTA6.exe", "G:\\GTA6.exe", kImageBase) == kImageBase);
    SetLastError(ERROR_SUCCESS);
    CHECK(register_main_module("Other.exe", "G:\\Other.exe", 0x150000000ULL) == kNullHandle);
    CHECK(GetLastError() == ERROR_ALREADY_EXISTS);
    CHECK(pg6_loader_register_main_module("GTA6.exe", "G:\\GTA6.exe", 0) == kNullHandle);

    // GetModuleFileNameA: NULL -> main image, truncation semantics.
    char buf[64] = {};
    CHECK(GetModuleFileNameA(kNullHandle, buf, sizeof(buf)) == 11);
    CHECK(std::string(buf) == "G:\\GTA6.exe");
    CHECK(pg6_win32_get_module_file_name_a(k32, buf, sizeof(buf)) == 32);
    CHECK(std::string(buf) == "C:\\Windows\\System32\\KERNEL32.dll");

    char small[5] = {'x', 'x', 'x', 'x', 'x'};
    SetLastError(ERROR_SUCCESS);
    CHECK(GetModuleFileNameA(kNullHandle, small, sizeof(small)) == 5);
    CHECK(GetLastError() == ERROR_INSUFFICIENT_BUFFER);
    CHECK(std::string(small) == "G:\\G");
    SetLastError(ERROR_SUCCESS);
    CHECK(GetModuleFileNameA(kNullHandle, buf, 0) == 0);
    CHECK(GetLastError() == ERROR_INSUFFICIENT_BUFFER);
    SetLastError(ERROR_SUCCESS);
    CHECK(GetModuleFileNameA(0xDEAD0000ULL, buf, sizeof(buf)) == 0);
    CHECK(GetLastError() == ERROR_MOD_NOT_FOUND);
}

// ---------------------------------------------------------------------------
// 8. Wide-string (char16_t) variants
// ---------------------------------------------------------------------------

void test_wide_variants() {
    const Handle k32 = GetModuleHandleA("kernel32.dll");
    CHECK(GetModuleHandleW(u"KERNEL32.DLL") == k32);
    CHECK(GetModuleHandleW(u"C:\\Windows\\System32\\kernel32") == k32);
    CHECK(GetModuleHandleW(u"api-ms-win-core-libraryloader-l1-2-0.dll") == k32);
    CHECK(pg6_win32_get_module_handle_w(u"Kernel32.dll") == k32);
    SetLastError(ERROR_SUCCESS);
    CHECK(GetModuleHandleW(u"missing_wide.dll") == kNullHandle);
    CHECK(GetLastError() == ERROR_MOD_NOT_FOUND);

    // Non-ASCII module names round-trip through UTF-8 storage.
    const Handle umlaut = register_virtual_module("Mod\xC3\xBCl.dll");
    CHECK(umlaut != kNullHandle);
    CHECK(GetModuleHandleW(u"mod\u00FCl.DLL") == umlaut);
    CHECK(LoadLibraryW(u"Mod\u00FCl") == umlaut);
    CHECK(FreeLibrary(umlaut));

    char16_t wbuf[64] = {};
    CHECK(GetModuleFileNameW(umlaut, wbuf, 64) == 29);
    CHECK(std::u16string(wbuf) == u"C:\\Windows\\System32\\Mod\u00FCl.dll");
    CHECK(pg6_win32_get_module_file_name_w(kNullHandle, wbuf, 64) == 11);
    CHECK(std::u16string(wbuf) == u"G:\\GTA6.exe");

    char16_t wsmall[4] = {u'x', u'x', u'x', u'x'};
    SetLastError(ERROR_SUCCESS);
    CHECK(GetModuleFileNameW(kNullHandle, wsmall, 4) == 4);
    CHECK(GetLastError() == ERROR_INSUFFICIENT_BUFFER);
    CHECK(std::u16string(wsmall) == u"G:\\");

    CHECK(LoadLibraryExW(u"API-MS-WIN-CORE-HEAP-L1-1-0", kNullHandle, 0) == k32);
    CHECK(pg6_win32_load_library_w(u"kernel32") == k32);
    CHECK(pg6_win32_load_library_ex_w(u"kernel32", kNullHandle, 0) == k32);
    CHECK(FreeLibrary(k32));
    CHECK(FreeLibrary(k32));
    CHECK(FreeLibrary(k32));
}

// ---------------------------------------------------------------------------
// 9. LoadLibrary / FreeLibrary reference counting and failures
// ---------------------------------------------------------------------------

void test_load_library_refcounts() {
    const Handle delay = register_virtual_module("delayload_test.dll", false);
    CHECK(delay != kNullHandle);
    CHECK(register_export("delayload_test.dll", "DelayFn", 0,
                          proc_address(&test_export_alpha)));

    // Unpinned modules are invisible until loaded.
    SetLastError(ERROR_SUCCESS);
    CHECK(GetModuleHandleA("delayload_test") == kNullHandle);
    CHECK(GetLastError() == ERROR_MOD_NOT_FOUND);
    CHECK(GetProcAddress(delay, heap_name("DelayFn")) == 0);

    CHECK(LoadLibraryA("delayload_test.dll") == delay);
    CHECK(pg6_win32_load_library_a("C:\\Somewhere\\DELAYLOAD_TEST") == delay);
    CHECK(GetModuleHandleA("delayload_test") == delay);
    CHECK(GetProcAddress(delay, heap_name("DelayFn")) == proc_address(&test_export_alpha));

    CHECK(FreeLibrary(delay));
    CHECK(GetModuleHandleA("delayload_test") == delay);  // Still one reference.
    CHECK(pg6_win32_free_library(delay) == 1);
    CHECK(GetModuleHandleA("delayload_test") == kNullHandle);  // Unloaded.
    SetLastError(ERROR_SUCCESS);
    CHECK(!FreeLibrary(delay));
    CHECK(GetLastError() == ERROR_MOD_NOT_FOUND);
    SetLastError(ERROR_SUCCESS);
    CHECK(GetProcAddress(delay, heap_name("DelayFn")) == 0);
    CHECK(GetLastError() == ERROR_MOD_NOT_FOUND);
    char buf[64] = {};
    CHECK(GetModuleFileNameA(delay, buf, sizeof(buf)) == 0);

    // Loading again restores visibility with the same HMODULE.
    CHECK(LoadLibraryExA("delayload_test.dll", kNullHandle, 0) == delay);
    CHECK(GetModuleHandleA("delayload_test.dll") == delay);
    CHECK(FreeLibrary(delay));

    // Pinned modules stay loaded no matter how often they are freed.
    const Handle k32 = LoadLibraryA("kernel32.dll");
    CHECK(k32 != kNullHandle);
    CHECK(FreeLibrary(k32));
    CHECK(FreeLibrary(k32));
    CHECK(GetModuleHandleA("kernel32.dll") == k32);

    // API-set names load their host module.
    CHECK(LoadLibraryA("api-ms-win-core-file-l1-2-0.dll") == k32);

    // Failures.
    SetLastError(ERROR_SUCCESS);
    CHECK(LoadLibraryA("definitely_missing.dll") == kNullHandle);
    CHECK(GetLastError() == ERROR_MOD_NOT_FOUND);
    SetLastError(ERROR_SUCCESS);
    CHECK(LoadLibraryW(u"definitely_missing") == kNullHandle);
    CHECK(GetLastError() == ERROR_MOD_NOT_FOUND);
    SetLastError(ERROR_SUCCESS);
    CHECK(LoadLibraryA(nullptr) == kNullHandle);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    SetLastError(ERROR_SUCCESS);
    CHECK(LoadLibraryExA("kernel32.dll", 0x1234, 0) == kNullHandle);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_load_library_ex_a("missing", kNullHandle, 0) == kNullHandle);
    SetLastError(ERROR_SUCCESS);
    CHECK(!FreeLibrary(kNullHandle));
    CHECK(GetLastError() == ERROR_MOD_NOT_FOUND);
    CHECK(!FreeLibrary(0xABCDEF000ULL));
}

// ---------------------------------------------------------------------------
// 10. GetProcAddress by name / ordinal / missing, registration helpers
// ---------------------------------------------------------------------------

void test_get_proc_address() {
    const Handle k32 = GetModuleHandleA("kernel32.dll");
    CHECK(GetProcAddress(k32, heap_name("HeapAlloc")) == proc_address(&pg6_win32_heap_alloc));
    CHECK(GetProcAddress(k32, heap_name("HeapReAlloc")) == proc_address(&pg6_win32_heap_realloc));
    CHECK(GetProcAddress(k32, heap_name("GetProcAddress")) ==
          proc_address(&pg6_win32_get_proc_address));
    CHECK(pg6_win32_get_proc_address(k32, heap_name("LoadLibraryExW")) ==
          proc_address(&pg6_win32_load_library_ex_w));

    // The explicit by-name / by-ordinal paths accept any pointer (string literals included).
    CHECK(GetProcAddressByName(k32, "GetModuleFileNameW") ==
          proc_address(&pg6_win32_get_module_file_name_w));
    CHECK(pg6_loader_get_proc_address_by_name(k32, "HeapFree") ==
          proc_address(&pg6_win32_heap_free));
    CHECK(pg6_loader_get_proc_address_by_name(k32, nullptr) == 0);
    SetLastError(ERROR_SUCCESS);
    CHECK(pg6_loader_get_proc_address_by_ordinal(k32, 1) == 0);
    CHECK(GetLastError() == ERROR_PROC_NOT_FOUND);

    // Call through a resolved address like an import thunk would.
    using HeapAllocFn = uint64_t (*)(Pg6Handle, uint32_t, uint64_t);
    using HeapFreeFn = int32_t (*)(Pg6Handle, uint32_t, uint64_t);
    auto heap_alloc = reinterpret_cast<HeapAllocFn>(
        static_cast<uintptr_t>(GetProcAddress(k32, heap_name("HeapAlloc"))));
    auto heap_free = reinterpret_cast<HeapFreeFn>(
        static_cast<uintptr_t>(resolve_import("api-ms-win-core-heap-l1-1-0.dll", "HeapFree")));
    CHECK(heap_alloc != nullptr);
    CHECK(heap_free != nullptr);
    if (heap_alloc != nullptr && heap_free != nullptr) {
        const uint64_t p = heap_alloc(GetProcessHeap(), HEAP_ZERO_MEMORY, 40);
        CHECK(p != 0);
        CHECK(all_equal(p, 40, 0));
        CHECK(heap_free(GetProcessHeap(), 0, p) == 1);
    }

    // Missing exports and case sensitivity.
    SetLastError(ERROR_SUCCESS);
    CHECK(GetProcAddress(k32, heap_name("NoSuchExport")) == 0);
    CHECK(GetLastError() == ERROR_PROC_NOT_FOUND);
    CHECK(GetProcAddress(k32, heap_name("heapalloc")) == 0);
    CHECK(GetProcAddress(k32, heap_name("")) == 0);
    SetLastError(ERROR_SUCCESS);
    CHECK(GetProcAddress(0x12345678000ULL, heap_name("HeapAlloc")) == 0);
    CHECK(GetLastError() == ERROR_MOD_NOT_FOUND);

    // Name + ordinal registration through the helper.
    const size_t n = register_exports(
        "ordtest.dll",
        {
            export_entry("Alpha", &test_export_alpha, 1),
            ExportEntry{nullptr, 7, proc_address(&test_export_beta)},
            export_entry("BetaByName", &test_export_beta),
        });
    CHECK(n == 3);
    const Handle ord = GetModuleHandleA("OrdTest");
    CHECK(ord != kNullHandle);
    CHECK(GetProcAddress(ord, heap_name("Alpha")) == proc_address(&test_export_alpha));
    CHECK(GetProcAddress(ord, reinterpret_cast<const char*>(uintptr_t{1})) ==
          proc_address(&test_export_alpha));
    CHECK(GetProcAddress(ord, reinterpret_cast<const char*>(uintptr_t{7})) ==
          proc_address(&test_export_beta));
    CHECK(GetProcAddressByOrdinal(ord, 7) == proc_address(&test_export_beta));
    SetLastError(ERROR_SUCCESS);
    CHECK(GetProcAddress(ord, reinterpret_cast<const char*>(uintptr_t{9})) == 0);
    CHECK(GetLastError() == ERROR_PROC_NOT_FOUND);
    CHECK(GetProcAddress(ord, nullptr) == 0);  // Ordinal 0 never exists.

    using IntFn = int32_t (*)();
    auto beta = reinterpret_cast<IntFn>(
        static_cast<uintptr_t>(resolve_import("ORDTEST", "BetaByName")));
    CHECK(beta != nullptr && beta() == 22);
    auto alpha = reinterpret_cast<IntFn>(
        static_cast<uintptr_t>(resolve_import_ordinal("ordtest.dll", 1)));
    CHECK(alpha != nullptr && alpha() == 11);

    // Invalid registrations.
    CHECK(!register_export("ordtest.dll", "Zero", 0, 0));
    CHECK(!register_export("ordtest.dll", "TooBig", 0x10000, 0x1000));
    CHECK(!register_export("ordtest.dll", "", 0, 0x1000));
    CHECK(!register_export("", "NoModule", 0, 0x1000));
    const ExportEntry mixed[] = {
        {"Good", 0, 0x2000},
        {nullptr, 0, 0x3000},
    };
    CHECK(register_exports("ordtest.dll", mixed, 2) == 1);
    CHECK(register_exports("ordtest.dll", nullptr, 2) == 0);

    // Re-registration replaces the address.
    CHECK(register_export("ordtest.dll", "Good", 0, 0x4000));
    CHECK(GetProcAddress(ord, heap_name("Good")) == 0x4000);

    // Exports of the main image via GetProcAddress(NULL, ...).
    CHECK(register_export("gta6.exe", "MainExport", 3, 0x140001000ULL));
    CHECK(GetProcAddress(kNullHandle, heap_name("MainExport")) == 0x140001000ULL);
    CHECK(GetProcAddressByOrdinal(kNullHandle, 3) == 0x140001000ULL);

    // C-ABI registration and resolution.
    CHECK(pg6_loader_register_export("d3d12_xs.dll", "D3D12CreateDevice", 0, 0x5000) == 1);
    CHECK(pg6_loader_register_export("d3d12_xs.dll", nullptr, 0, 0x5000) == 0);
    const Pg6ExportEntry c_entries[] = {
        {"D3D12GetDebugInterface", 0, 0x5010},
        {nullptr, 42, 0x5020},
    };
    CHECK(pg6_loader_register_exports("d3d12_xs.dll", c_entries, 2) == 2);
    CHECK(pg6_loader_resolve_import("D3D12_XS", "D3D12CreateDevice") == 0x5000);
    CHECK(pg6_loader_resolve_import("d3d12_xs.dll", "D3D12GetDebugInterface") == 0x5010);
    CHECK(pg6_loader_resolve_import_ordinal("d3d12_xs.dll", 42) == 0x5020);
    SetLastError(ERROR_SUCCESS);
    CHECK(pg6_loader_resolve_import("missing_dll.dll", "X") == 0);
    CHECK(GetLastError() == ERROR_MOD_NOT_FOUND);
    SetLastError(ERROR_SUCCESS);
    CHECK(pg6_loader_resolve_import("d3d12_xs.dll", "Missing") == 0);
    CHECK(GetLastError() == ERROR_PROC_NOT_FOUND);
    CHECK(pg6_loader_resolve_import(nullptr, nullptr) == 0);
    CHECK(pg6_loader_register_api_set_rule("api-ms-win-pg6c-", "d3d12_xs.dll") == 1);
    CHECK(pg6_loader_resolve_import("api-ms-win-pg6c-l1-1-0.dll", "D3D12CreateDevice") == 0x5000);
    CHECK(pg6_loader_register_api_set_rule(nullptr, "x.dll") == 0);
}

// ---------------------------------------------------------------------------
// 11. Concurrent lookups while exports are being registered
// ---------------------------------------------------------------------------

void test_concurrent_registration() {
    const Handle k32 = GetModuleHandleA("kernel32.dll");
    const uint64_t expected_alloc = proc_address(&pg6_win32_heap_alloc);
    constexpr int kExports = 1500;
    constexpr int kReaders = 6;

    std::atomic<bool> done{false};
    std::atomic<int> errors{0};
    std::atomic<int> hits{0};

    auto reader = [&](int id) {
        XorShift rng{0xC0FFEEULL + static_cast<uint64_t>(id) * 7919u};
        while (!done.load(std::memory_order_acquire)) {
            if (GetProcAddressByName(k32, "HeapAlloc") != expected_alloc) {
                errors.fetch_add(1);
            }
            if (GetModuleHandleA("api-ms-win-core-heap-l1-1-0.dll") != k32) {
                errors.fetch_add(1);
            }
            const int i = static_cast<int>(rng.next() % kExports);
            const std::string name = "Fn" + std::to_string(i);
            const uint64_t got = resolve_import("concurrent_test.dll", name);
            if (got != 0) {
                hits.fetch_add(1, std::memory_order_relaxed);
                if (got != 0x100000ULL + static_cast<uint64_t>(i) * 16u) {
                    errors.fetch_add(1);
                }
            }
            const uint64_t by_ord =
                resolve_import_ordinal("concurrent_test.dll", static_cast<uint32_t>(i + 1));
            if (by_ord != 0 && by_ord != 0x100000ULL + static_cast<uint64_t>(i) * 16u) {
                errors.fetch_add(1);
            }
        }
    };

    std::vector<std::thread> readers;
    for (int r = 0; r < kReaders; ++r) {
        readers.emplace_back(reader, r);
    }

    bool all_registered = true;
    for (int i = 0; i < kExports; ++i) {
        const std::string name = "Fn" + std::to_string(i);
        all_registered &= register_export("concurrent_test.dll", name,
                                          static_cast<uint32_t>(i + 1),
                                          0x100000ULL + static_cast<uint64_t>(i) * 16u);
        if (i % 100 == 0) {
            all_registered &= register_virtual_module("concurrent_extra_" + std::to_string(i)) !=
                              kNullHandle;
        }
    }
    done.store(true, std::memory_order_release);
    for (auto& t : readers) {
        t.join();
    }

    CHECK(all_registered);
    CHECK(errors.load() == 0);
    const Handle mod = GetModuleHandleA("concurrent_test.dll");
    CHECK(mod != kNullHandle);
    for (int i = 0; i < kExports; ++i) {
        const uint64_t expected = 0x100000ULL + static_cast<uint64_t>(i) * 16u;
        if (GetProcAddressByName(mod, "Fn" + std::to_string(i)) != expected ||
            GetProcAddressByOrdinal(mod, static_cast<uint32_t>(i + 1)) != expected) {
            CHECK(false);
            break;
        }
    }
}

}  // namespace

int main() {
    test_heap_basics();
    test_heap_zeroing();
    test_heap_realloc_semantics();
    test_private_heaps();
    test_heap_invalid_arguments();
    test_heap_stress();
    test_module_lookup();
    test_wide_variants();
    test_load_library_refcounts();
    test_get_proc_address();
    test_concurrent_registration();

    if (g_failures != 0) {
        std::fprintf(stderr, "test_win32_loader failed with %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("All playgta6_win32_loader_tests passed.\n");
    return 0;
}
