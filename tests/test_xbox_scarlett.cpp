#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "abi/playgta6_abi.h"
#include "os/game_input.h"
#include "os/httpfs_mailbox.h"
#include "os/surface_blocks.h"
#include "os/win32_kernel.h"
#include "os/win32_loader.h"
#include "os/xbox_scarlett.h"
#include "os/xgameruntime.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_xbox_scarlett.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6::os;
using namespace playgta6::os::scarlett;
using gdk::E_INVALIDARG_HR;
using gdk::E_POINTER_HR;
using gdk::S_OK_HR;

bool approx_eq(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

// Hook counters for testing XMemAlloc/XMemFree hook override.
int g_hook_alloc_calls = 0;
int g_hook_free_calls = 0;

void* test_alloc_hook(size_t size, uint64_t attrs) {
    ++g_hook_alloc_calls;
    return pg6_xbox_XMemAllocDefault(size, attrs);
}

void test_free_hook(void* ptr, uint64_t attrs) {
    ++g_hook_free_calls;
    pg6_xbox_XMemFreeDefault(ptr, attrs);
}

// ---------------------------------------------------------------------------
// 1. xmem.dll — all 12 imports
// ---------------------------------------------------------------------------

void test_xmem_heap_and_hooks() {
    xmem_reset_for_testing();

    // XMemAllocDefault with 256-byte alignment (code 7 -> 2^(7+1) = 256) and zero-init.
    const uint64_t attrs256 = make_xmem_attributes(42, 0, 7, 0, true);
    void* p1 = pg6_xbox_XMemAllocDefault(512, attrs256);
    CHECK(p1 != nullptr);
    CHECK((reinterpret_cast<uintptr_t>(p1) % 256u) == 0u);
    auto* b1 = static_cast<uint8_t*>(p1);
    bool all_zero = true;
    for (size_t i = 0; i < 512; ++i) {
        if (b1[i] != 0) all_zero = false;
    }
    CHECK(all_zero);

    XMemAllocInfo info{};
    CHECK(xmem_get_allocation_info(p1, &info));
    CHECK(info.size == 512);
    CHECK(info.alignment == 256);
    CHECK(info.attributes == attrs256);

    // XMemAlloc (unhooked) with 4 KiB alignment (code 11 -> 4096).
    const uint64_t attrs4k = make_xmem_attributes(7, 1, 11, 2, false);
    void* p2 = pg6_xbox_XMemAlloc(4096, attrs4k);
    CHECK(p2 != nullptr);
    CHECK((reinterpret_cast<uintptr_t>(p2) % 4096u) == 0u);
    CHECK(xmem_get_allocation_info(p2, &info));
    CHECK(info.size == 4096);
    CHECK(info.alignment == 4096);

    // Working set reflects both heap allocations.
    Pg6XMemWorkingSetStats stats{};
    CHECK(pg6_xbox_XMemGetWorkingSetStatistics(&stats) == S_OK_HR);
    CHECK(stats.heap_allocated_bytes == 512 + 4096);
    CHECK(stats.live_heap_allocations == 2);
    CHECK(stats.total_physical_bytes == kConsoleTotalRamBytes);
    CHECK(stats.title_budget_bytes == kConsoleTitleBudgetBytes);
    CHECK(stats.tools_mem_bytes == 0);
    CHECK(stats.available_title_bytes == kConsoleTitleBudgetBytes - (512 + 4096));

    // Tools mem must report unavailable (0).
    CHECK(pg6_xbox_XMemIsToolsMemAvailable() == 0);

    // Null stats rejected.
    CHECK(pg6_xbox_XMemGetWorkingSetStatistics(nullptr) == E_INVALIDARG_HR);

    // Free both and confirm accounting returns to 0.
    pg6_xbox_XMemFreeDefault(p1, attrs256);
    pg6_xbox_XMemFree(p2, attrs4k);
    CHECK(!xmem_get_allocation_info(p1, nullptr));
    CHECK(pg6_xbox_XMemGetWorkingSetStatistics(&stats) == S_OK_HR);
    CHECK(stats.heap_allocated_bytes == 0);
    CHECK(stats.live_heap_allocations == 0);
    CHECK(stats.peak_working_set_bytes >= 512 + 4096);

    // Safe no-op on null and double-free.
    pg6_xbox_XMemFree(nullptr, 0);
    pg6_xbox_XMemFreeDefault(p1, attrs256);

    // Custom __XMemAllocationHooks__ delegation.
    g_hook_alloc_calls = 0;
    g_hook_free_calls = 0;
    xmem_set_allocation_hooks(&test_alloc_hook, &test_free_hook);
    void* p3 = pg6_xbox_XMemAlloc(128, 0);
    CHECK(p3 != nullptr);
    CHECK(g_hook_alloc_calls == 1);
    pg6_xbox_XMemFree(p3, 0);
    CHECK(g_hook_free_calls == 1);
    xmem_set_allocation_hooks(nullptr, nullptr);
}

void test_xmem_physical_virtual_and_transfer() {
    xmem_reset_for_testing();

    // Allocate 3 x 4 KiB physical pages.
    size_t num_pages = 3;
    uint64_t handles[3]{};
    // (attributes, &count, pfns): the title's argument order (O27).
    CHECK(pg6_xbox_XMemAllocatePhysicalPages(kXMemFlagPage4K, &num_pages, handles) == 1);
    CHECK(handles[0] != 0 && handles[1] != 0 && handles[2] != 0);
    CHECK(handles[0] != handles[1] && handles[1] != handles[2]);

    // Invalid physical page allocation arguments rejected.
    size_t zero_pages = 0;
    CHECK(pg6_xbox_XMemAllocatePhysicalPages(0, &zero_pages, handles) == 0);
    CHECK(pg6_xbox_XMemAllocatePhysicalPages(0, nullptr, handles) == 0);

    // Contiguous mapping via XMemMapPhysicalPages (no address: a fresh block).
    void* mapped = pg6_xbox_XMemMapPhysicalPages(nullptr, 3, handles);
    CHECK(mapped != nullptr);
    auto* mapped_bytes = static_cast<uint8_t*>(mapped);
    std::memset(mapped_bytes + 0 * kXMemPageSize4K, 0x11, kXMemPageSize4K);
    std::memset(mapped_bytes + 1 * kXMemPageSize4K, 0x22, kXMemPageSize4K);
    std::memset(mapped_bytes + 2 * kXMemPageSize4K, 0x33, kXMemPageSize4K);

    // Unmap syncs written data back into the physical pages.
    CHECK(xmem_unmap_physical_pages(mapped));

    // Reserve a virtual range via XMemVirtualAlloc, then scatter-map pages in reverse order
    // with gaps (offsets: page 0 -> 8 KiB, page 2 -> 0 KiB).
    void* va_reserved = pg6_xbox_XMemVirtualAlloc(
        nullptr, 4 * kXMemPageSize4K, win32::MEM_RESERVE, 0, win32::PAGE_NOACCESS);
    CHECK(va_reserved != nullptr);

    const uint64_t scatter_handles[2] = {handles[2], handles[0]};
    const uint64_t base_va = win32::get_vmem_arena()->guest_address(va_reserved);
    const uint64_t scatter_vas[2] = {base_va, base_va + 2 * kXMemPageSize4K};
    CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(scatter_vas, 2, scatter_handles) == 1);

    const auto* scatter_bytes = static_cast<const uint8_t*>(va_reserved);
    CHECK(scatter_bytes[0] == 0x33);
    CHECK(scatter_bytes[kXMemPageSize4K - 1] == 0x33);
    CHECK(scatter_bytes[2 * kXMemPageSize4K] == 0x11);
    CHECK(scatter_bytes[3 * kXMemPageSize4K - 1] == 0x11);

    // O36: a page without contents mapped at a slot reads zero, whether the slot was never
    // committed (no memset needed) or held another page's bytes before (unmapped first).
    {
        size_t one = 2;
        uint64_t fresh[2]{};
        CHECK(pg6_xbox_XMemAllocatePhysicalPages(kXMemFlagPage4K, &one, fresh) == 1);
        const uint64_t at_gap[1] = {base_va + 1 * kXMemPageSize4K};
        CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(at_gap, 1, &fresh[0]) == 1);
        bool zero = true;
        for (uint64_t i = 0; i < kXMemPageSize4K; ++i) {
            zero = zero && scatter_bytes[kXMemPageSize4K + i] == 0;
        }
        CHECK(zero);
        const uint64_t at_first[1] = {base_va};
        const uint64_t none[1] = {0};
        CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(at_first, 1, none) == 1);  // page 2 leaves
        CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(at_first, 1, &fresh[1]) == 1);
        zero = true;
        for (uint64_t i = 0; i < kXMemPageSize4K; ++i) {
            zero = zero && scatter_bytes[i] == 0;
        }
        CHECK(zero);
        // Page 2 kept its bytes: mapped back at its slot it shows them again.
        const uint64_t back[1] = {handles[2]};
        CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(at_first, 1, back) == 1);
        CHECK(scatter_bytes[0] == 0x33 && scatter_bytes[kXMemPageSize4K - 1] == 0x33);
        CHECK(pg6_xbox_XMemMapPhysicalPagesScatter(at_gap, 1, none) == 1);
        CHECK(pg6_xbox_XMemFreePhysicalPages(2, fresh) == 1);
    }

    // Standalone XMemVirtualAlloc reserve + in-place commit + XMemTransferMemory.
    void* va_buf = pg6_xbox_XMemVirtualAlloc(
        nullptr, 8192, win32::MEM_RESERVE, 0, win32::PAGE_READWRITE);
    CHECK(va_buf != nullptr);
    void* va_committed = pg6_xbox_XMemVirtualAlloc(
        va_buf, 8192, win32::MEM_COMMIT, 0, win32::PAGE_READWRITE);
    CHECK(va_committed == va_buf);

    // The title's reservation shapes: MEM_64K_PAGES | MEM_RESERVE, XMem flags in the 4th argument,
    // the protection (with PAGE_GRAPHICS_* bits) in the 5th.
    void* big = pg6_xbox_XMemVirtualAlloc(nullptr, 256ull << 20, 0x20400000u | win32::MEM_RESERVE,
                                          0x21u, win32::PAGE_READWRITE);
    CHECK(big != nullptr);
    CHECK(reinterpret_cast<uintptr_t>(big) % kXMemPageSize64K == 0);
    void* gpu_only = pg6_xbox_XMemVirtualAlloc(nullptr, 64ull << 20, 0x20400000u | win32::MEM_RESERVE,
                                               0x3u, 0x800u | win32::PAGE_NOACCESS);
    CHECK(gpu_only != nullptr);
    void* graphics_rw = pg6_xbox_XMemVirtualAlloc(nullptr, 1ull << 20, win32::MEM_RESERVE | win32::MEM_COMMIT,
                                                  0u, 0x2000u /*PAGE_GRAPHICS_READWRITE*/);
    CHECK(graphics_rw != nullptr);
    if (graphics_rw != nullptr) {
        static_cast<uint8_t*>(graphics_rw)[100] = 7;  // the CPU view is writable
    }
    CHECK(pg6_xbox_XMemVirtualAlloc(nullptr, 1ull << 20, win32::MEM_RESERVE, 0u, 0x4000u) == nullptr);

    // The title's placeholder region: XMemVirtualAlloc(MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
    // PAGE_NOACCESS), split with VirtualFree(MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER), each piece
    // replaced with MEM_LARGE_PAGES | MEM_PHYSICAL | MEM_REPLACE_PLACEHOLDER | MEM_RESERVE.
    {
        constexpr uint64_t kPiece = 2ull << 20;
        void* region = pg6_xbox_XMemVirtualAlloc(nullptr, 3 * kPiece, 0x42000u, 7u, win32::PAGE_NOACCESS);
        CHECK(region != nullptr);
        // XMem takes and returns host pointers; VirtualFree takes guest addresses.
        auto& arena = *win32::get_vmem_arena();
        const uint64_t base = arena.guest_address(region);
        CHECK(win32::VirtualFree(base, kPiece, win32::MEM_RELEASE | 0x2u));
        CHECK(win32::VirtualFree(base + kPiece, kPiece, win32::MEM_RELEASE | 0x2u));
        for (int i = 0; i < 3; ++i) {
            void* want = arena.translate(base + i * kPiece);
            void* piece = pg6_xbox_XMemVirtualAlloc(want, kPiece, 0x20406000u, 0x27u, 0x1004u);
            CHECK(piece == want);
        }
        // Replacing twice fails: the range is no longer a placeholder.
        CHECK(pg6_xbox_XMemVirtualAlloc(region, kPiece, 0x20406000u, 0x27u, 0x1004u) == nullptr);
        CHECK(win32::VirtualFree(base + kPiece, kPiece, 0x1u | win32::MEM_RELEASE) == false);  // not placeholders
    }

    uint8_t src_pattern[256];
    for (size_t i = 0; i < sizeof(src_pattern); ++i) {
        src_pattern[i] = static_cast<uint8_t>((i * 7u + 3u) & 0xFFu);
    }
    CHECK(pg6_xbox_XMemTransferMemory(va_buf, src_pattern, sizeof(src_pattern), 0) == S_OK_HR);
    CHECK(std::memcmp(va_buf, src_pattern, sizeof(src_pattern)) == 0);
    // Overlapping self-transfer (memmove semantics).
    CHECK(pg6_xbox_XMemTransferMemory(
              static_cast<uint8_t*>(va_buf) + 16, va_buf, 128, 0) == S_OK_HR);
    CHECK(std::memcmp(static_cast<uint8_t*>(va_buf) + 16, src_pattern, 128) == 0);
    // Zero-size & invalid args.
    CHECK(pg6_xbox_XMemTransferMemory(nullptr, nullptr, 0, 0) == S_OK_HR);
    CHECK(pg6_xbox_XMemTransferMemory(nullptr, src_pattern, 16, 0) == E_INVALIDARG_HR);

    // Working-set stats verify physical + virtual + mapping accounting.
    Pg6XMemWorkingSetStats stats{};
    CHECK(pg6_xbox_XMemGetWorkingSetStatistics(&stats) == S_OK_HR);
    CHECK(stats.live_physical_pages == 3);
    CHECK(stats.physical_pages_bytes == 3 * kXMemPageSize4K);
    CHECK(stats.live_virtual_mappings == 2);  // mapped page slots
    CHECK(stats.mapped_physical_bytes == 2 * kXMemPageSize4K);
    CHECK(stats.virtual_committed_bytes >= 8192);

    // Cleanup virtual regions & physical pages.
    CHECK(xmem_virtual_free(va_reserved));
    CHECK(xmem_virtual_free(va_buf));
    CHECK(pg6_xbox_XMemFreePhysicalPages(3, handles) == 1);
    // Double free fails cleanly.
    CHECK(pg6_xbox_XMemFreePhysicalPages(3, handles) == 0);
}

// O36: the fast map path skips the zero-fill of a slot it just committed (fresh commits read as
// zero already; zeroing would make the whole pool view resident), but still zeroes a reused slot.
void test_xmem_fast_map_skips_fresh_zero_fill() {
    auto* arena = win32::get_vmem_arena();
    xmem_reset_for_testing();
    void* res = pg6_xbox_XMemVirtualAlloc(
        nullptr, 2 * kXMemPageSize64K, 0x20400000u | win32::MEM_RESERVE, 0x21u, win32::PAGE_READWRITE);
    CHECK(res != nullptr);
    if (res == nullptr) {
        return;
    }
    const uint64_t base = arena->guest_address(res);
    size_t n = 2;
    uint64_t pfn[2]{};
    CHECK(pg6_xbox_XMemAllocatePhysicalPages(kXMemFlagPage64K, &n, pfn) == 1);
    uint8_t* slot = arena->translate(base);
    const uint64_t before = xmem_window_contiguity().zero_fills;
    CHECK(pg6_xbox_XMemMapPhysicalPages(slot, 1, &pfn[0]) != nullptr);
    CHECK(xmem_window_contiguity().zero_fills == before);  // fresh commit: no memset
    bool zero = true;
    for (uint64_t i = 0; i < kXMemPageSize64K; ++i) {
        zero = zero && slot[i] == 0;
    }
    CHECK(zero);
    // Reused slot: dirty it, unmap, map another content-less page: it reads zero again.
    std::memset(slot, 0x5A, kXMemPageSize64K);
    CHECK(pg6_xbox_XMemMapPhysicalPages(slot, 1, nullptr) != nullptr);
    CHECK(pg6_xbox_XMemMapPhysicalPages(slot, 1, &pfn[1]) != nullptr);
    zero = true;
    for (uint64_t i = 0; i < kXMemPageSize64K; ++i) {
        zero = zero && slot[i] == 0;
    }
    CHECK(zero);
    CHECK(pg6_xbox_XMemMapPhysicalPages(slot, 1, nullptr) != nullptr);
    CHECK(pg6_xbox_XMemFreePhysicalPages(2, pfn) == 1);
    CHECK(xmem_virtual_free(res));
}

// R40: XMemMapPhysicalPages / unmap from many threads at once (their data work runs outside the
// XMem lock): disjoint slots keep every thread's bytes, and threads fighting over the same slots
// and pages leave every page's bytes intact and the accounting balanced. Both reservation kinds:
// a physical one (native: the arena's shared zone) and a plain one (committed slot by slot).
void test_xmem_concurrent_map_unmap() {
    constexpr int kThreads = 8;
    constexpr size_t kOwn = 4;     // slots (and pages) per thread, disjoint
    constexpr size_t kShared = 3;  // slots every thread maps into
    constexpr size_t kPool = 6;    // pages every thread maps
    constexpr int kRounds = 40;
    auto* arena = win32::get_vmem_arena();
    for (const uint32_t type : {0x20400000u | win32::MEM_RESERVE, kXMemFlagLargePages | win32::MEM_RESERVE}) {
        xmem_reset_for_testing();
        const size_t slots = kThreads * kOwn + kShared + kPool;
        void* res = pg6_xbox_XMemVirtualAlloc(nullptr, slots * kXMemPageSize64K, type, 0x21u, win32::PAGE_READWRITE);
        CHECK(res != nullptr);
        if (res == nullptr) {
            continue;
        }
        const uint64_t base = arena->guest_address(res);
        auto slot_va = [&](size_t i) { return base + i * kXMemPageSize64K; };
        auto bytes = [&](size_t i) { return arena->translate(slot_va(i)); };
        size_t n = kThreads * kOwn + kPool;
        std::vector<uint64_t> pfn(n);
        CHECK(pg6_xbox_XMemAllocatePhysicalPages(kXMemFlagPage64K, &n, pfn.data()) == 1);
        const uint64_t* pool = pfn.data() + kThreads * kOwn;
        // The pool pages' bytes: a marker per page, written through a slot of their own.
        const size_t pool_slot = kThreads * kOwn + kShared;
        for (size_t p = 0; p < kPool; ++p) {
            CHECK(pg6_xbox_XMemMapPhysicalPages(bytes(pool_slot + p), 1, &pool[p]) != nullptr);
            std::memset(bytes(pool_slot + p), static_cast<int>(0xA0 + p), kXMemPageSize64K);
            CHECK(pg6_xbox_XMemMapPhysicalPages(bytes(pool_slot + p), 1, nullptr) != nullptr);
        }
        std::atomic<int> bad{0};
        std::vector<std::thread> workers;
        for (int t = 0; t < kThreads; ++t) {
            workers.emplace_back([&, t] {
                for (int r = 0; r < kRounds; ++r) {
                    // Own slots: map, write, unmap (bytes to the store), map back, check.
                    for (size_t k = 0; k < kOwn; ++k) {
                        const size_t s = t * kOwn + k;
                        const uint64_t page = pfn[s];
                        const auto mark = static_cast<uint8_t>(t * 31 + r * 7 + k);
                        if (pg6_xbox_XMemMapPhysicalPages(bytes(s), 1, &page) == nullptr) {
                            ++bad;
                            continue;
                        }
                        bytes(s)[r * 97 + k] = mark;
                        pg6_xbox_XMemMapPhysicalPages(bytes(s), 1, nullptr);
                        pg6_xbox_XMemMapPhysicalPages(bytes(s), 1, &page);
                        bad += bytes(s)[r * 97 + k] == mark ? 0 : 1;
                        const uint64_t at[] = {slot_va(s)};
                        const uint64_t none[] = {0};
                        pg6_xbox_XMemMapPhysicalPagesScatter(at, 1, none);
                    }
                    // Shared slots and pages: map over whatever is there, unmap whatever is there.
                    const size_t s = kThreads * kOwn + static_cast<size_t>(t + r) % kShared;
                    const uint64_t page = pool[static_cast<size_t>(t * 5 + r) % kPool];
                    bad += pg6_xbox_XMemMapPhysicalPages(bytes(s), 1, &page) != nullptr ? 0 : 1;
                    if ((r & 1) != 0) {
                        pg6_xbox_XMemMapPhysicalPages(bytes(s), 1, nullptr);
                    }
                }
            });
        }
        for (auto& w : workers) {
            w.join();
        }
        CHECK(bad.load() == 0);
        for (size_t k = 0; k < kShared; ++k) {
            pg6_xbox_XMemMapPhysicalPages(bytes(kThreads * kOwn + k), 1, nullptr);
        }
        Pg6XMemWorkingSetStats stats{};
        CHECK(pg6_xbox_XMemGetWorkingSetStatistics(&stats) == S_OK_HR);
        CHECK(stats.live_virtual_mappings == 0);
        CHECK(stats.mapped_physical_bytes == 0);
        // Every page still holds its bytes: the threads' last marks, the pool's markers.
        for (int t = 0; t < kThreads; ++t) {
            for (size_t k = 0; k < kOwn; ++k) {
                const size_t s = t * kOwn + k;
                CHECK(pg6_xbox_XMemMapPhysicalPages(bytes(s), 1, &pfn[s]) != nullptr);
                for (int r = 0; r < kRounds; ++r) {
                    CHECK(bytes(s)[r * 97 + k] == static_cast<uint8_t>(t * 31 + r * 7 + k));
                }
            }
        }
        for (size_t p = 0; p < kPool; ++p) {
            CHECK(pg6_xbox_XMemMapPhysicalPages(bytes(pool_slot + p), 1, &pool[p]) != nullptr);
            const uint8_t* b = bytes(pool_slot + p);
            CHECK(b[0] == 0xA0 + p && b[kXMemPageSize64K / 2] == 0xA0 + p && b[kXMemPageSize64K - 1] == 0xA0 + p);
        }
        CHECK(xmem_virtual_free(res));
        CHECK(pg6_xbox_XMemFreePhysicalPages(pfn.size(), pfn.data()) == 1);
    }
    xmem_reset_for_testing();
}

// ---------------------------------------------------------------------------
// 2. XCRDAPI.dll — DirectStorage open/enqueue/submit round-trip
// ---------------------------------------------------------------------------

void test_dstorage_virtual_and_httpfs() {
    reset_dstorage_for_testing();

    // Obtain factory via DStorageGetFactory.
    CHECK(pg6_xbox_DStorageGetFactory(&kIidIDStorageFactory, nullptr) == E_POINTER_HR);
    void* bogus_iface = nullptr;
    CHECK(pg6_xbox_DStorageGetFactory(&kIidIGameInput, &bogus_iface) == E_NOINTERFACE_HR);

    Pg6IDStorageFactory* factory = nullptr;
    CHECK(pg6_xbox_DStorageGetFactory(
              &kIidIDStorageFactory, reinterpret_cast<void**>(&factory)) == S_OK_HR);
    CHECK(factory != nullptr);

    // QueryInterface & AddRef/Release.
    Pg6IUnknown* unk = nullptr;
    CHECK(factory->QueryInterface(kIidIUnknown, reinterpret_cast<void**>(&unk)) == S_OK_HR);
    CHECK(unk != nullptr);
    CHECK(unk->Release() == 1u);

    // Staging buffer size validation.
    CHECK(factory->SetStagingBufferSize(1024) == E_INVALIDARG_HR);
    CHECK(factory->SetStagingBufferSize(16u * 1024u * 1024u) == S_OK_HR);
    CHECK(factory->GetStagingBufferSize() == 16u * 1024u * 1024u);

    // Register a virtual file and open it via UTF-16 OpenFile.
    const std::string payload = "DIRECTSTORAGE-XBOX-SCARLETT-PAYLOAD-0123456789ABCDEF";
    register_dstorage_virtual_file(
        "data/common/world.rpf",
        std::vector<uint8_t>(payload.begin(), payload.end()));

    Pg6IDStorageFile* vfile = nullptr;
    CHECK(factory->OpenFile(u"G:\\data\\common\\world.rpf",
                            kIidIDStorageFile,
                            reinterpret_cast<void**>(&vfile)) == S_OK_HR);
    CHECK(vfile != nullptr);
    Pg6DStorageFileInfo finfo{};
    CHECK(vfile->GetFileInformation(&finfo) == S_OK_HR);
    CHECK(finfo.file_size == payload.size());

    // Missing file returns E_FILE_NOT_FOUND_HR.
    Pg6IDStorageFile* missing_file = nullptr;
    CHECK(factory->OpenFileA("data/does_not_exist.bin",
                             kIidIDStorageFile,
                             reinterpret_cast<void**>(&missing_file)) == E_FILE_NOT_FOUND_HR);
    CHECK(missing_file == nullptr);

    // Create queue & status array.
    Pg6DStorageQueueDesc qdesc{};
    qdesc.capacity = 64;
    qdesc.name = "test_dstorage_queue";
    Pg6IDStorageQueue* queue = nullptr;
    CHECK(factory->CreateQueue(
              &qdesc, kIidIDStorageQueue, reinterpret_cast<void**>(&queue)) == S_OK_HR);
    CHECK(queue != nullptr);

    Pg6IDStorageStatusArray* status = nullptr;
    CHECK(factory->CreateStatusArray(
              4, "test_status", kIidIDStorageStatusArray, reinterpret_cast<void**>(&status)) ==
          S_OK_HR);
    CHECK(status != nullptr);

    // Enqueue two reads (one normal, one cancelled by tag), a status slot, and a Win32 event signal.
    char out1[16]{};
    char out_cancelled[16]{};
    std::memset(out_cancelled, 'Z', sizeof(out_cancelled));

    queue->EnqueueRead(vfile, 14, out1, 13);  // "XBOX-SCARLETT"

    Pg6DStorageRequest cancelled_req{};
    cancelled_req.file = vfile;
    cancelled_req.offset = 0;
    cancelled_req.size = 13;
    cancelled_req.destination = out_cancelled;
    cancelled_req.cancellation_tag = 0xCAFE0001ULL;
    queue->EnqueueRequest(&cancelled_req);
    queue->CancelRequestsWithTag(0xFFFF0000ULL, 0xCAFE0000ULL);

    Pg6Handle done_event = win32::CreateEventA(nullptr, true, false, nullptr);
    CHECK(done_event != win32::kNullHandle);

    queue->EnqueueStatus(status, 0);
    queue->EnqueueSignal(done_event, 1);

    CHECK(queue->QueryPendingCount() == 4);
    CHECK(!status->IsComplete(0));

    queue->Submit();

    CHECK(queue->QueryPendingCount() == 0);
    CHECK(status->IsComplete(0));
    CHECK(status->GetHResult(0) == S_OK_HR);
    CHECK(std::memcmp(out1, "XBOX-SCARLETT", 13) == 0);
    CHECK(out_cancelled[0] == 'Z');  // Cancelled request was not executed.
    CHECK(win32::WaitForSingleObject(done_event, 0) == win32::WAIT_OBJECT_0);
    CHECK(win32::CloseHandle(done_event));

    // Test out-of-bounds read error recording.
    char small_buf[8]{};
    queue->EnqueueRead(vfile, 1000, small_buf, 8);
    queue->EnqueueStatus(status, 1);
    queue->Submit();
    CHECK(status->IsComplete(1));
    CHECK(status->GetHResult(1) == E_EOF_HR);
    Pg6DStorageErrorRecord err{};
    CHECK(queue->RetrieveErrorRecord(&err) == 1u);
    CHECK(err.first_failure_hresult == E_EOF_HR);
    CHECK(err.first_failure_offset == 1000u);

    // Also test HTTPFS mailbox backing end-to-end.
    HttpfsMailbox mailbox{};
    HttpfsMailboxClient::init_mailbox(mailbox);
    const std::string httpfs_data = "HTTPFS-STREAMED-ASSET-BLOCK-9999";
    HttpfsMailboxClient client(&mailbox, [&](HttpfsMailbox& mb) {
        if (mb.op == static_cast<uint32_t>(HttpfsOp::Open)) {
            const std::string req_path(reinterpret_cast<const char*>(mb.payload),
                                       static_cast<size_t>(mb.length));
            if (req_path == "stream/chunk0.bin") {
                HttpfsMailboxClient::complete_response(mb, 7);
            } else {
                HttpfsMailboxClient::complete_response(mb, -2, HttpfsStatus::Error);
            }
        } else if (mb.op == static_cast<uint32_t>(HttpfsOp::Read)) {
            if (mb.handle == 7 && mb.offset + mb.length <= httpfs_data.size()) {
                std::memcpy(mb.payload, httpfs_data.data() + mb.offset, static_cast<size_t>(mb.length));
                HttpfsMailboxClient::complete_response(mb, static_cast<int64_t>(mb.length));
            } else {
                HttpfsMailboxClient::complete_response(mb, -1, HttpfsStatus::Error);
            }
        } else if (mb.op == static_cast<uint32_t>(HttpfsOp::Close)) {
            HttpfsMailboxClient::complete_response(mb, 0);
        }
    });
    win32::set_httpfs_client(&client);

    Pg6IDStorageFile* http_file = nullptr;
    CHECK(factory->OpenFileA("stream/chunk0.bin",
                             kIidIDStorageFile,
                             reinterpret_cast<void**>(&http_file)) == S_OK_HR);
    CHECK(http_file != nullptr);
    char http_out[32]{};
    queue->EnqueueRead(http_file, 7, http_out, 14);  // "STREAMED-ASSET"
    queue->EnqueueStatus(status, 2);
    queue->Submit();
    CHECK(status->IsComplete(2));
    CHECK(status->GetHResult(2) == S_OK_HR);
    CHECK(std::memcmp(http_out, "STREAMED-ASSET", 14) == 0);

    http_file->Release();
    win32::set_httpfs_client(nullptr);
    vfile->Release();
    status->Release();
    queue->Release();
    factory->Release();
    reset_dstorage_for_testing();
}

// ---------------------------------------------------------------------------
// 3. xg_xs.dll — XGCreateTextureComputer & XGSuggestSwizzleMode
// ---------------------------------------------------------------------------

// os/surface_blocks.h: 64 KiB block dimensions and padded sizes; the XG computer's Standard64KB
// mode uses them for the surfaces they cover and keeps the linear layout otherwise.
void test_surface_blocks() {
    namespace sb = playgta6::os::surface;
    const uint32_t expected[5][3] = {{1, 256, 256}, {2, 256, 128}, {4, 128, 128}, {8, 128, 64}, {16, 64, 64}};
    for (const auto& e : expected) {
        sb::BlockDims d{};
        CHECK(sb::block64k_dims(e[0], &d) && d.width == e[1] && d.height == e[2]);
        CHECK(static_cast<uint64_t>(d.width) * d.height * e[0] == sb::kBlockBytes);
    }
    CHECK(!sb::block64k_dims(3, nullptr) && !sb::block64k_dims(12, nullptr));

    sb::Block64kLayout l{};
    CHECK(sb::block64k_layout_2d(3840, 2160, 4, &l));
    CHECK(l.padded_width == 3840 && l.padded_height == 2176 && l.row_pitch_bytes == 15360);
    CHECK(l.size_bytes == 33423360ull && l.size_bytes == 510 * sb::kBlockBytes);
    CHECK(sb::block64k_layout_2d(1920, 1080, 4, &l));
    CHECK(l.padded_width == 1920 && l.padded_height == 1152 && l.size_bytes == 8847360ull);
    CHECK(l.size_bytes == 135 * sb::kBlockBytes);
    // Smaller than one block in a dimension: not covered.
    CHECK(!sb::block64k_layout_2d(64, 4096, 4, &l));
    CHECK(!sb::block64k_layout_2d(4096, 100, 4, &l));

    // XG computer, Standard64KB: covered surfaces are padded, others keep the linear layout.
    Pg6XgTextureDesc desc{};
    desc.width = 3840;
    desc.height = 2160;
    desc.format = kDxgiFormatR8G8B8A8Unorm;
    desc.dimension = kXgDimensionTexture2D;
    desc.tile_mode = kXgTileModeStandard64KB;
    Pg6IXGTextureAddressComputer* computer = nullptr;
    CHECK(pg6_xbox_XGCreateTextureComputer(&desc, &computer) == S_OK_HR);
    Pg6XgResourceLayout layout{};
    CHECK(computer->GetResourceLayout(&layout) == S_OK_HR);
    CHECK(layout.size_bytes == 33423360ull && layout.base_alignment_bytes == sb::kBlockBytes);
    CHECK(layout.mips[0].row_pitch_bytes == 15360 && layout.mips[0].padded_rows == 2176);
    computer->Release();
    desc.mip_levels = 2;  // mip chains are not covered: the linear layout
    CHECK(pg6_xbox_XGCreateTextureComputer(&desc, &computer) == S_OK_HR);
    CHECK(computer->GetResourceLayout(&layout) == S_OK_HR);
    CHECK(layout.mips[0].padded_rows == 2160);
    computer->Release();
}

void test_xg_xs_texture_computer() {
    // XGSuggestSwizzleMode returns linear tile mode cleanly for valid descs, invalid for null.
    CHECK(pg6_xbox_XGSuggestSwizzleMode(nullptr) == kXgTileModeInvalid);

    Pg6XgTextureDesc desc{};
    desc.width = 256;
    desc.height = 128;
    desc.depth_or_array_size = 2;
    desc.mip_levels = 4;
    desc.format = kDxgiFormatR8G8B8A8Unorm;
    desc.dimension = kXgDimensionTexture2D;
    desc.tile_mode = pg6_xbox_XGSuggestSwizzleMode(&desc);
    CHECK(desc.tile_mode == kXgTileModeLinear);

    Pg6IXGTextureAddressComputer* computer = nullptr;
    CHECK(pg6_xbox_XGCreateTextureComputer(&desc, nullptr) == E_POINTER_HR);
    CHECK(pg6_xbox_XGCreateTextureComputer(nullptr, &computer) == E_INVALIDARG_HR);
    CHECK(pg6_xbox_XGCreateTextureComputer(&desc, &computer) == S_OK_HR);
    CHECK(computer != nullptr);

    Pg6XgResourceLayout layout{};
    CHECK(computer->GetResourceLayout(&layout) == S_OK_HR);
    CHECK(layout.mip_levels == 4);
    CHECK(layout.array_size == 2);
    CHECK(layout.tile_mode == kXgTileModeLinear);
    CHECK(layout.base_alignment_bytes == 256);

    // Mip 0: 256x128 RGBA8 -> row pitch 1024 (256-aligned), slice 131072, 2 slices = 262144.
    CHECK(layout.mips[0].offset_bytes == 0);
    CHECK(layout.mips[0].width == 256 && layout.mips[0].height == 128);
    CHECK(layout.mips[0].row_pitch_bytes == 1024);
    CHECK(layout.mips[0].slice_size_bytes == 1024ULL * 128ULL);
    CHECK(layout.mips[0].size_bytes == 1024ULL * 128ULL * 2ULL);

    // Mip 1: 128x64 RGBA8 -> row pitch 512, slice 32768.
    CHECK(layout.mips[1].offset_bytes == 262144);
    CHECK(layout.mips[1].row_pitch_bytes == 512);
    CHECK(layout.mips[1].slice_size_bytes == 512ULL * 64ULL);

    // Mip 2: 64x32 RGBA8 -> row pitch 256, slice 8192.
    CHECK(layout.mips[2].row_pitch_bytes == 256);
    CHECK(layout.mips[2].slice_size_bytes == 256ULL * 32ULL);

    // Mip 3: 32x16 RGBA8 -> 32*4 = 128, padded up to 256-byte row pitch!
    CHECK(layout.mips[3].row_pitch_bytes == 256);
    CHECK(layout.mips[3].slice_size_bytes == 256ULL * 16ULL);
    CHECK(computer->GetResourceSizeBytes() == layout.size_bytes);
    CHECK(computer->GetMipLevelOffsetBytes(0, 1) == layout.mips[1].offset_bytes);
    CHECK(computer->GetMipLevelRowPitchBytes(3) == 256);
    CHECK(computer->GetMipLevelSliceSizeBytes(0) == 1024ULL * 128ULL);

    // Texel element offset calculation (mip 0, x=5, y=3, slice=1).
    const uint64_t expected_off = 1ULL * (1024ULL * 128ULL) + 3ULL * 1024ULL + 5ULL * 4ULL;
    CHECK(computer->GetTexelElementOffsetBytes(0, 0, 5, 3, 1, 0) == expected_off);

    // Subresource copy round-trip (mip 2, slice 1 -> subresource = 1 * 4 + 2 = 6).
    std::vector<uint8_t> resource_mem(static_cast<size_t>(layout.size_bytes), 0u);
    std::vector<uint8_t> src_tex(64 * 32 * 4);
    for (size_t i = 0; i < src_tex.size(); ++i) {
        src_tex[i] = static_cast<uint8_t>((i * 13u + 5u) & 0xFFu);
    }
    CHECK(computer->CopyIntoSubresource(
              resource_mem.data(), 0, 6, src_tex.data(), 64 * 4, 64 * 32 * 4) == S_OK_HR);
    std::vector<uint8_t> dst_tex(64 * 32 * 4, 0u);
    CHECK(computer->CopyFromSubresource(
              dst_tex.data(), 64 * 4, 64 * 32 * 4, resource_mem.data(), 0, 6) == S_OK_HR);
    CHECK(dst_tex == src_tex);
    computer->Release();

    // BC7 4x4 block-compressed texture (64x64, full mip chain).
    Pg6XgTextureDesc bc7_desc{};
    bc7_desc.width = 64;
    bc7_desc.height = 64;
    bc7_desc.depth_or_array_size = 1;
    bc7_desc.mip_levels = 0;  // Full chain: 64, 32, 16, 8, 4, 2, 1 -> 7 mips.
    bc7_desc.format = kDxgiFormatBc7Unorm;
    bc7_desc.dimension = kXgDimensionTexture2D;
    bc7_desc.tile_mode = kXgTileModeLinear;

    Pg6IXGTextureAddressComputer* bc7_comp = nullptr;
    CHECK(pg6_xbox_XGCreateTextureComputer(&bc7_desc, &bc7_comp) == S_OK_HR);
    Pg6XgResourceLayout bc7_layout{};
    CHECK(bc7_comp->GetResourceLayout(&bc7_layout) == S_OK_HR);
    CHECK(bc7_layout.mip_levels == 7);
    // Mip 0: 64/4 = 16 blocks wide * 16 bytes = 256 row pitch, 16 block rows -> 4096 bytes.
    CHECK(bc7_layout.mips[0].row_pitch_bytes == 256);
    CHECK(bc7_layout.mips[0].padded_rows == 16);
    CHECK(bc7_layout.mips[0].slice_size_bytes == 4096);
    // Mip 6 (1x1): 1 block * 16 bytes -> row pitch aligned to 256, 1 block row -> 256 bytes.
    CHECK(bc7_layout.mips[6].padded_rows == 1);
    CHECK(bc7_layout.mips[6].row_pitch_bytes == 256);
    bc7_comp->Release();
}

// ---------------------------------------------------------------------------
// 4. GameInput.dll — GameInputCreate reading live GameInputReader SAB state
// ---------------------------------------------------------------------------

void test_game_input_bridge() {
    alignas(16) playgta6::InputState sab{};
    set_game_input_state(&sab, SeqlockMode::MonotonicSeq);
    set_game_input_deadzone(DeadzoneConfig{0.0f, 1.0f, 0.0f, 1.0f});

    CHECK(pg6_xbox_GameInputCreate(nullptr) == E_POINTER_HR);
    Pg6IGameInput* gi = nullptr;
    CHECK(pg6_xbox_GameInputCreate(&gi) == S_OK_HR);
    CHECK(gi != nullptr);

    const uint64_t t0 = gi->GetCurrentTimestamp();
    CHECK(t0 > 0);

    // Publish live gamepad & mouse state into the SAB.
    playgta6::InputState written{};
    written.buttons = input_button::kGamepadA | input_button::kRightBumper | input_button::kMenu;
    written.mouse_dx = 12;
    written.mouse_dy = -5;
    written.left_stick_x = 0.6f;
    written.left_stick_y = -0.4f;
    written.right_stick_x = -0.25f;
    written.right_stick_y = 0.75f;
    written.left_trigger = 0.25f;
    written.right_trigger = 0.9f;
    GameInputReader::write_monotonic(sab, written, false);

    Pg6IGameInputReading* reading1 = nullptr;
    CHECK(gi->GetCurrentReading(kGameInputKindGamepad, nullptr, &reading1) == S_OK_HR);
    CHECK(reading1 != nullptr);
    CHECK(reading1->GetInputKind() == kGameInputKindGamepad);
    CHECK(reading1->GetSequenceNumber(kGameInputKindGamepad) == 1u);
    CHECK(reading1->GetTimestamp() >= t0);
    CHECK(reading1->GetRawButtons() == written.buttons);

    Pg6GameInputGamepadState gp{};
    CHECK(reading1->GetGamepadState(&gp));
    CHECK((gp.buttons & gamepad_button::kA) != 0);
    CHECK((gp.buttons & gamepad_button::kRightShoulder) != 0);
    CHECK((gp.buttons & gamepad_button::kMenu) != 0);
    CHECK((gp.buttons & gamepad_button::kB) == 0);
    CHECK(approx_eq(gp.left_thumbstick_x, 0.6f));
    CHECK(approx_eq(gp.left_thumbstick_y, -0.4f));
    CHECK(approx_eq(gp.right_thumbstick_x, -0.25f));
    CHECK(approx_eq(gp.right_thumbstick_y, 0.75f));
    CHECK(approx_eq(gp.left_trigger, 0.25f));
    CHECK(approx_eq(gp.right_trigger, 0.9f));

    Pg6GameInputMouseState ms{};
    CHECK(reading1->GetMouseState(&ms));
    CHECK(ms.position_x == 12);
    CHECK(ms.position_y == -5);

    // GetNextReading with no new sequence returns GAMEINPUT_E_READING_NOT_FOUND.
    Pg6IGameInputReading* none_next = nullptr;
    CHECK(gi->GetNextReading(reading1, kGameInputKindGamepad, nullptr, &none_next) ==
          GAMEINPUT_E_READING_NOT_FOUND);
    CHECK(none_next == nullptr);

    // Write a second frame using WASD (neutral analog stick) -> synthesized left stick.
    playgta6::InputState wasd{};
    wasd.buttons = input_button::kMoveForward | input_button::kJump;
    GameInputReader::write_monotonic(sab, wasd, false);

    Pg6IGameInputReading* reading2 = nullptr;
    CHECK(gi->GetNextReading(reading1, kGameInputKindGamepad, nullptr, &reading2) == S_OK_HR);
    CHECK(reading2 != nullptr);
    CHECK(reading2->GetSequenceNumber(kGameInputKindGamepad) == 2u);
    CHECK(reading2->GetGamepadState(&gp));
    CHECK((gp.buttons & gamepad_button::kA) != 0);
    CHECK(approx_eq(gp.left_thumbstick_x, 0.0f));
    CHECK(approx_eq(gp.left_thumbstick_y, 1.0f));

    reading2->Release();
    reading1->Release();
    gi->Release();
    set_game_input_state(nullptr);
    set_game_input_deadzone(DeadzoneConfig{});
}

// ---------------------------------------------------------------------------
// 5. Module Registry Helper — pg6_xbox_scarlett_register_all & GetProcAddress
// ---------------------------------------------------------------------------

void test_module_registry_resolution() {
    const size_t count = pg6_xbox_scarlett_register_all();
    CHECK(count == 17u);
    // Idempotent second call.
    CHECK(pg6_xbox_scarlett_register_all() == 17u);

    // xmem.dll (12 exports)
    const Pg6Handle h_xmem = win32::GetModuleHandleA("xmem.dll");
    CHECK(h_xmem != win32::kNullHandle);
    CHECK(win32::GetProcAddressByName(h_xmem, "XMemAlloc") ==
          win32::proc_address(&pg6_xbox_XMemAlloc));
    CHECK(win32::GetProcAddressByName(h_xmem, "XMemAllocDefault") ==
          win32::proc_address(&pg6_xbox_XMemAllocDefault));
    CHECK(win32::GetProcAddressByName(h_xmem, "XMemAllocatePhysicalPages") ==
          win32::proc_address(&pg6_xbox_XMemAllocatePhysicalPages));
    CHECK(win32::GetProcAddressByName(h_xmem, "XMemFree") ==
          win32::proc_address(&pg6_xbox_XMemFree));
    CHECK(win32::GetProcAddressByName(h_xmem, "XMemFreeDefault") ==
          win32::proc_address(&pg6_xbox_XMemFreeDefault));
    CHECK(win32::GetProcAddressByName(h_xmem, "XMemFreePhysicalPages") ==
          win32::proc_address(&pg6_xbox_XMemFreePhysicalPages));
    CHECK(win32::GetProcAddressByName(h_xmem, "XMemGetWorkingSetStatistics") ==
          win32::proc_address(&pg6_xbox_XMemGetWorkingSetStatistics));
    CHECK(win32::GetProcAddressByName(h_xmem, "XMemIsToolsMemAvailable") ==
          win32::proc_address(&pg6_xbox_XMemIsToolsMemAvailable));
    CHECK(win32::GetProcAddressByName(h_xmem, "XMemMapPhysicalPages") ==
          win32::proc_address(&pg6_xbox_XMemMapPhysicalPages));
    CHECK(win32::GetProcAddressByName(h_xmem, "XMemMapPhysicalPagesScatter") ==
          win32::proc_address(&pg6_xbox_XMemMapPhysicalPagesScatter));
    CHECK(win32::GetProcAddressByName(h_xmem, "XMemTransferMemory") ==
          win32::proc_address(&pg6_xbox_XMemTransferMemory));
    CHECK(win32::GetProcAddressByName(h_xmem, "XMemVirtualAlloc") ==
          win32::proc_address(&pg6_xbox_XMemVirtualAlloc));

    // XCRDAPI.dll (1 export)
    const Pg6Handle h_xcrd = win32::GetModuleHandleA("XCRDAPI.dll");
    CHECK(h_xcrd != win32::kNullHandle);
    CHECK(win32::GetProcAddressByName(h_xcrd, "DStorageGetFactory") ==
          win32::proc_address(&pg6_xbox_DStorageGetFactory));

    // xg_xs.dll (2 exports)
    const Pg6Handle h_xg = win32::GetModuleHandleA("xg_xs.dll");
    CHECK(h_xg != win32::kNullHandle);
    CHECK(win32::GetProcAddressByName(h_xg, "XGCreateTextureComputer") ==
          win32::proc_address(&pg6_xbox_XGCreateTextureComputer));
    CHECK(win32::GetProcAddressByName(h_xg, "XGSuggestSwizzleMode") ==
          win32::proc_address(&pg6_xbox_XGSuggestSwizzleMode));

    // GameInput.dll (1 export: both ordinal #1 and name "GameInputCreate")
    const Pg6Handle h_gi = win32::GetModuleHandleA("GameInput.dll");
    CHECK(h_gi != win32::kNullHandle);
    CHECK(win32::GetProcAddressByName(h_gi, "GameInputCreate") ==
          win32::proc_address(&pg6_xbox_GameInputCreate));
    CHECK(win32::GetProcAddress(h_gi, reinterpret_cast<const char*>(uintptr_t{1})) ==
          win32::proc_address(&pg6_xbox_GameInputCreate));
    CHECK(win32::GetProcAddressByOrdinal(h_gi, 1u) ==
          win32::proc_address(&pg6_xbox_GameInputCreate));
    CHECK(win32::resolve_import_ordinal("GameInput.dll", 1u) ==
          win32::proc_address(&pg6_xbox_GameInputCreate));
}

}  // namespace

int main() {
    test_xmem_heap_and_hooks();
    test_xmem_physical_virtual_and_transfer();
    test_xmem_concurrent_map_unmap();
    test_xmem_fast_map_skips_fresh_zero_fill();
    test_dstorage_virtual_and_httpfs();
    test_surface_blocks();
    test_xg_xs_texture_computer();
    test_game_input_bridge();
    test_module_registry_resolution();

    if (g_failures > 0) {
        std::fprintf(stderr, "test_xbox_scarlett FAILED with %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_xbox_scarlett PASSED (all checks)\n");
    return 0;
}
