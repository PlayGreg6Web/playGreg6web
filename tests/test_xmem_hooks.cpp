// XMem allocation hooks (os/xbox_scarlett.h): XMemAlloc / XMemFree run a title's guest routines,
// taken from an exported __XMemAllocationHooks__ structure or from XMemSetAllocationHooks;
// XMemAllocDefault / XMemFreeDefault never do. The synthetic hook below has the shape of a
// title's hook: a table of handlers indexed by the allocator id (attribute bits 16..23); the
// handler for one id carves 64 KiB-rounded blocks out of an arena it allocates once through the
// next (default) handler; every other request falls through to the default allocator.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "abi/playgta6_abi.h"
#include "guest/com_bridge.h"
#include "guest/guest_runtime.h"
#include "os/guest_window.h"
#include "os/vmem.h"
#include "os/win32_kernel.h"
#include "os/xbox_scarlett.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_xmem_hooks.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6;
using namespace playgta6::guest;
namespace scarlett = playgta6::os::scarlett;
namespace win32 = playgta6::os::win32;

constexpr uint64_t kAllocHook = 0x140030000ULL;     // void* alloc(size_t, uint64_t attributes)
constexpr uint64_t kFreeHook = 0x140030100ULL;      // void free(void*, uint64_t attributes)
constexpr uint64_t kOtherAlloc = 0x140030200ULL;    // XMemSetAllocationHooks routine
constexpr uint64_t kWindowAlloc = 0x140030300ULL;   // hands out translated-window addresses
constexpr uint64_t kWindowFree = 0x140030400ULL;    // records the address it is given
constexpr uint64_t kWindowBlock = playgta6::os::guest_window::kWindowBase + 0x10040;
constexpr uint64_t kArenaId = 0x80;                 // allocator id served by the arena
constexpr uint64_t kArenaAttributes = (kArenaId << 16) | (2u << 14);
constexpr uint64_t kBudget = 8u << 20;

std::atomic<int> g_alloc_calls{0};
std::atomic<int> g_free_calls{0};
std::atomic<int> g_other_calls{0};
std::atomic<uint64_t> g_arena_base{0};
std::atomic<uint64_t> g_cursor{0};
std::atomic<uint64_t> g_window_freed{0};

uint64_t guest_of(void* host) {
    return host == nullptr ? 0 : win32::get_vmem_arena()->guest_address(host);
}

void ret(GuestState* s) {
    s->gpr[GuestReg::Rsp] += 8;
}

// The default handler at the end of every chain.
uint64_t default_handler(uint64_t size, uint64_t attributes) {
    return guest_of(pg6_xbox_XMemAllocDefault(static_cast<size_t>(size), attributes));
}

// The arena handler: other attributes fall through; the first matching request allocates the
// whole arena through the next handler; requests are rounded up to 64 KiB.
uint64_t arena_handler(uint64_t size, uint64_t attributes) {
    if ((attributes & 0xC000) != 0x8000) {
        return default_handler(size, attributes);
    }
    if (g_arena_base == 0) {
        g_arena_base = default_handler(kBudget, attributes);
        g_cursor = g_arena_base.load();
    }
    const uint64_t block = g_cursor;
    const uint64_t next = block + ((size + 0xFFFF) & ~uint64_t{0xFFFF});
    if (next > g_arena_base + kBudget) {
        return 0;
    }
    g_cursor = next;
    return block;
}

void sub_alloc_hook(GuestState* s) {
    g_alloc_calls.fetch_add(1);
    const uint64_t size = s->gpr[GuestReg::Rcx];
    const uint64_t attributes = s->gpr[GuestReg::Rdx];
    // Id-indexed table: only kArenaId has the arena handler in its chain.
    const uint64_t id = (attributes >> 16) & 0xFF;
    s->gpr[GuestReg::Rax] = id == kArenaId ? arena_handler(size, attributes) : default_handler(size, attributes);
    s->gpr[GuestReg::Rcx] = 0xDEAD;  // clobbers a volatile register: the caller's state is restored
    ret(s);
}

void sub_free_hook(GuestState* s) {
    g_free_calls.fetch_add(1);
    const uint64_t va = s->gpr[GuestReg::Rcx];
    const uint64_t attributes = s->gpr[GuestReg::Rdx];
    if (!(va >= g_arena_base && va < g_arena_base + kBudget)) {
        pg6_xbox_XMemFreeDefault(win32::get_vmem_arena()->translate(va), attributes);
    }
    ret(s);
}

void sub_other_alloc(GuestState* s) {
    g_other_calls.fetch_add(1);
    s->gpr[GuestReg::Rax] = default_handler(s->gpr[GuestReg::Rcx], s->gpr[GuestReg::Rdx]);
    ret(s);
}

void sub_window_alloc(GuestState* s) {
    s->gpr[GuestReg::Rax] = kWindowBlock;
    ret(s);
}

void sub_window_free(GuestState* s) {
    g_window_freed = s->gpr[GuestReg::Rcx];
    ret(s);
}

const LiftedFunctionEntry kFunctions[] = {
    {kAllocHook, sub_alloc_hook},
    {kFreeHook, sub_free_hook},
    {kOtherAlloc, sub_other_alloc},
    {kWindowAlloc, sub_window_alloc},
    {kWindowFree, sub_window_free},
};

// O31: a title allocator that hands out translated-window addresses. The import thunks of a
// windowed program translate the result back to the window address, and hand XMemFree the
// window address in their own form (the slot's backing page); the free routine gets the window
// address again.
void test_window_addresses() {
    namespace gw = playgta6::os::guest_window;
    gw::reset_for_testing();
    gw::set_enabled(true);
    std::vector<uint8_t> page(2 * 0x10000);
    auto* aligned = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(page.data()) + 0xFFFF) & ~uintptr_t{0xFFFF});
    CHECK(gw::map_slot(gw::kWindowBase + 0x10000, aligned));
    pg6_xbox_XMemSetAllocationHooks(kWindowAlloc, kWindowFree);
    void* p = pg6_xbox_XMemAlloc(64, 0);
    CHECK(gw::guest_address(p) == kWindowBlock);  // what the thunk returns to the guest
    pg6_xbox_XMemFree(gw::host_or_trap(kWindowBlock), 0);  // what the thunk passes
    CHECK(g_window_freed == kWindowBlock);
    pg6_xbox_XMemSetAllocationHooks(0, 0);
    gw::reset_for_testing();
}

void test_hooks() {
    auto& arena = *win32::get_vmem_arena();
    // The exported structure {alloc, free, 0} in guest memory.
    const uint64_t table = arena.virtual_alloc(0, 0x1000, os::kMemReserve | os::kMemCommit, os::PageProtect::ReadWrite);
    CHECK(table != 0);
    const uint64_t entries[3] = {kAllocHook, kFreeHook, 0};
    CHECK(arena.write(table, entries, sizeof(entries)));

    // No hooks: the default allocator.
    void* plain = pg6_xbox_XMemAlloc(100, 0);
    CHECK(plain != nullptr && g_alloc_calls == 0);
    pg6_xbox_XMemFree(plain, 0);

    scarlett::xmem_set_guest_hook_table(table);
    CHECK(scarlett::xmem_guest_hook_table() == table);

    // The arena: one large-page D3D-id allocation allocates the arena, blocks are 64 KiB apart.
    void* a = pg6_xbox_XMemAlloc(1000, kArenaAttributes);
    void* b = pg6_xbox_XMemAlloc(70000, kArenaAttributes);
    void* c = pg6_xbox_XMemAlloc(10, kArenaAttributes);
    CHECK(a != nullptr && guest_of(a) == g_arena_base);
    CHECK(guest_of(b) == g_arena_base + 0x10000);
    CHECK(guest_of(c) == g_arena_base + 0x30000);
    CHECK(g_cursor - g_arena_base == 0x40000);
    CHECK(g_alloc_calls == 3);

    // Other ids / page sizes fall through to the default allocator (still through the hook).
    void* other = pg6_xbox_XMemAlloc(256, kArenaId << 16);
    CHECK(other != nullptr && g_alloc_calls == 4);
    CHECK(!(guest_of(other) >= g_arena_base && guest_of(other) < g_arena_base + kBudget));

    // The default entry points bypass the hooks.
    void* direct = pg6_xbox_XMemAllocDefault(64, kArenaAttributes);
    CHECK(direct != nullptr && g_alloc_calls == 4);
    pg6_xbox_XMemFreeDefault(direct, kArenaAttributes);
    CHECK(g_free_calls == 0);

    // XMemFree goes through the free routine with the guest address.
    pg6_xbox_XMemFree(other, kArenaId << 16);
    pg6_xbox_XMemFree(a, kArenaAttributes);
    CHECK(g_free_calls == 2);

    // Concurrent callers (each host thread gets its own guest state).
    std::vector<std::thread> threads;
    std::atomic<int> ok{0};
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&] {
            for (int i = 0; i < 16; ++i) {
                void* p = pg6_xbox_XMemAlloc(32, 0);
                if (p != nullptr) {
                    ok.fetch_add(1);
                    pg6_xbox_XMemFree(p, 0);
                }
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    CHECK(ok == 64);
    CHECK(g_alloc_calls == 4 + 64 && g_free_calls == 2 + 64);

    // XMemSetAllocationHooks takes precedence; null routines fall back to the exported table.
    pg6_xbox_XMemSetAllocationHooks(kOtherAlloc, kFreeHook);
    void* via_other = pg6_xbox_XMemAlloc(48, 0);
    CHECK(via_other != nullptr && g_other_calls == 1 && g_alloc_calls == 68);
    pg6_xbox_XMemFree(via_other, 0);
    pg6_xbox_XMemSetAllocationHooks(0, 0);
    void* again = pg6_xbox_XMemAlloc(48, 0);
    CHECK(again != nullptr && g_alloc_calls == 69);
    pg6_xbox_XMemFree(again, 0);

    // Removing the table: back to the default allocator.
    scarlett::xmem_set_guest_hook_table(0);
    void* last = pg6_xbox_XMemAlloc(48, 0);
    CHECK(last != nullptr && g_alloc_calls == 69);
    pg6_xbox_XMemFree(last, 0);
}

}  // namespace

int main() {
    CHECK(register_lifted_functions(kFunctions));
    test_hooks();
    test_window_addresses();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_xmem_hooks failed with %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_xmem_hooks: all checks passed\n");
    return 0;
}
