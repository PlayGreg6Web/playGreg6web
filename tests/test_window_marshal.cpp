// O29: (pointer, length) marshaling of the generated thunks for code lifted with the translated
// CPU window. tests/fixtures/export_thunks_window/ is the real output of
// `playgta6-recomp export-thunks --memory-model window --only ...` (a Rust test keeps it in sync
// with the generator, recomp/src/stubs/exports/tests/fixture.rs). The test maps window slots onto
// arena pages so that slot 0 | slot 1 are NOT host-adjacent and slot 1 | slot 2 are, then calls the
// thunks the way guest code does with buffers that span those slots: a file read into a window
// buffer (CreateFileW with a split path, ReadFile), memcpy-class CRT calls (memcpy, memset,
// XMemTransferMemory), string functions (strlen, strncpy, strcpy, strcat), and a COM method's raw pointer argument
// (the COM bridge's Marshal). Split buffers go through bounce copies, contiguous ones are used in
// place, and pointer results map back to the guest addresses they stand for. Synthetic data only.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <span>
#include <string>
#include <vector>

#include "abi/playgta6_abi.h"
#include "guest/com_bridge.h"
#include "guest/export_thunks.h"
#include "guest/guest_runtime.h"
#include "os/guest_window.h"
#include "os/vmem.h"
#include "os/win32_kernel.h"
#include "os/win32_loader.h"
#include "pg6_export_thunks.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_window_marshal.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6;
using namespace playgta6::guest;
namespace com = playgta6::guest::com;
namespace gw = playgta6::os::guest_window;
namespace win32 = playgta6::os::win32;
using playgta6::os::VirtualMemoryArena;

constexpr uint64_t k64K = 64ull << 10;
constexpr uint64_t W = gw::kWindowBase;
constexpr uint64_t kReturnAddress = 0x140001000ULL;
constexpr uint32_t kRegs[4] = {GuestReg::Rcx, GuestReg::Rdx, GuestReg::R8, GuestReg::R9};

Pg6ExportThunkFn thunk(const char* name) {
    const std::span<const Pg6ExportThunk> table(pg6_export_thunks, pg6_export_thunk_count);
    for (const Pg6ExportThunk& e : table) {
        if (std::strcmp(e.name, name) == 0) {
            return e.thunk;
        }
    }
    std::fprintf(stderr, "no thunk for %s\n", name);
    ++g_failures;
    return nullptr;
}

struct Fixture {
    VirtualMemoryArena arena{playgta6::os::kDefaultArenaBase, uint64_t{1} << 30};
    GuestThread thread{};
    uint64_t pages_va = 0;
    uint8_t* pages = nullptr;

    Fixture() {
        reset_guest_runtime();
        com::reset_com_bridge();
        win32::set_vmem_arena(&arena);
        com::set_guest_memory(&arena);
        CHECK(create_guest_thread(arena, 256 * 1024, thread));
        // Slot 0 -> page 0, slot 1 -> page 2, slot 2 -> page 3: 0|1 split, 1|2 contiguous.
        pages_va = arena.virtual_alloc(0, 4 * k64K, playgta6::os::kMemReserve | playgta6::os::kMemCommit,
                                       playgta6::os::PageProtect::ReadWrite);
        pages = pages_va != 0 ? arena.translate(pages_va) : nullptr;
        CHECK(pages != nullptr);
        gw::reset_for_testing();
        gw::set_enabled(true);
        CHECK(gw::map_slot(W, page(0)));
        CHECK(gw::map_slot(W + k64K, page(2)));
        CHECK(gw::map_slot(W + 2 * k64K, page(3)));
    }

    ~Fixture() {
        gw::reset_for_testing();
        com::reset_com_bridge();
        destroy_guest_thread(arena, thread);
        com::set_guest_memory(nullptr);
        win32::set_vmem_arena(nullptr);
        reset_guest_runtime();
    }

    uint8_t* page(int i) const { return pages + i * k64K; }

    // Guest-convention call of a generated thunk: rcx rdx r8 r9, then stack arguments after the
    // shadow space, return address pushed; the thunk pops it. Returns rax.
    uint64_t call(Pg6ExportThunkFn fn, std::initializer_list<uint64_t> arg_list) {
        const std::vector<uint64_t> args(arg_list);
        GuestState& s = thread.state;
        const uint64_t saved_rsp = s.gpr[GuestReg::Rsp];
        const size_t stack_n = args.size() > 4 ? args.size() - 4 : 0;
        const uint64_t frame = ((0x20 + 8 * stack_n + 15) & ~uint64_t{15}) + 8;
        s.gpr[GuestReg::Rsp] -= frame;
        const uint64_t rsp = s.gpr[GuestReg::Rsp];
        for (size_t i = 0; i < args.size(); ++i) {
            if (i < 4) {
                s.gpr[kRegs[i]] = args[i];
            } else {
                CHECK(arena.write(rsp + 0x20 + 8 * (i - 4), &args[i], 8));
            }
        }
        s.gpr[GuestReg::Rsp] -= 8;
        CHECK(arena.write(s.gpr[GuestReg::Rsp], &kReturnAddress, 8));
        if (fn != nullptr) {
            fn(&s);
        }
        CHECK(s.gpr[GuestReg::Rsp] == rsp);
        s.gpr[GuestReg::Rsp] = saved_rsp;
        return s.gpr[GuestReg::Rax];
    }

    // A flat guest buffer in the arena.
    uint64_t flat(size_t bytes) {
        const uint64_t va = com::guest_alloc(bytes);
        CHECK(va != 0);
        return va;
    }
};

std::vector<uint8_t> pattern(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = static_cast<uint8_t>(i * 7 + seed);
    }
    return v;
}

std::vector<uint8_t> window_bytes(uint64_t va, size_t n) {
    std::vector<uint8_t> v(n);
    CHECK(gw::copy_from_guest(v.data(), va, n));
    return v;
}

void test_file_read_into_window_buffers() {
    Fixture fx;
    win32::clear_userdata_overlay();
    // The file, written by host code.
    const std::vector<uint8_t> data = pattern(5000, 3);
    {
        const Pg6Handle h = pg6_win32_create_file_w(u"T:\\userdata\\o29\\read.bin", win32::GENERIC_WRITE, 0, nullptr,
                                                    win32::CREATE_ALWAYS, win32::FILE_ATTRIBUTE_NORMAL, 0);
        CHECK(h != win32::INVALID_HANDLE_VALUE);
        uint32_t written = 0;
        CHECK(pg6_win32_write_file(h, data.data(), static_cast<uint32_t>(data.size()), &written, nullptr) == 1);
        pg6_win32_close_handle(h);
    }
    // The guest opens it with a path that straddles the split slots.
    const std::u16string path = u"T:\\userdata\\o29\\read.bin";
    const uint64_t path_va = W + k64K - 10;  // 5 characters in slot 0, the rest in slot 1
    CHECK(gw::copy_to_guest(path_va, path.c_str(), (path.size() + 1) * 2));
    const uint64_t handle = fx.call(thunk("CreateFileW"),
                                    {path_va, win32::GENERIC_READ, win32::FILE_SHARE_READ, 0, win32::OPEN_EXISTING,
                                     win32::FILE_ATTRIBUTE_NORMAL, 0});
    CHECK(handle != static_cast<uint64_t>(win32::INVALID_HANDLE_VALUE));

    // ReadFile into a buffer split across slots 0 | 1 (a bounce copy, copied back).
    const uint64_t got_va = fx.flat(8);
    const uint64_t split_buf = W + k64K - 1000;
    CHECK(fx.call(thunk("ReadFile"), {handle, split_buf, 3000, got_va, 0}) == 1);
    uint32_t got = 0;
    CHECK(com::read_guest(got_va, &got, 4) && got == 3000);
    CHECK(window_bytes(split_buf, 3000) == std::vector<uint8_t>(data.begin(), data.begin() + 3000));
    CHECK(std::memcmp(fx.page(0) + k64K - 1000, data.data(), 1000) == 0);  // slot 0's part
    CHECK(std::memcmp(fx.page(2), data.data() + 1000, 2000) == 0);        // slot 1's part
    // ... and into one across the contiguous slots 1 | 2 (in place), bytes_read in the window.
    const uint64_t run_buf = W + 2 * k64K - 600;
    const uint64_t got_window = W + 2 * k64K + 0x8000;
    CHECK(fx.call(thunk("ReadFile"), {handle, run_buf, 2000, got_window, 0}) == 1);
    CHECK(window_bytes(got_window, 4) == std::vector<uint8_t>({0xD0, 0x07, 0, 0}));  // 2000
    CHECK(std::memcmp(fx.page(2) + k64K - 600, data.data() + 3000, 600) == 0);
    CHECK(std::memcmp(fx.page(3), data.data() + 3600, 1400) == 0);
    CHECK(fx.call(thunk("CloseHandle"), {handle}) == 1);
}

void test_memcpy_class() {
    Fixture fx;
    const std::vector<uint8_t> src = pattern(256, 41);
    const uint64_t flat_src = fx.flat(src.size());
    CHECK(com::write_guest(flat_src, src.data(), src.size()));
    // Flat -> split window destination; the result is the guest destination address.
    const uint64_t dst = W + k64K - 100;
    CHECK(fx.call(thunk("memcpy"), {dst, flat_src, src.size()}) == dst);
    CHECK(window_bytes(dst, src.size()) == src);
    // Split window -> split window (both bounced).
    const uint64_t dst2 = W + k64K - 50;
    CHECK(fx.call(thunk("memcpy"), {dst2, dst, 40}) == dst2);
    CHECK(window_bytes(dst2, 40) == std::vector<uint8_t>(src.begin(), src.begin() + 40));
    // XMemTransferMemory: split source, contiguous destination (slots 1 | 2).
    const uint64_t dst3 = W + 2 * k64K - 64;
    const std::vector<uint8_t> before = window_bytes(dst, src.size());
    CHECK(fx.call(thunk("XMemTransferMemory"), {dst3, dst, src.size(), 0}) == 0);  // S_OK
    CHECK(window_bytes(dst3, src.size()) == before);
}

void test_string_functions() {
    Fixture fx;
    const char text[] = "a string across two window pages";
    const uint64_t s = W + k64K - 9;
    CHECK(gw::copy_to_guest(s, text, sizeof(text)));
    CHECK(fx.call(thunk("strlen"), {s}) == sizeof(text) - 1);
    // strncpy from the split string into the contiguous slots 1 | 2; returns the destination.
    const std::vector<uint8_t> fill(64, 0xEE);
    const uint64_t d = W + 2 * k64K - 16;
    CHECK(gw::copy_to_guest(d, fill.data(), fill.size()));
    CHECK(fx.call(thunk("strncpy"), {d, s, 40}) == d);
    const std::vector<uint8_t> out = window_bytes(d, 40);
    CHECK(std::memcmp(out.data(), text, sizeof(text)) == 0);
    CHECK(out[39] == 0 && window_bytes(d + 40, 1)[0] == 0xEE);  // NUL padding, nothing beyond
    // ... and back into a split destination.
    const uint64_t d2 = W + k64K - 5;
    CHECK(fx.call(thunk("strncpy"), {d2, d, sizeof(text)}) == d2);
    CHECK(std::memcmp(window_bytes(d2, sizeof(text)).data(), text, sizeof(text)) == 0);
}

// O31: memset (the length follows the fill value) and string destinations without a length
// (strcpy, strcat) on buffers split across non-adjacent slots.
void test_memset_and_string_destinations() {
    Fixture fx;
    const uint64_t dst = W + k64K - 40;  // 40 bytes in slot 0, the rest in slot 1
    CHECK(fx.call(thunk("memset"), {dst, 0x5A, 100}) == dst);
    CHECK(window_bytes(dst, 100) == std::vector<uint8_t>(100, 0x5A));
    CHECK(fx.page(2)[59] == 0x5A && fx.page(2)[60] != 0x5A);  // exactly 100 bytes, nothing beyond

    const char src[] = "a source string longer than what is left of the slot";
    const uint64_t s = W + 2 * k64K + 0x100;  // inside slot 2
    CHECK(gw::copy_to_guest(s, src, sizeof(src)));
    const uint64_t d = W + k64K - 10;
    const char start[] = "head:";
    CHECK(gw::copy_to_guest(d, start, sizeof(start)));
    CHECK(fx.call(thunk("strcat"), {d, s}) == d);
    const std::string expect = std::string(start) + src;
    CHECK(std::memcmp(window_bytes(d, expect.size() + 1).data(), expect.c_str(), expect.size() + 1) == 0);
    CHECK(fx.call(thunk("strcpy"), {d, s}) == d);
    CHECK(std::memcmp(window_bytes(d, sizeof(src)).data(), src, sizeof(src)) == 0);
}

// The COM bridge's raw pointer parameters (what a COM method with a buffer argument receives).
void test_com_raw_pointer_argument() {
    Fixture fx;
    struct Block {
        uint32_t words[32];
    };
    const uint64_t va = W + k64K - 64;  // 64 bytes in slot 0, 64 in slot 1
    std::vector<uint8_t> init = pattern(sizeof(Block), 9);
    CHECK(gw::copy_to_guest(va, init.data(), init.size()));
    GuestState state{};
    state.gpr[GuestReg::Rdx] = va;
    {
        com::CallContext ctx{state, 0, nullptr, nullptr, 0};
        Block* block = com::detail::Marshal<Block*>::load(ctx, 1);
        CHECK(block != nullptr);
        if (block != nullptr) {
            CHECK(std::memcmp(block, init.data(), sizeof(Block)) == 0);
            block->words[0] = 0x11111111u;
            block->words[31] = 0x22222222u;  // in slot 1
        }
        ctx.buffers.finish();
    }
    uint32_t first = 0;
    uint32_t last = 0;
    CHECK(gw::copy_from_guest(&first, va, 4) && first == 0x11111111u);
    CHECK(gw::copy_from_guest(&last, va + 31 * 4, 4) && last == 0x22222222u);
    // A const pointee is read, never written back.
    {
        com::CallContext ctx{state, 0, nullptr, nullptr, 0};
        const Block* block = com::detail::Marshal<const Block*>::load(ctx, 1);
        CHECK(block != nullptr && block->words[0] == 0x11111111u);
    }
}

}  // namespace

int main() {
    std::printf("test_window_marshal\n");
    test_file_read_into_window_buffers();
    test_memcpy_class();
    test_string_functions();
    test_memset_and_string_destinations();
    test_com_raw_pointer_argument();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_window_marshal: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_window_marshal: all tests passed\n");
    return 0;
}
