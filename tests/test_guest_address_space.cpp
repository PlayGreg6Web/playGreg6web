// End-to-end proof that the runtime's guest address space and lifted code agree
// (docs/wasm_abi.md "Guest Address Space"): a synthetic PGIM image is loaded by the image loader
// into a VirtualMemoryArena, the guest runtime creates a guest stack and calls a guest function
// that accesses memory exactly like lifted code (host = __playgta6_guest_base + VA): it reads a
// value from the image's .data via its guest VA, pushes it on the guest stack and writes results
// back into .data. The runtime then checks everything through the arena.
//
// The function exists as a hand-written C++ equivalent (always built) and, when the test is
// built with -DPLAYGTA6_E2E_LIFTED and linked with the object the Rust lifter produces for the
// same x86 code (recomp/src/lift/tests/exec.rs `writes_the_runtime_end_to_end_fixture`), as the
// real lifted `sub_140001000`.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "abi/playgta6_abi.h"
#include "guest/guest_runtime.h"
#include "os/image_loader.h"
#include "os/vmem.h"
#include "os/win32_kernel.h"

#ifdef PLAYGTA6_E2E_LIFTED
// Lifted from: mov rax,[0x140003000]; push rax; mov rdx,0x140003000; mov [rdx+16],rsp;
// mov rcx,[rsp]; add rcx,1; mov [rdx+8],rcx; pop rax; ret
extern "C" void sub_140001000(playgta6::GuestState* state);
#endif

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, const char* what, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_guest_address_space.cpp:%d] %s: %s\n", line, what, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, what, __LINE__)

using namespace playgta6;
using namespace playgta6::os;
using namespace playgta6::os::pgim;

constexpr uint64_t kImageBase = 0x140000000ULL;
constexpr uint64_t kEntry = kImageBase + 0x1000;
constexpr uint64_t kData = kImageBase + 0x3000;
constexpr uint64_t kValue = 0x0123456789ABCDEFULL;

// ---- what lifted code does: every guest access is host = __playgta6_guest_base + VA ----

uint64_t guest_load64(uint64_t va) {
    uint64_t v = 0;
    std::memcpy(&v, reinterpret_cast<const void*>(static_cast<uintptr_t>(__playgta6_guest_base + va)), 8);
    return v;
}

void guest_store64(uint64_t va, uint64_t v) {
    std::memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(__playgta6_guest_base + va)), &v, 8);
}

// Same semantics as the lifted sub_140001000 (registers rax, rcx, rdx, rsp).
void sub_handwritten(GuestState* s) {
    uint64_t& rsp = s->gpr[GuestReg::Rsp];
    s->gpr[GuestReg::Rax] = guest_load64(kData);  // mov rax, [0x140003000]
    rsp -= 8;                                     // push rax
    guest_store64(rsp, s->gpr[GuestReg::Rax]);
    s->gpr[GuestReg::Rdx] = kData;                // mov rdx, 0x140003000
    guest_store64(kData + 16, rsp);               // mov [rdx+16], rsp
    s->gpr[GuestReg::Rcx] = guest_load64(rsp) + 1;  // mov rcx, [rsp]; add rcx, 1
    guest_store64(kData + 8, s->gpr[GuestReg::Rcx]);  // mov [rdx+8], rcx
    s->gpr[GuestReg::Rax] = guest_load64(rsp);    // pop rax
    rsp += 8;
    rsp += 8;  // ret (pops the return address call_guest pushed)
}

// ---- a minimal PGIM v1 container (format: recomp/src/image.rs) ----

void put32(std::vector<uint8_t>& out, uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

void put64(std::vector<uint8_t>& out, uint64_t v) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

struct Section {
    const char* name;
    uint32_t rva;
    uint32_t vsize;
    uint32_t prot;
    std::vector<uint8_t> data;
};

std::vector<uint8_t> make_image() {
    std::vector<uint8_t> text(0x40, 0xCC);
    std::vector<uint8_t> data(8);
    std::memcpy(data.data(), &kValue, 8);
    const std::vector<Section> sections = {
        {".text", 0x1000, 0x1000, kProtRead | kProtExecute, text},
        {".data", 0x3000, 0x1000, kProtRead | kProtWrite, data},
    };
    std::vector<uint8_t> out = {'P', 'G', 'I', 'M'};
    put32(out, kVersion);
    put32(out, static_cast<uint32_t>(kHeaderBytes));
    put32(out, static_cast<uint32_t>(kSectionEntryBytes));
    put32(out, 0);  // flags
    put32(out, static_cast<uint32_t>(kPayloadAlignment));
    put64(out, kImageBase);
    put32(out, 0x5000);  // SizeOfImage
    put32(out, 0x1000);  // section alignment
    put32(out, 0x1000);  // entry point RVA
    put32(out, static_cast<uint32_t>(sections.size()));
    put32(out, 0);  // DIR64 count
    put32(out, 0);  // HIGHLOW count
    const uint64_t first_payload = 0x1000;
    put64(out, first_payload + 0x1000 * sections.size());  // file size (last payload is short)
    for (size_t i = 0; i < sections.size(); ++i) {
        const Section& s = sections[i];
        std::string name = s.name;
        name.resize(kSectionNameBytes, '\0');
        out.insert(out.end(), name.begin(), name.end());
        put32(out, s.rva);
        put32(out, s.vsize);
        put32(out, s.prot);
        put32(out, 0);  // section flags
        put64(out, first_payload + 0x1000 * i);
        put64(out, s.data.size());
        put64(out, fnv1a_64(s.data.data(), s.data.size()));
        put32(out, 0);  // characteristics
        put32(out, 0);
    }
    for (size_t i = 0; i < sections.size(); ++i) {
        out.resize(static_cast<size_t>(first_payload + 0x1000 * i), 0);
        out.insert(out.end(), sections[i].data.begin(), sections[i].data.end());
    }
    // Header field "file size" must match the container exactly.
    const uint64_t size = out.size();
    for (int i = 0; i < 8; ++i) out[56 + i] = static_cast<uint8_t>(size >> (8 * i));
    return out;
}

uint64_t arena_u64(VirtualMemoryArena& arena, uint64_t va) {
    uint64_t v = 0;
    return arena.read(va, &v, 8) ? v : ~uint64_t{0};
}

// Load the image into `arena`, publish it as the guest address space, run `fn` from a guest
// thread and check the results through the arena.
void run_scenario(const char* what, VirtualMemoryArena& arena, bool use_process_arena, LiftedFn fn) {
    win32::set_vmem_arena(use_process_arena ? nullptr : &arena);
    CHECK(__playgta6_guest_base == arena.guest_base());

    guest::reset_guest_runtime();
    static Pg6LiftedFunctionEntry table[1];
    table[0] = {kEntry, fn};
    CHECK(guest::register_lifted_functions(table));

    const std::vector<uint8_t> bytes = make_image();
    LoadedImage image;
    const ImageError err = load_image(bytes, arena, LoadOptions{}, image);
    CHECK(err.ok());
    if (!err.ok()) {
        std::fprintf(stderr, "  load_image: %s\n", error_name(err.code));
        win32::set_vmem_arena(nullptr);
        return;
    }
    CHECK(image.base == kImageBase && !image.rebased);  // preferred base honoured
    CHECK(arena.query(kEntry).protect == PageProtect::ExecuteRead);
    CHECK(arena.host_ptr_const(kData) ==
          reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(__playgta6_guest_base + kData)));

    guest::GuestThread thread;
    CHECK(guest::create_guest_thread(arena, 64 * 1024, thread));
    const uint64_t rax = guest::call_guest(arena, thread.state, kEntry);

    CHECK(rax == kValue);                                   // read .data through its guest VA
    CHECK(arena_u64(arena, kData + 8) == kValue + 1);       // write into .data
    const uint64_t pushed_at = arena_u64(arena, kData + 16);  // guest rsp after the push
    CHECK(pushed_at >= thread.stack_limit && pushed_at < thread.stack_top);
    CHECK(arena_u64(arena, pushed_at) == kValue);           // the guest stack slot
    CHECK(thread.state.gpr[GuestReg::Rsp] == thread.stack_top);
    std::printf("  %-28s guest_base=0x%016llx (%s) stack slot 0x%llx\n", what,
                static_cast<unsigned long long>(arena.guest_base()),
                arena.is_identity() ? "identity" : "offset",
                static_cast<unsigned long long>(pushed_at));

    guest::destroy_guest_thread(arena, thread);
    CHECK(unload_image(arena, image));
    guest::reset_guest_runtime();
    win32::set_vmem_arena(nullptr);
}

}  // namespace

int main() {
    std::printf("test_guest_address_space\n");
    VirtualMemoryArena& process = win32::default_vmem_arena();
    // An additional arena never takes the identity placement: always the offset translation.
    VirtualMemoryArena offset_arena(kDefaultArenaBase, uint64_t{2} << 30);
    {
        const char* what = "setup";
        CHECK(process.valid() && offset_arena.valid());
        CHECK(!offset_arena.is_identity());
#ifdef __EMSCRIPTEN__
        CHECK(process.is_identity());
#endif
    }

    run_scenario("hand-written, process arena", process, true, sub_handwritten);
    run_scenario("hand-written, offset arena", offset_arena, false, sub_handwritten);
#ifdef PLAYGTA6_E2E_LIFTED
    run_scenario("lifted, process arena", process, true, sub_140001000);
#ifndef PLAYGTA6_E2E_LIFTED_IDENTITY_ONLY
    run_scenario("lifted, offset arena", offset_arena, false, sub_140001000);
#endif
#endif

    if (g_failures != 0) {
        std::fprintf(stderr, "test_guest_address_space: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_guest_address_space: all tests passed\n");
    return 0;
}
