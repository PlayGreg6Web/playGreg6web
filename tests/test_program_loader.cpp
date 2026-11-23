// Program loader (runtime/guest/program_loader.h): synthetic PGIM images with hand-written
// guest-convention "lifted" functions, linked the way `playgta6-recomp build-program` links them
// (IAT slots of guest-to-guest imports pre-filled in the images, the rest bound to host thunks),
// booted through the boot sequence (guest/boot.h).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "abi/playgta6_abi.h"
#include "guest/boot.h"
#include "guest/guest_exceptions.h"
#include "guest/guest_runtime.h"
#include "guest/program_loader.h"
#include "os/image_loader.h"
#include "os/vmem.h"
#include "os/win32_kernel.h"
#include "os/win32_loader.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_program_loader.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6;
using namespace playgta6::guest;
using playgta6::os::VirtualMemoryArena;
namespace pgim = playgta6::os::pgim;
namespace win32 = playgta6::os::win32;

void check_code(const ProgramError& e, ProgramErrorCode code, int line) {
    if (e.code != code) {
        std::fprintf(stderr, "FAIL [test_program_loader.cpp:%d]: got '%s' (%s), expected '%s'\n",
                     line, program_error_name(e.code), e.detail.c_str(), program_error_name(code));
        ++g_failures;
    }
}

#define CHECK_CODE(e, code) check_code((e), (code), __LINE__)

// ---------------------------------------------------------------------------
// Synthetic images: <headers> (PE headers, export/TLS/.pdata directories), .data, .text
// ---------------------------------------------------------------------------

constexpr uint32_t kSize = 0x4000;
constexpr uint32_t kData = 0x1000;
constexpr uint32_t kText = 0x2000;
constexpr uint32_t kOptional = 0x80 + 24;

void put(std::vector<uint8_t>& b, size_t at, const void* src, size_t n) {
    if (b.size() < at + n) b.resize(at + n, 0);
    std::memcpy(b.data() + at, src, n);
}
void put16(std::vector<uint8_t>& b, size_t at, uint16_t v) { put(b, at, &v, 2); }
void put32(std::vector<uint8_t>& b, size_t at, uint32_t v) { put(b, at, &v, 4); }
void put64(std::vector<uint8_t>& b, size_t at, uint64_t v) { put(b, at, &v, 8); }
void put_str(std::vector<uint8_t>& b, size_t at, const std::string& s) {
    put(b, at, s.c_str(), s.size() + 1);
}

struct ExportSpec {
    std::string name;  // empty: ordinal only
    uint32_t ordinal;
    uint32_t rva;         // 0 for a forwarder
    std::string forward;  // "DLL.Name" / "DLL.#n"
};

struct TlsSpec {
    uint32_t start_rva;
    uint32_t end_rva;
    uint32_t index_rva;
    uint32_t callback_rva;
};

struct ModuleSpec {
    uint64_t base = 0;
    uint32_t entry_rva = 0;
    std::vector<ExportSpec> exports;
    const TlsSpec* tls = nullptr;
    bool pdata = false;
    std::vector<std::pair<uint32_t, uint64_t>> data;  // (rva, u64) in .data
};

void dir(std::vector<uint8_t>& h, size_t index, uint32_t rva, uint32_t size) {
    put32(h, kOptional + 112 + 8 * index, rva);
    put32(h, kOptional + 116 + 8 * index, size);
}

std::vector<uint8_t> headers(const ModuleSpec& spec) {
    std::vector<uint8_t> h(0x1000, 0);
    put16(h, 0, 0x5A4D);
    put32(h, 0x3C, 0x80);
    put32(h, 0x80, 0x00004550);
    put16(h, 0x84, 0x8664);
    put16(h, kOptional, 0x020B);
    put64(h, kOptional + 24, spec.base);
    put32(h, kOptional + 56, kSize);
    put32(h, kOptional + 108, 16);
    if (!spec.exports.empty()) {
        // Export directory at 0x200: EAT 0x240, names 0x280, ordinals 0x2C0, strings from 0x300.
        uint32_t lo = UINT32_MAX, hi = 0;
        for (const auto& e : spec.exports) {
            lo = std::min(lo, e.ordinal);
            hi = std::max(hi, e.ordinal);
        }
        size_t strings = 0x300;
        uint32_t names = 0;
        for (const auto& e : spec.exports) {
            const uint32_t slot = 0x240 + 4 * (e.ordinal - lo);
            if (e.forward.empty()) {
                put32(h, slot, e.rva);
            } else {
                put32(h, slot, static_cast<uint32_t>(strings));
                put_str(h, strings, e.forward);
                strings += e.forward.size() + 1;
            }
            if (!e.name.empty()) {
                put32(h, 0x280 + 4 * names, static_cast<uint32_t>(strings));
                put16(h, 0x2C0 + 2 * names, static_cast<uint16_t>(e.ordinal - lo));
                put_str(h, strings, e.name);
                strings += e.name.size() + 1;
                ++names;
            }
        }
        put32(h, 0x200 + 16, lo);
        put32(h, 0x200 + 20, hi - lo + 1);
        put32(h, 0x200 + 24, names);
        put32(h, 0x200 + 28, 0x240);
        put32(h, 0x200 + 32, 0x280);
        put32(h, 0x200 + 36, 0x2C0);
        dir(h, 0, 0x200, static_cast<uint32_t>(strings - 0x200));
    }
    if (spec.tls != nullptr) {
        put64(h, 0x600, spec.base + spec.tls->start_rva);
        put64(h, 0x608, spec.base + spec.tls->end_rva);
        put64(h, 0x610, spec.base + spec.tls->index_rva);
        put64(h, 0x618, spec.base + 0x640);  // callback array
        put64(h, 0x640, spec.base + spec.tls->callback_rva);
        dir(h, 9, 0x600, 40);
    }
    if (spec.pdata) {
        put32(h, 0x700, kText);
        put32(h, 0x704, kText + 0x10);
        put32(h, 0x708, 0x780);
        put32(h, 0x780, 0x00000001);  // UNWIND_INFO v1, no codes
        dir(h, 3, 0x700, 12);
    }
    return h;
}

void put_le(std::vector<uint8_t>& out, uint64_t v, int bytes) {
    for (int i = 0; i < bytes; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

std::vector<uint8_t> container(const ModuleSpec& spec) {
    struct Sec {
        const char* name;
        uint32_t rva, prot, flags;
        std::vector<uint8_t> bytes;
    };
    std::vector<uint8_t> data(0x200, 0);
    for (const auto& [rva, value] : spec.data) put64(data, rva - kData, value);
    std::vector<Sec> secs = {
        {"<headers>", 0, pgim::kProtRead, pgim::kSectionFlagHeaders, headers(spec)},
        {".data", kData, pgim::kProtRead | pgim::kProtWrite, 0, data},
        {".text", kText, pgim::kProtRead | pgim::kProtExecute, 0, {}},
    };
    const uint64_t tables = 64 + 64 * secs.size();
    std::vector<uint64_t> offsets;
    uint64_t cursor = tables;
    for (const Sec& s : secs) {
        if (s.bytes.empty()) {
            offsets.push_back(0);
            continue;
        }
        const uint64_t off = (cursor + 4095) / 4096 * 4096;
        offsets.push_back(off);
        cursor = off + s.bytes.size();
    }
    std::vector<uint8_t> out = {'P', 'G', 'I', 'M'};
    put_le(out, 1, 4);
    put_le(out, 64, 4);
    put_le(out, 64, 4);
    put_le(out, pgim::kFlagHeadersMapped, 4);
    put_le(out, 4096, 4);
    put_le(out, spec.base, 8);
    put_le(out, kSize, 4);
    put_le(out, 0x1000, 4);
    put_le(out, spec.entry_rva, 4);
    put_le(out, secs.size(), 4);
    put_le(out, 0, 4);
    put_le(out, 0, 4);
    put_le(out, cursor, 8);
    for (size_t i = 0; i < secs.size(); ++i) {
        std::string name = secs[i].name;
        name.resize(16, '\0');
        out.insert(out.end(), name.begin(), name.end());
        put_le(out, secs[i].rva, 4);
        put_le(out, 0x1000, 4);
        put_le(out, secs[i].prot, 4);
        put_le(out, secs[i].flags, 4);
        put_le(out, offsets[i], 8);
        put_le(out, secs[i].bytes.size(), 8);
        put_le(out, pgim::fnv1a_64(secs[i].bytes.data(), secs[i].bytes.size()), 8);
        put_le(out, 0, 4);
        put_le(out, 0, 4);
    }
    for (size_t i = 0; i < secs.size(); ++i) {
        if (!secs[i].bytes.empty()) {
            out.resize(static_cast<size_t>(offsets[i]), 0);
            out.insert(out.end(), secs[i].bytes.begin(), secs[i].bytes.end());
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Hand-written lifted functions
// ---------------------------------------------------------------------------

constexpr uint64_t kB = 0x3'0000'0000ULL;  // b.dll
constexpr uint64_t kA = 0x3'0001'0000ULL;  // a.dll (imports b.dll)
constexpr uint64_t kE = 0x3'0010'0000ULL;  // the executable (imports a.dll and b.dll)
constexpr uint64_t kC = 0x3'0020'0000ULL;  // c.dll (DllMain fails)
constexpr uint64_t kT = 0x3'0030'0000ULL;  // takeover.dll

VirtualMemoryArena* g_arena = nullptr;
std::string g_log;

uint64_t load64(uint64_t va) {
    uint64_t v = 0;
    CHECK(g_arena->read(va, &v, sizeof(v)));
    return v;
}

uint32_t load32(uint64_t va) {
    uint32_t v = 0;
    CHECK(g_arena->read(va, &v, sizeof(v)));
    return v;
}

uint64_t& reg(GuestState* s, uint32_t r) { return s->gpr[r]; }

void ret(GuestState* s) { reg(s, GuestReg::Rsp) += 8; }

// `call [slot]` as lifted code does it: push a return address, dispatch the slot's value.
void call_slot(GuestState* s, uint64_t slot) {
    reg(s, GuestReg::Rsp) -= 8;
    const uint64_t back = 0x1234;
    CHECK(g_arena->write(reg(s, GuestReg::Rsp), &back, 8));
    __playgta6_indirect_call(s, load64(slot));
}

void b_fb(GuestState* s) { reg(s, GuestReg::Rax) = 1; ret(s); }
void b_f3(GuestState* s) { reg(s, GuestReg::Rax) = 3; ret(s); }
void b_dllmain(GuestState* s) {
    CHECK(reg(s, GuestReg::Rcx) == kB);
    CHECK(reg(s, GuestReg::Rdx) == 1);  // DLL_PROCESS_ATTACH
    g_log += 'B';
    reg(s, GuestReg::Rax) = 1;
    ret(s);
}
void b_tls_callback(GuestState* s) {
    CHECK(reg(s, GuestReg::Rcx) == kB);
    g_log += 'b';
    ret(s);
}

// a!fa = b!fb() + 10 through a's pre-filled IAT slot.
void a_fa(GuestState* s) {
    call_slot(s, kA + kData);
    reg(s, GuestReg::Rax) += 10;
    ret(s);
}
void a_dllmain(GuestState* s) {
    g_log += 'A';
    reg(s, GuestReg::Rax) = 1;
    ret(s);
}

// main = a!fa() + *b!bdata + host() = 11 + 1000 + 77 (rcx = PEB from the boot).
void e_main(GuestState* s) {
    g_log += 'E';
    call_slot(s, kE + kData);
    reg(s, GuestReg::Rbx) = reg(s, GuestReg::Rax);
    reg(s, GuestReg::Rbx) += load64(load64(kE + kData + 8));
    call_slot(s, kE + kData + 0x10);
    reg(s, GuestReg::Rax) += reg(s, GuestReg::Rbx);
    ret(s);
}

void c_dllmain(GuestState* s) {
    g_log += 'C';
    reg(s, GuestReg::Rax) = 0;  // FALSE
    ret(s);
}

// Import bridge thunk (pops the return address like the generated ones).
void host_77(void* state) {
    auto* s = static_cast<GuestState*>(state);
    reg(s, GuestReg::Rax) = 77;
    ret(s);
}

const Pg6LiftedFunctionEntry kTableB[] = {{kB + kText, &b_fb},
                                          {kB + kText + 0x10, &b_f3},
                                          {kB + kText + 0x100, &b_dllmain},
                                          {kB + kText + 0x200, &b_tls_callback}};
const uint64_t kTableBCount = 4;
const Pg6LiftedFunctionEntry kTableA[] = {{kA + kText, &a_fa}, {kA + kText + 0x100, &a_dllmain}};
const uint64_t kTableACount = 2;
const Pg6LiftedFunctionEntry kTableE[] = {{kE + kText, &e_main}};
const uint64_t kTableECount = 1;
const Pg6LiftedFunctionEntry kTableC[] = {{kC + kText, &c_dllmain}};
const uint64_t kTableCCount = 1;

// Import bridge tables (pg6_import_thunks_<module> of the generated bridges): only the slots the
// build did not pre-fill.
const Pg6BootImport kImportsA[] = {
    {kData + 8, kPg6BootImportBound, 0, "KERNEL32.dll", "Host77", "pg6_win32_host77", &host_77}};
const Pg6BootImport* const kImportsAPtr = kImportsA;
const size_t kImportsACount = 1;
// The bridge has no shim for RtlCaptureContext (it works on the caller's registers): the program
// loader rebinds it to the unwinder's guest-state thunk, here through an API set.
const Pg6BootImport kImportsE[] = {
    {kData + 0x10, kPg6BootImportBound, 0, "KERNEL32.dll", "Host77", "pg6_win32_host77", &host_77},
    {kData + 0x18, 0, 0, "api-ms-win-core-rtlsupport-l1-1-0.dll", "RtlCaptureContext", nullptr,
     nullptr}};
const Pg6BootImport* const kImportsEPtr = kImportsE;
const size_t kImportsECount = 2;

const TlsSpec kTlsB{kData + 0x100, kData + 0x108, kData + 0x10, kText + 0x200};

ModuleSpec spec_b() {
    ModuleSpec m;
    m.base = kB;
    m.entry_rva = kText + 0x100;
    m.exports = {{"fb", 1, kText, ""}, {"bdata", 2, kData, ""}, {"", 3, kText + 0x10, ""}};
    m.tls = &kTlsB;
    m.pdata = true;
    m.data = {{kData, 1000}, {kData + 0x10, 0xFFFFFFFF}};
    return m;
}

ModuleSpec spec_a() {
    ModuleSpec m;
    m.base = kA;
    m.entry_rva = kText + 0x100;
    m.exports = {{"fa", 1, kText, ""}, {"fwd", 2, 0, "b.fb"}, {"fwdord", 3, 0, "B.#3"}};
    m.data = {{kData, kB + kText}};  // pre-filled: b.dll!fb
    return m;
}

ModuleSpec spec_e() {
    ModuleSpec m;
    m.base = kE;
    m.entry_rva = kText;
    m.exports = {{"MainExport", 1, kText, ""}};
    m.data = {{kData, kA + kText}, {kData + 8, kB + kData}};  // a.dll!#1, b.dll!bdata
    return m;
}

ModuleSpec plain_dll(uint64_t base, uint32_t entry_rva) {
    ModuleSpec m;
    m.base = base;
    m.entry_rva = entry_rva;
    m.exports = {{"GuestFn", 1, kText, ""}};
    return m;
}

Pg6ProgramModule exe_module() {
    return Pg6ProgramModule{"game.exe", "main", kE, kSize, kE + kText, kPg6ProgramModuleMain, 0,
                            kTableE, &kTableECount, &kImportsEPtr, &kImportsECount};
}

const std::vector<uint8_t>& exe_image() {
    static const std::vector<uint8_t> image = container(spec_e());
    return image;
}

BootResult boot(ProgramBoot& pb) {
    pb.config.arena = g_arena;
    pb.config.log_progress = false;
    g_log.clear();
    return run_boot(pb.config);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void test_boot_links_and_runs_a_program(VirtualMemoryArena& arena) {
    const auto b = container(spec_b());
    const auto a = container(spec_a());
    const Pg6ProgramModule modules[] = {
        {"b.dll", "b", kB, kSize, kB + kText + 0x100, kPg6ProgramModuleDll | kPg6ProgramModuleTls,
         0, kTableB, &kTableBCount, nullptr, nullptr},
        {"A.DLL", "a", kA, kSize, kA + kText + 0x100, kPg6ProgramModuleDll, 0, kTableA,
         &kTableACount, &kImportsAPtr, &kImportsACount},
        exe_module(),
    };
    const Pg6Program program{kPg6ProgramVersion, 3, modules};
    ProgramBoot pb;
    CHECK_CODE(prepare_program_boot(program,
                                    memory_image_source({{"b", b}, {"a", a}, {"main", exe_image()}}),
                                    ProgramLoadOptions{"D:\\", true}, pb),
               ProgramErrorCode::Ok);
    CHECK(pb.config.modules.size() == 3);
    CHECK(pb.config.modules[0].name == "game.exe" && pb.config.modules[1].name == "b.dll");
    CHECK(pb.config.call_dll_entry_points && pb.config.after_load);
    CHECK(pb.exception_imports == 1);
    const BootImport& capture = pb.config.modules[0].imports[1];
    CHECK(capture.thunk != nullptr && capture.thunk == exception_thunk_for("KERNEL32.dll", "RtlCaptureContext"));
    CHECK((capture.flags & kPg6BootImportBound) != 0);

    const BootResult result = boot(pb);
    CHECK(result.status == BootStatus::Exited);
    CHECK(result.stats.exit_code == 11 + 1000 + 77);
    // TLS callbacks and DllMain of each DLL in initialisation order, then the entry point.
    CHECK(g_log == "bBAE");
    if (result.status != BootStatus::Exited) {
        std::fprintf(stderr, "%s\n", result.summary.c_str());
    }

    // Module registry: guest modules by name, exports as guest VAs, forwarders followed.
    CHECK(win32::GetModuleHandleA("b.dll") == kB);
    CHECK(win32::GetModuleHandleA("a") == kA);
    CHECK(win32::GetModuleHandleA(nullptr) == kE);
    CHECK(win32::is_guest_module(kA) && win32::is_guest_module(kB) && win32::is_guest_module(kE));
    CHECK(win32::GetProcAddressByName(kA, "fa") == kA + kText);
    CHECK(win32::GetProcAddressByOrdinal(kA, 1) == kA + kText);
    CHECK(win32::GetProcAddressByName(kA, "fwd") == kB + kText);
    CHECK(win32::GetProcAddressByName(kA, "fwdord") == kB + kText + 0x10);
    CHECK(win32::GetProcAddressByOrdinal(kB, 3) == kB + kText + 0x10);
    CHECK(win32::GetProcAddressByName(kB, "bdata") == kB + kData);
    CHECK(win32::GetProcAddressByName(kE, "MainExport") == kE + kText);
    CHECK(win32::GetProcAddressByName(kB, "missing") == 0);
    CHECK(win32::resolve_import("a.dll", "fwd") == kB + kText);
    char path[64] = {};
    CHECK(win32::GetModuleFileNameA(kB, path, sizeof(path)) > 0);
    CHECK(std::string(path) == "D:\\b.dll");
    // Host shim exports are not mixed into a guest module.
    CHECK(!win32::register_export("b.dll", "HostFn", 0, 0x1000));

    // IAT slots: pre-filled ones untouched, the bridge's ones bound to synthetic addresses.
    CHECK(load64(kA + kData) == kB + kText);
    CHECK(is_synthetic_address(load64(kA + kData + 8)));
    CHECK(is_synthetic_address(load64(kE + kData + 0x10)));
    CHECK(is_synthetic_address(load64(kE + kData + 0x18)));
    // The boot gave b.dll the first TLS index (the executable has no TLS directory).
    CHECK(load32(kB + kData + 0x10) == 0);
    // b.dll's .pdata is registered with the unwinder.
    uint64_t image_base = 0;
    CHECK(lookup_function_entry(arena, kB + kText + 4, &image_base) == kB + 0x700);
    CHECK(image_base == kB);

    std::vector<ProgramModuleInfo> info;
    // Registering again is idempotent for the modules (exports are re-published).
    ProgramLoadOptions no_tables{"D:\\", false};
    CHECK_CODE(register_program_modules(arena, program, no_tables, &info), ProgramErrorCode::Ok);
    CHECK(info.size() == 3 && info[0].exports == 3 && info[1].forwarders == 2);
    CHECK(info[0].pdata_va == kB + 0x700 && info[0].pdata_size == 12);
    reset_boot();
}

void test_failures(VirtualMemoryArena& arena) {
    const auto c = container(plain_dll(kC, kText));
    const Pg6ProgramModule good[] = {
        {"c.dll", "c", kC, kSize, kC + kText, kPg6ProgramModuleDll, 0, kTableC, &kTableCCount,
         nullptr, nullptr},
        exe_module()};
    ProgramBoot pb;

    // Malformed descriptors.
    CHECK_CODE(prepare_program_boot({2, 2, good}, memory_image_source({}), {}, pb),
               ProgramErrorCode::InvalidProgram);
    const Pg6ProgramModule main_first[] = {exe_module(), good[0]};
    CHECK_CODE(prepare_program_boot({kPg6ProgramVersion, 2, main_first}, memory_image_source({}),
                                    {}, pb),
               ProgramErrorCode::InvalidProgram);
    Pg6ProgramModule overlap[] = {good[0], exe_module()};
    overlap[0].base = kE + 0x1000;
    CHECK_CODE(prepare_program_boot({kPg6ProgramVersion, 2, overlap}, memory_image_source({}), {},
                                    pb),
               ProgramErrorCode::InvalidProgram);
    // No image for a module.
    CHECK_CODE(prepare_program_boot({kPg6ProgramVersion, 2, good},
                                    memory_image_source({{"main", exe_image()}}), {}, pb),
               ProgramErrorCode::MissingImage);

    // DllMain returning FALSE stops the boot before the entry point.
    CHECK_CODE(prepare_program_boot({kPg6ProgramVersion, 2, good},
                                    memory_image_source({{"c", c}, {"main", exe_image()}}), {}, pb),
               ProgramErrorCode::Ok);
    BootResult result = boot(pb);
    CHECK(result.status == BootStatus::ProcessSetupFailed);
    CHECK(result.summary.find("DllMain of c.dll returned FALSE") != std::string::npos);
    CHECK(g_log == "C");
    reset_boot();

    // The descriptor's base disagrees with the image: the after_load step refuses the program.
    constexpr uint64_t kElsewhere = 0x3'0040'0000ULL;
    const auto moved = container(plain_dll(kElsewhere, 0));
    Pg6ProgramModule wrong[] = {good[0], exe_module()};
    wrong[0].file_name = "moved.dll";
    wrong[0].module = "moved";
    wrong[0].entry_point = 0;
    wrong[0].lifted_functions = nullptr;
    CHECK_CODE(prepare_program_boot({kPg6ProgramVersion, 2, wrong},
                                    memory_image_source({{"moved", moved}, {"main", exe_image()}}),
                                    {}, pb),
               ProgramErrorCode::Ok);
    result = boot(pb);
    CHECK(result.status == BootStatus::RegisterFailed);
    CHECK_CODE(pb.after_load_error, ProgramErrorCode::BadHeaders);
    CHECK(g_log.empty());
    reset_boot();
    (void)arena;
}

void test_guest_module_takes_over_a_virtual_name(VirtualMemoryArena& arena) {
    CHECK(win32::register_export("takeover.dll", "HostOnly", 0, 0x5000));
    const win32::Handle virtual_handle = win32::GetModuleHandleA("takeover.dll");
    CHECK(virtual_handle >= win32::kVirtualModuleBase);
    const auto t = container(plain_dll(kT, 0));
    const Pg6ProgramModule modules[] = {
        {"TakeOver.dll", "takeover", kT, kSize, 0, kPg6ProgramModuleDll, 0, nullptr, nullptr,
         nullptr, nullptr},
        exe_module()};
    ProgramBoot pb;
    CHECK_CODE(prepare_program_boot({kPg6ProgramVersion, 2, modules},
                                    memory_image_source({{"takeover", t}, {"main", exe_image()}}),
                                    {}, pb),
               ProgramErrorCode::Ok);
    pb.config.call_entry = false;  // the executable's imports are not part of this program
    const BootResult result = boot(pb);
    CHECK(result.status == BootStatus::Exited);
    CHECK(win32::GetModuleHandleA("takeover") == kT);
    CHECK(win32::GetProcAddressByName(kT, "GuestFn") == kT + kText);
    CHECK(win32::GetProcAddressByName(kT, "HostOnly") == 0);
    // The virtual module is still reachable through the handle handed out earlier.
    CHECK(win32::GetProcAddressByName(virtual_handle, "HostOnly") == 0x5000);
    CHECK(!win32::is_guest_module(virtual_handle));
    // A second guest module cannot take the name.
    CHECK(win32::register_guest_module("takeover.dll", "", kT + 0x100000, kSize) == 0);
    reset_boot();
    (void)arena;
}

}  // namespace

int main() {
    VirtualMemoryArena arena;
    g_arena = &arena;
    win32::set_vmem_arena(&arena);
    test_boot_links_and_runs_a_program(arena);
    test_failures(arena);
    test_guest_module_takes_over_a_virtual_name(arena);
    win32::set_vmem_arena(nullptr);
    if (g_failures != 0) {
        std::fprintf(stderr, "test_program_loader failed with %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_program_loader: all checks passed\n");
    return 0;
}
