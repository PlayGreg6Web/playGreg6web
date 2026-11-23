#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <string>
#include <thread>
#include <vector>

#include "abi/playgta6_abi.h"
#include "guest/cpu_helpers.h"
#include "guest/guest_runtime.h"
#include "os/win32_kernel.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_cpu_helpers.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6;
using namespace playgta6::guest;

constexpr uint64_t kHighGarbage = 0xDEADBEEF00000000ULL;

// A state whose registers all hold recognisable values with the upper halves set, so a helper
// that fails to zero-extend or clobbers an untouched register is caught.
GuestState make_state() {
    GuestState s{};
    for (unsigned i = 0; i < 16; ++i) {
        s.gpr[i] = kHighGarbage | (0x1000u + i);
    }
    s.rip = 0x1234;
    s.flags[GuestFlag::ZF] = 1;
    s.mxcsr = kGuestMxcsrDefault;
    s.xmm[3][0] = 0x1122334455667788ULL;
    s.ymm_hi[5][1] = 0x99AABBCCDDEEFF00ULL;
    return s;
}

// Calls the helper the way lifted code does: registers in the state, helper, registers back.
GuestState call_cpuid(uint32_t leaf, uint32_t subleaf) {
    GuestState s = make_state();
    s.gpr[GuestReg::Rax] = kHighGarbage | leaf;
    s.gpr[GuestReg::Rcx] = kHighGarbage | subleaf;
    __playgta6_cpuid(&s);
    return s;
}

CpuidResult cpuid_of(const GuestState& s) {
    return {static_cast<uint32_t>(s.gpr[GuestReg::Rax]), static_cast<uint32_t>(s.gpr[GuestReg::Rbx]),
            static_cast<uint32_t>(s.gpr[GuestReg::Rcx]), static_cast<uint32_t>(s.gpr[GuestReg::Rdx])};
}

CpuidResult cpuid(uint32_t leaf, uint32_t subleaf = 0) {
    return cpuid_of(call_cpuid(leaf, subleaf));
}

bool upper_halves_zero(const GuestState& s, std::initializer_list<uint32_t> regs) {
    for (uint32_t r : regs) {
        if ((s.gpr[r] >> 32) != 0) {
            return false;
        }
    }
    return true;
}

// Every register and state field except `touched` is as make_state() left it.
bool untouched_preserved(const GuestState& s, std::initializer_list<uint32_t> touched) {
    const GuestState ref = make_state();
    for (unsigned i = 0; i < 16; ++i) {
        if (std::find(touched.begin(), touched.end(), i) != touched.end()) {
            continue;
        }
        if (s.gpr[i] != ref.gpr[i]) {
            return false;
        }
    }
    GuestState a = s;
    GuestState b = ref;
    std::memset(a.gpr, 0, sizeof(a.gpr));
    std::memset(b.gpr, 0, sizeof(b.gpr));
    return std::memcmp(&a, &b, sizeof(a)) == 0;
}

std::string text_of(std::initializer_list<uint32_t> words) {
    std::string out;
    for (uint32_t w : words) {
        for (unsigned i = 0; i < 4; ++i) {
            out.push_back(static_cast<char>((w >> (8 * i)) & 0xFF));
        }
    }
    return out;
}

constexpr bool has(uint32_t value, unsigned n) {
    return ((value >> n) & 1u) != 0;
}

// ---------------------------------------------------------------------------

void test_leaf0_and_vendor() {
    reset_virtual_cpu();
    const GuestState s = call_cpuid(0, 0);
    const CpuidResult r = cpuid_of(s);
    CHECK(r.eax == 0xD);
    CHECK(text_of({r.ebx, r.edx, r.ecx}) == "AuthenticAMD");
    CHECK(r.ebx == 0x68747541 && r.edx == 0x69746E65 && r.ecx == 0x444D4163);
    CHECK(upper_halves_zero(s, {GuestReg::Rax, GuestReg::Rbx, GuestReg::Rcx, GuestReg::Rdx}));
    CHECK(untouched_preserved(
        s, {GuestReg::Rax, GuestReg::Rbx, GuestReg::Rcx, GuestReg::Rdx}));

    // Extended leaf 0 repeats the vendor on AMD and caps the extended range.
    const CpuidResult e = cpuid(0x80000000);
    CHECK(e.eax == 0x80000008);
    CHECK(text_of({e.ebx, e.edx, e.ecx}) == "AuthenticAMD");
}

void test_leaf1() {
    reset_virtual_cpu();
    const CpuidResult r = cpuid(1);
    CHECK(r.eax == 0x00830F10);  // family 17h model 31h stepping 0
    const uint32_t family = ((r.eax >> 8) & 0xF) + ((r.eax >> 20) & 0xFF);
    const uint32_t model = ((r.eax >> 4) & 0xF) | (((r.eax >> 16) & 0xF) << 4);
    CHECK(family == 0x17 && model == 0x31 && (r.eax & 0xF) == 0);
    CHECK(((r.ebx >> 8) & 0xFF) == 8);                                  // 64-byte CLFLUSH line
    CHECK(((r.ebx >> 16) & 0xFF) == virtual_cpu().logical_processors);  // logical processors
    CHECK((r.ebx >> 24) == 0 && (r.ebx & 0xFF) == 0);
    CHECK(has(r.edx, 28));  // HTT (more than one logical processor)
    CHECK(has(r.edx, 4) && has(r.edx, 8) && has(r.edx, 15) && has(r.edx, 19));  // TSC CX8 CMOV CLFSH
    CHECK(has(r.edx, 25) && has(r.edx, 26));                                    // SSE SSE2
    CHECK(!has(r.edx, 0) && !has(r.edx, 23) && !has(r.edx, 24));  // FPU MMX FXSR not lifted
    CHECK(has(r.ecx, 0) && has(r.ecx, 13) && has(r.ecx, 22) && has(r.ecx, 23));  // SSE3 CX16 MOVBE POPCNT
    CHECK(!has(r.ecx, 9) && !has(r.ecx, 19) && !has(r.ecx, 20));  // SSSE3 SSE4.1 SSE4.2
    CHECK(!has(r.ecx, 26) && !has(r.ecx, 27) && !has(r.ecx, 28));  // XSAVE OSXSAVE AVX
    CHECK(!has(r.ecx, 12) && !has(r.ecx, 29) && !has(r.ecx, 30));  // FMA F16C RDRAND
    CHECK(!has(r.ecx, 25) && !has(r.ecx, 1));                      // AES PCLMUL
    CHECK(!has(r.ecx, 31));                                        // hypervisor
}

void test_leaf7() {
    reset_virtual_cpu();
    const CpuidResult r = cpuid(7, 0);
    CHECK(r.eax == 0);
    CHECK(has(r.ebx, 3) && has(r.ebx, 8) && has(r.ebx, 19));    // BMI1 BMI2 ADX
    CHECK(has(r.ebx, 23) && has(r.ebx, 24));                    // CLFLUSHOPT CLWB
    CHECK(!has(r.ebx, 5) && !has(r.ebx, 18) && !has(r.ebx, 29));  // AVX2 RDSEED SHA
    CHECK(!has(r.ebx, 0) && !has(r.ebx, 7));                    // FSGSBASE SMEP
    CHECK(r.ecx == 0 && r.edx == 0);
    const CpuidResult s1 = cpuid(7, 1);
    CHECK(s1.eax == 0 && s1.ebx == 0 && s1.ecx == 0 && s1.edx == 0);
    const CpuidResult s2 = cpuid(7, 2);
    CHECK(s2.eax == 0 && s2.ebx == 0 && s2.ecx == 0 && s2.edx == 0);
}

void test_leaf_d() {
    reset_virtual_cpu();
    // Without XSAVE the whole leaf is empty.
    CpuidResult r = cpuid(0xD, 0);
    CHECK(r.eax == 0 && r.ebx == 0 && r.ecx == 0 && r.edx == 0);

    virtual_cpu().features.avx = true;  // implies XSAVE
    r = cpuid(0xD, 0);
    CHECK(r.eax == 7);        // x87 | SSE | AVX
    CHECK(r.ebx == 576 + 256);
    CHECK(r.ecx == 576 + 256);
    CHECK(r.edx == 0);
    r = cpuid(0xD, 1);
    CHECK(r.eax == 0 && r.ebx == 832 && r.ecx == 0 && r.edx == 0);
    r = cpuid(0xD, 2);
    CHECK(r.eax == 256 && r.ebx == 576 && r.ecx == 0 && r.edx == 0);
    r = cpuid(0xD, 3);
    CHECK(r.eax == 0 && r.ebx == 0 && r.ecx == 0 && r.edx == 0);
    CHECK(256 == sizeof(GuestState::ymm_hi));

    reset_virtual_cpu();
    virtual_cpu().features.xsave = true;  // XSAVE without AVX
    r = cpuid(0xD, 0);
    CHECK(r.eax == 3 && r.ebx == 576 && r.ecx == 576);
    r = cpuid(0xD, 2);
    CHECK(r.eax == 0 && r.ebx == 0);
    reset_virtual_cpu();
}

void test_extended_leaves() {
    reset_virtual_cpu();
    const CpuidResult e1 = cpuid(0x80000001);
    CHECK(e1.eax == cpuid(1).eax);
    CHECK(has(e1.ecx, 0));                 // LAHF/SAHF
    CHECK(has(e1.ecx, 5));                 // ABM (LZCNT)
    CHECK(has(e1.ecx, 8));                 // PREFETCHW
    CHECK(!has(e1.ecx, 6) && !has(e1.ecx, 22));  // SSE4A, TOPOEXT
    CHECK(has(e1.edx, 20) && has(e1.edx, 27) && has(e1.edx, 29));  // NX RDTSCP LM
    CHECK(!has(e1.edx, 11));  // SYSCALL instruction is not lifted

    // Brand string: three leaves, NUL terminated, matches the table.
    std::string brand;
    for (uint32_t leaf = 0x80000002; leaf <= 0x80000004; ++leaf) {
        const CpuidResult b = cpuid(leaf);
        brand += text_of({b.eax, b.ebx, b.ecx, b.edx});
    }
    CHECK(brand.size() == 48);
    CHECK(std::string(brand.c_str()) == virtual_cpu().brand);
    CHECK(brand.back() == '\0');

    CHECK(cpuid(0x80000005).eax == 0 && cpuid(0x80000006).ecx == 0);
    CHECK(has(cpuid(0x80000007).edx, 8));  // invariant TSC

    const CpuidResult e8 = cpuid(0x80000008);
    CHECK((e8.eax & 0xFF) == 48 && ((e8.eax >> 8) & 0xFF) == 48);
    CHECK(has(e8.ebx, 0));  // CLZERO
    CHECK((e8.ecx & 0xFF) + 1 == virtual_cpu().logical_processors);
    CHECK(((e8.ecx >> 12) & 0xF) == 3);  // 8 threads -> 3 APIC id bits
}

void test_unknown_leaves_are_zero() {
    reset_virtual_cpu();
    for (uint32_t leaf : {0x2u, 0x3u, 0x4u, 0x5u, 0x6u, 0xBu, 0xEu, 0x10u, 0x20u, 0x40000000u,
                          0x8000000Fu, 0x80000009u, 0x8000001Du, 0x8000001Eu, 0x80000020u,
                          0xFFFFFFFFu}) {
        const CpuidResult r = cpuid(leaf, 0);
        CHECK(r.eax == 0 && r.ebx == 0 && r.ecx == 0 && r.edx == 0);
    }
}

void test_zero_extension_and_preservation() {
    reset_virtual_cpu();
    for (uint32_t leaf : {0u, 1u, 7u, 0xDu, 0x80000001u, 0x80000002u, 0x80000008u, 0x1234u}) {
        const GuestState s = call_cpuid(leaf, 0);
        CHECK(upper_halves_zero(s, {GuestReg::Rax, GuestReg::Rbx, GuestReg::Rcx, GuestReg::Rdx}));
        CHECK(untouched_preserved(
            s, {GuestReg::Rax, GuestReg::Rbx, GuestReg::Rcx, GuestReg::Rdx}));
    }
    // rdtsc touches rax/rdx only, rdtscp also rcx.
    GuestState s = make_state();
    __playgta6_rdtsc(&s);
    CHECK(upper_halves_zero(s, {GuestReg::Rax, GuestReg::Rdx}));
    CHECK(untouched_preserved(s, {GuestReg::Rax, GuestReg::Rdx}));
    s = make_state();
    __playgta6_rdtscp(&s);
    CHECK(upper_halves_zero(s, {GuestReg::Rax, GuestReg::Rcx, GuestReg::Rdx}));
    CHECK(untouched_preserved(s, {GuestReg::Rax, GuestReg::Rcx, GuestReg::Rdx}));
    s = make_state();
    s.gpr[GuestReg::Rcx] = kHighGarbage | 0;  // only ECX counts
    __playgta6_xgetbv(&s);
    CHECK(upper_halves_zero(s, {GuestReg::Rax, GuestReg::Rdx}));
    CHECK(s.gpr[GuestReg::Rcx] == (kHighGarbage | 0));  // ECX is an input only
    s.gpr[GuestReg::Rcx] = make_state().gpr[GuestReg::Rcx];
    CHECK(untouched_preserved(s, {GuestReg::Rax, GuestReg::Rdx}));
}

void test_feature_override() {
    reset_virtual_cpu();
    CHECK(!has(cpuid(1).ecx, 20));
    virtual_cpu().features.ssse3 = true;
    virtual_cpu().features.sse41 = true;
    virtual_cpu().features.sse42 = true;
    CpuidResult r = cpuid(1);
    CHECK(has(r.ecx, 9) && has(r.ecx, 19) && has(r.ecx, 20));

    // Dependencies: SSE4.2 without SSE4.1 is not reported.
    virtual_cpu().features.sse41 = false;
    r = cpuid(1);
    CHECK(has(r.ecx, 9) && !has(r.ecx, 19) && !has(r.ecx, 20));

    virtual_cpu().features.avx = true;
    virtual_cpu().features.avx2 = true;
    virtual_cpu().features.fma = true;
    virtual_cpu().features.f16c = true;
    r = cpuid(1);
    CHECK(has(r.ecx, 26) && has(r.ecx, 27) && has(r.ecx, 28) && has(r.ecx, 12) && has(r.ecx, 29));
    CHECK(has(cpuid(7).ebx, 5));
    uint64_t xcr = 0;
    CHECK(virtual_cpu().xgetbv(0, &xcr) && xcr == 7);

    // Identity and processor count are table entries too.
    virtual_cpu().vendor = "GenuineIntel";
    virtual_cpu().logical_processors = 1;
    virtual_cpu().brand = "Test CPU";
    r = cpuid(0);
    CHECK(text_of({r.ebx, r.edx, r.ecx}) == "GenuineIntel");
    CHECK(!has(cpuid(1).edx, 28));
    CHECK((cpuid(0x80000008).ecx & 0xFFFF) == 0);
    CHECK(virtual_cpu().check_consistency());
    reset_virtual_cpu();
    CHECK(!has(cpuid(1).ecx, 28));
}

void test_consistency_all_switches() {
    using Member = bool CpuFeatures::*;
    // Exhaustive over every switch that takes part in a dependency chain or in XSAVE/AVX state.
    const Member chain[] = {&CpuFeatures::sse,  &CpuFeatures::sse2, &CpuFeatures::sse3,
                            &CpuFeatures::ssse3, &CpuFeatures::sse41, &CpuFeatures::sse42,
                            &CpuFeatures::xsave, &CpuFeatures::avx,  &CpuFeatures::avx2,
                            &CpuFeatures::fma,   &CpuFeatures::f16c, &CpuFeatures::aes,
                            &CpuFeatures::pclmul, &CpuFeatures::sha};
    constexpr unsigned kChain = sizeof(chain) / sizeof(chain[0]);
    for (uint32_t mask = 0; mask < (1u << kChain); ++mask) {
        VirtualCpu cpu;
        for (unsigned i = 0; i < kChain; ++i) {
            cpu.features.*chain[i] = ((mask >> i) & 1u) != 0;
        }
        std::string why;
        if (!cpu.check_consistency(&why)) {
            std::fprintf(stderr, "inconsistent for mask 0x%X: %s\n", mask, why.c_str());
            CHECK(false);
            return;
        }
    }

    // Pseudo-random combinations of every switch and of the processor count.
    const Member all[] = {
        &CpuFeatures::x87, &CpuFeatures::mmx, &CpuFeatures::fxsr, &CpuFeatures::sse,
        &CpuFeatures::sse2, &CpuFeatures::sse3, &CpuFeatures::ssse3, &CpuFeatures::sse41,
        &CpuFeatures::sse42, &CpuFeatures::popcnt, &CpuFeatures::lzcnt, &CpuFeatures::movbe,
        &CpuFeatures::cx8, &CpuFeatures::cx16, &CpuFeatures::cmov, &CpuFeatures::bmi1,
        &CpuFeatures::bmi2, &CpuFeatures::adx, &CpuFeatures::lahf, &CpuFeatures::prefetchw,
        &CpuFeatures::clflush, &CpuFeatures::clflushopt, &CpuFeatures::clwb, &CpuFeatures::clzero,
        &CpuFeatures::aes, &CpuFeatures::pclmul, &CpuFeatures::sha, &CpuFeatures::rdrand,
        &CpuFeatures::rdseed, &CpuFeatures::tsc, &CpuFeatures::rdtscp, &CpuFeatures::invariant_tsc,
        &CpuFeatures::xsave, &CpuFeatures::avx, &CpuFeatures::avx2, &CpuFeatures::fma,
        &CpuFeatures::f16c};
    uint64_t rng = 0x9E3779B97F4A7C15ULL;
    for (unsigned iter = 0; iter < 20000; ++iter) {
        VirtualCpu cpu;
        for (Member m : all) {
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            cpu.features.*m = (rng >> 33) & 1u;
        }
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        cpu.logical_processors = static_cast<uint32_t>((rng >> 33) % 300);  // includes 0 and > 255
        std::string why;
        if (!cpu.check_consistency(&why)) {
            std::fprintf(stderr, "inconsistent random combination %u: %s\n", iter, why.c_str());
            CHECK(false);
            return;
        }
    }

    // The check does reject a bad table: a brand string that cannot be encoded.
    VirtualCpu bad;
    bad.vendor = "short";
    std::string why;
    CHECK(!bad.check_consistency(&why) && !why.empty());
    reset_virtual_cpu();
    CHECK(virtual_cpu().check_consistency());
}

void test_xgetbv() {
    reset_virtual_cpu();
    GuestState s = make_state();
    s.gpr[GuestReg::Rcx] = 0;
    __playgta6_xgetbv(&s);
    CHECK(s.gpr[GuestReg::Rax] == 3 && s.gpr[GuestReg::Rdx] == 0);  // x87 | SSE

    virtual_cpu().features.avx = true;
    s = make_state();
    s.gpr[GuestReg::Rcx] = 0;
    __playgta6_xgetbv(&s);
    CHECK(s.gpr[GuestReg::Rax] == 7 && s.gpr[GuestReg::Rdx] == 0);
    reset_virtual_cpu();
}

struct SeenFault {
    FaultInfo info;
    std::string message;
};

void throwing_handler(const FaultInfo* info) {
    char message[512];
    format_fault(*info, message, sizeof(message));
    throw SeenFault{*info, message};
}

void test_xgetbv_invalid_index_faults() {
    reset_virtual_cpu();
    reset_guest_runtime();
    set_fault_handler(throwing_handler);
    for (uint64_t index : {1ULL, 2ULL, 0x80000000ULL, 0xFFFFFFFFULL}) {
        GuestState s = make_state();
        s.rip = 0xABCD0;
        s.gpr[GuestReg::Rcx] = kHighGarbage | index;
        bool faulted = false;
        bool returned = false;
        try {
            // Through a volatile pointer: MSVC assumes extern "C" functions never throw and would
            // drop the handler of a direct call.
            void (*volatile helper)(GuestState*) = __playgta6_xgetbv;
            helper(&s);
            returned = true;
        } catch (const SeenFault& fault) {
            faulted = true;
            CHECK(fault.info.kind == static_cast<uint32_t>(FaultKind::GeneralProtection));
            CHECK(fault.info.detail == index);
            CHECK(fault.info.va == 0xABCD0);
            CHECK(fault.info.state == &s);
            CHECK(fault.message.find("xgetbv") != std::string::npos);
            CHECK(fault.message.find("general protection") != std::string::npos);
        }
        CHECK(faulted && !returned);
        // The helper must not have written results before faulting.
        CHECK(s.gpr[GuestReg::Rax] == (kHighGarbage | 0x1000u));
        CHECK(s.gpr[GuestReg::Rdx] == (kHighGarbage | 0x1002u));
    }
    set_fault_handler(nullptr);
    reset_guest_runtime();
}

void test_rdtsc() {
    constexpr int kCalls = 100000;
    GuestState s = make_state();
    uint64_t prev = 0;
    bool increasing = true;
    for (int i = 0; i < kCalls; ++i) {
        __playgta6_rdtsc(&s);
        const uint64_t tsc = (s.gpr[GuestReg::Rdx] << 32) | s.gpr[GuestReg::Rax];
        increasing = increasing && (i == 0 || tsc > prev);
        prev = tsc;
    }
    CHECK(increasing);

    // Threads: each sees strictly increasing values, and no value is handed out twice.
    constexpr int kThreads = 4;
    constexpr int kPerThread = 25000;
    std::vector<std::vector<uint64_t>> seen(kThreads);
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&seen, t] {
            GuestState local = make_state();
            seen[t].reserve(kPerThread);
            for (int i = 0; i < kPerThread; ++i) {
                if (i % 2 == 0) {
                    __playgta6_rdtsc(&local);
                } else {
                    __playgta6_rdtscp(&local);
                }
                seen[t].push_back((local.gpr[GuestReg::Rdx] << 32) | local.gpr[GuestReg::Rax]);
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    std::vector<uint64_t> merged;
    for (const auto& v : seen) {
        CHECK(std::adjacent_find(v.begin(), v.end(), [](uint64_t a, uint64_t b) { return b <= a; }) ==
              v.end());
        merged.insert(merged.end(), v.begin(), v.end());
    }
    std::sort(merged.begin(), merged.end());
    CHECK(std::adjacent_find(merged.begin(), merged.end()) == merged.end());

    // The counter really runs: a short sleep advances it by roughly the nominal frequency.
    const uint64_t before = read_guest_tsc();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const uint64_t after = read_guest_tsc();
    const uint64_t ticks = after - before;
    CHECK(ticks >= kTscFrequencyHz / 20 * 9 / 10);  // >= ~45 ms worth
    CHECK(ticks < kTscFrequencyHz * 2);

    // Same clock as QueryPerformanceCounter: TSC / frequency tracks QPC nanoseconds.
    int64_t qpc0 = 0;
    int64_t qpc1 = 0;
    const uint64_t tsc0 = read_guest_tsc();
    os::win32::QueryPerformanceCounter(&qpc0);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    os::win32::QueryPerformanceCounter(&qpc1);
    const uint64_t tsc1 = read_guest_tsc();
    const double tsc_ns = static_cast<double>(tsc1 - tsc0) * 1e9 / static_cast<double>(kTscFrequencyHz);
    const double qpc_ns = static_cast<double>(qpc1 - qpc0);
    CHECK(tsc_ns >= qpc_ns * 0.99 && tsc_ns <= qpc_ns * 1.5 + 5e6);
}

void test_rdtscp_aux() {
    reset_virtual_cpu();
    GuestState s = make_state();
    __playgta6_rdtscp(&s);
    CHECK(s.gpr[GuestReg::Rcx] == 0);  // default: processor 0 for every thread
    const uint64_t tsc = (s.gpr[GuestReg::Rdx] << 32) | s.gpr[GuestReg::Rax];
    CHECK(tsc != 0);

    virtual_cpu().tsc_aux = 0xA5A5A5A5u;
    s = make_state();
    __playgta6_rdtscp(&s);
    CHECK(s.gpr[GuestReg::Rcx] == 0xA5A5A5A5u);
    reset_virtual_cpu();
}

void test_determinism() {
    auto sweep = [] {
        std::vector<CpuidResult> out;
        for (uint32_t base : {0u, 0x80000000u}) {
            for (uint32_t leaf = base; leaf < base + 0x30; ++leaf) {
                for (uint32_t sub = 0; sub < 4; ++sub) {
                    out.push_back(cpuid(leaf, sub));
                }
            }
        }
        return out;
    };
    reset_virtual_cpu();
    const auto first = sweep();
    reset_virtual_cpu();
    const auto second = sweep();
    CHECK(first.size() == second.size());
    CHECK(std::memcmp(first.data(), second.data(), first.size() * sizeof(CpuidResult)) == 0);

    // A fresh table gives the same answers as the shared default instance.
    VirtualCpu fresh;
    for (uint32_t leaf : {0u, 1u, 7u, 0xDu, 0x80000001u, 0x80000008u}) {
        const CpuidResult a = fresh.cpuid(leaf, 0);
        const CpuidResult b = virtual_cpu().cpuid(leaf, 0);
        CHECK(a.eax == b.eax && a.ebx == b.ebx && a.ecx == b.ecx && a.edx == b.edx);
    }
    CHECK(fresh.logical_processors == 8);  // fixed, not derived from the host
    CHECK(virtual_cpu().check_consistency());
}

}  // namespace

int main() {
    test_leaf0_and_vendor();
    test_leaf1();
    test_leaf7();
    test_leaf_d();
    test_extended_leaves();
    test_unknown_leaves_are_zero();
    test_zero_extension_and_preservation();
    test_feature_override();
    test_consistency_all_switches();
    test_xgetbv();
    test_xgetbv_invalid_index_faults();
    test_rdtsc();
    test_rdtscp_aux();
    test_determinism();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::puts("test_cpu_helpers: all checks passed");
    return 0;
}
