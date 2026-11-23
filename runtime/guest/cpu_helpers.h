#pragma once

#include <cstdint>
#include <string>

#include "abi/playgta6_abi.h"

// Virtual CPU behind the helpers lifted code calls for `cpuid`, `rdtsc`, `rdtscp` and `xgetbv`
// (`__playgta6_cpuid`, ... declared in abi/playgta6_abi.h, defined in cpu_helpers.cpp).
//
// One table, `VirtualCpu`, defines everything the guest can learn about the CPU. The default
// instance describes a Zen 2 class AMD CPU ("AuthenticAMD", family 17h) and never depends on the
// host: the same results on every machine and in the browser (the host's cpuid is never run).
//
// Feature policy: the advertised features must equal what the lifter implements, never more. A
// bit that is set while the lifter traps on the instruction turns "feature absent, take the
// fallback path" into a crash; a clear bit for a feature the code was compiled to assume is
// harmless. Every ISA level is an explicit switch in `CpuFeatures`; the defaults below follow
// recomp/src/lift on main:
//
//   set:   SSE SSE2 SSE3 (float/move/convert forms lifted), CMPXCHG16B, MOVBE, POPCNT, LZCNT
//          (ABM), BMI1, BMI2, ADX, CLFLUSH/CLFLUSHOPT/CLWB, CLZERO, PREFETCHW, LAHF/SAHF,
//          RDTSCP (+ invariant TSC), TSC, CMPXCHG8B, CMOV, NX, long mode.
//   clear: x87/FPU, MMX, FXSR (x87/MMX/fxsave trap in the lifter); SSSE3, SSE4.1, SSE4.2,
//          AES, PCLMULQDQ, SHA (integer SIMD / string compare / crypto are not lifted: only
//          the float subset of SSE4.1 is, so the SSE4.1 bit stays clear until pmulld, ptest,
//          pmovsx etc. are); XSAVE/OSXSAVE; AVX, AVX2, FMA, F16C (256-bit forms and FMA
//          trap); RDRAND, RDSEED, FSGSBASE, SYSCALL.
//
// When R8's AVX work lands (256-bit forms, FMA, F16C, integer SIMD), flip the matching switches
// here (`ssse3`, `sse41`, `sse42`, then `avx`, `avx2`, `fma`, `f16c`) and nothing else:
// `avx` implies `xsave`, which makes cpuid report XSAVE/OSXSAVE and leaf 0xD, and makes xgetbv
// report XCR0 bits 1-2 (SSE, AVX state), all derived from the same switch.
namespace playgta6::guest {

struct CpuidResult {
    uint32_t eax = 0;
    uint32_t ebx = 0;
    uint32_t ecx = 0;
    uint32_t edx = 0;
};

struct CpuFeatures {
    // Baseline.
    bool x87 = false;   // FPU (leaf 1 EDX[0])
    bool mmx = false;
    bool fxsr = false;  // fxsave/fxrstor
    bool sse = true;
    bool sse2 = true;
    bool sse3 = true;
    bool ssse3 = false;
    bool sse41 = false;
    bool sse42 = false;  // includes the crc32 instruction (lifted) and pcmpXstrX (not lifted)
    // Integer extensions.
    bool popcnt = true;
    bool lzcnt = true;  // ABM (leaf 0x80000001 ECX[5])
    bool movbe = true;
    bool cx8 = true;
    bool cx16 = true;
    bool cmov = true;
    bool bmi1 = true;
    bool bmi2 = true;
    bool adx = true;
    bool lahf = true;  // LAHF/SAHF in long mode
    bool prefetchw = true;
    bool clflush = true;
    bool clflushopt = true;
    bool clwb = true;
    bool clzero = true;
    // Crypto / random: not lifted.
    bool aes = false;
    bool pclmul = false;
    bool sha = false;
    bool rdrand = false;
    bool rdseed = false;
    // Time stamp.
    bool tsc = true;
    bool rdtscp = true;
    bool invariant_tsc = true;
    // Extended state. `avx` implies `xsave`; `avx2`, `fma`, `f16c` are dropped without `avx`.
    bool xsave = false;
    bool avx = false;
    bool avx2 = false;
    bool fma = false;
    bool f16c = false;
};

class VirtualCpu {
public:
    // Identity. family/model are the displayed values (family 0x17 = Zen 2, model 0x31).
    std::string vendor = "AuthenticAMD";  // exactly 12 characters
    std::string brand = "Virtual x86-64 Zen 2 class CPU";  // at most 47 characters
    uint32_t family = 0x17;
    uint32_t model = 0x31;
    uint32_t stepping = 0;
    // Logical processor count: the single source for cpuid and for GetSystemInfo-style shims.
    uint32_t logical_processors = 8;
    // IA32_TSC_AUX reported by rdtscp (processor id; one fixed id for every guest thread).
    uint32_t tsc_aux = 0;
    CpuFeatures features;

    // The feature set cpuid actually reports: dependencies applied (SSE2 needs SSE, SSE3 needs
    // SSE2, SSSE3 needs SSE3, SSE4.1 needs SSSE3, SSE4.2 needs SSE4.1; AVX turns on XSAVE; AVX2,
    // FMA, F16C need AVX; the crypto bits need SSE2). xgetbv and leaf 0xD derive from it too.
    CpuFeatures effective_features() const;

    // cpuid(leaf, subleaf). Unknown leaves return zeros (AMD behaviour, unlike Intel's "data of
    // the highest basic leaf"). Implemented leaves: 0, 1, 7 (sub-leaf 0; sub-leaf 1 and up are
    // zero), 0xD (sub-leaves 0, 1, 2), 0x80000000..0x80000008. Leaves 2-6, 8-0xC, 0xE+, the
    // cache/topology leaves (4, 0x8000001D, 0x8000001E, 0xB) and 0x80000005/6 return zeros;
    // TOPOEXT is not advertised.
    CpuidResult cpuid(uint32_t leaf, uint32_t subleaf) const;

    // XCR0 (x87 | SSE always; AVX state only when AVX is advertised).
    uint64_t xcr0() const;
    // xgetbv: false (and *value untouched) for an index without an XCR (everything but 0).
    bool xgetbv(uint32_t index, uint64_t* value) const;

    // Checks that cpuid and xgetbv agree with each other and with the guest state layout
    // (OSXSAVE vs XSAVE vs XCR0, leaf 0xD sizes, SSE/AVX dependency chains, hypervisor bit clear,
    // HTT vs the processor count, vendor/brand encodings). On failure returns false and, when
    // `why` is not null, describes the first problem.
    bool check_consistency(std::string* why = nullptr) const;
};

// The CPU instance the helpers consult. Not synchronised: change it only while no guest thread
// is running (start-up, tests).
VirtualCpu& virtual_cpu();
// Restores the default table.
void reset_virtual_cpu();

// Nominal TSC frequency implied by cpuid (Zen 2 console class clock, 3.6 GHz). rdtsc returns the
// runtime's QueryPerformanceCounter clock (nanoseconds since process start, +1) times 18/5, so a
// calibration loop comparing TSC to QPC measures exactly this frequency.
inline constexpr uint64_t kTscFrequencyHz = 3'600'000'000ULL;

// What rdtsc/rdtscp return: strictly increasing across all threads (each call returns more than
// every earlier call), lock-free.
uint64_t read_guest_tsc();

// Bytes of the XSAVE area described by leaf 0xD for the given XCR0 (legacy area + header, plus
// the AVX component when bit 2 is set). Derived from GuestState: the AVX component is the
// `ymm_hi` array.
uint32_t xsave_area_size(uint64_t xcr0);

}  // namespace playgta6::guest
