#include "guest/cpu_helpers.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>

#include "guest/guest_runtime.h"
#include "os/win32_kernel.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
// Keep the helpers in the link and export them (same as the helpers in guest_runtime.cpp).
#define PG6_GUEST_HELPER EMSCRIPTEN_KEEPALIVE
#else
#define PG6_GUEST_HELPER
#endif

namespace playgta6::guest {

namespace {

// XCR0 state components.
constexpr uint64_t kXcr0X87 = 1u << 0;
constexpr uint64_t kXcr0Sse = 1u << 1;
constexpr uint64_t kXcr0Avx = 1u << 2;

// XSAVE standard format: 512-byte legacy area + 64-byte header, then components in bit order.
constexpr uint32_t kXsaveLegacyBytes = 512;
constexpr uint32_t kXsaveHeaderBytes = 64;
constexpr uint32_t kXsaveAvxOffset = kXsaveLegacyBytes + kXsaveHeaderBytes;
constexpr uint32_t kXsaveAvxBytes = static_cast<uint32_t>(sizeof(GuestState::ymm_hi));
static_assert(kXsaveAvxBytes == 256, "the AVX XSAVE component is the 16 upper YMM halves");

constexpr uint32_t kMaxBasicLeaf = 0xD;
constexpr uint32_t kMaxExtendedLeaf = 0x80000008;

constexpr uint32_t bit(unsigned n) {
    return uint32_t{1} << n;
}
constexpr uint32_t flag(bool on, unsigned n) {
    return on ? bit(n) : 0u;
}

uint32_t pack4(const char* s) {
    return static_cast<uint32_t>(static_cast<unsigned char>(s[0])) |
           static_cast<uint32_t>(static_cast<unsigned char>(s[1])) << 8 |
           static_cast<uint32_t>(static_cast<unsigned char>(s[2])) << 16 |
           static_cast<uint32_t>(static_cast<unsigned char>(s[3])) << 24;
}

// Leaf 1 / 0x80000001 EAX: stepping, model, family with the extended fields.
uint32_t signature(const VirtualCpu& cpu) {
    const uint32_t family = cpu.family;
    const uint32_t base_family = family < 0xF ? family : 0xF;
    const uint32_t ext_family = family < 0xF ? 0 : (family - 0xF) & 0xFF;
    return (cpu.stepping & 0xF) | (cpu.model & 0xF) << 4 | base_family << 8 |
           ((cpu.model >> 4) & 0xF) << 16 | ext_family << 20;
}

uint32_t logical_count(const VirtualCpu& cpu) {
    return std::clamp<uint32_t>(cpu.logical_processors, 1, 255);
}

uint32_t apic_id_bits(uint32_t count) {
    uint32_t bits = 0;
    while ((uint32_t{1} << bits) < count) {
        ++bits;
    }
    return bits;
}

// Largest XSAVE area for the supported state components.
uint32_t xsave_max_size(const CpuFeatures& f) {
    return xsave_area_size(kXcr0X87 | kXcr0Sse | (f.avx ? kXcr0Avx : 0));
}

std::atomic<uint64_t> g_last_tsc{0};

VirtualCpu& default_cpu_instance() {
    static VirtualCpu instance;
    return instance;
}

}  // namespace

// ---------------------------------------------------------------------------
// VirtualCpu
// ---------------------------------------------------------------------------

CpuFeatures VirtualCpu::effective_features() const {
    CpuFeatures f = features;
    f.sse2 = f.sse2 && f.sse;
    f.sse3 = f.sse3 && f.sse2;
    f.ssse3 = f.ssse3 && f.sse3;
    f.sse41 = f.sse41 && f.ssse3;
    f.sse42 = f.sse42 && f.sse41;
    f.aes = f.aes && f.sse2;
    f.pclmul = f.pclmul && f.sse2;
    f.sha = f.sha && f.sse2;
    f.rdtscp = f.rdtscp && f.tsc;
    f.invariant_tsc = f.invariant_tsc && f.tsc;
    f.avx = f.avx && f.sse2;
    f.xsave = f.xsave || f.avx;
    f.avx2 = f.avx2 && f.avx;
    f.fma = f.fma && f.avx;
    f.f16c = f.f16c && f.avx;
    return f;
}

uint64_t VirtualCpu::xcr0() const {
    return kXcr0X87 | kXcr0Sse | (effective_features().avx ? kXcr0Avx : 0);
}

bool VirtualCpu::xgetbv(uint32_t index, uint64_t* value) const {
    if (index != 0) {
        return false;
    }
    *value = xcr0();
    return true;
}

CpuidResult VirtualCpu::cpuid(uint32_t leaf, uint32_t subleaf) const {
    const CpuFeatures f = effective_features();
    const uint32_t count = logical_count(*this);
    const bool htt = count > 1;
    CpuidResult r;

    switch (leaf) {
        case 0: {
            char vendor_chars[12] = {};
            std::memcpy(vendor_chars, vendor.data(), std::min<size_t>(vendor.size(), 12));
            r.eax = kMaxBasicLeaf;
            r.ebx = pack4(vendor_chars);
            r.edx = pack4(vendor_chars + 4);
            r.ecx = pack4(vendor_chars + 8);
            break;
        }
        case 1:
            r.eax = signature(*this);
            // [15:8] CLFLUSH line size in qwords, [23:16] logical processor count (valid with
            // HTT), [31:24] initial APIC id (0: one fixed id).
            r.ebx = (f.clflush ? 8u << 8 : 0u) | (htt ? count << 16 : 0u);
            r.ecx = flag(f.sse3, 0) | flag(f.pclmul, 1) | flag(f.ssse3, 9) | flag(f.fma, 12) |
                    flag(f.cx16, 13) | flag(f.sse41, 19) | flag(f.sse42, 20) | flag(f.movbe, 22) |
                    flag(f.popcnt, 23) | flag(f.aes, 25) | flag(f.xsave, 26) |
                    flag(f.xsave, 27) /* OSXSAVE */ | flag(f.avx, 28) | flag(f.f16c, 29) |
                    flag(f.rdrand, 30) /* bit 31 hypervisor: always clear */;
            r.edx = flag(f.x87, 0) | flag(f.tsc, 4) | flag(f.cx8, 8) | flag(f.cmov, 15) |
                    flag(f.clflush, 19) | flag(f.mmx, 23) | flag(f.fxsr, 24) | flag(f.sse, 25) |
                    flag(f.sse2, 26) | flag(htt, 28);
            break;
        case 7:
            if (subleaf == 0) {
                r.eax = 0;  // highest sub-leaf
                r.ebx = flag(f.bmi1, 3) | flag(f.avx2, 5) | flag(f.bmi2, 8) | flag(f.rdseed, 18) |
                        flag(f.adx, 19) | flag(f.clflushopt, 23) | flag(f.clwb, 24) |
                        flag(f.sha, 29);
            }
            break;
        case 0xD:
            if (!f.xsave) {
                break;
            }
            if (subleaf == 0) {
                const uint64_t xcr = xcr0();
                r.eax = static_cast<uint32_t>(xcr);
                r.ebx = xsave_area_size(xcr);
                r.ecx = xsave_max_size(f);
            } else if (subleaf == 1) {
                // No XSAVEOPT/XSAVEC/XSAVES: EAX = 0; EBX = size for XCR0 | IA32_XSS.
                r.ebx = xsave_area_size(xcr0());
            } else if (subleaf == 2 && f.avx) {
                r.eax = kXsaveAvxBytes;
                r.ebx = kXsaveAvxOffset;
            }
            break;
        case 0x80000000: {
            char vendor_chars[12] = {};
            std::memcpy(vendor_chars, vendor.data(), std::min<size_t>(vendor.size(), 12));
            r.eax = kMaxExtendedLeaf;
            r.ebx = pack4(vendor_chars);
            r.edx = pack4(vendor_chars + 4);
            r.ecx = pack4(vendor_chars + 8);
            break;
        }
        case 0x80000001:
            r.eax = signature(*this);
            // [0] LAHF/SAHF, [1] CmpLegacy (multi-core), [5] ABM/LZCNT, [8] PREFETCHW.
            r.ecx = flag(f.lahf, 0) | flag(htt, 1) | flag(f.lzcnt, 5) | flag(f.prefetchw, 8);
            // Duplicated leaf-1 bits, [20] NX, [27] RDTSCP, [29] long mode.
            r.edx = flag(f.x87, 0) | flag(f.tsc, 4) | flag(f.cx8, 8) | flag(f.cmov, 15) |
                    flag(f.mmx, 23) | flag(f.fxsr, 24) | bit(20) | flag(f.rdtscp, 27) | bit(29);
            break;
        case 0x80000002:
        case 0x80000003:
        case 0x80000004: {
            char text[48] = {};
            std::memcpy(text, brand.data(), std::min<size_t>(brand.size(), 47));
            const char* part = text + (leaf - 0x80000002) * 16;
            r.eax = pack4(part);
            r.ebx = pack4(part + 4);
            r.ecx = pack4(part + 8);
            r.edx = pack4(part + 12);
            break;
        }
        case 0x80000007:
            r.edx = flag(f.invariant_tsc, 8);
            break;
        case 0x80000008:
            r.eax = 48u | 48u << 8;  // physical / linear address bits
            r.ebx = flag(f.clzero, 0);
            // [7:0] NC = threads - 1, [15:12] APIC id size.
            r.ecx = (count - 1) | apic_id_bits(count) << 12;
            break;
        default:
            break;  // zeros
    }
    return r;
}

bool VirtualCpu::check_consistency(std::string* why) const {
    const CpuFeatures f = effective_features();
    auto fail = [&](const char* message) {
        if (why != nullptr) {
            *why = message;
        }
        return false;
    };

    const CpuidResult l0 = cpuid(0, 0);
    const CpuidResult l1 = cpuid(1, 0);
    const CpuidResult l7 = cpuid(7, 0);
    const CpuidResult e0 = cpuid(0x80000000, 0);
    const CpuidResult e1 = cpuid(0x80000001, 0);
    const CpuidResult e8 = cpuid(0x80000008, 0);

    if (vendor.size() != 12) {
        return fail("vendor string must be 12 characters");
    }
    if (brand.empty() || brand.size() > 47) {
        return fail("brand string must be 1..47 characters");
    }
    if (l0.eax < 7 || l0.eax > kMaxBasicLeaf || e0.eax < 0x80000008) {
        return fail("maximum leaves do not cover the implemented leaves");
    }
    if (l0.ebx != e0.ebx || l0.ecx != e0.ecx || l0.edx != e0.edx) {
        return fail("leaf 0 and 0x80000000 vendor strings differ");
    }
    if (l1.eax != e1.eax) {
        return fail("leaf 1 and 0x80000001 signatures differ");
    }
    if ((l1.ecx & bit(31)) != 0) {
        return fail("hypervisor bit must be clear");
    }
    // Dependency chains.
    if ((f.sse42 && !f.sse41) || (f.sse41 && !f.ssse3) || (f.ssse3 && !f.sse3) ||
        (f.sse3 && !f.sse2) || (f.sse2 && !f.sse)) {
        return fail("SSE feature chain is broken");
    }
    // XSAVE / OSXSAVE / XCR0 / leaf 0xD.
    const bool xsave_bit = (l1.ecx & bit(26)) != 0;
    const bool osxsave_bit = (l1.ecx & bit(27)) != 0;
    const bool avx_bit = (l1.ecx & bit(28)) != 0;
    const bool avx2_bit = (l7.ebx & bit(5)) != 0;
    const bool fma_bit = (l1.ecx & bit(12)) != 0;
    const bool f16c_bit = (l1.ecx & bit(29)) != 0;
    if (xsave_bit != osxsave_bit) {
        return fail("XSAVE and OSXSAVE differ (the OS enables XSAVE whenever it exists)");
    }
    if ((avx_bit || avx2_bit || fma_bit || f16c_bit) && !osxsave_bit) {
        return fail("AVX-class feature without OSXSAVE");
    }
    if ((avx2_bit || fma_bit || f16c_bit) && !avx_bit) {
        return fail("AVX2/FMA/F16C without AVX");
    }
    uint64_t xcr = 0;
    if (!xgetbv(0, &xcr)) {
        return fail("xgetbv(0) must succeed");
    }
    uint64_t unused = 0;
    if (xgetbv(1, &unused) || xgetbv(0xFFFFFFFFu, &unused)) {
        return fail("xgetbv must reject indices other than 0");
    }
    if ((xcr & (kXcr0X87 | kXcr0Sse)) != (kXcr0X87 | kXcr0Sse)) {
        return fail("XCR0 must always enable x87 and SSE state");
    }
    if (((xcr & kXcr0Avx) != 0) != avx_bit) {
        return fail("XCR0 AVX state bit and the AVX cpuid bit differ");
    }
    if ((xcr >> 3) != 0) {
        return fail("XCR0 enables a state component that is not implemented");
    }
    const CpuidResult d0 = cpuid(0xD, 0);
    const CpuidResult d1 = cpuid(0xD, 1);
    const CpuidResult d2 = cpuid(0xD, 2);
    if (xsave_bit) {
        if (d0.eax != static_cast<uint32_t>(xcr) || d0.edx != 0) {
            return fail("leaf 0xD.0 supported-state mask differs from XCR0");
        }
        if (d0.ebx != xsave_area_size(xcr) || d1.ebx != d0.ebx) {
            return fail("leaf 0xD XSAVE area size differs from the guest state layout");
        }
        if (d0.ecx < d0.ebx || d0.ecx != xsave_max_size(f)) {
            return fail("leaf 0xD.0 maximum XSAVE size is wrong");
        }
        if (avx_bit != (d2.eax == kXsaveAvxBytes && d2.ebx == kXsaveAvxOffset)) {
            return fail("leaf 0xD.2 does not describe the AVX component");
        }
    } else if (d0.eax | d0.ebx | d0.ecx | d0.edx | d1.ebx | d2.eax | d2.ebx) {
        return fail("leaf 0xD must be zero without XSAVE");
    }
    // Processor count.
    const uint32_t count = logical_count(*this);
    if (((l1.edx & bit(28)) != 0) != (count > 1)) {
        return fail("HTT bit and logical processor count differ");
    }
    if (count > 1 && ((l1.ebx >> 16) & 0xFF) != count) {
        return fail("leaf 1 logical processor count differs");
    }
    if (((e8.ecx & 0xFF) + 1) != count) {
        return fail("leaf 0x80000008 thread count differs");
    }
    // RDTSCP / TSC.
    if (((e1.edx & bit(27)) != 0) && !f.tsc) {
        return fail("RDTSCP without TSC");
    }
    return true;
}

VirtualCpu& virtual_cpu() {
    return default_cpu_instance();
}

void reset_virtual_cpu() {
    default_cpu_instance() = VirtualCpu{};
}

uint32_t xsave_area_size(uint64_t xcr0) {
    return kXsaveAvxOffset + ((xcr0 & kXcr0Avx) != 0 ? kXsaveAvxBytes : 0u);
}

uint64_t read_guest_tsc() {
    static_assert(std::atomic<uint64_t>::is_always_lock_free, "TSC state must be lock-free");
    int64_t qpc = 0;
    os::win32::QueryPerformanceCounter(&qpc);
    const uint64_t ns = qpc > 0 ? static_cast<uint64_t>(qpc) : 0;
    // ns * 3.6 without overflowing: kTscFrequencyHz / 1e9 == 18 / 5.
    static_assert(kTscFrequencyHz == 3'600'000'000ULL, "update the 18/5 scale with the frequency");
    const uint64_t now = ns / 5 * 18 + ns % 5 * 18 / 5;
    uint64_t last = g_last_tsc.load(std::memory_order_relaxed);
    for (;;) {
        const uint64_t next = now > last ? now : last + 1;
        if (g_last_tsc.compare_exchange_weak(last, next, std::memory_order_relaxed)) {
            return next;
        }
    }
}

}  // namespace playgta6::guest

// ---------------------------------------------------------------------------
// Helpers called by lifted code
// ---------------------------------------------------------------------------

namespace {

namespace GuestReg = playgta6::GuestReg;
using playgta6::GuestState;

uint32_t low32(const GuestState* state, uint32_t reg) {
    return static_cast<uint32_t>(state->gpr[reg]);
}

// 32-bit results are stored zero-extended into the 64-bit slots.
void set32(GuestState* state, uint32_t reg, uint32_t value) {
    state->gpr[reg] = value;
}

void set_tsc(GuestState* state) {
    const uint64_t tsc = playgta6::guest::read_guest_tsc();
    set32(state, GuestReg::Rax, static_cast<uint32_t>(tsc));
    set32(state, GuestReg::Rdx, static_cast<uint32_t>(tsc >> 32));
}

}  // namespace

extern "C" {

PG6_GUEST_HELPER void __playgta6_cpuid(GuestState* state) {
    const auto r = playgta6::guest::virtual_cpu().cpuid(low32(state, GuestReg::Rax),
                                                        low32(state, GuestReg::Rcx));
    set32(state, GuestReg::Rax, r.eax);
    set32(state, GuestReg::Rbx, r.ebx);
    set32(state, GuestReg::Rcx, r.ecx);
    set32(state, GuestReg::Rdx, r.edx);
}

PG6_GUEST_HELPER void __playgta6_rdtsc(GuestState* state) {
    set_tsc(state);
}

PG6_GUEST_HELPER void __playgta6_rdtscp(GuestState* state) {
    set_tsc(state);
    set32(state, GuestReg::Rcx, playgta6::guest::virtual_cpu().tsc_aux);
}

PG6_GUEST_HELPER void __playgta6_xgetbv(GuestState* state) {
    const uint32_t index = low32(state, GuestReg::Rcx);
    uint64_t value = 0;
    if (!playgta6::guest::virtual_cpu().xgetbv(index, &value)) {
        // #GP; the faulting va is the state's rip (lifted code does not maintain it, the
        // dispatcher leaves the entry of the running function there).
        playgta6::guest::raise_fault(playgta6::guest::FaultKind::GeneralProtection, state,
                                     state->rip, index);
    }
    set32(state, GuestReg::Rax, static_cast<uint32_t>(value));
    set32(state, GuestReg::Rdx, static_cast<uint32_t>(value >> 32));
}

}  // extern "C"
