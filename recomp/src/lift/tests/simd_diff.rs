//! Differential testing of the SSE/AVX lifter against the host CPU (x86-64 hosts only).
//!
//! Every instruction form under test is executed twice on identical inputs:
//! - natively: its encoding is copied into a small JIT stub ([`jit`]) that loads `ymm0..15`, the
//!   GPRs, RFLAGS and MXCSR (`0x1F80`) from a context block, executes the instruction and stores
//!   everything back;
//! - lifted: `instruction; ret` is discovered, lifted, compiled to wasm64 and run under node.
//!
//! The full YMM registers `ymm0..3` (so VEX zeroing and legacy preservation of the upper halves
//! are checked too; without AVX only the XMM halves), all GPRs except `rbx` (the memory operand base, a host buffer
//! natively and guest memory when lifted) and `rsp`, all six arithmetic flags and the 64-byte
//! memory buffer are compared. Inputs combine per-lane edge cases (signed zeros, denormals,
//! infinities, quiet and signalling NaNs with payloads, rounding and conversion boundaries,
//! integer extremes) such that every ordered pair of special values meets in some lane, plus
//! pseudo-random values. Instruction forms the host CPU lacks are skipped with a printed note.
//!
//! [`Check`] relaxes the comparison where the lifter documents a deviation: NaN results may
//! differ in payload and sign (but must be NaN), and `rcp`/`rsqrt` results are compared with a
//! relative error bound.

use iced_x86::code_asm::CodeAssembler;
use iced_x86::{Decoder, Formatter, IcedError, IntelFormatter};

use super::exec::{
    CODE_BASE, Case, DATA_ADDR, RAX, RBX, RDX, RSP, STACK_TOP, STATE_ADDR, build, exec_or_skip, run,
};
use super::ir::Lcg;
use super::{Program, assemble};
use crate::lift::{STATE_XMM_OFFSET, STATE_YMM_HI_OFFSET};

/// Executes single instructions on the host CPU.
pub(super) mod jit {
    use std::arch::asm;
    use std::sync::Mutex;

    use iced_x86::code_asm::*;

    /// Register file image read and written by a stub.
    #[repr(C, align(16))]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Ctx {
        pub ymm: [[u8; 32]; 16],
        pub gpr: [u64; 16],
        pub rflags: u64,
        pub mxcsr: u32,
        pub pad: u32,
    }

    const YMM: i32 = 0;
    const GPR: i32 = 512;
    const RFLAGS: i32 = 640;
    const MXCSR: i32 = 648;
    const ARENA_BYTES: usize = 16 << 20;

    /// A stub in executable memory.
    #[derive(Debug, Clone, Copy)]
    pub struct Stub(usize);

    static ARENA: Mutex<(usize, usize)> = Mutex::new((0, 0));

    #[cfg(windows)]
    fn alloc_exec(len: usize) -> Option<usize> {
        use std::ffi::c_void;
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn VirtualAlloc(addr: *mut c_void, size: usize, kind: u32, protect: u32)
            -> *mut c_void;
        }
        // MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE
        // SAFETY: plain allocation call; the result is checked.
        let p = unsafe { VirtualAlloc(std::ptr::null_mut(), len, 0x3000, 0x40) };
        (!p.is_null()).then_some(p as usize)
    }

    #[cfg(unix)]
    fn alloc_exec(len: usize) -> Option<usize> {
        use std::ffi::c_void;
        unsafe extern "C" {
            fn mmap(
                addr: *mut c_void,
                len: usize,
                prot: i32,
                flags: i32,
                fd: i32,
                off: i64,
            ) -> *mut c_void;
        }
        // PROT_READ | PROT_WRITE | PROT_EXEC; MAP_PRIVATE | MAP_ANONYMOUS
        let flags = if cfg!(target_os = "macos") {
            0x1002
        } else {
            0x22
        };
        // SAFETY: anonymous mapping; the result is checked.
        let p = unsafe { mmap(std::ptr::null_mut(), len, 7, flags, -1, 0) };
        (p as isize != -1 && !p.is_null()).then_some(p as usize)
    }

    #[cfg(not(any(windows, unix)))]
    fn alloc_exec(_len: usize) -> Option<usize> {
        None
    }

    fn prologue(avx: bool) -> Vec<u8> {
        let mut a = CodeAssembler::new(64).expect("assembler");
        let xmms = [
            xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13,
            xmm14, xmm15,
        ];
        let ymms = [
            ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12, ymm13,
            ymm14, ymm15,
        ];
        let gprs = [
            (rcx, 1),
            (rdx, 2),
            (rbx, 3),
            (rbp, 5),
            (rsi, 6),
            (rdi, 7),
            (r8, 8),
            (r9, 9),
            (r10, 10),
            (r11, 11),
            (r12, 12),
            (r13, 13),
            (r14, 14),
            (r15, 15),
        ];
        (|| -> Result<(), IcedError> {
            for r in [rbx, rbp, rsi, rdi, r12, r13, r14, r15] {
                a.push(r)?;
            }
            a.sub(rsp, 8)?;
            a.stmxcsr(dword_ptr(rsp))?;
            a.push(rax)?;
            a.ldmxcsr(dword_ptr(rax + MXCSR))?;
            for (i, (x, y)) in xmms.iter().zip(ymms).enumerate() {
                let at = rax + (YMM + 32 * i as i32);
                if avx {
                    a.vmovdqu(y, ymmword_ptr(at))?;
                } else {
                    a.movdqu(*x, xmmword_ptr(at))?;
                }
            }
            a.push(qword_ptr(rax + RFLAGS))?;
            a.popfq()?;
            for (r, i) in gprs {
                a.mov(r, qword_ptr(rax + (GPR + 8 * i)))?;
            }
            a.mov(rax, qword_ptr(rax + GPR))
        })()
        .expect("prologue assembles");
        a.assemble(0).expect("prologue encodes")
    }

    fn epilogue(avx: bool) -> Vec<u8> {
        let mut a = CodeAssembler::new(64).expect("assembler");
        let xmms = [
            xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13,
            xmm14, xmm15,
        ];
        let ymms = [
            ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12, ymm13,
            ymm14, ymm15,
        ];
        let gprs = [
            (rcx, 1),
            (rdx, 2),
            (rbx, 3),
            (rbp, 5),
            (rsi, 6),
            (rdi, 7),
            (r8, 8),
            (r9, 9),
            (r10, 10),
            (r11, 11),
            (r12, 12),
            (r13, 13),
            (r14, 14),
            (r15, 15),
        ];
        (|| -> Result<(), IcedError> {
            a.push(rax)?;
            a.mov(rax, qword_ptr(rsp + 8))?;
            a.pushfq()?;
            a.pop(qword_ptr(rax + RFLAGS))?;
            for (r, i) in gprs {
                a.mov(qword_ptr(rax + (GPR + 8 * i)), r)?;
            }
            a.pop(rcx)?;
            a.mov(qword_ptr(rax + GPR), rcx)?;
            for (i, (x, y)) in xmms.iter().zip(ymms).enumerate() {
                let at = rax + (YMM + 32 * i as i32);
                if avx {
                    a.vmovdqu(ymmword_ptr(at), y)?;
                } else {
                    a.movdqu(xmmword_ptr(at), *x)?;
                }
            }
            a.stmxcsr(dword_ptr(rax + MXCSR))?;
            a.pop(rax)?;
            a.ldmxcsr(dword_ptr(rsp))?;
            a.add(rsp, 8)?;
            for r in [r15, r14, r13, r12, rdi, rsi, rbp, rbx] {
                a.pop(r)?;
            }
            a.ret()
        })()
        .expect("epilogue assembles");
        a.assemble(0).expect("epilogue encodes")
    }

    /// Copies `prologue; instruction; epilogue` into executable memory. `instruction` must be
    /// position independent and must not touch `rsp`. `None` if no executable memory is available.
    pub fn stub(instruction: &[u8]) -> Option<Stub> {
        let avx = is_x86_feature_detected!("avx");
        let code = [prologue(avx), instruction.to_vec(), epilogue(avx)].concat();
        let mut arena = ARENA.lock().unwrap_or_else(|e| e.into_inner());
        if arena.0 == 0 {
            arena.0 = alloc_exec(ARENA_BYTES)?;
        }
        let at = (arena.1 + 15) & !15;
        if at + code.len() > ARENA_BYTES {
            return None;
        }
        // SAFETY: `at..at + len` lies inside the arena, which nothing else references.
        unsafe {
            std::ptr::copy_nonoverlapping(code.as_ptr(), (arena.0 + at) as *mut u8, code.len());
        }
        arena.1 = at + code.len();
        Some(Stub(arena.0 + at))
    }

    /// Runs a stub on `ctx`.
    pub fn run(stub: Stub, ctx: &mut Ctx) {
        // SAFETY: the stub preserves rbx, rbp, rsi, rdi, r12-r15 and MXCSR, keeps the stack
        // balanced and only touches `ctx` plus the memory its instruction addresses (callers pass
        // valid buffers). Every other register is declared clobbered.
        unsafe {
            asm!(
                "call {f}",
                f = in(reg) stub.0,
                inout("rax") ctx as *mut Ctx => _,
                out("xmm0") _, out("xmm1") _, out("xmm2") _, out("xmm3") _,
                out("xmm4") _, out("xmm5") _, out("xmm6") _, out("xmm7") _,
                out("xmm8") _, out("xmm9") _, out("xmm10") _, out("xmm11") _,
                out("xmm12") _, out("xmm13") _, out("xmm14") _, out("xmm15") _,
                clobber_abi("C"),
            );
        }
    }
}

/// CPU feature an instruction form needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Feature {
    Sse2,
    Sse3,
    Ssse3,
    Sse41,
    Sse42,
    Avx,
    Avx2,
    Fma,
    F16c,
    Aes,
    Pclmulqdq,
    Vaes,
    Vpclmulqdq,
}

impl Feature {
    fn available(self) -> bool {
        match self {
            Feature::Sse2 => true,
            Feature::Sse3 => is_x86_feature_detected!("sse3"),
            Feature::Ssse3 => is_x86_feature_detected!("ssse3"),
            Feature::Sse41 => is_x86_feature_detected!("sse4.1"),
            Feature::Sse42 => is_x86_feature_detected!("sse4.2"),
            Feature::Avx => is_x86_feature_detected!("avx"),
            Feature::Avx2 => is_x86_feature_detected!("avx2"),
            Feature::Fma => is_x86_feature_detected!("fma"),
            Feature::F16c => is_x86_feature_detected!("f16c"),
            Feature::Aes => is_x86_feature_detected!("aes"),
            Feature::Pclmulqdq => is_x86_feature_detected!("pclmulqdq"),
            Feature::Vaes => is_x86_feature_detected!("vaes") && is_x86_feature_detected!("avx2"),
            Feature::Vpclmulqdq => {
                is_x86_feature_detected!("vpclmulqdq") && is_x86_feature_detected!("avx2")
            }
        }
    }
}

/// Value pool of the XMM lanes and memory (or of `rax`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Kind {
    /// binary16 bit patterns (F16C).
    F16,
    F32,
    /// binary32 specials, random values within a factor of 8 of each other (products and
    /// addends of similar magnitude: cancellation) with often-short mantissas (rounding ties).
    F32Near,
    F64,
    /// binary64 counterpart of [`Kind::F32Near`].
    F64Near,
    /// binary64 specials and uniformly random bit patterns (every exponent: overflow,
    /// underflow, subnormal results).
    F64Any,
    I8,
    I16,
    I32,
    I64,
    /// 64-bit shift counts around every lane width.
    Count,
}

impl Kind {
    fn bytes(self) -> usize {
        match self {
            Kind::I8 => 1,
            Kind::I16 | Kind::F16 => 2,
            Kind::F32 | Kind::F32Near | Kind::I32 => 4,
            Kind::F64 | Kind::F64Near | Kind::F64Any | Kind::I64 | Kind::Count => 8,
        }
    }

    pub fn pool_len(self) -> usize {
        self.pool().len()
    }

    fn pool(self) -> Vec<u64> {
        match self {
            Kind::F32 | Kind::F32Near => F32_SPECIALS.iter().map(|&v| u64::from(v)).collect(),
            Kind::F16 => vec![
                0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3C00, 0xBC00, 0x3C01, 0x7BFF,
                0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01, 0x7D55, 0xFDFF, 0x3555, 0x0200,
            ],
            Kind::F64 | Kind::F64Near | Kind::F64Any => F64_SPECIALS.to_vec(),
            Kind::Count => vec![
                0,
                1,
                2,
                7,
                8,
                9,
                15,
                16,
                17,
                31,
                32,
                33,
                63,
                64,
                65,
                127,
                128,
                255,
                256,
                0x1_0000_0001,
                u64::MAX,
            ],
            int => {
                let bits = int.bytes() * 8;
                let m = if bits == 64 {
                    u64::MAX
                } else {
                    (1u64 << bits) - 1
                };
                let min = 1u64 << (bits - 1);
                let mut v = vec![
                    0,
                    1,
                    2,
                    3,
                    m,
                    m - 1,
                    min,
                    min + 1,
                    min - 1,
                    min - 2,
                    0x5555_5555_5555_5555 & m,
                    0xAAAA_AAAA_AAAA_AAAA & m,
                    0x7F & m,
                    0x80 & m,
                    0xFF & m,
                    0x100 & m,
                    0x7FFF & m,
                    0x8000 & m,
                    0xFFFF & m,
                    0x1234_5678_9ABC_DEF0 & m,
                ];
                if bits >= 32 {
                    // Integers that are not exactly representable as floats/doubles.
                    v.extend([
                        16_777_217,
                        16_777_219,
                        0x7FFF_FFC0,
                        (-16_777_217i64) as u64 & m,
                    ]);
                }
                if bits == 64 {
                    v.extend([
                        (1 << 53) + 1,
                        (1 << 62) + 1,
                        i64::MAX as u64 - 511,
                        (-(1i64 << 53) - 1) as u64,
                        0xFFFF_FFFF,
                        0x8000_0000,
                    ]);
                }
                v.sort_unstable();
                v.dedup();
                v
            }
        }
    }

    fn random(self, rng: &mut Lcg) -> u64 {
        let r = rng.next();
        match self {
            // Random magnitudes over a useful range, random sign.
            Kind::F32 => {
                let exp = 100 + (r >> 8) % 56; // 2^-27 .. 2^28
                ((r >> 63) << 31 | exp << 23 | (r & 0x7F_FFFF)) & 0xFFFF_FFFF
            }
            Kind::F64 => {
                let exp = 990 + (r >> 12) % 70;
                (r >> 63) << 63 | exp << 52 | (r & 0xF_FFFF_FFFF_FFFF)
            }
            Kind::Count => r % 80,
            Kind::F64Any => r,
            Kind::F64Near => {
                let exp = 1022 + (r >> 12) % 4;
                let mant = if r >> 50 & 1 == 1 {
                    r & 0xF_FFFF_FFF0_0000
                } else {
                    r & 0xF_FFFF_FFFF_FFFF
                };
                (r >> 63) << 63 | exp << 52 | mant
            }
            Kind::F32Near => {
                let exp = 126 + (r >> 8) % 4;
                let mant = if r >> 20 & 1 == 1 {
                    r & 0x7F_F000
                } else {
                    r & 0x7F_FFFF
                };
                ((r >> 63) << 31 | exp << 23 | mant) & 0xFFFF_FFFF
            }
            Kind::F16 => r & 0xFFFF,
            int => {
                let bits = int.bytes() * 8;
                if bits == 64 {
                    r
                } else {
                    r & ((1u64 << bits) - 1)
                }
            }
        }
    }
}

/// float bit patterns: signed zeros, denormals, normal boundaries, rounding ties, conversion
/// limits, infinities and NaNs (quiet/signalling, with payloads, both signs).
const F32_SPECIALS: [u32; 55] = [
    // binary16 conversion boundaries: max half, the overflow tie, min normal/denormal halves,
    // the underflow tie, the largest denormal half, round-to-nearest ties at 1.0.
    0x477F_E000,
    0x477F_F000,
    0x477F_EFFF,
    0xC77F_F000,
    0x3880_0000,
    0x3380_0000,
    0x3300_0000,
    0x3300_0001,
    0x387F_C000,
    0x3F80_1000,
    0x3F80_3000,
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x8000_0001,
    0x007F_FFFF,
    0x807F_FFFF,
    0x0080_0000,
    0x8080_0000,
    0x3F80_0000,
    0xBF80_0000,
    0x3FC0_0000,
    0x4020_0000,
    0xC020_0000,
    0x3F00_0000,
    0xBF00_0000,
    0x3EFF_FFFF,
    0x3F7F_FFFF,
    0x7F7F_FFFF,
    0xFF7F_FFFF,
    0x7F80_0000,
    0xFF80_0000,
    0x7FC0_0000,
    0xFFC0_0000,
    0x7F80_0001,
    0xFFA1_2345,
    0x7FD2_3456,
    0x4F00_0000,
    0xCF00_0000,
    0x4EFF_FFFF,
    0xCF00_0001,
    0x5F00_0000,
    0xDF00_0000,
    0x5EFF_FFFF,
    0xDF00_0001,
    0x4B7F_FFFF,
    0x4B00_0001,
    0x4040_0000,
    0x42F6_E979,
    0xC2F6_E979,
    0x7E80_0000,
    0x7E7F_FFFF,
    0x00FF_FFFF,
    0x4049_0FDB,
    0x3F00_0001,
];

/// double bit patterns (same idea as [`F32_SPECIALS`]).
const F64_SPECIALS: [u64; 46] = [
    0x0000_0000_0000_0000,
    0x8000_0000_0000_0000,
    0x0000_0000_0000_0001,
    0x8000_0000_0000_0001,
    0x000F_FFFF_FFFF_FFFF,
    0x0010_0000_0000_0000,
    0x3FF0_0000_0000_0000,
    0xBFF0_0000_0000_0000,
    0x3FF8_0000_0000_0000,
    0x4004_0000_0000_0000,
    0xC004_0000_0000_0000,
    0x3FE0_0000_0000_0000,
    0xBFE0_0000_0000_0000,
    0x3FDF_FFFF_FFFF_FFFF,
    0x7FEF_FFFF_FFFF_FFFF,
    0xFFEF_FFFF_FFFF_FFFF,
    0x7FF0_0000_0000_0000,
    0xFFF0_0000_0000_0000,
    0x7FF8_0000_0000_0000,
    0xFFF8_0000_0000_0000,
    0x7FF0_0000_0000_0001,
    0xFFF4_0000_0000_1234,
    0x7FFA_BCDE_F012_3456,
    0x41E0_0000_0000_0000, // 2^31
    0xC1E0_0000_0000_0000, // -2^31
    0x41DF_FFFF_FFE0_0000, // 2^31 - 1
    0x41DF_FFFF_FFF0_0000, // 2^31 - 0.5
    0xC1E0_0000_0010_0000, // -2^31 - 0.5
    0xC1E0_0000_0020_0000, // -2^31 - 1
    0x43E0_0000_0000_0000, // 2^63
    0xC3E0_0000_0000_0000, // -2^63
    0x43DF_FFFF_FFFF_FFFF,
    0xC3E0_0000_0000_0001,
    0x4330_0000_0000_0001, // 2^52 + 1
    0x47EF_FFFF_E000_0000, // FLT_MAX
    0x47EF_FFFF_F000_0000, // FLT_MAX + half an ulp: rounds to inf
    0x36A0_0000_0000_0000, // 2^-149 (smallest float denormal)
    0x3690_0000_0000_0000, // 2^-150 (ties to zero)
    0x3690_0000_0000_0001,
    0x3810_0000_0000_0000, // FLT_MIN
    0x3FF0_0000_1000_0000, // 1 + 2^-24 (float rounding tie)
    0x3FF0_0000_3000_0000, // 1 + 3*2^-24
    0x4059_0000_0000_0000,
    0x400C_0000_0000_0000,
    0x7E37_E43C_8800_759C, // 1e300
    0x8000_0000_0000_FFFF,
];

/// How lifted XMM results are compared with the host's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Check {
    /// Bit-exact.
    Exact,
    /// Bit-exact, except that a NaN host lane (of this many bits) only requires a NaN.
    Nan(u32),
    /// `rcp`/`rsqrt`: float lanes within a relative error of 2^-11 (hardware guarantees
    /// 1.5 * 2^-12 against the exact value the lifter computes); NaN/inf/zero classes must match.
    Approx,
}

/// One instruction form under test. Conventions: `xmm1` is the destination (and legacy first
/// source), `xmm2` or `[rbx]` the second source, `xmm3` the VEX first source, `xmm0` the
/// implicit `blendv` mask; GPR operands use `rax`/`rcx`.
pub(super) struct Op {
    pub name: String,
    pub bytes: Vec<u8>,
    pub feature: Feature,
    pub kind: Kind,
    pub gpr: Kind,
    pub check: Check,
    pub cases: usize,
    /// Put a shift count (around every lane width) into the low quadword of `xmm2`/`[rbx]`.
    pub shift_count: bool,
    /// Also draw `rdx` from the `gpr` pool (the explicit string lengths of `pcmpestr*`).
    pub lengths: bool,
}

impl Op {
    pub fn cases(mut self, n: usize) -> Self {
        self.cases = n;
        self
    }

    pub fn lengths(mut self) -> Self {
        self.lengths = true;
        self
    }

    pub fn shift_count(mut self) -> Self {
        self.shift_count = true;
        self
    }

    pub fn gpr(mut self, kind: Kind) -> Self {
        self.gpr = kind;
        self
    }

    /// Enough cases for every ordered pair of pool values to meet in lane 0 (`scalar`) or in some
    /// lane (packed).
    pub fn pairs(mut self, scalar: bool) -> Self {
        let p = self.kind.pool().len();
        let lanes = 16 / self.kind.bytes();
        self.cases = if scalar {
            p * p
        } else {
            (p * p).div_ceil(lanes)
        };
        self
    }
}

/// Builds an [`Op`] from assembler calls (named after the disassembly).
pub(super) fn op(
    feature: Feature,
    kind: Kind,
    check: Check,
    build: impl FnOnce(&mut CodeAssembler) -> Result<(), IcedError>,
) -> Op {
    let bytes = assemble(0, build);
    let mut name = String::new();
    let mut decoder = Decoder::with_ip(64, &bytes, 0, 0);
    let mut formatter = IntelFormatter::new();
    while decoder.can_decode() {
        if !name.is_empty() {
            name.push_str("; ");
        }
        formatter.format(&decoder.decode(), &mut name);
    }
    Op {
        name,
        bytes,
        feature,
        kind,
        gpr: Kind::I64,
        check,
        cases: 48,
        shift_count: false,
        lengths: false,
    }
}

/// Inputs of one case.
#[derive(Debug, Clone)]
struct Input {
    ymm: [[u8; 32]; 4],
    gpr: [u64; 16],
    mem: [u8; 64],
    flags: [bool; 6],
}

fn put(dst: &mut [u8], value: u64) {
    let bytes = value.to_le_bytes();
    dst.copy_from_slice(&bytes[..dst.len()]);
}

fn input(op: &Op, k: usize, rng: &mut Lcg) -> Input {
    let pool = op.kind.pool();
    let p = pool.len();
    let w = op.kind.bytes();
    let lanes = 32 / w;
    let span = (p * p).div_ceil(lanes);
    // Lane j of case k visits pair index c = k + j * span: lane 0 alone covers all ordered pairs
    // after p^2 cases, all lanes together after p^2 / lanes cases.
    let lane_values = |k: usize, j: usize, rng: &mut Lcg| -> (u64, u64) {
        if (k + j) % 7 == 6
            || (matches!(op.kind, Kind::F32Near | Kind::F64Near | Kind::F64Any)
                && !(k + j).is_multiple_of(7))
        {
            return (op.kind.random(rng), op.kind.random(rng));
        }
        let c = k + j * span;
        (pool[c % p], pool[(c / p) % p])
    };
    let mut xmm = [[0u8; 32]; 4];
    let mut mem = [0u8; 64];
    for (i, b) in mem.iter_mut().enumerate() {
        *b = (rng.next() >> (i % 8)) as u8;
    }
    for j in 0..lanes {
        let (a, b) = lane_values(k, j, rng);
        let (a3, _) = lane_values(k.wrapping_add(17), j, rng);
        put(&mut xmm[1][j * w..(j + 1) * w], a);
        put(&mut xmm[3][j * w..(j + 1) * w], a3);
        put(&mut xmm[2][j * w..(j + 1) * w], b);
        if (j + 1) * w <= 32 {
            put(&mut mem[j * w..(j + 1) * w], b);
        }
        // VEX forms read xmm3 as the first source: give it the "a" pattern of the pair too.
        if j % 2 == 0 {
            put(&mut xmm[3][j * w..(j + 1) * w], a);
        }
    }
    for b in xmm[0].iter_mut() {
        *b = rng.next() as u8;
    }
    let mut gpr = [0u64; 16];
    for (i, r) in gpr.iter_mut().enumerate() {
        *r = rng.next() ^ (i as u64) << 56;
    }
    if op.shift_count {
        let counts = Kind::Count.pool();
        let c = counts[k % counts.len()];
        put(&mut xmm[2][..8], c);
        put(&mut mem[..8], c);
    }
    let gpool = op.gpr.pool();
    gpr[RAX] = if k % 5 == 4 {
        op.gpr.random(rng)
    } else {
        gpool[k % gpool.len()]
    };
    // Sign-extend narrow integer pools so 32-bit GPR forms also see "negative" upper halves.
    if op.gpr.bytes() < 8 && k % 3 == 1 {
        gpr[RAX] |= 0xFFFF_FFFF_0000_0000;
    }
    if op.lengths {
        gpr[RDX] = gpool[(k / 3 + 1) % gpool.len()];
        if op.gpr.bytes() < 8 && k % 4 == 1 {
            gpr[RDX] |= 0xFFFF_FFFF_0000_0000;
        }
    }
    let f = rng.next();
    Input {
        ymm: xmm,
        gpr,
        mem,
        flags: std::array::from_fn(|i| f >> i & 1 == 1),
    }
}

/// Native result: xmm0..3, GPRs, flags `[CF PF AF ZF SF OF]`, memory buffer.
#[derive(Debug)]
struct Output {
    ymm: [[u8; 32]; 4],
    gpr: [u64; 16],
    flags: [bool; 6],
    mem: [u8; 64],
}

const FLAG_BITS: [u32; 6] = [0, 2, 4, 6, 7, 11];
const FLAG_NAMES: [&str; 6] = ["CF", "PF", "AF", "ZF", "SF", "OF"];

#[repr(C, align(64))]
struct Buffer([u8; 64]);

fn native(stub: jit::Stub, input: &Input) -> Output {
    let mut buf = Buffer(input.mem);
    let mut ctx = jit::Ctx {
        mxcsr: 0x1F80,
        ..Default::default()
    };
    ctx.ymm[..4].copy_from_slice(&input.ymm);
    ctx.gpr = input.gpr;
    ctx.gpr[RBX] = buf.0.as_mut_ptr() as u64;
    ctx.rflags = input
        .flags
        .iter()
        .zip(FLAG_BITS)
        .map(|(&f, bit)| u64::from(f) << bit)
        .sum();
    jit::run(stub, &mut ctx);
    let mut gpr = ctx.gpr;
    gpr[RBX] = input.gpr[RBX];
    Output {
        ymm: std::array::from_fn(|i| ctx.ymm[i]),
        gpr,
        flags: std::array::from_fn(|i| ctx.rflags >> FLAG_BITS[i] & 1 == 1),
        mem: buf.0,
    }
}

fn is_nan(bits: u64, width: u32) -> bool {
    match width {
        32 => f32::from_bits(bits as u32).is_nan(),
        _ => f64::from_bits(bits).is_nan(),
    }
}

/// Whether the lifted 16 bytes `got` are acceptable for the host's `host`.
fn ymm_matches(check: Check, host: &[u8; 32], got: &[u8; 32]) -> bool {
    if host == got {
        return true;
    }
    let lane = |b: &[u8; 32], i: usize, w: usize| {
        let mut v = [0u8; 8];
        v[..w].copy_from_slice(&b[i * w..(i + 1) * w]);
        u64::from_le_bytes(v)
    };
    match check {
        Check::Exact => false,
        Check::Nan(width) => {
            let w = width as usize / 8;
            (0..32 / w).all(|i| {
                let (h, g) = (lane(host, i, w), lane(got, i, w));
                h == g || (is_nan(h, width) && is_nan(g, width))
            })
        }
        Check::Approx => (0..8).all(|i| {
            let h = f32::from_bits(lane(host, i, 4) as u32);
            let g = f32::from_bits(lane(got, i, 4) as u32);
            // Results at the bottom of the normal range may flush to zero on one side only
            // (CPUs differ in where exactly they flush), so near-tiny results only need to be
            // tiny on both sides.
            let tiny = |v: f32| v.abs() <= 2.0 * f32::MIN_POSITIVE;
            let same_sign = h.is_sign_negative() == g.is_sign_negative();
            if h.is_nan() {
                g.is_nan()
            } else if h.is_infinite() || g.is_infinite() {
                h.to_bits() == g.to_bits()
            } else {
                let close = h != 0.0 && f64::from((h - g).abs()) <= f64::from(h.abs()) / 2048.0;
                same_sign && (close || (tiny(h) && tiny(g)))
            }
        }),
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().rev().map(|b| format!("{b:02x}")).collect()
}

/// Runs every op natively and lifted and asserts that all outputs agree.
pub(super) fn check_ops(test: &str, ops: Vec<Op>) {
    if !exec_or_skip(test) {
        return;
    }
    let (ops, skipped): (Vec<Op>, Vec<Op>) = ops.into_iter().partition(|op| op.feature.available());
    for op in &skipped {
        println!(
            "SKIP {test}: host CPU lacks {:?} for `{}`",
            op.feature, op.name
        );
    }
    let mut stubs = Vec::new();
    for op in &ops {
        match jit::stub(&op.bytes) {
            Some(stub) => stubs.push(stub),
            None => {
                println!("SKIP {test}: no executable memory for native reference runs");
                return;
            }
        }
    }
    let mut program = Program::new();
    for (i, op) in ops.iter().enumerate() {
        let va = CODE_BASE + i as u64 * 0x20;
        program = program.bytes(va, &[op.bytes.as_slice(), &[0xC3]].concat());
    }
    let functions = program.discover();
    assert_eq!(functions.len(), ops.len(), "one function per op");
    let built = build(test, functions.values());

    let mut rng = Lcg(0x55E_D1FF);
    let mut cases = Vec::new();
    let mut meta = Vec::new();
    let xmm_state = STATE_ADDR + STATE_XMM_OFFSET;
    let ymm_state = STATE_ADDR + STATE_YMM_HI_OFFSET;
    for (i, op) in ops.iter().enumerate() {
        let va = CODE_BASE + i as u64 * 0x20;
        for k in 0..op.cases {
            let inp = input(op, k, &mut rng);
            let mut case = Case::new(va);
            case.regs = inp.gpr;
            case.regs[RBX] = DATA_ADDR;
            case.regs[RSP] = STACK_TOP;
            case.flags = inp.flags;
            let lo: Vec<u8> = inp.ymm.iter().flat_map(|y| y[..16].to_vec()).collect();
            let hi: Vec<u8> = inp.ymm.iter().flat_map(|y| y[16..].to_vec()).collect();
            case.mem = vec![
                (xmm_state, lo),
                (ymm_state, hi),
                (DATA_ADDR, inp.mem.to_vec()),
            ];
            case.read = vec![(xmm_state, 64), (ymm_state, 64), (DATA_ADDR, 64)];
            case.out = (0..16).collect();
            cases.push(case);
            meta.push((i, inp));
        }
    }
    let outcomes = run(&built, &cases);

    let mut mismatches = Vec::new();
    for (o, (i, inp)) in outcomes.iter().zip(&meta) {
        let op = &ops[*i];
        let host = native(stubs[*i], inp);
        let label = format!(
            "`{}` ymm1={} ymm2={} ymm3={} ymm0={} rax={:#x} mem={}",
            op.name,
            hex(&inp.ymm[1]),
            hex(&inp.ymm[2]),
            hex(&inp.ymm[3]),
            hex(&inp.ymm[0]),
            inp.gpr[RAX],
            hex(&inp.mem[..32])
        );
        if !o.error.is_empty() {
            mismatches.push(format!("{label}: trapped: {}", o.error));
            continue;
        }
        for x in 0..4 {
            let mut got = [0u8; 32];
            got[..16].copy_from_slice(&o.mem[0][16 * x..16 * (x + 1)]);
            got[16..].copy_from_slice(&o.mem[1][16 * x..16 * (x + 1)]);
            let mut expected = host.ymm[x];
            if !is_x86_feature_detected!("avx") {
                // The host stub could not observe the upper halves; legacy code preserves them.
                expected[16..].copy_from_slice(&inp.ymm[x][16..]);
            }
            if !ymm_matches(op.check, &expected, &got) {
                mismatches.push(format!(
                    "{label}: ymm{x} = {}, host {}",
                    hex(&got),
                    hex(&expected)
                ));
            }
        }
        for r in 0..16 {
            if r == RSP || r == RBX {
                continue;
            }
            if o.regs[r] != host.gpr[r] {
                mismatches.push(format!(
                    "{label}: gpr{r} = {:#x}, host {:#x}",
                    o.regs[r], host.gpr[r]
                ));
            }
        }
        if o.regs[RBX] != DATA_ADDR || o.regs[RSP] != STACK_TOP {
            mismatches.push(format!("{label}: rbx/rsp changed"));
        }
        for f in 0..6 {
            if o.flags[f] != host.flags[f] {
                mismatches.push(format!(
                    "{label}: {} = {}, host {}",
                    FLAG_NAMES[f], o.flags[f], host.flags[f]
                ));
            }
        }
        if o.mem[2] != host.mem {
            mismatches.push(format!(
                "{label}: memory = {}, host {}",
                hex(&o.mem[2]),
                hex(&host.mem)
            ));
        }
    }
    println!(
        "{test}: {} instruction forms, {} cases, {} skipped forms",
        ops.len(),
        cases.len(),
        skipped.len()
    );
    assert!(
        mismatches.is_empty(),
        "{} mismatches, first ones:\n{}",
        mismatches.len(),
        mismatches
            .iter()
            .take(30)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
