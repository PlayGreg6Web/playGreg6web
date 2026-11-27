//! SSE/SSE2/SSE3/SSSE3/SSE4.1/SSE4.2, AVX, AVX2, FMA3 and F16C instruction lifting.
//!
//! # Vector registers
//! `ymm0..15` live in the guest state in two parts: bits 0..127 (`xmm0..15`) at
//! [`STATE_XMM_OFFSET`](super::super::STATE_XMM_OFFSET) and bits 128..255 at
//! [`STATE_YMM_HI_OFFSET`](super::super::STATE_YMM_HI_OFFSET), 16 bytes each, little endian.
//! Inside a function every 128-bit `xmm0..15` register the function touches gets an
//! `alloca <2 x i64>` (the canonical in-memory type of a 128-bit part), `%xmmN`, loaded from the
//! state at entry and after every call and stored back before every call, exit and trap, exactly
//! like the GPRs. Upper-half slots (`%ymmhN`) are allocated only for `ymm` registers whose upper
//! 128 bits the function actually reads; write-only upper halves (including VEX-128 zeroing and
//! `vzeroupper`/`vzeroall`) write directly to `%state.ymmhN`, eliminating dead allocas and
//! call-boundary spill/reload traffic. Instructions `bitcast` the value to the lane type they
//! operate on (`<4 x float>`, `<16 x i8>`, ...); LLVM promotes the allocas to SSA values and the
//! bitcasts are free on wasm (`v128`). The import thunks (`crate::stubs`) therefore always find
//! float arguments in the low lanes of the `xmm0..3` slots and leave float results in `xmm0`.
//! Legacy-SSE encodings leave bits 128..255 untouched (exact), VEX-128 encodings zero them
//! (exact), `vzeroupper`/`vzeroall` clear them.
//!
//! 256-bit instructions whose lanes are independent per 128-bit half (arithmetic, compares,
//! in-lane shuffles, integer ops, FMA, ...) are lifted as two 128-bit halves (per-lane immediate
//! bits such as `vshufpd`/`vblendps` are honoured); cross-lane ones (broadcasts, 128-bit
//! inserts/extracts, `vperm*`, widening/narrowing conversions, sign masks, tests, masked moves)
//! are lifted as a whole (`avx.rs`).
//!
//! # Supported instructions
//! Legacy-SSE, VEX-128 and (where they exist) VEX-256 forms of:
//! - moves: `movaps/movups/movapd/movupd/movdqa/movdqu/lddqu`, `movnt{ps,pd,dq,dqa,i}`,
//!   `movss/movsd`, `movd/movq`, `mov{l,h}{ps,pd}`, `movlhps/movhlps`, `movmskps/movmskpd/pmovmskb`,
//!   `movsldup/movshdup/movddup`, `vmaskmovps/pd`, `vpmaskmovd/q`;
//! - float arithmetic: `add/sub/mul/div/min/max/sqrt` `ss/sd/ps/pd`, `rcp/rsqrt` `ss/ps`,
//!   `and/andn/or/xor` `ps/pd`, `hadd/hsub/addsub` `ps/pd`, `round{ss,sd,ps,pd}`, `dpps/dppd`,
//!   FMA3 `vf{,n}m{add,sub}{132,213,231}{ps,pd,ss,sd}` and `vfm{addsub,subadd}*{ps,pd}`;
//! - compares: `cmp{ss,sd,ps,pd}` (8 legacy / 32 VEX predicates), `(u)comis{s,d}`, `ptest`,
//!   `vtestps/vtestpd`;
//! - conversions: `cvtsi2ss/sd`, `cvtss2sd/cvtsd2ss`, `cvt(t)ss2si/cvt(t)sd2si`,
//!   `cvtdq2ps/cvt(t)ps2dq/cvtdq2pd/cvt(t)pd2dq/cvtps2pd/cvtpd2ps`, F16C `vcvtph2ps/vcvtps2ph`
//!   (all rounding immediates);
//! - float shuffles: `shufps/shufpd`, `unpck{l,h}{ps,pd}`, `blend{ps,pd}`, `blendv{ps,pd}`,
//!   `insertps/extractps`, `vpermilps/vpermilpd` (immediate and variable), `vbroadcastss/sd/f128`,
//!   `vinsertf128/vextractf128`, `vperm2f128`, `vpermps/vpermpd`;
//! - integer: `padd/psub` `b/w/d/q`, `padds/paddus/psubs/psubus` `b/w`,
//!   `pmullw/pmulhw/pmulhuw/pmulhrsw/pmulld/pmuludq/pmuldq/pmaddwd/pmaddubsw`, `pand/pandn/por/
//!   pxor`, `pcmpeq/pcmpgt` `b/w/d/q`, `pmin/pmax` `s/u` `b/w/d`, `pavgb/w`, `psadbw`, `pabs*`,
//!   `psign*`, `psll/psrl/psra` (immediate and XMM count, any count), `pslldq/psrldq`,
//!   `vpsllv/vpsrlv/vpsrav`, `pshufd/pshuflw/pshufhw/pshufb`, `punpck*`, `packss*/packus*`,
//!   `palignr`, `pblendw/vpblendd/pblendvb`, `phadd*/phsub*`, `pextr*/pinsr*`,
//!   `pmovsx*/pmovzx*`, `vpbroadcast*`, `vbroadcasti128`, `vinserti128/vextracti128`,
//!   `vperm2i128`, `vpermd/vpermq`;
//! - SSE4.2 string compares `pcmp{i,e}str{i,m}` (every immediate, also VEX; `strcmp.rs`);
//! - AVX-512VL `vpmin/vpmax{s,u}q` on XMM/YMM registers without opmask or broadcast;
//! - AES-NI `aesenc/aesenclast/aesdec/aesdeclast/aesimc/aeskeygenassist` (also VAES 256-bit),
//!   `pclmulqdq` (also VPCLMULQDQ 256-bit), `phminposuw`;
//! - SSE4a `extrq/insertq` (immediate and register forms; the architecturally undefined upper
//!   quadword is preserved and field bits beyond bit 63 are dropped);
//! - `ldmxcsr/stmxcsr`, `vzeroupper/vzeroall`.
//!
//! Everything else that touches vector registers traps with a precise reason: AVX-512 (EVEX),
//! XOP/FMA4, MMX, x87 (not routed here), SHA, gathers, `mpsadbw`,
//! `maskmovdqu`.
//!
//! # Floating point semantics
//! Arithmetic, square root, conversions and rounding are IEEE-754 binary32/binary64 with
//! round-to-nearest-even, which is what x86 computes under the default MXCSR (`0x1F80`), so the
//! numerical results are bit-exact. Deviations:
//! - **NaN payloads**: when an arithmetic result is NaN, its payload and sign are not guaranteed
//!   (x86 returns the first NaN operand quieted, or the default `-QNaN`; LLVM may commute operands
//!   and wasm leaves NaN bits nondeterministic). Whether the result is NaN is always exact. Data
//!   movement, shuffles, blends, bitwise logic, `min`/`max` (compare + select) and F16C are
//!   bit-exact for NaNs too.
//! - **`min`/`max`** follow x86, not IEEE `minNum`: `minps a, b` is `a < b ? a : b` (the second
//!   operand when either is NaN or both are zero), `maxps a, b` is `a > b ? a : b`.
//! - **float -> int** conversions return the x86 "integer indefinite" value (`0x80000000` /
//!   `0x8000000000000000`) for NaN and out-of-range inputs, exactly like hardware; the rounding
//!   forms (`cvtss2si`, `cvtps2dq`, ...) round half to even, the `cvtt*` forms truncate.
//! - **FMA** (binary32) is computed with a single rounding without a hardware FMA: the product is
//!   exact in binary64, the sum is rounded to odd in binary64 (TwoSum error term) and then
//!   rounded to binary32, which is correctly rounded. Double-precision FMA uses an exact integer
//!   implementation (`__playgta6_fma_f64`: 106-bit product, aligned 128-bit sum with sticky
//!   bit, one rounding, subnormal and overflow handling).
//! - **`rcpss/rcpps/rsqrtss/rsqrtps`**: hardware returns an implementation-specific approximation
//!   (relative error <= 1.5 * 2^-12). The lifter returns the correctly rounded `1/x` and
//!   `1/sqrt(x)` (within the same bound of every conforming CPU) and reproduces the hardware's
//!   special cases: denormal inputs count as zero (result `+-inf`) and `rcp` of `|x| >= 2^126`
//!   (a result that is not a normal number) flushes to `+-0`.
//! - **MXCSR** is kept in the guest state ([`STATE_MXCSR_OFFSET`](super::super::STATE_MXCSR_OFFSET);
//!   the runtime initialises it to `0x1F80`): `ldmxcsr`/`stmxcsr` round-trip every accepted value.
//!   Lifted code always computes with round-to-nearest, no FTZ/DAZ and masked exceptions:
//!   - FTZ/DAZ bits are stored but not modelled: denormals are always computed per IEEE (games
//!     that enable FTZ/DAZ see different results on denormal operands/results only);
//!   - exception flags are never raised and unmasked exceptions never fault;
//!   - a value selecting another rounding mode (or setting reserved bits, which raises `#GP` on
//!     hardware) traps at run time through `__playgta6_unimplemented` instead of silently
//!     computing with the wrong rounding. `roundps`/`vcvtps2ph` immediates that select
//!     "MXCSR.RC" therefore round to nearest-even.
//! - **Alignment**: the aligned forms (`movaps`, `movdqa`, legacy-SSE memory operands, ...) do not
//!   raise `#GP` on misaligned addresses; all accesses are `align 1`. Masked moves never touch
//!   masked-out elements (like hardware, they cannot fault there).
//! - Non-temporal hints (`movnt*`) are plain loads/stores.

mod avx;
mod crypto;
mod float;
mod helpers;
mod int;
mod moves;
mod strcmp;

use std::fmt::Write as _;

use iced_x86::{
    Code, EncodingKind, Instruction, InstructionInfo, Mnemonic, OpAccess, OpKind, Register,
};

use super::super::Support;
use super::flag_liveness::{StateMask, xmm_state_bit, ymmh_state_bit};
use super::{ColdTrapKind, FunctionLifter, R, imm, unsupported};

/// The [`StateMask`] bit of `%state` vector field `field` (5: `xmm`, 6: `ymmh`) element `idx`.
fn vec_state_bit(field: u32, idx: usize) -> StateMask {
    if field == 5 {
        xmm_state_bit(idx)
    } else {
        ymmh_state_bit(idx)
    }
}

/// Canonical type of an XMM register value (allocas and guest state slots).
pub(super) const XMM_TYPE: &str = "<2 x i64>";

/// Lane element types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::lift) enum Elem {
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
}

impl Elem {
    fn bits(self) -> u32 {
        match self {
            Elem::I8 => 8,
            Elem::I16 => 16,
            Elem::I32 | Elem::F32 => 32,
            Elem::I64 | Elem::F64 => 64,
        }
    }

    /// Lanes in a 128-bit register.
    fn lanes(self) -> u32 {
        128 / self.bits()
    }

    fn scalar(self) -> &'static str {
        match self {
            Elem::I8 => "i8",
            Elem::I16 => "i16",
            Elem::I32 => "i32",
            Elem::I64 => "i64",
            Elem::F32 => "float",
            Elem::F64 => "double",
        }
    }

    /// The integer type of the same width.
    fn int(self) -> Elem {
        match self {
            Elem::F32 => Elem::I32,
            Elem::F64 => Elem::I64,
            e => e,
        }
    }

    fn of_bits(bits: u32) -> Elem {
        match bits {
            8 => Elem::I8,
            16 => Elem::I16,
            32 => Elem::I32,
            _ => Elem::I64,
        }
    }
}

/// An LLVM value type: a scalar (`n == 1`) or a vector of `n` lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::lift) struct Ty {
    e: Elem,
    n: u32,
}

impl Ty {
    /// A full 128-bit vector of `e`.
    fn v(e: Elem) -> Ty {
        Ty { e, n: e.lanes() }
    }

    fn s(e: Elem) -> Ty {
        Ty { e, n: 1 }
    }

    fn vn(e: Elem, n: u32) -> Ty {
        Ty { e, n }
    }

    fn ir(self) -> String {
        if self.n == 1 {
            self.e.scalar().to_string()
        } else {
            format!("<{} x {}>", self.n, self.e.scalar())
        }
    }

    /// Same shape with `i1` lanes (comparison results).
    fn bool_ir(self) -> String {
        if self.n == 1 {
            "i1".to_string()
        } else {
            format!("<{} x i1>", self.n)
        }
    }

    /// Same shape with integer lanes of the same width.
    fn int(self) -> Ty {
        Ty {
            e: self.e.int(),
            n: self.n,
        }
    }

    fn with(self, e: Elem) -> Ty {
        Ty { e, n: self.n }
    }

    /// `value` in every lane (`value` is a literal valid for the element type).
    fn splat(self, value: &str) -> String {
        if self.n == 1 {
            return value.to_string();
        }
        let lane = format!("{} {value}", self.e.scalar());
        let lanes: Vec<&str> = (0..self.n).map(|_| lane.as_str()).collect();
        format!("<{}>", lanes.join(", "))
    }

    /// LLVM intrinsic type suffix (`f32`, `v4f32`, `v16i8`, ...).
    fn suffix(self) -> String {
        let e = match self.e {
            Elem::F32 => "f32",
            Elem::F64 => "f64",
            Elem::I8 => "i8",
            Elem::I16 => "i16",
            Elem::I32 => "i32",
            Elem::I64 => "i64",
        };
        if self.n == 1 {
            e.to_string()
        } else {
            format!("v{}{e}", self.n)
        }
    }
}

/// A floating point literal (`v` must be exactly representable in the lane type).
fn fconst(v: f64) -> String {
    format!("0x{:016X}", v.to_bits())
}

/// `<k x i32> <i32 a, i32 b, ...>` shuffle mask.
fn mask(indices: &[u32]) -> String {
    let lanes: Vec<String> = indices.iter().map(|i| format!("i32 {i}")).collect();
    format!("<{} x i32> <{}>", indices.len(), lanes.join(", "))
}

/// Whether operands `a` and `b` name the exact same register.
fn same_reg(ins: &Instruction, a: u32, b: u32) -> bool {
    ins.op_kind(a) == OpKind::Register
        && ins.op_kind(b) == OpKind::Register
        && ins.op_register(a) == ins.op_register(b)
}

/// Whether a shuffle index list is the identity permutation `[0, 1, ..., n - 1]`.
fn is_identity_mask(idx: &[u32]) -> bool {
    idx.iter().enumerate().all(|(i, &k)| k == i as u32)
}

/// LLVM intrinsics (and wasm builtins) the SIMD lifter calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::lift) enum Intr {
    Sqrt,
    Rint,
    Floor,
    Ceil,
    Trunc,
    SAddSat,
    UAddSat,
    SSubSat,
    USubSat,
    SMin,
    SMax,
    UMin,
    UMax,
    /// `llvm.fptosi.sat.<dst>.<src>`: saturating float -> signed int conversion.
    FpToSiSat(Elem),
    /// `llvm.abs` with `is_int_min_poison = false` (`abs(MIN) == MIN`, like `pabs*`).
    Abs,
    /// `llvm.wasm.swizzle` (`i8x16.swizzle`): indices >= 16 select 0.
    Swizzle,
    /// `llvm.wasm.narrow.signed.<dst>.<src>` (`i8x16.narrow_i16x8_s` / `i16x8.narrow_i32x4_s`).
    NarrowSigned,
    /// `llvm.wasm.narrow.unsigned.<dst>.<src>` (`i8x16.narrow_i16x8_u` / `i16x8.narrow_i32x4_u`).
    NarrowUnsigned,
    /// `llvm.wasm.dot` (`i32x4.dot_i16x8_s`): pairwise signed `i16 * i16` dot product to `i32`.
    WasmDot,
    /// `llvm.wasm.avgr.unsigned.<ty>` (`i8x16.avgr_u` / `i16x8.avgr_u`).
    AvgrUnsigned,
    /// `llvm.wasm.bitmask.<ty>` (`i8x16`/`i32x4`/`i64x2.bitmask`): lane sign bits in low `i32`.
    Bitmask,
    /// `llvm.wasm.extadd.pairwise.unsigned.<dst>.<src>` (`i16x8`/`i32x4.extadd_pairwise_*_u`).
    ExtAddPairwiseUnsigned,
    /// `llvm.wasm.bitselect.<ty>` (`v128.bitselect`): `(a & c) | (b & ~c)` without 16B zero consts.
    Bitselect,
    /// `llvm.wasm.q15mulr.sat.signed` (`i16x8.q15mulr_sat_s`): saturating Q15 rounding multiply.
    Q15MulrSatSigned,
    /// Internal helper `@__playgta6_fma_f64` (double-precision FMA).
    FmaF64,
    /// Internal helper `@__playgta6_clmul` (carry-less multiplication).
    Clmul,
    /// The AES S-box tables.
    AesTables,
    /// Outlined `@__playgta6_aes_sub_bytes` helper (avoids 80-op inline expansion per AES round).
    AesSubBytes,
    /// Outlined `@__playgta6_aes_inv_sub_bytes` helper.
    AesInvSubBytes,
    /// Internal helper `@__playgta6_pcmpstr` (SSE4.2 string compare core).
    PcmpStr,
}

/// One module-level declaration required by lifted SIMD code (see `Support::Simd`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::lift) struct Decl {
    intr: Intr,
    ty: Ty,
}

impl Decl {
    fn name(self) -> String {
        let base = match self.intr {
            Intr::Sqrt => "sqrt",
            Intr::Rint => "rint",
            Intr::Floor => "floor",
            Intr::Ceil => "ceil",
            Intr::Trunc => "trunc",
            Intr::SAddSat => "sadd.sat",
            Intr::UAddSat => "uadd.sat",
            Intr::SSubSat => "ssub.sat",
            Intr::USubSat => "usub.sat",
            Intr::SMin => "smin",
            Intr::SMax => "smax",
            Intr::UMin => "umin",
            Intr::UMax => "umax",
            Intr::FpToSiSat(dst) => {
                return format!(
                    "llvm.fptosi.sat.{}.{}",
                    self.ty.with(dst).suffix(),
                    self.ty.suffix()
                );
            }
            Intr::Abs => "abs",
            Intr::Swizzle => return "llvm.wasm.swizzle".to_string(),
            Intr::NarrowSigned => {
                let dst = Ty::v(Elem::of_bits(self.ty.e.bits() / 2));
                return format!(
                    "llvm.wasm.narrow.signed.{}.{}",
                    dst.suffix(),
                    self.ty.suffix()
                );
            }
            Intr::NarrowUnsigned => {
                let dst = Ty::v(Elem::of_bits(self.ty.e.bits() / 2));
                return format!(
                    "llvm.wasm.narrow.unsigned.{}.{}",
                    dst.suffix(),
                    self.ty.suffix()
                );
            }
            Intr::WasmDot => return "llvm.wasm.dot".to_string(),
            Intr::AvgrUnsigned => {
                return format!("llvm.wasm.avgr.unsigned.{}", self.ty.suffix());
            }
            Intr::Bitmask => {
                return format!("llvm.wasm.bitmask.{}", self.ty.suffix());
            }
            Intr::ExtAddPairwiseUnsigned => {
                let dst = Ty::v(Elem::of_bits(self.ty.e.bits() * 2));
                return format!(
                    "llvm.wasm.extadd.pairwise.unsigned.{}.{}",
                    dst.suffix(),
                    self.ty.suffix()
                );
            }
            Intr::Bitselect => {
                return format!("llvm.wasm.bitselect.{}", self.ty.suffix());
            }
            Intr::Q15MulrSatSigned => return "llvm.wasm.q15mulr.sat.signed".to_string(),
            Intr::FmaF64 => return "__playgta6_fma_f64".to_string(),
            Intr::Clmul => return "__playgta6_clmul".to_string(),
            Intr::AesTables => return "__playgta6_aes_sbox".to_string(),
            Intr::AesSubBytes => return "__playgta6_aes_sub_bytes".to_string(),
            Intr::AesInvSubBytes => return "__playgta6_aes_inv_sub_bytes".to_string(),
            Intr::PcmpStr => return "__playgta6_pcmpstr".to_string(),
        };
        format!("llvm.{base}.{}", self.ty.suffix())
    }

    fn ret_ir(self) -> String {
        match self.intr {
            Intr::FpToSiSat(dst) => self.ty.with(dst).ir(),
            Intr::NarrowSigned | Intr::NarrowUnsigned => {
                Ty::v(Elem::of_bits(self.ty.e.bits() / 2)).ir()
            }
            Intr::WasmDot => Ty::v(Elem::I32).ir(),
            Intr::Bitmask => "i32".to_string(),
            Intr::ExtAddPairwiseUnsigned => Ty::v(Elem::of_bits(self.ty.e.bits() * 2)).ir(),
            _ => self.ty.ir(),
        }
    }

    fn params(self) -> String {
        let t = self.ty.ir();
        match self.intr {
            Intr::Sqrt
            | Intr::Rint
            | Intr::Floor
            | Intr::Ceil
            | Intr::Trunc
            | Intr::FpToSiSat(_)
            | Intr::Bitmask
            | Intr::ExtAddPairwiseUnsigned => t,
            Intr::Abs => format!("{t}, i1"),
            Intr::Bitselect => format!("{t}, {t}, {t}"),
            _ => format!("{t}, {t}"),
        }
    }

    /// The `declare` line (or, for internal helpers, the definition).
    pub(in crate::lift) fn declaration(self) -> String {
        match self.intr {
            Intr::FmaF64 => return helpers::fma_f64_ir(),
            Intr::Clmul => return helpers::CLMUL_IR.to_string(),
            Intr::AesTables => return helpers::aes_tables_ir(),
            Intr::AesSubBytes => return helpers::aes_sub_bytes_ir(false),
            Intr::AesInvSubBytes => return helpers::aes_sub_bytes_ir(true),
            Intr::PcmpStr => return helpers::PCMPSTR_IR.to_string(),
            _ => {}
        }
        format!(
            "declare {} @{}({})\n",
            self.ret_ir(),
            self.name(),
            self.params()
        )
    }
}

/// Operand positions of an SSE instruction: legacy encodings are destructive
/// (`dst = op0 = a`, `b = op1`), VEX encodings take three operands (`dst = op0, a = op1,
/// b = op2`).
#[derive(Debug, Clone, Copy)]
struct Form {
    dst: u32,
    a: u32,
    b: u32,
}

impl Form {
    fn of(vex: bool) -> Form {
        if vex {
            Form { dst: 0, a: 1, b: 2 }
        } else {
            Form { dst: 0, a: 0, b: 1 }
        }
    }
}

/// Whether the instruction is handled by the SIMD lifter: it has a vector (or MMX) register
/// operand, or it is one of the few SSE/AVX instructions without one.
pub(super) fn is_simd(ins: &Instruction) -> bool {
    use Mnemonic as M;
    let vector_operand = (0..ins.op_count()).any(|i| {
        ins.op_kind(i) == OpKind::Register && {
            let r = ins.op_register(i);
            r.is_xmm() || r.is_ymm() || r.is_zmm() || r.is_mm()
        }
    });
    vector_operand
        || matches!(
            ins.mnemonic(),
            M::Ldmxcsr
                | M::Vldmxcsr
                | M::Stmxcsr
                | M::Vstmxcsr
                | M::Vzeroupper
                | M::Vzeroall
                | M::Movnti
                | M::Emms
                // `r32/r64, m32/m64` forms have no vector operand.
                | M::Cvtss2si
                | M::Vcvtss2si
                | M::Cvttss2si
                | M::Vcvttss2si
                | M::Cvtsd2si
                | M::Vcvtsd2si
                | M::Cvttsd2si
                | M::Vcvttsd2si
        )
}

/// Vector registers a function uses, plus the SIMD lifter's per-instruction state.
#[derive(Debug, Default)]
pub(super) struct Regs {
    /// `xmm0..15` (bits 0..127 of `ymm0..15`) that get a slot.
    xmm: [bool; 16],
    /// Upper halves (bits 128..255) of `ymm0..15` that the function reads and therefore keep in a
    /// local `%ymmhN` alloca. Unread upper halves write directly to `%state.ymmhN`.
    ymmh: [bool; 16],
    /// The 128-bit lane the register/memory accessors address while a 256-bit instruction is
    /// lifted one lane at a time (0 = low, 1 = high).
    half: u32,
    /// Whether the current instruction is a 256-bit one lifted per 128-bit lane.
    wide: bool,
    /// R58: cached `(lo, hi)` `<2 x i64>` halves of a single 32-byte AVX memory load for the
    /// instruction currently being lifted, avoiding two separate 16-byte window-checked loads.
    mem32: Option<(String, String)>,
}

impl Regs {
    pub(super) fn clear_mem32(&mut self) {
        self.mem32 = None;
    }
}

fn is_read_access(access: OpAccess) -> bool {
    matches!(
        access,
        OpAccess::Read | OpAccess::CondRead | OpAccess::ReadWrite | OpAccess::ReadCondWrite
    )
}

/// Records the vector registers an instruction uses (`reg` and `access` from its used-register
/// info). Upper halves (`ymmh0..15`) only get a local alloca when read; write-only upper halves
/// (including VEX-128 zeroing) store directly to `%state.ymmhN`.
pub(super) fn note_register(reg: Register, access: OpAccess, regs: &mut Regs) {
    if (reg.is_xmm() || reg.is_ymm() || reg.is_zmm()) && reg.number() < 16 {
        regs.xmm[reg.number()] = true;
        if !reg.is_xmm() && is_read_access(access) {
            regs.ymmh[reg.number()] = true;
        }
    }
}

/// Records vector register halves an instruction uses beyond `used_registers`:
/// - `vzeroall` tracks all `xmm0..15` (its `ymmh0..15` clearing writes `%state.ymmh*` directly
///   when unread);
/// - legacy `blendvps`/`blendvpd`/`pblendvb` implicitly read `xmm0`;
/// - VEX-128 instructions with an XMM destination track `xmmN` (their upper-half zeroing writes
///   directly to `%state.ymmhN` when `ymmhN` is unread);
/// - explicit 256-bit `ymm` source operands (including zeroing idioms such as
///   `vxorps ymm0, ymm0, ymm0` where iced omits `Read` from `used_registers`) mark `ymmhN` as read
///   because the lifter reads source operands via `xget`/`xget_half`.
pub(super) fn note_instruction(ins: &Instruction, info: &InstructionInfo, regs: &mut Regs) {
    match ins.mnemonic() {
        Mnemonic::Vzeroall => {
            regs.xmm = [true; 16];
        }
        Mnemonic::Blendvps | Mnemonic::Blendvpd | Mnemonic::Pblendvb => {
            regs.xmm[0] = true;
        }
        Mnemonic::Pcmpistrm
        | Mnemonic::Pcmpestrm
        | Mnemonic::Pcmpestrm64
        | Mnemonic::Vpcmpistrm
        | Mnemonic::Vpcmpestrm
        | Mnemonic::Vpcmpestrm64 => {
            regs.xmm[0] = true;
        }
        _ => {}
    }
    if is_vex_encoded(ins)
        && ins.op_count() > 0
        && ins.op_kind(0) == OpKind::Register
        && ins.op_register(0).is_xmm()
        && ins.op_register(0).number() < 16
        && !reads_only_first_operand(ins.mnemonic())
    {
        regs.xmm[ins.op_register(0).number()] = true;
    }
    for i in 0..ins.op_count() {
        if ins.op_kind(i) == OpKind::Register {
            let r = ins.op_register(i);
            if r.is_ymm() && r.number() < 16 {
                regs.xmm[r.number()] = true;
                if i >= 1 || is_read_access(info.op_access(i)) {
                    regs.ymmh[r.number()] = true;
                }
            }
        }
    }
}

/// AVX-512VL 128/256-bit encodings that compute exactly what their VEX form does: the 64-bit
/// element `vpmin/vpmax` without an opmask, zeroing, broadcast, rounding or a register above 15
/// (those operands are rejected by [`FunctionLifter::check_encoding`]).
pub(super) fn is_plain_evex(ins: &Instruction) -> bool {
    use Mnemonic as M;
    ins.encoding() == EncodingKind::EVEX
        && matches!(
            ins.mnemonic(),
            M::Vpminsq | M::Vpminuq | M::Vpmaxsq | M::Vpmaxuq
        )
        && ins.op_mask() == Register::None
        && !ins.zeroing_masking()
        && !ins.is_broadcast()
        && !ins.suppress_all_exceptions()
        && ins.rounding_control() == iced_x86::RoundingControl::None
}

/// VEX encodings and the EVEX ones of [`is_plain_evex`].
pub(super) fn is_vex_encoded(ins: &Instruction) -> bool {
    ins.encoding() == EncodingKind::VEX || is_plain_evex(ins)
}

/// Whether a VEX instruction with an XMM first operand reads it without writing it.
pub(super) fn reads_only_first_operand(m: Mnemonic) -> bool {
    use Mnemonic as M;
    matches!(
        m,
        M::Vcomiss
            | M::Vucomiss
            | M::Vcomisd
            | M::Vucomisd
            | M::Vptest
            | M::Vtestps
            | M::Vtestpd
            | M::Vpcmpistri
            | M::Vpcmpistrm
            | M::Vpcmpestri
            | M::Vpcmpestri64
            | M::Vpcmpestrm
            | M::Vpcmpestrm64
    )
}

/// Whether `ins` is a vector constant-materialization idiom that the lifter lowers directly to
/// `zeroinitializer` or all-ones without reading any vector register alloca:
/// - `pxor`/`vpxor`/`xorps`/`vxorps`/`xorpd`/`vxorpd`/`pandn`/`vpandn`/`andnps`/`vandnps`/
///   `andnpd`/`vandnpd` with identical register sources -> `0`;
/// - `pcmpeq*`/`vpcmpeq*` with identical register sources -> `-1` (all-ones);
/// - `pcmpgt*`/`vpcmpgt*`/`psub*`/`vpsub*`/`psadbw`/`vpsadbw` with identical register sources -> `0`.
pub(super) fn is_const_idiom(ins: &Instruction) -> bool {
    use Mnemonic as M;
    let vex = is_vex_encoded(ins);
    let f = Form::of(vex);
    matches!(
        ins.mnemonic(),
        M::Pxor
            | M::Vpxor
            | M::Xorps
            | M::Vxorps
            | M::Xorpd
            | M::Vxorpd
            | M::Pandn
            | M::Vpandn
            | M::Andnps
            | M::Vandnps
            | M::Andnpd
            | M::Vandnpd
            | M::Pcmpeqb
            | M::Vpcmpeqb
            | M::Pcmpeqw
            | M::Vpcmpeqw
            | M::Pcmpeqd
            | M::Vpcmpeqd
            | M::Pcmpeqq
            | M::Vpcmpeqq
            | M::Pcmpgtb
            | M::Vpcmpgtb
            | M::Pcmpgtw
            | M::Vpcmpgtw
            | M::Pcmpgtd
            | M::Vpcmpgtd
            | M::Pcmpgtq
            | M::Vpcmpgtq
            | M::Psubb
            | M::Vpsubb
            | M::Psubw
            | M::Vpsubw
            | M::Psubd
            | M::Vpsubd
            | M::Psubq
            | M::Vpsubq
            | M::Psubsb
            | M::Vpsubsb
            | M::Psubsw
            | M::Vpsubsw
            | M::Psubusb
            | M::Vpsubusb
            | M::Psubusw
            | M::Vpsubusw
            | M::Psadbw
            | M::Vpsadbw
    ) && ins.op_count() > f.b
        && ins.op_kind(f.a) == OpKind::Register
        && ins.op_kind(f.b) == OpKind::Register
        && ins.op_register(f.a) == ins.op_register(f.b)
}

/// Whether a legacy-SSE (non-VEX) instruction with an XMM `op0` destination overwrites all 128
/// bits of `op0` without reading `op0`'s prior value (zeroing idioms are handled separately by
/// [`is_zeroing_idiom`]). Every partial-lane write (`movss`/`movsd` reg-reg, `movlps`/`movhps`/
/// `movlpd`/`movhpd`, `pinsr*`, `insertps`, `blend*`, `extrq`/`insertq`, etc.) returns `false`.
fn legacy_full_xmm_overwrite(ins: &Instruction) -> bool {
    use Mnemonic as M;
    match ins.mnemonic() {
        M::Movaps
        | M::Movups
        | M::Movapd
        | M::Movupd
        | M::Movdqa
        | M::Movdqu
        | M::Lddqu
        | M::Movntdqa
        | M::Movd
        | M::Movq
        | M::Movsldup
        | M::Movshdup
        | M::Movddup
        | M::Sqrtps
        | M::Sqrtpd
        | M::Rcpps
        | M::Rsqrtps
        | M::Cvtdq2ps
        | M::Cvtps2dq
        | M::Cvttps2dq
        | M::Cvtdq2pd
        | M::Cvtpd2dq
        | M::Cvttpd2dq
        | M::Cvtps2pd
        | M::Cvtpd2ps
        | M::Roundps
        | M::Roundpd
        | M::Pshufd
        | M::Pshuflw
        | M::Pshufhw
        | M::Pabsb
        | M::Pabsw
        | M::Pabsd
        | M::Pmovsxbw
        | M::Pmovsxbd
        | M::Pmovsxbq
        | M::Pmovsxwd
        | M::Pmovsxwq
        | M::Pmovsxdq
        | M::Pmovzxbw
        | M::Pmovzxbd
        | M::Pmovzxbq
        | M::Pmovzxwd
        | M::Pmovzxwq
        | M::Pmovzxdq
        | M::Aesimc
        | M::Aeskeygenassist
        | M::Phminposuw => true,
        // `movss/movsd xmm, m32/m64` zero-extends into all 128 bits (`zext_lanes`); reg-reg forms
        // preserve the upper lanes and MUST NOT be treated as a full overwrite.
        M::Movss | M::Movsd => ins.op_count() == 2 && ins.op_kind(1) == OpKind::Memory,
        _ => false,
    }
}

/// R64a: for a supported non-trapping SIMD instruction (`is_supported_non_trapping_simd(ins)`),
/// returns `(reads, full_writes)` bitmasks over `xmm0..15`:
/// - `reads`: superset of every `xmmN` whose `%xmmN` alloca `lift_simd` may load.
/// - `full_writes`: subset of every `xmmN` whose `%xmmN` alloca `lift_simd` unconditionally
///   overwrites (all 128 bits) without loading `%xmmN` first.
pub(super) fn xmm_rw_masks(ins: &Instruction, info: &InstructionInfo) -> (u16, u16) {
    use Mnemonic as M;
    match ins.mnemonic() {
        M::Vzeroall => return (0, 0xFFFF),
        M::Vzeroupper => return (0, 0),
        _ => {}
    }
    if is_const_idiom(ins)
        && ins.op_count() > 0
        && ins.op_kind(0) == OpKind::Register
    {
        let r0 = ins.op_register(0);
        if (r0.is_xmm() || r0.is_ymm()) && r0.number() < 16 {
            return (0, 1u16 << r0.number());
        }
    }
    let mut reads = 0u16;
    if matches!(ins.mnemonic(), M::Blendvps | M::Blendvpd | M::Pblendvb) {
        reads |= 1u16 << 0;
    }
    for ur in info.used_registers() {
        let r = ur.register();
        if (r.is_xmm() || r.is_ymm()) && r.number() < 16 && is_read_access(ur.access()) {
            reads |= 1u16 << r.number();
        }
    }
    // Conservatively mark every explicit source vector operand (1..op_count) as read, even for
    // dependency-breaking idioms that `iced_x86` omits from `used_registers`, unless
    // `is_const_idiom` short-circuits them above.
    for i in 1..ins.op_count() {
        if ins.op_kind(i) == OpKind::Register {
            let r = ins.op_register(i);
            if (r.is_xmm() || r.is_ymm()) && r.number() < 16 {
                reads |= 1u16 << r.number();
            }
        }
    }
    let mut full_writes = 0u16;
    if ins.op_count() > 0 && ins.op_kind(0) == OpKind::Register {
        let r0 = ins.op_register(0);
        if (r0.is_xmm() || r0.is_ymm()) && r0.number() < 16 {
            let bit = 1u16 << r0.number();
            let vex = is_vex_encoded(ins);
            let pure_full_write = info.op_access(0) == OpAccess::Write
                && !reads_only_first_operand(ins.mnemonic())
                && (vex || legacy_full_xmm_overwrite(ins));
            if pure_full_write && reads & bit == 0 {
                full_writes |= bit;
            } else {
                reads |= bit;
            }
        }
    }
    (reads, full_writes)
}

fn valid_mem_addr(ins: &Instruction) -> bool {
    let base = ins.memory_base();
    let index = ins.memory_index();
    let base_ok =
        matches!(base, Register::None | Register::RIP) || base.is_gpr64() || base.is_gpr32();
    let index_ok = index == Register::None || index.is_gpr64() || index.is_gpr32();
    base_ok && index_ok
}

/// Whether `ins` is a SIMD instruction that [`FunctionLifter::lift_simd`] is guaranteed to lift
/// without trapping (`cold_trap`, `trap_unsupported`, or any runtime helper call).
///
/// Used by `flag_liveness` to compute exact per-register `XMM0..15` / `YMMH0..15` call-boundary
/// summaries; any instruction returning `false` conservatively marks `ALL_STATE`.
pub(in crate::lift) fn is_supported_non_trapping_simd(ins: &Instruction) -> bool {
    use Mnemonic as M;
    if ins.code() == Code::INVALID || ins.has_lock_prefix() || ins.is_string_instruction() {
        return false;
    }
    if !is_simd(ins) {
        return false;
    }
    match ins.mnemonic() {
        // `ldmxcsr` traps at run time on non-default rounding modes / reserved bits.
        M::Ldmxcsr | M::Vldmxcsr | M::Emms => return false,
        M::Vzeroupper | M::Vzeroall => return true,
        M::Stmxcsr | M::Vstmxcsr => {
            return ins.op_count() == 1
                && ins.op_kind(0) == OpKind::Memory
                && ins.memory_size().size() == 4
                && valid_mem_addr(ins);
        }
        M::Movnti => {
            return ins.op_count() == 2
                && ins.op_kind(0) == OpKind::Memory
                && ins.op_kind(1) == OpKind::Register
                && valid_mem_addr(ins)
                && {
                    let r = ins.op_register(1);
                    r.is_gpr()
                        && r.full_register().is_gpr64()
                        && matches!(r.size(), 4 | 8)
                        && ins.memory_size().size() == r.size()
                };
        }
        _ => {}
    }
    if !(ins.encoding() == EncodingKind::Legacy || is_vex_encoded(ins)) {
        return false;
    }
    for i in 0..ins.op_count() {
        match ins.op_kind(i) {
            OpKind::Register => {
                let r = ins.op_register(i);
                let ok = (r.is_gpr() && r.full_register().is_gpr64())
                    || ((r.is_xmm() || r.is_ymm()) && r.number() < 16);
                if !ok {
                    return false;
                }
            }
            OpKind::Memory => {
                if ins.is_vsib() || ins.memory_size().size() > 32 || !valid_mem_addr(ins) {
                    return false;
                }
            }
            kind if super::is_immediate(kind) => {}
            _ => return false,
        }
    }
    let m = ins.mnemonic();
    if is_256(ins) {
        avx::is_avx_supported(ins) || avx::in_lane(m)
    } else {
        avx::is_avx_supported(ins)
            || moves::is_move_supported(m)
            || float::is_float_supported(m)
            || int::is_int_supported(m)
            || crypto::is_crypto_supported(m)
            || strcmp::is_strcmp_supported(m)
    }
}

/// Whether the instruction operates on 256-bit vectors (VEX.L = 1).
fn is_256(ins: &Instruction) -> bool {
    (0..ins.op_count()).any(|i| match ins.op_kind(i) {
        OpKind::Register => ins.op_register(i).is_ymm(),
        OpKind::Memory => ins.memory_size().size() == 32,
        _ => false,
    })
}

impl FunctionLifter<'_> {
    // ----- state ---------------------------------------------------------------------------------

    pub(super) fn xmm_allocas(&self, ir: &mut String) {
        for idx in 0..16 {
            if self.simd.xmm[idx] {
                let _ = writeln!(ir, "  %xmm{idx} = alloca {XMM_TYPE}");
            }
            if self.simd.ymmh[idx] {
                let _ = writeln!(ir, "  %ymmh{idx} = alloca {XMM_TYPE}");
            }
        }
    }

    /// Pointer to slot `idx` of state field 5 (`xmm`) or 6 (`ymm_hi`).
    fn state_vec_ptr(&mut self, field: u32, idx: usize) -> String {
        self.state_ptr(field, Some(idx))
    }

    /// The registers `entry:` loads from `%state` for `state` (see [`Self::sync_xmm_in`]).
    pub(super) fn vec_loaded(&self, state: StateMask) -> StateMask {
        self.vec_slots(state)
            .iter()
            .map(|&(field, idx, _)| vec_state_bit(field, idx))
            .fold(0, |a, b| a | b)
    }

    fn vec_slots(&self, state: StateMask) -> Vec<(u32, usize, String)> {
        let mut slots = Vec::new();
        for idx in 0..16 {
            if self.simd.xmm[idx] && state & xmm_state_bit(idx) != 0 {
                slots.push((5, idx, format!("%xmm{idx}")));
            }
            if self.simd.ymmh[idx] && state & ymmh_state_bit(idx) != 0 {
                slots.push((6, idx, format!("%ymmh{idx}")));
            }
        }
        slots
    }

    /// Loads the tracked vector registers present in `state` from `%state` into their allocas.
    pub(super) fn sync_xmm_in(&mut self, state: StateMask) {
        for (field, idx, slot) in self.vec_slots(state) {
            let ptr = self.state_vec_ptr(field, idx);
            let v = self.val(&format!("load {XMM_TYPE}, ptr {ptr}, align 8"));
            self.emit(&format!("store {XMM_TYPE} {v}, ptr {slot}"));
            self.note_synced(vec_state_bit(field, idx));
        }
    }

    /// Stores the tracked vector registers present in `state` back into `%state`.
    pub(super) fn sync_xmm_out(&mut self, state: StateMask) {
        for (field, idx, slot) in self.vec_slots(state) {
            let bit = vec_state_bit(field, idx);
            if self.is_synced(bit) {
                continue;
            }
            let v = self.val(&format!("load {XMM_TYPE}, ptr {slot}"));
            let ptr = self.state_vec_ptr(field, idx);
            self.emit(&format!("store {XMM_TYPE} {v}, ptr {ptr}, align 8"));
            self.note_synced(bit);
        }
    }

    // ----- registers and operands ----------------------------------------------------------------

    /// The alloca holding 128-bit lane `half` of the vector register `reg`.
    fn slot(&self, reg: Register, half: u32) -> R<String> {
        if !(reg.is_xmm() || reg.is_ymm()) {
            return unsupported(format!("operand {reg:?} is not an XMM/YMM register"));
        }
        let idx = reg.number();
        if idx >= 16 {
            return unsupported(format!(
                "register {reg:?} (only 16 vector registers exist without EVEX)"
            ));
        }
        let (tracked, name) = if half == 0 {
            (self.simd.xmm[idx], "xmm")
        } else {
            (self.simd.ymmh[idx], "ymmh")
        };
        if !tracked {
            return unsupported(format!("internal: {name}{idx} is not tracked"));
        }
        Ok(format!("%{name}{idx}"))
    }

    fn cast(&mut self, v: &str, from: Ty, to: Ty) -> String {
        if from == to {
            v.to_string()
        } else {
            self.val(&format!("bitcast {} {v} to {}", from.ir(), to.ir()))
        }
    }

    /// Lane `half` of a vector register as a 128-bit vector of `e`.
    fn xget_half(&mut self, reg: Register, half: u32, e: Elem) -> R<String> {
        let slot = self.slot(reg, half)?;
        let raw = self.val(&format!("load {XMM_TYPE}, ptr {slot}"));
        Ok(self.cast(&raw, Ty::v(Elem::I64), Ty::v(e)))
    }

    /// Writes lane `half` of a vector register.
    ///
    /// Upper halves (`half == 1`) that the function never reads have no local alloca and are
    /// written directly to `%state.ymmhN`, avoiding entry/call/exit spill and reload traffic.
    fn xset_half(&mut self, reg: Register, half: u32, v: &str, e: Elem) -> R<()> {
        if !(reg.is_xmm() || reg.is_ymm()) {
            return unsupported(format!("operand {reg:?} is not an XMM/YMM register"));
        }
        let idx = reg.number();
        if idx >= 16 {
            return unsupported(format!(
                "register {reg:?} (only 16 vector registers exist without EVEX)"
            ));
        }
        if half == 1 && !self.simd.ymmh[idx] {
            let bit = 1u16 << idx;
            if v == "zeroinitializer" {
                if self.ymmh_zeroed & bit != 0 {
                    return Ok(());
                }
                self.ymmh_zeroed |= bit;
                let ptr = self.state_vec_ptr(6, idx);
                self.emit(&format!(
                    "store {XMM_TYPE} zeroinitializer, ptr {ptr}, align 8"
                ));
                return Ok(());
            }
            self.ymmh_zeroed &= !bit;
            let raw = self.cast(v, Ty::v(e), Ty::v(Elem::I64));
            let ptr = self.state_vec_ptr(6, idx);
            self.emit(&format!("store {XMM_TYPE} {raw}, ptr {ptr}, align 8"));
            return Ok(());
        }
        let raw = self.cast(v, Ty::v(e), Ty::v(Elem::I64));
        let slot = self.slot(reg, half)?;
        self.emit(&format!("store {XMM_TYPE} {raw}, ptr {slot}"));
        Ok(())
    }

    /// Zeroes all 16 upper YMM halves (`ymm0..15` bits 128..255) for `vzeroupper` / `vzeroall`.
    ///
    /// When most or all upper halves are not locally allocated, clears the 256-byte `ymm_hi[16]`
    /// region in `%state` with a single `@llvm.memset.p0.i64` (lowering to one `memory.fill` on
    /// wasm64 with `+bulk-memory`) and zeroes any locally-tracked `%ymmhN` allocas.
    fn zero_all_ymmh(&mut self) -> R<()> {
        let local_count = self.simd.ymmh.iter().filter(|&&used| used).count();
        if local_count <= 13 {
            let mut unread_mask = 0u16;
            for idx in 0..16 {
                if !self.simd.ymmh[idx] {
                    unread_mask |= 1u16 << idx;
                }
            }
            if (self.ymmh_zeroed & unread_mask) != unread_mask {
                self.support.insert(Support::Memset);
                let ptr = self.state_vec_ptr(6, 0);
                self.emit(&format!(
                    "call void @llvm.memset.p0.i64(ptr align 8 {ptr}, i8 0, i64 256, i1 false)"
                ));
                self.ymmh_zeroed |= unread_mask;
            }
            for idx in 0..16u32 {
                if self.simd.ymmh[idx as usize] {
                    self.xset_half(Register::YMM0 + idx, 1, "zeroinitializer", Elem::I64)?;
                }
            }
            return Ok(());
        }
        for idx in 0..16u32 {
            self.xset_half(Register::YMM0 + idx, 1, "zeroinitializer", Elem::I64)?;
        }
        Ok(())
    }

    /// A vector register (the current 128-bit lane of a YMM register) as a vector of `e`.
    fn xget(&mut self, reg: Register, e: Elem) -> R<String> {
        if self.simd.half == 1 && !reg.is_ymm() {
            return unsupported("internal: upper lane of a 128-bit operand");
        }
        self.xget_half(reg, self.simd.half, e)
    }

    /// Writes a vector register (the current 128-bit lane of a YMM register).
    fn xset(&mut self, reg: Register, v: &str, e: Elem) -> R<()> {
        if self.simd.half == 1 && !reg.is_ymm() {
            return unsupported("internal: upper lane of a 128-bit operand");
        }
        self.xset_half(reg, self.simd.half, v, e)
    }

    fn mem_bytes(ins: &Instruction) -> usize {
        ins.memory_size().size()
    }

    /// Guest address of the memory operand plus `offset` bytes.
    fn mem_ea_at(&mut self, ins: &Instruction, offset: u64) -> R<String> {
        let ea = self.mem_ea(ins)?;
        Ok(self.offset_addr(&ea, offset))
    }

    /// R58: loads a 32-byte AVX memory operand once as `<4 x i64>`, caches both `<2 x i64>`
    /// halves for the current instruction, and returns lane `half` cast to `ty`.
    fn load_mem32_half(&mut self, ins: &Instruction, half: u32, ty: Ty) -> R<String> {
        let (lo, hi) = if let Some(pair) = &self.simd.mem32 {
            pair.clone()
        } else {
            let ea = self.mem_ea(ins)?;
            let full = self.guest_load(&ea, "<4 x i64>", 32);
            let lo = self.val(&format!(
                "shufflevector <4 x i64> {full}, <4 x i64> poison, <2 x i32> <i32 0, i32 1>"
            ));
            let hi = self.val(&format!(
                "shufflevector <4 x i64> {full}, <4 x i64> poison, <2 x i32> <i32 2, i32 3>"
            ));
            self.simd.mem32 = Some((lo.clone(), hi.clone()));
            (lo, hi)
        };
        let raw = if half == 0 { lo } else { hi };
        Ok(self.cast(&raw, Ty::v(Elem::I64), ty))
    }

    /// Byte offset of the current lane's part of the memory operand, checking its size: `bytes`
    /// for 128-bit instructions, twice that (one part per lane) for 256-bit ones.
    fn mem_part(&self, ins: &Instruction, bytes: usize) -> R<u64> {
        let total = if self.simd.wide { 2 * bytes } else { bytes };
        if Self::mem_bytes(ins) != total {
            return unsupported(format!(
                "internal: {}-byte memory operand where {total} bytes were expected",
                Self::mem_bytes(ins)
            ));
        }
        Ok(if self.simd.wide {
            u64::from(self.simd.half) * bytes as u64
        } else {
            0
        })
    }

    fn ty_bytes(ty: Ty) -> usize {
        (ty.e.bits() * ty.n / 8) as usize
    }

    /// Loads a value of type `ty` from the memory operand (the current lane's part of it).
    fn mem_load(&mut self, ins: &Instruction, ty: Ty) -> R<String> {
        let bytes = Self::ty_bytes(ty);
        let offset = self.mem_part(ins, bytes)?;
        if self.simd.wide && bytes == 16 && Self::mem_bytes(ins) == 32 {
            return self.load_mem32_half(ins, self.simd.half, ty);
        }
        let ea = self.mem_ea_at(ins, offset)?;
        Ok(self.guest_load(&ea, &ty.ir(), bytes as u64))
    }

    /// Stores a value of type `ty` to the memory operand (the current lane's part of it).
    fn mem_store(&mut self, ins: &Instruction, v: &str, ty: Ty) -> R<()> {
        let offset = self.mem_part(ins, Self::ty_bytes(ty))?;
        let ea = self.mem_ea_at(ins, offset)?;
        self.guest_store(&ea, &ty.ir(), v, Self::ty_bytes(ty) as u64);
        Ok(())
    }

    /// Loads `ty` at `offset` from a memory operand of exactly `total` bytes.
    fn mem_load_at(&mut self, ins: &Instruction, ty: Ty, offset: u64, total: usize) -> R<String> {
        if Self::mem_bytes(ins) != total {
            return unsupported(format!(
                "internal: {}-byte memory operand where {total} bytes were expected",
                Self::mem_bytes(ins)
            ));
        }
        let bytes = Self::ty_bytes(ty);
        if total == 32 && bytes == 16 && (offset == 0 || offset == 16) {
            return self.load_mem32_half(ins, (offset / 16) as u32, ty);
        }
        let ea = self.mem_ea_at(ins, offset)?;
        Ok(self.guest_load(&ea, &ty.ir(), bytes as u64))
    }

    /// Stores `ty` at `offset` into a memory operand of exactly `total` bytes.
    fn mem_store_at(
        &mut self,
        ins: &Instruction,
        v: &str,
        ty: Ty,
        offset: u64,
        total: usize,
    ) -> R<()> {
        if Self::mem_bytes(ins) != total {
            return unsupported(format!(
                "internal: {}-byte memory operand where {total} bytes were expected",
                Self::mem_bytes(ins)
            ));
        }
        let ea = self.mem_ea_at(ins, offset)?;
        self.guest_store(&ea, &ty.ir(), v, Self::ty_bytes(ty) as u64);
        Ok(())
    }

    /// A 128-bit operand (XMM register or `m128`, or the current lane of a YMM register or
    /// `m256`) as a vector of `e`.
    fn v128(&mut self, ins: &Instruction, i: u32, e: Elem) -> R<String> {
        match ins.op_kind(i) {
            OpKind::Register => self.xget(ins.op_register(i), e),
            OpKind::Memory => self.mem_load(ins, Ty::v(e)),
            kind => unsupported(format!("vector operand kind {kind:?}")),
        }
    }

    /// Writes a 128-bit result to a register or memory operand (current lane).
    fn set_v128(&mut self, ins: &Instruction, i: u32, v: &str, e: Elem) -> R<()> {
        match ins.op_kind(i) {
            OpKind::Register => self.xset(ins.op_register(i), v, e),
            OpKind::Memory => self.mem_store(ins, v, Ty::v(e)),
            kind => unsupported(format!("vector destination kind {kind:?}")),
        }
    }

    /// Lane `half` of a 128/256-bit register or memory operand.
    fn v_half(&mut self, ins: &Instruction, i: u32, half: u32, e: Elem) -> R<String> {
        match ins.op_kind(i) {
            OpKind::Register => self.xget_half(ins.op_register(i), half, e),
            OpKind::Memory => {
                let total = Self::mem_bytes(ins);
                if total != 16 && total != 32 {
                    return unsupported(format!("internal: {total}-byte vector memory operand"));
                }
                self.mem_load_at(ins, Ty::v(e), 16 * u64::from(half), total)
            }
            kind => unsupported(format!("vector operand kind {kind:?}")),
        }
    }

    /// The low `n` lanes of an XMM register, or a memory operand of that size, as `<n x e>`
    /// (a scalar when `n == 1`).
    fn low(&mut self, ins: &Instruction, i: u32, e: Elem, n: u32) -> R<String> {
        match ins.op_kind(i) {
            OpKind::Register => {
                let v = self.xget(ins.op_register(i), e)?;
                Ok(self.take_low(&v, e, n))
            }
            OpKind::Memory => self.mem_load(ins, Ty::vn(e, n)),
            kind => unsupported(format!("vector operand kind {kind:?}")),
        }
    }

    /// The low `n` lanes of a full vector of `e` (a scalar when `n == 1`).
    fn take_low(&mut self, v: &str, e: Elem, n: u32) -> String {
        let t = Ty::v(e).ir();
        if n == 1 {
            self.val(&format!("extractelement {t} {v}, i32 0"))
        } else if n == e.lanes() {
            v.to_string()
        } else {
            let idx: Vec<u32> = (0..n).collect();
            self.val(&format!(
                "shufflevector {t} {v}, {t} poison, {}",
                mask(&idx)
            ))
        }
    }

    /// Widens `<n x e>` (or a scalar) to a full vector, zeroing the upper lanes.
    fn zext_lanes(&mut self, v: &str, e: Elem, n: u32) -> String {
        let full = Ty::v(e);
        if n == 1 {
            return self.val(&format!(
                "insertelement {} zeroinitializer, {} {v}, i32 0",
                full.ir(),
                e.scalar()
            ));
        }
        if n == e.lanes() {
            return v.to_string();
        }
        let t = Ty::vn(e, n).ir();
        // Use consecutive indices `0..2*n` for the second (zero) operand so LLVM recognises
        // `concat_vectors(v, zero)` and emits native `f32x4.demote_f64x2_zero` /
        // `i32x4.trunc_sat_f64x2_s_zero`.
        let idx: Vec<u32> = (0..e.lanes())
            .map(|i| if i < 2 * n { i } else { n })
            .collect();
        self.val(&format!(
            "shufflevector {t} {v}, {t} zeroinitializer, {}",
            mask(&idx)
        ))
    }

    /// Expands four 32-bit lane indices (`<4 x i32>`, masked by `index_mask` such as `3` or `7`)
    /// into a `<16 x i8>` byte-index vector `[4*k, 4*k+1, 4*k+2, 4*k+3]` for `llvm.wasm.swizzle`.
    /// Multiplying `k <= 7` by `0x04040404` (`67372036`) computes `4*k` in all 4 bytes of each
    /// dword simultaneously with zero cross-byte carry (saves `shl 2` + `i8x16.shuffle`).
    fn dword_swizzle_indices(&mut self, idx: &str, index_mask: u32) -> String {
        let i32v = Ty::v(Elem::I32);
        let k = self.val(&format!(
            "and <4 x i32> {idx}, {}",
            i32v.splat(&index_mask.to_string())
        ));
        let rep32 = self.val(&format!("mul <4 x i32> {k}, {}", i32v.splat("67372036")));
        let rep = self.cast(&rep32, i32v, Ty::v(Elem::I8));
        self.val(&format!(
            "or <16 x i8> {rep}, <i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3>"
        ))
    }

    /// Replaces lane `lane` of a full vector of `e`.
    fn insert(&mut self, vec: &str, e: Elem, value: &str, lane: u32) -> String {
        self.val(&format!(
            "insertelement {} {vec}, {} {value}, i32 {lane}",
            Ty::v(e).ir(),
            e.scalar()
        ))
    }

    fn extract(&mut self, vec: &str, e: Elem, lane: u32) -> String {
        self.val(&format!(
            "extractelement {} {vec}, i32 {lane}",
            Ty::v(e).ir()
        ))
    }

    /// `shufflevector` of two full vectors of `e`.
    fn shuffle(&mut self, a: &str, b: &str, e: Elem, idx: &[u32]) -> String {
        let t = Ty::v(e).ir();
        self.val(&format!("shufflevector {t} {a}, {t} {b}, {}", mask(idx)))
    }

    /// The instruction's last operand as an 8-bit immediate.
    fn imm8(ins: &Instruction) -> R<u32> {
        let last = ins.op_count().saturating_sub(1);
        if !super::is_immediate(ins.op_kind(last)) {
            return unsupported("internal: missing imm8 operand");
        }
        match ins.try_immediate(last) {
            Ok(v) => Ok((v & 0xFF) as u32),
            Err(_) => unsupported("unreadable immediate"),
        }
    }

    /// The `per_lane` immediate bits that select within the current 128-bit lane (`vshufpd`,
    /// `vblendps`, ... use separate immediate bits for the upper lane).
    fn lane_imm(&self, imm8: u32, per_lane: u32) -> u32 {
        (imm8 >> (self.simd.half * per_lane)) & ((1 << per_lane) - 1)
    }

    /// Makes an internal helper (and what it needs) part of the module.
    fn use_helper(&mut self, intr: Intr) {
        self.support.insert(Support::Simd(Decl {
            intr,
            ty: Ty::s(Elem::I64),
        }));
        if intr == Intr::PcmpStr {
            self.support
                .insert(Support::BitIntrinsic(super::super::BitIntrinsic::Cttz, 32));
        }
        if intr == Intr::FmaF64 {
            self.support.insert(Support::Umul128);
            self.support
                .insert(Support::BitIntrinsic(super::super::BitIntrinsic::Ctlz, 64));
        }
        if matches!(intr, Intr::AesSubBytes | Intr::AesInvSubBytes) {
            self.support.insert(Support::Simd(Decl {
                intr: Intr::AesTables,
                ty: Ty::s(Elem::I64),
            }));
        }
    }

    /// Calls an intrinsic, declaring it at module level.
    fn intr(&mut self, intr: Intr, ty: Ty, args: &[&str]) -> String {
        let decl = Decl { intr, ty };
        self.support.insert(Support::Simd(decl));
        let t = ty.ir();
        let mut list: Vec<String> = args.iter().map(|a| format!("{t} {a}")).collect();
        if intr == Intr::Abs {
            list.push("i1 false".to_string());
        }
        self.val(&format!(
            "call {} @{}({})",
            decl.ret_ir(),
            decl.name(),
            list.join(", ")
        ))
    }

    // ----- dispatch ------------------------------------------------------------------------------

    /// Lifts an instruction for which [`is_simd`] holds.
    pub(super) fn lift_simd(&mut self, ins: &Instruction) -> R<()> {
        use Mnemonic as M;
        match ins.mnemonic() {
            M::Ldmxcsr | M::Vldmxcsr => return self.ldmxcsr(ins),
            M::Stmxcsr | M::Vstmxcsr => {
                let ptr = self.state_mxcsr_ptr();
                let v = self.val(&format!("load i32, ptr {ptr}"));
                return self.mem_store(ins, &v, Ty::s(Elem::I32));
            }
            M::Vzeroupper => return self.zero_all_ymmh(),
            M::Vzeroall => {
                for idx in 0..16u32 {
                    self.xset_half(Register::YMM0 + idx, 0, "zeroinitializer", Elem::I64)?;
                }
                return self.zero_all_ymmh();
            }
            M::Movnti => {
                let bits = self.op_bits(ins, 1)?;
                let v = self.read_op(ins, 1, bits)?;
                return self.write_op(ins, 0, &v, bits);
            }
            M::Emms => return unsupported("MMX instructions are not supported"),
            _ => {}
        }
        let vex = self.check_encoding(ins)?;
        self.simd.mem32 = None;
        let result = if is_256(ins) {
            self.lift_256(ins)
        } else {
            match self.lift_avx(ins) {
                Ok(true) => Ok(()),
                Ok(false) => self.lift_lane(ins, vex),
                Err(e) => Err(e),
            }
        };
        self.simd.half = 0;
        self.simd.wide = false;
        self.simd.mem32 = None;
        result?;
        // VEX encodings zero bits 128..255 of an XMM destination register.
        if vex
            && ins.op_kind(0) == OpKind::Register
            && ins.op_register(0).is_xmm()
            && !reads_only_first_operand(ins.mnemonic())
        {
            let ymm = Register::YMM0 + ins.op_register(0).number() as u32;
            self.xset_half(ymm, 1, "zeroinitializer", Elem::I64)?;
        }
        Ok(())
    }

    /// Lifts a 128-bit instruction (or, in a 256-bit loop, one lane of it).
    fn lift_lane(&mut self, ins: &Instruction, vex: bool) -> R<()> {
        if self.lift_moves(ins, vex)?
            || self.lift_float(ins, vex)?
            || self.lift_int(ins, vex)?
            || self.lift_crypto(ins, vex)?
            || self.lift_strcmp(ins, vex)?
        {
            return Ok(());
        }
        unsupported(format!(
            "instruction {:?} is not supported yet",
            ins.mnemonic()
        ))
    }

    /// 256-bit forms: cross-lane instructions are lifted as a whole, in-lane ones as two
    /// independent 128-bit halves.
    fn lift_256(&mut self, ins: &Instruction) -> R<()> {
        if self.lift_avx(ins)? {
            return Ok(());
        }
        // R58: 256-bit vector stores emit a single 32-byte guest store (1 window check, 1 dirty
        // mark) instead of two 16-byte halves.
        if ins.op_kind(0) == OpKind::Memory
            && Self::mem_bytes(ins) == 32
            && matches!(
                ins.mnemonic(),
                Mnemonic::Vmovaps
                    | Mnemonic::Vmovups
                    | Mnemonic::Vmovapd
                    | Mnemonic::Vmovupd
                    | Mnemonic::Vmovdqa
                    | Mnemonic::Vmovdqu
                    | Mnemonic::Vmovntps
                    | Mnemonic::Vmovntpd
                    | Mnemonic::Vmovntdq
            )
        {
            let lo = self.xget_half(ins.op_register(1), 0, Elem::I64)?;
            let hi = self.xget_half(ins.op_register(1), 1, Elem::I64)?;
            let val = self.val(&format!(
                "shufflevector <2 x i64> {lo}, <2 x i64> {hi}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>"
            ));
            let ea = self.mem_ea(ins)?;
            self.guest_store(&ea, "<4 x i64>", &val, 32);
            return Ok(());
        }
        if !avx::in_lane(ins.mnemonic()) {
            return unsupported(format!(
                "256-bit form of {:?} is not supported yet",
                ins.mnemonic()
            ));
        }
        self.simd.wide = true;
        for half in 0..2 {
            self.simd.half = half;
            self.lift_lane(ins, true)?;
        }
        Ok(())
    }

    /// Rejects encodings and operands outside the modelled SSE/AVX state; returns whether the
    /// instruction is VEX-encoded.
    fn check_encoding(&self, ins: &Instruction) -> R<bool> {
        let vex = match ins.encoding() {
            EncodingKind::Legacy => false,
            EncodingKind::VEX => true,
            EncodingKind::EVEX if is_plain_evex(ins) => true,
            EncodingKind::EVEX => {
                return unsupported("AVX-512 (EVEX) instructions are not supported");
            }
            other => {
                return unsupported(format!("{other:?}-encoded instructions are not supported"));
            }
        };
        for i in 0..ins.op_count() {
            match ins.op_kind(i) {
                OpKind::Register => {
                    let r = ins.op_register(i);
                    if r.is_mm() {
                        return unsupported("MMX instructions are not supported");
                    }
                    if r.is_zmm() || ((r.is_xmm() || r.is_ymm()) && r.number() >= 16) {
                        return unsupported(format!("register {r:?} requires EVEX"));
                    }
                }
                OpKind::Memory => {
                    if ins.is_vsib() {
                        return unsupported("gather (VSIB) instructions are not supported");
                    }
                    if Self::mem_bytes(ins) > 32 {
                        return unsupported("memory operand wider than 256 bits");
                    }
                }
                _ => {}
            }
        }
        Ok(vex)
    }

    fn state_mxcsr_ptr(&mut self) -> String {
        self.state_ptr(7, None)
    }

    /// `ldmxcsr m32`: stores the value in the guest state; only the default rounding mode is
    /// supported (see the module docs).
    fn ldmxcsr(&mut self, ins: &Instruction) -> R<()> {
        let v = self.mem_load(ins, Ty::s(Elem::I32))?;
        // Rounding control (bits 13-14) and the reserved bits 16-31 must be clear.
        let bad = self.val(&format!("and i32 {v}, {}", imm(0xFFFF_6000, 32)));
        let fault = self.val(&format!("icmp ne i32 {bad}, 0"));
        self.cold_trap_branch(&fault, ColdTrapKind::Unimplemented, ins.ip());
        let ptr = self.state_mxcsr_ptr();
        self.emit(&format!("store i32 {v}, ptr {ptr}"));
        Ok(())
    }
}
