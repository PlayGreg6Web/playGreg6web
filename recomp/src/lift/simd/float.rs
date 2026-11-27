//! Scalar and packed floating point: arithmetic, `min`/`max`, square root, reciprocal
//! approximations, bitwise logic, compares, conversions, rounding, horizontal ops and dot
//! products. See the parent module for the exactness guarantees and deviations.

use iced_x86::{Instruction, Mnemonic, OpKind};

use super::super::{Flag, FunctionLifter, R, imm, unsupported};
use super::{Elem, Form, Intr, Ty, fconst, same_reg};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    Sqrt,
    Rcp,
    Rsqrt,
}

/// `(operation, lane type, scalar form)` of the arithmetic instructions.
fn arith(m: Mnemonic) -> Option<(FOp, Elem, bool)> {
    use Elem::{F32, F64};
    use FOp::*;
    use Mnemonic as M;
    Some(match m {
        M::Addps | M::Vaddps => (Add, F32, false),
        M::Addpd | M::Vaddpd => (Add, F64, false),
        M::Addss | M::Vaddss => (Add, F32, true),
        M::Addsd | M::Vaddsd => (Add, F64, true),
        M::Subps | M::Vsubps => (Sub, F32, false),
        M::Subpd | M::Vsubpd => (Sub, F64, false),
        M::Subss | M::Vsubss => (Sub, F32, true),
        M::Subsd | M::Vsubsd => (Sub, F64, true),
        M::Mulps | M::Vmulps => (Mul, F32, false),
        M::Mulpd | M::Vmulpd => (Mul, F64, false),
        M::Mulss | M::Vmulss => (Mul, F32, true),
        M::Mulsd | M::Vmulsd => (Mul, F64, true),
        M::Divps | M::Vdivps => (Div, F32, false),
        M::Divpd | M::Vdivpd => (Div, F64, false),
        M::Divss | M::Vdivss => (Div, F32, true),
        M::Divsd | M::Vdivsd => (Div, F64, true),
        M::Minps | M::Vminps => (Min, F32, false),
        M::Minpd | M::Vminpd => (Min, F64, false),
        M::Minss | M::Vminss => (Min, F32, true),
        M::Minsd | M::Vminsd => (Min, F64, true),
        M::Maxps | M::Vmaxps => (Max, F32, false),
        M::Maxpd | M::Vmaxpd => (Max, F64, false),
        M::Maxss | M::Vmaxss => (Max, F32, true),
        M::Maxsd | M::Vmaxsd => (Max, F64, true),
        M::Sqrtps | M::Vsqrtps => (Sqrt, F32, false),
        M::Sqrtpd | M::Vsqrtpd => (Sqrt, F64, false),
        M::Sqrtss | M::Vsqrtss => (Sqrt, F32, true),
        M::Sqrtsd | M::Vsqrtsd => (Sqrt, F64, true),
        M::Rcpps | M::Vrcpps => (Rcp, F32, false),
        M::Rcpss | M::Vrcpss => (Rcp, F32, true),
        M::Rsqrtps | M::Vrsqrtps => (Rsqrt, F32, false),
        M::Rsqrtss | M::Vrsqrtss => (Rsqrt, F32, true),
        _ => return None,
    })
}

/// Operand order of an FMA3 instruction (`132`, `213`, `231`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FmaOrder {
    O132,
    O213,
    O231,
}

/// Signs of an FMA3 instruction: `-(x*y)` and `-z` per lane parity, plus lane type and width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Fma {
    order: FmaOrder,
    /// Negate the product.
    neg_product: bool,
    /// Negate the addend in even / odd lanes (`fmaddsub`: even subtract, odd add).
    neg_addend: [bool; 2],
    e: Elem,
    scalar: bool,
}

fn fma(m: Mnemonic) -> Option<Fma> {
    use FmaOrder::*;
    let name = format!("{m:?}");
    let rest = name.strip_prefix("Vf")?;
    let (kind, rest) = ["maddsub", "msubadd", "nmadd", "nmsub", "madd", "msub"]
        .iter()
        .find_map(|k| rest.strip_prefix(k).map(|r| (*k, r)))?;
    let (order, suffix) = if let Some(r) = rest.strip_prefix("132") {
        (O132, r)
    } else if let Some(r) = rest.strip_prefix("213") {
        (O213, r)
    } else {
        (O231, rest.strip_prefix("231")?)
    };
    let (e, scalar) = match suffix {
        "ps" => (Elem::F32, false),
        "pd" => (Elem::F64, false),
        "ss" => (Elem::F32, true),
        "sd" => (Elem::F64, true),
        _ => return None,
    };
    let (neg_product, neg_addend) = match kind {
        "madd" => (false, [false, false]),
        "msub" => (false, [true, true]),
        "nmadd" => (true, [false, false]),
        "nmsub" => (true, [true, true]),
        "maddsub" => (false, [true, false]),
        _ => (false, [false, true]), // msubadd
    };
    Some(Fma {
        order,
        neg_product,
        neg_addend,
        e,
        scalar,
    })
}

/// LLVM `fcmp` predicate for an SSE/AVX comparison predicate (`imm8[3:0]`; bit 4 only selects
/// the signalling variant, which makes no difference without exception flags).
fn predicate(p: u32) -> &'static str {
    match p & 0xF {
        0 => "oeq",
        1 => "olt",
        2 => "ole",
        3 => "uno",
        4 => "une",
        5 => "uge",
        6 => "ugt",
        7 => "ord",
        8 => "ueq",
        9 => "ult",
        10 => "ule",
        11 => "false",
        12 => "one",
        13 => "oge",
        14 => "ogt",
        _ => "true",
    }
}

/// Whether [`FunctionLifter::lift_float`] handles `m` (mirroring its dispatch).
pub(super) fn is_float_supported(m: Mnemonic) -> bool {
    use Mnemonic as M;
    if arith(m).is_some() || fma(m).is_some() {
        return true;
    }
    matches!(
        m,
        M::Andps
            | M::Vandps
            | M::Andpd
            | M::Vandpd
            | M::Andnps
            | M::Vandnps
            | M::Andnpd
            | M::Vandnpd
            | M::Orps
            | M::Vorps
            | M::Orpd
            | M::Vorpd
            | M::Xorps
            | M::Vxorps
            | M::Xorpd
            | M::Vxorpd
            | M::Cmpps
            | M::Vcmpps
            | M::Cmppd
            | M::Vcmppd
            | M::Cmpss
            | M::Vcmpss
            | M::Cmpsd
            | M::Vcmpsd
            | M::Comiss
            | M::Vcomiss
            | M::Ucomiss
            | M::Vucomiss
            | M::Comisd
            | M::Vcomisd
            | M::Ucomisd
            | M::Vucomisd
            | M::Cvtsi2ss
            | M::Vcvtsi2ss
            | M::Cvtsi2sd
            | M::Vcvtsi2sd
            | M::Cvtss2sd
            | M::Vcvtss2sd
            | M::Cvtsd2ss
            | M::Vcvtsd2ss
            | M::Cvtss2si
            | M::Vcvtss2si
            | M::Cvttss2si
            | M::Vcvttss2si
            | M::Cvtsd2si
            | M::Vcvtsd2si
            | M::Cvttsd2si
            | M::Vcvttsd2si
            | M::Cvtdq2ps
            | M::Vcvtdq2ps
            | M::Cvtps2dq
            | M::Vcvtps2dq
            | M::Cvttps2dq
            | M::Vcvttps2dq
            | M::Cvtdq2pd
            | M::Vcvtdq2pd
            | M::Cvtpd2dq
            | M::Vcvtpd2dq
            | M::Cvttpd2dq
            | M::Vcvttpd2dq
            | M::Cvtps2pd
            | M::Vcvtps2pd
            | M::Cvtpd2ps
            | M::Vcvtpd2ps
            | M::Haddps
            | M::Vhaddps
            | M::Haddpd
            | M::Vhaddpd
            | M::Hsubps
            | M::Vhsubps
            | M::Hsubpd
            | M::Vhsubpd
            | M::Addsubps
            | M::Vaddsubps
            | M::Addsubpd
            | M::Vaddsubpd
            | M::Roundps
            | M::Vroundps
            | M::Roundpd
            | M::Vroundpd
            | M::Roundss
            | M::Vroundss
            | M::Roundsd
            | M::Vroundsd
            | M::Dpps
            | M::Vdpps
            | M::Dppd
            | M::Vdppd
    )
}

impl FunctionLifter<'_> {
    pub(super) fn lift_float(&mut self, ins: &Instruction, vex: bool) -> R<bool> {
        use Elem::{F32, F64, I32};
        use Mnemonic as M;
        let f = Form::of(vex);
        let m = ins.mnemonic();
        if let Some((op, e, scalar)) = arith(m) {
            self.arith(ins, f, op, e, scalar)?;
            return Ok(true);
        }
        if let Some(fma) = fma(m) {
            self.fma(ins, fma)?;
            return Ok(true);
        }
        match m {
            M::Andps | M::Vandps | M::Andpd | M::Vandpd => self.logic(ins, f, "and", false)?,
            M::Andnps | M::Vandnps | M::Andnpd | M::Vandnpd => self.logic(ins, f, "and", true)?,
            M::Orps | M::Vorps | M::Orpd | M::Vorpd => self.logic(ins, f, "or", false)?,
            M::Xorps | M::Vxorps | M::Xorpd | M::Vxorpd => self.logic(ins, f, "xor", false)?,
            M::Cmpps | M::Vcmpps => self.compare(ins, f, vex, F32, false)?,
            M::Cmppd | M::Vcmppd => self.compare(ins, f, vex, F64, false)?,
            M::Cmpss | M::Vcmpss => self.compare(ins, f, vex, F32, true)?,
            M::Cmpsd | M::Vcmpsd => self.compare(ins, f, vex, F64, true)?,
            M::Comiss | M::Vcomiss | M::Ucomiss | M::Vucomiss => self.comis(ins, F32)?,
            M::Comisd | M::Vcomisd | M::Ucomisd | M::Vucomisd => self.comis(ins, F64)?,
            M::Cvtsi2ss | M::Vcvtsi2ss => self.cvt_int_to_scalar(ins, f, F32)?,
            M::Cvtsi2sd | M::Vcvtsi2sd => self.cvt_int_to_scalar(ins, f, F64)?,
            M::Cvtss2sd | M::Vcvtss2sd => {
                let a = self.v128(ins, f.a, F64)?;
                let s = self.low(ins, f.b, F32, 1)?;
                let d = self.val(&format!("fpext float {s} to double"));
                let r = self.insert(&a, F64, &d, 0);
                self.xset(ins.op_register(f.dst), &r, F64)?;
            }
            M::Cvtsd2ss | M::Vcvtsd2ss => {
                let a = self.v128(ins, f.a, F32)?;
                let s = self.low(ins, f.b, F64, 1)?;
                let d = self.val(&format!("fptrunc double {s} to float"));
                let r = self.insert(&a, F32, &d, 0);
                self.xset(ins.op_register(f.dst), &r, F32)?;
            }
            M::Cvtss2si | M::Vcvtss2si => self.cvt_scalar_to_int(ins, F32, false)?,
            M::Cvttss2si | M::Vcvttss2si => self.cvt_scalar_to_int(ins, F32, true)?,
            M::Cvtsd2si | M::Vcvtsd2si => self.cvt_scalar_to_int(ins, F64, false)?,
            M::Cvttsd2si | M::Vcvttsd2si => self.cvt_scalar_to_int(ins, F64, true)?,
            M::Cvtdq2ps | M::Vcvtdq2ps => {
                let v = self.v128(ins, 1, I32)?;
                let r = if v == "zeroinitializer" {
                    "zeroinitializer".to_string()
                } else {
                    self.val(&format!("sitofp <4 x i32> {v} to <4 x float>"))
                };
                self.xset(ins.op_register(0), &r, F32)?;
            }
            M::Cvtps2dq | M::Vcvtps2dq | M::Cvttps2dq | M::Vcvttps2dq => {
                let truncate = matches!(m, M::Cvttps2dq | M::Vcvttps2dq);
                let v = self.v128(ins, 1, F32)?;
                let r = self.float_to_int(&v, Ty::v(F32), 32, truncate);
                self.xset(ins.op_register(0), &r, I32)?;
            }
            M::Cvtdq2pd | M::Vcvtdq2pd => {
                let v = self.low(ins, 1, I32, 2)?;
                let r = if v == "zeroinitializer" {
                    "zeroinitializer".to_string()
                } else {
                    self.val(&format!("sitofp <2 x i32> {v} to <2 x double>"))
                };
                self.xset(ins.op_register(0), &r, F64)?;
            }
            M::Cvtpd2dq | M::Vcvtpd2dq | M::Cvttpd2dq | M::Vcvttpd2dq => {
                // Zero-extend the `<2 x i32>` `fptosi.sat` result to `<4 x i32>` and blend via
                // `@llvm.wasm.bitselect.v4i32` so LLVM emits native `i32x4.trunc_sat_f64x2_s_zero`
                // without legalizing `<4 x i1>` masks into `shl 31; shr_s 31`.
                let truncate = matches!(m, M::Cvttpd2dq | M::Vcvttpd2dq);
                let v = self.v128(ins, 1, F64)?;
                let x = if truncate {
                    v
                } else {
                    self.intr(Intr::Rint, Ty::v(F64), &[&v])
                };
                let sat2 = self.intr(Intr::FpToSiSat(I32), Ty::v(F64), &[&x]);
                let sat4 = self.zext_lanes(&sat2, I32, 2);
                let limit = Ty::v(F64).splat(&fconst(2f64.powi(31)));
                let ok2 = self.val(&format!("fcmp olt <2 x double> {x}, {limit}"));
                let ok64 = self.val(&format!("sext <2 x i1> {ok2} to <2 x i64>"));
                let ok32 = self.cast(&ok64, Ty::v(Elem::I64), Ty::v(I32));
                let mask = self.shuffle(&ok32, "poison", I32, &[0, 2, 0, 2]);
                let r = self.intr(
                    Intr::Bitselect,
                    Ty::v(I32),
                    &[
                        &sat4,
                        "<i32 -2147483648, i32 -2147483648, i32 0, i32 0>",
                        &mask,
                    ],
                );
                self.xset(ins.op_register(0), &r, I32)?;
            }
            M::Cvtps2pd | M::Vcvtps2pd => {
                let v = self.low(ins, 1, F32, 2)?;
                let r = self.val(&format!("fpext <2 x float> {v} to <2 x double>"));
                self.xset(ins.op_register(0), &r, F64)?;
            }
            M::Cvtpd2ps | M::Vcvtpd2ps => {
                let v = self.v128(ins, 1, F64)?;
                let r = self.val(&format!("fptrunc <2 x double> {v} to <2 x float>"));
                let r = self.zext_lanes(&r, F32, 2);
                self.xset(ins.op_register(0), &r, F32)?;
            }
            M::Haddps | M::Vhaddps => self.horizontal(ins, f, F32, "fadd")?,
            M::Haddpd | M::Vhaddpd => self.horizontal(ins, f, F64, "fadd")?,
            M::Hsubps | M::Vhsubps => self.horizontal(ins, f, F32, "fsub")?,
            M::Hsubpd | M::Vhsubpd => self.horizontal(ins, f, F64, "fsub")?,
            M::Addsubps | M::Vaddsubps | M::Addsubpd | M::Vaddsubpd => {
                let e = if matches!(m, M::Addsubps | M::Vaddsubps) {
                    F32
                } else {
                    F64
                };
                let t = Ty::v(e).ir();
                let a = self.v128(ins, f.a, e)?;
                let b = if same_reg(ins, f.a, f.b) {
                    a.clone()
                } else {
                    self.v128(ins, f.b, e)?
                };
                // Even lanes subtract, odd lanes add: in IEEE-754 round-to-nearest-even,
                // `a - b` is bit-exact with `a + (-b)` (including signed zeros, inf, NaN),
                // lowering to a single sign-bit `v128.xor` + `fadd` with zero shuffles.
                let nb = self.neg_alt_lanes(&b, e, true);
                let r = self.val(&format!("fadd {t} {a}, {nb}"));
                self.xset(ins.op_register(f.dst), &r, e)?;
            }
            M::Roundps | M::Vroundps => self.round(ins, f, F32, false)?,
            M::Roundpd | M::Vroundpd => self.round(ins, f, F64, false)?,
            M::Roundss | M::Vroundss => self.round(ins, f, F32, true)?,
            M::Roundsd | M::Vroundsd => self.round(ins, f, F64, true)?,
            M::Dpps | M::Vdpps => self.dot(ins, f, F32)?,
            M::Dppd | M::Vdppd => self.dot(ins, f, F64)?,
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn arith(&mut self, ins: &Instruction, f: Form, op: FOp, e: Elem, scalar: bool) -> R<()> {
        let unary = matches!(op, FOp::Sqrt | FOp::Rcp | FOp::Rsqrt);
        // x86 `min(a, a) === a` and `max(a, a) === a` for EVERY bit pattern (normals, +-0.0, inf,
        // and all NaNs) because `a < a` / `a > a` is always false, selecting the second operand `a`.
        if matches!(op, FOp::Min | FOp::Max) && same_reg(ins, f.a, f.b) {
            let a = self.v128(ins, f.a, e)?;
            return self.xset(ins.op_register(f.dst), &a, e);
        }
        if scalar {
            // Lane 0 from (a op b) or op(b); lanes 1.. from a.
            let a = self.v128(ins, f.a, e)?;
            let b = self.low(ins, f.b, e, 1)?;
            let r = if unary {
                self.fop1(op, &b, Ty::s(e))
            } else {
                let a0 = self.extract(&a, e, 0);
                self.fop2(op, &a0, &b, Ty::s(e))
            };
            let v = self.insert(&a, e, &r, 0);
            self.xset(ins.op_register(f.dst), &v, e)
        } else if unary {
            // sqrtps xmm1, xmm2/m128 (both encodings)
            let src = ins.op_count() - 1;
            let b = self.v128(ins, src, e)?;
            let r = self.fop1(op, &b, Ty::v(e));
            self.xset(ins.op_register(0), &r, e)
        } else {
            let a = self.v128(ins, f.a, e)?;
            let b = self.v128(ins, f.b, e)?;
            let r = self.fop2(op, &a, &b, Ty::v(e));
            self.xset(ins.op_register(f.dst), &r, e)
        }
    }

    fn fop2(&mut self, op: FOp, a: &str, b: &str, ty: Ty) -> String {
        let t = ty.ir();
        let opcode = match op {
            FOp::Add => "fadd",
            FOp::Sub => "fsub",
            FOp::Mul => "fmul",
            FOp::Div => "fdiv",
            FOp::Min | FOp::Max => {
                // x86: the second operand unless the comparison holds (NaNs, +-0 -> b).
                let pred = if op == FOp::Min { "olt" } else { "ogt" };
                let c = self.val(&format!("fcmp {pred} {t} {a}, {b}"));
                return self.val(&format!("select {} {c}, {t} {a}, {t} {b}", ty.bool_ir()));
            }
            _ => "fadd",
        };
        self.val(&format!("{opcode} {t} {a}, {b}"))
    }

    fn fop1(&mut self, op: FOp, x: &str, ty: Ty) -> String {
        let t = ty.ir();
        match op {
            FOp::Sqrt => self.intr(Intr::Sqrt, ty, &[x]),
            FOp::Rcp | FOp::Rsqrt => {
                // Correctly rounded 1/x or 1/sqrt(x) plus the hardware's special cases.
                let one = ty.splat(&fconst(1.0));
                let d = if op == FOp::Rcp {
                    x.to_string()
                } else {
                    self.intr(Intr::Sqrt, ty, &[x])
                };
                let q = self.val(&format!("fdiv {t} {one}, {d}"));
                let it = ty.int();
                let ti = it.ir();
                let bits = self.val(&format!("bitcast {t} {x} to {ti}"));
                let sign = self.val(&format!(
                    "and {ti} {bits}, {}",
                    it.splat(&imm(0x8000_0000, 32))
                ));
                let mag = self.val(&format!("xor {ti} {bits}, {sign}"));
                // Denormal inputs count as zero: +-inf.
                let den = self.val(&format!("icmp ult {ti} {mag}, {}", it.splat("8388608")));
                let inf = self.val(&format!("or {ti} {sign}, {}", it.splat("2139095040")));
                let inf = self.val(&format!("bitcast {ti} {inf} to {t}"));
                let r = self.val(&format!(
                    "select {} {den}, {t} {inf}, {t} {q}",
                    ty.bool_ir()
                ));
                if op == FOp::Rsqrt {
                    return r;
                }
                // rcp results that are not normal (|x| >= 2^126, finite) flush to +-0:
                // 0x7E80_0000 <= mag < 0x7F80_0000 <=> (mag - 0x7E80_0000) <_u 0x0100_0000.
                let off = self.val(&format!("sub {ti} {mag}, {}", it.splat("2122317824")));
                let tiny = self.val(&format!("icmp ult {ti} {off}, {}", it.splat("16777216")));
                let zero = self.val(&format!("bitcast {ti} {sign} to {t}"));
                self.val(&format!(
                    "select {} {tiny}, {t} {zero}, {t} {r}",
                    ty.bool_ir()
                ))
            }
            _ => x.to_string(),
        }
    }

    /// Bitwise logic on 128 bits (`andn` complements the first source).
    pub(super) fn logic(&mut self, ins: &Instruction, f: Form, op: &str, not_a: bool) -> R<()> {
        if same_reg(ins, f.a, f.b) {
            if (op == "xor" && !not_a) || (op == "and" && not_a) {
                return self.xset(ins.op_register(f.dst), "zeroinitializer", Elem::I64);
            }
            let a = self.v128(ins, f.a, Elem::I64)?;
            return self.xset(ins.op_register(f.dst), &a, Elem::I64);
        }
        let t = Ty::v(Elem::I64).ir();
        let mut a = self.v128(ins, f.a, Elem::I64)?;
        let b = self.v128(ins, f.b, Elem::I64)?;
        if a == "zeroinitializer" || b == "zeroinitializer" {
            let r = match (op, not_a) {
                ("and", false) => "zeroinitializer".to_string(),
                ("and", true) => {
                    if b == "zeroinitializer" {
                        "zeroinitializer".to_string()
                    } else {
                        b
                    }
                }
                _ => {
                    if a == "zeroinitializer" {
                        b
                    } else {
                        a
                    }
                }
            };
            return self.xset(ins.op_register(f.dst), &r, Elem::I64);
        }
        if not_a {
            a = self.val(&format!("xor {t} {a}, <i64 -1, i64 -1>"));
        }
        let r = self.val(&format!("{op} {t} {a}, {b}"));
        self.xset(ins.op_register(f.dst), &r, Elem::I64)
    }

    /// `cmpps/cmppd/cmpss/cmpsd`: all-ones lanes where the predicate holds.
    fn compare(&mut self, ins: &Instruction, f: Form, vex: bool, e: Elem, scalar: bool) -> R<()> {
        let i = Self::imm8(ins)?;
        // Legacy SSE encodings use imm8[2:0]; VEX uses imm8[4:0].
        let pred = predicate(if vex { i & 0x1F } else { i & 7 });
        let ie = e.int();
        let same = same_reg(ins, f.a, f.b);
        let const_result = match pred {
            "false" => Some(false),
            "true" => Some(true),
            "olt" | "ogt" | "one" if same => Some(false),
            "ueq" | "uge" | "ule" if same => Some(true),
            _ => None,
        };
        if scalar {
            let a = self.v128(ins, f.a, e)?;
            let b = self.low(ins, f.b, e, 1)?;
            let m = match const_result {
                Some(false) => "0".to_string(),
                Some(true) => "-1".to_string(),
                None => {
                    let a0 = self.extract(&a, e, 0);
                    let c = self.val(&format!("fcmp {pred} {} {a0}, {b}", e.scalar()));
                    self.val(&format!("sext i1 {c} to {}", ie.scalar()))
                }
            };
            let ai = self.cast(&a, Ty::v(e), Ty::v(ie));
            let r = self.insert(&ai, ie, &m, 0);
            self.xset(ins.op_register(f.dst), &r, ie)
        } else {
            let ty = Ty::v(e);
            if let Some(res) = const_result {
                if ins.op_kind(f.b) == OpKind::Memory {
                    let _ = self.v128(ins, f.b, e)?;
                }
                let m = if res {
                    ty.int().splat("-1")
                } else {
                    "zeroinitializer".to_string()
                };
                return self.xset(ins.op_register(f.dst), &m, ie);
            }
            let a = self.v128(ins, f.a, e)?;
            let b = if same {
                a.clone()
            } else {
                self.v128(ins, f.b, e)?
            };
            let c = self.val(&format!("fcmp {pred} {} {a}, {b}", ty.ir()));
            let m = self.val(&format!("sext {} {c} to {}", ty.bool_ir(), ty.int().ir()));
            self.xset(ins.op_register(f.dst), &m, ie)
        }
    }

    /// `comiss/ucomiss/comisd/ucomisd`: ZF/PF/CF = 111 unordered, 000 greater, 001 less,
    /// 100 equal; OF, SF and AF are cleared. Guards each `fcmp` with `flag_live` and installs
    /// `FlagSource::FloatCompare` so same-block `jcc`/`setcc`/`cmovcc` fuse to a single `fcmp`.
    fn comis(&mut self, ins: &Instruction, e: Elem) -> R<()> {
        let t = e.scalar();
        let a = self.low(ins, 0, e, 1)?;
        let b = self.low(ins, 1, e, 1)?;
        if self.flag_live(Flag::Zf) {
            let zf = self.val(&format!("fcmp ueq {t} {a}, {b}"));
            self.set_flag(Flag::Zf, &zf);
        }
        if self.flag_live(Flag::Pf) {
            let pf = self.val(&format!("fcmp uno {t} {a}, {b}"));
            self.set_flag(Flag::Pf, &pf);
        }
        if self.flag_live(Flag::Cf) {
            let cf = self.val(&format!("fcmp ult {t} {a}, {b}"));
            self.set_flag(Flag::Cf, &cf);
        }
        self.set_flag(Flag::Of, "false");
        self.set_flag(Flag::Sf, "false");
        self.set_flag(Flag::Af, "false");
        self.known.flag_source = Some(super::super::FlagSource::FloatCompare {
            a,
            b,
            scalar_ty: t,
        });
        Ok(())
    }

    /// `cvtsi2ss/cvtsi2sd xmm, r/m32|64`: lane 0 = (float) int, other lanes from the first source.
    fn cvt_int_to_scalar(&mut self, ins: &Instruction, f: Form, e: Elem) -> R<()> {
        let bits = self.op_bits(ins, f.b)?;
        if bits != 32 && bits != 64 {
            return unsupported("cvtsi2ss/sd source size");
        }
        let v = self.read_op(ins, f.b, bits)?;
        let c = self.val(&format!("sitofp i{bits} {v} to {}", e.scalar()));
        let a = self.v128(ins, f.a, e)?;
        let r = self.insert(&a, e, &c, 0);
        self.xset(ins.op_register(f.dst), &r, e)
    }

    /// `cvt(t)ss2si/cvt(t)sd2si r32/r64, xmm/m`.
    fn cvt_scalar_to_int(&mut self, ins: &Instruction, e: Elem, truncate: bool) -> R<()> {
        if ins.op_kind(0) != OpKind::Register {
            return unsupported("conversion destination");
        }
        let bits = self.op_bits(ins, 0)?;
        if bits != 32 && bits != 64 {
            return unsupported("conversion destination size");
        }
        let s = self.low(ins, 1, e, 1)?;
        let r = self.float_to_int(&s, Ty::s(e), bits, truncate);
        self.write_op(ins, 0, &r, bits)
    }

    /// x86 float -> signed int conversion: round half to even (or truncate), and the "integer
    /// indefinite" value `1 << (bits - 1)` for NaN and out-of-range inputs.
    /// `llvm.fptosi.sat` already saturates `x <= -2^(bits-1)` (including `-inf`) to the exact
    /// indefinite value `MIN_INT`, so a single ordered comparison `x < 2^(bits-1)` catches both
    /// positive overflow and NaN.
    pub(super) fn float_to_int(&mut self, v: &str, ty: Ty, bits: u32, truncate: bool) -> String {
        let t = ty.ir();
        let x = if truncate {
            v.to_string()
        } else {
            self.intr(Intr::Rint, ty, &[v])
        };
        let dst_elem = Elem::of_bits(bits);
        let sat = self.intr(Intr::FpToSiSat(dst_elem), ty, &[&x]);
        let limit = 2f64.powi(bits as i32 - 1);
        let ok = self.val(&format!("fcmp olt {t} {x}, {}", ty.splat(&fconst(limit))));
        let it = ty.with(dst_elem);
        let indefinite = it.splat(&imm(1u64 << (bits - 1), bits));
        self.val(&format!(
            "select {} {ok}, {} {sat}, {} {indefinite}",
            ty.bool_ir(),
            it.ir(),
            it.ir()
        ))
    }

    /// Flips the sign bit of even lanes (`neg_even == true`) or odd lanes (`neg_even == false`)
    /// using a single bitwise `xor` on the integer view (1 `v128.xor` on Wasm SIMD128).
    fn neg_alt_lanes(&mut self, v: &str, e: Elem, neg_even: bool) -> String {
        let ty = Ty::v(e);
        let ity = ty.int();
        let mask = match (e, neg_even) {
            (Elem::F32, true) => "<i32 -2147483648, i32 0, i32 -2147483648, i32 0>",
            (Elem::F32, false) => "<i32 0, i32 -2147483648, i32 0, i32 -2147483648>",
            (Elem::F64, true) => "<i64 -9223372036854775808, i64 0>",
            (Elem::F64, false) => "<i64 0, i64 -9223372036854775808>",
            _ => unreachable!(),
        };
        let iv = self.cast(v, ty, ity);
        let flipped = self.val(&format!("xor {} {iv}, {mask}", ity.ir()));
        self.cast(&flipped, ity, ty)
    }

    /// `haddps/hsubps/haddpd/hsubpd`: pairwise op of adjacent lanes of a, then of b.
    fn horizontal(&mut self, ins: &Instruction, f: Form, e: Elem, op: &str) -> R<()> {
        let a = self.v128(ins, f.a, e)?;
        let n = e.lanes();
        let (lo, hi) = if same_reg(ins, f.a, f.b) {
            let even: Vec<u32> = (0..n).map(|l| (2 * l) % n).collect();
            let odd: Vec<u32> = (0..n).map(|l| (2 * l + 1) % n).collect();
            (
                self.shuffle(&a, "poison", e, &even),
                self.shuffle(&a, "poison", e, &odd),
            )
        } else {
            let b = self.v128(ins, f.b, e)?;
            let even: Vec<u32> = (0..n).map(|l| 2 * l).collect();
            let odd: Vec<u32> = (0..n).map(|l| 2 * l + 1).collect();
            (
                self.shuffle(&a, &b, e, &even),
                self.shuffle(&a, &b, e, &odd),
            )
        };
        let r = self.val(&format!("{op} {} {lo}, {hi}", Ty::v(e).ir()));
        self.xset(ins.op_register(f.dst), &r, e)
    }

    /// `roundps/roundpd/roundss/roundsd`: imm8[1:0] = nearest-even, floor, ceil, truncate;
    /// imm8[2] selects MXCSR.RC, which is always round-to-nearest-even here.
    fn round(&mut self, ins: &Instruction, f: Form, e: Elem, scalar: bool) -> R<()> {
        let i = Self::imm8(ins)?;
        let intr = if i & 4 != 0 {
            Intr::Rint
        } else {
            [Intr::Rint, Intr::Floor, Intr::Ceil, Intr::Trunc][(i & 3) as usize]
        };
        if scalar {
            let a = self.v128(ins, f.a, e)?;
            let s = self.low(ins, f.b, e, 1)?;
            let r = self.intr(intr, Ty::s(e), &[&s]);
            let v = self.insert(&a, e, &r, 0);
            self.xset(ins.op_register(f.dst), &v, e)
        } else {
            let v = self.v128(ins, 1, e)?;
            let r = self.intr(intr, Ty::v(e), &[&v]);
            self.xset(ins.op_register(0), &r, e)
        }
    }

    /// `dpps/dppd`: products of the lanes selected by imm8[7:4] (others +0.0), summed in the
    /// hardware's order, broadcast to the lanes selected by imm8[3:0] (others +0.0).
    /// Specialises 1-, 2-, and 3-lane masks for minimal Wasm SIMD128 shuffles while preserving
    /// exact IEEE-754 `-0.0` sign rules and NaN isolation of masked-out lanes.
    fn dot(&mut self, ins: &Instruction, f: Form, e: Elem) -> R<()> {
        let i = Self::imm8(ins)?;
        let a = self.v128(ins, f.a, e)?;
        let b = if same_reg(ins, f.a, f.b) {
            a.clone()
        } else {
            self.v128(ins, f.b, e)?
        };
        let n = e.lanes();
        let all = (1u32 << n) - 1;
        let in_mask = (i >> 4) & all;
        let out_mask = i & all;
        if in_mask == 0 || out_mask == 0 {
            return self.xset(ins.op_register(f.dst), "zeroinitializer", e);
        }
        let vt = Ty::v(e).ir();
        let prod = self.val(&format!("fmul {vt} {a}, {b}"));
        let emit_out = |l: &mut Self, vec: &str, lane: u32, broadcast: bool| -> String {
            if out_mask == all {
                if broadcast {
                    vec.to_string()
                } else {
                    let idx: Vec<u32> = (0..n).map(|_| lane).collect();
                    l.shuffle(vec, "poison", e, &idx)
                }
            } else {
                let out_idx: Vec<u32> = (0..n)
                    .map(|k| if (out_mask >> k) & 1 == 1 { lane } else { n })
                    .collect();
                l.shuffle(vec, "zeroinitializer", e, &out_idx)
            }
        };
        let active = in_mask.count_ones();
        let r = if active == 1 {
            // Single active lane: adding `+0.0` normalises `-0.0` to `+0.0` (matching the
            // masked-out `+0.0` lanes in hardware) without any cross-lane shuffle.
            let lane = in_mask.trailing_zeros();
            let norm = self.val(&format!("fadd {vt} {prod}, zeroinitializer"));
            emit_out(self, &norm, lane, false)
        } else if n == 4 && active == 2 {
            // Two active lanes `l0 < l1`: splat `l1`, add into `l0`, then add `+0.0` to match
            // the hardware's masked-out `+0.0` lanes (5 ops vs 7).
            let l0 = in_mask.trailing_zeros();
            let l1 = (in_mask ^ (1 << l0)).trailing_zeros();
            let p1 = self.shuffle(&prod, "poison", e, &[l1, l1, l1, l1]);
            let s1 = self.val(&format!("fadd {vt} {prod}, {p1}"));
            let s2 = self.val(&format!("fadd {vt} {s1}, zeroinitializer"));
            emit_out(self, &s2, l0, false)
        } else if n == 4 && active == 3 && out_mask != all {
            // Three active lanes with partial output mask (e.g. `dpps 0x71`): combine the
            // input masking with the first pairwise shuffle so lane `pair_dst` gets the pair
            // sum and `lone` gets `prod[lone] + 0.0` in one step (saves 1 shuffle).
            let (pair_dst, lone, p1_idx) = match in_mask {
                0x7 => (0u32, 2u32, [1u32, 4, 4, 4]),
                0xB => (0, 3, [1, 4, 4, 4]),
                0xD => (2, 0, [4, 4, 3, 4]),
                _ => (2, 1, [4, 4, 3, 4]), // 0xE
            };
            let p1 = self.shuffle(&prod, "zeroinitializer", e, &p1_idx);
            let s1 = self.val(&format!("fadd {vt} {prod}, {p1}"));
            let p2 = self.shuffle(&s1, "poison", e, &[lone, lone, lone, lone]);
            let s2 = self.val(&format!("fadd {vt} {s1}, {p2}"));
            emit_out(self, &s2, pair_dst, false)
        } else {
            let masked = if in_mask == all {
                prod
            } else {
                let idx: Vec<u32> = (0..n)
                    .map(|l| if (in_mask >> l) & 1 == 1 { l } else { n })
                    .collect();
                self.shuffle(&prod, "zeroinitializer", e, &idx)
            };
            let sum_vec = if n == 4 {
                let p1 = self.shuffle(&masked, "poison", e, &[1, 0, 3, 2]);
                let s1 = self.val(&format!("fadd {vt} {masked}, {p1}"));
                let p2 = self.shuffle(&s1, "poison", e, &[2, 3, 0, 1]);
                self.val(&format!("fadd {vt} {s1}, {p2}"))
            } else {
                let p1 = self.shuffle(&masked, "poison", e, &[1, 0]);
                self.val(&format!("fadd {vt} {masked}, {p1}"))
            };
            emit_out(self, &sum_vec, 0, true)
        };
        self.xset(ins.op_register(f.dst), &r, e)
    }

    /// FMA3: `dst = (+-x) * y + (+-z)` with a single rounding. Operands: 132 `x=op0, y=op2,
    /// z=op1`; 213 `x=op1, y=op0, z=op2`; 231 `x=op1, y=op2, z=op0`. Scalar forms merge lane 0
    /// into the destination.
    fn fma(&mut self, ins: &Instruction, f: Fma) -> R<()> {
        let e = f.e;
        let n = if f.scalar { 1 } else { e.lanes() };
        let ty = if f.scalar { Ty::s(e) } else { Ty::v(e) };
        let mut ops: Vec<String> = Vec::with_capacity(3);
        let dst_vec = if f.scalar {
            // Load op0's full 128-bit register once; extract lane 0 for the FMA and reuse the
            // same vector to insert the scalar result back into lane 0.
            let d = self.v128(ins, 0, e)?;
            ops.push(self.extract(&d, e, 0));
            Some(d)
        } else {
            ops.push(self.v128(ins, 0, e)?);
            None
        };
        for i in 1..3 {
            let v = if same_reg(ins, i, 0) {
                ops[0].clone()
            } else if i == 2 && same_reg(ins, 2, 1) {
                ops[1].clone()
            } else if f.scalar {
                self.low(ins, i, e, 1)?
            } else {
                self.v128(ins, i, e)?
            };
            ops.push(v);
        }
        let (x, y, z) = match f.order {
            FmaOrder::O132 => (&ops[0], &ops[2], &ops[1]),
            FmaOrder::O213 => (&ops[1], &ops[0], &ops[2]),
            FmaOrder::O231 => (&ops[1], &ops[2], &ops[0]),
        };
        let t = ty.ir();
        let x = if f.neg_product {
            self.val(&format!("fneg {t} {x}"))
        } else {
            x.clone()
        };
        let z = match f.neg_addend {
            [false, false] => z.clone(),
            [true, true] => self.val(&format!("fneg {t} {z}")),
            [neg_even, _] => self.neg_alt_lanes(z, e, neg_even),
        };
        let r = if e == Elem::F32 {
            self.fma32(&x, y, &z, n)
        } else {
            self.fma64(&x, y, &z, n)
        };
        if let Some(d) = dst_vec {
            let v = self.insert(&d, e, &r, 0);
            self.xset(ins.op_register(0), &v, e)
        } else {
            self.xset(ins.op_register(0), &r, e)
        }
    }

    /// Correctly rounded binary64 `x * y + z` through the integer helper `__playgta6_fma_f64`.
    fn fma64(&mut self, x: &str, y: &str, z: &str, n: u32) -> String {
        self.use_helper(Intr::FmaF64);
        let call = |l: &mut Self, a: &str, b: &str, c: &str| {
            l.val(&format!(
                "call double @__playgta6_fma_f64(double {a}, double {b}, double {c})"
            ))
        };
        if n == 1 {
            return call(self, x, y, z);
        }
        let mut r = "poison".to_string();
        for lane in 0..n {
            let a = self.extract(x, Elem::F64, lane);
            let b = self.extract(y, Elem::F64, lane);
            let c = self.extract(z, Elem::F64, lane);
            let v = call(self, &a, &b, &c);
            r = self.insert(&r, Elem::F64, &v, lane);
        }
        r
    }

    /// Correctly rounded binary32 `x * y + z` without a hardware FMA: the product is exact in
    /// binary64, the sum is rounded to odd in binary64 (TwoSum error term), and the final
    /// rounding to binary32 is then correct (53 >= 2 * 24 + 2).
    ///
    /// For `<4 x float>`, splits into two native 128-bit `<2 x double>` halves so LLVM emits
    /// `f64x2.promote_low_f32x4` / `f32x4.demote_f64x2_zero` directly without 256-bit
    /// legalization.
    fn fma32(&mut self, x: &str, y: &str, z: &str, n: u32) -> String {
        if n == 4 {
            let mut halves = Vec::with_capacity(2);
            for half_mask in ["<2 x i32> <i32 0, i32 1>", "<2 x i32> <i32 2, i32 3>"] {
                let xh = self.val(&format!(
                    "shufflevector <4 x float> {x}, <4 x float> poison, {half_mask}"
                ));
                let yh = self.val(&format!(
                    "shufflevector <4 x float> {y}, <4 x float> poison, {half_mask}"
                ));
                let zh = self.val(&format!(
                    "shufflevector <4 x float> {z}, <4 x float> poison, {half_mask}"
                ));
                let h = self.fma32_core(&xh, &yh, &zh, 2);
                // Zero-extend `<2 x float>` to `<4 x float>` and combine via swapped `or` on
                // `<2 x i64>` so both zero upper halves stay demanded and LLVM emits native
                // `f32x4.demote_f64x2_zero` instead of scalar `f64x2.extract_lane` + `f32.demote_f64`.
                let z = self.zext_lanes(&h, Elem::F32, 2);
                halves.push(self.cast(&z, Ty::v(Elem::F32), Ty::v(Elem::I64)));
            }
            let hi_swap = self.shuffle(&halves[1], "poison", Elem::I64, &[1, 0]);
            let combined = self.val(&format!("or <2 x i64> {}, {hi_swap}", halves[0]));
            return self.cast(&combined, Ty::v(Elem::I64), Ty::v(Elem::F32));
        }
        self.fma32_core(x, y, z, n)
    }

    /// Scalar (`n == 1`) or native 128-bit pair (`n == 2`) binary32 FMA via binary64 TwoSum
    /// and branchless round-to-odd (zero `fcmp one`, zero double `select` chains, single `0.0`
    /// constant).
    fn fma32_core(&mut self, x: &str, y: &str, z: &str, n: u32) -> String {
        let (ft, dt, it, bt) = if n == 1 {
            ("float", "double", "i64", "i1")
        } else {
            ("<2 x float>", "<2 x double>", "<2 x i64>", "<2 x i1>")
        };
        let zero_d = if n == 1 {
            fconst(0.0)
        } else {
            "zeroinitializer".to_string()
        };
        let zero_i = if n == 1 { "0" } else { "zeroinitializer" };
        let sh63 = if n == 1 { "63" } else { "<i64 63, i64 63>" };

        let a = self.val(&format!("fpext {ft} {x} to {dt}"));
        let b = self.val(&format!("fpext {ft} {y} to {dt}"));
        let c = self.val(&format!("fpext {ft} {z} to {dt}"));
        let p = self.val(&format!("fmul {dt} {a}, {b}"));
        let s = self.val(&format!("fadd {dt} {p}, {c}"));
        // TwoSum: err = (p - (s - bb)) + (c - bb), bb = s - p.
        let bb = self.val(&format!("fsub {dt} {s}, {p}"));
        let t1 = self.val(&format!("fsub {dt} {s}, {bb}"));
        let e1 = self.val(&format!("fsub {dt} {p}, {t1}"));
        let e2 = self.val(&format!("fsub {dt} {c}, {bb}"));
        let err = self.val(&format!("fadd {dt} {e1}, {e2}"));
        // Sign of TwoSum error: d in {-1, 0, +1} (0 when err == 0 or NaN).
        let gt = self.val(&format!("fcmp ogt {dt} {err}, {zero_d}"));
        let lt = self.val(&format!("fcmp olt {dt} {err}, {zero_d}"));
        let d = if n == 1 {
            let gti = self.val(&format!("zext i1 {gt} to i64"));
            let lti = self.val(&format!("zext i1 {lt} to i64"));
            self.val(&format!("sub i64 {gti}, {lti}"))
        } else {
            // Vector comparisons produce all-ones (-1) for true, so `sext` is free on SIMD128.
            let gti = self.val(&format!("sext {bt} {gt} to {it}"));
            let lti = self.val(&format!("sext {bt} {lt} to {it}"));
            self.val(&format!("sub {it} {lti}, {gti}"))
        };
        // Step to the odd neighbour on the side of the exact value:
        // negate `d` when `s < 0`, and mask to 0 when `bits` is already odd.
        let bits = self.val(&format!("bitcast {dt} {s} to {it}"));
        let sneg_b = self.val(&format!("icmp slt {it} {bits}, {zero_i}"));
        let sneg = self.val(&format!("sext {bt} {sneg_b} to {it}"));
        let dx = self.val(&format!("xor {it} {d}, {sneg}"));
        let ds = self.val(&format!("sub {it} {dx}, {sneg}"));
        let lsb_top = self.val(&format!("shl {it} {bits}, {sh63}"));
        let odd = if n == 2 {
            // Avoid InstCombine turning `icmp eq (shl bits, 63), 0` back into `v128.const 0` +
            // `v128.const 1` + `i64x2.eq`: `ashr(shl(bits, 63), 63)` is all-ones iff `bits` is odd.
            let stepped = self.val(&format!("add {it} {bits}, {ds}"));
            let odd_m = self.val(&format!("ashr {it} {lsb_top}, {sh63}"));
            self.intr(Intr::Bitselect, Ty::v(Elem::I64), &[&bits, &stepped, &odd_m])
        } else {
            let even_b = self.val(&format!("icmp eq {it} {lsb_top}, {zero_i}"));
            let even = self.val(&format!("sext {bt} {even_b} to {it}"));
            let step = self.val(&format!("and {it} {ds}, {even}"));
            self.val(&format!("add {it} {bits}, {step}"))
        };
        let odd = self.val(&format!("bitcast {it} {odd} to {dt}"));
        self.val(&format!("fptrunc {dt} {odd} to {ft}"))
    }
}
