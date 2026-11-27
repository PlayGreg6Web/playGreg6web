//! Integer SIMD (SSE2 .. SSE4.2, AVX2): lane arithmetic, saturation, multiplies, compares,
//! min/max, shifts, byte shifts, shuffles, packs, unpacks, blends, horizontal ops, sums of
//! absolute differences, element inserts/extracts. Every operation is exact (no float involved).
//! In 256-bit (AVX2) forms all of these act on each 128-bit lane independently.

use iced_x86::{Instruction, Mnemonic, OpKind};

use super::super::{FunctionLifter, R, unsupported};
use super::{Elem, Form, Intr, Ty, is_identity_mask, same_reg};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IOp {
    Add,
    Sub,
    Mul,
    AddSatS,
    AddSatU,
    SubSatS,
    SubSatU,
    MulHiS,
    MulHiU,
    MulHrs,
    CmpEq,
    CmpGt,
    MinS,
    MinU,
    MaxS,
    MaxU,
    Avg,
    Sign,
    MulEvenU,
    MulEvenS,
}

/// Lane-wise binary operations `(op, lane type)`.
fn binary(m: Mnemonic) -> Option<(IOp, Elem)> {
    use Elem::{I8, I16, I32, I64};
    use IOp::*;
    use Mnemonic as M;
    Some(match m {
        M::Paddb | M::Vpaddb => (Add, I8),
        M::Paddw | M::Vpaddw => (Add, I16),
        M::Paddd | M::Vpaddd => (Add, I32),
        M::Paddq | M::Vpaddq => (Add, I64),
        M::Psubb | M::Vpsubb => (Sub, I8),
        M::Psubw | M::Vpsubw => (Sub, I16),
        M::Psubd | M::Vpsubd => (Sub, I32),
        M::Psubq | M::Vpsubq => (Sub, I64),
        M::Paddsb | M::Vpaddsb => (AddSatS, I8),
        M::Paddsw | M::Vpaddsw => (AddSatS, I16),
        M::Paddusb | M::Vpaddusb => (AddSatU, I8),
        M::Paddusw | M::Vpaddusw => (AddSatU, I16),
        M::Psubsb | M::Vpsubsb => (SubSatS, I8),
        M::Psubsw | M::Vpsubsw => (SubSatS, I16),
        M::Psubusb | M::Vpsubusb => (SubSatU, I8),
        M::Psubusw | M::Vpsubusw => (SubSatU, I16),
        M::Pmullw | M::Vpmullw => (Mul, I16),
        M::Pmulld | M::Vpmulld => (Mul, I32),
        M::Pmulhw | M::Vpmulhw => (MulHiS, I16),
        M::Pmulhuw | M::Vpmulhuw => (MulHiU, I16),
        M::Pmulhrsw | M::Vpmulhrsw => (MulHrs, I16),
        M::Pmuludq | M::Vpmuludq => (MulEvenU, I64),
        M::Pmuldq | M::Vpmuldq => (MulEvenS, I64),
        M::Pcmpeqb | M::Vpcmpeqb => (CmpEq, I8),
        M::Pcmpeqw | M::Vpcmpeqw => (CmpEq, I16),
        M::Pcmpeqd | M::Vpcmpeqd => (CmpEq, I32),
        M::Pcmpeqq | M::Vpcmpeqq => (CmpEq, I64),
        M::Pcmpgtb | M::Vpcmpgtb => (CmpGt, I8),
        M::Pcmpgtw | M::Vpcmpgtw => (CmpGt, I16),
        M::Pcmpgtd | M::Vpcmpgtd => (CmpGt, I32),
        M::Pcmpgtq | M::Vpcmpgtq => (CmpGt, I64),
        M::Pminsb | M::Vpminsb => (MinS, I8),
        M::Pminsw | M::Vpminsw => (MinS, I16),
        M::Pminsd | M::Vpminsd => (MinS, I32),
        M::Pminub | M::Vpminub => (MinU, I8),
        M::Pminuw | M::Vpminuw => (MinU, I16),
        M::Pminud | M::Vpminud => (MinU, I32),
        M::Pmaxsb | M::Vpmaxsb => (MaxS, I8),
        M::Pmaxsw | M::Vpmaxsw => (MaxS, I16),
        M::Pmaxsd | M::Vpmaxsd => (MaxS, I32),
        M::Pmaxub | M::Vpmaxub => (MaxU, I8),
        M::Pmaxuw | M::Vpmaxuw => (MaxU, I16),
        M::Pmaxud | M::Vpmaxud => (MaxU, I32),
        M::Vpminsq => (MinS, I64),
        M::Vpminuq => (MinU, I64),
        M::Vpmaxsq => (MaxS, I64),
        M::Vpmaxuq => (MaxU, I64),
        M::Pavgb | M::Vpavgb => (Avg, I8),
        M::Pavgw | M::Vpavgw => (Avg, I16),
        M::Psignb | M::Vpsignb => (Sign, I8),
        M::Psignw | M::Vpsignw => (Sign, I16),
        M::Psignd | M::Vpsignd => (Sign, I32),
        _ => return None,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShiftKind {
    Left,
    Logical,
    Arith,
}

/// `(kind, lane type)` of `psll/psrl/psra` (immediate or XMM count).
fn shift(m: Mnemonic) -> Option<(ShiftKind, Elem)> {
    use Elem::{I16, I32, I64};
    use Mnemonic as M;
    use ShiftKind::*;
    Some(match m {
        M::Psllw | M::Vpsllw => (Left, I16),
        M::Pslld | M::Vpslld => (Left, I32),
        M::Psllq | M::Vpsllq => (Left, I64),
        M::Psrlw | M::Vpsrlw => (Logical, I16),
        M::Psrld | M::Vpsrld => (Logical, I32),
        M::Psrlq | M::Vpsrlq => (Logical, I64),
        M::Psraw | M::Vpsraw => (Arith, I16),
        M::Psrad | M::Vpsrad => (Arith, I32),
        _ => return None,
    })
}

/// `(kind, lane type)` of the AVX2 per-element variable shifts.
fn variable_shift(m: Mnemonic) -> Option<(ShiftKind, Elem)> {
    use Elem::{I32, I64};
    use Mnemonic as M;
    use ShiftKind::*;
    Some(match m {
        M::Vpsllvd => (Left, I32),
        M::Vpsllvq => (Left, I64),
        M::Vpsrlvd => (Logical, I32),
        M::Vpsrlvq => (Logical, I64),
        M::Vpsravd => (Arith, I32),
        _ => return None,
    })
}

/// Unpack shuffle masks: interleave the low (`high == false`) or high halves of a and b.
fn unpack_mask(e: Elem, high: bool) -> Vec<u32> {
    let n = e.lanes();
    let base = if high { n / 2 } else { 0 };
    (0..n / 2).flat_map(|i| [base + i, n + base + i]).collect()
}

/// Whether [`FunctionLifter::lift_int`] handles `m` (mirroring its dispatch).
pub(super) fn is_int_supported(m: Mnemonic) -> bool {
    use Mnemonic as M;
    if binary(m).is_some() || shift(m).is_some() || variable_shift(m).is_some() {
        return true;
    }
    matches!(
        m,
        M::Pand
            | M::Vpand
            | M::Pandn
            | M::Vpandn
            | M::Por
            | M::Vpor
            | M::Pxor
            | M::Vpxor
            | M::Pslldq
            | M::Vpslldq
            | M::Psrldq
            | M::Vpsrldq
            | M::Pshufd
            | M::Vpshufd
            | M::Pshuflw
            | M::Vpshuflw
            | M::Pshufhw
            | M::Vpshufhw
            | M::Pshufb
            | M::Vpshufb
            | M::Punpcklbw
            | M::Vpunpcklbw
            | M::Punpckhbw
            | M::Vpunpckhbw
            | M::Punpcklwd
            | M::Vpunpcklwd
            | M::Punpckhwd
            | M::Vpunpckhwd
            | M::Punpckldq
            | M::Vpunpckldq
            | M::Punpckhdq
            | M::Vpunpckhdq
            | M::Punpcklqdq
            | M::Vpunpcklqdq
            | M::Punpckhqdq
            | M::Vpunpckhqdq
            | M::Packsswb
            | M::Vpacksswb
            | M::Packssdw
            | M::Vpackssdw
            | M::Packuswb
            | M::Vpackuswb
            | M::Packusdw
            | M::Vpackusdw
            | M::Pabsb
            | M::Vpabsb
            | M::Pabsw
            | M::Vpabsw
            | M::Pabsd
            | M::Vpabsd
            | M::Pmaddwd
            | M::Vpmaddwd
            | M::Pmaddubsw
            | M::Vpmaddubsw
            | M::Psadbw
            | M::Vpsadbw
            | M::Palignr
            | M::Vpalignr
            | M::Pblendw
            | M::Vpblendw
            | M::Vpblendd
            | M::Pblendvb
            | M::Vpblendvb
            | M::Phaddw
            | M::Vphaddw
            | M::Phaddd
            | M::Vphaddd
            | M::Phaddsw
            | M::Vphaddsw
            | M::Phsubw
            | M::Vphsubw
            | M::Phsubd
            | M::Vphsubd
            | M::Phsubsw
            | M::Vphsubsw
            | M::Pextrb
            | M::Vpextrb
            | M::Pextrw
            | M::Vpextrw
            | M::Pextrd
            | M::Vpextrd
            | M::Pextrq
            | M::Vpextrq
            | M::Pinsrb
            | M::Vpinsrb
            | M::Pinsrw
            | M::Vpinsrw
            | M::Pinsrd
            | M::Vpinsrd
            | M::Pinsrq
            | M::Vpinsrq
    )
}

impl FunctionLifter<'_> {
    pub(super) fn lift_int(&mut self, ins: &Instruction, vex: bool) -> R<bool> {
        use Elem::{I8, I16, I32, I64};
        use Mnemonic as M;
        let f = Form::of(vex);
        let m = ins.mnemonic();
        if let Some((op, e)) = binary(m) {
            self.int_binary(ins, f, op, e)?;
            return Ok(true);
        }
        if let Some((kind, e)) = shift(m) {
            self.int_shift(ins, f, kind, e)?;
            return Ok(true);
        }
        if let Some((kind, e)) = variable_shift(m) {
            self.variable_shift(ins, f, kind, e)?;
            return Ok(true);
        }
        match m {
            M::Pand | M::Vpand => self.logic(ins, f, "and", false)?,
            M::Pandn | M::Vpandn => self.logic(ins, f, "and", true)?,
            M::Por | M::Vpor => self.logic(ins, f, "or", false)?,
            M::Pxor | M::Vpxor => self.logic(ins, f, "xor", false)?,
            M::Pslldq | M::Vpslldq | M::Psrldq | M::Vpsrldq => {
                // pslldq xmm, imm / vpslldq xmm1, xmm2, imm
                let i = Self::imm8(ins)?;
                if i > 15 {
                    self.xset(ins.op_register(0), "zeroinitializer", I8)?;
                    return Ok(true);
                }
                let src = if ins.op_count() == 3 { 1 } else { 0 };
                let v = self.v128(ins, src, I8)?;
                let r = if i == 0 || v == "zeroinitializer" {
                    v
                } else {
                    let left = matches!(m, M::Pslldq | M::Vpslldq);
                    // Index 16 of the second operand is a zero byte.
                    let idx: Vec<u32> = (0..16)
                        .map(|b| {
                            if left {
                                if b >= i { b - i } else { 16 }
                            } else if b + i < 16 {
                                b + i
                            } else {
                                16
                            }
                        })
                        .collect();
                    self.shuffle(&v, "zeroinitializer", I8, &idx)
                };
                self.xset(ins.op_register(0), &r, I8)?;
            }
            M::Pshufd | M::Vpshufd => {
                let i = Self::imm8(ins)?;
                let v = self.v128(ins, 1, I32)?;
                let idx: [u32; 4] = std::array::from_fn(|l| (i >> (2 * l)) & 3);
                let r = if v == "zeroinitializer" || is_identity_mask(&idx) {
                    v
                } else {
                    self.shuffle(&v, "poison", I32, &idx)
                };
                self.xset(ins.op_register(0), &r, I32)?;
            }
            M::Pshuflw | M::Vpshuflw | M::Pshufhw | M::Vpshufhw => {
                let i = Self::imm8(ins)?;
                let v = self.v128(ins, 1, I16)?;
                let high = matches!(m, M::Pshufhw | M::Vpshufhw);
                let idx: [u32; 8] = std::array::from_fn(|l| {
                    let l = l as u32;
                    match (high, l < 4) {
                        (false, true) => (i >> (2 * l)) & 3,
                        (true, false) => 4 + ((i >> (2 * (l - 4))) & 3),
                        _ => l,
                    }
                });
                let r = if v == "zeroinitializer" || is_identity_mask(&idx) {
                    v
                } else {
                    self.shuffle(&v, "poison", I16, &idx)
                };
                self.xset(ins.op_register(0), &r, I16)?;
            }
            M::Pshufb | M::Vpshufb => {
                let a = self.v128(ins, f.a, I8)?;
                let b = self.v128(ins, f.b, I8)?;
                if a == "zeroinitializer" {
                    self.xset(ins.op_register(f.dst), "zeroinitializer", I8)?;
                    return Ok(true);
                }
                // Bit 7 set selects zero: keep it so the index is >= 16 (swizzle yields 0).
                let idx = self.val(&format!("and <16 x i8> {b}, {}", Ty::v(I8).splat("-113")));
                let r = self.intr(Intr::Swizzle, Ty::v(I8), &[&a, &idx]);
                self.xset(ins.op_register(f.dst), &r, I8)?;
            }
            M::Punpcklbw | M::Vpunpcklbw => self.shuffle_op(ins, f, I8, &unpack_mask(I8, false))?,
            M::Punpckhbw | M::Vpunpckhbw => self.shuffle_op(ins, f, I8, &unpack_mask(I8, true))?,
            M::Punpcklwd | M::Vpunpcklwd => {
                self.shuffle_op(ins, f, I16, &unpack_mask(I16, false))?
            }
            M::Punpckhwd | M::Vpunpckhwd => {
                self.shuffle_op(ins, f, I16, &unpack_mask(I16, true))?
            }
            M::Punpckldq | M::Vpunpckldq => {
                self.shuffle_op(ins, f, I32, &unpack_mask(I32, false))?
            }
            M::Punpckhdq | M::Vpunpckhdq => {
                self.shuffle_op(ins, f, I32, &unpack_mask(I32, true))?
            }
            M::Punpcklqdq | M::Vpunpcklqdq => {
                self.shuffle_op(ins, f, I64, &unpack_mask(I64, false))?
            }
            M::Punpckhqdq | M::Vpunpckhqdq => {
                self.shuffle_op(ins, f, I64, &unpack_mask(I64, true))?
            }
            M::Packsswb | M::Vpacksswb => self.pack(ins, f, I16, true)?,
            M::Packssdw | M::Vpackssdw => self.pack(ins, f, I32, true)?,
            M::Packuswb | M::Vpackuswb => self.pack(ins, f, I16, false)?,
            M::Packusdw | M::Vpackusdw => self.pack(ins, f, I32, false)?,
            M::Pabsb | M::Vpabsb => self.abs(ins, I8)?,
            M::Pabsw | M::Vpabsw => self.abs(ins, I16)?,
            M::Pabsd | M::Vpabsd => self.abs(ins, I32)?,
            M::Pmaddwd | M::Vpmaddwd => {
                let a = self.v128(ins, f.a, I16)?;
                let b = self.v128(ins, f.b, I16)?;
                let r = if a == "zeroinitializer" || b == "zeroinitializer" {
                    "zeroinitializer".to_string()
                } else {
                    self.intr(Intr::WasmDot, Ty::v(I16), &[&a, &b])
                };
                self.xset(ins.op_register(f.dst), &r, I32)?;
            }
            M::Pmaddubsw | M::Vpmaddubsw => {
                // View byte pairs as `<8 x i16>` (little-endian: low byte = even, high byte = odd)
                // to avoid cross-vector shuffles and wide `<16 x i16>` multiplies.
                let a = self.v128(ins, f.a, I16)?;
                let b = self.v128(ins, f.b, I16)?;
                let r = if a == "zeroinitializer" || b == "zeroinitializer" {
                    "zeroinitializer".to_string()
                } else {
                    let ty = Ty::v(I16);
                    let c8 = ty.splat("8");
                    let ae = self.val(&format!("and <8 x i16> {a}, {}", ty.splat("255")));
                    let ao = self.val(&format!("lshr <8 x i16> {a}, {c8}"));
                    let be_s = self.val(&format!("shl <8 x i16> {b}, {c8}"));
                    let be = self.val(&format!("ashr <8 x i16> {be_s}, {c8}"));
                    let bo = self.val(&format!("ashr <8 x i16> {b}, {c8}"));
                    // Each u8 * i8 product fits in signed i16; the pair sum saturates.
                    let pe = self.val(&format!("mul <8 x i16> {ae}, {be}"));
                    let po = self.val(&format!("mul <8 x i16> {ao}, {bo}"));
                    self.intr(Intr::SAddSat, ty, &[&pe, &po])
                };
                self.xset(ins.op_register(f.dst), &r, I16)?;
            }
            M::Psadbw | M::Vpsadbw => {
                if same_reg(ins, f.a, f.b) {
                    self.xset(ins.op_register(f.dst), "zeroinitializer", I64)?;
                    return Ok(true);
                }
                // Pure SIMD128 sum of absolute differences via native pairwise widening add:
                // |a - b| = umax(a, b) - umin(a, b), then `extadd_pairwise` (16x8 -> 8x16 -> 4x32)
                // and final 64-bit lane fold (drops 6 vector ops + 2 18-byte constants vs bitmask).
                let a = self.v128(ins, f.a, I8)?;
                let b = self.v128(ins, f.b, I8)?;
                if a == "zeroinitializer" && b == "zeroinitializer" {
                    self.xset(ins.op_register(f.dst), "zeroinitializer", I64)?;
                    return Ok(true);
                }
                let hi = self.intr(Intr::UMax, Ty::v(I8), &[&a, &b]);
                let lo = self.intr(Intr::UMin, Ty::v(I8), &[&a, &b]);
                let d = self.val(&format!("sub <16 x i8> {hi}, {lo}"));
                let s16 = self.intr(Intr::ExtAddPairwiseUnsigned, Ty::v(I8), &[&d]);
                let s32 = self.intr(Intr::ExtAddPairwiseUnsigned, Ty::v(I16), &[&s16]);
                let s64_v = self.cast(&s32, Ty::v(I32), Ty::v(I64));
                let s64_lo = self.val(&format!(
                    "and <2 x i64> {s64_v}, {}",
                    Ty::v(I64).splat("4294967295")
                ));
                let s64_hi =
                    self.val(&format!("lshr <2 x i64> {s64_v}, {}", Ty::v(I64).splat("32")));
                let r = self.val(&format!("add nuw <2 x i64> {s64_lo}, {s64_hi}"));
                self.xset(ins.op_register(f.dst), &r, I64)?;
            }
            M::Palignr | M::Vpalignr => {
                let i = Self::imm8(ins)?;
                let hi = self.v128(ins, f.a, I8)?;
                let lo = self.v128(ins, f.b, I8)?;
                // (hi:lo) >> 8*imm: byte k is lo[k+i] (< 16), hi[k+i-16] (< 32), else 0.
                let r = if hi == "zeroinitializer" && lo == "zeroinitializer" {
                    "zeroinitializer".to_string()
                } else {
                    match i {
                        0 => lo,
                        1..=15 => {
                            let idx: Vec<u32> = (0..16).map(|k| k + i).collect();
                            self.shuffle(&lo, &hi, I8, &idx)
                        }
                        16 => hi,
                        17..=31 => {
                            let shift = i - 16;
                            let idx: Vec<u32> = (0..16).map(|k| (k + shift).min(16)).collect();
                            self.shuffle(&hi, "zeroinitializer", I8, &idx)
                        }
                        _ => "zeroinitializer".to_string(),
                    }
                };
                self.xset(ins.op_register(f.dst), &r, I8)?;
            }
            M::Pblendw | M::Vpblendw => {
                let i = Self::imm8(ins)?;
                let idx: Vec<u32> = (0..8)
                    .map(|l| if i >> l & 1 == 1 { 8 + l } else { l })
                    .collect();
                self.shuffle_op(ins, f, I16, &idx)?;
            }
            M::Vpblendd => {
                let i = Self::imm8(ins)?;
                let i = self.lane_imm(i, 4);
                let idx: Vec<u32> = (0..4)
                    .map(|l| if i >> l & 1 == 1 { 4 + l } else { l })
                    .collect();
                self.shuffle_op(ins, f, I32, &idx)?;
            }
            M::Pblendvb | M::Vpblendvb => self.blend_var(ins, f, vex, I8)?,
            M::Phaddw | M::Vphaddw => self.int_horizontal(ins, f, I16, "add", false)?,
            M::Phaddd | M::Vphaddd => self.int_horizontal(ins, f, I32, "add", false)?,
            M::Phaddsw | M::Vphaddsw => self.int_horizontal(ins, f, I16, "add", true)?,
            M::Phsubw | M::Vphsubw => self.int_horizontal(ins, f, I16, "sub", false)?,
            M::Phsubd | M::Vphsubd => self.int_horizontal(ins, f, I32, "sub", false)?,
            M::Phsubsw | M::Vphsubsw => self.int_horizontal(ins, f, I16, "sub", true)?,
            M::Pextrb | M::Vpextrb => self.extract_elem(ins, I8)?,
            M::Pextrw | M::Vpextrw => self.extract_elem(ins, I16)?,
            M::Pextrd | M::Vpextrd => self.extract_elem(ins, I32)?,
            M::Pextrq | M::Vpextrq => self.extract_elem(ins, I64)?,
            M::Pinsrb | M::Vpinsrb => self.insert_elem(ins, f, I8)?,
            M::Pinsrw | M::Vpinsrw => self.insert_elem(ins, f, I16)?,
            M::Pinsrd | M::Vpinsrd => self.insert_elem(ins, f, I32)?,
            M::Pinsrq | M::Vpinsrq => self.insert_elem(ins, f, I64)?,
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn int_binary(&mut self, ins: &Instruction, f: Form, op: IOp, e: Elem) -> R<()> {
        let ty = Ty::v(e);
        if same_reg(ins, f.a, f.b) {
            match op {
                IOp::CmpEq => {
                    return self.xset(ins.op_register(f.dst), &ty.splat("-1"), e);
                }
                IOp::CmpGt | IOp::Sub | IOp::SubSatS | IOp::SubSatU => {
                    return self.xset(ins.op_register(f.dst), "zeroinitializer", e);
                }
                IOp::MinS | IOp::MinU | IOp::MaxS | IOp::MaxU | IOp::Avg => {
                    let a = self.v128(ins, f.a, e)?;
                    return self.xset(ins.op_register(f.dst), &a, e);
                }
                IOp::Sign => {
                    // `psign*(a, a)` negates negative lanes, keeps positive lanes, and preserves 0
                    // (and `-INT_MIN == INT_MIN`), which is bit-exact with `@llvm.abs(..., false)`.
                    let a = self.v128(ins, f.a, e)?;
                    let r = if a == "zeroinitializer" {
                        a
                    } else {
                        self.intr(Intr::Abs, ty, &[&a])
                    };
                    return self.xset(ins.op_register(f.dst), &r, e);
                }
                _ => {}
            }
        }
        if matches!(op, IOp::MulEvenU | IOp::MulEvenS) {
            // Load directly as `<4 x i32>` (avoiding `<2 x i64> -> <4 x i32>` bitcast) so LLVM
            // emits native `i64x2.extmul_low_i32x4_{u,s}` on `<4 x i32>` chains and memory loads.
            let a32 = self.v128(ins, f.a, Elem::I32)?;
            let same = same_reg(ins, f.a, f.b);
            let b32 = if same {
                a32.clone()
            } else {
                self.v128(ins, f.b, Elem::I32)?
            };
            if a32 == "zeroinitializer" || b32 == "zeroinitializer" {
                return self.xset(ins.op_register(f.dst), "zeroinitializer", Elem::I64);
            }
            let even = super::mask(&[0, 2]);
            let ae = self.val(&format!(
                "shufflevector <4 x i32> {a32}, <4 x i32> poison, {even}"
            ));
            let (ext, mflag) = if op == IOp::MulEvenU {
                ("zext", "nuw")
            } else {
                ("sext", "nsw")
            };
            let wa = self.val(&format!("{ext} <2 x i32> {ae} to <2 x i64>"));
            let wb = if same {
                wa.clone()
            } else {
                let be = self.val(&format!(
                    "shufflevector <4 x i32> {b32}, <4 x i32> poison, {even}"
                ));
                self.val(&format!("{ext} <2 x i32> {be} to <2 x i64>"))
            };
            let r = self.val(&format!("mul {mflag} <2 x i64> {wa}, {wb}"));
            return self.xset(ins.op_register(f.dst), &r, Elem::I64);
        }
        let t = ty.ir();
        let bt = ty.bool_ir();
        let a = self.v128(ins, f.a, e)?;
        let b = if same_reg(ins, f.a, f.b) {
            a.clone()
        } else {
            self.v128(ins, f.b, e)?
        };
        let a_zero = a == "zeroinitializer";
        let b_zero = b == "zeroinitializer";
        if a_zero || b_zero {
            let folded = match op {
                IOp::Add | IOp::AddSatS | IOp::AddSatU | IOp::MaxU => {
                    Some(if a_zero { b.clone() } else { a.clone() })
                }
                IOp::Sub | IOp::SubSatS if b_zero => Some(a.clone()),
                IOp::SubSatU => Some(if a_zero {
                    "zeroinitializer".to_string()
                } else {
                    a.clone()
                }),
                IOp::Mul | IOp::MulHiS | IOp::MulHiU | IOp::MulHrs | IOp::MinU | IOp::Sign => {
                    Some("zeroinitializer".to_string())
                }
                IOp::CmpEq if a_zero && b_zero => Some(ty.splat("-1")),
                IOp::CmpGt if a_zero && b_zero => Some("zeroinitializer".to_string()),
                _ => None,
            };
            if let Some(r) = folded {
                return self.xset(ins.op_register(f.dst), &r, e);
            }
        }
        let r = match op {
            IOp::Add => self.val(&format!("add {t} {a}, {b}")),
            IOp::Sub => self.val(&format!("sub {t} {a}, {b}")),
            IOp::Mul => self.val(&format!("mul {t} {a}, {b}")),
            IOp::AddSatS => self.intr(Intr::SAddSat, ty, &[&a, &b]),
            IOp::AddSatU => self.intr(Intr::UAddSat, ty, &[&a, &b]),
            IOp::SubSatS => self.intr(Intr::SSubSat, ty, &[&a, &b]),
            IOp::SubSatU => self.intr(Intr::USubSat, ty, &[&a, &b]),
            IOp::MulHrs => {
                // Native Wasm `i16x8.q15mulr_sat_s` matches x86 `pmulhrsw` on all 2^32 - 1 pairs
                // except `(-32768, -32768)` where Wasm saturates to +32767 (`0x7FFF`) while x86
                // wraps to -32768 (`0x8000`). Subtracting `sext(both == -32768)` (`0x7FFF - (-1)
                // = 0x8000`) restores bit-exact x86 semantics in 5 ops (3 ops when `a == b`).
                let q = self.intr(Intr::Q15MulrSatSigned, ty, &[&a, &b]);
                let min16 = ty.splat("-32768");
                let ea = self.val(&format!("icmp eq <8 x i16> {a}, {min16}"));
                let both = if same_reg(ins, f.a, f.b) {
                    ea
                } else {
                    let eb = self.val(&format!("icmp eq <8 x i16> {b}, {min16}"));
                    self.val(&format!("and <8 x i1> {ea}, {eb}"))
                };
                let m = self.val(&format!("sext <8 x i1> {both} to <8 x i16>"));
                self.val(&format!("sub <8 x i16> {q}, {m}"))
            }
            IOp::MulHiS | IOp::MulHiU => {
                // Split into low/high `<4 x i16>` -> `<4 x i32>` extmul halves, then pick the
                // odd `i16` lanes [1,3,5,7,9,11,13,15] directly with one `shufflevector`.
                let (ext, mflag) = if op == IOp::MulHiU {
                    ("zext", "nuw")
                } else {
                    ("sext", "nsw")
                };
                let same = same_reg(ins, f.a, f.b);
                let lo_mask = super::mask(&[0, 1, 2, 3]);
                let hi_mask = super::mask(&[4, 5, 6, 7]);
                let mut halves = Vec::with_capacity(2);
                for half_mask in [&lo_mask, &hi_mask] {
                    let ah = self.val(&format!(
                        "shufflevector <8 x i16> {a}, <8 x i16> poison, {half_mask}"
                    ));
                    let wa = self.val(&format!("{ext} <4 x i16> {ah} to <4 x i32>"));
                    let wb = if same {
                        wa.clone()
                    } else {
                        let bh = self.val(&format!(
                            "shufflevector <8 x i16> {b}, <8 x i16> poison, {half_mask}"
                        ));
                        self.val(&format!("{ext} <4 x i16> {bh} to <4 x i32>"))
                    };
                    let p = self.val(&format!("mul {mflag} <4 x i32> {wa}, {wb}"));
                    halves.push(self.cast(&p, Ty::v(Elem::I32), Ty::v(Elem::I16)));
                }
                self.shuffle(
                    &halves[0],
                    &halves[1],
                    Elem::I16,
                    &[1, 3, 5, 7, 9, 11, 13, 15],
                )
            }
            IOp::MulEvenU | IOp::MulEvenS => unreachable!(),
            IOp::CmpEq | IOp::CmpGt => {
                let pred = if op == IOp::CmpEq { "eq" } else { "sgt" };
                let c = self.val(&format!("icmp {pred} {t} {a}, {b}"));
                self.val(&format!("sext {bt} {c} to {t}"))
            }
            IOp::MinS => self.intr(Intr::SMin, ty, &[&a, &b]),
            IOp::MinU if e == Elem::I64 => {
                // Wasm SIMD128 lacks unsigned `i64x2` compare, so `llvm.umin.v2i64` scalarizes to
                // 21 ops. Flipping the sign bit reduces to `llvm.smin.v2i64` (5 pure SIMD128 ops).
                let bias = "<i64 -9223372036854775808, i64 -9223372036854775808>";
                let sa = self.val(&format!("xor <2 x i64> {a}, {bias}"));
                let sb = self.val(&format!("xor <2 x i64> {b}, {bias}"));
                let sm = self.intr(Intr::SMin, ty, &[&sa, &sb]);
                self.val(&format!("xor <2 x i64> {sm}, {bias}"))
            }
            IOp::MinU => self.intr(Intr::UMin, ty, &[&a, &b]),
            IOp::MaxS => self.intr(Intr::SMax, ty, &[&a, &b]),
            IOp::MaxU if e == Elem::I64 => {
                let bias = "<i64 -9223372036854775808, i64 -9223372036854775808>";
                let sa = self.val(&format!("xor <2 x i64> {a}, {bias}"));
                let sb = self.val(&format!("xor <2 x i64> {b}, {bias}"));
                let sm = self.intr(Intr::SMax, ty, &[&sa, &sb]);
                self.val(&format!("xor <2 x i64> {sm}, {bias}"))
            }
            IOp::MaxU => self.intr(Intr::UMax, ty, &[&a, &b]),
            IOp::Avg => self.intr(Intr::AvgrUnsigned, ty, &[&a, &b]),
            IOp::Sign => {
                let neg = self.val(&format!("sub {t} zeroinitializer, {a}"));
                let lt = self.val(&format!("icmp slt {t} {b}, zeroinitializer"));
                let eq = self.val(&format!("icmp eq {t} {b}, zeroinitializer"));
                let r = self.val(&format!("select {bt} {lt}, {t} {neg}, {t} {a}"));
                self.val(&format!("select {bt} {eq}, {t} zeroinitializer, {t} {r}"))
            }
        };
        self.xset(ins.op_register(f.dst), &r, e)
    }

    /// `psll/psrl/psra` by an immediate or by the low 64 bits of an XMM/`m128` count operand
    /// (also in 256-bit forms). Counts >= the lane width give 0 (logical) or the sign (arith).
    fn int_shift(&mut self, ins: &Instruction, f: Form, kind: ShiftKind, e: Elem) -> R<()> {
        let ty = Ty::v(e);
        let t = ty.ir();
        let last = ins.op_count() - 1;
        let width = e.bits();
        // Immediate forms: `psllw xmm, imm` / `vpsllw xmm1, xmm2, imm`.
        if super::super::is_immediate(ins.op_kind(last)) {
            let src = if ins.op_count() == 3 { 1 } else { 0 };
            let c = Self::imm8(ins)?;
            if c >= width && kind != ShiftKind::Arith {
                return self.xset(ins.op_register(0), "zeroinitializer", e);
            }
            let v = self.v128(ins, src, e)?;
            let r = if c == 0 || v == "zeroinitializer" {
                v
            } else if c >= width {
                self.val(&format!(
                    "ashr {t} {v}, {}",
                    ty.splat(&(width - 1).to_string())
                ))
            } else {
                let op = match kind {
                    ShiftKind::Left => "shl",
                    ShiftKind::Logical => "lshr",
                    ShiftKind::Arith => "ashr",
                };
                self.val(&format!("{op} {t} {v}, {}", ty.splat(&c.to_string())))
            };
            return self.xset(ins.op_register(0), &r, e);
        }
        let count = match ins.op_kind(f.b) {
            OpKind::Register => {
                // The count is always the low quadword of an XMM register, also for YMM data.
                let v = self.xget_half(ins.op_register(f.b), 0, Elem::I64)?;
                if v == "zeroinitializer" {
                    "0".to_string()
                } else {
                    self.extract(&v, Elem::I64, 0)
                }
            }
            OpKind::Memory => self.mem_load_at(ins, Ty::s(Elem::I64), 0, 16)?,
            k => return unsupported(format!("shift count operand {k:?}")),
        };
        let v = self.v128(ins, f.a, e)?;
        if v == "zeroinitializer" || count == "0" {
            return self.xset(ins.op_register(0), &v, e);
        }
        let big = self.val(&format!("icmp ugt i64 {count}, {}", width - 1));
        let r = match kind {
            ShiftKind::Arith => {
                let n = self.val(&format!("select i1 {big}, i64 {}, i64 {count}", width - 1));
                let n = self.narrow_count(&n, e);
                let s = self.splat_value(&n, e);
                self.val(&format!("ashr {t} {v}, {s}"))
            }
            _ => {
                // Masking mod `width` keeps the shift poison-free and lets LLVM fold the mask away
                // completely because Wasm SIMD128 shifts inherently mask counts mod `width`.
                let n = self.val(&format!("and i64 {count}, {}", width - 1));
                let n = self.narrow_count(&n, e);
                let s = self.splat_value(&n, e);
                let op = if kind == ShiftKind::Left {
                    "shl"
                } else {
                    "lshr"
                };
                let r = self.val(&format!("{op} {t} {v}, {s}"));
                self.val(&format!("select i1 {big}, {t} zeroinitializer, {t} {r}"))
            }
        };
        self.xset(ins.op_register(0), &r, e)
    }

    /// An `i64` shift count (already below the lane width) as a lane-typed scalar.
    fn narrow_count(&mut self, n: &str, e: Elem) -> String {
        if e == Elem::I64 {
            n.to_string()
        } else {
            self.val(&format!("trunc i64 {n} to {}", e.scalar()))
        }
    }

    /// AVX2 `vpsllv/vpsrlv/vpsrav`: per-element counts.
    fn variable_shift(&mut self, ins: &Instruction, f: Form, kind: ShiftKind, e: Elem) -> R<()> {
        let ty = Ty::v(e);
        let t = ty.ir();
        let bt = ty.bool_ir();
        let v = self.v128(ins, f.a, e)?;
        let c = self.v128(ins, f.b, e)?;
        if v == "zeroinitializer" || c == "zeroinitializer" {
            return self.xset(ins.op_register(f.dst), &v, e);
        }
        let max = ty.splat(&(e.bits() - 1).to_string());
        let big = self.val(&format!("icmp ugt {t} {c}, {max}"));
        let r = match kind {
            ShiftKind::Arith => {
                let n = self.val(&format!("select {bt} {big}, {t} {max}, {t} {c}"));
                self.val(&format!("ashr {t} {v}, {n}"))
            }
            _ => {
                // Masking count with `width - 1` keeps the shift poison-free while reusing `max`
                // and avoiding a vector `select` before the shift; out-of-range lanes are zeroed
                // by the final `select`.
                let n = self.val(&format!("and {t} {c}, {max}"));
                let op = if kind == ShiftKind::Left {
                    "shl"
                } else {
                    "lshr"
                };
                let r = self.val(&format!("{op} {t} {v}, {n}"));
                self.val(&format!("select {bt} {big}, {t} zeroinitializer, {t} {r}"))
            }
        };
        self.xset(ins.op_register(f.dst), &r, e)
    }

    /// A scalar broadcast to every lane of a 128-bit vector of `e`.
    pub(super) fn splat_value(&mut self, s: &str, e: Elem) -> String {
        let t = Ty::v(e).ir();
        let one = self.val(&format!(
            "insertelement {t} poison, {} {s}, i32 0",
            e.scalar()
        ));
        self.val(&format!(
            "shufflevector {t} {one}, {t} poison, <{} x i32> zeroinitializer",
            e.lanes()
        ))
    }

    /// `packss*`/`packus*`: saturate both signed sources to half-width lanes, `a`'s lanes first.
    /// Emits native Wasm SIMD128 narrowing intrinsics (`i8x16.narrow_i16x8_{s,u}`,
    /// `i16x8.narrow_i32x4_{s,u}`), which treat both inputs as signed identically to x86.
    fn pack(&mut self, ins: &Instruction, f: Form, e: Elem, signed: bool) -> R<()> {
        let a = self.v128(ins, f.a, e)?;
        let b = self.v128(ins, f.b, e)?;
        let ne = Elem::of_bits(e.bits() / 2);
        if a == "zeroinitializer" && b == "zeroinitializer" {
            return self.xset(ins.op_register(f.dst), "zeroinitializer", ne);
        }
        let intr = if signed {
            Intr::NarrowSigned
        } else {
            Intr::NarrowUnsigned
        };
        let r = self.intr(intr, Ty::v(e), &[&a, &b]);
        self.xset(ins.op_register(f.dst), &r, ne)
    }

    fn abs(&mut self, ins: &Instruction, e: Elem) -> R<()> {
        let v = self.v128(ins, 1, e)?;
        let r = if v == "zeroinitializer" {
            v
        } else {
            self.intr(Intr::Abs, Ty::v(e), &[&v])
        };
        self.xset(ins.op_register(0), &r, e)
    }

    /// `phadd*/phsub*`: adjacent pairs of a, then of b (optionally saturating).
    fn int_horizontal(
        &mut self,
        ins: &Instruction,
        f: Form,
        e: Elem,
        op: &str,
        saturate: bool,
    ) -> R<()> {
        let a = self.v128(ins, f.a, e)?;
        let n = e.lanes();
        let (lo, hi) = if same_reg(ins, f.a, f.b) {
            if a == "zeroinitializer" {
                return self.xset(ins.op_register(f.dst), "zeroinitializer", e);
            }
            let even: Vec<u32> = (0..n).map(|l| (2 * l) % n).collect();
            let odd: Vec<u32> = (0..n).map(|l| (2 * l + 1) % n).collect();
            (
                self.shuffle(&a, "poison", e, &even),
                self.shuffle(&a, "poison", e, &odd),
            )
        } else {
            let b = self.v128(ins, f.b, e)?;
            if a == "zeroinitializer" && b == "zeroinitializer" {
                return self.xset(ins.op_register(f.dst), "zeroinitializer", e);
            }
            let even: Vec<u32> = (0..n).map(|l| 2 * l).collect();
            let odd: Vec<u32> = (0..n).map(|l| 2 * l + 1).collect();
            (
                self.shuffle(&a, &b, e, &even),
                self.shuffle(&a, &b, e, &odd),
            )
        };
        let r = if saturate {
            let intr = if op == "add" {
                Intr::SAddSat
            } else {
                Intr::SSubSat
            };
            self.intr(intr, Ty::v(e), &[&lo, &hi])
        } else {
            self.val(&format!("{op} {} {lo}, {hi}", Ty::v(e).ir()))
        };
        self.xset(ins.op_register(f.dst), &r, e)
    }

    /// `pextrb/w/d/q r/m, xmm, imm`: zero-extended into a GPR or stored.
    fn extract_elem(&mut self, ins: &Instruction, e: Elem) -> R<()> {
        let i = Self::imm8(ins)? % e.lanes();
        let v = self.xget(ins.op_register(1), e)?;
        let s = self.extract(&v, e, i);
        self.write_gpr_or_mem(ins, 0, &s, e.bits())
    }

    /// `pinsrb/w/d/q xmm, r/m, imm` (VEX: merges into the first source).
    fn insert_elem(&mut self, ins: &Instruction, f: Form, e: Elem) -> R<()> {
        let i = Self::imm8(ins)? % e.lanes();
        let s = match ins.op_kind(f.b) {
            OpKind::Register => {
                let (v, bits) = self.read_reg(ins.op_register(f.b))?;
                if bits > e.bits() {
                    self.val(&format!("trunc i{bits} {v} to {}", e.scalar()))
                } else if bits == e.bits() {
                    v
                } else {
                    return unsupported("pinsr source register too narrow");
                }
            }
            OpKind::Memory => self.mem_load(ins, Ty::s(e))?,
            k => return unsupported(format!("pinsr source {k:?}")),
        };
        let a = self.xget(ins.op_register(f.a), e)?;
        let r = self.insert(&a, e, &s, i);
        self.xset(ins.op_register(f.dst), &r, e)
    }
}
