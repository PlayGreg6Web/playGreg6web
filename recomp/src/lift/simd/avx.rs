//! Instructions that need whole registers rather than one 128-bit lane at a time: broadcasts,
//! 128-bit inserts/extracts, cross-lane permutes, sign masks and tests of 256-bit registers,
//! masked loads/stores, widening/narrowing conversions, `pmovsx/pmovzx`, F16C half conversions.
//! Both their 128- and 256-bit forms are lifted here; [`in_lane`] lists the 256-bit forms that
//! the lane-wise lifters handle as two independent halves.

use iced_x86::{Instruction, Mnemonic, OpKind, Register};

use super::super::{Flag, FunctionLifter, R, imm, unsupported};
use super::{Elem, Intr, Ty, fconst, is_256, is_identity_mask, mask, same_reg};

/// 256-bit forms that operate on each 128-bit lane independently (with the same per-lane
/// semantics as the 128-bit form; per-lane immediate bits are handled by the lane lifters).
pub(super) fn in_lane(m: Mnemonic) -> bool {
    use Mnemonic as M;
    matches!(
        m,
        M::Vmovaps
            | M::Vmovups
            | M::Vmovapd
            | M::Vmovupd
            | M::Vmovdqa
            | M::Vmovdqu
            | M::Vlddqu
            | M::Vmovntps
            | M::Vmovntpd
            | M::Vmovntdq
            | M::Vmovntdqa
            | M::Vmovsldup
            | M::Vmovshdup
            | M::Vmovddup
            | M::Vshufps
            | M::Vshufpd
            | M::Vunpcklps
            | M::Vunpckhps
            | M::Vunpcklpd
            | M::Vunpckhpd
            | M::Vblendps
            | M::Vblendpd
            | M::Vblendvps
            | M::Vblendvpd
            | M::Vpermilps
            | M::Vpermilpd
            | M::Vaddps
            | M::Vaddpd
            | M::Vsubps
            | M::Vsubpd
            | M::Vmulps
            | M::Vmulpd
            | M::Vdivps
            | M::Vdivpd
            | M::Vminps
            | M::Vminpd
            | M::Vmaxps
            | M::Vmaxpd
            | M::Vsqrtps
            | M::Vsqrtpd
            | M::Vrcpps
            | M::Vrsqrtps
            | M::Vandps
            | M::Vandpd
            | M::Vandnps
            | M::Vandnpd
            | M::Vorps
            | M::Vorpd
            | M::Vxorps
            | M::Vxorpd
            | M::Vcmpps
            | M::Vcmppd
            | M::Vhaddps
            | M::Vhaddpd
            | M::Vhsubps
            | M::Vhsubpd
            | M::Vaddsubps
            | M::Vaddsubpd
            | M::Vroundps
            | M::Vroundpd
            | M::Vdpps
            | M::Vcvtdq2ps
            | M::Vcvtps2dq
            | M::Vcvttps2dq
            | M::Vfmadd132ps
            | M::Vfmadd213ps
            | M::Vfmadd231ps
            | M::Vfmsub132ps
            | M::Vfmsub213ps
            | M::Vfmsub231ps
            | M::Vfnmadd132ps
            | M::Vfnmadd213ps
            | M::Vfnmadd231ps
            | M::Vfnmsub132ps
            | M::Vfnmsub213ps
            | M::Vfnmsub231ps
            | M::Vfmaddsub132ps
            | M::Vfmaddsub213ps
            | M::Vfmaddsub231ps
            | M::Vfmsubadd132ps
            | M::Vfmsubadd213ps
            | M::Vfmsubadd231ps
            | M::Vfmadd132pd
            | M::Vfmadd213pd
            | M::Vfmadd231pd
            | M::Vfmsub132pd
            | M::Vfmsub213pd
            | M::Vfmsub231pd
            | M::Vfnmadd132pd
            | M::Vfnmadd213pd
            | M::Vfnmadd231pd
            | M::Vfnmsub132pd
            | M::Vfnmsub213pd
            | M::Vfnmsub231pd
            | M::Vfmaddsub132pd
            | M::Vfmaddsub213pd
            | M::Vfmaddsub231pd
            | M::Vfmsubadd132pd
            | M::Vfmsubadd213pd
            | M::Vfmsubadd231pd
            | M::Vaesenc
            | M::Vaesenclast
            | M::Vaesdec
            | M::Vaesdeclast
            | M::Vpclmulqdq
            | M::Vpaddb
            | M::Vpaddw
            | M::Vpaddd
            | M::Vpaddq
            | M::Vpsubb
            | M::Vpsubw
            | M::Vpsubd
            | M::Vpsubq
            | M::Vpaddsb
            | M::Vpaddsw
            | M::Vpaddusb
            | M::Vpaddusw
            | M::Vpsubsb
            | M::Vpsubsw
            | M::Vpsubusb
            | M::Vpsubusw
            | M::Vpmullw
            | M::Vpmulld
            | M::Vpmulhw
            | M::Vpmulhuw
            | M::Vpmulhrsw
            | M::Vpmuludq
            | M::Vpmuldq
            | M::Vpmaddwd
            | M::Vpmaddubsw
            | M::Vpcmpeqb
            | M::Vpcmpeqw
            | M::Vpcmpeqd
            | M::Vpcmpeqq
            | M::Vpcmpgtb
            | M::Vpcmpgtw
            | M::Vpcmpgtd
            | M::Vpcmpgtq
            | M::Vpminsb
            | M::Vpminsw
            | M::Vpminsd
            | M::Vpminub
            | M::Vpminuw
            | M::Vpminud
            | M::Vpmaxsb
            | M::Vpmaxsw
            | M::Vpmaxsd
            | M::Vpmaxub
            | M::Vpmaxuw
            | M::Vpmaxud
            | M::Vpminsq
            | M::Vpminuq
            | M::Vpmaxsq
            | M::Vpmaxuq
            | M::Vpavgb
            | M::Vpavgw
            | M::Vpsignb
            | M::Vpsignw
            | M::Vpsignd
            | M::Vpabsb
            | M::Vpabsw
            | M::Vpabsd
            | M::Vpand
            | M::Vpandn
            | M::Vpor
            | M::Vpxor
            | M::Vpsllw
            | M::Vpslld
            | M::Vpsllq
            | M::Vpsrlw
            | M::Vpsrld
            | M::Vpsrlq
            | M::Vpsraw
            | M::Vpsrad
            | M::Vpsllvd
            | M::Vpsllvq
            | M::Vpsrlvd
            | M::Vpsrlvq
            | M::Vpsravd
            | M::Vpslldq
            | M::Vpsrldq
            | M::Vpshufd
            | M::Vpshuflw
            | M::Vpshufhw
            | M::Vpshufb
            | M::Vpunpcklbw
            | M::Vpunpckhbw
            | M::Vpunpcklwd
            | M::Vpunpckhwd
            | M::Vpunpckldq
            | M::Vpunpckhdq
            | M::Vpunpcklqdq
            | M::Vpunpckhqdq
            | M::Vpacksswb
            | M::Vpackssdw
            | M::Vpackuswb
            | M::Vpackusdw
            | M::Vpsadbw
            | M::Vpalignr
            | M::Vpblendw
            | M::Vpblendd
            | M::Vpblendvb
            | M::Vphaddw
            | M::Vphaddd
            | M::Vphaddsw
            | M::Vphsubw
            | M::Vphsubd
            | M::Vphsubsw
    )
}

/// Whether [`FunctionLifter::lift_avx`] handles `ins` (mirroring its match arms).
pub(super) fn is_avx_supported(ins: &Instruction) -> bool {
    use Mnemonic as M;
    let wide = is_256(ins);
    match ins.mnemonic() {
        M::Vbroadcastss
        | M::Vbroadcastsd
        | M::Vpbroadcastb
        | M::Vpbroadcastw
        | M::Vpbroadcastd
        | M::Vpbroadcastq
        | M::Vbroadcastf128
        | M::Vbroadcasti128
        | M::Vinsertf128
        | M::Vinserti128
        | M::Vextractf128
        | M::Vextracti128
        | M::Vperm2f128
        | M::Vperm2i128
        | M::Vpermq
        | M::Vpermpd
        | M::Vpermd
        | M::Vpermps
        | M::Ptest
        | M::Vptest
        | M::Vtestps
        | M::Vtestpd
        | M::Vmaskmovps
        | M::Vpmaskmovd
        | M::Vmaskmovpd
        | M::Vpmaskmovq
        | M::Pmovsxbw
        | M::Vpmovsxbw
        | M::Pmovsxbd
        | M::Vpmovsxbd
        | M::Pmovsxbq
        | M::Vpmovsxbq
        | M::Pmovsxwd
        | M::Vpmovsxwd
        | M::Pmovsxwq
        | M::Vpmovsxwq
        | M::Pmovsxdq
        | M::Vpmovsxdq
        | M::Pmovzxbw
        | M::Vpmovzxbw
        | M::Pmovzxbd
        | M::Vpmovzxbd
        | M::Pmovzxbq
        | M::Vpmovzxbq
        | M::Pmovzxwd
        | M::Vpmovzxwd
        | M::Pmovzxwq
        | M::Vpmovzxwq
        | M::Pmovzxdq
        | M::Vpmovzxdq
        | M::Vcvtph2ps
        | M::Vcvtps2ph => true,
        M::Vmovmskps
        | M::Vmovmskpd
        | M::Vpmovmskb
        | M::Vcvtps2pd
        | M::Vcvtdq2pd
        | M::Vcvtpd2ps
        | M::Vcvtpd2dq
        | M::Vcvttpd2dq => wide,
        _ => false,
    }
}

/// F16C rounding control from `vcvtps2ph`'s immediate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Round {
    Nearest,
    Down,
    Up,
    Zero,
}

impl FunctionLifter<'_> {
    /// Writes the 128-bit halves of a destination register: both halves for a YMM register, the
    /// low half for an XMM register (VEX zeroing of the upper half follows in `lift_simd`).
    fn set_halves(&mut self, reg: Register, lo: &str, hi: Option<&str>, e: Elem) -> R<()> {
        self.xset_half(reg, 0, lo, e)?;
        if reg.is_ymm() {
            let Some(hi) = hi else {
                return unsupported("internal: missing upper half");
            };
            self.xset_half(reg, 1, hi, e)?;
        }
        Ok(())
    }

    /// Lifts the instructions of this module; `Ok(false)` if `ins` is not one of them.
    pub(super) fn lift_avx(&mut self, ins: &Instruction) -> R<bool> {
        use Elem::{F32, F64, I8, I16, I32, I64};
        use Mnemonic as M;
        let m = ins.mnemonic();
        let wide = is_256(ins);
        match m {
            M::Vbroadcastss => self.broadcast(ins, I32)?,
            M::Vbroadcastsd => self.broadcast(ins, I64)?,
            M::Vpbroadcastb => self.broadcast(ins, I8)?,
            M::Vpbroadcastw => self.broadcast(ins, I16)?,
            M::Vpbroadcastd => self.broadcast(ins, I32)?,
            M::Vpbroadcastq => self.broadcast(ins, I64)?,
            M::Vbroadcastf128 | M::Vbroadcasti128 => {
                let v = self.mem_load_at(ins, Ty::v(I64), 0, 16)?;
                self.set_halves(ins.op_register(0), &v, Some(&v), I64)?;
            }
            M::Vinsertf128 | M::Vinserti128 => {
                let sel = Self::imm8(ins)? & 1;
                let x = match ins.op_kind(2) {
                    OpKind::Register => self.xget_half(ins.op_register(2), 0, I64)?,
                    _ => self.mem_load_at(ins, Ty::v(I64), 0, 16)?,
                };
                let (lo, hi) = if sel == 0 {
                    (x, self.xget_half(ins.op_register(1), 1, I64)?)
                } else {
                    (self.xget_half(ins.op_register(1), 0, I64)?, x)
                };
                self.set_halves(ins.op_register(0), &lo, Some(&hi), I64)?;
            }
            M::Vextractf128 | M::Vextracti128 => {
                let sel = Self::imm8(ins)? & 1;
                let v = self.xget_half(ins.op_register(1), sel, I64)?;
                match ins.op_kind(0) {
                    OpKind::Register => self.xset_half(ins.op_register(0), 0, &v, I64)?,
                    _ => self.mem_store_at(ins, &v, Ty::v(I64), 0, 16)?,
                }
            }
            M::Vperm2f128 | M::Vperm2i128 => {
                let i = Self::imm8(ins)?;
                let cs = [i & 0xF, i >> 4];
                let need_mem2 = ins.op_kind(2) == OpKind::Memory;
                let mut loaded: [Option<String>; 4] = [None, None, None, None];
                let mut out = Vec::with_capacity(2);
                for c in cs {
                    if c & 8 != 0 {
                        out.push("zeroinitializer".to_string());
                    } else {
                        let slot = (c & 3) as usize;
                        if loaded[slot].is_none() {
                            let op = if slot < 2 { 1 } else { 2 };
                            loaded[slot] = Some(self.v_half(ins, op, (slot & 1) as u32, I64)?);
                        }
                        out.push(loaded[slot].clone().unwrap());
                    }
                }
                if need_mem2 && loaded[2].is_none() && loaded[3].is_none() {
                    let _ = self.v_half(ins, 2, 0, I64)?;
                }
                self.set_halves(ins.op_register(0), &out[0], Some(&out[1]), I64)?;
            }
            M::Vpermq | M::Vpermpd => {
                let i = Self::imm8(ins)?;
                let idx: [u32; 4] = std::array::from_fn(|l| (i >> (2 * l)) & 3);
                let need_lo = idx.iter().any(|&k| k < 2) || ins.op_kind(1) == OpKind::Memory;
                let need_hi = idx.iter().any(|&k| k >= 2);
                let lo = if need_lo {
                    Some(self.v_half(ins, 1, 0, I64)?)
                } else {
                    None
                };
                let hi = if need_hi {
                    Some(self.v_half(ins, 1, 1, I64)?)
                } else {
                    None
                };
                let mut build_half = |pair: [u32; 2]| -> String {
                    if is_identity_mask(&pair) {
                        lo.clone().unwrap()
                    } else if pair == [2, 3] {
                        hi.clone().unwrap()
                    } else if pair[0] < 2 && pair[1] < 2 {
                        self.shuffle(lo.as_deref().unwrap(), "poison", I64, &pair)
                    } else if pair[0] >= 2 && pair[1] >= 2 {
                        self.shuffle(
                            hi.as_deref().unwrap(),
                            "poison",
                            I64,
                            &[pair[0] - 2, pair[1] - 2],
                        )
                    } else {
                        self.shuffle(lo.as_deref().unwrap(), hi.as_deref().unwrap(), I64, &pair)
                    }
                };
                let out_lo = build_half([idx[0], idx[1]]);
                let out_hi = if [idx[2], idx[3]] == [idx[0], idx[1]] {
                    out_lo.clone()
                } else {
                    build_half([idx[2], idx[3]])
                };
                self.set_halves(ins.op_register(0), &out_lo, Some(&out_hi), I64)?;
            }
            M::Vpermd | M::Vpermps => {
                // vpermd ymm1, ymm2 (indices), ymm3/m256 (data):
                // Lower via two `i8x16.swizzle` ops per 128-bit half (zero stack spills!).
                // Byte indices in 0..15 hit `dlo` (and `idx ^ 16` in 16..31 zeroes `dhi`);
                // byte indices in 16..31 zero `dlo` (and `idx ^ 16` in 0..15 hits `dhi`).
                let dlo = self.v_half(ins, 2, 0, I8)?;
                let dhi = self.v_half(ins, 2, 1, I8)?;
                if dlo == "zeroinitializer" && dhi == "zeroinitializer" {
                    self.set_halves(
                        ins.op_register(0),
                        "zeroinitializer",
                        Some("zeroinitializer"),
                        I8,
                    )?;
                    return Ok(true);
                }
                let xor16 = Ty::v(I8).splat("16");
                let mut out = Vec::with_capacity(2);
                for half in 0..2 {
                    let idx_h = self.v_half(ins, 1, half, I32)?;
                    let lo_idx = self.dword_swizzle_indices(&idx_h, 7);
                    let hi_idx = self.val(&format!("xor <16 x i8> {lo_idx}, {xor16}"));
                    let rlo = self.intr(Intr::Swizzle, Ty::v(I8), &[&dlo, &lo_idx]);
                    let rhi = self.intr(Intr::Swizzle, Ty::v(I8), &[&dhi, &hi_idx]);
                    out.push(self.val(&format!("or <16 x i8> {rlo}, {rhi}")));
                }
                self.set_halves(ins.op_register(0), &out[0], Some(&out[1]), I8)?;
            }
            M::Vmovmskps | M::Vmovmskpd | M::Vpmovmskb if wide => {
                let e = match m {
                    M::Vmovmskps => I32,
                    M::Vmovmskpd => I64,
                    _ => I8,
                };
                // One native Wasm SIMD128 `bitmask` per 128-bit half, shifted and ORed.
                let n = e.lanes();
                let dst = self.op_bits(ins, 0)?;
                let v0 = self.v_half(ins, 1, 0, e)?;
                let lo = self.intr(Intr::Bitmask, Ty::v(e), &[&v0]);
                let v1 = self.v_half(ins, 1, 1, e)?;
                let hi = self.intr(Intr::Bitmask, Ty::v(e), &[&v1]);
                let hi_sh = self.val(&format!("shl i32 {hi}, {n}"));
                let combined = self.val(&format!("or i32 {lo}, {hi_sh}"));
                let r = if dst > 32 {
                    self.val(&format!("zext i32 {combined} to i{dst}"))
                } else {
                    combined
                };
                self.write_op(ins, 0, &r, dst)?;
            }
            M::Ptest | M::Vptest => self.test(ins, None)?,
            M::Vtestps => self.test(ins, Some(I32))?,
            M::Vtestpd => self.test(ins, Some(I64))?,
            M::Vmaskmovps | M::Vpmaskmovd => self.masked_move(ins, I32)?,
            M::Vmaskmovpd | M::Vpmaskmovq => self.masked_move(ins, I64)?,
            M::Pmovsxbw | M::Vpmovsxbw => self.extend(ins, I8, I16, "sext")?,
            M::Pmovsxbd | M::Vpmovsxbd => self.extend(ins, I8, I32, "sext")?,
            M::Pmovsxbq | M::Vpmovsxbq => self.extend(ins, I8, I64, "sext")?,
            M::Pmovsxwd | M::Vpmovsxwd => self.extend(ins, I16, I32, "sext")?,
            M::Pmovsxwq | M::Vpmovsxwq => self.extend(ins, I16, I64, "sext")?,
            M::Pmovsxdq | M::Vpmovsxdq => self.extend(ins, I32, I64, "sext")?,
            M::Pmovzxbw | M::Vpmovzxbw => self.extend(ins, I8, I16, "zext")?,
            M::Pmovzxbd | M::Vpmovzxbd => self.extend(ins, I8, I32, "zext")?,
            M::Pmovzxbq | M::Vpmovzxbq => self.extend(ins, I8, I64, "zext")?,
            M::Pmovzxwd | M::Vpmovzxwd => self.extend(ins, I16, I32, "zext")?,
            M::Pmovzxwq | M::Vpmovzxwq => self.extend(ins, I16, I64, "zext")?,
            M::Pmovzxdq | M::Vpmovzxdq => self.extend(ins, I32, I64, "zext")?,
            M::Vcvtps2pd | M::Vcvtdq2pd if wide => {
                let src = if m == M::Vcvtps2pd { F32 } else { I32 };
                let op = if m == M::Vcvtps2pd { "fpext" } else { "sitofp" };
                let v = self.v_half(ins, 1, 0, src)?;
                if v == "zeroinitializer" {
                    self.set_halves(
                        ins.op_register(0),
                        "zeroinitializer",
                        Some("zeroinitializer"),
                        F64,
                    )?;
                    return Ok(true);
                }
                let st = Ty::v(src).ir();
                let ht = Ty::vn(src, 2).ir();
                // Split into two native `<2 x src> -> <2 x double>` halves so LLVM emits
                // `f64x2.promote_low_f32x4` / `f64x2.convert_low_i32x4_s` without `<4 x double>`.
                let v_lo = self.val(&format!(
                    "shufflevector {st} {v}, {st} poison, <2 x i32> <i32 0, i32 1>"
                ));
                let v_hi = self.val(&format!(
                    "shufflevector {st} {v}, {st} poison, <2 x i32> <i32 2, i32 3>"
                ));
                let lo = self.val(&format!("{op} {ht} {v_lo} to <2 x double>"));
                let hi = self.val(&format!("{op} {ht} {v_hi} to <2 x double>"));
                self.set_halves(ins.op_register(0), &lo, Some(&hi), F64)?;
            }
            M::Vcvtpd2ps if wide => {
                let lo = self.v_half(ins, 1, 0, F64)?;
                let hi = self.v_half(ins, 1, 1, F64)?;
                if lo == "zeroinitializer" && hi == "zeroinitializer" {
                    self.xset_half(ins.op_register(0), 0, "zeroinitializer", I64)?;
                    return Ok(true);
                }
                // Demote each `<2 x double>` half via `zext_lanes` and combine through `<2 x i64>`
                // so LLVM keeps each half as `f32x4.demote_f64x2_zero` + one `i64x2.shuffle`.
                let rlo = self.val(&format!("fptrunc <2 x double> {lo} to <2 x float>"));
                let rhi = self.val(&format!("fptrunc <2 x double> {hi} to <2 x float>"));
                let zlo = self.zext_lanes(&rlo, F32, 2);
                let zhi = self.zext_lanes(&rhi, F32, 2);
                let lo64 = self.cast(&zlo, Ty::v(F32), Ty::v(I64));
                let hi64 = self.cast(&zhi, Ty::v(F32), Ty::v(I64));
                let hi_swap = self.shuffle(&hi64, "poison", I64, &[1, 0]);
                let r = self.val(&format!("or <2 x i64> {lo64}, {hi_swap}"));
                self.xset_half(ins.op_register(0), 0, &r, I64)?;
            }
            M::Vcvtpd2dq | M::Vcvttpd2dq if wide => {
                // Lower each `<2 x double>` half identically to 128-bit `cvt(t)pd2dq` using
                // `@llvm.wasm.bitselect.v4i32` so both halves emit native
                // `i32x4.trunc_sat_f64x2_s_zero` with zero scalarization or mask legalization.
                let truncate = m == M::Vcvttpd2dq;
                let limit = Ty::v(F64).splat(&fconst(2f64.powi(31)));
                let mut halves = Vec::with_capacity(2);
                for half in 0..2 {
                    let v = self.v_half(ins, 1, half, F64)?;
                    let x = if truncate {
                        v
                    } else {
                        self.intr(Intr::Rint, Ty::v(F64), &[&v])
                    };
                    let sat2 = self.intr(Intr::FpToSiSat(I32), Ty::v(F64), &[&x]);
                    let sat4 = self.zext_lanes(&sat2, I32, 2);
                    let ok2 = self.val(&format!("fcmp olt <2 x double> {x}, {limit}"));
                    let ok64 = self.val(&format!("sext <2 x i1> {ok2} to <2 x i64>"));
                    let ok32 = self.cast(&ok64, Ty::v(I64), Ty::v(I32));
                    let mask = self.shuffle(&ok32, "poison", I32, &[0, 2, 0, 2]);
                    let r4 = self.intr(
                        Intr::Bitselect,
                        Ty::v(I32),
                        &[
                            &sat4,
                            "<i32 -2147483648, i32 -2147483648, i32 0, i32 0>",
                            &mask,
                        ],
                    );
                    halves.push(self.cast(&r4, Ty::v(I32), Ty::v(I64)));
                }
                let hi_swap = self.shuffle(&halves[1], "poison", I64, &[1, 0]);
                let r = self.val(&format!("or <2 x i64> {}, {hi_swap}", halves[0]));
                self.xset_half(ins.op_register(0), 0, &r, I64)?;
            }
            M::Vcvtph2ps => {
                if ins.op_register(0).is_ymm() {
                    // Split `<8 x i16>` into lo/hi `<4 x i16>` -> `<4 x i32>` halves (matching
                    // `i32x4.extend_low_i16x8_u` / `i32x4.extend_high_i16x8_u`) to stay 128-bit.
                    let src = self.v_half(ins, 1, 0, I16)?;
                    let lo16 = self.val(&format!(
                        "shufflevector <8 x i16> {src}, <8 x i16> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>"
                    ));
                    let hi16 = self.val(&format!(
                        "shufflevector <8 x i16> {src}, <8 x i16> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>"
                    ));
                    let h_lo = self.val(&format!("zext <4 x i16> {lo16} to <4 x i32>"));
                    let h_hi = self.val(&format!("zext <4 x i16> {hi16} to <4 x i32>"));
                    let r_lo = self.half_to_float_v4(&h_lo);
                    let r_hi = self.half_to_float_v4(&h_hi);
                    self.set_halves(ins.op_register(0), &r_lo, Some(&r_hi), I32)?;
                } else {
                    let h16 = match ins.op_kind(1) {
                        OpKind::Register => {
                            let v = self.xget_half(ins.op_register(1), 0, I16)?;
                            self.take_low(&v, I16, 4)
                        }
                        _ => self.mem_load_at(ins, Ty::vn(I16, 4), 0, 8)?,
                    };
                    let h = self.val(&format!("zext <4 x i16> {h16} to <4 x i32>"));
                    let r = self.half_to_float_v4(&h);
                    self.xset_half(ins.op_register(0), 0, &r, I32)?;
                }
            }
            M::Vcvtps2ph => {
                let i = Self::imm8(ins)?;
                let mode = if i & 4 != 0 {
                    Round::Nearest
                } else {
                    [Round::Nearest, Round::Down, Round::Up, Round::Zero][(i & 3) as usize]
                };
                if ins.op_register(1).is_ymm() {
                    // Convert each 128-bit `<4 x i32>` half natively and pack low 16 bits into
                    // `<8 x i16>` with a single `shufflevector`.
                    let v_lo = self.v_half(ins, 1, 0, I32)?;
                    let v_hi = self.v_half(ins, 1, 1, I32)?;
                    let h_lo = self.float_to_half_v4(&v_lo, mode);
                    let h_hi = self.float_to_half_v4(&v_hi, mode);
                    let lo16 = self.cast(&h_lo, Ty::v(I32), Ty::v(I16));
                    let hi16 = self.cast(&h_hi, Ty::v(I32), Ty::v(I16));
                    let h8 = self.shuffle(&lo16, &hi16, I16, &[0, 2, 4, 6, 8, 10, 12, 14]);
                    match ins.op_kind(0) {
                        OpKind::Register => self.xset_half(ins.op_register(0), 0, &h8, I16)?,
                        _ => self.mem_store_at(ins, &h8, Ty::v(I16), 0, 16)?,
                    }
                } else {
                    let v = self.v_half(ins, 1, 0, I32)?;
                    let h = self.float_to_half_v4(&v, mode);
                    match ins.op_kind(0) {
                        OpKind::Register => {
                            // Pack low 16 bits of each lane + zero upper 64 bits in one
                            // `shufflevector` (2 Wasm ops vs 9 ops with `replace_lane`).
                            let h16 = self.cast(&h, Ty::v(I32), Ty::v(I16));
                            let r = self.shuffle(
                                &h16,
                                "zeroinitializer",
                                I16,
                                &[0, 2, 4, 6, 8, 8, 8, 8],
                            );
                            self.xset_half(ins.op_register(0), 0, &r, I16)?;
                        }
                        _ => {
                            let h4 = self.val(&format!("trunc <4 x i32> {h} to <4 x i16>"));
                            self.mem_store_at(ins, &h4, Ty::vn(I16, 4), 0, 8)?;
                        }
                    }
                }
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// `vbroadcastss/sd`, `vpbroadcastb/w/d/q`: the low element of an XMM register or memory in
    /// every lane.
    fn broadcast(&mut self, ins: &Instruction, e: Elem) -> R<()> {
        let v = match ins.op_kind(1) {
            OpKind::Register => {
                let src = self.xget_half(ins.op_register(1), 0, e)?;
                if src == "zeroinitializer" {
                    src
                } else {
                    let t = Ty::v(e).ir();
                    self.val(&format!(
                        "shufflevector {t} {src}, {t} poison, <{} x i32> zeroinitializer",
                        e.lanes()
                    ))
                }
            }
            _ => {
                let s = self.mem_load_at(ins, Ty::s(e), 0, (e.bits() / 8) as usize)?;
                self.splat_value(&s, e)
            }
        };
        self.set_halves(ins.op_register(0), &v, Some(&v), e)
    }

    /// `ptest/vptest` (`sign == None`) and `vtestps/vtestpd` (sign bits of `sign` lanes only):
    /// ZF = (a & b) == 0, CF = (!a & b) == 0, the other flags cleared.
    /// Guards ZF/CF computation with `flag_live` and uses native `@llvm.wasm.bitmask.*` for
    /// `vtestps`/`vtestpd` (eliminating 18-byte sign-mask constants and extra `v128.and`).
    fn test(&mut self, ins: &Instruction, sign: Option<Elem>) -> R<()> {
        let wide = is_256(ins);
        let t = Ty::v(Elem::I64).ir();
        let same = same_reg(ins, 0, 1);
        let want_zf = self.flag_live(Flag::Zf);
        let want_cf = self.flag_live(Flag::Cf);
        let mut acc_and = String::new();
        let mut acc_andn = String::new();
        let halves = if wide { 2 } else { 1 };
        if want_zf || (want_cf && !same) {
            for half in 0..halves {
                let a = self.v_half(ins, 0, half, Elem::I64)?;
                if same {
                    if want_zf {
                        acc_and = if half == 0 {
                            a
                        } else {
                            self.val(&format!("or {t} {acc_and}, {a}"))
                        };
                    }
                    continue;
                }
                let b = self.v_half(ins, 1, half, Elem::I64)?;
                if want_zf {
                    let v_and = self.val(&format!("and {t} {a}, {b}"));
                    acc_and = if half == 0 {
                        v_and
                    } else {
                        self.val(&format!("or {t} {acc_and}, {v_and}"))
                    };
                }
                if want_cf {
                    let na = self.val(&format!("xor {t} {a}, <i64 -1, i64 -1>"));
                    let v_andn = self.val(&format!("and {t} {na}, {b}"));
                    acc_andn = if half == 0 {
                        v_andn
                    } else {
                        self.val(&format!("or {t} {acc_andn}, {v_andn}"))
                    };
                }
            }
        } else if ins.op_kind(1) == OpKind::Memory {
            // Preserve memory access fault semantics even when both ZF and CF are dead.
            let _ = self.v_half(ins, 1, 0, Elem::I64)?;
        }
        let is_all_zero = |l: &mut Self, v: String| {
            if let Some(se) = sign {
                let cast = l.cast(&v, Ty::v(Elem::I64), Ty::v(se));
                let bm = l.intr(Intr::Bitmask, Ty::v(se), &[&cast]);
                l.val(&format!("icmp eq i32 {bm}, 0"))
            } else {
                let nz = l.val(&format!("icmp ne {t} {v}, zeroinitializer"));
                let bits = l.val(&format!("bitcast <2 x i1> {nz} to i2"));
                l.val(&format!("icmp eq i2 {bits}, 0"))
            }
        };
        if want_zf {
            let zf = is_all_zero(self, acc_and);
            self.set_flag(Flag::Zf, &zf);
        }
        if want_cf {
            let cf = if same {
                "true".to_string()
            } else {
                is_all_zero(self, acc_andn)
            };
            self.set_flag(Flag::Cf, &cf);
        }
        for flag in [Flag::Af, Flag::Of, Flag::Pf, Flag::Sf] {
            self.set_flag(flag, "false");
        }
        Ok(())
    }

    /// `vmaskmovps/pd`, `vpmaskmovd/q`: per-element loads/stores of the elements whose mask sign
    /// bit is set. Masked-out elements are never accessed (they cannot fault); loads zero them.
    /// Processes each 128-bit half natively with `@llvm.wasm.bitmask.*` (no 256-bit CFG live
    /// ranges and no empty `mpre` jump blocks).
    fn masked_move(&mut self, ins: &Instruction, e: Elem) -> R<()> {
        let load = ins.op_kind(0) == OpKind::Register;
        // load: dst, mask, mem ; store: mem, mask, src
        let wide = is_256(ins);
        let halves = if wide { 2u32 } else { 1u32 };
        let lanes = e.lanes();
        let vt = Ty::v(e).ir();
        let st = e.scalar();
        let bytes = u64::from(e.bits() / 8);
        let total = (halves as u64 * lanes as u64 * bytes) as usize;
        if Self::mem_bytes(ins) != total {
            return unsupported("internal: masked move memory size");
        }
        // Pre-read mask (and store source) halves before branching so `dst == mask` stays safe.
        let mut mask_bms = Vec::with_capacity(halves as usize);
        let mut src_halves = Vec::with_capacity(if load { 0 } else { halves as usize });
        for half in 0..halves {
            let mh = self.v_half(ins, 1, half, e)?;
            mask_bms.push(self.intr(Intr::Bitmask, Ty::v(e), &[&mh]));
            if !load {
                src_halves.push(self.v_half(ins, 2, half, e)?);
            }
        }
        // The address is computed once, in a block dominating every element access.
        self.mem_ea(ins)?;
        let mut out_halves = Vec::with_capacity(if load { halves as usize } else { 0 });
        for half in 0..halves {
            let bm = &mask_bms[half as usize];
            let mut acc = "zeroinitializer".to_string();
            for l in 0..lanes {
                let c = if l == 0 {
                    self.val(&format!("trunc i32 {bm} to i1"))
                } else {
                    let bit = self.val(&format!("and i32 {bm}, {}", 1u32 << l));
                    self.val(&format!("icmp ne i32 {bit}, 0"))
                };
                let pre = self.current_block.clone();
                let act = self.new_label("mact");
                let cont = self.new_label("mcont");
                self.emit(&format!("br i1 {c}, label %{act}, label %{cont}"));
                self.start_block(&act);
                let byte_off = u64::from(half * lanes + l) * bytes;
                let ea = self.mem_ea_at(ins, byte_off)?;
                let loaded = if load {
                    Some(self.guest_load(&ea, st, bytes))
                } else {
                    let src_h = &src_halves[half as usize];
                    if self.accesses_branch() {
                        let x = self.val(&format!("extractelement {vt} {src_h}, i32 {l}"));
                        self.guest_store(&ea, st, &x, bytes);
                    } else {
                        let ptr = self.guest_ptr(&ea);
                        let x = self.val(&format!("extractelement {vt} {src_h}, i32 {l}"));
                        self.emit(&format!("store {st} {x}, ptr {ptr}, align 1"));
                    }
                    None
                };
                // A table memory model ends the access in a block of its own (R19).
                let from = if self.accesses_branch() {
                    let end = self.new_label("mend");
                    self.emit(&format!("br label %{end}"));
                    self.start_block(&end);
                    end
                } else {
                    act
                };
                self.emit(&format!("br label %{cont}"));
                self.start_block(&cont);
                if let Some(x) = loaded {
                    let v = self.val(&format!("phi {st} [ {x}, %{from} ], [ 0, %{pre} ]"));
                    acc = self.val(&format!("insertelement {vt} {acc}, {st} {v}, i32 {l}"));
                }
            }
            if load {
                out_halves.push(acc);
            }
        }
        if load {
            let reg = ins.op_register(0);
            if wide {
                self.set_halves(reg, &out_halves[0], Some(&out_halves[1]), e)?;
            } else {
                self.xset_half(reg, 0, &out_halves[0], e)?;
            }
        }
        Ok(())
    }

    /// `pmovsx*/pmovzx*`: the low elements of an XMM register (or exactly-sized memory) widened
    /// into a 128- or 256-bit destination.
    ///
    /// For 256-bit destinations, stages widening in 128-bit power-of-2 steps so LLVM emits native
    /// `i16x8.load8x8_*` / `i32x4.load16x4_*` + `extend_low`/`extend_high` pairs with zero
    /// cross-lane `i8x16.shuffle`s.
    fn extend(&mut self, ins: &Instruction, from: Elem, to: Elem, op: &str) -> R<()> {
        let dst = ins.op_register(0);
        if !dst.is_ymm() {
            let n = to.lanes();
            let v = match ins.op_kind(1) {
                OpKind::Register => {
                    let x = self.xget_half(ins.op_register(1), 0, from)?;
                    if x == "zeroinitializer" {
                        return self.xset_half(dst, 0, "zeroinitializer", to);
                    }
                    self.take_low(&x, from, n)
                }
                _ => self.mem_load_at(ins, Ty::vn(from, n), 0, (n * from.bits() / 8) as usize)?,
            };
            let w = self.val(&format!(
                "{op} {} {v} to {}",
                Ty::vn(from, n).ir(),
                Ty::vn(to, n).ir()
            ));
            return self.xset_half(dst, 0, &w, to);
        }
        // 256-bit destination: fan out into two 128-bit halves without 256-bit intermediates.
        let n = 2 * to.lanes();
        let mut cur_elem = from;
        let mut cur = match ins.op_kind(1) {
            OpKind::Register => {
                let x = self.xget_half(ins.op_register(1), 0, from)?;
                if x == "zeroinitializer" {
                    return self.set_halves(dst, "zeroinitializer", Some("zeroinitializer"), to);
                }
                self.take_low(&x, from, n)
            }
            _ => self.mem_load_at(ins, Ty::vn(from, n), 0, (n * from.bits() / 8) as usize)?,
        };
        // Step up to the half-width element of `to` (always <= 128 bits for `n` lanes).
        let penult = Elem::of_bits(to.bits() / 2);
        while cur_elem.bits() < penult.bits() {
            let next_elem = Elem::of_bits(cur_elem.bits() * 2);
            cur = self.val(&format!(
                "{op} {} {cur} to {}",
                Ty::vn(cur_elem, n).ir(),
                Ty::vn(next_elem, n).ir()
            ));
            cur_elem = next_elem;
        }
        // `cur` is now a single 128-bit vector `<n x penult>`; split into lo/hi `<half_n x penult>`
        // and extend each directly to `<half_n x to>` (`extend_low` + `extend_high`).
        let half_n = to.lanes();
        let cur_t = Ty::vn(penult, n).ir();
        let half_t = Ty::vn(penult, half_n).ir();
        let dst_t = Ty::v(to).ir();
        let lo_idx: Vec<u32> = (0..half_n).collect();
        let hi_idx: Vec<u32> = (half_n..n).collect();
        let lo_in = self.val(&format!(
            "shufflevector {cur_t} {cur}, {cur_t} poison, {}",
            mask(&lo_idx)
        ));
        let hi_in = self.val(&format!(
            "shufflevector {cur_t} {cur}, {cur_t} poison, {}",
            mask(&hi_idx)
        ));
        let lo = self.val(&format!("{op} {half_t} {lo_in} to {dst_t}"));
        let hi = self.val(&format!("{op} {half_t} {hi_in} to {dst_t}"));
        self.set_halves(dst, &lo, Some(&hi), to)
    }

    /// binary16 (zero-extended in `<4 x i32>`) -> binary32 bits, bit-exact across all 65,536
    /// inputs (signalling NaNs are quieted like hardware does).
    fn half_to_float_v4(&mut self, h: &str) -> String {
        let ty = Ty::v(Elem::I32);
        let fty = Ty::v(Elem::F32);
        let t = "<4 x i32>";
        let bt = "<4 x i1>";
        let c = |v: i64| ty.splat(&v.to_string());
        let s16 = self.val(&format!("and {t} {h}, {}", c(0x8000)));
        let sign = self.val(&format!("shl {t} {s16}, {}", c(16)));
        let abs = self.val(&format!("xor {t} {h}, {s16}"));
        // Normals: `(abs << 13) + (112 << 23)`. Inf/NaN (`abs >= 0x7C00`) add a second `112 << 23`
        // (143 + 112 = 255 = 0xFF); NaN (`abs > 0x7C00`) also sets the quiet bit `0x0040_0000`.
        let sh = self.val(&format!("shl {t} {abs}, {}", c(13)));
        let bias = c(0x3800_0000);
        let base = self.val(&format!("add {t} {sh}, {bias}"));
        let is_spec = self.val(&format!("icmp sgt {t} {abs}, {}", c(0x7BFF)));
        let is_nan = self.val(&format!("icmp sgt {t} {abs}, {}", c(0x7C00)));
        let spec_m = self.val(&format!("sext {bt} {is_spec} to {t}"));
        let nan_m = self.val(&format!("sext {bt} {is_nan} to {t}"));
        let exp_fix = self.val(&format!("and {t} {spec_m}, {bias}"));
        let nan_fix = self.val(&format!("and {t} {nan_m}, {}", c(0x0040_0000)));
        let ns = self.val(&format!("add {t} {base}, {exp_fix}"));
        let ns = self.val(&format!("or {t} {ns}, {nan_fix}"));
        // Zero and denormals (`abs < 0x0400`): `abs * 2^-24` is exact in binary32.
        let f = self.val(&format!("sitofp {t} {abs} to <4 x float>"));
        let tiny_f = self.val(&format!(
            "fmul <4 x float> {f}, {}",
            fty.splat(&fconst(2f64.powi(-24)))
        ));
        let tiny = self.cast(&tiny_f, fty, ty);
        let is_sub = self.val(&format!("icmp slt {t} {abs}, {}", c(0x0400)));
        let mag = self.val(&format!("select {bt} {is_sub}, {t} {tiny}, {t} {ns}"));
        self.val(&format!("or {t} {mag}, {sign}"))
    }

    /// binary32 bits (`<4 x i32>`) -> binary16 in the low 16 bits of each lane, rounded with
    /// `mode`, bit-exact with x86 F16C `vcvtps2ph` (NaNs are quieted and keep top payload bits).
    fn float_to_half_v4(&mut self, x: &str, mode: Round) -> String {
        let ty = Ty::v(Elem::I32);
        let fty = Ty::v(Elem::F32);
        let t = "<4 x i32>";
        let bt = "<4 x i1>";
        let c = |v: i64| ty.splat(&v.to_string());
        let sign32 = self.val(&format!("and {t} {x}, {}", ty.splat(&imm(0x8000_0000, 32))));
        let sign16 = self.val(&format!("lshr {t} {sign32}, {}", c(16)));
        let abs = self.val(&format!("xor {t} {x}, {sign32}"));
        let sh13 = self.val(&format!("lshr {t} {abs}, {}", c(13)));
        let abs_f = self.cast(&abs, ty, fty);
        let is_sub = self.val(&format!("icmp slt {t} {abs}, {}", c(0x3880_0000)));
        let is_over = self.val(&format!("icmp sgt {t} {abs}, {}", c(0x477F_FFFF)));
        let is_nan = self.val(&format!("icmp sgt {t} {abs}, {}", c(0x7F80_0000)));
        let nan_p = self.val(&format!("and {t} {sh13}, {}", c(0x03FF)));
        let nan_h = self.val(&format!("or {t} {nan_p}, {}", c(0x7E00)));

        let r = if mode == Round::Nearest {
            // Normal range [2^-14, 65536): round-to-nearest-even via single-constant bias.
            // Values >= 65520.0 carry cleanly into 0x7C00 (+inf).
            let odd = self.val(&format!("and {t} {sh13}, {}", c(1)));
            let biased = self.val(&format!(
                "add {t} {abs}, {}",
                ty.splat(&imm(0xC800_0FFF, 32))
            ));
            let biased = self.val(&format!("add {t} {biased}, {odd}"));
            let norm = self.val(&format!("lshr {t} {biased}, {}", c(13)));
            // Subnormal/zero range [0, 2^-14): adding 0.5f32 (ulp = 2^-24) rounds with exact
            // IEEE round-to-nearest-even directly onto the binary16 subnormal grid (and carries
            // 1023.5*2^-24.. into 0x0400, the smallest normal half).
            let sub_f = self.val(&format!(
                "fadd <4 x float> {abs_f}, {}",
                fty.splat(&fconst(0.5))
            ));
            let sub_bits = self.cast(&sub_f, fty, ty);
            let sub_h = self.val(&format!("sub {t} {sub_bits}, {}", c(0x3F00_0000)));
            let r = self.val(&format!("select {bt} {is_sub}, {t} {sub_h}, {t} {norm}"));
            let r = self.val(&format!("select {bt} {is_over}, {t} {}, {t} {r}", c(0x7C00)));
            self.val(&format!("select {bt} {is_nan}, {t} {nan_h}, {t} {r}"))
        } else {
            // Directed / truncate rounding modes:
            // - Subnormal/zero: `|x| * 2^24` is exact in binary32 (even for f32 subnormals:
            //   `2^-149 * 2^24 = 2^-125` is normal), so `ceil` + `fptosi.sat` gives exact
            //   subnormal bits with zero variable shifts.
            // - Normal: truncate `(abs - 0x3800_0000) >> 13`, plus 1 when rounding away & rem != 0.
            let unbias = self.val(&format!("sub {t} {abs}, {}", c(0x3800_0000)));
            let trunc_h = self.val(&format!("lshr {t} {unbias}, {}", c(13)));
            let scaled = self.val(&format!(
                "fmul <4 x float> {abs_f}, {}",
                fty.splat(&fconst(2f64.powi(24)))
            ));
            let (norm, sub_h, overflow) = match mode {
                Round::Zero => {
                    let sub_h = self.intr(Intr::FpToSiSat(Elem::I32), fty, &[&scaled]);
                    (trunc_h, sub_h, c(0x7BFF))
                }
                Round::Down | Round::Up => {
                    let round_away = if mode == Round::Down {
                        self.val(&format!("icmp slt {t} {x}, zeroinitializer"))
                    } else {
                        self.val(&format!("icmp sge {t} {x}, zeroinitializer"))
                    };
                    let rem = self.val(&format!("and {t} {abs}, {}", c(0x1FFF)));
                    let inexact = self.val(&format!("icmp ne {t} {rem}, zeroinitializer"));
                    let inc = self.val(&format!("and {bt} {inexact}, {round_away}"));
                    let inc_m = self.val(&format!("sext {bt} {inc} to {t}"));
                    let norm = self.val(&format!("sub {t} {trunc_h}, {inc_m}"));
                    let ceiled = self.intr(Intr::Ceil, fty, &[&scaled]);
                    let sub_f = self.val(&format!(
                        "select {bt} {round_away}, <4 x float> {ceiled}, <4 x float> {scaled}"
                    ));
                    let sub_h = self.intr(Intr::FpToSiSat(Elem::I32), fty, &[&sub_f]);
                    let overflow = self.val(&format!(
                        "select {bt} {round_away}, {t} {}, {t} {}",
                        c(0x7C00),
                        c(0x7BFF)
                    ));
                    (norm, sub_h, overflow)
                }
                Round::Nearest => unreachable!(),
            };
            let is_inf = self.val(&format!("icmp eq {t} {abs}, {}", c(0x7F80_0000)));
            let r = self.val(&format!("select {bt} {is_sub}, {t} {sub_h}, {t} {norm}"));
            let r = self.val(&format!("select {bt} {is_over}, {t} {overflow}, {t} {r}"));
            let r = self.val(&format!("select {bt} {is_inf}, {t} {}, {t} {r}", c(0x7C00)));
            self.val(&format!("select {bt} {is_nan}, {t} {nan_h}, {t} {r}"))
        };
        self.val(&format!("or {t} {r}, {sign16}"))
    }
}
