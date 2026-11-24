//! Instructions that need whole registers rather than one 128-bit lane at a time: broadcasts,
//! 128-bit inserts/extracts, cross-lane permutes, sign masks and tests of 256-bit registers,
//! masked loads/stores, widening/narrowing conversions, `pmovsx/pmovzx`, F16C half conversions.
//! Both their 128- and 256-bit forms are lifted here; [`in_lane`] lists the 256-bit forms that
//! the lane-wise lifters handle as two independent halves.

use iced_x86::{Instruction, Mnemonic, OpKind, Register};

use super::super::{Flag, FunctionLifter, R, imm, unsupported};
use super::{Elem, Ty, fconst, is_256, mask};

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

    /// Concatenates two 128-bit halves of `e` into one `<2n x e>` vector.
    fn concat(&mut self, lo: &str, hi: &str, e: Elem) -> String {
        let n = e.lanes();
        let idx: Vec<u32> = (0..2 * n).collect();
        let t = Ty::v(e).ir();
        self.val(&format!("shufflevector {t} {lo}, {t} {hi}, {}", mask(&idx)))
    }

    /// Splits a `<2n x e>` vector into its two 128-bit halves.
    fn split(&mut self, v: &str, e: Elem) -> (String, String) {
        let n = e.lanes();
        let t = Ty::vn(e, 2 * n).ir();
        let lo: Vec<u32> = (0..n).collect();
        let hi: Vec<u32> = (n..2 * n).collect();
        let a = self.val(&format!("shufflevector {t} {v}, {t} poison, {}", mask(&lo)));
        let b = self.val(&format!("shufflevector {t} {v}, {t} poison, {}", mask(&hi)));
        (a, b)
    }

    /// The whole 128/256-bit operand `i` as `<n x e>` (n = 16 or 32 bytes worth).
    fn whole(&mut self, ins: &Instruction, i: u32, e: Elem) -> R<(String, bool)> {
        let wide = match ins.op_kind(i) {
            OpKind::Register => ins.op_register(i).is_ymm(),
            OpKind::Memory => Self::mem_bytes(ins) == 32,
            k => return unsupported(format!("vector operand kind {k:?}")),
        };
        let lo = self.v_half(ins, i, 0, e)?;
        if !wide {
            return Ok((lo, false));
        }
        let hi = self.v_half(ins, i, 1, e)?;
        Ok((self.concat(&lo, &hi, e), true))
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
                let lo = self.xget_half(ins.op_register(1), 0, I64)?;
                let hi = self.xget_half(ins.op_register(1), 1, I64)?;
                let x = match ins.op_kind(2) {
                    OpKind::Register => self.xget_half(ins.op_register(2), 0, I64)?,
                    _ => self.mem_load_at(ins, Ty::v(I64), 0, 16)?,
                };
                let (lo, hi) = if sel == 0 { (x, hi) } else { (lo, x) };
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
                let srcs = [
                    self.v_half(ins, 1, 0, I64)?,
                    self.v_half(ins, 1, 1, I64)?,
                    self.v_half(ins, 2, 0, I64)?,
                    self.v_half(ins, 2, 1, I64)?,
                ];
                let pick = |c: u32| -> String {
                    if c & 8 != 0 {
                        "zeroinitializer".to_string()
                    } else {
                        srcs[(c & 3) as usize].clone()
                    }
                };
                let (lo, hi) = (pick(i & 0xF), pick(i >> 4));
                self.set_halves(ins.op_register(0), &lo, Some(&hi), I64)?;
            }
            M::Vpermq | M::Vpermpd => {
                let i = Self::imm8(ins)?;
                let (v, _) = self.whole(ins, 1, I64)?;
                let idx: Vec<u32> = (0..4).map(|l| (i >> (2 * l)) & 3).collect();
                let t = Ty::vn(I64, 4).ir();
                let r = self.val(&format!(
                    "shufflevector {t} {v}, {t} poison, {}",
                    mask(&idx)
                ));
                let (lo, hi) = self.split(&r, I64);
                self.set_halves(ins.op_register(0), &lo, Some(&hi), I64)?;
            }
            M::Vpermd | M::Vpermps => {
                // vpermd ymm1, ymm2 (indices), ymm3/m256 (data)
                let (idx, _) = self.whole(ins, 1, I32)?;
                let (data, _) = self.whole(ins, 2, I32)?;
                let t = Ty::vn(I32, 8).ir();
                let mut r = "poison".to_string();
                for l in 0..8 {
                    let k = self.val(&format!("extractelement {t} {idx}, i32 {l}"));
                    let k = self.val(&format!("and i32 {k}, 7"));
                    let x = self.val(&format!("extractelement {t} {data}, i32 {k}"));
                    r = self.val(&format!("insertelement {t} {r}, i32 {x}, i32 {l}"));
                }
                let (lo, hi) = self.split(&r, I32);
                self.set_halves(ins.op_register(0), &lo, Some(&hi), I32)?;
            }
            M::Vmovmskps | M::Vmovmskpd | M::Vpmovmskb if wide => {
                let e = match m {
                    M::Vmovmskps => I32,
                    M::Vmovmskpd => I64,
                    _ => I8,
                };
                // One sign mask per 128-bit half (a single `<32 x i1>` bitcast is miscompiled
                // by the wasm backend of some LLVM versions).
                let n = e.lanes();
                let dst = self.op_bits(ins, 0)?;
                let mut r = "0".to_string();
                for half in 0..2 {
                    let v = self.v_half(ins, 1, half, e)?;
                    let neg = self.val(&format!("icmp slt {} {v}, zeroinitializer", Ty::v(e).ir()));
                    let bits = self.val(&format!("bitcast <{n} x i1> {neg} to i{n}"));
                    let bits = self.val(&format!("zext i{n} {bits} to i{dst}"));
                    let bits = self.val(&format!("shl i{dst} {bits}, {}", half * n));
                    r = self.val(&format!("or i{dst} {r}, {bits}"));
                }
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
                let v = self.v_half(ins, 1, 0, src)?;
                let op = if m == M::Vcvtps2pd { "fpext" } else { "sitofp" };
                let r = self.val(&format!("{op} {} {v} to <4 x double>", Ty::v(src).ir()));
                let (lo, hi) = self.split(&r, F64);
                self.set_halves(ins.op_register(0), &lo, Some(&hi), F64)?;
            }
            M::Vcvtpd2ps | M::Vcvtpd2dq | M::Vcvttpd2dq if wide => {
                let (v, _) = self.whole(ins, 1, F64)?;
                let r = if m == M::Vcvtpd2ps {
                    let r = self.val(&format!("fptrunc <4 x double> {v} to <4 x float>"));
                    self.cast(&r, Ty::v(F32), Ty::v(I32))
                } else {
                    self.float_to_int(&v, Ty::vn(F64, 4), 32, m == M::Vcvttpd2dq)
                };
                self.xset_half(ins.op_register(0), 0, &r, I32)?;
            }
            M::Vcvtph2ps => {
                let n = if ins.op_register(0).is_ymm() { 8 } else { 4 };
                let h = match ins.op_kind(1) {
                    OpKind::Register => {
                        let v = self.xget_half(ins.op_register(1), 0, I16)?;
                        self.take_low(&v, I16, n)
                    }
                    _ => self.mem_load_at(ins, Ty::vn(I16, n), 0, 2 * n as usize)?,
                };
                let h = self.val(&format!("zext <{n} x i16> {h} to <{n} x i32>"));
                let r = self.half_to_float(&h, n);
                if n == 8 {
                    let (lo, hi) = self.split(&r, I32);
                    self.set_halves(ins.op_register(0), &lo, Some(&hi), I32)?;
                } else {
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
                let (v, wide_src) = self.whole(ins, 1, I32)?;
                let n = if wide_src { 8 } else { 4 };
                let h = self.float_to_half(&v, n, mode);
                let h = self.val(&format!("trunc <{n} x i32> {h} to <{n} x i16>"));
                match ins.op_kind(0) {
                    OpKind::Register => {
                        let r = self.zext_lanes(&h, I16, n);
                        self.xset_half(ins.op_register(0), 0, &r, I16)?;
                    }
                    _ => self.mem_store_at(ins, &h, Ty::vn(I16, n), 0, 2 * n as usize)?,
                }
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// `vbroadcastss/sd`, `vpbroadcastb/w/d/q`: the low element of an XMM register or memory in
    /// every lane.
    fn broadcast(&mut self, ins: &Instruction, e: Elem) -> R<()> {
        let s = match ins.op_kind(1) {
            OpKind::Register => {
                let v = self.xget_half(ins.op_register(1), 0, e)?;
                self.extract(&v, e, 0)
            }
            _ => self.mem_load_at(ins, Ty::s(e), 0, (e.bits() / 8) as usize)?,
        };
        let v = self.splat_value(&s, e);
        self.set_halves(ins.op_register(0), &v, Some(&v), e)
    }

    /// `ptest/vptest` (`sign == None`) and `vtestps/vtestpd` (sign bits of `sign` lanes only):
    /// ZF = (a & b) == 0, CF = (!a & b) == 0, the other flags cleared.
    fn test(&mut self, ins: &Instruction, sign: Option<Elem>) -> R<()> {
        let wide = is_256(ins);
        let t = Ty::v(Elem::I64).ir();
        let sign_mask = match sign {
            Some(Elem::I32) => Some(Ty::v(Elem::I64).splat(&imm(0x8000_0000_8000_0000, 64))),
            Some(_) => Some(Ty::v(Elem::I64).splat(&imm(0x8000_0000_0000_0000, 64))),
            None => None,
        };
        let mut any_and = "false".to_string();
        let mut any_andn = "false".to_string();
        for half in 0..if wide { 2 } else { 1 } {
            let a = self.v_half(ins, 0, half, Elem::I64)?;
            let b = self.v_half(ins, 1, half, Elem::I64)?;
            let na = self.val(&format!("xor {t} {a}, <i64 -1, i64 -1>"));
            for (x, acc) in [(a, &mut any_and), (na, &mut any_andn)] {
                let mut v = self.val(&format!("and {t} {x}, {b}"));
                if let Some(m) = &sign_mask {
                    v = self.val(&format!("and {t} {v}, {m}"));
                }
                let l0 = self.val(&format!("extractelement {t} {v}, i32 0"));
                let l1 = self.val(&format!("extractelement {t} {v}, i32 1"));
                let o = self.val(&format!("or i64 {l0}, {l1}"));
                let nz = self.val(&format!("icmp ne i64 {o}, 0"));
                *acc = self.val(&format!("or i1 {acc}, {nz}"));
            }
        }
        let zf = self.val(&format!("xor i1 {any_and}, true"));
        let cf = self.val(&format!("xor i1 {any_andn}, true"));
        self.set_flag(Flag::Zf, &zf);
        self.set_flag(Flag::Cf, &cf);
        for flag in [Flag::Af, Flag::Of, Flag::Pf, Flag::Sf] {
            self.set_flag(flag, "false");
        }
        Ok(())
    }

    /// `vmaskmovps/pd`, `vpmaskmovd/q`: per-element loads/stores of the elements whose mask sign
    /// bit is set. Masked-out elements are never accessed (they cannot fault); loads zero them.
    fn masked_move(&mut self, ins: &Instruction, e: Elem) -> R<()> {
        let load = ins.op_kind(0) == OpKind::Register;
        // load: dst, mask, mem ; store: mem, mask, src
        let (mask_v, wide) = self.whole(ins, 1, e)?;
        let n = if wide { 2 * e.lanes() } else { e.lanes() };
        let vt = Ty::vn(e, n).ir();
        let st = e.scalar();
        let bytes = u64::from(e.bits() / 8);
        let total = (n as u64 * bytes) as usize;
        if Self::mem_bytes(ins) != total {
            return unsupported("internal: masked move memory size");
        }
        let neg = self.val(&format!("icmp slt {vt} {mask_v}, zeroinitializer"));
        let src = if load {
            None
        } else {
            Some(self.whole(ins, 2, e)?.0)
        };
        // The address is computed once, in a block dominating every element access.
        self.mem_ea(ins)?;
        let mut acc = "zeroinitializer".to_string();
        for l in 0..n {
            let c = self.val(&format!("extractelement <{n} x i1> {neg}, i32 {l}"));
            let pre = self.new_label("mpre");
            let act = self.new_label("mact");
            let cont = self.new_label("mcont");
            self.emit(&format!("br label %{pre}"));
            self.start_block(&pre);
            self.emit(&format!("br i1 {c}, label %{act}, label %{cont}"));
            self.start_block(&act);
            let ea = self.mem_ea_at(ins, u64::from(l) * bytes)?;
            let loaded = if let Some(src) = &src {
                if self.accesses_branch() {
                    let x = self.val(&format!("extractelement {vt} {src}, i32 {l}"));
                    self.guest_store(&ea, st, &x, bytes);
                } else {
                    let ptr = self.guest_ptr(&ea);
                    let x = self.val(&format!("extractelement {vt} {src}, i32 {l}"));
                    self.emit(&format!("store {st} {x}, ptr {ptr}, align 1"));
                }
                None
            } else {
                Some(self.guest_load(&ea, st, bytes))
            };
            // A table memory model ends the access in a block of its own (R19).
            let from = if self.accesses_branch() {
                let end = self.new_label("mend");
                self.emit(&format!("br label %{end}"));
                self.start_block(&end);
                end
            } else {
                act.clone()
            };
            self.emit(&format!("br label %{cont}"));
            self.start_block(&cont);
            if let Some(x) = loaded {
                let v = self.val(&format!("phi {st} [ {x}, %{from} ], [ 0, %{pre} ]"));
                acc = self.val(&format!("insertelement {vt} {acc}, {st} {v}, i32 {l}"));
            }
        }
        if load {
            let reg = ins.op_register(0);
            if wide {
                let (lo, hi) = self.split(&acc, e);
                self.set_halves(reg, &lo, Some(&hi), e)?;
            } else {
                self.xset_half(reg, 0, &acc, e)?;
            }
        }
        Ok(())
    }

    /// `pmovsx*/pmovzx*`: the low elements of an XMM register (or exactly-sized memory) widened
    /// into a 128- or 256-bit destination.
    fn extend(&mut self, ins: &Instruction, from: Elem, to: Elem, op: &str) -> R<()> {
        let dst = ins.op_register(0);
        let n = if dst.is_ymm() {
            2 * to.lanes()
        } else {
            to.lanes()
        };
        let v = match ins.op_kind(1) {
            OpKind::Register => {
                let x = self.xget_half(ins.op_register(1), 0, from)?;
                self.take_low(&x, from, n)
            }
            _ => self.mem_load_at(ins, Ty::vn(from, n), 0, (n * from.bits() / 8) as usize)?,
        };
        let w = self.val(&format!(
            "{op} {} {v} to {}",
            Ty::vn(from, n).ir(),
            Ty::vn(to, n).ir()
        ));
        if dst.is_ymm() {
            let (lo, hi) = self.split(&w, to);
            self.set_halves(dst, &lo, Some(&hi), to)
        } else {
            self.xset_half(dst, 0, &w, to)
        }
    }

    /// binary16 (zero-extended in `<n x i32>`) -> binary32 bits, exact (signalling NaNs are
    /// quieted like hardware does).
    fn half_to_float(&mut self, h: &str, n: u32) -> String {
        let ty = Ty::vn(Elem::I32, n);
        let t = ty.ir();
        let bt = ty.bool_ir();
        let c = |v: i64| ty.splat(&v.to_string());
        let sign = self.val(&format!("and {t} {h}, {}", c(0x8000)));
        let sign = self.val(&format!("shl {t} {sign}, {}", c(16)));
        let exp = self.val(&format!("lshr {t} {h}, {}", c(10)));
        let exp = self.val(&format!("and {t} {exp}, {}", c(31)));
        let man = self.val(&format!("and {t} {h}, {}", c(0x3FF)));
        let man13 = self.val(&format!("shl {t} {man}, {}", c(13)));
        let e32 = self.val(&format!("add {t} {exp}, {}", c(112)));
        let e32 = self.val(&format!("shl {t} {e32}, {}", c(23)));
        let normal = self.val(&format!("or {t} {e32}, {man13}"));
        let has_man = self.val(&format!("icmp ne {t} {man}, zeroinitializer"));
        let quiet = self.val(&format!(
            "select {bt} {has_man}, {t} {}, {t} zeroinitializer",
            c(0x0040_0000)
        ));
        let special = self.val(&format!("or {t} {man13}, {}", c(0x7F80_0000)));
        let special = self.val(&format!("or {t} {special}, {quiet}"));
        // Zero and denormals: man * 2^-24 is exact in binary32.
        let f = self.val(&format!("uitofp {t} {man} to <{n} x float>"));
        let scale = Ty::vn(Elem::F32, n).splat(&fconst(2f64.powi(-24)));
        let f = self.val(&format!("fmul <{n} x float> {f}, {scale}"));
        let tiny = self.val(&format!("bitcast <{n} x float> {f} to {t}"));
        let is0 = self.val(&format!("icmp eq {t} {exp}, zeroinitializer"));
        let is31 = self.val(&format!("icmp eq {t} {exp}, {}", c(31)));
        let r = self.val(&format!("select {bt} {is31}, {t} {special}, {t} {normal}"));
        let r = self.val(&format!("select {bt} {is0}, {t} {tiny}, {t} {r}"));
        self.val(&format!("or {t} {r}, {sign}"))
    }

    /// binary32 bits (`<n x i32>`) -> binary16 in the low 16 bits of each lane, rounded with
    /// `mode`, exactly like `vcvtps2ph` (NaNs are quieted and keep the top payload bits).
    fn float_to_half(&mut self, x: &str, n: u32, mode: Round) -> String {
        let ty = Ty::vn(Elem::I32, n);
        let t = ty.ir();
        let bt = ty.bool_ir();
        let c = |v: i64| ty.splat(&v.to_string());
        let s = self.val(&format!("lshr {t} {x}, {}", c(31)));
        let neg = self.val(&format!("icmp ne {t} {s}, zeroinitializer"));
        let sign16 = self.val(&format!("shl {t} {s}, {}", c(15)));
        let e = self.val(&format!("lshr {t} {x}, {}", c(23)));
        let e = self.val(&format!("and {t} {e}, {}", c(255)));
        let m = self.val(&format!("and {t} {x}, {}", c(0x7F_FFFF)));
        let e_zero = self.val(&format!("icmp eq {t} {e}, zeroinitializer"));
        let mi = self.val(&format!("or {t} {m}, {}", c(0x80_0000)));
        let full = self.val(&format!("select {bt} {e_zero}, {t} {m}, {t} {mi}"));
        // Denormal inputs share the exponent of e == 1 (without the implicit bit).
        let e_eff = self.val(&format!("select {bt} {e_zero}, {t} {}, {t} {e}", c(1)));
        let big_e = self.val(&format!("sub {t} {e_eff}, {}", c(112))); // half exponent
        let normal = self.val(&format!("icmp sgt {t} {big_e}, zeroinitializer"));
        // shift = 13 for normal results, 14 - E for denormal ones (capped at 31).
        let dshift = self.val(&format!("sub {t} {}, {big_e}", c(14)));
        let small = self.val(&format!("icmp ult {t} {dshift}, {}", c(31)));
        let dshift = self.val(&format!("select {bt} {small}, {t} {dshift}, {t} {}", c(31)));
        let shift = self.val(&format!(
            "select {bt} {normal}, {t} {}, {t} {dshift}",
            c(13)
        ));
        let q = self.val(&format!("lshr {t} {full}, {shift}"));
        let ones = self.val(&format!("shl {t} {}, {shift}", c(1)));
        let lowmask = self.val(&format!("sub {t} {ones}, {}", c(1)));
        let rem = self.val(&format!("and {t} {full}, {lowmask}"));
        let em1 = self.val(&format!("sub {t} {big_e}, {}", c(1)));
        let em1 = self.val(&format!("shl {t} {em1}, {}", c(10)));
        let base = self.val(&format!(
            "select {bt} {normal}, {t} {em1}, {t} zeroinitializer"
        ));
        let truncated = self.val(&format!("add {t} {base}, {q}"));
        let inexact = self.val(&format!("icmp ne {t} {rem}, zeroinitializer"));
        let inc = match mode {
            Round::Nearest => {
                let halfway = self.val(&format!("lshr {t} {ones}, {}", c(1)));
                let above = self.val(&format!("icmp ugt {t} {rem}, {halfway}"));
                let tie = self.val(&format!("icmp eq {t} {rem}, {halfway}"));
                let odd = self.val(&format!("and {t} {q}, {}", c(1)));
                let odd = self.val(&format!("icmp ne {t} {odd}, zeroinitializer"));
                let tie_odd = self.val(&format!("and {bt} {tie}, {odd}"));
                self.val(&format!("or {bt} {above}, {tie_odd}"))
            }
            Round::Down => self.val(&format!("and {bt} {inexact}, {neg}")),
            Round::Up => {
                let pos = self.val(&format!("xor {bt} {neg}, {}", ty.bool_splat(true)));
                self.val(&format!("and {bt} {inexact}, {pos}"))
            }
            Round::Zero => ty.bool_splat(false),
        };
        let inc = self.val(&format!("zext {bt} {inc} to {t}"));
        let rounded = self.val(&format!("add {t} {truncated}, {inc}"));
        // |x| >= 2^16 (half exponent >= 31): overflow per rounding mode.
        let over = self.val(&format!("icmp sgt {t} {big_e}, {}", c(30)));
        let (inf, max) = (c(0x7C00), c(0x7BFF));
        let overflow = match mode {
            Round::Nearest => inf.clone(),
            Round::Zero => max.clone(),
            Round::Down => self.val(&format!("select {bt} {neg}, {t} {inf}, {t} {max}")),
            Round::Up => self.val(&format!("select {bt} {neg}, {t} {max}, {t} {inf}")),
        };
        let r = self.val(&format!(
            "select {bt} {over}, {t} {overflow}, {t} {rounded}"
        ));
        // Infinity and NaN inputs.
        let special = self.val(&format!("icmp eq {t} {e}, {}", c(255)));
        let payload = self.val(&format!("lshr {t} {m}, {}", c(13)));
        let has_m = self.val(&format!("icmp ne {t} {m}, zeroinitializer"));
        let nan = self.val(&format!("or {t} {payload}, {}", c(0x7E00)));
        let special_v = self.val(&format!("select {bt} {has_m}, {t} {nan}, {t} {inf}"));
        let r = self.val(&format!("select {bt} {special}, {t} {special_v}, {t} {r}"));
        self.val(&format!("or {t} {r}, {sign16}"))
    }
}
