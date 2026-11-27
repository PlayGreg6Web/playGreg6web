//! Data movement: full/partial register moves, GPR <-> XMM moves, sign masks and the
//! floating-point shuffles, unpacks, blends and inserts. Moves use integer lane types so that no
//! value is ever interpreted as a float (NaN bits are preserved exactly).

use iced_x86::{Instruction, Mnemonic, OpKind, Register};

use super::super::{FunctionLifter, R, unsupported};
use super::{Elem, Form, Intr, Ty, is_identity_mask, same_reg};

/// Whether [`FunctionLifter::lift_moves`] handles `m` (mirroring its match arms).
pub(super) fn is_move_supported(m: Mnemonic) -> bool {
    use Mnemonic as M;
    matches!(
        m,
        M::Movaps
            | M::Vmovaps
            | M::Movups
            | M::Vmovups
            | M::Movapd
            | M::Vmovapd
            | M::Movupd
            | M::Vmovupd
            | M::Movdqa
            | M::Vmovdqa
            | M::Movdqu
            | M::Vmovdqu
            | M::Lddqu
            | M::Vlddqu
            | M::Movntps
            | M::Vmovntps
            | M::Movntpd
            | M::Vmovntpd
            | M::Movntdq
            | M::Vmovntdq
            | M::Movntdqa
            | M::Vmovntdqa
            | M::Movss
            | M::Vmovss
            | M::Movsd
            | M::Vmovsd
            | M::Movd
            | M::Vmovd
            | M::Movq
            | M::Vmovq
            | M::Movlps
            | M::Vmovlps
            | M::Movlpd
            | M::Vmovlpd
            | M::Movhps
            | M::Vmovhps
            | M::Movhpd
            | M::Vmovhpd
            | M::Movlhps
            | M::Vmovlhps
            | M::Movhlps
            | M::Vmovhlps
            | M::Movmskps
            | M::Vmovmskps
            | M::Movmskpd
            | M::Vmovmskpd
            | M::Pmovmskb
            | M::Vpmovmskb
            | M::Movsldup
            | M::Vmovsldup
            | M::Movshdup
            | M::Vmovshdup
            | M::Movddup
            | M::Vmovddup
            | M::Vpermilps
            | M::Vpermilpd
            | M::Shufps
            | M::Vshufps
            | M::Shufpd
            | M::Vshufpd
            | M::Unpcklps
            | M::Vunpcklps
            | M::Unpckhps
            | M::Vunpckhps
            | M::Unpcklpd
            | M::Vunpcklpd
            | M::Unpckhpd
            | M::Vunpckhpd
            | M::Blendps
            | M::Vblendps
            | M::Blendpd
            | M::Vblendpd
            | M::Blendvps
            | M::Vblendvps
            | M::Blendvpd
            | M::Vblendvpd
            | M::Insertps
            | M::Vinsertps
            | M::Extractps
            | M::Vextractps
    )
}

impl FunctionLifter<'_> {
    pub(super) fn lift_moves(&mut self, ins: &Instruction, vex: bool) -> R<bool> {
        use Elem::{I32, I64};
        use Mnemonic as M;
        let f = Form::of(vex);
        match ins.mnemonic() {
            M::Movaps
            | M::Vmovaps
            | M::Movups
            | M::Vmovups
            | M::Movapd
            | M::Vmovapd
            | M::Movupd
            | M::Vmovupd
            | M::Movdqa
            | M::Vmovdqa
            | M::Movdqu
            | M::Vmovdqu
            | M::Lddqu
            | M::Vlddqu
            | M::Movntps
            | M::Vmovntps
            | M::Movntpd
            | M::Vmovntpd
            | M::Movntdq
            | M::Vmovntdq
            | M::Movntdqa
            | M::Vmovntdqa => {
                let v = self.v128(ins, 1, I64)?;
                self.set_v128(ins, 0, &v, I64)?;
            }
            M::Movss | M::Vmovss => self.move_scalar(ins, I32)?,
            M::Movsd | M::Vmovsd => self.move_scalar(ins, I64)?,
            M::Movd | M::Vmovd => self.move_gpr(ins, 32)?,
            M::Movq | M::Vmovq => self.move_gpr(ins, 64)?,
            M::Movlps | M::Vmovlps | M::Movlpd | M::Vmovlpd => self.move_half(ins, 0)?,
            M::Movhps | M::Vmovhps | M::Movhpd | M::Vmovhpd => self.move_half(ins, 1)?,
            M::Movlhps | M::Vmovlhps | M::Movhlps | M::Vmovhlps => {
                let lh = matches!(ins.mnemonic(), M::Movlhps | M::Vmovlhps);
                let a = self.xget(ins.op_register(f.a), I64)?;
                let r = if same_reg(ins, f.a, f.b) {
                    let idx = if lh { [0, 0] } else { [1, 1] };
                    self.shuffle(&a, "poison", I64, &idx)
                } else {
                    let b = self.v128(ins, f.b, I64)?;
                    let idx = if lh { [0, 2] } else { [3, 1] };
                    self.shuffle(&a, &b, I64, &idx)
                };
                self.xset(ins.op_register(f.dst), &r, I64)?;
            }
            M::Movmskps | M::Vmovmskps => self.move_mask(ins, I32)?,
            M::Movmskpd | M::Vmovmskpd => self.move_mask(ins, I64)?,
            M::Pmovmskb | M::Vpmovmskb => self.move_mask(ins, Elem::I8)?,
            M::Movsldup | M::Vmovsldup | M::Movshdup | M::Vmovshdup => {
                let v = self.v128(ins, 1, I32)?;
                let r = if v == "zeroinitializer" {
                    v
                } else {
                    let idx = if matches!(ins.mnemonic(), M::Movsldup | M::Vmovsldup) {
                        [0, 0, 2, 2]
                    } else {
                        [1, 1, 3, 3]
                    };
                    self.shuffle(&v, "poison", I32, &idx)
                };
                self.xset(ins.op_register(0), &r, I32)?;
            }
            M::Movddup | M::Vmovddup => {
                // xmm, xmm/m64 ; ymm, ymm/m256 (even quadword of each lane).
                // For m64, emit `splat_value` directly so LLVM lowers to `v128.load64_splat`.
                let r = if ins.op_kind(1) == OpKind::Memory && Self::mem_bytes(ins) == 8 {
                    let s = self.low(ins, 1, I64, 1)?;
                    self.splat_value(&s, I64)
                } else {
                    let v = self.v128(ins, 1, I64)?;
                    if v == "zeroinitializer" {
                        v
                    } else {
                        self.shuffle(&v, "poison", I64, &[0, 0])
                    }
                };
                self.xset(ins.op_register(0), &r, I64)?;
            }
            M::Vpermilps | M::Vpermilpd => {
                let e = if ins.mnemonic() == M::Vpermilps {
                    I32
                } else {
                    I64
                };
                let v = self.v128(ins, 1, e)?;
                let r = if super::super::is_immediate(ins.op_kind(2)) {
                    let i = Self::imm8(ins)?;
                    let idx: Vec<u32> = if e == I32 {
                        (0..4).map(|l| (i >> (2 * l)) & 3).collect()
                    } else {
                        let i = self.lane_imm(i, 2);
                        (0..2).map(|l| (i >> l) & 1).collect()
                    };
                    if v == "zeroinitializer" || is_identity_mask(&idx) {
                        v
                    } else {
                        self.shuffle(&v, "poison", e, &idx)
                    }
                } else {
                    // Variable control: bits 1:0 (ps) -> `i8x16.swizzle`; bit 1 (pd) -> vector
                    // select between splatted lanes (zero stack spills).
                    let ctl = self.v128(ins, 2, e)?;
                    if v == "zeroinitializer" {
                        v
                    } else if e == I32 {
                        let idx = self.dword_swizzle_indices(&ctl, 3);
                        let vb = self.cast(&v, Ty::v(I32), Ty::v(Elem::I8));
                        let rb = self.intr(Intr::Swizzle, Ty::v(Elem::I8), &[&vb, &idx]);
                        self.cast(&rb, Ty::v(Elem::I8), Ty::v(I32))
                    } else {
                        let bit = self.val(&format!("and <2 x i64> {ctl}, <i64 2, i64 2>"));
                        let sel = self.val(&format!("icmp ne <2 x i64> {bit}, zeroinitializer"));
                        let v0 = self.shuffle(&v, "poison", I64, &[0, 0]);
                        let v1 = self.shuffle(&v, "poison", I64, &[1, 1]);
                        self.val(&format!("select <2 x i1> {sel}, <2 x i64> {v1}, <2 x i64> {v0}"))
                    }
                };
                self.xset(ins.op_register(0), &r, e)?;
            }
            M::Shufps | M::Vshufps => {
                let i = Self::imm8(ins)?;
                let idx = [i & 3, (i >> 2) & 3, 4 + ((i >> 4) & 3), 4 + ((i >> 6) & 3)];
                self.shuffle_op(ins, f, I32, &idx)?;
            }
            M::Shufpd | M::Vshufpd => {
                let i = Self::imm8(ins)?;
                let i = self.lane_imm(i, 2);
                self.shuffle_op(ins, f, I64, &[i & 1, 2 + ((i >> 1) & 1)])?;
            }
            M::Unpcklps | M::Vunpcklps => self.shuffle_op(ins, f, I32, &[0, 4, 1, 5])?,
            M::Unpckhps | M::Vunpckhps => self.shuffle_op(ins, f, I32, &[2, 6, 3, 7])?,
            M::Unpcklpd | M::Vunpcklpd => self.shuffle_op(ins, f, I64, &[0, 2])?,
            M::Unpckhpd | M::Vunpckhpd => self.shuffle_op(ins, f, I64, &[1, 3])?,
            M::Blendps | M::Vblendps => self.blend_imm(ins, f, I32)?,
            M::Blendpd | M::Vblendpd => self.blend_imm(ins, f, I64)?,
            M::Blendvps | M::Vblendvps => self.blend_var(ins, f, vex, I32)?,
            M::Blendvpd | M::Vblendvpd => self.blend_var(ins, f, vex, I64)?,
            M::Insertps | M::Vinsertps => {
                let i = Self::imm8(ins)?;
                let zmask = i & 0xF;
                if zmask == 0xF {
                    if ins.op_kind(f.b) == OpKind::Memory {
                        let _ = self.mem_load(ins, Ty::s(I32))?;
                    }
                    self.xset(ins.op_register(f.dst), "zeroinitializer", I32)?;
                    return Ok(true);
                }
                let src_lane = (i >> 6) & 3;
                let dst_lane = (i >> 4) & 3;
                // When every lane other than `dst_lane` is zeroed (e.g. `0x0E`/`0x4E`/`0x8E`/`0xCE`
                // extracting one lane to lane 0 and zeroing the rest), `a` is completely dead and
                // one shuffle/insert with `zeroinitializer` suffices.
                if (zmask | (1 << dst_lane)) == 0xF {
                    let r = match ins.op_kind(f.b) {
                        OpKind::Register => {
                            let b = self.xget(ins.op_register(f.b), I32)?;
                            let mut idx = [4u32; 4];
                            idx[dst_lane as usize] = src_lane;
                            self.shuffle(&b, "zeroinitializer", I32, &idx)
                        }
                        _ => {
                            let s = self.mem_load(ins, Ty::s(I32))?;
                            self.insert("zeroinitializer", I32, &s, dst_lane)
                        }
                    };
                    self.xset(ins.op_register(f.dst), &r, I32)?;
                    return Ok(true);
                }
                let a = self.xget(ins.op_register(f.a), I32)?;
                let mut r = match ins.op_kind(f.b) {
                    OpKind::Register => {
                        if (zmask >> dst_lane) & 1 == 1 {
                            a
                        } else if same_reg(ins, f.a, f.b) {
                            let mut idx = [0, 1, 2, 3];
                            idx[dst_lane as usize] = src_lane;
                            if is_identity_mask(&idx) {
                                a
                            } else {
                                self.shuffle(&a, "poison", I32, &idx)
                            }
                        } else {
                            let b = self.xget(ins.op_register(f.b), I32)?;
                            let mut idx = [0, 1, 2, 3];
                            idx[dst_lane as usize] = 4 + src_lane;
                            self.shuffle(&a, &b, I32, &idx)
                        }
                    }
                    _ => {
                        let s = self.mem_load(ins, Ty::s(I32))?;
                        if (zmask >> dst_lane) & 1 == 1 {
                            a
                        } else {
                            self.insert(&a, I32, &s, dst_lane)
                        }
                    }
                };
                if zmask != 0 {
                    let idx: Vec<u32> = (0..4)
                        .map(|l| if (zmask >> l) & 1 == 1 { 4 } else { l })
                        .collect();
                    r = self.shuffle(&r, "zeroinitializer", I32, &idx);
                }
                self.xset(ins.op_register(f.dst), &r, I32)?;
            }
            M::Extractps | M::Vextractps => {
                let i = Self::imm8(ins)?;
                let v = self.xget(ins.op_register(1), I32)?;
                let s = self.extract(&v, I32, i & 3);
                self.write_gpr_or_mem(ins, 0, &s, 32)?;
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// `dst = shufflevector(a, b, idx)` with the operand layout `f`.
    /// Recognises identical register operands (`poison` unary shuffle), single-source masks
    /// (such as trivial `blend` masks 0 / all-ones), identity permutations (direct move), and
    /// known-zero vector sources.
    pub(super) fn shuffle_op(&mut self, ins: &Instruction, f: Form, e: Elem, idx: &[u32]) -> R<()> {
        let n = e.lanes();
        if same_reg(ins, f.a, f.b) {
            let a = self.v128(ins, f.a, e)?;
            let mapped: Vec<u32> = idx.iter().map(|&k| k % n).collect();
            let r = if a == "zeroinitializer" || is_identity_mask(&mapped) {
                a
            } else {
                self.shuffle(&a, "poison", e, &mapped)
            };
            return self.xset(ins.op_register(f.dst), &r, e);
        }
        let only_a = idx.iter().all(|&k| k < n);
        let only_b = idx.iter().all(|&k| k >= n);
        let r = if only_a {
            let a = self.v128(ins, f.a, e)?;
            if ins.op_kind(f.b) == OpKind::Memory {
                let _ = self.v128(ins, f.b, e)?;
            }
            if a == "zeroinitializer" || is_identity_mask(idx) {
                a
            } else {
                self.shuffle(&a, "poison", e, idx)
            }
        } else if only_b {
            if ins.op_kind(f.a) == OpKind::Memory {
                let _ = self.v128(ins, f.a, e)?;
            }
            let b = self.v128(ins, f.b, e)?;
            let mapped: Vec<u32> = idx.iter().map(|&k| k - n).collect();
            if b == "zeroinitializer" || is_identity_mask(&mapped) {
                b
            } else {
                self.shuffle(&b, "poison", e, &mapped)
            }
        } else {
            let a = self.v128(ins, f.a, e)?;
            let b = self.v128(ins, f.b, e)?;
            if a == "zeroinitializer" && b == "zeroinitializer" {
                "zeroinitializer".to_string()
            } else {
                self.shuffle(&a, &b, e, idx)
            }
        };
        self.xset(ins.op_register(f.dst), &r, e)
    }

    /// Writes an `bits`-bit lane value to a GPR (zero-extended to the register width) or memory.
    pub(super) fn write_gpr_or_mem(
        &mut self,
        ins: &Instruction,
        i: u32,
        v: &str,
        bits: u32,
    ) -> R<()> {
        match ins.op_kind(i) {
            OpKind::Memory => self.mem_store(ins, v, Ty::s(Elem::of_bits(bits))),
            OpKind::Register => {
                let reg_bits = self.op_bits(ins, i)?;
                let v = if reg_bits > bits {
                    self.val(&format!("zext i{bits} {v} to i{reg_bits}"))
                } else if reg_bits == bits {
                    v.to_string()
                } else {
                    return unsupported("internal: destination register too narrow");
                };
                self.write_op(ins, i, &v, reg_bits)
            }
            kind => unsupported(format!("destination operand kind {kind:?}")),
        }
    }

    /// `movss`/`movsd` (lane type `I32`/`I64`): register merges keep the upper lanes of the
    /// first source, loads zero them, stores write the low lane only.
    fn move_scalar(&mut self, ins: &Instruction, e: Elem) -> R<()> {
        let merge_idx: &[u32] = if e == Elem::I32 {
            &[4, 1, 2, 3]
        } else {
            &[2, 1]
        };
        if ins.op_count() == 3 {
            // vmovss xmm1, xmm2, xmm3
            let a = self.xget(ins.op_register(1), e)?;
            if same_reg(ins, 1, 2) {
                return self.xset(ins.op_register(0), &a, e);
            }
            let b = self.xget(ins.op_register(2), e)?;
            let r = if a == "zeroinitializer" && b == "zeroinitializer" {
                "zeroinitializer".to_string()
            } else {
                self.shuffle(&a, &b, e, merge_idx)
            };
            return self.xset(ins.op_register(0), &r, e);
        }
        match (ins.op_kind(0), ins.op_kind(1)) {
            (OpKind::Memory, OpKind::Register) => {
                let v = self.xget(ins.op_register(1), e)?;
                let s = self.extract(&v, e, 0);
                self.mem_store(ins, &s, Ty::s(e))
            }
            (OpKind::Register, OpKind::Memory) => {
                let s = self.mem_load(ins, Ty::s(e))?;
                let r = self.zext_lanes(&s, e, 1);
                self.xset(ins.op_register(0), &r, e)
            }
            (OpKind::Register, OpKind::Register) => {
                let a = self.xget(ins.op_register(0), e)?;
                if same_reg(ins, 0, 1) {
                    return self.xset(ins.op_register(0), &a, e);
                }
                let b = self.xget(ins.op_register(1), e)?;
                let r = if a == "zeroinitializer" && b == "zeroinitializer" {
                    "zeroinitializer".to_string()
                } else {
                    self.shuffle(&a, &b, e, merge_idx)
                };
                self.xset(ins.op_register(0), &r, e)
            }
            _ => unsupported("movss/movsd operand form"),
        }
    }

    /// `movd`/`movq` between XMM and GPR/memory, and `movq xmm, xmm/m64`: writes to an XMM
    /// register zero the upper bits.
    fn move_gpr(&mut self, ins: &Instruction, bits: u32) -> R<()> {
        let dst_is_xmm = ins.op_kind(0) == OpKind::Register && ins.op_register(0).is_xmm();
        if dst_is_xmm {
            if ins.op_kind(1) == OpKind::Register && ins.op_register(1).is_xmm() {
                let x = self.xget(ins.op_register(1), Elem::I64)?;
                let r = if x == "zeroinitializer" {
                    x
                } else {
                    self.shuffle(&x, "zeroinitializer", Elem::I64, &[0, 2])
                };
                return self.xset(ins.op_register(0), &r, Elem::I64);
            }
            let v = self.read_op(ins, 1, bits)?;
            if v == "0" {
                return self.xset(ins.op_register(0), "zeroinitializer", Elem::I64);
            }
            // Insert directly at the operand's lane width (`<4 x i32>` for `movd`, `<2 x i64>` for
            // `movq`) to skip `zext i32 to i64` and enable native `v128.load32_zero` / `vmovd`.
            let elem = Elem::of_bits(bits);
            let r = self.zext_lanes(&v, elem, 1);
            return self.xset(ins.op_register(0), &r, elem);
        }
        if !(ins.op_kind(1) == OpKind::Register && ins.op_register(1).is_xmm()) {
            return unsupported("movd/movq operand form");
        }
        let e = Elem::of_bits(bits);
        let x = self.xget(ins.op_register(1), e)?;
        let s = if x == "zeroinitializer" {
            "0".to_string()
        } else {
            self.extract(&x, e, 0)
        };
        match ins.op_kind(0) {
            OpKind::Memory => self.mem_store(ins, &s, Ty::s(e)),
            _ => self.write_op(ins, 0, &s, bits),
        }
    }

    /// `movlps/movhps/movlpd/movhpd`: 64-bit loads into (VEX: merges with the first source) and
    /// stores from quadword lane `lane`.
    fn move_half(&mut self, ins: &Instruction, lane: u32) -> R<()> {
        let e = Elem::I64;
        if ins.op_kind(0) == OpKind::Memory {
            let v = self.xget(ins.op_register(1), e)?;
            let s = self.extract(&v, e, lane);
            return self.mem_store(ins, &s, Ty::s(e));
        }
        let base = if ins.op_count() == 3 { 1 } else { 0 };
        let a = self.xget(ins.op_register(base), e)?;
        let s = self.mem_load(ins, Ty::s(e))?;
        let r = self.insert(&a, e, &s, lane);
        self.xset(ins.op_register(0), &r, e)
    }

    /// `movmskps/movmskpd/pmovmskb`: the sign bit of every lane, zero-extended into a GPR.
    /// Emits native Wasm SIMD128 `i{8x16,32x4,64x2}.bitmask` (1 op, 2 bytes vs 22 bytes).
    fn move_mask(&mut self, ins: &Instruction, e: Elem) -> R<()> {
        let v = self.xget(ins.op_register(1), e)?;
        let dst_bits = self.op_bits(ins, 0)?;
        if v == "zeroinitializer" {
            return self.write_op(ins, 0, "0", dst_bits);
        }
        let bits = self.intr(Intr::Bitmask, Ty::v(e), &[&v]);
        let r = if dst_bits > 32 {
            self.val(&format!("zext i32 {bits} to i{dst_bits}"))
        } else {
            bits
        };
        self.write_op(ins, 0, &r, dst_bits)
    }

    /// `blendps/blendpd` (and `pblendw`): lane `i` comes from the second source when bit `i` of
    /// the immediate is set.
    pub(super) fn blend_imm(&mut self, ins: &Instruction, f: Form, e: Elem) -> R<()> {
        let i = Self::imm8(ins)?;
        let n = e.lanes();
        // 256-bit forms take the upper lane's selectors from the next immediate bits.
        let i = self.lane_imm(i, n);
        let idx: Vec<u32> = (0..n)
            .map(|l| if i >> (l % 8) & 1 == 1 { n + l } else { l })
            .collect();
        self.shuffle_op(ins, f, e, &idx)
    }

    /// `blendvps/blendvpd/pblendvb`: lane from the second source when the mask lane's sign bit is
    /// set. The mask is the implicit `xmm0` (legacy) or the fourth operand (VEX).
    /// Lowers via `ashr` sign-bit splat + `@llvm.wasm.bitselect.*` (6B vs 22B with `v128.const 0`
    /// + `icmp slt`, while still folding `ashr` away after `cmpps`/`pcmpeq*`/`pcmpgt*`).
    pub(super) fn blend_var(&mut self, ins: &Instruction, f: Form, vex: bool, e: Elem) -> R<()> {
        if same_reg(ins, f.a, f.b) {
            let a = self.v128(ins, f.a, e)?;
            return self.xset(ins.op_register(f.dst), &a, e);
        }
        let mask_reg = if vex {
            ins.op_register(3)
        } else {
            Register::XMM0
        };
        let ty = Ty::v(e);
        let t = ty.ir();
        let a = self.v128(ins, f.a, e)?;
        let b = self.v128(ins, f.b, e)?;
        let m = self.xget(mask_reg, e)?;
        if m == "zeroinitializer" {
            return self.xset(ins.op_register(f.dst), &a, e);
        }
        if a == "zeroinitializer" && b == "zeroinitializer" {
            return self.xset(ins.op_register(f.dst), "zeroinitializer", e);
        }
        let shift = ty.splat(&(e.bits() - 1).to_string());
        let mask = self.val(&format!("ashr {t} {m}, {shift}"));
        let r = self.intr(Intr::Bitselect, ty, &[&b, &a, &mask]);
        self.xset(ins.op_register(f.dst), &r, e)
    }
}
