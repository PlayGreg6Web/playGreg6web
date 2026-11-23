//! Bit instructions: `bt`/`bts`/`btr`/`btc`, `bsf`/`bsr`, `tzcnt`/`lzcnt`/`popcnt`, `bswap`,
//! BMI1/BMI2 (`andn blsi blsr blsmsk bextr bzhi shlx shrx sarx rorx mulx pdep pext`),
//! rotates through carry (`rcl`/`rcr`) and double-precision shifts (`shld`/`shrd`).
//!
//! Flags an instruction leaves *undefined* keep their previous value. Choices for
//! architecturally undefined results:
//! - `bswap r16` writes 0 to the 16-bit register (what Intel and AMD hardware do).
//! - `bsf`/`bsr` with a zero source leave the whole destination register unchanged (also for
//!   32-bit operands: no zero-extension), as hardware does and code relies on.
//! - `rcl`/`rcr`/`shld`/`shrd`: OF is computed with the count-1 formula for every non-zero count
//!   (undefined for counts > 1).
//! - `shld`/`shrd` with a 16-bit operand and a masked count of 17..31: the result is the 32-bit
//!   concatenation `dst:src` (resp. `src:dst`) shifted by the count, with flags set from it.

use iced_x86::{Instruction, Mnemonic, OpKind, Register};

use super::{Flag, FunctionLifter, R, RegRef, imm, is_immediate, mask, reg_ref, ty, unsupported};
use crate::lift::{BitIntrinsic, Support};

/// `pdep`: scatters the low bits of `src` to the set bits of `mask` (lowest first).
pub(in crate::lift) const PDEP_IR: &str = "\
define internal i64 @__playgta6_pdep(i64 %src, i64 %mask) {
entry:
  br label %loop
loop:
  %m = phi i64 [ %mask, %entry ], [ %m.next, %body ]
  %k = phi i64 [ 1, %entry ], [ %k.next, %body ]
  %r = phi i64 [ 0, %entry ], [ %r.next, %body ]
  %done = icmp eq i64 %m, 0
  br i1 %done, label %exit, label %body
body:
  %neg = sub i64 0, %m
  %low = and i64 %m, %neg
  %hit = and i64 %src, %k
  %take = icmp ne i64 %hit, 0
  %add = select i1 %take, i64 %low, i64 0
  %r.next = or i64 %r, %add
  %m.next = xor i64 %m, %low
  %k.next = shl i64 %k, 1
  br label %loop
exit:
  ret i64 %r
}
";

/// `pext`: gathers the bits of `src` at the set bits of `mask` into the low bits.
pub(in crate::lift) const PEXT_IR: &str = "\
define internal i64 @__playgta6_pext(i64 %src, i64 %mask) {
entry:
  br label %loop
loop:
  %m = phi i64 [ %mask, %entry ], [ %m.next, %body ]
  %k = phi i64 [ 1, %entry ], [ %k.next, %body ]
  %r = phi i64 [ 0, %entry ], [ %r.next, %body ]
  %done = icmp eq i64 %m, 0
  br i1 %done, label %exit, label %body
body:
  %neg = sub i64 0, %m
  %low = and i64 %m, %neg
  %hit = and i64 %src, %low
  %take = icmp ne i64 %hit, 0
  %add = select i1 %take, i64 %k, i64 0
  %r.next = or i64 %r, %add
  %m.next = xor i64 %m, %low
  %k.next = shl i64 %k, 1
  br label %loop
exit:
  ret i64 %r
}
";

impl FunctionLifter<'_> {
    /// Emits a call to `llvm.<kind>.iN` (`cttz`/`ctlz` with a defined result for zero).
    pub(super) fn bit_intrinsic(&mut self, kind: BitIntrinsic, v: &str, bits: u32) -> String {
        let (name, extra) = match kind {
            BitIntrinsic::Cttz => ("cttz", ", i1 false"),
            BitIntrinsic::Ctlz => ("ctlz", ", i1 false"),
            BitIntrinsic::Ctpop => ("ctpop", ""),
            BitIntrinsic::Bswap => ("bswap", ""),
        };
        if kind == BitIntrinsic::Ctpop && bits == 8 {
            self.support.insert(Support::Ctpop8);
        } else {
            self.support.insert(Support::BitIntrinsic(kind, bits));
        }
        let t = ty(bits);
        self.val(&format!("call {t} @llvm.{name}.{t}({t} {v}{extra})"))
    }

    pub(super) fn bswap_value(&mut self, v: &str, bits: u32) -> String {
        if bits == 8 {
            return v.to_string();
        }
        self.bit_intrinsic(BitIntrinsic::Bswap, v, bits)
    }

    /// Reads operand `i` as a bit offset/index immediate or register value.
    fn imm_operand(&self, ins: &Instruction, i: u32) -> R<Option<u64>> {
        if !is_immediate(ins.op_kind(i)) {
            return Ok(None);
        }
        match ins.try_immediate(i) {
            Ok(v) => Ok(Some(v)),
            Err(_) => unsupported("unreadable immediate"),
        }
    }

    /// `bt`/`bts`/`btr`/`btc`. Only CF is defined (ZF unaffected; OF SF AF PF undefined, left
    /// unchanged). A memory operand with a register bit offset addresses
    /// `ea + floor(offset / 8)` with the offset sign-extended from the operand size, so the bit
    /// may lie outside the operand; the access is done on that single byte, which is
    /// indistinguishable from the operand-size access (and is what `lock bts` etc. update
    /// atomically). Immediate offsets are masked to the operand size.
    pub(super) fn bit_test(&mut self, ins: &Instruction, locked: bool) -> R<()> {
        let m = ins.mnemonic();
        if locked && m == Mnemonic::Bt {
            return unsupported("lock prefix on bt (#UD)");
        }
        let bits = self.op_bits(ins, 0)?;
        let t = ty(bits);
        let offset_imm = self.imm_operand(ins, 1)?;
        let cf = match ins.op_kind(0) {
            OpKind::Register => {
                if locked {
                    return unsupported("lock prefix with a register destination (#UD)");
                }
                let v = self.read_op(ins, 0, bits)?;
                let index = match offset_imm {
                    Some(o) => imm(o & u64::from(bits - 1), bits),
                    None => {
                        let o = self.read_op(ins, 1, bits)?;
                        self.val(&format!("and {t} {o}, {}", bits - 1))
                    }
                };
                let one = self.val(&format!("shl {t} 1, {index}"));
                let hit = self.val(&format!("and {t} {v}, {one}"));
                let cf = self.val(&format!("icmp ne {t} {hit}, 0"));
                let new = match m {
                    Mnemonic::Bts => Some(self.val(&format!("or {t} {v}, {one}"))),
                    Mnemonic::Btr => {
                        let keep = self.val(&format!("xor {t} {one}, -1"));
                        Some(self.val(&format!("and {t} {v}, {keep}")))
                    }
                    Mnemonic::Btc => Some(self.val(&format!("xor {t} {v}, {one}"))),
                    _ => None,
                };
                if let Some(new) = new {
                    self.write_op(ins, 0, &new, bits)?;
                }
                cf
            }
            OpKind::Memory => {
                let ea = self.mem_ea(ins)?;
                let (addr, pos) = match offset_imm {
                    Some(o) => {
                        let o = o & u64::from(bits - 1);
                        let addr = if o >= 8 {
                            self.val(&format!("add i64 {ea}, {}", o >> 3))
                        } else {
                            ea
                        };
                        (addr, (o & 7).to_string())
                    }
                    None => {
                        let o = self.read_op(ins, 1, bits)?;
                        let wide = if bits == 64 {
                            o
                        } else {
                            self.val(&format!("sext {t} {o} to i64"))
                        };
                        let byte_offset = self.val(&format!("ashr i64 {wide}, 3"));
                        let addr = self.val(&format!("add i64 {ea}, {byte_offset}"));
                        let low = self.val(&format!("and i64 {wide}, 7"));
                        let pos = self.val(&format!("trunc i64 {low} to i8"));
                        (addr, pos)
                    }
                };
                let one = self.val(&format!("shl i8 1, {pos}"));
                let old = if locked {
                    let (op, operand) = match m {
                        Mnemonic::Bts => ("or", one.clone()),
                        Mnemonic::Btr => ("and", self.val(&format!("xor i8 {one}, -1"))),
                        _ => ("xor", one.clone()),
                    };
                    let ptr = self.guest_ptr(&addr);
                    self.atomic_rmw(op, &ptr, &operand, 8)
                } else {
                    let old = self.load_mem(&addr, 8);
                    let new = match m {
                        Mnemonic::Bts => Some(self.val(&format!("or i8 {old}, {one}"))),
                        Mnemonic::Btr => {
                            let keep = self.val(&format!("xor i8 {one}, -1"));
                            Some(self.val(&format!("and i8 {old}, {keep}")))
                        }
                        Mnemonic::Btc => Some(self.val(&format!("xor i8 {old}, {one}"))),
                        _ => None,
                    };
                    if let Some(new) = new {
                        self.store_mem(&addr, &new, 8);
                    }
                    old
                };
                let hit = self.val(&format!("and i8 {old}, {one}"));
                self.val(&format!("icmp ne i8 {hit}, 0"))
            }
            kind => return unsupported(format!("bit test destination {kind:?}")),
        };
        self.set_flag(Flag::Cf, &cf);
        Ok(())
    }

    /// `bsf`/`bsr`: ZF = (source == 0); with a zero source the destination register is left
    /// entirely unchanged (CF OF SF AF PF undefined, left unchanged).
    pub(super) fn bit_scan(&mut self, ins: &Instruction) -> R<()> {
        let bits = self.op_bits(ins, 0)?;
        let t = ty(bits);
        let src = self.read_op(ins, 1, bits)?;
        let zero = self.val(&format!("icmp eq {t} {src}, 0"));
        let index = if ins.mnemonic() == Mnemonic::Bsf {
            self.bit_intrinsic(BitIntrinsic::Cttz, &src, bits)
        } else {
            let lz = self.bit_intrinsic(BitIntrinsic::Ctlz, &src, bits);
            self.val(&format!("sub {t} {}, {lz}", bits - 1))
        };
        self.set_flag(Flag::Zf, &zero);
        if ins.op_kind(0) != OpKind::Register {
            return unsupported("bit scan destination");
        }
        match reg_ref(ins.op_register(0))? {
            RegRef::Low(idx, 64) | RegRef::Low(idx, 32) => {
                let old = self.load_gpr(idx)?;
                let wide = if bits == 64 {
                    index
                } else {
                    self.val(&format!("zext i32 {index} to i64"))
                };
                let v = self.val(&format!("select i1 {zero}, i64 {old}, i64 {wide}"));
                self.store_gpr(idx, &v)
            }
            _ => {
                let old = self.read_op(ins, 0, bits)?;
                let v = self.val(&format!("select i1 {zero}, {t} {old}, {t} {index}"));
                self.write_op(ins, 0, &v, bits)
            }
        }
    }

    /// `tzcnt`/`lzcnt` (CF = source is zero, ZF = result is zero) and `popcnt` (ZF = source is
    /// zero, CF OF SF AF PF cleared). OF SF AF PF are undefined for `tzcnt`/`lzcnt`.
    pub(super) fn bit_count(&mut self, ins: &Instruction) -> R<()> {
        let bits = self.op_bits(ins, 0)?;
        let t = ty(bits);
        let src = self.read_op(ins, 1, bits)?;
        let src_zero = self.val(&format!("icmp eq {t} {src}, 0"));
        let r = match ins.mnemonic() {
            Mnemonic::Popcnt => {
                let r = self.bit_intrinsic(BitIntrinsic::Ctpop, &src, bits);
                self.set_flag(Flag::Zf, &src_zero);
                for flag in [Flag::Cf, Flag::Of, Flag::Sf, Flag::Af, Flag::Pf] {
                    self.set_flag(flag, "false");
                }
                r
            }
            m => {
                let kind = if m == Mnemonic::Tzcnt {
                    BitIntrinsic::Cttz
                } else {
                    BitIntrinsic::Ctlz
                };
                let r = self.bit_intrinsic(kind, &src, bits);
                let r_zero = self.val(&format!("icmp eq {t} {r}, 0"));
                self.set_flag(Flag::Cf, &src_zero);
                self.set_flag(Flag::Zf, &r_zero);
                r
            }
        };
        self.write_op(ins, 0, &r, bits)
    }

    /// `bswap r32/r64`; `bswap r16` (undefined) writes 0 like hardware. No flags.
    pub(super) fn bswap(&mut self, ins: &Instruction) -> R<()> {
        let bits = self.op_bits(ins, 0)?;
        if bits == 16 {
            return self.write_op(ins, 0, "0", 16);
        }
        let v = self.read_op(ins, 0, bits)?;
        let r = self.bswap_value(&v, bits);
        self.write_op(ins, 0, &r, bits)
    }

    /// BMI1/BMI2 general-purpose register instructions (32/64-bit; results zero-extend).
    pub(super) fn bmi(&mut self, ins: &Instruction) -> R<()> {
        use Mnemonic as M;
        let bits = self.op_bits(ins, 0)?;
        let t = ty(bits);
        let m = ins.mnemonic();
        let r = match m {
            M::Andn => {
                // dst = ~src1 & src2; SF ZF from the result, CF = OF = 0, AF PF undefined.
                let a = self.read_op(ins, 1, bits)?;
                let b = self.read_op(ins, 2, bits)?;
                let na = self.val(&format!("xor {t} {a}, -1"));
                let r = self.val(&format!("and {t} {na}, {b}"));
                self.set_sign_zero(&r, bits);
                self.set_flag(Flag::Cf, "false");
                self.set_flag(Flag::Of, "false");
                r
            }
            M::Blsi | M::Blsr | M::Blsmsk => {
                let s = self.read_op(ins, 1, bits)?;
                let r = match m {
                    M::Blsi => {
                        let neg = self.val(&format!("sub {t} 0, {s}"));
                        self.val(&format!("and {t} {s}, {neg}"))
                    }
                    M::Blsr => {
                        let dec = self.val(&format!("sub {t} {s}, 1"));
                        self.val(&format!("and {t} {s}, {dec}"))
                    }
                    _ => {
                        let dec = self.val(&format!("sub {t} {s}, 1"));
                        self.val(&format!("xor {t} {s}, {dec}"))
                    }
                };
                // blsi: CF = (src != 0); blsr/blsmsk: CF = (src == 0). blsmsk clears ZF.
                let pred = if m == M::Blsi { "ne" } else { "eq" };
                let cf = self.val(&format!("icmp {pred} {t} {s}, 0"));
                self.set_sign_zero(&r, bits);
                if m == M::Blsmsk {
                    self.set_flag(Flag::Zf, "false");
                }
                self.set_flag(Flag::Cf, &cf);
                self.set_flag(Flag::Of, "false");
                r
            }
            M::Bextr => {
                // start = ctrl[7:0], len = ctrl[15:8]; bits past the operand read as zero.
                let s = self.read_op(ins, 1, bits)?;
                let ctrl = self.read_op(ins, 2, bits)?;
                let start = self.val(&format!("and {t} {ctrl}, 255"));
                let len_hi = self.val(&format!("lshr {t} {ctrl}, 8"));
                let len = self.val(&format!("and {t} {len_hi}, 255"));
                let shifted = self.shift_or(&s, &start, bits, "lshr", "0");
                let low = self.low_mask(&len, bits);
                let r = self.val(&format!("and {t} {shifted}, {low}"));
                let zf = self.val(&format!("icmp eq {t} {r}, 0"));
                self.set_flag(Flag::Zf, &zf);
                self.set_flag(Flag::Cf, "false");
                self.set_flag(Flag::Of, "false");
                r
            }
            M::Bzhi => {
                // n = idx[7:0]; n >= width: dst = src, CF = 1; else clear bits n.., CF = 0.
                let s = self.read_op(ins, 1, bits)?;
                let idx = self.read_op(ins, 2, bits)?;
                let n = self.val(&format!("and {t} {idx}, 255"));
                let low = self.low_mask(&n, bits);
                let r = self.val(&format!("and {t} {s}, {low}"));
                let big = self.val(&format!("icmp uge {t} {n}, {bits}"));
                self.set_sign_zero(&r, bits);
                self.set_flag(Flag::Cf, &big);
                self.set_flag(Flag::Of, "false");
                r
            }
            M::Shlx | M::Shrx | M::Sarx => {
                let s = self.read_op(ins, 1, bits)?;
                let c = self.read_op(ins, 2, bits)?;
                let c = self.val(&format!("and {t} {c}, {}", bits - 1));
                let op = match m {
                    M::Shlx => "shl",
                    M::Shrx => "lshr",
                    _ => "ashr",
                };
                self.val(&format!("{op} {t} {s}, {c}"))
            }
            M::Rorx => {
                let s = self.read_op(ins, 1, bits)?;
                let Some(c) = self.imm_operand(ins, 2)? else {
                    return unsupported("rorx without an immediate count");
                };
                let c = c & u64::from(bits - 1);
                self.support.insert(Support::Fshr(bits));
                self.val(&format!(
                    "call {t} @llvm.fshr.{t}({t} {s}, {t} {s}, {t} {c})"
                ))
            }
            M::Mulx => {
                // hi:lo = rdx * src (unsigned); dst1 = lo first, then dst0 = hi (hi wins).
                let src = self.read_op(ins, 2, bits)?;
                let d_reg = if bits == 64 {
                    Register::RDX
                } else {
                    Register::EDX
                };
                let (d, _) = self.read_reg(d_reg)?;
                let (lo, hi) = if bits == 64 {
                    self.mul128(&d, &src, false)
                } else {
                    let wa = self.val(&format!("zext i32 {d} to i64"));
                    let wb = self.val(&format!("zext i32 {src} to i64"));
                    let p = self.val(&format!("mul i64 {wa}, {wb}"));
                    let lo = self.val(&format!("trunc i64 {p} to i32"));
                    let high = self.val(&format!("lshr i64 {p}, 32"));
                    let hi = self.val(&format!("trunc i64 {high} to i32"));
                    (lo, hi)
                };
                self.write_op(ins, 1, &lo, bits)?;
                hi
            }
            M::Pdep | M::Pext => {
                let s = self.read_op(ins, 1, bits)?;
                let msk = self.read_op(ins, 2, bits)?;
                let (ws, wm) = if bits == 64 {
                    (s, msk)
                } else {
                    (
                        self.val(&format!("zext i32 {s} to i64")),
                        self.val(&format!("zext i32 {msk} to i64")),
                    )
                };
                let (name, support) = if m == M::Pdep {
                    ("__playgta6_pdep", Support::Pdep)
                } else {
                    ("__playgta6_pext", Support::Pext)
                };
                self.support.insert(support);
                let r = self.val(&format!("call i64 @{name}(i64 {ws}, i64 {wm})"));
                if bits == 64 {
                    r
                } else {
                    self.val(&format!("trunc i64 {r} to i32"))
                }
            }
            m => return unsupported(format!("instruction {m:?} is not supported yet")),
        };
        self.write_op(ins, 0, &r, bits)
    }

    fn set_sign_zero(&mut self, r: &str, bits: u32) {
        let zf = self.val(&format!("icmp eq i{bits} {r}, 0"));
        let sf = self.msb(r, bits);
        self.set_flag(Flag::Zf, &zf);
        self.set_flag(Flag::Sf, &sf);
    }

    /// `v <op> amount` when `amount < bits`, else `otherwise` (no poison shift amounts).
    fn shift_or(&mut self, v: &str, amount: &str, bits: u32, op: &str, otherwise: &str) -> String {
        let t = ty(bits);
        let small = self.val(&format!("icmp ult {t} {amount}, {bits}"));
        let safe = self.val(&format!("select i1 {small}, {t} {amount}, {t} 0"));
        let shifted = self.val(&format!("{op} {t} {v}, {safe}"));
        self.val(&format!(
            "select i1 {small}, {t} {shifted}, {t} {otherwise}"
        ))
    }

    /// `(1 << n) - 1`, all ones for `n >= bits`.
    fn low_mask(&mut self, n: &str, bits: u32) -> String {
        let t = ty(bits);
        let one = self.shift_or("1", n, bits, "shl", "0");
        // n >= bits: shift_or gave 0, and 0 - 1 is all ones.
        self.val(&format!("sub {t} {one}, 1"))
    }

    /// The masked shift/rotate count of operand `i` (imm8 or CL) as an i64 value.
    fn masked_count(&mut self, ins: &Instruction, i: u32, bits: u32) -> R<String> {
        let count_mask: u64 = if bits == 64 { 63 } else { 31 };
        match ins.op_kind(i) {
            OpKind::Register if ins.op_register(i) == Register::CL => {
                let (cl, _) = self.read_reg(Register::CL)?;
                let masked = self.val(&format!("and i8 {cl}, {count_mask}"));
                Ok(self.val(&format!("zext i8 {masked} to i64")))
            }
            kind if is_immediate(kind) => match ins.try_immediate(i) {
                Ok(v) => Ok((v & count_mask).to_string()),
                Err(_) => unsupported("unreadable shift count"),
            },
            kind => unsupported(format!("shift count operand {kind:?}")),
        }
    }

    /// `rcl`/`rcr`: rotate the `width + 1`-bit value `CF:dst` by the masked count (modulo 9 for
    /// bytes, 17 for words). A rotation by zero changes nothing; CF is the last bit rotated
    /// out. OF: `msb(result) ^ CF` for `rcl`, `msb(dst) ^ CF` (before) for `rcr`.
    pub(super) fn rotate_through_carry(&mut self, ins: &Instruction) -> R<()> {
        let bits = self.op_bits(ins, 0)?;
        let t = ty(bits);
        let left = ins.mnemonic() == Mnemonic::Rcl;
        let a = self.read_op(ins, 0, bits)?;
        let masked = self.masked_count(ins, 1, bits)?;
        let count = match bits {
            8 => self.val(&format!("urem i64 {masked}, 9")),
            16 => self.val(&format!("urem i64 {masked}, 17")),
            _ => masked.clone(),
        };
        let cf_in = self.get_flag(Flag::Cf);
        let cf_wide = self.val(&format!("zext i1 {cf_in} to i64"));
        let (r, cf) = if bits < 64 {
            // (w+1)-bit rotation inside an i64; count <= w, so every shift amount is in range.
            let a64 = self.val(&format!("zext {t} {a} to i64"));
            let top = self.val(&format!("shl i64 {cf_wide}, {bits}"));
            let v = self.val(&format!("or i64 {top}, {a64}"));
            let back = self.val(&format!("sub i64 {}, {count}", bits + 1));
            let (x, y) = if left {
                (
                    self.val(&format!("shl i64 {v}, {count}")),
                    self.val(&format!("lshr i64 {v}, {back}")),
                )
            } else {
                (
                    self.val(&format!("lshr i64 {v}, {count}")),
                    self.val(&format!("shl i64 {v}, {back}")),
                )
            };
            let joined = self.val(&format!("or i64 {x}, {y}"));
            let res = self.val(&format!("and i64 {joined}, {}", imm(mask(bits + 1), 64)));
            let r = self.val(&format!("trunc i64 {res} to {t}"));
            let cf = self.extract_bit(&res, 64, bits);
            (r, cf)
        } else {
            // count in 0..=63; amounts are masked to stay defined, count 0 is selected away.
            let dec = self.val(&format!("add i64 {count}, 63"));
            let dec = self.val(&format!("and i64 {dec}, 63"));
            let inv = self.val(&format!("sub i64 64, {count}"));
            let inv = self.val(&format!("and i64 {inv}, 63"));
            let (r, out) = if left {
                // (a << n) | (cf << (n - 1)) | (a >> (65 - n)); CF = bit 64 - n of a
                let x = self.val(&format!("shl i64 {a}, {count}"));
                let y = self.val(&format!("shl i64 {cf_wide}, {dec}"));
                let z1 = self.val(&format!("lshr i64 {a}, 1"));
                let z = self.val(&format!("lshr i64 {z1}, {inv}"));
                let xy = self.val(&format!("or i64 {x}, {y}"));
                let r = self.val(&format!("or i64 {xy}, {z}"));
                let out = self.val(&format!("lshr i64 {a}, {inv}"));
                (r, out)
            } else {
                // (a >> n) | (cf << (64 - n)) | (a << (65 - n)); CF = bit n - 1 of a
                let x = self.val(&format!("lshr i64 {a}, {count}"));
                let y = self.val(&format!("shl i64 {cf_wide}, {inv}"));
                let z1 = self.val(&format!("shl i64 {a}, 1"));
                let z = self.val(&format!("shl i64 {z1}, {inv}"));
                let xy = self.val(&format!("or i64 {x}, {y}"));
                let r = self.val(&format!("or i64 {xy}, {z}"));
                let out = self.val(&format!("lshr i64 {a}, {dec}"));
                (r, out)
            };
            let out = self.val(&format!("trunc i64 {out} to i1"));
            let nonzero = self.val(&format!("icmp ne i64 {count}, 0"));
            let r = self.val(&format!("select i1 {nonzero}, i64 {r}, i64 {a}"));
            let cf = self.val(&format!("select i1 {nonzero}, i1 {out}, i1 {cf_in}"));
            (r, cf)
        };
        let of = if left {
            let msb = self.msb(&r, bits);
            self.val(&format!("xor i1 {msb}, {cf}"))
        } else {
            let msb = self.msb(&a, bits);
            self.val(&format!("xor i1 {msb}, {cf_in}"))
        };
        // Flags change only for a non-zero masked count (CF already equals CF_in for a
        // rotation by a multiple of width + 1).
        let changed = self.val(&format!("icmp ne i64 {masked}, 0"));
        let old_of = self.get_flag(Flag::Of);
        let of = self.val(&format!("select i1 {changed}, i1 {of}, i1 {old_of}"));
        self.set_flag(Flag::Cf, &cf);
        self.set_flag(Flag::Of, &of);
        self.write_op(ins, 0, &r, bits)
    }

    /// `shld dst, src, count` / `shrd dst, src, count` (count masked to 5/6 bits; 0 changes
    /// nothing). CF = last bit shifted out of `dst`; SF ZF PF from the result; OF =
    /// `msb(result) ^ msb(dst)`; AF undefined (left unchanged).
    pub(super) fn double_shift(&mut self, ins: &Instruction) -> R<()> {
        let bits = self.op_bits(ins, 0)?;
        if bits == 8 || self.op_bits(ins, 1)? != bits {
            return unsupported("double shift operand size");
        }
        let t = ty(bits);
        let left = ins.mnemonic() == Mnemonic::Shld;
        let a = self.read_op(ins, 0, bits)?;
        let b = self.read_op(ins, 1, bits)?;
        let count = self.masked_count(ins, 2, bits)?;
        let (r, cf) = if bits == 16 {
            // 32-bit concatenation; counts 1..=16 are exact.
            let a32 = self.val(&format!("zext i16 {a} to i32"));
            let b32 = self.val(&format!("zext i16 {b} to i32"));
            let n = self.val(&format!("trunc i64 {count} to i32"));
            let dec = self.val(&format!("add i32 {n}, 31"));
            let dec = self.val(&format!("and i32 {dec}, 31"));
            if left {
                let hi = self.val(&format!("shl i32 {a32}, 16"));
                let wide = self.val(&format!("or i32 {hi}, {b32}"));
                let shifted = self.val(&format!("shl i32 {wide}, {n}"));
                let top = self.val(&format!("lshr i32 {shifted}, 16"));
                let r = self.val(&format!("trunc i32 {top} to i16"));
                // bit 16 - n of dst = bit 32 - n of wide = bit 31 of (wide << (n - 1))
                let pre = self.val(&format!("shl i32 {wide}, {dec}"));
                let cf = self.msb(&pre, 32);
                (r, cf)
            } else {
                let hi = self.val(&format!("shl i32 {b32}, 16"));
                let wide = self.val(&format!("or i32 {hi}, {a32}"));
                let shifted = self.val(&format!("lshr i32 {wide}, {n}"));
                let r = self.val(&format!("trunc i32 {shifted} to i16"));
                let pre = self.val(&format!("lshr i32 {wide}, {dec}"));
                let cf = self.val(&format!("trunc i32 {pre} to i1"));
                (r, cf)
            }
        } else {
            let n = if bits == 64 {
                count.clone()
            } else {
                self.val(&format!("trunc i64 {count} to i32"))
            };
            let dec = self.val(&format!("add {t} {n}, {}", bits - 1));
            let dec = self.val(&format!("and {t} {dec}, {}", bits - 1));
            if left {
                self.support.insert(Support::Fshl(bits));
                let r = self.val(&format!(
                    "call {t} @llvm.fshl.{t}({t} {a}, {t} {b}, {t} {n})"
                ));
                // bit width - n of dst = msb of (dst << (n - 1))
                let pre = self.val(&format!("shl {t} {a}, {dec}"));
                let cf = self.msb(&pre, bits);
                (r, cf)
            } else {
                self.support.insert(Support::Fshr(bits));
                let r = self.val(&format!(
                    "call {t} @llvm.fshr.{t}({t} {b}, {t} {a}, {t} {n})"
                ));
                let pre = self.val(&format!("lshr {t} {a}, {dec}"));
                let cf = self.val(&format!("trunc {t} {pre} to i1"));
                (r, cf)
            }
        };
        let msb_r = self.msb(&r, bits);
        let msb_a = self.msb(&a, bits);
        let of = self.val(&format!("xor i1 {msb_r}, {msb_a}"));
        let nonzero = self.val(&format!("icmp ne i64 {count}, 0"));
        let [zf, sf, pf] = self.zsp(&r, bits);
        for (flag, new) in [
            (Flag::Cf, cf),
            (Flag::Of, of),
            (Flag::Zf, zf),
            (Flag::Sf, sf),
            (Flag::Pf, pf),
        ] {
            let old = self.get_flag(flag);
            let v = self.val(&format!("select i1 {nonzero}, i1 {new}, i1 {old}"));
            self.set_flag(flag, &v);
        }
        self.write_op(ins, 0, &r, bits)
    }
}
