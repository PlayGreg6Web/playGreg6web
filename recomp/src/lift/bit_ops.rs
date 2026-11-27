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

use super::{
    Flag, FlagSource, FunctionLifter, R, RegRef, imm, is_immediate, mask, reg_ref, ty, unsupported,
};
use crate::lift::{BitIntrinsic, Support};

/// `pdep`: scatters the low bits of `src` to the set bits of `mask` (lowest first).
pub(in crate::lift) const PDEP_IR: &str = "\
define internal i64 @__playgta6_pdep(i64 %src, i64 %mask) unnamed_addr #0 {
entry:
  br label %loop
loop:
  %m = phi i64 [ %mask, %entry ], [ %m.next, %body ]
  %s = phi i64 [ %src, %entry ], [ %s.next, %body ]
  %r = phi i64 [ 0, %entry ], [ %r.next, %body ]
  %done = icmp eq i64 %m, 0
  br i1 %done, label %exit, label %body
body:
  %neg = sub i64 0, %m
  %low = and i64 %m, %neg
  %take = trunc i64 %s to i1
  %add = select i1 %take, i64 %low, i64 0
  %r.next = or i64 %r, %add
  %m.next = xor i64 %m, %low
  %s.next = lshr i64 %s, 1
  br label %loop
exit:
  ret i64 %r
}
";

/// `pext`: gathers the bits of `src` at the set bits of `mask` into the low bits.
pub(in crate::lift) const PEXT_IR: &str = "\
define internal i64 @__playgta6_pext(i64 %src, i64 %mask) unnamed_addr #0 {
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
        if let Some(c) = Self::const_u64(v, bits) {
            let swapped = match bits {
                16 => u64::from((c as u16).swap_bytes()),
                32 => u64::from((c as u32).swap_bytes()),
                64 => c.swap_bytes(),
                _ => c,
            };
            return imm(swapped, bits);
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

    /// Parses a lift-time constant integer literal masked to `bits`.
    fn const_u64(v: &str, bits: u32) -> Option<u64> {
        v.parse::<i64>().ok().map(|c| (c as u64) & mask(bits))
    }

    /// Computes `CF = (v >> bit) & 1` for a known bit position `bit < bits`, tightening
    /// bit 0 to `trunc` and bit `bits - 1` to `icmp slt ..., 0`.
    fn bit_test_cf_imm(&mut self, v: &str, bit: u32, bits: u32) -> String {
        let t = ty(bits);
        if bit == 0 {
            self.val(&format!("trunc {t} {v} to i1"))
        } else if bit == bits - 1 {
            self.msb(v, bits)
        } else {
            let m = imm(1u64 << bit, bits);
            let hit = self.val(&format!("and {t} {v}, {m}"));
            self.val(&format!("icmp ne {t} {hit}, 0"))
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
        let want_cf = self.flag_live(Flag::Cf);
        let (cf, lazy_source) = match ins.op_kind(0) {
            OpKind::Register => {
                if locked {
                    return unsupported("lock prefix with a register destination (#UD)");
                }
                let v = self.read_op(ins, 0, bits)?;
                // Register `bt*` masks both immediate and register bit offsets modulo `bits`.
                let (bit_imm, dyn_offset) = match offset_imm {
                    Some(o) => (Some((o & u64::from(bits - 1)) as u32), None),
                    None => {
                        let o = self.read_op(ins, 1, bits)?;
                        if let Some(c) = Self::const_u64(&o, bits) {
                            (Some((c & u64::from(bits - 1)) as u32), None)
                        } else {
                            (None, Some(o))
                        }
                    }
                };
                if let Some(bit) = bit_imm {
                    let bit_mask = 1u64 << bit;
                    let one = imm(bit_mask, bits);
                    let keep = imm(!bit_mask, bits);
                    let cf = want_cf.then(|| self.bit_test_cf_imm(&v, bit, bits));
                    let new = match m {
                        Mnemonic::Bts => Some(self.val(&format!("or {t} {v}, {one}"))),
                        Mnemonic::Btr => Some(self.val(&format!("and {t} {v}, {keep}"))),
                        Mnemonic::Btc => Some(self.val(&format!("xor {t} {v}, {one}"))),
                        _ => None,
                    };
                    if let Some(new) = new {
                        self.write_op(ins, 0, &new, bits)?;
                    }
                    let lazy = FlagSource::BitTestBit { val: v, bit, bits };
                    (cf, Some(lazy))
                } else if m == Mnemonic::Bt {
                    let o = dyn_offset.expect("dynamic bit offset");
                    let index = self.val(&format!("and {t} {o}, {}", bits - 1));
                    let cf = want_cf.then(|| {
                        let shifted = self.val(&format!("lshr {t} {v}, {index}"));
                        self.val(&format!("trunc {t} {shifted} to i1"))
                    });
                    let lazy = FlagSource::BitTestShift {
                        val: v,
                        index,
                        bits,
                    };
                    (cf, Some(lazy))
                } else {
                    let o = dyn_offset.expect("dynamic bit offset");
                    let index = self.val(&format!("and {t} {o}, {}", bits - 1));
                    let one = self.val(&format!("shl {t} 1, {index}"));
                    let cf = want_cf.then(|| {
                        let hit = self.val(&format!("and {t} {v}, {one}"));
                        self.val(&format!("icmp ne {t} {hit}, 0"))
                    });
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
                    let lazy = FlagSource::BitTest {
                        val: v,
                        mask: one,
                        bits,
                    };
                    (cf, Some(lazy))
                }
            }
            OpKind::Memory => {
                let ea = self.mem_ea(ins)?;
                let (addr, one, keep_imm, bit_imm_byte) = match offset_imm {
                    Some(o) => {
                        let o = o & u64::from(bits - 1);
                        let addr = self.offset_addr(&ea, o >> 3);
                        let bit_pos = (o & 7) as u32;
                        let bit = 1u64 << bit_pos;
                        (addr, imm(bit, 8), Some(imm(!bit, 8)), Some(bit_pos))
                    }
                    None => {
                        let o = self.read_op(ins, 1, bits)?;
                        if let Some(c) = Self::const_u64(&o, bits) {
                            let shift = 64 - bits;
                            let signed = (((c << shift) as i64) >> shift) as i64;
                            let byte_off = (signed >> 3) as u64;
                            let addr = self.offset_addr(&ea, byte_off);
                            let bit_pos = (signed & 7) as u32;
                            let bit = 1u64 << bit_pos;
                            (addr, imm(bit, 8), Some(imm(!bit, 8)), Some(bit_pos))
                        } else {
                            let wide = if bits == 64 {
                                o
                            } else {
                                self.val(&format!("sext {t} {o} to i64"))
                            };
                            let byte_offset = self.val(&format!("ashr i64 {wide}, 3"));
                            let addr = self.val(&format!("add i64 {ea}, {byte_offset}"));
                            let low = self.val(&format!("and i64 {wide}, 7"));
                            let pos = self.val(&format!("trunc i64 {low} to i8"));
                            let one = self.val(&format!("shl i8 1, {pos}"));
                            (addr, one, None, None)
                        }
                    }
                };
                let old = if locked {
                    let (op, operand) = match m {
                        Mnemonic::Bts => ("or", one.clone()),
                        Mnemonic::Btr => {
                            let keep = match keep_imm {
                                Some(k) => k,
                                None => self.val(&format!("xor i8 {one}, -1")),
                            };
                            ("and", keep)
                        }
                        _ => ("xor", one.clone()),
                    };
                    let (ptr, mark) = self.guest_ptr_write(&addr);
                    let old = self.atomic_rmw(op, &ptr, &operand, 8);
                    self.finish_write(mark);
                    old
                } else {
                    let old = self.load_mem(&addr, 8);
                    let new = match m {
                        Mnemonic::Bts => Some(self.val(&format!("or i8 {old}, {one}"))),
                        Mnemonic::Btr => {
                            let keep = match keep_imm {
                                Some(k) => k,
                                None => self.val(&format!("xor i8 {one}, -1")),
                            };
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
                let cf = want_cf.then(|| {
                    if let Some(bit) = bit_imm_byte {
                        self.bit_test_cf_imm(&old, bit, 8)
                    } else {
                        let hit = self.val(&format!("and i8 {old}, {one}"));
                        self.val(&format!("icmp ne i8 {hit}, 0"))
                    }
                });
                let lazy = if locked {
                    None
                } else if let Some(bit) = bit_imm_byte {
                    Some(FlagSource::BitTestBit {
                        val: old,
                        bit,
                        bits: 8,
                    })
                } else {
                    Some(FlagSource::BitTest {
                        val: old,
                        mask: one,
                        bits: 8,
                    })
                };
                (cf, lazy)
            }
            kind => return unsupported(format!("bit test destination {kind:?}")),
        };
        if let Some(cf) = cf {
            self.set_flag(Flag::Cf, &cf);
            if !locked {
                self.known.flag_source = Some(FlagSource::CarryBit { cf });
            }
        } else if let Some(lazy) = lazy_source {
            self.known.flag_source = Some(lazy);
        }
        Ok(())
    }

    /// `bsf`/`bsr`: ZF = (source == 0); with a zero source the destination register is left
    /// entirely unchanged (CF OF SF AF PF undefined, left unchanged).
    pub(super) fn bit_scan(&mut self, ins: &Instruction) -> R<()> {
        let bits = self.op_bits(ins, 0)?;
        let t = ty(bits);
        let src = self.read_op(ins, 1, bits)?;
        if ins.op_kind(0) != OpKind::Register {
            return unsupported("bit scan destination");
        }
        if let Some(c) = Self::const_u64(&src, bits) {
            if c == 0 {
                self.set_flag(Flag::Zf, "true");
                self.known.flag_source = Some(FlagSource::ZeroBit {
                    is_zero: "true".to_string(),
                });
                return Ok(());
            }
            let idx_val = if ins.mnemonic() == Mnemonic::Bsf {
                u64::from(c.trailing_zeros())
            } else {
                u64::from(bits - 1 - (c << (64 - bits)).leading_zeros())
            };
            self.set_flag(Flag::Zf, "false");
            let res = self.write_op(ins, 0, &imm(idx_val, bits), bits);
            self.known.flag_source = Some(FlagSource::ZeroBit {
                is_zero: "false".to_string(),
            });
            return res;
        }
        let zero = self.val(&format!("icmp eq {t} {src}, 0"));
        let index = if ins.mnemonic() == Mnemonic::Bsf {
            self.bit_intrinsic(BitIntrinsic::Cttz, &src, bits)
        } else {
            let lz = self.bit_intrinsic(BitIntrinsic::Ctlz, &src, bits);
            self.val(&format!("sub {t} {}, {lz}", bits - 1))
        };
        self.set_flag(Flag::Zf, &zero);
        let res = match reg_ref(ins.op_register(0))? {
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
        };
        self.known.flag_source = Some(FlagSource::ZeroBit { is_zero: zero });
        res
    }

    /// `tzcnt`/`lzcnt` (CF = source is zero, ZF = result is zero) and `popcnt` (ZF = source is
    /// zero, CF OF SF AF PF cleared). OF SF AF PF are undefined for `tzcnt`/`lzcnt`.
    pub(super) fn bit_count(&mut self, ins: &Instruction) -> R<()> {
        let bits = self.op_bits(ins, 0)?;
        let t = ty(bits);
        let src = self.read_op(ins, 1, bits)?;
        let r = match ins.mnemonic() {
            Mnemonic::Popcnt => {
                let r = if let Some(c) = Self::const_u64(&src, bits) {
                    imm(u64::from(c.count_ones()), bits)
                } else {
                    self.bit_intrinsic(BitIntrinsic::Ctpop, &src, bits)
                };
                if self.flag_live(Flag::Zf) {
                    let src_zero = self.val(&format!("icmp eq {t} {src}, 0"));
                    self.set_flag(Flag::Zf, &src_zero);
                }
                for flag in [Flag::Cf, Flag::Of, Flag::Sf, Flag::Af, Flag::Pf] {
                    self.set_flag(flag, "false");
                }
                self.known.flag_source = Some(FlagSource::ZeroClearRest { val: src, bits });
                r
            }
            m => {
                let is_tzcnt = m == Mnemonic::Tzcnt;
                let r = if let Some(c) = Self::const_u64(&src, bits) {
                    let cnt = if c == 0 {
                        bits
                    } else if is_tzcnt {
                        c.trailing_zeros()
                    } else {
                        (c << (64 - bits)).leading_zeros()
                    };
                    imm(u64::from(cnt), bits)
                } else {
                    let kind = if is_tzcnt {
                        BitIntrinsic::Cttz
                    } else {
                        BitIntrinsic::Ctlz
                    };
                    self.bit_intrinsic(kind, &src, bits)
                };
                if self.flag_live(Flag::Cf) {
                    let src_zero = self.val(&format!("icmp eq {t} {src}, 0"));
                    self.set_flag(Flag::Cf, &src_zero);
                }
                if self.flag_live(Flag::Zf) {
                    // `cttz(src) == 0 <=> lsb(src) == 1`; `ctlz(src) == 0 <=> msb(src) == 1`.
                    // Checking `src` directly breaks the data dependency on `cttz`/`ctlz`.
                    let r_zero = if is_tzcnt {
                        self.val(&format!("trunc {t} {src} to i1"))
                    } else {
                        self.msb(&src, bits)
                    };
                    self.set_flag(Flag::Zf, &r_zero);
                }
                self.known.flag_source = Some(FlagSource::BitCount {
                    src,
                    is_tzcnt,
                    bits,
                });
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
                let r = if a == b {
                    "0".to_string()
                } else {
                    let na = self.val(&format!("xor {t} {a}, -1"));
                    self.val(&format!("and {t} {na}, {b}"))
                };
                self.set_sign_zero(&r, bits);
                self.set_flag(Flag::Cf, "false");
                self.set_flag(Flag::Of, "false");
                self.known.flag_source = Some(FlagSource::LogicZero {
                    result: r.clone(),
                    bits,
                });
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
                if self.flag_live(Flag::Cf) {
                    let pred = if m == M::Blsi { "ne" } else { "eq" };
                    let cf = self.val(&format!("icmp {pred} {t} {s}, 0"));
                    self.set_flag(Flag::Cf, &cf);
                }
                if m == M::Blsmsk {
                    if self.flag_live(Flag::Sf) {
                        let sf = self.msb(&r, bits);
                        self.set_flag(Flag::Sf, &sf);
                    }
                    self.set_flag(Flag::Zf, "false");
                } else {
                    self.set_sign_zero(&r, bits);
                }
                self.set_flag(Flag::Of, "false");
                match m {
                    M::Blsi => {
                        self.known.flag_source = Some(FlagSource::Blsi {
                            result: r.clone(),
                            bits,
                        });
                    }
                    M::Blsr => {
                        self.known.flag_source = Some(FlagSource::Blsr {
                            src: s,
                            result: r.clone(),
                            bits,
                        });
                    }
                    _ => {}
                }
                r
            }
            M::Bextr => {
                // start = ctrl[7:0], len = ctrl[15:8]; bits past the operand read as zero.
                let s = self.read_op(ins, 1, bits)?;
                let ctrl = self.read_op(ins, 2, bits)?;
                let (r, always_zero) = if let Some(c) = Self::const_u64(&ctrl, bits) {
                    let start = (c & 0xFF) as u32;
                    let len = ((c >> 8) & 0xFF) as u32;
                    if start >= bits || len == 0 {
                        ("0".to_string(), true)
                    } else {
                        let eff_len = len.min(bits - start);
                        let shifted = if start == 0 {
                            s
                        } else {
                            self.val(&format!("lshr {t} {s}, {start}"))
                        };
                        let r = if eff_len >= bits - start {
                            shifted
                        } else {
                            let m = imm(mask(eff_len), bits);
                            self.val(&format!("and {t} {shifted}, {m}"))
                        };
                        (r, false)
                    }
                } else {
                    let start = self.val(&format!("and {t} {ctrl}, 255"));
                    let len_hi = self.val(&format!("lshr {t} {ctrl}, 8"));
                    let len = self.val(&format!("and {t} {len_hi}, 255"));
                    let shifted = self.shift_or(&s, &start, bits, "lshr", "0");
                    let low = self.low_mask(&len, bits);
                    (self.val(&format!("and {t} {shifted}, {low}")), false)
                };
                if always_zero {
                    self.set_flag(Flag::Zf, "true");
                } else if self.flag_live(Flag::Zf) {
                    let zf = self.val(&format!("icmp eq {t} {r}, 0"));
                    self.set_flag(Flag::Zf, &zf);
                }
                self.set_flag(Flag::Cf, "false");
                self.set_flag(Flag::Of, "false");
                self.known.flag_source = Some(if always_zero {
                    FlagSource::ZeroBit {
                        is_zero: "true".to_string(),
                    }
                } else {
                    FlagSource::ZeroOnly {
                        val: r.clone(),
                        bits,
                    }
                });
                r
            }
            M::Bzhi => {
                // n = idx[7:0]; n >= width: dst = src, CF = 1; else clear bits n.., CF = 0.
                let s = self.read_op(ins, 1, bits)?;
                let idx = self.read_op(ins, 2, bits)?;
                let (r, always_zero) = if let Some(c) = Self::const_u64(&idx, bits) {
                    let n = (c & 0xFF) as u32;
                    if n >= bits {
                        self.set_flag(Flag::Cf, "true");
                        (s, false)
                    } else if n == 0 {
                        self.set_flag(Flag::Cf, "false");
                        ("0".to_string(), true)
                    } else {
                        let m = imm(mask(n), bits);
                        let r = self.val(&format!("and {t} {s}, {m}"));
                        self.set_flag(Flag::Cf, "false");
                        (r, false)
                    }
                } else {
                    let n = self.val(&format!("and {t} {idx}, 255"));
                    let low = self.low_mask(&n, bits);
                    let r = self.val(&format!("and {t} {s}, {low}"));
                    if self.flag_live(Flag::Cf) {
                        let big = self.val(&format!("icmp uge {t} {n}, {bits}"));
                        self.set_flag(Flag::Cf, &big);
                    }
                    (r, false)
                };
                if always_zero {
                    self.set_flag(Flag::Zf, "true");
                    self.set_flag(Flag::Sf, "false");
                } else {
                    self.set_sign_zero(&r, bits);
                }
                self.set_flag(Flag::Of, "false");
                self.known.flag_source = Some(FlagSource::ResultZeroSign {
                    result: r.clone(),
                    bits,
                });
                r
            }
            M::Shlx | M::Shrx | M::Sarx => {
                let s = self.read_op(ins, 1, bits)?;
                let c = self.read_op(ins, 2, bits)?;
                let op = match m {
                    M::Shlx => "shl",
                    M::Shrx => "lshr",
                    _ => "ashr",
                };
                if let Some(cv) = Self::const_u64(&c, bits) {
                    let amt = cv & u64::from(bits - 1);
                    if amt == 0 {
                        s
                    } else {
                        self.val(&format!("{op} {t} {s}, {amt}"))
                    }
                } else {
                    let c = self.val(&format!("and {t} {c}, {}", bits - 1));
                    self.val(&format!("{op} {t} {s}, {c}"))
                }
            }
            M::Rorx => {
                let s = self.read_op(ins, 1, bits)?;
                let Some(c) = self.imm_operand(ins, 2)? else {
                    return unsupported("rorx without an immediate count");
                };
                let c = c & u64::from(bits - 1);
                if c == 0 {
                    s
                } else if let Some(vs) = Self::const_u64(&s, bits) {
                    let rot = c as u32;
                    let rotated = if bits == 32 {
                        u64::from((vs as u32).rotate_right(rot))
                    } else {
                        vs.rotate_right(rot)
                    };
                    imm(rotated, bits)
                } else {
                    self.support.insert(Support::Fshr(bits));
                    self.val(&format!(
                        "call {t} @llvm.fshr.{t}({t} {s}, {t} {s}, {t} {c})"
                    ))
                }
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
                let same_dst = ins.op_kind(0) == OpKind::Register
                    && ins.op_kind(1) == OpKind::Register
                    && ins.op_register(0) == ins.op_register(1);
                let (lo, hi) = if bits == 64 {
                    let (lo, hi) = self.mul128(&d, &src, false);
                    ((!same_dst).then_some(lo), hi)
                } else {
                    let wa = self.val(&format!("zext i32 {d} to i64"));
                    let wb = self.val(&format!("zext i32 {src} to i64"));
                    let p = self.val(&format!("mul i64 {wa}, {wb}"));
                    let lo = (!same_dst).then(|| self.val(&format!("trunc i64 {p} to i32")));
                    let high = self.val(&format!("lshr i64 {p}, 32"));
                    let hi = self.val(&format!("trunc i64 {high} to i32"));
                    (lo, hi)
                };
                if let Some(lo) = lo {
                    self.write_op(ins, 1, &lo, bits)?;
                }
                hi
            }
            M::Pdep | M::Pext => {
                let s = self.read_op(ins, 1, bits)?;
                let msk = self.read_op(ins, 2, bits)?;
                if let Some(m_val) = Self::const_u64(&msk, bits)
                    && let Some(fast) = self.try_const_pdep_pext(m == M::Pdep, &s, m_val, bits)
                {
                    fast
                } else {
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
            }
            m => return unsupported(format!("instruction {m:?} is not supported yet")),
        };
        self.write_op(ins, 0, &r, bits)
    }

    /// Fast path for `pdep`/`pext` with a compile-time constant mask `m_val`:
    /// - `0` -> `0`, `all_ones(bits)` -> `src`
    /// - 1..=4 contiguous runs of 1-bits -> direct shift + mask + or sequence without calling
    ///   `@__playgta6_pdep`/`@__playgta6_pext`.
    fn try_const_pdep_pext(
        &mut self,
        is_pdep: bool,
        src: &str,
        m_val: u64,
        bits: u32,
    ) -> Option<String> {
        let full = mask(bits);
        let m_val = m_val & full;
        if m_val == 0 {
            return Some("0".to_string());
        }
        if m_val == full {
            return Some(src.to_string());
        }
        // Decompose `m_val` into up to 4 contiguous runs `(pos, len)` from LSB to MSB.
        let mut runs: [(u32, u32); 4] = [(0, 0); 4];
        let mut run_count = 0usize;
        let mut rem = m_val;
        while rem != 0 {
            if run_count == runs.len() {
                return None;
            }
            let start = rem.trailing_zeros();
            let shifted = rem >> start;
            let len = (!shifted).trailing_zeros().min(bits - start);
            runs[run_count] = (start, len);
            run_count += 1;
            if start + len >= bits {
                break;
            }
            rem &= !(mask(len) << start);
        }
        let t = ty(bits);
        let mut packed_pos = 0u32;
        let mut acc: Option<String> = None;
        for &(pos, len) in &runs[..run_count] {
            let shift = pos - packed_pos;
            let part = if is_pdep {
                let shifted = if shift == 0 {
                    src.to_string()
                } else {
                    self.val(&format!("shl {t} {src}, {shift}"))
                };
                if packed_pos == 0 && pos + len == bits {
                    shifted
                } else {
                    let cm = imm(mask(len) << pos, bits);
                    self.val(&format!("and {t} {shifted}, {cm}"))
                }
            } else {
                let shifted = if shift == 0 {
                    src.to_string()
                } else {
                    self.val(&format!("lshr {t} {src}, {shift}"))
                };
                if packed_pos == 0 && pos + len == bits {
                    shifted
                } else {
                    let cm = imm(mask(len) << packed_pos, bits);
                    self.val(&format!("and {t} {shifted}, {cm}"))
                }
            };
            acc = Some(match acc {
                Some(prev) => self.val(&format!("or {t} {prev}, {part}")),
                None => part,
            });
            packed_pos += len;
        }
        acc
    }

    fn set_sign_zero(&mut self, r: &str, bits: u32) {
        if self.flag_live(Flag::Zf) {
            let zf = self.val(&format!("icmp eq i{bits} {r}, 0"));
            self.set_flag(Flag::Zf, &zf);
        }
        if self.flag_live(Flag::Sf) {
            let sf = self.msb(r, bits);
            self.set_flag(Flag::Sf, &sf);
        }
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

    /// The masked shift/rotate count of operand `i` (imm8 or CL) in `dst_bits` (`i32` or `i64`),
    /// along with the constant value when immediate or known constant in `rcx`.
    fn masked_count(
        &mut self,
        ins: &Instruction,
        i: u32,
        bits: u32,
        dst_bits: u32,
    ) -> R<(String, Option<u64>)> {
        let count_mask: u64 = if bits == 64 { 63 } else { 31 };
        match ins.op_kind(i) {
            OpKind::Register if ins.op_register(i) == Register::CL => {
                let rcx = self.load_gpr(super::integer::RCX)?;
                if let Some(v) = super::memory_access::const_addr(&rcx) {
                    let c = v & count_mask;
                    return Ok((c.to_string(), Some(c)));
                }
                let raw = if dst_bits == 64 {
                    rcx
                } else {
                    self.val(&format!("trunc i64 {rcx} to i{dst_bits}"))
                };
                Ok((
                    self.val(&format!("and i{dst_bits} {raw}, {count_mask}")),
                    None,
                ))
            }
            kind if is_immediate(kind) => match ins.try_immediate(i) {
                Ok(v) => {
                    let c = v & count_mask;
                    Ok((c.to_string(), Some(c)))
                }
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
        let (masked, imm_count) = self.masked_count(ins, 1, bits, 64)?;
        if imm_count == Some(0) {
            return self.write_op(ins, 0, &a, bits);
        }
        let (count, imm_mod) = match (bits, imm_count) {
            (8, Some(c)) => ((c % 9).to_string(), Some(c % 9)),
            (16, Some(c)) => ((c % 17).to_string(), Some(c % 17)),
            (8, None) => (self.val(&format!("urem i64 {masked}, 9")), None),
            (16, None) => (self.val(&format!("urem i64 {masked}, 17")), None),
            (_, imm_c) => (masked.clone(), imm_c),
        };
        let cf_in = self.get_flag(Flag::Cf);
        let cf_wide = self.val(&format!("zext i1 {cf_in} to i64"));
        let (r, cf) = if bits < 64 {
            if imm_mod == Some(0) {
                // Rotation by a multiple of width + 1 leaves dst and CF unchanged (OF still set).
                (a.clone(), cf_in.clone())
            } else {
                // (w+1)-bit rotation inside an i64; count <= w, so every shift amount is in range.
                let a64 = self.val(&format!("zext {t} {a} to i64"));
                let top = self.val(&format!("shl i64 {cf_wide}, {bits}"));
                let v = self.val(&format!("or i64 {top}, {a64}"));
                let back = match imm_mod {
                    Some(c) => (u64::from(bits + 1) - c).to_string(),
                    None => self.val(&format!("sub i64 {}, {count}", bits + 1)),
                };
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
            }
        } else {
            // count in 0..=63; amounts are masked to stay defined, count 0 is selected away.
            let (dec, inv) = match imm_count {
                Some(c) => ((c - 1).to_string(), (64 - c).to_string()),
                None => {
                    let dec = self.val(&format!("add i64 {count}, 63"));
                    let dec = self.val(&format!("and i64 {dec}, 63"));
                    let inv = self.val(&format!("sub i64 64, {count}"));
                    let inv = self.val(&format!("and i64 {inv}, 63"));
                    (dec, inv)
                }
            };
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
            if imm_count.is_some() {
                (r, out)
            } else {
                let nonzero = self.val(&format!("icmp ne i64 {count}, 0"));
                let r = self.val(&format!("select i1 {nonzero}, i64 {r}, i64 {a}"));
                let cf = self.val(&format!("select i1 {nonzero}, i1 {out}, i1 {cf_in}"));
                (r, cf)
            }
        };
        if self.flag_live(Flag::Of) {
            let of = if left {
                let msb = self.msb(&r, bits);
                self.val(&format!("xor i1 {msb}, {cf}"))
            } else {
                let msb = self.msb(&a, bits);
                self.val(&format!("xor i1 {msb}, {cf_in}"))
            };
            let of = if imm_count.is_some() {
                of
            } else {
                let changed = self.val(&format!("icmp ne i64 {masked}, 0"));
                let old_of = self.get_flag(Flag::Of);
                self.val(&format!("select i1 {changed}, i1 {of}, i1 {old_of}"))
            };
            self.set_flag(Flag::Of, &of);
        }
        self.set_flag(Flag::Cf, &cf);
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
        let cnt_bits = if bits == 64 { 64 } else { 32 };
        let (count, imm_count) = self.masked_count(ins, 2, bits, cnt_bits)?;
        if imm_count == Some(0) {
            return self.write_op(ins, 0, &a, bits);
        }
        let want_cf = self.flag_live(Flag::Cf);
        let want_of = self.flag_live(Flag::Of);
        let (r, cf) = if bits == 16 {
            // 32-bit concatenation; counts 1..=16 are exact.
            let a32 = self.val(&format!("zext i16 {a} to i32"));
            let b32 = self.val(&format!("zext i16 {b} to i32"));
            let dec = if want_cf {
                match imm_count {
                    Some(c) => ((c - 1) & 31).to_string(),
                    None => {
                        let dec = self.val(&format!("add i32 {count}, 31"));
                        self.val(&format!("and i32 {dec}, 31"))
                    }
                }
            } else {
                String::new()
            };
            if left {
                let hi = self.val(&format!("shl i32 {a32}, 16"));
                let wide = self.val(&format!("or i32 {hi}, {b32}"));
                let shifted = self.val(&format!("shl i32 {wide}, {count}"));
                let top = self.val(&format!("lshr i32 {shifted}, 16"));
                let r = self.val(&format!("trunc i32 {top} to i16"));
                // bit 16 - n of dst = bit 32 - n of wide = bit 31 of (wide << (n - 1))
                let cf = want_cf.then(|| {
                    let pre = self.val(&format!("shl i32 {wide}, {dec}"));
                    self.msb(&pre, 32)
                });
                (r, cf)
            } else {
                let hi = self.val(&format!("shl i32 {b32}, 16"));
                let wide = self.val(&format!("or i32 {hi}, {a32}"));
                let shifted = self.val(&format!("lshr i32 {wide}, {count}"));
                let r = self.val(&format!("trunc i32 {shifted} to i16"));
                let cf = want_cf.then(|| {
                    let pre = self.val(&format!("lshr i32 {wide}, {dec}"));
                    self.val(&format!("trunc i32 {pre} to i1"))
                });
                (r, cf)
            }
        } else if left {
            let r = if let (Some(va), Some(vb), Some(c)) = (
                Self::const_u64(&a, bits),
                Self::const_u64(&b, bits),
                imm_count,
            ) {
                let c = c as u32;
                let res = ((va << c) | (vb >> (bits - c))) & mask(bits);
                imm(res, bits)
            } else {
                self.support.insert(Support::Fshl(bits));
                self.val(&format!(
                    "call {t} @llvm.fshl.{t}({t} {a}, {t} {b}, {t} {count})"
                ))
            };
            let cf = want_cf.then(|| match imm_count {
                Some(1) => self.msb(&a, bits),
                Some(c) => {
                    let out = self.val(&format!("lshr {t} {a}, {}", u64::from(bits) - c));
                    self.val(&format!("trunc {t} {out} to i1"))
                }
                None => {
                    let dec = self.val(&format!("add {t} {count}, {}", bits - 1));
                    let dec = self.val(&format!("and {t} {dec}, {}", bits - 1));
                    let pre = self.val(&format!("shl {t} {a}, {dec}"));
                    self.msb(&pre, bits)
                }
            });
            (r, cf)
        } else {
            let r = if let (Some(va), Some(vb), Some(c)) = (
                Self::const_u64(&a, bits),
                Self::const_u64(&b, bits),
                imm_count,
            ) {
                let c = c as u32;
                let res = ((va >> c) | (vb << (bits - c))) & mask(bits);
                imm(res, bits)
            } else {
                self.support.insert(Support::Fshr(bits));
                self.val(&format!(
                    "call {t} @llvm.fshr.{t}({t} {b}, {t} {a}, {t} {count})"
                ))
            };
            let cf = want_cf.then(|| match imm_count {
                Some(1) => self.val(&format!("trunc {t} {a} to i1")),
                Some(c) => {
                    let out = self.val(&format!("lshr {t} {a}, {}", c - 1));
                    self.val(&format!("trunc {t} {out} to i1"))
                }
                None => {
                    let dec = self.val(&format!("add {t} {count}, {}", bits - 1));
                    let dec = self.val(&format!("and {t} {dec}, {}", bits - 1));
                    let pre = self.val(&format!("lshr {t} {a}, {dec}"));
                    self.val(&format!("trunc {t} {pre} to i1"))
                }
            });
            (r, cf)
        };
        let mut updates: Vec<(Flag, String)> = Vec::new();
        if let Some(cf) = cf {
            updates.push((Flag::Cf, cf));
        }
        if want_of {
            let msb_r = self.msb(&r, bits);
            let msb_a = self.msb(&a, bits);
            let of = self.val(&format!("xor i1 {msb_r}, {msb_a}"));
            updates.push((Flag::Of, of));
        }
        if self.flag_live(Flag::Zf) {
            let zf = self.val(&format!("icmp eq {t} {r}, 0"));
            updates.push((Flag::Zf, zf));
        }
        if self.flag_live(Flag::Sf) {
            let sf = self.msb(&r, bits);
            updates.push((Flag::Sf, sf));
        }
        if self.flag_live(Flag::Pf) {
            let pf = self.parity(&r, bits);
            updates.push((Flag::Pf, pf));
        }
        if !updates.is_empty() {
            if imm_count.is_some() {
                for (flag, new) in updates {
                    self.set_flag(flag, &new);
                }
            } else {
                let nonzero = self.val(&format!("icmp ne i{cnt_bits} {count}, 0"));
                for (flag, new) in updates {
                    let old = self.get_flag(flag);
                    let v = self.val(&format!("select i1 {nonzero}, i1 {new}, i1 {old}"));
                    self.set_flag(flag, &v);
                }
            }
        }
        if let Some(c) = imm_count {
            if bits >= 32 && c < u64::from(bits) {
                let cf_bit = if left {
                    bits - c as u32
                } else {
                    c as u32 - 1
                };
                self.known.flag_source = Some(FlagSource::ShiftImm {
                    result: r.clone(),
                    src: a,
                    cf_bit,
                    bits,
                });
            } else {
                self.known.flag_source = Some(FlagSource::ResultZeroSign {
                    result: r.clone(),
                    bits,
                });
            }
        }
        self.write_op(ins, 0, &r, bits)
    }
}
