//! String instructions: `movs`, `stos`, `lods`, `cmps`, `scas` (byte/word/dword/qword) with and
//! without `rep`/`repe`/`repne`, honouring DF.
//!
//! - One iteration reads `[seg:rsi]` (`fs`/`gs` overrides apply to the source only) and/or
//!   `[rdi]`, then moves `rsi`/`rdi` by `±size` (DF set: downwards).
//! - `rep`: `while rcx != 0 { iteration; rcx -= 1; [repe: stop if ZF = 0] [repne: stop if
//!   ZF = 1] }`. A zero count changes nothing (flags included). `cmps`/`scas` flags are those of
//!   the last executed comparison, `[rsi] - [rdi]` resp. `acc - [rdi]`.
//! - `rep movs` / `rep stos` take a bulk path (`llvm.memmove` / `llvm.memset`) only when it is
//!   observably identical to the element-by-element loop: the count is non-zero, the byte size
//!   does not overflow, and for `movs` the destination does not start inside the source range
//!   in the copy direction (an overlapping forward copy with `src < dst < src + n` replicates a
//!   pattern element by element, which neither `memcpy` nor `memmove` do); for `stos` the
//!   stored value must be one byte repeated. Everything else runs the exact loop.
//! - The 32-bit address-size forms (`ESI`/`EDI`/`ECX`) and `ins`/`outs` are unsupported; so is
//!   `repne` on `movs`/`stos`/`lods` (reserved).

use iced_x86::{Code, Instruction, OpKind, Register};

use super::integer::{RCX, RDI, RSI};
use super::{Alu, FunctionLifter, R, imm, unsupported};
use crate::lift::Support;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum StringOp {
    Movs,
    Stos,
    Lods,
    Cmps,
    Scas,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Repeat {
    Once,
    Rep,
    WhileEqual,
    WhileNotEqual,
}

fn accumulator(bits: u32) -> Register {
    match bits {
        8 => Register::AL,
        16 => Register::AX,
        32 => Register::EAX,
        _ => Register::RAX,
    }
}

impl FunctionLifter<'_> {
    pub(super) fn string_instruction(&mut self, ins: &Instruction) -> R<()> {
        use Code as C;
        use StringOp as S;
        let (op, bits) = match ins.code() {
            C::Movsb_m8_m8 => (S::Movs, 8),
            C::Movsw_m16_m16 => (S::Movs, 16),
            C::Movsd_m32_m32 => (S::Movs, 32),
            C::Movsq_m64_m64 => (S::Movs, 64),
            C::Stosb_m8_AL => (S::Stos, 8),
            C::Stosw_m16_AX => (S::Stos, 16),
            C::Stosd_m32_EAX => (S::Stos, 32),
            C::Stosq_m64_RAX => (S::Stos, 64),
            C::Lodsb_AL_m8 => (S::Lods, 8),
            C::Lodsw_AX_m16 => (S::Lods, 16),
            C::Lodsd_EAX_m32 => (S::Lods, 32),
            C::Lodsq_RAX_m64 => (S::Lods, 64),
            C::Cmpsb_m8_m8 => (S::Cmps, 8),
            C::Cmpsw_m16_m16 => (S::Cmps, 16),
            C::Cmpsd_m32_m32 => (S::Cmps, 32),
            C::Cmpsq_m64_m64 => (S::Cmps, 64),
            C::Scasb_AL_m8 => (S::Scas, 8),
            C::Scasw_AX_m16 => (S::Scas, 16),
            C::Scasd_EAX_m32 => (S::Scas, 32),
            C::Scasq_RAX_m64 => (S::Scas, 64),
            _ => {
                return unsupported(format!(
                    "string instruction {:?} is not supported",
                    ins.mnemonic()
                ));
            }
        };
        for i in 0..ins.op_count() {
            match ins.op_kind(i) {
                OpKind::Register | OpKind::MemorySegRSI | OpKind::MemoryESRDI => {}
                OpKind::MemorySegESI | OpKind::MemoryESEDI => {
                    return unsupported("string instruction with a 32-bit address size");
                }
                kind => return unsupported(format!("string operand kind {kind:?}")),
            }
        }
        let compares = matches!(op, S::Cmps | S::Scas);
        let repeat = match (ins.has_repe_prefix(), ins.has_repne_prefix()) {
            (false, false) => Repeat::Once,
            (true, false) if compares => Repeat::WhileEqual,
            (true, false) => Repeat::Rep,
            (false, true) if compares => Repeat::WhileNotEqual,
            _ => return unsupported(format!("reserved repeat prefix on {op:?}")),
        };
        // Source segment base (fs/gs overrides), computed once.
        let segment = if matches!(op, S::Movs | S::Lods | S::Cmps) {
            Some(ins.memory_segment())
        } else {
            None
        };
        // Signed element step from DF.
        let df = self.load_df();
        let k = bits / 8;
        let step = self.val(&format!("select i1 {df}, i64 {}, i64 {k}", -i64::from(k)));

        if repeat == Repeat::Once {
            return self.string_step(op, bits, &step, segment);
        }

        let head = self.new_label("rep_head");
        let body = self.new_label("rep_body");
        let done = self.new_label("rep_done");
        match op {
            S::Movs => self.rep_movs_bulk(bits, &df, segment, &head, &done)?,
            S::Stos => self.rep_stos_bulk(bits, &df, &head, &done)?,
            _ => self.emit(&format!("br label %{head}")),
        }

        self.start_block(&head);
        let count = self.load_gpr(RCX)?;
        let zero = self.val(&format!("icmp eq i64 {count}, 0"));
        self.emit(&format!("br i1 {zero}, label %{done}, label %{body}"));

        self.start_block(&body);
        self.string_step(op, bits, &step, segment)?;
        let next = self.val(&format!("sub i64 {count}, 1"));
        self.store_gpr(RCX, &next)?;
        match repeat {
            Repeat::WhileEqual | Repeat::WhileNotEqual => {
                let zf = self.get_flag(super::Flag::Zf);
                let (on_set, on_clear) = if repeat == Repeat::WhileEqual {
                    (&head, &done)
                } else {
                    (&done, &head)
                };
                self.emit(&format!("br i1 {zf}, label %{on_set}, label %{on_clear}"));
            }
            _ => self.emit(&format!("br label %{head}")),
        }
        self.start_block(&done);
        Ok(())
    }

    /// Source address `[seg:rsi]`.
    fn string_source(&mut self, segment: Option<Register>) -> R<(String, String)> {
        let rsi = self.load_gpr(RSI)?;
        let addr = match segment {
            Some(seg) => self.add_segment_base(&rsi, seg),
            None => rsi.clone(),
        };
        Ok((rsi, addr))
    }

    /// One iteration of the string instruction.
    fn string_step(
        &mut self,
        op: StringOp,
        bits: u32,
        step: &str,
        segment: Option<Register>,
    ) -> R<()> {
        let acc = accumulator(bits);
        match op {
            StringOp::Movs => {
                let (rsi, src) = self.string_source(segment)?;
                let rdi = self.load_gpr(RDI)?;
                let v = self.load_mem(&src, bits);
                self.store_mem(&rdi, &v, bits);
                self.advance(RSI, &rsi, step)?;
                self.advance(RDI, &rdi, step)?;
            }
            StringOp::Stos => {
                let (v, _) = self.read_reg(acc)?;
                let rdi = self.load_gpr(RDI)?;
                self.store_mem(&rdi, &v, bits);
                self.advance(RDI, &rdi, step)?;
            }
            StringOp::Lods => {
                let (rsi, src) = self.string_source(segment)?;
                let v = self.load_mem(&src, bits);
                self.write_reg(acc, &v, bits)?;
                self.advance(RSI, &rsi, step)?;
            }
            StringOp::Cmps => {
                let (rsi, src) = self.string_source(segment)?;
                let rdi = self.load_gpr(RDI)?;
                let a = self.load_mem(&src, bits);
                let b = self.load_mem(&rdi, bits);
                self.alu_value(Alu::Cmp, &a, &b, bits);
                self.advance(RSI, &rsi, step)?;
                self.advance(RDI, &rdi, step)?;
            }
            StringOp::Scas => {
                let (a, _) = self.read_reg(acc)?;
                let rdi = self.load_gpr(RDI)?;
                let b = self.load_mem(&rdi, bits);
                self.alu_value(Alu::Cmp, &a, &b, bits);
                self.advance(RDI, &rdi, step)?;
            }
        }
        Ok(())
    }

    fn advance(&mut self, reg: usize, old: &str, step: &str) -> R<()> {
        let new = self.val(&format!("add i64 {old}, {step}"));
        self.store_gpr(reg, &new)
    }

    /// Common guards of the bulk paths: `(count, byte length, count != 0 && length fits)`.
    fn bulk_length(&mut self, bits: u32) -> R<(String, String, String)> {
        let shift = (bits / 8).trailing_zeros();
        let count = self.load_gpr(RCX)?;
        let total = if shift == 0 {
            count.clone()
        } else {
            self.val(&format!("shl i64 {count}, {shift}"))
        };
        let nonzero = self.val(&format!("icmp ne i64 {count}, 0"));
        let fits = self.val(&format!(
            "icmp ule i64 {count}, {}",
            imm(u64::MAX >> shift, 64)
        ));
        let ok = self.val(&format!("and i1 {nonzero}, {fits}"));
        Ok((count, total, ok))
    }

    /// Lowest address of a run of `total` bytes walked from `start` (DF set: downwards).
    fn run_low(&mut self, df: &str, start: &str, total: &str, bits: u32) -> String {
        let span = self.val(&format!("sub i64 {total}, {}", bits / 8));
        let down = self.val(&format!("sub i64 {start}, {span}"));
        self.val(&format!("select i1 {df}, i64 {down}, i64 {start}"))
    }

    /// Final `rsi`/`rdi` adjustment of a bulk run: `±total`.
    fn run_delta(&mut self, df: &str, total: &str) -> String {
        let neg = self.val(&format!("sub i64 0, {total}"));
        self.val(&format!("select i1 {df}, i64 {neg}, i64 {total}"))
    }

    /// `rep movs` bulk path: `memmove` when the element loop provably equals it, i.e. the
    /// destination does not start strictly inside the not-yet-copied part of the source:
    /// forward (DF = 0) `(dst - src) mod 2^64 >= n`, backward `(src - dst) mod 2^64 >= n`.
    fn rep_movs_bulk(
        &mut self,
        bits: u32,
        df: &str,
        segment: Option<Register>,
        slow: &str,
        done: &str,
    ) -> R<()> {
        let (_, total, ok) = self.bulk_length(bits)?;
        let (rsi, src) = self.string_source(segment)?;
        let rdi = self.load_gpr(RDI)?;
        let forward_gap = self.val(&format!("sub i64 {rdi}, {src}"));
        let backward_gap = self.val(&format!("sub i64 {src}, {rdi}"));
        let gap = self.val(&format!(
            "select i1 {df}, i64 {backward_gap}, i64 {forward_gap}"
        ));
        let apart = self.val(&format!("icmp uge i64 {gap}, {total}"));
        let ok = self.val(&format!("and i1 {ok}, {apart}"));
        let fast = self.new_label("rep_bulk");
        self.emit(&format!("br i1 {ok}, label %{fast}, label %{slow}"));

        self.start_block(&fast);
        let src_low = self.run_low(df, &src, &total, bits);
        let dst_low = self.run_low(df, &rdi, &total, bits);
        if !self.guest_memmove(&dst_low, &src_low, &total)? {
            let src_ptr = self.guest_ptr(&src_low);
            let dst_ptr = self.guest_ptr(&dst_low);
            self.support.insert(Support::Memmove);
            self.emit(&format!(
                "call void @llvm.memmove.p0.p0.i64(ptr align 1 {dst_ptr}, ptr align 1 {src_ptr}, i64 {total}, i1 false)"
            ));
        }
        let delta = self.run_delta(df, &total);
        self.advance(RSI, &rsi, &delta)?;
        self.advance(RDI, &rdi, &delta)?;
        self.store_gpr(RCX, "0")?;
        self.emit(&format!("br label %{done}"));
        Ok(())
    }

    /// `rep stos` bulk path: `memset` when the stored value is one byte repeated (always for
    /// `stosb`; e.g. zeroing with `rep stosq`). Stores of one value do not depend on order.
    fn rep_stos_bulk(&mut self, bits: u32, df: &str, slow: &str, done: &str) -> R<()> {
        let (_, total, ok) = self.bulk_length(bits)?;
        let (v, _) = self.read_reg(accumulator(bits))?;
        let byte = if bits == 8 {
            v.clone()
        } else {
            self.val(&format!("trunc i{bits} {v} to i8"))
        };
        let ok = if bits == 8 {
            ok
        } else {
            let splat = imm(mask_bytes(bits), bits);
            let wide = self.val(&format!("zext i8 {byte} to i{bits}"));
            let repeated = self.val(&format!("mul i{bits} {wide}, {splat}"));
            let same = self.val(&format!("icmp eq i{bits} {repeated}, {v}"));
            self.val(&format!("and i1 {ok}, {same}"))
        };
        let fast = self.new_label("rep_bulk");
        self.emit(&format!("br i1 {ok}, label %{fast}, label %{slow}"));

        self.start_block(&fast);
        let rdi = self.load_gpr(RDI)?;
        let dst_low = self.run_low(df, &rdi, &total, bits);
        if !self.guest_memset(&dst_low, &byte, &total)? {
            let dst_ptr = self.guest_ptr(&dst_low);
            self.support.insert(Support::Memset);
            self.emit(&format!(
                "call void @llvm.memset.p0.i64(ptr align 1 {dst_ptr}, i8 {byte}, i64 {total}, i1 false)"
            ));
        }
        let delta = self.run_delta(df, &total);
        self.advance(RDI, &rdi, &delta)?;
        self.store_gpr(RCX, "0")?;
        self.emit(&format!("br label %{done}"));
        Ok(())
    }
}

/// `0x0101...01` with `bits / 8` bytes.
fn mask_bytes(bits: u32) -> u64 {
    (0..bits / 8).fold(0u64, |acc, i| acc | (1u64 << (8 * i)))
}
