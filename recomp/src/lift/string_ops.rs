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
use super::memory_access::{LIKELY, UNLIKELY};
use super::{Alu, FunctionLifter, R, imm, parse_const, unsupported};
use crate::lift::memory::{
    Decl, HELPER_GUEST_MEMSET, MemoryModel, SLOT_BYTES, SLOT_SHIFT, WINDOW_DIRTY_SYMBOL,
    WINDOW_TABLE_SYMBOL, Window,
};
use crate::lift::{GuestBase, Support};

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
        // Direction flag, computed once so bulk paths and scalar loops share it.
        let df = self.load_df();
        let k = bits / 8;

        if repeat == Repeat::Once {
            let step = self.step_value(&df, k);
            self.clear_window_cache();
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
        let step = self.step_value(&df, k);
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

    fn step_value(&mut self, df: &str, k: u32) -> String {
        if df == "false" {
            k.to_string()
        } else if df == "true" {
            (-i64::from(k)).to_string()
        } else {
            self.val(&format!("select i1 {df}, i64 {}, i64 {k}", -i64::from(k)))
        }
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
    /// Constant-folds `count` when statically known (e.g. `mov ecx, 16; rep movsq`).
    fn bulk_length(&mut self, bits: u32) -> R<(String, String, String)> {
        let shift = (bits / 8).trailing_zeros();
        let count = self.load_gpr(RCX)?;
        if let Some(c) = parse_const(&count, 64) {
            if c == 0 {
                return Ok((count, "0".to_string(), "false".to_string()));
            }
            if c <= u64::MAX >> shift {
                return Ok((count, imm(c << shift, 64), "true".to_string()));
            }
            return Ok((count, "0".to_string(), "false".to_string()));
        }
        let total = if shift == 0 {
            count.clone()
        } else {
            self.val(&format!("shl i64 {count}, {shift}"))
        };
        let nonzero = self.val(&format!("icmp ne i64 {count}, 0"));
        let ok = if shift == 0 {
            nonzero
        } else {
            let fits = self.val(&format!(
                "icmp ule i64 {count}, {}",
                imm(u64::MAX >> shift, 64)
            ));
            self.val(&format!("and i1 {nonzero}, {fits}"))
        };
        Ok((count, total, ok))
    }

    /// Lowest address of a run of `total` bytes walked from `start` (DF set: downwards).
    fn run_low(&mut self, df: &str, start: &str, total: &str, bits: u32) -> String {
        if df == "false" {
            return start.to_string();
        }
        let span = self.val(&format!("sub i64 {total}, {}", bits / 8));
        let down = self.val(&format!("sub i64 {start}, {span}"));
        if df == "true" {
            down
        } else {
            self.val(&format!("select i1 {df}, i64 {down}, i64 {start}"))
        }
    }

    /// Final `rsi`/`rdi` adjustment of a bulk run: `±total`.
    fn run_delta(&mut self, df: &str, total: &str) -> String {
        if df == "false" {
            return total.to_string();
        }
        let neg = self.val(&format!("sub i64 0, {total}"));
        if df == "true" {
            neg
        } else {
            self.val(&format!("select i1 {df}, i64 {neg}, i64 {total}"))
        }
    }

    /// Flat host address `addr + base` reusing the function's hoisted `@__playgta6_guest_base`.
    fn bulk_flat_host(&mut self, addr: &str, base: GuestBase) -> String {
        match base {
            GuestBase::Constant(0) => addr.to_string(),
            GuestBase::Constant(c) => self.val_raw(&format!("add i64 {addr}, {}", imm(c, 64))),
            GuestBase::Symbol => {
                self.support.insert(Support::GuestBase);
                let base_val = if let Some(name) = &self.guest_base {
                    name.clone()
                } else {
                    let name = self.fresh_name();
                    self.guest_base = Some(name.clone());
                    name
                };
                if addr == "0" {
                    base_val
                } else {
                    self.val(&format!("add i64 {addr}, {base_val}"))
                }
            }
        }
    }

    /// Loads the window table delta for an in-window offset `rel` using the short-mask byte offset
    /// `(rel >>a 13) & -8` (`getelementptr i8`), matching `memory_access::window_offset`.
    fn bulk_window_delta(&mut self, rel: &str) -> String {
        self.support.insert(Support::Memory(Decl::WindowTable));
        let shifted = self.val(&format!("ashr i64 {rel}, {}", SLOT_SHIFT - 3));
        let offset = self.val(&format!("and i64 {shifted}, -8"));
        let entry = self.val(&format!(
            "getelementptr i8, ptr @{WINDOW_TABLE_SYMBOL}, i64 {offset}"
        ));
        self.val(&format!("load i64, ptr {entry}, align 8"))
    }

    /// Marks window slot `slot` dirty with check-before-atomic-store (`monotonic` load + `release`
    /// store when clean) and branches to `done`.
    fn bulk_window_mark_then_br(&mut self, slot: &str, done: &str) {
        self.support.insert(Support::Memory(Decl::WindowDirty));
        self.support.insert(Support::Atomics);
        let m = self.val(&format!(
            "getelementptr i8, ptr @{WINDOW_DIRTY_SYMBOL}, i64 {slot}"
        ));
        let cur = self.val(&format!("load atomic i8, ptr {m} monotonic, align 1"));
        let clean = self.val(&format!("icmp eq i8 {cur}, 0"));
        let set = self.new_label("gms");
        self.emit(&format!(
            "br i1 {clean}, label %{set}, label %{done}, {UNLIKELY}"
        ));
        self.start_block(&set);
        self.emit(&format!("store atomic i8 1, ptr {m} release, align 1"));
        self.emit(&format!("br label %{done}"));
    }

    /// Emits a tight counted loop storing `v` (`i{bits}`) to `count` contiguous elements at
    /// `dst_ptr` (`count >= 1`, no per-iteration guest translation or alloca traffic; LLVM
    /// auto-vectorizes this directly to AVX2 / Wasm128 stores).
    fn emit_contiguous_fill_loop(&mut self, dst_ptr: &str, v: &str, count: &str, bits: u32) {
        let pre = self.current_block.clone();
        let loop_lbl = self.new_label("rep_fill");
        let after = self.new_label("rep_fill_done");
        self.emit(&format!("br label %{loop_lbl}"));

        self.start_block(&loop_lbl);
        let idx = self.fresh_name();
        let idx_next = self.fresh_name();
        self.emit(&format!(
            "{idx} = phi i64 [ 0, %{pre} ], [ {idx_next}, %{loop_lbl} ]"
        ));
        let elem_ptr = self.val(&format!(
            "getelementptr i{bits}, ptr {dst_ptr}, i64 {idx}"
        ));
        self.emit(&format!("store i{bits} {v}, ptr {elem_ptr}, align 1"));
        self.emit(&format!("{idx_next} = add nuw i64 {idx}, 1"));
        let more = self.val(&format!("icmp ult i64 {idx_next}, {count}"));
        self.emit(&format!(
            "br i1 {more}, label %{loop_lbl}, label %{after}, {LIKELY}"
        ));
        self.start_block(&after);
    }

    /// Tests whether `[dst_low, dst_low + total)` (`total > 0`) is wrap-free and completely outside
    /// `[w.base, w.base + w.size)`. Returns `(flat_ok, no_wrap)`.
    fn bulk_window_flat_test(&mut self, dst_low: &str, total: &str, w: Window) -> (String, String) {
        self.support.insert(Support::Memory(Decl::WindowTable));
        let end_addr = self.val(&format!("add i64 {dst_low}, {total}"));
        let no_wrap = self.val(&format!("icmp ugt i64 {end_addr}, {dst_low}"));
        let win_end = imm(w.base.wrapping_add(w.size), 64);
        let below_end = self.val(&format!("icmp ult i64 {dst_low}, {win_end}"));
        let after_start = self.val(&format!("icmp ugt i64 {end_addr}, {}", imm(w.base, 64)));
        let touches = self.val(&format!("and i1 {below_end}, {after_start}"));
        let outside = self.val(&format!("xor i1 {touches}, true"));
        let flat_ok = self.val(&format!("and i1 {no_wrap}, {outside}"));
        (flat_ok, no_wrap)
    }

    /// Tests whether `[dst_low, dst_low + total)` (`total > 0`) fits inside a single 64 KiB slot of
    /// window `w`. Returns `(slot_ok, rel)`.
    fn bulk_window_single_slot_test(
        &mut self,
        dst_low: &str,
        total: &str,
        no_wrap: &str,
        w: Window,
    ) -> (String, String) {
        let rel = self.val(&format!("sub i64 {dst_low}, {}", imm(w.base, 64)));
        let in_win = self.val(&format!("icmp ult i64 {rel}, {}", imm(w.size, 64)));
        let off = self.val(&format!("and i64 {dst_low}, {}", SLOT_BYTES - 1));
        let rem = self.val(&format!("sub i64 {SLOT_BYTES}, {off}"));
        let fits_slot = self.val(&format!("icmp ule i64 {total}, {rem}"));
        let ok0 = self.val(&format!("and i1 {in_win}, {fits_slot}"));
        let slot_ok = self.val(&format!("and i1 {ok0}, {no_wrap}"));
        (slot_ok, rel)
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
        if ok == "false" {
            self.emit(&format!("br label %{slow}"));
            return Ok(());
        }
        let (rsi, src) = self.string_source(segment)?;
        let rdi = self.load_gpr(RDI)?;
        let gap = if df == "false" {
            self.val(&format!("sub i64 {rdi}, {src}"))
        } else if df == "true" {
            self.val(&format!("sub i64 {src}, {rdi}"))
        } else {
            let forward_gap = self.val(&format!("sub i64 {rdi}, {src}"));
            let backward_gap = self.val(&format!("sub i64 {src}, {rdi}"));
            self.val(&format!(
                "select i1 {df}, i64 {backward_gap}, i64 {forward_gap}"
            ))
        };
        let apart = self.val(&format!("icmp uge i64 {gap}, {total}"));
        let ok = if ok == "true" {
            apart
        } else {
            self.val(&format!("and i1 {ok}, {apart}"))
        };
        let fast = self.new_label("rep_bulk");
        self.emit(&format!(
            "br i1 {ok}, label %{fast}, label %{slow}, {LIKELY}"
        ));

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

    /// `rep stos` bulk paths:
    /// 1. Byte-splatted fills (`stosb`, zero-fill `xor eax, eax; rep stosq`, or any repeated byte)
    ///    emit native `@llvm.memset.p0.i64` for flat memory AND single-slot in-window ranges
    ///    (with atomic dirty-bit marking), falling back to `@__playgta6_guest_memset` for
    ///    multi-slot window straddles and page-table models.
    /// 2. Non-splat multi-byte fills (`stosw`/`stosd`/`stosq`) emit a tight contiguous SSA
    ///    pointer-incrementing fill loop for flat memory and single-slot in-window ranges (with
    ///    atomic dirty-bit marking), falling back to the exact scalar loop on multi-slot straddles
    ///    or page-table models.
    fn rep_stos_bulk(&mut self, bits: u32, df: &str, slow: &str, done: &str) -> R<()> {
        let (count, total, ok) = self.bulk_length(bits)?;
        if ok == "false" {
            self.emit(&format!("br label %{slow}"));
            return Ok(());
        }
        let (v, _) = self.read_reg(accumulator(bits))?;
        let rdi = self.load_gpr(RDI)?;
        let static_splat = parse_const(&v, bits).map(|c| const_byte_splat(c, bits));

        let emit_memset_fast = |l: &mut Self, byte: &str| -> R<()> {
            let dst_low = l.run_low(df, &rdi, &total, bits);
            match l.memory {
                MemoryModel::Windowed(w) => {
                    let (flat_ok, no_wrap) = l.bulk_window_flat_test(&dst_low, &total, w);
                    let flat_blk = l.new_label("rep_mflat");
                    let non_flat = l.new_label("rep_mwin_chk");
                    let win_blk = l.new_label("rep_mwin");
                    let helper_blk = l.new_label("rep_mhlp");
                    let finish = l.new_label("rep_mfin");
                    l.emit(&format!(
                        "br i1 {flat_ok}, label %{flat_blk}, label %{non_flat}, {LIKELY}"
                    ));

                    l.start_block(&flat_blk);
                    let host = l.bulk_flat_host(&dst_low, w.flat);
                    let dst_ptr = l.val(&format!("inttoptr i64 {host} to ptr"));
                    l.support.insert(Support::Memset);
                    l.emit(&format!(
                        "call void @llvm.memset.p0.i64(ptr align 1 {dst_ptr}, i8 {byte}, i64 {total}, i1 false)"
                    ));
                    l.emit(&format!("br label %{finish}"));

                    l.start_block(&non_flat);
                    let (slot_ok, rel) =
                        l.bulk_window_single_slot_test(&dst_low, &total, &no_wrap, w);
                    l.emit(&format!(
                        "br i1 {slot_ok}, label %{win_blk}, label %{helper_blk}, {LIKELY}"
                    ));

                    l.start_block(&win_blk);
                    let slot = l.val(&format!("lshr i64 {rel}, {SLOT_SHIFT}"));
                    let delta_win = l.bulk_window_delta(&rel);
                    let host_win = l.val(&format!("add i64 {dst_low}, {delta_win}"));
                    let ptr_win = l.val(&format!("inttoptr i64 {host_win} to ptr"));
                    l.support.insert(Support::Memset);
                    l.emit(&format!(
                        "call void @llvm.memset.p0.i64(ptr align 1 {ptr_win}, i8 {byte}, i64 {total}, i1 false)"
                    ));
                    l.bulk_window_mark_then_br(&slot, &finish);

                    l.start_block(&helper_blk);
                    l.support.insert(Support::Memory(Decl::Memset));
                    l.emit(&format!(
                        "call void @{HELPER_GUEST_MEMSET}(i64 {dst_low}, i8 {byte}, i64 {total})"
                    ));
                    l.emit(&format!("br label %{finish}"));

                    l.start_block(&finish);
                }
                _ => {
                    if !l.guest_memset(&dst_low, byte, &total)? {
                        let dst_ptr = l.guest_ptr(&dst_low);
                        l.support.insert(Support::Memset);
                        l.emit(&format!(
                            "call void @llvm.memset.p0.i64(ptr align 1 {dst_ptr}, i8 {byte}, i64 {total}, i1 false)"
                        ));
                    }
                }
            }
            let delta = l.run_delta(df, &total);
            l.advance(RDI, &rdi, &delta)?;
            l.store_gpr(RCX, "0")?;
            l.emit(&format!("br label %{done}"));
            Ok(())
        };

        let emit_fill_fast = |l: &mut Self| -> R<()> {
            let dst_low = l.run_low(df, &rdi, &total, bits);
            match l.memory {
                MemoryModel::Base(_) => {
                    let dst_ptr = l.guest_ptr(&dst_low);
                    l.emit_contiguous_fill_loop(&dst_ptr, &v, &count, bits);
                    let delta = l.run_delta(df, &total);
                    l.advance(RDI, &rdi, &delta)?;
                    l.store_gpr(RCX, "0")?;
                    l.emit(&format!("br label %{done}"));
                }
                MemoryModel::Windowed(w) => {
                    let (flat_ok, no_wrap) = l.bulk_window_flat_test(&dst_low, &total, w);
                    let flat_blk = l.new_label("rep_fflat");
                    let non_flat = l.new_label("rep_fwin_chk");
                    let win_blk = l.new_label("rep_fwin");
                    let finish = l.new_label("rep_ffin");
                    l.emit(&format!(
                        "br i1 {flat_ok}, label %{flat_blk}, label %{non_flat}, {LIKELY}"
                    ));

                    l.start_block(&flat_blk);
                    let host = l.bulk_flat_host(&dst_low, w.flat);
                    let dst_ptr = l.val(&format!("inttoptr i64 {host} to ptr"));
                    l.emit_contiguous_fill_loop(&dst_ptr, &v, &count, bits);
                    l.emit(&format!("br label %{finish}"));

                    l.start_block(&non_flat);
                    let (slot_ok, rel) =
                        l.bulk_window_single_slot_test(&dst_low, &total, &no_wrap, w);
                    l.emit(&format!(
                        "br i1 {slot_ok}, label %{win_blk}, label %{slow}, {LIKELY}"
                    ));

                    l.start_block(&win_blk);
                    let slot = l.val(&format!("lshr i64 {rel}, {SLOT_SHIFT}"));
                    let delta_win = l.bulk_window_delta(&rel);
                    let host_win = l.val(&format!("add i64 {dst_low}, {delta_win}"));
                    let ptr_win = l.val(&format!("inttoptr i64 {host_win} to ptr"));
                    l.emit_contiguous_fill_loop(&ptr_win, &v, &count, bits);
                    l.bulk_window_mark_then_br(&slot, &finish);

                    l.start_block(&finish);
                    let delta = l.run_delta(df, &total);
                    l.advance(RDI, &rdi, &delta)?;
                    l.store_gpr(RCX, "0")?;
                    l.emit(&format!("br label %{done}"));
                }
                MemoryModel::PageTable { .. } => {
                    l.emit(&format!("br label %{slow}"));
                }
            }
            Ok(())
        };

        // 1. Always-byte (`stosb`) or compile-time constant byte-splat (e.g. `xor eax, eax; rep stosq`).
        if bits == 8 {
            let fast = self.new_label("rep_bulk");
            if ok == "true" {
                self.emit(&format!("br label %{fast}"));
            } else {
                self.emit(&format!(
                    "br i1 {ok}, label %{fast}, label %{slow}, {LIKELY}"
                ));
            }
            self.start_block(&fast);
            return emit_memset_fast(self, &v);
        }
        if let Some(Some(b)) = static_splat {
            let fast = self.new_label("rep_bulk");
            if ok == "true" {
                self.emit(&format!("br label %{fast}"));
            } else {
                self.emit(&format!(
                    "br i1 {ok}, label %{fast}, label %{slow}, {LIKELY}"
                ));
            }
            self.start_block(&fast);
            let byte = imm(u64::from(b), 8);
            return emit_memset_fast(self, &byte);
        }
        // 2. Compile-time constant non-splat multi-byte value (e.g. `mov eax, 0x3F800000; rep stosd`).
        if let Some(None) = static_splat {
            if matches!(self.memory, MemoryModel::PageTable { .. }) {
                self.emit(&format!("br label %{slow}"));
                return Ok(());
            }
            let fast = self.new_label("rep_bulk_fill");
            if ok == "true" {
                self.emit(&format!("br label %{fast}"));
            } else {
                self.emit(&format!(
                    "br i1 {ok}, label %{fast}, label %{slow}, {LIKELY}"
                ));
            }
            self.start_block(&fast);
            return emit_fill_fast(self);
        }
        // 3. Dynamic multi-byte value: dispatch byte-splats to `@llvm.memset` and non-splat fills
        //    to the tight contiguous fill loop (or scalar loop under PageTable).
        let byte = self.val(&format!("trunc i{bits} {v} to i8"));
        let splat = imm(mask_bytes(bits), bits);
        let wide = self.val(&format!("zext i8 {byte} to i{bits}"));
        let repeated = self.val(&format!("mul i{bits} {wide}, {splat}"));
        let same = self.val(&format!("icmp eq i{bits} {repeated}, {v}"));
        if matches!(self.memory, MemoryModel::PageTable { .. }) {
            let cond = if ok == "true" {
                same
            } else {
                self.val(&format!("and i1 {ok}, {same}"))
            };
            let fast = self.new_label("rep_bulk");
            self.emit(&format!(
                "br i1 {cond}, label %{fast}, label %{slow}, {LIKELY}"
            ));
            self.start_block(&fast);
            return emit_memset_fast(self, &byte);
        }
        let dispatch = self.new_label("rep_stos_kind");
        if ok == "true" {
            self.emit(&format!("br label %{dispatch}"));
        } else {
            self.emit(&format!(
                "br i1 {ok}, label %{dispatch}, label %{slow}, {LIKELY}"
            ));
        }
        self.start_block(&dispatch);
        let fast_memset = self.new_label("rep_bulk");
        let fast_fill = self.new_label("rep_bulk_fill");
        self.emit(&format!(
            "br i1 {same}, label %{fast_memset}, label %{fast_fill}"
        ));
        self.start_block(&fast_memset);
        emit_memset_fast(self, &byte)?;
        self.start_block(&fast_fill);
        emit_fill_fast(self)
    }
}

/// `0x0101...01` with `bits / 8` bytes.
fn mask_bytes(bits: u32) -> u64 {
    (0..bits / 8).fold(0u64, |acc, i| acc | (1u64 << (8 * i)))
}

/// Returns `Some(b)` if `val`'s low `bits` bits consist of `b` repeated in every byte.
fn const_byte_splat(val: u64, bits: u32) -> Option<u8> {
    let b = val as u8;
    let rep = u64::from(b).wrapping_mul(mask_bytes(bits));
    let mask = if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    ((val & mask) == (rep & mask)).then_some(b)
}
