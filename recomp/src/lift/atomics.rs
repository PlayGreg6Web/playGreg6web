//! Exchange / compare-exchange instructions and every `lock`-prefixed form.
//!
//! Lifted code runs on real threads (wasm threads on a shared `WebAssembly.Memory`), so:
//! - `lock add/sub/and/or/xor/adc/sbb/inc/dec/not/xadd/bts/btr/btc [mem]` and `xchg [mem], reg`
//!   (implicitly locked) become one LLVM `atomicrmw` (`adc`/`sbb` add/subtract `src + CF`);
//!   `lock neg` is a `cmpxchg` loop; `lock cmpxchg`/`lock cmpxchg8b` are one `cmpxchg`. All use
//!   `seq_cst` ordering (x86 locked instructions are full barriers) and natural alignment; the
//!   flags are computed exactly from the old value the atomic operation returned.
//! - wasm atomics trap on a misaligned address, so a locked access that is not naturally
//!   aligned traps instead of being split (x86 "split locks" are legal but never used by
//!   compiled code). `lock bts/btr/btc` update the single byte that holds the bit, which is
//!   always aligned.
//! - `lock cmpxchg16b` has no 16-byte wasm atomic: it runs under a spinlock taken from a lock
//!   table shared by every lifted module (`__playgta6_cmpxchg16b_locks`, indexed by the address
//!   / 16). It is therefore atomic with respect to every other `cmpxchg16b` on the same 16
//!   bytes, and its two 8-byte halves are read and written with 8-byte atomic accesses (no
//!   torn halves), but it is *not* atomic with respect to plain or 8-byte locked stores another
//!   thread makes to the same 16 bytes while the lock is held. A misaligned `cmpxchg16b`
//!   operand raises #GP (`__playgta6_trap`).
//! - Without the `lock` prefix, `xadd`, `cmpxchg`, `cmpxchg8b` and `cmpxchg16b` are ordinary
//!   (non-atomic) read-modify-writes, like on hardware.

use iced_x86::{Code, Instruction, Mnemonic, OpKind, Register};

use super::integer::{RAX, RBX, RCX, RDX};
use super::{Alu, Flag, FunctionLifter, R, RegRef, imm, reg_ref, unsupported};
use crate::lift::{HELPER_TRAP, Support};

/// `{lo, hi, success} = cmpxchg16b(addr, expected lo/hi, new lo/hi)` under a spinlock from the
/// shared lock table (see the module docs for exactly what this guarantees). `linkonce_odr`
/// gives every linked module the same table.
pub(in crate::lift) const CMPXCHG16B_IR: &str = "\
@__playgta6_cmpxchg16b_locks = linkonce_odr hidden global [64 x i32] zeroinitializer, align 4

define internal { i64, i64, i1 } @__playgta6_cmpxchg16b(i64 %addr, i64 %cmp_lo, i64 %cmp_hi, i64 %new_lo, i64 %new_hi) {
entry:
  %slot = lshr i64 %addr, 4
  %idx = and i64 %slot, 63
  %lock = getelementptr inbounds [64 x i32], ptr @__playgta6_cmpxchg16b_locks, i64 0, i64 %idx
  br label %acquire
acquire:
  %try = cmpxchg ptr %lock, i32 0, i32 1 seq_cst seq_cst, align 4
  %held = extractvalue { i32, i1 } %try, 1
  br i1 %held, label %locked, label %acquire
locked:
  %lo.p = inttoptr i64 %addr to ptr
  %hi.addr = add i64 %addr, 8
  %hi.p = inttoptr i64 %hi.addr to ptr
  %lo = load atomic i64, ptr %lo.p seq_cst, align 8
  %hi = load atomic i64, ptr %hi.p seq_cst, align 8
  %lo.eq = icmp eq i64 %lo, %cmp_lo
  %hi.eq = icmp eq i64 %hi, %cmp_hi
  %eq = and i1 %lo.eq, %hi.eq
  br i1 %eq, label %store, label %release
store:
  store atomic i64 %new_lo, ptr %lo.p seq_cst, align 8
  store atomic i64 %new_hi, ptr %hi.p seq_cst, align 8
  br label %release
release:
  store atomic i32 0, ptr %lock seq_cst, align 4
  %r0 = insertvalue { i64, i64, i1 } poison, i64 %lo, 0
  %r1 = insertvalue { i64, i64, i1 } %r0, i64 %hi, 1
  %r2 = insertvalue { i64, i64, i1 } %r1, i1 %eq, 2
  ret { i64, i64, i1 } %r2
}
";

fn accumulator(bits: u32) -> Register {
    match bits {
        8 => Register::AL,
        16 => Register::AX,
        32 => Register::EAX,
        _ => Register::RAX,
    }
}

impl FunctionLifter<'_> {
    /// `atomicrmw <op> ptr, iN value seq_cst` (natural alignment); returns the old value.
    pub(super) fn atomic_rmw(&mut self, op: &str, ptr: &str, value: &str, bits: u32) -> String {
        self.support.insert(Support::Atomics);
        self.val(&format!(
            "atomicrmw {op} ptr {ptr}, i{bits} {value} seq_cst, align {}",
            bits / 8
        ))
    }

    /// `cmpxchg ptr, expected, new seq_cst seq_cst`; returns `(old, success)`.
    fn atomic_cmpxchg(
        &mut self,
        ptr: &str,
        expected: &str,
        new: &str,
        bits: u32,
    ) -> (String, String) {
        self.support.insert(Support::Atomics);
        let pair = self.val(&format!(
            "cmpxchg ptr {ptr}, i{bits} {expected}, i{bits} {new} seq_cst seq_cst, align {}",
            bits / 8
        ));
        let old = self.val(&format!("extractvalue {{ i{bits}, i1 }} {pair}, 0"));
        let ok = self.val(&format!("extractvalue {{ i{bits}, i1 }} {pair}, 1"));
        (old, ok)
    }

    /// A new `%tN` name without emitting anything.
    fn fresh_name(&mut self) -> String {
        let name = format!("%t{}", self.tmp);
        self.tmp += 1;
        name
    }

    fn memory_ptr(&mut self, ins: &Instruction) -> R<String> {
        let ea = self.mem_ea(ins)?;
        Ok(self.guest_ptr(&ea))
    }

    /// Every `lock`-prefixed instruction.
    pub(super) fn lift_locked(&mut self, ins: &Instruction) -> R<()> {
        use Mnemonic as M;
        if ins.op_count() == 0 || ins.op_kind(0) != OpKind::Memory {
            return unsupported("lock prefix without a memory destination (#UD)");
        }
        match ins.mnemonic() {
            M::Add => self.atomic_alu(ins, Alu::Add),
            M::Adc => self.atomic_alu(ins, Alu::Adc),
            M::Sub => self.atomic_alu(ins, Alu::Sub),
            M::Sbb => self.atomic_alu(ins, Alu::Sbb),
            M::And => self.atomic_alu(ins, Alu::And),
            M::Or => self.atomic_alu(ins, Alu::Or),
            M::Xor => self.atomic_alu(ins, Alu::Xor),
            M::Inc | M::Dec | M::Not | M::Neg => self.atomic_unary(ins),
            M::Xadd => self.xadd(ins, true),
            M::Cmpxchg => self.cmpxchg(ins, true),
            M::Cmpxchg8b | M::Cmpxchg16b => self.cmpxchg_double(ins, true),
            M::Bts | M::Btr | M::Btc => self.bit_test(ins, true),
            M::Xchg => self.xchg_memory(ins),
            m => unsupported(format!("lock prefix on {m:?} (#UD)")),
        }
    }

    /// `lock <alu> [mem], src`: one `atomicrmw`, flags from the old value.
    fn atomic_alu(&mut self, ins: &Instruction, op: Alu) -> R<()> {
        let bits = self.op_bits(ins, 0)?;
        let t = format!("i{bits}");
        let b = self.read_op(ins, 1, bits)?;
        let (rmw, operand) = match op {
            Alu::Add => ("add", b.clone()),
            Alu::Sub => ("sub", b.clone()),
            Alu::And => ("and", b.clone()),
            Alu::Or => ("or", b.clone()),
            Alu::Xor => ("xor", b.clone()),
            Alu::Adc | Alu::Sbb => {
                // a + b + c == a + (b + c) modulo 2^N (likewise for subtraction).
                let c = self.get_flag(Flag::Cf);
                let cz = self.val(&format!("zext i1 {c} to {t}"));
                let sum = self.val(&format!("add {t} {b}, {cz}"));
                (if matches!(op, Alu::Adc) { "add" } else { "sub" }, sum)
            }
            Alu::Cmp | Alu::Test => return unsupported("lock prefix on cmp/test (#UD)"),
        };
        let ptr = self.memory_ptr(ins)?;
        let old = self.atomic_rmw(rmw, &ptr, &operand, bits);
        self.alu_value(op, &old, &b, bits);
        Ok(())
    }

    /// `lock inc/dec/not/neg [mem]`.
    fn atomic_unary(&mut self, ins: &Instruction) -> R<()> {
        let bits = self.op_bits(ins, 0)?;
        let ptr = self.memory_ptr(ins)?;
        let m = ins.mnemonic();
        let old = match m {
            Mnemonic::Inc => self.atomic_rmw("add", &ptr, "1", bits),
            Mnemonic::Dec => self.atomic_rmw("sub", &ptr, "1", bits),
            Mnemonic::Not => self.atomic_rmw("xor", &ptr, "-1", bits),
            _ => {
                // neg: compare-exchange loop.
                let t = format!("i{bits}");
                let pre = self.new_label("neg_pre");
                let retry = self.new_label("neg_retry");
                let done = self.new_label("neg_done");
                self.emit(&format!("br label %{pre}"));
                self.start_block(&pre);
                self.support.insert(Support::Atomics);
                let first = self.val(&format!(
                    "load atomic {t}, ptr {ptr} seq_cst, align {}",
                    bits / 8
                ));
                self.emit(&format!("br label %{retry}"));
                self.start_block(&retry);
                // Names are reserved up front because the phi refers to `seen` before it is
                // defined.
                let cur = self.fresh_name();
                let seen = self.fresh_name();
                self.emit(&format!(
                    "{cur} = phi {t} [ {first}, %{pre} ], [ {seen}, %{retry} ]"
                ));
                let neg = self.val(&format!("sub {t} 0, {cur}"));
                let pair = self.val(&format!(
                    "cmpxchg ptr {ptr}, {t} {cur}, {t} {neg} seq_cst seq_cst, align {}",
                    bits / 8
                ));
                self.emit(&format!("{seen} = extractvalue {{ {t}, i1 }} {pair}, 0"));
                let ok = self.val(&format!("extractvalue {{ {t}, i1 }} {pair}, 1"));
                self.emit(&format!("br i1 {ok}, label %{done}, label %{retry}"));
                self.start_block(&done);
                seen
            }
        };
        self.unary_value(m, &old, bits);
        Ok(())
    }

    /// `xchg [mem], reg` / `xchg reg, [mem]`: always atomic (implicit lock).
    pub(super) fn xchg_memory(&mut self, ins: &Instruction) -> R<()> {
        let (mem, reg) = if ins.op_kind(0) == OpKind::Memory {
            (0, 1)
        } else {
            (1, 0)
        };
        if ins.op_kind(reg) != OpKind::Register {
            return unsupported("xchg without a register operand");
        }
        let bits = self.op_bits(ins, mem)?;
        let v = self.read_op(ins, reg, bits)?;
        let ptr = self.memory_ptr(ins)?;
        let old = self.atomic_rmw("xchg", &ptr, &v, bits);
        self.write_op(ins, reg, &old, bits)
    }

    /// `xadd dst, src`: `tmp = dst + src; src = dst; dst = tmp` (flags as `add`).
    pub(super) fn xadd(&mut self, ins: &Instruction, locked: bool) -> R<()> {
        let bits = self.op_bits(ins, 0)?;
        let b = self.read_op(ins, 1, bits)?;
        if locked {
            let ptr = self.memory_ptr(ins)?;
            let old = self.atomic_rmw("add", &ptr, &b, bits);
            self.alu_value(Alu::Add, &old, &b, bits);
            return self.write_op(ins, 1, &old, bits);
        }
        let a = self.read_op(ins, 0, bits)?;
        let r = self.alu_value(Alu::Add, &a, &b, bits);
        // The destination is written last (`xadd eax, eax` leaves the sum).
        self.write_op(ins, 1, &a, bits)?;
        self.write_op(ins, 0, &r, bits)
    }

    /// `cmpxchg dst, src`: flags of `cmp acc, dst`; equal: `dst = src`; else `acc = dst`. A
    /// register destination is not written on failure and the accumulator is not written on
    /// success (so neither is zero-extended by a 32-bit form then), as measured on hardware.
    pub(super) fn cmpxchg(&mut self, ins: &Instruction, locked: bool) -> R<()> {
        let bits = self.op_bits(ins, 0)?;
        let acc = accumulator(bits);
        let src = self.read_op(ins, 1, bits)?;
        let (expected, _) = self.read_reg(acc)?;
        let (old, eq) = if locked {
            let ptr = self.memory_ptr(ins)?;
            self.atomic_cmpxchg(&ptr, &expected, &src, bits)
        } else {
            let old = self.read_op(ins, 0, bits)?;
            let eq = self.val(&format!("icmp eq i{bits} {expected}, {old}"));
            match (ins.op_kind(0), reg_ref(ins.op_register(0))) {
                // A 32-bit register destination is only written (zero-extended) on success.
                (OpKind::Register, Ok(RegRef::Low(idx, 32))) => {
                    let full = self.load_gpr(idx)?;
                    let wide = self.val(&format!("zext i32 {src} to i64"));
                    let v = self.val(&format!("select i1 {eq}, i64 {wide}, i64 {full}"));
                    self.store_gpr(idx, &v)?;
                }
                _ => {
                    let new = self.val(&format!("select i1 {eq}, i{bits} {src}, i{bits} {old}"));
                    self.write_op(ins, 0, &new, bits)?;
                }
            }
            (old, eq)
        };
        self.alu_value(Alu::Cmp, &expected, &old, bits);
        self.write_accumulator_on_failure(acc, &eq, &old, bits)
    }

    /// `acc = old` when `eq` is false; leaves the register untouched otherwise.
    fn write_accumulator_on_failure(
        &mut self,
        acc: Register,
        eq: &str,
        old: &str,
        bits: u32,
    ) -> R<()> {
        let idx = match acc {
            Register::EAX | Register::RAX => RAX,
            Register::EDX | Register::RDX => RDX,
            _ => {
                // 8/16-bit: re-read the current value (the destination may be the same register).
                let (cur, _) = self.read_reg(acc)?;
                let v = self.val(&format!("select i1 {eq}, i{bits} {cur}, i{bits} {old}"));
                return self.write_reg(acc, &v, bits);
            }
        };
        let cur = self.load_gpr(idx)?;
        let wide = if bits == 64 {
            old.to_string()
        } else {
            self.val(&format!("zext i32 {old} to i64"))
        };
        let v = self.val(&format!("select i1 {eq}, i64 {cur}, i64 {wide}"));
        self.store_gpr(idx, &v)
    }

    /// `cmpxchg8b m64` (`edx:eax` vs memory, store `ecx:ebx`) and `cmpxchg16b m128` (`rdx:rax`,
    /// `rcx:rbx`). Only ZF changes. On failure the memory value is loaded into `edx:eax` /
    /// `rdx:rax` (32-bit writes zero-extend); on success those registers are untouched.
    pub(super) fn cmpxchg_double(&mut self, ins: &Instruction, locked: bool) -> R<()> {
        let wide = match ins.code() {
            Code::Cmpxchg8b_m64 => false,
            Code::Cmpxchg16b_m128 => true,
            _ => return unsupported("cmpxchg8b/16b operand"),
        };
        let ea = self.mem_ea(ins)?;
        let (eq, lo, hi) = if wide {
            // #GP for a misaligned operand.
            let low = self.val(&format!("and i64 {ea}, 15"));
            let misaligned = self.val(&format!("icmp ne i64 {low}, 0"));
            let fault = self.new_label("gp");
            let ok = self.new_label("cont");
            self.emit(&format!("br i1 {misaligned}, label %{fault}, label %{ok}"));
            self.start_block(&fault);
            self.sync_out();
            self.emit(&format!(
                "call void @{HELPER_TRAP}(ptr %state, i64 {})",
                imm(ins.ip(), 64)
            ));
            self.emit("unreachable");
            self.start_block(&ok);
            let rax = self.load_gpr(RAX)?;
            let rdx = self.load_gpr(RDX)?;
            let rbx = self.load_gpr(RBX)?;
            let rcx = self.load_gpr(RCX)?;
            if locked {
                self.support.insert(Support::Atomics);
                self.support.insert(Support::Cmpxchg16b);
                let res = self.val(&format!(
                    "call {{ i64, i64, i1 }} @__playgta6_cmpxchg16b(i64 {ea}, i64 {rax}, i64 {rdx}, i64 {rbx}, i64 {rcx})"
                ));
                let lo = self.val(&format!("extractvalue {{ i64, i64, i1 }} {res}, 0"));
                let hi = self.val(&format!("extractvalue {{ i64, i64, i1 }} {res}, 1"));
                let eq = self.val(&format!("extractvalue {{ i64, i64, i1 }} {res}, 2"));
                (eq, lo, hi)
            } else {
                let hi_addr = self.val(&format!("add i64 {ea}, 8"));
                let lo = self.load_mem(&ea, 64);
                let hi = self.load_mem(&hi_addr, 64);
                let lo_eq = self.val(&format!("icmp eq i64 {lo}, {rax}"));
                let hi_eq = self.val(&format!("icmp eq i64 {hi}, {rdx}"));
                let eq = self.val(&format!("and i1 {lo_eq}, {hi_eq}"));
                let new_lo = self.val(&format!("select i1 {eq}, i64 {rbx}, i64 {lo}"));
                let new_hi = self.val(&format!("select i1 {eq}, i64 {rcx}, i64 {hi}"));
                self.store_mem(&ea, &new_lo, 64);
                self.store_mem(&hi_addr, &new_hi, 64);
                (eq, lo, hi)
            }
        } else {
            let (eax, _) = self.read_reg(Register::EAX)?;
            let (edx, _) = self.read_reg(Register::EDX)?;
            let (ebx, _) = self.read_reg(Register::EBX)?;
            let (ecx, _) = self.read_reg(Register::ECX)?;
            let expected = self.join32(&edx, &eax);
            let new = self.join32(&ecx, &ebx);
            let (old, eq) = if locked {
                let ptr = self.guest_ptr(&ea);
                self.atomic_cmpxchg(&ptr, &expected, &new, 64)
            } else {
                let old = self.load_mem(&ea, 64);
                let eq = self.val(&format!("icmp eq i64 {old}, {expected}"));
                let stored = self.val(&format!("select i1 {eq}, i64 {new}, i64 {old}"));
                self.store_mem(&ea, &stored, 64);
                (old, eq)
            };
            let lo = self.val(&format!("trunc i64 {old} to i32"));
            let shifted = self.val(&format!("lshr i64 {old}, 32"));
            let hi = self.val(&format!("trunc i64 {shifted} to i32"));
            (eq, lo, hi)
        };
        self.set_flag(Flag::Zf, &eq);
        let bits = if wide { 64 } else { 32 };
        let (lo_reg, hi_reg) = if wide {
            (Register::RAX, Register::RDX)
        } else {
            (Register::EAX, Register::EDX)
        };
        self.write_accumulator_on_failure(lo_reg, &eq, &lo, bits)?;
        self.write_accumulator_on_failure(hi_reg, &eq, &hi, bits)
    }

    /// `(hi << 32) | lo` from two i32 values.
    fn join32(&mut self, hi: &str, lo: &str) -> String {
        let h = self.val(&format!("zext i32 {hi} to i64"));
        let l = self.val(&format!("zext i32 {lo} to i64"));
        let h = self.val(&format!("shl i64 {h}, 32"));
        self.val(&format!("or i64 {h}, {l}"))
    }
}
