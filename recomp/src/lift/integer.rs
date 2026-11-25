//! Integer instructions beyond the core move/ALU set: the dispatch hook for the string
//! ([`string_ops`](super::string_ops)), bit ([`bit_ops`](super::bit_ops)) and atomic
//! ([`atomics`](super::atomics)) families, plus flag manipulation, `pushfq`/`popfq`,
//! `lahf`/`sahf`, `enter`, `xlat`, `movbe`, `adcx`/`adox`, `crc32`, `loop*`/`jrcxz`/`jecxz`
//! conditions, fences, hint no-ops and the runtime-provided `cpuid`/`rdtsc`/`rdtscp`/`xgetbv`.

use iced_x86::{Code, Instruction, Mnemonic, OpKind, Register};

use super::{Flag, FunctionLifter, R, RBP, RSP, imm, unsupported};
use crate::lift::{GUEST_STATE_TYPE, RuntimeHelper, Support};

pub(super) const RAX: usize = 0;
pub(super) const RCX: usize = 1;
pub(super) const RDX: usize = 2;
pub(super) const RBX: usize = 3;
pub(super) const RSI: usize = 6;
pub(super) const RDI: usize = 7;

/// Index of `DF` in the guest state's flag bytes (`CF PF AF ZF SF OF DF pad`).
const DF_INDEX: u32 = 6;

/// RFLAGS bit positions of the arithmetic flags in guest-state order.
const RFLAGS_BITS: [(Flag, u32); 6] = [
    (Flag::Cf, 0),
    (Flag::Pf, 2),
    (Flag::Af, 4),
    (Flag::Zf, 6),
    (Flag::Sf, 7),
    (Flag::Of, 11),
];
const RFLAGS_DF_BIT: u32 = 10;
/// Bit 1 is always set; IF (bit 9) is always set in user mode.
const RFLAGS_FIXED: u64 = 0x202;

/// `{crc'} = crc32c(crc, byte)`: one byte of the reflected CRC-32C (Castagnoli, polynomial
/// `0x82F63B78`) that the SSE4.2 `crc32` instruction computes.
pub(in crate::lift) const CRC32C_IR: &str = "\
define internal i32 @__playgta6_crc32c_u8(i32 %crc, i8 %byte) unnamed_addr #0 {
entry:
  %b = zext i8 %byte to i32
  %c0 = xor i32 %crc, %b
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %c = phi i32 [ %c0, %entry ], [ %c.next, %loop ]
  %low = and i32 %c, 1
  %neg = sub i32 0, %low
  %poly = and i32 %neg, -2097792136
  %half = lshr i32 %c, 1
  %c.next = xor i32 %half, %poly
  %i.next = add i32 %i, 1
  %done = icmp eq i32 %i.next, 8
  br i1 %done, label %exit, label %loop
exit:
  ret i32 %c.next
}
";

impl FunctionLifter<'_> {
    /// Lifts the instructions this module family owns; `None` hands the instruction back to the
    /// core lifter. Every `lock`-prefixed instruction is handled here (atomically or as a trap).
    pub(super) fn lift_integer(&mut self, ins: &Instruction) -> Option<R<()>> {
        use Mnemonic as M;
        if ins.has_lock_prefix() {
            return Some(self.lift_locked(ins));
        }
        if ins.is_string_instruction() {
            return Some(self.string_instruction(ins));
        }
        let result = match ins.mnemonic() {
            // `xchg` with a memory operand is implicitly locked.
            M::Xchg if ins.op_kind(0) == OpKind::Memory || ins.op_kind(1) == OpKind::Memory => {
                self.xchg_memory(ins)
            }
            M::Xadd => self.xadd(ins, false),
            M::Cmpxchg => self.cmpxchg(ins, false),
            M::Cmpxchg8b | M::Cmpxchg16b => self.cmpxchg_double(ins, false),
            M::Bt | M::Bts | M::Btr | M::Btc => self.bit_test(ins, false),
            M::Bsf | M::Bsr => self.bit_scan(ins),
            M::Tzcnt | M::Lzcnt | M::Popcnt => self.bit_count(ins),
            M::Bswap => self.bswap(ins),
            M::Andn
            | M::Blsi
            | M::Blsr
            | M::Blsmsk
            | M::Bextr
            | M::Bzhi
            | M::Shlx
            | M::Shrx
            | M::Sarx
            | M::Rorx
            | M::Mulx
            | M::Pdep
            | M::Pext => self.bmi(ins),
            M::Rcl | M::Rcr => self.rotate_through_carry(ins),
            M::Shld | M::Shrd => self.double_shift(ins),
            M::Movbe => self.movbe(ins),
            M::Adcx | M::Adox => self.add_with_flag(ins),
            M::Crc32 => self.crc32(ins),
            M::Clc | M::Stc | M::Cmc => {
                let v = match ins.mnemonic() {
                    M::Clc => "false".to_string(),
                    M::Stc => "true".to_string(),
                    _ => {
                        let cf = self.get_flag(Flag::Cf);
                        self.val(&format!("xor i1 {cf}, true"))
                    }
                };
                self.set_flag(Flag::Cf, &v);
                Ok(())
            }
            M::Cld | M::Std => {
                let v = if ins.mnemonic() == M::Std { "true" } else { "false" };
                self.store_df(v);
                Ok(())
            }
            M::Lahf => self.lahf(),
            M::Sahf => self.sahf(),
            M::Pushfq => self.pushfq(),
            M::Popfq => self.popfq(),
            M::Pushf | M::Popf => unsupported("16-bit pushf/popf"),
            M::Enter => self.enter(ins),
            M::Xlatb => self.xlat(ins),
            M::Cpuid => self.runtime_helper(RuntimeHelper::Cpuid),
            M::Rdtsc => self.runtime_helper(RuntimeHelper::Rdtsc),
            M::Rdtscp => self.runtime_helper(RuntimeHelper::Rdtscp),
            M::Xgetbv => self.runtime_helper(RuntimeHelper::Xgetbv),
            // Ordering only: one sequentially consistent fence (`atomic.fence` on wasm).
            M::Mfence | M::Lfence | M::Sfence => {
                self.support.insert(Support::Atomics);
                self.emit("fence seq_cst");
                Ok(())
            }
            // `pause` marks a spin-wait: a compiler barrier, like every loop header (see
            // `LOOP_BARRIER`).
            M::Pause => {
                self.emit(crate::lift::LOOP_BARRIER);
                Ok(())
            }
            // Hints without an architectural effect on registers, flags or memory contents.
            M::Prefetchnta
            | M::Prefetcht0
            | M::Prefetcht1
            | M::Prefetcht2
            | M::Prefetchw
            | M::Prefetchwt1
            | M::Prefetch
            | M::Prefetchit0
            | M::Prefetchit1
            | M::Clflush
            | M::Clflushopt
            | M::Clwb
            | M::Cldemote
            | M::Reservednop
            // `monitorx` only arms an address monitor; `mwaitx` may return at any time
            // (spurious wake-ups are architectural), so returning at once is exact.
            | M::Monitorx
            | M::Mwaitx => Ok(()),
            M::Clzero => self.clzero(ins),
            _ => return None,
        };
        Some(result)
    }

    // ----- DF and RFLAGS ----------------------------------------------------------------------

    /// DF is not cached in an alloca: it lives only in the guest state (it changes rarely and
    /// is read only by string instructions and `pushfq`).
    fn df_ptr(&mut self) -> String {
        self.val(&format!(
            "getelementptr inbounds {GUEST_STATE_TYPE}, ptr %state, i64 0, i32 2, i64 {DF_INDEX}"
        ))
    }

    /// Loads DF as an `i1`.
    pub(super) fn load_df(&mut self) -> String {
        let ptr = self.df_ptr();
        let byte = self.val(&format!("load i8, ptr {ptr}"));
        self.val(&format!("trunc i8 {byte} to i1"))
    }

    fn store_df(&mut self, bit: &str) {
        let byte = if bit == "true" || bit == "false" {
            u8::from(bit == "true").to_string()
        } else {
            self.val(&format!("zext i1 {bit} to i8"))
        };
        let ptr = self.df_ptr();
        self.emit(&format!("store i8 {byte}, ptr {ptr}"));
    }

    /// RFLAGS as seen by `pushfq`: the arithmetic flags, DF, bit 1 and IF. TF, AC, ID and the
    /// system flags read as zero.
    fn rflags(&mut self) -> String {
        let mut acc = imm(RFLAGS_FIXED, 64);
        let mut bits: Vec<(String, u32)> = Vec::new();
        for (flag, bit) in RFLAGS_BITS {
            bits.push((self.get_flag(flag), bit));
        }
        bits.push((self.load_df(), RFLAGS_DF_BIT));
        for (value, bit) in bits {
            let wide = self.val(&format!("zext i1 {value} to i64"));
            let placed = self.val(&format!("shl i64 {wide}, {bit}"));
            acc = self.val(&format!("or i64 {acc}, {placed}"));
        }
        acc
    }

    fn pushfq(&mut self) -> R<()> {
        let flags = self.rflags();
        self.push(&flags, 64)
    }

    /// `popfq` loads the arithmetic flags and DF. Writes to TF, AC, ID and the system flags are
    /// ignored (user-mode code cannot observe most of them; single-stepping and alignment checks
    /// are not emulated).
    fn popfq(&mut self) -> R<()> {
        let rsp = self.load_gpr(RSP)?;
        let v = self.load_mem(&rsp, 64);
        let new_rsp = self.val(&format!("add i64 {rsp}, 8"));
        self.store_gpr(RSP, &new_rsp)?;
        for (flag, bit) in RFLAGS_BITS {
            let b = self.extract_bit(&v, 64, bit);
            self.set_flag(flag, &b);
        }
        let df = self.extract_bit(&v, 64, RFLAGS_DF_BIT);
        self.store_df(&df);
        Ok(())
    }

    /// `(v >> bit) & 1` as an `i1`.
    pub(super) fn extract_bit(&mut self, v: &str, bits: u32, bit: u32) -> String {
        let shifted = if bit == 0 {
            v.to_string()
        } else {
            self.val(&format!("lshr i{bits} {v}, {bit}"))
        };
        self.val(&format!("trunc i{bits} {shifted} to i1"))
    }

    /// `AH = SF:ZF:0:AF:0:PF:1:CF`.
    fn lahf(&mut self) -> R<()> {
        let mut acc = "2".to_string();
        for (flag, bit) in RFLAGS_BITS {
            if bit >= 8 {
                continue;
            }
            let f = self.get_flag(flag);
            let wide = self.val(&format!("zext i1 {f} to i8"));
            let placed = self.val(&format!("shl i8 {wide}, {bit}"));
            acc = self.val(&format!("or i8 {acc}, {placed}"));
        }
        self.write_reg(Register::AH, &acc, 8)
    }

    /// Loads SF, ZF, AF, PF and CF from AH (OF is unaffected).
    fn sahf(&mut self) -> R<()> {
        let (ah, _) = self.read_reg(Register::AH)?;
        for (flag, bit) in RFLAGS_BITS {
            if bit >= 8 {
                continue;
            }
            let b = self.extract_bit(&ah, 8, bit);
            self.set_flag(flag, &b);
        }
        Ok(())
    }

    // ----- stack frames ------------------------------------------------------------------------

    /// `enter size, 0`: `push rbp; rbp = rsp; rsp -= size`. Nesting levels above zero (which copy
    /// outer frame pointers) are reported as unsupported.
    fn enter(&mut self, ins: &Instruction) -> R<()> {
        if ins.code() != Code::Enterq_imm16_imm8 {
            return unsupported("16/32-bit enter");
        }
        let level = ins.immediate8_2nd() & 31;
        if level != 0 {
            return unsupported(format!("enter with nesting level {level}"));
        }
        let size = u64::from(ins.immediate16());
        let rbp = self.load_gpr(RBP)?;
        self.push(&rbp, 64)?;
        let frame = self.load_gpr(RSP)?;
        let new_rsp = self.val(&format!("sub i64 {frame}, {size}"));
        self.store_gpr(RBP, &frame)?;
        self.store_gpr(RSP, &new_rsp)
    }

    // ----- misc ----------------------------------------------------------------------------------

    /// `xlat`: `AL = [seg: rbx + zx(AL)]`.
    fn xlat(&mut self, ins: &Instruction) -> R<()> {
        let base = ins.memory_base();
        if !(base == Register::RBX || base == Register::EBX) {
            return unsupported(format!("xlat base register {base:?}"));
        }
        let (rbx, _) = self.read_reg(Register::RBX)?;
        let (al, _) = self.read_reg(Register::AL)?;
        let index = self.val(&format!("zext i8 {al} to i64"));
        let mut addr = self.val(&format!("add i64 {rbx}, {index}"));
        if base == Register::EBX {
            addr = self.val(&format!("and i64 {addr}, 4294967295"));
        }
        addr = self.add_segment_base(&addr, ins.memory_segment());
        let v = self.load_mem(&addr, 8);
        self.write_reg(Register::AL, &v, 8)
    }

    /// Adds the `fs`/`gs` base for those segments (all other segments have base 0 in 64-bit mode).
    pub(super) fn add_segment_base(&mut self, addr: &str, segment: Register) -> String {
        let field = match segment {
            Register::FS => 3,
            Register::GS => 4,
            _ => return addr.to_string(),
        };
        let seg = self.segment_base(field);
        self.val(&format!("add i64 {addr}, {seg}"))
    }

    /// AMD `clzero`: zeroes the 64-byte cache line (the line size of the target CPU) that
    /// contains `[ds:rax]`.
    fn clzero(&mut self, ins: &Instruction) -> R<()> {
        if ins.code() != Code::Clzeroq {
            return unsupported("clzero with a 32-bit address size");
        }
        let rax = self.load_gpr(RAX)?;
        let line = self.val(&format!("and i64 {rax}, -64"));
        let line = self.add_segment_base(&line, ins.segment_prefix());
        for offset in (0..64).step_by(8) {
            let addr = if offset == 0 {
                line.clone()
            } else {
                self.val(&format!("add i64 {line}, {offset}"))
            };
            self.store_mem(&addr, "0", 64);
        }
        Ok(())
    }

    /// `movbe r, m` / `movbe m, r`: a byte-swapping load or store (no flags).
    fn movbe(&mut self, ins: &Instruction) -> R<()> {
        let bits = self.op_bits(ins, 0)?;
        if self.op_bits(ins, 1)? != bits {
            return unsupported("movbe operand size");
        }
        let v = self.read_op(ins, 1, bits)?;
        let swapped = self.bswap_value(&v, bits);
        self.write_op(ins, 0, &swapped, bits)
    }

    /// `adcx` (carry in/out through CF) and `adox` (through OF); no other flag changes.
    fn add_with_flag(&mut self, ins: &Instruction) -> R<()> {
        let flag = if ins.mnemonic() == Mnemonic::Adcx {
            Flag::Cf
        } else {
            Flag::Of
        };
        let bits = self.op_bits(ins, 0)?;
        let t = format!("i{bits}");
        let a = self.read_op(ins, 0, bits)?;
        let b = self.read_op(ins, 1, bits)?;
        let c = self.get_flag(flag);
        let cz = self.val(&format!("zext i1 {c} to {t}"));
        let s = self.val(&format!("add {t} {a}, {b}"));
        let r = self.val(&format!("add {t} {s}, {cz}"));
        let lt = self.val(&format!("icmp ult {t} {r}, {a}"));
        let eq = self.val(&format!("icmp eq {t} {r}, {a}"));
        let ceq = self.val(&format!("and i1 {c}, {eq}"));
        let carry = self.val(&format!("or i1 {lt}, {ceq}"));
        self.set_flag(flag, &carry);
        self.write_op(ins, 0, &r, bits)
    }

    /// SSE4.2 `crc32 r32/r64, r/m8/16/32/64` (CRC-32C, no flags; the result is zero-extended).
    fn crc32(&mut self, ins: &Instruction) -> R<()> {
        let dst_bits = self.op_bits(ins, 0)?;
        let src_bits = self.op_bits(ins, 1)?;
        let src = self.read_op(ins, 1, src_bits)?;
        let (dst, _) = self.read_reg(ins.op_register(0))?;
        let mut crc = if dst_bits == 64 {
            self.val(&format!("trunc i64 {dst} to i32"))
        } else {
            dst
        };
        self.support.insert(Support::Crc32c);
        for byte in 0..src_bits / 8 {
            let b = if src_bits == 8 {
                src.clone()
            } else {
                let shifted = self.val(&format!("lshr i{src_bits} {src}, {}", byte * 8));
                self.val(&format!("trunc i{src_bits} {shifted} to i8"))
            };
            crc = self.val(&format!(
                "call i32 @__playgta6_crc32c_u8(i32 {crc}, i8 {b})"
            ));
        }
        if dst_bits == 64 {
            let wide = self.val(&format!("zext i32 {crc} to i64"));
            self.write_op(ins, 0, &wide, 64)
        } else {
            self.write_op(ins, 0, &crc, 32)
        }
    }

    /// Instructions whose results the runtime chooses: the registers are synced to the state,
    /// the helper reads/writes the state's GPRs (see the module docs for each contract) and the
    /// registers are reloaded.
    fn runtime_helper(&mut self, helper: RuntimeHelper) -> R<()> {
        self.support.insert(Support::Helper(helper));
        self.sync_out();
        self.emit(&format!("call void @{}(ptr %state)", helper.symbol()));
        self.sync_in();
        Ok(())
    }

    // ----- counter branches ------------------------------------------------------------------------

    /// Condition of `loop`/`loope`/`loopne` (which first decrement RCX, without touching flags)
    /// and `jrcxz`/`jecxz`. Forms with a 32-bit address size (`ECX` counter) are unsupported.
    pub(super) fn counter_branch_condition(&mut self, ins: &Instruction) -> R<String> {
        use Code as C;
        match ins.code() {
            C::Jrcxz_rel8_64 => {
                let rcx = self.load_gpr(RCX)?;
                Ok(self.val(&format!("icmp eq i64 {rcx}, 0")))
            }
            C::Jecxz_rel8_64 => {
                let rcx = self.load_gpr(RCX)?;
                let ecx = self.val(&format!("trunc i64 {rcx} to i32"));
                Ok(self.val(&format!("icmp eq i32 {ecx}, 0")))
            }
            C::Loop_rel8_64_RCX | C::Loope_rel8_64_RCX | C::Loopne_rel8_64_RCX => {
                let rcx = self.load_gpr(RCX)?;
                let next = self.val(&format!("sub i64 {rcx}, 1"));
                self.store_gpr(RCX, &next)?;
                let nonzero = self.val(&format!("icmp ne i64 {next}, 0"));
                let zf = match ins.code() {
                    C::Loope_rel8_64_RCX => self.get_flag(Flag::Zf),
                    C::Loopne_rel8_64_RCX => {
                        let zf = self.get_flag(Flag::Zf);
                        self.val(&format!("xor i1 {zf}, true"))
                    }
                    _ => return Ok(nonzero),
                };
                Ok(self.val(&format!("and i1 {nonzero}, {zf}")))
            }
            C::Loop_rel8_64_ECX | C::Loope_rel8_64_ECX | C::Loopne_rel8_64_ECX => {
                unsupported("loop with a 32-bit address size (ECX counter)")
            }
            _ => unsupported(format!(
                "conditional branch {:?} is not supported yet",
                ins.mnemonic()
            )),
        }
    }
}
