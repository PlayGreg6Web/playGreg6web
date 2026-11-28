//! Integer instructions beyond the core move/ALU set: the dispatch hook for the string
//! ([`string_ops`](super::string_ops)), bit ([`bit_ops`](super::bit_ops)) and atomic
//! ([`atomics`](super::atomics)) families, plus flag manipulation, `pushfq`/`popfq`,
//! `lahf`/`sahf`, `enter`, `xlat`, `movbe`, `adcx`/`adox`, `crc32`, `loop*`/`jrcxz`/`jecxz`
//! conditions, fences, hint no-ops and the runtime-provided `cpuid`/`rdtsc`/`rdtscp`/`xgetbv`.

use iced_x86::{Code, Instruction, Mnemonic, OpKind, Register};

use super::{
    ColdTrapKind, Flag, FlagSource, FunctionLifter, R, RBP, RSP, RegRef, imm, mask, parse_const,
    reg_ref, sign_extend_u64, ty, unsupported,
};
use crate::lift::{RuntimeHelper, Support};

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
/// `0x82F63B78`) that the SSE4.2 `crc32` instruction computes, via a 256-entry lookup table
/// (~10x faster than an 8-iteration bit loop per byte).
pub(in crate::lift) const CRC32C_IR: &str = "\
@__playgta6_crc32c_tbl = internal unnamed_addr constant [256 x i32] [\
i32 0, i32 -227835133, i32 -516198153, i32 324072436, i32 -946170081, i32 904991772, i32 648144872, i32 -724933397, \
i32 -1965467441, i32 2024987596, i32 1809983544, i32 -1719030981, i32 1296289744, i32 -1087877933, i32 -1401372889, i32 1578318884, \
i32 274646895, i32 -499825556, i32 -244992104, i32 51262619, i32 -675000208, i32 632279923, i32 922689671, i32 -996891772, \
i32 -1702387808, i32 1760304291, i32 2075979607, i32 -1982370732, i32 1562183871, i32 -1351185476, i32 -1138329528, i32 1313733451, \
i32 549293790, i32 -757723683, i32 -1048117719, i32 871202090, i32 -416867903, i32 357341890, i32 102525238, i32 -193467851, \
i32 -1436232175, i32 1477399826, i32 1264559846, i32 -1187764763, i32 1845379342, i32 -1617575411, i32 -1933233671, i32 2125378298, \
i32 820201905, i32 -1031222606, i32 -774358714, i32 598981189, i32 -143008082, i32 85089709, i32 373468761, i32 -467063462, \
i32 -1170599554, i32 1213305469, i32 1526817161, i32 -1452612982, i32 2107672161, i32 -1882520222, i32 -1667500394, i32 1861252501, \
i32 1098587580, i32 -1290756417, i32 -1606390453, i32 1378610760, i32 -2032039261, i32 1955203488, i32 1742404180, i32 -1783531177, \
i32 -878557837, i32 969524848, i32 714683780, i32 -655182201, i32 205050476, i32 -28094097, i32 -318528869, i32 526918040, \
i32 1361435347, i32 -1555146288, i32 -1340167644, i32 1114974503, i32 -1765847604, i32 1691668175, i32 2005155131, i32 -2047885768, \
i32 -604208612, i32 697762079, i32 986182379, i32 -928222744, i32 476452099, i32 -301099520, i32 -44210700, i32 255256311, \
i32 1640403810, i32 -1817374623, i32 -2130844779, i32 1922457750, i32 -1503918979, i32 1412925310, i32 1197962378, i32 -1257441399, \
i32 -350237779, i32 427051182, i32 170179418, i32 -129025959, i32 746937522, i32 -554770511, i32 -843174843, i32 1070968646, \
i32 1905808397, i32 -2081171698, i32 -1868356358, i32 1657317369, i32 -1241332974, i32 1147748369, i32 1463399397, i32 -1521340186, \
i32 -79622974, i32 153784257, i32 444234805, i32 -401473738, i32 1021025245, i32 -827320098, i32 -572462294, i32 797665321, \
i32 -2097792136, i32 1889384571, i32 1674398607, i32 -1851340660, i32 1164749927, i32 -1224265884, i32 -1537745776, i32 1446797203, \
i32 137323447, i32 -96149324, i32 -384560320, i32 461344835, i32 -810158936, i32 1037989803, i32 781091935, i32 -588970148, \
i32 -1834419177, i32 1623424788, i32 1939049696, i32 -2114449437, i32 1429367560, i32 -1487280117, i32 -1274471425, i32 1180866812, \
i32 410100952, i32 -367384613, i32 -112536529, i32 186734380, i32 -538233913, i32 763408580, i32 1053836080, i32 -860110797, \
i32 -1572096602, i32 1344288421, i32 1131464017, i32 -1323612590, i32 1708204729, i32 -1749376582, i32 -2065018290, i32 1988219213, \
i32 680717673, i32 -621187478, i32 -911630946, i32 1002577565, i32 -284657034, i32 493091189, i32 238226049, i32 -61306494, \
i32 -1307217207, i32 1082061258, i32 1395524158, i32 -1589280451, i32 1972364758, i32 -2015074603, i32 -1800104671, i32 1725896226, \
i32 952904198, i32 -894981883, i32 -638100751, i32 731699698, i32 -11092711, i32 222117402, i32 510512622, i32 -335130899, \
i32 -1014159676, i32 837199303, i32 582374963, i32 -790768336, i32 68661723, i32 -159632680, i32 -450051796, i32 390545967, \
i32 1230274059, i32 -1153434360, i32 -1469116676, i32 1510247935, i32 -1899042540, i32 2091215383, i32 1878366691, i32 -1650582816, \
i32 -741088853, i32 565732008, i32 854102364, i32 -1065151905, i32 340358836, i32 -433916489, i32 -177076669, i32 119113024, \
i32 1493875044, i32 -1419691417, i32 -1204696685, i32 1247431312, i32 -1634718085, i32 1828433272, i32 2141937292, i32 -1916740209, \
i32 -483350502, i32 291187481, i32 34330861, i32 -262120466, i32 615137029, i32 -691946490, i32 -980332558, i32 939183345, \
i32 1776939221, i32 -1685949482, i32 -1999470558, i32 2058945313, i32 -1368168502, i32 1545135305, i32 1330124605, i32 -1121741762, \
i32 -210866315, i32 17165430, i32 307568514, i32 -532767615, i32 888469610, i32 -962626711, i32 -707819363, i32 665062302, \
i32 2042050490, i32 -1948470087, i32 -1735637171, i32 1793573966, i32 -1104306011, i32 1279665062, i32 1595330642, i32 -1384295599], align 4
define internal i32 @__playgta6_crc32c_u8(i32 %crc, i8 %byte) unnamed_addr #0 {
entry:
  %b = zext i8 %byte to i32
  %x = xor i32 %crc, %b
  %idx32 = and i32 %x, 255
  %idx = zext i32 %idx32 to i64
  %ptr = getelementptr inbounds [256 x i32], ptr @__playgta6_crc32c_tbl, i64 0, i64 %idx
  %t = load i32, ptr %ptr, align 4
  %hi = lshr i32 %crc, 8
  %r = xor i32 %hi, %t
  ret i32 %r
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
                        let cf = self.take_carry_flag();
                        self.negate_i1(&cf)
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
                self.clear_window_cache();
                self.support.insert(Support::Atomics);
                self.emit("fence seq_cst");
                Ok(())
            }
            // `pause` marks a spin-wait: a compiler barrier, like every loop header (see
            // `LOOP_BARRIER`).
            M::Pause => {
                self.clear_window_cache();
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
            // No shadow stack (CET off): `rdssp` leaves its destination unchanged.
            | M::Rdsspd
            | M::Rdsspq
            // `monitorx` only arms an address monitor; `mwaitx` may return at any time
            // (spurious wake-ups are architectural), so returning at once is exact.
            | M::Monitorx
            | M::Mwaitx => Ok(()),
            M::Clzero => self.clzero(ins),
            M::Imul if ins.op_count() >= 2 => self.opt_imul_truncating(ins),
            M::Imul | M::Mul => self.opt_mul_widening(ins, ins.mnemonic() == M::Imul),
            M::Div | M::Idiv => self.opt_divide(ins, ins.mnemonic() == M::Idiv),
            M::Neg => self.opt_neg(ins),
            M::Cwd | M::Cdq | M::Cqo => self.opt_sign_extend_dx(ins),
            M::Cbw | M::Cwde | M::Cdqe => self.opt_sign_extend_ax(ins),
            M::Movzx | M::Movsx | M::Movsxd => self.opt_mov_extend(ins),
            _ => return None,
        };
        Some(result)
    }

    // ----- DF and RFLAGS ----------------------------------------------------------------------

    /// DF is not cached in an alloca: it lives only in the guest state (it changes rarely and
    /// is read only by string instructions and `pushfq`).
    fn df_ptr(&mut self) -> String {
        self.state_ptr(2, Some(DF_INDEX as usize))
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
        let c = if matches!(flag, Flag::Cf) {
            self.take_carry_flag()
        } else {
            self.get_flag(flag)
        };
        let want_flag = self.flag_live(flag);
        let r = match c.as_str() {
            "false" => {
                let r = self.val(&format!("add {t} {a}, {b}"));
                if want_flag {
                    let carry = self.val(&format!("icmp ult {t} {r}, {a}"));
                    self.set_flag(flag, &carry);
                }
                r
            }
            "true" => {
                let s = self.val(&format!("add {t} {a}, {b}"));
                let r = self.val(&format!("add {t} {s}, 1"));
                if want_flag {
                    let carry = self.val(&format!("icmp ule {t} {r}, {a}"));
                    self.set_flag(flag, &carry);
                }
                r
            }
            _ => {
                let cz = self.val(&format!("zext i1 {c} to {t}"));
                let s = self.val(&format!("add {t} {a}, {b}"));
                let r = self.val(&format!("add {t} {s}, {cz}"));
                if want_flag {
                    let lt = self.val(&format!("icmp ult {t} {r}, {a}"));
                    let eq = self.val(&format!("icmp eq {t} {r}, {a}"));
                    let ceq = self.val(&format!("and i1 {c}, {eq}"));
                    let carry = self.val(&format!("or i1 {lt}, {ceq}"));
                    self.set_flag(flag, &carry);
                }
                r
            }
        };
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
        self.clear_window_cache();
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

    // ----- bit-exact constant folding & strength reduction for mul/div/neg/ext -----------------

    /// Truncating `imul r, r/m` and `imul r, r/m, imm` with bit-exact constant folding and
    /// identity / `-1` / `±2^k` strength reduction (eliminating `mul` and 64-bit `@__playgta6_umul128`
    /// even when `CF`/`OF` are live). Always evaluates memory operands first so `#PF` is preserved.
    fn opt_imul_truncating(&mut self, ins: &Instruction) -> R<()> {
        let bits = self.op_bits(ins, 0)?;
        let t = ty(bits);
        let (a, b) = if ins.op_count() == 3 {
            let src_bits = self.op_bits(ins, 1)?;
            if src_bits != bits {
                return unsupported("imul operand size");
            }
            (self.read_op(ins, 1, bits)?, self.read_op(ins, 2, bits)?)
        } else {
            (self.read_op(ins, 0, bits)?, self.read_op(ins, 1, bits)?)
        };
        let ca = parse_const(&a, bits);
        let cb = parse_const(&b, bits);
        let want_ov = self.flag_live(Flag::Cf) || self.flag_live(Flag::Of);

        if let (Some(va), Some(vb)) = (ca, cb) {
            let sa = i128::from(sign_extend_u64(va, bits) as i64);
            let sb = i128::from(sign_extend_u64(vb, bits) as i64);
            let prod = sa * sb;
            let trunc = (prod as u64) & mask(bits);
            let ov = i128::from(sign_extend_u64(trunc, bits) as i64) != prod;
            let ov_s = if ov { "true" } else { "false" };
            self.set_flag(Flag::Cf, ov_s);
            self.set_flag(Flag::Of, ov_s);
            return self.write_op(ins, 0, &imm(trunc, bits), bits);
        }

        // Normalize `(dyn_val, const_val)` when exactly one operand is a lift-time constant.
        let one_const = match (ca, cb) {
            (None, Some(c)) => Some((a.as_str(), c)),
            (Some(c), None) => Some((b.as_str(), c)),
            _ => None,
        };
        if let Some((x, c)) = one_const {
            let m = mask(bits);
            let c_masked = c & m;
            if c_masked == 0 {
                self.set_flag(Flag::Cf, "false");
                self.set_flag(Flag::Of, "false");
                return self.write_op(ins, 0, "0", bits);
            }
            if c_masked == 1 {
                self.set_flag(Flag::Cf, "false");
                self.set_flag(Flag::Of, "false");
                return self.write_op(ins, 0, x, bits);
            }
            if c_masked == m {
                // `x * -1 = -x`; overflows signed `bits` iff `x == INT_MIN`.
                let r = self.val(&format!("sub {t} 0, {x}"));
                if want_ov {
                    let min_val = imm(1u64 << (bits - 1), bits);
                    let ov = self.val(&format!("icmp eq {t} {x}, {min_val}"));
                    self.set_flag(Flag::Cf, &ov);
                    self.set_flag(Flag::Of, &ov);
                }
                return self.write_op(ins, 0, &r, bits);
            }
            let c_signed = sign_extend_u64(c_masked, bits) as i64;
            if c_signed > 0 && (c_signed as u64).is_power_of_two() {
                // Positive power of two `2^k` (`1 <= k <= bits - 2`): `r = x << k`,
                // overflows signed `bits` iff `(r >>a k) != x`.
                let k = (c_signed as u64).trailing_zeros();
                let r = self.val(&format!("shl {t} {x}, {k}"));
                if want_ov {
                    let back = self.val(&format!("ashr {t} {r}, {k}"));
                    let ov = self.val(&format!("icmp ne {t} {back}, {x}"));
                    self.set_flag(Flag::Cf, &ov);
                    self.set_flag(Flag::Of, &ov);
                }
                return self.write_op(ins, 0, &r, bits);
            }
            if c_signed < 0 && c_signed.unsigned_abs().is_power_of_two() {
                let k = c_signed.unsigned_abs().trailing_zeros();
                if k == bits - 1 {
                    // `c == INT_MIN = -2^(bits-1)`: truncated product is `x << (bits - 1)`;
                    // fits in signed `bits` iff `x == 0` or `x == 1` (`x <=u 1`).
                    let r = self.val(&format!("shl {t} {x}, {k}"));
                    if want_ov {
                        let ov = self.val(&format!("icmp ugt {t} {x}, 1"));
                        self.set_flag(Flag::Cf, &ov);
                        self.set_flag(Flag::Of, &ov);
                    }
                    return self.write_op(ins, 0, &r, bits);
                }
                // Negative power of two `-2^k` (`1 <= k <= bits - 2`):
                // `r = -(x << k)`; fits in signed `bits` iff
                // `-(2^(bits-1-k)) + 1 <=s x <=s 2^(bits-1-k)`, i.e.
                // `(x + (2^(bits-1-k) - 1)) <=u (2^(bits-k) - 1)`.
                let sh = self.val(&format!("shl {t} {x}, {k}"));
                let r = self.val(&format!("sub {t} 0, {sh}"));
                if want_ov {
                    let lo_neg = imm((1u64 << (bits - 1 - k)) - 1, bits);
                    let span = imm((1u64 << (bits - k)) - 1, bits);
                    let shifted = self.val(&format!("add {t} {x}, {lo_neg}"));
                    let ov = self.val(&format!("icmp ugt {t} {shifted}, {span}"));
                    self.set_flag(Flag::Cf, &ov);
                    self.set_flag(Flag::Of, &ov);
                }
                return self.write_op(ins, 0, &r, bits);
            }
        }

        if !want_ov {
            let r = self.val(&format!("mul {t} {a}, {b}"));
            return self.write_op(ins, 0, &r, bits);
        }
        let (r, overflow) = if bits == 64 {
            let r = self.val(&format!("mul i64 {a}, {b}"));
            let (_, hi) = self.mul128(&a, &b, true);
            let sign = self.val(&format!("ashr i64 {r}, 63"));
            let ov = self.val(&format!("icmp ne i64 {hi}, {sign}"));
            (r, ov)
        } else {
            let wa = self.val(&format!("sext {t} {a} to i64"));
            let wb = self.val(&format!("sext {t} {b} to i64"));
            let p = self.val(&format!("mul i64 {wa}, {wb}"));
            let r = self.val(&format!("trunc i64 {p} to {t}"));
            let back = self.val(&format!("sext {t} {r} to i64"));
            let ov = self.val(&format!("icmp ne i64 {back}, {p}"));
            (r, ov)
        };
        self.set_flag(Flag::Cf, &overflow);
        self.set_flag(Flag::Of, &overflow);
        self.write_op(ins, 0, &r, bits)
    }

    /// One-operand widening `mul`/`imul` with bit-exact constant folding and `0`/`1`/`-1`/`2^k`
    /// strength reduction (eliminating 64-bit `@__playgta6_umul128` for all special factors).
    fn opt_mul_widening(&mut self, ins: &Instruction, signed: bool) -> R<()> {
        let bits = self.op_bits(ins, 0)?;
        let t = ty(bits);
        let src = self.read_op(ins, 0, bits)?;
        let (acc, dst_hi) = match bits {
            8 => (Register::AL, Register::AH),
            16 => (Register::AX, Register::DX),
            32 => (Register::EAX, Register::EDX),
            _ => (Register::RAX, Register::RDX),
        };
        let (a, _) = self.read_reg(acc)?;
        let want_ov = self.flag_live(Flag::Cf) || self.flag_live(Flag::Of);
        let ca = parse_const(&a, bits);
        let cb = parse_const(&src, bits);
        let m = mask(bits);

        if let (Some(va), Some(vb)) = (ca, cb) {
            let (prod_lo, prod_hi, ax16, ov) = if signed {
                let sa = i128::from(sign_extend_u64(va, bits) as i64);
                let sb = i128::from(sign_extend_u64(vb, bits) as i64);
                let prod = sa * sb;
                let lo = (prod as u64) & m;
                let hi = ((prod >> bits) as u64) & m;
                let ov = i128::from(sign_extend_u64(lo, bits) as i64) != prod;
                (lo, hi, (prod as u64) & 0xFFFF, ov)
            } else {
                let prod = u128::from(va & m) * u128::from(vb & m);
                let lo = (prod as u64) & m;
                let hi = ((prod >> bits) as u64) & m;
                let ov = hi != 0;
                (lo, hi, (prod as u64) & 0xFFFF, ov)
            };
            if bits == 8 {
                self.write_reg(Register::AX, &imm(ax16, 16), 16)?;
            } else {
                self.write_reg(acc, &imm(prod_lo, bits), bits)?;
                self.write_reg(dst_hi, &imm(prod_hi, bits), bits)?;
            }
            if want_ov {
                let ov_s = if ov { "true" } else { "false" };
                self.set_flag(Flag::Cf, ov_s);
                self.set_flag(Flag::Of, ov_s);
            }
            return Ok(());
        }

        let one_const = match (ca, cb) {
            (None, Some(c)) => Some((a.as_str(), c & m)),
            (Some(c), None) => Some((src.as_str(), c & m)),
            _ => None,
        };
        if let Some((x, c)) = one_const {
            if c == 0 {
                if bits == 8 {
                    self.write_reg(Register::AX, "0", 16)?;
                } else {
                    self.write_reg(acc, "0", bits)?;
                    self.write_reg(dst_hi, "0", bits)?;
                }
                if want_ov {
                    self.set_flag(Flag::Cf, "false");
                    self.set_flag(Flag::Of, "false");
                }
                return Ok(());
            }
            if c == 1 {
                if bits == 8 {
                    let ext = if signed { "sext" } else { "zext" };
                    let ax = self.val(&format!("{ext} i8 {x} to i16"));
                    self.write_reg(Register::AX, &ax, 16)?;
                } else {
                    let hi = if signed {
                        self.val(&format!("ashr {t} {x}, {}", bits - 1))
                    } else {
                        "0".to_string()
                    };
                    self.write_reg(acc, x, bits)?;
                    self.write_reg(dst_hi, &hi, bits)?;
                }
                if want_ov {
                    self.set_flag(Flag::Cf, "false");
                    self.set_flag(Flag::Of, "false");
                }
                return Ok(());
            }
            if signed && c == m {
                // Signed `x * -1`: `lo = -x`, `hi = (x >s 0) ? -1 : 0` (note: for `x == INT_MIN`,
                // product is `+2^(bits-1)` so `lo = INT_MIN` and `hi = 0`!), `ov = (x == INT_MIN)`.
                let ov = want_ov.then(|| {
                    let min_val = imm(1u64 << (bits - 1), bits);
                    self.val(&format!("icmp eq {t} {x}, {min_val}"))
                });
                if bits == 8 {
                    let ext = self.val(&format!("sext i8 {x} to i16"));
                    let ax = self.val(&format!("sub i16 0, {ext}"));
                    self.write_reg(Register::AX, &ax, 16)?;
                } else {
                    let lo = self.val(&format!("sub {t} 0, {x}"));
                    let pos = self.val(&format!("icmp sgt {t} {x}, 0"));
                    let hi = self.val(&format!("sext i1 {pos} to {t}"));
                    self.write_reg(acc, &lo, bits)?;
                    self.write_reg(dst_hi, &hi, bits)?;
                }
                if let Some(ov) = ov {
                    self.set_flag(Flag::Cf, &ov);
                    self.set_flag(Flag::Of, &ov);
                }
                return Ok(());
            }
            if !signed && bits > 8 && c.is_power_of_two() {
                // Unsigned `x * 2^k` (`1 <= k <= bits - 1`): `lo = x << k`, `hi = x >>u (bits - k)`.
                let k = c.trailing_zeros();
                let lo = self.val(&format!("shl {t} {x}, {k}"));
                let hi = self.val(&format!("lshr {t} {x}, {}", bits - k));
                let ov = want_ov.then(|| self.val(&format!("icmp ne {t} {hi}, 0")));
                self.write_reg(acc, &lo, bits)?;
                self.write_reg(dst_hi, &hi, bits)?;
                if let Some(ov) = ov {
                    self.set_flag(Flag::Cf, &ov);
                    self.set_flag(Flag::Of, &ov);
                }
                return Ok(());
            }
            if signed && bits > 8 {
                let c_signed = sign_extend_u64(c, bits) as i64;
                if c_signed > 0 && (c_signed as u64).is_power_of_two() {
                    // Signed `x * (+2^k)` (`1 <= k <= bits - 2`):
                    // `lo = x << k`, `hi = x >>a (bits - k)`, `ov = (hi != (lo >>a (bits - 1)))`.
                    let k = (c_signed as u64).trailing_zeros();
                    let lo = self.val(&format!("shl {t} {x}, {k}"));
                    let hi = self.val(&format!("ashr {t} {x}, {}", bits - k));
                    let ov = want_ov.then(|| {
                        let sign = self.val(&format!("ashr {t} {lo}, {}", bits - 1));
                        self.val(&format!("icmp ne {t} {hi}, {sign}"))
                    });
                    self.write_reg(acc, &lo, bits)?;
                    self.write_reg(dst_hi, &hi, bits)?;
                    if let Some(ov) = ov {
                        self.set_flag(Flag::Cf, &ov);
                        self.set_flag(Flag::Of, &ov);
                    }
                    return Ok(());
                }
            }
        }

        // General widening multiply (same lowering as `FunctionLifter::mul_widening`, reusing
        // already-loaded `src` and `a` without re-reading any memory operand).
        let overflow = if bits == 64 {
            let (lo, hi) = self.mul128(&a, &src, signed);
            let ov = want_ov.then(|| {
                if signed {
                    let sign = self.val(&format!("ashr i64 {lo}, 63"));
                    self.val(&format!("icmp ne i64 {hi}, {sign}"))
                } else {
                    self.val(&format!("icmp ne i64 {hi}, 0"))
                }
            });
            self.write_reg(Register::RAX, &lo, 64)?;
            self.write_reg(Register::RDX, &hi, 64)?;
            ov
        } else {
            let ext = if signed { "sext" } else { "zext" };
            let wa = self.val(&format!("{ext} {t} {a} to i64"));
            let wb = self.val(&format!("{ext} {t} {src} to i64"));
            let p = self.val(&format!("mul i64 {wa}, {wb}"));
            let lo = self.val(&format!("trunc i64 {p} to {t}"));
            let high = (bits != 8 || (want_ov && !signed))
                .then(|| self.val(&format!("lshr i64 {p}, {bits}")));
            let ov = want_ov.then(|| {
                if signed {
                    let back = self.val(&format!("sext {t} {lo} to i64"));
                    self.val(&format!("icmp ne i64 {back}, {p}"))
                } else {
                    self.val(&format!("icmp ne i64 {}, 0", high.as_ref().unwrap()))
                }
            });
            if bits == 8 {
                let ax = self.val(&format!("trunc i64 {p} to i16"));
                self.write_reg(Register::AX, &ax, 16)?;
            } else {
                let hi = self.val(&format!("trunc i64 {} to {t}", high.unwrap()));
                self.write_reg(acc, &lo, bits)?;
                self.write_reg(dst_hi, &hi, bits)?;
            }
            ov
        };
        if let Some(overflow) = overflow {
            self.set_flag(Flag::Cf, &overflow);
            self.set_flag(Flag::Of, &overflow);
        }
        Ok(())
    }

    /// `div`/`idiv` with bit-exact constant folding, power-of-two / identity strength reduction,
    /// and statically provable trap elision while strictly preserving `#DE` divide-by-zero and
    /// `INT_MIN / -1` / quotient-overflow trap semantics.
    fn opt_divide(&mut self, ins: &Instruction, signed: bool) -> R<()> {
        let bits = self.op_bits(ins, 0)?;
        let t = ty(bits);
        let divisor = self.read_op(ins, 0, bits)?;
        let cd = parse_const(&divisor, bits);

        // Constant divide by zero: trap `#DE` unconditionally without writing RAX/RDX.
        if cd == Some(0) {
            self.cold_trap_branch("true", ColdTrapKind::DivideError, ins.ip());
            return Ok(());
        }

        let (qreg, rreg) = match bits {
            8 => (Register::AL, Register::AH),
            16 => (Register::AX, Register::DX),
            32 => (Register::EAX, Register::EDX),
            _ => (Register::RAX, Register::RDX),
        };

        // Fast paths when `divisor` is a known non-zero constant and high half is known zero
        // (unsigned 32/64-bit) or sign-extended via `cdq`/`cqo` (signed 32/64-bit).
        if matches!(bits, 32 | 64)
            && let Some(vd) = cd
        {
            if !signed
                && self.known.gprs[RDX]
                    .as_deref()
                    .and_then(super::memory_access::const_addr)
                    .is_some_and(|v| (v & mask(bits)) == 0)
            {
                let (lo, _) = self.read_reg(qreg)?;
                if let Some(vlo) = parse_const(&lo, bits) {
                    let q = imm(vlo / vd, bits);
                    let r = imm(vlo % vd, bits);
                    self.write_reg(qreg, &q, bits)?;
                    return self.write_reg(rreg, &r, bits);
                }
                let (q, r) = if vd == 1 {
                    (lo, "0".to_string())
                } else if vd.is_power_of_two() {
                    let k = vd.trailing_zeros();
                    let q = self.val(&format!("lshr {t} {lo}, {k}"));
                    let r = self.val(&format!("and {t} {lo}, {}", imm(vd - 1, bits)));
                    (q, r)
                } else {
                    // `vd != 0` and `hi == 0`, so `#DE` is impossible.
                    let q = self.val(&format!("udiv {t} {lo}, {divisor}"));
                    let r = self.val(&format!("urem {t} {lo}, {divisor}"));
                    (q, r)
                };
                self.write_reg(qreg, &q, bits)?;
                return self.write_reg(rreg, &r, bits);
            }
            if signed
                && matches!(
                    &self.known.sign_ext_rdx,
                    Some((rdx64, rax64, b))
                        if *b == bits
                            && self.known.gprs[RDX].as_deref() == Some(rdx64.as_str())
                            && self.known.gprs[RAX].as_deref() == Some(rax64.as_str())
                )
            {
                let (lo, _) = self.read_reg(qreg)?;
                let sd = sign_extend_u64(vd, bits) as i64;
                if sd == 1 {
                    // `lo / 1 == lo`, `lo % 1 == 0`; never overflows (even for `INT_MIN`).
                    self.write_reg(qreg, &lo, bits)?;
                    return self.write_reg(rreg, "0", bits);
                }
                if sd == -1 {
                    let min_u = 1u64 << (bits - 1);
                    if let Some(vlo) = parse_const(&lo, bits) {
                        if vlo == min_u {
                            self.cold_trap_branch("true", ColdTrapKind::DivideError, ins.ip());
                            return Ok(());
                        }
                        let q = imm(0u64.wrapping_sub(vlo) & mask(bits), bits);
                        self.write_reg(qreg, &q, bits)?;
                        return self.write_reg(rreg, "0", bits);
                    }
                    let min_imm = imm(min_u, bits);
                    let fault = self.val(&format!("icmp eq {t} {lo}, {min_imm}"));
                    self.cold_trap_branch(&fault, ColdTrapKind::DivideError, ins.ip());
                    let q = self.val(&format!("sub {t} 0, {lo}"));
                    self.write_reg(qreg, &q, bits)?;
                    return self.write_reg(rreg, "0", bits);
                }
                // `sd != 0 && sd != -1` with sign-extended `rdx`: `#DE` is impossible!
                if let Some(vlo) = parse_const(&lo, bits) {
                    let slo = sign_extend_u64(vlo, bits) as i64;
                    let q = imm((slo / sd) as u64, bits);
                    let r = imm((slo % sd) as u64, bits);
                    self.write_reg(qreg, &q, bits)?;
                    return self.write_reg(rreg, &r, bits);
                }
                let q = self.val(&format!("sdiv {t} {lo}, {divisor}"));
                let r = self.val(&format!("srem {t} {lo}, {divisor}"));
                self.write_reg(qreg, &q, bits)?;
                return self.write_reg(rreg, &r, bits);
            }
        }

        // Full constant folding for any width (8/16/32/64-bit) when the entire 2N-bit dividend
        // and non-zero divisor are lift-time constants.
        if let Some(vd) = cd {
            let const_hi_lo = if bits == 8 {
                let (ax, _) = self.read_reg(Register::AX)?;
                parse_const(&ax, 16).map(|v| ((v >> 8) & 0xFF, v & 0xFF))
            } else {
                let (h, _) = self.read_reg(rreg)?;
                let (l, _) = self.read_reg(qreg)?;
                match (parse_const(&h, bits), parse_const(&l, bits)) {
                    (Some(vh), Some(vl)) => Some((vh, vl)),
                    _ => None,
                }
            };
            if let Some((vh, vl)) = const_hi_lo {
                if !signed {
                    if vh >= vd {
                        self.cold_trap_branch("true", ColdTrapKind::DivideError, ins.ip());
                        return Ok(());
                    }
                    let dividend = (u128::from(vh) << bits) | u128::from(vl);
                    let d128 = u128::from(vd);
                    let q = imm((dividend / d128) as u64, bits);
                    let r = imm((dividend % d128) as u64, bits);
                    self.write_reg(qreg, &q, bits)?;
                    return self.write_reg(rreg, &r, bits);
                }
                let sh = i128::from(sign_extend_u64(vh, bits) as i64);
                let dividend = (sh << bits) | i128::from(vl);
                let sd = i128::from(sign_extend_u64(vd, bits) as i64);
                if bits == 64 && dividend == i128::MIN && sd == -1 {
                    self.cold_trap_branch("true", ColdTrapKind::DivideError, ins.ip());
                    return Ok(());
                }
                let q128 = dividend / sd;
                let r128 = dividend % sd;
                let min_q = -(1i128 << (bits - 1));
                let max_q = (1i128 << (bits - 1)) - 1;
                if q128 < min_q || q128 > max_q {
                    self.cold_trap_branch("true", ColdTrapKind::DivideError, ins.ip());
                    return Ok(());
                }
                let q = imm(q128 as u64, bits);
                let r = imm(r128 as u64, bits);
                self.write_reg(qreg, &q, bits)?;
                return self.write_reg(rreg, &r, bits);
            }
        }

        // General dynamic divide path (reuses already-loaded `divisor` without re-reading memory).
        let ext = if signed { "sext" } else { "zext" };
        if matches!(bits, 32 | 64) {
            if !signed
                && self.known.gprs[RDX]
                    .as_deref()
                    .and_then(super::memory_access::const_addr)
                    .is_some_and(|v| (v & mask(bits)) == 0)
            {
                let (lo, _) = self.read_reg(qreg)?;
                let fault = self.val(&format!("icmp eq {t} {divisor}, 0"));
                self.cold_trap_branch(&fault, ColdTrapKind::DivideError, ins.ip());
                let q = self.val(&format!("udiv {t} {lo}, {divisor}"));
                let r = self.val(&format!("urem {t} {lo}, {divisor}"));
                self.write_reg(qreg, &q, bits)?;
                return self.write_reg(rreg, &r, bits);
            }
            if signed
                && matches!(
                    &self.known.sign_ext_rdx,
                    Some((rdx64, rax64, b))
                        if *b == bits
                            && self.known.gprs[RDX].as_deref() == Some(rdx64.as_str())
                            && self.known.gprs[RAX].as_deref() == Some(rax64.as_str())
                )
            {
                let (lo, _) = self.read_reg(qreg)?;
                let d_zero = self.val(&format!("icmp eq {t} {divisor}, 0"));
                let min_val = imm(1u64 << (bits - 1), bits);
                let is_min = self.val(&format!("icmp eq {t} {lo}, {min_val}"));
                let is_neg1 = self.val(&format!("icmp eq {t} {divisor}, -1"));
                let ov = self.val(&format!("and i1 {is_min}, {is_neg1}"));
                let fault = self.val(&format!("or i1 {d_zero}, {ov}"));
                self.cold_trap_branch(&fault, ColdTrapKind::DivideError, ins.ip());
                let q = self.val(&format!("sdiv {t} {lo}, {divisor}"));
                let r = self.val(&format!("srem {t} {lo}, {divisor}"));
                self.write_reg(qreg, &q, bits)?;
                return self.write_reg(rreg, &r, bits);
            }
        }

        if bits < 64 {
            let (dividend, hi_part) = if bits == 8 {
                let (ax, _) = self.read_reg(Register::AX)?;
                let div = self.val(&format!("{ext} i16 {ax} to i64"));
                let hi = (!signed).then(|| self.val(&format!("lshr i64 {div}, 8")));
                (div, hi)
            } else {
                let (h, _) = self.read_reg(rreg)?;
                let (l, _) = self.read_reg(qreg)?;
                let h64 = self.val(&format!("zext {t} {h} to i64"));
                let l64 = self.val(&format!("zext {t} {l} to i64"));
                let hsh = self.val(&format!("shl i64 {h64}, {bits}"));
                let joined = self.val(&format!("or i64 {hsh}, {l64}"));
                let div = if signed && bits == 16 {
                    let narrow = self.val(&format!("trunc i64 {joined} to i32"));
                    self.val(&format!("sext i32 {narrow} to i64"))
                } else {
                    joined
                };
                (div, (!signed).then_some(h64))
            };
            let d = self.val(&format!("{ext} {t} {divisor} to i64"));
            let (q, r) = if !signed {
                let hi_part = hi_part.expect("unsigned hi_part");
                let fault = self.val(&format!("icmp uge i64 {hi_part}, {d}"));
                self.cold_trap_branch(&fault, ColdTrapKind::DivideError, ins.ip());
                let q64 = self.val(&format!("udiv i64 {dividend}, {d}"));
                let r64 = self.val(&format!("urem i64 {dividend}, {d}"));
                (
                    self.val(&format!("trunc i64 {q64} to {t}")),
                    self.val(&format!("trunc i64 {r64} to {t}")),
                )
            } else {
                let d_zero = self.val(&format!("icmp eq i64 {d}, 0"));
                let pre_fault = if bits == 32 {
                    let is_min =
                        self.val(&format!("icmp eq i64 {dividend}, -9223372036854775808"));
                    let is_neg1 = self.val(&format!("icmp eq i64 {d}, -1"));
                    let sdiv_ov = self.val(&format!("and i1 {is_min}, {is_neg1}"));
                    self.val(&format!("or i1 {d_zero}, {sdiv_ov}"))
                } else {
                    d_zero
                };
                self.cold_trap_branch(&pre_fault, ColdTrapKind::DivideError, ins.ip());
                let q64 = self.val(&format!("sdiv i64 {dividend}, {d}"));
                let r64 = self.val(&format!("srem i64 {dividend}, {d}"));
                let qn = self.val(&format!("trunc i64 {q64} to {t}"));
                let rn = self.val(&format!("trunc i64 {r64} to {t}"));
                let back = self.val(&format!("sext {t} {qn} to i64"));
                let range = self.val(&format!("icmp ne i64 {back}, {q64}"));
                self.cold_trap_branch(&range, ColdTrapKind::DivideError, ins.ip());
                (qn, rn)
            };
            self.write_reg(qreg, &q, bits)?;
            return self.write_reg(rreg, &r, bits);
        }

        let (hi, _) = self.read_reg(Register::RDX)?;
        let (lo, _) = self.read_reg(Register::RAX)?;
        let helper = if signed {
            self.support.insert(Support::Sdiv128);
            "__playgta6_sdiv128"
        } else {
            self.support.insert(Support::Udiv128);
            "__playgta6_udiv128"
        };
        let res = self.val(&format!(
            "call {{ i64, i64, i1 }} @{helper}(i64 {hi}, i64 {lo}, i64 {divisor})"
        ));
        let q = self.val(&format!("extractvalue {{ i64, i64, i1 }} {res}, 0"));
        let r = self.val(&format!("extractvalue {{ i64, i64, i1 }} {res}, 1"));
        let fault = self.val(&format!("extractvalue {{ i64, i64, i1 }} {res}, 2"));
        self.cold_trap_branch(&fault, ColdTrapKind::DivideError, ins.ip());
        self.write_reg(Register::RAX, &q, 64)?;
        self.write_reg(Register::RDX, &r, 64)
    }

    /// `neg` with full constant folding (result + all 6 arithmetic flags) and simplified dynamic
    /// `AF = ((a & 0xF) != 0)` (2 ops instead of `xor` + `and 16` + `icmp ne 0`).
    fn opt_neg(&mut self, ins: &Instruction) -> R<()> {
        let bits = self.op_bits(ins, 0)?;
        let t = ty(bits);
        let a = self.read_op(ins, 0, bits)?;
        let min_u = 1u64 << (bits - 1);
        if let Some(va) = parse_const(&a, bits) {
            let r_u = 0u64.wrapping_sub(va) & mask(bits);
            let r_s = imm(r_u, bits);
            let b_str = |b: bool| if b { "true" } else { "false" };
            self.set_flag(Flag::Cf, b_str(va != 0));
            self.set_flag(Flag::Of, b_str(va == min_u));
            self.set_flag(Flag::Af, b_str((va & 0xF) != 0));
            self.set_flag(Flag::Zf, b_str(r_u == 0));
            self.set_flag(Flag::Sf, b_str((r_u & min_u) != 0));
            self.set_flag(Flag::Pf, b_str(((r_u as u8).count_ones() & 1) == 0));
            self.known.flag_source = Some(if va == 0 {
                FlagSource::ConstZero
            } else {
                FlagSource::Compare {
                    a: "0".to_string(),
                    b: a,
                    diff: Some(r_s.clone()),
                    bits,
                }
            });
            return self.write_op(ins, 0, &r_s, bits);
        }
        let r = self.val(&format!("sub {t} 0, {a}"));
        if self.flag_live(Flag::Cf) {
            let cf = self.val(&format!("icmp ne {t} {a}, 0"));
            self.set_flag(Flag::Cf, &cf);
        }
        if self.flag_live(Flag::Of) {
            let min = imm(min_u, bits);
            let of = self.val(&format!("icmp eq {t} {a}, {min}"));
            self.set_flag(Flag::Of, &of);
        }
        if self.flag_live(Flag::Af) {
            let low = self.val(&format!("and {t} {a}, 15"));
            let af = self.val(&format!("icmp ne {t} {low}, 0"));
            self.set_flag(Flag::Af, &af);
        }
        self.set_zsp(&r, bits);
        self.known.flag_source = Some(FlagSource::Compare {
            a: "0".to_string(),
            b: a,
            diff: Some(r.clone()),
            bits,
        });
        self.write_op(ins, 0, &r, bits)
    }

    /// `cwd`/`cdq`/`cqo`: folds constant and provably non-negative inputs (`sign bit known 0`)
    /// directly to `0` (or `-1`) without emitting `ashr`, while preserving `known.sign_ext_rdx`.
    fn opt_sign_extend_dx(&mut self, ins: &Instruction) -> R<()> {
        let (src, dst) = match ins.mnemonic() {
            Mnemonic::Cwd => (Register::AX, Register::DX),
            Mnemonic::Cdq => (Register::EAX, Register::EDX),
            _ => (Register::RAX, Register::RDX),
        };
        let (v, bits) = self.read_reg(src)?;
        let sign_bit = 1u64 << (bits - 1);
        let sign = if let Some(c) = parse_const(&v, bits) {
            if (c & sign_bit) != 0 {
                "-1".to_string()
            } else {
                "0".to_string()
            }
        } else if (self.nonzero_bits(&v, bits) & sign_bit) == 0 {
            "0".to_string()
        } else {
            self.val(&format!("ashr i{bits} {v}, {}", bits - 1))
        };
        self.write_reg(dst, &sign, bits)?;
        if matches!(bits, 32 | 64)
            && let (Some(rdx64), Some(rax64)) = (
                self.known.gprs[RDX].clone(),
                self.known.gprs[RAX].clone(),
            )
        {
            self.known.sign_ext_rdx = Some((rdx64, rax64, bits));
        }
        Ok(())
    }

    /// `cbw`/`cwde`/`cdqe`: folds constants and elides `trunc`/`sext` when the sign bit is
    /// provably zero (e.g. `movzx eax, word ptr [rcx]; cdqe` becomes a pure no-op).
    fn opt_sign_extend_ax(&mut self, ins: &Instruction) -> R<()> {
        let (src, dst, dst_bits) = match ins.mnemonic() {
            Mnemonic::Cbw => (Register::AL, Register::AX, 16),
            Mnemonic::Cwde => (Register::AX, Register::EAX, 32),
            _ => (Register::EAX, Register::RAX, 64),
        };
        if ins.mnemonic() == Mnemonic::Cdqe {
            let full = self.load_gpr(RAX)?;
            if let Some(c) = parse_const(&full, 64) {
                return self.store_gpr(RAX, &imm(sign_extend_u64(c & mask(32), 32), 64));
            }
            let nz = self.nonzero_bits(&full, 64);
            if (nz & !0x7FFF_FFFF) == 0 {
                // Already zero-extended and bit 31 == 0: `cdqe` is an architectural no-op!
                return Ok(());
            }
            if (nz & 0x8000_0000) == 0 {
                // Bit 31 == 0, upper 32 may be dirty: mask to 32 bits instead of `trunc` + `sext`.
                let wide = self.val(&format!("and i64 {full}, 4294967295"));
                return self.store_gpr(RAX, &wide);
            }
        }
        let (v, src_bits) = self.read_reg(src)?;
        let sign_bit = 1u64 << (src_bits - 1);
        let ext = if let Some(c) = parse_const(&v, src_bits) {
            imm(sign_extend_u64(c, src_bits), dst_bits)
        } else if (self.nonzero_bits(&v, src_bits) & sign_bit) == 0 {
            self.val(&format!("zext i{src_bits} {v} to i{dst_bits}"))
        } else {
            self.val(&format!("sext i{src_bits} {v} to i{dst_bits}"))
        };
        self.write_reg(dst, &ext, dst_bits)
    }

    /// `movzx`/`movsx`/`movsxd` with constant folding, direct 64-bit masking for high-byte
    /// `movzx r32/r64, r8h`, and zero-cost `sext` elision when the source sign bit is known zero.
    fn opt_mov_extend(&mut self, ins: &Instruction) -> R<()> {
        let dst_bits = self.op_bits(ins, 0)?;
        let src_bits = self.op_bits(ins, 1)?;
        if src_bits > dst_bits {
            return unsupported("narrowing extension");
        }
        let is_zx = ins.mnemonic() == Mnemonic::Movzx;

        if ins.op_kind(0) == OpKind::Register && matches!(dst_bits, 32 | 64) {
            let RegRef::Low(dst_idx, _) = reg_ref(ins.op_register(0))? else {
                return unsupported("mov extend destination");
            };
            if is_zx && src_bits < 32 {
                let wide = if ins.op_kind(1) == OpKind::Register {
                    match reg_ref(ins.op_register(1))? {
                        RegRef::Low(src_idx, sb) if sb == src_bits => {
                            let full = self.load_gpr(src_idx)?;
                            self.val(&format!("and i64 {full}, {}", mask(src_bits)))
                        }
                        RegRef::High8(src_idx) if src_bits == 8 => {
                            let full = self.load_gpr(src_idx)?;
                            if let Some(c) = parse_const(&full, 64) {
                                imm((c >> 8) & 0xFF, 64)
                            } else {
                                let shifted = self.val(&format!("lshr i64 {full}, 8"));
                                self.val(&format!("and i64 {shifted}, 255"))
                            }
                        }
                        _ => {
                            let v = self.read_op(ins, 1, src_bits)?;
                            self.val(&format!("zext i{src_bits} {v} to i64"))
                        }
                    }
                } else {
                    let v = self.read_op(ins, 1, src_bits)?;
                    self.val(&format!("zext i{src_bits} {v} to i64"))
                };
                return self.store_gpr(dst_idx, &wide);
            }
            if !is_zx && ins.op_kind(1) == OpKind::Register {
                match reg_ref(ins.op_register(1))? {
                    RegRef::Low(src_idx, sb) if sb == src_bits => {
                        let full = self.load_gpr(src_idx)?;
                        if let Some(c) = parse_const(&full, 64) {
                            let ext = sign_extend_u64(c & mask(src_bits), src_bits);
                            let final_v = if dst_bits == 32 { ext & mask(32) } else { ext };
                            return self.store_gpr(dst_idx, &imm(final_v, 64));
                        }
                        let sign_bit = 1u64 << (src_bits - 1);
                        let nz = self.nonzero_bits(&full, 64);
                        if (nz & sign_bit) == 0 {
                            // Sign bit of source is provably 0: `movsx`/`movsxd` is identical to
                            // zero-extension, avoiding `trunc` + `sext` (+ `zext` for 32-bit)!
                            let wide = self.val(&format!("and i64 {full}, {}", mask(src_bits)));
                            return self.store_gpr(dst_idx, &wide);
                        }
                    }
                    RegRef::High8(src_idx) if src_bits == 8 => {
                        let full = self.load_gpr(src_idx)?;
                        if let Some(c) = parse_const(&full, 64) {
                            let ext = sign_extend_u64((c >> 8) & 0xFF, 8);
                            let final_v = if dst_bits == 32 { ext & mask(32) } else { ext };
                            return self.store_gpr(dst_idx, &imm(final_v, 64));
                        }
                    }
                    _ => {}
                }
            }
        }

        let v = self.read_op(ins, 1, src_bits)?;
        let out = if src_bits == dst_bits {
            v
        } else if let Some(c) = parse_const(&v, src_bits) {
            let ext = if is_zx { c } else { sign_extend_u64(c, src_bits) };
            imm(ext, dst_bits)
        } else {
            let sign_bit = 1u64 << (src_bits - 1);
            let ext = if is_zx || (self.nonzero_bits(&v, src_bits) & sign_bit) == 0 {
                "zext"
            } else {
                "sext"
            };
            self.val(&format!("{ext} i{src_bits} {v} to i{dst_bits}"))
        };
        self.write_op(ins, 0, &out, dst_bits)
    }
}
