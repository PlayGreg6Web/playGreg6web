//! Per-function translation of decoded x86-64 instructions into LLVM IR text.
//!
//! See the parent module for the state, memory, call and flag model. Every instruction is lifted
//! into the function body inside a checkpoint: if any part of it turns out to be unsupported, the
//! partial output is rolled back and replaced by an `__playgta6_unimplemented` trap.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::Write as _;

use iced_x86::{
    Code, ConditionCode, FlowControl, Formatter, Instruction, InstructionInfoFactory,
    IntelFormatter, Mnemonic, OpAccess, OpKind, Register, RflagsBits,
};

use super::{
    GUEST_BASE_SYMBOL, GUEST_STATE_TYPE, HELPER_DIVIDE_ERROR, HELPER_INDIRECT_CALL, HELPER_TRAP,
    HELPER_UNIMPLEMENTED, LiftError, LiftedFunction, MemoryModel, Support, TBAA_GUEST_TAG,
    TBAA_STATE_TAG, UnsupportedInstruction, function_symbol,
};
use crate::cfg::{self, CallTarget, DiagnosticKind, Edge, Terminator};

#[path = "simd/mod.rs"]
pub(super) mod simd;

// Integer instruction families beyond the core set (child modules so they can use the private
// lifter primitives; the files live next to this one).
#[path = "atomics.rs"]
mod atomics;
#[path = "bit_ops.rs"]
mod bit_ops;
#[path = "flag_liveness.rs"]
pub(super) mod flag_liveness;
#[path = "integer.rs"]
mod integer;
#[path = "memory_access.rs"]
mod memory_access;
#[path = "string_ops.rs"]
mod string_ops;
#[path = "unwind.rs"]
mod unwind;

use flag_liveness::{
    ALL_FLAGS, ALL_STATE, ALL_XMM_STATE, ALL_YMMH_STATE, FlagMask, FlagSummaries, StateMask,
};

pub(super) use atomics::CMPXCHG16B_IR;
pub(super) use bit_ops::{PDEP_IR, PEXT_IR};
pub(super) use integer::CRC32C_IR;

const GPR_NAMES: [&str; 16] = [
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13",
    "r14", "r15",
];
const RSP: usize = 4;
const RBP: usize = 5;

/// Maximum x86 instruction count for a tiny internal true-leaf function to receive the LLVM
/// `alwaysinline` function attribute (R65b).
const TINY_LEAF_MAX_INSNS: usize = 10;

/// Arithmetic flags in guest-state order (`STATE_FLAGS_OFFSET + index`).
#[derive(Debug, Clone, Copy)]
enum Flag {
    Cf = 0,
    Pf = 1,
    Af = 2,
    Zf = 3,
    Sf = 4,
    Of = 5,
}

const FLAGS: [Flag; 6] = [Flag::Cf, Flag::Pf, Flag::Af, Flag::Zf, Flag::Sf, Flag::Of];

impl Flag {
    fn slot(self) -> &'static str {
        match self {
            Flag::Cf => "%cf",
            Flag::Pf => "%pf",
            Flag::Af => "%af",
            Flag::Zf => "%zf",
            Flag::Sf => "%sf",
            Flag::Of => "%of",
        }
    }

    fn rflags_bit(self) -> u32 {
        match self {
            Flag::Cf => RflagsBits::CF,
            Flag::Pf => RflagsBits::PF,
            Flag::Af => RflagsBits::AF,
            Flag::Zf => RflagsBits::ZF,
            Flag::Sf => RflagsBits::SF,
            Flag::Of => RflagsBits::OF,
        }
    }
}

/// Kind of cold noreturn runtime trap routed through a function's shared `cold_trap` sink.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ColdTrapKind {
    Trap,
    Unimplemented,
    DivideError,
}

impl ColdTrapKind {
    fn helper(self) -> &'static str {
        match self {
            Self::Trap => HELPER_TRAP,
            Self::Unimplemented => HELPER_UNIMPLEMENTED,
            Self::DivideError => HELPER_DIVIDE_ERROR,
        }
    }

    fn tag(self) -> u8 {
        match self {
            Self::Trap => 0,
            Self::Unimplemented => 1,
            Self::DivideError => 2,
        }
    }
}

/// Why a single instruction could not be lifted.
enum InsnError {
    /// Replace the instruction with an `__playgta6_unimplemented` trap.
    Unsupported(String),
    /// The whole function is malformed.
    Cfg(LiftError),
}

impl From<LiftError> for InsnError {
    fn from(err: LiftError) -> Self {
        InsnError::Cfg(err)
    }
}

type R<T> = Result<T, InsnError>;

fn unsupported<T>(reason: impl Into<String>) -> R<T> {
    Err(InsnError::Unsupported(reason.into()))
}

/// A general purpose register reference: `(gpr index, width in bits)` or a legacy high-byte reg.
enum RegRef {
    Low(usize, u32),
    High8(usize),
}

#[derive(Clone, Copy)]
enum Alu {
    Add,
    Adc,
    Sub,
    Sbb,
    Cmp,
    And,
    Or,
    Xor,
    Test,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Shift {
    Shl,
    Shr,
    Sar,
    Rol,
    Ror,
}

fn ty(bits: u32) -> String {
    format!("i{bits}")
}

fn mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

/// An integer literal of width `bits`, printed as a signed decimal so it is valid for any width.
fn imm(value: u64, bits: u32) -> String {
    if bits >= 64 {
        return (value as i64).to_string();
    }
    let v = value & mask(bits);
    let signed = if v >> (bits - 1) & 1 == 1 {
        (v | !mask(bits)) as i64
    } else {
        v as i64
    };
    signed.to_string()
}

fn block_label(va: u64) -> String {
    format!("bb_{va:X}")
}

fn exit_label(va: u64) -> String {
    format!("ext_{va:X}")
}

fn reg_ref(reg: Register) -> R<RegRef> {
    if !reg.is_gpr() {
        return unsupported(format!(
            "register {reg:?} is not a general purpose register"
        ));
    }
    let full = reg.full_register();
    if !full.is_gpr64() {
        return unsupported(format!("register {reg:?} has no 64-bit parent"));
    }
    let idx = full.number();
    Ok(match reg {
        Register::AH | Register::CH | Register::DH | Register::BH => RegRef::High8(idx),
        _ => RegRef::Low(idx, (reg.size() * 8) as u32),
    })
}

fn is_immediate(kind: OpKind) -> bool {
    matches!(
        kind,
        OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64
            | OpKind::Immediate32to64
    )
}

/// Lifts one function (see `super::lift_function`).
pub(super) fn lift(
    func: &cfg::Function,
    memory: MemoryModel,
    flag_summaries: Option<&FlagSummaries>,
) -> Result<LiftedFunction, LiftError> {
    if !func.blocks.contains_key(&func.entry) {
        return Err(LiftError::MissingEntryBlock { entry: func.entry });
    }
    let mut regs = [false; 16];
    regs[RSP] = true;
    let mut flags = [false; 6];
    let mut vector_regs = simd::Regs::default();
    let mut faulting_calls = BTreeSet::new();
    let mut return_blocks = 0usize;
    let mut info = InstructionInfoFactory::new();
    for (&start, block) in &func.blocks {
        let first = block
            .instructions
            .first()
            .map_or(block.start, Instruction::ip);
        if block.start != start || first != start {
            return Err(LiftError::BlockStartMismatch {
                block: start,
                first_instruction: first,
            });
        }
        if block.instructions.is_empty() && block.terminator != Terminator::Invalid {
            return Err(LiftError::EmptyBlock { block: start });
        }
        if block.terminator == Terminator::Return {
            return_blocks += 1;
        }
        for ins in &block.instructions {
            let mut touched =
                ins.rflags_read() | ins.rflags_written() | ins.rflags_cleared() | ins.rflags_set();
            match ins.mnemonic() {
                Mnemonic::Rol | Mnemonic::Ror | Mnemonic::Rcl | Mnemonic::Rcr => {
                    touched |= RflagsBits::CF | RflagsBits::OF;
                }
                Mnemonic::Shl
                | Mnemonic::Sal
                | Mnemonic::Shr
                | Mnemonic::Sar
                | Mnemonic::Shld
                | Mnemonic::Shrd => {
                    touched |= RflagsBits::CF
                        | RflagsBits::OF
                        | RflagsBits::ZF
                        | RflagsBits::SF
                        | RflagsBits::PF;
                }
                _ => {}
            }
            if touched != 0 {
                for flag in FLAGS {
                    if touched & flag.rflags_bit() != 0 {
                        flags[flag as usize] = true;
                    }
                }
            }
            let ins_info = info.info(ins);
            for used in ins_info.used_registers() {
                let reg = used.register();
                if reg.is_gpr() && reg.full_register().is_gpr64() {
                    regs[reg.full_register().number()] = true;
                }
                if !matches!(ins.mnemonic(), Mnemonic::Vzeroupper | Mnemonic::Vzeroall) {
                    simd::note_register(reg, used.access(), &mut vector_regs);
                }
            }
            simd::note_instruction(ins, ins_info, &mut vector_regs);
        }
        // Direct calls whose target discovery found to be unmapped or non-executable fault.
        for call in &block.calls {
            if let CallTarget::Direct(target) = call.target
                && block.diagnostics.iter().any(|d| {
                    d.site == call.site
                        && d.address == target
                        && matches!(
                            d.kind,
                            DiagnosticKind::Unmapped | DiagnosticKind::NotExecutable
                        )
                })
            {
                faulting_calls.insert(call.site);
            }
        }
    }

    let (written_state, written_flags) = flag_liveness::own_written_state_and_flags(func);
    let mut lifter = FunctionLifter {
        func,
        regs,
        flags,
        written_state,
        written_flags,
        simd: vector_regs,
        faulting_calls,
        body: String::new(),
        current_block: String::new(),
        tmp: 0,
        label: 0,
        exits: Vec::new(),
        callees: Vec::new(),
        support: BTreeSet::new(),
        unsupported: Vec::new(),
        ea: None,
        memory,
        stack_values: BTreeMap::new(),
        window_load_cache: Vec::new(),
        burst_override: None,
        last_access: None,
        guest_base: None,
        cold_traps: Vec::new(),
        share_ret: return_blocks > 1,
        has_shared_ret: false,
        unwind_sites: 0,
        state_ptrs: BTreeMap::new(),
        flag_summaries,
        block_live_out: flag_liveness::block_live_out(func, flag_summaries),
        dead_flags: 0,
        carry_source: None,
        ymmh_zeroed: 0,
        known: Known::default(),
        block_entry_facts: [GprFact::Bottom; 16],
        edge_known: Vec::new(),
        cse: BTreeMap::new(),
    };
    lifter.lift_blocks()?;
    Ok(lifter.finish())
}

/// Removes stores to a GPR or flag alloca that a later store to the same alloca overwrites in the
/// same LLVM block with no use of the alloca in between. The allocas never escape (only `load`
/// and `store` name them), so nothing else can read the overwritten value; a terminator or label
/// ends the block and counts as a use of every alloca, as a successor may load it. With the load
/// forwarding of [`FunctionLifter::known`], a register written twice in a block (`push`/`pop`
/// pairs, `rsp` updates, flags of consecutive ALU instructions) keeps only its last store.
fn drop_overwritten_slot_stores(ir: &str) -> String {
    const TERMINATORS: [&str; 8] = [
        "br ",
        "switch ",
        "ret ",
        "unreachable",
        "indirectbr ",
        "invoke ",
        "resume ",
        "callbr ",
    ];
    let slot_of = |name: &str| -> Option<usize> {
        GPR_NAMES.iter().position(|&g| g == name).or_else(|| {
            FLAGS
                .iter()
                .position(|f| &f.slot()[1..] == name)
                .map(|i| GPR_NAMES.len() + i)
        })
    };
    let lines: Vec<&str> = ir.lines().collect();
    let mut keep = vec![true; lines.len()];
    let mut pending: [Option<usize>; 22] = [None; 22];
    for (i, line) in lines.iter().enumerate() {
        let code = line.trim();
        if !line.starts_with(' ') || TERMINATORS.iter().any(|t| code.starts_with(t)) {
            pending = [None; 22];
            continue;
        }
        let stored = code
            .strip_prefix("store ")
            .and_then(|rest| rest.rsplit_once(", ptr %"))
            .and_then(|(_, slot)| slot_of(slot));
        // Every alloca the line names (as a whole `%name` token) is read, except the stored one.
        let mut rest = code;
        while let Some(at) = rest.find('%') {
            rest = &rest[at + 1..];
            let end = rest
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '.'))
                .unwrap_or(rest.len());
            if let Some(slot) = slot_of(&rest[..end])
                && Some(slot) != stored
            {
                pending[slot] = None;
            }
            rest = &rest[end..];
        }
        if let Some(slot) = stored {
            if let Some(earlier) = pending[slot] {
                keep[earlier] = false;
            }
            pending[slot] = Some(i);
        }
    }
    let mut out = String::with_capacity(ir.len());
    for (line, keep) in lines.iter().zip(keep) {
        if keep {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// Every `%name` token a line mentions (labels included; they never name a value definition).
fn value_tokens(code: &str) -> impl Iterator<Item = &str> {
    code.match_indices('%').map(move |(at, _)| {
        let rest = &code[at + 1..];
        let end = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '.'))
            .unwrap_or(rest.len());
        &rest[..end]
    })
}

/// Whether the right-hand side of a value definition has no effect besides its result, so the
/// definition can go once nothing uses it. Loads count only when they read guest state or a
/// register/flag alloca, never guest memory (a guest load may fault) and never atomic or volatile
/// ones (the memory-model barriers of #719/#722 order those); calls, divisions and every other
/// instruction stay.
fn is_pure_definition(rhs: &str) -> bool {
    const PURE: [&str; 21] = [
        "add ",
        "sub ",
        "mul ",
        "and ",
        "or ",
        "xor ",
        "shl ",
        "lshr ",
        "ashr ",
        "icmp ",
        "select ",
        "trunc ",
        "zext ",
        "sext ",
        "getelementptr ",
        "inttoptr ",
        "ptrtoint ",
        "bitcast ",
        "phi ",
        "extractvalue ",
        "extractelement ",
    ];
    if PURE.iter().any(|op| rhs.starts_with(op)) {
        return true;
    }
    let Some(load) = rhs.strip_prefix("load ") else {
        return false;
    };
    if load.starts_with("atomic ") || load.starts_with("volatile ") {
        return false;
    }
    let Some((_, ptr)) = load.split_once(", ptr %") else {
        return false;
    };
    let ptr = ptr.split([',', ' ']).next().unwrap_or_default();
    ptr.starts_with("state.")
        || GPR_NAMES.contains(&ptr)
        || FLAGS.iter().any(|f| &f.slot()[1..] == ptr)
}

/// Removes pure value definitions ([`is_pure_definition`]) that nothing uses, transitively. The
/// state syncs around calls reload every register and flag the callee may write, and the load
/// forwarding plus [`drop_overwritten_slot_stores`] then leave the reloads of values the caller
/// overwrites before reading (typically the flags after a call) without a single use.
/// Appends [`TBAA_STATE_TAG`] to every load/store of a `%state.*` field and [`TBAA_GUEST_TAG`]
/// to every load/store through an `inttoptr` result (a guest pointer), see
/// [`super::TBAA_METADATA`]. With both classes tagged LLVM drops state stores of values that are
/// still in the state and keeps state values in registers across guest stores, which removes
/// about 5% of the wasm code. Every other access stays untagged and aliases everything.
fn tag_state_and_guest_accesses(ir: &str) -> String {
    let mut guest_ptrs: BTreeSet<&str> = BTreeSet::new();
    let mut out = String::with_capacity(ir.len() + ir.len() / 16);
    for line in ir.lines() {
        out.push_str(line);
        let code = line.trim_start();
        let rhs = match code.split_once(" = ") {
            Some((lhs, rhs)) if lhs.starts_with('%') && !lhs.contains(' ') => {
                if rhs.starts_with("inttoptr ") {
                    guest_ptrs.insert(&lhs[1..]);
                }
                rhs
            }
            _ => code,
        };
        if (rhs.starts_with("load ") || rhs.starts_with("store "))
            && !rhs.contains(" volatile ")
            && !rhs.contains("!tbaa")
            && let Some((_, after)) = rhs.split_once(", ptr %")
        {
            let ptr = &after[..after.find([' ', ',']).unwrap_or(after.len())];
            let tag = if ptr.starts_with("state.") {
                Some(TBAA_STATE_TAG)
            } else if guest_ptrs.contains(ptr) {
                Some(TBAA_GUEST_TAG)
            } else {
                None
            };
            if let Some(tag) = tag {
                out.push_str(", !tbaa ");
                out.push_str(tag);
            }
        }
        out.push('\n');
    }
    out
}

fn drop_unused_values(ir: &str) -> String {
    let lines: Vec<&str> = ir.lines().collect();
    let mut defs: BTreeMap<&str, usize> = BTreeMap::new();
    let mut uses: BTreeMap<&str, usize> = BTreeMap::new();
    for (i, line) in lines.iter().enumerate() {
        let code = line.trim();
        let (def, rhs) = match code.split_once(" = ") {
            Some((lhs, rhs)) if lhs.starts_with('%') && !lhs.contains(' ') => {
                (Some(&lhs[1..]), rhs)
            }
            _ => (None, code),
        };
        if let Some(def) = def
            && is_pure_definition(rhs)
        {
            defs.insert(def, i);
        }
        for token in value_tokens(rhs) {
            *uses.entry(token).or_default() += 1;
        }
    }
    let mut keep = vec![true; lines.len()];
    let mut work: Vec<&str> = defs
        .keys()
        .copied()
        .filter(|d| uses.get(d).copied().unwrap_or(0) == 0)
        .collect();
    while let Some(def) = work.pop() {
        let Some(&i) = defs.get(def) else { continue };
        if !keep[i] {
            continue;
        }
        keep[i] = false;
        let rhs = lines[i].trim().split_once(" = ").map_or("", |(_, r)| r);
        for token in value_tokens(rhs) {
            if let Some(n) = uses.get_mut(token) {
                *n -= 1;
                if *n == 0 && defs.contains_key(token) {
                    work.push(token);
                }
            }
        }
    }
    let mut out = String::with_capacity(ir.len());
    for (line, keep) in lines.iter().zip(keep) {
        if keep {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// Terminators whose last instruction is the control transfer itself (lowered from the edges).
fn ends_in_transfer(terminator: Terminator) -> bool {
    matches!(
        terminator,
        Terminator::Branch
            | Terminator::Jump
            | Terminator::IndirectJump
            | Terminator::Return
            | Terminator::Trap
    )
}

/// Instructions that may only appear as the last instruction of a block.
fn is_control_transfer(ins: &Instruction) -> bool {
    matches!(
        ins.flow_control(),
        FlowControl::UnconditionalBranch
            | FlowControl::IndirectBranch
            | FlowControl::ConditionalBranch
            | FlowControl::Return
    )
}

struct FunctionLifter<'a> {
    func: &'a cfg::Function,
    /// GPRs the function touches; only these get an alloca and are synced with the state.
    regs: [bool; 16],
    /// Arithmetic flags the function reads or writes; untouched flags stay in `%state`.
    flags: [bool; 6],
    /// Registers (GPR/XMM/YMMH) this function's own instructions may write; read-only registers
    /// skip redundant write-backs in `sync_out_parts`.
    written_state: StateMask,
    /// Arithmetic flags this function's own instructions may write; read-only flags skip
    /// redundant write-backs in `sync_out_parts`.
    written_flags: FlagMask,
    /// Vector registers the function touches (an `alloca <2 x i64>` per XMM register and per
    /// YMM upper half, synced like the GPRs) and the SIMD lifter's per-instruction state.
    simd: simd::Regs,
    /// Sites of direct calls into unmapped or non-executable memory (lifted as traps).
    faulting_calls: BTreeSet<u64>,
    body: String,
    /// Label of the LLVM basic block currently being emitted (used for phi predecessors).
    pub(super) current_block: String,
    tmp: u32,
    label: u32,
    /// Targets of edges that leave the function (become tail calls).
    exits: Vec<u64>,
    callees: Vec<u64>,
    support: BTreeSet<Support>,
    unsupported: Vec<UnsupportedInstruction>,
    /// Effective address (with segment base) of the current instruction's memory operand.
    ea: Option<String>,
    /// Guest -> host address translation (`memory_access.rs`).
    memory: MemoryModel,
    /// SSA values known to be stack-pointer-relative guest addresses -> proven displacement
    /// bound from `rsp` / `entry_rsp` (table models with `stack_flat` only).
    stack_values: BTreeMap<String, memory_access::StackDisp>,
    /// R58/R71: cached dynamic window translations (up to 4 entries) for same-address RMW /
    /// repeated accesses within the same basic block (cleared at block boundaries and
    /// calls/atomics/fences).
    window_load_cache: Vec<memory_access::CachedWindowAddr>,
    /// R58: active fast-path translation override for a coalesced same-base window burst.
    burst_override: Option<memory_access::BurstOverride>,
    /// The last plain guest load or store, for forwarding its value to a load of the same address
    /// later in the same LLVM block (`memory_access::LastAccess`).
    last_access: Option<memory_access::LastAccess>,
    /// SSA value holding `@__playgta6_guest_base`, loaded once in `entry:` when needed.
    pub(super) guest_base: Option<String>,
    /// Incoming edges `(kind, va, pred_block)` into this function's shared `cold_trap` sink.
    cold_traps: Vec<(ColdTrapKind, u64, String)>,
    /// True when the function has >1 `ret` blocks and shares a single `ret_epilog` sync-out.
    share_ret: bool,
    /// True once at least one `ret` branched to `ret_epilog`.
    has_shared_ret: bool,
    /// Calls followed by a resume-protocol check (see `unwind`).
    unwind_sites: u32,
    /// Pointers into `%state` the function uses (name -> `getelementptr` text). Each is defined
    /// once at the top of `entry:` (which dominates every block) instead of per use.
    state_ptrs: BTreeMap<String, String>,
    /// Flags each known direct callee may observe on entry and the state it may touch
    /// (`flag_liveness`); without it every call and tail jump stores every tracked flag and
    /// register and every call reloads them.
    flag_summaries: Option<&'a FlagSummaries>,
    /// Flags live at the exit of each basic block (`flag_liveness::block_live_out`), before the
    /// block's own instructions or `Branch` terminator are accounted for.
    block_live_out: BTreeMap<u64, FlagMask>,
    /// Per-instruction dead flags mask (0 outside modelled instructions):
    /// these flags are neither computed nor stored.
    dead_flags: FlagMask,
    /// Pending `FlagSource` captured right before a carry-consuming instruction (`sbb`/`adc`/
    /// `adcx`/`cmc`) invalidates `known.flag_source`, enabling direct carry-flag fusion.
    carry_source: Option<FlagSource>,
    /// R63a: bitmask of unread upper YMM halves (`!self.simd.ymmh[i]`) whose `%state.ymmh{i}`
    /// slot is already known to hold `zeroinitializer` in the current straight-line region.
    /// Cleared on every `start_block`, call, and non-zero 256-bit write to an unread `ymmh`.
    pub(super) ymmh_zeroed: u16,
    /// Load forwarding: the value each GPR / flag alloca holds, as far as it is known in the
    /// current LLVM basic block (the last value stored or loaded there). The allocas are private
    /// to the function and never escape, so a later load in the same block reads exactly that
    /// value; every store is still emitted. Cleared at each new block (see [`Self::start_block`]).
    known: Known,
    /// R71: per-GPR cross-block proven entry facts (`StackOffset`/`StackFrame`/`Const`) for the
    /// current block, cleared individually on writes/reloads so `load_gpr` and `lift_window_burst`
    /// preserve stack-pointer provenance across loops and diamond joins.
    pub(super) block_entry_facts: [GprFact; 16],
    /// [`Self::known`] at each branch emitted so far into a block of this function, as
    /// `(target, known)`: a block whose only predecessor is such an edge starts with it.
    edge_known: Vec<(u64, Known)>,
    /// Local value numbering: pure definitions already emitted in the current dominating region,
    /// by right-hand side. Cleared whenever a block starts or lifting rolls back.
    cse: BTreeMap<String, String>,
}

/// R71: intra-procedural forward dataflow lattice fact for one 64-bit GPR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GprFact {
    /// Unvisited / unreachable (identity element for join).
    Top,
    /// Provably holds exact 64-bit constant `u64` on all incoming paths.
    Const(u64),
    /// Provably equals `entry_rsp.wrapping_add(offset as u64)` with
    /// `offset.abs() <= STACK_FRAME_MAX_DISP`.
    StackOffset(i64),
    /// Provably within `STACK_FRAME_HALF` of `entry_rsp` across non-identical incoming paths.
    StackFrame,
    /// Unknown / varying across incoming paths.
    Bottom,
}

impl GprFact {
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Top, x) | (x, Self::Top) => x,
            (Self::Const(a), Self::Const(b)) if a == b => Self::Const(a),
            (Self::StackOffset(a), Self::StackOffset(b)) if a == b => Self::StackOffset(a),
            (Self::StackOffset(a), Self::StackOffset(b))
                if a.abs() <= memory_access::STACK_FRAME_HALF
                    && b.abs() <= memory_access::STACK_FRAME_HALF =>
            {
                Self::StackFrame
            }
            (Self::StackOffset(a), Self::StackFrame)
            | (Self::StackFrame, Self::StackOffset(a))
                if a.abs() <= memory_access::STACK_FRAME_HALF =>
            {
                Self::StackFrame
            }
            (Self::StackFrame, Self::StackFrame) => Self::StackFrame,
            _ => Self::Bottom,
        }
    }
}

/// High-level origin of the current block's live arithmetic flags, used to fuse
/// `cmp`/`test`/ALU/shift/bit-scan -> `jcc`/`setcc`/`cmovcc` into a single direct `icmp`
/// instead of materializing and combining individual flag slots.
#[derive(Clone, PartialEq, Eq)]
pub(super) enum FlagSource {
    /// `a - b` (`cmp`, `sub`, `neg`): all 10 relational CCs lower directly to `icmp <pred> a, b`;
    /// `e`/`ne`/`s`/`ns` use `diff` (when available) or `a` vs 0 (when `b == "0"`).
    Compare {
        a: String,
        b: String,
        diff: Option<String>,
        bits: u32,
    },
    /// Logical result (`test`, `and`, `or`, `xor`, `andn`): `CF = 0, OF = 0`, `ZF`/`SF` from
    /// `result`; all non-parity CCs lower directly to a constant or one `icmp <pred> result, 0`.
    LogicZero { result: String, bits: u32 },
    /// `add a, b` (`result = a + b`): `e`/`ne`/`s`/`ns` test `result` vs 0; `b`/`ae` test
    /// `result <u a` / `result >=u a`.
    Add {
        a: String,
        result: String,
        bits: u32,
    },
    /// Result (`blsi`, `blsr`, `bzhi`, `adc`, `sbb`, non-zero imm `shl`/`shr`/`sar`/`shld`/`shrd`):
    /// `e`/`ne`/`s`/`ns` lower directly to `icmp eq/ne/slt/sge result, 0`.
    ResultZeroSign { result: String, bits: u32 },
    /// `dec a` (`result = a - 1`, preserving `CF`): `e`/`ne`/`s`/`ns` test `result` vs 0;
    /// `l`/`ge`/`le`/`g` test `a` vs 1.
    Dec {
        a: String,
        result: String,
        bits: u32,
    },
    /// `inc a` (`result = a + 1`, preserving `CF`): `e`/`ne`/`s`/`ns` test `result` vs 0;
    /// `l`/`ge`/`le`/`g` test `a` vs -1.
    Inc {
        a: String,
        result: String,
        bits: u32,
    },
    /// `ZF = (val == 0)` (`popcnt`, `bextr`): `e`/`ne` test `val` vs 0.
    ZeroOnly { val: String, bits: u32 },
    /// Precomputed `ZF = is_zero` (`bsf`, `bsr`, constant-zero `bextr`): `e` uses `is_zero`, `ne` negates it.
    ZeroBit { is_zero: String },
    /// `tzcnt`/`lzcnt`: `CF = (src == 0)`, `ZF = (lsb/msb(src) == 1)`.
    BitCount {
        src: String,
        is_tzcnt: bool,
        bits: u32,
    },
    /// Precomputed `CF = cf` (`bt`/`bts`/`btr`/`btc` when CF is also live): `b` uses `cf`, `ae` negates it.
    CarryBit { cf: String },
    /// Lazy `CF = ((val & mask) != 0)` (`bt`/`bts`/`btr`/`btc` with dynamic bit mask).
    BitTest {
        val: String,
        mask: String,
        bits: u32,
    },
    /// Lazy `CF = ((val >> bit) & 1) != 0` (`bt`/`bts`/`btr`/`btc` with known bit index `bit < bits`).
    BitTestBit { val: String, bit: u32, bits: u32 },
    /// `(v)(u)comis{s,d}` (`ZF/PF/CF = 111` unordered, `000` >, `001` <, `100` ==; `OF=SF=AF=0`):
    /// every x86 condition code lowers directly to a single `fcmp` or boolean constant.
    FloatCompare {
        a: String,
        b: String,
        scalar_ty: &'static str,
    },
}

/// See [`FunctionLifter::known`].
#[derive(Clone, Default)]
struct Known {
    gprs: [Option<String>; 16],
    flags: [Option<String>; 6],
    flag_source: Option<FlagSource>,
    /// Tracks `(rdx64_ssa, rax64_ssa, bits)` immediately after `cdq` (32) or `cqo` (64) so a
    /// subsequent `idiv` lowers to native `sdiv`/`srem i32`/`i64` instead of widening/`sdiv128`.
    sign_ext_rdx: Option<(String, String, u32)>,
}

struct Checkpoint {
    body: usize,
    current_block: String,
    exits: usize,
    callees: usize,
    cold_traps: usize,
    has_shared_ret: bool,
    support: BTreeSet<Support>,
    known: Known,
    block_entry_facts: [GprFact; 16],
    edge_known: usize,
    window_load_cache: Vec<memory_access::CachedWindowAddr>,
    ymmh_zeroed: u16,
}

impl FunctionLifter<'_> {
    // ----- driver -------------------------------------------------------------------------------

    fn lift_blocks(&mut self) -> Result<(), LiftError> {
        let func = self.func;
        // Loop heads: targets of a backward (or self) edge in guest address order.
        let loop_heads: BTreeSet<u64> =
            func.blocks
                .iter()
                .flat_map(|(&start, block)| {
                    block.successors.iter().filter_map(move |edge| {
                        edge.intra_target().filter(|&target| target <= start)
                    })
                })
                .collect();
        // Incoming edges per block: CFG edges, the function entry and resume-protocol re-entry.
        let mut preds: BTreeMap<u64, usize> = BTreeMap::new();
        for block in func.blocks.values() {
            for target in block.successors.iter().filter_map(Edge::intra_target) {
                *preds.entry(target).or_default() += 1;
            }
        }
        for &target in std::iter::once(&func.entry).chain(&func.resume_targets) {
            *preds.entry(target).or_default() += 1;
        }
        let entry_gpr_facts =
            compute_block_entry_gpr_facts(func, self.flag_summaries, &self.faulting_calls);
        for (&start, block) in &func.blocks {
            let label = block_label(start);
            self.start_block(&label);
            // Extended basic block: with a single incoming edge, already lifted, the values the
            // predecessor's branch knew dominate this block and the allocas still hold them.
            if preds.get(&start) == Some(&1)
                && let Some(at) = self.edge_known.iter().position(|(t, _)| *t == start)
            {
                self.known = self.edge_known.swap_remove(at).1;
            }
            if let Some(&raw_facts) = entry_gpr_facts.get(&start) {
                let mut facts = [GprFact::Bottom; 16];
                for idx in 0..16 {
                    let fact = match raw_facts[idx] {
                        GprFact::Top => GprFact::Bottom,
                        other => other,
                    };
                    facts[idx] = fact;
                    match fact {
                        GprFact::Const(c) => {
                            self.known.gprs[idx] = Some(imm(c, 64));
                        }
                        GprFact::StackOffset(d) => {
                            if let Some(v) = self.known.gprs[idx].clone()
                                && !self.stack_values.contains_key(&v)
                            {
                                self.note_stack_value(&v, d);
                            }
                        }
                        GprFact::StackFrame => {
                            if let Some(v) = self.known.gprs[idx].clone()
                                && !self.stack_values.contains_key(&v)
                            {
                                self.note_stack_bounded(&v, memory_access::STACK_FRAME_HALF);
                            }
                        }
                        GprFact::Top | GprFact::Bottom => {}
                    }
                }
                self.block_entry_facts = facts;
            }
            if loop_heads.contains(&start) {
                self.emit(crate::lift::LOOP_BARRIER);
            }
            let (body, transfer) = match block.instructions.split_last() {
                Some((last, rest)) if ends_in_transfer(block.terminator) => (rest, Some(last)),
                _ => (&block.instructions[..], None),
            };
            let mut trapped = false;
            let bursts = self.find_window_bursts(body);
            let live_out = self
                .block_live_out
                .get(&start)
                .copied()
                .unwrap_or(ALL_FLAGS);
            let block_dead =
                flag_liveness::dead_flags_for_block(block, live_out, self.flag_summaries, &bursts);
            let mut burst_cursor = 0usize;
            let mut idx = 0usize;
            while idx < body.len() {
                let ins = &body[idx];
                if is_control_transfer(ins) {
                    return Err(LiftError::InstructionAfterTerminator {
                        block: start,
                        address: ins.ip(),
                    });
                }
                while burst_cursor < bursts.len() && bursts[burst_cursor].start < idx {
                    burst_cursor += 1;
                }
                if let Some(&burst) = bursts.get(burst_cursor)
                    && burst.start == idx
                {
                    let slice = &body[burst.start..=burst.end];
                    if let Some(bad) = slice.iter().find(|i| is_control_transfer(i)) {
                        return Err(LiftError::InstructionAfterTerminator {
                            block: start,
                            address: bad.ip(),
                        });
                    }
                    match self.lift_window_burst(
                        burst,
                        slice,
                        &block_dead[burst.start..=burst.end],
                    ) {
                        Ok(true) => {
                            idx = burst.end + 1;
                            burst_cursor += 1;
                            continue;
                        }
                        Ok(false) => {
                            burst_cursor += 1;
                        }
                        Err(InsnError::Cfg(err)) => return Err(err),
                        Err(InsnError::Unsupported(_)) => {
                            burst_cursor += 1;
                        }
                    }
                }
                let checkpoint = self.checkpoint();
                self.ea = None;
                self.dead_flags = block_dead[idx];
                let lifted = self.lift_instruction(ins);
                self.dead_flags = 0;
                match lifted {
                    Ok(()) => {}
                    Err(InsnError::Cfg(err)) => return Err(err),
                    Err(InsnError::Unsupported(reason)) => {
                        self.rollback(checkpoint);
                        self.trap_unsupported(ins, reason);
                        // The trap does not return; the rest of the block is dead.
                        trapped = true;
                        break;
                    }
                }
                idx += 1;
            }
            if !trapped {
                self.lift_terminator(block, transfer)?;
            }
        }
        Ok(())
    }

    /// Ends the LLVM block for `block` according to its terminator and typed successor edges.
    /// `transfer` is the block's last instruction when it is the control transfer itself.
    ///
    /// Terminator kinds added to `cfg` later degrade like an unresolved indirect jump (runtime
    /// dispatch) when the block ends in an indirect `jmp`, and to an unsupported trap otherwise.
    fn lift_terminator(
        &mut self,
        block: &cfg::BasicBlock,
        transfer: Option<&Instruction>,
    ) -> Result<(), LiftError> {
        let checkpoint = self.checkpoint();
        self.ea = None;
        let result = match (block.terminator, transfer) {
            (Terminator::Fallthrough | Terminator::Jump, _) => {
                self.goto(block, block.successors.first())
            }
            (Terminator::Branch, Some(ins)) => self.branch(block, ins),
            (Terminator::IndirectJump, Some(ins)) => self.indirect_jump(block, ins),
            (Terminator::Return, Some(ins)) => self.ret(ins),
            (Terminator::Trap, Some(ins)) => {
                self.trap(ins.ip());
                Ok(())
            }
            // The non-returning call itself was lifted with the block body. If it returns
            // anyway, execution would continue at the next instruction.
            (Terminator::NoReturnCall, _) => {
                self.trap(block.end);
                Ok(())
            }
            // Decoding stopped at `block.end` (the undecodable or unmapped address).
            (Terminator::Invalid, _) => {
                self.trap(block.end);
                Ok(())
            }
            // Fallback for terminator kinds added to `cfg` later (none exist today).
            #[allow(unreachable_patterns)]
            (_, Some(ins)) if ins.is_jmp_near_indirect() => self.indirect_jump(block, ins),
            (terminator, _) => unsupported(format!(
                "block terminator '{}' is not supported",
                terminator.as_str()
            )),
        };
        match result {
            Ok(()) => Ok(()),
            Err(InsnError::Cfg(err)) => Err(err),
            Err(InsnError::Unsupported(reason)) => {
                self.rollback(checkpoint);
                match transfer.or(block.instructions.last()) {
                    Some(ins) => self.trap_unsupported(ins, reason),
                    None => self.trap(block.end),
                }
                Ok(())
            }
        }
    }

    /// Resolves a successor edge to an LLVM label: a block of this function, or a tail-call exit.
    /// Edge kinds without a static target are reported as unsupported.
    fn successor(&mut self, block: &cfg::BasicBlock, edge: Option<&Edge>) -> R<String> {
        let Some(edge) = edge else {
            return unsupported("block has no successor edge");
        };
        if let Edge::TailCall(target) = *edge {
            self.exits.push(target);
            return Ok(exit_label(target));
        }
        match edge.intra_target() {
            Some(target) if self.func.blocks.contains_key(&target) => {
                self.edge_known.push((target, self.known.clone()));
                Ok(block_label(target))
            }
            Some(target) => Err(InsnError::Cfg(LiftError::UnknownSuccessor {
                block: block.start,
                target,
            })),
            None => unsupported(format!(
                "successor edge '{}' has no static target",
                edge.kind_str()
            )),
        }
    }

    fn goto(&mut self, block: &cfg::BasicBlock, edge: Option<&Edge>) -> R<()> {
        let target = self.successor(block, edge)?;
        self.emit(&format!("br label %{target}"));
        Ok(())
    }

    fn finish(mut self) -> LiftedFunction {
        let entry = self.func.entry;
        let symbol = function_symbol(entry);
        self.unwind_blocks();

        let mut exits: Vec<u64> = std::mem::take(&mut self.exits);
        exits.sort_unstable();
        exits.dedup();
        for &target in &exits {
            let exit = exit_label(target);
            self.start_block(&exit);
            let flags = self.flags_needed_by(target);
            self.sync_out_parts(ALL_STATE, flags);
            self.emit(&format!(
                "tail call void @{}(ptr %state)",
                function_symbol(target)
            ));
            self.emit("ret void");
            self.callees.push(target);
        }

        if self.has_shared_ret {
            self.start_block("ret_epilog");
            self.sync_out();
            self.emit("ret void");
        }

        let has_cold_traps = !self.cold_traps.is_empty();
        self.emit_cold_trap_sink();

        let body = std::mem::take(&mut self.body);
        let mut ir = String::new();
        let _ = writeln!(
            ir,
            "define void @{symbol}(ptr noalias noundef %state) unnamed_addr #0 {{"
        );
        ir.push_str("entry:\n");
        for (idx, name) in GPR_NAMES.iter().enumerate() {
            if self.regs[idx] {
                let _ = writeln!(ir, "  %{name} = alloca i64");
            }
        }
        for flag in FLAGS {
            if self.flags[flag as usize] {
                let _ = writeln!(ir, "  {} = alloca i1", flag.slot());
            }
        }
        self.xmm_allocas(&mut ir);
        // The emit helpers write to `self.body`, which is empty again here: build the entry
        // block's state loads there and splice them in front of the lifted blocks.
        self.known = Known::default();
        let (entry_dead_state, entry_dead_flags) = self.entry_overwritten_state();
        self.sync_in_parts(ALL_STATE & !entry_dead_state, ALL_FLAGS & !entry_dead_flags);
        self.unwind_entry();
        if self.support.contains(&Support::GuestBase)
            && let Some(base_name) = &self.guest_base
        {
            let line = format!(
                "{base_name} = load i64, ptr @{GUEST_BASE_SYMBOL}, align 8, !invariant.load !{{}}"
            );
            self.emit(&line);
        }
        // Every state pointer the function uses, including the entry loads' just above.
        for (name, gep) in &self.state_ptrs {
            let _ = writeln!(ir, "  {name} = {gep}");
        }
        ir.push_str(&std::mem::take(&mut self.body));
        // A loop can also span functions (tail jumps from one to the next and back). Once LLVM
        // inlines the callee, the self tail call becomes a loop whose head is this entry, so
        // the entry of a function that tail-jumps is a barrier too (see `LOOP_BARRIER`). Every
        // function on such a cycle has a tail jump; leaf functions get none, as a barrier there
        // would sit inside the caller's loop after inlining and block its optimisation.
        if !exits.is_empty() {
            let _ = writeln!(ir, "  {}", crate::lift::LOOP_BARRIER);
        }
        let _ = writeln!(ir, "  br label %{}", block_label(entry));
        ir.push_str(&body);
        ir.push_str("}\n");
        let ir = drop_overwritten_slot_stores(&ir);
        let ir = drop_unused_values(&ir);
        let ir = tag_state_and_guest_accesses(&ir);

        // R65b: tag tiny internal true-leaf functions (no calls, tail calls, unwind handlers,
        // traps, or unsupported ops, and `<= TINY_LEAF_MAX_INSNS` x86 instructions) for
        // `alwaysinline` when emitted with `internal` linkage in a dispatch shard.
        let total_insns: usize = self
            .func
            .blocks
            .values()
            .map(|b| b.instructions.len())
            .sum();
        let has_non_leaf_block = self.func.blocks.values().any(|b| {
            matches!(
                b.terminator,
                Terminator::IndirectJump
                    | Terminator::NoReturnCall
                    | Terminator::Trap
                    | Terminator::Invalid
            ) || !b.calls.is_empty()
                || b.instructions
                    .iter()
                    .any(|ins| ins.mnemonic() == Mnemonic::Call)
        });
        let tiny_leaf = total_insns > 0
            && total_insns <= TINY_LEAF_MAX_INSNS
            && self.callees.is_empty()
            && exits.is_empty()
            && self.unwind_sites == 0
            && self.func.resume_targets.is_empty()
            && !has_cold_traps
            && self.unsupported.is_empty()
            && !has_non_leaf_block;

        let mut unsupported = std::mem::take(&mut self.unsupported);
        unsupported.sort_by_key(|u| u.address);
        LiftedFunction {
            entry,
            symbol,
            ir,
            callees: self.callees.iter().copied().collect(),
            unsupported,
            support: std::mem::take(&mut self.support),
            tiny_leaf,
        }
    }

    /// Emits the function's shared cold trap sink (`cold_trap:`), deduplicating `sync_out()`
    /// across all trap, unimplemented, divide-error, misaligned-`cmpxchg16b` and `ldmxcsr` sites.
    fn emit_cold_trap_sink(&mut self) {
        let traps = std::mem::take(&mut self.cold_traps);
        if traps.is_empty() {
            return;
        }
        self.start_block("cold_trap");
        let first_va = traps[0].1;
        let va = if traps.iter().all(|(_, va, _)| *va == first_va) {
            imm(first_va, 64)
        } else {
            let inputs: Vec<String> = traps
                .iter()
                .map(|(_, va, pred)| format!("[ {}, %{pred} ]", imm(*va, 64)))
                .collect();
            self.val(&format!("phi i64 {}", inputs.join(", ")))
        };
        let first_kind = traps[0].0;
        if traps.iter().all(|(kind, _, _)| *kind == first_kind) {
            self.sync_out();
            self.emit(&format!(
                "call void @{}(ptr %state, i64 {va})",
                first_kind.helper()
            ));
            self.emit("unreachable");
            return;
        }
        let kind_inputs: Vec<String> = traps
            .iter()
            .map(|(kind, _, pred)| format!("[ {}, %{pred} ]", kind.tag()))
            .collect();
        let kind_val = self.val(&format!("phi i8 {}", kind_inputs.join(", ")));
        self.sync_out();
        let mut kinds = Vec::new();
        for (kind, _, _) in &traps {
            if !kinds.contains(kind) {
                kinds.push(*kind);
            }
        }
        let default_kind = kinds[0];
        let mut arms = String::new();
        for &kind in &kinds[1..] {
            let _ = write!(arms, " i8 {}, label %cold_trap_{}", kind.tag(), kind.tag());
        }
        self.emit(&format!(
            "switch i8 {kind_val}, label %cold_trap_{} [{arms} ]",
            default_kind.tag()
        ));
        for kind in kinds {
            let label = format!("cold_trap_{}", kind.tag());
            self.start_block(&label);
            self.emit(&format!(
                "call void @{}(ptr %state, i64 {va})",
                kind.helper()
            ));
            self.emit("unreachable");
        }
    }

    fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            body: self.body.len(),
            current_block: self.current_block.clone(),
            exits: self.exits.len(),
            callees: self.callees.len(),
            cold_traps: self.cold_traps.len(),
            has_shared_ret: self.has_shared_ret,
            support: self.support.clone(),
            known: self.known.clone(),
            block_entry_facts: self.block_entry_facts,
            edge_known: self.edge_known.len(),
            window_load_cache: self.window_load_cache.clone(),
            ymmh_zeroed: self.ymmh_zeroed,
        }
    }

    fn rollback(&mut self, checkpoint: Checkpoint) {
        self.body.truncate(checkpoint.body);
        self.current_block = checkpoint.current_block;
        self.exits.truncate(checkpoint.exits);
        self.callees.truncate(checkpoint.callees);
        self.cold_traps.truncate(checkpoint.cold_traps);
        self.has_shared_ret = checkpoint.has_shared_ret;
        self.support = checkpoint.support;
        self.known = checkpoint.known;
        self.block_entry_facts = checkpoint.block_entry_facts;
        self.cse.clear();
        self.edge_known.truncate(checkpoint.edge_known);
        self.window_load_cache = checkpoint.window_load_cache;
        self.ymmh_zeroed = checkpoint.ymmh_zeroed;
        self.simd.clear_mem32();
    }

    fn trap_unsupported(&mut self, ins: &Instruction, reason: String) {
        let mut text = String::new();
        IntelFormatter::new().format(ins, &mut text);
        self.unsupported.push(UnsupportedInstruction {
            address: ins.ip(),
            mnemonic: format!("{:?}", ins.mnemonic()).to_lowercase(),
            instruction: text,
            reason,
        });
        self.cold_traps.push((
            ColdTrapKind::Unimplemented,
            ins.ip(),
            self.current_block.clone(),
        ));
        self.emit("br label %cold_trap");
    }

    /// Ends the current LLVM block with a branch to the function's shared cold trap sink.
    fn trap(&mut self, va: u64) {
        self.cold_traps
            .push((ColdTrapKind::Trap, va, self.current_block.clone()));
        self.emit("br label %cold_trap");
    }

    /// Branches to the function's shared cold trap sink when `fault` is true (`!prof UNLIKELY`)
    /// and continues lifting in a fresh continuation block.
    pub(super) fn cold_trap_branch(&mut self, fault: &str, kind: ColdTrapKind, va: u64) {
        let ok = self.new_label("cont");
        self.cold_traps.push((kind, va, self.current_block.clone()));
        self.emit(&format!(
            "br i1 {fault}, label %cold_trap, label %{ok}, {}",
            memory_access::UNLIKELY
        ));
        // `ok` has this block as its only predecessor: the known values stay valid there.
        let known = std::mem::take(&mut self.known);
        let facts = self.block_entry_facts;
        let cse = std::mem::take(&mut self.cse);
        self.start_block(&ok);
        self.known = known;
        self.block_entry_facts = facts;
        self.cse = cse;
    }

    // ----- IR emission primitives ---------------------------------------------------------------

    fn emit(&mut self, line: &str) {
        self.body.push_str("  ");
        self.body.push_str(line);
        self.body.push('\n');
    }

    /// A new `%tN` name without emitting anything.
    pub(super) fn fresh_name(&mut self) -> String {
        let name = format!("%t{}", self.tmp);
        self.tmp += 1;
        name
    }

    /// Emits `%tN = <rhs>` and returns `%tN`.
    fn val(&mut self, rhs: &str) -> String {
        const CSE_OPS: [&str; 10] = [
            "add ", "sub ", "and ", "or ", "xor ", "shl ", "lshr ", "trunc ", "zext ", "sext ",
        ];
        let cacheable = CSE_OPS.iter().any(|op| rhs.starts_with(op));
        if cacheable && let Some(prev) = self.cse.get(rhs) {
            return prev.clone();
        }
        let name = self.fresh_name();
        self.emit(&format!("{name} = {rhs}"));
        if cacheable {
            self.cse.insert(rhs.to_string(), name.clone());
        }
        name
    }

    fn new_label(&mut self, prefix: &str) -> String {
        let name = format!("{prefix}{}", self.label);
        self.label += 1;
        name
    }

    fn start_block(&mut self, label: &str) {
        self.cse.clear();
        self.known = Known::default();
        self.block_entry_facts = [GprFact::Bottom; 16];
        self.window_load_cache.clear();
        self.last_access = None;
        self.ymmh_zeroed = 0;
        self.current_block.clear();
        self.current_block.push_str(label);
        let _ = writeln!(self.body, "{label}:");
    }

    /// Starts an internal CFG block (e.g. a memory-access slow path or join) without discarding
    /// `self.known` or `self.window_load_cache`, which stay valid across single-entry regions that
    /// never write GPR/flag allocas or invoke runtime remapping helpers.
    pub(super) fn start_internal_block(&mut self, label: &str) {
        self.cse.clear();
        self.current_block.clear();
        self.current_block.push_str(label);
        let _ = writeln!(self.body, "{label}:");
    }

    pub(super) fn clear_window_cache(&mut self) {
        self.window_load_cache.clear();
    }

    pub(super) fn known_gpr(&self, idx: usize) -> Option<&str> {
        self.known.gprs[idx].as_deref()
    }

    /// True when GPR `idx` is currently provably stack-pointer-relative (either via its active
    /// SSA value in `stack_values`, or via cross-block `block_entry_facts` if not yet loaded or
    /// modified in this block).
    pub(super) fn is_gpr_stack_proven(&self, idx: usize) -> bool {
        if idx == RSP {
            return true;
        }
        match self.known.gprs[idx].as_deref() {
            Some(v) => self.stack_values.contains_key(v),
            None => matches!(
                self.block_entry_facts[idx],
                GprFact::StackOffset(_) | GprFact::StackFrame
            ),
        }
    }

    /// Preserves `block_entry_facts` across a window burst join for GPRs untouched by both arms.
    pub(super) fn restore_unchanged_block_entry_facts(
        &mut self,
        entry: &[GprFact; 16],
        a: &[GprFact; 16],
        b: &[GprFact; 16],
    ) {
        for i in 0..16 {
            if a[i] == entry[i] && b[i] == entry[i] {
                self.block_entry_facts[i] = entry[i];
            }
        }
    }

    /// Restores entries from `entry` that neither `a` nor `b` modified (the entry block strictly
    /// dominates the join block, so unchanged SSA values remain valid and still match their allocas).
    pub(super) fn restore_unchanged_known(&mut self, entry: &Known, a: &Known, b: &Known) {
        for i in 0..16 {
            if entry.gprs[i].is_some()
                && a.gprs[i] == entry.gprs[i]
                && b.gprs[i] == entry.gprs[i]
            {
                self.known.gprs[i] = entry.gprs[i].clone();
            }
        }
        for i in 0..6 {
            if entry.flags[i].is_some()
                && a.flags[i] == entry.flags[i]
                && b.flags[i] == entry.flags[i]
            {
                self.known.flags[i] = entry.flags[i].clone();
            }
        }
        if entry.flag_source.is_some()
            && a.flag_source == entry.flag_source
            && b.flag_source == entry.flag_source
        {
            self.known.flag_source = entry.flag_source.clone();
        }
    }

    // ----- guest state ---------------------------------------------------------------------------

    /// Pointer to `%state` field `field` (element `idx` of it when given), named after the slot
    /// (`%state.gpr3`, `%state.flag0`, `%state.xmm5`, `%state.f7`, ...). The `getelementptr` is
    /// emitted once at the top of `entry:` by [`Self::finish`], so every use shares it.
    pub(super) fn state_ptr(&mut self, field: u32, idx: Option<usize>) -> String {
        let name = match (field, idx) {
            (0, Some(i)) => format!("%state.gpr{i}"),
            (2, Some(i)) => format!("%state.flag{i}"),
            (5, Some(i)) => format!("%state.xmm{i}"),
            (6, Some(i)) => format!("%state.ymmh{i}"),
            (f, Some(i)) => format!("%state.f{f}.{i}"),
            (f, None) => format!("%state.f{f}"),
        };
        self.state_ptrs.entry(name.clone()).or_insert_with(|| {
            let elem = idx.map_or(String::new(), |i| format!(", i64 {i}"));
            format!(
                "getelementptr inbounds {GUEST_STATE_TYPE}, ptr %state, i64 0, i32 {field}{elem}"
            )
        });
        name
    }

    fn state_gpr_ptr(&mut self, idx: usize) -> String {
        self.state_ptr(0, Some(idx))
    }

    fn state_flag_ptr(&mut self, flag: Flag) -> String {
        self.state_ptr(2, Some(flag as usize))
    }

    /// R64a + R65a: `(dead_state, dead_flags)` that the function's entry block unconditionally
    /// overwrites before any read, call, branch, or potentially trapping/unmodelled instruction.
    /// Skipping their entry-block `%state.gprN` / `%state.flagN` / `%state.xmmN` loads is 100%
    /// sound because every path from `entry:` writes the full slot before any read or `sync_out()`
    /// (and `rsp` is never elided).
    fn entry_overwritten_state(&self) -> (StateMask, FlagMask) {
        let Some(entry_block) = self.func.blocks.get(&self.func.entry) else {
            return (0, 0);
        };
        let body = match entry_block.instructions.split_last() {
            Some((_, rest)) if ends_in_transfer(entry_block.terminator) => rest,
            _ => &entry_block.instructions[..],
        };
        let mut factory = InstructionInfoFactory::new();
        let mut gpr_read_before_write: u16 = 1u16 << RSP;
        let mut gpr_written_first: u16 = 0;
        let mut flag_read_before_write: FlagMask = 0;
        let mut flag_written_first: FlagMask = 0;
        let mut xmm_read_before_write: u16 = 0;
        let mut xmm_written_first: u16 = 0;
        for ins in body {
            if is_control_transfer(ins) || self.unsupported.iter().any(|u| u.address == ins.ip()) {
                break;
            }
            let Some((flag_reads, flag_writes)) = flag_liveness::flag_effect(ins) else {
                break;
            };
            flag_read_before_write |= flag_reads & !flag_written_first;
            flag_written_first |= flag_writes & !flag_read_before_write;

            let info = factory.info(ins);
            let (gpr_reads, gpr_full_writes) = gpr_rw_masks(ins, info);
            gpr_read_before_write |= gpr_reads & !gpr_written_first;
            gpr_written_first |= gpr_full_writes & !gpr_read_before_write;

            if simd::is_simd(ins) {
                let (xmm_reads, xmm_full_writes) = simd::xmm_rw_masks(ins, info);
                xmm_read_before_write |= xmm_reads & !xmm_written_first;
                xmm_written_first |= xmm_full_writes & !xmm_read_before_write;
            }
        }
        let mut state_mask: StateMask = u64::from(gpr_written_first & !(1u16 << RSP));
        for idx in 0..16 {
            if xmm_written_first & (1u16 << idx) != 0 {
                state_mask |= flag_liveness::xmm_state_bit(idx);
            }
        }
        (state_mask, flag_written_first)
    }

    /// Loads all tracked registers and flags from `%state` into their allocas.
    fn sync_in(&mut self) {
        self.sync_in_parts(ALL_STATE, ALL_FLAGS);
    }

    /// [`Self::sync_in`] loading only the GPRs, `xmm` and `ymmh` registers in `state` and the
    /// flags in `flags`.
    fn sync_in_parts(&mut self, state: StateMask, flags: FlagMask) {
        for (idx, name) in GPR_NAMES.iter().enumerate() {
            if self.regs[idx] && state & (1u64 << idx) != 0 {
                let ptr = self.state_gpr_ptr(idx);
                let v = self.val(&format!("load i64, ptr {ptr}"));
                self.emit(&format!("store i64 {v}, ptr %{name}"));
                self.known.gprs[idx] = Some(v);
                self.block_entry_facts[idx] = GprFact::Bottom;
            }
        }
        self.known.flag_source = None;
        for flag in FLAGS {
            if self.flags[flag as usize] && flags & (1 << flag as usize) != 0 {
                let ptr = self.state_flag_ptr(flag);
                let byte = self.val(&format!("load i8, ptr {ptr}, !range !{{i8 0, i8 2}}"));
                let bit = self.val(&format!("trunc i8 {byte} to i1"));
                self.emit(&format!("store i1 {bit}, ptr {}", flag.slot()));
                self.known.flags[flag as usize] = Some(bit);
            } else {
                self.known.flags[flag as usize] = None;
            }
        }
        if state & (ALL_XMM_STATE | ALL_YMMH_STATE) != 0 {
            self.sync_xmm_in(state);
        }
    }

    /// Stores all tracked registers and flags from their allocas back into `%state`.
    fn sync_out(&mut self) {
        self.sync_out_parts(ALL_STATE, ALL_FLAGS);
    }

    /// Flags a tail jump to `target` has to store (see `flag_liveness`).
    fn flags_needed_by(&self, target: u64) -> FlagMask {
        self.flag_summaries
            .map_or(ALL_FLAGS, |summaries| summaries.needed_by(target))
    }

    /// Flags a direct call to `target` has to store before calling (`touched_flags & live_in`).
    fn flags_sync_out_by(&self, target: u64) -> FlagMask {
        self.flag_summaries
            .map_or(ALL_FLAGS, |summaries| summaries.sync_out_flags_by(target))
    }

    /// Flags a direct call to `target` may overwrite (`written_flags`).
    fn flags_written_by(&self, target: u64) -> FlagMask {
        self.flag_summaries
            .map_or(ALL_FLAGS, |summaries| summaries.written_flags_by(target))
    }

    /// Guest state a direct call to `target` must store before calling (`touched & live_in_state`).
    fn state_sync_out_by(&self, target: u64) -> StateMask {
        self.flag_summaries
            .map_or(ALL_STATE, |summaries| summaries.sync_out_state_by(target))
    }

    /// Guest state a direct call to `target` must reload after returning (`written`).
    fn state_written_by(&self, target: u64) -> StateMask {
        self.flag_summaries
            .map_or(ALL_STATE, |summaries| summaries.written_by(target))
    }

    /// [`Self::sync_out`] storing only the tracked GPRs, `xmm` and `ymmh` registers in `state`
    /// and the tracked flags in `flags` that this function may actually have modified. Registers
    /// and flags the function only reads keep their exact values in `%state` throughout the call.
    fn sync_out_parts(&mut self, state: StateMask, flags: FlagMask) {
        let state = state & self.written_state;
        let flags = flags & self.written_flags;
        for idx in 0..16 {
            if self.regs[idx] && state & (1u64 << idx) != 0 {
                let Ok(v) = self.load_gpr(idx) else {
                    unreachable!("register {idx} is tracked")
                };
                let ptr = self.state_gpr_ptr(idx);
                self.emit(&format!("store i64 {v}, ptr {ptr}"));
            }
        }
        for flag in FLAGS {
            if self.flags[flag as usize] && flags & (1 << flag as usize) != 0 {
                let bit = self.load_flag(flag);
                let byte = match bit.as_str() {
                    "true" => "1".to_string(),
                    "false" => "0".to_string(),
                    _ => self.val(&format!("zext i1 {bit} to i8")),
                };
                let ptr = self.state_flag_ptr(flag);
                self.emit(&format!("store i8 {byte}, ptr {ptr}"));
            }
        }
        if state & (ALL_XMM_STATE | ALL_YMMH_STATE) != 0 {
            self.sync_xmm_out(state);
        }
    }

    fn segment_base(&mut self, field: u32) -> String {
        let ptr = self.state_ptr(field, None);
        self.val(&format!("load i64, ptr {ptr}"))
    }

    // ----- registers -----------------------------------------------------------------------------

    fn gpr_slot(&self, idx: usize) -> R<String> {
        if idx < 16 && self.regs[idx] {
            Ok(format!("%{}", GPR_NAMES[idx]))
        } else {
            unsupported(format!("internal: register index {idx} is not tracked"))
        }
    }

    fn load_gpr(&mut self, idx: usize) -> R<String> {
        let slot = self.gpr_slot(idx)?;
        let v = match &self.known.gprs[idx] {
            Some(v) => v.clone(),
            None => {
                let v = self.val(&format!("load i64, ptr {slot}"));
                self.known.gprs[idx] = Some(v.clone());
                if idx != RSP {
                    match self.block_entry_facts[idx] {
                        GprFact::StackOffset(d) => self.note_stack_value(&v, d),
                        GprFact::StackFrame => {
                            self.note_stack_bounded(&v, memory_access::STACK_FRAME_HALF)
                        }
                        GprFact::Top | GprFact::Const(_) | GprFact::Bottom => {}
                    }
                }
                v
            }
        };
        if idx == RSP {
            self.note_stack_value(&v, 0);
        }
        Ok(v)
    }

    fn store_gpr(&mut self, idx: usize, value: &str) -> R<()> {
        let slot = self.gpr_slot(idx)?;
        self.emit(&format!("store i64 {value}, ptr {slot}"));
        self.known.gprs[idx] = Some(value.to_string());
        self.block_entry_facts[idx] = GprFact::Bottom;
        if idx == integer::RAX || idx == integer::RDX {
            self.known.sign_ext_rdx = None;
        }
        Ok(())
    }

    /// Reads a GPR (any width) and returns `(value, bits)`.
    fn read_reg(&mut self, reg: Register) -> R<(String, u32)> {
        match reg_ref(reg)? {
            RegRef::Low(idx, 64) => Ok((self.load_gpr(idx)?, 64)),
            RegRef::Low(idx, bits) => {
                let full = self.load_gpr(idx)?;
                if let Some(c) = memory_access::const_addr(&full) {
                    return Ok((imm(c, bits), bits));
                }
                Ok((self.val(&format!("trunc i64 {full} to i{bits}")), bits))
            }
            RegRef::High8(idx) => {
                let full = self.load_gpr(idx)?;
                if let Some(c) = memory_access::const_addr(&full) {
                    return Ok((imm(c >> 8, 8), 8));
                }
                let shifted = self.val(&format!("lshr i64 {full}, 8"));
                Ok((self.val(&format!("trunc i64 {shifted} to i8")), 8))
            }
        }
    }

    /// Writes a GPR with x86-64 semantics (32-bit writes zero-extend, 8/16-bit writes merge).
    fn write_reg(&mut self, reg: Register, value: &str, bits: u32) -> R<()> {
        match reg_ref(reg)? {
            RegRef::Low(idx, reg_bits) if reg_bits != bits => unsupported(format!(
                "internal: writing {bits}-bit value to {reg_bits}-bit register {reg:?} ({idx})"
            )),
            RegRef::Low(idx, 64) => self.store_gpr(idx, value),
            RegRef::Low(idx, 32) => {
                // A constant 32-bit result (e.g. `xor r9d, r9d`) zero-extends at lift time.
                let wide = match value.parse::<i64>() {
                    Ok(c) => (c as u32 as u64).to_string(),
                    Err(_) => self.val(&format!("zext i32 {value} to i64")),
                };
                self.store_gpr(idx, &wide)
            }
            RegRef::Low(idx, bits) => self.merge_gpr(idx, value, bits, 0),
            RegRef::High8(idx) if bits == 8 => self.merge_gpr(idx, value, 8, 8),
            RegRef::High8(_) => unsupported("internal: high-byte register width mismatch"),
        }
    }

    /// Replaces bits `shift..shift+bits` of a 64-bit register, preserving the others.
    fn merge_gpr(&mut self, idx: usize, value: &str, bits: u32, shift: u32) -> R<()> {
        let old = self.load_gpr(idx)?;
        let keep = !(mask(bits) << shift);
        if let Some(old_c) = memory_access::const_addr(&old) {
            let kept_c = old_c & keep;
            if let Ok(c) = value.parse::<i64>() {
                let placed = ((c as u64) & mask(bits)) << shift;
                return self.store_gpr(idx, &imm(kept_c | placed, 64));
            }
            let wide = self.val(&format!("zext i{bits} {value} to i64"));
            let placed = if shift == 0 {
                wide
            } else {
                self.val(&format!("shl i64 {wide}, {shift}"))
            };
            if kept_c == 0 {
                return self.store_gpr(idx, &placed);
            }
            let merged = self.val(&format!("or i64 {placed}, {}", imm(kept_c, 64)));
            return self.store_gpr(idx, &merged);
        }
        let kept = self.val(&format!("and i64 {old}, {}", imm(keep, 64)));
        if let Ok(c) = value.parse::<i64>() {
            let placed = ((c as u64) & mask(bits)) << shift;
            if placed == 0 {
                return self.store_gpr(idx, &kept);
            }
            let merged = self.val(&format!("or i64 {kept}, {}", imm(placed, 64)));
            return self.store_gpr(idx, &merged);
        }
        let wide = self.val(&format!("zext i{bits} {value} to i64"));
        let placed = if shift == 0 {
            wide
        } else {
            self.val(&format!("shl i64 {wide}, {shift}"))
        };
        let merged = self.val(&format!("or i64 {kept}, {placed}"));
        self.store_gpr(idx, &merged)
    }

    // ----- memory --------------------------------------------------------------------------------

    /// Computes the memory operand's address as an `i64`, optionally adding the `fs`/`gs` base.
    fn address(&mut self, ins: &Instruction, with_segment: bool) -> R<String> {
        let base = ins.memory_base();
        let index = ins.memory_index();
        // For RIP-relative operands iced already returns the absolute target here.
        let disp = ins.memory_displacement64();
        let mut addr32 = false;
        let mut acc: Option<String> = None;

        if base == Register::EIP {
            return unsupported("EIP-relative addressing");
        }
        if base != Register::None && base != Register::RIP {
            if !(base.is_gpr64() || base.is_gpr32()) {
                return unsupported(format!("memory base register {base:?}"));
            }
            addr32 |= base.is_gpr32();
            acc = Some(self.read_addr_reg(base)?);
        }
        if index != Register::None {
            if !(index.is_gpr64() || index.is_gpr32()) {
                return unsupported(format!("memory index register {index:?}"));
            }
            addr32 |= index.is_gpr32();
            let mut scaled = self.read_addr_reg(index)?;
            let scale = ins.memory_index_scale();
            if scale > 1 {
                if let Some(c) = memory_access::const_addr(&scaled) {
                    scaled = imm(c.wrapping_shl(scale.trailing_zeros()), 64);
                } else {
                    scaled = self.val(&format!("shl i64 {scaled}, {}", scale.trailing_zeros()));
                }
            }
            acc = Some(match acc {
                Some(sum) => {
                    if let (Some(a), Some(b)) = (
                        memory_access::const_addr(&sum),
                        memory_access::const_addr(&scaled),
                    ) {
                        imm(a.wrapping_add(b), 64)
                    } else {
                        self.val(&format!("add i64 {sum}, {scaled}"))
                    }
                }
                None => scaled,
            });
        }
        let acc_base = if index == Register::None {
            acc.clone()
        } else {
            None
        };
        let mut addr = match acc {
            Some(sum) if disp != 0 => {
                if let Some(base_va) = memory_access::const_addr(&sum) {
                    imm(base_va.wrapping_add(disp), 64)
                } else {
                    self.val(&format!("add i64 {sum}, {}", imm(disp, 64)))
                }
            }
            Some(sum) => sum,
            None => imm(if addr32 { disp & mask(32) } else { disp }, 64),
        };
        let has_fs_gs = with_segment
            && matches!(ins.memory_segment(), Register::FS | Register::GS);
        if addr32 {
            if let Some(va) = memory_access::const_addr(&addr) {
                addr = imm(va & mask(32), 64);
            } else {
                addr = self.val(&format!("and i64 {addr}, 4294967295"));
            }
        } else if base == Register::RSP && index == Register::None {
            self.note_stack_value(&addr, disp as i64);
        } else if index == Register::None
            && let Some(base_val) = &acc_base
        {
            self.propagate_stack_offset(base_val, &addr, disp as i64);
        }
        if !addr32
            && !has_fs_gs
            && base.is_gpr64()
            && (index == Register::None || index.is_gpr64())
            && let Some(burst) = &mut self.burst_override
        {
            let key = memory_access::BurstKey {
                base: base.number(),
                index: (index != Register::None)
                    .then(|| (index.number(), ins.memory_index_scale())),
            };
            if let Some(slot) = burst.bases.iter_mut().find(|b| b.key == key) {
                slot.addrs.insert(addr.clone(), disp as i64);
            }
        }
        if with_segment {
            let field = match ins.memory_segment() {
                Register::FS => Some(3),
                Register::GS => Some(4),
                _ => None,
            };
            if let Some(field) = field {
                let seg = self.segment_base(field);
                addr = self.val(&format!("add i64 {addr}, {seg}"));
            }
        }
        Ok(addr)
    }

    /// Computes `addr + offset` (64-bit wrapping), folding at lift time when `addr` is a constant
    /// literal and preserving bounded stack-pointer provenance and burst-override tracking.
    pub(super) fn offset_addr(&mut self, addr: &str, offset: u64) -> String {
        if offset == 0 {
            return addr.to_string();
        }
        if let Some(va) = memory_access::const_addr(addr) {
            return imm(va.wrapping_add(offset), 64);
        }
        let out = self.val(&format!("add i64 {addr}, {}", imm(offset, 64)));
        self.propagate_stack_offset(addr, &out, offset as i64);
        if let Some(burst) = &mut self.burst_override {
            for slot in &mut burst.bases {
                if let Some(&base_disp) = slot.addrs.get(addr)
                    && let Some(new_disp) = base_disp.checked_add(offset as i64)
                {
                    slot.addrs.insert(out.clone(), new_disp);
                }
            }
        }
        out
    }

    fn read_addr_reg(&mut self, reg: Register) -> R<String> {
        match reg_ref(reg)? {
            RegRef::Low(idx, 64) => self.load_gpr(idx),
            RegRef::Low(idx, 32) => {
                let full = self.load_gpr(idx)?;
                if let Some(c) = memory_access::const_addr(&full) {
                    return Ok(imm(c & mask(32), 64));
                }
                Ok(self.val(&format!("and i64 {full}, 4294967295")))
            }
            _ => {
                let (v, bits) = self.read_reg(reg)?;
                Ok(self.val(&format!("zext i{bits} {v} to i64")))
            }
        }
    }

    /// Effective (segmented) address of the current instruction's memory operand, computed once
    /// per instruction so read-modify-write and `xchg` see a consistent address.
    fn mem_ea(&mut self, ins: &Instruction) -> R<String> {
        if let Some(ea) = &self.ea {
            return Ok(ea.clone());
        }
        let ea = self.address(ins, true)?;
        self.ea = Some(ea.clone());
        Ok(ea)
    }

    // Guest addresses become host pointers only in `memory_access.rs` (`guest_ptr`,
    // `guest_load`, `guest_store`, `guest_memmove`, `guest_memset`).

    fn load_mem(&mut self, addr: &str, bits: u32) -> String {
        self.guest_load(addr, &format!("i{bits}"), u64::from(bits / 8))
    }

    fn store_mem(&mut self, addr: &str, value: &str, bits: u32) {
        self.guest_store(addr, &format!("i{bits}"), value, u64::from(bits / 8));
    }

    fn push(&mut self, value: &str, bits: u32) -> R<()> {
        let rsp = self.load_gpr(RSP)?;
        let new_rsp = self.val(&format!("sub i64 {rsp}, {}", bits / 8));
        self.note_stack_value(&new_rsp, -i64::from(bits / 8));
        self.store_mem(&new_rsp, value, bits);
        self.store_gpr(RSP, &new_rsp)
    }

    /// R58/R63/R64/R67: scans a basic block body for non-overlapping bursts of consecutive plain
    /// or indexed memory accesses sharing up to [`memory_access::MAX_BURST_BASES`] (3) 64-bit
    /// addressing bases (`[base + disp]` or `[base + idx*scale + disp]`) under
    /// `WindowCheck::Branch` (`window`, `window-stack`, `window-runs`), so the entire burst pays
    /// one window range/page check per active base instead of one per access.
    fn find_window_bursts(&self, body: &[Instruction]) -> Vec<memory_access::WindowBurst> {
        use crate::lift::memory::WindowCheck;
        let MemoryModel::Windowed(w) = self.memory else {
            return Vec::new();
        };
        if w.check != WindowCheck::Branch || body.len() < 2 {
            return Vec::new();
        }
        const MAX_BURST_INSNS: usize = 64;
        #[derive(Clone, Copy)]
        struct Active {
            start: usize,
            end: usize,
            min_d: i64,
            max_end: i64,
            count: usize,
            has_write: bool,
            has_simd_mem: bool,
        }
        #[derive(Clone, Copy)]
        struct SingleCandidate {
            start: usize,
            end: usize,
            base: memory_access::BurstBase,
        }
        let span_ok = |min_d: i64, max_end: i64| -> bool {
            max_end
                .checked_sub(min_d)
                .and_then(|s| u64::try_from(s).ok())
                .is_some_and(|s| s > 0 && s <= memory_access::BURST_MAX_SPAN)
        };
        let meets_threshold = |count: usize, has_write: bool, has_simd_mem: bool| -> bool {
            count >= 3 || (count >= 2 && (has_write || has_simd_mem))
        };

        let mut barriers = vec![false; body.len()];
        let mut written = vec![0u16; body.len()];
        let mut accesses = vec![None; body.len()];
        let mut active: BTreeMap<memory_access::BurstKey, Active> = BTreeMap::new();
        let mut candidates: Vec<SingleCandidate> = Vec::new();
        let mut factory = InstructionInfoFactory::new();

        let emit_candidate =
            |key: memory_access::BurstKey, a: Active, out: &mut Vec<SingleCandidate>| {
                if a.end > a.start
                    && a.end - a.start < MAX_BURST_INSNS
                    && meets_threshold(a.count, a.has_write, a.has_simd_mem)
                    && span_ok(a.min_d, a.max_end)
                {
                    out.push(SingleCandidate {
                        start: a.start,
                        end: a.end,
                        base: memory_access::BurstBase {
                            key,
                            min_d: a.min_d,
                            max_end: a.max_end,
                            has_write: a.has_write,
                        },
                    });
                }
            };
        let close_all = |active: &mut BTreeMap<memory_access::BurstKey, Active>,
                         out: &mut Vec<SingleCandidate>| {
            for (key, a) in std::mem::take(active) {
                emit_candidate(key, a, out);
            }
        };

        for (i, ins) in body.iter().enumerate() {
            if is_burst_barrier(ins) {
                barriers[i] = true;
                close_all(&mut active, &mut candidates);
                continue;
            }
            let info = factory.info(ins);
            let mut written_gprs = 0u16;
            for ur in info.used_registers() {
                if matches!(
                    ur.access(),
                    OpAccess::Write
                        | OpAccess::CondWrite
                        | OpAccess::ReadWrite
                        | OpAccess::ReadCondWrite
                ) {
                    let r = ur.register();
                    if r.is_gpr() && r.full_register().is_gpr64() {
                        written_gprs |= 1u16 << r.full_register().number();
                    }
                }
            }
            written[i] = written_gprs;
            // Check if this instruction performs a plain `[base64 + disp]` or indexed
            // `[base64 + idx64*scale + disp]` memory access.
            let mem_access = plain_burst_mem_access(ins, info, w.stack_flat);
            accesses[i] = mem_access;
            let mem_key = mem_access.map(|(k, _, _, _, _)| k);

            // Any active key whose base or index GPR is written WITHOUT being this instruction's
            // own addressing key closes BEFORE instruction `i`.
            if written_gprs != 0 {
                let to_close: Vec<memory_access::BurstKey> = active
                    .keys()
                    .copied()
                    .filter(|&k| written_gprs & k.gpr_mask() != 0 && mem_key != Some(k))
                    .collect();
                for k in to_close {
                    if let Some(a) = active.remove(&k) {
                        emit_candidate(k, a, &mut candidates);
                    }
                }
            }

            if let Some((key, disp, size, is_write, is_simd_mem)) = mem_access {
                let end_d = disp.saturating_add(size as i64);
                let extend = match active.get(&key).copied() {
                    Some(a) if i - a.start < MAX_BURST_INSNS => {
                        let new_min = a.min_d.min(disp);
                        let new_max = a.max_end.max(end_d);
                        if span_ok(new_min, new_max) {
                            Some(Active {
                                start: a.start,
                                end: i,
                                min_d: new_min,
                                max_end: new_max,
                                count: a.count + 1,
                                has_write: a.has_write || is_write,
                                has_simd_mem: a.has_simd_mem || is_simd_mem,
                            })
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                if let Some(updated) = extend {
                    active.insert(key, updated);
                } else {
                    if let Some(old) = active.remove(&key) {
                        emit_candidate(key, old, &mut candidates);
                    }
                    active.insert(
                        key,
                        Active {
                            start: i,
                            end: i,
                            min_d: disp,
                            max_end: end_d,
                            count: 1,
                            has_write: is_write,
                            has_simd_mem: is_simd_mem,
                        },
                    );
                }
            }

            // If this instruction also wrote its own memory base or index GPR (e.g.
            // `mov rdx, [rdx+8]`), close its burst right AFTER `i` so `i` is the final
            // instruction of the burst.
            if let Some(k) = mem_key
                && written_gprs & k.gpr_mask() != 0
                && let Some(a) = active.remove(&k)
            {
                emit_candidate(k, a, &mut candidates);
            }
        }
        close_all(&mut active, &mut candidates);

        // Pick non-overlapping bursts in program order (preferring earlier start, then larger span),
        // iteratively merging up to `MAX_BURST_BASES` (3) overlapping unmodified keys (R63b/R64b/R67).
        candidates.sort_by_key(|b| (b.start, std::cmp::Reverse(b.end)));
        let mut chosen: Vec<memory_access::WindowBurst> = Vec::new();
        while !candidates.is_empty() {
            let a = candidates.remove(0);
            let limit = (a.start + MAX_BURST_INSNS - 1).min(body.len() - 1);
            let mut bases = [a.base; memory_access::MAX_BURST_BASES];
            let mut base_count = 1usize;
            let mut merged_end = a.end;
            let mut chosen_mask = a.base.key.gpr_mask();

            #[derive(Clone, Copy)]
            struct Partner {
                last_i: usize,
                min_d: i64,
                max_end: i64,
                count: usize,
                has_write: bool,
                has_simd_mem: bool,
                frozen: bool,
                dead: bool,
            }

            while base_count < memory_access::MAX_BURST_BASES {
                let mut max_end_idx = merged_end;
                if written[merged_end] & chosen_mask == 0 {
                    while max_end_idx < limit
                        && !barriers[max_end_idx + 1]
                        && written[max_end_idx + 1] & chosen_mask == 0
                    {
                        max_end_idx += 1;
                    }
                }

                let mut partners: BTreeMap<memory_access::BurstKey, Partner> = BTreeMap::new();
                let mut written_since_start: u16 = 0;

                for k in a.start..=max_end_idx {
                    if let Some((key, disp, size, is_write, is_simd_mem)) = accesses[k]
                        && !bases[..base_count].iter().any(|b| b.key == key)
                    {
                        let end_d = disp.saturating_add(size as i64);
                        if let Some(p) = partners.get_mut(&key) {
                            if !p.dead && !p.frozen {
                                let new_min = p.min_d.min(disp);
                                let new_max = p.max_end.max(end_d);
                                if span_ok(new_min, new_max) {
                                    p.last_i = k;
                                    p.min_d = new_min;
                                    p.max_end = new_max;
                                    p.count += 1;
                                    p.has_write |= is_write;
                                    p.has_simd_mem |= is_simd_mem;
                                } else {
                                    p.frozen = true;
                                }
                            }
                        } else if k <= merged_end
                            && written_since_start & key.gpr_mask() == 0
                            && span_ok(disp, end_d)
                        {
                            // Only merge a partner key that overlaps the burst and whose base/index
                            // GPRs were unmodified since `a.start`.
                            partners.insert(
                                key,
                                Partner {
                                    last_i: k,
                                    min_d: disp,
                                    max_end: end_d,
                                    count: 1,
                                    has_write: is_write,
                                    has_simd_mem: is_simd_mem,
                                    frozen: false,
                                    dead: false,
                                },
                            );
                        }
                    }
                    let w_mask = written[k];
                    written_since_start |= w_mask;
                    if w_mask != 0 {
                        for (&key2, p) in &mut partners {
                            if w_mask & key2.gpr_mask() != 0 && !p.dead {
                                if k < merged_end {
                                    // Modified strictly inside `[a.start, merged_end)`: unsafe to hoist.
                                    p.dead = true;
                                } else {
                                    // Modified at or after `merged_end`: valid through `k` as the
                                    // final instruction of the burst, so freeze further extension.
                                    p.frozen = true;
                                }
                            }
                        }
                    }
                }

                let best_partner = partners
                    .iter()
                    .filter(|(_, p)| {
                        !p.dead
                            && meets_threshold(p.count, p.has_write, p.has_simd_mem)
                            && span_ok(p.min_d, p.max_end)
                    })
                    .max_by_key(|(_, p)| {
                        (p.count, usize::from(p.has_write), usize::from(p.has_simd_mem))
                    })
                    .map(|(&key2, &p)| (key2, p));

                let Some((key2, p)) = best_partner else {
                    break;
                };
                bases[base_count] = memory_access::BurstBase {
                    key: key2,
                    min_d: p.min_d,
                    max_end: p.max_end,
                    has_write: p.has_write,
                };
                base_count += 1;
                chosen_mask |= key2.gpr_mask();
                merged_end = merged_end.max(p.last_i);
            }

            chosen.push(memory_access::WindowBurst {
                start: a.start,
                end: merged_end,
                bases,
                base_count: base_count as u8,
            });

            // Drop covered candidates and trim any candidate whose tail extends past `merged_end`.
            let mut needs_sort = false;
            let mut next_candidates = Vec::with_capacity(candidates.len());
            for c in candidates {
                if c.start > merged_end {
                    next_candidates.push(c);
                } else if c.end > merged_end {
                    let mut trimmed: Option<Active> = None;
                    for (k, acc) in accesses.iter().enumerate().take(c.end + 1).skip(merged_end + 1)
                    {
                        if let Some((key, disp, size, is_write, is_simd_mem)) = *acc
                            && key == c.base.key
                        {
                            let end_d = disp.saturating_add(size as i64);
                            trimmed = Some(match trimmed {
                                Some(t) => Active {
                                    start: t.start,
                                    end: k,
                                    min_d: t.min_d.min(disp),
                                    max_end: t.max_end.max(end_d),
                                    count: t.count + 1,
                                    has_write: t.has_write || is_write,
                                    has_simd_mem: t.has_simd_mem || is_simd_mem,
                                },
                                None => Active {
                                    start: k,
                                    end: k,
                                    min_d: disp,
                                    max_end: end_d,
                                    count: 1,
                                    has_write: is_write,
                                    has_simd_mem: is_simd_mem,
                                },
                            });
                        }
                    }
                    if let Some(t) = trimmed
                        && t.end > t.start
                        && t.end - t.start < MAX_BURST_INSNS
                        && meets_threshold(t.count, t.has_write, t.has_simd_mem)
                        && span_ok(t.min_d, t.max_end)
                    {
                        next_candidates.push(SingleCandidate {
                            start: t.start,
                            end: t.end,
                            base: memory_access::BurstBase {
                                key: c.base.key,
                                min_d: t.min_d,
                                max_end: t.max_end,
                                has_write: t.has_write,
                            },
                        });
                        needs_sort = true;
                    }
                }
            }
            candidates = next_candidates;
            if needs_sort {
                candidates.sort_by_key(|b| (b.start, std::cmp::Reverse(b.end)));
            }
        }
        chosen
    }

    // ----- operands ------------------------------------------------------------------------------

    fn op_bits(&self, ins: &Instruction, i: u32) -> R<u32> {
        match ins.op_kind(i) {
            OpKind::Register => {
                let reg = ins.op_register(i);
                if !reg.is_gpr() {
                    return unsupported(format!("register operand {reg:?}"));
                }
                Ok((reg.size() * 8) as u32)
            }
            OpKind::Memory => match ins.memory_size().size() {
                size @ (1 | 2 | 4 | 8) => Ok((size * 8) as u32),
                size => unsupported(format!("{size}-byte memory operand")),
            },
            kind => unsupported(format!("operand kind {kind:?}")),
        }
    }

    fn read_op(&mut self, ins: &Instruction, i: u32, bits: u32) -> R<String> {
        match ins.op_kind(i) {
            OpKind::Register => {
                let (v, reg_bits) = self.read_reg(ins.op_register(i))?;
                if reg_bits != bits {
                    return unsupported("internal: operand width mismatch");
                }
                Ok(v)
            }
            OpKind::Memory => {
                let ea = self.mem_ea(ins)?;
                Ok(self.load_mem(&ea, bits))
            }
            kind if is_immediate(kind) => match ins.try_immediate(i) {
                Ok(value) => Ok(imm(value, bits)),
                Err(_) => unsupported("unreadable immediate"),
            },
            kind => unsupported(format!("operand kind {kind:?}")),
        }
    }

    fn write_op(&mut self, ins: &Instruction, i: u32, value: &str, bits: u32) -> R<()> {
        match ins.op_kind(i) {
            OpKind::Register => self.write_reg(ins.op_register(i), value, bits),
            OpKind::Memory => {
                let ea = self.mem_ea(ins)?;
                self.store_mem(&ea, value, bits);
                Ok(())
            }
            kind => unsupported(format!("destination operand kind {kind:?}")),
        }
    }

    // ----- flags ---------------------------------------------------------------------------------

    /// False when the current instruction's value of `flag` is dead (see [`Self::dead_flags`]).
    fn flag_live(&self, flag: Flag) -> bool {
        self.dead_flags & (1 << flag as usize) == 0
    }

    fn set_flag(&mut self, flag: Flag, value: &str) {
        debug_assert!(self.flags[flag as usize], "flag {flag:?} was not tracked");
        self.known.flag_source = None;
        if !self.flag_live(flag) {
            return;
        }
        self.emit(&format!("store i1 {value}, ptr {}", flag.slot()));
        self.known.flags[flag as usize] = Some(value.to_string());
    }

    fn get_flag(&mut self, flag: Flag) -> String {
        debug_assert!(self.flags[flag as usize], "flag {flag:?} was not tracked");
        self.load_flag(flag)
    }

    /// The flag's alloca value, forwarded from [`Self::known`] when the block already has it.
    fn load_flag(&mut self, flag: Flag) -> String {
        if let Some(v) = &self.known.flags[flag as usize] {
            return v.clone();
        }
        let v = self.val(&format!("load i1, ptr {}", flag.slot()));
        self.known.flags[flag as usize] = Some(v.clone());
        v
    }

    fn msb(&mut self, v: &str, bits: u32) -> String {
        self.val(&format!("icmp slt i{bits} {v}, 0"))
    }

    /// PF: set when the low byte of the result has an even number of 1 bits.
    fn parity(&mut self, r: &str, bits: u32) -> String {
        let low = if bits == 8 {
            r.to_string()
        } else {
            self.val(&format!("trunc i{bits} {r} to i8"))
        };
        self.support.insert(Support::Ctpop8);
        let count = self.val(&format!("call i8 @llvm.ctpop.i8(i8 {low})"));
        let odd = self.val(&format!("trunc i8 {count} to i1"));
        self.val(&format!("xor i1 {odd}, true"))
    }

    /// AF: carry/borrow out of bit 3, `((a ^ b ^ r) & 0x10) != 0`.
    fn aux_flag(&mut self, a: &str, b: &str, r: &str, bits: u32) -> String {
        let t = ty(bits);
        let x = self.val(&format!("xor {t} {a}, {b}"));
        let x = self.val(&format!("xor {t} {x}, {r}"));
        let x = self.val(&format!("and {t} {x}, 16"));
        self.val(&format!("icmp ne {t} {x}, 0"))
    }

    /// Computes and stores the live ones of ZF, SF and PF of `r`.
    fn set_zsp(&mut self, r: &str, bits: u32) {
        if self.flag_live(Flag::Zf) {
            let zf = self.val(&format!("icmp eq i{bits} {r}, 0"));
            self.set_flag(Flag::Zf, &zf);
        }
        if self.flag_live(Flag::Sf) {
            let sf = self.msb(r, bits);
            self.set_flag(Flag::Sf, &sf);
        }
        if self.flag_live(Flag::Pf) {
            let pf = self.parity(r, bits);
            self.set_flag(Flag::Pf, &pf);
        }
    }

    /// Computes and stores OF and AF of `r = a + b` (or `a - b` when `sub`) when they are live.
    fn set_of_af(&mut self, a: &str, b: &str, r: &str, bits: u32, sub: bool) {
        let t = ty(bits);
        if self.flag_live(Flag::Of) {
            let (x, y) = if sub {
                // OF: operands differ in sign and the result's sign differs from a.
                (
                    self.val(&format!("xor {t} {a}, {b}")),
                    self.val(&format!("xor {t} {a}, {r}")),
                )
            } else {
                // OF: both inputs differ in sign from the result.
                (
                    self.val(&format!("xor {t} {a}, {r}")),
                    self.val(&format!("xor {t} {b}, {r}")),
                )
            };
            let xy = self.val(&format!("and {t} {x}, {y}"));
            let of = self.msb(&xy, bits);
            self.set_flag(Flag::Of, &of);
        }
        if self.flag_live(Flag::Af) {
            let af = self.aux_flag(a, b, r, bits);
            self.set_flag(Flag::Af, &af);
        }
    }

    fn negate_i1(&mut self, v: &str) -> String {
        match v {
            "true" => "false".to_string(),
            "false" => "true".to_string(),
            _ => self.val(&format!("xor i1 {v}, true")),
        }
    }

    fn or_i1(&mut self, a: &str, b: &str) -> String {
        match (a, b) {
            ("true", _) | (_, "true") => "true".to_string(),
            ("false", x) | (x, "false") => x.to_string(),
            _ => self.val(&format!("or i1 {a}, {b}")),
        }
    }

    fn xor_i1(&mut self, a: &str, b: &str) -> String {
        match (a, b) {
            ("false", x) | (x, "false") => x.to_string(),
            ("true", x) | (x, "true") => self.negate_i1(x),
            _ => self.val(&format!("xor i1 {a}, {b}")),
        }
    }

    /// Evaluates `cc` directly from the current block's [`FlagSource`] as a single `icmp` (or
    /// constant), avoiding flag-slot loads and boolean flag chains.
    fn try_fused_condition(&mut self, cc: ConditionCode) -> Option<String> {
        use ConditionCode as C;
        let src = self.known.flag_source.clone()?;
        match src {
            FlagSource::Compare { a, b, diff, bits } => {
                let t = ty(bits);
                match cc {
                    C::e => Some(match diff {
                        Some(r) => self.val(&format!("icmp eq {t} {r}, 0")),
                        None => self.val(&format!("icmp eq {t} {a}, {b}")),
                    }),
                    C::ne => Some(match diff {
                        Some(r) => self.val(&format!("icmp ne {t} {r}, 0")),
                        None => self.val(&format!("icmp ne {t} {a}, {b}")),
                    }),
                    C::b => Some(if b == "0" {
                        "false".to_string()
                    } else {
                        self.val(&format!("icmp ult {t} {a}, {b}"))
                    }),
                    C::ae => Some(if b == "0" {
                        "true".to_string()
                    } else {
                        self.val(&format!("icmp uge {t} {a}, {b}"))
                    }),
                    C::be => Some(self.val(&format!("icmp ule {t} {a}, {b}"))),
                    C::a => Some(self.val(&format!("icmp ugt {t} {a}, {b}"))),
                    C::l => Some(self.val(&format!("icmp slt {t} {a}, {b}"))),
                    C::ge => Some(self.val(&format!("icmp sge {t} {a}, {b}"))),
                    C::le => Some(self.val(&format!("icmp sle {t} {a}, {b}"))),
                    C::g => Some(self.val(&format!("icmp sgt {t} {a}, {b}"))),
                    C::s => match diff {
                        Some(r) => Some(self.val(&format!("icmp slt {t} {r}, 0"))),
                        None if b == "0" => Some(self.val(&format!("icmp slt {t} {a}, 0"))),
                        None => None,
                    },
                    C::ns => match diff {
                        Some(r) => Some(self.val(&format!("icmp sge {t} {r}, 0"))),
                        None if b == "0" => Some(self.val(&format!("icmp sge {t} {a}, 0"))),
                        None => None,
                    },
                    _ => None,
                }
            }
            FlagSource::LogicZero { result, bits } => {
                let t = ty(bits);
                match cc {
                    C::o | C::b => Some("false".to_string()),
                    C::no | C::ae => Some("true".to_string()),
                    C::e | C::be => Some(self.val(&format!("icmp eq {t} {result}, 0"))),
                    C::ne | C::a => Some(self.val(&format!("icmp ne {t} {result}, 0"))),
                    C::s | C::l => Some(self.val(&format!("icmp slt {t} {result}, 0"))),
                    C::ns | C::ge => Some(self.val(&format!("icmp sge {t} {result}, 0"))),
                    C::le => Some(self.val(&format!("icmp sle {t} {result}, 0"))),
                    C::g => Some(self.val(&format!("icmp sgt {t} {result}, 0"))),
                    _ => None,
                }
            }
            FlagSource::Add { a, result, bits } => {
                let t = ty(bits);
                match cc {
                    C::e => Some(self.val(&format!("icmp eq {t} {result}, 0"))),
                    C::ne => Some(self.val(&format!("icmp ne {t} {result}, 0"))),
                    C::s => Some(self.val(&format!("icmp slt {t} {result}, 0"))),
                    C::ns => Some(self.val(&format!("icmp sge {t} {result}, 0"))),
                    C::b => Some(self.val(&format!("icmp ult {t} {result}, {a}"))),
                    C::ae => Some(self.val(&format!("icmp uge {t} {result}, {a}"))),
                    _ => None,
                }
            }
            FlagSource::ResultZeroSign { result, bits } => {
                let t = ty(bits);
                match cc {
                    C::e => Some(self.val(&format!("icmp eq {t} {result}, 0"))),
                    C::ne => Some(self.val(&format!("icmp ne {t} {result}, 0"))),
                    C::s => Some(self.val(&format!("icmp slt {t} {result}, 0"))),
                    C::ns => Some(self.val(&format!("icmp sge {t} {result}, 0"))),
                    _ => None,
                }
            }
            FlagSource::Dec { a, result, bits } => {
                let t = ty(bits);
                match cc {
                    C::e => Some(self.val(&format!("icmp eq {t} {result}, 0"))),
                    C::ne => Some(self.val(&format!("icmp ne {t} {result}, 0"))),
                    C::s => Some(self.val(&format!("icmp slt {t} {result}, 0"))),
                    C::ns => Some(self.val(&format!("icmp sge {t} {result}, 0"))),
                    C::l => Some(self.val(&format!("icmp slt {t} {a}, 1"))),
                    C::ge => Some(self.val(&format!("icmp sge {t} {a}, 1"))),
                    C::le => Some(self.val(&format!("icmp sle {t} {a}, 1"))),
                    C::g => Some(self.val(&format!("icmp sgt {t} {a}, 1"))),
                    _ => None,
                }
            }
            FlagSource::Inc { a, result, bits } => {
                let t = ty(bits);
                match cc {
                    C::e => Some(self.val(&format!("icmp eq {t} {result}, 0"))),
                    C::ne => Some(self.val(&format!("icmp ne {t} {result}, 0"))),
                    C::s => Some(self.val(&format!("icmp slt {t} {result}, 0"))),
                    C::ns => Some(self.val(&format!("icmp sge {t} {result}, 0"))),
                    C::l => Some(self.val(&format!("icmp slt {t} {a}, -1"))),
                    C::ge => Some(self.val(&format!("icmp sge {t} {a}, -1"))),
                    C::le => Some(self.val(&format!("icmp sle {t} {a}, -1"))),
                    C::g => Some(self.val(&format!("icmp sgt {t} {a}, -1"))),
                    _ => None,
                }
            }
            FlagSource::ZeroOnly { val, bits } => {
                let t = ty(bits);
                match cc {
                    C::e => Some(self.val(&format!("icmp eq {t} {val}, 0"))),
                    C::ne => Some(self.val(&format!("icmp ne {t} {val}, 0"))),
                    _ => None,
                }
            }
            FlagSource::ZeroBit { is_zero } => match cc {
                C::e => Some(is_zero),
                C::ne => Some(self.negate_i1(&is_zero)),
                _ => None,
            },
            FlagSource::BitCount {
                src,
                is_tzcnt,
                bits,
            } => {
                let t = ty(bits);
                match cc {
                    C::e => Some(if is_tzcnt {
                        self.val(&format!("trunc {t} {src} to i1"))
                    } else {
                        self.msb(&src, bits)
                    }),
                    C::ne => Some(if is_tzcnt {
                        let hit = self.val(&format!("and {t} {src}, 1"));
                        self.val(&format!("icmp eq {t} {hit}, 0"))
                    } else {
                        self.val(&format!("icmp sge {t} {src}, 0"))
                    }),
                    C::b => Some(self.val(&format!("icmp eq {t} {src}, 0"))),
                    C::ae => Some(self.val(&format!("icmp ne {t} {src}, 0"))),
                    _ => None,
                }
            }
            FlagSource::CarryBit { cf } => match cc {
                C::b => Some(cf),
                C::ae => Some(self.negate_i1(&cf)),
                _ => None,
            },
            FlagSource::BitTest { val, mask, bits } => {
                let t = ty(bits);
                match cc {
                    C::b => {
                        let hit = self.val(&format!("and {t} {val}, {mask}"));
                        Some(self.val(&format!("icmp ne {t} {hit}, 0")))
                    }
                    C::ae => {
                        let hit = self.val(&format!("and {t} {val}, {mask}"));
                        Some(self.val(&format!("icmp eq {t} {hit}, 0")))
                    }
                    _ => None,
                }
            }
            FlagSource::BitTestBit { val, bit, bits } => {
                let t = ty(bits);
                match cc {
                    C::b => Some(if bit == 0 {
                        self.val(&format!("trunc {t} {val} to i1"))
                    } else if bit == bits - 1 {
                        self.msb(&val, bits)
                    } else {
                        let m = imm(1u64 << bit, bits);
                        let hit = self.val(&format!("and {t} {val}, {m}"));
                        self.val(&format!("icmp ne {t} {hit}, 0"))
                    }),
                    C::ae => Some(if bit == bits - 1 {
                        self.val(&format!("icmp sge {t} {val}, 0"))
                    } else {
                        let m = imm(1u64 << bit, bits);
                        let hit = self.val(&format!("and {t} {val}, {m}"));
                        self.val(&format!("icmp eq {t} {hit}, 0"))
                    }),
                    _ => None,
                }
            }
            FlagSource::FloatCompare { a, b, scalar_ty } => match cc {
                C::o | C::s | C::l => Some("false".to_string()),
                C::no | C::ns | C::ge => Some("true".to_string()),
                C::b => Some(self.val(&format!("fcmp ult {scalar_ty} {a}, {b}"))),
                C::ae => Some(self.val(&format!("fcmp oge {scalar_ty} {a}, {b}"))),
                C::e | C::le => Some(self.val(&format!("fcmp ueq {scalar_ty} {a}, {b}"))),
                C::ne | C::g => Some(self.val(&format!("fcmp one {scalar_ty} {a}, {b}"))),
                C::be => Some(self.val(&format!("fcmp ule {scalar_ty} {a}, {b}"))),
                C::a => Some(self.val(&format!("fcmp ogt {scalar_ty} {a}, {b}"))),
                C::p => Some(self.val(&format!("fcmp uno {scalar_ty} {a}, {b}"))),
                C::np => Some(self.val(&format!("fcmp ord {scalar_ty} {a}, {b}"))),
                C::None => None,
            },
        }
    }

    fn condition(&mut self, cc: ConditionCode) -> R<String> {
        use ConditionCode as C;
        if let Some(fused) = self.try_fused_condition(cc) {
            return Ok(fused);
        }
        let (flag, negate) = match cc {
            C::o => (Flag::Of, false),
            C::no => (Flag::Of, true),
            C::b => (Flag::Cf, false),
            C::ae => (Flag::Cf, true),
            C::e => (Flag::Zf, false),
            C::ne => (Flag::Zf, true),
            C::s => (Flag::Sf, false),
            C::ns => (Flag::Sf, true),
            C::p => (Flag::Pf, false),
            C::np => (Flag::Pf, true),
            C::be | C::a => {
                let cf = self.get_flag(Flag::Cf);
                let zf = self.get_flag(Flag::Zf);
                let be = self.or_i1(&cf, &zf);
                return Ok(if cc == C::be {
                    be
                } else {
                    self.negate_i1(&be)
                });
            }
            C::l | C::ge => {
                let sf = self.get_flag(Flag::Sf);
                let of = self.get_flag(Flag::Of);
                let l = self.xor_i1(&sf, &of);
                return Ok(if cc == C::l { l } else { self.negate_i1(&l) });
            }
            C::le | C::g => {
                let sf = self.get_flag(Flag::Sf);
                let of = self.get_flag(Flag::Of);
                let zf = self.get_flag(Flag::Zf);
                let l = self.xor_i1(&sf, &of);
                let le = self.or_i1(&zf, &l);
                return Ok(if cc == C::le {
                    le
                } else {
                    self.negate_i1(&le)
                });
            }
            C::None => return unsupported("missing condition code"),
        };
        let v = self.get_flag(flag);
        Ok(if negate { self.negate_i1(&v) } else { v })
    }

    /// Reads the incoming carry flag (`CF`), preferring direct fusion from the immediately
    /// preceding producer's `FlagSource` (captured in `self.carry_source`) over loading `%cf`.
    pub(super) fn take_carry_flag(&mut self) -> String {
        if let Some(src) = self.carry_source.take() {
            let saved = self.known.flag_source.replace(src);
            let fused = self.try_fused_condition(ConditionCode::b);
            self.known.flag_source = saved;
            if let Some(cf) = fused {
                return cf;
            }
        }
        self.get_flag(Flag::Cf)
    }

    // ----- instructions --------------------------------------------------------------------------

    /// Lifts one non-terminating instruction (calls included).
    fn lift_instruction(&mut self, ins: &Instruction) -> R<()> {
        use Mnemonic as M;
        self.carry_source = None;
        if ins.code() == Code::INVALID {
            return unsupported("invalid instruction encoding");
        }
        // Any instruction that can modify flags (or a call) invalidates the pending `FlagSource`
        // before it runs; fusable producers reinstall their own `FlagSource` after updating flags.
        // Carry-consuming non-locked instructions (`adc`/`sbb`/`adcx`/`cmc`) preserve the prior
        // `FlagSource` in `carry_source` so they can consume `CF` without touching `%cf`.
        if ins.rflags_modified() != 0 || ins.mnemonic() == M::Call {
            let prev = self.known.flag_source.take();
            if !ins.has_lock_prefix()
                && matches!(ins.mnemonic(), M::Adc | M::Sbb | M::Adcx | M::Cmc)
            {
                self.carry_source = prev;
            }
        }
        if simd::is_simd(ins) {
            return self.lift_simd(ins);
        }
        // String, bit, atomic and miscellaneous integer instructions (incl. every `lock` form).
        if let Some(result) = self.lift_integer(ins) {
            return result;
        }
        match ins.mnemonic() {
            M::Nop | M::Endbr64 => {}
            // x87 control/status plumbing that `setjmp`/`longjmp` and CRT startup use. No x87
            // arithmetic is modelled, so the unit stays in its reset state: control word 0x027F
            // (the Windows default, 53-bit precision), status word 0, nothing pending.
            M::Fnstcw | M::Fstcw => self.write_op(ins, 0, "639", 16)?,
            M::Fldcw => {
                // Read for the access check; the value does not change anything modelled.
                self.read_op(ins, 0, 16)?;
            }
            M::Fnstsw | M::Fstsw => self.write_op(ins, 0, "0", 16)?,
            M::Fnclex | M::Fclex | M::Fninit | M::Finit | M::Wait | M::Fnop => {}
            M::Mov => {
                let bits = self.op_bits(ins, 0)?;
                let v = self.read_op(ins, 1, bits)?;
                self.write_op(ins, 0, &v, bits)?;
            }
            M::Movzx | M::Movsx | M::Movsxd => {
                let dst_bits = self.op_bits(ins, 0)?;
                let src_bits = self.op_bits(ins, 1)?;
                if src_bits > dst_bits {
                    return unsupported("narrowing extension");
                }
                if ins.mnemonic() == M::Movzx
                    && ins.op_kind(0) == OpKind::Register
                    && matches!(dst_bits, 32 | 64)
                    && src_bits < 32
                {
                    let RegRef::Low(dst_idx, _) = reg_ref(ins.op_register(0))? else {
                        return unsupported("movzx destination");
                    };
                    // `movzx r32/r64, r8_low/r16`: mask the 64-bit source directly instead of
                    // `trunc -> zext -> zext`. Memory / high-byte sources extend straight to i64.
                    let wide = if ins.op_kind(1) == OpKind::Register
                        && let Ok(RegRef::Low(src_idx, sb)) = reg_ref(ins.op_register(1))
                        && sb == src_bits
                    {
                        let full = self.load_gpr(src_idx)?;
                        self.val(&format!("and i64 {full}, {}", mask(src_bits)))
                    } else {
                        let v = self.read_op(ins, 1, src_bits)?;
                        self.val(&format!("zext i{src_bits} {v} to i64"))
                    };
                    self.store_gpr(dst_idx, &wide)?;
                    return Ok(());
                }
                let v = self.read_op(ins, 1, src_bits)?;
                let v = if src_bits == dst_bits {
                    v
                } else {
                    let ext = if ins.mnemonic() == M::Movzx {
                        "zext"
                    } else {
                        "sext"
                    };
                    self.val(&format!("{ext} i{src_bits} {v} to i{dst_bits}"))
                };
                self.write_op(ins, 0, &v, dst_bits)?;
            }
            M::Lea => {
                let bits = self.op_bits(ins, 0)?;
                if ins.op_kind(1) != OpKind::Memory {
                    return unsupported("lea without memory operand");
                }
                let addr = self.address(ins, false)?;
                if bits == 32 && ins.op_kind(0) == OpKind::Register {
                    let RegRef::Low(dst_idx, 32) = reg_ref(ins.op_register(0))? else {
                        return unsupported("lea destination");
                    };
                    let masked = if let Ok(c) = addr.parse::<i64>() {
                        imm((c as u64) & 0xFFFF_FFFF, 64)
                    } else {
                        self.val(&format!("and i64 {addr}, 4294967295"))
                    };
                    self.store_gpr(dst_idx, &masked)?;
                    return Ok(());
                }
                let v = if bits == 64 {
                    addr
                } else {
                    self.val(&format!("trunc i64 {addr} to i{bits}"))
                };
                self.write_op(ins, 0, &v, bits)?;
            }
            M::Xchg => {
                let bits = self.op_bits(ins, 0)?;
                let a = self.read_op(ins, 0, bits)?;
                let b = self.read_op(ins, 1, bits)?;
                self.write_op(ins, 0, &b, bits)?;
                self.write_op(ins, 1, &a, bits)?;
            }
            M::Add => self.alu(ins, Alu::Add)?,
            M::Adc => self.alu(ins, Alu::Adc)?,
            M::Sub => self.alu(ins, Alu::Sub)?,
            M::Sbb => self.alu(ins, Alu::Sbb)?,
            M::Cmp => self.alu(ins, Alu::Cmp)?,
            M::And => self.alu(ins, Alu::And)?,
            M::Or => self.alu(ins, Alu::Or)?,
            M::Xor => self.alu(ins, Alu::Xor)?,
            M::Test => self.alu(ins, Alu::Test)?,
            M::Inc | M::Dec | M::Neg | M::Not => self.unary(ins)?,
            M::Shl | M::Sal => self.shift(ins, Shift::Shl)?,
            M::Shr => self.shift(ins, Shift::Shr)?,
            M::Sar => self.shift(ins, Shift::Sar)?,
            M::Rol => self.shift(ins, Shift::Rol)?,
            M::Ror => self.shift(ins, Shift::Ror)?,
            M::Imul if ins.op_count() >= 2 => self.imul_truncating(ins)?,
            M::Imul | M::Mul => self.mul_widening(ins, ins.mnemonic() == M::Imul)?,
            M::Div | M::Idiv => self.divide(ins, ins.mnemonic() == M::Idiv)?,
            M::Cbw | M::Cwde | M::Cdqe => {
                let (src, dst, bits) = match ins.mnemonic() {
                    M::Cbw => (Register::AL, Register::AX, 16),
                    M::Cwde => (Register::AX, Register::EAX, 32),
                    _ => (Register::EAX, Register::RAX, 64),
                };
                let (v, src_bits) = self.read_reg(src)?;
                let ext = self.val(&format!("sext i{src_bits} {v} to i{bits}"));
                self.write_reg(dst, &ext, bits)?;
            }
            M::Cwd | M::Cdq | M::Cqo => {
                let (src, dst) = match ins.mnemonic() {
                    M::Cwd => (Register::AX, Register::DX),
                    M::Cdq => (Register::EAX, Register::EDX),
                    _ => (Register::RAX, Register::RDX),
                };
                let (v, bits) = self.read_reg(src)?;
                let sign = self.val(&format!("ashr i{bits} {v}, {}", bits - 1));
                self.write_reg(dst, &sign, bits)?;
                if matches!(bits, 32 | 64)
                    && let (Some(rdx64), Some(rax64)) = (
                        self.known.gprs[integer::RDX].clone(),
                        self.known.gprs[integer::RAX].clone(),
                    )
                {
                    self.known.sign_ext_rdx = Some((rdx64, rax64, bits));
                }
            }
            M::Seto
            | M::Setno
            | M::Setb
            | M::Setae
            | M::Sete
            | M::Setne
            | M::Setbe
            | M::Seta
            | M::Sets
            | M::Setns
            | M::Setp
            | M::Setnp
            | M::Setl
            | M::Setge
            | M::Setle
            | M::Setg => {
                let c = self.condition(ins.condition_code())?;
                let byte = match c.as_str() {
                    "true" => "1".to_string(),
                    "false" => "0".to_string(),
                    _ => self.val(&format!("zext i1 {c} to i8")),
                };
                self.write_op(ins, 0, &byte, 8)?;
            }
            M::Cmovo
            | M::Cmovno
            | M::Cmovb
            | M::Cmovae
            | M::Cmove
            | M::Cmovne
            | M::Cmovbe
            | M::Cmova
            | M::Cmovs
            | M::Cmovns
            | M::Cmovp
            | M::Cmovnp
            | M::Cmovl
            | M::Cmovge
            | M::Cmovle
            | M::Cmovg => {
                let bits = self.op_bits(ins, 0)?;
                if ins.op_kind(0) == OpKind::Register
                    && ins.op_kind(1) == OpKind::Register
                    && ins.op_register(0) == ins.op_register(1)
                {
                    // Same-register `cmovcc`: 64/16-bit is a pure no-op; 32-bit unconditionally
                    // zero-extends the 32-bit register into 64-bit regardless of flags.
                    if bits == 32 {
                        let RegRef::Low(dst_idx, 32) = reg_ref(ins.op_register(0))? else {
                            return unsupported("cmov destination");
                        };
                        let old64 = self.load_gpr(dst_idx)?;
                        let wide = if let Some(c) = memory_access::const_addr(&old64) {
                            imm(c & mask(32), 64)
                        } else {
                            self.val(&format!("and i64 {old64}, 4294967295"))
                        };
                        self.store_gpr(dst_idx, &wide)?;
                    }
                    return Ok(());
                }
                let c = self.condition(ins.condition_code())?;
                if bits == 32
                    && ins.op_kind(0) == OpKind::Register
                    && ins.op_kind(1) == OpKind::Register
                {
                    let RegRef::Low(dst_idx, 32) = reg_ref(ins.op_register(0))? else {
                        return unsupported("cmov destination");
                    };
                    let RegRef::Low(src_idx, 32) = reg_ref(ins.op_register(1))? else {
                        return unsupported("cmov source");
                    };
                    let src64 = self.load_gpr(src_idx)?;
                    let old64 = self.load_gpr(dst_idx)?;
                    let sel = match c.as_str() {
                        "true" => src64,
                        "false" => old64,
                        _ if src64 == old64 => old64,
                        _ => self.val(&format!("select i1 {c}, i64 {src64}, i64 {old64}")),
                    };
                    // A 32-bit cmov zero-extends the destination even when the condition is false.
                    let wide = if let Some(v) = memory_access::const_addr(&sel) {
                        imm(v & mask(32), 64)
                    } else {
                        self.val(&format!("and i64 {sel}, 4294967295"))
                    };
                    self.store_gpr(dst_idx, &wide)?;
                } else {
                    let src = self.read_op(ins, 1, bits)?;
                    let old = self.read_op(ins, 0, bits)?;
                    let v = match c.as_str() {
                        "true" => src,
                        "false" => old,
                        _ if src == old => old,
                        _ => self.val(&format!("select i1 {c}, i{bits} {src}, i{bits} {old}")),
                    };
                    // A 32-bit cmov zero-extends the destination even when the condition is false.
                    self.write_op(ins, 0, &v, bits)?;
                }
            }
            M::Push => {
                let bits = match ins.stack_pointer_increment() {
                    -8 => 64,
                    -2 => 16,
                    other => return unsupported(format!("push adjusting rsp by {other}")),
                };
                if matches!(ins.op_kind(0), OpKind::Register | OpKind::Memory)
                    && self.op_bits(ins, 0)? != bits
                {
                    return unsupported("push operand size");
                }
                // The value is read before rsp changes (`push rsp`, `push [rsp+8]`).
                let v = self.read_op(ins, 0, bits)?;
                self.push(&v, bits)?;
            }
            M::Pop => {
                let bits = match ins.stack_pointer_increment() {
                    8 => 64,
                    2 => 16,
                    other => return unsupported(format!("pop adjusting rsp by {other}")),
                };
                if self.op_bits(ins, 0)? != bits {
                    return unsupported("pop operand size");
                }
                let rsp = self.load_gpr(RSP)?;
                let v = self.load_mem(&rsp, bits);
                let new_rsp = self.val(&format!("add i64 {rsp}, {}", bits / 8));
                self.store_gpr(RSP, &new_rsp)?;
                // The destination is written (and a memory destination addressed) after the
                // increment, so `pop rsp` and `pop [rsp+8]` behave like on hardware.
                self.write_op(ins, 0, &v, bits)?;
            }
            M::Leave => {
                if ins.code() != Code::Leaveq {
                    return unsupported("16/32-bit leave");
                }
                let rbp = self.load_gpr(RBP)?;
                let saved = self.load_mem(&rbp, 64);
                let new_rsp = self.val(&format!("add i64 {rbp}, 8"));
                self.store_gpr(RSP, &new_rsp)?;
                self.store_gpr(RBP, &saved)?;
            }
            M::Call => self.call(ins)?,
            m => return unsupported(format!("instruction {m:?} is not supported yet")),
        }
        Ok(())
    }

    fn alu(&mut self, ins: &Instruction, op: Alu) -> R<()> {
        let bits = self.op_bits(ins, 0)?;
        if ins.op_kind(0) == OpKind::Register
            && ins.op_kind(1) == OpKind::Register
            && ins.op_register(0) == ins.op_register(1)
        {
            match op {
                Alu::Xor | Alu::Sub | Alu::Cmp => {
                    self.set_flag(Flag::Cf, "false");
                    self.set_flag(Flag::Of, "false");
                    if !matches!(op, Alu::Xor) {
                        self.set_flag(Flag::Af, "false");
                    }
                    self.set_flag(Flag::Zf, "true");
                    self.set_flag(Flag::Sf, "false");
                    self.set_flag(Flag::Pf, "true");
                    if !matches!(op, Alu::Cmp) {
                        self.write_op(ins, 0, "0", bits)?;
                    }
                    return Ok(());
                }
                Alu::And | Alu::Or | Alu::Test => {
                    let a = self.read_op(ins, 0, bits)?;
                    self.set_flag(Flag::Cf, "false");
                    self.set_flag(Flag::Of, "false");
                    self.set_zsp(&a, bits);
                    if !matches!(op, Alu::Test) {
                        self.write_op(ins, 0, &a, bits)?;
                    }
                    self.known.flag_source = Some(FlagSource::LogicZero { result: a, bits });
                    return Ok(());
                }
                Alu::Sbb => {
                    // `sbb r, r` computes `-CF` (`0` or all-ones): CF=c, OF=0, AF=c, ZF=!c, SF=c,
                    // PF=1 (both `0x00` and `0xFF` have even parity) without reading `r` (32/64b).
                    let c = self.take_carry_flag();
                    self.set_flag(Flag::Cf, &c);
                    self.set_flag(Flag::Of, "false");
                    self.set_flag(Flag::Af, &c);
                    if self.flag_live(Flag::Zf) {
                        let zf = self.negate_i1(&c);
                        self.set_flag(Flag::Zf, &zf);
                    }
                    self.set_flag(Flag::Sf, &c);
                    self.set_flag(Flag::Pf, "true");
                    let r = match c.as_str() {
                        "false" => "0".to_string(),
                        "true" => "-1".to_string(),
                        _ => self.val(&format!("sext i1 {c} to {}", ty(bits))),
                    };
                    self.known.flag_source = Some(FlagSource::ResultZeroSign {
                        result: r.clone(),
                        bits,
                    });
                    return self.write_op(ins, 0, &r, bits);
                }
                Alu::Add | Alu::Adc => {}
            }
        }
        let a = self.read_op(ins, 0, bits)?;
        let b = self.read_op(ins, 1, bits)?;
        let r = self.alu_value(op, &a, &b, bits);
        if !matches!(op, Alu::Cmp | Alu::Test) {
            self.write_op(ins, 0, &r, bits)?;
        }
        Ok(())
    }

    /// Computes `a <op> b` (reading CF for `adc`/`sbb`), sets every flag the operation defines
    /// and returns the result. Also used with the old value of an atomic read-modify-write.
    fn alu_value(&mut self, op: Alu, a: &str, b: &str, bits: u32) -> String {
        let t = ty(bits);
        match op {
            Alu::Add => {
                let r = if b == "0" {
                    self.set_flag(Flag::Cf, "false");
                    self.set_flag(Flag::Of, "false");
                    self.set_flag(Flag::Af, "false");
                    a.to_string()
                } else {
                    let r = self.val(&format!("add {t} {a}, {b}"));
                    if self.flag_live(Flag::Cf) {
                        let cf = self.val(&format!("icmp ult {t} {r}, {a}"));
                        self.set_flag(Flag::Cf, &cf);
                    }
                    self.set_of_af(a, b, &r, bits, false);
                    r
                };
                self.set_zsp(&r, bits);
                self.known.flag_source = Some(FlagSource::Add {
                    a: a.to_string(),
                    result: r.clone(),
                    bits,
                });
                r
            }
            Alu::Adc => {
                // carry out of a + b + c: r < a, or r == a with a carry in.
                let c = self.take_carry_flag();
                if c == "false" {
                    return self.alu_value(Alu::Add, a, b, bits);
                }
                let r = if b == "0" {
                    let r = if c == "true" {
                        self.val(&format!("add {t} {a}, 1"))
                    } else {
                        let cz = self.val(&format!("zext i1 {c} to {t}"));
                        self.val(&format!("add {t} {a}, {cz}"))
                    };
                    if self.flag_live(Flag::Cf) {
                        let cf = self.val(&format!("icmp ult {t} {r}, {a}"));
                        self.set_flag(Flag::Cf, &cf);
                    }
                    r
                } else if c == "true" {
                    let s = self.val(&format!("add {t} {a}, {b}"));
                    let r = self.val(&format!("add {t} {s}, 1"));
                    if self.flag_live(Flag::Cf) {
                        let cf = self.val(&format!("icmp ule {t} {r}, {a}"));
                        self.set_flag(Flag::Cf, &cf);
                    }
                    r
                } else {
                    let cz = self.val(&format!("zext i1 {c} to {t}"));
                    let s = self.val(&format!("add {t} {a}, {b}"));
                    let r = self.val(&format!("add {t} {s}, {cz}"));
                    if self.flag_live(Flag::Cf) {
                        let lt = self.val(&format!("icmp ult {t} {r}, {a}"));
                        let eq = self.val(&format!("icmp eq {t} {r}, {a}"));
                        let ceq = self.val(&format!("and i1 {c}, {eq}"));
                        let cf = self.val(&format!("or i1 {lt}, {ceq}"));
                        self.set_flag(Flag::Cf, &cf);
                    }
                    r
                };
                self.set_of_af(a, b, &r, bits, false);
                self.set_zsp(&r, bits);
                self.known.flag_source = Some(FlagSource::ResultZeroSign {
                    result: r.clone(),
                    bits,
                });
                r
            }
            Alu::Cmp => {
                let diff = if b == "0" {
                    self.set_flag(Flag::Cf, "false");
                    self.set_flag(Flag::Of, "false");
                    self.set_flag(Flag::Af, "false");
                    self.set_zsp(a, bits);
                    None
                } else {
                    if self.flag_live(Flag::Cf) {
                        let cf = self.val(&format!("icmp ult {t} {a}, {b}"));
                        self.set_flag(Flag::Cf, &cf);
                    }
                    if self.flag_live(Flag::Zf) {
                        let zf = self.val(&format!("icmp eq {t} {a}, {b}"));
                        self.set_flag(Flag::Zf, &zf);
                    }
                    let need_diff = self.flag_live(Flag::Of)
                        || self.flag_live(Flag::Af)
                        || self.flag_live(Flag::Sf)
                        || self.flag_live(Flag::Pf);
                    if need_diff {
                        let r = self.val(&format!("sub {t} {a}, {b}"));
                        self.set_of_af(a, b, &r, bits, true);
                        if self.flag_live(Flag::Sf) {
                            let sf = self.msb(&r, bits);
                            self.set_flag(Flag::Sf, &sf);
                        }
                        if self.flag_live(Flag::Pf) {
                            let pf = self.parity(&r, bits);
                            self.set_flag(Flag::Pf, &pf);
                        }
                        Some(r)
                    } else {
                        None
                    }
                };
                self.known.flag_source = Some(FlagSource::Compare {
                    a: a.to_string(),
                    b: b.to_string(),
                    diff,
                    bits,
                });
                String::new()
            }
            Alu::Sub => {
                let r = if b == "0" {
                    self.set_flag(Flag::Cf, "false");
                    self.set_flag(Flag::Of, "false");
                    self.set_flag(Flag::Af, "false");
                    a.to_string()
                } else {
                    let r = self.val(&format!("sub {t} {a}, {b}"));
                    if self.flag_live(Flag::Cf) {
                        let cf = self.val(&format!("icmp ult {t} {a}, {b}"));
                        self.set_flag(Flag::Cf, &cf);
                    }
                    self.set_of_af(a, b, &r, bits, true);
                    r
                };
                self.set_zsp(&r, bits);
                self.known.flag_source = Some(FlagSource::Compare {
                    a: a.to_string(),
                    b: b.to_string(),
                    diff: Some(r.clone()),
                    bits,
                });
                r
            }
            Alu::Sbb => {
                // borrow out of a - b - c: a < b, or a == b with a borrow in.
                let c = self.take_carry_flag();
                if c == "false" {
                    return self.alu_value(Alu::Sub, a, b, bits);
                }
                let r = if b == "0" {
                    let r = if c == "true" {
                        self.val(&format!("sub {t} {a}, 1"))
                    } else {
                        let cz = self.val(&format!("zext i1 {c} to {t}"));
                        self.val(&format!("sub {t} {a}, {cz}"))
                    };
                    if self.flag_live(Flag::Cf) {
                        let cf = self.val(&format!("icmp ugt {t} {r}, {a}"));
                        self.set_flag(Flag::Cf, &cf);
                    }
                    r
                } else if c == "true" {
                    let d = self.val(&format!("sub {t} {a}, {b}"));
                    let r = self.val(&format!("sub {t} {d}, 1"));
                    if self.flag_live(Flag::Cf) {
                        let cf = self.val(&format!("icmp ule {t} {a}, {b}"));
                        self.set_flag(Flag::Cf, &cf);
                    }
                    r
                } else {
                    let cz = self.val(&format!("zext i1 {c} to {t}"));
                    let d = self.val(&format!("sub {t} {a}, {b}"));
                    let r = self.val(&format!("sub {t} {d}, {cz}"));
                    if self.flag_live(Flag::Cf) {
                        let lt = self.val(&format!("icmp ult {t} {a}, {b}"));
                        let eq = self.val(&format!("icmp eq {t} {a}, {b}"));
                        let ceq = self.val(&format!("and i1 {c}, {eq}"));
                        let cf = self.val(&format!("or i1 {lt}, {ceq}"));
                        self.set_flag(Flag::Cf, &cf);
                    }
                    r
                };
                self.set_of_af(a, b, &r, bits, true);
                self.set_zsp(&r, bits);
                self.known.flag_source = Some(FlagSource::ResultZeroSign {
                    result: r.clone(),
                    bits,
                });
                r
            }
            Alu::And | Alu::Or | Alu::Xor | Alu::Test => {
                let opcode = match op {
                    Alu::Or => "or",
                    Alu::Xor => "xor",
                    _ => "and",
                };
                let r = self.val(&format!("{opcode} {t} {a}, {b}"));
                // CF and OF are cleared; AF is undefined and left unchanged.
                self.set_flag(Flag::Cf, "false");
                self.set_flag(Flag::Of, "false");
                self.set_zsp(&r, bits);
                self.known.flag_source = Some(FlagSource::LogicZero {
                    result: r.clone(),
                    bits,
                });
                r
            }
        }
    }

    fn unary(&mut self, ins: &Instruction) -> R<()> {
        let bits = self.op_bits(ins, 0)?;
        let a = self.read_op(ins, 0, bits)?;
        let r = self.unary_value(ins.mnemonic(), &a, bits);
        self.write_op(ins, 0, &r, bits)
    }

    /// Computes `inc`/`dec`/`neg`/`not` of `a`, sets the flags the operation defines and returns
    /// the result. Also used with the old value of an atomic read-modify-write.
    fn unary_value(&mut self, mnemonic: Mnemonic, a: &str, bits: u32) -> String {
        let t = ty(bits);
        let min = imm(1u64 << (bits - 1), bits);
        let (r, source) = match mnemonic {
            Mnemonic::Not => {
                // `not` does not modify any flag.
                return self.val(&format!("xor {t} {a}, -1"));
            }
            Mnemonic::Inc => {
                // CF is preserved by inc/dec.
                let r = self.val(&format!("add {t} {a}, 1"));
                if self.flag_live(Flag::Of) {
                    let of = self.val(&format!("icmp eq {t} {r}, {min}"));
                    self.set_flag(Flag::Of, &of);
                }
                if self.flag_live(Flag::Af) {
                    let af = self.aux_flag(a, "1", &r, bits);
                    self.set_flag(Flag::Af, &af);
                }
                let src = FlagSource::Inc {
                    a: a.to_string(),
                    result: r.clone(),
                    bits,
                };
                (r, src)
            }
            Mnemonic::Dec => {
                let r = self.val(&format!("sub {t} {a}, 1"));
                if self.flag_live(Flag::Of) {
                    let of = self.val(&format!("icmp eq {t} {a}, {min}"));
                    self.set_flag(Flag::Of, &of);
                }
                if self.flag_live(Flag::Af) {
                    let af = self.aux_flag(a, "1", &r, bits);
                    self.set_flag(Flag::Af, &af);
                }
                let src = FlagSource::Dec {
                    a: a.to_string(),
                    result: r.clone(),
                    bits,
                };
                (r, src)
            }
            _ => {
                // neg
                let r = self.val(&format!("sub {t} 0, {a}"));
                if self.flag_live(Flag::Cf) {
                    let cf = self.val(&format!("icmp ne {t} {a}, 0"));
                    self.set_flag(Flag::Cf, &cf);
                }
                if self.flag_live(Flag::Of) {
                    let of = self.val(&format!("icmp eq {t} {a}, {min}"));
                    self.set_flag(Flag::Of, &of);
                }
                if self.flag_live(Flag::Af) {
                    let af = self.aux_flag("0", a, &r, bits);
                    self.set_flag(Flag::Af, &af);
                }
                let src = FlagSource::Compare {
                    a: "0".to_string(),
                    b: a.to_string(),
                    diff: Some(r.clone()),
                    bits,
                };
                (r, src)
            }
        };
        self.set_zsp(&r, bits);
        self.known.flag_source = Some(source);
        r
    }

    /// Shifts and rotates. The count is masked to 5 bits (6 for 64-bit operands); a masked count
    /// of zero leaves all flags unchanged. OF is computed with the 1-bit formula for every count
    /// (it is architecturally undefined for counts > 1); AF is undefined and left unchanged.
    fn shift(&mut self, ins: &Instruction, kind: Shift) -> R<()> {
        let bits = self.op_bits(ins, 0)?;
        let t = ty(bits);
        let count_mask: u64 = if bits == 64 { 63 } else { 31 };
        let a = self.read_op(ins, 0, bits)?;

        // Native count width: `i32` for 32-bit shifts/rotates, `i{bits}` for 8/16-bit rotates,
        // `i64` for 64-bit and for 8/16-bit shifts (which widen to i64 so counts 8..31 never poison).
        let cnt_bits = match (kind, bits) {
            (Shift::Rol | Shift::Ror, b) => b,
            (_, 32) => 32,
            _ => 64,
        };
        let (count, imm_count) = match ins.op_kind(1) {
            OpKind::Register if ins.op_register(1) == Register::CL => {
                let rcx = self.load_gpr(integer::RCX)?;
                if let Some(v) = memory_access::const_addr(&rcx) {
                    let c = v & count_mask;
                    (c.to_string(), Some(c))
                } else {
                    let raw = if cnt_bits == 64 {
                        rcx
                    } else {
                        self.val(&format!("trunc i64 {rcx} to i{cnt_bits}"))
                    };
                    (
                        self.val(&format!("and i{cnt_bits} {raw}, {count_mask}")),
                        None,
                    )
                }
            }
            kind if is_immediate(kind) => match ins.try_immediate(1) {
                Ok(v) => {
                    let c = v & count_mask;
                    (c.to_string(), Some(c))
                }
                Err(_) => return unsupported("unreadable shift count"),
            },
            kind => return unsupported(format!("shift count operand {kind:?}")),
        };

        // A masked immediate count of zero leaves all flags unchanged (still write_op for 32-bit zext).
        if imm_count == Some(0) {
            return self.write_op(ins, 0, &a, bits);
        }

        let want_cf = self.flag_live(Flag::Cf);
        let want_of = self.flag_live(Flag::Of);
        let (r, cf, of): (String, Option<String>, Option<String>) = match kind {
            Shift::Rol | Shift::Ror => {
                // fshl/fshr take the amount modulo the width, like x86 rotates (after masking).
                let (name, support) = if kind == Shift::Rol {
                    ("fshl", Support::Fshl(bits))
                } else {
                    ("fshr", Support::Fshr(bits))
                };
                self.support.insert(support);
                let r = self.val(&format!(
                    "call {t} @llvm.{name}.{t}({t} {a}, {t} {a}, {t} {count})"
                ));
                if kind == Shift::Rol {
                    let cf =
                        (want_cf || want_of).then(|| self.val(&format!("trunc {t} {r} to i1")));
                    let of = if want_of {
                        let msb = self.msb(&r, bits);
                        Some(self.val(&format!("xor i1 {msb}, {}", cf.as_ref().unwrap())))
                    } else {
                        None
                    };
                    (r, cf.filter(|_| want_cf), of)
                } else {
                    let cf = want_cf.then(|| self.msb(&r, bits));
                    let of = if want_of {
                        let msb = self.msb(&r, bits);
                        let below = self.val(&format!("lshr {t} {r}, {}", bits - 2));
                        let below = self.val(&format!("trunc {t} {below} to i1"));
                        Some(self.val(&format!("xor i1 {msb}, {below}")))
                    } else {
                        None
                    };
                    (r, cf, of)
                }
            }
            _ => {
                let op = match kind {
                    Shift::Shl => "shl",
                    Shift::Shr => "lshr",
                    _ => "ashr",
                };
                if bits == 64 || bits == 32 {
                    // Masked counts are in 0..bits, so native `i32`/`i64` shifts never poison.
                    let w = u64::from(bits);
                    let m = w - 1;
                    let r = self.val(&format!("{op} {t} {a}, {count}"));
                    match kind {
                        Shift::Shl => {
                            let cf = (want_cf || want_of).then(|| match imm_count {
                                Some(1) => self.msb(&a, bits),
                                Some(c) => {
                                    let out = self.val(&format!("lshr {t} {a}, {}", (w - c) & m));
                                    self.val(&format!("trunc {t} {out} to i1"))
                                }
                                None => {
                                    let back = self.val(&format!("sub {t} {w}, {count}"));
                                    let back = self.val(&format!("and {t} {back}, {m}"));
                                    let out = self.val(&format!("lshr {t} {a}, {back}"));
                                    self.val(&format!("trunc {t} {out} to i1"))
                                }
                            });
                            let of = if want_of {
                                let msb = self.msb(&r, bits);
                                Some(self.val(&format!("xor i1 {msb}, {}", cf.as_ref().unwrap())))
                            } else {
                                None
                            };
                            (r, cf.filter(|_| want_cf), of)
                        }
                        _ => {
                            let cf = want_cf.then(|| match imm_count {
                                Some(1) => self.val(&format!("trunc {t} {a} to i1")),
                                Some(c) => {
                                    let out = self.val(&format!("{op} {t} {a}, {}", c - 1));
                                    self.val(&format!("trunc {t} {out} to i1"))
                                }
                                None => {
                                    let prev = self.val(&format!("add {t} {count}, {m}"));
                                    let prev = self.val(&format!("and {t} {prev}, {m}"));
                                    let out = self.val(&format!("{op} {t} {a}, {prev}"));
                                    self.val(&format!("trunc {t} {out} to i1"))
                                }
                            });
                            let of = if kind == Shift::Shr {
                                want_of.then(|| self.msb(&a, bits))
                            } else {
                                want_of.then(|| "false".to_string())
                            };
                            (r, cf, of)
                        }
                    }
                } else {
                    // 8/16-bit operands are shifted in 64 bits so masked counts up to 31 never poison.
                    let prev = if !want_cf || kind == Shift::Shl {
                        String::new()
                    } else {
                        match imm_count {
                            Some(c) => (c - 1).to_string(),
                            None => {
                                let prev = self.val(&format!("add i64 {count}, 63"));
                                self.val(&format!("and i64 {prev}, 63"))
                            }
                        }
                    };
                    let ext = if kind == Shift::Sar { "sext" } else { "zext" };
                    let wide = self.val(&format!("{ext} {t} {a} to i64"));
                    let res = self.val(&format!("{op} i64 {wide}, {count}"));
                    let r = self.val(&format!("trunc i64 {res} to {t}"));
                    match kind {
                        Shift::Shl => {
                            let cf = (want_cf || want_of).then(|| {
                                let out = self.val(&format!("lshr i64 {res}, {bits}"));
                                self.val(&format!("trunc i64 {out} to i1"))
                            });
                            let of = if want_of {
                                let msb = self.msb(&r, bits);
                                Some(self.val(&format!("xor i1 {msb}, {}", cf.as_ref().unwrap())))
                            } else {
                                None
                            };
                            (r, cf.filter(|_| want_cf), of)
                        }
                        _ => {
                            let cf = want_cf.then(|| {
                                let out = self.val(&format!("{op} i64 {wide}, {prev}"));
                                self.val(&format!("trunc i64 {out} to i1"))
                            });
                            let of = if kind == Shift::Shr {
                                want_of.then(|| self.msb(&a, bits))
                            } else {
                                want_of.then(|| "false".to_string())
                            };
                            (r, cf, of)
                        }
                    }
                }
            }
        };

        // Flags only change when the masked count is non-zero.
        let mut updates: Vec<(Flag, String)> = Vec::new();
        updates.extend(cf.map(|v| (Flag::Cf, v)));
        updates.extend(of.map(|v| (Flag::Of, v)));
        if !matches!(kind, Shift::Rol | Shift::Ror) {
            if self.flag_live(Flag::Zf) {
                let zf = self.val(&format!("icmp eq i{bits} {r}, 0"));
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
        if !matches!(kind, Shift::Rol | Shift::Ror) && imm_count.is_some() {
            self.known.flag_source = Some(FlagSource::ResultZeroSign {
                result: r.clone(),
                bits,
            });
        }
        self.write_op(ins, 0, &r, bits)
    }

    /// Full 64x64 -> 128-bit product as `(lo, hi)`; `signed` adjusts the high half.
    fn mul128(&mut self, a: &str, b: &str, signed: bool) -> (String, String) {
        if memory_access::const_addr(a) == Some(0) || memory_access::const_addr(b) == Some(0) {
            return ("0".to_string(), "0".to_string());
        }
        // `UMUL128_IR` places its inline-asm barrier on `%a1` (`a >> 32`); keeping a lift-time
        // constant in `%b` (e.g. MSVC's `mov rax, MAGIC; mul rcx`) lets LLVM constant-fold `%b0`
        // and `%b1`, cutting Wasm ops and `i64.mul` count in half.
        let (a, b) = if memory_access::const_addr(a).is_some()
            && memory_access::const_addr(b).is_none()
        {
            (b, a)
        } else {
            (a, b)
        };
        self.support.insert(Support::Umul128);
        let p = self.val(&format!(
            "call {{ i64, i64 }} @__playgta6_umul128(i64 {a}, i64 {b})"
        ));
        let lo = self.val(&format!("extractvalue {{ i64, i64 }} {p}, 0"));
        let mut hi = self.val(&format!("extractvalue {{ i64, i64 }} {p}, 1"));
        if signed {
            // signed_hi = unsigned_hi - (a < 0 ? b : 0) - (b < 0 ? a : 0)
            let sa = self.val(&format!("ashr i64 {a}, 63"));
            let fa = self.val(&format!("and i64 {sa}, {b}"));
            let sb = self.val(&format!("ashr i64 {b}, 63"));
            let fb = self.val(&format!("and i64 {sb}, {a}"));
            hi = self.val(&format!("sub i64 {hi}, {fa}"));
            hi = self.val(&format!("sub i64 {hi}, {fb}"));
        }
        (lo, hi)
    }

    /// `imul r, r/m` and `imul r, r/m, imm`: truncated product; CF = OF = signed overflow.
    fn imul_truncating(&mut self, ins: &Instruction) -> R<()> {
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
        if !self.flag_live(Flag::Cf) && !self.flag_live(Flag::Of) {
            let r = if bits == 64 {
                self.val(&format!("mul i64 {a}, {b}"))
            } else {
                self.val(&format!("mul {t} {a}, {b}"))
            };
            return self.write_op(ins, 0, &r, bits);
        }
        let (r, overflow) = if bits == 64 {
            // The truncated product is a plain `mul`: only CF/OF need the 128-bit high half, so
            // when they are dead (nearly always) the 32-bit-halves helper folds away entirely.
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
        // SF, ZF, AF and PF are undefined and left unchanged.
        self.set_flag(Flag::Cf, &overflow);
        self.set_flag(Flag::Of, &overflow);
        self.write_op(ins, 0, &r, bits)
    }

    /// One-operand `mul`/`imul`: `rdx:rax = rax * src` (AX = AL * src for bytes).
    fn mul_widening(&mut self, ins: &Instruction, signed: bool) -> R<()> {
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
        // SF, ZF, AF and PF are undefined and left unchanged.
        if let Some(overflow) = overflow {
            self.set_flag(Flag::Cf, &overflow);
            self.set_flag(Flag::Of, &overflow);
        }
        Ok(())
    }

    /// `div`/`idiv`: divides `rdx:rax` (AX for bytes) by the operand. Division by zero or a
    /// quotient that does not fit raises `#DE` through `__playgta6_divide_error`. All flags are
    /// undefined and left unchanged.
    fn divide(&mut self, ins: &Instruction, signed: bool) -> R<()> {
        let bits = self.op_bits(ins, 0)?;
        let t = ty(bits);
        let divisor = self.read_op(ins, 0, bits)?;
        let ext = if signed { "sext" } else { "zext" };

        // Fast paths for compiler-standard `xor edx, edx; div` and `cdq`/`cqo; idiv` (32/64-bit):
        // emit native `udiv`/`urem`/`sdiv`/`srem i{bits}` directly, avoiding 32->64 widening and
        // 64-bit `@__playgta6_{u,s}div128` calls (`sret` shadow-stack spills + 64-iter loops).
        if matches!(bits, 32 | 64) {
            let (qreg, rreg) = if bits == 32 {
                (Register::EAX, Register::EDX)
            } else {
                (Register::RAX, Register::RDX)
            };
            if !signed
                && self.known.gprs[integer::RDX]
                    .as_deref()
                    .and_then(memory_access::const_addr)
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
                            && self.known.gprs[integer::RDX].as_deref() == Some(rdx64.as_str())
                            && self.known.gprs[integer::RAX].as_deref() == Some(rax64.as_str())
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

        // 8/16/32-bit `div`/`idiv`: the 2N-bit dividend fits natively in `i64`, so lower straight
        // to Wasm64/native `udiv`/`sdiv`/`urem`/`srem i64` without calling the 128-bit helpers.
        if bits < 64 {
            let (dividend, hi_part) = if bits == 8 {
                let (ax, _) = self.read_reg(Register::AX)?;
                let div = self.val(&format!("{ext} i16 {ax} to i64"));
                let hi = (!signed).then(|| self.val(&format!("lshr i64 {div}, 8")));
                (div, hi)
            } else {
                let (hreg, lreg) = if bits == 16 {
                    (Register::DX, Register::AX)
                } else {
                    (Register::EDX, Register::EAX)
                };
                let (h, _) = self.read_reg(hreg)?;
                let (l, _) = self.read_reg(lreg)?;
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
                // `d == 0` OR `floor(dividend / d) >= 2^bits` iff `hi_part >=u d`.
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
                    // Guard LLVM/wasm `sdiv`/`srem` against `INT64_MIN / -1` (also #DE on x86).
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
            let (qreg, rreg) = match bits {
                8 => (Register::AL, Register::AH),
                16 => (Register::AX, Register::DX),
                _ => (Register::EAX, Register::EDX),
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

    // ----- control flow --------------------------------------------------------------------------

    /// Reads the 64-bit target of an indirect `call`/`jmp`.
    fn indirect_target(&mut self, ins: &Instruction) -> R<String> {
        if self.op_bits(ins, 0)? != 64 {
            return unsupported("indirect branch operand size");
        }
        self.read_op(ins, 0, 64)
    }

    fn call(&mut self, ins: &Instruction) -> R<()> {
        if !ins.is_call_near() && !ins.is_call_near_indirect() {
            return unsupported("far call");
        }
        let return_address = imm(ins.next_ip(), 64);
        let live_flags_after = ALL_FLAGS & !self.dead_flags;
        if ins.op_kind(0) == OpKind::NearBranch64 {
            let target = ins.near_branch_target();
            self.push(&return_address, 64)?;
            if self.faulting_calls.contains(&ins.ip()) {
                // The target is not executable guest code: the call faults on arrival. Code after
                // it is unreachable, so continue in a fresh block without predecessors.
                self.trap(target);
                let dead = self.new_label("dead");
                self.start_block(&dead);
                return Ok(());
            }
            let flags = self.flags_sync_out_by(target);
            let written_flags = self.flags_written_by(target);
            let out_state = self.state_sync_out_by(target);
            let in_state = self.state_written_by(target);
            self.sync_out_parts(out_state, flags);
            let pre_known_gprs = self.known.gprs.clone();
            let pre_known_flags = self.known.flags.clone();
            let pre_entry_facts = self.block_entry_facts;
            self.guest_call(&format!(
                "call void @{}(ptr %state)",
                function_symbol(target)
            ));
            self.callees.push(target);
            self.sync_in_parts(in_state, written_flags & live_flags_after);
            // `after_call` is dominated solely by the pre-call block: GPRs and flags the callee
            // never writes (other than `rsp`) still hold their pre-call SSA values and facts.
            for idx in 0..16 {
                if idx != RSP && self.regs[idx] && in_state & (1u64 << idx) == 0 {
                    self.known.gprs[idx] = pre_known_gprs[idx].clone();
                    self.block_entry_facts[idx] = pre_entry_facts[idx];
                }
            }
            for flag in FLAGS {
                let f = flag as usize;
                if self.flags[f] && written_flags & (1 << f) == 0 {
                    self.known.flags[f] = pre_known_flags[f].clone();
                }
            }
        } else {
            // The target is read before the return address is pushed (`call [rsp+8]`).
            let target = self.indirect_target(ins)?;
            self.push(&return_address, 64)?;
            self.sync_out();
            self.guest_call(&format!(
                "call void @{HELPER_INDIRECT_CALL}(ptr %state, i64 {target})"
            ));
            self.sync_in_parts(ALL_STATE, live_flags_after);
        }
        Ok(())
    }

    fn ret(&mut self, ins: &Instruction) -> R<()> {
        let extra = match ins.code() {
            Code::Retnq => 0,
            Code::Retnq_imm16 => u64::from(ins.immediate16()),
            _ => return unsupported("far or 16/32-bit return"),
        };
        // Pop the return address (not checked: the host call stack already knows where to return).
        let rsp = self.load_gpr(RSP)?;
        let new_rsp = self.val(&format!("add i64 {rsp}, {}", 8 + extra));
        self.store_gpr(RSP, &new_rsp)?;
        if self.share_ret {
            self.has_shared_ret = true;
            self.emit("br label %ret_epilog");
        } else {
            self.sync_out();
            self.emit("ret void");
        }
        Ok(())
    }

    /// `jmp reg` / `jmp [mem]`. A resolved switch (`Edge::JumpTable` successors) becomes an LLVM
    /// `switch` on the computed guest target over the case blocks of this function; every other
    /// value (an incomplete table proof, a patched table) and unresolved indirect jumps leave the
    /// function through the runtime dispatcher as a tail call.
    fn indirect_jump(&mut self, block: &cfg::BasicBlock, ins: &Instruction) -> R<()> {
        if !ins.is_jmp_near_indirect() {
            return unsupported("far or non-indirect jump");
        }
        let target = self.indirect_target(ins)?;
        let mut cases: Vec<(u64, bool)> = block
            .successors
            .iter()
            .filter_map(|edge| match *edge {
                Edge::JumpTable(case) => Some((case, false)),
                // A case that is another function's entry (function-pointer table).
                Edge::TailCall(case) if block.terminator == Terminator::IndirectJump => {
                    Some((case, true))
                }
                _ => None,
            })
            .collect();
        cases.sort_unstable();
        cases.dedup();
        if !cases.is_empty() {
            let mut arms = String::new();
            for (case, is_exit) in cases {
                if is_exit {
                    self.exits.push(case);
                    let _ = write!(arms, " i64 {}, label %{}", imm(case, 64), exit_label(case));
                    continue;
                }
                if !self.func.blocks.contains_key(&case) {
                    return Err(InsnError::Cfg(LiftError::UnknownSuccessor {
                        block: block.start,
                        target: case,
                    }));
                }
                let _ = write!(arms, " i64 {}, label %{}", imm(case, 64), block_label(case));
                self.edge_known.push((case, self.known.clone()));
            }
            let dispatch = self.new_label("dispatch");
            self.emit(&format!("switch i64 {target}, label %{dispatch} [{arms} ]"));
            self.start_block(&dispatch);
        }
        self.sync_out();
        self.emit(&format!(
            "tail call void @{HELPER_INDIRECT_CALL}(ptr %state, i64 {target})"
        ));
        self.emit("ret void");
        Ok(())
    }

    /// `jcc`: the condition comes from the instruction, the targets from the `[taken, not taken]`
    /// successor edges (either may be a tail call into another function).
    fn branch(&mut self, block: &cfg::BasicBlock, ins: &Instruction) -> R<()> {
        let c = if ins.is_jcc_short_or_near() {
            self.condition(ins.condition_code())?
        } else {
            // `loop*` / `jrcxz` / `jecxz` (anything else is reported as unsupported).
            self.counter_branch_condition(ins)?
        };
        let taken = self.successor(block, block.successors.first())?;
        let not_taken = self.successor(block, block.successors.get(1))?;
        self.emit(&format!("br i1 {c}, label %{taken}, label %{not_taken}"));
        Ok(())
    }
}

/// R71: intra-procedural forward dataflow analysis computing per-block entry [`GprFact`]s
/// (compile-time constants and bounded `entry_rsp`-relative stack offsets) over `func.blocks`.
///
/// Soundness invariants:
/// - `func.entry` starts with `RSP = StackOffset(0)` and all other GPRs `Bottom`.
/// - Every block in `func.resume_targets` starts with all GPRs `Bottom` (re-entry from unwind).
/// - Direct calls kill `RSP` and every GPR in `flag_summaries.written_by(target)` (or all GPRs
///   when summaries are absent); indirect calls and runtime helpers kill all GPRs.
/// - Every unmodelled instruction conservatively kills all GPRs reported written by `iced_x86`.
fn compute_block_entry_gpr_facts(
    func: &cfg::Function,
    flag_summaries: Option<&FlagSummaries>,
    faulting_calls: &BTreeSet<u64>,
) -> BTreeMap<u64, [GprFact; 16]> {
    let mut in_state: BTreeMap<u64, [GprFact; 16]> = func
        .blocks
        .keys()
        .map(|&va| (va, [GprFact::Top; 16]))
        .collect();
    let mut work: VecDeque<u64> = VecDeque::new();
    let mut queued: BTreeSet<u64> = BTreeSet::new();

    if let Some(entry_in) = in_state.get_mut(&func.entry) {
        *entry_in = [GprFact::Bottom; 16];
        entry_in[RSP] = GprFact::StackOffset(0);
        work.push_back(func.entry);
        queued.insert(func.entry);
    }
    for &resume_va in &func.resume_targets {
        if let Some(resume_in) = in_state.get_mut(&resume_va) {
            *resume_in = [GprFact::Bottom; 16];
            if queued.insert(resume_va) {
                work.push_back(resume_va);
            }
        }
    }

    let mut factory = InstructionInfoFactory::new();
    while let Some(va) = work.pop_front() {
        queued.remove(&va);
        let Some(block) = func.blocks.get(&va) else {
            continue;
        };
        let mut state = in_state[&va];
        let (body, transfer) = match block.instructions.split_last() {
            Some((last, rest)) if ends_in_transfer(block.terminator) => (rest, Some(last)),
            _ => (&block.instructions[..], None),
        };
        let mut falls_through = true;
        for ins in body {
            if !transfer_gpr_instruction(&mut state, ins, flag_summaries, faulting_calls, &mut factory)
            {
                falls_through = false;
                break;
            }
        }
        if !falls_through {
            continue;
        }
        if let (Terminator::Branch, Some(term)) = (block.terminator, transfer)
            && matches!(
                term.code(),
                Code::Loop_rel8_64_RCX | Code::Loope_rel8_64_RCX | Code::Loopne_rel8_64_RCX
            )
        {
            state[integer::RCX] = GprFact::Bottom;
        }
        for target in block.successors.iter().filter_map(Edge::intra_target) {
            if let Some(succ_in) = in_state.get_mut(&target) {
                let mut changed = false;
                for i in 0..16 {
                    let joined = succ_in[i].join(state[i]);
                    if joined != succ_in[i] {
                        succ_in[i] = joined;
                        changed = true;
                    }
                }
                if changed && queued.insert(target) {
                    work.push_back(target);
                }
            }
        }
    }
    in_state
}

fn add_stack_offset(base: i64, delta: i64) -> GprFact {
    base.checked_add(delta)
        .filter(|total| total.abs() <= memory_access::STACK_FRAME_MAX_DISP)
        .map_or(GprFact::Bottom, GprFact::StackOffset)
}

fn add_fact_offset(fact: GprFact, delta: i64) -> GprFact {
    match fact {
        GprFact::Const(c) => GprFact::Const(c.wrapping_add(delta as u64)),
        GprFact::StackOffset(d) => add_stack_offset(d, delta),
        GprFact::StackFrame if delta == 0 => GprFact::StackFrame,
        _ => GprFact::Bottom,
    }
}

/// True if `m` is a power-of-two stack-alignment mask `-(1 << k)` with `1 <= k <= 12` (<= 4 KiB).
fn align_mask_slack(m: u64) -> Option<i64> {
    let slack = (!m) as i64;
    if slack > 0 && slack <= 4095 && ((slack as u64) & (slack as u64 + 1)) == 0 {
        Some(slack)
    } else {
        None
    }
}

fn read_reg_const(state: &[GprFact; 16], reg: Register) -> Option<u64> {
    match reg_ref(reg).ok()? {
        RegRef::Low(idx, bits) => match state[idx] {
            GprFact::Const(c) => Some(c & mask(bits)),
            _ => None,
        },
        RegRef::High8(idx) => match state[idx] {
            GprFact::Const(c) => Some((c >> 8) & 0xFF),
            _ => None,
        },
    }
}

fn write_gpr_fact(state: &mut [GprFact; 16], reg: Register, val: GprFact) {
    let Ok(r) = reg_ref(reg) else {
        return;
    };
    match r {
        RegRef::Low(idx, 64) => state[idx] = val,
        RegRef::Low(idx, 32) => {
            state[idx] = match val {
                GprFact::Const(c) => GprFact::Const(c & mask(32)),
                _ => GprFact::Bottom,
            };
        }
        RegRef::Low(idx, bits) => {
            state[idx] = match (state[idx], val) {
                (GprFact::Const(old), GprFact::Const(new)) => {
                    let m = mask(bits);
                    GprFact::Const((old & !m) | (new & m))
                }
                _ => GprFact::Bottom,
            };
        }
        RegRef::High8(idx) => {
            state[idx] = match (state[idx], val) {
                (GprFact::Const(old), GprFact::Const(new)) => {
                    GprFact::Const((old & !0xFF00) | ((new & 0xFF) << 8))
                }
                _ => GprFact::Bottom,
            };
        }
    }
}

fn kill_written_gprs(
    state: &mut [GprFact; 16],
    ins: &Instruction,
    factory: &mut InstructionInfoFactory,
) {
    let info = factory.info(ins);
    for ur in info.used_registers() {
        if matches!(
            ur.access(),
            OpAccess::Write
                | OpAccess::CondWrite
                | OpAccess::ReadWrite
                | OpAccess::ReadCondWrite
        ) {
            let r = ur.register();
            if r.is_gpr() && r.full_register().is_gpr64() {
                state[r.full_register().number()] = GprFact::Bottom;
            }
        }
    }
    for i in 0..ins.op_count() {
        if ins.op_kind(i) == OpKind::Register
            && matches!(
                info.op_access(i),
                OpAccess::Write
                    | OpAccess::CondWrite
                    | OpAccess::ReadWrite
                    | OpAccess::ReadCondWrite
            )
        {
            let r = ins.op_register(i);
            if r.is_gpr() && r.full_register().is_gpr64() {
                state[r.full_register().number()] = GprFact::Bottom;
            }
        }
    }
}

fn sign_extend_u64(val: u64, from_bits: u32) -> u64 {
    if from_bits >= 64 {
        return val;
    }
    let shift = 64 - from_bits;
    (((val << shift) as i64) >> shift) as u64
}

fn is_cmovcc_mnemonic(m: Mnemonic) -> bool {
    use Mnemonic as M;
    matches!(
        m,
        M::Cmovo
            | M::Cmovno
            | M::Cmovb
            | M::Cmovae
            | M::Cmove
            | M::Cmovne
            | M::Cmovbe
            | M::Cmova
            | M::Cmovs
            | M::Cmovns
            | M::Cmovp
            | M::Cmovnp
            | M::Cmovl
            | M::Cmovge
            | M::Cmovle
            | M::Cmovg
    )
}

/// Applies one body instruction to `state`. Returns `false` only when the instruction is known to
/// trap unconditionally (a direct call into unmapped/non-executable memory).
fn transfer_gpr_instruction(
    state: &mut [GprFact; 16],
    ins: &Instruction,
    flag_summaries: Option<&FlagSummaries>,
    faulting_calls: &BTreeSet<u64>,
    factory: &mut InstructionInfoFactory,
) -> bool {
    use Mnemonic as M;
    if ins.code() == Code::INVALID {
        *state = [GprFact::Bottom; 16];
        return true;
    }
    let m = ins.mnemonic();
    if m == M::Call {
        if ins.op_kind(0) == OpKind::NearBranch64 {
            if faulting_calls.contains(&ins.ip()) {
                return false;
            }
            let target = ins.near_branch_target();
            if let Some(summaries) = flag_summaries {
                let written = summaries.written_by(target);
                for (idx, slot) in state.iter_mut().enumerate() {
                    if idx == RSP || (written & (1u64 << idx)) != 0 {
                        *slot = GprFact::Bottom;
                    }
                }
            } else {
                *state = [GprFact::Bottom; 16];
            }
        } else {
            *state = [GprFact::Bottom; 16];
        }
        return true;
    }
    if matches!(m, M::Cpuid | M::Rdtsc | M::Rdtscp | M::Xgetbv) {
        *state = [GprFact::Bottom; 16];
        return true;
    }
    if ins.has_lock_prefix() || ins.is_string_instruction() {
        kill_written_gprs(state, ins, factory);
        return true;
    }
    match m {
        M::Nop
        | M::Endbr64
        | M::Cmp
        | M::Test
        | M::Bt
        | M::Mfence
        | M::Lfence
        | M::Sfence
        | M::Pause
        | M::Prefetchnta
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
        | M::Rdsspd
        | M::Rdsspq
        | M::Monitorx
        | M::Mwaitx
        | M::Cld
        | M::Std
        | M::Clc
        | M::Stc
        | M::Cmc
        | M::Fnclex
        | M::Fclex
        | M::Fninit
        | M::Finit
        | M::Wait
        | M::Fnop
        | M::Fldcw
        | M::Clzero => {}
        M::Mov => {
            if ins.op_kind(0) == OpKind::Register {
                let r0 = ins.op_register(0);
                let val = match ins.op_kind(1) {
                    kind if is_immediate(kind) => ins
                        .try_immediate(1)
                        .ok()
                        .map_or(GprFact::Bottom, GprFact::Const),
                    OpKind::Register => {
                        let r1 = ins.op_register(1);
                        if r0.is_gpr64() && r1.is_gpr64() {
                            state[r1.number()]
                        } else {
                            read_reg_const(state, r1).map_or(GprFact::Bottom, GprFact::Const)
                        }
                    }
                    _ => GprFact::Bottom,
                };
                write_gpr_fact(state, r0, val);
            }
        }
        M::Movzx | M::Movsx | M::Movsxd => {
            if ins.op_kind(0) == OpKind::Register {
                let r0 = ins.op_register(0);
                let val = if ins.op_kind(1) == OpKind::Register {
                    let r1 = ins.op_register(1);
                    let src_bits = (r1.size() * 8) as u32;
                    read_reg_const(state, r1).map_or(GprFact::Bottom, |c| {
                        let ext = if m == M::Movzx {
                            c & mask(src_bits)
                        } else {
                            sign_extend_u64(c & mask(src_bits), src_bits)
                        };
                        GprFact::Const(ext)
                    })
                } else {
                    GprFact::Bottom
                };
                write_gpr_fact(state, r0, val);
            }
        }
        M::Lea => {
            if ins.op_kind(0) == OpKind::Register && ins.op_kind(1) == OpKind::Memory {
                let base = ins.memory_base();
                let index = ins.memory_index();
                let disp = ins.memory_displacement64();
                let addr32 = base.is_gpr32() || index.is_gpr32();
                let res = if base == Register::EIP {
                    GprFact::Bottom
                } else if base == Register::RIP && index == Register::None {
                    GprFact::Const(disp)
                } else if base == Register::None && index == Register::None {
                    GprFact::Const(if addr32 { disp & mask(32) } else { disp })
                } else {
                    let base_fact = match base {
                        Register::None | Register::RIP => GprFact::Const(0),
                        r if r.is_gpr64() => state[r.number()],
                        r if r.is_gpr32() => match state[r.full_register().number()] {
                            GprFact::Const(c) => GprFact::Const(c & mask(32)),
                            _ => GprFact::Bottom,
                        },
                        _ => GprFact::Bottom,
                    };
                    let idx_fact = match index {
                        Register::None => GprFact::Const(0),
                        r if r.is_gpr64() => match state[r.number()] {
                            GprFact::Const(c) => {
                                GprFact::Const(c.wrapping_mul(u64::from(ins.memory_index_scale())))
                            }
                            _ => GprFact::Bottom,
                        },
                        r if r.is_gpr32() => match state[r.full_register().number()] {
                            GprFact::Const(c) => GprFact::Const(
                                (c & mask(32)).wrapping_mul(u64::from(ins.memory_index_scale())),
                            ),
                            _ => GprFact::Bottom,
                        },
                        _ => GprFact::Bottom,
                    };
                    match (base_fact, idx_fact) {
                        (GprFact::Const(a), GprFact::Const(b)) => {
                            let sum = a.wrapping_add(b).wrapping_add(disp);
                            GprFact::Const(if addr32 { sum & mask(32) } else { sum })
                        }
                        (GprFact::StackOffset(d), GprFact::Const(b)) if !addr32 => {
                            add_stack_offset(d, b.wrapping_add(disp) as i64)
                        }
                        (GprFact::StackFrame, GprFact::Const(b))
                            if !addr32 && b.wrapping_add(disp) == 0 =>
                        {
                            GprFact::StackFrame
                        }
                        _ => GprFact::Bottom,
                    }
                };
                write_gpr_fact(state, ins.op_register(0), res);
            } else {
                kill_written_gprs(state, ins, factory);
            }
        }
        M::Xor | M::Sub
            if ins.op_count() == 2
                && ins.op_kind(0) == OpKind::Register
                && ins.op_kind(1) == OpKind::Register
                && ins.op_register(0) == ins.op_register(1) =>
        {
            write_gpr_fact(state, ins.op_register(0), GprFact::Const(0));
        }
        M::Add | M::Sub | M::And | M::Or | M::Xor => {
            if ins.op_kind(0) == OpKind::Register {
                let r0 = ins.op_register(0);
                let bits = (r0.size() * 8) as u32;
                let a = if r0.is_gpr64() {
                    state[r0.number()]
                } else {
                    read_reg_const(state, r0).map_or(GprFact::Bottom, GprFact::Const)
                };
                let b = match ins.op_kind(1) {
                    kind if is_immediate(kind) => ins
                        .try_immediate(1)
                        .ok()
                        .map_or(GprFact::Bottom, |v| GprFact::Const(v & mask(bits))),
                    OpKind::Register => {
                        let r1 = ins.op_register(1);
                        if r0.is_gpr64() && r1.is_gpr64() {
                            state[r1.number()]
                        } else {
                            read_reg_const(state, r1).map_or(GprFact::Bottom, GprFact::Const)
                        }
                    }
                    _ => GprFact::Bottom,
                };
                let res = match m {
                    M::Add => match (a, b) {
                        (GprFact::Const(x), GprFact::Const(y)) => {
                            GprFact::Const(x.wrapping_add(y) & mask(bits))
                        }
                        (GprFact::StackOffset(d), GprFact::Const(y))
                        | (GprFact::Const(y), GprFact::StackOffset(d))
                            if bits == 64 =>
                        {
                            add_stack_offset(d, y as i64)
                        }
                        (GprFact::StackFrame, GprFact::Const(0))
                        | (GprFact::Const(0), GprFact::StackFrame)
                            if bits == 64 =>
                        {
                            GprFact::StackFrame
                        }
                        _ => GprFact::Bottom,
                    },
                    M::Sub => match (a, b) {
                        (GprFact::Const(x), GprFact::Const(y)) => {
                            GprFact::Const(x.wrapping_sub(y) & mask(bits))
                        }
                        (GprFact::StackOffset(d), GprFact::Const(y)) if bits == 64 => d
                            .checked_sub(y as i64)
                            .filter(|t| t.abs() <= memory_access::STACK_FRAME_MAX_DISP)
                            .map_or(GprFact::Bottom, GprFact::StackOffset),
                        (GprFact::StackOffset(d1), GprFact::StackOffset(d2)) if bits == 64 => {
                            GprFact::Const(d1.wrapping_sub(d2) as u64)
                        }
                        (GprFact::StackFrame, GprFact::Const(0)) if bits == 64 => {
                            GprFact::StackFrame
                        }
                        _ => GprFact::Bottom,
                    },
                    M::And => match (a, b) {
                        (GprFact::Const(x), GprFact::Const(y)) => {
                            GprFact::Const((x & y) & mask(bits))
                        }
                        (GprFact::StackOffset(d), GprFact::Const(y))
                            if bits == 64
                                && align_mask_slack(y).is_some_and(|slack| {
                                    d.abs()
                                        .checked_add(slack)
                                        .is_some_and(|t| t <= memory_access::STACK_FRAME_HALF)
                                }) =>
                        {
                            GprFact::StackFrame
                        }
                        _ => GprFact::Bottom,
                    },
                    M::Or => match (a, b) {
                        (GprFact::Const(x), GprFact::Const(y)) => {
                            GprFact::Const((x | y) & mask(bits))
                        }
                        _ => GprFact::Bottom,
                    },
                    M::Xor => match (a, b) {
                        (GprFact::Const(x), GprFact::Const(y)) => {
                            GprFact::Const((x ^ y) & mask(bits))
                        }
                        _ => GprFact::Bottom,
                    },
                    _ => GprFact::Bottom,
                };
                write_gpr_fact(state, r0, res);
            }
        }
        M::Inc | M::Dec | M::Neg | M::Not => {
            if ins.op_kind(0) == OpKind::Register {
                let r0 = ins.op_register(0);
                let bits = (r0.size() * 8) as u32;
                let a = if r0.is_gpr64() {
                    state[r0.number()]
                } else {
                    read_reg_const(state, r0).map_or(GprFact::Bottom, GprFact::Const)
                };
                let res = match (m, a) {
                    (M::Inc, GprFact::Const(x)) => GprFact::Const(x.wrapping_add(1) & mask(bits)),
                    (M::Dec, GprFact::Const(x)) => GprFact::Const(x.wrapping_sub(1) & mask(bits)),
                    (M::Inc, GprFact::StackOffset(d)) if bits == 64 => add_stack_offset(d, 1),
                    (M::Dec, GprFact::StackOffset(d)) if bits == 64 => add_stack_offset(d, -1),
                    (M::Neg, GprFact::Const(x)) => GprFact::Const(0u64.wrapping_sub(x) & mask(bits)),
                    (M::Not, GprFact::Const(x)) => GprFact::Const((!x) & mask(bits)),
                    _ => GprFact::Bottom,
                };
                write_gpr_fact(state, r0, res);
            }
        }
        M::Shl | M::Sal | M::Shr | M::Sar => {
            if ins.op_kind(0) == OpKind::Register {
                let r0 = ins.op_register(0);
                let bits = (r0.size() * 8) as u32;
                let count_mask = if bits == 64 { 63 } else { 31 };
                let res = if is_immediate(ins.op_kind(1))
                    && let Ok(raw) = ins.try_immediate(1)
                    && let Some(c) = read_reg_const(state, r0)
                {
                    let amt = (raw & count_mask) as u32;
                    let v = match m {
                        M::Shl | M::Sal => {
                            if amt >= bits {
                                0
                            } else {
                                (c << amt) & mask(bits)
                            }
                        }
                        M::Shr => {
                            if amt >= bits {
                                0
                            } else {
                                (c & mask(bits)) >> amt
                            }
                        }
                        M::Sar => {
                            let sx = sign_extend_u64(c & mask(bits), bits) as i64;
                            ((sx >> amt.min(63)) as u64) & mask(bits)
                        }
                        _ => unreachable!(),
                    };
                    GprFact::Const(v)
                } else {
                    GprFact::Bottom
                };
                write_gpr_fact(state, r0, res);
            }
        }
        M::Push => {
            let inc = ins.stack_pointer_increment();
            state[RSP] = if matches!(inc, -8 | -2) {
                add_fact_offset(state[RSP], i64::from(inc))
            } else {
                GprFact::Bottom
            };
        }
        M::Pop => {
            let inc = ins.stack_pointer_increment();
            state[RSP] = if matches!(inc, 8 | 2) {
                add_fact_offset(state[RSP], i64::from(inc))
            } else {
                GprFact::Bottom
            };
            if ins.op_kind(0) == OpKind::Register {
                write_gpr_fact(state, ins.op_register(0), GprFact::Bottom);
            }
        }
        M::Pushfq => {
            state[RSP] = add_fact_offset(state[RSP], -8);
        }
        M::Popfq => {
            state[RSP] = add_fact_offset(state[RSP], 8);
        }
        M::Enter if ins.code() == Code::Enterq_imm16_imm8 && (ins.immediate8_2nd() & 31) == 0 => {
            let frame = add_fact_offset(state[RSP], -8);
            state[RBP] = frame;
            state[RSP] = add_fact_offset(frame, -i64::from(ins.immediate16()));
        }
        M::Leave if ins.code() == Code::Leaveq => {
            state[RSP] = add_fact_offset(state[RBP], 8);
            state[RBP] = GprFact::Bottom;
        }
        M::Xchg if ins.op_kind(0) == OpKind::Register && ins.op_kind(1) == OpKind::Register => {
            let r0 = ins.op_register(0);
            let r1 = ins.op_register(1);
            if r0.is_gpr64() && r1.is_gpr64() {
                state.swap(r0.number(), r1.number());
            } else {
                kill_written_gprs(state, ins, factory);
            }
        }
        cm if is_cmovcc_mnemonic(cm) && ins.op_kind(0) == OpKind::Register => {
            let r0 = ins.op_register(0);
            if ins.op_kind(1) == OpKind::Register {
                let r1 = ins.op_register(1);
                if r0 == r1 {
                    if r0.is_gpr32() {
                        let idx = r0.full_register().number();
                        state[idx] = match state[idx] {
                            GprFact::Const(c) => GprFact::Const(c & mask(32)),
                            _ => GprFact::Bottom,
                        };
                    }
                } else if r0.is_gpr64() && r1.is_gpr64() {
                    let joined = state[r0.number()].join(state[r1.number()]);
                    state[r0.number()] = joined;
                } else if r0.is_gpr32() && r1.is_gpr32() {
                    let a = match state[r0.full_register().number()] {
                        GprFact::Const(c) => GprFact::Const(c & mask(32)),
                        _ => GprFact::Bottom,
                    };
                    let b = match state[r1.full_register().number()] {
                        GprFact::Const(c) => GprFact::Const(c & mask(32)),
                        _ => GprFact::Bottom,
                    };
                    state[r0.full_register().number()] = a.join(b);
                } else {
                    write_gpr_fact(state, r0, GprFact::Bottom);
                }
            } else {
                write_gpr_fact(state, r0, GprFact::Bottom);
            }
        }
        _ => {
            kill_written_gprs(state, ins, factory);
        }
    }
    true
}

/// R65a: for a modelled non-trapping instruction (`flag_liveness::flag_effect(ins).is_some()`),
/// returns `(reads, full_writes)` bitmasks over GPRs `rax..r15` (`0..15`):
/// - `reads`: superset of every GPR whose `%gpr` alloca lifting `ins` may load (including memory
///   base/index registers, implicit sources, and partial 8/16-bit or conditional destinations).
/// - `full_writes`: subset of every GPR (excluding `rsp`) whose full 64-bit `%gpr` alloca lifting
///   `ins` unconditionally overwrites (64-bit writes and zero-extending 32-bit writes) without
///   loading `%gpr` first.
pub(super) fn gpr_rw_masks(ins: &Instruction, info: &iced_x86::InstructionInfo) -> (u16, u16) {
    use Mnemonic as M;
    let rsp_bit = 1u16 << RSP;
    // Register zeroing/borrow-mask idioms (`xor`/`sub`/`sbb r32/r64, r32/r64`) write directly
    // in `alu()` without loading the register. 8/16-bit forms go through `merge_gpr` and
    // therefore read the full 64-bit register.
    if matches!(ins.mnemonic(), M::Xor | M::Sub | M::Sbb)
        && ins.op_count() == 2
        && ins.op_kind(0) == OpKind::Register
        && ins.op_kind(1) == OpKind::Register
        && ins.op_register(0) == ins.op_register(1)
    {
        let r = ins.op_register(0);
        if r.is_gpr() && r.full_register().is_gpr64() {
            let bit = 1u16 << r.full_register().number();
            return if r.is_gpr64() || r.is_gpr32() {
                (0, bit & !rsp_bit)
            } else {
                (bit, 0)
            };
        }
    }

    let mut reads = 0u16;
    for ur in info.used_registers() {
        let r = ur.register();
        if r.is_gpr() && r.full_register().is_gpr64() {
            let bit = 1u16 << r.full_register().number();
            let access = ur.access();
            let is_read = matches!(
                access,
                OpAccess::Read
                    | OpAccess::CondRead
                    | OpAccess::ReadWrite
                    | OpAccess::ReadCondWrite
                    | OpAccess::CondWrite
            );
            let is_partial_write =
                access == OpAccess::Write && !(r.is_gpr64() || r.is_gpr32());
            if is_read || is_partial_write {
                reads |= bit;
            }
        }
    }
    for mem_reg in [ins.memory_base(), ins.memory_index()] {
        if mem_reg.is_gpr() && mem_reg.full_register().is_gpr64() {
            reads |= 1u16 << mem_reg.full_register().number();
        }
    }
    for i in 1..ins.op_count() {
        if ins.op_kind(i) == OpKind::Register {
            let r = ins.op_register(i);
            if r.is_gpr()
                && r.full_register().is_gpr64()
                && !matches!(info.op_access(i), OpAccess::Write | OpAccess::None)
            {
                reads |= 1u16 << r.full_register().number();
            }
        }
    }

    let mut full_writes = 0u16;
    let is_unconditional_op0_full_write = matches!(
        ins.mnemonic(),
        M::Mov
            | M::Movzx
            | M::Movsx
            | M::Movsxd
            | M::Lea
            | M::Movbe
            | M::Pop
            | M::Shlx
            | M::Shrx
            | M::Sarx
            | M::Rorx
            | M::Mulx
            | M::Andn
            | M::Blsi
            | M::Blsr
            | M::Blsmsk
            | M::Bzhi
            | M::Bextr
            | M::Tzcnt
            | M::Lzcnt
            | M::Popcnt
            | M::Movd
            | M::Vmovd
            | M::Movq
            | M::Vmovq
            | M::Movmskps
            | M::Vmovmskps
            | M::Movmskpd
            | M::Vmovmskpd
            | M::Pmovmskb
            | M::Vpmovmskb
            | M::Cvtss2si
            | M::Vcvtss2si
            | M::Cvttss2si
            | M::Vcvttss2si
            | M::Cvtsd2si
            | M::Vcvtsd2si
            | M::Cvttsd2si
            | M::Vcvttsd2si
            | M::Pextrd
            | M::Vpextrd
            | M::Pextrq
            | M::Vpextrq
    ) || (ins.mnemonic() == M::Imul && ins.op_count() == 3);

    if is_unconditional_op0_full_write
        && ins.op_count() > 0
        && ins.op_kind(0) == OpKind::Register
        && info.op_access(0) == OpAccess::Write
    {
        let r0 = ins.op_register(0);
        if r0.is_gpr() && (r0.is_gpr64() || r0.is_gpr32()) {
            let bit = 1u16 << r0.full_register().number();
            if reads & bit == 0 {
                full_writes |= bit;
            }
        }
    }
    if ins.mnemonic() == M::Mulx
        && ins.op_count() >= 2
        && ins.op_kind(1) == OpKind::Register
        && info.op_access(1) == OpAccess::Write
    {
        let r1 = ins.op_register(1);
        if r1.is_gpr() && (r1.is_gpr64() || r1.is_gpr32()) {
            let bit = 1u16 << r1.full_register().number();
            if reads & bit == 0 {
                full_writes |= bit;
            }
        }
    }
    if matches!(ins.mnemonic(), M::Cdq | M::Cqo) {
        let rdx_bit = 1u16 << 2;
        if reads & rdx_bit == 0 {
            full_writes |= rdx_bit;
        }
    }
    // Any other GPR destination (e.g. `bsf`/`bsr`, `cmovcc`, 8/16-bit writes) is conservatively
    // treated as a read so a later full write cannot bypass an earlier partial/conditional access.
    if ins.op_count() > 0 && ins.op_kind(0) == OpKind::Register {
        let r0 = ins.op_register(0);
        if r0.is_gpr() && r0.full_register().is_gpr64() {
            let bit = 1u16 << r0.full_register().number();
            if full_writes & bit == 0 {
                reads |= bit;
            }
        }
    }
    (reads, full_writes & !rsp_bit)
}

/// True if `ins` cannot appear inside a coalesced window burst (control flow, calls, atomics,
/// fences, trapping/helper instructions, string/stack/masked-vector ops, or bit-test offsets).
fn is_burst_barrier(ins: &Instruction) -> bool {
    use Mnemonic as M;
    if ins.code() == Code::INVALID
        || ins.has_lock_prefix()
        || ins.is_string_instruction()
        || is_control_transfer(ins)
        || ins.is_vsib()
    {
        return true;
    }
    if ins.mnemonic() == M::Xchg
        && (ins.op_kind(0) == OpKind::Memory || ins.op_kind(1) == OpKind::Memory)
    {
        return true;
    }
    matches!(
        ins.mnemonic(),
        M::Call
            | M::Div
            | M::Idiv
            | M::Cmpxchg
            | M::Cmpxchg8b
            | M::Cmpxchg16b
            | M::Bt
            | M::Bts
            | M::Btr
            | M::Btc
            | M::Push
            | M::Pop
            | M::Pushfq
            | M::Popfq
            | M::Pushf
            | M::Popf
            | M::Enter
            | M::Leave
            | M::Xlatb
            | M::Clzero
            | M::Cpuid
            | M::Rdtsc
            | M::Rdtscp
            | M::Xgetbv
            | M::Mfence
            | M::Lfence
            | M::Sfence
            | M::Pause
            | M::Ldmxcsr
            | M::Vldmxcsr
            | M::Stmxcsr
            | M::Vstmxcsr
            | M::Vmaskmovps
            | M::Vmaskmovpd
            | M::Vpmaskmovd
            | M::Vpmaskmovq
            | M::Maskmovdqu
            | M::Emms
            | M::Int3
            | M::Ud2
            | M::Hlt
    )
}

/// If `ins` performs a plain `[base64 + disp]` or indexed `[base64 + idx64*scale + disp]` memory
/// access (no `fs:`/`gs:`, not plain `rsp` when `stack_flat` is enabled, fixed byte width 1..=32),
/// returns `(burst_key, disp, bytes, is_write, is_simd_mem)`.
fn plain_burst_mem_access(
    ins: &Instruction,
    info: &iced_x86::InstructionInfo,
    stack_flat: bool,
) -> Option<(memory_access::BurstKey, i64, usize, bool, bool)> {
    use Mnemonic as M;
    if matches!(
        ins.mnemonic(),
        M::Lea
            | M::Nop
            | M::Endbr64
            | M::Prefetchnta
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
    ) {
        return None;
    }
    let has_mem_op = (0..ins.op_count()).any(|i| ins.op_kind(i) == OpKind::Memory);
    if !has_mem_op {
        return None;
    }
    let base = ins.memory_base();
    if !base.is_gpr64() {
        return None;
    }
    let index_reg = ins.memory_index();
    let index = if index_reg == Register::None {
        None
    } else if index_reg.is_gpr64() {
        let scale = ins.memory_index_scale();
        if !matches!(scale, 1 | 2 | 4 | 8) {
            return None;
        }
        Some((index_reg.number(), scale))
    } else {
        return None;
    };
    if matches!(ins.memory_segment(), Register::FS | Register::GS) {
        return None;
    }
    let reg = base.number();
    // Only plain `[rsp + disp]` (no index) is guaranteed stack-flat.
    if stack_flat && index.is_none() && reg == RSP {
        return None;
    }
    let bytes = ins.memory_size().size();
    if bytes == 0 || bytes > 32 {
        return None;
    }
    let is_write = info.used_memory().iter().any(|m| {
        matches!(
            m.access(),
            OpAccess::Write | OpAccess::CondWrite | OpAccess::ReadWrite | OpAccess::ReadCondWrite
        )
    });
    let is_simd_mem = bytes >= 16 || simd::is_simd(ins);
    let key = memory_access::BurstKey { base: reg, index };
    Some((
        key,
        ins.memory_displacement64() as i64,
        bytes,
        is_write,
        is_simd_mem,
    ))
}

/// `{lo, hi} = a * b` (unsigned 64x64 -> 128) built from 32-bit halves, so no `i128` multiply
/// (and therefore no compiler-rt libcall) is needed on wasm64. The empty inline-asm copy of
/// `a`'s high half hides the pattern from LLVM's multiply-high recognition (LLVM 20+), which
/// would otherwise turn a high-half-only use (`mul` for a division by a constant, `mulx` with
/// equal destinations) back into an `i128` multiply, i.e. a `__multi3` import. The copy is
/// `readnone`, so it folds away with the rest of the helper when a caller drops both halves
/// (a truncating `imul` whose CF/OF are dead).
pub(super) const UMUL128_IR: &str = "\
define internal { i64, i64 } @__playgta6_umul128(i64 %a, i64 %b) unnamed_addr #0 {
entry:
  %a0 = and i64 %a, 4294967295
  %a1.raw = lshr i64 %a, 32
  %a1 = call i64 asm \"\", \"=r,0\"(i64 %a1.raw) nounwind readnone willreturn
  %b0 = and i64 %b, 4294967295
  %b1 = lshr i64 %b, 32
  %p00 = mul i64 %a0, %b0
  %p01 = mul i64 %a0, %b1
  %p10 = mul i64 %a1, %b0
  %p11 = mul i64 %a1, %b1
  %p00h = lshr i64 %p00, 32
  %p01l = and i64 %p01, 4294967295
  %p10l = and i64 %p10, 4294967295
  %mid0 = add i64 %p00h, %p01l
  %mid = add i64 %mid0, %p10l
  %midl = shl i64 %mid, 32
  %p00l = and i64 %p00, 4294967295
  %lo = or i64 %midl, %p00l
  %p01h = lshr i64 %p01, 32
  %p10h = lshr i64 %p10, 32
  %midh = lshr i64 %mid, 32
  %hi0 = add i64 %p11, %p01h
  %hi1 = add i64 %hi0, %p10h
  %hi = add i64 %hi1, %midh
  %r0 = insertvalue { i64, i64 } poison, i64 %lo, 0
  %r1 = insertvalue { i64, i64 } %r0, i64 %hi, 1
  ret { i64, i64 } %r1
}
";

/// `{q, r, fault} = (hi:lo) / d` (unsigned 128 / 64). `fault` is set when `d == 0` or the
/// quotient does not fit in 64 bits (`hi >= d`). Restoring division, no `i128` libcalls.
pub(super) const UDIV128_IR: &str = "\
define internal { i64, i64, i1 } @__playgta6_udiv128(i64 %hi, i64 %lo, i64 %d) unnamed_addr #0 {
entry:
  %fault = icmp uge i64 %hi, %d
  br i1 %fault, label %fail, label %ok
fail:
  ret { i64, i64, i1 } { i64 0, i64 0, i1 true }
ok:
  %small = icmp eq i64 %hi, 0
  br i1 %small, label %fast, label %loop
fast:
  %fq = udiv i64 %lo, %d
  %fr = urem i64 %lo, %d
  %f0 = insertvalue { i64, i64, i1 } { i64 0, i64 0, i1 false }, i64 %fq, 0
  %f1 = insertvalue { i64, i64, i1 } %f0, i64 %fr, 1
  ret { i64, i64, i1 } %f1
loop:
  %i = phi i32 [ 0, %ok ], [ %i.next, %loop ]
  %h = phi i64 [ %hi, %ok ], [ %h.next, %loop ]
  %l = phi i64 [ %lo, %ok ], [ %l.next, %loop ]
  %carry = icmp slt i64 %h, 0
  %h.shl = shl i64 %h, 1
  %l.top = lshr i64 %l, 63
  %h2 = or i64 %h.shl, %l.top
  %l2 = shl i64 %l, 1
  %ge = icmp uge i64 %h2, %d
  %take = or i1 %carry, %ge
  %h.sub = sub i64 %h2, %d
  %h.next = select i1 %take, i64 %h.sub, i64 %h2
  %bit = zext i1 %take to i64
  %l.next = or i64 %l2, %bit
  %i.next = add i32 %i, 1
  %done = icmp eq i32 %i.next, 64
  br i1 %done, label %exit, label %loop
exit:
  %e0 = insertvalue { i64, i64, i1 } { i64 0, i64 0, i1 false }, i64 %l.next, 0
  %e1 = insertvalue { i64, i64, i1 } %e0, i64 %h.next, 1
  ret { i64, i64, i1 } %e1
}
";

/// `{q, r, fault} = (hi:lo) / d` (signed 128 / 64, truncating). The remainder has the sign of the
/// dividend; `fault` is set for `d == 0` or a quotient outside the i64 range.
pub(super) const SDIV128_IR: &str = "\
define internal { i64, i64, i1 } @__playgta6_sdiv128(i64 %hi, i64 %lo, i64 %d) unnamed_addr #0 {
entry:
  %sign = ashr i64 %lo, 63
  %narrow = icmp eq i64 %hi, %sign
  %nz = icmp ne i64 %d, 0
  %is_min = icmp eq i64 %lo, -9223372036854775808
  %is_neg1 = icmp eq i64 %d, -1
  %ov = and i1 %is_min, %is_neg1
  %not_ov = xor i1 %ov, true
  %ok0 = and i1 %narrow, %nz
  %fast_ok = and i1 %ok0, %not_ov
  br i1 %fast_ok, label %fast, label %slow
fast:
  %fq = sdiv i64 %lo, %d
  %fr = srem i64 %lo, %d
  %f0 = insertvalue { i64, i64, i1 } { i64 0, i64 0, i1 false }, i64 %fq, 0
  %f1 = insertvalue { i64, i64, i1 } %f0, i64 %fr, 1
  ret { i64, i64, i1 } %f1
slow:
  %nneg = icmp slt i64 %hi, 0
  %dneg = icmp slt i64 %d, 0
  %lo.neg = sub i64 0, %lo
  %lo.zero = icmp eq i64 %lo, 0
  %hi.not = xor i64 %hi, -1
  %hi.inc = zext i1 %lo.zero to i64
  %hi.neg = add i64 %hi.not, %hi.inc
  %ahi = select i1 %nneg, i64 %hi.neg, i64 %hi
  %alo = select i1 %nneg, i64 %lo.neg, i64 %lo
  %d.neg = sub i64 0, %d
  %ad = select i1 %dneg, i64 %d.neg, i64 %d
  %u = call { i64, i64, i1 } @__playgta6_udiv128(i64 %ahi, i64 %alo, i64 %ad)
  %uq = extractvalue { i64, i64, i1 } %u, 0
  %ur = extractvalue { i64, i64, i1 } %u, 1
  %ufault = extractvalue { i64, i64, i1 } %u, 2
  %qneg = xor i1 %nneg, %dneg
  %limit = select i1 %qneg, i64 -9223372036854775808, i64 9223372036854775807
  %range = icmp ugt i64 %uq, %limit
  %fault = or i1 %ufault, %range
  %q.neg = sub i64 0, %uq
  %q = select i1 %qneg, i64 %q.neg, i64 %uq
  %r.neg = sub i64 0, %ur
  %r = select i1 %nneg, i64 %r.neg, i64 %ur
  %s0 = insertvalue { i64, i64, i1 } poison, i64 %q, 0
  %s1 = insertvalue { i64, i64, i1 } %s0, i64 %r, 1
  %s2 = insertvalue { i64, i64, i1 } %s1, i1 %fault, 2
  ret { i64, i64, i1 } %s2
}
";
