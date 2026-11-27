//! Interprocedural liveness of the six arithmetic flags at function entry.
//!
//! A call or tail jump stores every flag the caller tracks into `%state` so the callee can read
//! it. Nearly every callee overwrites the flags before reading them (`sub rsp, N` in the prologue),
//! so those stores are dead, and so is everything computed only for them: PF's `ctpop`, AF's xor
//! chain, the 128-bit high half behind a truncating `imul`'s CF/OF. Once the callee is not inlined
//! LLVM cannot see that; this pass can.
//!
//! [`FlagSummaries`] holds, per function entry, the flags whose incoming value the function may
//! observe: read by an instruction, passed on to a callee that may observe it, or still unchanged
//! when the function returns, jumps through the dispatcher, traps or meets an instruction this pass
//! does not model. A flag outside that set is therefore overwritten on every path before any of
//! that happens, and a caller may skip storing it before a direct call or tail jump: the callee
//! never looks at the stale value in `%state`, and when it returns the flag holds the callee's own
//! value, which is exactly what the caller loads back. Behaviour at every return, trap and call is
//! unchanged; only stores nobody can read disappear.
//!
//! The one place a stale value can show is an exception unwind or `longjmp` that leaves the callee
//! before it wrote the flag: the resumed frame (a catch continuation, `__except` body or `setjmp`
//! return site) starts with flags that are undefined on hardware too, since neither the unwinder
//! nor `jmp_buf` carries the status flags across frames.
//!
//! The analysis is deliberately small: only the integer instructions every prologue is made of
//! are modelled (`mov`/`movzx`/`movsx`/`movsxd`/`lea`/`push`/`pop`/`nop`, the ALU, `inc`/`dec`/
//! `neg`, shifts and rotates, truncating `imul`), with the flag effects the lifter implements and
//! only for operand forms the lifter supports. Any other instruction counts as reading every flag.
//! Calls to functions outside the analysed set, indirect transfers, returns and traps read every
//! flag. The summaries are the least fixed point over the call graph (a may-analysis: they only
//! grow), which is sound for every execution that returns.
//!
//! # Registers a callee never touches
//!
//! The same summaries also record, per function, the guest state a call to it may touch:
//! - bits `0..15`: general purpose registers `rax..r15`,
//! - bits `16..31`: `xmm0..xmm15` (low 128 bits of `ymm0..15`),
//! - bits `32..47`: `ymmh0..ymmh15` (upper 128 bits of `ymm0..15`, including VEX-128 zeroing),
//! - bit `48` ([`OTHER_STATE`]): non-GPR/non-vector state (`mxcsr`, etc.) and any whole-state
//!   observation (an instruction the pass does not model, an indirect call or jump, a call outside
//!   the analysed set, a trap, invalid code, a non-returning call — which set [`ALL_STATE`]).
//!
//! The analysis is flow-insensitive (the union over every block of the function) and the least
//! fixed point over the call graph. A register outside that set is neither read nor written by
//! the callee, and nothing can look at `%state` while it runs, since every such place sets every
//! bit: a caller keeps untouched GPRs, XMMs and YMM upper halves in its allocas across a direct
//! call instead of storing them before and reloading them after. The stale copy in `%state` is
//! refreshed by the caller's next full write-back (its return, tail jump, indirect call, helper
//! or trap) or by the store before a call that may touch it. A host fault (wasm out-of-bounds
//! access, native access violation) inside such a callee is fatal, not a guest exception, so no
//! unwinder or handler ever reads the stale copy.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use iced_x86::{
    Code, ConditionCode, Instruction, InstructionInfo, InstructionInfoFactory, Mnemonic, OpAccess,
    OpKind, Register,
};

use super::memory_access::WindowBurst;
use super::{FLAGS, ends_in_transfer, is_immediate, mask, simd};
use crate::cfg::{self, CallTarget, DiagnosticKind, Edge, Terminator};

/// Bit `Flag as usize` per arithmetic flag.
pub(super) type FlagMask = u8;

/// Every arithmetic flag.
pub(super) const ALL_FLAGS: FlagMask = (1 << FLAGS.len()) - 1;

/// Per-register guest-state touch mask:
/// - bits `0..15`: GPRs `rax..r15`,
/// - bits `16..31`: `xmm0..xmm15` ([`XMM_STATE_SHIFT`]),
/// - bits `32..47`: `ymmh0..ymmh15` ([`YMMH_STATE_SHIFT`]),
/// - bit `48`: [`OTHER_STATE`] (non-GPR/non-vector state such as `mxcsr`).
pub(super) type StateMask = u64;

/// Bit shift of `xmm0..xmm15` within [`StateMask`].
pub(super) const XMM_STATE_SHIFT: u32 = 16;

/// All 16 `xmm0..xmm15` bits (`16..31`).
pub(super) const ALL_XMM_STATE: StateMask = 0xFFFF_u64 << XMM_STATE_SHIFT;

/// Bit shift of `ymmh0..ymmh15` within [`StateMask`].
pub(super) const YMMH_STATE_SHIFT: u32 = 32;

/// All 16 `ymmh0..ymmh15` bits (`32..47`).
pub(super) const ALL_YMMH_STATE: StateMask = 0xFFFF_u64 << YMMH_STATE_SHIFT;

/// Non-GPR/non-XMM/non-YMMH state (`mxcsr`, x87, etc.) and whole-state observations.
pub(super) const OTHER_STATE: StateMask = 1u64 << 48;

/// Everything a call may touch (bits `0..=48`).
pub(super) const ALL_STATE: StateMask = (OTHER_STATE << 1) - 1;
/// The Win64 integer argument registers `rcx rdx r8 r9` and `rsp`, which the entry probe reads
/// (`rsp` pairs the exit hook with its entry).
pub(super) const PROBE_ARG_STATE: StateMask = (1 << 1) | (1 << 2) | (1 << 4) | (1 << 8) | (1 << 9);

/// Bit in [`StateMask`] for `xmm{idx}` (`idx < 16`).
pub(super) const fn xmm_state_bit(idx: usize) -> StateMask {
    1u64 << (XMM_STATE_SHIFT + idx as u32)
}

/// Bit in [`StateMask`] for `ymmh{idx}` (`idx < 16`).
pub(super) const fn ymmh_state_bit(idx: usize) -> StateMask {
    1u64 << (YMMH_STATE_SHIFT + idx as u32)
}

/// Per-function flags whose value at entry may be observed, and the guest state a call may touch
/// (see the module docs).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FlagSummaries {
    live_in: BTreeMap<u64, FlagMask>,
    touched_flags: BTreeMap<u64, FlagMask>,
    written_flags: BTreeMap<u64, FlagMask>,
    touched: BTreeMap<u64, StateMask>,
    written: BTreeMap<u64, StateMask>,
    live_in_state: BTreeMap<u64, StateMask>,
}

impl FlagSummaries {
    /// Analyses `functions` together: calls between them use each other's summaries, calls to
    /// anything else count as reading every flag.
    pub fn compute<'a>(functions: impl IntoIterator<Item = &'a cfg::Function>) -> Self {
        Self::compute_with_probes(functions, &BTreeSet::new())
    }

    /// [`Self::compute`] for a build that inserts the entry probe
    /// ([`super::LiftedFunction::insert_entry_probe`]) into every function in `probes`. The probe
    /// logs `rcx rdx r8 r9`, so each probed function counts as reading them at entry: callers
    /// then store them before a direct call and the probe sees the synced arguments, not a stale
    /// copy (docs/LIFTER_STATE_CONTRACT.md).
    pub fn compute_with_probes<'a>(
        functions: impl IntoIterator<Item = &'a cfg::Function>,
        probes: &BTreeSet<u64>,
    ) -> Self {
        let functions: Vec<&cfg::Function> = functions.into_iter().collect();
        let (touched, written) = state_effects(&functions, probes);
        let live_in_state = state_live_in(&functions, &touched, probes);
        let (touched_flags, written_flags) = flag_effects(&functions);
        let functions: BTreeMap<u64, Model> = functions
            .into_iter()
            .map(|f| (f.entry, Model::new(f)))
            .collect();
        let mut live_in: BTreeMap<u64, FlagMask> =
            functions.keys().map(|&entry| (entry, 0)).collect();
        // Callers to revisit when a function's summary grows.
        let mut callers: BTreeMap<u64, BTreeSet<u64>> = BTreeMap::new();
        for (&entry, model) in &functions {
            for callee in model.callees() {
                if functions.contains_key(&callee) {
                    callers.entry(callee).or_default().insert(entry);
                }
            }
        }
        let mut queued: BTreeSet<u64> = functions.keys().copied().collect();
        let mut work: VecDeque<u64> = functions.keys().copied().collect();
        while let Some(entry) = work.pop_front() {
            queued.remove(&entry);
            let old = live_in[&entry];
            let mask = old | functions[&entry].entry_live(&live_in, &touched_flags);
            live_in.insert(entry, mask);
            if mask != old {
                for &caller in callers.get(&entry).into_iter().flatten() {
                    if queued.insert(caller) {
                        work.push_back(caller);
                    }
                }
            }
        }
        Self {
            live_in,
            touched_flags,
            written_flags,
            touched,
            written,
            live_in_state,
        }
    }

    /// Test hook: overwrites one summary field of `entry` (0 live_in, 1 touched_flags,
    /// 2 written_flags, 3 touched, 4 written, 5 live_in_state) to `value`.
    #[cfg(test)]
    pub(crate) fn set_field_for_test(&mut self, entry: u64, field: u8, value: u64) {
        match field {
            0 => drop(self.live_in.insert(entry, value as u8)),
            1 => drop(self.touched_flags.insert(entry, value as u8)),
            2 => drop(self.written_flags.insert(entry, value as u8)),
            3 => drop(self.touched.insert(entry, value)),
            4 => drop(self.written.insert(entry, value)),
            _ => drop(self.live_in_state.insert(entry, value)),
        }
    }

    /// Flags the function at `entry` may observe on entry; `None` for a function outside the set.
    pub fn live_in(&self, entry: u64) -> Option<u8> {
        self.live_in.get(&entry).copied()
    }

    /// Arithmetic flags a call to the function at `entry` may read or write; `None` for a function
    /// outside the set.
    pub fn touched_flags(&self, entry: u64) -> Option<u8> {
        self.touched_flags.get(&entry).copied()
    }

    /// Arithmetic flags a call to the function at `entry` may write (subset of
    /// [`Self::touched_flags`]); `None` for a function outside the set.
    pub fn written_flags(&self, entry: u64) -> Option<u8> {
        self.written_flags.get(&entry).copied()
    }

    /// Guest state a call to the function at `entry` may read or write: GPR `n` as bit `n`,
    /// `xmmN` as bit `16 + n`, `ymmhN` as bit `32 + n`, other state as bit `48`. `None` for a
    /// function outside the set.
    pub fn touched(&self, entry: u64) -> Option<u64> {
        self.touched.get(&entry).copied()
    }

    /// Guest state a call to the function at `entry` may write (subset of [`Self::touched`]);
    /// `None` for a function outside the set.
    pub fn written(&self, entry: u64) -> Option<u64> {
        self.written.get(&entry).copied()
    }

    /// Guest state whose value at `entry` may be observed (read before unconditional full
    /// overwrite, or still unchanged at a return, indirect transfer, trap, or unmodelled
    /// instruction); `None` for a function outside the set.
    pub fn live_in_state(&self, entry: u64) -> Option<u64> {
        self.live_in_state.get(&entry).copied()
    }

    /// Guest state a direct call to `target` must store before it (`touched & live_in_state`).
    pub(super) fn sync_out_state_by(&self, target: u64) -> StateMask {
        let touched = self.touched(target).unwrap_or(ALL_STATE);
        let live_in = self.live_in_state(target).unwrap_or(ALL_STATE);
        touched & live_in
    }

    /// Guest state a direct call to `target` must reload after it (`written`).
    pub(super) fn written_by(&self, target: u64) -> StateMask {
        self.written(target).unwrap_or(ALL_STATE)
    }

    /// Flags a tail jump to `target` must store first (`live_in`).
    pub(super) fn needed_by(&self, target: u64) -> FlagMask {
        self.live_in(target).unwrap_or(ALL_FLAGS)
    }

    /// Flags a direct call to `target` may touch (`touched_flags`).
    pub(super) fn touched_flags_by(&self, target: u64) -> FlagMask {
        self.touched_flags(target).unwrap_or(ALL_FLAGS)
    }

    /// Flags a direct call to `target` must store before calling (`touched_flags & live_in`).
    pub(super) fn sync_out_flags_by(&self, target: u64) -> FlagMask {
        self.touched_flags_by(target) & self.needed_by(target)
    }

    /// Flags a direct call to `target` may overwrite (`written_flags`).
    pub(super) fn written_flags_by(&self, target: u64) -> FlagMask {
        self.written_flags(target).unwrap_or(ALL_FLAGS)
    }
}

/// One backward step of a block.
#[derive(Debug, Clone, Copy)]
enum Step {
    /// `live = gen | (live & !kill)`.
    Effect { gen_: FlagMask, kill: FlagMask },
    /// A direct call that returns: `live = live_in(target)` (every flag outside it is written by
    /// the callee on each returning path, so the caller's value of it is dead across the call).
    Call(u64),
}

/// Where a block's flags go when it ends.
#[derive(Debug, Clone, Copy)]
enum Exit {
    Block(usize),
    Function(u64),
    Everything,
}

struct BlockModel {
    /// In program order; applied in reverse.
    steps: Vec<Step>,
    exits: Vec<Exit>,
}

struct Model {
    entry: usize,
    blocks: Vec<BlockModel>,
}

impl Model {
    fn new(func: &cfg::Function) -> Self {
        let index: BTreeMap<u64, usize> = func
            .blocks
            .keys()
            .enumerate()
            .map(|(i, &start)| (start, i))
            .collect();
        let blocks = func
            .blocks
            .values()
            .map(|block| block_model(block, &index))
            .collect();
        let entry = index.get(&func.entry).copied().unwrap_or(usize::MAX);
        Self { entry, blocks }
    }

    fn callees(&self) -> impl Iterator<Item = u64> + '_ {
        self.blocks.iter().flat_map(|b| {
            let calls = b.steps.iter().filter_map(|s| match *s {
                Step::Call(target) => Some(target),
                Step::Effect { .. } => None,
            });
            let tails = b.exits.iter().filter_map(|e| match *e {
                Exit::Function(target) => Some(target),
                _ => None,
            });
            calls.chain(tails)
        })
    }

    /// Flags live at the entry block under the current summaries (least fixed point).
    fn entry_live(
        &self,
        summaries: &BTreeMap<u64, FlagMask>,
        touched: &BTreeMap<u64, FlagMask>,
    ) -> FlagMask {
        if self.entry >= self.blocks.len() {
            return ALL_FLAGS;
        }
        let needed = |target: u64| summaries.get(&target).copied().unwrap_or(ALL_FLAGS);
        let touched = |target: u64| touched.get(&target).copied().unwrap_or(ALL_FLAGS);
        self.block_live_in(needed, touched)[self.entry]
    }

    /// Flags live at the start of every block, given the flags each direct callee or tail-jump
    /// target needs and touches (least fixed point).
    fn block_live_in(
        &self,
        needed: impl Fn(u64) -> FlagMask,
        touched: impl Fn(u64) -> FlagMask,
    ) -> Vec<FlagMask> {
        let mut live_in = vec![0 as FlagMask; self.blocks.len()];
        loop {
            let mut changed = false;
            for (i, block) in self.blocks.iter().enumerate().rev() {
                let mut live = 0;
                for exit in &block.exits {
                    live |= match *exit {
                        Exit::Block(j) => live_in[j],
                        Exit::Function(target) => needed(target),
                        Exit::Everything => ALL_FLAGS,
                    };
                }
                for step in block.steps.iter().rev() {
                    live = match *step {
                        Step::Effect { gen_, kill } => gen_ | (live & !kill),
                        Step::Call(target) => (touched(target) | live) & needed(target),
                    };
                }
                let grown = live_in[i] | live;
                if grown != live_in[i] {
                    live_in[i] = grown;
                    changed = true;
                }
            }
            if !changed {
                return live_in;
            }
        }
    }
}

/// Flags live at the exit of each basic block (keyed by block start address), BEFORE accounting
/// for the block's own instructions or a `Terminator::Branch` condition.
pub(super) fn block_live_out(
    func: &cfg::Function,
    summaries: Option<&FlagSummaries>,
) -> BTreeMap<u64, FlagMask> {
    let needed = |target: u64| summaries.map_or(ALL_FLAGS, |s| s.needed_by(target));
    let touched = |target: u64| summaries.map_or(ALL_FLAGS, |s| s.touched_flags_by(target));
    let model = Model::new(func);
    let live_in = model.block_live_in(needed, touched);
    let mut out = BTreeMap::new();
    for (block, model_block) in func.blocks.values().zip(&model.blocks) {
        let mut live = 0;
        for exit in &model_block.exits {
            live |= match *exit {
                Exit::Block(j) => live_in[j],
                Exit::Function(target) => needed(target),
                Exit::Everything => ALL_FLAGS,
            };
        }
        out.insert(block.start, live);
    }
    out
}

/// Same-block flag-producer shapes whose flags can be fused directly into `jcc`/`setcc`/`cmovcc`
/// without materializing or reading individual flag allocas.
///
/// Every `(ProducerKind, ConditionCode)` accepted by [`ProducerKind::supports`] MUST:
/// 1. Be matched by `FunctionLifter::try_fused_condition` with identical semantics, and
/// 2. Only read flags that the producer unconditionally kills in [`flag_effect`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProducerKind {
    /// `cmp` / `sub` / `neg`: `a - b`. `has_diff_or_zero` is true when `diff` (or `b == 0`) is
    /// available for `s`/`ns` without extra arithmetic.
    Compare { has_diff_or_zero: bool },
    /// `test` / `and` / `or` / `xor` / `andn`: `CF = 0, OF = 0`, `ZF`/`SF` from result.
    LogicZero,
    /// `add a, b`: `ZF`/`SF` from `result`, `CF` from `result <u a`.
    Add,
    /// `blsi` / `blsr` / `bzhi` / `adc` / `sbb` / non-zero imm `shl`/`shr`/`sar`/`shld`/`shrd`:
    /// `ZF`/`SF` from result.
    ResultZeroSign,
    /// `dec a`: `ZF`/`SF`/`OF` from `a - 1` (`CF` preserved).
    Dec,
    /// `inc a`: `ZF`/`SF`/`OF` from `a + 1` (`CF` preserved).
    Inc,
    /// `bsf` / `bsr` / `popcnt` / `bextr`: `ZF` from input/result.
    ZeroOnly,
    /// `tzcnt` / `lzcnt`: `CF = (src == 0)`, `ZF = (result == 0)`.
    BitCount,
    /// `bt` / `bts` / `btr` / `btc`: `CF = tested bit`.
    CarryOnly,
    /// `(v)(u)comis{s,d}`: all 16 x86 CCs fuse to a single `fcmp` or constant.
    FloatCompare,
}

impl ProducerKind {
    fn supports(self, cc: ConditionCode) -> bool {
        use ConditionCode as C;
        match self {
            Self::Compare { has_diff_or_zero } => match cc {
                C::e | C::ne | C::b | C::ae | C::be | C::a | C::l | C::ge | C::le | C::g => true,
                C::s | C::ns => has_diff_or_zero,
                _ => false,
            },
            Self::LogicZero => matches!(
                cc,
                C::o | C::no
                    | C::b
                    | C::ae
                    | C::e
                    | C::ne
                    | C::be
                    | C::a
                    | C::s
                    | C::ns
                    | C::l
                    | C::ge
                    | C::le
                    | C::g
            ),
            Self::Add => matches!(cc, C::e | C::ne | C::s | C::ns | C::b | C::ae),
            Self::ResultZeroSign => matches!(cc, C::e | C::ne | C::s | C::ns),
            Self::Dec | Self::Inc => {
                matches!(cc, C::e | C::ne | C::s | C::ns | C::l | C::ge | C::le | C::g)
            }
            Self::ZeroOnly => matches!(cc, C::e | C::ne),
            Self::BitCount => matches!(cc, C::e | C::ne | C::b | C::ae),
            Self::CarryOnly => matches!(cc, C::b | C::ae),
            Self::FloatCompare => cc != C::None,
        }
    }
}

fn is_setcc(m: Mnemonic) -> bool {
    use Mnemonic as M;
    matches!(
        m,
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
            | M::Setg
    )
}

fn is_cmovcc(m: Mnemonic) -> bool {
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

fn same_reg_operands(ins: &Instruction) -> bool {
    ins.op_count() >= 2
        && ins.op_kind(0) == OpKind::Register
        && ins.op_kind(1) == OpKind::Register
        && ins.op_register(0) == ins.op_register(1)
}

fn producer_kind(ins: &Instruction) -> Option<ProducerKind> {
    use Mnemonic as M;
    flag_effect(ins)?;
    match ins.mnemonic() {
        M::Cmp => {
            if same_reg_operands(ins) {
                return None;
            }
            let bits = operand_bits(ins)?;
            let is_zero = is_immediate(ins.op_kind(1))
                && ins
                    .try_immediate(1)
                    .ok()
                    .is_some_and(|v| (v & mask(bits)) == 0);
            Some(ProducerKind::Compare {
                has_diff_or_zero: is_zero,
            })
        }
        M::Sub => {
            if same_reg_operands(ins) {
                return None;
            }
            Some(ProducerKind::Compare {
                has_diff_or_zero: true,
            })
        }
        M::Neg => Some(ProducerKind::Compare {
            has_diff_or_zero: true,
        }),
        M::Xor => {
            if same_reg_operands(ins) {
                return None;
            }
            Some(ProducerKind::LogicZero)
        }
        M::Test | M::And | M::Or | M::Andn => Some(ProducerKind::LogicZero),
        M::Add => Some(ProducerKind::Add),
        M::Adc | M::Sbb | M::Blsi | M::Blsr | M::Bzhi => Some(ProducerKind::ResultZeroSign),
        M::Shl | M::Sal | M::Shr | M::Sar => {
            let bits = operand_bits(ins)?;
            let count_mask = if bits == 64 { 63 } else { 31 };
            if is_immediate(ins.op_kind(1))
                && ins
                    .try_immediate(1)
                    .ok()
                    .is_some_and(|v| (v & count_mask) != 0)
            {
                Some(ProducerKind::ResultZeroSign)
            } else {
                None
            }
        }
        M::Shld | M::Shrd => {
            let bits = operand_bits(ins)?;
            let count_mask = if bits == 64 { 63 } else { 31 };
            if is_immediate(ins.op_kind(2))
                && ins
                    .try_immediate(2)
                    .ok()
                    .is_some_and(|v| (v & count_mask) != 0)
            {
                Some(ProducerKind::ResultZeroSign)
            } else {
                None
            }
        }
        M::Dec => Some(ProducerKind::Dec),
        M::Inc => Some(ProducerKind::Inc),
        M::Bsf | M::Bsr | M::Popcnt | M::Bextr => Some(ProducerKind::ZeroOnly),
        M::Tzcnt | M::Lzcnt => Some(ProducerKind::BitCount),
        M::Bt | M::Bts | M::Btr | M::Btc => Some(ProducerKind::CarryOnly),
        M::Comiss
        | M::Vcomiss
        | M::Ucomiss
        | M::Vucomiss
        | M::Comisd
        | M::Vcomisd
        | M::Ucomisd
        | M::Vucomisd => Some(ProducerKind::FloatCompare),
        _ => None,
    }
}

/// Per body instruction of `block`, the flags whose value is dead right after it, accounting for
/// same-block producer -> `jcc`/`setcc`/`cmovcc` predicate fusion.
///
/// `bursts` are the coalesced window bursts in `block`: a producer inside a burst only elides
/// flags for consumers inside the *same* burst (because `lift_window_burst` clears modified
/// `known` entries at its `join` block).
pub(super) fn dead_flags_for_block(
    block: &cfg::BasicBlock,
    live_out: FlagMask,
    summaries: Option<&FlagSummaries>,
    bursts: &[WindowBurst],
) -> Vec<FlagMask> {
    let needed = |target: u64| summaries.map_or(ALL_FLAGS, |s| s.needed_by(target));
    let touched = |target: u64| summaries.map_or(ALL_FLAGS, |s| s.touched_flags_by(target));
    let (body, transfer) = match block.instructions.split_last() {
        Some((last, rest)) if ends_in_transfer(block.terminator) => (rest, Some(last)),
        _ => (&block.instructions[..], None),
    };
    let burst_of = |idx: usize| -> Option<usize> {
        bursts
            .iter()
            .position(|b| idx >= b.start && idx <= b.end)
    };

    // Identify same-block `setcc`/`cmovcc`/`jcc` and carry-consuming (`sbb`/`adc`/`adcx`/`cmc`)
    // consumers that will be answered directly from an earlier producer's `FlagSource` without
    // reading any flag alloca.
    let mut fused_body = vec![false; body.len()];
    let mut fused_term = false;
    for (p, prod) in body.iter().enumerate() {
        let Some(kind) = producer_kind(prod) else {
            continue;
        };
        let p_burst = burst_of(p);
        let mut reaches_end = true;
        for j in (p + 1)..body.len() {
            let cur = &body[j];
            if flag_effect(cur).is_some() && (is_setcc(cur.mnemonic()) || is_cmovcc(cur.mnemonic()))
            {
                if kind.supports(cur.condition_code())
                    && (p_burst.is_none() || p_burst == burst_of(j))
                {
                    fused_body[j] = true;
                }
                continue;
            }
            if flag_effect(cur).is_some()
                && matches!(
                    cur.mnemonic(),
                    Mnemonic::Sbb | Mnemonic::Adc | Mnemonic::Adcx | Mnemonic::Cmc
                )
            {
                if kind.supports(ConditionCode::b)
                    && (p_burst.is_none() || p_burst == burst_of(j))
                {
                    fused_body[j] = true;
                }
                reaches_end = false;
                break;
            }
            if cur.mnemonic() != Mnemonic::Call
                && cur.rflags_modified() == 0
                && flag_effect(cur) == Some((0, 0))
            {
                continue;
            }
            reaches_end = false;
            break;
        }
        if reaches_end
            && p_burst.is_none()
            && let (Terminator::Branch, Some(term)) = (block.terminator, transfer)
            && term.is_jcc_short_or_near()
            && kind.supports(term.condition_code())
        {
            fused_term = true;
        }
    }

    let mut live = live_out;
    if !fused_term
        && let (Terminator::Branch, Some(term)) = (block.terminator, transfer)
    {
        live |= read_mask(term);
    }
    let mut dead = vec![0 as FlagMask; body.len()];
    for (idx, ins) in body.iter().enumerate().rev() {
        live = match instruction_step(block, ins) {
            Step::Effect { gen_, kill } => {
                if ins.mnemonic() == Mnemonic::Call || flag_effect(ins).is_some() {
                    dead[idx] = ALL_FLAGS & !live;
                }
                let effective_gen = if fused_body[idx] { 0 } else { gen_ };
                effective_gen | (live & !kill)
            }
            Step::Call(target) => {
                dead[idx] = ALL_FLAGS & !live;
                (touched(target) | live) & needed(target)
            }
        };
    }
    dead
}

fn block_model(block: &cfg::BasicBlock, index: &BTreeMap<u64, usize>) -> BlockModel {
    let (body, transfer) = match block.instructions.split_last() {
        Some((last, rest)) if ends_in_transfer(block.terminator) => (rest, Some(last)),
        _ => (&block.instructions[..], None),
    };
    let mut steps: Vec<Step> = Vec::with_capacity(body.len() + 1);
    let mut push = |step: Step| {
        // Merge consecutive effects: A then B is `gen_A | (gen_B & !kill_A)`, `kill_A | kill_B`.
        if let (Some(Step::Effect { gen_: g1, kill: k1 }), Step::Effect { gen_: g2, kill: k2 }) =
            (steps.last_mut(), step)
        {
            *g1 |= g2 & !*k1;
            *k1 |= k2;
        } else {
            steps.push(step);
        }
    };
    for ins in body {
        push(instruction_step(block, ins));
    }
    let everything = vec![Exit::Everything];
    let exits = match (block.terminator, transfer) {
        (Terminator::Fallthrough | Terminator::Jump, _) => successor_exits(block, index),
        (Terminator::Branch, Some(ins)) => {
            push(Step::Effect {
                gen_: read_mask(ins),
                kill: 0,
            });
            successor_exits(block, index)
        }
        // Returns, the dispatcher, traps, non-returning calls, invalid code.
        _ => everything,
    };
    BlockModel { steps, exits }
}

fn successor_exits(block: &cfg::BasicBlock, index: &BTreeMap<u64, usize>) -> Vec<Exit> {
    if block.successors.is_empty() {
        return vec![Exit::Everything];
    }
    block
        .successors
        .iter()
        .map(|edge| match *edge {
            Edge::TailCall(target) => Exit::Function(target),
            _ => match edge.intra_target().and_then(|t| index.get(&t)) {
                Some(&i) => Exit::Block(i),
                None => Exit::Everything,
            },
        })
        .collect()
}

fn instruction_step(block: &cfg::BasicBlock, ins: &Instruction) -> Step {
    const READS_ALL: Step = Step::Effect {
        gen_: ALL_FLAGS,
        kill: 0,
    };
    if ins.mnemonic() == Mnemonic::Call {
        if ins.op_kind(0) != OpKind::NearBranch64 {
            return READS_ALL;
        }
        let target = ins.near_branch_target();
        // A direct call into unmapped or non-executable memory is lifted as a trap.
        let faults = block.calls.iter().any(|call| {
            call.site == ins.ip()
                && call.target == CallTarget::Direct(target)
                && block.diagnostics.iter().any(|d| {
                    d.site == call.site
                        && d.address == target
                        && matches!(
                            d.kind,
                            DiagnosticKind::Unmapped | DiagnosticKind::NotExecutable
                        )
                })
        });
        return if faults {
            READS_ALL
        } else {
            Step::Call(target)
        };
    }
    match flag_effect(ins) {
        Some((gen_, kill)) => Step::Effect { gen_, kill },
        None => READS_ALL,
    }
}

/// `(read, written)` flags of the modelled non-trapping, non-calling instructions; `None` for
/// everything else (anything that may branch to `cold_trap`, call another function, or touch
/// unmodelled state).
pub(super) fn flag_effect(ins: &Instruction) -> Option<(FlagMask, FlagMask)> {
    use Mnemonic as M;
    if ins.code() == Code::INVALID || ins.has_lock_prefix() || ins.is_string_instruction() {
        return None;
    }
    if matches!(
        ins.mnemonic(),
        M::Nop
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
            | M::Rdsspd
            | M::Rdsspq
            | M::Monitorx
            | M::Mwaitx
            | M::Mfence
            | M::Lfence
            | M::Sfence
            | M::Pause
            | M::Cld
            | M::Std
    ) {
        return Some((0, 0));
    }
    if let Some(effect) = simd_flag_effect(ins) {
        return Some(effect);
    }
    if !plain_operands(ins) {
        return None;
    }
    // CF OF ZF SF PF: what a shift by a non-zero count writes (AF is left unchanged).
    let shift_flags = mask_of(&[0, 5, 3, 4, 1]);
    // CF OF: what a rotate by a non-zero count (or mul/imul) writes.
    let rotate_flags = mask_of(&[0, 5]);
    let cf = mask_of(&[0]);
    let zf = mask_of(&[3]);
    let of = mask_of(&[5]);
    let zf_sf_cf_of = mask_of(&[0, 3, 4, 5]);
    let zf_cf_of = mask_of(&[0, 3, 5]);
    match ins.mnemonic() {
        M::Mov
        | M::Movzx
        | M::Movsx
        | M::Movsxd
        | M::Lea
        | M::Push
        | M::Pop
        | M::Not
        | M::Cbw
        | M::Cwde
        | M::Cdqe
        | M::Cwd
        | M::Cdq
        | M::Cqo
        | M::Bswap => Some((0, 0)),
        M::Xchg
            if ins.op_kind(0) == OpKind::Register && ins.op_kind(1) == OpKind::Register =>
        {
            Some((0, 0))
        }
        M::Movbe if operand_bits(ins).is_some_and(|b| b >= 16) => Some((0, 0)),
        M::Shlx | M::Shrx | M::Sarx | M::Mulx | M::Pdep | M::Pext
            if operand_bits(ins).is_some_and(|b| matches!(b, 32 | 64)) =>
        {
            Some((0, 0))
        }
        M::Crc32 if operand_bits(ins).is_some_and(|b| matches!(b, 32 | 64)) => Some((0, 0)),
        M::Leave if ins.code() == Code::Leaveq => Some((0, 0)),
        M::Rorx
            if operand_bits(ins).is_some_and(|b| matches!(b, 32 | 64))
                && is_immediate(ins.op_kind(2)) =>
        {
            Some((0, 0))
        }
        m if is_cmovcc(m) && same_reg_operands(ins) => Some((0, 0)),
        m if is_setcc(m) || is_cmovcc(m) => Some((read_mask(ins), 0)),
        M::Clc | M::Stc => Some((0, cf)),
        M::Cmc => Some((cf, cf)),
        M::Adcx if operand_bits(ins).is_some_and(|b| matches!(b, 32 | 64)) => Some((cf, cf)),
        M::Adox if operand_bits(ins).is_some_and(|b| matches!(b, 32 | 64)) => Some((of, of)),
        M::Bsf | M::Bsr
            if ins.op_kind(0) == OpKind::Register
                && operand_bits(ins).is_some_and(|b| b >= 16) =>
        {
            Some((0, zf))
        }
        M::Tzcnt | M::Lzcnt if operand_bits(ins).is_some_and(|b| b >= 16) => Some((0, cf | zf)),
        M::Popcnt if operand_bits(ins).is_some_and(|b| b >= 16) => Some((0, ALL_FLAGS)),
        M::Bt | M::Bts | M::Btr | M::Btc if operand_bits(ins).is_some_and(|b| b >= 16) => {
            Some((0, cf))
        }
        M::Andn | M::Blsi | M::Blsr | M::Blsmsk | M::Bzhi
            if operand_bits(ins).is_some_and(|b| matches!(b, 32 | 64)) =>
        {
            Some((0, zf_sf_cf_of))
        }
        M::Bextr if operand_bits(ins).is_some_and(|b| matches!(b, 32 | 64)) => {
            Some((0, zf_cf_of))
        }
        M::Add
        | M::Adc
        | M::Sub
        | M::Sbb
        | M::Cmp
        | M::And
        | M::Or
        | M::Xor
        | M::Test
        | M::Inc
        | M::Dec
        | M::Neg => Some((read_mask(ins), written_mask(ins))),
        M::Mul | M::Imul => Some((0, rotate_flags)),
        M::Shl | M::Sal | M::Shr | M::Sar | M::Rol | M::Ror => {
            let flags = if matches!(ins.mnemonic(), M::Rol | M::Ror) {
                rotate_flags
            } else {
                shift_flags
            };
            match ins.op_kind(1) {
                // Variable count in CL: a zero count preserves flags and `shift()` only reads
                // `get_flag(f)` when `flag_live(f)` is true, so `live_in(f) == live_out(f)`
                // (`gen = 0, kill = 0`). `may_write_flags()` separately tracks that CL shifts
                // may overwrite `flags` for interprocedural `touched_flags`/`written_flags`.
                OpKind::Register if ins.op_register(1) == Register::CL => Some((0, 0)),
                OpKind::Register => None,
                _ => {
                    let count_mask = if operand_bits(ins) == Some(64) {
                        63
                    } else {
                        31
                    };
                    let count = ins.try_immediate(1).ok()? & count_mask;
                    Some((0, if count == 0 { 0 } else { flags }))
                }
            }
        }
        M::Shld | M::Shrd
            if ins.op_kind(1) == OpKind::Register
                && operand_bits(ins).is_some_and(|b| b >= 16) =>
        {
            match ins.op_kind(2) {
                // See `Shl..Ror` above: `double_shift()` only reads `get_flag(f)` when live.
                OpKind::Register if ins.op_register(2) == Register::CL => Some((0, 0)),
                OpKind::Register => None,
                _ => {
                    let count_mask = if operand_bits(ins) == Some(64) {
                        63
                    } else {
                        31
                    };
                    let count = ins.try_immediate(2).ok()? & count_mask;
                    Some((0, if count == 0 { 0 } else { shift_flags }))
                }
            }
        }
        _ => None,
    }
}

/// `(read, written)` arithmetic flags for supported non-trapping SIMD/AVX instructions.
fn simd_flag_effect(ins: &Instruction) -> Option<(FlagMask, FlagMask)> {
    if !simd::is_supported_non_trapping_simd(ins) {
        return None;
    }
    use Mnemonic as M;
    match ins.mnemonic() {
        M::Comiss
        | M::Vcomiss
        | M::Ucomiss
        | M::Vucomiss
        | M::Comisd
        | M::Vcomisd
        | M::Ucomisd
        | M::Vucomisd
        | M::Ptest
        | M::Vptest
        | M::Vtestps
        | M::Vtestpd
        | M::Pcmpistri
        | M::Vpcmpistri
        | M::Pcmpistrm
        | M::Vpcmpistrm
        | M::Pcmpestri
        | M::Pcmpestri64
        | M::Vpcmpestri
        | M::Vpcmpestri64
        | M::Pcmpestrm
        | M::Pcmpestrm64
        | M::Vpcmpestrm
        | M::Vpcmpestrm64 => Some((0, ALL_FLAGS)),
        _ => Some((0, 0)),
    }
}

/// Arithmetic flags a modelled non-trapping instruction (`flag_effect(ins) == Some((gen_, kill))`)
/// may write at run time. Matches `kill` except for variable-count `CL` shifts/rotates, which do
/// not unconditionally kill flags (for liveness `kill == 0`) yet DO overwrite them when `CL != 0`.
fn may_write_flags(ins: &Instruction, kill: FlagMask) -> FlagMask {
    use Mnemonic as M;
    match ins.mnemonic() {
        M::Rol | M::Ror
            if ins.op_kind(1) == OpKind::Register && ins.op_register(1) == Register::CL =>
        {
            mask_of(&[0, 5])
        }
        M::Shl | M::Sal | M::Shr | M::Sar
            if ins.op_kind(1) == OpKind::Register && ins.op_register(1) == Register::CL =>
        {
            mask_of(&[0, 5, 3, 4, 1])
        }
        M::Shld | M::Shrd
            if ins.op_kind(2) == OpKind::Register && ins.op_register(2) == Register::CL =>
        {
            mask_of(&[0, 5, 3, 4, 1])
        }
        _ => kill,
    }
}

/// [`FlagSummaries::touched_flags`] and [`FlagSummaries::written_flags`] for every function:
/// least fixed point of "own flag reads/writes plus the flag effects of every direct callee and
/// tail-jump target".
fn flag_effects(
    functions: &[&cfg::Function],
) -> (BTreeMap<u64, FlagMask>, BTreeMap<u64, FlagMask>) {
    let own: BTreeMap<u64, (FlagMask, FlagMask, BTreeSet<u64>)> = functions
        .iter()
        .map(|f| (f.entry, own_flag_effects(f)))
        .collect();
    let mut touched: BTreeMap<u64, FlagMask> = own.keys().map(|&entry| (entry, 0)).collect();
    let mut written: BTreeMap<u64, FlagMask> = own.keys().map(|&entry| (entry, 0)).collect();
    let mut callers: BTreeMap<u64, BTreeSet<u64>> = BTreeMap::new();
    for (&entry, (_, _, callees)) in &own {
        for &callee in callees {
            if own.contains_key(&callee) {
                callers.entry(callee).or_default().insert(entry);
            }
        }
    }
    let mut queued: BTreeSet<u64> = own.keys().copied().collect();
    let mut work: VecDeque<u64> = own.keys().copied().collect();
    while let Some(entry) = work.pop_front() {
        queued.remove(&entry);
        let (own_t, own_w, callees) = &own[&entry];
        let old_t = touched[&entry];
        let old_w = written[&entry];
        let mut new_t = old_t | own_t;
        let mut new_w = old_w | own_w;
        for callee in callees {
            new_t |= touched.get(callee).copied().unwrap_or(ALL_FLAGS);
            new_w |= written.get(callee).copied().unwrap_or(ALL_FLAGS);
        }
        if new_t != old_t || new_w != old_w {
            touched.insert(entry, new_t);
            written.insert(entry, new_w);
            for &caller in callers.get(&entry).into_iter().flatten() {
                if queued.insert(caller) {
                    work.push_back(caller);
                }
            }
        }
    }
    (touched, written)
}

fn own_flag_effects(func: &cfg::Function) -> (FlagMask, FlagMask, BTreeSet<u64>) {
    let mut callees = BTreeSet::new();
    if !func.blocks.contains_key(&func.entry) {
        return (ALL_FLAGS, ALL_FLAGS, callees);
    }
    let mut touched: FlagMask = 0;
    let mut written: FlagMask = 0;
    for block in func.blocks.values() {
        let (body, transfer) = match block.instructions.split_last() {
            Some((last, rest)) if ends_in_transfer(block.terminator) => (rest, Some(last)),
            _ => (&block.instructions[..], None),
        };
        for ins in body {
            match instruction_step(block, ins) {
                Step::Call(target) => {
                    callees.insert(target);
                }
                Step::Effect { .. } if ins.mnemonic() == Mnemonic::Call => {
                    touched = ALL_FLAGS;
                    written = ALL_FLAGS;
                }
                Step::Effect { gen_, kill } => match flag_effect(ins) {
                    Some(_) => {
                        let w = may_write_flags(ins, kill);
                        touched |= gen_ | w;
                        written |= w;
                    }
                    None => {
                        touched = ALL_FLAGS;
                        written = ALL_FLAGS;
                    }
                },
            }
        }
        match (block.terminator, transfer) {
            (Terminator::Fallthrough | Terminator::Jump, _) => {}
            (Terminator::Branch, Some(ins)) if ins.is_jcc_short_or_near() => {
                touched |= read_mask(ins);
            }
            (Terminator::Return, Some(ins))
                if matches!(ins.code(), Code::Retnq | Code::Retnq_imm16) => {}
            _ => {
                touched = ALL_FLAGS;
                written = ALL_FLAGS;
            }
        }
        if matches!(
            block.terminator,
            Terminator::Fallthrough | Terminator::Jump | Terminator::Branch
        ) {
            if block.successors.is_empty() {
                touched = ALL_FLAGS;
                written = ALL_FLAGS;
            }
            for edge in &block.successors {
                match *edge {
                    Edge::TailCall(target) => {
                        callees.insert(target);
                    }
                    _ => match edge.intra_target() {
                        Some(target) if func.blocks.contains_key(&target) => {}
                        _ => {
                            touched = ALL_FLAGS;
                            written = ALL_FLAGS;
                        }
                    },
                }
            }
        }
    }
    (touched, written, callees)
}

/// Registers and arithmetic flags that `func`'s own instructions may write into its local allocas
/// (excluding callee writes, which go straight to `%state` and are mirrored back via `sync_in`).
/// Registers/flags outside this mask are strictly read-only in `func`, so `sync_out_parts` can
/// skip storing them back to `%state` (their `%state` slots already hold the exact live values).
pub(super) fn own_written_state_and_flags(func: &cfg::Function) -> (StateMask, FlagMask) {
    if !func.blocks.contains_key(&func.entry) {
        return (ALL_STATE, ALL_FLAGS);
    }
    let rsp: StateMask = 1 << Register::RSP.number();
    let mut factory = InstructionInfoFactory::new();
    let mut written_state: StateMask = 0;
    let mut written_flags: FlagMask = 0;
    for block in func.blocks.values() {
        let (body, transfer) = match block.instructions.split_last() {
            Some((last, rest)) if ends_in_transfer(block.terminator) => (rest, Some(last)),
            _ => (&block.instructions[..], None),
        };
        for ins in body {
            if ins.mnemonic() == Mnemonic::Call {
                written_state |= rsp;
                continue;
            }
            let Some((_, kill)) = flag_effect(ins) else {
                return (ALL_STATE, ALL_FLAGS);
            };
            let (_, w) = used_and_written_state(ins, factory.info(ins));
            written_state |= w;
            written_flags |= may_write_flags(ins, kill);
        }
        match (block.terminator, transfer) {
            (Terminator::Fallthrough | Terminator::Jump | Terminator::IndirectJump, _) => {}
            (Terminator::Branch, Some(ins)) if ins.is_jcc_short_or_near() => {}
            (Terminator::Return, Some(ins))
                if matches!(ins.code(), Code::Retnq | Code::Retnq_imm16) =>
            {
                written_state |= rsp;
            }
            _ => return (ALL_STATE, ALL_FLAGS),
        }
    }
    (written_state, written_flags)
}

/// [`FlagSummaries::touched`] and [`FlagSummaries::written`] for every function: least fixed point
/// of "own effects plus the effects of every direct callee and tail-jump target".
fn state_effects(
    functions: &[&cfg::Function],
    probes: &BTreeSet<u64>,
) -> (BTreeMap<u64, StateMask>, BTreeMap<u64, StateMask>) {
    let mut factory = InstructionInfoFactory::new();
    let own: BTreeMap<u64, (StateMask, StateMask, BTreeSet<u64>)> = functions
        .iter()
        .map(|f| {
            let (mut t, w, callees) = own_effects(f, &mut factory);
            if probes.contains(&f.entry) {
                t |= PROBE_ARG_STATE;
            }
            (f.entry, (t, w, callees))
        })
        .collect();
    let mut touched: BTreeMap<u64, StateMask> = own.keys().map(|&entry| (entry, 0)).collect();
    let mut written: BTreeMap<u64, StateMask> = own.keys().map(|&entry| (entry, 0)).collect();
    let mut callers: BTreeMap<u64, BTreeSet<u64>> = BTreeMap::new();
    for (&entry, (_, _, callees)) in &own {
        for &callee in callees {
            if own.contains_key(&callee) {
                callers.entry(callee).or_default().insert(entry);
            }
        }
    }
    let mut queued: BTreeSet<u64> = own.keys().copied().collect();
    let mut work: VecDeque<u64> = own.keys().copied().collect();
    while let Some(entry) = work.pop_front() {
        queued.remove(&entry);
        let (own_t, own_w, callees) = &own[&entry];
        let old_t = touched[&entry];
        let old_w = written[&entry];
        let mut new_t = old_t | own_t;
        let mut new_w = old_w | own_w;
        for callee in callees {
            new_t |= touched.get(callee).copied().unwrap_or(ALL_STATE);
            new_w |= written.get(callee).copied().unwrap_or(ALL_STATE);
        }
        if new_t != old_t || new_w != old_w {
            touched.insert(entry, new_t);
            written.insert(entry, new_w);
            for &caller in callers.get(&entry).into_iter().flatten() {
                if queued.insert(caller) {
                    work.push_back(caller);
                }
            }
        }
    }
    (touched, written)
}

/// The state one function touches and writes itself, and its direct callees and tail-jump targets.
fn own_effects(
    func: &cfg::Function,
    factory: &mut InstructionInfoFactory,
) -> (StateMask, StateMask, BTreeSet<u64>) {
    let mut callees = BTreeSet::new();
    if !func.blocks.contains_key(&func.entry) {
        return (ALL_STATE, ALL_STATE, callees);
    }
    let rsp: StateMask = 1 << Register::RSP.number();
    let mut touched: StateMask = 0;
    let mut written: StateMask = 0;
    for block in func.blocks.values() {
        let (body, transfer) = match block.instructions.split_last() {
            Some((last, rest)) if ends_in_transfer(block.terminator) => (rest, Some(last)),
            _ => (&block.instructions[..], None),
        };
        for ins in body {
            let (t, w) = match instruction_step(block, ins) {
                // The call pushes the return address; the callee's own effects come from its
                // summary.
                Step::Call(target) => {
                    callees.insert(target);
                    (rsp, rsp)
                }
                Step::Effect { .. } if ins.mnemonic() == Mnemonic::Call => (ALL_STATE, ALL_STATE),
                Step::Effect { .. } if flag_effect(ins).is_none() => (ALL_STATE, ALL_STATE),
                Step::Effect { .. } => used_and_written_state(ins, factory.info(ins)),
            };
            touched |= t;
            written |= w;
        }
        let (term_t, term_w) = match (block.terminator, transfer) {
            (Terminator::Fallthrough | Terminator::Jump, _) => (0, 0),
            (Terminator::Branch, Some(ins)) if ins.is_jcc_short_or_near() => (0, 0),
            (Terminator::Return, Some(ins))
                if matches!(ins.code(), Code::Retnq | Code::Retnq_imm16) =>
            {
                (rsp, rsp)
            }
            // Indirect jumps (the dispatcher), traps, non-returning calls, invalid code, `loop`,
            // `jrcxz`, `xbegin`, far returns.
            _ => (ALL_STATE, ALL_STATE),
        };
        touched |= term_t;
        written |= term_w;
        if matches!(
            block.terminator,
            Terminator::Fallthrough | Terminator::Jump | Terminator::Branch
        ) {
            if block.successors.is_empty() {
                touched = ALL_STATE;
                written = ALL_STATE;
            }
            for edge in &block.successors {
                match *edge {
                    Edge::TailCall(target) => {
                        callees.insert(target);
                    }
                    _ => match edge.intra_target() {
                        Some(target) if func.blocks.contains_key(&target) => {}
                        _ => {
                            touched = ALL_STATE;
                            written = ALL_STATE;
                        }
                    },
                }
            }
        }
    }
    (touched, written, callees)
}

/// Guest registers (GPRs, `xmm0..15`, `ymmh0..15`) and other state a modelled non-trapping
/// instruction `(touches, writes)` (explicitly, implicitly, in an address, or via VEX-128
/// upper-half zeroing). Any non-segment register outside GPR/XMM/YMM/ZMM0..15 sets [`OTHER_STATE`].
fn used_and_written_state(ins: &Instruction, info: &InstructionInfo) -> (StateMask, StateMask) {
    match ins.mnemonic() {
        Mnemonic::Vzeroall => {
            let m = ALL_XMM_STATE | ALL_YMMH_STATE;
            return (m, m);
        }
        Mnemonic::Vzeroupper => return (ALL_YMMH_STATE, ALL_YMMH_STATE),
        _ => {}
    }
    let mut touched: StateMask = 0;
    let mut written: StateMask = 0;
    for used in info.used_registers() {
        let reg = used.register();
        let is_write = matches!(
            used.access(),
            OpAccess::Write
                | OpAccess::CondWrite
                | OpAccess::ReadWrite
                | OpAccess::ReadCondWrite
        );
        let bits = if reg.is_gpr() && reg.full_register().is_gpr64() {
            1u64 << reg.full_register().number()
        } else if (reg.is_xmm() || reg.is_ymm() || reg.is_zmm()) && reg.number() < 16 {
            let idx = reg.number();
            let mut b = xmm_state_bit(idx);
            if !reg.is_xmm() {
                b |= ymmh_state_bit(idx);
            }
            b
        } else if !reg.is_segment_register() {
            OTHER_STATE
        } else {
            0
        };
        touched |= bits;
        if is_write {
            written |= bits;
        }
    }
    match ins.mnemonic() {
        Mnemonic::Pcmpistrm | Mnemonic::Pcmpestrm | Mnemonic::Pcmpestrm64 => {
            // The byte/word mask result goes to XMM0.
            touched |= xmm_state_bit(0);
            written |= xmm_state_bit(0);
        }
        Mnemonic::Vpcmpistrm | Mnemonic::Vpcmpestrm | Mnemonic::Vpcmpestrm64 => {
            // The byte/word mask result goes to XMM0 and VEX-128 zeroes YMMH0.
            let b = xmm_state_bit(0) | ymmh_state_bit(0);
            touched |= b;
            written |= b;
        }
        Mnemonic::Blendvps | Mnemonic::Blendvpd | Mnemonic::Pblendvb => {
            // Legacy SSE4.1 variable blends implicitly read XMM0 as the mask.
            touched |= xmm_state_bit(0);
        }
        Mnemonic::Stmxcsr | Mnemonic::Vstmxcsr => {
            touched |= OTHER_STATE;
        }
        Mnemonic::Cld | Mnemonic::Std => {
            touched |= OTHER_STATE;
            written |= OTHER_STATE;
        }
        _ => {}
    }
    // VEX encodings with an XMM destination register zero bits 128..255 (`ymmhN`).
    if simd::is_vex_encoded(ins)
        && ins.op_count() > 0
        && ins.op_kind(0) == OpKind::Register
        && ins.op_register(0).is_xmm()
        && ins.op_register(0).number() < 16
        && !simd::reads_only_first_operand(ins.mnemonic())
    {
        let idx = ins.op_register(0).number();
        let b = xmm_state_bit(idx) | ymmh_state_bit(idx);
        touched |= b;
        written |= b;
    }
    (touched, written)
}

/// One backward register-state liveness step of a block.
#[derive(Debug, Clone, Copy)]
enum StateStep {
    /// `live = gen_ | (live & !kill)`.
    Effect { gen_: StateMask, kill: StateMask },
    /// A direct returning call: `live = ((touched(target) | live) & live_in_state(target)) | rsp`.
    Call(u64),
}

struct StateBlockModel {
    steps: Vec<StateStep>,
    exits: Vec<Exit>,
}

struct StateModel {
    entry: usize,
    blocks: Vec<StateBlockModel>,
}

impl StateModel {
    fn new(func: &cfg::Function, factory: &mut InstructionInfoFactory) -> Self {
        let index: BTreeMap<u64, usize> = func
            .blocks
            .keys()
            .enumerate()
            .map(|(i, &start)| (start, i))
            .collect();
        let blocks = func
            .blocks
            .values()
            .map(|block| state_block_model(block, &index, factory))
            .collect();
        let entry = index.get(&func.entry).copied().unwrap_or(usize::MAX);
        Self { entry, blocks }
    }

    fn callees(&self) -> impl Iterator<Item = u64> + '_ {
        self.blocks.iter().flat_map(|b| {
            let calls = b.steps.iter().filter_map(|s| match *s {
                StateStep::Call(target) => Some(target),
                StateStep::Effect { .. } => None,
            });
            let tails = b.exits.iter().filter_map(|e| match *e {
                Exit::Function(target) => Some(target),
                _ => None,
            });
            calls.chain(tails)
        })
    }

    /// Guest state live at the entry block under the current summaries (least fixed point).
    fn entry_live(
        &self,
        summaries: &BTreeMap<u64, StateMask>,
        touched: &BTreeMap<u64, StateMask>,
    ) -> StateMask {
        if self.entry >= self.blocks.len() {
            return ALL_STATE;
        }
        let rsp: StateMask = 1u64 << Register::RSP.number();
        let mut live_in = vec![0 as StateMask; self.blocks.len()];
        loop {
            let mut changed = false;
            for (i, block) in self.blocks.iter().enumerate().rev() {
                let mut live: StateMask = 0;
                for exit in &block.exits {
                    live |= match *exit {
                        Exit::Block(j) => live_in[j],
                        Exit::Function(target) => {
                            summaries.get(&target).copied().unwrap_or(ALL_STATE)
                        }
                        Exit::Everything => ALL_STATE,
                    };
                }
                for step in block.steps.iter().rev() {
                    live = match *step {
                        StateStep::Effect { gen_, kill } => gen_ | (live & !kill),
                        StateStep::Call(target) => {
                            let callee_touched =
                                touched.get(&target).copied().unwrap_or(ALL_STATE);
                            let callee_live =
                                summaries.get(&target).copied().unwrap_or(ALL_STATE);
                            ((callee_touched | live) & callee_live) | rsp
                        }
                    };
                }
                let grown = live_in[i] | live;
                if grown != live_in[i] {
                    live_in[i] = grown;
                    changed = true;
                }
            }
            if !changed {
                return live_in[self.entry];
            }
        }
    }
}

fn state_block_model(
    block: &cfg::BasicBlock,
    index: &BTreeMap<u64, usize>,
    factory: &mut InstructionInfoFactory,
) -> StateBlockModel {
    const READS_ALL: StateStep = StateStep::Effect {
        gen_: ALL_STATE,
        kill: 0,
    };
    let (body, transfer) = match block.instructions.split_last() {
        Some((last, rest)) if ends_in_transfer(block.terminator) => (rest, Some(last)),
        _ => (&block.instructions[..], None),
    };
    let mut steps: Vec<StateStep> = Vec::with_capacity(body.len());
    let mut push = |step: StateStep| {
        if let (
            Some(StateStep::Effect { gen_: g1, kill: k1 }),
            StateStep::Effect { gen_: g2, kill: k2 },
        ) = (steps.last_mut(), step)
        {
            *g1 |= g2 & !*k1;
            *k1 |= k2;
        } else {
            steps.push(step);
        }
    };
    for ins in body {
        let step = match instruction_step(block, ins) {
            Step::Call(target) => StateStep::Call(target),
            Step::Effect { .. } if ins.mnemonic() == Mnemonic::Call => READS_ALL,
            Step::Effect { .. } if flag_effect(ins).is_none() => READS_ALL,
            Step::Effect { .. } => {
                let (gen_, kill) = state_rw_masks(ins, factory.info(ins));
                StateStep::Effect { gen_, kill }
            }
        };
        push(step);
    }
    let exits = match (block.terminator, transfer) {
        (Terminator::Fallthrough | Terminator::Jump, _) => successor_exits(block, index),
        (Terminator::Branch, Some(ins)) if ins.is_jcc_short_or_near() => {
            successor_exits(block, index)
        }
        // Returns, indirect jumps, traps, non-returning calls, invalid code, `loop`/`jrcxz`:
        // every register reaching here may be observed.
        _ => vec![Exit::Everything],
    };
    StateBlockModel { steps, exits }
}

/// [`FlagSummaries::live_in_state`] for every function (least fixed point over the call graph).
fn state_live_in(
    functions: &[&cfg::Function],
    touched: &BTreeMap<u64, StateMask>,
    probes: &BTreeSet<u64>,
) -> BTreeMap<u64, StateMask> {
    let mut factory = InstructionInfoFactory::new();
    let models: BTreeMap<u64, StateModel> = functions
        .iter()
        .map(|f| (f.entry, StateModel::new(f, &mut factory)))
        .collect();
    let mut live_in: BTreeMap<u64, StateMask> = models.keys().map(|&entry| (entry, 0)).collect();
    let mut callers: BTreeMap<u64, BTreeSet<u64>> = BTreeMap::new();
    for (&entry, model) in &models {
        for callee in model.callees() {
            if models.contains_key(&callee) {
                callers.entry(callee).or_default().insert(entry);
            }
        }
    }
    let mut queued: BTreeSet<u64> = models.keys().copied().collect();
    let mut work: VecDeque<u64> = models.keys().copied().collect();
    while let Some(entry) = work.pop_front() {
        queued.remove(&entry);
        let old = live_in[&entry];
        let probe = if probes.contains(&entry) {
            PROBE_ARG_STATE
        } else {
            0
        };
        let mask = old | probe | models[&entry].entry_live(&live_in, touched);
        live_in.insert(entry, mask);
        if mask != old {
            for &caller in callers.get(&entry).into_iter().flatten() {
                if queued.insert(caller) {
                    work.push_back(caller);
                }
            }
        }
    }
    live_in
}

/// `(reads, full_writes)` bitmasks over [`StateMask`] for a modelled non-trapping instruction
/// (`flag_effect(ins).is_some()`). A bit is set in `full_writes` only when `ins` unconditionally
/// overwrites the entire register (64/32-bit GPR, 128-bit XMM, or 128-bit YMMH) without reading
/// it first; `rsp` is never killed.
fn state_rw_masks(ins: &Instruction, info: &InstructionInfo) -> (StateMask, StateMask) {
    match ins.mnemonic() {
        Mnemonic::Vzeroall => return (0, ALL_XMM_STATE | ALL_YMMH_STATE),
        Mnemonic::Vzeroupper => return (0, ALL_YMMH_STATE),
        _ => {}
    }
    let (gpr_reads, gpr_full_writes) = super::gpr_rw_masks(ins, info);
    let mut reads: StateMask = u64::from(gpr_reads);
    let mut full_writes: StateMask = u64::from(gpr_full_writes);

    if simd::is_simd(ins) {
        let (xmm_reads, xmm_full_writes) = simd::xmm_rw_masks(ins, info);
        reads |= u64::from(xmm_reads) << XMM_STATE_SHIFT;
        full_writes |= u64::from(xmm_full_writes) << XMM_STATE_SHIFT;

        let mut ymmh_reads: u16 = 0;
        let mut ymmh_full_writes: u16 = 0;
        if simd::is_const_idiom(ins)
            && ins.op_count() > 0
            && ins.op_kind(0) == OpKind::Register
        {
            let r0 = ins.op_register(0);
            if (r0.is_ymm() || (simd::is_vex_encoded(ins) && r0.is_xmm())) && r0.number() < 16 {
                ymmh_full_writes |= 1u16 << r0.number();
            }
        } else {
            for ur in info.used_registers() {
                let r = ur.register();
                if (r.is_ymm() || r.is_zmm())
                    && r.number() < 16
                    && matches!(
                        ur.access(),
                        OpAccess::Read
                            | OpAccess::CondRead
                            | OpAccess::ReadWrite
                            | OpAccess::ReadCondWrite
                            | OpAccess::CondWrite
                    )
                {
                    ymmh_reads |= 1u16 << r.number();
                }
            }
            for i in 1..ins.op_count() {
                if ins.op_kind(i) == OpKind::Register {
                    let r = ins.op_register(i);
                    if (r.is_ymm() || r.is_zmm()) && r.number() < 16 {
                        ymmh_reads |= 1u16 << r.number();
                    }
                }
            }
            if ins.op_count() > 0 && ins.op_kind(0) == OpKind::Register {
                let r0 = ins.op_register(0);
                if r0.number() < 16 {
                    let bit = 1u16 << r0.number();
                    if r0.is_xmm()
                        && simd::is_vex_encoded(ins)
                        && !simd::reads_only_first_operand(ins.mnemonic())
                    {
                        // VEX-128 with XMM destination unconditionally zeroes ymmhN without reading it.
                        ymmh_full_writes |= bit;
                    } else if r0.is_ymm() {
                        if xmm_full_writes & bit != 0 && ymmh_reads & bit == 0 {
                            ymmh_full_writes |= bit;
                        } else {
                            ymmh_reads |= bit;
                        }
                    }
                }
            }
        }
        if matches!(
            ins.mnemonic(),
            Mnemonic::Vpcmpistrm | Mnemonic::Vpcmpestrm | Mnemonic::Vpcmpestrm64
        ) {
            // Implicitly writes xmm0 and zeroes ymmh0 (VEX-128).
            ymmh_full_writes |= 1u16 << 0;
        }
        reads |= u64::from(ymmh_reads) << YMMH_STATE_SHIFT;
        full_writes |= u64::from(ymmh_full_writes & !ymmh_reads) << YMMH_STATE_SHIFT;
    }

    if matches!(ins.mnemonic(), Mnemonic::Stmxcsr | Mnemonic::Vstmxcsr)
        || info.used_registers().iter().any(|ur| {
            let r = ur.register();
            !(r.is_gpr() || r.is_xmm() || r.is_ymm() || r.is_zmm() || r.is_segment_register())
        })
    {
        reads |= OTHER_STATE;
    }

    (reads, full_writes & !reads)
}

fn mask_of(flags: &[usize]) -> FlagMask {
    flags.iter().fold(0, |m, &f| m | 1 << f)
}

fn read_mask(ins: &Instruction) -> FlagMask {
    rflags_mask(ins.rflags_read())
}

fn written_mask(ins: &Instruction) -> FlagMask {
    rflags_mask(ins.rflags_written() | ins.rflags_cleared() | ins.rflags_set())
}

fn rflags_mask(bits: u32) -> FlagMask {
    FLAGS
        .iter()
        .filter(|f| bits & f.rflags_bit() != 0)
        .fold(0, |m, &f| m | 1 << f as usize)
}

fn operand_bits(ins: &Instruction) -> Option<u32> {
    match ins.op_kind(0) {
        OpKind::Register => Some(ins.op_register(0).size() as u32 * 8),
        OpKind::Memory => Some(ins.memory_size().size() as u32 * 8),
        _ => None,
    }
}

/// Operands the lifter accepts for the modelled instructions (general purpose registers,
/// immediates, 1/2/4/8-byte memory addressed by general purpose registers, `lea` addresses), so a
/// modelled instruction never becomes an unsupported-instruction trap.
fn plain_operands(ins: &Instruction) -> bool {
    (0..ins.op_count()).all(|i| match ins.op_kind(i) {
        OpKind::Register => {
            let reg = ins.op_register(i);
            reg.is_gpr() && reg.full_register().is_gpr64()
        }
        OpKind::Memory => {
            let base = ins.memory_base();
            let index = ins.memory_index();
            let base_ok = matches!(base, Register::None | Register::RIP)
                || base.is_gpr64()
                || base.is_gpr32();
            let index_ok = index == Register::None || index.is_gpr64() || index.is_gpr32();
            let size_ok = ins.mnemonic() == Mnemonic::Lea
                || matches!(ins.memory_size().size(), 1 | 2 | 4 | 8);
            base_ok && index_ok && size_ok
        }
        OpKind::Immediate8
        | OpKind::Immediate16
        | OpKind::Immediate32
        | OpKind::Immediate64
        | OpKind::Immediate8to16
        | OpKind::Immediate8to32
        | OpKind::Immediate8to64
        | OpKind::Immediate32to64 => true,
        _ => false,
    })
}
