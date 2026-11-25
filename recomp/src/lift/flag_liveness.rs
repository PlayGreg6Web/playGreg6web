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

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use iced_x86::{Instruction, Mnemonic, OpKind, Register};

use super::{FLAGS, ends_in_transfer};
use crate::cfg::{self, CallTarget, DiagnosticKind, Edge, Terminator};

/// Bit `Flag as usize` per arithmetic flag.
pub(super) type FlagMask = u8;

/// Every arithmetic flag.
pub(super) const ALL_FLAGS: FlagMask = (1 << FLAGS.len()) - 1;

/// Per-function flags whose value at entry may be observed (see the module docs).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FlagSummaries {
    live_in: BTreeMap<u64, FlagMask>,
}

impl FlagSummaries {
    /// Analyses `functions` together: calls between them use each other's summaries, calls to
    /// anything else count as reading every flag.
    pub fn compute<'a>(functions: impl IntoIterator<Item = &'a cfg::Function>) -> Self {
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
            let mask = old | functions[&entry].entry_live(&live_in);
            live_in.insert(entry, mask);
            if mask != old {
                for &caller in callers.get(&entry).into_iter().flatten() {
                    if queued.insert(caller) {
                        work.push_back(caller);
                    }
                }
            }
        }
        Self { live_in }
    }

    /// Flags the function at `entry` may observe on entry; `None` for a function outside the set.
    pub fn live_in(&self, entry: u64) -> Option<u8> {
        self.live_in.get(&entry).copied()
    }

    /// Flags a direct call or tail jump to `target` must store first.
    pub(super) fn needed_by(&self, target: u64) -> FlagMask {
        self.live_in(target).unwrap_or(ALL_FLAGS)
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
    fn entry_live(&self, summaries: &BTreeMap<u64, FlagMask>) -> FlagMask {
        if self.entry >= self.blocks.len() {
            return ALL_FLAGS;
        }
        let needed = |target: u64| summaries.get(&target).copied().unwrap_or(ALL_FLAGS);
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
                        Step::Call(target) => needed(target),
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

/// `(read, written)` flags of the modelled instructions; `None` for everything else.
fn flag_effect(ins: &Instruction) -> Option<(FlagMask, FlagMask)> {
    use Mnemonic as M;
    if matches!(ins.mnemonic(), M::Nop | M::Endbr64) {
        return Some((0, 0));
    }
    if ins.has_lock_prefix() || ins.is_string_instruction() || !plain_operands(ins) {
        return None;
    }
    // CF OF ZF SF PF: what a shift by a non-zero count writes (AF is left unchanged).
    let shift_flags = mask_of(&[0, 5, 3, 4, 1]);
    // CF OF: what a rotate by a non-zero count writes.
    let rotate_flags = mask_of(&[0, 5]);
    match ins.mnemonic() {
        M::Mov | M::Movzx | M::Movsx | M::Movsxd | M::Lea | M::Push | M::Pop => Some((0, 0)),
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
        M::Imul if ins.op_count() >= 2 => Some((0, rotate_flags)),
        M::Shl | M::Sal | M::Shr | M::Sar | M::Rol | M::Ror => {
            let flags = if matches!(ins.mnemonic(), M::Rol | M::Ror) {
                rotate_flags
            } else {
                shift_flags
            };
            match ins.op_kind(1) {
                // A zero count at run time leaves the flags unchanged: they are read.
                OpKind::Register if ins.op_register(1) == Register::CL => Some((flags, 0)),
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
        _ => None,
    }
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
