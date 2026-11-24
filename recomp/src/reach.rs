//! Register values that reach a block over every intra-function path (for jump-table recovery).
//!
//! [`crate::discover`] slices an indirect `jmp` together with its chain of sole predecessors. When
//! that chain stops at a merge (a loop header, or a block with several incoming edges), the table
//! base or the index may be defined before the merge, on several paths. [`reaching_values`] runs
//! a forward dataflow over every block that can reach the merge block and reports, for each
//! requested 64-bit register, what holds on *all* incoming paths:
//!
//! * a single value (`[v, v]`), e.g. a RIP-relative `lea` of the table, possibly carried around a
//!   loop through register copies, or
//! * an unsigned interval `[lo, hi]` of the full 64-bit register, from constants, masks, shifts,
//!   zero extension and the `cmp` / `test` + conditional-branch guards on the edges, or
//! * nothing (the register is unknown on at least one path).
//!
//! The lattice per register is unvisited < interval < unknown; the join is the interval hull.
//! Each block's input is the join over its incoming edges, each refined by the guard of the edge.
//! Function entries start with every register unknown. A block whose input changes more than
//! [`WIDEN_AFTER`] times widens every interval it keeps changing to unknown, so loops terminate.
//!
//! Soundness: every transfer either computes an over-approximation of the register's value or
//! yields unknown; anything not modelled (memory loads into 64-bit registers, partial 8/16-bit
//! writes, other arithmetic, `pop`, calls for Win64-volatile registers) yields unknown. Guards
//! refine only the compared register, only for unsigned conditions or when the interval is
//! already non-negative, and only when no instruction between the compare and the branch writes
//! that register. Blocks that cannot reach the merge never contribute (the analysis only sees the
//! backward-reachable subgraph), so a register reused for something else there does not matter.
//!
//! Cost: one backward walk to collect at most [`MAX_REACH_BLOCKS`] blocks, then a worklist over
//! them; every block's input can change at most `WIDEN_AFTER + 1` times per register before it is
//! unknown, so the work is `O(blocks * instructions per block * (WIDEN_AFTER + 1) * 16)` per
//! queried merge.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use iced_x86::{
    Decoder, DecoderOptions, FlowControl, Instruction, InstructionInfoFactory, Mnemonic, OpAccess,
    OpKind, Register,
};

use crate::cfg::{BasicBlock, Edge, Terminator};

/// Blocks of the backward-reachable subgraph examined for one merge (beyond: no result).
pub(crate) const MAX_REACH_BLOCKS: usize = 4096;

/// Input changes of one block after which still-changing intervals widen to unknown.
const WIDEN_AFTER: u32 = 8;

/// What is known about a full 64-bit register on every path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Known {
    /// No path seen yet (only during the analysis).
    Unvisited,
    /// Inclusive unsigned interval of the full 64-bit value.
    Range { lo: u64, hi: u64 },
    /// Not known on at least one path.
    Unknown,
}

impl Known {
    fn join(self, other: Known) -> Known {
        match (self, other) {
            (Known::Unvisited, x) | (x, Known::Unvisited) => x,
            (Known::Unknown, _) | (_, Known::Unknown) => Known::Unknown,
            (Known::Range { lo: a, hi: b }, Known::Range { lo: c, hi: d }) => Known::Range {
                lo: a.min(c),
                hi: b.max(d),
            },
        }
    }

    fn exact(v: u64) -> Known {
        Known::Range { lo: v, hi: v }
    }

    fn within_u32(self) -> bool {
        matches!(self, Known::Range { hi, .. } if hi <= u64::from(u32::MAX))
    }
}

const GPR_COUNT: usize = 16;

/// Register file state: one [`Known`] per 64-bit GPR (index = `Register::number()` of RAX..R15).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct State([Known; GPR_COUNT]);

impl State {
    fn unvisited() -> Self {
        Self([Known::Unvisited; GPR_COUNT])
    }
    fn unknown() -> Self {
        Self([Known::Unknown; GPR_COUNT])
    }
    fn get(&self, reg: Register) -> Known {
        slot(reg).map_or(Known::Unknown, |i| self.0[i])
    }
    fn set(&mut self, reg: Register, value: Known) {
        if let Some(i) = slot(reg) {
            self.0[i] = value;
        }
    }
    fn join(&self, other: &State) -> State {
        let mut out = *self;
        for (o, x) in out.0.iter_mut().zip(other.0.iter()) {
            *o = o.join(*x);
        }
        out
    }
}

fn slot(reg: Register) -> Option<usize> {
    let full = reg.full_register();
    full.is_gpr64().then(|| full.number())
}

fn is_win64_volatile(reg: Register) -> bool {
    matches!(
        reg.full_register(),
        Register::RAX
            | Register::RCX
            | Register::RDX
            | Register::R8
            | Register::R9
            | Register::R10
            | Register::R11
    )
}

fn mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

/// Values of `regs` on entry to `target`, over every intra-function path that reaches it.
/// `preds(b)` lists the blocks with an intra-function edge into `b`; `is_entry(b)` marks function
/// entries (where every register is unknown). `None` when the reachable subgraph exceeds
/// [`MAX_REACH_BLOCKS`] or `target` is unknown.
pub(crate) fn reaching_values(
    blocks: &BTreeMap<u64, BasicBlock>,
    preds: &dyn Fn(u64) -> Vec<u64>,
    is_entry: &dyn Fn(u64) -> bool,
    target: u64,
    regs: &[Register],
) -> Option<Vec<(Register, Known)>> {
    blocks.get(&target)?;
    // Backward-reachable subgraph (closed under predecessors; stops at function entries).
    let mut sub: HashSet<u64> = HashSet::new();
    let mut stack = vec![target];
    while let Some(b) = stack.pop() {
        if !sub.insert(b) {
            continue;
        }
        if sub.len() > MAX_REACH_BLOCKS {
            return None;
        }
        blocks.get(&b)?;
        if is_entry(b) {
            continue;
        }
        stack.extend(preds(b));
    }

    let mut info = InstructionInfoFactory::new();
    let mut inputs: HashMap<u64, State> = sub.iter().map(|&b| (b, State::unvisited())).collect();
    let mut changes: HashMap<u64, u32> = HashMap::new();
    let mut queue: VecDeque<u64> = VecDeque::new();
    let mut queued: HashSet<u64> = HashSet::new();
    for &b in &sub {
        if is_entry(b) {
            inputs.insert(b, State::unknown());
            queue.push_back(b);
            queued.insert(b);
        }
    }
    // Blocks whose predecessors are all outside the function's known code (no entry above them)
    // can still start a path: treat a block without predecessors as unknown on entry.
    for &b in &sub {
        if !is_entry(b) && preds(b).is_empty() {
            inputs.insert(b, State::unknown());
            if queued.insert(b) {
                queue.push_back(b);
            }
        }
    }

    while let Some(b) = queue.pop_front() {
        queued.remove(&b);
        let Some(block) = blocks.get(&b) else {
            continue;
        };
        let mut state = inputs[&b];
        for instr in &block.instructions {
            transfer(&mut info, instr, &mut state);
        }
        for edge in &block.successors {
            let Some(next) = edge.intra_target() else {
                continue;
            };
            if !sub.contains(&next) || is_entry(next) {
                continue;
            }
            let mut out = state;
            if block.terminator == Terminator::Branch {
                refine_edge(&mut info, block, *edge, &mut out);
            }
            let old = inputs[&next];
            let mut new = old.join(&out);
            if new == old {
                continue;
            }
            let n = changes.entry(next).or_insert(0);
            *n += 1;
            if *n > WIDEN_AFTER {
                for (i, k) in new.0.iter_mut().enumerate() {
                    if *k != old.0[i] {
                        *k = Known::Unknown;
                    }
                }
            }
            inputs.insert(next, new);
            if queued.insert(next) {
                queue.push_back(next);
            }
        }
    }

    let input = inputs[&target];
    Some(
        regs.iter()
            .map(|&r| {
                let k = input.get(r);
                (
                    r,
                    if k == Known::Unvisited {
                        Known::Unknown
                    } else {
                        k
                    },
                )
            })
            .collect(),
    )
}

fn writes(info: &mut InstructionInfoFactory, instr: &Instruction, reg: Register) -> bool {
    let reg = reg.full_register();
    info.info(instr).used_registers().iter().any(|u| {
        u.register().full_register() == reg
            && matches!(
                u.access(),
                OpAccess::Write
                    | OpAccess::CondWrite
                    | OpAccess::ReadWrite
                    | OpAccess::ReadCondWrite
            )
    })
}

/// Apply `instr` to `state`.
fn transfer(info: &mut InstructionInfoFactory, instr: &Instruction, state: &mut State) {
    if matches!(
        instr.flow_control(),
        FlowControl::Call | FlowControl::IndirectCall
    ) {
        for (i, known) in state.0.iter_mut().enumerate() {
            if is_win64_volatile(Register::RAX + i as u32) {
                *known = Known::Unknown;
            }
        }
        return;
    }
    let written: Vec<Register> = {
        let used = info.info(instr).used_registers();
        let mut out = Vec::new();
        for u in used {
            let full = u.register().full_register();
            if full.is_gpr64()
                && matches!(
                    u.access(),
                    OpAccess::Write
                        | OpAccess::CondWrite
                        | OpAccess::ReadWrite
                        | OpAccess::ReadCondWrite
                )
                && !out.contains(&full)
            {
                out.push(full);
            }
        }
        out
    };
    if written.is_empty() {
        return;
    }
    // The one register an exactly modelled instruction writes (operand 0).
    let modelled = model(instr, state);
    for reg in written {
        let value = match modelled {
            Some((dst, v)) if dst == reg => v,
            _ => Known::Unknown,
        };
        state.set(reg, value);
    }
}

/// Exact effect of `instr` on its destination register (operand 0), if modelled.
fn model(instr: &Instruction, state: &State) -> Option<(Register, Known)> {
    if instr.op_count() < 1 || instr.op0_kind() != OpKind::Register {
        return None;
    }
    let dst = instr.op0_register();
    let full = dst.full_register();
    if !full.is_gpr64() {
        return None;
    }
    let d64 = dst.is_gpr64();
    let d32 = dst.is_gpr32();
    if !(d64 || d32) {
        // 8/16-bit partial writes keep the other bits: not modelled.
        return None;
    }
    let width = if d64 { 64 } else { 32 };
    let src_reg = (instr.op_count() == 2 && instr.op1_kind() == OpKind::Register)
        .then(|| instr.op1_register());
    let imm = (instr.op_count() == 2)
        .then(|| instr.try_immediate(1).ok())
        .flatten();
    let cur = state.get(full);
    let known = match instr.mnemonic() {
        Mnemonic::Lea if d64 && instr.is_ip_rel_memory_operand() => {
            Known::exact(instr.ip_rel_memory_address())
        }
        Mnemonic::Mov if imm.is_some() => Known::exact(imm? & mask(width)),
        Mnemonic::Xor if src_reg == Some(dst) => Known::exact(0),
        Mnemonic::Mov if src_reg.is_some_and(|s| s.is_gpr64()) && d64 => state.get(src_reg?),
        Mnemonic::Mov if src_reg.is_some_and(|s| s.is_gpr32()) && d32 => {
            let s = state.get(src_reg?);
            if s.within_u32() {
                s
            } else {
                Known::Range {
                    lo: 0,
                    hi: mask(32),
                }
            }
        }
        Mnemonic::Mov if instr.op_count() == 2 && instr.op1_kind() == OpKind::Memory && d32 => {
            Known::Range {
                lo: 0,
                hi: mask(32),
            }
        }
        Mnemonic::Movzx if instr.op_count() == 2 => {
            let bits = match instr.op1_kind() {
                OpKind::Register => {
                    let s = instr.op1_register();
                    if matches!(s, Register::AH | Register::BH | Register::CH | Register::DH) {
                        8
                    } else {
                        s.size() as u32 * 8
                    }
                }
                OpKind::Memory => instr.memory_size().size() as u32 * 8,
                _ => return None,
            };
            let m = mask(bits);
            match (instr.op1_kind(), src_reg.map(|s| state.get(s))) {
                (OpKind::Register, Some(Known::Range { lo, hi }))
                    if hi <= m
                        && !matches!(
                            src_reg,
                            Some(Register::AH | Register::BH | Register::CH | Register::DH)
                        ) =>
                {
                    Known::Range { lo, hi }
                }
                _ => Known::Range { lo: 0, hi: m },
            }
        }
        Mnemonic::Movsxd if d64 && src_reg.is_some_and(|s| s.is_gpr32()) => {
            match state.get(src_reg?) {
                Known::Range { lo, hi } if hi <= 0x7fff_ffff => Known::Range { lo, hi },
                _ => Known::Unknown,
            }
        }
        Mnemonic::And if imm.is_some() => {
            let m = imm? & mask(width);
            let top = if width == 64 && (m as i64) < 0 {
                None
            } else {
                Some(m)
            };
            match (top, cur) {
                (Some(m), Known::Range { hi, .. }) => Known::Range {
                    lo: 0,
                    hi: hi.min(m),
                },
                (Some(m), _) => Known::Range { lo: 0, hi: m },
                (None, _) => Known::Unknown,
            }
        }
        Mnemonic::Shr if imm.is_some() => {
            let s = (imm? & 63) as u32;
            let base = if d32 {
                if cur.within_u32() {
                    cur
                } else {
                    Known::Range {
                        lo: 0,
                        hi: mask(32),
                    }
                }
            } else {
                cur
            };
            match base {
                Known::Range { lo, hi } => Known::Range {
                    lo: lo >> s,
                    hi: hi >> s,
                },
                _ => Known::Range {
                    lo: 0,
                    hi: mask(width) >> s,
                },
            }
        }
        Mnemonic::Add | Mnemonic::Sub | Mnemonic::Inc | Mnemonic::Dec => {
            let delta: i128 = match instr.mnemonic() {
                Mnemonic::Inc => 1,
                Mnemonic::Dec => -1,
                _ => {
                    let v = imm?;
                    let signed = if width == 64 {
                        i128::from(v as i64)
                    } else {
                        i128::from(v as u32 as i32)
                    };
                    if instr.mnemonic() == Mnemonic::Sub {
                        -signed
                    } else {
                        signed
                    }
                }
            };
            let base = if d32 && !cur.within_u32() {
                Known::Unknown
            } else {
                cur
            };
            match base {
                Known::Range { lo, hi } => {
                    let nlo = i128::from(lo) + delta;
                    let nhi = i128::from(hi) + delta;
                    if nlo < 0 || nhi > i128::from(mask(width)) {
                        if d32 {
                            Known::Range {
                                lo: 0,
                                hi: mask(32),
                            }
                        } else {
                            Known::Unknown
                        }
                    } else {
                        Known::Range {
                            lo: nlo as u64,
                            hi: nhi as u64,
                        }
                    }
                }
                _ if d32 => Known::Range {
                    lo: 0,
                    hi: mask(32),
                },
                _ => Known::Unknown,
            }
        }
        // Any other unconditional write to a 32-bit register zero-extends: bounded by 2^32.
        _ if d32 => Known::Range {
            lo: 0,
            hi: mask(32),
        },
        _ => return None,
    };
    Some((full, known))
}

/// Refine `state` for the edge `edge` out of `block` (which ends in a conditional branch).
fn refine_edge(
    info: &mut InstructionInfoFactory,
    block: &BasicBlock,
    edge: Edge,
    state: &mut State,
) {
    let Some(jcc) = block.instructions.last() else {
        return;
    };
    let taken = match edge {
        Edge::Taken(_) => true,
        Edge::Fallthrough(_) => false,
        _ => return,
    };
    // Taken and fallthrough to the same block: no refinement.
    if block
        .successors
        .iter()
        .filter(|e| e.intra_target() == edge.intra_target())
        .count()
        > 1
    {
        return;
    }
    let n = block.instructions.len();
    for i in (0..n.saturating_sub(1)).rev() {
        let instr = &block.instructions[i];
        if instr.rflags_modified() == 0 {
            continue;
        }
        if instr.op_count() < 1 || instr.op0_kind() != OpKind::Register {
            return;
        }
        let reg = instr.op0_register();
        if !(reg.is_gpr32() || reg.is_gpr64())
            || block.instructions[i + 1..n - 1]
                .iter()
                .any(|k| writes(info, k, reg))
        {
            return;
        }
        let bits = reg.size() as u32 * 8;
        let full = reg.full_register();
        let cur = state.get(full);
        // A 32-bit compare bounds the full register only when its upper half is known zero.
        if bits == 32 && !cur.within_u32() {
            return;
        }
        let (lo, hi) = match cur {
            Known::Range { lo, hi } => (lo, hi),
            _ => (0, mask(bits)),
        };
        let fact = match instr.mnemonic() {
            Mnemonic::Cmp => {
                let Ok(raw) = instr.try_immediate(1) else {
                    return;
                };
                let imm = raw & mask(bits);
                unsigned_fact(jcc.mnemonic(), taken, imm, lo, hi)
            }
            Mnemonic::Test
                if instr.op_count() == 2
                    && instr.op1_kind() == OpKind::Register
                    && instr.op1_register() == reg =>
            {
                match (jcc.mnemonic(), taken) {
                    (Mnemonic::Je, true) | (Mnemonic::Jne, false) => Some((0, 0)),
                    _ => None,
                }
            }
            _ => None,
        };
        if let Some((nlo, nhi)) = fact {
            let nlo = nlo.max(lo);
            let nhi = nhi.min(hi);
            if nlo <= nhi {
                state.set(full, Known::Range { lo: nlo, hi: nhi });
            }
        }
        return;
    }
}

/// `[lo, hi]` implied for the compared register by `cmp reg, imm` + `jcc` on the `taken` side,
/// given its current range `[cur_lo, cur_hi]` (signed conditions only when that is non-negative
/// below 2^31).
fn unsigned_fact(
    jcc: Mnemonic,
    taken: bool,
    imm: u64,
    cur_lo: u64,
    cur_hi: u64,
) -> Option<(u64, u64)> {
    let nonneg = cur_hi <= 0x7fff_ffff && imm <= 0x7fff_ffff;
    let (op, holds) = match jcc {
        Mnemonic::Ja => ("gt", taken),
        Mnemonic::Jae => ("ge", taken),
        Mnemonic::Jb => ("ge", !taken),
        Mnemonic::Jbe => ("gt", !taken),
        Mnemonic::Jg if nonneg => ("gt", taken),
        Mnemonic::Jge if nonneg => ("ge", taken),
        Mnemonic::Jl if nonneg => ("ge", !taken),
        Mnemonic::Jle if nonneg => ("gt", !taken),
        Mnemonic::Je if taken => return Some((imm, imm)),
        Mnemonic::Jne if !taken => return Some((imm, imm)),
        _ => return None,
    };
    let _ = cur_lo;
    // holds: reg > imm (gt) / reg >= imm (ge); otherwise the negation.
    match (op, holds) {
        ("gt", true) => imm.checked_add(1).map(|l| (l, u64::MAX)),
        ("gt", false) => Some((0, imm)),
        ("ge", true) => Some((imm, u64::MAX)),
        ("ge", false) => imm.checked_sub(1).map(|h| (0, h)),
        _ => None,
    }
}

/// Synthetic instructions that state `known` for `reg` in a form the jump-table slicer reads:
/// `lea reg, [rip+v]` for a single in-image address, else `cmp reg, hi` with a not-taken `ja`,
/// plus `cmp reg, lo` with a not-taken `jb` when `lo > 0`.
///
/// Returns the instructions with whether each is a not-taken conditional branch; empty when
/// nothing useful can be stated.
pub(crate) fn synthesize(
    reg: Register,
    known: Known,
    image_contains: &dyn Fn(u64) -> bool,
) -> Vec<(Instruction, bool)> {
    let Known::Range { lo, hi } = known else {
        return Vec::new();
    };
    let n = reg.full_register().number() as u8;
    if lo == hi && image_contains(lo) {
        let bytes = [
            0x48 | ((n >> 3) << 2),
            0x8D,
            ((n & 7) << 3) | 0x05,
            0,
            0,
            0,
            0,
        ];
        return decode(&bytes, lo.wrapping_sub(7))
            .filter(|i| i.op0_register() == reg.full_register() && i.ip_rel_memory_address() == lo)
            .map(|i| vec![(i, false)])
            .unwrap_or_default();
    }
    if hi > u64::from(u32::MAX) {
        return Vec::new();
    }
    // `cmp r32, imm32` compares the low half; valid because `hi < 2^32` bounds the whole register.
    let cmp = |imm: u64| -> Option<Instruction> {
        let mut bytes = Vec::with_capacity(7);
        if n >= 8 {
            bytes.push(0x41);
        }
        bytes.push(0x81);
        bytes.push(0xF8 | (n & 7));
        bytes.extend_from_slice(&(imm as u32).to_le_bytes());
        decode(&bytes, 0)
    };
    let jcc = |opcode: u8| decode(&[0x0F, opcode, 0, 0, 0, 0], 0);
    let mut out = Vec::new();
    match (cmp(hi), jcc(0x87)) {
        (Some(c), Some(j)) => {
            out.push((c, false));
            out.push((j, true));
        }
        _ => return Vec::new(),
    }
    if lo > 0
        && let (Some(c), Some(j)) = (cmp(lo), jcc(0x82))
    {
        out.push((c, false));
        out.push((j, true));
    }
    out
}

fn decode(bytes: &[u8], ip: u64) -> Option<Instruction> {
    let mut decoder = Decoder::with_ip(64, bytes, ip, DecoderOptions::NONE);
    let instr = decoder.decode();
    (!instr.is_invalid() && instr.len() == bytes.len()).then_some(instr)
}
