//! Per-function control-flow graph produced by [`crate::discover`] and consumed by the lifter.
//!
//! Invariants:
//! - Every [`Edge::Fallthrough`], [`Edge::Taken`] and [`Edge::Jump`] target of a block has a block
//!   with that start address in the same [`Function`].
//! - [`Edge::TailCall`] targets are entries of other discovered functions.
//! - A block with no instructions marks an address that could not be decoded; its
//!   [`BasicBlock::diagnostics`] say why and its terminator is [`Terminator::Invalid`].

use std::collections::{BTreeMap, BTreeSet};

use iced_x86::Instruction;

/// Why a basic block ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Terminator {
    /// No control-flow instruction: the next instruction starts another block.
    Fallthrough,
    /// Conditional branch (`jcc`, `loop*`, `jrcxz`, `xbegin`).
    Branch,
    /// Direct unconditional `jmp`.
    Jump,
    /// Indirect `jmp reg` / `jmp [mem]`.
    IndirectJump,
    /// `ret` (and other return-class instructions).
    Return,
    /// Call to a target listed in [`crate::code_image::CodeImage::no_return`].
    NoReturnCall,
    /// `int3`, `int1`, `int 0x29` (`__fastfail`), `ud0`/`ud1`/`ud2`, `hlt`.
    Trap,
    /// Decoding stopped on an invalid, truncated or unmapped instruction (see diagnostics). The
    /// faulting instruction is not part of the block.
    Invalid,
}

impl Terminator {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fallthrough => "fallthrough",
            Self::Branch => "branch",
            Self::Jump => "jump",
            Self::IndirectJump => "indirect_jump",
            Self::Return => "return",
            Self::NoReturnCall => "no_return_call",
            Self::Trap => "trap",
            Self::Invalid => "invalid",
        }
    }
}

/// Typed successor edge of a basic block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Edge {
    /// Sequential flow into the next block (also the not-taken side of a conditional branch).
    Fallthrough(u64),
    /// Taken side of a conditional branch.
    Taken(u64),
    /// Direct unconditional jump inside the function.
    Jump(u64),
    /// Control leaves to the entry of another function without pushing a return address
    /// (`jmp`, `jcc` or fallthrough into a known function entry). The block's terminator tells
    /// which form it was.
    TailCall(u64),
    /// Unresolved indirect jump. `slot` is the VA read by a RIP-relative `jmp [rip+x]` (import
    /// thunk / IAT entry); `None` for register or computed memory operands (e.g. jump tables).
    IndirectJump { slot: Option<u64> },
    /// Return to the caller.
    Return,
}

impl Edge {
    /// Target VA for direct edges (including tail calls).
    pub fn target(&self) -> Option<u64> {
        match *self {
            Self::Fallthrough(t) | Self::Taken(t) | Self::Jump(t) | Self::TailCall(t) => Some(t),
            Self::IndirectJump { .. } | Self::Return => None,
        }
    }

    /// Target VA for edges that stay inside the function.
    pub fn intra_target(&self) -> Option<u64> {
        match *self {
            Self::Fallthrough(t) | Self::Taken(t) | Self::Jump(t) => Some(t),
            _ => None,
        }
    }

    pub fn kind_str(&self) -> &'static str {
        match self {
            Self::Fallthrough(_) => "fallthrough",
            Self::Taken(_) => "taken",
            Self::Jump(_) => "jump",
            Self::TailCall(_) => "tail_call",
            Self::IndirectJump { .. } => "indirect_jump",
            Self::Return => "return",
        }
    }
}

/// Callee of a `call` instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CallTarget {
    /// `call rel32`.
    Direct(u64),
    /// `call reg` / `call [mem]`. `slot` is the VA read by a RIP-relative `call [rip+x]`.
    Indirect { slot: Option<u64> },
}

/// A `call` instruction inside a block. Calls do not end blocks unless the target is known not to
/// return.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CallSite {
    /// VA of the `call` instruction.
    pub site: u64,
    pub target: CallTarget,
}

/// Category of a discovery problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DiagnosticKind {
    /// The bytes at `address` do not decode to a valid instruction.
    InvalidInstruction,
    /// The instruction at `address` runs past the end of its section's bytes.
    TruncatedInstruction,
    /// `address` is not backed by executable section bytes.
    NotExecutable,
    /// `address` is not inside any section.
    Unmapped,
    /// Decoding at `site` overlaps a differently aligned instruction stream at `address`.
    OverlappingInstruction,
}

impl DiagnosticKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InvalidInstruction => "invalid_instruction",
            Self::TruncatedInstruction => "truncated_instruction",
            Self::NotExecutable => "not_executable",
            Self::Unmapped => "unmapped",
            Self::OverlappingInstruction => "overlapping_instruction",
        }
    }
}

/// A non-fatal problem found while decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Diagnostic {
    /// VA of the responsible instruction or block start.
    pub site: u64,
    /// The offending address (equal to `site` for decode failures).
    pub address: u64,
    pub kind: DiagnosticKind,
}

/// Straight-line run of instructions with a single entry at `start`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasicBlock {
    /// VA of the first instruction.
    pub start: u64,
    /// VA one past the last instruction (`== start` for an empty block).
    pub end: u64,
    /// Instructions in address order; the last one is the terminator unless the block falls
    /// through or ends in [`Terminator::Invalid`].
    pub instructions: Vec<Instruction>,
    pub terminator: Terminator,
    /// Successors: `[Taken, Fallthrough]` for branches, one edge otherwise, none for traps,
    /// non-returning calls and invalid blocks.
    pub successors: Vec<Edge>,
    /// Calls inside this block, in address order.
    pub calls: Vec<CallSite>,
    /// Problems found in this block, in address order.
    pub diagnostics: Vec<Diagnostic>,
}

impl BasicBlock {
    pub(crate) fn new(start: u64) -> Self {
        Self {
            start,
            end: start,
            instructions: Vec::new(),
            terminator: Terminator::Fallthrough,
            successors: Vec::new(),
            calls: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    /// Whether `va` is inside `[start, end)`.
    pub fn contains(&self, va: u64) -> bool {
        va >= self.start && va < self.end
    }
}

/// A discovered function: its entry plus every block reachable through intra-function edges.
/// Blocks reachable from several functions (shared tails) are copied into each of them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub entry: u64,
    /// Keyed by block start VA.
    pub blocks: BTreeMap<u64, BasicBlock>,
    /// Entries of functions reached through direct calls.
    pub callees: BTreeSet<u64>,
}

impl Function {
    pub fn entry_block(&self) -> Option<&BasicBlock> {
        self.blocks.get(&self.entry)
    }

    pub fn instruction_count(&self) -> usize {
        self.blocks.values().map(|b| b.instructions.len()).sum()
    }

    /// All calls in address order.
    pub fn calls(&self) -> impl Iterator<Item = &CallSite> {
        self.blocks.values().flat_map(|b| b.calls.iter())
    }

    /// All diagnostics in block order.
    pub fn diagnostics(&self) -> impl Iterator<Item = &Diagnostic> {
        self.blocks.values().flat_map(|b| b.diagnostics.iter())
    }

    /// All tail-call targets in block order.
    pub fn tail_calls(&self) -> impl Iterator<Item = u64> + '_ {
        self.blocks
            .values()
            .flat_map(|b| b.successors.iter())
            .filter_map(|e| match e {
                Edge::TailCall(t) => Some(*t),
                _ => None,
            })
    }
}
