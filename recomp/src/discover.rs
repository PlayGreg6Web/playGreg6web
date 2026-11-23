//! Recursive-descent function discovery over a [`CodeImage`].
//!
//! Discovery runs in two phases so that results do not depend on the order in which code is
//! visited:
//! 1. A worklist decodes every block reachable from the entry points into one global block map.
//!    Direct `call` targets become function entries, direct branch targets become block leaders,
//!    and a leader found inside an already decoded block splits that block in place. A leader
//!    inside an instruction starts a second, overlapping stream (flagged with a diagnostic) that
//!    stops once it falls back into step with already decoded code.
//!
//!    Once the worklist is empty, every new register `jmp` is handed to [`crate::jump_table`]
//!    together with its chain of sole predecessor blocks. A resolved switch table turns the jump's
//!    successors into [`Edge::JumpTable`] edges and its targets into leaders, and the worklist
//!    runs again; this repeats until a round resolves nothing new. Each round only sees the
//!    complete block map of the previous fixpoint, so results stay independent of visiting order.
//!    Finally every resolution is re-checked against the final CFG (a later-discovered jump into
//!    the dispatch path voids the bound) and demoted back to [`Edge::IndirectJump`] if it no
//!    longer holds.
//! 2. Each function collects the blocks reachable from its entry through intra-function edges.
//!    Edges into another function's entry are reclassified as [`Edge::TailCall`].
//!
//! Both phases are iterative (no recursion proportional to code size). Block splitting is
//! `O(log n + block length)` and the per-instruction hot path does no heap formatting. Each
//! jump-table round costs one `O(blocks)` predecessor pass.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use iced_x86::{Decoder, DecoderError, DecoderOptions, FlowControl, Instruction, Mnemonic, OpKind};

use crate::cfg::{
    BasicBlock, CallSite, CallTarget, Diagnostic, DiagnosticKind, Edge, Function, JumpTable,
    Terminator,
};
use crate::code_image::CodeImage;
use crate::jump_table::{self, Step};
use crate::reloc::{RelocKind, Relocation, RelocationMap, instruction_relocations};

/// `int 0x29` is the Windows `__fastfail` trap and never returns.
const FASTFAIL_VECTOR: u8 = 0x29;

/// Preceding blocks examined when looking for the block that contains an address.
const CONTAINING_BLOCK_SCAN: usize = 8;

/// Longest chain of blocks (the jump's block plus sole predecessors) sliced for a jump table.
const MAX_TRACE_BLOCKS: usize = 4;

/// Incoming intra-function edges of a block: how many, and one of them.
#[derive(Clone, Copy)]
struct Pred {
    count: u32,
    from: u64,
    edge: Edge,
}

/// Where an address falls relative to already decoded blocks.
enum Decoded {
    /// An instruction of the block at `start` (at `index`) begins at the address.
    Boundary { start: u64, index: usize },
    /// The address is inside a decoded instruction.
    InsideInstruction,
    /// Not decoded yet.
    No,
}

/// Result of [`discover`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Analysis {
    /// Discovered functions keyed by entry VA.
    pub functions: BTreeMap<u64, Function>,
    /// Address references of every decoded instruction and of every resolved jump table.
    pub relocations: RelocationMap,
    /// Resolved switch jump tables keyed by the VA of their indirect `jmp`.
    pub jump_tables: BTreeMap<u64, JumpTable>,
}

impl Analysis {
    /// Function whose entry is `entry`.
    pub fn function(&self, entry: u64) -> Option<&Function> {
        self.functions.get(&entry)
    }
}

/// Discover functions, per-function CFGs, jump tables and the relocation map starting from
/// `image.entry_points`. Never panics on malformed code: problems become [`Diagnostic`]s.
pub fn discover(image: &CodeImage<'_>) -> Analysis {
    let mut state = Discovery::new(image);
    for &entry in &image.entry_points {
        state.add_function(entry);
    }
    state.run();
    while state.resolve_jump_tables() {
        state.run();
    }
    state.verify_jump_tables();

    let mut relocations = Vec::new();
    for instr in state.blocks.values().flat_map(|b| b.instructions.iter()) {
        instruction_relocations(image, instr, &mut relocations);
    }
    let jump_tables = std::mem::take(&mut state.jump_tables);
    for table in jump_tables.values() {
        jump_table_relocations(image, table, &mut relocations);
    }
    let relocations = RelocationMap::from_entries(relocations);
    let functions = state.into_functions();
    Analysis {
        functions,
        relocations,
        jump_tables,
    }
}

/// Relocations of a resolved table: the table and index-table reads plus one per entry.
fn jump_table_relocations(image: &CodeImage<'_>, table: &JumpTable, out: &mut Vec<Relocation>) {
    let mut push = |site, kind, target| {
        out.push(Relocation {
            site,
            kind,
            target,
            region: image.region_of(target),
        });
    };
    push(table.load_site, RelocKind::JumpTable, table.table);
    if let Some(index) = table.index {
        push(index.load_site, RelocKind::JumpTableIndex, index.table);
    }
    for (slot, &target) in (table.table..).step_by(4).zip(&table.targets) {
        push(slot, RelocKind::JumpTableEntry, target);
    }
}

struct Discovery<'i, 'a> {
    image: &'i CodeImage<'a>,
    no_return: HashSet<u64>,
    /// Every block decoded so far, keyed by start VA.
    blocks: BTreeMap<u64, BasicBlock>,
    /// Every known block start (decoded or still queued).
    leaders: BTreeSet<u64>,
    worklist: Vec<u64>,
    entries: BTreeSet<u64>,
    /// `jmp reg` sites decoded since the last jump-table round.
    pending_jumps: BTreeSet<u64>,
    jump_tables: BTreeMap<u64, JumpTable>,
}

impl<'i, 'a> Discovery<'i, 'a> {
    fn new(image: &'i CodeImage<'a>) -> Self {
        Self {
            image,
            no_return: image.no_return.iter().copied().collect(),
            blocks: BTreeMap::new(),
            leaders: BTreeSet::new(),
            worklist: Vec::new(),
            entries: BTreeSet::new(),
            pending_jumps: BTreeSet::new(),
            jump_tables: BTreeMap::new(),
        }
    }

    fn add_leader(&mut self, va: u64) {
        if self.leaders.insert(va) {
            self.worklist.push(va);
        }
    }

    fn add_function(&mut self, va: u64) {
        self.entries.insert(va);
        self.add_leader(va);
    }

    fn run(&mut self) {
        while let Some(va) = self.worklist.pop() {
            if self.blocks.contains_key(&va) {
                continue;
            }
            match self.locate(va) {
                Decoded::Boundary { start, index } => self.split(start, index),
                // `va` is inside an instruction: decode a second, overlapping stream.
                Decoded::InsideInstruction => self.decode_block(va, true),
                Decoded::No => self.decode_block(va, false),
            }
        }
    }

    /// Find an already decoded block covering `va`. Only overlapping instruction streams put
    /// other blocks between `va` and its containing block, so a short backwards scan suffices.
    fn locate(&self, va: u64) -> Decoded {
        let mut inside = false;
        for (&start, block) in self.blocks.range(..=va).rev().take(CONTAINING_BLOCK_SCAN) {
            if !block.contains(va) {
                continue;
            }
            match block
                .instructions
                .binary_search_by_key(&va, Instruction::ip)
            {
                Ok(index) => return Decoded::Boundary { start, index },
                Err(_) => inside = true,
            }
        }
        if inside {
            Decoded::InsideInstruction
        } else {
            Decoded::No
        }
    }

    /// Split the block at `start` so that its `index`-th instruction starts a new block.
    fn split(&mut self, start: u64, index: usize) {
        let Some(head) = self.blocks.get_mut(&start) else {
            return;
        };
        if index == 0 || index >= head.instructions.len() {
            return;
        }
        let at = head.instructions[index].ip();
        let call_split = head.calls.partition_point(|c| c.site < at);
        let diag_split = head.diagnostics.partition_point(|d| d.site < at);
        let tail = BasicBlock {
            start: at,
            end: head.end,
            instructions: head.instructions.split_off(index),
            terminator: head.terminator,
            successors: std::mem::replace(&mut head.successors, vec![Edge::Fallthrough(at)]),
            calls: head.calls.split_off(call_split),
            diagnostics: head.diagnostics.split_off(diag_split),
        };
        head.end = at;
        head.terminator = Terminator::Fallthrough;
        self.blocks.insert(at, tail);
    }

    /// First known leader at or after `va` (`u64::MAX` if none).
    fn next_leader_from(&self, va: u64) -> u64 {
        self.leaders.range(va..).next().copied().unwrap_or(u64::MAX)
    }

    fn decode_block(&mut self, start: u64, overlaps: bool) {
        let mut block = BasicBlock::new(start);
        let bytes = match self.image.code_bytes(start) {
            Ok(bytes) => bytes,
            Err(kind) => {
                block.terminator = Terminator::Invalid;
                block.diagnostics.push(Diagnostic {
                    site: start,
                    address: start,
                    kind,
                });
                self.blocks.insert(start, block);
                return;
            }
        };
        if overlaps {
            block.diagnostics.push(Diagnostic {
                site: start,
                address: start,
                kind: DiagnosticKind::OverlappingInstruction,
            });
        }

        let mut decoder = Decoder::with_ip(64, bytes, start, DecoderOptions::NONE);
        let mut instr = Instruction::default();
        let mut stop_at = start
            .checked_add(1)
            .map_or(u64::MAX, |va| self.next_leader_from(va));
        // Set once this stream is out of step with already decoded code; it then stops as soon as
        // it lands on an existing instruction boundary instead of duplicating that block's tail.
        let mut misaligned = overlaps;
        loop {
            let ip = decoder.ip();
            if ip == stop_at {
                block.successors.push(Edge::Fallthrough(ip));
                break;
            }
            if misaligned
                && ip != start
                && let Decoded::Boundary {
                    start: other,
                    index,
                } = self.locate(ip)
            {
                self.leaders.insert(ip);
                self.split(other, index);
                block.successors.push(Edge::Fallthrough(ip));
                break;
            }
            if !decoder.can_decode() {
                // Ran off the end of the section bytes: continue at the next address, which is
                // either an adjacent executable section or becomes an invalid block.
                block.successors.push(Edge::Fallthrough(ip));
                self.add_leader(ip);
                break;
            }
            decoder.decode_out(&mut instr);
            if instr.is_invalid() {
                let kind = if decoder.last_error() == DecoderError::NoMoreBytes {
                    DiagnosticKind::TruncatedInstruction
                } else {
                    DiagnosticKind::InvalidInstruction
                };
                block.terminator = Terminator::Invalid;
                block.diagnostics.push(Diagnostic {
                    site: ip,
                    address: ip,
                    kind,
                });
                break;
            }
            let next = instr.next_ip();
            if stop_at > ip && stop_at < next {
                // This instruction straddles a known leader: keep decoding this stream.
                block.diagnostics.push(Diagnostic {
                    site: ip,
                    address: stop_at,
                    kind: DiagnosticKind::OverlappingInstruction,
                });
                stop_at = self.next_leader_from(next);
                misaligned = true;
            }
            block.instructions.push(instr);
            block.end = next;

            if let Some(terminator) = self.classify(&instr, &mut block, &mut stop_at) {
                block.terminator = terminator;
                break;
            }
        }
        self.blocks.insert(start, block);
    }

    /// Record control flow of `instr` (already appended to `block`). Returns the terminator if
    /// the instruction ends the block.
    fn classify(
        &mut self,
        instr: &Instruction,
        block: &mut BasicBlock,
        stop_at: &mut u64,
    ) -> Option<Terminator> {
        let ip = instr.ip();
        match instr.flow_control() {
            FlowControl::Call if instr.op_kind(0) == OpKind::NearBranch64 => {
                let target = instr.near_branch64();
                block.calls.push(CallSite {
                    site: ip,
                    target: CallTarget::Direct(target),
                });
                match self.image.code_bytes(target) {
                    Ok(_) => {
                        self.add_function(target);
                        if target > ip && target < *stop_at {
                            *stop_at = target;
                        }
                    }
                    Err(kind) => block.diagnostics.push(Diagnostic {
                        site: ip,
                        address: target,
                        kind,
                    }),
                }
                self.no_return
                    .contains(&target)
                    .then_some(Terminator::NoReturnCall)
            }
            FlowControl::IndirectCall => {
                let slot = rip_slot(instr);
                block.calls.push(CallSite {
                    site: ip,
                    target: CallTarget::Indirect { slot },
                });
                slot.filter(|s| self.no_return.contains(s))
                    .map(|_| Terminator::NoReturnCall)
            }
            FlowControl::UnconditionalBranch if instr.op_kind(0) == OpKind::NearBranch64 => {
                let target = instr.near_branch64();
                block.successors.push(Edge::Jump(target));
                self.add_leader(target);
                Some(Terminator::Jump)
            }
            FlowControl::UnconditionalBranch | FlowControl::IndirectBranch => {
                block.successors.push(Edge::IndirectJump {
                    slot: rip_slot(instr),
                });
                if instr.op0_kind() == OpKind::Register {
                    self.pending_jumps.insert(ip);
                }
                Some(Terminator::IndirectJump)
            }
            FlowControl::ConditionalBranch => self.branch(instr, block),
            FlowControl::XbeginXabortXend if instr.mnemonic() == Mnemonic::Xbegin => {
                self.branch(instr, block)
            }
            FlowControl::Return => {
                block.successors.push(Edge::Return);
                Some(Terminator::Return)
            }
            FlowControl::Interrupt => {
                let traps = match instr.mnemonic() {
                    Mnemonic::Int3 | Mnemonic::Int1 | Mnemonic::Into => true,
                    Mnemonic::Int => instr.immediate8() == FASTFAIL_VECTOR,
                    _ => false,
                };
                traps.then_some(Terminator::Trap)
            }
            FlowControl::Exception => Some(Terminator::Trap),
            _ if instr.mnemonic() == Mnemonic::Hlt => Some(Terminator::Trap),
            // Plain instructions, `syscall`, `int n`, `xabort`/`xend`.
            _ => None,
        }
    }

    fn branch(&mut self, instr: &Instruction, block: &mut BasicBlock) -> Option<Terminator> {
        let taken = instr.near_branch_target();
        let next = instr.next_ip();
        block.successors.push(Edge::Taken(taken));
        block.successors.push(Edge::Fallthrough(next));
        self.add_leader(taken);
        self.add_leader(next);
        Some(Terminator::Branch)
    }

    /// Incoming intra-function edges of every block.
    fn predecessors(&self) -> HashMap<u64, Pred> {
        let mut preds: HashMap<u64, Pred> = HashMap::new();
        for (&from, block) in &self.blocks {
            for &edge in &block.successors {
                if let Some(target) = edge.intra_target() {
                    preds
                        .entry(target)
                        .and_modify(|p| p.count += 1)
                        .or_insert(Pred {
                            count: 1,
                            from,
                            edge,
                        });
                }
            }
        }
        preds
    }

    /// The block whose last instruction is the indirect jump at `site`.
    fn jump_block(&self, site: u64) -> Option<u64> {
        let Decoded::Boundary { start, index } = self.locate(site) else {
            return None;
        };
        let block = self.blocks.get(&start)?;
        (index + 1 == block.instructions.len() && block.terminator == Terminator::IndirectJump)
            .then_some(start)
    }

    /// Slice the jump at `site` together with its chain of sole predecessors.
    fn match_jump_table(&self, site: u64, preds: &HashMap<u64, Pred>) -> Option<JumpTable> {
        let start = self.jump_block(site)?;
        let mut chain = vec![(self.blocks.get(&start)?, None)];
        let mut current = start;
        while chain.len() < MAX_TRACE_BLOCKS && !self.entries.contains(&current) {
            let Some(pred) = preds.get(&current).filter(|p| p.count == 1) else {
                break;
            };
            let Some(prev) = self.blocks.get(&pred.from) else {
                break;
            };
            let taken = match pred.edge {
                Edge::Taken(_) => Some(true),
                Edge::Fallthrough(_) if prev.terminator == Terminator::Branch => Some(false),
                Edge::Fallthrough(_) | Edge::Jump(_) => None,
                _ => break,
            };
            chain.push((prev, taken));
            current = pred.from;
        }
        let mut trace = Vec::new();
        for (block, taken) in chain.into_iter().rev() {
            let last = block.instructions.len().saturating_sub(1);
            trace.extend(
                block
                    .instructions
                    .iter()
                    .enumerate()
                    .map(|(i, instr)| Step {
                        instr,
                        taken: if i == last { taken } else { None },
                    }),
            );
        }
        jump_table::resolve(self.image, &trace)
    }

    /// Try every jump decoded since the last round. Returns whether any table was resolved.
    fn resolve_jump_tables(&mut self) -> bool {
        let pending = std::mem::take(&mut self.pending_jumps);
        if pending.is_empty() {
            return false;
        }
        // One predecessor snapshot per round keeps the outcome independent of visiting order.
        let preds = self.predecessors();
        let resolved: Vec<JumpTable> = pending
            .into_iter()
            .filter_map(|site| self.match_jump_table(site, &preds))
            .collect();
        let progress = !resolved.is_empty();
        for table in resolved {
            if let Some(block) = self
                .jump_block(table.site)
                .and_then(|start| self.blocks.get_mut(&start))
            {
                let targets: BTreeSet<u64> = table.targets.iter().copied().collect();
                block.successors = targets.into_iter().map(Edge::JumpTable).collect();
            }
            for &target in &table.targets {
                self.add_leader(target);
            }
            self.jump_tables.insert(table.site, table);
        }
        progress
    }

    /// Demote resolutions that no longer hold in the final CFG (e.g. the dispatch block gained a
    /// second predecessor), until the remaining set is stable.
    fn verify_jump_tables(&mut self) {
        loop {
            let preds = self.predecessors();
            let stale: Vec<u64> = self
                .jump_tables
                .iter()
                .filter(|(site, table)| {
                    self.match_jump_table(**site, &preds).as_ref() != Some(*table)
                })
                .map(|(site, _)| *site)
                .collect();
            if stale.is_empty() {
                return;
            }
            for site in stale {
                self.jump_tables.remove(&site);
                if let Some(block) = self
                    .jump_block(site)
                    .and_then(|start| self.blocks.get_mut(&start))
                {
                    block.successors = vec![Edge::IndirectJump { slot: None }];
                }
            }
        }
    }

    /// Phase 2: assign blocks to functions and classify tail calls.
    fn into_functions(mut self) -> BTreeMap<u64, Function> {
        let entries: Vec<u64> = self.entries.iter().copied().collect();

        // Membership: blocks reachable from each entry through intra-function edges.
        let mut members: Vec<Vec<u64>> = Vec::with_capacity(entries.len());
        let mut refcount: HashMap<u64, u32> = HashMap::new();
        let mut seen: HashSet<u64> = HashSet::new();
        let mut stack: Vec<u64> = Vec::new();
        for &entry in &entries {
            seen.clear();
            stack.push(entry);
            let mut list = Vec::new();
            while let Some(va) = stack.pop() {
                if !seen.insert(va) {
                    continue;
                }
                let Some(block) = self.blocks.get(&va) else {
                    continue;
                };
                list.push(va);
                for edge in &block.successors {
                    if let Some(target) = edge.intra_target()
                        && (target == entry || !self.entries.contains(&target))
                    {
                        stack.push(target);
                    }
                }
            }
            for &va in &list {
                *refcount.entry(va).or_default() += 1;
            }
            members.push(list);
        }

        // Materialise: move a block into its last owner, clone it for earlier ones.
        let mut functions = BTreeMap::new();
        for (entry, list) in entries.into_iter().zip(members) {
            let mut function = Function {
                entry,
                blocks: BTreeMap::new(),
                callees: BTreeSet::new(),
            };
            for va in list {
                let remaining = refcount.get_mut(&va).map(|count| {
                    *count -= 1;
                    *count
                });
                let block = if remaining == Some(0) {
                    self.blocks.remove(&va)
                } else {
                    self.blocks.get(&va).cloned()
                };
                let Some(mut block) = block else {
                    continue;
                };
                for edge in &mut block.successors {
                    if let Some(target) = edge.intra_target()
                        && target != entry
                        && self.entries.contains(&target)
                    {
                        *edge = Edge::TailCall(target);
                    }
                }
                for call in &block.calls {
                    if let CallTarget::Direct(target) = call.target
                        && self.entries.contains(&target)
                    {
                        function.callees.insert(target);
                    }
                }
                function.blocks.insert(va, block);
            }
            functions.insert(entry, function);
        }
        functions
    }
}

/// Slot VA read by a RIP-relative indirect `call`/`jmp`.
fn rip_slot(instr: &Instruction) -> Option<u64> {
    (instr.op_count() > 0 && instr.op_kind(0) == OpKind::Memory && instr.is_ip_rel_memory_operand())
        .then(|| instr.ip_rel_memory_address())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code_image::{CodeSection, Region};
    use crate::reloc::{RelocKind, Relocation};

    const BASE: u64 = 0x1_4000_0000;
    const TEXT: u64 = BASE + 0x1000;
    const DATA: u64 = BASE + 0x2000;
    const DATA_BYTES: [u8; 0x20] = [0; 0x20];

    /// Analyse `code` mapped executable at `TEXT` (plus a data section at `DATA`) starting at
    /// the given `TEXT`-relative entry offsets.
    fn analyze(code: &[u8], entries: &[u64]) -> Analysis {
        analyze_with(code, entries, &[])
    }

    fn analyze_with(code: &[u8], entries: &[u64], no_return: &[u64]) -> Analysis {
        let mut image = CodeImage::new(
            BASE,
            vec![
                CodeSection::new(TEXT, code, true),
                CodeSection::new(DATA, &DATA_BYTES, false),
            ],
            entries.iter().map(|e| TEXT + e).collect(),
        );
        image.no_return = no_return.to_vec();
        discover(&image)
    }

    fn func(analysis: &Analysis, offset: u64) -> &Function {
        analysis
            .function(TEXT + offset)
            .unwrap_or_else(|| panic!("no function at TEXT+{offset:#x}"))
    }

    fn block(function: &Function, offset: u64) -> &BasicBlock {
        function
            .blocks
            .get(&(TEXT + offset))
            .unwrap_or_else(|| panic!("no block at TEXT+{offset:#x}"))
    }

    fn block_starts(function: &Function) -> Vec<u64> {
        function.blocks.keys().map(|va| va - TEXT).collect()
    }

    fn instr_offsets(block: &BasicBlock) -> Vec<u64> {
        block.instructions.iter().map(|i| i.ip() - TEXT).collect()
    }

    fn le64(value: u64) -> [u8; 8] {
        value.to_le_bytes()
    }

    #[test]
    fn straight_line_function() {
        let code = [
            0x48, 0x89, 0xc8, // 0: mov rax, rcx
            0x48, 0x83, 0xc0, 0x01, // 3: add rax, 1
            0xc3, // 7: ret
        ];
        let analysis = analyze(&code, &[0]);
        assert_eq!(analysis.functions.len(), 1);
        let f = func(&analysis, 0);
        assert_eq!(block_starts(f), [0]);
        let b = block(f, 0);
        assert_eq!(instr_offsets(b), [0, 3, 7]);
        assert_eq!(b.end, TEXT + 8);
        assert_eq!(b.terminator, Terminator::Return);
        assert_eq!(b.successors, [Edge::Return]);
        assert!(f.callees.is_empty());
        assert_eq!(f.diagnostics().count(), 0);
        assert!(analysis.relocations.is_empty());
    }

    #[test]
    fn if_else_diamond() {
        let code = [
            0x85, 0xc9, // 0: test ecx, ecx
            0x74, 0x07, // 2: je 0xb
            0xb8, 0x01, 0x00, 0x00, 0x00, // 4: mov eax, 1
            0xeb, 0x05, // 9: jmp 0x10
            0xb8, 0x02, 0x00, 0x00, 0x00, // b: mov eax, 2
            0xc3, // 10: ret
        ];
        let analysis = analyze(&code, &[0]);
        let f = func(&analysis, 0);
        assert_eq!(block_starts(f), [0x0, 0x4, 0xb, 0x10]);

        let head = block(f, 0);
        assert_eq!(head.terminator, Terminator::Branch);
        assert_eq!(
            head.successors,
            [Edge::Taken(TEXT + 0xb), Edge::Fallthrough(TEXT + 0x4)]
        );
        let then_block = block(f, 0x4);
        assert_eq!(then_block.terminator, Terminator::Jump);
        assert_eq!(then_block.successors, [Edge::Jump(TEXT + 0x10)]);
        let else_block = block(f, 0xb);
        assert_eq!(instr_offsets(else_block), [0xb]);
        assert_eq!(else_block.terminator, Terminator::Fallthrough);
        assert_eq!(else_block.successors, [Edge::Fallthrough(TEXT + 0x10)]);
        assert_eq!(block(f, 0x10).terminator, Terminator::Return);

        let branch_targets: Vec<_> = analysis
            .relocations
            .entries()
            .iter()
            .map(|r| (r.site - TEXT, r.kind, r.target - TEXT, r.region))
            .collect();
        assert_eq!(
            branch_targets,
            [
                (0x2, RelocKind::BranchTarget, 0xb, Region::Code),
                (0x9, RelocKind::BranchTarget, 0x10, Region::Code),
            ]
        );
    }

    #[test]
    fn loop_with_back_edge_splits_entry_block() {
        let code = [
            0x31, 0xc0, // 0: xor eax, eax
            0x01, 0xc8, // 2: add eax, ecx   <- loop head
            0xff, 0xc9, // 4: dec ecx
            0x75, 0xfa, // 6: jne 0x2
            0xc3, // 8: ret
        ];
        let analysis = analyze(&code, &[0]);
        let f = func(&analysis, 0);
        assert_eq!(block_starts(f), [0x0, 0x2, 0x8]);
        let entry = block(f, 0);
        assert_eq!(instr_offsets(entry), [0]);
        assert_eq!(entry.end, TEXT + 2);
        assert_eq!(entry.terminator, Terminator::Fallthrough);
        assert_eq!(entry.successors, [Edge::Fallthrough(TEXT + 2)]);
        let body = block(f, 0x2);
        assert_eq!(instr_offsets(body), [0x2, 0x4, 0x6]);
        assert_eq!(
            body.successors,
            [Edge::Taken(TEXT + 0x2), Edge::Fallthrough(TEXT + 0x8)]
        );
    }

    #[test]
    fn direct_call_discovers_second_function() {
        let code = [
            0x48, 0x83, 0xec, 0x28, // 0: sub rsp, 0x28
            0xe8, 0x07, 0x00, 0x00, 0x00, // 4: call 0x10
            0x48, 0x83, 0xc4, 0x28, // 9: add rsp, 0x28
            0xc3, // d: ret
            0xcc, 0xcc, // e: int3 padding (never decoded)
            0x8d, 0x41, 0x01, // 10: lea eax, [rcx+1]
            0xc3, // 13: ret
        ];
        let analysis = analyze(&code, &[0]);
        assert_eq!(
            analysis.functions.keys().copied().collect::<Vec<_>>(),
            [TEXT, TEXT + 0x10]
        );
        let caller = func(&analysis, 0);
        assert_eq!(block_starts(caller), [0], "calls do not end blocks");
        assert_eq!(instr_offsets(block(caller, 0)), [0x0, 0x4, 0x9, 0xd]);
        assert_eq!(
            caller.calls().copied().collect::<Vec<_>>(),
            [CallSite {
                site: TEXT + 4,
                target: CallTarget::Direct(TEXT + 0x10),
            }]
        );
        assert_eq!(
            caller.callees.iter().copied().collect::<Vec<_>>(),
            [TEXT + 0x10]
        );

        let callee = func(&analysis, 0x10);
        assert_eq!(block_starts(callee), [0x10]);
        assert_eq!(callee.instruction_count(), 2);
        assert_eq!(
            analysis.relocations.at(TEXT + 4),
            &[Relocation {
                site: TEXT + 4,
                kind: RelocKind::CallTarget,
                target: TEXT + 0x10,
                region: Region::Code,
            }]
        );
    }

    #[test]
    fn tail_calls_leave_the_function() {
        let code = [
            0xe8, 0x0b, 0x00, 0x00, 0x00, // 0: call 0x10
            0x85, 0xc9, // 5: test ecx, ecx
            0x74, 0x07, // 7: je 0x10        (conditional tail call)
            0xe9, 0x02, 0x00, 0x00, 0x00, // 9: jmp 0x10 (tail call)
            0xcc, 0xcc, // e: padding
            0xff, 0xc9, // 10: dec ecx
            0x75, 0xfc, // 12: jne 0x10     (jump to own entry: not a tail call)
            0xc3, // 14: ret
        ];
        let analysis = analyze(&code, &[0]);
        let caller = func(&analysis, 0);
        assert_eq!(block_starts(caller), [0x0, 0x9]);
        let head = block(caller, 0);
        assert_eq!(head.terminator, Terminator::Branch);
        assert_eq!(
            head.successors,
            [Edge::TailCall(TEXT + 0x10), Edge::Fallthrough(TEXT + 0x9)]
        );
        let jump = block(caller, 0x9);
        assert_eq!(jump.terminator, Terminator::Jump);
        assert_eq!(jump.successors, [Edge::TailCall(TEXT + 0x10)]);
        assert_eq!(
            caller.tail_calls().collect::<Vec<_>>(),
            [TEXT + 0x10, TEXT + 0x10]
        );

        let callee = func(&analysis, 0x10);
        assert_eq!(block_starts(callee), [0x10, 0x14]);
        assert_eq!(
            block(callee, 0x10).successors,
            [Edge::Taken(TEXT + 0x10), Edge::Fallthrough(TEXT + 0x14)]
        );
        assert_eq!(callee.tail_calls().count(), 0);
    }

    #[test]
    fn jump_into_another_entry_point_is_a_tail_call() {
        let code = [
            0x31, 0xc0, // 0: xor eax, eax
            0xeb, 0x00, // 2: jmp 0x4
            0xc3, // 4: ret  (second entry point)
        ];
        let analysis = analyze(&code, &[0, 4]);
        let first = func(&analysis, 0);
        assert_eq!(block_starts(first), [0]);
        assert_eq!(block(first, 0).successors, [Edge::TailCall(TEXT + 4)]);
        assert_eq!(block_starts(func(&analysis, 4)), [4]);
    }

    #[test]
    fn jump_into_middle_of_block_splits_it_and_moves_calls() {
        let code = [
            0xb8, 0x01, 0x00, 0x00, 0x00, // 0: mov eax, 1
            0x01, 0xc8, // 5: add eax, ecx   <- loop head
            0xe8, 0x09, 0x00, 0x00, 0x00, // 7: call 0x15
            0x85, 0xc0, // c: test eax, eax
            0x75, 0xf5, // e: jne 0x5
            0xc3, // 10: ret
            0xcc, 0xcc, 0xcc, 0xcc, // 11: padding
            0xc3, // 15: ret
        ];
        let analysis = analyze(&code, &[0]);
        let f = func(&analysis, 0);
        assert_eq!(block_starts(f), [0x0, 0x5, 0x10]);

        let head = block(f, 0);
        assert_eq!(instr_offsets(head), [0x0]);
        assert_eq!(head.end, TEXT + 5);
        assert_eq!(head.successors, [Edge::Fallthrough(TEXT + 5)]);
        assert!(head.calls.is_empty());

        let tail = block(f, 0x5);
        assert_eq!(instr_offsets(tail), [0x5, 0x7, 0xc, 0xe]);
        assert_eq!(tail.end, TEXT + 0x10);
        assert_eq!(tail.terminator, Terminator::Branch);
        assert_eq!(
            tail.successors,
            [Edge::Taken(TEXT + 0x5), Edge::Fallthrough(TEXT + 0x10)]
        );
        assert_eq!(tail.calls.len(), 1);
        assert_eq!(tail.calls[0].site, TEXT + 7);
        assert!(analysis.function(TEXT + 0x15).is_some());
    }

    #[test]
    fn jump_into_middle_of_instruction_decodes_overlapping_stream() {
        let code = [
            0x74, 0x01, // 0: je 0x3 (into the immediate of the next mov)
            0xb8, 0xc3, 0x00, 0x00, 0x00, // 2: mov eax, 0xc3 (byte 3 is also `ret`)
            0xc3, // 7: ret
        ];
        let analysis = analyze(&code, &[0]);
        let f = func(&analysis, 0);
        assert_eq!(block_starts(f), [0x0, 0x2, 0x3]);
        assert_eq!(instr_offsets(block(f, 0x2)), [0x2, 0x7]);
        let hidden = block(f, 0x3);
        assert_eq!(instr_offsets(hidden), [0x3]);
        assert_eq!(hidden.terminator, Terminator::Return);
        assert!(
            f.diagnostics()
                .all(|d| d.kind == DiagnosticKind::OverlappingInstruction)
        );
        assert!(f.diagnostics().any(|d| d.address == TEXT + 3));

        // A misaligned stream that falls back into step stops at the shared boundary, which
        // splits the original block instead of duplicating its tail.
        let code = [
            0x74, 0x01, // 0: je 0x3
            0xb8, 0x90, 0x90, 0x90, 0x90, // 2: mov eax, 0x90909090 (3..6 are also nops)
            0xc3, // 7: ret
        ];
        let analysis = analyze(&code, &[0]);
        let f = func(&analysis, 0);
        assert_eq!(block_starts(f), [0x0, 0x2, 0x3, 0x7]);
        assert_eq!(instr_offsets(block(f, 0x2)), [0x2]);
        assert_eq!(block(f, 0x2).successors, [Edge::Fallthrough(TEXT + 7)]);
        let nops = block(f, 0x3);
        assert_eq!(instr_offsets(nops), [0x3, 0x4, 0x5, 0x6]);
        assert_eq!(nops.successors, [Edge::Fallthrough(TEXT + 7)]);
        assert_eq!(block(f, 0x7).terminator, Terminator::Return);
    }

    #[test]
    fn indirect_call_through_rip_relative_slot() {
        let code = [
            0x48, 0x83, 0xec, 0x28, // 0: sub rsp, 0x28
            0xff, 0x15, 0xf6, 0x0f, 0x00, 0x00, // 4: call [rip+0xff6] -> DATA
            0x48, 0x83, 0xc4, 0x28, // a: add rsp, 0x28
            0xc3, // e: ret
        ];
        let analysis = analyze(&code, &[0]);
        let f = func(&analysis, 0);
        assert_eq!(block_starts(f), [0]);
        assert_eq!(block(f, 0).terminator, Terminator::Return);
        assert_eq!(
            f.calls().copied().collect::<Vec<_>>(),
            [CallSite {
                site: TEXT + 4,
                target: CallTarget::Indirect { slot: Some(DATA) },
            }]
        );
        assert!(f.callees.is_empty());
        assert_eq!(
            analysis.relocations.entries(),
            &[Relocation {
                site: TEXT + 4,
                kind: RelocKind::CallSlot,
                target: DATA,
                region: Region::Data,
            }]
        );

        // The same call through a slot known not to return ends the block.
        let analysis = analyze_with(&code, &[0], &[DATA]);
        let f = func(&analysis, 0);
        let b = block(f, 0);
        assert_eq!(instr_offsets(b), [0x0, 0x4]);
        assert_eq!(b.terminator, Terminator::NoReturnCall);
        assert!(b.successors.is_empty());
    }

    #[test]
    fn direct_call_to_no_return_function_ends_block() {
        let code = [
            0xe8, 0x01, 0x00, 0x00, 0x00, // 0: call 0x6
            0xcc, // 5: garbage after the call
            0xcc, // 6: int3 (the non-returning callee)
        ];
        let analysis = analyze_with(&code, &[0], &[TEXT + 6]);
        let b = block(func(&analysis, 0), 0);
        assert_eq!(b.terminator, Terminator::NoReturnCall);
        assert_eq!(instr_offsets(b), [0]);
        assert_eq!(block(func(&analysis, 6), 6).terminator, Terminator::Trap);
    }

    #[test]
    fn indirect_jumps_are_unresolved_edges() {
        let code = [
            0xff, 0xe0, // 0: jmp rax
            0xff, 0x25, 0x00, 0x10, 0x00,
            0x00, // 2: jmp [rip+0x1000] -> DATA+8 (import thunk)
        ];
        let analysis = analyze(&code, &[0, 2]);
        let by_reg = block(func(&analysis, 0), 0);
        assert_eq!(by_reg.terminator, Terminator::IndirectJump);
        assert_eq!(by_reg.successors, [Edge::IndirectJump { slot: None }]);
        let thunk = block(func(&analysis, 2), 2);
        assert_eq!(thunk.terminator, Terminator::IndirectJump);
        assert_eq!(
            thunk.successors,
            [Edge::IndirectJump {
                slot: Some(DATA + 8)
            }]
        );
        assert_eq!(
            analysis.relocations.entries(),
            &[Relocation {
                site: TEXT + 2,
                kind: RelocKind::JumpSlot,
                target: DATA + 8,
                region: Region::Data,
            }]
        );
    }

    #[test]
    fn trap_instructions_terminate_blocks() {
        let code = [
            0xcc, // 0: int3
            0x0f, 0x0b, // 1: ud2
            0xf4, // 3: hlt
            0xcd, 0x29, // 4: int 0x29 (__fastfail)
            0xcd, 0x2e, // 6: int 0x2e (returns)
            0xc3, // 8: ret
        ];
        let analysis = analyze(&code, &[0, 1, 3, 4, 6]);
        for offset in [0, 1, 3, 4] {
            let b = block(func(&analysis, offset), offset);
            assert_eq!(b.terminator, Terminator::Trap, "offset {offset:#x}");
            assert_eq!(b.instructions.len(), 1);
            assert!(b.successors.is_empty());
        }
        let b = block(func(&analysis, 6), 6);
        assert_eq!(instr_offsets(b), [0x6, 0x8]);
        assert_eq!(b.terminator, Terminator::Return);
    }

    #[test]
    fn invalid_and_truncated_instructions_are_diagnostics() {
        // 0: nop; 1: 0x06 (`push es`, invalid in 64-bit mode)
        let analysis = analyze(&[0x90, 0x06, 0xc3], &[0]);
        let b = block(func(&analysis, 0), 0);
        assert_eq!(instr_offsets(b), [0]);
        assert_eq!(b.end, TEXT + 1);
        assert_eq!(b.terminator, Terminator::Invalid);
        assert!(b.successors.is_empty());
        assert_eq!(
            b.diagnostics,
            [Diagnostic {
                site: TEXT + 1,
                address: TEXT + 1,
                kind: DiagnosticKind::InvalidInstruction,
            }]
        );

        // 0: nop; 1: 48 8b (mov missing its ModRM byte at the end of the section)
        let analysis = analyze(&[0x90, 0x48, 0x8b], &[0]);
        let b = block(func(&analysis, 0), 0);
        assert_eq!(instr_offsets(b), [0]);
        assert_eq!(b.terminator, Terminator::Invalid);
        assert_eq!(b.diagnostics[0].kind, DiagnosticKind::TruncatedInstruction);
    }

    #[test]
    fn running_off_the_end_of_a_section() {
        let analysis = analyze(&[0x90, 0x90], &[0]);
        let f = func(&analysis, 0);
        assert_eq!(block_starts(f), [0, 2]);
        let b = block(f, 0);
        assert_eq!(b.terminator, Terminator::Fallthrough);
        assert_eq!(b.successors, [Edge::Fallthrough(TEXT + 2)]);
        let off_end = block(f, 2);
        assert!(off_end.instructions.is_empty());
        assert_eq!(off_end.terminator, Terminator::Invalid);
        assert_eq!(off_end.diagnostics[0].kind, DiagnosticKind::Unmapped);
    }

    #[test]
    fn decoding_continues_into_adjacent_executable_section() {
        let first = [0x90, 0x90];
        let second = [0xc3];
        let image = CodeImage::new(
            BASE,
            vec![
                CodeSection::new(TEXT, &first, true),
                CodeSection::new(TEXT + 2, &second, true),
            ],
            vec![TEXT],
        );
        let analysis = discover(&image);
        let f = func(&analysis, 0);
        assert_eq!(block_starts(f), [0, 2]);
        assert_eq!(block(f, 2).terminator, Terminator::Return);
        assert_eq!(f.diagnostics().count(), 0);
    }

    #[test]
    fn bad_targets_are_diagnostics_not_functions() {
        let code = [
            0xe8, 0x00, 0x00, 0x00, 0x10, // 0: call TEXT+5+0x10000000 (outside the image)
            0x85, 0xc9, // 5: test ecx, ecx
            0x0f, 0x84, 0xf3, 0x0f, 0x00, 0x00, // 7: je DATA (0xd + 0xff3 = 0x1000)
            0xc3, // d: ret
        ];
        let outside = TEXT + 5 + 0x1000_0000;
        let analysis = analyze(&code, &[0]);
        assert_eq!(
            analysis.functions.len(),
            1,
            "no function at bad call target"
        );
        let f = func(&analysis, 0);
        assert!(f.callees.is_empty());
        let head = block(f, 0);
        assert_eq!(
            head.diagnostics,
            [Diagnostic {
                site: TEXT,
                address: outside,
                kind: DiagnosticKind::Unmapped,
            }]
        );
        assert_eq!(
            head.successors,
            [Edge::Taken(DATA), Edge::Fallthrough(TEXT + 0xd)]
        );
        let data_block = &f.blocks[&DATA];
        assert!(data_block.instructions.is_empty());
        assert_eq!(data_block.terminator, Terminator::Invalid);
        assert_eq!(
            data_block.diagnostics[0].kind,
            DiagnosticKind::NotExecutable
        );
        assert_eq!(
            analysis.relocations.at(TEXT)[0],
            Relocation {
                site: TEXT,
                kind: RelocKind::CallTarget,
                target: outside,
                region: Region::Outside,
            }
        );
    }

    #[test]
    fn entry_point_in_data_section_yields_invalid_function() {
        let image = CodeImage::new(
            BASE,
            vec![CodeSection::new(DATA, &DATA_BYTES, false)],
            vec![DATA, BASE + 0x9000_0000],
        );
        let analysis = discover(&image);
        let data_fn = &analysis.functions[&DATA];
        assert_eq!(data_fn.blocks.len(), 1);
        assert_eq!(
            data_fn.entry_block().map(|b| b.terminator),
            Some(Terminator::Invalid)
        );
        assert_eq!(
            data_fn.diagnostics().next().map(|d| d.kind),
            Some(DiagnosticKind::NotExecutable)
        );
        let outside_fn = &analysis.functions[&(BASE + 0x9000_0000)];
        assert_eq!(
            outside_fn.diagnostics().next().map(|d| d.kind),
            Some(DiagnosticKind::Unmapped)
        );
    }

    #[test]
    fn relocation_map_contents() {
        let code = [
            &[0x48, 0x8d, 0x0d, 0xf9, 0x0f, 0x00, 0x00][..], // 0: lea rcx, [rip+0xff9] -> DATA
            &[0x48, 0x8b, 0x05, 0xfa, 0x0f, 0x00, 0x00],     // 7: mov rax, [rip+0xffa] -> DATA+8
            &[0x48, 0xb8],
            &le64(TEXT + 0x40), // e: mov rax, TEXT+0x40
            &[0x48, 0xba],
            &le64(0x1234), // 18: mov rdx, 0x1234 (not an image address)
            &[0x48, 0xa1],
            &le64(DATA + 0x10),              // 22: mov rax, [DATA+0x10] (moffs64)
            &[0xe8, 0x0f, 0x00, 0x00, 0x00], // 2c: call 0x40
            &[0x48, 0x8d, 0x15, 0x08, 0x00, 0x00, 0x00], // 31: lea rdx, [rip+8] -> 0x40
            &[0xc3],                         // 38: ret
            &[0xcc; 7],                      // 39: padding
            &[0xc3],                         // 40: ret
        ]
        .concat();
        let analysis = analyze(&code, &[0]);
        let got: Vec<_> = analysis
            .relocations
            .entries()
            .iter()
            .map(|r| (r.site - TEXT, r.kind, r.target, r.region))
            .collect();
        assert_eq!(
            got,
            [
                (0x00, RelocKind::RipAddress, DATA, Region::Data),
                (0x07, RelocKind::RipData, DATA + 8, Region::Data),
                (
                    0x0e,
                    RelocKind::AbsoluteImmediate,
                    TEXT + 0x40,
                    Region::Code
                ),
                (
                    0x22,
                    RelocKind::AbsoluteDisplacement,
                    DATA + 0x10,
                    Region::Data
                ),
                (0x2c, RelocKind::CallTarget, TEXT + 0x40, Region::Code),
                (0x31, RelocKind::RipAddress, TEXT + 0x40, Region::Code),
            ]
        );
        let possible: Vec<_> = analysis
            .relocations
            .entries()
            .iter()
            .filter(|r| r.kind.is_possible())
            .map(|r| r.site - TEXT)
            .collect();
        assert_eq!(possible, [0x0e, 0x22]);
        assert!(analysis.relocations.at(TEXT + 0x18).is_empty());
    }

    #[test]
    fn discovers_functions_in_synthetic_pe() {
        use crate::pe::{Pe64, test_util::build_synthetic_pe64};

        let mut file = build_synthetic_pe64(false);
        // .text is at RVA 0x1000 / file offset 0x200. Entry (+0x10): call +0x50; ret.
        file[0x210..0x216].copy_from_slice(&[0xe8, 0x3b, 0x00, 0x00, 0x00, 0xc3]);
        file[0x250] = 0xc3;
        // Export (+0x20): call [rip+0x109a] -> KERNEL32 IAT slot at RVA 0x20c0; ret.
        file[0x220..0x227].copy_from_slice(&[0xff, 0x15, 0x9a, 0x10, 0x00, 0x00, 0xc3]);
        let pe = Pe64::parse(&file).expect("synthetic PE parses");
        let image = CodeImage::from_pe(&pe, &file);
        let analysis = discover(&image);

        let base = 0x1_4000_0000u64;
        assert_eq!(
            analysis.functions.keys().copied().collect::<Vec<_>>(),
            [0x1010, 0x1020, 0x1030, 0x1040, 0x1050].map(|rva| base + rva)
        );
        let entry = &analysis.functions[&(base + 0x1010)];
        assert_eq!(
            entry.callees.iter().copied().collect::<Vec<_>>(),
            [base + 0x1050]
        );
        let export = &analysis.functions[&(base + 0x1020)];
        assert_eq!(
            export.calls().map(|c| c.target).collect::<Vec<_>>(),
            [CallTarget::Indirect {
                slot: Some(base + 0x20c0)
            }]
        );
        assert!(
            analysis
                .functions
                .values()
                .all(|f| f.diagnostics().next().is_none())
        );
        assert_eq!(
            analysis.relocations.at(base + 0x1020),
            &[Relocation {
                site: base + 0x1020,
                kind: RelocKind::CallSlot,
                target: base + 0x20c0,
                region: Region::Data,
            }]
        );
    }
    /// Fixture mixing calls, tail calls, loops and block splits (no overlapping code).
    fn mixed_fixture() -> Vec<u8> {
        [
            &[0xe8, 0x0b, 0x00, 0x00, 0x00][..],   // 0: call 0x10
            &[0x85, 0xc9],                         // 5: test ecx, ecx
            &[0x74, 0x07],                         // 7: je 0x10
            &[0xe9, 0x02, 0x00, 0x00, 0x00],       // 9: jmp 0x10
            &[0xcc, 0xcc],                         // e: padding
            &[0xb8, 0x01, 0x00, 0x00, 0x00],       // 10: mov eax, 1
            &[0x01, 0xc8],                         // 15: add eax, ecx   <- loop head
            &[0xff, 0x15, 0xe3, 0x0f, 0x00, 0x00], // 17: call [rip+0xfe3] -> DATA
            &[0x85, 0xc0],                         // 1d: test eax, eax
            &[0x75, 0xf4],                         // 1f: jne 0x15
            &[0xc3],                               // 21: ret
        ]
        .concat()
    }

    #[test]
    fn discovery_is_deterministic_and_order_independent() {
        let code = mixed_fixture();
        let forward = analyze(&code, &[0, 0x10]);
        let again = analyze(&code, &[0, 0x10]);
        let reversed = analyze(&code, &[0x10, 0]);
        let only_first = analyze(&code, &[0]);
        for other in [&again, &reversed, &only_first] {
            assert_eq!(&forward, other);
            for (f, g) in forward.functions.values().zip(other.functions.values()) {
                for (a, b) in f.blocks.values().zip(g.blocks.values()) {
                    assert_eq!(instr_offsets(a), instr_offsets(b));
                }
            }
        }

        let f = func(&forward, 0x10);
        assert_eq!(block_starts(f), [0x10, 0x15, 0x21]);
        assert_eq!(f.calls().count(), 1);
        assert_eq!(forward.relocations.len(), 5);
    }

    #[test]
    fn long_call_chain_does_not_recurse() {
        // 50k functions, each `call next; ret`, the last one just `ret`.
        const COUNT: usize = 50_000;
        let mut code = Vec::with_capacity(COUNT * 6 + 1);
        for _ in 0..COUNT {
            code.extend_from_slice(&[0xe8, 0x01, 0x00, 0x00, 0x00, 0xc3]);
        }
        code.push(0xc3);
        let image = CodeImage::new(TEXT, vec![CodeSection::new(TEXT, &code, true)], vec![TEXT]);
        let analysis = discover(&image);
        assert_eq!(analysis.functions.len(), COUNT + 1);
        assert_eq!(analysis.relocations.len(), COUNT);
        let first = func(&analysis, 0);
        assert_eq!(block_starts(first), [0]);
        assert_eq!(
            first.callees.iter().copied().collect::<Vec<_>>(),
            [TEXT + 6]
        );
    }
}
