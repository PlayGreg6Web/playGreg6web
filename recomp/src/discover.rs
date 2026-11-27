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
//! Resume targets (where `RtlRestoreContext` / `longjmp` continue inside a function, see
//! [`Function::resume_targets`]) are block leaders from three sources: the EH metadata seeds in
//! [`CodeImage::eh`] (catch continuations and `__except` bodies, each with its owning function;
//! the owner's membership walk starts from them as well, since they are often reachable only
//! through the unwinder), the instruction after a call through a `setjmp`-family IAT slot (or a
//! direct call to an import thunk of one), and, when `.pdata` function extents are known,
//! every address-taken interior code address: a RIP-relative `lea` of a non-entry address
//! inside a `.pdata` function (e.g. the continuation an FH3 or FH4 catch funclet returns in
//! `rax`), owned by that function. Interior addresses that fall inside a decoded instruction
//! are ignored. The `setjmp`-thunk and `lea` sources need the complete block map, so they run
//! after each jump-table fixpoint and the worklist runs again until nothing new is found.
//!
//! Both phases are iterative (no recursion proportional to code size). Block splitting is
//! `O(log n + block length)` and the per-instruction hot path does no heap formatting. Each
//! jump-table round costs one `O(blocks)` predecessor pass.
//!
//! Parallelism: block decoding stays one serial worklist (leaders and splits make it stateful),
//! but the expensive read-only work runs in waves over a frozen snapshot on scoped threads:
//! matching each jump-table round's pending jumps, revising and verifying resolved tables,
//! classifying unresolved jumps, gathering relocations and the per-function membership walks.
//! Every wave maps an ordered input to an ordered output ([`par_map`]) and the results are merged
//! serially in that order, so the analysis is identical for every thread count.
//! [`discover`] uses [`default_threads`]; [`discover_with_threads`] takes an explicit count.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use iced_x86::{
    Decoder, DecoderError, DecoderOptions, FlowControl, Instruction, Mnemonic, OpKind, Register,
};

use crate::cfg::{
    BasicBlock, CallSite, CallTarget, Diagnostic, DiagnosticKind, Edge, Function, JumpTable,
    Terminator, UnresolvedJump,
};
use crate::code_image::{CodeImage, Region};
use crate::jump_table::{self, Step};
use crate::pointer_slot;
use crate::reach;
use crate::reloc::{RelocKind, Relocation, RelocationMap, instruction_relocations};

mod pred_index;
use pred_index::PredIndex;

/// `int 0x29` is the Windows `__fastfail` trap and never returns.
const FASTFAIL_VECTOR: u8 = 0x29;

/// Preceding blocks examined when looking for the block that contains an address (16 covers the
/// full 15-byte maximum x86-64 instruction width plus the containing block).
const CONTAINING_BLOCK_SCAN: usize = 16;

/// Longest chain of blocks (the jump's block plus sole predecessors) sliced for a jump table.
const MAX_TRACE_BLOCKS: usize = 4;

/// Times one resolved jump table may be replaced by a changed resolution.
const MAX_TABLE_REVISIONS: u32 = 8;

/// Maximum consecutive 8-byte relocated pointer entries walked per `.rdata` vtable/callback table.
const MAX_VTABLE_ENTRIES: usize = 1024;

/// Incoming intra-function edges of a block: how many, one of them, and every source block
/// (`others` holds the sources beyond `from`).
#[derive(Clone)]

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
    /// Why each indirect `jmp` without an import slot that is left to run-time dispatch did not
    /// resolve, keyed by its VA.
    pub unresolved_jumps: BTreeMap<u64, UnresolvedJump>,
    /// Unresolved tail-call `jmp`s whose pointer is loaded from a statically initialised,
    /// relocated slot (`mov reg, [rip+slot]` ... `jmp reg`), keyed by the jump's VA. The value is
    /// `(slot, initial code target)`; the target was seeded as a function entry. The jump itself
    /// still leaves through the run-time dispatcher, since the slot can be rewritten.
    pub pointer_slots: BTreeMap<u64, (u64, u64)>,
}

impl Analysis {
    /// Function whose entry is `entry`.
    pub fn function(&self, entry: u64) -> Option<&Function> {
        self.functions.get(&entry)
    }
}

/// Environment variable overriding the discovery thread count (a positive integer).
pub const THREADS_ENV: &str = "PLAYGTA6_RECOMP_THREADS";

/// Threads [`discover`] uses: [`THREADS_ENV`] when set to a positive integer, else the available
/// parallelism, else 1.
pub fn default_threads() -> usize {
    std::env::var(THREADS_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .or_else(|| std::thread::available_parallelism().ok().map(usize::from))
        .unwrap_or(1)
}

/// Discover functions, per-function CFGs, jump tables and the relocation map starting from
/// `image.entry_points`. Never panics on malformed code: problems become [`Diagnostic`]s.
pub fn discover(image: &CodeImage<'_>) -> Analysis {
    discover_with_threads(image, default_threads())
}

/// [`discover`] with an explicit thread count (`0` and `1` run serially). The result does not
/// depend on `threads`.
pub fn discover_with_threads(image: &CodeImage<'_>, threads: usize) -> Analysis {
    let mut state = Discovery::new(image, threads);
    for &entry in &image.entry_points {
        state.add_function(entry);
    }
    for &(owner, target) in &image.eh.resume_targets {
        state.seed_resume_target(owner, target);
    }
    state.run();
    loop {
        while state.resolve_jump_tables()
            || state.revise_jump_tables()
            || state.requeue_changed_jumps()
        {
            state.run();
        }
        let late = state.add_late_resume_targets();
        let seeded = state.seed_pointer_slots();
        let vtables = state.seed_vtable_callbacks();
        if !late && !seeded && !vtables {
            break;
        }
        state.run();
    }
    state.verify_jump_tables();
    let unresolved_jumps = state.classify_unresolved_jumps();
    let pointer_slots = std::mem::take(&mut state.pointer_slots);

    let blocks: Vec<&BasicBlock> = state.blocks.values().collect();
    let mut relocations: Vec<Relocation> = par_map(threads, &blocks, |block| {
        let mut out = Vec::new();
        for instr in &block.instructions {
            instruction_relocations(image, instr, &mut out);
        }
        out
    })
    .into_iter()
    .flatten()
    .collect();
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
        unresolved_jumps,
        pointer_slots,
    }
}

/// `f` over `items` on up to `threads` scoped threads, results in input order. Workers claim small
/// index chunks from a shared counter (load balancing for uneven items); the output order, and so
/// everything merged from it, is independent of scheduling and thread count.
fn par_map<T: Sync, R: Send>(threads: usize, items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = threads.min(items.len());
    if threads <= 1 {
        return items.iter().map(f).collect();
    }
    let chunk = (items.len() / (threads * 8)).max(1);
    let next = AtomicUsize::new(0);
    let parts: Mutex<Vec<(usize, Vec<R>)>> = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let mut local = Vec::new();
                loop {
                    let start = next.fetch_add(chunk, Ordering::Relaxed);
                    if start >= items.len() {
                        break;
                    }
                    let end = (start + chunk).min(items.len());
                    local.push((start, items[start..end].iter().map(&f).collect::<Vec<R>>()));
                }
                parts
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .extend(local);
            });
        }
    });
    let mut parts = parts
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    parts.sort_unstable_by_key(|(start, _)| *start);
    parts.into_iter().flat_map(|(_, part)| part).collect()
}

/// Relocations of a resolved table: the table and index-table reads plus one per entry.
/// An inferred table (no proven index bound) is read as far as its entries stay inside the
/// function, which runs over into the next table when tables are packed back to back: the
/// entries past the neighbour's base are that table's, and decode to addresses in the middle of
/// instructions. Cuts the extent at the lowest known table base inside it, keeping at least two
/// entries. Tables with a proven bound and two-level tables are left alone.
fn clip_inferred(
    blocks: &BTreeMap<u64, BasicBlock>,
    mut table: JumpTable,
    bases: &BTreeSet<u64>,
) -> JumpTable {
    table.alternates = std::mem::take(&mut table.alternates)
        .into_iter()
        .map(|t| clip_inferred(blocks, t, bases))
        .collect();
    if !table.inferred || table.index.is_some() {
        return table;
    }
    let stride = blocks
        .range(..=table.load_site)
        .next_back()
        .and_then(|(_, b)| b.instructions.iter().find(|i| i.ip() == table.load_site))
        .filter(|i| i.memory_index() != Register::None)
        .map(|i| u64::from(i.memory_index_scale()));
    let Some(stride @ (4 | 8)) = stride else {
        return table;
    };
    let Some(end) = u64::try_from(table.targets.len())
        .ok()
        .and_then(|n| n.checked_mul(stride))
        .and_then(|n| table.table.checked_add(n))
    else {
        return table;
    };
    let Some(&next) = bases.range(table.table.saturating_add(1)..end).next() else {
        return table;
    };
    let keep = (next - table.table) / stride;
    if keep >= 2
        && let Ok(keep) = usize::try_from(keep)
    {
        table.targets.truncate(keep);
    }
    table
}

fn jump_table_relocations(image: &CodeImage<'_>, table: &JumpTable, out: &mut Vec<Relocation>) {
    for alternate in &table.alternates {
        jump_table_relocations(image, alternate, out);
    }
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
    for (index, &target) in table.targets.iter().enumerate() {
        let Some(offset) = u64::try_from(index).ok().and_then(|i| i.checked_mul(4)) else {
            break;
        };
        let Some(slot) = table.table.checked_add(offset) else {
            break;
        };
        push(slot, RelocKind::JumpTableEntry, target);
    }
}

/// `(first block, trace length, incoming edges of the first block)` of a jump's slice.
type JumpShape = (u64, usize, u32);

struct Discovery<'i, 'a> {
    image: &'i CodeImage<'a>,
    /// Threads for the read-only waves ([`par_map`]).
    threads: usize,
    no_return: BTreeSet<u64>,
    /// Every block decoded so far, keyed by start VA. Change it only through
    /// [`Self::insert_block`], [`Self::set_successors`] and [`Self::split`], which keep
    /// [`Self::preds`] in step.
    blocks: BTreeMap<u64, BasicBlock>,
    /// Incoming intra-function edges of every block in [`Self::blocks`], maintained on every edge
    /// change (always equal to [`PredIndex::from_blocks`]).
    preds: PredIndex,
    /// Every known block start (decoded or still queued).
    leaders: BTreeSet<u64>,
    /// Pending block starts popped in ascending VA order for input-order independence.
    worklist: BTreeSet<u64>,
    entries: BTreeSet<u64>,
    /// `jmp reg` sites decoded since the last jump-table round.
    pending_jumps: BTreeSet<u64>,
    /// Jumps that did not resolve: the shape of their slice when last tried
    /// ([`Self::jump_shape`]). A jump is tried again when it changes, e.g. a merge in front of it
    /// gained the incoming edge of a table resolved later ([`Self::requeue_changed_jumps`]).
    failed_jumps: BTreeMap<u64, JumpShape>,
    /// Retries per jump (at most [`MAX_TABLE_REVISIONS`]).
    jump_retries: BTreeMap<u64, u32>,
    jump_tables: BTreeMap<u64, JumpTable>,
    /// How often each resolved table was revised (see `revise_jump_tables`).
    table_revisions: BTreeMap<u64, u32>,
    /// `setjmp`-family IAT slots ([`CodeImage::eh`]).
    setjmp_slots: BTreeSet<u64>,
    /// Resume targets with their owning function: `(owner entry, target)`.
    resume_roots: BTreeSet<(u64, u64)>,
    /// Distinct resume target VAs across [`Self::resume_roots`] for `O(log n)` entry checks.
    resume_targets: BTreeSet<u64>,
    /// EH resume targets whose owner has not been discovered as a function entry yet.
    deferred_resume_targets: BTreeSet<(u64, u64)>,
    /// Return sites of `setjmp`-family calls (owned by every function containing the block).
    return_sites: BTreeSet<u64>,
    /// Pointer-slot tail calls seen so far: jump VA -> `(slot, seeded target)`.
    pointer_slots: BTreeMap<u64, (u64, u64)>,
    /// Unresolved jumps/calls whose pointer slot was already checked by `seed_pointer_slots`.
    slot_checked: BTreeSet<u64>,
    /// Referenced `Region::Data` VAs already scanned for vtables / function pointer tables.
    scanned_data_refs: BTreeSet<u64>,
    /// `(block.start, block.end)` snapshots already scanned for RIP-relative vtable/callback refs.
    vtable_scanned_blocks: BTreeSet<(u64, u64)>,
}

impl<'i, 'a> Discovery<'i, 'a> {
    fn new(image: &'i CodeImage<'a>, threads: usize) -> Self {
        Self {
            image,
            threads,
            no_return: image.no_return.iter().copied().collect(),
            blocks: BTreeMap::new(),
            preds: PredIndex::default(),
            leaders: BTreeSet::new(),
            worklist: BTreeSet::new(),
            entries: BTreeSet::new(),
            pending_jumps: BTreeSet::new(),
            failed_jumps: BTreeMap::new(),
            jump_retries: BTreeMap::new(),
            jump_tables: BTreeMap::new(),
            table_revisions: BTreeMap::new(),
            setjmp_slots: image.eh.setjmp_slots.iter().copied().collect(),
            resume_roots: BTreeSet::new(),
            resume_targets: BTreeSet::new(),
            deferred_resume_targets: BTreeSet::new(),
            return_sites: BTreeSet::new(),
            pointer_slots: BTreeMap::new(),
            slot_checked: BTreeSet::new(),
            scanned_data_refs: BTreeSet::new(),
            vtable_scanned_blocks: BTreeSet::new(),
        }
    }

    /// Seeds an EH resume target `(owner, target)`. Non-decodable or out-of-range VAs are ignored;
    /// targets whose owner is not yet a known function entry are deferred until `owner` is
    /// discovered so malformed seeds never decode orphan blocks.
    fn seed_resume_target(&mut self, owner: u64, target: u64) {
        if self.image.code_bytes(owner).is_err() || self.image.code_bytes(target).is_err() {
            return;
        }
        if self.entries.contains(&owner) {
            self.add_resume_target(owner, target);
        } else {
            self.deferred_resume_targets.insert((owner, target));
        }
    }

    /// Records `target` as a resume target of `owner` (ignored when it is another function's
    /// entry or not backed by executable bytes). Returns whether it was new.
    fn add_resume_target(&mut self, owner: u64, target: u64) -> bool {
        if (target != owner && self.entries.contains(&target))
            || self.image.code_bytes(owner).is_err()
            || self.image.code_bytes(target).is_err()
        {
            return false;
        }
        self.add_leader(target);
        self.resume_targets.insert(target);
        self.resume_roots.insert((owner, target))
    }

    /// Records the return site of a `setjmp`-family call when it maps to executable code.
    /// Returns whether it was new.
    fn add_return_site(&mut self, va: u64) -> bool {
        if self.image.code_bytes(va).is_err() {
            return false;
        }
        self.add_leader(va);
        self.return_sites.insert(va)
    }

    /// Whether the block at `va` is an import thunk `jmp [rip+slot]` of a `setjmp`-family slot.
    fn is_setjmp_thunk(&self, va: u64) -> bool {
        self.blocks.get(&va).is_some_and(|block| {
            block.instructions.len() == 1
                && matches!(block.successors[..], [Edge::IndirectJump { slot: Some(slot) }]
                    if self.setjmp_slots.contains(&slot))
        })
    }

    /// Resume targets that need the complete block map: deferred EH resume targets whose owner
    /// was discovered via calls, return sites of direct calls to `setjmp` thunks, and
    /// address-taken interior code (see the module docs). Returns whether anything new was found.
    fn add_late_resume_targets(&mut self) -> bool {
        let mut progress = false;
        if !self.deferred_resume_targets.is_empty() {
            let ready: Vec<(u64, u64)> = self
                .deferred_resume_targets
                .iter()
                .copied()
                .filter(|(owner, _)| self.entries.contains(owner))
                .collect();
            for pair in ready {
                self.deferred_resume_targets.remove(&pair);
                progress |= self.add_resume_target(pair.0, pair.1);
            }
        }
        let table = &self.image.function_table;
        if table.is_empty() && self.setjmp_slots.is_empty() {
            return progress;
        }
        let mut sites = Vec::new();
        let mut interior = Vec::new();
        for block in self.blocks.values() {
            for call in &block.calls {
                if let CallTarget::Direct(target) = call.target
                    && !self.no_return.contains(&target)
                    && self.is_setjmp_thunk(target)
                    && let Ok(index) = block
                        .instructions
                        .binary_search_by_key(&call.site, Instruction::ip)
                {
                    let instr = &block.instructions[index];
                    if let Some(next) = instr.ip().checked_add(instr.len() as u64) {
                        sites.push(next);
                    }
                }
            }
            if table.is_empty() {
                continue;
            }
            for instr in &block.instructions {
                if instr.mnemonic() != Mnemonic::Lea || !instr.is_ip_rel_memory_operand() {
                    continue;
                }
                let target = instr.ip_rel_memory_address();
                if self.entries.contains(&target)
                    || self.image.code_bytes(target).is_err()
                    || matches!(self.locate(target), Decoded::InsideInstruction)
                {
                    continue;
                }
                if let Some(owner) = table.function_at(target)
                    && self.entries.contains(&owner)
                {
                    interior.push((owner, target));
                }
            }
        }
        for site in sites {
            progress |= self.add_return_site(site);
        }
        for (owner, target) in interior {
            progress |= self.add_resume_target(owner, target);
        }
        progress
    }

    fn add_leader(&mut self, va: u64) {
        if self.leaders.insert(va) {
            self.worklist.insert(va);
        }
    }

    fn add_function(&mut self, va: u64) {
        self.entries.insert(va);
        self.add_leader(va);
    }

    fn run(&mut self) {
        while let Some(va) = self.worklist.pop_first() {
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
        if index == 0 {
            return;
        }
        let Some(at) = self
            .blocks
            .get(&start)
            .and_then(|head| head.instructions.get(index))
            .map(Instruction::ip)
        else {
            return;
        };
        if self.blocks.contains_key(&at) {
            return;
        }
        let Some(head) = self.blocks.get_mut(&start) else {
            return;
        };
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
        self.preds.remove(start, &tail.successors);
        self.preds.add(start, &head.successors);
        self.insert_block(at, tail);
    }

    /// Store a decoded block (replacing any block at `start`) and index its edges.
    fn insert_block(&mut self, start: u64, block: BasicBlock) {
        if let Some(old) = self.blocks.remove(&start) {
            self.preds.remove(start, &old.successors);
        }
        self.preds.add(start, &block.successors);
        self.blocks.insert(start, block);
    }

    /// Replace the successors of the block at `start` (no-op when there is none).
    fn set_successors(&mut self, start: u64, successors: Vec<Edge>) {
        let Some(block) = self.blocks.get_mut(&start) else {
            return;
        };
        self.preds.remove(start, &block.successors);
        self.preds.add(start, &successors);
        block.successors = successors;
    }

    /// Unit tests check the incremental predecessor index against a full rebuild before every
    /// phase that reads it.
    fn check_preds(&self) {
        #[cfg(test)]
        assert_eq!(
            self.preds,
            PredIndex::from_blocks(&self.blocks),
            "incremental predecessor index out of step with the block map"
        );
    }

    /// First known leader at or after `va`, if any.
    fn next_leader_from(&self, va: u64) -> Option<u64> {
        self.leaders.range(va..).next().copied()
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
                self.insert_block(start, block);
                return;
            }
        };
        // Clamp to the end of the 64-bit address space so instruction IPs cannot wrap past
        // `u64::MAX`.
        let max_len = usize::try_from(u64::MAX - start).unwrap_or(usize::MAX);
        let bytes = &bytes[..bytes.len().min(max_len)];
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
            .and_then(|va| self.next_leader_from(va));
        // Set once this stream is out of step with already decoded code; it then stops as soon as
        // it lands on an existing instruction boundary instead of duplicating that block's tail.
        let mut misaligned = overlaps;
        loop {
            let ip = decoder.ip();
            if stop_at == Some(ip) {
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
            let Some(next) = ip.checked_add(instr.len() as u64) else {
                block.terminator = Terminator::Invalid;
                block.diagnostics.push(Diagnostic {
                    site: ip,
                    address: ip,
                    kind: DiagnosticKind::TruncatedInstruction,
                });
                break;
            };
            if let Some(leader) = stop_at
                && leader > ip
                && leader < next
            {
                // This instruction straddles a known leader: keep decoding this stream.
                block.diagnostics.push(Diagnostic {
                    site: ip,
                    address: leader,
                    kind: DiagnosticKind::OverlappingInstruction,
                });
                stop_at = self.next_leader_from(next);
                misaligned = true;
            }
            block.instructions.push(instr);
            block.end = next;

            if let Some(terminator) = self.classify(&instr, next, &mut block, &mut stop_at) {
                block.terminator = terminator;
                break;
            }
        }
        self.insert_block(start, block);
    }

    /// Record control flow of `instr` (already appended to `block`). Returns the terminator if
    /// the instruction ends the block.
    fn classify(
        &mut self,
        instr: &Instruction,
        next: u64,
        block: &mut BasicBlock,
        stop_at: &mut Option<u64>,
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
                        if target >= next && stop_at.is_none_or(|s| target < s) {
                            *stop_at = Some(target);
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
                let is_no_return = slot.is_some_and(|s| self.no_return.contains(&s));
                if !is_no_return
                    && slot.is_some_and(|s| self.setjmp_slots.contains(&s))
                    && self.add_return_site(next)
                {
                    *stop_at = Some(stop_at.map_or(next, |s| s.min(next)));
                }
                is_no_return.then_some(Terminator::NoReturnCall)
            }
            FlowControl::UnconditionalBranch if instr.op_kind(0) == OpKind::NearBranch64 => {
                let target = instr.near_branch64();
                block.successors.push(Edge::Jump(target));
                self.seed_cross_function_tail_call(ip, target);
                self.add_leader(target);
                Some(Terminator::Jump)
            }
            FlowControl::UnconditionalBranch | FlowControl::IndirectBranch => {
                block.successors.push(Edge::IndirectJump {
                    slot: rip_slot(instr),
                });
                // `jmp reg`, and `jmp qword ptr [base + index*8 (+ disp)]`: a pointer table read
                // by the jump itself (a 64-bit absolute switch table, e.g. a threaded
                // interpreter's opcode dispatch). A `[base + disp]` slot without an index is a
                // vtable/callback tail call and stays on run-time dispatch unexamined.
                let indexed_table_read = instr.op0_kind() == OpKind::Memory
                    && !instr.is_ip_rel_memory_operand()
                    && instr.memory_index() != Register::None;
                if instr.op0_kind() == OpKind::Register || indexed_table_read {
                    self.pending_jumps.insert(ip);
                }
                Some(Terminator::IndirectJump)
            }
            FlowControl::ConditionalBranch => self.branch(instr, next, block),
            FlowControl::XbeginXabortXend if instr.mnemonic() == Mnemonic::Xbegin => {
                self.branch(instr, next, block)
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

    fn branch(
        &mut self,
        instr: &Instruction,
        next: u64,
        block: &mut BasicBlock,
    ) -> Option<Terminator> {
        let taken = instr.near_branch_target();
        block.successors.push(Edge::Taken(taken));
        block.successors.push(Edge::Fallthrough(next));
        self.seed_cross_function_tail_call(instr.ip(), taken);
        self.add_leader(taken);
        self.add_leader(next);
        Some(Terminator::Branch)
    }

    /// If a direct branch at `ip` inside a `.pdata`-owned function jumps straight to the entry of
    /// a different `.pdata` function, seed that target as a function entry so it is discovered
    /// separately and classified as `Edge::TailCall`.
    fn seed_cross_function_tail_call(&mut self, ip: u64, target: u64) {
        let table = &self.image.function_table;
        if table.is_empty() {
            return;
        }
        if let Some(src_fn) = table.function_at(ip)
            && let Some(dst_fn) = table.function_at(target)
            && dst_fn != src_fn
            && target == dst_fn
            && self.image.code_bytes(dst_fn).is_ok()
        {
            self.add_function(dst_fn);
        }
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

    /// The jump at `site` together with its chain of sole predecessors, and the first block of
    /// that chain.
    fn jump_trace(&self, site: u64, preds: &PredIndex) -> Option<(Vec<Step<'_>>, u64)> {
        let start = self.jump_block(site)?;
        let mut chain = vec![(self.blocks.get(&start)?, None)];
        let mut current = start;
        while chain.len() < MAX_TRACE_BLOCKS && !self.entries.contains(&current) {
            let Some(pred) = preds.get(&current).filter(|p| p.count == 1) else {
                break;
            };
            if chain.iter().any(|(b, _)| b.start == pred.from) {
                break;
            }
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
        Some((trace, current))
    }

    /// Facts about the registers `trace` reads before writing them that hold on every
    /// intra-function path into `first_block` ([`reach::reaching_values`]), as synthetic
    /// instructions to put in front of the trace: `(constants, all)`, where `constants` holds only
    /// the single in-image values (`lea reg, [rip+X]`, the table bases) and `all` adds the interval
    /// bounds (`cmp reg, hi` + not-taken `ja`). `None` when nothing reaches or the subgraph is too
    /// large.
    #[allow(clippy::type_complexity)]
    fn reaching_prefix(
        &self,
        trace: &[Step<'_>],
        first_block: u64,
        preds: &PredIndex,
    ) -> Option<(Vec<(Instruction, bool)>, Vec<(Instruction, bool)>)> {
        let live = live_in_registers(trace);
        if live.is_empty() {
            return None;
        }
        let preds_of = |b: u64| -> Vec<u64> {
            preds.get(&b).map_or_else(Vec::new, |p| p.sources())
        };
        let is_entry = |b: u64| {
            self.entries.contains(&b)
                || self.return_sites.contains(&b)
                || self.resume_targets.contains(&b)
        };
        let facts =
            reach::reaching_facts(&self.blocks, &preds_of, &is_entry, first_block, &live)?;
        let contains = |va: u64| self.image.contains(va);
        let constants: Vec<(Instruction, bool)> = facts
            .values
            .iter()
            .filter(|(_, known)| {
                matches!(known, reach::Known::Range { lo, hi } if lo == hi && contains(*lo))
            })
            .flat_map(|&(reg, known)| reach::synthesize(reg, known, &contains))
            .collect();
        // Register copies (`mov r9d, eax` / `mov r9, rax` before a loop or merge) go in front of
        // the bounds, so a guard on the source reaches an index held in the copy.
        let mut all: Vec<(Instruction, bool)> = facts
            .copies
            .iter()
            .filter_map(|&(dst, src, is_64)| {
                if is_64 {
                    reach::synthesize_copy64(dst, src)
                } else {
                    reach::synthesize_copy(dst, src)
                }
            })
            .collect();
        for (&(reg, known), &(_, u32_known)) in facts.values.iter().zip(&facts.u32_values) {
            let synth = reach::synthesize(reg, known, &contains);
            if !synth.is_empty() {
                all.extend(synth);
            } else {
                all.extend(reach::synthesize_u32(reg, u32_known));
            }
        }
        Some((constants, all))
    }

    /// `[entry, next entry)` of the discovered function entries around `site` (both must exist):
    /// the extent of a leaf function, which has no `.pdata` range ([`jump_table::resolve_in`]).
    fn leaf_extent(&self, site: u64) -> Option<(u64, u64)> {
        let entry = *self.entries.range(..=site).next_back()?;
        let after = site.checked_add(1)?;
        let next = *self.entries.range(after..).next()?;
        Some((entry, next))
    }

    /// Base address of every resolved jump table.
    fn table_bases(&self) -> BTreeSet<u64> {
        self.jump_tables
            .values()
            .flat_map(JumpTable::tables)
            .map(|t| t.table)
            .collect()
    }

    /// [`Self::match_jump_table_raw`] with an inferred extent cut at the start of the next known
    /// table ([`clip_inferred`]).
    fn match_jump_table(
        &self,
        site: u64,
        preds: &PredIndex,
        bases: &BTreeSet<u64>,
    ) -> Option<JumpTable> {
        let table = self.match_jump_table_raw(site, preds)?;
        Some(clip_inferred(&self.blocks, table, bases))
    }

    /// Slice the jump at `site` together with its chain of sole predecessors.
    fn match_jump_table_raw(&self, site: u64, preds: &PredIndex) -> Option<JumpTable> {
        let (trace, first_block) = self.jump_trace(site, preds)?;
        let leaf = self.leaf_extent(site);
        let direct = jump_table::resolve_in(self.image, &trace, leaf);
        if direct.as_ref().is_some_and(|t| !t.inferred) {
            return direct;
        }
        // The chain stops at a merge (e.g. a loop header): a table base or an index bound
        // established before the merge is invisible to the slice. What holds on every path into
        // the merge is stated in front of the trace and the slice runs again: first only the
        // constants, then also the bounds, so tables resolved by the base alone keep their
        // resolution. An extent inferred without those facts (no bound, or only the width of a
        // zero-extended byte index) yields to a bound they prove; the first table found wins
        // when none is proven.
        let Some((constants, all)) = self.reaching_prefix(&trace, first_block, preds) else {
            return direct;
        };
        let mut first = direct;
        if !constants.is_empty()
            && let Some(table) =
                jump_table::resolve_in(self.image, &prefixed(&constants, &trace), leaf)
        {
            if !table.inferred {
                return Some(table);
            }
            first.get_or_insert(table);
        }
        if all.len() > constants.len()
            && let Some(table) = jump_table::resolve_in(self.image, &prefixed(&all, &trace), leaf)
        {
            if !table.inferred {
                return Some(table);
            }
            first.get_or_insert(table);
        }
        self.match_constant_sets(&trace, first_block, preds, leaf, &all)
            .or(first)
    }

    /// A dispatch whose table base (or another register the slice reads) holds one of a few
    /// constants on the paths into the merge ([`reach::reaching_constant_sets`]): the slice runs
    /// once per constant, with that constant stated last in front of the trace (after the facts
    /// that hold on every path). The jump resolves only when every constant yields a table; equal
    /// tables collapse, distinct ones become [`JumpTable::alternates`] of the one with the lowest
    /// base. Since every path brings one of the constants, the union of the tables is every case
    /// the jump can take under the proofs of the single tables.
    fn match_constant_sets(
        &self,
        trace: &[Step<'_>],
        first_block: u64,
        preds: &PredIndex,
        leaf: Option<(u64, u64)>,
        all: &[(Instruction, bool)],
    ) -> Option<JumpTable> {
        let live = live_in_registers(trace);
        if live.is_empty() {
            return None;
        }
        let preds_of = |b: u64| -> Vec<u64> {
            preds.get(&b).map_or_else(Vec::new, |p| p.sources())
        };
        let is_entry = |b: u64| {
            self.entries.contains(&b)
                || self.return_sites.contains(&b)
                || self.resume_targets.contains(&b)
        };
        let sets =
            reach::reaching_constant_sets(&self.blocks, &preds_of, &is_entry, first_block, &live)?;
        let contains = |va: u64| self.image.contains(va);
        'regs: for (reg, values) in sets {
            let mut tables: Vec<JumpTable> = Vec::with_capacity(values.len());
            for value in values {
                let mut prefix = all.to_vec();
                let stated = reach::synthesize(
                    reg,
                    reach::Known::Range {
                        lo: value,
                        hi: value,
                    },
                    &contains,
                );
                if stated.is_empty() {
                    continue 'regs;
                }
                prefix.extend(stated);
                let Some(table) =
                    jump_table::resolve_in(self.image, &prefixed(&prefix, trace), leaf)
                else {
                    continue 'regs;
                };
                tables.push(table);
            }
            tables.sort();
            tables.dedup();
            let mut tables = tables.into_iter();
            let mut primary = tables.next()?;
            primary.alternates = tables.collect();
            return Some(primary);
        }
        None
    }

    /// Why each indirect jump without an import slot that is left to run-time dispatch did not
    /// resolve ([`jump_table::classify`] on the same slice and reaching facts the match used).
    fn classify_unresolved_jumps(&self) -> BTreeMap<u64, UnresolvedJump> {
        self.check_preds();
        let preds = &self.preds;
        let jumps: Vec<u64> = self
            .blocks
            .values()
            .filter(|block| {
                block.terminator == Terminator::IndirectJump
                    && block
                        .successors
                        .contains(&Edge::IndirectJump { slot: None })
            })
            .filter_map(|block| block.instructions.last().map(Instruction::ip))
            .collect();
        par_map(self.threads, &jumps, |&site| {
            let (trace, first_block) = self.jump_trace(site, preds)?;
            let kind = match self.reaching_prefix(&trace, first_block, preds) {
                Some((_, all)) if !all.is_empty() => {
                    jump_table::classify(self.image, &prefixed(&all, &trace))
                }
                _ => jump_table::classify(self.image, &trace),
            };
            Some((site, kind))
        })
        .into_iter()
        .flatten()
        .collect()
    }

    /// Validates a relocated 64-bit pointer slot in `Region::Data` and returns its code target if
    /// it lands on a valid, non-padding, non-interior code instruction.
    fn relocated_code_target(&self, slot: u64) -> Option<u64> {
        if !self.image.data_relocs.contains(&slot) || self.image.region_of(slot) != Region::Data {
            return None;
        }
        let bytes = self.image.bytes_at(slot, 8)?;
        let target = u64::from_le_bytes(bytes.try_into().ok()?);
        if self.image.region_of(target) != Region::Code {
            return None;
        }
        let code = self.image.code_bytes(target).ok()?;
        if !self.image.function_table.is_empty()
            && self
                .image
                .function_table
                .function_at(target)
                .is_some_and(|owner| owner != target)
        {
            return None;
        }
        let mut decoder = Decoder::with_ip(64, code, target, DecoderOptions::NONE);
        if !decoder.can_decode() {
            return None;
        }
        let instr = decoder.decode();
        if instr.is_invalid() || matches!(instr.mnemonic(), Mnemonic::Int3 | Mnemonic::Ud2) {
            return None;
        }
        Some(target)
    }

    /// Tail calls and indirect calls through a statically initialised, relocated pointer slot:
    /// when the slot holds a relocated code address, that address is a real function entry, so
    /// seed it ahead of time. The jump/call stays on run-time dispatch. Returns whether any new
    /// entry was added.
    fn seed_pointer_slots(&mut self) -> bool {
        self.check_preds();
        let preds = &self.preds;
        let jumps: Vec<u64> = self
            .blocks
            .values()
            .filter(|block| {
                block.terminator == Terminator::IndirectJump
                    && block
                        .successors
                        .contains(&Edge::IndirectJump { slot: None })
            })
            .filter_map(|block| block.instructions.last().map(Instruction::ip))
            .filter(|site| !self.slot_checked.contains(site))
            .collect();
        let found: Vec<(u64, Option<(u64, u64)>)> = par_map(self.threads, &jumps, |&site| {
            let hit = self.jump_trace(site, preds).and_then(|(trace, _)| {
                let instrs: Vec<&Instruction> = trace.iter().map(|s| s.instr).collect();
                pointer_slot::static_target(self.image, &instrs)
            });
            (site, hit)
        });
        let mut progress = false;
        for (site, hit) in found {
            if let Some((slot, target)) = hit {
                self.slot_checked.insert(site);
                self.pointer_slots.insert(site, (slot, target));
                if !self.entries.contains(&target) {
                    self.add_function(target);
                    progress = true;
                }
            }
        }

        if self.image.data_relocs.is_empty() {
            return progress;
        }

        let mut rip_jumps: Vec<(u64, u64)> = Vec::new();
        let mut rip_calls: Vec<(u64, u64)> = Vec::new();
        for block in self.blocks.values() {
            if block.terminator == Terminator::IndirectJump
                && let Some(site) = block.instructions.last().map(Instruction::ip)
                && !self.slot_checked.contains(&site)
            {
                for edge in &block.successors {
                    if let Edge::IndirectJump { slot: Some(slot) } = *edge {
                        rip_jumps.push((site, slot));
                    }
                }
            }
            for call in &block.calls {
                if let CallTarget::Indirect { slot: Some(slot) } = call.target
                    && !self.slot_checked.contains(&call.site)
                {
                    rip_calls.push((call.site, slot));
                }
            }
        }
        for (site, slot) in rip_jumps {
            self.slot_checked.insert(site);
            if let Some(target) = self.relocated_code_target(slot) {
                self.pointer_slots.insert(site, (slot, target));
                if !self.entries.contains(&target) {
                    self.add_function(target);
                    progress = true;
                }
            }
        }
        for (site, slot) in rip_calls {
            self.slot_checked.insert(site);
            if let Some(target) = self.relocated_code_target(slot)
                && !self.entries.contains(&target)
            {
                self.add_function(target);
                progress = true;
            }
        }
        progress
    }

    /// Scans newly decoded blocks for RIP-relative references to `.rdata`/data vtables and
    /// function-pointer tables (strictly gated by PE `DIR64` `data_relocs`) as well as direct
    /// address-taken `.pdata` function entries (`lea reg, [rip + code_va]`). Returns whether any
    /// new function entry was seeded.
    fn seed_vtable_callbacks(&mut self) -> bool {
        let has_relocs = !self.image.data_relocs.is_empty();
        let table = &self.image.function_table;
        if !has_relocs && table.is_empty() {
            return false;
        }
        let mut data_vas: BTreeSet<u64> = BTreeSet::new();
        let mut code_entries: BTreeSet<u64> = BTreeSet::new();
        for block in self.blocks.values() {
            if !self.vtable_scanned_blocks.insert((block.start, block.end)) {
                continue;
            }
            for instr in &block.instructions {
                if !instr.is_ip_rel_memory_operand() {
                    continue;
                }
                let next_ip = instr.next_ip();
                let target = instr.ip_rel_memory_address();
                let rel = target.wrapping_sub(next_ip) as i64;
                if !(i64::from(i32::MIN)..=i64::from(i32::MAX)).contains(&rel)
                    || next_ip.checked_add_signed(rel) != Some(target)
                {
                    continue;
                }
                match self.image.region_of(target) {
                    Region::Data if has_relocs && !self.scanned_data_refs.contains(&target) => {
                        data_vas.insert(target);
                    }
                    Region::Code
                        if instr.mnemonic() == Mnemonic::Lea
                            && !table.is_empty()
                            && !self.entries.contains(&target)
                            && table.function_at(target) == Some(target)
                            && self.image.code_bytes(target).is_ok() =>
                    {
                        code_entries.insert(target);
                    }
                    _ => {}
                }
            }
        }

        let mut progress = false;
        for data_va in data_vas {
            self.scanned_data_refs.insert(data_va);
            if !data_va.is_multiple_of(8) && !self.image.data_relocs.contains(&data_va) {
                continue;
            }
            let mut slot = data_va;
            for _ in 0..MAX_VTABLE_ENTRIES {
                let Some(target) = self.relocated_code_target(slot) else {
                    break;
                };
                if !self.entries.contains(&target) {
                    self.add_function(target);
                    progress = true;
                }
                let Some(next_slot) = slot.checked_add(8) else {
                    break;
                };
                slot = next_slot;
            }
        }
        for code_va in code_entries {
            if !self.entries.contains(&code_va) {
                self.add_function(code_va);
                progress = true;
            }
        }
        progress
    }

    /// Try every jump decoded since the last round. Returns whether any table was resolved.
    fn resolve_jump_tables(&mut self) -> bool {
        let pending = std::mem::take(&mut self.pending_jumps);
        if pending.is_empty() {
            return false;
        }
        // One predecessor snapshot per round keeps the outcome independent of visiting order.
        self.check_preds();
        let preds = &self.preds;
        let pending: Vec<u64> = pending.into_iter().collect();
        let matched: Vec<JumpTable> = par_map(self.threads, &pending, |&site| {
            self.match_jump_table_raw(site, preds)
        })
        .into_iter()
        .flatten()
        .collect();
        // Tables found in the same round bound each other's inferred extents too.
        let mut bases = self.table_bases();
        bases.extend(matched.iter().flat_map(JumpTable::tables).map(|t| t.table));
        let resolved: Vec<JumpTable> = matched
            .into_iter()
            .map(|t| clip_inferred(&self.blocks, t, &bases))
            .collect();
        let progress = !resolved.is_empty();
        let done: BTreeSet<u64> = resolved.iter().map(|t| t.site).collect();
        for &site in &pending {
            if done.contains(&site) {
                self.failed_jumps.remove(&site);
            } else if let Some(shape) = self.jump_shape(site, preds) {
                self.failed_jumps.insert(site, shape);
            }
        }
        for table in resolved {
            if let Some(start) = self.jump_block(table.site) {
                let targets: BTreeSet<u64> = table.all_targets().collect();
                self.set_successors(start, targets.into_iter().map(Edge::JumpTable).collect());
            }
            let tail_calls = jump_table::is_fn_table(self.image, &table);
            for target in table.unique_targets() {
                // A function-pointer table's cases are other functions' entries (tail calls).
                if tail_calls {
                    self.add_function(target);
                } else {
                    self.add_leader(target);
                }
            }
            self.jump_tables.insert(table.site, table);
        }
        progress
    }

    /// What a jump's match depends on locally: the first block of its slice, the slice length and
    /// the incoming edges of that block.
    fn jump_shape(&self, site: u64, preds: &PredIndex) -> Option<JumpShape> {
        let (trace, first_block) = self.jump_trace(site, preds)?;
        let incoming = preds.get(&first_block).map_or(0, |p| p.count);
        Some((first_block, trace.len(), incoming))
    }

    /// Queue again every unresolved jump whose slice changed since it was last tried. Returns
    /// whether any was queued.
    fn requeue_changed_jumps(&mut self) -> bool {
        if self.failed_jumps.is_empty() {
            return false;
        }
        self.check_preds();
        let preds = &self.preds;
        let changed: Vec<u64> = self
            .failed_jumps
            .iter()
            .filter(|(site, shape)| {
                !self.jump_tables.contains_key(site)
                    && self.jump_retries.get(site).copied().unwrap_or(0) < MAX_TABLE_REVISIONS
                    && self.jump_shape(**site, preds).is_some_and(|now| now != **shape)
            })
            .map(|(&site, _)| site)
            .collect();
        for &site in &changed {
            let retries = self.jump_retries.entry(site).or_insert(0);
            *retries = retries.saturating_add(1);
            self.pending_jumps.insert(site);
        }
        !changed.is_empty()
    }

    /// Re-match every resolved table against the current CFG and adopt a changed resolution.
    /// A table resolved early can be too narrow: e.g. a switch in a loop whose index is zeroed
    /// before the loop resolves to the first case only until the loop's back edges are known.
    /// New targets become leaders (the caller runs the worklist again). Resolutions that no
    /// longer match at all are left to [`Self::verify_jump_tables`]. Each table is revised at most
    /// [`MAX_TABLE_REVISIONS`] times, so the rounds terminate.
    fn revise_jump_tables(&mut self) -> bool {
        self.check_preds();
        let preds = &self.preds;
        let candidates: Vec<(&u64, &JumpTable)> = self
            .jump_tables
            .iter()
            .filter(|(site, _)| {
                self.table_revisions.get(site).copied().unwrap_or(0) < MAX_TABLE_REVISIONS
            })
            .collect();
        let bases = self.table_bases();
        let changed: Vec<JumpTable> = par_map(self.threads, &candidates, |&(site, table)| {
            let now = self.match_jump_table(*site, preds, &bases)?;
            (now != *table).then_some(now)
        })
        .into_iter()
        .flatten()
        .collect();
        let progress = !changed.is_empty();
        for table in changed {
            let revisions = self.table_revisions.entry(table.site).or_insert(0);
            *revisions = revisions.saturating_add(1);
            if let Some(start) = self.jump_block(table.site) {
                let targets: BTreeSet<u64> = table.all_targets().collect();
                self.set_successors(start, targets.into_iter().map(Edge::JumpTable).collect());
            }
            let tail_calls = jump_table::is_fn_table(self.image, &table);
            for target in table.unique_targets() {
                // A function-pointer table's cases are other functions' entries (tail calls).
                if tail_calls {
                    self.add_function(target);
                } else {
                    self.add_leader(target);
                }
            }
            self.jump_tables.insert(table.site, table);
        }
        progress
    }

    /// Demote resolutions that no longer hold in the final CFG (e.g. the dispatch block gained a
    /// second predecessor), until the remaining set is stable.
    fn verify_jump_tables(&mut self) {
        loop {
            self.check_preds();
            let preds = &self.preds;
            let tables: Vec<(&u64, &JumpTable)> = self.jump_tables.iter().collect();
            let bases = self.table_bases();
            let stale: Vec<u64> = par_map(self.threads, &tables, |&(site, table)| {
                (self.match_jump_table(*site, preds, &bases).as_ref() != Some(table))
                    .then_some(*site)
            })
            .into_iter()
            .flatten()
            .collect();
            if stale.is_empty() {
                return;
            }
            for site in stale {
                self.jump_tables.remove(&site);
                if let Some(start) = self.jump_block(site) {
                    self.set_successors(start, vec![Edge::IndirectJump { slot: None }]);
                }
            }
        }
    }

    /// Phase 2: assign blocks to functions and classify tail calls.
    fn into_functions(mut self) -> BTreeMap<u64, Function> {
        let entries: Vec<u64> = self.entries.iter().copied().collect();
        let mut roots: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
        // A target that became another function's entry after it was recorded is that
        // function's entry, not a resume target.
        for &(owner, target) in &self.resume_roots {
            if target == owner || !self.entries.contains(&target) {
                roots.entry(owner).or_default().push(target);
            }
        }

        // Membership: blocks reachable from each entry through intra-function edges (one
        // independent walk per entry, run as a parallel wave).
        let members: Vec<Vec<u64>> = par_map(self.threads, &entries, |&entry| {
            let mut seen: BTreeSet<u64> = BTreeSet::new();
            let mut stack: Vec<u64> = vec![entry];
            stack.extend(roots.get(&entry).into_iter().flatten().rev());
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
            list
        });
        let mut refcount: BTreeMap<u64, usize> = BTreeMap::new();
        for &va in members.iter().flatten() {
            let count = refcount.entry(va).or_default();
            *count = count.saturating_add(1);
        }

        // Materialise: move a block into its last owner, clone it for earlier ones.
        let mut functions = BTreeMap::new();
        for (entry, list) in entries.into_iter().zip(members) {
            let mut function = Function {
                entry,
                blocks: BTreeMap::new(),
                callees: BTreeSet::new(),
                resume_targets: BTreeSet::new(),
            };
            for va in list {
                let remaining = refcount.get_mut(&va).map(|count| {
                    *count = count.saturating_sub(1);
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
            let owned = roots.get(&entry).into_iter().flatten();
            function.resume_targets = owned
                .chain(self.return_sites.iter())
                .copied()
                .filter(|va| function.blocks.contains_key(va))
                .collect();
            functions.insert(entry, function);
        }
        functions
    }
}

/// `prefix` (synthetic instructions, `true` = a branch known not taken) followed by `trace`.
fn prefixed<'a>(prefix: &'a [(Instruction, bool)], trace: &[Step<'a>]) -> Vec<Step<'a>> {
    let mut out: Vec<Step<'a>> = prefix
        .iter()
        .map(|(instr, not_taken)| Step {
            instr,
            taken: not_taken.then_some(false),
        })
        .collect();
    out.extend(trace.iter().copied());
    out
}

/// 64-bit general purpose registers (other than `rsp`) the trace reads before it writes them.
fn live_in_registers(trace: &[Step<'_>]) -> Vec<iced_x86::Register> {
    use iced_x86::OpAccess;
    let mut info = iced_x86::InstructionInfoFactory::new();
    let mut written: Vec<iced_x86::Register> = Vec::new();
    let mut live: Vec<iced_x86::Register> = Vec::new();
    for step in trace {
        let used = info.info(step.instr).used_registers().to_vec();
        for u in &used {
            let full = u.register().full_register();
            if !full.is_gpr64() || full == iced_x86::Register::RSP {
                continue;
            }
            let reads = matches!(
                u.access(),
                OpAccess::Read | OpAccess::CondRead | OpAccess::ReadWrite | OpAccess::ReadCondWrite
            );
            if reads && !written.contains(&full) && !live.contains(&full) {
                live.push(full);
            }
        }
        for u in &used {
            let full = u.register().full_register();
            if full.is_gpr64()
                && matches!(u.access(), OpAccess::Write | OpAccess::ReadWrite)
                && !written.contains(&full)
            {
                written.push(full);
            }
        }
    }
    live
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

    /// An inferred extent that runs into the next table is cut at that table's base.
    #[test]
    fn inferred_extent_is_clipped_at_the_next_table() {
        let mut load =
            Decoder::with_ip(64, &[0x48, 0x63, 0x04, 0x81], TEXT, DecoderOptions::NONE).decode(); // movsxd rax, dword ptr [rcx+rax*4]
        load.set_ip(TEXT);
        let mut block = BasicBlock::new(TEXT);
        block.instructions.push(load);
        let blocks = BTreeMap::from([(TEXT, block)]);
        let table = |n: usize, inferred: bool| JumpTable {
            site: TEXT + 4,
            load_site: TEXT,
            table: DATA,
            targets: (0..n as u64).map(|i| TEXT + 0x100 + i).collect(),
            index: None,
            inferred,
            alternates: Vec::new(),
        };
        let bases = BTreeSet::from([DATA, DATA + 0x10, DATA + 0x20]);
        let clipped = clip_inferred(&blocks, table(20, true), &bases);
        assert_eq!(clipped.targets.len(), 4);
        // A proven extent, and one without a neighbour inside, are untouched.
        assert_eq!(
            clip_inferred(&blocks, table(20, false), &bases)
                .targets
                .len(),
            20
        );
        assert_eq!(
            clip_inferred(&blocks, table(3, true), &bases).targets.len(),
            3
        );
        // Fewer than two entries would remain: keep the table as inferred.
        let near = BTreeSet::from([DATA + 4]);
        assert_eq!(
            clip_inferred(&blocks, table(20, true), &near).targets.len(),
            20
        );
    }

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

    fn analyze_eh(
        code: &[u8],
        entries: &[u64],
        eh: crate::eh::EhSeeds,
        table: crate::pdata::FunctionTable,
    ) -> Analysis {
        let mut image = CodeImage::new(
            BASE,
            vec![
                CodeSection::new(TEXT, code, true),
                CodeSection::new(DATA, &DATA_BYTES, false),
            ],
            entries.iter().map(|e| TEXT + e).collect(),
        );
        image.eh = eh;
        image.function_table = table;
        discover(&image)
    }

    fn resume_offsets(function: &Function) -> Vec<u64> {
        function.resume_targets.iter().map(|va| va - TEXT).collect()
    }

    #[test]
    fn eh_resume_targets_split_blocks_and_root_the_owner() {
        let code = [
            0x48, 0x83, 0xec, 0x28, // 0: sub rsp, 0x28
            0xe8, 0x17, 0x00, 0x00, 0x00, // 4: call 0x20
            0x90, // 9: nop          <- resume target (splits the block)
            0x48, 0x83, 0xc4, 0x28, // a: add rsp, 0x28
            0xc3, // e: ret
            0xcc, // f: padding
            0xb8, 0x01, 0x00, 0x00, 0x00, // 10: mov eax, 1 <- reachable only by unwinding
            0xeb, 0xf3, // 15: jmp 0xa
            0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, // 17: padding
            0xc3, // 20: ret
        ];
        let eh = crate::eh::EhSeeds {
            resume_targets: vec![
                (TEXT, TEXT + 0x9),
                (TEXT, TEXT + 0x10),
                // Another function's entry is never a resume target.
                (TEXT, TEXT + 0x20),
            ],
            ..Default::default()
        };
        let analysis = analyze_eh(&code, &[0], eh, Default::default());
        let f = func(&analysis, 0);
        assert_eq!(block_starts(f), [0x0, 0x9, 0xa, 0x10]);
        assert_eq!(resume_offsets(f), [0x9, 0x10]);
        assert_eq!(block(f, 0x0).successors, [Edge::Fallthrough(TEXT + 0x9)]);
        assert_eq!(block(f, 0x10).successors, [Edge::Jump(TEXT + 0xa)]);
        assert!(func(&analysis, 0x20).resume_targets.is_empty());

        // Without the seeds, the unwinder-only block does not exist.
        let plain = analyze(&code, &[0]);
        assert_eq!(block_starts(func(&plain, 0)), [0x0]);
        assert!(func(&plain, 0).resume_targets.is_empty());
    }

    #[test]
    fn setjmp_return_sites_are_resume_targets() {
        let slot_call = (DATA as i64 - (TEXT as i64 + 6)) as i32;
        let slot_jump = (DATA as i64 - (TEXT as i64 + 0x26)) as i32;
        let code = [
            &[0xff, 0x15][..],
            &slot_call.to_le_bytes(), // 0: call [rip+x] -> DATA (setjmp slot)
            &[0x85, 0xc0],            // 6: test eax, eax   <- return site
            &[0xe8, 0x13, 0x00, 0x00, 0x00], // 8: call 0x20 (setjmp import thunk)
            &[0xc3],                  // d: ret             <- return site
            &[0xcc; 0x12],            // e: padding
            &[0xff, 0x25],
            &slot_jump.to_le_bytes(), // 20: jmp [rip+x] -> DATA
        ]
        .concat();
        let eh = crate::eh::EhSeeds {
            setjmp_slots: vec![DATA],
            ..Default::default()
        };
        let analysis = analyze_eh(&code, &[0], eh, Default::default());
        let f = func(&analysis, 0);
        assert_eq!(block_starts(f), [0x0, 0x6, 0xd]);
        assert_eq!(resume_offsets(f), [0x6, 0xd]);
        assert_eq!(block(f, 0x0).successors, [Edge::Fallthrough(TEXT + 0x6)]);
        assert!(func(&analysis, 0x20).resume_targets.is_empty());
    }

    #[test]
    fn address_taken_interior_code_is_a_resume_target_of_its_pdata_function() {
        let lea = |reg: u8, at: u64, target: u64| {
            let disp = (target as i64 - (at as i64 + 7)) as i32;
            [&[0x48, 0x8d, reg][..], &disp.to_le_bytes()].concat()
        };
        let code = [
            lea(0x05, 0x0, 0x1c),   // 0: lea rax, [B+4]  (interior instruction boundary)
            lea(0x0d, 0x7, 0x1d),   // 7: lea rcx, [B+5]  (inside an instruction: ignored)
            lea(0x15, 0xe, 0x18),   // e: lea rdx, [B]    (function entry: ignored)
            vec![0xc3, 0xcc, 0xcc], // 15: ret
            vec![0x90, 0x90, 0x90, 0x90], // 18: B: nop x4
            vec![0xb8, 0x02, 0x00, 0x00, 0x00], // 1c: mov eax, 2
            vec![0xc3],             // 21: ret
        ]
        .concat();
        let range = |start: u64, end: u64| crate::pdata::FunctionRange {
            start: TEXT + start,
            end: TEXT + end,
            function: TEXT + start,
        };
        let table = crate::pdata::FunctionTable::new(vec![range(0, 0x18), range(0x18, 0x22)]);
        let analysis = analyze_eh(&code, &[0, 0x18], Default::default(), table);
        let b = func(&analysis, 0x18);
        assert_eq!(block_starts(b), [0x18, 0x1c]);
        assert_eq!(resume_offsets(b), [0x1c]);
        assert!(func(&analysis, 0).resume_targets.is_empty());

        // Without .pdata extents the owner is unknown: nothing is added.
        let analysis = analyze(&code, &[0, 0x18]);
        assert_eq!(block_starts(func(&analysis, 0x18)), [0x18]);
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

    #[test]
    fn overlapping_entries_are_order_independent() {
        // 0: mov eax, 0xc3 (bytes 0..5; byte 1 is also `ret`), 5: ret.
        let code = [0xb8, 0xc3, 0x00, 0x00, 0x00, 0xc3];
        let forward = analyze(&code, &[0, 1]);
        let reversed = analyze(&code, &[1, 0]);
        let duplicated = analyze(&code, &[1, 0, 1, 0]);
        assert_eq!(forward, reversed);
        assert_eq!(forward, duplicated);

        let f0 = func(&forward, 0);
        let f1 = func(&forward, 1);
        assert!(
            f0.diagnostics()
                .any(|d| d.kind == DiagnosticKind::OverlappingInstruction && d.address == TEXT + 1)
        );
        assert!(
            f1.diagnostics()
                .any(|d| d.kind == DiagnosticKind::OverlappingInstruction && d.address == TEXT + 1)
        );
        assert_eq!(block(f1, 1).terminator, Terminator::Return);
    }

    #[test]
    fn direct_call_into_own_interior_preserves_subsequent_leader_boundary() {
        // 0: call TEXT+1 (5 bytes: e8 fc ff ff ff), 5: nop, 6: ret.
        let code = [0xe8, 0xfc, 0xff, 0xff, 0xff, 0x90, 0xc3];
        let forward = analyze(&code, &[0, 6]);
        let reversed = analyze(&code, &[6, 0]);
        assert_eq!(forward, reversed);

        let f0 = func(&forward, 0);
        assert_eq!(block_starts(f0), [0]);
        let b0 = block(f0, 0);
        assert_eq!(instr_offsets(b0), [0, 5]);
        assert_eq!(b0.end, TEXT + 6);
        assert_eq!(b0.terminator, Terminator::Fallthrough);
        assert_eq!(b0.successors, [Edge::TailCall(TEXT + 6)]);

        let f1 = func(&forward, 1);
        assert!(
            f1.diagnostics()
                .any(|d| d.kind == DiagnosticKind::OverlappingInstruction && d.address == TEXT + 1)
        );
        assert_eq!(block_starts(func(&forward, 6)), [6]);
    }

    #[test]
    fn malformed_eh_resume_targets_are_ignored_and_deferred_callees_are_attached() {
        // 0: call 0x10; ret; 10: nop; nop; ret; 18: mov eax, 1; jmp 0x11
        let code = [
            0xe8, 0x0b, 0x00, 0x00, 0x00, // 0: call 0x10
            0xc3, // 5: ret
            0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, // 6..f: padding
            0x90, // 10: nop (callee entry, discovered via call)
            0x90, // 11: nop
            0xc3, // 12: ret
            0xcc, 0xcc, 0xcc, 0xcc, 0xcc, // 13..17: padding
            0xb8, 0x01, 0x00, 0x00, 0x00, // 18: mov eax, 1 (unwinder-only continuation)
            0xeb, 0xf2, // 1d: jmp 0x11
        ];
        let mut text_section = CodeSection::new(TEXT, &code, true);
        // Executable zero-fill tail at TEXT+0x20..TEXT+0x100 (Region::Code, not decodable).
        text_section.virtual_size = 0x100;

        let build = |resume_targets: Vec<(u64, u64)>, entries: Vec<u64>| {
            let mut image = CodeImage::new(
                BASE,
                vec![text_section, CodeSection::new(DATA, &DATA_BYTES, false)],
                entries,
            );
            image.eh = crate::eh::EhSeeds {
                resume_targets,
                ..Default::default()
            };
            discover(&image)
        };

        let valid_deferred = (TEXT + 0x10, TEXT + 0x18);
        let seeds_a = vec![
            valid_deferred,
            (TEXT, DATA),                  // non-executable target
            (TEXT, 0),                     // unmapped target
            (TEXT, u64::MAX),              // max VA target
            (TEXT, TEXT + 0x40),           // zero-fill executable tail
            (DATA, TEXT + 0x11),           // non-executable owner
            (u64::MAX, TEXT + 0x11),       // unmapped owner
            (TEXT + 0x13, TEXT + 0x18),    // uncalled code owner (never a function)
            valid_deferred,                // duplicate
        ];
        let mut seeds_b = seeds_a.clone();
        seeds_b.reverse();

        let a = build(seeds_a, vec![TEXT]);
        let b = build(seeds_b, vec![TEXT, TEXT]);
        assert_eq!(a, b);

        assert_eq!(
            a.functions.keys().copied().collect::<Vec<_>>(),
            [TEXT, TEXT + 0x10]
        );
        assert!(func(&a, 0).resume_targets.is_empty());
        assert_eq!(func(&a, 0).diagnostics().count(), 0);
        let callee = func(&a, 0x10);
        assert_eq!(resume_offsets(callee), [0x18]);
        assert_eq!(block_starts(callee), [0x10, 0x11, 0x18]);
        assert_eq!(callee.diagnostics().count(), 0);
    }

    #[test]
    fn late_lea_into_zero_fill_tail_and_setjmp_at_eof_are_bounds_checked() {
        // 0: lea rax, [rip + 0x39] -> TEXT + 0x40 (zero-fill tail of .text)
        // 7: call [rip + slot] -> DATA (setjmp at section EOF)
        let slot_call = (DATA as i64 - (TEXT as i64 + 0xd)) as i32;
        let code = [
            &[0x48, 0x8d, 0x05, 0x39, 0x00, 0x00, 0x00][..],
            &[0xff, 0x15],
            &slot_call.to_le_bytes(),
        ]
        .concat();
        let mut text_section = CodeSection::new(TEXT, &code, true);
        text_section.virtual_size = 0x100;
        let mut image = CodeImage::new(
            BASE,
            vec![text_section, CodeSection::new(DATA, &DATA_BYTES, false)],
            vec![TEXT],
        );
        image.eh = crate::eh::EhSeeds {
            setjmp_slots: vec![DATA],
            ..Default::default()
        };
        image.function_table = crate::pdata::FunctionTable::new(vec![crate::pdata::FunctionRange {
            start: TEXT,
            end: TEXT + 0x100,
            function: TEXT,
        }]);

        let analysis = discover(&image);
        let f = func(&analysis, 0);
        assert!(
            f.resume_targets.is_empty(),
            "unmapped EOF and zero-fill targets must not be resume targets"
        );
        assert_eq!(block_starts(f), [0, 0xd]);
        assert_eq!(
            block(f, 0xd).diagnostics[0].kind,
            DiagnosticKind::NotExecutable
        );
    }

    #[test]
    fn addresses_near_u64_max_do_not_overflow_or_wrap() {
        // 3 bytes placed at u64::MAX - 2: byte 2 would sit at u64::MAX and wrap to 0 if decoded.
        let high_start = u64::MAX - 2;
        let nops = [0x90, 0x90, 0x90];
        let image = CodeImage::new(
            high_start,
            vec![CodeSection::new(high_start, &nops, true)],
            vec![high_start, 0],
        );
        let analysis = discover(&image);

        let f_high = analysis.function(high_start).expect("high function");
        let b_high = f_high.blocks.get(&high_start).expect("high block");
        assert_eq!(b_high.start, high_start);
        assert_eq!(b_high.end, u64::MAX);
        assert_eq!(b_high.instructions.len(), 2);
        assert_eq!(b_high.successors, [Edge::Fallthrough(u64::MAX)]);

        let b_max = f_high.blocks.get(&u64::MAX).expect("u64::MAX block");
        assert_eq!(b_max.terminator, Terminator::Invalid);
        assert_eq!(b_max.diagnostics[0].kind, DiagnosticKind::Unmapped);

        // When u64::MAX is itself an entry point, fallthrough becomes a tail call cleanly.
        let image_with_max = CodeImage::new(
            high_start,
            vec![CodeSection::new(high_start, &nops, true)],
            vec![high_start, u64::MAX, 0],
        );
        let analysis_with_max = discover(&image_with_max);
        assert_eq!(
            analysis_with_max
                .function(high_start)
                .and_then(|f| f.entry_block())
                .map(|b| b.successors.as_slice()),
            Some(&[Edge::TailCall(u64::MAX)][..])
        );
        assert_eq!(
            analysis_with_max
                .function(u64::MAX)
                .and_then(|f| f.entry_block())
                .map(|b| b.diagnostics[0].kind),
            Some(DiagnosticKind::Unmapped)
        );

        assert_eq!(
            analysis
                .function(0)
                .and_then(|f| f.entry_block())
                .map(|b| b.terminator),
            Some(Terminator::Invalid)
        );

        // Multi-byte instruction crossing u64::MAX becomes TruncatedInstruction, never wraps.
        let trunc_start = u64::MAX - 1;
        let mov = [0x48, 0x89, 0xc8];
        let trunc_image = CodeImage::new(
            trunc_start,
            vec![CodeSection::new(trunc_start, &mov, true)],
            vec![trunc_start],
        );
        let trunc_analysis = discover(&trunc_image);
        let b_trunc = trunc_analysis
            .function(trunc_start)
            .and_then(|f| f.entry_block())
            .expect("truncated block");
        assert!(b_trunc.instructions.is_empty());
        assert_eq!(b_trunc.terminator, Terminator::Invalid);
        assert_eq!(
            b_trunc.diagnostics[0].kind,
            DiagnosticKind::TruncatedInstruction
        );

        // `leaf_extent` at u64::MAX must not overflow `site + 1`.
        let mut state = Discovery::new(&image, 1);
        state.add_function(high_start);
        state.add_function(u64::MAX);
        assert_eq!(state.leaf_extent(u64::MAX), None);

        // `jump_table_relocations` near u64::MAX must not overflow `(table.table..).step_by(4)`.
        let table = JumpTable {
            site: high_start,
            load_site: high_start,
            table: u64::MAX - 2,
            targets: vec![high_start, high_start + 1, high_start + 2],
            index: None,
            inferred: false,
            alternates: Vec::new(),
        };
        let mut relocs = Vec::new();
        jump_table_relocations(&image, &table, &mut relocs);
        assert_eq!(
            relocs
                .iter()
                .filter(|r| r.kind == RelocKind::JumpTableEntry)
                .map(|r| r.site)
                .collect::<Vec<_>>(),
            vec![u64::MAX - 2]
        );
    }

    #[test]
    fn jump_trace_breaks_cycles_without_duplicating_blocks() {
        // 0: ret (entry)
        // 2: jmp 4 (unwinder-only cycle head; sole predecessor is block 4)
        // 4: test ecx, ecx; je 2; jmp rax (sole predecessor is block 2)
        let code = [
            0xc3, // 0: ret
            0xcc, // 1: padding
            0xeb, 0x00, // 2: jmp 4
            0x85, 0xc9, // 4: test ecx, ecx
            0x74, 0xfa, // 6: je 2
            0xff, 0xe0, // 8: jmp rax
        ];
        let eh = crate::eh::EhSeeds {
            resume_targets: vec![(TEXT, TEXT + 2)],
            ..Default::default()
        };
        let mut image = CodeImage::new(
            BASE,
            vec![
                CodeSection::new(TEXT, &code, true),
                CodeSection::new(DATA, &DATA_BYTES, false),
            ],
            vec![TEXT],
        );
        image.eh = eh;
        let mut state = Discovery::new(&image, 1);
        state.add_function(TEXT);
        state.seed_resume_target(TEXT, TEXT + 2);
        state.run();
        state.check_preds();
        let preds = &state.preds;
        let (trace, first_block) = state
            .jump_trace(TEXT + 8, preds)
            .expect("jump trace exists");
        assert_eq!(first_block, TEXT + 2);
        let ips: Vec<u64> = trace.iter().map(|s| s.instr.ip() - TEXT).collect();
        assert_eq!(ips, [2, 4, 6, 8]);
    }

    #[test]
    fn cross_function_pdata_jmp_and_jcc_tail_calls_are_seeded() {
        // Entry at 0x00 only; 0x10 (callee A) and 0x20 (callee B) are in .pdata but NOT initially
        // in `image.entry_points`. 0x30 is a cold chunk owned by 0x00 (must stay intra-function!).
        //   0x00: test ecx, ecx; je 0x10 (jcc tail call); jmp 0x30 (same primary -> intra)
        //   0x10: mov eax, 1; ret
        //   0x20: mov eax, 2; ret
        //   0x30: jmp 0x20 (unconditional tail call to another .pdata function)
        let mut code = vec![0xccu8; 0x40];
        code[0x00..0x09].copy_from_slice(&[
            0x85, 0xc9, // 0x00: test ecx, ecx
            0x74, 0x0c, // 0x02: je 0x10
            0xe9, 0x27, 0x00, 0x00, 0x00, // 0x04: jmp 0x30
        ]);
        code[0x10..0x16].copy_from_slice(&[0xb8, 0x01, 0x00, 0x00, 0x00, 0xc3]);
        code[0x20..0x26].copy_from_slice(&[0xb8, 0x02, 0x00, 0x00, 0x00, 0xc3]);
        code[0x30..0x35].copy_from_slice(&[0xe9, 0xeb, 0xff, 0xff, 0xff]); // jmp 0x20

        let table = crate::pdata::FunctionTable::new(vec![
            crate::pdata::FunctionRange {
                start: TEXT,
                end: TEXT + 0x10,
                function: TEXT,
            },
            crate::pdata::FunctionRange {
                start: TEXT + 0x10,
                end: TEXT + 0x20,
                function: TEXT + 0x10,
            },
            crate::pdata::FunctionRange {
                start: TEXT + 0x20,
                end: TEXT + 0x30,
                function: TEXT + 0x20,
            },
            crate::pdata::FunctionRange {
                start: TEXT + 0x30,
                end: TEXT + 0x40,
                function: TEXT, // cold chunk of TEXT
            },
        ]);

        let analysis = analyze_eh(&code, &[0], Default::default(), table);
        assert_eq!(
            analysis.functions.keys().copied().collect::<Vec<_>>(),
            [TEXT, TEXT + 0x10, TEXT + 0x20]
        );
        let main_fn = func(&analysis, 0);
        // Cold chunk 0x30 remains part of main_fn; 0x10 and 0x20 are tail calls.
        assert_eq!(block_starts(main_fn), [0x0, 0x4, 0x30]);
        assert_eq!(
            block(main_fn, 0x0).successors,
            [Edge::TailCall(TEXT + 0x10), Edge::Fallthrough(TEXT + 0x4)]
        );
        assert_eq!(block(main_fn, 0x4).successors, [Edge::Jump(TEXT + 0x30)]);
        assert_eq!(
            block(main_fn, 0x30).successors,
            [Edge::TailCall(TEXT + 0x20)]
        );
        assert_eq!(block_starts(func(&analysis, 0x10)), [0x10]);
        assert_eq!(block_starts(func(&analysis, 0x20)), [0x20]);
    }

    #[test]
    fn rdata_vtable_and_direct_code_callbacks_are_discovered_and_reloc_gated() {
        // Code layout:
        //   0x00: lea rax, [rip -> DATA+0x00]; mov [rcx], rax; lea rdx, [rip -> TEXT+0x40]; ret
        //   0x20: virtual method 0 (.pdata function): mov eax, 1; ret
        //   0x30: virtual method 1 (leaf function in gap): mov eax, 2; ret
        //   0x40: direct address-taken .pdata callback: mov eax, 3; ret
        //   0x50: padding (`int3`): must NOT be seeded even if relocated
        //   0x60: method after stopper: must NOT be reached past the padding/non-reloc entry
        let mut code = vec![0xccu8; 0x70];
        let lea_vtable = (DATA as i64 - (TEXT as i64 + 7)) as i32;
        let lea_cb = ((TEXT + 0x40) as i64 - (TEXT as i64 + 0x11)) as i32;
        code[0x00..0x03].copy_from_slice(&[0x48, 0x8d, 0x05]);
        code[0x03..0x07].copy_from_slice(&lea_vtable.to_le_bytes());
        code[0x07..0x0a].copy_from_slice(&[0x48, 0x89, 0x01]); // mov [rcx], rax
        code[0x0a..0x0d].copy_from_slice(&[0x48, 0x8d, 0x15]);
        code[0x0d..0x11].copy_from_slice(&lea_cb.to_le_bytes());
        code[0x11] = 0xc3; // ret
        code[0x20..0x26].copy_from_slice(&[0xb8, 0x01, 0x00, 0x00, 0x00, 0xc3]);
        code[0x30..0x36].copy_from_slice(&[0xb8, 0x02, 0x00, 0x00, 0x00, 0xc3]);
        code[0x40..0x46].copy_from_slice(&[0xb8, 0x03, 0x00, 0x00, 0x00, 0xc3]);
        code[0x50] = 0xcc; // int3 padding
        code[0x60..0x66].copy_from_slice(&[0xb8, 0x04, 0x00, 0x00, 0x00, 0xc3]);

        // VTable at DATA+0x00:
        //   [0]: TEXT+0x20 (.pdata entry, relocated) -> SEEDED
        //   [1]: TEXT+0x30 (leaf function, relocated) -> SEEDED
        //   [2]: TEXT+0x50 (int3 padding, relocated) -> STOPS walk
        //   [3]: TEXT+0x60 (valid code past stopper, relocated) -> NOT reached
        let mut data = vec![0u8; 0x40];
        data[0x00..0x08].copy_from_slice(&(TEXT + 0x20).to_le_bytes());
        data[0x08..0x10].copy_from_slice(&(TEXT + 0x30).to_le_bytes());
        data[0x10..0x18].copy_from_slice(&(TEXT + 0x50).to_le_bytes());
        data[0x18..0x20].copy_from_slice(&(TEXT + 0x60).to_le_bytes());

        let pdata = crate::pdata::FunctionTable::new(vec![
            crate::pdata::FunctionRange {
                start: TEXT,
                end: TEXT + 0x18,
                function: TEXT,
            },
            crate::pdata::FunctionRange {
                start: TEXT + 0x20,
                end: TEXT + 0x28,
                function: TEXT + 0x20,
            },
            crate::pdata::FunctionRange {
                start: TEXT + 0x40,
                end: TEXT + 0x48,
                function: TEXT + 0x40,
            },
        ]);

        // 1) With data_relocs set: discovers vtable entries [0], [1] and direct callback 0x40.
        let mut img_ok = CodeImage::new(
            BASE,
            vec![
                CodeSection::new(TEXT, &code, true),
                CodeSection::new(DATA, &data, false),
            ],
            vec![TEXT],
        );
        img_ok.function_table = pdata.clone();
        for off in [0x00u64, 0x08, 0x10, 0x18] {
            img_ok.data_relocs.insert(DATA + off);
        }
        let a1 = discover_with_threads(&img_ok, 1);
        let a4 = discover_with_threads(&img_ok, 4);
        assert_eq!(a1, a4, "discovery must be thread-count invariant");
        assert_eq!(
            a1.functions.keys().copied().collect::<Vec<_>>(),
            [TEXT, TEXT + 0x20, TEXT + 0x30, TEXT + 0x40]
        );

        // 2) Without data_relocs: vtable entries are NOT seeded (only direct .pdata callback 0x40).
        let mut img_no_reloc = CodeImage::new(
            BASE,
            vec![
                CodeSection::new(TEXT, &code, true),
                CodeSection::new(DATA, &data, false),
            ],
            vec![TEXT],
        );
        img_no_reloc.function_table = pdata.clone();
        let a_no_reloc = discover(&img_no_reloc);
        assert_eq!(
            a_no_reloc.functions.keys().copied().collect::<Vec<_>>(),
            [TEXT, TEXT + 0x40]
        );

        // 3) Interior pointer to a .pdata function (TEXT+0x25, inside [0x20, 0x28)) stops the walk.
        let mut bad_data = data.clone();
        bad_data[0x00..0x08].copy_from_slice(&(TEXT + 0x25).to_le_bytes());
        let mut img_interior = CodeImage::new(
            BASE,
            vec![
                CodeSection::new(TEXT, &code, true),
                CodeSection::new(DATA, &bad_data, false),
            ],
            vec![TEXT],
        );
        img_interior.function_table = pdata;
        img_interior.data_relocs.insert(DATA);
        img_interior.data_relocs.insert(DATA + 0x08);
        let a_interior = discover(&img_interior);
        assert_eq!(
            a_interior.functions.keys().copied().collect::<Vec<_>>(),
            [TEXT, TEXT + 0x40]
        );
    }

    #[test]
    fn rip_relative_indirect_call_and_jmp_pointer_slots_seed_targets() {
        // 0x00: call [rip -> DATA+0x00] (relocated -> TEXT+0x20)
        // 0x06: call [rip -> DATA+0x08] (NOT relocated, e.g. IAT -> ignored)
        // 0x0c: jmp  [rip -> DATA+0x10] (relocated -> TEXT+0x30)
        let mut code = vec![0xccu8; 0x40];
        let d_call_ok = (DATA as i64 - (TEXT as i64 + 6)) as i32;
        let d_call_iat = ((DATA + 8) as i64 - (TEXT as i64 + 0x0c)) as i32;
        let d_jmp_ok = ((DATA + 0x10) as i64 - (TEXT as i64 + 0x12)) as i32;
        code[0x00..0x02].copy_from_slice(&[0xff, 0x15]);
        code[0x02..0x06].copy_from_slice(&d_call_ok.to_le_bytes());
        code[0x06..0x08].copy_from_slice(&[0xff, 0x15]);
        code[0x08..0x0c].copy_from_slice(&d_call_iat.to_le_bytes());
        code[0x0c..0x0e].copy_from_slice(&[0xff, 0x25]);
        code[0x0e..0x12].copy_from_slice(&d_jmp_ok.to_le_bytes());
        code[0x20] = 0xc3;
        code[0x28] = 0xc3;
        code[0x30] = 0xc3;

        let mut data = vec![0u8; 0x20];
        data[0x00..0x08].copy_from_slice(&(TEXT + 0x20).to_le_bytes());
        data[0x08..0x10].copy_from_slice(&(TEXT + 0x28).to_le_bytes());
        data[0x10..0x18].copy_from_slice(&(TEXT + 0x30).to_le_bytes());

        let mut image = CodeImage::new(
            BASE,
            vec![
                CodeSection::new(TEXT, &code, true),
                CodeSection::new(DATA, &data, false),
            ],
            vec![TEXT],
        );
        // Relocate only DATA+0x00 and DATA+0x10; leave DATA+0x08 unrelocated (like an IAT slot).
        image.data_relocs.insert(DATA);
        image.data_relocs.insert(DATA + 0x10);

        let a1 = discover_with_threads(&image, 1);
        let a4 = discover_with_threads(&image, 4);
        assert_eq!(a1, a4);
        assert_eq!(
            a1.functions.keys().copied().collect::<Vec<_>>(),
            [TEXT, TEXT + 0x20, TEXT + 0x30]
        );
        assert_eq!(
            a1.pointer_slots.get(&(TEXT + 0x0c)),
            Some(&(DATA + 0x10, TEXT + 0x30))
        );
        assert!(!a1.functions.contains_key(&(TEXT + 0x28)));
    }
}
