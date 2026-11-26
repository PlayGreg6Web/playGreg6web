//! Refinements between the pipeline stages. Each one is a separate, deterministic step whose
//! result is reported in the build manifest.
//!
//! 1. **No-return imports** ([`no_return_imports`]): IAT slots of a curated list of APIs that
//!    never return to their call site, so discovery ends the block at `call [rip+slot]` instead of
//!    decoding whatever follows. Import thunks (`jmp [rip+slot]` / `jmp` to such a target) are
//!    added during the fixpoint ([`no_return_thunks`]).
//! 2. **Code-pointer entry points** ([`discover_refined`]): functions referenced only as data
//!    (vtables, callbacks, function-pointer tables) are never reached by recursive descent. Two
//!    sources propose candidates: RIP-relative `lea` of a code address ([`lea_pointers`]) and
//!    `DIR64` base relocations in non-executable sections whose stored value points into code
//!    ([`base_reloc_pointers`]). See [`PointerStatus`] for the acceptance rule.
//! 3. **Dispatcher-reachable set**: every discovered function entry is lifted and registered;
//!    candidates that did not become entries are reported (see [`CodePointer::lifted`]).
//! 4. **Import slot consistency** ([`check_import_slots`]): slot reads found by discovery versus
//!    the IAT slots of the import bridge plan.

use std::collections::{BTreeMap, BTreeSet};

use iced_x86::{Decoder, DecoderOptions, FlowControl, Mnemonic};

use crate::cfg::{BasicBlock, Edge, Terminator};
use crate::code_image::{CodeImage, Region};
use crate::discover::{Analysis, discover};
use crate::image::{BaseRelocKind, GuestImage};
use crate::pe::{ImportedSymbol, Pe64};
use crate::reloc::{RelocKind, RelocationMap};
use crate::stubs::{BridgePlan, Resolution};

// ---------------------------------------------------------------------------------------------
// 1. No-return imports
// ---------------------------------------------------------------------------------------------

/// DLL families the no-return list is keyed by (a name only counts when imported from its
/// family, so an unrelated DLL exporting e.g. `exit` is not affected).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DllFamily {
    /// `kernel32`, `kernelbase`, `api-ms-win-core-*`.
    Kernel,
    /// `ntdll`.
    Ntdll,
    /// `ucrtbase(d)`, `msvcrt`, `msvcr*`, `vcruntime*`, `api-ms-win-crt-*`.
    Crt,
}

fn dll_family(dll: &str) -> Option<DllFamily> {
    let lower = dll.to_ascii_lowercase();
    let name = lower.strip_suffix(".dll").unwrap_or(&lower);
    if matches!(name, "kernel32" | "kernelbase") || name.starts_with("api-ms-win-core-") {
        Some(DllFamily::Kernel)
    } else if name == "ntdll" {
        Some(DllFamily::Ntdll)
    } else if matches!(name, "ucrtbase" | "ucrtbased" | "msvcrt")
        || name.starts_with("msvcr")
        || name.starts_with("vcruntime")
        || name.starts_with("api-ms-win-crt-")
    {
        Some(DllFamily::Crt)
    } else {
        None
    }
}

/// APIs that never return to their call site, by DLL family (`kernel`: `kernel32`,
/// `kernelbase`, `api-ms-win-core-*`; `ntdll`; `crt`: `ucrtbase(d)`, `msvcrt`, `msvcr*`,
/// `vcruntime*`, `api-ms-win-crt-*`; names compare case-sensitively). Only functions whose contract
/// guarantees this unconditionally are listed. Deliberately absent: `TerminateProcess` /
/// `TerminateThread` / `NtTerminateProcess` (return when the handle names another process or
/// thread, which is not provable statically), `RaiseException` (continuable exceptions return),
/// `RtlUnwind*` (resume at a caller-chosen target that may be the call site), and
/// `__fastfail` (an instruction, `int 0x29`, which discovery already treats as a trap).
pub const NO_RETURN_APIS: &[(&str, &[&str])] = &[
    (
        "kernel",
        &[
            "ExitProcess",
            "ExitThread",
            "FatalExit",
            "FatalAppExitA",
            "FatalAppExitW",
            "FreeLibraryAndExitThread",
            "RaiseFailFastException",
        ],
    ),
    (
        "ntdll",
        &["RtlExitUserProcess", "RtlExitUserThread", "RtlRaiseStatus"],
    ),
    (
        "crt",
        &[
            "exit",
            "_exit",
            "_Exit",
            "quick_exit",
            "abort",
            "terminate",
            "__std_terminate",
            "_amsg_exit",
            "_invalid_parameter_noinfo_noreturn",
            "_invoke_watson",
            "_CxxThrowException",
            "longjmp",
        ],
    ),
];

fn is_no_return(dll: &str, name: &str) -> bool {
    let family = match dll_family(dll) {
        Some(DllFamily::Kernel) => "kernel",
        Some(DllFamily::Ntdll) => "ntdll",
        Some(DllFamily::Crt) => "crt",
        None => return false,
    };
    NO_RETURN_APIS
        .iter()
        .any(|(f, names)| *f == family && names.contains(&name))
}

/// An imported API known never to return, with the IAT slot calls go through.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct NoReturnImport {
    /// VA of the 8-byte IAT slot.
    pub slot: u64,
    /// `DLL!name`.
    pub label: String,
}

/// Computes the checked VA of an 8-byte IAT slot at `base + rva`, verifying that the full 8-byte
/// slot `[va, va + 8)` does not overflow `u64`.
fn checked_slot_va(base: u64, rva: u32) -> Option<u64> {
    let va = base.checked_add(u64::from(rva))?;
    va.checked_add(8)?;
    Some(va)
}

/// Computes the checked VA of the `index`-th 8-byte IAT slot starting at `base + iat_rva`.
fn checked_iat_slot(base: u64, iat_rva: u32, index: usize) -> Option<u64> {
    if iat_rva == 0 {
        return None;
    }
    let offset = u64::try_from(index).ok()?.checked_mul(8)?;
    let va = base
        .checked_add(u64::from(iat_rva))?
        .checked_add(offset)?;
    va.checked_add(8)?;
    Some(va)
}

/// Computes the checked half-open byte range `[start, end)` of an IAT containing `symbol_count`
/// entries plus its 8-byte null terminator.
fn checked_iat_range(base: u64, iat_rva: u32, symbol_count: usize) -> Option<(u64, u64)> {
    if iat_rva == 0 {
        return None;
    }
    let start = base.checked_add(u64::from(iat_rva))?;
    let slots = u64::try_from(symbol_count).ok()?.checked_add(1)?;
    let byte_len = slots.checked_mul(8)?;
    let end = start.checked_add(byte_len)?;
    Some((start, end))
}

/// Every by-name import (normal and delay-load) listed in [`NO_RETURN_APIS`], keyed by its IAT
/// slot VA (`image_base + iat_rva + 8 * index`). Slots whose VA calculation overflows `u64` are
/// skipped. Sorted by `(slot, label)` and deduplicated.
pub fn no_return_imports(pe: &Pe64) -> Vec<NoReturnImport> {
    if pe.imports.is_empty() && pe.delay_imports.is_empty() {
        return Vec::new();
    }
    let base = pe.optional.image_base;
    let normal = pe
        .imports
        .iter()
        .map(|m| (&m.dll_name, m.iat_rva, &m.symbols));
    let delayed = pe
        .delay_imports
        .iter()
        .map(|m| (&m.dll_name, m.iat_rva, &m.symbols));
    let mut out = Vec::new();
    for (dll, iat_rva, symbols) in normal.chain(delayed) {
        if iat_rva == 0 {
            continue;
        }
        for (i, symbol) in symbols.iter().enumerate() {
            if let ImportedSymbol::ByName { name, .. } = symbol
                && is_no_return(dll, name)
            {
                let Some(slot) = checked_iat_slot(base, iat_rva, i) else {
                    continue;
                };
                out.push(NoReturnImport {
                    slot,
                    label: format!("{dll}!{name}"),
                });
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Entries of discovered functions that only forward to a no-return target: a single
/// `jmp [rip+slot]` with `slot` in `no_return`, or a single direct `jmp` to an entry in
/// `no_return`. Excludes entries already in `no_return`.
pub fn no_return_thunks(analysis: &Analysis, no_return: &BTreeSet<u64>) -> BTreeSet<u64> {
    if analysis.functions.is_empty() || no_return.is_empty() {
        return BTreeSet::new();
    }
    analysis
        .functions
        .values()
        .filter(|f| !no_return.contains(&f.entry))
        .filter(|f| {
            let Some(block) = f.entry_block() else {
                return false;
            };
            block.instructions.len() == 1
                && match block.successors.as_slice() {
                    [Edge::IndirectJump { slot: Some(slot) }] => no_return.contains(slot),
                    [Edge::TailCall(target)] => {
                        block.terminator == Terminator::Jump && no_return.contains(target)
                    }
                    _ => false,
                }
        })
        .map(|f| f.entry)
        .collect()
}

// ---------------------------------------------------------------------------------------------
// 2. Code-pointer entry points
// ---------------------------------------------------------------------------------------------

/// Instructions decoded when checking that a candidate without `.pdata` looks like code.
/// Large enough for long straight-line leaf functions (e.g. static initializers that fill tables
/// with thousands of `lea`/`mov` pairs before their `ret`); zero runs and other data that decodes
/// without ever reaching a control-flow instruction still fail at this bound.
pub const PROBE_INSTRUCTIONS: usize = 65536;

/// Upper bound on discovery rounds of the fixpoint (each round adds at least one entry or
/// no-return thunk, so real inputs converge in a handful).
pub const MAX_ROUNDS: usize = 64;

/// Where a code-pointer candidate came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PointerSource {
    /// RIP-relative `lea` of a code address (`RelocKind::RipAddress`, region code).
    Lea,
    /// `DIR64` base relocation in a non-executable section whose value points into code.
    BaseReloc,
}

impl PointerSource {
    pub const ALL: [Self; 2] = [Self::Lea, Self::BaseReloc];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lea => "lea",
            Self::BaseReloc => "base_reloc",
        }
    }
}

/// Outcome for one candidate. The acceptance rule, in order:
/// 1. Already a discovered function entry: [`Self::KnownEntry`] (or [`Self::Accepted`] if this
///    refinement added it in an earlier round).
/// 2. Inside a `.pdata` function chunk but not at its primary function start: rejected
///    ([`Self::PdataInterior`]). `.pdata` starts are exact and already seeded, so with `.pdata`
///    nothing inside a described function is ever guessed.
/// 3. Inside bytes of a resolved jump table or its index table: [`Self::JumpTableData`].
/// 4. Strictly inside an instruction of a block discovered so far, or of the probe of a candidate
///    accepted earlier in the same round (ascending order): [`Self::InsideInstruction`].
/// 5. Otherwise (no `.pdata`, or a leaf function `.pdata` does not describe): a linear decode
///    from the candidate must succeed and reach a control-flow instruction within
///    [`PROBE_INSTRUCTIONS`] instructions without running off the section
///    ([`Self::NotDecodable`] otherwise); it must not start with `00 00` (zero padding) or with a
///    trap/`hlt` ([`Self::Implausible`]). Passing candidates are [`Self::Accepted`].
///
/// A candidate at an instruction boundary of an already discovered block (outside `.pdata`
/// functions) is accepted too: the code is already decoded and the block is split there, the
/// fallthrough becoming a tail call, which preserves the guest's behaviour exactly. In real
/// binaries these are leaf functions without unwind data that follow a call discovery could not
/// prove non-returning, so the previous function's decoding ran into them.
///
/// Accepted candidates become discovery entry points; discovery is re-run (new code adds new
/// `lea`s and no-return thunks) until nothing changes. The rule is unsound only where data that
/// decodes plausibly is referenced like code (e.g. an unresolved jump table inside `.text`
/// outside every `.pdata` function): such a candidate becomes a function that traps or computes
/// garbage if ever called, which a correct program never does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PointerStatus {
    KnownEntry,
    Accepted,
    PdataInterior,
    JumpTableData,
    InsideInstruction,
    NotDecodable,
    Implausible,
}

impl PointerStatus {
    pub const ALL: [Self; 7] = [
        Self::KnownEntry,
        Self::Accepted,
        Self::PdataInterior,
        Self::JumpTableData,
        Self::InsideInstruction,
        Self::NotDecodable,
        Self::Implausible,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::KnownEntry => "known_entry",
            Self::Accepted => "accepted",
            Self::PdataInterior => "pdata_interior",
            Self::JumpTableData => "jump_table_data",
            Self::InsideInstruction => "inside_instruction",
            Self::NotDecodable => "not_decodable",
            Self::Implausible => "implausible",
        }
    }
}

/// One address-taken code address and what became of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodePointer {
    pub va: u64,
    /// Sources that proposed it (at least one).
    pub sources: BTreeSet<PointerSource>,
    pub status: PointerStatus,
    /// Whether it is a discovered (hence lifted and registered) function entry.
    pub lifted: bool,
}

/// Result of the discovery fixpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodePointerOutcome {
    /// Discovery runs performed (1 when no refinement added anything).
    pub rounds: usize,
    /// Whether the fixpoint stopped at [`MAX_ROUNDS`] instead of converging.
    pub round_limit_hit: bool,
    /// Function entries found to be thunks to no-return targets, ascending.
    pub no_return_thunks: Vec<u64>,
    /// Every candidate of the final round, ascending by VA.
    pub pointers: Vec<CodePointer>,
}

impl CodePointerOutcome {
    /// Number of candidates of `source` with `status`.
    pub fn count(&self, source: PointerSource, status: PointerStatus) -> usize {
        self.pointers
            .iter()
            .filter(|p| p.status == status && p.sources.contains(&source))
            .count()
    }

    /// Candidates that are not registered function entries (dispatcher gaps).
    pub fn not_lifted(&self) -> impl Iterator<Item = &CodePointer> {
        self.pointers.iter().filter(|p| !p.lifted)
    }

    /// Entry points this refinement added.
    pub fn added_entries(&self) -> impl Iterator<Item = u64> + '_ {
        self.pointers
            .iter()
            .filter(|p| p.status == PointerStatus::Accepted)
            .map(|p| p.va)
    }
}

/// Code addresses taken by RIP-relative `lea` anywhere in the discovered code.
pub fn lea_pointers(analysis: &Analysis) -> BTreeSet<u64> {
    if analysis.relocations.is_empty() {
        return BTreeSet::new();
    }
    analysis
        .relocations
        .entries()
        .iter()
        .filter(|r| r.kind == RelocKind::RipAddress && r.region == Region::Code)
        .map(|r| r.target)
        .collect()
}

/// Values of `DIR64` base relocations stored in non-executable sections that point into
/// executable code (pointers as stored at the preferred image base).
pub fn base_reloc_pointers(image: &CodeImage<'_>, guest: &GuestImage<'_>) -> BTreeSet<u64> {
    if image.sections().is_empty() || guest.sections.is_empty() || guest.relocations.is_empty() {
        return BTreeSet::new();
    }
    let mut out = BTreeSet::new();
    for reloc in &guest.relocations {
        if reloc.kind != BaseRelocKind::Dir64 {
            continue;
        }
        let rva = u64::from(reloc.rva);
        let Some(section) = guest.section_at(rva) else {
            continue;
        };
        if section.protection.execute || section.is_headers || section.bytes_omitted {
            continue;
        }
        let Some(offset) = rva
            .checked_sub(u64::from(section.rva))
            .and_then(|o| usize::try_from(o).ok())
        else {
            continue;
        };
        let Some(end) = offset.checked_add(8) else {
            continue;
        };
        let Some(bytes) = section.data.get(offset..end) else {
            continue; // zero-fill: a null pointer
        };
        let value = u64::from_le_bytes(bytes.try_into().expect("8 bytes"));
        if image.region_of(value) == Region::Code {
            out.insert(value);
        }
    }
    out
}

/// Blocks of all functions keyed by start (shared blocks once).
struct BlockIndex<'a> {
    blocks: BTreeMap<u64, &'a BasicBlock>,
}

/// Preceding blocks examined for one containing an address (16 covers the full 15-byte maximum
/// x86-64 instruction width plus the containing block when overlapping streams interleave).
const CONTAINING_BLOCK_SCAN: usize = 16;

impl<'a> BlockIndex<'a> {
    fn new(analysis: &'a Analysis) -> Self {
        let mut blocks = BTreeMap::new();
        for function in analysis.functions.values() {
            for (&start, block) in &function.blocks {
                blocks.entry(start).or_insert(block);
            }
        }
        Self { blocks }
    }

    /// `Some(true)` at an instruction boundary of a block, `Some(false)` inside an instruction.
    fn locate(&self, va: u64) -> Option<bool> {
        if self.blocks.is_empty() {
            return None;
        }
        let mut inside = false;
        for block in self
            .blocks
            .range(..=va)
            .rev()
            .take(CONTAINING_BLOCK_SCAN)
            .map(|(_, b)| *b)
        {
            if block.contains(va) {
                if block.instructions.iter().any(|i| i.ip() == va) {
                    return Some(true);
                }
                inside = true;
            }
        }
        inside.then_some(false)
    }
}

/// Inserts half-open range `[start, end)` into a canonical set of disjoint, non-adjacent
/// intervals so lookups are exact and independent of insertion order even when tables overlap or
/// nest.
fn insert_merged_range(ranges: &mut BTreeMap<u64, u64>, start: u64, end: u64) {
    if start >= end {
        return;
    }
    let mut merged_start = start;
    let mut merged_end = end;
    if let Some((&prev_start, &prev_end)) = ranges.range(..=start).next_back()
        && prev_end >= start
    {
        merged_start = prev_start;
        merged_end = merged_end.max(prev_end);
    }
    let absorbed: Vec<(u64, u64)> = ranges
        .range(merged_start..=merged_end)
        .map(|(&s, &e)| (s, e))
        .collect();
    for (s, e) in absorbed {
        merged_end = merged_end.max(e);
        ranges.remove(&s);
    }
    ranges.insert(merged_start, merged_end);
}

/// Checked exclusive end VA for a jump table with `entries` elements of `entry_size` bytes.
fn checked_table_end(start: u64, entries: usize, entry_size: u64) -> Option<u64> {
    let count = u64::try_from(entries).ok()?;
    let byte_len = count.checked_mul(entry_size)?;
    start.checked_add(byte_len)
}

/// Byte ranges of resolved jump tables and their index tables, normalised into disjoint sorted
/// half-open intervals. Tables whose extent overflows `u64` are ignored safely.
fn jump_table_ranges(analysis: &Analysis) -> BTreeMap<u64, u64> {
    if analysis.jump_tables.is_empty() {
        return BTreeMap::new();
    }
    let mut ranges = BTreeMap::new();
    for table in analysis.jump_tables.values() {
        if let Some(end) = checked_table_end(table.table, table.targets.len(), 4) {
            insert_merged_range(&mut ranges, table.table, end);
        }
        if let Some(index) = table.index
            && let Some(end) = index.table.checked_add(index.len)
        {
            insert_merged_range(&mut ranges, index.table, end);
        }
    }
    ranges
}

fn in_ranges(ranges: &BTreeMap<u64, u64>, va: u64) -> bool {
    ranges
        .range(..=va)
        .next_back()
        .is_some_and(|(_, &end)| va < end)
}

/// Decoded span of one linearly probed candidate: instruction starts (ascending) and exclusive end
/// VA of the final instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ProbeSpan {
    starts: Vec<u64>,
    end: u64,
}

/// Linear decode used for candidates without `.pdata`: instruction starts and exclusive end VA on
/// success.
fn probe(image: &CodeImage<'_>, va: u64) -> Result<ProbeSpan, PointerStatus> {
    let bytes = image
        .code_bytes(va)
        .map_err(|_| PointerStatus::NotDecodable)?;
    if bytes.is_empty() {
        return Err(PointerStatus::NotDecodable);
    }
    if bytes.starts_with(&[0, 0]) {
        return Err(PointerStatus::Implausible);
    }
    let mut decoder = Decoder::with_ip(64, bytes, va, DecoderOptions::NONE);
    let mut starts = Vec::new();
    for i in 0..PROBE_INSTRUCTIONS {
        if !decoder.can_decode() {
            return Err(PointerStatus::NotDecodable);
        }
        let instr = decoder.decode();
        if instr.is_invalid() {
            return Err(PointerStatus::NotDecodable);
        }
        let ip = instr.ip();
        let len = instr.len() as u64;
        if len == 0 {
            return Err(PointerStatus::NotDecodable);
        }
        let Some(next_ip) = ip.checked_add(len) else {
            return Err(PointerStatus::NotDecodable);
        };
        starts.push(ip);
        let traps = matches!(
            instr.flow_control(),
            FlowControl::Interrupt | FlowControl::Exception
        ) || instr.mnemonic() == Mnemonic::Hlt;
        if i == 0 && traps {
            return Err(PointerStatus::Implausible);
        }
        if instr.flow_control() != FlowControl::Next {
            return Ok(ProbeSpan {
                starts,
                end: next_ip,
            });
        }
    }
    Err(PointerStatus::Implausible)
}

/// Whether `va` falls strictly inside an instruction of a candidate accepted earlier in this
/// round. Matching an instruction boundary returns `false` (same boundary-wins rule as
/// [`BlockIndex::locate`]).
fn inside_accepted_probe(accepted: &BTreeMap<u64, ProbeSpan>, va: u64) -> bool {
    let mut inside = false;
    for (_, span) in accepted.range(..va).rev().take(CONTAINING_BLOCK_SCAN) {
        if va < span.end {
            if span.starts.binary_search(&va).is_ok() {
                return false;
            }
            inside = true;
        }
    }
    inside
}

/// Classifies every candidate against one discovery result (see [`PointerStatus`]).
fn classify(
    image: &CodeImage<'_>,
    analysis: &Analysis,
    candidates: &BTreeMap<u64, BTreeSet<PointerSource>>,
    added: &BTreeSet<u64>,
) -> Vec<CodePointer> {
    if candidates.is_empty() {
        return Vec::new();
    }
    let blocks = BlockIndex::new(analysis);
    let tables = jump_table_ranges(analysis);
    // Probed spans of candidates accepted earlier in this round, keyed by start VA.
    let mut accepted: BTreeMap<u64, ProbeSpan> = BTreeMap::new();
    let mut out = Vec::with_capacity(candidates.len());
    for (&va, sources) in candidates {
        let lifted = analysis.functions.contains_key(&va);
        let status = if lifted {
            if added.contains(&va) {
                PointerStatus::Accepted
            } else {
                PointerStatus::KnownEntry
            }
        } else if image
            .function_table
            .range_at(va)
            .is_some_and(|r| !(r.start == va && r.function == va))
        {
            PointerStatus::PdataInterior
        } else if in_ranges(&tables, va) {
            PointerStatus::JumpTableData
        } else if blocks.locate(va) == Some(false) || inside_accepted_probe(&accepted, va) {
            PointerStatus::InsideInstruction
        } else {
            match probe(image, va) {
                Ok(span) => {
                    accepted.insert(va, span);
                    PointerStatus::Accepted
                }
                Err(status) => status,
            }
        };
        out.push(CodePointer {
            va,
            sources: sources.clone(),
            status,
            lifted,
        });
    }
    out
}

/// Runs discovery on `image` to a fixpoint with the code-pointer and no-return-thunk
/// refinements, bounded by [`MAX_ROUNDS`]. Accepted candidates are appended to
/// `image.entry_points` and thunks to `image.no_return` (both kept sorted and deduplicated),
/// so `image` reflects the final seeds.
pub fn discover_refined(
    image: &mut CodeImage<'_>,
    guest: &GuestImage<'_>,
) -> (Analysis, CodePointerOutcome) {
    discover_refined_with_max_rounds(image, guest, MAX_ROUNDS)
}

/// Bounded variant of [`discover_refined`] with an explicit maximum round budget.
///
/// Canonicalises `image.entry_points` and `image.no_return` before round 1 so output is
/// independent of seed ordering or duplicates, short-circuits empty graphs without running
/// discovery, and enforces strict per-round growth so stagnant iterations terminate immediately.
pub fn discover_refined_with_max_rounds(
    image: &mut CodeImage<'_>,
    guest: &GuestImage<'_>,
    max_rounds: usize,
) -> (Analysis, CodePointerOutcome) {
    let mut known_entries: BTreeSet<u64> = image.entry_points.iter().copied().collect();
    let mut known_no_return: BTreeSet<u64> = image.no_return.iter().copied().collect();
    image.entry_points = known_entries.iter().copied().collect();
    image.no_return = known_no_return.iter().copied().collect();

    let relocated = base_reloc_pointers(image, guest);
    let empty_graph =
        known_entries.is_empty() && image.eh.resume_targets.is_empty() && relocated.is_empty();
    if empty_graph || max_rounds == 0 {
        let analysis = Analysis {
            functions: BTreeMap::new(),
            relocations: RelocationMap::default(),
            jump_tables: BTreeMap::new(),
            unresolved_jumps: BTreeMap::new(),
            pointer_slots: BTreeMap::new(),
        };
        let outcome = CodePointerOutcome {
            rounds: 0,
            round_limit_hit: !empty_graph,
            no_return_thunks: Vec::new(),
            pointers: Vec::new(),
        };
        return (analysis, outcome);
    }

    let mut added = BTreeSet::new();
    let mut thunks = BTreeSet::new();
    for rounds in 1..=max_rounds {
        let analysis = discover(image);
        let mut candidates: BTreeMap<u64, BTreeSet<PointerSource>> = BTreeMap::new();
        for va in lea_pointers(&analysis) {
            candidates.entry(va).or_default().insert(PointerSource::Lea);
        }
        for &va in &relocated {
            candidates
                .entry(va)
                .or_default()
                .insert(PointerSource::BaseReloc);
        }
        let pointers = classify(image, &analysis, &candidates, &added);
        let new_entries: BTreeSet<u64> = pointers
            .iter()
            .filter(|p| {
                p.status == PointerStatus::Accepted
                    && !p.lifted
                    && !known_entries.contains(&p.va)
                    && !added.contains(&p.va)
            })
            .map(|p| p.va)
            .collect();
        let new_thunks = no_return_thunks(&analysis, &known_no_return);
        let converged = new_entries.is_empty() && new_thunks.is_empty();
        if converged || rounds == max_rounds {
            let outcome = CodePointerOutcome {
                rounds,
                round_limit_hit: !converged,
                no_return_thunks: thunks.into_iter().collect(),
                pointers,
            };
            return (analysis, outcome);
        }
        added.extend(new_entries.iter().copied());
        known_entries.extend(new_entries.iter().copied());
        image.entry_points = known_entries.iter().copied().collect();
        thunks.extend(new_thunks.iter().copied());
        known_no_return.extend(new_thunks.iter().copied());
        image.no_return = known_no_return.iter().copied().collect();
    }
    unreachable!("max_rounds >= 1 returns on the final iteration");
}

// ---------------------------------------------------------------------------------------------
// 4. Import slot consistency
// ---------------------------------------------------------------------------------------------

/// Cross-check of slot reads in discovered code against the import bridge plan.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SlotCheck {
    /// Distinct slots read by `call [rip+x]` / `jmp [rip+x]`.
    pub slot_reads: usize,
    /// Of those, slots that are IAT entries of the plan.
    pub iat_reads: usize,
    /// Slot reads outside every import address table (function-pointer variables, vtables in
    /// data); dispatched by value at run time, ascending.
    pub non_iat_reads: Vec<u64>,
    /// Addresses read by code that fall inside an import address table but not on a symbol's
    /// slot (misaligned or terminator reads): a real inconsistency, ascending.
    pub misaligned: Vec<u64>,
    /// Imports whose slot code references (any reference kind) but that have no shim:
    /// `(dll, label, reason code)` in plan order.
    pub referenced_unresolved: Vec<(String, String, &'static str)>,
    /// Imports whose slot no discovered code references: `(label, bound)` in plan order.
    pub unreferenced: Vec<(String, bool)>,
}

/// Cross-checks the slots discovery found against the IAT slots of `plan`.
pub fn check_import_slots(pe: &Pe64, plan: &BridgePlan, analysis: &Analysis) -> SlotCheck {
    if plan.imports.is_empty() && analysis.relocations.is_empty() {
        return SlotCheck::default();
    }
    let base = pe.optional.image_base;
    let slot_vas: BTreeSet<u64> = plan
        .imports
        .iter()
        .filter_map(|i| i.slot_rva)
        .filter_map(|rva| checked_slot_va(base, rva))
        .collect();
    let mut iat_ranges = BTreeMap::new();
    let tables = pe
        .imports
        .iter()
        .map(|m| (m.iat_rva, m.symbols.len()))
        .chain(
            pe.delay_imports
                .iter()
                .map(|m| (m.iat_rva, m.symbols.len())),
        );
    for (iat_rva, count) in tables {
        if let Some((start, end)) = checked_iat_range(base, iat_rva, count) {
            insert_merged_range(&mut iat_ranges, start, end);
        }
    }

    let mut referenced = BTreeSet::new();
    let mut slot_reads = BTreeSet::new();
    let mut misaligned = BTreeSet::new();
    for reloc in analysis.relocations.entries() {
        let target = reloc.target;
        if matches!(reloc.kind, RelocKind::CallSlot | RelocKind::JumpSlot) {
            slot_reads.insert(target);
        }
        if slot_vas.contains(&target) {
            referenced.insert(target);
        } else if in_ranges(&iat_ranges, target) {
            misaligned.insert(target);
        }
    }

    let mut check = SlotCheck {
        slot_reads: slot_reads.len(),
        iat_reads: slot_reads.intersection(&slot_vas).count(),
        non_iat_reads: slot_reads
            .iter()
            .copied()
            .filter(|va| !slot_vas.contains(va) && !misaligned.contains(va))
            .collect(),
        misaligned: misaligned.into_iter().collect(),
        ..SlotCheck::default()
    };
    for import in &plan.imports {
        let used = import
            .slot_rva
            .and_then(|rva| checked_slot_va(base, rva))
            .is_some_and(|va| referenced.contains(&va));
        match (&import.resolution, used) {
            // Bound to another guest module by a program build: as good as bound.
            (_, true) if import.guest.is_some() => {}
            (_, false) if import.guest.is_some() => {
                check.unreferenced.push((import.label(), true));
            }
            (Resolution::Unresolved(why), true) => {
                check
                    .referenced_unresolved
                    .push((import.dll.clone(), import.label(), why.code()))
            }
            (_, false) => check
                .unreferenced
                .push((import.label(), import.bound_shim().is_some())),
            _ => {}
        }
    }
    check
}

#[cfg(test)]
mod probe_tests {
    use super::*;
    use crate::cfg::{JumpTable, JumpTableIndex};
    use crate::code_image::CodeSection;
    use crate::image::{BaseRelocation, ExecutableBytes, ImageSection, Protection};
    use crate::pe::{CoffHeader, DelayImportModule, ImportModule, OptionalHeader64};
    use crate::reloc::Relocation;
    use crate::stubs::{ImportBinding, Unresolved};

    const BASE: u64 = 0x1_4000_0000;
    const TEXT: u64 = BASE + 0x1000;
    const RDATA_RVA: u32 = 0x2000;

    fn probe_code(code: &[u8]) -> Result<Vec<u64>, PointerStatus> {
        let image = CodeImage::new(BASE, vec![CodeSection::new(TEXT, code, true)], vec![]);
        probe(&image, TEXT).map(|span| span.starts)
    }

    fn empty_guest<'a>(image_base: u64) -> GuestImage<'a> {
        GuestImage {
            image_base,
            size_of_image: 0,
            section_alignment: 0x1000,
            entry_point_rva: 0,
            headers_mapped: false,
            executable_bytes: ExecutableBytes::Keep,
            relocs_stripped: false,
            sections: Vec::new(),
            relocations: Vec::new(),
        }
    }

    fn empty_pe(image_base: u64) -> Pe64 {
        Pe64 {
            file_size: 0,
            e_lfanew: 0,
            coff: CoffHeader {
                machine: 0x8664,
                number_of_sections: 0,
                time_date_stamp: 0,
                pointer_to_symbol_table: 0,
                number_of_symbols: 0,
                size_of_optional_header: 0,
                characteristics: 0,
            },
            optional: OptionalHeader64 {
                magic: 0x020B,
                major_linker_version: 0,
                minor_linker_version: 0,
                size_of_code: 0,
                size_of_initialized_data: 0,
                size_of_uninitialized_data: 0,
                entry_point_rva: 0,
                base_of_code: 0,
                image_base,
                section_alignment: 0x1000,
                file_alignment: 0x200,
                size_of_image: 0x1_0000,
                size_of_headers: 0x200,
                checksum: 0,
                subsystem: 0,
                dll_characteristics: 0,
                number_of_rva_and_sizes: 0,
                data_directories: Vec::new(),
            },
            sections: Vec::new(),
            imports: Vec::new(),
            delay_imports: Vec::new(),
            exports: None,
            tls: None,
        }
    }

    /// Appends `lea rax, [rip + disp32]` targeting `target_va`, followed by `ret` (8 bytes total).
    fn emit_lea_ret(code: &mut Vec<u8>, base_va: u64, target_va: u64) {
        let next_ip = base_va + code.len() as u64 + 7;
        let disp = i32::try_from(target_va as i64 - next_ip as i64).expect("rip disp32");
        code.extend_from_slice(&[0x48, 0x8d, 0x05]);
        code.extend_from_slice(&disp.to_le_bytes());
        code.push(0xc3);
    }

    /// Appends `jmp qword ptr [rip + disp32]` targeting `slot_va` (6 bytes total).
    fn emit_jmp_slot(code: &mut Vec<u8>, base_va: u64, slot_va: u64) {
        let next_ip = base_va + code.len() as u64 + 6;
        let disp = i32::try_from(slot_va as i64 - next_ip as i64).expect("rip disp32");
        code.extend_from_slice(&[0xff, 0x25]);
        code.extend_from_slice(&disp.to_le_bytes());
    }

    /// Appends `jmp rel32` targeting `target_va` (5 bytes total).
    fn emit_jmp_rel(code: &mut Vec<u8>, base_va: u64, target_va: u64) {
        let next_ip = base_va + code.len() as u64 + 5;
        let disp = i32::try_from(target_va as i64 - next_ip as i64).expect("jmp rel32");
        code.push(0xe9);
        code.extend_from_slice(&disp.to_le_bytes());
    }

    #[test]
    fn long_straight_line_leaf_function_is_code() {
        // 6000 x `mov eax, 1` (5 bytes each), then `ret`: a static initializer shape.
        let mut code: Vec<u8> = std::iter::repeat_n([0xb8, 1, 0, 0, 0], 6000).flatten().collect();
        code.push(0xc3);
        let starts = probe_code(&code).expect("plausible");
        assert_eq!(starts.len(), 6001);
    }

    #[test]
    fn data_without_control_flow_is_still_implausible() {
        // A `nop` sled longer than the probe never reaches a control-flow instruction.
        let code = vec![0x90u8; PROBE_INSTRUCTIONS + 10];
        assert_eq!(probe_code(&code), Err(PointerStatus::Implausible));
        assert_eq!(probe_code(&[0, 0, 0xc3]), Err(PointerStatus::Implausible));
        assert_eq!(probe_code(&[0xcc, 0xc3]), Err(PointerStatus::Implausible));
    }

    #[test]
    fn discover_refined_is_permutation_deterministic() {
        // Synthetic image with:
        // - f0 (entry): lea f1; ret
        // - f1 (discovered via lea): lea f2; ret
        // - f2 (discovered via lea): ret
        // - vt0 (discovered via DIR64 reloc): lea vt1; ret
        // - vt1 (discovered via lea in vt0): ret
        // - thunk1 (entry): jmp [rip+exit_slot]
        // - thunk2 (entry): jmp thunk1
        let exit_slot = BASE + u64::from(RDATA_RVA) + 0x20;
        let f0 = TEXT;
        let f1 = TEXT + 0x08;
        let f2 = TEXT + 0x10;
        let vt0 = TEXT + 0x18;
        let vt1 = TEXT + 0x20;
        let thunk1 = TEXT + 0x28;
        let thunk2 = TEXT + 0x30;

        let mut code = Vec::new();
        emit_lea_ret(&mut code, TEXT, f1); // 0x00..0x08
        emit_lea_ret(&mut code, TEXT, f2); // 0x08..0x10
        code.extend_from_slice(&[0xc3, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90]); // 0x10..0x18
        emit_lea_ret(&mut code, TEXT, vt1); // 0x18..0x20
        code.extend_from_slice(&[0xc3, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90]); // 0x20..0x28
        emit_jmp_slot(&mut code, TEXT, exit_slot); // 0x28..0x2e
        code.extend_from_slice(&[0x90, 0x90]); // 0x2e..0x30
        emit_jmp_rel(&mut code, TEXT, thunk1); // 0x30..0x35
        code.extend_from_slice(&[0x90, 0x90, 0x90]); // 0x35..0x38

        let mut rdata = Vec::new();
        rdata.extend_from_slice(&vt0.to_le_bytes()); // +0x00
        rdata.extend_from_slice(&f2.to_le_bytes()); // +0x08
        rdata.extend_from_slice(&0u64.to_le_bytes()); // +0x10
        rdata.extend_from_slice(&0u64.to_le_bytes()); // +0x18
        rdata.extend_from_slice(&0u64.to_le_bytes()); // +0x20 (exit_slot)

        let run_case = |entries: Vec<u64>,
                        no_ret: Vec<u64>,
                        relocs: Vec<BaseRelocation>|
         -> (Analysis, CodePointerOutcome, Vec<u64>, Vec<u64>) {
            let mut image = CodeImage::new(
                BASE,
                vec![
                    CodeSection::new(TEXT, &code, true),
                    CodeSection::new(BASE + u64::from(RDATA_RVA), &rdata, false),
                ],
                entries,
            );
            image.no_return = no_ret;
            let guest = GuestImage {
                image_base: BASE,
                size_of_image: 0x4000,
                section_alignment: 0x1000,
                entry_point_rva: 0x1000,
                headers_mapped: false,
                executable_bytes: ExecutableBytes::Keep,
                relocs_stripped: false,
                sections: vec![ImageSection {
                    name: ".rdata".to_string(),
                    rva: RDATA_RVA,
                    virtual_size: rdata.len() as u32,
                    protection: Protection::READ_ONLY,
                    is_headers: false,
                    bytes_omitted: false,
                    characteristics: 0,
                    data: &rdata,
                }],
                relocations: relocs,
            };
            let (analysis, outcome) = discover_refined(&mut image, &guest);
            (analysis, outcome, image.entry_points, image.no_return)
        };

        let reloc_a = BaseRelocation {
            rva: RDATA_RVA,
            kind: BaseRelocKind::Dir64,
        };
        let reloc_b = BaseRelocation {
            rva: RDATA_RVA + 8,
            kind: BaseRelocKind::Dir64,
        };

        let baseline = run_case(
            vec![f0, thunk1, thunk2],
            vec![exit_slot],
            vec![reloc_a, reloc_b],
        );
        let permuted = run_case(
            vec![thunk2, f0, thunk1, f0, thunk2],
            vec![exit_slot, exit_slot],
            vec![reloc_b, reloc_a, reloc_b],
        );

        assert_eq!(baseline, permuted);
        let (analysis, outcome, final_entries, final_no_return) = baseline;
        assert!(!outcome.round_limit_hit);
        assert_eq!(outcome.no_return_thunks, vec![thunk1, thunk2]);
        assert_eq!(final_entries, vec![f0, f1, f2, vt0, vt1, thunk1, thunk2]);
        assert_eq!(final_no_return, vec![thunk1, thunk2, exit_slot]);
        assert_eq!(analysis.functions.len(), 7);
    }

    #[test]
    fn bounded_fixed_point_respects_explicit_and_default_max_rounds() {
        // Build a 66-function lea chain: f_0 -> f_1 -> ... -> f_65 (each 8 bytes).
        // Each round discovers one more link, so converging requires 66 rounds (> MAX_ROUNDS = 64).
        let chain_len = MAX_ROUNDS + 2;
        let mut code = Vec::with_capacity(chain_len * 8);
        for i in 0..chain_len - 1 {
            let next_va = TEXT + ((i + 1) * 8) as u64;
            emit_lea_ret(&mut code, TEXT, next_va);
        }
        code.extend_from_slice(&[0xc3, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90]);
        let guest = empty_guest(BASE);

        // Zero budget on non-empty graph returns immediately with round_limit_hit = true.
        let mut img_zero = CodeImage::new(BASE, vec![CodeSection::new(TEXT, &code, true)], vec![TEXT]);
        let (analysis_zero, out_zero) = discover_refined_with_max_rounds(&mut img_zero, &guest, 0);
        assert_eq!(out_zero.rounds, 0);
        assert!(out_zero.round_limit_hit);
        assert!(analysis_zero.functions.is_empty());

        // Tight budget (3 rounds) stops cleanly at 3 rounds with round_limit_hit = true.
        let mut img_small = CodeImage::new(BASE, vec![CodeSection::new(TEXT, &code, true)], vec![TEXT]);
        let (analysis_small, out_small) = discover_refined_with_max_rounds(&mut img_small, &guest, 3);
        assert_eq!(out_small.rounds, 3);
        assert!(out_small.round_limit_hit);
        assert_eq!(analysis_small.functions.len(), 3);

        // Default MAX_ROUNDS (64) stops cleanly on the 66-link chain without spinning forever.
        let mut img_default =
            CodeImage::new(BASE, vec![CodeSection::new(TEXT, &code, true)], vec![TEXT]);
        let (analysis_default, out_default) = discover_refined(&mut img_default, &guest);
        assert_eq!(out_default.rounds, MAX_ROUNDS);
        assert!(out_default.round_limit_hit);
        assert_eq!(analysis_default.functions.len(), MAX_ROUNDS);

        // Generous budget converges cleanly.
        let mut img_full = CodeImage::new(BASE, vec![CodeSection::new(TEXT, &code, true)], vec![TEXT]);
        let (analysis_full, out_full) =
            discover_refined_with_max_rounds(&mut img_full, &guest, chain_len + 1);
        assert_eq!(out_full.rounds, chain_len);
        assert!(!out_full.round_limit_hit);
        assert_eq!(analysis_full.functions.len(), chain_len);
    }

    #[test]
    fn empty_graph_and_same_round_final_instruction_boundary_guards() {
        // 1. Completely empty CodeImage and GuestImage short-circuit with 0 rounds.
        let mut empty_img = CodeImage::new(BASE, Vec::new(), Vec::new());
        let guest = empty_guest(BASE);
        let (analysis, outcome) = discover_refined(&mut empty_img, &guest);
        assert_eq!(outcome.rounds, 0);
        assert!(!outcome.round_limit_hit);
        assert!(outcome.pointers.is_empty());
        assert!(outcome.no_return_thunks.is_empty());
        assert!(analysis.functions.is_empty());
        assert!(lea_pointers(&analysis).is_empty());
        assert!(no_return_thunks(&analysis, &BTreeSet::from([TEXT])).is_empty());
        assert!(base_reloc_pointers(&empty_img, &guest).is_empty());
        assert!(jump_table_ranges(&analysis).is_empty());

        let pe = empty_pe(BASE);
        assert!(no_return_imports(&pe).is_empty());
        assert_eq!(
            check_import_slots(&pe, &BridgePlan { imports: Vec::new() }, &analysis),
            SlotCheck::default()
        );

        // 2. Candidate strictly inside the final (and only) multi-byte instruction of an earlier
        //    accepted probe in the same round must be classified as InsideInstruction.
        //    `jmp TEXT` is 5 bytes: e9 fb ff ff ff at TEXT..TEXT+5.
        let mut code = Vec::new();
        emit_jmp_rel(&mut code, TEXT, TEXT);
        code.push(0xc3);
        let img = CodeImage::new(BASE, vec![CodeSection::new(TEXT, &code, true)], Vec::new());
        let mut candidates = BTreeMap::new();
        candidates.insert(TEXT, BTreeSet::from([PointerSource::BaseReloc]));
        candidates.insert(TEXT + 1, BTreeSet::from([PointerSource::BaseReloc]));
        candidates.insert(TEXT + 5, BTreeSet::from([PointerSource::BaseReloc]));

        let classified = classify(&img, &analysis, &candidates, &BTreeSet::new());
        assert_eq!(classified.len(), 3);
        assert_eq!(classified[0].status, PointerStatus::Accepted);
        assert_eq!(classified[1].status, PointerStatus::InsideInstruction);
        assert_eq!(classified[2].status, PointerStatus::Accepted);
    }

    #[test]
    fn checked_va_overflow_guards_never_panic() {
        // 1. no_return_imports skips slots that overflow u64 while keeping valid ones.
        let mut pe = empty_pe(u64::MAX - 0x10);
        pe.imports.push(ImportModule {
            dll_name: "KERNEL32.dll".to_string(),
            iat_rva: 0x08, // slot 0 at u64::MAX - 0x08 (valid, end = u64::MAX), slot 1 overflows
            symbols: vec![
                ImportedSymbol::ByName {
                    hint: 0,
                    name: "ExitProcess".to_string(),
                },
                ImportedSymbol::ByName {
                    hint: 0,
                    name: "ExitThread".to_string(),
                },
            ],
        });
        pe.delay_imports.push(DelayImportModule {
            dll_name: "ntdll.dll".to_string(),
            attributes: 1,
            uses_rva: true,
            iat_rva: 0x20, // base + 0x20 overflows u64 immediately
            symbols: vec![ImportedSymbol::ByName {
                hint: 0,
                name: "RtlExitUserProcess".to_string(),
            }],
        });
        let found = no_return_imports(&pe);
        assert_eq!(
            found,
            vec![NoReturnImport {
                slot: u64::MAX - 0x08,
                label: "KERNEL32.dll!ExitProcess".to_string(),
            }]
        );

        // 2. check_import_slots handles overflowing image_base + rva cleanly.
        let plan = BridgePlan {
            imports: vec![
                ImportBinding {
                    index: 0,
                    dll: "KERNEL32.dll".to_string(),
                    symbol: ImportedSymbol::ByName {
                        hint: 0,
                        name: "ExitProcess".to_string(),
                    },
                    slot_rva: Some(0x08),
                    delay_load: false,
                    resolution: Resolution::Unresolved(Unresolved::UnknownDll),
                    guest: None,
                },
                ImportBinding {
                    index: 1,
                    dll: "KERNEL32.dll".to_string(),
                    symbol: ImportedSymbol::ByName {
                        hint: 0,
                        name: "ExitThread".to_string(),
                    },
                    slot_rva: Some(0x40), // overflows u64
                    delay_load: false,
                    resolution: Resolution::Unresolved(Unresolved::UnknownDll),
                    guest: None,
                },
            ],
        };
        let analysis = Analysis {
            functions: BTreeMap::new(),
            relocations: RelocationMap::from_entries(vec![Relocation {
                site: TEXT,
                kind: RelocKind::CallSlot,
                target: u64::MAX - 0x08,
                region: Region::Data,
            }]),
            jump_tables: BTreeMap::new(),
            unresolved_jumps: BTreeMap::new(),
            pointer_slots: BTreeMap::new(),
        };
        let slot_check = check_import_slots(&pe, &plan, &analysis);
        assert_eq!(slot_check.slot_reads, 1);
        assert_eq!(slot_check.iat_reads, 1);
        assert!(slot_check.misaligned.is_empty());
        assert_eq!(slot_check.referenced_unresolved.len(), 1);
        assert_eq!(slot_check.unreferenced.len(), 1);

        // 3. probe rejects instructions whose end VA overflows u64, while accepting a 1-byte `ret`
        //    at u64::MAX - 1 (exclusive end == u64::MAX).
        let ret_bytes = [0xc3u8];
        let ret_img = CodeImage::new(
            u64::MAX - 1,
            vec![CodeSection::new(u64::MAX - 1, &ret_bytes, true)],
            Vec::new(),
        );
        assert_eq!(
            probe(&ret_img, u64::MAX - 1),
            Ok(ProbeSpan {
                starts: vec![u64::MAX - 1],
                end: u64::MAX,
            })
        );

        let jmp_bytes = [0xebu8, 0xfe]; // 2-byte `jmp -2` at u64::MAX - 1 overflows u64
        let wrap_img = CodeImage::new(
            u64::MAX - 1,
            vec![CodeSection::new(u64::MAX - 1, &jmp_bytes, true)],
            Vec::new(),
        );
        assert_eq!(probe(&wrap_img, u64::MAX - 1), Err(PointerStatus::NotDecodable));

        // 4. base_reloc_pointers ignores truncated 8-byte slots and bytes_omitted sections.
        let short_data = [0u8; 4];
        let text_bytes = [0xc3u8];
        let code_img = CodeImage::new(BASE, vec![CodeSection::new(TEXT, &text_bytes, true)], vec![]);
        let guest = GuestImage {
            image_base: BASE,
            size_of_image: 0x4000,
            section_alignment: 0x1000,
            entry_point_rva: 0,
            headers_mapped: false,
            executable_bytes: ExecutableBytes::Keep,
            relocs_stripped: false,
            sections: vec![ImageSection {
                name: ".rdata".to_string(),
                rva: RDATA_RVA,
                virtual_size: 0x100,
                protection: Protection::READ_ONLY,
                is_headers: false,
                bytes_omitted: false,
                characteristics: 0,
                data: &short_data,
            }],
            relocations: vec![BaseRelocation {
                rva: RDATA_RVA,
                kind: BaseRelocKind::Dir64,
            }],
        };
        assert!(base_reloc_pointers(&code_img, &guest).is_empty());
    }

    #[test]
    fn overlapping_and_nested_ranges_are_canonical_and_order_independent() {
        // Outer jump table [0x1000, 0x1020) contains nested index table [0x1008, 0x100c), plus
        // an overflowing table near u64::MAX that must be ignored without panicking.
        let mut analysis_a = Analysis {
            functions: BTreeMap::new(),
            relocations: RelocationMap::default(),
            jump_tables: BTreeMap::new(),
            unresolved_jumps: BTreeMap::new(),
            pointer_slots: BTreeMap::new(),
        };
        let outer = JumpTable {
            site: TEXT,
            load_site: TEXT,
            table: 0x1000,
            targets: vec![TEXT; 8], // 32 bytes -> [0x1000, 0x1020)
            index: Some(JumpTableIndex {
                load_site: TEXT,
                table: 0x1008,
                len: 4, // nested [0x1008, 0x100c)
            }),
            inferred: false,
        };
        let overlapping = JumpTable {
            site: TEXT + 0x10,
            load_site: TEXT + 0x10,
            table: 0x1018,
            targets: vec![TEXT; 4], // [0x1018, 0x1028), extends outer
            index: Some(JumpTableIndex {
                load_site: TEXT + 0x10,
                table: u64::MAX - 2,
                len: 8, // overflows u64 -> safely skipped
            }),
            inferred: false,
        };
        let overflowing = JumpTable {
            site: TEXT + 0x20,
            load_site: TEXT + 0x20,
            table: u64::MAX - 2,
            targets: vec![TEXT; 2], // overflows u64 -> safely skipped
            index: None,
            inferred: false,
        };

        analysis_a.jump_tables.insert(outer.site, outer.clone());
        analysis_a
            .jump_tables
            .insert(overlapping.site, overlapping.clone());
        analysis_a
            .jump_tables
            .insert(overflowing.site, overflowing.clone());

        let mut analysis_b = Analysis {
            functions: BTreeMap::new(),
            relocations: RelocationMap::default(),
            jump_tables: BTreeMap::new(),
            unresolved_jumps: BTreeMap::new(),
            pointer_slots: BTreeMap::new(),
        };
        // Insert with swapped keys to change iteration order.
        let mut outer_swapped = outer;
        outer_swapped.site = TEXT + 0x30;
        let mut overlap_swapped = overlapping;
        overlap_swapped.site = TEXT;
        analysis_b.jump_tables.insert(overlap_swapped.site, overlap_swapped);
        analysis_b.jump_tables.insert(outer_swapped.site, outer_swapped);
        analysis_b.jump_tables.insert(overflowing.site, overflowing);

        let ranges_a = jump_table_ranges(&analysis_a);
        let ranges_b = jump_table_ranges(&analysis_b);
        assert_eq!(ranges_a, ranges_b);
        assert_eq!(ranges_a, BTreeMap::from([(0x1000, 0x1028)]));
        // Address after the nested index table [0x1008, 0x100c) is still inside the merged range.
        assert!(in_ranges(&ranges_a, 0x1000));
        assert!(in_ranges(&ranges_a, 0x1010));
        assert!(in_ranges(&ranges_a, 0x1027));
        assert!(!in_ranges(&ranges_a, 0x1028));
        assert!(!in_ranges(&ranges_a, u64::MAX));
    }

    #[test]
    fn no_return_imports_and_slot_checks_are_permutation_deterministic() {
        let mut pe_a = empty_pe(BASE);
        let mod_k32 = ImportModule {
            dll_name: "KERNEL32.dll".to_string(),
            iat_rva: RDATA_RVA,
            symbols: vec![
                ImportedSymbol::ByName {
                    hint: 0,
                    name: "ExitProcess".to_string(),
                },
                ImportedSymbol::ByName {
                    hint: 0,
                    name: "GetTickCount64".to_string(),
                },
            ],
        };
        let mod_crt = ImportModule {
            dll_name: "ucrtbase.dll".to_string(),
            iat_rva: RDATA_RVA + 0x20,
            symbols: vec![ImportedSymbol::ByName {
                hint: 0,
                name: "abort".to_string(),
            }],
        };
        let delay_nt = DelayImportModule {
            dll_name: "ntdll.dll".to_string(),
            attributes: 1,
            uses_rva: true,
            iat_rva: RDATA_RVA + 0x08, // overlaps KERNEL32 IAT intentionally to verify merged range determinism
            symbols: vec![ImportedSymbol::ByName {
                hint: 0,
                name: "RtlExitUserProcess".to_string(),
            }],
        };
        pe_a.imports = vec![mod_k32.clone(), mod_crt.clone()];
        pe_a.delay_imports = vec![delay_nt.clone()];

        let mut pe_b = empty_pe(BASE);
        pe_b.imports = vec![mod_crt, mod_k32];
        pe_b.delay_imports = vec![delay_nt];

        assert_eq!(no_return_imports(&pe_a), no_return_imports(&pe_b));
        assert_eq!(no_return_imports(&pe_a).len(), 3);

        let plan = BridgePlan {
            imports: vec![ImportBinding {
                index: 0,
                dll: "KERNEL32.dll".to_string(),
                symbol: ImportedSymbol::ByName {
                    hint: 0,
                    name: "ExitProcess".to_string(),
                },
                slot_rva: Some(RDATA_RVA),
                delay_load: false,
                resolution: Resolution::Bound {
                    shim: "pg6_win32_exit_process".to_string(),
                },
                guest: None,
            }],
        };
        let analysis = Analysis {
            functions: BTreeMap::new(),
            relocations: RelocationMap::from_entries(vec![
                Relocation {
                    site: TEXT,
                    kind: RelocKind::CallSlot,
                    target: BASE + u64::from(RDATA_RVA),
                    region: Region::Data,
                },
                Relocation {
                    site: TEXT + 6,
                    kind: RelocKind::CallSlot,
                    target: BASE + u64::from(RDATA_RVA) + 0x14, // misaligned inside merged IAT
                    region: Region::Data,
                },
            ]),
            jump_tables: BTreeMap::new(),
            unresolved_jumps: BTreeMap::new(),
            pointer_slots: BTreeMap::new(),
        };
        let check_a = check_import_slots(&pe_a, &plan, &analysis);
        let check_b = check_import_slots(&pe_b, &plan, &analysis);
        assert_eq!(check_a, check_b);
        assert_eq!(check_a.iat_reads, 1);
        assert_eq!(check_a.misaligned, vec![BASE + u64::from(RDATA_RVA) + 0x14]);
    }
}
