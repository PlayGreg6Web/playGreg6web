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
use crate::reloc::RelocKind;
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

/// Every by-name import (normal and delay-load) listed in [`NO_RETURN_APIS`], keyed by its IAT
/// slot VA (`image_base + iat_rva + 8 * index`). Sorted by slot.
pub fn no_return_imports(pe: &Pe64) -> Vec<NoReturnImport> {
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
                out.push(NoReturnImport {
                    slot: base + u64::from(iat_rva) + 8 * i as u64,
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
/// Large enough for long straight-line leaf functions (e.g. static initializers that fill a table
/// with hundreds of `lea`/`mov` pairs before their `ret`); zero runs and other data that decodes
/// without ever reaching a control-flow instruction still fail at this bound.
pub const PROBE_INSTRUCTIONS: usize = 4096;

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
    let mut out = BTreeSet::new();
    for reloc in &guest.relocations {
        if reloc.kind != BaseRelocKind::Dir64 {
            continue;
        }
        let rva = u64::from(reloc.rva);
        let Some(section) = guest.section_at(rva) else {
            continue;
        };
        if section.protection.execute || section.is_headers {
            continue;
        }
        let offset = (rva - u64::from(section.rva)) as usize;
        let Some(bytes) = section.data.get(offset..offset + 8) else {
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

/// Preceding blocks examined for one containing an address (overlapping streams interleave).
const CONTAINING_BLOCK_SCAN: usize = 8;

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

/// Byte ranges of resolved jump tables and their index tables.
fn jump_table_ranges(analysis: &Analysis) -> BTreeMap<u64, u64> {
    let mut ranges = BTreeMap::new();
    for table in analysis.jump_tables.values() {
        let end = table.table + 4 * table.targets.len() as u64;
        ranges.insert(table.table, end);
        if let Some(index) = table.index {
            ranges.insert(index.table, index.table + index.len);
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

/// Linear decode used for candidates without `.pdata`: instruction starts on success.
fn probe(image: &CodeImage<'_>, va: u64) -> Result<Vec<u64>, PointerStatus> {
    let bytes = image
        .code_bytes(va)
        .map_err(|_| PointerStatus::NotDecodable)?;
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
        starts.push(instr.ip());
        let traps = matches!(
            instr.flow_control(),
            FlowControl::Interrupt | FlowControl::Exception
        ) || instr.mnemonic() == Mnemonic::Hlt;
        if i == 0 && traps {
            return Err(PointerStatus::Implausible);
        }
        if instr.flow_control() != FlowControl::Next {
            return Ok(starts);
        }
    }
    Err(PointerStatus::Implausible)
}

/// Classifies every candidate against one discovery result (see [`PointerStatus`]).
fn classify(
    image: &CodeImage<'_>,
    analysis: &Analysis,
    candidates: &BTreeMap<u64, BTreeSet<PointerSource>>,
    added: &BTreeSet<u64>,
) -> Vec<CodePointer> {
    let blocks = BlockIndex::new(analysis);
    let tables = jump_table_ranges(analysis);
    // Instruction starts of candidates accepted earlier in this round, keyed by start.
    let mut accepted: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
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
        } else if blocks.locate(va) == Some(false)
            || accepted.range(..va).next_back().is_some_and(|(_, starts)| {
                starts.last().is_some_and(|&last| va <= last) && !starts.contains(&va)
            })
        {
            PointerStatus::InsideInstruction
        } else {
            match probe(image, va) {
                Ok(starts) => {
                    accepted.insert(va, starts);
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
/// refinements. Accepted candidates are appended to `image.entry_points` and thunks to
/// `image.no_return` (both kept sorted and deduplicated), so `image` reflects the final seeds.
pub fn discover_refined(
    image: &mut CodeImage<'_>,
    guest: &GuestImage<'_>,
) -> (Analysis, CodePointerOutcome) {
    let relocated = base_reloc_pointers(image, guest);
    let mut added = BTreeSet::new();
    let mut thunks = BTreeSet::new();
    let mut rounds = 0;
    loop {
        rounds += 1;
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
        let new_entries: Vec<u64> = pointers
            .iter()
            .filter(|p| p.status == PointerStatus::Accepted && !p.lifted)
            .map(|p| p.va)
            .collect();
        let no_return: BTreeSet<u64> = image.no_return.iter().copied().collect();
        let new_thunks = no_return_thunks(&analysis, &no_return);
        let converged = new_entries.is_empty() && new_thunks.is_empty();
        if converged || rounds >= MAX_ROUNDS {
            let outcome = CodePointerOutcome {
                rounds,
                round_limit_hit: !converged,
                no_return_thunks: thunks.into_iter().collect(),
                pointers,
            };
            return (analysis, outcome);
        }
        added.extend(new_entries.iter().copied());
        image.entry_points.extend(new_entries);
        image.entry_points.sort_unstable();
        image.entry_points.dedup();
        thunks.extend(new_thunks.iter().copied());
        image.no_return.extend(new_thunks);
        image.no_return.sort_unstable();
        image.no_return.dedup();
    }
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
    let base = pe.optional.image_base;
    let slot_vas: BTreeSet<u64> = plan
        .imports
        .iter()
        .filter_map(|i| i.slot_rva)
        .map(|rva| base + u64::from(rva))
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
        if iat_rva != 0 {
            let start = base + u64::from(iat_rva);
            iat_ranges.insert(start, start + 8 * (count as u64 + 1));
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
            .is_some_and(|rva| referenced.contains(&(base + u64::from(rva))));
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
    use crate::code_image::CodeSection;

    const TEXT: u64 = 0x1_4000_1000;

    fn probe_code(code: &[u8]) -> Result<Vec<u64>, PointerStatus> {
        let image = CodeImage::new(0x1_4000_0000, vec![CodeSection::new(TEXT, code, true)], vec![]);
        probe(&image, TEXT)
    }

    #[test]
    fn long_straight_line_leaf_function_is_code() {
        // 1500 x `mov eax, 1` (5 bytes each), then `ret`: a static initializer shape.
        let mut code: Vec<u8> = std::iter::repeat_n([0xb8, 1, 0, 0, 0], 1500).flatten().collect();
        code.push(0xc3);
        let starts = probe_code(&code).expect("plausible");
        assert_eq!(starts.len(), 1501);
    }

    #[test]
    fn data_without_control_flow_is_still_implausible() {
        // A `nop` sled longer than the probe never reaches a control-flow instruction.
        let code = vec![0x90u8; PROBE_INSTRUCTIONS + 10];
        assert_eq!(probe_code(&code), Err(PointerStatus::Implausible));
        assert_eq!(probe_code(&[0, 0, 0xc3]), Err(PointerStatus::Implausible));
        assert_eq!(probe_code(&[0xcc, 0xc3]), Err(PointerStatus::Implausible));
    }
}
