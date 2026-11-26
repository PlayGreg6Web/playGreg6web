//! Survey of indirect `jmp`s the lifter leaves on run-time dispatch.
//!
//! [`crate::discover`] resolves MSVC switch jump tables; every other indirect `jmp` without an
//! import slot stays in [`Analysis::unresolved_jumps`]. This module re-reads each such jump's
//! local instruction window (its block plus the chain of sole predecessors inside the function)
//! and groups the jumps by pattern: the lifter's reason, the role (switch / tail call), the jump
//! form, the index scaling, the guard shape and where the table or pointer lives. It also walks
//! the direct call graph from the entry point and TLS callbacks so the groups can be ranked by
//! what a boot reaches. Output is addresses, mnemonic-level shapes, section names and counts only.
//!
//! Indirect calls (vtables, callbacks) are not followed, so "reachable" is a lower bound.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::Write as _;

use iced_x86::{ConditionCode, FlowControl, Instruction, Mnemonic, OpKind, Register};

use crate::cfg::{Edge, Function, Terminator, UnresolvedJump};
use crate::code_image::CodeImage;
use crate::discover::Analysis;
use crate::pe::Pe64;

/// Predecessor blocks followed backwards from the jump.
const MAX_WINDOW_BLOCKS: usize = 6;
const MAX_DEPTH: u32 = 8;

/// Section map used to label table locations.
#[derive(Debug, Clone, Default)]
pub struct Layout {
    pub image_base: u64,
    /// `(name, rva, size, characteristics)`.
    pub sections: Vec<(String, u64, u64, u32)>,
}

impl Layout {
    pub fn from_pe(pe: &Pe64) -> Self {
        Self {
            image_base: pe.optional.image_base,
            sections: pe
                .sections
                .iter()
                .map(|s| {
                    (
                        s.name.clone(),
                        u64::from(s.virtual_address),
                        u64::from(s.virtual_extent()),
                        s.characteristics,
                    )
                })
                .collect(),
        }
    }
}

/// One grouping key. Fields are short descriptive labels.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Pattern {
    pub reason: &'static str,
    pub role: &'static str,
    pub form: String,
    pub scale: String,
    pub guard: String,
    pub location: String,
}

impl Pattern {
    pub fn label(&self) -> String {
        format!(
            "{} | {} | form={} scale={} guard={} loc={}",
            self.reason, self.role, self.form, self.scale, self.guard, self.location
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Group {
    pub sites: Vec<u64>,
    pub reachable: usize,
    /// Reachable sites by function entry.
    pub functions: BTreeMap<u64, usize>,
    /// Shallowest boot call-graph depth among reachable sites in this group.
    pub min_depth: Option<u32>,
    /// Count of jump sites in this group whose sole-predecessor window terminated because a
    /// block had `> 1` intra-function predecessors (a CFG merge point).
    pub merge_predecessor_sites: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Location {
    pub function: u64,
    pub depth: u32,
    pub callers: usize,
    pub sites: usize,
    pub patterns: BTreeSet<usize>,
}

/// Priority tier for unresolved indirect jump patterns, ordered highest-priority first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PriorityTier {
    /// Boot-reachable switch with a guard, merge-predecessor window, or unknown/unbounded table.
    CriticalBootSwitch,
    /// Boot-reachable switch without the critical indicators above.
    BootSwitch,
    /// Boot-reachable non-switch (`tail_call` or `computed`).
    BootOther,
    /// Unreachable switch pattern.
    UnreachableSwitch,
    /// Unreachable non-switch pattern.
    UnreachableOther,
}

impl PriorityTier {
    /// Stable snake_case identifier for reports and JSON output.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CriticalBootSwitch => "critical_boot_switch",
            Self::BootSwitch => "boot_switch",
            Self::BootOther => "boot_other",
            Self::UnreachableSwitch => "unreachable_switch",
            Self::UnreachableOther => "unreachable_other",
        }
    }

    /// Classifies a `(Pattern, Group)` pair into its priority tier.
    pub fn classify(pattern: &Pattern, group: &Group) -> Self {
        if pattern.role == "switch" && group.reachable > 0 {
            if pattern.guard != "none"
                || group.merge_predecessor_sites > 0
                || pattern.reason == "table_base_unknown"
                || pattern.reason == "table_unbounded"
            {
                Self::CriticalBootSwitch
            } else {
                Self::BootSwitch
            }
        } else if group.reachable > 0 {
            Self::BootOther
        } else if pattern.role == "switch" {
            Self::UnreachableSwitch
        } else {
            Self::UnreachableOther
        }
    }
}

/// Deterministic numeric priority score (strictly monotonic across [`PriorityTier`] bands).
pub fn priority_score(pattern: &Pattern, group: &Group) -> u64 {
    let tier = PriorityTier::classify(pattern, group);
    let tier_base: u64 = match tier {
        PriorityTier::CriticalBootSwitch => 4_000_000_000,
        PriorityTier::BootSwitch => 3_000_000_000,
        PriorityTier::BootOther => 2_000_000_000,
        PriorityTier::UnreachableSwitch => 1_000_000_000,
        PriorityTier::UnreachableOther => 0,
    };
    let reachable_bonus = (group.reachable as u64).min(1_000_000) * 500;
    let depth_bonus = group
        .min_depth
        .map_or(0, |d| 256u64.saturating_sub(u64::from(d).min(256)) * 1_000);
    let merge_bonus = (group.merge_predecessor_sites as u64).min(100_000) * 100;
    let guard_bonus = if pattern.role == "switch" && pattern.guard != "none" {
        50_000
    } else {
        0
    };
    let total_bonus = (group.sites.len() as u64).min(100_000_000);
    tier_base + reachable_bonus + depth_bonus + merge_bonus + guard_bonus + total_bonus
}

/// One pattern ranked by boot-first priority.
#[derive(Debug, Clone)]
pub struct PrioritizedPattern<'a> {
    /// 1-based index into [`JumpSurvey::patterns`].
    pub pattern_index: usize,
    pub tier: PriorityTier,
    pub score: u64,
    pub pattern: &'a Pattern,
    pub group: &'a Group,
}

#[derive(Debug, Clone, Default)]
pub struct JumpSurvey {
    pub total_unresolved: usize,
    pub reachable_unresolved: usize,
    pub functions_total: usize,
    pub functions_reachable: usize,
    /// Unresolved jumps whose pointer slot is statically initialised (target seeded as an entry).
    pub pointer_slot_jumps: usize,
    pub by_reason: BTreeMap<&'static str, usize>,
    pub by_reason_reachable: BTreeMap<&'static str, usize>,
    /// Role (`"switch"`, `"tail_call"`, `"computed"`) -> `(total, reachable)`.
    pub by_role: BTreeMap<&'static str, (usize, usize)>,
    /// Ranked: reachable desc, total desc, label.
    pub patterns: Vec<(Pattern, Group)>,
    pub locations: Vec<Location>,
}

impl JumpSurvey {
    /// Returns patterns ranked by boot-first priority:
    /// `(tier asc, reachable desc, min_depth.unwrap_or(u32::MAX) asc, total desc, pattern asc)`.
    pub fn prioritized_patterns(&self) -> Vec<PrioritizedPattern<'_>> {
        let mut out: Vec<PrioritizedPattern<'_>> = self
            .patterns
            .iter()
            .enumerate()
            .map(|(i, (pattern, group))| PrioritizedPattern {
                pattern_index: i + 1,
                tier: PriorityTier::classify(pattern, group),
                score: priority_score(pattern, group),
                pattern,
                group,
            })
            .collect();
        out.sort_by(|a, b| {
            a.tier
                .cmp(&b.tier)
                .then_with(|| b.group.reachable.cmp(&a.group.reachable))
                .then_with(|| {
                    a.group
                        .min_depth
                        .unwrap_or(u32::MAX)
                        .cmp(&b.group.min_depth.unwrap_or(u32::MAX))
                })
                .then_with(|| b.group.sites.len().cmp(&a.group.sites.len()))
                .then_with(|| a.pattern.cmp(b.pattern))
        });
        out
    }

    /// Returns `(total_switches, reachable_switches, merge_predecessor_switches)`.
    pub fn switch_counts(&self) -> (usize, usize, usize) {
        let mut total = 0usize;
        let mut reachable = 0usize;
        let mut merge_preds = 0usize;
        for (p, g) in &self.patterns {
            if p.role == "switch" {
                total += g.sites.len();
                reachable += g.reachable;
                merge_preds += g.merge_predecessor_sites;
            }
        }
        (total, reachable, merge_preds)
    }

    /// Returns `(1-based pattern_index, pattern, group)` for every pattern matching `role`.
    pub fn filtered_by_role(&self, role: &str) -> Vec<(usize, &Pattern, &Group)> {
        self.patterns
            .iter()
            .enumerate()
            .filter(|(_, (p, _))| p.role == role)
            .map(|(i, (p, g))| (i + 1, p, g))
            .collect()
    }

    /// Returns a derived survey restricted strictly to patterns with `role == "switch"`.
    pub fn switches_only(&self) -> Self {
        let mut patterns: Vec<(Pattern, Group)> = Vec::new();
        let mut by_reason: BTreeMap<&'static str, usize> = BTreeMap::new();
        let mut by_reason_reachable: BTreeMap<&'static str, usize> = BTreeMap::new();
        let mut total_unresolved = 0usize;
        let mut reachable_unresolved = 0usize;

        for (p, g) in &self.patterns {
            if p.role != "switch" {
                continue;
            }
            total_unresolved += g.sites.len();
            reachable_unresolved += g.reachable;
            *by_reason.entry(p.reason).or_default() += g.sites.len();
            if g.reachable > 0 {
                *by_reason_reachable.entry(p.reason).or_default() += g.reachable;
            }
            patterns.push((p.clone(), g.clone()));
        }

        let mut by_role: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
        if total_unresolved > 0 {
            by_role.insert("switch", (total_unresolved, reachable_unresolved));
        }

        let mut func_info: BTreeMap<u64, (usize, BTreeSet<usize>)> = BTreeMap::new();
        for (pi, (_, g)) in patterns.iter().enumerate() {
            for (&func, &count) in &g.functions {
                let entry = func_info.entry(func).or_default();
                entry.0 += count;
                entry.1.insert(pi + 1);
            }
        }

        let locations: Vec<Location> = self
            .locations
            .iter()
            .filter_map(|loc| {
                let (sites, pats) = func_info.get(&loc.function)?;
                Some(Location {
                    function: loc.function,
                    depth: loc.depth,
                    callers: loc.callers,
                    sites: *sites,
                    patterns: pats.clone(),
                })
            })
            .collect();

        Self {
            total_unresolved,
            reachable_unresolved,
            functions_total: self.functions_total,
            functions_reachable: self.functions_reachable,
            pointer_slot_jumps: self.pointer_slot_jumps,
            by_reason,
            by_reason_reachable,
            by_role,
            patterns,
            locations,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Def {
    Load {
        scale: u8,
        loc: String,
    },
    RelOffset {
        scale: u8,
        loc: String,
    },
    Anchor(u64),
    /// Return value (or clobber) of a `call` in the window: the target comes from a function.
    CallResult,
    LiveIn,
    Other,
}

/// Win64 volatile GPRs: a `call` returns in RAX and may clobber the others.
pub(crate) fn is_win64_volatile(reg: Register) -> bool {
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

/// Image bytes and function entries, used to read the leading entries of a constant table.
#[derive(Clone, Copy)]
pub(crate) struct TableProbe<'a> {
    pub(crate) image: &'a CodeImage<'a>,
    pub(crate) functions: &'a BTreeMap<u64, Function>,
}

/// Leading entries read to tell a function-pointer table from a switch table.
const PROBE_ENTRIES: u64 = 4;

pub(crate) struct Ctx<'a> {
    pub(crate) layout: &'a Layout,
    pub(crate) probe: Option<TableProbe<'a>>,
}

impl Ctx<'_> {
    fn section_label(&self, va: u64) -> String {
        let base = self.layout.image_base;
        if va == base {
            return "image_base".into();
        }
        let Some(rva) = va.checked_sub(base) else {
            return "outside".into();
        };
        for (name, start, size, ch) in &self.layout.sections {
            if rva >= *start && rva < start + size {
                let name: String = name
                    .trim_matches('\0')
                    .chars()
                    .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
                    .collect();
                let x = if ch & 0x2000_0000 != 0 { "x" } else { "" };
                let w = if ch & 0x8000_0000 != 0 { "w" } else { "" };
                return format!("{name}({x}{w})");
            }
        }
        "outside".into()
    }

    /// Where a memory operand's base comes from.
    fn mem_loc(&self, ins: &[&Instruction], at: usize, i: &Instruction, depth: u32) -> String {
        if i.is_ip_rel_memory_operand() {
            return self.section_label(i.ip_rel_memory_address());
        }
        let b = i.memory_base();
        if b == Register::None {
            return if i.memory_index() == Register::None {
                "abs".into()
            } else {
                "abs_disp".into()
            };
        }
        if depth == 0 {
            return "deep".into();
        }
        match self.describe(ins, at, b, depth - 1) {
            Def::Anchor(a) => format!("anchor:{}", self.section_label(a)),
            Def::Load { .. } => "loaded_pointer".into(),
            Def::RelOffset { .. } => "computed_pointer".into(),
            Def::CallResult => "call_result".into(),
            Def::LiveIn => "live_in_reg".into(),
            Def::Other => "other_reg".into(),
        }
    }

    /// Definition of `reg` just before `ins[upto]`.
    fn describe(&self, ins: &[&Instruction], upto: usize, reg: Register, depth: u32) -> Def {
        let reg = reg.full_register();
        for idx in (0..upto).rev() {
            let i = ins[idx];
            if matches!(
                i.flow_control(),
                FlowControl::Call | FlowControl::IndirectCall
            ) {
                if is_win64_volatile(reg) {
                    return Def::CallResult;
                }
                continue;
            }
            if matches!(
                i.mnemonic(),
                Mnemonic::Cmp | Mnemonic::Test | Mnemonic::Push | Mnemonic::Jmp | Mnemonic::Bt
            ) || i.op_count() == 0
                || i.op0_kind() != OpKind::Register
                || i.op0_register().full_register() != reg
            {
                continue;
            }
            let loadish = matches!(
                i.mnemonic(),
                Mnemonic::Mov | Mnemonic::Movsxd | Mnemonic::Movzx | Mnemonic::Movsx
            );
            return match i.mnemonic() {
                _ if loadish && i.op1_kind() == OpKind::Memory => Def::Load {
                    scale: if i.memory_index() == Register::None {
                        0
                    } else {
                        i.memory_index_scale() as u8
                    },
                    loc: self.mem_loc(ins, idx, i, depth),
                },
                _ if loadish && i.op1_kind() == OpKind::Register && depth > 0 => {
                    self.describe(ins, idx, i.op1_register(), depth - 1)
                }
                Mnemonic::Lea if i.is_ip_rel_memory_operand() => {
                    Def::Anchor(i.ip_rel_memory_address())
                }
                Mnemonic::Lea | Mnemonic::Add if depth > 0 => {
                    let (a, b) = if i.mnemonic() == Mnemonic::Lea {
                        (i.memory_base(), i.memory_index())
                    } else if i.op1_kind() == OpKind::Register {
                        (reg, i.op1_register())
                    } else {
                        return Def::Other;
                    };
                    let da = self.describe(ins, idx, a, depth - 1);
                    let db = self.describe(ins, idx, b, depth - 1);
                    for d in [da, db] {
                        match d {
                            Def::Load { scale, loc } | Def::RelOffset { scale, loc }
                                if scale > 0 =>
                            {
                                return Def::RelOffset { scale, loc };
                            }
                            _ => {}
                        }
                    }
                    Def::Other
                }
                _ => Def::Other,
            };
        }
        Def::LiveIn
    }
}

/// What a scale-8 table read through a constant base holds, when that is not a switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PointerTable {
    /// Read-only table whose leading non-null entries are all discovered function entries: a
    /// tail call through a function-pointer table (handler array). Its targets are functions,
    /// which the run-time dispatcher already knows; a switch's targets are case labels.
    Functions,
    /// Table in a writable section: pointers stored at run time (callback arrays), never a
    /// compiled switch table.
    Writable,
    /// Base loaded from object or global memory (`mov r, [obj]` then `[r+i*8+d]`): a vtable
    /// slot or pointer array picked by an index. A compiled switch's table base is a constant
    /// (`lea`), a non-volatile register holding one, or at most a stack spill of it.
    Loaded,
}

impl Ctx<'_> {
    /// Constant base VA of the 8-byte table read by the jump ending `ins` (`jmp [base+i*8+d]` or
    /// `mov r, [base+i*8+d]; jmp r`, across register copies), when the base is a `lea` of a
    /// fixed address.
    fn table_base(&self, ins: &[&Instruction], at: usize, jmp: &Instruction) -> Option<u64> {
        let (pos, read) = if jmp.op0_kind() == OpKind::Memory {
            (at, jmp)
        } else {
            self.load_of(ins, at, jmp.op0_register())?
        };
        if read.memory_size().size() != 8
            || read.memory_index() == Register::None
            || read.memory_index_scale() != 8
            || read.memory_base() == Register::None
            || read.is_ip_rel_memory_operand()
        {
            return None;
        }
        match self.describe(ins, pos, read.memory_base(), MAX_DEPTH) {
            Def::Anchor(a) => Some(a.wrapping_add(read.memory_displacement64())),
            _ => None,
        }
    }

    /// Whether the base register of the 8-byte table read by the jump ending `ins` was loaded
    /// from non-stack memory (an object field, a vtable pointer or a global) in the window.
    fn base_loaded_from_memory(&self, ins: &[&Instruction], at: usize, jmp: &Instruction) -> bool {
        let (pos, read) = if jmp.op0_kind() == OpKind::Memory {
            (at, jmp)
        } else {
            match self.load_of(ins, at, jmp.op0_register()) {
                Some(found) => found,
                None => return false,
            }
        };
        if read.memory_index() == Register::None || read.memory_index_scale() != 8 {
            return false;
        }
        let base = read.memory_base();
        if base == Register::None || base == Register::RIP {
            return false;
        }
        match self.load_of(ins, pos, base) {
            Some((_, load)) => !matches!(
                load.memory_base().full_register(),
                Register::RSP | Register::RBP
            ),
            None => false,
        }
    }

    /// The `mov reg, [mem]` defining `reg` before `ins[upto]`, following `mov` register copies.
    fn load_of<'i>(
        &self,
        ins: &[&'i Instruction],
        upto: usize,
        reg: Register,
    ) -> Option<(usize, &'i Instruction)> {
        let mut reg = reg.full_register();
        let mut before = upto;
        for _ in 0..MAX_DEPTH {
            let idx = (0..before).rev().find(|&idx| {
                let i = ins[idx];
                matches!(
                    i.flow_control(),
                    FlowControl::Call | FlowControl::IndirectCall
                ) || (i.op_count() >= 1
                    && i.op0_kind() == OpKind::Register
                    && i.op0_register().full_register() == reg
                    && !matches!(
                        i.mnemonic(),
                        Mnemonic::Cmp | Mnemonic::Test | Mnemonic::Bt | Mnemonic::Push
                    ))
            })?;
            let i = ins[idx];
            if i.mnemonic() != Mnemonic::Mov || i.op_count() != 2 {
                return None;
            }
            match i.op1_kind() {
                OpKind::Register if i.op1_register().is_gpr64() => {
                    reg = i.op1_register().full_register();
                    before = idx;
                }
                OpKind::Memory if i.op0_register().is_gpr64() => return Some((idx, i)),
                _ => return None,
            }
        }
        None
    }

    /// Classifies the table read by the jump ending `ins` as a function-pointer or run-time data
    /// table; `None` keeps it a switch candidate (no probe, unknown base, or any leading entry
    /// that is neither null nor a discovered function entry).
    fn pointer_table(
        &self,
        ins: &[&Instruction],
        at: usize,
        jmp: &Instruction,
    ) -> Option<PointerTable> {
        if self.base_loaded_from_memory(ins, at, jmp) {
            return Some(PointerTable::Loaded);
        }
        let probe = self.probe?;
        let base = self.table_base(ins, at, jmp)?;
        let rva = base.checked_sub(self.layout.image_base)?;
        let (_, _, _, ch) = self
            .layout
            .sections
            .iter()
            .find(|(_, start, size, _)| rva >= *start && rva < start + size)?;
        if ch & 0x8000_0000 != 0 {
            return Some(PointerTable::Writable);
        }
        let mut functions = 0;
        for n in 0..PROBE_ENTRIES {
            let Some(bytes) = base
                .checked_add(n * 8)
                .and_then(|va| probe.image.bytes_at(va, 8))
            else {
                break;
            };
            let v = u64::from_le_bytes(bytes.try_into().ok()?);
            if v == 0 {
                continue;
            }
            if !probe.functions.contains_key(&v) {
                return None;
            }
            functions += 1;
        }
        (functions > 0).then_some(PointerTable::Functions)
    }
}

fn guard_shape(ins: &[&Instruction]) -> String {
    let n = ins.len();
    for j in (0..n.saturating_sub(1)).rev() {
        let i = ins[j];
        let cc = i.condition_code();
        if cc == ConditionCode::None
            || !matches!(
                i.mnemonic(),
                Mnemonic::Ja
                    | Mnemonic::Jae
                    | Mnemonic::Jb
                    | Mnemonic::Jbe
                    | Mnemonic::Jg
                    | Mnemonic::Jge
                    | Mnemonic::Jl
                    | Mnemonic::Jle
                    | Mnemonic::Je
                    | Mnemonic::Jne
                    | Mnemonic::Js
                    | Mnemonic::Jns
            )
        {
            continue;
        }
        let kind = match cc {
            ConditionCode::a | ConditionCode::ae | ConditionCode::b | ConditionCode::be => {
                "unsigned"
            }
            ConditionCode::g | ConditionCode::ge | ConditionCode::l | ConditionCode::le => "signed",
            ConditionCode::e | ConditionCode::ne => "eq",
            _ => "sign",
        };
        let mut cmp = "flags";
        let mut biased = false;
        for k in (0..j).rev().take(4) {
            let c = ins[k];
            match c.mnemonic() {
                Mnemonic::Cmp | Mnemonic::Test => {
                    cmp = match (c.mnemonic(), c.op1_kind()) {
                        (Mnemonic::Test, _) => "test",
                        (_, OpKind::Memory) => "cmp_mem",
                        (_, OpKind::Register) => "cmp_reg",
                        _ => "cmp_imm",
                    };
                    break;
                }
                Mnemonic::Sub | Mnemonic::Add | Mnemonic::Lea => biased = true,
                _ => {}
            }
        }
        // Equality guards are null/sentinel checks (lazy-pointer thunks), not bounds checks.
        return match kind {
            "eq" => "null_check".to_string(),
            "sign" => "sign_check".to_string(),
            _ => format!("{cmp}{}_{kind}", if biased { "+bias" } else { "" }),
        };
    }
    let tail = &ins[n.saturating_sub(8)..];
    if tail.iter().any(|i| {
        i.mnemonic() == Mnemonic::And && !matches!(i.op1_kind(), OpKind::Register | OpKind::Memory)
    }) {
        return "mask".into();
    }
    if tail.iter().any(|i| i.mnemonic() == Mnemonic::Shr) {
        return "shift".into();
    }
    "none".into()
}

/// Classifies the jump at the end of `ins`.
pub(crate) fn classify(ctx: &Ctx<'_>, reason: UnresolvedJump, ins: &[&Instruction]) -> Pattern {
    let at = ins.len() - 1;
    let jmp = ins[at];
    let guard = guard_shape(ins);
    let (form, scale, location, indexed): (String, String, String, bool);
    if jmp.op0_kind() == OpKind::Memory {
        let sc = if jmp.memory_index() == Register::None {
            0
        } else {
            jmp.memory_index_scale() as u8
        };
        form = if jmp.is_ip_rel_memory_operand() {
            "jmp [rip+x]"
        } else {
            "jmp [mem]"
        }
        .to_string();
        scale = sc.to_string();
        location = ctx.mem_loc(ins, at, jmp, MAX_DEPTH);
        indexed = sc > 0;
    } else {
        let (f, s, l, ix) = match ctx.describe(ins, at, jmp.op0_register(), MAX_DEPTH) {
            Def::Load { scale, loc } => ("jmp reg<-load", scale.to_string(), loc, scale > 0),
            Def::RelOffset { scale, loc } => {
                ("jmp reg<-base+offset", scale.to_string(), loc, scale > 0)
            }
            Def::Anchor(a) => ("jmp reg<-lea", "0".into(), ctx.section_label(a), false),
            Def::CallResult => ("jmp reg<-call", "0".into(), "call_result".into(), false),
            Def::LiveIn => ("jmp reg<-live_in", "0".into(), "none".into(), false),
            Def::Other => ("jmp reg<-computed", "0".into(), "none".into(), false),
        };
        form = f.to_string();
        scale = s;
        location = l;
        indexed = ix;
    }
    // Compiled switch tables are read with a scale of 4 (offsets) or 8 (pointers); `[reg+reg]`
    // with scale 1/2 is pointer arithmetic on an object, not a table index.
    let table_scale = matches!(scale.as_str(), "4" | "8");
    let role = if indexed && table_scale {
        match ctx.pointer_table(ins, at, jmp) {
            Some(PointerTable::Functions) => "fn_table",
            Some(PointerTable::Writable) => "data_table",
            Some(PointerTable::Loaded) => "ptr_array",
            None => "switch",
        }
    } else if indexed
        || form.starts_with("jmp [")
        || form.contains("load")
        || form.contains("live_in")
        || form.contains("call")
    {
        "tail_call"
    } else {
        "computed"
    };
    Pattern {
        reason: reason.name(),
        role,
        form,
        scale,
        guard,
        location,
    }
}

/// Instruction window for the jump ending the block at `start`: the block, then sole-predecessor
/// blocks.
pub(crate) fn window<'a>(
    func: &'a Function,
    preds: &BTreeMap<u64, Vec<u64>>,
    start: u64,
) -> Vec<&'a Instruction> {
    window_with_merge(func, preds, start).0
}

/// Instruction window for the jump ending the block at `start`, also returning `true` if the
/// sole-predecessor chain terminated because a block had `> 1` intra-function predecessors.
fn window_with_merge<'a>(
    func: &'a Function,
    preds: &BTreeMap<u64, Vec<u64>>,
    start: u64,
) -> (Vec<&'a Instruction>, bool) {
    let mut chain = vec![start];
    let mut cur = start;
    let mut hit_merge_predecessor = false;
    while chain.len() < MAX_WINDOW_BLOCKS {
        match preds.get(&cur).map(Vec::as_slice) {
            Some([p]) if !chain.contains(p) => {
                chain.push(*p);
                cur = *p;
            }
            Some(ps) if ps.len() > 1 => {
                hit_merge_predecessor = true;
                break;
            }
            _ => break,
        }
    }
    let mut out = Vec::new();
    for b in chain.iter().rev() {
        if let Some(block) = func.blocks.get(b) {
            out.extend(block.instructions.iter());
        }
    }
    (out, hit_merge_predecessor)
}

/// Builds the survey from a finished analysis. `roots` are the VAs the boot starts at;
/// `top` bounds the number of listed locations.
pub fn survey_jumps(layout: &Layout, analysis: &Analysis, roots: &[u64], top: usize) -> JumpSurvey {
    survey_jumps_probed(layout, analysis, roots, top, None)
}

/// [`survey_jumps`] that also reads the leading entries of constant 8-byte tables from `image`
/// to tell function-pointer tables (`fn_table`) and run-time data tables (`data_table`) apart
/// from switch candidates.
pub fn survey_jumps_probed(
    layout: &Layout,
    analysis: &Analysis,
    roots: &[u64],
    top: usize,
    image: Option<&CodeImage<'_>>,
) -> JumpSurvey {
    let ctx = Ctx {
        layout,
        probe: image.map(|image| TableProbe {
            image,
            functions: &analysis.functions,
        }),
    };
    let mut callers: BTreeMap<u64, BTreeSet<u64>> = BTreeMap::new();
    let mut succ: BTreeMap<u64, BTreeSet<u64>> = BTreeMap::new();
    for (entry, f) in &analysis.functions {
        let mut out: BTreeSet<u64> = f.callees.clone();
        for b in f.blocks.values() {
            for e in &b.successors {
                if let Edge::TailCall(t) = e {
                    out.insert(*t);
                }
            }
        }
        for t in &out {
            callers.entry(*t).or_default().insert(*entry);
        }
        succ.insert(*entry, out);
    }
    let mut depth: BTreeMap<u64, u32> = BTreeMap::new();
    let mut q = VecDeque::new();
    for &r in roots {
        if analysis.functions.contains_key(&r) && depth.insert(r, 0).is_none() {
            q.push_back(r);
        }
    }
    while let Some(f) = q.pop_front() {
        let d = depth[&f];
        for &t in succ.get(&f).into_iter().flatten() {
            if let std::collections::btree_map::Entry::Vacant(e) = depth.entry(t) {
                e.insert(d + 1);
                q.push_back(t);
            }
        }
    }

    let mut survey = JumpSurvey {
        functions_total: analysis.functions.len(),
        functions_reachable: depth.len(),
        pointer_slot_jumps: analysis.pointer_slots.len(),
        ..Default::default()
    };
    let mut groups: BTreeMap<Pattern, Group> = BTreeMap::new();
    let mut seen: BTreeSet<u64> = BTreeSet::new();
    for (entry, f) in &analysis.functions {
        let mut preds: Option<BTreeMap<u64, Vec<u64>>> = None;
        for (start, b) in &f.blocks {
            if b.terminator != Terminator::IndirectJump {
                continue;
            }
            let Some(last) = b.instructions.last() else {
                continue;
            };
            let site = last.ip();
            let Some(&reason) = analysis.unresolved_jumps.get(&site) else {
                continue;
            };
            if !seen.insert(site) {
                continue;
            }
            let preds = preds.get_or_insert_with(|| {
                let mut m: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
                for (s, blk) in &f.blocks {
                    for t in blk.intra_successors() {
                        m.entry(t).or_default().push(*s);
                    }
                }
                m
            });
            let (ins, hit_merge) = window_with_merge(f, preds, *start);
            let pat = classify(&ctx, reason, &ins);
            let role = pat.role;
            let g = groups.entry(pat).or_default();
            g.sites.push(site);
            if hit_merge {
                g.merge_predecessor_sites += 1;
            }
            survey.total_unresolved += 1;
            *survey.by_reason.entry(reason.name()).or_default() += 1;
            let role_counts = survey.by_role.entry(role).or_default();
            role_counts.0 += 1;
            if let Some(&d) = depth.get(entry) {
                g.reachable += 1;
                survey.reachable_unresolved += 1;
                *survey.by_reason_reachable.entry(reason.name()).or_default() += 1;
                role_counts.1 += 1;
                *g.functions.entry(*entry).or_default() += 1;
                g.min_depth = Some(g.min_depth.map_or(d, |cur| cur.min(d)));
            }
        }
    }
    let mut patterns: Vec<(Pattern, Group)> = groups.into_iter().collect();
    patterns.sort_by(|a, b| {
        b.1.reachable
            .cmp(&a.1.reachable)
            .then(b.1.sites.len().cmp(&a.1.sites.len()))
            .then(a.0.cmp(&b.0))
    });
    let mut locs: BTreeMap<u64, Location> = BTreeMap::new();
    for (pi, (_, g)) in patterns.iter().enumerate() {
        for (&func, &n) in &g.functions {
            let l = locs.entry(func).or_insert_with(|| Location {
                function: func,
                depth: depth[&func],
                callers: callers.get(&func).map_or(0, BTreeSet::len),
                sites: 0,
                patterns: BTreeSet::new(),
            });
            l.sites += n;
            l.patterns.insert(pi + 1);
        }
    }
    let mut locations: Vec<Location> = locs.into_values().collect();
    locations.sort_by_key(|l| (l.depth, std::cmp::Reverse(l.callers), l.function));
    locations.truncate(top);
    survey.patterns = patterns;
    survey.locations = locations;
    survey
}

fn push_json_str(out: &mut String, input: &str) {
    out.push('"');
    for ch in input.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Plain-text report (addresses, labels and counts only).
pub fn render(s: &JumpSurvey, examples: usize) -> String {
    let mut o = String::new();
    let _ = writeln!(o, "unresolved indirect jmp survey");
    let _ = writeln!(
        o,
        "unresolved jumps: {} total, {} in functions reachable from boot ({} of {} functions, direct calls and tail calls only)",
        s.total_unresolved, s.reachable_unresolved, s.functions_reachable, s.functions_total
    );
    let _ = writeln!(
        o,
        "pointer-slot tail calls with a statically seeded target: {}",
        s.pointer_slot_jumps
    );
    let _ = writeln!(o, "by lifter reason:");
    for (k, v) in &s.by_reason {
        let reachable = s.by_reason_reachable.get(k).copied().unwrap_or(0);
        let _ = writeln!(o, "  {k}: {v} (reachable={reachable})");
    }
    let (sw_total, sw_reach, sw_merge) = s.switch_counts();
    let _ = writeln!(o, "by role:");
    for (role, (total, reachable)) in &s.by_role {
        if *role == "switch" {
            let _ = writeln!(
                o,
                "  {role}: total={total} reachable={reachable} merge_preds={sw_merge}"
            );
        } else {
            let _ = writeln!(o, "  {role}: total={total} reachable={reachable}");
        }
    }
    if sw_total > 0 && !s.by_role.contains_key("switch") {
        let _ = writeln!(
            o,
            "  switch: total={sw_total} reachable={sw_reach} merge_preds={sw_merge}"
        );
    }
    let _ = writeln!(o, "patterns (ranked by reachable count):");
    for (i, (p, g)) in s.patterns.iter().enumerate() {
        let tier = PriorityTier::classify(p, g);
        let min_depth = g
            .min_depth
            .map_or_else(|| "none".to_string(), |d| d.to_string());
        let ex: Vec<String> = g
            .sites
            .iter()
            .take(examples)
            .map(|a| format!("{a:#x}"))
            .collect();
        let _ = writeln!(
            o,
            "  #{} total={} reachable={} tier={} min_depth={} merge_preds={} :: {}\n      e.g. {}",
            i + 1,
            g.sites.len(),
            g.reachable,
            tier.as_str(),
            min_depth,
            g.merge_predecessor_sites,
            p.label(),
            ex.join(" ")
        );
    }
    let _ = writeln!(
        o,
        "locations (shallowest boot-reachable functions holding unresolved jumps):"
    );
    for l in &s.locations {
        let pats: Vec<String> = l.patterns.iter().map(|p| format!("#{p}")).collect();
        let _ = writeln!(
            o,
            "  fn {:#x} depth={} callers={} sites={} patterns={}",
            l.function,
            l.depth,
            l.callers,
            l.sites,
            pats.join(",")
        );
    }
    o
}

/// Deterministic, strict RFC 8259 JSON report (zero external crates, zero host paths).
pub fn render_json(s: &JumpSurvey, examples: usize) -> String {
    render_json_with_trace(s, examples, None)
}

fn render_json_with_trace(
    s: &JumpSurvey,
    examples: usize,
    trace: Option<(
        &str,
        &crate::trace_seed::SeedDiagnostics,
        &[crate::trace_seed::CorrelatedJumpHit],
        usize,
    )>,
) -> String {
    let (sw_total, sw_reach, sw_merge) = s.switch_counts();
    let prioritized = s.prioritized_patterns();

    let mut out = String::new();
    out.push_str("{\n");
    let _ = writeln!(out, "  \"total_unresolved\": {},", s.total_unresolved);
    let _ = writeln!(
        out,
        "  \"reachable_unresolved\": {},",
        s.reachable_unresolved
    );
    let _ = writeln!(out, "  \"functions_total\": {},", s.functions_total);
    let _ = writeln!(
        out,
        "  \"functions_reachable\": {},",
        s.functions_reachable
    );
    let _ = writeln!(out, "  \"pointer_slot_jumps\": {},", s.pointer_slot_jumps);

    out.push_str("  \"by_reason\": {");
    for (i, (k, v)) in s.by_reason.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        push_json_str(&mut out, k);
        let _ = write!(out, ": {v}");
    }
    out.push_str("},\n");

    out.push_str("  \"by_reason_reachable\": {");
    for (i, (k, v)) in s.by_reason_reachable.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        push_json_str(&mut out, k);
        let _ = write!(out, ": {v}");
    }
    out.push_str("},\n");

    out.push_str("  \"by_role\": {");
    for (i, (role, (total, reachable))) in s.by_role.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        push_json_str(&mut out, role);
        let _ = write!(out, ": {{\"total\": {total}, \"reachable\": {reachable}}}");
    }
    out.push_str("},\n");

    let _ = writeln!(
        out,
        "  \"switch_summary\": {{\"total\": {sw_total}, \"reachable\": {sw_reach}, \"merge_predecessor_sites\": {sw_merge}}},"
    );

    out.push_str("  \"patterns\": [");
    for (idx, item) in prioritized.iter().enumerate() {
        out.push_str(if idx == 0 { "\n" } else { ",\n" });
        out.push_str("    {");
        let _ = write!(
            out,
            "\"rank\": {}, \"pattern_index\": {}, \"tier\": ",
            idx + 1,
            item.pattern_index
        );
        push_json_str(&mut out, item.tier.as_str());
        let _ = write!(
            out,
            ", \"priority_score\": {}, \"total\": {}, \"reachable\": {}, \"min_depth\": ",
            item.score,
            item.group.sites.len(),
            item.group.reachable
        );
        match item.group.min_depth {
            Some(d) => {
                let _ = write!(out, "{d}");
            }
            None => out.push_str("null"),
        }
        let _ = write!(
            out,
            ", \"merge_predecessor_sites\": {}, \"reason\": ",
            item.group.merge_predecessor_sites
        );
        push_json_str(&mut out, item.pattern.reason);
        out.push_str(", \"role\": ");
        push_json_str(&mut out, item.pattern.role);
        out.push_str(", \"form\": ");
        push_json_str(&mut out, &item.pattern.form);
        out.push_str(", \"scale\": ");
        push_json_str(&mut out, &item.pattern.scale);
        out.push_str(", \"guard\": ");
        push_json_str(&mut out, &item.pattern.guard);
        out.push_str(", \"location\": ");
        push_json_str(&mut out, &item.pattern.location);
        out.push_str(", \"examples\": [");
        for (ei, site) in item.group.sites.iter().take(examples).enumerate() {
            if ei > 0 {
                out.push_str(", ");
            }
            let _ = write!(out, "\"0x{site:x}\"");
        }
        out.push_str("]}");
    }
    if prioritized.is_empty() {
        out.push_str("],\n");
    } else {
        out.push_str("\n  ],\n");
    }

    out.push_str("  \"locations\": [");
    for (i, loc) in s.locations.iter().enumerate() {
        out.push_str(if i == 0 { "\n" } else { ",\n" });
        out.push_str("    {");
        let _ = write!(
            out,
            "\"function\": \"0x{:x}\", \"depth\": {}, \"callers\": {}, \"sites\": {}, \"patterns\": [",
            loc.function, loc.depth, loc.callers, loc.sites
        );
        for (pi, pat) in loc.patterns.iter().enumerate() {
            if pi > 0 {
                out.push_str(", ");
            }
            let _ = write!(out, "{pat}");
        }
        out.push_str("]}");
    }
    if s.locations.is_empty() {
        out.push(']');
    } else {
        out.push_str("\n  ]");
    }

    if let Some((module, diag, correlated, top)) = trace {
        out.push_str(",\n  \"trace_seed\": {\n");
        let diag_json = diag.render_json(module, top);
        let indented = diag_json
            .trim_end_matches('\n')
            .replace('\n', "\n    ");
        let _ = writeln!(out, "    \"diagnostics\": {indented},");
        let _ = writeln!(out, "    \"correlated_hits_total\": {},", correlated.len());
        out.push_str("    \"correlated_hits\": [");
        let shown = correlated.len().min(top);
        for (i, hit) in correlated.iter().take(shown).enumerate() {
            out.push_str(if i == 0 { "\n" } else { ",\n" });
            out.push_str("      {");
            let _ = write!(
                out,
                "\"function\": \"0x{:x}\", \"jump_va\": \"0x{:x}\", \"reason\": ",
                hit.function, hit.jump_va
            );
            push_json_str(&mut out, hit.reason.name());
            let _ = write!(out, ", \"pattern_rank\": {}, \"pattern_label\": ", hit.pattern_rank);
            push_json_str(&mut out, &hit.pattern_label);
            out.push_str(", \"role\": ");
            push_json_str(&mut out, hit.role);
            out.push_str(", \"guard\": ");
            push_json_str(&mut out, &hit.guard);
            let _ = write!(
                out,
                ", \"boot_reachable\": {}, \"accepted_targets\": {}, \"hit_targets\": [",
                hit.boot_reachable, hit.accepted_targets
            );
            for (ti, t) in hit.hit_targets.iter().enumerate() {
                if ti > 0 {
                    out.push_str(", ");
                }
                let _ = write!(out, "\"0x{t:x}\"");
            }
            out.push_str("], \"distinct_sites\": [");
            for (si, site) in hit.distinct_sites.iter().enumerate() {
                if si > 0 {
                    out.push_str(", ");
                }
                let _ = write!(out, "\"0x{site:x}\"");
            }
            out.push_str("]}");
        }
        if shown == 0 {
            out.push_str("]\n");
        } else {
            out.push_str("\n    ]\n");
        }
        out.push_str("  }\n");
    } else {
        out.push('\n');
    }

    out.push_str("}\n");
    out
}

fn render_trace_correlation_text(
    module: &str,
    diag: &crate::trace_seed::SeedDiagnostics,
    correlated: &[crate::trace_seed::CorrelatedJumpHit],
    top: usize,
) -> String {
    let mut out = String::new();
    out.push('\n');
    out.push_str(&diag.render_text(module, top));
    let _ = writeln!(
        out,
        "trace-seed survey correlation (top {} of {} matched unresolved jumps):",
        top.min(correlated.len()),
        correlated.len()
    );
    for hit in correlated.iter().take(top) {
        let _ = writeln!(
            out,
            "  fn {:#x} jump {:#x} pattern=#{} role={} guard={} boot_reachable={} accepted={} hits={} sites={} :: {}",
            hit.function,
            hit.jump_va,
            hit.pattern_rank,
            hit.role,
            hit.guard,
            hit.boot_reachable,
            hit.accepted_targets,
            hit.hit_targets.len(),
            hit.distinct_sites.len(),
            hit.pattern_label
        );
    }
    out
}

/// Boot roots: the PE entry point and TLS callbacks.
pub fn boot_roots(pe: &Pe64) -> Vec<u64> {
    let mut v = vec![pe.optional.image_base + u64::from(pe.optional.entry_point_rva)];
    if let Some(t) = &pe.tls {
        v.extend(t.callbacks.iter().map(|c| c.va));
    }
    v
}

/// Extra reachability roots from a boot's recorded indirect-call targets: a recorded target is code
/// the boot actually ran, so everything it calls directly is boot-reachable too. Targets that are
/// not function entries are ignored by [`survey_jumps`].
pub fn recorded_roots(seeds: &crate::trace_seed::TraceSeeds) -> Vec<u64> {
    let set: BTreeSet<u64> = seeds.pairs.iter().map(|p| p.target).collect();
    set.into_iter().collect()
}

/// `survey-jumps <pe-path> [--top N] [--examples N] [--no-refine] [--json] [--switches-only] [--seed-targets <file>] [--out <file>]`.
pub const USAGE: &str = "survey-jumps <pe-path> [--top N] [--examples N] [--no-refine] [--json] [--switches-only] [--seed-targets <file>] [--out <file>]";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    pub pe_path: std::path::PathBuf,
    pub top: usize,
    pub examples: usize,
    pub no_refine: bool,
    pub json: bool,
    pub switches_only: bool,
    pub seed_targets: Option<std::path::PathBuf>,
    pub out: Option<std::path::PathBuf>,
}

pub fn parse_args(args: &[String]) -> Result<Option<Args>, String> {
    let mut out = Args {
        pe_path: Default::default(),
        top: 40,
        examples: 4,
        no_refine: false,
        json: false,
        switches_only: false,
        seed_targets: None,
        out: None,
    };
    let mut path = None;
    let mut i = 0;
    let num = |i: &mut usize, flag: &str| -> Result<usize, String> {
        *i += 1;
        args.get(*i)
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| format!("{flag} requires a number"))
    };
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => return Ok(None),
            "--no-refine" => out.no_refine = true,
            "--json" => out.json = true,
            "--switches-only" => out.switches_only = true,
            "--top" => out.top = num(&mut i, "--top")?,
            "--examples" => out.examples = num(&mut i, "--examples")?,
            "--seed-targets" => {
                i += 1;
                out.seed_targets = Some(
                    args.get(i)
                        .ok_or_else(|| "--seed-targets requires a value".to_string())?
                        .into(),
                );
            }
            "--out" => {
                i += 1;
                out.out = Some(
                    args.get(i)
                        .ok_or_else(|| "--out requires a value".to_string())?
                        .into(),
                );
            }
            f if f.starts_with('-') => {
                return Err(format!("unknown option '{f}' for 'survey-jumps'"));
            }
            p => {
                if path.is_some() {
                    return Err(format!("unexpected extra positional argument '{p}'"));
                }
                path = Some(p.into());
            }
        }
        i += 1;
    }
    out.pe_path = path.ok_or_else(|| "missing input PE path <pe-path>".to_string())?;
    Ok(Some(out))
}

/// Runs discovery and the survey; returns the report text.
pub fn run(args: &Args) -> Result<String, String> {
    let bytes = std::fs::read(&args.pe_path)
        .map_err(|e| format!("failed to read '{}': {e}", args.pe_path.display()))?;
    let pe = Pe64::parse(&bytes).map_err(|e| format!("failed to parse PE64: {e}"))?;
    let mut seeded = crate::pipeline::seed(&pe, &bytes);
    let analysis = if args.no_refine {
        crate::discover::discover(&seeded.image)
    } else {
        let guest =
            crate::image::build_guest_image(&pe, &bytes, crate::image::ImageOptions::default())
                .map_err(|e| format!("failed to build the guest image: {e:?}"))?;
        crate::pipeline::refine::discover_refined(&mut seeded.image, &guest.image).0
    };
    let seeds = match &args.seed_targets {
        Some(path) => Some(crate::trace_seed::load(path)?),
        None => None,
    };
    let mut roots = boot_roots(&pe);
    if let Some(seeds) = &seeds {
        roots.extend(recorded_roots(seeds));
    }
    let full_survey = survey_jumps_probed(
        &Layout::from_pe(&pe),
        &analysis,
        &roots,
        args.top,
        Some(&seeded.image),
    );
    let survey = if args.switches_only {
        full_survey.switches_only()
    } else {
        full_survey
    };

    let module = crate::summary::sanitize_label(&args.pe_path.to_string_lossy());
    let trace_data = if let Some(seeds) = &seeds {
        let outcome = crate::trace_seed::evaluate(&seeded.image, &analysis, seeds);
        let diag = outcome.diagnostics();
        let correlated = crate::trace_seed::correlate_with_survey(&outcome, &survey);
        Some((diag, correlated))
    } else {
        None
    };

    let text = if args.json {
        let trace_ref = trace_data
            .as_ref()
            .map(|(diag, corr)| (module.as_str(), diag, corr.as_slice(), args.top));
        render_json_with_trace(&survey, args.examples, trace_ref)
    } else {
        let mut s = render(&survey, args.examples);
        if let Some((diag, corr)) = &trace_data {
            s.push_str(&render_trace_correlation_text(&module, diag, corr, args.top));
        }
        s
    };

    match &args.out {
        Some(f) => {
            std::fs::write(f, &text)
                .map_err(|e| format!("failed to write '{}': {e}", f.display()))?;
            Ok(format!("Wrote jump survey to {}\n", f.display()))
        }
        None => Ok(text),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code_image::{CodeImage, CodeSection};
    use crate::discover::discover;
    use iced_x86::code_asm::*;

    const BASE: u64 = 0x1_4000_0000;
    const TEXT: u64 = BASE + 0x1000;

    fn analyse(build: impl FnOnce(&mut CodeAssembler)) -> Analysis {
        let mut a = CodeAssembler::new(64).unwrap();
        build(&mut a);
        let code = a.assemble(TEXT).unwrap();
        let text: &'static [u8] = Box::leak([code, vec![0xCC; 16]].concat().into_boxed_slice());
        let rdata: &'static [u8] = Box::leak(vec![0u8; 0x100].into_boxed_slice());
        let image = CodeImage::new(
            BASE,
            vec![
                CodeSection::new(TEXT, text, true),
                CodeSection::new(BASE + 0x2000, rdata, false),
            ],
            vec![TEXT],
        );
        discover(&image)
    }

    fn layout() -> Layout {
        Layout {
            image_base: BASE,
            sections: vec![
                (".text".into(), 0x1000, 0x1000, 0x6000_0020),
                (".rdata".into(), 0x2000, 0x100, 0x4000_0040),
            ],
        }
    }

    /// Image with `.rdata` holding `rdata` (writable when `writable`), the jump's role probed.
    fn probed_role(
        build: impl FnOnce(&mut CodeAssembler),
        rdata: Vec<u8>,
        writable: bool,
    ) -> &'static str {
        let mut a = CodeAssembler::new(64).unwrap();
        build(&mut a);
        let code = a.assemble(TEXT).unwrap();
        let text: &'static [u8] = Box::leak([code, vec![0xCC; 16]].concat().into_boxed_slice());
        let rdata: &'static [u8] = Box::leak(rdata.into_boxed_slice());
        let image = Box::leak(Box::new(CodeImage::new(
            BASE,
            vec![
                CodeSection::new(TEXT, text, true),
                CodeSection::new(BASE + 0x2000, rdata, false),
            ],
            vec![TEXT],
        )));
        let an = discover(image);
        let mut l = layout();
        if writable {
            l.sections[1].3 |= 0x8000_0000;
        }
        let s = survey_jumps_probed(&l, &an, &[TEXT], 5, Some(image));
        assert_eq!(s.total_unresolved, 1);
        s.patterns[0].0.role
    }

    fn table_of(entries: &[u64]) -> Vec<u8> {
        let mut v: Vec<u8> = entries.iter().flat_map(|e| e.to_le_bytes()).collect();
        v.resize(0x100, 0);
        v
    }

    fn lea_table_jump(a: &mut CodeAssembler) {
        a.movsxd(rax, ecx).unwrap();
        let lea = iced_x86::Instruction::with2(
            iced_x86::Code::Lea_r64_m,
            Register::RCX,
            iced_x86::MemoryOperand::with_base_displ(Register::RIP, (BASE + 0x2000) as i64),
        )
        .unwrap();
        a.add_instruction(lea).unwrap();
        a.jmp(qword_ptr(rcx + rax * 8)).unwrap();
    }

    #[test]
    fn read_only_table_of_function_entries_is_a_fn_table() {
        assert_eq!(
            probed_role(lea_table_jump, table_of(&[TEXT, 0, TEXT]), false),
            "fn_table"
        );
    }

    #[test]
    fn read_only_table_of_case_labels_stays_a_switch() {
        // An entry that is not a function entry (inside the function) keeps the switch label.
        assert_eq!(
            probed_role(lea_table_jump, table_of(&[TEXT + 4, TEXT]), false),
            "switch"
        );
        // No probe: never reclassified.
        let an = analyse(lea_table_jump);
        assert_eq!(
            survey_jumps(&layout(), &an, &[TEXT], 5).patterns[0].0.role,
            "switch"
        );
    }

    #[test]
    fn writable_table_is_a_data_table() {
        assert_eq!(
            probed_role(lea_table_jump, table_of(&[]), true),
            "data_table"
        );
    }

    #[test]
    fn index_into_a_loaded_vtable_is_a_pointer_array() {
        let role = probed_role(
            |a| {
                a.mov(rdx, qword_ptr(rcx)).unwrap();
                a.movzx(eax, al).unwrap();
                a.jmp(qword_ptr(rdx + rax * 8 + 0x38)).unwrap();
            },
            table_of(&[]),
            false,
        );
        assert_eq!(role, "ptr_array");
    }

    #[test]
    fn unscaled_pointer_read_and_call_result_are_tail_calls() {
        let an = analyse(|a| {
            a.mov(rdx, qword_ptr(rcx)).unwrap();
            a.jmp(qword_ptr(rdx + rax)).unwrap();
        });
        let s = survey_jumps(&layout(), &an, &[TEXT], 5);
        assert_eq!(s.patterns[0].0.role, "tail_call");
        let an = analyse(|a| {
            a.sub(rsp, 0x28).unwrap();
            a.call(TEXT + 0x40).unwrap();
            a.add(rsp, 0x28).unwrap();
            a.jmp(rax).unwrap();
        });
        let s = survey_jumps(&layout(), &an, &[TEXT], 5);
        let p = &s
            .patterns
            .iter()
            .find(|(p, _)| p.form == "jmp reg<-call")
            .unwrap()
            .0;
        assert_eq!((p.role, p.location.as_str()), ("tail_call", "call_result"));
    }

    #[test]
    fn recorded_targets_extend_boot_reachability() {
        let an = analyse(|a| {
            a.jmp(qword_ptr(rax + 0x10)).unwrap();
        });
        let seeds = crate::trace_seed::TraceSeeds {
            pairs: vec![crate::trace_seed::TracePair {
                site: 0x1234,
                target: TEXT,
            }],
        };
        let roots = recorded_roots(&seeds);
        assert_eq!(roots, vec![TEXT]);
        assert_eq!(survey_jumps(&layout(), &an, &[], 5).reachable_unresolved, 0);
        assert_eq!(
            survey_jumps(&layout(), &an, &roots, 5).reachable_unresolved,
            1
        );
    }

    #[test]
    fn vtable_tail_call_is_grouped_as_tail_call() {
        let an = analyse(|a| {
            a.jmp(qword_ptr(rax + 0x10)).unwrap();
        });
        let s = survey_jumps(&layout(), &an, &[TEXT], 5);
        assert_eq!(s.total_unresolved, 1);
        assert_eq!(s.reachable_unresolved, 1);
        let (p, g) = &s.patterns[0];
        assert_eq!(p.role, "tail_call");
        assert_eq!(p.guard, "none");
        assert_eq!(g.sites.len(), 1);
        assert!(render(&s, 2).contains("tail_call"));
    }

    #[test]
    fn scaled_load_from_spilled_pointer_is_a_switch() {
        let an = analyse(|a| {
            a.mov(rdx, qword_ptr(rsp + 0x8)).unwrap();
            a.mov(rax, qword_ptr(rdx + rax * 8)).unwrap();
            a.jmp(rax).unwrap();
        });
        let s = survey_jumps(&layout(), &an, &[TEXT], 5);
        let (p, _) = &s.patterns[0];
        assert_eq!(p.role, "switch");
        assert_eq!(p.scale, "8");
        assert_eq!(p.location, "loaded_pointer");
        assert_eq!(p.guard, "none");
    }

    #[test]
    fn unreachable_roots_count_nothing_reachable() {
        let an = analyse(|a| {
            a.jmp(qword_ptr(rax + 0x10)).unwrap();
        });
        let s = survey_jumps(&layout(), &an, &[], 5);
        assert_eq!(s.total_unresolved, 1);
        assert_eq!(s.reachable_unresolved, 0);
    }

    #[test]
    fn parse_args_reads_flags() {
        let a = parse_args(&["x.exe".into(), "--top".into(), "3".into()])
            .unwrap()
            .unwrap();
        assert_eq!(a.top, 3);
        assert!(!a.json);
        assert!(!a.switches_only);
        assert!(a.seed_targets.is_none());
        assert!(parse_args(&["--bogus".into()]).is_err());
    }

    #[test]
    fn parse_args_supports_json_switches_only_and_seed_targets() {
        let a = parse_args(&[
            "game.exe".into(),
            "--json".into(),
            "--switches-only".into(),
            "--no-refine".into(),
            "--top".into(),
            "15".into(),
            "--examples".into(),
            "2".into(),
            "--seed-targets".into(),
            "seeds.txt".into(),
            "--out".into(),
            "report.json".into(),
        ])
        .unwrap()
        .unwrap();
        assert_eq!(a.pe_path, std::path::PathBuf::from("game.exe"));
        assert!(a.json);
        assert!(a.switches_only);
        assert!(a.no_refine);
        assert_eq!(a.top, 15);
        assert_eq!(a.examples, 2);
        assert_eq!(a.seed_targets, Some(std::path::PathBuf::from("seeds.txt")));
        assert_eq!(a.out, Some(std::path::PathBuf::from("report.json")));

        assert_eq!(parse_args(&["--help".into()]).unwrap(), None);
        assert_eq!(parse_args(&["-h".into()]).unwrap(), None);
        assert!(parse_args(&["game.exe".into(), "--seed-targets".into()]).is_err());
        assert!(parse_args(&["game.exe".into(), "--out".into()]).is_err());
        assert!(parse_args(&["game.exe".into(), "--top".into(), "abc".into()]).is_err());
        assert!(parse_args(&["game.exe".into(), "--examples".into()]).is_err());
        assert!(parse_args(&["a.exe".into(), "b.exe".into()]).is_err());
        assert!(parse_args(&["--json".into()]).is_err());
    }

    /// Strict RFC 8259 JSON syntax validator (zero external dependencies).
    struct JsonValidator<'a> {
        bytes: &'a [u8],
        pos: usize,
    }

    impl<'a> JsonValidator<'a> {
        fn validate(input: &'a str) -> Result<(), String> {
            let mut v = Self {
                bytes: input.as_bytes(),
                pos: 0,
            };
            v.skip_ws();
            v.parse_value()?;
            v.skip_ws();
            if v.pos != v.bytes.len() {
                return Err(format!("trailing bytes at offset {}", v.pos));
            }
            Ok(())
        }

        fn peek(&self) -> Option<u8> {
            self.bytes.get(self.pos).copied()
        }

        fn skip_ws(&mut self) {
            while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
                self.pos += 1;
            }
        }

        fn expect_byte(&mut self, expected: u8) -> Result<(), String> {
            match self.peek() {
                Some(b) if b == expected => {
                    self.pos += 1;
                    Ok(())
                }
                other => Err(format!(
                    "expected '{}' at offset {}, got {:?}",
                    expected as char, self.pos, other
                )),
            }
        }

        fn parse_value(&mut self) -> Result<(), String> {
            self.skip_ws();
            match self.peek() {
                Some(b'{') => self.parse_object(),
                Some(b'[') => self.parse_array(),
                Some(b'"') => self.parse_string(),
                Some(b't') => self.parse_literal(b"true"),
                Some(b'f') => self.parse_literal(b"false"),
                Some(b'n') => self.parse_literal(b"null"),
                Some(b'-' | b'0'..=b'9') => self.parse_number(),
                other => Err(format!("unexpected token {other:?} at offset {}", self.pos)),
            }
        }

        fn parse_literal(&mut self, lit: &[u8]) -> Result<(), String> {
            if self.bytes[self.pos..].starts_with(lit) {
                self.pos += lit.len();
                Ok(())
            } else {
                Err(format!("invalid literal at offset {}", self.pos))
            }
        }

        fn parse_string(&mut self) -> Result<(), String> {
            self.expect_byte(b'"')?;
            while let Some(b) = self.peek() {
                self.pos += 1;
                match b {
                    b'"' => return Ok(()),
                    b'\\' => {
                        let esc = self
                            .peek()
                            .ok_or_else(|| format!("unterminated escape at {}", self.pos))?;
                        self.pos += 1;
                        match esc {
                            b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {}
                            b'u' => {
                                for _ in 0..4 {
                                    let h = self.peek().ok_or_else(|| {
                                        format!("truncated unicode escape at {}", self.pos)
                                    })?;
                                    if !h.is_ascii_hexdigit() {
                                        return Err(format!("invalid hex escape at {}", self.pos));
                                    }
                                    self.pos += 1;
                                }
                            }
                            _ => return Err(format!("invalid escape '\\{}' at {}", esc as char, self.pos)),
                        }
                    }
                    0x00..=0x1F => {
                        return Err(format!("unescaped control byte 0x{b:02x} at {}", self.pos));
                    }
                    _ => {}
                }
            }
            Err(format!("unterminated string at {}", self.pos))
        }

        fn parse_number(&mut self) -> Result<(), String> {
            if self.peek() == Some(b'-') {
                self.pos += 1;
            }
            match self.peek() {
                Some(b'0') => self.pos += 1,
                Some(b'1'..=b'9') => {
                    while matches!(self.peek(), Some(b'0'..=b'9')) {
                        self.pos += 1;
                    }
                }
                _ => return Err(format!("invalid number at {}", self.pos)),
            }
            if self.peek() == Some(b'.') {
                self.pos += 1;
                if !matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(format!("expected fraction digit at {}", self.pos));
                }
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            if matches!(self.peek(), Some(b'e' | b'E')) {
                self.pos += 1;
                if matches!(self.peek(), Some(b'+' | b'-')) {
                    self.pos += 1;
                }
                if !matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(format!("expected exponent digit at {}", self.pos));
                }
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            Ok(())
        }

        fn parse_array(&mut self) -> Result<(), String> {
            self.expect_byte(b'[')?;
            self.skip_ws();
            if self.peek() == Some(b']') {
                self.pos += 1;
                return Ok(());
            }
            loop {
                self.parse_value()?;
                self.skip_ws();
                match self.peek() {
                    Some(b',') => self.pos += 1,
                    Some(b']') => {
                        self.pos += 1;
                        return Ok(());
                    }
                    other => {
                        return Err(format!(
                            "expected ',' or ']' in array at {}, got {other:?}",
                            self.pos
                        ));
                    }
                }
            }
        }

        fn parse_object(&mut self) -> Result<(), String> {
            self.expect_byte(b'{')?;
            self.skip_ws();
            if self.peek() == Some(b'}') {
                self.pos += 1;
                return Ok(());
            }
            loop {
                self.skip_ws();
                self.parse_string()?;
                self.skip_ws();
                self.expect_byte(b':')?;
                self.parse_value()?;
                self.skip_ws();
                match self.peek() {
                    Some(b',') => self.pos += 1,
                    Some(b'}') => {
                        self.pos += 1;
                        return Ok(());
                    }
                    other => {
                        return Err(format!(
                            "expected ',' or '}}' in object at {}, got {other:?}",
                            self.pos
                        ));
                    }
                }
            }
        }
    }

    fn analyse_multi(entries: Vec<u64>, code: Vec<u8>) -> Analysis {
        let text: &'static [u8] = Box::leak([code, vec![0xCC; 16]].concat().into_boxed_slice());
        let rdata: &'static [u8] = Box::leak(vec![0u8; 0x100].into_boxed_slice());
        let image = CodeImage::new(
            BASE,
            vec![
                CodeSection::new(TEXT, text, true),
                CodeSection::new(BASE + 0x2000, rdata, false),
            ],
            entries,
        );
        discover(&image)
    }

    #[test]
    fn merge_predecessor_shift_guarded_switch_is_critical_boot_switch() {
        let an = analyse(|a| {
            let mut alt = a.create_label();
            let mut merge = a.create_label();
            a.test(ecx, ecx).unwrap();
            a.jz(alt).unwrap();
            a.mov(rdx, qword_ptr(rsp + 0x8)).unwrap();
            a.jmp(merge).unwrap();
            a.set_label(&mut alt).unwrap();
            a.mov(rdx, qword_ptr(rsp + 0x10)).unwrap();
            a.set_label(&mut merge).unwrap();
            a.shr(rax, 2).unwrap();
            a.jmp(qword_ptr(rdx + rax * 8)).unwrap();
        });
        let s = survey_jumps(&layout(), &an, &[TEXT], 10);
        assert_eq!(s.total_unresolved, 1);
        assert_eq!(s.reachable_unresolved, 1);
        assert_eq!(s.switch_counts(), (1, 1, 1));
        assert_eq!(s.by_role.get("switch"), Some(&(1, 1)));
        let (p, g) = &s.patterns[0];
        assert_eq!(p.role, "switch");
        assert_eq!(p.guard, "shift");
        assert_eq!(p.scale, "8");
        assert_eq!(g.merge_predecessor_sites, 1);
        assert_eq!(g.min_depth, Some(0));
        assert_eq!(PriorityTier::classify(p, g), PriorityTier::CriticalBootSwitch);

        let prio = s.prioritized_patterns();
        assert_eq!(prio.len(), 1);
        assert_eq!(prio[0].tier, PriorityTier::CriticalBootSwitch);
        assert_eq!(prio[0].pattern_index, 1);
    }

    #[test]
    fn priority_ranking_puts_boot_switch_ahead_of_higher_count_tail_call() {
        // Build 3 functions in one code buffer:
        // fn0 (boot root, depth 0): calls fn1, fn2, fn3, fn4
        // fn1, fn2, fn3 (depth 1): each does a vtable tail call `jmp [rax + 0x10]` (3 reachable tail_calls)
        // fn4 (depth 1): boot-reachable switch (`mov rdx, [rcx+8]; mov rax, [rdx+rax*8]; jmp rax`)
        // fn5 (unreachable): another switch
        let mut buf = Vec::new();
        let assemble_at = |va: u64, build: fn(&mut CodeAssembler)| -> Vec<u8> {
            let mut a = CodeAssembler::new(64).unwrap();
            build(&mut a);
            a.assemble(va).unwrap()
        };

        let fn0 = TEXT;
        let fn1 = TEXT + 0x40;
        let fn2 = TEXT + 0x60;
        let fn3 = TEXT + 0x80;
        let fn4 = TEXT + 0xA0;
        let fn5 = TEXT + 0xE0;

        let mut a0 = CodeAssembler::new(64).unwrap();
        a0.call(fn1).unwrap();
        a0.call(fn2).unwrap();
        a0.call(fn3).unwrap();
        a0.call(fn4).unwrap();
        a0.ret().unwrap();
        let b0 = a0.assemble(fn0).unwrap();
        buf.extend_from_slice(&b0);
        buf.resize((fn1 - TEXT) as usize, 0xCC);

        for target in [fn1, fn2, fn3] {
            let b = assemble_at(target, |a| {
                a.jmp(qword_ptr(rax + 0x10)).unwrap();
            });
            buf.extend_from_slice(&b);
            buf.resize(((target - TEXT) + 0x20) as usize, 0xCC);
        }

        let b4 = assemble_at(fn4, |a| {
            a.mov(rdx, qword_ptr(rsp + 0x8)).unwrap();
            a.mov(rax, qword_ptr(rdx + rax * 8)).unwrap();
            a.jmp(rax).unwrap();
        });
        buf.extend_from_slice(&b4);
        buf.resize((fn5 - TEXT) as usize, 0xCC);

        let b5 = assemble_at(fn5, |a| {
            a.mov(rdx, qword_ptr(rsp + 0x10)).unwrap();
            a.shr(rax, 1).unwrap();
            a.jmp(qword_ptr(rdx + rax * 8)).unwrap();
        });
        buf.extend_from_slice(&b5);

        let an = analyse_multi(vec![fn0, fn5], buf);
        let s = survey_jumps(&layout(), &an, &[fn0], 10);

        assert_eq!(s.total_unresolved, 5);
        assert_eq!(s.reachable_unresolved, 4);
        assert_eq!(s.by_role.get("tail_call"), Some(&(3, 3)));
        assert_eq!(s.by_role.get("switch"), Some(&(2, 1)));
        assert_eq!(s.switch_counts(), (2, 1, 0));
        assert_eq!(s.by_reason_reachable.values().sum::<usize>(), 4);

        // Raw patterns are sorted by reachable count desc -> tail_call (3) is first.
        assert_eq!(s.patterns[0].0.role, "tail_call");
        assert_eq!(s.patterns[0].1.reachable, 3);

        // Prioritized patterns put the boot-reachable switch ahead of the higher-count tail_call!
        let prio = s.prioritized_patterns();
        assert_eq!(prio.len(), 3);
        assert_eq!(prio[0].pattern.role, "switch");
        assert!(matches!(
            prio[0].tier,
            PriorityTier::CriticalBootSwitch | PriorityTier::BootSwitch
        ));
        assert_eq!(prio[0].group.min_depth, Some(1));
        assert_eq!(prio[1].pattern.role, "tail_call");
        assert_eq!(prio[1].tier, PriorityTier::BootOther);
        assert_eq!(prio[2].pattern.role, "switch");
        assert_eq!(prio[2].tier, PriorityTier::UnreachableSwitch);
        assert_eq!(prio[2].group.min_depth, None);
        assert!(prio[0].score > prio[1].score);
        assert!(prio[1].score > prio[2].score);

        // Role filtering & switches_only()
        assert_eq!(s.filtered_by_role("switch").len(), 2);
        assert_eq!(s.filtered_by_role("tail_call").len(), 1);
        let sw_only = s.switches_only();
        assert_eq!(sw_only.total_unresolved, 2);
        assert_eq!(sw_only.reachable_unresolved, 1);
        assert_eq!(sw_only.patterns.len(), 2);
        assert!(sw_only.patterns.iter().all(|(p, _)| p.role == "switch"));
        assert_eq!(sw_only.locations.len(), 1);
        assert_eq!(sw_only.locations[0].function, fn4);
        assert_eq!(sw_only.by_role.len(), 1);
        assert_eq!(sw_only.by_role.get("switch"), Some(&(2, 1)));

        // Text rendering preserves headers and adds role/tier/depth/merge metadata.
        let text = render(&s, 3);
        assert!(text.contains("unresolved indirect jmp survey"));
        assert!(text.contains("by lifter reason:"));
        assert!(text.contains("by role:"));
        assert!(text.contains("patterns (ranked by reachable count):"));
        assert!(text.contains("locations (shallowest boot-reachable functions holding unresolved jumps):"));
        assert!(text.contains("tier="));
        assert!(text.contains("min_depth="));
        assert!(text.contains("merge_preds="));
        assert!(!crate::summary::contains_host_path(&text));

        // JSON rendering is strict RFC 8259, deterministic, and host-path-free.
        let json1 = render_json(&s, 3);
        let json2 = render_json(&s, 3);
        assert_eq!(json1, json2);
        JsonValidator::validate(&json1).expect("render_json must emit valid RFC 8259 JSON");
        assert!(!crate::summary::contains_host_path(&json1));
        assert!(json1.contains("\"by_reason_reachable\""));
        assert!(json1.contains("\"by_role\""));
        assert!(json1.contains("\"switch_summary\""));
        assert!(json1.contains("\"priority_score\""));
        assert!(json1.contains("\"merge_predecessor_sites\""));

        let sw_json = render_json(&sw_only, 3);
        JsonValidator::validate(&sw_json).expect("switches_only JSON must be valid");
        assert!(!sw_json.contains("\"tail_call\""));
    }

    #[test]
    fn priority_tiers_and_min_depth_tiebreak_order_cleanly() {
        let mk_pat = |role: &'static str, reason: &'static str, guard: &str, form: &str| Pattern {
            reason,
            role,
            form: form.into(),
            scale: if role == "switch" { "8".into() } else { "0".into() },
            guard: guard.into(),
            location: "loaded_pointer".into(),
        };
        let mk_grp = |sites: Vec<u64>, reachable: usize, min_depth: Option<u32>, merge: usize| Group {
            sites,
            reachable,
            functions: BTreeMap::new(),
            min_depth,
            merge_predecessor_sites: merge,
        };

        let p_crit_d0 = mk_pat("switch", "table_base_unknown", "shift", "jmp [mem]");
        let g_crit_d0 = mk_grp(vec![0x10], 1, Some(0), 1);

        let p_crit_d2 = mk_pat("switch", "table_unbounded", "none", "jmp [mem]");
        let g_crit_d2 = mk_grp(vec![0x20], 1, Some(2), 0);

        let p_boot_sw = mk_pat("switch", "memory_operand", "none", "jmp [mem]");
        let g_boot_sw = mk_grp(vec![0x30], 2, Some(1), 0);

        let p_boot_other = mk_pat("tail_call", "memory_operand", "none", "jmp [mem]");
        let g_boot_other = mk_grp(vec![0x40, 0x41, 0x42], 3, Some(0), 0);

        let p_unreach_sw = mk_pat("switch", "table_base_unknown", "shift", "jmp [mem]");
        let g_unreach_sw = mk_grp(vec![0x50], 0, None, 1);

        let p_unreach_other = mk_pat("computed", "computed", "none", "jmp reg<-computed");
        let g_unreach_other = mk_grp(vec![0x60], 0, None, 0);

        let survey = JumpSurvey {
            total_unresolved: 8,
            reachable_unresolved: 7,
            patterns: vec![
                (p_unreach_other, g_unreach_other),
                (p_boot_other, g_boot_other),
                (p_unreach_sw, g_unreach_sw),
                (p_crit_d2, g_crit_d2),
                (p_boot_sw, g_boot_sw),
                (p_crit_d0, g_crit_d0),
            ],
            ..Default::default()
        };

        let prio = survey.prioritized_patterns();
        let tiers: Vec<PriorityTier> = prio.iter().map(|p| p.tier).collect();
        assert_eq!(
            tiers,
            vec![
                PriorityTier::CriticalBootSwitch,
                PriorityTier::CriticalBootSwitch,
                PriorityTier::BootSwitch,
                PriorityTier::BootOther,
                PriorityTier::UnreachableSwitch,
                PriorityTier::UnreachableOther,
            ]
        );
        // Same reachable count (1) in CriticalBootSwitch -> shallower min_depth (0 < 2) wins.
        assert_eq!(prio[0].group.min_depth, Some(0));
        assert_eq!(prio[1].group.min_depth, Some(2));
        for w in prio.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
        assert_eq!(PriorityTier::CriticalBootSwitch.as_str(), "critical_boot_switch");
        assert_eq!(PriorityTier::BootSwitch.as_str(), "boot_switch");
        assert_eq!(PriorityTier::BootOther.as_str(), "boot_other");
        assert_eq!(PriorityTier::UnreachableSwitch.as_str(), "unreachable_switch");
        assert_eq!(PriorityTier::UnreachableOther.as_str(), "unreachable_other");
    }

    #[test]
    fn render_json_with_trace_sanitizes_host_paths_and_escapes_strings() {
        let s = JumpSurvey {
            total_unresolved: 1,
            reachable_unresolved: 1,
            by_reason: BTreeMap::from([("table_base_unknown", 1)]),
            by_reason_reachable: BTreeMap::from([("table_base_unknown", 1)]),
            by_role: BTreeMap::from([("switch", (1, 1))]),
            patterns: vec![(
                Pattern {
                    reason: "table_base_unknown",
                    role: "switch",
                    form: "jmp \"quoted\"\tform\x01".into(),
                    scale: "8".into(),
                    guard: "shift".into(),
                    location: "anchor:.rdata(r)\n\\".into(),
                },
                Group {
                    sites: vec![TEXT + 0x10],
                    reachable: 1,
                    functions: BTreeMap::from([(TEXT, 1)]),
                    min_depth: Some(0),
                    merge_predecessor_sites: 1,
                },
            )],
            ..Default::default()
        };
        let outcome = crate::trace_seed::SeedOutcome {
            pairs_in_range: 1,
            rows: vec![crate::trace_seed::TargetRow {
                target: TEXT + 0x20,
                sites: vec![TEXT + 0x4],
                verdict: crate::trace_seed::Verdict::Accepted,
                container: Some(TEXT),
                unresolved_jumps: vec![(TEXT + 0x10, UnresolvedJump::TableBaseUnknown)],
            }],
        };
        let diag = outcome.diagnostics();
        let correlated = crate::trace_seed::correlate_with_survey(&outcome, &s);
        assert_eq!(correlated.len(), 1);

        for host_path in [
            r"C:\Users\dev\secret\game.exe",
            r"\\fileserver\share\game.exe",
            "/home/dev/builds/game.exe",
            "file:///Users/dev/game.exe",
        ] {
            let json = render_json_with_trace(&s, 4, Some((host_path, &diag, &correlated, 10)));
            JsonValidator::validate(&json).expect("embedded trace JSON must be valid RFC 8259");
            assert!(
                !crate::summary::contains_host_path(&json),
                "host path leaked for {host_path}: {json}"
            );
            let txt = render_trace_correlation_text(host_path, &diag, &correlated, 10);
            assert!(
                !crate::summary::contains_host_path(&txt),
                "host path leaked in text for {host_path}: {txt}"
            );
        }
    }
}
