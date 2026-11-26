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

use iced_x86::{ConditionCode, Instruction, Mnemonic, OpKind, Register};

use crate::cfg::{Edge, Function, Terminator, UnresolvedJump};
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

#[derive(Debug, Clone, Default)]
pub struct Group {
    pub sites: Vec<u64>,
    pub reachable: usize,
    /// Reachable sites by function entry.
    pub functions: BTreeMap<u64, usize>,
}

#[derive(Debug, Clone)]
pub struct Location {
    pub function: u64,
    pub depth: u32,
    pub callers: usize,
    pub sites: usize,
    pub patterns: BTreeSet<usize>,
}

#[derive(Debug, Default)]
pub struct JumpSurvey {
    pub total_unresolved: usize,
    pub reachable_unresolved: usize,
    pub functions_total: usize,
    pub functions_reachable: usize,
    /// Unresolved jumps whose pointer slot is statically initialised (target seeded as an entry).
    pub pointer_slot_jumps: usize,
    pub by_reason: BTreeMap<&'static str, usize>,
    /// Ranked: reachable desc, total desc, label.
    pub patterns: Vec<(Pattern, Group)>,
    pub locations: Vec<Location>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Def {
    Load { scale: u8, loc: String },
    RelOffset { scale: u8, loc: String },
    Anchor(u64),
    LiveIn,
    Other,
}

struct Ctx<'a> {
    layout: &'a Layout,
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
fn classify(ctx: &Ctx<'_>, reason: UnresolvedJump, ins: &[&Instruction]) -> Pattern {
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
            Def::LiveIn => ("jmp reg<-live_in", "0".into(), "none".into(), false),
            Def::Other => ("jmp reg<-computed", "0".into(), "none".into(), false),
        };
        form = f.to_string();
        scale = s;
        location = l;
        indexed = ix;
    }
    let role = if indexed {
        "switch"
    } else if form.starts_with("jmp [") || form.contains("load") || form.contains("live_in") {
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
fn window<'a>(
    func: &'a Function,
    preds: &BTreeMap<u64, Vec<u64>>,
    start: u64,
) -> Vec<&'a Instruction> {
    let mut chain = vec![start];
    let mut cur = start;
    while chain.len() < MAX_WINDOW_BLOCKS {
        match preds.get(&cur).map(Vec::as_slice) {
            Some([p]) if !chain.contains(p) => {
                chain.push(*p);
                cur = *p;
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
    out
}

/// Builds the survey from a finished analysis. `roots` are the VAs the boot starts at;
/// `top` bounds the number of listed locations.
pub fn survey_jumps(layout: &Layout, analysis: &Analysis, roots: &[u64], top: usize) -> JumpSurvey {
    let ctx = Ctx { layout };
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
            let ins = window(f, preds, *start);
            let g = groups.entry(classify(&ctx, reason, &ins)).or_default();
            g.sites.push(site);
            survey.total_unresolved += 1;
            *survey.by_reason.entry(reason.name()).or_default() += 1;
            if depth.contains_key(entry) {
                g.reachable += 1;
                survey.reachable_unresolved += 1;
                *g.functions.entry(*entry).or_default() += 1;
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
        let _ = writeln!(o, "  {k}: {v}");
    }
    let _ = writeln!(o, "patterns (ranked by reachable count):");
    for (i, (p, g)) in s.patterns.iter().enumerate() {
        let ex: Vec<String> = g
            .sites
            .iter()
            .take(examples)
            .map(|a| format!("{a:#x}"))
            .collect();
        let _ = writeln!(
            o,
            "  #{} total={} reachable={} :: {}\n      e.g. {}",
            i + 1,
            g.sites.len(),
            g.reachable,
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

/// Boot roots: the PE entry point and TLS callbacks.
pub fn boot_roots(pe: &Pe64) -> Vec<u64> {
    let mut v = vec![pe.optional.image_base + u64::from(pe.optional.entry_point_rva)];
    if let Some(t) = &pe.tls {
        v.extend(t.callbacks.iter().map(|c| c.va));
    }
    v
}

/// `survey-jumps <pe-path> [--top N] [--examples N] [--no-refine] [--out <file>]`.
pub const USAGE: &str =
    "survey-jumps <pe-path> [--top N] [--examples N] [--no-refine] [--out <file>]";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    pub pe_path: std::path::PathBuf,
    pub top: usize,
    pub examples: usize,
    pub no_refine: bool,
    pub out: Option<std::path::PathBuf>,
}

pub fn parse_args(args: &[String]) -> Result<Option<Args>, String> {
    let mut out = Args {
        pe_path: Default::default(),
        top: 40,
        examples: 4,
        no_refine: false,
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
            "--top" => out.top = num(&mut i, "--top")?,
            "--examples" => out.examples = num(&mut i, "--examples")?,
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
    let s = survey_jumps(&Layout::from_pe(&pe), &analysis, &boot_roots(&pe), args.top);
    let text = render(&s, args.examples);
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
    fn scaled_load_from_loaded_pointer_is_a_switch() {
        let an = analyse(|a| {
            a.mov(rdx, qword_ptr(rcx + 0x8)).unwrap();
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
        assert!(parse_args(&["--bogus".into()]).is_err());
    }
}
