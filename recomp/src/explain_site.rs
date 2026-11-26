//! `explain-site`: read-only explanation of one indirect `jmp`/`call` site, for fault triage.
//!
//! Given a guest address (an indirect site, or the return address of an indirect call), runs the
//! same discovery as `survey-jumps` and prints the site's instruction window (mnemonics and
//! operands only, no raw bytes), the definitions reaching its index/base registers, the reason
//! the jump-table matcher left it on run-time dispatch (see [`crate::cfg::UnresolvedJump`]), the
//! survey pattern class and a suggested fix category. It never changes resolution behaviour.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use iced_x86::{Formatter, Instruction, IntelFormatter, Mnemonic, OpKind, Register};

use crate::cfg::{Function, UnresolvedJump};
use crate::code_image::CodeImage;
use crate::discover::Analysis;
use crate::jump_survey::{classify, is_win64_volatile, window, Ctx, Layout, TableProbe};
use crate::pe::Pe64;

pub const USAGE: &str =
    "explain-site <pe-path> <addr>[,<addr>...] [--return-address] [--no-refine]";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    pub pe_path: std::path::PathBuf,
    pub addr: u64,
    /// Further sites from a comma-separated `<addr>` list, explained after `addr` with the same
    /// (single) discovery run.
    pub more: Vec<u64>,
    /// `addr` is the return address of an indirect call (the site is the instruction before it).
    pub return_address: bool,
    pub no_refine: bool,
}

pub fn parse_addr(s: &str) -> Option<u64> {
    let t = s.trim();
    match t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        Some(h) => u64::from_str_radix(h, 16).ok(),
        None => t.parse().ok(),
    }
}

pub fn parse_args(args: &[String]) -> Result<Option<Args>, String> {
    let mut path = None;
    let mut addr = None;
    let mut out = Args {
        pe_path: Default::default(),
        addr: 0,
        more: Vec::new(),
        return_address: false,
        no_refine: false,
    };
    for a in args {
        match a.as_str() {
            "-h" | "--help" => return Ok(None),
            "--return-address" => out.return_address = true,
            "--no-refine" => out.no_refine = true,
            f if f.starts_with('-') => {
                return Err(format!("unknown option '{f}' for 'explain-site'"));
            }
            p if path.is_none() => path = Some(p.into()),
            p if addr.is_none() => {
                for (n, part) in p.split(',').filter(|t| !t.trim().is_empty()).enumerate() {
                    let v = parse_addr(part).ok_or_else(|| format!("bad address '{part}'"))?;
                    if n == 0 {
                        addr = Some(v);
                    } else {
                        out.more.push(v);
                    }
                }
            }
            p => return Err(format!("unexpected extra positional argument '{p}'")),
        }
    }
    out.pe_path = path.ok_or_else(|| "missing input PE path <pe-path>".to_string())?;
    out.addr = addr.ok_or_else(|| "missing <addr>".to_string())?;
    Ok(Some(out))
}

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
    let layout = Layout::from_pe(&pe);
    let mut text = String::new();
    for (n, &addr) in std::iter::once(&args.addr).chain(&args.more).enumerate() {
        if n > 0 {
            text.push_str(
                "
----
",
            );
        }
        text.push_str(&explain_probed(
            &layout,
            &analysis,
            addr,
            args.return_address,
            Some(&seeded.image),
        ));
    }
    Ok(text)
}

fn is_indirect(i: &Instruction) -> Option<&'static str> {
    let kind = match i.mnemonic() {
        Mnemonic::Jmp => "jmp",
        Mnemonic::Call => "call",
        _ => return None,
    };
    matches!(i.op0_kind(), OpKind::Register | OpKind::Memory).then_some(kind)
}

fn fmt(i: &Instruction) -> String {
    let mut f = IntelFormatter::new();
    let mut s = String::new();
    f.format(i, &mut s);
    s
}

/// Registers an instruction reads for its address or operand value, excluding rip/rsp.
fn read_regs(i: &Instruction) -> Vec<Register> {
    let mut v = Vec::new();
    let mut push = |r: Register| {
        let r = r.full_register();
        if r != Register::None && r != Register::RIP && r != Register::RSP && !v.contains(&r) {
            v.push(r);
        }
    };
    for n in 0..i.op_count() {
        match i.op_kind(n) {
            OpKind::Memory => {
                push(i.memory_base());
                push(i.memory_index());
            }
            OpKind::Register if n > 0 => push(i.op_register(n)),
            _ => {}
        }
    }
    // `jmp reg` / `call reg`, and read-modify-write on op0 (`add rax, rdx`).
    if i.op0_kind() == OpKind::Register
        && (is_indirect(i).is_some()
            || matches!(
                i.mnemonic(),
                Mnemonic::Add | Mnemonic::Sub | Mnemonic::Or | Mnemonic::And | Mnemonic::Xor
            ))
    {
        push(i.op0_register());
    }
    v
}

fn writes_reg(i: &Instruction, reg: Register) -> bool {
    if is_indirect(i) == Some("call") || i.mnemonic() == Mnemonic::Call {
        // Win64: a call returns in RAX and may clobber every other volatile register.
        return is_win64_volatile(reg);
    }
    !matches!(
        i.mnemonic(),
        Mnemonic::Cmp
            | Mnemonic::Test
            | Mnemonic::Push
            | Mnemonic::Jmp
            | Mnemonic::Bt
            | Mnemonic::Call
    ) && i.op_count() > 0
        && i.op0_kind() == OpKind::Register
        && i.op0_register().full_register() == reg
}

fn defs(o: &mut String, ins: &[&Instruction], upto: usize, reg: Register, depth: usize) {
    let pad = "  ".repeat(depth + 1);
    let reg = reg.full_register();
    for idx in (0..upto).rev() {
        let i = ins[idx];
        if writes_reg(i, reg) {
            if i.mnemonic() == Mnemonic::Call {
                let what = if reg == Register::RAX {
                    "return value"
                } else {
                    "clobbered"
                };
                let _ = writeln!(o, "{pad}{reg:?} <- {:#x}: {} ({what})", i.ip(), fmt(i));
                return;
            }
            let _ = writeln!(o, "{pad}{reg:?} <- {:#x}: {}", i.ip(), fmt(i));
            let zeroing = i.mnemonic() == Mnemonic::Xor
                && i.op1_kind() == OpKind::Register
                && i.op1_register().full_register() == reg;
            if depth < 3 && !zeroing {
                for r in read_regs(i) {
                    if r != reg {
                        defs(o, ins, idx, r, depth + 1);
                    }
                }
            }
            return;
        }
    }
    let _ = writeln!(o, "{pad}{reg:?} <- live-in (not written in the window)");
}

/// Why the jump-table matcher rejected the site, and which fix family it belongs to.
fn rule(reason: UnresolvedJump) -> (&'static str, &'static str) {
    match reason {
        UnresolvedJump::MemoryOperand => (
            "no table shape: jmp [reg+disp] is a function-pointer read (vtable/callback slot)",
            "runtime-dispatch",
        ),
        UnresolvedJump::LoadedPointer => (
            "no table shape: jmp reg with reg loaded from memory without a scaled index",
            "runtime-dispatch",
        ),
        UnresolvedJump::TableBaseUnknown => (
            "table-shaped read ([base+index*4/8]) but the base is not a proven constant on every path",
            "base-proof",
        ),
        UnresolvedJump::TableUnbounded => (
            "table base proven but the index has no proven bound (no cmp/ja guard, mask, shr or movzx) and no .pdata-derived extent",
            "bound-proof",
        ),
        UnresolvedJump::RegisterLiveIn => (
            "jump register is not written in the slice (set before a merge or passed in)",
            "runtime-dispatch",
        ),
        UnresolvedJump::Computed => (
            "jump target is some other computation that matches no rule",
            "new-shape",
        ),
    }
}

fn category(cat: &str, kind: &str) -> &'static str {
    if kind == "call" {
        return "seed-entry: indirect call (vtable/callback); the dispatcher needs the observed target as a function entry (--entry / dispatched-target feedback); no recompiler change";
    }
    match cat {
        "runtime-dispatch" => {
            "seed-entry: tail call through a pointer; seed the observed target as a function entry; no jump_table change"
        }
        "base-proof" => {
            "jump_table-fix: extend the base reaching-definition proof (see the reaching defs above: merge, load or clobber of the base register)"
        }
        "bound-proof" => {
            "jump_table-fix: add an index-bound shape (guard/mask) or infer the table extent from .pdata or a table terminator"
        }
        "resolved" => "none: this site resolves; the gap is elsewhere (check the call stack / dispatched entries)",
        _ => "new-shape: unknown dispatch computation; read the window above and add a survey/jump_table shape or a hand-seeded entry",
    }
}

/// Finds `(function, block start, instruction ip)` of the instruction at `addr`
/// (or the one ending at `addr` when `ret`).
fn locate(analysis: &Analysis, addr: u64, ret: bool) -> Option<(&Function, u64, u64)> {
    for f in analysis.functions.values() {
        for (start, b) in &f.blocks {
            for i in &b.instructions {
                let hit = if ret {
                    i.next_ip() == addr
                } else {
                    i.ip() == addr
                };
                if hit {
                    return Some((f, *start, i.ip()));
                }
            }
        }
    }
    None
}

pub fn explain(layout: &Layout, analysis: &Analysis, addr: u64, ret: bool) -> String {
    explain_probed(layout, analysis, addr, ret, None)
}

/// [`explain`] that reads the leading entries of a constant table from `image`, so a tail call
/// through a function-pointer or run-time data table is not reported as a switch gap.
pub fn explain_probed(
    layout: &Layout,
    analysis: &Analysis,
    addr: u64,
    ret: bool,
    image: Option<&CodeImage<'_>>,
) -> String {
    let mut o = String::new();
    let Some((func, bstart, site)) = locate(analysis, addr, ret) else {
        let _ = writeln!(
            o,
            "address {addr:#x} is not inside any discovered function's decoded code ({} functions known)",
            analysis.functions.len()
        );
        let _ = writeln!(
            o,
            "category: seed-entry: seed a function entry covering it (--entry), then re-run"
        );
        return o;
    };
    let mut preds: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    for (s, blk) in &func.blocks {
        for t in blk.intra_successors() {
            preds.entry(t).or_default().push(*s);
        }
    }
    let mut ins: Vec<&Instruction> = window(func, &preds, bstart);
    let pos = ins
        .iter()
        .position(|i| i.ip() == site)
        .unwrap_or(ins.len() - 1);
    ins.truncate(pos + 1);
    let jmp = ins[pos];
    let _ = writeln!(
        o,
        "site {site:#x} in sub_{:X}+{:#x}: {}",
        func.entry,
        site - func.entry,
        fmt(jmp)
    );
    let Some(kind) = is_indirect(jmp) else {
        let _ = writeln!(
            o,
            "not an indirect jmp/call; unresolved indirect jumps in this function:"
        );
        for (s, why) in &analysis.unresolved_jumps {
            let inside = func
                .blocks
                .values()
                .any(|b| b.instructions.last().is_some_and(|l| l.ip() == *s));
            if inside {
                let _ = writeln!(o, "  {s:#x} {}", why.name());
            }
        }
        let _ = writeln!(o, "category: inspect: no indirect site at this address");
        return o;
    };
    let _ = writeln!(o, "kind: indirect {kind}");
    let _ = writeln!(
        o,
        "window ({} instructions, from {:#x}):",
        ins.len(),
        ins[0].ip()
    );
    for i in &ins {
        let mark = if i.ip() == site { "   <== site" } else { "" };
        let _ = writeln!(o, "  {:#x}  {}{mark}", i.ip(), fmt(i));
    }
    let _ = writeln!(o, "reaching definitions of the site's registers:");
    let regs = read_regs(jmp);
    if regs.is_empty() {
        let _ = writeln!(o, "  (none: rip-relative or absolute operand)");
    }
    for r in regs {
        defs(&mut o, &ins, pos, r, 0);
    }
    let ctx = Ctx {
        layout,
        probe: image.map(|image| TableProbe {
            image,
            functions: &analysis.functions,
        }),
    };
    let reason = if kind == "jmp" {
        analysis.unresolved_jumps.get(&site).copied()
    } else {
        None
    };
    let pat = classify(&ctx, reason.unwrap_or(UnresolvedJump::Computed), &ins);
    let cat;
    match reason {
        Some(r) => {
            let (why, c) = rule(r);
            let _ = writeln!(o, "jump_table: not resolved, reason {}: {why}", r.name());
            cat = match pat.role {
                "fn_table" => {
                    let _ = writeln!(
                        o,
                        "table: read-only, its leading entries are function entries (a handler table, not case labels)"
                    );
                    "runtime-dispatch"
                }
                "ptr_array" => {
                    let _ = writeln!(
                        o,
                        "table: base loaded from object/global memory (a vtable slot or pointer array picked by an index, not a switch table)"
                    );
                    "runtime-dispatch"
                }
                "data_table" => {
                    let _ = writeln!(
                        o,
                        "table: in a writable section (pointers stored at run time, not a switch table)"
                    );
                    "runtime-dispatch"
                }
                _ if pat.form == "jmp reg<-call" => {
                    let _ = writeln!(o, "target: returned by the preceding call");
                    "runtime-dispatch"
                }
                _ => c,
            };
        }
        None if kind == "jmp" => {
            if let Some(t) = analysis.jump_tables.get(&site) {
                let _ = writeln!(
                    o,
                    "jump_table: RESOLVED ({} case targets); this site is not the gap",
                    t.targets.len()
                );
                cat = "resolved";
            } else if let Some((slot, tgt)) = analysis.pointer_slots.get(&site) {
                let _ = writeln!(
                    o,
                    "jump_table: pointer-slot tail call, slot {slot:#x}, seeded target {tgt:#x}"
                );
                cat = "runtime-dispatch";
            } else {
                let _ = writeln!(
                    o,
                    "jump_table: no unresolved record (import slot or resolved elsewhere)"
                );
                cat = "runtime-dispatch";
            }
        }
        None => {
            let _ = writeln!(
                o,
                "jump_table: not applicable to indirect calls (never table-resolved)"
            );
            cat = "runtime-dispatch";
        }
    }
    let _ = writeln!(o, "survey pattern: {}", pat.label());
    let _ = writeln!(o, "category: {}", category(cat, kind));
    o
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
    fn unbounded_loaded_table_reports_reason_defs_and_category() {
        let an = analyse(|a| {
            a.mov(rdx, qword_ptr(rsp + 0x8)).unwrap();
            a.mov(rax, qword_ptr(rdx + rax * 8)).unwrap();
            a.jmp(rax).unwrap();
        });
        let site = *an.unresolved_jumps.keys().next().unwrap();
        let t = explain(&layout(), &an, site, false);
        assert!(t.contains("indirect jmp"), "{t}");
        assert!(t.contains("<== site"), "{t}");
        assert!(t.contains("RAX <-"), "{t}");
        assert!(t.contains("RDX <-"), "{t}");
        assert!(t.contains("not resolved, reason "), "{t}");
        assert!(t.contains("switch"), "{t}");
        assert!(t.contains("category: jump_table-fix"), "{t}");
    }

    #[test]
    fn vtable_tail_call_is_a_seed_entry() {
        let an = analyse(|a| {
            a.jmp(qword_ptr(rax + 0x10)).unwrap();
        });
        let site = *an.unresolved_jumps.keys().next().unwrap();
        let t = explain(&layout(), &an, site, false);
        assert!(t.contains("memory_operand"), "{t}");
        assert!(t.contains("category: seed-entry"), "{t}");
        assert!(t.contains("tail_call"), "{t}");
    }

    #[test]
    fn indirect_call_found_by_return_address() {
        let an = analyse(|a| {
            a.mov(rax, qword_ptr(rcx)).unwrap();
            a.call(qword_ptr(rax + 0x18)).unwrap();
            a.ret().unwrap();
        });
        let mut ret = 0;
        for f in an.functions.values() {
            for b in f.blocks.values() {
                for i in &b.instructions {
                    if i.mnemonic() == Mnemonic::Call {
                        ret = i.next_ip();
                    }
                }
            }
        }
        assert_ne!(ret, 0);
        let t = explain(&layout(), &an, ret, true);
        assert!(t.contains("indirect call"), "{t}");
        assert!(t.contains("RAX <-"), "{t}");
        assert!(t.contains("category: seed-entry: indirect call"), "{t}");
    }

    #[test]
    fn unknown_address_and_args() {
        let an = analyse(|a| {
            a.ret().unwrap();
        });
        assert!(explain(&layout(), &an, 0x1234, false).contains("not inside"));
        let a = parse_args(&[
            "x.exe".into(),
            "0x1400010a0".into(),
            "--return-address".into(),
        ])
        .unwrap()
        .unwrap();
        assert_eq!(a.addr, 0x1_4000_10a0);
        assert!(a.return_address);
        assert!(parse_args(&["x.exe".into()]).is_err());
        let m = parse_args(&["x.exe".into(), "0x10,0x20,0x30".into()])
            .unwrap()
            .unwrap();
        assert_eq!((m.addr, m.more.as_slice()), (0x10, &[0x20, 0x30][..]));
        assert!(parse_args(&["x.exe".into(), "0x10,zz".into()]).is_err());
    }
}
