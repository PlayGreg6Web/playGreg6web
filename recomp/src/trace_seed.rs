//! Guided targets: indirect-call targets a boot actually dispatched (`playgta6_boot
//! --record-indirect-targets <file>`) fed back into the next lift as function entries.
//!
//! File format (text, one pair per line, hex; `#` starts a comment): `<site> <target>` where
//! `site` is the return address the guest `call` pushed (what `[rsp]` held at the dispatch) and
//! `target` the guest address that was dispatched. Addresses are absolute guest VAs, so a file
//! is only valid for the image layout it was recorded with; pairs outside a module's range are
//! ignored by that module.
//!
//! A target is only ever accepted when it is real code: inside an executable section of the
//! module, and an instruction boundary (of the discovered code when the address is already part
//! of it, else a decodable instruction). Nothing else is invented: rejected targets are only
//! reported. Accepted targets that are not yet function entries become discovery entry points;
//! the report also lists which targets land inside a function that holds an unresolved indirect
//! `jmp` (a switch the jump-table resolver did not handle, see `survey-jumps`).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;

use iced_x86::{Decoder, DecoderOptions};

use crate::cfg::{BasicBlock, UnresolvedJump};
use crate::code_image::{CodeImage, Region};
use crate::discover::Analysis;

/// One recorded dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TracePair {
    pub site: u64,
    pub target: u64,
}

/// Parsed seed file (sorted, deduplicated).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TraceSeeds {
    pub pairs: Vec<TracePair>,
}

fn parse_hex(text: &str) -> Option<u64> {
    let t = text
        .strip_prefix("0x")
        .or_else(|| text.strip_prefix("0X"))
        .unwrap_or(text);
    u64::from_str_radix(t, 16).ok()
}

/// Parses the file text; a malformed line is an error naming its number.
pub fn parse(text: &str) -> Result<TraceSeeds, String> {
    let mut set = BTreeSet::new();
    for (n, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let (Some(site), Some(target), None) = (it.next(), it.next(), it.next()) else {
            return Err(format!("line {}: expected '<site> <target>'", n + 1));
        };
        let (Some(site), Some(target)) = (parse_hex(site), parse_hex(target)) else {
            return Err(format!("line {}: expected hexadecimal addresses", n + 1));
        };
        set.insert(TracePair { site, target });
    }
    Ok(TraceSeeds {
        pairs: set.into_iter().collect(),
    })
}

pub fn load(path: &Path) -> Result<TraceSeeds, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read seed targets '{}': {e}", path.display()))?;
    parse(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// Why a target was or was not seeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// New function entry.
    Accepted,
    /// Already a discovered function entry (nothing to do).
    KnownEntry,
    NotExecutable,
    /// Inside discovered code but not at an instruction start.
    NotInstructionBoundary,
    /// Not part of discovered code and not a decodable instruction.
    Undecodable,
}

impl Verdict {
    pub fn name(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::KnownEntry => "known_entry",
            Self::NotExecutable => "not_executable",
            Self::NotInstructionBoundary => "not_instruction_boundary",
            Self::Undecodable => "undecodable",
        }
    }
}

/// One target of the module and what became of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetRow {
    pub target: u64,
    pub sites: Vec<u64>,
    pub verdict: Verdict,
    /// Function containing the target when it is inside discovered code (not its entry), and the
    /// unresolved indirect jumps of that function: `(jump va, reason)`. Non-empty means the
    /// target is very likely a case of one of those jumps.
    pub container: Option<u64>,
    pub unresolved_jumps: Vec<(u64, UnresolvedJump)>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SeedOutcome {
    /// Pairs that fell in this module's image range.
    pub pairs_in_range: usize,
    pub rows: Vec<TargetRow>,
}

impl SeedOutcome {
    /// Targets to add as discovery entry points.
    pub fn accepted(&self) -> Vec<u64> {
        self.rows
            .iter()
            .filter(|r| r.verdict == Verdict::Accepted)
            .map(|r| r.target)
            .collect()
    }
}

/// Judges every recorded target inside `image` against the analysis of the unseeded lift.
pub fn evaluate(image: &CodeImage<'_>, analysis: &Analysis, seeds: &TraceSeeds) -> SeedOutcome {
    // Block lookup by start VA; shared tails are copied into several functions, any copy has the
    // same instructions. Remember one owning function per block.
    let mut blocks: BTreeMap<u64, (&BasicBlock, u64)> = BTreeMap::new();
    for f in analysis.functions.values() {
        for (start, b) in &f.blocks {
            blocks.entry(*start).or_insert((b, f.entry));
        }
    }
    let mut by_target: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    let mut in_range = 0;
    for p in &seeds.pairs {
        if image.contains(p.target) {
            in_range += 1;
            by_target.entry(p.target).or_default().push(p.site);
        }
    }
    let mut rows = Vec::with_capacity(by_target.len());
    for (target, sites) in by_target {
        let mut row = TargetRow {
            target,
            sites,
            verdict: Verdict::Accepted,
            container: None,
            unresolved_jumps: Vec::new(),
        };
        let inside = blocks
            .range(..=target)
            .next_back()
            .map(|(_, v)| *v)
            .filter(|(b, _)| target < b.end);
        if analysis.functions.contains_key(&target) {
            row.verdict = Verdict::KnownEntry;
        } else if image.region_of(target) != Region::Code {
            row.verdict = Verdict::NotExecutable;
        } else if let Some((block, owner)) = inside {
            let boundary = block
                .instructions
                .binary_search_by_key(&target, |i| i.ip())
                .is_ok();
            if boundary {
                row.container = Some(owner);
                if let Some(f) = analysis.functions.get(&owner) {
                    row.unresolved_jumps = analysis
                        .unresolved_jumps
                        .iter()
                        .filter(|(va, _)| f.blocks.values().any(|b| b.contains(**va)))
                        .map(|(va, k)| (*va, *k))
                        .collect();
                }
            } else {
                row.verdict = Verdict::NotInstructionBoundary;
            }
        } else {
            let decodes = image.code_bytes(target).is_ok_and(|bytes| {
                let mut d = Decoder::with_ip(64, bytes, target, DecoderOptions::NONE);
                !d.decode().is_invalid()
            });
            if !decodes {
                row.verdict = Verdict::Undecodable;
            }
        }
        rows.push(row);
    }
    SeedOutcome {
        pairs_in_range: in_range,
        rows,
    }
}

/// One JSON line (no trailing newline) describing the module's outcome. Addresses and reason
/// names only.
pub fn report_line(module: &str, o: &SeedOutcome) -> String {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for r in &o.rows {
        *counts.entry(r.verdict.name()).or_default() += 1;
    }
    let mut s = String::new();
    let _ = write!(
        s,
        "{{\"module\":\"{}\",\"pairs_in_range\":{},\"targets\":{},\"verdicts\":{{",
        module.replace(['\\', '"'], "_"),
        o.pairs_in_range,
        o.rows.len()
    );
    for (i, (k, n)) in counts.iter().enumerate() {
        let _ = write!(s, "{}\"{k}\":{n}", if i > 0 { "," } else { "" });
    }
    s.push_str("},\"rejected\":[");
    let mut first = true;
    for r in o
        .rows
        .iter()
        .filter(|r| !matches!(r.verdict, Verdict::Accepted | Verdict::KnownEntry))
    {
        let _ = write!(
            s,
            "{}{{\"target\":\"0x{:x}\",\"reason\":\"{}\"}}",
            if first { "" } else { "," },
            r.target,
            r.verdict.name()
        );
        first = false;
    }
    s.push_str("],\"unresolved_jump_table_targets\":[");
    first = true;
    for r in o.rows.iter().filter(|r| !r.unresolved_jumps.is_empty()) {
        let jumps: Vec<String> = r
            .unresolved_jumps
            .iter()
            .map(|(va, k)| format!("{{\"jump\":\"0x{va:x}\",\"reason\":\"{}\"}}", k.name()))
            .collect();
        let sites: Vec<String> = r.sites.iter().map(|v| format!("\"0x{v:x}\"")).collect();
        let _ = write!(
            s,
            "{}{{\"target\":\"0x{:x}\",\"function\":\"0x{:x}\",\"sites\":[{}],\"jumps\":[{}]}}",
            if first { "" } else { "," },
            r.target,
            r.container.unwrap_or(0),
            sites.join(","),
            jumps.join(",")
        );
        first = false;
    }
    s.push_str("]}");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code_image::CodeSection;
    use crate::discover::discover;

    const BASE: u64 = 0x1_4000_0000;
    const TEXT: u64 = BASE + 0x1000;

    #[test]
    fn parses_pairs_comments_and_dedupes() {
        let s = parse("# hdr\n0x10 0x20\n0X10 20 # dup\n\n30 40\n").unwrap();
        assert_eq!(
            s.pairs,
            vec![
                TracePair {
                    site: 0x10,
                    target: 0x20
                },
                TracePair {
                    site: 0x30,
                    target: 0x40
                }
            ]
        );
    }

    #[test]
    fn rejects_malformed_lines() {
        assert!(parse("0x10\n").unwrap_err().contains("line 1"));
        assert!(parse("1 2\nzz 3\n").unwrap_err().contains("line 2"));
        assert!(parse("1 2 3\n").is_err());
    }

    fn image<'a>(code: &'a [u8], data: &'a [u8]) -> CodeImage<'a> {
        CodeImage::new(
            BASE,
            vec![
                CodeSection::new(TEXT, code, true),
                CodeSection::new(BASE + 0x2000, data, false),
            ],
            vec![TEXT],
        )
    }

    #[test]
    fn validates_targets() {
        // 0x0: xor eax,eax ; ret.  0x10: mov eax,imm32 ; ret (undiscovered).
        let mut c = vec![0x31, 0xC0, 0xC3];
        c.resize(0x10, 0xCC);
        c.extend([0xB8, 0x78, 0x56, 0x34, 0x12, 0xC3]);
        let img = image(&c, &[0xC3; 8]);
        let analysis = discover(&img);
        let seeds = TraceSeeds {
            pairs: [TEXT, TEXT + 0x10, BASE + 0x2000, BASE + 0x9_0000]
                .iter()
                .map(|&target| TracePair { site: 1, target })
                .collect(),
        };
        let o = evaluate(&img, &analysis, &seeds);
        assert_eq!(o.pairs_in_range, 3);
        let v: Vec<_> = o.rows.iter().map(|r| (r.target, r.verdict)).collect();
        assert!(v.contains(&(TEXT, Verdict::KnownEntry)));
        assert!(v.contains(&(TEXT + 0x10, Verdict::Accepted)));
        assert!(v.contains(&(BASE + 0x2000, Verdict::NotExecutable)));
        assert!(!v.iter().any(|(t, _)| *t == BASE + 0x9_0000));
    }

    #[test]
    fn mid_instruction_in_discovered_code_is_rejected() {
        // mov eax, imm32 ; ret as the entry: TEXT+1 is inside the mov.
        let c = vec![0xB8, 0x78, 0x56, 0x34, 0x12, 0xC3];
        let img = image(&c, &[]);
        let analysis = discover(&img);
        let seeds = TraceSeeds {
            pairs: vec![
                TracePair {
                    site: 1,
                    target: TEXT + 1,
                },
                TracePair {
                    site: 1,
                    target: TEXT + 5,
                },
            ],
        };
        let o = evaluate(&img, &analysis, &seeds);
        let verdict = |t| o.rows.iter().find(|r| r.target == t).unwrap().verdict;
        assert_eq!(verdict(TEXT + 1), Verdict::NotInstructionBoundary);
        assert_eq!(verdict(TEXT + 5), Verdict::Accepted);
        assert_eq!(o.accepted(), vec![TEXT + 5]);
        assert_eq!(
            o.rows
                .iter()
                .find(|r| r.target == TEXT + 5)
                .unwrap()
                .container,
            Some(TEXT)
        );
    }

    #[test]
    fn report_lists_rejections_and_jump_table_candidates() {
        let o = SeedOutcome {
            pairs_in_range: 2,
            rows: vec![
                TargetRow {
                    target: 0x20,
                    sites: vec![0x5],
                    verdict: Verdict::NotExecutable,
                    container: None,
                    unresolved_jumps: vec![],
                },
                TargetRow {
                    target: 0x30,
                    sites: vec![0x6],
                    verdict: Verdict::Accepted,
                    container: Some(0x28),
                    unresolved_jumps: vec![(0x2a, UnresolvedJump::TableUnbounded)],
                },
            ],
        };
        let line = report_line("game.exe", &o);
        assert!(line.contains("\"reason\":\"not_executable\""), "{line}");
        assert!(
            line.contains("\"unresolved_jump_table_targets\":[{\"target\":\"0x30\""),
            "{line}"
        );
        assert!(line.contains("\"jump\":\"0x2a\""), "{line}");
        assert!(!line.contains('\n'));
    }
}
