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

    /// Computes structured, deterministic diagnostics summarizing verdicts, unresolved-jump
    /// candidates, and rejected targets.
    pub fn diagnostics(&self) -> SeedDiagnostics {
        let mut verdict_counts: BTreeMap<&'static str, usize> = BTreeMap::new();
        let mut accepted_in_discovered_code = 0usize;
        let mut accepted_new_entries = 0usize;
        let mut rejected_targets: Vec<(u64, Verdict)> = Vec::new();

        #[derive(Default)]
        struct JumpAcc {
            reason: Option<UnresolvedJump>,
            hit_targets: BTreeSet<u64>,
            distinct_sites: BTreeSet<u64>,
            accepted_targets: BTreeSet<u64>,
            known_targets: BTreeSet<u64>,
        }

        let mut by_jump: BTreeMap<(u64, u64), JumpAcc> = BTreeMap::new();

        for row in &self.rows {
            *verdict_counts.entry(row.verdict.name()).or_default() += 1;
            match row.verdict {
                Verdict::Accepted => {
                    if row.container.is_some() {
                        accepted_in_discovered_code += 1;
                    } else {
                        accepted_new_entries += 1;
                    }
                }
                Verdict::KnownEntry => {}
                other => {
                    rejected_targets.push((row.target, other));
                }
            }

            if !row.unresolved_jumps.is_empty() {
                let func = row.container.unwrap_or(0);
                for &(jump_va, reason) in &row.unresolved_jumps {
                    let acc = by_jump.entry((func, jump_va)).or_default();
                    acc.reason.get_or_insert(reason);
                    acc.hit_targets.insert(row.target);
                    acc.distinct_sites.extend(row.sites.iter().copied());
                    match row.verdict {
                        Verdict::Accepted => {
                            acc.accepted_targets.insert(row.target);
                        }
                        Verdict::KnownEntry => {
                            acc.known_targets.insert(row.target);
                        }
                        _ => {}
                    }
                }
            }
        }

        rejected_targets.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| a.1.name().cmp(b.1.name()))
        });

        let mut jump_candidates: Vec<JumpCandidateSummary> = by_jump
            .into_iter()
            .filter_map(|((function, jump_va), acc)| {
                let reason = acc.reason?;
                Some(JumpCandidateSummary {
                    function,
                    jump_va,
                    reason,
                    hit_targets: acc.hit_targets.into_iter().collect(),
                    distinct_sites: acc.distinct_sites,
                    accepted_targets: acc.accepted_targets.len(),
                    known_targets: acc.known_targets.len(),
                })
            })
            .collect();

        jump_candidates.sort_by(|a, b| {
            b.accepted_targets
                .cmp(&a.accepted_targets)
                .then_with(|| b.hit_targets.len().cmp(&a.hit_targets.len()))
                .then_with(|| b.distinct_sites.len().cmp(&a.distinct_sites.len()))
                .then_with(|| a.function.cmp(&b.function))
                .then_with(|| a.jump_va.cmp(&b.jump_va))
        });

        SeedDiagnostics {
            pairs_in_range: self.pairs_in_range,
            distinct_targets: self.rows.len(),
            verdict_counts,
            accepted_in_discovered_code,
            accepted_new_entries,
            jump_candidates,
            rejected_targets,
        }
    }
}

/// Aggregated trace-seed hits associated with one unresolved indirect jump.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JumpCandidateSummary {
    pub function: u64,
    pub jump_va: u64,
    pub reason: UnresolvedJump,
    /// Distinct target VAs hitting this candidate (sorted ascending, deduplicated).
    pub hit_targets: Vec<u64>,
    /// Distinct dispatch return-site VAs recorded across all hitting targets.
    pub distinct_sites: BTreeSet<u64>,
    pub accepted_targets: usize,
    pub known_targets: usize,
}

/// Structured, deterministic diagnostics for a [`SeedOutcome`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SeedDiagnostics {
    pub pairs_in_range: usize,
    pub distinct_targets: usize,
    pub verdict_counts: BTreeMap<&'static str, usize>,
    pub accepted_in_discovered_code: usize,
    pub accepted_new_entries: usize,
    pub jump_candidates: Vec<JumpCandidateSummary>,
    pub rejected_targets: Vec<(u64, Verdict)>,
}

/// Correlation between a live trace-seed unresolved-jump candidate and its classified
/// [`crate::jump_survey::Pattern`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorrelatedJumpHit {
    pub function: u64,
    pub jump_va: u64,
    pub reason: UnresolvedJump,
    /// 1-based index of the matching pattern in [`crate::jump_survey::JumpSurvey::patterns`].
    pub pattern_rank: usize,
    pub pattern_label: String,
    pub role: &'static str,
    pub guard: String,
    /// Whether `function` is recorded as boot-reachable in the matching pattern group.
    pub boot_reachable: bool,
    pub hit_targets: Vec<u64>,
    pub distinct_sites: BTreeSet<u64>,
    pub accepted_targets: usize,
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

fn push_hex_array(out: &mut String, values: impl IntoIterator<Item = u64>) {
    out.push('[');
    for (i, v) in values.into_iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        let _ = write!(out, "\"0x{v:x}\"");
    }
    out.push(']');
}

impl SeedDiagnostics {
    /// Deterministic, sanitized plain-text diagnostics report.
    pub fn render_text(&self, module: &str, top: usize) -> String {
        let safe_module = crate::summary::sanitize_label(module);
        let mut out = String::new();
        let _ = writeln!(out, "trace-seed diagnostics ({safe_module})");
        let _ = writeln!(
            out,
            "pairs_in_range={} distinct_targets={} accepted_in_discovered_code={} accepted_new_entries={} jump_candidates={} rejected={}",
            self.pairs_in_range,
            self.distinct_targets,
            self.accepted_in_discovered_code,
            self.accepted_new_entries,
            self.jump_candidates.len(),
            self.rejected_targets.len()
        );
        let _ = writeln!(out, "verdicts:");
        if self.verdict_counts.is_empty() {
            let _ = writeln!(out, "  none");
        } else {
            for (k, v) in &self.verdict_counts {
                let _ = writeln!(out, "  {k}: {v}");
            }
        }
        let _ = writeln!(out, "unresolved jump candidates (top {} of {}):", top.min(self.jump_candidates.len()), self.jump_candidates.len());
        for (idx, c) in self.jump_candidates.iter().take(top).enumerate() {
            let targets: Vec<String> = c
                .hit_targets
                .iter()
                .take(8)
                .map(|t| format!("{t:#x}"))
                .collect();
            let extra = if c.hit_targets.len() > 8 {
                format!(" (+{} more)", c.hit_targets.len() - 8)
            } else {
                String::new()
            };
            let _ = writeln!(
                out,
                "  #{} fn {:#x} jump {:#x} reason={} accepted={} known={} hits={} sites={} targets=[{}{}]",
                idx + 1,
                c.function,
                c.jump_va,
                c.reason.name(),
                c.accepted_targets,
                c.known_targets,
                c.hit_targets.len(),
                c.distinct_sites.len(),
                targets.join(", "),
                extra
            );
        }
        if !self.rejected_targets.is_empty() {
            let _ = writeln!(
                out,
                "rejected targets (top {} of {}):",
                top.min(self.rejected_targets.len()),
                self.rejected_targets.len()
            );
            for (target, verdict) in self.rejected_targets.iter().take(top) {
                let _ = writeln!(out, "  {target:#x} reason={}", verdict.name());
            }
        }
        out
    }

    /// Deterministic, strict RFC 8259 JSON diagnostics report with sanitized module label.
    pub fn render_json(&self, module: &str, top: usize) -> String {
        let safe_module = crate::summary::sanitize_label(module);
        let mut out = String::new();
        out.push_str("{\n");
        out.push_str("  \"module\": ");
        push_json_str(&mut out, &safe_module);
        out.push_str(",\n");
        let _ = writeln!(out, "  \"pairs_in_range\": {},", self.pairs_in_range);
        let _ = writeln!(out, "  \"distinct_targets\": {},", self.distinct_targets);
        let _ = writeln!(
            out,
            "  \"accepted_in_discovered_code\": {},",
            self.accepted_in_discovered_code
        );
        let _ = writeln!(
            out,
            "  \"accepted_new_entries\": {},",
            self.accepted_new_entries
        );

        out.push_str("  \"verdicts\": {");
        for (i, (k, v)) in self.verdict_counts.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            push_json_str(&mut out, k);
            let _ = write!(out, ": {v}");
        }
        out.push_str("},\n");

        let _ = writeln!(
            out,
            "  \"jump_candidates_total\": {},",
            self.jump_candidates.len()
        );
        out.push_str("  \"jump_candidates\": [");
        let shown_cands = self.jump_candidates.len().min(top);
        for (i, c) in self.jump_candidates.iter().take(shown_cands).enumerate() {
            out.push_str(if i == 0 { "\n" } else { ",\n" });
            out.push_str("    {");
            let _ = write!(
                out,
                "\"function\": \"0x{:x}\", \"jump_va\": \"0x{:x}\", \"reason\": ",
                c.function, c.jump_va
            );
            push_json_str(&mut out, c.reason.name());
            let _ = write!(
                out,
                ", \"accepted_targets\": {}, \"known_targets\": {}, \"hit_targets\": ",
                c.accepted_targets, c.known_targets
            );
            push_hex_array(&mut out, c.hit_targets.iter().copied());
            out.push_str(", \"distinct_sites\": ");
            push_hex_array(&mut out, c.distinct_sites.iter().copied());
            out.push('}');
        }
        if shown_cands == 0 {
            out.push_str("],\n");
        } else {
            out.push_str("\n  ],\n");
        }

        let _ = writeln!(out, "  \"rejected_total\": {},", self.rejected_targets.len());
        out.push_str("  \"rejected_targets\": [");
        let shown_rej = self.rejected_targets.len().min(top);
        for (i, (target, verdict)) in self.rejected_targets.iter().take(shown_rej).enumerate() {
            out.push_str(if i == 0 { "\n" } else { ",\n" });
            out.push_str("    {\"target\": ");
            let _ = write!(out, "\"0x{target:x}\"");
            out.push_str(", \"reason\": ");
            push_json_str(&mut out, verdict.name());
            out.push('}');
        }
        if shown_rej == 0 {
            out.push_str("]\n");
        } else {
            out.push_str("\n  ]\n");
        }
        out.push_str("}\n");
        out
    }
}

/// Correlates unresolved-jump candidates in `outcome` with their classified patterns in `survey`.
pub fn correlate_with_survey(
    outcome: &SeedOutcome,
    survey: &crate::jump_survey::JumpSurvey,
) -> Vec<CorrelatedJumpHit> {
    let mut by_site: BTreeMap<u64, (usize, &crate::jump_survey::Pattern, &crate::jump_survey::Group)> =
        BTreeMap::new();
    for (idx, (pattern, group)) in survey.patterns.iter().enumerate() {
        let rank = idx + 1;
        for &site in &group.sites {
            by_site.entry(site).or_insert((rank, pattern, group));
        }
    }

    let diag = outcome.diagnostics();
    let mut hits = Vec::new();
    for cand in diag.jump_candidates {
        let Some(&(pattern_rank, pattern, group)) = by_site.get(&cand.jump_va) else {
            continue;
        };
        hits.push(CorrelatedJumpHit {
            function: cand.function,
            jump_va: cand.jump_va,
            reason: cand.reason,
            pattern_rank,
            pattern_label: pattern.label(),
            role: pattern.role,
            guard: pattern.guard.clone(),
            boot_reachable: group.functions.contains_key(&cand.function),
            hit_targets: cand.hit_targets,
            distinct_sites: cand.distinct_sites,
            accepted_targets: cand.accepted_targets,
        });
    }

    hits.sort_by(|a, b| {
        b.accepted_targets
            .cmp(&a.accepted_targets)
            .then_with(|| b.hit_targets.len().cmp(&a.hit_targets.len()))
            .then_with(|| b.boot_reachable.cmp(&a.boot_reachable))
            .then_with(|| a.function.cmp(&b.function))
            .then_with(|| a.jump_va.cmp(&b.jump_va))
    });

    hits
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

    fn validate_rfc8259_json(input: &str) -> Result<(), String> {
        struct Parser<'a> {
            b: &'a [u8],
            i: usize,
        }
        impl<'a> Parser<'a> {
            fn ws(&mut self) {
                while matches!(self.b.get(self.i), Some(b' ' | b'\t' | b'\n' | b'\r')) {
                    self.i += 1;
                }
            }
            fn val(&mut self) -> Result<(), String> {
                self.ws();
                match self.b.get(self.i).copied() {
                    Some(b'{') => self.obj(),
                    Some(b'[') => self.arr(),
                    Some(b'"') => self.str_lit(),
                    Some(b't') => self.lit(b"true"),
                    Some(b'f') => self.lit(b"false"),
                    Some(b'n') => self.lit(b"null"),
                    Some(b'-' | b'0'..=b'9') => self.num(),
                    other => Err(format!("bad token {other:?} at {}", self.i)),
                }
            }
            fn lit(&mut self, s: &[u8]) -> Result<(), String> {
                if self.b[self.i..].starts_with(s) {
                    self.i += s.len();
                    Ok(())
                } else {
                    Err(format!("bad literal at {}", self.i))
                }
            }
            fn str_lit(&mut self) -> Result<(), String> {
                if self.b.get(self.i) != Some(&b'"') {
                    return Err(format!("expected quote at {}", self.i));
                }
                self.i += 1;
                while let Some(&c) = self.b.get(self.i) {
                    self.i += 1;
                    match c {
                        b'"' => return Ok(()),
                        b'\\' => {
                            let e = *self.b.get(self.i).ok_or("eof in escape")?;
                            self.i += 1;
                            match e {
                                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {}
                                b'u' => {
                                    for _ in 0..4 {
                                        let h = *self.b.get(self.i).ok_or("eof in \\u")?;
                                        if !h.is_ascii_hexdigit() {
                                            return Err(format!("bad hex at {}", self.i));
                                        }
                                        self.i += 1;
                                    }
                                }
                                _ => return Err(format!("bad escape at {}", self.i)),
                            }
                        }
                        0x00..=0x1F => return Err(format!("raw ctrl char at {}", self.i)),
                        _ => {}
                    }
                }
                Err("unterminated string".into())
            }
            fn num(&mut self) -> Result<(), String> {
                if self.b.get(self.i) == Some(&b'-') {
                    self.i += 1;
                }
                match self.b.get(self.i) {
                    Some(b'0') => self.i += 1,
                    Some(b'1'..=b'9') => {
                        while matches!(self.b.get(self.i), Some(b'0'..=b'9')) {
                            self.i += 1;
                        }
                    }
                    _ => return Err(format!("bad number at {}", self.i)),
                }
                Ok(())
            }
            fn arr(&mut self) -> Result<(), String> {
                self.i += 1;
                self.ws();
                if self.b.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(());
                }
                loop {
                    self.val()?;
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(());
                        }
                        _ => return Err(format!("bad array at {}", self.i)),
                    }
                }
            }
            fn obj(&mut self) -> Result<(), String> {
                self.i += 1;
                self.ws();
                if self.b.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(());
                }
                loop {
                    self.ws();
                    self.str_lit()?;
                    self.ws();
                    if self.b.get(self.i) != Some(&b':') {
                        return Err(format!("expected ':' at {}", self.i));
                    }
                    self.i += 1;
                    self.val()?;
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(());
                        }
                        _ => return Err(format!("bad object at {}", self.i)),
                    }
                }
            }
        }
        let mut p = Parser {
            b: input.as_bytes(),
            i: 0,
        };
        p.val()?;
        p.ws();
        if p.i != p.b.len() {
            return Err(format!("trailing bytes at {}", p.i));
        }
        Ok(())
    }

    #[test]
    fn diagnostics_aggregates_multi_target_hits_and_ranks_order_independently() {
        let row_a1 = TargetRow {
            target: 0x1050,
            sites: vec![0x9001, 0x9002],
            verdict: Verdict::Accepted,
            container: Some(0x1000),
            unresolved_jumps: vec![(0x1010, UnresolvedJump::TableBaseUnknown)],
        };
        let row_a2 = TargetRow {
            target: 0x1060,
            sites: vec![0x9002, 0x9003],
            verdict: Verdict::Accepted,
            container: Some(0x1000),
            unresolved_jumps: vec![(0x1010, UnresolvedJump::TableBaseUnknown)],
        };
        let row_b = TargetRow {
            target: 0x2030,
            sites: vec![0x9010],
            verdict: Verdict::Accepted,
            container: Some(0x2000),
            unresolved_jumps: vec![(0x2010, UnresolvedJump::TableUnbounded)],
        };
        let row_new = TargetRow {
            target: 0x3000,
            sites: vec![0x9020],
            verdict: Verdict::Accepted,
            container: None,
            unresolved_jumps: vec![],
        };
        let row_rej1 = TargetRow {
            target: 0x5000,
            sites: vec![0x9030],
            verdict: Verdict::NotExecutable,
            container: None,
            unresolved_jumps: vec![],
        };
        let row_rej2 = TargetRow {
            target: 0x4000,
            sites: vec![0x9031],
            verdict: Verdict::Undecodable,
            container: None,
            unresolved_jumps: vec![],
        };

        let o1 = SeedOutcome {
            pairs_in_range: 7,
            rows: vec![
                row_a1.clone(),
                row_a2.clone(),
                row_b.clone(),
                row_new.clone(),
                row_rej1.clone(),
                row_rej2.clone(),
            ],
        };
        let o2 = SeedOutcome {
            pairs_in_range: 7,
            rows: vec![row_rej1, row_b, row_a2, row_new, row_rej2, row_a1],
        };

        let d1 = o1.diagnostics();
        let d2 = o2.diagnostics();
        assert_eq!(d1, d2);
        assert_eq!(d1.pairs_in_range, 7);
        assert_eq!(d1.distinct_targets, 6);
        assert_eq!(d1.accepted_in_discovered_code, 3);
        assert_eq!(d1.accepted_new_entries, 1);
        assert_eq!(d1.verdict_counts.get("accepted"), Some(&4));
        assert_eq!(d1.verdict_counts.get("not_executable"), Some(&1));
        assert_eq!(d1.verdict_counts.get("undecodable"), Some(&1));

        // Rejected targets are sorted ascending by target VA.
        assert_eq!(
            d1.rejected_targets,
            vec![
                (0x4000, Verdict::Undecodable),
                (0x5000, Verdict::NotExecutable),
            ]
        );

        // Jump candidates aggregate multiple targets onto 0x1010 and rank it first.
        assert_eq!(d1.jump_candidates.len(), 2);
        let top = &d1.jump_candidates[0];
        assert_eq!(top.function, 0x1000);
        assert_eq!(top.jump_va, 0x1010);
        assert_eq!(top.reason, UnresolvedJump::TableBaseUnknown);
        assert_eq!(top.hit_targets, vec![0x1050, 0x1060]);
        assert_eq!(top.distinct_sites, BTreeSet::from([0x9001, 0x9002, 0x9003]));
        assert_eq!(top.accepted_targets, 2);
        assert_eq!(top.known_targets, 0);
        assert_eq!(d1.jump_candidates[1].jump_va, 0x2010);

        // Deterministic & host-path-sanitized text and JSON output.
        for adversarial in [
            r"C:\Users\dev\builds\game.exe",
            r"\\server\share\game.exe",
            "/home/dev/builds/game.exe",
            "file:///tmp/game.exe",
        ] {
            let txt1 = d1.render_text(adversarial, 10);
            let txt2 = d2.render_text(adversarial, 10);
            assert_eq!(txt1, txt2);
            assert!(
                !crate::summary::contains_host_path(&txt1),
                "host path leaked in text: {txt1}"
            );
            assert!(txt1.contains("game.exe"));

            let json1 = d1.render_json(adversarial, 10);
            let json2 = d2.render_json(adversarial, 10);
            assert_eq!(json1, json2);
            validate_rfc8259_json(&json1).expect("SeedDiagnostics JSON must be valid RFC 8259");
            assert!(
                !crate::summary::contains_host_path(&json1),
                "host path leaked in json: {json1}"
            );
            assert!(json1.contains("\"module\": \"game.exe\""));
        }
    }

    #[test]
    fn diagnostics_handles_empty_and_all_known_entries() {
        let empty = SeedOutcome::default().diagnostics();
        assert_eq!(empty.pairs_in_range, 0);
        assert_eq!(empty.distinct_targets, 0);
        assert_eq!(empty.accepted_in_discovered_code, 0);
        assert_eq!(empty.accepted_new_entries, 0);
        assert!(empty.jump_candidates.is_empty());
        assert!(empty.rejected_targets.is_empty());
        let empty_json = empty.render_json("mod.exe", 5);
        validate_rfc8259_json(&empty_json).expect("empty diagnostics JSON must be valid");

        let known = SeedOutcome {
            pairs_in_range: 2,
            rows: vec![
                TargetRow {
                    target: TEXT,
                    sites: vec![0x1],
                    verdict: Verdict::KnownEntry,
                    container: None,
                    unresolved_jumps: vec![],
                },
                TargetRow {
                    target: TEXT + 0x20,
                    sites: vec![0x2],
                    verdict: Verdict::KnownEntry,
                    container: None,
                    unresolved_jumps: vec![],
                },
            ],
        };
        let kd = known.diagnostics();
        assert_eq!(kd.accepted_in_discovered_code, 0);
        assert_eq!(kd.accepted_new_entries, 0);
        assert!(kd.rejected_targets.is_empty());
        assert!(kd.jump_candidates.is_empty());
        assert_eq!(kd.verdict_counts.get("known_entry"), Some(&2));
        let known_json = kd.render_json("mod.exe", 5);
        validate_rfc8259_json(&known_json).expect("all-known diagnostics JSON must be valid");
    }

    #[test]
    fn correlate_with_survey_matches_live_hits_to_classified_patterns() {
        use crate::jump_survey::{Layout, survey_jumps};
        use iced_x86::code_asm::*;

        // Function 1 (boot root at TEXT): shift-guarded switch with a reachable inner block
        // Function 2 (unreachable at TEXT+0x60): vtable tail call with a reachable inner block
        let fn1 = TEXT;
        let fn2 = TEXT + 0x60;

        let mut a1 = CodeAssembler::new(64).unwrap();
        let mut lbl_case = a1.create_label();
        a1.cmp(eax, 8).unwrap();
        a1.ja(lbl_case).unwrap();
        a1.mov(rdx, qword_ptr(rsp + 0x8)).unwrap();
        a1.jmp(qword_ptr(rdx + rax * 8)).unwrap();
        a1.set_label(&mut lbl_case).unwrap();
        a1.nop().unwrap();
        a1.ret().unwrap();
        let b1 = a1.assemble(fn1).unwrap();

        let mut a2 = CodeAssembler::new(64).unwrap();
        let mut lbl_alt = a2.create_label();
        a2.test(edx, edx).unwrap();
        a2.jz(lbl_alt).unwrap();
        a2.jmp(qword_ptr(rax + 0x10)).unwrap();
        a2.set_label(&mut lbl_alt).unwrap();
        a2.nop().unwrap();
        a2.ret().unwrap();
        let b2 = a2.assemble(fn2).unwrap();

        let mut code = b1.clone();
        code.resize((fn2 - TEXT) as usize, 0xCC);
        code.extend_from_slice(&b2);
        code.extend(std::iter::repeat_n(0xCC, 16));

        let img = CodeImage::new(
            BASE,
            vec![
                CodeSection::new(TEXT, &code, true),
                CodeSection::new(BASE + 0x2000, &[0u8; 0x40], false),
            ],
            vec![fn1, fn2],
        );
        let analysis = discover(&img);

        // Locate the two `ret` targets inside fn1 and fn2 (instruction boundaries inside discovered code).
        let fn1_ret = analysis.functions[&fn1]
            .blocks
            .values()
            .flat_map(|b| b.instructions.iter())
            .find(|i| i.mnemonic() == iced_x86::Mnemonic::Ret)
            .unwrap()
            .ip();
        let fn1_nop = analysis.functions[&fn1]
            .blocks
            .values()
            .flat_map(|b| b.instructions.iter())
            .find(|i| i.mnemonic() == iced_x86::Mnemonic::Nop)
            .unwrap()
            .ip();
        let fn2_ret = analysis.functions[&fn2]
            .blocks
            .values()
            .flat_map(|b| b.instructions.iter())
            .find(|i| i.mnemonic() == iced_x86::Mnemonic::Ret)
            .unwrap()
            .ip();

        let seeds = TraceSeeds {
            pairs: vec![
                TracePair {
                    site: 0x100,
                    target: fn1_nop,
                },
                TracePair {
                    site: 0x104,
                    target: fn1_ret,
                },
                TracePair {
                    site: 0x200,
                    target: fn2_ret,
                },
            ],
        };
        let outcome = evaluate(&img, &analysis, &seeds);
        let layout = Layout {
            image_base: BASE,
            sections: vec![
                (".text".into(), 0x1000, 0x1000, 0x6000_0020),
                (".rdata".into(), 0x2000, 0x40, 0x4000_0040),
            ],
        };
        let survey = survey_jumps(&layout, &analysis, &[fn1], 10);

        let hits = correlate_with_survey(&outcome, &survey);
        assert_eq!(hits.len(), 2);

        // First hit: fn1 (2 accepted targets, boot-reachable switch).
        assert_eq!(hits[0].function, fn1);
        assert_eq!(hits[0].role, "switch");
        assert_eq!(hits[0].guard, "cmp_imm_unsigned");
        assert!(hits[0].boot_reachable);
        assert_eq!(hits[0].accepted_targets, 2);
        assert_eq!(hits[0].hit_targets, vec![fn1_nop, fn1_ret]);
        assert_eq!(hits[0].distinct_sites, BTreeSet::from([0x100, 0x104]));
        assert_eq!(hits[0].pattern_rank, 1);
        assert!(hits[0].pattern_label.contains("switch"));

        // Second hit: fn2 (1 accepted target, unreachable tail_call).
        assert_eq!(hits[1].function, fn2);
        assert_eq!(hits[1].role, "tail_call");
        assert!(!hits[1].boot_reachable);
        assert_eq!(hits[1].accepted_targets, 1);
        assert_eq!(hits[1].hit_targets, vec![fn2_ret]);

        // Correlating against switches_only() keeps only the switch hit.
        let sw_hits = correlate_with_survey(&outcome, &survey.switches_only());
        assert_eq!(sw_hits.len(), 1);
        assert_eq!(sw_hits[0].function, fn1);
        assert_eq!(sw_hits[0].role, "switch");
    }
}
