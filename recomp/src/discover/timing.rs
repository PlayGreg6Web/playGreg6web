//! Wall time per discovery phase ([`super::discover_timed`],
//! [`crate::pipeline::refine::discover_refined_timed`]). Diagnostic only: the timings never
//! influence the analysis.

use std::fmt::Write as _;
use std::time::{Duration, Instant};

/// A timed part of discovery. The order is the report order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Phase {
    /// Seeding entries and EH resume targets.
    Seed,
    /// The block decoding worklist (`run`).
    Decode,
    ResolveJumpTables,
    ReviseJumpTables,
    RequeueChangedJumps,
    LateResumeTargets,
    PointerSlots,
    VtableCallbacks,
    VerifyJumpTables,
    ClassifyUnresolved,
    Relocations,
    /// Function membership walks (`into_functions`).
    Functions,
    /// Refinement: base-relocation pointer scan (once).
    RefineBaseRelocs,
    /// Refinement: code-pointer classification and no-return thunks (per round).
    RefineClassify,
}

impl Phase {
    pub const ALL: [Phase; 14] = [
        Phase::Seed,
        Phase::Decode,
        Phase::ResolveJumpTables,
        Phase::ReviseJumpTables,
        Phase::RequeueChangedJumps,
        Phase::LateResumeTargets,
        Phase::PointerSlots,
        Phase::VtableCallbacks,
        Phase::VerifyJumpTables,
        Phase::ClassifyUnresolved,
        Phase::Relocations,
        Phase::Functions,
        Phase::RefineBaseRelocs,
        Phase::RefineClassify,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Phase::Seed => "seed",
            Phase::Decode => "decode",
            Phase::ResolveJumpTables => "resolve_jump_tables",
            Phase::ReviseJumpTables => "revise_jump_tables",
            Phase::RequeueChangedJumps => "requeue_changed_jumps",
            Phase::LateResumeTargets => "late_resume_targets",
            Phase::PointerSlots => "pointer_slots",
            Phase::VtableCallbacks => "vtable_callbacks",
            Phase::VerifyJumpTables => "verify_jump_tables",
            Phase::ClassifyUnresolved => "classify_unresolved",
            Phase::Relocations => "relocations",
            Phase::Functions => "functions",
            Phase::RefineBaseRelocs => "refine_base_relocs",
            Phase::RefineClassify => "refine_classify",
        }
    }

    fn index(self) -> usize {
        Phase::ALL.iter().position(|&p| p == self).unwrap_or(0)
    }
}

/// Calls and accumulated wall time of one phase.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PhaseTime {
    pub calls: u64,
    pub time: Duration,
}

/// Accumulated wall time per [`Phase`], plus the wall time of each `discover()` round.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DiscoveryTimings {
    phases: [PhaseTime; Phase::ALL.len()],
    /// Wall time of each `discover()` call (one per refinement round).
    pub rounds: Vec<Duration>,
    /// Wall time of the whole stage (all rounds and the refinement).
    pub total: Duration,
    /// Rounds that continued the previous round's state instead of starting over.
    pub continued_rounds: usize,
    /// Continued rounds after which the state had to be dropped (see
    /// [`super::Session::diverged`]).
    pub diverged_rounds: usize,
}

impl DiscoveryTimings {
    /// Runs `f`, adding its wall time to `phase`.
    pub fn time<R>(&mut self, phase: Phase, f: impl FnOnce() -> R) -> R {
        let start = Instant::now();
        let out = f();
        self.add(phase, start.elapsed());
        out
    }

    pub fn add(&mut self, phase: Phase, time: Duration) {
        let entry = &mut self.phases[phase.index()];
        entry.calls += 1;
        entry.time += time;
    }

    pub fn get(&self, phase: Phase) -> PhaseTime {
        self.phases[phase.index()]
    }

    /// Phases that ran, in report order.
    pub fn phases(&self) -> impl Iterator<Item = (Phase, PhaseTime)> + '_ {
        Phase::ALL
            .into_iter()
            .map(|p| (p, self.get(p)))
            .filter(|(_, t)| t.calls > 0)
    }

    /// Adds every phase and round of `other`.
    pub fn merge(&mut self, other: &DiscoveryTimings) {
        for (phase, t) in other.phases() {
            let entry = &mut self.phases[phase.index()];
            entry.calls += t.calls;
            entry.time += t.time;
        }
        self.rounds.extend_from_slice(&other.rounds);
        self.total += other.total;
        self.continued_rounds += other.continued_rounds;
        self.diverged_rounds += other.diverged_rounds;
    }

    /// Human-readable table (one line per phase, then the rounds).
    pub fn render(&self) -> String {
        let mut out = String::new();
        let total = self.total.as_secs_f64();
        let _ = writeln!(out, "{total:.2} s total");
        for (phase, t) in self.phases() {
            let secs = t.time.as_secs_f64();
            let share = if total > 0.0 {
                100.0 * secs / total
            } else {
                0.0
            };
            let _ = writeln!(
                out,
                "  {:<24} {:>6} calls {:>9.2} s {:>5.1} %",
                phase.as_str(),
                t.calls,
                secs,
                share
            );
        }
        let rounds: Vec<String> = self
            .rounds
            .iter()
            .map(|d| format!("{:.2}", d.as_secs_f64()))
            .collect();
        let _ = writeln!(
            out,
            "  rounds (s): [{}], {} continued, {} diverged",
            rounds.join(", "),
            self.continued_rounds,
            self.diverged_rounds
        );
        out
    }

    /// One JSON object: `{"total_s":..,"rounds_s":[..],"continued_rounds":..,"diverged_rounds":..,"phases":{"<name>":{"calls":..,"s":..}}}`.
    pub fn to_json(&self) -> String {
        let phases: Vec<String> = self
            .phases()
            .map(|(p, t)| {
                format!(
                    "\"{}\":{{\"calls\":{},\"s\":{:.3}}}",
                    p.as_str(),
                    t.calls,
                    t.time.as_secs_f64()
                )
            })
            .collect();
        let rounds: Vec<String> = self
            .rounds
            .iter()
            .map(|d| format!("{:.3}", d.as_secs_f64()))
            .collect();
        format!(
            "{{\"total_s\":{:.3},\"rounds_s\":[{}],\"continued_rounds\":{},\"diverged_rounds\":{},\"phases\":{{{}}}}}",
            self.total.as_secs_f64(),
            rounds.join(","),
            self.continued_rounds,
            self.diverged_rounds,
            phases.join(",")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulates_merges_and_renders_in_phase_order() {
        let mut a = DiscoveryTimings::default();
        a.add(Phase::ReviseJumpTables, Duration::from_millis(30));
        a.add(Phase::Decode, Duration::from_millis(10));
        a.add(Phase::Decode, Duration::from_millis(5));
        a.rounds.push(Duration::from_millis(45));
        a.total = Duration::from_millis(50);
        let mut b = DiscoveryTimings::default();
        b.merge(&a);
        b.merge(&a);
        assert_eq!(b.get(Phase::Decode).calls, 4);
        assert_eq!(b.get(Phase::Decode).time, Duration::from_millis(30));
        assert_eq!(b.rounds.len(), 2);
        let names: Vec<_> = b.phases().map(|(p, _)| p).collect();
        assert_eq!(names, [Phase::Decode, Phase::ReviseJumpTables]);
        let json = b.to_json();
        assert!(json.starts_with("{\"total_s\":0.100,\"rounds_s\":[0.045,0.045]"));
        assert!(json.contains("\"decode\":{\"calls\":4,\"s\":0.030}"));
        let text = b.render();
        assert!(text.find("decode").unwrap() < text.find("revise_jump_tables").unwrap());
    }
}
