//! R40 guard for the R19 hot paths: an instruction-count budget per benchmark function and memory
//! model, taken from the lifted (unoptimised) IR, so it is deterministic and needs no LLVM tools.
//! Timing is too noisy for CI; a growth of the lifted IR is what a lifter regression looks like.
//!
//! The structural properties behind the R19 speedups are asserted next to the code that has them:
//! `truncating_imul64_result_is_a_plain_mul` (ir.rs), the contiguous-runs single dirty mark
//! (memory_models.rs) and `calls_store_only_the_flags_the_callee_may_observe`
//! (flags_across_calls.rs).
//!
//! Fails if a function's instruction count exceeds its snapshot by more than 10%. After an
//! intended change, regenerate `ir_budget.snap` with
//! `PLAYGTA6_UPDATE_IR_BUDGET=1 cargo test -p playgta6-recomp ir_budget` (run in `recomp/`) and
//! commit it.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::{env, fs};

use super::memory_bench::{CHASE, COPY, LEAF, REP_COPY, STRUCTS, SUM, program};
use super::memory_models::models;
use crate::lift::lift_module_with;

const SNAPSHOT: &str = include_str!("ir_budget.snap");
/// Allowed growth, in percent.
const TOLERANCE_PERCENT: usize = 10;

/// Instructions of one `define` text: indented lines that are neither labels nor comments.
fn instruction_count(ir: &str) -> usize {
    ir.lines()
        .filter(|l| l.starts_with(' ') || l.starts_with('\t'))
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with(';') && !l.ends_with(':'))
        .count()
}

fn measure() -> BTreeMap<String, usize> {
    let names = [
        (SUM, "array_sum"),
        (CHASE, "pointer_chase"),
        (COPY, "copy_loop"),
        (STRUCTS, "typical_structs"),
        (LEAF, "leaf"),
        (REP_COPY, "rep_movsq"),
    ];
    let functions = program().discover();
    let mut out = BTreeMap::new();
    for (model_name, model) in models() {
        let module = lift_module_with(functions.values(), model).expect("lifts");
        for (entry, name) in names {
            let f = module
                .functions
                .iter()
                .find(|f| f.entry == entry)
                .expect("function lifted");
            out.insert(format!("{model_name}/{name}"), instruction_count(&f.ir));
        }
    }
    out
}

#[test]
fn r19_hot_paths_stay_within_their_ir_instruction_budget() {
    let now = measure();
    if env::var_os("PLAYGTA6_UPDATE_IR_BUDGET").is_some() {
        let mut text = String::from(
            "# model/function instruction-count; regenerate with PLAYGTA6_UPDATE_IR_BUDGET=1\n",
        );
        for (k, v) in &now {
            let _ = writeln!(text, "{k} {v}");
        }
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/lift/tests/ir_budget.snap");
        fs::write(path, text).expect("write snapshot");
        return;
    }
    let snap: BTreeMap<&str, usize> = SNAPSHOT
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let (k, v) = l.rsplit_once(' ').expect("`key count` line");
            (k, v.parse().expect("count"))
        })
        .collect();
    let mut failures = Vec::new();
    for (key, &count) in &now {
        match snap.get(key.as_str()) {
            None => failures.push(format!("{key}: missing from snapshot (now {count})")),
            Some(&budget) if count * 100 > budget * (100 + TOLERANCE_PERCENT) => failures.push(
                format!("{key}: {count} instructions, budget {budget} (+10%)"),
            ),
            _ => {}
        }
    }
    assert!(
        failures.is_empty(),
        "lifted IR grew past the R19 budget (PLAYGTA6_UPDATE_IR_BUDGET=1 to accept):\n{}",
        failures.join("\n")
    );
}
