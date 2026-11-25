//! R19 measurements on real lifted code (tool-gated, local only): lifts a sample of the functions
//! of an executable under every guest memory model, writes one module per shard and model,
//! compiles each to a wasm64 object and reports IR and object sizes.
//!
//! Enabled only when `PLAYGTA6_R19_EXE` (the PE file, a read-only input) and `PLAYGTA6_R19_OUT`
//! (an output directory, by convention under the gitignored `work/`) are set. Optional:
//! `PLAYGTA6_R19_STRIDE` (lift every Nth discovered function, default 100),
//! `PLAYGTA6_R19_SHARD` (functions per module, default 250), `PLAYGTA6_R19_MODELS`
//! (comma-separated model names, default all), `PLAYGTA6_R19_NO_OBJ` (IR only).
//! Nothing derived from the input goes anywhere but the output directory; the printed summary is
//! counts and sizes only.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::{env, fs};

use super::ir::random_program;
use crate::cfg;
use crate::discover::discover;
use crate::lift::compile::{compile_to_wasm64_object, find_llvm_tool};
use crate::lift::memory::{Window, WindowCheck};
use crate::lift::{
    GuestBase, MemoryModel, emit_module, lift_function_with, lift_module, lift_module_with,
};
use crate::pe::Pe64;
use crate::pipeline::{Seeded, seed};

/// The CPU window as the browser would place it: above the 16 GiB wasm64 memory, sized for the
/// larger console profile (24 GiB; Series X uses 16 GiB of it).
pub(super) const REAL_WINDOW_BASE: u64 = 0x4_0000_0000;
pub(super) const REAL_WINDOW_SIZE: u64 = 0x6_0000_0000;
/// Page-table slots covering guest VAs [0, 64 GiB).
pub(super) const REAL_PAGES: u64 = 1 << 20;

fn window(check: WindowCheck, stack_flat: bool, contiguous_runs: bool) -> MemoryModel {
    MemoryModel::Windowed(Window {
        flat: GuestBase::Constant(0),
        base: REAL_WINDOW_BASE,
        size: REAL_WINDOW_SIZE,
        check,
        stack_flat,
        contiguous_runs,
    })
}

/// The models measured, by name.
pub(super) fn models() -> Vec<(&'static str, MemoryModel)> {
    vec![
        ("identity", GuestBase::Constant(0).into()),
        ("symbol", GuestBase::Symbol.into()),
        ("window", window(WindowCheck::Branch, false, false)),
        ("window-stack", window(WindowCheck::Branch, true, false)),
        ("window-runs", window(WindowCheck::Branch, true, true)),
        (
            "window-outline-stack",
            window(WindowCheck::Outline, true, false),
        ),
        ("window-select", window(WindowCheck::Select, false, false)),
        (
            "page",
            MemoryModel::PageTable {
                pages: REAL_PAGES,
                stack: None,
                contiguous_runs: false,
            },
        ),
        (
            "page-runs",
            MemoryModel::PageTable {
                pages: REAL_PAGES,
                stack: Some(GuestBase::Constant(0)),
                contiguous_runs: true,
            },
        ),
    ]
}

/// Symbols only the table models emit.
const TABLE_MODEL_SYMBOLS: [&str; 7] = [
    "__playgta6_window_table",
    "__playgta6_page_table",
    "__playgta6_guest_read",
    "__playgta6_guest_scratch",
    "__playgta6_guest_write",
    "__playgta6_guest_memmove",
    "__playgta6_guest_memset",
];

#[test]
fn default_model_output_is_the_base_translation() {
    // The default is MemoryModel::Base(GuestBase::Symbol), emitted exactly as before R19 (the
    // byte-for-byte comparison against the pre-R19 lifter on 3,562 real functions is recorded in
    // the R19 report); nothing of the table models leaks into it.
    let functions = random_program(150);
    let default = lift_module(functions.values()).expect("lifts").ir;
    let explicit = lift_module_with(functions.values(), GuestBase::Symbol)
        .expect("lifts")
        .ir;
    assert_eq!(default, explicit);
    assert_eq!(MemoryModel::default(), MemoryModel::Base(GuestBase::Symbol));
    for symbol in TABLE_MODEL_SYMBOLS {
        assert!(!default.contains(symbol), "{symbol} in default output");
    }
}

#[test]
fn table_entries_are_ordinary_loads_and_the_base_stays_invariant() {
    let functions = random_program(60);
    for (name, model) in models().into_iter().skip(2) {
        let ir = lift_module_with(functions.values(), model)
            .expect("lifts")
            .ir;
        let mut entry_loads = 0;
        for line in ir.lines() {
            // An entry load reads a pointer computed from the table (`getelementptr ... @table`).
            if line.contains("load i64, ptr %") && line.contains("align 8") {
                assert!(
                    !line.contains("invariant"),
                    "{name}: table entry load {line}"
                );
                entry_loads += 1;
            }
            if line.contains("@__playgta6_guest_base,") {
                assert!(line.contains("!invariant.load"), "{name}: {line}");
            }
        }
        assert!(entry_loads > 0, "{name}: no table loads");
        assert!(
            ir.contains("@__playgta6_window_table") || ir.contains("@__playgta6_page_table"),
            "{name}"
        );
    }
}

#[test]
fn contiguous_runs_store_marks_its_last_slot_only_when_it_crosses() {
    // `mov [rcx], rax` under window-runs: one unconditional mark of the store's slot; the mark of
    // the last byte's slot sits in its own block behind `icmp ne` (taken only by a store that
    // leaves its slot), not on the hot path.
    use iced_x86::code_asm::*;
    let f = super::asm_function(0x1000, |a| {
        a.mov(qword_ptr(rcx), rax)?;
        a.ret()
    });
    let ir = lift_module_with([&f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    // Labels of the blocks holding a mark.
    let mut label = "";
    let mut marked = Vec::new();
    for line in ir.lines() {
        if let Some(l) = line.strip_suffix(':') {
            label = l;
        } else if line.contains("store atomic i8 1, ptr ") {
            marked.push(label);
        }
    }
    assert_eq!(marked.len(), 2, "{ir}");
    assert_ne!(marked[0], marked[1], "both marks on the hot path:\n{ir}");
    let cold = marked[1];
    assert!(cold.starts_with("gm"), "{ir}");
    // The cold mark block is entered only by an `icmp ne` of the two slot indices.
    let branch = ir
        .lines()
        .find(|l| l.contains(&format!("label %{cold},")))
        .expect("branch to the cold mark");
    let cond = branch
        .split_whitespace()
        .nth(2)
        .expect("condition")
        .trim_end_matches(',');
    assert!(
        ir.lines()
            .any(|l| l.trim_start().starts_with(&format!("{cond} = icmp ne i64"))),
        "{ir}"
    );
}

/// FNV-1a 64 of a byte string (stable fingerprint of emitted IR).
fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

#[test]
fn r19_measure_real_code() {
    let (Some(exe), Some(out)) = (
        env::var_os("PLAYGTA6_R19_EXE"),
        env::var_os("PLAYGTA6_R19_OUT"),
    ) else {
        println!("SKIP r19_measure_real_code: PLAYGTA6_R19_EXE / PLAYGTA6_R19_OUT not set");
        return;
    };
    let out = PathBuf::from(out);
    let stride: usize = env::var("PLAYGTA6_R19_STRIDE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);
    let shard: usize = env::var("PLAYGTA6_R19_SHARD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(250);
    let wanted: Option<Vec<String>> = env::var("PLAYGTA6_R19_MODELS")
        .ok()
        .map(|v| v.split(',').map(str::to_string).collect());
    let emit_obj = env::var_os("PLAYGTA6_R19_NO_OBJ").is_none();
    let tool = if emit_obj { find_llvm_tool() } else { None };

    let bytes = fs::read(&exe).expect("read executable");
    let pe = Pe64::parse(&bytes).expect("parse PE");
    let Seeded { image, .. } = seed(&pe, &bytes);
    let analysis = discover(&image);
    let sample: Vec<&cfg::Function> = analysis.functions.values().step_by(stride).collect();
    let instructions: usize = sample
        .iter()
        .flat_map(|f| f.blocks.values())
        .map(|b| b.instructions.len())
        .sum();
    println!(
        "r19: {} functions discovered, {} sampled (every {stride}th), {instructions} instructions",
        analysis.functions.len(),
        sample.len()
    );

    let mut report = String::new();
    let mut totals: BTreeMap<&str, (usize, u64, u64, u64)> = BTreeMap::new();
    for (name, model) in models() {
        if wanted
            .as_ref()
            .is_some_and(|w| !w.iter().any(|m| m == name))
        {
            continue;
        }
        let dir = out.join(name);
        fs::create_dir_all(&dir).expect("create output dir");
        let (mut ir_bytes, mut obj_bytes, mut hash, mut lifted) = (0u64, 0u64, 0u64, 0usize);
        for (i, chunk) in sample.chunks(shard).enumerate() {
            let mut functions = Vec::new();
            for f in chunk {
                if let Ok(l) = lift_function_with(f, model) {
                    functions.push(l);
                }
            }
            lifted += functions.len();
            let ir = emit_module(&functions).expect("emit module");
            ir_bytes += ir.len() as u64;
            hash = hash.rotate_left(5) ^ fnv(ir.as_bytes());
            let ll = dir.join(format!("shard-{i:04}.ll"));
            fs::write(&ll, &ir).expect("write .ll");
            if let Some(tool) = &tool {
                let obj = ll.with_extension("o");
                compile_to_wasm64_object(tool, &ll, &obj).expect("compile shard");
                obj_bytes += fs::metadata(&obj).map(|m| m.len()).unwrap_or(0);
            }
        }
        let _ = writeln!(
            report,
            "{name}\tfunctions={lifted}\tir_bytes={ir_bytes}\tobj_bytes={obj_bytes}\tir_hash={hash:016x}"
        );
        totals.insert(name, (lifted, ir_bytes, obj_bytes, hash));
    }
    fs::write(out.join("sizes.tsv"), &report).expect("write report");
    print!("{report}");
}
