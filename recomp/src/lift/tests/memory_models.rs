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
use crate::lift::memory::{SLOT_SHIFT, Window, WindowCheck};
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
    // `mov [rcx], rax` under window-runs (R57 + R63c):
    // - Primary slot: one hot dirty check (`load atomic i8 ... monotonic`) with its atomic release
    //   store behind `icmp eq i8 ..., 0` in `gms*`.
    // - Last byte's slot (R63c): sits in its own cold `gm*` block behind `icmp ne` (taken only by
    //   a store that leaves its slot) and emits a compact direct `store atomic i8 1 ... release`
    //   with no extra load/branch diamond.
    use iced_x86::code_asm::*;
    let f = super::asm_function(0x1000, |a| {
        a.mov(qword_ptr(rcx), rax)?;
        a.ret()
    });
    let ir = lift_module_with([&f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    // Labels of the blocks holding a dirty check and a release store.
    let mut label = "";
    let mut checked = Vec::new();
    let mut stored = Vec::new();
    for line in ir.lines() {
        if let Some(l) = line.strip_suffix(':') {
            label = l;
        } else if line.contains("load atomic i8, ptr ") && line.contains(" monotonic, align 1") {
            checked.push(label);
        } else if line.contains("store atomic i8 1, ptr ") && line.contains(" release, align 1") {
            stored.push(label);
        }
    }
    assert_eq!(checked.len(), 1, "{ir}");
    assert_eq!(stored.len(), 2, "{ir}");
    let primary_set = stored[0];
    assert!(primary_set.starts_with("gms"), "{ir}");
    let set_br = ir
        .lines()
        .find(|l| l.contains(&format!("label %{primary_set},")))
        .expect("branch to mark-set block");
    assert!(
        set_br.contains("!prof !{!\"branch_weights\", i32 1, i32 2000}"),
        "{ir}"
    );
    let set_cond = set_br
        .split_whitespace()
        .nth(2)
        .expect("condition")
        .trim_end_matches(',');
    assert!(
        ir.lines()
            .any(|l| l.trim_start().starts_with(&format!("{set_cond} = icmp eq i8"))
                && l.ends_with(", 0")),
        "{ir}"
    );
    // The cold cross-slot mark block is entered only by an `icmp ne` of the two slot indices and
    // stores directly without a second `gms*` diamond.
    let cold = stored[1];
    assert!(cold.starts_with("gm") && !cold.starts_with("gms"), "{ir}");
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

#[test]
fn window_table_offsets_use_a_short_mask() {
    // Every dynamic window table entry is addressed by its byte offset `(rel >>a 13) & -8`
    // (`getelementptr i8`), never by a scaled slot index (`getelementptr i64, ..., rel >> 16`):
    // wasm lowers the scaled index to `(rel >> 13) & 0x7fff_ffff_ffff_fff8`, a 9-byte
    // `i64.const` per window check. Same entry for every address; the slot index the dirty
    // marks use is unchanged.
    use iced_x86::code_asm::*;
    let f = super::asm_function(0x1000, |a| {
        a.mov(rax, qword_ptr(rcx))?;
        a.mov(qword_ptr(rdx), rax)?;
        a.lock().add(dword_ptr(rsi), eax)?;
        a.mov(byte_ptr(rdi), al)?;
        a.ret()
    });
    let table = "ptr @__playgta6_window_table, i64 ";
    let tool = find_llvm_tool();
    for check in [WindowCheck::Branch, WindowCheck::Select] {
        for runs in [false, true] {
            let tag = format!("{check:?}/{runs}");
            let ir = lift_module_with([&f], window(check, false, runs))
                .expect("lifts")
                .ir;
            let defs: BTreeMap<&str, &str> = ir
                .lines()
                .filter_map(|l| l.trim_start().split_once(" = "))
                .collect();
            let def = |v: &str| defs.get(v).copied().unwrap_or("");
            let mut entries = 0;
            for line in ir.lines().filter(|l| l.contains(table)) {
                let index = line.split(table).nth(1).expect("index").trim();
                if !index.starts_with('%') {
                    continue; // lift-time constant slot
                }
                assert!(line.contains("getelementptr i8, "), "{tag}: {line}
{ir}");
                // Select: `select i1 %inside, i64 %offset, i64 0`.
                let offset = def(index)
                    .strip_prefix("select i1 ")
                    .and_then(|r| r.split(", i64 ").nth(1))
                    .unwrap_or(index);
                let and = def(offset);
                let shifted = and
                    .strip_prefix("and i64 ")
                    .and_then(|r| r.strip_suffix(", -8"))
                    .unwrap_or_else(|| panic!("{tag}: offset {and}
{ir}"));
                assert!(
                    def(shifted).starts_with("ashr i64 ")
                        && def(shifted).ends_with(&format!(", {}", SLOT_SHIFT - 3)),
                    "{tag}: {shifted} = {}
{ir}",
                    def(shifted)
                );
                entries += 1;
            }
            assert!(entries >= 3, "{tag}: {entries} dynamic table entries
{ir}");
            if let Some(tool) = &tool {
                let code = compile(tool, &ir, &tag);
                let wide = sleb_i64_const((u64::MAX >> (SLOT_SHIFT - 3)) & !7);
                assert!(
                    !code.windows(wide.len()).any(|w| w == wide.as_slice()),
                    "{tag}: wasm object still holds the 64-bit slot mask"
                );
            }
        }
    }
}

/// `i64.const value` as wasm encodes it (opcode `0x42`, signed LEB128).
fn sleb_i64_const(value: u64) -> Vec<u8> {
    let mut v = value as i64;
    let mut out = vec![0x42];
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        let done = (v == 0 && byte & 0x40 == 0) || (v == -1 && byte & 0x40 != 0);
        out.push(if done { byte } else { byte | 0x80 });
        if done {
            return out;
        }
    }
}

/// The wasm64 object of `ir` at `-O2`.
fn compile(tool: &crate::lift::compile::LlvmTool, ir: &str, tag: &str) -> Vec<u8> {
    let dir = env::temp_dir().join(format!(
        "pg6_window_entry_{}_{}",
        std::process::id(),
        tag.replace('/', "-")
    ));
    fs::create_dir_all(&dir).expect("temp dir");
    let ll = dir.join("m.ll");
    let obj = dir.join("m.o");
    fs::write(&ll, ir).expect("write ir");
    compile_to_wasm64_object(tool, &ll, &obj).expect("compiles");
    let bytes = fs::read(&obj).expect("object");
    let _ = fs::remove_dir_all(&dir);
    bytes
}

#[test]
fn window_dirty_marks_guard_atomic_release_store_with_monotonic_check() {
    // R57 + R63c: every primary dirty release store is guarded by a `load atomic i8 ... monotonic`
    // + `icmp eq i8 ..., 0` cold branch so already-dirty slots avoid `LOCK XCHG`. Under
    // `window-runs`, the rare (~0.005%) cross-slot fallback for multi-byte stores uses a compact
    // direct `store atomic i8 1 ... release` inside `%gm`.
    use iced_x86::code_asm::*;
    let f = super::asm_function(0x1000, |a| {
        a.mov(qword_ptr(rcx), rax)?;
        a.lock().xadd(qword_ptr(rdx), rax)?;
        a.ret()
    });
    for name in ["window", "window-outline"] {
        let model = MemoryModel::from_name(name).expect("model");
        let ir = lift_module_with([&f], model).expect("lifts").ir;
        let loads = ir.matches("load atomic i8, ptr ").count();
        let stores = ir.matches("store atomic i8 1, ptr ").count();
        assert!(stores > 0, "{name}: expected dirty marks:\n{ir}");
        assert_eq!(
            loads, stores,
            "{name}: every dirty release store must have a matching monotonic load guard:\n{ir}"
        );
    }
    let runs_ir = lift_module_with([&f], MemoryModel::from_name("window-runs").expect("model"))
        .expect("lifts")
        .ir;
    assert_eq!(
        runs_ir.matches("load atomic i8, ptr ").count(),
        2,
        "window-runs: 2 guarded primary marks (store + lock xadd):\n{runs_ir}"
    );
    assert_eq!(
        runs_ir.matches("store atomic i8 1, ptr ").count(),
        3,
        "window-runs: 2 primary release stores + 1 compact cross-slot release store:\n{runs_ir}"
    );
    let select_ir = lift_module_with([&f], window(WindowCheck::Select, false, false))
        .expect("lifts")
        .ir;
    let loads = select_ir.matches("load atomic i8, ptr ").count();
    let stores = select_ir.matches("store atomic i8 1, ptr ").count();
    assert!(stores > 0, "window-select:\n{select_ir}");
    assert_eq!(loads, stores, "window-select:\n{select_ir}");
}

#[test]
fn constant_addresses_outside_window_elide_window_checks_and_atomics() {
    use iced_x86::{Code, Instruction, MemoryOperand, Register};
    // RIP-relative globals in the executable image (0x1_4AA3_6C18..0x1_4AA3_6C80), strictly below
    // GUEST_WINDOW_BASE (0x4_0000_0000), plus an address above GUEST_WINDOW_BASE + SIZE.
    let g_below = 0x1_4AA3_6C18u64;
    let g_below2 = 0x1_4AA3_6C20u64;
    let g_vec = 0x1_4AA3_6C40u64;
    let g_vec2 = 0x1_4AA3_6C60u64;
    let bytes = super::encode(
        0x1_4000_1000,
        &[
            Instruction::with2(
                Code::Mov_r64_rm64,
                Register::RAX,
                MemoryOperand::with_base_displ(Register::RIP, g_below as i64),
            )
            .expect("mov load"),
            Instruction::with2(
                Code::Mov_rm64_r64,
                MemoryOperand::with_base_displ(Register::RIP, g_below2 as i64),
                Register::RAX,
            )
            .expect("mov store"),
            Instruction::with2(
                Code::VEX_Vmovdqu_ymm_ymmm256,
                Register::YMM0,
                MemoryOperand::with_base_displ(Register::RIP, g_vec as i64),
            )
            .expect("vmovdqu load"),
            Instruction::with2(
                Code::VEX_Vmovdqu_ymmm256_ymm,
                MemoryOperand::with_base_displ(Register::RIP, g_vec2 as i64),
                Register::YMM0,
            )
            .expect("vmovdqu store"),
            Instruction::with(Code::Retnq),
        ],
    );
    let f = super::Program::new()
        .bytes(0x1_4000_1000, &bytes)
        .function(0x1_4000_1000);

    for (name, model) in [
        ("window", MemoryModel::from_name("window").expect("m")),
        (
            "window-runs",
            MemoryModel::from_name("window-runs").expect("m"),
        ),
        (
            "window-outline",
            MemoryModel::from_name("window-outline").expect("m"),
        ),
        ("window-select", window(WindowCheck::Select, true, false)),
    ] {
        let module = lift_module_with([&f], model).expect("lifts");
        let ir = &module.ir;
        assert!(
            !ir.contains("__playgta6_window_table"),
            "{name}: constant out-of-window addresses must not reference window_table:\n{ir}"
        );
        assert!(
            !ir.contains("__playgta6_window_dirty"),
            "{name}: constant out-of-window stores must not reference window_dirty:\n{ir}"
        );
        assert!(
            !ir.contains("icmp ult"),
            "{name}: constant out-of-window addresses must not emit range checks:\n{ir}"
        );
        assert!(
            !module.requires_shared_memory(),
            "{name}: non-atomic out-of-window stores must not pull in +atomics:\n{ir}"
        );
    }

    // Locked atomic to a constant out-of-window global also skips window_table and window_dirty.
    let mut lock_xadd = Instruction::with2(
        Code::Xadd_rm64_r64,
        MemoryOperand::with_base_displ(Register::RIP, g_below as i64),
        Register::RAX,
    )
    .expect("xadd");
    lock_xadd.set_has_lock_prefix(true);
    let atomic_bytes = super::encode(
        0x1_4000_2000,
        &[lock_xadd, Instruction::with(Code::Retnq)],
    );
    let atomic_f = super::Program::new()
        .bytes(0x1_4000_2000, &atomic_bytes)
        .function(0x1_4000_2000);
    let atomic_ir = lift_module_with(
        [&atomic_f],
        MemoryModel::from_name("window-runs").expect("m"),
    )
    .expect("lifts")
    .ir;
    assert!(
        !atomic_ir.contains("__playgta6_window_table")
            && !atomic_ir.contains("__playgta6_window_dirty"),
        "atomic on out-of-window global must not touch window tables:\n{atomic_ir}"
    );
    assert!(atomic_ir.contains("atomicrmw add ptr"), "{atomic_ir}");
}

#[test]
fn constant_addresses_inside_window_fold_slot_and_straddle_checks() {
    use iced_x86::{Code, Instruction, MemoryOperand, Register};
    // Code near the window so RIP-relative addresses land inside [0x4_0000_0000, 0xA_0000_0000):
    // aligned slot 1 (+0x10008), and straddling slot 1|2 (+0x1FFFC, 8 bytes).
    let base = REAL_WINDOW_BASE;
    let aligned = base + 0x1_0008;
    let straddle = base + 0x1_FFFC;
    let code_va = base - 0x1000;
    let bytes = super::encode(
        code_va,
        &[
            Instruction::with2(
                Code::Mov_r64_rm64,
                Register::RAX,
                MemoryOperand::with_base_displ(Register::RIP, aligned as i64),
            )
            .expect("aligned load"),
            Instruction::with2(
                Code::Mov_rm64_r64,
                MemoryOperand::with_base_displ(Register::RIP, aligned as i64),
                Register::RAX,
            )
            .expect("aligned store"),
            Instruction::with2(
                Code::Mov_rm64_r64,
                MemoryOperand::with_base_displ(Register::RIP, straddle as i64),
                Register::RAX,
            )
            .expect("straddle store"),
            Instruction::with(Code::Retnq),
        ],
    );
    let f = super::Program::new().bytes(code_va, &bytes).function(code_va);

    // window-runs: no dynamic range branch (`icmp ult`), direct constant slot GEPs, 1 dirty mark
    // for aligned store + 2 static dirty marks (slots 1 and 2) for straddling store (no `icmp ne`).
    let runs_ir = lift_module_with([&f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    assert!(
        !runs_ir.contains("icmp ult"),
        "constant in-window must elide range check:\n{runs_ir}"
    );
    assert!(
        !runs_ir.contains("icmp ne"),
        "constant in-window store must elide runtime crosses check:\n{runs_ir}"
    );
    assert!(
        runs_ir.contains("getelementptr i64, ptr @__playgta6_window_table, i64 1"),
        "{runs_ir}"
    );
    assert!(
        runs_ir.contains("getelementptr i8, ptr @__playgta6_window_dirty, i64 2"),
        "straddling store marks slot 2 statically:\n{runs_ir}"
    );
    assert_eq!(runs_ir.matches("store atomic i8 1, ptr ").count(), 3, "{runs_ir}");

    // window (!contiguous_runs): straddling constant store goes directly to `@__playgta6_slow_store8`.
    let win_ir = lift_module_with([&f], window(WindowCheck::Branch, true, false))
        .expect("lifts")
        .ir;
    assert!(!win_ir.contains("icmp ult"), "{win_ir}");
    assert!(!win_ir.contains("icmp ule"), "{win_ir}");
    assert!(
        win_ir.contains("call void @__playgta6_slow_store8("),
        "{win_ir}"
    );

    // Exact window boundary and u64::MAX wraparound checks:
    //  - W - 1 (0x3_FFFF_FFFF), W + S (0xA_0000_0000), and u64::MAX - 7 (0xFFFF_FFFF_FFFF_FFF8)
    //    must all stay flat (outside window).
    //  - W (0x4_0000_0000) and W + S - 8 (0x9_FFFF_FFF8) must fold to window_table slots 0 and
    //    (S >> 16) - 1 with zero range checks.
    for (label, va, expect_window, expect_slot) in [
        ("below_w", REAL_WINDOW_BASE - 1, false, 0u64),
        ("at_w", REAL_WINDOW_BASE, true, 0u64),
        (
            "at_end_minus_8",
            REAL_WINDOW_BASE + REAL_WINDOW_SIZE - 8,
            true,
            (REAL_WINDOW_SIZE >> 16) - 1,
        ),
        ("at_end", REAL_WINDOW_BASE + REAL_WINDOW_SIZE, false, 0u64),
        ("near_u64_max", u64::MAX - 7, false, 0u64),
    ] {
        let code_at = va.wrapping_sub(0x100);
        let b = super::encode(
            code_at,
            &[
                Instruction::with2(
                    Code::Mov_r64_rm64,
                    Register::RAX,
                    MemoryOperand::with_base_displ(Register::RIP, va as i64),
                )
                .expect("mov"),
                Instruction::with(Code::Retnq),
            ],
        );
        let func = super::Program::new().bytes(code_at, &b).function(code_at);
        let ir = lift_module_with([&func], window(WindowCheck::Branch, true, true))
            .expect("lifts")
            .ir;
        assert!(!ir.contains("icmp ult"), "{label}: no dynamic range check:\n{ir}");
        if expect_window {
            assert!(
                ir.contains(&format!(
                    "getelementptr i64, ptr @__playgta6_window_table, i64 {expect_slot}"
                )),
                "{label}: expected slot {expect_slot}:\n{ir}"
            );
        } else {
            assert!(
                !ir.contains("__playgta6_window_table"),
                "{label}: outside window must stay flat:\n{ir}"
            );
        }
    }
}

#[test]
fn avx256_stack_frame_and_burst_optimizations_reduce_window_checks() {
    use iced_x86::code_asm::*;
    // 1. 256-bit AVX stack spill/reload and frame-pointer `[rbp - 0x10]` skip window checks.
    let stack_f = super::asm_function(0x1000, |a| {
        a.push(rbp)?;
        a.mov(rbp, rsp)?;
        a.sub(rsp, 0x60)?;
        a.vmovdqu(ymmword_ptr(rsp + 0x20), ymm0)?;
        a.vmovdqu(ymm1, ymmword_ptr(rsp + 0x20))?;
        a.mov(qword_ptr(rbp - 0x10), rax)?;
        a.add(rax, qword_ptr(rbp - 0x10))?;
        a.add(rsp, 0x60)?;
        a.pop(rbp)?;
        a.ret()
    });
    let stack_ir = lift_module_with([&stack_f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    assert!(
        !stack_ir.contains("__playgta6_window_table")
            && !stack_ir.contains("__playgta6_window_dirty"),
        "AVX256 stack spills and frame-pointer accesses must stay flat:\n{stack_ir}"
    );

    // 2. Single 256-bit AVX load + store uses one 32-byte `<4 x i64>` access each (not two 16-byte).
    let avx_f = super::asm_function(0x2000, |a| {
        a.vmovdqu(ymm0, ymmword_ptr(rcx))?;
        a.vmovdqu(ymmword_ptr(rdx), ymm0)?;
        a.ret()
    });
    let avx_ir = lift_module_with([&avx_f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    assert_eq!(avx_ir.matches("load <4 x i64>, ptr ").count(), 1, "{avx_ir}");
    assert_eq!(
        avx_ir.matches("store <4 x i64> ").count(),
        2,
        "one flat store + one window store (32 bytes each):\n{avx_ir}"
    );
    assert_eq!(
        avx_ir.matches("@__playgta6_window_table").count(),
        3,
        "1 decl + 1 load lookup + 1 store lookup:\n{avx_ir}"
    );

    // 3. Burst of 4 AVX loads from same base `[rcx + 0/32/64/96]` emits only ONE window table load.
    let burst_f = super::asm_function(0x3000, |a| {
        a.vmovdqu(ymm0, ymmword_ptr(rcx))?;
        a.vmovdqu(ymm1, ymmword_ptr(rcx + 32))?;
        a.vmovdqu(ymm2, ymmword_ptr(rcx + 64))?;
        a.vmovdqu(ymm3, ymmword_ptr(rcx + 96))?;
        a.vpaddq(ymm0, ymm0, ymm1)?;
        a.vpaddq(ymm2, ymm2, ymm3)?;
        a.vpaddq(ymm0, ymm0, ymm2)?;
        a.ret()
    });
    let burst_ir = lift_module_with([&burst_f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    assert_eq!(
        burst_ir
            .matches("ptr @__playgta6_window_table, i64 ")
            .count(),
        1,
        "4 coalesced AVX loads must share one window_table lookup:\n{burst_ir}"
    );

    // 4. Read-modify-write `add [rcx], rax` under window-runs reuses the load's table lookup.
    let rmw_f = super::asm_function(0x4000, |a| {
        a.add(qword_ptr(rcx), rax)?;
        a.ret()
    });
    let rmw_ir = lift_module_with([&rmw_f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    assert_eq!(
        rmw_ir
            .matches("ptr @__playgta6_window_table, i64 ")
            .count(),
        1,
        "RMW under window-runs must reuse the load's window_table lookup:\n{rmw_ir}"
    );
}

#[test]
fn r63b_dual_base_interleaved_and_long_bursts() {
    use iced_x86::code_asm::*;

    // 1. Interleaved dual-base read/write burst (>8 instructions: 12 insns total):
    //    reads `[rcx + 0/16/32/48]` interleaved with writes `[rbx + 0/16/32/48]`.
    //    Must emit ONE combined fast path (`gbf*`), 2 window_table lookups (1 per base), and 1
    //    dirty mark (for written base `rbx` only).
    let dual_f = super::asm_function(0x1000, |a| {
        a.vmovdqu(xmm1, xmmword_ptr(rcx))?;
        a.vpaddd(xmm0, xmm1, xmm1)?;
        a.vmovdqu(xmmword_ptr(rbx), xmm0)?;
        a.vmovdqu(xmm2, xmmword_ptr(rcx + 16))?;
        a.vpaddd(xmm0, xmm2, xmm1)?;
        a.vmovdqu(xmmword_ptr(rbx + 16), xmm0)?;
        a.vmovdqu(xmm3, xmmword_ptr(rcx + 32))?;
        a.vpaddd(xmm0, xmm3, xmm2)?;
        a.vmovdqu(xmmword_ptr(rbx + 32), xmm0)?;
        a.vmovdqu(xmm4, xmmword_ptr(rcx + 48))?;
        a.vpaddd(xmm0, xmm4, xmm3)?;
        a.vmovdqu(xmmword_ptr(rbx + 48), xmm0)?;
        a.ret()
    });
    let dual_ir = lift_module_with([&dual_f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    assert_eq!(
        dual_ir.matches("\ngbf").count(),
        1,
        "interleaved dual-base burst must emit one combined fast path:\n{dual_ir}"
    );
    assert_eq!(
        dual_ir.matches("\ngbs").count(),
        1,
        "interleaved dual-base burst must emit one shared slow fallback:\n{dual_ir}"
    );
    assert_eq!(
        dual_ir
            .matches("ptr @__playgta6_window_table, i64 ")
            .count(),
        2,
        "dual-base burst must perform exactly 2 window_table lookups (1 per base):\n{dual_ir}"
    );
    assert_eq!(
        dual_ir.matches("load atomic i8, ptr ").count(),
        1,
        "only the written base rbx needs a dirty mark:\n{dual_ir}"
    );
    assert_eq!(
        dual_ir.matches("store atomic i8 1, ptr ").count(),
        1,
        "single coalesced dirty mark for rbx:\n{dual_ir}"
    );

    // 2. Long >8-instruction single-base burst (20 instructions, up to MAX_BURST_INSNS = 24):
    //    10 loads + 10 ALU ops from `[rcx + 0..80]` stay inside ONE burst.
    let long_f = super::asm_function(0x2000, |a| {
        a.xor(eax, eax)?;
        for k in 0..10 {
            a.mov(rdx, qword_ptr(rcx + (k * 8)))?;
            a.add(rax, rdx)?;
        }
        a.ret()
    });
    let long_ir = lift_module_with([&long_f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    assert_eq!(
        long_ir.matches("\ngbf").count(),
        1,
        "20-instruction single-base burst must stay in one burst (MAX_BURST_INSNS=24):\n{long_ir}"
    );
    assert_eq!(
        long_ir
            .matches("ptr @__playgta6_window_table, i64 ")
            .count(),
        1,
        "{long_ir}"
    );

    // 3. Dual-base burst where BOTH bases write emits 2 conditional dirty marks (`gbm*`).
    let both_write_f = super::asm_function(0x3000, |a| {
        a.mov(qword_ptr(rcx), rax)?;
        a.mov(qword_ptr(rbx), rax)?;
        a.mov(qword_ptr(rcx + 8), rdx)?;
        a.mov(qword_ptr(rbx + 8), rdx)?;
        a.ret()
    });
    let both_ir = lift_module_with([&both_write_f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    assert_eq!(both_ir.matches("\ngbf").count(), 1, "{both_ir}");
    assert_eq!(
        both_ir
            .matches("ptr @__playgta6_window_table, i64 ")
            .count(),
        2,
        "{both_ir}"
    );
    assert_eq!(both_ir.matches("\ngbm").count(), 2, "{both_ir}");
    assert_eq!(both_ir.matches("store atomic i8 1, ptr ").count(), 2, "{both_ir}");

    // 4. Stack-flat base (`rbp` copied from `rsp`) is skipped from a dual-base burst so only the
    //    non-stack base `rcx` pays a window check.
    let stack_dual_f = super::asm_function(0x4000, |a| {
        a.push(rbp)?;
        a.mov(rbp, rsp)?;
        a.sub(rsp, 0x40)?;
        a.vmovdqu(xmm0, xmmword_ptr(rcx))?;
        a.vmovdqu(xmmword_ptr(rbp - 0x10), xmm0)?;
        a.vmovdqu(xmm1, xmmword_ptr(rcx + 16))?;
        a.vmovdqu(xmmword_ptr(rbp - 0x20), xmm1)?;
        a.add(rsp, 0x40)?;
        a.pop(rbp)?;
        a.ret()
    });
    let stack_dual_ir = lift_module_with([&stack_dual_f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    assert_eq!(
        stack_dual_ir
            .matches("ptr @__playgta6_window_table, i64 ")
            .count(),
        1,
        "stack-flat rbp must be dropped from dual-base burst leaving 1-base burst on rcx:\n{stack_dual_ir}"
    );
    assert!(
        !stack_dual_ir.contains("__playgta6_window_dirty"),
        "stack-flat rbp stores must not emit window dirty marks:\n{stack_dual_ir}"
    );

    // 5. Soundness: modifying the second base GPR strictly inside the first base's burst must NOT
    //    merge it into a dual-base burst.
    let mod_base_f = super::asm_function(0x5000, |a| {
        a.vmovdqu(xmm0, xmmword_ptr(rcx))?;
        a.vmovdqu(xmmword_ptr(rbx), xmm0)?;
        a.add(rbx, 32)?;
        a.vmovdqu(xmm1, xmmword_ptr(rcx + 16))?;
        a.vmovdqu(xmmword_ptr(rbx), xmm1)?;
        a.ret()
    });
    let mod_ir = lift_module_with([&mod_base_f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    assert_eq!(
        mod_ir.matches("\ngbf").count(),
        1,
        "rcx still forms a 1-base burst:\n{mod_ir}"
    );
    assert_eq!(
        mod_ir.matches("\ngbc").count(),
        0,
        "modified rbx must NOT be merged as a second base in rcx's burst:\n{mod_ir}"
    );
}

#[test]
fn r64b_triple_base_and_32_insn_bursts() {
    use iced_x86::code_asm::*;

    // 1. Interleaved triple-base filter/blend burst (`[rcx + d]` + `[rdx + d]` -> `[rbx + d]`,
    //    16 instructions): must coalesce into ONE fast path (`gbf*`), ONE shared cold fallback
    //    (`gbs*`), 2 chained base-check continuations (`gbc*`), 3 `window_table` lookups (1 per
    //    base), and 1 coalesced dirty mark (written base `rbx` only).
    let triple_f = super::asm_function(0x1000, |a| {
        for k in 0..4i32 {
            let off = k * 16;
            a.vmovdqu(xmm0, xmmword_ptr(rcx + off))?;
            a.vmovdqu(xmm1, xmmword_ptr(rdx + off))?;
            a.vpaddw(xmm2, xmm0, xmm1)?;
            a.vmovdqu(xmmword_ptr(rbx + off), xmm2)?;
        }
        a.ret()
    });
    let triple_ir = lift_module_with([&triple_f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    assert_eq!(
        triple_ir.matches("\ngbf").count(),
        1,
        "interleaved 3-base burst must emit one combined fast path:\n{triple_ir}"
    );
    assert_eq!(
        triple_ir.matches("\ngbs").count(),
        1,
        "interleaved 3-base burst must emit one shared slow fallback:\n{triple_ir}"
    );
    assert_eq!(
        triple_ir.matches("\ngbc").count(),
        2,
        "3-base burst chains 2 intermediate base checks before gbf:\n{triple_ir}"
    );
    assert_eq!(
        triple_ir
            .matches("ptr @__playgta6_window_table, i64 ")
            .count(),
        3,
        "3-base burst must perform exactly 3 window_table lookups (1 per base):\n{triple_ir}"
    );
    assert_eq!(
        triple_ir.matches("load atomic i8, ptr ").count(),
        1,
        "only written base rbx needs a dirty mark:\n{triple_ir}"
    );
    assert_eq!(
        triple_ir.matches("store atomic i8 1, ptr ").count(),
        1,
        "single coalesced dirty mark for rbx:\n{triple_ir}"
    );

    // 2. Long 30-instruction (>24, <=32) 3-base burst stays inside ONE single burst.
    let long30_f = super::asm_function(0x2000, |a| {
        for k in 0..6i32 {
            let off = k * 16;
            a.vmovdqu(xmm0, xmmword_ptr(rcx + off))?;
            a.vmovdqu(xmm1, xmmword_ptr(rdx + off))?;
            a.vpaddw(xmm2, xmm0, xmm1)?;
            a.vpavgw(xmm2, xmm2, xmm0)?;
            a.vmovdqu(xmmword_ptr(rbx + off), xmm2)?;
        }
        a.ret()
    });
    let long30_ir = lift_module_with([&long30_f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    assert_eq!(
        long30_ir.matches("\ngbf").count(),
        1,
        "30-instruction 3-base burst must stay in ONE burst (MAX_BURST_INSNS=32):\n{long30_ir}"
    );
    assert_eq!(
        long30_ir
            .matches("ptr @__playgta6_window_table, i64 ")
            .count(),
        3,
        "30-instruction 3-base burst must emit 3 window_table lookups:\n{long30_ir}"
    );

    // 3. 3-base burst where 2 bases write (`rdx` and `rbx`) emits 2 conditional dirty marks.
    let two_writes_f = super::asm_function(0x3000, |a| {
        for k in 0..2i32 {
            let off = k * 16;
            a.vmovdqu(xmm0, xmmword_ptr(rcx + off))?;
            a.vmovdqu(xmmword_ptr(rdx + off), xmm0)?;
            a.vmovdqu(xmmword_ptr(rbx + off), xmm0)?;
        }
        a.ret()
    });
    let two_writes_ir = lift_module_with([&two_writes_f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    assert_eq!(two_writes_ir.matches("\ngbf").count(), 1, "{two_writes_ir}");
    assert_eq!(
        two_writes_ir
            .matches("ptr @__playgta6_window_table, i64 ")
            .count(),
        3,
        "{two_writes_ir}"
    );
    assert_eq!(two_writes_ir.matches("\ngbm").count(), 2, "{two_writes_ir}");
    assert_eq!(
        two_writes_ir.matches("store atomic i8 1, ptr ").count(),
        2,
        "{two_writes_ir}"
    );

    // 4. Soundness: modifying the 3rd base (`rbx`) strictly inside `[rcx, rdx]` burst must NOT
    //    merge `rbx` into the burst (only `rcx` + `rdx` form a 2-base burst).
    let mod3_f = super::asm_function(0x4000, |a| {
        a.vmovdqu(xmm0, xmmword_ptr(rcx))?;
        a.vmovdqu(xmm1, xmmword_ptr(rdx))?;
        a.vmovdqu(xmmword_ptr(rbx), xmm0)?;
        a.add(rbx, 32)?;
        a.vmovdqu(xmm2, xmmword_ptr(rcx + 16))?;
        a.vmovdqu(xmm3, xmmword_ptr(rdx + 16))?;
        a.vmovdqu(xmmword_ptr(rbx), xmm2)?;
        a.ret()
    });
    let mod3_ir = lift_module_with([&mod3_f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    assert_eq!(mod3_ir.matches("\ngbf").count(), 1, "{mod3_ir}");
    assert_eq!(
        mod3_ir.matches("\ngbc").count(),
        1,
        "only unmodified rcx+rdx may merge (2 bases -> 1 gbc); modified rbx stays outside:\n{mod3_ir}"
    );

    // 5. Soundness: modifying partner 2 (`rdx`) at `k` must stop partner 3 (`rbx`) from extending
    //    the burst past `k` (preventing stale `base_host` reuse if `rdx` were accessed after `k`).
    let mod_p2_f = super::asm_function(0x5000, |a| {
        a.vmovdqu(xmm0, xmmword_ptr(rcx))?;
        a.vmovdqu(xmm1, xmmword_ptr(rdx))?;
        a.vmovdqu(xmmword_ptr(rbx), xmm0)?;
        a.vmovdqu(xmm2, xmmword_ptr(rcx + 16))?;
        a.vmovdqu(xmm3, xmmword_ptr(rdx + 16))?;
        a.add(rdx, 64)?; // modifies rdx after rcx's burst ends
        a.vmovdqu(xmm4, xmmword_ptr(rdx))?; // uses NEW rdx!
        a.vmovdqu(xmmword_ptr(rbx + 16), xmm4)?;
        a.ret()
    });
    let mod_p2_ir = lift_module_with([&mod_p2_f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    assert_eq!(
        mod_p2_ir.matches("\ngbc").count(),
        1,
        "rdx modification must prevent rbx from extending the burst past `add rdx, 64`:\n{mod_p2_ir}"
    );
}

#[test]
fn r67_extended_64_insn_and_indexed_window_bursts() {
    use iced_x86::code_asm::*;

    // 1. Long 48-instruction (>32, <=64) unrolled single-base burst coalesces into ONE burst.
    let long48_f = super::asm_function(0x1000, |a| {
        a.xor(eax, eax)?;
        for k in 0..24i32 {
            a.mov(rdx, qword_ptr(rcx + k * 8))?;
            a.add(rax, rdx)?;
        }
        a.ret()
    });
    let long48_ir = lift_module_with([&long48_f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    assert_eq!(
        long48_ir.matches("\ngbf").count(),
        1,
        "48-instruction single-base burst must coalesce into ONE burst (MAX_BURST_INSNS=64):\n{long48_ir}"
    );
    assert_eq!(
        long48_ir
            .matches("ptr @__playgta6_window_table, i64 ")
            .count(),
        1,
        "48-instruction burst must emit 1 window_table lookup:\n{long48_ir}"
    );

    // 2. Indexed `[rcx + rsi*8 + disp]` read/write burst coalesces into ONE burst with ONE table
    //    lookup and ONE dirty mark.
    let idx_f = super::asm_function(0x2000, |a| {
        a.mov(rax, qword_ptr(rcx + rsi * 8))?;
        a.add(rax, qword_ptr(rcx + rsi * 8 + 8))?;
        a.add(rax, qword_ptr(rcx + rsi * 8 + 16))?;
        a.mov(qword_ptr(rcx + rsi * 8 + 24), rax)?;
        a.ret()
    });
    let idx_ir = lift_module_with([&idx_f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    assert_eq!(
        idx_ir.matches("\ngbf").count(),
        1,
        "indexed [rcx + rsi*8 + disp] burst must coalesce into ONE fast path:\n{idx_ir}"
    );
    assert_eq!(
        idx_ir.matches("\ngbs").count(),
        1,
        "indexed burst must emit ONE shared slow fallback:\n{idx_ir}"
    );
    assert_eq!(
        idx_ir
            .matches("ptr @__playgta6_window_table, i64 ")
            .count(),
        1,
        "indexed burst must emit ONE window_table lookup:\n{idx_ir}"
    );
    assert_eq!(
        idx_ir.matches("store atomic i8 1, ptr ").count(),
        1,
        "indexed burst must emit ONE coalesced dirty mark:\n{idx_ir}"
    );

    // 3. Multi-base burst mixing plain `[rdx + d]` + indexed `[rcx + r8*4 + d]` -> `[rbx + r8*4 + d]`.
    let mixed_f = super::asm_function(0x3000, |a| {
        for k in 0..4i32 {
            let off = k * 16;
            a.vmovdqu(xmm0, xmmword_ptr(rcx + r8 * 4 + off))?;
            a.vmovdqu(xmm1, xmmword_ptr(rdx + off))?;
            a.vpaddd(xmm2, xmm0, xmm1)?;
            a.vmovdqu(xmmword_ptr(rbx + r8 * 4 + off), xmm2)?;
        }
        a.ret()
    });
    let mixed_ir = lift_module_with([&mixed_f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    assert_eq!(mixed_ir.matches("\ngbf").count(), 1, "{mixed_ir}");
    assert_eq!(mixed_ir.matches("\ngbc").count(), 2, "{mixed_ir}");
    assert_eq!(
        mixed_ir
            .matches("ptr @__playgta6_window_table, i64 ")
            .count(),
        3,
        "3-base plain+indexed burst must emit 3 window_table lookups:\n{mixed_ir}"
    );
    assert_eq!(
        mixed_ir.matches("store atomic i8 1, ptr ").count(),
        1,
        "only written indexed base rbx+r8*4 needs a dirty mark:\n{mixed_ir}"
    );

    // 4. Soundness: modifying the index register `rsi` breaks an indexed burst (never reuses stale
    //    `eff_base`).
    let mod_idx_f = super::asm_function(0x4000, |a| {
        a.vmovdqu(xmm0, xmmword_ptr(rcx + rsi * 8))?;
        a.add(rsi, 2)?;
        a.vmovdqu(xmm1, xmmword_ptr(rcx + rsi * 8))?;
        a.ret()
    });
    let mod_idx_ir = lift_module_with([&mod_idx_f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    assert_eq!(
        mod_idx_ir.matches("\ngbf").count(),
        0,
        "modifying index reg rsi must prevent coalescing across `add rsi, 2`:\n{mod_idx_ir}"
    );

    // 5. Soundness: indexed `[rsp + rax*8 + disp]` must NOT be skipped as stack-flat even under
    //    `stack_flat = true` (it must still perform a window check!).
    let rsp_idx_f = super::asm_function(0x5000, |a| {
        a.mov(rcx, qword_ptr(rsp + rax * 8))?;
        a.mov(qword_ptr(rsp + rax * 8 + 8), rcx)?;
        a.ret()
    });
    let rsp_idx_ir = lift_module_with([&rsp_idx_f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    assert_eq!(
        rsp_idx_ir.matches("\ngbf").count(),
        1,
        "indexed [rsp + rax*8 + disp] must form a checked window burst (not stack-flat):\n{rsp_idx_ir}"
    );
    assert_eq!(
        rsp_idx_ir
            .matches("ptr @__playgta6_window_table, i64 ")
            .count(),
        1,
        "indexed [rsp + rax*8 + disp] must emit a window_table lookup:\n{rsp_idx_ir}"
    );
    assert_eq!(
        rsp_idx_ir.matches("store atomic i8 1, ptr ").count(),
        1,
        "indexed [rsp + rax*8 + disp] store must emit a dirty mark:\n{rsp_idx_ir}"
    );
}

#[test]
fn r71_cross_bb_stack_and_const_base_propagation_elides_window_checks() {
    use iced_x86::code_asm::*;

    // 1. Frame pointer `rbp` + derived stack base `r12` survive across a loop AND a diamond join
    //    -> ZERO window table lookups and ZERO dirty marks under `window-runs` and `window-stack`.
    let loop_stack_f = super::asm_function(0x1_4000_1000, |a| {
        let mut arm_b = a.create_label();
        let mut join = a.create_label();
        let mut loop_hdr = a.create_label();
        a.push(rbp)?;
        a.mov(rbp, rsp)?;
        a.sub(rsp, 0x60)?;
        a.test(ecx, ecx)?;
        a.jz(arm_b)?;
        a.lea(r12, qword_ptr(rbp - 0x10))?;
        a.jmp(join)?;
        a.set_label(&mut arm_b)?;
        a.lea(r12, qword_ptr(rbp - 0x20))?;
        a.set_label(&mut join)?;
        a.mov(eax, 4)?;
        a.set_label(&mut loop_hdr)?;
        a.mov(qword_ptr(rbp - 0x30), rax)?;
        a.add(rdx, qword_ptr(rbp - 0x30))?;
        a.mov(qword_ptr(r12 + 8), rdx)?;
        a.dec(eax)?;
        a.jnz(loop_hdr)?;
        a.add(rsp, 0x60)?;
        a.pop(rbp)?;
        a.ret()
    });
    for (name, model) in [
        ("window-runs", window(WindowCheck::Branch, true, true)),
        ("window-stack", window(WindowCheck::Branch, true, false)),
    ] {
        let ir = lift_module_with([&loop_stack_f], model).expect("lifts").ir;
        assert!(
            !ir.contains("__playgta6_window_table") && !ir.contains("__playgta6_window_dirty"),
            "{name}: cross-BB rbp & merged stack base r12 must stay flat:\n{ir}"
        );
    }

    // 2. Soundness: if one arm of a diamond overwrites `rbp` with a non-stack register `rcx`,
    //    the join block MUST NOT treat `[rbp - 8]` as stack-flat!
    let unsound_guard_f = super::asm_function(0x1_4000_2000, |a| {
        let mut use_rcx = a.create_label();
        let mut join = a.create_label();
        a.push(rbp)?;
        a.mov(rbp, rsp)?;
        a.test(edx, edx)?;
        a.jnz(use_rcx)?;
        a.jmp(join)?;
        a.set_label(&mut use_rcx)?;
        a.mov(rbp, rcx)?;
        a.set_label(&mut join)?;
        a.mov(rax, qword_ptr(rbp - 8))?;
        a.pop(rbp)?;
        a.ret()
    });
    let unsound_ir = lift_module_with([&unsound_guard_f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    assert!(
        unsound_ir.contains("__playgta6_window_table"),
        "modified rbp on one diamond arm must emit a window check at join:\n{unsound_ir}"
    );

    // 3. Cross-BB constant base pointer (`lea rbx, [rip + global_below_window]`) hoisted before a
    //    loop: all accesses `[rbx + 0/8/16]` inside the loop fold to constant out-of-window VAs
    //    with ZERO window checks and ZERO dirty marks.
    let g_base = 0x1_4AA0_1000u64;
    let code_const = 0x1_4000_3000u64;
    let const_loop_bytes = {
        use iced_x86::{Code, Instruction, MemoryOperand, Register};
        // Preheader: lea rbx, [rip + g_base]; xor eax, eax; mov ecx, 4
        let pre = super::encode(
            code_const,
            &[
                Instruction::with2(
                    Code::Lea_r64_m,
                    Register::RBX,
                    MemoryOperand::with_base_displ(Register::RIP, g_base as i64),
                )
                .expect("lea rip"),
                Instruction::with2(Code::Xor_r32_rm32, Register::EAX, Register::EAX).expect("xor"),
                Instruction::with2(Code::Mov_r32_imm32, Register::ECX, 4u32).expect("mov ecx"),
            ],
        );
        let loop_va = code_const + pre.len() as u64;
        let mut a = CodeAssembler::new(64).expect("asm");
        let mut loop_hdr = a.create_label();
        a.set_label(&mut loop_hdr).expect("lbl");
        a.add(rax, qword_ptr(rbx)).expect("add0");
        a.add(rax, qword_ptr(rbx + 8)).expect("add8");
        a.mov(qword_ptr(rbx + 16), rax).expect("st16");
        a.dec(ecx).expect("dec");
        a.jnz(loop_hdr).expect("jnz");
        a.ret().expect("ret");
        let mut out = pre;
        out.extend_from_slice(&a.assemble(loop_va).expect("assemble loop"));
        out
    };
    let const_loop_f = super::Program::new()
        .bytes(code_const, &const_loop_bytes)
        .function(code_const);
    let const_loop_ir =
        lift_module_with([&const_loop_f], window(WindowCheck::Branch, true, true))
            .expect("lifts")
            .ir;
    assert!(
        !const_loop_ir.contains("__playgta6_window_table")
            && !const_loop_ir.contains("__playgta6_window_dirty"),
        "loop-invariant constant global base must elide all window checks:\n{const_loop_ir}"
    );

    // 4. Soundness: modifying `rbx` inside the loop (`add rbx, 8`) invalidates its constant fact
    //    at the loop header so `[rbx]` MUST emit a window check.
    let mut_loop_f = super::asm_function(0x1_4000_4000, |a| {
        let mut loop_hdr = a.create_label();
        a.mov(rbx, g_base)?;
        a.xor(eax, eax)?;
        a.mov(ecx, 4)?;
        a.set_label(&mut loop_hdr)?;
        a.add(rax, qword_ptr(rbx))?;
        a.add(rbx, 8)?;
        a.dec(ecx)?;
        a.jnz(loop_hdr)?;
        a.ret()
    });
    let mut_loop_ir = lift_module_with([&mut_loop_f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    assert!(
        mut_loop_ir.contains("__playgta6_window_table"),
        "loop-variant rbx must NOT be treated as constant:\n{mut_loop_ir}"
    );

    // 5. Multi-entry window_load_cache: interleaved loads then stores to `[rcx]` and `[rdx]`
    //    under `window-runs` reuse both cached translations (only 2 table lookups total).
    let cache_f = super::asm_function(0x1_4000_5000, |a| {
        a.mov(rax, qword_ptr(rcx))?;
        a.mov(r8, qword_ptr(rdx))?;
        a.add(rax, r8)?;
        a.mov(qword_ptr(rcx), rax)?;
        a.mov(qword_ptr(rdx), r8)?;
        a.ret()
    });
    let cache_ir = lift_module_with([&cache_f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    assert_eq!(
        cache_ir
            .matches("ptr @__playgta6_window_table, i64 ")
            .count(),
        2,
        "interleaved load/store on rcx and rdx must reuse both cached translations:\n{cache_ir}"
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
