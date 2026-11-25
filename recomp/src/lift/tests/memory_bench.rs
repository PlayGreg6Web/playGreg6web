//! R19 run-time measurements (tool-gated, local only): memory-heavy lifted micro-benchmarks under
//! node for every memory model. Runs only with `PLAYGTA6_R19_BENCH=1` (it takes minutes);
//! `PLAYGTA6_R19_SAMPLES` sets the samples per bench (default 15). Prints one table row per
//! model, scenario and bench: median / min / max nanoseconds per element.

use std::env;
use std::fmt::Write as _;

use iced_x86::code_asm::*;

use super::Program;
use super::exec::{
    CODE_BASE, GUEST_BASE, HarnessMemory, STATE_ADDR, WINDOW_BASE, build_with_memory, exec_or_skip,
    run_script,
};
use crate::lift::memory::WindowCheck;
use crate::lift::{GuestBase, STATE_GPR_OFFSET};

const BENCH_JS: &str = include_str!("bench.js");

pub(super) const SUM: u64 = CODE_BASE;
pub(super) const CHASE: u64 = CODE_BASE + 0x100;
pub(super) const COPY: u64 = CODE_BASE + 0x200;
pub(super) const STRUCTS: u64 = CODE_BASE + 0x300;
pub(super) const LEAF: u64 = CODE_BASE + 0x400;
pub(super) const REP_COPY: u64 = CODE_BASE + 0x500;

/// Guest VA of data outside the harness window (linear 0xC0_0000 under the base translation).
const FLAT_DATA: u64 = 0x40_0000;
/// Guest stack top outside the window.
const BENCH_STACK: u64 = 0x3F_0000;

pub(super) fn program() -> Program {
    Program::new()
        // rax = sum of rcx qwords at rdx
        .asm(SUM, |a| {
            let mut top = a.create_label();
            a.xor(eax, eax)?;
            a.set_label(&mut top)?;
            a.add(rax, qword_ptr(rdx + rcx * 8 - 8))?;
            a.dec(rcx)?;
            a.jnz(top)?;
            a.ret()
        })
        // follow rcx links from rdx
        .asm(CHASE, |a| {
            let mut top = a.create_label();
            a.set_label(&mut top)?;
            a.mov(rdx, qword_ptr(rdx))?;
            a.dec(rcx)?;
            a.jnz(top)?;
            a.mov(rax, rdx)?;
            a.ret()
        })
        // copy rcx qwords from rdx to r8, one at a time
        .asm(COPY, |a| {
            let mut top = a.create_label();
            a.set_label(&mut top)?;
            a.mov(rax, qword_ptr(rdx))?;
            a.mov(qword_ptr(r8), rax)?;
            a.add(rdx, 8)?;
            a.add(r8, 8)?;
            a.dec(rcx)?;
            a.jnz(top)?;
            a.ret()
        })
        // "typical function": per 32-byte struct load two fields, multiply by a third, store the
        // result, call a leaf (guest stack traffic), accumulate
        .asm(STRUCTS, |a| {
            let mut top = a.create_label();
            a.push(rbx)?;
            a.xor(r9d, r9d)?;
            a.set_label(&mut top)?;
            a.mov(rax, qword_ptr(rdx))?;
            a.add(rax, qword_ptr(rdx + 8))?;
            a.imul_2(rax, qword_ptr(rdx + 16))?;
            a.mov(qword_ptr(rdx + 24), rax)?;
            a.call(LEAF)?;
            a.add(r9, rax)?;
            a.add(rdx, 32)?;
            a.dec(rcx)?;
            a.jnz(top)?;
            a.mov(rax, r9)?;
            a.pop(rbx)?;
            a.ret()
        })
        .asm(LEAF, |a| {
            a.push(rbx)?;
            a.mov(rbx, rax)?;
            a.shr(rbx, 3)?;
            a.xor(rax, rbx)?;
            a.pop(rbx)?;
            a.ret()
        })
        // rep movsq of rcx qwords from rdx to r8 (bulk path)
        .asm(REP_COPY, |a| {
            a.push(rsi)?;
            a.push(rdi)?;
            a.mov(rsi, rdx)?;
            a.mov(rdi, r8)?;
            a.rep().movsq()?;
            a.pop(rdi)?;
            a.pop(rsi)?;
            a.ret()
        })
}

fn benches(base: u64) -> String {
    let n = 1u64 << 16; // 64 Ki qwords = 512 KiB
    let rows = [
        format!(
            "{{\"name\":\"array_sum\",\"fn\":\"sub_{SUM:X}\",\"kind\":\"array\",\"base\":\"{base:#x}\",\"n\":{n},\"calls\":20}}"
        ),
        format!(
            "{{\"name\":\"pointer_chase\",\"fn\":\"sub_{CHASE:X}\",\"kind\":\"chase\",\"base\":\"{base:#x}\",\"n\":8192,\"steps\":65536,\"calls\":20}}"
        ),
        format!(
            "{{\"name\":\"copy_loop\",\"fn\":\"sub_{COPY:X}\",\"kind\":\"copy\",\"base\":\"{base:#x}\",\"n\":{},\"dst\":\"{:#x}\",\"calls\":20}}",
            n / 2,
            base + 0x4_0000
        ),
        format!(
            "{{\"name\":\"typical_structs\",\"fn\":\"sub_{STRUCTS:X}\",\"kind\":\"structs\",\"base\":\"{base:#x}\",\"n\":{},\"calls\":20}}",
            n / 4
        ),
        format!(
            "{{\"name\":\"rep_movsq\",\"fn\":\"sub_{REP_COPY:X}\",\"kind\":\"copy\",\"base\":\"{base:#x}\",\"n\":{},\"dst\":\"{:#x}\",\"calls\":20}}",
            n / 2,
            base + 0x4_0000
        ),
    ];
    rows.join(",")
}

#[test]
fn r19_benchmark_memory_models() {
    if env::var_os("PLAYGTA6_R19_BENCH").is_none() {
        println!("SKIP r19_benchmark_memory_models: PLAYGTA6_R19_BENCH not set");
        return;
    }
    if !exec_or_skip("r19_benchmark_memory_models") {
        return;
    }
    let samples: u32 = env::var("PLAYGTA6_R19_SAMPLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(15);
    let functions = program().discover();
    let standard = HarnessMemory::standard_slots();
    let stack = HarnessMemory::stack_identity_slots();
    let runs = HarnessMemory::contiguous_slots();
    // (model, scenario, data base)
    let runs: Vec<(&str, HarnessMemory, &str, u64)> = vec![
        (
            "identity",
            HarnessMemory::base(GuestBase::Constant(0)),
            "flat",
            FLAT_DATA,
        ),
        (
            "symbol",
            HarnessMemory::base(GuestBase::Symbol),
            "flat",
            FLAT_DATA,
        ),
        (
            "window",
            HarnessMemory::window(&standard),
            "flat",
            FLAT_DATA,
        ),
        (
            "window",
            HarnessMemory::window(&standard),
            "in-window",
            WINDOW_BASE,
        ),
        (
            "window-stack",
            HarnessMemory::window_opts(&stack, WindowCheck::Branch, true),
            "flat",
            FLAT_DATA,
        ),
        (
            "window-stack",
            HarnessMemory::window_opts(&stack, WindowCheck::Branch, true),
            "in-window",
            WINDOW_BASE,
        ),
        (
            "window-outline-stack",
            HarnessMemory::window_opts(&stack, WindowCheck::Outline, true),
            "flat",
            FLAT_DATA,
        ),
        (
            "window-outline-stack",
            HarnessMemory::window_opts(&stack, WindowCheck::Outline, true),
            "in-window",
            WINDOW_BASE,
        ),
        (
            "window-runs",
            HarnessMemory::window_full(&runs, WindowCheck::Branch, true, true),
            "flat",
            FLAT_DATA,
        ),
        (
            "window-runs",
            HarnessMemory::window_full(&runs, WindowCheck::Branch, true, true),
            "in-window",
            WINDOW_BASE,
        ),
        (
            "page-runs",
            HarnessMemory::page_full(&runs, Some(GuestBase::Symbol), true),
            "flat",
            FLAT_DATA,
        ),
        (
            "page-runs",
            HarnessMemory::page_full(&runs, Some(GuestBase::Symbol), true),
            "in-window",
            WINDOW_BASE,
        ),
        (
            "window-select",
            HarnessMemory::window_with(&standard, WindowCheck::Select),
            "flat",
            FLAT_DATA,
        ),
        (
            "window-select",
            HarnessMemory::window_with(&standard, WindowCheck::Select),
            "in-window",
            WINDOW_BASE,
        ),
        ("page", HarnessMemory::page(&standard), "flat", FLAT_DATA),
        (
            "page",
            HarnessMemory::page(&standard),
            "in-window",
            WINDOW_BASE,
        ),
        (
            "page-stack",
            HarnessMemory::page_opts(&stack, Some(GuestBase::Symbol)),
            "flat",
            FLAT_DATA,
        ),
        (
            "page-stack",
            HarnessMemory::page_opts(&stack, Some(GuestBase::Symbol)),
            "in-window",
            WINDOW_BASE,
        ),
    ];
    let mut table = String::from("model\tscenario\tbench\tmedian_ns\tmin_ns\tmax_ns\tchecksum\n");
    for (model, memory, scenario, base) in runs {
        let label = format!("bench_{model}_{scenario}");
        // The constant-0 identity build addresses linear memory directly: keep its data where the
        // other models' flat translation puts it.
        let (data, guest_base) = match memory.model {
            crate::lift::MemoryModel::Base(GuestBase::Constant(0)) => (base + GUEST_BASE, 0),
            _ => (base, GUEST_BASE),
        };
        let stack = if guest_base == 0 {
            BENCH_STACK + GUEST_BASE
        } else {
            BENCH_STACK
        };
        if let Some(dir) = env::var_os("PLAYGTA6_R19_DUMP") {
            // The lifted IR of each model, for inspecting what LLVM makes of the translation.
            let module =
                crate::lift::lift_module_with(functions.values(), memory.model).expect("lifts");
            let path = std::path::Path::new(&dir).join(format!("{label}.ll"));
            std::fs::write(path, module.ir).expect("write IR");
        }
        let built = build_with_memory(&label, functions.values(), &[], &memory, GUEST_BASE);
        let memory_json = if built.memory_json.is_empty() {
            String::new()
        } else {
            format!("\"memory\":{},", built.memory_json)
        };
        let spec = format!(
            "{{\"state\":\"{STATE_ADDR:#x}\",\"gprOffset\":{STATE_GPR_OFFSET},\"guestBase\":\"{:#x}\",{memory_json}\"rsp\":\"{stack:#x}\",\"samples\":{samples},\"benches\":[{}]}}",
            built.guest_base,
            benches(data)
        );
        let out = run_script(&built, "bench.js", BENCH_JS, &[spec]);
        for line in out.lines() {
            let _ = writeln!(table, "{model}\t{scenario}\t{line}");
        }
    }
    print!("{table}");
}
