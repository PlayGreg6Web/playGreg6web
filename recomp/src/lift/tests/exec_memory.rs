//! Execution tests of the R19 table memory models (`lift/memory.rs`) under node: accesses that
//! straddle two window slots backed by non-adjacent pages (every width, SSE/AVX, `rep movs` /
//! `rep stos`), unmapped slots, a remap between two accesses, and atomics on threads inside the
//! window. Every case also runs under the base model as the reference. Tool-gated like `exec`.

use iced_x86::code_asm::*;

use super::asm_function;
use super::exec::{
    CODE_BASE, Case, GUEST_BASE, HarnessMemory, Outcome, RAX, RBX, RCX, RDI, RDX, RSP, STACK_TOP,
    WINDOW_BASE, WINDOW_SIZE, build_with_memory, exec_or_skip, run, window_backing,
};
use super::exec_integer::atomics_across_threads;
use crate::lift::GuestBase;
use crate::lift::memory::{Window, WindowCheck};
use crate::lift::target::LiftTarget;
use crate::lift::{MemoryModel, lift_module_with};

/// Window slot boundary `k` (slot `k - 1` | slot `k`); neighbours map to non-adjacent pages.
const fn boundary(k: u64) -> u64 {
    WINDOW_BASE + k * 0x1_0000
}

fn models() -> Vec<(&'static str, HarnessMemory)> {
    vec![
        ("base", HarnessMemory::base(GuestBase::Symbol)),
        (
            "window",
            HarnessMemory::window(&HarnessMemory::standard_slots()),
        ),
        (
            "window_select",
            HarnessMemory::window_with(&HarnessMemory::standard_slots(), WindowCheck::Select),
        ),
        (
            "window_outline",
            HarnessMemory::window_with(&HarnessMemory::standard_slots(), WindowCheck::Outline),
        ),
        (
            "window_stack",
            HarnessMemory::window_opts(
                &HarnessMemory::stack_identity_slots(),
                WindowCheck::Branch,
                true,
            ),
        ),
        (
            "page_stack",
            HarnessMemory::page_opts(
                &HarnessMemory::stack_identity_slots(),
                Some(GuestBase::Symbol),
            ),
        ),
        (
            "page",
            HarnessMemory::page(&HarnessMemory::standard_slots()),
        ),
    ]
}

fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(37).wrapping_add(seed) | 1)
        .collect()
}

fn le(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .rev()
        .fold(0u64, |acc, &b| (acc << 8) | u64::from(b))
}

fn ok(outcomes: &[Outcome], what: &str) {
    for (i, o) in outcomes.iter().enumerate() {
        assert!(o.error.is_empty(), "{what}: case {i} trapped: {}", o.error);
        assert_eq!(o.regs[RSP], STACK_TOP, "{what}: case {i}: rsp");
    }
}

#[test]
fn straddling_accesses_of_every_width_execute() {
    if !exec_or_skip("straddling_accesses_of_every_width_execute") {
        return;
    }
    // Loads at boundary(2) - k for k = 1, 3, 5, 7, 15 straddle slots 1 | 2; the stores straddle
    // boundary(3..=7). Widths 2, 4, 8 (GPR), 16 (SSE), 32 (AVX).
    let src = boundary(2);
    let f = asm_function(CODE_BASE, |a| {
        a.mov(rsi, src)?;
        a.mov(ax, word_ptr(rsi - 1))?;
        a.mov(ebx, dword_ptr(rsi - 3))?;
        a.mov(rcx, qword_ptr(rsi - 5))?;
        a.movdqu(xmm0, xmmword_ptr(rsi - 7))?;
        a.vmovdqu(ymm1, ymmword_ptr(rsi - 15))?;
        a.mov(rdi, boundary(3))?;
        a.mov(word_ptr(rdi - 1), ax)?;
        a.mov(rdi, boundary(4))?;
        a.mov(dword_ptr(rdi - 3), ebx)?;
        a.mov(rdi, boundary(5))?;
        a.mov(qword_ptr(rdi - 5), rcx)?;
        a.mov(rdi, boundary(6))?;
        a.movdqu(xmmword_ptr(rdi - 7), xmm0)?;
        a.mov(rdi, boundary(7))?;
        a.vmovdqu(ymmword_ptr(rdi - 15), ymm1)?;
        a.vzeroupper()?;
        a.ret()
    });
    let data = pattern(64, 11);
    let at = |k: u64| -> &[u8] { &data[(32 - k) as usize..] };
    for (name, memory) in models() {
        let built = build_with_memory(name, [&f], &[], &memory, GUEST_BASE);
        let case = Case::new(CODE_BASE)
            .mem(src - 32, &data)
            .read(boundary(3) - 1, 2)
            .read(boundary(4) - 3, 4)
            .read(boundary(5) - 5, 8)
            .read(boundary(6) - 7, 16)
            .read(boundary(7) - 15, 32);
        let o = run(&built, &[case]);
        ok(&o, name);
        assert_eq!(o[0].regs[RAX] & 0xFFFF, le(&at(1)[..2]), "{name}: word");
        assert_eq!(o[0].regs[RBX], le(&at(3)[..4]), "{name}: dword");
        assert_eq!(o[0].regs[RCX], le(&at(5)[..8]), "{name}: qword");
        assert_eq!(o[0].mem[0], at(1)[..2], "{name}: word store");
        assert_eq!(o[0].mem[1], at(3)[..4], "{name}: dword store");
        assert_eq!(o[0].mem[2], at(5)[..8], "{name}: qword store");
        assert_eq!(o[0].mem[3], at(7)[..16], "{name}: xmm round trip");
        assert_eq!(o[0].mem[4], at(15)[..32], "{name}: ymm round trip");
    }
}

#[test]
fn rep_movs_and_stos_across_slots_execute() {
    if !exec_or_skip("rep_movs_and_stos_across_slots_execute") {
        return;
    }
    // 0x200-byte rep movsb from across boundary(2) to across boundary(9), then rep stosq of a
    // repeated byte across boundary(10) (both bulk paths), and a byte-wise movsb loop (DF set).
    let (src, dst, fill) = (boundary(2) - 0x100, boundary(9) - 0x80, boundary(10) - 0x40);
    let f = asm_function(CODE_BASE, |a| {
        a.mov(rsi, src)?;
        a.mov(rdi, dst)?;
        a.mov(rcx, 0x200u64)?;
        a.rep().movsb()?;
        a.mov(rdi, fill)?;
        a.mov(rax, 0x5A5A_5A5A_5A5A_5A5Au64)?;
        a.mov(rcx, 0x10u64)?;
        a.rep().stosq()?;
        a.ret()
    });
    let data = pattern(0x200, 3);
    for (name, memory) in models() {
        let built = build_with_memory(name, [&f], &[], &memory, GUEST_BASE);
        let case = Case::new(CODE_BASE)
            .mem(src, &data)
            .read(dst, 0x200)
            .read(fill, 0x80);
        let o = run(&built, &[case]);
        ok(&o, name);
        assert_eq!(o[0].mem[0], data, "{name}: rep movsb");
        assert_eq!(o[0].mem[1], vec![0x5A; 0x80], "{name}: rep stosq");
    }
}

#[test]
fn unmapped_slots_trap() {
    if !exec_or_skip("unmapped_slots_trap") {
        return;
    }
    let load = asm_function(CODE_BASE, |a| {
        a.mov(rax, qword_ptr(rdi))?;
        a.ret()
    });
    // Window: slot 3 unmapped. Page table: slot 3 of the window and the null page unmapped.
    let mut slots = HarnessMemory::standard_slots();
    slots[3].1 = None;
    let window = HarnessMemory::window(&slots);
    let window_select = HarnessMemory::window_with(&slots, WindowCheck::Select);
    slots.push((0, None));
    let page = HarnessMemory::page(&slots);
    let inside = boundary(3) + 0x10;
    let straddle = boundary(3) - 2;
    for (name, memory) in [
        ("window_unmapped", window),
        ("window_select_unmapped", window_select),
        ("page_unmapped", page),
    ] {
        let built = build_with_memory(name, [&load], &[], &memory, GUEST_BASE);
        let mut cases = vec![
            Case::new(CODE_BASE).reg(RDI, inside),
            Case::new(CODE_BASE).reg(RDI, straddle),
            Case::new(CODE_BASE).reg(RDI, boundary(4) + 8),
        ];
        if name == "page_unmapped" {
            cases.push(Case::new(CODE_BASE).reg(RDI, 0x10)); // null page
            cases.push(Case::new(CODE_BASE).reg(RDI, 0x1_0000_0010)); // beyond the table
        }
        let o = run(&built, &cases);
        assert!(
            o[0].error.contains("out of bounds"),
            "{name}: unmapped slot must trap: {:?}",
            o[0].error
        );
        assert!(
            o[1].error.contains("unmapped guest page"),
            "{name}: straddling into an unmapped slot must trap: {:?}",
            o[1].error
        );
        assert!(o[2].error.is_empty(), "{name}: mapped slot: {}", o[2].error);
        for extra in &o[3..] {
            assert!(
                extra.error.contains("out of bounds"),
                "{name}: must trap: {:?}",
                extra.error
            );
        }
    }
}

#[test]
fn remapped_slot_is_reloaded_after_a_call() {
    if !exec_or_skip("remapped_slot_is_reloaded_after_a_call") {
        return;
    }
    // rax = [x]; remap x's slot onto the page of slot 6 (host function 0x7FFE0030); rbx = [x].
    let x = boundary(5) + 0x40;
    let twin = boundary(6) + 0x40; // same offset in slot 6
    let f = asm_function(CODE_BASE, |a| {
        a.mov(rax, qword_ptr(rdi))?;
        a.push(rax)?;
        a.sub(rsp, 0x28)?;
        a.mov(rcx, rdi)?;
        a.mov(rdx, window_backing(6))?;
        a.mov(r11, 0x7FFE_0030u64)?;
        a.call(r11)?;
        a.add(rsp, 0x28)?;
        a.mov(rbx, qword_ptr(rdi))?;
        a.pop(rax)?;
        a.ret()
    });
    for (name, memory) in models().into_iter().skip(1) {
        let built = build_with_memory(name, [&f], &[], &memory, GUEST_BASE);
        let case = Case::new(CODE_BASE)
            .reg(RDI, x)
            .mem(x, &0x1111_1111_1111_1111u64.to_le_bytes())
            .mem(twin, &0x2222_2222_2222_2222u64.to_le_bytes());
        let o = run(&built, &[case]);
        ok(&o, name);
        assert_eq!(
            o[0].regs[RAX], 0x1111_1111_1111_1111,
            "{name}: before the remap"
        );
        assert_eq!(
            o[0].regs[RBX], 0x2222_2222_2222_2222,
            "{name}: after the remap"
        );
    }
}

#[test]
fn multi_return_and_mixed_traps_execute_identically_across_memory_models() {
    if !exec_or_skip("multi_return_and_mixed_traps_execute_identically_across_memory_models") {
        return;
    }
    // Exercises shared ret_epilog, shared mixed cold_trap (int3 + divide error), hoisted
    // guest_base, constant-count shifts, same-register xor/test, and windowed/straddling loads &
    // stores across every memory model.
    let f = asm_function(CODE_BASE, |a| {
        let mut early = a.create_label();
        let mut trap_lbl = a.create_label();
        a.test(rcx, rcx)?;
        a.jz(early)?;
        a.mov(rax, qword_ptr(rcx))?;
        a.shl(rax, 2)?;
        a.mov(qword_ptr(rdi), rax)?;
        a.test(r8, r8)?;
        a.js(trap_lbl)?;
        a.xor(edx, edx)?;
        a.div(r8)?;
        a.ret()?;
        a.set_label(&mut early)?;
        a.xor(eax, eax)?;
        a.ret()?;
        a.set_label(&mut trap_lbl)?;
        a.int3()
    });
    let src = boundary(2) - 3; // straddles slots 1 | 2
    let dst = boundary(4) - 5; // straddles slots 3 | 4
    let input = 0x1234_5678u64;
    for (name, memory) in models() {
        let built = build_with_memory(name, [&f], &[], &memory, GUEST_BASE);
        let cases = [
            // Normal path: loads straddling qword, shifts << 2, stores straddling qword, divides by 4.
            Case::new(CODE_BASE)
                .reg(RCX, src)
                .reg(RDI, dst)
                .reg(super::exec::R8, 4)
                .mem(src, &input.to_le_bytes())
                .read(dst, 8),
            // Early-return path (shared ret_epilog).
            Case::new(CODE_BASE).reg(RCX, 0),
            // Divide-by-zero path (shared cold_trap -> divide_error).
            Case::new(CODE_BASE)
                .reg(RCX, src)
                .reg(RDI, dst)
                .reg(super::exec::R8, 0)
                .mem(src, &input.to_le_bytes()),
            // Trap path (shared cold_trap -> trap).
            Case::new(CODE_BASE)
                .reg(RCX, src)
                .reg(RDI, dst)
                .reg(super::exec::R8, u64::MAX)
                .mem(src, &input.to_le_bytes()),
        ];
        let o = run(&built, &cases);
        ok(&o[..2], name);
        assert_eq!(o[0].regs[RAX], input, "{name}: (input << 2) / 4");
        assert_eq!(
            o[0].mem[0],
            (input << 2).to_le_bytes(),
            "{name}: straddling store"
        );
        assert_eq!(o[1].regs[RAX], 0, "{name}: early return");
        assert!(
            o[2].error.contains("divide error"),
            "{name}: divide error: {:?}",
            o[2].error
        );
        assert!(
            o[3].error.contains("trap at"),
            "{name}: int3 trap: {:?}",
            o[3].error
        );
    }
}

#[test]
fn atomics_in_a_window_across_threads() {
    // The counters (DATA_ADDR) and the four guest stacks lie inside the harness window.
    atomics_across_threads(
        "atomics_in_a_window_across_threads",
        &HarnessMemory::window(&HarnessMemory::standard_slots()),
    );
}

/// Writes the lifted fixture of the runtime's window end-to-end test
/// (tests/test_guest_window_e2e.cpp) into the directory named by `PLAYGTA6_E2E_LIFT_DIR`:
/// `sub_140001000` reads two qwords at `[rcx]` and `[rcx + 8]` (window addresses, possibly
/// straddling a slot), stores their sum to `[rdx]` (window) and `[r8]` (outside), returns it in
/// rax. Lifted with the window at the runtime's placement (16 GiB, 24 GiB) under the flat
/// translations the runtime uses: identity (wasm) and the base symbol (native), as wasm64 IR and,
/// for the symbol variant, retargeted for the host.
#[test]
fn writes_the_runtime_window_fixture() {
    let Some(dir) = std::env::var_os("PLAYGTA6_E2E_LIFT_DIR") else {
        println!("SKIP writes_the_runtime_window_fixture: PLAYGTA6_E2E_LIFT_DIR not set");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    std::fs::create_dir_all(&dir).expect("create output dir");
    let f = asm_function(0x1_4000_1000, |a| {
        a.mov(rax, qword_ptr(rcx))?;
        a.add(rax, qword_ptr(rcx + 8))?;
        a.mov(qword_ptr(rdx), rax)?;
        a.mov(qword_ptr(r8), rax)?;
        a.ret()
    });
    let window = |flat| {
        MemoryModel::Windowed(Window {
            flat,
            base: 0x4_0000_0000,
            size: 0x6_0000_0000,
            check: WindowCheck::Branch,
            stack_flat: true,
            contiguous_runs: false,
        })
    };
    for (name, model, native) in [
        (
            "e2e_window_identity.ll",
            window(GuestBase::Constant(0)),
            false,
        ),
        ("e2e_window_symbol.ll", window(GuestBase::Symbol), false),
        (
            "e2e_window_symbol_native.ll",
            window(GuestBase::Symbol),
            true,
        ),
    ] {
        let module = lift_module_with([&f], model).expect("lifts");
        assert!(module.functions[0].unsupported.is_empty());
        let ir = if native {
            LiftTarget::native_host().retarget(module.ir)
        } else {
            module.ir
        };
        std::fs::write(dir.join(name), ir).expect("write .ll");
    }
}

// ----- O37: window write marks ------------------------------------------------------------------

/// A guest VA outside the harness window (flat translation), free in the harness memory map.
const FLAT_ADDR: u64 = 0x28_0000;

/// Window slot of harness VA `va`.
fn slot(va: u64) -> u64 {
    (va - WINDOW_BASE) >> 16
}

/// The models the write marks apply to, plus the reference and a table model without a window.
fn mark_models() -> Vec<(&'static str, HarnessMemory)> {
    vec![
        ("base", HarnessMemory::base(GuestBase::Symbol)),
        (
            "page",
            HarnessMemory::page(&HarnessMemory::standard_slots()),
        ),
        (
            "window",
            HarnessMemory::window(&HarnessMemory::standard_slots()),
        ),
        (
            "window_select",
            HarnessMemory::window_with(&HarnessMemory::standard_slots(), WindowCheck::Select),
        ),
        (
            "window_outline",
            HarnessMemory::window_with(&HarnessMemory::standard_slots(), WindowCheck::Outline),
        ),
        (
            "window_runs",
            HarnessMemory::window_full(
                &HarnessMemory::contiguous_slots(),
                WindowCheck::Branch,
                true,
                true,
            ),
        ),
    ]
}

/// The marks a case should leave under `model`: `slots` for the window models, nothing without a
/// window; the branch-free check also marks its spare entry (the slot count) for a store outside.
fn expected_marks(model: &str, slots: &[u64], outside_store: bool) -> Vec<u64> {
    match model {
        "base" | "page" => Vec::new(),
        _ => {
            let mut v: Vec<u64> = slots.to_vec();
            if model == "window_select" && outside_store {
                v.push(WINDOW_SIZE >> 16);
            }
            v.sort_unstable();
            v.dedup();
            v
        }
    }
}

#[test]
fn stores_into_the_window_mark_their_slots() {
    if !exec_or_skip("stores_into_the_window_mark_their_slots") {
        return;
    }
    // f(rcx, rdx): mov [rcx], rax; mov [rdx], eax; ret - one store into the window (or a
    // straddling one), one outside.
    let f = asm_function(CODE_BASE, |a| {
        a.mov(qword_ptr(rcx), rax)?;
        a.mov(dword_ptr(rdx), eax)?;
        a.ret()
    });
    let aligned = boundary(2) + 0x100;
    let straddle = boundary(4) - 3;
    for (name, memory) in mark_models() {
        let built = build_with_memory(name, [&f], &[], &memory, GUEST_BASE);
        let cases = [
            Case::new(CODE_BASE)
                .reg(RAX, 0x1122_3344_5566_7788)
                .reg(RCX, aligned)
                .reg(RDX, FLAT_ADDR)
                .read(aligned, 8),
            // A straddling store: the slow path (or, with contiguous runs, the lifted code) marks
            // both slots.
            Case::new(CODE_BASE)
                .reg(RAX, 0x0102_0304_0506_0708)
                .reg(RCX, straddle)
                .reg(RDX, FLAT_ADDR)
                .read(straddle, 8),
            // Both stores outside the window: no slot marked.
            Case::new(CODE_BASE)
                .reg(RAX, 7)
                .reg(RCX, FLAT_ADDR + 0x40)
                .reg(RDX, FLAT_ADDR),
        ];
        let o = run(&built, &cases);
        ok(&o, name);
        assert_eq!(le(&o[0].mem[0]), 0x1122_3344_5566_7788, "{name}");
        assert_eq!(le(&o[1].mem[0]), 0x0102_0304_0506_0708, "{name}");
        assert_eq!(
            o[0].dirty,
            expected_marks(name, &[slot(aligned)], true),
            "{name}: aligned"
        );
        assert_eq!(
            o[1].dirty,
            expected_marks(name, &[slot(straddle), slot(straddle) + 1], true),
            "{name}: straddling"
        );
        assert_eq!(
            o[2].dirty,
            expected_marks(name, &[], true),
            "{name}: outside"
        );
    }
}

#[test]
fn loads_mark_nothing() {
    if !exec_or_skip("loads_mark_nothing") {
        return;
    }
    let f = asm_function(CODE_BASE, |a| {
        a.mov(rax, qword_ptr(rcx))?;
        a.add(rax, qword_ptr(rcx + 0xFFF8))?;
        a.ret()
    });
    let at = boundary(3) - 0x10;
    for (name, memory) in mark_models() {
        let built = build_with_memory(name, [&f], &[], &memory, GUEST_BASE);
        let o = run(
            &built,
            &[Case::new(CODE_BASE).reg(RCX, at).mem(at, &[1; 8])],
        );
        ok(&o, name);
        assert!(o[0].dirty.is_empty(), "{name}: {:?}", o[0].dirty);
    }
}

#[test]
fn rep_movs_and_stos_into_the_window_mark_the_destination() {
    if !exec_or_skip("rep_movs_and_stos_into_the_window_mark_the_destination") {
        return;
    }
    // rep movsb from outside the window to a range over slots 5 | 6, then rep stosb over 7 | 8.
    let f = asm_function(CODE_BASE, |a| {
        a.rep().movsb()?;
        a.mov(rdi, r8)?;
        a.mov(rcx, r9)?;
        a.rep().stosb()?;
        a.ret()
    });
    let dst = boundary(6) - 100;
    let fill = boundary(8) - 10;
    let src_bytes = pattern(300, 3);
    for (name, memory) in mark_models() {
        let built = build_with_memory(name, [&f], &[], &memory, GUEST_BASE);
        let o = run(
            &built,
            &[Case::new(CODE_BASE)
                .reg(super::exec::RSI, FLAT_ADDR)
                .reg(RDI, dst)
                .reg(RCX, 300)
                .reg(RAX, 0x5A)
                .reg(super::exec::R8, fill)
                .reg(super::exec::R9, 20)
                .mem(FLAT_ADDR, &src_bytes)
                .read(dst, 300)
                .read(fill, 20)],
        );
        ok(&o, name);
        assert_eq!(o[0].mem[0], src_bytes, "{name}: copy");
        assert_eq!(o[0].mem[1], vec![0x5A; 20], "{name}: fill");
        assert_eq!(
            o[0].dirty,
            expected_marks(
                name,
                &[slot(dst), slot(dst) + 1, slot(fill), slot(fill) + 1],
                false
            ),
            "{name}"
        );
    }
}

#[test]
fn atomics_into_the_window_mark_their_slot() {
    if !exec_or_skip("atomics_into_the_window_mark_their_slot") {
        return;
    }
    // lock xadd [rcx], rax; xchg [rdx], rbx; lock cmpxchg [r8], r9 (rax holds the old value of
    // [rcx] after the xadd, so the cmpxchg compares with that); lock bts [r10], 3; ret.
    let f = asm_function(CODE_BASE, |a| {
        a.lock().xadd(qword_ptr(rcx), rax)?;
        a.xchg(qword_ptr(rdx), rbx)?;
        a.lock().cmpxchg(qword_ptr(r8), r9)?;
        a.lock().bts(qword_ptr(r10), 3)?;
        a.ret()
    });
    let xadd_at = boundary(1) + 0x40;
    let xchg_at = boundary(3) + 0x80;
    let cmpxchg_at = boundary(5) + 0x100;
    let bts_at = boundary(7) + 0x200;
    for (name, memory) in mark_models() {
        let built = build_with_memory(name, [&f], &[], &memory, GUEST_BASE);
        let inside = Case::new(CODE_BASE)
            .reg(RAX, 1)
            .reg(RCX, xadd_at)
            .reg(super::exec::RDX, xchg_at)
            .reg(RBX, 9)
            .reg(super::exec::R8, cmpxchg_at)
            .reg(super::exec::R9, 0x77)
            .reg(super::exec::R10, bts_at)
            .mem(xadd_at, &[0; 8])
            .mem(cmpxchg_at, &[5, 0, 0, 0, 0, 0, 0, 0])
            .mem(bts_at, &[0; 8])
            .read(xchg_at, 8)
            .read(bts_at, 8);
        let outside = Case::new(CODE_BASE)
            .reg(RAX, 1)
            .reg(RCX, FLAT_ADDR)
            .reg(super::exec::RDX, FLAT_ADDR + 8)
            .reg(super::exec::R8, FLAT_ADDR + 16)
            .reg(super::exec::R10, FLAT_ADDR + 24)
            .mem(FLAT_ADDR, &[0; 32]);
        let o = run(&built, &[inside, outside]);
        ok(&o, name);
        assert_eq!(le(&o[0].mem[0]), 9, "{name}: xchg");
        assert_eq!(le(&o[0].mem[1]), 8, "{name}: bts");
        assert_eq!(
            o[0].dirty,
            expected_marks(
                name,
                &[slot(xadd_at), slot(xchg_at), slot(cmpxchg_at), slot(bts_at)],
                false
            ),
            "{name}: inside"
        );
        assert_eq!(
            o[1].dirty,
            expected_marks(name, &[], true),
            "{name}: outside"
        );
    }
}

#[test]
fn default_output_has_no_write_marks() {
    let f = asm_function(CODE_BASE, |a| {
        a.mov(qword_ptr(rcx), rax)?;
        a.lock().xadd(qword_ptr(rcx), rax)?;
        a.ret()
    });
    let ir = lift_module_with([&f], MemoryModel::default())
        .expect("lifts")
        .ir;
    assert!(!ir.contains("__playgta6_window_dirty"), "{ir}");
    assert!(!ir.contains("store atomic i8 1"), "{ir}");
}
