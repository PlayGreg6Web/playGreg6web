//! Store-backs of registers whose alloca still matches their `%state` slot (`Known::synced_state`)
//! are skipped: a register stored before a call to a callee that only reads it is not stored
//! again before the next call, a tail jump or the return, across extended blocks, joins whose
//! predecessors all agree, the shared `ret_epilog` and tail-call exits. A write in between, or a
//! loop head, keeps the store. The results stay exact.

use iced_x86::code_asm::*;

use super::Program;
use super::exec::{CODE_BASE, Case, RAX, RCX, RDX, build, exec_or_skip, run};
use crate::lift::{function_symbol, lift_module};

/// `mov rax, rdx; ret`: reads `rdx`, writes only `rax` (and `rsp`).
const READER: u64 = CODE_BASE;
/// `add rdx, 1; call READER; call READER; add rax, rdx; ret`.
const TWICE: u64 = CODE_BASE + 0x100;
/// `add rdx, 1; call READER; add rdx, 1; call READER; ret`: `rdx` changes between the calls.
const CHANGED: u64 = CODE_BASE + 0x200;
/// `add rdx, 1; call READER; test rcx, rcx; jz L; add rax, 1; L: call READER; ret`.
const JOIN: u64 = CODE_BASE + 0x300;
/// `L: add rdx, 1; call READER; dec rcx; jnz L; ret`: the loop head starts without facts.
const LOOP: u64 = CODE_BASE + 0x400;
/// `add rdx, 1; call READER; jmp READER`.
const TAIL: u64 = CODE_BASE + 0x500;
/// `add rdx, 1; call READER; test rcx, rcx; jz L; ret; L: add rax, 1; ret` (shared epilog).
const SHARED_RET: u64 = CODE_BASE + 0x600;
/// `setc al; ret`: reads CF, writes no flag.
const SETC: u64 = CODE_BASE + 0x700;
/// `cmp rcx, rdx; call SETC; call SETC; ret`.
const FLAGS_TWICE: u64 = CODE_BASE + 0x800;
/// `movq rax, xmm1; ret`: reads `xmm1` only.
const XREAD: u64 = CODE_BASE + 0x900;
/// `paddq xmm1, xmm1; call XREAD; call XREAD; ret`.
const XTWICE: u64 = CODE_BASE + 0xA00;

fn program() -> Program {
    Program::new()
        .asm(READER, |a| {
            a.mov(rax, rdx)?;
            a.ret()
        })
        .asm(TWICE, |a| {
            a.add(rdx, 1)?;
            a.call(READER)?;
            a.call(READER)?;
            a.add(rax, rdx)?;
            a.ret()
        })
        .asm(CHANGED, |a| {
            a.add(rdx, 1)?;
            a.call(READER)?;
            a.add(rdx, 1)?;
            a.call(READER)?;
            a.ret()
        })
        .asm(JOIN, |a| {
            let mut l = a.create_label();
            a.add(rdx, 1)?;
            a.call(READER)?;
            a.test(rcx, rcx)?;
            a.jz(l)?;
            a.add(rax, 1)?;
            a.set_label(&mut l)?;
            a.call(READER)?;
            a.ret()
        })
        .asm(LOOP, |a| {
            let mut l = a.create_label();
            a.set_label(&mut l)?;
            a.add(rdx, 1)?;
            a.call(READER)?;
            a.dec(rcx)?;
            a.jnz(l)?;
            a.ret()
        })
        .asm(TAIL, |a| {
            a.add(rdx, 1)?;
            a.call(READER)?;
            a.jmp(READER)
        })
        .asm(SHARED_RET, |a| {
            let mut l = a.create_label();
            a.add(rdx, 1)?;
            a.call(READER)?;
            a.test(rcx, rcx)?;
            a.jz(l)?;
            a.ret()?;
            a.set_label(&mut l)?;
            a.add(rax, 1)?;
            a.ret()
        })
        .asm(SETC, |a| {
            a.setc(al)?;
            a.ret()
        })
        .asm(FLAGS_TWICE, |a| {
            a.cmp(rcx, rdx)?;
            a.call(SETC)?;
            a.call(SETC)?;
            a.ret()
        })
        .asm(XREAD, |a| {
            a.movq(rax, xmm1)?;
            a.ret()
        })
        .asm(XTWICE, |a| {
            a.paddq(xmm1, xmm1)?;
            a.call(XREAD)?;
            a.call(XREAD)?;
            a.ret()
        })
}

/// The text of the function at `va`.
fn function_ir(ir: &str, va: u64) -> &str {
    ir.split("define ")
        .find(|f| f.contains(&format!("@{}(ptr noalias", function_symbol(va))))
        .expect("defined")
}

/// Stores into `%state.<slot>` in the function at `va`.
fn stores(ir: &str, va: u64, slot: &str) -> usize {
    let needle = format!("ptr %state.{slot},");
    function_ir(ir, va)
        .lines()
        .filter(|l| l.trim_start().starts_with("store ") && l.contains(&needle))
        .count()
}

#[test]
fn unchanged_registers_are_stored_back_once() {
    let functions = program().discover();
    let ir = lift_module(functions.values()).expect("lifts").ir;
    let gpr2 = |va| stores(&ir, va, "gpr2");
    assert_eq!(gpr2(TWICE), 1, "second call and return skip rdx");
    assert_eq!(gpr2(CHANGED), 2, "a write between the calls keeps the store");
    assert_eq!(gpr2(JOIN), 1, "both edges into the join agree");
    assert_eq!(gpr2(LOOP), 1, "inside the loop; the exit after it skips");
    assert_eq!(gpr2(TAIL), 1, "the tail-call exit skips rdx");
    assert_eq!(gpr2(SHARED_RET), 1, "the shared epilog skips rdx");
    // CF is stored for the first SETC, not for the second; the return stores the other five.
    assert_eq!(stores(&ir, FLAGS_TWICE, "flag0"), 1);
    for flag in 1..6 {
        assert_eq!(stores(&ir, FLAGS_TWICE, &format!("flag{flag}")), 1);
    }
    assert_eq!(stores(&ir, XTWICE, "xmm1"), 1, "vector registers too");
    // Without summaries every call may write anything, so every call reloads `rdx`: the reload
    // (not the callee summary) is what keeps the second call and the return from storing it.
    let lifted = crate::lift::lift_function(&functions[&TWICE]).expect("lifts");
    assert_eq!(stores(&lifted.ir, TWICE, "gpr2"), 1);
    assert_eq!(
        function_ir(&lifted.ir, TWICE)
            .matches("load i64, ptr %state.gpr2,")
            .count(),
        3,
        "entry and after each call"
    );
}

#[test]
fn skipped_store_backs_keep_results_exact() {
    if !exec_or_skip("skipped_store_backs_keep_results_exact") {
        return;
    }
    let functions = program().discover();
    let built = build("synced_store_backs", functions.values());
    let inputs = [(0u64, 5u64), (1, u64::MAX), (3, 0x1234_5678), (7, 2)];
    let mut cases = Vec::new();
    for &(c, d) in &inputs {
        for va in [TWICE, CHANGED, JOIN, LOOP, TAIL, SHARED_RET, FLAGS_TWICE] {
            let c = if va == LOOP { (c & 3) + 1 } else { c };
            cases.push(Case::new(va).reg(RCX, c).reg(RDX, d));
        }
    }
    let outcomes = run(&built, &cases);
    for (i, &(c, d)) in inputs.iter().enumerate() {
        let o = &outcomes[i * 7..i * 7 + 7];
        for (k, out) in o.iter().enumerate() {
            assert!(out.error.is_empty(), "input {i} case {k}: {}", out.error);
        }
        let start = |va: u64, c: u64| Case::new(va).reg(RCX, c).reg(RDX, d).regs;
        let d1 = d.wrapping_add(1);
        let mut want = start(TWICE, c);
        want[RDX] = d1;
        want[RAX] = d1.wrapping_add(d1);
        assert_eq!(o[0].regs, want, "twice ({c:#x}, {d:#x})");
        let mut want = start(CHANGED, c);
        want[RDX] = d.wrapping_add(2);
        want[RAX] = d.wrapping_add(2);
        assert_eq!(o[1].regs, want, "changed ({c:#x}, {d:#x})");
        let mut want = start(JOIN, c);
        want[RDX] = d1;
        want[RAX] = d1;
        assert_eq!(o[2].regs, want, "join ({c:#x}, {d:#x})");
        assert_eq!(o[2].flags[3], c == 0, "join zf");
        let n = (c & 3) + 1;
        let mut want = start(LOOP, n);
        want[RDX] = d.wrapping_add(n);
        want[RAX] = d.wrapping_add(n);
        want[RCX] = 0;
        assert_eq!(o[3].regs, want, "loop ({n:#x}, {d:#x})");
        let mut want = start(TAIL, c);
        want[RDX] = d1;
        want[RAX] = d1;
        assert_eq!(o[4].regs, want, "tail ({c:#x}, {d:#x})");
        let mut want = start(SHARED_RET, c);
        want[RDX] = d1;
        want[RAX] = if c == 0 { d1.wrapping_add(1) } else { d1 };
        assert_eq!(o[5].regs, want, "shared ret ({c:#x}, {d:#x})");
        let mut want = start(FLAGS_TWICE, c);
        want[RAX] = (want[RAX] & !0xFF) | u64::from(c < d);
        assert_eq!(o[6].regs, want, "flags twice ({c:#x}, {d:#x})");
        assert_eq!(o[6].flags[0], c < d, "cf");
        assert_eq!(o[6].flags[3], c == d, "zf");
    }
}
