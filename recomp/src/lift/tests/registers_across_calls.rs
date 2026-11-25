//! Guest state a callee may touch (`flag_liveness`, "Registers a callee never touches"): the
//! caller keeping the registers a direct callee never reads or writes out of `%state` across the
//! call, and the results staying exact.

use iced_x86::code_asm::*;

use super::exec::{
    CODE_BASE, Case, R8, R9, RAX, RBX, RCX, RDX, RSI, RSP, build, exec_or_skip, run,
};
use super::{Program, asm_function};
use crate::lift::{FlagSummaries, function_symbol, lift_module};

const OTHER: u32 = 1 << 16;
const ALL: u32 = (OTHER << 1) - 1;

const fn bit(reg: usize) -> u32 {
    1 << reg
}

/// Keeps `rcx`, `rdx`, `rsi` and `r9` live across a call to LEAF.
const CALLER: u64 = CODE_BASE;
/// `push rbx; mov rbx, rax; shr rbx, 3; xor rax, rbx; pop rbx; ret`.
const LEAF: u64 = CODE_BASE + 0x100;
/// Adds `r8`, then calls LEAF: touches LEAF's registers and `r8`.
const MID: u64 = CODE_BASE + 0x200;
/// Calls MID with `r8`, `r9` live.
const MID_CALLER: u64 = CODE_BASE + 0x300;
/// Counts `rcx` down recursively.
const REC: u64 = CODE_BASE + 0x400;
/// Tail-jumps to LEAF after touching `rdx`.
const TAIL: u64 = CODE_BASE + 0x500;
/// Uses a vector register.
const VECTOR: u64 = CODE_BASE + 0x600;
/// Calls VECTOR with `rsi` live: everything is stored and reloaded.
const VECTOR_CALLER: u64 = CODE_BASE + 0x700;
/// Calls TAIL with `rsi` live.
const TAIL_CALLER: u64 = CODE_BASE + 0x800;

fn program() -> Program {
    Program::new()
        .asm(CALLER, |a| {
            a.mov(r9, rcx)?;
            a.add(rsi, rdx)?;
            a.call(LEAF)?;
            a.add(rax, r9)?;
            a.add(rax, rsi)?;
            a.add(rcx, rdx)?;
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
        .asm(MID, |a| {
            a.add(rax, r8)?;
            a.call(LEAF)?;
            a.ret()
        })
        .asm(MID_CALLER, |a| {
            a.mov(r9, r8)?;
            a.call(MID)?;
            a.sub(rax, r9)?;
            a.ret()
        })
        .asm(REC, |a| {
            let mut done = a.create_label();
            a.dec(rcx)?;
            a.jz(done)?;
            a.call(REC)?;
            a.set_label(&mut done)?;
            a.ret()
        })
        .asm(TAIL, |a| {
            a.add(rax, rdx)?;
            a.jmp(LEAF)
        })
        .asm(VECTOR, |a| {
            a.movq(xmm0, rax)?;
            a.movq(rax, xmm0)?;
            a.ret()
        })
        .asm(VECTOR_CALLER, |a| {
            a.add(rsi, 1)?;
            a.call(VECTOR)?;
            a.add(rax, rsi)?;
            a.ret()
        })
        .asm(TAIL_CALLER, |a| {
            a.add(rsi, 1)?;
            a.call(TAIL)?;
            a.add(rax, rsi)?;
            a.ret()
        })
}

#[test]
fn summaries_hold_the_state_a_call_may_touch() {
    let functions = program().discover();
    let summaries = FlagSummaries::compute(functions.values());
    let touched = |va: u64| summaries.touched(va).expect("analysed");
    let leaf = bit(RAX) | bit(RBX) | bit(RSP);
    assert_eq!(touched(LEAF), leaf);
    assert_eq!(
        touched(MID),
        leaf | bit(R8),
        "a call adds the callee's state"
    );
    assert_eq!(
        touched(REC),
        bit(RCX) | bit(RSP),
        "recursion is a fixed point"
    );
    assert_eq!(
        touched(TAIL),
        leaf | bit(RDX),
        "a tail jump adds the target's state"
    );
    assert_eq!(touched(VECTOR), ALL, "vector code is not modelled");
    assert_eq!(touched(VECTOR_CALLER), ALL);
    assert_eq!(summaries.touched(CODE_BASE + 0xF000), None);
}

#[test]
fn summaries_treat_unknown_code_as_touching_everything() {
    let outside = asm_function(0x1000, |a| {
        a.call(0x2000)?;
        a.ret()
    });
    let indirect = asm_function(0x3000, |a| {
        a.call(rax)?;
        a.ret()
    });
    let unmodelled = asm_function(0x4000, |a| {
        a.bswap(rax)?;
        a.ret()
    });
    let trap = asm_function(0x5000, |a| a.int3());
    let dispatch = asm_function(0x6000, |a| a.jmp(rax));
    let looping = asm_function(0x7000, |a| {
        let mut top = a.create_label();
        a.set_label(&mut top)?;
        a.loop_(top)?;
        a.ret()
    });
    let summaries =
        FlagSummaries::compute([&outside, &indirect, &unmodelled, &trap, &dispatch, &looping]);
    for va in [0x1000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000] {
        assert_eq!(summaries.touched(va), Some(ALL), "{va:#x}");
    }
}

/// The caller's function text.
fn function_ir(ir: &str, va: u64) -> &str {
    ir.split("define ")
        .find(|f| f.contains(&format!("@{}(ptr noalias", function_symbol(va))))
        .expect("defined")
}

/// GPR indices whose state slot `text` addresses.
fn gprs(text: &str) -> Vec<usize> {
    (0..16)
        .filter(|i| text.contains(&format!("i32 0, i64 {i}\n")))
        .collect()
}

/// `(before, after)`: the caller's text from its first block up to the call of `callee`, and
/// from the call to the first instruction after the call's pending test.
fn around_call(ir: &str, caller: u64, callee: u64) -> (String, String) {
    let func = function_ir(ir, caller);
    let call = format!("call void @{}(ptr %state)", function_symbol(callee));
    let at = func.find(&call).expect("calls the callee");
    let start = func.find(&format!("bb_{caller:X}:")).expect("first block");
    let after = &func[at..];
    let resumed = after.find("after_call").expect("pending test");
    let body = &after[resumed..];
    let body = &body[body.find(":\n").expect("label") + 2..];
    // Up to the first lifted instruction after the reloads (each caller continues with add/sub).
    let end = ["= add i64", "= sub i64"]
        .iter()
        .filter_map(|op| body.find(op))
        .min()
        .unwrap_or(body.len());
    (func[start..at].to_string(), body[..end].to_string())
}

#[test]
fn calls_keep_untouched_registers_out_of_the_state() {
    let functions = program().discover();
    let ir = lift_module(functions.values()).expect("lifts").ir;
    // LEAF touches rax, rbx, rsp: rcx, rdx, rsi, r9 stay in the caller.
    let (before, after) = around_call(&ir, CALLER, LEAF);
    assert_eq!(gprs(&before), vec![RAX, RSP], "stores before the call");
    assert_eq!(gprs(&after), vec![RAX, RSP], "reloads after the call");
    // MID also touches r8.
    let (before, after) = around_call(&ir, MID_CALLER, MID);
    assert_eq!(gprs(&before), vec![RAX, RSP, R8]);
    assert_eq!(gprs(&after), vec![RAX, RSP, R8]);
    // TAIL also touches rdx; rsi stays.
    let (before, _) = around_call(&ir, TAIL_CALLER, TAIL);
    assert_eq!(gprs(&before), vec![RAX, RSP]);
    // VECTOR may touch anything: every tracked register goes through the state.
    let (before, after) = around_call(&ir, VECTOR_CALLER, VECTOR);
    assert_eq!(gprs(&before), vec![RAX, RSP, RSI]);
    assert_eq!(gprs(&after), vec![RAX, RSP, RSI]);
    // The caller's own return still writes every tracked register back.
    let caller = function_ir(&ir, CALLER);
    let ret = caller.rfind("ret void").expect("returns");
    let tail = &caller[..ret];
    let last_call = tail.rfind("after_call").expect("after the call");
    assert_eq!(gprs(&tail[last_call..]), vec![RAX, RCX, RDX, RSP, RSI, R9]);
}

#[test]
fn calls_without_summaries_sync_every_register() {
    let functions = program().discover();
    let lifted = crate::lift::lift_function(&functions[&CALLER]).expect("lifts");
    let (before, after) = around_call(&lifted.ir, CALLER, LEAF);
    assert_eq!(gprs(&before), vec![RAX, RCX, RDX, RSP, RSI, R9]);
    assert_eq!(gprs(&after), vec![RAX, RCX, RDX, RSP, RSI, R9]);
}

#[test]
fn registers_stay_exact_across_calls() {
    if !exec_or_skip("registers_stay_exact_across_calls") {
        return;
    }
    let functions = program().discover();
    let built = build("registers_across_calls", functions.values());
    let inputs = [
        (1u64, 2u64, 3u64),
        (0x1234_5678_9ABC, u64::MAX, 7),
        (5, 0, 1 << 63),
    ];
    let mut cases = Vec::new();
    for &(c, d, s) in &inputs {
        for va in [CALLER, MID_CALLER, TAIL_CALLER, VECTOR_CALLER] {
            cases.push(
                Case::new(va)
                    .reg(RCX, c)
                    .reg(RDX, d)
                    .reg(RSI, s)
                    .reg(R8, d ^ 0x55)
                    .reg(RAX, c ^ 0x5A),
            );
        }
        cases.push(Case::new(REC).reg(RCX, (c & 0xF) + 1));
    }
    let outcomes = run(&built, &cases);
    let leaf = |x: u64| x ^ (x >> 3);
    for (i, &(c, d, s)) in inputs.iter().enumerate() {
        let o = &outcomes[i * 5..i * 5 + 5];
        for (k, out) in o.iter().enumerate() {
            assert!(out.error.is_empty(), "input {i} case {k}: {}", out.error);
        }
        let start = |va: u64| {
            Case::new(va)
                .reg(RCX, c)
                .reg(RDX, d)
                .reg(RSI, s)
                .reg(R8, d ^ 0x55)
                .reg(RAX, c ^ 0x5A)
                .regs
        };
        // CALLER
        let mut want = start(CALLER);
        want[R9] = c;
        want[RSI] = s.wrapping_add(d);
        want[RAX] = leaf(c ^ 0x5A)
            .wrapping_add(c)
            .wrapping_add(s.wrapping_add(d));
        want[RCX] = c.wrapping_add(d);
        assert_eq!(o[0].regs, want, "caller ({c:#x}, {d:#x}, {s:#x})");
        // MID_CALLER
        let mut want = start(MID_CALLER);
        want[R9] = d ^ 0x55;
        want[RAX] = leaf((c ^ 0x5A).wrapping_add(d ^ 0x55)).wrapping_sub(d ^ 0x55);
        assert_eq!(o[1].regs, want, "mid ({c:#x}, {d:#x}, {s:#x})");
        // TAIL_CALLER
        let mut want = start(TAIL_CALLER);
        want[RSI] = s.wrapping_add(1);
        want[RAX] = leaf((c ^ 0x5A).wrapping_add(d)).wrapping_add(s.wrapping_add(1));
        assert_eq!(o[2].regs, want, "tail ({c:#x}, {d:#x}, {s:#x})");
        // VECTOR_CALLER
        let mut want = start(VECTOR_CALLER);
        want[RSI] = s.wrapping_add(1);
        want[RAX] = (c ^ 0x5A).wrapping_add(s.wrapping_add(1));
        assert_eq!(o[3].regs, want, "vector ({c:#x}, {d:#x}, {s:#x})");
        // REC counts rcx down to zero; rsp is back where it started.
        assert_eq!(o[4].regs[RCX], 0);
        assert_eq!(o[4].regs[RSP], Case::new(REC).regs[RSP]);
    }
}
