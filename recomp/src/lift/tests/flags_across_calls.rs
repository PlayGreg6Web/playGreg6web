//! Flag liveness across calls (`flag_liveness`): which flags a callee may observe on entry, the
//! caller storing only those before a direct call or tail jump, and the results staying exact.

use iced_x86::code_asm::*;

use super::exec::{CODE_BASE, Case, RAX, RCX, RDX, build, exec_or_skip, run};
use super::{Program, asm_function};
use crate::lift::{FlagSummaries, function_symbol, lift_module};

const CF: u8 = 1 << 0;
const PF: u8 = 1 << 1;
const AF: u8 = 1 << 2;
const ZF: u8 = 1 << 3;
const SF: u8 = 1 << 4;
const OF: u8 = 1 << 5;
const ALL: u8 = CF | PF | AF | ZF | SF | OF;

const CALLER: u64 = CODE_BASE;
/// Overwrites every flag but AF before returning (`shr`, `xor`).
const KILL: u64 = CODE_BASE + 0x100;
/// Reads CF (`adc`).
const ADC: u64 = CODE_BASE + 0x200;
/// Touches no flag: all of them reach the caller unchanged.
const PASS: u64 = CODE_BASE + 0x300;
/// Reads ZF (`setz`); also reached by a tail jump.
const SETZ: u64 = CODE_BASE + 0x400;
const ADC_CALLER: u64 = CODE_BASE + 0x500;
const PASS_CALLER: u64 = CODE_BASE + 0x600;
const TAIL_CALLER: u64 = CODE_BASE + 0x700;
const SETZ_CALLER: u64 = CODE_BASE + 0x800;
/// Calls KILL first, then uses CF: needs nothing from its caller but AF (KILL keeps AF).
const NESTED: u64 = CODE_BASE + 0x900;
/// `shl rax, cl`: a zero count keeps the flags, so they are read.
const SHIFT_CL: u64 = CODE_BASE + 0xA00;

fn program() -> Program {
    Program::new()
        // rax = rcx * 3, flags from the compare and imul, then the leaf
        .asm(CALLER, |a| {
            a.cmp(rcx, rdx)?;
            a.imul_3(rax, rcx, 3)?;
            a.call(KILL)?;
            a.ret()
        })
        .asm(KILL, |a| {
            a.push(rbx)?;
            a.mov(rbx, rax)?;
            a.shr(rbx, 3)?;
            a.xor(rax, rbx)?;
            a.pop(rbx)?;
            a.ret()
        })
        .asm(ADC, |a| {
            a.mov(eax, 0)?;
            a.adc(rax, 0)?;
            a.ret()
        })
        .asm(PASS, |a| {
            a.mov(rax, rcx)?;
            a.ret()
        })
        .asm(SETZ, |a| {
            a.mov(eax, 0)?;
            a.setz(al)?;
            a.ret()
        })
        .asm(ADC_CALLER, |a| {
            a.cmp(rcx, rdx)?;
            a.call(ADC)?;
            a.ret()
        })
        .asm(PASS_CALLER, |a| {
            a.cmp(rcx, rdx)?;
            a.call(PASS)?;
            a.ret()
        })
        .asm(TAIL_CALLER, |a| {
            a.cmp(rcx, rdx)?;
            a.jmp(SETZ)
        })
        .asm(SETZ_CALLER, |a| {
            a.call(SETZ)?;
            a.ret()
        })
        .asm(NESTED, |a| {
            a.call(KILL)?;
            a.adc(rax, 0)?;
            a.ret()
        })
        .asm(SHIFT_CL, |a| {
            a.shl(rax, cl)?;
            a.add(rax, 1)?;
            a.ret()
        })
}

#[test]
fn summaries_hold_the_flags_a_callee_may_observe() {
    let functions = program().discover();
    let summaries = FlagSummaries::compute(functions.values());
    let live = |va: u64| summaries.live_in(va).expect("analysed");
    assert_eq!(live(KILL), AF, "shr/xor overwrite all but AF");
    assert_eq!(live(ADC), CF, "adc reads CF and overwrites the rest");
    assert_eq!(live(PASS), ALL);
    assert_eq!(live(SETZ), ALL);
    assert_eq!(live(CALLER), 0, "cmp overwrites every flag first");
    assert_eq!(live(TAIL_CALLER), 0);
    assert_eq!(
        live(SETZ_CALLER),
        ALL,
        "a call passes the callee's live-in on"
    );
    assert_eq!(live(NESTED), AF, "only AF survives the call to KILL");
    assert_eq!(
        live(SHIFT_CL),
        CF | PF | ZF | SF | OF,
        "shl by cl may keep them"
    );
    assert_eq!(summaries.live_in(CODE_BASE + 0xF000), None);
}

#[test]
fn summaries_treat_unknown_code_as_reading_every_flag() {
    // A call to a function outside the analysed set, an indirect call and an instruction the
    // analysis does not model all count as reading every flag.
    let outside = asm_function(0x1000, |a| {
        a.call(0x2000)?;
        a.xor(eax, eax)?;
        a.ret()
    });
    let indirect = asm_function(0x3000, |a| {
        a.call(rax)?;
        a.xor(eax, eax)?;
        a.ret()
    });
    let unmodelled = asm_function(0x4000, |a| {
        a.bswap(rax)?;
        a.xor(eax, eax)?;
        a.ret()
    });
    let summaries = FlagSummaries::compute([&outside, &indirect, &unmodelled]);
    assert_eq!(summaries.live_in(0x1000), Some(ALL));
    assert_eq!(summaries.live_in(0x3000), Some(ALL));
    assert_eq!(summaries.live_in(0x4000), Some(ALL));
}

/// The caller's LLVM block from its first guest block up to the call or tail call of `callee`.
fn before_transfer(ir: &str, caller: u64, callee: u64) -> String {
    let func = ir
        .split("define ")
        .find(|f| f.contains(&format!("@{}(ptr noalias", function_symbol(caller))))
        .expect("caller defined");
    let call = format!("call void @{}(ptr %state)", function_symbol(callee));
    let end = func.find(&call).expect("calls the callee");
    let start = func.find(&format!("bb_{caller:X}:")).expect("first block");
    func[start..end].to_string()
}

fn stored_flags(text: &str) -> Vec<usize> {
    (0..6)
        .filter(|i| text.contains(&format!("i32 2, i64 {i}\n")))
        .collect()
}

#[test]
fn calls_store_only_the_flags_the_callee_may_observe() {
    let functions = program().discover();
    let ir = lift_module(functions.values()).expect("lifts").ir;
    // KILL observes AF only: CF/OF of the imul (and the rest) are not stored before the call.
    assert_eq!(stored_flags(&before_transfer(&ir, CALLER, KILL)), vec![2]);
    // ADC observes CF, PASS every flag.
    assert_eq!(
        stored_flags(&before_transfer(&ir, ADC_CALLER, ADC)),
        vec![0]
    );
    assert_eq!(
        stored_flags(&before_transfer(&ir, PASS_CALLER, PASS)),
        vec![0, 1, 2, 3, 4, 5]
    );
    // The tail jump into SETZ stores the flags SETZ observes (all of them).
    let tail = ir
        .split("define ")
        .find(|f| f.contains(&format!("@{}(ptr noalias", function_symbol(TAIL_CALLER))))
        .expect("tail caller");
    let exit = tail.find(&format!("ext_{SETZ:X}:")).expect("tail exit");
    assert_eq!(stored_flags(&tail[exit..]), vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn calls_without_summaries_store_every_flag() {
    let functions = program().discover();
    let lifted = crate::lift::lift_function(&functions[&CALLER]).expect("lifts");
    assert_eq!(
        stored_flags(&before_transfer(&lifted.ir, CALLER, KILL)),
        vec![0, 1, 2, 3, 4, 5]
    );
}

#[test]
fn flags_stay_exact_across_calls() {
    if !exec_or_skip("flags_stay_exact_across_calls") {
        return;
    }
    let functions = program().discover();
    let built = build("flags_across_calls", functions.values());
    let pairs = [
        (1u64, 2u64),
        (2, 1),
        (7, 7),
        (0, u64::MAX),
        (0x8000_0000_0000_0000, 1),
    ];
    let mut cases = Vec::new();
    for &(c, d) in &pairs {
        for va in [CALLER, ADC_CALLER, PASS_CALLER, TAIL_CALLER, NESTED] {
            cases.push(Case::new(va).reg(RCX, c).reg(RDX, d).reg(RAX, c ^ 0x5A));
        }
    }
    let outcomes = run(&built, &cases);
    for (i, &(c, d)) in pairs.iter().enumerate() {
        let o = &outcomes[i * 5..i * 5 + 5];
        for (k, out) in o.iter().enumerate() {
            assert!(out.error.is_empty(), "pair {i} case {k}: {}", out.error);
        }
        let diff = c.wrapping_sub(d);
        let cmp_cf = c < d;
        let cmp_zf = c == d;
        let cmp_af = (c ^ d ^ diff) & 0x10 != 0;
        let cmp_sf = diff >> 63 != 0;
        // CALLER: KILL's result and flags, AF from the cmp (imul leaves AF unchanged).
        let x = c.wrapping_mul(3);
        let r = x ^ (x >> 3);
        let parity = (r as u8).count_ones().is_multiple_of(2);
        assert_eq!(o[0].regs[RAX], r, "caller rax ({c:#x}, {d:#x})");
        assert_eq!(
            o[0].flags,
            [false, parity, cmp_af, r == 0, r >> 63 != 0, false],
            "caller flags ({c:#x}, {d:#x})"
        );
        // ADC_CALLER: the callee adds the caller's CF.
        assert_eq!(o[1].regs[RAX], u64::from(cmp_cf), "adc ({c:#x}, {d:#x})");
        // PASS_CALLER: the compare's flags come back unchanged.
        assert_eq!(o[2].regs[RAX], c);
        assert_eq!(
            (o[2].flags[0], o[2].flags[2], o[2].flags[3], o[2].flags[4]),
            (cmp_cf, cmp_af, cmp_zf, cmp_sf),
            "pass ({c:#x}, {d:#x})"
        );
        // TAIL_CALLER: setz after the tail jump sees the compare's ZF.
        assert_eq!(o[3].regs[RAX], u64::from(cmp_zf), "tail ({c:#x}, {d:#x})");
        // NESTED: KILL clears CF, so the adc adds nothing.
        let x = c ^ 0x5A;
        assert_eq!(o[4].regs[RAX], x ^ (x >> 3), "nested ({c:#x}, {d:#x})");
    }
}
