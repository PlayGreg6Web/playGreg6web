//! The lifted-code / runtime boundary contract on `%state` (docs/LIFTER_STATE_CONTRACT.md).
//!
//! A caller may keep a register in its alloca across a direct call (and an optimisation such as
//! #2033 may treat "alloca still equals the slot" as surviving that call) only for the slots
//! outside the callee's `written` summary. That is sound only if nothing the callee runs can write
//! `%state` behind the summary's back. Inside lifted code, the only calls that hand `%state` to
//! the runtime are the ones listed in [`RUNTIME_STATE_CALLS`]. These tests pin:
//! - every such call that can return (and so lets the runtime write the state: imports, nested
//!   guest callbacks, exception and context-restore thunks, `cpuid`/`rdtsc`/`xgetbv`) only appears
//!   in functions whose summary already says "writes everything" ([`ALL`]);
//! - every direct call or tail jump to another lifted function only appears in a function whose
//!   `written` summary covers the callee's;
//! - no new runtime helper receives `%state` without being classified here and in the doc.

use iced_x86::code_asm::*;

use super::Program;
use super::exec::CODE_BASE;
use crate::lift::{
    FlagSummaries, HELPER_CPUID, HELPER_DIVIDE_ERROR, HELPER_INDIRECT_CALL, HELPER_PROBE_EXIT, HELPER_PROBE_HIT,
    HELPER_RDTSC, HELPER_RDTSCP, HELPER_TRAP, HELPER_UNIMPLEMENTED, HELPER_XGETBV, function_symbol,
    lift_module,
};

const ALL: u64 = (1u64 << 49) - 1;

/// How a runtime helper that receives `%state` may affect it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Effect {
    /// Returns and may write any field (imports, callbacks, context restores, unwinds).
    WritesAny,
    /// Never returns (the run ends); reads the state for the fault report only.
    NoReturn,
    /// Returns, reads the state, never writes it (diagnostics).
    ReadOnly,
}

/// Every runtime helper lifted code passes `%state` to, with its effect.
const RUNTIME_STATE_CALLS: &[(&str, Effect)] = &[
    (HELPER_INDIRECT_CALL, Effect::WritesAny),
    (HELPER_CPUID, Effect::WritesAny),
    (HELPER_RDTSC, Effect::WritesAny),
    (HELPER_RDTSCP, Effect::WritesAny),
    (HELPER_XGETBV, Effect::WritesAny),
    (HELPER_TRAP, Effect::NoReturn),
    (HELPER_UNIMPLEMENTED, Effect::NoReturn),
    (HELPER_DIVIDE_ERROR, Effect::NoReturn),
    (HELPER_PROBE_HIT, Effect::ReadOnly),
    (HELPER_PROBE_EXIT, Effect::ReadOnly),
];

const PURE: u64 = CODE_BASE;
const PURE_CALLER: u64 = CODE_BASE + 0x100;
const PURE_TAIL: u64 = CODE_BASE + 0x200;
const VECTOR: u64 = CODE_BASE + 0x300;
const IMPORT_CALL: u64 = CODE_BASE + 0x400;
const IAT_CALL: u64 = CODE_BASE + 0x500;
const CALLS_IMPORT: u64 = CODE_BASE + 0x600;
const DISPATCH: u64 = CODE_BASE + 0x700;
const CPUID: u64 = CODE_BASE + 0x800;
const RDTSC: u64 = CODE_BASE + 0x900;
const LDMXCSR: u64 = CODE_BASE + 0xA00;
const DIVIDE: u64 = CODE_BASE + 0xB00;
const BREAKPOINT: u64 = CODE_BASE + 0xC00;
const NESTED: u64 = CODE_BASE + 0xD00;

/// One function per kind of transfer lifted code emits.
fn program() -> Program {
    Program::new()
        .asm(PURE, |a| {
            a.push(rbx)?;
            a.lea(rax, qword_ptr(rcx + rdx))?;
            a.mov(rbx, rax)?;
            a.add(rax, rbx)?;
            a.pop(rbx)?;
            a.ret()
        })
        .asm(PURE_CALLER, |a| {
            a.add(rsi, 1)?;
            a.call(PURE)?;
            a.add(rax, rsi)?;
            a.ret()
        })
        .asm(PURE_TAIL, |a| {
            a.add(rdx, 2)?;
            a.jmp(PURE)
        })
        .asm(VECTOR, |a| {
            a.movq(xmm1, rax)?;
            a.addpd(xmm0, xmm1)?;
            a.ret()
        })
        .asm(IMPORT_CALL, |a| {
            a.add(rsi, 1)?;
            a.call(rax)?;
            a.add(rax, rsi)?;
            a.ret()
        })
        .asm(IAT_CALL, |a| {
            a.call(qword_ptr(rbx + 8))?;
            a.ret()
        })
        .asm(CALLS_IMPORT, |a| {
            a.add(rsi, 1)?;
            a.call(IMPORT_CALL)?;
            a.add(rax, rsi)?;
            a.ret()
        })
        .asm(DISPATCH, |a| a.jmp(rax))
        .asm(CPUID, |a| {
            a.cpuid()?;
            a.ret()
        })
        .asm(RDTSC, |a| {
            a.rdtsc()?;
            a.ret()
        })
        .asm(LDMXCSR, |a| {
            a.ldmxcsr(dword_ptr(rsp + 8))?;
            a.ret()
        })
        .asm(DIVIDE, |a| {
            a.xor(edx, edx)?;
            a.div(rcx)?;
            a.ret()
        })
        .asm(BREAKPOINT, |a| a.int3())
        .asm(NESTED, |a| {
            a.call(PURE_CALLER)?;
            a.call(PURE_TAIL)?;
            a.ret()
        })
}

/// The text of the lifted function at `va`.
fn function_ir(ir: &str, va: u64) -> &str {
    ir.split("define ")
        .find(|f| f.contains(&format!("@{}(ptr noalias", function_symbol(va))))
        .expect("defined")
}

/// `(callee symbol, line index)` of every call in `func` that passes `%state`.
fn state_calls(lines: &[&str]) -> Vec<(String, usize)> {
    lines
        .iter()
        .enumerate()
        .filter_map(|(i, line)| {
            let line = line.trim();
            let call = line
                .strip_prefix("tail call void @")
                .or_else(|| line.strip_prefix("call void @"))?;
            let (callee, args) = call.split_once('(')?;
            args.starts_with("ptr %state")
                .then(|| (callee.to_string(), i))
        })
        .collect()
}

#[test]
fn only_functions_summarised_as_writing_everything_let_the_runtime_write_state() {
    let functions = program().discover();
    let summaries = FlagSummaries::compute(functions.values());
    let mut module = lift_module(functions.values()).expect("lifts");
    // The probe is inserted after lifting, outside the summaries: it must stay read-only.
    for f in &mut module.functions {
        if f.entry == PURE {
            f.insert_entry_probe();
            f.insert_exit_probe();
        }
    }
    let ir: String = module
        .functions
        .iter()
        .map(|f| f.ir.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let sub_va = |symbol: &str| {
        functions
            .keys()
            .copied()
            .find(|&va| function_symbol(va) == symbol)
    };
    let mut seen = Vec::new();
    for &va in functions.keys() {
        let written = summaries.written(va).expect("analysed");
        let lines: Vec<&str> = function_ir(&ir, va).lines().collect();
        for (callee, at) in state_calls(&lines) {
            if let Some(target) = sub_va(&callee) {
                let callee_written = summaries.written(target).expect("analysed");
                assert_eq!(
                    callee_written & !written,
                    0,
                    "{va:#x}: the summary must cover what its callee {target:#x} writes"
                );
                continue;
            }
            let effect = RUNTIME_STATE_CALLS
                .iter()
                .find(|(name, _)| *name == callee)
                .unwrap_or_else(|| {
                    panic!(
                        "{va:#x} passes %state to @{callee}, which is not classified in \
                         RUNTIME_STATE_CALLS / docs/LIFTER_STATE_CONTRACT.md"
                    )
                })
                .1;
            seen.push(callee.clone());
            match effect {
                Effect::WritesAny => assert_eq!(
                    written, ALL,
                    "{va:#x} calls @{callee}, which may write any state field, but its summary \
                     claims it writes only {written:#x}: callers would keep stale registers"
                ),
                Effect::NoReturn => assert_eq!(
                    lines.get(at + 1).map(|l| l.trim()),
                    Some("unreachable"),
                    "{va:#x}: @{callee} must not return into lifted code"
                ),
                Effect::ReadOnly => {}
            }
        }
    }
    // The program reaches every class of boundary call.
    for name in [
        HELPER_INDIRECT_CALL,
        HELPER_CPUID,
        HELPER_RDTSC,
        HELPER_PROBE_HIT,
        HELPER_PROBE_EXIT,
    ] {
        assert!(seen.iter().any(|s| s == name), "no call of @{name}");
    }
    assert!(
        seen.iter()
            .any(|s| s == HELPER_TRAP || s == HELPER_UNIMPLEMENTED || s == HELPER_DIVIDE_ERROR),
        "no trap"
    );
}

#[test]
fn precise_summaries_exist_only_for_code_the_runtime_never_enters() {
    let functions = program().discover();
    let summaries = FlagSummaries::compute(functions.values());
    let written = |va: u64| summaries.written(va).expect("analysed");
    // The functions a caller may keep registers across: pure guest code.
    for va in [PURE, PURE_CALLER, PURE_TAIL, VECTOR, NESTED] {
        assert_ne!(written(va), ALL, "{va:#x} should have a precise summary");
    }
    // Anything that reaches the runtime with %state, directly or through a callee.
    for va in [
        IMPORT_CALL,
        IAT_CALL,
        CALLS_IMPORT,
        DISPATCH,
        CPUID,
        RDTSC,
        LDMXCSR,
        DIVIDE,
        BREAKPOINT,
    ] {
        assert_eq!(written(va), ALL, "{va:#x} must count as writing everything");
    }
}

/// The entry probe logs `rcx rdx r8 r9`. A callee that does not read `r8`/`r9` would otherwise
/// let its callers skip storing them before a direct call, and the probe would log a stale copy:
/// a probed build counts the probed function as reading all four at entry, transitively.
#[test]
fn probed_functions_get_their_arguments_synced_before_direct_calls() {
    let r8_r9 = (1u64 << 8) | (1u64 << 9);
    let functions = program().discover();
    let plain = FlagSummaries::compute(functions.values());
    // PURE reads only rcx and rdx, so without a probe nobody syncs r8/r9 for it.
    let synced = |s: &FlagSummaries, va: u64| {
        s.touched(va).expect("analysed") & s.live_in_state(va).expect("analysed")
    };
    assert_eq!(synced(&plain, PURE) & r8_r9, 0);
    let probes = [PURE].into_iter().collect();
    let probed = FlagSummaries::compute_with_probes(functions.values(), &probes);
    let args = 0b11_0000_0110u64;
    assert_eq!(synced(&probed, PURE) & args, args);
    // Callers that reach PURE without setting r8/r9 pass the requirement on to their callers.
    assert_eq!(synced(&probed, PURE_CALLER) & r8_r9, r8_r9);
    assert_eq!(synced(&probed, NESTED) & r8_r9, r8_r9);
    // Written summaries are unchanged: the probe only reads.
    for &va in functions.keys() {
        assert_eq!(probed.written(va), plain.written(va), "{va:#x}");
    }
    // A caller that sets r8 itself must store it to the slot before the direct call, or the
    // probe reads the value from before the caller ran.
    const PROBED: u64 = CODE_BASE;
    const SETS_R8: u64 = CODE_BASE + 0x100;
    let functions = Program::new()
        .asm(PROBED, |a| {
            a.lea(rax, qword_ptr(rcx + rdx))?;
            a.ret()
        })
        .asm(SETS_R8, |a| {
            a.mov(r8, 5u64)?;
            a.call(PROBED)?;
            a.add(rax, r8)?;
            a.ret()
        })
        .discover();
    let plain = FlagSummaries::compute(functions.values());
    let probed = FlagSummaries::compute_with_probes(functions.values(), &[PROBED].into());
    let call = format!("@{}(", function_symbol(PROBED));
    let stores_r8_before_call = |summaries: &FlagSummaries| {
        let ir = crate::lift::lift_function_in(
            &functions[&SETS_R8],
            crate::lift::GuestBase::default(),
            summaries,
        )
        .expect("lifts")
        .ir;
        ir[..ir.find(&call).expect("direct call")].contains("ptr %state.gpr8")
    };
    assert!(!stores_r8_before_call(&plain));
    assert!(stores_r8_before_call(&probed));
}
