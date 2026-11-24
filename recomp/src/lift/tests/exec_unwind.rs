//! Resume protocol (`lift/unwind.rs`): IR shape, and execution under node with a host function
//! standing in for `RtlRestoreContext` (`0x7FFE0020` in `runner.js`: restores the GPRs listed in a
//! descriptor `{rip, mask, gpr[16]}` and leaves the restore pending).

use iced_x86::code_asm::*;
use iced_x86::{BlockEncoderOptions, IcedError};

use super::Program;
use super::exec::{
    CODE_BASE, Case, DATA_ADDR, Outcome, RAX, RBX, RSI, RSP, STACK_TOP, STATE_ADDR, build,
    exec_or_skip, run,
};
use crate::lift::{STATE_UNWIND_PENDING_OFFSET, lift_function};

const R12: usize = 12;
const HOST_RESTORE: u64 = 0x7FFE_0020;
/// Descriptor read by the host restore: `[rip][mask][gpr0..gpr15]`.
const DESC: u64 = DATA_ADDR;
const DESC_GPR: i32 = 16;
/// Where the test functions record what they saw at a continuation.
const SEEN: u64 = DATA_ADDR + 0x100;
const JMP_BUF: u64 = DATA_ADDR + 0x200;

const F: u64 = CODE_BASE;
const G: u64 = CODE_BASE + 0x100;
const H: u64 = CODE_BASE + 0x200;
const REC: u64 = CODE_BASE + 0x400;
const SJ: u64 = CODE_BASE + 0x600;
const SETJMP: u64 = CODE_BASE + 0x700;
const LONGJMP: u64 = CODE_BASE + 0x780;

const fn bit(reg: usize) -> i32 {
    1 << reg
}

/// Assembles at `va`; `build` returns the labels whose addresses the test needs.
fn assemble_labeled(
    va: u64,
    build: impl FnOnce(&mut CodeAssembler) -> Result<Vec<CodeLabel>, IcedError>,
) -> (Vec<u8>, Vec<u64>) {
    let mut a = CodeAssembler::new(64).expect("64-bit assembler");
    let labels = build(&mut a).expect("test code assembles");
    let result = a
        .assemble_options(va, BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS)
        .expect("test code encodes");
    let ips = labels
        .iter()
        .map(|l| result.label_ip(l).expect("label ip"))
        .collect();
    (result.inner.code_buffer, ips)
}

/// `call [host restore]` with rcx = descriptor.
fn restore(a: &mut CodeAssembler) -> Result<(), IcedError> {
    a.mov(rcx, DESC)?;
    a.mov(rax, HOST_RESTORE)?;
    a.call(rax)
}

/// G: push rsi; sub rsp, 20h; clobber rsi and rbx; call H; (unreachable on restore)
fn intermediate() -> Vec<u8> {
    super::assemble(G, |a| {
        a.push(rsi)?;
        a.sub(rsp, 0x20)?;
        a.mov(rsi, 0xBAD_u64)?;
        a.mov(rbx, 0xBAD_u64)?;
        a.call(H)?;
        a.mov(rax, 0xDEAD_u64)?;
        a.add(rsp, 0x20)?;
        a.pop(rsi)?;
        a.ret()
    })
}

/// H: sub rsp, 28h; restore(DESC); (unreachable on restore)
fn thrower() -> Vec<u8> {
    super::assemble(H, |a| {
        a.sub(rsp, 0x28)?;
        restore(a)?;
        a.mov(rax, 0xBEEF_u64)?;
        a.add(rsp, 0x28)?;
        a.ret()
    })
}

/// F: try { G() } with continuation `cont`: the descriptor resumes at `resume_at(cont, done)`
/// with rsp = F's rsp + `rsp_delta`, rax = 40h, rbx = 1111h, r12 = 2222h, rsi = F's rsi.
/// Normal path: rax += 1; continuation: [SEEN] = rbx, rax += 100h. Returns (code, [cont, done]).
fn catcher(resume_at_done: bool, rsp_delta: i32) -> (Vec<u8>, Vec<u64>) {
    assemble_labeled(F, |a| {
        let mut cont = a.create_label();
        let mut done = a.create_label();
        a.push(rbx)?;
        a.push(r12)?;
        a.sub(rsp, 0x28)?;
        a.mov(rbx, 0x1111_u64)?;
        a.mov(r12, 0x2222_u64)?;
        a.mov(rax, DESC)?;
        if resume_at_done {
            a.lea(rcx, ptr(done))?;
        } else {
            a.lea(rcx, ptr(cont))?;
        }
        a.mov(qword_ptr(rax), rcx)?;
        let mask = bit(RAX) | bit(RBX) | bit(RSP) | bit(RSI) | bit(R12);
        a.mov(qword_ptr(rax + 8), mask)?;
        a.mov(qword_ptr(rax + DESC_GPR), 0x40)?;
        a.mov(qword_ptr(rax + DESC_GPR + 3 * 8), rbx)?;
        a.lea(rcx, qword_ptr(rsp + rsp_delta))?;
        a.mov(qword_ptr(rax + DESC_GPR + 4 * 8), rcx)?;
        a.mov(qword_ptr(rax + DESC_GPR + 6 * 8), rsi)?;
        a.mov(qword_ptr(rax + DESC_GPR + 12 * 8), r12)?;
        a.mov(rax, 0x10_u64)?;
        a.call(G)?;
        a.add(rax, 1)?;
        a.jmp(done)?;
        a.set_label(&mut cont)?;
        a.mov(rcx, SEEN)?;
        a.mov(qword_ptr(rcx), rbx)?;
        a.add(rax, 0x100)?;
        a.set_label(&mut done)?;
        a.add(rsp, 0x28)?;
        a.pop(r12)?;
        a.pop(rbx)?;
        a.ret()?;
        Ok(vec![cont, done])
    })
}

fn catcher_program(resume_at_done: bool, rsp_delta: i32) -> (Program, Vec<u64>) {
    let (code, labels) = catcher(resume_at_done, rsp_delta);
    let program = Program::new()
        .bytes(F, &code)
        .bytes(G, &intermediate())
        .bytes(H, &thrower())
        .resume(F, labels[0]);
    (program, labels)
}

/// REC(n = rcx, k = rdx): recursion; the activation with n == k records itself as the restore
/// target, the base case (n == 0) restores. Normal return: rax += 10h; continuation:
/// rax = n + 1000h.
fn recursion() -> (Vec<u8>, Vec<u64>) {
    assemble_labeled(REC, |a| {
        let mut skip = a.create_label();
        let mut base = a.create_label();
        let mut cont = a.create_label();
        let mut done = a.create_label();
        a.push(rbx)?;
        a.push(rsi)?;
        a.sub(rsp, 0x28)?;
        a.mov(rbx, rcx)?;
        a.mov(rsi, rdx)?;
        a.cmp(rbx, rsi)?;
        a.jne(skip)?;
        a.mov(rax, DESC)?;
        a.lea(rcx, ptr(cont))?;
        a.mov(qword_ptr(rax), rcx)?;
        a.mov(
            qword_ptr(rax + 8),
            bit(RAX) | bit(RBX) | bit(RSP) | bit(RSI),
        )?;
        a.mov(qword_ptr(rax + DESC_GPR), 0)?;
        a.mov(qword_ptr(rax + DESC_GPR + 3 * 8), rbx)?;
        a.mov(qword_ptr(rax + DESC_GPR + 4 * 8), rsp)?;
        a.mov(qword_ptr(rax + DESC_GPR + 6 * 8), rsi)?;
        a.set_label(&mut skip)?;
        a.test(rbx, rbx)?;
        a.jz(base)?;
        a.lea(rcx, qword_ptr(rbx - 1))?;
        a.mov(rdx, rsi)?;
        a.call(REC)?;
        a.add(rax, 0x10)?;
        a.jmp(done)?;
        a.set_label(&mut cont)?;
        a.lea(rax, qword_ptr(rbx + 0x1000))?;
        a.jmp(done)?;
        a.set_label(&mut base)?;
        restore(a)?;
        a.mov(rax, 0xBAD_u64)?;
        a.set_label(&mut done)?;
        a.add(rsp, 0x28)?;
        a.pop(rsi)?;
        a.pop(rbx)?;
        a.ret()?;
        Ok(vec![cont])
    })
}

/// SJ: rbx = 5151h; if (setjmp(jb) == 0) { rbx = clobbered; longjmp(jb, 7); } else
/// { [SEEN+8] = rbx; rax += 500h }
fn setjmp_program() -> (Program, u64) {
    let (code, labels) = assemble_labeled(SJ, |a| {
        let mut after = a.create_label();
        let mut second = a.create_label();
        let mut done = a.create_label();
        a.push(rbx)?;
        a.sub(rsp, 0x20)?;
        a.mov(rbx, 0x5151_u64)?;
        a.mov(rcx, JMP_BUF)?;
        a.call(SETJMP)?;
        a.set_label(&mut after)?;
        a.test(rax, rax)?;
        a.jnz(second)?;
        a.mov(rbx, 0xC0FFEE_u64)?;
        a.mov(rcx, JMP_BUF)?;
        a.call(LONGJMP)?;
        a.mov(rax, 0xBAD_u64)?;
        a.jmp(done)?;
        a.set_label(&mut second)?;
        a.add(rax, 0x500)?;
        a.mov(rcx, SEEN)?;
        a.mov(qword_ptr(rcx + 8), rbx)?;
        a.set_label(&mut done)?;
        a.add(rsp, 0x20)?;
        a.pop(rbx)?;
        a.ret()?;
        Ok(vec![after])
    });
    // setjmp(jb = rcx): jb = { return address, rsp after return, rbx }; rax = 0
    let setjmp = super::assemble(SETJMP, |a| {
        a.mov(rax, qword_ptr(rsp))?;
        a.mov(qword_ptr(rcx), rax)?;
        a.lea(rax, qword_ptr(rsp + 8))?;
        a.mov(qword_ptr(rcx + 8), rax)?;
        a.mov(qword_ptr(rcx + 16), rbx)?;
        a.xor(eax, eax)?;
        a.ret()
    });
    // longjmp(jb = rcx, 7): restore rip, rsp, rbx from jb with rax = 7
    let longjmp = super::assemble(LONGJMP, |a| {
        a.sub(rsp, 0x28)?;
        a.mov(rdx, DESC)?;
        a.mov(rax, qword_ptr(rcx))?;
        a.mov(qword_ptr(rdx), rax)?;
        a.mov(qword_ptr(rdx + 8), bit(RAX) | bit(RBX) | bit(RSP))?;
        a.mov(qword_ptr(rdx + DESC_GPR), 7)?;
        a.mov(rax, qword_ptr(rcx + 16))?;
        a.mov(qword_ptr(rdx + DESC_GPR + 3 * 8), rax)?;
        a.mov(rax, qword_ptr(rcx + 8))?;
        a.mov(qword_ptr(rdx + DESC_GPR + 4 * 8), rax)?;
        restore(a)?;
        a.add(rsp, 0x28)?;
        a.ret()
    });
    let program = Program::new()
        .bytes(SJ, &code)
        .bytes(SETJMP, &setjmp)
        .bytes(LONGJMP, &longjmp)
        .resume(SJ, labels[0]);
    (program, labels[0])
}

const PENDING: u64 = STATE_ADDR + STATE_UNWIND_PENDING_OFFSET;

/// Reads `SEEN[0..16]` and the pending byte after the run.
fn observed(case: Case) -> Case {
    case.read(SEEN, 16).read(PENDING, 1)
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().expect("8 bytes"))
}

fn assert_clean(o: &Outcome, label: &str) {
    assert!(o.error.is_empty(), "{label}: {}", o.error);
    assert_eq!(o.regs[RSP], STACK_TOP, "{label}: rsp");
    assert_eq!(o.mem[1], [0], "{label}: restore still pending");
}

// ----- IR shape -----------------------------------------------------------------------------------

#[test]
fn protocol_ir_depends_on_calls_and_resume_targets() {
    // A leaf has no check at all.
    let leaf = super::asm_function(0x1000, |a| {
        a.mov(eax, 1)?;
        a.ret()
    });
    let ir = lift_function(&leaf).expect("lifts").ir;
    assert!(!ir.contains("unwind"), "{ir}");

    // Calls without resume targets: every call is followed by the pending test, the shared
    // block just returns (no register write-back), no entry rsp, no dispatch.
    let (program, labels) = catcher_program(false, 0);
    let functions = program.discover();
    let g = lift_function(&functions[&G]).expect("lifts").ir;
    assert!(
        g.contains("i32 10\n") && g.contains("label %unwind, label %after_call"),
        "{g}"
    );
    assert!(g.contains("unwind:\n  ret void\n"), "{g}");
    assert!(!g.contains("%entry_rsp") && !g.contains("resume"), "{g}");

    // With resume targets: entry rsp, ownership test, dispatch on the resume rip.
    let f = &functions[&F];
    assert_eq!(
        f.resume_targets.iter().copied().collect::<Vec<_>>(),
        [labels[0]]
    );
    let ir = lift_function(f).expect("lifts").ir;
    assert!(ir.contains("%entry_rsp = load i64"), "{ir}");
    assert!(ir.contains("icmp ugt i64 %entry_rsp"), "{ir}");
    assert!(
        ir.contains(&format!(
            "label %resume_unknown [ i64 {}, label %bb_{:X} ]",
            labels[0], labels[0]
        )),
        "{ir}"
    );
    assert_eq!(ir.matches("label %unwind,").count(), 1);
}

// ----- execution ----------------------------------------------------------------------------------

#[test]
fn restore_unwinds_intermediate_frames_to_the_resume_target() {
    if !exec_or_skip("restore_unwinds_intermediate_frames_to_the_resume_target") {
        return;
    }
    let (program, _) = catcher_program(false, 0);
    let built = build("unwind_basic", program.discover().values());
    let case = observed(
        Case::new(F)
            .reg(RBX, 0xB0B0)
            .reg(RSI, 0x5151)
            .reg(R12, 0x1212),
    );
    let o = &run(&built, &[case])[0];
    assert_clean(o, "basic");
    // 40h from the restored context + 100h at the continuation (not the normal +1 path).
    assert_eq!(o.regs[RAX], 0x140);
    // The continuation saw the restored rbx, not the 0xBAD the abandoned frame left in its
    // registers (non-owners return without writing back).
    assert_eq!(u64_at(&o.mem[0], 0), 0x1111);
    // F's epilog restored the caller's nonvolatiles; G never got to restore rsi, but the
    // restored context carries F's value.
    assert_eq!(
        (o.regs[RBX], o.regs[R12], o.regs[RSI]),
        (0xB0B0, 0x1212, 0x5151)
    );
}

#[test]
fn recursion_resumes_the_activation_that_owns_the_frame() {
    if !exec_or_skip("recursion_resumes_the_activation_that_owns_the_frame") {
        return;
    }
    let (code, labels) = recursion();
    let functions = Program::new()
        .bytes(REC, &code)
        .resume(REC, labels[0])
        .discover();
    let built = build("unwind_recursion", functions.values());
    // (n, k) -> activation k resumes with n + 1000h, then each outer activation adds 10h.
    let cases: Vec<Case> = [(5u64, 2u64), (5, 5), (5, 0), (1, 1)]
        .iter()
        .map(|&(n, k)| {
            observed(
                Case::new(REC)
                    .reg(super::exec::RCX, n)
                    .reg(super::exec::RDX, k),
            )
        })
        .collect();
    let outcomes = run(&built, &cases);
    let expected = [0x1002 + 3 * 0x10, 0x1005, 0x1000 + 5 * 0x10, 0x1001];
    for (o, want) in outcomes.iter().zip(expected) {
        assert_clean(o, "recursion");
        assert_eq!(o.regs[RAX], want);
    }
}

#[test]
fn longjmp_shaped_restore_resumes_after_the_setjmp_call() {
    if !exec_or_skip("longjmp_shaped_restore_resumes_after_the_setjmp_call") {
        return;
    }
    let (program, after) = setjmp_program();
    let functions = program.discover();
    assert!(functions[&SJ].resume_targets.contains(&after));
    let built = build("unwind_setjmp", functions.values());
    let o = &run(&built, &[observed(Case::new(SJ).reg(RBX, 0x77))])[0];
    assert_clean(o, "setjmp");
    assert_eq!(o.regs[RAX], 0x507, "setjmp returned 7 the second time");
    assert_eq!(u64_at(&o.mem[0], 8), 0x5151, "rbx from the jump buffer");
    assert_eq!(o.regs[RBX], 0x77);
}

#[test]
fn unknown_resume_targets_reach_the_trap_helper() {
    if !exec_or_skip("unknown_resume_targets_reach_the_trap_helper") {
        return;
    }
    // The owner has no block registered as a resume target at the restored rip.
    let (program, labels) = catcher_program(true, 0);
    let built = build("unwind_unknown", program.discover().values());
    let o = &run(&built, &[observed(Case::new(F))])[0];
    assert_eq!(o.error, format!("trap at {:#x}", labels[1]));
    assert_eq!(o.mem[1], [1], "the runtime sees the restore still pending");

    // A restored rsp above every frame: nobody owns it, the restore leaves the lifted code
    // pending (the runtime's call_guest reports it).
    let (program, _) = catcher_program(false, 0x1000);
    let built = build("unwind_no_owner", program.discover().values());
    let o = &run(&built, &[observed(Case::new(F))])[0];
    assert!(o.error.is_empty(), "{}", o.error);
    assert_eq!(o.mem[1], [1]);
    assert_eq!(o.regs[RAX], 0x40, "registers are the restored context");
}
