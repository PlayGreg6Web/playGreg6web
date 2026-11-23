//! Always-run tests: IR structure, determinism, register/flag modelling, CFG handling and error
//! reporting.

use std::collections::BTreeMap;

use iced_x86::code_asm::*;
use iced_x86::{Code, Instruction, MemoryOperand, Register};

use super::{
    Program, SWITCH_CASES, SWITCH_TEXT_RVA, asm_function, assemble, encode, switch_program,
};
use crate::cfg::{self, Edge, Terminator};
use crate::lift::{
    HELPER_DIVIDE_ERROR, HELPER_INDIRECT_CALL, HELPER_TRAP, HELPER_UNIMPLEMENTED, LiftError,
    WASM64_TRIPLE, emit_module, lift_function, lift_module,
};

/// Instructions of the LLVM block `label` (up to the next label or the end of the function).
fn block_text(ir: &str, label: &str) -> String {
    let header = format!("{label}:");
    let mut lines = ir.lines().skip_while(|l| *l != header);
    assert!(lines.next().is_some(), "label {label} missing in:\n{ir}");
    let mut body = String::new();
    for line in lines.take_while(|l| l.starts_with("  ")) {
        body.push_str(line);
        body.push('\n');
    }
    body
}

/// Number of `store i1 ..., ptr %<flag>` lines (flag updates, not state syncs).
fn flag_stores(body: &str, flag: &str) -> usize {
    let suffix = format!(", ptr %{flag}");
    body.lines()
        .filter(|l| l.trim_start().starts_with("store i1 ") && l.ends_with(&suffix))
        .count()
}

fn trap_call(va: u64) -> String {
    format!("call void @{HELPER_TRAP}(ptr %state, i64 {va})\n  unreachable")
}

#[test]
fn raw_bytes_function_lifts_to_a_complete_module() {
    // lea rax, [rcx+rdx*4+0x10] ; ret
    let bytes = [0x48, 0x8D, 0x44, 0x91, 0x10, 0xC3];
    let func = Program::new().bytes(0x1000, &bytes).function(0x1000);
    let module = lift_module([&func]).expect("lifts");
    let ir = &module.ir;

    assert!(ir.contains(&format!("target triple = \"{WASM64_TRIPLE}\"")));
    assert!(ir.contains("target datalayout = \"e-m:e-p:64:64"));
    assert!(ir.contains(
        "%playgta6.GuestState = type { [16 x i64], i64, [8 x i8], i64, i64, [16 x [2 x i64]] }"
    ));
    assert!(ir.contains(&format!("declare void @{HELPER_INDIRECT_CALL}(ptr, i64)")));
    for helper in [HELPER_UNIMPLEMENTED, HELPER_DIVIDE_ERROR, HELPER_TRAP] {
        assert!(ir.contains(&format!("declare void @{helper}(ptr, i64) noreturn")));
    }
    assert!(ir.contains("define void @sub_1000(ptr noalias %state) {\nentry:\n"));
    assert!(ir.contains("  br label %bb_1000\nbb_1000:\n"));
    // Only the registers the function touches get a slot (rax, rcx, rdx + always rsp).
    for reg in ["rax", "rcx", "rdx", "rsp"] {
        assert!(ir.contains(&format!("  %{reg} = alloca i64\n")), "{reg}");
    }
    assert!(!ir.contains("%rbx = alloca"));
    let body = block_text(ir, "bb_1000");
    assert!(body.contains("shl i64"), "index scaled by 4:\n{body}");
    assert!(body.contains(", 16\n"), "displacement added:\n{body}");
    assert!(body.trim_end().ends_with("ret void"));
    assert!(module.functions[0].unsupported.is_empty());
}

#[test]
fn output_is_deterministic_and_independent_of_input_order() {
    let functions = Program::new()
        .asm(0x2000, |a| {
            a.mov(rax, rcx)?;
            a.call(0x3000)?;
            a.ret()
        })
        .asm(0x3000, |a| {
            a.add(rcx, rdx)?;
            a.jmp(0x4000)
        })
        .asm(0x4000, |a| a.ret())
        .discover();
    let (a, b) = (&functions[&0x2000], &functions[&0x3000]);
    let first = lift_module([a, b]).expect("lifts");
    let second = lift_module([b, a]).expect("lifts");
    let third = lift_module([a, b]).expect("lifts");
    assert_eq!(first.ir, second.ir);
    assert_eq!(first.ir, third.ir);
    // sub_3000 is defined in the module, sub_4000 is only tail-called: declared once.
    assert!(!first.ir.contains("declare void @sub_3000(ptr)"));
    assert_eq!(first.ir.matches("declare void @sub_4000(ptr)").count(), 1);
    assert!(
        first.ir.find("define void @sub_2000").expect("sub_2000")
            < first.ir.find("define void @sub_3000").expect("sub_3000")
    );
    // Lifting the whole discovered program is deterministic too.
    let all = lift_module(functions.values()).expect("lifts");
    assert_eq!(
        all.ir,
        lift_module(functions.values().rev()).expect("lifts").ir
    );
}

#[test]
fn sub_register_writes_follow_x86_64_rules() {
    let func = asm_function(0x1000, |a| {
        a.mov(eax, ecx)?; // zero-extends into rax
        a.mov(bl, cl)?; // merges bits 0..8
        a.mov(dh, cl)?; // merges bits 8..16 of rdx
        a.mov(si, cx)?; // merges bits 0..16
        a.movzx(edi, ch)?; // reads bits 8..16 of rcx
        a.ret()
    });
    let lifted = lift_function(&func).expect("lifts");
    let body = block_text(&lifted.ir, "bb_1000");
    assert!(
        body.contains("zext i32"),
        "32-bit write zero-extends:\n{body}"
    );
    assert!(
        body.contains("and i64 %t") && body.contains(", -256\n"),
        "low byte mask:\n{body}"
    );
    assert!(body.contains(", -65281\n"), "high byte mask:\n{body}");
    assert!(
        body.contains("shl i64") && body.contains(", 8\n"),
        "high byte placed:\n{body}"
    );
    assert!(body.contains(", -65536\n"), "word mask:\n{body}");
    assert!(
        body.contains("lshr i64") && body.contains("trunc i64"),
        "ch read:\n{body}"
    );
}

#[test]
fn flags_are_written_only_by_instructions_that_define_them() {
    let count_stores = |func: &cfg::Function| -> BTreeMap<&'static str, usize> {
        let ir = lift_function(func).expect("lifts").ir;
        let body = block_text(&ir, &format!("bb_{:X}", func.entry));
        ["cf", "pf", "af", "zf", "sf", "of"]
            .into_iter()
            .map(|f| (f, flag_stores(&body, f)))
            .collect()
    };
    // add defines all six arithmetic flags.
    let add = count_stores(&asm_function(0x1000, |a| {
        a.add(eax, ecx)?;
        a.ret()
    }));
    assert!(add.values().all(|&n| n == 1), "{add:?}");
    // logic ops leave the undefined AF alone.
    let and = count_stores(&asm_function(0x1000, |a| {
        a.and(eax, ecx)?;
        a.ret()
    }));
    assert_eq!(and["af"], 0, "{and:?}");
    assert_eq!(and["cf"], 1, "{and:?}");
    // inc preserves CF.
    let inc = count_stores(&asm_function(0x1000, |a| {
        a.inc(eax)?;
        a.ret()
    }));
    assert_eq!(inc["cf"], 0, "{inc:?}");
    assert_eq!(inc["of"], 1, "{inc:?}");
    // not and mov touch no flags.
    let not = count_stores(&asm_function(0x1000, |a| {
        a.not(eax)?;
        a.mov(ecx, 5)?;
        a.ret()
    }));
    assert!(not.values().all(|&n| n == 0), "{not:?}");
}

#[test]
fn conditional_branches_and_loops_become_llvm_branches() {
    // sum = 0; while (ecx) { sum += ecx; ecx--; }
    let func = asm_function(0x1000, |a| {
        let mut top = a.create_label();
        let mut done = a.create_label();
        a.xor(eax, eax)?;
        a.test(ecx, ecx)?;
        a.jz(done)?;
        a.set_label(&mut top)?;
        a.add(eax, ecx)?;
        a.dec(ecx)?;
        a.jnz(top)?;
        a.set_label(&mut done)?;
        a.ret()
    });
    assert_eq!(func.blocks.len(), 3);
    let ir = lift_function(&func).expect("lifts").ir;
    let labels: Vec<String> = func.blocks.keys().map(|k| format!("bb_{k:X}")).collect();
    // entry block: jz done / fallthrough into the loop
    assert!(ir.contains(&format!("label %{}, label %{}", labels[2], labels[1])));
    // loop block: jnz top / fallthrough to done
    let loop_body = block_text(&ir, &labels[1]);
    assert!(loop_body.contains(&format!("label %{}, label %{}", labels[1], labels[2])));
}

#[test]
fn calls_push_return_addresses_and_sync_state() {
    let program = Program::new()
        .asm(0x1000, |a| {
            a.sub(rsp, 0x28)?;
            a.call(0x5000)?;
            a.add(rsp, 0x28)?;
            a.ret()
        })
        .asm(0x5000, |a| a.ret());
    let lifted = lift_function(&program.function(0x1000)).expect("lifts");
    let body = block_text(&lifted.ir, "bb_1000");
    // return address = address after the 5-byte call at 0x1004
    assert!(
        body.contains("store i64 4105, ptr"),
        "return address pushed:\n{body}"
    );
    let call = body
        .find("call void @sub_5000(ptr %state)")
        .expect("direct call");
    let before = &body[..call];
    let after = &body[call..];
    assert!(before.rfind("store i64").is_some() && after.contains("load i64, ptr %t"));
    assert_eq!(lifted.callees.iter().copied().collect::<Vec<_>>(), [0x5000]);
}

#[test]
fn direct_calls_into_unmapped_memory_trap() {
    // call 0x9000 (nothing mapped there) ; ret
    let func = asm_function(0x1000, |a| {
        a.call(0x9000)?;
        a.ret()
    });
    let lifted = lift_function(&func).expect("lifts");
    assert!(lifted.ir.contains(&trap_call(0x9000)), "{}", lifted.ir);
    assert!(
        !lifted.ir.contains("@sub_9000"),
        "no call to a non-function"
    );
    assert!(lifted.callees.is_empty());
    // The instructions after the faulting call land in a predecessor-less block.
    assert!(lifted.ir.contains("\ndead"), "{}", lifted.ir);
}

#[test]
fn indirect_and_rip_relative_calls_use_the_runtime_dispatcher() {
    // call qword [rip+X] (import thunk style), then jmp rax (indirect tail jump)
    let slot = 0x1100u64;
    let bytes = encode(
        0x1000,
        &[
            Instruction::with1(
                Code::Call_rm64,
                MemoryOperand::with_base_displ(Register::RIP, slot as i64),
            )
            .expect("call"),
            Instruction::with1(Code::Jmp_rm64, Register::RAX).expect("jmp"),
        ],
    );
    let func = Program::new().bytes(0x1000, &bytes).function(0x1000);
    let block = &func.blocks[&0x1000];
    assert_eq!(block.terminator, Terminator::IndirectJump);
    let ir = lift_function(&func).expect("lifts").ir;
    assert!(
        ir.contains(&format!("inttoptr i64 {slot} to ptr")),
        "RIP-relative slot address:\n{ir}"
    );
    assert!(ir.contains(&format!(
        "  call void @{HELPER_INDIRECT_CALL}(ptr %state, i64 %t"
    )));
    assert!(ir.contains(&format!(
        "  tail call void @{HELPER_INDIRECT_CALL}(ptr %state, i64 %t"
    )));
}

#[test]
fn tail_call_edges_become_tail_calls() {
    let program = Program::new()
        .asm(0x1000, |a| {
            let mut skip = a.create_label();
            a.test(ecx, ecx)?;
            a.jz(skip)?;
            a.jmp(0x9000)?;
            a.set_label(&mut skip)?;
            a.jne(0x9000)?;
            a.ret()
        })
        .asm(0x9000, |a| a.ret())
        // mov eax, 1 at 0x2000 falls through into the function at 0x2005
        .bytes(0x2000, &[0xB8, 0x01, 0x00, 0x00, 0x00])
        .bytes(0x2005, &[0xC3]);
    let functions = program.discover();

    let jumps = &functions[&0x1000];
    assert!(jumps.tail_calls().all(|t| t == 0x9000) && jumps.tail_calls().count() == 2);
    let lifted = lift_function(jumps).expect("lifts");
    assert_eq!(lifted.ir.matches("\next_9000:\n").count(), 1);
    assert!(
        lifted
            .ir
            .contains("tail call void @sub_9000(ptr %state)\n  ret void")
    );
    assert!(lifted.callees.contains(&0x9000));

    let falls = &functions[&0x2000];
    assert_eq!(falls.blocks[&0x2000].successors, [Edge::TailCall(0x2005)]);
    let lifted = lift_function(falls).expect("lifts");
    assert!(
        lifted
            .ir
            .contains("tail call void @sub_2005(ptr %state)\n  ret void"),
        "{}",
        lifted.ir
    );
}

#[test]
fn blocks_shared_by_several_functions_are_lifted_into_each() {
    // Two functions jump into one shared tail (not a function entry).
    let shared = 0x3000u64;
    let functions = Program::new()
        .asm(0x1000, |a| {
            a.mov(eax, 1)?;
            a.jmp(shared)
        })
        .asm(0x2000, |a| {
            a.mov(eax, 2)?;
            a.jmp(shared)
        })
        .code(
            shared,
            &assemble(shared, |a| {
                a.add(eax, 10)?;
                a.ret()
            }),
        )
        .discover();
    assert_eq!(functions.len(), 2);
    let module = lift_module(functions.values()).expect("lifts");
    assert_eq!(module.ir.matches(&format!("\nbb_{shared:X}:\n")).count(), 2);
    for f in &module.functions {
        assert!(
            f.callees.is_empty(),
            "the shared block is not a call: {f:?}"
        );
    }
}

#[test]
fn trap_terminators_and_invalid_code_reach_the_trap_helper() {
    let program = Program::new()
        .asm(0x1000, |a| {
            a.nop()?;
            a.int3()
        })
        .asm(0x1100, |a| a.ud2())
        .asm(0x1200, |a| a.hlt())
        // int 0x29 (__fastfail)
        .bytes(0x1300, &[0xCD, 0x29])
        // nop ; then an invalid opcode (0x06 is `push es`, invalid in 64-bit mode)
        .bytes(0x1400, &[0x90, 0x06])
        // entry that is invalid right away: an empty block
        .bytes(0x1500, &[0x06]);
    let functions = program.discover();
    assert!(functions[&0x1500].blocks[&0x1500].instructions.is_empty());
    let module = lift_module(functions.values()).expect("lifts");
    for va in [0x1001u64, 0x1100, 0x1200, 0x1300, 0x1401, 0x1500] {
        assert!(
            module.ir.contains(&trap_call(va)),
            "trap at {va:#x}:\n{}",
            module.ir
        );
    }
    assert!(module.functions.iter().all(|f| f.unsupported.is_empty()));
}

#[test]
fn code_after_a_non_returning_call_traps() {
    let exit_va = 0x8000u64;
    let program = Program::new()
        .asm(0x1000, |a| {
            a.mov(ecx, 3)?;
            a.call(exit_va)?;
            a.mov(eax, 1)?; // never executed if exit really does not return
            a.ret()
        })
        .asm(exit_va, |a| a.ret())
        .no_return(exit_va);
    let func = program.function(0x1000);
    let block = &func.blocks[&0x1000];
    assert_eq!(block.terminator, Terminator::NoReturnCall);
    let ir = lift_function(&func).expect("lifts").ir;
    let body = block_text(&ir, "bb_1000");
    let call = body.find("call void @sub_8000(ptr %state)").expect("call");
    assert!(body[call..].contains(&trap_call(block.end)), "{body}");
}

#[test]
fn unsupported_instructions_trap_and_are_reported() {
    let func = asm_function(0x1000, |a| {
        let mut skip = a.create_label();
        a.test(ecx, ecx)?;
        a.jz(skip)?;
        a.cpuid()?;
        a.mov(eax, 1)?;
        a.set_label(&mut skip)?;
        a.movaps(xmm0, xmmword_ptr(rax))?;
        a.ret()
    });
    let module = lift_module([&func]).expect("lifts despite unsupported instructions");
    let report = &module.functions[0].unsupported;
    assert_eq!(report.len(), 2, "{report:?}");
    assert_eq!(report[0].instruction, "cpuid");
    assert_eq!(report[0].mnemonic, "cpuid");
    assert!(report[0].reason.contains("Cpuid"), "{:?}", report[0]);
    assert_eq!(report[1].mnemonic, "movaps");
    assert!(
        report[1].instruction.starts_with("movaps xmm0"),
        "{:?}",
        report[1]
    );
    let cpuid_va = report[0].address;
    let trap =
        format!("call void @{HELPER_UNIMPLEMENTED}(ptr %state, i64 {cpuid_va})\n  unreachable");
    assert!(module.ir.contains(&trap), "{}", module.ir);
    // The instruction after the trap in the same block is not lifted.
    let body = block_text(&module.ir, &format!("bb_{cpuid_va:X}"));
    assert!(!body.contains("store i64 1,"), "{body}");
}

#[test]
fn unsupported_conditional_branches_trap() {
    // loop rel8 is a conditional branch R2 understands but the lifter does not support yet.
    let func = asm_function(0x1000, |a| {
        let mut top = a.create_label();
        a.set_label(&mut top)?;
        a.inc(eax)?;
        a.loop_(top)?;
        a.ret()
    });
    let lifted = lift_function(&func).expect("lifts");
    assert_eq!(lifted.unsupported.len(), 1, "{:?}", lifted.unsupported);
    assert_eq!(lifted.unsupported[0].mnemonic, "loop");
}

#[test]
fn partially_lifted_instructions_are_rolled_back() {
    // `cmove eax, [eip+0x10]`: the condition (ZF) is read before the EIP-relative operand is
    // found to be unsupported; none of that partial IR may remain in front of the trap.
    // 67 0F 44 05 10 00 00 00 = cmove eax, [eip+0x10] ; C3 = ret
    let bytes = [0x67, 0x0F, 0x44, 0x05, 0x10, 0x00, 0x00, 0x00, 0xC3];
    let func = Program::new().bytes(0x1000, &bytes).function(0x1000);
    let lifted = lift_function(&func).expect("lifts");
    assert_eq!(lifted.unsupported.len(), 1, "{:?}", lifted.unsupported);
    assert_eq!(lifted.unsupported[0].reason, "EIP-relative addressing");
    let body = block_text(&lifted.ir, "bb_1000");
    // The only ZF read left is the state sync in front of the trap call.
    assert_eq!(body.matches("load i1, ptr %zf").count(), 1, "{body}");
    assert!(!body.contains("select"), "{body}");
    assert!(body.contains(HELPER_UNIMPLEMENTED), "{body}");
}

#[test]
fn division_has_a_divide_error_path_and_wide_ops_use_internal_helpers() {
    let func = asm_function(0x1000, |a| {
        a.xor(edx, edx)?;
        a.div(rcx)?;
        a.mul(rcx)?;
        a.cqo()?;
        a.idiv(r8)?;
        a.rol(eax, 3)?;
        a.ret()
    });
    let module = lift_module([&func]).expect("lifts");
    let ir = &module.ir;
    assert_eq!(
        ir.matches(&format!("call void @{HELPER_DIVIDE_ERROR}"))
            .count(),
        2
    );
    for helper in [
        "__playgta6_umul128",
        "__playgta6_udiv128",
        "__playgta6_sdiv128",
    ] {
        assert!(
            ir.lines()
                .any(|l| l.starts_with("define internal ") && l.contains(&format!("@{helper}("))),
            "{helper} defined"
        );
    }
    assert!(ir.contains("declare i32 @llvm.fshl.i32(i32, i32, i32)"));
    assert!(ir.contains("declare i8 @llvm.ctpop.i8(i8)"));
    assert!(
        !ir.contains(" i128 "),
        "no i128 arithmetic (would need compiler-rt on wasm)"
    );
}

/// A block built by hand (for CFG shapes discovery never produces).
fn block(
    start: u64,
    bytes: &[u8],
    terminator: Terminator,
    successors: Vec<Edge>,
) -> cfg::BasicBlock {
    let mut decoder = iced_x86::Decoder::with_ip(64, bytes, start, 0);
    let mut b = cfg::BasicBlock::new(start);
    while decoder.can_decode() {
        b.instructions.push(decoder.decode());
    }
    b.end = start + bytes.len() as u64;
    b.terminator = terminator;
    b.successors = successors;
    b
}

fn function(entry: u64, blocks: Vec<cfg::BasicBlock>) -> cfg::Function {
    cfg::Function {
        entry,
        blocks: blocks.into_iter().map(|b| (b.start, b)).collect(),
        callees: Default::default(),
    }
}

#[test]
fn malformed_cfgs_are_errors_not_panics() {
    let ret = |start| block(start, &[0xC3], Terminator::Return, vec![Edge::Return]);

    assert_eq!(
        lift_function(&function(0x2000, vec![ret(0x1000)])),
        Err(LiftError::MissingEntryBlock { entry: 0x2000 })
    );
    assert_eq!(
        lift_function(&function(
            0x1000,
            vec![block(0x1000, &[], Terminator::Return, vec![])]
        )),
        Err(LiftError::EmptyBlock { block: 0x1000 })
    );
    // ret ; nop in one block
    assert_eq!(
        lift_function(&function(
            0x1000,
            vec![block(
                0x1000,
                &[0xC3, 0x90],
                Terminator::Fallthrough,
                vec![]
            )]
        )),
        Err(LiftError::InstructionAfterTerminator {
            block: 0x1000,
            address: 0x1000
        })
    );
    // block keyed at an address its contents do not start at
    let mut moved = function(0x1004, vec![]);
    moved.blocks.insert(0x1004, ret(0x1000));
    assert_eq!(
        lift_function(&moved),
        Err(LiftError::BlockStartMismatch {
            block: 0x1004,
            first_instruction: 0x1000
        })
    );
    // jmp to an intra-function target that has no block
    assert_eq!(
        lift_function(&function(
            0x1000,
            vec![block(
                0x1000,
                &[0xEB, 0x10],
                Terminator::Jump,
                vec![Edge::Jump(0x1012)]
            )]
        )),
        Err(LiftError::UnknownSuccessor {
            block: 0x1000,
            target: 0x1012
        })
    );
    // the same function twice in one module
    let lifted = lift_function(&function(0x1000, vec![ret(0x1000)])).expect("lifts");
    assert_eq!(
        emit_module(&[lifted.clone(), lifted]),
        Err(LiftError::DuplicateFunction { entry: 0x1000 })
    );
}

#[test]
fn edges_without_a_static_target_degrade_to_traps_or_the_dispatcher() {
    // A fallthrough block without any successor edge cannot be lowered: unsupported trap.
    let no_edge = function(
        0x1000,
        vec![block(0x1000, &[0x90], Terminator::Fallthrough, vec![])],
    );
    let lifted = lift_function(&no_edge).expect("lifts");
    assert_eq!(lifted.unsupported.len(), 1);
    assert!(lifted.ir.contains(HELPER_UNIMPLEMENTED));

    // An indirect jump is always dispatched at run time, whatever edges are attached to it.
    let indirect = function(
        0x1000,
        vec![block(
            0x1000,
            &[0xFF, 0xE0], // jmp rax
            Terminator::IndirectJump,
            vec![Edge::IndirectJump { slot: None }, Edge::Return],
        )],
    );
    let lifted = lift_function(&indirect).expect("lifts");
    assert!(lifted.unsupported.is_empty());
    assert!(
        lifted
            .ir
            .contains(&format!("tail call void @{HELPER_INDIRECT_CALL}"))
    );
}

/// Small deterministic generator for fuzz-style inputs.
pub(super) struct Lcg(pub u64);

impl Lcg {
    pub fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 ^ (self.0 >> 29)
    }
}

/// Random byte soup run through discovery: invalid code, overlapping streams, branches into
/// other pieces and unmapped memory.
pub(super) fn random_program(count: usize) -> BTreeMap<u64, cfg::Function> {
    let mut rng = Lcg(0x5EED);
    let mut program = Program::new();
    for i in 0..count {
        let len = 16 + (rng.next() % 96) as usize;
        let bytes: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
        program = program.bytes(0x40_0000 + (i as u64) * 0x100, &bytes);
    }
    program.discover()
}

#[test]
fn random_byte_sequences_never_panic() {
    let functions = random_program(400);
    let module = lift_module(functions.values()).expect("discovered random code lifts");
    let traps: usize = module.functions.iter().map(|f| f.unsupported.len()).sum();
    assert!(
        traps > 0,
        "random code should hit some unsupported instructions"
    );
    assert!(module.ir.contains(HELPER_TRAP), "and some invalid code");
    assert_eq!(module.functions.len(), functions.len());
}

#[test]
fn resolved_jump_tables_become_switches() {
    let base = 0x1_4000_0000u64;
    let text = base + u64::from(SWITCH_TEXT_RVA);
    let functions = switch_program(base).discover();
    let lifted = lift_function(&functions[&text]).expect("lifts");
    let switch_line = lifted
        .ir
        .lines()
        .find(|l| l.trim_start().starts_with("switch i64 "))
        .unwrap_or_else(|| panic!("no switch:\n{}", lifted.ir));
    for case in SWITCH_CASES {
        let va = text + u64::from(case);
        assert!(
            switch_line.contains(&format!("i64 {va}, label %bb_{va:X}")),
            "{switch_line}"
        );
    }
    // Values outside the proven targets fall back to the runtime dispatcher.
    let default = switch_line
        .split("label %")
        .nth(1)
        .and_then(|rest| rest.split(' ').next())
        .expect("default label");
    let dispatch = block_text(&lifted.ir, default);
    assert!(
        dispatch.contains(&format!("tail call void @{HELPER_INDIRECT_CALL}")),
        "{dispatch}"
    );

    // Without the table bytes the jump stays unresolved: dispatcher only, no switch.
    let unresolved = Program::new()
        .base(base)
        .bytes(text, &super::switch_code())
        .function(text);
    let lifted = lift_function(&unresolved).expect("lifts");
    assert!(!lifted.ir.contains("switch i64"));
    assert!(
        lifted
            .ir
            .contains(&format!("tail call void @{HELPER_INDIRECT_CALL}"))
    );
}
#[test]
fn state_layout_constants_match_the_ir_type_and_the_import_seam() {
    use crate::lift::{
        CALLEE_STACK_ARG_OFFSET, STATE_FLAGS_OFFSET, STATE_FS_BASE_OFFSET, STATE_GPR_OFFSET,
        STATE_GS_BASE_OFFSET, STATE_RIP_OFFSET, STATE_SIZE, STATE_XMM_OFFSET, STATE_XMM_STRIDE,
    };
    use crate::stubs;

    // { [16 x i64], i64, [8 x i8], i64, i64, [16 x [2 x i64]] }, naturally aligned.
    let module = emit_module(&[]).expect("empty module");
    assert!(module.contains(
        "%playgta6.GuestState = type { [16 x i64], i64, [8 x i8], i64, i64, [16 x [2 x i64]] }"
    ));
    assert_eq!(STATE_GPR_OFFSET, 0);
    assert_eq!(STATE_RIP_OFFSET, STATE_GPR_OFFSET + 16 * 8);
    assert_eq!(STATE_FLAGS_OFFSET, STATE_RIP_OFFSET + 8);
    assert_eq!(STATE_FS_BASE_OFFSET, STATE_FLAGS_OFFSET + 8);
    assert_eq!(STATE_GS_BASE_OFFSET, STATE_FS_BASE_OFFSET + 8);
    assert_eq!(STATE_XMM_OFFSET, STATE_GS_BASE_OFFSET + 8);
    assert_eq!(STATE_XMM_STRIDE, 16);
    assert_eq!(STATE_SIZE, STATE_XMM_OFFSET + 16 * STATE_XMM_STRIDE);

    // The import thunks generated by `stubs` read the same state and stack layout.
    assert_eq!(stubs::SEAM_GPR_OFFSET, STATE_GPR_OFFSET);
    assert_eq!(stubs::SEAM_XMM_OFFSET, STATE_XMM_OFFSET);
    assert_eq!(stubs::SEAM_XMM_STRIDE, STATE_XMM_STRIDE);
    assert_eq!(stubs::STACK_ARG_OFFSET, CALLEE_STACK_ARG_OFFSET);
}
