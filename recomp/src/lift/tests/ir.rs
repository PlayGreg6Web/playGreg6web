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
    GUEST_BASE_SYMBOL, GuestBase, HELPER_DIVIDE_ERROR, HELPER_INDIRECT_CALL, HELPER_TRAP,
    HELPER_UNIMPLEMENTED, LiftError, WASM64_TRIPLE, emit_module, lift_function, lift_module,
    lift_module_with,
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
        "%playgta6.GuestState = type { [16 x i64], i64, [8 x i8], i64, i64, [16 x [2 x i64]], \
         [16 x [2 x i64]], i32, [3 x i32], i64, i8, [7 x i8] }"
    ));
    assert!(ir.contains(&format!(
        "declare void @{HELPER_INDIRECT_CALL}(ptr noundef, i64 noundef)"
    )));
    for helper in [HELPER_UNIMPLEMENTED, HELPER_DIVIDE_ERROR, HELPER_TRAP] {
        assert!(ir.contains(&format!(
            "declare void @{helper}(ptr noundef, i64 noundef) noreturn cold"
        )));
    }
    assert!(
        ir.contains("define void @sub_1000(ptr noalias noundef %state) unnamed_addr #0 {\nentry:\n")
    );
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
    assert!(!first.ir.contains("declare void @sub_3000(ptr noundef)"));
    assert_eq!(
        first.ir.matches("declare void @sub_4000(ptr noundef)").count(),
        1
    );
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
    // The registers are reloaded after the call (past the resume-protocol check).
    let call_at = lifted
        .ir
        .find("call void @sub_5000(ptr %state)")
        .expect("call");
    let after = &lifted.ir[call_at..];
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
    // The slot address goes through the guest-base seam (default: the runtime's base symbol).
    assert!(
        ir.contains(&format!("add i64 {slot}, %t")),
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
fn guest_addresses_go_through_the_selected_guest_base() {
    // mov rax, [0x2000]; mov [rcx + 8], rax; ret
    let func = asm_function(0x1000, |a| {
        a.mov(rax, qword_ptr(0x2000u64))?;
        a.mov(qword_ptr(rcx + 8), rax)?;
        a.ret()
    });
    let decl = format!("@{GUEST_BASE_SYMBOL} = external global i64, align 8");

    // Default: the runtime-defined base, an invariant load, declared once per module.
    let module = lift_module([&func]).expect("lifts").ir;
    let load = format!("load i64, ptr @{GUEST_BASE_SYMBOL}, align 8, !invariant.load !{{}}");
    assert!(module.contains(&load), "{module}");
    assert_eq!(module.matches(&decl).count(), 1, "{module}");
    assert!(
        module.contains("add i64 8192, %t"),
        "absolute address translated:\n{module}"
    );
    assert!(!module.contains("inttoptr i64 8192 to ptr"), "{module}");

    // Identity: plain inttoptr of the guest address, no base symbol.
    let identity = lift_module_with([&func], GuestBase::Constant(0))
        .expect("lifts")
        .ir;
    assert!(identity.contains("inttoptr i64 8192 to ptr"), "{identity}");
    assert!(!identity.contains(GUEST_BASE_SYMBOL), "{identity}");

    // A constant base folds into the address arithmetic (wrapping: shown as a signed i64).
    let base = 0xFFFF_FFFF_C000_0000u64;
    let constant = lift_module_with([&func], GuestBase::Constant(base))
        .expect("lifts")
        .ir;
    assert!(
        constant.contains(&format!("add i64 8192, {}", base as i64)),
        "{constant}"
    );
    assert!(!constant.contains(GUEST_BASE_SYMBOL), "{constant}");
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
    let call = ir.find("call void @sub_8000(ptr %state)").expect("call");
    assert!(ir[call..].contains(&trap_call(block.end)), "{ir}");
}

#[test]
fn unsupported_instructions_trap_and_are_reported() {
    let func = asm_function(0x1000, |a| {
        let mut skip = a.create_label();
        a.test(ecx, ecx)?;
        a.jz(skip)?;
        a.in_(eax, dx)?;
        a.mov(eax, 1)?;
        a.set_label(&mut skip)?;
        a.fld(dword_ptr(rax))?;
        a.ret()
    });
    let module = lift_module([&func]).expect("lifts despite unsupported instructions");
    let report = &module.functions[0].unsupported;
    assert_eq!(report.len(), 2, "{report:?}");
    assert_eq!(report[0].instruction, "in eax,dx");
    assert_eq!(report[0].mnemonic, "in");
    assert!(report[0].reason.contains("In"), "{:?}", report[0]);
    assert_eq!(report[1].mnemonic, "fld");
    assert!(
        report[1].instruction.starts_with("fld st,dword ptr"),
        "{:?}",
        report[1]
    );
    let in_va = report[0].address;
    let fld_va = report[1].address;
    // Both unsupported sites share one cold_trap sync-out and call via a phi of their VAs.
    let cold = block_text(&module.ir, "cold_trap");
    assert!(
        cold.contains(&format!("[ {in_va}, %bb_{in_va:X} ]"))
            && cold.contains(&format!("[ {fld_va}, %bb_{fld_va:X} ]")),
        "{cold}"
    );
    assert_eq!(
        module
            .ir
            .matches(&format!("call void @{HELPER_UNIMPLEMENTED}("))
            .count(),
        1,
        "{}",
        module.ir
    );
    assert!(
        cold.contains(&format!("call void @{HELPER_UNIMPLEMENTED}(ptr %state, i64 %t")),
        "{cold}"
    );
    assert!(cold.trim_end().ends_with("unreachable"), "{cold}");
    // The instruction after the trap in the same block is not lifted.
    let body = block_text(&module.ir, &format!("bb_{in_va:X}"));
    assert!(!body.contains("store i64 1,"), "{body}");
    assert_eq!(body.trim(), "br label %cold_trap", "{body}");
}

#[test]
fn unsupported_conditional_branches_trap() {
    // `addr32 loop` (ECX counter) is a conditional branch R2 understands but the lifter does not
    // support: inc eax ; addr32 loop 0x1000 ; ret
    let bytes = [0xFF, 0xC0, 0x67, 0xE2, 0xFB, 0xC3];
    let func = Program::new().bytes(0x1000, &bytes).function(0x1000);
    let lifted = lift_function(&func).expect("lifts");
    assert_eq!(lifted.unsupported.len(), 1, "{:?}", lifted.unsupported);
    assert_eq!(lifted.unsupported[0].mnemonic, "loop");
    assert!(
        lifted.unsupported[0].reason.contains("32-bit address size"),
        "{:?}",
        lifted.unsupported[0]
    );
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
    // No partial cmove IR remains in bb_1000: only the branch to the shared cold trap sink.
    assert_eq!(body.trim(), "br label %cold_trap", "{body}");
    assert!(!body.contains("select"), "{body}");
    let cold = block_text(&lifted.ir, "cold_trap");
    // The only ZF read left is the state sync in front of the trap call.
    assert_eq!(cold.matches("load i1, ptr %zf").count(), 1, "{cold}");
    assert!(cold.contains(HELPER_UNIMPLEMENTED), "{cold}");
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
        a.add(eax, ecx)?;
        a.ret()
    });
    let module = lift_module([&func]).expect("lifts");
    let ir = &module.ir;
    // Both div/idiv divide-error paths branch to the shared cold_trap sink (1 helper call).
    assert_eq!(
        ir.matches(&format!("call void @{HELPER_DIVIDE_ERROR}"))
            .count(),
        1
    );
    assert_eq!(ir.matches("label %cold_trap,").count(), 2, "{ir}");
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

#[test]
fn truncating_imul64_result_is_a_plain_mul() {
    let func = asm_function(0x1000, |a| {
        a.imul_2(rax, rcx)?;
        a.ret()
    });
    let lifted = lift_function(&func).expect("lifts");
    let ir = &lifted.ir;
    // RAX gets a plain `mul i64`; the 128-bit helper only feeds CF/OF (dead here, so LLVM drops
    // it and its four 32-bit multiplies).
    let mul = ir
        .lines()
        .find(|l| l.contains(" = mul i64 "))
        .expect("plain mul");
    let r = mul.trim().split(' ').next().expect("name");
    assert!(ir.contains(&format!("store i64 {r}, ptr %rax")), "{ir}");
    // CF/OF still use the high half.
    assert!(ir.contains("@__playgta6_umul128("), "{ir}");
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
        resume_targets: Default::default(),
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
        STATE_GS_BASE_OFFSET, STATE_MXCSR_OFFSET, STATE_RESUME_RIP_OFFSET, STATE_RIP_OFFSET,
        STATE_SIZE, STATE_UNWIND_PENDING_OFFSET, STATE_XMM_OFFSET, STATE_XMM_STRIDE,
        STATE_YMM_HI_OFFSET, STATE_YMM_HI_STRIDE,
    };
    use crate::stubs;

    // { [16 x i64], i64, [8 x i8], i64, i64, [16 x [2 x i64]], [16 x [2 x i64]], i32, [3 x i32],
    //   i64, i8, [7 x i8] }, naturally aligned.
    let module = emit_module(&[]).expect("empty module");
    assert!(module.contains(
        "%playgta6.GuestState = type { [16 x i64], i64, [8 x i8], i64, i64, [16 x [2 x i64]], \
         [16 x [2 x i64]], i32, [3 x i32], i64, i8, [7 x i8] }"
    ));
    assert_eq!(STATE_GPR_OFFSET, 0);
    assert_eq!(STATE_RIP_OFFSET, STATE_GPR_OFFSET + 16 * 8);
    assert_eq!(STATE_FLAGS_OFFSET, STATE_RIP_OFFSET + 8);
    assert_eq!(STATE_FS_BASE_OFFSET, STATE_FLAGS_OFFSET + 8);
    assert_eq!(STATE_GS_BASE_OFFSET, STATE_FS_BASE_OFFSET + 8);
    assert_eq!(STATE_XMM_OFFSET, STATE_GS_BASE_OFFSET + 8);
    assert_eq!(STATE_XMM_STRIDE, 16);
    assert_eq!(
        STATE_YMM_HI_OFFSET,
        STATE_XMM_OFFSET + 16 * STATE_XMM_STRIDE
    );
    assert_eq!(STATE_YMM_HI_STRIDE, 16);
    assert_eq!(
        STATE_MXCSR_OFFSET,
        STATE_YMM_HI_OFFSET + 16 * STATE_YMM_HI_STRIDE
    );
    assert_eq!(STATE_RESUME_RIP_OFFSET, STATE_MXCSR_OFFSET + 16);
    assert_eq!(STATE_UNWIND_PENDING_OFFSET, STATE_RESUME_RIP_OFFSET + 8);
    assert_eq!(STATE_SIZE, STATE_UNWIND_PENDING_OFFSET + 8);
    assert_eq!(STATE_SIZE % 16, 0);

    // The import thunks generated by `stubs` read the same state and stack layout.
    assert_eq!(stubs::SEAM_GPR_OFFSET, STATE_GPR_OFFSET);
    assert_eq!(stubs::SEAM_XMM_OFFSET, STATE_XMM_OFFSET);
    assert_eq!(stubs::SEAM_XMM_STRIDE, STATE_XMM_STRIDE);
    assert_eq!(stubs::STACK_ARG_OFFSET, CALLEE_STACK_ARG_OFFSET);
}

#[test]
fn r36_lossless_ir_size_optimizations_for_windowed_multi_block_functions() {
    use crate::lift::MemoryModel;
    use crate::lift::compile::{compile_to_wasm64_object, find_llvm_tool};

    // 1. Flag-free leaf function: zero flag allocas and zero flag state syncs.
    let flag_free = asm_function(0x1000, |a| {
        a.mov(rax, qword_ptr(rcx + 8))?;
        a.mov(qword_ptr(rdx), rax)?;
        a.ret()
    });
    let flag_free_mod = lift_module_with(
        [&flag_free],
        MemoryModel::from_name("window").expect("window model"),
    )
    .expect("lifts");
    for flag in ["%cf", "%pf", "%af", "%zf", "%sf", "%of"] {
        assert!(
            !flag_free_mod.ir.contains(&format!("{flag} = alloca")),
            "flag-free function must not allocate {flag}:\n{}",
            flag_free_mod.ir
        );
    }

    // 2. Representative multi-block windowed function with multiple returns, mixed cold traps
    //    (int3 + ud2 + div), multiple non-stack loads & stores, immediate shift, xor/test idioms.
    let multi = asm_function(0x2000, |a| {
        let mut slow_err = a.create_label();
        let mut early_ret = a.create_label();
        let mut second_trap = a.create_label();
        a.test(rcx, rcx)?;
        a.jz(early_ret)?;
        a.mov(rax, qword_ptr(rcx))?;
        a.mov(r8d, dword_ptr(rcx + 8))?;
        a.shl(rax, 3)?;
        a.mov(qword_ptr(rdx), rax)?;
        a.mov(dword_ptr(rdx + 8), r8d)?;
        a.test(r9, r9)?;
        a.jz(slow_err)?;
        a.xor(edx, edx)?;
        a.div(r9)?;
        a.test(r10, r10)?;
        a.jz(second_trap)?;
        a.ret()?;
        a.set_label(&mut early_ret)?;
        a.xor(eax, eax)?;
        a.ret()?;
        a.set_label(&mut slow_err)?;
        a.int3()?;
        a.set_label(&mut second_trap)?;
        a.ud2()
    });
    let windowed = lift_module_with(
        [&multi],
        MemoryModel::from_name("window").expect("window model"),
    )
    .expect("lifts");
    let ir = &windowed.ir;

    // Attributes: unnamed_addr, noundef, cold noreturn on trap helpers, cold on straddle helpers.
    assert!(ir.contains("define void @sub_2000(ptr noalias noundef %state) unnamed_addr #0 {"));
    assert!(ir.contains("declare void @__playgta6_trap(ptr noundef, i64 noundef) noreturn cold"));
    assert!(ir.contains(
        "declare void @__playgta6_divide_error(ptr noundef, i64 noundef) noreturn cold"
    ));
    assert!(ir.contains("declare ptr @__playgta6_guest_read(i64 noundef, i64 noundef) cold"));
    assert!(ir.contains("declare ptr @__playgta6_guest_scratch() cold"));
    assert!(ir.contains("declare void @__playgta6_guest_write(i64 noundef, i64 noundef) cold"));

    // Hoisted @__playgta6_guest_base load: strictly ONCE in entry:, not per memory access!
    assert_eq!(
        ir.matches("load i64, ptr @__playgta6_guest_base, align 8, !invariant.load !{}")
            .count(),
        1,
        "guest base must be hoisted once into entry:\n{ir}"
    );
    let fn_start = ir.find("define void @sub_2000(").expect("sub_2000");
    let fn_ir = &ir[fn_start..];
    let entry = block_text(fn_ir, "entry");
    assert!(
        entry.contains("@__playgta6_guest_base, align 8, !invariant.load !{}"),
        "hoisted base must live in entry:\n{entry}"
    );
    assert!(
        entry.contains("!range !{i8 0, i8 2}"),
        "flag loads carry binary range metadata:\n{entry}"
    );

    // Shared cold_trap sink: 1 sync_out shared across int3 + ud2 + div (#DE), 1 call per kind.
    assert_eq!(ir.matches("\ncold_trap:\n").count(), 1, "{ir}");
    assert_eq!(ir.matches("call void @__playgta6_trap(").count(), 1, "{ir}");
    assert_eq!(
        ir.matches("call void @__playgta6_divide_error(").count(),
        1,
        "{ir}"
    );
    assert!(ir.contains("switch i8 "), "mixed trap kinds share sync_out:\n{ir}");

    // Shared ret_epilog sink across both `ret` instructions.
    assert_eq!(ir.matches("\nret_epilog:\n").count(), 1, "{ir}");
    assert_eq!(ir.matches("br label %ret_epilog").count(), 2, "{ir}");

    // Compact windowed accesses: 2 non-stack loads -> 2 `align 1` loads total (not 6!),
    // 2 non-stack stores -> one `align 1` store on the flat and one on the window path each (O37:
    // the window path marks its slot after the store, without a second branch) + outlined cold
    // slow_store helpers.
    assert_eq!(
        fn_ir.matches(", align 1\n").count(),
        8,
        "2 loads + 2 x 2 stores + 2 slot marks = 8 align-1 accesses total:\n{fn_ir}"
    );
    assert_eq!(
        fn_ir.matches("store atomic i8 1, ptr ").count(),
        2,
        "each window store marks its slot:\n{fn_ir}"
    );
    assert!(
        ir.contains("define internal void @__playgta6_slow_store8(i64 noundef %va, i64 noundef %v) unnamed_addr noinline cold #0 {"),
        "{ir}"
    );
    assert!(
        ir.contains("define internal void @__playgta6_slow_store4(i64 noundef %va, i32 noundef %v) unnamed_addr noinline cold #0 {"),
        "{ir}"
    );
    assert!(
        !fn_ir.contains("@__playgta6_guest_scratch"),
        "slow store must be outlined out of lifted functions:\n{fn_ir}"
    );

    // Immediate shift (`shl rax, 3`) does not emit conditional flag select muxes.
    assert!(
        !fn_ir.contains("select i1 "),
        "constant-count shift must not emit flag selects:\n{fn_ir}"
    );

    // 3. `window-outline` CLI model emits compact outlined helpers with native v128 for 16-byte ops.
    let simd_mem = asm_function(0x3000, |a| {
        a.movdqu(xmm0, xmmword_ptr(rcx))?;
        a.movdqu(xmmword_ptr(rdx), xmm0)?;
        a.ret()
    });
    let outlined = lift_module_with(
        [&simd_mem],
        MemoryModel::from_name("window-outline").expect("window-outline model"),
    )
    .expect("lifts");
    assert!(
        outlined.ir.contains(
            "define internal <2 x i64> @__playgta6_wload16_400000000(i64 noundef %va) unnamed_addr noinline cold #0 {"
        ),
        "{}",
        outlined.ir
    );
    assert!(
        outlined.ir.contains(
            "define internal void @__playgta6_wstore16_400000000(i64 noundef %va, <2 x i64> noundef %v) unnamed_addr noinline cold #0 {"
        ),
        "{}",
        outlined.ir
    );
    assert!(
        !outlined.ir.contains(" i128 ") && !outlined.ir.contains("(i128"),
        "16-byte outlined helpers use native <2 x i64>, not i128:\n{}",
        outlined.ir
    );

    // 4. Verify all three emitted modules compile cleanly to valid wasm64 objects via LLVM.
    if let Some(tool) = find_llvm_tool() {
        let dir = std::env::temp_dir().join(format!(
            "playgta6_r36_ir_size_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        for (name, text) in [
            ("flag_free", &flag_free_mod.ir),
            ("windowed", &windowed.ir),
            ("outlined", &outlined.ir),
        ] {
            let ll = dir.join(format!("{name}.ll"));
            let obj = dir.join(format!("{name}.o"));
            std::fs::write(&ll, text).expect("write .ll");
            compile_to_wasm64_object(&tool, &ll, &obj)
                .unwrap_or_else(|e| panic!("{name} failed LLVM compilation: {e}"));
            let bytes = std::fs::metadata(&obj).expect("obj metadata").len();
            assert!(bytes > 0, "{name} object empty");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
fn shard_dispatch_emits_switch_internal_linkage_and_empty_shard_trap() {
    use std::collections::HashSet;

    use crate::lift::compile::{compile_to_wasm64_object, find_llvm_tool};
    use crate::lift::{ShardDispatch, emit_module_with_dispatch};

    let f0 = asm_function(0x1000, |a| {
        a.mov(eax, 1)?;
        a.ret()
    });
    let f1 = asm_function(0x2000, |a| {
        a.mov(eax, 2)?;
        a.ret()
    });
    // Pass unsorted to confirm ascending VA ordering determines local_idx.
    let lifted = vec![
        lift_function(&f1).expect("lift f1"),
        lift_function(&f0).expect("lift f0"),
    ];

    // `None` is byte-identical to `emit_module`.
    assert_eq!(
        emit_module(&lifted).expect("emit_module"),
        emit_module_with_dispatch(&lifted, None).expect("emit_module_with_dispatch(None)")
    );

    // Only 0x2000 is cross-shard external; 0x1000 becomes `internal`.
    let mut external = HashSet::new();
    external.insert(0x2000);
    let shard_ir = emit_module_with_dispatch(
        &lifted,
        Some(ShardDispatch {
            dispatcher_symbol: "__playgta6_dispatch_shard_main_00000",
            external_entries: &external,
        }),
    )
    .expect("emit shard");
    assert!(
        shard_ir.contains("define internal void @sub_1000(ptr noalias noundef %state) unnamed_addr #0 {"),
        "shard-local function should have internal linkage:\n{shard_ir}"
    );
    assert!(
        shard_ir.contains("\ndefine void @sub_2000(ptr noalias noundef %state) unnamed_addr #0 {"),
        "cross-shard callee must keep external linkage:\n{shard_ir}"
    );
    assert!(
        shard_ir.contains(
            "define void @__playgta6_dispatch_shard_main_00000(i32 noundef %local_idx, ptr noundef %state) #0 {\n\
             entry:\n  \
               switch i32 %local_idx, label %invalid [\n    \
                 i32 0, label %f0\n    \
                 i32 1, label %f1\n  \
               ]\n\
             f0:\n  \
               tail call void @sub_1000(ptr noundef %state)\n  \
               ret void\n\
             f1:\n  \
               tail call void @sub_2000(ptr noundef %state)\n  \
               ret void\n\
             invalid:\n  \
               tail call void @__playgta6_trap(ptr noundef %state, i64 noundef 0)\n  \
               unreachable\n\
             }\n"
        ),
        "unexpected dispatcher IR:\n{shard_ir}"
    );

    // Empty shard emits a direct trap without a switch.
    let empty_ir = emit_module_with_dispatch(
        &[],
        Some(ShardDispatch {
            dispatcher_symbol: "__playgta6_dispatch_shard_main_00001",
            external_entries: &external,
        }),
    )
    .expect("emit empty shard");
    assert!(!empty_ir.contains("switch i32"), "{empty_ir}");
    assert!(
        empty_ir.contains(
            "define void @__playgta6_dispatch_shard_main_00001(i32 noundef %local_idx, ptr noundef %state) #0 {\n\
             entry:\n  \
               tail call void @__playgta6_trap(ptr noundef %state, i64 noundef 0)\n  \
               unreachable\n\
             }\n"
        ),
        "unexpected empty shard dispatcher IR:\n{empty_ir}"
    );

    // Compile both through LLVM when available.
    if let Some(tool) = find_llvm_tool() {
        let dir = std::env::temp_dir().join(format!(
            "playgta6_r37b_shard_ir_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        for (name, text) in [("shard", &shard_ir), ("empty_shard", &empty_ir)] {
            let ll = dir.join(format!("{name}.ll"));
            let obj = dir.join(format!("{name}.o"));
            std::fs::write(&ll, text).expect("write .ll");
            compile_to_wasm64_object(&tool, &ll, &obj)
                .unwrap_or_else(|e| panic!("{name} failed LLVM compilation: {e}"));
            assert!(std::fs::metadata(&obj).expect("obj metadata").len() > 0);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
