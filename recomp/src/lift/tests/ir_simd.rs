//! IR-level tests of the SSE/AVX lifter: XMM state modelling, unsupported forms and fuzzing of
//! the SSE/VEX opcode space.

use std::collections::BTreeMap;

use iced_x86::code_asm::*;

use super::ir::Lcg;
use super::{Program, asm_function};
use crate::cfg;
use crate::lift::{HELPER_UNIMPLEMENTED, LIFTED_FUNCTION_ATTRIBUTES, lift_function, lift_module};

#[test]
fn used_xmm_registers_get_slots_synced_like_gprs() {
    let func = Program::new()
        .asm(0x1000, |a| {
            a.addps(xmm0, xmm5)?;
            a.call(0x2000)?;
            a.movaps(xmm5, xmm0)?;
            a.ret()
        })
        .asm(0x2000, |a| a.ret())
        .function(0x1000);
    let lifted = lift_function(&func).expect("lifts");
    let ir = &lifted.ir;
    assert!(lifted.unsupported.is_empty(), "{:?}", lifted.unsupported);
    assert!(ir.contains("  %xmm0 = alloca <2 x i64>\n"), "{ir}");
    assert!(ir.contains("  %xmm5 = alloca <2 x i64>\n"), "{ir}");
    assert!(
        !ir.contains("%xmm1 = alloca"),
        "only used registers get a slot"
    );
    let state_slot = |idx: u32| format!("ptr %state.xmm{idx}, align 8");
    // entry + after the call + before the call + before the return
    assert_eq!(ir.matches(&state_slot(0)).count(), 4, "{ir}");
    assert_eq!(ir.matches(&state_slot(5)).count(), 4, "{ir}");
    let call = ir.find("call void @sub_2000").expect("call");
    let before = &ir[..call];
    let after = &ir[call..];
    assert!(
        before.rfind("store <2 x i64>").unwrap()
            > before.rfind("load <2 x i64>, ptr %xmm0").unwrap()
    );
    assert!(
        after.contains("load <2 x i64>, ptr %state.xmm"),
        "reloaded after the call: {after}"
    );

    let plain = asm_function(0x1000, |a| {
        a.add(rax, rcx)?;
        a.ret()
    });
    let lifted = lift_function(&plain).expect("lifts");
    assert!(!lifted.ir.contains("xmm"), "{}", lifted.ir);
}

#[test]
fn ymm_upper_halves_get_slots_only_where_avx_touches_them() {
    let func = asm_function(0x1000, |a| {
        a.addps(xmm0, xmm1)?; // legacy SSE: upper halves untouched
        a.vaddps(ymm2, ymm3, ymm4)?; // 256-bit: reads ymm3/ymm4, writes ymm2 only
        a.vmulps(xmm5, xmm5, xmm5)?; // VEX-128: zeroes ymm5's upper half directly in state
        a.ret()
    });
    let lifted = lift_function(&func).expect("lifts");
    let ir = &lifted.ir;
    assert!(lifted.unsupported.is_empty(), "{:?}", lifted.unsupported);
    // Read upper halves (`ymm3`, `ymm4`) keep local allocas with entry load + exit store.
    for idx in [3, 4] {
        assert!(
            ir.contains(&format!("  %ymmh{idx} = alloca <2 x i64>\n")),
            "{ir}"
        );
        assert_eq!(
            ir.matches(&format!("ptr %state.ymmh{idx}, align 8"))
                .count(),
            2,
            "{ir}"
        );
    }
    // Write-only (`ymm2`) and VEX-128-zeroed (`ymm5`) upper halves skip local allocas and write
    // directly into `%state.ymmh*` (no entry load, no exit store).
    for idx in [2, 5] {
        assert!(!ir.contains(&format!("%ymmh{idx} = alloca")), "{ir}");
        assert!(
            !ir.contains(&format!("load <2 x i64>, ptr %state.ymmh{idx}")),
            "{ir}"
        );
        assert_eq!(
            ir.matches(&format!("ptr %state.ymmh{idx}, align 8"))
                .count(),
            1,
            "{ir}"
        );
    }
    for idx in [0, 1] {
        assert!(!ir.contains(&format!("%ymmh{idx} =")), "{ir}");
        assert!(!ir.contains(&format!("%state.ymmh{idx}")), "{ir}");
    }
    assert!(
        ir.contains("store <2 x i64> zeroinitializer, ptr %state.ymmh5, align 8"),
        "{ir}"
    );
}

#[test]
fn r62_unread_ymmh_elides_allocas_and_uses_memset_for_vzeroupper() {
    // 1. VEX-128 + vzeroupper + direct & indirect calls, reading zero ymmh registers.
    let prog = Program::new()
        .asm(0x1000, |a| {
            a.vxorps(xmm0, xmm0, xmm0)?;
            a.call(0x2000)?;
            a.call(rax)?;
            a.vzeroupper()?;
            a.ret()
        })
        .asm(0x2000, |a| a.ret());
    let f_vex_vzu = prog.function(0x1000);
    let lifted = lift_function(&f_vex_vzu).expect("lifts");
    let ir = &lifted.ir;
    assert!(lifted.unsupported.is_empty(), "{:?}", lifted.unsupported);
    assert!(
        !ir.contains("%ymmh"),
        "unread VEX-128/vzeroupper must emit zero %ymmh allocas:\n{ir}"
    );
    assert!(
        !ir.contains("load <2 x i64>, ptr %state.ymmh"),
        "unread VEX-128/vzeroupper must emit zero %state.ymmh loads:\n{ir}"
    );
    assert!(
        ir.contains("store <2 x i64> zeroinitializer, ptr %state.ymmh0, align 8"),
        "VEX-128 must still zero %state.ymmh0 directly:\n{ir}"
    );
    assert!(
        ir.contains("call void @llvm.memset.p0.i64(ptr align 8 %state.ymmh0, i8 0, i64 256, i1 false)"),
        "vzeroupper with no local ymmh must emit a single 256-byte memset:\n{ir}"
    );

    // 2. Reads ONLY ymm0 (`vmovdqu [rcx], ymm0`) + vzeroupper + call -> only %ymmh0 allocated!
    let prog_one = Program::new()
        .asm(0x1000, |a| {
            a.vmovdqu(ymmword_ptr(rcx), ymm0)?;
            a.call(rax)?;
            a.vzeroupper()?;
            a.ret()
        });
    let lifted_one = lift_function(&prog_one.function(0x1000)).expect("lifts");
    let ir_one = &lifted_one.ir;
    assert!(
        lifted_one.unsupported.is_empty(),
        "{:?}",
        lifted_one.unsupported
    );
    assert!(
        ir_one.contains("  %ymmh0 = alloca <2 x i64>\n"),
        "read ymm0 must allocate %ymmh0:\n{ir_one}"
    );
    for idx in 1..16 {
        assert!(
            !ir_one.contains(&format!("%ymmh{idx} = alloca")),
            "unread ymmh{idx} must not allocate:\n{ir_one}"
        );
        assert!(
            !ir_one.contains(&format!("load <2 x i64>, ptr %state.ymmh{idx}")),
            "unread ymmh{idx} must not reload from state:\n{ir_one}"
        );
    }
    assert!(
        ir_one.contains("call void @llvm.memset.p0.i64(ptr align 8 %state.ymmh0, i8 0, i64 256, i1 false)"),
        "vzeroupper with 1 local ymmh must bulk-zero state:\n{ir_one}"
    );
    assert!(
        ir_one.contains("store <2 x i64> zeroinitializer, ptr %ymmh0"),
        "vzeroupper must also zero local %ymmh0:\n{ir_one}"
    );

    // 3. Write-only 256-bit `vmovdqu ymm0, [rcx]` -> zero %ymmh allocas, direct store to %state.ymmh0.
    let f_write_only = asm_function(0x1000, |a| {
        a.vmovdqu(ymm0, ymmword_ptr(rcx))?;
        a.ret()
    });
    let lifted_wo = lift_function(&f_write_only).expect("lifts");
    let ir_wo = &lifted_wo.ir;
    assert!(
        lifted_wo.unsupported.is_empty(),
        "{:?}",
        lifted_wo.unsupported
    );
    assert!(
        !ir_wo.contains("%ymmh"),
        "write-only ymm0 must not allocate %ymmh:\n{ir_wo}"
    );
    assert!(
        !ir_wo.contains("load <2 x i64>, ptr %state.ymmh"),
        "write-only ymm0 must not load %state.ymmh:\n{ir_wo}"
    );
    assert_eq!(
        ir_wo.matches("ptr %state.ymmh0, align 8").count(),
        1,
        "write-only ymm0 must store directly to %state.ymmh0 once:\n{ir_wo}"
    );

    // 4. `vzeroall` without ymmh reads -> zero %ymmh allocas, single 256-byte memset.
    let f_vza = asm_function(0x1000, |a| {
        a.vzeroall()?;
        a.ret()
    });
    let lifted_vza = lift_function(&f_vza).expect("lifts");
    assert!(lifted_vza.unsupported.is_empty());
    assert!(!lifted_vza.ir.contains("%ymmh"), "{}", lifted_vza.ir);
    assert!(
        lifted_vza
            .ir
            .contains("call void @llvm.memset.p0.i64(ptr align 8 %state.ymmh0, i8 0, i64 256, i1 false)"),
        "{}",
        lifted_vza.ir
    );

    // 5. 256-bit zeroing idiom `vxorps ymm0, ymm0, ymm0` must still lift cleanly without trapping.
    let f_idiom = asm_function(0x1000, |a| {
        a.vxorps(ymm0, ymm0, ymm0)?;
        a.ret()
    });
    let lifted_idiom = lift_function(&f_idiom).expect("lifts");
    assert!(
        lifted_idiom.unsupported.is_empty(),
        "{:?}",
        lifted_idiom.unsupported
    );
}

#[test]
fn r63a_elides_redundant_unread_ymmh_zero_stores() {
    use crate::lift::{MemoryModel, lift_module_with};

    // 1. Multiple VEX-128 writes to the same unread xmm0 in one basic block (even across memory
    //    stores / atomics that block LLVM DSE) emit only ONE `%state.ymmh0` zero store.
    let f_same_bb = asm_function(0x1000, |a| {
        a.vxorps(xmm0, xmm0, xmm0)?;
        a.mov(qword_ptr(rcx), rax)?;
        a.vaddps(xmm0, xmm0, xmm1)?;
        a.lock().xadd(qword_ptr(rcx), rax)?;
        a.vmulps(xmm0, xmm0, xmm2)?;
        a.vsubps(xmm0, xmm0, xmm1)?;
        a.ret()
    });
    let ir_same = lift_module_with(
        [&f_same_bb],
        MemoryModel::from_name("window-runs").expect("model"),
    )
    .expect("lifts")
    .ir;
    assert_eq!(
        ir_same
            .matches("store <2 x i64> zeroinitializer, ptr %state.ymmh0, align 8")
            .count(),
        1,
        "repeated VEX-128 writes to xmm0 in one block must coalesce to one ymmh0 zero store:\n{ir_same}"
    );

    // 2. Calls (direct & indirect) reset `ymmh_zeroed`, so the first VEX-128 write after each
    //    call emits a zero store again, while subsequent ones in the same block are elided.
    let prog_call = Program::new()
        .asm(0x1000, |a| {
            a.vxorps(xmm0, xmm0, xmm0)?;
            a.vaddps(xmm0, xmm0, xmm1)?;
            a.call(0x2000)?;
            a.vmulps(xmm0, xmm0, xmm1)?;
            a.vsubps(xmm0, xmm0, xmm1)?;
            a.call(rax)?;
            a.vaddps(xmm0, xmm0, xmm1)?;
            a.vmulps(xmm0, xmm0, xmm1)?;
            a.ret()
        })
        .asm(0x2000, |a| a.ret());
    let ir_call = lift_function(&prog_call.function(0x1000)).expect("lifts").ir;
    assert_eq!(
        ir_call
            .matches("store <2 x i64> zeroinitializer, ptr %state.ymmh0, align 8")
            .count(),
        3,
        "one zero store before direct call, one after direct call, one after indirect call:\n{ir_call}"
    );

    // 3. Basic block boundaries reset `ymmh_zeroed`.
    let f_blocks = asm_function(0x1000, |a| {
        let mut target = a.create_label();
        a.vxorps(xmm0, xmm0, xmm0)?;
        a.vaddps(xmm0, xmm0, xmm1)?;
        a.test(rcx, rcx)?;
        a.jz(target)?;
        a.vmulps(xmm0, xmm0, xmm1)?;
        a.vsubps(xmm0, xmm0, xmm1)?;
        a.set_label(&mut target)?;
        a.vaddps(xmm0, xmm0, xmm1)?;
        a.vmulps(xmm0, xmm0, xmm1)?;
        a.ret()
    });
    let ir_blocks = lift_function(&f_blocks).expect("lifts").ir;
    assert_eq!(
        ir_blocks
            .matches("store <2 x i64> zeroinitializer, ptr %state.ymmh0, align 8")
            .count(),
        3,
        "each of the 3 basic blocks emits exactly 1 ymmh0 zero store:\n{ir_blocks}"
    );

    // 4. A 256-bit write to an unread `ymm0` clears bit 0 of `ymmh_zeroed`, so the next VEX-128
    //    write to `xmm0` MUST re-zero `%state.ymmh0` (soundness check!).
    let f_wide_reset = asm_function(0x1000, |a| {
        a.vxorps(xmm0, xmm0, xmm0)?;
        a.vaddps(xmm0, xmm0, xmm1)?; // elided
        a.vmovdqu(ymm0, ymmword_ptr(rcx))?; // non-zero store to %state.ymmh0, clears bit 0
        a.vaddps(xmm0, xmm0, xmm1)?; // must re-emit zero store!
        a.vmulps(xmm0, xmm0, xmm1)?; // elided again
        a.ret()
    });
    let ir_wide = lift_function(&f_wide_reset).expect("lifts").ir;
    assert!(!ir_wide.contains("%ymmh0 = alloca"), "{ir_wide}");
    assert_eq!(
        ir_wide
            .matches("store <2 x i64> zeroinitializer, ptr %state.ymmh0, align 8")
            .count(),
        2,
        "256-bit write to unread ymm0 must clear ymmh_zeroed bit 0:\n{ir_wide}"
    );
    assert_eq!(
        ir_wide.matches("ptr %state.ymmh0, align 8").count(),
        3,
        "2 zero stores + 1 non-zero 256-bit store to %state.ymmh0:\n{ir_wide}"
    );

    // 5. `vzeroupper` marks all unread `ymmh` slots zeroed, so following VEX-128 ops emit zero
    //    redundant `%state.ymmh*` stores.
    let f_vzu_then_vex = asm_function(0x1000, |a| {
        a.vzeroupper()?;
        a.vxorps(xmm0, xmm0, xmm0)?;
        a.vaddps(xmm1, xmm1, xmm2)?;
        a.ret()
    });
    let ir_vzu = lift_function(&f_vzu_then_vex).expect("lifts").ir;
    assert_eq!(
        ir_vzu
            .matches("call void @llvm.memset.p0.i64(ptr align 8 %state.ymmh0, i8 0, i64 256, i1 false)")
            .count(),
        1,
        "{ir_vzu}"
    );
    assert_eq!(
        ir_vzu
            .matches("store <2 x i64> zeroinitializer, ptr %state.ymmh")
            .count(),
        0,
        "VEX-128 ops after vzeroupper in the same block must elide redundant zero stores:\n{ir_vzu}"
    );

    // 6. `ymmh_zeroed` survives across a coalesced window burst (entry -> fast & slow -> join).
    let f_burst_zero = asm_function(0x1000, |a| {
        a.vxorps(xmm0, xmm0, xmm0)?;
        a.vmovdqu(xmm1, xmmword_ptr(rcx))?;
        a.vmovdqu(xmm2, xmmword_ptr(rcx + 16))?;
        a.vpaddd(xmm0, xmm1, xmm2)?;
        a.vmovdqu(xmmword_ptr(rbx), xmm0)?;
        a.vmovdqu(xmmword_ptr(rbx + 16), xmm0)?;
        a.vpaddd(xmm0, xmm0, xmm1)?;
        a.ret()
    });
    let ir_burst_zero = lift_module_with(
        [&f_burst_zero],
        MemoryModel::from_name("window-runs").expect("model"),
    )
    .expect("lifts")
    .ir;
    assert_eq!(
        ir_burst_zero
            .matches("store <2 x i64> zeroinitializer, ptr %state.ymmh0, align 8")
            .count(),
        1,
        "ymmh0 zeroed before burst must not be re-zeroed inside or after the burst:\n{ir_burst_zero}"
    );
}

#[test]
fn r64a_elides_dead_entry_xmm_loads_soundly() {
    let has_entry_load = |ir: &str, idx: usize| -> bool {
        let entry_end = ir.find("\nbb_").expect("entry block ends before bb_");
        let entry = &ir[..entry_end];
        entry.contains(&format!("load <2 x i64>, ptr %state.xmm{idx}, align 8"))
    };

    // 1. Entry-block overwrites (after standard prologue): vmovdqu, movdqu, vmovd, vmovq,
    //    zeroing idioms (vpxor/pxor/vxorps/vxorpd), and 3-op VEX non-merging arithmetic/unpack
    //    elide their entry `%state.xmmN` loads while keeping allocas and exit stores.
    let f_elide = asm_function(0x1000, |a| {
        a.push(rbx)?;
        a.sub(rsp, 0x20)?;
        a.vmovdqu(xmm0, xmmword_ptr(rcx))?; // full load -> elided
        a.movdqu(xmm1, xmmword_ptr(rdx))?; // legacy full load -> elided
        a.vmovd(xmm2, eax)?; // zero-extending move -> elided
        a.vmovq(xmm3, rax)?; // zero-extending move -> elided
        a.vpxor(xmm4, xmm4, xmm4)?; // VEX zeroing idiom -> elided
        a.pxor(xmm5, xmm5)?; // legacy zeroing idiom -> elided
        a.vxorps(xmm6, xmm6, xmm6)?; // VEX float zeroing idiom -> elided
        a.vxorpd(xmm7, xmm7, xmm7)?; // VEX double zeroing idiom -> elided
        a.vpaddw(xmm8, xmm9, xmm10)?; // 3-op VEX dst!=src -> xmm8 elided, xmm9/xmm10 kept!
        a.vpunpcklbw(xmm11, xmm0, xmm4)?; // 3-op VEX dst!=src -> xmm11 elided
        a.add(rsp, 0x20)?;
        a.pop(rbx)?;
        a.ret()
    });
    let lifted = lift_function(&f_elide).expect("lifts");
    let ir = &lifted.ir;
    assert!(lifted.unsupported.is_empty(), "{:?}", lifted.unsupported);
    for idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 11] {
        assert!(
            ir.contains(&format!("  %xmm{idx} = alloca <2 x i64>\n")),
            "alloca must remain for xmm{idx}:\n{ir}"
        );
        assert!(
            !has_entry_load(ir, idx),
            "dead entry load for overwritten xmm{idx} must be elided:\n{ir}"
        );
        assert!(
            ir.contains(&format!("ptr %state.xmm{idx}, align 8")),
            "exit sync_out store must remain for xmm{idx}:\n{ir}"
        );
    }
    for idx in [9, 10] {
        assert!(
            has_entry_load(ir, idx),
            "read-before-write xmm{idx} must keep its entry load:\n{ir}"
        );
    }

    // 2. Partial-register writes and read-modify-write instructions MUST preserve entry loads!
    let f_partial = asm_function(0x2000, |a| {
        a.movss(xmm0, xmm1)?; // reg-reg partial write (preserves upper 96 bits of xmm0)
        a.movsd_2(xmm2, xmm3)?; // reg-reg partial write (preserves upper 64 bits of xmm2)
        a.movlps(xmm4, qword_ptr(rcx))?; // partial half write
        a.pinsrd(xmm5, eax, 1)?; // partial lane insert
        a.insertps(xmm6, xmm7, 0)?; // partial lane insert
        a.blendps(xmm8, xmm9, 0x5)?; // legacy blend reads dst
        a.blendps(xmm10, xmm11, 0xF)?; // legacy blend even with 0xF mask reads dst in lifter
        a.paddw(xmm12, xmm13)?; // legacy RMW
        a.vpaddw(xmm14, xmm14, xmm15)?; // VEX RMW (dst == src1)
        a.ret()
    });
    let lifted_p = lift_function(&f_partial).expect("lifts");
    let ir_p = &lifted_p.ir;
    assert!(lifted_p.unsupported.is_empty(), "{:?}", lifted_p.unsupported);
    for idx in 0..16 {
        assert!(
            has_entry_load(ir_p, idx),
            "partial/RMW xmm{idx} must preserve its entry load:\n{ir_p}"
        );
    }

    // 3. FMA3, legacy blendvps (implicit xmm0), and pcmpeqd self-compare MUST preserve entry loads.
    let f_special = asm_function(0x3000, |a| {
        a.blendvps(xmm1, xmm2)?; // implicitly reads xmm0 + RMW xmm1
        a.vfmadd213ps(xmm3, xmm4, xmm5)?; // FMA reads op0 (xmm3)
        a.pcmpeqd(xmm6, xmm6)?; // lifter loads operands for pcmpeqd -> must NOT elide!
        a.ret()
    });
    let ir_s = lift_function(&f_special).expect("lifts").ir;
    for idx in [0, 1, 2, 3, 4, 5, 6] {
        assert!(
            has_entry_load(&ir_s, idx),
            "xmm{idx} must preserve its entry load:\n{ir_s}"
        );
    }

    // 4. Post-branch writes MUST preserve entry loads (only the entry block is scanned).
    let f_branch = asm_function(0x4000, |a| {
        let mut skip = a.create_label();
        a.test(rcx, rcx)?;
        a.jz(skip)?;
        a.vmovdqu(xmm0, xmmword_ptr(rdx))?;
        a.set_label(&mut skip)?;
        a.ret()
    });
    let ir_b = lift_function(&f_branch).expect("lifts").ir;
    assert!(
        has_entry_load(&ir_b, 0),
        "post-branch write to xmm0 must preserve entry load:\n{ir_b}"
    );

    // 5. Post-call writes MUST preserve entry loads (callee may observe `%state.xmm0`), while
    //    writes BEFORE the call are safely elided.
    let prog_call = Program::new()
        .asm(0x5000, |a| {
            a.vpxor(xmm1, xmm1, xmm1)?; // before call -> elided
            a.call(0x6000)?;
            a.vmovdqu(xmm0, xmmword_ptr(rcx))?; // after call -> preserved!
            a.ret()
        })
        .asm(0x6000, |a| a.ret());
    let ir_c = lift_function(&prog_call.function(0x5000)).expect("lifts").ir;
    assert!(
        !has_entry_load(&ir_c, 1),
        "xmm1 written before call should elide entry load:\n{ir_c}"
    );
    assert!(
        has_entry_load(&ir_c, 0),
        "xmm0 written only after call must preserve entry load:\n{ir_c}"
    );

    // 6. Writes AFTER a potentially-trapping instruction (`div`, `ldmxcsr`) MUST preserve entry
    //    loads so `cold_trap`'s `sync_out()` sees live `%state.xmmN`.
    let f_div = asm_function(0x7000, |a| {
        a.div(rcx)?;
        a.vmovdqu(xmm0, xmmword_ptr(rdx))?;
        a.ret()
    });
    let ir_div = lift_function(&f_div).expect("lifts").ir;
    assert!(
        has_entry_load(&ir_div, 0),
        "xmm0 written after potentially-trapping div must preserve entry load:\n{ir_div}"
    );
    let f_mxcsr = asm_function(0x8000, |a| {
        a.ldmxcsr(dword_ptr(rcx))?;
        a.vmovdqu(xmm0, xmmword_ptr(rdx))?;
        a.ret()
    });
    let ir_mxcsr = lift_function(&f_mxcsr).expect("lifts").ir;
    assert!(
        has_entry_load(&ir_mxcsr, 0),
        "xmm0 written after potentially-trapping ldmxcsr must preserve entry load:\n{ir_mxcsr}"
    );
}

#[test]
fn lifted_functions_require_wasm_simd() {
    let func = asm_function(0x1000, |a| {
        a.sqrtps(xmm0, xmm1)?;
        a.ret()
    });
    let module = lift_module([
        &func,
        &asm_function(0x2000, |a| {
            a.sqrtps(xmm2, xmm3)?;
            a.sqrtss(xmm2, xmm3)?;
            a.ret()
        }),
    ])
    .expect("lifts");
    assert!(
        module
            .ir
            .contains("define void @sub_1000(ptr noalias noundef %state) unnamed_addr #0 {")
    );
    assert!(module.ir.contains(&format!(
        "attributes #0 = {{ {LIFTED_FUNCTION_ATTRIBUTES} }}"
    )));
    assert!(LIFTED_FUNCTION_ATTRIBUTES.contains("+simd128"));
    assert_eq!(
        module
            .ir
            .matches("declare <4 x float> @llvm.sqrt.v4f32(<4 x float>)")
            .count(),
        1,
        "intrinsics are declared once"
    );
    assert!(module.ir.contains("declare float @llvm.sqrt.f32(float)"));
    assert!(!module.ir.contains("llvm.x86"), "no x86 intrinsics on wasm");
}

#[test]
fn unsupported_vector_forms_trap_with_precise_reasons() {
    let cases: Vec<(Vec<u8>, &str)> = vec![
        (
            super::assemble(0x1000, |a| a.vmpsadbw(ymm0, ymm1, ymm2, 0)),
            "256-bit form of Vmpsadbw is not supported yet",
        ),
        (
            super::assemble(0x1000, |a| a.sha1rnds4(xmm0, xmm1, 0)),
            "Sha1rnds4 is not supported yet",
        ),
        (
            vec![0x62, 0xF1, 0x74, 0x08, 0x58, 0xC2], // EVEX vaddps xmm0, xmm1, xmm2
            "AVX-512 (EVEX)",
        ),
        (
            super::assemble(0x1000, |a| a.paddb(mm0, mm1)),
            "MMX instructions",
        ),
        (super::assemble(0x1000, |a| a.emms()), "MMX instructions"),
        (
            super::assemble(0x1000, |a| a.mpsadbw(xmm0, xmm1, 0)),
            "Mpsadbw is not supported yet",
        ),
        (
            super::assemble(0x1000, |a| a.maskmovdqu(xmm0, xmm1)),
            "Maskmovdqu is not supported yet",
        ),
        (
            super::assemble(0x1000, |a| {
                a.vpgatherdd_3(xmm0, dword_ptr(rax + xmm1), xmm2)
            }),
            "gather",
        ),
    ];
    for (bytes, reason) in cases {
        let code = [bytes.as_slice(), &[0xC3]].concat();
        let func = Program::new().bytes(0x1000, &code).function(0x1000);
        let lifted = lift_function(&func).expect("lifts");
        assert_eq!(
            lifted.unsupported.len(),
            1,
            "{reason}: {:?}
{}",
            lifted.unsupported,
            lifted.ir
        );
        assert!(
            lifted.unsupported[0].reason.contains(reason),
            "{:?} should mention {reason:?}",
            lifted.unsupported[0]
        );
        assert!(lifted.ir.contains(HELPER_UNIMPLEMENTED));
    }
}

#[test]
fn ldmxcsr_checks_the_rounding_mode_at_run_time() {
    let func = asm_function(0x1000, |a| {
        a.ldmxcsr(dword_ptr(rcx))?;
        a.ret()
    });
    let lifted = lift_function(&func).expect("lifts");
    assert!(lifted.unsupported.is_empty());
    assert!(lifted.ir.contains("and i32 %"), "{}", lifted.ir);
    assert!(
        lifted.ir.contains("-40960"),
        "RC + reserved bit mask: {}",
        lifted.ir
    );
    assert!(
        lifted.ir.contains(&format!(
            "call void @{HELPER_UNIMPLEMENTED}(ptr %state, i64 4096)"
        )),
        "{}",
        lifted.ir
    );
}

/// Random instructions from the SSE/SSE2..4.2 (`[66|F2|F3] [REX] 0F xx`, `0F 38 xx`, `0F 3A xx`)
/// and VEX (`C5`/`C4`) opcode spaces with random ModRM/SIB/displacement/immediate bytes, run
/// through discovery.
pub(super) fn random_sse_program(count: usize) -> BTreeMap<u64, cfg::Function> {
    let mut rng = Lcg(0x55E_F022);
    let mut program = Program::new();
    for i in 0..count {
        let mut bytes = Vec::new();
        for _ in 0..24 {
            let r = rng.next();
            match r % 8 {
                0..=3 => {
                    let prefix = [None, Some(0x66), Some(0xF2), Some(0xF3)][(r >> 8) as usize % 4];
                    bytes.extend(prefix);
                    if r >> 12 & 3 == 0 {
                        bytes.push(0x40 | (r >> 16) as u8 & 0xF);
                    }
                    bytes.push(0x0F);
                    match r >> 20 & 7 {
                        0 => bytes.extend([0x38, (r >> 24) as u8]),
                        1 => bytes.extend([0x3A, (r >> 24) as u8]),
                        _ => bytes.push(0x10 | (r >> 24) as u8),
                    }
                }
                4 | 5 => bytes.extend([0xC5, (r >> 8) as u8 | 0x80, (r >> 16) as u8]),
                _ => bytes.extend([
                    0xC4,
                    (r >> 8) as u8 & 0xE3 | 0xE0,
                    (r >> 16) as u8 & 0x7F,
                    (r >> 24) as u8,
                ]),
            }
            // ModRM (mostly register-register), then SIB/displacement/immediate material.
            let modrm = (r >> 32) as u8;
            bytes.push(if r >> 40 & 3 == 0 {
                modrm
            } else {
                modrm | 0xC0
            });
            let tail = (r >> 44) as usize % 3;
            bytes.extend((0..tail).map(|_| rng.next() as u8));
        }
        bytes.push(0xC3);
        program = program.bytes(0x50_0000 + (i as u64) * 0x400, &bytes);
    }
    program.discover()
}

#[test]
fn random_sse_byte_sequences_never_panic() {
    let functions = random_sse_program(800);
    let module = lift_module(functions.values()).expect("discovered random SSE code lifts");
    assert_eq!(module.functions.len(), functions.len());
    let traps: usize = module.functions.iter().map(|f| f.unsupported.len()).sum();
    let lifted = module.ir.matches("load <2 x i64>, ptr %xmm").count();
    assert!(
        traps > 0 && lifted > 200,
        "traps {traps}, xmm loads {lifted}"
    );
}

#[test]
fn r61_supported_non_trapping_simd_never_traps_on_random_sse() {
    use crate::lift::function::simd::is_supported_non_trapping_simd;
    let functions = random_sse_program(800);
    let mut checked = 0usize;
    for func in functions.values() {
        let lifted = lift_function(func).expect("lifts");
        let trapped_addrs: std::collections::BTreeSet<u64> =
            lifted.unsupported.iter().map(|u| u.address).collect();
        for block in func.blocks.values() {
            for ins in &block.instructions {
                if is_supported_non_trapping_simd(ins) {
                    checked += 1;
                    assert!(
                        !trapped_addrs.contains(&ins.ip()),
                        "is_supported_non_trapping_simd accepted an instruction that trapped: {ins} ({:?}) at {:#x}",
                        ins.code(),
                        ins.ip()
                    );
                }
            }
        }
    }
    assert!(
        checked > 100,
        "expected >100 checked SIMD instructions, got {checked}"
    );
}

#[test]
fn r59_simd_vector_lowering_avoids_scalar_loops() {
    use crate::lift::target::LiftTarget;

    let cases: Vec<(&str, crate::cfg::Function)> = vec![
        (
            "psadbw",
            asm_function(0x1000, |a| {
                a.psadbw(xmm0, xmm1)?;
                a.ret()
            }),
        ),
        (
            "vpermd",
            asm_function(0x1100, |a| {
                a.vpermd(ymm0, ymm1, ymm2)?;
                a.ret()
            }),
        ),
        (
            "vpermilps_var",
            asm_function(0x1200, |a| {
                a.vpermilps(xmm0, xmm1, xmm2)?;
                a.ret()
            }),
        ),
        (
            "vpermilpd_var",
            asm_function(0x1300, |a| {
                a.vpermilpd(xmm0, xmm1, xmm2)?;
                a.ret()
            }),
        ),
        (
            "ptest",
            asm_function(0x1400, |a| {
                a.ptest(xmm0, xmm1)?;
                a.ret()
            }),
        ),
        (
            "vptest256",
            asm_function(0x1500, |a| {
                a.vptest(ymm0, ymm1)?;
                a.ret()
            }),
        ),
        (
            "phminposuw",
            asm_function(0x1600, |a| {
                a.phminposuw(xmm0, xmm1)?;
                a.ret()
            }),
        ),
        (
            "packsswb",
            asm_function(0x1700, |a| {
                a.packsswb(xmm0, xmm1)?;
                a.ret()
            }),
        ),
        (
            "pmulhrsw",
            asm_function(0x1800, |a| {
                a.pmulhrsw(xmm0, xmm1)?;
                a.ret()
            }),
        ),
        (
            "pmulhw",
            asm_function(0x1900, |a| {
                a.pmulhw(xmm0, xmm1)?;
                a.ret()
            }),
        ),
        (
            "pmuldq",
            asm_function(0x1A00, |a| {
                a.pmuldq(xmm0, xmm1)?;
                a.ret()
            }),
        ),
        (
            "cvtps2dq",
            asm_function(0x1B00, |a| {
                a.cvtps2dq(xmm0, xmm1)?;
                a.ret()
            }),
        ),
        (
            "cvtpd2dq",
            asm_function(0x1C00, |a| {
                a.cvtpd2dq(xmm0, xmm1)?;
                a.ret()
            }),
        ),
        (
            "cvtpd2ps",
            asm_function(0x1D00, |a| {
                a.cvtpd2ps(xmm0, xmm1)?;
                a.ret()
            }),
        ),
    ];

    let module = lift_module(cases.iter().map(|(_, f)| f)).expect("lift module");
    for (name, f) in cases.iter().zip(&module.functions) {
        assert!(f.unsupported.is_empty(), "{}: {:?}", name.0, f.unsupported);
        let ir = &f.ir;
        match name.0 {
            "psadbw" => {
                assert!(!ir.contains("extractelement"), "psadbw scalarized:\n{ir}");
                assert!(ir.contains("@llvm.umax.v16i8"), "{ir}");
                assert!(ir.contains("@llvm.umin.v16i8"), "{ir}");
            }
            "vpermd" | "vpermilps_var" => {
                assert!(!ir.contains("extractelement"), "{} scalarized:\n{ir}", name.0);
                assert!(ir.contains("@llvm.wasm.swizzle("), "{ir}");
            }
            "vpermilpd_var" => {
                assert!(!ir.contains("extractelement"), "vpermilpd scalarized:\n{ir}");
                assert!(ir.contains("select <2 x i1>"), "{ir}");
            }
            "ptest" | "vptest256" => {
                assert!(!ir.contains("extractelement"), "{} scalarized:\n{ir}", name.0);
                assert!(ir.contains("bitcast <2 x i1>"), "{ir}");
            }
            "phminposuw" => {
                assert!(ir.contains("@llvm.umin.v8i16"), "{ir}");
                assert!(ir.contains("@llvm.cttz.i16"), "{ir}");
            }
            "packsswb" => {
                assert!(ir.contains("@llvm.smax.v8i16"), "{ir}");
                assert!(ir.contains("@llvm.smin.v8i16"), "{ir}");
            }
            "pmulhrsw" => {
                assert!(ir.contains("16384"), "{ir}");
                assert!(ir.contains("lshr <4 x i32>"), "{ir}");
            }
            "pmulhw" => {
                assert!(!ir.contains("lshr"), "pmulhw should shuffle odd lanes directly:\n{ir}");
            }
            "pmuldq" => {
                assert!(ir.contains("sext <2 x i32>"), "{ir}");
            }
            "cvtps2dq" => {
                assert!(ir.contains("@llvm.fptosi.sat.v4i32.v4f32"), "{ir}");
            }
            "cvtpd2dq" => {
                assert!(ir.contains("@llvm.fptosi.sat.v2i32.v2f64"), "{ir}");
                assert!(
                    ir.contains("<4 x i32> <i32 0, i32 1, i32 2, i32 3>"),
                    "{ir}"
                );
            }
            "cvtpd2ps" => {
                assert!(
                    ir.contains("<4 x i32> <i32 0, i32 1, i32 2, i32 3>"),
                    "{ir}"
                );
            }
            _ => {}
        }
    }

    // Native retargeting must remain free of wasm-only intrinsics.
    let native_ir = LiftTarget::native_host().retarget(module.ir);
    assert!(
        !native_ir.contains("llvm.wasm."),
        "wasm-only intrinsic leaked into native IR"
    );
}
