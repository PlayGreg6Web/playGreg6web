//! Tests for issue #2194:
//! 1. Native LLVM bit-manipulation intrinsics & constant folding (`recomp/src/lift/bit_ops.rs`,
//!    `recomp/src/lift/function.rs`):
//!    - `rol`/`ror` -> `@llvm.fshl`/`@llvm.fshr`, dead-flag dynamic-count mask elision, full-width
//!      constant rotate elision (`rol al, 8`/`ror ax, 16`) with exact CF/OF, count-0 flag
//!      preservation + 32-bit zero-extension, and constant-operand folding for `rol`/`ror`/`rorx`/
//!      `shld`/`shrd`/`andn(r, r)`.
//!    - `bsf`/`bsr`/`tzcnt`/`lzcnt`/`popcnt`/`bswap` -> `@llvm.cttz`/`@llvm.ctlz`/`@llvm.ctpop`/
//!      `@llvm.bswap` with `bsf`/`bsr` zero-input full-register preservation, `nonzero_bits`
//!      propagation on count intrinsics, and read-only dynamic `bt reg, reg` `lshr+trunc` lowering.
//! 2. Sound intra-block `cmp`/`test`/bitop + `jcc`/`setcc`/`cmovcc`/`adc`/`sbb` flag fusion:
//!    - Direct `icmp` without redundant `and r, r` / `sub r, 0` / flag-alloca stores when flags are
//!      dead, `test r, SIGN_BIT` -> `icmp slt/sge r, 0`, `test r, 1; jne` -> `trunc r to i1`,
//!      same-reg `cmp`/`sub`/`xor` constant fusion, `rol`/`ror`/`shl`/`shr`/`shld`/`shrd` carry
//!      fusion, `popcnt`/`lzcnt`/`blsi`/`blsr` extended fusion, 32-bit `cmovcc` upper-32 zeroing
//!      even when false, and exact live architectural flag + `%state` store-backs.
//! 3. Redundant `lea` / address-calc cleanup:
//!    - Identity 64-bit `lea r64, [r64]` no-op, 32-bit `lea r32, [r32 + 0]` zero-extension,
//!      single final mask for 32-bit addressing, affine offset cancellation, and nested `shl` fold.
//! 4. Tool-gated Wasm64 execution and x86-64 host-CPU differential verification.

use iced_x86::code_asm::*;

use super::exec::{
    CODE_BASE, Case, Outcome, R8, R9, R10, R11, RAX, RBX, RCX, RDI, RDX, RSI, RSP, STACK_TOP,
    build, exec_or_skip, run,
};
use super::{Program, asm_function};
use crate::lift::lift_function;

fn lift_ir(build_fn: impl FnOnce(&mut CodeAssembler) -> Result<(), iced_x86::IcedError>) -> String {
    let f = asm_function(0x1000, build_fn);
    let lifted = lift_function(&f).expect("lifts");
    assert!(lifted.unsupported.is_empty(), "{:?}", lifted.unsupported);
    lifted.ir
}

fn block_ir<'a>(ir: &'a str, label: &str) -> &'a str {
    let header = format!("{label}:\n");
    let start = ir
        .find(&header)
        .unwrap_or_else(|| panic!("missing block {label} in:\n{ir}"))
        + header.len();
    let rest = &ir[start..];
    let end = rest
        .lines()
        .scan(0usize, |pos, line| {
            let line_start = *pos;
            *pos += line.len() + 1;
            Some((line_start, line))
        })
        .find(|(_, line)| !line.starts_with(' ') && !line.starts_with('\t'))
        .map_or(rest.len(), |(pos, _)| pos);
    &rest[..end]
}

fn with_flags(mut c: Case, flags: [bool; 6]) -> Case {
    c.flags = flags;
    c
}

fn ok(outcomes: &[Outcome], what: &str) {
    for (i, o) in outcomes.iter().enumerate() {
        assert!(o.error.is_empty(), "{what}: case {i} trapped: {}", o.error);
        assert_eq!(o.regs[RSP], STACK_TOP, "{what}: case {i}: rsp");
    }
}

// ================================================================================================
// 1. IR-shape tests for native bit-manipulation LLVM intrinsics & constant folding
// ================================================================================================

#[test]
fn bitops_native_llvm_intrinsics_and_folding_ir() {
    // 1. `rol`/`ror` with constant and dynamic `cl` counts lower directly to `@llvm.fshl`/`@llvm.fshr`.
    //    When flags are overwritten right after (`xor edx, edx`), dynamic `cl` needs NO `and` count mask.
    let rot_dead_flags = lift_ir(|a| {
        a.rol(rax, 7)?;
        a.ror(r8d, 13)?;
        a.rol(r9, cl)?;
        a.ror(r10d, cl)?;
        a.xor(edx, edx)?; // kills CF/OF before ret
        a.ret()
    });
    assert!(
        rot_dead_flags.contains("call i64 @llvm.fshl.i64(")
            && rot_dead_flags.contains("call i32 @llvm.fshr.i32(")
            && rot_dead_flags.contains("call i64 @llvm.fshl.i64(i64 ")
            && rot_dead_flags.contains("call i32 @llvm.fshr.i32(i32 "),
        "rol/ror must emit @llvm.fshl/@llvm.fshr:\n{rot_dead_flags}"
    );
    assert!(
        !rot_dead_flags.contains("and i64 %") && !rot_dead_flags.contains(", 63\n"),
        "dynamic rol/ror with dead CF/OF should omit redundant count mask:\n{rot_dead_flags}"
    );

    // 2. Dynamic `rol rax, cl` with LIVE CF/OF at `ret` MUST mask `cl & 63` and guard flag updates
    //    on `count != 0` so `cl == 0` or `cl == 64` leaves flags untouched!
    let rot_live_flags = lift_ir(|a| {
        a.rol(rax, cl)?;
        a.ret()
    });
    assert!(
        rot_live_flags.contains("and i64 ")
            && rot_live_flags.contains(", 63")
            && rot_live_flags.contains("icmp ne i64 ")
            && rot_live_flags.contains("select i1 ")
            && rot_live_flags.contains("ptr %state.flag0")
            && rot_live_flags.contains("ptr %state.flag5"),
        "dynamic rol with live flags must guard CF/OF on (cl & 63) != 0 and store to %state:\n{rot_live_flags}"
    );

    // 3. Full-width immediate rotate (`rol al, 8`, `ror ax, 16`) skips `@llvm.fshl`/`@llvm.fshr`,
    //    while `rol eax, 0` leaves flags untouched and preserves 32-bit zero-extension!
    let rot_special = lift_ir(|a| {
        a.rol(al, 8)?;
        a.ror(dx, 16)?;
        a.rol(r8d, 0)?;
        a.ret()
    });
    assert!(
        !rot_special.contains("@llvm.fshl.") && !rot_special.contains("@llvm.fshr."),
        "full-width and zero-count immediate rotates should not call fshl/fshr:\n{rot_special}"
    );
    assert!(
        rot_special.contains("zext i32 ") || rot_special.contains("4294967295"),
        "rol r32, 0 must still zero-extend upper 32 bits of r8:\n{rot_special}"
    );

    // 4. `bsf`/`bsr`/`tzcnt`/`lzcnt`/`popcnt`/`bswap` emit native LLVM intrinsics.
    //    `bsf`/`bsr` preserve full 64-bit destination via `select i1 %zero, i64 %old, i64 %wide`.
    let bit_intrinsics = lift_ir(|a| {
        a.bsf(eax, ecx)?;
        a.bsr(rdx, r8)?;
        a.tzcnt(r9d, r10d)?;
        a.lzcnt(r11, rsi)?;
        a.popcnt(rdi, rbx)?;
        a.bswap(rsi)?;
        a.ret()
    });
    assert!(
        bit_intrinsics.contains("call i32 @llvm.cttz.i32(i32 ")
            && bit_intrinsics.contains("call i64 @llvm.ctlz.i64(i64 ")
            && bit_intrinsics.contains("call i64 @llvm.ctpop.i64(i64 ")
            && bit_intrinsics.contains("call i64 @llvm.bswap.i64(i64 "),
        "bitops must emit native cttz/ctlz/ctpop/bswap intrinsics:\n{bit_intrinsics}"
    );
    assert!(
        bit_intrinsics.contains("select i1 "),
        "bsf/bsr must preserve destination on zero input via select:\n{bit_intrinsics}"
    );

    // 5. Known-zero-bits propagation from `@llvm.ctpop`/`@llvm.cttz`/`@llvm.ctlz`:
    //    - `popcnt eax, ecx` produces `0..=32` (`nonzero_bits = 0x3F`):
    //      `and eax, 63` folds away, and `cmp eax, 100; sete dl` folds directly to `0`!
    let count_nz = lift_ir(|a| {
        a.popcnt(eax, ecx)?;
        a.and(eax, 63)?;
        a.cmp(eax, 100)?;
        a.sete(dl)?;
        a.sub(r8d, r8d)?; // kills all 6 flags (including AF) before ret
        a.ret()
    });
    assert!(
        !count_nz.contains("and i32 ") && !count_nz.contains("icmp eq i32 "),
        "popcnt nonzero_bits (0x3F) should fold redundant mask and impossible cmp == 100:\n{count_nz}"
    );

    // 6. Constant folding for `rol`, `ror`, `rorx`, `shld`, `shrd`, and same-reg `andn`.
    let const_bitops = lift_ir(|a| {
        a.mov(eax, 0x1234_5678u32 as i32)?;
        a.rol(eax, 8)?;
        a.ror(eax, 4)?;
        a.rorx(ecx, eax, 12)?;
        a.mov(edx, 0xABCD_0001u32 as i32)?;
        a.shld(eax, edx, 16)?;
        a.shrd(ecx, edx, 8)?;
        a.andn(r8d, r9d, r9d)?;
        a.add(eax, ecx)?;
        a.add(eax, r8d)?;
        a.ret()
    });
    assert!(
        !const_bitops.contains("@llvm.fshl.") && !const_bitops.contains("@llvm.fshr."),
        "constant rol/ror/rorx/shld/shrd should fold without fshl/fshr calls:\n{const_bitops}"
    );

    // 7. Read-only dynamic `bt rax, rcx` emits concise `and` + `lshr` + `trunc` (no `shl 1, idx`).
    let bt_dyn = lift_ir(|a| {
        a.bt(rax, rcx)?;
        a.setb(dl)?;
        a.xor(eax, eax)?;
        a.ret()
    });
    assert!(
        bt_dyn.contains("lshr i64 ")
            && bt_dyn.contains("trunc i64 ")
            && !bt_dyn.contains("shl i64 1, "),
        "read-only dynamic bt reg, reg should use lshr+trunc instead of shl 1:\n{bt_dyn}"
    );
}

// ================================================================================================
// 2. IR-shape tests for intra-block cmp/test/bitop flag fusion & cmovcc rules
// ================================================================================================

#[test]
fn intra_block_cmp_test_and_bitop_flag_fusion_ir() {
    // 1. `test rax, rax; jle` emits a single `icmp sle i64 %t, 0` in bb_1000 with NO `and` and NO flag stores!
    let test_jle = lift_ir(|a| {
        let mut taken = a.create_label();
        a.test(rax, rax)?;
        a.jle(taken)?;
        a.xor(eax, eax)?; // kills flags on both paths
        a.ret()?;
        a.set_label(&mut taken)?;
        a.mov(eax, 1)?;
        a.xor(ecx, ecx)?;
        a.ret()
    });
    let test_jle_bb = block_ir(&test_jle, "bb_1000");
    assert!(
        test_jle_bb.contains("icmp sle i64 ")
            && !test_jle_bb.contains("and i64 ")
            && !test_jle_bb.contains("store i1 "),
        "test r, r; jle must fuse into single icmp sle with zero `and` or flag stores in bb_1000:\n{test_jle}"
    );

    // 2. `test eax, 0x80000000; jne` folds `and + icmp ne` directly to `icmp slt i32 ..., 0`,
    //    and `test ecx, 1; setne dl` folds to `trunc i32 ... to i1`.
    let test_bit_masks = lift_ir(|a| {
        let mut taken = a.create_label();
        a.test(ecx, 1)?;
        a.setne(dl)?;
        a.test(eax, 0x8000_0000u32 as i32)?;
        a.jne(taken)?;
        a.xor(eax, eax)?;
        a.ret()?;
        a.set_label(&mut taken)?;
        a.mov(eax, 1)?;
        a.xor(ecx, ecx)?;
        a.ret()
    });
    let test_bit_bb = block_ir(&test_bit_masks, "bb_1000");
    assert!(
        test_bit_bb.contains("icmp slt i32 ")
            && test_bit_bb.contains("trunc i32 ")
            && !test_bit_bb.contains("-2147483648")
            && !test_bit_bb.contains("store i1 "),
        "test SIGN_BIT and test 1 must fold directly without masking constants or flag stores:\n{test_bit_masks}"
    );

    // 3. `cmp rax, 0; jg` and `cmp rcx, 42; setl / cmovge`: zero `sub`, zero flag stores in bb_1000!
    //    (`sub` on both successors kills all 6 flags including AF; `xor` leaves AF live.)
    let cmp_fused = lift_ir(|a| {
        let mut taken = a.create_label();
        a.cmp(rcx, 42)?;
        a.setl(r8b)?;
        a.cmovge(rdx, r9)?;
        a.cmp(rax, 0)?;
        a.jg(taken)?;
        a.sub(eax, eax)?;
        a.ret()?;
        a.set_label(&mut taken)?;
        a.mov(eax, 1)?;
        a.sub(ecx, ecx)?;
        a.ret()
    });
    let cmp_fused_bb = block_ir(&cmp_fused, "bb_1000");
    assert!(
        cmp_fused_bb.contains("icmp slt i64 ")
            && cmp_fused_bb.contains("icmp sge i64 ")
            && cmp_fused_bb.contains("icmp sgt i64 ")
            && !cmp_fused_bb.contains("sub i64 ")
            && !cmp_fused_bb.contains("store i1 "),
        "cmp+setcc/cmovcc/jcc must fuse into direct icmps with zero sub or flag stores in bb_1000:\n{cmp_fused}"
    );

    // 4. `cmp rax, rdx; js` (even with `b != 0`) fuses with zero `%sf` alloca store in bb_1000!
    let cmp_js = lift_ir(|a| {
        let mut taken = a.create_label();
        a.cmp(rax, rdx)?;
        a.js(taken)?;
        a.sub(eax, eax)?;
        a.ret()?;
        a.set_label(&mut taken)?;
        a.mov(eax, 1)?;
        a.sub(ecx, ecx)?;
        a.ret()
    });
    let cmp_js_bb = block_ir(&cmp_js, "bb_1000");
    assert!(
        cmp_js_bb.contains("icmp slt i64 ") && !cmp_js_bb.contains("store i1 "),
        "cmp r, r; js must fuse without storing %sf in bb_1000:\n{cmp_js}"
    );

    // 5. Same-reg `cmp rax, rax; sete dl` and `xor eax, eax; sbb rcx, rdx` fold flags to constants
    //    with zero `%zf`/`%cf` flag loads (and zero `icmp` for cmp/sete)!
    let same_reg_const = lift_ir(|a| {
        a.cmp(rax, rax)?;
        a.sete(dl)?;
        a.xor(eax, eax)?;
        a.sbb(rcx, rdx)?;
        a.sub(r8d, r8d)?; // kills all 6 flags (including AF) before ret
        a.ret()
    });
    let same_reg_bb = block_ir(&same_reg_const, "bb_1000");
    assert!(
        !same_reg_bb.contains("load i1 ") && !same_reg_bb.contains("icmp "),
        "same-reg cmp/xor feeding setcc/sbb must fold without flag loads or icmps:\n{same_reg_const}"
    );

    // 6. Immediate `rol`/`ror`/`shr`/`shl` carry fusion (`rol r, 1; jc`, `shr r, 1; adc`):
    //    zero `%cf` alloca store in bb_1000 when flags are dead!
    let shift_carry_fusion = lift_ir(|a| {
        let mut taken = a.create_label();
        a.shr(rcx, 1)?;
        a.adc(rdx, 0)?;
        a.ror(rax, 1)?;
        a.jb(taken)?;
        a.sub(eax, eax)?;
        a.ret()?;
        a.set_label(&mut taken)?;
        a.mov(eax, 1)?;
        a.sub(ecx, ecx)?;
        a.ret()
    });
    let shift_carry_bb = block_ir(&shift_carry_fusion, "bb_1000");
    assert!(
        shift_carry_bb.contains("@llvm.fshr.i64(")
            && !shift_carry_bb.contains("store i1 ")
            && !shift_carry_bb.contains("load i1 "),
        "imm shr+adc and ror+jb must fuse CF directly without %cf alloca traffic in bb_1000:\n{shift_carry_fusion}"
    );

    // 7. Extended bitop flag fusion: `popcnt + ja`, `lzcnt + jbe`, `blsi + jb`, `blsr + jle`.
    let bitop_fusion = lift_ir(|a| {
        let mut l1 = a.create_label();
        a.popcnt(r8, rcx)?;
        a.seta(dl)?;
        a.blsi(r9, rcx)?;
        a.setb(r10b)?;
        a.blsr(r11, rcx)?;
        a.setle(r11b)?;
        a.lzcnt(rax, rcx)?;
        a.jbe(l1)?;
        a.sub(eax, eax)?;
        a.ret()?;
        a.set_label(&mut l1)?;
        a.mov(eax, 1)?;
        a.sub(ecx, ecx)?;
        a.ret()
    });
    let bitop_fusion_bb = block_ir(&bitop_fusion, "bb_1000");
    assert!(
        bitop_fusion_bb.contains("icmp sle i64 ")
            && !bitop_fusion_bb.contains("store i1 ")
            && !bitop_fusion_bb.contains("load i1 "),
        "popcnt/blsi/blsr/lzcnt must fuse directly with zero flag alloca traffic in bb_1000:\n{bitop_fusion}"
    );

    // 8. Soundness: 32-bit `cmovcc r32, r32` (both distinct and same-reg `cmove eax, eax`)
    //    MUST zero-extend upper 32 bits of entry registers even when condition is false!
    let cmov32_same = lift_ir(|a| {
        a.cmove(eax, eax)?;
        a.ret()
    });
    assert!(
        cmov32_same.contains("4294967295") || cmov32_same.contains("zext i32 "),
        "same-reg 32-bit cmovcc MUST still zero upper 32 bits:\n{cmov32_same}"
    );
    let cmov32_distinct = lift_ir(|a| {
        a.cmp(ecx, edx)?;
        a.cmove(eax, r8d)?;
        a.ret()
    });
    assert!(
        cmov32_distinct.contains("4294967295") || cmov32_distinct.contains("zext i32 "),
        "32-bit cmovcc on unknown-upper-bits regs MUST zero upper 32 bits:\n{cmov32_distinct}"
    );

    // 9. Soundness (LIFTER_STATE_CONTRACT): live flags at `ret` after fused `cmp`/`test`/`setcc`
    //    MUST still be materialized and stored back to `%state.flag*`!
    let live_flags_exit = lift_ir(|a| {
        a.cmp(rax, rcx)?;
        a.setl(dl)?;
        a.ret()
    });
    for f in 0..6 {
        assert!(
            live_flags_exit.contains(&format!("ptr %state.flag{f}")),
            "live flag {f} after cmp+setcc must still store back to %state.flag{f}:\n{live_flags_exit}"
        );
    }
    assert!(
        live_flags_exit.contains("ptr %state.gpr2"),
        "written rdx must still store back to %state.gpr2:\n{live_flags_exit}"
    );
}

// ================================================================================================
// 3. IR-shape tests for redundant `lea` / address-calc cleanup
// ================================================================================================

#[test]
fn lea_and_address_calc_cleanup_ir() {
    // 1. Identity 64-bit `lea rax, [rax]` is a pure no-op (zero arithmetic or %rax alloca store in bb_1000).
    let id64 = lift_ir(|a| {
        a.lea(rax, qword_ptr(rax))?;
        a.ret()
    });
    let id64_bb = block_ir(&id64, "bb_1000");
    assert!(
        id64_bb.matches("add i64 ").count() == 1 // only `rsp + 8` from `ret`
            && !id64_bb.contains("shl i64 ")
            && !id64_bb
                .lines()
                .any(|l| l.trim().starts_with("store ") && l.trim().ends_with(", ptr %rax")),
        "identity 64-bit lea rax, [rax] should emit no arithmetic or redundant %rax store in bb_1000:\n{id64}"
    );

    // 2. Identity 32-bit `lea eax, [rax + 0]` on an unknown entry reg MUST still zero upper 32 bits,
    //    while `mov eax, [rcx]; lea eax, [rax + 0]` elides the mask (already known 32-bit)!
    let id32_entry = lift_ir(|a| {
        a.lea(eax, dword_ptr(rax))?;
        a.ret()
    });
    assert!(
        id32_entry.contains("4294967295"),
        "32-bit identity lea on entry reg MUST zero upper 32 bits:\n{id32_entry}"
    );
    let id32_known = lift_ir(|a| {
        a.mov(eax, dword_ptr(rcx))?;
        a.lea(eax, dword_ptr(rax))?;
        a.ret()
    });
    assert_eq!(
        id32_known.matches("4294967295").count(),
        0,
        "32-bit lea on already zero-extended reg needs no extra mask:\n{id32_known}"
    );

    // 3. 32-bit-addressed `lea eax, [ecx + edx*4 + 16]` emits at most ONE 32-bit mask (not 3).
    let addr32_lea = lift_ir(|a| {
        a.lea(eax, dword_ptr(ecx + edx * 4 + 16))?;
        a.ret()
    });
    assert_eq!(
        addr32_lea.matches("4294967295").count(),
        1,
        "32-bit-addressed lea should emit only a single final 32-bit mask:\n{addr32_lea}"
    );

    // 4. Affine offset cancellation across `lea` chain:
    //    `lea rcx, [rsp + 0x20]; lea rax, [rcx + rdx*4 - 0x20]` -> cancelling offsets disappear!
    let affine_cancel = lift_ir(|a| {
        a.lea(rcx, qword_ptr(rsp + 0x20))?;
        a.lea(rax, qword_ptr(rcx + rdx * 4 - 0x20))?;
        a.ret()
    });
    let affine_bb = block_ir(&affine_cancel, "bb_1000");
    assert!(
        !affine_bb.contains("-32") && affine_bb.matches("add i64 ").count() == 3,
        "cancelling affine offsets in lea should fold cleanly (only rcx=rsp+32, rax=rsp+rdx*4, ret rsp+8):\n{affine_cancel}"
    );

    // 5. `lea rax, [rcx + rcx]` emits `shl i64 ..., 1` and chains with `lea rax, [rax*4]`
    //    into a single `shl i64 ..., 3`!
    let double_scale = lift_ir(|a| {
        a.lea(rax, qword_ptr(rcx + rcx))?;
        a.lea(rax, qword_ptr(rax * 4))?;
        a.ret()
    });
    assert!(
        double_scale.contains("shl i64 ") && double_scale.contains(", 3\n"),
        "lea [rcx+rcx] followed by lea [rax*4] should fold to shl i64 ..., 3:\n{double_scale}"
    );
}

// ================================================================================================
// 4. Tool-gated Wasm64 execution tests
// ================================================================================================

#[test]
fn bitops_and_rotate_edge_cases_execute_on_wasm64() {
    if !exec_or_skip("bitops_and_rotate_edge_cases_execute_on_wasm64") {
        return;
    }

    // F1: bsf/bsr 32-bit zero-input preserves FULL 64-bit destination (including upper 32 bits!),
    //     non-zero input zero-extends to 64-bit and clears ZF.
    let f_bsf_bsr = asm_function(CODE_BASE, |a| {
        a.bsf(eax, ecx)?;
        a.setz(r8b)?;
        a.bsr(edx, r9d)?;
        a.setz(r10b)?;
        a.ret()
    });

    // F2: rotate count 0 / 1 / 8 / 31 / 32 / 63 / 64 via CL (32-bit & 64-bit) with live CF/OF!
    let f_rol32_cl = asm_function(CODE_BASE + 0x100, |a| {
        a.rol(eax, cl)?;
        a.ret()
    });
    let f_ror64_cl = asm_function(CODE_BASE + 0x200, |a| {
        a.ror(rax, cl)?;
        a.ret()
    });

    // F3: full-width 8-bit `rol al, 8` and 16-bit `ror ax, 16` + count-0 `rol eax, 0`
    let f_rot_imm = asm_function(CODE_BASE + 0x300, |a| {
        a.rol(eax, 0)?; // leaves CF/OF untouched, zeroes upper 32 bits of rax!
        a.rol(dl, 8)?; // value unchanged, CF = bit 0 of dl
        a.setb(r8b)?;
        a.ror(r9w, 16)?; // value unchanged, CF = bit 15 of r9w
        a.setb(r10b)?;
        a.ret()
    });

    // F4: fused carry from immediate `rol`/`ror`/`shl`/`shr`/`bt` into `adc`/`sbb`/`setc`
    let f_shift_carry = asm_function(CODE_BASE + 0x400, |a| {
        a.shr(rcx, 1)?;
        a.adc(rax, 0)?;
        a.rol(rdx, 1)?;
        a.adc(rax, 0)?;
        a.ror(r8, 1)?;
        a.adc(rax, 0)?;
        a.bt(r9, r10)?;
        a.adc(rax, 0)?;
        a.ret()
    });

    // F5: tzcnt/lzcnt/popcnt on 0, 1, INT_MIN, and fused flags
    let f_counts = asm_function(CODE_BASE + 0x500, |a| {
        a.tzcnt(r8, rcx)?;
        a.setb(al)?; // CF = (rcx == 0)
        a.sete(ah)?; // ZF = (lsb == 1)
        a.lzcnt(r9, rdx)?;
        a.setbe(bl)?; // CF|ZF = (rdx <=s 0)
        a.popcnt(r10, rsi)?;
        a.seta(r11b)?; // !CF & !ZF = (rsi != 0)
        a.ret()
    });

    let built = build(
        "hyper_bitops_exec",
        [
            &f_bsf_bsr,
            &f_rol32_cl,
            &f_ror64_cl,
            &f_rot_imm,
            &f_shift_carry,
            &f_counts,
        ],
    );

    let cases = vec![
        // 0: bsf(0) preserves full 64-bit rax, bsr(0) preserves full 64-bit rdx, both set ZF=1
        Case::new(CODE_BASE)
            .reg(RAX, 0xDEAD_BEEF_1234_5678)
            .reg(RCX, 0)
            .reg(RDX, 0xCAFE_BABE_8765_4321)
            .reg(R9, 0)
            .reg(R8, 0)
            .reg(R10, 0),
        // 1: bsf(0x8000_0000) -> 31 (zero-extended!), bsr(0x8000_0000) -> 31, ZF=0
        Case::new(CODE_BASE)
            .reg(RAX, 0xDEAD_BEEF_1234_5678)
            .reg(RCX, 0x8000_0000)
            .reg(RDX, 0xCAFE_BABE_8765_4321)
            .reg(R9, 0x8000_0000)
            .reg(R8, 0)
            .reg(R10, 0),
        // 2: rol32 with cl=32 (masked 0!): upper 32 of rax zeroed, incoming CF=true/OF=true UNTOUCHED!
        with_flags(
            Case::new(CODE_BASE + 0x100)
                .reg(RAX, 0xFFFF_FFFF_8000_0001)
                .reg(RCX, 32),
            [true, false, false, false, false, true],
        ),
        // 3: rol32 with cl=1: 0x8000_0001 rol 1 = 0x0000_0003, CF=1, OF=msb(r)^CF=0^1=1
        with_flags(
            Case::new(CODE_BASE + 0x100)
                .reg(RAX, 0xFFFF_FFFF_8000_0001)
                .reg(RCX, 1),
            [false; 6],
        ),
        // 4: ror64 with cl=64 (masked 0!): value unchanged, incoming CF=true/OF=false UNTOUCHED!
        with_flags(
            Case::new(CODE_BASE + 0x200)
                .reg(RAX, 0x0123_4567_89AB_CDEF)
                .reg(RCX, 64),
            [true, false, false, false, false, false],
        ),
        // 5: ror64 with cl=1 on INT_MIN (0x8000..0): result 0x4000..0, CF=0, OF=0^1=1
        with_flags(
            Case::new(CODE_BASE + 0x200)
                .reg(RAX, 1u64 << 63)
                .reg(RCX, 1),
            [true, false, false, false, false, false],
        ),
        // 6: f_rot_imm: rol eax, 0 zeroes upper 32; rol dl, 8 sets CF=lsb(dl)=1; ror r9w, 16 sets CF=msb(r9w)=1
        Case::new(CODE_BASE + 0x300)
            .reg(RAX, 0xAAAA_BBBB_1122_3344)
            .reg(RDX, 0x55)
            .reg(R9, 0x8001)
            .reg(R8, 0)
            .reg(R10, 0),
        // 7: f_shift_carry: all 4 carries set -> rax = 4
        Case::new(CODE_BASE + 0x400)
            .reg(RAX, 0)
            .reg(RCX, 1) // shr 1 -> CF=1
            .reg(RDX, 1u64 << 63) // rol 1 -> CF=1
            .reg(R8, 1) // ror 1 -> CF=1
            .reg(R9, 1u64 << 42) // bt bit 42 (mod 64) -> CF=1
            .reg(R10, 42 + 64), // verifies modulo-64 masking!
        // 8: f_counts: tzcnt(0)=64 CF=1 ZF=0; lzcnt(INT_MIN)=0 CF|ZF=1; popcnt(0)=0 !CF&!ZF=0
        Case::new(CODE_BASE + 0x500)
            .reg(RAX, 0)
            .reg(RBX, 0)
            .reg(RCX, 0)
            .reg(RDX, 1u64 << 63)
            .reg(RSI, 0)
            .reg(R11, 0),
        // 9: f_counts: tzcnt(1)=0 CF=0 ZF=1; lzcnt(5)=61 CF|ZF=0; popcnt(0xF0)=4 !CF&!ZF=1
        Case::new(CODE_BASE + 0x500)
            .reg(RAX, 0)
            .reg(RBX, 0)
            .reg(RCX, 1)
            .reg(RDX, 5)
            .reg(RSI, 0xF0)
            .reg(R11, 0),
    ];

    let o = run(&built, &cases);
    ok(&o, "hyper_bitops_exec");

    assert_eq!(o[0].regs[RAX], 0xDEAD_BEEF_1234_5678, "bsf(0) preserves rax");
    assert_eq!(o[0].regs[RDX], 0xCAFE_BABE_8765_4321, "bsr(0) preserves rdx");
    assert_eq!(o[0].regs[R8] & 0xFF, 1, "bsf(0) ZF=1");
    assert_eq!(o[0].regs[R10] & 0xFF, 1, "bsr(0) ZF=1");

    assert_eq!(o[1].regs[RAX], 31, "bsf(0x8000_0000) zero-extends");
    assert_eq!(o[1].regs[RDX], 31, "bsr(0x8000_0000) zero-extends");
    assert_eq!(o[1].regs[R8] & 0xFF, 0, "bsf nonzero ZF=0");
    assert_eq!(o[1].regs[R10] & 0xFF, 0, "bsr nonzero ZF=0");

    assert_eq!(o[2].regs[RAX], 0x8000_0001, "rol32 cl=32 zeroes upper 32");
    assert!(o[2].flags[0] && o[2].flags[5], "rol32 cl=32 leaves CF/OF untouched");

    assert_eq!(o[3].regs[RAX], 0x0000_0003, "rol32 cl=1 value");
    assert!(o[3].flags[0] && o[3].flags[5], "rol32 cl=1 CF=1 OF=1");

    assert_eq!(o[4].regs[RAX], 0x0123_4567_89AB_CDEF, "ror64 cl=64 value");
    assert!(o[4].flags[0] && !o[4].flags[5], "ror64 cl=64 leaves CF/OF untouched");

    assert_eq!(o[5].regs[RAX], 1u64 << 62, "ror64 cl=1 INT_MIN");
    assert!(!o[5].flags[0] && o[5].flags[5], "ror64 cl=1 INT_MIN CF=0 OF=1");

    assert_eq!(o[6].regs[RAX], 0x1122_3344, "rol eax, 0 zeroes upper 32");
    assert_eq!(o[6].regs[R8] & 0xFF, 1, "rol dl, 8 sets CF=lsb");
    assert_eq!(o[6].regs[R10] & 0xFF, 1, "ror r9w, 16 sets CF=msb");

    assert_eq!(o[7].regs[RAX], 4, "fused shift/rotate/bt carries");

    assert_eq!(o[8].regs[R8], 64, "tzcnt(0)");
    assert_eq!(o[8].regs[RAX] & 0xFFFF, 0x0001, "tzcnt(0) CF=1 ZF=0");
    assert_eq!(o[8].regs[R9], 0, "lzcnt(INT_MIN)");
    assert_eq!(o[8].regs[RBX] & 0xFF, 1, "lzcnt(INT_MIN) CF|ZF=1");
    assert_eq!(o[8].regs[R10], 0, "popcnt(0)");
    assert_eq!(o[8].regs[R11] & 0xFF, 0, "popcnt(0) !CF&!ZF=0");

    assert_eq!(o[9].regs[R8], 0, "tzcnt(1)");
    assert_eq!(o[9].regs[RAX] & 0xFFFF, 0x0100, "tzcnt(1) CF=0 ZF=1");
    assert_eq!(o[9].regs[R9], 61, "lzcnt(5)");
    assert_eq!(o[9].regs[RBX] & 0xFF, 0, "lzcnt(5) CF|ZF=0");
    assert_eq!(o[9].regs[R10], 4, "popcnt(0xF0)");
    assert_eq!(o[9].regs[R11] & 0xFF, 1, "popcnt(0xF0) !CF&!ZF=1");
}

#[test]
fn flag_fusion_cmov_and_lea_edge_cases_execute_on_wasm64() {
    if !exec_or_skip("flag_fusion_cmov_and_lea_edge_cases_execute_on_wasm64") {
        return;
    }

    // F1: Signed overflow cmp (`INT_MIN` vs `1` / `INT_MAX` vs `-1`):
    //     packs [jl, jge, jle, jg, js, jns, jb, jae] into rax AND keeps cmp flags live at `ret`!
    let f_cmp_overflow = asm_function(CODE_BASE, |a| {
        a.xor(eax, eax)?;
        a.cmp(rcx, rdx)?;
        a.setl(r8b)?;
        a.setge(r9b)?;
        a.setle(r10b)?;
        a.setg(r11b)?;
        a.sets(bl)?;
        a.setb(sil)?;
        a.ret() // cmp flags remain LIVE at exit!
    });

    // F2: 32-bit `cmovcc` upper-32 zeroing when FALSE (distinct & same-reg) + 64-bit cmovcc!
    let f_cmov = asm_function(CODE_BASE + 0x100, |a| {
        a.cmp(rcx, rdx)?;
        a.cmove(eax, r8d)?; // when rcx != rdx, keeps low 32 of eax BUT zeroes upper 32!
        a.cmove(r9d, r9d)?; // same-reg 32-bit ALWAYS zeroes upper 32!
        a.cmove(r10, r11)?; // 64-bit preserves full 64-bit r10 when false!
        a.ret()
    });

    // F3: `test r, SIGN_BIT` & `test r, 1` & `test r, r` + `lea` edge cases
    let f_test_lea = asm_function(CODE_BASE + 0x200, |a| {
        a.test(ecx, 0x8000_0000u32 as i32)?;
        a.setne(al)?;
        a.test(ecx, 1)?;
        a.setne(ah)?;
        a.lea(r8, qword_ptr(r8))?; // 64-bit identity: preserves full 64-bit r8
        a.lea(r9d, dword_ptr(r9))?; // 32-bit identity: zeroes upper 32 bits of r9
        a.lea(r10d, dword_ptr(ecx + edx * 4 + 16))?; // 32-bit addressing wrap-around
        a.lea(r11, qword_ptr(rsi + rsi))?;
        a.lea(r11, qword_ptr(r11 * 4))?; // r11 = rsi * 8
        a.ret()
    });

    let built = build(
        "hyper_fusion_lea_exec",
        [&f_cmp_overflow, &f_cmov, &f_test_lea],
    );

    let cases = vec![
        // 0: cmp(i64::MIN, 1): diff = i64::MAX > 0 (SF=0, OF=1!), so jl=1, jge=0, jle=1, jg=0, js=0, jb=0
        Case::new(CODE_BASE)
            .reg(RCX, i64::MIN as u64)
            .reg(RDX, 1)
            .reg(R8, 0)
            .reg(R9, 0)
            .reg(R10, 0)
            .reg(R11, 0)
            .reg(RBX, 0)
            .reg(RSI, 0),
        // 1: cmp(i64::MAX, -1): diff = i64::MIN < 0 (SF=1, OF=1!), so jl=0, jge=1, jle=0, jg=1, js=1, jb=1
        Case::new(CODE_BASE)
            .reg(RCX, i64::MAX as u64)
            .reg(RDX, u64::MAX)
            .reg(R8, 0)
            .reg(R9, 0)
            .reg(R10, 0)
            .reg(R11, 0)
            .reg(RBX, 0)
            .reg(RSI, 0),
        // 2: cmov with condition FALSE (rcx=1 != rdx=2)
        Case::new(CODE_BASE + 0x100)
            .reg(RCX, 1)
            .reg(RDX, 2)
            .reg(RAX, 0xDEAD_BEEF_1122_3344)
            .reg(R8, 0xAAAA_BBBB_5566_7788)
            .reg(R9, 0xCAFE_BABE_99AA_BBCC)
            .reg(R10, 0x1234_5678_9ABC_DEF0)
            .reg(R11, 0x0FED_CBA9_8765_4321),
        // 3: cmov with condition TRUE (rcx=7 == rdx=7)
        Case::new(CODE_BASE + 0x100)
            .reg(RCX, 7)
            .reg(RDX, 7)
            .reg(RAX, 0xDEAD_BEEF_1122_3344)
            .reg(R8, 0xAAAA_BBBB_5566_7788)
            .reg(R9, 0xCAFE_BABE_99AA_BBCC)
            .reg(R10, 0x1234_5678_9ABC_DEF0)
            .reg(R11, 0x0FED_CBA9_8765_4321),
        // 4: test SIGN_BIT & bit 0 + lea identities & 32-bit wrap-around with dirty upper 32 bits
        Case::new(CODE_BASE + 0x200)
            .reg(RAX, 0)
            .reg(RCX, 0xFFFF_FFFF_8000_0001)
            .reg(RDX, 0xDEAD_BEEF_4000_0000) // (0x8000_0001 + 0x4000_0000*4 + 16) mod 2^32 = 0x8000_0011
            .reg(R8, 0x1111_2222_3333_4444)
            .reg(R9, 0x5555_6666_7777_8888)
            .reg(RSI, 0x1234_5678_9ABC_DEF1),
    ];

    let o = run(&built, &cases);
    ok(&o, "hyper_fusion_lea_exec");

    // Case 0: i64::MIN vs 1
    assert_eq!(o[0].regs[R8] & 0xFF, 1, "INT_MIN <s 1");
    assert_eq!(o[0].regs[R9] & 0xFF, 0, "INT_MIN >=s 1");
    assert_eq!(o[0].regs[R10] & 0xFF, 1, "INT_MIN <=s 1");
    assert_eq!(o[0].regs[R11] & 0xFF, 0, "INT_MIN >s 1");
    assert_eq!(o[0].regs[RBX] & 0xFF, 0, "SF(INT_MIN - 1) == 0");
    assert_eq!(o[0].regs[RSI] & 0xFF, 0, "INT_MIN <u 1 is false");
    assert!(
        !o[0].flags[0] && !o[0].flags[3] && !o[0].flags[4] && o[0].flags[5],
        "live cmp flags at exit: CF=0 ZF=0 SF=0 OF=1"
    );

    // Case 1: i64::MAX vs -1
    assert_eq!(o[1].regs[R8] & 0xFF, 0, "INT_MAX <s -1");
    assert_eq!(o[1].regs[R9] & 0xFF, 1, "INT_MAX >=s -1");
    assert_eq!(o[1].regs[R10] & 0xFF, 0, "INT_MAX <=s -1");
    assert_eq!(o[1].regs[R11] & 0xFF, 1, "INT_MAX >s -1");
    assert_eq!(o[1].regs[RBX] & 0xFF, 1, "SF(INT_MAX - -1) == 1");
    assert_eq!(o[1].regs[RSI] & 0xFF, 1, "INT_MAX <u u64::MAX is true");
    assert!(
        o[1].flags[0] && !o[1].flags[3] && o[1].flags[4] && o[1].flags[5],
        "live cmp flags at exit: CF=1 ZF=0 SF=1 OF=1"
    );

    // Case 2: cmov FALSE
    assert_eq!(o[2].regs[RAX], 0x1122_3344, "32-bit cmov false zeroes upper 32");
    assert_eq!(o[2].regs[R9], 0x99AA_BBCC, "same-reg 32-bit cmov zeroes upper 32");
    assert_eq!(o[2].regs[R10], 0x1234_5678_9ABC_DEF0, "64-bit cmov false preserves r10");

    // Case 3: cmov TRUE
    assert_eq!(o[3].regs[RAX], 0x5566_7788, "32-bit cmov true zeroes upper 32");
    assert_eq!(o[3].regs[R9], 0x99AA_BBCC, "same-reg 32-bit cmov zeroes upper 32");
    assert_eq!(o[3].regs[R10], 0x0FED_CBA9_8765_4321, "64-bit cmov true moves r11");

    // Case 4: test + lea
    assert_eq!(o[4].regs[RAX] & 0xFFFF, 0x0101, "test SIGN_BIT and test 1 both true");
    assert_eq!(o[4].regs[R8], 0x1111_2222_3333_4444, "64-bit identity lea preserves r8");
    assert_eq!(o[4].regs[R9], 0x7777_8888, "32-bit identity lea zeroes upper 32");
    assert_eq!(o[4].regs[R10], 0x8000_0011, "32-bit addr wrap-around");
    assert_eq!(
        o[4].regs[R11],
        0x1234_5678_9ABC_DEF1u64.wrapping_mul(8),
        "chained lea [rsi+rsi] + lea [r11*4]"
    );
}

// ================================================================================================
// 5. Tool-gated x86-64 host-CPU differential verification
// ================================================================================================

#[cfg(target_arch = "x86_64")]
#[test]
fn host_diff_bitops_flag_fusion_and_lea() {
    use std::arch::asm;
    use std::ffi::c_void;

    if !exec_or_skip("host_diff_bitops_flag_fusion_and_lea") {
        return;
    }

    const REGS: [usize; 10] = [RAX, RCX, RDX, RBX, RSI, RDI, R8, R9, R10, R11];
    const FLAG_BITS: [u32; 6] = [0, 2, 4, 6, 7, 11];
    const FLAG_NAMES: [&str; 6] = ["CF", "PF", "AF", "ZF", "SF", "OF"];

    #[repr(C)]
    #[derive(Default)]
    struct Ctx {
        regs: [u64; 10],
        rflags: u64,
    }

    fn alloc_rx(len: usize) -> *mut u8 {
        #[cfg(windows)]
        let ptr = {
            #[link(name = "kernel32")]
            unsafe extern "system" {
                fn VirtualAlloc(addr: *mut c_void, size: usize, kind: u32, protect: u32)
                -> *mut c_void;
            }
            // SAFETY: allocates RWX page for synthetic test snippets.
            unsafe { VirtualAlloc(std::ptr::null_mut(), len, 0x3000, 0x40) }
        };
        #[cfg(target_os = "linux")]
        let ptr = {
            unsafe extern "C" {
                fn mmap(
                    addr: *mut c_void,
                    len: usize,
                    prot: i32,
                    flags: i32,
                    fd: i32,
                    off: i64,
                ) -> *mut c_void;
            }
            // SAFETY: allocates RWX anonymous mapping for synthetic test snippets.
            let p = unsafe { mmap(std::ptr::null_mut(), len, 7, 0x22, -1, 0) };
            if p as isize == -1 {
                std::ptr::null_mut()
            } else {
                p
            }
        };
        assert!(!ptr.is_null(), "executable memory allocation failed");
        ptr.cast()
    }

    unsafe fn run_native(code: *const u8, regs: [u64; 10], flags: [bool; 6]) -> ([u64; 10], [bool; 6]) {
        let mut ctx = Ctx { regs, rflags: 0x202 };
        for k in 0..6 {
            ctx.rflags |= u64::from(flags[k]) << FLAG_BITS[k];
        }
        // SAFETY: snippet only touches caller-saved + rbx registers (restored) and ends in `ret`.
        unsafe {
            asm!(
                "push rbx",
                "push r12",
                "mov rax, [r12]",
                "mov rcx, [r12 + 8]",
                "mov rdx, [r12 + 16]",
                "mov rbx, [r12 + 24]",
                "mov rsi, [r12 + 32]",
                "mov rdi, [r12 + 40]",
                "mov r8, [r12 + 48]",
                "mov r9, [r12 + 56]",
                "mov r10, [r12 + 64]",
                "mov r11, [r12 + 72]",
                "push qword ptr [r12 + 80]",
                "popfq",
                "call r13",
                "pushfq",
                "mov r12, [rsp + 8]",
                "pop qword ptr [r12 + 80]",
                "mov [r12], rax",
                "mov [r12 + 8], rcx",
                "mov [r12 + 16], rdx",
                "mov [r12 + 24], rbx",
                "mov [r12 + 32], rsi",
                "mov [r12 + 40], rdi",
                "mov [r12 + 48], r8",
                "mov [r12 + 56], r9",
                "mov [r12 + 64], r10",
                "mov [r12 + 72], r11",
                "pop r12",
                "pop rbx",
                "cld",
                in("r12") &mut ctx as *mut Ctx,
                in("r13") code,
                out("rax") _, out("rcx") _, out("rdx") _, out("rsi") _, out("rdi") _,
                out("r8") _, out("r9") _, out("r10") _, out("r11") _,
            );
        }
        let out_flags = std::array::from_fn(|k| (ctx.rflags >> FLAG_BITS[k]) & 1 == 1);
        (ctx.regs, out_flags)
    }

    struct Snippet {
        name: &'static str,
        code: Vec<u8>,
        /// Which of `[CF, PF, AF, ZF, SF, OF]` are architecturally defined at `ret`.
        check_flags: [bool; 6],
        inputs: Vec<([u64; 10], [bool; 6])>,
    }

    fn mk_input(rax_v: u64, rcx_v: u64, rdx_v: u64, r8_v: u64, flags: [bool; 6]) -> ([u64; 10], [bool; 6]) {
        (
            [
                rax_v,
                rcx_v,
                rdx_v,
                0xCAFE_BABE_1122_3344,
                0xDEAD_BEEF_5566_7788,
                0x0123_4567_89AB_CDEF,
                r8_v,
                0xFEDC_BA98_7654_3210,
                0x8000_0000_0000_0001,
                0x0000_0000_7FFF_FFFF,
            ],
            flags,
        )
    }

    let edge_pairs = vec![
        mk_input(0, 0, 0, 0, [false; 6]),
        mk_input(0, 1, 0, 1, [true, false, false, true, false, true]),
        mk_input(i64::MIN as u64, 1, i64::MAX as u64, 31, [false, true, false, false, true, false]),
        mk_input(i64::MAX as u64, 32, u64::MAX, 32, [true, false, false, false, false, true]),
        mk_input(0xFFFF_FFFF_8000_0001, 63, 0x8000_0000_0000_0000, 64, [false; 6]),
        mk_input(0xDEAD_BEEF_0000_0000, 64, 0x0000_0000_8000_0000, 8, [true; 6]),
        mk_input(0x1234_5678_9ABC_DEF0, 7, 0x1234_5678_9ABC_DEF0, 16, [false; 6]),
    ];

    let mut snippets: Vec<Snippet> = Vec::new();
    let mut add = |name: &'static str,
                   check_flags: [bool; 6],
                   inputs: Vec<([u64; 10], [bool; 6])>,
                   build_fn: fn(&mut CodeAssembler) -> Result<(), iced_x86::IcedError>| {
        let mut a = CodeAssembler::new(64).unwrap();
        build_fn(&mut a).unwrap();
        a.ret().unwrap();
        let va = CODE_BASE + snippets.len() as u64 * 0x80;
        let code = a.assemble(va).unwrap();
        snippets.push(Snippet {
            name,
            code,
            check_flags,
            inputs,
        });
    };

    // 1. rol/ror cl (32-bit & 64-bit) with counts 0, 1, 7, 31, 32, 63, 64 (checks CF & value; OF undef for >1)
    add(
        "rol32_cl_and_ror64_cl",
        [true, false, false, false, false, false],
        edge_pairs.clone(),
        |a| {
            a.rol(eax, cl)?;
            a.ror(rdx, cl)
        },
    );
    // 2. rol/ror imm 1 + fused carry into adc + full-width 8/16-bit rotates + count 0
    add(
        "rol_ror_imm_and_carry_fusion",
        [true, true, true, true, true, true],
        edge_pairs.clone(),
        |a| {
            a.rol(eax, 0)?;
            a.rol(rdx, 1)?;
            a.adc(r8, 0)?;
            a.ror(r9, 1)?;
            a.sbb(r10, 0)?;
            a.rol(bl, 8)?;
            a.setb(sil)?;
            a.ror(di, 16)?;
            a.setb(r11b)?;
            a.cmp(rax, rdx)
        },
    );
    // 3. bsf/bsr 32-bit & 64-bit including 0 input preserving full 64-bit dst + ZF
    add(
        "bsf_bsr_zero_and_nonzero",
        [false, false, false, true, false, false],
        edge_pairs.clone(),
        |a| {
            a.bsf(edx, eax)?;
            a.setz(r10b)?;
            a.bsr(r8, rax)?;
            a.setz(r11b)?;
            a.bsr(ebx, ecx)
        },
    );
    // 4. bswap 16/32/64 + bt/bts/btr/btc fused with setc/cmovc
    add(
        "bswap_and_bt_fusion",
        [true, false, false, false, false, false],
        edge_pairs.clone(),
        |a| {
            a.bswap(eax)?;
            a.bswap(rdx)?;
            a.bswap(bx)?;
            a.bt(rsi, rcx)?;
            a.setb(r10b)?;
            a.bts(rdi, 17)?;
            a.cmovb(r8, r9)?;
            a.btr(r9, rcx)?;
            a.btc(rsi, 63)
        },
    );
    // 5. cmp/test + jcc/setcc/cmovcc (signed overflow, test r,r, test SIGN_BIT, test 1, 32-bit cmovcc)
    add(
        "cmp_test_cmov_fusion_and_live_flags",
        [true, true, true, true, true, true],
        edge_pairs.clone(),
        |a| {
            a.test(eax, eax)?;
            a.setle(r10b)?;
            a.test(edx, 0x8000_0000u32 as i32)?;
            a.setne(r11b)?;
            a.test(ecx, 1)?;
            a.setne(bl)?;
            a.cmp(rax, rdx)?;
            a.cmovl(esi, r8d)?;
            a.cmovge(edi, edi)?;
            a.cmovg(r8, r9)?;
            a.cmp(rax, rdx)
        },
    );
    // 6. lea identities, 32-bit wrap-around, affine cancel, [rcx+rcx]
    add(
        "lea_identities_and_addr32",
        [true, true, true, true, true, true],
        edge_pairs.clone(),
        |a| {
            a.lea(rax, qword_ptr(rax))?;
            a.lea(r8d, dword_ptr(r8))?;
            a.lea(r9d, dword_ptr(ecx + edx * 4 + 16))?;
            a.lea(r10, qword_ptr(rsi + rsi))?;
            a.lea(r10, qword_ptr(r10 * 4))?;
            a.lea(r11, qword_ptr(rdi + 32))?;
            a.lea(rbx, qword_ptr(r11 + rcx * 2 - 32))
        },
    );
    if is_x86_feature_detected!("popcnt") {
        add(
            "popcnt_full_flag_fusion",
            [true, true, true, true, true, true],
            edge_pairs.clone(),
            |a| {
                a.popcnt(r8, rax)?;
                a.seta(r10b)?;
                a.setle(r11b)?;
                a.popcnt(r9d, ecx)
            },
        );
    }
    if is_x86_feature_detected!("bmi1") {
        add(
            "tzcnt_andn_blsi_blsr_fusion",
            [true, false, false, true, true, true],
            edge_pairs.clone(),
            |a| {
                a.tzcnt(r8, rax)?;
                a.setb(r10b)?;
                a.sete(r11b)?;
                a.andn(rsi, rcx, rcx)?;
                a.blsi(r9, rdx)?;
                a.setb(bl)?;
                a.blsr(rdi, rax)
            },
        );
    }
    if is_x86_feature_detected!("lzcnt") {
        add(
            "lzcnt_jbe_ja_fusion",
            [true, false, false, true, false, false],
            edge_pairs.clone(),
            |a| {
                a.lzcnt(r8, rax)?;
                a.setbe(r10b)?;
                a.seta(r11b)?;
                a.lzcnt(r9d, ecx)
            },
        );
    }

    let exec_buf = alloc_rx(snippets.len() * 128);
    let mut program = Program::new();
    let mut cases = Vec::new();
    let mut meta = Vec::new();
    for (i, s) in snippets.iter().enumerate() {
        assert!(s.code.len() <= 128, "snippet {} too large", s.name);
        let va = CODE_BASE + i as u64 * 0x80;
        // SAFETY: slot `i` lies inside the allocated `snippets.len() * 128` buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(s.code.as_ptr(), exec_buf.add(i * 128), s.code.len());
        }
        program = program.bytes(va, &s.code);
        for (k, &(regs, flags)) in s.inputs.iter().enumerate() {
            let mut c = Case::new(va);
            for (slot, &g) in REGS.iter().enumerate() {
                c.regs[g] = regs[slot];
            }
            c.flags = flags;
            c.out = REGS.to_vec();
            c.out.push(RSP);
            cases.push(c);
            meta.push((i, k));
        }
    }
    let funcs = program.discover();
    let built = build("hyper_bitops_host_diff", funcs.values());
    let outcomes = run(&built, &cases);
    ok(&outcomes, "hyper_bitops_host_diff");

    for (out, (snip_idx, case_idx)) in outcomes.iter().zip(meta) {
        let s = &snippets[snip_idx];
        let (in_regs, in_flags) = s.inputs[case_idx];
        // SAFETY: snippet `snip_idx` was copied to `exec_buf.add(snip_idx * 128)` and ends in `ret`.
        let (want_regs, want_flags) =
            unsafe { run_native(exec_buf.add(snip_idx * 128), in_regs, in_flags) };
        for (slot, &g) in REGS.iter().enumerate() {
            assert_eq!(
                out.regs[g], want_regs[slot],
                "{} case {case_idx}: reg slot {slot} mismatch (got {:#x}, host {:#x})",
                s.name, out.regs[g], want_regs[slot]
            );
        }
        for f in 0..6 {
            if s.check_flags[f] {
                assert_eq!(
                    out.flags[f], want_flags[f],
                    "{} case {case_idx}: flag {} mismatch (got {}, host {})",
                    s.name, FLAG_NAMES[f], out.flags[f], want_flags[f]
                );
            }
        }
    }
}
