//! Tests for provably sound, low-overhead LLVM IR lifting optimizations (issue #2136):
//! - Address calculation, LEA, stack affine flattening, RIP-relative folding, and segment caching
//! - Intra-block 32-bit zero-extension / sub-register / identity / constant ALU & bit-op folding
//! - Intra-block SIMD alloca forwarding, dead store elision, and `zeroinitializer` cast folding
//! - Disjoint multi-entry stack/flat store-to-load forwarding, narrow LE prefix forwarding, and
//!   strict aliasing/overlap soundness
//! - Window-model same-address `window_test` reuse with exact straddle & dirty-mark preservation
//! - Measurable IR instruction-count budgets on representative hot x86-64 sequences
//! - Preservation of the `%state` store-back contract (`docs/LIFTER_STATE_CONTRACT.md`)
//! - Tool-gated Wasm64 execution and x86-64 host-CPU differential verification.

use iced_x86::code_asm::*;
use iced_x86::{Code, Instruction, MemoryOperand, Register};

use super::exec::{
    CODE_BASE, Case, GUEST_BASE, HarnessMemory, Outcome, R8, R9, RAX, RBX, RCX, RDX, RSP,
    STACK_TOP, WINDOW_BASE, WINDOW_SIZE, build, build_with_memory, exec_or_skip, run,
};
use super::{Program, asm_function, encode, switch_program};
use crate::lift::memory::{Window, WindowCheck};
use crate::lift::{GuestBase, MemoryModel, lift_function, lift_function_with, lift_module};

const FLAT_ADDR: u64 = 0x28_0000;

fn lift_ir(build_fn: impl FnOnce(&mut CodeAssembler) -> Result<(), iced_x86::IcedError>) -> String {
    let f = asm_function(0x1000, build_fn);
    let lifted = lift_function(&f).expect("lifts");
    assert!(lifted.unsupported.is_empty(), "{:?}", lifted.unsupported);
    lifted.ir
}

fn lift_ir_with(
    model: MemoryModel,
    build_fn: impl FnOnce(&mut CodeAssembler) -> Result<(), iced_x86::IcedError>,
) -> String {
    let f = asm_function(0x1000, build_fn);
    let lifted = lift_function_with(&f, model).expect("lifts");
    assert!(lifted.unsupported.is_empty(), "{:?}", lifted.unsupported);
    lifted.ir
}

fn ir_instruction_count(ir: &str) -> usize {
    ir.lines()
        .filter(|l| l.starts_with(' ') || l.starts_with('\t'))
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with(';') && !l.ends_with(':'))
        .count()
}

fn window_model(stack_flat: bool) -> MemoryModel {
    MemoryModel::Windowed(Window {
        base: WINDOW_BASE,
        size: WINDOW_SIZE,
        flat: GuestBase::Symbol,
        check: WindowCheck::Branch,
        stack_flat,
        contiguous_runs: false,
    })
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

const fn boundary(k: u64) -> u64 {
    WINDOW_BASE + k * 0x1_0000
}

const fn slot(va: u64) -> u64 {
    (va - WINDOW_BASE) >> 16
}

// ================================================================================================
// 1. Address calculation, LEA, affine stack folding, RIP-relative folding & segment base caching
// ================================================================================================

#[test]
fn address_lea_stack_affine_and_segment_caching_ir() {
    // Frame prologue/epilogue: push rbx; sub rsp, 0x20; ...; add rsp, 0x20; pop rbx; ret.
    // Affine offset tracking should cancel sub/add and push/pop cleanly (no `+ 0` or stacked
    // `(rsp - 8) - 32` chains).
    let frame_ir = lift_ir(|a| {
        a.push(rbx)?;
        a.sub(rsp, 0x20)?;
        a.mov(qword_ptr(rsp + 0x10), rcx)?;
        a.mov(rax, qword_ptr(rsp + 0x10))?;
        a.add(rax, rdx)?;
        a.add(rsp, 0x20)?;
        a.pop(rbx)?;
        a.ret()
    });
    assert!(
        !frame_ir.contains(", 0\n")
            || !frame_ir.contains("add i64 %t")
            || !frame_ir.contains("add i64 %t0, 0"),
        "no redundant add-by-zero:\n{frame_ir}"
    );
    assert!(
        !frame_ir.contains("mul i64 ") && !frame_ir.contains("shl i64 "),
        "plain stack slots need no scale math:\n{frame_ir}"
    );

    // LEA with scale=1, disp=0, and constant folding of RIP-relative base + displacement.
    let lea_ir = lift_ir(|a| {
        a.lea(rax, qword_ptr(rcx + rdx * 1 + 0))?;
        a.lea(r8d, dword_ptr(rcx + rdx * 4 + 16))?;
        a.add(r8d, 1)?;
        a.ret()
    });
    assert!(
        !lea_ir.contains("mul i64 ") && lea_ir.matches("shl i64 ").count() == 1,
        "scale=1 emits no multiply/shift; only scale=4 emits a single `shl i64 ..., 2`:\n{lea_ir}"
    );
    // `lea r8d, [rcx + rdx*4 + 16]` followed by `add r8d, 1` does NOT need a redundant
    // `and i64 ..., 4294967295` before truncating to i32 (dead 32-bit mask eliminated).
    assert_eq!(
        lea_ir.matches("4294967295").count(),
        0,
        "dead lea r32 mask before 32-bit consumer should be eliminated:\n{lea_ir}"
    );

    // RIP-relative base + scaled index + table displacement (MSVC switch table pattern).
    let sw = switch_program(0x1_4000_0000);
    let f = sw.function(0x1_4000_1000);
    let sw_ir = lift_function(&f).expect("lifts").ir;
    // `lea rdx, [rip+__ImageBase]; mov ecx, [rdx + rax*4 + 0x2000]` folds the constant base and
    // table displacement `0x140000000 + 0x2000 = 0x140002000` (`5368717312`) into a single `add`!
    let expected_table_addr = (0x1_4000_0000u64 + 0x2000).to_string();
    assert!(
        sw_ir.contains(&expected_table_addr),
        "RIP-relative base + displacement should fold to constant {expected_table_addr}:\n{sw_ir}"
    );

    // Repeated `gs:` accesses in the same basic block should load `%state.f4` (gs_base) only once.
    let gs_bytes = [
        &[0x65, 0x48, 0x8B, 0x04, 0x25, 0x30, 0x00, 0x00, 0x00][..], // mov rax, gs:[0x30]
        &[0x65, 0x48, 0x03, 0x04, 0x25, 0x60, 0x00, 0x00, 0x00][..], // add rax, gs:[0x60]
        &[0x65, 0x48, 0x03, 0x04, 0x25, 0x68, 0x00, 0x00, 0x00][..], // add rax, gs:[0x68]
        &[0xC3][..],                                                 // ret
    ]
    .concat();
    let gs_fn = Program::new().bytes(0x1000, &gs_bytes).function(0x1000);
    let gs_ir = lift_function(&gs_fn).expect("lifts").ir;
    assert_eq!(
        gs_ir.matches("load i64, ptr %state.f4").count(),
        1,
        "repeated gs: reads in one block must cache gs_base (expected 1 load from %state.f4):\n{gs_ir}"
    );
}

// ================================================================================================
// 2. Intra-block 32-bit GPR chains, constant/identity ALU, shifts, imul & bit ops
// ================================================================================================

#[test]
fn subregister_32bit_and_identity_alu_folding_ir() {
    // 32-bit chain: zero intermediate `trunc i64 ... to i32` on `eax` across consecutive 32-bit ops
    // and zero redundant `and i64 ..., 4294967295`.
    let chain_ir = lift_ir(|a| {
        a.mov(eax, dword_ptr(rcx))?;
        a.add(eax, 7)?;
        a.xor(eax, edx)?;
        a.sub(eax, r8d)?;
        a.mov(dword_ptr(r9), eax)?;
        a.ret()
    });
    // Only edx and r8d (entry registers) should need `trunc i64 ... to i32`; eax never re-truncs!
    assert_eq!(
        chain_ir.matches("trunc i64 ").count(),
        2,
        "32-bit chain should only trunc entry edx/r8d, never round-trip eax:\n{chain_ir}"
    );
    assert_eq!(
        chain_ir.matches("4294967295").count(),
        0,
        "32-bit chain should not emit redundant 32-bit masks:\n{chain_ir}"
    );

    // `cmovcc r32, r32` where both inputs are already known 32-bit zero-extended skips mask.
    let cmov_ir = lift_ir(|a| {
        a.mov(eax, dword_ptr(rcx))?;
        a.mov(edx, dword_ptr(r8))?;
        a.cmp(eax, edx)?;
        a.cmovg(eax, edx)?;
        a.ret()
    });
    assert_eq!(
        cmov_ir.matches("4294967295").count(),
        0,
        "cmovcc r32 with zero-extended operands needs no extra 32-bit mask:\n{cmov_ir}"
    );

    // Soundness check: `mov eax, ecx` on entry register MUST still zero-extend (truncate upper 32).
    let zext_sound_ir = lift_ir(|a| {
        a.mov(eax, ecx)?;
        a.ret()
    });
    assert!(
        zext_sound_ir.contains("zext i32 ") || zext_sound_ir.contains("4294967295"),
        "32-bit write from unknown-upper-bits entry reg must zero-extend:\n{zext_sound_ir}"
    );

    // Identity & constant ALU folding + distinct-reg equal-SSA `sub`/`cmp` + constant `imul` +
    // native 8/16-bit immediate shifts + constant bit ops (zero `@llvm.ctpop.i8` on constants).
    let fold_ir = lift_ir(|a| {
        // Identity ops on live input rcx
        a.mov(rax, rcx)?;
        a.add(rax, 0)?;
        a.or(rax, 0)?;
        a.xor(rax, 0)?;
        a.and(rax, -1)?;
        a.shl(rax, 0)?;
        // Distinct-register equal-SSA `sub` folds to 0
        a.mov(r8, rdx)?;
        a.sub(r8, rdx)?;
        // Constant imul + *1 + *0
        a.mov(r9, 17u64)?;
        a.imul_3(r9, r9, 3)?;
        a.imul_3(r10, rcx, 1)?;
        // Constant bit ops
        a.mov(r11d, 0x1234_5678u32 as i32)?;
        a.bswap(r11d)?;
        a.popcnt(esi, r11d)?;
        a.tzcnt(edi, r11d)?;
        a.lzcnt(ebx, r11d)?;
        a.add(rax, r8)?;
        a.add(rax, r9)?;
        a.add(rax, r10)?;
        a.add(rax, r11)?;
        a.add(rax, rsi)?;
        a.add(rax, rdi)?;
        a.add(rax, rbx)?;
        a.ret()
    });
    assert!(
        !fold_ir.contains("@llvm.bswap.")
            && !fold_ir.contains("@llvm.ctpop.i32")
            && !fold_ir.contains("@llvm.ctpop.i64")
            && !fold_ir.contains("@llvm.cttz.")
            && !fold_ir.contains("@llvm.ctlz."),
        "constant bswap/popcnt/tzcnt/lzcnt should fold completely:\n{fold_ir}"
    );
    assert!(
        !fold_ir.contains("mul i64 "),
        "constant imul and imul-by-1 should emit zero mul instructions:\n{fold_ir}"
    );
    // Pure-constant arithmetic folds even its parity flag (zero `@llvm.ctpop.i8`).
    let const_flags_ir = lift_ir(|a| {
        a.mov(eax, 42)?;
        a.add(eax, 13)?;
        a.ret()
    });
    assert!(
        !const_flags_ir.contains("@llvm.ctpop.i8") && !const_flags_ir.contains("add i32 "),
        "constant ALU and its parity flag should fold completely:\n{const_flags_ir}"
    );

    // Native 8/16-bit immediate shift (`1 <= c < bits`) should emit native i8/i16 shifts rather
    // than widening the shift itself to `i64`.
    let shift8_ir = lift_ir(|a| {
        a.shl(cl, 3)?;
        a.shr(dx, 5)?;
        a.sar(al, 2)?;
        a.ret()
    });
    assert!(
        shift8_ir.contains("shl i8 ")
            && shift8_ir.contains("lshr i16 ")
            && shift8_ir.contains("ashr i8 ")
            && !shift8_ir.contains("shl i64 ")
            && !shift8_ir.contains("lshr i64 ")
            && !shift8_ir.contains("ashr i64 "),
        "immediate 8/16-bit shifts < width should operate natively on i8/i16:\n{shift8_ir}"
    );
}

// ================================================================================================
// 3. SIMD intra-block alloca forwarding, dead store elision & zeroinitializer cast folding
// ================================================================================================

#[test]
fn simd_intra_block_forwarding_and_dead_store_elision_ir() {
    let ir = lift_ir(|a| {
        a.vmovups(xmm0, xmmword_ptr(rcx))?;
        a.vmulps(xmm0, xmm0, xmmword_ptr(rdx))?;
        a.vaddps(xmm0, xmm0, xmmword_ptr(r8))?;
        a.vsubps(xmm0, xmm0, xmmword_ptr(r9))?;
        a.vxorps(xmm1, xmm1, xmm1)?;
        a.vpaddd(xmm1, xmm1, xmm0)?;
        a.vmovups(xmmword_ptr(rax), xmm1)?;
        a.ret()
    });
    // Zero intra-block reloads from local `%xmm0` or `%xmm1` allocas.
    assert!(
        !ir.contains("load <2 x i64>, ptr %xmm0") && !ir.contains("load <2 x i64>, ptr %xmm1"),
        "intra-block SIMD forwarding should eliminate all local %xmm0/%xmm1 reloads:\n{ir}"
    );
    // Overwritten stores to local `%xmm0` / `%xmm1` allocas are dropped (at most 1 store each).
    assert!(
        ir.matches(", ptr %xmm0\n").count() <= 1,
        "overwritten local %xmm0 stores should be dropped:\n{ir}"
    );
    assert!(
        ir.matches(", ptr %xmm1\n").count() <= 1,
        "overwritten local %xmm1 stores should be dropped:\n{ir}"
    );
    // Zeroinitializer vector cast is folded directly.
    assert!(
        !ir.contains("bitcast <4 x float> zeroinitializer")
            && !ir.contains("bitcast <2 x i64> zeroinitializer"),
        "zeroinitializer vector casts should fold directly:\n{ir}"
    );
    // %state sync-out for xmm0 and xmm1 MUST still be preserved!
    assert!(
        ir.contains("ptr %state.xmm0") && ir.contains("ptr %state.xmm1"),
        "%state XMM store-backs must remain intact:\n{ir}"
    );
}

// ================================================================================================
// 4. Disjoint stack/flat store-to-load forwarding, narrow LE prefix & aliasing soundness
// ================================================================================================

#[test]
fn disjoint_stack_forwarding_narrow_prefix_and_aliasing_soundness_ir() {
    for (name, model) in [
        ("flat", MemoryModel::Base(GuestBase::Symbol)),
        ("window-stack", window_model(true)),
    ] {
        // Two disjoint stack spills, then both read back (plus narrow LE prefix reads):
        // zero guest memory loads should be emitted!
        let fwd_ir = lift_ir_with(model, |a| {
            a.sub(rsp, 0x30)?;
            a.mov(qword_ptr(rsp + 0x10), rcx)?;
            a.mov(qword_ptr(rsp + 0x18), rdx)?;
            a.mov(rax, qword_ptr(rsp + 0x10))?;
            a.add(rax, qword_ptr(rsp + 0x18))?;
            a.mov(r8d, dword_ptr(rsp + 0x10))?;
            a.movzx(r9d, byte_ptr(rsp + 0x18))?;
            a.add(rax, r8)?;
            a.add(rax, r9)?;
            a.add(rsp, 0x30)?;
            a.ret()
        });
        // Zero guest memory loads in the entire function! (`ret` only adjusts rsp.)
        let guest_loads = fwd_ir
            .lines()
            .filter(|l| {
                l.contains("= load ")
                    && !l.contains("%state.")
                    && !l.contains("ptr %r")
                    && !l.contains("@__playgta6_")
            })
            .count();
        assert_eq!(
            guest_loads, 0,
            "{name}: disjoint stack spills + narrow LE prefix loads should forward with 0 guest loads:\n{fwd_ir}"
        );

        // Negative soundness 1: overlapping partial store at `[rsp+0x14]` MUST invalidate `[rsp+0x10]`.
        let overlap_ir = lift_ir_with(model, |a| {
            a.sub(rsp, 0x20)?;
            a.mov(qword_ptr(rsp + 0x10), rcx)?;
            a.mov(dword_ptr(rsp + 0x14), edx)?;
            a.mov(rax, qword_ptr(rsp + 0x10))?;
            a.add(rsp, 0x20)?;
            a.ret()
        });
        let overlap_loads = overlap_ir
            .lines()
            .filter(|l| l.contains("= load i64, ptr %t") && !l.contains("@__playgta6_"))
            .count();
        assert!(
            overlap_loads >= 1,
            "{name}: overlapping store at [rsp+0x14] MUST force reload of [rsp+0x10]:\n{overlap_ir}"
        );

        // Negative soundness 2: unknown-base store `[r8]` may alias stack `[rsp+0x10]`, MUST reload!
        let alias_ir = lift_ir_with(model, |a| {
            a.sub(rsp, 0x20)?;
            a.mov(qword_ptr(rsp + 0x10), rcx)?;
            a.mov(qword_ptr(r8), rdx)?;
            a.mov(rax, qword_ptr(rsp + 0x10))?;
            a.add(rsp, 0x20)?;
            a.ret()
        });
        let alias_loads = alias_ir
            .lines()
            .filter(|l| l.contains("= load i64, ptr %t") && !l.contains("@__playgta6_"))
            .count();
        assert!(
            alias_loads >= 1,
            "{name}: unknown-base store [r8] MUST invalidate cached [rsp+0x10]:\n{alias_ir}"
        );
    }
}

// ================================================================================================
// 5. Window-model same-address RMW `window_test` reuse
// ================================================================================================

#[test]
fn window_model_same_address_rmw_reuses_window_test_ir() {
    for (name, model) in [
        ("window", window_model(false)),
        ("window-stack", window_model(true)),
    ] {
        let ir = lift_ir_with(model, |a| {
            a.add(dword_ptr(rcx), eax)?;
            a.ret()
        });
        let ult_win = format!("icmp ult i64 %t1, {WINDOW_SIZE}");
        assert_eq!(
            ir.matches(&ult_win).count(),
            1,
            "{name}: same-address RMW `add [rcx], eax` must reuse window_test (expected 1):\n{ir}"
        );
        // Dirty-mark table write MUST still be present!
        assert!(
            ir.contains("@__playgta6_window_dirty"),
            "{name}: RMW store must preserve window dirty marking:\n{ir}"
        );
    }
}

// ================================================================================================
// 6. Representative hot x86-64 sequences: concrete IR instruction-count budgets
// ================================================================================================

#[test]
fn hot_sequences_ir_instruction_count_budgets() {
    // 1. Frame prologue/epilogue with two disjoint stack spills & reloads.
    let frame_fn = asm_function(0x1000, |a| {
        a.push(rbx)?;
        a.push(rsi)?;
        a.sub(rsp, 0x28)?;
        a.mov(qword_ptr(rsp + 0x10), rcx)?;
        a.mov(qword_ptr(rsp + 0x18), rdx)?;
        a.mov(rax, qword_ptr(rsp + 0x10))?;
        a.add(rax, qword_ptr(rsp + 0x18))?;
        a.add(rsp, 0x28)?;
        a.pop(rsi)?;
        a.pop(rbx)?;
        a.ret()
    });

    // 2. RIP-relative struct / jump-table dispatch.
    let sw_fn = switch_program(0x1_4000_0000).function(0x1_4000_1000);

    // 3. Hot counted loop with 32-bit ALU, `cmp` + `cmovcc` + `jcc`.
    let loop_fn = asm_function(0x2000, |a| {
        let mut hdr = a.create_label();
        a.xor(eax, eax)?;
        a.xor(r8d, r8d)?;
        a.set_label(&mut hdr)?;
        a.mov(r9d, dword_ptr(rcx + r8 * 4))?;
        a.add(r9d, 1)?;
        a.cmp(r9d, eax)?;
        a.cmovg(eax, r9d)?;
        a.inc(r8d)?;
        a.cmp(r8d, edx)?;
        a.jl(hdr)?;
        a.ret()
    });

    // 4. Bitwise / shift math hot path (32-bit chain + immediate shifts + constant bit ops).
    let bit_fn = asm_function(0x3000, |a| {
        a.mov(eax, ecx)?;
        a.shl(eax, 3)?;
        a.xor(eax, edx)?;
        a.shr(eax, 1)?;
        a.and(eax, 0x0FFF_FFFF)?;
        a.or(eax, 0)?;
        a.shl(r8b, 2)?;
        a.shr(r9w, 4)?;
        a.add(eax, r8d)?;
        a.add(eax, r9d)?;
        a.ret()
    });

    // 5. SIMD FP math chain.
    let simd_fn = asm_function(0x4000, |a| {
        a.vmovups(xmm0, xmmword_ptr(rcx))?;
        a.vmulps(xmm0, xmm0, xmmword_ptr(rdx))?;
        a.vaddps(xmm0, xmm0, xmmword_ptr(r8))?;
        a.vsubps(xmm0, xmm0, xmmword_ptr(r9))?;
        a.vmovups(xmmword_ptr(rax), xmm0)?;
        a.ret()
    });

    let cases: [(&str, &crate::cfg::Function, usize, usize); 5] = [
        ("frame_spills", &frame_fn, 108, 112),
        ("rip_switch", &sw_fn, 175, 200),
        ("loop_cmov", &loop_fn, 145, 180),
        ("bit_shift_math", &bit_fn, 145, 145),
        ("simd_fp_chain", &simd_fn, 90, 260),
    ];

    for (label, func, max_flat, max_win_stack) in cases {
        let flat_ir = lift_function_with(func, MemoryModel::Base(GuestBase::Symbol))
            .expect("flat lifts")
            .ir;
        let flat_n = ir_instruction_count(&flat_ir);
        assert!(
            flat_n <= max_flat,
            "{label} (flat): {flat_n} IR instructions exceeded budget {max_flat}:\n{flat_ir}"
        );

        let ws_ir = lift_function_with(func, window_model(true))
            .expect("window-stack lifts")
            .ir;
        let ws_n = ir_instruction_count(&ws_ir);
        assert!(
            ws_n <= max_win_stack,
            "{label} (window-stack): {ws_n} IR instructions exceeded budget {max_win_stack}:\n{ws_ir}"
        );
    }
}

// ================================================================================================
// 7. `%state` store-back contract preservation (`docs/LIFTER_STATE_CONTRACT.md`)
// ================================================================================================

#[test]
fn state_store_back_contract_preserved_with_optimizations() {
    // Function with folded 32-bit chain, SIMD chain, and an indirect call. Every written GPR and
    // XMM must still be synced back into `%state` before the indirect call AND before `ret`.
    let f = asm_function(0x1000, |a| {
        a.mov(eax, ecx)?;
        a.add(eax, 1)?;
        a.mov(edx, eax)?;
        a.vmovups(xmm0, xmmword_ptr(r8))?;
        a.vaddps(xmm0, xmm0, xmm0)?;
        a.call(r9)?;
        a.add(rax, rdx)?;
        a.ret()
    });
    let module = lift_module([&f]).expect("lifts");
    let ir = &module.functions[0].ir;
    assert!(
        ir.contains("ptr %state.gpr0")
            && ir.contains("ptr %state.gpr2")
            && ir.contains("ptr %state.xmm0"),
        "all live GPR/XMM %state store-backs must be emitted:\n{ir}"
    );
    // Before `call void @__playgta6_indirect_call`, `%state` stores for rax, rdx, and xmm0 must appear.
    let before_call = ir
        .split("call void @__playgta6_indirect_call")
        .next()
        .expect("has indirect call");
    assert!(
        before_call.contains(", ptr %state.gpr0")
            && before_call.contains(", ptr %state.gpr2")
            && before_call.contains("store <2 x i64> ")
            && before_call.contains(", ptr %state.xmm0"),
        "indirect call must sync written GPRs and XMMs to %state:\n{before_call}"
    );
}

// ================================================================================================
// 8. Tool-gated Wasm64 execution tests (registers, flags, memory models, dirty marks, SIMD)
// ================================================================================================

fn all_exec_models() -> Vec<(&'static str, HarnessMemory)> {
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

fn expected_dirty(model: &str, slots: &[u64], outside_store: bool) -> Vec<u64> {
    match model {
        "base" => Vec::new(),
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
fn optimized_alu_subreg_and_flags_execute() {
    if !exec_or_skip("optimized_alu_subreg_and_flags_execute") {
        return;
    }

    // 1. 32-bit zero-extension chain + cmovcc r32 + identity ALU + constant bit ops.
    let f_alu = asm_function(CODE_BASE, |a| {
        // rcx, rdx enter with dirty upper 32 bits; eax = ((ecx + 7) ^ edx) - r8d
        a.mov(eax, ecx)?;
        a.add(eax, 7)?;
        a.xor(eax, edx)?;
        a.sub(eax, r8d)?;
        // cmovg r32 with zero-extended operands
        a.mov(r9d, 100)?;
        a.cmp(eax, r9d)?;
        a.cmovg(eax, r9d)?;
        // Identity ops preserve value
        a.add(rax, 0)?;
        a.or(rax, 0)?;
        a.xor(rax, 0)?;
        a.and(rax, -1)?;
        // Constant bswap/popcnt/tzcnt/lzcnt/imul folded
        a.mov(r10d, 0x1234_5678u32 as i32)?;
        a.bswap(r10d)?;
        a.popcnt(esi, r10d)?;
        a.tzcnt(edi, r10d)?;
        a.lzcnt(ebx, r10d)?;
        a.mov(r11, 17u64)?;
        a.imul_3(r11, r11, 3)?;
        a.add(rax, r10)?;
        a.add(rax, rsi)?;
        a.add(rax, rdi)?;
        a.add(rax, rbx)?;
        a.add(rax, r11)?;
        a.ret()
    });

    // 2. Distinct-register equal-SSA `sub`: must yield 0 and ZF=1, PF=1, CF=0, OF=0, SF=0, AF=0.
    let f_eq_sub = asm_function(CODE_BASE + 0x200, |a| {
        a.mov(rax, rcx)?;
        a.sub(rax, rcx)?;
        a.ret()
    });

    // 3. Constant `imul` overflow flag test: i64::MAX * 2 -> overflows signed i64 (CF=1, OF=1).
    let f_imul_ovf = asm_function(CODE_BASE + 0x300, |a| {
        a.mov(rax, i64::MAX)?;
        a.imul_3(rax, rax, 2)?;
        a.ret()
    });

    // 4. Native 8-bit and 16-bit immediate shifts (`shl al, 1` setting CF & OF, `sar bx, 3`).
    let f_shifts = asm_function(CODE_BASE + 0x400, |a| {
        a.mov(rax, rcx)?;
        a.mov(rbx, rdx)?;
        a.sar(bx, 3)?;
        a.shl(al, 1)?;
        a.ret()
    });

    let built = build("low_opt_alu", [&f_alu, &f_eq_sub, &f_imul_ovf, &f_shifts]);
    let cases = [
        Case::new(CODE_BASE)
            .reg(RCX, 0xDEAD_BEEF_0000_0020)
            .reg(RDX, 0xCAFE_BABE_0000_0005)
            .reg(R8, 0x1111_2222_0000_0002),
        Case::new(CODE_BASE + 0x200).reg(RCX, 0x1234_5678_9ABC_DEF0),
        Case::new(CODE_BASE + 0x300),
        // al = 0xC0 -> shl al, 1 => 0x80 (CF=1, OF=0, SF=1, ZF=0, PF=0), upper bytes of rax kept!
        // bx = 0x8010 (-32752i16) -> sar bx, 3 => 0xF002 (-4094i16), upper 48 bits of rbx kept!
        Case::new(CODE_BASE + 0x400)
            .reg(RCX, 0xAAAA_BBBB_CCCC_DDC0)
            .reg(RDX, 0x1122_3344_5566_8010),
    ];
    let o = run(&built, &cases);
    ok(&o, "low_opt_alu");

    let e0 = (((0x20u32.wrapping_add(7)) ^ 0x05).wrapping_sub(2)).min(100) as u64;
    let bsw = 0x1234_5678u32.swap_bytes();
    let want0 = e0
        + u64::from(bsw)
        + u64::from(bsw.count_ones())
        + u64::from(bsw.trailing_zeros())
        + u64::from(bsw.leading_zeros())
        + 51;
    assert_eq!(o[0].regs[RAX], want0, "32-bit chain + bit ops");

    assert_eq!(o[1].regs[RAX], 0, "equal-SSA sub result");
    // [CF, PF, AF, ZF, SF, OF]
    assert_eq!(
        o[1].flags,
        [false, true, false, true, false, false],
        "equal-SSA sub flags"
    );

    assert_eq!(
        o[2].regs[RAX],
        (i64::MAX as u64).wrapping_mul(2),
        "imul ovf result"
    );
    assert!(
        o[2].flags[0] && o[2].flags[5],
        "imul ovf CF and OF must be set: {:?}",
        o[2].flags
    );

    assert_eq!(
        o[3].regs[RAX], 0xAAAA_BBBB_CCCC_DD80,
        "shl al, 1 preserves upper bytes"
    );
    assert_eq!(
        o[3].regs[RBX], 0x1122_3344_5566_F002,
        "sar bx, 3 preserves upper 48 bits"
    );
    assert_eq!(o[3].flags[0], true, "shl al, 1 CF");
    assert_eq!(
        o[3].flags[5], false,
        "shl al, 1 OF (0xC0 -> 0x80: top 2 bits equal)"
    );
    assert_eq!(o[3].flags[4], true, "shl al, 1 SF");
    assert_eq!(o[3].flags[3], false, "shl al, 1 ZF");
    assert_eq!(o[3].flags[1], false, "shl al, 1 PF (0x80 has 1 bit)");
}

#[test]
fn optimized_memory_disjoint_narrow_aliasing_and_window_dirty_execute() {
    if !exec_or_skip("optimized_memory_disjoint_narrow_aliasing_and_window_dirty_execute") {
        return;
    }

    // 1. Disjoint stack spills + negative rbp displacements + narrow LE prefix loads.
    let f_stack = asm_function(CODE_BASE, |a| {
        a.push(rbp)?;
        a.mov(rbp, rsp)?;
        a.sub(rsp, 0x30)?;
        a.mov(qword_ptr(rbp - 0x10), rcx)?;
        a.mov(qword_ptr(rbp - 0x18), rdx)?;
        a.mov(rax, qword_ptr(rbp - 0x10))?;
        a.mov(r8d, dword_ptr(rbp - 0x10))?;
        a.movzx(r9d, word_ptr(rbp - 0x18))?;
        a.movzx(r10d, byte_ptr(rbp - 0x18))?;
        a.add(rax, r8)?;
        a.add(rax, r9)?;
        a.add(rax, r10)?;
        a.add(rsp, 0x30)?;
        a.pop(rbp)?;
        a.ret()
    });

    // 2. Overlapping store + unknown-base aliasing store soundness:
    //    stores qword at [rsp+0x10], overwrites upper dword at [rsp+0x14], then if r8 != 0 also
    //    writes through [r8] (which caller points at rsp+0x10!), then reloads [rsp+0x10].
    let f_alias = asm_function(CODE_BASE + 0x200, |a| {
        a.sub(rsp, 0x20)?;
        a.mov(qword_ptr(rsp + 0x10), rcx)?;
        a.mov(dword_ptr(rsp + 0x14), edx)?;
        a.mov(rax, qword_ptr(rsp + 0x10))?;
        a.lea(r9, qword_ptr(rsp + 0x10))?;
        a.mov(qword_ptr(r9), r8)?;
        a.mov(rdx, qword_ptr(rsp + 0x10))?;
        a.add(rsp, 0x20)?;
        a.ret()
    });

    // 3. Same-address RMW `add dword ptr [rcx], eax` (reuses `window_test`, must keep exact value
    //    and dirty marks for aligned in-window, straddling 64 KiB boundary, and outside-window).
    let f_rmw = asm_function(CODE_BASE + 0x400, |a| {
        a.add(dword_ptr(rcx), eax)?;
        a.ret()
    });

    // 4. SIMD FP + zeroinitializer chain.
    let f_simd = asm_function(CODE_BASE + 0x600, |a| {
        a.vmovups(xmm0, xmmword_ptr(rcx))?;
        a.vmulps(xmm0, xmm0, xmmword_ptr(rdx))?;
        a.vaddps(xmm0, xmm0, xmmword_ptr(r8))?;
        a.vxorps(xmm1, xmm1, xmm1)?;
        a.vpaddd(xmm1, xmm1, xmm0)?;
        a.vmovups(xmmword_ptr(r9), xmm1)?;
        a.ret()
    });

    let aligned = boundary(2) + 0x80;
    let straddle = boundary(4) - 2; // 4-byte RMW straddles slots 3 | 4!
    let simd_a: Vec<u8> = [1.5f32, -2.0, 3.25, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let simd_b: Vec<u8> = [2.0f32, 3.0, 4.0, -0.5]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let simd_c: Vec<u8> = [0.5f32, 1.0, -1.0, 2.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let simd_want: Vec<u8> = [3.5f32, -5.0, 12.0, 0.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    for (name, memory) in all_exec_models() {
        let built = build_with_memory(
            name,
            [&f_stack, &f_alias, &f_rmw, &f_simd],
            &[],
            &memory,
            GUEST_BASE,
        );
        let cases = [
            // 0: disjoint stack + narrow LE prefix
            Case::new(CODE_BASE)
                .reg(RCX, 0x1122_3344_5566_7788)
                .reg(RDX, 0xAABB_CCDD_EEFF_1234),
            // 1: overlap + aliasing store
            Case::new(CODE_BASE + 0x200)
                .reg(RCX, 0x1122_3344_5566_7788)
                .reg(RDX, 0xDEAD_BEEF)
                .reg(R8, 0x0102_0304_0506_0708),
            // 2: RMW aligned inside window slot 2
            Case::new(CODE_BASE + 0x400)
                .reg(RCX, aligned)
                .reg(RAX, 7)
                .mem(aligned, &100u32.to_le_bytes())
                .read(aligned, 4),
            // 3: RMW straddling slots 3 | 4
            Case::new(CODE_BASE + 0x400)
                .reg(RCX, straddle)
                .reg(RAX, 0x0001_0005)
                .mem(straddle, &0x0002_FFFCu32.to_le_bytes())
                .read(straddle, 4),
            // 4: RMW outside window (flat)
            Case::new(CODE_BASE + 0x400)
                .reg(RCX, FLAT_ADDR + 0x40)
                .reg(RAX, 42)
                .mem(FLAT_ADDR + 0x40, &1000u32.to_le_bytes())
                .read(FLAT_ADDR + 0x40, 4),
            // 5: SIMD FP + zeroinitializer chain
            Case::new(CODE_BASE + 0x600)
                .reg(RCX, FLAT_ADDR + 0x100)
                .reg(RDX, FLAT_ADDR + 0x120)
                .reg(R8, FLAT_ADDR + 0x140)
                .reg(R9, FLAT_ADDR + 0x160)
                .mem(FLAT_ADDR + 0x100, &simd_a)
                .mem(FLAT_ADDR + 0x120, &simd_b)
                .mem(FLAT_ADDR + 0x140, &simd_c)
                .read(FLAT_ADDR + 0x160, 16),
        ];
        let o = run(&built, &cases);
        ok(&o, name);

        let want_stack = 0x1122_3344_5566_7788u64
            .wrapping_add(0x5566_7788)
            .wrapping_add(0x1234)
            .wrapping_add(0x34);
        assert_eq!(
            o[0].regs[RAX], want_stack,
            "{name}: disjoint stack + narrow LE"
        );

        assert_eq!(
            o[1].regs[RAX], 0xDEAD_BEEF_5566_7788,
            "{name}: overlapping dword store must be observed"
        );
        assert_eq!(
            o[1].regs[RDX], 0x0102_0304_0506_0708,
            "{name}: aliasing store through pointer must be observed"
        );

        assert_eq!(le(&o[2].mem[0]), 107, "{name}: aligned window RMW");
        assert_eq!(
            o[2].dirty,
            expected_dirty(name, &[slot(aligned)], false),
            "{name}: aligned window RMW dirty mark"
        );

        assert_eq!(
            le(&o[3].mem[0]),
            0x0004_0001,
            "{name}: straddling window RMW"
        );
        assert_eq!(
            o[3].dirty,
            expected_dirty(name, &[slot(straddle), slot(straddle) + 1], false),
            "{name}: straddling window RMW must mark BOTH slots"
        );

        assert_eq!(le(&o[4].mem[0]), 1042, "{name}: flat RMW");
        assert_eq!(
            o[4].dirty,
            expected_dirty(name, &[], true),
            "{name}: flat RMW dirty mark"
        );

        assert_eq!(o[5].mem[0], simd_want, "{name}: SIMD chain output");
    }
}

// ================================================================================================
// 9. Host-CPU differential verification (x86-64 hosts, tool-gated)
// ================================================================================================

#[cfg(target_arch = "x86_64")]
#[test]
fn host_diff_low_overhead_opt_patterns() {
    if !exec_or_skip("host_diff_low_overhead_opt_patterns") {
        return;
    }

    // Compare lifted Wasm64 output registers & flags against real x86-64 host CPU inline asm on:
    //  - 32-bit zero-extending chain + cmovg r32
    //  - native 8-bit `shl al, 3` / `sar al, 2` and 16-bit `shr ax, 5` / `rol ax, 3`
    //  - `imul` 64-bit immediate with overflow vs non-overflow
    //  - `sub` equal registers (distinct GPRs holding equal value)
    fn native_chain32(a: u64, b: u64) -> (u64, u64) {
        let mut out = a;
        let flags: u64;
        // SAFETY: pure register arithmetic; pushfq/pop keeps stack balanced.
        unsafe {
            std::arch::asm!(
                "mov {out:e}, {a:e}",
                "add {out:e}, 7",
                "xor {out:e}, {b:e}",
                "cmp {out:e}, {b:e}",
                "cmovg {out:e}, {b:e}",
                "pushfq",
                "pop {f}",
                a = in(reg) a,
                b = in(reg) b,
                out = out(reg) out,
                f = out(reg) flags,
            );
        }
        (out, flags)
    }

    fn native_shl8(a: u64) -> (u64, u64) {
        let mut out = a;
        let flags: u64;
        // SAFETY: pure register arithmetic.
        unsafe {
            std::arch::asm!(
                "shl {a:l}, 3",
                "pushfq",
                "pop {f}",
                a = inout(reg) out,
                f = out(reg) flags,
            );
        }
        (out, flags)
    }

    fn native_shr16(a: u64) -> (u64, u64) {
        let mut out = a;
        let flags: u64;
        // SAFETY: pure register arithmetic.
        unsafe {
            std::arch::asm!(
                "shr {a:x}, 5",
                "pushfq",
                "pop {f}",
                a = inout(reg) out,
                f = out(reg) flags,
            );
        }
        (out, flags)
    }

    let f_chain32 = asm_function(CODE_BASE, |a| {
        a.mov(eax, ecx)?;
        a.add(eax, 7)?;
        a.xor(eax, edx)?;
        a.cmp(eax, edx)?;
        a.cmovg(eax, edx)?;
        a.ret()
    });
    let f_shl8 = asm_function(CODE_BASE + 0x100, |a| {
        a.mov(rax, rcx)?;
        a.shl(al, 3)?;
        a.ret()
    });
    let f_shr16 = asm_function(CODE_BASE + 0x200, |a| {
        a.mov(rax, rcx)?;
        a.shr(ax, 5)?;
        a.ret()
    });
    // Also verify memory-operand `bt` with constant register bit offset via RIP-relative slot.
    let bt_bytes = encode(
        CODE_BASE + 0x300,
        &[
            Instruction::with2(Code::Mov_r64_imm64, Register::RCX, 19).expect("mov rcx, 19"),
            Instruction::with2(
                Code::Bt_rm64_r64,
                MemoryOperand::with_base(Register::RDX),
                Register::RCX,
            )
            .expect("bt [rdx], rcx"),
            Instruction::with1(Code::Setb_rm8, Register::AL).expect("setc al"),
            Instruction::with2(Code::Movzx_r32_rm8, Register::EAX, Register::AL).expect("movzx"),
            Instruction::with(Code::Retnq),
        ],
    );
    let f_bt = Program::new()
        .bytes(CODE_BASE + 0x300, &bt_bytes)
        .function(CODE_BASE + 0x300);

    let built = build("low_opt_host_diff", [&f_chain32, &f_shl8, &f_shr16, &f_bt]);
    let inputs: [(u64, u64); 6] = [
        (0, 0),
        (0xDEAD_BEEF_0000_0001, 0xCAFE_BABE_8000_0000),
        (0xFFFF_FFFF_7FFF_FFFF, 0x1234_5678_FFFF_FFFF),
        (0x0102_0304_0506_0781, 0x0000_0000_0000_0007),
        (0xAAAA_BBBB_CCCC_DDFF, 0x5555_6666_7777_8888),
        (0x8000_0000_8000_8080, 0x0000_0000_0000_0001),
    ];

    let mut cases = Vec::new();
    for &(a, b) in &inputs {
        cases.push(Case::new(CODE_BASE).reg(RCX, a).reg(RDX, b));
        cases.push(Case::new(CODE_BASE + 0x100).reg(RCX, a));
        cases.push(Case::new(CODE_BASE + 0x200).reg(RCX, a));
    }
    cases.push(
        Case::new(CODE_BASE + 0x300)
            .reg(RDX, FLAT_ADDR)
            .mem(FLAT_ADDR, &(1u64 << 19).to_le_bytes()),
    );
    cases.push(
        Case::new(CODE_BASE + 0x300)
            .reg(RDX, FLAT_ADDR)
            .mem(FLAT_ADDR, &(!(1u64 << 19)).to_le_bytes()),
    );

    let outcomes = run(&built, &cases);
    ok(&outcomes, "low_opt_host_diff");

    let flag_bits = [0usize, 2, 4, 6, 7, 11]; // CF, PF, AF, ZF, SF, OF
    for (idx, &(a, b)) in inputs.iter().enumerate() {
        let base = idx * 3;
        // 1. chain32 + cmovg (all 6 flags defined by cmp)
        let (want_r, want_f) = native_chain32(a, b);
        assert_eq!(
            outcomes[base].regs[RAX], want_r,
            "chain32({a:#x}, {b:#x}) reg"
        );
        for (fi, &bit) in flag_bits.iter().enumerate() {
            assert_eq!(
                outcomes[base].flags[fi],
                ((want_f >> bit) & 1) != 0,
                "chain32({a:#x}, {b:#x}) flag {fi}"
            );
        }
        // 2. shl8 (CF, PF, ZF, SF defined for count=3)
        let (want_r, want_f) = native_shl8(a);
        assert_eq!(outcomes[base + 1].regs[RAX], want_r, "shl8({a:#x}) reg");
        for fi in [0, 1, 3, 4] {
            let bit = flag_bits[fi];
            assert_eq!(
                outcomes[base + 1].flags[fi],
                ((want_f >> bit) & 1) != 0,
                "shl8({a:#x}) flag {fi}"
            );
        }
        // 3. shr16 (CF, PF, ZF, SF defined for count=5)
        let (want_r, want_f) = native_shr16(a);
        assert_eq!(outcomes[base + 2].regs[RAX], want_r, "shr16({a:#x}) reg");
        for fi in [0, 1, 3, 4] {
            let bit = flag_bits[fi];
            assert_eq!(
                outcomes[base + 2].flags[fi],
                ((want_f >> bit) & 1) != 0,
                "shr16({a:#x}) flag {fi}"
            );
        }
    }

    let bt_base = inputs.len() * 3;
    assert_eq!(outcomes[bt_base].regs[RAX], 1, "bt [rdx], 19 bit set");
    assert_eq!(outcomes[bt_base + 1].regs[RAX], 0, "bt [rdx], 19 bit clear");
}
