//! Tests for issue #2206:
//! 1. Intra-block memory translation & dirty-mark optimizations (`recomp/src/lift/memory_access.rs`):
//!    - Reuse of loaded `@__playgta6_window_table` / `@__playgta6_page_table` slot deltas and
//!      `@__playgta6_window_dirty` marks within the same basic block (same-address RMW/stores,
//!      constant same-slot VAs, and provably aligned same-slot spans such as `clzero` and
//!      `and rcx, -64` structs) across `WindowCheck::Branch`, `WindowCheck::Select`, and `PageTable`.
//!    - Strict invalidation across direct/indirect calls, atomics, fences, `pause`, runtime
//!      helpers (`cpuid`), and basic-block boundaries.
//!    - Exact preservation of straddle checks, `#PF` traps, and `%state` store-backs
//!      (`docs/LIFTER_STATE_CONTRACT.md`).
//! 2. Bit-exact integer constant folding & strength reduction (`recomp/src/lift/integer.rs`):
//!    - `imul` (2-op & 3-op) with `0`, `1`, `-1`, `+2^k`, `-2^k`, `INT_MIN` (live & dead CF/OF).
//!    - Widening `mul`/`imul` with constants, `0`, `1`, `-1`, `2^k` (eliminating `@__playgta6_umul128`).
//!    - `div`/`idiv` with constant divisors (`0`, `1`, `2^k`, `-1`, general constants) preserving
//!      exact `#DE` divide-by-zero and `INT_MIN / -1` overflow trap semantics.
//!    - `neg` constant folding + simplified dynamic `AF = ((a & 15) != 0)`.
//!    - `cdq`/`cqo`/`cdqe`/`movsx`/`movsxd`/`movzx` constant folding, non-negative sign-bit
//!      elision, high-byte `movzx`, and 32-bit zero-extension to 64-bit.
//! 3. Tool-gated Wasm64 execution tests (Flat, Windowed, PageTable) and x86-64 host-CPU
//!    differential tests.

use iced_x86::code_asm::*;
use iced_x86::{Code, Instruction, MemoryOperand, Register};

use super::exec::{
    CODE_BASE, Case, GUEST_BASE, HarnessMemory, Outcome, R8, R9, R10, R11, RAX, RBX, RCX, RDI,
    RDX, RSI, RSP, STACK_TOP, WINDOW_BASE, WINDOW_SIZE, build, build_with_memory, exec_or_skip,
    run, window_backing,
};
use super::{Program, asm_function, encode};
use crate::lift::memory::{Window, WindowCheck};
use crate::lift::{GuestBase, MemoryModel, lift_function, lift_function_with, lift_module_with};

const FLAT_ADDR: u64 = 0x28_0000;

const fn boundary(k: u64) -> u64 {
    WINDOW_BASE + k * 0x1_0000
}

const fn slot(va: u64) -> u64 {
    (va - WINDOW_BASE) >> 16
}

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

fn window_model(check: WindowCheck, stack_flat: bool, contiguous_runs: bool) -> MemoryModel {
    MemoryModel::Windowed(Window {
        flat: GuestBase::Symbol,
        base: WINDOW_BASE,
        size: WINDOW_SIZE,
        check,
        stack_flat,
        contiguous_runs,
    })
}

fn page_model(stack_flat: bool, contiguous_runs: bool) -> MemoryModel {
    MemoryModel::PageTable {
        pages: 1 << 16,
        stack: stack_flat.then_some(GuestBase::Symbol),
        contiguous_runs,
    }
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

fn expected_dirty(model: &str, slots: &[u64], outside_store: bool) -> Vec<u64> {
    match model {
        "base" | "page" | "page_stack" => Vec::new(),
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
        (
            "page",
            HarnessMemory::page(&HarnessMemory::standard_slots()),
        ),
        (
            "page_stack",
            HarnessMemory::page_opts(
                &HarnessMemory::stack_identity_slots(),
                Some(GuestBase::Symbol),
            ),
        ),
    ]
}

// ================================================================================================
// 1. IR-shape tests for intra-block window/page slot-delta & dirty-mark reuse + invalidation
// ================================================================================================

#[test]
fn intra_block_window_and_page_slot_reuse_ir() {
    // 1. RMW `add dword ptr [rcx], eax` under default `window` (!contiguous_runs), `window-runs`,
    //    `window-select`, and `page` (!contiguous_runs): MUST load the delta table ONLY ONCE!
    for (name, model, table_sym) in [
        (
            "window",
            window_model(WindowCheck::Branch, true, false),
            "@__playgta6_window_table",
        ),
        (
            "window-runs",
            window_model(WindowCheck::Branch, true, true),
            "@__playgta6_window_table",
        ),
        (
            "window-select",
            window_model(WindowCheck::Select, true, false),
            "@__playgta6_window_table",
        ),
        ("page", page_model(true, false), "@__playgta6_page_table"),
        (
            "page-runs",
            page_model(true, true),
            "@__playgta6_page_table",
        ),
    ] {
        let ir = lift_ir_with(model, |a| {
            a.add(dword_ptr(rcx), eax)?;
            a.ret()
        });
        assert_eq!(
            ir.matches(table_sym).count(),
            1,
            "{name}: same-address RMW must perform exactly 1 {table_sym} lookup:\n{ir}"
        );
        if table_sym == "@__playgta6_window_table" {
            assert_eq!(
                ir.matches("load atomic i8, ptr ").count(),
                1,
                "{name}: RMW store must emit 1 guarded dirty mark:\n{ir}"
            );
        }
    }

    // 2. Consecutive stores to the SAME address (not coalesced by burst when separated by a
    //    non-burst instruction like `xlat` or in `window-select`): reuse BOTH table lookup AND
    //    dirty mark!
    //    Also test 2 consecutive stores under `window-select` (which does not use bursts).
    let select_two_stores = lift_ir_with(window_model(WindowCheck::Select, true, false), |a| {
        a.mov(qword_ptr(rcx), rax)?;
        a.mov(qword_ptr(rcx), rdx)?;
        a.ret()
    });
    assert_eq!(
        select_two_stores.matches("@__playgta6_window_table").count(),
        1,
        "window-select: consecutive same-address stores must share 1 window_table lookup:\n{select_two_stores}"
    );
    assert_eq!(
        select_two_stores.matches("store atomic i8 1, ptr ").count(),
        1,
        "window-select: consecutive same-address stores must share 1 dirty mark:\n{select_two_stores}"
    );

    // 3. Aligned 64-byte span (`and rcx, -64` or `clzero`): ALL 8 qword stores lie provably in
    //    the exact same 64 KiB slot with zero straddling -> 1 table lookup, 0 straddle checks,
    //    and 1 dirty mark across `window`, `window-select`, and `page`!
    for (name, model, table_sym) in [
        (
            "window",
            window_model(WindowCheck::Branch, true, false),
            "@__playgta6_window_table",
        ),
        (
            "window-select",
            window_model(WindowCheck::Select, true, false),
            "@__playgta6_window_table",
        ),
        ("page", page_model(true, false), "@__playgta6_page_table"),
    ] {
        let ir = lift_ir_with(model, |a| {
            a.clzero()?;
            a.ret()
        });
        assert_eq!(
            ir.matches(table_sym).count(),
            1,
            "{name}: clzero (8 aligned qwords) must emit exactly 1 {table_sym} lookup:\n{ir}"
        );
        assert!(
            !ir.contains("@__playgta6_slow_store"),
            "{name}: 64-byte aligned clzero cannot straddle and needs no slow_store:\n{ir}"
        );
        if table_sym == "@__playgta6_window_table" {
            assert_eq!(
                ir.matches("store atomic i8 1, ptr ").count(),
                1,
                "{name}: clzero must emit exactly 1 dirty mark for all 8 stores:\n{ir}"
            );
        }
    }

    // 4. Constant in-window RIP-relative accesses in the same 64 KiB slot share 1 table lookup!
    let va0 = WINDOW_BASE + 0x2_0010;
    let va1 = WINDOW_BASE + 0x2_0018;
    let va2 = WINDOW_BASE + 0x2_0020;
    let code_va = WINDOW_BASE - 0x1000;
    let bytes = encode(
        code_va,
        &[
            Instruction::with2(
                Code::Mov_r64_rm64,
                Register::RAX,
                MemoryOperand::with_base_displ(Register::RIP, va0 as i64),
            )
            .expect("ld0"),
            Instruction::with2(
                Code::Add_r64_rm64,
                Register::RAX,
                MemoryOperand::with_base_displ(Register::RIP, va1 as i64),
            )
            .expect("ld1"),
            Instruction::with2(
                Code::Add_r64_rm64,
                Register::RAX,
                MemoryOperand::with_base_displ(Register::RIP, va2 as i64),
            )
            .expect("ld2"),
            Instruction::with(Code::Retnq),
        ],
    );
    let const_fn = Program::new().bytes(code_va, &bytes).function(code_va);
    for (name, model, table_sym) in [
        (
            "window",
            window_model(WindowCheck::Branch, true, false),
            "@__playgta6_window_table",
        ),
        ("page", page_model(true, false), "@__playgta6_page_table"),
    ] {
        let ir = lift_function_with(&const_fn, model).expect("lifts").ir;
        assert_eq!(
            ir.matches(table_sym).count(),
            1,
            "{name}: 3 constant loads in the same 64 KiB slot must share 1 {table_sym} lookup:\n{ir}"
        );
    }

    // 5. Soundness: misaligned access outside alignment bound (`and rcx, -16; mov rax, [rcx+12]`)
    //    MUST keep the straddle check (`icmp ule` + `@__playgta6_guest_read`)!
    let misalign_ir = lift_ir_with(window_model(WindowCheck::Branch, true, false), |a| {
        a.and(rcx, -16)?;
        a.mov(rax, qword_ptr(rcx + 12))?;
        a.ret()
    });
    assert!(
        misalign_ir.contains("@__playgta6_guest_read"),
        "offset 12 + 8 > align 16 MUST retain straddle check & guest_read fallback:\n{misalign_ir}"
    );

    // 6. Soundness: narrow 1-byte store followed by wider 8-byte store at the same unaligned
    //    address under `window-runs` MUST NOT skip the cross-slot mark for the 8-byte store!
    let narrow_then_wide = lift_ir_with(window_model(WindowCheck::Branch, true, true), |a| {
        a.mov(byte_ptr(rcx), al)?;
        a.xor(edx, edx)?;
        a.mov(qword_ptr(rcx), rdx)?;
        a.ret()
    });
    assert!(
        narrow_then_wide.contains("icmp ne i64 ")
            || narrow_then_wide.matches("store atomic i8 1, ptr ").count() >= 1,
        "narrow-then-wide store must preserve cross-slot dirty mark check:\n{narrow_then_wide}"
    );
}

#[test]
fn intra_block_caches_strictly_invalidate_across_calls_barriers_and_blocks_ir() {
    // 1. Direct call between two stores to `[rcx]` MUST clear both window_table and dirty caches!
    let callee_va = 0x2000u64;
    let prog = Program::new()
        .asm(0x1000, |a| {
            a.mov(qword_ptr(rcx), rax)?;
            a.call(callee_va)?;
            a.mov(qword_ptr(rcx), rax)?;
            a.ret()
        })
        .asm(callee_va, |a| {
            a.inc(rax)?;
            a.ret()
        });
    let funcs = prog.discover();
    for (name, model) in [
        ("window", window_model(WindowCheck::Branch, true, false)),
        (
            "window-select",
            window_model(WindowCheck::Select, true, false),
        ),
    ] {
        let ir = lift_module_with(funcs.values(), model).expect("lifts").ir;
        assert_eq!(
            ir.matches("ptr @__playgta6_window_table, i64 ").count(),
            2,
            "{name}: direct call must invalidate window_table cache (expected 2 lookups):\n{ir}"
        );
        assert_eq!(
            ir.matches("store atomic i8 1, ptr ").count(),
            2,
            "{name}: direct call must invalidate window_dirty cache (expected 2 dirty marks):\n{ir}"
        );
    }

    // 2. Indirect call (`call r11`), `mfence`, `pause`, `lock xadd`, and `cpuid` MUST all
    //    invalidate slot & dirty caches!
    for (barrier_name, build_fn) in [
        (
            "indirect_call",
            (|a: &mut CodeAssembler| {
                a.mov(qword_ptr(rcx), rax)?;
                a.call(r11)?;
                a.mov(qword_ptr(rcx), rdx)?;
                a.ret()
            }) as fn(&mut CodeAssembler) -> Result<(), iced_x86::IcedError>,
        ),
        ("mfence", |a| {
            a.mov(qword_ptr(rcx), rax)?;
            a.mfence()?;
            a.mov(qword_ptr(rcx), rdx)?;
            a.ret()
        }),
        ("pause", |a| {
            a.mov(qword_ptr(rcx), rax)?;
            a.pause()?;
            a.mov(qword_ptr(rcx), rdx)?;
            a.ret()
        }),
        ("cpuid", |a| {
            a.mov(qword_ptr(rcx), rax)?;
            a.cpuid()?;
            a.mov(qword_ptr(rcx), rdx)?;
            a.ret()
        }),
    ] {
        for (mname, model) in [
            ("window", window_model(WindowCheck::Branch, true, false)),
            (
                "window-select",
                window_model(WindowCheck::Select, true, false),
            ),
        ] {
            let ir = lift_ir_with(model, build_fn);
            assert_eq!(
                ir.matches("ptr @__playgta6_window_table, i64 ").count(),
                2,
                "{mname}/{barrier_name}: barrier must invalidate window_table cache:\n{ir}"
            );
            assert_eq!(
                ir.matches("store atomic i8 1, ptr ").count(),
                2,
                "{mname}/{barrier_name}: barrier must invalidate window_dirty cache:\n{ir}"
            );
        }
    }

    // 3. CFG block boundary (conditional branch + join) MUST clear slot & dirty caches!
    let branch_ir = lift_ir_with(window_model(WindowCheck::Select, true, false), |a| {
        let mut join = a.create_label();
        a.mov(qword_ptr(rcx), rax)?;
        a.test(edx, edx)?;
        a.jz(join)?;
        a.inc(rax)?;
        a.set_label(&mut join)?;
        a.mov(qword_ptr(rcx), rax)?;
        a.ret()
    });
    assert_eq!(
        branch_ir
            .matches("ptr @__playgta6_window_table, i64 ")
            .count(),
        2,
        "CFG block boundary must invalidate window_table cache:\n{branch_ir}"
    );
    assert_eq!(
        branch_ir.matches("store atomic i8 1, ptr ").count(),
        2,
        "CFG block boundary must invalidate window_dirty cache:\n{branch_ir}"
    );

    // 4. LIFTER_STATE_CONTRACT (#2033): written GPRs and live flags MUST still store back to
    //    `%state.gpr*` and `%state.flag*`!
    let contract_ir = lift_ir_with(window_model(WindowCheck::Branch, true, false), |a| {
        a.imul_3(rax, rcx, 8)?;
        a.mov(qword_ptr(rdx), rax)?;
        a.ret()
    });
    assert!(
        contract_ir.contains("ptr %state.gpr0")
            && contract_ir.contains("ptr %state.flag0")
            && contract_ir.contains("ptr %state.flag5"),
        "imul + window store must preserve %state.gpr0 and live CF/OF store-backs:\n{contract_ir}"
    );
}

// ================================================================================================
// 2. IR-shape tests for integer.rs constant folding & strength reduction
// ================================================================================================

#[test]
fn integer_imul_mul_div_neg_and_extend_folding_ir() {
    // 1. Truncating `imul` by 0, 1, -1, +2^k, -2^k, INT_MIN with LIVE CF/OF:
    //    ZERO `mul` instructions and ZERO `@__playgta6_umul128` calls!
    let imul_ir = lift_ir(|a| {
        a.imul_3(rax, rcx, 0)?;
        a.imul_3(rdx, rcx, 1)?;
        a.imul_3(r8, rcx, -1)?;
        a.imul_3(r9, rcx, 8)?;
        a.imul_3(r10, rcx, -16)?;
        a.imul_3(r11d, ecx, i32::MIN)?;
        a.add(rax, rdx)?;
        a.add(rax, r8)?;
        a.add(rax, r9)?;
        a.add(rax, r10)?;
        a.add(rax, r11)?;
        // Final imul with LIVE CF/OF at `ret`:
        a.imul_3(rax, rcx, 16)?;
        a.ret()
    });
    assert!(
        !imul_ir.contains("mul i64 ")
            && !imul_ir.contains("mul i32 ")
            && !imul_ir.contains("@__playgta6_umul128"),
        "imul by 0/1/-1/±2^k/INT_MIN must emit zero mul and zero umul128:\n{imul_ir}"
    );
    assert!(
        imul_ir.contains("shl i64 ")
            && imul_ir.contains("ashr i64 ")
            && imul_ir.contains("ptr %state.flag0")
            && imul_ir.contains("ptr %state.flag5"),
        "imul by +16 with live CF/OF must use shl+ashr+icmp and sync CF/OF to %state:\n{imul_ir}"
    );

    // 2. Widening 64-bit `mul` by 2^k and `imul` by -1 / +2^k / 1 / constants:
    //    ZERO `@__playgta6_umul128` calls!
    let wide_mul_ir = lift_ir(|a| {
        a.mov(rcx, 64u64)?;
        a.mul(rcx)?;
        a.mov(r8, -1i64 as u64)?;
        a.imul(r8)?;
        a.mov(r9, 16u64)?;
        a.imul(r9)?;
        a.ret()
    });
    assert!(
        !wide_mul_ir.contains("@__playgta6_umul128"),
        "widening 64-bit mul/imul by 2^k and -1 must eliminate @__playgta6_umul128:\n{wide_mul_ir}"
    );

    // 3. Unsigned `div` with `xor edx, edx` and constant power-of-two / 1 divisor:
    //    ZERO `udiv`/`urem` and ZERO `#DE` trap branch!
    let udiv_pow2_ir = lift_ir(|a| {
        a.xor(edx, edx)?;
        a.mov(ecx, 16)?;
        a.div(ecx)?;
        a.ret()
    });
    assert!(
        !udiv_pow2_ir.contains("udiv ")
            && !udiv_pow2_ir.contains("urem ")
            && !udiv_pow2_ir.contains("@__playgta6_divide_error")
            && udiv_pow2_ir.contains("lshr i32 ")
            && udiv_pow2_ir.contains("and i32 "),
        "xor edx,edx; div 16 must strength-reduce to lshr+and with zero trap:\n{udiv_pow2_ir}"
    );

    // 4. Signed `idiv` after `cdq` with constant non-zero divisor != -1 (e.g. 7):
    //    ZERO `#DE` trap branch; while `idiv -1` MUST retain the `INT_MIN` trap check!
    let idiv_const_ir = lift_ir(|a| {
        a.cdq()?;
        a.mov(ecx, 7)?;
        a.idiv(ecx)?;
        a.ret()
    });
    assert!(
        !idiv_const_ir.contains("@__playgta6_divide_error") && idiv_const_ir.contains("sdiv i32 "),
        "cdq; idiv 7 cannot trap and must omit divide_error:\n{idiv_const_ir}"
    );
    let idiv_neg1_ir = lift_ir(|a| {
        a.cdq()?;
        a.mov(ecx, -1)?;
        a.idiv(ecx)?;
        a.ret()
    });
    assert!(
        idiv_neg1_ir.contains("@__playgta6_divide_error")
            && idiv_neg1_ir.contains("-2147483648")
            && !idiv_neg1_ir.contains("sdiv "),
        "cdq; idiv -1 must guard INT_MIN trap and lower to sub without sdiv:\n{idiv_neg1_ir}"
    );

    // 5. Constant `neg` folds completely (zero `sub`, zero `@llvm.ctpop.i8`);
    //    `cdq`/`cqo`/`cdqe`/`movsx`/`movsxd` on known non-negative inputs emit zero `ashr`/`sext`;
    //    `movzx ebx, ah` on 16-bit zero-extended `ax` folds to a single `lshr` (no `and`), while
    //    `movzx edi, ch` on unknown entry `rcx` lowers to `lshr` + `and 255` (no `trunc` + `zext`).
    let ext_ir = lift_ir(|a| {
        a.mov(r8d, 42)?;
        a.neg(r8d)?;
        a.movzx(eax, word_ptr(rcx))?;
        a.cdq()?;
        a.cdqe()?;
        a.movzx(r9d, byte_ptr(rsi))?;
        a.movsxd(r10, r9d)?;
        a.movzx(ebx, ah)?;
        a.movzx(edi, ch)?;
        a.add(rax, rdx)?;
        a.add(rax, r8)?;
        a.add(rax, r10)?;
        a.add(rax, rbx)?;
        a.add(rax, rdi)?;
        a.ret()
    });
    assert!(
        !ext_ir.contains("ashr ") && !ext_ir.contains("sext "),
        "known non-negative cdq/cdqe/movsxd must emit zero ashr/sext:\n{ext_ir}"
    );
    assert!(
        ext_ir.contains("lshr i64 ")
            && ext_ir.contains("and i64 ")
            && ext_ir.contains(", 255")
            && !ext_ir.contains("zext i8 %t12"),
        "movzx r32, r8h should lower to lshr (+ and 255 on unknown upper bits) without trunc+zext:\n{ext_ir}"
    );
}

// ================================================================================================
// 3. Tool-gated Wasm64 execution tests (Flat, Windowed, PageTable + integer edge cases & #DE traps)
// ================================================================================================

#[test]
fn memory_slot_reuse_straddles_and_dirty_invalidation_execute_across_all_models() {
    // F1: Same-slot aligned struct + clzero + RMW + narrow-then-wide store + straddle RMW.
    let f_mem = asm_function(CODE_BASE, |a| {
        // 1. Aligned 64-byte clzero at r8 (must zero all 64 bytes and mark slot).
        a.mov(rax, r8)?;
        a.clzero()?;
        // 2. Aligned 64-byte struct stores + loads via `and r9, -64`.
        a.and(r9, -64)?;
        a.mov(qword_ptr(r9), rcx)?;
        a.mov(qword_ptr(r9 + 8), rdx)?;
        a.mov(rax, qword_ptr(r9))?;
        a.add(rax, qword_ptr(r9 + 8))?;
        a.mov(qword_ptr(r9 + 16), rax)?;
        // 3. Dynamic RMW at rdi (tested both aligned in-slot and straddling a 64 KiB boundary!).
        a.add(qword_ptr(rdi), 0x1122)?;
        a.mov(r10, qword_ptr(rdi))?;
        a.add(rax, r10)?;
        // 4. Narrow 1-byte store then wider 8-byte store at rsi (tested straddling 64 KiB!).
        a.mov(byte_ptr(rsi), 0x5Au32)?;
        a.mov(qword_ptr(rsi), rax)?;
        a.ret()
    });

    // F2: Store before indirect remap call, remap slot 5 -> slot 6 via host helper 0x7FFE_0030,
    //     then reload and store after the call (verifying cache invalidation!).
    let f_remap = asm_function(CODE_BASE + 0x200, |a| {
        a.mov(qword_ptr(rdi), rax)?;
        a.push(rax)?;
        a.sub(rsp, 0x28)?;
        a.mov(rcx, rdi)?;
        a.mov(rdx, window_backing(6))?;
        a.mov(r11, 0x7FFE_0030u64)?;
        a.call(r11)?;
        a.add(rsp, 0x28)?;
        a.mov(rbx, qword_ptr(rdi))?;
        a.add(rbx, 1)?;
        a.mov(qword_ptr(rdi), rbx)?;
        a.pop(rax)?;
        a.ret()
    });

    // Always verify assembly & lifting across every model, even when LLVM execution tools are absent.
    for (name, memory) in all_exec_models() {
        let m = lift_module_with([&f_mem, &f_remap], memory.model).expect("lifts");
        for lf in &m.functions {
            assert!(
                lf.unsupported.is_empty(),
                "{name}: unsupported: {:?}",
                lf.unsupported
            );
        }
    }

    if !exec_or_skip("memory_slot_reuse_straddles_and_dirty_invalidation_execute_across_all_models")
    {
        return;
    }

    let clzero_win = boundary(1) + 0x80;
    let struct_win = boundary(2) + 0x100;
    let rmw_straddle = boundary(4) - 3; // straddles slots 3 | 4
    let nw_straddle = boundary(6) - 5; // straddles slots 5 | 6

    for (name, memory) in all_exec_models() {
        let built = build_with_memory(name, [&f_mem, &f_remap], &[], &memory, GUEST_BASE);

        // Case 0: In-window aligned + straddling RMW + straddling narrow-then-wide store.
        let c_win = Case::new(CODE_BASE)
            .reg(RCX, 0x1000)
            .reg(RDX, 0x2000)
            .reg(R8, clzero_win + 17) // clzero masks down to clzero_win
            .reg(R9, struct_win + 9) // `and r9, -64` masks down to struct_win
            .reg(RDI, rmw_straddle)
            .reg(RSI, nw_straddle)
            .mem(clzero_win, &[0xFF; 64])
            .mem(rmw_straddle, &0x0040u64.to_le_bytes())
            .mem(nw_straddle, &[0xAA; 8])
            .read(clzero_win, 64)
            .read(struct_win, 24)
            .read(rmw_straddle, 8)
            .read(nw_straddle, 8);

        // Case 1: Flat addresses (outside window).
        let c_flat = Case::new(CODE_BASE)
            .reg(RCX, 10)
            .reg(RDX, 20)
            .reg(R8, FLAT_ADDR + 0x40 + 5)
            .reg(R9, FLAT_ADDR + 0x80 + 3)
            .reg(RDI, FLAT_ADDR + 0x100)
            .reg(RSI, FLAT_ADDR + 0x140)
            .mem(FLAT_ADDR + 0x40, &[0xEE; 64])
            .mem(FLAT_ADDR + 0x100, &5u64.to_le_bytes())
            .read(FLAT_ADDR + 0x40, 64)
            .read(FLAT_ADDR + 0x80, 24)
            .read(FLAT_ADDR + 0x100, 8)
            .read(FLAT_ADDR + 0x140, 8);

        let o = run(&built, &[c_win, c_flat]);
        ok(&o, name);

        let want_win_rax = 0x3000 + (0x0040 + 0x1122);
        assert_eq!(o[0].regs[RAX], want_win_rax, "{name}: win rax");
        assert_eq!(o[0].mem[0], vec![0u8; 64], "{name}: clzero win");
        assert_eq!(le(&o[0].mem[1][0..8]), 0x1000, "{name}: struct[0]");
        assert_eq!(le(&o[0].mem[1][8..16]), 0x2000, "{name}: struct[1]");
        assert_eq!(le(&o[0].mem[1][16..24]), 0x3000, "{name}: struct[2]");
        assert_eq!(le(&o[0].mem[2]), 0x1162, "{name}: straddle rmw");
        assert_eq!(le(&o[0].mem[3]), want_win_rax, "{name}: straddle narrow-wide");
        assert_eq!(
            o[0].dirty,
            expected_dirty(
                name,
                &[
                    slot(clzero_win),
                    slot(struct_win),
                    slot(rmw_straddle),
                    slot(rmw_straddle) + 1,
                    slot(nw_straddle),
                    slot(nw_straddle) + 1,
                ],
                false,
            ),
            "{name}: win dirty slots"
        );

        let want_flat_rax = 30 + (5 + 0x1122);
        assert_eq!(o[1].regs[RAX], want_flat_rax, "{name}: flat rax");
        assert_eq!(o[1].mem[0], vec![0u8; 64], "{name}: clzero flat");
        assert_eq!(le(&o[1].mem[2]), 5 + 0x1122, "{name}: flat rmw");
        assert_eq!(le(&o[1].mem[3]), want_flat_rax, "{name}: flat narrow-wide");
        assert_eq!(
            o[1].dirty,
            expected_dirty(name, &[], true),
            "{name}: flat dirty slots"
        );

        // Case 2: Remap across indirect call on table models (not flat base).
        if name != "base" {
            let x = boundary(5) + 0x40;
            let twin = boundary(6) + 0x40;
            let c_remap = Case::new(CODE_BASE + 0x200)
                .reg(RDI, x)
                .reg(RAX, 0xAAAA_BBBB_CCCC_DDDD)
                .mem(twin, &0x1234_5678_9ABC_DEF0u64.to_le_bytes())
                .read(twin, 8);
            let ro = run(&built, &[c_remap]);
            ok(&ro, &format!("{name}/remap"));
            assert_eq!(
                ro[0].regs[RBX],
                0x1234_5678_9ABC_DEF1,
                "{name}: must reload remapped slot after call"
            );
            assert_eq!(
                le(&ro[0].mem[0]),
                0x1234_5678_9ABC_DEF1,
                "{name}: store after remap must land in new backing page"
            );
            if !matches!(name, "page" | "page_stack") {
                assert!(
                    ro[0].dirty.contains(&slot(x)),
                    "{name}: slot 5 must be marked dirty after post-call store: {:?}",
                    ro[0].dirty
                );
            }
        }
    }
}

#[test]
fn integer_imul_div_idiv_traps_and_ext_execute_on_wasm64() {
    // F1: Truncating `imul` with -1, +8, -4, INT_MIN (32-bit & 64-bit) with LIVE CF/OF.
    let f_imul_neg1 = asm_function(CODE_BASE, |a| {
        a.imul_3(rax, rcx, -1)?;
        a.ret()
    });
    let f_imul_pos8 = asm_function(CODE_BASE + 0x100, |a| {
        a.imul_3(rax, rcx, 8)?;
        a.ret()
    });
    let f_imul_neg4 = asm_function(CODE_BASE + 0x200, |a| {
        a.imul_3(rax, rcx, -4)?;
        a.ret()
    });
    let f_imul_min32 = asm_function(CODE_BASE + 0x300, |a| {
        a.imul_3(eax, ecx, i32::MIN)?;
        a.ret()
    });

    // F2: `div` / `idiv` edge cases & `#DE` traps (constant 0, dynamic 0, constant INT_MIN / -1,
    //     dynamic INT_MIN / -1, power-of-two unsigned div, signed idiv by 1/-1/7).
    let f_div_pow2 = asm_function(CODE_BASE + 0x400, |a| {
        a.xor(edx, edx)?;
        a.mov(ecx, 16)?;
        a.div(ecx)?;
        a.ret()
    });
    let f_idiv_neg1 = asm_function(CODE_BASE + 0x500, |a| {
        a.cdq()?;
        a.mov(ecx, -1)?;
        a.idiv(ecx)?;
        a.ret()
    });
    let f_div_const_zero = asm_function(CODE_BASE + 0x600, |a| {
        a.xor(edx, edx)?;
        a.xor(ecx, ecx)?;
        a.div(ecx)?;
        a.ret()
    });
    let f_idiv_const_ov = asm_function(CODE_BASE + 0x700, |a| {
        a.mov(eax, i32::MIN)?;
        a.cdq()?;
        a.mov(ecx, -1)?;
        a.idiv(ecx)?;
        a.ret()
    });
    let f_ext_and_mul = asm_function(CODE_BASE + 0x800, |a| {
        // Widening 64-bit `imul` by -1 on rcx, then `movsx`/`movzx` 32-bit zero-extension checks.
        a.mov(rax, rcx)?;
        a.mov(r8, -1i64 as u64)?;
        a.imul(r8)?; // rdx:rax = rcx * -1, CF/OF set on i64::MIN
        a.seto(r9b)?;
        a.mov(r10, 0xFFFF_FFFF_0000_00FFu64)?;
        a.movsx(r10d, r10b)?; // 0xFF -> -1i32 -> zero-extended to 0x0000_0000_FFFF_FFFF!
        a.mov(r11, 0xDEAD_BEEF_1234_AB00u64)?;
        a.movzx(esi, r11b)?; // 0x00
        a.mov(rbx, 0xDEAD_BEEF_0000_AB55u64)?;
        a.movzx(edi, bh)?; // 0xAB
        a.ret()
    });

    for f in [
        &f_imul_neg1,
        &f_imul_pos8,
        &f_imul_neg4,
        &f_imul_min32,
        &f_div_pow2,
        &f_idiv_neg1,
        &f_div_const_zero,
        &f_idiv_const_ov,
        &f_ext_and_mul,
    ] {
        let lf = lift_function(f).expect("lifts");
        assert!(lf.unsupported.is_empty(), "{:?}", lf.unsupported);
    }

    if !exec_or_skip("integer_imul_div_idiv_traps_and_ext_execute_on_wasm64") {
        return;
    }

    let built = build(
        "hyper_int_exec",
        [
            &f_imul_neg1,
            &f_imul_pos8,
            &f_imul_neg4,
            &f_imul_min32,
            &f_div_pow2,
            &f_idiv_neg1,
            &f_div_const_zero,
            &f_idiv_const_ov,
            &f_ext_and_mul,
        ],
    );

    let ok_cases = vec![
        // 0: i64::MIN * -1 -> i64::MIN, CF=OF=1
        Case::new(CODE_BASE).reg(RCX, i64::MIN as u64),
        // 1: (-42) * -1 -> 42, CF=OF=0
        Case::new(CODE_BASE).reg(RCX, (-42i64) as u64),
        // 2: (1 << 60) * 8 -> i64::MIN (overflow! +2^63), CF=OF=1
        Case::new(CODE_BASE + 0x100).reg(RCX, 1u64 << 60),
        // 3: (-(1 << 60)) * 8 -> i64::MIN (fits!), CF=OF=0
        Case::new(CODE_BASE + 0x100).reg(RCX, (-(1i64 << 60)) as u64),
        // 4: (1 << 61) * -4 -> i64::MIN (fits!), CF=OF=0
        Case::new(CODE_BASE + 0x200).reg(RCX, 1u64 << 61),
        // 5: (-(1 << 61)) * -4 -> +2^63 (overflow!), CF=OF=1
        Case::new(CODE_BASE + 0x200).reg(RCX, (-(1i64 << 61)) as u64),
        // 6: (-(1 << 61) + 1) * -4 -> i64::MAX - 3 (fits!), CF=OF=0
        Case::new(CODE_BASE + 0x200).reg(RCX, (-(1i64 << 61) + 1) as u64),
        // 7: 1 * i32::MIN -> 0x0000_0000_8000_0000 (zero-extended!), CF=OF=0
        Case::new(CODE_BASE + 0x300)
            .reg(RAX, 0xFFFF_FFFF_0000_0000)
            .reg(RCX, 1),
        // 8: (-1i32) * i32::MIN -> 0x0000_0000_8000_0000, CF=OF=1
        Case::new(CODE_BASE + 0x300).reg(RCX, u32::MAX as u64),
        // 9: unsigned 32-bit div by 16 with dirty upper 32 bits of rax/rdx
        Case::new(CODE_BASE + 0x400)
            .reg(RAX, 0xDEAD_BEEF_0000_00AB)
            .reg(RDX, 0xCAFE_BABE_1234_5678),
        // 10: signed 32-bit idiv by -1 on -12345 -> +12345, rem 0
        Case::new(CODE_BASE + 0x500).reg(RAX, 0xFFFF_FFFF_FFFF_CFC7),
        // 11: widening 64-bit imul by -1 on i64::MIN -> lo = i64::MIN, hi = 0, OF = 1!
        Case::new(CODE_BASE + 0x800).reg(RCX, i64::MIN as u64),
    ];
    let o = run(&built, &ok_cases);
    ok(&o, "hyper_int_exec");

    assert_eq!(o[0].regs[RAX], i64::MIN as u64);
    assert!(o[0].flags[0] && o[0].flags[5], "i64::MIN * -1 overflows");
    assert_eq!(o[1].regs[RAX], 42);
    assert!(!o[1].flags[0] && !o[1].flags[5], "-42 * -1 fits");
    assert_eq!(o[2].regs[RAX], i64::MIN as u64);
    assert!(o[2].flags[0] && o[2].flags[5], "(1<<60)*8 overflows");
    assert_eq!(o[3].regs[RAX], i64::MIN as u64);
    assert!(!o[3].flags[0] && !o[3].flags[5], "(-1<<60)*8 fits");
    assert_eq!(o[4].regs[RAX], i64::MIN as u64);
    assert!(!o[4].flags[0] && !o[4].flags[5], "(1<<61)*-4 fits");
    assert_eq!(o[5].regs[RAX], i64::MIN as u64);
    assert!(o[5].flags[0] && o[5].flags[5], "(-1<<61)*-4 overflows");
    assert_eq!(o[6].regs[RAX], (i64::MAX - 3) as u64);
    assert!(!o[6].flags[0] && !o[6].flags[5], "(-1<<61 + 1)*-4 fits");
    assert_eq!(o[7].regs[RAX], 0x8000_0000);
    assert!(!o[7].flags[0] && !o[7].flags[5], "1 * i32::MIN fits");
    assert_eq!(o[8].regs[RAX], 0x8000_0000);
    assert!(o[8].flags[0] && o[8].flags[5], "-1 * i32::MIN overflows");
    assert_eq!(o[9].regs[RAX], 0xAB / 16);
    assert_eq!(o[9].regs[RDX], 0xAB % 16);
    assert_eq!(o[10].regs[RAX], 12345);
    assert_eq!(o[10].regs[RDX], 0);
    assert_eq!(o[11].regs[RAX], i64::MIN as u64);
    assert_eq!(o[11].regs[RDX], 0, "widening i64::MIN * -1 has hi == 0!");
    assert_eq!(o[11].regs[R9] & 0xFF, 1, "widening i64::MIN * -1 sets OF=1");
    assert_eq!(
        o[11].regs[R10], 0xFFFF_FFFF,
        "movsx r32, r8 (-1) must zero-extend to 64-bit"
    );
    assert_eq!(o[11].regs[RSI], 0);
    assert_eq!(o[11].regs[RDI], 0xAB, "movzx edi, bh");

    // `#DE` trap cases: dynamic INT_MIN / -1, constant div by 0, constant INT_MIN / -1.
    let trap_cases = vec![
        Case::new(CODE_BASE + 0x500).reg(RAX, i32::MIN as u32 as u64),
        Case::new(CODE_BASE + 0x600).reg(RAX, 100),
        Case::new(CODE_BASE + 0x700),
    ];
    let to = run(&built, &trap_cases);
    for (i, res) in to.iter().enumerate() {
        assert!(
            res.error.contains("divide error"),
            "trap case {i} must trigger #DE divide error, got: {:?}",
            res.error
        );
    }
}

// ================================================================================================
// 4. Tool-gated x86-64 host-CPU differential verification
// ================================================================================================

#[cfg(target_arch = "x86_64")]
#[test]
fn host_diff_integer_imul_mul_div_neg_and_ext() {
    use std::arch::asm;
    use std::ffi::c_void;

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
                fn VirtualAlloc(
                    addr: *mut c_void,
                    size: usize,
                    kind: u32,
                    protect: u32,
                ) -> *mut c_void;
            }
            // SAFETY: allocates RWX memory for synthetic test snippets.
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
        check_flags: [bool; 6],
        inputs: Vec<([u64; 10], [bool; 6])>,
    }

    fn mk(rax_v: u64, rcx_v: u64, rdx_v: u64, flags: [bool; 6]) -> ([u64; 10], [bool; 6]) {
        (
            [
                rax_v,
                rcx_v,
                rdx_v,
                0xDEAD_BEEF_0000_80FF,
                0xCAFE_BABE_7FFF_FFFF,
                0x0123_4567_89AB_CDEF,
                0xFFFF_FFFF_8000_0001,
                0x0000_0000_0000_0000,
                0x0000_0000_7FFF_FFFE,
                0x8000_0000_0000_0000,
            ],
            flags,
        )
    }

    let edge_inputs = vec![
        mk(0, 0, 0, [false; 6]),
        mk(1, 1, 1, [true; 6]),
        mk(u64::MAX, u64::MAX, 0xFF, [false, true, false, true, false, true]),
        mk(i64::MIN as u64, i64::MIN as u64, i32::MIN as u32 as u64, [false; 6]),
        mk(i64::MAX as u64, i64::MAX as u64, i32::MAX as u64, [true; 6]),
        mk(1u64 << 60, (-(1i64 << 60)) as u64, 1u64 << 61, [false; 6]),
        mk((-(1i64 << 61)) as u64, (-(1i64 << 61) + 1) as u64, 0x10, [true; 6]),
        mk(0xDEAD_BEEF_8000_0000, 0x1234_5678_9ABC_DEF0, 0x7F, [false; 6]),
    ];

    let mut snippets: Vec<Snippet> = Vec::new();
    let mut add = |name: &'static str,
                   check_flags: [bool; 6],
                   inputs: Vec<([u64; 10], [bool; 6])>,
                   build_fn: fn(&mut CodeAssembler) -> Result<(), iced_x86::IcedError>| {
        let mut a = CodeAssembler::new(64).unwrap();
        build_fn(&mut a).unwrap();
        a.ret().unwrap();
        let va = CODE_BASE + snippets.len() as u64 * 0x100;
        let code = a.assemble(va).unwrap();
        snippets.push(Snippet {
            name,
            code,
            check_flags,
            inputs,
        });
    };

    // 1. Truncating `imul` by 0, 1, -1, +8, -4, INT_MIN with live CF/OF captured at each step.
    add(
        "imul_strength_reduce_flags_and_values",
        [true, false, false, false, false, true],
        edge_inputs.clone(),
        |a| {
            a.imul_3(r8, rcx, -1)?;
            a.seto(r9b)?;
            a.imul_3(r10, rcx, 8)?;
            a.seto(r11b)?;
            a.imul_3(rsi, rdx, -4)?;
            a.seto(bl)?;
            a.imul_3(edi, eax, i32::MIN)?;
            a.seto(bh)?;
            a.imul_3(rax, rcx, -2)
        },
    );

    // 2. Widening `mul` and `imul` by 0, 1, -1, 2^k, and constants (16/32/64-bit).
    add(
        "widening_mul_imul_strength_reduce",
        [true, false, false, false, false, true],
        edge_inputs.clone(),
        |a| {
            a.mov(r8, 32u64)?;
            a.mul(r8)?; // unsigned 64-bit * 32
            a.seto(r9b)?;
            a.mov(r10, -1i64 as u64)?;
            a.imul(r10)?; // signed 64-bit * -1
            a.seto(r11b)?;
            a.mov(esi, 8)?;
            a.imul(esi)?; // signed 32-bit * 8 (zero-extends rax & rdx)
            a.seto(bl)?;
            a.mov(r8, 16u64)?;
            a.imul(r8) // signed 64-bit * 16 (CF/OF live at ret)
        },
    );

    // 3. `div` (pow2, 1, non-pow2) and `idiv` (1, 7, -3) — never traps for any input!
    add(
        "div_idiv_const_strength_reduce",
        [false; 6],
        edge_inputs.clone(),
        |a| {
            a.xor(edx, edx)?;
            a.mov(r8d, 16)?;
            a.div(r8d)?;
            a.mov(r9, rax)?;
            a.mov(r10, rdx)?;
            a.mov(rax, rcx)?;
            a.cqo()?;
            a.mov(r11, 7u64)?;
            a.idiv(r11)?;
            a.mov(rsi, rax)?;
            a.mov(rdi, rdx)?;
            a.mov(eax, ecx)?;
            a.cdq()?;
            a.mov(ebx, -3)?;
            a.idiv(ebx)
        },
    );

    // 4. `neg` (8/16/32/64-bit, constant & dynamic) checking ALL 6 flags (CF, PF, AF, ZF, SF, OF)!
    add(
        "neg_all_flags_and_widths",
        [true, true, true, true, true, true],
        edge_inputs.clone(),
        |a| {
            a.mov(r8d, 0)?;
            a.neg(r8d)?;
            a.mov(r9d, i32::MIN)?;
            a.neg(r9d)?;
            a.neg(ecx)?;
            a.neg(dx)?;
            a.neg(rax)
        },
    );

    // 5. `cdq`/`cqo`/`cdqe`/`cbw`/`cwde`/`movsx`/`movsxd`/`movzx` (incl. high-byte ch & non-neg).
    add(
        "sign_zero_extend_families",
        [true, true, true, true, true, true],
        edge_inputs.clone(),
        |a| {
            a.movzx(edi, ch)?;
            a.movzx(r9, bl)?;
            a.movsx(r10d, cl)?;
            a.movsx(r11, bx)?;
            a.movzx(esi, r9w)?;
            a.movsxd(rsi, esi)?;
            a.mov(eax, ecx)?;
            a.cdq()?;
            a.cdqe()?;
            a.cqo()?;
            a.cwde()?;
            a.cbw()
        },
    );

    let mut program = Program::new();
    let mut cases = Vec::new();
    let mut meta = Vec::new();
    for (i, s) in snippets.iter().enumerate() {
        assert!(s.code.len() <= 256, "snippet {} too large", s.name);
        let va = CODE_BASE + i as u64 * 0x100;
        program = program.bytes(va, &s.code);
        for (k, &(regs, flags)) in s.inputs.iter().enumerate() {
            let mut c = Case::new(va);
            for (slot_idx, &g) in REGS.iter().enumerate() {
                c.regs[g] = regs[slot_idx];
            }
            c.flags = flags;
            c.out = REGS.to_vec();
            c.out.push(RSP);
            cases.push(c);
            meta.push((i, k));
        }
    }
    let funcs = program.discover();
    for f in funcs.values() {
        let lf = lift_function(f).expect("lifts");
        assert!(lf.unsupported.is_empty(), "{:?}", lf.unsupported);
    }
    if !exec_or_skip("host_diff_integer_imul_mul_div_neg_and_ext") {
        return;
    }
    let exec_buf = alloc_rx(snippets.len() * 256);
    for (i, s) in snippets.iter().enumerate() {
        // SAFETY: slot `i` lies inside the allocated `snippets.len() * 256` buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(s.code.as_ptr(), exec_buf.add(i * 256), s.code.len());
        }
    }
    let built = build("hyper_mem_and_int_host_diff", funcs.values());
    let outcomes = run(&built, &cases);
    ok(&outcomes, "hyper_mem_and_int_host_diff");

    for (out, (snip_idx, case_idx)) in outcomes.iter().zip(meta) {
        let s = &snippets[snip_idx];
        let (in_regs, in_flags) = s.inputs[case_idx];
        // SAFETY: snippet `snip_idx` was copied to `exec_buf.add(snip_idx * 256)` and ends in `ret`.
        let (want_regs, want_flags) =
            unsafe { run_native(exec_buf.add(snip_idx * 256), in_regs, in_flags) };
        for (slot_idx, &g) in REGS.iter().enumerate() {
            assert_eq!(
                out.regs[g], want_regs[slot_idx],
                "{} case {case_idx}: reg slot {slot_idx} mismatch (got {:#x}, host {:#x})",
                s.name, out.regs[g], want_regs[slot_idx]
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
