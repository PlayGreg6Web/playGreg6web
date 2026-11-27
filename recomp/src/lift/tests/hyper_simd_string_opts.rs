//! Tests for issue #2179:
//! 1. Bulk `rep movs`/`rep stos` fast paths (`recomp/src/lift/string_ops.rs`):
//!    - Constant-count folding, static byte-splat zero-fill (`xor eax, eax; rep stosq`),
//!      non-splat multi-byte contiguous fill loops (`rep stosw/stosd/stosq`), single-slot
//!      in-window fast paths with exact atomic dirty-bit marking, overlap/DF=1/rcx=0 soundness,
//!      and exact final `rcx`/`rsi`/`rdi` register values.
//! 2. Native LLVM vector SIMD lowering (`recomp/src/lift/simd/`):
//!    - Native vector IR for packed min/max/abs/compares/shuffles/blends, `psign*(x, x) -> abs`,
//!      `vpsllv`/`vpsrlv` mask simplification, same-reg `pmulhw`/`pmulhuw`/`minps`/`maxps`/logic/
//!      float compares, direct 32-bit `movd` lane insert, and zero-vector algebraic folding with
//!      100% bit-exact NaN/`-0.0`/`INT_MIN`/VEX-128 upper-half zeroing semantics.
//! 3. Tool-gated Wasm64 execution across memory models (`base`, `window`, `window_select`,
//!    `window_outline`, `window_runs`, `page`) and x86-64 host-CPU differential verification.

use iced_x86::code_asm::*;

use super::asm_function;
use super::exec::{
    CODE_BASE, Case, GUEST_BASE, HarnessMemory, Outcome, R8, R9, RAX, RCX, RDI, RDX, RSI, RSP,
    STACK_TOP, STATE_ADDR, WINDOW_BASE, WINDOW_SIZE, build, build_with_memory, exec_or_skip, run,
};
use crate::lift::memory::{Window, WindowCheck};
use crate::lift::target::LiftTarget;
use crate::lift::{
    BULK_MEMORY_TARGET_FEATURES, GuestBase, MemoryModel, STATE_FLAGS_OFFSET, lift_function,
    lift_module, lift_module_with,
};

const FLAT_ADDR: u64 = 0x28_0000;
const DF_ADDR: u64 = STATE_ADDR + STATE_FLAGS_OFFSET + 6;

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

fn lift_module_ir(
    model: MemoryModel,
    build_fn: impl FnOnce(&mut CodeAssembler) -> Result<(), iced_x86::IcedError>,
) -> String {
    let f = asm_function(0x1000, build_fn);
    let module = lift_module_with([&f], model).expect("lifts");
    assert!(
        module.functions[0].unsupported.is_empty(),
        "{:?}",
        module.functions[0].unsupported
    );
    module.ir
}

fn window_model(check: WindowCheck, stack_flat: bool, contiguous_runs: bool) -> MemoryModel {
    MemoryModel::Windowed(Window {
        base: WINDOW_BASE,
        size: WINDOW_SIZE,
        flat: GuestBase::Symbol,
        check,
        stack_flat,
        contiguous_runs,
    })
}

fn ok(outcomes: &[Outcome], what: &str) {
    for (i, o) in outcomes.iter().enumerate() {
        assert!(o.error.is_empty(), "{what}: case {i} trapped: {}", o.error);
        assert_eq!(o.regs[RSP], STACK_TOP, "{what}: case {i}: rsp");
    }
}

fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(37).wrapping_add(seed) | 1)
        .collect()
}

fn exec_models() -> Vec<(&'static str, HarnessMemory)> {
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
            "window_runs",
            HarnessMemory::window_full(
                &HarnessMemory::contiguous_slots(),
                WindowCheck::Branch,
                true,
                true,
            ),
        ),
        ("page", HarnessMemory::page(&HarnessMemory::standard_slots())),
    ]
}

fn expected_dirty(model: &str, slots: &[u64]) -> Vec<u64> {
    match model {
        "base" | "page" => Vec::new(),
        _ => {
            let mut v: Vec<u64> = slots.to_vec();
            v.sort_unstable();
            v.dedup();
            v
        }
    }
}

// ================================================================================================
// 1. IR-shape tests for bulk `rep movs` / `rep stos` fast paths
// ================================================================================================

#[test]
fn bulk_rep_movs_and_stos_ir_shapes() {
    // 1. `xor eax, eax; rep stosq` (static zero byte-splat):
    //    - emits `@llvm.memset.p0.i64(..., i8 0, ...)` directly
    //    - zero runtime splat-check `mul i64` or non-splat `rep_fill` loop
    //    - carries `+bulk-memory` and `declare void @llvm.memset.p0.i64`
    let zero_stosq = lift_module_ir(MemoryModel::default(), |a| {
        a.xor(eax, eax)?;
        a.rep().stosq()?;
        a.ret()
    });
    assert!(
        zero_stosq.contains("call void @llvm.memset.p0.i64(")
            && zero_stosq.contains(", i8 0, i64 "),
        "zero-fill rep stosq must emit direct memset with i8 0:\n{zero_stosq}"
    );
    assert!(
        zero_stosq.contains("declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)"),
        "module must declare llvm.memset:\n{zero_stosq}"
    );
    assert!(
        zero_stosq.contains(BULK_MEMORY_TARGET_FEATURES),
        "module must enable +bulk-memory:\n{zero_stosq}"
    );
    assert!(
        !zero_stosq.contains("mul i64 ") && !zero_stosq.contains("rep_fill"),
        "static zero rep stosq must skip runtime splat check and non-splat fill loop:\n{zero_stosq}"
    );

    // 2. `mov eax, 0x3F800000; rep stosd` (static non-splat fill, e.g. 1.0f buffer fill):
    //    - emits tight contiguous `rep_fill` loop with `getelementptr i32` + `store i32`
    //    - skips unreachable `@llvm.memset` and runtime splat-check `mul`
    let float_stosd = lift_module_ir(MemoryModel::default(), |a| {
        a.mov(eax, 0x3F80_0000u32 as i32)?;
        a.rep().stosd()?;
        a.ret()
    });
    assert!(
        float_stosd.contains("rep_fill")
            && float_stosd.contains("getelementptr i32, ptr ")
            && float_stosd.contains("store i32 1065353216, ptr "),
        "non-splat rep stosd must emit tight contiguous fill loop:\n{float_stosd}"
    );
    assert!(
        !float_stosd.contains("@llvm.memset") && !float_stosd.contains("mul i32 "),
        "static non-splat rep stosd must not emit memset or splat-check mul:\n{float_stosd}"
    );

    // 3. Dynamic `rep stosq` under Base and Windowed models:
    //    - dispatches both `@llvm.memset.p0.i64` (splat) and `rep_fill` (non-splat)
    //    - under Windowed, emits single-slot in-window fast paths with short-mask table lookup
    //      (`getelementptr i8, ptr @__playgta6_window_table`) and atomic dirty mark
    //      (`@__playgta6_window_dirty`), plus `@__playgta6_guest_memset` for multi-slot straddles.
    let dyn_base = lift_module_ir(MemoryModel::default(), |a| {
        a.rep().stosq()?;
        a.ret()
    });
    assert!(
        dyn_base.contains("call void @llvm.memset.p0.i64(")
            && dyn_base.contains("rep_fill")
            && dyn_base.contains("getelementptr i64, ptr "),
        "dynamic rep stosq must emit both memset and tight contiguous fill loop:\n{dyn_base}"
    );
    let dyn_win = lift_module_ir(window_model(WindowCheck::Branch, true, false), |a| {
        a.rep().stosq()?;
        a.ret()
    });
    assert!(
        dyn_win.contains("call void @llvm.memset.p0.i64(")
            && dyn_win.contains("call void @__playgta6_guest_memset(")
            && dyn_win.contains("getelementptr i8, ptr @__playgta6_window_table")
            && dyn_win.contains("getelementptr i8, ptr @__playgta6_window_dirty")
            && dyn_win.contains("store atomic i8 1, ptr ")
            && dyn_win.contains("rep_fill"),
        "windowed rep stosq must emit flat + single-slot window memset/fill + dirty mark:\n{dyn_win}"
    );

    // 4. Constant-count `mov ecx, 16; rep movsq` folds byte length `128` without `shl`/`icmp ule`.
    let const_movsq = lift_module_ir(MemoryModel::default(), |a| {
        a.mov(ecx, 16)?;
        a.rep().movsq()?;
        a.ret()
    });
    assert!(
        const_movsq.contains("call void @llvm.memmove.p0.p0.i64(")
            && const_movsq.contains(", i64 128, i1 false)"),
        "constant-count rep movsq must fold byte length to 128:\n{const_movsq}"
    );
    assert!(
        !const_movsq.contains("shl i64 "),
        "constant-count rep movsq needs no dynamic shift:\n{const_movsq}"
    );

    // 5. `%state` store-backs for rcx/rsi/rdi remain intact (LIFTER_STATE_CONTRACT), and byte-size
    //    `rep movsb` elides the redundant `icmp ule i64 ..., -1` overflow check.
    let movsb_ir = lift_ir(|a| {
        a.rep().movsb()?;
        a.ret()
    });
    assert!(
        movsb_ir.contains(", ptr %state.gpr1")
            && movsb_ir.contains(", ptr %state.gpr6")
            && movsb_ir.contains(", ptr %state.gpr7"),
        "%state store-backs for rcx/rsi/rdi must remain intact:\n{movsb_ir}"
    );
    assert!(
        !movsb_ir.contains("icmp ule i64 "),
        "byte-width rep movsb needs no overflow check against u64::MAX:\n{movsb_ir}"
    );
}

// ================================================================================================
// 2. IR-shape tests for native LLVM vector SIMD lowering
// ================================================================================================

#[test]
fn native_vector_simd_lowering_ir_shapes() {
    let cases: Vec<(&str, crate::cfg::Function)> = vec![
        (
            "int_minmax_abs_cmp",
            asm_function(0x1000, |a| {
                a.pminub(xmm0, xmm1)?;
                a.pmaxub(xmm0, xmm2)?;
                a.pminsw(xmm0, xmm1)?;
                a.pmaxsw(xmm0, xmm2)?;
                a.pminsb(xmm0, xmm1)?;
                a.pmaxsb(xmm0, xmm2)?;
                a.pminsd(xmm0, xmm1)?;
                a.pmaxsd(xmm0, xmm2)?;
                a.pminuw(xmm0, xmm1)?;
                a.pmaxuw(xmm0, xmm2)?;
                a.pminud(xmm0, xmm1)?;
                a.pmaxud(xmm0, xmm2)?;
                a.pabsb(xmm3, xmm0)?;
                a.pabsw(xmm4, xmm0)?;
                a.pabsd(xmm5, xmm0)?;
                a.pcmpeqb(xmm6, xmm1)?;
                a.pcmpgtd(xmm7, xmm2)?;
                a.ret()
            }),
        ),
        (
            "psign_same_reg_to_abs",
            asm_function(0x1100, |a| {
                a.psignb(xmm0, xmm0)?;
                a.psignw(xmm1, xmm1)?;
                a.psignd(xmm2, xmm2)?;
                a.vpsignd(ymm3, ymm4, ymm4)?;
                a.ret()
            }),
        ),
        (
            "var_shift_mask",
            asm_function(0x1200, |a| {
                a.vpsllvd(xmm0, xmm1, xmm2)?;
                a.vpsrlvq(ymm3, ymm4, ymm5)?;
                a.ret()
            }),
        ),
        (
            "pmulhw_same_reg",
            asm_function(0x1300, |a| {
                a.pmulhw(xmm0, xmm0)?;
                a.vpmulhuw(xmm1, xmm2, xmm2)?;
                a.ret()
            }),
        ),
        (
            "shuffles_and_blends",
            asm_function(0x1400, |a| {
                a.pshufd(xmm0, xmm1, 0x1B)?;
                a.pshuflw(xmm2, xmm1, 0xB1)?;
                a.pshufhw(xmm3, xmm1, 0x4E)?;
                a.shufps(xmm4, xmm5, 0x88)?;
                a.shufpd(xmm6, xmm7, 0x01)?;
                a.unpcklps(xmm0, xmm2)?;
                a.punpckhbw(xmm3, xmm4)?;
                a.pblendw(xmm5, xmm6, 0x55)?;
                a.blendps(xmm7, xmm0, 0x0A)?;
                a.blendpd(xmm1, xmm2, 0x02)?;
                a.pblendvb(xmm3, xmm4)?;
                a.blendvps(xmm5, xmm6)?;
                a.blendvpd(xmm7, xmm1)?;
                a.ret()
            }),
        ),
        (
            "float_minmax_and_cmp_exact",
            asm_function(0x1500, |a| {
                // Distinct operands: must keep exact x86 `fcmp olt/ogt` + `select` order!
                a.minps(xmm0, xmm1)?;
                a.maxpd(xmm2, xmm3)?;
                // Same-register min/max: identity (zero fcmp/select)!
                a.vminps(xmm4, xmm5, xmm5)?;
                a.vmaxsd(xmm6, xmm7, xmm7)?;
                // VEX FALSE_OQ (11), TRUE_UQ (15), and same-reg OLT (1) / UEQ (8): constant!
                a.vcmpps(xmm8, xmm0, xmm1, 11)?;
                a.vcmppd(xmm9, xmm2, xmm3, 15)?;
                a.vcmpps(xmm10, xmm0, xmm0, 1)?;
                a.vcmpps(xmm11, xmm0, xmm0, 8)?;
                // NaN-sensitive same-reg OEQ (0) MUST keep `fcmp oeq`!
                a.vcmpps(xmm12, xmm0, xmm0, 0)?;
                a.ret()
            }),
        ),
        (
            "movd_and_zero_vec_folding",
            asm_function(0x1600, |a| {
                a.movd(xmm0, eax)?;
                a.vmovd(xmm1, dword_ptr(rcx))?;
                a.vxorps(xmm2, xmm2, xmm2)?;
                a.vpaddd(xmm3, xmm0, xmm2)?; // + 0 -> xmm0
                a.vpmulld(xmm4, xmm0, xmm2)?; // * 0 -> 0
                a.vpminub(xmm5, xmm0, xmm2)?; // umin(x, 0) -> 0
                a.vpmaxub(xmm6, xmm0, xmm2)?; // umax(x, 0) -> xmm0
                a.vpshufd(xmm7, xmm2, 0x1B)?; // shuffle(0) -> 0
                a.vblendvps(xmm8, xmm0, xmm1, xmm2)?; // zero mask -> xmm0
                a.vpmovsxbd(xmm9, xmm2)?; // sext(0) -> 0
                a.vbroadcastss(ymm10, xmm2)?; // broadcast(0) -> 0
                a.vcvtdq2ps(xmm11, xmm2)?; // sitofp(0) -> 0
                a.pand(xmm0, xmm0)?; // x & x -> x
                a.por(xmm1, xmm1)?; // x | x -> x
                a.movss(xmm3, xmm3)?; // identity
                a.ret()
            }),
        ),
    ];

    let module = lift_module(cases.iter().map(|(_, f)| f)).expect("lift module");
    for ((name, _), f) in cases.iter().zip(&module.functions) {
        assert!(f.unsupported.is_empty(), "{name}: {:?}", f.unsupported);
        let ir = &f.ir;
        match *name {
            "int_minmax_abs_cmp" => {
                assert!(
                    !ir.contains("extractelement") && !ir.contains("insertelement"),
                    "int min/max/abs/cmp must be 100% vector IR:\n{ir}"
                );
                for intr in [
                    "@llvm.umin.v16i8(",
                    "@llvm.umax.v16i8(",
                    "@llvm.smin.v8i16(",
                    "@llvm.smax.v8i16(",
                    "@llvm.smin.v16i8(",
                    "@llvm.smax.v16i8(",
                    "@llvm.smin.v4i32(",
                    "@llvm.smax.v4i32(",
                    "@llvm.umin.v8i16(",
                    "@llvm.umax.v8i16(",
                    "@llvm.umin.v4i32(",
                    "@llvm.umax.v4i32(",
                    "@llvm.abs.v16i8(",
                    "@llvm.abs.v8i16(",
                    "@llvm.abs.v4i32(",
                ] {
                    assert!(ir.contains(intr), "missing {intr}:\n{ir}");
                }
                assert!(
                    ir.contains("icmp eq <16 x i8>")
                        && ir.contains("sext <16 x i1> ")
                        && ir.contains("icmp sgt <4 x i32>")
                        && ir.contains("sext <4 x i1> "),
                    "pcmpeq/pcmpgt must use vector icmp + sext:\n{ir}"
                );
            }
            "psign_same_reg_to_abs" => {
                assert!(
                    ir.contains("@llvm.abs.v16i8(")
                        && ir.contains("@llvm.abs.v8i16(")
                        && ir.contains("@llvm.abs.v4i32("),
                    "psign*(x, x) must lower to @llvm.abs:\n{ir}"
                );
                assert!(
                    !ir.contains("select ") && !ir.contains("icmp "),
                    "psign*(x, x) needs zero icmp/select:\n{ir}"
                );
            }
            "var_shift_mask" => {
                // 1 select per lane (3 total across 128-bit vpsllvd + 256-bit vpsrlvq), not 2!
                assert_eq!(
                    ir.matches("select ").count(),
                    3,
                    "vpsllv/vpsrlv should mask count with `and` (1 select per 128-bit lane):\n{ir}"
                );
                assert!(
                    ir.contains("and <4 x i32>") && ir.contains("and <2 x i64>"),
                    "vpsllv/vpsrlv must use `and` count mask:\n{ir}"
                );
            }
            "pmulhw_same_reg" => {
                // Each same-reg pmulhw/vpmulhuw needs 2 half-shuffles + 1 pack shuffle = 3 total,
                // so 6 total across both instructions (instead of 10 with duplicate b shuffles).
                assert_eq!(
                    ir.matches("shufflevector ").count(),
                    6,
                    "same-reg pmulhw/vpmulhuw must deduplicate source shuffles:\n{ir}"
                );
            }
            "shuffles_and_blends" => {
                assert!(
                    !ir.contains("extractelement") && !ir.contains("insertelement"),
                    "shuffles and blends must be 100% native vector IR:\n{ir}"
                );
                assert!(
                    ir.matches("shufflevector ").count() >= 10,
                    "expected native shufflevector lowering:\n{ir}"
                );
                assert!(
                    ir.contains("@llvm.wasm.bitselect.v16i8(")
                        && ir.contains("@llvm.wasm.bitselect.v4i32(")
                        && ir.contains("@llvm.wasm.bitselect.v2i64("),
                    "blendv* must use native vector bitselect:\n{ir}"
                );
            }
            "float_minmax_and_cmp_exact" => {
                assert!(
                    ir.contains("fcmp olt <4 x float>") && ir.contains("fcmp ogt <2 x double>"),
                    "distinct minps/maxpd must keep exact fcmp olt/ogt:\n{ir}"
                );
                // Only 3 fcmp instructions total: minps, maxpd, and NaN-sensitive `vcmpps oeq`!
                assert_eq!(
                    ir.matches("fcmp ").count(),
                    3,
                    "same-reg min/max and constant/tautology vcmpps must fold cleanly:\n{ir}"
                );
                assert!(
                    ir.contains("fcmp oeq <4 x float>"),
                    "NaN-sensitive vcmpps oeq must NOT fold away:\n{ir}"
                );
            }
            "movd_and_zero_vec_folding" => {
                assert!(
                    ir.contains("insertelement <4 x i32> zeroinitializer, i32 "),
                    "movd xmm, r32/m32 must insert directly into <4 x i32>:\n{ir}"
                );
                assert!(
                    !ir.contains("zext i32 "),
                    "movd xmm, r32/m32 must not zext i32 to i64:\n{ir}"
                );
                assert!(
                    !ir.contains("add <4 x i32>")
                        && !ir.contains("mul <4 x i32>")
                        && !ir.contains("@llvm.umin.")
                        && !ir.contains("@llvm.umax.")
                        && !ir.contains("shufflevector ")
                        && !ir.contains("@llvm.wasm.bitselect")
                        && !ir.contains("sext <4 x i8>")
                        && !ir.contains("sitofp ")
                        && !ir.contains("and <2 x i64>")
                        && !ir.contains("or <2 x i64>"),
                    "zero-vector and same-reg identities must fold cleanly:\n{ir}"
                );
                // VEX-128 upper-half zeroing must still be emitted!
                assert!(
                    ir.contains("store <2 x i64> zeroinitializer, ptr %state.ymmh1, align 8"),
                    "VEX-128 upper-half zeroing must remain intact:\n{ir}"
                );
            }
            _ => {}
        }
    }

    // Native x86-64 retargeting must remain free of wasm-only intrinsics.
    let native_ir = LiftTarget::native_host().retarget(module.ir);
    assert!(
        !native_ir.contains("llvm.wasm."),
        "wasm-only intrinsic leaked into native IR"
    );
}

// ================================================================================================
// 3. Tool-gated Wasm64 execution tests across all memory models (including window dirty marks)
// ================================================================================================

#[test]
fn bulk_rep_movs_and_stos_execute_across_all_models_and_edge_cases() {
    if !exec_or_skip("bulk_rep_movs_and_stos_execute_across_all_models_and_edge_cases") {
        return;
    }

    let f_movsb = asm_function(CODE_BASE, |a| {
        a.rep().movsb()?;
        a.ret()
    });
    let f_movsq = asm_function(CODE_BASE + 0x100, |a| {
        a.rep().movsq()?;
        a.ret()
    });
    let f_stosb = asm_function(CODE_BASE + 0x200, |a| {
        a.rep().stosb()?;
        a.ret()
    });
    let f_stosq_dyn = asm_function(CODE_BASE + 0x300, |a| {
        a.rep().stosq()?;
        a.ret()
    });
    let f_stosq_zero = asm_function(CODE_BASE + 0x400, |a| {
        a.xor(eax, eax)?;
        a.rep().stosq()?;
        a.ret()
    });
    let f_stosd_const = asm_function(CODE_BASE + 0x500, |a| {
        a.mov(eax, 0x3F80_0000u32 as i32)?;
        a.rep().stosd()?;
        a.ret()
    });
    let f_stosw_dyn = asm_function(CODE_BASE + 0x600, |a| {
        a.rep().stosw()?;
        a.ret()
    });

    let funcs = [
        &f_movsb,
        &f_movsq,
        &f_stosb,
        &f_stosq_dyn,
        &f_stosq_zero,
        &f_stosd_const,
        &f_stosw_dyn,
    ];

    let flat_src = FLAT_ADDR;
    let flat_dst = FLAT_ADDR + 0x2000;
    let flat_ovl = FLAT_ADDR + 0x4000;
    let win_single_a = boundary(2) + 0x100; // strictly inside slot 2
    let win_single_b = boundary(3) + 0x200; // strictly inside slot 3
    let win_straddle_a = boundary(5) - 0x20; // straddles slots 4 | 5
    let win_straddle_b = boundary(7) - 0x10; // straddles slots 6 | 7

    let src_256 = pattern(256, 17);
    let mut ovl_init = vec![0xCCu8; 64];
    ovl_init[0] = 0xAB;
    ovl_init[1] = 0xCD;
    let mut back_init = pattern(64, 5);
    let back_expected = {
        let mut buf = back_init.clone();
        for i in 0..32 {
            buf[i] = buf[i + 5];
        }
        buf
    };
    let mut df1_src_init = pattern(64, 9);
    let df1_expected = {
        let mut buf = df1_src_init.clone();
        // rep movsb DF=1: src=base+31, dst=base+34, rcx=16 -> replicates [31..34] downwards
        for i in 0..16 {
            buf[34 - i] = buf[31 - i];
        }
        buf
    };
    // Avoid unused mut warnings
    back_init[0] = back_init[0];
    df1_src_init[0] = df1_src_init[0];

    let qword_pat: u64 = 0x0123_4567_89AB_CDEF;
    let word_pat: u16 = 0xBEEF;
    let f32_one: u32 = 0x3F80_0000;
    let with_df = |c: Case, df: bool| c.mem(DF_ADDR, &[u8::from(df)]);

    for (name, memory) in exec_models() {
        let built = build_with_memory(name, funcs, &[], &memory, GUEST_BASE);
        let cases = vec![
            // 0: rcx == 0 no-op on rep movsb with unmapped/bogus addresses (must not fault!)
            with_df(Case::new(CODE_BASE), false)
                .reg(RSI, 0xFFFF_0000_0000_0000)
                .reg(RDI, 0xFFFF_0000_0000_1000)
                .reg(RCX, 0),
            // 1: rcx == 0 no-op on rep stosq with unmapped/bogus address
            with_df(Case::new(CODE_BASE + 0x300), false)
                .reg(RDI, 0xFFFF_0000_0000_2000)
                .reg(RAX, qword_pat)
                .reg(RCX, 0),
            // 2: DF == 0 forward non-overlapping rep movsq (256 bytes = 32 qwords)
            with_df(Case::new(CODE_BASE + 0x100), false)
                .reg(RSI, flat_src)
                .reg(RDI, flat_dst)
                .reg(RCX, 32)
                .mem(flat_src, &src_256)
                .read(flat_dst, 256),
            // 3: DF == 0 forward overlap `dst = src + 1` (MUST replicate byte 0 via scalar loop!)
            with_df(Case::new(CODE_BASE), false)
                .reg(RSI, flat_ovl)
                .reg(RDI, flat_ovl + 1)
                .reg(RCX, 31)
                .mem(flat_ovl, &ovl_init)
                .read(flat_ovl, 32),
            // 4: DF == 0 backward overlap `dst = src - 5` (safe memmove direction!)
            with_df(Case::new(CODE_BASE), false)
                .reg(RSI, flat_ovl + 5)
                .reg(RDI, flat_ovl)
                .reg(RCX, 32)
                .mem(flat_ovl, &back_init)
                .read(flat_ovl, 64),
            // 5: DF == 1 downwards overlap `dst = src + 3` (replicates downwards!)
            with_df(Case::new(CODE_BASE), true)
                .reg(RSI, flat_ovl + 31)
                .reg(RDI, flat_ovl + 34)
                .reg(RCX, 16)
                .mem(flat_ovl, &df1_src_init)
                .read(flat_ovl, 64),
            // 6: Flat static zero-fill `xor eax, eax; rep stosq` (64 qwords = 512 bytes)
            with_df(Case::new(CODE_BASE + 0x400), false)
                .reg(RDI, flat_dst)
                .reg(RAX, 0xDEAD_BEEF_CAFE_BABE)
                .reg(RCX, 64)
                .mem(flat_dst, &vec![0xFF; 520])
                .read(flat_dst, 520),
            // 7: Flat static non-splat `mov eax, 0x3F800000; rep stosd` (32 dwords = 128 bytes)
            with_df(Case::new(CODE_BASE + 0x500), false)
                .reg(RDI, flat_dst)
                .reg(RCX, 32)
                .mem(flat_dst, &vec![0x11; 132])
                .read(flat_dst, 132),
            // 8: Flat dynamic non-splat `rep stosq` with DF == 1 (16 qwords downwards)
            with_df(Case::new(CODE_BASE + 0x300), true)
                .reg(RDI, flat_dst + 15 * 8)
                .reg(RAX, qword_pat)
                .reg(RCX, 16)
                .mem(flat_dst, &vec![0x22; 136])
                .read(flat_dst, 136),
            // 9: Single-slot in-window `rep stosb` (marks slot 2 only!)
            with_df(Case::new(CODE_BASE + 0x200), false)
                .reg(RDI, win_single_a)
                .reg(RAX, 0x7E)
                .reg(RCX, 128)
                .read(win_single_a, 128),
            // 10: Single-slot in-window non-splat `rep stosq` (marks slot 3 only!)
            with_df(Case::new(CODE_BASE + 0x300), false)
                .reg(RDI, win_single_b)
                .reg(RAX, qword_pat)
                .reg(RCX, 24)
                .read(win_single_b, 24 * 8),
            // 11: Multi-slot straddling in-window zero-fill `rep stosq` (marks slots 4 & 5!)
            with_df(Case::new(CODE_BASE + 0x400), false)
                .reg(RDI, win_straddle_a)
                .reg(RCX, 16)
                .mem(win_straddle_a, &vec![0xDD; 128])
                .read(win_straddle_a, 128),
            // 12: Multi-slot straddling in-window non-splat `rep stosw` (marks slots 6 & 7!)
            with_df(Case::new(CODE_BASE + 0x600), false)
                .reg(RDI, win_straddle_b)
                .reg(RAX, u64::from(word_pat))
                .reg(RCX, 20)
                .read(win_straddle_b, 40),
        ];

        let o = run(&built, &cases);
        ok(&o, name);

        // 0 & 1: rcx == 0 no-op preserves registers and marks nothing
        assert_eq!(o[0].regs[RCX], 0, "{name}: case 0 rcx");
        assert_eq!(o[0].regs[RSI], 0xFFFF_0000_0000_0000, "{name}: case 0 rsi");
        assert_eq!(o[0].regs[RDI], 0xFFFF_0000_0000_1000, "{name}: case 0 rdi");
        assert_eq!(o[1].regs[RCX], 0, "{name}: case 1 rcx");
        assert_eq!(o[1].regs[RDI], 0xFFFF_0000_0000_2000, "{name}: case 1 rdi");

        // 2: non-overlapping rep movsq
        assert_eq!(o[2].mem[0], src_256, "{name}: rep movsq bytes");
        assert_eq!(o[2].regs[RCX], 0, "{name}: rep movsq rcx");
        assert_eq!(o[2].regs[RSI], flat_src + 256, "{name}: rep movsq rsi");
        assert_eq!(o[2].regs[RDI], flat_dst + 256, "{name}: rep movsq rdi");
        assert!(o[2].dirty.is_empty(), "{name}: flat movsq marks nothing");

        // 3: forward overlap `dst = src + 1` replicates 0xAB
        assert_eq!(
            o[3].mem[0],
            vec![0xAB; 32],
            "{name}: forward overlap must replicate pattern byte"
        );
        assert_eq!(o[3].regs[RCX], 0);
        assert_eq!(o[3].regs[RSI], flat_ovl + 31);
        assert_eq!(o[3].regs[RDI], flat_ovl + 32);

        // 4: backward overlap `dst = src - 5`
        assert_eq!(o[4].mem[0], back_expected, "{name}: backward overlap");
        assert_eq!(o[4].regs[RCX], 0);
        assert_eq!(o[4].regs[RSI], flat_ovl + 37);
        assert_eq!(o[4].regs[RDI], flat_ovl + 32);

        // 5: DF == 1 downwards overlap
        assert_eq!(o[5].mem[0], df1_expected, "{name}: DF=1 overlap");
        assert_eq!(o[5].regs[RCX], 0);
        assert_eq!(o[5].regs[RSI], flat_ovl + 15);
        assert_eq!(o[5].regs[RDI], flat_ovl + 18);

        // 6: flat static zero-fill
        assert!(o[6].mem[0][..512].iter().all(|&b| b == 0), "{name}: zero");
        assert_eq!(&o[6].mem[0][512..], &[0xFF; 8], "{name}: zero guard");
        assert_eq!(o[6].regs[RDI], flat_dst + 512);
        assert_eq!(o[6].regs[RCX], 0);
        assert!(o[6].dirty.is_empty(), "{name}: flat zero marks nothing");

        // 7: flat static non-splat f32 1.0 fill
        let want_f32: Vec<u8> = (0..32).flat_map(|_| f32_one.to_le_bytes()).collect();
        assert_eq!(&o[7].mem[0][..128], &want_f32[..], "{name}: f32 fill");
        assert_eq!(&o[7].mem[0][128..], &[0x11; 4], "{name}: f32 guard");
        assert_eq!(o[7].regs[RDI], flat_dst + 128);
        assert_eq!(o[7].regs[RCX], 0);
        assert!(o[7].dirty.is_empty(), "{name}: flat f32 marks nothing");

        // 8: flat dynamic non-splat qword DF=1 fill
        let want_q16: Vec<u8> = (0..16).flat_map(|_| qword_pat.to_le_bytes()).collect();
        assert_eq!(&o[8].mem[0][..128], &want_q16[..], "{name}: df1 qword fill");
        assert_eq!(&o[8].mem[0][128..], &[0x22; 8], "{name}: df1 guard");
        assert_eq!(o[8].regs[RDI], flat_dst - 8);
        assert_eq!(o[8].regs[RCX], 0);

        // 9: single-slot in-window rep stosb
        assert_eq!(o[9].mem[0], vec![0x7E; 128], "{name}: win stosb");
        assert_eq!(o[9].regs[RDI], win_single_a + 128);
        assert_eq!(
            o[9].dirty,
            expected_dirty(name, &[slot(win_single_a)]),
            "{name}: win stosb dirty"
        );

        // 10: single-slot in-window non-splat rep stosq
        let want_q24: Vec<u8> = (0..24).flat_map(|_| qword_pat.to_le_bytes()).collect();
        assert_eq!(o[10].mem[0], want_q24, "{name}: win non-splat stosq");
        assert_eq!(o[10].regs[RDI], win_single_b + 24 * 8);
        assert_eq!(
            o[10].dirty,
            expected_dirty(name, &[slot(win_single_b)]),
            "{name}: win non-splat stosq dirty"
        );

        // 11: multi-slot straddling in-window zero-fill rep stosq
        assert_eq!(o[11].mem[0], vec![0u8; 128], "{name}: straddle zero stosq");
        assert_eq!(o[11].regs[RDI], win_straddle_a + 128);
        assert_eq!(
            o[11].dirty,
            expected_dirty(name, &[slot(win_straddle_a), slot(win_straddle_a) + 1]),
            "{name}: straddle zero stosq dirty"
        );

        // 12: multi-slot straddling in-window non-splat rep stosw
        let want_w20: Vec<u8> = (0..20).flat_map(|_| word_pat.to_le_bytes()).collect();
        assert_eq!(o[12].mem[0], want_w20, "{name}: straddle non-splat stosw");
        assert_eq!(o[12].regs[RDI], win_straddle_b + 40);
        assert_eq!(
            o[12].dirty,
            expected_dirty(name, &[slot(win_straddle_b), slot(win_straddle_b) + 1]),
            "{name}: straddle non-splat stosw dirty"
        );
    }
}

#[test]
fn upgraded_simd_edge_cases_execute_on_wasm64() {
    if !exec_or_skip("upgraded_simd_edge_cases_execute_on_wasm64") {
        return;
    }

    // Tests INT_MIN for pabs* & psign*(x, x), signed/unsigned min/max boundaries, NaN & -0.0 for
    // minps/maxps/vcmpps, vpsllvd/vpsrlvd overshifts, movd 32-bit zero-extension, zero-vector
    // folding chains, and VEX-128 upper-half zeroing.
    let f = asm_function(CODE_BASE, |a| {
        // 1. pabsd & psignd(x, x) on [INT_MIN, -1, 0, INT_MAX]
        a.vmovdqu(xmm0, xmmword_ptr(rcx))?;
        a.vpabsd(xmm1, xmm0)?;
        a.vpsignd(xmm2, xmm0, xmm0)?;
        a.vmovdqu(xmmword_ptr(rdx), xmm1)?;
        a.vmovdqu(xmmword_ptr(rdx + 16), xmm2)?;

        // 2. minps / maxps with NaN, -0.0, +0.0 (second-operand x86 rule!) & same-reg identity
        a.vmovups(xmm3, xmmword_ptr(rcx + 16))?;
        a.vmovups(xmm4, xmmword_ptr(rcx + 32))?;
        a.vminps(xmm5, xmm3, xmm4)?;
        a.vmaxps(xmm6, xmm3, xmm4)?;
        a.vminps(xmm7, xmm3, xmm3)?;
        a.vmovups(xmmword_ptr(rdx + 32), xmm5)?;
        a.vmovups(xmmword_ptr(rdx + 48), xmm6)?;
        a.vmovups(xmmword_ptr(rdx + 64), xmm7)?;

        // 3. vpsllvd / vpsrlvd with counts [0, 1, 31, 32]
        a.vmovdqu(xmm8, xmmword_ptr(rcx + 48))?;
        a.vmovdqu(xmm9, xmmword_ptr(rcx + 64))?;
        a.vpsllvd(xmm10, xmm8, xmm9)?;
        a.vpsrlvd(xmm11, xmm8, xmm9)?;
        a.vmovdqu(xmmword_ptr(rdx + 80), xmm10)?;
        a.vmovdqu(xmmword_ptr(rdx + 96), xmm11)?;

        // 4. movd xmm, r32 + zero-vector fold chain + VEX-128 upper-half zeroing check
        a.vmovdqu(ymm12, ymmword_ptr(rcx))?; // populate upper 128 bits with non-zero bytes
        a.vmovd(xmm12, r8d)?; // VEX-128: lane 0 = r8d, lanes 1..3 = 0, ymm12[128..255] = 0!
        a.vmovdqu(ymmword_ptr(rdx + 112), ymm12)?;

        // 5. vcmpps constant/tautology/NaN-sensitive predicates on [NaN, -0.0, 1.0, -2.0]
        a.vcmpps(xmm13, xmm3, xmm3, 1)?; // OLT(x, x) -> all 0
        a.vcmpps(xmm14, xmm3, xmm3, 8)?; // UEQ(x, x) -> all -1
        a.vcmpps(xmm15, xmm3, xmm3, 0)?; // OEQ(x, x) -> 0 for NaN, -1 for non-NaN
        a.vmovdqu(xmmword_ptr(rdx + 144), xmm13)?;
        a.vmovdqu(xmmword_ptr(rdx + 160), xmm14)?;
        a.vmovdqu(xmmword_ptr(rdx + 176), xmm15)?;

        a.vzeroupper()?;
        a.ret()
    });

    let built = build("hyper_simd_edge", [&f]);
    let mut input = Vec::new();
    // [0..16]: [INT_MIN, -1, 0, INT_MAX]
    for v in [i32::MIN, -1, 0, i32::MAX] {
        input.extend_from_slice(&v.to_le_bytes());
    }
    // [16..32]: [qNaN(0x7FC0_1234), -0.0, +0.0, 2.0]
    let a_bits: [u32; 4] = [0x7FC0_1234, 0x8000_0000, 0x0000_0000, 2.0f32.to_bits()];
    for b in a_bits {
        input.extend_from_slice(&b.to_le_bytes());
    }
    // [32..48]: [1.0, +0.0, -0.0, sNaN(0x7F80_0001)]
    let b_bits: [u32; 4] = [1.0f32.to_bits(), 0x0000_0000, 0x8000_0000, 0x7F80_0001];
    for b in b_bits {
        input.extend_from_slice(&b.to_le_bytes());
    }
    // [48..64]: shift values
    let sh_vals: [u32; 4] = [0x8000_0001, 0x1234_5678, 0x8000_0001, 0xDEAD_BEEF];
    for v in sh_vals {
        input.extend_from_slice(&v.to_le_bytes());
    }
    // [64..80]: shift counts [0, 1, 31, 32]
    for c in [0u32, 1, 31, 32] {
        input.extend_from_slice(&c.to_le_bytes());
    }

    let out_addr = FLAT_ADDR + 0x1000;
    let case = Case::new(CODE_BASE)
        .reg(RCX, FLAT_ADDR)
        .reg(RDX, out_addr)
        .reg(R8, 0xFFFF_FFFF_CAFE_BABE)
        .reg(R9, 0)
        .mem(FLAT_ADDR, &input)
        .read(out_addr, 192);
    let o = run(&built, &[case]);
    ok(&o, "hyper_simd_edge");
    let got = &o[0].mem[0];
    let u32s = |off: usize| -> [u32; 4] {
        std::array::from_fn(|i| {
            u32::from_le_bytes(got[off + 4 * i..off + 4 * i + 4].try_into().unwrap())
        })
    };

    // 1. pabsd & psignd(x, x) both yield [INT_MIN, 1, 0, INT_MAX]
    let want_abs = [i32::MIN as u32, 1, 0, i32::MAX as u32];
    assert_eq!(u32s(0), want_abs, "vpabsd INT_MIN");
    assert_eq!(u32s(16), want_abs, "vpsignd(x, x) INT_MIN");

    // 2. minps / maxps return b on NaN or +-0.0 ties, exact NaN bits preserved!
    assert_eq!(u32s(32), b_bits, "vminps NaN & +-0.0 second-operand rule");
    assert_eq!(u32s(48), b_bits, "vmaxps NaN & +-0.0 second-operand rule");
    assert_eq!(u32s(64), a_bits, "vminps(a, a) bit-exact identity");

    // 3. vpsllvd / vpsrlvd with count=32 zeroing lane 3
    assert_eq!(
        u32s(80),
        [0x8000_0001, 0x1234_5678 << 1, 0x8000_0001 << 31, 0],
        "vpsllvd"
    );
    assert_eq!(
        u32s(96),
        [0x8000_0001, 0x1234_5678 >> 1, 0x8000_0001 >> 31, 0],
        "vpsrlvd"
    );

    // 4. vmovd xmm12, r8d: low 32 bits only, upper 96 bits of xmm12 + upper 128 bits of ymm12 = 0!
    assert_eq!(u32s(112), [0xCAFE_BABE, 0, 0, 0], "vmovd low 128");
    assert_eq!(u32s(128), [0, 0, 0, 0], "vmovd VEX-128 upper 128 zeroing");

    // 5. vcmpps predicates
    assert_eq!(u32s(144), [0, 0, 0, 0], "vcmpps OLT(x, x)");
    assert_eq!(
        u32s(160),
        [u32::MAX, u32::MAX, u32::MAX, u32::MAX],
        "vcmpps UEQ(x, x)"
    );
    assert_eq!(
        u32s(176),
        [0, u32::MAX, u32::MAX, u32::MAX],
        "vcmpps OEQ(x, x) must return 0 for NaN lane 0"
    );
}

// ================================================================================================
// 4. Tool-gated x86-64 host-CPU differential verification
// ================================================================================================

#[cfg(target_arch = "x86_64")]
#[test]
fn host_diff_upgraded_vector_simd_ops() {
    use super::simd_diff::{Check, Feature, Kind, Op, check_ops, op};

    let mut ops: Vec<Op> = Vec::new();
    let (s2, s3, s41, avx, avx2) = (
        Feature::Sse2,
        Feature::Ssse3,
        Feature::Sse41,
        Feature::Avx,
        Feature::Avx2,
    );

    // Same-register psign* (-> @llvm.abs) across 8/16/32-bit, legacy SSE & AVX2
    ops.push(op(s3, Kind::I8, Check::Exact, |a| a.psignb(xmm1, xmm1)));
    ops.push(op(s3, Kind::I16, Check::Exact, |a| a.psignw(xmm1, xmm1)));
    ops.push(op(s3, Kind::I32, Check::Exact, |a| a.psignd(xmm1, xmm1)));
    ops.push(op(avx, Kind::I32, Check::Exact, |a| {
        a.vpsignd(xmm1, xmm2, xmm2)
    }));
    ops.push(op(avx2, Kind::I16, Check::Exact, |a| {
        a.vpsignw(ymm1, ymm2, ymm2)
    }));

    // Same-register pmulhw / pmulhuw
    ops.push(op(s2, Kind::I16, Check::Exact, |a| a.pmulhw(xmm1, xmm1)));
    ops.push(op(s2, Kind::I16, Check::Exact, |a| a.pmulhuw(xmm1, xmm1)));
    ops.push(op(avx2, Kind::I16, Check::Exact, |a| {
        a.vpmulhw(ymm1, ymm2, ymm2)
    }));
    ops.push(op(avx2, Kind::I16, Check::Exact, |a| {
        a.vpmulhuw(ymm1, ymm2, ymm2)
    }));

    // Variable shifts with `and` mask simplification
    for k in [Kind::I32, Kind::Count] {
        ops.push(op(avx2, k, Check::Exact, |a| a.vpsllvd(xmm1, xmm3, xmm2)).pairs(false));
        ops.push(op(avx2, k, Check::Exact, |a| a.vpsrlvd(ymm1, ymm3, ymm2)).pairs(false));
    }
    for k in [Kind::I64, Kind::Count] {
        ops.push(op(avx2, k, Check::Exact, |a| a.vpsllvq(ymm1, ymm3, ymm2)).pairs(false));
        ops.push(op(avx2, k, Check::Exact, |a| a.vpsrlvq(ymm1, ymm3, ymm2)).pairs(false));
    }

    // Same-register min/max float (bit-exact including NaNs and -0.0) & logic same-reg
    ops.push(op(s2, Kind::F32, Check::Exact, |a| a.minps(xmm1, xmm1)));
    ops.push(op(s2, Kind::F32, Check::Exact, |a| a.maxps(xmm1, xmm1)));
    ops.push(op(s2, Kind::F64, Check::Exact, |a| a.minpd(xmm1, xmm1)));
    ops.push(op(s2, Kind::F64, Check::Exact, |a| a.maxpd(xmm1, xmm1)));
    ops.push(op(avx, Kind::F32, Check::Exact, |a| {
        a.vminss(xmm1, xmm2, xmm2)
    }));
    ops.push(op(avx, Kind::F64, Check::Exact, |a| {
        a.vmaxsd(xmm1, xmm2, xmm2)
    }));
    ops.push(op(s2, Kind::I64, Check::Exact, |a| a.pand(xmm1, xmm1)));
    ops.push(op(s2, Kind::I64, Check::Exact, |a| a.por(xmm1, xmm1)));
    ops.push(op(s2, Kind::F32, Check::Exact, |a| a.andps(xmm1, xmm1)));
    ops.push(op(s2, Kind::F32, Check::Exact, |a| a.orps(xmm1, xmm1)));

    // Same-register movss/movsd & direct 32-bit movd
    ops.push(op(s2, Kind::I32, Check::Exact, |a| a.movss(xmm1, xmm1)));
    ops.push(op(avx, Kind::I64, Check::Exact, |a| {
        a.vmovsd_3(xmm1, xmm2, xmm2)
    }));
    ops.push(op(s2, Kind::I32, Check::Exact, |a| a.movd(xmm1, eax)).gpr(Kind::I32));
    ops.push(op(avx, Kind::I32, Check::Exact, |a| {
        a.vmovd(xmm1, dword_ptr(rbx))
    }));

    // Same-register float compares across all key predicates (0=oeq, 1=olt, 3=uno, 5=uge,
    // 7=ord, 8=ueq, 10=ule, 11=false, 12=one, 14=ogt, 15=true)
    for pred in [0u32, 1, 3, 5, 7, 8, 10, 11, 12, 14, 15] {
        ops.push(op(avx, Kind::F32, Check::Exact, |a| {
            a.vcmpps(xmm1, xmm2, xmm2, pred)
        }));
        ops.push(op(avx, Kind::F64, Check::Exact, |a| {
            a.vcmppd(xmm1, xmm2, xmm2, pred)
        }));
        ops.push(op(avx, Kind::F32, Check::Exact, |a| {
            a.vcmpss(xmm1, xmm2, xmm2, pred)
        }));
    }

    // Zero-vector algebraic folding sequences
    ops.push(op(avx, Kind::I32, Check::Exact, |a| {
        a.vpxor(xmm0, xmm0, xmm0)?;
        a.vpaddd(xmm1, xmm2, xmm0)
    }));
    ops.push(op(avx, Kind::I8, Check::Exact, |a| {
        a.vpxor(xmm0, xmm0, xmm0)?;
        a.vpmaxub(xmm1, xmm2, xmm0)
    }));
    ops.push(op(s41, Kind::I8, Check::Exact, |a| {
        a.pxor(xmm0, xmm0)?;
        a.pblendvb(xmm1, xmm2)
    }));

    check_ops("host_diff_upgraded_vector_simd_ops", ops);
}
