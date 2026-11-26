//! Execution tests for lifted SSE code under node (tool-gated like [`exec`](super::exec)): the
//! Win64 floating point calling convention across lifted calls and the import seam, the MXCSR
//! policy and `vzeroall`/`vzeroupper`.

use iced_x86::code_asm::*;

use super::exec::{
    CODE_BASE, Case, RAX, RCX, RDX, RSP, STACK_TOP, STATE_ADDR, build, exec_or_skip, run,
};
use super::ir_simd::random_sse_program;
use super::{Program, asm_function};
use crate::lift::{STATE_XMM_OFFSET, lift_module};

const XMM_STATE: u64 = STATE_ADDR + STATE_XMM_OFFSET;
/// Host function in `runner.js`: `xmm0 = xmm0 * xmm1 + r8` (doubles), then `ret`.
const HOST_FMA_IMPORT: u64 = 0x7FFE_0010;

fn xmm_bytes(lo: u64, hi: u64) -> Vec<u8> {
    [lo.to_le_bytes(), hi.to_le_bytes()].concat()
}

#[test]
fn win64_float_calls_execute() {
    if !exec_or_skip("win64_float_calls_execute") {
        return;
    }
    let caller = CODE_BASE;
    let callee_d = CODE_BASE + 0x1000;
    let callee_f = CODE_BASE + 0x2000;
    // double callee_d(double a, float b, int c, double d) = 2a + b + c + d.
    // Uses callee-saved xmm6 (saved/restored through the stack) and clobbers volatile xmm2..5.
    let program = Program::new()
        .asm(callee_d, |a| {
            a.sub(rsp, 0x28)?;
            a.movaps(xmmword_ptr(rsp + 0x10), xmm6)?;
            a.cvtss2sd(xmm6, xmm1)?;
            a.addsd(xmm0, xmm0)?;
            a.addsd(xmm0, xmm6)?;
            a.cvtsi2sd(xmm6, r8d)?;
            a.addsd(xmm0, xmm6)?;
            a.addsd(xmm0, xmm3)?;
            a.xorps(xmm2, xmm2)?;
            a.movaps(xmm4, xmm2)?;
            a.cmpeqps(xmm4, xmm2)?;
            a.movaps(xmm5, xmm4)?;
            a.movaps(xmm6, xmmword_ptr(rsp + 0x10))?;
            a.add(rsp, 0x28)?;
            a.ret()
        })
        // float callee_f(float x, int y) = x * y
        .asm(callee_f, |a| {
            a.cvtsi2ss(xmm1, edx)?;
            a.mulss(xmm0, xmm1)?;
            a.ret()
        })
        .asm(caller, |a| {
            a.push(rbx)?;
            a.sub(rsp, 0x40)?;
            a.movaps(xmmword_ptr(rsp + 0x20), xmm6)?;
            a.movaps(xmmword_ptr(rsp + 0x30), xmm7)?;
            a.movq(xmm6, rcx)?;
            a.mov(rax, 2.5f64.to_bits())?;
            a.movq(xmm0, rax)?;
            a.mov(eax, 1.5f32.to_bits())?;
            a.movd(xmm1, eax)?;
            a.mov(r8d, 7)?;
            a.mov(rax, 1.0f64.to_bits())?;
            a.movq(xmm3, rax)?;
            a.call(callee_d)?; // 14.5
            a.movapd(xmm7, xmm0)?;
            a.mov(eax, 3.0f32.to_bits())?;
            a.movd(xmm0, eax)?;
            a.mov(edx, -4)?;
            a.call(callee_f)?; // -12.0f
            a.cvtss2sd(xmm0, xmm0)?;
            a.addsd(xmm7, xmm0)?; // 2.5
            // Import through the runtime dispatcher: 2.5 * 4.0 + 3 = 13.0 in xmm0.
            a.movapd(xmm0, xmm7)?;
            a.mov(rax, 4.0f64.to_bits())?;
            a.movq(xmm1, rax)?;
            a.mov(r8d, 3)?;
            a.mov(rax, HOST_FMA_IMPORT)?;
            a.call(rax)?;
            a.movq(rax, xmm0)?;
            a.movq(rdx, xmm6)?;
            a.movaps(xmm6, xmmword_ptr(rsp + 0x20))?;
            a.movaps(xmm7, xmmword_ptr(rsp + 0x30))?;
            a.add(rsp, 0x40)?;
            a.pop(rbx)?;
            a.ret()
        });
    let functions = program.discover();
    let built = build("win64_float", functions.values());

    let saved6 = xmm_bytes(0x0606_0606_0606_0606, 0x6666_6666_6666_6666);
    let saved7 = xmm_bytes(0x0707_0707_0707_0707, 0x7777_7777_7777_7777);
    let mut case = Case::new(caller).reg(RCX, 0x4142_4344_4546_4748);
    let mut state = vec![0u8; 16 * 16];
    state[16 * 6..16 * 7].copy_from_slice(&saved6);
    state[16 * 7..16 * 8].copy_from_slice(&saved7);
    case.mem.push((XMM_STATE, state));
    case.read.push((XMM_STATE, 16 * 16));
    let o = run(&built, &[case]);
    assert!(o[0].error.is_empty(), "trapped: {}", o[0].error);
    assert_eq!(o[0].regs[RSP], STACK_TOP);
    assert_eq!(f64::from_bits(o[0].regs[RAX]), 13.0);
    assert_eq!(
        o[0].regs[RDX], 0x4142_4344_4546_4748,
        "xmm6 preserved by callees"
    );
    let xmm = &o[0].mem[0];
    assert_eq!(
        xmm[..16],
        xmm_bytes(13.0f64.to_bits(), 0)[..],
        "xmm0 result"
    );
    assert_eq!(xmm[16 * 6..16 * 7], saved6[..], "caller restores its xmm6");
    assert_eq!(xmm[16 * 7..16 * 8], saved7[..], "caller restores its xmm7");
    assert_eq!(
        xmm[16 * 2..16 * 3],
        [0u8; 16][..],
        "volatile xmm2 clobbered by the callee"
    );
    assert_eq!(
        xmm[16 * 5..16 * 6],
        [0xFFu8; 16][..],
        "volatile xmm5 clobbered by the callee"
    );
}

#[test]
fn mxcsr_policy_executes() {
    if !exec_or_skip("mxcsr_policy_executes") {
        return;
    }
    let store = CODE_BASE;
    let load = CODE_BASE + 0x100;
    let outer = CODE_BASE + 0x200;
    let program = Program::new()
        .asm(store, |a| {
            a.stmxcsr(dword_ptr(rcx))?;
            a.vstmxcsr(dword_ptr(rcx + 4))?;
            a.mov(eax, dword_ptr(rcx))?;
            a.mov(edx, dword_ptr(rcx + 4))?;
            a.ret()
        })
        .asm(load, |a| {
            a.ldmxcsr(dword_ptr(rcx))?;
            a.vldmxcsr(dword_ptr(rcx))?;
            a.stmxcsr(dword_ptr(rcx + 4))?;
            a.mov(eax, dword_ptr(rcx + 4))?;
            a.ret()
        })
        // MXCSR lives in the guest state: a value loaded by a callee is seen by its caller.
        .asm(outer, |a| {
            a.sub(rsp, 0x28)?;
            a.call(load)?;
            a.add(rsp, 0x28)?;
            a.vstmxcsr(dword_ptr(rcx + 8))?;
            a.mov(edx, dword_ptr(rcx + 8))?;
            a.ret()
        });
    let functions = program.discover();
    let ldmxcsr_va = functions[&load].blocks[&load].instructions[0].ip();
    let built = build("mxcsr", functions.values());
    let buf = 0x18_0000u64;
    let with = |va: u64, value: u32| {
        let mut c = Case::new(va).reg(RCX, buf);
        c.mem.push((buf, value.to_le_bytes().to_vec()));
        c
    };
    let values = [
        0x1F80u32, 0x9FC0, 0x1F00, 0x0000, 0x3F80, 0x5F80, 0x7F80, 0x1_1F80,
    ];
    let mut cases = vec![with(store, 0)];
    cases.extend(values.iter().map(|&v| with(outer, v)));
    let o = run(&built, &cases);
    assert!(o[0].error.is_empty(), "{}", o[0].error);
    assert_eq!(o[0].regs[RAX], 0x1F80, "stmxcsr of a new thread's MXCSR");
    assert_eq!(o[0].regs[RDX], 0x1F80, "vstmxcsr of a new thread's MXCSR");
    for (v, out) in values.iter().zip(&o[1..]) {
        if v & 0xFFFF_6000 == 0 {
            // FTZ/DAZ and exception masks are stored (and round-trip) but not modelled.
            assert!(out.error.is_empty(), "{v:#x}: {}", out.error);
            assert_eq!(out.regs[RAX], u64::from(*v), "stmxcsr after ldmxcsr");
            assert_eq!(
                out.regs[RDX],
                u64::from(*v),
                "the caller sees the callee's MXCSR"
            );
        } else {
            // Non-default rounding modes and reserved bits fail loudly.
            assert_eq!(
                out.error,
                format!("unimplemented instruction at {ldmxcsr_va:#x}"),
                "{v:#x}"
            );
        }
    }
}

#[test]
fn vzeroall_and_vzeroupper_execute() {
    if !exec_or_skip("vzeroall_and_vzeroupper_execute") {
        return;
    }
    let zero_all = asm_function(CODE_BASE, |a| {
        a.vzeroall()?;
        a.ret()
    });
    let zero_upper = asm_function(CODE_BASE + 0x100, |a| {
        a.vzeroupper()?;
        a.vaddps(xmm1, xmm1, xmm1)?;
        a.ret()
    });
    let built = build("vzero", [&zero_all, &zero_upper]);
    let state: Vec<u8> = (0..256).map(|i| (i * 7 + 1) as u8).collect();
    let mut cases = Vec::new();
    for va in [CODE_BASE, CODE_BASE + 0x100] {
        let mut c = Case::new(va);
        c.mem.push((XMM_STATE, state.clone()));
        c.read.push((XMM_STATE, 256));
        cases.push(c);
    }
    let o = run(&built, &cases);
    assert!(o.iter().all(|o| o.error.is_empty()));
    assert_eq!(o[0].mem[0], vec![0u8; 256], "vzeroall clears xmm0..15");
    let mut expected = state.clone();
    for lane in 0..4 {
        let at = 16 + 4 * lane;
        let v = f32::from_le_bytes(state[at..at + 4].try_into().unwrap());
        expected[at..at + 4].copy_from_slice(&(v + v).to_le_bytes());
    }
    // vaddps xmm1, xmm1, xmm1 (the NaN-free lanes of this pattern) after vzeroupper
    let got = &o[1].mem[0];
    assert_eq!(got[..16], state[..16]);
    assert_eq!(got[32..], state[32..]);
    for lane in 0..4 {
        let at = 16 + 4 * lane;
        let e = f32::from_le_bytes(expected[at..at + 4].try_into().unwrap());
        let g = f32::from_le_bytes(got[at..at + 4].try_into().unwrap());
        assert!(
            e.to_bits() == g.to_bits() || (e.is_nan() && g.is_nan()),
            "lane {lane}"
        );
    }
}

#[test]
fn lifted_random_sse_code_compiles() {
    if !exec_or_skip("lifted_random_sse_code_compiles") {
        return;
    }
    let functions = random_sse_program(200);
    let module = lift_module(functions.values()).expect("lifts");
    let lifted: usize = module
        .functions
        .iter()
        .map(|f| f.ir.matches("load <2 x i64>, ptr %xmm").count())
        .sum();
    assert!(lifted > 100, "random SSE code should lift XMM operations");
    // `build` compiles and links the module; a lifter IR bug fails here.
    let _ = build("random_sse", functions.values());
}

/// Reference model of SSE4a `extrq`/`insertq` (AMD only, so not differentially testable on
/// every host): low-quadword bit fields; the undefined parts as the lifter defines them.
fn sse4a_model(dst: u64, src: u64, len: u64, idx: u64, insert: bool) -> u64 {
    let (len, idx) = (len & 63, idx & 63);
    let mask = if len == 0 { u64::MAX } else { (1 << len) - 1 };
    if insert {
        (dst & !(mask << idx)) | ((src & mask) << idx)
    } else {
        (dst >> idx) & mask
    }
}

#[test]
fn sse4a_bit_fields_execute() {
    if !exec_or_skip("sse4a_bit_fields_execute") {
        return;
    }
    let fields = [
        (0u32, 0u32),
        (1, 0),
        (8, 4),
        (16, 48),
        (32, 31),
        (63, 1),
        (5, 59),
        (0, 7),
    ];
    let mut program = Program::new();
    for (i, &(len, idx)) in fields.iter().enumerate() {
        let va = CODE_BASE + 0x100 * i as u64;
        program = program
            .asm(va, |a| {
                a.extrq_3(xmm1, len, idx)?;
                a.ret()
            })
            .asm(va + 0x40, |a| {
                a.insertq_4(xmm1, xmm2, len, idx)?;
                a.ret()
            });
    }
    let reg_extrq = CODE_BASE + 0x2000;
    let reg_insertq = CODE_BASE + 0x2040;
    let program = program
        .asm(reg_extrq, |a| {
            a.extrq(xmm1, xmm2)?;
            a.ret()
        })
        .asm(reg_insertq, |a| {
            a.insertq(xmm1, xmm2)?;
            a.ret()
        });
    let functions = program.discover();
    let built = build("sse4a", functions.values());
    let (d0, d1) = (0x0123_4567_89AB_CDEFu64, 0xFEDC_BA98_7654_3210u64);
    let s0 = 0xA5A5_5A5A_F00F_0FF0u64;
    let state = |x1: [u64; 2], x2: [u64; 2]| {
        let mut b = vec![0u8; 48];
        b[16..24].copy_from_slice(&x1[0].to_le_bytes());
        b[24..32].copy_from_slice(&x1[1].to_le_bytes());
        b[32..40].copy_from_slice(&x2[0].to_le_bytes());
        b[40..48].copy_from_slice(&x2[1].to_le_bytes());
        b
    };
    let case = |va: u64, x2: [u64; 2]| {
        let mut c = Case::new(va);
        c.mem.push((XMM_STATE, state([d0, d1], x2)));
        c.read.push((XMM_STATE + 16, 16));
        c
    };
    let mut cases = Vec::new();
    let mut expected = Vec::new();
    for (i, &(len, idx)) in fields.iter().enumerate() {
        let va = CODE_BASE + 0x100 * i as u64;
        let (l, x) = (u64::from(len), u64::from(idx));
        cases.push(case(va, [s0, 0x77]));
        expected.push(sse4a_model(d0, s0, l, x, false));
        cases.push(case(va + 0x40, [s0, 0x77]));
        expected.push(sse4a_model(d0, s0, l, x, true));
        // Register forms take the field from xmm2 (bits 5:0 / 13:8, or 69:64 / 77:72).
        cases.push(case(reg_extrq, [l | x << 8 | 0xC0C0, 0]));
        expected.push(sse4a_model(d0, 0, l, x, false));
        cases.push(case(reg_insertq, [s0, l | x << 8 | 0xC0C0]));
        expected.push(sse4a_model(d0, s0, l, x, true));
    }
    let o = run(&built, &cases);
    for (i, (out, want)) in o.iter().zip(&expected).enumerate() {
        assert!(out.error.is_empty(), "case {i}: {}", out.error);
        let lo = u64::from_le_bytes(out.mem[0][..8].try_into().unwrap());
        let hi = u64::from_le_bytes(out.mem[0][8..].try_into().unwrap());
        assert_eq!(lo, *want, "case {i}");
        assert_eq!(hi, d1, "case {i}: upper quadword preserved");
    }
}

#[test]
fn r59_wasm64_simd128_codegen_and_edge_cases() {
    if !exec_or_skip("r59_wasm64_simd128_codegen_and_edge_cases") {
        return;
    }
    use crate::lift::STATE_YMM_HI_OFFSET;
    use crate::lift::compile::{LlvmTool, find_llvm_tool};

    const YMMH_STATE: u64 = STATE_ADDR + STATE_YMM_HI_OFFSET;

    // 1) Verify LLVM emits native Wasm64 SIMD128 instructions (no stack spills or scalar loops).
    if let Some(LlvmTool::Llc { llc, opt }) = find_llvm_tool() {
        let dir = std::env::temp_dir().join(format!(
            "playgta6_r59_asm_{}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let probes: Vec<(&str, crate::cfg::Function)> = vec![
            (
                "psadbw",
                asm_function(CODE_BASE, |a| {
                    a.psadbw(xmm0, xmm1)?;
                    a.ret()
                }),
            ),
            (
                "vpermd",
                asm_function(CODE_BASE + 0x100, |a| {
                    a.vpermd(ymm0, ymm1, ymm2)?;
                    a.ret()
                }),
            ),
            (
                "vpermilps",
                asm_function(CODE_BASE + 0x200, |a| {
                    a.vpermilps(xmm0, xmm1, xmm2)?;
                    a.ret()
                }),
            ),
            (
                "vpermilpd",
                asm_function(CODE_BASE + 0x300, |a| {
                    a.vpermilpd(xmm0, xmm1, xmm2)?;
                    a.ret()
                }),
            ),
            (
                "vptest",
                asm_function(CODE_BASE + 0x400, |a| {
                    a.vptest(ymm0, ymm1)?;
                    a.ret()
                }),
            ),
            (
                "pmuludq",
                asm_function(CODE_BASE + 0x500, |a| {
                    a.paddd(xmm0, xmm2)?;
                    a.paddd(xmm1, xmm2)?;
                    a.pmuludq(xmm0, xmm1)?;
                    a.ret()
                }),
            ),
            (
                "pmuldq",
                asm_function(CODE_BASE + 0x600, |a| {
                    a.paddd(xmm0, xmm2)?;
                    a.paddd(xmm1, xmm2)?;
                    a.pmuldq(xmm0, xmm1)?;
                    a.ret()
                }),
            ),
            (
                "cvtpd2ps",
                asm_function(CODE_BASE + 0x700, |a| {
                    a.cvtpd2ps(xmm0, xmm1)?;
                    a.ret()
                }),
            ),
            (
                "cvtpd2dq",
                asm_function(CODE_BASE + 0x800, |a| {
                    a.cvtpd2dq(xmm0, xmm1)?;
                    a.ret()
                }),
            ),
            (
                "phminposuw",
                asm_function(CODE_BASE + 0x900, |a| {
                    a.phminposuw(xmm0, xmm1)?;
                    a.ret()
                }),
            ),
        ];
        for (name, func) in &probes {
            let m = lift_module([func]).expect("lift");
            let ll = dir.join(format!("{name}.ll"));
            let bc = dir.join(format!("{name}.bc"));
            let asm_path = dir.join(format!("{name}.s"));
            std::fs::write(&ll, &m.ir).expect("write ll");
            let input = if let Some(opt_bin) = &opt {
                let st = std::process::Command::new(opt_bin)
                    .args(["-O2"])
                    .arg(&ll)
                    .arg("-o")
                    .arg(&bc)
                    .status()
                    .expect("run opt");
                assert!(st.success(), "opt failed on {name}");
                &bc
            } else {
                &ll
            };
            let st = std::process::Command::new(&llc)
                .args(["-mtriple=wasm64-unknown-unknown", "-O2", "-filetype=asm"])
                .arg(input)
                .arg("-o")
                .arg(&asm_path)
                .status()
                .expect("run llc");
            assert!(st.success(), "llc failed on {name}");
            let asm = std::fs::read_to_string(&asm_path).expect("read asm");
            match *name {
                "psadbw" => {
                    assert!(asm.contains("i8x16.max_u"), "{asm}");
                    assert!(asm.contains("i8x16.min_u"), "{asm}");
                    assert!(!asm.contains("extract_lane"), "psadbw scalarized:\n{asm}");
                }
                "vpermd" | "vpermilps" => {
                    assert!(asm.contains("i8x16.swizzle"), "{name}:\n{asm}");
                    assert!(!asm.contains("__stack_pointer"), "{name} spilled stack:\n{asm}");
                }
                "vpermilpd" => {
                    assert!(asm.contains("v128.bitselect"), "{asm}");
                    assert!(!asm.contains("__stack_pointer"), "vpermilpd spilled stack:\n{asm}");
                }
                "vptest" => {
                    assert!(asm.contains("v128.any_true"), "{asm}");
                    assert!(!asm.contains("i64x2.extract_lane"), "vptest extracted lanes:\n{asm}");
                }
                "pmuludq" => {
                    // Newer LLVM (23+) `opt -O2` canonicalizes zext(shuffle) into an `and` with
                    // 0xFFFFFFFF that the wasm backend no longer folds back into `extmul`; the
                    // masked `i64x2.mul` is still a pure SIMD128 sequence (no scalarization).
                    assert!(
                        asm.contains("i64x2.extmul_low_i32x4_u")
                            || (asm.contains("v128.and") && asm.contains("4294967295")),
                        "{asm}"
                    );
                    assert!(!asm.contains("extract_lane"), "pmuludq scalarized:
{asm}");
                }
                "pmuldq" => {
                    // See pmuludq: newer LLVM may keep a shl/shr_s sign-extension + `i64x2.mul`.
                    assert!(
                        asm.contains("i64x2.extmul_low_i32x4_s")
                            || (asm.contains("i64x2.shr_s") && asm.contains("i64x2.mul")),
                        "{asm}"
                    );
                    assert!(!asm.contains("extract_lane"), "pmuldq scalarized:
{asm}");
                }
                "cvtpd2ps" => {
                    assert!(asm.contains("f32x4.demote_f64x2_zero"), "{asm}");
                }
                "cvtpd2dq" => {
                    assert!(asm.contains("i32x4.trunc_sat_f64x2_s_zero"), "{asm}");
                }
                "phminposuw" => {
                    assert!(asm.contains("i16x8.min_u"), "{asm}");
                    assert!(asm.contains("i16x8.bitmask"), "{asm}");
                }
                _ => {}
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 2) Execute critical edge cases under Wasm64 (Node):
    //    - pmulhrsw(-32768, -32768) == -32768 (wraps, does NOT saturate to 32767)
    //    - vpmulhrsw (VEX-128) zeroes ymm0[128..255]
    //    - pmaddwd(-32768, -32768) == -2147483648
    //    - psadbw max byte diff (0 vs 255 -> [2040, 2040])
    //    - cvttpd2dq on -2147483648.9 and +inf -> [-2147483648, -2147483648, 0, 0]
    let f_pmulhrsw = CODE_BASE;
    let f_vpmulhrsw = CODE_BASE + 0x100;
    let f_pmaddwd = CODE_BASE + 0x200;
    let f_psadbw = CODE_BASE + 0x300;
    let f_cvttpd2dq = CODE_BASE + 0x400;
    let f_phminposuw = CODE_BASE + 0x500;
    let program = Program::new()
        .asm(f_pmulhrsw, |a| {
            a.pmulhrsw(xmm0, xmm1)?;
            a.ret()
        })
        .asm(f_vpmulhrsw, |a| {
            a.vpmulhrsw(xmm0, xmm1, xmm2)?;
            a.ret()
        })
        .asm(f_pmaddwd, |a| {
            a.pmaddwd(xmm0, xmm1)?;
            a.ret()
        })
        .asm(f_psadbw, |a| {
            a.psadbw(xmm0, xmm1)?;
            a.ret()
        })
        .asm(f_cvttpd2dq, |a| {
            a.cvttpd2dq(xmm0, xmm1)?;
            a.ret()
        })
        .asm(f_phminposuw, |a| {
            a.phminposuw(xmm0, xmm1)?;
            a.ret()
        });
    let functions = program.discover();
    let built = build("r59_edge", functions.values());

    let words = |ws: [i16; 8]| -> Vec<u8> { ws.iter().flat_map(|w| w.to_le_bytes()).collect() };

    // pmulhrsw edge pairs
    let mut c_hrsw = Case::new(f_pmulhrsw);
    let mut xmms = vec![0u8; 48];
    xmms[0..16].copy_from_slice(&words([-32768, -32768, -1, 32767, 16384, -16384, 0, 1]));
    xmms[16..32].copy_from_slice(&words([-32768, 32767, 1, 32767, 16384, 16384, -32768, 16384]));
    c_hrsw.mem.push((XMM_STATE, xmms.clone()));
    c_hrsw.read.push((XMM_STATE, 16));

    // vpmulhrsw VEX-128 zeroing of upper 128 bits
    let mut c_vhrsw = Case::new(f_vpmulhrsw);
    let mut vxmms = vec![0u8; 48];
    vxmms[16..32].copy_from_slice(&words([-32768; 8]));
    vxmms[32..48].copy_from_slice(&words([-32768; 8]));
    c_vhrsw.mem.push((XMM_STATE, vxmms));
    c_vhrsw.mem.push((YMMH_STATE, vec![0xA5u8; 16]));
    c_vhrsw.read.push((XMM_STATE, 16));
    c_vhrsw.read.push((YMMH_STATE, 16));

    // pmaddwd(-32768, -32768)
    let mut c_madd = Case::new(f_pmaddwd);
    let mut mxmms = vec![0u8; 32];
    mxmms[0..16].copy_from_slice(&words([-32768; 8]));
    mxmms[16..32].copy_from_slice(&words([-32768; 8]));
    c_madd.mem.push((XMM_STATE, mxmms));
    c_madd.read.push((XMM_STATE, 16));

    // psadbw(0x00.., 0xFF..) -> [2040, 2040]
    let mut c_sad = Case::new(f_psadbw);
    let mut sxmms = vec![0u8; 32];
    sxmms[16..32].fill(0xFF);
    c_sad.mem.push((XMM_STATE, sxmms));
    c_sad.read.push((XMM_STATE, 16));

    // cvttpd2dq([-2147483648.9, +inf]) -> [0x80000000, 0x80000000, 0, 0]
    let mut c_cvt = Case::new(f_cvttpd2dq);
    let mut cxmms = vec![0u8; 32];
    cxmms[16..24].copy_from_slice(&(-2147483648.9f64).to_le_bytes());
    cxmms[24..32].copy_from_slice(&f64::INFINITY.to_le_bytes());
    c_cvt.mem.push((XMM_STATE, cxmms));
    c_cvt.read.push((XMM_STATE, 16));

    // phminposuw tie-breaking: lanes 2 and 5 both hold 7 -> min=7, index=2
    let mut c_minpos = Case::new(f_phminposuw);
    let mut pxmms = vec![0u8; 32];
    pxmms[16..32].copy_from_slice(&words([100, 50, 7, 99, 8, 7, 1000, 12]));
    c_minpos.mem.push((XMM_STATE, pxmms));
    c_minpos.read.push((XMM_STATE, 16));

    let out = run(&built, &[c_hrsw, c_vhrsw, c_madd, c_sad, c_cvt, c_minpos]);
    for (i, o) in out.iter().enumerate() {
        assert!(o.error.is_empty(), "case {i} trapped: {}", o.error);
    }
    assert_eq!(
        out[0].mem[0],
        words([-32768, -32767, 0, 32766, 8192, -8192, 0, 1]),
        "pmulhrsw edge cases"
    );
    assert_eq!(out[1].mem[0], words([-32768; 8]), "vpmulhrsw lo");
    assert_eq!(out[1].mem[1], vec![0u8; 16], "vpmulhrsw VEX-128 zeroes ymm0_hi");
    assert_eq!(
        out[2].mem[0],
        [-2147483648i32; 4]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
        "pmaddwd -32768*-32768 wrap"
    );
    assert_eq!(out[3].mem[0], xmm_bytes(2040, 2040), "psadbw max diff");
    assert_eq!(
        out[4].mem[0],
        [-2147483648i32, -2147483648, 0, 0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
        "cvttpd2dq edge cases"
    );
    assert_eq!(
        out[5].mem[0],
        words([7, 2, 0, 0, 0, 0, 0, 0]),
        "phminposuw tie-breaking"
    );
}

/// AVX-512VL `vpmin/vpmax{s,u}q` on XMM and YMM registers (no opmask, no broadcast) compute their
/// VEX-style lane-wise result and zero the upper half of an XMM destination.
#[test]
fn evex_quadword_min_max_execute() {
    if !exec_or_skip("evex_quadword_min_max_execute") {
        return;
    }
    const YMMH_STATE: u64 = STATE_ADDR + crate::lift::STATE_YMM_HI_OFFSET;
    let f_minsq = CODE_BASE;
    let f_minuq = CODE_BASE + 0x100;
    let f_maxsq = CODE_BASE + 0x200;
    let f_maxuq = CODE_BASE + 0x300;
    let f_ymm = CODE_BASE + 0x400;
    let program = Program::new()
        .asm(f_minsq, |a| {
            a.vpminsq(xmm1, xmm1, xmm2)?;
            a.ret()
        })
        .asm(f_minuq, |a| {
            a.vpminuq(xmm1, xmm1, xmm2)?;
            a.ret()
        })
        .asm(f_maxsq, |a| {
            a.vpmaxsq(xmm1, xmm1, xmm2)?;
            a.ret()
        })
        .asm(f_maxuq, |a| {
            a.vpmaxuq(xmm1, xmm0, xmm2)?;
            a.ret()
        })
        .asm(f_ymm, |a| {
            a.vpminuq(ymm1, ymm1, ymm2)?;
            a.ret()
        });
    let functions = program.discover();
    let built = build("evex_qword_minmax", functions.values());

    let a = [i64::MIN as u64, 5, u64::MAX, 0x8000_0000_0000_0001];
    let b = [i64::MAX as u64, (-3i64) as u64, 7, 0x8000_0000_0000_0000];
    // Registers xmm0..2 (16 bytes each), then the upper halves ymmh0..2.
    let state = |x0: [u64; 2], x1: [u64; 2], x2: [u64; 2]| {
        [x0, x1, x2]
            .iter()
            .flat_map(|x| xmm_bytes(x[0], x[1]))
            .collect::<Vec<u8>>()
    };
    let case = |func: u64| {
        let mut c = Case::new(func);
        c.mem
            .push((XMM_STATE, state([a[2], b[2]], [a[0], a[1]], [b[0], b[1]])));
        c.mem.push((YMMH_STATE, vec![0xA5u8; 3 * 16]));
        c.read.push((XMM_STATE + 16, 16));
        c.read.push((YMMH_STATE + 16, 16));
        c
    };
    let mut cases = vec![case(f_minsq), case(f_minuq), case(f_maxsq), case(f_maxuq)];
    // 256-bit: both halves are processed.
    let mut wide = Case::new(f_ymm);
    wide.mem
        .push((XMM_STATE, state([0, 0], [a[0], a[1]], [b[0], b[1]])));
    wide.mem.push((
        YMMH_STATE,
        [
            xmm_bytes(0, 0),
            xmm_bytes(a[2], a[3]),
            xmm_bytes(b[2], b[3]),
        ]
        .concat(),
    ));
    wide.read.push((XMM_STATE + 16, 16));
    wide.read.push((YMMH_STATE + 16, 16));
    cases.push(wide);
    let out = run(&built, &cases);
    for (i, o) in out.iter().enumerate() {
        assert!(o.error.is_empty(), "case {i} trapped: {}", o.error);
    }
    let lanes =
        |f: fn(u64, u64) -> u64, x: [u64; 2], y: [u64; 2]| xmm_bytes(f(x[0], y[0]), f(x[1], y[1]));
    let (x1, x2) = ([a[0], a[1]], [b[0], b[1]]);
    assert_eq!(
        out[0].mem[0],
        lanes(|p, q| (p as i64).min(q as i64) as u64, x1, x2),
        "vpminsq"
    );
    assert_eq!(out[1].mem[0], lanes(u64::min, x1, x2), "vpminuq");
    assert_eq!(
        out[2].mem[0],
        lanes(|p, q| (p as i64).max(q as i64) as u64, x1, x2),
        "vpmaxsq"
    );
    // vpmaxuq xmm1, xmm0, xmm2: first source is xmm0.
    assert_eq!(
        out[3].mem[0],
        lanes(u64::max, [a[2], b[2]], x2),
        "vpmaxuq (three operand)"
    );
    for o in &out[..4] {
        assert_eq!(
            o.mem[1],
            vec![0u8; 16],
            "VEX/EVEX-128 zeroes the upper half"
        );
    }
    assert_eq!(
        out[4].mem[0],
        lanes(u64::min, x1, x2),
        "vpminuq ymm low half"
    );
    assert_eq!(
        out[4].mem[1],
        lanes(u64::min, [a[2], a[3]], [b[2], b[3]]),
        "vpminuq ymm high half"
    );
}

/// `rdsspq` / `rdsspd` without an enabled shadow stack leave the destination register unchanged.
#[test]
fn rdssp_without_shadow_stack_is_a_nop() {
    if !exec_or_skip("rdssp_without_shadow_stack_is_a_nop") {
        return;
    }
    let f = CODE_BASE;
    let program = Program::new().asm(f, |a| {
        a.rdsspq(r10)?;
        a.rdsspd(r11d)?;
        a.ret()
    });
    let functions = program.discover();
    let built = build("rdssp_nop", functions.values());
    let c = Case::new(f)
        .reg(10, 0x1122_3344_5566_7788)
        .reg(11, 0x99AA_BBCC_DDEE_FF00);
    let out = run(&built, &[c]);
    assert!(out[0].error.is_empty(), "trapped: {}", out[0].error);
    assert_eq!(out[0].regs[10], 0x1122_3344_5566_7788);
    assert_eq!(out[0].regs[11], 0x99AA_BBCC_DDEE_FF00);
}
