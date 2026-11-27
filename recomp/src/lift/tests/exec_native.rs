//! Tool-gated native execution: lifted code compiled for the host triple ([`LiftTarget::Native`])
//! and run as a tiny host executable, without wasm or node. Needs `llc` (or `clang`) and a host
//! linker (`rust-lld -flavor link` / `lld-link` on Windows, `cc` elsewhere); skips with a printed
//! note when either is missing or the host is not x86-64.
//!
//! The driver is hand-written LLVM IR: it provides the runtime helpers the lifted module calls,
//! sets up a guest state whose stack and `gs` base are host globals (lifted code addresses guest
//! memory 1:1 here, so a guest VA *is* a host address), calls the lifted entry and turns the
//! outcome into the process exit code.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

use iced_x86::code_asm::*;

use super::Program;
use crate::lift::compile::{compile_to_object, find_llvm_tool, find_on_path, rustc_tool_dir};
use crate::lift::target::LiftTarget;
use crate::lift::{
    STATE_GPR_OFFSET, STATE_GS_BASE_OFFSET, STATE_SIZE, STATE_XMM_OFFSET, STATE_YMM_HI_OFFSET,
    emit_module, lift_module,
};

const F: u64 = 0x1_4000_1000;
const G: u64 = 0x1_4000_1100;
const TEB_SELF: u64 = 0x1234;

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let dir = env::temp_dir().join(format!(
            "playgta6_lift_native_{}_{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("create temp dir");
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// `f(a, b)`: reads `gs:[0x30]`, calls `g` directly and through a register, runs a short loop,
/// an SSE add and a `pshufb`; `g(a, b) = a + 2b`.
fn program() -> Program {
    Program::new()
        .asm(F, |a| {
            let mut top = a.create_label();
            a.push(rbx)?;
            a.mov(rbx, rcx)?;
            a.db(&[0x65, 0x48, 0x8B, 0x04, 0x25, 0x30, 0x00, 0x00, 0x00])?; // mov rax, gs:[0x30]
            a.add(rbx, rax)?;
            a.mov(rcx, rbx)?;
            a.sub(rsp, 0x20)?;
            a.call(G)?;
            a.mov(r8, rax)?;
            a.mov(rax, G)?;
            a.call(rax)?;
            a.add(rsp, 0x20)?;
            a.add(rax, r8)?;
            a.imul_3(rax, rax, 3)?;
            a.movq(xmm0, rax)?;
            a.paddq(xmm0, xmm0)?;
            // pshufb (lifted through the wasm swizzle, mapped back for native targets): byte 7
            // index 0x80 -> 0, byte 6 index 0x15 -> byte 5, the rest identity.
            a.mov(r9, PSHUFB_INDICES)?;
            a.movq(xmm1, r9)?;
            a.pshufb(xmm0, xmm1)?;
            a.movq(rax, xmm0)?;
            a.xor(ecx, ecx)?;
            a.mov(edx, 10)?;
            a.set_label(&mut top)?;
            a.add(rax, rcx)?;
            a.inc(rcx)?;
            a.cmp(rcx, rdx)?;
            a.jne(top)?;
            a.pop(rbx)?;
            a.ret()
        })
        .asm(G, |a| {
            a.lea(rax, qword_ptr(rcx + rdx * 2))?;
            a.ret()
        })
}

const PSHUFB_INDICES: u64 = 0x8015_0504_0302_0100;

fn expected(a: u64, b: u64) -> u64 {
    let g = a.wrapping_add(TEB_SELF).wrapping_add(2 * b);
    let v = (2 * g).wrapping_mul(3).wrapping_mul(2).to_le_bytes();
    let mut out = [0u8; 8];
    for (i, idx) in PSHUFB_INDICES.to_le_bytes().into_iter().enumerate() {
        // xmm0's upper qword is zero (movq), so indices 8..15 read 0.
        out[i] = if idx & 0x80 != 0 {
            0
        } else {
            v.get(usize::from(idx & 15)).copied().unwrap_or(0)
        };
    }
    u64::from_le_bytes(out) + 45
}

fn driver_ll(triple: &str, a: u64, b: u64) -> String {
    let o_rsp = STATE_GPR_OFFSET + 8 * 4;
    let o_rcx = STATE_GPR_OFFSET + 8;
    let o_rdx = STATE_GPR_OFFSET + 8 * 2;
    let o_rax = STATE_GPR_OFFSET;
    let o_gs = STATE_GS_BASE_OFFSET;
    let want = expected(a, b) as i64;
    format!(
        r#"target triple = "{triple}"

; Guest VAs are host addresses here (identity: the runtime's base is 0).
@__playgta6_guest_base = global i64 0
@state = global [{STATE_SIZE} x i8] zeroinitializer, align 16
@stack = global [4096 x i64] zeroinitializer, align 16
@teb = global [16 x i64] zeroinitializer, align 16

declare void @sub_{F:X}(ptr)
declare void @sub_{G:X}(ptr)
declare void @llvm.trap() noreturn

define void @__playgta6_indirect_call(ptr %s, i64 %t) {{
entry:
  %is_g = icmp eq i64 %t, {G}
  br i1 %is_g, label %call_g, label %bad
call_g:
  call void @sub_{G:X}(ptr %s)
  ret void
bad:
  call void @llvm.trap()
  unreachable
}}

define void @__playgta6_trap(ptr %s, i64 %va) {{
  call void @llvm.trap()
  unreachable
}}

define void @__playgta6_unimplemented(ptr %s, i64 %va) {{
  call void @llvm.trap()
  unreachable
}}

define void @__playgta6_divide_error(ptr %s, i64 %va) {{
  call void @llvm.trap()
  unreachable
}}

define i32 @main() {{
entry:
  store i64 {TEB_SELF}, ptr getelementptr (i8, ptr @teb, i64 48)
  %teb = ptrtoint ptr @teb to i64
  store i64 %teb, ptr getelementptr (i8, ptr @state, i64 {o_gs})
  %stack = ptrtoint ptr @stack to i64
  %top = add i64 %stack, 32768
  %entry_rsp = sub i64 %top, 8
  store i64 %entry_rsp, ptr getelementptr (i8, ptr @state, i64 {o_rsp})
  store i64 {a}, ptr getelementptr (i8, ptr @state, i64 {o_rcx})
  store i64 {b}, ptr getelementptr (i8, ptr @state, i64 {o_rdx})
  call void @sub_{F:X}(ptr @state)
  %rax = load i64, ptr getelementptr (i8, ptr @state, i64 {o_rax})
  %rsp_after = load i64, ptr getelementptr (i8, ptr @state, i64 {o_rsp})
  %ok_rax = icmp eq i64 %rax, {want}
  %ok_rsp = icmp eq i64 %rsp_after, %top
  %code_rax = select i1 %ok_rax, i32 0, i32 1
  %code_rsp = select i1 %ok_rsp, i32 0, i32 2
  %code = or i32 %code_rax, %code_rsp
  ret i32 %code
}}
"#
    )
}

/// Links `objects` into an executable, or explains why no host linker is available.
fn link(dir: &Path, objects: &[PathBuf], triple: &str) -> Result<PathBuf, String> {
    let exe = dir.join(if triple.contains("windows") {
        "native.exe"
    } else {
        "native"
    });
    let mut cmd = if triple.contains("windows-msvc") {
        let lld = find_on_path("lld-link")
            .map(|p| (p, false))
            .or_else(|| {
                rustc_tool_dir()
                    .map(|d| d.join(format!("rust-lld{}", env::consts::EXE_SUFFIX)))
                    .filter(|p| p.is_file())
                    .map(|p| (p, true))
            })
            .ok_or("no lld-link or rust-lld found")?;
        let mut cmd = Command::new(&lld.0);
        if lld.1 {
            cmd.args(["-flavor", "link"]);
        }
        // No CRT: `main` is the entry point and its return value the exit code.
        cmd.args([
            "/entry:main",
            "/subsystem:console",
            "/nodefaultlib",
            "/incremental:no",
        ])
        .arg(format!("/out:{}", exe.display()));
        cmd
    } else {
        let cc = find_on_path("cc").ok_or("no cc found")?;
        let mut cmd = Command::new(cc);
        cmd.arg("-o").arg(&exe);
        cmd
    };
    cmd.args(objects);
    let out = cmd
        .output()
        .map_err(|e| format!("cannot run the linker: {e}"))?;
    if !out.status.success() {
        panic!(
            "link failed: {}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(exe)
}

#[test]
fn lifted_code_runs_natively() {
    if !cfg!(target_arch = "x86_64") {
        eprintln!("skipping native execution test: host is not x86-64");
        return;
    }
    let Some(tool) = find_llvm_tool() else {
        eprintln!("skipping native execution test: no LLVM tool (llc/clang) found");
        return;
    };
    let target = LiftTarget::native_host();
    let triple = target.triple().to_string();
    let dir = TempDir::new();

    let functions = program().discover();
    let module = lift_module(functions.values()).expect("lift");
    assert!(module.functions.iter().all(|f| f.unsupported.is_empty()));
    let native_ir = target.retarget(module.ir.clone());
    assert!(native_ir.contains(&format!("target triple = \"{triple}\"")));
    assert!(!native_ir.contains("target datalayout"));
    assert!(!native_ir.contains("simd128"));
    assert!(
        module.ir.contains("@llvm.wasm.swizzle("),
        "the test covers the swizzle mapping"
    );
    assert!(
        !native_ir.contains("llvm.wasm."),
        "no wasm-only intrinsic survives retargeting"
    );

    let lifted_ll = dir.0.join("lifted.ll");
    let driver_ll_path = dir.0.join("driver.ll");
    let (a, b) = (5, 7);
    fs::write(&lifted_ll, &native_ir).expect("write lifted.ll");
    fs::write(&driver_ll_path, driver_ll(&triple, a, b)).expect("write driver.ll");
    let obj_ext = if triple.contains("windows") {
        "obj"
    } else {
        "o"
    };
    let mut objects = Vec::new();
    for ll in [&lifted_ll, &driver_ll_path] {
        let obj = ll.with_extension(obj_ext);
        if let Err(err) = compile_to_object(&tool, &target, ll, &obj) {
            // The tool compiles wasm64 elsewhere; a build without the X86 backend is a skip.
            eprintln!(
                "skipping native execution test: {} cannot compile for {triple}: {err}",
                tool.path().display()
            );
            return;
        }
        objects.push(obj);
    }
    let exe = match link(&dir.0, &objects, &triple) {
        Ok(exe) => exe,
        Err(reason) => {
            eprintln!("skipping native execution test: {reason}");
            return;
        }
    };
    let status = Command::new(&exe)
        .status()
        .expect("run the native executable");
    assert_eq!(
        status.code(),
        Some(0),
        "native run failed (1 = wrong rax, expected {:#x}; 2 = rsp not restored)",
        expected(a, b)
    );
}

#[test]
fn r60_scalar_bitmanip_and_fused_branches_run_natively() {
    if !cfg!(target_arch = "x86_64") {
        eprintln!("skipping native R60 test: host is not x86-64");
        return;
    }
    let Some(tool) = find_llvm_tool() else {
        eprintln!("skipping native R60 test: no LLVM tool (llc/clang) found");
        return;
    };
    let target = LiftTarget::native_host();
    let triple = target.triple().to_string();
    let dir = TempDir::new();

    // Exercises R60 paths natively: bsf(0)/bsr(0) preservation, tzcnt/lzcnt/popcnt, rol/ror,
    // shld/shrd, bswap, 32-bit div/idiv, 32-bit shifts, movzx, and fused signed INT_MIN/INT_MAX cmp+jle/cmovg.
    let prog = Program::new().asm(F, |a| {
        let mut lt_ok = a.create_label();
        let mut bad = a.create_label();
        // 1. cmp INT_MIN, INT_MAX -> jle must be taken!
        a.mov(r8, i64::MIN as u64)?;
        a.mov(r9, i64::MAX as u64)?;
        a.cmp(r8, r9)?;
        a.jle(lt_ok)?;
        a.jmp(bad)?;
        a.set_label(&mut lt_ok)?;
        // 2. bsf(0) preserves 64-bit r10 = 0x1234_5678_9ABC_DEF0
        a.mov(r10, 0x1234_5678_9ABC_DEF0u64)?;
        a.xor(ecx, ecx)?;
        a.bsf(r10d, ecx)?;
        // 3. tzcnt(0) = 64, lzcnt(0) = 64, popcnt(0xF0F0) = 8
        a.tzcnt(rax, rcx)?;
        a.lzcnt(r11, rcx)?;
        a.add(rax, r11)?; // 128
        a.mov(ecx, 0xF0F0)?;
        a.popcnt(ecx, ecx)?;
        a.add(rax, rcx)?; // 136
        // 4. rol32(0x8000_0001, 1) = 3, bswap32(0x01020304) = 0x04030201
        a.mov(edx, 0x8000_0001u32)?;
        a.rol(edx, 1)?;
        a.add(rax, rdx)?; // 139
        a.mov(edx, 0x0102_0304u32)?;
        a.bswap(edx)?;
        a.shr(edx, 24)?; // 4
        a.add(rax, rdx)?; // 143
        // 5. shld32(0x1234_5678, 0x9ABC_DEF0, 4) = 0x2345_6789 -> low byte 0x89
        a.mov(ecx, 0x1234_5678u32)?;
        a.mov(edx, 0x9ABC_DEF0u32)?;
        a.shld(ecx, edx, 4)?;
        a.movzx(ecx, cl)?;
        a.add(rax, rcx)?; // 143 + 0x89 = 280
        a.mov(rbx, rax)?; // save 280
        // 6. 32-bit idiv: -100 / 7 = -14 rem -2
        a.mov(eax, (-100i32) as u32)?;
        a.cdq()?;
        a.mov(ecx, 7)?;
        a.idiv(ecx)?; // eax = -14, edx = -2
        a.add(eax, edx)?; // -16
        a.movsxd(rax, eax)?;
        a.add(rax, rbx)?; // 264
        a.add(rax, r10)?; // 0x1234_5678_9ABC_DEF0 + 264
        a.ret()?;
        a.set_label(&mut bad)?;
        a.xor(eax, eax)?;
        a.ret()
    });
    let functions = prog.discover();
    let module = lift_module(functions.values()).expect("lift");
    assert!(module.functions.iter().all(|f| f.unsupported.is_empty()));
    let native_ir = target.retarget(module.ir);
    let want = 0x1234_5678_9ABC_DEF0u64.wrapping_add(264) as i64;
    let o_rsp = STATE_GPR_OFFSET + 8 * 4;
    let o_rax = STATE_GPR_OFFSET;
    let driver = format!(
        r#"target triple = "{triple}"
@__playgta6_guest_base = global i64 0
@state = global [{STATE_SIZE} x i8] zeroinitializer, align 16
@stack = global [4096 x i64] zeroinitializer, align 16
declare void @sub_{F:X}(ptr)
declare void @llvm.trap() noreturn
define void @__playgta6_indirect_call(ptr %s, i64 %t) {{ call void @llvm.trap() unreachable }}
define void @__playgta6_trap(ptr %s, i64 %va) {{ call void @llvm.trap() unreachable }}
define void @__playgta6_unimplemented(ptr %s, i64 %va) {{ call void @llvm.trap() unreachable }}
define void @__playgta6_divide_error(ptr %s, i64 %va) {{ call void @llvm.trap() unreachable }}
define i32 @main() {{
entry:
  %stack = ptrtoint ptr @stack to i64
  %top = add i64 %stack, 32768
  %entry_rsp = sub i64 %top, 8
  store i64 %entry_rsp, ptr getelementptr (i8, ptr @state, i64 {o_rsp})
  call void @sub_{F:X}(ptr @state)
  %rax = load i64, ptr getelementptr (i8, ptr @state, i64 {o_rax})
  %ok = icmp eq i64 %rax, {want}
  %code = select i1 %ok, i32 0, i32 1
  ret i32 %code
}}
"#
    );
    let lifted_ll = dir.0.join("r60_lifted.ll");
    let driver_ll_path = dir.0.join("r60_driver.ll");
    fs::write(&lifted_ll, &native_ir).expect("write lifted.ll");
    fs::write(&driver_ll_path, &driver).expect("write driver.ll");
    let obj_ext = if triple.contains("windows") { "obj" } else { "o" };
    let mut objects = Vec::new();
    for ll in [&lifted_ll, &driver_ll_path] {
        let obj = ll.with_extension(obj_ext);
        if let Err(err) = compile_to_object(&tool, &target, ll, &obj) {
            eprintln!("skipping native R60 test: {err}");
            return;
        }
        objects.push(obj);
    }
    let exe = match link(&dir.0, &objects, &triple) {
        Ok(exe) => exe,
        Err(reason) => {
            eprintln!("skipping native R60 test: {reason}");
            return;
        }
    };
    let status = Command::new(&exe).status().expect("run native R60");
    assert_eq!(status.code(), Some(0), "native R60 run failed");
}

#[test]
fn r61_vex128_and_vzeroupper_across_calls_run_natively() {
    if !cfg!(target_arch = "x86_64") {
        eprintln!("skipping native R61 test: host is not x86-64");
        return;
    }
    let Some(tool) = find_llvm_tool() else {
        eprintln!("skipping native R61 test: no LLVM tool (llc/clang) found");
        return;
    };
    let target = LiftTarget::native_host();
    let triple = target.triple().to_string();
    let dir = TempDir::new();

    const VX_CALLEE: u64 = G;
    const VZU_CALLEE: u64 = F + 0x200;
    const XMM1_CALLEE: u64 = F + 0x300;

    // Entry F:
    // - ymm0 starts with [1, 2, 3, 4], ymm2 starts with [10, 20, 30, 40], xmm1 starts with [5, 6]
    // - doubles ymm0 and ymm2, calls XMM1_CALLEE (doubles xmm1; disjoint from ymm0/ymm2)
    // - calls VX_CALLEE (`vxorps xmm2, xmm2, xmm2` -> must zero BOTH xmm2 and ymmh2!)
    // - adds ymm2 into ymm0 (must stay [2, 4, 6, 8])
    // - extracts ymmh2 low qword via vextracti128 + movq -> r8 (must be 0!)
    // - extracts ymmh0 low qword -> r9 (must be 6!)
    // - calls VZU_CALLEE (`vzeroupper` -> clears ymmh0..15, preserves xmm0..15!)
    // - extracts ymmh0 low qword -> r10 (must be 0!)
    // - extracts xmm0 low qword -> rax (must be 2), xmm1 low qword -> r11 (must be 10)
    // - packs result into rax: rax + (r8 << 8) + (r9 << 16) + (r10 << 24) + (r11 << 32)
    //   expected = 2 | (0 << 8) | (6 << 16) | (0 << 24) | (10 << 32) = 0x0000_000A_0006_0002
    let prog = Program::new()
        .asm(F, |a| {
            a.vpaddq(ymm0, ymm0, ymm0)?;
            a.vpaddq(ymm2, ymm2, ymm2)?;
            a.call(XMM1_CALLEE)?;
            a.call(VX_CALLEE)?;
            a.vpaddq(ymm0, ymm0, ymm2)?;
            a.vextracti128(xmm4, ymm2, 1)?;
            a.movq(r8, xmm4)?;
            a.vextracti128(xmm5, ymm0, 1)?;
            a.movq(r9, xmm5)?;
            a.call(VZU_CALLEE)?;
            a.vextracti128(xmm6, ymm0, 1)?;
            a.movq(r10, xmm6)?;
            a.movq(rax, xmm0)?;
            a.movq(r11, xmm1)?;
            a.shl(r8, 8)?;
            a.shl(r9, 16)?;
            a.shl(r10, 24)?;
            a.shl(r11, 32)?;
            a.or(rax, r8)?;
            a.or(rax, r9)?;
            a.or(rax, r10)?;
            a.or(rax, r11)?;
            a.ret()
        })
        .asm(VX_CALLEE, |a| {
            a.vxorps(xmm2, xmm2, xmm2)?;
            a.ret()
        })
        .asm(VZU_CALLEE, |a| {
            a.vzeroupper()?;
            a.ret()
        })
        .asm(XMM1_CALLEE, |a| {
            a.paddq(xmm1, xmm1)?;
            a.ret()
        });

    let functions = prog.discover();
    let module = lift_module(functions.values()).expect("lift");
    assert!(module.functions.iter().all(|f| f.unsupported.is_empty()));
    let native_ir = target.retarget(module.ir);
    let want: i64 = 0x0000_000A_0006_0002;
    let o_rsp = STATE_GPR_OFFSET + 8 * 4;
    let o_rax = STATE_GPR_OFFSET;
    let o_xmm0 = STATE_XMM_OFFSET;
    let o_xmm1 = STATE_XMM_OFFSET + 16;
    let o_xmm2 = STATE_XMM_OFFSET + 32;
    let o_ymmh0 = STATE_YMM_HI_OFFSET;
    let o_ymmh2 = STATE_YMM_HI_OFFSET + 32;
    let driver = format!(
        r#"target triple = "{triple}"
@__playgta6_guest_base = global i64 0
@state = global [{STATE_SIZE} x i8] zeroinitializer, align 16
@stack = global [4096 x i64] zeroinitializer, align 16
declare void @sub_{F:X}(ptr)
declare void @llvm.trap() noreturn
define void @__playgta6_indirect_call(ptr %s, i64 %t) {{ call void @llvm.trap() unreachable }}
define void @__playgta6_trap(ptr %s, i64 %va) {{ call void @llvm.trap() unreachable }}
define void @__playgta6_unimplemented(ptr %s, i64 %va) {{ call void @llvm.trap() unreachable }}
define void @__playgta6_divide_error(ptr %s, i64 %va) {{ call void @llvm.trap() unreachable }}
define i32 @main() {{
entry:
  %stack = ptrtoint ptr @stack to i64
  %top = add i64 %stack, 32768
  %entry_rsp = sub i64 %top, 8
  store i64 %entry_rsp, ptr getelementptr (i8, ptr @state, i64 {o_rsp})
  store <2 x i64> <i64 1, i64 2>, ptr getelementptr (i8, ptr @state, i64 {o_xmm0}), align 8
  store <2 x i64> <i64 3, i64 4>, ptr getelementptr (i8, ptr @state, i64 {o_ymmh0}), align 8
  store <2 x i64> <i64 5, i64 6>, ptr getelementptr (i8, ptr @state, i64 {o_xmm1}), align 8
  store <2 x i64> <i64 10, i64 20>, ptr getelementptr (i8, ptr @state, i64 {o_xmm2}), align 8
  store <2 x i64> <i64 30, i64 40>, ptr getelementptr (i8, ptr @state, i64 {o_ymmh2}), align 8
  call void @sub_{F:X}(ptr @state)
  %rax = load i64, ptr getelementptr (i8, ptr @state, i64 {o_rax})
  %ok = icmp eq i64 %rax, {want}
  %code = select i1 %ok, i32 0, i32 1
  ret i32 %code
}}
"#
    );
    let lifted_ll = dir.0.join("r61_lifted.ll");
    let driver_ll_path = dir.0.join("r61_driver.ll");
    fs::write(&lifted_ll, &native_ir).expect("write lifted.ll");
    fs::write(&driver_ll_path, &driver).expect("write driver.ll");
    let obj_ext = if triple.contains("windows") {
        "obj"
    } else {
        "o"
    };
    let mut objects = Vec::new();
    for ll in [&lifted_ll, &driver_ll_path] {
        let obj = ll.with_extension(obj_ext);
        if let Err(err) = compile_to_object(&tool, &target, ll, &obj) {
            eprintln!("skipping native R61 test: {err}");
            return;
        }
        objects.push(obj);
    }
    let exe = match link(&dir.0, &objects, &triple) {
        Ok(exe) => exe,
        Err(reason) => {
            eprintln!("skipping native R61 test: {reason}");
            return;
        }
    };
    let status = Command::new(&exe).status().expect("run native R61");
    assert_eq!(status.code(), Some(0), "native R61 run failed");
}

#[test]
fn r62_unread_ymmh_write_and_vzeroupper_memset_run_natively() {
    if !cfg!(target_arch = "x86_64") {
        eprintln!("skipping native R62 test: host is not x86-64");
        return;
    }
    let Some(tool) = find_llvm_tool() else {
        eprintln!("skipping native R62 test: no LLVM tool (llc/clang) found");
        return;
    };
    let target = LiftTarget::native_host();
    let triple = target.triple().to_string();
    let dir = TempDir::new();

    const WRITE_ONLY: u64 = G;
    const VEX_ONLY: u64 = F + 0x200;
    const VZU_ONLY: u64 = F + 0x300;

    // Entry F:
    // 1. Calls WRITE_ONLY (`vmovdqu ymm0, [rcx]` — writes ymm0 without reading ymm0!)
    //    and checks ymmh0 low qword == 0x33 (`r8`) and xmm0 low qword == 0x11 (`rax`).
    // 2. Calls VEX_ONLY (`vxorps xmm1, xmm1, xmm1` — zeroes only ymmh1, leaves ymmh0 untouched!)
    //    and checks ymmh1 low qword == 0 (`r9`) and ymmh0 low qword == 0x33 (`r10`).
    // 3. Calls VZU_ONLY (`vzeroupper` — bulk-memset clears ymmh0..15, preserves xmm0!)
    //    and checks ymmh0 low qword == 0 (`r11`) and xmm0 low qword == 0x11 (`rdx`).
    // Packs: rax | (r8 << 8) | (r9 << 16) | (r10 << 24) | (r11 << 32) | (rdx << 40)
    //      = 0x0000_1100_3300_3311
    let prog = Program::new()
        .asm(F, |a| {
            a.call(WRITE_ONLY)?;
            a.movq(rax, xmm0)?;
            a.vextracti128(xmm4, ymm0, 1)?;
            a.movq(r8, xmm4)?;
            a.call(VEX_ONLY)?;
            a.vextracti128(xmm5, ymm1, 1)?;
            a.movq(r9, xmm5)?;
            a.vextracti128(xmm6, ymm0, 1)?;
            a.movq(r10, xmm6)?;
            a.call(VZU_ONLY)?;
            a.vextracti128(xmm7, ymm0, 1)?;
            a.movq(r11, xmm7)?;
            a.movq(rdx, xmm0)?;
            a.shl(r8, 8)?;
            a.shl(r9, 16)?;
            a.shl(r10, 24)?;
            a.shl(r11, 32)?;
            a.shl(rdx, 40)?;
            a.or(rax, r8)?;
            a.or(rax, r9)?;
            a.or(rax, r10)?;
            a.or(rax, r11)?;
            a.or(rax, rdx)?;
            a.ret()
        })
        .asm(WRITE_ONLY, |a| {
            a.vmovdqu(ymm0, ymmword_ptr(rcx))?;
            a.ret()
        })
        .asm(VEX_ONLY, |a| {
            a.vxorps(xmm1, xmm1, xmm1)?;
            a.ret()
        })
        .asm(VZU_ONLY, |a| {
            a.vzeroupper()?;
            a.ret()
        });

    let functions = prog.discover();
    let module = lift_module(functions.values()).expect("lift");
    assert!(module.functions.iter().all(|f| f.unsupported.is_empty()));
    let native_ir = target.retarget(module.ir);
    let want: i64 = 0x0000_1100_3300_3311;
    let o_rsp = STATE_GPR_OFFSET + 8 * 4;
    let o_rax = STATE_GPR_OFFSET;
    let o_rcx = STATE_GPR_OFFSET + 8;
    let o_ymmh1 = STATE_YMM_HI_OFFSET + 16;
    let driver = format!(
        r#"target triple = "{triple}"
@__playgta6_guest_base = global i64 0
@state = global [{STATE_SIZE} x i8] zeroinitializer, align 16
@stack = global [4096 x i64] zeroinitializer, align 16
@vec_in = global [4 x i64] [i64 17, i64 34, i64 51, i64 68], align 32
declare void @sub_{F:X}(ptr)
declare void @llvm.trap() noreturn
define void @__playgta6_indirect_call(ptr %s, i64 %t) {{ call void @llvm.trap() unreachable }}
define void @__playgta6_trap(ptr %s, i64 %va) {{ call void @llvm.trap() unreachable }}
define void @__playgta6_unimplemented(ptr %s, i64 %va) {{ call void @llvm.trap() unreachable }}
define void @__playgta6_divide_error(ptr %s, i64 %va) {{ call void @llvm.trap() unreachable }}
define i32 @main() {{
entry:
  %stack = ptrtoint ptr @stack to i64
  %top = add i64 %stack, 32768
  %entry_rsp = sub i64 %top, 8
  store i64 %entry_rsp, ptr getelementptr (i8, ptr @state, i64 {o_rsp})
  %in_va = ptrtoint ptr @vec_in to i64
  store i64 %in_va, ptr getelementptr (i8, ptr @state, i64 {o_rcx})
  store <2 x i64> <i64 99, i64 100>, ptr getelementptr (i8, ptr @state, i64 {o_ymmh1}), align 8
  call void @sub_{F:X}(ptr @state)
  %rax = load i64, ptr getelementptr (i8, ptr @state, i64 {o_rax})
  %ok = icmp eq i64 %rax, {want}
  %code = select i1 %ok, i32 0, i32 1
  ret i32 %code
}}
"#
    );
    let lifted_ll = dir.0.join("r62_lifted.ll");
    let driver_ll_path = dir.0.join("r62_driver.ll");
    fs::write(&lifted_ll, &native_ir).expect("write lifted.ll");
    fs::write(&driver_ll_path, &driver).expect("write driver.ll");
    let obj_ext = if triple.contains("windows") {
        "obj"
    } else {
        "o"
    };
    let mut objects = Vec::new();
    for ll in [&lifted_ll, &driver_ll_path] {
        let obj = ll.with_extension(obj_ext);
        if let Err(err) = compile_to_object(&tool, &target, ll, &obj) {
            eprintln!("skipping native R62 test: {err}");
            return;
        }
        objects.push(obj);
    }
    let exe = match link(&dir.0, &objects, &triple) {
        Ok(exe) => exe,
        Err(reason) => {
            eprintln!("skipping native R62 test: {reason}");
            return;
        }
    };
    let status = Command::new(&exe).status().expect("run native R62");
    assert_eq!(status.code(), Some(0), "native R62 run failed");
}

#[test]
fn exit_probe_reports_return_values_including_a_tail_call_natively() {
    if !cfg!(target_arch = "x86_64") {
        eprintln!("skipping native exit probe test: host is not x86-64");
        return;
    }
    let Some(tool) = find_llvm_tool() else {
        eprintln!("skipping native exit probe test: no LLVM tool (llc/clang) found");
        return;
    };
    let target = LiftTarget::native_host();
    let triple = target.triple().to_string();
    let dir = TempDir::new();

    const OUTER: u64 = F;
    const TAILER: u64 = F + 0x100;
    const LEAF: u64 = F + 0x200;
    // OUTER(x) calls TAILER(x); TAILER sets rax = x + 5 and tail-jumps to LEAF, which doubles it.
    // OUTER and TAILER are listed, LEAF is not: expected exits are TAILER -> 24, OUTER -> 24.
    let prog = Program::new()
        .asm(OUTER, |a| {
            a.sub(rsp, 0x28)?;
            a.call(TAILER)?;
            a.add(rsp, 0x28)?;
            a.ret()
        })
        .asm(TAILER, |a| {
            a.lea(rax, qword_ptr(rcx + 5))?;
            a.jmp(LEAF)
        })
        .asm(LEAF, |a| {
            a.add(rax, rax)?;
            a.ret()
        });
    let functions = prog.discover();
    let mut module = lift_module(functions.values()).expect("lift");
    for f in &mut module.functions {
        if f.entry == OUTER || f.entry == TAILER {
            f.insert_exit_probe();
        }
    }
    let ir = emit_module(&module.functions).expect("emit");
    let leaf = &ir[ir.find(&format!("define void @sub_{LEAF:X}")).expect("leaf")..];
    assert!(!leaf.contains("__playgta6_probe_exit"), "unlisted function stays hook-free");
    let native_ir = target.retarget(ir);

    let o_rsp = STATE_GPR_OFFSET + 8 * 4;
    let o_rcx = STATE_GPR_OFFSET + 8;
    let o_rax = STATE_GPR_OFFSET;
    let driver = format!(
        r#"target triple = "{triple}"
@__playgta6_guest_base = global i64 0
@state = global [{STATE_SIZE} x i8] zeroinitializer, align 16
@stack = global [4096 x i64] zeroinitializer, align 16
@n = global i64 0
@vas = global [8 x i64] zeroinitializer
@vals = global [8 x i64] zeroinitializer
declare void @sub_{OUTER:X}(ptr)
declare void @llvm.trap() noreturn
define void @__playgta6_indirect_call(ptr %s, i64 %t) {{ call void @llvm.trap() unreachable }}
define void @__playgta6_trap(ptr %s, i64 %va) {{ call void @llvm.trap() unreachable }}
define void @__playgta6_unimplemented(ptr %s, i64 %va) {{ call void @llvm.trap() unreachable }}
define void @__playgta6_divide_error(ptr %s, i64 %va) {{ call void @llvm.trap() unreachable }}
define void @__playgta6_probe_exit(ptr %s, i64 %va) {{
  %i = load i64, ptr @n
  %pv = getelementptr i64, ptr @vas, i64 %i
  store i64 %va, ptr %pv
  %pa = getelementptr i8, ptr %s, i64 {o_rax}
  %rax = load i64, ptr %pa
  %pr = getelementptr i64, ptr @vals, i64 %i
  store i64 %rax, ptr %pr
  %i2 = add i64 %i, 1
  store i64 %i2, ptr @n
  ret void
}}
define i32 @main() {{
entry:
  %stack = ptrtoint ptr @stack to i64
  %top = add i64 %stack, 32768
  %entry_rsp = sub i64 %top, 8
  store i64 %entry_rsp, ptr getelementptr (i8, ptr @state, i64 {o_rsp})
  store i64 7, ptr getelementptr (i8, ptr @state, i64 {o_rcx})
  call void @sub_{OUTER:X}(ptr @state)
  %n = load i64, ptr @n
  %v0 = load i64, ptr @vas
  %v1 = load i64, ptr getelementptr (i64, ptr @vas, i64 1)
  %r0 = load i64, ptr @vals
  %r1 = load i64, ptr getelementptr (i64, ptr @vals, i64 1)
  %c0 = icmp eq i64 %n, 2
  %c1 = icmp eq i64 %v0, {TAILER}
  %c2 = icmp eq i64 %v1, {OUTER}
  %c3 = icmp eq i64 %r0, 24
  %c4 = icmp eq i64 %r1, 24
  %a0 = and i1 %c0, %c1
  %a1 = and i1 %a0, %c2
  %a2 = and i1 %a1, %c3
  %a3 = and i1 %a2, %c4
  %code = select i1 %a3, i32 0, i32 1
  ret i32 %code
}}
"#
    );
    let lifted_ll = dir.0.join("exit_lifted.ll");
    let driver_ll_path = dir.0.join("exit_driver.ll");
    fs::write(&lifted_ll, &native_ir).expect("write lifted.ll");
    fs::write(&driver_ll_path, &driver).expect("write driver.ll");
    let obj_ext = if triple.contains("windows") { "obj" } else { "o" };
    let mut objects = Vec::new();
    for ll in [&lifted_ll, &driver_ll_path] {
        let obj = ll.with_extension(obj_ext);
        if let Err(err) = compile_to_object(&tool, &target, ll, &obj) {
            eprintln!("skipping native exit probe test: {err}");
            return;
        }
        objects.push(obj);
    }
    let exe = match link(&dir.0, &objects, &triple) {
        Ok(exe) => exe,
        Err(reason) => {
            eprintln!("skipping native exit probe test: {reason}");
            return;
        }
    };
    let status = Command::new(&exe).status().expect("run native exit probe");
    assert_eq!(status.code(), Some(0), "exit hooks saw the wrong sequence or values");
}
