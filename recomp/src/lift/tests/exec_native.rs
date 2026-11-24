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
use crate::lift::{STATE_GPR_OFFSET, STATE_GS_BASE_OFFSET, STATE_SIZE, lift_module};

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
