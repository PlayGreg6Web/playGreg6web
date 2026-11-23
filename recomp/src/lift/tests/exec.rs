//! Tool-gated tests. They compile lifted modules to wasm64 objects (needs `llc` or `clang`), link
//! them with `wasm-ld`/`rust-lld -flavor wasm`, and execute them under `node` (memory64), checking
//! results against expected values and, on x86-64 hosts, against the host CPU itself.
//!
//! Every test skips with a printed note when a tool is missing. All outputs go to a temp dir.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{env, fs};

use iced_x86::code_asm::*;
use iced_x86::{Code, Instruction, MemoryOperand, Register};

use super::ir::random_program;
use super::{
    Program, SWITCH_TABLE_RVA, SWITCH_TEXT_RVA, asm_function, assemble, encode, switch_program,
    switch_table,
};
use crate::cfg;
use crate::lift::compile::{
    LlvmTool, compile_to_wasm64_object, find_llvm_tool, find_on_path, rustc_tool_dir,
};
use crate::lift::{
    STATE_FLAGS_OFFSET, STATE_GPR_OFFSET, STATE_GS_BASE_OFFSET, STATE_XMM_OFFSET, lift_module,
};

pub(super) const RAX: usize = 0;
pub(super) const RCX: usize = 1;
pub(super) const RDX: usize = 2;
pub(super) const RBX: usize = 3;
pub(super) const RSP: usize = 4;
pub(super) const RBP: usize = 5;
pub(super) const RSI: usize = 6;
pub(super) const RDI: usize = 7;
pub(super) const R8: usize = 8;
pub(super) const R9: usize = 9;
pub(super) const R10: usize = 10;
pub(super) const R11: usize = 11;

/// Guest memory map used by the harness (all inside the 8 MiB linear memory).
pub(super) const STATE_ADDR: u64 = 0x10_0000;
pub(super) const DATA_ADDR: u64 = 0x18_0000;
pub(super) const STACK_TOP: u64 = 0x1F_0000;
pub(super) const CODE_BASE: u64 = 0x30_0000;
pub(super) const MEMORY_BYTES: u64 = 8 << 20;
/// Host-implemented function in `runner.js`: `rax = rcx * 10 + rdx`.
const HOST_MUL10_ADD: u64 = 0x7FFE_0000;

const RUNNER_JS: &str = include_str!("runner.js");

// ----- tool discovery ----------------------------------------------------------------------------
//
// Gating checks capability, not presence: an LLVM tool must compile a trivial wasm64 module, a
// linker must link it and node must instantiate and run it. A tool that is present but too old or
// built without the WebAssembly target turns into a skip with a note, never a failure. The probe
// module is hand-written (no lifter output), so lifter bugs cannot hide behind a skip. Lifted
// functions require WebAssembly SIMD (`simd128`, for SSE code), so the probe uses it as well.

const PROBE_LL: &str = "\
target triple = \"wasm64-unknown-unknown\"

define void @probe(ptr %p) #0 {
entry:
  %v = load <2 x i64>, ptr %p, align 1
  %w = add <2 x i64> %v, <i64 1, i64 1>
  store <2 x i64> %w, ptr %p, align 1
  ret void
}

attributes #0 = { \"target-features\"=\"+simd128\" }
";

const PROBE_JS: &str = "\
const fs = require('fs');
const instance = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(process.argv[2])), {});
const view = new DataView(instance.exports.memory.buffer);
view.setBigUint64(1024, 41n, true);
instance.exports.probe(1024n);
process.stdout.write(String(view.getBigUint64(1024, true)));
";

struct Linker {
    program: PathBuf,
    /// `rust-lld` needs `-flavor wasm`.
    flavor: bool,
}

struct Exec {
    linker: Linker,
    node: PathBuf,
    /// Flags node needs for memory64 (empty once it is enabled by default).
    node_flags: Vec<&'static str>,
}

struct Tools {
    /// An LLVM tool that compiled the probe module, or why there is none.
    llvm: Result<LlvmTool, String>,
    /// A linker + node pair that linked and ran the probe module, or why there is none.
    exec: Result<Exec, String>,
}

fn tools() -> &'static Tools {
    static TOOLS: OnceLock<Tools> = OnceLock::new();
    TOOLS.get_or_init(probe_tools)
}

fn probe_tools() -> Tools {
    let dir = TempDir::new("probe");
    let ll = dir.0.join("probe.ll");
    let obj = dir.0.join("probe.o");
    if let Err(err) = fs::write(&ll, PROBE_LL) {
        let reason = format!("cannot write probe files: {err}");
        return Tools {
            llvm: Err(reason.clone()),
            exec: Err(reason),
        };
    }
    let llvm = match find_llvm_tool() {
        None => Err(
            "no LLVM tool found (set PLAYGTA6_LLC, put llc/clang on PATH, or run \
                     `rustup component add llvm-tools`)"
                .to_string(),
        ),
        Some(tool) => match compile_to_wasm64_object(&tool, &ll, &obj) {
            Ok(()) => Ok(tool),
            Err(err) => Err(format!(
                "{} cannot compile a trivial wasm64 module: {}",
                tool.path().display(),
                first_line(&err.to_string())
            )),
        },
    };
    let exec = match &llvm {
        Err(reason) => Err(reason.clone()),
        Ok(_) => probe_exec(&dir.0, &obj),
    };
    Tools { llvm, exec }
}

fn probe_exec(dir: &Path, obj: &Path) -> Result<Exec, String> {
    let wasm = dir.join("probe.wasm");
    let candidates = linker_candidates();
    if candidates.is_empty() {
        return Err("no wasm linker found (wasm-ld, or rust-lld from the Rust toolchain)".into());
    }
    let mut failures = Vec::new();
    let mut linker = None;
    for candidate in candidates {
        match link(&candidate, obj, &["probe".to_string()], &wasm, &[]) {
            Ok(()) => {
                linker = Some(candidate);
                break;
            }
            Err(err) => failures.push(format!(
                "{}: {}",
                candidate.program.display(),
                first_line(&err)
            )),
        }
    }
    let linker = linker.ok_or_else(|| {
        format!(
            "no linker could link a wasm64 module ({})",
            failures.join("; ")
        )
    })?;

    let node = find_on_path("node").ok_or("node not found on PATH")?;
    let script = dir.join("probe.js");
    fs::write(&script, PROBE_JS).map_err(|e| format!("cannot write probe script: {e}"))?;
    for node_flags in [vec![], vec!["--experimental-wasm-memory64"]] {
        let ran = Command::new(&node)
            .args(&node_flags)
            .arg(&script)
            .arg(&wasm)
            .output()
            .is_ok_and(|o| o.status.success() && o.stdout == b"42");
        if ran {
            return Ok(Exec {
                linker,
                node,
                node_flags,
            });
        }
    }
    Err(format!(
        "{} cannot run a wasm64 (memory64) module",
        node.display()
    ))
}

/// `PLAYGTA6_WASM_LD`, `wasm-ld` (also versioned, e.g. `wasm-ld-18`) on PATH, then `rust-lld`
/// from the active Rust toolchain or PATH.
fn linker_candidates() -> Vec<Linker> {
    let is_rust_lld = |p: &Path| {
        p.file_stem()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.contains("rust-lld"))
    };
    let mut out = Vec::new();
    if let Some(path) = env::var_os("PLAYGTA6_WASM_LD").filter(|v| !v.is_empty()) {
        let program = PathBuf::from(path);
        let flavor = is_rust_lld(&program);
        out.push(Linker { program, flavor });
    }
    let wasm_ld_names = std::iter::once("wasm-ld".to_string())
        .chain((14..=30).rev().map(|v| format!("wasm-ld-{v}")));
    for name in wasm_ld_names {
        if let Some(program) = find_on_path(&name) {
            out.push(Linker {
                program,
                flavor: false,
            });
        }
    }
    let rust_lld = rustc_tool_dir()
        .map(|dir| dir.join(format!("rust-lld{}", env::consts::EXE_SUFFIX)))
        .filter(|p| p.is_file());
    for program in rust_lld.into_iter().chain(find_on_path("rust-lld")) {
        out.push(Linker {
            program,
            flavor: true,
        });
    }
    out
}

fn link(
    linker: &Linker,
    obj: &Path,
    exports: &[String],
    wasm: &Path,
    extra: &[String],
) -> Result<(), String> {
    let mut cmd = Command::new(&linker.program);
    if linker.flavor {
        cmd.args(["-flavor", "wasm"]);
    }
    cmd.args(["-mwasm64", "--no-entry", "--allow-undefined"])
        .arg(format!("--initial-memory={MEMORY_BYTES}"))
        .args(extra);
    for name in exports {
        cmd.arg(format!("--export={name}"));
    }
    cmd.arg("-o").arg(wasm).arg(obj);
    let output = cmd.output().map_err(|e| e.to_string())?;
    if output.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).into_owned())
    }
}

fn first_line(text: &str) -> &str {
    text.lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim()
}

fn llvm_or_skip(test: &str) -> Option<&'static LlvmTool> {
    match &tools().llvm {
        Ok(tool) => Some(tool),
        Err(reason) => {
            println!("SKIP {test}: {reason}");
            None
        }
    }
}

pub(super) fn exec_or_skip(test: &str) -> bool {
    match &tools().exec {
        Ok(_) => true,
        Err(reason) => {
            println!("SKIP {test}: {reason}");
            false
        }
    }
}

// ----- temp dirs, compile, link -----------------------------------------------------------------

struct TempDir(PathBuf);

impl TempDir {
    fn new(label: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = env::temp_dir().join(format!(
            "playgta6_recomp_{label}_{}_{nanos}",
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

fn compile(dir: &Path, ir: &str, tool: &LlvmTool) -> PathBuf {
    let ll = dir.join("module.ll");
    let obj = dir.join("module.o");
    fs::write(&ll, ir).expect("write .ll");
    if let Err(err) = compile_to_wasm64_object(tool, &ll, &obj) {
        panic!("compiling lifted IR failed: {err}\n--- IR ---\n{ir}");
    }
    obj
}

/// A linked, runnable wasm64 module.
pub(super) struct Built {
    dir: TempDir,
    wasm: PathBuf,
}

/// Linker arguments for a module with atomics (wasm-ld requires a shared memory for them).
pub(super) fn shared_memory_link_args() -> Vec<String> {
    vec![
        "--shared-memory".to_string(),
        format!("--max-memory={MEMORY_BYTES}"),
    ]
}

/// Lifts, compiles and links `functions`, exporting every lifted function. Modules with atomic
/// instructions are linked against a shared memory.
pub(super) fn build<'a>(
    label: &str,
    functions: impl IntoIterator<Item = &'a cfg::Function>,
) -> Built {
    build_with(label, functions, &[])
}

/// [`build`] with extra linker arguments.
pub(super) fn build_with<'a>(
    label: &str,
    functions: impl IntoIterator<Item = &'a cfg::Function>,
    extra: &[String],
) -> Built {
    let tools = tools();
    let module = lift_module(functions).expect("lifts");
    let mut extra = extra.to_vec();
    if module.requires_shared_memory() && !extra.iter().any(|a| a == "--shared-memory") {
        extra.extend(shared_memory_link_args());
    }
    let dir = TempDir::new(label);
    let llvm = tools.llvm.as_ref().expect("probed LLVM tool");
    let obj = compile(&dir.0, &module.ir, llvm);
    let exec = tools.exec.as_ref().expect("probed linker and node");
    let wasm = dir.0.join("module.wasm");
    let exports: Vec<String> = module.functions.iter().map(|f| f.symbol.clone()).collect();
    if let Err(err) = link(&exec.linker, &obj, &exports, &wasm, &extra) {
        panic!("linking lifted module failed: {err}");
    }
    Built { dir, wasm }
}

// ----- execution ---------------------------------------------------------------------------------

/// One invocation of a lifted function.
#[derive(Clone)]
pub(super) struct Case {
    pub func: u64,
    pub regs: [u64; 16],
    pub flags: [bool; 6],
    pub mem: Vec<(u64, Vec<u8>)>,
    pub read: Vec<(u64, usize)>,
    pub out: Vec<usize>,
}

impl Case {
    /// All registers hold distinct sentinels, `rsp = STACK_TOP`, all flags clear.
    pub fn new(func: u64) -> Self {
        let mut regs = [0u64; 16];
        for (i, r) in regs.iter_mut().enumerate() {
            *r = 0x0101_0101_0101_0101u64.wrapping_mul(i as u64 + 1) ^ 0x1000_0000_0000_0000;
        }
        regs[RSP] = STACK_TOP;
        Self {
            func,
            regs,
            flags: [false; 6],
            mem: Vec::new(),
            read: Vec::new(),
            out: (0..16).collect(),
        }
    }

    pub fn reg(mut self, idx: usize, value: u64) -> Self {
        self.regs[idx] = value;
        self
    }

    pub fn mem(mut self, addr: u64, bytes: &[u8]) -> Self {
        self.mem.push((addr, bytes.to_vec()));
        self
    }

    pub fn read(mut self, addr: u64, len: usize) -> Self {
        self.read.push((addr, len));
        self
    }
}

#[derive(Debug)]
pub(super) struct Outcome {
    pub regs: [u64; 16],
    /// CF PF AF ZF SF OF
    pub flags: [bool; 6],
    pub mem: Vec<Vec<u8>>,
    pub error: String,
}

pub(super) fn run(built: &Built, cases: &[Case]) -> Vec<Outcome> {
    let mut json = String::new();
    let _ = write!(
        json,
        "{{\"state\":\"{STATE_ADDR:#x}\",\"gprOffset\":{STATE_GPR_OFFSET},\"flagsOffset\":{STATE_FLAGS_OFFSET},\"xmmOffset\":{STATE_XMM_OFFSET},\"cases\":["
    );
    for (i, case) in cases.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        let regs: Vec<String> = (0..16)
            .filter(|&i| case.regs[i] != 0)
            .map(|i| format!("[{i},\"{:#x}\"]", case.regs[i]))
            .collect();
        let flags: Vec<&str> = case
            .flags
            .iter()
            .map(|&f| if f { "1" } else { "0" })
            .collect();
        let mem: Vec<String> = case
            .mem
            .iter()
            .map(|(addr, bytes)| {
                let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
                format!("[\"{addr:#x}\",\"{hex}\"]")
            })
            .collect();
        let read: Vec<String> = case
            .read
            .iter()
            .map(|(addr, len)| format!("[\"{addr:#x}\",{len}]"))
            .collect();
        let out: Vec<String> = case.out.iter().map(|i| i.to_string()).collect();
        let _ = write!(
            json,
            "{{\"fn\":\"sub_{:X}\",\"regs\":[{}],\"flags\":[{}],\"mem\":[{}],\"read\":[{}],\"out\":[{}]}}",
            case.func,
            regs.join(","),
            flags.join(","),
            mem.join(","),
            read.join(","),
            out.join(",")
        );
    }
    json.push_str("]}");

    let runner = built.dir.0.join("runner.js");
    let cases_path = built.dir.0.join("cases.json");
    fs::write(&runner, RUNNER_JS).expect("write runner");
    fs::write(&cases_path, json).expect("write cases");
    let exec = tools().exec.as_ref().expect("probed node");
    let output = Command::new(&exec.node)
        .args(&exec.node_flags)
        .arg(&runner)
        .arg(&built.wasm)
        .arg(&cases_path)
        .output()
        .expect("run node");
    assert!(
        output.status.success(),
        "node failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf-8 output");
    let outcomes: Vec<Outcome> = stdout
        .lines()
        .zip(cases)
        .map(|(line, case)| {
            let mut parts = line.splitn(5, '\t');
            let _index = parts.next();
            let mut regs = [0u64; 16];
            let values = parts.next().unwrap_or_default();
            for (idx, v) in case
                .out
                .iter()
                .zip(values.split(',').filter(|s| !s.is_empty()))
            {
                regs[*idx] = u64::from_str_radix(v.trim_start_matches("0x"), 16).expect("hex");
            }
            let flag_text = parts.next().unwrap_or_default().as_bytes();
            let flags = std::array::from_fn(|i| flag_text.get(i) == Some(&b'1'));
            let mem = parts
                .next()
                .unwrap_or_default()
                .split(',')
                .filter(|s| !s.is_empty())
                .map(|hex| {
                    (0..hex.len() / 2)
                        .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).expect("hex"))
                        .collect()
                })
                .collect();
            let error = parts.next().unwrap_or_default().to_string();
            Outcome {
                regs,
                flags,
                mem,
                error,
            }
        })
        .collect();
    assert_eq!(outcomes.len(), cases.len(), "runner output:\n{stdout}");
    outcomes
}

/// Runs a custom node script (written next to the module) with `args` after the module path.
pub(super) fn run_script(built: &Built, name: &str, script: &str, args: &[String]) -> String {
    let path = built.dir.0.join(name);
    fs::write(&path, script).expect("write script");
    let exec = tools().exec.as_ref().expect("probed node");
    let output = Command::new(&exec.node)
        .args(&exec.node_flags)
        .arg(&path)
        .arg(&built.wasm)
        .args(args)
        .output()
        .expect("run node");
    assert!(
        output.status.success(),
        "node {name} failed: {}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("utf-8 output")
}

const PROBE_ATOMICS_LL: &str = "\
target triple = \"wasm64-unknown-unknown\"

define void @probe(ptr %p) #0 {
entry:
  %v = atomicrmw add ptr %p, i64 1 seq_cst, align 8
  ret void
}

attributes #0 = { \"target-features\"=\"+atomics,+bulk-memory\" }
";

/// Whether modules with atomics can be compiled, linked against a shared memory and run (a
/// capability probe like the one for plain modules; skips with a note otherwise).
pub(super) fn shared_memory_or_skip(test: &str) -> bool {
    static SHARED: OnceLock<Result<(), String>> = OnceLock::new();
    let result = SHARED.get_or_init(|| {
        let tools = tools();
        let llvm = tools.llvm.as_ref().map_err(Clone::clone)?;
        let exec = tools.exec.as_ref().map_err(Clone::clone)?;
        let dir = TempDir::new("probe_shared");
        let ll = dir.0.join("probe.ll");
        let obj = dir.0.join("probe.o");
        let wasm = dir.0.join("probe.wasm");
        fs::write(&ll, PROBE_ATOMICS_LL).map_err(|e| e.to_string())?;
        compile_to_wasm64_object(llvm, &ll, &obj).map_err(|e| {
            format!(
                "{} cannot compile wasm64 atomics: {}",
                llvm.path().display(),
                first_line(&e.to_string())
            )
        })?;
        link(
            &exec.linker,
            &obj,
            &["probe".to_string()],
            &wasm,
            &shared_memory_link_args(),
        )
        .map_err(|e| format!("cannot link a shared wasm64 memory: {}", first_line(&e)))?;
        let script = dir.0.join("probe.js");
        fs::write(&script, PROBE_JS).map_err(|e| e.to_string())?;
        let ran = Command::new(&exec.node)
            .args(&exec.node_flags)
            .arg(&script)
            .arg(&wasm)
            .output()
            .is_ok_and(|o| o.status.success() && o.stdout == b"42");
        if ran {
            Ok(())
        } else {
            Err(format!(
                "{} cannot run a wasm64 module with a shared memory",
                exec.node.display()
            ))
        }
    });
    match result {
        Ok(()) => true,
        Err(reason) => {
            println!("SKIP {test}: {reason}");
            false
        }
    }
}

fn run_ok(built: &Built, cases: &[Case]) -> Vec<Outcome> {
    let outcomes = run(built, cases);
    for (i, o) in outcomes.iter().enumerate() {
        assert!(o.error.is_empty(), "case {i} trapped: {}", o.error);
        assert_eq!(o.regs[RSP], STACK_TOP, "case {i}: rsp not restored by ret");
    }
    outcomes
}

// ----- compile-only tests ------------------------------------------------------------------------

/// Checks the wasm magic and that the object imports a 64-bit linear memory.
fn assert_wasm64_object(obj: &[u8]) {
    assert_eq!(&obj[..8], b"\0asm\x01\0\0\0", "wasm magic + version 1");
    let name = b"__linear_memory";
    let at = obj
        .windows(name.len())
        .position(|w| w == name)
        .expect("object imports __linear_memory");
    let kind = obj[at + name.len()];
    let limits = obj[at + name.len() + 1];
    assert_eq!(kind, 0x02, "memory import");
    assert_eq!(limits & 0x04, 0x04, "memory64 limits flag");
}

#[test]
fn lifted_module_compiles_to_a_wasm64_object() {
    let Some(tool) = llvm_or_skip("lifted_module_compiles_to_a_wasm64_object") else {
        return;
    };
    let program = Program::new().asm(0x1000, |a| {
        let mut top = a.create_label();
        a.set_label(&mut top)?;
        a.imul_3(rax, rcx, 7)?;
        a.mul(rdx)?;
        a.div(r8)?;
        a.idiv(r9b)?;
        a.ror(r10w, cl)?;
        a.sbb(byte_ptr(rcx + rdx * 2 + 1), ah)?;
        a.cmovg(r11d, dword_ptr(rsp + 8))?;
        a.dec(r12)?;
        a.jnz(top)?;
        a.call(0x2000)?;
        a.jmp(rax)
    });
    let functions = program.asm(0x2000, |a| a.ret()).discover();
    let dir = TempDir::new("compile");
    let module = lift_module(functions.values()).expect("lifts");
    assert!(module.functions.iter().all(|f| f.unsupported.is_empty()));
    let obj = compile(&dir.0, &module.ir, tool);
    assert_wasm64_object(&fs::read(obj).expect("read object"));
}

#[test]
fn lifted_random_code_compiles() {
    let Some(tool) = llvm_or_skip("lifted_random_code_compiles") else {
        return;
    };
    let functions = random_program(150);
    let module = lift_module(functions.values()).expect("lifts");
    let dir = TempDir::new("random");
    let obj = compile(&dir.0, &module.ir, tool);
    assert_wasm64_object(&fs::read(obj).expect("read object"));
}

// ----- execution tests ---------------------------------------------------------------------------

#[test]
fn arithmetic_function_executes() {
    if !exec_or_skip("arithmetic_function_executes") {
        return;
    }
    // rax = ((rcx + rdx*4 + 0x10) * r8 - r9) ; rax ^= rax >> 3
    let f = asm_function(CODE_BASE, |a| {
        a.lea(rax, qword_ptr(rcx + rdx * 4 + 0x10))?;
        a.imul_2(rax, r8)?;
        a.sub(rax, r9)?;
        a.mov(r10, rax)?;
        a.shr(r10, 3)?;
        a.xor(rax, r10)?;
        a.ret()
    });
    let built = build("arith", &[f]);
    let inputs = [
        (1u64, 2u64, 3u64, 4u64),
        (0, 0, 0, 0),
        (u64::MAX, 1, 1, 0),
        (0x7FFF_FFFF_FFFF_FFF0, 0, 2, 1),
        (
            0x1234_5678,
            0x9ABC_DEF0,
            0xFFFF_FFFF_FFFF_FFFF,
            0x8000_0000_0000_0000,
        ),
    ];
    let cases: Vec<Case> = inputs
        .iter()
        .map(|&(c, d, e, f)| {
            Case::new(CODE_BASE)
                .reg(RCX, c)
                .reg(RDX, d)
                .reg(R8, e)
                .reg(R9, f)
        })
        .collect();
    for (o, &(c, d, e, f)) in run_ok(&built, &cases).iter().zip(&inputs) {
        let v = c
            .wrapping_add(d.wrapping_mul(4))
            .wrapping_add(0x10)
            .wrapping_mul(e)
            .wrapping_sub(f);
        let expected = v ^ (v >> 3);
        assert_eq!(o.regs[RAX], expected);
        assert_eq!(o.regs[R10], v >> 3);
        // xor: CF = OF = 0, ZF/SF from the result
        assert!(!o.flags[0] && !o.flags[5]);
        assert_eq!(o.flags[3], expected == 0);
        assert_eq!(o.flags[4], (expected as i64) < 0);
        assert_eq!(o.regs[RBX], Case::new(0).regs[RBX], "untouched register");
    }
}

#[test]
fn loops_execute() {
    if !exec_or_skip("loops_execute") {
        return;
    }
    // eax = sum(1..=ecx)
    let sum = asm_function(CODE_BASE, |a| {
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
    // rax = rcx! (wrapping)
    let fact = asm_function(CODE_BASE + 0x100, |a| {
        let mut top = a.create_label();
        let mut done = a.create_label();
        a.mov(eax, 1)?;
        a.cmp(rcx, 1)?;
        a.jbe(done)?;
        a.set_label(&mut top)?;
        a.imul_2(rax, rcx)?;
        a.dec(rcx)?;
        a.cmp(rcx, 1)?;
        a.ja(top)?;
        a.set_label(&mut done)?;
        a.ret()
    });
    let built = build("loops", &[sum, fact]);
    let ns = [0u64, 1, 2, 10, 1000, 100_000];
    let mut cases: Vec<Case> = ns
        .iter()
        .map(|&n| Case::new(CODE_BASE).reg(RCX, n))
        .collect();
    let facts = [0u64, 1, 5, 20, 21, 25];
    cases.extend(
        facts
            .iter()
            .map(|&n| Case::new(CODE_BASE + 0x100).reg(RCX, n)),
    );
    let outcomes = run_ok(&built, &cases);
    for (o, &n) in outcomes.iter().zip(&ns) {
        let expected = (n * (n + 1) / 2) as u32 as u64;
        assert_eq!(o.regs[RAX], expected, "sum 1..={n}");
        assert_eq!(o.regs[RCX], 0);
    }
    for (o, &n) in outcomes[ns.len()..].iter().zip(&facts) {
        let expected = (1..=n.max(1)).fold(1u64, |acc, k| acc.wrapping_mul(k));
        assert_eq!(o.regs[RAX], expected, "{n}!");
    }
}

#[test]
fn signed_and_unsigned_branches_execute() {
    if !exec_or_skip("signed_and_unsigned_branches_execute") {
        return;
    }
    let base = CODE_BASE;
    // max_signed / max_unsigned with branches
    let branch_max = |va: u64, signed: bool| {
        asm_function(va, move |a| {
            let mut done = a.create_label();
            a.mov(rax, rcx)?;
            a.cmp(rcx, rdx)?;
            if signed {
                a.jge(done)?;
            } else {
                a.jae(done)?;
            }
            a.mov(rax, rdx)?;
            a.set_label(&mut done)?;
            a.ret()
        })
    };
    // min_signed / min_unsigned with cmov, plus setl/setb into al/ah of r8
    let cmov_min = asm_function(base + 0x200, |a| {
        a.mov(rax, rcx)?;
        a.mov(r9, rcx)?;
        a.cmp(rcx, rdx)?;
        a.cmovg(rax, rdx)?;
        a.cmova(r9, rdx)?;
        a.mov(ebx, 0)?;
        a.setl(bl)?;
        a.setb(bh)?;
        a.ret()
    });
    // sign(rcx) as -1/0/1 via js/je chain
    let sign = asm_function(base + 0x300, |a| {
        let mut neg = a.create_label();
        let mut zero = a.create_label();
        let mut out = a.create_label();
        a.test(rcx, rcx)?;
        a.js(neg)?;
        a.je(zero)?;
        a.mov(rax, 1i64)?;
        a.jmp(out)?;
        a.set_label(&mut neg)?;
        a.mov(rax, -1i64)?;
        a.jmp(out)?;
        a.set_label(&mut zero)?;
        a.xor(eax, eax)?;
        a.set_label(&mut out)?;
        a.ret()
    });
    let built = build(
        "branches",
        &[
            branch_max(base, true),
            branch_max(base + 0x100, false),
            cmov_min,
            sign,
        ],
    );
    let pairs = [
        (1u64, 2u64),
        (u64::MAX, 1),
        (1, u64::MAX),
        (5, 5),
        (i64::MIN as u64, i64::MAX as u64),
        (0, u64::MAX),
    ];
    let mut cases = Vec::new();
    for &(x, y) in &pairs {
        for va in [base, base + 0x100, base + 0x200] {
            cases.push(Case::new(va).reg(RCX, x).reg(RDX, y));
        }
    }
    let signs = [0u64, 7, u64::MAX, i64::MIN as u64];
    for &x in &signs {
        cases.push(Case::new(base + 0x300).reg(RCX, x));
    }
    let outcomes = run_ok(&built, &cases);
    for (chunk, &(x, y)) in outcomes.chunks(3).zip(&pairs) {
        let (sx, sy) = (x as i64, y as i64);
        assert_eq!(
            chunk[0].regs[RAX],
            sx.max(sy) as u64,
            "max_signed({sx}, {sy})"
        );
        assert_eq!(chunk[1].regs[RAX], x.max(y), "max_unsigned({x}, {y})");
        assert_eq!(chunk[2].regs[RAX], sx.min(sy) as u64, "min_signed");
        assert_eq!(chunk[2].regs[R9], x.min(y), "min_unsigned");
        let expected_bx = u64::from(sx < sy) | (u64::from(x < y) << 8);
        assert_eq!(chunk[2].regs[RBX], expected_bx, "setl bl / setb bh");
    }
    for (o, &x) in outcomes[pairs.len() * 3..].iter().zip(&signs) {
        assert_eq!(
            o.regs[RAX] as i64,
            (x as i64).signum(),
            "sign({})",
            x as i64
        );
    }
}

#[test]
fn memory_loads_and_stores_execute() {
    if !exec_or_skip("memory_loads_and_stores_execute") {
        return;
    }
    let base = CODE_BASE;
    // Stores through several addressing modes/widths, then reads them back with extensions.
    let roundtrip = asm_function(base, |a| {
        a.mov(qword_ptr(rcx + rdx * 8 + 0x10), r8)?;
        a.mov(dword_ptr(rcx + rdx * 8 + 0x18), 0x8765_4321u32)?;
        a.mov(byte_ptr(rcx + 0x40), r8b)?;
        a.mov(word_ptr(rcx + 0x42), r8w)?;
        a.movzx(eax, byte_ptr(rcx + 0x40))?;
        a.movsx(r9, word_ptr(rcx + 0x42))?;
        a.movsxd(r10, dword_ptr(rcx + rdx * 8 + 0x18))?;
        a.add(rax, qword_ptr(rcx + rdx * 8 + 0x10))?;
        a.add(dword_ptr(rcx + 0x44), 5)?; // read-modify-write
        a.push(qword_ptr(rcx + 0x10 + rdx * 8))?; // memory push / pop
        a.pop(r11)?;
        a.ret()
    });
    // Byte copy loop: memcpy(rcx, rdx, r8)
    let copy = asm_function(base + 0x100, |a| {
        let mut top = a.create_label();
        let mut done = a.create_label();
        a.test(r8, r8)?;
        a.jz(done)?;
        a.set_label(&mut top)?;
        a.mov(al, byte_ptr(rdx))?;
        a.mov(byte_ptr(rcx), al)?;
        a.inc(rdx)?;
        a.inc(rcx)?;
        a.dec(r8)?;
        a.jnz(top)?;
        a.set_label(&mut done)?;
        a.ret()
    });
    // RIP-relative global counter + gs-relative (TEB-style) load.
    let global = base + 0x280;
    let gs_load = Instruction::with2(
        Code::Mov_r64_rm64,
        Register::RDX,
        MemoryOperand {
            segment_prefix: Register::GS,
            ..MemoryOperand::with_displ(0x30, 4)
        },
    )
    .expect("gs load");
    let rip = |code, reg, disp: u64| {
        Instruction::with2(
            code,
            reg,
            MemoryOperand::with_base_displ(Register::RIP, disp as i64),
        )
        .expect("rip-relative")
    };
    let counter = Program::new().bytes(
        base + 0x200,
        &encode(
            base + 0x200,
            &[
                rip(Code::Mov_r64_rm64, Register::RAX, global),
                Instruction::with1(Code::Inc_rm64, Register::RAX).expect("inc"),
                Instruction::with2(
                    Code::Mov_rm64_r64,
                    MemoryOperand::with_base_displ(Register::RIP, global as i64),
                    Register::RAX,
                )
                .expect("store"),
                gs_load,
                Instruction::with(Code::Retnq),
            ],
        ),
    );
    let counter = counter.function(base + 0x200);
    let built = build("memory", &[roundtrip, copy, counter]);

    let buf = DATA_ADDR;
    let value = 0xF1E2_D3C4_B5A6_9788u64;
    let src: Vec<u8> = (0..37u8)
        .map(|i| i.wrapping_mul(7).wrapping_add(3))
        .collect();
    let teb = DATA_ADDR + 0x2000;
    let cases = [
        Case::new(base)
            .reg(RCX, buf)
            .reg(RDX, 3)
            .reg(R8, value)
            .mem(buf, &[0u8; 0x60])
            .mem(buf + 0x44, &100u32.to_le_bytes())
            .read(buf, 0x60),
        Case::new(base + 0x100)
            .reg(RCX, buf + 0x1000)
            .reg(RDX, buf + 0x800)
            .reg(R8, src.len() as u64)
            .mem(buf + 0x800, &src)
            .mem(buf + 0x1000, &[0xEE; 40])
            .read(buf + 0x1000, 40),
        Case::new(base + 0x200)
            .mem(global, &41u64.to_le_bytes())
            .mem(STATE_ADDR + STATE_GS_BASE_OFFSET, &teb.to_le_bytes())
            .mem(teb + 0x30, &0xABCD_EF01u64.to_le_bytes())
            .read(global, 8),
    ];
    let o = run_ok(&built, &cases);

    let m = &o[0].mem[0];
    let slot = 0x10 + 3 * 8;
    assert_eq!(&m[slot..slot + 8], &value.to_le_bytes(), "qword store");
    assert_eq!(
        &m[slot + 8..slot + 12],
        &0x8765_4321u32.to_le_bytes(),
        "imm32 store"
    );
    assert_eq!(m[0x40], value as u8, "byte store");
    assert_eq!(&m[0x42..0x44], &(value as u16).to_le_bytes(), "word store");
    assert_eq!(&m[0x44..0x48], &105u32.to_le_bytes(), "add [mem], imm");
    assert_eq!(
        o[0].regs[RAX],
        u64::from(value as u8).wrapping_add(value),
        "movzx + add"
    );
    assert_eq!(o[0].regs[R9], value as u16 as i16 as i64 as u64, "movsx");
    assert_eq!(
        o[0].regs[R10], 0x8765_4321u32 as i32 as i64 as u64,
        "movsxd"
    );
    assert_eq!(o[0].regs[R11], value, "push [mem] / pop");

    let mut expected = src.clone();
    expected.extend([0xEE; 3]);
    assert_eq!(o[1].mem[0], expected, "byte copy loop");
    assert_eq!(o[1].regs[R8], 0);

    assert_eq!(o[2].mem[0], 42u64.to_le_bytes(), "RIP-relative increment");
    assert_eq!(o[2].regs[RAX], 42);
    assert_eq!(o[2].regs[RDX], 0xABCD_EF01, "gs:[0x30]");
}

#[test]
fn win64_call_chain_executes() {
    if !exec_or_skip("win64_call_chain_executes") {
        return;
    }
    let caller_va = CODE_BASE;
    let callee_va = CODE_BASE + 0x1000;
    let fact_va = CODE_BASE + 0x2000;
    let tail_va = CODE_BASE + 0x3000;
    let iat_slot = CODE_BASE + 0x800;

    // callee(a=rcx, b=rdx, c=r8, d=r9, e=[rsp+0x28]) = (a + 2b + c) * d + e
    // Uses callee-saved rbx (saved/restored), homes rdx in its shadow slot, clobbers volatiles.
    let program = Program::new().asm(callee_va, |a| {
        a.push(rbx)?;
        a.mov(qword_ptr(rsp + 0x18), rdx)?;
        a.mov(rbx, rcx)?;
        a.lea(rax, qword_ptr(rbx + rdx * 2))?;
        a.add(rax, r8)?;
        a.imul_2(rax, r9)?;
        a.add(rax, qword_ptr(rsp + 0x30))?;
        a.mov(r10, -1i64)?;
        a.mov(r11, -1i64)?;
        a.mov(ecx, 0x55)?;
        a.pop(rbx)?;
        a.ret()
    });
    // caller(rcx = x): r = callee(3, 4, 5, 6, 7) + x + 1000; return host_mul10_add(r, 2)
    let mut caller_code = vec![];
    {
        let mut a = CodeAssembler::new(64).expect("asm");
        a.push(rbx).unwrap();
        a.push(rsi).unwrap();
        a.sub(rsp, 0x38).unwrap();
        a.mov(rbx, rcx).unwrap();
        a.mov(esi, 1000).unwrap();
        a.mov(ecx, 3).unwrap();
        a.mov(edx, 4).unwrap();
        a.mov(r8d, 5).unwrap();
        a.mov(r9d, 6).unwrap();
        a.mov(qword_ptr(rsp + 0x20), 7).unwrap();
        a.call(callee_va).unwrap();
        a.add(rax, rbx).unwrap();
        a.add(rax, rsi).unwrap();
        a.mov(rcx, rax).unwrap();
        a.mov(edx, 2).unwrap();
        a.add_instruction(
            Instruction::with1(
                Code::Call_rm64,
                MemoryOperand::with_base_displ(Register::RIP, iat_slot as i64),
            )
            .unwrap(),
        )
        .unwrap();
        a.add(rsp, 0x38).unwrap();
        a.pop(rsi).unwrap();
        a.pop(rbx).unwrap();
        a.ret().unwrap();
        caller_code.extend(a.assemble(caller_va).unwrap());
    }
    // Recursive factorial through real guest stack frames.
    let program = program.bytes(caller_va, &caller_code).asm(fact_va, |a| {
        let mut rec = a.create_label();
        a.cmp(rcx, 1)?;
        a.ja(rec)?;
        a.mov(eax, 1)?;
        a.ret()?;
        a.set_label(&mut rec)?;
        a.push(rcx)?;
        a.sub(rsp, 0x20)?;
        a.dec(rcx)?;
        a.call(fact_va)?;
        a.add(rsp, 0x20)?;
        a.pop(rcx)?;
        a.imul_2(rax, rcx)?;
        a.ret()
    });
    // Tail call: tail(rcx) = callee(rcx + 1, ...) via jmp; and an indirect call through a register.
    let functions = program
        .asm(tail_va, |a| {
            a.add(rcx, 1)?;
            a.mov(rax, callee_va)?;
            a.mov(r9d, 1)?;
            a.mov(qword_ptr(rsp + 0x28), 0)?;
            a.jmp(rax)
        })
        .discover();
    assert_eq!(functions.len(), 4);
    let built = build("win64", functions.values());

    let cases = [
        Case::new(caller_va)
            .reg(RCX, 11)
            .mem(iat_slot, &HOST_MUL10_ADD.to_le_bytes())
            .read(STACK_TOP - 72, 8),
        Case::new(fact_va).reg(RCX, 10),
        Case::new(fact_va).reg(RCX, 20),
        Case::new(tail_va).reg(RCX, 9).reg(RDX, 2).reg(R8, 1),
    ];
    let o = run_ok(&built, &cases);
    let sentinels = Case::new(0).regs;

    // (3 + 8 + 5) * 6 + 7 = 103; + 11 + 1000 = 1114; host: 1114 * 10 + 2
    assert_eq!(o[0].regs[RAX], 11142);
    for reg in [RBX, RSI, RDI, RBP, 12, 13, 14, 15] {
        assert_eq!(
            o[0].regs[reg], sentinels[reg],
            "callee-saved register {reg} preserved"
        );
    }
    assert_eq!(o[0].regs[R10], u64::MAX, "volatile r10 clobbered by callee");
    assert_eq!(
        o[0].mem[0],
        4u64.to_le_bytes(),
        "callee homed rdx in the caller's shadow space"
    );

    assert_eq!(o[1].regs[RAX], 3_628_800);
    assert_eq!(o[1].regs[RCX], 10, "rcx restored from the guest stack");
    assert_eq!(o[2].regs[RAX], 2_432_902_008_176_640_000);

    // callee(10, 2, 1, 1, [rsp+0x28] = 0) = (10 + 4 + 1) * 1 + 0
    assert_eq!(o[3].regs[RAX], 15);
}

#[test]
fn traps_reach_the_runtime_helpers() {
    if !exec_or_skip("traps_reach_the_runtime_helpers") {
        return;
    }
    let div = asm_function(CODE_BASE, |a| {
        a.div(rcx)?;
        a.ret()
    });
    let idiv = asm_function(CODE_BASE + 0x100, |a| {
        a.cqo()?;
        a.idiv(rcx)?;
        a.ret()
    });
    // an unsupported instruction (port I/O) only runs when ecx != 0
    let guarded = asm_function(CODE_BASE + 0x200, |a| {
        let mut skip = a.create_label();
        a.test(ecx, ecx)?;
        a.jz(skip)?;
        a.in_(eax, dx)?;
        a.set_label(&mut skip)?;
        a.mov(eax, 7)?;
        a.ret()
    });
    let unsupported_va = guarded
        .blocks
        .values()
        .nth(1)
        .expect("unsupported block")
        .start;
    let built = build("traps", &[div, idiv, guarded]);
    let cases = [
        // 128-bit dividend: 2^64 / 3
        Case::new(CODE_BASE).reg(RDX, 1).reg(RAX, 0).reg(RCX, 3),
        Case::new(CODE_BASE).reg(RDX, 0).reg(RAX, 100).reg(RCX, 0),
        Case::new(CODE_BASE).reg(RDX, 5).reg(RAX, 0).reg(RCX, 3),
        Case::new(CODE_BASE + 0x100)
            .reg(RAX, -100i64 as u64)
            .reg(RCX, 7),
        Case::new(CODE_BASE + 0x100)
            .reg(RAX, i64::MIN as u64)
            .reg(RCX, u64::MAX),
        Case::new(CODE_BASE + 0x200).reg(RCX, 0),
        Case::new(CODE_BASE + 0x200).reg(RCX, 1),
    ];
    let o = run(&built, &cases);
    assert_eq!(o[0].error, "");
    assert_eq!((o[0].regs[RAX], o[0].regs[RDX]), (0x5555_5555_5555_5555, 1));
    assert_eq!(
        o[1].error,
        format!("divide error at {CODE_BASE:#x}"),
        "division by zero"
    );
    assert_eq!(
        o[2].error,
        format!("divide error at {CODE_BASE:#x}"),
        "quotient overflow"
    );
    assert_eq!(o[3].error, "");
    assert_eq!(o[3].regs[RAX] as i64, -14);
    assert_eq!(o[3].regs[RDX] as i64, -2);
    assert!(o[4].error.starts_with("divide error"), "INT64_MIN / -1");
    assert_eq!(o[5].error, "", "unsupported instruction on an untaken path");
    assert_eq!(o[5].regs[RAX], 7);
    assert_eq!(
        o[6].error,
        format!("unimplemented instruction at {unsupported_va:#x}")
    );
}

#[test]
fn sub_registers_execute() {
    if !exec_or_skip("sub_registers_execute") {
        return;
    }
    let f = asm_function(CODE_BASE, |a| {
        a.mov(rax, rcx)?;
        a.mov(al, 0x99)?; // ...7799 -> ...7799 with low byte 0x99
        a.mov(ah, dl)?; // bits 8..16 from dl
        a.mov(r8, rcx)?;
        a.mov(r8w, 0x1234)?;
        a.mov(r9, rcx)?;
        a.mov(r9d, 0xFFFF_FFFFu32)?; // zero-extends
        a.mov(r10, rcx)?;
        a.add(r10d, 1)?; // 32-bit arithmetic zero-extends
        a.movzx(esi, ch)?;
        a.mov(rbx, rcx)?;
        a.xchg(bl, dh)?; // swaps bits 0..8 of rbx with bits 8..16 of rdx
        a.ret()
    });
    let built = build("subregs", &[f]);
    let rcx_in = 0x1122_3344_5566_7788u64;
    let rdx_in = 0xAAAA_BBBB_CCCC_DDAAu64;
    let o = &run_ok(
        &built,
        &[Case::new(CODE_BASE).reg(RCX, rcx_in).reg(RDX, rdx_in)],
    )[0];
    assert_eq!(o.regs[RAX], 0x1122_3344_5566_AA99);
    assert_eq!(o.regs[R8], 0x1122_3344_5566_1234);
    assert_eq!(o.regs[R9], 0x0000_0000_FFFF_FFFF);
    assert_eq!(o.regs[R10], 0x0000_0000_5566_7789);
    assert_eq!(o.regs[RSI], 0x77);
    assert_eq!(o.regs[RBX], 0x1122_3344_5566_77DD);
    assert_eq!(o.regs[RDX], 0xAAAA_BBBB_CCCC_88AA);
}

#[test]
fn discovered_control_flow_executes() {
    if !exec_or_skip("discovered_control_flow_executes") {
        return;
    }
    let shared = CODE_BASE + 0x300;
    let exit_va = CODE_BASE + 0x500;
    let functions = Program::new()
        // Two functions jumping into one shared tail block (copied into both by discovery).
        .asm(CODE_BASE, |a| {
            a.mov(eax, 1)?;
            a.jmp(shared)
        })
        .asm(CODE_BASE + 0x100, |a| {
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
        // `mov ecx, 5` falls through into the next function (a tail call edge).
        .bytes(
            CODE_BASE + 0x200,
            &assemble(CODE_BASE + 0x200, |a| a.mov(ecx, 5)),
        )
        .asm(CODE_BASE + 0x205, |a| {
            a.lea(eax, dword_ptr(rcx + 1))?;
            a.ret()
        })
        .asm(CODE_BASE + 0x400, |a| a.ud2())
        // A call to a function marked non-returning that returns anyway.
        .asm(CODE_BASE + 0x480, |a| {
            a.mov(ecx, 3)?;
            a.call(exit_va)?;
            a.mov(eax, 1)?;
            a.ret()
        })
        .asm(exit_va, |a| a.ret())
        .no_return(exit_va)
        .discover();
    let after_exit_call = functions[&(CODE_BASE + 0x480)].blocks[&(CODE_BASE + 0x480)].end;
    let built = build("discovered", functions.values());
    let cases = [
        Case::new(CODE_BASE),
        Case::new(CODE_BASE + 0x100),
        Case::new(CODE_BASE + 0x200),
        Case::new(CODE_BASE + 0x400),
        Case::new(CODE_BASE + 0x480),
    ];
    let o = run(&built, &cases);
    assert_eq!(
        (o[0].error.as_str(), o[0].regs[RAX]),
        ("", 11),
        "shared tail"
    );
    assert_eq!(
        (o[1].error.as_str(), o[1].regs[RAX]),
        ("", 12),
        "shared tail"
    );
    assert_eq!(
        (o[2].error.as_str(), o[2].regs[RAX]),
        ("", 6),
        "fallthrough tail call"
    );
    assert_eq!(o[2].regs[RSP], STACK_TOP);
    assert_eq!(
        o[3].error,
        format!("trap at {:#x}", CODE_BASE + 0x400),
        "ud2"
    );
    assert_eq!(
        o[4].error,
        format!("trap at {after_exit_call:#x}"),
        "returned from a no-return call"
    );
}

#[test]
fn resolved_switch_executes() {
    if !exec_or_skip("resolved_switch_executes") {
        return;
    }
    // Image base inside the 8 MiB test memory so the table can live at its real address.
    let base = 0x40_0000u64;
    let text = base + u64::from(SWITCH_TEXT_RVA);
    let table = base + u64::from(SWITCH_TABLE_RVA);
    let other = base + 0x1100;
    let functions = switch_program(base)
        .asm(other, |a| {
            a.mov(eax, 77)?;
            a.ret()
        })
        .discover();
    assert!(
        functions[&text].blocks.values().any(|b| b
            .successors
            .iter()
            .any(|e| matches!(e, cfg::Edge::JumpTable(_)))),
        "the fixture's switch must be resolved by discovery"
    );
    let built = build("switch", functions.values());

    let inputs: [(u64, u64); 7] = [
        (0, 10),
        (1, 11),
        (2, 12),
        (3, 99),
        (4, 99),
        (100, 99),
        (u64::MAX, 99),
    ];
    let mut cases: Vec<Case> = inputs
        .iter()
        .map(|&(value, _)| Case::new(text).reg(RCX, value).mem(table, &switch_table()))
        .collect();
    // A table entry patched at run time to a target outside the proven set takes the default
    // arm: the runtime dispatcher runs the function at that address.
    let mut patched = switch_table();
    patched[..4].copy_from_slice(&0x1100u32.to_le_bytes());
    cases.push(Case::new(text).reg(RCX, 0).mem(table, &patched));

    let o = run_ok(&built, &cases);
    for (outcome, &(value, expected)) in o.iter().zip(&inputs) {
        assert_eq!(outcome.regs[RAX], expected, "switch({value:#x})");
    }
    assert_eq!(o[inputs.len()].regs[RAX], 77, "dispatcher default arm");
}
