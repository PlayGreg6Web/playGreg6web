//! Compiles emitted `.ll` text to a wasm64 object file with an external LLVM tool.
//!
//! No LLVM libraries are linked into the recompiler; instead an installed tool is located:
//! 1. the [`LLC_ENV`] environment variable (path to `llc` or `clang`),
//! 2. `llc` on `PATH`,
//! 3. `clang` on `PATH`,
//! 4. `llc` from the rustup `llvm-tools` component (`<sysroot>/lib/rustlib/<host>/bin/llc`).
//!
//! Code is always optimised at `-O2` so that the per-register `alloca`s become SSA values:
//! `clang` does this itself; for `llc` a sibling `opt` (same directory) runs the middle-end first
//! when present.
//!
//! No target-feature flags are passed: a module that needs wasm features (atomics for `lock`
//! instructions, bulk memory for `rep movs`/`rep stos`) names them in its own
//! `"target-features"` function attribute (see [`super::LIFTED_FUNCTION_ATTRIBUTES`] and
//! [`super::ATOMICS_TARGET_FEATURES`]), so `llc`,
//! `opt` and `clang` all see the same requirement and other modules compile unchanged. Objects
//! with atomics must be linked with `wasm-ld --shared-memory`.

use std::env;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::WASM64_TRIPLE;

/// Environment variable overriding the LLVM tool (path to `llc` or `clang`).
pub const LLC_ENV: &str = "PLAYGTA6_LLC";

/// An LLVM tool able to turn `.ll` into a wasm64 object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlvmTool {
    /// `llc`, plus the `opt` next to it (if any) for the `-O2` middle-end.
    Llc { llc: PathBuf, opt: Option<PathBuf> },
    /// `clang` (compiles IR directly with `--target=wasm64-unknown-unknown`).
    Clang(PathBuf),
}

impl LlvmTool {
    /// Path of the main executable.
    pub fn path(&self) -> &Path {
        match self {
            Self::Llc { llc, .. } => llc,
            Self::Clang(clang) => clang,
        }
    }

    fn from_path(path: PathBuf) -> Self {
        let is_clang = path
            .file_stem()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.contains("clang"));
        if is_clang {
            Self::Clang(path)
        } else {
            let opt = path
                .parent()
                .map(|dir| dir.join(exe_name("opt")))
                .filter(|p| p.is_file());
            Self::Llc { llc: path, opt }
        }
    }
}

/// Error from [`compile_to_wasm64_object`].
#[derive(Debug)]
pub enum CompileError {
    /// The tool could not be started.
    Spawn {
        tool: PathBuf,
        error: std::io::Error,
    },
    /// The tool ran and failed.
    Failed {
        tool: PathBuf,
        status: String,
        stderr: String,
    },
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spawn { tool, error } => write!(f, "failed to run {}: {error}", tool.display()),
            Self::Failed {
                tool,
                status,
                stderr,
            } => write!(f, "{} failed ({status}):\n{stderr}", tool.display()),
        }
    }
}

impl std::error::Error for CompileError {}

/// Locates an LLVM tool (see the module docs for the search order). `None` if nothing is found.
pub fn find_llvm_tool() -> Option<LlvmTool> {
    if let Some(path) = env::var_os(LLC_ENV).filter(|v| !v.is_empty()) {
        return Some(LlvmTool::from_path(PathBuf::from(path)));
    }
    if let Some(llc) = find_on_path("llc") {
        return Some(LlvmTool::from_path(llc));
    }
    if let Some(clang) = find_on_path("clang") {
        return Some(LlvmTool::Clang(clang));
    }
    rustc_tool_dir()
        .map(|dir| dir.join(exe_name("llc")))
        .filter(|p| p.is_file())
        .map(LlvmTool::from_path)
}

/// Compiles the `.ll` file `ll` into the wasm64 object file `object` at `-O2`.
pub fn compile_to_wasm64_object(
    tool: &LlvmTool,
    ll: &Path,
    object: &Path,
) -> Result<(), CompileError> {
    match tool {
        LlvmTool::Clang(clang) => {
            let mut cmd = Command::new(clang);
            cmd.arg(format!("--target={WASM64_TRIPLE}"))
                .args(["-O2", "-c", "-x", "ir"])
                .arg(ll)
                .arg("-o")
                .arg(object);
            run(clang, &mut cmd)
        }
        LlvmTool::Llc { llc, opt } => {
            let optimized = object.with_extension("opt.bc");
            let input = match opt {
                Some(opt) => {
                    let mut cmd = Command::new(opt);
                    cmd.arg("-O2").arg(ll).arg("-o").arg(&optimized);
                    run(opt, &mut cmd)?;
                    optimized.as_path()
                }
                None => ll,
            };
            let mut cmd = Command::new(llc);
            cmd.arg(format!("-mtriple={WASM64_TRIPLE}"))
                .args(["-O2", "-filetype=obj"])
                .arg(input)
                .arg("-o")
                .arg(object);
            let result = run(llc, &mut cmd);
            if opt.is_some() {
                let _ = std::fs::remove_file(&optimized);
            }
            result
        }
    }
}

/// Directory of the rustup `llvm-tools` binaries (and `rust-lld`) for the active toolchain.
pub fn rustc_tool_dir() -> Option<PathBuf> {
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let sysroot = Command::new(&rustc)
        .args(["--print", "sysroot"])
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    let sysroot = String::from_utf8(sysroot.stdout).ok()?;
    let version = Command::new(&rustc)
        .arg("-vV")
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    let version = String::from_utf8(version.stdout).ok()?;
    let host = version.lines().find_map(|l| l.strip_prefix("host: "))?;
    let dir = Path::new(sysroot.trim())
        .join("lib")
        .join("rustlib")
        .join(host.trim())
        .join("bin");
    dir.is_dir().then_some(dir)
}

/// Finds an executable on `PATH`.
pub fn find_on_path(name: &str) -> Option<PathBuf> {
    let file = exe_name(name);
    env::split_paths(&env::var_os("PATH")?)
        .map(|dir| dir.join(&file))
        .find(|p| p.is_file())
}

fn exe_name(name: &str) -> String {
    format!("{name}{}", env::consts::EXE_SUFFIX)
}

fn run(tool: &Path, cmd: &mut Command) -> Result<(), CompileError> {
    let output = cmd.output().map_err(|error| CompileError::Spawn {
        tool: tool.to_path_buf(),
        error,
    })?;
    if output.status.success() {
        Ok(())
    } else {
        Err(CompileError::Failed {
            tool: tool.to_path_buf(),
            status: output.status.to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }
}
