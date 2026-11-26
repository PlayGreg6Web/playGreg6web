//! Compiles emitted `.ll` text to a wasm64 object file with an external LLVM tool.
//!
//! No LLVM libraries are linked into the recompiler; instead an installed tool is located:
//! 1. the [`LLC_ENV`] environment variable (path to `llc` or `clang`),
//! 2. the [`LLVM_BIN_DIR_ENV`] environment variable (a directory holding `llc` + `opt`, or `clang`),
//! 3. `llc` on `PATH`,
//! 4. `clang` on `PATH`,
//! 5. `llc` from the rustup `llvm-tools` component (`<sysroot>/lib/rustlib/<host>/bin/llc`).
//!
//! Which LLVM is found changes codegen, so the chosen tool's major version is compared with
//! [`EXPECTED_LLVM_MAJOR`] (override with [`LLVM_MAJOR_ENV`]) and a warning is printed on a
//! mismatch. See `docs/LLVM_TOOLCHAIN.md`.
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
use super::target::LiftTarget;

/// Environment variable overriding the LLVM tool (path to `llc` or `clang`).
pub const LLC_ENV: &str = "PLAYGTA6_LLC";

/// Environment variable naming a directory with matching `llc` and `opt` (or `clang`) binaries.
pub const LLVM_BIN_DIR_ENV: &str = "PLAYGTA6_LLVM_BIN_DIR";

/// Environment variable overriding the expected LLVM major version.
pub const LLVM_MAJOR_ENV: &str = "PLAYGTA6_LLVM_MAJOR";

/// LLVM major version the CI runners (`ubuntu-latest` = Ubuntu 24.04, `llvm-18`) provide.
/// `.github/workflows/ci.yml` does not install LLVM itself, so this tracks the runner image.
pub const EXPECTED_LLVM_MAJOR: u32 = 18;

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
    let tool = find_llvm_tool_unchecked()?;
    warn_on_version_mismatch(&tool);
    Some(tool)
}

fn find_llvm_tool_unchecked() -> Option<LlvmTool> {
    if let Some(path) = env::var_os(LLC_ENV).filter(|v| !v.is_empty()) {
        return Some(LlvmTool::from_path(PathBuf::from(path)));
    }
    if let Some(dir) = env::var_os(LLVM_BIN_DIR_ENV).filter(|v| !v.is_empty()) {
        if let Some(tool) = tool_in_dir(Path::new(&dir)) {
            return Some(tool);
        }
        eprintln!(
            "warning: {LLVM_BIN_DIR_ENV}={} holds no llc/clang; falling back to PATH discovery",
            Path::new(&dir).display()
        );
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

/// Finds `llc` (preferred, with its sibling `opt`) or `clang` inside `dir`.
pub fn tool_in_dir(dir: &Path) -> Option<LlvmTool> {
    let llc = dir.join(exe_name("llc"));
    if llc.is_file() {
        return Some(LlvmTool::from_path(llc));
    }
    let clang = dir.join(exe_name("clang"));
    clang.is_file().then_some(LlvmTool::Clang(clang))
}

/// Parses the major version out of `llc --version` / `opt --version` / `clang --version` output
/// (`LLVM version 18.1.3`, `clang version 18.1.3 (...)`, `Ubuntu LLVM version 18.1.3`).
pub fn parse_llvm_major(output: &str) -> Option<u32> {
    output.lines().find_map(|line| {
        if !(line.contains("LLVM") || line.contains("clang")) {
            return None;
        }
        let rest = line.split("version ").nth(1)?;
        let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
        digits.parse().ok()
    })
}

/// The expected major: [`LLVM_MAJOR_ENV`] if set and numeric, else [`EXPECTED_LLVM_MAJOR`].
pub fn expected_llvm_major(override_value: Option<&str>) -> u32 {
    override_value
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(EXPECTED_LLVM_MAJOR)
}

/// Warning text for a tool whose major differs from the expected one (`None` if it matches).
pub fn version_mismatch_warning(tool: &Path, found: u32, expected: u32) -> Option<String> {
    (found != expected).then(|| {
        format!(
            "warning: {} is LLVM {found} but CI/production expects LLVM {expected}; codegen and \
             test results may differ. Point {LLVM_BIN_DIR_ENV} at an LLVM {expected} bin dir, or set \
             {LLVM_MAJOR_ENV} to silence this (see docs/LLVM_TOOLCHAIN.md)",
            tool.display()
        )
    })
}

fn tool_major(path: &Path) -> Option<u32> {
    let out = Command::new(path).arg("--version").output().ok()?;
    parse_llvm_major(&String::from_utf8_lossy(&out.stdout))
}

fn warn_on_version_mismatch(tool: &LlvmTool) {
    let expected = expected_llvm_major(env::var(LLVM_MAJOR_ENV).ok().as_deref());
    let mut paths = vec![tool.path().to_path_buf()];
    if let LlvmTool::Llc { opt: Some(opt), .. } = tool {
        paths.push(opt.clone());
    }
    for path in paths {
        match tool_major(&path) {
            Some(found) => {
                if let Some(msg) = version_mismatch_warning(&path, found, expected) {
                    eprintln!("{msg}");
                }
            }
            None => eprintln!(
                "warning: could not determine the LLVM version of {}",
                path.display()
            ),
        }
    }
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

/// Compiles `ll` into an object for `target` at `-O2`: [`compile_to_wasm64_object`] for
/// [`LiftTarget::Wasm64`]; for a native target the same `opt` + `llc` (or `clang`) steps with the
/// native triple and CPU (`-relocation-model=pic` outside Windows, so objects link into PIE
/// executables).
pub fn compile_to_object(
    tool: &LlvmTool,
    target: &LiftTarget,
    ll: &Path,
    object: &Path,
) -> Result<(), CompileError> {
    let LiftTarget::Native { triple, cpu } = target else {
        return compile_to_wasm64_object(tool, ll, object);
    };
    let pic = !triple.contains("windows");
    match tool {
        LlvmTool::Clang(clang) => {
            let mut cmd = Command::new(clang);
            cmd.arg(format!("--target={triple}"))
                .arg(format!("-march={cpu}"))
                .args(["-O2", "-c", "-x", "ir"]);
            if pic {
                cmd.arg("-fPIC");
            }
            cmd.arg(ll).arg("-o").arg(object);
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
            cmd.arg(format!("-mtriple={triple}"))
                .arg(format!("-mcpu={cpu}"))
                .args(["-O2", "-filetype=obj"]);
            if pic {
                cmd.arg("-relocation-model=pic");
            }
            cmd.arg(input).arg("-o").arg(object);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_llc_clang_and_distro_banners() {
        assert_eq!(
            parse_llvm_major("LLVM (http://llvm.org/):\n  LLVM version 18.1.3\n"),
            Some(18)
        );
        assert_eq!(
            parse_llvm_major("Ubuntu clang version 18.1.3 (1ubuntu1)\n"),
            Some(18)
        );
        assert_eq!(
            parse_llvm_major("  LLVM version 23.1.5-rust-1.97\n"),
            Some(23)
        );
        assert_eq!(parse_llvm_major("rustc version 1.97\n"), None);
        assert_eq!(parse_llvm_major(""), None);
    }

    #[test]
    fn expected_major_override() {
        assert_eq!(expected_llvm_major(None), EXPECTED_LLVM_MAJOR);
        assert_eq!(expected_llvm_major(Some(" 24 ")), 24);
        assert_eq!(expected_llvm_major(Some("abc")), EXPECTED_LLVM_MAJOR);
    }

    #[test]
    fn mismatch_warning_only_on_difference() {
        let p = Path::new("llc");
        assert!(version_mismatch_warning(p, 18, 18).is_none());
        let w = version_mismatch_warning(p, 23, 18).unwrap();
        assert!(w.contains("LLVM 23") && w.contains("LLVM 18") && w.contains(LLVM_BIN_DIR_ENV));
    }

    #[test]
    fn bin_dir_prefers_llc_with_sibling_opt() {
        let dir = env::temp_dir().join(format!("pg6_llvm_dir_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(tool_in_dir(&dir), None);
        std::fs::write(dir.join(exe_name("clang")), b"").unwrap();
        assert_eq!(
            tool_in_dir(&dir),
            Some(LlvmTool::Clang(dir.join(exe_name("clang"))))
        );
        std::fs::write(dir.join(exe_name("llc")), b"").unwrap();
        std::fs::write(dir.join(exe_name("opt")), b"").unwrap();
        assert_eq!(
            tool_in_dir(&dir),
            Some(LlvmTool::Llc {
                llc: dir.join(exe_name("llc")),
                opt: Some(dir.join(exe_name("opt")))
            })
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
