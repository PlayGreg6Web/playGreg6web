//! `playgta6-recomp build-program`: the executable plus its shipped DLLs as one guest program
//! (see `playgta6_recomp::pipeline::program`). Outputs are derived from the inputs' code and
//! data, so `--out` must be a gitignored directory such as `work/` or lie outside the repo.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use playgta6_recomp::lift::MemoryModel;
use playgta6_recomp::lift::target::LiftTarget;
use playgta6_recomp::pipeline::{self, IndirectDispatch, program};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramArgs {
    pub exe: PathBuf,
    pub out_dir: PathBuf,
    pub dlls: Vec<PathBuf>,
    pub dll_dirs: Vec<PathBuf>,
    pub overrides: program::Overrides,
    pub policy_files: Vec<PathBuf>,
    pub allow_allocator_split: bool,
    pub address_pool: Option<(u64, u64)>,
    pub shims: Vec<PathBuf>,
    pub emit_obj: bool,
    pub jobs: Option<usize>,
    pub shard_size: Option<usize>,
    pub shard_instructions: Option<usize>,
    /// Code-generation target of every module (`--target`, `--target-cpu`).
    pub codegen: LiftTarget,
    /// Memory model of every module (`--memory-model`).
    pub memory: MemoryModel,
    /// Indirect-call dispatch mode of every module (`--indirect-dispatch`).
    pub indirect_dispatch: IndirectDispatch,
    /// Recorded indirect-call targets to seed (`--seed-targets`) and the report file (`--seed-report`).
    pub seed_targets: Option<PathBuf>,
    pub seed_report: Option<PathBuf>,
}

pub const USAGE: &str = "  playgta6-recomp build-program <exe> --out <dir> [--dll <file>]... [--dll-dir <dir>]... [--guest-dll <name>]... [--host-dll <name>]... [--bind <dll>!<symbol>=guest|host|prefer-host]... [--policy <file>]... [--allow-allocator-split] [--address-pool <lo>-<hi>] [--shims <header>...] [--emit-obj] [--jobs N] [--shard-size N] [--shard-instructions N] [--target wasm64|native|native:<triple>] [--target-cpu <cpu>] [--memory-model <m>] [--indirect-dispatch table|shard] [--seed-targets <file>] [--seed-report <file>]";

pub const OPTIONS: &[&str] = &[
    "  --target <t> / --target-cpu <cpu> build-program: code-generation target of every module (as for build)",
    "  --memory-model <m>     build-program: memory model of every module (as for build), recorded in pg6_program.cpp",
    "  --indirect-dispatch <m> build-program: indirect-call dispatch mode of every module (table|shard), wired in pg6_program.cpp",
    "  --dll <file>           build-program: offer a DLL as a guest module (used when the program imports it)",
    "  --dll-dir <dir>        build-program: offer every *.dll in <dir>",
    "  --guest-dll <name>     build-program: build <name> as a guest module even if host-only by default or unreferenced",
    "  --host-dll <name>      build-program: never build <name>; bind its imports to host shims",
    "  --bind <rule>          build-program: binding rule <dll>!<symbol-pattern>=guest|host|prefer-host (first match wins)",
    "  --policy <file>        build-program: rules from a file (<dll>!<symbol>=<action>, <dll>=guest|host, # comments)",
    "  --allow-allocator-split build-program: report heap functions bound to different allocators instead of failing",
    "  --seed-targets <file>  build-program: indirect-call targets recorded by a boot (playgta6_boot --record-indirect-targets); validated, then lifted as function entries",
    "  --seed-report <file>   build-program: with --seed-targets, a JSON line per module: rejected targets and targets inside functions with unresolved jump tables",
    "  --address-pool <lo>-<hi> build-program: hex range every module image must lie in (default 0x100000000-0x400000000)",
];

/// Parses the arguments after `build-program`. `Ok(None)` means help was requested.
pub fn parse_program_args(args: &[String]) -> Result<Option<ProgramArgs>, String> {
    let mut exe = None;
    let mut out_dir = None;
    let mut parsed = ProgramArgs {
        exe: PathBuf::new(),
        out_dir: PathBuf::new(),
        dlls: Vec::new(),
        dll_dirs: Vec::new(),
        overrides: program::Overrides::default(),
        policy_files: Vec::new(),
        allow_allocator_split: false,
        address_pool: None,
        shims: Vec::new(),
        emit_obj: false,
        jobs: None,
        shard_size: None,
        shard_instructions: None,
        codegen: LiftTarget::Wasm64,
        memory: MemoryModel::default(),
        indirect_dispatch: IndirectDispatch::default(),
        seed_targets: None,
        seed_report: None,
    };
    let value = |i: usize, flag: &str, what: &str| -> Result<String, String> {
        args.get(i)
            .filter(|v| !v.starts_with('-'))
            .cloned()
            .ok_or_else(|| format!("{flag} requires a {what} argument"))
    };
    let count = |i: usize, flag: &str| -> Result<usize, String> {
        args.get(i)
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .ok_or_else(|| format!("{flag} requires a positive integer"))
    };
    let mut target_spec: Option<String> = None;
    let mut target_cpu: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        match flag {
            "-h" | "--help" => return Ok(None),
            "--emit-obj" => parsed.emit_obj = true,
            "--allow-allocator-split" => parsed.allow_allocator_split = true,
            "--out" => {
                i += 1;
                out_dir = Some(PathBuf::from(value(i, flag, "<dir>")?));
            }
            "--dll" => {
                i += 1;
                parsed.dlls.push(PathBuf::from(value(i, flag, "<file>")?));
            }
            "--dll-dir" => {
                i += 1;
                parsed
                    .dll_dirs
                    .push(PathBuf::from(value(i, flag, "<dir>")?));
            }
            "--guest-dll" => {
                i += 1;
                parsed.overrides.add_guest_dll(&value(i, flag, "<name>")?)?;
            }
            "--host-dll" => {
                i += 1;
                parsed.overrides.add_host_dll(&value(i, flag, "<name>")?)?;
            }
            "--bind" => {
                i += 1;
                parsed.overrides.add_bind(&value(i, flag, "<rule>")?)?;
            }
            "--address-pool" => {
                i += 1;
                let text = value(i, flag, "<lo>-<hi>")?;
                let parse = |t: &str| {
                    let t = t.trim();
                    u64::from_str_radix(t.strip_prefix("0x").unwrap_or(t), 16).ok()
                };
                parsed.address_pool = Some(
                    text.split_once('-')
                        .and_then(|(lo, hi)| Some((parse(lo)?, parse(hi)?)))
                        .filter(|(lo, hi)| lo < hi)
                        .ok_or(
                            "--address-pool expects <lo>-<hi> in hex, e.g. 0x100000000-0x400000000",
                        )?,
                );
            }
            "--seed-targets" => {
                i += 1;
                parsed.seed_targets = Some(PathBuf::from(value(i, flag, "<file>")?));
            }
            "--seed-report" => {
                i += 1;
                parsed.seed_report = Some(PathBuf::from(value(i, flag, "<file>")?));
            }
            "--policy" => {
                i += 1;
                parsed
                    .policy_files
                    .push(PathBuf::from(value(i, flag, "<file>")?));
            }
            "--shims" => {
                let start = parsed.shims.len();
                while let Some(next) = args.get(i + 1).filter(|a| !a.starts_with('-')) {
                    parsed.shims.push(PathBuf::from(next));
                    i += 1;
                }
                if parsed.shims.len() == start {
                    return Err("--shims requires at least one <header> path".to_string());
                }
            }
            "--jobs" => {
                i += 1;
                parsed.jobs = Some(count(i, flag)?);
            }
            "--shard-size" => {
                i += 1;
                parsed.shard_size = Some(count(i, flag)?);
            }
            "--shard-instructions" => {
                i += 1;
                parsed.shard_instructions = Some(count(i, flag)?);
            }
            "--target" => {
                i += 1;
                target_spec = Some(value(i, flag, "<target>")?);
            }
            "--target-cpu" => {
                i += 1;
                target_cpu = Some(value(i, flag, "<cpu>")?);
            }
            "--memory-model" => {
                i += 1;
                parsed.memory = MemoryModel::from_name(&value(i, flag, "<model>")?)?;
            }
            "--indirect-dispatch" => {
                i += 1;
                parsed.indirect_dispatch =
                    IndirectDispatch::from_name(&value(i, flag, "<mode>")?)?;
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown option '{other}' for 'build-program'"));
            }
            positional => {
                if exe.is_some() {
                    return Err(format!(
                        "unexpected extra positional argument '{positional}'"
                    ));
                }
                exe = Some(PathBuf::from(positional));
            }
        }
        i += 1;
    }
    parsed.codegen = LiftTarget::parse(target_spec.as_deref().unwrap_or("wasm64"), target_cpu.as_deref())?;
    if target_cpu.is_some() && !parsed.codegen.is_native() {
        return Err("--target-cpu needs --target native".to_string());
    }
    parsed.exe = exe.ok_or("missing input executable path")?;
    parsed.out_dir = out_dir.ok_or(
        "missing --out <dir> (program outputs are derived from game code; use the gitignored work/)",
    )?;
    Ok(Some(parsed))
}

/// Builds the program and returns the report to print.
pub fn run_program(args: &ProgramArgs) -> Result<String, String> {
    let mut options = program::ProgramOptions::new(&args.exe, &args.out_dir);
    options.dlls = args.dlls.clone();
    options.dll_dirs = args.dll_dirs.clone();
    options.overrides = args.overrides.clone();
    for file in &args.policy_files {
        let text = fs::read_to_string(file)
            .map_err(|e| format!("failed to read policy file '{}': {e}", file.display()))?;
        options
            .overrides
            .add_file(&text)
            .map_err(|e| format!("{}: {e}", file.display()))?;
    }
    options.allow_allocator_split = args.allow_allocator_split;
    if let Some(pool) = args.address_pool {
        options.address_pool = pool;
    }
    let build = &mut options.build;
    if args.shims.is_empty() {
        let cwd = env::current_dir()
            .map_err(|e| format!("failed to determine current directory: {e}"))?;
        let (headers, root) = pipeline::checkout_shim_headers(&cwd).ok_or(
            "missing --shims <header>... (no playgta6web checkout found to take runtime/os/*.h from)",
        )?;
        build.shim_headers = headers;
        build.include_root = Some(root);
    } else {
        build.shim_headers = args.shims.clone();
    }
    build.emit_obj = args.emit_obj;
    build.target = args.codegen.clone();
    build.memory = args.memory;
    build.indirect_dispatch = args.indirect_dispatch;
    if let Some(jobs) = args.jobs {
        build.jobs = jobs;
    }
    if let Some(n) = args.shard_size {
        build.shard_size = n;
    }
    if let Some(n) = args.shard_instructions {
        build.shard_instructions = n;
    }
    if let Some(file) = &args.seed_targets {
        build.trace_seeds = Some(playgta6_recomp::trace_seed::load(file)?);
        build.trace_seed_report = args.seed_report.clone();
        if let Some(report) = &args.seed_report {
            fs::write(report, "")
                .map_err(|e| format!("failed to write '{}': {e}", report.display()))?;
        }
    } else if args.seed_report.is_some() {
        return Err("--seed-report needs --seed-targets".to_string());
    }
    let start = Instant::now();
    let report = program::build_program(&options, &mut |module, stage| {
        let module = if module.is_empty() { "program" } else { module };
        eprintln!(
            "[{:>9.2}s] {module}: {stage} done",
            start.elapsed().as_secs_f64()
        );
    })
    .map_err(|e| e.to_string())?;
    eprintln!(
        "Wrote {} modules and {} to {}",
        report.modules.len(),
        program::PROGRAM_MANIFEST_FILE,
        args.out_dir.display()
    );
    Ok(report.report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn parses_seed_options() {
        let a = parse_program_args(&args(&[
            "a.exe", "--out", "x", "--seed-targets", "t.txt", "--seed-report", "r.jsonl",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(a.seed_targets, Some(PathBuf::from("t.txt")));
        assert_eq!(a.seed_report, Some(PathBuf::from("r.jsonl")));
        assert!(parse_program_args(&args(&["a.exe", "--out", "x", "--seed-targets"])).is_err());
        let none = parse_program_args(&args(&["a.exe", "--out", "x"])).unwrap().unwrap();
        assert_eq!(none.seed_targets, None);
    }

    #[test]
    fn parses_program_options() {
        let parsed = parse_program_args(&args(&[
            "game.exe",
            "--out",
            "work/p",
            "--dll",
            "a.dll",
            "--dll-dir",
            "pkg",
            "--guest-dll",
            "Bink2Scarlett.dll",
            "--host-dll",
            "XCurl",
            "--bind",
            "VCRUNTIME140.dll!memcpy=host",
            "--allow-allocator-split",
            "--jobs",
            "3",
            "--address-pool",
            "0x10000-0x40000000",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(parsed.exe, PathBuf::from("game.exe"));
        assert_eq!(parsed.dlls, vec![PathBuf::from("a.dll")]);
        assert_eq!(parsed.dll_dirs, vec![PathBuf::from("pkg")]);
        assert_eq!(parsed.overrides.force_guest, vec!["bink2scarlett.dll"]);
        assert_eq!(parsed.overrides.roles.len(), 2);
        assert_eq!(
            parsed.overrides.rules[0].to_string(),
            "vcruntime140.dll!memcpy=host"
        );
        assert!(parsed.allow_allocator_split);
        assert_eq!(parsed.jobs, Some(3));
        assert_eq!(parsed.address_pool, Some((0x10000, 0x4000_0000)));
        assert!(
            parse_program_args(&args(&["a.exe", "--out", "x", "--address-pool", "5-1"])).is_err()
        );
        assert_eq!(parse_program_args(&args(&["--help"])), Ok(None));
        assert!(parse_program_args(&args(&["game.exe"])).is_err());
        assert!(parse_program_args(&args(&["--out", "x"])).is_err());
        assert!(parse_program_args(&args(&["a.exe", "--out", "x", "--bind", "nope"])).is_err());
        assert!(
            parse_program_args(&args(&["a.exe", "--out", "x", "--bind", "a!b=sideways"])).is_err()
        );
        assert_eq!(parsed.codegen, LiftTarget::Wasm64);
    }

    #[test]
    fn parses_the_codegen_target() {
        let parsed = parse_program_args(&args(&[
            "game.exe",
            "--out",
            "w",
            "--target",
            "native:x86_64-pc-windows-msvc",
            "--target-cpu",
            "znver2",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(
            parsed.codegen,
            LiftTarget::Native {
                triple: "x86_64-pc-windows-msvc".to_string(),
                cpu: "znver2".to_string()
            }
        );
        assert!(parse_program_args(&args(&["a.exe", "--out", "x", "--target", "arm"])).is_err());
        assert_eq!(parsed.memory, MemoryModel::default());
        let windowed =
            parse_program_args(&args(&["a.exe", "--out", "x", "--memory-model", "window"]))
                .unwrap()
                .unwrap();
        assert_eq!(windowed.memory, MemoryModel::translated_window(false));
        let outlined = parse_program_args(&args(&[
            "a.exe",
            "--out",
            "x",
            "--memory-model",
            "window-outline",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(outlined.memory, MemoryModel::translated_window_outline());
        assert!(
            parse_program_args(&args(&["a.exe", "--out", "x", "--memory-model", "flat"])).is_err()
        );
        assert!(
            parse_program_args(&args(&["a.exe", "--out", "x", "--target-cpu", "znver2"])).is_err()
        );
        assert_eq!(parsed.indirect_dispatch, IndirectDispatch::Table);
        let shard = parse_program_args(&args(&[
            "a.exe",
            "--out",
            "x",
            "--indirect-dispatch",
            "shard",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(shard.indirect_dispatch, IndirectDispatch::Shard);
        let table = parse_program_args(&args(&[
            "a.exe",
            "--out",
            "x",
            "--indirect-dispatch",
            "table",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(table.indirect_dispatch, IndirectDispatch::Table);
        assert!(
            parse_program_args(&args(&[
                "a.exe",
                "--out",
                "x",
                "--indirect-dispatch",
                "invalid",
            ]))
            .is_err()
        );
        assert!(
            parse_program_args(&args(&["a.exe", "--out", "x", "--indirect-dispatch"])).is_err()
        );
    }
}
