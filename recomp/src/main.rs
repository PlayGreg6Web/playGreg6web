mod lift_cmd;
mod program_cmd;
mod report;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use playgta6_recomp::discover::discover;
use playgta6_recomp::image::{self, ExecutableBytes, ImageOptions};
use playgta6_recomp::lift::target::LiftTarget;
use playgta6_recomp::pe;
use playgta6_recomp::pipeline::{self, PipelineOptions};
use playgta6_recomp::stubs;
use playgta6_recomp::summary::render_summary_json;

use pe::Pe64;

const ABI_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
enum InspectTarget {
    Path(PathBuf),
    FromEnv,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct InspectArgs {
    target: InspectTarget,
    markdown_out: Option<PathBuf>,
    json: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DiscoverArgs {
    pe_path: PathBuf,
    json_out: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PackImageArgs {
    pe_path: PathBuf,
    out_dir: PathBuf,
    options: ImageOptions,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StubsArgs {
    pe_path: PathBuf,
    shims: Vec<PathBuf>,
    out_dir: PathBuf,
    include_root: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BuildArgs {
    target: InspectTarget,
    out_dir: PathBuf,
    module: Option<String>,
    shims: Vec<PathBuf>,
    entries: Vec<u64>,
    emit_obj: bool,
    jobs: Option<usize>,
    shard_size: Option<usize>,
    shard_instructions: Option<usize>,
    codegen: LiftTarget,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Command {
    Help,
    Inspect(InspectArgs),
    Discover(DiscoverArgs),
    PackImage(PackImageArgs),
    Stubs(StubsArgs),
    Lift(lift_cmd::LiftArgs),
    SurveyInterfaces(playgta6_recomp::survey::cli::Args),
    Build(BuildArgs),
    BuildProgram(program_cmd::ProgramArgs),
}

fn print_usage() {
    eprintln!("playgta6-recomp (ABI v{ABI_VERSION})");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  playgta6-recomp inspect <pe-path> [--markdown <out-file>] [--json]");
    eprintln!("  playgta6-recomp inspect --from-env [--markdown <out-file>] [--json]");
    eprintln!("  playgta6-recomp discover <pe-path> [--out <json-file>]");
    eprintln!("  playgta6-recomp pack-image <pe-path> --out <dir> [--omit-code-bytes] [--no-headers]");
    eprintln!(
        "  playgta6-recomp stubs <pe-path> --shims <header>... --out <dir> [--include-root <dir>]"
    );
    eprintln!("  playgta6-recomp lift <pe-path> --out <dir> [--entry <va>]... [--emit-obj]");
    eprintln!("  playgta6-recomp {}", playgta6_recomp::survey::cli::USAGE);
    eprintln!(
        "  playgta6-recomp build <pe-path>|--from-env --out <dir> [--module <name>] [--shims <header>...] [--entry <va>]... [--emit-obj] [--jobs N] [--shard-size N] [--shard-instructions N] [--target wasm64|native|native:<triple>] [--target-cpu <cpu>]"
    );
    eprintln!("{}", program_cmd::USAGE);
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --from-env             Read GTA6_GAME_DIR from environment or .env and inspect top-level .exe files");
    eprintln!("  --markdown <out-file>  Write sanitized Markdown report to <out-file>");
    eprintln!("  --json                 Print sanitized JSON report to stdout");
    eprintln!("  --out <json-file>      Write the discovery summary (addresses and counts only) to <json-file>");
    eprintln!("  --out <dir>            pack-image: write <name>.pgim (contains game bytes; use gitignored work/) and <name>.pgim.json");
    eprintln!("  --omit-code-bytes      pack-image: map executable sections as zero-fill instead of storing their bytes");
    eprintln!("  --no-headers           pack-image: do not map the PE headers at the image base");
    eprintln!("  --shims <header>...    stubs: runtime headers declaring the extern \"C\" shims");
    eprintln!("  --out <dir>            stubs: write import thunks, seam header and manifest");
    eprintln!("  --include-root <dir>   stubs: spell generated #includes relative to <dir>");
    eprintln!("  --out <dir>            lift: write lifted.ll (and lifted.o) derived from game code; use gitignored work/");
    eprintln!("  --entry <va>           lift: extra function entry point (hex VA), repeatable");
    eprintln!("  --emit-obj             lift: also compile a wasm64 object with llc/clang");
    eprintln!("  --out <dir>            build: write lifted shards, registration table, import bridge, guest image,");
    eprintln!("                         report.txt and build-manifest.json (game-derived; gitignored work/ or outside the repo)");
    eprintln!("  --from-env             build: use the single top-level .exe of GTA6_GAME_DIR");
    eprintln!("  --module <name>        build: module name ([A-Za-z0-9_]): outputs go to <dir>/<name>/, table symbols get _<name> (default {})", pipeline::DEFAULT_MODULE);
    eprintln!("  --shims <header>...    build: runtime shim headers (default: runtime/os/*.h + runtime/gpu/d3d12_xs.h + runtime/guest/guest_callbacks.h of this checkout)");
    eprintln!("  --entry <va>           build: extra function entry point (hex VA), repeatable");
    eprintln!("  --emit-obj             build: also compile every shard and the registration table to wasm64 objects");
    eprintln!("  --jobs N               build: worker threads for lifting/compiling (default: CPU count; output is identical)");
    eprintln!("  --shard-size N         build: maximum functions per lifted shard (default {})", pipeline::DEFAULT_SHARD_SIZE);
    eprintln!("  --shard-instructions N build: maximum instructions per lifted shard (default {})", pipeline::DEFAULT_SHARD_INSTRUCTIONS);
    eprintln!("  --target <t>           build: wasm64 (default), native (this host's x86-64 triple) or native:<triple>");
    eprintln!("  --target-cpu <cpu>     build: LLVM CPU of a native target (default {})", playgta6_recomp::lift::target::NATIVE_DEFAULT_CPU);
    for line in program_cmd::OPTIONS {
        eprintln!("{line}");
    }
    eprintln!("  -h, --help            Show this help message");
    eprintln!();
    eprintln!("Pipeline target: Xbox Series x86-64 PE -> LLVM IR -> Wasm64 (memory64).");
}

fn parse_cli(args: &[String]) -> Result<Command, String> {
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        return Ok(Command::Help);
    }

    match args[0].as_str() {
        "inspect" => parse_inspect_args(&args[1..]),
        "discover" => parse_discover_args(&args[1..]),
        "pack-image" => parse_pack_image_args(&args[1..]),
        "stubs" => parse_stubs_args(&args[1..]),
        "lift" => {
            lift_cmd::parse_lift_args(&args[1..]).map(|a| a.map_or(Command::Help, Command::Lift))
        }
        "survey-interfaces" => playgta6_recomp::survey::cli::parse_args(&args[1..])
            .map(|a| a.map_or(Command::Help, Command::SurveyInterfaces)),
        "build" => parse_build_args(&args[1..]),
        "build-program" => program_cmd::parse_program_args(&args[1..])
            .map(|a| a.map_or(Command::Help, Command::BuildProgram)),
        other => Err(format!(
            "unknown subcommand '{other}'. Expected 'inspect', 'discover', 'pack-image', 'stubs', 'lift', 'survey-interfaces', 'build' or 'build-program'. Run with --help for usage."
        )),
    }
}

fn parse_build_args(args: &[String]) -> Result<Command, String> {
    let mut pe_path: Option<PathBuf> = None;
    let mut from_env = false;
    let mut out_dir: Option<PathBuf> = None;
    let mut module = None;
    let mut shims: Vec<PathBuf> = Vec::new();
    let mut entries = Vec::new();
    let mut emit_obj = false;
    let mut jobs = None;
    let mut shard_size = None;
    let mut shard_instructions = None;
    let mut target_spec: Option<String> = None;
    let mut target_cpu: Option<String> = None;

    let count = |flag: &str, value: Option<&String>| -> Result<usize, String> {
        value
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .ok_or_else(|| format!("{flag} requires a positive integer"))
    };

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--from-env" => from_env = true,
            "--emit-obj" => emit_obj = true,
            "--out" => {
                i += 1;
                match args.get(i) {
                    Some(dir) if !dir.starts_with('-') => out_dir = Some(PathBuf::from(dir)),
                    _ => return Err("--out requires an output <dir> argument".to_string()),
                }
            }
            "--shims" => {
                let start = shims.len();
                while let Some(next) = args.get(i + 1).filter(|a| !a.starts_with('-')) {
                    shims.push(PathBuf::from(next));
                    i += 1;
                }
                if shims.len() == start {
                    return Err("--shims requires at least one <header> path".to_string());
                }
            }
            "--module" => {
                i += 1;
                match args.get(i) {
                    Some(name) if pipeline::is_valid_module_name(name) => {
                        module = Some(name.clone());
                    }
                    _ => {
                        return Err(
                            "--module requires a <name> of 1-64 characters from [A-Za-z0-9_]"
                                .to_string(),
                        );
                    }
                }
            }
            "--entry" => {
                i += 1;
                let Some(text) = args.get(i) else {
                    return Err("--entry requires a hexadecimal <va> argument".to_string());
                };
                entries.push(lift_cmd::parse_va(text)?);
            }
            "--jobs" => {
                i += 1;
                jobs = Some(count("--jobs", args.get(i))?);
            }
            "--shard-size" => {
                i += 1;
                shard_size = Some(count("--shard-size", args.get(i))?);
            }
            "--shard-instructions" => {
                i += 1;
                shard_instructions = Some(count("--shard-instructions", args.get(i))?);
            }
            "--target" | "--target-cpu" => {
                let flag = args[i].clone();
                i += 1;
                let Some(value) = args.get(i).filter(|v| !v.starts_with('-')) else {
                    return Err(format!("{flag} requires a value"));
                };
                if flag == "--target" {
                    target_spec = Some(value.clone());
                } else {
                    target_cpu = Some(value.clone());
                }
            }
            flag if flag.starts_with('-') => {
                return Err(format!("unknown option '{flag}' for 'build'"));
            }
            positional => {
                if pe_path.is_some() {
                    return Err(format!(
                        "unexpected extra positional argument '{positional}'"
                    ));
                }
                pe_path = Some(PathBuf::from(positional));
            }
        }
        i += 1;
    }

    let target = match (pe_path, from_env) {
        (Some(_), true) => return Err("cannot specify both <pe-path> and --from-env".to_string()),
        (None, false) => {
            return Err("missing input PE path (provide <pe-path> or --from-env)".to_string());
        }
        (Some(path), false) => InspectTarget::Path(path),
        (None, true) => InspectTarget::FromEnv,
    };
    let out_dir = out_dir.ok_or_else(|| {
        "missing --out <dir> (build outputs are derived from game code; use the gitignored work/)"
            .to_string()
    })?;
    let codegen = LiftTarget::parse(
        target_spec.as_deref().unwrap_or("wasm64"),
        target_cpu.as_deref(),
    )?;
    if target_cpu.is_some() && !codegen.is_native() {
        return Err("--target-cpu needs --target native".to_string());
    }
    Ok(Command::Build(BuildArgs {
        target,
        out_dir,
        module,
        shims,
        entries,
        emit_obj,
        jobs,
        shard_size,
        shard_instructions,
        codegen,
    }))
}

fn parse_inspect_args(args: &[String]) -> Result<Command, String> {
    let mut pe_path: Option<PathBuf> = None;
    let mut from_env = false;
    let mut markdown_out: Option<PathBuf> = None;
    let mut json = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--from-env" => {
                from_env = true;
            }
            "--json" => {
                json = true;
            }
            "--markdown" => {
                i += 1;
                let Some(path_str) = args.get(i) else {
                    return Err("--markdown requires an <out-file> path argument".to_string());
                };
                if path_str.starts_with('-') {
                    return Err(format!(
                        "--markdown requires an <out-file> path argument, got '{path_str}'"
                    ));
                }
                markdown_out = Some(PathBuf::from(path_str));
            }
            flag if flag.starts_with('-') => {
                return Err(format!("unknown option '{flag}' for 'inspect'"));
            }
            positional => {
                if pe_path.is_some() {
                    return Err(format!(
                        "unexpected extra positional argument '{positional}'"
                    ));
                }
                pe_path = Some(PathBuf::from(positional));
            }
        }
        i += 1;
    }

    let target = match (pe_path, from_env) {
        (Some(_), true) => {
            return Err("cannot specify both <pe-path> and --from-env".to_string());
        }
        (None, false) => {
            return Err("missing input PE path (provide <pe-path> or --from-env)".to_string());
        }
        (Some(path), false) => InspectTarget::Path(path),
        (None, true) => InspectTarget::FromEnv,
    };

    Ok(Command::Inspect(InspectArgs {
        target,
        markdown_out,
        json,
    }))
}

fn parse_discover_args(args: &[String]) -> Result<Command, String> {
    let mut pe_path: Option<PathBuf> = None;
    let mut json_out: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--out" => {
                i += 1;
                let Some(path_str) = args.get(i) else {
                    return Err("--out requires a <json-file> path argument".to_string());
                };
                if path_str.starts_with('-') {
                    return Err(format!(
                        "--out requires a <json-file> path argument, got '{path_str}'"
                    ));
                }
                json_out = Some(PathBuf::from(path_str));
            }
            flag if flag.starts_with('-') => {
                return Err(format!("unknown option '{flag}' for 'discover'"));
            }
            positional => {
                if pe_path.is_some() {
                    return Err(format!(
                        "unexpected extra positional argument '{positional}'"
                    ));
                }
                pe_path = Some(PathBuf::from(positional));
            }
        }
        i += 1;
    }

    let pe_path = pe_path.ok_or_else(|| "missing input PE path <pe-path>".to_string())?;
    Ok(Command::Discover(DiscoverArgs { pe_path, json_out }))
}

fn parse_pack_image_args(args: &[String]) -> Result<Command, String> {
    let mut pe_path: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut options = ImageOptions::default();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--omit-code-bytes" => options.executable_bytes = ExecutableBytes::Omit,
            "--no-headers" => options.map_headers = false,
            "--out" => {
                i += 1;
                let Some(path_str) = args.get(i) else {
                    return Err("--out requires a <dir> path argument".to_string());
                };
                if path_str.starts_with('-') {
                    return Err(format!(
                        "--out requires a <dir> path argument, got '{path_str}'"
                    ));
                }
                out_dir = Some(PathBuf::from(path_str));
            }
            flag if flag.starts_with('-') => {
                return Err(format!("unknown option '{flag}' for 'pack-image'"));
            }
            positional => {
                if pe_path.is_some() {
                    return Err(format!(
                        "unexpected extra positional argument '{positional}'"
                    ));
                }
                pe_path = Some(PathBuf::from(positional));
            }
        }
        i += 1;
    }

    let pe_path = pe_path.ok_or_else(|| "missing input PE path <pe-path>".to_string())?;
    let out_dir = out_dir.ok_or_else(|| {
        "missing --out <dir> (the packed image contains game bytes; use a gitignored directory such as work/)"
            .to_string()
    })?;
    Ok(Command::PackImage(PackImageArgs {
        pe_path,
        out_dir,
        options,
    }))
}

fn parse_stubs_args(args: &[String]) -> Result<Command, String> {
    let mut pe_path: Option<PathBuf> = None;
    let mut shims: Vec<PathBuf> = Vec::new();
    let mut out_dir: Option<PathBuf> = None;
    let mut include_root: Option<PathBuf> = None;

    let dir_value = |flag: &str, value: Option<&String>| -> Result<PathBuf, String> {
        match value {
            Some(v) if !v.starts_with('-') => Ok(PathBuf::from(v)),
            Some(v) => Err(format!("{flag} requires a <dir> path argument, got '{v}'")),
            None => Err(format!("{flag} requires a <dir> path argument")),
        }
    };

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--shims" => {
                let start = shims.len();
                while let Some(next) = args.get(i + 1).filter(|a| !a.starts_with('-')) {
                    shims.push(PathBuf::from(next));
                    i += 1;
                }
                if shims.len() == start {
                    return Err("--shims requires at least one <header> path".to_string());
                }
            }
            "--out" => {
                i += 1;
                out_dir = Some(dir_value("--out", args.get(i))?);
            }
            "--include-root" => {
                i += 1;
                include_root = Some(dir_value("--include-root", args.get(i))?);
            }
            flag if flag.starts_with('-') => {
                return Err(format!("unknown option '{flag}' for 'stubs'"));
            }
            positional => {
                if pe_path.is_some() {
                    return Err(format!(
                        "unexpected extra positional argument '{positional}'"
                    ));
                }
                pe_path = Some(PathBuf::from(positional));
            }
        }
        i += 1;
    }

    let pe_path = pe_path.ok_or_else(|| "missing input PE path <pe-path>".to_string())?;
    if shims.is_empty() {
        return Err("missing --shims <header>... (runtime shim headers)".to_string());
    }
    let out_dir = out_dir.ok_or_else(|| "missing --out <dir>".to_string())?;
    Ok(Command::Stubs(StubsArgs {
        pe_path,
        shims,
        out_dir,
        include_root,
    }))
}

fn parse_dotenv_key(content: &str, target_key: &str) -> Option<String> {
    let mut found: Option<String> = None;
    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, val)) = line.split_once('=') else {
            continue;
        };
        if key.trim() != target_key {
            continue;
        }
        let mut cleaned = val.trim();
        if (cleaned.starts_with('"') && cleaned.ends_with('"') && cleaned.len() >= 2)
            || (cleaned.starts_with('\'') && cleaned.ends_with('\'') && cleaned.len() >= 2)
        {
            cleaned = &cleaned[1..cleaned.len() - 1];
        }
        let cleaned = cleaned.trim();
        if !cleaned.is_empty() {
            found = Some(cleaned.to_string());
        }
    }
    found
}

fn resolve_game_dir_from_env(start_dir: &Path) -> Result<PathBuf, String> {
    resolve_game_dir_with_env(env::var("GTA6_GAME_DIR").ok(), start_dir)
}

fn resolve_game_dir_with_env(
    env_var: Option<String>,
    start_dir: &Path,
) -> Result<PathBuf, String> {
    if let Some(val) = env_var {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            return Ok(PathBuf::from(trimmed));
        }
    }

    let mut cur = Some(start_dir);
    while let Some(dir) = cur {
        let env_path = dir.join(".env");
        if env_path.is_file()
            && let Ok(contents) = fs::read_to_string(&env_path)
            && let Some(val) = parse_dotenv_key(&contents, "GTA6_GAME_DIR")
        {
            return Ok(PathBuf::from(val));
        }
        cur = dir.parent();
    }

    Err(
        "GTA6_GAME_DIR is not set in the process environment or .env. \
         Set GTA6_GAME_DIR or run: python tools/local_env.py --set-game \"<path>\""
            .to_string(),
    )
}

fn find_top_level_exes(game_dir: &Path) -> Result<Vec<PathBuf>, String> {
    if !game_dir.is_dir() {
        return Err(format!(
            "GTA6_GAME_DIR does not exist or is not a directory: {}",
            game_dir.display()
        ));
    }

    let entries = fs::read_dir(game_dir).map_err(|e| {
        format!(
            "failed to read GTA6_GAME_DIR '{}': {e}",
            game_dir.display()
        )
    })?;

    let mut exes = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| {
            format!(
                "failed to inspect directory entry in '{}': {e}",
                game_dir.display()
            )
        })?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let is_exe = path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("exe"));
        if is_exe {
            exes.push(path);
        }
    }

    exes.sort_by(|a, b| {
        let a_name = a
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let b_name = b
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        a_name.cmp(&b_name).then_with(|| a.cmp(b))
    });

    if exes.is_empty() {
        return Err(format!(
            "no top-level .exe files found in GTA6_GAME_DIR: {}",
            game_dir.display()
        ));
    }

    Ok(exes)
}

fn sanitized_source_label(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| "unknown.exe".to_string())
}

fn run_inspect(args: &InspectArgs) -> Result<(), String> {
    let paths: Vec<PathBuf> = match &args.target {
        InspectTarget::Path(p) => vec![p.clone()],
        InspectTarget::FromEnv => {
            let cwd = env::current_dir()
                .map_err(|e| format!("failed to determine current directory: {e}"))?;
            let game_dir = resolve_game_dir_from_env(&cwd)?;
            find_top_level_exes(&game_dir)?
        }
    };

    let mut parsed: Vec<(String, Pe64)> = Vec::with_capacity(paths.len());
    for path in &paths {
        let bytes = fs::read(path)
            .map_err(|e| format!("failed to read PE file '{}': {e}", path.display()))?;
        let pe = Pe64::parse(&bytes)
            .map_err(|e| format!("failed to parse PE64 '{}': {e}", path.display()))?;
        parsed.push((sanitized_source_label(path), pe));
    }

    let markdown_combined = || -> String {
        parsed
            .iter()
            .map(|(label, pe)| report::render_markdown(pe, label))
            .collect::<Vec<_>>()
            .join("\n---\n\n")
    };

    if let Some(out_path) = &args.markdown_out {
        if let Some(parent) = out_path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "failed to create output directory '{}': {e}",
                    parent.display()
                )
            })?;
        }
        let md = markdown_combined();
        fs::write(out_path, md)
            .map_err(|e| format!("failed to write Markdown report '{}': {e}", out_path.display()))?;
        eprintln!(
            "Wrote sanitized PE64 Markdown report ({} file{}) to {}",
            parsed.len(),
            if parsed.len() == 1 { "" } else { "s" },
            out_path.display()
        );
    }

    if args.json {
        if parsed.len() == 1 {
            let (label, pe) = &parsed[0];
            print!("{}", report::render_json(pe, label));
        } else {
            print!("{}", report::render_json_many(&parsed));
        }
    } else if args.markdown_out.is_none() {
        print!("{}", markdown_combined());
    }

    Ok(())
}

fn run_discover(args: &DiscoverArgs) -> Result<(), String> {
    let path = &args.pe_path;
    let bytes =
        fs::read(path).map_err(|e| format!("failed to read PE file '{}': {e}", path.display()))?;
    let pe = Pe64::parse(&bytes)
        .map_err(|e| format!("failed to parse PE64 '{}': {e}", path.display()))?;
    let pipeline::Seeded {
        image, pdata, eh, ..
    } = pipeline::seed(&pe, &bytes);
    if let Some(first) = pdata.diagnostics.first() {
        eprintln!(
            "Warning: {} .pdata diagnostic(s), first: {first}",
            pdata.diagnostics.len()
        );
    }
    if let Some(first) = eh.diagnostics.first() {
        eprintln!(
            "Warning: {} EH metadata diagnostic(s), first: {first}",
            eh.diagnostics.len()
        );
    }
    let summary = render_summary_json(&image, &discover(&image));

    match &args.json_out {
        Some(out_path) => {
            if let Some(parent) = out_path.parent()
                && !parent.as_os_str().is_empty()
            {
                fs::create_dir_all(parent).map_err(|e| {
                    format!(
                        "failed to create output directory '{}': {e}",
                        parent.display()
                    )
                })?;
            }
            fs::write(out_path, summary).map_err(|e| {
                format!(
                    "failed to write discovery summary '{}': {e}",
                    out_path.display()
                )
            })?;
            eprintln!(
                "Wrote discovery summary for {} to {}",
                sanitized_source_label(path),
                out_path.display()
            );
        }
        None => print!("{summary}"),
    }
    Ok(())
}

fn run_pack_image(args: &PackImageArgs) -> Result<(), String> {
    let path = &args.pe_path;
    let label = sanitized_source_label(path);
    let bytes =
        fs::read(path).map_err(|e| format!("failed to read PE file '{}': {e}", path.display()))?;
    let pe = Pe64::parse(&bytes)
        .map_err(|e| format!("failed to parse PE64 '{}': {e}", path.display()))?;
    let built = image::build_guest_image(&pe, &bytes, args.options)
        .map_err(|e| format!("failed to build guest image for {label}: {e}"))?;
    let container = image::write_image(&built.image)
        .map_err(|e| format!("failed to serialise guest image for {label}: {e}"))?;
    let manifest = image::render_manifest_json(&built.image, &built.diagnostics)
        .map_err(|e| format!("failed to render manifest for {label}: {e}"))?;

    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("image");
    let image_path = args.out_dir.join(format!("{stem}.pgim"));
    let manifest_path = args.out_dir.join(format!("{stem}.pgim.json"));
    fs::create_dir_all(&args.out_dir).map_err(|e| {
        format!(
            "failed to create output directory '{}': {e}",
            args.out_dir.display()
        )
    })?;
    fs::write(&image_path, &container)
        .map_err(|e| format!("failed to write '{}': {e}", image_path.display()))?;
    fs::write(&manifest_path, manifest)
        .map_err(|e| format!("failed to write '{}': {e}", manifest_path.display()))?;

    print!(
        "Guest image for {label}: {}",
        image::render_summary_text(&built.image, &built.diagnostics)
    );
    eprintln!(
        "Wrote {} ({} bytes) and {}",
        image_path.display(),
        container.len(),
        manifest_path.display()
    );
    Ok(())
}

fn run_stubs(args: &StubsArgs) -> Result<(), String> {
    let path = &args.pe_path;
    let bytes =
        fs::read(path).map_err(|e| format!("failed to read PE file '{}': {e}", path.display()))?;
    let pe = Pe64::parse(&bytes)
        .map_err(|e| format!("failed to parse PE64 '{}': {e}", path.display()))?;

    let mut headers = Vec::with_capacity(args.shims.len());
    for header in &args.shims {
        headers.push(
            stubs::load_shim_header(header, args.include_root.as_deref())
                .map_err(|e| format!("failed to read shim header '{}': {e}", header.display()))?,
        );
    }
    let catalogue = stubs::Catalogue::scan(&headers);
    let plan = stubs::plan_bridge(&pe, &catalogue);
    stubs::render_bridge(&plan, &catalogue)
        .write_to(&args.out_dir)
        .map_err(|e| {
            format!(
                "failed to write import bridge to '{}': {e}",
                args.out_dir.display()
            )
        })?;
    print!("{}", stubs::render_coverage(&plan, &catalogue));
    eprintln!(
        "Wrote import bridge for {} to {}",
        sanitized_source_label(path),
        args.out_dir.display()
    );
    Ok(())
}

fn run_build(args: &BuildArgs) -> Result<String, String> {
    let pe_path = match &args.target {
        InspectTarget::Path(path) => path.clone(),
        InspectTarget::FromEnv => {
            let cwd = env::current_dir()
                .map_err(|e| format!("failed to determine current directory: {e}"))?;
            let mut exes = find_top_level_exes(&resolve_game_dir_from_env(&cwd)?)?;
            if exes.len() != 1 {
                return Err(format!(
                    "--from-env needs exactly one top-level .exe in GTA6_GAME_DIR, found {}; pass <pe-path> instead",
                    exes.len()
                ));
            }
            exes.remove(0)
        }
    };
    let mut options = PipelineOptions::new(pe_path, &args.out_dir);
    if let Some(module) = &args.module {
        options.module = module.clone();
    }
    if args.shims.is_empty() {
        let cwd = env::current_dir()
            .map_err(|e| format!("failed to determine current directory: {e}"))?;
        let (headers, root) = pipeline::checkout_shim_headers(&cwd).ok_or(
            "missing --shims <header>... (no playgta6web checkout found to take runtime/os/*.h from)",
        )?;
        options.shim_headers = headers;
        options.include_root = Some(root);
    } else {
        options.shim_headers = args.shims.clone();
    }
    options.extra_entries = args.entries.clone();
    options.emit_obj = args.emit_obj;
    options.target = args.codegen.clone();
    if let Some(jobs) = args.jobs {
        options.jobs = jobs;
    }
    if let Some(shard_size) = args.shard_size {
        options.shard_size = shard_size;
    }
    if let Some(limit) = args.shard_instructions {
        options.shard_instructions = limit;
    }
    let start = std::time::Instant::now();
    let report = pipeline::run_with_progress(&options, &mut |stage| {
        eprintln!("[{:>9.2}s] {stage} done", start.elapsed().as_secs_f64());
    })
    .map_err(|e| e.to_string())?;
    eprintln!(
        "Wrote {} files and {} to {}",
        report.files.len(),
        pipeline::MANIFEST_FILE,
        options.module_dir().display()
    );
    Ok(report.report)
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match parse_cli(&args) {
        Ok(Command::Help) => {
            print_usage();
            ExitCode::SUCCESS
        }
        Ok(Command::Inspect(inspect_args)) => match run_inspect(&inspect_args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("Error: {err}");
                ExitCode::FAILURE
            }
        },
        Ok(Command::Discover(discover_args)) => match run_discover(&discover_args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("Error: {err}");
                ExitCode::FAILURE
            }
        },
        Ok(Command::PackImage(pack_args)) => match run_pack_image(&pack_args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("Error: {err}");
                ExitCode::FAILURE
            }
        },
        Ok(Command::Stubs(stubs_args)) => match run_stubs(&stubs_args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("Error: {err}");
                ExitCode::FAILURE
            }
        },
        Ok(Command::SurveyInterfaces(a)) => match playgta6_recomp::survey::cli::run(&a) {
            Ok(report) => {
                print!("{report}");
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("Error: {err}");
                ExitCode::FAILURE
            }
        },
        Ok(Command::Lift(lift_args)) => match lift_cmd::run_lift(&lift_args) {
            Ok(summary) => {
                print!("{summary}");
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("Error: {err}");
                ExitCode::FAILURE
            }
        },
        Ok(Command::Build(build_args)) => match run_build(&build_args) {
            Ok(report) => {
                print!("{report}");
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("Error: {err}");
                ExitCode::FAILURE
            }
        },
        Ok(Command::BuildProgram(program_args)) => match program_cmd::run_program(&program_args) {
            Ok(report) => {
                print!("{report}");
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("Error: {err}");
                ExitCode::FAILURE
            }
        },
        Err(err) => {
            eprintln!("Error: {err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pe::test_util::build_synthetic_pe64;

    fn vec_s(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn cli_parses_inspect_modes() {
        assert_eq!(parse_cli(&[]), Ok(Command::Help));
        assert_eq!(parse_cli(&vec_s(&["--help"])), Ok(Command::Help));
        assert_eq!(
            parse_cli(&vec_s(&[
                "inspect",
                "game.exe",
                "--markdown",
                "out/report.md",
                "--json"
            ])),
            Ok(Command::Inspect(InspectArgs {
                target: InspectTarget::Path(PathBuf::from("game.exe")),
                markdown_out: Some(PathBuf::from("out/report.md")),
                json: true,
            }))
        );
        assert_eq!(
            parse_cli(&vec_s(&["inspect", "--from-env", "--json"])),
            Ok(Command::Inspect(InspectArgs {
                target: InspectTarget::FromEnv,
                markdown_out: None,
                json: true,
            }))
        );
    }

    #[test]
    fn cli_rejects_invalid_combinations() {
        assert!(parse_cli(&vec_s(&["inspect"])).is_err());
        assert!(parse_cli(&vec_s(&["inspect", "a.exe", "--from-env"])).is_err());
        assert!(parse_cli(&vec_s(&["inspect", "a.exe", "--markdown"])).is_err());
        assert!(parse_cli(&vec_s(&["inspect", "a.exe", "--unknown"])).is_err());
        assert!(parse_cli(&vec_s(&["bogus"])).is_err());
    }

    #[test]
    fn dotenv_parser_extracts_game_dir() {
        let sample = "\
# comment
GTA6_DATA_DIR=local-data
GTA6_GAME_DIR=\"C:/Games/GTA6\"
";
        assert_eq!(
            parse_dotenv_key(sample, "GTA6_GAME_DIR").as_deref(),
            Some("C:/Games/GTA6")
        );
        assert_eq!(parse_dotenv_key("GTA6_GAME_DIR=\n", "GTA6_GAME_DIR"), None);
    }

    #[test]
    fn find_top_level_exes_and_end_to_end_markdown_output() {
        let unique = format!(
            "playgta6_recomp_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        );
        let temp_dir = env::temp_dir().join(unique);
        fs::create_dir_all(temp_dir.join("nested")).expect("create temp dir");

        let pe_bytes = build_synthetic_pe64(false);
        let exe_path = temp_dir.join("SyntheticGame.exe");
        fs::write(&exe_path, &pe_bytes).expect("write synthetic pe");
        // Nested exe should NOT be picked up by top-level scan
        fs::write(temp_dir.join("nested").join("Sub.exe"), &pe_bytes).expect("write nested pe");
        fs::write(temp_dir.join("readme.txt"), b"ignore").expect("write txt");

        let found = find_top_level_exes(&temp_dir).expect("should find top-level exe");
        assert_eq!(found, vec![exe_path.clone()]);

        let md_out = temp_dir.join("report.md");
        run_inspect(&InspectArgs {
            target: InspectTarget::Path(exe_path),
            markdown_out: Some(md_out.clone()),
            json: false,
        })
        .expect("run_inspect should succeed");

        let written = fs::read_to_string(&md_out).expect("read written report");
        assert!(written.contains("SyntheticGame.exe"));
        assert!(written.contains("VirtualAlloc"));
        assert!(written.contains("D3D12CreateDevice"));
        assert!(written.contains("GameMain"));

        // Verify .env resolution (including walking up from child subdir) and env var precedence
        let dotenv_path = temp_dir.join(".env");
        fs::write(&dotenv_path, format!("GTA6_GAME_DIR={}\n", temp_dir.display()))
            .expect("write .env");
        assert_eq!(
            resolve_game_dir_with_env(None, &temp_dir.join("nested")).expect("read from parent .env"),
            temp_dir
        );
        assert_eq!(
            resolve_game_dir_with_env(Some("/override/path".to_string()), &temp_dir)
                .expect("env var should take precedence"),
            PathBuf::from("/override/path")
        );

        let _ = fs::remove_file(&dotenv_path);
        let err = resolve_game_dir_with_env(None, &temp_dir)
            .expect_err("should error clearly when GTA6_GAME_DIR is unset");
        assert!(err.contains("GTA6_GAME_DIR"));

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn cli_parses_discover() {
        assert_eq!(
            parse_cli(&vec_s(&["discover", "game.exe", "--out", "work/cfg.json"])),
            Ok(Command::Discover(DiscoverArgs {
                pe_path: PathBuf::from("game.exe"),
                json_out: Some(PathBuf::from("work/cfg.json")),
            }))
        );
        assert_eq!(
            parse_cli(&vec_s(&["discover", "game.exe"])),
            Ok(Command::Discover(DiscoverArgs {
                pe_path: PathBuf::from("game.exe"),
                json_out: None,
            }))
        );
        assert!(parse_cli(&vec_s(&["discover"])).is_err());
        assert!(parse_cli(&vec_s(&["discover", "a.exe", "--out"])).is_err());
        assert!(parse_cli(&vec_s(&["discover", "a.exe", "b.exe"])).is_err());
        assert!(parse_cli(&vec_s(&["discover", "a.exe", "--json"])).is_err());
    }

    #[test]
    fn discover_end_to_end_writes_sanitized_summary() {
        let temp_dir = env::temp_dir().join(format!(
            "playgta6_recomp_discover_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&temp_dir).expect("create temp dir");
        let exe_path = temp_dir.join("SyntheticGame.exe");
        fs::write(&exe_path, build_synthetic_pe64(false)).expect("write synthetic pe");

        let out = temp_dir.join("out").join("cfg.json");
        run_discover(&DiscoverArgs {
            pe_path: exe_path,
            json_out: Some(out.clone()),
        })
        .expect("run_discover should succeed");

        let json = fs::read_to_string(&out).expect("read summary");
        assert!(json.contains("\"image_base\": \"0x140000000\""));
        assert!(json.contains("\"function_count\": 4"));
        assert!(json.contains("\"entry\": \"0x140001010\""));
        assert!(!json.contains("SyntheticGame"), "no file names or paths");

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn cli_parses_pack_image() {
        assert_eq!(
            parse_cli(&vec_s(&["pack-image", "game.exe", "--out", "work/image"])),
            Ok(Command::PackImage(PackImageArgs {
                pe_path: PathBuf::from("game.exe"),
                out_dir: PathBuf::from("work/image"),
                options: ImageOptions::default(),
            }))
        );
        assert_eq!(
            parse_cli(&vec_s(&[
                "pack-image",
                "--omit-code-bytes",
                "game.exe",
                "--no-headers",
                "--out",
                "w"
            ])),
            Ok(Command::PackImage(PackImageArgs {
                pe_path: PathBuf::from("game.exe"),
                out_dir: PathBuf::from("w"),
                options: ImageOptions {
                    map_headers: false,
                    executable_bytes: ExecutableBytes::Omit,
                },
            }))
        );
        assert!(
            parse_cli(&vec_s(&["pack-image", "game.exe"])).is_err(),
            "--out is required"
        );
        assert!(parse_cli(&vec_s(&["pack-image", "--out", "w"])).is_err());
        assert!(parse_cli(&vec_s(&["pack-image", "a.exe", "--out"])).is_err());
        assert!(parse_cli(&vec_s(&["pack-image", "a.exe", "b.exe", "--out", "w"])).is_err());
        assert!(parse_cli(&vec_s(&["pack-image", "a.exe", "--out", "w", "--json"])).is_err());
    }

    #[test]
    fn pack_image_end_to_end_writes_container_and_manifest() {
        let temp_dir = env::temp_dir().join(format!(
            "playgta6_recomp_pack_image_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&temp_dir).expect("create temp dir");
        let exe_path = temp_dir.join("SyntheticGame.exe");
        let pe_bytes = build_synthetic_pe64(false);
        fs::write(&exe_path, &pe_bytes).expect("write synthetic pe");

        let out_dir = temp_dir.join("out");
        run_pack_image(&PackImageArgs {
            pe_path: exe_path,
            out_dir: out_dir.clone(),
            options: ImageOptions::default(),
        })
        .expect("run_pack_image should succeed");

        let container = fs::read(out_dir.join("SyntheticGame.pgim")).expect("read container");
        let packed = image::read_image(&container).expect("container validates");
        assert_eq!(packed.image_base, 0x1_4000_0000);
        assert_eq!(packed.sections.len(), 3);
        let text = packed.section_at(0x1000).expect(".text mapped");
        assert_eq!(text.data, &pe_bytes[0x200..0x200 + text.data.len()]);

        let manifest =
            fs::read_to_string(out_dir.join("SyntheticGame.pgim.json")).expect("read manifest");
        assert!(manifest.contains("\"format\": \"PGIM\""));
        assert!(manifest.contains(&format!("\"container_bytes\": {}", container.len())));
        assert!(
            !manifest.contains("SyntheticGame"),
            "no file names or paths"
        );
        assert!(!manifest.contains("KERNEL32"), "no data bytes");

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn cli_parses_stubs() {
        assert_eq!(
            parse_cli(&vec_s(&[
                "stubs",
                "game.exe",
                "--shims",
                "runtime/os/win32_kernel.h",
                "runtime/os/xgameruntime.h",
                "--out",
                "work/bridge",
                "--include-root",
                "runtime",
            ])),
            Ok(Command::Stubs(StubsArgs {
                pe_path: PathBuf::from("game.exe"),
                shims: vec![
                    PathBuf::from("runtime/os/win32_kernel.h"),
                    PathBuf::from("runtime/os/xgameruntime.h"),
                ],
                out_dir: PathBuf::from("work/bridge"),
                include_root: Some(PathBuf::from("runtime")),
            }))
        );
        assert!(parse_cli(&vec_s(&["stubs", "a.exe", "--out", "o"])).is_err());
        assert!(parse_cli(&vec_s(&["stubs", "a.exe", "--shims", "--out", "o"])).is_err());
        assert!(parse_cli(&vec_s(&["stubs", "a.exe", "--shims", "h.h"])).is_err());
        assert!(parse_cli(&vec_s(&["stubs", "--shims", "h.h", "--out", "o"])).is_err());
        assert!(parse_cli(&vec_s(&["stubs", "a.exe", "--shims", "h.h", "--out"])).is_err());
        assert!(parse_cli(&vec_s(&["stubs", "a.exe", "--shims", "h.h", "--x"])).is_err());
    }

    #[test]
    fn cli_parses_build() {
        assert_eq!(
            parse_cli(&vec_s(&[
                "build",
                "game.exe",
                "--out",
                "work/build",
                "--module",
                "game",
                "--shims",
                "a.h",
                "b.h",
                "--entry",
                "0x140001000",
                "--emit-obj",
                "--jobs",
                "8",
                "--shard-size",
                "512",
                "--shard-instructions",
                "1000",
            ])),
            Ok(Command::Build(BuildArgs {
                target: InspectTarget::Path(PathBuf::from("game.exe")),
                out_dir: PathBuf::from("work/build"),
                module: Some("game".to_string()),
                shims: vec![PathBuf::from("a.h"), PathBuf::from("b.h")],
                entries: vec![0x1_4000_1000],
                emit_obj: true,
                jobs: Some(8),
                shard_size: Some(512),
                shard_instructions: Some(1000),
                codegen: LiftTarget::Wasm64,
            }))
        );
        assert_eq!(
            parse_cli(&vec_s(&["build", "--from-env", "--out", "w"])),
            Ok(Command::Build(BuildArgs {
                target: InspectTarget::FromEnv,
                out_dir: PathBuf::from("w"),
                module: None,
                shims: Vec::new(),
                entries: Vec::new(),
                emit_obj: false,
                jobs: None,
                shard_size: None,
                shard_instructions: None,
                codegen: LiftTarget::Wasm64,
            }))
        );
        assert!(parse_cli(&vec_s(&["build", "game.exe"])).is_err(), "--out is required");
        assert!(parse_cli(&vec_s(&["build", "--out", "w"])).is_err());
        assert!(parse_cli(&vec_s(&["build", "a.exe", "--from-env", "--out", "w"])).is_err());
        assert!(parse_cli(&vec_s(&["build", "a.exe", "--out", "w", "--jobs", "0"])).is_err());
        assert!(parse_cli(&vec_s(&["build", "a.exe", "--out", "w", "--shard-size"])).is_err());
        assert!(parse_cli(&vec_s(&["build", "a.exe", "--out", "w", "--shims"])).is_err());
        assert!(parse_cli(&vec_s(&["build", "a.exe", "--out", "w", "--entry", "zz"])).is_err());
        assert!(parse_cli(&vec_s(&["build", "a.exe", "--out", "w", "--bogus"])).is_err());
        assert!(parse_cli(&vec_s(&["build", "a.exe", "--out", "w", "--module", "a/b"])).is_err());
        match parse_cli(&vec_s(&[
            "build", "a.exe", "--out", "w", "--target", "native:x86_64-unknown-linux-gnu",
            "--target-cpu", "znver2",
        ])) {
            Ok(Command::Build(args)) => assert_eq!(
                args.codegen,
                LiftTarget::Native {
                    triple: "x86_64-unknown-linux-gnu".to_string(),
                    cpu: "znver2".to_string()
                }
            ),
            other => panic!("unexpected {other:?}"),
        }
        assert!(parse_cli(&vec_s(&["build", "a.exe", "--out", "w", "--target", "arm"])).is_err());
        assert!(parse_cli(&vec_s(&["build", "a.exe", "--out", "w", "--target-cpu", "x"])).is_err());
        assert!(parse_cli(&vec_s(&["build", "a.exe", "--out", "w", "--target"])).is_err());
    }

    #[test]
    fn build_end_to_end_prints_the_report() {
        let temp_dir = env::temp_dir().join(format!(
            "playgta6_recomp_build_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&temp_dir).expect("create temp dir");
        let exe_path = temp_dir.join("SyntheticGame.exe");
        fs::write(&exe_path, build_synthetic_pe64(false)).expect("write synthetic pe");
        let header = temp_dir.join("mini.h");
        fs::write(&header, "extern \"C\" void pg6_win32_virtual_alloc(void);\n").expect("header");

        let out = temp_dir.join("out");
        let report = run_build(&BuildArgs {
            target: InspectTarget::Path(exe_path),
            out_dir: out.clone(),
            module: None,
            shims: vec![header],
            entries: Vec::new(),
            emit_obj: false,
            jobs: Some(2),
            shard_size: None,
            shard_instructions: None,
            codegen: LiftTarget::Wasm64,
        })
        .expect("run_build should succeed");
        assert!(report.starts_with("playgta6-recomp build report: module main\n"), "{report}");
        assert!(report.contains("discovery: 4 functions"), "{report}");
        assert!(!report.contains("SyntheticGame"), "no file names");
        assert!(out.join("main").join(pipeline::MANIFEST_FILE).is_file());

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn stubs_end_to_end_writes_bridge() {
        let temp_dir = env::temp_dir().join(format!(
            "playgta6_recomp_stubs_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(temp_dir.join("rt").join("os")).expect("create temp dir");
        let exe_path = temp_dir.join("SyntheticGame.exe");
        fs::write(&exe_path, build_synthetic_pe64(false)).expect("write synthetic pe");
        let header = temp_dir.join("rt").join("os").join("mini.h");
        fs::write(
            &header,
            "extern \"C\" uint64_t pg6_win32_virtual_alloc(uint64_t a, uint64_t s, uint32_t t, uint32_t p);\n",
        )
        .expect("write header");

        let out = temp_dir.join("out");
        run_stubs(&StubsArgs {
            pe_path: exe_path,
            shims: vec![header],
            out_dir: out.clone(),
            include_root: Some(temp_dir.join("rt")),
        })
        .expect("run_stubs should succeed");

        let source = fs::read_to_string(out.join(stubs::SOURCE_FILE)).expect("read source");
        assert!(source.contains("#include \"os/mini.h\""));
        assert!(source.contains("pg6_win32_virtual_alloc("));
        let manifest = fs::read_to_string(out.join(stubs::MANIFEST_FILE)).expect("read manifest");
        assert!(manifest.contains("\"bound\": 1"));
        assert!(!manifest.contains("SyntheticGame"), "no paths");
        assert!(out.join(stubs::HEADER_FILE).is_file());
        assert!(out.join(stubs::SEAM_FILE).is_file());

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
