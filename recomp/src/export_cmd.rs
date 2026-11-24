//! `playgta6-recomp export-thunks`: guest-callable thunks for every shim export the runtime
//! registers (see `playgta6_recomp::stubs`, "Export thunks"). The output depends on the runtime
//! sources only (no game input), so it may go anywhere; `build-program` writes the same files to
//! `<out>/exports/` and references the table from `pg6_program.cpp`.

use std::env;
use std::path::PathBuf;

use playgta6_recomp::pipeline;
use playgta6_recomp::stubs;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportArgs {
    pub out_dir: PathBuf,
    pub shims: Vec<PathBuf>,
    pub include_root: Option<PathBuf>,
    /// `DLL!export` selectors; empty = every registered export.
    pub only: Vec<String>,
}

pub const USAGE: &str = "  playgta6-recomp export-thunks --out <dir> [--shims <header>...] [--include-root <dir>] [--only <DLL>!<export>]...";

pub const OPTIONS: &[&str] = &[
    "  --out <dir>            export-thunks: write pg6_export_thunks.{h,cpp}, the seam header and pg6_export_manifest.json:",
    "                         a marshaling thunk for every export the runtime registers (bindable under the static-import rules)",
    "  --only <DLL>!<name>    export-thunks: emit only this export (repeatable); for slim tables and test fixtures",
    "  --shims <header>...    export-thunks: runtime headers and export-table sources (default: this checkout's runtime/os/*.{h,cpp}, d3d12_xs, guest_callbacks.h)",
];

/// Parses the arguments after `export-thunks`. `Ok(None)` means help was requested.
pub fn parse_export_args(args: &[String]) -> Result<Option<ExportArgs>, String> {
    let mut out_dir = None;
    let mut shims = Vec::new();
    let mut include_root = None;
    let mut only = Vec::new();
    let dir_value = |flag: &str, value: Option<&String>| -> Result<PathBuf, String> {
        match value {
            Some(v) if !v.starts_with('-') => Ok(PathBuf::from(v)),
            _ => Err(format!("{flag} requires a <dir> path argument")),
        }
    };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => return Ok(None),
            "--out" => {
                i += 1;
                out_dir = Some(dir_value("--out", args.get(i))?);
            }
            "--include-root" => {
                i += 1;
                include_root = Some(dir_value("--include-root", args.get(i))?);
            }
            "--only" => {
                i += 1;
                match args.get(i) {
                    Some(v) if !v.starts_with('-') => only.push(v.clone()),
                    _ => return Err("--only requires a <DLL>!<name> argument".to_string()),
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
            other => return Err(format!("unknown argument '{other}' for 'export-thunks'")),
        }
        i += 1;
    }
    Ok(Some(ExportArgs {
        out_dir: out_dir.ok_or_else(|| "missing --out <dir>".to_string())?,
        shims,
        include_root,
        only,
    }))
}

/// Generates the export table; returns the coverage summary.
pub fn run_export_thunks(args: &ExportArgs) -> Result<String, String> {
    let (paths, default_root) = if args.shims.is_empty() {
        let cwd = env::current_dir()
            .map_err(|e| format!("failed to determine current directory: {e}"))?;
        let (headers, root) = pipeline::checkout_shim_headers(&cwd).ok_or(
            "missing --shims <header>... (no playgta6web checkout found to take runtime/os/*.h from)",
        )?;
        (headers, Some(root))
    } else {
        (args.shims.clone(), None)
    };
    let include_root = args.include_root.clone().or(default_root);
    let mut headers = Vec::with_capacity(paths.len());
    for header in &paths {
        headers.push(
            stubs::load_shim_header(header, include_root.as_deref())
                .map_err(|e| format!("failed to read shim header '{}': {e}", header.display()))?,
        );
    }
    let catalogue = stubs::Catalogue::scan(&headers);
    let mut plan = stubs::plan_exports(&catalogue);
    if !args.only.is_empty() {
        plan = plan.only(&args.only)?;
    }
    stubs::render_exports(&plan, &catalogue)
        .write_to(&args.out_dir)
        .map_err(|e| {
            format!(
                "failed to write export thunks to '{}': {e}",
                args.out_dir.display()
            )
        })?;
    Ok(format!(
        "{}Wrote export thunks to {}\n",
        stubs::render_export_coverage(&plan, &catalogue),
        args.out_dir.display()
    ))
}
