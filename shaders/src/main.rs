use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use playgta6_shaders::compat::render_compat_audit_report_md;
use playgta6_shaders::convert::{
    ConvertOptions, ConvertedShader, ShaderStage, audit_spirv_file,
    convert_spirv_bytes_with_options, format_shader_hash, format_wgsl_filename,
};
use playgta6_shaders::dxbc::{
    DxbcFileEntry, parse_dxbc, render_dxbc_batch_json, render_dxbc_batch_md,
    render_dxbc_info_json, render_dxbc_info_md,
};
use playgta6_shaders::dxil_reflect::{
    DxilReflectionErrorRecord, build_dxil_reflection_manifest, reflect_dxil_bytes_with_options,
    sanitize_source_path,
};
use playgta6_shaders::pack::{
    BlockerReportInput, FailedShaderRecord, PACK_ENTRY_BYTES, PACK_HEADER_BYTES, PackEntry,
    SHADER_INDEX_ABI_VERSION, ShaderIndexEntry, read_pack, render_blocker_report_md,
    render_shader_index_json, write_pack_with_layout,
};
use playgta6_shaders::reflect::{
    LAYOUT_MANIFEST_FILE_NAME, ReflectError, RegisterConvention, build_layout_manifest,
    parse_pipeline_list, reflect_spirv_bytes,
};

#[derive(Debug, PartialEq, Eq)]
enum Command {
    Help,
    Convert {
        spirv_dir: PathBuf,
        out_dir: PathBuf,
        pack_path: Option<PathBuf>,
        report_path: Option<PathBuf>,
        no_sanitize: bool,
        no_lower: bool,
    },
    Audit {
        spirv_dir: PathBuf,
        report_path: Option<PathBuf>,
        no_sanitize: bool,
    },
    InspectPack {
        pack_path: PathBuf,
    },
    InspectDxbc {
        target_path: PathBuf,
        json: bool,
    },
    Reflect {
        spirv_dir: PathBuf,
        out_dir: PathBuf,
        pipelines_path: Option<PathBuf>,
        convention: RegisterConvention,
        no_sanitize: bool,
        no_lower: bool,
    },
    ReflectDxil {
        target_path: PathBuf,
        out_path: Option<PathBuf>,
        json: bool,
    },
}

fn print_usage() {
    eprintln!("playgta6-shaders (ABI v{SHADER_INDEX_ABI_VERSION})");
    eprintln!("Usage:");
    eprintln!(
        "  playgta6-shaders convert --spirv <dir> --out <dir> [--pack <file.wgsl.pack>] [--report <blockers.md>] [--no-sanitize] [--no-lower]"
    );
    eprintln!(
        "  playgta6-shaders audit --spirv <dir> [--report <audit.md>] [--no-sanitize]"
    );
    eprintln!("  playgta6-shaders inspect-pack <file.wgsl.pack>");
    eprintln!("  playgta6-shaders inspect-dxbc <path-or-dir> [--json]");
    eprintln!(
        "  playgta6-shaders reflect --spirv <dir> --out <dir> [--pipelines <list.txt>] [--register-shift <b|t|u|s>=<n>]... [--no-sanitize] [--no-lower]"
    );
    eprintln!("  playgta6-shaders reflect-dxil <file-or-dir> [--out <json>] [--json]");
    eprintln!();
    eprintln!(
        "Pipeline: RDNA2 -> SPIR-V (--mesh-compute, --wave64-emulate) -> compat sanitize -> Naga lower -> WGSL -> .wgsl.pack"
    );
}

fn parse_args(args: &[String]) -> Result<Command, String> {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        return Ok(Command::Help);
    }

    match args[0].as_str() {
        "convert" => {
            let mut spirv_dir: Option<PathBuf> = None;
            let mut out_dir: Option<PathBuf> = None;
            let mut pack_path: Option<PathBuf> = None;
            let mut report_path: Option<PathBuf> = None;
            let mut no_sanitize = false;
            let mut no_lower = false;

            let mut i = 1;
            while i < args.len() {
                match args[i].as_str() {
                    "--spirv" => {
                        i += 1;
                        let val = args
                            .get(i)
                            .ok_or_else(|| "missing value for --spirv".to_string())?;
                        spirv_dir = Some(PathBuf::from(val));
                    }
                    "--out" => {
                        i += 1;
                        let val = args
                            .get(i)
                            .ok_or_else(|| "missing value for --out".to_string())?;
                        out_dir = Some(PathBuf::from(val));
                    }
                    "--pack" => {
                        i += 1;
                        let val = args
                            .get(i)
                            .ok_or_else(|| "missing value for --pack".to_string())?;
                        pack_path = Some(PathBuf::from(val));
                    }
                    "--report" => {
                        i += 1;
                        let val = args
                            .get(i)
                            .ok_or_else(|| "missing value for --report".to_string())?;
                        report_path = Some(PathBuf::from(val));
                    }
                    "--no-sanitize" => {
                        no_sanitize = true;
                    }
                    "--no-lower" => {
                        no_lower = true;
                    }
                    unknown => {
                        return Err(format!("unknown option for convert: {unknown}"));
                    }
                }
                i += 1;
            }

            let spirv_dir =
                spirv_dir.ok_or_else(|| "missing required option --spirv <dir>".to_string())?;
            let out_dir =
                out_dir.ok_or_else(|| "missing required option --out <dir>".to_string())?;

            Ok(Command::Convert {
                spirv_dir,
                out_dir,
                pack_path,
                report_path,
                no_sanitize,
                no_lower,
            })
        }
        "audit" => {
            let mut spirv_dir: Option<PathBuf> = None;
            let mut report_path: Option<PathBuf> = None;
            let mut no_sanitize = false;

            let mut i = 1;
            while i < args.len() {
                match args[i].as_str() {
                    "--spirv" => {
                        i += 1;
                        let val = args
                            .get(i)
                            .ok_or_else(|| "missing value for --spirv".to_string())?;
                        spirv_dir = Some(PathBuf::from(val));
                    }
                    "--report" => {
                        i += 1;
                        let val = args
                            .get(i)
                            .ok_or_else(|| "missing value for --report".to_string())?;
                        report_path = Some(PathBuf::from(val));
                    }
                    "--no-sanitize" => {
                        no_sanitize = true;
                    }
                    unknown => {
                        return Err(format!("unknown option for audit: {unknown}"));
                    }
                }
                i += 1;
            }

            let spirv_dir =
                spirv_dir.ok_or_else(|| "missing required option --spirv <dir>".to_string())?;

            Ok(Command::Audit {
                spirv_dir,
                report_path,
                no_sanitize,
            })
        }
        "inspect-pack" => {
            if args.len() != 2 {
                return Err(
                    "inspect-pack requires exactly one argument: <file.wgsl.pack>".to_string(),
                );
            }
            Ok(Command::InspectPack {
                pack_path: PathBuf::from(&args[1]),
            })
        }
        "inspect-dxbc" => {
            let mut target_path: Option<PathBuf> = None;
            let mut json = false;

            for arg in &args[1..] {
                match arg.as_str() {
                    "--json" => json = true,
                    other if other.starts_with('-') => {
                        return Err(format!("unknown option for inspect-dxbc: {other}"));
                    }
                    path_arg => {
                        if target_path.is_some() {
                            return Err(format!(
                                "unexpected extra argument for inspect-dxbc: {path_arg}"
                            ));
                        }
                        target_path = Some(PathBuf::from(path_arg));
                    }
                }
            }

            let target_path = target_path.ok_or_else(|| {
                "inspect-dxbc requires <path-or-dir> [--json]".to_string()
            })?;

            Ok(Command::InspectDxbc { target_path, json })
        }
        "reflect" => {
            let mut spirv_dir: Option<PathBuf> = None;
            let mut out_dir: Option<PathBuf> = None;
            let mut pipelines_path: Option<PathBuf> = None;
            let mut convention = RegisterConvention::default();
            let mut no_sanitize = false;
            let mut no_lower = false;

            let mut i = 1;
            while i < args.len() {
                let flag = args[i].as_str();
                match flag {
                    "--spirv" | "--out" | "--pipelines" | "--register-shift" => {
                        i += 1;
                        let val = args
                            .get(i)
                            .ok_or_else(|| format!("missing value for {flag}"))?;
                        match flag {
                            "--spirv" => spirv_dir = Some(PathBuf::from(val)),
                            "--out" => out_dir = Some(PathBuf::from(val)),
                            "--pipelines" => pipelines_path = Some(PathBuf::from(val)),
                            _ => convention.set_shift_arg(val)?,
                        }
                    }
                    "--no-sanitize" => no_sanitize = true,
                    "--no-lower" => no_lower = true,
                    unknown => {
                        return Err(format!("unknown option for reflect: {unknown}"));
                    }
                }
                i += 1;
            }

            Ok(Command::Reflect {
                spirv_dir: spirv_dir
                    .ok_or_else(|| "missing required option --spirv <dir>".to_string())?,
                out_dir: out_dir.ok_or_else(|| "missing required option --out <dir>".to_string())?,
                pipelines_path,
                convention,
                no_sanitize,
                no_lower,
            })
        }
        "reflect-dxil" => {
            let mut target_path: Option<PathBuf> = None;
            let mut out_path: Option<PathBuf> = None;
            let mut json = false;

            let mut i = 1;
            while i < args.len() {
                match args[i].as_str() {
                    "--out" => {
                        i += 1;
                        let val = args
                            .get(i)
                            .ok_or_else(|| "missing value for --out".to_string())?;
                        out_path = Some(PathBuf::from(val));
                    }
                    "--json" => json = true,
                    other if other.starts_with('-') => {
                        return Err(format!("unknown option for reflect-dxil: {other}"));
                    }
                    path_arg => {
                        if target_path.is_some() {
                            return Err(format!(
                                "unexpected extra argument for reflect-dxil: {path_arg}"
                            ));
                        }
                        target_path = Some(PathBuf::from(path_arg));
                    }
                }
                i += 1;
            }

            let target_path = target_path.ok_or_else(|| {
                "reflect-dxil requires <file-or-dir> [--out <json>] [--json]".to_string()
            })?;

            Ok(Command::ReflectDxil {
                target_path,
                out_path,
                json,
            })
        }
        other => Err(format!(
            "unknown subcommand '{other}' (expected 'convert', 'audit', 'inspect-pack', 'inspect-dxbc', 'reflect', or 'reflect-dxil')"
        )),
    }
}

fn collect_spv_files(root: &Path) -> Result<Vec<PathBuf>, String> {
    if !root.is_dir() {
        return Err(format!(
            "SPIR-V input directory does not exist or is not a directory: {}",
            root.display()
        ));
    }

    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir)
            .map_err(|err| format!("failed to read directory {}: {err}", dir.display()))?;
        for entry in entries {
            let entry = entry.map_err(|err| format!("failed to read directory entry: {err}"))?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("spv"))
            {
                files.push(path);
            }
        }
    }

    files.sort();
    Ok(files)
}

fn normalize_rel_path(path: &Path, base: &Path) -> String {
    let rel = path.strip_prefix(base).unwrap_or(path);
    rel.to_string_lossy().replace('\\', "/")
}

fn run_convert(
    spirv_dir: &Path,
    out_dir: &Path,
    pack_path: Option<&Path>,
    report_path: Option<&Path>,
    options: ConvertOptions,
) -> Result<(), String> {
    let spv_files = collect_spv_files(spirv_dir)?;
    fs::create_dir_all(out_dir).map_err(|err| {
        format!(
            "failed to create output directory {}: {err}",
            out_dir.display()
        )
    })?;

    let mut converted_by_hash: BTreeMap<u64, ConvertedShader> = BTreeMap::new();
    let mut converted_file_count = 0usize;
    let mut sanitized_file_count = 0usize;
    let mut rewritten_instruction_count = 0usize;
    let mut lowered_push_constant_count = 0usize;
    let mut lowered_bind_group_count = 0usize;
    let mut failures: Vec<FailedShaderRecord> = Vec::new();

    for path in &spv_files {
        let rel_path = normalize_rel_path(path, spirv_dir);
        let bytes = fs::read(path)
            .map_err(|err| format!("failed to read SPIR-V file {}: {err}", path.display()))?;

        match convert_spirv_bytes_with_options(&bytes, &options) {
            Ok(shader) => {
                converted_file_count += 1;
                if shader.audit.rewritten_instructions > 0 {
                    sanitized_file_count += 1;
                    rewritten_instruction_count += shader.audit.rewritten_instructions;
                }
                lowered_push_constant_count += shader.lowering.push_constant_remaps.len();
                lowered_bind_group_count += shader.lowering.bind_group_remaps.len();
                converted_by_hash.entry(shader.hash).or_insert(shader);
            }
            Err(error) => {
                failures.push(FailedShaderRecord {
                    source_path: rel_path,
                    error,
                });
            }
        }
    }

    let mut vertex_count = 0usize;
    let mut fragment_count = 0usize;
    let mut compute_count = 0usize;

    let mut pack_entries = Vec::with_capacity(converted_by_hash.len());
    for shader in converted_by_hash.values() {
        match shader.stage {
            ShaderStage::Vertex => vertex_count += 1,
            ShaderStage::Fragment => fragment_count += 1,
            ShaderStage::Compute => compute_count += 1,
        }

        let wgsl_file = format_wgsl_filename(shader.hash);
        let wgsl_dest = out_dir.join(&wgsl_file);
        fs::write(&wgsl_dest, shader.wgsl.as_bytes()).map_err(|err| {
            format!(
                "failed to write WGSL file {}: {err}",
                wgsl_dest.display()
            )
        })?;

        pack_entries.push(PackEntry {
            hash: shader.hash,
            stage: shader.stage,
            wgsl: shader.wgsl.clone(),
        });
    }

    let (pack_label, pack_ranges) = if let Some(pack_dest) = pack_path {
        if let Some(parent) = pack_dest.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "failed to create pack directory {}: {err}",
                    parent.display()
                )
            })?;
        }
        let written = write_pack_with_layout(&pack_entries, 0)
            .map_err(|err| format!("failed to build shader pack: {err}"))?;
        fs::write(pack_dest, &written.bytes).map_err(|err| {
            format!(
                "failed to write shader pack {}: {err}",
                pack_dest.display()
            )
        })?;
        let label = normalize_rel_path(pack_dest, out_dir);
        (Some(label), Some(written.ranges))
    } else {
        (None, None)
    };

    let mut index_entries = Vec::with_capacity(converted_by_hash.len());
    for (idx, shader) in converted_by_hash.values().enumerate() {
        let (pack_offset, pack_length) = match &pack_ranges {
            Some(ranges) => (Some(ranges[idx].0), Some(ranges[idx].1)),
            None => (None, None),
        };
        index_entries.push(ShaderIndexEntry {
            hash: shader.hash,
            stage: shader.stage,
            wgsl_path: format_wgsl_filename(shader.hash),
            entry_point: shader.entry_point.clone(),
            pack_offset,
            pack_length,
        });
    }

    let index_json = render_shader_index_json(pack_label.as_deref(), &index_entries);
    let index_dest = out_dir.join("shader-index.json");
    fs::write(&index_dest, index_json.as_bytes()).map_err(|err| {
        format!(
            "failed to write shader index {}: {err}",
            index_dest.display()
        )
    })?;

    let report_md = render_blocker_report_md(&BlockerReportInput {
        total_files: spv_files.len(),
        converted_count: converted_file_count,
        vertex_count,
        fragment_count,
        compute_count,
        failures,
    });

    if let Some(report_dest) = report_path {
        if let Some(parent) = report_dest.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "failed to create report directory {}: {err}",
                    parent.display()
                )
            })?;
        }
        fs::write(report_dest, report_md.as_bytes()).map_err(|err| {
            format!(
                "failed to write blocker report {}: {err}",
                report_dest.display()
            )
        })?;
    }

    println!(
        "Converted {converted_file_count}/{} SPIR-V files ({} unique shaders: vertex={vertex_count}, fragment={fragment_count}, compute={compute_count}; sanitized={sanitized_file_count} files / {rewritten_instruction_count} instructions; lowered={lowered_push_constant_count} push-constants / {lowered_bind_group_count} bind-groups); index: {}",
        spv_files.len(),
        converted_by_hash.len(),
        index_dest.display()
    );
    if let Some(pack_dest) = pack_path {
        println!("Wrote shader pack: {}", pack_dest.display());
    }
    if let Some(report_dest) = report_path {
        println!("Wrote blocker report: {}", report_dest.display());
    }

    Ok(())
}

fn run_audit(
    spirv_dir: &Path,
    report_path: Option<&Path>,
    options: ConvertOptions,
) -> Result<(), String> {
    let spv_files = collect_spv_files(spirv_dir)?;
    let mut records = Vec::with_capacity(spv_files.len());

    for path in &spv_files {
        let rel_path = normalize_rel_path(path, spirv_dir);
        let bytes = fs::read(path)
            .map_err(|err| format!("failed to read SPIR-V file {}: {err}", path.display()))?;
        records.push(audit_spirv_file(&rel_path, &bytes, &options));
    }

    let lowered_ok = records.iter().filter(|r| r.lowered_to_wgsl).count();
    let rewritten_files = records
        .iter()
        .filter(|r| r.audit.rewritten_instructions > 0)
        .count();
    let rewritten_insts: usize = records.iter().map(|r| r.audit.rewritten_instructions).sum();
    let blocker_files = records
        .iter()
        .filter(|r| r.audit.has_blockers() || r.stream_error.is_some() || !r.lowered_to_wgsl)
        .count();
    let warning_files = records
        .iter()
        .filter(|r| r.audit.warning_count() > 0)
        .count();

    let report_md = render_compat_audit_report_md(&records);
    if let Some(report_dest) = report_path {
        if let Some(parent) = report_dest.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "failed to create audit report directory {}: {err}",
                    parent.display()
                )
            })?;
        }
        fs::write(report_dest, report_md.as_bytes()).map_err(|err| {
            format!(
                "failed to write audit report {}: {err}",
                report_dest.display()
            )
        })?;
    }

    println!(
        "Audited {} SPIR-V files: lowered={lowered_ok}, rewritten={rewritten_files} ({rewritten_insts} instructions), blockers={blocker_files}, warnings={warning_files}",
        records.len()
    );
    if let Some(report_dest) = report_path {
        println!("Wrote compatibility audit report: {}", report_dest.display());
    }

    Ok(())
}

fn run_inspect_pack(pack_path: &Path) -> Result<(), String> {
    let bytes = fs::read(pack_path)
        .map_err(|err| format!("failed to read shader pack {}: {err}", pack_path.display()))?;
    let pack = read_pack(&bytes)
        .map_err(|err| format!("invalid shader pack {}: {err}", pack_path.display()))?;

    let payload_bytes: usize = pack.entries.iter().map(|e| e.length as usize).sum();
    let table_bytes = pack.entries.len() * PACK_ENTRY_BYTES;
    let mut vertex = 0usize;
    let mut fragment = 0usize;
    let mut compute = 0usize;
    for entry in &pack.entries {
        match entry.stage {
            ShaderStage::Vertex => vertex += 1,
            ShaderStage::Fragment => fragment += 1,
            ShaderStage::Compute => compute += 1,
        }
    }

    println!("Pack: {}", pack_path.display());
    println!("  Magic:   WGPK");
    println!("  Version: {}", pack.version);
    println!("  Flags:   0x{:08x}", pack.flags);
    println!(
        "  Entries: {} (vertex: {vertex}, fragment: {fragment}, compute: {compute})",
        pack.entries.len()
    );
    println!(
        "  Layout:  header={PACK_HEADER_BYTES}B, table={table_bytes}B, payload={payload_bytes}B, total={}B",
        bytes.len()
    );

    if !pack.entries.is_empty() {
        println!();
        println!("  {:<18}  {:<8}  {:>10}  {:>10}", "Hash", "Stage", "Offset", "Length");
        for entry in &pack.entries {
            println!(
                "  {:<18}  {:<8}  {:>10}  {:>10}",
                format_shader_hash(entry.hash),
                entry.stage.as_str(),
                entry.offset,
                entry.length
            );
        }
    }

    Ok(())
}

fn collect_dxbc_files(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir)
            .map_err(|err| format!("failed to read directory {}: {err}", dir.display()))?;
        for entry in entries {
            let entry = entry.map_err(|err| format!("failed to read directory entry: {err}"))?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|e| e.to_str()).is_some_and(|ext| {
                ext.eq_ignore_ascii_case("dxbc")
                    || ext.eq_ignore_ascii_case("dxil")
                    || ext.eq_ignore_ascii_case("cso")
            }) {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

fn format_inspect_dxbc(target_path: &Path, json: bool) -> Result<String, String> {
    if target_path.is_file() {
        let bytes = fs::read(target_path).map_err(|err| {
            format!(
                "failed to read DXBC/DXIL file {}: {err}",
                target_path.display()
            )
        })?;
        let info = parse_dxbc(&bytes).map_err(|err| {
            format!(
                "invalid DXBC/DXIL container {}: {err}",
                target_path.display()
            )
        })?;
        let display_path = target_path.to_string_lossy().replace('\\', "/");
        if json {
            Ok(render_dxbc_info_json(Some(&display_path), &info))
        } else {
            Ok(render_dxbc_info_md(Some(&display_path), &info))
        }
    } else if target_path.is_dir() {
        let files = collect_dxbc_files(target_path)?;
        let mut entries = Vec::with_capacity(files.len());
        for path in &files {
            let rel_path = normalize_rel_path(path, target_path);
            let result = match fs::read(path) {
                Ok(bytes) => parse_dxbc(&bytes).map_err(|err| err.to_string()),
                Err(err) => Err(format!("read error: {err}")),
            };
            entries.push(DxbcFileEntry {
                path: rel_path,
                result,
            });
        }
        if json {
            Ok(render_dxbc_batch_json(&entries))
        } else {
            Ok(render_dxbc_batch_md(&entries))
        }
    } else {
        Err(format!(
            "DXBC/DXIL path does not exist: {}",
            target_path.display()
        ))
    }
}

fn run_inspect_dxbc(target_path: &Path, json: bool) -> Result<(), String> {
    let rendered = format_inspect_dxbc(target_path, json)?;
    print!("{rendered}");
    Ok(())
}

fn run_reflect(
    spirv_dir: &Path,
    out_dir: &Path,
    pipelines_path: Option<&Path>,
    convention: &RegisterConvention,
    options: ConvertOptions,
) -> Result<(), String> {
    let spv_files = collect_spv_files(spirv_dir)?;
    let pipelines = match pipelines_path {
        Some(path) => {
            let text = fs::read_to_string(path).map_err(|err| {
                format!("failed to read pipeline list {}: {err}", path.display())
            })?;
            parse_pipeline_list(&text)
                .map_err(|err| format!("invalid pipeline list {}: {err}", path.display()))?
        }
        None => Vec::new(),
    };

    let mut shaders = Vec::with_capacity(spv_files.len());
    let mut failed = 0usize;
    for path in &spv_files {
        let bytes = fs::read(path)
            .map_err(|err| format!("failed to read SPIR-V file {}: {err}", path.display()))?;
        match reflect_spirv_bytes(&bytes, &options, convention) {
            Ok(shader) => shaders.push(shader),
            // Conversion failures are already reported by `convert --report`.
            Err(ReflectError::Conversion(_)) => failed += 1,
            Err(err) => return Err(format!("failed to reflect {}: {err}", path.display())),
        }
    }

    let manifest = build_layout_manifest(&shaders, &pipelines, convention)
        .map_err(|err| format!("failed to build layout manifest: {err}"))?;
    fs::create_dir_all(out_dir).map_err(|err| {
        format!(
            "failed to create output directory {}: {err}",
            out_dir.display()
        )
    })?;
    let manifest_dest = out_dir.join(LAYOUT_MANIFEST_FILE_NAME);
    fs::write(&manifest_dest, manifest.render_json().as_bytes()).map_err(|err| {
        format!(
            "failed to write layout manifest {}: {err}",
            manifest_dest.display()
        )
    })?;

    let incomplete = manifest
        .shaders
        .values()
        .filter(|record| !record.entry.layout_complete())
        .count();
    let over_limit = manifest
        .shaders
        .values()
        .filter(|record| !record.limit_violations.is_empty())
        .count();
    println!(
        "Reflected {}/{} SPIR-V files ({} unique shaders, {failed} not convertible; {} bind group layouts, {} pipeline layouts, {} pipelines; {incomplete} need lowering, {over_limit} exceed WebGPU limits); manifest: {}",
        spv_files.len() - failed,
        spv_files.len(),
        manifest.shaders.len(),
        manifest.bind_group_layouts.len(),
        manifest.pipeline_layouts.len(),
        manifest.pipelines.len(),
        manifest_dest.display()
    );
    Ok(())
}

fn build_reflect_dxil_json(target_path: &Path) -> Result<(String, usize, usize, usize, usize), String> {
    let convention = RegisterConvention::default();
    let mut reflections = Vec::new();
    let mut errors = Vec::new();

    if target_path.is_file() {
        let source = sanitize_source_path(target_path, None);
        let bytes = fs::read(target_path)
            .map_err(|err| format!("failed to read DXBC/DXIL file {source}: {err}"))?;
        let refl = reflect_dxil_bytes_with_options(&bytes, Some(&source), &convention)
            .map_err(|err| format!("failed to reflect DXBC/DXIL container {source}: {err}"))?;
        reflections.push(refl);
    } else if target_path.is_dir() {
        let files = collect_dxbc_files(target_path)?;
        for path in &files {
            let source = sanitize_source_path(path, Some(target_path));
            match fs::read(path) {
                Ok(bytes) => match reflect_dxil_bytes_with_options(&bytes, Some(&source), &convention) {
                    Ok(refl) => reflections.push(refl),
                    Err(err) => errors.push(DxilReflectionErrorRecord {
                        source,
                        error: err.to_string(),
                    }),
                },
                Err(err) => errors.push(DxilReflectionErrorRecord {
                    source,
                    error: format!("read error: {err}"),
                }),
            }
        }
    } else {
        return Err(format!(
            "DXBC/DXIL path does not exist: {}",
            sanitize_source_path(target_path, None)
        ));
    }

    let manifest = build_dxil_reflection_manifest(&reflections, &errors, &convention)
        .map_err(|err| format!("failed to build DXIL reflection manifest: {err}"))?;
    let json = manifest.render_json();
    Ok((
        json,
        manifest.shaders.len(),
        manifest.bind_group_layouts.len(),
        manifest.pipeline_layouts.len(),
        manifest.errors.len(),
    ))
}

fn run_reflect_dxil(target_path: &Path, out_path: Option<&Path>, json_flag: bool) -> Result<(), String> {
    let (json, shader_count, bgl_count, pl_count, err_count) = build_reflect_dxil_json(target_path)?;
    if let Some(dest) = out_path {
        if let Some(parent) = dest.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "failed to create output directory {}: {err}",
                    parent.display()
                )
            })?;
        }
        fs::write(dest, json.as_bytes()).map_err(|err| {
            format!(
                "failed to write DXIL reflection manifest {}: {err}",
                dest.display()
            )
        })?;
        if json_flag {
            print!("{json}");
        } else {
            println!(
                "Reflected {shader_count} DXBC/DXIL shader(s) ({bgl_count} bind group layouts, {pl_count} pipeline layouts, {err_count} errors); manifest: {}",
                dest.display()
            );
        }
    } else {
        print!("{json}");
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match parse_args(&args) {
        Ok(Command::Help) => {
            print_usage();
            ExitCode::SUCCESS
        }
        Ok(Command::Convert {
            spirv_dir,
            out_dir,
            pack_path,
            report_path,
            no_sanitize,
            no_lower,
        }) => {
            let mut options = if no_sanitize {
                ConvertOptions::without_sanitization()
            } else {
                ConvertOptions::default()
            };
            if no_lower {
                options = options.without_lowering();
            }
            match run_convert(
                &spirv_dir,
                &out_dir,
                pack_path.as_deref(),
                report_path.as_deref(),
                options,
            ) {
                Ok(()) => ExitCode::SUCCESS,
                Err(err) => {
                    eprintln!("error: {err}");
                    ExitCode::FAILURE
                }
            }
        }
        Ok(Command::Audit {
            spirv_dir,
            report_path,
            no_sanitize,
        }) => {
            let options = if no_sanitize {
                ConvertOptions::without_sanitization()
            } else {
                ConvertOptions::default()
            };
            match run_audit(&spirv_dir, report_path.as_deref(), options) {
                Ok(()) => ExitCode::SUCCESS,
                Err(err) => {
                    eprintln!("error: {err}");
                    ExitCode::FAILURE
                }
            }
        }
        Ok(Command::InspectPack { pack_path }) => match run_inspect_pack(&pack_path) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("error: {err}");
                ExitCode::FAILURE
            }
        },
        Ok(Command::InspectDxbc { target_path, json }) => {
            match run_inspect_dxbc(&target_path, json) {
                Ok(()) => ExitCode::SUCCESS,
                Err(err) => {
                    eprintln!("error: {err}");
                    ExitCode::FAILURE
                }
            }
        }
        Ok(Command::Reflect {
            spirv_dir,
            out_dir,
            pipelines_path,
            convention,
            no_sanitize,
            no_lower,
        }) => {
            let mut options = if no_sanitize {
                ConvertOptions::without_sanitization()
            } else {
                ConvertOptions::default()
            };
            if no_lower {
                options = options.without_lowering();
            }
            match run_reflect(
                &spirv_dir,
                &out_dir,
                pipelines_path.as_deref(),
                &convention,
                options,
            ) {
                Ok(()) => ExitCode::SUCCESS,
                Err(err) => {
                    eprintln!("error: {err}");
                    ExitCode::FAILURE
                }
            }
        }
        Ok(Command::ReflectDxil {
            target_path,
            out_path,
            json,
        }) => match run_reflect_dxil(&target_path, out_path.as_deref(), json) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("error: {err}");
                ExitCode::FAILURE
            }
        },
        Err(err) => {
            eprintln!("error: {err}");
            eprintln!();
            print_usage();
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn make_temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = env::temp_dir().join(format!("{prefix}_{}_{nanos}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn words_to_bytes(words: &[u32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for &w in words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        bytes
    }

    fn encode_spirv_string_words(s: &str) -> Vec<u32> {
        let mut raw = s.as_bytes().to_vec();
        raw.push(0);
        while !raw.len().is_multiple_of(4) {
            raw.push(0);
        }
        raw.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn build_float_controls_spv() -> Vec<u8> {
        let ext_words = encode_spirv_string_words("SPV_KHR_float_controls");
        let mut words: Vec<u32> = vec![
            0x0723_0203, // magic
            0x0001_0300, // SPIR-V 1.3
            0,           // generator
            5,           // bound
            0,           // schema
            (2 << 16) | 17, 1,    // OpCapability Shader
            (2 << 16) | 17, 4466, // OpCapability SignedZeroInfNanPreserve
        ];
        words.push((((1 + ext_words.len()) as u32) << 16) | 10);
        words.extend_from_slice(&ext_words);
        words.extend_from_slice(&[
            (3 << 16) | 14, 0, 1, // OpMemoryModel Logical GLSL450
            (5 << 16) | 15, 5, 3, u32::from_le_bytes(*b"main"), 0, // OpEntryPoint GLCompute %3 "main"
            (6 << 16) | 16, 3, 17, 1, 1, 1, // OpExecutionMode %3 LocalSize 1 1 1
            (4 << 16) | 16, 3, 4461, 32, // OpExecutionMode %3 SignedZeroInfNanPreserve 32
            (2 << 16) | 19, 1, // %1 = OpTypeVoid
            (3 << 16) | 33, 2, 1, // %2 = OpTypeFunction %1
            (5 << 16) | 54, 1, 3, 0, 2, // %3 = OpFunction %1 None %2
            (2 << 16) | 248, 4, // %4 = OpLabel
            (1 << 16) | 253, // OpReturn
            (1 << 16) | 56, // OpFunctionEnd
        ]);
        words_to_bytes(&words)
    }

    #[test]
    fn cli_parses_convert_audit_and_inspect_pack_arguments() {
        let cmd = parse_args(&[
            "convert".to_string(),
            "--spirv".to_string(),
            "in_spv".to_string(),
            "--out".to_string(),
            "out_wgsl".to_string(),
            "--pack".to_string(),
            "out_wgsl/all.wgsl.pack".to_string(),
            "--report".to_string(),
            "out_wgsl/blockers.md".to_string(),
            "--no-sanitize".to_string(),
            "--no-lower".to_string(),
        ])
        .unwrap();

        assert_eq!(
            cmd,
            Command::Convert {
                spirv_dir: PathBuf::from("in_spv"),
                out_dir: PathBuf::from("out_wgsl"),
                pack_path: Some(PathBuf::from("out_wgsl/all.wgsl.pack")),
                report_path: Some(PathBuf::from("out_wgsl/blockers.md")),
                no_sanitize: true,
                no_lower: true,
            }
        );

        let inspect_dxbc = parse_args(&[
            "inspect-dxbc".to_string(),
            "shaders_dir".to_string(),
            "--json".to_string(),
        ])
        .unwrap();
        assert_eq!(
            inspect_dxbc,
            Command::InspectDxbc {
                target_path: PathBuf::from("shaders_dir"),
                json: true,
            }
        );

        let inspect_dxbc_md =
            parse_args(&["inspect-dxbc".to_string(), "pixel.dxil".to_string()]).unwrap();
        assert_eq!(
            inspect_dxbc_md,
            Command::InspectDxbc {
                target_path: PathBuf::from("pixel.dxil"),
                json: false,
            }
        );
        assert!(parse_args(&["inspect-dxbc".to_string()]).is_err());

        let audit = parse_args(&[
            "audit".to_string(),
            "--spirv".to_string(),
            "in_spv".to_string(),
            "--report".to_string(),
            "out_wgsl/audit.md".to_string(),
        ])
        .unwrap();
        assert_eq!(
            audit,
            Command::Audit {
                spirv_dir: PathBuf::from("in_spv"),
                report_path: Some(PathBuf::from("out_wgsl/audit.md")),
                no_sanitize: false,
            }
        );

        let inspect = parse_args(&[
            "inspect-pack".to_string(),
            "out_wgsl/all.wgsl.pack".to_string(),
        ])
        .unwrap();
        assert_eq!(
            inspect,
            Command::InspectPack {
                pack_path: PathBuf::from("out_wgsl/all.wgsl.pack"),
            }
        );

        assert!(parse_args(&["convert".to_string(), "--spirv".to_string(), "x".to_string()]).is_err());
        assert!(parse_args(&["audit".to_string()]).is_err());
        assert!(parse_args(&["inspect-pack".to_string()]).is_err());
    }

    #[test]
    fn end_to_end_convert_pack_inspect_and_audit_temp_dir() {
        let root = make_temp_dir("playgta6_shaders_e2e");
        let spirv_dir = root.join("spirv");
        let out_dir = root.join("out");
        let out_no_san = root.join("out_no_san");
        let pack_file = out_dir.join("bundle.wgsl.pack");
        let report_file = out_dir.join("blockers.md");
        let no_san_report = out_no_san.join("blockers.md");
        let audit_report = root.join("audit.md");
        fs::create_dir_all(&spirv_dir).unwrap();

        // Minimal compute shader with SPV_KHR_float_controls + SignedZeroInfNanPreserve
        // plus one corrupt file.
        let float_ctrl_bytes = build_float_controls_spv();
        fs::write(spirv_dir.join("cs_float_ctrl.spv"), &float_ctrl_bytes).unwrap();
        fs::write(spirv_dir.join("broken.spv"), b"not-spirv").unwrap();

        // 1. Default convert sanitizes float controls -> succeeds for cs_float_ctrl.spv.
        run_convert(
            &spirv_dir,
            &out_dir,
            Some(&pack_file),
            Some(&report_file),
            ConvertOptions::default(),
        )
        .expect("batch conversion with sanitization should succeed");

        let pack_bytes = fs::read(&pack_file).unwrap();
        let decoded = read_pack(&pack_bytes).unwrap();
        assert_eq!(decoded.entries.len(), 1);
        assert_eq!(decoded.entries[0].stage, ShaderStage::Compute);
        assert!(decoded.entries[0].wgsl.contains("@compute"));

        let index_text = fs::read_to_string(out_dir.join("shader-index.json")).unwrap();
        assert!(index_text.contains("\"abi_version\": 1"));
        assert!(index_text.contains("\"pack\": \"bundle.wgsl.pack\""));

        let report_text = fs::read_to_string(&report_file).unwrap();
        assert!(report_text.contains("InvalidSpirvHeader"));
        assert!(report_text.contains("broken.spv"));

        // 2. --no-sanitize convert fails cs_float_ctrl.spv because SPV_KHR_float_controls is unstripped.
        run_convert(
            &spirv_dir,
            &out_no_san,
            None,
            Some(&no_san_report),
            ConvertOptions::without_sanitization(),
        )
        .expect("batch conversion without sanitization should finish and report failures");
        let no_san_text = fs::read_to_string(&no_san_report).unwrap();
        assert!(no_san_text.contains("cs_float_ctrl.spv"));

        // 3. audit subcommand writes Markdown compatibility report with rewrite counts.
        run_audit(&spirv_dir, Some(&audit_report), ConvertOptions::default())
            .expect("audit subcommand should succeed");
        let audit_text = fs::read_to_string(&audit_report).unwrap();
        assert!(audit_text.contains("# SPIR-V WebGPU Compatibility Audit Report"));
        assert!(audit_text.contains("FloatControlsStripped"));
        assert!(audit_text.contains("cs_float_ctrl.spv"));
        assert!(audit_text.contains("broken.spv"));

        run_inspect_pack(&pack_file).expect("inspect-pack should succeed");

        // 4. inspect-dxbc works on both a single file and a directory in Markdown and JSON modes.
        let dxbc_dir = root.join("dxbc");
        fs::create_dir_all(&dxbc_dir).unwrap();
        let dxbc_bytes = build_minimal_dxil_container();
        let dxil_file = dxbc_dir.join("cs_main.dxil");
        fs::write(&dxil_file, &dxbc_bytes).unwrap();
        fs::write(dxbc_dir.join("invalid.cso"), b"bad-dxbc").unwrap();

        let single_md = format_inspect_dxbc(&dxil_file, false).unwrap();
        assert!(single_md.contains("# DXBC / DXIL Container Summary"));
        assert!(single_md.contains("`compute`"));
        assert!(single_md.contains("| `DXIL` |"));

        let single_json = format_inspect_dxbc(&dxil_file, true).unwrap();
        assert!(single_json.contains("\"is_dxil\": true"));
        assert!(single_json.contains("\"has_llvm_bitcode\": true"));

        let dir_md = format_inspect_dxbc(&dxbc_dir, false).unwrap();
        assert!(dir_md.contains("| Total containers | 2 |"));
        assert!(dir_md.contains("`cs_main.dxil`"));
        assert!(dir_md.contains("`invalid.cso`"));

        let dir_json = format_inspect_dxbc(&dxbc_dir, true).unwrap();
        assert!(dir_json.contains("\"total\": 2"));
        assert!(dir_json.contains("\"error\":"));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn cli_parses_reflect_arguments() {
        let args: Vec<String> = [
            "reflect",
            "--spirv",
            "in_spv",
            "--out",
            "out_wgsl",
            "--pipelines",
            "pipelines.txt",
            "--register-shift",
            "t=1000",
            "--register-shift",
            "s=2000",
            "--no-lower",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let convention = RegisterConvention {
            shift_t: 1000,
            shift_s: 2000,
            ..RegisterConvention::default()
        };
        assert_eq!(
            parse_args(&args).unwrap(),
            Command::Reflect {
                spirv_dir: PathBuf::from("in_spv"),
                out_dir: PathBuf::from("out_wgsl"),
                pipelines_path: Some(PathBuf::from("pipelines.txt")),
                convention,
                no_sanitize: false,
                no_lower: true,
            }
        );
        assert!(parse_args(&["reflect".to_string(), "--spirv".to_string(), "x".to_string()]).is_err());
        assert!(parse_args(&args[..8]).unwrap_err().contains("--register-shift"));
        let mut bad_shift = args.clone();
        bad_shift[8] = "q=1".to_string();
        assert!(parse_args(&bad_shift).is_err());
    }

    /// Compute shader (`LocalSize 8 4 1`) storing into a `StorageBuffer` runtime array at
    /// `DescriptorSet 1` / `Binding 2`.
    fn build_storage_buffer_compute_spv() -> Vec<u8> {
        // %1 void, %2 fn, %3 u32, %4 u32[], %5 struct, %6/%7 StorageBuffer pointers, %8 buffer,
        // %9/%10 constants 0/7, %11 main, %12 label, %13 element pointer.
        let words: Vec<u32> = vec![
            0x0723_0203, 0x0001_0300, 0, 14, 0, // header (bound 14)
            (2 << 16) | 17, 1, // OpCapability Shader
            (3 << 16) | 14, 0, 1, // OpMemoryModel Logical GLSL450
            (5 << 16) | 15, 5, 11, u32::from_le_bytes(*b"main"), 0, // OpEntryPoint GLCompute %11 "main"
            (6 << 16) | 16, 11, 17, 8, 4, 1, // OpExecutionMode %11 LocalSize 8 4 1
            (4 << 16) | 71, 4, 6, 4, // OpDecorate %4 ArrayStride 4
            (5 << 16) | 72, 5, 0, 35, 0, // OpMemberDecorate %5 0 Offset 0
            (3 << 16) | 71, 5, 2, // OpDecorate %5 Block
            (4 << 16) | 71, 8, 34, 1, // OpDecorate %8 DescriptorSet 1
            (4 << 16) | 71, 8, 33, 2, // OpDecorate %8 Binding 2
            (2 << 16) | 19, 1, // %1 = OpTypeVoid
            (3 << 16) | 33, 2, 1, // %2 = OpTypeFunction %1
            (4 << 16) | 21, 3, 32, 0, // %3 = OpTypeInt 32 0
            (3 << 16) | 29, 4, 3, // %4 = OpTypeRuntimeArray %3
            (3 << 16) | 30, 5, 4, // %5 = OpTypeStruct %4
            (4 << 16) | 32, 6, 12, 5, // %6 = OpTypePointer StorageBuffer %5
            (4 << 16) | 32, 7, 12, 3, // %7 = OpTypePointer StorageBuffer %3
            (4 << 16) | 59, 6, 8, 12, // %8 = OpVariable %6 StorageBuffer
            (4 << 16) | 43, 3, 9, 0, // %9 = OpConstant %3 0
            (4 << 16) | 43, 3, 10, 7, // %10 = OpConstant %3 7
            (5 << 16) | 54, 1, 11, 0, 2, // %11 = OpFunction %1 None %2
            (2 << 16) | 248, 12, // %12 = OpLabel
            (6 << 16) | 65, 7, 13, 8, 9, 9, // %13 = OpAccessChain %7 %8 %9 %9
            (3 << 16) | 62, 13, 10, // OpStore %13 %10
            (1 << 16) | 253, // OpReturn
            (1 << 16) | 56, // OpFunctionEnd
        ];
        words_to_bytes(&words)
    }

    #[test]
    fn end_to_end_reflect_writes_layout_manifest_next_to_shader_index() {
        use playgta6_shaders::convert::fnv1a_64;

        let root = make_temp_dir("playgta6_shaders_reflect");
        let spirv_dir = root.join("spirv");
        let out_dir = root.join("out");
        let pipelines = root.join("pipelines.txt");
        fs::create_dir_all(spirv_dir.join("nested")).unwrap();

        let storage = build_storage_buffer_compute_spv();
        let storage_key = format_shader_hash(fnv1a_64(&storage));
        fs::write(spirv_dir.join("nested/cs_storage.spv"), &storage).unwrap();
        fs::write(spirv_dir.join("cs_float_ctrl.spv"), build_float_controls_spv()).unwrap();
        fs::write(spirv_dir.join("broken.spv"), b"not-spirv").unwrap();
        fs::write(&pipelines, format!("# compute pipelines\n{storage_key}\n")).unwrap();

        run_convert(&spirv_dir, &out_dir, None, None, ConvertOptions::default())
            .expect("convert should succeed");
        let mut convention = RegisterConvention::default();
        convention.set_shift_arg("u=1").unwrap();
        let reflect = |pipelines_path: &Path| {
            run_reflect(
                &spirv_dir,
                &out_dir,
                Some(pipelines_path),
                &convention,
                ConvertOptions::default(),
            )
        };
        reflect(&pipelines).expect("reflect should succeed");

        let index = fs::read_to_string(out_dir.join("shader-index.json")).unwrap();
        let manifest_path = out_dir.join(LAYOUT_MANIFEST_FILE_NAME);
        let manifest = fs::read_to_string(&manifest_path).unwrap();
        assert!(index.contains(&storage_key));
        assert!(manifest.starts_with("{\n  \"schema_version\": 1,\n"));
        assert!(manifest.contains(&format!("\"{storage_key}\": {{\n      \"stage\": \"compute\"")));
        assert!(manifest.contains(
            r#"{"binding":2,"visibility":4,"buffer":{"type":"storage","hasDynamicOffset":false,"minBindingSize":4}}"#
        ));
        // Binding 2 with `-fvk-u-shift 1` -> register u1, space 1.
        assert!(manifest.contains(
            r#""register":{"class":"u","index":1,"space":1},"spirv":{"set":1,"binding":2}"#
        ));
        assert!(manifest.contains("\"workgroup_size\": [8, 4, 1]"));
        assert!(manifest.contains("\"shifts\":{\"b\":0,\"t\":0,\"u\":1,\"s\":0}"));
        assert!(manifest.contains(&format!(
            "\"{storage_key}\": {{\n      \"stages\": {{\"compute\":\"{storage_key}\"}}"
        )));
        // Both convertible shaders are listed; the broken file is skipped.
        assert_eq!(manifest.matches("\"entry_point\": \"main\"").count(), 2);

        // Re-running is byte-identical.
        reflect(&pipelines).unwrap();
        assert_eq!(fs::read_to_string(&manifest_path).unwrap(), manifest);

        fs::write(&pipelines, "0x1234\n").unwrap();
        assert!(reflect(&pipelines).unwrap_err().contains("unknown shader"));

        let _ = fs::remove_dir_all(&root);
    }

    fn build_minimal_dxil_container() -> Vec<u8> {
        let mut dxil_payload = Vec::new();
        let program_version: u32 = (5u32 << 16) | (6u32 << 4) | 6u32; // cs_6_6
        dxil_payload.extend_from_slice(&program_version.to_le_bytes());
        dxil_payload.extend_from_slice(&8u32.to_le_bytes()); // size_in_uint32
        dxil_payload.extend_from_slice(b"DXIL");
        dxil_payload.extend_from_slice(&0x0000_0106u32.to_le_bytes()); // DXIL 1.6
        dxil_payload.extend_from_slice(&16u32.to_le_bytes()); // bitcode_offset
        dxil_payload.extend_from_slice(&8u32.to_le_bytes()); // bitcode_size
        dxil_payload.extend_from_slice(&[0x42, 0x43, 0xC0, 0xDE, 0, 0, 0, 0]);

        let chunk_offset = 36u32; // 32B header + 4B offset table
        let total_size = chunk_offset + 8 + dxil_payload.len() as u32;
        let mut out = Vec::new();
        out.extend_from_slice(b"DXBC");
        out.extend_from_slice(&[0x55u8; 16]); // digest
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&total_size.to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes()); // chunk_count
        out.extend_from_slice(&chunk_offset.to_le_bytes());
        out.extend_from_slice(b"DXIL");
        out.extend_from_slice(&(dxil_payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&dxil_payload);
        out
    }

    #[test]
    fn cli_parses_reflect_dxil_arguments() {
        let cmd = parse_args(&[
            "reflect-dxil".to_string(),
            "shaders_dir".to_string(),
            "--out".to_string(),
            "out/dxil-layouts.json".to_string(),
            "--json".to_string(),
        ])
        .unwrap();
        assert_eq!(
            cmd,
            Command::ReflectDxil {
                target_path: PathBuf::from("shaders_dir"),
                out_path: Some(PathBuf::from("out/dxil-layouts.json")),
                json: true,
            }
        );

        let cmd_min = parse_args(&["reflect-dxil".to_string(), "vs.dxil".to_string()]).unwrap();
        assert_eq!(
            cmd_min,
            Command::ReflectDxil {
                target_path: PathBuf::from("vs.dxil"),
                out_path: None,
                json: false,
            }
        );

        assert!(parse_args(&["reflect-dxil".to_string()]).is_err());
        assert!(parse_args(&["reflect-dxil".to_string(), "vs.dxil".to_string(), "--out".to_string()]).is_err());
        assert!(parse_args(&["reflect-dxil".to_string(), "vs.dxil".to_string(), "--bogus".to_string()]).is_err());
    }

    #[test]
    fn end_to_end_reflect_dxil_single_file_and_dir_emits_sanitized_s6_json() {
        let root = make_temp_dir("playgta6_shaders_reflect_dxil");
        let nested = root.join("sub");
        fs::create_dir_all(&nested).unwrap();

        let dxil_file = nested.join("cs_sample.dxil");
        fs::write(&dxil_file, build_minimal_dxil_container()).unwrap();
        fs::write(root.join("corrupt.cso"), b"bad-dxbc").unwrap();

        let out_file = root.join("out/manifest.json");
        run_reflect_dxil(&dxil_file, Some(&out_file), false).expect("single-file reflect-dxil should succeed");
        let single_json = fs::read_to_string(&out_file).unwrap();
        assert!(single_json.starts_with("{\n  \"schema_version\": 1,\n"));
        assert!(single_json.contains("\"name\": \"cs_sample.dxil\""));
        assert!(single_json.contains("\"stage\": \"compute\""));
        // Strict sanitization: zero host directory paths leaked into JSON!
        let root_str = root.to_string_lossy().replace('\\', "/");
        assert!(
            !single_json.contains(&root_str),
            "host path must never appear in sanitized reflection JSON"
        );

        let (dir_json, shaders, _, _, errors) =
            build_reflect_dxil_json(&root).expect("directory reflect-dxil should succeed");
        assert_eq!(shaders, 1);
        assert_eq!(errors, 1);
        assert!(dir_json.contains("\"name\": \"sub/cs_sample.dxil\""));
        assert!(dir_json.contains("\"source\":\"corrupt.cso\""));
        assert!(
            !dir_json.contains(&root_str),
            "host path must never appear in directory reflection JSON"
        );

        let _ = fs::remove_dir_all(&root);
    }
}
