use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
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
    BlockerReportInput, FailedShaderRecord, PACK_HEADER_BYTES, PackEntry, PackVersion,
    SHADER_INDEX_ABI_VERSION, ShaderIndexEntry, checked_table_data_start_for_version,
    pack_entry_bytes, parse_pack_header, read_pack_toc, render_blocker_report_md,
    render_shader_index_json, write_pack_with_layout_versioned,
};
use playgta6_shaders::rdna2::{
    analyze_instructions, decode_program, format_disasm_with_summary, to_json_with_disasm,
};
use playgta6_shaders::rdna2_pack::{
    BudgetOrder, Rdna2PackOptions, Rdna2PackResult, load_priority_hashes_file,
    pack_rdna2_sources_to_file,
};
use playgta6_shaders::rdna2_to_wgsl::{
    Rdna2StageOverride, Rdna2TranslateOptions, parse_fssm_stage_metadata, translate_rdna2_to_wgsl,
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
        pack_version: PackVersion,
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
    DisasmRdna2 {
        file_path: PathBuf,
        json: bool,
        max_instructions: Option<usize>,
    },
    TranslateRdna2 {
        file_path: PathBuf,
        out_path: Option<PathBuf>,
        stage: Rdna2StageOverride,
        json: bool,
    },
    PackRdna2 {
        input_dir: Option<PathBuf>,
        manifest_path: Option<PathBuf>,
        out_path: PathBuf,
        pack_version: PackVersion,
        best_effort: bool,
        max_wgsl_bytes: Option<usize>,
        max_pack_bytes: Option<u64>,
        budget_order: BudgetOrder,
        priority_hashes_path: Option<PathBuf>,
        jobs: Option<usize>,
        json: bool,
    },
}

fn print_usage() {
    eprintln!("playgta6-shaders (ABI v{SHADER_INDEX_ABI_VERSION})");
    eprintln!("Usage:");
    eprintln!(
        "  playgta6-shaders convert --spirv <dir> --out <dir> [--pack <file.wgsl.pack>] [--pack-version auto|1|2] [--report <blockers.md>] [--no-sanitize] [--no-lower]"
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
    eprintln!("  playgta6-shaders disasm-rdna2 <file> [--json] [--max-instructions <N>]");
    eprintln!(
        "  playgta6-shaders translate-rdna2 <file> [--out <file.wgsl>] [--stage auto|vertex|pixel|compute|mesh] [--json]"
    );
    eprintln!(
        "  playgta6-shaders pack-rdna2 [<input-dir>] [--manifest <manifest.json>] --out <out.wgsl.pack> [--pack-version auto|1|2] [--best-effort] [--max-wgsl-bytes <N>] [--max-pack-bytes <N>] [--budget-order hash-order|smallest-first|manifest-order] [--priority-hashes <file>] [--jobs <N>] [--json]"
    );
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
            let mut pack_version = PackVersion::Auto;
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
                    "--pack-version" => {
                        i += 1;
                        let val = args
                            .get(i)
                            .ok_or_else(|| "missing value for --pack-version".to_string())?;
                        pack_version = PackVersion::parse_str(val)?;
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
                pack_version,
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
        "disasm-rdna2" => {
            let mut file_path: Option<PathBuf> = None;
            let mut json = false;
            let mut max_instructions: Option<usize> = None;

            let mut i = 1;
            while i < args.len() {
                match args[i].as_str() {
                    "--json" => json = true,
                    "--max-instructions" => {
                        i += 1;
                        let val = args
                            .get(i)
                            .ok_or_else(|| "missing value for --max-instructions".to_string())?;
                        let parsed = val.parse::<usize>().map_err(|_| {
                            format!("invalid integer for --max-instructions: {val}")
                        })?;
                        max_instructions = Some(parsed);
                    }
                    other if other.starts_with('-') => {
                        return Err(format!("unknown option for disasm-rdna2: {other}"));
                    }
                    path_arg => {
                        if file_path.is_some() {
                            return Err(format!(
                                "unexpected extra argument for disasm-rdna2: {path_arg}"
                            ));
                        }
                        file_path = Some(PathBuf::from(path_arg));
                    }
                }
                i += 1;
            }

            let file_path = file_path.ok_or_else(|| {
                "disasm-rdna2 requires <file> [--json] [--max-instructions <N>]".to_string()
            })?;

            Ok(Command::DisasmRdna2 {
                file_path,
                json,
                max_instructions,
            })
        }
        "translate-rdna2" => {
            let mut file_path: Option<PathBuf> = None;
            let mut out_path: Option<PathBuf> = None;
            let mut stage = Rdna2StageOverride::Auto;
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
                    "--stage" => {
                        i += 1;
                        let val = args
                            .get(i)
                            .ok_or_else(|| "missing value for --stage".to_string())?;
                        stage = Rdna2StageOverride::parse_str(val)?;
                    }
                    "--json" => json = true,
                    other if other.starts_with('-') => {
                        return Err(format!("unknown option for translate-rdna2: {other}"));
                    }
                    path_arg => {
                        if file_path.is_some() {
                            return Err(format!(
                                "unexpected extra argument for translate-rdna2: {path_arg}"
                            ));
                        }
                        file_path = Some(PathBuf::from(path_arg));
                    }
                }
                i += 1;
            }

            let file_path = file_path.ok_or_else(|| {
                "translate-rdna2 requires <file> [--out <file.wgsl>] [--stage auto|vertex|pixel|compute|mesh] [--json]".to_string()
            })?;

            Ok(Command::TranslateRdna2 {
                file_path,
                out_path,
                stage,
                json,
            })
        }
        "pack-rdna2" => {
            let mut input_dir: Option<PathBuf> = None;
            let mut manifest_path: Option<PathBuf> = None;
            let mut out_path: Option<PathBuf> = None;
            let mut pack_version = PackVersion::Auto;
            let mut best_effort = false;
            let mut max_wgsl_bytes: Option<usize> = None;
            let mut max_pack_bytes: Option<u64> = None;
            let mut budget_order = BudgetOrder::HashOrder;
            let mut priority_hashes_path: Option<PathBuf> = None;
            let mut jobs: Option<usize> = None;
            let mut json = false;

            let mut i = 1;
            while i < args.len() {
                match args[i].as_str() {
                    "--manifest" => {
                        i += 1;
                        let val = args
                            .get(i)
                            .ok_or_else(|| "missing value for --manifest".to_string())?;
                        if manifest_path.is_some() {
                            return Err("duplicate option --manifest for pack-rdna2".to_string());
                        }
                        manifest_path = Some(PathBuf::from(val));
                    }
                    "--out" => {
                        i += 1;
                        let val = args
                            .get(i)
                            .ok_or_else(|| "missing value for --out".to_string())?;
                        out_path = Some(PathBuf::from(val));
                    }
                    "--pack-version" => {
                        i += 1;
                        let val = args
                            .get(i)
                            .ok_or_else(|| "missing value for --pack-version".to_string())?;
                        pack_version = PackVersion::parse_str(val)?;
                    }
                    "--best-effort" => best_effort = true,
                    "--max-wgsl-bytes" => {
                        i += 1;
                        let val = args
                            .get(i)
                            .ok_or_else(|| "missing value for --max-wgsl-bytes".to_string())?;
                        let parsed = val.parse::<usize>().map_err(|_| {
                            format!("invalid integer for --max-wgsl-bytes: {val}")
                        })?;
                        max_wgsl_bytes = Some(parsed);
                    }
                    "--max-pack-bytes" => {
                        i += 1;
                        let val = args
                            .get(i)
                            .ok_or_else(|| "missing value for --max-pack-bytes".to_string())?;
                        let parsed = val.parse::<u64>().map_err(|_| {
                            format!("invalid integer for --max-pack-bytes: {val}")
                        })?;
                        max_pack_bytes = Some(parsed);
                    }
                    "--budget-order" => {
                        i += 1;
                        let val = args
                            .get(i)
                            .ok_or_else(|| "missing value for --budget-order".to_string())?;
                        budget_order = BudgetOrder::parse_str(val)?;
                    }
                    "--priority-hashes" => {
                        i += 1;
                        let val = args
                            .get(i)
                            .ok_or_else(|| "missing value for --priority-hashes".to_string())?;
                        if priority_hashes_path.is_some() {
                            return Err(
                                "duplicate option --priority-hashes for pack-rdna2".to_string(),
                            );
                        }
                        priority_hashes_path = Some(PathBuf::from(val));
                    }
                    flag @ ("--jobs" | "-j") => {
                        i += 1;
                        let val = args
                            .get(i)
                            .ok_or_else(|| format!("missing value for {flag}"))?;
                        let parsed = val
                            .parse::<usize>()
                            .map_err(|_| format!("invalid integer for {flag}: {val}"))?;
                        if parsed == 0 {
                            return Err(format!("{flag} must be >= 1 (got 0)"));
                        }
                        jobs = Some(parsed);
                    }
                    "--json" => json = true,
                    other if other.starts_with('-') => {
                        return Err(format!("unknown option for pack-rdna2: {other}"));
                    }
                    dir_arg => {
                        if input_dir.is_some() {
                            return Err(format!(
                                "unexpected extra argument for pack-rdna2: {dir_arg}"
                            ));
                        }
                        input_dir = Some(PathBuf::from(dir_arg));
                    }
                }
                i += 1;
            }

            if input_dir.is_none() && manifest_path.is_none() {
                return Err(
                    "pack-rdna2 requires <input-dir> and/or --manifest <manifest.json> --out <out.wgsl.pack> [--pack-version auto|1|2] [--best-effort] [--max-wgsl-bytes <N>] [--max-pack-bytes <N>] [--budget-order hash-order|smallest-first|manifest-order] [--priority-hashes <file>] [--jobs <N>] [--json]"
                        .to_string(),
                );
            }
            let out_path = out_path.ok_or_else(|| {
                "pack-rdna2 requires --out <out.wgsl.pack>".to_string()
            })?;

            Ok(Command::PackRdna2 {
                input_dir,
                manifest_path,
                out_path,
                pack_version,
                best_effort,
                max_wgsl_bytes,
                max_pack_bytes,
                budget_order,
                priority_hashes_path,
                jobs,
                json,
            })
        }
        other => Err(format!(
            "unknown subcommand '{other}' (expected 'convert', 'audit', 'inspect-pack', 'inspect-dxbc', 'reflect', 'reflect-dxil', 'disasm-rdna2', 'translate-rdna2', or 'pack-rdna2')"
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
    pack_version: PackVersion,
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
        let written = write_pack_with_layout_versioned(&pack_entries, 0, pack_version)
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
    let mut file = File::open(pack_path)
        .map_err(|err| format!("failed to read shader pack {}: {err}", pack_path.display()))?;
    let total_file_size = file
        .metadata()
        .map_err(|err| format!("failed to stat shader pack {}: {err}", pack_path.display()))?
        .len();

    let mut header_buf = [0u8; PACK_HEADER_BYTES];
    let header_read = file
        .read(&mut header_buf)
        .map_err(|err| format!("failed to read shader pack {}: {err}", pack_path.display()))?;
    let (version, entry_count, _flags) = parse_pack_header(&header_buf[..header_read])
        .map_err(|err| format!("invalid shader pack {}: {err}", pack_path.display()))?;
    let data_start = checked_table_data_start_for_version(version, entry_count as usize)
        .map_err(|err| format!("invalid shader pack {}: {err}", pack_path.display()))?;
    let toc_len = usize::try_from(data_start.min(total_file_size))
        .map_err(|_| format!("invalid shader pack {}: TOC exceeds platform usize", pack_path.display()))?;

    file.seek(SeekFrom::Start(0))
        .map_err(|err| format!("failed to seek shader pack {}: {err}", pack_path.display()))?;
    let mut toc_buf = vec![0u8; toc_len];
    file.read_exact(&mut toc_buf)
        .map_err(|err| format!("failed to read shader pack TOC {}: {err}", pack_path.display()))?;

    let toc = read_pack_toc(&toc_buf, total_file_size)
        .map_err(|err| format!("invalid shader pack {}: {err}", pack_path.display()))?;

    // Validate UTF-8 payload bytes per entry without loading >4 GiB archives wholesale into RAM.
    let mut scratch = Vec::new();
    for (idx, entry) in toc.entries.iter().enumerate() {
        let len = entry.length as usize;
        if len == 0 {
            continue;
        }
        scratch.resize(len, 0);
        file.seek(SeekFrom::Start(entry.offset))
            .map_err(|err| format!("failed to seek shader pack {}: {err}", pack_path.display()))?;
        file.read_exact(&mut scratch)
            .map_err(|err| format!("failed to read shader pack {}: {err}", pack_path.display()))?;
        if let Err(err) = std::str::from_utf8(&scratch) {
            return Err(format!(
                "invalid shader pack {}: entry {idx} payload is not valid UTF-8: {err}",
                pack_path.display()
            ));
        }
    }

    let entry_bytes = pack_entry_bytes(toc.version).expect("validated version");
    let payload_bytes: u64 = toc.entries.iter().map(|e| u64::from(e.length)).sum();
    let table_bytes = (toc.entries.len() as u64) * (entry_bytes as u64);
    let mut vertex = 0usize;
    let mut fragment = 0usize;
    let mut compute = 0usize;
    for entry in &toc.entries {
        match entry.stage {
            ShaderStage::Vertex => vertex += 1,
            ShaderStage::Fragment => fragment += 1,
            ShaderStage::Compute => compute += 1,
        }
    }

    println!("Pack: {}", pack_path.display());
    println!("  Magic:   WGPK");
    println!("  Version: {}", toc.version);
    println!("  Flags:   0x{:08x}", toc.flags);
    println!(
        "  Entries: {} (vertex: {vertex}, fragment: {fragment}, compute: {compute})",
        toc.entries.len()
    );
    println!(
        "  Layout:  header={PACK_HEADER_BYTES}B, table={table_bytes}B, payload={payload_bytes}B, total={total_file_size}B"
    );

    if !toc.entries.is_empty() {
        println!();
        println!("  {:<18}  {:<8}  {:>10}  {:>10}", "Hash", "Stage", "Offset", "Length");
        for entry in &toc.entries {
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

fn format_disasm_rdna2(
    file_path: &Path,
    json: bool,
    max_instructions: Option<usize>,
) -> Result<String, String> {
    let source = sanitize_source_path(file_path, None);
    let bytes = fs::read(file_path)
        .map_err(|err| format!("failed to read RDNA2 shader file {source}: {err}"))?;
    let instructions = decode_program(&bytes);
    let summary = analyze_instructions(&bytes, &instructions);
    if json {
        Ok(to_json_with_disasm(&summary, &instructions, max_instructions))
    } else {
        Ok(format_disasm_with_summary(
            &summary,
            &instructions,
            max_instructions,
        ))
    }
}

fn run_disasm_rdna2(
    file_path: &Path,
    json: bool,
    max_instructions: Option<usize>,
) -> Result<(), String> {
    let rendered = format_disasm_rdna2(file_path, json, max_instructions)?;
    print!("{rendered}");
    Ok(())
}

fn format_translate_rdna2(
    file_path: &Path,
    stage: Rdna2StageOverride,
    json: bool,
) -> Result<(String, playgta6_shaders::rdna2_to_wgsl::Rdna2WgslOutput, String), String> {
    let source = sanitize_source_path(file_path, None);
    let bytes = fs::read(file_path)
        .map_err(|err| format!("failed to read RDNA2 shader file {source}: {err}"))?;
    let fssm_meta = {
        let meta_path = file_path.with_extension("meta");
        if meta_path != file_path && fs::metadata(&meta_path).is_ok_and(|m| m.is_file()) {
            fs::read(&meta_path)
                .ok()
                .and_then(|meta_bytes| parse_fssm_stage_metadata(&meta_bytes))
        } else {
            None
        }
    };
    let options = Rdna2TranslateOptions {
        stage,
        fssm_meta,
        ..Rdna2TranslateOptions::default()
    };
    let output = translate_rdna2_to_wgsl(&bytes, &options)
        .map_err(|err| format!("failed to translate RDNA2 shader {source}: {err}"))?;
    let rendered = if json {
        output.to_json_with_source(Some(&source))
    } else {
        output.wgsl.clone()
    };
    Ok((rendered, output, source))
}

fn run_translate_rdna2(
    file_path: &Path,
    out_path: Option<&Path>,
    stage: Rdna2StageOverride,
    json: bool,
) -> Result<(), String> {
    let (rendered, output, source) = format_translate_rdna2(file_path, stage, json)?;
    if let Some(dest) = out_path {
        let dest_label = sanitize_source_path(dest, None);
        if let Some(parent) = dest.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).map_err(|err| {
                format!("failed to create output directory for {dest_label}: {err}")
            })?;
        }
        fs::write(dest, output.wgsl.as_bytes())
            .map_err(|err| format!("failed to write WGSL file {dest_label}: {err}"))?;
        if json {
            print!("{rendered}");
        } else {
            println!(
                "Translated {source} -> {dest_label} (stage={}, hash={}, translated_ops={}, unsupported_ops={}, naga_validated={})",
                output.stage.as_str(),
                format_shader_hash(output.hash_fnv1a64),
                output.translated_op_count,
                output.unsupported_op_count,
                output.naga_validated,
            );
        }
    } else {
        print!("{rendered}");
    }
    Ok(())
}

fn format_and_write_pack_rdna2_sources(
    input_dir: Option<&Path>,
    manifest_path: Option<&Path>,
    out_path: &Path,
    options: &Rdna2PackOptions,
    json: bool,
) -> Result<(String, Rdna2PackResult), String> {
    let result = pack_rdna2_sources_to_file(input_dir, manifest_path, out_path, options)
        .map_err(|err| err.to_string())?;
    let rendered = if json {
        result.summary.to_json()
    } else {
        result.summary.to_text()
    };
    Ok((rendered, result))
}

#[cfg(test)]
fn format_and_write_pack_rdna2(
    input_dir: &Path,
    out_path: &Path,
    json: bool,
) -> Result<(String, Rdna2PackResult), String> {
    format_and_write_pack_rdna2_sources(
        Some(input_dir),
        None,
        out_path,
        &Rdna2PackOptions::default(),
        json,
    )
}

fn load_cli_priority_hashes(path: Option<&Path>) -> Result<Vec<u64>, String> {
    match path {
        Some(p) => load_priority_hashes_file(p).map_err(|err| err.to_string()),
        None => Ok(Vec::new()),
    }
}

fn run_pack_rdna2(
    input_dir: Option<&Path>,
    manifest_path: Option<&Path>,
    out_path: &Path,
    options: &Rdna2PackOptions,
    json: bool,
) -> Result<(), String> {
    let (rendered, _) =
        format_and_write_pack_rdna2_sources(input_dir, manifest_path, out_path, options, json)?;
    print!("{rendered}");
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
            pack_version,
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
                pack_version,
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
        Ok(Command::DisasmRdna2 {
            file_path,
            json,
            max_instructions,
        }) => match run_disasm_rdna2(&file_path, json, max_instructions) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("error: {err}");
                ExitCode::FAILURE
            }
        },
        Ok(Command::TranslateRdna2 {
            file_path,
            out_path,
            stage,
            json,
        }) => match run_translate_rdna2(&file_path, out_path.as_deref(), stage, json) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("error: {err}");
                ExitCode::FAILURE
            }
        },
        Ok(Command::PackRdna2 {
            input_dir,
            manifest_path,
            out_path,
            pack_version,
            best_effort,
            max_wgsl_bytes,
            max_pack_bytes,
            budget_order,
            priority_hashes_path,
            jobs,
            json,
        }) => {
            let priority_hashes = match load_cli_priority_hashes(priority_hashes_path.as_deref()) {
                Ok(hashes) => hashes,
                Err(err) => {
                    eprintln!("error: {err}");
                    return ExitCode::FAILURE;
                }
            };
            let options = Rdna2PackOptions {
                pack_version,
                best_effort,
                max_wgsl_bytes,
                max_pack_bytes,
                budget_order,
                priority_hashes,
                jobs,
                ..Rdna2PackOptions::default()
            };
            match run_pack_rdna2(
                input_dir.as_deref(),
                manifest_path.as_deref(),
                &out_path,
                &options,
                json,
            ) {
                Ok(()) => ExitCode::SUCCESS,
                Err(err) => {
                    eprintln!("error: {err}");
                    ExitCode::FAILURE
                }
            }
        }
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
    use playgta6_shaders::pack::{
        PACK_ENTRY_BYTES, PACK_ENTRY_BYTES_V2, PACK_VERSION_V1, PACK_VERSION_V2, read_pack,
    };
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
                pack_version: PackVersion::Auto,
                report_path: Some(PathBuf::from("out_wgsl/blockers.md")),
                no_sanitize: true,
                no_lower: true,
            }
        );

        let cmd_v2 = parse_args(&[
            "convert".to_string(),
            "--spirv".to_string(),
            "in_spv".to_string(),
            "--out".to_string(),
            "out_wgsl".to_string(),
            "--pack".to_string(),
            "out_wgsl/v2.wgsl.pack".to_string(),
            "--pack-version".to_string(),
            "2".to_string(),
        ])
        .unwrap();
        assert_eq!(
            cmd_v2,
            Command::Convert {
                spirv_dir: PathBuf::from("in_spv"),
                out_dir: PathBuf::from("out_wgsl"),
                pack_path: Some(PathBuf::from("out_wgsl/v2.wgsl.pack")),
                pack_version: PackVersion::V2,
                report_path: None,
                no_sanitize: false,
                no_lower: false,
            }
        );
        assert!(
            parse_args(&[
                "convert".to_string(),
                "--spirv".to_string(),
                "in_spv".to_string(),
                "--out".to_string(),
                "out_wgsl".to_string(),
                "--pack-version".to_string(),
                "3".to_string(),
            ])
            .is_err()
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
            PackVersion::Auto,
            Some(&report_file),
            ConvertOptions::default(),
        )
        .expect("batch conversion with sanitization should succeed");

        let pack_bytes = fs::read(&pack_file).unwrap();
        let decoded = read_pack(&pack_bytes).unwrap();
        assert_eq!(decoded.version, PACK_VERSION_V1);
        assert_eq!(decoded.entries.len(), 1);
        assert_eq!(decoded.entries[0].stage, ShaderStage::Compute);
        assert!(decoded.entries[0].wgsl.contains("@compute"));
        run_inspect_pack(&pack_file).expect("inspect-pack should accept v1 pack");

        // 1b. Explicit --pack-version 2 emits WGPK v2 and passes inspect-pack.
        let pack_file_v2 = out_dir.join("bundle_v2.wgsl.pack");
        run_convert(
            &spirv_dir,
            &out_dir,
            Some(&pack_file_v2),
            PackVersion::V2,
            None,
            ConvertOptions::default(),
        )
        .expect("batch conversion with WGPK v2 should succeed");
        let pack_bytes_v2 = fs::read(&pack_file_v2).unwrap();
        let decoded_v2 = read_pack(&pack_bytes_v2).unwrap();
        assert_eq!(decoded_v2.version, PACK_VERSION_V2);
        assert_eq!(pack_bytes_v2.len() - pack_bytes.len(), 4);
        assert_eq!(decoded_v2.entries[0].wgsl, decoded.entries[0].wgsl);
        run_inspect_pack(&pack_file_v2).expect("inspect-pack should accept v2 pack");

        let index_text = fs::read_to_string(out_dir.join("shader-index.json")).unwrap();
        assert!(index_text.contains("\"abi_version\": 1"));
        assert!(index_text.contains("\"pack\": \"bundle_v2.wgsl.pack\""));

        let report_text = fs::read_to_string(&report_file).unwrap();
        assert!(report_text.contains("InvalidSpirvHeader"));
        assert!(report_text.contains("broken.spv"));

        // 2. --no-sanitize convert fails cs_float_ctrl.spv because SPV_KHR_float_controls is unstripped.
        run_convert(
            &spirv_dir,
            &out_no_san,
            None,
            PackVersion::Auto,
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

        run_convert(
            &spirv_dir,
            &out_dir,
            None,
            PackVersion::Auto,
            None,
            ConvertOptions::default(),
        )
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

    #[test]
    fn cli_parses_disasm_rdna2_arguments() {
        let cmd = parse_args(&[
            "disasm-rdna2".to_string(),
            "splash_ps.bin".to_string(),
            "--json".to_string(),
            "--max-instructions".to_string(),
            "64".to_string(),
        ])
        .unwrap();
        assert_eq!(
            cmd,
            Command::DisasmRdna2 {
                file_path: PathBuf::from("splash_ps.bin"),
                json: true,
                max_instructions: Some(64),
            }
        );

        let cmd_min = parse_args(&["disasm-rdna2".to_string(), "vs.bin".to_string()]).unwrap();
        assert_eq!(
            cmd_min,
            Command::DisasmRdna2 {
                file_path: PathBuf::from("vs.bin"),
                json: false,
                max_instructions: None,
            }
        );

        assert!(parse_args(&["disasm-rdna2".to_string()]).is_err());
        assert!(
            parse_args(&[
                "disasm-rdna2".to_string(),
                "vs.bin".to_string(),
                "--max-instructions".to_string(),
                "not-a-number".to_string(),
            ])
            .is_err()
        );
        assert!(
            parse_args(&[
                "disasm-rdna2".to_string(),
                "vs.bin".to_string(),
                "--bogus".to_string(),
            ])
            .is_err()
        );
    }

    #[test]
    fn end_to_end_disasm_rdna2_emits_sanitized_text_and_json() {
        let root = make_temp_dir("playgta6_shaders_disasm_rdna2");
        let bin_path = root.join("splash_vs.bin");

        // Synthetic VS: v_mov_b32 v0, 1.0; exp pos0 v0, v0, v0, v0 done; s_endpgm
        let words: Vec<u32> = vec![
            (0b0111_111u32 << 25) | (0u32 << 17) | (0x01u32 << 9) | 242, // v_mov_b32 v0, 1.0
            (0b111110u32 << 26) | (1u32 << 11) | (12u32 << 4) | 0xF,     // exp pos0 ... done
            0x0000_0000,                                                 // v0, v0, v0, v0
            (0b1011_1111_1u32 << 23) | (0x01u32 << 16),                  // s_endpgm
        ];
        fs::write(&bin_path, words_to_bytes(&words)).unwrap();

        let text = format_disasm_rdna2(&bin_path, false, Some(2)).unwrap();
        assert!(text.contains("; stage:        vertex"));
        assert!(text.contains("v_mov_b32 v0, 1.0"));
        assert!(text.contains("exp pos0 v0, v0, v0, v0 done"));
        assert!(text.contains("1 more instruction(s) omitted"));

        let json = format_disasm_rdna2(&bin_path, true, Some(2)).unwrap();
        assert!(json.contains("\"isa\": \"rdna2_gfx10_3\""));
        assert!(json.contains("\"inferred_stage\": \"vertex\""));
        assert!(json.contains("\"instruction_count\": 3"));
        assert!(json.contains("\"shown_instructions\": 2"));

        // Strictly sanitized: never leak host directory paths into text or JSON.
        let root_str = root.to_string_lossy().replace('\\', "/");
        assert!(!text.contains(&root_str));
        assert!(!json.contains(&root_str));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn cli_parses_translate_rdna2_arguments() {
        let cmd = parse_args(&[
            "translate-rdna2".to_string(),
            "splash_ps.bin".to_string(),
            "--out".to_string(),
            "out/splash.wgsl".to_string(),
            "--stage".to_string(),
            "pixel".to_string(),
            "--json".to_string(),
        ])
        .unwrap();
        assert_eq!(
            cmd,
            Command::TranslateRdna2 {
                file_path: PathBuf::from("splash_ps.bin"),
                out_path: Some(PathBuf::from("out/splash.wgsl")),
                stage: Rdna2StageOverride::Pixel,
                json: true,
            }
        );

        let cmd_min = parse_args(&["translate-rdna2".to_string(), "vs.bin".to_string()]).unwrap();
        assert_eq!(
            cmd_min,
            Command::TranslateRdna2 {
                file_path: PathBuf::from("vs.bin"),
                out_path: None,
                stage: Rdna2StageOverride::Auto,
                json: false,
            }
        );

        assert!(parse_args(&["translate-rdna2".to_string()]).is_err());
        assert!(
            parse_args(&[
                "translate-rdna2".to_string(),
                "vs.bin".to_string(),
                "--stage".to_string(),
                "bogus".to_string(),
            ])
            .is_err()
        );
        assert!(
            parse_args(&[
                "translate-rdna2".to_string(),
                "vs.bin".to_string(),
                "--out".to_string(),
            ])
            .is_err()
        );
    }

    #[test]
    fn end_to_end_translate_rdna2_emits_validated_wgsl_and_sanitized_json() {
        let root = make_temp_dir("playgta6_shaders_translate_rdna2");
        let bin_path = root.join("splash_vs.bin");
        let out_wgsl = root.join("out/splash_vs.wgsl");

        // Synthetic VS: v_mov_b32 v1, 1.0; exp pos0 v0, v1, v1, v1 done; s_endpgm
        let words: Vec<u32> = vec![
            (0b0111_111u32 << 25) | (1u32 << 17) | (0x01u32 << 9) | 242, // v_mov_b32 v1, 1.0
            (0b111110u32 << 26) | (1u32 << 11) | (12u32 << 4) | 0xF,     // exp pos0 ... done
            (1u32 << 24) | (1u32 << 16) | (1u32 << 8),                   // v0, v1, v1, v1
            (0b1011_1111_1u32 << 23) | (0x01u32 << 16),                  // s_endpgm
        ];
        fs::write(&bin_path, words_to_bytes(&words)).unwrap();

        let (wgsl, output, source) =
            format_translate_rdna2(&bin_path, Rdna2StageOverride::Auto, false).unwrap();
        assert_eq!(source, "splash_vs.bin");
        assert!(output.naga_validated);
        assert!(wgsl.contains("@vertex"));
        assert!(wgsl.contains("@builtin(position)"));

        let (json, _, _) =
            format_translate_rdna2(&bin_path, Rdna2StageOverride::Auto, true).unwrap();
        assert!(json.contains("\"isa\": \"rdna2_gfx10_3\""));
        assert!(json.contains("\"source\": \"splash_vs.bin\""));
        assert!(json.contains("\"stage\": \"vertex\""));
        assert!(json.contains("\"naga_validated\": true"));
        assert!(json.contains("\"bindings\": []"));

        run_translate_rdna2(&bin_path, Some(&out_wgsl), Rdna2StageOverride::Auto, false)
            .expect("translate-rdna2 --out should write validated WGSL");
        let written_wgsl = fs::read_to_string(&out_wgsl).unwrap();
        assert_eq!(written_wgsl, wgsl);

        // Verify round-trip parse with naga WGSL front-end.
        let parsed = naga::front::wgsl::parse_str(&written_wgsl).expect("written WGSL must parse");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );
        validator
            .validate(&parsed)
            .expect("written WGSL must pass Naga validation");

        // Strictly sanitized: never leak host directory paths into WGSL or JSON.
        let root_str = root.to_string_lossy().replace('\\', "/");
        assert!(!wgsl.contains(&root_str));
        assert!(!json.contains(&root_str));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn cli_parses_pack_rdna2_arguments() {
        let cmd = parse_args(&[
            "pack-rdna2".to_string(),
            "rdna2_blobs".to_string(),
            "--out".to_string(),
            "out/splash.wgsl.pack".to_string(),
            "--json".to_string(),
        ])
        .unwrap();
        assert_eq!(
            cmd,
            Command::PackRdna2 {
                input_dir: Some(PathBuf::from("rdna2_blobs")),
                manifest_path: None,
                out_path: PathBuf::from("out/splash.wgsl.pack"),
                pack_version: PackVersion::Auto,
                best_effort: false,
                max_wgsl_bytes: None,
                max_pack_bytes: None,
                budget_order: BudgetOrder::HashOrder,
                priority_hashes_path: None,
                jobs: None,
                json: true,
            }
        );

        let cmd_text = parse_args(&[
            "pack-rdna2".to_string(),
            "--out".to_string(),
            "out/all.wgsl.pack".to_string(),
            "rdna2_blobs".to_string(),
        ])
        .unwrap();
        assert_eq!(
            cmd_text,
            Command::PackRdna2 {
                input_dir: Some(PathBuf::from("rdna2_blobs")),
                manifest_path: None,
                out_path: PathBuf::from("out/all.wgsl.pack"),
                pack_version: PackVersion::Auto,
                best_effort: false,
                max_wgsl_bytes: None,
                max_pack_bytes: None,
                budget_order: BudgetOrder::HashOrder,
                priority_hashes_path: None,
                jobs: None,
                json: false,
            }
        );

        // Standalone --manifest mode with size controls, --pack-version 2, --budget-order, --priority-hashes, and --jobs
        let cmd_manifest = parse_args(&[
            "pack-rdna2".to_string(),
            "--manifest".to_string(),
            "work/dump/manifest.json".to_string(),
            "--out".to_string(),
            "out/splash.wgsl.pack".to_string(),
            "--pack-version".to_string(),
            "2".to_string(),
            "--best-effort".to_string(),
            "--max-wgsl-bytes".to_string(),
            "262144".to_string(),
            "--max-pack-bytes".to_string(),
            "536870912".to_string(),
            "--budget-order".to_string(),
            "smallest-first".to_string(),
            "--priority-hashes".to_string(),
            "work/priority_hashes.txt".to_string(),
            "--jobs".to_string(),
            "4".to_string(),
            "--json".to_string(),
        ])
        .unwrap();
        assert_eq!(
            cmd_manifest,
            Command::PackRdna2 {
                input_dir: None,
                manifest_path: Some(PathBuf::from("work/dump/manifest.json")),
                out_path: PathBuf::from("out/splash.wgsl.pack"),
                pack_version: PackVersion::V2,
                best_effort: true,
                max_wgsl_bytes: Some(262_144),
                max_pack_bytes: Some(536_870_912),
                budget_order: BudgetOrder::SmallestFirst,
                priority_hashes_path: Some(PathBuf::from("work/priority_hashes.txt")),
                jobs: Some(4),
                json: true,
            }
        );

        // Explicit --budget-order manifest-order and hash-order
        let cmd_manifest_order = parse_args(&[
            "pack-rdna2".to_string(),
            "--manifest".to_string(),
            "work/dump/manifest.json".to_string(),
            "--out".to_string(),
            "out/early.wgsl.pack".to_string(),
            "--budget-order".to_string(),
            "manifest-order".to_string(),
        ])
        .unwrap();
        assert_eq!(
            cmd_manifest_order,
            Command::PackRdna2 {
                input_dir: None,
                manifest_path: Some(PathBuf::from("work/dump/manifest.json")),
                out_path: PathBuf::from("out/early.wgsl.pack"),
                pack_version: PackVersion::Auto,
                best_effort: false,
                max_wgsl_bytes: None,
                max_pack_bytes: None,
                budget_order: BudgetOrder::ManifestOrder,
                priority_hashes_path: None,
                jobs: None,
                json: false,
            }
        );

        let cmd_hash_order = parse_args(&[
            "pack-rdna2".to_string(),
            "rdna2_blobs".to_string(),
            "--out".to_string(),
            "out/hash.wgsl.pack".to_string(),
            "--budget-order".to_string(),
            "hash-order".to_string(),
        ])
        .unwrap();
        assert_eq!(
            cmd_hash_order,
            Command::PackRdna2 {
                input_dir: Some(PathBuf::from("rdna2_blobs")),
                manifest_path: None,
                out_path: PathBuf::from("out/hash.wgsl.pack"),
                pack_version: PackVersion::Auto,
                best_effort: false,
                max_wgsl_bytes: None,
                max_pack_bytes: None,
                budget_order: BudgetOrder::HashOrder,
                priority_hashes_path: None,
                jobs: None,
                json: false,
            }
        );

        // Single-threaded --jobs 1 and short -j 4 alias
        let cmd_jobs_one = parse_args(&[
            "pack-rdna2".to_string(),
            "rdna2_blobs".to_string(),
            "--out".to_string(),
            "out/single.wgsl.pack".to_string(),
            "--jobs".to_string(),
            "1".to_string(),
        ])
        .unwrap();
        assert_eq!(
            cmd_jobs_one,
            Command::PackRdna2 {
                input_dir: Some(PathBuf::from("rdna2_blobs")),
                manifest_path: None,
                out_path: PathBuf::from("out/single.wgsl.pack"),
                pack_version: PackVersion::Auto,
                best_effort: false,
                max_wgsl_bytes: None,
                max_pack_bytes: None,
                budget_order: BudgetOrder::HashOrder,
                priority_hashes_path: None,
                jobs: Some(1),
                json: false,
            }
        );

        let cmd_jobs_short = parse_args(&[
            "pack-rdna2".to_string(),
            "rdna2_blobs".to_string(),
            "--out".to_string(),
            "out/short.wgsl.pack".to_string(),
            "-j".to_string(),
            "4".to_string(),
        ])
        .unwrap();
        assert_eq!(
            cmd_jobs_short,
            Command::PackRdna2 {
                input_dir: Some(PathBuf::from("rdna2_blobs")),
                manifest_path: None,
                out_path: PathBuf::from("out/short.wgsl.pack"),
                pack_version: PackVersion::Auto,
                best_effort: false,
                max_wgsl_bytes: None,
                max_pack_bytes: None,
                budget_order: BudgetOrder::HashOrder,
                priority_hashes_path: None,
                jobs: Some(4),
                json: false,
            }
        );

        // Combined --manifest + positional input directory mode
        let cmd_combined = parse_args(&[
            "pack-rdna2".to_string(),
            "extra_blobs".to_string(),
            "--manifest".to_string(),
            "work/dump/manifest.json".to_string(),
            "--out".to_string(),
            "out/combined.wgsl.pack".to_string(),
        ])
        .unwrap();
        assert_eq!(
            cmd_combined,
            Command::PackRdna2 {
                input_dir: Some(PathBuf::from("extra_blobs")),
                manifest_path: Some(PathBuf::from("work/dump/manifest.json")),
                out_path: PathBuf::from("out/combined.wgsl.pack"),
                pack_version: PackVersion::Auto,
                best_effort: false,
                max_wgsl_bytes: None,
                max_pack_bytes: None,
                budget_order: BudgetOrder::HashOrder,
                priority_hashes_path: None,
                jobs: None,
                json: false,
            }
        );

        assert!(parse_args(&["pack-rdna2".to_string()]).is_err());
        assert!(parse_args(&["pack-rdna2".to_string(), "rdna2_blobs".to_string()]).is_err());
        assert!(
            parse_args(&[
                "pack-rdna2".to_string(),
                "--out".to_string(),
                "out.wgsl.pack".to_string(),
            ])
            .is_err()
        );
        assert!(
            parse_args(&[
                "pack-rdna2".to_string(),
                "--manifest".to_string(),
            ])
            .is_err()
        );
        assert!(
            parse_args(&[
                "pack-rdna2".to_string(),
                "--manifest".to_string(),
                "a.json".to_string(),
                "--manifest".to_string(),
                "b.json".to_string(),
                "--out".to_string(),
                "out.wgsl.pack".to_string(),
            ])
            .is_err()
        );
        assert!(
            parse_args(&[
                "pack-rdna2".to_string(),
                "rdna2_blobs".to_string(),
                "--out".to_string(),
            ])
            .is_err()
        );
        assert!(
            parse_args(&[
                "pack-rdna2".to_string(),
                "rdna2_blobs".to_string(),
                "--out".to_string(),
                "out.wgsl.pack".to_string(),
                "--max-wgsl-bytes".to_string(),
            ])
            .unwrap_err()
            .contains("--max-wgsl-bytes")
        );
        assert!(
            parse_args(&[
                "pack-rdna2".to_string(),
                "rdna2_blobs".to_string(),
                "--out".to_string(),
                "out.wgsl.pack".to_string(),
                "--max-wgsl-bytes".to_string(),
                "not-a-number".to_string(),
            ])
            .unwrap_err()
            .contains("--max-wgsl-bytes")
        );
        assert!(
            parse_args(&[
                "pack-rdna2".to_string(),
                "rdna2_blobs".to_string(),
                "--out".to_string(),
                "out.wgsl.pack".to_string(),
                "--max-wgsl-bytes".to_string(),
                "-10".to_string(),
            ])
            .is_err()
        );
        assert!(
            parse_args(&[
                "pack-rdna2".to_string(),
                "rdna2_blobs".to_string(),
                "--out".to_string(),
                "out.wgsl.pack".to_string(),
                "--max-pack-bytes".to_string(),
            ])
            .unwrap_err()
            .contains("--max-pack-bytes")
        );
        assert!(
            parse_args(&[
                "pack-rdna2".to_string(),
                "rdna2_blobs".to_string(),
                "--out".to_string(),
                "out.wgsl.pack".to_string(),
                "--max-pack-bytes".to_string(),
                "512mb".to_string(),
            ])
            .unwrap_err()
            .contains("--max-pack-bytes")
        );
        assert!(
            parse_args(&[
                "pack-rdna2".to_string(),
                "rdna2_blobs".to_string(),
                "--out".to_string(),
                "out.wgsl.pack".to_string(),
                "--max-pack-bytes".to_string(),
                "-1".to_string(),
            ])
            .is_err()
        );
        assert!(
            parse_args(&[
                "pack-rdna2".to_string(),
                "rdna2_blobs".to_string(),
                "extra_dir".to_string(),
                "--out".to_string(),
                "out.wgsl.pack".to_string(),
            ])
            .is_err()
        );
        assert!(
            parse_args(&[
                "pack-rdna2".to_string(),
                "rdna2_blobs".to_string(),
                "--out".to_string(),
                "out.wgsl.pack".to_string(),
                "--jobs".to_string(),
            ])
            .unwrap_err()
            .contains("--jobs")
        );
        assert!(
            parse_args(&[
                "pack-rdna2".to_string(),
                "rdna2_blobs".to_string(),
                "--out".to_string(),
                "out.wgsl.pack".to_string(),
                "--jobs".to_string(),
                "0".to_string(),
            ])
            .unwrap_err()
            .contains("--jobs")
        );
        assert!(
            parse_args(&[
                "pack-rdna2".to_string(),
                "rdna2_blobs".to_string(),
                "--out".to_string(),
                "out.wgsl.pack".to_string(),
                "-j".to_string(),
                "0".to_string(),
            ])
            .unwrap_err()
            .contains("-j")
        );
        assert!(
            parse_args(&[
                "pack-rdna2".to_string(),
                "rdna2_blobs".to_string(),
                "--out".to_string(),
                "out.wgsl.pack".to_string(),
                "--jobs".to_string(),
                "abc".to_string(),
            ])
            .unwrap_err()
            .contains("--jobs")
        );
        assert!(
            parse_args(&[
                "pack-rdna2".to_string(),
                "rdna2_blobs".to_string(),
                "--out".to_string(),
                "out.wgsl.pack".to_string(),
                "--jobs".to_string(),
                "-2".to_string(),
            ])
            .is_err()
        );
        assert!(
            parse_args(&[
                "pack-rdna2".to_string(),
                "rdna2_blobs".to_string(),
                "--out".to_string(),
                "out.wgsl.pack".to_string(),
                "--budget-order".to_string(),
            ])
            .unwrap_err()
            .contains("--budget-order")
        );
        assert!(
            parse_args(&[
                "pack-rdna2".to_string(),
                "rdna2_blobs".to_string(),
                "--out".to_string(),
                "out.wgsl.pack".to_string(),
                "--budget-order".to_string(),
                "largest-first".to_string(),
            ])
            .unwrap_err()
            .contains("--budget-order")
        );
        assert!(
            parse_args(&[
                "pack-rdna2".to_string(),
                "rdna2_blobs".to_string(),
                "--out".to_string(),
                "out.wgsl.pack".to_string(),
                "--priority-hashes".to_string(),
            ])
            .unwrap_err()
            .contains("--priority-hashes")
        );
        assert!(
            parse_args(&[
                "pack-rdna2".to_string(),
                "rdna2_blobs".to_string(),
                "--out".to_string(),
                "out.wgsl.pack".to_string(),
                "--priority-hashes".to_string(),
                "a.txt".to_string(),
                "--priority-hashes".to_string(),
                "b.txt".to_string(),
            ])
            .unwrap_err()
            .contains("--priority-hashes")
        );
        assert!(
            parse_args(&[
                "pack-rdna2".to_string(),
                "rdna2_blobs".to_string(),
                "--out".to_string(),
                "out.wgsl.pack".to_string(),
                "--bogus".to_string(),
            ])
            .is_err()
        );
    }

    #[test]
    fn end_to_end_pack_rdna2_writes_wgpk_and_emits_sanitized_summaries() {
        let root = make_temp_dir("playgta6_shaders_pack_rdna2");
        let in_dir = root.join("blobs");
        let out_pack = root.join("out/splash.wgsl.pack");
        fs::create_dir_all(&in_dir).unwrap();

        // Synthetic fullscreen-triangle VS:
        //   v_lshlrev_b32 v1, 1, v0; v_and_b32 v1, 2, v1; v_and_b32 v2, 2, v0;
        //   v_cvt_f32_u32 v1, v1; v_cvt_f32_u32 v2, v2;
        //   v_fma_f32 v3, v1, 2.0, -1.0; v_fma_f32 v4, -v2, 2.0, 1.0;
        //   v_mov_b32 v5, 0; v_mov_b32 v6, 1.0;
        //   exp pos0 v3, v4, v5, v6; exp param0 v1, v2, off, off done; s_endpgm
        let vs_words: Vec<u32> = vec![
            (0x18u32 << 25) | (1u32 << 17) | (128 + 1),
            (0x19u32 << 25) | (1u32 << 17) | (1u32 << 9) | (128 + 2),
            (0x19u32 << 25) | (2u32 << 17) | (128 + 2),
            (0b011_1111_u32 << 25) | (1u32 << 17) | (0x06u32 << 9) | (256 + 1),
            (0b011_1111_u32 << 25) | (2u32 << 17) | (0x06u32 << 9) | (256 + 2),
            (0b110101u32 << 26) | (0x14Bu32 << 16) | 3,
            (243u32 << 18) | (244u32 << 9) | (256 + 1),
            (0b110101u32 << 26) | (0x14Bu32 << 16) | 4,
            (0b001u32 << 29) | (242u32 << 18) | (244u32 << 9) | (256 + 2),
            (0b011_1111_u32 << 25) | (5u32 << 17) | (0x01u32 << 9) | 128,
            (0b011_1111_u32 << 25) | (6u32 << 17) | (0x01u32 << 9) | 242,
            (0b111110u32 << 26) | (12u32 << 4) | 0xF,
            3 | (4 << 8) | (5 << 16) | (6 << 24),
            (0b111110u32 << 26) | (1u32 << 11) | (32u32 << 4) | 0x3,
            1 | (2 << 8),
            (0b1_0111_1111_u32 << 23) | (0x01u32 << 16),
        ];
        let vs_bytes = words_to_bytes(&vs_words);

        // Synthetic textured splash PS:
        //   v_interp_p1/p2 attr0.xy; image_sample; s_waitcnt; exp mrt0 done vm; s_endpgm
        let ps_words: Vec<u32> = vec![
            (0b110010u32 << 26) | (2u32 << 18),
            (0b110010u32 << 26) | (2u32 << 18) | (1u32 << 16) | 1,
            (0b110010u32 << 26) | (3u32 << 18) | (1u32 << 8),
            (0b110010u32 << 26) | (3u32 << 18) | (1u32 << 16) | (1u32 << 8) | 1,
            (0b111100u32 << 26) | (0x20u32 << 18) | (0xFu32 << 8) | (1u32 << 3),
            (1u32 << 21) | (2u32 << 16) | (4u32 << 8) | 2,
            (0b1_0111_1111_u32 << 23) | (0x0Cu32 << 16) | 0x3F70,
            (0b111110u32 << 26) | (1u32 << 12) | (1u32 << 11) | 0xF,
            4 | (5 << 8) | (6 << 16) | (7 << 24),
            (0b1_0111_1111_u32 << 23) | (0x01u32 << 16),
        ];
        let ps_bytes = words_to_bytes(&ps_words);

        // Synthetic compute blob (unlabeled filename -> FNV-1a hash + inferred compute stage)
        let cs_words: Vec<u32> = vec![
            (0b011_1111_u32 << 25) | (1u32 << 17) | (0x01u32 << 9) | 242,
            (0b1_0111_1111_u32 << 23) | (0x0Au32 << 16),
            (0b1_0111_1111_u32 << 23) | (0x01u32 << 16),
        ];
        let cs_bytes = words_to_bytes(&cs_words);
        let cs_hash = playgta6_shaders::convert::fnv1a_64(&cs_bytes);

        fs::write(in_dir.join("ps_0000000000000020.rdna2"), &ps_bytes).unwrap();
        fs::write(in_dir.join("vs_0000000000000010.rdna2"), &vs_bytes).unwrap();
        fs::write(in_dir.join("auto_compute.bin"), &cs_bytes).unwrap();
        fs::write(in_dir.join("truncated.shader"), &[0x01, 0x02]).unwrap();
        fs::write(in_dir.join("readme.md"), b"ignored").unwrap();

        let (json_out, result) =
            format_and_write_pack_rdna2(&in_dir, &out_pack, true).expect("pack-rdna2 JSON should succeed");
        let (text_out, _) =
            format_and_write_pack_rdna2(&in_dir, &out_pack, false).expect("pack-rdna2 text should succeed");

        assert_eq!(result.summary.total_blobs, 4);
        assert_eq!(result.summary.translated_ok, 3);
        assert_eq!(result.summary.skipped_or_failed, 1);
        assert_eq!(result.summary.vertex_count, 1);
        assert_eq!(result.summary.pixel_count, 1);
        assert_eq!(result.summary.compute_count, 1);
        assert_eq!(result.summary.pack_bytes, result.pack_bytes.len());

        // Strictly sanitized JSON & text summaries: zero host paths.
        let root_str = root.to_string_lossy().replace('\\', "/");
        assert!(json_out.contains("\"total_blobs\": 4"));
        assert!(json_out.contains("\"translated_ok\": 3"));
        assert!(json_out.contains("\"skipped_or_failed\": 1"));
        assert!(json_out.contains("\"vertex_count\": 1"));
        assert!(json_out.contains("\"pixel_count\": 1"));
        assert!(json_out.contains("\"compute_count\": 1"));
        assert!(json_out.contains(&format!("\"pack_bytes\": {}", result.pack_bytes.len())));
        assert!(json_out.contains("\"entries\": ["));
        assert!(json_out.contains("\"entry_point\": \"fs_main\""));
        assert!(json_out.contains("\"bindings\": ["));
        assert!(!json_out.contains(&root_str));
        assert!(!text_out.contains(&root_str));
        assert!(text_out.contains("total_blobs:"));
        assert!(text_out.contains("pack_bytes:"));

        // Verify WGPK v1 round-trip from disk via crate::pack::read_pack + Naga WGSL validation.
        let disk_bytes = fs::read(&out_pack).unwrap();
        assert_eq!(disk_bytes, result.pack_bytes);
        let decoded = read_pack(&disk_bytes).expect("disk WGPK pack must parse cleanly");
        assert_eq!(decoded.entries.len(), 3);

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32,
        );
        for entry in &decoded.entries {
            let parsed =
                naga::front::wgsl::parse_str(&entry.wgsl).expect("packed WGSL must parse");
            validator
                .validate(&parsed)
                .expect("packed WGSL must pass Naga validation");
        }

        assert!(decoded.entries.iter().any(|e| e.hash == 0x10 && e.stage == ShaderStage::Vertex));
        let ps_packed = decoded
            .entries
            .iter()
            .find(|e| e.hash == 0x20 && e.stage == ShaderStage::Fragment)
            .expect("packed PS entry must exist");
        assert!(ps_packed.wgsl.contains("fn fs_main("));
        assert!(decoded.entries.iter().any(|e| e.hash == cs_hash && e.stage == ShaderStage::Compute));

        // Verify inspect-pack and run_pack_rdna2 both succeed on the generated pack.
        run_inspect_pack(&out_pack).expect("inspect-pack should accept pack-rdna2 output");
        run_pack_rdna2(
            Some(&in_dir),
            None,
            &out_pack,
            &Rdna2PackOptions::default(),
            false,
        )
        .expect("run_pack_rdna2 should succeed");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn end_to_end_translate_and_pack_rdna2_json_emit_tight_cbv_binding_sizes() {
        let root = make_temp_dir("playgta6_shaders_tight_cbv_cli");
        let in_dir = root.join("blobs");
        let out_pack = root.join("out/tight_cbv.wgsl.pack");
        fs::create_dir_all(&in_dir).unwrap();

        let enc_smem = |op: u8, sdata: u8, sbase_pair: u8, offset: i32| -> [u32; 2] {
            let w0 = (0b111101u32 << 26)
                | (u32::from(op) << 18)
                | (u32::from(sdata & 0x7F) << 6)
                | u32::from(sbase_pair & 0x3F);
            let w1 = (0x7Du32 << 25) | ((offset as u32) & 0x001F_FFFF);
            [w0, w1]
        };

        // 1) Synthetic PS reading cb[0..2] (offsets 0..44) -> 48 B (array<vec4<u32>, 3>)
        let mut small_words = Vec::new();
        small_words.extend_from_slice(&enc_smem(0x0A, 4, 0, 0x00));
        small_words.extend_from_slice(&enc_smem(0x08, 8, 0, 0x2C));
        small_words.push((0b0111_111u32 << 25) | (0u32 << 17) | (0x01u32 << 9) | 4);
        small_words.push((0b0111_111u32 << 25) | (1u32 << 17) | (0x01u32 << 9) | 8);
        small_words.extend_from_slice(&[
            (0b111110u32 << 26) | (1u32 << 12) | (1u32 << 11) | 0xF,
            0 | (1 << 8) | (0 << 16) | (1 << 24),
        ]);
        small_words.push((0b1011_1111_1u32 << 23) | (0x01u32 << 16));

        // 2) Synthetic PS with two CBVs: cb0 (vec4 1 -> 32 B) + cb1 (vec4 5 -> 96 B)
        let mut multi_words = Vec::new();
        multi_words.extend_from_slice(&enc_smem(0x0A, 4, 0, 0x10));
        multi_words.extend_from_slice(&enc_smem(0x0A, 8, 2, 0x50));
        multi_words.push((0b0111_111u32 << 25) | (0u32 << 17) | (0x01u32 << 9) | 4);
        multi_words.push((0b0111_111u32 << 25) | (1u32 << 17) | (0x01u32 << 9) | 8);
        multi_words.extend_from_slice(&[
            (0b111110u32 << 26) | (1u32 << 12) | (1u32 << 11) | 0xF,
            0 | (1 << 8) | (0 << 16) | (1 << 24),
        ]);
        multi_words.push((0b1011_1111_1u32 << 23) | (0x01u32 << 16));

        let small_path = in_dir.join("ps_0000000000000201.rdna2");
        let multi_path = in_dir.join("ps_0000000000000202.rdna2");
        fs::write(&small_path, words_to_bytes(&small_words)).unwrap();
        fs::write(&multi_path, words_to_bytes(&multi_words)).unwrap();

        // translate-rdna2 --json for small 48 B CBV
        let (small_json, small_out, _) =
            format_translate_rdna2(&small_path, Rdna2StageOverride::Auto, true)
                .expect("translate-rdna2 --json should succeed");
        assert!(small_out.wgsl.contains("array<vec4<u32>, 3>"));
        assert!(
            small_json.contains(
                "{\"group\": 0, \"binding\": 0, \"kind\": \"uniform\", \"name\": \"cb0\", \"min_binding_size_bytes\": 48, \"highest_vec4_offset\": 2}"
            ),
            "JSON:\n{small_json}"
        );

        // translate-rdna2 --json for multi-CBV (32 B + 96 B)
        let (multi_json, multi_out, _) =
            format_translate_rdna2(&multi_path, Rdna2StageOverride::Auto, true)
                .expect("translate-rdna2 --json should succeed for multi-CBV");
        assert!(multi_out.wgsl.contains("array<vec4<u32>, 2>"));
        assert!(multi_out.wgsl.contains("array<vec4<u32>, 6>"));
        assert!(
            multi_json.contains(
                "{\"group\": 0, \"binding\": 0, \"kind\": \"uniform\", \"name\": \"cb0\", \"min_binding_size_bytes\": 32, \"highest_vec4_offset\": 1}"
            ),
            "JSON:\n{multi_json}"
        );
        assert!(
            multi_json.contains(
                "{\"group\": 0, \"binding\": 16, \"kind\": \"uniform\", \"name\": \"cb1\", \"min_binding_size_bytes\": 96, \"highest_vec4_offset\": 5}"
            ),
            "JSON:\n{multi_json}"
        );

        // pack-rdna2 --json emits the same tight binding metadata for all packed shaders
        let (pack_json, _) = format_and_write_pack_rdna2(&in_dir, &out_pack, true)
            .expect("pack-rdna2 --json should succeed");
        assert!(
            pack_json.contains("\"min_binding_size_bytes\": 48, \"highest_vec4_offset\": 2"),
            "pack JSON:\n{pack_json}"
        );
        assert!(
            pack_json.contains("\"min_binding_size_bytes\": 32, \"highest_vec4_offset\": 1"),
            "pack JSON:\n{pack_json}"
        );
        assert!(
            pack_json.contains("\"min_binding_size_bytes\": 96, \"highest_vec4_offset\": 5"),
            "pack JSON:\n{pack_json}"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn end_to_end_pack_rdna2_manifest_cli_packs_vs_ps_cs_and_supports_best_effort() {
        use playgta6_shaders::convert::fnv1a_64;

        let root = make_temp_dir("playgta6_shaders_pack_rdna2_manifest_cli");
        let dump_dir = root.join("pipeline_dump");
        let native_dir = dump_dir.join("native");
        let out_pack = root.join("out/splash.wgsl.pack");
        fs::create_dir_all(&native_dir).unwrap();

        // Synthetic fullscreen-triangle VS
        let vs_words: Vec<u32> = vec![
            (0x18u32 << 25) | (1u32 << 17) | (128 + 1),
            (0x19u32 << 25) | (1u32 << 17) | (1u32 << 9) | (128 + 2),
            (0x19u32 << 25) | (2u32 << 17) | (128 + 2),
            (0b011_1111_u32 << 25) | (1u32 << 17) | (0x06u32 << 9) | (256 + 1),
            (0b011_1111_u32 << 25) | (2u32 << 17) | (0x06u32 << 9) | (256 + 2),
            (0b110101u32 << 26) | (0x14Bu32 << 16) | 3,
            (243u32 << 18) | (244u32 << 9) | (256 + 1),
            (0b110101u32 << 26) | (0x14Bu32 << 16) | 4,
            (0b001u32 << 29) | (242u32 << 18) | (244u32 << 9) | (256 + 2),
            (0b011_1111_u32 << 25) | (5u32 << 17) | (0x01u32 << 9) | 128,
            (0b011_1111_u32 << 25) | (6u32 << 17) | (0x01u32 << 9) | 242,
            (0b111110u32 << 26) | (12u32 << 4) | 0xF,
            3 | (4 << 8) | (5 << 16) | (6 << 24),
            (0b111110u32 << 26) | (1u32 << 11) | (32u32 << 4) | 0x3,
            1 | (2 << 8),
            (0b1_0111_1111_u32 << 23) | (0x01u32 << 16),
        ];
        let vs_bytes = words_to_bytes(&vs_words);
        let vs_hash = fnv1a_64(&vs_bytes);
        let vs_id = format_shader_hash(vs_hash);

        // Synthetic PS with tight 48 B CBV (cb0[0..2]) + exp mrt0
        let enc_smem = |op: u8, sdata: u8, sbase_pair: u8, offset: i32| -> [u32; 2] {
            let w0 = (0b111101u32 << 26)
                | (u32::from(op) << 18)
                | (u32::from(sdata & 0x7F) << 6)
                | u32::from(sbase_pair & 0x3F);
            let w1 = (0x7Du32 << 25) | ((offset as u32) & 0x001F_FFFF);
            [w0, w1]
        };
        let mut ps_words = Vec::new();
        ps_words.extend_from_slice(&enc_smem(0x0A, 4, 0, 0x00));
        ps_words.extend_from_slice(&enc_smem(0x08, 8, 0, 0x2C));
        ps_words.push((0b011_1111_u32 << 25) | (0x01u32 << 9) | 4);
        ps_words.push((0b011_1111_u32 << 25) | (1u32 << 17) | (0x01u32 << 9) | 8);
        ps_words.extend_from_slice(&[
            (0b111110u32 << 26) | (1u32 << 12) | (1u32 << 11) | 0xF,
            (1 << 8) | (1 << 24),
        ]);
        ps_words.push((0b1_0111_1111_u32 << 23) | (0x01u32 << 16));
        let ps_bytes = words_to_bytes(&ps_words);
        let ps_hash = fnv1a_64(&ps_bytes);
        let ps_id = format_shader_hash(ps_hash);

        // Synthetic compute shader
        let cs_words: Vec<u32> = vec![
            (0b011_1111_u32 << 25) | (1u32 << 17) | (0x01u32 << 9) | 242,
            (0b1_0111_1111_u32 << 23) | (0x0Au32 << 16),
            (0b1_0111_1111_u32 << 23) | (0x01u32 << 16),
        ];
        let cs_bytes = words_to_bytes(&cs_words);
        let cs_hash = fnv1a_64(&cs_bytes);
        let cs_id = format_shader_hash(cs_hash);

        fs::write(native_dir.join(format!("{vs_id}.bin")), &vs_bytes).unwrap();
        fs::write(native_dir.join(format!("{ps_id}.bin")), &ps_bytes).unwrap();
        fs::write(native_dir.join(format!("{cs_id}.bin")), &cs_bytes).unwrap();

        let manifest_path = dump_dir.join("manifest.json");
        fs::write(
            &manifest_path,
            format!(
                r#"{{
  "schema": "playgta6-pipeline-dump",
  "schema_version": 1,
  "native_stages": [
    {{"code_identity": "{vs_id}", "identity": "0x0000000000000001", "stage": "vertex", "layout": "A", "code_bytes": {}, "metadata_bytes": 0, "code_file": "native/{vs_id}.bin", "metadata_file": "native/{vs_id}.meta"}},
    {{"code_identity": "{ps_id}", "identity": "0x0000000000000002", "stage": "pixel", "layout": "A", "code_bytes": {}, "metadata_bytes": 0, "code_file": "native/{ps_id}.bin", "metadata_file": "native/{ps_id}.meta"}},
    {{"code_identity": "{cs_id}", "identity": "0x0000000000000003", "stage": "compute", "layout": "A", "code_bytes": {}, "metadata_bytes": 0, "code_file": "native/{cs_id}.bin", "metadata_file": "native/{cs_id}.meta"}}
  ],
  "dxil": [],
  "root_signatures": [],
  "pipelines": []
}}"#,
                vs_bytes.len(),
                ps_bytes.len(),
                cs_bytes.len()
            ),
        )
        .unwrap();

        let parsed_cmd = parse_args(&[
            "pack-rdna2".to_string(),
            "--manifest".to_string(),
            manifest_path.to_string_lossy().into_owned(),
            "--out".to_string(),
            out_pack.to_string_lossy().into_owned(),
            "--json".to_string(),
        ])
        .expect("pack-rdna2 --manifest CLI args should parse");

        let Command::PackRdna2 {
            input_dir,
            manifest_path: parsed_manifest,
            out_path,
            pack_version,
            best_effort,
            max_wgsl_bytes,
            max_pack_bytes,
            budget_order,
            priority_hashes_path: _,
            jobs,
            json,
        } = parsed_cmd
        else {
            panic!("expected Command::PackRdna2");
        };

        let parsed_options = Rdna2PackOptions {
            pack_version,
            best_effort,
            max_wgsl_bytes,
            max_pack_bytes,
            budget_order,
            jobs,
            ..Rdna2PackOptions::default()
        };
        let (json_out, result) = format_and_write_pack_rdna2_sources(
            input_dir.as_deref(),
            parsed_manifest.as_deref(),
            &out_path,
            &parsed_options,
            json,
        )
        .expect("pack-rdna2 --manifest should succeed");

        assert_eq!(result.summary.vertex_packed, 1);
        assert_eq!(result.summary.fragment_packed, 1);
        assert_eq!(result.summary.compute_packed, 1);
        assert_eq!(result.summary.failed, 0);
        assert_eq!(result.summary.translated_ok, 3);

        // Verify WGPK v1 on disk + Naga validation + all 3 hashes & entry points
        let disk_bytes = fs::read(&out_pack).unwrap();
        let decoded = read_pack(&disk_bytes).expect("WGPK pack from --manifest must parse");
        assert_eq!(decoded.entries.len(), 3);
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32,
        );
        for entry in &decoded.entries {
            let parsed =
                naga::front::wgsl::parse_str(&entry.wgsl).expect("packed WGSL must parse");
            validator
                .validate(&parsed)
                .expect("packed WGSL must pass Naga validation");
        }
        assert!(decoded
            .entries
            .iter()
            .any(|e| e.hash == vs_hash && e.stage == ShaderStage::Vertex && e.wgsl.contains("fn vs_main(")));
        assert!(decoded
            .entries
            .iter()
            .any(|e| e.hash == ps_hash && e.stage == ShaderStage::Fragment && e.wgsl.contains("fn fs_main(")));
        assert!(decoded
            .entries
            .iter()
            .any(|e| e.hash == cs_hash && e.stage == ShaderStage::Compute && e.wgsl.contains("fn cs_main(")));

        // Verify --json output & sanitization
        let root_str = root.to_string_lossy().replace('\\', "/");
        assert!(json_out.contains("\"vertex_packed\": 1"), "JSON:\n{json_out}");
        assert!(json_out.contains("\"fragment_packed\": 1"), "JSON:\n{json_out}");
        assert!(json_out.contains("\"compute_packed\": 1"), "JSON:\n{json_out}");
        assert!(json_out.contains("\"failed\": 0"), "JSON:\n{json_out}");
        assert!(json_out.contains("\"naga_validated\": true"), "JSON:\n{json_out}");
        assert!(json_out.contains("\"entry_point\": \"vs_main\""), "JSON:\n{json_out}");
        assert!(json_out.contains("\"entry_point\": \"fs_main\""), "JSON:\n{json_out}");
        assert!(json_out.contains("\"entry_point\": \"cs_main\""), "JSON:\n{json_out}");
        assert!(json_out.contains("\"min_binding_size_bytes\": 48"), "JSON:\n{json_out}");
        assert!(!json_out.contains(&root_str), "JSON must not leak host path");
        run_inspect_pack(&out_pack).expect("inspect-pack should accept manifest pack");

        // Now add a broken entry (missing file) and verify strict aborts while --best-effort succeeds.
        let broken_manifest = dump_dir.join("manifest_with_broken.json");
        fs::write(
            &broken_manifest,
            format!(
                r#"{{
  "schema": "playgta6-pipeline-dump",
  "schema_version": 1,
  "native_stages": [
    {{"code_identity": "{vs_id}", "stage": "vertex", "code_file": "native/{vs_id}.bin"}},
    {{"code_identity": "0x000000000000dead", "stage": "pixel", "code_file": "native/0x000000000000dead.bin"}}
  ]
}}"#
            ),
        )
        .unwrap();

        assert!(
            format_and_write_pack_rdna2_sources(
                None,
                Some(&broken_manifest),
                &out_pack,
                &Rdna2PackOptions::default(),
                true,
            )
            .is_err(),
            "strict --manifest mode should fail on missing code_file"
        );
        let be_opts = Rdna2PackOptions {
            best_effort: true,
            ..Rdna2PackOptions::default()
        };
        let (be_json, be_res) = format_and_write_pack_rdna2_sources(
            None,
            Some(&broken_manifest),
            &out_pack,
            &be_opts,
            true,
        )
        .expect("--best-effort should skip missing shader and pack valid ones");
        assert_eq!(be_res.summary.vertex_packed, 1);
        assert_eq!(be_res.summary.fragment_packed, 0);
        assert_eq!(be_res.summary.compute_packed, 0);
        assert_eq!(be_res.summary.failed, 1);
        assert!(be_json.contains("\"failed\": 1"), "JSON:\n{be_json}");
        assert!(be_json.contains("\"0x000000000000dead\""), "JSON:\n{be_json}");
        assert!(!be_json.contains(&root_str));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn end_to_end_pack_rdna2_cli_size_controls_and_json_diagnostics() {
        let root = make_temp_dir("playgta6_shaders_pack_rdna2_size_controls");
        let in_dir = root.join("blobs");
        let out_pack = root.join("out/capped.wgsl.pack");
        fs::create_dir_all(&in_dir).unwrap();

        let cs_bytes = words_to_bytes(&[
            (0b011_1111_u32 << 25) | (1u32 << 17) | (0x01u32 << 9) | 242,
            (0b1_0111_1111_u32 << 23) | (0x0Au32 << 16),
            (0b1_0111_1111_u32 << 23) | (0x01u32 << 16),
        ]);
        let mut big_cs_words = Vec::new();
        for _ in 0..80 {
            big_cs_words.push((0b011_1111_u32 << 25) | (1u32 << 17) | (0x01u32 << 9) | 242);
            big_cs_words.push((0x18u32 << 25) | (2u32 << 17) | (1u32 << 9) | (128 + 1));
        }
        big_cs_words.push((0b1_0111_1111_u32 << 23) | (0x0Au32 << 16));
        big_cs_words.push((0b1_0111_1111_u32 << 23) | (0x01u32 << 16));
        let big_cs_bytes = words_to_bytes(&big_cs_words);

        fs::write(in_dir.join("a_cs_0000000000003001.rdna2"), &cs_bytes).unwrap();
        fs::write(in_dir.join("b_cs_0000000000003002.rdna2"), &cs_bytes).unwrap();
        fs::write(in_dir.join("c_cs_0000000000003003.rdna2"), &big_cs_bytes).unwrap();

        let (_, baseline) =
            format_and_write_pack_rdna2(&in_dir, &out_pack, true).expect("baseline pack succeeds");
        let small_len = baseline
            .entries
            .iter()
            .find(|e| e.hash == 0x3001)
            .unwrap()
            .wgsl
            .len();
        let big_len = baseline
            .entries
            .iter()
            .find(|e| e.hash == 0x3003)
            .unwrap()
            .wgsl
            .len();
        assert!(big_len > small_len);

        // Fit only 1 small shader by pack budget, and also cap individual WGSL size below big_len.
        // Note: c_cs comes last in sorted order, so test --max-wgsl-bytes first on its own, then combined.
        let one_entry_budget = (PACK_HEADER_BYTES + PACK_ENTRY_BYTES + small_len) as u64;
        let parsed = parse_args(&[
            "pack-rdna2".to_string(),
            in_dir.to_string_lossy().into_owned(),
            "--out".to_string(),
            out_pack.to_string_lossy().into_owned(),
            "--max-wgsl-bytes".to_string(),
            small_len.to_string(),
            "--max-pack-bytes".to_string(),
            (one_entry_budget + PACK_ENTRY_BYTES as u64 + small_len as u64).to_string(),
            "--json".to_string(),
        ])
        .expect("size control flags should parse");
        let Command::PackRdna2 {
            input_dir,
            manifest_path,
            out_path,
            pack_version,
            best_effort,
            max_wgsl_bytes,
            max_pack_bytes,
            budget_order,
            priority_hashes_path: _,
            jobs,
            json,
        } = parsed
        else {
            panic!("expected Command::PackRdna2");
        };
        let opts = Rdna2PackOptions {
            pack_version,
            best_effort,
            max_wgsl_bytes,
            max_pack_bytes,
            budget_order,
            jobs,
            ..Rdna2PackOptions::default()
        };
        let (json_out, res) = format_and_write_pack_rdna2_sources(
            input_dir.as_deref(),
            manifest_path.as_deref(),
            &out_path,
            &opts,
            json,
        )
        .expect("pack-rdna2 with --max-wgsl-bytes and --max-pack-bytes should succeed");

        assert_eq!(res.summary.translated_ok, 2);
        assert_eq!(res.summary.skipped_oversize, 1);
        assert_eq!(res.summary.skipped_budget, 0);
        assert_eq!(res.summary.largest_wgsl_bytes, small_len);
        assert_eq!(res.summary.total_wgsl_bytes, (small_len * 2) as u64);
        assert!(json_out.contains("\"skipped_oversize\": 1"), "JSON:\n{json_out}");
        assert!(json_out.contains("\"skipped_budget\": 0"), "JSON:\n{json_out}");
        assert!(
            json_out.contains(&format!("\"largest_wgsl_bytes\": {small_len}")),
            "JSON:\n{json_out}"
        );
        assert!(
            json_out.contains(&format!("\"total_wgsl_bytes\": {}", (small_len * 2) as u64)),
            "JSON:\n{json_out}"
        );

        // Now enforce a tight --max-pack-bytes that fits only 1 shader.
        let tight_opts = Rdna2PackOptions {
            max_pack_bytes: Some(one_entry_budget),
            ..Rdna2PackOptions::default()
        };
        let (tight_json, tight_res) = format_and_write_pack_rdna2_sources(
            Some(&in_dir),
            None,
            &out_pack,
            &tight_opts,
            true,
        )
        .expect("tight --max-pack-bytes should succeed");
        assert_eq!(tight_res.summary.translated_ok, 1);
        assert_eq!(tight_res.summary.skipped_budget, 2);
        assert_eq!(fs::read(&out_pack).unwrap().len() as u64, one_entry_budget);
        assert!(
            tight_json.contains("\"skipped_budget\": 2"),
            "JSON:\n{tight_json}"
        );
        let root_str = root.to_string_lossy().replace('\\', "/");
        assert!(!tight_json.contains(&root_str));
        assert_eq!(read_pack(&fs::read(&out_pack).unwrap()).unwrap().entries.len(), 1);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn end_to_end_pack_rdna2_cli_jobs_4_matches_jobs_1_bit_for_bit() {
        let root = make_temp_dir("playgta6_shaders_pack_rdna2_jobs_cli");
        let in_dir = root.join("blobs");
        let out_j1 = root.join("out/j1.wgsl.pack");
        let out_j4 = root.join("out/j4.wgsl.pack");
        fs::create_dir_all(&in_dir).unwrap();

        let cs_bytes = words_to_bytes(&[
            (0b011_1111_u32 << 25) | (1u32 << 17) | (0x01u32 << 9) | 242,
            (0b1_0111_1111_u32 << 23) | (0x0Au32 << 16),
            (0b1_0111_1111_u32 << 23) | (0x01u32 << 16),
        ]);
        let mut big_cs_words = Vec::new();
        for _ in 0..80 {
            big_cs_words.push((0b011_1111_u32 << 25) | (1u32 << 17) | (0x01u32 << 9) | 242);
            big_cs_words.push((0x18u32 << 25) | (2u32 << 17) | (1u32 << 9) | (128 + 1));
        }
        big_cs_words.push((0b1_0111_1111_u32 << 23) | (0x0Au32 << 16));
        big_cs_words.push((0b1_0111_1111_u32 << 23) | (0x01u32 << 16));
        let big_cs_bytes = words_to_bytes(&big_cs_words);

        fs::write(in_dir.join("a_cs_0000000000005001.rdna2"), &cs_bytes).unwrap();
        fs::write(in_dir.join("b_cs_0000000000005002.rdna2"), &cs_bytes).unwrap();
        fs::write(in_dir.join("c_cs_dup_0000000000005001.rdna2"), &cs_bytes).unwrap();
        fs::write(in_dir.join("d_cs_0000000000005003.rdna2"), &big_cs_bytes).unwrap();
        fs::write(in_dir.join("e_truncated.bin"), &[0x01, 0x02]).unwrap();

        let run_cli = |jobs_arg: &str, out_file: &Path| -> (String, Vec<u8>) {
            let cmd = parse_args(&[
                "pack-rdna2".to_string(),
                in_dir.to_string_lossy().into_owned(),
                "--out".to_string(),
                out_file.to_string_lossy().into_owned(),
                "--best-effort".to_string(),
                "--jobs".to_string(),
                jobs_arg.to_string(),
                "--json".to_string(),
            ])
            .expect("pack-rdna2 --jobs CLI args should parse");
            let Command::PackRdna2 {
                input_dir,
                manifest_path,
                out_path,
                pack_version,
                best_effort,
                max_wgsl_bytes,
                max_pack_bytes,
                budget_order,
                priority_hashes_path: _,
                jobs,
                json,
            } = cmd
            else {
                panic!("expected Command::PackRdna2");
            };
            let opts = Rdna2PackOptions {
                pack_version,
                best_effort,
                max_wgsl_bytes,
                max_pack_bytes,
                budget_order,
                jobs,
                ..Rdna2PackOptions::default()
            };
            let (json_out, _) = format_and_write_pack_rdna2_sources(
                input_dir.as_deref(),
                manifest_path.as_deref(),
                &out_path,
                &opts,
                json,
            )
            .expect("pack-rdna2 should succeed");
            let bytes = fs::read(out_file).unwrap();
            (json_out, bytes)
        };

        let (json_j1, bytes_j1) = run_cli("1", &out_j1);
        let (json_j4, bytes_j4) = run_cli("4", &out_j4);
        assert_eq!(bytes_j1, bytes_j4);
        assert_eq!(json_j1, json_j4);
        assert_eq!(read_pack(&bytes_j4).unwrap().entries.len(), 3);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn end_to_end_pack_rdna2_cli_budget_order_smallest_first_emits_json_and_packs_small_high_hash()
    {
        let root = make_temp_dir("playgta6_shaders_pack_rdna2_budget_order_cli");
        let in_dir = root.join("blobs");
        let out_pack = root.join("out/smallest.wgsl.pack");
        fs::create_dir_all(&in_dir).unwrap();

        let small_cs = words_to_bytes(&[
            (0b011_1111_u32 << 25) | (1u32 << 17) | (0x01u32 << 9) | 242,
            (0b1_0111_1111_u32 << 23) | (0x0Au32 << 16),
            (0b1_0111_1111_u32 << 23) | (0x01u32 << 16),
        ]);
        let mut huge_cs_words = Vec::new();
        for _ in 0..160 {
            huge_cs_words.push((0b011_1111_u32 << 25) | (1u32 << 17) | (0x01u32 << 9) | 242);
            huge_cs_words.push((0x18u32 << 25) | (2u32 << 17) | (1u32 << 9) | (128 + 1));
        }
        huge_cs_words.push((0b1_0111_1111_u32 << 23) | (0x0Au32 << 16));
        huge_cs_words.push((0b1_0111_1111_u32 << 23) | (0x01u32 << 16));
        let huge_cs = words_to_bytes(&huge_cs_words);

        // Low-hash huge shader + 2 high-hash compact shaders.
        fs::write(in_dir.join("cs_0000000000000001.rdna2"), &huge_cs).unwrap();
        fs::write(in_dir.join("cs_a000000000000002.rdna2"), &small_cs).unwrap();
        fs::write(in_dir.join("cs_f000000000000003.rdna2"), &small_cs).unwrap();

        let (_, baseline) =
            format_and_write_pack_rdna2(&in_dir, &out_pack, true).expect("baseline pack succeeds");
        let small_len = baseline
            .entries
            .iter()
            .find(|e| e.hash == 0xa000_0000_0000_0002)
            .unwrap()
            .wgsl
            .len() as u64;
        let two_small_budget =
            PACK_HEADER_BYTES as u64 + 2 * (PACK_ENTRY_BYTES as u64 + small_len);

        let cmd = parse_args(&[
            "pack-rdna2".to_string(),
            in_dir.to_string_lossy().into_owned(),
            "--out".to_string(),
            out_pack.to_string_lossy().into_owned(),
            "--max-pack-bytes".to_string(),
            two_small_budget.to_string(),
            "--budget-order".to_string(),
            "smallest-first".to_string(),
            "--json".to_string(),
        ])
        .expect("pack-rdna2 --budget-order smallest-first should parse");

        let Command::PackRdna2 {
            input_dir,
            manifest_path,
            out_path,
            pack_version,
            best_effort,
            max_wgsl_bytes,
            max_pack_bytes,
            budget_order,
            priority_hashes_path: _,
            jobs,
            json,
        } = cmd
        else {
            panic!("expected Command::PackRdna2");
        };
        assert_eq!(budget_order, BudgetOrder::SmallestFirst);

        let opts = Rdna2PackOptions {
            pack_version,
            best_effort,
            max_wgsl_bytes,
            max_pack_bytes,
            budget_order,
            jobs,
            ..Rdna2PackOptions::default()
        };
        let (json_out, res) = format_and_write_pack_rdna2_sources(
            input_dir.as_deref(),
            manifest_path.as_deref(),
            &out_path,
            &opts,
            json,
        )
        .expect("pack-rdna2 --budget-order smallest-first should succeed");

        assert_eq!(res.summary.translated_ok, 2);
        assert_eq!(res.summary.omitted_by_budget, 1);
        assert_eq!(res.summary.skipped_budget, 1);
        assert_eq!(res.summary.budget_order, BudgetOrder::SmallestFirst);
        assert!(
            json_out.contains("\"budget_order\": \"smallest-first\""),
            "JSON:\n{json_out}"
        );
        assert!(
            json_out.contains("\"omitted_by_budget\": 1"),
            "JSON:\n{json_out}"
        );

        let decoded = read_pack(&fs::read(&out_pack).unwrap()).unwrap();
        let hashes: Vec<u64> = decoded.entries.iter().map(|e| e.hash).collect();
        assert_eq!(hashes, vec![0xa000_0000_0000_0002, 0xf000_0000_0000_0003]);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn end_to_end_pack_rdna2_cli_priority_hashes_packs_large_priority_ahead_of_smallest_first() {
        let root = make_temp_dir("playgta6_shaders_pack_rdna2_priority_hashes_cli");
        let in_dir = root.join("blobs");
        let out_pack = root.join("out/prioritized.wgsl.pack");
        let priority_file = root.join("priority_hashes.txt");
        fs::create_dir_all(&in_dir).unwrap();

        let small_cs = words_to_bytes(&[
            (0b011_1111_u32 << 25) | (1u32 << 17) | (0x01u32 << 9) | 242,
            (0b1_0111_1111_u32 << 23) | (0x0Au32 << 16),
            (0b1_0111_1111_u32 << 23) | (0x01u32 << 16),
        ]);
        let mut large_cs_words = Vec::new();
        for _ in 0..160 {
            large_cs_words.push((0b011_1111_u32 << 25) | (1u32 << 17) | (0x01u32 << 9) | 242);
            large_cs_words.push((0x18u32 << 25) | (2u32 << 17) | (1u32 << 9) | (128 + 1));
        }
        large_cs_words.push((0b1_0111_1111_u32 << 23) | (0x0Au32 << 16));
        large_cs_words.push((0b1_0111_1111_u32 << 23) | (0x01u32 << 16));
        let large_cs = words_to_bytes(&large_cs_words);

        fs::write(in_dir.join("cs_f000000000000001.rdna2"), &large_cs).unwrap();
        fs::write(in_dir.join("cs_0000000000000002.rdna2"), &small_cs).unwrap();
        fs::write(in_dir.join("cs_0000000000000003.rdna2"), &small_cs).unwrap();

        let (_, baseline) =
            format_and_write_pack_rdna2(&in_dir, &out_pack, true).expect("baseline pack succeeds");
        let large_len = baseline
            .entries
            .iter()
            .find(|e| e.hash == 0xf000_0000_0000_0001)
            .unwrap()
            .wgsl
            .len() as u64;
        let small_len = baseline
            .entries
            .iter()
            .find(|e| e.hash == 0x0000_0000_0000_0002)
            .unwrap()
            .wgsl
            .len() as u64;

        // Budget fits `large + 1 small`, forcing 1 small shader out when `large` is prioritized.
        let budget = PACK_HEADER_BYTES as u64
            + (PACK_ENTRY_BYTES as u64 + large_len)
            + (PACK_ENTRY_BYTES as u64 + small_len);

        fs::write(
            &priority_file,
            "# Critical hot shaders\n\n0xF000000000000001 # large shader\ndeadbeef00000009\n",
        )
        .unwrap();

        let cmd = parse_args(&[
            "pack-rdna2".to_string(),
            in_dir.to_string_lossy().into_owned(),
            "--out".to_string(),
            out_pack.to_string_lossy().into_owned(),
            "--max-pack-bytes".to_string(),
            budget.to_string(),
            "--budget-order".to_string(),
            "smallest-first".to_string(),
            "--priority-hashes".to_string(),
            priority_file.to_string_lossy().into_owned(),
            "--json".to_string(),
        ])
        .expect("pack-rdna2 --priority-hashes should parse");

        let Command::PackRdna2 {
            input_dir,
            manifest_path,
            out_path,
            pack_version,
            best_effort,
            max_wgsl_bytes,
            max_pack_bytes,
            budget_order,
            priority_hashes_path,
            jobs,
            json,
        } = cmd
        else {
            panic!("expected Command::PackRdna2");
        };

        let priority_hashes =
            load_cli_priority_hashes(priority_hashes_path.as_deref()).expect("priority file valid");
        assert_eq!(
            priority_hashes,
            vec![0xf000_0000_0000_0001, 0xdead_beef_0000_0009]
        );

        let opts = Rdna2PackOptions {
            pack_version,
            best_effort,
            max_wgsl_bytes,
            max_pack_bytes,
            budget_order,
            priority_hashes,
            jobs,
            ..Rdna2PackOptions::default()
        };
        let (json_out, res) = format_and_write_pack_rdna2_sources(
            input_dir.as_deref(),
            manifest_path.as_deref(),
            &out_path,
            &opts,
            json,
        )
        .expect("pack-rdna2 --priority-hashes should succeed");

        assert_eq!(res.summary.translated_ok, 2);
        assert_eq!(res.summary.omitted_by_budget, 1);
        assert_eq!(res.summary.priority_hashes_requested, 2);
        assert_eq!(res.summary.priority_hashes_packed, 1);
        assert!(
            json_out.contains("\"priority_hashes_requested\": 2"),
            "JSON:\n{json_out}"
        );
        assert!(
            json_out.contains("\"priority_hashes_packed\": 1"),
            "JSON:\n{json_out}"
        );
        let root_str = root.to_string_lossy().replace('\\', "/");
        assert!(!json_out.contains(&root_str), "JSON must not leak host path");

        let decoded = read_pack(&fs::read(&out_pack).unwrap()).unwrap();
        let hashes: Vec<u64> = decoded.entries.iter().map(|e| e.hash).collect();
        assert_eq!(hashes, vec![0x0000_0000_0000_0002, 0xf000_0000_0000_0001]);

        // Verify invalid hex line returns a clean error without leaking host path.
        let bad_file = root.join("bad_priority.txt");
        fs::write(&bad_file, "0xnot_valid_hex\n").unwrap();
        let err = load_cli_priority_hashes(Some(&bad_file))
            .expect_err("invalid priority hashes file must error");
        assert!(err.contains("line 1"), "err: {err}");
        assert!(!err.contains(&root_str), "err must not leak host path: {err}");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn cli_parses_and_translates_mesh_stage_override() {
        let cmd = parse_args(&[
            "translate-rdna2".to_string(),
            "mesh.ms.bin".to_string(),
            "--stage".to_string(),
            "mesh".to_string(),
            "--json".to_string(),
        ])
        .expect("translate-rdna2 --stage mesh must parse");
        assert_eq!(
            cmd,
            Command::TranslateRdna2 {
                file_path: PathBuf::from("mesh.ms.bin"),
                out_path: None,
                stage: Rdna2StageOverride::Mesh,
                json: true,
            }
        );

        let root = make_temp_dir("playgta6_cli_mesh");
        let bin_path = root.join("sample.ms.bin");
        // s_mov_b32 m0, (1 << 12) | 3 ; s_sendmsg MSG_GS_ALLOC_REQ ; exp pos0 ; exp prim ; s_endpgm
        let words: Vec<u32> = vec![
            (0b1011_1110_1u32 << 23) | (124u32 << 16) | (0x03u32 << 8) | 0xFFu32,
            (1u32 << 12) | 3u32,
            (0b1011_1111_1u32 << 23) | (0x10u32 << 16) | 9u32,
            (0b111110u32 << 26) | (12u32 << 4) | 0xFu32,
            0x0302_0100,
            (0b111110u32 << 26) | (1u32 << 11) | (20u32 << 4) | 0x1u32,
            0x0000_0004,
            (0b1011_1111_1u32 << 23) | (0x01u32 << 16),
        ];
        fs::write(&bin_path, words_to_bytes(&words)).unwrap();

        let (json_out, output, _) =
            format_translate_rdna2(&bin_path, Rdna2StageOverride::Mesh, true)
                .expect("CLI mesh translation must succeed");
        assert_eq!(output.stage, ShaderStage::Compute);
        assert!(output.mesh_output.is_some());
        assert!(
            json_out.contains(
                "\"mesh_output\": {\"max_vertices\": 3, \"max_primitives\": 1, \"vertex_words\": 4}"
            ),
            "JSON:\n{json_out}"
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn end_to_end_pack_rdna2_cli_pack_version_2_and_inspect_pack() {
        let root = make_temp_dir("playgta6_cli_pack_v2");
        let in_dir = root.join("blobs");
        let out_v2 = root.join("out/shaders_v2.wgsl.pack");
        fs::create_dir_all(&in_dir).unwrap();

        let cs_bytes = words_to_bytes(&[
            (0b011_1111_u32 << 25) | (1u32 << 17) | (0x01u32 << 9) | 242,
            (0b1_0111_1111_u32 << 23) | (0x0Au32 << 16),
            (0b1_0111_1111_u32 << 23) | (0x01u32 << 16),
        ]);
        fs::write(in_dir.join("cs_0000000000007001.rdna2"), &cs_bytes).unwrap();
        fs::write(in_dir.join("cs_0000000000007002.rdna2"), &cs_bytes).unwrap();

        let cmd = parse_args(&[
            "pack-rdna2".to_string(),
            in_dir.to_string_lossy().into_owned(),
            "--out".to_string(),
            out_v2.to_string_lossy().into_owned(),
            "--pack-version".to_string(),
            "2".to_string(),
            "--json".to_string(),
        ])
        .expect("pack-rdna2 --pack-version 2 should parse");
        let Command::PackRdna2 {
            input_dir,
            manifest_path,
            out_path,
            pack_version,
            best_effort,
            max_wgsl_bytes,
            max_pack_bytes,
            budget_order,
            priority_hashes_path: _,
            jobs,
            json,
        } = cmd
        else {
            panic!("expected Command::PackRdna2");
        };
        assert_eq!(pack_version, PackVersion::V2);

        let opts = Rdna2PackOptions {
            pack_version,
            best_effort,
            max_wgsl_bytes,
            max_pack_bytes,
            budget_order,
            jobs,
            ..Rdna2PackOptions::default()
        };
        let (json_out, res) = format_and_write_pack_rdna2_sources(
            input_dir.as_deref(),
            manifest_path.as_deref(),
            &out_path,
            &opts,
            json,
        )
        .expect("pack-rdna2 --pack-version 2 should succeed");

        assert_eq!(res.summary.pack_version, PACK_VERSION_V2);
        assert!(json_out.contains("\"format\": \"wgpk_v2\""), "JSON:\n{json_out}");

        let disk_bytes = fs::read(&out_v2).unwrap();
        let decoded = read_pack(&disk_bytes).expect("WGPK v2 on disk must decode");
        assert_eq!(decoded.version, PACK_VERSION_V2);
        assert_eq!(decoded.entries.len(), 2);
        assert_eq!(
            decoded.entries[0].offset,
            (PACK_HEADER_BYTES + 2 * PACK_ENTRY_BYTES_V2) as u64
        );
        run_inspect_pack(&out_v2).expect("inspect-pack must succeed on WGPK v2");

        // Corrupt UTF-8 payload byte in v2 file must be rejected by inspect-pack.
        let mut corrupt = disk_bytes;
        let payload_start = PACK_HEADER_BYTES + 2 * PACK_ENTRY_BYTES_V2;
        corrupt[payload_start] = 0xff;
        let corrupt_path = root.join("out/corrupt_v2.wgsl.pack");
        fs::write(&corrupt_path, &corrupt).unwrap();
        assert!(run_inspect_pack(&corrupt_path).is_err());

        let _ = fs::remove_dir_all(&root);
    }
}
