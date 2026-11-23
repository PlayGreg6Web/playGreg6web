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
use playgta6_shaders::pack::{
    BlockerReportInput, FailedShaderRecord, PACK_ENTRY_BYTES, PACK_HEADER_BYTES, PackEntry,
    SHADER_INDEX_ABI_VERSION, ShaderIndexEntry, read_pack, render_blocker_report_md,
    render_shader_index_json, write_pack_with_layout,
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
    },
    Audit {
        spirv_dir: PathBuf,
        report_path: Option<PathBuf>,
        no_sanitize: bool,
    },
    InspectPack {
        pack_path: PathBuf,
    },
}

fn print_usage() {
    eprintln!("playgta6-shaders (ABI v{SHADER_INDEX_ABI_VERSION})");
    eprintln!("Usage:");
    eprintln!(
        "  playgta6-shaders convert --spirv <dir> --out <dir> [--pack <file.wgsl.pack>] [--report <blockers.md>] [--no-sanitize]"
    );
    eprintln!(
        "  playgta6-shaders audit --spirv <dir> [--report <audit.md>] [--no-sanitize]"
    );
    eprintln!("  playgta6-shaders inspect-pack <file.wgsl.pack>");
    eprintln!();
    eprintln!(
        "Pipeline: RDNA2 -> SPIR-V (--mesh-compute, --wave64-emulate) -> compat sanitize -> WGSL (naga) -> .wgsl.pack"
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
        other => Err(format!(
            "unknown subcommand '{other}' (expected 'convert', 'audit', or 'inspect-pack')"
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
        "Converted {converted_file_count}/{} SPIR-V files ({} unique shaders: vertex={vertex_count}, fragment={fragment_count}, compute={compute_count}; sanitized={sanitized_file_count} files / {rewritten_instruction_count} instructions); index: {}",
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
        }) => {
            let options = if no_sanitize {
                ConvertOptions::without_sanitization()
            } else {
                ConvertOptions::default()
            };
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
            }
        );

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

        let _ = fs::remove_dir_all(&root);
    }
}
