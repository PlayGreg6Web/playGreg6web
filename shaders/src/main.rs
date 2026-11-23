use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use playgta6_shaders::convert::{
    ConvertedShader, ShaderStage, convert_spirv_bytes, format_shader_hash, format_wgsl_filename,
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
    },
    InspectPack {
        pack_path: PathBuf,
    },
}

fn print_usage() {
    eprintln!("playgta6-shaders (ABI v{SHADER_INDEX_ABI_VERSION})");
    eprintln!("Usage:");
    eprintln!(
        "  playgta6-shaders convert --spirv <dir> --out <dir> [--pack <file.wgsl.pack>] [--report <blockers.md>]"
    );
    eprintln!("  playgta6-shaders inspect-pack <file.wgsl.pack>");
    eprintln!();
    eprintln!(
        "Pipeline: RDNA2 -> SPIR-V (--mesh-compute, --wave64-emulate) -> WGSL (naga) -> .wgsl.pack"
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
            "unknown subcommand '{other}' (expected 'convert' or 'inspect-pack')"
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
    let mut failures: Vec<FailedShaderRecord> = Vec::new();

    for path in &spv_files {
        let rel_path = normalize_rel_path(path, spirv_dir);
        let bytes = fs::read(path)
            .map_err(|err| format!("failed to read SPIR-V file {}: {err}", path.display()))?;

        match convert_spirv_bytes(&bytes) {
            Ok(shader) => {
                converted_file_count += 1;
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
        "Converted {converted_file_count}/{} SPIR-V files ({} unique shaders: vertex={vertex_count}, fragment={fragment_count}, compute={compute_count}); index: {}",
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
        }) => match run_convert(
            &spirv_dir,
            &out_dir,
            pack_path.as_deref(),
            report_path.as_deref(),
        ) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("error: {err}");
                ExitCode::FAILURE
            }
        },
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

    #[test]
    fn cli_parses_convert_and_inspect_pack_arguments() {
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
        ])
        .unwrap();

        assert_eq!(
            cmd,
            Command::Convert {
                spirv_dir: PathBuf::from("in_spv"),
                out_dir: PathBuf::from("out_wgsl"),
                pack_path: Some(PathBuf::from("out_wgsl/all.wgsl.pack")),
                report_path: Some(PathBuf::from("out_wgsl/blockers.md")),
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
        assert!(parse_args(&["inspect-pack".to_string()]).is_err());
    }

    #[test]
    fn end_to_end_convert_pack_and_inspect_temp_dir() {
        let root = make_temp_dir("playgta6_shaders_e2e");
        let spirv_dir = root.join("spirv");
        let out_dir = root.join("out");
        let pack_file = out_dir.join("bundle.wgsl.pack");
        let report_file = out_dir.join("blockers.md");
        fs::create_dir_all(&spirv_dir).unwrap();

        // Write a minimal valid compute SPIR-V module and one corrupt SPIR-V file.
        let mut compute_words: Vec<u32> = vec![
            0x0723_0203, // magic
            0x0001_0300, // SPIR-V 1.3
            0,           // generator
            5,           // bound
            0,           // schema
            (2 << 16) | 17, 1, // OpCapability Shader
            (3 << 16) | 14, 0, 1, // OpMemoryModel Logical GLSL450
            (5 << 16) | 15, 5, 3, u32::from_le_bytes(*b"main"), 0, // OpEntryPoint GLCompute %3 "main"
            (6 << 16) | 16, 3, 17, 1, 1, 1, // OpExecutionMode %3 LocalSize 1 1 1
            (2 << 16) | 19, 1, // %1 = OpTypeVoid
            (3 << 16) | 33, 2, 1, // %2 = OpTypeFunction %1
            (5 << 16) | 54, 1, 3, 0, 2, // %3 = OpFunction %1 None %2
            (2 << 16) | 248, 4, // %4 = OpLabel
            (1 << 16) | 253, // OpReturn
            (1 << 16) | 56, // OpFunctionEnd
        ];
        let mut compute_bytes = Vec::with_capacity(compute_words.len() * 4);
        for w in compute_words.drain(..) {
            compute_bytes.extend_from_slice(&w.to_le_bytes());
        }

        fs::write(spirv_dir.join("cs_test.spv"), &compute_bytes).unwrap();
        fs::write(spirv_dir.join("broken.spv"), b"not-spirv").unwrap();

        run_convert(
            &spirv_dir,
            &out_dir,
            Some(&pack_file),
            Some(&report_file),
        )
        .expect("batch conversion should succeed");

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

        run_inspect_pack(&pack_file).expect("inspect-pack should succeed");

        let _ = fs::remove_dir_all(&root);
    }
}
