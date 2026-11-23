//! `playgta6-recomp lift`: PE64 -> code image (seeded from `.pdata` like `discover`) -> function
//! discovery -> LLVM IR for wasm64, and optionally a wasm64 object.
//!
//! Lifted IR is derived from the input binary's code, so it is only ever written into the
//! directory passed with `--out` (by convention the gitignored `work/`); there is no default
//! output location. The printed summary contains counts, mnemonics and addresses only.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use playgta6_recomp::cfg::Terminator;
use playgta6_recomp::code_image::CodeImage;
use playgta6_recomp::discover::discover;
use playgta6_recomp::eh::EhMetadata;
use playgta6_recomp::lift::compile::{compile_to_wasm64_object, find_llvm_tool};
use playgta6_recomp::lift::lift_module;
use playgta6_recomp::pdata::ExceptionDirectory;
use playgta6_recomp::pe::Pe64;

/// File name of the lifted module inside `--out`.
pub const LL_FILE: &str = "lifted.ll";
/// File name of the wasm64 object inside `--out` (with `--emit-obj`).
pub const OBJ_FILE: &str = "lifted.o";

/// Addresses listed per mnemonic in the unsupported-instruction histogram.
const ADDRESSES_PER_MNEMONIC: usize = 5;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiftArgs {
    pub pe_path: PathBuf,
    pub out_dir: PathBuf,
    /// Extra discovery entry points (VAs) on top of the PE entry point, exports and TLS callbacks.
    pub entries: Vec<u64>,
    pub emit_obj: bool,
}

/// Parses the arguments after `lift`. `Ok(None)` means help was requested.
pub fn parse_lift_args(args: &[String]) -> Result<Option<LiftArgs>, String> {
    let mut pe_path: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut entries = Vec::new();
    let mut emit_obj = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => return Ok(None),
            "--emit-obj" => emit_obj = true,
            "--out" => {
                i += 1;
                match args.get(i) {
                    Some(dir) if !dir.starts_with('-') => out_dir = Some(PathBuf::from(dir)),
                    _ => return Err("--out requires an output <dir> argument".to_string()),
                }
            }
            "--entry" => {
                i += 1;
                let Some(text) = args.get(i) else {
                    return Err("--entry requires a hexadecimal <va> argument".to_string());
                };
                entries.push(parse_va(text)?);
            }
            flag if flag.starts_with('-') => {
                return Err(format!("unknown option '{flag}' for 'lift'"));
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
        "missing --out <dir> (lifted IR is derived from game code; use the gitignored work/)"
            .to_string()
    })?;
    Ok(Some(LiftArgs {
        pe_path,
        out_dir,
        entries,
        emit_obj,
    }))
}

/// Parses a virtual address written in hex, with or without a `0x` prefix.
fn parse_va(text: &str) -> Result<u64, String> {
    let digits = text
        .strip_prefix("0x")
        .or_else(|| text.strip_prefix("0X"))
        .unwrap_or(text);
    u64::from_str_radix(digits, 16)
        .map_err(|_| format!("--entry expects a hexadecimal virtual address, got '{text}'"))
}

/// Runs the pipeline and returns the summary to print.
pub fn run_lift(args: &LiftArgs) -> Result<String, String> {
    let path = &args.pe_path;
    let bytes =
        fs::read(path).map_err(|e| format!("failed to read PE file '{}': {e}", path.display()))?;
    let pe = Pe64::parse(&bytes)
        .map_err(|e| format!("failed to parse PE64 '{}': {e}", path.display()))?;
    // Seed discovery exactly like `discover` does, so both report the same function set.
    let mut image = CodeImage::from_pe(&pe, &bytes);
    let pdata = ExceptionDirectory::from_pe(&pe, &bytes);
    pdata.seed(&mut image);
    EhMetadata::from_pe(&pe, &bytes, &pdata).seed(&mut image);
    image.entry_points.extend(&args.entries);
    image.entry_points.sort_unstable();
    image.entry_points.dedup();
    let analysis = discover(&image);
    let module =
        lift_module(analysis.functions.values()).map_err(|e| format!("lifting failed: {e}"))?;

    fs::create_dir_all(&args.out_dir).map_err(|e| {
        format!(
            "failed to create output directory '{}': {e}",
            args.out_dir.display()
        )
    })?;
    let ll = args.out_dir.join(LL_FILE);
    fs::write(&ll, &module.ir).map_err(|e| format!("failed to write '{}': {e}", ll.display()))?;
    let mut summary = String::new();
    let blocks: Vec<_> = analysis
        .functions
        .values()
        .flat_map(|f| f.blocks.values())
        .collect();
    let instructions: usize = blocks.iter().map(|b| b.instructions.len()).sum();
    summary.push_str(&format!(
        "Lifted {} functions ({} blocks, {} instructions)\n",
        module.functions.len(),
        blocks.len(),
        instructions
    ));
    summary.push_str(&format!("LLVM IR: {}\n", ll.display()));
    summary.push_str(&format!(
        ".pdata: {} functions, {} diagnostics; {} jump tables resolved\n",
        pdata.entries.len(),
        pdata.diagnostics.len(),
        analysis.jump_tables.len()
    ));

    if args.emit_obj {
        let tool = find_llvm_tool().ok_or(
            "--emit-obj needs llc or clang (PLAYGTA6_LLC, PATH, or `rustup component add llvm-tools`)",
        )?;
        let obj = args.out_dir.join(OBJ_FILE);
        compile_to_wasm64_object(&tool, &ll, &obj).map_err(|e| e.to_string())?;
        summary.push_str(&format!(
            "wasm64 object: {} (via {})\n",
            obj.display(),
            tool.path().display()
        ));
    }

    let invalid = blocks
        .iter()
        .filter(|b| b.terminator == Terminator::Invalid)
        .count();
    let traps = blocks
        .iter()
        .filter(|b| b.terminator == Terminator::Trap)
        .count();
    summary.push_str(&format!(
        "Trap blocks: {traps} trap instructions, {invalid} invalid/unmapped code\n"
    ));

    let mut histogram: BTreeMap<&str, Vec<u64>> = BTreeMap::new();
    for func in &module.functions {
        for u in &func.unsupported {
            histogram.entry(&u.mnemonic).or_default().push(u.address);
        }
    }
    let total: usize = histogram.values().map(Vec::len).sum();
    summary.push_str(&format!(
        "Unsupported instructions: {total} ({} distinct mnemonics)\n",
        histogram.len()
    ));
    let mut rows: Vec<(&str, Vec<u64>)> = histogram.into_iter().collect();
    rows.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(b.0)));
    for (mnemonic, mut addresses) in rows {
        addresses.sort_unstable();
        addresses.dedup();
        let shown: Vec<String> = addresses
            .iter()
            .take(ADDRESSES_PER_MNEMONIC)
            .map(|va| format!("{va:#x}"))
            .collect();
        let more = if addresses.len() > ADDRESSES_PER_MNEMONIC {
            ", ..."
        } else {
            ""
        };
        summary.push_str(&format!(
            "  {mnemonic:<12} {:>6}  {}{more}\n",
            addresses.len(),
            shown.join(", ")
        ));
    }
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use playgta6_recomp::pe::test_util::build_synthetic_pe64;

    fn vec_s(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn parses_lift_arguments() {
        assert_eq!(
            parse_lift_args(&vec_s(&[
                "game.exe",
                "--out",
                "work/lift",
                "--entry",
                "0x140001000",
                "--entry",
                "140002000",
                "--emit-obj"
            ])),
            Ok(Some(LiftArgs {
                pe_path: PathBuf::from("game.exe"),
                out_dir: PathBuf::from("work/lift"),
                entries: vec![0x1_4000_1000, 0x1_4000_2000],
                emit_obj: true,
            }))
        );
        assert_eq!(parse_lift_args(&vec_s(&["--help"])), Ok(None));
        assert!(
            parse_lift_args(&vec_s(&["game.exe"])).is_err(),
            "--out is required"
        );
        assert!(parse_lift_args(&vec_s(&["--out", "work"])).is_err());
        assert!(parse_lift_args(&vec_s(&["a.exe", "--out"])).is_err());
        assert!(parse_lift_args(&vec_s(&["a.exe", "--out", "w", "--entry", "zz"])).is_err());
        assert!(parse_lift_args(&vec_s(&["a.exe", "--out", "w", "--bogus"])).is_err());
        assert!(parse_lift_args(&vec_s(&["a.exe", "b.exe", "--out", "w"])).is_err());
    }

    #[test]
    fn lifts_a_synthetic_pe_into_the_output_directory() {
        let temp_dir = std::env::temp_dir().join(format!(
            "playgta6_recomp_lift_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&temp_dir).expect("create temp dir");
        let exe_path = temp_dir.join("SyntheticGame.exe");
        fs::write(&exe_path, build_synthetic_pe64(false)).expect("write synthetic pe");

        let out_dir = temp_dir.join("work").join("lift");
        let summary = run_lift(&LiftArgs {
            pe_path: exe_path,
            out_dir: out_dir.clone(),
            entries: Vec::new(),
            emit_obj: false,
        })
        .expect("run_lift should succeed");

        let ir = fs::read_to_string(out_dir.join(LL_FILE)).expect("read lifted IR");
        assert!(ir.contains("target triple = \"wasm64-unknown-unknown\""));
        assert!(ir.contains("define void @sub_140001010(ptr noalias %state)"));
        assert_eq!(ir.matches("\ndefine void @sub_").count(), 4);
        assert!(summary.starts_with("Lifted 4 functions"), "{summary}");
        assert!(summary.contains("Unsupported instructions: "), "{summary}");
        assert!(
            summary.contains(".pdata: "),
            "seeded like discover: {summary}"
        );
        assert!(
            !summary.contains("SyntheticGame"),
            "no file names in the summary"
        );
        let entries: Vec<_> = fs::read_dir(&out_dir)
            .expect("list output")
            .filter_map(|e| e.ok().map(|e| e.file_name()))
            .collect();
        assert_eq!(
            entries,
            [LL_FILE],
            "only the IR is written without --emit-obj"
        );

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
