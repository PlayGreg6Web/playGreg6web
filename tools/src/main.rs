mod awc;
mod rpf9;
mod scan;
mod xvd;

use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use rpf9::{
    BootsetEntry, DEFAULT_BOOTSET_HEAD_BYTES, inspect_rpf_file, json_escape, render_bootset_json,
    write_bootset,
};
use awc::{
    DEFAULT_MAX_LISTED_STREAMS, dir_format_human, dir_to_json, inspect_awc_dir, inspect_awc_file,
};
use scan::{is_file_complete, resolve_env_key, resolve_game_dir_from_env, scan_directory};
use xvd::inspect_xvd_file;

const ABI_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
struct FileEntry {
    rel_path: String,
    size: u64,
}

fn collect_files(
    root: &Path,
    dir: &Path,
    skip_incomplete: bool,
    out: &mut Vec<FileEntry>,
    skipped: &mut usize,
) -> io::Result<()> {
    let entries = match fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(_) if skip_incomplete => {
            *skipped += 1;
            return Ok(());
        }
        Err(err) => return Err(err),
    };

    for entry_res in entries {
        let entry = match entry_res {
            Ok(e) => e,
            Err(_) if skip_incomplete => {
                *skipped += 1;
                continue;
            }
            Err(err) => return Err(err),
        };
        let path = entry.path();
        let meta = match entry.metadata() {
            Ok(m) => m,
            Err(_) if skip_incomplete => {
                *skipped += 1;
                continue;
            }
            Err(err) => return Err(err),
        };
        if meta.is_dir() {
            collect_files(root, &path, skip_incomplete, out, skipped)?;
        } else if meta.is_file() {
            let size = meta.len();
            if skip_incomplete && !is_file_complete(&path, size) {
                *skipped += 1;
                continue;
            }
            let rel = path
                .strip_prefix(root)
                .map_err(io::Error::other)?
                .to_string_lossy()
                .replace('\\', "/");
            out.push(FileEntry {
                rel_path: rel,
                size,
            });
        }
    }
    Ok(())
}

fn render_manifest_json(entries: &[FileEntry]) -> String {
    let total_bytes: u64 = entries.iter().map(|e| e.size).sum();
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"abi_version\": {ABI_VERSION},\n"));
    out.push_str(&format!("  \"file_count\": {},\n", entries.len()));
    out.push_str(&format!("  \"total_bytes\": {},\n", total_bytes));
    out.push_str("  \"files\": [\n");
    for (idx, e) in entries.iter().enumerate() {
        let comma = if idx + 1 == entries.len() { "" } else { "," };
        out.push_str(&format!(
            "    {{\"path\": \"{}\", \"size\": {}}}{}\n",
            json_escape(&e.rel_path),
            e.size,
            comma
        ));
    }
    out.push_str("  ]\n");
    out.push_str("}\n");
    out
}

fn build_manifest_with_options(
    game_dir: &Path,
    out_file: &Path,
    skip_incomplete: bool,
) -> io::Result<(usize, usize)> {
    let root_meta = fs::metadata(game_dir)?;
    if !root_meta.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "game directory path is not a directory",
        ));
    }

    let mut entries = Vec::new();
    let mut skipped = 0usize;
    collect_files(
        game_dir,
        game_dir,
        skip_incomplete,
        &mut entries,
        &mut skipped,
    )?;
    entries.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
    if let Some(parent) = out_file.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let json = render_manifest_json(&entries);
    fs::write(out_file, json)?;
    Ok((entries.len(), skipped))
}

#[cfg(test)]
fn build_manifest(game_dir: &Path, out_file: &Path) -> io::Result<usize> {
    build_manifest_with_options(game_dir, out_file, false).map(|(count, _)| count)
}

fn is_bootset_candidate(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("rpf") || ext.eq_ignore_ascii_case("config"))
}

fn collect_complete_bootset_entries(
    root: &Path,
    dir: &Path,
    max_header_bytes: u64,
    out: &mut Vec<BootsetEntry>,
    skipped: &mut usize,
) -> io::Result<()> {
    let entries = match fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(_) => {
            *skipped += 1;
            return Ok(());
        }
    };

    for entry_res in entries {
        let Ok(entry) = entry_res else {
            *skipped += 1;
            continue;
        };
        let path = entry.path();
        let Ok(meta) = entry.metadata() else {
            *skipped += 1;
            continue;
        };
        if meta.is_dir() {
            collect_complete_bootset_entries(root, &path, max_header_bytes, out, skipped)?;
        } else if meta.is_file() && is_bootset_candidate(&path) {
            let size = meta.len();
            if !is_file_complete(&path, size) {
                *skipped += 1;
                continue;
            }
            let clamped_len = size.min(max_header_bytes);
            if clamped_len == 0 {
                *skipped += 1;
                continue;
            }
            let rel = path
                .strip_prefix(root)
                .map_err(io::Error::other)?
                .to_string_lossy()
                .replace('\\', "/");
            out.push(BootsetEntry {
                path: rel,
                offset: 0,
                length: clamped_len,
            });
        }
    }
    Ok(())
}

fn build_bootset_with_options(
    game_dir: &Path,
    out_file: &Path,
    head_bytes: u64,
    skip_incomplete: bool,
) -> io::Result<(usize, usize)> {
    if !skip_incomplete {
        let count = write_bootset(game_dir, out_file, head_bytes)?;
        return Ok((count, 0));
    }

    let root_meta = fs::metadata(game_dir)?;
    if !root_meta.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "game directory path is not a directory",
        ));
    }

    let mut entries = Vec::new();
    let mut skipped = 0usize;
    collect_complete_bootset_entries(game_dir, game_dir, head_bytes, &mut entries, &mut skipped)?;
    entries.sort_by(|a, b| a.path.cmp(&b.path));
    if let Some(parent) = out_file.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let json = render_bootset_json(&entries);
    fs::write(out_file, json)?;
    Ok((entries.len(), skipped))
}

fn read_dotenv_key(key: &str) -> Option<String> {
    resolve_env_key(key)
}

fn print_usage() {
    eprintln!("Usage:");
    eprintln!(
        "  playgta6-manifest [manifest] [--game <game-dir>] [--out <local-data/data-manifest.json>] [--skip-incomplete]"
    );
    eprintln!(
        "  playgta6-manifest bootset [--game <game-dir>] [--out <local-data/bootset.json>] [--head-bytes {DEFAULT_BOOTSET_HEAD_BYTES}] [--skip-incomplete]"
    );
    eprintln!("  playgta6-manifest scan (--dir <game-dir> | --from-env) [--json]");
    eprintln!("  playgta6-manifest inspect-rpf <file.rpf>");
    eprintln!("  playgta6-manifest xvd <file.xvd|file.xvc> [--json]");
    eprintln!(
        "  playgta6-manifest awc (<file.awc> | --dir <game-dir> | --from-env) [--json] [--max-streams {DEFAULT_MAX_LISTED_STREAMS}]"
    );
    eprintln!("Falls back to GTA6_GAME_DIR and GTA6_DATA_DIR from env or .env.");
}

#[derive(Debug, PartialEq, Eq)]
enum Command {
    Help,
    Manifest {
        game_dir: Option<PathBuf>,
        out_file: Option<PathBuf>,
        skip_incomplete: bool,
    },
    Bootset {
        game_dir: Option<PathBuf>,
        out_file: Option<PathBuf>,
        head_bytes: u64,
        skip_incomplete: bool,
    },
    InspectRpf {
        rpf_path: PathBuf,
    },
    Xvd {
        path: PathBuf,
        json: bool,
    },
    Scan {
        dir: Option<PathBuf>,
        from_env: bool,
        json: bool,
    },
    Awc {
        path: Option<PathBuf>,
        dir: Option<PathBuf>,
        from_env: bool,
        json: bool,
        max_streams: usize,
    },
}

fn parse_manifest_flags(args: &[String], start_idx: usize) -> Result<Command, String> {
    let mut game_dir: Option<PathBuf> = None;
    let mut out_file: Option<PathBuf> = None;
    let mut skip_incomplete = false;

    let mut i = start_idx;
    while i < args.len() {
        match args[i].as_str() {
            "--game" if i + 1 < args.len() => {
                game_dir = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--out" if i + 1 < args.len() => {
                out_file = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--skip-incomplete" => {
                skip_incomplete = true;
                i += 1;
            }
            other => {
                return Err(format!("Unknown argument: {other}"));
            }
        }
    }
    Ok(Command::Manifest {
        game_dir,
        out_file,
        skip_incomplete,
    })
}

fn parse_args(args: &[String]) -> Result<Command, String> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        return Ok(Command::Help);
    }

    match args.first().map(String::as_str) {
        Some("manifest") => parse_manifest_flags(args, 1),
        Some("bootset") => {
            let mut game_dir: Option<PathBuf> = None;
            let mut out_file: Option<PathBuf> = None;
            let mut head_bytes: u64 = DEFAULT_BOOTSET_HEAD_BYTES;
            let mut skip_incomplete = false;

            let mut i = 1;
            while i < args.len() {
                match args[i].as_str() {
                    "--game" if i + 1 < args.len() => {
                        game_dir = Some(PathBuf::from(&args[i + 1]));
                        i += 2;
                    }
                    "--out" if i + 1 < args.len() => {
                        out_file = Some(PathBuf::from(&args[i + 1]));
                        i += 2;
                    }
                    "--head-bytes" if i + 1 < args.len() => {
                        head_bytes = args[i + 1]
                            .parse::<u64>()
                            .map_err(|_| format!("Invalid --head-bytes value: {}", args[i + 1]))?;
                        i += 2;
                    }
                    "--skip-incomplete" => {
                        skip_incomplete = true;
                        i += 1;
                    }
                    other => {
                        return Err(format!("Unknown argument for bootset: {other}"));
                    }
                }
            }
            Ok(Command::Bootset {
                game_dir,
                out_file,
                head_bytes,
                skip_incomplete,
            })
        }
        Some("scan") => {
            let mut dir: Option<PathBuf> = None;
            let mut from_env = false;
            let mut json = false;

            let mut i = 1;
            while i < args.len() {
                match args[i].as_str() {
                    "--dir" => {
                        if i + 1 >= args.len() {
                            return Err("Missing value for --dir".to_string());
                        }
                        dir = Some(PathBuf::from(&args[i + 1]));
                        i += 2;
                    }
                    "--from-env" => {
                        from_env = true;
                        i += 1;
                    }
                    "--json" => {
                        json = true;
                        i += 1;
                    }
                    other => {
                        return Err(format!("Unknown argument for scan: {other}"));
                    }
                }
            }
            if dir.is_some() && from_env {
                return Err("Cannot specify both --dir and --from-env for scan".to_string());
            }
            Ok(Command::Scan {
                dir,
                from_env,
                json,
            })
        }
        Some("inspect-rpf") => {
            if args.len() != 2 {
                return Err("Usage: playgta6-manifest inspect-rpf <file.rpf>".to_string());
            }
            Ok(Command::InspectRpf {
                rpf_path: PathBuf::from(&args[1]),
            })
        }
        Some("xvd") => {
            let mut path: Option<PathBuf> = None;
            let mut json = false;

            for arg in &args[1..] {
                match arg.as_str() {
                    "--json" => json = true,
                    other if other.starts_with('-') => {
                        return Err(format!("Unknown argument for xvd: {other}"));
                    }
                    other => {
                        if path.is_some() {
                            return Err(format!("Unexpected extra argument for xvd: {other}"));
                        }
                        path = Some(PathBuf::from(other));
                    }
                }
            }

            let Some(path) = path else {
                return Err("Usage: playgta6-manifest xvd <file.xvd|file.xvc> [--json]".to_string());
            };
            Ok(Command::Xvd { path, json })
        }
        Some("awc") => parse_awc_args(&args[1..]),
        _ => parse_manifest_flags(args, 0),
    }
}

fn parse_awc_args(args: &[String]) -> Result<Command, String> {
    const USAGE: &str =
        "Usage: playgta6-manifest awc (<file.awc> | --dir <game-dir> | --from-env) [--json] [--max-streams N]";
    let mut path: Option<PathBuf> = None;
    let mut dir: Option<PathBuf> = None;
    let mut from_env = false;
    let mut json = false;
    let mut max_streams = DEFAULT_MAX_LISTED_STREAMS;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--json" => json = true,
            "--from-env" => from_env = true,
            "--dir" => {
                i += 1;
                let v = args.get(i).ok_or("Missing value for --dir")?;
                dir = Some(PathBuf::from(v));
            }
            "--max-streams" => {
                i += 1;
                let v = args.get(i).ok_or("Missing value for --max-streams")?;
                max_streams = v
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid --max-streams value: {v}"))?;
            }
            other if other.starts_with('-') => {
                return Err(format!("Unknown argument for awc: {other}"));
            }
            other => {
                if path.is_some() {
                    return Err(format!("Unexpected extra argument for awc: {other}"));
                }
                path = Some(PathBuf::from(other));
            }
        }
        i += 1;
    }
    let modes = usize::from(path.is_some()) + usize::from(dir.is_some()) + usize::from(from_env);
    if modes != 1 {
        return Err(USAGE.to_string());
    }
    Ok(Command::Awc {
        path,
        dir,
        from_env,
        json,
        max_streams,
    })
}

fn default_data_path(file_name: &str) -> PathBuf {
    let data_dir = read_dotenv_key("GTA6_DATA_DIR").unwrap_or_else(|| "local-data".to_string());
    PathBuf::from(data_dir).join(file_name)
}

fn resolve_game_dir(cli_game_dir: Option<PathBuf>) -> Option<PathBuf> {
    cli_game_dir.or_else(resolve_game_dir_from_env)
}

fn run_awc(path: Option<PathBuf>, dir: Option<PathBuf>, json: bool, max_streams: usize) -> ExitCode {
    if let Some(path) = path {
        return match inspect_awc_file(&path) {
            Ok(report) => {
                if json {
                    print!("{}", report.to_json(max_streams));
                } else {
                    print!("{}", report.format_human(max_streams));
                }
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("AWC inspection failed: {err}");
                ExitCode::FAILURE
            }
        };
    }
    let Some(target_dir) = resolve_game_dir(dir) else {
        eprintln!(
            "Error: --dir <path> or --from-env (with GTA6_GAME_DIR in env or .env) is required."
        );
        return ExitCode::FAILURE;
    };
    match inspect_awc_dir(&target_dir) {
        Ok(results) => {
            if json {
                print!("{}", dir_to_json(&results, max_streams));
            } else {
                print!("{}", dir_format_human(&results));
            }
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("AWC directory scan failed: {err}");
            ExitCode::FAILURE
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let command = match parse_args(&args) {
        Ok(cmd) => cmd,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::FAILURE;
        }
    };

    match command {
        Command::Help => {
            print_usage();
            ExitCode::SUCCESS
        }
        Command::Manifest {
            game_dir,
            out_file,
            skip_incomplete,
        } => {
            let Some(game_dir) = resolve_game_dir(game_dir) else {
                eprintln!("Error: --game <dir> or GTA6_GAME_DIR (in .env) is required.");
                return ExitCode::FAILURE;
            };
            let out_file = out_file.unwrap_or_else(|| default_data_path("data-manifest.json"));
            match build_manifest_with_options(&game_dir, &out_file, skip_incomplete) {
                Ok((count, skipped)) => {
                    if skip_incomplete && skipped > 0 {
                        println!(
                            "Wrote {} entries to {} (skipped {} incomplete)",
                            count,
                            out_file.display(),
                            skipped
                        );
                    } else {
                        println!("Wrote {} entries to {}", count, out_file.display());
                    }
                    ExitCode::SUCCESS
                }
                Err(err) => {
                    eprintln!("Manifest generation failed: {err}");
                    ExitCode::FAILURE
                }
            }
        }
        Command::Bootset {
            game_dir,
            out_file,
            head_bytes,
            skip_incomplete,
        } => {
            let Some(game_dir) = resolve_game_dir(game_dir) else {
                eprintln!("Error: --game <dir> or GTA6_GAME_DIR (in .env) is required.");
                return ExitCode::FAILURE;
            };
            let out_file = out_file.unwrap_or_else(|| default_data_path("bootset.json"));
            match build_bootset_with_options(&game_dir, &out_file, head_bytes, skip_incomplete) {
                Ok((count, skipped)) => {
                    if skip_incomplete && skipped > 0 {
                        println!(
                            "Wrote {} bootset entries to {} (skipped {} incomplete)",
                            count,
                            out_file.display(),
                            skipped
                        );
                    } else {
                        println!("Wrote {} bootset entries to {}", count, out_file.display());
                    }
                    ExitCode::SUCCESS
                }
                Err(err) => {
                    eprintln!("Bootset generation failed: {err}");
                    ExitCode::FAILURE
                }
            }
        }
        Command::Scan {
            dir,
            from_env: _,
            json,
        } => {
            let Some(target_dir) = resolve_game_dir(dir) else {
                eprintln!(
                    "Error: --dir <path> or --from-env (with GTA6_GAME_DIR in env or .env) is required."
                );
                return ExitCode::FAILURE;
            };
            match scan_directory(&target_dir) {
                Ok(report) => {
                    if json {
                        print!("{}", report.to_json());
                    } else {
                        print!("{}", report.format_human());
                    }
                    ExitCode::SUCCESS
                }
                Err(err) => {
                    eprintln!("Scan failed: {err}");
                    ExitCode::FAILURE
                }
            }
        }
        Command::InspectRpf { rpf_path } => match inspect_rpf_file(&rpf_path) {
            Ok(header) => {
                println!("{}", header.format_summary());
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("RPF inspection failed: {err}");
                ExitCode::FAILURE
            }
        },
        Command::Awc {
            path,
            dir,
            from_env: _,
            json,
            max_streams,
        } => run_awc(path, dir, json, max_streams),
        Command::Xvd { path, json } => match inspect_xvd_file(&path) {
            Ok(report) => {
                if json {
                    print!("{}", report.to_json());
                } else {
                    print!("{}", report.format_human());
                }
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("XVD inspection failed: {err}");
                ExitCode::FAILURE
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_temp_dir(label: &str) -> PathBuf {
        let n = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = env::temp_dir().join(format!(
            "playgta6_manifest_{label}_{}_{}",
            std::process::id(),
            n
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn make_synthetic_pe64() -> Vec<u8> {
        let mut buf = vec![0u8; 0x120];
        buf[0..2].copy_from_slice(b"MZ");
        buf[0x3C..0x40].copy_from_slice(&0x80u32.to_le_bytes());
        buf[0x80..0x84].copy_from_slice(b"PE\0\0");
        buf[0x84..0x86].copy_from_slice(&0x8664u16.to_le_bytes());
        buf[0x94..0x96].copy_from_slice(&0x00F0u16.to_le_bytes());
        buf[0x98..0x9A].copy_from_slice(&0x020Bu16.to_le_bytes());
        buf
    }

    fn make_synthetic_rpf9() -> Vec<u8> {
        let mut buf = vec![0u8; 64];
        buf[0..4].copy_from_slice(b"RPF9");
        buf
    }

    #[test]
    fn builds_sorted_manifest_json() {
        let base = unique_temp_dir("sorted");
        let game = base.join("game");
        let sub = game.join("sub");
        fs::create_dir_all(&sub).unwrap();
        fs::write(game.join("b.bin"), b"12345").unwrap();
        fs::write(sub.join("a.bin"), b"xy").unwrap();

        let out_file = base.join("out/data-manifest.json");
        let count = build_manifest(&game, &out_file).unwrap();
        assert_eq!(count, 2);

        let content = fs::read_to_string(&out_file).unwrap();
        assert!(content.contains("\"abi_version\": 1"));
        assert!(content.contains("\"file_count\": 2"));
        assert!(content.contains("\"total_bytes\": 7"));
        assert!(content.contains("\"path\": \"b.bin\""));
        assert!(content.contains("\"path\": \"sub/a.bin\""));

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn skips_incomplete_files_for_manifest_and_bootset() {
        let base = unique_temp_dir("skip_incomplete");
        let game = base.join("game");
        let sub = game.join("x64");
        fs::create_dir_all(&sub).unwrap();

        // Completed files
        fs::write(game.join("GTA6.exe"), make_synthetic_pe64()).unwrap();
        fs::write(game.join("MicrosoftGame.Config"), b"<Game/>").unwrap();
        fs::write(sub.join("levels.rpf"), make_synthetic_rpf9()).unwrap();

        // Incomplete files
        fs::write(sub.join("audio.rpf.!qB"), b"partial").unwrap();
        fs::write(sub.join("world.rpf.part"), b"partial").unwrap();
        fs::write(sub.join("sparse.rpf"), vec![0u8; 2048]).unwrap();
        fs::write(game.join("sparse.exe"), vec![0u8; 512]).unwrap();
        fs::write(game.join("empty.meta"), b"").unwrap();

        let manifest_out = base.join("out/data-manifest.json");
        let (manifest_count, manifest_skipped) =
            build_manifest_with_options(&game, &manifest_out, true).unwrap();
        assert_eq!(manifest_count, 3);
        assert_eq!(manifest_skipped, 5);

        let manifest_json = fs::read_to_string(&manifest_out).unwrap();
        assert!(manifest_json.contains("\"path\": \"GTA6.exe\""));
        assert!(manifest_json.contains("\"path\": \"MicrosoftGame.Config\""));
        assert!(manifest_json.contains("\"path\": \"x64/levels.rpf\""));
        assert!(!manifest_json.contains("audio.rpf.!qB"));
        assert!(!manifest_json.contains("world.rpf.part"));
        assert!(!manifest_json.contains("sparse.rpf"));
        assert!(!manifest_json.contains("sparse.exe"));
        assert!(!manifest_json.contains("empty.meta"));

        let bootset_out = base.join("out/bootset.json");
        let (bootset_count, bootset_skipped) =
            build_bootset_with_options(&game, &bootset_out, DEFAULT_BOOTSET_HEAD_BYTES, true)
                .unwrap();
        assert_eq!(bootset_count, 2);
        assert_eq!(bootset_skipped, 1); // sparse.rpf skipped

        let bootset_json = fs::read_to_string(&bootset_out).unwrap();
        assert!(bootset_json.contains("\"path\": \"MicrosoftGame.Config\""));
        assert!(bootset_json.contains("\"path\": \"x64/levels.rpf\""));
        assert!(!bootset_json.contains("sparse.rpf"));
        assert!(!bootset_json.contains("audio.rpf.!qB"));

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn parses_cli_subcommands_and_default_manifest() {
        let default_args: Vec<String> = vec!["--game", "C:/game", "--out", "out.json"]
            .into_iter()
            .map(String::from)
            .collect();
        assert_eq!(
            parse_args(&default_args).unwrap(),
            Command::Manifest {
                game_dir: Some(PathBuf::from("C:/game")),
                out_file: Some(PathBuf::from("out.json")),
                skip_incomplete: false,
            }
        );

        let manifest_skip_args: Vec<String> =
            vec!["manifest", "--game", "C:/game", "--skip-incomplete"]
                .into_iter()
                .map(String::from)
                .collect();
        assert_eq!(
            parse_args(&manifest_skip_args).unwrap(),
            Command::Manifest {
                game_dir: Some(PathBuf::from("C:/game")),
                out_file: None,
                skip_incomplete: true,
            }
        );

        let bootset_args: Vec<String> = vec![
            "bootset",
            "--game",
            "C:/game",
            "--out",
            "local-data/bootset.json",
            "--head-bytes",
            "32768",
            "--skip-incomplete",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(
            parse_args(&bootset_args).unwrap(),
            Command::Bootset {
                game_dir: Some(PathBuf::from("C:/game")),
                out_file: Some(PathBuf::from("local-data/bootset.json")),
                head_bytes: 32768,
                skip_incomplete: true,
            }
        );

        let scan_dir_args: Vec<String> = vec!["scan", "--dir", "C:/game", "--json"]
            .into_iter()
            .map(String::from)
            .collect();
        assert_eq!(
            parse_args(&scan_dir_args).unwrap(),
            Command::Scan {
                dir: Some(PathBuf::from("C:/game")),
                from_env: false,
                json: true,
            }
        );

        let scan_env_args: Vec<String> = vec!["scan", "--from-env"]
            .into_iter()
            .map(String::from)
            .collect();
        assert_eq!(
            parse_args(&scan_env_args).unwrap(),
            Command::Scan {
                dir: None,
                from_env: true,
                json: false,
            }
        );

        let conflicting_scan: Vec<String> = vec!["scan", "--dir", "C:/game", "--from-env"]
            .into_iter()
            .map(String::from)
            .collect();
        assert!(parse_args(&conflicting_scan).is_err());

        let inspect_args: Vec<String> = vec!["inspect-rpf", "levels.rpf"]
            .into_iter()
            .map(String::from)
            .collect();
        assert_eq!(
            parse_args(&inspect_args).unwrap(),
            Command::InspectRpf {
                rpf_path: PathBuf::from("levels.rpf"),
            }
        );

        let bad_inspect: Vec<String> = vec!["inspect-rpf".to_string()];
        assert!(parse_args(&bad_inspect).is_err());

        let xvd_args: Vec<String> = vec!["xvd", "package.xvc"]
            .into_iter()
            .map(String::from)
            .collect();
        assert_eq!(
            parse_args(&xvd_args).unwrap(),
            Command::Xvd {
                path: PathBuf::from("package.xvc"),
                json: false,
            }
        );

        let xvd_json_args: Vec<String> = vec!["xvd", "package.xvc", "--json"]
            .into_iter()
            .map(String::from)
            .collect();
        assert_eq!(
            parse_args(&xvd_json_args).unwrap(),
            Command::Xvd {
                path: PathBuf::from("package.xvc"),
                json: true,
            }
        );

        let xvd_json_first_args: Vec<String> = vec!["xvd", "--json", "container.xvd"]
            .into_iter()
            .map(String::from)
            .collect();
        assert_eq!(
            parse_args(&xvd_json_first_args).unwrap(),
            Command::Xvd {
                path: PathBuf::from("container.xvd"),
                json: true,
            }
        );

        let sv = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<String>>();
        assert_eq!(
            parse_args(&sv(&["awc", "a.awc", "--json", "--max-streams", "5"])).unwrap(),
            Command::Awc {
                path: Some(PathBuf::from("a.awc")),
                dir: None,
                from_env: false,
                json: true,
                max_streams: 5,
            }
        );
        assert_eq!(
            parse_args(&sv(&["awc", "--from-env"])).unwrap(),
            Command::Awc {
                path: None,
                dir: None,
                from_env: true,
                json: false,
                max_streams: DEFAULT_MAX_LISTED_STREAMS,
            }
        );
        assert_eq!(
            parse_args(&sv(&["awc", "--dir", "C:/game"])).unwrap(),
            Command::Awc {
                path: None,
                dir: Some(PathBuf::from("C:/game")),
                from_env: false,
                json: false,
                max_streams: DEFAULT_MAX_LISTED_STREAMS,
            }
        );
        for bad in [
            vec!["awc"],
            vec!["awc", "a.awc", "b.awc"],
            vec!["awc", "a.awc", "--dir", "x"],
            vec!["awc", "--dir"],
            vec!["awc", "a.awc", "--max-streams"],
            vec!["awc", "a.awc", "--max-streams", "x"],
            vec!["awc", "a.awc", "--bogus"],
            vec!["awc", "--dir", "x", "--from-env"],
        ] {
            assert!(parse_args(&sv(&bad)).is_err(), "{bad:?}");
        }

        let bad_xvd_missing: Vec<String> = vec!["xvd".to_string()];
        assert!(parse_args(&bad_xvd_missing).is_err());

        let bad_xvd_flag: Vec<String> = vec!["xvd", "package.xvc", "--unknown"]
            .into_iter()
            .map(String::from)
            .collect();
        assert!(parse_args(&bad_xvd_flag).is_err());

        let bad_xvd_extra: Vec<String> = vec!["xvd", "a.xvc", "b.xvc"]
            .into_iter()
            .map(String::from)
            .collect();
        assert!(parse_args(&bad_xvd_extra).is_err());
    }
}
