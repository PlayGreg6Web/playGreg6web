mod rpf9;

use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use rpf9::{DEFAULT_BOOTSET_HEAD_BYTES, inspect_rpf_file, json_escape, write_bootset};

const ABI_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
struct FileEntry {
    rel_path: String,
    size: u64,
}

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<FileEntry>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let meta = entry.metadata()?;
        if meta.is_dir() {
            collect_files(root, &path, out)?;
        } else if meta.is_file() {
            let rel = path
                .strip_prefix(root)
                .map_err(io::Error::other)?
                .to_string_lossy()
                .replace('\\', "/");
            out.push(FileEntry {
                rel_path: rel,
                size: meta.len(),
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

fn build_manifest(game_dir: &Path, out_file: &Path) -> io::Result<usize> {
    let mut entries = Vec::new();
    collect_files(game_dir, game_dir, &mut entries)?;
    entries.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
    if let Some(parent) = out_file.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let json = render_manifest_json(&entries);
    fs::write(out_file, json)?;
    Ok(entries.len())
}

fn read_dotenv_key(key: &str) -> Option<String> {
    if let Ok(val) = env::var(key)
        && !val.trim().is_empty()
    {
        return Some(val);
    }
    let text = fs::read_to_string(".env").ok()?;
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = line.split_once('=')
            && k.trim() == key
        {
            let cleaned = v.trim().trim_matches('"').trim_matches('\'');
            if !cleaned.is_empty() {
                return Some(cleaned.to_string());
            }
        }
    }
    None
}

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  playgta6-manifest [--game <game-dir>] [--out <local-data/data-manifest.json>]");
    eprintln!(
        "  playgta6-manifest bootset [--game <game-dir>] [--out <local-data/bootset.json>] [--head-bytes {DEFAULT_BOOTSET_HEAD_BYTES}]"
    );
    eprintln!("  playgta6-manifest inspect-rpf <file.rpf>");
    eprintln!("Falls back to GTA6_GAME_DIR and GTA6_DATA_DIR from env or .env.");
}

#[derive(Debug, PartialEq, Eq)]
enum Command {
    Help,
    Manifest {
        game_dir: Option<PathBuf>,
        out_file: Option<PathBuf>,
    },
    Bootset {
        game_dir: Option<PathBuf>,
        out_file: Option<PathBuf>,
        head_bytes: u64,
    },
    InspectRpf {
        rpf_path: PathBuf,
    },
}

fn parse_args(args: &[String]) -> Result<Command, String> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        return Ok(Command::Help);
    }

    match args.first().map(String::as_str) {
        Some("bootset") => {
            let mut game_dir: Option<PathBuf> = None;
            let mut out_file: Option<PathBuf> = None;
            let mut head_bytes: u64 = DEFAULT_BOOTSET_HEAD_BYTES;

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
                        head_bytes = args[i + 1].parse::<u64>().map_err(|_| {
                            format!("Invalid --head-bytes value: {}", args[i + 1])
                        })?;
                        i += 2;
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
        _ => {
            let mut game_dir: Option<PathBuf> = None;
            let mut out_file: Option<PathBuf> = None;

            let mut i = 0;
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
                    other => {
                        return Err(format!("Unknown argument: {other}"));
                    }
                }
            }
            Ok(Command::Manifest { game_dir, out_file })
        }
    }
}

fn default_data_path(file_name: &str) -> PathBuf {
    let data_dir = read_dotenv_key("GTA6_DATA_DIR").unwrap_or_else(|| "local-data".to_string());
    PathBuf::from(data_dir).join(file_name)
}

fn resolve_game_dir(cli_game_dir: Option<PathBuf>) -> Option<PathBuf> {
    cli_game_dir.or_else(|| read_dotenv_key("GTA6_GAME_DIR").map(PathBuf::from))
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
        Command::Manifest { game_dir, out_file } => {
            let Some(game_dir) = resolve_game_dir(game_dir) else {
                eprintln!("Error: --game <dir> or GTA6_GAME_DIR (in .env) is required.");
                return ExitCode::FAILURE;
            };
            let out_file = out_file.unwrap_or_else(|| default_data_path("data-manifest.json"));
            match build_manifest(&game_dir, &out_file) {
                Ok(count) => {
                    println!("Wrote {} entries to {}", count, out_file.display());
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
        } => {
            let Some(game_dir) = resolve_game_dir(game_dir) else {
                eprintln!("Error: --game <dir> or GTA6_GAME_DIR (in .env) is required.");
                return ExitCode::FAILURE;
            };
            let out_file = out_file.unwrap_or_else(|| default_data_path("bootset.json"));
            match write_bootset(&game_dir, &out_file, head_bytes) {
                Ok(count) => {
                    println!("Wrote {} bootset entries to {}", count, out_file.display());
                    ExitCode::SUCCESS
                }
                Err(err) => {
                    eprintln!("Bootset generation failed: {err}");
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_sorted_manifest_json() {
        let base = env::temp_dir().join(format!("playgta6_manifest_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
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
            }
        );

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
    }
}
