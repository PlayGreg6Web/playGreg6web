use std::collections::BTreeMap;
use std::env;
use std::fmt;
use std::fs;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::rpf9::{RPF7_MAGIC, RPF8_MAGIC, RPF9_MAGIC, json_escape};

const ABI_VERSION: u32 = 1;

/// Number of header bytes probed per file.
///
/// 520 bytes (`0x208`) covers DOS/PE headers at common offsets, full 24-byte RPF headers,
/// shader/audio magic bytes, text prefixes, and Xbox `msft-xvd` magic at both offset `0x000`
/// and the standard XVD header offset `0x200`.
const HEADER_PROBE_BYTES: usize = 0x208;

/// Case-insensitive temporary download suffixes used by torrent clients and browsers.
pub const INCOMPLETE_SUFFIXES: &[&str] = &[
    ".!qb",
    ".part",
    ".crdownload",
    ".tmp",
    ".download",
    ".aria2",
    ".bc!",
];

const XVD_MAGIC: &[u8; 8] = b"msft-xvd";
const XVD_STANDARD_OFFSET: usize = 0x200;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FileCategory {
    PeExecutable,
    RpfArchive,
    XboxContainer,
    ShaderOrAudio,
    MetadataOrConfig,
    Other,
}

impl FileCategory {
    pub const ALL: [FileCategory; 6] = [
        FileCategory::PeExecutable,
        FileCategory::RpfArchive,
        FileCategory::XboxContainer,
        FileCategory::ShaderOrAudio,
        FileCategory::MetadataOrConfig,
        FileCategory::Other,
    ];

    pub fn json_key(self) -> &'static str {
        match self {
            FileCategory::PeExecutable => "pe_executable",
            FileCategory::RpfArchive => "rpf_archive",
            FileCategory::XboxContainer => "xbox_container",
            FileCategory::ShaderOrAudio => "shader_or_audio",
            FileCategory::MetadataOrConfig => "metadata_or_config",
            FileCategory::Other => "other",
        }
    }

    pub fn display_label(self) -> &'static str {
        match self {
            FileCategory::PeExecutable => "PeExecutable",
            FileCategory::RpfArchive => "RpfArchive",
            FileCategory::XboxContainer => "XboxContainer",
            FileCategory::ShaderOrAudio => "ShaderOrAudio",
            FileCategory::MetadataOrConfig => "MetadataOrConfig",
            FileCategory::Other => "Other",
        }
    }
}

impl fmt::Display for FileCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.display_label())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IncompleteReason {
    TempSuffix(String),
    EmptyFile,
    SparseOrZeroHeader,
    Unreadable(String),
}

impl IncompleteReason {
    pub fn json_code(&self) -> &'static str {
        match self {
            IncompleteReason::TempSuffix(_) => "temp_suffix",
            IncompleteReason::EmptyFile => "empty_file",
            IncompleteReason::SparseOrZeroHeader => "sparse_zero_header",
            IncompleteReason::Unreadable(_) => "unreadable",
        }
    }

    pub fn detail(&self) -> String {
        match self {
            IncompleteReason::TempSuffix(suffix) => format!("temporary download suffix {suffix}"),
            IncompleteReason::EmptyFile => "zero-byte file".to_string(),
            IncompleteReason::SparseOrZeroHeader => {
                "unallocated/zero-filled header bytes".to_string()
            }
            IncompleteReason::Unreadable(kind) => format!("file unreadable ({kind})"),
        }
    }
}

impl fmt::Display for IncompleteReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.detail())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScannedFile {
    pub rel_path: String,
    pub size: u64,
    pub category: FileCategory,
    pub format_tag: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IncompleteFile {
    pub rel_path: String,
    pub size: u64,
    pub reason: IncompleteReason,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NextStep {
    pub id: String,
    pub title: String,
    pub command: String,
    pub ready_files: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanReport {
    pub total_files: usize,
    pub total_bytes: u64,
    pub completed_count: usize,
    pub completed_bytes: u64,
    pub incomplete_count: usize,
    pub incomplete_bytes: u64,
    pub skipped_dirs: usize,
    pub completed: Vec<ScannedFile>,
    pub incomplete: Vec<IncompleteFile>,
    pub next_steps: Vec<NextStep>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileProbe {
    Complete {
        category: FileCategory,
        format_tag: Option<String>,
    },
    Incomplete(IncompleteReason),
}

/// Parses a `.env` file body into key-value pairs, ignoring blank lines and `#` comments,
/// stripping optional `export ` prefixes, matching surrounding single/double quotes,
/// and removing trailing inline comments.
pub fn parse_dotenv_str(content: &str) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    let content = content.strip_prefix('\u{FEFF}').unwrap_or(content);

    for raw_line in content.lines() {
        let mut line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("export ") {
            line = rest.trim_start();
        }
        let Some((raw_key, raw_val)) = line.split_once('=') else {
            continue;
        };
        let key = raw_key.trim();
        if key.is_empty() {
            continue;
        }
        let val = parse_dotenv_value(raw_val);
        map.insert(key.to_string(), val);
    }

    map
}

fn parse_dotenv_value(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    if let Some(rest) = trimmed.strip_prefix('"') {
        if let Some(end_idx) = rest.find('"') {
            return rest[..end_idx].to_string();
        }
        return rest.trim_end_matches('"').to_string();
    }

    if let Some(rest) = trimmed.strip_prefix('\'') {
        if let Some(end_idx) = rest.find('\'') {
            return rest[..end_idx].to_string();
        }
        return rest.trim_end_matches('\'').to_string();
    }

    // Unquoted value: strip inline comment when `#` starts the value or follows whitespace.
    let mut prev_ws = true;
    for (idx, ch) in trimmed.char_indices() {
        if ch == '#' && prev_ws {
            return trimmed[..idx].trim_end().to_string();
        }
        prev_ws = ch.is_whitespace();
    }

    trimmed.to_string()
}

/// Reads a single key from a specific `.env` file if present and non-empty.
pub fn read_dotenv_file_key(env_path: &Path, key: &str) -> Option<String> {
    let text = fs::read_to_string(env_path).ok()?;
    let values = parse_dotenv_str(&text);
    values.get(key).and_then(|v| {
        let trimmed = v.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

/// Walks upward from `start_dir` to locate a `.env` file (stopping after checking the repo root).
pub fn find_dotenv_path(start_dir: &Path) -> Option<PathBuf> {
    for dir in start_dir.ancestors() {
        let candidate = dir.join(".env");
        if candidate.is_file() {
            return Some(candidate);
        }
        if dir.join(".git").exists() {
            break;
        }
    }
    None
}

/// Resolves a config key from the process environment first, then `.env` in CWD or repo root.
pub fn resolve_env_key(key: &str) -> Option<String> {
    if let Ok(val) = env::var(key) {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }

    if let Ok(cwd) = env::current_dir()
        && let Some(env_path) = find_dotenv_path(&cwd)
        && let Some(val) = read_dotenv_file_key(&env_path, key)
    {
        return Some(val);
    }

    if let Ok(exe) = env::current_exe()
        && let Some(exe_dir) = exe.parent()
        && let Some(env_path) = find_dotenv_path(exe_dir)
        && let Some(val) = read_dotenv_file_key(&env_path, key)
    {
        return Some(val);
    }

    None
}

/// Resolves `GTA6_GAME_DIR` from environment or `.env`.
pub fn resolve_game_dir_from_env() -> Option<PathBuf> {
    resolve_env_key("GTA6_GAME_DIR").map(PathBuf::from)
}

/// Returns the matching temporary download suffix (lowercased) if `path` looks like an in-progress download.
pub fn matching_incomplete_suffix(path: &Path) -> Option<&'static str> {
    let name = path.file_name()?.to_str()?;
    let lower = name.to_ascii_lowercase();
    INCOMPLETE_SUFFIXES
        .iter()
        .copied()
        .find(|suffix| lower.ends_with(suffix))
}

fn lower_extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
}

fn read_prefix(file: &mut fs::File, max_bytes: usize) -> io::Result<Vec<u8>> {
    let mut buf = vec![0u8; max_bytes];
    let mut total = 0;
    while total < max_bytes {
        match file.read(&mut buf[total..]) {
            Ok(0) => break,
            Ok(n) => total += n,
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(err),
        }
    }
    buf.truncate(total);
    Ok(buf)
}

fn read_exact_at(file: &mut fs::File, offset: u64, len: usize) -> io::Result<Option<Vec<u8>>> {
    file.seek(SeekFrom::Start(offset))?;
    let mut buf = vec![0u8; len];
    let mut total = 0;
    while total < len {
        match file.read(&mut buf[total..]) {
            Ok(0) => return Ok(None),
            Ok(n) => total += n,
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(err),
        }
    }
    Ok(Some(buf))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PeKind {
    Pe64X86_64,
    Pe64Arm64,
    Pe64Other,
    Pe32,
    NotPe,
}

fn inspect_pe_header(file: &mut fs::File, prefix: &[u8], file_size: u64) -> io::Result<PeKind> {
    if prefix.len() < 0x40 || &prefix[0..2] != b"MZ" {
        return Ok(PeKind::NotPe);
    }

    let e_lfanew =
        u32::from_le_bytes([prefix[0x3C], prefix[0x3D], prefix[0x3E], prefix[0x3F]]) as u64;
    let Some(min_pe_end) = e_lfanew.checked_add(26) else {
        return Ok(PeKind::NotPe);
    };
    if min_pe_end > file_size {
        return Ok(PeKind::NotPe);
    }

    let pe_slice_owned;
    let pe_bytes: &[u8] = if let Ok(end_usize) = usize::try_from(min_pe_end)
        && end_usize <= prefix.len()
    {
        let start_usize = e_lfanew as usize;
        &prefix[start_usize..end_usize]
    } else {
        match read_exact_at(file, e_lfanew, 26)? {
            Some(bytes) => {
                pe_slice_owned = bytes;
                &pe_slice_owned
            }
            None => return Ok(PeKind::NotPe),
        }
    };

    if &pe_bytes[0..4] != b"PE\0\0" {
        return Ok(PeKind::NotPe);
    }

    let machine = u16::from_le_bytes([pe_bytes[4], pe_bytes[5]]);
    let opt_header_size = u16::from_le_bytes([pe_bytes[20], pe_bytes[21]]);
    if opt_header_size < 2 {
        return Ok(PeKind::NotPe);
    }

    let opt_magic = u16::from_le_bytes([pe_bytes[24], pe_bytes[25]]);
    match opt_magic {
        0x020B => match machine {
            0x8664 => Ok(PeKind::Pe64X86_64),
            0xAA64 => Ok(PeKind::Pe64Arm64),
            _ => Ok(PeKind::Pe64Other),
        },
        0x010B => Ok(PeKind::Pe32),
        _ => Ok(PeKind::NotPe),
    }
}

fn has_xvd_magic(prefix: &[u8]) -> bool {
    if prefix.len() >= XVD_MAGIC.len() && &prefix[..XVD_MAGIC.len()] == XVD_MAGIC {
        return true;
    }
    if prefix.len() >= XVD_STANDARD_OFFSET + XVD_MAGIC.len()
        && &prefix[XVD_STANDARD_OFFSET..XVD_STANDARD_OFFSET + XVD_MAGIC.len()] == XVD_MAGIC
    {
        return true;
    }
    false
}

fn is_all_zero(bytes: &[u8]) -> bool {
    !bytes.is_empty() && bytes.iter().all(|&b| b == 0)
}

fn sanitized_io_kind(err: &io::Error) -> String {
    match err.kind() {
        io::ErrorKind::NotFound => "not_found".to_string(),
        io::ErrorKind::PermissionDenied => "permission_denied".to_string(),
        io::ErrorKind::WouldBlock => "would_block".to_string(),
        io::ErrorKind::Interrupted => "interrupted".to_string(),
        io::ErrorKind::UnexpectedEof => "unexpected_eof".to_string(),
        _ => "io_error".to_string(),
    }
}

/// Probes a single file read-only to determine whether it is incomplete or completed
/// (and if completed, its [`FileCategory`] and optional format tag).
pub fn probe_file(path: &Path, size: u64) -> FileProbe {
    if let Some(suffix) = matching_incomplete_suffix(path) {
        return FileProbe::Incomplete(IncompleteReason::TempSuffix(suffix.to_string()));
    }

    if size == 0 {
        return FileProbe::Incomplete(IncompleteReason::EmptyFile);
    }

    let mut file = match fs::File::open(path) {
        Ok(f) => f,
        Err(err) => {
            return FileProbe::Incomplete(IncompleteReason::Unreadable(sanitized_io_kind(&err)));
        }
    };

    let to_read =
        usize::try_from(size.min(HEADER_PROBE_BYTES as u64)).unwrap_or(HEADER_PROBE_BYTES);
    let prefix = match read_prefix(&mut file, to_read) {
        Ok(buf) => buf,
        Err(err) => {
            return FileProbe::Incomplete(IncompleteReason::Unreadable(sanitized_io_kind(&err)));
        }
    };

    if prefix.is_empty() {
        return FileProbe::Incomplete(IncompleteReason::EmptyFile);
    }

    let ext = lower_extension(path);

    // 1. Sparse / zero-filled header checks for known binary formats.
    if let Some(ext_str) = ext.as_deref() {
        match ext_str {
            "exe" | "dll" => {
                if prefix.len() < 2 || (prefix[0] == 0x00 && prefix[1] == 0x00) {
                    return FileProbe::Incomplete(IncompleteReason::SparseOrZeroHeader);
                }
            }
            "rpf" => {
                if prefix.len() < 4 || prefix[0..4] == [0x00, 0x00, 0x00, 0x00] {
                    return FileProbe::Incomplete(IncompleteReason::SparseOrZeroHeader);
                }
            }
            "xvd" | "xvc" | "msixvc" => {
                if !has_xvd_magic(&prefix) && is_all_zero(&prefix) {
                    return FileProbe::Incomplete(IncompleteReason::SparseOrZeroHeader);
                }
            }
            "spv" | "cso" | "dxil" | "awc" | "opus" | "wem" => {
                let check_len = prefix.len().min(4);
                if is_all_zero(&prefix[..check_len]) {
                    return FileProbe::Incomplete(IncompleteReason::SparseOrZeroHeader);
                }
            }
            "xml" | "json" | "meta" | "txt" | "ini" | "config" => {
                if is_all_zero(&prefix) {
                    return FileProbe::Incomplete(IncompleteReason::SparseOrZeroHeader);
                }
            }
            _ => {}
        }
    }

    // 2. Magic-based classification for completed files.
    if prefix.len() >= 4 {
        let magic4 = [prefix[0], prefix[1], prefix[2], prefix[3]];
        if magic4 == RPF9_MAGIC {
            return FileProbe::Complete {
                category: FileCategory::RpfArchive,
                format_tag: Some("RPF9".to_string()),
            };
        }
        if magic4 == RPF8_MAGIC {
            return FileProbe::Complete {
                category: FileCategory::RpfArchive,
                format_tag: Some("RPF8".to_string()),
            };
        }
        if magic4 == RPF7_MAGIC {
            return FileProbe::Complete {
                category: FileCategory::RpfArchive,
                format_tag: Some("RPF7".to_string()),
            };
        }
    }

    if matches!(ext.as_deref(), Some("exe" | "dll")) {
        match inspect_pe_header(&mut file, &prefix, size) {
            Ok(PeKind::Pe64X86_64) => {
                return FileProbe::Complete {
                    category: FileCategory::PeExecutable,
                    format_tag: Some("PE64/x86-64".to_string()),
                };
            }
            Ok(PeKind::Pe64Arm64) => {
                return FileProbe::Complete {
                    category: FileCategory::PeExecutable,
                    format_tag: Some("PE64/ARM64".to_string()),
                };
            }
            Ok(PeKind::Pe64Other) => {
                return FileProbe::Complete {
                    category: FileCategory::PeExecutable,
                    format_tag: Some("PE64".to_string()),
                };
            }
            Ok(PeKind::Pe32) => {
                return FileProbe::Complete {
                    category: FileCategory::Other,
                    format_tag: Some("PE32".to_string()),
                };
            }
            Ok(PeKind::NotPe) => {}
            Err(err) => {
                return FileProbe::Incomplete(IncompleteReason::Unreadable(sanitized_io_kind(
                    &err,
                )));
            }
        }
    }

    if has_xvd_magic(&prefix) {
        return FileProbe::Complete {
            category: FileCategory::XboxContainer,
            format_tag: Some("msft-xvd".to_string()),
        };
    }

    // 3. Extension-based classification for completed files.
    if let Some(ext_str) = ext.as_deref() {
        match ext_str {
            "xvd" | "xvc" | "msixvc" => {
                return FileProbe::Complete {
                    category: FileCategory::XboxContainer,
                    format_tag: Some(ext_str.to_string()),
                };
            }
            "spv" | "cso" | "dxil" | "awc" | "opus" | "wem" => {
                return FileProbe::Complete {
                    category: FileCategory::ShaderOrAudio,
                    format_tag: Some(ext_str.to_string()),
                };
            }
            "xml" | "json" | "meta" | "txt" | "ini" | "config" => {
                return FileProbe::Complete {
                    category: FileCategory::MetadataOrConfig,
                    format_tag: Some(ext_str.to_string()),
                };
            }
            _ => {}
        }
    }

    FileProbe::Complete {
        category: FileCategory::Other,
        format_tag: None,
    }
}

/// Returns `true` if `path` appears complete and safe to include in an incremental manifest/bootset.
pub fn is_file_complete(path: &Path, size: u64) -> bool {
    matches!(probe_file(path, size), FileProbe::Complete { .. })
}

fn normalize_rel_path(root: &Path, path: &Path) -> Option<String> {
    let rel = path.strip_prefix(root).ok()?;
    Some(rel.to_string_lossy().replace('\\', "/"))
}

fn scan_dir_recursive(
    root: &Path,
    dir: &Path,
    completed: &mut Vec<ScannedFile>,
    incomplete: &mut Vec<IncompleteFile>,
    skipped_dirs: &mut usize,
) {
    let entries = match fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(_) => {
            *skipped_dirs += 1;
            return;
        }
    };

    for entry_res in entries {
        let Ok(entry) = entry_res else {
            continue;
        };
        let path = entry.path();
        let meta = match entry.metadata() {
            Ok(m) => m,
            Err(err) => {
                if let Some(rel_path) = normalize_rel_path(root, &path) {
                    incomplete.push(IncompleteFile {
                        rel_path,
                        size: 0,
                        reason: IncompleteReason::Unreadable(sanitized_io_kind(&err)),
                    });
                }
                continue;
            }
        };

        if meta.is_dir() {
            scan_dir_recursive(root, &path, completed, incomplete, skipped_dirs);
        } else if meta.is_file() {
            let Some(rel_path) = normalize_rel_path(root, &path) else {
                continue;
            };
            let size = meta.len();
            match probe_file(&path, size) {
                FileProbe::Complete {
                    category,
                    format_tag,
                } => completed.push(ScannedFile {
                    rel_path,
                    size,
                    category,
                    format_tag,
                }),
                FileProbe::Incomplete(reason) => incomplete.push(IncompleteFile {
                    rel_path,
                    size,
                    reason,
                }),
            }
        }
    }
}

fn build_next_steps(completed: &[ScannedFile], incomplete: &[IncompleteFile]) -> Vec<NextStep> {
    let mut steps = Vec::new();

    // 1. PE64 executables ready for Track 1 (R1) import inspection.
    let mut pe_files: Vec<&ScannedFile> = completed
        .iter()
        .filter(|f| f.category == FileCategory::PeExecutable)
        .collect();
    pe_files.sort_by(|a, b| {
        let a_is_exe = a.rel_path.to_ascii_lowercase().ends_with(".exe");
        let b_is_exe = b.rel_path.to_ascii_lowercase().ends_with(".exe");
        b_is_exe
            .cmp(&a_is_exe)
            .then_with(|| a.rel_path.cmp(&b.rel_path))
    });
    if let Some(first_pe) = pe_files.first() {
        let ready_files = pe_files.iter().map(|f| f.rel_path.clone()).collect();
        steps.push(NextStep {
            id: "recomp-imports".to_string(),
            title: "Inspect completed PE64 executables with playgta6-recomp (R1)".to_string(),
            command: format!(
                "cargo run -p playgta6-recomp -- <GTA6_GAME_DIR>/{} --out work",
                first_pe.rel_path
            ),
            ready_files,
        });
    }

    // 2. RPF archives ready for header inspection & bootset generation.
    let rpf_files: Vec<&ScannedFile> = completed
        .iter()
        .filter(|f| f.category == FileCategory::RpfArchive)
        .collect();
    if !rpf_files.is_empty() {
        let first_rpf = rpf_files
            .iter()
            .find(|f| f.format_tag.as_deref() == Some("RPF9"))
            .unwrap_or(&rpf_files[0]);
        let ready_files = rpf_files.iter().map(|f| f.rel_path.clone()).collect();
        steps.push(NextStep {
            id: "rpf9-inspect".to_string(),
            title: "Inspect completed RPF archive headers (W4)".to_string(),
            command: format!(
                "cargo run -p playgta6-tools -- inspect-rpf <GTA6_GAME_DIR>/{}",
                first_rpf.rel_path
            ),
            ready_files,
        });
    }

    // 3. Bootset generation if any completed .rpf or .config file exists.
    let bootset_ready: Vec<String> = completed
        .iter()
        .filter(|f| {
            let lower = f.rel_path.to_ascii_lowercase();
            lower.ends_with(".rpf") || lower.ends_with(".config")
        })
        .map(|f| f.rel_path.clone())
        .collect();
    if !bootset_ready.is_empty() {
        let flag = if incomplete.is_empty() {
            ""
        } else {
            " --skip-incomplete"
        };
        steps.push(NextStep {
            id: "bootset".to_string(),
            title: if incomplete.is_empty() {
                "Generate bootset.json from completed RPF/config files".to_string()
            } else {
                "Generate incremental bootset.json from already-completed RPF/config files"
                    .to_string()
            },
            command: format!("cargo run -p playgta6-tools -- bootset{flag}"),
            ready_files: bootset_ready,
        });
    }

    // 4. SPIR-V shaders ready for Track 4 (S1) WGSL conversion.
    let spv_ready: Vec<String> = completed
        .iter()
        .filter(|f| {
            f.category == FileCategory::ShaderOrAudio && f.format_tag.as_deref() == Some("spv")
        })
        .map(|f| f.rel_path.clone())
        .collect();
    if !spv_ready.is_empty() {
        steps.push(NextStep {
            id: "shader-convert".to_string(),
            title: "Convert completed SPIR-V shaders to WGSL packs (S1/S2)".to_string(),
            command:
                "cargo run -p playgta6-shaders -- convert --in <GTA6_GAME_DIR> --out work/shaders"
                    .to_string(),
            ready_files: spv_ready,
        });
    }

    // 5. Incremental or full manifest generation.
    if !completed.is_empty() {
        let is_partial = !incomplete.is_empty();
        steps.push(NextStep {
            id: if is_partial {
                "incremental-manifest".to_string()
            } else {
                "full-manifest".to_string()
            },
            title: if is_partial {
                "Build incremental data-manifest.json while remaining files download".to_string()
            } else {
                "Build full data-manifest.json for local streaming server".to_string()
            },
            command: if is_partial {
                "cargo run -p playgta6-tools -- manifest --skip-incomplete".to_string()
            } else {
                "cargo run -p playgta6-tools -- manifest".to_string()
            },
            ready_files: Vec::new(),
        });
    } else {
        steps.push(NextStep {
            id: "wait-for-downloads".to_string(),
            title: "No completed files detected yet — re-run scan as downloads finish".to_string(),
            command: "cargo run -p playgta6-tools -- scan --from-env".to_string(),
            ready_files: Vec::new(),
        });
    }

    steps
}

/// Scans `root` read-only, classifying completed vs in-progress/sparse/unreadable files.
pub fn scan_directory(root: &Path) -> io::Result<ScanReport> {
    let meta = fs::metadata(root)?;
    if !meta.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "scan target is not a directory",
        ));
    }

    let mut completed = Vec::new();
    let mut incomplete = Vec::new();
    let mut skipped_dirs = 0usize;

    scan_dir_recursive(
        root,
        root,
        &mut completed,
        &mut incomplete,
        &mut skipped_dirs,
    );

    completed.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
    incomplete.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));

    let completed_count = completed.len();
    let completed_bytes: u64 = completed.iter().map(|f| f.size).sum();
    let incomplete_count = incomplete.len();
    let incomplete_bytes: u64 = incomplete.iter().map(|f| f.size).sum();
    let total_files = completed_count + incomplete_count;
    let total_bytes = completed_bytes.saturating_add(incomplete_bytes);

    let next_steps = build_next_steps(&completed, &incomplete);

    Ok(ScanReport {
        total_files,
        total_bytes,
        completed_count,
        completed_bytes,
        incomplete_count,
        incomplete_bytes,
        skipped_dirs,
        completed,
        incomplete,
        next_steps,
    })
}

pub fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

    if bytes as f64 >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB)
    } else if bytes as f64 >= MIB {
        format!("{:.2} MiB", bytes as f64 / MIB)
    } else if bytes as f64 >= KIB {
        format!("{:.2} KiB", bytes as f64 / KIB)
    } else {
        format!("{bytes} B")
    }
}

impl ScanReport {
    pub fn category_stats(&self, category: FileCategory) -> (usize, u64) {
        let mut count = 0usize;
        let mut bytes = 0u64;
        for f in &self.completed {
            if f.category == category {
                count += 1;
                bytes = bytes.saturating_add(f.size);
            }
        }
        (count, bytes)
    }

    /// Formats a sanitized JSON report using normalized relative paths only (never host paths or raw bytes).
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        out.push_str("{\n");
        out.push_str(&format!("  \"abi_version\": {ABI_VERSION},\n"));
        out.push_str("  \"summary\": {\n");
        out.push_str(&format!("    \"total_files\": {},\n", self.total_files));
        out.push_str(&format!("    \"total_bytes\": {},\n", self.total_bytes));
        out.push_str(&format!(
            "    \"completed_files\": {},\n",
            self.completed_count
        ));
        out.push_str(&format!(
            "    \"completed_bytes\": {},\n",
            self.completed_bytes
        ));
        out.push_str(&format!(
            "    \"incomplete_files\": {},\n",
            self.incomplete_count
        ));
        out.push_str(&format!(
            "    \"incomplete_bytes\": {},\n",
            self.incomplete_bytes
        ));
        out.push_str(&format!("    \"skipped_dirs\": {}\n", self.skipped_dirs));
        out.push_str("  },\n");

        out.push_str("  \"categories\": {\n");
        for (idx, cat) in FileCategory::ALL.iter().enumerate() {
            let (count, bytes) = self.category_stats(*cat);
            let comma = if idx + 1 == FileCategory::ALL.len() {
                ""
            } else {
                ","
            };
            out.push_str(&format!(
                "    \"{}\": {{\"count\": {}, \"bytes\": {}}}{}\n",
                cat.json_key(),
                count,
                bytes,
                comma
            ));
        }
        out.push_str("  },\n");

        out.push_str("  \"completed\": [");
        if !self.completed.is_empty() {
            out.push('\n');
            for (idx, file) in self.completed.iter().enumerate() {
                let comma = if idx + 1 == self.completed.len() {
                    ""
                } else {
                    ","
                };
                let format_field = match &file.format_tag {
                    Some(tag) => format!("\"{}\"", json_escape(tag)),
                    None => "null".to_string(),
                };
                out.push_str(&format!(
                    "    {{\"path\": \"{}\", \"size\": {}, \"category\": \"{}\", \"format\": {}}}{}\n",
                    json_escape(&file.rel_path),
                    file.size,
                    file.category.json_key(),
                    format_field,
                    comma
                ));
            }
            out.push_str("  ");
        }
        out.push_str("],\n");

        out.push_str("  \"incomplete\": [");
        if !self.incomplete.is_empty() {
            out.push('\n');
            for (idx, file) in self.incomplete.iter().enumerate() {
                let comma = if idx + 1 == self.incomplete.len() {
                    ""
                } else {
                    ","
                };
                out.push_str(&format!(
                    "    {{\"path\": \"{}\", \"size\": {}, \"reason\": \"{}\", \"detail\": \"{}\"}}{}\n",
                    json_escape(&file.rel_path),
                    file.size,
                    file.reason.json_code(),
                    json_escape(&file.reason.detail()),
                    comma
                ));
            }
            out.push_str("  ");
        }
        out.push_str("],\n");

        out.push_str("  \"next_steps\": [");
        if !self.next_steps.is_empty() {
            out.push('\n');
            for (idx, step) in self.next_steps.iter().enumerate() {
                let comma = if idx + 1 == self.next_steps.len() {
                    ""
                } else {
                    ","
                };
                out.push_str("    {\n");
                out.push_str(&format!("      \"id\": \"{}\",\n", json_escape(&step.id)));
                out.push_str(&format!(
                    "      \"title\": \"{}\",\n",
                    json_escape(&step.title)
                ));
                out.push_str(&format!(
                    "      \"command\": \"{}\",\n",
                    json_escape(&step.command)
                ));
                out.push_str("      \"ready_files\": [");
                for (f_idx, rel) in step.ready_files.iter().enumerate() {
                    if f_idx > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(&format!("\"{}\"", json_escape(rel)));
                }
                out.push_str("]\n");
                out.push_str(&format!("    }}{comma}\n"));
            }
            out.push_str("  ");
        }
        out.push_str("]\n");
        out.push_str("}\n");
        out
    }

    /// Formats a clean human-readable terminal summary using normalized relative paths only.
    pub fn format_human(&self) -> String {
        const MAX_LISTED: usize = 20;
        let mut out = String::new();
        out.push_str("=== playgta6web Game Folder Scan ===\n");
        out.push_str(&format!(
            "Total files      : {} ({} / {} bytes)\n",
            self.total_files,
            format_bytes(self.total_bytes),
            self.total_bytes
        ));
        out.push_str(&format!(
            "Completed        : {} ({} / {} bytes)\n",
            self.completed_count,
            format_bytes(self.completed_bytes),
            self.completed_bytes
        ));
        out.push_str(&format!(
            "Incomplete       : {} ({} / {} bytes)\n",
            self.incomplete_count,
            format_bytes(self.incomplete_bytes),
            self.incomplete_bytes
        ));
        if self.skipped_dirs > 0 {
            out.push_str(&format!(
                "Skipped dirs     : {} (unreadable)\n",
                self.skipped_dirs
            ));
        }

        out.push_str("\nCompleted by category:\n");
        for cat in FileCategory::ALL {
            let (count, bytes) = self.category_stats(cat);
            if count > 0 {
                out.push_str(&format!(
                    "  {:<18} {:>5} file(s)  ({})\n",
                    cat.display_label(),
                    count,
                    format_bytes(bytes)
                ));
            }
        }
        if self.completed_count == 0 {
            out.push_str("  (none yet)\n");
        }

        if !self.incomplete.is_empty() {
            out.push_str("\nIncomplete / downloading files:\n");
            for file in self.incomplete.iter().take(MAX_LISTED) {
                out.push_str(&format!(
                    "  [incomplete] {} ({}) - {}\n",
                    file.rel_path,
                    format_bytes(file.size),
                    file.reason
                ));
            }
            if self.incomplete.len() > MAX_LISTED {
                out.push_str(&format!(
                    "  ... and {} more incomplete file(s)\n",
                    self.incomplete.len() - MAX_LISTED
                ));
            }
        }

        out.push_str("\nActionable next steps:\n");
        for (idx, step) in self.next_steps.iter().enumerate() {
            out.push_str(&format!("  {}. {}\n", idx + 1, step.title));
            if !step.ready_files.is_empty() {
                let shown: Vec<&str> = step
                    .ready_files
                    .iter()
                    .take(5)
                    .map(String::as_str)
                    .collect();
                let suffix = if step.ready_files.len() > 5 {
                    format!(" (+{} more)", step.ready_files.len() - 5)
                } else {
                    String::new()
                };
                out.push_str(&format!("     Ready: {}{}\n", shown.join(", "), suffix));
            }
            out.push_str(&format!("     Run  : {}\n", step.command));
        }

        out
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
            "playgta6_scan_{label}_{}_{}",
            std::process::id(),
            n
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn make_synthetic_pe64(machine: u16) -> Vec<u8> {
        let mut buf = vec![0u8; 0x120];
        buf[0..2].copy_from_slice(b"MZ");
        let e_lfanew: u32 = 0x80;
        buf[0x3C..0x40].copy_from_slice(&e_lfanew.to_le_bytes());
        let pe = e_lfanew as usize;
        buf[pe..pe + 4].copy_from_slice(b"PE\0\0");
        buf[pe + 4..pe + 6].copy_from_slice(&machine.to_le_bytes());
        let opt_size: u16 = 0x00F0;
        buf[pe + 20..pe + 22].copy_from_slice(&opt_size.to_le_bytes());
        let opt_magic: u16 = 0x020B; // PE32+ (PE64)
        buf[pe + 24..pe + 26].copy_from_slice(&opt_magic.to_le_bytes());
        buf
    }

    fn make_synthetic_pe32() -> Vec<u8> {
        let mut buf = vec![0u8; 0x120];
        buf[0..2].copy_from_slice(b"MZ");
        let e_lfanew: u32 = 0x80;
        buf[0x3C..0x40].copy_from_slice(&e_lfanew.to_le_bytes());
        let pe = e_lfanew as usize;
        buf[pe..pe + 4].copy_from_slice(b"PE\0\0");
        buf[pe + 4..pe + 6].copy_from_slice(&0x014Cu16.to_le_bytes());
        buf[pe + 20..pe + 22].copy_from_slice(&0x00E0u16.to_le_bytes());
        buf[pe + 24..pe + 26].copy_from_slice(&0x010Bu16.to_le_bytes()); // PE32
        buf
    }

    fn make_synthetic_rpf9() -> Vec<u8> {
        let mut buf = vec![0u8; 64];
        buf[0..4].copy_from_slice(b"RPF9");
        buf[4..8].copy_from_slice(&1u32.to_le_bytes());
        buf[8..12].copy_from_slice(&16u32.to_le_bytes());
        buf
    }

    #[test]
    fn scans_mixed_completed_and_incomplete_game_folder() {
        let dir = unique_temp_dir("mixed");
        let game = dir.join("game");
        let sub_shaders = game.join("shaders");
        let sub_rpf = game.join("x64");
        fs::create_dir_all(&sub_shaders).unwrap();
        fs::create_dir_all(&sub_rpf).unwrap();

        // Completed files across all categories
        fs::write(game.join("GTA6.exe"), make_synthetic_pe64(0x8664)).unwrap();
        fs::write(game.join("helper32.dll"), make_synthetic_pe32()).unwrap();
        fs::write(sub_rpf.join("levels.rpf"), make_synthetic_rpf9()).unwrap();
        fs::write(game.join("MicrosoftGame.Config"), b"<Game></Game>").unwrap();
        fs::write(
            sub_shaders.join("pass.spv"),
            b"\x03\x02\x23\x07\x00\x00\x01\x00",
        )
        .unwrap();

        let mut xvd = vec![0u8; 0x220];
        xvd[0x200..0x208].copy_from_slice(b"msft-xvd");
        fs::write(game.join("package.xvc"), &xvd).unwrap();

        // Incomplete files: temp suffixes, sparse/zero headers, empty file
        fs::write(sub_rpf.join("audio.rpf.!qB"), b"partial data").unwrap();
        fs::write(sub_rpf.join("world.rpf.part"), b"partial data 2").unwrap();
        fs::write(sub_rpf.join("textures.rpf"), vec![0u8; 1024]).unwrap();
        fs::write(game.join("engine.dll"), vec![0u8; 512]).unwrap();
        fs::write(game.join("empty.bin"), b"").unwrap();

        let report = scan_directory(&game).unwrap();

        assert_eq!(report.completed_count, 6);
        assert_eq!(report.incomplete_count, 5);
        assert_eq!(report.total_files, 11);
        assert_eq!(
            report.total_bytes,
            report.completed_bytes + report.incomplete_bytes
        );

        // Verify categories
        assert_eq!(report.category_stats(FileCategory::PeExecutable).0, 1);
        assert_eq!(report.category_stats(FileCategory::RpfArchive).0, 1);
        assert_eq!(report.category_stats(FileCategory::XboxContainer).0, 1);
        assert_eq!(report.category_stats(FileCategory::ShaderOrAudio).0, 1);
        assert_eq!(report.category_stats(FileCategory::MetadataOrConfig).0, 1);
        assert_eq!(report.category_stats(FileCategory::Other).0, 1); // helper32.dll (PE32)

        // Verify PE64 classification
        let pe = report
            .completed
            .iter()
            .find(|f| f.rel_path == "GTA6.exe")
            .unwrap();
        assert_eq!(pe.category, FileCategory::PeExecutable);
        assert_eq!(pe.format_tag.as_deref(), Some("PE64/x86-64"));

        // Verify RPF9 classification
        let rpf = report
            .completed
            .iter()
            .find(|f| f.rel_path == "x64/levels.rpf")
            .unwrap();
        assert_eq!(rpf.category, FileCategory::RpfArchive);
        assert_eq!(rpf.format_tag.as_deref(), Some("RPF9"));

        // Verify incomplete reasons
        let qb = report
            .incomplete
            .iter()
            .find(|f| f.rel_path == "x64/audio.rpf.!qB")
            .unwrap();
        assert_eq!(qb.reason, IncompleteReason::TempSuffix(".!qb".to_string()));

        let part = report
            .incomplete
            .iter()
            .find(|f| f.rel_path == "x64/world.rpf.part")
            .unwrap();
        assert_eq!(
            part.reason,
            IncompleteReason::TempSuffix(".part".to_string())
        );

        let sparse_rpf = report
            .incomplete
            .iter()
            .find(|f| f.rel_path == "x64/textures.rpf")
            .unwrap();
        assert_eq!(sparse_rpf.reason, IncompleteReason::SparseOrZeroHeader);

        let sparse_dll = report
            .incomplete
            .iter()
            .find(|f| f.rel_path == "engine.dll")
            .unwrap();
        assert_eq!(sparse_dll.reason, IncompleteReason::SparseOrZeroHeader);

        let empty = report
            .incomplete
            .iter()
            .find(|f| f.rel_path == "empty.bin")
            .unwrap();
        assert_eq!(empty.reason, IncompleteReason::EmptyFile);

        // Verify actionable next steps
        assert!(report.next_steps.iter().any(|s| s.id == "recomp-imports"));
        assert!(report.next_steps.iter().any(|s| s.id == "rpf9-inspect"));
        assert!(report.next_steps.iter().any(|s| s.id == "bootset"));
        assert!(report.next_steps.iter().any(|s| s.id == "shader-convert"));
        assert!(
            report
                .next_steps
                .iter()
                .any(|s| s.id == "incremental-manifest")
        );

        // Verify JSON has no absolute host paths and includes expected fields
        let json = report.to_json();
        let abs_dir = game.to_string_lossy().to_string();
        assert!(!json.contains(&abs_dir));
        assert!(!json.contains(&abs_dir.replace('\\', "/")));
        assert!(json.contains("\"abi_version\": 1"));
        assert!(json.contains("\"completed_files\": 6"));
        assert!(json.contains("\"incomplete_files\": 5"));
        assert!(json.contains("\"path\": \"GTA6.exe\""));
        assert!(json.contains("\"path\": \"x64/levels.rpf\""));
        assert!(json.contains("\"reason\": \"temp_suffix\""));
        assert!(json.contains("\"reason\": \"sparse_zero_header\""));
        assert!(json.contains("\"reason\": \"empty_file\""));

        // Verify human report mentions key summaries and next steps
        let human = report.format_human();
        assert!(human.contains("Completed        : 6"));
        assert!(human.contains("Incomplete       : 5"));
        assert!(human.contains("PeExecutable"));
        assert!(human.contains("RpfArchive"));
        assert!(human.contains("GTA6.exe"));
        assert!(human.contains("--skip-incomplete"));
        assert!(!human.contains(&abs_dir));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn detects_all_temporary_download_suffixes_case_insensitively() {
        for name in [
            "a.rpf.!qB",
            "b.rpf.!QB",
            "c.bin.PART",
            "d.exe.crdownload",
            "e.tmp",
            "f.Download",
            "g.rpf.aria2",
            "h.rpf.BC!",
        ] {
            assert!(
                matching_incomplete_suffix(Path::new(name)).is_some(),
                "expected {name} to match incomplete suffix"
            );
        }

        assert!(matching_incomplete_suffix(Path::new("levels.rpf")).is_none());
        assert!(matching_incomplete_suffix(Path::new("GTA6.exe")).is_none());
    }

    #[test]
    fn detects_pe64_with_large_e_lfanew_beyond_initial_probe() {
        let dir = unique_temp_dir("large_pe");
        let exe_path = dir.join("big_stub.exe");

        let mut buf = vec![0u8; 0x500];
        buf[0..2].copy_from_slice(b"MZ");
        let e_lfanew: u32 = 0x400; // Beyond HEADER_PROBE_BYTES (0x208)
        buf[0x3C..0x40].copy_from_slice(&e_lfanew.to_le_bytes());
        let pe = e_lfanew as usize;
        buf[pe..pe + 4].copy_from_slice(b"PE\0\0");
        buf[pe + 4..pe + 6].copy_from_slice(&0x8664u16.to_le_bytes());
        buf[pe + 20..pe + 22].copy_from_slice(&0x00F0u16.to_le_bytes());
        buf[pe + 24..pe + 26].copy_from_slice(&0x020Bu16.to_le_bytes());
        fs::write(&exe_path, &buf).unwrap();

        let probe = probe_file(&exe_path, buf.len() as u64);
        assert_eq!(
            probe,
            FileProbe::Complete {
                category: FileCategory::PeExecutable,
                format_tag: Some("PE64/x86-64".to_string()),
            }
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn parses_dotenv_with_quotes_comments_and_exports() {
        let sample = "\u{FEFF}# Local config\n\
            GTA6_GAME_DIR=\"C:\\Games\\GTA 6\" # inline comment\n\
            export GTA6_DATA_DIR='local-data'\n\
            GTA6_WORK_DIR=work # trailing comment\n\
            HASH_IN_PATH=C:\\Games\\build#1\n\
            EMPTY_VAL=\n\
            # COMMENTED=ignored\n";

        let parsed = parse_dotenv_str(sample);
        assert_eq!(
            parsed.get("GTA6_GAME_DIR").map(String::as_str),
            Some("C:\\Games\\GTA 6")
        );
        assert_eq!(
            parsed.get("GTA6_DATA_DIR").map(String::as_str),
            Some("local-data")
        );
        assert_eq!(
            parsed.get("GTA6_WORK_DIR").map(String::as_str),
            Some("work")
        );
        assert_eq!(
            parsed.get("HASH_IN_PATH").map(String::as_str),
            Some("C:\\Games\\build#1")
        );
        assert_eq!(parsed.get("EMPTY_VAL").map(String::as_str), Some(""));
        assert!(!parsed.contains_key("COMMENTED"));

        let dir = unique_temp_dir("dotenv_file");
        let env_file = dir.join(".env");
        fs::write(&env_file, sample).unwrap();
        assert_eq!(
            read_dotenv_file_key(&env_file, "GTA6_GAME_DIR").as_deref(),
            Some("C:\\Games\\GTA 6")
        );
        assert_eq!(read_dotenv_file_key(&env_file, "EMPTY_VAL"), None);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn escapes_special_characters_in_scan_json() {
        let report = ScanReport {
            total_files: 1,
            total_bytes: 12,
            completed_count: 1,
            completed_bytes: 12,
            incomplete_count: 0,
            incomplete_bytes: 0,
            skipped_dirs: 0,
            completed: vec![ScannedFile {
                rel_path: "dir/quote\"name.meta".to_string(),
                size: 12,
                category: FileCategory::MetadataOrConfig,
                format_tag: Some("meta".to_string()),
            }],
            incomplete: Vec::new(),
            next_steps: Vec::new(),
        };

        let json = report.to_json();
        assert!(json.contains("\"path\": \"dir/quote\\\"name.meta\""));
    }

    #[test]
    fn handles_all_incomplete_and_all_completed_folders() {
        let dir = unique_temp_dir("edge_states");

        // 1. All-incomplete folder
        let only_partial = dir.join("partial_only");
        fs::create_dir_all(&only_partial).unwrap();
        fs::write(only_partial.join("big.rpf.!qB"), b"chunk").unwrap();
        fs::write(only_partial.join("zero.xvd"), vec![0u8; 64]).unwrap();

        let partial_report = scan_directory(&only_partial).unwrap();
        assert_eq!(partial_report.completed_count, 0);
        assert_eq!(partial_report.incomplete_count, 2);
        assert!(
            partial_report
                .next_steps
                .iter()
                .any(|s| s.id == "wait-for-downloads")
        );
        let partial_json = partial_report.to_json();
        assert!(partial_json.contains("\"completed\": [],"));

        // 2. All-completed folder covering RPF7/8, extensionless msft-xvd, audio/shaders, metadata
        let all_done = dir.join("all_done");
        fs::create_dir_all(&all_done).unwrap();
        fs::write(all_done.join("legacy.rpf"), b"RPF7\x00\x00\x00\x00").unwrap();
        fs::write(all_done.join("mid.rpf"), b"RPF8\x00\x00\x00\x00").unwrap();
        fs::write(all_done.join("guid_container"), b"msft-xvd\x01\x02\x03\x04").unwrap();
        fs::write(all_done.join("voice.wem"), b"RIFF\x24\x00\x00\x00").unwrap();
        fs::write(all_done.join("settings.ini"), b"[display]\nvsync=1\n").unwrap();

        let done_report = scan_directory(&all_done).unwrap();
        assert_eq!(done_report.completed_count, 5);
        assert_eq!(done_report.incomplete_count, 0);
        assert_eq!(done_report.category_stats(FileCategory::RpfArchive).0, 2);
        assert_eq!(done_report.category_stats(FileCategory::XboxContainer).0, 1);
        assert_eq!(done_report.category_stats(FileCategory::ShaderOrAudio).0, 1);
        assert_eq!(
            done_report.category_stats(FileCategory::MetadataOrConfig).0,
            1
        );
        assert!(
            done_report
                .next_steps
                .iter()
                .any(|s| s.id == "full-manifest")
        );
        let done_json = done_report.to_json();
        assert!(done_json.contains("\"incomplete\": [],"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn finds_dotenv_in_parent_repo_root() {
        let dir = unique_temp_dir("dotenv_walk");
        let repo_root = dir.join("repo");
        let nested = repo_root.join("tools").join("src");
        fs::create_dir_all(&nested).unwrap();
        fs::write(repo_root.join(".git"), b"gitdir: .git/worktrees/x\n").unwrap();
        let env_path = repo_root.join(".env");
        fs::write(&env_path, "GTA6_GAME_DIR=\"D:/Games/GTA6\"\n").unwrap();

        let found = find_dotenv_path(&nested).unwrap();
        assert_eq!(found, env_path);
        assert_eq!(
            read_dotenv_file_key(&found, "GTA6_GAME_DIR").as_deref(),
            Some("D:/Games/GTA6")
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
