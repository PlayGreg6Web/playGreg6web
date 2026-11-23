use std::fmt;
use std::fs;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

pub const RPF9_MAGIC: [u8; 4] = *b"RPF9";
pub const RPF8_MAGIC: [u8; 4] = *b"RPF8";
pub const RPF7_MAGIC: [u8; 4] = *b"RPF7";

/// Minimum header size for RPF9 archives inspected by this tool (24 bytes):
/// - `0x00..0x04`: magic (`"RPF9"`)
/// - `0x04..0x08`: `entry_count` (`u32` LE)
/// - `0x08..0x0C`: `names_size` (`u32` LE)
/// - `0x0C..0x10`: `encryption_tag` (`u32` LE)
/// - `0x10..0x14`: `metadata_size` (`u32` LE)
/// - `0x14..0x18`: `flags` (`u32` LE)
pub const RPF9_HEADER_SIZE: usize = 24;

/// Conservative minimum size (in bytes) per TOC entry in RPF7/8/9 archives.
pub const RPF9_ENTRY_STRIDE: u64 = 16;

/// Default prefix byte length prefetched per archive/config in `bootset.json`.
pub const DEFAULT_BOOTSET_HEAD_BYTES: u64 = 65_536;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RpfVersion {
    Rpf7,
    Rpf8,
    Rpf9,
    Other(u8),
}

impl fmt::Display for RpfVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RpfVersion::Rpf7 => write!(f, "RPF7"),
            RpfVersion::Rpf8 => write!(f, "RPF8"),
            RpfVersion::Rpf9 => write!(f, "RPF9"),
            RpfVersion::Other(d) => write!(f, "RPF{}", *d as char),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RpfHeader {
    pub version: RpfVersion,
    pub entry_count: u32,
    pub names_size: u32,
    pub encryption_tag: u32,
    pub metadata_size: u32,
    pub flags: u32,
    pub total_size: u64,
    pub min_index_bytes: u64,
}

impl RpfHeader {
    /// Human-readable classification for known RAGE/RPF encryption tags.
    pub fn encryption_label(&self) -> &'static str {
        match self.encryption_tag {
            0x0000_0000 => "none",
            0x4E45_504F => "open",
            0x0FFF_FFF9 => "aes",
            0x0FEF_FFFF => "ng",
            _ => "custom",
        }
    }

    /// Formats a sanitized, metadata-only summary without ever dumping archive bytes.
    pub fn format_summary(&self) -> String {
        format!(
            concat!(
                "{} Archive Header\n",
                "  Version:          {}\n",
                "  File Size:        {} bytes\n",
                "  Entry Count:      {}\n",
                "  Names Size:       {} bytes\n",
                "  Metadata Size:    {} bytes\n",
                "  Min Index Bytes:  {} bytes\n",
                "  Flags:            0x{:08X}\n",
                "  Encryption Tag:   0x{:08X} ({})"
            ),
            self.version,
            self.version,
            self.total_size,
            self.entry_count,
            self.names_size,
            self.metadata_size,
            self.min_index_bytes,
            self.flags,
            self.encryption_tag,
            self.encryption_label()
        )
    }
}

impl fmt::Display for RpfHeader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.format_summary())
    }
}

#[derive(Debug)]
pub enum RpfError {
    Io(io::Error),
    TruncatedHeader { actual: usize, expected: usize },
    UnsupportedVersion { found: RpfVersion, expected: RpfVersion },
    InvalidMagic([u8; 4]),
    SizeOverflow,
    TruncatedArchive { required_bytes: u64, actual_bytes: u64 },
}

impl fmt::Display for RpfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RpfError::Io(err) => write!(f, "I/O error reading RPF archive: {err}"),
            RpfError::TruncatedHeader { actual, expected } => write!(
                f,
                "truncated RPF header: got {actual} bytes, need at least {expected} bytes"
            ),
            RpfError::UnsupportedVersion { found, expected } => write!(
                f,
                "unsupported RPF archive version {found} (expected {expected})"
            ),
            RpfError::InvalidMagic(magic) => write!(
                f,
                "invalid RPF magic bytes: {:02X} {:02X} {:02X} {:02X} (expected RPF9)",
                magic[0], magic[1], magic[2], magic[3]
            ),
            RpfError::SizeOverflow => {
                write!(f, "RPF header metadata sizes overflow 64-bit bounds")
            }
            RpfError::TruncatedArchive {
                required_bytes,
                actual_bytes,
            } => write!(
                f,
                "truncated RPF9 archive: header requires at least {required_bytes} bytes, but file is {actual_bytes} bytes"
            ),
        }
    }
}

impl std::error::Error for RpfError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RpfError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for RpfError {
    fn from(value: io::Error) -> Self {
        RpfError::Io(value)
    }
}

fn detect_rpf_version(magic: [u8; 4]) -> Result<RpfVersion, RpfError> {
    if magic == RPF9_MAGIC {
        Ok(RpfVersion::Rpf9)
    } else if magic == RPF8_MAGIC {
        Ok(RpfVersion::Rpf8)
    } else if magic == RPF7_MAGIC {
        Ok(RpfVersion::Rpf7)
    } else if magic[0..3] == *b"RPF" && magic[3].is_ascii_alphanumeric() {
        Ok(RpfVersion::Other(magic[3]))
    } else {
        Err(RpfError::InvalidMagic(magic))
    }
}

fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

/// Inspects an RPF9 header from an in-memory prefix slice and known total archive size.
///
/// Validates the magic version (`RPF9` only; `RPF8`/`RPF7`/other `RPF*` return
/// [`RpfError::UnsupportedVersion`]) and checks that declared TOC/name/metadata sizes
/// fit within `total_size` using checked arithmetic.
pub fn inspect_rpf_bytes(header_bytes: &[u8], total_size: u64) -> Result<RpfHeader, RpfError> {
    if header_bytes.len() < 4 {
        return Err(RpfError::TruncatedHeader {
            actual: header_bytes.len(),
            expected: RPF9_HEADER_SIZE,
        });
    }

    let magic = [
        header_bytes[0],
        header_bytes[1],
        header_bytes[2],
        header_bytes[3],
    ];
    let version = detect_rpf_version(magic)?;
    if version != RpfVersion::Rpf9 {
        return Err(RpfError::UnsupportedVersion {
            found: version,
            expected: RpfVersion::Rpf9,
        });
    }

    if header_bytes.len() < RPF9_HEADER_SIZE {
        return Err(RpfError::TruncatedHeader {
            actual: header_bytes.len(),
            expected: RPF9_HEADER_SIZE,
        });
    }

    if total_size < RPF9_HEADER_SIZE as u64 {
        return Err(RpfError::TruncatedArchive {
            required_bytes: RPF9_HEADER_SIZE as u64,
            actual_bytes: total_size,
        });
    }

    let entry_count = read_u32_le(header_bytes, 4);
    let names_size = read_u32_le(header_bytes, 8);
    let encryption_tag = read_u32_le(header_bytes, 12);
    let metadata_size = read_u32_le(header_bytes, 16);
    let flags = read_u32_le(header_bytes, 20);

    let entries_bytes = u64::from(entry_count)
        .checked_mul(RPF9_ENTRY_STRIDE)
        .ok_or(RpfError::SizeOverflow)?;
    let min_index_bytes = (RPF9_HEADER_SIZE as u64)
        .checked_add(entries_bytes)
        .and_then(|v| v.checked_add(u64::from(names_size)))
        .and_then(|v| v.checked_add(u64::from(metadata_size)))
        .ok_or(RpfError::SizeOverflow)?;

    if min_index_bytes > total_size {
        return Err(RpfError::TruncatedArchive {
            required_bytes: min_index_bytes,
            actual_bytes: total_size,
        });
    }

    Ok(RpfHeader {
        version,
        entry_count,
        names_size,
        encryption_tag,
        metadata_size,
        flags,
        total_size,
        min_index_bytes,
    })
}

/// Non-extracting RPF9 header inspector for local files of any size (including 100+ GB archives).
///
/// Only reads at most [`RPF9_HEADER_SIZE`] bytes from offset 0 after querying file length.
pub fn inspect_rpf_file(path: &Path) -> Result<RpfHeader, RpfError> {
    let mut file = fs::File::open(path)?;
    let total_size = file.metadata()?.len();
    file.seek(SeekFrom::Start(0))?;

    let to_read = usize::try_from(total_size.min(RPF9_HEADER_SIZE as u64)).unwrap_or(RPF9_HEADER_SIZE);
    let mut buf = vec![0u8; to_read];
    file.read_exact(&mut buf)?;

    inspect_rpf_bytes(&buf, total_size)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootsetEntry {
    pub path: String,
    pub offset: u64,
    pub length: u64,
}

fn is_bootset_candidate(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("rpf") || ext.eq_ignore_ascii_case("config"))
}

fn collect_bootset_entries(
    root: &Path,
    dir: &Path,
    max_header_bytes: u64,
    out: &mut Vec<BootsetEntry>,
) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let meta = entry.metadata()?;
        if meta.is_dir() {
            collect_bootset_entries(root, &path, max_header_bytes, out)?;
        } else if meta.is_file() && is_bootset_candidate(&path) {
            let clamped_len = meta.len().min(max_header_bytes);
            if clamped_len == 0 {
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

/// Scans `game_dir` for `.rpf` and `.config` files (case-insensitive) and returns sorted
/// `bootset.json` range entries with each `length` clamped to the file's actual size.
pub fn generate_bootset(game_dir: &Path, max_header_bytes: u64) -> io::Result<Vec<BootsetEntry>> {
    let mut entries = Vec::new();
    collect_bootset_entries(game_dir, game_dir, max_header_bytes, &mut entries)?;
    entries.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(entries)
}

pub(crate) fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Renders `bootset.json` matching `docs/io_manifest.md`.
pub fn render_bootset_json(entries: &[BootsetEntry]) -> String {
    if entries.is_empty() {
        return "[]\n".to_string();
    }
    let mut out = String::from("[\n");
    for (idx, entry) in entries.iter().enumerate() {
        let comma = if idx + 1 == entries.len() { "" } else { "," };
        out.push_str(&format!(
            "  {{\"path\": \"{}\", \"offset\": {}, \"length\": {}}}{}\n",
            json_escape(&entry.path),
            entry.offset,
            entry.length,
            comma
        ));
    }
    out.push_str("]\n");
    out
}

/// Generates and writes `bootset.json` for `game_dir` to `out_file`.
pub fn write_bootset(
    game_dir: &Path,
    out_file: &Path,
    max_header_bytes: u64,
) -> io::Result<usize> {
    let entries = generate_bootset(game_dir, max_header_bytes)?;
    if let Some(parent) = out_file.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let json = render_bootset_json(&entries);
    fs::write(out_file, json)?;
    Ok(entries.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_temp_dir(label: &str) -> std::path::PathBuf {
        let n = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = env::temp_dir().join(format!(
            "playgta6_rpf9_{label}_{}_{}",
            std::process::id(),
            n
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn make_rpf_header(
        magic: [u8; 4],
        entry_count: u32,
        names_size: u32,
        encryption_tag: u32,
        metadata_size: u32,
        flags: u32,
    ) -> [u8; RPF9_HEADER_SIZE] {
        let mut buf = [0u8; RPF9_HEADER_SIZE];
        buf[0..4].copy_from_slice(&magic);
        buf[4..8].copy_from_slice(&entry_count.to_le_bytes());
        buf[8..12].copy_from_slice(&names_size.to_le_bytes());
        buf[12..16].copy_from_slice(&encryption_tag.to_le_bytes());
        buf[16..20].copy_from_slice(&metadata_size.to_le_bytes());
        buf[20..24].copy_from_slice(&flags.to_le_bytes());
        buf
    }

    #[test]
    fn parses_valid_synthetic_rpf9_header_and_summary() {
        let header = make_rpf_header(RPF9_MAGIC, 128, 4096, 0x4E45_504F, 1024, 0x0000_0003);
        let total_size = 10 * 1024 * 1024;
        let parsed = inspect_rpf_bytes(&header, total_size).unwrap();

        assert_eq!(parsed.version, RpfVersion::Rpf9);
        assert_eq!(parsed.entry_count, 128);
        assert_eq!(parsed.names_size, 4096);
        assert_eq!(parsed.encryption_tag, 0x4E45_504F);
        assert_eq!(parsed.encryption_label(), "open");
        assert_eq!(parsed.metadata_size, 1024);
        assert_eq!(parsed.flags, 0x0000_0003);
        assert_eq!(parsed.total_size, total_size);
        assert_eq!(parsed.min_index_bytes, 24 + 128 * 16 + 4096 + 1024);

        let summary = parsed.format_summary();
        assert!(summary.contains("RPF9"));
        assert!(summary.contains("128"));
        assert!(summary.contains("4096 bytes"));
        assert!(summary.contains("1024 bytes"));
        assert!(summary.contains("0x00000003"));
        assert!(summary.contains("0x4E45504F (open)"));
    }

    #[test]
    fn rejects_rpf8_and_rpf7_with_clear_version_error() {
        let rpf8 = make_rpf_header(RPF8_MAGIC, 10, 128, 0, 0, 0);
        let err8 = inspect_rpf_bytes(&rpf8, 4096).unwrap_err();
        match err8 {
            RpfError::UnsupportedVersion { found, expected } => {
                assert_eq!(found, RpfVersion::Rpf8);
                assert_eq!(expected, RpfVersion::Rpf9);
            }
            other => panic!("expected UnsupportedVersion(Rpf8), got {other:?}"),
        }
        assert!(err8.to_string().contains("RPF8"));
        assert!(err8.to_string().contains("RPF9"));

        let rpf7 = make_rpf_header(RPF7_MAGIC, 10, 128, 0, 0, 0);
        let err7 = inspect_rpf_bytes(&rpf7, 4096).unwrap_err();
        match err7 {
            RpfError::UnsupportedVersion { found, expected } => {
                assert_eq!(found, RpfVersion::Rpf7);
                assert_eq!(expected, RpfVersion::Rpf9);
            }
            other => panic!("expected UnsupportedVersion(Rpf7), got {other:?}"),
        }
        assert!(err7.to_string().contains("RPF7"));
    }

    #[test]
    fn rejects_bad_magic() {
        let bad = make_rpf_header(*b"PK\x03\x04", 0, 0, 0, 0, 0);
        let err = inspect_rpf_bytes(&bad, 1024).unwrap_err();
        assert!(matches!(err, RpfError::InvalidMagic([0x50, 0x4B, 0x03, 0x04])));
        assert!(err.to_string().contains("invalid RPF magic"));
    }

    #[test]
    fn rejects_truncated_header_and_truncated_archive() {
        let short_err = inspect_rpf_bytes(b"RPF9\x01\x00\x00\x00", 8).unwrap_err();
        assert!(matches!(
            short_err,
            RpfError::TruncatedHeader {
                actual: 8,
                expected: RPF9_HEADER_SIZE
            }
        ));

        // Header declares 100 entries (1600 B) + 512 B names + 24 B header = 2136 B,
        // but archive size is only 1000 B.
        let header = make_rpf_header(RPF9_MAGIC, 100, 512, 0, 0, 0);
        let trunc_err = inspect_rpf_bytes(&header, 1000).unwrap_err();
        assert!(matches!(
            trunc_err,
            RpfError::TruncatedArchive {
                required_bytes: 2136,
                actual_bytes: 1000
            }
        ));
    }

    #[test]
    fn inspects_rpf_file_without_reading_payload() {
        let dir = unique_temp_dir("inspect_file");
        let rpf_path = dir.join("test_archive.rpf");

        let header = make_rpf_header(RPF9_MAGIC, 2, 64, 0x0FFF_FFF9, 32, 0x10);
        let mut content = Vec::with_capacity(2048);
        content.extend_from_slice(&header);
        content.extend_from_slice(&vec![0xAA; 2048 - RPF9_HEADER_SIZE]);
        fs::write(&rpf_path, &content).unwrap();

        let parsed = inspect_rpf_file(&rpf_path).unwrap();
        assert_eq!(parsed.version, RpfVersion::Rpf9);
        assert_eq!(parsed.entry_count, 2);
        assert_eq!(parsed.encryption_label(), "aes");
        assert_eq!(parsed.total_size, 2048);

        // Summary must not leak payload bytes.
        assert!(!parsed.format_summary().contains("AAAA"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn generates_bootset_clamping_file_sizes() {
        let dir = unique_temp_dir("bootset");
        let game = dir.join("game");
        let sub = game.join("x64").join("audio");
        fs::create_dir_all(&sub).unwrap();

        // Larger than head_bytes (clamped down to 65536)
        fs::write(game.join("levels.rpf"), vec![0u8; 100_000]).unwrap();
        // Smaller than head_bytes (clamped to real size 576, mixed-case .Config)
        fs::write(game.join("MicrosoftGame.Config"), vec![b'x'; 576]).unwrap();
        // Nested uppercase .RPF smaller than head_bytes
        fs::write(sub.join("sfx.RPF"), vec![1u8; 4096]).unwrap();
        // Empty .rpf should be skipped (0-byte range is invalid)
        fs::write(game.join("empty.rpf"), b"").unwrap();
        // Non-matching files should be ignored
        fs::write(game.join("game.exe"), vec![0u8; 8192]).unwrap();
        fs::write(game.join("readme.txt"), b"hello").unwrap();

        let entries = generate_bootset(&game, DEFAULT_BOOTSET_HEAD_BYTES).unwrap();
        assert_eq!(
            entries,
            vec![
                BootsetEntry {
                    path: "MicrosoftGame.Config".to_string(),
                    offset: 0,
                    length: 576,
                },
                BootsetEntry {
                    path: "levels.rpf".to_string(),
                    offset: 0,
                    length: 65_536,
                },
                BootsetEntry {
                    path: "x64/audio/sfx.RPF".to_string(),
                    offset: 0,
                    length: 4096,
                },
            ]
        );

        let out_file = dir.join("out").join("bootset.json");
        let count = write_bootset(&game, &out_file, DEFAULT_BOOTSET_HEAD_BYTES).unwrap();
        assert_eq!(count, 3);

        let json = fs::read_to_string(&out_file).unwrap();
        assert!(json.starts_with("[\n"));
        assert!(json.ends_with("]\n"));
        assert!(json.contains("{\"path\": \"MicrosoftGame.Config\", \"offset\": 0, \"length\": 576},"));
        assert!(json.contains("{\"path\": \"levels.rpf\", \"offset\": 0, \"length\": 65536},"));
        assert!(json.contains("{\"path\": \"x64/audio/sfx.RPF\", \"offset\": 0, \"length\": 4096}\n"));

        let _ = fs::remove_dir_all(&dir);
    }
}
