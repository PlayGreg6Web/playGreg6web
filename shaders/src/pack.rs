use std::collections::BTreeMap;
use std::fmt::{self, Write as _};

use crate::convert::{ConversionError, ConversionErrorKind, ShaderStage, format_shader_hash};

/// Binary pack file magic (`"WGPK"`).
pub const PACK_MAGIC: [u8; 4] = *b"WGPK";

/// Binary pack format version.
pub const PACK_VERSION: u32 = 1;

/// Fixed pack header size in bytes: magic(4) + version(4) + entry_count(4) + flags(4).
pub const PACK_HEADER_BYTES: usize = 16;

/// Fixed entry table record size in bytes: hash(8) + stage(1) + reserved(3) + offset(4) + length(4).
pub const PACK_ENTRY_BYTES: usize = 20;

/// Shader index ABI version matching `docs/gpu_commands.md`.
pub const SHADER_INDEX_ABI_VERSION: u32 = 1;

/// Input/output entry in a `.wgsl.pack` archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackEntry {
    pub hash: u64,
    pub stage: ShaderStage,
    pub wgsl: String,
}

/// Decoded entry metadata plus payload from `read_pack`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedPackEntry {
    pub hash: u64,
    pub stage: ShaderStage,
    pub offset: u32,
    pub length: u32,
    pub wgsl: String,
}

/// Decoded `.wgsl.pack` container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderPack {
    pub version: u32,
    pub flags: u32,
    pub entries: Vec<DecodedPackEntry>,
}

/// Metadata recorded per shader for `shader-index.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderIndexEntry {
    pub hash: u64,
    pub stage: ShaderStage,
    pub wgsl_path: String,
    pub entry_point: String,
    pub pack_offset: Option<u32>,
    pub pack_length: Option<u32>,
}

/// Record of a single shader that failed SPIR-V -> WGSL conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailedShaderRecord {
    pub source_path: String,
    pub error: ConversionError,
}

/// Summary statistics for `render_blocker_report_md`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BlockerReportInput {
    pub total_files: usize,
    pub converted_count: usize,
    pub vertex_count: usize,
    pub fragment_count: usize,
    pub compute_count: usize,
    pub failures: Vec<FailedShaderRecord>,
}

/// Errors produced when reading or writing `.wgsl.pack` bundles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackError {
    TooShort { actual: usize, expected_min: usize },
    InvalidMagic([u8; 4]),
    UnsupportedVersion(u32),
    InvalidStage { entry_index: usize, stage: u8 },
    OutOfBounds {
        entry_index: usize,
        offset: u32,
        length: u32,
        total_len: usize,
    },
    InvalidUtf8 { entry_index: usize, detail: String },
    Overflow(String),
}

impl fmt::Display for PackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort {
                actual,
                expected_min,
            } => write!(
                f,
                "pack truncated: {actual} bytes (minimum required: {expected_min})"
            ),
            Self::InvalidMagic(magic) => write!(
                f,
                "invalid pack magic {:?} (expected {:?})",
                magic, PACK_MAGIC
            ),
            Self::UnsupportedVersion(v) => {
                write!(f, "unsupported pack version {v} (expected {PACK_VERSION})")
            }
            Self::InvalidStage { entry_index, stage } => {
                write!(f, "invalid stage value {stage} at entry {entry_index}")
            }
            Self::OutOfBounds {
                entry_index,
                offset,
                length,
                total_len,
            } => write!(
                f,
                "entry {entry_index} payload [{offset}..+{length}] exceeds pack bounds (total {total_len} bytes)"
            ),
            Self::InvalidUtf8 {
                entry_index,
                detail,
            } => write!(f, "entry {entry_index} payload is not valid UTF-8: {detail}"),
            Self::Overflow(msg) => write!(f, "pack size overflow: {msg}"),
        }
    }
}

impl std::error::Error for PackError {}

/// Layout result returned by [`write_pack_with_layout`]: raw binary bytes plus `(offset, length)`
/// per input entry in the same order as `entries`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrittenPack {
    pub bytes: Vec<u8>,
    pub ranges: Vec<(u32, u32)>,
}

/// Serialize WGSL shader entries into the compact binary `.wgsl.pack` format.
pub fn write_pack(entries: &[PackEntry], flags: u32) -> Result<Vec<u8>, PackError> {
    Ok(write_pack_with_layout(entries, flags)?.bytes)
}

/// Serialize WGSL shader entries into `.wgsl.pack` and also return each entry's `(offset, length)`.
pub fn write_pack_with_layout(entries: &[PackEntry], flags: u32) -> Result<WrittenPack, PackError> {
    let entry_count = u32::try_from(entries.len())
        .map_err(|_| PackError::Overflow("entry count exceeds u32::MAX".to_string()))?;

    let table_bytes = entries
        .len()
        .checked_mul(PACK_ENTRY_BYTES)
        .ok_or_else(|| PackError::Overflow("entry table byte length overflow".to_string()))?;
    let data_start = PACK_HEADER_BYTES
        .checked_add(table_bytes)
        .ok_or_else(|| PackError::Overflow("header + entry table size overflow".to_string()))?;

    let mut total_payload_bytes: usize = 0;
    for entry in entries {
        total_payload_bytes = total_payload_bytes
            .checked_add(entry.wgsl.len())
            .ok_or_else(|| PackError::Overflow("total payload size overflow".to_string()))?;
    }
    let total_size = data_start
        .checked_add(total_payload_bytes)
        .ok_or_else(|| PackError::Overflow("total pack size overflow".to_string()))?;
    if u32::try_from(total_size).is_err() {
        return Err(PackError::Overflow(
            "pack size exceeds u32::MAX byte offsets".to_string(),
        ));
    }

    let mut out = Vec::with_capacity(total_size);
    // Header (16 bytes)
    out.extend_from_slice(&PACK_MAGIC);
    out.extend_from_slice(&PACK_VERSION.to_le_bytes());
    out.extend_from_slice(&entry_count.to_le_bytes());
    out.extend_from_slice(&flags.to_le_bytes());

    // Entry table (20 bytes per entry)
    let mut ranges = Vec::with_capacity(entries.len());
    let mut cursor = data_start;
    for entry in entries {
        let offset = u32::try_from(cursor)
            .map_err(|_| PackError::Overflow("entry offset exceeds u32::MAX".to_string()))?;
        let length = u32::try_from(entry.wgsl.len())
            .map_err(|_| PackError::Overflow("entry payload exceeds u32::MAX".to_string()))?;

        out.extend_from_slice(&entry.hash.to_le_bytes());
        out.push(entry.stage.to_u8());
        out.extend_from_slice(&[0u8; 3]); // reserved
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&length.to_le_bytes());

        ranges.push((offset, length));
        cursor += entry.wgsl.len();
    }

    // Concatenated UTF-8 WGSL payloads
    for entry in entries {
        out.extend_from_slice(entry.wgsl.as_bytes());
    }

    debug_assert_eq!(out.len(), total_size);
    Ok(WrittenPack { bytes: out, ranges })
}

/// Deserialize and validate a binary `.wgsl.pack` buffer.
pub fn read_pack(bytes: &[u8]) -> Result<ShaderPack, PackError> {
    if bytes.len() < PACK_HEADER_BYTES {
        return Err(PackError::TooShort {
            actual: bytes.len(),
            expected_min: PACK_HEADER_BYTES,
        });
    }

    let magic: [u8; 4] = bytes[0..4].try_into().unwrap();
    if magic != PACK_MAGIC {
        return Err(PackError::InvalidMagic(magic));
    }

    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != PACK_VERSION {
        return Err(PackError::UnsupportedVersion(version));
    }

    let entry_count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let flags = u32::from_le_bytes(bytes[12..16].try_into().unwrap());

    let table_bytes = entry_count
        .checked_mul(PACK_ENTRY_BYTES)
        .ok_or_else(|| PackError::Overflow("entry table byte length overflow".to_string()))?;
    let data_start = PACK_HEADER_BYTES
        .checked_add(table_bytes)
        .ok_or_else(|| PackError::Overflow("header + entry table size overflow".to_string()))?;

    if bytes.len() < data_start {
        return Err(PackError::TooShort {
            actual: bytes.len(),
            expected_min: data_start,
        });
    }

    let mut entries = Vec::with_capacity(entry_count);
    for idx in 0..entry_count {
        let base = PACK_HEADER_BYTES + idx * PACK_ENTRY_BYTES;
        let hash = u64::from_le_bytes(bytes[base..base + 8].try_into().unwrap());
        let stage_raw = bytes[base + 8];
        let stage = ShaderStage::from_u8(stage_raw).ok_or(PackError::InvalidStage {
            entry_index: idx,
            stage: stage_raw,
        })?;
        // bytes[base + 9..base + 12] are reserved for forward compatibility.
        let offset = u32::from_le_bytes(bytes[base + 12..base + 16].try_into().unwrap());
        let length = u32::from_le_bytes(bytes[base + 16..base + 20].try_into().unwrap());

        let start = offset as usize;
        let end = start.checked_add(length as usize).ok_or(PackError::OutOfBounds {
            entry_index: idx,
            offset,
            length,
            total_len: bytes.len(),
        })?;

        if start < data_start || end > bytes.len() {
            return Err(PackError::OutOfBounds {
                entry_index: idx,
                offset,
                length,
                total_len: bytes.len(),
            });
        }

        let wgsl = std::str::from_utf8(&bytes[start..end])
            .map_err(|err| PackError::InvalidUtf8 {
                entry_index: idx,
                detail: err.to_string(),
            })?
            .to_string();

        entries.push(DecodedPackEntry {
            hash,
            stage,
            offset,
            length,
            wgsl,
        });
    }

    Ok(ShaderPack {
        version,
        flags,
        entries,
    })
}

/// Render `shader-index.json` matching `docs/gpu_commands.md` (`abi_version: 1`, `pack`, `shaders`).
pub fn render_shader_index_json(pack_path: Option<&str>, entries: &[ShaderIndexEntry]) -> String {
    // Deterministic ordering by canonical hash key.
    let mut sorted: Vec<&ShaderIndexEntry> = entries.iter().collect();
    sorted.sort_by_key(|e| e.hash);

    let mut out = String::new();
    out.push_str("{\n");
    let _ = writeln!(out, "  \"abi_version\": {SHADER_INDEX_ABI_VERSION},");
    match pack_path {
        Some(path) => {
            out.push_str("  \"pack\": ");
            write_json_string(&mut out, path);
            out.push_str(",\n");
        }
        None => {
            out.push_str("  \"pack\": null,\n");
        }
    }

    if sorted.is_empty() {
        out.push_str("  \"shaders\": {}\n");
    } else {
        out.push_str("  \"shaders\": {\n");
        for (i, entry) in sorted.iter().enumerate() {
            let key = format_shader_hash(entry.hash);
            out.push_str("    ");
            write_json_string(&mut out, &key);
            out.push_str(": {\n");

            out.push_str("      \"stage\": ");
            write_json_string(&mut out, entry.stage.as_str());
            out.push_str(",\n");

            out.push_str("      \"wgsl\": ");
            write_json_string(&mut out, &entry.wgsl_path);
            out.push_str(",\n");

            out.push_str("      \"entry_point\": ");
            write_json_string(&mut out, &entry.entry_point);

            if let (Some(offset), Some(length)) = (entry.pack_offset, entry.pack_length) {
                out.push_str(",\n");
                let _ = writeln!(out, "      \"pack_offset\": {offset},");
                let _ = writeln!(out, "      \"pack_length\": {length}");
            } else {
                out.push('\n');
            }

            if i + 1 < sorted.len() {
                out.push_str("    },\n");
            } else {
                out.push_str("    }\n");
            }
        }
        out.push_str("  }\n");
    }

    out.push_str("}\n");
    out
}

/// Write a properly escaped JSON string literal (including quotes) into `out`.
pub fn write_json_string(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Render a structured Markdown WebGPU blocker report summarizing converted vs failed shaders.
pub fn render_blocker_report_md(input: &BlockerReportInput) -> String {
    let failed_count = input.failures.len();
    let total = input.total_files.max(input.converted_count + failed_count);
    let pass_rate = if total == 0 {
        0.0
    } else {
        (input.converted_count as f64 / total as f64) * 100.0
    };

    let mut out = String::new();
    out.push_str("# SPIR-V -> WGSL WebGPU Blocker Report\n\n");
    out.push_str("## Summary\n\n");
    out.push_str("| Metric | Count |\n");
    out.push_str("| :--- | ---: |\n");
    let _ = writeln!(out, "| Total `.spv` files | {total} |");
    let _ = writeln!(
        out,
        "| Converted to WGSL | {} ({pass_rate:.1}%) |",
        input.converted_count
    );
    let _ = writeln!(out, "| Failed | {failed_count} |");
    let _ = writeln!(out, "| Vertex shaders | {} |", input.vertex_count);
    let _ = writeln!(out, "| Fragment shaders | {} |", input.fragment_count);
    let _ = writeln!(out, "| Compute shaders | {} |", input.compute_count);
    out.push('\n');

    if input.failures.is_empty() {
        out.push_str("## Blockers\n\n");
        out.push_str("No conversion failures detected.\n");
        return out;
    }

    // Group by high-level error kind.
    let mut by_kind: BTreeMap<ConversionErrorKind, usize> = BTreeMap::new();
    for fail in &input.failures {
        *by_kind.entry(fail.error.kind).or_insert(0) += 1;
    }

    out.push_str("## Failures by Error Category\n\n");
    out.push_str("| Category | Count |\n");
    out.push_str("| :--- | ---: |\n");
    for (kind, count) in &by_kind {
        let _ = writeln!(out, "| `{kind}` | {count} |");
    }
    out.push('\n');

    // Group by normalized blocker key, sorted by descending count then alphabetically.
    struct BlockerGroup<'a> {
        kind: ConversionErrorKind,
        count: usize,
        keywords: Vec<String>,
        samples: Vec<&'a FailedShaderRecord>,
    }

    let mut grouped: BTreeMap<&str, BlockerGroup<'_>> = BTreeMap::new();
    for fail in &input.failures {
        let entry = grouped
            .entry(fail.error.blocker_key.as_str())
            .or_insert_with(|| BlockerGroup {
                kind: fail.error.kind,
                count: 0,
                keywords: Vec::new(),
                samples: Vec::new(),
            });
        entry.count += 1;
        for kw in &fail.error.keywords {
            if !entry.keywords.iter().any(|existing| existing == kw) {
                entry.keywords.push(kw.clone());
            }
        }
        if entry.samples.len() < 5 {
            entry.samples.push(fail);
        }
    }

    let mut ranked: Vec<(&str, BlockerGroup<'_>)> = grouped.into_iter().collect();
    ranked.sort_by(|(k1, g1), (k2, g2)| g2.count.cmp(&g1.count).then_with(|| k1.cmp(k2)));

    out.push_str("## Grouped WebGPU Blockers\n\n");
    out.push_str("| Blocker | Category | Count | Keywords | Sample Shaders |\n");
    out.push_str("| :--- | :--- | ---: | :--- | :--- |\n");
    for (key, group) in &ranked {
        let kw_cell = if group.keywords.is_empty() {
            "-".to_string()
        } else {
            group
                .keywords
                .iter()
                .map(|k| format!("`{}`", escape_md_cell(k)))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let samples_cell = group
            .samples
            .iter()
            .map(|s| format!("`{}`", escape_md_cell(&s.source_path)))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(
            out,
            "| `{}` | `{}` | {} | {} | {} |",
            escape_md_cell(key),
            group.kind,
            group.count,
            kw_cell,
            samples_cell
        );
    }
    out.push('\n');

    out.push_str("## Failure Details\n\n");
    for fail in &input.failures {
        let first_line = fail
            .error
            .detail
            .lines()
            .find(|l| !l.trim().is_empty())
            .unwrap_or(&fail.error.detail)
            .trim();
        let _ = writeln!(
            out,
            "- `{}` — `{}` (`{}`): {}",
            escape_md_cell(&fail.source_path),
            fail.error.kind,
            escape_md_cell(&fail.error.blocker_key),
            first_line
        );
    }

    out
}

fn escape_md_cell(text: &str) -> String {
    text.replace('|', "\\|").replace(['\r', '\n'], " ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::{
        convert_spirv_bytes, format_wgsl_filename,
        test_spirv::{
            build_minimal_compute_spv, build_minimal_fragment_spv, build_minimal_vertex_spv,
            build_unsupported_capability_spv, build_unsupported_extension_spv,
        },
    };

    #[test]
    fn pack_round_trip_with_converted_spirv_shaders() {
        let compute = convert_spirv_bytes(&build_minimal_compute_spv()).unwrap();
        let vertex = convert_spirv_bytes(&build_minimal_vertex_spv()).unwrap();
        let fragment = convert_spirv_bytes(&build_minimal_fragment_spv()).unwrap();

        let pack_entries = vec![
            PackEntry {
                hash: compute.hash,
                stage: compute.stage,
                wgsl: compute.wgsl.clone(),
            },
            PackEntry {
                hash: vertex.hash,
                stage: vertex.stage,
                wgsl: vertex.wgsl.clone(),
            },
            PackEntry {
                hash: fragment.hash,
                stage: fragment.stage,
                wgsl: fragment.wgsl.clone(),
            },
        ];

        let written = write_pack_with_layout(&pack_entries, 0).expect("pack write should succeed");
        assert_eq!(&written.bytes[0..4], b"WGPK");
        assert_eq!(
            u32::from_le_bytes(written.bytes[4..8].try_into().unwrap()),
            1
        );
        assert_eq!(
            u32::from_le_bytes(written.bytes[8..12].try_into().unwrap()),
            3
        );
        assert_eq!(
            u32::from_le_bytes(written.bytes[12..16].try_into().unwrap()),
            0
        );

        let expected_data_start = (PACK_HEADER_BYTES + 3 * PACK_ENTRY_BYTES) as u32;
        assert_eq!(written.ranges[0].0, expected_data_start);

        let decoded = read_pack(&written.bytes).expect("pack read should succeed");
        assert_eq!(decoded.version, PACK_VERSION);
        assert_eq!(decoded.flags, 0);
        assert_eq!(decoded.entries.len(), 3);

        for (i, original) in pack_entries.iter().enumerate() {
            let entry = &decoded.entries[i];
            assert_eq!(entry.hash, original.hash);
            assert_eq!(entry.stage, original.stage);
            assert_eq!(entry.wgsl, original.wgsl);
            assert_eq!((entry.offset, entry.length), written.ranges[i]);
        }

        // Verify shader-index.json output matches docs/gpu_commands.md schema.
        let index_entries = vec![
            ShaderIndexEntry {
                hash: compute.hash,
                stage: compute.stage,
                wgsl_path: format!("shaders/{}", format_wgsl_filename(compute.hash)),
                entry_point: compute.entry_point.clone(),
                pack_offset: Some(written.ranges[0].0),
                pack_length: Some(written.ranges[0].1),
            },
            ShaderIndexEntry {
                hash: vertex.hash,
                stage: vertex.stage,
                wgsl_path: format!("shaders/{}", format_wgsl_filename(vertex.hash)),
                entry_point: vertex.entry_point.clone(),
                pack_offset: Some(written.ranges[1].0),
                pack_length: Some(written.ranges[1].1),
            },
        ];

        let json = render_shader_index_json(Some("shaders.wgsl.pack"), &index_entries);
        assert!(json.contains("\"abi_version\": 1"));
        assert!(json.contains("\"pack\": \"shaders.wgsl.pack\""));
        assert!(json.contains(&format_shader_hash(compute.hash)));
        assert!(json.contains(&format_shader_hash(vertex.hash)));
        assert!(json.contains("\"stage\": \"compute\""));
        assert!(json.contains("\"stage\": \"vertex\""));
        assert!(json.contains("\"pack_offset\":"));
        assert!(json.contains("\"pack_length\":"));
    }

    #[test]
    fn read_pack_validates_magic_version_bounds_and_utf8() {
        let valid = write_pack(
            &[PackEntry {
                hash: 0x1122_3344_5566_7788,
                stage: ShaderStage::Compute,
                wgsl: "@compute @workgroup_size(1) fn main() {}".to_string(),
            }],
            0,
        )
        .unwrap();

        // Too short
        assert!(matches!(
            read_pack(&valid[..8]),
            Err(PackError::TooShort { .. })
        ));

        // Bad magic
        let mut bad_magic = valid.clone();
        bad_magic[0..4].copy_from_slice(b"NOPE");
        assert!(matches!(
            read_pack(&bad_magic),
            Err(PackError::InvalidMagic(_))
        ));

        // Bad version
        let mut bad_version = valid.clone();
        bad_version[4..8].copy_from_slice(&99u32.to_le_bytes());
        assert_eq!(
            read_pack(&bad_version),
            Err(PackError::UnsupportedVersion(99))
        );

        // Invalid stage
        let mut bad_stage = valid.clone();
        bad_stage[PACK_HEADER_BYTES + 8] = 9;
        assert_eq!(
            read_pack(&bad_stage),
            Err(PackError::InvalidStage {
                entry_index: 0,
                stage: 9
            })
        );

        // Out-of-bounds offset
        let mut oob = valid.clone();
        oob[PACK_HEADER_BYTES + 12..PACK_HEADER_BYTES + 16]
            .copy_from_slice(&9999u32.to_le_bytes());
        assert!(matches!(read_pack(&oob), Err(PackError::OutOfBounds { .. })));

        // Offset pointing inside header/table
        let mut overlap = valid.clone();
        overlap[PACK_HEADER_BYTES + 12..PACK_HEADER_BYTES + 16]
            .copy_from_slice(&4u32.to_le_bytes());
        assert!(matches!(
            read_pack(&overlap),
            Err(PackError::OutOfBounds { .. })
        ));

        // Invalid UTF-8 payload
        let mut bad_utf8 = valid.clone();
        let payload_start = PACK_HEADER_BYTES + PACK_ENTRY_BYTES;
        bad_utf8[payload_start] = 0xff;
        assert!(matches!(
            read_pack(&bad_utf8),
            Err(PackError::InvalidUtf8 { entry_index: 0, .. })
        ));
    }

    #[test]
    fn json_string_escaping_handles_special_chars() {
        let mut buf = String::new();
        write_json_string(&mut buf, "a\"b\\c\nd\te\u{0007}");
        assert_eq!(buf, "\"a\\\"b\\\\c\\nd\\te\\u0007\"");

        let json = render_shader_index_json(None, &[]);
        assert!(json.contains("\"abi_version\": 1"));
        assert!(json.contains("\"pack\": null"));
        assert!(json.contains("\"shaders\": {}"));
    }

    #[test]
    fn blocker_report_groups_failures_and_summarizes_counts() {
        let err_ext = convert_spirv_bytes(&build_unsupported_extension_spv()).unwrap_err();
        let err_cap = convert_spirv_bytes(&build_unsupported_capability_spv(true)).unwrap_err();
        let err_hdr = convert_spirv_bytes(&[1, 2, 3]).unwrap_err();

        let report = render_blocker_report_md(&BlockerReportInput {
            total_files: 5,
            converted_count: 2,
            vertex_count: 1,
            fragment_count: 0,
            compute_count: 1,
            failures: vec![
                FailedShaderRecord {
                    source_path: "rt/ray1.spv".to_string(),
                    error: err_ext.clone(),
                },
                FailedShaderRecord {
                    source_path: "rt/ray2.spv".to_string(),
                    error: err_ext,
                },
                FailedShaderRecord {
                    source_path: "rt/query.spv".to_string(),
                    error: err_cap,
                },
                FailedShaderRecord {
                    source_path: "corrupt.spv".to_string(),
                    error: err_hdr,
                },
            ],
        });

        assert!(report.contains("# SPIR-V -> WGSL WebGPU Blocker Report"));
        assert!(report.contains("SPV_KHR_ray_tracing"));
        assert!(report.contains("InvalidSpirvHeader"));
        assert!(report.contains("SpirvParseError"));
        assert!(report.contains("rt/ray1.spv"));
    }
}
