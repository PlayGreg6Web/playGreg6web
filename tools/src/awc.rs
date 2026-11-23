//! Read-only RAGE `.awc` streaming audio bank header inspector (W28).
//!
//! This module parses only the container header and the small per-stream tables of an audio
//! wave container. It never reads, copies, decodes or prints audio payload bytes, never touches
//! key material, and contains no decryption code.
//!
//! # Provenance and confidence
//!
//! There was no real bank available when this was written. The layout below is reconstructed
//! from memory of the publicly documented `AWC` layout used by earlier RAGE titles (community
//! modding/research notes) and has NOT been verified against any file of the newest title. The
//! format revision used by that title is not publicly documented, so every field that is not
//! recognised is reported as unrecognised together with its raw numeric value instead of being
//! guessed at. Confidence is given per part:
//!
//! * `0x00` magic: `ADAT` = little-endian bank, `TADA` = big-endian bank. (high)
//! * `0x04` `u16` version, `0x06` `u16` flags, `0x08` `u32` stream count, `0x0C` `u32` data
//!   offset (the end of the header region). (medium: the version/flags split and the
//!   semantics of the data offset are the least certain part of the header)
//! * Flags (see [`FLAG_TABLE`]): chunk-indices (`0x0002`), multi-channel (`0x0004`) (low
//!   confidence on the exact bit positions), wide chunk descriptors (`0x0008`) (a guess made
//!   for this tool, see below). Any other set bit is reported as an unknown flag bit.
//! * Tables, directly after the header, in this order: optional `u16[stream_count]` chunk
//!   index table (only if the chunk-indices flag is set), `u32[stream_count]` stream id table
//!   (low 29 bits = stream id, high 3 bits = chunk count), then the chunk descriptors for all
//!   streams back to back. (medium)
//! * Chunk descriptor, narrow (default): one `u64` packed as tag = bits 0..8, size = bits
//!   8..36 (28 bits), offset = bits 36..64 (28 bits); offsets are taken as absolute from the
//!   start of the file. (medium)
//! * Chunk descriptor, wide (flag `0x0008`): this tool assumes 16 bytes, `u32` tag word (low
//!   byte = tag), `u32` size, `u64` offset. This is an assumption made to cover banks larger
//!   than the 28-bit limit; output for banks using it carries a warning. (low)
//! * Chunk tags: see [`CHUNK_TAG_TABLE`]. Unlisted tags are shown as `unknown(0xNN)`.
//! * Format chunk (tag `0xFA`, 20 bytes): `u32` sample count, `i32` loop point, `u16` sample
//!   rate, `i16` headroom, `u16` loop begin, `u16` loop end, `u16` play end, `u8` play begin,
//!   `u8` codec id. (medium)
//! * Stream-format chunk (tag `0x48`): `u32` block count, `u32` block size, `u32` channel
//!   count, then 16 bytes per channel: `u32` id, `u32` sample count, `i16` headroom, `u16`
//!   sample rate, `u8` codec id, 3 reserved bytes. (low)
//! * Codec ids: only ids with public documentation that this tool's author was confident about
//!   are listed in [`CODEC_TABLE`]. XMA2, Opus, Vorbis, MP3 and others are expected to exist in
//!   newer revisions but their numeric ids are not listed until verified; they print as
//!   `unknown(N)`. Extending the table is a one-line change.
//!
//! # Read bound
//!
//! At most [`MAX_READ_BYTES`] (16 MiB) are read from any file regardless of its size: the
//! 16-byte header, the three tables (at most [`MAX_STREAMS`] streams, so at most about 7.8 MiB),
//! and for the first [`MAX_DETAIL_STREAMS`] streams the format chunk (20 bytes) and the
//! stream-format chunk (at most 1036 bytes). Chunk ranges that overlap a data chunk are never
//! read. A read budget enforces the limit and returns [`AwcError::ReadBudgetExceeded`] if a
//! corrupt file would exceed it. Allocations are sized only after bounds checks against both
//! the file length and the caps.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

use crate::rpf9::json_escape;
use crate::scan::format_bytes;

const ABI_VERSION: u32 = 1;

pub const AWC_MAGIC_LE: [u8; 4] = *b"ADAT";
pub const AWC_MAGIC_BE: [u8; 4] = *b"TADA";

/// Fixed header: magic, version, flags, stream count, data offset.
pub const AWC_HEADER_SIZE: u64 = 16;

/// Hard cap on declared streams; real banks hold far fewer.
pub const MAX_STREAMS: u64 = 65_536;
/// Streams whose format chunks are read; later streams only get their chunk list.
pub const MAX_DETAIL_STREAMS: usize = 4_096;
/// Maximum channels accepted in a stream-format chunk.
pub const MAX_CHANNELS: u32 = 64;
/// Absolute ceiling on bytes read from a bank.
pub const MAX_READ_BYTES: u64 = 16 * 1024 * 1024;
/// Maximum warnings stored; further ones are only counted.
pub const MAX_WARNINGS: usize = 64;
/// Default number of streams listed in output.
pub const DEFAULT_MAX_LISTED_STREAMS: usize = 32;

const FORMAT_CHUNK_SIZE: u64 = 20;
const STREAM_FORMAT_HEADER_SIZE: u64 = 12;
const STREAM_FORMAT_CHANNEL_SIZE: u64 = 16;
const STREAM_FORMAT_MAX_READ: u64 =
    STREAM_FORMAT_HEADER_SIZE + STREAM_FORMAT_CHANNEL_SIZE * MAX_CHANNELS as u64;

pub const TAG_DATA: u8 = 0x55;
pub const TAG_FORMAT: u8 = 0xFA;
pub const TAG_STREAM_FORMAT: u8 = 0x48;

pub const FLAG_CHUNK_INDICES: u16 = 0x0002;
pub const FLAG_MULTI_CHANNEL: u16 = 0x0004;
pub const FLAG_WIDE_DESCRIPTORS: u16 = 0x0008;

/// Versions this tool knows how to interpret. Anything else is reported as unrecognised and
/// the tables are not interpreted.
pub const KNOWN_VERSIONS: &[u16] = &[1];

/// Flag bits this tool understands (single table; extend here).
pub const FLAG_TABLE: &[(u16, &str)] = &[
    (FLAG_CHUNK_INDICES, "chunk_indices"),
    (FLAG_MULTI_CHANNEL, "multi_channel"),
    (FLAG_WIDE_DESCRIPTORS, "wide_descriptors"),
];

/// Chunk tags this tool names (single table; extend here).
pub const CHUNK_TAG_TABLE: &[(u8, &str)] = &[
    (TAG_DATA, "data"),
    (TAG_FORMAT, "format"),
    (TAG_STREAM_FORMAT, "stream_format"),
    (0x5B, "seek_table"),
    (0x5A, "markers"),
    (0x5C, "animation"),
    (0x36, "peak"),
    (0x68, "midi"),
    (0xD9, "granular_grains"),
    (0xBD, "granular_loops"),
];

/// Whether the web audio path (`web/workers/audio_mixer.js`) can decode a codec today.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebSupport {
    Yes,
    /// Not decodable yet (for example XMA2); no listed codec uses it until its id is verified.
    #[allow(dead_code)]
    No,
    Unknown,
}

impl WebSupport {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Yes => "yes",
            Self::No => "no",
            Self::Unknown => "unknown",
        }
    }
}

/// Codec ids this tool names (single table; extend here once an id is verified).
/// Entries: id, name, web decodable today.
pub const CODEC_TABLE: &[(u8, &str, WebSupport)] = &[
    (0x00, "PCM16", WebSupport::Yes),
    (0x04, "IMA-ADPCM", WebSupport::Yes),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endian {
    Little,
    Big,
}

impl Endian {
    fn u16(self, b: &[u8]) -> Option<u16> {
        let a: [u8; 2] = b.get(..2)?.try_into().ok()?;
        Some(match self {
            Self::Little => u16::from_le_bytes(a),
            Self::Big => u16::from_be_bytes(a),
        })
    }
    fn u32(self, b: &[u8]) -> Option<u32> {
        let a: [u8; 4] = b.get(..4)?.try_into().ok()?;
        Some(match self {
            Self::Little => u32::from_le_bytes(a),
            Self::Big => u32::from_be_bytes(a),
        })
    }
    fn u64(self, b: &[u8]) -> Option<u64> {
        let a: [u8; 8] = b.get(..8)?.try_into().ok()?;
        Some(match self {
            Self::Little => u64::from_le_bytes(a),
            Self::Big => u64::from_be_bytes(a),
        })
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Little => "little",
            Self::Big => "big",
        }
    }
}

/// Codec id resolved through [`CODEC_TABLE`]; unknown ids keep their raw value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Codec {
    pub id: u8,
}

impl Codec {
    pub fn entry(self) -> Option<&'static (u8, &'static str, WebSupport)> {
        CODEC_TABLE.iter().find(|e| e.0 == self.id)
    }
    pub fn is_known(self) -> bool {
        self.entry().is_some()
    }
    pub fn web_support(self) -> WebSupport {
        self.entry().map_or(WebSupport::Unknown, |e| e.2)
    }
    /// Label; PCM16 carries the bank endianness.
    pub fn label(self, endian: Endian) -> String {
        match self.entry() {
            Some((_, "PCM16", _)) => match endian {
                Endian::Little => "PCM16-LE".to_string(),
                Endian::Big => "PCM16-BE".to_string(),
            },
            Some((_, name, _)) => (*name).to_string(),
            None => format!("unknown({})", self.id),
        }
    }
}

pub fn chunk_tag_name(tag: u8) -> Option<&'static str> {
    CHUNK_TAG_TABLE.iter().find(|e| e.0 == tag).map(|e| e.1)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkInfo {
    pub tag: u8,
    pub offset: u64,
    pub size: u64,
}

impl ChunkInfo {
    pub fn tag_name(&self) -> Option<&'static str> {
        chunk_tag_name(self.tag)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormatInfo {
    pub samples: u32,
    pub loop_point: i32,
    pub sample_rate: u16,
    pub headroom: i16,
    pub loop_begin: u16,
    pub loop_end: u16,
    pub play_end: u16,
    pub play_begin: u8,
    pub codec: Codec,
}

impl FormatInfo {
    pub fn duration_ms(&self) -> Option<u64> {
        duration_ms(self.samples, self.sample_rate)
    }
}

fn duration_ms(samples: u32, rate: u16) -> Option<u64> {
    if rate == 0 {
        None
    } else {
        Some(u64::from(samples) * 1000 / u64::from(rate))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelFormat {
    pub id: u32,
    pub samples: u32,
    pub headroom: i16,
    pub sample_rate: u16,
    pub codec: Codec,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamFormatInfo {
    pub block_count: u32,
    pub block_size: u32,
    pub channel_count: u32,
    pub channels: Vec<ChannelFormat>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamInfo {
    pub id: u32,
    pub chunk_index: Option<u16>,
    pub chunks: Vec<ChunkInfo>,
    pub format: Option<FormatInfo>,
    pub stream_format: Option<StreamFormatInfo>,
}

impl StreamInfo {
    pub fn data_bytes(&self) -> u64 {
        self.chunks
            .iter()
            .filter(|c| c.tag == TAG_DATA)
            .fold(0u64, |a, c| a.saturating_add(c.size))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AwcReport {
    pub file_name: String,
    pub file_size: u64,
    pub endian: Endian,
    pub version: u16,
    pub version_known: bool,
    pub flags: u16,
    pub unknown_flag_bits: u16,
    pub stream_count: u32,
    pub data_offset: u32,
    /// False when the version is unrecognised and the tables were not interpreted.
    pub tables_parsed: bool,
    pub table_end: u64,
    pub streams: Vec<StreamInfo>,
    pub streams_inspected: usize,
    pub chunk_count: u64,
    /// Sum of data-chunk sizes per codec label (deterministic order).
    pub payload_bytes_by_codec: BTreeMap<String, u64>,
    pub bytes_read: u64,
    pub warnings: Vec<String>,
    pub warnings_suppressed: u64,
}

impl AwcReport {
    pub fn flag_names(&self) -> Vec<&'static str> {
        FLAG_TABLE
            .iter()
            .filter(|(bit, _)| self.flags & bit != 0)
            .map(|(_, n)| *n)
            .collect()
    }
}

#[derive(Debug)]
pub enum AwcError {
    Io(io::Error),
    TruncatedHeader {
        actual: u64,
        expected: u64,
    },
    InvalidMagic {
        found: [u8; 4],
        looks_encrypted: bool,
    },
    TooManyStreams {
        count: u64,
        max: u64,
    },
    TruncatedTable {
        table: &'static str,
        required: u64,
        actual: u64,
    },
    SizeOverflow,
    ReadBudgetExceeded {
        limit: u64,
    },
}

impl fmt::Display for AwcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error reading AWC bank: {err}"),
            Self::TruncatedHeader { actual, expected } => write!(
                f,
                "truncated AWC header: got {actual} bytes, need at least {expected} bytes"
            ),
            Self::InvalidMagic {
                found,
                looks_encrypted,
            } => {
                write!(
                    f,
                    "not a plain AWC header: magic {:02X} {:02X} {:02X} {:02X} (expected \"ADAT\" or \"TADA\")",
                    found[0], found[1], found[2], found[3]
                )?;
                if *looks_encrypted {
                    write!(
                        f,
                        "; the leading bytes look encrypted or compressed, which this tool does not handle"
                    )?;
                }
                Ok(())
            }
            Self::TooManyStreams { count, max } => write!(
                f,
                "implausible AWC stream count {count} (limit {max}); header is corrupt or an unrecognised revision"
            ),
            Self::TruncatedTable {
                table,
                required,
                actual,
            } => write!(
                f,
                "truncated AWC {table}: needs {required} bytes but file is {actual} bytes"
            ),
            Self::SizeOverflow => write!(f, "AWC table sizes overflow 64-bit bounds"),
            Self::ReadBudgetExceeded { limit } => {
                write!(f, "AWC inspection would exceed the {limit} byte read limit")
            }
        }
    }
}

impl std::error::Error for AwcError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for AwcError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

struct Src<'a, R: Read + Seek> {
    reader: &'a mut R,
    file_size: u64,
    limit: u64,
    read: u64,
}

impl<R: Read + Seek> Src<'_, R> {
    /// Positioned bounded read; the allocation happens only after all checks pass.
    fn read_at(&mut self, offset: u64, len: u64, what: &'static str) -> Result<Vec<u8>, AwcError> {
        let end = offset.checked_add(len).ok_or(AwcError::SizeOverflow)?;
        if end > self.file_size {
            return Err(AwcError::TruncatedTable {
                table: what,
                required: end,
                actual: self.file_size,
            });
        }
        let total = self.read.checked_add(len).ok_or(AwcError::SizeOverflow)?;
        if total > self.limit {
            return Err(AwcError::ReadBudgetExceeded { limit: self.limit });
        }
        let n = usize::try_from(len).map_err(|_| AwcError::SizeOverflow)?;
        let mut buf = vec![0u8; n];
        self.reader.seek(SeekFrom::Start(offset))?;
        self.reader.read_exact(&mut buf)?;
        self.read = total;
        Ok(buf)
    }
}

struct Warnings {
    list: Vec<String>,
    suppressed: u64,
}

impl Warnings {
    fn push(&mut self, msg: String) {
        if self.list.len() < MAX_WARNINGS {
            self.list.push(msg);
        } else {
            self.suppressed += 1;
        }
    }
}

fn looks_encrypted(head: &[u8]) -> bool {
    if head.len() < 16 {
        return false;
    }
    let mut seen = [false; 256];
    let mut distinct = 0;
    for &b in &head[..16] {
        if !seen[b as usize] {
            seen[b as usize] = true;
            distinct += 1;
        }
    }
    distinct >= 12
}

/// Core inspector over any `Read + Seek` source, bounded by [`MAX_READ_BYTES`].
pub fn inspect_awc_reader<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
    file_name: &str,
) -> Result<AwcReport, AwcError> {
    inspect_awc_reader_with_limit(reader, file_size, file_name, MAX_READ_BYTES)
}

fn inspect_awc_reader_with_limit<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
    file_name: &str,
    limit: u64,
) -> Result<AwcReport, AwcError> {
    let mut src = Src {
        reader,
        file_size,
        limit,
        read: 0,
    };
    let head_len = file_size.min(AWC_HEADER_SIZE);
    let head = src.read_at(0, head_len, "header")?;
    if head.len() < 4 {
        return Err(AwcError::TruncatedHeader {
            actual: head_len,
            expected: AWC_HEADER_SIZE,
        });
    }
    let magic = [head[0], head[1], head[2], head[3]];
    let endian = if magic == AWC_MAGIC_LE {
        Endian::Little
    } else if magic == AWC_MAGIC_BE {
        Endian::Big
    } else {
        return Err(AwcError::InvalidMagic {
            found: magic,
            looks_encrypted: looks_encrypted(&head),
        });
    };
    if head_len < AWC_HEADER_SIZE {
        return Err(AwcError::TruncatedHeader {
            actual: head_len,
            expected: AWC_HEADER_SIZE,
        });
    }
    let trunc = || AwcError::TruncatedHeader {
        actual: head_len,
        expected: AWC_HEADER_SIZE,
    };
    let version = endian.u16(&head[4..]).ok_or_else(trunc)?;
    let flags = endian.u16(&head[6..]).ok_or_else(trunc)?;
    let stream_count = endian.u32(&head[8..]).ok_or_else(trunc)?;
    let data_offset = endian.u32(&head[12..]).ok_or_else(trunc)?;

    let known_bits = FLAG_TABLE.iter().fold(0u16, |a, (b, _)| a | b);
    let unknown_flag_bits = flags & !known_bits;
    let version_known = KNOWN_VERSIONS.contains(&version);
    let mut warn = Warnings {
        list: Vec::new(),
        suppressed: 0,
    };
    if !version_known {
        warn.push(format!(
            "unrecognised AWC version {version}; stream tables were not interpreted"
        ));
    }
    if unknown_flag_bits != 0 {
        warn.push(format!(
            "unrecognised flag bits 0x{unknown_flag_bits:04X} (raw flags 0x{flags:04X})"
        ));
    }

    let mut report = AwcReport {
        file_name: file_name.to_string(),
        file_size,
        endian,
        version,
        version_known,
        flags,
        unknown_flag_bits,
        stream_count,
        data_offset,
        tables_parsed: false,
        table_end: AWC_HEADER_SIZE,
        streams: Vec::new(),
        streams_inspected: 0,
        chunk_count: 0,
        payload_bytes_by_codec: BTreeMap::new(),
        bytes_read: 0,
        warnings: Vec::new(),
        warnings_suppressed: 0,
    };

    if !version_known {
        finish(&mut report, warn, src.read);
        return Ok(report);
    }

    let n = u64::from(stream_count);
    if n > MAX_STREAMS {
        return Err(AwcError::TooManyStreams {
            count: n,
            max: MAX_STREAMS,
        });
    }
    let has_indices = flags & FLAG_CHUNK_INDICES != 0;
    let wide = flags & FLAG_WIDE_DESCRIPTORS != 0;
    if wide {
        warn.push(
            "wide chunk descriptors use an assumed layout (unverified); chunk values may be wrong"
                .to_string(),
        );
    }
    let desc_size: u64 = if wide { 16 } else { 8 };

    let mut pos = AWC_HEADER_SIZE;
    let indices = if has_indices {
        let b = src.read_at(pos, n * 2, "chunk index table")?;
        pos += n * 2;
        Some(b)
    } else {
        None
    };
    let id_bytes = src.read_at(pos, n * 4, "stream id table")?;
    pos += n * 4;

    let mut ids = Vec::with_capacity(id_bytes.len() / 4);
    let mut total_chunks: u64 = 0;
    for k in 0..n as usize {
        let v = id_bytes
            .get(k * 4..)
            .and_then(|c| endian.u32(c))
            .ok_or(AwcError::SizeOverflow)?;
        let count = v >> 29;
        total_chunks += u64::from(count);
        ids.push((v & 0x1FFF_FFFF, count as usize));
    }
    let desc_len = total_chunks
        .checked_mul(desc_size)
        .ok_or(AwcError::SizeOverflow)?;
    let desc_bytes = src.read_at(pos, desc_len, "chunk descriptor table")?;
    pos += desc_len;
    report.table_end = pos;
    report.tables_parsed = true;
    report.chunk_count = total_chunks;
    if u64::from(data_offset) < pos {
        warn.push(format!(
            "data offset {data_offset} lies before the end of the header tables ({pos})"
        ));
    }
    if n == 0 {
        warn.push("bank declares zero streams".to_string());
    }

    // Duplicate ids.
    let mut seen: BTreeMap<u32, u32> = BTreeMap::new();
    for (id, _) in &ids {
        *seen.entry(*id).or_insert(0) += 1;
    }
    for (id, c) in &seen {
        if *c > 1 {
            warn.push(format!("duplicate stream id 0x{id:08X} appears {c} times"));
        }
    }

    // Build streams.
    let mut cursor = 0usize;
    for (i, (id, count)) in ids.iter().enumerate() {
        let mut chunks = Vec::with_capacity(*count);
        for _ in 0..*count {
            let d = desc_bytes
                .get(cursor * desc_size as usize..)
                .ok_or(AwcError::SizeOverflow)?;
            cursor += 1;
            let chunk = if wide {
                let tagword = endian.u32(d).ok_or(AwcError::SizeOverflow)?;
                ChunkInfo {
                    tag: (tagword & 0xFF) as u8,
                    size: u64::from(endian.u32(&d[4..]).ok_or(AwcError::SizeOverflow)?),
                    offset: endian.u64(&d[8..]).ok_or(AwcError::SizeOverflow)?,
                }
            } else {
                let v = endian.u64(d).ok_or(AwcError::SizeOverflow)?;
                ChunkInfo {
                    tag: (v & 0xFF) as u8,
                    size: (v >> 8) & 0x0FFF_FFFF,
                    offset: (v >> 36) & 0x0FFF_FFFF,
                }
            };
            chunks.push(chunk);
        }
        let chunk_index = indices.as_ref().and_then(|b| endian.u16(b.get(i * 2..)?));
        report.streams.push(StreamInfo {
            id: *id,
            chunk_index,
            chunks,
            format: None,
            stream_format: None,
        });
    }

    // Detail pass: only small format chunks are read, never data chunks.
    let detailed = report.streams.len().min(MAX_DETAIL_STREAMS);
    if report.streams.len() > detailed {
        warn.push(format!(
            "only the first {detailed} of {} streams had their format chunks read",
            report.streams.len()
        ));
    }
    for stream in report.streams.iter_mut().take(detailed) {
        inspect_stream(&mut src, endian, stream, &mut warn)?;
    }
    report.streams_inspected = detailed;

    // Payload totals.
    let mut totals: BTreeMap<String, u64> = BTreeMap::new();
    for s in &report.streams[..detailed] {
        let bytes = s.data_bytes();
        if bytes == 0 {
            continue;
        }
        let label = if let Some(f) = &s.format {
            f.codec.label(endian)
        } else if let Some(sf) = &s.stream_format {
            let mut labels: Vec<String> =
                sf.channels.iter().map(|c| c.codec.label(endian)).collect();
            labels.dedup();
            match labels.len() {
                0 => "no format information".to_string(),
                1 => labels.remove(0),
                _ => "mixed".to_string(),
            }
        } else {
            "no format information".to_string()
        };
        let e = totals.entry(label).or_insert(0);
        *e = e.saturating_add(bytes);
    }
    report.payload_bytes_by_codec = totals;

    finish(&mut report, warn, src.read);
    Ok(report)
}

fn finish(report: &mut AwcReport, warn: Warnings, bytes_read: u64) {
    report.warnings = warn.list;
    report.warnings_suppressed = warn.suppressed;
    report.bytes_read = bytes_read;
}

fn overlaps(a_off: u64, a_len: u64, b_off: u64, b_len: u64) -> bool {
    a_off < b_off.saturating_add(b_len) && b_off < a_off.saturating_add(a_len)
}

fn inspect_stream<R: Read + Seek>(
    src: &mut Src<'_, R>,
    endian: Endian,
    stream: &mut StreamInfo,
    warn: &mut Warnings,
) -> Result<(), AwcError> {
    let id = stream.id;
    let file_size = src.file_size;
    for c in &stream.chunks {
        let end = c.offset.saturating_add(c.size);
        if end > file_size {
            warn.push(format!(
                "stream 0x{id:08X} chunk 0x{:02X} range {}..{} exceeds file size {file_size}",
                c.tag, c.offset, end
            ));
        }
    }
    let data_ranges: Vec<(u64, u64)> = stream
        .chunks
        .iter()
        .filter(|c| c.tag == TAG_DATA)
        .map(|c| (c.offset, c.size))
        .collect();
    let readable = |c: &ChunkInfo| -> bool {
        c.offset.saturating_add(c.size) <= file_size
            && !data_ranges
                .iter()
                .any(|(o, l)| overlaps(c.offset, c.size, *o, *l))
    };

    if let Some(c) = stream.chunks.iter().find(|c| c.tag == TAG_FORMAT).cloned() {
        if c.size < FORMAT_CHUNK_SIZE {
            warn.push(format!(
                "stream 0x{id:08X} format chunk is {} bytes, expected at least {FORMAT_CHUNK_SIZE}",
                c.size
            ));
        } else if !readable(&c) {
            warn.push(format!(
                "stream 0x{id:08X} format chunk is out of bounds or overlaps audio data; not read"
            ));
        } else {
            let b = src.read_at(c.offset, FORMAT_CHUNK_SIZE, "format chunk")?;
            let bad = || AwcError::SizeOverflow;
            stream.format = Some(FormatInfo {
                samples: endian.u32(&b).ok_or_else(bad)?,
                loop_point: endian.u32(&b[4..]).ok_or_else(bad)? as i32,
                sample_rate: endian.u16(&b[8..]).ok_or_else(bad)?,
                headroom: endian.u16(&b[10..]).ok_or_else(bad)? as i16,
                loop_begin: endian.u16(&b[12..]).ok_or_else(bad)?,
                loop_end: endian.u16(&b[14..]).ok_or_else(bad)?,
                play_end: endian.u16(&b[16..]).ok_or_else(bad)?,
                play_begin: b[18],
                codec: Codec { id: b[19] },
            });
        }
    }

    if let Some(c) = stream
        .chunks
        .iter()
        .find(|c| c.tag == TAG_STREAM_FORMAT)
        .cloned()
    {
        if c.size < STREAM_FORMAT_HEADER_SIZE {
            warn.push(format!(
                "stream 0x{id:08X} stream-format chunk is only {} bytes",
                c.size
            ));
        } else if !readable(&c) {
            warn.push(format!(
                "stream 0x{id:08X} stream-format chunk is out of bounds or overlaps audio data; not read"
            ));
        } else {
            let take = c.size.min(STREAM_FORMAT_MAX_READ);
            let b = src.read_at(c.offset, take, "stream-format chunk")?;
            let bad = || AwcError::SizeOverflow;
            let block_count = endian.u32(&b).ok_or_else(bad)?;
            let block_size = endian.u32(&b[4..]).ok_or_else(bad)?;
            let channel_count = endian.u32(&b[8..]).ok_or_else(bad)?;
            if channel_count > MAX_CHANNELS {
                warn.push(format!(
                    "stream 0x{id:08X} declares {channel_count} channels (limit {MAX_CHANNELS}); channel table ignored"
                ));
            } else {
                let need = STREAM_FORMAT_HEADER_SIZE
                    + u64::from(channel_count) * STREAM_FORMAT_CHANNEL_SIZE;
                if need > take {
                    warn.push(format!(
                        "stream 0x{id:08X} stream-format chunk too small for {channel_count} channels"
                    ));
                } else {
                    let mut channels = Vec::new();
                    for k in 0..channel_count as usize {
                        let o = STREAM_FORMAT_HEADER_SIZE as usize
                            + k * STREAM_FORMAT_CHANNEL_SIZE as usize;
                        let e = b.get(o..o + 16).ok_or_else(bad)?;
                        channels.push(ChannelFormat {
                            id: endian.u32(e).ok_or_else(bad)?,
                            samples: endian.u32(&e[4..]).ok_or_else(bad)?,
                            headroom: endian.u16(&e[8..]).ok_or_else(bad)? as i16,
                            sample_rate: endian.u16(&e[10..]).ok_or_else(bad)?,
                            codec: Codec { id: e[12] },
                        });
                    }
                    stream.stream_format = Some(StreamFormatInfo {
                        block_count,
                        block_size,
                        channel_count,
                        channels,
                    });
                }
            }
        }
    }
    Ok(())
}

fn sanitize_file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().replace('\\', "/"))
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "unknown.awc".to_string())
}

/// Strictly read-only inspector for local `.awc` files of any size; see the module docs for the
/// read bound. Only the basename is stored in the report.
pub fn inspect_awc_file(path: &Path) -> Result<AwcReport, AwcError> {
    let mut file = fs::File::open(path)?;
    let size = file.metadata()?.len();
    inspect_awc_reader(&mut file, size, &sanitize_file_name(path))
}

/// One directory-mode result: path relative to the scanned root plus outcome.
pub type DirResult = (String, Result<AwcReport, AwcError>);

/// Recursively inspects every `*.awc` under `dir`, sorted by relative path.
pub fn inspect_awc_dir(dir: &Path) -> io::Result<Vec<DirResult>> {
    let mut found = Vec::new();
    collect_awc(dir, dir, &mut found)?;
    found.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out = Vec::with_capacity(found.len());
    for (rel, path) in found {
        let result = fs::File::open(&path)
            .map_err(AwcError::from)
            .and_then(|mut f| {
                let size = f.metadata()?.len();
                inspect_awc_reader(&mut f, size, &rel)
            });
        out.push((rel, result));
    }
    Ok(out)
}

fn collect_awc(
    root: &Path,
    dir: &Path,
    out: &mut Vec<(String, std::path::PathBuf)>,
) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let meta = entry.metadata()?;
        if meta.is_dir() {
            collect_awc(root, &path, out)?;
        } else if meta.is_file()
            && path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("awc"))
        {
            let rel = path
                .strip_prefix(root)
                .map_err(io::Error::other)?
                .to_string_lossy()
                .replace('\\', "/");
            out.push((rel, path));
        }
    }
    Ok(())
}

fn json_codec(c: Codec, endian: Endian) -> String {
    format!(
        "{{\"id\": {}, \"name\": \"{}\", \"known\": {}, \"web_decodable\": \"{}\"}}",
        c.id,
        json_escape(&c.label(endian)),
        c.is_known(),
        c.web_support().as_str()
    )
}

impl AwcReport {
    /// Metadata-only JSON with deterministic field order (no audio bytes, basename only).
    /// `max_streams` limits how many streams are listed; totals still cover all inspected ones.
    pub fn to_json(&self, max_streams: usize) -> String {
        let listed = self.streams.len().min(max_streams);
        let mut o = String::new();
        o.push_str("{\n");
        o.push_str(&format!("  \"abi_version\": {ABI_VERSION},\n"));
        o.push_str(&format!(
            "  \"file_name\": \"{}\",\n",
            json_escape(&self.file_name)
        ));
        o.push_str(&format!("  \"file_size\": {},\n", self.file_size));
        o.push_str(&format!(
            "  \"endianness\": \"{}\",\n",
            self.endian.as_str()
        ));
        o.push_str(&format!("  \"version\": {},\n", self.version));
        o.push_str(&format!("  \"version_known\": {},\n", self.version_known));
        let names: Vec<String> = self
            .flag_names()
            .iter()
            .map(|n| format!("\"{n}\""))
            .collect();
        o.push_str(&format!(
            "  \"flags\": {{\"raw\": {}, \"hex\": \"0x{:04X}\", \"decoded\": [{}], \"unknown_bits\": {}}},\n",
            self.flags,
            self.flags,
            names.join(", "),
            self.unknown_flag_bits
        ));
        o.push_str(&format!("  \"stream_count\": {},\n", self.stream_count));
        o.push_str(&format!("  \"data_offset\": {},\n", self.data_offset));
        o.push_str(&format!("  \"tables_parsed\": {},\n", self.tables_parsed));
        o.push_str(&format!("  \"table_end\": {},\n", self.table_end));
        o.push_str(&format!("  \"chunk_count\": {},\n", self.chunk_count));
        o.push_str(&format!(
            "  \"streams_inspected\": {},\n",
            self.streams_inspected
        ));
        o.push_str(&format!("  \"streams_listed\": {listed},\n"));
        o.push_str(&format!("  \"bytes_read\": {},\n", self.bytes_read));
        o.push_str("  \"payload_bytes_by_codec\": {");
        let parts: Vec<String> = self
            .payload_bytes_by_codec
            .iter()
            .map(|(k, v)| format!("\"{}\": {v}", json_escape(k)))
            .collect();
        o.push_str(&parts.join(", "));
        o.push_str("},\n");
        o.push_str("  \"streams\": [");
        for (i, s) in self.streams[..listed].iter().enumerate() {
            o.push_str(if i == 0 { "\n" } else { ",\n" });
            o.push_str(&format!("    {{\"id\": {}, ", s.id));
            match s.chunk_index {
                Some(ci) => o.push_str(&format!("\"chunk_index\": {ci}, ")),
                None => o.push_str("\"chunk_index\": null, "),
            }
            o.push_str("\"chunks\": [");
            let chunks: Vec<String> = s
                .chunks
                .iter()
                .map(|c| {
                    let name = c.tag_name().map_or_else(
                        || format!("unknown(0x{:02X})", c.tag),
                        std::string::ToString::to_string,
                    );
                    format!(
                        "{{\"tag\": {}, \"name\": \"{}\", \"known\": {}, \"offset\": {}, \"size\": {}}}",
                        c.tag,
                        name,
                        c.tag_name().is_some(),
                        c.offset,
                        c.size
                    )
                })
                .collect();
            o.push_str(&chunks.join(", "));
            o.push_str("], ");
            match &s.format {
                Some(f) => {
                    o.push_str(&format!(
                        "\"format\": {{\"samples\": {}, \"sample_rate\": {}, \"channels\": 1, \"codec\": {}, \"loop_point\": {}, \"headroom\": {}, \"duration_ms\": {}}}, ",
                        f.samples,
                        f.sample_rate,
                        json_codec(f.codec, self.endian),
                        f.loop_point,
                        f.headroom,
                        f.duration_ms().map_or("null".to_string(), |d| d.to_string())
                    ));
                }
                None => o.push_str("\"format\": null, "),
            }
            match &s.stream_format {
                Some(sf) => {
                    let ch: Vec<String> = sf
                        .channels
                        .iter()
                        .map(|c| {
                            format!(
                                "{{\"id\": {}, \"samples\": {}, \"sample_rate\": {}, \"headroom\": {}, \"codec\": {}, \"duration_ms\": {}}}",
                                c.id,
                                c.samples,
                                c.sample_rate,
                                c.headroom,
                                json_codec(c.codec, self.endian),
                                duration_ms(c.samples, c.sample_rate)
                                    .map_or("null".to_string(), |d| d.to_string())
                            )
                        })
                        .collect();
                    o.push_str(&format!(
                        "\"stream_format\": {{\"block_count\": {}, \"block_size\": {}, \"channel_count\": {}, \"channels\": [{}]}}}}",
                        sf.block_count,
                        sf.block_size,
                        sf.channel_count,
                        ch.join(", ")
                    ));
                }
                None => o.push_str("\"stream_format\": null}"),
            }
        }
        if listed > 0 {
            o.push_str("\n  ");
        }
        o.push_str("],\n");
        o.push_str("  \"warnings\": [");
        let w: Vec<String> = self
            .warnings
            .iter()
            .map(|w| format!("\"{}\"", json_escape(w)))
            .collect();
        o.push_str(&w.join(", "));
        o.push_str("],\n");
        o.push_str(&format!(
            "  \"warnings_suppressed\": {}\n",
            self.warnings_suppressed
        ));
        o.push_str("}\n");
        o
    }

    /// Compact one-line summary used in directory mode.
    pub fn format_line(&self) -> String {
        format!(
            "{}: {} endian, v{}{}, {} streams, {} chunks, {}",
            self.file_name,
            self.endian.as_str(),
            self.version,
            if self.version_known {
                ""
            } else {
                " (unrecognised)"
            },
            self.stream_count,
            self.chunk_count,
            format_bytes(self.file_size)
        )
    }

    pub fn format_human(&self, max_streams: usize) -> String {
        let mut o = String::new();
        o.push_str("AWC Audio Bank Header\n");
        o.push_str(&format!("  File:             {}\n", self.file_name));
        o.push_str(&format!(
            "  File Size:        {} ({} bytes)\n",
            format_bytes(self.file_size),
            self.file_size
        ));
        o.push_str(&format!(
            "  Endianness:       {} ({})\n",
            self.endian.as_str(),
            match self.endian {
                Endian::Little => "ADAT",
                Endian::Big => "TADA",
            }
        ));
        o.push_str(&format!(
            "  Version:          {} ({})\n",
            self.version,
            if self.version_known {
                "known"
            } else {
                "UNRECOGNISED"
            }
        ));
        let names = self.flag_names();
        let mut flags = if names.is_empty() {
            "none".to_string()
        } else {
            names.join(", ")
        };
        if self.unknown_flag_bits != 0 {
            flags.push_str(&format!(" + unknown bits 0x{:04X}", self.unknown_flag_bits));
        }
        o.push_str(&format!(
            "  Flags:            0x{:04X} ({flags})\n",
            self.flags
        ));
        o.push_str(&format!("  Streams:          {}\n", self.stream_count));
        o.push_str(&format!("  Chunks:           {}\n", self.chunk_count));
        o.push_str(&format!("  Data Offset:      {}\n", self.data_offset));
        o.push_str(&format!("  Tables End:       {}\n", self.table_end));
        o.push_str(&format!("  Bytes Read:       {}\n", self.bytes_read));
        if !self.tables_parsed {
            o.push_str("  Tables:           not interpreted (unrecognised version)\n");
        }
        let listed = self.streams.len().min(max_streams);
        if listed > 0 {
            o.push_str(&format!(
                "Streams (showing {listed} of {}):\n",
                self.streams.len()
            ));
        }
        for (i, s) in self.streams[..listed].iter().enumerate() {
            o.push_str(&format!(
                "  #{i} id=0x{:08X} chunks={}{}\n",
                s.id,
                s.chunks.len(),
                s.chunk_index
                    .map_or(String::new(), |c| format!(" chunk_index={c}"))
            ));
            for c in &s.chunks {
                let name = c.tag_name().map_or_else(
                    || format!("unknown(0x{:02X})", c.tag),
                    std::string::ToString::to_string,
                );
                o.push_str(&format!(
                    "      {name:<18} tag=0x{:02X} offset={} size={}\n",
                    c.tag, c.offset, c.size
                ));
            }
            if let Some(f) = &s.format {
                o.push_str(&format!(
                    "      format: {} Hz, {} samples{}, codec {} (web: {})\n",
                    f.sample_rate,
                    f.samples,
                    f.duration_ms()
                        .map_or(String::new(), |d| format!(" ({d} ms)")),
                    f.codec.label(self.endian),
                    f.codec.web_support().as_str()
                ));
            }
            if let Some(sf) = &s.stream_format {
                o.push_str(&format!(
                    "      stream format: {} channels, {} blocks of {} bytes\n",
                    sf.channel_count, sf.block_count, sf.block_size
                ));
                for (k, c) in sf.channels.iter().enumerate() {
                    o.push_str(&format!(
                        "        ch{k}: id={} {} Hz, {} samples, codec {} (web: {})\n",
                        c.id,
                        c.sample_rate,
                        c.samples,
                        c.codec.label(self.endian),
                        c.codec.web_support().as_str()
                    ));
                }
            }
        }
        if !self.payload_bytes_by_codec.is_empty() {
            o.push_str("Payload bytes by codec:\n");
            for (k, v) in &self.payload_bytes_by_codec {
                o.push_str(&format!("  {k:<18} {} ({v} bytes)\n", format_bytes(*v)));
            }
        }
        if !self.warnings.is_empty() {
            o.push_str("Warnings:\n");
            for w in &self.warnings {
                o.push_str(&format!("  - {w}\n"));
            }
            if self.warnings_suppressed > 0 {
                o.push_str(&format!(
                    "  - {} further warnings suppressed\n",
                    self.warnings_suppressed
                ));
            }
        }
        o
    }
}

/// Renders directory-mode results as JSON (deterministic; relative paths only).
pub fn dir_to_json(results: &[DirResult], max_streams: usize) -> String {
    let errors = results.iter().filter(|r| r.1.is_err()).count();
    let mut o = String::new();
    o.push_str("{\n");
    o.push_str(&format!("  \"abi_version\": {ABI_VERSION},\n"));
    o.push_str(&format!("  \"file_count\": {},\n", results.len()));
    o.push_str(&format!("  \"error_count\": {errors},\n"));
    o.push_str("  \"files\": [");
    for (i, (rel, res)) in results.iter().enumerate() {
        o.push_str(if i == 0 { "\n" } else { ",\n" });
        match res {
            Ok(r) => {
                let body = r.to_json(max_streams);
                let body = body.trim_end();
                let indented: Vec<String> = body.lines().map(|l| format!("    {l}")).collect();
                o.push_str(&indented.join("\n"));
            }
            Err(e) => o.push_str(&format!(
                "    {{\"file_name\": \"{}\", \"error\": \"{}\"}}",
                json_escape(rel),
                json_escape(&e.to_string())
            )),
        }
    }
    if !results.is_empty() {
        o.push_str("\n  ");
    }
    o.push_str("]\n}\n");
    o
}

/// Renders directory-mode results as one line per file.
pub fn dir_format_human(results: &[DirResult]) -> String {
    let mut o = format!("AWC banks found: {}\n", results.len());
    for (rel, res) in results {
        match res {
            Ok(r) => o.push_str(&format!("  {}\n", r.format_line())),
            Err(e) => o.push_str(&format!("  {rel}: ERROR {e}\n")),
        }
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(label: &str) -> std::path::PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let d =
            std::env::temp_dir().join(format!("playgta6_awc_{label}_{}_{}", std::process::id(), n));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    // ---- in-test synthetic bank writer ----

    type Chunks = Vec<(u8, Vec<u8>)>;

    struct B {
        endian: Endian,
        version: u16,
        flags: u16,
        streams: Vec<(u32, Chunks)>,
        indices: Vec<u16>,
    }

    struct Built {
        bytes: Vec<u8>,
        table_end: usize,
        payload: Vec<(u64, u64)>,
    }

    fn put16(e: Endian, v: u16) -> [u8; 2] {
        match e {
            Endian::Little => v.to_le_bytes(),
            Endian::Big => v.to_be_bytes(),
        }
    }
    fn put32(e: Endian, v: u32) -> [u8; 4] {
        match e {
            Endian::Little => v.to_le_bytes(),
            Endian::Big => v.to_be_bytes(),
        }
    }
    fn put64(e: Endian, v: u64) -> [u8; 8] {
        match e {
            Endian::Little => v.to_le_bytes(),
            Endian::Big => v.to_be_bytes(),
        }
    }

    impl B {
        fn new(endian: Endian) -> Self {
            B {
                endian,
                version: 1,
                flags: 0,
                streams: Vec::new(),
                indices: Vec::new(),
            }
        }
        fn stream(mut self, id: u32, chunks: Chunks) -> Self {
            self.streams.push((id, chunks));
            self
        }
        fn flags(mut self, f: u16) -> Self {
            self.flags |= f;
            self
        }
        fn build(&self) -> Built {
            let e = self.endian;
            let wide = self.flags & FLAG_WIDE_DESCRIPTORS != 0;
            let has_idx = self.flags & FLAG_CHUNK_INDICES != 0;
            let n = self.streams.len();
            let total: usize = self.streams.iter().map(|s| s.1.len()).sum();
            let table_end =
                16 + if has_idx { n * 2 } else { 0 } + n * 4 + total * if wide { 16 } else { 8 };
            let mut out = Vec::new();
            out.extend_from_slice(match e {
                Endian::Little => &AWC_MAGIC_LE,
                Endian::Big => &AWC_MAGIC_BE,
            });
            out.extend_from_slice(&put16(e, self.version));
            out.extend_from_slice(&put16(e, self.flags));
            out.extend_from_slice(&put32(e, n as u32));
            out.extend_from_slice(&put32(e, table_end as u32));
            if has_idx {
                for i in 0..n {
                    let v = self.indices.get(i).copied().unwrap_or(i as u16);
                    out.extend_from_slice(&put16(e, v));
                }
            }
            for (id, chunks) in &self.streams {
                out.extend_from_slice(&put32(e, id | ((chunks.len() as u32) << 29)));
            }
            let mut body = Vec::new();
            let mut payload = Vec::new();
            let mut descs = Vec::new();
            for (_, chunks) in &self.streams {
                for (tag, data) in chunks {
                    let off = (table_end + body.len()) as u64;
                    let size = data.len() as u64;
                    if *tag == TAG_DATA {
                        payload.push((off, size));
                    }
                    if wide {
                        descs.extend_from_slice(&put32(e, u32::from(*tag)));
                        descs.extend_from_slice(&put32(e, size as u32));
                        descs.extend_from_slice(&put64(e, off));
                    } else {
                        descs.extend_from_slice(&put64(
                            e,
                            u64::from(*tag) | (size << 8) | (off << 36),
                        ));
                    }
                    body.extend_from_slice(data);
                }
            }
            out.extend_from_slice(&descs);
            assert_eq!(out.len(), table_end);
            out.extend_from_slice(&body);
            Built {
                bytes: out,
                table_end,
                payload,
            }
        }
    }

    fn fmt_chunk(e: Endian, samples: u32, rate: u16, codec: u8) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&put32(e, samples));
        v.extend_from_slice(&put32(e, (-1i32) as u32));
        v.extend_from_slice(&put16(e, rate));
        v.extend_from_slice(&put16(e, (-5i16) as u16));
        v.extend_from_slice(&put16(e, 1));
        v.extend_from_slice(&put16(e, 2));
        v.extend_from_slice(&put16(e, 3));
        v.push(4);
        v.push(codec);
        assert_eq!(v.len(), 20);
        v
    }

    fn sfmt_chunk(e: Endian, blocks: u32, bsize: u32, chans: &[(u32, u32, u16, u8)]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&put32(e, blocks));
        v.extend_from_slice(&put32(e, bsize));
        v.extend_from_slice(&put32(e, chans.len() as u32));
        for (id, samples, rate, codec) in chans {
            v.extend_from_slice(&put32(e, *id));
            v.extend_from_slice(&put32(e, *samples));
            v.extend_from_slice(&put16(e, 0));
            v.extend_from_slice(&put16(e, *rate));
            v.push(*codec);
            v.extend_from_slice(&[0, 0, 0]);
        }
        v
    }

    fn inspect(bytes: &[u8]) -> Result<AwcReport, AwcError> {
        let mut c = Cursor::new(bytes.to_vec());
        inspect_awc_reader(&mut c, bytes.len() as u64, "t.awc")
    }

    fn simple(e: Endian, codec: u8) -> Built {
        B::new(e)
            .stream(
                0x1234,
                vec![
                    (TAG_FORMAT, fmt_chunk(e, 48_000, 24_000, codec)),
                    (TAG_DATA, vec![0xEE; 256]),
                ],
            )
            .build()
    }

    // ---- reader wrappers ----

    struct Guard {
        inner: Cursor<Vec<u8>>,
        forbidden: Vec<(u64, u64)>,
        total: u64,
    }
    impl Read for Guard {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let pos = self.inner.position();
            if overlaps_any(pos, buf.len() as u64, &self.forbidden) {
                return Err(io::Error::other("touched payload"));
            }
            let n = self.inner.read(buf)?;
            self.total += n as u64;
            Ok(n)
        }
    }
    impl Seek for Guard {
        fn seek(&mut self, p: SeekFrom) -> io::Result<u64> {
            self.inner.seek(p)
        }
    }
    fn overlaps_any(pos: u64, len: u64, r: &[(u64, u64)]) -> bool {
        r.iter().any(|(o, l)| overlaps(pos, len, *o, *l))
    }

    // ---- tests ----

    #[test]
    fn parses_little_endian_single_stream() {
        let b = simple(Endian::Little, 0);
        let r = inspect(&b.bytes).unwrap();
        assert_eq!(r.endian, Endian::Little);
        assert_eq!(r.version, 1);
        assert!(r.version_known && r.tables_parsed);
        assert_eq!(r.stream_count, 1);
        assert_eq!(r.chunk_count, 2);
        assert_eq!(r.table_end, b.table_end as u64);
        let s = &r.streams[0];
        assert_eq!(s.id, 0x1234);
        assert_eq!(s.chunks.len(), 2);
        assert_eq!(s.chunks[0].tag_name(), Some("format"));
        assert_eq!(s.chunks[1].size, 256);
        let f = s.format.as_ref().unwrap();
        assert_eq!(f.samples, 48_000);
        assert_eq!(f.sample_rate, 24_000);
        assert_eq!(f.loop_point, -1);
        assert_eq!(f.headroom, -5);
        assert_eq!(
            (f.loop_begin, f.loop_end, f.play_end, f.play_begin),
            (1, 2, 3, 4)
        );
        assert_eq!(f.duration_ms(), Some(2000));
        assert_eq!(r.payload_bytes_by_codec.get("PCM16-LE"), Some(&256));
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
    }

    #[test]
    fn big_endian_matches_little_endian_contents() {
        let l = inspect(&simple(Endian::Little, 4).bytes).unwrap();
        let g = inspect(&simple(Endian::Big, 4).bytes).unwrap();
        assert_eq!(g.endian, Endian::Big);
        assert_eq!(l.streams, g.streams);
        assert_eq!(g.payload_bytes_by_codec.get("IMA-ADPCM"), Some(&256));
        let pcm_be = inspect(&simple(Endian::Big, 0).bytes).unwrap();
        assert!(pcm_be.payload_bytes_by_codec.contains_key("PCM16-BE"));
    }

    #[test]
    fn multi_channel_streaming_bank() {
        for e in [Endian::Little, Endian::Big] {
            let b = B::new(e)
                .flags(FLAG_MULTI_CHANNEL)
                .stream(
                    7,
                    vec![
                        (
                            TAG_STREAM_FORMAT,
                            sfmt_chunk(e, 10, 2048, &[(1, 1000, 32_000, 4), (2, 1000, 32_000, 4)]),
                        ),
                        (TAG_DATA, vec![0x11; 4096]),
                        (0x5B, vec![0; 16]),
                    ],
                )
                .build();
            let r = inspect(&b.bytes).unwrap();
            assert_eq!(r.flag_names(), vec!["multi_channel"]);
            let sf = r.streams[0].stream_format.as_ref().unwrap();
            assert_eq!(
                (sf.block_count, sf.block_size, sf.channel_count),
                (10, 2048, 2)
            );
            assert_eq!(sf.channels[1].id, 2);
            assert_eq!(sf.channels[1].sample_rate, 32_000);
            assert_eq!(r.payload_bytes_by_codec.get("IMA-ADPCM"), Some(&4096));
        }
    }

    #[test]
    fn mixed_channel_codecs_are_labelled_mixed() {
        let e = Endian::Little;
        let b = B::new(e)
            .stream(
                1,
                vec![
                    (
                        TAG_STREAM_FORMAT,
                        sfmt_chunk(e, 1, 1, &[(1, 1, 1, 0), (2, 1, 1, 4)]),
                    ),
                    (TAG_DATA, vec![0; 10]),
                ],
            )
            .build();
        let r = inspect(&b.bytes).unwrap();
        assert_eq!(r.payload_bytes_by_codec.get("mixed"), Some(&10));
    }

    #[test]
    fn every_listed_chunk_tag_is_named_and_unknown_is_raw() {
        for (tag, name) in CHUNK_TAG_TABLE {
            let b = B::new(Endian::Little)
                .stream(1, vec![(*tag, vec![0; 8])])
                .build();
            let r = inspect(&b.bytes).unwrap();
            assert_eq!(r.streams[0].chunks[0].tag_name(), Some(*name));
        }
        let b = B::new(Endian::Little)
            .stream(1, vec![(0x01, vec![0; 8])])
            .build();
        let r = inspect(&b.bytes).unwrap();
        assert_eq!(r.streams[0].chunks[0].tag_name(), None);
        let j = r.to_json(10);
        assert!(j.contains("\"name\": \"unknown(0x01)\""));
        assert!(r.format_human(10).contains("unknown(0x01)"));
    }

    #[test]
    fn chunk_tag_table_has_no_duplicates() {
        for (i, a) in CHUNK_TAG_TABLE.iter().enumerate() {
            for b in &CHUNK_TAG_TABLE[i + 1..] {
                assert_ne!(a.0, b.0);
            }
        }
        for (i, a) in CODEC_TABLE.iter().enumerate() {
            for b in &CODEC_TABLE[i + 1..] {
                assert_ne!(a.0, b.0);
            }
        }
    }

    #[test]
    fn known_and_unknown_codecs() {
        for (id, name, web) in CODEC_TABLE {
            let r = inspect(&simple(Endian::Little, *id).bytes).unwrap();
            let c = r.streams[0].format.as_ref().unwrap().codec;
            assert!(c.is_known());
            assert_eq!(c.web_support(), *web);
            assert!(c.label(Endian::Little).starts_with(name));
        }
        for id in [1u8, 7, 8, 0xFF] {
            let r = inspect(&simple(Endian::Little, id).bytes).unwrap();
            let c = r.streams[0].format.as_ref().unwrap().codec;
            assert!(!c.is_known());
            assert_eq!(c.web_support(), WebSupport::Unknown);
            assert_eq!(c.label(Endian::Little), format!("unknown({id})"));
            let j = r.to_json(4);
            assert!(j.contains(&format!("\"name\": \"unknown({id})\", \"known\": false")));
            assert!(
                r.payload_bytes_by_codec
                    .contains_key(&format!("unknown({id})"))
            );
        }
    }

    #[test]
    fn wide_descriptors_parse_and_warn() {
        for e in [Endian::Little, Endian::Big] {
            let b = B::new(e)
                .flags(FLAG_WIDE_DESCRIPTORS)
                .stream(
                    9,
                    vec![
                        (TAG_FORMAT, fmt_chunk(e, 100, 8000, 0)),
                        (TAG_DATA, vec![1; 64]),
                    ],
                )
                .build();
            let r = inspect(&b.bytes).unwrap();
            assert_eq!(r.table_end, b.table_end as u64);
            assert_eq!(r.streams[0].format.as_ref().unwrap().sample_rate, 8000);
            assert_eq!(r.streams[0].data_bytes(), 64);
            assert!(
                r.warnings
                    .iter()
                    .any(|w| w.contains("wide chunk descriptors"))
            );
        }
    }

    #[test]
    fn chunk_index_table_is_read() {
        let mut b = B::new(Endian::Little)
            .flags(FLAG_CHUNK_INDICES)
            .stream(1, vec![(TAG_DATA, vec![0; 4])])
            .stream(2, vec![(TAG_DATA, vec![0; 4])]);
        b.indices = vec![5, 9];
        let r = inspect(&b.build().bytes).unwrap();
        assert_eq!(r.streams[0].chunk_index, Some(5));
        assert_eq!(r.streams[1].chunk_index, Some(9));
        assert!(r.format_human(5).contains("chunk_index=9"));
    }

    #[test]
    fn unknown_version_is_reported_not_guessed() {
        let mut b = B::new(Endian::Little).stream(1, vec![(TAG_DATA, vec![0; 4])]);
        b.version = 0x77;
        let r = inspect(&b.build().bytes).unwrap();
        assert!(!r.version_known);
        assert!(!r.tables_parsed);
        assert!(r.streams.is_empty());
        assert_eq!(r.version, 0x77);
        assert!(r.warnings[0].contains("unrecognised AWC version 119"));
        assert!(r.format_human(5).contains("UNRECOGNISED"));
        assert!(r.to_json(5).contains("\"version_known\": false"));
    }

    #[test]
    fn unknown_flag_bits_are_reported() {
        let b = B::new(Endian::Little)
            .flags(0x8001)
            .stream(1, vec![(TAG_DATA, vec![0; 4])])
            .build();
        let r = inspect(&b.bytes).unwrap();
        assert_eq!(r.unknown_flag_bits, 0x8001);
        assert!(r.warnings.iter().any(|w| w.contains("0x8001")));
        assert!(r.to_json(5).contains("\"unknown_bits\": 32769"));
    }

    #[test]
    fn truncation_at_every_length_never_panics_and_errors_before_tables_end() {
        for e in [Endian::Little, Endian::Big] {
            let b = B::new(e)
                .flags(FLAG_CHUNK_INDICES)
                .stream(
                    1,
                    vec![
                        (TAG_FORMAT, fmt_chunk(e, 10, 100, 0)),
                        (TAG_DATA, vec![0xAA; 32]),
                    ],
                )
                .stream(
                    2,
                    vec![(TAG_STREAM_FORMAT, sfmt_chunk(e, 1, 1, &[(1, 1, 1, 0)]))],
                )
                .build();
            for len in 0..=b.bytes.len() {
                let res = inspect(&b.bytes[..len]);
                if len < b.table_end {
                    assert!(res.is_err(), "len {len} should fail");
                } else if let Ok(r) = res {
                    assert!(r.bytes_read <= len as u64);
                }
            }
            assert!(inspect(&b.bytes).unwrap().warnings.is_empty());
        }
    }

    #[test]
    fn truncation_errors_are_structured() {
        let b = simple(Endian::Little, 0);
        assert!(matches!(
            inspect(&b.bytes[..2]).unwrap_err(),
            AwcError::TruncatedHeader { .. } | AwcError::InvalidMagic { .. }
        ));
        assert!(matches!(
            inspect(&b.bytes[..10]).unwrap_err(),
            AwcError::TruncatedHeader {
                actual: 10,
                expected: 16
            }
        ));
        assert!(matches!(
            inspect(&b.bytes[..18]).unwrap_err(),
            AwcError::TruncatedTable { .. }
        ));
        assert!(inspect(&[]).is_err());
    }

    #[test]
    fn absurd_counts_error_cleanly() {
        let mut b = simple(Endian::Little, 0).bytes;
        b[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            inspect(&b).unwrap_err(),
            AwcError::TooManyStreams { .. }
        ));
        // Within cap but far larger than the file.
        b[8..12].copy_from_slice(&60_000u32.to_le_bytes());
        assert!(matches!(
            inspect(&b).unwrap_err(),
            AwcError::TruncatedTable {
                table: "stream id table",
                ..
            }
        ));
        // Max chunk counts in the id table exceed the file.
        let mut b = simple(Endian::Little, 0).bytes;
        b.truncate(40);
        b[16..20].copy_from_slice(&(0x1234u32 | (7 << 29)).to_le_bytes());
        assert!(matches!(
            inspect(&b).unwrap_err(),
            AwcError::TruncatedTable {
                table: "chunk descriptor table",
                ..
            }
        ));
    }

    #[test]
    fn absurd_chunk_offsets_warn_without_reading() {
        let mut b = simple(Endian::Little, 0).bytes;
        let v: u64 = u64::from(TAG_FORMAT) | (0x0FFF_FFFFu64 << 8) | (0x0FFF_FFFFu64 << 36);
        b[20..28].copy_from_slice(&v.to_le_bytes());
        let r = inspect(&b).unwrap();
        assert!(r.streams[0].format.is_none());
        assert!(r.warnings.iter().any(|w| w.contains("exceeds file size")));
    }

    #[test]
    fn format_chunk_overlapping_data_is_not_read() {
        let e = Endian::Little;
        let mut b = B::new(e)
            .stream(
                1,
                vec![(TAG_FORMAT, vec![0; 20]), (TAG_DATA, vec![0xEE; 64])],
            )
            .build();
        // Point the format chunk at the data chunk offset.
        let data_off = b.payload[0].0;
        let v = u64::from(TAG_FORMAT) | (20u64 << 8) | (data_off << 36);
        b.bytes[20..28].copy_from_slice(&v.to_le_bytes());
        let mut g = Guard {
            inner: Cursor::new(b.bytes.clone()),
            forbidden: b.payload.clone(),
            total: 0,
        };
        let r = inspect_awc_reader(&mut g, b.bytes.len() as u64, "x.awc").unwrap();
        assert!(r.streams[0].format.is_none());
        assert!(r.warnings.iter().any(|w| w.contains("overlaps audio data")));
    }

    #[test]
    fn zero_streams_is_valid_with_warning() {
        let b = B::new(Endian::Little).build();
        let r = inspect(&b.bytes).unwrap();
        assert_eq!(r.stream_count, 0);
        assert!(r.streams.is_empty());
        assert!(r.warnings.iter().any(|w| w.contains("zero streams")));
        assert!(r.to_json(5).contains("\"streams\": []"));
        assert!(r.format_human(5).contains("zero streams"));
    }

    #[test]
    fn duplicate_stream_ids_warn() {
        let b = B::new(Endian::Little)
            .stream(5, vec![(TAG_DATA, vec![0; 4])])
            .stream(5, vec![(TAG_DATA, vec![0; 4])])
            .build();
        let r = inspect(&b.bytes).unwrap();
        assert_eq!(r.streams.len(), 2);
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("duplicate stream id 0x00000005"))
        );
    }

    #[test]
    fn payload_is_never_read_and_read_bound_holds() {
        for e in [Endian::Little, Endian::Big] {
            let mut bld = B::new(e);
            for i in 0..50u32 {
                bld = bld.stream(
                    i,
                    vec![
                        (TAG_FORMAT, fmt_chunk(e, 1000, 16_000, 4)),
                        (TAG_DATA, vec![0xEE; 10_000]),
                    ],
                );
            }
            bld = bld.stream(
                99,
                vec![
                    (TAG_STREAM_FORMAT, sfmt_chunk(e, 1, 1, &[(1, 1, 1, 0)])),
                    (TAG_DATA, vec![0xEE; 10_000]),
                ],
            );
            let b = bld.build();
            let mut g = Guard {
                inner: Cursor::new(b.bytes.clone()),
                forbidden: b.payload.clone(),
                total: 0,
            };
            let r = inspect_awc_reader(&mut g, b.bytes.len() as u64, "x.awc").unwrap();
            assert_eq!(r.streams.len(), 51);
            assert_eq!(g.total, r.bytes_read);
            let expected = 16 + 51 * 4 + 102 * 8 + 50 * 20 + (12 + 16);
            assert_eq!(r.bytes_read, expected as u64);
            assert!(r.bytes_read < MAX_READ_BYTES);
            assert!(r.bytes_read < (b.bytes.len() / 10) as u64);
        }
    }

    #[test]
    fn worst_case_bound_is_below_limit() {
        let tables = AWC_HEADER_SIZE + MAX_STREAMS * (2 + 4 + 7 * 16);
        let detail = MAX_DETAIL_STREAMS as u64 * (FORMAT_CHUNK_SIZE + STREAM_FORMAT_MAX_READ);
        assert!(tables + detail <= MAX_READ_BYTES);
    }

    #[test]
    fn read_budget_is_enforced() {
        let b = simple(Endian::Little, 0);
        let mut c = Cursor::new(b.bytes.clone());
        let err = inspect_awc_reader_with_limit(&mut c, b.bytes.len() as u64, "x", 20).unwrap_err();
        assert!(matches!(err, AwcError::ReadBudgetExceeded { limit: 20 }));
    }

    #[test]
    fn bad_magic_and_encrypted_looking_header() {
        let mut plain = vec![0u8; 64];
        plain[..4].copy_from_slice(b"RPF9");
        match inspect(&plain).unwrap_err() {
            AwcError::InvalidMagic {
                looks_encrypted, ..
            } => assert!(!looks_encrypted),
            other => panic!("{other:?}"),
        }
        let noisy: Vec<u8> = (0..64u32).map(|i| (i * 37 + 11) as u8).collect();
        let err = inspect(&noisy).unwrap_err();
        assert!(matches!(
            err,
            AwcError::InvalidMagic {
                looks_encrypted: true,
                ..
            }
        ));
        assert!(err.to_string().contains("look encrypted"));
    }

    #[test]
    fn json_is_deterministic_and_metadata_only() {
        let b = B::new(Endian::Little)
            .stream(
                1,
                vec![
                    (TAG_FORMAT, fmt_chunk(Endian::Little, 100, 50, 0)),
                    (TAG_DATA, vec![0xEE; 128]),
                ],
            )
            .stream(2, vec![(TAG_DATA, vec![0xDD; 128])])
            .build();
        let a = inspect(&b.bytes).unwrap().to_json(10);
        let c = inspect(&b.bytes).unwrap().to_json(10);
        assert_eq!(a, c);
        assert!(a.starts_with("{\n  \"abi_version\": 1,\n  \"file_name\": \"t.awc\","));
        assert!(!a.contains("EE") && !a.contains("DD"));
        assert!(a.contains(
            "\"payload_bytes_by_codec\": {\"PCM16-LE\": 128, \"no format information\": 128}"
        ));
        assert!(a.contains("\"duration_ms\": 2000"));
        // Stream listing limit.
        let limited = inspect(&b.bytes).unwrap().to_json(1);
        assert!(limited.contains("\"streams_listed\": 1"));
        assert!(!limited.contains("\"id\": 2,"));
        let human = inspect(&b.bytes).unwrap().format_human(1);
        assert!(human.contains("showing 1 of 2"));
    }

    #[test]
    fn warnings_are_capped() {
        let mut bld = B::new(Endian::Little);
        for i in 0..200u32 {
            bld = bld.stream(i, vec![(TAG_FORMAT, vec![0; 4])]);
        }
        let r = inspect(&bld.build().bytes).unwrap();
        assert_eq!(r.warnings.len(), MAX_WARNINGS);
        assert_eq!(r.warnings_suppressed, 200 - MAX_WARNINGS as u64);
        assert!(r.format_human(1).contains("further warnings suppressed"));
    }

    #[test]
    fn file_and_dir_inspection_use_relative_names() {
        let dir = temp_dir("files");
        let sub = dir.join("sfx");
        fs::create_dir_all(&sub).unwrap();
        fs::write(sub.join("b.AWC"), simple(Endian::Little, 0).bytes).unwrap();
        fs::write(dir.join("a.awc"), simple(Endian::Big, 4).bytes).unwrap();
        fs::write(dir.join("bad.awc"), b"nope nope nope nope").unwrap();
        fs::write(dir.join("skip.txt"), b"x").unwrap();

        let r = inspect_awc_file(&dir.join("a.awc")).unwrap();
        assert_eq!(r.file_name, "a.awc");
        assert!(
            !r.to_json(5)
                .contains(&dir.to_string_lossy().replace('\\', "\\\\"))
        );

        let res = inspect_awc_dir(&dir).unwrap();
        let names: Vec<&str> = res.iter().map(|r| r.0.as_str()).collect();
        assert_eq!(names, vec!["a.awc", "bad.awc", "sfx/b.AWC"]);
        assert!(res[1].1.is_err());
        let j = dir_to_json(&res, 5);
        assert!(j.contains("\"file_count\": 3"));
        assert!(j.contains("\"error_count\": 1"));
        assert_eq!(j, dir_to_json(&res, 5));
        let h = dir_format_human(&res);
        assert!(h.contains("AWC banks found: 3"));
        assert!(h.contains("bad.awc: ERROR"));
        assert!(inspect_awc_file(&dir.join("missing.awc")).is_err());
        let _ = fs::remove_dir_all(&dir);
    }
}
