//! PE64 exception directory (`.pdata`) parser.
//!
//! `IMAGE_DIRECTORY_ENTRY_EXCEPTION` is an array of 12-byte `RUNTIME_FUNCTION` records
//! (`BeginAddress`, `EndAddress`, `UnwindInfoAddress`, all RVAs), one per contiguous chunk of
//! every function that has unwind data. Each record points at an `UNWIND_INFO`:
//!
//! ```text
//! u8  Version:3, Flags:5          (EHANDLER = 1, UHANDLER = 2, CHAININFO = 4)
//! u8  SizeOfProlog
//! u8  CountOfCodes                (2-byte slots)
//! u8  FrameRegister:4, FrameOffset:4
//! UNWIND_CODE[CountOfCodes]       (padded to an even count)
//! u32 ExceptionHandler + language-specific data   if EHANDLER | UHANDLER
//! RUNTIME_FUNCTION parent                          if CHAININFO
//! ```
//!
//! A chained record describes a secondary chunk (cold code, shrink-wrapped regions) of the
//! function its parent belongs to; an `UnwindInfoAddress` with bit 0 set points at another
//! `RUNTIME_FUNCTION` (at `rva - 1`) instead. Following either link to a record with plain
//! unwind info yields the chunk's primary function.
//!
//! [`ExceptionDirectory::from_pe`] never fails: problems become [`PdataDiagnostic`]s and the
//! affected record is dropped (bad range) or kept without unwind info (bad unwind data). All
//! reads go through R1's bounds-checked [`Pe64::rva_to_file_offset`]. The kept records are sorted,
//! deduplicated and non-overlapping, so [`ExceptionDirectory::function_table`] can map every
//! chunk to its primary function and answer "which function owns this address" in `O(log n)`.
//! Everything is `O(n log n)` in the number of records.

use std::fmt;

use crate::code_image::{CodeImage, Region};
use crate::pe::{IMAGE_SCN_MEM_EXECUTE, Pe64};

/// Data directory index of the exception table.
pub const DIR_EXCEPTION: usize = 3;
/// Size of one `RUNTIME_FUNCTION` record.
pub const RUNTIME_FUNCTION_LEN: usize = 12;

pub const UNW_FLAG_EHANDLER: u8 = 0x1;
pub const UNW_FLAG_UHANDLER: u8 = 0x2;
pub const UNW_FLAG_CHAININFO: u8 = 0x4;

/// Bit 0 of `UnwindInfoAddress`: the record at `rva - 1` holds the unwind data.
const RUNTIME_FUNCTION_INDIRECT: u32 = 0x1;
/// Longest chain of parent links followed before giving up (guards against cycles).
const MAX_CHAIN_DEPTH: usize = 32;
const UNWIND_HEADER_LEN: usize = 4;

/// One `RUNTIME_FUNCTION` record (RVAs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RuntimeFunction {
    pub begin: u32,
    /// One past the last byte of the chunk.
    pub end: u32,
    pub unwind: u32,
}

impl RuntimeFunction {
    /// Decode a record from the first [`RUNTIME_FUNCTION_LEN`] bytes of `bytes`.
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        Some(Self {
            begin: le_u32(bytes, 0)?,
            end: le_u32(bytes, 4)?,
            unwind: le_u32(bytes, 8)?,
        })
    }
}

/// Decoded `UNWIND_CODE` operation. Register numbers use the x64 encoding (0 = RAX ... 15 = R15,
/// XMM0 ... XMM15). Far/near variants of the same operation are folded together.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UnwindOp {
    /// `UWOP_PUSH_NONVOL`.
    PushNonVolatile { register: u8 },
    /// `UWOP_ALLOC_SMALL` / `UWOP_ALLOC_LARGE`: `sub rsp, size`.
    Alloc { size: u32 },
    /// `UWOP_SET_FPREG`: frame register = `rsp + frame_offset`.
    SetFramePointer,
    /// `UWOP_SAVE_NONVOL` / `UWOP_SAVE_NONVOL_FAR`: `mov [rsp+offset], reg`.
    SaveNonVolatile { register: u8, offset: u32 },
    /// `UWOP_SAVE_XMM128` / `UWOP_SAVE_XMM128_FAR`: `movaps [rsp+offset], xmm`.
    SaveXmm128 { register: u8, offset: u32 },
    /// `UWOP_PUSH_MACHFRAME` (hardware interrupt/exception frame).
    PushMachineFrame { error_code: bool },
    /// `UWOP_EPILOG` (version 2 only), raw: the first one carries the epilog size and flags,
    /// the following ones the epilog offsets from the end of the function.
    Epilog { code_offset: u8, info: u8 },
}

/// One decoded unwind code with its prolog offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnwindCode {
    /// Offset from the function start of the end of the prolog instruction it describes.
    pub prolog_offset: u8,
    pub op: UnwindOp,
}

/// Language-specific exception handler of an `UNWIND_INFO`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExceptionHandler {
    /// RVA of the handler routine (code).
    pub rva: u32,
    /// RVA of the language-specific handler data that follows the handler RVA.
    pub data_rva: u32,
}

/// Decoded `UNWIND_INFO`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnwindInfo {
    /// 1 or 2.
    pub version: u8,
    /// `UNW_FLAG_*` bits.
    pub flags: u8,
    pub prolog_size: u8,
    /// Frame register number (0 = none).
    pub frame_register: u8,
    /// Scaled frame register offset in bytes (`FrameOffset * 16`).
    pub frame_offset: u8,
    /// Unwind codes in table order (reverse prolog order).
    pub codes: Vec<UnwindCode>,
    /// Set when `EHANDLER` or `UHANDLER` is set and `CHAININFO` is not.
    pub handler: Option<ExceptionHandler>,
    /// Parent record when `CHAININFO` is set (takes precedence over handler flags).
    pub chained: Option<RuntimeFunction>,
}

/// Why an `UNWIND_INFO` could not be decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UnwindError {
    /// The unwind RVA does not map to file bytes.
    Unmapped,
    /// The structure runs past the mapped bytes.
    Truncated,
    /// Version other than 1 or 2.
    UnsupportedVersion(u8),
    /// Reserved opcode, invalid operation info, or a code that overruns `CountOfCodes`.
    InvalidCode { slot: u8, op: u8 },
}

impl UnwindInfo {
    /// Bytes needed by the structure described by the 4-byte header in `bytes`.
    pub fn encoded_len(bytes: &[u8]) -> Result<usize, UnwindError> {
        let header = bytes
            .get(..UNWIND_HEADER_LEN)
            .ok_or(UnwindError::Truncated)?;
        let flags = header[0] >> 3;
        let count = usize::from(header[2]);
        let mut len = UNWIND_HEADER_LEN + 2 * (count + (count & 1));
        if flags & UNW_FLAG_CHAININFO != 0 {
            len += RUNTIME_FUNCTION_LEN;
        } else if flags & (UNW_FLAG_EHANDLER | UNW_FLAG_UHANDLER) != 0 {
            len += 4;
        }
        Ok(len)
    }

    /// Decode an `UNWIND_INFO` located at `rva` from `bytes` (which may extend past it).
    pub fn parse(bytes: &[u8], rva: u32) -> Result<Self, UnwindError> {
        let len = Self::encoded_len(bytes)?;
        let bytes = bytes.get(..len).ok_or(UnwindError::Truncated)?;
        let version = bytes[0] & 0x7;
        if version != 1 && version != 2 {
            return Err(UnwindError::UnsupportedVersion(version));
        }
        let flags = bytes[0] >> 3;
        let count = usize::from(bytes[2]);
        let slots = &bytes[UNWIND_HEADER_LEN..UNWIND_HEADER_LEN + 2 * count];
        let codes = decode_codes(slots, version)?;

        let trailer = UNWIND_HEADER_LEN + 2 * (count + (count & 1));
        let (mut handler, mut chained) = (None, None);
        if flags & UNW_FLAG_CHAININFO != 0 {
            chained = RuntimeFunction::parse(&bytes[trailer..]);
        } else if flags & (UNW_FLAG_EHANDLER | UNW_FLAG_UHANDLER) != 0 {
            let data_rva = (trailer as u32 + 4)
                .checked_add(rva)
                .ok_or(UnwindError::Truncated)?;
            handler = le_u32(bytes, trailer).map(|rva| ExceptionHandler { rva, data_rva });
        }
        Ok(Self {
            version,
            flags,
            prolog_size: bytes[1],
            frame_register: bytes[3] & 0xf,
            frame_offset: (bytes[3] >> 4) * 16,
            codes,
            handler,
            chained,
        })
    }
}

/// Decode `CountOfCodes` slots. `UWOP_EPILOG` (6) only exists in version 2; the obsolete
/// version-1 meanings of 6 and 7 and the reserved 7, 11..15 are rejected.
fn decode_codes(slots: &[u8], version: u8) -> Result<Vec<UnwindCode>, UnwindError> {
    let count = slots.len() / 2;
    let slot = |i: usize| u16::from_le_bytes([slots[2 * i], slots[2 * i + 1]]);
    let mut codes = Vec::with_capacity(count);
    let mut i = 0;
    while i < count {
        let (prolog_offset, op_byte) = (slots[2 * i], slots[2 * i + 1]);
        let (op, info) = (op_byte & 0xf, op_byte >> 4);
        let invalid = UnwindError::InvalidCode { slot: i as u8, op };
        let used = match op {
            1 if info == 1 => 3,
            1 if info == 0 => 2,
            4 | 8 => 2,
            5 | 9 => 3,
            0 | 2 | 3 | 10 => 1,
            6 if version == 2 => 1,
            _ => return Err(invalid),
        };
        if i + used > count {
            return Err(invalid);
        }
        let far = || u32::from(slot(i + 1)) | (u32::from(slot(i + 2)) << 16);
        let op = match op {
            0 => UnwindOp::PushNonVolatile { register: info },
            1 if info == 0 => UnwindOp::Alloc {
                size: u32::from(slot(i + 1)) * 8,
            },
            1 => UnwindOp::Alloc { size: far() },
            2 => UnwindOp::Alloc {
                size: u32::from(info) * 8 + 8,
            },
            3 => UnwindOp::SetFramePointer,
            4 => UnwindOp::SaveNonVolatile {
                register: info,
                offset: u32::from(slot(i + 1)) * 8,
            },
            5 => UnwindOp::SaveNonVolatile {
                register: info,
                offset: far(),
            },
            6 => UnwindOp::Epilog {
                code_offset: prolog_offset,
                info,
            },
            8 => UnwindOp::SaveXmm128 {
                register: info,
                offset: u32::from(slot(i + 1)) * 16,
            },
            9 => UnwindOp::SaveXmm128 {
                register: info,
                offset: far(),
            },
            10 if info <= 1 => UnwindOp::PushMachineFrame {
                error_code: info == 1,
            },
            _ => return Err(invalid),
        };
        codes.push(UnwindCode { prolog_offset, op });
        i += used;
    }
    Ok(codes)
}

/// Category of an exception-directory problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PdataDiagnosticKind {
    /// The directory RVA does not map to file bytes; nothing was read.
    DirectoryUnmapped,
    /// The directory runs past its section's file bytes; the records before that were read.
    DirectoryTruncated,
    /// The directory size is not a multiple of 12; the trailing bytes were ignored.
    TrailingBytes,
    /// Records are not sorted by `BeginAddress` (the loader binary-searches them); sorted here.
    Unsorted,
    /// `BeginAddress >= EndAddress`; record dropped.
    EmptyRange,
    /// `EndAddress` is past `SizeOfImage`; record dropped.
    OutsideImage,
    /// `BeginAddress` is not inside an executable section; record dropped.
    NotExecutable,
    /// The range overlaps an earlier record; record dropped.
    Overlap,
    /// The record's `UNWIND_INFO` could not be decoded; the range is kept without unwind info.
    Unwind(UnwindError),
    /// A chained/indirect parent could not be read or is not a record of this directory; the
    /// chunk is treated as its own function.
    ChainBroken,
    /// More than 32 chained links (or a cycle); the chunk is treated as its own function.
    ChainTooDeep,
    /// The exception handler RVA is not inside an executable section.
    HandlerNotCode,
}

impl PdataDiagnosticKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DirectoryUnmapped => "directory_unmapped",
            Self::DirectoryTruncated => "directory_truncated",
            Self::TrailingBytes => "trailing_bytes",
            Self::Unsorted => "unsorted",
            Self::EmptyRange => "empty_range",
            Self::OutsideImage => "outside_image",
            Self::NotExecutable => "not_executable",
            Self::Overlap => "overlap",
            Self::Unwind(UnwindError::Unmapped) => "unwind_unmapped",
            Self::Unwind(UnwindError::Truncated) => "unwind_truncated",
            Self::Unwind(UnwindError::UnsupportedVersion(_)) => "unwind_unsupported_version",
            Self::Unwind(UnwindError::InvalidCode { .. }) => "unwind_invalid_code",
            Self::ChainBroken => "chain_broken",
            Self::ChainTooDeep => "chain_too_deep",
            Self::HandlerNotCode => "handler_not_code",
        }
    }
}

/// A non-fatal exception-directory problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PdataDiagnostic {
    /// `BeginAddress` of the affected record, or the directory RVA for directory-level problems.
    pub rva: u32,
    pub kind: PdataDiagnosticKind,
}

impl fmt::Display for PdataDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at RVA 0x{:08X}", self.kind.as_str(), self.rva)
    }
}

/// A validated `RUNTIME_FUNCTION` record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdataEntry {
    pub function: RuntimeFunction,
    /// `None` for indirect records and when the unwind info is malformed (see diagnostics).
    pub unwind: Option<UnwindInfo>,
    /// `BeginAddress` of the primary function this chunk belongs to (its own begin unless
    /// chained or indirect).
    pub primary: u32,
}

/// Parsed exception directory.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExceptionDirectory {
    /// Number of records read from the directory (before validation).
    pub record_count: usize,
    /// Valid records sorted by `BeginAddress`, deduplicated and non-overlapping.
    pub entries: Vec<PdataEntry>,
    /// Problems in directory order, then per record.
    pub diagnostics: Vec<PdataDiagnostic>,
}

/// Bounds-checked RVA reads through R1's section table.
struct Reader<'p, 'f> {
    pe: &'p Pe64,
    file: &'f [u8],
}

impl<'f> Reader<'_, 'f> {
    /// `len` bytes at `rva`, only if they are contiguous file bytes of one mapping.
    fn read(&self, rva: u32, len: usize) -> Option<&'f [u8]> {
        let offset = self.pe.rva_to_file_offset(rva)?;
        if len > 1 {
            let last = rva.checked_add(u32::try_from(len - 1).ok()?)?;
            if self.pe.rva_to_file_offset(last)? != offset.checked_add(len - 1)? {
                return None;
            }
        }
        self.file.get(offset..offset.checked_add(len)?)
    }

    fn unwind_info(&self, rva: u32) -> Result<UnwindInfo, UnwindError> {
        let header = self
            .read(rva, UNWIND_HEADER_LEN)
            .ok_or(UnwindError::Unmapped)?;
        let len = UnwindInfo::encoded_len(header)?;
        let bytes = self.read(rva, len).ok_or(UnwindError::Truncated)?;
        UnwindInfo::parse(bytes, rva)
    }

    /// Record pointed to by an indirect `UnwindInfoAddress`.
    fn indirect(&self, unwind: u32) -> Option<RuntimeFunction> {
        RuntimeFunction::parse(self.read(unwind - 1, RUNTIME_FUNCTION_LEN)?)
    }

    fn is_executable(&self, rva: u32) -> bool {
        self.pe.sections.iter().any(|s| {
            let extent = if s.virtual_size == 0 {
                s.raw_size
            } else {
                s.virtual_size
            };
            s.characteristics & IMAGE_SCN_MEM_EXECUTE != 0
                && rva
                    .checked_sub(s.virtual_address)
                    .is_some_and(|delta| delta < extent)
        })
    }

    /// Follow the parent links starting at `link` to the record with plain unwind info.
    fn primary_of(&self, mut link: RuntimeFunction) -> Result<u32, PdataDiagnosticKind> {
        for _ in 0..MAX_CHAIN_DEPTH {
            let next = if link.unwind & RUNTIME_FUNCTION_INDIRECT != 0 {
                self.indirect(link.unwind)
            } else {
                match self.unwind_info(link.unwind) {
                    Ok(info) => match info.chained {
                        Some(parent) => Some(parent),
                        None => return Ok(link.begin),
                    },
                    Err(_) => None,
                }
            };
            link = next.ok_or(PdataDiagnosticKind::ChainBroken)?;
        }
        Err(PdataDiagnosticKind::ChainTooDeep)
    }
}

impl ExceptionDirectory {
    /// Parse the exception directory of `pe`; `file` must be the bytes `pe` was parsed from.
    /// Returns an empty directory when the PE has none.
    pub fn from_pe(pe: &Pe64, file: &[u8]) -> Self {
        let mut out = Self::default();
        let Some(dir) = pe.optional.data_directory(DIR_EXCEPTION) else {
            return out;
        };
        let reader = Reader { pe, file };
        let mut diag = |rva, kind| out.diagnostics.push(PdataDiagnostic { rva, kind });

        let size = dir.size as usize;
        if !size.is_multiple_of(RUNTIME_FUNCTION_LEN) {
            diag(dir.virtual_address, PdataDiagnosticKind::TrailingBytes);
        }
        let mut records = Vec::new();
        for i in 0..size / RUNTIME_FUNCTION_LEN {
            let record = u32::try_from(i * RUNTIME_FUNCTION_LEN)
                .ok()
                .and_then(|offset| dir.virtual_address.checked_add(offset))
                .and_then(|rva| reader.read(rva, RUNTIME_FUNCTION_LEN))
                .and_then(RuntimeFunction::parse);
            match record {
                Some(record) => records.push(record),
                None => {
                    let kind = if i == 0 {
                        PdataDiagnosticKind::DirectoryUnmapped
                    } else {
                        PdataDiagnosticKind::DirectoryTruncated
                    };
                    diag(dir.virtual_address, kind);
                    break;
                }
            }
        }
        out.record_count = records.len();
        if records.windows(2).any(|w| w[1].begin < w[0].begin) {
            diag(dir.virtual_address, PdataDiagnosticKind::Unsorted);
        }

        // Range validation, then sort / dedup / drop overlaps.
        let mut valid = Vec::with_capacity(records.len());
        for record in records {
            let kind = if record.begin >= record.end {
                PdataDiagnosticKind::EmptyRange
            } else if record.end > pe.optional.size_of_image {
                PdataDiagnosticKind::OutsideImage
            } else if !reader.is_executable(record.begin) {
                PdataDiagnosticKind::NotExecutable
            } else {
                valid.push(record);
                continue;
            };
            diag(record.begin, kind);
        }
        valid.sort_unstable();
        valid.dedup();
        let mut kept: Vec<RuntimeFunction> = Vec::with_capacity(valid.len());
        for record in valid {
            if kept.last().is_some_and(|prev| record.begin < prev.end) {
                diag(record.begin, PdataDiagnosticKind::Overlap);
            } else {
                kept.push(record);
            }
        }

        // Unwind info and primary function of every kept record.
        let mut entries = Vec::with_capacity(kept.len());
        for function in kept {
            let (unwind, link) = if function.unwind & RUNTIME_FUNCTION_INDIRECT != 0 {
                (None, Some(function))
            } else {
                match reader.unwind_info(function.unwind) {
                    Ok(info) => {
                        if info.handler.is_some_and(|h| !reader.is_executable(h.rva)) {
                            diag(function.begin, PdataDiagnosticKind::HandlerNotCode);
                        }
                        let link = info.chained.is_some().then_some(function);
                        (Some(info), link)
                    }
                    Err(err) => {
                        diag(function.begin, PdataDiagnosticKind::Unwind(err));
                        (None, None)
                    }
                }
            };
            let primary = match link.map(|link| reader.primary_of(link)) {
                None => function.begin,
                Some(Ok(primary)) => primary,
                Some(Err(kind)) => {
                    diag(function.begin, kind);
                    function.begin
                }
            };
            entries.push(PdataEntry {
                function,
                unwind,
                primary,
            });
        }

        // A primary must itself be a record of this directory.
        let begins: Vec<u32> = entries.iter().map(|e| e.function.begin).collect();
        for entry in &mut entries {
            if entry.primary != entry.function.begin
                && begins.binary_search(&entry.primary).is_err()
            {
                diag(entry.function.begin, PdataDiagnosticKind::ChainBroken);
                entry.primary = entry.function.begin;
            }
        }
        out.entries = entries;
        out
    }

    /// Exact function chunk ranges as absolute VAs, each mapped to its primary function.
    pub fn function_table(&self, image_base: u64) -> FunctionTable {
        let va = |rva: u32| image_base.saturating_add(u64::from(rva));
        FunctionTable::new(
            self.entries
                .iter()
                .map(|e| FunctionRange {
                    start: va(e.function.begin),
                    end: va(e.function.end),
                    function: va(e.primary),
                })
                .collect(),
        )
    }

    /// RVAs of the exception handlers referenced by unwind info, sorted and deduplicated.
    pub fn handler_rvas(&self) -> Vec<u32> {
        let mut rvas: Vec<u32> = self
            .entries
            .iter()
            .filter_map(|e| e.unwind.as_ref()?.handler.map(|h| h.rva))
            .collect();
        rvas.sort_unstable();
        rvas.dedup();
        rvas
    }

    /// Feed the directory into discovery: install [`Self::function_table`] on `image` and add
    /// every primary function start and exception handler that lies in executable code to its
    /// entry points (kept sorted and deduplicated).
    pub fn seed(&self, image: &mut CodeImage<'_>) {
        let base = image.base();
        image.function_table = self.function_table(base);
        let handlers = self
            .handler_rvas()
            .into_iter()
            .map(|rva| base.saturating_add(u64::from(rva)));
        let seeds: Vec<u64> = image
            .function_table
            .function_entries()
            .chain(handlers)
            .filter(|&va| image.region_of(va) == Region::Code)
            .collect();
        let mut entries = std::mem::take(&mut image.entry_points);
        entries.extend(seeds);
        entries.sort_unstable();
        entries.dedup();
        image.entry_points = entries;
    }
}

/// One function chunk `[start, end)` owned by the function whose entry is `function` (VAs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FunctionRange {
    pub start: u64,
    pub end: u64,
    pub function: u64,
}

impl FunctionRange {
    pub fn contains(&self, va: u64) -> bool {
        va >= self.start && va < self.end
    }
}

/// Sorted, non-overlapping function chunk ranges (from `.pdata`), used to bound analysis.
/// Empty when the image has no exception directory.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FunctionTable {
    ranges: Vec<FunctionRange>,
}

impl FunctionTable {
    /// Build a table from arbitrary ranges: empty ranges are dropped, the rest sorted and
    /// deduplicated, and a range overlapping an earlier one is dropped.
    pub fn new(mut ranges: Vec<FunctionRange>) -> Self {
        ranges.retain(|r| r.start < r.end);
        ranges.sort_unstable();
        ranges.dedup();
        let mut kept: Vec<FunctionRange> = Vec::with_capacity(ranges.len());
        for range in ranges {
            if kept.last().is_none_or(|prev| range.start >= prev.end) {
                kept.push(range);
            }
        }
        Self { ranges: kept }
    }

    pub fn ranges(&self) -> &[FunctionRange] {
        &self.ranges
    }

    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Chunk containing `va`.
    pub fn range_at(&self, va: u64) -> Option<&FunctionRange> {
        let idx = self.ranges.partition_point(|r| r.start <= va);
        let range = self.ranges.get(idx.checked_sub(1)?)?;
        range.contains(va).then_some(range)
    }

    /// Entry of the function owning `va`, or `None` when no range covers it.
    pub fn function_at(&self, va: u64) -> Option<u64> {
        self.range_at(va).map(|r| r.function)
    }

    /// Primary function entries in ascending order.
    pub fn function_entries(&self) -> impl Iterator<Item = u64> + '_ {
        self.ranges
            .iter()
            .filter(|r| r.start == r.function)
            .map(|r| r.start)
    }

    /// Number of distinct primary functions.
    pub fn function_count(&self) -> usize {
        self.function_entries().count()
    }
}

fn le_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let b = bytes.get(offset..offset.checked_add(4)?)?;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::pe::test_util::build_synthetic_pe64;

    pub(crate) const BASE: u64 = 0x1_4000_0000;
    /// File offset of the exception data directory entry in the synthetic PE.
    const DIR_ENTRY: usize = 0x108 + DIR_EXCEPTION * 8;
    /// `.pdata` records are placed at this `.rdata` RVA.
    pub(crate) const PDATA_RVA: u32 = 0x2560;

    /// File offset of `rva` in the synthetic PE (`.text` at 0x1000 / file 0x200, `.rdata` at
    /// 0x2000 / file 0x400).
    pub(crate) fn file_offset(rva: u32) -> usize {
        match rva {
            0x1000..0x1200 => rva as usize - 0x1000 + 0x200,
            0x2000..0x2800 => rva as usize - 0x2000 + 0x400,
            _ => panic!("RVA {rva:#x} is not backed by the synthetic PE"),
        }
    }

    pub(crate) fn put(file: &mut [u8], rva: u32, bytes: &[u8]) {
        let at = file_offset(rva);
        file[at..at + bytes.len()].copy_from_slice(bytes);
    }

    /// R1's synthetic PE with `.text` grown to 0x200 bytes, `.rdata` to 0x800, and an exception
    /// directory at [`PDATA_RVA`] holding `records` (`begin, end, unwind` RVAs).
    pub(crate) fn pe_with_pdata(records: &[(u32, u32, u32)]) -> Vec<u8> {
        let mut file = build_synthetic_pe64(false);
        file[0x190..0x194].copy_from_slice(&0x200u32.to_le_bytes()); // .text VirtualSize
        file[0x1b8..0x1bc].copy_from_slice(&0x800u32.to_le_bytes()); // .rdata VirtualSize
        for (i, &(begin, end, unwind)) in records.iter().enumerate() {
            let rva = PDATA_RVA + 12 * i as u32;
            put(&mut file, rva, &begin.to_le_bytes());
            put(&mut file, rva + 4, &end.to_le_bytes());
            put(&mut file, rva + 8, &unwind.to_le_bytes());
        }
        set_pdata_dir(&mut file, PDATA_RVA, 12 * records.len() as u32);
        file
    }

    pub(crate) fn set_pdata_dir(file: &mut [u8], rva: u32, size: u32) {
        file[DIR_ENTRY..DIR_ENTRY + 4].copy_from_slice(&rva.to_le_bytes());
        file[DIR_ENTRY + 4..DIR_ENTRY + 8].copy_from_slice(&size.to_le_bytes());
    }

    /// Minimal version-1 `UNWIND_INFO` (`sub rsp, 0x28`) with the given flags and trailer.
    pub(crate) fn unwind_bytes(flags: u8, trailer: &[u8]) -> Vec<u8> {
        let mut bytes = vec![
            0x01 | (flags << 3),
            0x04,
            0x01,
            0x00,
            0x04,
            0x42,
            0x00,
            0x00,
        ];
        bytes.extend_from_slice(trailer);
        bytes
    }

    fn parse(file: &[u8]) -> ExceptionDirectory {
        let pe = Pe64::parse(file).expect("synthetic PE parses");
        ExceptionDirectory::from_pe(&pe, file)
    }

    fn kinds(dir: &ExceptionDirectory) -> Vec<(u32, PdataDiagnosticKind)> {
        dir.diagnostics.iter().map(|d| (d.rva, d.kind)).collect()
    }

    const UNWIND_A: u32 = 0x2700;
    const UNWIND_B: u32 = 0x2720;
    const UNWIND_C: u32 = 0x2740;
    const UNWIND_D: u32 = 0x2760;

    #[test]
    fn decodes_every_unwind_code() {
        let bytes = [
            0x02 | (UNW_FLAG_EHANDLER << 3), // version 2, EHANDLER
            0x30,                            // prolog size
            0x15,                            // 21 slots
            0x35,                            // frame register RBP, offset 3 * 16
            0x05,
            0x16, // epilog: size 5, flags 1
            0x20,
            0x06, // epilog: offset 0x20
            0x01,
            0x50, // push rbp
            0x05,
            0x32, // alloc small 0x20
            0x0c,
            0x01,
            0x00,
            0x02, // alloc large 0x200 * 8
            0x10,
            0x11,
            0x00,
            0x00,
            0x01,
            0x00, // alloc large far 0x10000
            0x12,
            0x03, // set fpreg
            0x14,
            0x34,
            0x06,
            0x00, // save rbx at 6 * 8
            0x18,
            0x65,
            0x45,
            0x23,
            0x01,
            0x00, // save rsi far at 0x12345
            0x1c,
            0x68,
            0x02,
            0x00, // save xmm6 at 2 * 16
            0x20,
            0x79,
            0x00,
            0x10,
            0x00,
            0x00, // save xmm7 far at 0x1000
            0x22,
            0x1a, // push machframe with error code
            0x00,
            0x00, // padding to an even slot count
            0x00,
            0x15,
            0x00,
            0x00, // handler RVA 0x1500
            0xaa,
            0xbb, // language-specific data (not part of the structure)
        ];
        assert_eq!(UnwindInfo::encoded_len(&bytes), Ok(4 + 44 + 4));
        let info = UnwindInfo::parse(&bytes, 0x2600).expect("valid unwind info");
        assert_eq!(
            (info.version, info.flags, info.prolog_size),
            (2, UNW_FLAG_EHANDLER, 0x30)
        );
        assert_eq!((info.frame_register, info.frame_offset), (5, 0x30));
        let ops: Vec<_> = info.codes.iter().map(|c| (c.prolog_offset, c.op)).collect();
        assert_eq!(
            ops,
            [
                (
                    0x05,
                    UnwindOp::Epilog {
                        code_offset: 5,
                        info: 1
                    }
                ),
                (
                    0x20,
                    UnwindOp::Epilog {
                        code_offset: 0x20,
                        info: 0
                    }
                ),
                (0x01, UnwindOp::PushNonVolatile { register: 5 }),
                (0x05, UnwindOp::Alloc { size: 0x20 }),
                (0x0c, UnwindOp::Alloc { size: 0x1000 }),
                (0x10, UnwindOp::Alloc { size: 0x10000 }),
                (0x12, UnwindOp::SetFramePointer),
                (
                    0x14,
                    UnwindOp::SaveNonVolatile {
                        register: 3,
                        offset: 0x30
                    }
                ),
                (
                    0x18,
                    UnwindOp::SaveNonVolatile {
                        register: 6,
                        offset: 0x12345
                    }
                ),
                (
                    0x1c,
                    UnwindOp::SaveXmm128 {
                        register: 6,
                        offset: 0x20
                    }
                ),
                (
                    0x20,
                    UnwindOp::SaveXmm128 {
                        register: 7,
                        offset: 0x1000
                    }
                ),
                (0x22, UnwindOp::PushMachineFrame { error_code: true }),
            ]
        );
        assert_eq!(
            info.handler,
            Some(ExceptionHandler {
                rva: 0x1500,
                data_rva: 0x2600 + 52,
            })
        );
        assert_eq!(info.chained, None);
    }

    #[test]
    fn rejects_malformed_unwind_info() {
        assert_eq!(
            UnwindInfo::parse(&[0x01, 0x00], 0),
            Err(UnwindError::Truncated)
        );
        assert_eq!(
            UnwindInfo::parse(&[0x03, 0, 0, 0], 0),
            Err(UnwindError::UnsupportedVersion(3))
        );
        // CountOfCodes = 2 but only one slot present.
        assert_eq!(
            UnwindInfo::parse(&[0x01, 0, 2, 0, 0x01, 0x50], 0),
            Err(UnwindError::Truncated)
        );
        // UWOP_ALLOC_LARGE needs 2 slots, only 1 counted.
        assert_eq!(
            UnwindInfo::parse(&[0x01, 0, 1, 0, 0x04, 0x01, 0, 0], 0),
            Err(UnwindError::InvalidCode { slot: 0, op: 1 })
        );
        // UWOP_EPILOG in version 1, reserved opcode 11, machframe info > 1.
        for (op_byte, op) in [(0x06u8, 6u8), (0x0b, 11), (0x2a, 10)] {
            assert_eq!(
                UnwindInfo::parse(&[0x01, 0, 1, 0, 0x00, op_byte, 0, 0], 0),
                Err(UnwindError::InvalidCode { slot: 0, op })
            );
        }
        // CHAININFO set but the parent record is cut off.
        assert_eq!(
            UnwindInfo::parse(&unwind_bytes(UNW_FLAG_CHAININFO, &[0; 8]), 0),
            Err(UnwindError::Truncated)
        );
    }

    #[test]
    fn parses_runtime_functions_into_sorted_ranges() {
        let mut file = pe_with_pdata(&[
            (0x1010, 0x1018, UNWIND_A),
            (0x1060, 0x1070, UNWIND_A), // shares unwind info (linker folding)
            (0x1020, 0x1030, UNWIND_B),
        ]);
        put(&mut file, UNWIND_A, &unwind_bytes(0, &[]));
        put(
            &mut file,
            UNWIND_B,
            &unwind_bytes(UNW_FLAG_UHANDLER, &[0x40, 0x10, 0, 0]),
        );
        let dir = parse(&file);
        assert_eq!(dir.record_count, 3);
        assert_eq!(kinds(&dir), [(PDATA_RVA, PdataDiagnosticKind::Unsorted)]);
        let begins: Vec<u32> = dir.entries.iter().map(|e| e.function.begin).collect();
        assert_eq!(begins, [0x1010, 0x1020, 0x1060]);
        assert!(dir.entries.iter().all(|e| e.primary == e.function.begin));
        let info = dir.entries[0].unwind.as_ref().expect("unwind info");
        assert_eq!(
            info.codes,
            [UnwindCode {
                prolog_offset: 4,
                op: UnwindOp::Alloc { size: 0x28 },
            }]
        );
        assert_eq!(dir.handler_rvas(), [0x1040]);
        assert_eq!(
            dir.entries[1].unwind.as_ref().and_then(|u| u.handler),
            Some(ExceptionHandler {
                rva: 0x1040,
                data_rva: UNWIND_B + 12,
            })
        );

        let table = dir.function_table(BASE);
        assert_eq!(table.function_count(), 3);
        assert_eq!(table.function_at(BASE + 0x1017), Some(BASE + 0x1010));
        assert_eq!(table.function_at(BASE + 0x1018), None);
        assert_eq!(table.function_at(BASE + 0x106f), Some(BASE + 0x1060));
        assert_eq!(table.function_at(BASE + 0x100f), None);
    }

    #[test]
    fn chained_and_indirect_chunks_map_to_their_primary() {
        let mut file = pe_with_pdata(&[
            (0x1010, 0x1020, UNWIND_A),
            (0x1080, 0x1090, UNWIND_B),      // chained to 0x1010
            (0x10a0, 0x10b0, UNWIND_C),      // chained to 0x1080 (two links)
            (0x10c0, 0x10c8, PDATA_RVA + 1), // indirect: record 0 holds the unwind data
        ]);
        put(&mut file, UNWIND_A, &unwind_bytes(0, &[]));
        let parent = |begin: u32, end: u32, unwind: u32| {
            [begin.to_le_bytes(), end.to_le_bytes(), unwind.to_le_bytes()].concat()
        };
        put(
            &mut file,
            UNWIND_B,
            &unwind_bytes(UNW_FLAG_CHAININFO, &parent(0x1010, 0x1020, UNWIND_A)),
        );
        put(
            &mut file,
            UNWIND_C,
            &unwind_bytes(UNW_FLAG_CHAININFO, &parent(0x1080, 0x1090, UNWIND_B)),
        );
        let dir = parse(&file);
        assert!(dir.diagnostics.is_empty(), "{:?}", dir.diagnostics);
        let primaries: Vec<(u32, u32)> = dir
            .entries
            .iter()
            .map(|e| (e.function.begin, e.primary))
            .collect();
        assert_eq!(
            primaries,
            [
                (0x1010, 0x1010),
                (0x1080, 0x1010),
                (0x10a0, 0x1010),
                (0x10c0, 0x1010)
            ]
        );
        assert_eq!(
            dir.entries[1].unwind.as_ref().and_then(|u| u.chained),
            Some(RuntimeFunction {
                begin: 0x1010,
                end: 0x1020,
                unwind: UNWIND_A,
            })
        );
        assert_eq!(dir.entries[3].unwind, None, "indirect record");

        let table = dir.function_table(BASE);
        assert_eq!(table.function_count(), 1);
        assert_eq!(table.ranges().len(), 4);
        assert_eq!(table.function_at(BASE + 0x10a8), Some(BASE + 0x1010));
        assert_eq!(
            table.function_entries().collect::<Vec<_>>(),
            [BASE + 0x1010]
        );
    }

    #[test]
    fn broken_and_cyclic_chains_fall_back_to_own_function() {
        let mut file = pe_with_pdata(&[
            (0x1010, 0x1020, UNWIND_A), // chained to 0x1030, which chains back to 0x1010
            (0x1030, 0x1040, UNWIND_B),
            (0x1050, 0x1060, UNWIND_C), // chained to a record that is not in the directory
            (0x1070, 0x1078, 0x1fff),   // indirect into nothing (RVA 0x1ffe is unmapped)
        ]);
        let link = |begin: u32, end: u32, unwind: u32| {
            [begin.to_le_bytes(), end.to_le_bytes(), unwind.to_le_bytes()].concat()
        };
        put(
            &mut file,
            UNWIND_A,
            &unwind_bytes(UNW_FLAG_CHAININFO, &link(0x1030, 0x1040, UNWIND_B)),
        );
        put(
            &mut file,
            UNWIND_B,
            &unwind_bytes(UNW_FLAG_CHAININFO, &link(0x1010, 0x1020, UNWIND_A)),
        );
        put(
            &mut file,
            UNWIND_C,
            &unwind_bytes(UNW_FLAG_CHAININFO, &link(0x1100, 0x1110, UNWIND_D)),
        );
        put(&mut file, UNWIND_D, &unwind_bytes(0, &[]));
        let dir = parse(&file);
        assert_eq!(
            kinds(&dir),
            [
                (0x1010, PdataDiagnosticKind::ChainTooDeep),
                (0x1030, PdataDiagnosticKind::ChainTooDeep),
                (0x1070, PdataDiagnosticKind::ChainBroken),
                (0x1050, PdataDiagnosticKind::ChainBroken),
            ]
        );
        assert!(dir.entries.iter().all(|e| e.primary == e.function.begin));
    }

    #[test]
    fn invalid_records_are_dropped_with_diagnostics() {
        let mut file = pe_with_pdata(&[
            (0x1010, 0x1020, UNWIND_A),
            (0x1010, 0x1020, UNWIND_A), // exact duplicate: silently merged
            (0x1018, 0x1028, UNWIND_A), // overlaps
            (0x1030, 0x1030, UNWIND_A), // empty
            (0x1040, 0x4000, UNWIND_A), // past SizeOfImage
            (0x1050, 0x1058, 0x2ff0),   // unwind info unmapped (past .rdata)
            (0x1060, 0x1068, UNWIND_B), // bad version
            (0x1070, 0x1078, 0x27fe),   // unwind header crosses the end of .rdata
            (0x2010, 0x2020, UNWIND_A), // in .rdata
        ]);
        put(&mut file, UNWIND_A, &unwind_bytes(0, &[]));
        put(&mut file, UNWIND_B, &[0x05, 0, 0, 0]);
        let dir = parse(&file);
        assert_eq!(
            kinds(&dir),
            [
                (0x1030, PdataDiagnosticKind::EmptyRange),
                (0x1040, PdataDiagnosticKind::OutsideImage),
                (0x2010, PdataDiagnosticKind::NotExecutable),
                (0x1018, PdataDiagnosticKind::Overlap),
                (0x1050, PdataDiagnosticKind::Unwind(UnwindError::Unmapped)),
                (
                    0x1060,
                    PdataDiagnosticKind::Unwind(UnwindError::UnsupportedVersion(5))
                ),
                (0x1070, PdataDiagnosticKind::Unwind(UnwindError::Unmapped)),
            ]
        );
        let kept: Vec<(u32, bool)> = dir
            .entries
            .iter()
            .map(|e| (e.function.begin, e.unwind.is_some()))
            .collect();
        assert_eq!(
            kept,
            [
                (0x1010, true),
                (0x1050, false),
                (0x1060, false),
                (0x1070, false)
            ]
        );
    }

    #[test]
    fn malformed_directories_never_panic() {
        // Missing directory.
        let file = build_synthetic_pe64(false);
        assert_eq!(parse(&file), ExceptionDirectory::default());

        // Directory RVA outside every section.
        let mut file = pe_with_pdata(&[]);
        set_pdata_dir(&mut file, 0x9000, 24);
        let dir = parse(&file);
        assert!(dir.entries.is_empty());
        assert_eq!(
            kinds(&dir),
            [(0x9000, PdataDiagnosticKind::DirectoryUnmapped)]
        );

        // Directory running past the end of .rdata, with a size that is not a multiple of 12.
        let mut file = pe_with_pdata(&[(0x1010, 0x1020, UNWIND_A)]);
        put(&mut file, UNWIND_A, &unwind_bytes(0, &[]));
        set_pdata_dir(&mut file, 0x27f4, 12 * 3 + 5);
        put(
            &mut file,
            0x27f4,
            &[0x10, 0x10, 0, 0, 0x20, 0x10, 0, 0, 0x00, 0x27, 0, 0],
        );
        let dir = parse(&file);
        assert_eq!(dir.record_count, 1);
        assert_eq!(
            kinds(&dir),
            [
                (0x27f4, PdataDiagnosticKind::TrailingBytes),
                (0x27f4, PdataDiagnosticKind::DirectoryTruncated),
            ]
        );
        assert_eq!(dir.entries.len(), 1);

        // Truncated file: records past the end of the file are not read.
        let mut file = pe_with_pdata(&[(0x1010, 0x1020, UNWIND_A); 4]);
        let pe = Pe64::parse(&file).expect("parses");
        file.truncate(file_offset(PDATA_RVA) + 18);
        let dir = ExceptionDirectory::from_pe(&pe, &file);
        assert_eq!(dir.record_count, 1);

        // Garbage everywhere (headers parsed before the bytes are scrambled).
        let mut file = pe_with_pdata(&[]);
        set_pdata_dir(&mut file, 0x2000, 0x7f0);
        let pe = Pe64::parse(&file).expect("parses");
        let mut seed = 0x1234_5678u32;
        for rva in 0x2000..0x2800 {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            put(&mut file, rva, &[(seed >> 16) as u8]);
        }
        let dir = ExceptionDirectory::from_pe(&pe, &file);
        assert!(
            dir.entries
                .windows(2)
                .all(|w| w[0].function.end <= w[1].function.begin)
        );
    }

    #[test]
    fn function_table_lookup_and_overlap_handling() {
        let r = |start, end, function| FunctionRange {
            start,
            end,
            function,
        };
        let table = FunctionTable::new(vec![
            r(0x30, 0x40, 0x10),
            r(0x10, 0x20, 0x10),
            r(0x18, 0x28, 0x18), // overlaps: dropped
            r(0x50, 0x50, 0x50), // empty: dropped
            r(0x10, 0x20, 0x10),
        ]);
        assert_eq!(table.ranges(), &[r(0x10, 0x20, 0x10), r(0x30, 0x40, 0x10)]);
        assert_eq!(table.function_count(), 1);
        assert_eq!(table.function_at(0x35), Some(0x10));
        assert_eq!(table.function_at(0x20), None);
        assert_eq!(table.function_at(0x0), None);
        assert!(FunctionTable::default().function_at(0x10).is_none());
    }

    #[test]
    fn seed_adds_pdata_functions_and_handlers_as_entry_points() {
        let mut file = pe_with_pdata(&[
            (0x1060, 0x1070, UNWIND_A),
            (0x1080, 0x1090, UNWIND_B),
            (0x10b0, 0x10b8, UNWIND_C),
        ]);
        put(&mut file, UNWIND_A, &unwind_bytes(0, &[]));
        // Handler at .text+0xa0, and one that points into .rdata (reported, not seeded).
        put(
            &mut file,
            UNWIND_B,
            &unwind_bytes(UNW_FLAG_EHANDLER, &[0xa0, 0x10, 0, 0]),
        );
        put(
            &mut file,
            UNWIND_C,
            &unwind_bytes(UNW_FLAG_EHANDLER, &[0x10, 0x20, 0, 0]),
        );
        let pe = Pe64::parse(&file).expect("parses");
        let dir = ExceptionDirectory::from_pe(&pe, &file);
        assert_eq!(kinds(&dir), [(0x10b0, PdataDiagnosticKind::HandlerNotCode)]);
        assert_eq!(dir.handler_rvas(), [0x10a0, 0x2010]);
        let mut image = CodeImage::from_pe(&pe, &file);
        dir.seed(&mut image);
        assert_eq!(
            image.entry_points,
            [
                0x1010, 0x1020, 0x1030, 0x1040, 0x1060, 0x1080, 0x10a0, 0x10b0
            ]
            .map(|rva| BASE + rva)
        );
        assert_eq!(image.function_table.function_count(), 3);
    }

    #[test]
    fn parse_is_deterministic() {
        let mut file = pe_with_pdata(&[(0x1080, 0x1090, UNWIND_B), (0x1010, 0x1020, UNWIND_A)]);
        put(&mut file, UNWIND_A, &unwind_bytes(0, &[]));
        put(
            &mut file,
            UNWIND_B,
            &unwind_bytes(UNW_FLAG_EHANDLER, &[0x10, 0x10, 0, 0]),
        );
        assert_eq!(parse(&file), parse(&file));
    }
}
