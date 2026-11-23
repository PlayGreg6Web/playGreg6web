//! Guest memory image packer: lays out a PE32+ image exactly as the Windows loader would map it,
//! parses its base-relocation directory, and serialises both into a deterministic container
//! (`PGIM`) that the runtime streams into `VirtualMemoryArena`.
//!
//! # Guest image model
//!
//! [`build_guest_image`] turns a parsed [`Pe64`] plus the file bytes into a [`GuestImage`]:
//!
//! - Every section with a non-zero in-memory size is placed at its exact RVA. Its in-memory size is
//!   `VirtualSize`, or `SizeOfRawData` when `VirtualSize` is 0 (same rule as
//!   [`CodeImage::from_pe`](crate::code_image::CodeImage::from_pe)). Initialised bytes come from
//!   the raw data (clamped to the in-memory size and the file); everything past them is zero-fill.
//!   Trailing zero bytes of the initialised data are trimmed because they are zero-fill anyway.
//!   `.bss`-style sections (`SizeOfRawData == 0`) are pure zero-fill. Discardable sections such as
//!   `.reloc` stay mapped, like the Windows loader does.
//! - Protection comes from `IMAGE_SCN_MEM_READ` / `_WRITE` / `_EXECUTE`.
//! - **PE headers are mapped by default** ([`ImageOptions::map_headers`]) as a read-only
//!   pseudo-section `<headers>` at RVA 0 with `min(SizeOfHeaders, first section RVA)` bytes. The
//!   Windows loader maps them too, and real binaries read their own headers through
//!   `__ImageBase` / `GetModuleHandle(NULL)` (MSVC CRT `_IsNonwritableInCurrentImage`, `.pdata`
//!   and resource lookups, TLS setup). Recompiled code therefore needs them at the image base.
//! - **Executable section bytes are kept by default** ([`ExecutableBytes::Keep`]). Guest code is
//!   recompiled to Wasm and never executed from these bytes, but compilers place data in `.text`
//!   (switch jump tables, literal pools, constants) and recompiled code reads it through guest
//!   addresses; some code also hashes or inspects its own code. [`ExecutableBytes::Omit`] maps
//!   executable sections as zero-fill with their original protection (so every address stays
//!   valid) and drops relocation sites inside them. That shrinks the container by the size of
//!   `.text`, but is only correct once the lifter has proven no guest read targets code.
//!
//! # Base relocations
//!
//! [`parse_base_relocations`] decodes `IMAGE_DIRECTORY_ENTRY_BASERELOC`: blocks of
//! `page_rva: u32, block_size: u32` (block size includes the 8-byte header) followed by
//! `u16` entries `type << 12 | offset`. `IMAGE_REL_BASED_ABSOLUTE` (0) is padding,
//! `IMAGE_REL_BASED_HIGHLOW` (3) and `IMAGE_REL_BASED_DIR64` (10) become [`BaseRelocation`]s, and
//! every other type is reported as an [`ImageDiagnostic`] and skipped (`HIGHADJ` also skips its
//! parameter slot). A zero block header followed only by zero bytes ends the directory.
//! Truncated or malformed blocks and overlapping sites are [`ImageError`]s; nothing panics.
//!
//! The table is a sorted list of RVAs. Rebasing from the preferred `image_base` to `new_base`
//! adds `delta = new_base - image_base` (two's complement, so negative deltas work) to the
//! little-endian `u64` at each `DIR64` site and the low 32 bits of `delta` to the `u32` at each
//! `HIGHLOW` site. Sites need not be aligned and may straddle a page boundary.
//! [`apply_relocations`] is the reference implementation.
//!
//! Relocation sites inside executable sections also mark absolute addresses baked into guest
//! code; the recompiled code embeds those as constants. Mapping at the preferred base avoids
//! rebasing entirely and is what the runtime should try first (`VirtualMemoryArena::reserve` with
//! the image base as hint); rebasing data alone is only sound if the lifter rebases its constants
//! by the same delta.
//!
//! # Container format (`PGIM`, version 1)
//!
//! All integers are little-endian. Guest RVAs and sizes are `u32` (a PE32+ `SizeOfImage` is a
//! `u32`); container offsets and lengths are `u64`. The file is laid out in streaming order:
//! header, section table, relocation table, then section payloads.
//!
//! Header (64 bytes, offset 0):
//!
//! | Offset | Type     | Field                                                              |
//! | -----: | :------- | :----------------------------------------------------------------- |
//! | 0x00   | `[u8;4]` | magic `"PGIM"`                                                     |
//! | 0x04   | `u32`    | version = 1                                                        |
//! | 0x08   | `u32`    | header size = 64                                                   |
//! | 0x0C   | `u32`    | section entry size = 64                                            |
//! | 0x10   | `u32`    | flags: bit 0 headers mapped, bit 1 executable bytes omitted, bit 2 relocations stripped (`IMAGE_FILE_RELOCS_STRIPPED`: must map at `image_base`) |
//! | 0x14   | `u32`    | payload alignment = 4096 (guest page size)                         |
//! | 0x18   | `u64`    | preferred image base                                               |
//! | 0x20   | `u32`    | `SizeOfImage`                                                      |
//! | 0x24   | `u32`    | `SectionAlignment` (informational)                                 |
//! | 0x28   | `u32`    | entry point RVA (informational, 0 if none)                         |
//! | 0x2C   | `u32`    | section count `N`                                                  |
//! | 0x30   | `u32`    | `DIR64` relocation count `D`                                       |
//! | 0x34   | `u32`    | `HIGHLOW` relocation count `H`                                     |
//! | 0x38   | `u64`    | total container size in bytes                                      |
//!
//! Section table (`N` x 64 bytes at offset 64), sorted by RVA, non-overlapping:
//!
//! | Offset | Type      | Field                                                             |
//! | -----: | :-------- | :---------------------------------------------------------------- |
//! | 0x00   | `[u8;16]` | name, UTF-8, NUL-padded (truncated to 16 bytes)                   |
//! | 0x10   | `u32`     | RVA                                                               |
//! | 0x14   | `u32`     | in-memory size (> 0, `RVA + size <= SizeOfImage`)                 |
//! | 0x18   | `u32`     | protection: bit 0 read, bit 1 write, bit 2 execute                |
//! | 0x1C   | `u32`     | flags: bit 0 PE headers pseudo-section, bit 1 bytes omitted       |
//! | 0x20   | `u64`     | payload offset (multiple of 4096; 0 when payload length is 0)     |
//! | 0x28   | `u64`     | payload length = initialised bytes (`<=` in-memory size)          |
//! | 0x30   | `u64`     | FNV-1a 64 of the payload bytes                                    |
//! | 0x38   | `u32`     | raw PE section characteristics (informational)                    |
//! | 0x3C   | `u32`     | reserved = 0                                                      |
//!
//! Relocation table (offset `64 + 64 * N`): `D` x `u32` `DIR64` site RVAs, then `H` x `u32`
//! `HIGHLOW` site RVAs, each list strictly ascending. Sites never overlap each other and lie
//! entirely in mapped, non-omitted sections.
//!
//! Payloads follow, in section order, each starting at the next 4096-byte boundary after the
//! previous structure (gaps are zero). Zero-fill is never stored. The container ends right after
//! the last payload.
//!
//! To load, the runtime reserves `[base, base + SizeOfImage)`, commits for every section the pages
//! covering `[RVA, RVA + size)` (fresh pages are zero), copies each payload to `base + RVA`,
//! applies relocations if `base != image_base`, and finally sets page protections as computed by
//! [`GuestImage::page_protections`] (a page shared by several sections gets the union of their
//! protections). Pages outside every section stay reserved and inaccessible. [`read_image`]
//! performs every validation a loader needs and is the reference for the C++ reader.

use std::fmt::{self, Write as _};
use std::ops::BitOr;

use crate::pe::{IMAGE_SCN_MEM_EXECUTE, IMAGE_SCN_MEM_READ, IMAGE_SCN_MEM_WRITE, Pe64};

/// Container magic.
pub const IMAGE_MAGIC: [u8; 4] = *b"PGIM";
/// Container format version.
pub const IMAGE_VERSION: u32 = 1;
/// Container header size in bytes.
pub const IMAGE_HEADER_BYTES: usize = 64;
/// Section table entry size in bytes.
pub const IMAGE_SECTION_ENTRY_BYTES: usize = 64;
/// Guest page size and payload alignment (matches `playgta6::os::kPageSize`).
pub const IMAGE_PAGE_SIZE: u64 = 4096;
/// Maximum section name length stored in the container.
pub const SECTION_NAME_BYTES: usize = 16;
/// Name of the PE headers pseudo-section.
pub const HEADERS_SECTION_NAME: &str = "<headers>";

/// Header flag: the PE headers are mapped as the first section.
pub const FLAG_HEADERS_MAPPED: u32 = 1 << 0;
/// Header flag: executable sections are mapped as zero-fill ([`ExecutableBytes::Omit`]).
pub const FLAG_EXECUTABLE_BYTES_OMITTED: u32 = 1 << 1;
/// Header flag: the PE has `IMAGE_FILE_RELOCS_STRIPPED`; it can only run at its preferred base.
pub const FLAG_RELOCS_STRIPPED: u32 = 1 << 2;
const KNOWN_HEADER_FLAGS: u32 =
    FLAG_HEADERS_MAPPED | FLAG_EXECUTABLE_BYTES_OMITTED | FLAG_RELOCS_STRIPPED;

/// Section flag: PE headers pseudo-section.
pub const SECTION_FLAG_HEADERS: u32 = 1 << 0;
/// Section flag: initialised bytes were dropped ([`ExecutableBytes::Omit`]).
pub const SECTION_FLAG_BYTES_OMITTED: u32 = 1 << 1;
const KNOWN_SECTION_FLAGS: u32 = SECTION_FLAG_HEADERS | SECTION_FLAG_BYTES_OMITTED;

/// `IMAGE_DIRECTORY_ENTRY_BASERELOC`.
pub const DIR_BASERELOC: usize = 5;
/// `IMAGE_FILE_RELOCS_STRIPPED` COFF characteristic.
pub const IMAGE_FILE_RELOCS_STRIPPED: u16 = 0x0001;

const IMAGE_REL_BASED_ABSOLUTE: u8 = 0;
const IMAGE_REL_BASED_HIGHLOW: u8 = 3;
const IMAGE_REL_BASED_HIGHADJ: u8 = 4;
const IMAGE_REL_BASED_DIR64: u8 = 10;
const RELOC_BLOCK_HEADER_BYTES: usize = 8;

const FNV1A_64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A_64_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Errors from building, validating, serialising, or reading a guest image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageError {
    Truncated {
        context: &'static str,
        needed: u64,
        available: u64,
    },
    BadMagic([u8; 4]),
    UnsupportedVersion(u32),
    BadHeaderField {
        field: &'static str,
        value: u64,
    },
    Overflow {
        context: &'static str,
    },
    SectionOutOfImage {
        name: String,
        rva: u32,
        virtual_size: u32,
        size_of_image: u32,
    },
    OverlappingSections {
        first: String,
        second: String,
    },
    InvalidSection {
        index: usize,
        reason: &'static str,
    },
    PayloadOutOfRange {
        index: usize,
        offset: u64,
        length: u64,
        file_len: u64,
    },
    PayloadMisaligned {
        index: usize,
        offset: u64,
    },
    PayloadOverlap {
        index: usize,
    },
    ChecksumMismatch {
        index: usize,
        expected: u64,
        actual: u64,
    },
    RelocDirectoryUnmapped {
        rva: u32,
        size: u32,
    },
    RelocBlockTruncated {
        offset: usize,
        needed: usize,
        available: usize,
    },
    MalformedRelocBlock {
        offset: usize,
        page_rva: u32,
        block_size: u32,
        reason: &'static str,
    },
    RelocSiteOutOfRange {
        rva: u32,
        width: u32,
        limit: u64,
    },
    RelocSiteUnmapped {
        rva: u32,
    },
    RelocationsUnsorted {
        rva: u32,
    },
    OverlappingRelocations {
        first: u32,
        second: u32,
    },
}

impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated {
                context,
                needed,
                available,
            } => write!(
                f,
                "truncated {context}: needed {needed} bytes, only {available} available"
            ),
            Self::BadMagic(m) => write!(f, "invalid image magic {m:02X?} (expected \"PGIM\")"),
            Self::UnsupportedVersion(v) => {
                write!(
                    f,
                    "unsupported image version {v} (expected {IMAGE_VERSION})"
                )
            }
            Self::BadHeaderField { field, value } => {
                write!(f, "invalid header field {field} = 0x{value:x}")
            }
            Self::Overflow { context } => write!(f, "integer overflow while computing {context}"),
            Self::SectionOutOfImage {
                name,
                rva,
                virtual_size,
                size_of_image,
            } => write!(
                f,
                "section {name} [0x{rva:x}..+0x{virtual_size:x}] exceeds SizeOfImage 0x{size_of_image:x}"
            ),
            Self::OverlappingSections { first, second } => {
                write!(f, "sections {first} and {second} overlap")
            }
            Self::InvalidSection { index, reason } => write!(f, "section {index}: {reason}"),
            Self::PayloadOutOfRange {
                index,
                offset,
                length,
                file_len,
            } => write!(
                f,
                "section {index} payload [0x{offset:x}..+0x{length:x}] is outside the container (0x{file_len:x} bytes) or its tables"
            ),
            Self::PayloadMisaligned { index, offset } => write!(
                f,
                "section {index} payload offset 0x{offset:x} is not {IMAGE_PAGE_SIZE}-byte aligned"
            ),
            Self::PayloadOverlap { index } => {
                write!(f, "section {index} payload overlaps another payload")
            }
            Self::ChecksumMismatch {
                index,
                expected,
                actual,
            } => write!(
                f,
                "section {index} checksum mismatch: stored 0x{expected:016x}, computed 0x{actual:016x}"
            ),
            Self::RelocDirectoryUnmapped { rva, size } => write!(
                f,
                "base relocation directory [0x{rva:x}..+0x{size:x}] is not backed by file bytes"
            ),
            Self::RelocBlockTruncated {
                offset,
                needed,
                available,
            } => write!(
                f,
                "base relocation block at directory offset 0x{offset:x} needs {needed} bytes, only {available} available"
            ),
            Self::MalformedRelocBlock {
                offset,
                page_rva,
                block_size,
                reason,
            } => write!(
                f,
                "malformed base relocation block at directory offset 0x{offset:x} (page 0x{page_rva:x}, size {block_size}): {reason}"
            ),
            Self::RelocSiteOutOfRange { rva, width, limit } => write!(
                f,
                "relocation site 0x{rva:x} (+{width}) is outside 0x{limit:x}"
            ),
            Self::RelocSiteUnmapped { rva } => write!(
                f,
                "relocation site 0x{rva:x} is not inside a mapped section with stored bytes"
            ),
            Self::RelocationsUnsorted { rva } => {
                write!(f, "relocation table is not strictly ascending at 0x{rva:x}")
            }
            Self::OverlappingRelocations { first, second } => {
                write!(f, "relocation sites 0x{first:x} and 0x{second:x} overlap")
            }
        }
    }
}

impl std::error::Error for ImageError {}

/// Non-fatal findings while building an image (addresses and names only).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageDiagnostic {
    /// Relocation type other than `ABSOLUTE`/`HIGHLOW`/`DIR64`; the site is not rebased.
    UnsupportedRelocType { rva: u32, type_code: u8 },
    /// The same site and kind appeared more than once; it is kept once.
    DuplicateRelocation { rva: u32 },
    /// The file ends before the section's raw data does; the missing bytes become zero-fill.
    RawDataTruncated {
        section: String,
        expected: u32,
        available: u32,
    },
    /// Section with zero in-memory size; not mapped.
    EmptySection { section: String },
    /// Relocation sites dropped because the section's bytes were omitted.
    RelocationsInOmittedSection { section: String, count: usize },
}

impl fmt::Display for ImageDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedRelocType { rva, type_code } => write!(
                f,
                "unsupported base relocation type {type_code} at 0x{rva:x} (not rebased)"
            ),
            Self::DuplicateRelocation { rva } => {
                write!(f, "duplicate base relocation at 0x{rva:x}")
            }
            Self::RawDataTruncated {
                section,
                expected,
                available,
            } => write!(
                f,
                "section {section} raw data truncated by end of file: 0x{available:x} of 0x{expected:x} bytes"
            ),
            Self::EmptySection { section } => write!(f, "section {section} is empty (not mapped)"),
            Self::RelocationsInOmittedSection { section, count } => write!(
                f,
                "{count} relocation site(s) in section {section} dropped with its omitted bytes"
            ),
        }
    }
}

/// What to do with the initialised bytes of executable sections. See the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutableBytes {
    /// Store them, so guest reads of data embedded in code see the original bytes.
    #[default]
    Keep,
    /// Map executable sections as zero-fill and drop relocation sites inside them.
    Omit,
}

impl ExecutableBytes {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Keep => "keep",
            Self::Omit => "omit",
        }
    }
}

/// Options for [`build_guest_image`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageOptions {
    /// Map the PE headers read-only at RVA 0 (default `true`).
    pub map_headers: bool,
    pub executable_bytes: ExecutableBytes,
}

impl Default for ImageOptions {
    fn default() -> Self {
        Self {
            map_headers: true,
            executable_bytes: ExecutableBytes::Keep,
        }
    }
}

/// Guest page protection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Protection {
    pub read: bool,
    pub write: bool,
    pub execute: bool,
}

impl Protection {
    pub const READ_ONLY: Self = Self {
        read: true,
        write: false,
        execute: false,
    };

    /// Protection implied by PE section characteristics.
    pub fn from_characteristics(characteristics: u32) -> Self {
        Self {
            read: characteristics & IMAGE_SCN_MEM_READ != 0,
            write: characteristics & IMAGE_SCN_MEM_WRITE != 0,
            execute: characteristics & IMAGE_SCN_MEM_EXECUTE != 0,
        }
    }

    /// Container encoding: bit 0 read, bit 1 write, bit 2 execute.
    pub fn bits(self) -> u32 {
        u32::from(self.read) | u32::from(self.write) << 1 | u32::from(self.execute) << 2
    }

    pub fn from_bits(bits: u32) -> Option<Self> {
        (bits & !0b111 == 0).then_some(Self {
            read: bits & 1 != 0,
            write: bits & 2 != 0,
            execute: bits & 4 != 0,
        })
    }

    /// `rwx`-style string.
    pub fn as_str(self) -> &'static str {
        const NAMES: [&str; 8] = ["---", "r--", "-w-", "rw-", "--x", "r-x", "-wx", "rwx"];
        NAMES[self.bits() as usize]
    }

    /// Closest Win32 `PAGE_*` constant (as used by `playgta6::os::PageProtect`). Write implies
    /// read, as in Win32 where no write-only protection exists.
    pub fn win32_page_protect(self) -> u32 {
        match (self.read || self.write, self.write, self.execute) {
            (_, true, true) => 0x40,       // PAGE_EXECUTE_READWRITE
            (_, false, true) => 0x20,      // PAGE_EXECUTE_READ
            (_, true, false) => 0x04,      // PAGE_READWRITE
            (true, false, false) => 0x02,  // PAGE_READONLY
            (false, false, false) => 0x01, // PAGE_NOACCESS
        }
    }
}

impl BitOr for Protection {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self {
            read: self.read || rhs.read,
            write: self.write || rhs.write,
            execute: self.execute || rhs.execute,
        }
    }
}

/// Width of a relocated value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BaseRelocKind {
    /// `IMAGE_REL_BASED_HIGHLOW`: 32-bit value, adds the low 32 bits of the delta.
    HighLow,
    /// `IMAGE_REL_BASED_DIR64`: 64-bit value.
    Dir64,
}

impl BaseRelocKind {
    pub fn width(self) -> u32 {
        match self {
            Self::HighLow => 4,
            Self::Dir64 => 8,
        }
    }
}

/// One base relocation site. Ordering is `(rva, kind)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BaseRelocation {
    pub rva: u32,
    pub kind: BaseRelocKind,
}

impl BaseRelocation {
    fn end(self) -> u64 {
        u64::from(self.rva) + u64::from(self.kind.width())
    }
}

/// Result of [`parse_base_relocations`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RelocParse {
    /// Sorted by RVA, deduplicated, non-overlapping.
    pub relocations: Vec<BaseRelocation>,
    pub diagnostics: Vec<ImageDiagnostic>,
}

/// One region of guest memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageSection<'a> {
    pub name: String,
    pub rva: u32,
    /// In-memory size; bytes past `data.len()` are zero-fill.
    pub virtual_size: u32,
    pub protection: Protection,
    /// PE headers pseudo-section.
    pub is_headers: bool,
    /// Initialised bytes were dropped ([`ExecutableBytes::Omit`]).
    pub bytes_omitted: bool,
    /// Raw PE section characteristics (0 for the headers).
    pub characteristics: u32,
    /// Initialised bytes, placed at `rva`.
    pub data: &'a [u8],
}

impl ImageSection<'_> {
    /// One past the last RVA.
    pub fn end(&self) -> u64 {
        u64::from(self.rva) + u64::from(self.virtual_size)
    }

    fn flags(&self) -> u32 {
        (if self.is_headers {
            SECTION_FLAG_HEADERS
        } else {
            0
        }) | (if self.bytes_omitted {
            SECTION_FLAG_BYTES_OMITTED
        } else {
            0
        })
    }
}

/// A run of guest pages sharing one protection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageRun {
    pub rva: u64,
    pub size: u64,
    pub protection: Protection,
}

/// Guest memory image: what the runtime maps into `VirtualMemoryArena`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuestImage<'a> {
    /// Preferred load address.
    pub image_base: u64,
    pub size_of_image: u32,
    pub section_alignment: u32,
    pub entry_point_rva: u32,
    pub headers_mapped: bool,
    pub executable_bytes: ExecutableBytes,
    pub relocs_stripped: bool,
    /// Sorted by RVA, non-overlapping.
    pub sections: Vec<ImageSection<'a>>,
    /// Sorted by RVA, non-overlapping.
    pub relocations: Vec<BaseRelocation>,
}

/// Result of [`build_guest_image`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageBuild<'a> {
    pub image: GuestImage<'a>,
    pub diagnostics: Vec<ImageDiagnostic>,
}

impl<'a> GuestImage<'a> {
    /// Section containing `rva`, if any (binary search).
    pub fn section_at(&self, rva: u64) -> Option<&ImageSection<'a>> {
        let idx = self
            .sections
            .partition_point(|s| u64::from(s.rva) <= rva)
            .checked_sub(1)?;
        let section = &self.sections[idx];
        (rva < section.end()).then_some(section)
    }

    /// Relocation counts as `(dir64, highlow)`.
    pub fn relocation_counts(&self) -> (usize, usize) {
        let dir64 = self
            .relocations
            .iter()
            .filter(|r| r.kind == BaseRelocKind::Dir64)
            .count();
        (dir64, self.relocations.len() - dir64)
    }

    /// Total initialised bytes stored in the container.
    pub fn stored_bytes(&self) -> u64 {
        self.sections.iter().map(|s| s.data.len() as u64).sum()
    }

    /// Page-granular protections the runtime applies after copying payloads, merged into runs.
    /// Pages shared by several sections (section alignment below the page size) get the union of
    /// their protections. Pages not touched by any section are not listed (reserved, no access).
    pub fn page_protections(&self) -> Vec<PageRun> {
        let spans: Vec<(u64, u64, Protection)> = self
            .sections
            .iter()
            .map(|s| {
                let start = u64::from(s.rva) / IMAGE_PAGE_SIZE * IMAGE_PAGE_SIZE;
                let end = s.end().div_ceil(IMAGE_PAGE_SIZE) * IMAGE_PAGE_SIZE;
                (start, end, s.protection)
            })
            .collect();
        let mut cuts: Vec<u64> = spans.iter().flat_map(|&(s, e, _)| [s, e]).collect();
        cuts.sort_unstable();
        cuts.dedup();

        // Sections are sorted and disjoint, so span starts and ends are both non-decreasing and
        // the spans covering a cut form a window that only moves forward.
        let mut runs: Vec<PageRun> = Vec::new();
        let mut lo = 0;
        for pair in cuts.windows(2) {
            let (start, end) = (pair[0], pair[1]);
            while lo < spans.len() && spans[lo].1 <= start {
                lo += 1;
            }
            let mut protection: Option<Protection> = None;
            for span in spans[lo..].iter().take_while(|span| span.0 <= start) {
                protection = Some(protection.map_or(span.2, |p| p | span.2));
            }
            let Some(protection) = protection else {
                continue;
            };
            match runs.last_mut() {
                Some(last) if last.rva + last.size == start && last.protection == protection => {
                    last.size += end - start;
                }
                _ => runs.push(PageRun {
                    rva: start,
                    size: end - start,
                    protection,
                }),
            }
        }
        runs
    }

    /// Flat guest memory `[0, SizeOfImage)` at the preferred base of a validated image. Allocates
    /// the whole image, so this is a reference/test helper; the runtime commits only the pages
    /// sections cover.
    pub fn materialize(&self) -> Vec<u8> {
        let mut memory = vec![0u8; self.size_of_image as usize];
        for section in &self.sections {
            let start = section.rva as usize;
            memory[start..start + section.data.len()].copy_from_slice(section.data);
        }
        memory
    }

    /// [`materialize`](Self::materialize) and rebase to `new_base`.
    pub fn rebased(&self, new_base: u64) -> Result<Vec<u8>, ImageError> {
        let mut memory = self.materialize();
        apply_relocations(&mut memory, &self.relocations, self.image_base, new_base)?;
        Ok(memory)
    }

    /// Check every invariant of the model (the container reader relies on this).
    pub fn validate(&self) -> Result<(), ImageError> {
        self.image_base
            .checked_add(u64::from(self.size_of_image))
            .ok_or(ImageError::BadHeaderField {
                field: "image_base + size_of_image",
                value: self.image_base,
            })?;

        if self.headers_mapped != self.sections.first().is_some_and(|s| s.is_headers) {
            return Err(ImageError::InvalidSection {
                index: 0,
                reason: "headers flag does not match the first section",
            });
        }

        let omit = self.executable_bytes == ExecutableBytes::Omit;
        for (index, section) in self.sections.iter().enumerate() {
            let invalid = |reason| Err(ImageError::InvalidSection { index, reason });
            if section.name.len() > SECTION_NAME_BYTES || section.name.contains('\0') {
                return invalid("name longer than 16 bytes or contains NUL");
            }
            if section.virtual_size == 0 {
                return invalid("zero in-memory size");
            }
            if section.data.len() as u64 > u64::from(section.virtual_size) {
                return invalid("initialised bytes exceed in-memory size");
            }
            if section.is_headers && (index != 0 || section.rva != 0) {
                return invalid("headers pseudo-section must be first and at RVA 0");
            }
            if section.bytes_omitted
                && !(omit && section.protection.execute && section.data.is_empty())
            {
                return invalid("omitted bytes only allowed for empty executable sections");
            }
            if omit && section.protection.execute && !section.is_headers && !section.bytes_omitted {
                return invalid("executable section bytes must be omitted");
            }
            if section.end() > u64::from(self.size_of_image) {
                return Err(ImageError::SectionOutOfImage {
                    name: section.name.clone(),
                    rva: section.rva,
                    virtual_size: section.virtual_size,
                    size_of_image: self.size_of_image,
                });
            }
            if index > 0 {
                let prev = &self.sections[index - 1];
                if prev.end() > u64::from(section.rva) {
                    return Err(ImageError::OverlappingSections {
                        first: prev.name.clone(),
                        second: section.name.clone(),
                    });
                }
            }
        }

        for (i, reloc) in self.relocations.iter().enumerate() {
            if i > 0 {
                let prev = self.relocations[i - 1];
                if prev.rva >= reloc.rva {
                    return Err(ImageError::RelocationsUnsorted { rva: reloc.rva });
                }
                if prev.end() > u64::from(reloc.rva) {
                    return Err(ImageError::OverlappingRelocations {
                        first: prev.rva,
                        second: reloc.rva,
                    });
                }
            }
            self.check_site(*reloc)?;
        }
        Ok(())
    }

    /// A site must lie in mapped sections whose bytes are stored (contiguous if it straddles two).
    fn check_site(&self, reloc: BaseRelocation) -> Result<(), ImageError> {
        if reloc.end() > u64::from(self.size_of_image) {
            return Err(ImageError::RelocSiteOutOfRange {
                rva: reloc.rva,
                width: reloc.kind.width(),
                limit: u64::from(self.size_of_image),
            });
        }
        let first = self.section_at(u64::from(reloc.rva));
        let last = self.section_at(reloc.end() - 1);
        match (first, last) {
            (Some(first), Some(last))
                if !first.bytes_omitted
                    && !last.bytes_omitted
                    && (first.rva == last.rva || first.end() == u64::from(last.rva)) =>
            {
                Ok(())
            }
            _ => Err(ImageError::RelocSiteUnmapped { rva: reloc.rva }),
        }
    }
}

/// Build the guest image of `pe`. `file` must be the bytes `pe` was parsed from.
pub fn build_guest_image<'a>(
    pe: &Pe64,
    file: &'a [u8],
    options: ImageOptions,
) -> Result<ImageBuild<'a>, ImageError> {
    let mut diagnostics = Vec::new();
    let omit = options.executable_bytes == ExecutableBytes::Omit;

    let mut sections = Vec::with_capacity(pe.sections.len() + 1);
    for header in &pe.sections {
        let memory_size = if header.virtual_size == 0 {
            header.raw_size
        } else {
            header.virtual_size
        };
        if memory_size == 0 {
            diagnostics.push(ImageDiagnostic::EmptySection {
                section: header.name.clone(),
            });
            continue;
        }
        let protection = Protection::from_characteristics(header.characteristics);
        let bytes_omitted = omit && protection.execute;
        let data = if bytes_omitted {
            &[][..]
        } else {
            let expected = header.raw_size.min(memory_size);
            let raw = file_slice(file, u64::from(header.raw_offset), u64::from(expected));
            if raw.len() < expected as usize {
                diagnostics.push(ImageDiagnostic::RawDataTruncated {
                    section: header.name.clone(),
                    expected,
                    available: raw.len() as u32,
                });
            }
            trim_trailing_zeros(raw)
        };
        sections.push(ImageSection {
            name: truncate_name(&header.name),
            rva: header.virtual_address,
            virtual_size: memory_size,
            protection,
            is_headers: false,
            bytes_omitted,
            characteristics: header.characteristics,
            data,
        });
    }
    sections.sort_by_key(|s| s.rva);

    if options.map_headers {
        let first_rva = sections.first().map_or(u32::MAX, |s| s.rva);
        let size = pe
            .optional
            .size_of_headers
            .min(first_rva)
            .min(pe.optional.size_of_image);
        if size > 0 {
            sections.insert(
                0,
                ImageSection {
                    name: HEADERS_SECTION_NAME.to_string(),
                    rva: 0,
                    virtual_size: size,
                    protection: Protection::READ_ONLY,
                    is_headers: true,
                    bytes_omitted: false,
                    characteristics: 0,
                    data: trim_trailing_zeros(file_slice(file, 0, u64::from(size))),
                },
            );
        }
    }

    let mut image = GuestImage {
        image_base: pe.optional.image_base,
        size_of_image: pe.optional.size_of_image,
        section_alignment: pe.optional.section_alignment,
        entry_point_rva: pe.optional.entry_point_rva,
        headers_mapped: sections.first().is_some_and(|s| s.is_headers),
        executable_bytes: options.executable_bytes,
        relocs_stripped: pe.coff.characteristics & IMAGE_FILE_RELOCS_STRIPPED != 0,
        sections,
        relocations: Vec::new(),
    };

    if let Some(dir) = pe.optional.data_directory(DIR_BASERELOC) {
        let dir_bytes = reloc_directory_bytes(pe, file, dir.virtual_address, dir.size).ok_or(
            ImageError::RelocDirectoryUnmapped {
                rva: dir.virtual_address,
                size: dir.size,
            },
        )?;
        let parsed = parse_base_relocations(dir_bytes)?;
        diagnostics.extend(parsed.diagnostics);
        let mut relocations = parsed.relocations;
        if omit {
            let mut dropped: Vec<(u32, usize)> = Vec::new();
            relocations.retain(|reloc| {
                let omitted = [u64::from(reloc.rva), reloc.end() - 1]
                    .into_iter()
                    .filter_map(|rva| image.section_at(rva))
                    .find(|s| s.bytes_omitted);
                match omitted {
                    Some(section) => {
                        match dropped.last_mut() {
                            Some((rva, count)) if *rva == section.rva => *count += 1,
                            _ => dropped.push((section.rva, 1)),
                        }
                        false
                    }
                    None => true,
                }
            });
            for (rva, count) in dropped {
                let section = image
                    .section_at(u64::from(rva))
                    .map_or(String::new(), |s| s.name.clone());
                diagnostics.push(ImageDiagnostic::RelocationsInOmittedSection { section, count });
            }
        }
        image.relocations = relocations;
    }

    image.validate()?;
    Ok(ImageBuild { image, diagnostics })
}

/// Parse the raw bytes of a base relocation directory into a sorted, deduplicated table.
pub fn parse_base_relocations(dir: &[u8]) -> Result<RelocParse, ImageError> {
    let mut relocations = Vec::new();
    let mut diagnostics = Vec::new();
    let mut offset = 0usize;

    while offset < dir.len() {
        let rest = &dir[offset..];
        if rest.iter().all(|&b| b == 0) {
            break; // zero padding after the last block
        }
        if rest.len() < RELOC_BLOCK_HEADER_BYTES {
            return Err(ImageError::RelocBlockTruncated {
                offset,
                needed: RELOC_BLOCK_HEADER_BYTES,
                available: rest.len(),
            });
        }
        let page_rva = le_u32(&rest[0..4]);
        let block_size = le_u32(&rest[4..8]);
        let malformed = |reason| ImageError::MalformedRelocBlock {
            offset,
            page_rva,
            block_size,
            reason,
        };
        let size = block_size as usize;
        if size < RELOC_BLOCK_HEADER_BYTES {
            return Err(malformed("block size smaller than its 8-byte header"));
        }
        if !(size - RELOC_BLOCK_HEADER_BYTES).is_multiple_of(2) {
            return Err(malformed("entry area is not a whole number of u16 entries"));
        }
        if size > rest.len() {
            return Err(ImageError::RelocBlockTruncated {
                offset,
                needed: size,
                available: rest.len(),
            });
        }

        let mut entries = rest[RELOC_BLOCK_HEADER_BYTES..size]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&e| u16::from_le_bytes(e));
        while let Some(entry) = entries.next() {
            let type_code = (entry >> 12) as u8;
            let rva = page_rva
                .checked_add(u32::from(entry & 0x0FFF))
                .ok_or_else(|| malformed("site RVA overflows u32"))?;
            let kind = match type_code {
                IMAGE_REL_BASED_ABSOLUTE => continue,
                IMAGE_REL_BASED_HIGHLOW => BaseRelocKind::HighLow,
                IMAGE_REL_BASED_DIR64 => BaseRelocKind::Dir64,
                other => {
                    if other == IMAGE_REL_BASED_HIGHADJ {
                        entries.next(); // parameter slot
                    }
                    diagnostics.push(ImageDiagnostic::UnsupportedRelocType {
                        rva,
                        type_code: other,
                    });
                    continue;
                }
            };
            relocations.push(BaseRelocation { rva, kind });
        }
        offset += size;
    }

    relocations.sort_unstable();
    let mut unique: Vec<BaseRelocation> = Vec::with_capacity(relocations.len());
    for reloc in relocations {
        match unique.last() {
            Some(&prev) if prev == reloc => {
                if diagnostics.last()
                    != Some(&ImageDiagnostic::DuplicateRelocation { rva: reloc.rva })
                {
                    diagnostics.push(ImageDiagnostic::DuplicateRelocation { rva: reloc.rva });
                }
            }
            Some(&prev) if prev.end() > u64::from(reloc.rva) => {
                return Err(ImageError::OverlappingRelocations {
                    first: prev.rva,
                    second: reloc.rva,
                });
            }
            _ => unique.push(reloc),
        }
    }
    Ok(RelocParse {
        relocations: unique,
        diagnostics,
    })
}

/// Rebase flat guest memory (`memory[rva]` is the byte at `base + rva`) from `preferred_base` to
/// `new_base`. All sites are bounds-checked before anything is written.
pub fn apply_relocations(
    memory: &mut [u8],
    relocations: &[BaseRelocation],
    preferred_base: u64,
    new_base: u64,
) -> Result<(), ImageError> {
    if let Some(bad) = relocations.iter().find(|r| r.end() > memory.len() as u64) {
        return Err(ImageError::RelocSiteOutOfRange {
            rva: bad.rva,
            width: bad.kind.width(),
            limit: memory.len() as u64,
        });
    }
    let delta = new_base.wrapping_sub(preferred_base);
    for reloc in relocations {
        let start = reloc.rva as usize;
        match reloc.kind {
            BaseRelocKind::Dir64 => {
                let slot = &mut memory[start..start + 8];
                let value = u64::from_le_bytes(slot.try_into().expect("8-byte slot"));
                slot.copy_from_slice(&value.wrapping_add(delta).to_le_bytes());
            }
            BaseRelocKind::HighLow => {
                let slot = &mut memory[start..start + 4];
                let value = u32::from_le_bytes(slot.try_into().expect("4-byte slot"));
                slot.copy_from_slice(&value.wrapping_add(delta as u32).to_le_bytes());
            }
        }
    }
    Ok(())
}

/// Container layout: payload offset per section and total size.
struct Layout {
    data_offsets: Vec<u64>,
    file_size: u64,
}

fn layout(image: &GuestImage<'_>) -> Result<Layout, ImageError> {
    let overflow = |context| ImageError::Overflow { context };
    let tables = (IMAGE_HEADER_BYTES as u64)
        .checked_add(image.sections.len() as u64 * IMAGE_SECTION_ENTRY_BYTES as u64)
        .and_then(|v| v.checked_add(image.relocations.len() as u64 * 4))
        .ok_or(overflow("container table size"))?;
    let mut cursor = tables;
    let mut data_offsets = Vec::with_capacity(image.sections.len());
    for section in &image.sections {
        if section.data.is_empty() {
            data_offsets.push(0);
            continue;
        }
        let offset = cursor
            .checked_next_multiple_of(IMAGE_PAGE_SIZE)
            .ok_or(overflow("payload offset"))?;
        data_offsets.push(offset);
        cursor = offset
            .checked_add(section.data.len() as u64)
            .ok_or(overflow("payload end"))?;
    }
    Ok(Layout {
        data_offsets,
        file_size: cursor,
    })
}

/// Serialise a validated image into a `PGIM` container. Same image, same bytes.
pub fn write_image(image: &GuestImage<'_>) -> Result<Vec<u8>, ImageError> {
    image.validate()?;
    let overflow = |context| ImageError::Overflow { context };
    let section_count =
        u32::try_from(image.sections.len()).map_err(|_| overflow("section count"))?;
    let (dir64, highlow) = image.relocation_counts();
    let dir64 = u32::try_from(dir64).map_err(|_| overflow("DIR64 count"))?;
    let highlow = u32::try_from(highlow).map_err(|_| overflow("HIGHLOW count"))?;
    let layout = layout(image)?;
    let capacity = usize::try_from(layout.file_size).map_err(|_| overflow("container size"))?;

    let mut flags = 0;
    if image.headers_mapped {
        flags |= FLAG_HEADERS_MAPPED;
    }
    if image.executable_bytes == ExecutableBytes::Omit {
        flags |= FLAG_EXECUTABLE_BYTES_OMITTED;
    }
    if image.relocs_stripped {
        flags |= FLAG_RELOCS_STRIPPED;
    }

    let mut out = Vec::with_capacity(capacity);
    out.extend_from_slice(&IMAGE_MAGIC);
    out.extend_from_slice(&IMAGE_VERSION.to_le_bytes());
    out.extend_from_slice(&(IMAGE_HEADER_BYTES as u32).to_le_bytes());
    out.extend_from_slice(&(IMAGE_SECTION_ENTRY_BYTES as u32).to_le_bytes());
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&(IMAGE_PAGE_SIZE as u32).to_le_bytes());
    out.extend_from_slice(&image.image_base.to_le_bytes());
    out.extend_from_slice(&image.size_of_image.to_le_bytes());
    out.extend_from_slice(&image.section_alignment.to_le_bytes());
    out.extend_from_slice(&image.entry_point_rva.to_le_bytes());
    out.extend_from_slice(&section_count.to_le_bytes());
    out.extend_from_slice(&dir64.to_le_bytes());
    out.extend_from_slice(&highlow.to_le_bytes());
    out.extend_from_slice(&layout.file_size.to_le_bytes());
    debug_assert_eq!(out.len(), IMAGE_HEADER_BYTES);

    for (section, &offset) in image.sections.iter().zip(&layout.data_offsets) {
        let mut name = [0u8; SECTION_NAME_BYTES];
        name[..section.name.len()].copy_from_slice(section.name.as_bytes());
        out.extend_from_slice(&name);
        out.extend_from_slice(&section.rva.to_le_bytes());
        out.extend_from_slice(&section.virtual_size.to_le_bytes());
        out.extend_from_slice(&section.protection.bits().to_le_bytes());
        out.extend_from_slice(&section.flags().to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&(section.data.len() as u64).to_le_bytes());
        out.extend_from_slice(&fnv1a_64(section.data).to_le_bytes());
        out.extend_from_slice(&section.characteristics.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
    }

    for kind in [BaseRelocKind::Dir64, BaseRelocKind::HighLow] {
        for reloc in image.relocations.iter().filter(|r| r.kind == kind) {
            out.extend_from_slice(&reloc.rva.to_le_bytes());
        }
    }

    for (section, &offset) in image.sections.iter().zip(&layout.data_offsets) {
        if !section.data.is_empty() {
            out.resize(offset as usize, 0);
            out.extend_from_slice(section.data);
        }
    }
    debug_assert_eq!(out.len() as u64, layout.file_size);
    Ok(out)
}

/// Parse and fully validate a `PGIM` container. Payloads borrow from `bytes`.
pub fn read_image(bytes: &[u8]) -> Result<GuestImage<'_>, ImageError> {
    let file_len = bytes.len() as u64;
    if bytes.len() < IMAGE_HEADER_BYTES {
        return Err(ImageError::Truncated {
            context: "image header",
            needed: IMAGE_HEADER_BYTES as u64,
            available: file_len,
        });
    }
    let magic: [u8; 4] = bytes[0..4].try_into().expect("4-byte magic");
    if magic != IMAGE_MAGIC {
        return Err(ImageError::BadMagic(magic));
    }
    let version = le_u32(&bytes[4..8]);
    if version != IMAGE_VERSION {
        return Err(ImageError::UnsupportedVersion(version));
    }
    let expect_u32 = |at: usize, field, expected: u32| {
        let value = le_u32(&bytes[at..at + 4]);
        if value == expected {
            Ok(())
        } else {
            Err(ImageError::BadHeaderField {
                field,
                value: u64::from(value),
            })
        }
    };
    expect_u32(8, "header_bytes", IMAGE_HEADER_BYTES as u32)?;
    expect_u32(12, "section_entry_bytes", IMAGE_SECTION_ENTRY_BYTES as u32)?;
    expect_u32(20, "page_size", IMAGE_PAGE_SIZE as u32)?;
    let flags = le_u32(&bytes[16..20]);
    if flags & !KNOWN_HEADER_FLAGS != 0 {
        return Err(ImageError::BadHeaderField {
            field: "flags",
            value: u64::from(flags),
        });
    }
    let image_base = le_u64(&bytes[24..32]);
    let size_of_image = le_u32(&bytes[32..36]);
    let section_alignment = le_u32(&bytes[36..40]);
    let entry_point_rva = le_u32(&bytes[40..44]);
    let section_count = le_u32(&bytes[44..48]) as u64;
    let dir64_count = le_u32(&bytes[48..52]) as u64;
    let highlow_count = le_u32(&bytes[52..56]) as u64;
    let file_size = le_u64(&bytes[56..64]);
    if file_size != file_len {
        return Err(ImageError::BadHeaderField {
            field: "file_size",
            value: file_size,
        });
    }

    // All counts are u32, so these products cannot overflow u64.
    let table_start = IMAGE_HEADER_BYTES as u64 + section_count * IMAGE_SECTION_ENTRY_BYTES as u64;
    let tables_end = table_start + (dir64_count + highlow_count) * 4;
    if tables_end > file_len {
        return Err(ImageError::Truncated {
            context: "section and relocation tables",
            needed: tables_end,
            available: file_len,
        });
    }

    let mut sections = Vec::with_capacity(section_count as usize);
    let mut payloads: Vec<(u64, u64, usize)> = Vec::new();
    for index in 0..section_count as usize {
        let at = IMAGE_HEADER_BYTES + index * IMAGE_SECTION_ENTRY_BYTES;
        let entry = &bytes[at..at + IMAGE_SECTION_ENTRY_BYTES];
        let invalid = |reason| ImageError::InvalidSection { index, reason };

        let raw_name = &entry[0..SECTION_NAME_BYTES];
        let name_len = raw_name
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(SECTION_NAME_BYTES);
        if raw_name[name_len..].iter().any(|&b| b != 0) {
            return Err(invalid("name padding is not zero"));
        }
        let name = std::str::from_utf8(&raw_name[..name_len])
            .map_err(|_| invalid("name is not UTF-8"))?
            .to_string();
        let protection = Protection::from_bits(le_u32(&entry[24..28]))
            .ok_or_else(|| invalid("unknown protection bits"))?;
        let section_flags = le_u32(&entry[28..32]);
        if section_flags & !KNOWN_SECTION_FLAGS != 0 {
            return Err(invalid("unknown section flags"));
        }
        if le_u32(&entry[60..64]) != 0 {
            return Err(invalid("reserved field is not zero"));
        }
        let offset = le_u64(&entry[32..40]);
        let length = le_u64(&entry[40..48]);
        let checksum = le_u64(&entry[48..56]);

        let data = if length == 0 {
            if offset != 0 {
                return Err(invalid("empty payload with non-zero offset"));
            }
            &[][..]
        } else {
            if !offset.is_multiple_of(IMAGE_PAGE_SIZE) {
                return Err(ImageError::PayloadMisaligned { index, offset });
            }
            let end = offset.checked_add(length).filter(|&end| end <= file_len);
            let Some(end) = end.filter(|_| offset >= tables_end) else {
                return Err(ImageError::PayloadOutOfRange {
                    index,
                    offset,
                    length,
                    file_len,
                });
            };
            payloads.push((offset, end, index));
            &bytes[offset as usize..end as usize]
        };
        let actual = fnv1a_64(data);
        if actual != checksum {
            return Err(ImageError::ChecksumMismatch {
                index,
                expected: checksum,
                actual,
            });
        }

        sections.push(ImageSection {
            name,
            rva: le_u32(&entry[16..20]),
            virtual_size: le_u32(&entry[20..24]),
            protection,
            is_headers: section_flags & SECTION_FLAG_HEADERS != 0,
            bytes_omitted: section_flags & SECTION_FLAG_BYTES_OMITTED != 0,
            characteristics: le_u32(&entry[56..60]),
            data,
        });
    }

    payloads.sort_unstable();
    for pair in payloads.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(ImageError::PayloadOverlap { index: pair[1].2 });
        }
    }

    let read_rvas = |start: u64, count: u64, kind| -> Result<Vec<BaseRelocation>, ImageError> {
        let table = &bytes[start as usize..(start + count * 4) as usize];
        let mut out: Vec<BaseRelocation> = Vec::with_capacity(count as usize);
        for &chunk in table.as_chunks::<4>().0 {
            let rva = u32::from_le_bytes(chunk);
            if out.last().is_some_and(|prev| prev.rva >= rva) {
                return Err(ImageError::RelocationsUnsorted { rva });
            }
            out.push(BaseRelocation { rva, kind });
        }
        Ok(out)
    };
    let mut relocations = read_rvas(table_start, dir64_count, BaseRelocKind::Dir64)?;
    relocations.extend(read_rvas(
        table_start + dir64_count * 4,
        highlow_count,
        BaseRelocKind::HighLow,
    )?);
    relocations.sort_unstable();

    let image = GuestImage {
        image_base,
        size_of_image,
        section_alignment,
        entry_point_rva,
        headers_mapped: flags & FLAG_HEADERS_MAPPED != 0,
        executable_bytes: if flags & FLAG_EXECUTABLE_BYTES_OMITTED != 0 {
            ExecutableBytes::Omit
        } else {
            ExecutableBytes::Keep
        },
        relocs_stripped: flags & FLAG_RELOCS_STRIPPED != 0,
        sections,
        relocations,
    };
    image.validate()?;
    Ok(image)
}

/// Deterministic JSON manifest of a packed image: addresses, sizes, counts and checksums only.
pub fn render_manifest_json(
    image: &GuestImage<'_>,
    diagnostics: &[ImageDiagnostic],
) -> Result<String, ImageError> {
    let layout = layout(image)?;
    let (dir64, highlow) = image.relocation_counts();
    let mut out = String::new();
    out.push_str("{\n");
    let _ = writeln!(out, "  \"format\": \"PGIM\",");
    let _ = writeln!(out, "  \"version\": {IMAGE_VERSION},");
    let _ = writeln!(out, "  \"image_base\": \"0x{:x}\",", image.image_base);
    let _ = writeln!(out, "  \"size_of_image\": \"0x{:x}\",", image.size_of_image);
    let _ = writeln!(
        out,
        "  \"section_alignment\": \"0x{:x}\",",
        image.section_alignment
    );
    let _ = writeln!(
        out,
        "  \"entry_point_rva\": \"0x{:x}\",",
        image.entry_point_rva
    );
    let _ = writeln!(out, "  \"headers_mapped\": {},", image.headers_mapped);
    let _ = writeln!(
        out,
        "  \"executable_bytes\": \"{}\",",
        image.executable_bytes.as_str()
    );
    let _ = writeln!(out, "  \"relocs_stripped\": {},", image.relocs_stripped);
    let _ = writeln!(out, "  \"container_bytes\": {},", layout.file_size);
    let _ = writeln!(out, "  \"stored_bytes\": {},", image.stored_bytes());
    out.push_str("  \"sections\": [");
    for (i, (section, offset)) in image.sections.iter().zip(&layout.data_offsets).enumerate() {
        out.push_str(if i == 0 { "\n" } else { ",\n" });
        let _ = write!(
            out,
            "    {{\"name\": {}, \"rva\": \"0x{:x}\", \"virtual_size\": \"0x{:x}\", \
             \"protection\": \"{}\", \"data_offset\": \"0x{offset:x}\", \"data_length\": \"0x{:x}\", \
             \"zero_fill\": \"0x{:x}\", \"checksum\": \"0x{:016x}\"}}",
            json_string(&section.name),
            section.rva,
            section.virtual_size,
            section.protection.as_str(),
            section.data.len(),
            u64::from(section.virtual_size) - section.data.len() as u64,
            fnv1a_64(section.data),
        );
    }
    out.push_str(if image.sections.is_empty() {
        "],\n"
    } else {
        "\n  ],\n"
    });
    let _ = writeln!(
        out,
        "  \"relocations\": {{\"total\": {}, \"dir64\": {dir64}, \"highlow\": {highlow}}},",
        image.relocations.len()
    );
    out.push_str("  \"diagnostics\": [");
    for (i, diagnostic) in diagnostics.iter().enumerate() {
        out.push_str(if i == 0 { "\n    " } else { ",\n    " });
        out.push_str(&json_string(&diagnostic.to_string()));
    }
    out.push_str(if diagnostics.is_empty() {
        "]\n"
    } else {
        "\n  ]\n"
    });
    out.push_str("}\n");
    Ok(out)
}

/// Human-readable summary (names, addresses and counts only).
pub fn render_summary_text(image: &GuestImage<'_>, diagnostics: &[ImageDiagnostic]) -> String {
    let (dir64, highlow) = image.relocation_counts();
    let mut out = String::new();
    let _ = writeln!(
        out,
        "image base 0x{:x}, size of image 0x{:x}, {} section(s), {} relocation(s) ({dir64} dir64, {highlow} highlow), executable bytes {}, {} diagnostic(s)",
        image.image_base,
        image.size_of_image,
        image.sections.len(),
        image.relocations.len(),
        image.executable_bytes.as_str(),
        diagnostics.len()
    );
    for section in &image.sections {
        let _ = writeln!(
            out,
            "  {:<16} rva 0x{:08x} size 0x{:08x} stored 0x{:08x} {}",
            section.name,
            section.rva,
            section.virtual_size,
            section.data.len(),
            section.protection.as_str()
        );
    }
    for diagnostic in diagnostics {
        let _ = writeln!(out, "  warning: {diagnostic}");
    }
    out
}

/// 64-bit FNV-1a (same parameters as the shader pack hash).
pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV1A_64_OFFSET_BASIS, |hash, &b| {
        (hash ^ u64::from(b)).wrapping_mul(FNV1A_64_PRIME)
    })
}

/// Initialised bytes of the reloc directory (in a section's raw data or in the headers).
fn reloc_directory_bytes<'a>(pe: &Pe64, file: &'a [u8], rva: u32, size: u32) -> Option<&'a [u8]> {
    let end = u64::from(rva) + u64::from(size);
    let file_offset = pe
        .sections
        .iter()
        .find_map(|s| {
            let memory_size = if s.virtual_size == 0 {
                s.raw_size
            } else {
                s.virtual_size
            };
            let initialised = u64::from(s.raw_size.min(memory_size));
            let start = u64::from(s.virtual_address);
            (rva >= s.virtual_address && end <= start + initialised)
                .then(|| u64::from(s.raw_offset) + u64::from(rva - s.virtual_address))
        })
        .or_else(|| (end <= u64::from(pe.optional.size_of_headers)).then_some(u64::from(rva)))?;
    let slice = file_slice(file, file_offset, u64::from(size));
    (slice.len() == size as usize).then_some(slice)
}

/// `file[offset..offset + len]` clamped to the file.
fn file_slice(file: &[u8], offset: u64, len: u64) -> &[u8] {
    let start = offset.min(file.len() as u64) as usize;
    let end = offset.saturating_add(len).min(file.len() as u64) as usize;
    &file[start..end]
}

fn trim_trailing_zeros(data: &[u8]) -> &[u8] {
    let len = data.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
    &data[..len]
}

/// Truncate to [`SECTION_NAME_BYTES`] on a char boundary.
fn truncate_name(name: &str) -> String {
    let mut end = name.len().min(SECTION_NAME_BYTES);
    while !name.is_char_boundary(end) {
        end -= 1;
    }
    name[..end].to_string()
}

fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn le_u32(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

fn le_u64(b: &[u8]) -> u64 {
    u64::from_le_bytes(b[..8].try_into().expect("8 bytes"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pe::test_util::build_synthetic_pe64;

    const R: u32 = IMAGE_SCN_MEM_READ;
    const W: u32 = IMAGE_SCN_MEM_WRITE;
    const X: u32 = IMAGE_SCN_MEM_EXECUTE;
    const BASE: u64 = 0x1_4000_0000;

    struct Sec {
        name: &'static str,
        rva: u32,
        virtual_size: u32,
        raw: Vec<u8>,
        characteristics: u32,
    }

    impl Sec {
        fn new(name: &'static str, rva: u32, virtual_size: u32, raw: Vec<u8>, chars: u32) -> Self {
            Self {
                name,
                rva,
                virtual_size,
                raw,
                characteristics: chars,
            }
        }
    }

    struct Spec {
        image_base: u64,
        section_alignment: u32,
        file_alignment: u32,
        size_of_image: u32,
        size_of_headers: u32,
        coff_characteristics: u16,
        sections: Vec<Sec>,
        /// Base relocation directory `(rva, size)`.
        reloc_dir: Option<(u32, u32)>,
    }

    impl Spec {
        fn new(size_of_image: u32, sections: Vec<Sec>) -> Self {
            Self {
                image_base: BASE,
                section_alignment: 0x1000,
                file_alignment: 0x200,
                size_of_image,
                size_of_headers: 0x400,
                coff_characteristics: 0x0022,
                sections,
                reloc_dir: None,
            }
        }
    }

    fn put(buf: &mut [u8], off: usize, bytes: &[u8]) {
        buf[off..off + bytes.len()].copy_from_slice(bytes);
    }

    /// Minimal PE32+ writer: headers in `[0, size_of_headers)`, raw data packed after them.
    fn build_pe(spec: &Spec) -> Vec<u8> {
        let e_lfanew = 0x80usize;
        let opt = e_lfanew + 24;
        let table = opt + 240;
        assert!(table + spec.sections.len() * 40 <= spec.size_of_headers as usize);

        let align = |v: usize| v.next_multiple_of(spec.file_alignment as usize);
        let mut file_len = align(spec.size_of_headers as usize);
        let mut raw_offsets = Vec::new();
        for sec in &spec.sections {
            raw_offsets.push(if sec.raw.is_empty() { 0 } else { file_len });
            file_len = align(file_len + sec.raw.len());
        }
        let mut buf = vec![0u8; file_len];

        put(&mut buf, 0, b"MZ");
        put(&mut buf, 0x3C, &(e_lfanew as u32).to_le_bytes());
        put(&mut buf, e_lfanew, b"PE\0\0");
        put(&mut buf, e_lfanew + 4, &0x8664u16.to_le_bytes());
        put(
            &mut buf,
            e_lfanew + 6,
            &(spec.sections.len() as u16).to_le_bytes(),
        );
        put(&mut buf, e_lfanew + 20, &240u16.to_le_bytes());
        put(
            &mut buf,
            e_lfanew + 22,
            &spec.coff_characteristics.to_le_bytes(),
        );
        put(&mut buf, opt, &0x020Bu16.to_le_bytes());
        put(&mut buf, opt + 16, &0x1010u32.to_le_bytes()); // entry point
        put(&mut buf, opt + 24, &spec.image_base.to_le_bytes());
        put(&mut buf, opt + 32, &spec.section_alignment.to_le_bytes());
        put(&mut buf, opt + 36, &spec.file_alignment.to_le_bytes());
        put(&mut buf, opt + 56, &spec.size_of_image.to_le_bytes());
        put(&mut buf, opt + 60, &spec.size_of_headers.to_le_bytes());
        put(&mut buf, opt + 108, &16u32.to_le_bytes());
        if let Some((rva, size)) = spec.reloc_dir {
            put(&mut buf, opt + 112 + DIR_BASERELOC * 8, &rva.to_le_bytes());
            put(&mut buf, opt + 116 + DIR_BASERELOC * 8, &size.to_le_bytes());
        }

        for (i, (sec, &raw_off)) in spec.sections.iter().zip(&raw_offsets).enumerate() {
            let h = table + i * 40;
            put(&mut buf, h, sec.name.as_bytes());
            put(&mut buf, h + 8, &sec.virtual_size.to_le_bytes());
            put(&mut buf, h + 12, &sec.rva.to_le_bytes());
            put(&mut buf, h + 16, &(sec.raw.len() as u32).to_le_bytes());
            put(&mut buf, h + 20, &(raw_off as u32).to_le_bytes());
            put(&mut buf, h + 36, &sec.characteristics.to_le_bytes());
            put(&mut buf, raw_off, &sec.raw);
        }
        buf
    }

    /// One relocation block; `(type, page offset)` entries, padded with `ABSOLUTE` to 4 bytes.
    fn reloc_block(page_rva: u32, entries: &[(u8, u16)]) -> Vec<u8> {
        let mut words: Vec<u16> = entries
            .iter()
            .map(|&(ty, off)| u16::from(ty) << 12 | off)
            .collect();
        if words.len() % 2 == 1 {
            words.push(0);
        }
        let mut out = page_rva.to_le_bytes().to_vec();
        out.extend_from_slice(&(8 + 2 * words.len() as u32).to_le_bytes());
        for w in words {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    fn build(file: &[u8], options: ImageOptions) -> Result<ImageBuild<'_>, ImageError> {
        let pe = Pe64::parse(file).expect("synthetic PE parses");
        build_guest_image(&pe, file, options)
    }

    fn reloc(rva: u32, kind: BaseRelocKind) -> BaseRelocation {
        BaseRelocation { rva, kind }
    }

    /// `.data` with pointers (one unaligned across a page boundary), `.text` with one absolute
    /// pointer, and `.reloc` describing them.
    fn rebase_fixture() -> Vec<u8> {
        let mut data = vec![0u8; 0x2000];
        put(&mut data, 0x008, &(BASE + 0x1800).to_le_bytes()); // DIR64 -> .data
        put(&mut data, 0xFFC, &(BASE + 0x3100).to_le_bytes()); // DIR64 straddling 0x2000
        put(&mut data, 0x100, &((BASE as u32) + 0x1234).to_le_bytes()); // HIGHLOW
        put(&mut data, 0x200, b"not a pointer");
        let mut text = vec![0xCCu8; 0x40];
        put(&mut text, 0x10, &(BASE + 0x1008).to_le_bytes()); // DIR64 in code
        let mut relocs = reloc_block(
            0x1000,
            &[
                (IMAGE_REL_BASED_DIR64, 0x008),
                (IMAGE_REL_BASED_HIGHLOW, 0x100),
                (IMAGE_REL_BASED_DIR64, 0xFFC),
            ],
        );
        relocs.extend(reloc_block(0x3000, &[(IMAGE_REL_BASED_DIR64, 0x010)]));
        let reloc_len = relocs.len() as u32;
        let mut spec = Spec::new(
            0x5000,
            vec![
                Sec::new(".data", 0x1000, 0x2000, data, R | W),
                Sec::new(".text", 0x3000, 0x40, text, R | X),
                Sec::new(".reloc", 0x4000, reloc_len, relocs, R),
            ],
        );
        spec.reloc_dir = Some((0x4000, reloc_len));
        build_pe(&spec)
    }

    #[test]
    fn synthetic_pe64_bytes_land_at_their_rvas() {
        let file = build_synthetic_pe64(false);
        let pe = Pe64::parse(&file).expect("parses");
        let built = build_guest_image(&pe, &file, ImageOptions::default()).expect("builds");
        let image = &built.image;
        assert!(built.diagnostics.is_empty());
        assert_eq!(image.image_base, BASE);
        assert_eq!(image.size_of_image, 0x3000);
        assert_eq!(image.entry_point_rva, 0x1010);
        assert!(image.headers_mapped && !image.relocs_stripped);
        assert!(image.relocations.is_empty());

        let names: Vec<_> = image.sections.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, [HEADERS_SECTION_NAME, ".text", ".rdata"]);
        let s = &image.sections;
        assert_eq!((s[0].rva, s[0].virtual_size), (0, 0x200));
        assert_eq!(s[0].protection, Protection::READ_ONLY);
        assert_eq!(&s[0].data[..2], b"MZ");
        assert_eq!((s[1].rva, s[1].virtual_size), (0x1000, 0x120));
        assert_eq!(s[1].protection.as_str(), "r-x");
        assert_eq!(s[1].data, trim_trailing_zeros(&file[0x200..0x320]));
        assert_eq!((s[2].rva, s[2].virtual_size), (0x2000, 0x700));
        assert_eq!(s[2].protection.as_str(), "r--");

        // Every byte the PE maps lands at its RVA; everything else is zero.
        let memory = image.materialize();
        assert_eq!(memory.len(), 0x3000);
        for rva in 0..0x3000u32 {
            let expected = pe.rva_to_file_offset(rva).map_or(0, |off| file[off]);
            assert_eq!(memory[rva as usize], expected, "byte at RVA 0x{rva:x}");
        }

        // Without headers the header page reads as zero and is not mapped.
        let no_headers = ImageOptions {
            map_headers: false,
            ..ImageOptions::default()
        };
        let image = build_guest_image(&pe, &file, no_headers)
            .expect("builds")
            .image;
        assert!(!image.headers_mapped);
        assert_eq!(image.sections.len(), 2);
        assert!(image.section_at(0).is_none());
        assert!(image.materialize()[..0x200].iter().all(|&b| b == 0));
    }

    #[test]
    fn zero_fill_bss_and_sub_page_alignment() {
        let mut data = vec![0x11u8; 0x80];
        data.extend_from_slice(&[0u8; 0x40]); // trailing zeros are zero-fill, not stored
        let mut spec = Spec::new(
            0x2800,
            vec![
                Sec::new(".text", 0x200, 0x180, vec![0x90; 0x200], R | X),
                Sec::new(".data", 0x400, 0x300, data, R | W),
                Sec::new(".bss", 0x800, 0x2000, Vec::new(), R | W | 0x80),
            ],
        );
        spec.section_alignment = 0x200;
        spec.size_of_headers = 0x200;
        let file = build_pe(&spec);
        let built = build(&file, ImageOptions::default()).expect("builds");
        let image = &built.image;
        assert!(built.diagnostics.is_empty());

        let s = &image.sections;
        assert_eq!(s.len(), 4);
        assert_eq!(s[1].data.len(), 0x180, "raw data clamped to VirtualSize");
        assert_eq!(s[2].data.len(), 0x80, "trailing zeros trimmed");
        assert_eq!((s[3].virtual_size, s[3].data.len()), (0x2000, 0));

        let memory = image.materialize();
        assert!(memory[0x200..0x380].iter().all(|&b| b == 0x90));
        assert!(memory[0x380..0x400].iter().all(|&b| b == 0));
        assert!(memory[0x400..0x480].iter().all(|&b| b == 0x11));
        assert!(memory[0x480..].iter().all(|&b| b == 0));

        // Headers, .text, .data and .bss share the first page: its protection is the union.
        let rw = Protection {
            read: true,
            write: true,
            execute: false,
        };
        assert_eq!(
            image.page_protections(),
            [
                PageRun {
                    rva: 0,
                    size: 0x1000,
                    protection: rw | Protection::from_characteristics(X),
                },
                PageRun {
                    rva: 0x1000,
                    size: 0x2000,
                    protection: rw,
                },
            ]
        );
    }

    #[test]
    fn page_protections_merge_runs_and_skip_gaps() {
        let mut spec = Spec::new(
            0x9000,
            vec![
                Sec::new(".text", 0x1000, 0x1800, vec![1], R | X),
                Sec::new(".rdata", 0x3000, 0x1000, vec![2], R),
                Sec::new(".data", 0x4000, 0x10, vec![3], R | W),
                Sec::new(".data2", 0x5000, 0x10, vec![4], R | W),
                Sec::new(".gap", 0x8000, 0x10, vec![5], 0),
            ],
        );
        spec.size_of_headers = 0x400;
        let file = build_pe(&spec);
        let image = build(&file, ImageOptions::default()).expect("builds").image;
        let runs: Vec<_> = image
            .page_protections()
            .iter()
            .map(|r| (r.rva, r.size, r.protection.as_str()))
            .collect();
        assert_eq!(
            runs,
            [
                (0, 0x1000, "r--"),
                (0x1000, 0x2000, "r-x"),
                (0x3000, 0x1000, "r--"),
                (0x4000, 0x2000, "rw-"),
                (0x8000, 0x1000, "---"),
            ]
        );
    }

    #[test]
    fn protection_encodings() {
        for bits in 0..8 {
            let p = Protection::from_bits(bits).expect("valid bits");
            assert_eq!(p.bits(), bits);
        }
        assert_eq!(Protection::from_bits(8), None);
        let p = |c| Protection::from_characteristics(c).win32_page_protect();
        assert_eq!(p(0), 0x01);
        assert_eq!(p(R), 0x02);
        assert_eq!(p(R | W), 0x04);
        assert_eq!(p(W), 0x04);
        assert_eq!(p(R | X), 0x20);
        assert_eq!(p(X), 0x20);
        assert_eq!(p(R | W | X), 0x40);
        assert_eq!(Protection::from_characteristics(R | W | X).as_str(), "rwx");
        assert_eq!(truncate_name("abcdefghijklmnopqrstu"), "abcdefghijklmnop");
        assert_eq!(truncate_name("abcdefghijklmno\u{e9}"), "abcdefghijklmno");
    }

    #[test]
    fn parses_reloc_blocks_padding_unknown_types_and_duplicates() {
        let mut dir = reloc_block(
            0x3000,
            &[
                (IMAGE_REL_BASED_DIR64, 0x010),
                (IMAGE_REL_BASED_ABSOLUTE, 0x000),
                (5, 0x030),
                (IMAGE_REL_BASED_HIGHADJ, 0x040),
                (1, 0x234), // HIGHADJ parameter slot, not an entry
                (IMAGE_REL_BASED_HIGHLOW, 0x050),
            ],
        );
        dir.extend(reloc_block(
            0x1000,
            &[
                (IMAGE_REL_BASED_DIR64, 0xFF8),
                (IMAGE_REL_BASED_DIR64, 0x000),
            ],
        ));
        dir.extend(reloc_block(0x3000, &[(IMAGE_REL_BASED_DIR64, 0x010)]));
        dir.extend([0u8; 12]); // zero padding after the last block

        let parsed = parse_base_relocations(&dir).expect("parses");
        assert_eq!(
            parsed.relocations,
            [
                reloc(0x1000, BaseRelocKind::Dir64),
                reloc(0x1FF8, BaseRelocKind::Dir64),
                reloc(0x3010, BaseRelocKind::Dir64),
                reloc(0x3050, BaseRelocKind::HighLow),
            ]
        );
        assert_eq!(
            parsed.diagnostics,
            [
                ImageDiagnostic::UnsupportedRelocType {
                    rva: 0x3030,
                    type_code: 5
                },
                ImageDiagnostic::UnsupportedRelocType {
                    rva: 0x3040,
                    type_code: IMAGE_REL_BASED_HIGHADJ
                },
                ImageDiagnostic::DuplicateRelocation { rva: 0x3010 },
            ]
        );
        assert_eq!(parse_base_relocations(&[]), Ok(RelocParse::default()));
    }

    #[test]
    fn rejects_malformed_reloc_blocks() {
        let block = reloc_block(0x1000, &[(IMAGE_REL_BASED_DIR64, 0x10)]);

        let mut tiny = block.clone();
        tiny[4..8].copy_from_slice(&4u32.to_le_bytes());
        assert!(matches!(
            parse_base_relocations(&tiny),
            Err(ImageError::MalformedRelocBlock { offset: 0, .. })
        ));

        let mut odd = block.clone();
        odd[4..8].copy_from_slice(&11u32.to_le_bytes());
        odd.extend([0u8; 4]);
        assert!(matches!(
            parse_base_relocations(&odd),
            Err(ImageError::MalformedRelocBlock { .. })
        ));

        assert_eq!(
            parse_base_relocations(&block[..block.len() - 2]),
            Err(ImageError::RelocBlockTruncated {
                offset: 0,
                needed: 12,
                available: 10
            })
        );

        let mut trailing = block.clone();
        trailing.extend([0, 0, 0x10, 0, 0]);
        assert_eq!(
            parse_base_relocations(&trailing),
            Err(ImageError::RelocBlockTruncated {
                offset: 12,
                needed: 8,
                available: 5
            })
        );

        // A zero header followed by real data is not padding.
        let mut hidden = vec![0u8; 8];
        hidden.extend(&block);
        assert!(matches!(
            parse_base_relocations(&hidden),
            Err(ImageError::MalformedRelocBlock { offset: 0, .. })
        ));

        let overflow = reloc_block(0xFFFF_FF00, &[(IMAGE_REL_BASED_DIR64, 0x100)]);
        assert!(matches!(
            parse_base_relocations(&overflow),
            Err(ImageError::MalformedRelocBlock { .. })
        ));

        let overlap = reloc_block(
            0x1000,
            &[
                (IMAGE_REL_BASED_DIR64, 0x10),
                (IMAGE_REL_BASED_HIGHLOW, 0x14),
            ],
        );
        assert_eq!(
            parse_base_relocations(&overlap),
            Err(ImageError::OverlappingRelocations {
                first: 0x1010,
                second: 0x1014
            })
        );

        // Arbitrary bytes never panic.
        let mut state = 0x1234_5678_9abc_def0u64;
        for len in 0..512 {
            let bytes: Vec<u8> = (0..len)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    (state >> 56) as u8
                })
                .collect();
            let _ = parse_base_relocations(&bytes);
        }
    }

    #[test]
    fn build_rejects_bad_sites_directories_and_sections() {
        let sections = |relocs: Vec<u8>| {
            vec![
                Sec::new(".data", 0x1000, 0x100, vec![7; 0x100], R | W),
                Sec::new(".reloc", 0x3000, relocs.len() as u32, relocs, R),
            ]
        };
        let with_relocs = |relocs: Vec<u8>| {
            let len = relocs.len() as u32;
            let mut spec = Spec::new(0x4000, sections(relocs));
            spec.reloc_dir = Some((0x3000, len));
            build_pe(&spec)
        };

        // Site in the gap between .data and .reloc.
        let file = with_relocs(reloc_block(0x2000, &[(IMAGE_REL_BASED_DIR64, 0)]));
        assert_eq!(
            build(&file, ImageOptions::default()).map(|_| ()),
            Err(ImageError::RelocSiteUnmapped { rva: 0x2000 })
        );
        // Site crossing the end of .data into the gap.
        let file = with_relocs(reloc_block(0x1000, &[(IMAGE_REL_BASED_DIR64, 0xFC)]));
        assert_eq!(
            build(&file, ImageOptions::default()).map(|_| ()),
            Err(ImageError::RelocSiteUnmapped { rva: 0x10FC })
        );
        // Site past SizeOfImage.
        let file = with_relocs(reloc_block(0x3000, &[(IMAGE_REL_BASED_DIR64, 0xFFC)]));
        assert!(matches!(
            build(&file, ImageOptions::default()),
            Err(ImageError::RelocSiteOutOfRange { rva: 0x3FFC, .. })
        ));
        // Site in the headers is fine only while they are mapped.
        let file = with_relocs(reloc_block(0, &[(IMAGE_REL_BASED_HIGHLOW, 0x40)]));
        assert!(build(&file, ImageOptions::default()).is_ok());
        let no_headers = ImageOptions {
            map_headers: false,
            ..ImageOptions::default()
        };
        assert_eq!(
            build(&file, no_headers).map(|_| ()),
            Err(ImageError::RelocSiteUnmapped { rva: 0x40 })
        );

        // Directory larger than the initialised bytes of its section.
        let mut spec = Spec::new(0x4000, sections(reloc_block(0x1000, &[(10, 0)])));
        spec.reloc_dir = Some((0x3000, 0x400));
        let file = build_pe(&spec);
        assert_eq!(
            build(&file, ImageOptions::default()).map(|_| ()),
            Err(ImageError::RelocDirectoryUnmapped {
                rva: 0x3000,
                size: 0x400
            })
        );

        let overlapping = Spec::new(
            0x4000,
            vec![
                Sec::new(".a", 0x1000, 0x1001, vec![1], R),
                Sec::new(".b", 0x2000, 0x10, vec![1], R),
            ],
        );
        assert_eq!(
            build(&build_pe(&overlapping), ImageOptions::default()).map(|_| ()),
            Err(ImageError::OverlappingSections {
                first: ".a".to_string(),
                second: ".b".to_string()
            })
        );

        let too_big = Spec::new(0x2000, vec![Sec::new(".a", 0x1000, 0x1001, vec![1], R)]);
        assert!(matches!(
            build(&build_pe(&too_big), ImageOptions::default()),
            Err(ImageError::SectionOutOfImage { .. })
        ));
    }

    #[test]
    fn empty_and_truncated_sections_are_diagnosed() {
        let spec = Spec::new(
            0x4000,
            vec![
                Sec::new(".empty", 0x1000, 0, Vec::new(), R),
                Sec::new(".data", 0x2000, 0x400, vec![9; 0x400], R | W),
            ],
        );
        let mut file = build_pe(&spec);
        let pe = Pe64::parse(&file).expect("parses");
        file.truncate(0x500); // .data raw starts at 0x400
        let built = build_guest_image(&pe, &file, ImageOptions::default()).expect("builds");
        assert_eq!(
            built.diagnostics,
            [
                ImageDiagnostic::EmptySection {
                    section: ".empty".to_string()
                },
                ImageDiagnostic::RawDataTruncated {
                    section: ".data".to_string(),
                    expected: 0x400,
                    available: 0x100
                },
            ]
        );
        let data = built.image.section_at(0x2000).expect(".data mapped");
        assert_eq!((data.virtual_size, data.data.len()), (0x400, 0x100));
    }

    #[test]
    fn apply_relocations_rebases_up_and_down() {
        let file = rebase_fixture();
        let built = build(&file, ImageOptions::default()).expect("builds");
        assert!(built.diagnostics.is_empty());
        let image = built.image;
        assert_eq!(
            image.relocations,
            [
                reloc(0x1008, BaseRelocKind::Dir64),
                reloc(0x1100, BaseRelocKind::HighLow),
                reloc(0x1FFC, BaseRelocKind::Dir64),
                reloc(0x3010, BaseRelocKind::Dir64),
            ]
        );
        let original = image.materialize();
        assert_eq!(image.rebased(BASE), Ok(original.clone()), "zero delta");

        let u64_at = |m: &[u8], rva: usize| u64::from_le_bytes(m[rva..rva + 8].try_into().unwrap());
        let u32_at = |m: &[u8], rva: usize| u32::from_le_bytes(m[rva..rva + 4].try_into().unwrap());
        for new_base in [BASE + 0x1_0000_0000, 0x1_0010_0000, 0x7FF0_0000_0000] {
            let rebased = image.rebased(new_base).expect("rebases");
            assert_eq!(u64_at(&rebased, 0x1008), new_base + 0x1800);
            assert_eq!(
                u64_at(&rebased, 0x1FFC),
                new_base + 0x3100,
                "straddles a page"
            );
            assert_eq!(u64_at(&rebased, 0x3010), new_base + 0x1008);
            assert_eq!(
                u32_at(&rebased, 0x1100),
                (new_base as u32).wrapping_add(0x1234)
            );
            for (rva, (&a, &b)) in original.iter().zip(&rebased).enumerate() {
                let is_site = image
                    .relocations
                    .iter()
                    .any(|r| (r.rva as usize..r.end() as usize).contains(&rva));
                if !is_site {
                    assert_eq!(a, b, "non-site byte 0x{rva:x} changed");
                }
            }
            // Rebasing back restores the original image.
            let mut back = rebased;
            apply_relocations(&mut back, &image.relocations, new_base, BASE).expect("rebases");
            assert_eq!(back, original);
        }

        // The table read back from the container rebases identically.
        let bytes = write_image(&image).expect("writes");
        let read = read_image(&bytes).expect("reads");
        assert_eq!(read.rebased(0x1_0010_0000), image.rebased(0x1_0010_0000));

        // Out-of-range sites are rejected before anything is written.
        let mut memory = original.clone();
        let sites = [
            reloc(0x10, BaseRelocKind::Dir64),
            reloc(0x4FFC, BaseRelocKind::Dir64),
        ];
        assert_eq!(
            apply_relocations(&mut memory, &sites, BASE, 0),
            Err(ImageError::RelocSiteOutOfRange {
                rva: 0x4FFC,
                width: 8,
                limit: 0x5000
            })
        );
        assert_eq!(memory, original);
    }

    #[test]
    fn omitting_executable_bytes_drops_code_and_its_relocations() {
        let file = rebase_fixture();
        let options = ImageOptions {
            map_headers: true,
            executable_bytes: ExecutableBytes::Omit,
        };
        let built = build(&file, options).expect("builds");
        assert_eq!(
            built.diagnostics,
            [ImageDiagnostic::RelocationsInOmittedSection {
                section: ".text".to_string(),
                count: 1
            }]
        );
        let image = built.image;
        let text = image.section_at(0x3000).expect(".text mapped");
        assert!(text.bytes_omitted && text.data.is_empty());
        assert_eq!((text.virtual_size, text.protection.as_str()), (0x40, "r-x"));
        assert_eq!(image.relocations.len(), 3);
        assert!(image.materialize()[0x3000..0x3040].iter().all(|&b| b == 0));

        let bytes = write_image(&image).expect("writes");
        assert_eq!(read_image(&bytes), Ok(image.clone()));
        assert_ne!(
            bytes,
            write_image(&build(&file, ImageOptions::default()).unwrap().image).unwrap()
        );

        let mut bad = image.clone();
        bad.relocations.push(reloc(0x3010, BaseRelocKind::Dir64));
        assert_eq!(
            bad.validate(),
            Err(ImageError::RelocSiteUnmapped { rva: 0x3010 })
        );
    }

    #[test]
    fn container_round_trips_deterministically() {
        let file = rebase_fixture();
        let image = build(&file, ImageOptions::default()).expect("builds").image;
        let bytes = write_image(&image).expect("writes");
        assert_eq!(
            write_image(&image).expect("writes"),
            bytes,
            "same input, same bytes"
        );
        assert_eq!(&bytes[0..4], b"PGIM");
        assert_eq!(le_u64(&bytes[56..64]), bytes.len() as u64);

        let read = read_image(&bytes).expect("reads back");
        assert_eq!(read, image);
        assert_eq!(read.materialize(), image.materialize());

        // Payloads are page aligned, sorted by RVA, and never store zero-fill.
        let mut last_end = 0;
        for index in 0..image.sections.len() {
            let entry = &bytes[64 + index * 64..128 + index * 64];
            let (offset, length) = (le_u64(&entry[32..40]), le_u64(&entry[40..48]));
            assert_eq!(length, image.sections[index].data.len() as u64);
            if length > 0 {
                assert!(offset.is_multiple_of(IMAGE_PAGE_SIZE) && offset >= last_end);
                last_end = offset + length;
            }
        }
        assert_eq!(last_end, bytes.len() as u64);
        let stored: u64 = image.stored_bytes();
        assert!(stored < 0x2100, "zero-fill not stored ({stored} bytes)");

        // Section header order in the PE does not matter (headers left unmapped because their
        // bytes differ between the two files).
        let a = || Sec::new(".a", 0x1000, 0x10, vec![1; 0x10], R);
        let b = || Sec::new(".b", 0x2000, 0x10, vec![2; 0x10], R | W);
        let pack = |sections| {
            let file = build_pe(&Spec::new(0x3000, sections));
            let options = ImageOptions {
                map_headers: false,
                ..ImageOptions::default()
            };
            write_image(&build(&file, options).unwrap().image).unwrap()
        };
        assert_eq!(pack(vec![a(), b()]), pack(vec![b(), a()]));

        // RELOCS_STRIPPED survives the round trip.
        let mut spec = Spec::new(0x2000, vec![Sec::new(".a", 0x1000, 0x10, vec![1], R)]);
        spec.coff_characteristics |= IMAGE_FILE_RELOCS_STRIPPED;
        let file = build_pe(&spec);
        let image = build(&file, ImageOptions::default()).unwrap().image;
        assert!(image.relocs_stripped);
        let bytes = write_image(&image).unwrap();
        assert_eq!(
            le_u32(&bytes[16..20]) & FLAG_RELOCS_STRIPPED,
            FLAG_RELOCS_STRIPPED
        );
        assert_eq!(read_image(&bytes), Ok(image));
    }

    #[test]
    fn manifest_is_deterministic_and_has_no_data_bytes() {
        let mut data = b"SECRET-GAME-BYTES".to_vec();
        data.resize(0x20, 0);
        let spec = Spec::new(
            0x3000,
            vec![
                Sec::new(".data", 0x1000, 0x40, data, R | W),
                Sec::new(".bss", 0x2000, 0x800, Vec::new(), R | W),
            ],
        );
        let file = build_pe(&spec);
        let built = build(&file, ImageOptions::default()).unwrap();
        let json = render_manifest_json(&built.image, &built.diagnostics).unwrap();
        assert_eq!(
            json,
            render_manifest_json(&built.image, &built.diagnostics).unwrap()
        );
        assert!(!json.contains("SECRET"));
        assert!(json.contains("\"image_base\": \"0x140000000\""));
        assert!(json.contains("\"container_bytes\": "));
        assert!(json.contains(&format!(
            "{{\"name\": \".data\", \"rva\": \"0x1000\", \"virtual_size\": \"0x40\", \
             \"protection\": \"rw-\", \"data_offset\": \"0x2000\", \"data_length\": \"0x11\", \
             \"zero_fill\": \"0x2f\", \"checksum\": \"0x{:016x}\"}}",
            fnv1a_64(b"SECRET-GAME-BYTES")
        )));
        assert!(json.contains("\"data_offset\": \"0x0\", \"data_length\": \"0x0\""));
        assert!(json.contains("\"relocations\": {\"total\": 0, \"dir64\": 0, \"highlow\": 0}"));
        assert!(json.ends_with("\"diagnostics\": []\n}\n"));

        let text = render_summary_text(&built.image, &built.diagnostics);
        assert!(text.starts_with("image base 0x140000000, size of image 0x3000, 3 section(s)"));
        assert!(text.contains(".bss"));
        assert!(!text.contains("SECRET"));
        assert_eq!(json_string("a\"b\\c\n"), "\"a\\\"b\\\\c\\u000a\"");
    }

    fn corrupt(bytes: &[u8], at: usize, patch: &[u8]) -> Vec<u8> {
        let mut out = bytes.to_vec();
        out[at..at + patch.len()].copy_from_slice(patch);
        out
    }

    #[test]
    fn reader_rejects_corrupted_containers() {
        let file = rebase_fixture();
        let image = build(&file, ImageOptions::default()).unwrap().image;
        assert_eq!(image.sections.len(), 4); // headers, .data, .text, .reloc
        let good = write_image(&image).unwrap();
        let entry = |i: usize| 64 + 64 * i;
        let relocs = 64 + 64 * 4;

        assert!(matches!(
            read_image(&good[..63]),
            Err(ImageError::Truncated { .. })
        ));
        assert_eq!(
            read_image(&corrupt(&good, 0, b"MZ\0\0")),
            Err(ImageError::BadMagic(*b"MZ\0\0"))
        );
        assert_eq!(
            read_image(&corrupt(&good, 4, &2u32.to_le_bytes())),
            Err(ImageError::UnsupportedVersion(2))
        );
        assert!(matches!(
            read_image(&good[..good.len() - 1]),
            Err(ImageError::BadHeaderField {
                field: "file_size",
                ..
            })
        ));
        assert!(matches!(
            read_image(&corrupt(&good, 16, &0x80u32.to_le_bytes())),
            Err(ImageError::BadHeaderField { field: "flags", .. })
        ));
        assert!(matches!(
            read_image(&corrupt(&good, 44, &0x00FF_FFFFu32.to_le_bytes())),
            Err(ImageError::Truncated { .. })
        ));

        // .text moved onto .data.
        assert!(matches!(
            read_image(&corrupt(&good, entry(2) + 16, &0x1800u32.to_le_bytes())),
            Err(ImageError::OverlappingSections { .. })
        ));
        // Initialised bytes larger than the in-memory size.
        assert!(matches!(
            read_image(&corrupt(&good, entry(1) + 20, &0x10u32.to_le_bytes())),
            Err(ImageError::InvalidSection { index: 1, .. })
        ));
        assert!(matches!(
            read_image(&corrupt(&good, entry(1) + 24, &8u32.to_le_bytes())),
            Err(ImageError::InvalidSection { index: 1, .. })
        ));
        assert!(matches!(
            read_image(&corrupt(&good, entry(1) + 60, &1u32.to_le_bytes())),
            Err(ImageError::InvalidSection { index: 1, .. })
        ));
        // Payload offsets: far out of range, misaligned, inside the tables.
        assert!(matches!(
            read_image(&corrupt(&good, entry(1) + 32, &(1u64 << 40).to_le_bytes())),
            Err(ImageError::PayloadOutOfRange { index: 1, .. })
        ));
        assert!(matches!(
            read_image(&corrupt(&good, entry(1) + 40, &u64::MAX.to_le_bytes())),
            Err(ImageError::PayloadOutOfRange { index: 1, .. })
        ));
        let data_offset = le_u64(&good[entry(1) + 32..entry(1) + 40]);
        assert_eq!(
            read_image(&corrupt(
                &good,
                entry(1) + 32,
                &(data_offset + 1).to_le_bytes()
            )),
            Err(ImageError::PayloadMisaligned {
                index: 1,
                offset: data_offset + 1
            })
        );
        assert!(matches!(
            read_image(&corrupt(&good, entry(1) + 32, &0u64.to_le_bytes())),
            Err(ImageError::PayloadOutOfRange { index: 1, .. })
        ));
        // .text pointing at .data's payload (with matching length and checksum).
        let aliased = corrupt(&good, entry(2) + 32, &good[entry(1) + 32..entry(1) + 56]);
        assert_eq!(
            read_image(&aliased),
            Err(ImageError::PayloadOverlap { index: 2 })
        );
        // Flipped payload byte.
        let mut flipped = good.clone();
        flipped[data_offset as usize + 8] ^= 0xFF;
        assert!(matches!(
            read_image(&flipped),
            Err(ImageError::ChecksumMismatch { index: 1, .. })
        ));
        // Relocation table: DIR64 list is [0x1008, 0x1FFC, 0x3010], then HIGHLOW [0x1100].
        assert_eq!(
            read_image(&corrupt(&good, relocs, &0x2000u32.to_le_bytes())),
            Err(ImageError::RelocationsUnsorted { rva: 0x1FFC })
        );
        assert!(matches!(
            read_image(&corrupt(&good, relocs + 8, &0xFFFF_FFF0u32.to_le_bytes())),
            Err(ImageError::RelocSiteOutOfRange { .. })
        ));
        assert_eq!(
            read_image(&corrupt(&good, relocs + 12, &0x1006u32.to_le_bytes())),
            Err(ImageError::OverlappingRelocations {
                first: 0x1006,
                second: 0x1008
            })
        );

        // Truncations and single-byte corruptions of the header and tables never panic.
        let tables_end = relocs + 16;
        for len in 0..tables_end + 8 {
            assert!(read_image(&good[..len]).is_err());
        }
        for at in 0..tables_end {
            for mask in [0x01, 0x80, 0xFF] {
                let mut bytes = good.clone();
                bytes[at] ^= mask;
                let _ = read_image(&bytes);
            }
        }
    }

    #[test]
    fn large_sparse_image_stays_small_and_linear() {
        const PAGES: u32 = 4096;
        const PER_PAGE: u16 = 64;
        const BSS: u32 = 0x10_0000;
        let mut relocs = Vec::new();
        for page in 0..PAGES {
            let entries: Vec<(u8, u16)> = (0..PER_PAGE)
                .map(|i| (IMAGE_REL_BASED_DIR64, i * 64))
                .collect();
            relocs.extend(reloc_block(BSS + page * 0x1_0000, &entries));
        }
        let reloc_len = relocs.len() as u32;
        let size_of_image = BSS + 0xE000_0000;
        let mut spec = Spec::new(
            size_of_image,
            vec![
                Sec::new(".data", 0x1000, 0x1000, vec![0xAB; 0x10], R | W),
                Sec::new(".reloc", 0x2000, reloc_len, relocs, R),
                Sec::new(".bss", BSS, 0xE000_0000, Vec::new(), R | W),
            ],
        );
        spec.reloc_dir = Some((0x2000, reloc_len));
        let file = build_pe(&spec);

        let image = build(&file, ImageOptions::default()).expect("builds").image;
        assert_eq!(
            image.relocations.len(),
            (PAGES * u32::from(PER_PAGE)) as usize
        );
        let bytes = write_image(&image).expect("writes");
        // Tables + payloads only: ~1.6 MiB for a 3.5 GiB guest image.
        assert!(bytes.len() < 4 << 20, "container is {} bytes", bytes.len());
        assert!(image.stored_bytes() < u64::from(reloc_len) + 0x2000);
        assert_eq!(read_image(&bytes).expect("reads"), image);
        assert_eq!(
            image.page_protections().last().map(|r| r.rva + r.size),
            Some(u64::from(size_of_image))
        );
    }
}
