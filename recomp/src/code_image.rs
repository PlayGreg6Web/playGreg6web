//! Minimal in-memory code image consumed by function discovery.
//!
//! The types are file-format agnostic (tests build them from raw byte arrays);
//! [`CodeImage::from_pe`] maps a parsed [`Pe64`] onto them.

use std::cmp::Ordering;

use crate::cfg::DiagnosticKind;
use crate::eh::EhSeeds;
use crate::pdata::FunctionTable;
use crate::pe::{IMAGE_SCN_MEM_EXECUTE, Pe64};

/// One mapped section of the image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodeSection<'a> {
    /// Absolute virtual address of the first byte.
    pub va: u64,
    /// Size of the section in memory (PE `VirtualSize`). Bytes past `data.len()` are zero-filled
    /// in memory but are never decoded as code.
    pub virtual_size: u64,
    /// Initialised bytes of the section (PE raw data).
    pub data: &'a [u8],
    /// Whether the section may contain code (PE `IMAGE_SCN_MEM_EXECUTE`).
    pub executable: bool,
}

impl<'a> CodeSection<'a> {
    /// Section whose in-memory size equals its initialised bytes.
    pub fn new(va: u64, data: &'a [u8], executable: bool) -> Self {
        Self {
            va,
            virtual_size: data.len() as u64,
            data,
            executable,
        }
    }

    /// Number of representable mapped bytes in `[va, end())`, clamped so `va + mapped_size()`
    /// never overflows `u64::MAX`.
    pub fn mapped_size(&self) -> u64 {
        let raw_size = self.virtual_size.max(self.data.len() as u64);
        raw_size.min(u64::MAX - self.va)
    }

    /// One past the last mapped VA (saturating at `u64::MAX`).
    pub fn end(&self) -> u64 {
        self.va
            .checked_add(self.mapped_size())
            .unwrap_or(u64::MAX)
    }

    /// Whether this section maps zero bytes in the representable `u64` address space.
    pub fn is_empty(&self) -> bool {
        self.mapped_size() == 0
    }

    /// Initialised bytes that lie strictly within `[va, end())` without wrapping past `u64::MAX`.
    pub fn initialized_data(&self) -> &'a [u8] {
        let max_len = usize::try_from(self.mapped_size()).unwrap_or(usize::MAX);
        &self.data[..self.data.len().min(max_len)]
    }

    /// One past the last initialised byte VA (always `<= self.end()`).
    pub fn initialized_end(&self) -> u64 {
        self.va
            .checked_add(self.initialized_data().len() as u64)
            .unwrap_or(self.end())
    }

    /// Whether `va` lies inside the mapped range of this section.
    pub fn contains(&self, va: u64) -> bool {
        va >= self.va && va < self.end()
    }

    /// Whether `[va, va + len)` lies completely within `[self.va, self.end())` without `u64`
    /// overflow.
    pub fn contains_range(&self, va: u64, len: u64) -> bool {
        let Some(range_end) = va.checked_add(len) else {
            return false;
        };
        if len == 0 {
            return self.contains(va);
        }
        va >= self.va && range_end <= self.end()
    }

    /// Initialised bytes covering `[va, va + len)` strictly within this section.
    ///
    /// Returns `None` on `u64` overflow, out-of-bounds VAs, or uninitialised zero-fill tails.
    pub fn bytes_at(&self, va: u64, len: u64) -> Option<&'a [u8]> {
        let range_end = va.checked_add(len)?;
        if !self.contains(va) || range_end > self.end() {
            return None;
        }
        let init = self.initialized_data();
        let start = usize::try_from(va.checked_sub(self.va)?).ok()?;
        if len == 0 {
            return (start < init.len()).then_some(&init[start..start]);
        }
        let count = usize::try_from(len).ok()?;
        let end = start.checked_add(count)?;
        init.get(start..end)
    }

    fn normalized(self) -> Option<Self> {
        let mapped_size = self.mapped_size();
        if mapped_size == 0 {
            return None;
        }
        let data = self.initialized_data();
        Some(Self {
            va: self.va,
            virtual_size: mapped_size,
            data,
            executable: self.executable,
        })
    }
}

/// Deterministic total order over normalized sections, independent of input permutation.
fn canonical_section_cmp(a: &CodeSection<'_>, b: &CodeSection<'_>) -> Ordering {
    a.va.cmp(&b.va)
        .then_with(|| b.end().cmp(&a.end()))
        .then_with(|| b.executable.cmp(&a.executable))
        .then_with(|| b.data.len().cmp(&a.data.len()))
        .then_with(|| a.data.cmp(b.data))
}

/// Classification of an address relative to the image layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Region {
    /// Inside an executable section.
    Code,
    /// Inside a non-executable section.
    Data,
    /// Inside the image range but outside every section (PE headers, `__ImageBase`, gaps).
    Image,
    /// Outside the image range.
    Outside,
}

impl Region {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Code => "code",
            Self::Data => "data",
            Self::Image => "image",
            Self::Outside => "outside",
        }
    }
}

/// Code image layout plus discovery seeds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeImage<'a> {
    base: u64,
    end: u64,
    /// Sorted by `va`, non-empty, and strictly non-overlapping (`prev.end() <= next.va`).
    sections: Vec<CodeSection<'a>>,
    /// Addresses where function discovery starts (PE entry point, exports, TLS callbacks, ...).
    pub entry_points: Vec<u64>,
    /// Call targets known never to return. Matches either a direct call target VA or the slot VA
    /// read by a RIP-relative indirect call (an IAT entry such as `ExitProcess`).
    pub no_return: Vec<u64>,
    /// Exact function extents (from `.pdata`, see [`crate::pdata::ExceptionDirectory::seed`]);
    /// empty when unknown. Jump-table targets must stay inside the jump's function.
    pub function_table: FunctionTable,
    /// Exception-handling seeds (resume targets, `setjmp` slots; see
    /// [`crate::eh::EhMetadata::seed`]); empty when unknown.
    pub eh: EhSeeds,
    /// VAs of 64-bit base-relocation (`DIR64`) sites in non-executable sections: the pointer
    /// slots the loader rewrites. Empty when unknown (see [`Self::from_pe`]); a pointer slot
    /// whose initial value is a code address is only trusted when it is listed here.
    pub data_relocs: std::collections::BTreeSet<u64>,
}

impl<'a> CodeImage<'a> {
    /// Build an image from its load base, sections, and entry points. The image range is
    /// `[base, max section end)`.
    ///
    /// Empty sections (zero mapped bytes or `va == u64::MAX`) are dropped, section data is
    /// clamped so it never wraps past `u64::MAX`, sections are canonically sorted independent of
    /// input order, and any overlapping later section (`section.va < prev.end()`) is discarded so
    /// lookups remain unambiguous and single-section.
    pub fn new(base: u64, sections: Vec<CodeSection<'a>>, entry_points: Vec<u64>) -> Self {
        let mut normalized: Vec<CodeSection<'a>> = sections
            .into_iter()
            .filter_map(CodeSection::normalized)
            .collect();
        normalized.sort_by(canonical_section_cmp);

        let mut non_overlapping: Vec<CodeSection<'a>> = Vec::with_capacity(normalized.len());
        for section in normalized {
            if let Some(prev) = non_overlapping.last() {
                if section.va < prev.end() {
                    continue;
                }
            }
            non_overlapping.push(section);
        }

        let end = non_overlapping
            .iter()
            .map(CodeSection::end)
            .max()
            .unwrap_or(base)
            .max(base);
        Self {
            base,
            end,
            sections: non_overlapping,
            entry_points,
            no_return: Vec::new(),
            function_table: FunctionTable::default(),
            eh: EhSeeds::default(),
            data_relocs: Default::default(),
        }
    }

    /// Map a parsed PE64 onto a code image. `file` must be the bytes `pe` was parsed from.
    ///
    /// Sections borrow their raw data from `file` (clamped to the file and to `VirtualSize`) and
    /// are executable when `IMAGE_SCN_MEM_EXECUTE` is set. Sections or entry points whose
    /// `image_base + rva` would overflow `u64` are skipped. The image range covers `SizeOfImage`.
    /// Entry points are the PE entry point, every non-forwarded export that lands in an executable
    /// section (exported data is skipped), and every TLS callback; sorted and deduplicated.
    pub fn from_pe(pe: &Pe64, file: &'a [u8]) -> Self {
        let base = pe.optional.image_base;
        let sections = pe
            .sections
            .iter()
            .filter_map(|header| {
                let va = base.checked_add(u64::from(header.virtual_address))?;
                let memory_size = if header.virtual_size == 0 {
                    header.raw_size
                } else {
                    header.virtual_size
                };
                if memory_size == 0 {
                    return None;
                }
                let start = usize::try_from(header.raw_offset)
                    .unwrap_or(file.len())
                    .min(file.len());
                let len = usize::try_from(header.raw_size.min(memory_size)).unwrap_or(file.len());
                let end = start.saturating_add(len).min(file.len());
                Some(CodeSection {
                    va,
                    virtual_size: u64::from(memory_size),
                    data: &file[start..end],
                    executable: header.characteristics & IMAGE_SCN_MEM_EXECUTE != 0,
                })
            })
            .collect();

        let mut image = Self::new(base, sections, Vec::new());
        image.end = image
            .end
            .max(base.saturating_add(u64::from(pe.optional.size_of_image)));

        let mut entries = Vec::new();
        if pe.optional.entry_point_rva != 0 {
            if let Some(entry_va) = base.checked_add(u64::from(pe.optional.entry_point_rva)) {
                entries.push(entry_va);
            }
        }
        if let Some(exports) = &pe.exports {
            entries.extend(
                exports
                    .symbols
                    .iter()
                    .filter(|sym| sym.forwarder.is_none() && sym.rva != 0)
                    .filter_map(|sym| base.checked_add(u64::from(sym.rva)))
                    .filter(|&va| image.region_of(va) == Region::Code),
            );
        }
        if let Some(tls) = &pe.tls {
            entries.extend(tls.callbacks.iter().map(|cb| cb.va));
        }
        entries.sort_unstable();
        entries.dedup();
        image.entry_points = entries;
        if let Ok(table) = crate::reloc::BaseRelocTable::from_pe(pe, file) {
            let relocs: std::collections::BTreeSet<u64> = table
                .sites()
                .iter()
                .filter(|site| site.kind == crate::reloc::BaseRelocType::Dir64)
                .filter_map(|site| site.checked_va(base))
                .filter(|&va| image.region_of(va) == Region::Data)
                .collect();
            image.data_relocs = relocs;
        }
        image
    }

    pub fn base(&self) -> u64 {
        self.base
    }

    /// One past the last VA of the image range.
    pub fn end(&self) -> u64 {
        self.end
    }

    pub fn sections(&self) -> &[CodeSection<'a>] {
        &self.sections
    }

    /// Whether `va` lies inside `[base, end)`.
    pub fn contains(&self, va: u64) -> bool {
        va >= self.base && va < self.end
    }

    /// Whether `[va, va + len)` lies inside `[base, end)` without `u64` overflow.
    pub fn contains_range(&self, va: u64, len: u64) -> bool {
        let Some(range_end) = va.checked_add(len) else {
            return false;
        };
        if len == 0 {
            return self.contains(va);
        }
        va >= self.base && range_end <= self.end
    }

    /// Section mapping `va`, if any.
    pub fn section_at(&self, va: u64) -> Option<&CodeSection<'a>> {
        let idx = self.sections.partition_point(|s| s.va <= va);
        let section = self.sections.get(idx.checked_sub(1)?)?;
        section.contains(va).then_some(section)
    }

    /// Single section completely covering `[va, va + len)` without `u64` overflow.
    ///
    /// Never bridges across gaps or adjacent sections.
    pub fn section_for_range(&self, va: u64, len: u64) -> Option<&CodeSection<'a>> {
        let section = self.section_at(va)?;
        section.contains_range(va, len).then_some(section)
    }

    pub fn region_of(&self, va: u64) -> Region {
        match self.section_at(va) {
            Some(s) if s.executable => Region::Code,
            Some(_) => Region::Data,
            None if self.contains(va) => Region::Image,
            None => Region::Outside,
        }
    }

    /// Initialised bytes covering `[va, va + len)` strictly within a single mapped section.
    ///
    /// Returns `None` on `u64` overflow, unmapped or gap addresses, uninitialised zero-fill tails,
    /// or ranges spanning more than one section.
    pub fn bytes_at(&self, va: u64, len: u64) -> Option<&'a [u8]> {
        let section = self.section_for_range(va, len)?;
        section.bytes_at(va, len)
    }

    /// Decodable bytes starting at `va` up to the end of its section's initialised data.
    ///
    /// Never spans into an adjacent section, gap, or past `u64::MAX`.
    pub fn code_bytes(&self, va: u64) -> Result<&'a [u8], DiagnosticKind> {
        let Some(section) = self.section_at(va) else {
            return Err(DiagnosticKind::Unmapped);
        };
        if !section.executable {
            return Err(DiagnosticKind::NotExecutable);
        }
        let init = section.initialized_data();
        let Some(offset) = va
            .checked_sub(section.va)
            .and_then(|off| usize::try_from(off).ok())
        else {
            return Err(DiagnosticKind::Unmapped);
        };
        if offset >= init.len() {
            return Err(DiagnosticKind::NotExecutable);
        }
        Ok(&init[offset..])
    }

    /// Decodable bytes covering `[va, va + len)` strictly within a single executable section.
    ///
    /// Rejects `u64` overflow, unmapped/gap addresses, and cross-section ranges with
    /// [`DiagnosticKind::Unmapped`], and non-executable sections or uninitialised zero-fill tails
    /// with [`DiagnosticKind::NotExecutable`].
    pub fn code_bytes_range(&self, va: u64, len: u64) -> Result<&'a [u8], DiagnosticKind> {
        let Some(section) = self.section_for_range(va, len) else {
            return Err(DiagnosticKind::Unmapped);
        };
        if !section.executable {
            return Err(DiagnosticKind::NotExecutable);
        }
        section.bytes_at(va, len).ok_or(DiagnosticKind::NotExecutable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pe::test_util::build_synthetic_pe64;

    #[test]
    fn section_lookup_and_regions() {
        let text = [0x90u8; 0x10];
        let data = [0u8; 0x8];
        let mut bss = CodeSection::new(0x3000, &data, false);
        bss.virtual_size = 0x100;
        let image = CodeImage::new(
            0x1000,
            vec![bss, CodeSection::new(0x2000, &text, true)],
            vec![0x2000],
        );

        assert_eq!(image.end(), 0x3100);
        assert_eq!(image.sections()[0].va, 0x2000, "sections are sorted");
        assert_eq!(image.region_of(0x1000), Region::Image);
        assert_eq!(image.region_of(0x200f), Region::Code);
        assert_eq!(image.region_of(0x2010), Region::Image);
        assert_eq!(image.region_of(0x30ff), Region::Data);
        assert_eq!(image.region_of(0x3100), Region::Outside);
        assert_eq!(image.region_of(0xfff), Region::Outside);

        assert_eq!(image.code_bytes(0x200e).map(<[u8]>::len), Ok(2));
        assert_eq!(image.code_bytes(0x2010), Err(DiagnosticKind::Unmapped));
        assert_eq!(image.code_bytes(0x3000), Err(DiagnosticKind::NotExecutable));
        assert_eq!(image.code_bytes(0x9000), Err(DiagnosticKind::Unmapped));
    }

    #[test]
    fn from_pe_maps_sections_and_entry_points() {
        let file = build_synthetic_pe64(false);
        let pe = Pe64::parse(&file).expect("synthetic PE parses");
        let image = CodeImage::from_pe(&pe, &file);
        let base = 0x1_4000_0000;

        assert_eq!(image.base(), base);
        assert_eq!(image.end(), base + 0x3000, "covers SizeOfImage");
        let text = image.sections()[0];
        assert_eq!((text.va, text.virtual_size), (base + 0x1000, 0x120));
        assert_eq!(text.data.len(), 0x120, "raw data clamped to VirtualSize");
        assert_eq!(text.data, &file[0x200..0x320]);
        assert!(text.executable);
        assert!(!image.sections()[1].executable);
        assert_eq!(image.region_of(base + 0x100), Region::Image, "headers");
        // Entry point + exports (.text+0x10, +0x20; forwarder skipped) + TLS callbacks.
        assert_eq!(
            image.entry_points,
            [base + 0x1010, base + 0x1020, base + 0x1030, base + 0x1040]
        );
    }

    #[test]
    fn from_pe_clamps_section_data_to_file() {
        let mut file = build_synthetic_pe64(false);
        let pe = Pe64::parse(&file).expect("synthetic PE parses");
        file.truncate(0x280);
        let image = CodeImage::from_pe(&pe, &file);
        assert_eq!(image.sections()[0].data.len(), 0x80);
        assert!(image.sections()[1].data.is_empty());
    }

    #[test]
    fn executable_zero_fill_tail_is_not_decodable() {
        let text = [0xc3u8; 4];
        let mut section = CodeSection::new(0x1000, &text, true);
        section.virtual_size = 0x1000;
        let image = CodeImage::new(0x1000, vec![section], vec![]);
        assert_eq!(image.region_of(0x1004), Region::Code);
        assert_eq!(image.code_bytes(0x1004), Err(DiagnosticKind::NotExecutable));
    }

    #[test]
    fn near_u64_max_va_and_length_arithmetic_never_wraps() {
        let bytes = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, // representable in [u64::MAX - 8, u64::MAX)
            0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99, // past u64::MAX; must be clamped
        ];
        let base = u64::MAX - 16;
        let sec_va = u64::MAX - 8;
        let image = CodeImage::new(
            base,
            vec![
                CodeSection::new(sec_va, &bytes, true),
                // Placed directly at u64::MAX: maps 0 representable bytes and must be dropped.
                CodeSection::new(u64::MAX, &bytes, true),
            ],
            vec![sec_va],
        );

        assert_eq!(image.sections().len(), 1);
        let sec = image.sections()[0];
        assert_eq!(sec.va, sec_va);
        assert_eq!(sec.end(), u64::MAX);
        assert_eq!(sec.mapped_size(), 8);
        assert_eq!(sec.initialized_data(), &bytes[..8]);
        assert_eq!(sec.initialized_end(), u64::MAX);
        assert_eq!(image.end(), u64::MAX);

        // Checked range containment near u64::MAX.
        assert!(image.contains(sec_va));
        assert!(image.contains(u64::MAX - 1));
        assert!(!image.contains(u64::MAX));
        assert!(image.contains_range(sec_va, 8));
        assert!(!image.contains_range(sec_va, 9));
        assert!(!image.contains_range(u64::MAX, 0));
        assert!(!image.contains_range(u64::MAX, 1));
        assert!(!image.contains_range(sec_va, u64::MAX));

        // Checked byte and code slicing never wraps or exposes clamped tail bytes.
        assert_eq!(image.code_bytes(sec_va), Ok(&bytes[..8]));
        assert_eq!(image.code_bytes(u64::MAX - 1), Ok(&bytes[7..8]));
        assert_eq!(image.code_bytes(u64::MAX), Err(DiagnosticKind::Unmapped));

        assert_eq!(image.bytes_at(sec_va, 8), Some(&bytes[..8]));
        assert_eq!(image.bytes_at(sec_va, 9), None);
        assert_eq!(image.bytes_at(u64::MAX - 1, 2), None);
        assert_eq!(image.bytes_at(sec_va, u64::MAX), None);
        assert_eq!(image.bytes_at(u64::MAX, 0), None);
        assert_eq!(image.bytes_at(u64::MAX, 1), None);

        assert_eq!(image.code_bytes_range(sec_va, 8), Ok(&bytes[..8]));
        assert_eq!(
            image.code_bytes_range(sec_va, 9),
            Err(DiagnosticKind::Unmapped)
        );
        assert_eq!(
            image.code_bytes_range(u64::MAX, 1),
            Err(DiagnosticKind::Unmapped)
        );
        assert_eq!(image.section_at(u64::MAX), None);
        assert_eq!(image.region_of(u64::MAX), Region::Outside);
    }

    #[test]
    fn unmapped_gap_and_zero_fill_tail_reads_are_rejected() {
        let text = [0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0xc3];
        let mut code_sec = CodeSection::new(0x2000, &text, true);
        code_sec.virtual_size = 0x10; // [0x2008, 0x2010) is uninitialised zero-fill tail
        let rdata = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        let data_sec = CodeSection::new(0x4000, &rdata, false);

        let image = CodeImage::new(0x1000, vec![code_sec, data_sec], vec![0x2000]);

        // Unmapped headers, gaps, and addresses outside the image.
        for va in [0x0, 0xfff, 0x1000, 0x1fff, 0x2010, 0x3000, 0x3fff, 0x4008, u64::MAX] {
            assert_eq!(image.section_at(va), None, "va=0x{va:x}");
            assert_eq!(image.bytes_at(va, 0), None, "zero-len at va=0x{va:x}");
            assert_eq!(image.bytes_at(va, 1), None, "1-byte at va=0x{va:x}");
            assert_eq!(
                image.code_bytes_range(va, 1),
                Err(DiagnosticKind::Unmapped),
                "code range at va=0x{va:x}"
            );
        }

        // Reads straddling section-to-gap boundaries are rejected.
        assert_eq!(image.bytes_at(0x2004, 0x10), None);
        assert_eq!(
            image.code_bytes_range(0x2004, 0x10),
            Err(DiagnosticKind::Unmapped)
        );
        assert_eq!(image.bytes_at(0x4004, 8), None);
        assert_eq!(
            image.code_bytes_range(0x4004, 8),
            Err(DiagnosticKind::Unmapped)
        );

        // Zero-fill tail is mapped as Region::Code, but unreadable/non-decodable.
        assert_eq!(image.region_of(0x2008), Region::Code);
        assert!(image.section_for_range(0x2004, 8).is_some());
        assert_eq!(image.bytes_at(0x2008, 0), None);
        assert_eq!(image.bytes_at(0x2008, 1), None);
        assert_eq!(image.bytes_at(0x2004, 5), None);
        assert_eq!(
            image.code_bytes_range(0x2008, 1),
            Err(DiagnosticKind::NotExecutable)
        );
        assert_eq!(
            image.code_bytes_range(0x2004, 5),
            Err(DiagnosticKind::NotExecutable)
        );
        assert_eq!(image.bytes_at(0x2000, 8), Some(&text[..]));
        assert_eq!(image.code_bytes_range(0x2000, 8), Ok(&text[..]));

        // Non-executable data section supports bytes_at but rejects code_bytes_range.
        assert_eq!(image.bytes_at(0x4000, 8), Some(&rdata[..]));
        assert_eq!(image.bytes_at(0x4000, 0), Some(&rdata[0..0]));
        assert_eq!(
            image.code_bytes_range(0x4000, 8),
            Err(DiagnosticKind::NotExecutable)
        );
    }

    #[test]
    fn cross_section_slices_never_bridge_adjacent_or_gapped_sections() {
        let first = [0x90, 0x90, 0x90, 0x90];
        let second = [0xcc, 0xcc, 0xcc, 0xc3];
        let third = [0xc3, 0xc3, 0xc3, 0xc3];
        let image = CodeImage::new(
            0x1000,
            vec![
                CodeSection::new(0x2000, &first, true),
                CodeSection::new(0x2004, &second, true), // immediately adjacent to first
                CodeSection::new(0x3000, &third, true),  // separated by gap
            ],
            vec![0x2000],
        );

        // Single-section boundary reads succeed.
        assert_eq!(image.bytes_at(0x2000, 4), Some(&first[..]));
        assert_eq!(image.bytes_at(0x2004, 4), Some(&second[..]));
        assert_eq!(image.bytes_at(0x2003, 1), Some(&first[3..4]));
        assert_eq!(image.code_bytes(0x2000), Ok(&first[..]));
        assert_eq!(image.code_bytes(0x2004), Ok(&second[..]));
        assert_eq!(image.code_bytes_range(0x2000, 4), Ok(&first[..]));
        assert_eq!(image.code_bytes_range(0x2004, 4), Ok(&second[..]));

        // Overall image contains [0x2000, 0x2008), but single-section slicing must NOT bridge.
        assert!(image.contains_range(0x2000, 8));
        for (va, len) in [
            (0x2000, 5),
            (0x2000, 8),
            (0x2002, 4),
            (0x2003, 2),
            (0x2004, 0x1000),
            (0x2000, 0x1004),
        ] {
            assert_eq!(
                image.section_for_range(va, len),
                None,
                "cross-section range [0x{va:x}, +0x{len:x}) must not match a single section"
            );
            assert_eq!(image.bytes_at(va, len), None);
            assert_eq!(
                image.code_bytes_range(va, len),
                Err(DiagnosticKind::Unmapped)
            );
        }
    }

    #[test]
    fn empty_sections_are_filtered_and_do_not_shadow_lookups() {
        let text = [0x90u8; 0x10];
        let mut bss_only = CodeSection::new(0x3000, &[], false);
        bss_only.virtual_size = 0x20;

        let empty_same_va = CodeSection::new(0x2000, &[], false);
        let empty_mid_section = CodeSection::new(0x2004, &[], false);
        let empty_far_past = CodeSection::new(0x9000_0000, &[], true);
        let empty_at_max = CodeSection::new(u64::MAX, &[0x90], true);

        assert!(empty_same_va.is_empty());
        assert!(!empty_same_va.contains(0x2000));
        assert!(!empty_same_va.contains_range(0x2000, 0));
        assert_eq!(empty_same_va.bytes_at(0x2000, 0), None);
        assert!(empty_at_max.is_empty());

        let image = CodeImage::new(
            0x1000,
            vec![
                empty_far_past,
                empty_mid_section,
                CodeSection::new(0x2000, &text, true),
                empty_same_va,
                bss_only,
                empty_at_max,
            ],
            vec![0x2000],
        );

        // Only the two non-empty sections (.text and BSS-only) remain; image.end() is not inflated.
        assert_eq!(image.sections().len(), 2);
        assert_eq!(image.sections()[0].va, 0x2000);
        assert_eq!(image.sections()[1].va, 0x3000);
        assert_eq!(image.end(), 0x3020);

        // Lookups across .text are never shadowed by mid-section or same-VA empty sections.
        assert_eq!(image.section_at(0x2000).map(|s| s.va), Some(0x2000));
        assert_eq!(image.section_at(0x2004).map(|s| s.va), Some(0x2000));
        assert_eq!(image.section_at(0x200f).map(|s| s.va), Some(0x2000));
        assert_eq!(image.bytes_at(0x2000, 0x10), Some(&text[..]));

        // BSS-only section is mapped as Region::Data but has no initialised bytes.
        assert_eq!(image.region_of(0x3000), Region::Data);
        assert_eq!(image.bytes_at(0x3000, 0), None);
        assert_eq!(image.bytes_at(0x3000, 1), None);
    }

    #[test]
    fn overlapping_sections_are_guarded_and_lookup_order_is_deterministic() {
        let wide_code = [0x90u8; 0x100];
        let narrow_data = [0xaa; 0x20];
        let nested_data = [0xbb; 0x20];
        let partial_tail = [0xcc; 0x100];
        let disjoint_after = [0xc3; 0x10];

        let sec_wide = CodeSection::new(0x2000, &wide_code, true); // [0x2000, 0x2100)
        let sec_same_va = CodeSection::new(0x2000, &narrow_data, false); // [0x2000, 0x2020)
        let sec_nested = CodeSection::new(0x2040, &nested_data, false); // [0x2040, 0x2060)
        let sec_partial = CodeSection::new(0x2080, &partial_tail, false); // [0x2080, 0x2180)
        let sec_after = CodeSection::new(0x2200, &disjoint_after, true); // [0x2200, 0x2210)

        let permutations = [
            vec![sec_wide, sec_same_va, sec_nested, sec_partial, sec_after],
            vec![sec_after, sec_partial, sec_nested, sec_same_va, sec_wide],
            vec![sec_same_va, sec_nested, sec_after, sec_wide, sec_partial],
            vec![sec_partial, sec_wide, sec_after, sec_same_va, sec_nested],
        ];

        let reference = CodeImage::new(0x1000, permutations[0].clone(), vec![0x2000]);

        // Kept sections are strictly sorted and non-overlapping.
        assert_eq!(reference.sections().len(), 2);
        assert_eq!(reference.sections()[0], sec_wide);
        assert_eq!(reference.sections()[1], sec_after);
        for pair in reference.sections().windows(2) {
            assert!(pair[0].end() <= pair[1].va);
        }

        // Tail past the nested section resolves cleanly to sec_wide; rejected partial tail is unmapped.
        assert_eq!(reference.region_of(0x2000), Region::Code);
        assert_eq!(reference.region_of(0x2050), Region::Code);
        assert_eq!(reference.region_of(0x20f0), Region::Code);
        assert_eq!(reference.region_of(0x2120), Region::Image);
        assert_eq!(reference.region_of(0x2200), Region::Code);
        assert_eq!(reference.bytes_at(0x2200, 0x10), Some(&disjoint_after[..]));

        // Every permutation yields the exact same CodeImage and query outcomes.
        for perm in &permutations[1..] {
            let candidate = CodeImage::new(0x1000, perm.clone(), vec![0x2000]);
            assert_eq!(candidate, reference);
            for va in [0x1000, 0x2000, 0x2040, 0x2080, 0x20ff, 0x2100, 0x2120, 0x2200] {
                assert_eq!(candidate.section_at(va), reference.section_at(va));
                assert_eq!(candidate.region_of(va), reference.region_of(va));
            }
        }

        // Equal-range tie-breaking: executable wins over data, then lexicographic data bytes.
        let eq_data = CodeSection::new(0x5000, &[0x01; 8], false);
        let eq_code_b = CodeSection::new(0x5000, &[0x22; 8], true);
        let eq_code_a = CodeSection::new(0x5000, &[0x11; 8], true);
        let tie_1 = CodeImage::new(0x5000, vec![eq_data, eq_code_b, eq_code_a], vec![]);
        let tie_2 = CodeImage::new(0x5000, vec![eq_code_a, eq_data, eq_code_b], vec![]);
        let tie_3 = CodeImage::new(0x5000, vec![eq_code_b, eq_code_a, eq_data], vec![]);
        assert_eq!(tie_1, tie_2);
        assert_eq!(tie_2, tie_3);
        assert_eq!(tie_1.sections(), &[eq_code_a]);
    }

    #[test]
    fn from_pe_skips_overflowing_rvas_near_u64_max() {
        let file = build_synthetic_pe64(false);
        let mut pe = Pe64::parse(&file).expect("synthetic PE parses");
        // Image base so high that every section/export/entry RVA (>= 0x1000) overflows u64.
        pe.optional.image_base = u64::MAX - 0x100;
        let image = CodeImage::from_pe(&pe, &file);

        assert!(image.sections().is_empty());
        assert_eq!(image.end(), u64::MAX);
        // Only TLS callbacks (which carry absolute VAs, not base + RVA) can remain; no bogus u64::MAX.
        assert!(!image.entry_points.contains(&u64::MAX));
        assert_eq!(image.section_at(u64::MAX), None);
        assert_eq!(image.bytes_at(u64::MAX - 0x10, 0x20), None);
    }
}
