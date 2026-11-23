//! Minimal in-memory code image consumed by function discovery.
//!
//! The types are file-format agnostic (tests build them from raw byte arrays);
//! [`CodeImage::from_pe`] maps a parsed [`Pe64`] onto them.

use crate::cfg::DiagnosticKind;
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

    /// One past the last mapped VA (saturating at `u64::MAX`).
    pub fn end(&self) -> u64 {
        let size = self.virtual_size.max(self.data.len() as u64);
        self.va.saturating_add(size)
    }

    /// Whether `va` lies inside the mapped range of this section.
    pub fn contains(&self, va: u64) -> bool {
        va >= self.va && va < self.end()
    }
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
    /// Sorted by `va`; sections are expected not to overlap.
    sections: Vec<CodeSection<'a>>,
    /// Addresses where function discovery starts (PE entry point, exports, TLS callbacks, ...).
    pub entry_points: Vec<u64>,
    /// Call targets known never to return. Matches either a direct call target VA or the slot VA
    /// read by a RIP-relative indirect call (an IAT entry such as `ExitProcess`).
    pub no_return: Vec<u64>,
    /// Exact function extents (from `.pdata`, see [`crate::pdata::ExceptionDirectory::seed`]);
    /// empty when unknown. Jump-table targets must stay inside the jump's function.
    pub function_table: FunctionTable,
}

impl<'a> CodeImage<'a> {
    /// Build an image from its load base, sections, and entry points. The image range is
    /// `[base, max section end)`.
    pub fn new(base: u64, mut sections: Vec<CodeSection<'a>>, entry_points: Vec<u64>) -> Self {
        sections.sort_by_key(|s| s.va);
        let end = sections
            .iter()
            .map(CodeSection::end)
            .max()
            .unwrap_or(base)
            .max(base);
        Self {
            base,
            end,
            sections,
            entry_points,
            no_return: Vec::new(),
            function_table: FunctionTable::default(),
        }
    }

    /// Map a parsed PE64 onto a code image. `file` must be the bytes `pe` was parsed from.
    ///
    /// Sections borrow their raw data from `file` (clamped to the file and to `VirtualSize`) and
    /// are executable when `IMAGE_SCN_MEM_EXECUTE` is set. The image range covers `SizeOfImage`.
    /// Entry points are the PE entry point, every non-forwarded export that lands in an executable
    /// section (exported data is skipped), and every TLS callback; sorted and deduplicated.
    pub fn from_pe(pe: &Pe64, file: &'a [u8]) -> Self {
        let base = pe.optional.image_base;
        let sections = pe
            .sections
            .iter()
            .map(|header| {
                let memory_size = if header.virtual_size == 0 {
                    header.raw_size
                } else {
                    header.virtual_size
                };
                let start = (header.raw_offset as usize).min(file.len());
                let len = header.raw_size.min(memory_size) as usize;
                let end = start.saturating_add(len).min(file.len());
                CodeSection {
                    va: base.saturating_add(u64::from(header.virtual_address)),
                    virtual_size: u64::from(memory_size),
                    data: &file[start..end],
                    executable: header.characteristics & IMAGE_SCN_MEM_EXECUTE != 0,
                }
            })
            .collect();

        let mut image = Self::new(base, sections, Vec::new());
        image.end = image
            .end
            .max(base.saturating_add(u64::from(pe.optional.size_of_image)));

        let mut entries = Vec::new();
        if pe.optional.entry_point_rva != 0 {
            entries.push(base.saturating_add(u64::from(pe.optional.entry_point_rva)));
        }
        if let Some(exports) = &pe.exports {
            entries.extend(
                exports
                    .symbols
                    .iter()
                    .filter(|sym| sym.forwarder.is_none() && sym.rva != 0)
                    .map(|sym| base.saturating_add(u64::from(sym.rva)))
                    .filter(|&va| image.region_of(va) == Region::Code),
            );
        }
        if let Some(tls) = &pe.tls {
            entries.extend(tls.callbacks.iter().map(|cb| cb.va));
        }
        entries.sort_unstable();
        entries.dedup();
        image.entry_points = entries;
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

    /// Section mapping `va`, if any.
    pub fn section_at(&self, va: u64) -> Option<&CodeSection<'a>> {
        let idx = self.sections.partition_point(|s| s.va <= va);
        let section = self.sections.get(idx.checked_sub(1)?)?;
        section.contains(va).then_some(section)
    }

    pub fn region_of(&self, va: u64) -> Region {
        match self.section_at(va) {
            Some(s) if s.executable => Region::Code,
            Some(_) => Region::Data,
            None if self.contains(va) => Region::Image,
            None => Region::Outside,
        }
    }

    /// Decodable bytes starting at `va` up to the end of its section's initialised data.
    pub fn code_bytes(&self, va: u64) -> Result<&'a [u8], DiagnosticKind> {
        let Some(section) = self.section_at(va) else {
            return Err(DiagnosticKind::Unmapped);
        };
        let offset = va - section.va;
        if !section.executable || offset >= section.data.len() as u64 {
            return Err(DiagnosticKind::NotExecutable);
        }
        Ok(&section.data[offset as usize..])
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
}
