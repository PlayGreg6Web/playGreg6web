//! Image base assignment and build-time rebasing of a program's modules.

use crate::image::{self, BaseRelocKind, DIR_BASERELOC, IMAGE_FILE_RELOCS_STRIPPED};
use crate::pe::Pe64;

/// Image ranges are reserved in units of the Windows allocation granularity.
pub const ALLOCATION_GRANULARITY: u64 = 0x1_0000;
/// Every module must end below the runtime's virtual module handles
/// (`kVirtualModuleBase` in `runtime/os/win32_loader.h`).
pub const ADDRESS_LIMIT: u64 = 0x0000_7FF8_0000_0000;

/// One module to place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    pub name: String,
    pub preferred_base: u64,
    pub size_of_image: u32,
    /// `IMAGE_FILE_RELOCS_STRIPPED` is not set (the module can be moved).
    pub relocatable: bool,
}

/// Reserved extent of an image: `SizeOfImage` rounded up to the allocation granularity.
pub fn reserved_size(size_of_image: u32) -> u64 {
    u64::from(size_of_image).next_multiple_of(ALLOCATION_GRANULARITY)
}

/// Assigns every module a disjoint range, in request order: a module keeps its preferred base
/// when `[base, base + reserved)` overlaps nothing placed before it; otherwise it goes to the
/// lowest granularity-aligned address at or above its preferred base where it fits (a module
/// that cannot be moved is an error). The first request (the executable) therefore always keeps
/// its base, and the result depends only on the requests and their order.
pub fn assign_bases(requests: &[Request]) -> Result<Vec<u64>, String> {
    let mut placed: Vec<(u64, u64)> = Vec::new();
    let mut bases = Vec::with_capacity(requests.len());
    for request in requests {
        let size = reserved_size(request.size_of_image);
        let fits = |start: u64, placed: &[(u64, u64)]| {
            start != 0
                && start
                    .checked_add(size)
                    .is_some_and(|end| end <= ADDRESS_LIMIT)
                && placed.iter().all(|&(s, e)| start + size <= s || e <= start)
        };
        let base = if fits(request.preferred_base, &placed) {
            request.preferred_base
        } else if !request.relocatable {
            return Err(format!(
                "{} has its relocations stripped and cannot move from its preferred base 0x{:x}, \
                 which overlaps another module",
                request.name, request.preferred_base
            ));
        } else {
            let start = request
                .preferred_base
                .next_multiple_of(ALLOCATION_GRANULARITY);
            let mut candidates: Vec<u64> = placed
                .iter()
                .map(|&(_, end)| end.next_multiple_of(ALLOCATION_GRANULARITY))
                .filter(|&c| c >= start)
                .collect();
            candidates.push(start);
            candidates.sort_unstable();
            candidates
                .into_iter()
                .find(|&c| fits(c, &placed))
                .ok_or_else(|| {
                    format!(
                        "no free range of 0x{size:x} bytes at or above 0x{start:x} for {}",
                        request.name
                    )
                })?
        };
        placed.push((base, base + size));
        bases.push(base);
    }
    Ok(bases)
}

/// Whether the PE can be moved from its preferred base.
pub fn relocatable(pe: &Pe64) -> bool {
    pe.coff.characteristics & IMAGE_FILE_RELOCS_STRIPPED == 0
}

/// The file bytes of `pe` as if the loader had mapped it at `new_base`: every base relocation
/// site gets `new_base - preferred` added (`DIR64` 64-bit, `HIGHLOW` low 32 bits, exactly like
/// [`image::apply_relocations`]) and `OptionalHeader.ImageBase` becomes `new_base`. Everything
/// derived from the result (discovery, lifted constants, `.pdata` and EH VAs, export and IAT
/// VAs, TLS directory, the packed image) is then consistent with `new_base`, and the guest image
/// loads there without further relocation. Sites outside file-backed bytes are an error (the
/// file cannot carry the relocated value).
pub fn rebase_pe(file: &[u8], pe: &Pe64, new_base: u64) -> Result<Vec<u8>, String> {
    let preferred = pe.optional.image_base;
    let mut bytes = file.to_vec();
    if new_base == preferred {
        return Ok(bytes);
    }
    if !relocatable(pe) {
        return Err("relocations stripped (IMAGE_FILE_RELOCS_STRIPPED)".to_string());
    }
    let relocations = match pe.optional.data_directory(DIR_BASERELOC) {
        Some(dir) => {
            let start = pe
                .rva_to_file_offset(dir.virtual_address)
                .ok_or("base relocation directory is not file-backed")?;
            let end = start + dir.size as usize;
            if dir.size == 0
                || pe.rva_to_file_offset(dir.virtual_address + dir.size - 1) != Some(end - 1)
            {
                return Err("base relocation directory is not contiguous in the file".to_string());
            }
            image::parse_base_relocations(&file[start..end])
                .map_err(|e| format!("bad base relocations: {e}"))?
                .relocations
        }
        None => Vec::new(),
    };
    let delta = new_base.wrapping_sub(preferred);
    for reloc in &relocations {
        let width = reloc.kind.width();
        let offset = pe
            .rva_to_file_offset(reloc.rva)
            .filter(|&o| {
                pe.rva_to_file_offset(reloc.rva + width - 1) == Some(o + width as usize - 1)
            })
            .ok_or_else(|| {
                format!(
                    "base relocation at RVA 0x{:x} lies outside file-backed bytes",
                    reloc.rva
                )
            })?;
        match reloc.kind {
            BaseRelocKind::Dir64 => {
                let slot = &mut bytes[offset..offset + 8];
                let value = u64::from_le_bytes(slot.try_into().expect("8 bytes"));
                slot.copy_from_slice(&value.wrapping_add(delta).to_le_bytes());
            }
            BaseRelocKind::HighLow => {
                let slot = &mut bytes[offset..offset + 4];
                let value = u32::from_le_bytes(slot.try_into().expect("4 bytes"));
                slot.copy_from_slice(&value.wrapping_add(delta as u32).to_le_bytes());
            }
        }
    }
    // OptionalHeader64.ImageBase: PE signature (4) + COFF header (20) + 24.
    let field = pe.e_lfanew as usize + 4 + 20 + 24;
    bytes[field..field + 8].copy_from_slice(&new_base.to_le_bytes());
    Ok(bytes)
}
