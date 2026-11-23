//! Relocation map: every place in discovered code that references an image address, so later
//! stages can retarget or rebase it.

use iced_x86::{FlowControl, Instruction, Mnemonic, OpKind, Register};

use crate::code_image::{CodeImage, Region};

/// How an instruction references its target address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RelocKind {
    /// Direct `call rel32` target.
    CallTarget,
    /// Direct `jmp` / `jcc` / `loop` / `jrcxz` / `xbegin` target.
    BranchTarget,
    /// RIP-relative memory operand that is read or written as data.
    RipData,
    /// RIP-relative address computed by `lea` (no memory access).
    RipAddress,
    /// Slot read by an indirect `call [rip+x]` (import thunk / IAT entry).
    CallSlot,
    /// Slot read by an indirect `jmp [rip+x]` (import thunk / IAT entry).
    JumpSlot,
    /// Heuristic: 64-bit immediate (`mov r64, imm64`, sign-extended imm32) inside the image.
    AbsoluteImmediate,
    /// Heuristic: absolute memory displacement without base register (`[disp32]`,
    /// `[index*scale+disp32]`, `moffs64`) inside the image.
    AbsoluteDisplacement,
    /// Jump table read by a resolved switch dispatch (`site` is the entry load).
    JumpTable,
    /// Byte index table of a two-level switch (`site` is the `movzx` reading it).
    JumpTableIndex,
    /// One jump-table entry: `site` is the VA of the 32-bit entry itself (data, not an
    /// instruction) and `target` the case it selects.
    JumpTableEntry,
}

impl RelocKind {
    /// Whether this entry is a guess (a constant that merely falls inside the image range).
    pub fn is_possible(self) -> bool {
        matches!(self, Self::AbsoluteImmediate | Self::AbsoluteDisplacement)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::CallTarget => "call_target",
            Self::BranchTarget => "branch_target",
            Self::RipData => "rip_data",
            Self::RipAddress => "rip_address",
            Self::CallSlot => "call_slot",
            Self::JumpSlot => "jump_slot",
            Self::AbsoluteImmediate => "absolute_immediate",
            Self::AbsoluteDisplacement => "absolute_displacement",
            Self::JumpTable => "jump_table",
            Self::JumpTableIndex => "jump_table_index",
            Self::JumpTableEntry => "jump_table_entry",
        }
    }
}

/// One address reference. Ordering is `(site, kind, target)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Relocation {
    /// VA of the referencing instruction (of the table entry for
    /// [`RelocKind::JumpTableEntry`]).
    pub site: u64,
    pub kind: RelocKind,
    /// Absolute referenced VA.
    pub target: u64,
    /// Where `target` lies (code vs data vs outside).
    pub region: Region,
}

/// Sorted, deduplicated relocations.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RelocationMap {
    entries: Vec<Relocation>,
}

impl RelocationMap {
    /// Build a map from arbitrary relocations (sorts and deduplicates).
    pub fn from_entries(mut entries: Vec<Relocation>) -> Self {
        entries.sort_unstable();
        entries.dedup();
        Self { entries }
    }

    /// Collect relocations from decoded instructions.
    pub fn from_instructions<'i>(
        image: &CodeImage<'_>,
        instructions: impl IntoIterator<Item = &'i Instruction>,
    ) -> Self {
        let mut entries = Vec::new();
        for instr in instructions {
            instruction_relocations(image, instr, &mut entries);
        }
        Self::from_entries(entries)
    }

    pub fn entries(&self) -> &[Relocation] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Relocations of the instruction at `site`.
    pub fn at(&self, site: u64) -> &[Relocation] {
        let lo = self.entries.partition_point(|r| r.site < site);
        let hi = lo + self.entries[lo..].partition_point(|r| r.site == site);
        &self.entries[lo..hi]
    }
}

/// Append the address references made by `instr` to `out` (unsorted).
pub fn instruction_relocations(
    image: &CodeImage<'_>,
    instr: &Instruction,
    out: &mut Vec<Relocation>,
) {
    let site = instr.ip();
    let mut push = |kind: RelocKind, target: u64| {
        out.push(Relocation {
            site,
            kind,
            target,
            region: image.region_of(target),
        });
    };

    for operand in 0..instr.op_count() {
        match instr.op_kind(operand) {
            OpKind::NearBranch64 => {
                let kind = if instr.flow_control() == FlowControl::Call {
                    RelocKind::CallTarget
                } else {
                    RelocKind::BranchTarget
                };
                push(kind, instr.near_branch64());
            }
            OpKind::Memory if instr.is_ip_rel_memory_operand() => {
                let kind = match instr.flow_control() {
                    FlowControl::IndirectCall => RelocKind::CallSlot,
                    FlowControl::IndirectBranch => RelocKind::JumpSlot,
                    _ if instr.mnemonic() == Mnemonic::Lea => RelocKind::RipAddress,
                    _ => RelocKind::RipData,
                };
                push(kind, instr.ip_rel_memory_address());
            }
            OpKind::Memory if instr.memory_base() == Register::None => {
                let target = instr.memory_displacement64();
                if image.contains(target) {
                    push(RelocKind::AbsoluteDisplacement, target);
                }
            }
            OpKind::Immediate64 => {
                let value = instr.immediate64();
                if image.contains(value) {
                    push(RelocKind::AbsoluteImmediate, value);
                }
            }
            OpKind::Immediate32to64 => {
                let value = instr.immediate32to64() as u64;
                if image.contains(value) {
                    push(RelocKind::AbsoluteImmediate, value);
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code_image::CodeSection;
    use iced_x86::{Decoder, DecoderOptions};

    fn decode_one(bytes: &[u8], ip: u64) -> Instruction {
        let instr = Decoder::with_ip(64, bytes, ip, DecoderOptions::NONE).decode();
        assert!(!instr.is_invalid());
        instr
    }

    #[test]
    fn low_image_jump_table_displacement_and_imm32() {
        // Image loaded below 2 GiB so 32-bit absolutes can reach it.
        let text = [0u8; 0x100];
        let image = CodeImage::new(
            0x40_0000,
            vec![CodeSection::new(0x40_1000, &text, true)],
            vec![],
        );
        let mut out = Vec::new();
        // jmp qword [rax*8+0x401080]
        instruction_relocations(
            &image,
            &decode_one(&[0xff, 0x24, 0xc5, 0x80, 0x10, 0x40, 0x00], 0x40_1000),
            &mut out,
        );
        // push 0x401010 (imm32 sign-extended to 64)
        instruction_relocations(
            &image,
            &decode_one(&[0x68, 0x10, 0x10, 0x40, 0x00], 0x40_1007),
            &mut out,
        );
        // mov eax, [rax+0x401010] has a base register: not a candidate.
        instruction_relocations(
            &image,
            &decode_one(&[0x8b, 0x80, 0x10, 0x10, 0x40, 0x00], 0x40_100c),
            &mut out,
        );
        let map = RelocationMap::from_entries(out);
        assert_eq!(
            map.entries(),
            &[
                Relocation {
                    site: 0x40_1000,
                    kind: RelocKind::AbsoluteDisplacement,
                    target: 0x40_1080,
                    region: Region::Code,
                },
                Relocation {
                    site: 0x40_1007,
                    kind: RelocKind::AbsoluteImmediate,
                    target: 0x40_1010,
                    region: Region::Code,
                },
            ]
        );
        assert!(map.entries().iter().all(|r| r.kind.is_possible()));
    }

    #[test]
    fn map_sorts_dedups_and_looks_up_by_site() {
        let r = |site, target| Relocation {
            site,
            kind: RelocKind::RipData,
            target,
            region: Region::Data,
        };
        let map = RelocationMap::from_entries(vec![r(3, 9), r(1, 5), r(3, 9), r(3, 7)]);
        assert_eq!(map.entries(), &[r(1, 5), r(3, 7), r(3, 9)]);
        assert_eq!(map.at(3), &[r(3, 7), r(3, 9)]);
        assert_eq!(map.at(1), &[r(1, 5)]);
        assert!(map.at(2).is_empty());
        assert!(map.at(4).is_empty());
    }
}
