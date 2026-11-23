//! MSVC x64 switch jump-table recovery.
//!
//! [`crate::discover`] hands over the instruction trace that ends in an unresolved `jmp reg`: the
//! jump's block preceded by its chain of sole predecessors, with the direction taken through
//! every conditional branch between them. A backward slice over that trace matches the dispatch
//! MSVC emits for dense x64 `switch` statements:
//!
//! ```text
//!     cmp    ecx, N                              ; guard, usually in the predecessor block
//!     ja     default                             ; (or jae / taken jbe / taken jb)
//!     movsxd rax, ecx                            ; optional zero/sign extension
//!     lea    rdx, [rip+__ImageBase]
//!   [ movzx  eax, byte ptr [rdx+rax+index_rva] ] ; two-level variant
//!     mov    ecx, dword ptr [rdx+rax*4+table_rva]
//!     add    rcx, rdx
//!     jmp    rcx
//! ```
//!
//! Register order, the choice of registers and interleaved unrelated instructions do not matter;
//! what is matched is the data flow. The number of entries read comes only from a proven bound:
//! an unsigned compare of the same register against an immediate guarding the path with no write
//! in between, a zero-extension of a byte/word, or, for the second level, the largest byte of a
//! bounded index table. Every target must be decodable code owned by the same `.pdata` function
//! as the jump (or, for code no `.pdata` range covers, by none). Anything else leaves the jump
//! unresolved; nothing is guessed.

use iced_x86::{
    FlowControl, Instruction, InstructionInfoFactory, Mnemonic, OpAccess, OpKind, Register,
};

use crate::cfg::{JumpTable, JumpTableIndex};
use crate::code_image::CodeImage;

/// Largest number of jump-table (or index-table) entries read for one dispatch.
pub const MAX_JUMP_TABLE_ENTRIES: u64 = 1 << 16;

/// Bound on nested slice steps (register copies, extensions) followed for one value.
const MAX_SLICE_DEPTH: u32 = 16;

/// One instruction of the trace. `taken` is set on a conditional branch that ends a predecessor
/// block: whether the path to the jump took it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Step<'b> {
    pub instr: &'b Instruction,
    pub taken: Option<bool>,
}

/// Symbolic value of a register at a trace position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Value {
    /// `__ImageBase`.
    ImageBase,
    /// A 32-bit table entry (zero-extended).
    Entry(Load),
    /// `__ImageBase + entry`: the dispatch target.
    Target(Load),
}

/// `[base + index*scale + disp]` table read at trace position `at`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Load {
    at: usize,
    table: u64,
    index: Register,
}

/// Proven inclusive upper bound of an index.
struct Bound {
    max: u64,
    index: Option<JumpTableIndex>,
}

/// Resolve the jump table dispatched by the `jmp reg` that ends `trace`.
pub(crate) fn resolve(image: &CodeImage<'_>, trace: &[Step<'_>]) -> Option<JumpTable> {
    let at = trace.len().checked_sub(1)?;
    let jmp = trace[at].instr;
    if jmp.flow_control() != FlowControl::IndirectBranch
        || jmp.op0_kind() != OpKind::Register
        || !jmp.op0_register().is_gpr64()
    {
        return None;
    }
    let mut slicer = Slicer {
        image,
        trace,
        info: InstructionInfoFactory::new(),
    };
    let Some(Value::Target(load)) = slicer.value(jmp.op0_register(), at, 0) else {
        return None;
    };
    let bound = slicer.bound(load.index, load.at, 64, 0)?;
    let count = bound.max.checked_add(1)?;
    if count > MAX_JUMP_TABLE_ENTRIES {
        return None;
    }
    let bytes = read(image, load.table, count * 4)?;
    let owner = image.function_table.function_at(jmp.ip());
    let mut targets = Vec::with_capacity(count as usize);
    for &entry in bytes.as_chunks::<4>().0 {
        let target = image
            .base()
            .checked_add(u64::from(u32::from_le_bytes(entry)))?;
        if image.code_bytes(target).is_err() || image.function_table.function_at(target) != owner {
            return None;
        }
        targets.push(target);
    }
    Some(JumpTable {
        site: jmp.ip(),
        load_site: trace[load.at].instr.ip(),
        table: load.table,
        targets,
        index: bound.index,
    })
}

/// `len` initialised bytes at `va` from a single section.
fn read<'a>(image: &CodeImage<'a>, va: u64, len: u64) -> Option<&'a [u8]> {
    let section = image.section_at(va)?;
    let start = usize::try_from(va - section.va).ok()?;
    let end = start.checked_add(usize::try_from(len).ok()?)?;
    section.data.get(start..end)
}

fn mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1 << bits) - 1
    }
}

/// `AH`/`CH`/`DH`/`BH` alias bits 8..16 of their full register.
fn is_high_byte(reg: Register) -> bool {
    matches!(
        reg,
        Register::AH | Register::CH | Register::DH | Register::BH
    )
}

struct Slicer<'t, 'b, 'i, 'a> {
    image: &'i CodeImage<'a>,
    trace: &'t [Step<'b>],
    info: InstructionInfoFactory,
}

impl Slicer<'_, '_, '_, '_> {
    /// Whether `instr` may write any part of the full register `reg`. Calls clobber everything.
    fn writes(&mut self, instr: &Instruction, reg: Register) -> bool {
        if matches!(
            instr.flow_control(),
            FlowControl::Call | FlowControl::IndirectCall
        ) {
            return true;
        }
        self.info.info(instr).used_registers().iter().any(|used| {
            used.register().full_register() == reg
                && matches!(
                    used.access(),
                    OpAccess::Write
                        | OpAccess::CondWrite
                        | OpAccess::ReadWrite
                        | OpAccess::ReadCondWrite
                )
        })
    }

    /// Position of the last write to `reg` before `before`.
    fn def(&mut self, reg: Register, before: usize) -> Option<usize> {
        let trace = self.trace;
        (0..before)
            .rev()
            .find(|&i| self.writes(trace[i].instr, reg))
    }

    /// Symbolic value of the full register `reg` just before position `before`.
    fn value(&mut self, reg: Register, before: usize, depth: u32) -> Option<Value> {
        if depth > MAX_SLICE_DEPTH {
            return None;
        }
        let at = self.def(reg.full_register(), before)?;
        let instr = self.trace[at].instr;
        let dst = instr.op0_register();
        if instr.op_count() != 2 || instr.op0_kind() != OpKind::Register {
            return None;
        }
        match (instr.mnemonic(), instr.op1_kind()) {
            (Mnemonic::Lea, OpKind::Memory)
                if dst.is_gpr64()
                    && instr.is_ip_rel_memory_operand()
                    && instr.ip_rel_memory_address() == self.image.base() =>
            {
                Some(Value::ImageBase)
            }
            (Mnemonic::Mov, OpKind::Register)
                if dst.is_gpr64() && instr.op1_register().is_gpr64() =>
            {
                self.value(instr.op1_register(), at, depth + 1)
            }
            (Mnemonic::Mov, OpKind::Memory) if dst.is_gpr32() => {
                self.table_load(at, 4, depth).map(Value::Entry)
            }
            (Mnemonic::Add, OpKind::Register)
                if dst.is_gpr64() && instr.op1_register().is_gpr64() =>
            {
                let lhs = self.value(dst, at, depth + 1)?;
                let rhs = self.value(instr.op1_register(), at, depth + 1)?;
                match (lhs, rhs) {
                    (Value::ImageBase, Value::Entry(load))
                    | (Value::Entry(load), Value::ImageBase) => Some(Value::Target(load)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Match an `element`-byte read of `[base + index*element + disp]` with
    /// `base == __ImageBase` in operand 1 of the instruction at `at`.
    fn table_load(&mut self, at: usize, element: u32, depth: u32) -> Option<Load> {
        let instr = self.trace[at].instr;
        let (base, index) = (instr.memory_base(), instr.memory_index());
        let segment = instr.memory_segment();
        if instr.op1_kind() != OpKind::Memory
            || instr.memory_size().size() != element as usize
            || instr.memory_index_scale() != element
            || !base.is_gpr64()
            || !index.is_gpr64()
            || segment == Register::FS
            || segment == Register::GS
        {
            return None;
        }
        if self.value(base, at, depth + 1)? != Value::ImageBase {
            return None;
        }
        let rva = instr.memory_displacement64();
        if rva >= self.image.end() - self.image.base() {
            return None;
        }
        Some(Load {
            at,
            table: self.image.base() + rva,
            index,
        })
    }

    /// Proven inclusive bound of the low `width` bits of the full register `reg` just before
    /// position `before` (`width == 64`: of the whole register).
    fn bound(&mut self, reg: Register, before: usize, width: u32, depth: u32) -> Option<Bound> {
        if depth > MAX_SLICE_DEPTH {
            return None;
        }
        let reg = reg.full_register();
        for i in (0..before).rev() {
            let step = self.trace[i];
            if self.writes(step.instr, reg) {
                return self.def_bound(i, width, depth);
            }
            if let Some(taken) = step.taken
                && let Some(max) = self.guard(i, taken, reg, width)
            {
                return Some(Bound { max, index: None });
            }
        }
        None
    }

    /// Bound of the register written by the instruction at `at`.
    fn def_bound(&mut self, at: usize, width: u32, depth: u32) -> Option<Bound> {
        let instr = self.trace[at].instr;
        let dst = instr.op0_register();
        let src = instr.op1_register();
        let src_is_reg = instr.op_count() == 2 && instr.op1_kind() == OpKind::Register;
        let sign_extended = |bound: Bound| {
            // Sign extension preserves an unsigned bound only below 2^31.
            (width <= 32 || bound.max < 0x8000_0000).then_some(bound)
        };
        match instr.mnemonic() {
            Mnemonic::Mov if src_is_reg && dst.is_gpr32() && src.is_gpr32() => {
                self.bound(src, at, 32, depth + 1)
            }
            Mnemonic::Mov if src_is_reg && dst.is_gpr64() && src.is_gpr64() => {
                self.bound(src, at, width, depth + 1)
            }
            Mnemonic::Movsxd if src_is_reg && dst.is_gpr64() && src.is_gpr32() => {
                self.bound(src, at, 32, depth + 1).and_then(sign_extended)
            }
            Mnemonic::Cdqe => self
                .bound(Register::RAX, at, 32, depth + 1)
                .and_then(sign_extended),
            Mnemonic::Movzx if src_is_reg && (src.is_gpr8() || src.is_gpr16()) => {
                let bits = src.size() as u32 * 8;
                let deeper = if is_high_byte(src) {
                    None
                } else {
                    self.bound(src, at, bits, depth + 1)
                };
                let max = deeper.map_or(mask(bits), |b| b.max.min(mask(bits)));
                Some(Bound { max, index: None })
            }
            Mnemonic::Movzx
                if instr.op1_kind() == OpKind::Memory
                    && instr.memory_size().size() == 1
                    && (dst.is_gpr32() || dst.is_gpr64()) =>
            {
                self.index_table(at, depth)
            }
            _ => None,
        }
    }

    /// Two-level dispatch: `movzx r, byte [__ImageBase + index + index_rva]` at `at`.
    fn index_table(&mut self, at: usize, depth: u32) -> Option<Bound> {
        let load = self.table_load(at, 1, depth)?;
        let outer = self.bound(load.index, at, 64, depth + 1)?;
        if outer.index.is_some() {
            return None;
        }
        let len = outer.max.checked_add(1)?;
        if len > MAX_JUMP_TABLE_ENTRIES {
            return None;
        }
        let max = read(self.image, load.table, len)?.iter().copied().max()?;
        Some(Bound {
            max: u64::from(max),
            index: Some(JumpTableIndex {
                load_site: self.trace[at].instr.ip(),
                table: load.table,
                len,
            }),
        })
    }

    /// Bound on `reg` implied by the conditional branch at `at` (taken or not, per `taken`):
    /// the last flag write before it must be `cmp reg, imm` with no write to `reg` in between.
    fn guard(&mut self, at: usize, taken: bool, reg: Register, width: u32) -> Option<u64> {
        let jcc = self.trace[at].instr.mnemonic();
        for i in (0..at).rev() {
            let instr = self.trace[i].instr;
            if self.writes(instr, reg) {
                return None;
            }
            if instr.rflags_modified() == 0 {
                continue;
            }
            let cmp_reg = instr.op0_register();
            if instr.mnemonic() != Mnemonic::Cmp
                || instr.op0_kind() != OpKind::Register
                || cmp_reg.full_register() != reg
                || is_high_byte(cmp_reg)
            {
                return None;
            }
            let bits = cmp_reg.size() as u32 * 8;
            if bits < width {
                return None;
            }
            let imm = instr.try_immediate(1).ok()? & mask(bits);
            let max = match (jcc, taken) {
                (Mnemonic::Ja, false) | (Mnemonic::Jbe, true) => imm,
                (Mnemonic::Jae, false) | (Mnemonic::Jb, true) => imm.checked_sub(1)?,
                _ => return None,
            };
            return Some(max.min(mask(width)));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::cfg::{Edge, Function, JumpTable, JumpTableIndex, Terminator};
    use crate::code_image::{CodeImage, CodeSection, Region};
    use crate::discover::{Analysis, discover};
    use crate::pdata::{FunctionRange, FunctionTable};
    use crate::reloc::{RelocKind, Relocation};

    const BASE: u64 = 0x1_4000_0000;
    const TEXT_RVA: u32 = 0x1000;
    const DATA_RVA: u32 = 0x2000;
    const TEXT: u64 = BASE + TEXT_RVA as u64;
    const DATA: u64 = BASE + DATA_RVA as u64;
    /// Case blocks of [`switch_code`] (offsets); the last one is also the default.
    const CASES: [u32; 4] = [0x1b, 0x21, 0x27, 0x2d];

    /// `lea rdx, [rip+x]` at `rva` addressing `__ImageBase`.
    fn lea_rdx_image_base(rva: u32) -> Vec<u8> {
        let disp = -i64::from(rva + 7) as i32;
        [&[0x48, 0x8d, 0x15][..], &disp.to_le_bytes()].concat()
    }

    /// MSVC-style `switch (ecx)` with three cases and a default at `text_rva`, reading a 32-bit
    /// RVA table at `table_rva`.
    fn switch_code(text_rva: u32, table_rva: u32) -> Vec<u8> {
        [
            &[0x83, 0xf9, 0x03][..],           // 0: cmp ecx, 3
            &[0x77, 0x28],                     // 3: ja 0x2d
            &[0x48, 0x63, 0xc1],               // 5: movsxd rax, ecx
            &lea_rdx_image_base(text_rva + 8), // 8: lea rdx, [__ImageBase]
            &[0x8b, 0x8c, 0x82],
            &table_rva.to_le_bytes(), // f: mov ecx, [rdx+rax*4+table_rva]
            &[0x48, 0x01, 0xd1],      // 16: add rcx, rdx
            &[0xff, 0xe1],            // 19: jmp rcx
            &[0xb8, 0x00, 0x00, 0x00, 0x00, 0xc3], // 1b: mov eax, 0; ret
            &[0xb8, 0x01, 0x00, 0x00, 0x00, 0xc3], // 21: mov eax, 1; ret
            &[0xb8, 0x02, 0x00, 0x00, 0x00, 0xc3], // 27: mov eax, 2; ret
            &[0x31, 0xc0, 0xc3],      // 2d: default: xor eax, eax; ret
        ]
        .concat()
    }

    fn rva_table(rvas: impl IntoIterator<Item = u32>) -> Vec<u8> {
        rvas.into_iter().flat_map(u32::to_le_bytes).collect()
    }

    fn case_table(text_rva: u32) -> Vec<u8> {
        rva_table(CASES.map(|c| text_rva + c))
    }

    fn analyze(code: &[u8], data: &[u8], entries: &[u32], ranges: &[(u32, u32, u32)]) -> Analysis {
        let mut image = CodeImage::new(
            BASE,
            vec![
                CodeSection::new(TEXT, code, true),
                CodeSection::new(DATA, data, false),
            ],
            entries.iter().map(|&e| TEXT + u64::from(e)).collect(),
        );
        image.function_table = FunctionTable::new(
            ranges
                .iter()
                .map(|&(start, end, function)| FunctionRange {
                    start: TEXT + u64::from(start),
                    end: TEXT + u64::from(end),
                    function: TEXT + u64::from(function),
                })
                .collect(),
        );
        discover(&image)
    }

    /// Successors of the block of `function` that ends in the jump at `jmp`.
    fn jump_successors(function: &Function, jmp: u64) -> &[Edge] {
        let block = function
            .blocks
            .values()
            .find(|b| b.instructions.last().map(|i| i.ip()) == Some(jmp))
            .expect("block ending in the jump");
        assert_eq!(block.terminator, Terminator::IndirectJump);
        &block.successors
    }

    fn assert_unresolved(analysis: &Analysis, jmp: u32) {
        assert!(analysis.jump_tables.is_empty());
        assert_eq!(
            jump_successors(&analysis.functions[&TEXT], TEXT + u64::from(jmp)),
            [Edge::IndirectJump { slot: None }]
        );
    }

    #[test]
    fn resolves_rva_table_switch() {
        let code = switch_code(TEXT_RVA, DATA_RVA);
        let data = case_table(TEXT_RVA);
        let targets = CASES.map(|c| TEXT + u64::from(c));
        for ranges in [&[][..], &[(0, 0x30, 0)]] {
            let analysis = analyze(&code, &data, &[0], ranges);
            assert_eq!(
                analysis.jump_tables.values().collect::<Vec<_>>(),
                [&JumpTable {
                    site: TEXT + 0x19,
                    load_site: TEXT + 0xf,
                    table: DATA,
                    targets: targets.to_vec(),
                    index: None,
                }]
            );
            assert_eq!(analysis.functions.len(), 1);
            let f = &analysis.functions[&TEXT];
            assert_eq!(
                jump_successors(f, TEXT + 0x19),
                targets.map(Edge::JumpTable)
            );
            assert_eq!(
                f.blocks.keys().map(|va| va - TEXT).collect::<Vec<_>>(),
                [0x0, 0x5, 0x1b, 0x21, 0x27, 0x2d]
            );
            let table_relocs: Vec<_> = analysis
                .relocations
                .entries()
                .iter()
                .filter(|r| matches!(r.kind, RelocKind::JumpTable | RelocKind::JumpTableEntry))
                .map(|r| (r.site, r.kind, r.target, r.region))
                .collect();
            assert_eq!(
                table_relocs,
                [
                    (TEXT + 0xf, RelocKind::JumpTable, DATA, Region::Data),
                    (DATA, RelocKind::JumpTableEntry, targets[0], Region::Code),
                    (
                        DATA + 4,
                        RelocKind::JumpTableEntry,
                        targets[1],
                        Region::Code
                    ),
                    (
                        DATA + 8,
                        RelocKind::JumpTableEntry,
                        targets[2],
                        Region::Code
                    ),
                    (
                        DATA + 12,
                        RelocKind::JumpTableEntry,
                        targets[3],
                        Region::Code
                    ),
                ]
            );
        }
    }

    #[test]
    fn resolves_two_level_byte_index_switch() {
        let code = [
            &[0x83, 0xf9, 0x07][..],           // 0: cmp ecx, 7
            &[0x77, 0x2a],                     // 3: ja 0x2f
            &[0x48, 0x63, 0xc1],               // 5: movsxd rax, ecx
            &lea_rdx_image_base(TEXT_RVA + 8), // 8: lea rdx, [__ImageBase]
            &[0x0f, 0xb6, 0x84, 0x02],
            &(DATA_RVA + 0x20).to_le_bytes(), // f: movzx eax, byte [rdx+rax+index_rva]
            &[0x8b, 0x8c, 0x82],
            &DATA_RVA.to_le_bytes(), // 17: mov ecx, [rdx+rax*4+table_rva]
            &[0x48, 0x01, 0xd1],     // 1e: add rcx, rdx
            &[0xff, 0xe1],           // 21: jmp rcx
            &[0xb8, 0x0a, 0x00, 0x00, 0x00, 0xc3], // 23: mov eax, 10; ret
            &[0xb8, 0x0b, 0x00, 0x00, 0x00, 0xc3], // 29: mov eax, 11; ret
            &[0x31, 0xc0, 0xc3],     // 2f: default
        ]
        .concat();
        let mut data = rva_table([0x1023, 0x1029, 0x102f]);
        data.resize(0x20, 0xcc); // bytes past the proven 3 entries are never read
        data.extend_from_slice(&[0, 1, 0, 2, 2, 1, 0, 2]);
        data.extend_from_slice(&[0xff; 8]); // index bytes past the bound are never read

        let analysis = analyze(&code, &data, &[0], &[(0, 0x32, 0)]);
        let targets = [0x23, 0x29, 0x2f].map(|c| TEXT + c);
        assert_eq!(
            analysis.jump_tables[&(TEXT + 0x21)],
            JumpTable {
                site: TEXT + 0x21,
                load_site: TEXT + 0x17,
                table: DATA,
                targets: targets.to_vec(),
                index: Some(JumpTableIndex {
                    load_site: TEXT + 0xf,
                    table: DATA + 0x20,
                    len: 8,
                }),
            }
        );
        assert_eq!(
            jump_successors(&analysis.functions[&TEXT], TEXT + 0x21),
            targets.map(Edge::JumpTable)
        );
        assert_eq!(
            analysis.relocations.at(TEXT + 0xf),
            &[Relocation {
                site: TEXT + 0xf,
                kind: RelocKind::JumpTableIndex,
                target: DATA + 0x20,
                region: Region::Data,
            }]
        );
    }

    #[test]
    fn rejects_targets_outside_the_function() {
        let code = switch_code(TEXT_RVA, DATA_RVA);
        let data = case_table(TEXT_RVA);
        // The default case belongs to another .pdata function.
        let analysis = analyze(&code, &data, &[0], &[(0, 0x2d, 0), (0x2d, 0x30, 0x2d)]);
        assert_unresolved(&analysis, 0x19);

        // An entry pointing into a data section.
        let mut data = case_table(TEXT_RVA);
        data[4..8].copy_from_slice(&(DATA_RVA + 0x10).to_le_bytes());
        data.resize(0x20, 0);
        assert_unresolved(&analyze(&code, &data, &[0], &[]), 0x19);

        // Table past the end of its section's bytes.
        assert_unresolved(&analyze(&code, &data[..12], &[0], &[]), 0x19);
    }

    #[test]
    fn unprovable_bounds_stay_unresolved() {
        let data = case_table(TEXT_RVA);
        let base_code = switch_code(TEXT_RVA, DATA_RVA);

        // No guard at all.
        let mut code = base_code.clone();
        code[0..5].copy_from_slice(&[0x0f, 0x1f, 0x44, 0x00, 0x00]); // 5-byte nop
        assert_unresolved(&analyze(&code, &data, &[0], &[]), 0x19);

        // The branch bounds the wrong side (`jb default`).
        let mut code = base_code.clone();
        code[3] = 0x72;
        assert_unresolved(&analyze(&code, &data, &[0], &[]), 0x19);

        // 32-bit compare, but the full 64-bit register indexes the table (upper half unknown).
        let mut code = base_code.clone();
        code[5..8].copy_from_slice(&[0x0f, 0x1f, 0x00]); // 3-byte nop instead of movsxd
        code[0x11] = 0x8a; // mov ecx, [rdx+rcx*4+table]
        assert_unresolved(&analyze(&code, &data, &[0], &[]), 0x19);

        // The index comes from a register the guard did not check.
        let mut code = base_code.clone();
        code[5..8].copy_from_slice(&[0x48, 0x63, 0xc2]); // movsxd rax, edx
        assert_unresolved(&analyze(&code, &data, &[0], &[]), 0x19);

        // The table base is not __ImageBase.
        let mut code = base_code;
        code[0x0b..0x0f].copy_from_slice(&0i32.to_le_bytes()); // lea rdx, [rip]
        assert_unresolved(&analyze(&code, &data, &[0], &[]), 0x19);
    }

    #[test]
    fn resolution_is_demoted_when_a_case_jumps_back_into_the_dispatch() {
        // Case 1 decrements the index and re-enters the dispatch without a guard, so the bound
        // proven in the first round no longer holds once the case code is decoded.
        let mut code = switch_code(TEXT_RVA, DATA_RVA);
        code[0x21..0x27].copy_from_slice(&[0xff, 0xc9, 0xeb, 0xe0, 0x90, 0x90]); // dec ecx; jmp 0x5
        let analysis = analyze(&code, &case_table(TEXT_RVA), &[0], &[]);
        assert_unresolved(&analysis, 0x19);
        assert!(
            analysis
                .relocations
                .entries()
                .iter()
                .all(|r| !matches!(r.kind, RelocKind::JumpTable | RelocKind::JumpTableEntry))
        );
    }

    #[test]
    fn nested_switch_found_through_a_case_and_order_independent() {
        // The outer switch's case 0 is a second switch placed after the outer one; it is only
        // reachable through the outer table.
        let outer = switch_code(TEXT_RVA, DATA_RVA);
        let inner = switch_code(TEXT_RVA + 0x30, DATA_RVA + 0x10);
        let code = [outer, inner].concat();
        let mut data = rva_table([0x1030, 0x1021, 0x1027, 0x102d]);
        data.extend(case_table(TEXT_RVA + 0x30));

        let forward = analyze(&code, &data, &[0], &[(0, 0x60, 0)]);
        assert_eq!(
            forward.jump_tables.keys().copied().collect::<Vec<_>>(),
            [TEXT + 0x19, TEXT + 0x49]
        );
        assert_eq!(
            forward.jump_tables[&(TEXT + 0x49)].targets,
            CASES.map(|c| TEXT + 0x30 + u64::from(c))
        );
        let again = analyze(&code, &data, &[0], &[(0, 0x60, 0)]);
        assert_eq!(forward, again);
        // Seeding the inner switch directly resolves the same tables.
        let with_inner_entry = analyze(&code, &data, &[0x30, 0], &[(0, 0x60, 0)]);
        assert_eq!(forward.jump_tables, with_inner_entry.jump_tables);
        assert_eq!(forward.relocations, with_inner_entry.relocations);
    }

    #[test]
    fn pdata_only_function_with_switch_in_synthetic_pe() {
        use crate::pdata::ExceptionDirectory;
        use crate::pdata::tests::{pe_with_pdata, put, unwind_bytes};
        use crate::pe::Pe64;
        use crate::summary::render_summary_json;

        let mut file = pe_with_pdata(&[(0x1100, 0x1130, 0x2700)]);
        put(&mut file, 0x2700, &unwind_bytes(0, &[]));
        put(&mut file, 0x1100, &switch_code(0x1100, 0x2600));
        put(&mut file, 0x2600, &case_table(0x1100));
        let pe = Pe64::parse(&file).expect("synthetic PE parses");
        let mut image = CodeImage::from_pe(&pe, &file);
        assert!(discover(&image).function(BASE + 0x1100).is_none());

        let pdata = ExceptionDirectory::from_pe(&pe, &file);
        assert!(pdata.diagnostics.is_empty());
        pdata.seed(&mut image);
        let analysis = discover(&image);
        let f = analysis
            .function(BASE + 0x1100)
            .expect("found through .pdata only");
        assert_eq!(
            jump_successors(f, BASE + 0x1119),
            CASES.map(|c| Edge::JumpTable(BASE + 0x1100 + u64::from(c)))
        );
        let json = render_summary_json(&image, &analysis);
        assert!(json.contains("\"pdata\": {\"functions\": 1, \"ranges\": 1},"));
        assert!(json.contains("\"jump_tables\": {\"resolved\": 1, \"unresolved\": 0},"));
        assert!(json.contains("\"jump_table_entry\": 4"));
    }
}
