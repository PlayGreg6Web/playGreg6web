//! Static pointer-slot lookup for tail calls through a loaded pointer.
//!
//! `mov reg, [rip+slot]` (then optionally register copies, a null check, ...) followed by
//! `jmp reg` is a tail call through a global function pointer. When the slot is a relocated
//! 64-bit pointer in a data section (listed in [`CodeImage::data_relocs`]) whose initial value is a
//! code address, that address is a real function entry and can be lifted ahead of time. The jump
//! itself stays on run-time dispatch: the slot may be rewritten later. Nothing is invented: a slot
//! without a base relocation, or one whose value is not in an executable section, yields nothing.

use std::collections::{BTreeMap, BTreeSet};

use iced_x86::{
    FlowControl, Instruction, InstructionInfoFactory, Mnemonic, OpAccess, OpKind, Register,
};

use crate::cfg::{Edge, Function, Terminator};
use crate::code_image::{CodeImage, Region};
use crate::discover::Analysis;

/// Backward steps (register copies followed) before giving up.
const MAX_COPIES: u32 = 4;

/// `(slot, initial code target)` for the indirect `jmp` ending `trace`, if it jumps through a
/// register loaded from a relocated data slot that statically holds a code address.
pub(crate) fn static_target(image: &CodeImage<'_>, trace: &[&Instruction]) -> Option<(u64, u64)> {
    slot_target(image, loaded_slot(trace)?)
}

/// The RIP-relative 8-byte slot whose value the indirect `jmp reg` ending `trace` jumps to
/// (`mov reg, [rip+slot]`, then only register copies of it), if any.
fn loaded_slot(trace: &[&Instruction]) -> Option<u64> {
    let jmp = *trace.last()?;
    if jmp.flow_control() != FlowControl::IndirectBranch
        || jmp.op0_kind() != OpKind::Register
        || !jmp.op0_register().is_gpr64()
    {
        return None;
    }
    let mut info = InstructionInfoFactory::new();
    let mut reg = jmp.op0_register().full_register();
    let mut before = trace.len() - 1;
    for _ in 0..=MAX_COPIES {
        let def = (0..before).rev().find(|&i| {
            let instr = trace[i];
            matches!(
                instr.flow_control(),
                FlowControl::Call | FlowControl::IndirectCall
            ) || info.info(instr).used_registers().iter().any(|u| {
                u.register().full_register() == reg
                    && matches!(
                        u.access(),
                        OpAccess::Write
                            | OpAccess::ReadWrite
                            | OpAccess::CondWrite
                            | OpAccess::ReadCondWrite
                    )
            })
        })?;
        let instr = trace[def];
        if instr.mnemonic() != Mnemonic::Mov || instr.op_count() != 2 {
            return None;
        }
        match instr.op1_kind() {
            OpKind::Register
                if instr.op0_register().is_gpr64() && instr.op1_register().is_gpr64() =>
            {
                reg = instr.op1_register().full_register();
                before = def;
            }
            OpKind::Memory
                if instr.is_ip_rel_memory_operand()
                    && instr.memory_size().size() == 8
                    && instr.op0_register().is_gpr64() =>
            {
                return Some(instr.ip_rel_memory_address());
            }
            _ => return None,
        }
    }
    None
}

/// Initial code target of the relocated pointer slot at `slot`.
fn slot_target(image: &CodeImage<'_>, slot: u64) -> Option<(u64, u64)> {
    if !image.data_relocs.contains(&slot) || image.region_of(slot) != Region::Data {
        return None;
    }
    let bytes = image.bytes_at(slot, 8)?;
    let target = u64::from_le_bytes(bytes.try_into().ok()?);
    (image.region_of(target) == Region::Code && image.code_bytes(target).is_ok())
        .then_some((slot, target))
}

/// Most known targets a slot may have for its jumps to get direct tail-call arms.
const MAX_ARMS: usize = 8;

/// Bytes below a slot in which an address-taking `lea` counts as the slot escaping (a field of
/// a global object whose address is passed on).
const ESCAPE_BELOW: u64 = 0x40;

/// What every instruction in the analysed code stores to one pointer slot.
#[derive(Debug, Default)]
struct SlotStores {
    /// Code addresses stored to (or statically held by) the slot.
    targets: BTreeSet<u64>,
    /// A store of an unknown value, a partial or read-modify-write access, or an address-taking
    /// `lea` at or just below the slot: the slot may hold something else.
    unknown: bool,
}

/// Tail calls through a pointer slot (`mov reg, [rip+slot]` ... `jmp reg`, backlog 43) whose
/// stores are visible: every `mov qword [rip+slot], reg` in the analysed code whose `reg` comes
/// from `lea reg, [rip+code]` in the same block adds a known target, as does a relocated
/// initial code address. Each known target that is a function entry becomes an
/// [`Edge::TailCall`] of the jump's block, which the lifter turns into a direct tail call with
/// the run-time dispatcher as the fallback for any other value. When the known targets are all
/// the slot can hold (no other store, no address-taking `lea` near it, a zero or relocated code
/// initial value) the jump leaves [`Analysis::unresolved_jumps`]; otherwise it keeps its reason.
pub(crate) fn resolve_slot_stores(image: &CodeImage<'_>, analysis: &mut Analysis) {
    let mut sites: BTreeSet<u64> = analysis.unresolved_jumps.keys().copied().collect();
    sites.extend(analysis.pointer_slots.keys().copied());
    sites.retain(|site| !analysis.jump_tables.contains_key(site));
    if sites.is_empty() {
        return;
    }
    // Every copy `(function, block)` of the block each site ends.
    let mut homes: BTreeMap<u64, Vec<(u64, u64)>> = BTreeMap::new();
    for (&entry, function) in &analysis.functions {
        for (&start, block) in &function.blocks {
            if block.terminator == Terminator::IndirectJump
                && let Some(site) = block.instructions.last().map(Instruction::ip)
                && sites.contains(&site)
            {
                homes.entry(site).or_default().push((entry, start));
            }
        }
    }
    let mut site_slot: BTreeMap<u64, u64> = BTreeMap::new();
    for (&site, copies) in &homes {
        let (entry, start) = copies[0];
        let function = &analysis.functions[&entry];
        if let Some(slot) = loaded_slot(&trace_into(function, start))
            && image.region_of(slot) == Region::Data
        {
            site_slot.insert(site, slot);
        }
    }
    if site_slot.is_empty() {
        return;
    }
    let slots: BTreeSet<u64> = site_slot.values().copied().collect();
    let stores = scan_stores(image, analysis, &slots);
    for (site, slot) in site_slot {
        let Some(info) = stores.get(&slot) else {
            continue;
        };
        let arms: Vec<u64> = info
            .targets
            .iter()
            .copied()
            .filter(|t| analysis.functions.contains_key(t))
            .collect();
        if arms.is_empty() || arms.len() > MAX_ARMS {
            continue;
        }
        for &(entry, start) in &homes[&site] {
            if let Some(block) = analysis
                .functions
                .get_mut(&entry)
                .and_then(|f| f.blocks.get_mut(&start))
            {
                for &t in &arms {
                    if !block.successors.contains(&Edge::TailCall(t)) {
                        block.successors.push(Edge::TailCall(t));
                    }
                }
            }
        }
        if !info.unknown && arms.len() == info.targets.len() {
            analysis.unresolved_jumps.remove(&site);
        }
    }
}

/// `(closed, open)`: indirect `jmp`s outside a resolved jump table with known tail-call targets
/// from [`resolve_slot_stores`], closed ones having left [`Analysis::unresolved_jumps`].
pub(crate) fn slot_store_counts(analysis: &Analysis) -> (usize, usize) {
    let mut seen = BTreeSet::new();
    let (mut closed, mut open) = (0, 0);
    for function in analysis.functions.values() {
        for block in function.blocks.values() {
            if block.terminator == Terminator::IndirectJump
                && block
                    .successors
                    .iter()
                    .any(|e| matches!(e, Edge::TailCall(_)))
                && let Some(site) = block.instructions.last().map(Instruction::ip)
                && !analysis.jump_tables.contains_key(&site)
                && seen.insert(site)
            {
                if analysis.unresolved_jumps.contains_key(&site) {
                    open += 1;
                } else {
                    closed += 1;
                }
            }
        }
    }
    (closed, open)
}

/// The instructions of the block at `start` of `function`, preceded by those of its sole
/// predecessors (up to [`MAX_COPIES`] blocks back).
fn trace_into(function: &Function, start: u64) -> Vec<&Instruction> {
    let mut chain = vec![start];
    let mut at = start;
    for _ in 0..MAX_COPIES {
        let mut preds = function.blocks.values().filter(|b| {
            b.successors.iter().any(
                |e| matches!(*e, Edge::Fallthrough(t) | Edge::Taken(t) | Edge::Jump(t) if t == at),
            )
        });
        let (Some(pred), None) = (preds.next(), preds.next()) else {
            break;
        };
        if chain.contains(&pred.start) {
            break;
        }
        chain.push(pred.start);
        at = pred.start;
    }
    chain
        .iter()
        .rev()
        .flat_map(|s| function.blocks[s].instructions.iter())
        .collect()
}

/// What the analysed code (each block once) and the image's initial values store to `slots`.
fn scan_stores(
    image: &CodeImage<'_>,
    analysis: &Analysis,
    slots: &BTreeSet<u64>,
) -> BTreeMap<u64, SlotStores> {
    let mut out: BTreeMap<u64, SlotStores> = slots
        .iter()
        .map(|&slot| (slot, initial_value(image, slot)))
        .collect();
    let mut seen = BTreeSet::new();
    let mut info = InstructionInfoFactory::new();
    for function in analysis.functions.values() {
        for block in function.blocks.values() {
            if !seen.insert(block.start) {
                continue;
            }
            for (index, instr) in block.instructions.iter().enumerate() {
                if !instr.is_ip_rel_memory_operand() {
                    continue;
                }
                let addr = instr.ip_rel_memory_address();
                let low = addr.saturating_sub(7);
                let high = addr.saturating_add(ESCAPE_BELOW);
                for &slot in slots.range(low..=high) {
                    let access =
                        classify(image, &block.instructions[..index], instr, slot, &mut info);
                    let entry = out.get_mut(&slot).expect("every slot has an entry");
                    match access {
                        Access::None => {}
                        Access::Store(target) => {
                            entry.targets.insert(target);
                        }
                        Access::Unknown => entry.unknown = true,
                    }
                }
            }
        }
    }
    out
}

/// The slot's initial value: nothing for zero (or zero-fill), a relocated code address as a
/// known target, anything else unknown.
fn initial_value(image: &CodeImage<'_>, slot: u64) -> SlotStores {
    let mut stores = SlotStores::default();
    let value = image
        .bytes_at(slot, 8)
        .and_then(|b| b.try_into().ok())
        .map_or(0, u64::from_le_bytes);
    if image.data_relocs.contains(&slot) {
        match slot_target(image, slot) {
            Some((_, target)) => {
                stores.targets.insert(target);
            }
            None => stores.unknown = true,
        }
    } else if value != 0 {
        stores.unknown = true;
    }
    stores
}

/// How one instruction touches an 8-byte slot.
enum Access {
    /// Reads it, or touches a neighbour only.
    None,
    /// Stores this code address to it.
    Store(u64),
    /// Anything that may leave another value in it.
    Unknown,
}

/// How `instr` (preceded in its block by `before`) touches the 8-byte slot at `slot`.
fn classify(
    image: &CodeImage<'_>,
    before: &[Instruction],
    instr: &Instruction,
    slot: u64,
    info: &mut InstructionInfoFactory,
) -> Access {
    let addr = instr.ip_rel_memory_address();
    if instr.mnemonic() == Mnemonic::Lea {
        return if addr <= slot && slot - addr <= ESCAPE_BELOW {
            Access::Unknown
        } else {
            Access::None
        };
    }
    let size = instr.memory_size().size() as u64;
    let overlaps = addr < slot.saturating_add(8) && addr.saturating_add(size.max(1)) > slot;
    if !overlaps {
        return Access::None;
    }
    if addr == slot && size == 8 {
        match (instr.mnemonic(), instr.op0_kind(), instr.op1_kind()) {
            (Mnemonic::Mov, OpKind::Register, OpKind::Memory)
            | (Mnemonic::Cmp | Mnemonic::Test, OpKind::Memory, _) => return Access::None,
            (Mnemonic::Mov, OpKind::Memory, OpKind::Immediate32to64) => {
                return if instr.immediate32to64() == 0 {
                    Access::None
                } else {
                    Access::Unknown
                };
            }
            (Mnemonic::Mov, OpKind::Memory, OpKind::Register)
                if instr.op1_register().is_gpr64() =>
            {
                return stored_code(image, before, instr.op1_register(), info);
            }
            _ => {}
        }
    }
    if let Some(access) = vector_store(image, before, instr, slot, info) {
        return access;
    }
    // Only the RIP-relative operand counts (a call's return-address push is not the slot).
    let read_only = info
        .info(instr)
        .used_memory()
        .iter()
        .filter(|m| m.base() == Register::RIP)
        .all(|m| {
            matches!(
                m.access(),
                OpAccess::Read | OpAccess::CondRead | OpAccess::NoMemAccess
            )
        });
    if read_only {
        Access::None
    } else {
        Access::Unknown
    }
}

/// Whether `mnemonic` is a plain whole-register vector move.
fn is_vector_move(mnemonic: Mnemonic) -> bool {
    matches!(
        mnemonic,
        Mnemonic::Movups
            | Mnemonic::Movaps
            | Mnemonic::Movdqu
            | Mnemonic::Movdqa
            | Mnemonic::Vmovups
            | Mnemonic::Vmovaps
            | Mnemonic::Vmovdqu
            | Mnemonic::Vmovdqa
    )
}

/// A vector store over the slot (a global object copied from a constant, e.g. a callable's
/// `{invoker, manager}` pair): the 8 bytes it leaves in the slot when its register was loaded
/// from initialised data, or zeroed, in the same block. `None` for any other instruction.
fn vector_store(
    image: &CodeImage<'_>,
    before: &[Instruction],
    instr: &Instruction,
    slot: u64,
    info: &mut InstructionInfoFactory,
) -> Option<Access> {
    if !is_vector_move(instr.mnemonic())
        || instr.op0_kind() != OpKind::Memory
        || instr.op1_kind() != OpKind::Register
    {
        return None;
    }
    let offset = slot.checked_sub(instr.ip_rel_memory_address())?;
    let size = instr.memory_size().size() as u64;
    if offset + 8 > size {
        return Some(Access::Unknown);
    }
    let reg = instr.op1_register().full_register();
    let def = before.iter().rev().find(|i| {
        matches!(
            i.flow_control(),
            FlowControl::Call | FlowControl::IndirectCall
        ) || info.info(i).used_registers().iter().any(|u| {
            u.register().full_register() == reg
                && !matches!(u.access(), OpAccess::Read | OpAccess::CondRead)
        })
    });
    let Some(def) = def else {
        return Some(Access::Unknown);
    };
    let zeroing = matches!(
        def.mnemonic(),
        Mnemonic::Xorps | Mnemonic::Vxorps | Mnemonic::Pxor | Mnemonic::Vpxor
    ) && (1..def.op_count())
        .all(|i| def.op_kind(i) == OpKind::Register && def.op_register(i).full_register() == reg);
    if zeroing {
        return Some(Access::None);
    }
    if !is_vector_move(def.mnemonic())
        || def.op1_kind() != OpKind::Memory
        || !def.is_ip_rel_memory_operand()
        || (def.memory_size().size() as u64) < size
    {
        return Some(Access::Unknown);
    }
    let src = def.ip_rel_memory_address() + offset;
    if image.region_of(src) != Region::Data {
        return Some(Access::Unknown);
    }
    if image.data_relocs.contains(&src) {
        return Some(slot_target(image, src).map_or(Access::Unknown, |(_, t)| Access::Store(t)));
    }
    let value = image
        .bytes_at(src, 8)
        .and_then(|b| b.try_into().ok())
        .map(u64::from_le_bytes);
    Some(if value == Some(0) {
        Access::None
    } else {
        Access::Unknown
    })
}

/// What storing `reg` (as it is at the end of `before`) leaves in a slot: its last write there
/// (no call in between) is `lea reg, [rip+code]` (that code address) or `xor reg, reg` (null).
fn stored_code(
    image: &CodeImage<'_>,
    before: &[Instruction],
    reg: Register,
    info: &mut InstructionInfoFactory,
) -> Access {
    let reg = reg.full_register();
    let def = before.iter().rev().find(|instr| {
        matches!(
            instr.flow_control(),
            FlowControl::Call | FlowControl::IndirectCall
        ) || info.info(instr).used_registers().iter().any(|u| {
            u.register().full_register() == reg
                && !matches!(u.access(), OpAccess::Read | OpAccess::CondRead)
        })
    });
    let Some(def) = def else {
        return Access::Unknown;
    };
    if def.mnemonic() == Mnemonic::Xor
        && def.op0_kind() == OpKind::Register
        && def.op1_kind() == OpKind::Register
        && def.op0_register().full_register() == reg
        && def.op1_register().full_register() == reg
    {
        return Access::None;
    }
    lea_code(image, def).map_or(Access::Unknown, Access::Store)
}

/// The code address `def` (`lea reg, [rip+code]`) loads.
fn lea_code(image: &CodeImage<'_>, def: &Instruction) -> Option<u64> {
    if def.mnemonic() != Mnemonic::Lea || !def.is_ip_rel_memory_operand() {
        return None;
    }
    let target = def.ip_rel_memory_address();
    (image.region_of(target) == Region::Code && image.code_bytes(target).is_ok()).then_some(target)
}

#[cfg(test)]
mod tests {
    use iced_x86::Instruction;

    use crate::cfg::Edge;
    use crate::code_image::{CodeImage, CodeSection};
    use crate::discover::discover;

    const BASE: u64 = 0x1_4000_0000;
    const TEXT: u64 = BASE + 0x1000;
    const DATA: u64 = BASE + 0x2000;

    /// `mov rax,[rip+slot]; test rax,rax; jz +2; jmp rax; ret` at 0, a lone `ret` at 0x20.
    fn code() -> Vec<u8> {
        let mut c = vec![0x48, 0x8b, 0x05, 0x01, 0x10, 0x00, 0x00]; // mov rax,[DATA+8]
        c.extend([0x48, 0x85, 0xc0, 0x74, 0x02, 0xff, 0xe0, 0xc3]);
        c.resize(0x20, 0xcc);
        c.push(0xc3);
        c
    }

    fn run(code: &[u8], value: u64, relocated: bool) -> crate::discover::Analysis {
        let mut data = [0u8; 16];
        data[8..].copy_from_slice(&value.to_le_bytes());
        let mut image = CodeImage::new(
            BASE,
            vec![
                CodeSection::new(TEXT, code, true),
                CodeSection::new(DATA, &data, false),
            ],
            vec![TEXT],
        );
        if relocated {
            image.data_relocs.insert(DATA + 8);
        }
        discover(&image)
    }

    #[test]
    fn relocated_slot_seeds_its_target_as_an_entry() {
        let a = run(&code(), TEXT + 0x20, true);
        assert_eq!(
            a.pointer_slots.get(&(TEXT + 12)),
            Some(&(DATA + 8, TEXT + 0x20))
        );
        assert!(a.functions.contains_key(&(TEXT + 0x20)));
        // Still on run-time dispatch: the slot may be rewritten.
        assert!(a.unresolved_jumps.contains_key(&(TEXT + 12)));
    }

    #[test]
    fn slot_without_relocation_is_not_trusted() {
        let a = run(&code(), TEXT + 0x20, false);
        assert!(a.pointer_slots.is_empty());
        assert!(!a.functions.contains_key(&(TEXT + 0x20)));
    }

    #[test]
    fn slot_value_outside_code_or_null_is_ignored() {
        for value in [0, DATA, BASE + 0x9000_0000] {
            let a = run(&code(), value, true);
            assert!(a.pointer_slots.is_empty(), "{value:#x}");
        }
    }

    #[test]
    fn register_rewritten_after_the_load_is_not_followed() {
        // mov rax, rbx between the load and the jump replaces the loaded value.
        let mut c = code();
        c.splice(10..12, []); // drop jz
        c.splice(10..10, [0x48, 0x89, 0xd8]); // mov rax, rbx
        c[13] = 0xff;
        c[14] = 0xe0;
        let a = run(&c, TEXT + 0x20, true);
        assert!(a.pointer_slots.is_empty());
    }

    /// `code()` plus a store function at 0x30 (`lea rax,[rip+0x20]; mov [rip+slot],rax; ret`)
    /// and the `extra` bytes from 0x40 on, with 0x20 and every 0x10-aligned function listed in
    /// `entries` as entries; the slot starts zero and unrelocated.
    fn run_stores(extra: &[u8], entries: &[u64]) -> crate::discover::Analysis {
        let mut c = code();
        c.resize(0x30, 0xcc);
        let rel = |at: u64, len: u64, to: u64| ((to as i64) - (TEXT + at + len) as i64) as i32;
        c.extend([0x48, 0x8d, 0x05]);
        c.extend(rel(0x30, 7, TEXT + 0x20).to_le_bytes());
        c.extend([0x48, 0x89, 0x05]);
        c.extend(rel(0x37, 7, DATA + 8).to_le_bytes());
        c.push(0xc3);
        c.resize(0x40, 0xcc);
        c.extend(extra);
        let data = [0u8; 16];
        let mut seeds = vec![TEXT, TEXT + 0x20, TEXT + 0x30];
        seeds.extend(entries);
        let image = CodeImage::new(
            BASE,
            vec![
                CodeSection::new(TEXT, &c, true),
                CodeSection::new(DATA, &data, false),
            ],
            seeds,
        );
        let mut a = discover(&image);
        super::resolve_slot_stores(&image, &mut a);
        a
    }

    fn arms(a: &crate::discover::Analysis) -> Vec<Edge> {
        a.functions[&TEXT]
            .blocks
            .values()
            .find(|b| b.instructions.last().map(Instruction::ip) == Some(TEXT + 12))
            .expect("jump block")
            .successors
            .iter()
            .filter(|e| matches!(e, Edge::TailCall(_)))
            .copied()
            .collect()
    }

    #[test]
    fn slot_stored_only_with_a_code_address_resolves_its_jump() {
        let a = run_stores(&[], &[]);
        assert_eq!(arms(&a), vec![Edge::TailCall(TEXT + 0x20)]);
        assert!(!a.unresolved_jumps.contains_key(&(TEXT + 12)));
        assert_eq!(super::slot_store_counts(&a), (1, 0));
    }

    #[test]
    fn unknown_store_keeps_the_jump_on_dispatch_with_the_known_arm() {
        // 0x40: mov [rip+slot], rcx; ret (rcx is a parameter).
        let mut extra = vec![0x48, 0x89, 0x0d];
        extra.extend((((DATA + 8) as i64 - (TEXT + 0x47) as i64) as i32).to_le_bytes());
        extra.push(0xc3);
        let a = run_stores(&extra, &[TEXT + 0x40]);
        assert_eq!(arms(&a), vec![Edge::TailCall(TEXT + 0x20)]);
        assert!(a.unresolved_jumps.contains_key(&(TEXT + 12)));
        assert_eq!(super::slot_store_counts(&a), (0, 1));
    }

    #[test]
    fn address_taken_slot_or_null_store_is_handled() {
        // 0x40: lea rcx, [rip+slot-0x10] (the enclosing object escapes); ret.
        let mut extra = vec![0x48, 0x8d, 0x0d];
        extra.extend((((DATA - 8) as i64 - (TEXT + 0x47) as i64) as i32).to_le_bytes());
        extra.push(0xc3);
        let a = run_stores(&extra, &[TEXT + 0x40]);
        assert!(a.unresolved_jumps.contains_key(&(TEXT + 12)));
        // 0x40: xor eax, eax; mov [rip+slot], rax; ret (a reset to null keeps the set closed).
        let mut extra = vec![0x31, 0xc0, 0x48, 0x89, 0x05];
        extra.extend((((DATA + 8) as i64 - (TEXT + 0x49) as i64) as i32).to_le_bytes());
        extra.push(0xc3);
        let a = run_stores(&extra, &[TEXT + 0x40]);
        assert!(!a.unresolved_jumps.contains_key(&(TEXT + 12)));
    }
}
