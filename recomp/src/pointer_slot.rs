//! Static pointer-slot lookup for tail calls through a loaded pointer.
//!
//! `mov reg, [rip+slot]` (then optionally register copies, a null check, ...) followed by
//! `jmp reg` is a tail call through a global function pointer. When the slot is a relocated
//! 64-bit pointer in a data section (listed in [`CodeImage::data_relocs`]) whose initial value is a
//! code address, that address is a real function entry and can be lifted ahead of time. The jump
//! itself stays on run-time dispatch: the slot may be rewritten later. Nothing is invented: a slot
//! without a base relocation, or one whose value is not in an executable section, yields nothing.

use iced_x86::{FlowControl, Instruction, InstructionInfoFactory, Mnemonic, OpAccess, OpKind};

use crate::code_image::{CodeImage, Region};

/// Backward steps (register copies followed) before giving up.
const MAX_COPIES: u32 = 4;

/// `(slot, initial code target)` for the indirect `jmp` ending `trace`, if it jumps through a
/// register loaded from a relocated data slot that statically holds a code address.
pub(crate) fn static_target(image: &CodeImage<'_>, trace: &[&Instruction]) -> Option<(u64, u64)> {
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
                return slot_target(image, instr.ip_rel_memory_address());
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

#[cfg(test)]
mod tests {
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
}
