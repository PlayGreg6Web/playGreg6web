//! Per-function translation of decoded x86-64 instructions into LLVM IR text.
//!
//! See the parent module for the state, memory, call and flag model. Every instruction is lifted
//! into the function body inside a checkpoint: if any part of it turns out to be unsupported, the
//! partial output is rolled back and replaced by an `__playgta6_unimplemented` trap.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use iced_x86::{
    Code, ConditionCode, FlowControl, Formatter, Instruction, InstructionInfoFactory,
    IntelFormatter, Mnemonic, OpKind, Register,
};

use super::{
    GUEST_STATE_TYPE, HELPER_DIVIDE_ERROR, HELPER_INDIRECT_CALL, HELPER_TRAP, HELPER_UNIMPLEMENTED,
    LiftError, LiftedFunction, Support, UnsupportedInstruction, function_symbol,
};
use crate::cfg::{self, CallTarget, DiagnosticKind, Edge, Terminator};

#[path = "simd/mod.rs"]
pub(super) mod simd;

// Integer instruction families beyond the core set (child modules so they can use the private
// lifter primitives; the files live next to this one).
#[path = "atomics.rs"]
mod atomics;
#[path = "bit_ops.rs"]
mod bit_ops;
#[path = "integer.rs"]
mod integer;
#[path = "string_ops.rs"]
mod string_ops;

pub(super) use atomics::CMPXCHG16B_IR;
pub(super) use bit_ops::{PDEP_IR, PEXT_IR};
pub(super) use integer::CRC32C_IR;

const GPR_NAMES: [&str; 16] = [
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13",
    "r14", "r15",
];
const RSP: usize = 4;
const RBP: usize = 5;

/// Arithmetic flags in guest-state order (`STATE_FLAGS_OFFSET + index`).
#[derive(Debug, Clone, Copy)]
enum Flag {
    Cf = 0,
    Pf = 1,
    Af = 2,
    Zf = 3,
    Sf = 4,
    Of = 5,
}

const FLAGS: [Flag; 6] = [Flag::Cf, Flag::Pf, Flag::Af, Flag::Zf, Flag::Sf, Flag::Of];

impl Flag {
    fn slot(self) -> &'static str {
        match self {
            Flag::Cf => "%cf",
            Flag::Pf => "%pf",
            Flag::Af => "%af",
            Flag::Zf => "%zf",
            Flag::Sf => "%sf",
            Flag::Of => "%of",
        }
    }
}

/// Why a single instruction could not be lifted.
enum InsnError {
    /// Replace the instruction with an `__playgta6_unimplemented` trap.
    Unsupported(String),
    /// The whole function is malformed.
    Cfg(LiftError),
}

impl From<LiftError> for InsnError {
    fn from(err: LiftError) -> Self {
        InsnError::Cfg(err)
    }
}

type R<T> = Result<T, InsnError>;

fn unsupported<T>(reason: impl Into<String>) -> R<T> {
    Err(InsnError::Unsupported(reason.into()))
}

/// A general purpose register reference: `(gpr index, width in bits)` or a legacy high-byte reg.
enum RegRef {
    Low(usize, u32),
    High8(usize),
}

#[derive(Clone, Copy)]
enum Alu {
    Add,
    Adc,
    Sub,
    Sbb,
    Cmp,
    And,
    Or,
    Xor,
    Test,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Shift {
    Shl,
    Shr,
    Sar,
    Rol,
    Ror,
}

fn ty(bits: u32) -> String {
    format!("i{bits}")
}

fn mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

/// An integer literal of width `bits`, printed as a signed decimal so it is valid for any width.
fn imm(value: u64, bits: u32) -> String {
    if bits >= 64 {
        return (value as i64).to_string();
    }
    let v = value & mask(bits);
    let signed = if v >> (bits - 1) & 1 == 1 {
        (v | !mask(bits)) as i64
    } else {
        v as i64
    };
    signed.to_string()
}

fn block_label(va: u64) -> String {
    format!("bb_{va:X}")
}

fn exit_label(va: u64) -> String {
    format!("ext_{va:X}")
}

fn reg_ref(reg: Register) -> R<RegRef> {
    if !reg.is_gpr() {
        return unsupported(format!(
            "register {reg:?} is not a general purpose register"
        ));
    }
    let full = reg.full_register();
    if !full.is_gpr64() {
        return unsupported(format!("register {reg:?} has no 64-bit parent"));
    }
    let idx = full.number();
    Ok(match reg {
        Register::AH | Register::CH | Register::DH | Register::BH => RegRef::High8(idx),
        _ => RegRef::Low(idx, (reg.size() * 8) as u32),
    })
}

fn is_immediate(kind: OpKind) -> bool {
    matches!(
        kind,
        OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64
            | OpKind::Immediate32to64
    )
}

/// Lifts one function (see `super::lift_function`).
pub(super) fn lift(func: &cfg::Function) -> Result<LiftedFunction, LiftError> {
    if !func.blocks.contains_key(&func.entry) {
        return Err(LiftError::MissingEntryBlock { entry: func.entry });
    }
    let mut regs = [false; 16];
    regs[RSP] = true;
    let mut xmms = [false; 16];
    let mut faulting_calls = BTreeSet::new();
    let mut info = InstructionInfoFactory::new();
    for (&start, block) in &func.blocks {
        let first = block
            .instructions
            .first()
            .map_or(block.start, Instruction::ip);
        if block.start != start || first != start {
            return Err(LiftError::BlockStartMismatch {
                block: start,
                first_instruction: first,
            });
        }
        if block.instructions.is_empty() && block.terminator != Terminator::Invalid {
            return Err(LiftError::EmptyBlock { block: start });
        }
        for ins in &block.instructions {
            for used in info.info(ins).used_registers() {
                let reg = used.register();
                if reg.is_gpr() && reg.full_register().is_gpr64() {
                    regs[reg.full_register().number()] = true;
                }
                simd::note_register(reg, &mut xmms);
            }
            simd::note_instruction(ins, &mut xmms);
        }
        // Direct calls whose target discovery found to be unmapped or non-executable fault.
        for call in &block.calls {
            if let CallTarget::Direct(target) = call.target
                && block.diagnostics.iter().any(|d| {
                    d.site == call.site
                        && d.address == target
                        && matches!(
                            d.kind,
                            DiagnosticKind::Unmapped | DiagnosticKind::NotExecutable
                        )
                })
            {
                faulting_calls.insert(call.site);
            }
        }
    }

    let mut lifter = FunctionLifter {
        func,
        regs,
        xmms,
        faulting_calls,
        body: String::new(),
        tmp: 0,
        label: 0,
        exits: Vec::new(),
        callees: Vec::new(),
        support: BTreeSet::new(),
        unsupported: Vec::new(),
        ea: None,
    };
    lifter.lift_blocks()?;
    Ok(lifter.finish())
}

/// Terminators whose last instruction is the control transfer itself (lowered from the edges).
fn ends_in_transfer(terminator: Terminator) -> bool {
    matches!(
        terminator,
        Terminator::Branch
            | Terminator::Jump
            | Terminator::IndirectJump
            | Terminator::Return
            | Terminator::Trap
    )
}

/// Instructions that may only appear as the last instruction of a block.
fn is_control_transfer(ins: &Instruction) -> bool {
    matches!(
        ins.flow_control(),
        FlowControl::UnconditionalBranch
            | FlowControl::IndirectBranch
            | FlowControl::ConditionalBranch
            | FlowControl::Return
    )
}

struct FunctionLifter<'a> {
    func: &'a cfg::Function,
    /// GPRs the function touches; only these get an alloca and are synced with the state.
    regs: [bool; 16],
    /// XMM registers the function touches (`alloca <2 x i64>` each, synced like the GPRs).
    xmms: [bool; 16],
    /// Sites of direct calls into unmapped or non-executable memory (lifted as traps).
    faulting_calls: BTreeSet<u64>,
    body: String,
    tmp: u32,
    label: u32,
    /// Targets of edges that leave the function (become tail calls).
    exits: Vec<u64>,
    callees: Vec<u64>,
    support: BTreeSet<Support>,
    unsupported: Vec<UnsupportedInstruction>,
    /// Effective address (with segment base) of the current instruction's memory operand.
    ea: Option<String>,
}

struct Checkpoint {
    body: usize,
    exits: usize,
    callees: usize,
    support: BTreeSet<Support>,
}

impl FunctionLifter<'_> {
    // ----- driver -------------------------------------------------------------------------------

    fn lift_blocks(&mut self) -> Result<(), LiftError> {
        let func = self.func;
        for (&start, block) in &func.blocks {
            let _ = writeln!(self.body, "{}:", block_label(start));
            let (body, transfer) = match block.instructions.split_last() {
                Some((last, rest)) if ends_in_transfer(block.terminator) => (rest, Some(last)),
                _ => (&block.instructions[..], None),
            };
            let mut trapped = false;
            for ins in body {
                if is_control_transfer(ins) {
                    return Err(LiftError::InstructionAfterTerminator {
                        block: start,
                        address: ins.ip(),
                    });
                }
                let checkpoint = self.checkpoint();
                self.ea = None;
                match self.lift_instruction(ins) {
                    Ok(()) => {}
                    Err(InsnError::Cfg(err)) => return Err(err),
                    Err(InsnError::Unsupported(reason)) => {
                        self.rollback(checkpoint);
                        self.trap_unsupported(ins, reason);
                        // The trap does not return; the rest of the block is dead.
                        trapped = true;
                        break;
                    }
                }
            }
            if !trapped {
                self.lift_terminator(block, transfer)?;
            }
        }
        Ok(())
    }

    /// Ends the LLVM block for `block` according to its terminator and typed successor edges.
    /// `transfer` is the block's last instruction when it is the control transfer itself.
    ///
    /// Terminator kinds added to `cfg` later degrade like an unresolved indirect jump (runtime
    /// dispatch) when the block ends in an indirect `jmp`, and to an unsupported trap otherwise.
    fn lift_terminator(
        &mut self,
        block: &cfg::BasicBlock,
        transfer: Option<&Instruction>,
    ) -> Result<(), LiftError> {
        let checkpoint = self.checkpoint();
        self.ea = None;
        let result = match (block.terminator, transfer) {
            (Terminator::Fallthrough | Terminator::Jump, _) => {
                self.goto(block, block.successors.first())
            }
            (Terminator::Branch, Some(ins)) => self.branch(block, ins),
            (Terminator::IndirectJump, Some(ins)) => self.indirect_jump(block, ins),
            (Terminator::Return, Some(ins)) => self.ret(ins),
            (Terminator::Trap, Some(ins)) => {
                self.trap(ins.ip());
                Ok(())
            }
            // The non-returning call itself was lifted with the block body. If it returns
            // anyway, execution would continue at the next instruction.
            (Terminator::NoReturnCall, _) => {
                self.trap(block.end);
                Ok(())
            }
            // Decoding stopped at `block.end` (the undecodable or unmapped address).
            (Terminator::Invalid, _) => {
                self.trap(block.end);
                Ok(())
            }
            // Fallback for terminator kinds added to `cfg` later (none exist today).
            #[allow(unreachable_patterns)]
            (_, Some(ins)) if ins.is_jmp_near_indirect() => self.indirect_jump(block, ins),
            (terminator, _) => unsupported(format!(
                "block terminator '{}' is not supported",
                terminator.as_str()
            )),
        };
        match result {
            Ok(()) => Ok(()),
            Err(InsnError::Cfg(err)) => Err(err),
            Err(InsnError::Unsupported(reason)) => {
                self.rollback(checkpoint);
                match transfer.or(block.instructions.last()) {
                    Some(ins) => self.trap_unsupported(ins, reason),
                    None => self.trap(block.end),
                }
                Ok(())
            }
        }
    }

    /// Resolves a successor edge to an LLVM label: a block of this function, or a tail-call exit.
    /// Edge kinds without a static target are reported as unsupported.
    fn successor(&mut self, block: &cfg::BasicBlock, edge: Option<&Edge>) -> R<String> {
        let Some(edge) = edge else {
            return unsupported("block has no successor edge");
        };
        if let Edge::TailCall(target) = *edge {
            self.exits.push(target);
            return Ok(exit_label(target));
        }
        match edge.intra_target() {
            Some(target) if self.func.blocks.contains_key(&target) => Ok(block_label(target)),
            Some(target) => Err(InsnError::Cfg(LiftError::UnknownSuccessor {
                block: block.start,
                target,
            })),
            None => unsupported(format!(
                "successor edge '{}' has no static target",
                edge.kind_str()
            )),
        }
    }

    fn goto(&mut self, block: &cfg::BasicBlock, edge: Option<&Edge>) -> R<()> {
        let target = self.successor(block, edge)?;
        self.emit(&format!("br label %{target}"));
        Ok(())
    }

    fn finish(mut self) -> LiftedFunction {
        let entry = self.func.entry;
        let symbol = function_symbol(entry);

        let mut exits: Vec<u64> = std::mem::take(&mut self.exits);
        exits.sort_unstable();
        exits.dedup();
        for &target in &exits {
            let _ = writeln!(self.body, "{}:", exit_label(target));
            self.sync_out();
            self.emit(&format!(
                "tail call void @{}(ptr %state)",
                function_symbol(target)
            ));
            self.emit("ret void");
            self.callees.push(target);
        }

        let body = std::mem::take(&mut self.body);
        let mut ir = String::new();
        let _ = writeln!(ir, "define void @{symbol}(ptr noalias %state) #0 {{");
        ir.push_str("entry:\n");
        for (idx, name) in GPR_NAMES.iter().enumerate() {
            if self.regs[idx] {
                let _ = writeln!(ir, "  %{name} = alloca i64");
            }
        }
        for flag in FLAGS {
            let _ = writeln!(ir, "  {} = alloca i1", flag.slot());
        }
        self.xmm_allocas(&mut ir);
        // The emit helpers write to `self.body`, which is empty again here: build the entry
        // block's state loads there and splice them in front of the lifted blocks.
        self.sync_in();
        ir.push_str(&std::mem::take(&mut self.body));
        let _ = writeln!(ir, "  br label %{}", block_label(entry));
        ir.push_str(&body);
        ir.push_str("}\n");

        let mut unsupported = std::mem::take(&mut self.unsupported);
        unsupported.sort_by_key(|u| u.address);
        LiftedFunction {
            entry,
            symbol,
            ir,
            callees: self.callees.iter().copied().collect(),
            unsupported,
            support: std::mem::take(&mut self.support),
        }
    }

    fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            body: self.body.len(),
            exits: self.exits.len(),
            callees: self.callees.len(),
            support: self.support.clone(),
        }
    }

    fn rollback(&mut self, checkpoint: Checkpoint) {
        self.body.truncate(checkpoint.body);
        self.exits.truncate(checkpoint.exits);
        self.callees.truncate(checkpoint.callees);
        self.support = checkpoint.support;
    }

    fn trap_unsupported(&mut self, ins: &Instruction, reason: String) {
        let mut text = String::new();
        IntelFormatter::new().format(ins, &mut text);
        self.unsupported.push(UnsupportedInstruction {
            address: ins.ip(),
            mnemonic: format!("{:?}", ins.mnemonic()).to_lowercase(),
            instruction: text,
            reason,
        });
        self.sync_out();
        self.emit(&format!(
            "call void @{HELPER_UNIMPLEMENTED}(ptr %state, i64 {})",
            imm(ins.ip(), 64)
        ));
        self.emit("unreachable");
    }

    /// Ends the current LLVM block with a call to the guest-trap helper.
    fn trap(&mut self, va: u64) {
        self.sync_out();
        self.emit(&format!(
            "call void @{HELPER_TRAP}(ptr %state, i64 {})",
            imm(va, 64)
        ));
        self.emit("unreachable");
    }

    // ----- IR emission primitives ---------------------------------------------------------------

    fn emit(&mut self, line: &str) {
        self.body.push_str("  ");
        self.body.push_str(line);
        self.body.push('\n');
    }

    /// Emits `%tN = <rhs>` and returns `%tN`.
    fn val(&mut self, rhs: &str) -> String {
        let name = format!("%t{}", self.tmp);
        self.tmp += 1;
        self.emit(&format!("{name} = {rhs}"));
        name
    }

    fn new_label(&mut self, prefix: &str) -> String {
        let name = format!("{prefix}{}", self.label);
        self.label += 1;
        name
    }

    fn start_block(&mut self, label: &str) {
        let _ = writeln!(self.body, "{label}:");
    }

    // ----- guest state ---------------------------------------------------------------------------

    fn state_gpr_ptr(&mut self, idx: usize) -> String {
        self.val(&format!(
            "getelementptr inbounds {GUEST_STATE_TYPE}, ptr %state, i64 0, i32 0, i64 {idx}"
        ))
    }

    fn state_flag_ptr(&mut self, flag: Flag) -> String {
        self.val(&format!(
            "getelementptr inbounds {GUEST_STATE_TYPE}, ptr %state, i64 0, i32 2, i64 {}",
            flag as usize
        ))
    }

    /// Loads all tracked registers and flags from `%state` into their allocas.
    fn sync_in(&mut self) {
        for idx in 0..16 {
            if self.regs[idx] {
                let ptr = self.state_gpr_ptr(idx);
                let v = self.val(&format!("load i64, ptr {ptr}"));
                self.emit(&format!("store i64 {v}, ptr %{}", GPR_NAMES[idx]));
            }
        }
        for flag in FLAGS {
            let ptr = self.state_flag_ptr(flag);
            let byte = self.val(&format!("load i8, ptr {ptr}"));
            let bit = self.val(&format!("trunc i8 {byte} to i1"));
            self.emit(&format!("store i1 {bit}, ptr {}", flag.slot()));
        }
        self.sync_xmm_in();
    }

    /// Stores all tracked registers and flags from their allocas back into `%state`.
    fn sync_out(&mut self) {
        for idx in 0..16 {
            if self.regs[idx] {
                let v = self.val(&format!("load i64, ptr %{}", GPR_NAMES[idx]));
                let ptr = self.state_gpr_ptr(idx);
                self.emit(&format!("store i64 {v}, ptr {ptr}"));
            }
        }
        for flag in FLAGS {
            let bit = self.val(&format!("load i1, ptr {}", flag.slot()));
            let byte = self.val(&format!("zext i1 {bit} to i8"));
            let ptr = self.state_flag_ptr(flag);
            self.emit(&format!("store i8 {byte}, ptr {ptr}"));
        }
        self.sync_xmm_out();
    }

    fn segment_base(&mut self, field: u32) -> String {
        let ptr = self.val(&format!(
            "getelementptr inbounds {GUEST_STATE_TYPE}, ptr %state, i64 0, i32 {field}"
        ));
        self.val(&format!("load i64, ptr {ptr}"))
    }

    // ----- registers -----------------------------------------------------------------------------

    fn gpr_slot(&self, idx: usize) -> R<String> {
        if idx < 16 && self.regs[idx] {
            Ok(format!("%{}", GPR_NAMES[idx]))
        } else {
            unsupported(format!("internal: register index {idx} is not tracked"))
        }
    }

    fn load_gpr(&mut self, idx: usize) -> R<String> {
        let slot = self.gpr_slot(idx)?;
        Ok(self.val(&format!("load i64, ptr {slot}")))
    }

    fn store_gpr(&mut self, idx: usize, value: &str) -> R<()> {
        let slot = self.gpr_slot(idx)?;
        self.emit(&format!("store i64 {value}, ptr {slot}"));
        Ok(())
    }

    /// Reads a GPR (any width) and returns `(value, bits)`.
    fn read_reg(&mut self, reg: Register) -> R<(String, u32)> {
        match reg_ref(reg)? {
            RegRef::Low(idx, 64) => Ok((self.load_gpr(idx)?, 64)),
            RegRef::Low(idx, bits) => {
                let full = self.load_gpr(idx)?;
                Ok((self.val(&format!("trunc i64 {full} to i{bits}")), bits))
            }
            RegRef::High8(idx) => {
                let full = self.load_gpr(idx)?;
                let shifted = self.val(&format!("lshr i64 {full}, 8"));
                Ok((self.val(&format!("trunc i64 {shifted} to i8")), 8))
            }
        }
    }

    /// Writes a GPR with x86-64 semantics (32-bit writes zero-extend, 8/16-bit writes merge).
    fn write_reg(&mut self, reg: Register, value: &str, bits: u32) -> R<()> {
        match reg_ref(reg)? {
            RegRef::Low(idx, reg_bits) if reg_bits != bits => unsupported(format!(
                "internal: writing {bits}-bit value to {reg_bits}-bit register {reg:?} ({idx})"
            )),
            RegRef::Low(idx, 64) => self.store_gpr(idx, value),
            RegRef::Low(idx, 32) => {
                let wide = self.val(&format!("zext i32 {value} to i64"));
                self.store_gpr(idx, &wide)
            }
            RegRef::Low(idx, bits) => self.merge_gpr(idx, value, bits, 0),
            RegRef::High8(idx) if bits == 8 => self.merge_gpr(idx, value, 8, 8),
            RegRef::High8(_) => unsupported("internal: high-byte register width mismatch"),
        }
    }

    /// Replaces bits `shift..shift+bits` of a 64-bit register, preserving the others.
    fn merge_gpr(&mut self, idx: usize, value: &str, bits: u32, shift: u32) -> R<()> {
        let old = self.load_gpr(idx)?;
        let keep = !(mask(bits) << shift);
        let kept = self.val(&format!("and i64 {old}, {}", imm(keep, 64)));
        let wide = self.val(&format!("zext i{bits} {value} to i64"));
        let placed = if shift == 0 {
            wide
        } else {
            self.val(&format!("shl i64 {wide}, {shift}"))
        };
        let merged = self.val(&format!("or i64 {kept}, {placed}"));
        self.store_gpr(idx, &merged)
    }

    // ----- memory --------------------------------------------------------------------------------

    /// Computes the memory operand's address as an `i64`, optionally adding the `fs`/`gs` base.
    fn address(&mut self, ins: &Instruction, with_segment: bool) -> R<String> {
        let base = ins.memory_base();
        let index = ins.memory_index();
        // For RIP-relative operands iced already returns the absolute target here.
        let disp = ins.memory_displacement64();
        let mut addr32 = false;
        let mut acc: Option<String> = None;

        if base == Register::EIP {
            return unsupported("EIP-relative addressing");
        }
        if base != Register::None && base != Register::RIP {
            if !(base.is_gpr64() || base.is_gpr32()) {
                return unsupported(format!("memory base register {base:?}"));
            }
            addr32 |= base.is_gpr32();
            acc = Some(self.read_addr_reg(base)?);
        }
        if index != Register::None {
            if !(index.is_gpr64() || index.is_gpr32()) {
                return unsupported(format!("memory index register {index:?}"));
            }
            addr32 |= index.is_gpr32();
            let mut scaled = self.read_addr_reg(index)?;
            let scale = ins.memory_index_scale();
            if scale > 1 {
                scaled = self.val(&format!("shl i64 {scaled}, {}", scale.trailing_zeros()));
            }
            acc = Some(match acc {
                Some(sum) => self.val(&format!("add i64 {sum}, {scaled}")),
                None => scaled,
            });
        }
        let mut addr = match acc {
            Some(sum) if disp != 0 => self.val(&format!("add i64 {sum}, {}", imm(disp, 64))),
            Some(sum) => sum,
            None => imm(if addr32 { disp & mask(32) } else { disp }, 64),
        };
        if addr32 {
            addr = self.val(&format!("and i64 {addr}, 4294967295"));
        }
        if with_segment {
            let field = match ins.memory_segment() {
                Register::FS => Some(3),
                Register::GS => Some(4),
                _ => None,
            };
            if let Some(field) = field {
                let seg = self.segment_base(field);
                addr = self.val(&format!("add i64 {addr}, {seg}"));
            }
        }
        Ok(addr)
    }

    fn read_addr_reg(&mut self, reg: Register) -> R<String> {
        let (v, bits) = self.read_reg(reg)?;
        Ok(if bits == 64 {
            v
        } else {
            self.val(&format!("zext i{bits} {v} to i64"))
        })
    }

    /// Effective (segmented) address of the current instruction's memory operand, computed once
    /// per instruction so read-modify-write and `xchg` see a consistent address.
    fn mem_ea(&mut self, ins: &Instruction) -> R<String> {
        if let Some(ea) = &self.ea {
            return Ok(ea.clone());
        }
        let ea = self.address(ins, true)?;
        self.ea = Some(ea.clone());
        Ok(ea)
    }

    /// The single place where a guest address becomes a host pointer (identity mapping of guest
    /// virtual addresses onto wasm64 linear memory).
    fn guest_ptr(&mut self, addr: &str) -> String {
        self.val(&format!("inttoptr i64 {addr} to ptr"))
    }

    fn load_mem(&mut self, addr: &str, bits: u32) -> String {
        let ptr = self.guest_ptr(addr);
        self.val(&format!("load i{bits}, ptr {ptr}, align 1"))
    }

    fn store_mem(&mut self, addr: &str, value: &str, bits: u32) {
        let ptr = self.guest_ptr(addr);
        self.emit(&format!("store i{bits} {value}, ptr {ptr}, align 1"));
    }

    fn push(&mut self, value: &str, bits: u32) -> R<()> {
        let rsp = self.load_gpr(RSP)?;
        let new_rsp = self.val(&format!("sub i64 {rsp}, {}", bits / 8));
        self.store_mem(&new_rsp, value, bits);
        self.store_gpr(RSP, &new_rsp)
    }

    // ----- operands ------------------------------------------------------------------------------

    fn op_bits(&self, ins: &Instruction, i: u32) -> R<u32> {
        match ins.op_kind(i) {
            OpKind::Register => {
                let reg = ins.op_register(i);
                if !reg.is_gpr() {
                    return unsupported(format!("register operand {reg:?}"));
                }
                Ok((reg.size() * 8) as u32)
            }
            OpKind::Memory => match ins.memory_size().size() {
                size @ (1 | 2 | 4 | 8) => Ok((size * 8) as u32),
                size => unsupported(format!("{size}-byte memory operand")),
            },
            kind => unsupported(format!("operand kind {kind:?}")),
        }
    }

    fn read_op(&mut self, ins: &Instruction, i: u32, bits: u32) -> R<String> {
        match ins.op_kind(i) {
            OpKind::Register => {
                let (v, reg_bits) = self.read_reg(ins.op_register(i))?;
                if reg_bits != bits {
                    return unsupported("internal: operand width mismatch");
                }
                Ok(v)
            }
            OpKind::Memory => {
                let ea = self.mem_ea(ins)?;
                Ok(self.load_mem(&ea, bits))
            }
            kind if is_immediate(kind) => match ins.try_immediate(i) {
                Ok(value) => Ok(imm(value, bits)),
                Err(_) => unsupported("unreadable immediate"),
            },
            kind => unsupported(format!("operand kind {kind:?}")),
        }
    }

    fn write_op(&mut self, ins: &Instruction, i: u32, value: &str, bits: u32) -> R<()> {
        match ins.op_kind(i) {
            OpKind::Register => self.write_reg(ins.op_register(i), value, bits),
            OpKind::Memory => {
                let ea = self.mem_ea(ins)?;
                self.store_mem(&ea, value, bits);
                Ok(())
            }
            kind => unsupported(format!("destination operand kind {kind:?}")),
        }
    }

    // ----- flags ---------------------------------------------------------------------------------

    fn set_flag(&mut self, flag: Flag, value: &str) {
        self.emit(&format!("store i1 {value}, ptr {}", flag.slot()));
    }

    fn get_flag(&mut self, flag: Flag) -> String {
        self.val(&format!("load i1, ptr {}", flag.slot()))
    }

    fn msb(&mut self, v: &str, bits: u32) -> String {
        self.val(&format!("icmp slt i{bits} {v}, 0"))
    }

    /// PF: set when the low byte of the result has an even number of 1 bits.
    fn parity(&mut self, r: &str, bits: u32) -> String {
        let low = if bits == 8 {
            r.to_string()
        } else {
            self.val(&format!("trunc i{bits} {r} to i8"))
        };
        self.support.insert(Support::Ctpop8);
        let count = self.val(&format!("call i8 @llvm.ctpop.i8(i8 {low})"));
        let odd = self.val(&format!("trunc i8 {count} to i1"));
        self.val(&format!("xor i1 {odd}, true"))
    }

    /// AF: carry/borrow out of bit 3, `((a ^ b ^ r) & 0x10) != 0`.
    fn aux_flag(&mut self, a: &str, b: &str, r: &str, bits: u32) -> String {
        let t = ty(bits);
        let x = self.val(&format!("xor {t} {a}, {b}"));
        let x = self.val(&format!("xor {t} {x}, {r}"));
        let x = self.val(&format!("and {t} {x}, 16"));
        self.val(&format!("icmp ne {t} {x}, 0"))
    }

    /// Computes ZF, SF and PF of `r` (not stored).
    fn zsp(&mut self, r: &str, bits: u32) -> [String; 3] {
        let zf = self.val(&format!("icmp eq i{bits} {r}, 0"));
        let sf = self.msb(r, bits);
        let pf = self.parity(r, bits);
        [zf, sf, pf]
    }

    fn set_zsp(&mut self, r: &str, bits: u32) {
        let [zf, sf, pf] = self.zsp(r, bits);
        self.set_flag(Flag::Zf, &zf);
        self.set_flag(Flag::Sf, &sf);
        self.set_flag(Flag::Pf, &pf);
    }

    fn condition(&mut self, cc: ConditionCode) -> R<String> {
        use ConditionCode as C;
        let (flag, negate) = match cc {
            C::o => (Flag::Of, false),
            C::no => (Flag::Of, true),
            C::b => (Flag::Cf, false),
            C::ae => (Flag::Cf, true),
            C::e => (Flag::Zf, false),
            C::ne => (Flag::Zf, true),
            C::s => (Flag::Sf, false),
            C::ns => (Flag::Sf, true),
            C::p => (Flag::Pf, false),
            C::np => (Flag::Pf, true),
            C::be | C::a => {
                let cf = self.get_flag(Flag::Cf);
                let zf = self.get_flag(Flag::Zf);
                let be = self.val(&format!("or i1 {cf}, {zf}"));
                return Ok(if cc == C::be {
                    be
                } else {
                    self.val(&format!("xor i1 {be}, true"))
                });
            }
            C::l | C::ge => {
                let sf = self.get_flag(Flag::Sf);
                let of = self.get_flag(Flag::Of);
                let l = self.val(&format!("xor i1 {sf}, {of}"));
                return Ok(if cc == C::l {
                    l
                } else {
                    self.val(&format!("xor i1 {l}, true"))
                });
            }
            C::le | C::g => {
                let sf = self.get_flag(Flag::Sf);
                let of = self.get_flag(Flag::Of);
                let zf = self.get_flag(Flag::Zf);
                let l = self.val(&format!("xor i1 {sf}, {of}"));
                let le = self.val(&format!("or i1 {zf}, {l}"));
                return Ok(if cc == C::le {
                    le
                } else {
                    self.val(&format!("xor i1 {le}, true"))
                });
            }
            C::None => return unsupported("missing condition code"),
        };
        let v = self.get_flag(flag);
        Ok(if negate {
            self.val(&format!("xor i1 {v}, true"))
        } else {
            v
        })
    }

    // ----- instructions --------------------------------------------------------------------------

    /// Lifts one non-terminating instruction (calls included).
    fn lift_instruction(&mut self, ins: &Instruction) -> R<()> {
        use Mnemonic as M;
        if ins.code() == Code::INVALID {
            return unsupported("invalid instruction encoding");
        }
        if simd::is_simd(ins) {
            return self.lift_simd(ins);
        }
        // String, bit, atomic and miscellaneous integer instructions (incl. every `lock` form).
        if let Some(result) = self.lift_integer(ins) {
            return result;
        }
        match ins.mnemonic() {
            M::Nop | M::Pause | M::Endbr64 => {}
            M::Mov => {
                let bits = self.op_bits(ins, 0)?;
                let v = self.read_op(ins, 1, bits)?;
                self.write_op(ins, 0, &v, bits)?;
            }
            M::Movzx | M::Movsx | M::Movsxd => {
                let dst_bits = self.op_bits(ins, 0)?;
                let src_bits = self.op_bits(ins, 1)?;
                let v = self.read_op(ins, 1, src_bits)?;
                let v = if src_bits == dst_bits {
                    v
                } else if src_bits > dst_bits {
                    return unsupported("narrowing extension");
                } else {
                    let ext = if ins.mnemonic() == M::Movzx {
                        "zext"
                    } else {
                        "sext"
                    };
                    self.val(&format!("{ext} i{src_bits} {v} to i{dst_bits}"))
                };
                self.write_op(ins, 0, &v, dst_bits)?;
            }
            M::Lea => {
                let bits = self.op_bits(ins, 0)?;
                if ins.op_kind(1) != OpKind::Memory {
                    return unsupported("lea without memory operand");
                }
                let addr = self.address(ins, false)?;
                let v = if bits == 64 {
                    addr
                } else {
                    self.val(&format!("trunc i64 {addr} to i{bits}"))
                };
                self.write_op(ins, 0, &v, bits)?;
            }
            M::Xchg => {
                let bits = self.op_bits(ins, 0)?;
                let a = self.read_op(ins, 0, bits)?;
                let b = self.read_op(ins, 1, bits)?;
                self.write_op(ins, 0, &b, bits)?;
                self.write_op(ins, 1, &a, bits)?;
            }
            M::Add => self.alu(ins, Alu::Add)?,
            M::Adc => self.alu(ins, Alu::Adc)?,
            M::Sub => self.alu(ins, Alu::Sub)?,
            M::Sbb => self.alu(ins, Alu::Sbb)?,
            M::Cmp => self.alu(ins, Alu::Cmp)?,
            M::And => self.alu(ins, Alu::And)?,
            M::Or => self.alu(ins, Alu::Or)?,
            M::Xor => self.alu(ins, Alu::Xor)?,
            M::Test => self.alu(ins, Alu::Test)?,
            M::Inc | M::Dec | M::Neg | M::Not => self.unary(ins)?,
            M::Shl | M::Sal => self.shift(ins, Shift::Shl)?,
            M::Shr => self.shift(ins, Shift::Shr)?,
            M::Sar => self.shift(ins, Shift::Sar)?,
            M::Rol => self.shift(ins, Shift::Rol)?,
            M::Ror => self.shift(ins, Shift::Ror)?,
            M::Imul if ins.op_count() >= 2 => self.imul_truncating(ins)?,
            M::Imul | M::Mul => self.mul_widening(ins, ins.mnemonic() == M::Imul)?,
            M::Div | M::Idiv => self.divide(ins, ins.mnemonic() == M::Idiv)?,
            M::Cbw | M::Cwde | M::Cdqe => {
                let (src, dst, bits) = match ins.mnemonic() {
                    M::Cbw => (Register::AL, Register::AX, 16),
                    M::Cwde => (Register::AX, Register::EAX, 32),
                    _ => (Register::EAX, Register::RAX, 64),
                };
                let (v, src_bits) = self.read_reg(src)?;
                let ext = self.val(&format!("sext i{src_bits} {v} to i{bits}"));
                self.write_reg(dst, &ext, bits)?;
            }
            M::Cwd | M::Cdq | M::Cqo => {
                let (src, dst) = match ins.mnemonic() {
                    M::Cwd => (Register::AX, Register::DX),
                    M::Cdq => (Register::EAX, Register::EDX),
                    _ => (Register::RAX, Register::RDX),
                };
                let (v, bits) = self.read_reg(src)?;
                let sign = self.val(&format!("ashr i{bits} {v}, {}", bits - 1));
                self.write_reg(dst, &sign, bits)?;
            }
            M::Seto
            | M::Setno
            | M::Setb
            | M::Setae
            | M::Sete
            | M::Setne
            | M::Setbe
            | M::Seta
            | M::Sets
            | M::Setns
            | M::Setp
            | M::Setnp
            | M::Setl
            | M::Setge
            | M::Setle
            | M::Setg => {
                let c = self.condition(ins.condition_code())?;
                let byte = self.val(&format!("zext i1 {c} to i8"));
                self.write_op(ins, 0, &byte, 8)?;
            }
            M::Cmovo
            | M::Cmovno
            | M::Cmovb
            | M::Cmovae
            | M::Cmove
            | M::Cmovne
            | M::Cmovbe
            | M::Cmova
            | M::Cmovs
            | M::Cmovns
            | M::Cmovp
            | M::Cmovnp
            | M::Cmovl
            | M::Cmovge
            | M::Cmovle
            | M::Cmovg => {
                let bits = self.op_bits(ins, 0)?;
                let c = self.condition(ins.condition_code())?;
                let src = self.read_op(ins, 1, bits)?;
                let old = self.read_op(ins, 0, bits)?;
                let v = self.val(&format!("select i1 {c}, i{bits} {src}, i{bits} {old}"));
                // A 32-bit cmov zero-extends the destination even when the condition is false.
                self.write_op(ins, 0, &v, bits)?;
            }
            M::Push => {
                let bits = match ins.stack_pointer_increment() {
                    -8 => 64,
                    -2 => 16,
                    other => return unsupported(format!("push adjusting rsp by {other}")),
                };
                if matches!(ins.op_kind(0), OpKind::Register | OpKind::Memory)
                    && self.op_bits(ins, 0)? != bits
                {
                    return unsupported("push operand size");
                }
                // The value is read before rsp changes (`push rsp`, `push [rsp+8]`).
                let v = self.read_op(ins, 0, bits)?;
                self.push(&v, bits)?;
            }
            M::Pop => {
                let bits = match ins.stack_pointer_increment() {
                    8 => 64,
                    2 => 16,
                    other => return unsupported(format!("pop adjusting rsp by {other}")),
                };
                if self.op_bits(ins, 0)? != bits {
                    return unsupported("pop operand size");
                }
                let rsp = self.load_gpr(RSP)?;
                let v = self.load_mem(&rsp, bits);
                let new_rsp = self.val(&format!("add i64 {rsp}, {}", bits / 8));
                self.store_gpr(RSP, &new_rsp)?;
                // The destination is written (and a memory destination addressed) after the
                // increment, so `pop rsp` and `pop [rsp+8]` behave like on hardware.
                self.write_op(ins, 0, &v, bits)?;
            }
            M::Leave => {
                if ins.code() != Code::Leaveq {
                    return unsupported("16/32-bit leave");
                }
                let rbp = self.load_gpr(RBP)?;
                let saved = self.load_mem(&rbp, 64);
                let new_rsp = self.val(&format!("add i64 {rbp}, 8"));
                self.store_gpr(RSP, &new_rsp)?;
                self.store_gpr(RBP, &saved)?;
            }
            M::Call => self.call(ins)?,
            m => return unsupported(format!("instruction {m:?} is not supported yet")),
        }
        Ok(())
    }

    fn alu(&mut self, ins: &Instruction, op: Alu) -> R<()> {
        let bits = self.op_bits(ins, 0)?;
        let a = self.read_op(ins, 0, bits)?;
        let b = self.read_op(ins, 1, bits)?;
        let r = self.alu_value(op, &a, &b, bits);
        if !matches!(op, Alu::Cmp | Alu::Test) {
            self.write_op(ins, 0, &r, bits)?;
        }
        Ok(())
    }

    /// Computes `a <op> b` (reading CF for `adc`/`sbb`), sets every flag the operation defines
    /// and returns the result. Also used with the old value of an atomic read-modify-write.
    fn alu_value(&mut self, op: Alu, a: &str, b: &str, bits: u32) -> String {
        let t = ty(bits);
        let r = match op {
            Alu::Add | Alu::Adc => {
                let (r, cf) = if matches!(op, Alu::Add) {
                    let r = self.val(&format!("add {t} {a}, {b}"));
                    let cf = self.val(&format!("icmp ult {t} {r}, {a}"));
                    (r, cf)
                } else {
                    // carry out of a + b + c: r < a, or r == a with a carry in.
                    let c = self.get_flag(Flag::Cf);
                    let cz = self.val(&format!("zext i1 {c} to {t}"));
                    let s = self.val(&format!("add {t} {a}, {b}"));
                    let r = self.val(&format!("add {t} {s}, {cz}"));
                    let lt = self.val(&format!("icmp ult {t} {r}, {a}"));
                    let eq = self.val(&format!("icmp eq {t} {r}, {a}"));
                    let ceq = self.val(&format!("and i1 {c}, {eq}"));
                    let cf = self.val(&format!("or i1 {lt}, {ceq}"));
                    (r, cf)
                };
                // OF: both inputs differ in sign from the result.
                let x = self.val(&format!("xor {t} {a}, {r}"));
                let y = self.val(&format!("xor {t} {b}, {r}"));
                let xy = self.val(&format!("and {t} {x}, {y}"));
                let of = self.msb(&xy, bits);
                let af = self.aux_flag(a, b, &r, bits);
                self.set_flag(Flag::Cf, &cf);
                self.set_flag(Flag::Of, &of);
                self.set_flag(Flag::Af, &af);
                r
            }
            Alu::Sub | Alu::Cmp | Alu::Sbb => {
                let (r, cf) = if matches!(op, Alu::Sbb) {
                    // borrow out of a - b - c: a < b, or a == b with a borrow in.
                    let c = self.get_flag(Flag::Cf);
                    let cz = self.val(&format!("zext i1 {c} to {t}"));
                    let d = self.val(&format!("sub {t} {a}, {b}"));
                    let r = self.val(&format!("sub {t} {d}, {cz}"));
                    let lt = self.val(&format!("icmp ult {t} {a}, {b}"));
                    let eq = self.val(&format!("icmp eq {t} {a}, {b}"));
                    let ceq = self.val(&format!("and i1 {c}, {eq}"));
                    let cf = self.val(&format!("or i1 {lt}, {ceq}"));
                    (r, cf)
                } else {
                    let r = self.val(&format!("sub {t} {a}, {b}"));
                    let cf = self.val(&format!("icmp ult {t} {a}, {b}"));
                    (r, cf)
                };
                // OF: operands differ in sign and the result's sign differs from a.
                let x = self.val(&format!("xor {t} {a}, {b}"));
                let y = self.val(&format!("xor {t} {a}, {r}"));
                let xy = self.val(&format!("and {t} {x}, {y}"));
                let of = self.msb(&xy, bits);
                let af = self.aux_flag(a, b, &r, bits);
                self.set_flag(Flag::Cf, &cf);
                self.set_flag(Flag::Of, &of);
                self.set_flag(Flag::Af, &af);
                r
            }
            Alu::And | Alu::Or | Alu::Xor | Alu::Test => {
                let opcode = match op {
                    Alu::Or => "or",
                    Alu::Xor => "xor",
                    _ => "and",
                };
                let r = self.val(&format!("{opcode} {t} {a}, {b}"));
                // CF and OF are cleared; AF is undefined and left unchanged.
                self.set_flag(Flag::Cf, "false");
                self.set_flag(Flag::Of, "false");
                r
            }
        };
        self.set_zsp(&r, bits);
        r
    }

    fn unary(&mut self, ins: &Instruction) -> R<()> {
        let bits = self.op_bits(ins, 0)?;
        let a = self.read_op(ins, 0, bits)?;
        let r = self.unary_value(ins.mnemonic(), &a, bits);
        self.write_op(ins, 0, &r, bits)
    }

    /// Computes `inc`/`dec`/`neg`/`not` of `a`, sets the flags the operation defines and returns
    /// the result. Also used with the old value of an atomic read-modify-write.
    fn unary_value(&mut self, mnemonic: Mnemonic, a: &str, bits: u32) -> String {
        let t = ty(bits);
        let min = imm(1u64 << (bits - 1), bits);
        let r = match mnemonic {
            Mnemonic::Not => {
                // `not` does not modify any flag.
                return self.val(&format!("xor {t} {a}, -1"));
            }
            Mnemonic::Inc => {
                // CF is preserved by inc/dec.
                let r = self.val(&format!("add {t} {a}, 1"));
                let of = self.val(&format!("icmp eq {t} {r}, {min}"));
                let af = self.aux_flag(a, "1", &r, bits);
                self.set_flag(Flag::Of, &of);
                self.set_flag(Flag::Af, &af);
                r
            }
            Mnemonic::Dec => {
                let r = self.val(&format!("sub {t} {a}, 1"));
                let of = self.val(&format!("icmp eq {t} {a}, {min}"));
                let af = self.aux_flag(a, "1", &r, bits);
                self.set_flag(Flag::Of, &of);
                self.set_flag(Flag::Af, &af);
                r
            }
            _ => {
                // neg
                let r = self.val(&format!("sub {t} 0, {a}"));
                let cf = self.val(&format!("icmp ne {t} {a}, 0"));
                let of = self.val(&format!("icmp eq {t} {a}, {min}"));
                let af = self.aux_flag("0", a, &r, bits);
                self.set_flag(Flag::Cf, &cf);
                self.set_flag(Flag::Of, &of);
                self.set_flag(Flag::Af, &af);
                r
            }
        };
        self.set_zsp(&r, bits);
        r
    }

    /// Shifts and rotates. The count is masked to 5 bits (6 for 64-bit operands); a masked count
    /// of zero leaves all flags unchanged. OF is computed with the 1-bit formula for every count
    /// (it is architecturally undefined for counts > 1); AF is undefined and left unchanged.
    fn shift(&mut self, ins: &Instruction, kind: Shift) -> R<()> {
        let bits = self.op_bits(ins, 0)?;
        let t = ty(bits);
        let count_mask: u64 = if bits == 64 { 63 } else { 31 };
        let a = self.read_op(ins, 0, bits)?;

        // Masked count as i64.
        let count = match ins.op_kind(1) {
            OpKind::Register if ins.op_register(1) == Register::CL => {
                let (cl, _) = self.read_reg(Register::CL)?;
                let masked = self.val(&format!("and i8 {cl}, {count_mask}"));
                self.val(&format!("zext i8 {masked} to i64"))
            }
            kind if is_immediate(kind) => match ins.try_immediate(1) {
                Ok(v) => (v & count_mask).to_string(),
                Err(_) => return unsupported("unreadable shift count"),
            },
            kind => return unsupported(format!("shift count operand {kind:?}")),
        };

        let (r, cf, of) = match kind {
            Shift::Rol | Shift::Ror => {
                // fshl/fshr take the amount modulo the width, like x86 rotates (after masking).
                let amount = if bits == 64 {
                    count.clone()
                } else {
                    self.val(&format!("trunc i64 {count} to {t}"))
                };
                let (name, support) = if kind == Shift::Rol {
                    ("fshl", Support::Fshl(bits))
                } else {
                    ("fshr", Support::Fshr(bits))
                };
                self.support.insert(support);
                let r = self.val(&format!(
                    "call {t} @llvm.{name}.{t}({t} {a}, {t} {a}, {t} {amount})"
                ));
                let msb = self.msb(&r, bits);
                if kind == Shift::Rol {
                    let cf = self.val(&format!("trunc {t} {r} to i1"));
                    let of = self.val(&format!("xor i1 {msb}, {cf}"));
                    (r, cf, of)
                } else {
                    let below = self.val(&format!("lshr {t} {r}, {}", bits - 2));
                    let below = self.val(&format!("trunc {t} {below} to i1"));
                    let of = self.val(&format!("xor i1 {msb}, {below}"));
                    (r, msb, of)
                }
            }
            _ => {
                // count - 1 (masked so a zero count never produces a poison shift amount).
                let prev = self.val(&format!("add i64 {count}, 63"));
                let prev = self.val(&format!("and i64 {prev}, 63"));
                if bits == 64 {
                    match kind {
                        Shift::Shl => {
                            let r = self.val(&format!("shl i64 {a}, {count}"));
                            let back = self.val(&format!("sub i64 64, {count}"));
                            let back = self.val(&format!("and i64 {back}, 63"));
                            let out = self.val(&format!("lshr i64 {a}, {back}"));
                            let cf = self.val(&format!("trunc i64 {out} to i1"));
                            let msb = self.msb(&r, 64);
                            let of = self.val(&format!("xor i1 {msb}, {cf}"));
                            (r, cf, of)
                        }
                        Shift::Shr => {
                            let r = self.val(&format!("lshr i64 {a}, {count}"));
                            let out = self.val(&format!("lshr i64 {a}, {prev}"));
                            let cf = self.val(&format!("trunc i64 {out} to i1"));
                            let of = self.msb(&a, 64);
                            (r, cf, of)
                        }
                        _ => {
                            let r = self.val(&format!("ashr i64 {a}, {count}"));
                            let out = self.val(&format!("ashr i64 {a}, {prev}"));
                            let cf = self.val(&format!("trunc i64 {out} to i1"));
                            (r, cf, "false".to_string())
                        }
                    }
                } else {
                    // Narrow operands are shifted in 64 bits; counts are at most 31 here.
                    let ext = if kind == Shift::Sar { "sext" } else { "zext" };
                    let wide = self.val(&format!("{ext} {t} {a} to i64"));
                    match kind {
                        Shift::Shl => {
                            let res = self.val(&format!("shl i64 {wide}, {count}"));
                            let r = self.val(&format!("trunc i64 {res} to {t}"));
                            let out = self.val(&format!("lshr i64 {res}, {bits}"));
                            let cf = self.val(&format!("trunc i64 {out} to i1"));
                            let msb = self.msb(&r, bits);
                            let of = self.val(&format!("xor i1 {msb}, {cf}"));
                            (r, cf, of)
                        }
                        Shift::Shr => {
                            let res = self.val(&format!("lshr i64 {wide}, {count}"));
                            let r = self.val(&format!("trunc i64 {res} to {t}"));
                            let out = self.val(&format!("lshr i64 {wide}, {prev}"));
                            let cf = self.val(&format!("trunc i64 {out} to i1"));
                            let of = self.msb(&a, bits);
                            (r, cf, of)
                        }
                        _ => {
                            let res = self.val(&format!("ashr i64 {wide}, {count}"));
                            let r = self.val(&format!("trunc i64 {res} to {t}"));
                            let out = self.val(&format!("ashr i64 {wide}, {prev}"));
                            let cf = self.val(&format!("trunc i64 {out} to i1"));
                            (r, cf, "false".to_string())
                        }
                    }
                }
            }
        };

        // Flags only change when the masked count is non-zero.
        let nonzero = self.val(&format!("icmp ne i64 {count}, 0"));
        let mut updates = vec![(Flag::Cf, cf), (Flag::Of, of)];
        if !matches!(kind, Shift::Rol | Shift::Ror) {
            let [zf, sf, pf] = self.zsp(&r, bits);
            updates.extend([(Flag::Zf, zf), (Flag::Sf, sf), (Flag::Pf, pf)]);
        }
        for (flag, new) in updates {
            let old = self.get_flag(flag);
            let v = self.val(&format!("select i1 {nonzero}, i1 {new}, i1 {old}"));
            self.set_flag(flag, &v);
        }
        self.write_op(ins, 0, &r, bits)
    }

    /// Full 64x64 -> 128-bit product as `(lo, hi)`; `signed` adjusts the high half.
    fn mul128(&mut self, a: &str, b: &str, signed: bool) -> (String, String) {
        self.support.insert(Support::Umul128);
        let p = self.val(&format!(
            "call {{ i64, i64 }} @__playgta6_umul128(i64 {a}, i64 {b})"
        ));
        let lo = self.val(&format!("extractvalue {{ i64, i64 }} {p}, 0"));
        let mut hi = self.val(&format!("extractvalue {{ i64, i64 }} {p}, 1"));
        if signed {
            // signed_hi = unsigned_hi - (a < 0 ? b : 0) - (b < 0 ? a : 0)
            let sa = self.val(&format!("ashr i64 {a}, 63"));
            let fa = self.val(&format!("and i64 {sa}, {b}"));
            let sb = self.val(&format!("ashr i64 {b}, 63"));
            let fb = self.val(&format!("and i64 {sb}, {a}"));
            hi = self.val(&format!("sub i64 {hi}, {fa}"));
            hi = self.val(&format!("sub i64 {hi}, {fb}"));
        }
        (lo, hi)
    }

    /// `imul r, r/m` and `imul r, r/m, imm`: truncated product; CF = OF = signed overflow.
    fn imul_truncating(&mut self, ins: &Instruction) -> R<()> {
        let bits = self.op_bits(ins, 0)?;
        let t = ty(bits);
        let (a, b) = if ins.op_count() == 3 {
            let src_bits = self.op_bits(ins, 1)?;
            if src_bits != bits {
                return unsupported("imul operand size");
            }
            (self.read_op(ins, 1, bits)?, self.read_op(ins, 2, bits)?)
        } else {
            (self.read_op(ins, 0, bits)?, self.read_op(ins, 1, bits)?)
        };
        let (r, overflow) = if bits == 64 {
            let (lo, hi) = self.mul128(&a, &b, true);
            let sign = self.val(&format!("ashr i64 {lo}, 63"));
            let ov = self.val(&format!("icmp ne i64 {hi}, {sign}"));
            (lo, ov)
        } else {
            let wa = self.val(&format!("sext {t} {a} to i64"));
            let wb = self.val(&format!("sext {t} {b} to i64"));
            let p = self.val(&format!("mul i64 {wa}, {wb}"));
            let r = self.val(&format!("trunc i64 {p} to {t}"));
            let back = self.val(&format!("sext {t} {r} to i64"));
            let ov = self.val(&format!("icmp ne i64 {back}, {p}"));
            (r, ov)
        };
        // SF, ZF, AF and PF are undefined and left unchanged.
        self.set_flag(Flag::Cf, &overflow);
        self.set_flag(Flag::Of, &overflow);
        self.write_op(ins, 0, &r, bits)
    }

    /// One-operand `mul`/`imul`: `rdx:rax = rax * src` (AX = AL * src for bytes).
    fn mul_widening(&mut self, ins: &Instruction, signed: bool) -> R<()> {
        let bits = self.op_bits(ins, 0)?;
        let t = ty(bits);
        let src = self.read_op(ins, 0, bits)?;
        let (acc, dst_hi) = match bits {
            8 => (Register::AL, Register::AH),
            16 => (Register::AX, Register::DX),
            32 => (Register::EAX, Register::EDX),
            _ => (Register::RAX, Register::RDX),
        };
        let (a, _) = self.read_reg(acc)?;
        let overflow = if bits == 64 {
            let (lo, hi) = self.mul128(&a, &src, signed);
            let ov = if signed {
                let sign = self.val(&format!("ashr i64 {lo}, 63"));
                self.val(&format!("icmp ne i64 {hi}, {sign}"))
            } else {
                self.val(&format!("icmp ne i64 {hi}, 0"))
            };
            self.write_reg(Register::RAX, &lo, 64)?;
            self.write_reg(Register::RDX, &hi, 64)?;
            ov
        } else {
            let ext = if signed { "sext" } else { "zext" };
            let wa = self.val(&format!("{ext} {t} {a} to i64"));
            let wb = self.val(&format!("{ext} {t} {src} to i64"));
            let p = self.val(&format!("mul i64 {wa}, {wb}"));
            let lo = self.val(&format!("trunc i64 {p} to {t}"));
            let ov = if signed {
                let back = self.val(&format!("sext {t} {lo} to i64"));
                self.val(&format!("icmp ne i64 {back}, {p}"))
            } else {
                let high = self.val(&format!("lshr i64 {p}, {bits}"));
                self.val(&format!("icmp ne i64 {high}, 0"))
            };
            if bits == 8 {
                let ax = self.val(&format!("trunc i64 {p} to i16"));
                self.write_reg(Register::AX, &ax, 16)?;
            } else {
                let high = self.val(&format!("lshr i64 {p}, {bits}"));
                let hi = self.val(&format!("trunc i64 {high} to {t}"));
                self.write_reg(acc, &lo, bits)?;
                self.write_reg(dst_hi, &hi, bits)?;
            }
            ov
        };
        // SF, ZF, AF and PF are undefined and left unchanged.
        self.set_flag(Flag::Cf, &overflow);
        self.set_flag(Flag::Of, &overflow);
        Ok(())
    }

    /// `div`/`idiv`: divides `rdx:rax` (AX for bytes) by the operand. Division by zero or a
    /// quotient that does not fit raises `#DE` through `__playgta6_divide_error`. All flags are
    /// undefined and left unchanged.
    fn divide(&mut self, ins: &Instruction, signed: bool) -> R<()> {
        let bits = self.op_bits(ins, 0)?;
        let t = ty(bits);
        let divisor = self.read_op(ins, 0, bits)?;
        let ext = if signed { "sext" } else { "zext" };

        let (hi, lo, d) = if bits == 64 {
            let (hi, _) = self.read_reg(Register::RDX)?;
            let (lo, _) = self.read_reg(Register::RAX)?;
            (hi, lo, divisor)
        } else {
            // Assemble the 2N-bit dividend in an i64, then extend it to 128 bits.
            let dividend = if bits == 8 {
                let (ax, _) = self.read_reg(Register::AX)?;
                self.val(&format!("{ext} i16 {ax} to i64"))
            } else {
                let (hreg, lreg) = if bits == 16 {
                    (Register::DX, Register::AX)
                } else {
                    (Register::EDX, Register::EAX)
                };
                let (h, _) = self.read_reg(hreg)?;
                let (l, _) = self.read_reg(lreg)?;
                let h = self.val(&format!("zext {t} {h} to i64"));
                let l = self.val(&format!("zext {t} {l} to i64"));
                let h = self.val(&format!("shl i64 {h}, {bits}"));
                let joined = self.val(&format!("or i64 {h}, {l}"));
                if signed && bits == 16 {
                    let narrow = self.val(&format!("trunc i64 {joined} to i32"));
                    self.val(&format!("sext i32 {narrow} to i64"))
                } else {
                    joined
                }
            };
            let hi = if signed {
                self.val(&format!("ashr i64 {dividend}, 63"))
            } else {
                "0".to_string()
            };
            let d = self.val(&format!("{ext} {t} {divisor} to i64"));
            (hi, dividend, d)
        };

        let helper = if signed {
            self.support.insert(Support::Sdiv128);
            "__playgta6_sdiv128"
        } else {
            self.support.insert(Support::Udiv128);
            "__playgta6_udiv128"
        };
        let res = self.val(&format!(
            "call {{ i64, i64, i1 }} @{helper}(i64 {hi}, i64 {lo}, i64 {d})"
        ));
        let q = self.val(&format!("extractvalue {{ i64, i64, i1 }} {res}, 0"));
        let r = self.val(&format!("extractvalue {{ i64, i64, i1 }} {res}, 1"));
        let mut fault = self.val(&format!("extractvalue {{ i64, i64, i1 }} {res}, 2"));
        let (q, r) = if bits == 64 {
            (q, r)
        } else {
            let qn = self.val(&format!("trunc i64 {q} to {t}"));
            let rn = self.val(&format!("trunc i64 {r} to {t}"));
            let back = self.val(&format!("{ext} {t} {qn} to i64"));
            let range = self.val(&format!("icmp ne i64 {back}, {q}"));
            fault = self.val(&format!("or i1 {fault}, {range}"));
            (qn, rn)
        };

        let de = self.new_label("de");
        let ok = self.new_label("cont");
        self.emit(&format!("br i1 {fault}, label %{de}, label %{ok}"));
        self.start_block(&de);
        self.sync_out();
        self.emit(&format!(
            "call void @{HELPER_DIVIDE_ERROR}(ptr %state, i64 {})",
            imm(ins.ip(), 64)
        ));
        self.emit("unreachable");
        self.start_block(&ok);

        let (qreg, rreg) = match bits {
            8 => (Register::AL, Register::AH),
            16 => (Register::AX, Register::DX),
            32 => (Register::EAX, Register::EDX),
            _ => (Register::RAX, Register::RDX),
        };
        self.write_reg(qreg, &q, bits)?;
        self.write_reg(rreg, &r, bits)
    }

    // ----- control flow --------------------------------------------------------------------------

    /// Reads the 64-bit target of an indirect `call`/`jmp`.
    fn indirect_target(&mut self, ins: &Instruction) -> R<String> {
        if self.op_bits(ins, 0)? != 64 {
            return unsupported("indirect branch operand size");
        }
        self.read_op(ins, 0, 64)
    }

    fn call(&mut self, ins: &Instruction) -> R<()> {
        if !ins.is_call_near() && !ins.is_call_near_indirect() {
            return unsupported("far call");
        }
        let return_address = imm(ins.next_ip(), 64);
        if ins.op_kind(0) == OpKind::NearBranch64 {
            let target = ins.near_branch_target();
            self.push(&return_address, 64)?;
            if self.faulting_calls.contains(&ins.ip()) {
                // The target is not executable guest code: the call faults on arrival. Code after
                // it is unreachable, so continue in a fresh block without predecessors.
                self.trap(target);
                let dead = self.new_label("dead");
                self.start_block(&dead);
                return Ok(());
            }
            self.sync_out();
            self.emit(&format!(
                "call void @{}(ptr %state)",
                function_symbol(target)
            ));
            self.callees.push(target);
        } else {
            // The target is read before the return address is pushed (`call [rsp+8]`).
            let target = self.indirect_target(ins)?;
            self.push(&return_address, 64)?;
            self.sync_out();
            self.emit(&format!(
                "call void @{HELPER_INDIRECT_CALL}(ptr %state, i64 {target})"
            ));
        }
        self.sync_in();
        Ok(())
    }

    fn ret(&mut self, ins: &Instruction) -> R<()> {
        let extra = match ins.code() {
            Code::Retnq => 0,
            Code::Retnq_imm16 => u64::from(ins.immediate16()),
            _ => return unsupported("far or 16/32-bit return"),
        };
        // Pop the return address (not checked: the host call stack already knows where to return).
        let rsp = self.load_gpr(RSP)?;
        let new_rsp = self.val(&format!("add i64 {rsp}, {}", 8 + extra));
        self.store_gpr(RSP, &new_rsp)?;
        self.sync_out();
        self.emit("ret void");
        Ok(())
    }

    /// `jmp reg` / `jmp [mem]`. A resolved switch (`Edge::JumpTable` successors) becomes an LLVM
    /// `switch` on the computed guest target over the case blocks of this function; every other
    /// value (an incomplete table proof, a patched table) and unresolved indirect jumps leave the
    /// function through the runtime dispatcher as a tail call.
    fn indirect_jump(&mut self, block: &cfg::BasicBlock, ins: &Instruction) -> R<()> {
        if !ins.is_jmp_near_indirect() {
            return unsupported("far or non-indirect jump");
        }
        let target = self.indirect_target(ins)?;
        let mut cases: Vec<u64> = block
            .successors
            .iter()
            .filter_map(|edge| match *edge {
                Edge::JumpTable(case) => Some(case),
                _ => None,
            })
            .collect();
        cases.sort_unstable();
        cases.dedup();
        if !cases.is_empty() {
            let mut arms = String::new();
            for case in cases {
                if !self.func.blocks.contains_key(&case) {
                    return Err(InsnError::Cfg(LiftError::UnknownSuccessor {
                        block: block.start,
                        target: case,
                    }));
                }
                let _ = write!(arms, " i64 {}, label %{}", imm(case, 64), block_label(case));
            }
            let dispatch = self.new_label("dispatch");
            self.emit(&format!("switch i64 {target}, label %{dispatch} [{arms} ]"));
            self.start_block(&dispatch);
        }
        self.sync_out();
        self.emit(&format!(
            "tail call void @{HELPER_INDIRECT_CALL}(ptr %state, i64 {target})"
        ));
        self.emit("ret void");
        Ok(())
    }

    /// `jcc`: the condition comes from the instruction, the targets from the `[taken, not taken]`
    /// successor edges (either may be a tail call into another function).
    fn branch(&mut self, block: &cfg::BasicBlock, ins: &Instruction) -> R<()> {
        let c = if ins.is_jcc_short_or_near() {
            self.condition(ins.condition_code())?
        } else {
            // `loop*` / `jrcxz` / `jecxz` (anything else is reported as unsupported).
            self.counter_branch_condition(ins)?
        };
        let taken = self.successor(block, block.successors.first())?;
        let not_taken = self.successor(block, block.successors.get(1))?;
        self.emit(&format!("br i1 {c}, label %{taken}, label %{not_taken}"));
        Ok(())
    }
}

/// `{lo, hi} = a * b` (unsigned 64x64 -> 128) built from 32-bit halves, so no `i128` multiply
/// (and therefore no compiler-rt libcall) is needed on wasm64. The empty inline-asm copy of
/// `a`'s high half hides the pattern from LLVM's multiply-high recognition (LLVM 20+), which
/// would otherwise turn a high-half-only use (`mul` for a division by a constant, `mulx` with
/// equal destinations) back into an `i128` multiply, i.e. a `__multi3` import.
pub(super) const UMUL128_IR: &str = "\
define internal { i64, i64 } @__playgta6_umul128(i64 %a, i64 %b) {
entry:
  %a0 = and i64 %a, 4294967295
  %a1.raw = lshr i64 %a, 32
  %a1 = call i64 asm \"\", \"=r,0\"(i64 %a1.raw)
  %b0 = and i64 %b, 4294967295
  %b1 = lshr i64 %b, 32
  %p00 = mul i64 %a0, %b0
  %p01 = mul i64 %a0, %b1
  %p10 = mul i64 %a1, %b0
  %p11 = mul i64 %a1, %b1
  %p00h = lshr i64 %p00, 32
  %p01l = and i64 %p01, 4294967295
  %p10l = and i64 %p10, 4294967295
  %mid0 = add i64 %p00h, %p01l
  %mid = add i64 %mid0, %p10l
  %midl = shl i64 %mid, 32
  %p00l = and i64 %p00, 4294967295
  %lo = or i64 %midl, %p00l
  %p01h = lshr i64 %p01, 32
  %p10h = lshr i64 %p10, 32
  %midh = lshr i64 %mid, 32
  %hi0 = add i64 %p11, %p01h
  %hi1 = add i64 %hi0, %p10h
  %hi = add i64 %hi1, %midh
  %r0 = insertvalue { i64, i64 } poison, i64 %lo, 0
  %r1 = insertvalue { i64, i64 } %r0, i64 %hi, 1
  ret { i64, i64 } %r1
}
";

/// `{q, r, fault} = (hi:lo) / d` (unsigned 128 / 64). `fault` is set when `d == 0` or the
/// quotient does not fit in 64 bits (`hi >= d`). Restoring division, no `i128` libcalls.
pub(super) const UDIV128_IR: &str = "\
define internal { i64, i64, i1 } @__playgta6_udiv128(i64 %hi, i64 %lo, i64 %d) {
entry:
  %fault = icmp uge i64 %hi, %d
  br i1 %fault, label %fail, label %ok
fail:
  ret { i64, i64, i1 } { i64 0, i64 0, i1 true }
ok:
  %small = icmp eq i64 %hi, 0
  br i1 %small, label %fast, label %loop
fast:
  %fq = udiv i64 %lo, %d
  %fr = urem i64 %lo, %d
  %f0 = insertvalue { i64, i64, i1 } { i64 0, i64 0, i1 false }, i64 %fq, 0
  %f1 = insertvalue { i64, i64, i1 } %f0, i64 %fr, 1
  ret { i64, i64, i1 } %f1
loop:
  %i = phi i32 [ 0, %ok ], [ %i.next, %loop ]
  %h = phi i64 [ %hi, %ok ], [ %h.next, %loop ]
  %l = phi i64 [ %lo, %ok ], [ %l.next, %loop ]
  %carry = icmp slt i64 %h, 0
  %h.shl = shl i64 %h, 1
  %l.top = lshr i64 %l, 63
  %h2 = or i64 %h.shl, %l.top
  %l2 = shl i64 %l, 1
  %ge = icmp uge i64 %h2, %d
  %take = or i1 %carry, %ge
  %h.sub = sub i64 %h2, %d
  %h.next = select i1 %take, i64 %h.sub, i64 %h2
  %bit = zext i1 %take to i64
  %l.next = or i64 %l2, %bit
  %i.next = add i32 %i, 1
  %done = icmp eq i32 %i.next, 64
  br i1 %done, label %exit, label %loop
exit:
  %e0 = insertvalue { i64, i64, i1 } { i64 0, i64 0, i1 false }, i64 %l.next, 0
  %e1 = insertvalue { i64, i64, i1 } %e0, i64 %h.next, 1
  ret { i64, i64, i1 } %e1
}
";

/// `{q, r, fault} = (hi:lo) / d` (signed 128 / 64, truncating). The remainder has the sign of the
/// dividend; `fault` is set for `d == 0` or a quotient outside the i64 range.
pub(super) const SDIV128_IR: &str = "\
define internal { i64, i64, i1 } @__playgta6_sdiv128(i64 %hi, i64 %lo, i64 %d) {
entry:
  %nneg = icmp slt i64 %hi, 0
  %dneg = icmp slt i64 %d, 0
  %lo.neg = sub i64 0, %lo
  %lo.zero = icmp eq i64 %lo, 0
  %hi.not = xor i64 %hi, -1
  %hi.inc = zext i1 %lo.zero to i64
  %hi.neg = add i64 %hi.not, %hi.inc
  %ahi = select i1 %nneg, i64 %hi.neg, i64 %hi
  %alo = select i1 %nneg, i64 %lo.neg, i64 %lo
  %d.neg = sub i64 0, %d
  %ad = select i1 %dneg, i64 %d.neg, i64 %d
  %u = call { i64, i64, i1 } @__playgta6_udiv128(i64 %ahi, i64 %alo, i64 %ad)
  %uq = extractvalue { i64, i64, i1 } %u, 0
  %ur = extractvalue { i64, i64, i1 } %u, 1
  %ufault = extractvalue { i64, i64, i1 } %u, 2
  %qneg = xor i1 %nneg, %dneg
  %limit = select i1 %qneg, i64 -9223372036854775808, i64 9223372036854775807
  %range = icmp ugt i64 %uq, %limit
  %fault = or i1 %ufault, %range
  %q.neg = sub i64 0, %uq
  %q = select i1 %qneg, i64 %q.neg, i64 %uq
  %r.neg = sub i64 0, %ur
  %r = select i1 %nneg, i64 %r.neg, i64 %ur
  %s0 = insertvalue { i64, i64, i1 } poison, i64 %q, 0
  %s1 = insertvalue { i64, i64, i1 } %s0, i64 %r, 1
  %s2 = insertvalue { i64, i64, i1 } %s1, i1 %fault, 2
  ret { i64, i64, i1 } %s2
}
";
