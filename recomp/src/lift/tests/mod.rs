//! Lifter tests. Synthetic x86-64 code is assembled with iced-x86's `code_asm` (dev-only feature)
//! or given as raw bytes, mapped into a [`CodeImage`] and run through R2's real
//! [`discover`](crate::discover::discover) before lifting, so the tests exercise the actual
//! `bytes -> CodeImage -> discover -> lift_module` pipeline.

mod exec;
mod exec_integer;
mod exec_memory;
mod exec_native;
mod exec_simd;
mod exec_unwind;
mod exec_x87;
mod flags_across_calls;
mod registers_across_calls;
mod state_boundary;
mod synced_store_backs;
#[cfg(target_arch = "x86_64")]
mod host_diff;
#[cfg(target_arch = "x86_64")]
mod host_diff_avx;
#[cfg(target_arch = "x86_64")]
mod host_diff_crypto;
#[cfg(target_arch = "x86_64")]
mod host_diff_integer;
#[cfg(target_arch = "x86_64")]
mod host_diff_simd;
#[cfg(target_arch = "x86_64")]
mod host_diff_simd_int;
#[cfg(target_arch = "x86_64")]
mod host_diff_strcmp;
mod ir;
mod ir_budget;
mod ir_integer;
mod ir_simd;
mod memory_bench;
mod memory_models;
#[cfg(target_arch = "x86_64")]
mod simd_diff;

use std::collections::BTreeMap;

use iced_x86::code_asm::CodeAssembler;
use iced_x86::{IcedError, Instruction};

use crate::cfg;
use crate::code_image::{CodeImage, CodeSection};
use crate::discover::discover;

/// Synthetic code at fixed addresses. Every piece is its own executable section and (unless added
/// with [`Program::code`]) a discovery entry point.
#[derive(Default)]
pub(super) struct Program {
    base: Option<u64>,
    pieces: Vec<(u64, Vec<u8>)>,
    data: Vec<(u64, Vec<u8>)>,
    entries: Vec<u64>,
    no_return: Vec<u64>,
    resume: Vec<(u64, u64)>,
}

impl Program {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `bytes` at `va` as a function entry.
    pub fn bytes(mut self, va: u64, bytes: &[u8]) -> Self {
        self.entries.push(va);
        self.code(va, bytes)
    }

    /// Adds `bytes` at `va` without making it an entry point (e.g. a shared tail block).
    pub fn code(mut self, va: u64, bytes: &[u8]) -> Self {
        self.pieces.push((va, bytes.to_vec()));
        self
    }

    /// Assembles a function at `va` with `code_asm`.
    pub fn asm(
        self,
        va: u64,
        build: impl FnOnce(&mut CodeAssembler) -> Result<(), IcedError>,
    ) -> Self {
        let bytes = assemble(va, build);
        self.bytes(va, &bytes)
    }

    /// Sets the image base (`__ImageBase`, the base of RVAs); defaults to the lowest piece.
    pub fn base(mut self, va: u64) -> Self {
        self.base = Some(va);
        self
    }

    /// Adds a non-executable section (e.g. a jump table) at `va`.
    pub fn data(mut self, va: u64, bytes: &[u8]) -> Self {
        self.data.push((va, bytes.to_vec()));
        self
    }

    /// Marks a call target as never returning (like `ExitProcess`).
    pub fn no_return(mut self, va: u64) -> Self {
        self.no_return.push(va);
        self
    }

    /// Makes `target` a resume target of the function at `owner` (like an EH continuation).
    pub fn resume(mut self, owner: u64, target: u64) -> Self {
        self.resume.push((owner, target));
        self
    }

    /// Runs function discovery over the program.
    pub fn discover(&self) -> BTreeMap<u64, cfg::Function> {
        let lowest = self.pieces.iter().map(|(va, _)| *va).min().unwrap_or(0);
        let base = self.base.unwrap_or(lowest);
        let code = self
            .pieces
            .iter()
            .map(|(va, b)| CodeSection::new(*va, b, true));
        let data = self
            .data
            .iter()
            .map(|(va, b)| CodeSection::new(*va, b, false));
        let sections = code.chain(data).collect();
        let mut image = CodeImage::new(base, sections, self.entries.clone());
        image.no_return = self.no_return.clone();
        image.eh.resume_targets = self.resume.clone();
        image.eh.resume_targets.sort_unstable();
        discover(&image).functions
    }

    /// The discovered function at `entry`.
    pub fn function(&self, entry: u64) -> cfg::Function {
        self.discover()
            .remove(&entry)
            .unwrap_or_else(|| panic!("no function discovered at {entry:#x}"))
    }
}

/// Assembles code at `va`.
pub(super) fn assemble(
    va: u64,
    build: impl FnOnce(&mut CodeAssembler) -> Result<(), IcedError>,
) -> Vec<u8> {
    let mut a = CodeAssembler::new(64).expect("64-bit assembler");
    build(&mut a).expect("test code assembles");
    a.assemble(va).expect("test code encodes")
}

/// Assembles a single function at `entry` and discovers it.
pub(super) fn asm_function(
    entry: u64,
    build: impl FnOnce(&mut CodeAssembler) -> Result<(), IcedError>,
) -> cfg::Function {
    Program::new().asm(entry, build).function(entry)
}

/// Encodes prebuilt instructions (e.g. RIP-relative ones) at `va`.
pub(super) fn encode(va: u64, instructions: &[Instruction]) -> Vec<u8> {
    let mut a = CodeAssembler::new(64).expect("64-bit assembler");
    for ins in instructions {
        a.add_instruction(*ins).expect("instruction accepted");
    }
    a.assemble(va).expect("instructions encode")
}

/// Text RVA of the [`switch_code`] fixture (relative to the program's image base).
pub(super) const SWITCH_TEXT_RVA: u32 = 0x1000;
/// RVA of the fixture's 32-bit jump table.
pub(super) const SWITCH_TABLE_RVA: u32 = 0x2000;
/// Case block offsets of the fixture; the last one is case 3 and also the default.
pub(super) const SWITCH_CASES: [u32; 4] = [0x1b, 0x21, 0x27, 0x2d];

/// MSVC-style dense `switch (ecx)` that discovery resolves from its jump table: cases 0, 1, 2
/// return 10, 11, 12; case 3 and every value above 3 return 99.
pub(super) fn switch_code() -> Vec<u8> {
    let image_base_disp = -i64::from(SWITCH_TEXT_RVA + 8 + 7) as i32;
    [
        &[0x83, 0xf9, 0x03][..], // 0: cmp ecx, 3
        &[0x77, 0x28],           // 3: ja 0x2d
        &[0x48, 0x63, 0xc1],     // 5: movsxd rax, ecx
        &[0x48, 0x8d, 0x15],     // 8: lea rdx, [rip+__ImageBase]
        &image_base_disp.to_le_bytes(),
        &[0x8b, 0x8c, 0x82], // f: mov ecx, [rdx+rax*4+table_rva]
        &SWITCH_TABLE_RVA.to_le_bytes(),
        &[0x48, 0x01, 0xd1],        // 16: add rcx, rdx
        &[0xff, 0xe1],              // 19: jmp rcx
        &[0xb8, 10, 0, 0, 0, 0xc3], // 1b: mov eax, 10 ; ret
        &[0xb8, 11, 0, 0, 0, 0xc3], // 21: mov eax, 11 ; ret
        &[0xb8, 12, 0, 0, 0, 0xc3], // 27: mov eax, 12 ; ret
        &[0xb8, 99, 0, 0, 0, 0xc3], // 2d: mov eax, 99 ; ret
    ]
    .concat()
}

/// The fixture's jump table: one 32-bit RVA per case.
pub(super) fn switch_table() -> Vec<u8> {
    SWITCH_CASES
        .iter()
        .flat_map(|case| (SWITCH_TEXT_RVA + case).to_le_bytes())
        .collect()
}

/// [`switch_code`] at `base + SWITCH_TEXT_RVA` plus its table in a data section.
pub(super) fn switch_program(base: u64) -> Program {
    Program::new()
        .base(base)
        .bytes(base + u64::from(SWITCH_TEXT_RVA), &switch_code())
        .data(base + u64::from(SWITCH_TABLE_RVA), &switch_table())
}
