//! Differential tests against the host CPU (x86-64 hosts only) for the string, bit, BMI,
//! exchange/atomic, rotate-through-carry, double-shift, flag and miscellaneous integer
//! instructions.
//!
//! The *same machine code* runs on both sides: natively (copied into executable memory and
//! called through a trampoline that loads ten GPRs, RFLAGS and DF from a context block and
//! stores them back) and lifted to wasm64 under node. Registers, the memory buffer, DF and every
//! architecturally defined flag are compared; pointer registers are compared as offsets into
//! the buffer. Forms the host CPU lacks (BMI2, ADX, ...) are skipped with a printed note.

use iced_x86::code_asm::*;
use iced_x86::{Code, IcedError, Instruction, Register};

use super::exec::{
    CODE_BASE, Case, DATA_ADDR, R8, R9, R10, R11, RAX, RBX, RCX, RDI, RDX, RSI, RSP, STACK_TOP,
    STATE_ADDR, build, exec_or_skip, run, shared_memory_or_skip,
};
use super::ir::Lcg;
use super::{Program, assemble, encode};
use crate::lift::STATE_FLAGS_OFFSET;

/// Guest GPR index of each trampoline register: `rax rcx rdx rbx rsi rdi r8 r9 r10 r11`.
const REGS: [usize; 10] = [RAX, RCX, RDX, RBX, RSI, RDI, R8, R9, R10, R11];
const REG_NAMES: [&str; 10] = [
    "rax", "rcx", "rdx", "rbx", "rsi", "rdi", "r8", "r9", "r10", "r11",
];
// Trampoline register slots.
const AX: usize = 0;
const CX: usize = 1;
const DX: usize = 2;
const BX: usize = 3;
const SI: usize = 4;
const DI: usize = 5;
const X8: usize = 6;
const X9: usize = 7;
const FLAG_NAMES: [&str; 6] = ["CF", "PF", "AF", "ZF", "SF", "OF"];
const FLAG_BITS: [u32; 6] = [0, 2, 4, 6, 7, 11];
const DF_ADDR: u64 = STATE_ADDR + STATE_FLAGS_OFFSET + 6;

/// Native execution of a code snippet (`<instruction>; ret`) with a full register context.
mod native {
    use std::arch::asm;
    use std::ffi::c_void;

    #[repr(C)]
    #[derive(Default)]
    pub struct Ctx {
        /// rax rcx rdx rbx rsi rdi r8 r9 r10 r11
        pub regs: [u64; 10],
        pub rflags: u64,
    }

    /// Read/write/execute memory for the snippets (never freed; the test process exits).
    pub fn exec_memory(len: usize) -> Option<*mut u8> {
        #[cfg(windows)]
        let ptr = {
            #[link(name = "kernel32")]
            unsafe extern "system" {
                fn VirtualAlloc(
                    addr: *mut c_void,
                    size: usize,
                    kind: u32,
                    protect: u32,
                ) -> *mut c_void;
            }
            // MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE
            // SAFETY: plain allocation call.
            unsafe { VirtualAlloc(std::ptr::null_mut(), len, 0x3000, 0x40) }
        };
        #[cfg(target_os = "linux")]
        let ptr = {
            unsafe extern "C" {
                fn mmap(
                    addr: *mut c_void,
                    len: usize,
                    prot: i32,
                    flags: i32,
                    fd: i32,
                    off: i64,
                ) -> *mut c_void;
            }
            // PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS
            // SAFETY: plain allocation call.
            let p = unsafe { mmap(std::ptr::null_mut(), len, 7, 0x22, -1, 0) };
            if p as isize == -1 {
                std::ptr::null_mut()
            } else {
                p
            }
        };
        #[cfg(not(any(windows, target_os = "linux")))]
        let ptr: *mut c_void = {
            let _ = len;
            std::ptr::null_mut()
        };
        (!ptr.is_null()).then_some(ptr.cast())
    }

    /// Calls `code` with the registers and RFLAGS from `ctx` and stores them back.
    ///
    /// # Safety
    /// `code` must be executable, end in `ret`, keep rsp/rbp/r12..r15 intact and only touch
    /// memory the caller owns.
    pub unsafe fn call(code: *const u8, ctx: &mut Ctx) {
        // SAFETY: see the function contract; rbx is saved manually (it cannot be an operand),
        // DF is cleared again before leaving the block.
        unsafe {
            asm!(
                "push rbx",
                "push r12",
                "mov rax, [r12]",
                "mov rcx, [r12 + 8]",
                "mov rdx, [r12 + 16]",
                "mov rbx, [r12 + 24]",
                "mov rsi, [r12 + 32]",
                "mov rdi, [r12 + 40]",
                "mov r8, [r12 + 48]",
                "mov r9, [r12 + 56]",
                "mov r10, [r12 + 64]",
                "mov r11, [r12 + 72]",
                "push qword ptr [r12 + 80]",
                "popfq",
                "call r13",
                "pushfq",
                "mov r12, [rsp + 8]",
                "pop qword ptr [r12 + 80]",
                "mov [r12], rax",
                "mov [r12 + 8], rcx",
                "mov [r12 + 16], rdx",
                "mov [r12 + 24], rbx",
                "mov [r12 + 32], rsi",
                "mov [r12 + 40], rdi",
                "mov [r12 + 48], r8",
                "mov [r12 + 56], r9",
                "mov [r12 + 64], r10",
                "mov [r12 + 72], r11",
                "pop r12",
                "pop rbx",
                "cld",
                in("r12") ctx as *mut Ctx,
                in("r13") code,
                out("rax") _, out("rcx") _, out("rdx") _, out("rsi") _, out("rdi") _,
                out("r8") _, out("r9") _, out("r10") _, out("r11") _,
            );
        }
    }
}

/// One test input. Registers whose bit is set in `ptrs` hold an offset into `mem`.
#[derive(Clone, Debug)]
struct Input {
    regs: [u64; 10],
    ptrs: u16,
    flags: [bool; 6],
    df: bool,
    mem: Vec<u8>,
}

/// What an instruction leaves undefined for an input (not compared).
#[derive(Clone, Copy, Default)]
struct Undef {
    flags: [bool; 6],
    regs: u16,
}

const CF: usize = 0;
const PF: usize = 1;
const AF: usize = 2;
const SF: usize = 4;
const OF: usize = 5;

fn undef_flags(list: &[usize]) -> Undef {
    let mut u = Undef::default();
    for &f in list {
        u.flags[f] = true;
    }
    u
}

struct Op {
    name: String,
    code: Vec<u8>,
    feature: Option<&'static str>,
    cases: Vec<Input>,
    undefined: Box<dyn Fn(&Input) -> Undef>,
}

struct Results {
    regs: [u64; 10],
    flags: [bool; 6],
    df: bool,
    mem: Vec<u8>,
}

fn host_has(feature: &str) -> bool {
    match feature {
        "bmi1" => is_x86_feature_detected!("bmi1"),
        "bmi2" => is_x86_feature_detected!("bmi2"),
        "lzcnt" => is_x86_feature_detected!("lzcnt"),
        "popcnt" => is_x86_feature_detected!("popcnt"),
        "adx" => is_x86_feature_detected!("adx"),
        "movbe" => is_x86_feature_detected!("movbe"),
        "sse4.2" => is_x86_feature_detected!("sse4.2"),
        "cmpxchg16b" => is_x86_feature_detected!("cmpxchg16b"),
        _ => false,
    }
}

fn run_native(code: *mut u8, input: &Input) -> Results {
    // 16-byte aligned buffer (cmpxchg16b).
    let mut buf = vec![0u128; input.mem.len().div_ceil(16) + 1];
    let base = buf.as_mut_ptr().cast::<u8>();
    // SAFETY: `buf` holds at least `mem.len()` bytes.
    unsafe { std::ptr::copy_nonoverlapping(input.mem.as_ptr(), base, input.mem.len()) };
    let mut ctx = native::Ctx::default();
    for i in 0..10 {
        ctx.regs[i] = if input.ptrs >> i & 1 == 1 {
            (base as u64).wrapping_add(input.regs[i])
        } else {
            input.regs[i]
        };
    }
    ctx.rflags = 0x202 | (u64::from(input.df) << 10);
    for k in 0..6 {
        ctx.rflags |= u64::from(input.flags[k]) << FLAG_BITS[k];
    }
    // SAFETY: the snippet is one instruction plus `ret`; the inputs keep every memory access
    // inside `buf`.
    unsafe { native::call(code, &mut ctx) };
    let mut regs = ctx.regs;
    for (i, r) in regs.iter_mut().enumerate() {
        if input.ptrs >> i & 1 == 1 {
            *r = r.wrapping_sub(base as u64);
        }
    }
    let mut mem = vec![0u8; input.mem.len()];
    // SAFETY: as above.
    unsafe { std::ptr::copy_nonoverlapping(base, mem.as_mut_ptr(), mem.len()) };
    Results {
        regs,
        flags: std::array::from_fn(|k| ctx.rflags >> FLAG_BITS[k] & 1 == 1),
        df: ctx.rflags >> 10 & 1 == 1,
        mem,
    }
}

fn lifted_case(va: u64, input: &Input) -> Case {
    let mut c = Case::new(va);
    for i in 0..10 {
        c.regs[REGS[i]] = if input.ptrs >> i & 1 == 1 {
            DATA_ADDR + input.regs[i]
        } else {
            input.regs[i]
        };
    }
    c.regs[RSP] = STACK_TOP;
    c.flags = input.flags;
    c.out = REGS.to_vec();
    c.out.push(RSP);
    let mut c = c.mem(DF_ADDR, &[u8::from(input.df)]);
    if !input.mem.is_empty() {
        c = c
            .mem(DATA_ADDR, &input.mem)
            .read(DATA_ADDR, input.mem.len());
    }
    c.read(DF_ADDR, 1)
}

/// Runs every op natively and lifted (one module) and returns the mismatch descriptions.
fn differential(label: &str, ops: &[Op]) -> Vec<String> {
    let mut code_mem = native::exec_memory(ops.len() * 64).expect("executable memory");
    let mut cases = Vec::new();
    let mut meta = Vec::new();
    let mut pieces = Vec::new();
    for (i, op) in ops.iter().enumerate() {
        assert!(op.code.len() < 63, "{}", op.name);
        let va = CODE_BASE + i as u64 * 0x40;
        let mut bytes = op.code.clone();
        bytes.push(0xC3);
        // SAFETY: slot `i` of the executable buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), code_mem.add(i * 64), bytes.len());
        }
        pieces.push((va, bytes));
        for (k, input) in op.cases.iter().enumerate() {
            cases.push(lifted_case(va, input));
            meta.push((i, k));
        }
    }
    let program = pieces
        .iter()
        .fold(Program::new(), |p, (va, bytes)| p.bytes(*va, bytes));
    let functions = program.discover();
    assert_eq!(functions.len(), ops.len());
    for f in functions.values() {
        let lifted = crate::lift::lift_function(f).expect("lifts");
        assert!(
            lifted.unsupported.is_empty(),
            "{label}: {:?}",
            lifted.unsupported
        );
    }
    let built = build(label, functions.values());
    let outcomes = run(&built, &cases);

    let mut mismatches = Vec::new();
    for (o, &(i, k)) in outcomes.iter().zip(&meta) {
        let op = &ops[i];
        let input = &op.cases[k];
        // SAFETY: slot `i`, written above.
        let host = run_native(unsafe { code_mem.add(i * 64) }, input);
        let label = format!("{} case {k} {:x?}", op.name, input.regs);
        if !o.error.is_empty() {
            mismatches.push(format!("{label}: trapped: {}", o.error));
            continue;
        }
        let undef = (op.undefined)(input);
        for r in 0..10 {
            if undef.regs >> r & 1 == 1 {
                continue;
            }
            let mut lifted = o.regs[REGS[r]];
            if input.ptrs >> r & 1 == 1 {
                lifted = lifted.wrapping_sub(DATA_ADDR);
            }
            if lifted != host.regs[r] {
                mismatches.push(format!(
                    "{label}: {} = {lifted:#x}, host {:#x}",
                    REG_NAMES[r], host.regs[r]
                ));
            }
        }
        for f in 0..6 {
            if !undef.flags[f] && o.flags[f] != host.flags[f] {
                mismatches.push(format!(
                    "{label}: {} = {}, host {} (flags in {:?})",
                    FLAG_NAMES[f], o.flags[f], host.flags[f], input.flags
                ));
            }
        }
        let df = o.mem.last().is_some_and(|m| m.first() == Some(&1));
        if df != host.df {
            mismatches.push(format!("{label}: DF = {df}, host {}", host.df));
        }
        if !input.mem.is_empty() && o.mem[0] != host.mem {
            let at = o.mem[0]
                .iter()
                .zip(&host.mem)
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            mismatches.push(format!(
                "{label}: memory differs at offset {at}: {:02x?} vs host {:02x?}",
                &o.mem[0][at..(at + 8).min(host.mem.len())],
                &host.mem[at..(at + 8).min(host.mem.len())]
            ));
        }
        if o.regs[RSP] != STACK_TOP {
            mismatches.push(format!("{label}: rsp not restored"));
        }
    }
    // Keep the allocation reachable until the end.
    code_mem = code_mem.wrapping_add(0);
    let _ = code_mem;
    println!(
        "{label}: {} ops, {} cases, {} mismatches",
        ops.len(),
        cases.len(),
        mismatches.len()
    );
    mismatches
}

// ----- input generation --------------------------------------------------------------------------

const VALUES: [u64; 16] = [
    0,
    1,
    2,
    0x7F,
    0x80,
    0xFF,
    0x7FFF,
    0x8000,
    0xFFFF,
    0x7FFF_FFFF,
    0x8000_0000,
    0xFFFF_FFFF,
    0x1_0000_0000,
    i64::MAX as u64,
    i64::MIN as u64,
    u64::MAX,
];
const COUNTS: [u64; 19] = [
    0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 18, 31, 32, 33, 63, 64, 65, 127, 255,
];

/// How an active register is drawn.
#[derive(Clone, Copy)]
enum Src {
    /// Width boundaries plus random values.
    Value,
    /// Shift/rotate counts, with random upper bits half of the time.
    Count,
    /// `bextr` control words: start and length in bytes 0 and 1.
    Control,
}

fn values_for(src: Src, rng: &mut Lcg) -> Vec<u64> {
    match src {
        Src::Value => {
            let mut v = VALUES.to_vec();
            v.extend((0..4).map(|_| rng.next()));
            v
        }
        Src::Count => COUNTS
            .iter()
            .enumerate()
            .map(|(i, &c)| if i % 2 == 1 { c | (rng.next() << 8) } else { c })
            .collect(),
        Src::Control => {
            let parts = [0u64, 1, 4, 8, 15, 16, 31, 32, 33, 63, 64, 65, 255];
            let mut v = Vec::new();
            for (i, &start) in parts.iter().enumerate() {
                for (j, &len) in parts.iter().enumerate() {
                    if (i + j) % 2 == 0 {
                        v.push(start | (len << 8) | (rng.next() << 16));
                    }
                }
            }
            v
        }
    }
}

fn random_input(rng: &mut Lcg, mem_len: usize) -> Input {
    Input {
        regs: std::array::from_fn(|_| rng.next()),
        ptrs: 0,
        flags: std::array::from_fn(|_| rng.next() & 1 == 1),
        df: false,
        mem: (0..mem_len).map(|_| rng.next() as u8).collect(),
    }
}

/// Cross product of the active registers' value lists (sampled when more than two), each with
/// CF clear and set; every other register and flag random. `fix` adjusts each input.
fn reg_cases(
    rng: &mut Lcg,
    active: &[(usize, Src)],
    mem_len: usize,
    fix: impl Fn(&mut Input, &mut Lcg),
) -> Vec<Input> {
    let lists: Vec<Vec<u64>> = active.iter().map(|&(_, s)| values_for(s, rng)).collect();
    let mut combos: Vec<Vec<u64>> = vec![vec![]];
    if active.len() <= 2 {
        for list in &lists {
            combos = combos
                .iter()
                .flat_map(|c| {
                    list.iter().map(move |&v| {
                        let mut c = c.clone();
                        c.push(v);
                        c
                    })
                })
                .collect();
        }
    } else {
        combos = (0..500)
            .map(|_| {
                lists
                    .iter()
                    .map(|l| l[(rng.next() % l.len() as u64) as usize])
                    .collect()
            })
            .collect();
    }
    let mut out = Vec::new();
    for combo in combos {
        for cf in [false, true] {
            let mut input = random_input(rng, mem_len);
            for (&(slot, _), &v) in active.iter().zip(&combo) {
                input.regs[slot] = v;
            }
            input.flags[CF] = cf;
            fix(&mut input, rng);
            out.push(input);
        }
    }
    out
}

/// `[rsi]`-style memory operand inputs: a random buffer of `mem_len` bytes with the pointer
/// register at `offset`, plus the active registers.
fn mem_cases(
    rng: &mut Lcg,
    ptr: usize,
    offset: u64,
    active: &[(usize, Src)],
    mem_len: usize,
    fix: impl Fn(&mut Input, &mut Lcg),
) -> Vec<Input> {
    reg_cases(rng, active, mem_len, |input, rng| {
        input.ptrs |= 1 << ptr;
        input.regs[ptr] = offset;
        fix(input, rng);
    })
}

fn asm(build: impl FnOnce(&mut CodeAssembler) -> Result<(), IcedError>) -> Vec<u8> {
    assemble(0x1000, build)
}

fn no_undef() -> Box<dyn Fn(&Input) -> Undef> {
    Box::new(|_| Undef::default())
}

fn undef(list: &'static [usize]) -> Box<dyn Fn(&Input) -> Undef> {
    Box::new(move |_| undef_flags(list))
}

fn op(
    name: impl Into<String>,
    code: Vec<u8>,
    feature: Option<&'static str>,
    cases: Vec<Input>,
    undefined: Box<dyn Fn(&Input) -> Undef>,
) -> Op {
    Op {
        name: name.into(),
        code,
        feature,
        cases,
        undefined,
    }
}

fn report(test: &str, mismatches: &[String]) {
    assert!(
        mismatches.is_empty(),
        "{test}: {} mismatches, first ones:\n{}",
        mismatches.len(),
        mismatches
            .iter()
            .take(30)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// Drops (with a note) ops whose CPU feature the host lacks.
fn supported(test: &str, ops: Vec<Op>) -> Vec<Op> {
    ops.into_iter()
        .filter(|op| match op.feature {
            Some(f) if !host_has(f) => {
                println!("SKIP {test}: {} (host CPU lacks {f})", op.name);
                false
            }
            _ => true,
        })
        .collect()
}

// ----- string instructions ---------------------------------------------------------------------

const STRING_MEM: usize = 512;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Str {
    Movs,
    Stos,
    Lods,
    Cmps,
    Scas,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Repeat {
    No,
    Rep,
    Ne,
}

fn string_code(kind: Str, size: u64, rep: Repeat) -> Vec<u8> {
    let base = match kind {
        Str::Movs => 0xA4,
        Str::Cmps => 0xA6,
        Str::Stos => 0xAA,
        Str::Lods => 0xAC,
        Str::Scas => 0xAE,
    };
    let mut code = Vec::new();
    match rep {
        Repeat::No => {}
        Repeat::Rep => code.push(0xF3),
        Repeat::Ne => code.push(0xF2),
    }
    match size {
        2 => code.push(0x66),
        8 => code.push(0x48),
        _ => {}
    }
    code.push(if size == 1 { base } else { base + 1 });
    code
}

/// Writes the low `size` bytes of `v` at `at`.
fn put(mem: &mut [u8], at: u64, size: u64, v: u64) {
    for b in 0..size {
        mem[(at + b) as usize] = (v >> (8 * b)) as u8;
    }
}

fn get(mem: &[u8], at: u64, size: u64) -> u64 {
    (0..size).fold(0, |acc, b| {
        acc | (u64::from(mem[(at + b) as usize]) << (8 * b))
    })
}

fn string_cases(rng: &mut Lcg, kind: Str, size: u64, rep: Repeat) -> Vec<Input> {
    let max_bytes = 192u64;
    let counts: Vec<u64> = if rep == Repeat::No {
        vec![rng.next()]
    } else {
        [0u64, 1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 24, 33, 64, 100, 192]
            .into_iter()
            .filter(|c| c * size <= max_bytes)
            .collect()
    };
    let gaps: Vec<i64> = match kind {
        Str::Movs | Str::Cmps => vec![
            -100, -17, -9, -8, -7, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 100,
        ],
        _ => vec![0],
    };
    let accs = [
        0u64,
        0x4242_4242_4242_4242,
        u64::MAX,
        0x00FF_00FF_00FF_00FF,
        rng.next(),
    ];
    let mut out = Vec::new();
    for df in [false, true] {
        for &count in &counts {
            for &gap in &gaps {
                for (variant, &acc) in accs.iter().enumerate() {
                    // Only the stored/compared value matters beyond the first variant.
                    if variant > 0 && !matches!(kind, Str::Stos | Str::Scas | Str::Cmps) {
                        break;
                    }
                    let mut input = random_input(rng, STRING_MEM);
                    input.df = df;
                    let src = (256 - gap / 2) as u64;
                    let dst = (src as i64 + gap) as u64;
                    input.ptrs = (1 << SI) | (1 << DI);
                    input.regs[SI] = src;
                    input.regs[DI] = dst;
                    input.regs[CX] = count;
                    input.regs[AX] = acc;
                    let step = |j: u64, start: u64| {
                        if df {
                            start.wrapping_sub(j * size)
                        } else {
                            start + j * size
                        }
                    };
                    let n = if rep == Repeat::No { 1 } else { count };
                    match kind {
                        Str::Cmps if n > 0 => {
                            // variant 0: random; 1: equal; 2: equal then a mismatch at j;
                            // 3: different but equal at j; 4: equal, large rcx stops at j
                            let j = rng.next() % n;
                            if variant >= 1 && gap.unsigned_abs() >= n * size {
                                for e in 0..n {
                                    let v = get(&input.mem, step(e, src), size);
                                    put(&mut input.mem, step(e, dst), size, v);
                                }
                                match variant {
                                    2 | 4 => {
                                        let v = get(&input.mem, step(j, dst), size);
                                        put(&mut input.mem, step(j, dst), size, v ^ 0x10);
                                        if variant == 4 && rep == Repeat::Rep {
                                            input.regs[CX] = u64::MAX;
                                        }
                                    }
                                    3 => {
                                        for e in 0..n {
                                            if e != j {
                                                let v = get(&input.mem, step(e, dst), size);
                                                put(&mut input.mem, step(e, dst), size, !v);
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                        Str::Scas if n > 0 => {
                            let j = rng.next() % n;
                            match variant {
                                1 => {
                                    // equal until j (repe stops at j)
                                    for e in 0..n {
                                        let v = if e == j { !acc } else { acc };
                                        put(&mut input.mem, step(e, dst), size, v);
                                    }
                                }
                                2 | 4 => {
                                    // different except at j (repne stops at j)
                                    for e in 0..n {
                                        let v = if e == j { acc } else { !acc };
                                        put(&mut input.mem, step(e, dst), size, v);
                                    }
                                    if variant == 4 && rep == Repeat::Ne {
                                        input.regs[CX] = u64::MAX;
                                    }
                                }
                                _ => {}
                            }
                        }
                        _ => {}
                    }
                    out.push(input);
                }
            }
        }
    }
    out
}

fn string_ops(rng: &mut Lcg) -> Vec<Op> {
    let mut ops = Vec::new();
    for (kind, name) in [
        (Str::Movs, "movs"),
        (Str::Stos, "stos"),
        (Str::Lods, "lods"),
        (Str::Cmps, "cmps"),
        (Str::Scas, "scas"),
    ] {
        let reps: &[Repeat] = if matches!(kind, Str::Cmps | Str::Scas) {
            &[Repeat::No, Repeat::Rep, Repeat::Ne]
        } else {
            &[Repeat::No, Repeat::Rep]
        };
        for &rep in reps {
            for size in [1u64, 2, 4, 8] {
                let prefix = match rep {
                    Repeat::No => "",
                    Repeat::Rep if matches!(kind, Str::Cmps | Str::Scas) => "repe ",
                    Repeat::Rep => "rep ",
                    Repeat::Ne => "repne ",
                };
                ops.push(op(
                    format!("{prefix}{name}{}", size * 8),
                    string_code(kind, size, rep),
                    None,
                    string_cases(rng, kind, size, rep),
                    no_undef(),
                ));
            }
        }
    }
    // fs-relative source is covered by the exec tests (fs base lives in the state).
    ops
}

#[test]
fn string_instructions_match_the_host_cpu() {
    let test = "string_instructions_match_the_host_cpu";
    if !exec_or_skip(test) {
        return;
    }
    let mut rng = Lcg(0x57A1);
    let ops = string_ops(&mut rng);
    report(test, &differential("hostdiff_string", &ops));
}

// ----- bit instructions ------------------------------------------------------------------------

/// Register value for a bit offset `e` used at `bits` width: only the low `bits` bits matter
/// (sign-extended), the rest is garbage.
fn bit_offset_value(e: i64, bits: u32, rng: &mut Lcg) -> u64 {
    if bits == 64 {
        e as u64
    } else {
        let m = (1u64 << bits) - 1;
        (rng.next() & !m) | (e as u64 & m)
    }
}

fn bit_ops(rng: &mut Lcg) -> Vec<Op> {
    let mut ops = Vec::new();
    type Gpr = fn(usize) -> AsmRegister64;
    let _: Option<Gpr> = None;
    let r16 = [dx, r8w];
    let r32 = [edx, r8d];
    let r64 = [rdx, r8];
    let bt_undef = || undef(&[PF, AF, SF, OF]);
    for (m, name) in [(0, "bt"), (1, "bts"), (2, "btr"), (3, "btc")] {
        for bits in [16u32, 32, 64] {
            // register, register
            let code = asm(|a| match (m, bits) {
                (0, 16) => a.bt(r16[0], r16[1]),
                (0, 32) => a.bt(r32[0], r32[1]),
                (0, _) => a.bt(r64[0], r64[1]),
                (1, 16) => a.bts(r16[0], r16[1]),
                (1, 32) => a.bts(r32[0], r32[1]),
                (1, _) => a.bts(r64[0], r64[1]),
                (2, 16) => a.btr(r16[0], r16[1]),
                (2, 32) => a.btr(r32[0], r32[1]),
                (2, _) => a.btr(r64[0], r64[1]),
                (_, 16) => a.btc(r16[0], r16[1]),
                (_, 32) => a.btc(r32[0], r32[1]),
                _ => a.btc(r64[0], r64[1]),
            });
            let cases = reg_cases(rng, &[(DX, Src::Value), (X8, Src::Count)], 0, |_, _| {});
            ops.push(op(
                format!("{name}{bits} r,r"),
                code,
                None,
                cases,
                bt_undef(),
            ));
            // register, imm8 (masked to the operand size)
            for imm in [3u32, 45, 200] {
                let code = asm(|a| match (m, bits) {
                    (0, 16) => a.bt(r16[0], imm),
                    (0, 32) => a.bt(r32[0], imm),
                    (0, _) => a.bt(r64[0], imm),
                    (1, 16) => a.bts(r16[0], imm),
                    (1, 32) => a.bts(r32[0], imm),
                    (1, _) => a.bts(r64[0], imm),
                    (2, 16) => a.btr(r16[0], imm),
                    (2, 32) => a.btr(r32[0], imm),
                    (2, _) => a.btr(r64[0], imm),
                    (_, 16) => a.btc(r16[0], imm),
                    (_, 32) => a.btc(r32[0], imm),
                    _ => a.btc(r64[0], imm),
                });
                let cases = reg_cases(rng, &[(DX, Src::Value)], 0, |_, _| {});
                ops.push(op(
                    format!("{name}{bits} r,{imm}"),
                    code,
                    None,
                    cases,
                    bt_undef(),
                ));
            }
            // memory, register: the signed bit offset reaches other bytes
            for lock in [false, true] {
                if lock && m == 0 {
                    continue;
                }
                let mem = |bits| match bits {
                    16 => word_ptr(rsi),
                    32 => dword_ptr(rsi),
                    _ => qword_ptr(rsi),
                };
                let code = asm(|a| {
                    if lock {
                        let _ = a.lock();
                    }
                    match (m, bits) {
                        (0, 16) => a.bt(mem(16), r16[1]),
                        (0, 32) => a.bt(mem(32), r32[1]),
                        (0, _) => a.bt(mem(64), r64[1]),
                        (1, 16) => a.bts(mem(16), r16[1]),
                        (1, 32) => a.bts(mem(32), r32[1]),
                        (1, _) => a.bts(mem(64), r64[1]),
                        (2, 16) => a.btr(mem(16), r16[1]),
                        (2, 32) => a.btr(mem(32), r32[1]),
                        (2, _) => a.btr(mem(64), r64[1]),
                        (_, 16) => a.btc(mem(16), r16[1]),
                        (_, 32) => a.btc(mem(32), r32[1]),
                        _ => a.btc(mem(64), r64[1]),
                    }
                });
                let offsets: [i64; 22] = [
                    0, 1, 7, 8, 9, 15, 16, 31, 32, 63, 64, 65, -1, -8, -9, -16, -63, -64, -65,
                    -100, 1000, -1500,
                ];
                let mut cases = Vec::new();
                for &e in &offsets {
                    for cf in [false, true] {
                        let mut input = random_input(rng, 512);
                        input.ptrs = 1 << SI;
                        input.regs[SI] = 256;
                        input.regs[X8] = bit_offset_value(e, bits, rng);
                        input.flags[CF] = cf;
                        cases.push(input);
                    }
                }
                let prefix = if lock { "lock " } else { "" };
                ops.push(op(
                    format!("{prefix}{name}{bits} [m],r"),
                    code,
                    None,
                    cases,
                    bt_undef(),
                ));
                // memory, imm8
                let code = asm(|a| {
                    if lock {
                        let _ = a.lock();
                    }
                    match (m, bits) {
                        (0, 16) => a.bt(mem(16), 13),
                        (0, 32) => a.bt(mem(32), 45),
                        (0, _) => a.bt(mem(64), 200),
                        (1, 16) => a.bts(mem(16), 13),
                        (1, 32) => a.bts(mem(32), 45),
                        (1, _) => a.bts(mem(64), 200),
                        (2, 16) => a.btr(mem(16), 13),
                        (2, 32) => a.btr(mem(32), 45),
                        (2, _) => a.btr(mem(64), 200),
                        (_, 16) => a.btc(mem(16), 13),
                        (_, 32) => a.btc(mem(32), 45),
                        _ => a.btc(mem(64), 200),
                    }
                });
                let cases = mem_cases(rng, SI, 16, &[], 64, |_, _| {});
                let cases = cases
                    .into_iter()
                    .chain(mem_cases(rng, SI, 16, &[], 64, |_, _| {}))
                    .collect();
                ops.push(op(
                    format!("{prefix}{name}{bits} [m],imm"),
                    code,
                    None,
                    cases,
                    bt_undef(),
                ));
            }
        }
    }
    // bsf / bsr / tzcnt / lzcnt / popcnt
    for (name, feature, undefined) in [
        ("bsf", None, &[CF, PF, AF, SF, OF][..]),
        ("bsr", None, &[CF, PF, AF, SF, OF][..]),
        ("tzcnt", Some("bmi1"), &[PF, AF, SF, OF][..]),
        ("lzcnt", Some("lzcnt"), &[PF, AF, SF, OF][..]),
        ("popcnt", Some("popcnt"), &[][..]),
    ] {
        for bits in [16u32, 32, 64] {
            let code = asm(|a| match (name, bits) {
                ("bsf", 16) => a.bsf(dx, r8w),
                ("bsf", 32) => a.bsf(edx, r8d),
                ("bsf", _) => a.bsf(rdx, r8),
                ("bsr", 16) => a.bsr(dx, r8w),
                ("bsr", 32) => a.bsr(edx, r8d),
                ("bsr", _) => a.bsr(rdx, r8),
                ("tzcnt", 16) => a.tzcnt(dx, r8w),
                ("tzcnt", 32) => a.tzcnt(edx, r8d),
                ("tzcnt", _) => a.tzcnt(rdx, r8),
                ("lzcnt", 16) => a.lzcnt(dx, r8w),
                ("lzcnt", 32) => a.lzcnt(edx, r8d),
                ("lzcnt", _) => a.lzcnt(rdx, r8),
                (_, 16) => a.popcnt(dx, r8w),
                (_, 32) => a.popcnt(edx, r8d),
                _ => a.popcnt(rdx, r8),
            });
            let list: &'static [usize] = undefined;
            let cases = reg_cases(rng, &[(X8, Src::Value)], 0, |_, _| {});
            ops.push(op(
                format!("{name}{bits}"),
                code,
                feature,
                cases,
                undef(list),
            ));
        }
        // memory source
        let code = asm(|a| match name {
            "bsf" => a.bsf(edx, dword_ptr(rsi)),
            "bsr" => a.bsr(rdx, qword_ptr(rsi)),
            "tzcnt" => a.tzcnt(rdx, qword_ptr(rsi)),
            "lzcnt" => a.lzcnt(dx, word_ptr(rsi)),
            _ => a.popcnt(edx, dword_ptr(rsi)),
        });
        let list: &'static [usize] = undefined;
        let cases = mem_cases(rng, SI, 8, &[(DX, Src::Value)], 32, |input, rng| {
            if rng.next() % 4 == 0 {
                put(&mut input.mem, 8, 8, 0);
            }
        });
        ops.push(op(
            format!("{name} r,[m]"),
            code,
            feature,
            cases,
            undef(list),
        ));
    }
    // bswap (16-bit: undefined, hardware zeroes the register)
    ops.push(op(
        "bswap16",
        encode(
            0x1000,
            &[Instruction::with1(Code::Bswap_r16, Register::DX).expect("bswap")],
        ),
        None,
        reg_cases(rng, &[(DX, Src::Value)], 0, |_, _| {}),
        no_undef(),
    ));
    ops.push(op(
        "bswap32",
        asm(|a| a.bswap(edx)),
        None,
        reg_cases(rng, &[(DX, Src::Value)], 0, |_, _| {}),
        no_undef(),
    ));
    ops.push(op(
        "bswap64",
        asm(|a| a.bswap(rdx)),
        None,
        reg_cases(rng, &[(DX, Src::Value)], 0, |_, _| {}),
        no_undef(),
    ));
    ops
}

fn bmi_ops(rng: &mut Lcg) -> Vec<Op> {
    let mut ops = Vec::new();
    for bits in [32u32, 64] {
        let w = bits == 64;
        let two = [(X8, Src::Value), (X9, Src::Value)];
        let mut add = |name: &str, feature, code: Vec<u8>, cases, undefined| {
            ops.push(op(
                format!("{name}{bits}"),
                code,
                Some(feature),
                cases,
                undefined,
            ));
        };
        add(
            "andn",
            "bmi1",
            asm(|a| {
                if w {
                    a.andn(rdx, r8, r9)
                } else {
                    a.andn(edx, r8d, r9d)
                }
            }),
            reg_cases(rng, &two, 0, |_, _| {}),
            undef(&[AF, PF]),
        );
        add(
            "andn [m]",
            "bmi1",
            asm(|a| {
                if w {
                    a.andn(rdx, r8, qword_ptr(rsi))
                } else {
                    a.andn(edx, r8d, dword_ptr(rsi))
                }
            }),
            mem_cases(rng, SI, 8, &[(X8, Src::Value)], 32, |_, _| {}),
            undef(&[AF, PF]),
        );
        for name in ["blsi", "blsr", "blsmsk"] {
            add(
                name,
                "bmi1",
                asm(|a| match (name, w) {
                    ("blsi", true) => a.blsi(rdx, r8),
                    ("blsi", false) => a.blsi(edx, r8d),
                    ("blsr", true) => a.blsr(rdx, r8),
                    ("blsr", false) => a.blsr(edx, r8d),
                    (_, true) => a.blsmsk(rdx, r8),
                    _ => a.blsmsk(edx, r8d),
                }),
                reg_cases(rng, &[(X8, Src::Value)], 0, |_, _| {}),
                undef(&[AF, PF]),
            );
        }
        add(
            "bextr",
            "bmi1",
            asm(|a| {
                if w {
                    a.bextr(rdx, r8, r9)
                } else {
                    a.bextr(edx, r8d, r9d)
                }
            }),
            reg_cases(rng, &[(X8, Src::Value), (X9, Src::Control)], 0, |_, _| {}),
            undef(&[AF, SF, PF]),
        );
        add(
            "bzhi",
            "bmi2",
            asm(|a| {
                if w {
                    a.bzhi(rdx, r8, r9)
                } else {
                    a.bzhi(edx, r8d, r9d)
                }
            }),
            reg_cases(rng, &[(X8, Src::Value), (X9, Src::Count)], 0, |_, _| {}),
            undef(&[AF, PF]),
        );
        for name in ["shlx", "shrx", "sarx"] {
            add(
                name,
                "bmi2",
                asm(|a| match (name, w) {
                    ("shlx", true) => a.shlx(rdx, r8, r9),
                    ("shlx", false) => a.shlx(edx, r8d, r9d),
                    ("shrx", true) => a.shrx(rdx, r8, r9),
                    ("shrx", false) => a.shrx(edx, r8d, r9d),
                    (_, true) => a.sarx(rdx, r8, r9),
                    _ => a.sarx(edx, r8d, r9d),
                }),
                reg_cases(rng, &[(X8, Src::Value), (X9, Src::Count)], 0, |_, _| {}),
                no_undef(),
            );
        }
        for imm in [0u32, 1, 13, 31, 63] {
            add(
                &format!("rorx {imm} "),
                "bmi2",
                asm(|a| {
                    if w {
                        a.rorx(rdx, r8, imm)
                    } else {
                        a.rorx(edx, r8d, imm)
                    }
                }),
                reg_cases(rng, &[(X8, Src::Value)], 0, |_, _| {}),
                no_undef(),
            );
        }
        add(
            "mulx",
            "bmi2",
            asm(|a| {
                if w {
                    a.mulx(rcx, r8, r9)
                } else {
                    a.mulx(ecx, r8d, r9d)
                }
            }),
            reg_cases(rng, &[(DX, Src::Value), (X9, Src::Value)], 0, |_, _| {}),
            no_undef(),
        );
        add(
            "mulx same-dst ",
            "bmi2",
            asm(|a| {
                if w {
                    a.mulx(rcx, rcx, r9)
                } else {
                    a.mulx(ecx, ecx, r9d)
                }
            }),
            reg_cases(rng, &[(DX, Src::Value), (X9, Src::Value)], 0, |_, _| {}),
            no_undef(),
        );
        for name in ["pdep", "pext"] {
            add(
                name,
                "bmi2",
                asm(|a| match (name, w) {
                    ("pdep", true) => a.pdep(rdx, r8, r9),
                    ("pdep", false) => a.pdep(edx, r8d, r9d),
                    (_, true) => a.pext(rdx, r8, r9),
                    _ => a.pext(edx, r8d, r9d),
                }),
                reg_cases(rng, &two, 0, |_, _| {}),
                no_undef(),
            );
        }
    }
    ops
}

#[test]
fn bit_instructions_match_the_host_cpu() {
    let test = "bit_instructions_match_the_host_cpu";
    if !exec_or_skip(test) {
        return;
    }
    let mut rng = Lcg(0xB175);
    let mut ops = bit_ops(&mut rng);
    ops.extend(bmi_ops(&mut rng));
    // The locked bit-test forms need a shared memory; they run in the atomics test.
    let ops: Vec<Op> = ops
        .into_iter()
        .filter(|o| !o.name.starts_with("lock "))
        .collect();
    let ops = supported(test, ops);
    report(test, &differential("hostdiff_bits", &ops));
}

// ----- bit/BMI forms with aliased operands, memory sources and in-block flag consumers ----------

/// Bit offsets for `bt* [rsi + rdx*8 - 8], rdx` (the offset doubles as the index register).
const ALIAS_OFFSETS: [i64; 15] = [0, 1, 7, 8, 31, 32, 63, 64, 100, -1, -8, -9, -64, -65, -100];

/// Operand-aliasing and memory-source forms of the BMI/bit-count/bit-test instructions (the
/// destination doubling as a source, count or control register), as compilers emit them.
fn alias_ops(rng: &mut Lcg) -> Vec<Op> {
    let mut ops = Vec::new();
    for bits in [32u32, 64] {
        let w = bits == 64;
        let mut add = |name: &str, feature, code: Vec<u8>, cases, undefined| {
            ops.push(op(
                format!("{name}{bits}"),
                code,
                Some(feature),
                cases,
                undefined,
            ));
        };
        for name in ["shlx", "shrx", "sarx"] {
            let pick = |a: &mut CodeAssembler, d: usize, s: usize, c: usize| {
                let r64 = [rdx, r8, r9];
                let r32 = [edx, r8d, r9d];
                match (name, w) {
                    ("shlx", true) => a.shlx(r64[d], r64[s], r64[c]),
                    ("shlx", false) => a.shlx(r32[d], r32[s], r32[c]),
                    ("shrx", true) => a.shrx(r64[d], r64[s], r64[c]),
                    ("shrx", false) => a.shrx(r32[d], r32[s], r32[c]),
                    (_, true) => a.sarx(r64[d], r64[s], r64[c]),
                    _ => a.sarx(r32[d], r32[s], r32[c]),
                }
            };
            add(
                &format!("{name} d=s "),
                "bmi2",
                asm(|a| pick(a, 1, 1, 2)),
                reg_cases(rng, &[(X8, Src::Value), (X9, Src::Count)], 0, |_, _| {}),
                no_undef(),
            );
            add(
                &format!("{name} d=c "),
                "bmi2",
                asm(|a| pick(a, 2, 1, 2)),
                reg_cases(rng, &[(X8, Src::Value), (X9, Src::Count)], 0, |_, _| {}),
                no_undef(),
            );
            add(
                &format!("{name} d=s=c "),
                "bmi2",
                asm(|a| pick(a, 1, 1, 1)),
                reg_cases(rng, &[(X8, Src::Value)], 0, |_, _| {})
                    .into_iter()
                    .chain(reg_cases(rng, &[(X8, Src::Count)], 0, |_, _| {}))
                    .collect(),
                no_undef(),
            );
            add(
                &format!("{name} [m] "),
                "bmi2",
                asm(|a| match (name, w) {
                    ("shlx", true) => a.shlx(rdx, qword_ptr(rsi), r9),
                    ("shlx", false) => a.shlx(edx, dword_ptr(rsi), r9d),
                    ("shrx", true) => a.shrx(rdx, qword_ptr(rsi), r9),
                    ("shrx", false) => a.shrx(edx, dword_ptr(rsi), r9d),
                    (_, true) => a.sarx(rdx, qword_ptr(rsi), r9),
                    _ => a.sarx(edx, dword_ptr(rsi), r9d),
                }),
                mem_cases(rng, SI, 8, &[(X9, Src::Count)], 32, |_, _| {}),
                no_undef(),
            );
        }
        // bextr with the destination as control (`bextr r9d, esi, r9d`) or as source.
        for (label, d, s) in [("d=ctl ", 2usize, 1usize), ("d=s ", 1, 1)] {
            add(
                &format!("bextr {label}"),
                "bmi1",
                asm(|a| {
                    if w {
                        let r = [rdx, r8, r9];
                        a.bextr(r[d], r[s], r9)
                    } else {
                        let r = [edx, r8d, r9d];
                        a.bextr(r[d], r[s], r9d)
                    }
                }),
                reg_cases(rng, &[(X8, Src::Value), (X9, Src::Control)], 0, |_, _| {}),
                undef(&[AF, SF, PF]),
            );
        }
        add(
            "bextr [m] d=ctl ",
            "bmi1",
            asm(|a| {
                if w {
                    a.bextr(r9, qword_ptr(rsi), r9)
                } else {
                    a.bextr(r9d, dword_ptr(rsi), r9d)
                }
            }),
            mem_cases(rng, SI, 8, &[(X9, Src::Control)], 32, |_, _| {}),
            undef(&[AF, SF, PF]),
        );
        add(
            "bzhi d=idx ",
            "bmi2",
            asm(|a| {
                if w {
                    a.bzhi(r9, r8, r9)
                } else {
                    a.bzhi(r9d, r8d, r9d)
                }
            }),
            reg_cases(rng, &[(X8, Src::Value), (X9, Src::Count)], 0, |_, _| {}),
            undef(&[AF, PF]),
        );
        add(
            "bzhi [m] ",
            "bmi2",
            asm(|a| {
                if w {
                    a.bzhi(rdx, qword_ptr(rsi), r9)
                } else {
                    a.bzhi(edx, dword_ptr(rsi), r9d)
                }
            }),
            mem_cases(rng, SI, 8, &[(X9, Src::Count)], 32, |_, _| {}),
            undef(&[AF, PF]),
        );
        for (label, d) in [("d=s1 ", 1usize), ("d=s2 ", 2)] {
            add(
                &format!("andn {label}"),
                "bmi1",
                asm(|a| {
                    if w {
                        let r = [rdx, r8, r9];
                        a.andn(r[d], r8, r9)
                    } else {
                        let r = [edx, r8d, r9d];
                        a.andn(r[d], r8d, r9d)
                    }
                }),
                reg_cases(rng, &[(X8, Src::Value), (X9, Src::Value)], 0, |_, _| {}),
                undef(&[AF, PF]),
            );
        }
        for name in ["blsi", "blsr", "blsmsk"] {
            add(
                &format!("{name} d=s "),
                "bmi1",
                asm(|a| match (name, w) {
                    ("blsi", true) => a.blsi(r8, r8),
                    ("blsi", false) => a.blsi(r8d, r8d),
                    ("blsr", true) => a.blsr(r8, r8),
                    ("blsr", false) => a.blsr(r8d, r8d),
                    (_, true) => a.blsmsk(r8, r8),
                    _ => a.blsmsk(r8d, r8d),
                }),
                reg_cases(rng, &[(X8, Src::Value)], 0, |_, _| {}),
                undef(&[AF, PF]),
            );
            add(
                &format!("{name} [m] "),
                "bmi1",
                asm(|a| match (name, w) {
                    ("blsi", true) => a.blsi(rdx, qword_ptr(rsi)),
                    ("blsi", false) => a.blsi(edx, dword_ptr(rsi)),
                    ("blsr", true) => a.blsr(rdx, qword_ptr(rsi)),
                    ("blsr", false) => a.blsr(edx, dword_ptr(rsi)),
                    (_, true) => a.blsmsk(rdx, qword_ptr(rsi)),
                    _ => a.blsmsk(edx, dword_ptr(rsi)),
                }),
                mem_cases(rng, SI, 8, &[], 32, |input, rng| {
                    if rng.next() % 4 == 0 {
                        put(&mut input.mem, 8, 8, 0);
                    }
                }),
                undef(&[AF, PF]),
            );
        }
    }
    // lzcnt/tzcnt/popcnt in place, at every width.
    for (name, feature, list) in [
        ("lzcnt", "lzcnt", &[PF, AF, SF, OF][..]),
        ("tzcnt", "bmi1", &[PF, AF, SF, OF][..]),
        ("popcnt", "popcnt", &[][..]),
    ] {
        for bits in [16u32, 32, 64] {
            let code = asm(|a| match (name, bits) {
                ("lzcnt", 16) => a.lzcnt(r8w, r8w),
                ("lzcnt", 32) => a.lzcnt(r8d, r8d),
                ("lzcnt", _) => a.lzcnt(r8, r8),
                ("tzcnt", 16) => a.tzcnt(r8w, r8w),
                ("tzcnt", 32) => a.tzcnt(r8d, r8d),
                ("tzcnt", _) => a.tzcnt(r8, r8),
                (_, 16) => a.popcnt(r8w, r8w),
                (_, 32) => a.popcnt(r8d, r8d),
                _ => a.popcnt(r8, r8),
            });
            let list: &'static [usize] = list;
            ops.push(op(
                format!("{name}{bits} d=s"),
                code,
                Some(feature),
                reg_cases(rng, &[(X8, Src::Value)], 0, |_, _| {}),
                undef(list),
            ));
        }
    }
    // bt* with the offset register aliasing the destination register.
    let bt_undef = || undef(&[PF, AF, SF, OF]);
    for (m, name) in [(0, "bt"), (1, "bts"), (2, "btr"), (3, "btc")] {
        for bits in [16u32, 32, 64] {
            let code = asm(|a| match (m, bits) {
                (0, 16) => a.bt(dx, dx),
                (0, 32) => a.bt(edx, edx),
                (0, _) => a.bt(rdx, rdx),
                (1, 16) => a.bts(dx, dx),
                (1, 32) => a.bts(edx, edx),
                (1, _) => a.bts(rdx, rdx),
                (2, 16) => a.btr(dx, dx),
                (2, 32) => a.btr(edx, edx),
                (2, _) => a.btr(rdx, rdx),
                (_, 16) => a.btc(dx, dx),
                (_, 32) => a.btc(edx, edx),
                _ => a.btc(rdx, rdx),
            });
            let cases = reg_cases(rng, &[(DX, Src::Value)], 0, |_, _| {})
                .into_iter()
                .chain(reg_cases(rng, &[(DX, Src::Count)], 0, |_, _| {}))
                .collect();
            ops.push(op(
                format!("{name}{bits} r,r d=o"),
                code,
                None,
                cases,
                bt_undef(),
            ));
            // [rsi + rdx*8 - 8], rdx: the offset is also the scaled index, so the bit lands in a
            // different element than the one the address names (or before it, when negative).
            let code = asm(|a| match (m, bits) {
                (0, 16) => a.bt(word_ptr(rsi + rdx * 8 - 8), dx),
                (0, 32) => a.bt(dword_ptr(rsi + rdx * 8 - 8), edx),
                (0, _) => a.bt(qword_ptr(rsi + rdx * 8 - 8), rdx),
                (1, 16) => a.bts(word_ptr(rsi + rdx * 8 - 8), dx),
                (1, 32) => a.bts(dword_ptr(rsi + rdx * 8 - 8), edx),
                (1, _) => a.bts(qword_ptr(rsi + rdx * 8 - 8), rdx),
                (2, 16) => a.btr(word_ptr(rsi + rdx * 8 - 8), dx),
                (2, 32) => a.btr(dword_ptr(rsi + rdx * 8 - 8), edx),
                (2, _) => a.btr(qword_ptr(rsi + rdx * 8 - 8), rdx),
                (_, 16) => a.btc(word_ptr(rsi + rdx * 8 - 8), dx),
                (_, 32) => a.btc(dword_ptr(rsi + rdx * 8 - 8), edx),
                _ => a.btc(qword_ptr(rsi + rdx * 8 - 8), rdx),
            });
            let mut cases = Vec::new();
            for &e in &ALIAS_OFFSETS {
                for cf in [false, true] {
                    let mut input = random_input(rng, 2048);
                    input.ptrs = 1 << SI;
                    input.regs[SI] = 1024;
                    // The index uses all 64 bits, so the offset must be a clean sign extension.
                    input.regs[DX] = e as u64;
                    input.flags[CF] = cf;
                    cases.push(input);
                }
            }
            ops.push(op(
                format!("{name}{bits} [b+o*8-8],o"),
                code,
                None,
                cases,
                bt_undef(),
            ));
        }
    }
    ops
}

/// Short sequences that consume the flags of a bit/BMI instruction inside the same block
/// (setcc / cmovcc / adc / sbb / jcc), which exercises the lifter's lazy flag sources rather
/// than the flags materialised at the function exit, plus the shapes of a real buddy allocator.
fn flag_consumer_ops(rng: &mut Lcg) -> Vec<Op> {
    let mut ops = Vec::new();
    let two = [(X8, Src::Value), (X9, Src::Value)];
    let one = [(X8, Src::Value)];
    let bt_undef = || undef(&[PF, AF, SF, OF]);
    let cnt_undef = || undef(&[PF, AF, SF, OF]);
    for bits in [16u32, 32, 64] {
        for (name, feature) in [("lzcnt", "lzcnt"), ("tzcnt", "bmi1")] {
            let code = asm(|a| {
                match (name, bits) {
                    ("lzcnt", 16) => a.lzcnt(dx, r8w)?,
                    ("lzcnt", 32) => a.lzcnt(edx, r8d)?,
                    ("lzcnt", _) => a.lzcnt(rdx, r8)?,
                    (_, 16) => a.tzcnt(dx, r8w)?,
                    (_, 32) => a.tzcnt(edx, r8d)?,
                    _ => a.tzcnt(rdx, r8)?,
                }
                a.setb(al)?;
                a.sete(cl)?;
                a.setbe(bl)?;
                a.cmova(rdi, r9)
            });
            ops.push(op(
                format!("{name}{bits}; setb sete setbe cmova"),
                code,
                Some(feature),
                reg_cases(rng, &two, 0, |_, _| {}),
                cnt_undef(),
            ));
        }
        let code = asm(|a| {
            match bits {
                16 => a.popcnt(dx, r8w)?,
                32 => a.popcnt(edx, r8d)?,
                _ => a.popcnt(rdx, r8)?,
            }
            a.sete(al)?;
            a.setb(cl)?;
            a.sets(bl)?;
            a.setp(dil)
        });
        ops.push(op(
            format!("popcnt{bits}; sete setb sets setp"),
            code,
            Some("popcnt"),
            reg_cases(rng, &one, 0, |_, _| {}),
            no_undef(),
        ));
        for (m, name) in [(0, "bt"), (1, "bts"), (2, "btr"), (3, "btc")] {
            let code = asm(|a| {
                match (m, bits) {
                    (0, 16) => a.bt(dx, r8w)?,
                    (0, 32) => a.bt(edx, r8d)?,
                    (0, _) => a.bt(rdx, r8)?,
                    (1, 16) => a.bts(dx, r8w)?,
                    (1, 32) => a.bts(edx, r8d)?,
                    (1, _) => a.bts(rdx, r8)?,
                    (2, 16) => a.btr(dx, r8w)?,
                    (2, 32) => a.btr(edx, r8d)?,
                    (2, _) => a.btr(rdx, r8)?,
                    (_, 16) => a.btc(dx, r8w)?,
                    (_, 32) => a.btc(edx, r8d)?,
                    _ => a.btc(rdx, r8)?,
                }
                a.setb(al)?;
                a.cmovb(rdi, r9)?;
                a.adc(rcx, 0)
            });
            ops.push(op(
                format!("{name}{bits} r,r; setb cmovb adc"),
                code,
                None,
                reg_cases(rng, &[(DX, Src::Value), (X8, Src::Count)], 0, |_, _| {}),
                no_undef(),
            ));
            let code = asm(|a| {
                match (m, bits) {
                    (0, 16) => a.bt(dx, 13)?,
                    (0, 32) => a.bt(edx, 45)?,
                    (0, _) => a.bt(rdx, 63)?,
                    (1, 16) => a.bts(dx, 13)?,
                    (1, 32) => a.bts(edx, 45)?,
                    (1, _) => a.bts(rdx, 63)?,
                    (2, 16) => a.btr(dx, 13)?,
                    (2, 32) => a.btr(edx, 45)?,
                    (2, _) => a.btr(rdx, 63)?,
                    (_, 16) => a.btc(dx, 13)?,
                    (_, 32) => a.btc(edx, 45)?,
                    _ => a.btc(rdx, 63)?,
                }
                a.setae(al)?;
                a.cmovae(rdi, r9)?;
                a.sbb(rcx, rcx)
            });
            ops.push(op(
                format!("{name}{bits} r,imm; setae cmovae sbb"),
                code,
                None,
                reg_cases(rng, &[(DX, Src::Value)], 0, |_, _| {}),
                no_undef(),
            ));
            let code = asm(|a| {
                match (m, bits) {
                    (0, 16) => a.bt(word_ptr(rsi), r8w)?,
                    (0, 32) => a.bt(dword_ptr(rsi), r8d)?,
                    (0, _) => a.bt(qword_ptr(rsi), r8)?,
                    (1, 16) => a.bts(word_ptr(rsi), r8w)?,
                    (1, 32) => a.bts(dword_ptr(rsi), r8d)?,
                    (1, _) => a.bts(qword_ptr(rsi), r8)?,
                    (2, 16) => a.btr(word_ptr(rsi), r8w)?,
                    (2, 32) => a.btr(dword_ptr(rsi), r8d)?,
                    (2, _) => a.btr(qword_ptr(rsi), r8)?,
                    (_, 16) => a.btc(word_ptr(rsi), r8w)?,
                    (_, 32) => a.btc(dword_ptr(rsi), r8d)?,
                    _ => a.btc(qword_ptr(rsi), r8)?,
                }
                a.setb(al)?;
                a.cmovb(rdi, r9)
            });
            let offsets: [i64; 12] = [0, 5, 31, 32, 63, 64, 200, -1, -8, -64, -65, -300];
            let mut cases = Vec::new();
            for &e in &offsets {
                for cf in [false, true] {
                    let mut input = random_input(rng, 512);
                    input.ptrs = 1 << SI;
                    input.regs[SI] = 256;
                    input.regs[X8] = bit_offset_value(e, bits, rng);
                    input.flags[CF] = cf;
                    cases.push(input);
                }
            }
            ops.push(op(
                format!("{name}{bits} [m],r; setb cmovb"),
                code,
                None,
                cases,
                bt_undef(),
            ));
        }
    }
    for bits in [32u32, 64] {
        let w = bits == 64;
        for name in ["blsi", "blsr", "blsmsk"] {
            let code = asm(|a| {
                match (name, w) {
                    ("blsi", true) => a.blsi(rdx, r8)?,
                    ("blsi", false) => a.blsi(edx, r8d)?,
                    ("blsr", true) => a.blsr(rdx, r8)?,
                    ("blsr", false) => a.blsr(edx, r8d)?,
                    (_, true) => a.blsmsk(rdx, r8)?,
                    _ => a.blsmsk(edx, r8d)?,
                }
                a.setb(al)?;
                a.sete(cl)?;
                a.sets(bl)?;
                a.seto(dil)?;
                a.cmovle(r9, r8)
            });
            ops.push(op(
                format!("{name}{bits}; setb sete sets seto cmovle"),
                code,
                Some("bmi1"),
                reg_cases(rng, &two, 0, |_, _| {}),
                undef(&[AF, PF]),
            ));
        }
        let code = asm(|a| {
            if w {
                a.andn(rdx, r8, r9)?;
            } else {
                a.andn(edx, r8d, r9d)?;
            }
            a.sete(al)?;
            a.sets(cl)?;
            a.setb(bl)?;
            a.setg(dil)
        });
        ops.push(op(
            format!("andn{bits}; sete sets setb setg"),
            code,
            Some("bmi1"),
            reg_cases(rng, &two, 0, |_, _| {}),
            undef(&[AF, PF]),
        ));
        let code = asm(|a| {
            if w {
                a.bextr(rdx, r8, r9)?;
            } else {
                a.bextr(edx, r8d, r9d)?;
            }
            a.sete(al)?;
            a.setb(cl)?;
            a.seto(bl)?;
            a.cmova(rdi, rdx)
        });
        ops.push(op(
            format!("bextr{bits}; sete setb seto cmova"),
            code,
            Some("bmi1"),
            reg_cases(rng, &[(X8, Src::Value), (X9, Src::Control)], 0, |_, _| {}),
            undef(&[AF, SF, PF]),
        ));
        let code = asm(|a| {
            if w {
                a.bzhi(rdx, r8, r9)?;
            } else {
                a.bzhi(edx, r8d, r9d)?;
            }
            a.setb(al)?;
            a.sete(cl)?;
            a.sets(bl)?;
            a.setl(dil)
        });
        ops.push(op(
            format!("bzhi{bits}; setb sete sets setl"),
            code,
            Some("bmi2"),
            reg_cases(rng, &[(X8, Src::Value), (X9, Src::Count)], 0, |_, _| {}),
            undef(&[AF, PF]),
        ));
    }
    // lzcnt feeding branches (CF, then CF|ZF).
    let code = asm(|a| {
        let mut zero = a.create_label();
        let mut done = a.create_label();
        a.lzcnt(eax, ecx)?;
        a.jb(zero)?;
        a.jbe(done)?;
        a.mov(edx, 1)?;
        a.jmp(done)?;
        a.set_label(&mut zero)?;
        a.mov(edx, 2)?;
        a.set_label(&mut done)?;
        a.nop()
    });
    ops.push(op(
        "lzcnt32; jb jbe",
        code,
        Some("lzcnt"),
        reg_cases(rng, &[(CX, Src::Value)], 0, |_, _| {}),
        cnt_undef(),
    ));
    // Buddy-allocator shapes: order = 31 - lzcnt(size); a shrx loop bound; a shlx order
    // weight; a bextr of a packed order field with the control in the destination register.
    let code = asm(|a| {
        a.lzcnt(eax, ecx)?;
        a.mov(esi, 0x1f)?;
        a.sub(esi, eax)?;
        a.shrx(r9d, edx, esi)?;
        a.mov(eax, 1)?;
        a.shlx(eax, eax, esi)?;
        a.mov(r10d, 0x50f)?;
        a.bextr(r10d, r8d, r10d)?;
        a.cmp(r10d, 0x17)?;
        a.seta(bl)?;
        a.test(r9d, r9d)?;
        a.sete(dil)
    });
    ops.push(op(
        "buddy order/shrx/shlx/bextr",
        code,
        Some("bmi2"),
        reg_cases(
            rng,
            &[(CX, Src::Value), (DX, Src::Value), (X8, Src::Value)],
            0,
            |_, _| {},
        ),
        // `test` leaves AF undefined.
        undef(&[AF]),
    ));
    ops
}

#[test]
fn aliased_bit_forms_and_flag_consumers_match_the_host_cpu() {
    let test = "aliased_bit_forms_and_flag_consumers_match_the_host_cpu";
    if !exec_or_skip(test) {
        return;
    }
    let mut rng = Lcg(0xA11A5);
    let mut ops = alias_ops(&mut rng);
    ops.extend(flag_consumer_ops(&mut rng));
    let ops = supported(test, ops);
    report(test, &differential("hostdiff_bit_alias", &ops));
}

// ----- rotates through carry, double shifts ----------------------------------------------------

fn masked(count: u64, bits: u32) -> u64 {
    count & if bits == 64 { 63 } else { 31 }
}

fn shift_ops(rng: &mut Lcg) -> Vec<Op> {
    let mut ops = Vec::new();
    for left in [true, false] {
        let name = if left { "rcl" } else { "rcr" };
        for bits in [8u32, 16, 32, 64] {
            let undefined = move |input: &Input| {
                let m = masked(input.regs[CX], bits);
                let mut u = Undef::default();
                u.flags[OF] = m > 1;
                u
            };
            let code = asm(|a| match (left, bits) {
                (true, 8) => a.rcl(dl, cl),
                (true, 16) => a.rcl(dx, cl),
                (true, 32) => a.rcl(edx, cl),
                (true, _) => a.rcl(rdx, cl),
                (false, 8) => a.rcr(dl, cl),
                (false, 16) => a.rcr(dx, cl),
                (false, 32) => a.rcr(edx, cl),
                _ => a.rcr(rdx, cl),
            });
            let cases = reg_cases(rng, &[(DX, Src::Value), (CX, Src::Count)], 0, |_, _| {});
            ops.push(op(
                format!("{name}{bits} cl"),
                code,
                None,
                cases,
                Box::new(undefined),
            ));
            for imm in [1u32, 9, 17, 33] {
                let code = asm(|a| match (left, bits) {
                    (true, 8) => a.rcl(dl, imm),
                    (true, 16) => a.rcl(dx, imm),
                    (true, 32) => a.rcl(edx, imm),
                    (true, _) => a.rcl(rdx, imm),
                    (false, 8) => a.rcr(dl, imm),
                    (false, 16) => a.rcr(dx, imm),
                    (false, 32) => a.rcr(edx, imm),
                    _ => a.rcr(rdx, imm),
                });
                let m = masked(u64::from(imm), bits);
                let cases = reg_cases(rng, &[(DX, Src::Value)], 0, |_, _| {});
                ops.push(op(
                    format!("{name}{bits} {imm}"),
                    code,
                    None,
                    cases,
                    if m > 1 { undef(&[OF]) } else { no_undef() },
                ));
            }
        }
        // the dedicated count-1 encoding (D0 /2, D0 /3)
        ops.push(op(
            format!("{name}8 1 (D0)"),
            vec![0xD0, if left { 0xD2 } else { 0xDA }],
            None,
            reg_cases(rng, &[(DX, Src::Value)], 0, |_, _| {}),
            no_undef(),
        ));
    }
    for left in [true, false] {
        let name = if left { "shld" } else { "shrd" };
        for bits in [16u32, 32, 64] {
            let undefined =
                move |input: &Input| shift_double_undef(masked(input.regs[CX], bits), bits);
            let code = asm(|a| match (left, bits) {
                (true, 16) => a.shld(dx, r8w, cl),
                (true, 32) => a.shld(edx, r8d, cl),
                (true, _) => a.shld(rdx, r8, cl),
                (false, 16) => a.shrd(dx, r8w, cl),
                (false, 32) => a.shrd(edx, r8d, cl),
                _ => a.shrd(rdx, r8, cl),
            });
            let cases = reg_cases(
                rng,
                &[(DX, Src::Value), (X8, Src::Value), (CX, Src::Count)],
                0,
                |_, _| {},
            );
            ops.push(op(
                format!("{name}{bits} cl"),
                code,
                None,
                cases,
                Box::new(undefined),
            ));
            for imm in [1u32, 7, 16] {
                let code = asm(|a| match (left, bits) {
                    (true, 16) => a.shld(dx, r8w, imm),
                    (true, 32) => a.shld(edx, r8d, imm),
                    (true, _) => a.shld(rdx, r8, imm),
                    (false, 16) => a.shrd(dx, r8w, imm),
                    (false, 32) => a.shrd(edx, r8d, imm),
                    _ => a.shrd(rdx, r8, imm),
                });
                let u = shift_double_undef(masked(u64::from(imm), bits), bits);
                let cases = reg_cases(rng, &[(DX, Src::Value), (X8, Src::Value)], 0, |_, _| {});
                ops.push(op(
                    format!("{name}{bits} {imm}"),
                    code,
                    None,
                    cases,
                    Box::new(move |_| u),
                ));
            }
        }
    }
    ops
}

fn shift_double_undef(count: u64, bits: u32) -> Undef {
    let mut u = Undef::default();
    if count == 0 {
        return u;
    }
    u.flags[AF] = true;
    u.flags[OF] = count > 1;
    if bits == 16 && count > 16 {
        u.flags = [true; 6];
        u.regs = 1 << DX;
    }
    u
}

#[test]
fn rotates_and_double_shifts_match_the_host_cpu() {
    let test = "rotates_and_double_shifts_match_the_host_cpu";
    if !exec_or_skip(test) {
        return;
    }
    let mut rng = Lcg(0x5417);
    let ops = shift_ops(&mut rng);
    report(test, &differential("hostdiff_shifts", &ops));
}

// ----- exchange, compare-exchange, misc -----------------------------------------------------------

/// Sets the accumulator (low `bits`) equal to the destination value half of the time.
fn match_acc(input: &mut Input, rng: &mut Lcg, dest: u64, bits: u32) {
    if rng.next().is_multiple_of(2) {
        let m = if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        input.regs[AX] = (input.regs[AX] & !m) | (dest & m);
    }
}

fn exchange_ops(rng: &mut Lcg, locked: bool) -> Vec<Op> {
    let mut ops = Vec::new();
    let prefix = if locked { "lock " } else { "" };
    let lock = |a: &mut CodeAssembler| {
        if locked {
            let _ = a.lock();
        }
    };
    for bits in [8u32, 16, 32, 64] {
        let k = u64::from(bits / 8);
        // xadd [m], r
        let code = asm(|a| {
            lock(a);
            match bits {
                8 => a.xadd(byte_ptr(rsi), r8b),
                16 => a.xadd(word_ptr(rsi), r8w),
                32 => a.xadd(dword_ptr(rsi), r8d),
                _ => a.xadd(qword_ptr(rsi), r8),
            }
        });
        ops.push(op(
            format!("{prefix}xadd{bits} [m],r"),
            code,
            None,
            mem_cases(rng, SI, 16, &[(X8, Src::Value)], 48, |_, _| {}),
            no_undef(),
        ));
        // cmpxchg [m], r
        let code = asm(|a| {
            lock(a);
            match bits {
                8 => a.cmpxchg(byte_ptr(rsi), r8b),
                16 => a.cmpxchg(word_ptr(rsi), r8w),
                32 => a.cmpxchg(dword_ptr(rsi), r8d),
                _ => a.cmpxchg(qword_ptr(rsi), r8),
            }
        });
        ops.push(op(
            format!("{prefix}cmpxchg{bits} [m],r"),
            code,
            None,
            mem_cases(rng, SI, 16, &[(X8, Src::Value)], 48, move |input, rng| {
                let dest = get(&input.mem, 16, k);
                match_acc(input, rng, dest, bits);
            }),
            no_undef(),
        ));
        if locked {
            // alu [m], r
            for (name, und) in [
                ("add", &[][..]),
                ("adc", &[][..]),
                ("sub", &[][..]),
                ("sbb", &[][..]),
                ("and", &[AF][..]),
                ("or", &[AF][..]),
                ("xor", &[AF][..]),
            ] {
                let code = asm(|a| {
                    let _ = a.lock();
                    let m8 = byte_ptr(rsi);
                    let m16 = word_ptr(rsi);
                    let m32 = dword_ptr(rsi);
                    let m64 = qword_ptr(rsi);
                    match (name, bits) {
                        ("add", 8) => a.add(m8, r8b),
                        ("add", 16) => a.add(m16, r8w),
                        ("add", 32) => a.add(m32, r8d),
                        ("add", _) => a.add(m64, r8),
                        ("adc", 8) => a.adc(m8, r8b),
                        ("adc", 16) => a.adc(m16, r8w),
                        ("adc", 32) => a.adc(m32, r8d),
                        ("adc", _) => a.adc(m64, r8),
                        ("sub", 8) => a.sub(m8, r8b),
                        ("sub", 16) => a.sub(m16, r8w),
                        ("sub", 32) => a.sub(m32, r8d),
                        ("sub", _) => a.sub(m64, r8),
                        ("sbb", 8) => a.sbb(m8, r8b),
                        ("sbb", 16) => a.sbb(m16, r8w),
                        ("sbb", 32) => a.sbb(m32, r8d),
                        ("sbb", _) => a.sbb(m64, r8),
                        ("and", 8) => a.and(m8, r8b),
                        ("and", 16) => a.and(m16, r8w),
                        ("and", 32) => a.and(m32, r8d),
                        ("and", _) => a.and(m64, r8),
                        ("or", 8) => a.or(m8, r8b),
                        ("or", 16) => a.or(m16, r8w),
                        ("or", 32) => a.or(m32, r8d),
                        ("or", _) => a.or(m64, r8),
                        (_, 8) => a.xor(m8, r8b),
                        (_, 16) => a.xor(m16, r8w),
                        (_, 32) => a.xor(m32, r8d),
                        _ => a.xor(m64, r8),
                    }
                });
                // Memory operand values at the width boundaries.
                let cases = mem_cases(
                    rng,
                    SI,
                    16,
                    &[(X8, Src::Value), (X9, Src::Value)],
                    48,
                    move |input, _| {
                        let v = input.regs[X9];
                        put(&mut input.mem, 16, k, v);
                    },
                );
                let list: &'static [usize] = und;
                ops.push(op(
                    format!("lock {name}{bits} [m],r"),
                    code,
                    None,
                    cases,
                    undef(list),
                ));
            }
            for name in ["inc", "dec", "not", "neg"] {
                let code = asm(|a| {
                    let _ = a.lock();
                    let m8 = byte_ptr(rsi);
                    let m16 = word_ptr(rsi);
                    let m32 = dword_ptr(rsi);
                    let m64 = qword_ptr(rsi);
                    match (name, bits) {
                        ("inc", 8) => a.inc(m8),
                        ("inc", 16) => a.inc(m16),
                        ("inc", 32) => a.inc(m32),
                        ("inc", _) => a.inc(m64),
                        ("dec", 8) => a.dec(m8),
                        ("dec", 16) => a.dec(m16),
                        ("dec", 32) => a.dec(m32),
                        ("dec", _) => a.dec(m64),
                        ("not", 8) => a.not(m8),
                        ("not", 16) => a.not(m16),
                        ("not", 32) => a.not(m32),
                        ("not", _) => a.not(m64),
                        (_, 8) => a.neg(m8),
                        (_, 16) => a.neg(m16),
                        (_, 32) => a.neg(m32),
                        _ => a.neg(m64),
                    }
                });
                let cases = mem_cases(rng, SI, 16, &[(X9, Src::Value)], 48, move |input, _| {
                    let v = input.regs[X9];
                    put(&mut input.mem, 16, k, v);
                });
                ops.push(op(
                    format!("lock {name}{bits} [m]"),
                    code,
                    None,
                    cases,
                    no_undef(),
                ));
            }
            // xchg [m], r (implicitly locked; also without the prefix)
            for explicit in [false, true] {
                let code = asm(|a| {
                    if explicit {
                        let _ = a.lock();
                    }
                    match bits {
                        8 => a.xchg(byte_ptr(rsi), r8b),
                        16 => a.xchg(word_ptr(rsi), r8w),
                        32 => a.xchg(dword_ptr(rsi), r8d),
                        _ => a.xchg(qword_ptr(rsi), r8),
                    }
                });
                ops.push(op(
                    format!("{}xchg{bits} [m],r", if explicit { "lock " } else { "" }),
                    code,
                    None,
                    mem_cases(rng, SI, 16, &[(X8, Src::Value)], 48, |_, _| {}),
                    no_undef(),
                ));
            }
        } else {
            // register forms
            let code = asm(|a| match bits {
                8 => a.xadd(dl, r8b),
                16 => a.xadd(dx, r8w),
                32 => a.xadd(edx, r8d),
                _ => a.xadd(rdx, r8),
            });
            ops.push(op(
                format!("xadd{bits} r,r"),
                code,
                None,
                reg_cases(rng, &[(DX, Src::Value), (X8, Src::Value)], 0, |_, _| {}),
                no_undef(),
            ));
            let code = asm(|a| match bits {
                8 => a.xadd(dl, dl),
                16 => a.xadd(dx, dx),
                32 => a.xadd(edx, edx),
                _ => a.xadd(rdx, rdx),
            });
            ops.push(op(
                format!("xadd{bits} same"),
                code,
                None,
                reg_cases(rng, &[(DX, Src::Value)], 0, |_, _| {}),
                no_undef(),
            ));
            let code = asm(|a| match bits {
                8 => a.cmpxchg(dl, r8b),
                16 => a.cmpxchg(dx, r8w),
                32 => a.cmpxchg(edx, r8d),
                _ => a.cmpxchg(rdx, r8),
            });
            ops.push(op(
                format!("cmpxchg{bits} r,r"),
                code,
                None,
                reg_cases(
                    rng,
                    &[(DX, Src::Value), (X8, Src::Value)],
                    0,
                    move |input, rng| {
                        let dest = input.regs[DX];
                        match_acc(input, rng, dest, bits);
                    },
                ),
                no_undef(),
            ));
            // destination is the accumulator itself
            let code = asm(|a| match bits {
                8 => a.cmpxchg(al, r8b),
                16 => a.cmpxchg(ax, r8w),
                32 => a.cmpxchg(eax, r8d),
                _ => a.cmpxchg(rax, r8),
            });
            ops.push(op(
                format!("cmpxchg{bits} acc,r"),
                code,
                None,
                reg_cases(rng, &[(AX, Src::Value), (X8, Src::Value)], 0, |_, _| {}),
                no_undef(),
            ));
        }
    }
    // cmpxchg8b / cmpxchg16b (16-byte aligned operand)
    let code = asm(|a| {
        lock(a);
        a.cmpxchg8b(qword_ptr(rsi))
    });
    ops.push(op(
        format!("{prefix}cmpxchg8b"),
        code,
        None,
        mem_cases(
            rng,
            SI,
            16,
            &[(AX, Src::Value), (DX, Src::Value)],
            48,
            |input, rng| {
                if rng.next() % 2 == 0 {
                    let lo = input.regs[AX] & 0xFFFF_FFFF;
                    let hi = input.regs[DX] & 0xFFFF_FFFF;
                    put(&mut input.mem, 16, 8, lo | (hi << 32));
                }
            },
        ),
        no_undef(),
    ));
    let code = asm(|a| {
        lock(a);
        a.cmpxchg16b(xmmword_ptr(rsi))
    });
    ops.push(op(
        format!("{prefix}cmpxchg16b"),
        code,
        Some("cmpxchg16b"),
        mem_cases(
            rng,
            SI,
            16,
            &[(AX, Src::Value), (DX, Src::Value)],
            48,
            |input, rng| {
                match rng.next() % 4 {
                    0 | 1 => {
                        let (lo, hi) = (input.regs[AX], input.regs[DX]);
                        put(&mut input.mem, 16, 8, lo);
                        put(&mut input.mem, 24, 8, hi);
                    }
                    2 => {
                        // only the low half matches
                        let lo = input.regs[AX];
                        put(&mut input.mem, 16, 8, lo);
                    }
                    _ => {}
                }
            },
        ),
        no_undef(),
    ));
    ops
}

fn misc_ops(rng: &mut Lcg) -> Vec<Op> {
    let mut ops = Vec::new();
    for (name, code) in [
        ("clc", asm(|a| a.clc())),
        ("stc", asm(|a| a.stc())),
        ("cmc", asm(|a| a.cmc())),
        ("cld", asm(|a| a.cld())),
        ("std", asm(|a| a.std())),
        ("lahf", asm(|a| a.lahf())),
        ("sahf", asm(|a| a.sahf())),
    ] {
        let mut cases = reg_cases(rng, &[(AX, Src::Value)], 0, |_, _| {});
        for (i, c) in cases.iter_mut().enumerate() {
            c.df = i % 3 == 0;
        }
        // `std` sets DF; the trampoline clears it again only after reading RFLAGS.
        ops.push(op(name, code, None, cases, no_undef()));
    }
    ops.push(op(
        "xlat",
        asm(|a| a.xlatb()),
        None,
        reg_cases(rng, &[(AX, Src::Value)], 256, |input, _| {
            input.ptrs |= 1 << BX;
            input.regs[BX] = 0;
        }),
        no_undef(),
    ));
    for bits in [32u32, 64] {
        let w = bits == 64;
        ops.push(op(
            format!("adcx{bits}"),
            asm(|a| if w { a.adcx(rdx, r8) } else { a.adcx(edx, r8d) }),
            Some("adx"),
            reg_cases(rng, &[(DX, Src::Value), (X8, Src::Value)], 0, |_, _| {}),
            no_undef(),
        ));
        ops.push(op(
            format!("adox{bits}"),
            asm(|a| if w { a.adox(rdx, r8) } else { a.adox(edx, r8d) }),
            Some("adx"),
            reg_cases(
                rng,
                &[(DX, Src::Value), (X8, Src::Value)],
                0,
                |input, rng| {
                    input.flags[OF] = rng.next() & 1 == 1;
                },
            ),
            no_undef(),
        ));
    }
    for bits in [16u32, 32, 64] {
        ops.push(op(
            format!("movbe{bits} r,[m]"),
            asm(|a| match bits {
                16 => a.movbe(dx, word_ptr(rsi)),
                32 => a.movbe(edx, dword_ptr(rsi)),
                _ => a.movbe(rdx, qword_ptr(rsi)),
            }),
            Some("movbe"),
            mem_cases(rng, SI, 8, &[(DX, Src::Value)], 32, |_, _| {}),
            no_undef(),
        ));
        ops.push(op(
            format!("movbe{bits} [m],r"),
            asm(|a| match bits {
                16 => a.movbe(word_ptr(rsi), r8w),
                32 => a.movbe(dword_ptr(rsi), r8d),
                _ => a.movbe(qword_ptr(rsi), r8),
            }),
            Some("movbe"),
            mem_cases(rng, SI, 8, &[(X8, Src::Value)], 32, |_, _| {}),
            no_undef(),
        ));
    }
    for (name, code) in [
        ("crc32 r32,r8", asm(|a| a.crc32(edx, r8b))),
        ("crc32 r32,r16", asm(|a| a.crc32(edx, r8w))),
        ("crc32 r32,r32", asm(|a| a.crc32(edx, r8d))),
        ("crc32 r64,r8", asm(|a| a.crc32(rdx, r8b))),
        ("crc32 r64,r64", asm(|a| a.crc32(rdx, r8))),
        ("crc32 r32,[m]", asm(|a| a.crc32(edx, dword_ptr(rsi)))),
    ] {
        let cases = mem_cases(
            rng,
            SI,
            8,
            &[(DX, Src::Value), (X8, Src::Value)],
            32,
            |_, _| {},
        );
        ops.push(op(name, code, Some("sse4.2"), cases, no_undef()));
    }
    ops
}

#[test]
fn exchange_and_misc_instructions_match_the_host_cpu() {
    let test = "exchange_and_misc_instructions_match_the_host_cpu";
    if !exec_or_skip(test) {
        return;
    }
    let mut rng = Lcg(0xC4A6);
    let mut ops = exchange_ops(&mut rng, false);
    ops.extend(misc_ops(&mut rng));
    let ops = supported(test, ops);
    report(test, &differential("hostdiff_exchange", &ops));
}

#[test]
fn locked_instructions_match_the_host_cpu() {
    let test = "locked_instructions_match_the_host_cpu";
    if !exec_or_skip(test) || !shared_memory_or_skip(test) {
        return;
    }
    let mut rng = Lcg(0x10CC);
    let mut ops = exchange_ops(&mut rng, true);
    ops.extend(
        bit_ops(&mut rng)
            .into_iter()
            .filter(|o| o.name.starts_with("lock ")),
    );
    let ops = supported(test, ops);
    report(test, &differential("hostdiff_locked", &ops));
}
