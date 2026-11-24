//! Tool-gated execution tests for the string, bit, atomic and miscellaneous integer
//! instructions: counter branches, stack frames and RFLAGS, the runtime-provided instructions,
//! large and overlapping `rep movs`/`rep stos`, AMD hints, multiply-high, a multi-threaded test of
//! the atomics on a real shared memory, and compilation of random code from these families.

use std::fmt::Write as _;

use iced_x86::code_asm::*;

use super::exec::{
    CODE_BASE, Case, DATA_ADDR, GUEST_BASE, HarnessMemory, MEMORY_BYTES, Outcome, R8, R9, R10, R11,
    RAX, RBP, RBX, RCX, RDI, RDX, RSI, RSP, STACK_TOP, STATE_ADDR, build, build_with_memory,
    exec_or_skip, run, run_script, shared_memory_link_args, shared_memory_or_skip,
};
use super::ir_integer::random_integer_program;
use super::{Program, asm_function};
use crate::lift::{STATE_FLAGS_OFFSET, STATE_FS_BASE_OFFSET, STATE_GPR_OFFSET};

const RUNNER_THREADS_JS: &str = include_str!("runner_threads.js");
const DF_ADDR: u64 = STATE_ADDR + STATE_FLAGS_OFFSET + 6;

fn run_clean(built: &super::exec::Built, cases: &[Case]) -> Vec<Outcome> {
    let outcomes = run(built, cases);
    for (i, o) in outcomes.iter().enumerate() {
        assert!(o.error.is_empty(), "case {i} trapped: {}", o.error);
        assert_eq!(o.regs[RSP], STACK_TOP, "case {i}: rsp not restored by ret");
    }
    outcomes
}

#[test]
fn counter_branches_execute() {
    if !exec_or_skip("counter_branches_execute") {
        return;
    }
    let base = CODE_BASE;
    let functions = Program::new()
        // eax = sum(1..=rcx) with loop, guarded by jrcxz (a 32-bit sum: LLVM rewrites a 64-bit
        // one into a closed form with an i128 multiply, i.e. a compiler-rt `__multi3` call)
        .asm(base, |a| {
            let mut top = a.create_label();
            let mut done = a.create_label();
            a.xor(eax, eax)?;
            a.jrcxz(done)?;
            a.set_label(&mut top)?;
            a.add(eax, ecx)?;
            a.loop_(top)?;
            a.set_label(&mut done)?;
            a.ret()
        })
        // mov eax, 1 ; jecxz +5 (67 E3: the 64-bit-mode encoding) ; mov eax, 2 ; ret
        .bytes(
            base + 0x100,
            &[0xB8, 1, 0, 0, 0, 0x67, 0xE3, 0x05, 0xB8, 2, 0, 0, 0, 0xC3],
        )
        .asm(base + 0x200, |a| {
            let mut zero = a.create_label();
            a.mov(eax, 1)?;
            a.jrcxz(zero)?;
            a.mov(eax, 2)?;
            a.set_label(&mut zero)?;
            a.ret()
        })
        // scan while the byte is zero (loope) / until it is 0x55 (loopne)
        .asm(base + 0x300, |a| {
            let mut top = a.create_label();
            a.mov(rax, -1i64)?;
            a.set_label(&mut top)?;
            a.inc(rax)?;
            a.cmp(byte_ptr(rdx + rax), 0)?;
            a.loope(top)?;
            a.ret()
        })
        .asm(base + 0x400, |a| {
            let mut top = a.create_label();
            a.mov(rax, -1i64)?;
            a.set_label(&mut top)?;
            a.inc(rax)?;
            a.cmp(byte_ptr(rdx + rax), 0x55)?;
            a.loopne(top)?;
            a.ret()
        })
        .discover();
    let built = build("counter_branches", functions.values());

    let sums = [0u64, 1, 5, 1000];
    let tests = [0u64, 1, 0x1_0000_0000, 0xFFFF_FFFF_0000_0000];
    let bytes = [0u8, 0, 0, 7, 0, 0x55, 0, 9, 0x55, 0];
    let counts = [1u64, 2, 3, 4, 5, 6, 7, 10];
    let mut cases = Vec::new();
    cases.extend(sums.iter().map(|&n| Case::new(base).reg(RCX, n)));
    for &v in &tests {
        cases.push(Case::new(base + 0x100).reg(RCX, v));
        cases.push(Case::new(base + 0x200).reg(RCX, v));
    }
    for va in [base + 0x300, base + 0x400] {
        for &n in &counts {
            cases.push(
                Case::new(va)
                    .reg(RCX, n)
                    .reg(RDX, DATA_ADDR)
                    .mem(DATA_ADDR, &bytes),
            );
        }
    }
    let o = run_clean(&built, &cases);
    for (i, &n) in sums.iter().enumerate() {
        assert_eq!(o[i].regs[RAX], n * (n + 1) / 2, "sum({n})");
        assert_eq!(o[i].regs[RCX], 0);
    }
    let o = &o[sums.len()..];
    for (i, &v) in tests.iter().enumerate() {
        let jecxz = o[2 * i].regs[RAX];
        let jrcxz = o[2 * i + 1].regs[RAX];
        assert_eq!(jecxz, if v as u32 == 0 { 1 } else { 2 }, "jecxz {v:#x}");
        assert_eq!(jrcxz, if v == 0 { 1 } else { 2 }, "jrcxz {v:#x}");
    }
    let o = &o[2 * tests.len()..];
    for (k, (target, while_equal)) in [(0u8, true), (0x55u8, false)].into_iter().enumerate() {
        for (i, &n) in counts.iter().enumerate() {
            // Reference model of loope/loopne.
            let (mut idx, mut left) = (u64::MAX, n);
            let zf = loop {
                idx = idx.wrapping_add(1);
                let zf = bytes[idx as usize] == target;
                left -= 1;
                if left == 0 || zf != while_equal {
                    break zf;
                }
            };
            let out = &o[k * counts.len() + i];
            assert_eq!(
                (out.regs[RAX], out.regs[RCX], out.flags[3]),
                (idx, left, zf),
                "loop{} with rcx={n}",
                if while_equal { "e" } else { "ne" }
            );
        }
    }
}

#[test]
fn stack_frames_and_rflags_execute() {
    if !exec_or_skip("stack_frames_and_rflags_execute") {
        return;
    }
    let base = CODE_BASE;
    let functions = Program::new()
        .asm(base, |a| {
            a.enter(0x20u32, 0u32)?;
            a.mov(qword_ptr(rbp - 8), rcx)?;
            a.mov(rax, qword_ptr(rbp - 8))?;
            a.mov(rdx, rsp)?;
            a.mov(r8, rbp)?;
            a.leave()?;
            a.ret()
        })
        .asm(base + 0x100, |a| {
            a.pushfq()?;
            a.pop(rax)?;
            a.ret()
        })
        .asm(base + 0x200, |a| {
            a.push(rcx)?;
            a.popfq()?;
            a.ret()
        })
        .asm(base + 0x300, |a| {
            a.std()?;
            a.pushfq()?;
            a.pop(rax)?;
            a.cld()?;
            a.pushfq()?;
            a.pop(rdx)?;
            a.ret()
        })
        .asm(base + 0x400, |a| {
            a.enter(0x10u32, 1u32)?;
            a.leave()?;
            a.ret()
        })
        .discover();
    let enter_level1 = functions[&(base + 0x400)].entry;
    let built = build("frames", functions.values());
    let rbp_in = 0x1234_5678_9ABC_DEF0u64;
    let flags = [true, false, true, true, false, true]; // CF AF ZF OF set
    let popped = 0x8D5u64 | (1 << 10) | (1 << 9) | (1 << 8); // CF PF AF ZF SF OF, DF, IF, TF
    let cases = [
        Case::new(base)
            .reg(RCX, 0xFEED)
            .reg(RBP, rbp_in)
            .read(STACK_TOP - 16, 8),
        {
            let mut c = Case::new(base + 0x100).mem(DF_ADDR, &[1]);
            c.flags = flags;
            c
        },
        Case::new(base + 0x200)
            .reg(RCX, popped)
            .mem(DF_ADDR, &[0])
            .read(DF_ADDR, 1),
        Case::new(base + 0x300).mem(DF_ADDR, &[0]).read(DF_ADDR, 1),
    ];
    let o = run_clean(&built, &cases);
    assert_eq!(o[0].regs[RAX], 0xFEED);
    assert_eq!(o[0].regs[R8], STACK_TOP - 16, "enter: rbp = frame");
    assert_eq!(
        o[0].regs[RDX],
        STACK_TOP - 16 - 0x20,
        "enter: rsp = frame - size"
    );
    assert_eq!(o[0].regs[RBP], rbp_in, "leave restored rbp");
    assert_eq!(o[0].mem[0], rbp_in.to_le_bytes(), "enter pushed rbp");
    // pushfq: CF(0) PF(2)=0 AF(4) ZF(6) SF(7)=0 OF(11), bit 1, IF, DF
    assert_eq!(
        o[1].regs[RAX],
        1 | 2 | (1 << 4) | (1 << 6) | (1 << 9) | (1 << 10) | (1 << 11)
    );
    // popfq: the six flags and DF come from the popped value; TF/IF are ignored.
    assert_eq!(o[2].flags, [true; 6]);
    assert_eq!(o[2].mem[0], [1], "DF from bit 10");
    assert_eq!(o[3].regs[RAX] >> 10 & 1, 1, "std visible to pushfq");
    assert_eq!(o[3].regs[RDX] >> 10 & 1, 0, "cld visible to pushfq");
    assert_eq!(o[3].mem[0], [0]);

    let trapped = run(&built, &[Case::new(base + 0x400)]);
    assert_eq!(
        trapped[0].error,
        format!("unimplemented instruction at {enter_level1:#x}"),
        "enter with a nesting level traps"
    );
}

#[test]
fn runtime_helper_instructions_execute() {
    if !exec_or_skip("runtime_helper_instructions_execute") {
        return;
    }
    let f = asm_function(CODE_BASE, |a| {
        a.mov(rbx, -1i64)?;
        a.mov(rcx, -1i64)?;
        a.mov(eax, 7)?;
        a.xor(ecx, ecx)?;
        a.cpuid()?;
        a.mov(r8, rax)?;
        a.mov(r9, rbx)?;
        a.mov(r10, rcx)?;
        a.mov(r11, rdx)?;
        a.rdtsc()?;
        a.shl(rdx, 32)?;
        a.or(rax, rdx)?;
        a.mov(rsi, rax)?;
        a.rdtscp()?;
        a.shl(rdx, 32)?;
        a.or(rax, rdx)?;
        a.mov(rdi, rax)?;
        a.mov(rbp, rcx)?;
        a.xor(ecx, ecx)?;
        a.xgetbv()?;
        a.ret()
    });
    let built = build("helpers", &[f]);
    let o = &run_clean(&built, &[Case::new(CODE_BASE)])[0];
    // runner.js stand-ins: cpuid -> leaf ^ 0x11111111, subleaf ^ 0x22222222, 0x33.., 0x44..
    assert_eq!(o.regs[R8], 7 ^ 0x1111_1111);
    assert_eq!(o.regs[R9], 0x2222_2222, "rbx zero-extended");
    assert_eq!(o.regs[R10], 0x3333_3333);
    assert_eq!(o.regs[R11], 0x4444_4444);
    assert!(
        o.regs[RDI] > o.regs[RSI] && o.regs[RSI] > 0,
        "tsc increases"
    );
    assert_eq!(o.regs[RBP], 7, "rdtscp: ecx = TSC_AUX");
    assert_eq!((o.regs[RAX], o.regs[RDX]), (7, 0), "xgetbv(0)");
}

/// Reference model of `rep movs` / `rep stos` on a flat byte array (element by element).
fn model_movs(mem: &mut [u8], base: u64, src: u64, dst: u64, n: u64, size: u64, df: bool) {
    for j in 0..n {
        let (s, d) = if df {
            (src - j * size, dst - j * size)
        } else {
            (src + j * size, dst + j * size)
        };
        let (s, d) = ((s - base) as usize, (d - base) as usize);
        let element: Vec<u8> = mem[s..s + size as usize].to_vec();
        mem[d..d + size as usize].copy_from_slice(&element);
    }
}

fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed) ^ (i >> 8) as u8)
        .collect()
}

#[test]
fn large_and_overlapping_rep_string_operations_execute() {
    if !exec_or_skip("large_and_overlapping_rep_string_operations_execute") {
        return;
    }
    let base = CODE_BASE;
    // fs: rep movsb (segment override on the source)
    let fs_movsb = [0x64, 0xF3, 0xA4, 0xC3];
    let functions = Program::new()
        .asm(base, |a| a.rep().movsb().and_then(|_| a.ret()))
        .asm(base + 0x100, |a| a.rep().movsq().and_then(|_| a.ret()))
        .asm(base + 0x200, |a| a.rep().stosb().and_then(|_| a.ret()))
        .asm(base + 0x300, |a| a.rep().stosq().and_then(|_| a.ret()))
        .asm(base + 0x400, |a| a.rep().stosd().and_then(|_| a.ret()))
        .bytes(base + 0x500, &fs_movsb)
        .discover();
    let built = build("big_strings", functions.values());

    let a_addr = 0x40_0000u64;
    let b_addr = 0x50_0000u64;
    let big = 100_000usize;
    let src = pattern(big, 3);
    let with_df = |c: Case, df: bool| c.mem(DF_ADDR, &[u8::from(df)]);
    // (case, region start, region length, initial bytes, model)
    type Model = Box<dyn Fn(&mut [u8], u64)>;
    let mut cases: Vec<(Case, u64, usize, Model, u64, u64)> = Vec::new();
    // forward, non-overlapping, 100k bytes (bulk path)
    let region = |len: usize| pattern(len, 9);
    cases.push((
        with_df(Case::new(base), false)
            .reg(RSI, a_addr)
            .reg(RDI, b_addr)
            .reg(RCX, big as u64)
            .mem(a_addr, &src),
        b_addr,
        big,
        Box::new(|_, _| {}),
        a_addr + big as u64,
        b_addr + big as u64,
    ));
    // forward overlap by one byte: replicates the first byte (element loop)
    for (gap, n, df, size, va) in [
        (1i64, 1000u64, false, 1u64, base),
        (3, 500, true, 1, base),
        (-5, 700, false, 1, base),
        (4, 300, false, 8, base + 0x100),
        (16, 300, true, 8, base + 0x100),
        (-8, 300, true, 8, base + 0x100),
    ] {
        let src_va = a_addr + 0x8000;
        let dst_va = (src_va as i64 + gap) as u64;
        let model: Model = Box::new(move |mem: &mut [u8], start: u64| {
            model_movs(mem, start, src_va, dst_va, n, size, df);
        });
        let delta = n * size;
        let (rsi_out, rdi_out) = if df {
            (src_va - delta, dst_va - delta)
        } else {
            (src_va + delta, dst_va + delta)
        };
        cases.push((
            with_df(Case::new(va), df)
                .reg(RSI, src_va)
                .reg(RDI, dst_va)
                .reg(RCX, n)
                .mem(a_addr, &region(0x10000)),
            a_addr,
            0x10000,
            model,
            rsi_out,
            rdi_out,
        ));
    }
    let mut outcomes_expected = Vec::new();
    let mut run_cases = Vec::new();
    for (case, start, len, model, rsi_out, rdi_out) in cases {
        let mut initial = vec![0u8; len];
        // The initial bytes of the observed region are whatever the case wrote there.
        for (addr, bytes) in &case.mem {
            if *addr <= start && start < *addr + bytes.len() as u64 {
                let off = (start - addr) as usize;
                let n = len.min(bytes.len() - off);
                initial[..n].copy_from_slice(&bytes[off..off + n]);
            }
        }
        if start == b_addr {
            initial = src.clone();
        } else {
            model(&mut initial, start);
        }
        outcomes_expected.push((initial, rsi_out, rdi_out));
        run_cases.push(case.read(start, len));
    }
    // stores: 100k `stosb`, 10k zero qwords, non-splat dwords downwards
    let c_addr = 0x60_0000u64;
    run_cases.push(
        with_df(Case::new(base + 0x200), false)
            .reg(RDI, c_addr)
            .reg(RCX, big as u64)
            .reg(RAX, 0x5A)
            .read(c_addr, big + 1),
    );
    run_cases.push(
        with_df(Case::new(base + 0x300), false)
            .reg(RDI, c_addr)
            .reg(RCX, 10_000)
            .reg(RAX, 0)
            .mem(c_addr, &vec![0xEE; 80_008])
            .read(c_addr, 80_008),
    );
    run_cases.push(
        with_df(Case::new(base + 0x400), true)
            .reg(RDI, c_addr + 4 * 999)
            .reg(RCX, 1000)
            .reg(RAX, 0x1234_5678)
            .mem(c_addr, &vec![0xEE; 4004])
            .read(c_addr, 4004),
    );
    // fs-relative source
    let fs_base = a_addr;
    run_cases.push(
        with_df(Case::new(base + 0x500), false)
            .reg(RSI, 0x10)
            .reg(RDI, b_addr)
            .reg(RCX, 64)
            .mem(a_addr, &src[..0x100])
            .mem(STATE_ADDR + STATE_FS_BASE_OFFSET, &fs_base.to_le_bytes())
            .read(b_addr, 64),
    );
    let o = run_clean(&built, &run_cases);
    for (i, (expected, rsi_out, rdi_out)) in outcomes_expected.iter().enumerate() {
        assert_eq!(o[i].regs[RCX], 0, "case {i}: rcx");
        assert_eq!(
            (o[i].regs[RSI], o[i].regs[RDI]),
            (*rsi_out, *rdi_out),
            "case {i}"
        );
        assert!(
            o[i].mem[0] == *expected,
            "case {i}: memory differs from the element model"
        );
    }
    let s = &o[outcomes_expected.len()..];
    assert!(s[0].mem[0][..big].iter().all(|&b| b == 0x5A));
    assert_eq!(s[0].mem[0][big], 0, "stosb stops after rcx bytes");
    assert_eq!(s[0].regs[RDI], c_addr + big as u64);
    assert!(s[1].mem[0][..80_000].iter().all(|&b| b == 0));
    assert_eq!(&s[1].mem[0][80_000..], &[0xEE; 8]);
    let dwords: Vec<u32> = s[2].mem[0][..4000]
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert!(dwords.iter().all(|&d| d == 0x1234_5678), "stosd downwards");
    assert_eq!(&s[2].mem[0][4000..], &[0xEE; 4]);
    assert_eq!(
        s[2].regs[RDI],
        c_addr - 4,
        "rdi after 1000 dwords downwards"
    );
    assert_eq!(s[3].mem[0], src[0x10..0x50], "fs: source");
    assert_eq!(s[3].regs[RSI], 0x10 + 64, "rsi excludes the segment base");
}

#[test]
fn amd_hints_clzero_and_multiply_high_execute() {
    if !exec_or_skip("amd_hints_clzero_and_multiply_high_execute") {
        return;
    }
    // clzero; monitorx; mwaitx; prefetchw [rax]; clflush [rax]; ret
    let hints = [
        0x0F, 0x01, 0xFC, 0x0F, 0x01, 0xFA, 0x0F, 0x01, 0xFB, 0x0F, 0x0D, 0x08, 0x0F, 0xAE, 0x38,
        0xC3,
    ];
    let functions = Program::new()
        .bytes(CODE_BASE, &hints)
        // unsigned division by 10 through the high half of a multiply only
        .asm(CODE_BASE + 0x100, |a| {
            a.mov(rax, 0xCCCC_CCCC_CCCC_CCCDu64)?;
            a.mul(rcx)?;
            a.shr(rdx, 3)?;
            a.mov(rax, rdx)?;
            a.ret()
        })
        // mulx with equal destinations keeps only the high half
        .asm(CODE_BASE + 0x200, |a| {
            a.mulx(r8, r8, rcx)?;
            a.ret()
        })
        .discover();
    let built = build("hints", functions.values());
    let line = DATA_ADDR + 0x40;
    let values = [0u64, 9, 10, 99, 12345, u64::MAX, 1 << 63];
    let mut cases = vec![
        Case::new(CODE_BASE)
            .reg(RAX, line + 7)
            .mem(DATA_ADDR, &[0xFF; 0x100])
            .read(DATA_ADDR, 0x100),
    ];
    for &v in &values {
        cases.push(Case::new(CODE_BASE + 0x100).reg(RCX, v));
        cases.push(
            Case::new(CODE_BASE + 0x200)
                .reg(RCX, v)
                .reg(RDX, 0xDEAD_BEEF_1234_5678),
        );
    }
    let o = run_clean(&built, &cases);
    let mut expected = vec![0xFFu8; 0x100];
    expected[0x40..0x80].fill(0);
    assert_eq!(o[0].mem[0], expected, "clzero zeroes the 64-byte line");
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(o[1 + 2 * i].regs[RAX], v / 10, "{v} / 10");
        let hi = ((u128::from(v) * 0xDEAD_BEEF_1234_5678u128) >> 64) as u64;
        assert_eq!(o[2 + 2 * i].regs[R8], hi, "mulx high half of {v:#x}");
    }
}

#[test]
fn misaligned_cmpxchg16b_and_fences_execute() {
    let test = "misaligned_cmpxchg16b_and_fences_execute";
    if !exec_or_skip(test) || !shared_memory_or_skip(test) {
        return;
    }
    let f = asm_function(CODE_BASE, |a| {
        a.mfence()?;
        a.lfence()?;
        a.sfence()?;
        a.lock().cmpxchg16b(xmmword_ptr(rsi))?;
        a.ret()
    });
    let site = f.blocks[&CODE_BASE].instructions[3].ip();
    let built = build("cx16", &[f]);
    let o = run(
        &built,
        &[
            Case::new(CODE_BASE)
                .reg(RSI, DATA_ADDR)
                .reg(RAX, 1)
                .reg(RDX, 2)
                .reg(RBX, 3)
                .reg(RCX, 4)
                .mem(DATA_ADDR, &[1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0])
                .read(DATA_ADDR, 16),
            Case::new(CODE_BASE).reg(RSI, DATA_ADDR + 8),
        ],
    );
    assert_eq!(o[0].error, "");
    assert!(o[0].flags[3], "ZF on success");
    assert_eq!(
        o[0].mem[0],
        [3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(
        o[1].error,
        format!("trap at {site:#x}"),
        "#GP on a misaligned operand"
    );
}

#[test]
fn atomics_have_no_lost_updates_across_threads() {
    atomics_across_threads(
        "atomics_have_no_lost_updates_across_threads",
        &HarnessMemory::from_env(),
    );
}

/// The threads test under a given memory model (R19 runs it with the counters and the guest
/// stacks inside a translated window: `exec_memory.rs`).
pub(super) fn atomics_across_threads(test: &str, memory: &HarnessMemory) {
    if !exec_or_skip(test) || !shared_memory_or_skip(test) {
        return;
    }
    let base = CODE_BASE;
    let functions = Program::new()
        // lock inc qword [rdx], rcx times
        .asm(base, |a| {
            let mut top = a.create_label();
            a.set_label(&mut top)?;
            a.lock().inc(qword_ptr(rdx))?;
            a.dec(rcx)?;
            a.jnz(top)?;
            a.ret()
        })
        // compare-exchange increment loop
        .asm(base + 0x100, |a| {
            let mut top = a.create_label();
            let mut retry = a.create_label();
            a.set_label(&mut top)?;
            a.mov(rax, qword_ptr(rdx))?;
            a.set_label(&mut retry)?;
            a.lea(r8, qword_ptr(rax + 1))?;
            a.lock().cmpxchg(qword_ptr(rdx), r8)?;
            a.jnz(retry)?;
            a.dec(rcx)?;
            a.jnz(top)?;
            a.ret()
        })
        // lock xadd dword [rdx], 1
        .asm(base + 0x200, |a| {
            let mut top = a.create_label();
            a.set_label(&mut top)?;
            a.mov(r8d, 1)?;
            a.lock().xadd(dword_ptr(rdx), r8d)?;
            a.dec(rcx)?;
            a.jnz(top)?;
            a.ret()
        })
        // 128-bit counter with lock cmpxchg16b (carries into the high half)
        .asm(base + 0x300, |a| {
            let mut top = a.create_label();
            let mut retry = a.create_label();
            a.set_label(&mut top)?;
            a.mov(rax, qword_ptr(rsi))?;
            a.mov(rdx, qword_ptr(rsi + 8))?;
            a.set_label(&mut retry)?;
            a.mov(rbx, rax)?;
            a.mov(rcx, rdx)?;
            a.add(rbx, 1)?;
            a.adc(rcx, 0)?;
            a.lock().cmpxchg16b(xmmword_ptr(rsi))?;
            a.jnz(retry)?;
            a.dec(r9)?;
            a.jnz(top)?;
            a.ret()
        })
        // spinlock: acquire with xchg, plain increment inside, release with lock btr
        .asm(base + 0x400, |a| {
            let mut acquire = a.create_label();
            a.set_label(&mut acquire)?;
            a.mov(eax, 1)?;
            a.xchg(dword_ptr(rdx), eax)?;
            a.test(eax, eax)?;
            a.jnz(acquire)?;
            a.inc(qword_ptr(rdx + 8))?;
            a.lock().btr(dword_ptr(rdx), 0)?;
            a.dec(rcx)?;
            a.jnz(acquire)?;
            a.ret()
        })
        // negative control: a plain (non-atomic) increment; the fence keeps LLVM from merging
        // the iterations, so concurrent workers lose updates
        .asm(base + 0x500, |a| {
            let mut top = a.create_label();
            a.set_label(&mut top)?;
            a.inc(qword_ptr(rdx))?;
            a.mfence()?;
            a.dec(rcx)?;
            a.jnz(top)?;
            a.ret()
        })
        .discover();
    let mut extra = vec!["--import-memory".to_string()];
    extra.extend(shared_memory_link_args());
    extra.push("--export-if-defined=__stack_pointer".to_string());
    let built = build_with_memory("threads", functions.values(), &extra, memory, GUEST_BASE);

    const WORKERS: u64 = 4;
    // The lifted loops are short, so on a loaded or small machine one worker can finish before
    // the next is even scheduled. Each attempt is a fresh run with four times the iterations,
    // until the non-atomic control proves that calls really interleaved (it lost updates) or the
    // attempts run out.
    const ATTEMPTS: [u64; 4] = [100_000, 400_000, 1_600_000, 3_200_000];
    let counters = DATA_ADDR;
    let (inc, cas, xadd, wide, spin, plain) = (
        counters,
        counters + 16,
        counters + 32,
        counters + 48,
        counters + 64,
        counters + 80,
    );
    let wide_start: u128 = u128::from(u64::MAX - 1000) | (5u128 << 64);
    let init: String = wide_start
        .to_le_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let mut last_note = String::new();
    for iterations in ATTEMPTS {
        let mut workers = String::new();
        for w in 0..WORKERS {
            if w > 0 {
                workers.push(',');
            }
            let calls = [
                (base, vec![(RCX, iterations), (RDX, inc)]),
                (base + 0x100, vec![(RCX, iterations), (RDX, cas)]),
                (base + 0x200, vec![(RCX, iterations), (RDX, xadd)]),
                (base + 0x300, vec![(R9, iterations), (RSI, wide)]),
                (base + 0x400, vec![(RCX, iterations), (RDX, spin)]),
                (base + 0x500, vec![(RCX, iterations), (RDX, plain)]),
            ];
            let calls: Vec<String> = calls
                .iter()
                .map(|(va, regs)| {
                    let regs: Vec<String> = regs
                        .iter()
                        .map(|(i, v)| format!("[{i},\"{v:#x}\"]"))
                        .collect();
                    format!("{{\"fn\":\"sub_{va:X}\",\"regs\":[{}]}}", regs.join(","))
                })
                .collect();
            let _ = write!(
                workers,
                "{{\"state\":\"{:#x}\",\"stackTop\":\"{:#x}\",\"shadowStack\":\"{:#x}\",\"calls\":[{}]}}",
                STATE_ADDR + w * 0x200,
                STACK_TOP - w * 0x1_0000,
                0x70_0000 + (w + 1) * 0x1_0000,
                calls.join(",")
            );
        }
        let spec = format!(
            "{{\"pages\":{},\"gprOffset\":{STATE_GPR_OFFSET},\"guestBase\":\"{:#x}\",{}\"init\":[[\"{wide:#x}\",\"{init}\"]],\"read\":[[\"{counters:#x}\",88]],\"workers\":[{workers}]}}",
            MEMORY_BYTES / 65536,
            built.guest_base,
            if built.memory_json.is_empty() {
                String::new()
            } else {
                format!("\"memory\":{},", built.memory_json)
            }
        );
        let out = run_script(&built, "runner_threads.js", RUNNER_THREADS_JS, &[spec]);
        if let Some(reason) = out.trim().strip_prefix("skip: ") {
            println!("SKIP {test}: {reason}");
            return;
        }
        println!("{out}");
        let first = out.lines().next().unwrap_or_default().trim();
        assert!(!first.starts_with("error"), "{first}");
        let bytes: Vec<u8> = (0..first.len() / 2)
            .map(|i| u8::from_str_radix(&first[2 * i..2 * i + 2], 16).expect("hex"))
            .collect();
        let u64_at = |off: usize| u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        // The no-lost-updates checks are strict on every attempt, whether or not the threads
        // overlapped: a lost update is a bug either way.
        let total = WORKERS * iterations;
        assert_eq!(u64_at(0), total, "lock inc");
        assert_eq!(u64_at(16), total, "lock cmpxchg loop");
        assert_eq!(u64_at(32) & 0xFFFF_FFFF, total, "lock xadd");
        let wide_out = u128::from_le_bytes(bytes[48..64].try_into().unwrap());
        assert_eq!(wide_out, wide_start + u128::from(total), "lock cmpxchg16b");
        assert_eq!(u64_at(64), 0, "spinlock released");
        assert_eq!(
            u64_at(72),
            total,
            "xchg/btr spinlock protects a plain increment"
        );
        // Peak number of calls in flight at once, per call (see runner_threads.js).
        let peaks: Vec<u64> = out
            .lines()
            .filter_map(|l| l.strip_prefix("overlap "))
            .filter_map(|l| l.split(' ').nth(1)?.parse().ok())
            .collect();
        let survived = u64_at(80);
        println!(
            "non-atomic control: {survived} of {total} increments survived; peak concurrent calls {peaks:?}"
        );
        // The mutation check: a plain increment loses updates exactly when calls interleave, so
        // a lost update in the control is proof that the atomic loops above ran under real
        // contention and the harness can see a lost update. Observed overlap alone is not
        // enough (a late worker may just catch the tail of an early one's call).
        if survived < total {
            assert!(
                peaks[5] >= 2,
                "the control lost updates, so calls must have overlapped: {out}"
            );
            println!(
                "OVERLAP OBSERVED {test}: control lost {} updates",
                total - survived
            );
            return;
        }
        last_note = format!("peak concurrent calls {peaks:?}");
    }
    println!(
        "SKIP {test}: the workers never interleaved (the non-atomic control lost no update in \
         any attempt up to {} iterations; {last_note}), so this machine offers no real \
         concurrency; the atomic counters were still exact",
        ATTEMPTS[ATTEMPTS.len() - 1]
    );
}

#[test]
fn lifted_random_integer_code_compiles_and_links() {
    let test = "lifted_random_integer_code_compiles_and_links";
    if !exec_or_skip(test) || !shared_memory_or_skip(test) {
        return;
    }
    for seed in [0x1A7E, 0xBEEF] {
        let functions = random_integer_program(150, seed);
        let _ = build("random_integer", functions.values());
    }
}
