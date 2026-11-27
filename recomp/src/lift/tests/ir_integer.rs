//! Always-run IR tests for the string, bit, atomic and miscellaneous integer instructions:
//! the atomics/target-features policy, bulk string paths, runtime helper declarations, precise
//! traps for unsupported forms, and a fuzz test biased towards these instruction families.

use std::collections::BTreeMap;

use iced_x86::code_asm::*;

use super::ir::Lcg;
use super::{Program, asm_function};
use crate::cfg;
use crate::lift::{
    ATOMICS_TARGET_FEATURES, BULK_MEMORY_TARGET_FEATURES, CMPXCHG16B_LOCKS, HELPER_CPUID,
    HELPER_RDTSC, HELPER_RDTSCP, HELPER_XGETBV, LIFTED_FUNCTION_ATTRIBUTES, lift_function,
    lift_module,
};

fn module_ir(build: impl FnOnce(&mut CodeAssembler) -> Result<(), iced_x86::IcedError>) -> String {
    let f = asm_function(0x1000, build);
    let module = lift_module([&f]).expect("lifts");
    assert!(
        module.functions[0].unsupported.is_empty(),
        "{:?}",
        module.functions[0].unsupported
    );
    module.ir
}

/// The single unsupported instruction of a function made of `bytes` followed by `ret`.
fn unsupported_reason(bytes: &[u8]) -> (String, String) {
    let mut code = bytes.to_vec();
    code.push(0xC3);
    let f = Program::new().bytes(0x1000, &code).function(0x1000);
    let lifted = lift_function(&f).expect("lifts");
    assert_eq!(lifted.unsupported.len(), 1, "{:?}", lifted.unsupported);
    let u = &lifted.unsupported[0];
    (u.mnemonic.clone(), u.reason.clone())
}

#[test]
fn constant_32_bit_register_writes_fold_the_zero_extension() {
    let ir = module_ir(|a| {
        a.xor(r9d, r9d)?;
        a.mov(eax, -1i32)?;
        a.ret()
    });
    assert!(!ir.contains("zext i32 0 to i64"), "{ir}");
    assert!(!ir.contains("zext i32 -1 to i64"), "{ir}");
    assert!(ir.contains("store i64 4294967295, ptr %rax"), "{ir}");
}

#[test]
fn non_atomic_modules_keep_the_default_target_features() {
    let ir = module_ir(|a| {
        a.add(rax, rcx)?;
        a.xadd(rdx, r8)?;
        a.cmpxchg(qword_ptr(rcx), rdx)?;
        a.bts(dword_ptr(rcx), eax)?;
        a.ret()
    });
    assert!(
        ir.contains(&format!(
            "attributes #0 = {{ {LIFTED_FUNCTION_ATTRIBUTES} }}"
        )),
        "{ir}"
    );
    assert!(
        !ir.contains("+atomics") && !ir.contains("+bulk-memory"),
        "{ir}"
    );
    assert!(
        !ir.contains("atomicrmw") && !ir.contains("cmpxchg ptr"),
        "{ir}"
    );
}

#[test]
fn locked_instructions_become_seq_cst_atomics() {
    let program = Program::new().asm(0x1000, |a| {
        a.lock().add(qword_ptr(rcx), rdx)?;
        a.lock().xadd(dword_ptr(rcx + 8), eax)?;
        a.lock().cmpxchg(word_ptr(rcx + 16), dx)?;
        a.lock().inc(byte_ptr(rcx + 18))?;
        a.lock().bts(qword_ptr(rcx), rax)?;
        a.lock().neg(qword_ptr(rcx + 24))?;
        a.xchg(qword_ptr(rcx + 32), r8)?;
        a.lock().cmpxchg16b(xmmword_ptr(rcx + 48))?;
        a.mfence()?;
        a.ret()
    });
    let functions = program.discover();
    let module = lift_module(functions.values()).expect("lifts");
    assert!(module.functions[0].unsupported.is_empty());
    assert!(module.requires_shared_memory());
    let ir = &module.ir;
    for needle in [
        "atomicrmw add ptr %t",
        "atomicrmw xchg ptr %t",
        "atomicrmw or ptr %t",
        "cmpxchg ptr %t",
        "fence seq_cst",
    ] {
        assert!(ir.contains(needle), "{needle}:\n{ir}");
    }
    // Every atomic is sequentially consistent and naturally aligned.
    for line in ir.lines().filter(|l| l.contains("atomicrmw ")) {
        assert!(line.contains(" seq_cst, align "), "{line}");
    }
    assert!(ir.contains(" i64 1 seq_cst, align 1") || ir.contains("i8 1 seq_cst, align 1"));
    assert!(
        ir.contains("i16 %t") && ir.contains("seq_cst seq_cst, align 2"),
        "{ir}"
    );
    assert!(ir.contains(&format!("@{CMPXCHG16B_LOCKS} = linkonce_odr hidden global")));
    assert!(ir.contains(&format!(
        "attributes #0 = {{ \"target-features\"=\"+simd128,{ATOMICS_TARGET_FEATURES}\" }}"
    )));
    assert!(
        ir.contains("define void @sub_1000(ptr noalias noundef %state) unnamed_addr #0 {")
    );
    assert!(
        ir.lines()
            .filter(|l| l.starts_with("define "))
            .all(|l| l.ends_with(" #0 {")),
        "every definition carries the features"
    );
}

#[test]
fn rep_movs_and_stos_have_bulk_paths_and_exact_loops() {
    let ir = module_ir(|a| {
        a.rep().movsb()?;
        a.rep().stosq()?;
        a.repne().scasb()?;
        a.ret()
    });
    assert!(ir.contains("call void @llvm.memmove.p0.p0.i64("), "{ir}");
    assert!(ir.contains("call void @llvm.memset.p0.i64("), "{ir}");
    assert!(
        !ir.contains("llvm.memcpy"),
        "memcpy is never equivalent: {ir}"
    );
    assert!(ir.contains("declare void @llvm.memmove.p0.p0.i64(ptr, ptr, i64, i1)"));
    assert!(ir.contains(&format!(
        "attributes #0 = {{ \"target-features\"=\"+simd128,{BULK_MEMORY_TARGET_FEATURES}\" }}"
    )));
    // Each repeated instruction keeps its element loop.
    assert_eq!(ir.matches("\nrep_head").count(), 3, "{ir}");
    // DF is read from the guest state.
    assert!(ir.contains("i32 2, i64 6"), "{ir}");
}

#[test]
fn runtime_helpers_are_declared_only_when_used() {
    let plain = module_ir(|a| {
        a.mov(eax, 1)?;
        a.ret()
    });
    for helper in [HELPER_CPUID, HELPER_RDTSC, HELPER_RDTSCP, HELPER_XGETBV] {
        assert!(!plain.contains(helper));
    }
    let ir = module_ir(|a| {
        a.cpuid()?;
        a.rdtsc()?;
        a.rdtscp()?;
        a.xgetbv()?;
        a.ret()
    });
    for helper in [HELPER_CPUID, HELPER_RDTSC, HELPER_RDTSCP, HELPER_XGETBV] {
        assert!(
            ir.contains(&format!("declare void @{helper}(ptr noundef)")),
            "{helper}"
        );
        assert!(
            ir.contains(&format!("call void @{helper}(ptr %state)")),
            "{helper}"
        );
    }
}

#[test]
fn loop_and_jrcxz_become_llvm_branches() {
    let f = asm_function(0x1000, |a| {
        let mut top = a.create_label();
        let mut done = a.create_label();
        a.jrcxz(done)?;
        a.set_label(&mut top)?;
        a.inc(eax)?;
        a.loopne(top)?;
        a.set_label(&mut done)?;
        a.ret()
    });
    let lifted = lift_function(&f).expect("lifts");
    assert!(lifted.unsupported.is_empty(), "{:?}", lifted.unsupported);
    assert!(lifted.ir.contains("br i1 %t"), "{}", lifted.ir);
}

/// `while (*(u16*)rcx < 0x100) pause;`: the shape of the title's page-commit wait.
fn pause_spin_module() -> String {
    module_ir(|a| {
        let mut top = a.create_label();
        a.set_label(&mut top)?;
        a.pause()?;
        a.movzx(eax, word_ptr(rcx))?;
        a.cmp(ax, 0x100)?;
        a.jb(top)?;
        a.ret()
    })
}

/// `lock inc [rax]; while (*(u32*)(rsp + 0x48) != 8);`: a join wait without `pause`.
fn plain_spin_module() -> String {
    module_ir(|a| {
        let mut top = a.create_label();
        a.lock().inc(dword_ptr(rax))?;
        a.set_label(&mut top)?;
        a.mov(eax, dword_ptr(rsp + 0x48))?;
        a.cmp(eax, 8)?;
        a.jne(top)?;
        a.ret()
    })
}

#[test]
fn pause_and_loop_heads_are_compiler_barriers() {
    for ir in [pause_spin_module(), plain_spin_module()] {
        assert!(ir.contains(crate::lift::LOOP_BARRIER), "{ir}");
    }
    // Straight-line code without tail jumps gets none, and the barrier needs no wasm atomics.
    let straight = module_ir(|a| {
        a.mov(eax, dword_ptr(rcx))?;
        a.ret()
    });
    assert!(!straight.contains("fence"), "{straight}");
    assert!(!pause_spin_module().contains(ATOMICS_TARGET_FEATURES));
}

/// After the middle-end a spin loop must still reload the guest word on every iteration: a
/// plain load in a loop without stores is loop-invariant and LLVM turns the wait into `jmp .`.
#[test]
fn spin_waits_keep_their_load_inside_the_loop_after_optimisation() {
    for ir in [pause_spin_module(), plain_spin_module()] {
        spin_wait_reloads(&ir);
    }
}

fn spin_wait_reloads(ir: &str) {
    let Some(crate::lift::compile::LlvmTool::Llc { opt: Some(opt), .. }) =
        crate::lift::compile::find_llvm_tool()
    else {
        eprintln!("skipping: no LLVM opt found");
        return;
    };
    let ll = std::env::temp_dir().join(format!("playgta6_recomp_spin_{}.ll", std::process::id()));
    std::fs::write(&ll, ir).expect("write ll");
    let out = std::process::Command::new(opt)
        .args(["-O2", "-S"])
        .arg(&ll)
        .output()
        .expect("runs opt");
    let _ = std::fs::remove_file(&ll);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let optimized = String::from_utf8_lossy(&out.stdout);
    // The basic block holding the fence must load before it branches back.
    let mut in_fence_block = false;
    let mut load_after_fence = false;
    for line in optimized.lines() {
        let line = line.trim();
        if line.starts_with(crate::lift::LOOP_BARRIER) {
            in_fence_block = true;
        } else if in_fence_block && line.contains(" = load ") {
            load_after_fence = true;
        } else if in_fence_block && line.starts_with("br ") {
            break;
        }
    }
    assert!(load_after_fence, "{optimized}");
}

/// A spin built from tail jumps between two functions: `sub_1000` polls `*(u16*)rcx` and jumps
/// to `sub_2000` while it is below 0x100, and `sub_2000` jumps straight back. Neither function
/// has a backward edge of its own, but once LLVM inlines `sub_2000` into `sub_1000` the self
/// tail call becomes a loop (tail-recursion elimination).
fn tail_jump_spin_module() -> String {
    let program = Program::new()
        .asm(0x1000, |a| {
            let mut done = a.create_label();
            a.movzx(eax, word_ptr(rcx))?;
            a.cmp(ax, 0x100)?;
            a.jae(done)?;
            a.jmp(0x2000)?;
            a.set_label(&mut done)?;
            a.ret()
        })
        .asm(0x2000, |a| a.jmp(0x1000));
    let functions = program.discover();
    let module = lift_module(functions.values()).expect("lifts");
    for f in &module.functions {
        assert!(f.unsupported.is_empty(), "{:?}", f.unsupported);
    }
    module.ir
}

#[test]
fn function_entries_are_compiler_barriers() {
    let ir = tail_jump_spin_module();
    for name in ["sub_1000", "sub_2000"] {
        let header = format!("define void @{name}(");
        let start = ir.find(&header).unwrap_or_else(|| panic!("{name} missing:\n{ir}"));
        let entry: String = ir[start..]
            .lines()
            .skip(2)
            .take_while(|l| !l.ends_with(':'))
            .collect::<Vec<_>>()
            .join("\n");
        let fence = entry.find(crate::lift::LOOP_BARRIER);
        let branch = entry.find("  br label %bb_");
        assert!(
            matches!((fence, branch), (Some(f), Some(b)) if f < b),
            "{name}: entry block ends with the barrier and its branch:\n{entry}"
        );
    }
}

/// After the middle-end the inlined tail-jump cycle must still reload the polled word.
#[test]
fn tail_jump_spins_keep_their_load_after_optimisation() {
    let ir = tail_jump_spin_module();
    let Some(crate::lift::compile::LlvmTool::Llc { opt: Some(opt), .. }) =
        crate::lift::compile::find_llvm_tool()
    else {
        eprintln!("skipping: no LLVM opt found");
        return;
    };
    let ll = std::env::temp_dir().join(format!(
        "playgta6_recomp_tail_spin_{}.ll",
        std::process::id()
    ));
    std::fs::write(&ll, &ir).expect("write ll");
    let out = std::process::Command::new(opt)
        .args(["-O2", "-S"])
        .arg(&ll)
        .output()
        .expect("runs opt");
    let _ = std::fs::remove_file(&ll);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let optimized = String::from_utf8_lossy(&out.stdout);
    let start = optimized
        .find("@sub_1000(")
        .and_then(|i| optimized[..i].rfind("define "))
        .expect("sub_1000 survives");
    let body: String = optimized[start..]
        .lines()
        .take_while(|l| *l != "}")
        .collect::<Vec<_>>()
        .join("\n");
    // Every block that holds the barrier loads the polled word after it, before it branches.
    let mut fences = 0;
    let mut in_fence_block = false;
    for line in body.lines().map(str::trim) {
        if line.starts_with(crate::lift::LOOP_BARRIER) {
            fences += 1;
            in_fence_block = true;
        } else if in_fence_block && line.contains(" = load i16") {
            in_fence_block = false;
        } else if in_fence_block && (line.starts_with("br ") || line.starts_with("ret ")) {
            panic!("barrier block without the polled load:\n{body}");
        }
    }
    assert!(fences > 0, "the barrier survives optimisation:\n{body}");
    // The cycle really was closed into a loop (otherwise the test proves nothing).
    assert!(
        !body.contains("@sub_2000(") || body.contains("br label"),
        "{body}"
    );
}

#[test]
fn unsupported_integer_forms_trap_with_precise_reasons() {
    let cases: [(&[u8], &str, &str); 9] = [
        // addr32 movsb (ESI/EDI)
        (&[0x67, 0xA4], "movsb", "32-bit address size"),
        // repne movsb (reserved prefix)
        (&[0xF2, 0xA4], "movsb", "reserved repeat prefix"),
        // enter 0x10, 1
        (&[0xC8, 0x10, 0x00, 0x01], "enter", "nesting level 1"),
        // pushf (16-bit)
        (&[0x66, 0x9C], "pushf", "16-bit pushf/popf"),
        // insb
        (&[0x6C], "insb", "string instruction Insb"),
        // in al, dx
        (&[0xEC], "in", "not supported yet"),
        // rdrand eax
        (&[0x0F, 0xC7, 0xF0], "rdrand", "not supported yet"),
        // addr32 clzero
        (&[0x67, 0x0F, 0x01, 0xFC], "clzero", "32-bit address size"),
        // data16 leave
        (&[0x66, 0xC9], "leave", "16/32-bit leave"),
    ];
    for (bytes, mnemonic, reason) in cases {
        let (m, r) = unsupported_reason(bytes);
        assert_eq!(m, mnemonic, "{bytes:02x?}");
        assert!(r.contains(reason), "{bytes:02x?}: {r}");
    }
}

/// Random code built from the opcodes of the integer families this module covers, with random
/// legacy/REX prefixes (lock, rep, operand/address size, segment overrides), ModRM bytes and
/// trailing bytes.
pub(super) fn random_integer_program(count: usize, seed: u64) -> BTreeMap<u64, cfg::Function> {
    const OPCODES: &[&[u8]] = &[
        &[0xA4],
        &[0xA5],
        &[0xA6],
        &[0xA7],
        &[0xAA],
        &[0xAB],
        &[0xAC],
        &[0xAD],
        &[0xAE],
        &[0xAF],
        &[0x0F, 0xA3],
        &[0x0F, 0xAB],
        &[0x0F, 0xB3],
        &[0x0F, 0xBB],
        &[0x0F, 0xBA],
        &[0x0F, 0xBC],
        &[0x0F, 0xBD],
        &[0x0F, 0xB8],
        &[0x0F, 0xC8],
        &[0x0F, 0xCB],
        &[0x0F, 0xA4],
        &[0x0F, 0xA5],
        &[0x0F, 0xAC],
        &[0x0F, 0xAD],
        &[0xD0],
        &[0xD1],
        &[0xD2],
        &[0xD3],
        &[0xC0],
        &[0xC1],
        &[0x0F, 0xC0],
        &[0x0F, 0xC1],
        &[0x0F, 0xB0],
        &[0x0F, 0xB1],
        &[0x0F, 0xC7],
        &[0x86],
        &[0x87],
        &[0x80],
        &[0x81],
        &[0x83],
        &[0xF6],
        &[0xF7],
        &[0xFE],
        &[0xFF],
        &[0x00],
        &[0x01],
        &[0x10],
        &[0x11],
        &[0x18],
        &[0x19],
        &[0x20],
        &[0x21],
        &[0x28],
        &[0x29],
        &[0x30],
        &[0x31],
        &[0xE0],
        &[0xE1],
        &[0xE2],
        &[0xE3],
        &[0xC8],
        &[0xC9],
        &[0x9C],
        &[0x9D],
        &[0x9E],
        &[0x9F],
        &[0xF5],
        &[0xF8],
        &[0xF9],
        &[0xFC],
        &[0xFD],
        &[0xD7],
        &[0x0F, 0xA2],
        &[0x0F, 0x31],
        &[0x0F, 0x01],
        &[0x0F, 0x38, 0xF0],
        &[0x0F, 0x38, 0xF1],
        &[0x0F, 0x38, 0xF6],
        &[0x0F, 0xAE],
        &[0x0F, 0x18],
        &[0x0F, 0x0D],
        &[0xC4, 0xE2, 0x78],
        &[0xC4, 0xE2, 0xF8],
        &[0xC4, 0xE2, 0x7B],
        &[0xC4, 0xE2, 0xFA],
        &[0xC4, 0xE2, 0x79],
        &[0xC4, 0xE2, 0xFB],
        &[0xC4, 0xE3, 0x7B],
        &[0xC4, 0xE3, 0xFB],
    ];
    const PREFIXES: [u8; 10] = [0xF0, 0xF2, 0xF3, 0x66, 0x67, 0x64, 0x65, 0x48, 0x4C, 0x41];
    let mut rng = Lcg(seed);
    let mut program = Program::new();
    for i in 0..count {
        let mut bytes = Vec::new();
        for _ in 0..(4 + rng.next() % 9) {
            for _ in 0..rng.next() % 3 {
                bytes.push(PREFIXES[(rng.next() % PREFIXES.len() as u64) as usize]);
            }
            bytes.extend_from_slice(OPCODES[(rng.next() % OPCODES.len() as u64) as usize]);
            // ModRM (register and memory forms) plus displacement/immediate bytes.
            for _ in 0..1 + rng.next() % 6 {
                bytes.push(rng.next() as u8);
            }
        }
        bytes.push(0xC3);
        program = program.bytes(0x40_0000 + (i as u64) * 0x100, &bytes);
    }
    program.discover()
}

#[test]
fn random_integer_family_code_never_panics() {
    for seed in [0x1A7E, 0xBEEF, 0x5151] {
        let functions = random_integer_program(400, seed);
        let module = lift_module(functions.values()).expect("discovered random code lifts");
        assert_eq!(module.functions.len(), functions.len());
        let lifted: usize = module.functions.iter().map(|f| f.unsupported.len()).sum();
        // Mostly supported instructions: far fewer traps than functions.
        assert!(
            lifted < functions.len(),
            "{lifted} traps in {} functions",
            functions.len()
        );
    }
}

#[test]
fn r60_bit_manip_rotates_bswap_and_double_shifts_emit_compact_intrinsics_and_elide_dead_flags() {
    // Every bit-manipulation/rotate/double-shift/bswap op followed by a flag-overwriting `xor`
    // must emit native LLVM intrinsics and elide dead flag computations (no parity ctpop.i8, no
    // constant-count flag selects, constant-folded bit-test masks).
    let ir = module_ir(|a| {
        a.bsf(eax, ecx)?;
        a.bsr(edx, ecx)?;
        a.tzcnt(r8, rcx)?;
        a.lzcnt(r9, rcx)?;
        a.popcnt(r10d, ecx)?;
        a.popcnt(r11, rcx)?;
        a.rol(eax, 7)?;
        a.ror(r8, 13)?;
        a.shld(eax, edx, 5)?;
        a.shrd(r8, r9, 9)?;
        a.bswap(eax)?;
        a.bswap(r8)?;
        a.bts(eax, 5)?;
        a.btr(r8, 17)?;
        a.rorx(ecx, eax, 0)?;
        a.andn(r10, r8, r9)?;
        a.blsi(r11d, eax)?;
        a.bzhi(edx, eax, ecx)?;
        // Overwrites CF/OF/ZF/SF/PF so all preceding instructions' flags are dead.
        a.xor(eax, r10d)?;
        a.ret()
    });
    for intrinsic in [
        "@llvm.cttz.i32(",
        "@llvm.ctlz.i32(",
        "@llvm.cttz.i64(",
        "@llvm.ctlz.i64(",
        "@llvm.ctpop.i32(",
        "@llvm.ctpop.i64(",
        "@llvm.fshl.i32(",
        "@llvm.fshr.i64(",
        "@llvm.bswap.i32(",
        "@llvm.bswap.i64(",
    ] {
        assert!(ir.contains(intrinsic), "missing {intrinsic}:\n{ir}");
    }
    let fn_start = ir.find("define void @sub_1000(").expect("sub_1000");
    let fn_ir = &ir[fn_start..];
    // Only the final `xor` computes parity (not `shld`/`shrd`!).
    assert_eq!(
        fn_ir.matches("@llvm.ctpop.i8(").count(),
        1,
        "dead shld/shrd PF must be elided:\n{fn_ir}"
    );
    // Constant-count `shld`/`shrd` and `rol`/`ror` never emit `select i1 ..., i1` flag muxes.
    assert!(
        !fn_ir.lines().any(|l| l.contains("select i1 ") && l.contains(", i1 ")),
        "dead/imm shift flags must not emit flag selects:\n{fn_ir}"
    );
    // `bts eax, 5` -> `or i32 %..., 32`, `btr r8, 17` -> `and i64 %..., -131073` (no `shl`!).
    assert!(
        fn_ir.contains("or i32 ") && fn_ir.contains(", 32\n"),
        "bts imm mask should be constant-folded:\n{fn_ir}"
    );
    assert!(
        fn_ir.contains("and i64 ") && fn_ir.contains(", -131073\n"),
        "btr imm mask should be constant-folded:\n{fn_ir}"
    );
    // `rorx ecx, eax, 0` must not emit an `fshr.i32` call.
    assert!(
        !fn_ir.contains("@llvm.fshr.i32("),
        "rorx by 0 should not emit fshr:\n{fn_ir}"
    );
    // Native retargeting keeps the exact same standard LLVM intrinsics (no llvm.wasm.*).
    let native = crate::lift::target::LiftTarget::native_host().retarget(ir);
    assert!(!native.contains("llvm.wasm."), "{native}");
}

#[test]
fn r60_same_block_predicate_fusion_and_scalar_lowering_compactness() {
    // 1. `cmp rax, rbx; jle taken` with flags overwritten in both successors:
    //    must emit a single `icmp sle i64` and ZERO `sub`, ZERO `OF` chain, ZERO flag stores!
    let cmp_jle = module_ir(|a| {
        let mut taken = a.create_label();
        a.cmp(rax, rbx)?;
        a.jle(taken)?;
        a.add(rax, 1)?;
        a.ret()?;
        a.set_label(&mut taken)?;
        a.add(rax, 2)?;
        a.ret()
    });
    let entry_body = {
        let start = cmp_jle.find("bb_1000:\n").expect("bb_1000");
        let rest = &cmp_jle[start + "bb_1000:\n".len()..];
        let end = rest.find(":\n").map_or(rest.len(), |i| rest[..i].rfind('\n').unwrap_or(0));
        &rest[..end]
    };
    assert!(
        entry_body.contains("icmp sle i64 "),
        "cmp+jle must fuse to icmp sle:\n{entry_body}"
    );
    assert!(
        !entry_body.contains(" = sub i64 "),
        "fused cmp+jle with dead flags must not emit sub:\n{entry_body}"
    );
    assert!(
        !entry_body.contains("store i1 "),
        "fused cmp+jle with dead flags must not store flags:\n{entry_body}"
    );
    assert!(
        !entry_body.contains(" = xor i1 ") && !entry_body.contains(" = or i1 "),
        "fused cmp+jle must not combine boolean flags:\n{entry_body}"
    );

    // 2. `test eax, eax; jg taken` with flags overwritten in both successors:
    //    must emit a single `icmp sgt i32` and ZERO flag stores!
    let test_jg = module_ir(|a| {
        let mut taken = a.create_label();
        a.test(eax, eax)?;
        a.jg(taken)?;
        a.xor(eax, eax)?;
        a.ret()?;
        a.set_label(&mut taken)?;
        a.add(eax, 1)?;
        a.ret()
    });
    let test_body = {
        let start = test_jg.find("bb_1000:\n").expect("bb_1000");
        let rest = &test_jg[start + "bb_1000:\n".len()..];
        let end = rest.find(":\n").map_or(rest.len(), |i| rest[..i].rfind('\n').unwrap_or(0));
        &rest[..end]
    };
    assert!(
        test_body.contains("icmp sgt i32 "),
        "test+jg must fuse to icmp sgt:\n{test_body}"
    );
    assert!(
        !test_body.contains("store i1 "),
        "fused test+jg with dead flags must not store flags:\n{test_body}"
    );

    // 3. `cmp` -> `setl` + `cmovg` + `dec` -> `jle` + `movzx` + `lea r32` + 32-bit shifts + 32-bit div/idiv.
    let scalar = module_ir(|a| {
        let mut done = a.create_label();
        a.movzx(eax, cl)?;
        a.movzx(edx, byte_ptr(rsi))?;
        a.lea(r8d, dword_ptr(rcx + rdx * 4 + 16))?;
        a.lea(r10d, dword_ptr(rcx + rdx * 4 + 16))?;
        a.shl(eax, 3)?;
        a.shr(edx, cl)?;
        a.sar(r8d, 5)?;
        a.cmp(eax, edx)?;
        a.setl(r9b)?;
        a.cmovg(eax, r8d)?;
        a.xor(edx, edx)?;
        a.div(r8d)?;
        a.cdq()?;
        a.idiv(r8d)?;
        a.dec(rcx)?;
        a.jle(done)?;
        a.add(eax, 1)?;
        a.ret()?;
        a.set_label(&mut done)?;
        a.add(eax, 2)?;
        a.ret()
    });
    let fn_start = scalar.find("define void @sub_1000(").expect("sub_1000");
    let fn_ir = &scalar[fn_start..];
    // `movzx eax, cl` -> `and i64 ..., 255` (no trunc + double zext).
    assert!(fn_ir.contains("and i64 ") && fn_ir.contains(", 255\n"), "{fn_ir}");
    // `movzx edx, byte ptr [rsi]` -> single `zext i8 ... to i64` (no `zext i8->i32` then `zext i32->i64`).
    assert!(
        fn_ir.lines().any(|l| l.contains(" = zext i8 ") && l.ends_with(" to i64")),
        "32-bit movzx from memory should extend directly to i64:\n{fn_ir}"
    );
    // `lea r8d, [rcx + rdx*4 + 16]` -> `and i64 ..., 4294967295`.
    assert!(fn_ir.contains(", 4294967295\n"), "{fn_ir}");
    // 32-bit shifts stay natively in `i32`.
    assert!(fn_ir.contains(" = shl i32 "), "{fn_ir}");
    assert!(fn_ir.contains(" = lshr i32 "), "{fn_ir}");
    assert!(fn_ir.contains(" = ashr i32 "), "{fn_ir}");
    // `cmp` -> `setl` + `cmovg` fused directly to `icmp slt i32` and `icmp sgt i32`.
    assert!(fn_ir.contains(" = icmp slt i32 "), "{fn_ir}");
    assert!(fn_ir.contains(" = icmp sgt i32 "), "{fn_ir}");
    // `dec rcx` -> `jle` fused directly to `icmp sle i64 ..., 1`.
    assert!(fn_ir.contains(" = icmp sle i64 ") && fn_ir.contains(", 1\n"), "{fn_ir}");
    // 32-bit `xor edx, edx; div` and `cdq; idiv` use native `udiv i32` / `sdiv i32` directly!
    assert!(fn_ir.contains(" = udiv i32 ") && fn_ir.contains(" = urem i32 "), "{fn_ir}");
    assert!(fn_ir.contains(" = sdiv i32 ") && fn_ir.contains(" = srem i32 "), "{fn_ir}");
    assert!(
        !scalar.contains("__playgta6_udiv128") && !scalar.contains("__playgta6_sdiv128"),
        "32-bit div/idiv must not pull in 128-bit helpers:\n{scalar}"
    );
}

#[test]
fn repeated_effective_address_math_in_a_block_is_computed_once() {
    let ir = module_ir(|a| {
        a.mov(eax, dword_ptr(rbx + rcx * 4 + 8))?;
        a.add(dword_ptr(rbx + rcx * 4 + 8), eax)?;
        a.ret()
    });
    assert_eq!(ir.matches("shl i64 ").count(), 1, "{ir}");
}

#[test]
fn r71_bit_ops_const_fast_paths_lzcnt_tzcnt_mulx_and_adcx_adox() {
    // 1. `bextr` and `bzhi` with known constant control/index emit compact shift/mask ops (no
    //    `select` bounds clamping from `shift_or`/`low_mask`).
    let bmi_const = module_ir(|a| {
        a.mov(ecx, 0x0804)?; // start=4, len=8
        a.bextr(eax, edx, ecx)?;
        a.mov(r8d, 12)?;
        a.bzhi(r9d, edx, r8d)?;
        a.xor(ecx, ecx)?; // kill flags
        a.ret()
    });
    assert!(
        !bmi_const.contains("select i1 "),
        "const bextr/bzhi must avoid dynamic shift_or/low_mask selects:\n{bmi_const}"
    );
    assert!(
        bmi_const.contains("lshr i32 ") && bmi_const.contains(", 4\n"),
        "const bextr should shift right by 4:\n{bmi_const}"
    );
    assert!(
        bmi_const.contains("and i32 ") && bmi_const.contains(", 255\n"),
        "const bextr should mask 8 bits:\n{bmi_const}"
    );
    assert!(
        bmi_const.contains("and i32 ") && bmi_const.contains(", 4095\n"),
        "const bzhi(12) should mask 12 bits:\n{bmi_const}"
    );

    // 2. `pext` / `pdep` with constant masks (0, all-ones, 1 run, 4 runs) inline without pulling
    //    in `@__playgta6_pext` or `@__playgta6_pdep`!
    let pext_fast = module_ir(|a| {
        a.mov(r8, 0x00FF_00FF_00FF_00FFu64)?; // 4 runs
        a.pext(rax, rcx, r8)?;
        a.pdep(rdx, rax, r8)?;
        a.mov(r9d, 0x00FF_0000u32)?; // 1 run
        a.pext(r10d, ecx, r9d)?;
        a.pdep(r11d, r10d, r9d)?;
        a.ret()
    });
    assert!(
        !pext_fast.contains("__playgta6_pext") && !pext_fast.contains("__playgta6_pdep"),
        "1..=4 run constant masks must bypass loop helpers:\n{pext_fast}"
    );

    // 3. `pext` with >4 runs (`0x5555_5555_5555_5555`) falls back to `@__playgta6_pext`.
    let pext_many = module_ir(|a| {
        a.mov(r8, 0x5555_5555_5555_5555u64)?;
        a.pext(rax, rcx, r8)?;
        a.ret()
    });
    assert!(
        pext_many.contains("@__playgta6_pext("),
        ">4 runs mask must keep helper call:\n{pext_many}"
    );

    // 4. `lzcnt`/`tzcnt` compute ZF from `src` directly (breaking dependency on ctlz/cttz).
    let cnt_ir = module_ir(|a| {
        a.lzcnt(rax, rcx)?;
        a.setz(dl)?;
        a.tzcnt(r8, r9)?;
        a.setz(r10b)?;
        a.xor(ecx, ecx)?;
        a.ret()
    });
    assert!(
        cnt_ir.contains("icmp slt i64 ") && cnt_ir.contains(", 0\n"),
        "lzcnt ZF should test msb(src) directly:\n{cnt_ir}"
    );
    assert!(
        cnt_ir.contains("trunc i64 ") && cnt_ir.contains(" to i1\n"),
        "tzcnt ZF should test lsb(src) directly:\n{cnt_ir}"
    );

    // 5. `mulx` with identical 32-bit destinations skips computing/storing the unused low half
    //    (only 2 `trunc i64` for `ecx` and `edx`; `lshr i64 %p, 32` already has zero upper 32
    //    bits so its `trunc`+`zext` round-trip folds away too).
    let mulx_ir = module_ir(|a| {
        a.mulx(r8d, r8d, ecx)?;
        a.ret()
    });
    assert_eq!(
        mulx_ir.matches("trunc i64 ").count(),
        2,
        "equal-dst 32-bit mulx should skip low-half and redundant high-half trunc:\n{mulx_ir}"
    );
    assert_eq!(
        mulx_ir.matches(", ptr %r8\n").count(),
        1,
        "equal-dst mulx must write %r8 only once (high half):\n{mulx_ir}"
    );

    // 6. `adcx`/`adox` chain: first limb folds constant CF=0 (no zext), trailing dead-flag limb
    //    elides carry calculation completely.
    let adcx_ir = module_ir(|a| {
        a.xor(eax, eax)?; // CF = 0, OF = 0
        a.adcx(rcx, rdx)?;
        a.adcx(r8, r9)?;
        a.xor(eax, eax)?; // kills trailing CF
        a.ret()
    });
    assert_eq!(
        adcx_ir.matches("icmp ult i64 ").count(),
        1,
        "only the first adcx needs carry-out; trailing dead CF must be elided:\n{adcx_ir}"
    );
    assert_eq!(
        adcx_ir.matches("zext i1 ").count(),
        1,
        "first adcx with known CF=0 must skip zext i1:\n{adcx_ir}"
    );
}

#[test]
fn wasm64_batch2_scalar_carry_fusion_div_mul128_and_crc32_ir() {
    // 1. Carry consumption fusion: `cmp; sbb eax, eax`, `neg; sbb`, `add; adc rdx, 0`,
    //    `bt; sbb`, `cmp; cmc; adc`, `cmp; adcx` — zero `%cf` alloca traffic in body!
    let carry_ir = module_ir(|a| {
        a.cmp(rcx, rdx)?;
        a.sbb(eax, eax)?;
        a.neg(r8)?;
        a.sbb(r8, r8)?;
        a.add(rax, rbx)?;
        a.adc(rdx, 0)?;
        a.bt(rsi, 3)?;
        a.sbb(r9d, r9d)?;
        a.cmp(rcx, rdx)?;
        a.cmc()?;
        a.adc(r10, 0)?;
        a.cmp(rcx, rdx)?;
        a.adcx(r11, rbx)?;
        a.xor(ecx, ecx)?; // kill trailing flags
        a.ret()
    });
    let body = carry_ir
        .split("bb_1000:\n")
        .nth(1)
        .expect("bb_1000")
        .split("store i64 ")
        .take(999)
        .collect::<Vec<_>>()
        .join("store i64 ");
    let before_ret = body.split("add i64 %").next().unwrap_or(&body);
    assert!(
        !before_ret.contains("ptr %cf"),
        "fused carry consumers must not touch %cf alloca:\n{before_ret}"
    );
    assert!(
        carry_ir.contains("sext i1 ") && carry_ir.contains(" to i32"),
        "sbb r, r should emit sext i1:\n{carry_ir}"
    );

    // 2. `mul128` constant-operand swap (`mov rax, MAGIC; mul rcx` puts constant in 2nd arg)
    //    and zero-fold (`xor eax, eax; mul rcx` emits zero umul128 calls!).
    let mul_swap_ir = module_ir(|a| {
        a.mov(rax, 0xAAAA_AAAA_AAAA_AAABu64)?;
        a.mul(rcx)?;
        a.ret()
    });
    assert!(
        mul_swap_ir.contains(", i64 -6148914691236517205)"),
        "constant RAX must be swapped into 2nd arg of @__playgta6_umul128:\n{mul_swap_ir}"
    );
    let mul_zero_ir = module_ir(|a| {
        a.xor(eax, eax)?;
        a.mul(rcx)?;
        a.ret()
    });
    assert!(
        !mul_zero_ir.contains("@__playgta6_umul128"),
        "zero-operand mul must fold to (0, 0) without umul128:\n{mul_zero_ir}"
    );

    // 3. Table-driven `@__playgta6_crc32c_u8` (256-entry lookup table, zero bit loop).
    let crc_ir = module_ir(|a| {
        a.crc32(rax, rcx)?;
        a.ret()
    });
    assert!(
        crc_ir.contains("@__playgta6_crc32c_tbl = internal unnamed_addr constant [256 x i32]"),
        "crc32c must use 256-entry lookup table:\n{crc_ir}"
    );
    assert!(
        !crc_ir.contains("br i1 %done"),
        "crc32c must not emit 8-iteration bit loop:\n{crc_ir}"
    );

    // 4. `merge_gpr` constant-`old` folding: `xor eax, eax; setcc al` skips `and`/`or`.
    let merge_ir = module_ir(|a| {
        a.xor(eax, eax)?;
        a.cmp(rcx, rdx)?;
        a.sete(al)?;
        a.xor(ecx, ecx)?;
        a.ret()
    });
    let merge_bb = merge_ir.split("bb_1000:\n").nth(1).expect("bb_1000");
    assert!(
        !merge_bb.contains("-256")
            && !merge_bb.contains(" = or i64 ")
            && merge_bb.contains("zext i8 ")
            && merge_bb.contains(" to i64"),
        "xor eax, eax; sete al must fold directly to zext i8->i64 without and/or in merge_gpr:\n{merge_bb}"
    );
}
