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
    assert!(ir.contains("define void @sub_1000(ptr noalias %state) #0 {"));
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
            ir.contains(&format!("declare void @{helper}(ptr)")),
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
