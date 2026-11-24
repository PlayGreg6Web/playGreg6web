//! Synthetic tests: small functions assembled with `code_asm` inside a synthetic image with an
//! import table.

use iced_x86::code_asm::*;
use iced_x86::{Code, Instruction, MemoryOperand, Register};

use super::cli::{Args, parse_args};
use super::report::{render_json, render_text};
use super::*;
use crate::code_image::CodeSection;
use crate::discover::discover;

const BASE: u64 = 0x1_4000_0000;
const TEXT: u64 = BASE + 0x1000;
const RDATA: u64 = BASE + 0x2000;
const DATA: u64 = BASE + 0x3000;
const GUARD: u64 = DATA + 0x100;
const CHECK: u64 = DATA + 0x108;
const G0: u64 = DATA;
const G1: u64 = DATA + 8;

type Asm = CodeAssembler;
type R = Result<(), iced_x86::IcedError>;

fn rip(va: u64) -> MemoryOperand {
    MemoryOperand::with_base_displ(Register::RIP, va as i64)
}

fn call_slot(a: &mut Asm, va: u64) -> R {
    a.add_instruction(Instruction::with1(Code::Call_rm64, rip(va))?)
}

fn load_g(a: &mut Asm, reg: Register, va: u64) -> R {
    a.add_instruction(Instruction::with2(Code::Mov_r64_rm64, reg, rip(va))?)
}

fn store_g(a: &mut Asm, va: u64, reg: Register) -> R {
    a.add_instruction(Instruction::with2(Code::Mov_rm64_r64, rip(va), reg)?)
}

fn lea_g(a: &mut Asm, reg: Register, va: u64) -> R {
    a.add_instruction(Instruction::with2(Code::Lea_r64_m, reg, rip(va))?)
}

#[derive(Default)]
struct Fx {
    text: Vec<u8>,
    entries: Vec<u64>,
    rdata: Vec<u8>,
    data: Vec<u8>,
    imports: Vec<ImportEntry>,
    next_const: usize,
}

impl Fx {
    fn new() -> Self {
        Self {
            rdata: vec![0; 0x400],
            data: vec![0; 0x400],
            next_const: 0x200,
            ..Self::default()
        }
    }

    fn import(&mut self, dll: &str, name: &str) -> u64 {
        let slot = RDATA + 8 * self.imports.len() as u64;
        self.imports.push(ImportEntry {
            dll: dll.to_string(),
            name: ImportRef::Name(name.to_string()),
            slot,
        });
        slot
    }

    fn bytes(&mut self, b: &[u8]) -> u64 {
        let va = RDATA + self.next_const as u64;
        self.rdata[self.next_const..self.next_const + b.len()].copy_from_slice(b);
        self.next_const = (self.next_const + b.len()).next_multiple_of(16);
        va
    }

    fn guid(&mut self, seed: u8) -> u64 {
        let mut b = [0u8; 16];
        for (i, x) in b.iter_mut().enumerate() {
            *x = seed.wrapping_add(i as u8 * 0x11);
        }
        self.bytes(&b)
    }

    fn func(&mut self, build: impl FnOnce(&mut Asm) -> R) -> u64 {
        while !self.text.len().is_multiple_of(0x20) {
            self.text.push(0xcc);
        }
        let va = TEXT + self.text.len() as u64;
        let mut a = Asm::new(64).unwrap();
        build(&mut a).expect("assemble");
        self.text.extend(a.assemble(va).expect("encode"));
        self.entries.push(va);
        va
    }

    fn run(&self, config: Config) -> Survey {
        let sections = vec![
            CodeSection::new(TEXT, &self.text, true),
            CodeSection::new(RDATA, &self.rdata, false),
            CodeSection::new(DATA, &self.data, false),
        ];
        let image = CodeImage::new(BASE, sections, self.entries.clone());
        let analysis = discover(&image);
        let env = Env {
            imports: self.imports.clone(),
            readonly: vec![(RDATA, RDATA + self.rdata.len() as u64)],
        };
        survey(&env, &image, &analysis, &config).expect("survey")
    }
}

fn src(spec: &str) -> Source {
    parse_source_spec(spec).expect("valid spec")
}

fn config(specs: &[&str]) -> Config {
    Config {
        sources: specs.iter().map(|s| src(s)).collect(),
        ..Config::default()
    }
}

fn key<'a>(s: &'a Survey, source: &str, index: usize) -> &'a KeyReport {
    &s.sources.iter().find(|x| x.name == source).unwrap().keys[index]
}

fn slot(k: &KeyReport, n: u32) -> Option<&SlotReport> {
    k.slots.iter().find(|s| s.slot == n)
}

fn slot_numbers(k: &KeyReport) -> Vec<u32> {
    k.slots.iter().map(|s| s.slot).collect()
}

/// `sub rsp,0x38; lea rdx,[rsp+0x30]; call [create]` -- out pointer in argument 1.
fn create_dev(a: &mut Asm, create: u64) -> R {
    a.sub(rsp, 0x38)?;
    a.lea(rdx, qword_ptr(rsp + 0x30))?;
    call_slot(a, create)
}

fn epilogue(a: &mut Asm) -> R {
    a.add(rsp, 0x38)?;
    a.ret()
}

const DEV: &str = "name=dev,import=dev.dll!Create,out=1";

#[test]
fn out_pointer_on_stack_then_direct_vtable_call() {
    let mut fx = Fx::new();
    let create = fx.import("DEV.dll", "Create");
    fx.func(|a| {
        create_dev(a, create)?;
        a.mov(rcx, qword_ptr(rsp + 0x30))?;
        a.mov(rax, qword_ptr(rcx))?;
        a.call(qword_ptr(rax + 0x48))?;
        epilogue(a)
    });
    let s = fx.run(config(&[DEV]));
    let k = key(&s, "dev", 0);
    assert_eq!((k.source_calls, k.out_stack, k.source_functions), (1, 1, 1));
    assert_eq!(slot_numbers(k), [9]);
    let sl = slot(k, 9).unwrap();
    assert_eq!((sl.sites, sl.unconfirmed_sites, sl.functions), (1, 0, 1));
    assert_eq!((sl.min_args, sl.max_args), (1, 1));
    assert_eq!(evidence_str(sl.evidence), "this_only");
    assert_eq!(sl.confidence, Confidence::High);
    assert_eq!(s.totals.sites, 1);
    assert!(s.unresolved.is_empty());
}

#[test]
fn pointer_spilled_and_reloaded_through_calls() {
    let mut fx = Fx::new();
    let create = fx.import("dev.dll", "Create");
    let other = fx.import("other.dll", "Other");
    fx.func(|a| {
        a.push(rbx)?;
        a.sub(rsp, 0x40)?;
        a.lea(rdx, qword_ptr(rsp + 0x30))?;
        call_slot(a, create)?;
        // Spill into another slot and a callee-saved register, then call something else.
        a.mov(rax, qword_ptr(rsp + 0x30))?;
        a.mov(qword_ptr(rsp + 0x38), rax)?;
        a.mov(rbx, rax)?;
        call_slot(a, other)?;
        // Reload from the stack.
        a.mov(rcx, qword_ptr(rsp + 0x38))?;
        a.mov(rax, qword_ptr(rcx))?;
        a.call(qword_ptr(rax + 0x10))?;
        // Callee-saved register.
        a.mov(rcx, rbx)?;
        a.mov(rax, qword_ptr(rcx))?;
        a.call(qword_ptr(rax + 0x18))?;
        a.add(rsp, 0x40)?;
        a.pop(rbx)?;
        a.ret()
    });
    let s = fx.run(config(&[DEV]));
    let k = key(&s, "dev", 0);
    assert_eq!(slot_numbers(k), [2, 3]);
    assert!(
        k.slots
            .iter()
            .all(|s| s.sites == 1 && s.confidence == Confidence::High)
    );
}

#[test]
fn pointer_in_register_survives_joins_and_is_lost_when_paths_disagree() {
    let mut fx = Fx::new();
    let create = fx.import("dev.dll", "Create");
    fx.func(|a| {
        create_dev(a, create)?;
        a.mov(rbx, qword_ptr(rsp + 0x30))?;
        let mut skip = a.create_label();
        a.test(esi, esi)?;
        a.jz(skip)?;
        a.nop()?;
        a.set_label(&mut skip)?;
        // Both paths still hold the pointer in rbx.
        a.mov(rcx, rbx)?;
        a.mov(rax, qword_ptr(rcx))?;
        a.call(qword_ptr(rax + 0x20))?;
        epilogue(a)
    });
    // One path overwrites rbx: the join forgets the pointer, nothing is guessed.
    fx.func(|a| {
        create_dev(a, create)?;
        a.mov(rbx, qword_ptr(rsp + 0x30))?;
        let mut skip = a.create_label();
        a.test(esi, esi)?;
        a.jz(skip)?;
        a.mov(rbx, rdi)?;
        a.set_label(&mut skip)?;
        a.mov(rcx, rbx)?;
        a.mov(rax, qword_ptr(rcx))?;
        a.call(qword_ptr(rax + 0x28))?;
        epilogue(a)
    });
    let s = fx.run(config(&[DEV]));
    let k = key(&s, "dev", 0);
    assert_eq!(slot_numbers(k), [4]);
}

#[test]
fn pointer_stored_to_global_in_one_function_and_used_in_another() {
    let mut fx = Fx::new();
    let create = fx.import("dev.dll", "Create");
    fx.func(|a| {
        create_dev(a, create)?;
        a.mov(rax, qword_ptr(rsp + 0x30))?;
        store_g(a, G0, Register::RAX)?;
        epilogue(a)
    });
    fx.func(|a| {
        load_g(a, Register::RCX, G0)?;
        a.mov(rax, qword_ptr(rcx))?;
        a.sub(rsp, 0x28)?;
        a.call(qword_ptr(rax + 0x20))?;
        a.add(rsp, 0x28)?;
        a.ret()
    });
    let s = fx.run(config(&[DEV]));
    let k = key(&s, "dev", 0);
    assert_eq!(k.globals, [G0]);
    let sl = slot(k, 4).expect("slot 4 via the global");
    assert_eq!(sl.sites, 1);
    assert_eq!(sl.confidence, Confidence::Medium);
}

#[test]
fn out_pointer_stored_directly_to_a_global_and_global_source() {
    let mut fx = Fx::new();
    let create = fx.import("dev.dll", "Create");
    fx.func(|a| {
        a.sub(rsp, 0x28)?;
        lea_g(a, Register::RDX, G0)?;
        call_slot(a, create)?;
        a.add(rsp, 0x28)?;
        a.ret()
    });
    fx.func(|a| {
        load_g(a, Register::RCX, G0)?;
        a.mov(rax, qword_ptr(rcx))?;
        a.sub(rsp, 0x28)?;
        a.call(qword_ptr(rax + 0x30))?;
        a.add(rsp, 0x28)?;
        a.ret()
    });
    // A second global the user declares holds an interface of its own.
    fx.func(|a| {
        load_g(a, Register::RCX, G1)?;
        a.mov(rax, qword_ptr(rcx))?;
        a.sub(rsp, 0x28)?;
        a.call(qword_ptr(rax + 0x38))?;
        a.add(rsp, 0x28)?;
        a.ret()
    });
    let spec_global = format!("name=gl,global={G1:#x}");
    let s = fx.run(config(&[DEV, &spec_global]));
    let k = key(&s, "dev", 0);
    assert_eq!((k.out_global, k.out_stack), (1, 0));
    assert_eq!(slot_numbers(k), [6]);
    let g = key(&s, "gl", 0);
    assert_eq!(slot_numbers(g), [7]);
    assert_eq!(g.source_calls, 0);
}

#[test]
fn control_flow_guard_dispatch_through_a_data_slot() {
    let mut fx = Fx::new();
    let create = fx.import("dev.dll", "Create");
    fx.func(|a| {
        create_dev(a, create)?;
        a.mov(rcx, qword_ptr(rsp + 0x30))?;
        a.mov(rax, qword_ptr(rcx))?;
        a.mov(rax, qword_ptr(rax + 0x38))?;
        call_slot(a, GUARD)?;
        epilogue(a)
    });
    let s = fx.run(config(&[DEV]));
    let k = key(&s, "dev", 0);
    assert_eq!(slot_numbers(k), [7]);
    assert_eq!(slot(k, 7).unwrap().confidence, Confidence::High);
    assert_eq!(
        s.guard_slots,
        [GUARD],
        "the guard slot is detected automatically"
    );
}

#[test]
fn control_flow_guard_check_then_call_through_register() {
    let mut fx = Fx::new();
    let create = fx.import("dev.dll", "Create");
    fx.func(|a| {
        a.push(rbx)?;
        a.sub(rsp, 0x30)?;
        a.lea(rdx, qword_ptr(rsp + 0x28))?;
        call_slot(a, create)?;
        a.mov(rcx, qword_ptr(rsp + 0x28))?;
        a.mov(rax, qword_ptr(rcx))?;
        a.mov(rbx, qword_ptr(rax + 0x40))?;
        a.mov(rcx, rbx)?;
        call_slot(a, CHECK)?; // check call: preserves every register
        a.mov(rcx, qword_ptr(rsp + 0x28))?;
        a.call(rbx)?;
        a.add(rsp, 0x30)?;
        a.pop(rbx)?;
        a.ret()
    });
    let s = fx.run(config(&[DEV]));
    let k = key(&s, "dev", 0);
    assert_eq!(slot_numbers(k), [8]);
    assert_eq!(
        slot(k, 8).unwrap().sites,
        1,
        "the check call is not a call site"
    );
    assert_eq!(s.guard_slots, [CHECK]);
}

#[test]
fn dispatch_thunk_function_and_forced_guard_slot() {
    let mut fx = Fx::new();
    let create = fx.import("dev.dll", "Create");
    // `jmp rax` thunk and a `jmp [guard]` thunk.
    let jmp_rax = fx.func(|a| a.jmp(rax));
    let jmp_slot = fx.func(|a| a.add_instruction(Instruction::with1(Code::Jmp_rm64, rip(GUARD))?));
    for thunk in [jmp_rax, jmp_slot] {
        fx.func(|a| {
            create_dev(a, create)?;
            a.mov(rcx, qword_ptr(rsp + 0x30))?;
            a.mov(rax, qword_ptr(rcx))?;
            a.mov(
                rax,
                qword_ptr(rax + if thunk == jmp_rax { 0x10 } else { 0x18 }),
            )?;
            a.call(thunk)?;
            epilogue(a)
        });
    }
    let mut cfg = config(&[DEV]);
    cfg.guard_slots.push(GUARD);
    let s = fx.run(cfg);
    let k = key(&s, "dev", 0);
    assert_eq!(slot_numbers(k), [2, 3]);
}

#[test]
fn wrapper_per_api_counts_its_callers() {
    let mut fx = Fx::new();
    let create = fx.import("dev.dll", "Create");
    let wrapper = fx.func(|a| {
        create_dev(a, create)?;
        a.mov(rcx, qword_ptr(rsp + 0x30))?;
        a.mov(rax, qword_ptr(rcx))?;
        a.call(qword_ptr(rax + 0x18))?;
        epilogue(a)
    });
    for _ in 0..3 {
        fx.func(|a| {
            a.sub(rsp, 0x28)?;
            a.call(wrapper)?;
            a.add(rsp, 0x28)?;
            a.ret()
        });
    }
    // A tail call counts too.
    fx.func(|a| a.jmp(wrapper));
    let s = fx.run(config(&[DEV]));
    let sl = slot(key(&s, "dev", 0), 3).unwrap();
    assert_eq!((sl.sites, sl.functions, sl.caller_sites), (1, 1, 4));
}

#[test]
fn interface_passed_to_a_small_wrapper_in_rcx() {
    let mut fx = Fx::new();
    let create = fx.import("dev.dll", "Create");
    let helper = fx.func(|a| {
        a.mov(rax, qword_ptr(rcx))?;
        a.sub(rsp, 0x28)?;
        a.call(qword_ptr(rax + 0x28))?;
        a.add(rsp, 0x28)?;
        a.ret()
    });
    fx.func(|a| {
        create_dev(a, create)?;
        a.mov(rcx, qword_ptr(rsp + 0x30))?;
        a.call(helper)?;
        epilogue(a)
    });
    let s = fx.run(config(&[DEV]));
    let k = key(&s, "dev", 0);
    let sl = slot(k, 5).expect("slot attributed through the wrapper");
    assert_eq!((sl.sites, sl.functions, sl.caller_sites), (1, 1, 1));
    assert_eq!(sl.confidence, Confidence::Medium);

    // With the depth limit at zero the pointer is reported as escaping, not guessed.
    let mut cfg = config(&[DEV]);
    cfg.max_depth = 0;
    let s = fx.run(cfg);
    let k = key(&s, "dev", 0);
    assert!(slot(k, 5).is_none());
    assert_eq!(k.unresolved.get("passed_beyond_limit"), Some(&1));
}

#[test]
fn chained_sources_follow_method_outputs() {
    let mut fx = Fx::new();
    let create = fx.import("dev.dll", "Create");
    fx.func(|a| {
        create_dev(a, create)?;
        a.mov(rcx, qword_ptr(rsp + 0x30))?;
        a.mov(rax, qword_ptr(rcx))?;
        a.lea(r8, qword_ptr(rsp + 0x28))?;
        a.call(qword_ptr(rax + 0x40))?; // slot 8 creates the queue (out arg 2)
        a.mov(rcx, qword_ptr(rsp + 0x28))?;
        a.mov(rax, qword_ptr(rcx))?;
        a.call(qword_ptr(rax + 0x50))?; // queue slot 10
        epilogue(a)
    });
    let s = fx.run(config(&[DEV, "name=queue,method=dev:8,out=2"]));
    let dev = key(&s, "dev", 0);
    assert_eq!(slot_numbers(dev), [8]);
    let sl = slot(dev, 8).unwrap();
    assert_eq!(
        (sl.min_args, evidence_str(sl.evidence)),
        (3, "regs".to_string())
    );
    let queue = key(&s, "queue", 0);
    assert_eq!((queue.source_calls, queue.out_stack), (1, 1));
    assert_eq!(slot_numbers(queue), [10]);
    assert_eq!(s.totals.keys, 2);
}

#[test]
fn method_source_with_global_out_and_return_value() {
    let mut fx = Fx::new();
    let create = fx.import("dev.dll", "Create");
    fx.func(|a| {
        create_dev(a, create)?;
        a.mov(rcx, qword_ptr(rsp + 0x30))?;
        a.mov(rax, qword_ptr(rcx))?;
        a.call(qword_ptr(rax + 0x58))?; // slot 11 returns the child in rax
        a.mov(rcx, rax)?;
        a.mov(rax, qword_ptr(rcx))?;
        a.call(qword_ptr(rax + 0x60))?;
        epilogue(a)
    });
    let s = fx.run(config(&[DEV, "name=child,method=dev:11,out=ret"]));
    assert_eq!(slot_numbers(key(&s, "child", 0)), [12]);
    assert_eq!(key(&s, "child", 0).out_ret, 1);
}

#[test]
fn unprovable_this_is_reported_as_unconfirmed() {
    let mut fx = Fx::new();
    let create = fx.import("dev.dll", "Create");
    fx.func(|a| {
        a.push(rbx)?;
        a.sub(rsp, 0x30)?;
        a.lea(rdx, qword_ptr(rsp + 0x28))?;
        call_slot(a, create)?;
        a.mov(rax, qword_ptr(rsp + 0x28))?;
        a.mov(rax, qword_ptr(rax))?; // vtable of the interface ...
        a.mov(rcx, rdi)?; // ... but `this` is something else
        a.call(qword_ptr(rax + 0x18))?;
        a.add(rsp, 0x30)?;
        a.pop(rbx)?;
        a.ret()
    });
    let s = fx.run(config(&[DEV]));
    let sl = slot(key(&s, "dev", 0), 3).unwrap();
    assert_eq!((sl.sites, sl.unconfirmed_sites), (0, 1));
    assert_eq!(sl.confidence, Confidence::Low);
    assert_eq!(sl.unconfirmed_examples.len(), 1);
    assert_eq!(s.totals.sites, 0);
    assert_eq!(s.totals.unconfirmed_sites, 1);
}

#[test]
fn argument_counts_are_lower_bounds_with_evidence() {
    let mut fx = Fx::new();
    let create = fx.import("dev.dll", "Create");
    fx.func(|a| {
        a.push(rbx)?;
        a.sub(rsp, 0x50)?;
        a.lea(rdx, qword_ptr(rsp + 0x48))?;
        call_slot(a, create)?;
        a.mov(rbx, qword_ptr(rsp + 0x48))?;
        let begin = |a: &mut Asm| -> R {
            a.mov(rcx, rbx)?;
            a.mov(rax, qword_ptr(rcx))
        };
        // slot 2: no argument
        begin(a)?;
        a.call(qword_ptr(rax + 0x10))?;
        // slot 3: one argument
        begin(a)?;
        a.mov(edx, 1)?;
        a.call(qword_ptr(rax + 0x18))?;
        // slot 4: three arguments
        begin(a)?;
        a.mov(edx, 1)?;
        a.mov(r8d, 2)?;
        a.mov(r9d, 3)?;
        a.call(qword_ptr(rax + 0x20))?;
        // slot 5: five arguments, two of them on the stack
        begin(a)?;
        a.mov(edx, 1)?;
        a.mov(r8d, 2)?;
        a.mov(r9d, 3)?;
        a.mov(qword_ptr(rsp + 0x20), 4)?;
        a.mov(qword_ptr(rsp + 0x28), 5)?;
        a.call(qword_ptr(rax + 0x28))?;
        // slot 6: a float in xmm2
        begin(a)?;
        a.xorps(xmm2, xmm2)?;
        a.call(qword_ptr(rax + 0x30))?;
        // slot 7: nothing again; the earlier arguments belong to earlier calls
        begin(a)?;
        a.call(qword_ptr(rax + 0x38))?;
        a.add(rsp, 0x50)?;
        a.pop(rbx)?;
        a.ret()
    });
    let s = fx.run(config(&[DEV]));
    let k = key(&s, "dev", 0);
    let args = |n: u32| {
        let sl = slot(k, n).unwrap();
        assert_eq!(sl.min_args, sl.max_args);
        (sl.min_args, evidence_str(sl.evidence))
    };
    assert_eq!(args(2), (1, "this_only".to_string()));
    assert_eq!(args(3), (2, "regs".to_string()));
    assert_eq!(args(4), (4, "regs".to_string()));
    assert_eq!(args(5), (6, "regs+stack".to_string()));
    assert_eq!(args(6), (3, "xmm".to_string()));
    assert_eq!(args(7), (1, "this_only".to_string()));
}

#[test]
fn argument_count_takes_min_and_max_over_sites() {
    let mut fx = Fx::new();
    let create = fx.import("dev.dll", "Create");
    for extra in [false, true] {
        fx.func(|a| {
            create_dev(a, create)?;
            a.mov(rcx, qword_ptr(rsp + 0x30))?;
            a.mov(rax, qword_ptr(rcx))?;
            if extra {
                a.mov(edx, 1)?;
                a.mov(r8d, 1)?;
            }
            a.call(qword_ptr(rax + 0x18))?;
            epilogue(a)
        });
    }
    let s = fx.run(config(&[DEV]));
    let sl = slot(key(&s, "dev", 0), 3).unwrap();
    assert_eq!(
        (sl.sites, sl.functions, sl.min_args, sl.max_args),
        (2, 2, 1, 3)
    );
}

fn proc_fixture() -> (Fx, u64, [u64; 3]) {
    let mut fx = Fx::new();
    let getproc = fx.import("kernel32.dll", "GetProcAddress");
    let name = fx.bytes(b"QueryApiImpl\0");
    let other_name = fx.bytes(b"SomethingElse\0");
    let guids = [fx.guid(0x10), fx.guid(0x30), fx.guid(0x50)];
    fx.func(|a| {
        a.sub(rsp, 0x28)?;
        a.xor(ecx, ecx)?;
        lea_g(a, Register::RDX, name)?;
        call_slot(a, getproc)?;
        store_g(a, G0, Register::RAX)?;
        // A different export goes to another variable and must not become a source.
        a.xor(ecx, ecx)?;
        lea_g(a, Register::RDX, other_name)?;
        call_slot(a, getproc)?;
        store_g(a, G1, Register::RAX)?;
        a.add(rsp, 0x28)?;
        a.ret()
    });
    (fx, getproc, guids)
}

const PROC: &str = "name=q,proc=QueryApiImpl,out=2,key=0+1";

#[test]
fn getprocaddress_function_pointer_source_keyed_by_guid() {
    let (mut fx, _, guids) = proc_fixture();
    // `call [var]` with guid 0, `mov rax,[var]; call rax` with guid 1.
    fx.func(|a| {
        a.sub(rsp, 0x48)?;
        a.lea(r8, qword_ptr(rsp + 0x40))?;
        lea_g(a, Register::RCX, guids[0])?;
        a.mov(rdx, rcx)?;
        call_slot(a, G0)?;
        a.mov(rcx, qword_ptr(rsp + 0x40))?;
        a.mov(rax, qword_ptr(rcx))?;
        a.call(qword_ptr(rax + 0x18))?;
        a.add(rsp, 0x48)?;
        a.ret()
    });
    fx.func(|a| {
        a.sub(rsp, 0x48)?;
        a.lea(r8, qword_ptr(rsp + 0x40))?;
        lea_g(a, Register::RCX, guids[1])?;
        a.mov(rdx, rcx)?;
        load_g(a, Register::RAX, G0)?;
        a.call(rax)?;
        a.mov(rcx, qword_ptr(rsp + 0x40))?;
        a.mov(rax, qword_ptr(rcx))?;
        a.call(qword_ptr(rax + 0x20))?;
        a.add(rsp, 0x48)?;
        a.ret()
    });
    // Calls through the other export's variable are not a source.
    fx.func(|a| {
        a.sub(rsp, 0x48)?;
        a.lea(r8, qword_ptr(rsp + 0x40))?;
        lea_g(a, Register::RCX, guids[2])?;
        a.mov(rdx, rcx)?;
        call_slot(a, G1)?;
        a.mov(rcx, qword_ptr(rsp + 0x40))?;
        a.mov(rax, qword_ptr(rcx))?;
        a.call(qword_ptr(rax + 0x28))?;
        a.add(rsp, 0x48)?;
        a.ret()
    });
    let s = fx.run(config(&[PROC]));
    let keys = &s.sources[0].keys;
    assert_eq!(keys.len(), 2, "two GUIDs, two interface keys");
    let text = |k: &KeyReport| {
        k.key_args
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" ")
    };
    // Bytes 10 21 32 43 54 65 76 87 98 a9 ba cb dc ed fe 0f and 30 41 52 63 74 85 96 a7 ...
    let (g0, g1) = (
        "{43322110-6554-8776-98a9-bacbdcedfe0f}",
        "{63524130-8574-a796-b8c9-daebfc0d1e2f}",
    );
    assert_eq!(text(&keys[0]), format!("{g0} {g0}"));
    assert_eq!(text(&keys[1]), format!("{g1} {g1}"));
    assert_eq!(slot_numbers(&keys[0]), [3]);
    assert_eq!(slot_numbers(&keys[1]), [4]);
    assert_eq!(s.totals.keys, 2);
}

#[test]
fn function_forwarding_its_arguments_to_the_function_pointer_is_a_source() {
    let (mut fx, _, guids) = proc_fixture();
    let forwarder = fx.func(|a| {
        a.sub(rsp, 0x28)?;
        load_g(a, Register::RAX, G0)?;
        a.call(rax)?;
        a.add(rsp, 0x28)?;
        a.ret()
    });
    for (g, disp) in [(guids[0], 0x18), (guids[0], 0x20), (guids[1], 0x18)] {
        fx.func(|a| {
            a.sub(rsp, 0x48)?;
            a.lea(r8, qword_ptr(rsp + 0x40))?;
            lea_g(a, Register::RCX, g)?;
            a.mov(rdx, rcx)?;
            a.call(forwarder)?;
            a.mov(rcx, qword_ptr(rsp + 0x40))?;
            a.mov(rax, qword_ptr(rcx))?;
            a.call(qword_ptr(rax + disp))?;
            a.add(rsp, 0x48)?;
            a.ret()
        });
    }
    let s = fx.run(config(&[PROC]));
    let keys = &s.sources[0].keys;
    assert_eq!(keys.len(), 2);
    assert_eq!(
        (keys[0].source_calls, slot_numbers(&keys[0])),
        (2, vec![3, 4])
    );
    assert_eq!((keys[1].source_calls, slot_numbers(&keys[1])), (1, vec![3]));
}

#[test]
fn guid_arguments_outside_read_only_sections_stay_unknown() {
    let mut fx = Fx::new();
    let create = fx.import("dev.dll", "Create");
    fx.func(|a| {
        a.sub(rsp, 0x38)?;
        lea_g(a, Register::RCX, G1)?; // writable data, not a GUID constant
        a.xor(edx, edx)?;
        a.lea(r8, qword_ptr(rsp + 0x30))?;
        call_slot(a, create)?;
        a.add(rsp, 0x38)?;
        a.ret()
    });
    let s = fx.run(config(&["name=dev,import=dev.dll!Create,out=2,key=0+1"]));
    assert_eq!(
        key(&s, "dev", 0).key_args,
        [ArgKey::Unknown, ArgKey::Const(0)]
    );
}

#[test]
fn unrelated_vtable_calls_are_not_reported() {
    let mut fx = Fx::new();
    let create = fx.import("dev.dll", "Create");
    let other = fx.import("other.dll", "GetThing");
    fx.func(|a| {
        a.mov(rax, qword_ptr(rdi))?;
        a.sub(rsp, 0x28)?;
        a.call(qword_ptr(rax + 0x48))?;
        a.add(rsp, 0x28)?;
        a.ret()
    });
    fx.func(|a| {
        create_dev(a, create)?;
        // An unrelated object next to the interface.
        a.mov(rcx, rbx)?;
        a.mov(rax, qword_ptr(rcx))?;
        a.call(qword_ptr(rax + 0x48))?;
        // Another interface from a different import.
        a.lea(rdx, qword_ptr(rsp + 0x28))?;
        call_slot(a, other)?;
        a.mov(rcx, qword_ptr(rsp + 0x28))?;
        a.mov(rax, qword_ptr(rcx))?;
        a.call(qword_ptr(rax + 0x50))?;
        // A computed slot is unresolved, not a slot.
        a.mov(rcx, qword_ptr(rsp + 0x30))?;
        a.mov(rax, qword_ptr(rcx))?;
        a.call(qword_ptr(rax + rsi * 8))?;
        epilogue(a)
    });
    let s = fx.run(config(&[DEV]));
    let k = key(&s, "dev", 0);
    assert!(k.slots.is_empty(), "{:?}", k.slots);
    assert_eq!(s.totals.sites, 0);
    assert_eq!(k.unresolved.get("computed_slot"), Some(&1));
}

#[test]
fn escapes_are_counted_by_reason() {
    let mut fx = Fx::new();
    let create = fx.import("dev.dll", "Create");
    let other = fx.import("other.dll", "Take");
    fx.func(|a| {
        create_dev(a, create)?;
        a.mov(rcx, qword_ptr(rsp + 0x30))?;
        call_slot(a, other)?; // import taking the interface
        a.mov(rcx, qword_ptr(rsp + 0x30))?;
        a.mov(qword_ptr(rdi + 0x10), rcx)?; // stored into a field of an incoming object
        a.mov(rdx, qword_ptr(rdi + 0x20))?;
        a.mov(qword_ptr(rdx), rcx)?; // stored through a pointer of unknown provenance
        a.call(rbx)?; // unknown indirect callee
        a.mov(rax, qword_ptr(rsp + 0x30))?; // returned
        epilogue(a)
    });
    // The out pointer is a register: untracked.
    fx.func(|a| {
        a.sub(rsp, 0x28)?;
        a.mov(rdx, rdi)?;
        call_slot(a, create)?;
        a.add(rsp, 0x28)?;
        a.ret()
    });
    let s = fx.run(config(&[DEV]));
    let k = key(&s, "dev", 0);
    for reason in [
        "passed_to_import",
        "stored_to_field",
        "stored_to_unknown",
        "passed_to_indirect_call",
        "returned_to_caller",
    ] {
        assert_eq!(
            k.unresolved.get(reason),
            Some(&1),
            "{reason}: {:?}",
            k.unresolved
        );
    }
    assert_eq!(k.unresolved.get("out_param_untracked"), Some(&1));
    assert_eq!(s.unresolved.get("out_param_untracked"), Some(&1));
}

#[test]
fn object_fields_are_followed_inside_a_function_only_until_a_call() {
    let mut fx = Fx::new();
    let create = fx.import("dev.dll", "Create");
    let other = fx.import("other.dll", "Other");
    fx.func(|a| {
        create_dev(a, create)?;
        a.mov(rax, qword_ptr(rsp + 0x30))?;
        a.mov(qword_ptr(rdi + 0x10), rax)?;
        a.mov(rcx, qword_ptr(rdi + 0x10))?; // reload: same object, same field
        a.mov(rax, qword_ptr(rcx))?;
        a.call(qword_ptr(rax + 0x10))?;
        a.mov(rax, qword_ptr(rsp + 0x30))?;
        a.mov(qword_ptr(rdi + 0x10), rax)?;
        call_slot(a, other)?; // a call may change the field
        a.mov(rcx, qword_ptr(rdi + 0x10))?;
        a.mov(rax, qword_ptr(rcx))?;
        a.call(qword_ptr(rax + 0x18))?;
        epilogue(a)
    });
    let s = fx.run(config(&[DEV]));
    assert_eq!(slot_numbers(key(&s, "dev", 0)), [2]);
}

#[test]
fn results_are_deterministic_and_the_json_has_the_documented_shape() {
    let build = || {
        let (mut fx, _, guids) = proc_fixture();
        let create = fx.import("dev.dll", "Create");
        for (g, disp) in [(guids[0], 0x18), (guids[1], 0x20)] {
            fx.func(|a| {
                a.sub(rsp, 0x48)?;
                a.lea(r8, qword_ptr(rsp + 0x40))?;
                lea_g(a, Register::RCX, g)?;
                a.mov(rdx, rcx)?;
                call_slot(a, G0)?;
                a.mov(rcx, qword_ptr(rsp + 0x40))?;
                a.mov(rax, qword_ptr(rcx))?;
                a.call(qword_ptr(rax + disp))?;
                a.add(rsp, 0x48)?;
                a.ret()
            });
        }
        fx.func(|a| {
            create_dev(a, create)?;
            a.mov(rcx, qword_ptr(rsp + 0x30))?;
            a.mov(rax, qword_ptr(rcx))?;
            a.call(qword_ptr(rax + 0x48))?;
            epilogue(a)
        });
        fx.run(config(&[PROC, DEV]))
    };
    let (first, second) = (build(), build());
    assert_eq!(first, second);
    assert_eq!(render_text(&first), render_text(&second));
    let json = render_json(&first);
    assert_eq!(json, render_json(&second));
    assert!(json.starts_with("{\n  \"stats\": {"));
    assert!(json.ends_with("}\n"));
    for needle in [
        "\"guard_slots\": [",
        "\"totals\": {\"keys\": 3, \"source_calls\": 3, ",
        "\"unresolved\": {}",
        "\"sources\": [",
        "{\"name\": \"q\", \"description\": \"GetProcAddress \\\"QueryApiImpl\\\", out arg 2, key args 0+1\"",
        "\"key\": [\"{43322110-6554-8776-98a9-bacbdcedfe0f}\", ",
        "{\"slot\": 9, \"sites\": 1, \"unconfirmed_sites\": 0, \"functions\": 1, \"caller_sites\": 0, ",
        "\"min_args\": 1, \"max_args\": 1, \"evidence\": \"this_only\", \"confidence\": \"high\", ",
        "\"examples\": [\"0x",
    ] {
        assert!(json.contains(needle), "missing {needle:?} in\n{json}");
    }
    // Balanced outside of strings and no code bytes: only structure, hex addresses and counts.
    let (mut depth, mut in_str, mut esc) = (0i32, false, false);
    for c in json.chars() {
        if in_str {
            match (esc, c) {
                (true, _) => esc = false,
                (false, '\\') => esc = true,
                (false, '"') => in_str = false,
                _ => {}
            }
        } else {
            match c {
                '"' => in_str = true,
                '{' | '[' => depth += 1,
                '}' | ']' => depth -= 1,
                _ => {}
            }
            assert!(depth >= 0);
        }
    }
    assert_eq!((depth, in_str), (0, false));
    let text = render_text(&first);
    assert!(text.contains("source dev (import dev.dll!Create, out arg 1)"));
    assert!(text.contains("totals: 3 keys, 3 source calls, 3 slots, 3 confirmed sites"));
}

#[test]
fn source_errors_are_reported() {
    let mut fx = Fx::new();
    fx.import("dev.dll", "Create");
    let try_run = |specs: &[&str]| {
        let sections = vec![CodeSection::new(TEXT, &fx.text, true)];
        let image = CodeImage::new(BASE, sections, vec![]);
        let analysis = discover(&image);
        let env = Env {
            imports: fx.imports.clone(),
            readonly: vec![],
        };
        survey(&env, &image, &analysis, &config(specs)).err()
    };
    assert_eq!(try_run(&[]), Some(SurveyError::NoSources));
    assert_eq!(
        try_run(&["name=a,import=nope.dll!X,out=0"]),
        Some(SurveyError::ImportNotFound("nope.dll!X".to_string()))
    );
    assert_eq!(
        try_run(&[DEV, "name=dev,fn=0x10,out=0"]),
        Some(SurveyError::DuplicateSource("dev".to_string()))
    );
    assert!(matches!(
        try_run(&["name=m,method=later:3,out=0", "name=later,fn=0x10,out=0"]),
        Some(SurveyError::UnknownParent { .. })
    ));
    assert_eq!(
        try_run(&["name=p,proc=Foo,out=0"]),
        Some(SurveyError::NoGetProcAddress)
    );
    assert_eq!(try_run(&[DEV]), None);
}

#[test]
fn function_sources_by_entry_address() {
    let mut fx = Fx::new();
    let make = fx.func(|a| a.ret());
    fx.func(|a| {
        a.sub(rsp, 0x28)?;
        a.call(make)?;
        a.mov(rcx, rax)?;
        a.mov(rax, qword_ptr(rcx))?;
        a.call(qword_ptr(rax + 0x20))?;
        a.add(rsp, 0x28)?;
        a.ret()
    });
    let s = fx.run(config(&[&format!("name=mk,fn={make:#x},out=ret")]));
    assert_eq!(slot_numbers(key(&s, "mk", 0)), [4]);
}

#[test]
fn env_from_pe_matches_the_import_directory() {
    use crate::pe::test_util::build_synthetic_pe64;
    let file = build_synthetic_pe64(false);
    let pe = Pe64::parse(&file).unwrap();
    let env = Env::from_pe(&pe);
    let base = pe.optional.image_base;
    let k32 = &pe.imports[0];
    assert_eq!(
        env.find_import("kernel32", &ImportRef::Name("VirtualAlloc".to_string())),
        Some(base + u64::from(k32.iat_rva))
    );
    assert_eq!(
        env.find_import("KERNEL32.DLL", &ImportRef::Ordinal(42)),
        Some(base + u64::from(k32.iat_rva) + 8)
    );
    assert_eq!(
        env.find_import("kernel32.dll", &ImportRef::Name("Nope".into())),
        None
    );
    // Delay-load imports are slots too.
    let delay_slots: usize = pe
        .delay_imports
        .iter()
        .filter(|m| m.iat_rva != 0)
        .map(|m| m.symbols.len())
        .sum();
    let k32_slots: usize = pe.imports.iter().map(|m| m.symbols.len()).sum();
    assert_eq!(env.imports.len(), k32_slots + delay_slots);
    // `.rdata` is readable but not writable or executable.
    assert!(!env.readonly.is_empty());
    assert!(env.readonly.iter().all(|&(lo, hi)| lo >= base && hi > lo));
}

fn tmp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("playgta6-survey-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn end_to_end_run_on_a_pe_file() {
    use crate::pe::test_util::build_synthetic_pe64;
    let dir = tmp_dir("e2e");
    let pe = dir.join("synthetic.exe");
    std::fs::write(&pe, build_synthetic_pe64(false)).unwrap();
    let out = dir.join("out").join("survey.json");
    let argv: Vec<String> = [
        pe.to_str().unwrap(),
        "--source",
        "name=va,import=kernel32.dll!VirtualAlloc,out=ret",
        "--json",
        "--out",
        out.to_str().unwrap(),
        "--no-refine",
    ]
    .iter()
    .map(ToString::to_string)
    .collect();
    let args = parse_args(&argv).unwrap().unwrap();
    let msg = super::cli::run(&args).unwrap();
    assert!(msg.starts_with("Wrote interface survey to "));
    let json = std::fs::read_to_string(&out).unwrap();
    assert!(json.contains("\"name\": \"va\""));
    assert!(json.contains("\"keys\": []"));
    // The refined discovery path works too and prints the table to stdout.
    let mut refined = args.clone();
    refined.no_refine = false;
    refined.json = false;
    refined.out = None;
    let text = super::cli::run(&refined).unwrap();
    assert!(text.contains("source va (import kernel32.dll!VirtualAlloc, out ret)"));
    let _ = std::fs::remove_dir_all(&dir);
}

fn strings(v: &[&str]) -> Vec<String> {
    v.iter().map(ToString::to_string).collect()
}

#[test]
fn cli_parsing() {
    let args = parse_args(&strings(&[
        "game.exe",
        "--source",
        "name=dev,import=d3d12_xs.dll!D3D12XboxCreateDevice,out=3,key=2",
        "--source",
        "name=q,method=dev:8,out=3,key=2",
        "--guard-slot",
        "0x1000",
        "--max-depth",
        "5",
        "--examples",
        "2",
        "--json",
    ]))
    .unwrap()
    .unwrap();
    assert_eq!(args.pe_path, std::path::PathBuf::from("game.exe"));
    assert_eq!(args.sources.len(), 2);
    assert_eq!(args.guard_slots, [0x1000]);
    assert_eq!((args.max_depth, args.examples), (5, 2));
    assert!(args.json && !args.no_refine && args.out.is_none());
    assert_eq!(
        args.sources[0],
        Source {
            name: "dev".to_string(),
            kind: SourceKind::Import {
                dll: "d3d12_xs.dll".to_string(),
                name: ImportRef::Name("D3D12XboxCreateDevice".to_string())
            },
            out: Some(OutParam::Arg(3)),
            key_args: vec![2],
        }
    );
    assert_eq!(
        args.sources[1].kind,
        SourceKind::Method {
            parent: "dev".to_string(),
            slot: 8
        }
    );
    assert_eq!(parse_args(&strings(&["--help"])).unwrap(), None::<Args>);

    let err = |v: &[&str]| parse_args(&strings(v)).unwrap_err();
    assert!(err(&["a.exe"]).contains("--source"));
    assert!(err(&["--source", "name=a,fn=1,out=0"]).contains("missing input PE path"));
    assert!(err(&["a.exe", "b.exe", "--source", "name=a,fn=1,out=0"]).contains("extra positional"));
    assert!(err(&["a.exe", "--bogus"]).contains("unknown option"));
    assert!(err(&["a.exe", "--source"]).contains("requires a value"));
    assert!(err(&["a.exe", "--max-depth", "x"]).contains("--max-depth"));
    assert!(err(&["a.exe", "--guard-slot", "zz"]).contains("hexadecimal"));
}

#[test]
fn source_spec_parsing() {
    let ord = src("name=o,import=ws2_32.dll!#115,out=ret");
    assert_eq!(
        ord.kind,
        SourceKind::Import {
            dll: "ws2_32.dll".to_string(),
            name: ImportRef::Ordinal(115)
        }
    );
    assert_eq!(ord.out, Some(OutParam::Ret));
    assert_eq!(
        src("name=f,fn=0x140001000,out=0").kind,
        SourceKind::Function {
            entry: 0x1_4000_1000
        }
    );
    let g = src("name=g,global=140002000");
    assert_eq!(
        (g.kind, g.out),
        (SourceKind::Global { va: 0x1_4000_2000 }, None)
    );
    assert_eq!(
        src("name=p,proc=QueryApiImpl,out=2,key=0+1").key_args,
        [0, 1]
    );
    assert_eq!(
        src("name=d,import=a.dll!B,out=3,key=2").describe(),
        "import a.dll!B, out arg 3, key args 2"
    );

    let bad = |s: &str| parse_source_spec(s).unwrap_err();
    assert!(bad("import=a.dll!B,out=1").contains("name="));
    assert!(bad("name=a,out=1").contains("needs one of"));
    assert!(bad("name=a,import=a.dll!B").contains("out="));
    assert!(bad("name=a,global=10,out=1").contains("no out="));
    assert!(bad("name=a,fn=1,proc=X,out=0").contains("several"));
    assert!(bad("name=a,import=nobang,out=0").contains("<dll>!<name>"));
    assert!(bad("name=a,fn=1,out=16").contains("too large"));
    assert!(bad("name=a,fn=1,out=0,key=1+99").contains("too large"));
    assert!(bad("name=a,method=x:9999,out=0").contains("out of range"));
    assert!(bad("name=a,method=x,out=0").contains("<source>:<slot>"));
    assert!(bad("name=a b,fn=1,out=0").contains("must match"));
    assert!(bad("name=a,fn=1,out=0,color=red").contains("unknown source option"));
    assert!(bad("name=a,fn=zz,out=0").contains("hexadecimal"));
    assert!(bad("nonsense").contains("key=value"));
}
