//! Pipeline tests on synthetic PE32+ images written in-test (sections, imports, `.pdata`, base
//! relocations); all outputs go to temp dirs. Tests that compile, link or execute need `llc` or
//! `clang`, a wasm linker and `node`, and skip with a printed note when one is missing.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use super::refine::{PointerSource, PointerStatus};
use super::*;
use crate::cfg::Terminator;
use crate::discover::discover;
use crate::lift::compile::{compile_to_wasm64_object, find_on_path, rustc_tool_dir};

const BASE: u64 = 0x1_4000_0000;
const RDATA: u32 = 0x1000;
const DATA: u32 = 0x2000;
const TEXT: u32 = 0x3000;
const SECTION_SPAN: u32 = 0x1000;
/// Shared minimal `UNWIND_INFO` (version 1, `sub rsp, 0x28`) for every `.pdata` record.
const UNWIND_RVA: u32 = RDATA + 0xF00;
const UNWIND: [u8; 8] = [0x01, 0x04, 0x01, 0x00, 0x04, 0x42, 0x00, 0x00];

const T: u64 = BASE + TEXT as u64;
const D: u64 = BASE + DATA as u64;

// ----- synthetic PE writer -----------------------------------------------------------------------

fn put(buf: &mut [u8], off: usize, bytes: &[u8]) {
    buf[off..off + bytes.len()].copy_from_slice(bytes);
}

/// `.rdata` bytes holding an import directory for `modules` at offset 0, plus the IAT slot RVAs
/// per module. The shared unwind info lives at [`UNWIND_RVA`].
fn layout_rdata(modules: &[(&str, &[&str])]) -> (Vec<u8>, Vec<Vec<u32>>) {
    let mut buf = vec![0u8; SECTION_SPAN as usize];
    let mut cur = ((modules.len() + 1) * 20).next_multiple_of(8);
    let mut slots = Vec::new();
    for (m, (dll, names)) in modules.iter().enumerate() {
        let ilt = cur;
        let iat = ilt + 8 * (names.len() + 1);
        cur = iat + 8 * (names.len() + 1);
        let name = cur;
        put(&mut buf, name, dll.as_bytes());
        cur = (cur + dll.len() + 1).next_multiple_of(2);
        let mut module_slots = Vec::new();
        for (i, symbol) in names.iter().enumerate() {
            let hint_name = RDATA + cur as u32;
            put(&mut buf, cur + 2, symbol.as_bytes());
            cur = (cur + 2 + symbol.len() + 1).next_multiple_of(2);
            put(&mut buf, ilt + 8 * i, &u64::from(hint_name).to_le_bytes());
            put(&mut buf, iat + 8 * i, &u64::from(hint_name).to_le_bytes());
            module_slots.push(RDATA + (iat + 8 * i) as u32);
        }
        let desc = 20 * m;
        put(&mut buf, desc, &(RDATA + ilt as u32).to_le_bytes());
        put(&mut buf, desc + 12, &(RDATA + name as u32).to_le_bytes());
        put(&mut buf, desc + 16, &(RDATA + iat as u32).to_le_bytes());
        slots.push(module_slots);
        cur = cur.next_multiple_of(8);
    }
    assert!(
        cur <= (UNWIND_RVA - RDATA) as usize,
        "import directory too large"
    );
    put(&mut buf, (UNWIND_RVA - RDATA) as usize, &UNWIND);
    (buf, slots)
}

/// A synthetic PE32+: `.rdata` (imports) at 0x1000, `.data` at 0x2000, `.text` at 0x3000, then
/// optional `.pdata` and `.reloc`.
#[derive(Default)]
struct TestPe<'a> {
    imports: Vec<(&'a str, &'a [&'a str])>,
    data: Vec<u8>,
    text: Vec<u8>,
    /// Entry point RVA (defaults to the start of `.text`).
    entry: Option<u32>,
    /// `.pdata` records `(begin, end)` as VAs.
    pdata: Vec<(u64, u64)>,
    /// `DIR64` relocation sites as VAs.
    relocs: Vec<u64>,
}

impl TestPe<'_> {
    fn slots(&self) -> Vec<Vec<u64>> {
        layout_rdata(&self.imports)
            .1
            .into_iter()
            .map(|m| m.into_iter().map(|rva| BASE + u64::from(rva)).collect())
            .collect()
    }

    fn build(&self) -> Vec<u8> {
        const R: u32 = 0x4000_0000;
        const W: u32 = 0x8000_0000;
        const X: u32 = 0x2000_0000;
        let rva = |va: u64| u32::try_from(va - BASE).expect("rva");
        let (rdata, _) = layout_rdata(&self.imports);
        let mut sections: Vec<(&str, u32, u32, Vec<u8>, u32)> = vec![
            (".rdata", RDATA, SECTION_SPAN, rdata, R | 0x40),
            (".data", DATA, SECTION_SPAN, self.data.clone(), R | W | 0x40),
            (
                ".text",
                TEXT,
                self.text.len() as u32,
                self.text.clone(),
                R | X | 0x20,
            ),
        ];
        let mut next = (TEXT + self.text.len() as u32).next_multiple_of(SECTION_SPAN);
        let mut pdata_dir = None;
        if !self.pdata.is_empty() {
            let mut records = self.pdata.clone();
            records.sort_unstable();
            let mut raw = Vec::new();
            for (begin, end) in records {
                raw.extend_from_slice(&rva(begin).to_le_bytes());
                raw.extend_from_slice(&rva(end).to_le_bytes());
                raw.extend_from_slice(&UNWIND_RVA.to_le_bytes());
            }
            pdata_dir = Some((next, raw.len() as u32));
            let size = raw.len() as u32;
            sections.push((".pdata", next, size, raw, R | 0x40));
            next = (next + size).next_multiple_of(SECTION_SPAN);
        }
        let mut reloc_dir = None;
        if !self.relocs.is_empty() {
            let mut by_page: BTreeMap<u32, Vec<u16>> = BTreeMap::new();
            for &site in &self.relocs {
                let r = rva(site);
                by_page
                    .entry(r & !0xFFF)
                    .or_default()
                    .push(0xA000 | (r & 0xFFF) as u16);
            }
            let mut raw = Vec::new();
            for (page, mut entries) in by_page {
                if entries.len() % 2 == 1 {
                    entries.push(0);
                }
                raw.extend_from_slice(&page.to_le_bytes());
                raw.extend_from_slice(&(8 + 2 * entries.len() as u32).to_le_bytes());
                for e in entries {
                    raw.extend_from_slice(&e.to_le_bytes());
                }
            }
            reloc_dir = Some((next, raw.len() as u32));
            let size = raw.len() as u32;
            sections.push((".reloc", next, size, raw, R | 0x0200_0000 | 0x40));
            next = (next + size).next_multiple_of(SECTION_SPAN);
        }
        let size_of_image = next;

        let e_lfanew = 0x80usize;
        let opt = e_lfanew + 24;
        let table = opt + 240;
        let mut file_len = 0x400usize;
        let mut raw_offsets = Vec::new();
        for (_, _, _, raw, _) in &sections {
            raw_offsets.push(if raw.is_empty() { 0 } else { file_len });
            file_len = (file_len + raw.len()).next_multiple_of(0x200);
        }
        let mut buf = vec![0u8; file_len];
        put(&mut buf, 0, b"MZ");
        put(&mut buf, 0x3C, &(e_lfanew as u32).to_le_bytes());
        put(&mut buf, e_lfanew, b"PE\0\0");
        put(&mut buf, e_lfanew + 4, &0x8664u16.to_le_bytes());
        put(
            &mut buf,
            e_lfanew + 6,
            &(sections.len() as u16).to_le_bytes(),
        );
        put(&mut buf, e_lfanew + 20, &240u16.to_le_bytes());
        put(&mut buf, e_lfanew + 22, &0x0022u16.to_le_bytes());
        put(&mut buf, opt, &0x020Bu16.to_le_bytes());
        put(
            &mut buf,
            opt + 16,
            &self.entry.unwrap_or(TEXT).to_le_bytes(),
        );
        put(&mut buf, opt + 20, &TEXT.to_le_bytes());
        put(&mut buf, opt + 24, &BASE.to_le_bytes());
        put(&mut buf, opt + 32, &SECTION_SPAN.to_le_bytes());
        put(&mut buf, opt + 36, &0x200u32.to_le_bytes());
        put(&mut buf, opt + 56, &size_of_image.to_le_bytes());
        put(&mut buf, opt + 60, &0x400u32.to_le_bytes());
        put(&mut buf, opt + 68, &14u16.to_le_bytes());
        put(&mut buf, opt + 108, &16u32.to_le_bytes());
        let dir = |buf: &mut Vec<u8>, index: usize, (rva, size): (u32, u32)| {
            put(buf, opt + 112 + index * 8, &rva.to_le_bytes());
            put(buf, opt + 116 + index * 8, &size.to_le_bytes());
        };
        if !self.imports.is_empty() {
            dir(&mut buf, 1, (RDATA, 20 * (self.imports.len() as u32 + 1)));
        }
        if let Some(d) = pdata_dir {
            dir(&mut buf, 3, d);
        }
        if let Some(d) = reloc_dir {
            dir(&mut buf, 5, d);
        }
        for (i, ((name, rva, vsize, raw, chars), &off)) in
            sections.iter().zip(&raw_offsets).enumerate()
        {
            let h = table + i * 40;
            put(&mut buf, h, name.as_bytes());
            put(&mut buf, h + 8, &vsize.to_le_bytes());
            put(&mut buf, h + 12, &rva.to_le_bytes());
            put(&mut buf, h + 16, &(raw.len() as u32).to_le_bytes());
            put(&mut buf, h + 20, &(off as u32).to_le_bytes());
            put(&mut buf, h + 36, &chars.to_le_bytes());
            put(&mut buf, off, raw);
        }
        buf
    }
}

/// Tiny x86-64 emitter for the handful of encodings the fixtures need.
struct Asm {
    base: u64,
    bytes: Vec<u8>,
}

impl Asm {
    fn new(base: u64) -> Self {
        Self {
            base,
            bytes: Vec::new(),
        }
    }

    fn here(&self) -> u64 {
        self.base + self.bytes.len() as u64
    }

    fn raw(&mut self, bytes: &[u8]) -> &mut Self {
        self.bytes.extend_from_slice(bytes);
        self
    }

    /// `opcode` followed by a rel32 to `target` measured from the end of the instruction.
    fn rel32(&mut self, opcode: &[u8], target: u64) -> &mut Self {
        let end = self.here() + opcode.len() as u64 + 4;
        let rel = i32::try_from(target.wrapping_sub(end) as i64).expect("rel32 in range");
        self.raw(opcode).raw(&rel.to_le_bytes())
    }

    fn call(&mut self, target: u64) -> &mut Self {
        self.rel32(&[0xE8], target)
    }

    fn call_rip(&mut self, slot: u64) -> &mut Self {
        self.rel32(&[0xFF, 0x15], slot)
    }

    fn jmp_rip(&mut self, slot: u64) -> &mut Self {
        self.rel32(&[0xFF, 0x25], slot)
    }

    fn lea_rcx(&mut self, target: u64) -> &mut Self {
        self.rel32(&[0x48, 0x8D, 0x0D], target)
    }

    fn lea_rdx(&mut self, target: u64) -> &mut Self {
        self.rel32(&[0x48, 0x8D, 0x15], target)
    }

    fn lea_r8(&mut self, target: u64) -> &mut Self {
        self.rel32(&[0x4C, 0x8D, 0x05], target)
    }

    fn lea_rax(&mut self, target: u64) -> &mut Self {
        self.rel32(&[0x48, 0x8D, 0x05], target)
    }

    fn mov_rax_mem(&mut self, target: u64) -> &mut Self {
        self.rel32(&[0x48, 0x8B, 0x05], target)
    }

    fn mov_eax(&mut self, imm: u32) -> &mut Self {
        self.raw(&[0xB8]).raw(&imm.to_le_bytes())
    }

    fn add_rax(&mut self, imm: i8) -> &mut Self {
        self.raw(&[0x48, 0x83, 0xC0, imm as u8])
    }

    fn ret(&mut self) -> &mut Self {
        self.raw(&[0xC3])
    }

    /// Pads with `int3` up to `base + offset`.
    fn pad_to(&mut self, offset: u64) -> &mut Self {
        assert!(self.here() <= self.base + offset, "code overflows its slot");
        while self.here() < self.base + offset {
            self.bytes.push(0xCC);
        }
        self
    }
}

// ----- fixtures ----------------------------------------------------------------------------------

const KERNEL32: (&str, &[&str]) = (
    "KERNEL32.dll",
    &["ExitProcess", "VirtualAlloc", "GetTickCount64"],
);
const XGAMERUNTIME: (&str, &[&str]) = ("xgameruntime.dll", &["XTaskQueueCreate"]);

/// Addresses of the basic fixture.
struct Basic {
    main: u64,
    helper: u64,
    cb: u64,
    vt: u64,
    deep: u64,
    exit_thunk: u64,
    fatal: u64,
    /// `int3` padding after `helper` (a `lea` points here).
    pad: u64,
    /// Garbage right after `call [rip+ExitProcess]` in `main`.
    main_garbage: u64,
    /// Garbage right after `call exit_thunk` in `fatal`.
    fatal_garbage: u64,
    exit_slot: u64,
    valloc_slot: u64,
    xtq_slot: u64,
    /// `.data` function-pointer variable read by `call [rip+x]` (not relocated).
    fn_ptr_var: u64,
}

const BASIC: Basic = Basic {
    main: T,
    helper: T + 0x60,
    cb: T + 0x80,
    vt: T + 0xA0,
    deep: T + 0xC0,
    exit_thunk: T + 0xE0,
    fatal: T + 0xF0,
    pad: T + 0x70,
    main_garbage: 0,
    fatal_garbage: 0,
    exit_slot: 0,
    valloc_slot: 0,
    xtq_slot: 0,
    fn_ptr_var: D + 0x40,
};

/// The basic fixture: `main` (PE entry) calls imports, `helper`, `fatal`, a function-pointer
/// variable and `ExitProcess` (followed by garbage); `cb` is only `lea`-referenced; `vt` only
/// through a relocated `.data` pointer and itself `lea`s `deep`; `fatal` calls `exit_thunk`
/// (`jmp [rip+ExitProcess]`) followed by garbage. Bogus candidates: a `lea` into the middle of
/// `main`'s first instruction, a `lea` of `int3` padding, and relocated pointers into `helper`.
fn basic_fixture(with_pdata: bool) -> (Vec<u8>, Basic) {
    let mut pe = TestPe {
        imports: vec![KERNEL32, XGAMERUNTIME],
        ..TestPe::default()
    };
    let slots = pe.slots();
    let (exit, valloc, xtq) = (slots[0][0], slots[0][1], slots[1][0]);
    let b = &BASIC;

    let mut a = Asm::new(T);
    a.raw(&[0x48, 0x83, 0xEC, 0x28]) // sub rsp, 0x28
        .lea_rcx(b.cb)
        .lea_rdx(b.main + 1) // inside `sub rsp, 0x28`
        .lea_r8(b.pad)
        .call_rip(valloc)
        .call_rip(xtq)
        .call(b.helper)
        .call(b.fatal)
        .call_rip(b.fn_ptr_var)
        .mov_rax_mem(exit + 4) // misaligned IAT read
        .raw(&[0x31, 0xC9]) // xor ecx, ecx
        .call_rip(exit);
    let main_garbage = a.here();
    a.raw(&[0x06, 0x06, 0x06, 0x06]).pad_to(0x60);
    a.mov_eax(1).ret().pad_to(0x80); // helper (ret at +5)
    a.mov_eax(2).ret().pad_to(0xA0); // cb
    a.lea_rcx(b.deep).ret().pad_to(0xC0); // vt
    a.mov_eax(3).ret().pad_to(0xE0); // deep
    a.jmp_rip(exit).pad_to(0xF0); // exit_thunk
    a.call(b.exit_thunk);
    let fatal_garbage = a.here();
    a.raw(&[0x06, 0x06]).pad_to(0x100);
    pe.text = a.bytes;

    let mut data = vec![0u8; 0x48];
    for (i, value) in [b.vt, b.helper + 5, b.helper + 1, BASE + u64::from(RDATA), 0]
        .into_iter()
        .enumerate()
    {
        put(&mut data, 8 * i, &value.to_le_bytes());
        pe.relocs.push(D + 8 * i as u64);
    }
    put(&mut data, 0x40, &b.helper.to_le_bytes());
    pe.data = data;
    if with_pdata {
        pe.pdata = vec![
            (b.main, main_garbage + 4),
            (b.helper, b.helper + 6),
            (b.vt, b.vt + 8),
            (b.fatal, fatal_garbage + 2),
        ];
    }
    let fixture = Basic {
        main_garbage,
        fatal_garbage,
        exit_slot: exit,
        valloc_slot: valloc,
        xtq_slot: xtq,
        ..BASIC
    };
    (pe.build(), fixture)
}

/// `count` tiny functions 16 bytes apart; function `i` does `mov eax, i; call f(i+1); ret`, so
/// every call crosses into the next function (and, at bucket edges, the next shard).
fn chain_fixture(count: usize, with_pdata: bool) -> Vec<u8> {
    let mut a = Asm::new(T);
    let mut pdata = Vec::new();
    for i in 0..count {
        let start = a.here();
        a.mov_eax(i as u32);
        if i + 1 < count {
            a.call(T + 16 * (i as u64 + 1));
        }
        a.ret();
        pdata.push((start, a.here()));
        a.pad_to(16 * (i as u64 + 1));
    }
    TestPe {
        text: a.bytes,
        pdata: if with_pdata { pdata } else { Vec::new() },
        ..TestPe::default()
    }
    .build()
}

// ----- harness -----------------------------------------------------------------------------------

struct TempDir(PathBuf);

impl TempDir {
    fn new(label: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = env::temp_dir().join(format!(
            "playgta6_pipeline_{label}_{}_{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("create temp dir");
        Self(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

const MINI_SHIM: &str = "extern \"C\" uint64_t pg6_win32_virtual_alloc(uint64_t a, uint64_t s, uint32_t t, uint32_t p);\n";

/// Writes `pe` and a one-shim header into `dir` and returns options building into `dir/out`.
fn options_for(dir: &Path, pe: &[u8]) -> PipelineOptions {
    fs::create_dir_all(dir.join("rt").join("os")).expect("create header dir");
    let pe_path = dir.join("Synthetic.exe");
    fs::write(&pe_path, pe).expect("write PE");
    let header = dir.join("rt").join("os").join("mini.h");
    fs::write(&header, MINI_SHIM).expect("write header");
    let mut options = PipelineOptions::new(pe_path, dir.join("out"));
    options.shim_headers = vec![header];
    options.jobs = 2;
    options
}

/// Every file under `dir` as `(relative '/'-path, bytes)`, sorted.
fn read_tree(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
        for entry in fs::read_dir(dir).expect("list dir") {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                walk(root, &path, out);
            } else {
                let rel = path
                    .strip_prefix(root)
                    .expect("below root")
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
                    .join("/");
                out.insert(rel, fs::read(&path).expect("read file"));
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(dir, dir, &mut out);
    out
}

fn status_of(report: &PipelineReport, va: u64) -> PointerStatus {
    report
        .code_pointers
        .pointers
        .iter()
        .find(|p| p.va == va)
        .unwrap_or_else(|| panic!("no candidate {va:#x}"))
        .status
}

// ----- end to end --------------------------------------------------------------------------------

#[test]
fn end_to_end_build_writes_consistent_artifacts() {
    let tmp = TempDir::new("e2e");
    let (pe, b) = basic_fixture(false);
    let options = options_for(tmp.path(), &pe);
    let report = run(&options).expect("build succeeds");
    let out = options.module_dir();

    let tree = read_tree(&out);
    let expected_files = [
        "build-manifest.json",
        "image/guest.pgim",
        "image/guest.pgim.json",
        "lift/functions.ll",
        "lift/shard-00000.ll",
        "report.txt",
        "stubs/pg6_guest_seam.h",
        "stubs/pg6_import_manifest.json",
        "stubs/pg6_import_thunks.cpp",
        "stubs/pg6_import_thunks.h",
    ];
    assert_eq!(
        tree.keys().map(String::as_str).collect::<Vec<_>>(),
        expected_files
    );

    // The manifest lists every other file with its size and hash.
    let manifest = String::from_utf8(tree["build-manifest.json"].clone()).expect("utf8");
    assert_eq!(manifest, report.manifest);
    assert_eq!(report.files.len(), tree.len() - 1);
    for (path, bytes) in &tree {
        if path == MANIFEST_FILE {
            continue;
        }
        let line = format!(
            "{{\"path\": \"{path}\", \"size\": {}, \"fnv1a64\": \"0x{:016x}\"}}",
            bytes.len(),
            image::fnv1a_64(bytes)
        );
        assert!(
            manifest.contains(&line),
            "manifest lacks {line}\n{manifest}"
        );
    }
    assert!(
        manifest
            .starts_with("{\n  \"schema\": \"playgta6-recomp-build\",\n  \"schema_version\": 1,\n")
    );
    assert!(manifest.contains(&format!(
        "\"input\": {{\"size\": {}, \"fnv1a64\": \"0x{:016x}\"}}",
        pe.len(),
        image::fnv1a_64(&pe)
    )));
    assert!(manifest.contains("\"shim_headers\": [\"os/mini.h\"]"));
    assert!(manifest.contains("\"llvm\": null"));

    // No paths, temp names or code bytes leak into the manifest or the report.
    let temp_name = tmp
        .path()
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    for text in [&report.manifest, &report.report] {
        assert!(!text.contains(&temp_name), "temp dir name leaked");
        assert!(!text.contains("Synthetic"), "input file name leaked");
        assert!(
            !text.contains(":\\") && !text.contains("/tmp"),
            "absolute path leaked"
        );
    }
    assert_eq!(tree["report.txt"], report.report.as_bytes());

    // Discovery found every function, including those only reachable through code pointers.
    let d = &report.discovery;
    assert_eq!(d.functions, 8, "{}", report.report);
    assert_eq!(report.lift.functions, 8);
    assert_eq!(report.lift.shards.len(), 1);
    assert_eq!(
        d.rounds, 3,
        "pointers, then the pointer inside `vt`, then stable"
    );
    assert_eq!(
        d.invalid_blocks, 0,
        "no garbage decoded after no-return calls"
    );
    assert_eq!(d.no_return_calls, 2);

    // Registration table: every function, sorted.
    let table = String::from_utf8(tree["lift/functions.ll"].clone()).expect("utf8");
    let registered: Vec<u64> = table
        .lines()
        .filter_map(|l| l.trim().strip_prefix("%Pg6LiftedFunctionEntry { i64 "))
        .map(|l| l.split(',').next().unwrap().parse::<i64>().unwrap() as u64)
        .collect();
    // `helper + 5` (a relocated pointer at an instruction boundary) splits `helper`.
    let mut expected = vec![
        b.main,
        b.helper,
        b.helper + 5,
        b.cb,
        b.vt,
        b.deep,
        b.exit_thunk,
        b.fatal,
    ];
    expected.sort_unstable();
    assert_eq!(registered, expected);
    assert!(table.contains("@__playgta6_lifted_function_count_main = constant i64 8, align 8"));

    // Imports: one bound, ExitProcess and XTaskQueueCreate called but unresolved.
    assert_eq!(report.imports.bound, 1);
    assert_eq!(report.imports.total, 4);
    assert!(report.report.contains("KERNEL32.dll!ExitProcess (no_shim)"));
    assert!(report.report.contains("coverage:"));
    assert!(report.report.contains("dispatcher: 8 registered functions"));

    // The packed image is the R6 container of the same PE.
    let packed = image::read_image(&tree["image/guest.pgim"]).expect("container validates");
    assert_eq!(packed.image_base, BASE);
}

#[test]
fn rejects_bad_options_and_missing_input() {
    let tmp = TempDir::new("options");
    let (pe, _) = basic_fixture(false);
    let mut options = options_for(tmp.path(), &pe);
    options.jobs = 0;
    assert!(matches!(run(&options), Err(PipelineError::Options(_))));
    options.jobs = 1;
    options.shard_size = 0;
    assert!(matches!(run(&options), Err(PipelineError::Options(_))));
    options.shard_size = 4;
    options.pe_path = tmp.path().join("missing.exe");
    assert!(matches!(run(&options), Err(PipelineError::Io { .. })));
    fs::write(&options.pe_path, b"MZ not a PE").expect("write");
    assert!(matches!(run(&options), Err(PipelineError::Parse(_))));
}

#[test]
fn outputs_are_identical_across_runs_and_job_counts() {
    let tmp = TempDir::new("determinism");
    let pe = chain_fixture(600, true);
    let mut trees = Vec::new();
    for (run_index, jobs) in [1, 8, 8].into_iter().enumerate() {
        let mut options = options_for(tmp.path(), &pe);
        options.out_dir = tmp.path().join(format!("out{run_index}"));
        options.jobs = jobs;
        options.shard_size = 64;
        let report = run(&options).expect("build succeeds");
        assert_eq!(report.lift.shards.len(), 10);
        trees.push(read_tree(&options.module_dir()));
    }
    assert_eq!(trees[0].len(), 10 + 9);
    assert_eq!(trees[0], trees[1], "jobs 1 vs 8");
    assert_eq!(trees[1], trees[2], "repeated run");
}

#[test]
fn native_target_changes_only_the_target_lines() {
    let tmp = TempDir::new("target");
    let pe = chain_fixture(300, true);
    let build = |name: &str, target: Option<LiftTarget>| {
        let mut options = options_for(tmp.path(), &pe);
        options.out_dir = tmp.path().join(name);
        options.shard_size = 64;
        if let Some(target) = target {
            options.target = target;
        }
        run(&options).expect("build succeeds");
        read_tree(&options.module_dir())
    };
    let default = build("default", None);
    let explicit = build("wasm64", Some(LiftTarget::Wasm64));
    assert_eq!(default, explicit, "the default target is wasm64 and changes nothing");
    let manifest = String::from_utf8(default[MANIFEST_FILE].clone()).expect("utf-8");
    assert!(!manifest.contains("\"target\""), "wasm64 manifests keep their key set");
    for (path, bytes) in &default {
        if path.ends_with(".ll") {
            let text = String::from_utf8_lossy(bytes);
            assert!(text.contains(&format!("target triple = \"{}\"", lift::WASM64_TRIPLE)));
            assert!(text.contains(&format!("target datalayout = \"{WASM64_DATALAYOUT}\"")));
        }
    }

    let native = LiftTarget::parse("native:x86_64-unknown-linux-gnu", None).expect("target");
    let retargeted = build("native", Some(native.clone()));
    assert_eq!(
        default.keys().collect::<Vec<_>>(),
        retargeted.keys().collect::<Vec<_>>()
    );
    for (path, bytes) in &default {
        let other = &retargeted[path];
        if path.ends_with(".ll") {
            let wasm = String::from_utf8(bytes.clone()).expect("utf-8");
            assert_eq!(
                String::from_utf8(other.clone()).expect("utf-8"),
                native.retarget(wasm),
                "{path}: only the target lines differ"
            );
        } else if path != MANIFEST_FILE && path != REPORT_FILE {
            assert_eq!(bytes, other, "{path} does not depend on the target");
        }
    }
    let manifest = String::from_utf8(retargeted[MANIFEST_FILE].clone()).expect("utf-8");
    assert!(manifest.contains("\"target\": \"native:x86_64-unknown-linux-gnu\""), "{manifest}");
    assert!(manifest.contains("\"target_cpu\": \"x86-64-v3\""));
}

#[test]
fn shards_partition_functions_and_declare_cross_shard_callees() {
    let tmp = TempDir::new("shards");
    let count = 1000;
    let mut options = options_for(tmp.path(), &chain_fixture(count, false));
    options.shard_size = 96;
    options.jobs = 4;
    let report = run(&options).expect("build succeeds");
    assert_eq!(report.discovery.functions, count);
    let shards = &report.lift.shards;
    assert_eq!(shards.len(), count.div_ceil(96));
    let mut next = 0;
    for (i, shard) in shards.iter().enumerate() {
        assert_eq!(shard.index, i);
        assert_eq!(shard.first, T + 16 * next as u64, "contiguous, ascending");
        next += shard.functions;
        assert_eq!(shard.last, T + 16 * (next as u64 - 1));
        assert_eq!(
            shard.functions,
            if i + 1 < shards.len() { 96 } else { count % 96 }
        );
    }
    assert_eq!(next, count);

    let mut defined = BTreeMap::new();
    let mut declared = Vec::new();
    for shard in shards {
        let ir = fs::read_to_string(options.module_dir().join(&shard.ll.path)).expect("read shard");
        for line in ir.lines() {
            if let Some(rest) = line.strip_prefix("define void @") {
                defined.insert(rest.split('(').next().unwrap().to_string(), shard.index);
            } else if let Some(rest) = line.strip_prefix("declare void @sub_") {
                declared.push((
                    shard.index,
                    format!("sub_{}", rest.split('(').next().unwrap()),
                ));
            }
        }
    }
    assert_eq!(defined.len(), count);
    assert_eq!(
        declared.len(),
        shards.len() - 1,
        "one cross-shard call per shard edge"
    );
    for (shard, symbol) in declared {
        assert_eq!(
            defined.get(&symbol),
            Some(&(shard + 1)),
            "{symbol} defined in the next shard"
        );
    }
}

#[test]
fn stale_shards_of_a_previous_build_are_removed() {
    let tmp = TempDir::new("stale");
    let mut options = options_for(tmp.path(), &chain_fixture(40, false));
    options.shard_size = 4;
    run(&options).expect("first build");
    let lift_dir = options.module_dir().join(LIFT_DIR);
    assert!(lift_dir.join("shard-00009.ll").is_file());
    fs::write(lift_dir.join("notes.txt"), b"kept").expect("write unrelated file");
    options.shard_size = 64;
    run(&options).expect("second build");
    let mut names: Vec<String> = fs::read_dir(&lift_dir)
        .expect("list")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(names, ["functions.ll", "notes.txt", "shard-00000.ll"]);
}

#[test]
fn plan_shards_covers_every_function_once() {
    assert!(plan_shards(&[], 8, 100).is_empty());
    assert_eq!(
        plan_shards(&[1; 5], 8, 100),
        vec![Range { start: 0, end: 5 }]
    );
    assert_eq!(plan_shards(&[1; 16], 8, 100), [0..8, 8..16]);
    assert_eq!(plan_shards(&[1; 17], 8, 100), [0..8, 8..16, 16..17]);
    // Instruction budget: close before exceeding it; oversized functions stand alone.
    assert_eq!(
        plan_shards(&[40, 40, 40, 500, 10, 90, 1], 8, 100),
        [0..2, 2..3, 3..4, 4..6, 6..7]
    );
    assert_eq!(shard_file_name(7, "ll"), "shard-00007.ll");
}

#[test]
fn registration_table_layout() {
    let mut out = Vec::new();
    write_registration_table("main", &[0x1_4000_1000, 0x1_4000_2000], &mut out).expect("render");
    let text = String::from_utf8(out).expect("utf8");
    assert!(text.contains("%Pg6LiftedFunctionEntry = type { i64, ptr }\n"));
    assert!(text.contains("declare void @sub_140001000(ptr)\n"));
    assert!(text.contains(
        "@__playgta6_lifted_functions_main = constant [2 x %Pg6LiftedFunctionEntry] [\n  \
         %Pg6LiftedFunctionEntry { i64 5368713216, ptr @sub_140001000 },\n  \
         %Pg6LiftedFunctionEntry { i64 5368717312, ptr @sub_140002000 }\n], align 8\n"
    ));
    assert!(text.contains("@__playgta6_lifted_function_count_main = constant i64 2, align 8\n"));
    assert!(text.ends_with(
        "declare i32 @pg6_guest_register_lifted_functions(ptr, i64)\n\n\
         define i32 @__playgta6_register_lifted_functions_main() {\n\
         entry:\n  \
         %ok = call i32 @pg6_guest_register_lifted_functions(ptr @__playgta6_lifted_functions_main, i64 2)\n  \
         ret i32 %ok\n}\n"
    ));

    let mut empty = Vec::new();
    write_registration_table("main", &[], &mut empty).expect("render");
    let text = String::from_utf8(empty).expect("utf8");
    assert!(text.contains("constant [0 x %Pg6LiftedFunctionEntry] zeroinitializer, align 8"));
    assert!(text.contains("constant i64 0, align 8"));
}

// ----- refinements -------------------------------------------------------------------------------

#[test]
fn no_return_imports_stop_decoding_after_exit_process() {
    let (file, b) = basic_fixture(false);
    let pe = Pe64::parse(&file).expect("parses");
    assert_eq!(
        refine::no_return_imports(&pe),
        [refine::NoReturnImport {
            slot: b.exit_slot,
            label: "KERNEL32.dll!ExitProcess".to_string(),
        }]
    );

    // Without the refinement, discovery decodes the garbage after `call [ExitProcess]`.
    let plain = discover(&CodeImage::from_pe(&pe, &file));
    let main = plain.function(b.main).expect("main");
    let block = main.entry_block().expect("entry block");
    assert_eq!(block.terminator, Terminator::Invalid);
    assert_eq!(block.end, b.main_garbage, "decoding ran into the garbage");
    let diagnostics: Vec<_> = main.diagnostics().map(|d| (d.address, d.kind)).collect();
    assert_eq!(
        diagnostics,
        [(b.main_garbage, DiagnosticKind::InvalidInstruction)]
    );

    // With it, the call ends the block and nothing after it is decoded.
    let seeded = seed(&pe, &file);
    assert_eq!(seeded.image.no_return, [b.exit_slot]);
    let refined = discover(&seeded.image);
    let main = refined.function(b.main).expect("main");
    let block = main.entry_block().expect("entry block");
    assert_eq!(block.terminator, Terminator::NoReturnCall);
    assert_eq!(block.end, b.main_garbage);
    assert!(block.successors.is_empty());
    assert_eq!(main.diagnostics().count(), 0);

    // Only listed names from the right DLL family count.
    let mut other = pe.clone();
    other.imports[0].dll_name = "mylib.dll".to_string();
    assert!(refine::no_return_imports(&other).is_empty());
}

#[test]
fn no_return_thunks_are_found_by_the_fixpoint() {
    let (file, b) = basic_fixture(false);
    let pe = Pe64::parse(&file).expect("parses");
    let guest = image::build_guest_image(&pe, &file, ImageOptions::default()).expect("image");
    let mut seeded = seed(&pe, &file);

    // Round one alone still decodes the garbage after `call exit_thunk`.
    let first = discover(&seeded.image);
    let entry = first.function(b.fatal).unwrap().entry_block().unwrap();
    assert_eq!(entry.terminator, Terminator::Invalid);
    assert_eq!(entry.diagnostics[0].address, b.fatal_garbage);

    let (analysis, outcome) = refine::discover_refined(&mut seeded.image, &guest.image);
    assert_eq!(outcome.no_return_thunks, [b.exit_thunk]);
    assert!(seeded.image.no_return.contains(&b.exit_thunk));
    let fatal = analysis.function(b.fatal).expect("fatal");
    let entry = fatal.entry_block().unwrap();
    assert_eq!(entry.terminator, Terminator::NoReturnCall);
    assert_eq!(entry.end, b.fatal_garbage);
    assert_eq!(fatal.diagnostics().count(), 0);
}

#[test]
fn code_pointers_without_pdata() {
    let tmp = TempDir::new("cp_nopdata");
    let (pe, b) = basic_fixture(false);
    let report = run(&options_for(tmp.path(), &pe)).expect("build");
    let cp = &report.code_pointers;
    assert!(!report.pdata_present);
    assert_eq!(
        status_of(&report, b.cb),
        PointerStatus::Accepted,
        "lea only"
    );
    assert_eq!(
        status_of(&report, b.vt),
        PointerStatus::Accepted,
        "data pointer only"
    );
    assert_eq!(
        status_of(&report, b.deep),
        PointerStatus::Accepted,
        "lea in vt, round 2"
    );
    assert_eq!(
        status_of(&report, b.main + 1),
        PointerStatus::InsideInstruction
    );
    assert_eq!(status_of(&report, b.pad), PointerStatus::Implausible);
    assert_eq!(
        status_of(&report, b.helper + 5),
        PointerStatus::Accepted,
        "instruction boundary of a discovered block: split there"
    );
    assert_eq!(
        status_of(&report, b.helper + 1),
        PointerStatus::InsideInstruction
    );
    assert!(
        cp.pointers.iter().all(|p| p.va != BASE + u64::from(RDATA)),
        "data addresses are not candidates"
    );
    let vt = cp.pointers.iter().find(|p| p.va == b.vt).unwrap();
    assert_eq!(
        vt.sources.iter().copied().collect::<Vec<_>>(),
        [PointerSource::BaseReloc]
    );
    let inside = cp.pointers.iter().find(|p| p.va == b.helper + 1).unwrap();
    assert_eq!(
        inside.sources.iter().copied().collect::<Vec<_>>(),
        [PointerSource::BaseReloc]
    );

    assert_eq!(cp.count(PointerSource::Lea, PointerStatus::Accepted), 2);
    assert_eq!(
        cp.count(PointerSource::BaseReloc, PointerStatus::Accepted),
        2
    );
    assert_eq!(
        cp.added_entries().collect::<Vec<_>>(),
        [b.helper + 5, b.cb, b.vt, b.deep]
    );
    let gaps: Vec<u64> = cp.not_lifted().map(|p| p.va).collect();
    assert_eq!(gaps, [b.main + 1, b.helper + 1, b.pad]);
    assert!(report.manifest.contains(&format!(
        "{{\"va\": \"0x{:x}\", \"sources\": \"lea\", \"status\": \"inside_instruction\"}}",
        b.main + 1
    )));
    assert!(
        report
            .manifest
            .contains("\"lea\": {\"known_entry\": 0, \"accepted\": 2,")
    );
}

#[test]
fn code_pointers_with_pdata_trust_exact_function_starts() {
    let tmp = TempDir::new("cp_pdata");
    let (pe, b) = basic_fixture(true);
    let report = run(&options_for(tmp.path(), &pe)).expect("build");
    assert!(report.pdata_present);
    assert_eq!(report.pdata.functions, 4);
    assert_eq!(
        status_of(&report, b.vt),
        PointerStatus::KnownEntry,
        "a .pdata start"
    );
    assert_eq!(status_of(&report, b.main + 1), PointerStatus::PdataInterior);
    assert_eq!(
        status_of(&report, b.helper + 5),
        PointerStatus::PdataInterior
    );
    assert_eq!(
        status_of(&report, b.helper + 1),
        PointerStatus::PdataInterior
    );
    // Leaf functions without unwind data fall back to the decode rule.
    assert_eq!(status_of(&report, b.cb), PointerStatus::Accepted);
    assert_eq!(status_of(&report, b.deep), PointerStatus::Accepted);
    assert_eq!(status_of(&report, b.pad), PointerStatus::Implausible);
    assert_eq!(report.discovery.functions, 7);
    assert_eq!(
        report.discovery.rounds, 2,
        "vt is seeded, so deep is found in round 1"
    );
}

#[test]
fn import_slot_consistency_is_reported() {
    let tmp = TempDir::new("slots");
    let (pe, b) = basic_fixture(false);
    let report = run(&options_for(tmp.path(), &pe)).expect("build");
    let s = &report.slots;
    assert_eq!(
        s.slot_reads, 4,
        "VirtualAlloc, XTaskQueueCreate, ExitProcess, fn pointer var"
    );
    assert_eq!(s.iat_reads, 3);
    assert_eq!(s.non_iat_reads, [b.fn_ptr_var]);
    assert_eq!(s.misaligned, [b.exit_slot + 4]);
    assert_eq!(
        s.referenced_unresolved,
        [
            (
                "KERNEL32.dll".to_string(),
                "KERNEL32.dll!ExitProcess".to_string(),
                "no_shim"
            ),
            (
                "xgameruntime.dll".to_string(),
                "xgameruntime.dll!XTaskQueueCreate".to_string(),
                "no_shim"
            ),
        ]
    );
    assert_eq!(
        s.unreferenced,
        [("KERNEL32.dll!GetTickCount64".to_string(), false)]
    );
    assert!(b.valloc_slot != 0 && b.xtq_slot != 0);
}

// ----- hygiene -----------------------------------------------------------------------------------

#[test]
fn output_dir_hygiene() {
    let tmp = TempDir::new("hygiene");
    let repo = tmp.path().join("repo");
    fs::create_dir_all(repo.join(".git").join("info")).expect("fake repo");
    fs::create_dir_all(repo.join("sub")).expect("subdir");
    fs::write(
        repo.join(".gitignore"),
        "# outputs\nwork/\n/build-*\nout*/\n!out-keep/\n",
    )
    .expect("gitignore");
    fs::write(repo.join("sub").join(".gitignore"), "local/\n").expect("nested gitignore");
    fs::write(repo.join(".git").join("info").join("exclude"), "scratch\n").expect("exclude");

    let ok = |p: PathBuf| check_output_dir(&p).map_err(|e| format!("{}: {e}", p.display()));
    ok(repo.join("work")).expect("ignored dir");
    ok(repo.join("work").join("recomp").join("deep")).expect("below an ignored dir");
    ok(repo.join("build-7")).expect("anchored glob");
    ok(repo.join("out-a")).expect("glob");
    ok(repo.join("sub").join("local").join("x")).expect("nested .gitignore");
    ok(repo.join("scratch")).expect(".git/info/exclude");
    ok(repo.join("src").join("..").join("work")).expect("normalised");
    ok(tmp.path().join("elsewhere")).expect("outside the repository");

    for bad in [
        repo.clone(),
        repo.join("src"),
        repo.join("out-keep"),
        repo.join("sub").join("build-7"),
        repo.join("local"),
        repo.join(".git").join("objects"),
        repo.join("work").join("..").join("src"),
    ] {
        let err = check_output_dir(&bad).expect_err(&format!("{} must be refused", bad.display()));
        assert!(err.contains("refusing"), "{err}");
    }
    let err = check_output_dir(&repo.join("src")).unwrap_err();
    assert!(err.contains("work/"), "points at work/: {err}");

    // Linked worktrees and submodules have a `.git` file.
    let worktree = tmp.path().join("worktree");
    fs::create_dir_all(&worktree).expect("worktree");
    fs::write(worktree.join(".git"), "gitdir: elsewhere\n").expect("git file");
    assert!(check_output_dir(&worktree.join("out")).is_err());

    // An existing file is not an output directory.
    fs::write(tmp.path().join("file"), b"x").expect("file");
    assert!(check_output_dir(&tmp.path().join("file")).is_err());

    // The pipeline checks before reading anything.
    let mut options = options_for(tmp.path(), &chain_fixture(2, false));
    options.out_dir = repo.join("src").join("gen");
    assert!(matches!(run(&options), Err(PipelineError::OutputDir(_))));
    assert!(!repo.join("src").exists(), "nothing written");
    options.out_dir = repo.join("work").join("build");
    run(&options).expect("gitignored work/ is accepted");
}

#[test]
fn checkout_shim_headers_come_from_runtime() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let (headers, root) = checkout_shim_headers(crate_dir).expect("inside the checkout");
    assert!(root.ends_with("runtime"));
    assert!(headers.iter().any(|h| h.ends_with("win32_kernel.h")));
    assert!(headers.last().unwrap().ends_with("guest_callbacks.h"));
    assert!(headers[headers.len() - 2].ends_with("d3d12_xs.h"));
    let mut sorted = headers[..headers.len() - 2].to_vec();
    sorted.sort();
    assert_eq!(sorted, headers[..headers.len() - 2]);

    let tmp = TempDir::new("checkout");
    assert!(checkout_shim_headers(tmp.path()).is_none());
    let (pe, _) = basic_fixture(false);
    let mut options = options_for(tmp.path(), &pe);
    options.shim_headers = headers;
    options.include_root = Some(root);
    let report = run(&options).expect("build with the real runtime headers");
    assert!(report.imports.shims > 0);
    assert!(
        report.imports.headers.iter().all(|h| h.starts_with("os/")
            || h.starts_with("gpu/")
            || h == "guest/guest_callbacks.h")
    );
}

// ----- tool-gated: compile, link and execute -----------------------------------------------------

const PROBE_LL: &str = "\
target triple = \"wasm64-unknown-unknown\"

define void @probe(ptr %p) {
entry:
  %v = load i64, ptr %p, align 1
  %w = add i64 %v, 1
  store i64 %w, ptr %p, align 1
  ret void
}
";

/// An LLVM tool that compiles a trivial wasm64 module, or a skip reason.
fn llvm_tool() -> Result<LlvmTool, String> {
    let tool = find_llvm_tool().ok_or("no LLVM tool (llc/clang/llvm-tools)")?;
    let dir = TempDir::new("probe_llvm");
    let ll = dir.path().join("probe.ll");
    fs::write(&ll, PROBE_LL).map_err(|e| e.to_string())?;
    compile_to_wasm64_object(&tool, &ll, &dir.path().join("probe.o"))
        .map_err(|e| format!("{} cannot compile wasm64: {e}", tool.path().display()))?;
    Ok(tool)
}

/// `(linker, needs -flavor wasm)` candidates.
fn linkers() -> Vec<(PathBuf, bool)> {
    let mut out = Vec::new();
    if let Some(path) = env::var_os("PLAYGTA6_WASM_LD").filter(|v| !v.is_empty()) {
        let path = PathBuf::from(path);
        let flavor = path.to_string_lossy().contains("rust-lld");
        out.push((path, flavor));
    }
    out.extend(find_on_path("wasm-ld").map(|p| (p, false)));
    let rust_lld = rustc_tool_dir()
        .map(|dir| dir.join(format!("rust-lld{}", env::consts::EXE_SUFFIX)))
        .filter(|p| p.is_file());
    out.extend(
        rust_lld
            .into_iter()
            .chain(find_on_path("rust-lld"))
            .map(|p| (p, true)),
    );
    out
}

fn link(
    linker: &(PathBuf, bool),
    objects: &[PathBuf],
    exports: &[&str],
    wasm: &Path,
) -> Result<(), String> {
    let mut cmd = Command::new(&linker.0);
    if linker.1 {
        cmd.args(["-flavor", "wasm"]);
    }
    cmd.args([
        "-mwasm64",
        "--no-entry",
        "--allow-undefined",
        "--export-table",
    ])
    .arg(format!("--initial-memory={}", 8u64 << 20));
    for name in exports {
        cmd.arg(format!("--export={name}"));
    }
    cmd.arg("-o").arg(wasm).args(objects);
    let output = cmd.output().map_err(|e| e.to_string())?;
    if output.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).into_owned())
    }
}

const RUNNER_JS: &str = r#"'use strict';
// Runs `entry` after the module registered its function table through the generated
// `__playgta6_register_lifted_functions_<module>`, which calls the runtime's
// `pg6_guest_register_lifted_functions(entries, count)`; this stands in for the runtime.
const fs = require('fs');
const [wasmPath, entry, registerSymbol, countSymbol] = process.argv.slice(2);
const STATE = 0x100000, STACK_TOP = 0x1F0000, RSP = 4;
const byVa = new Map();
const vas = [];
let ex;
const fail = (what, va) => { throw new Error(what + ' 0x' + BigInt(va).toString(16)); };
const view = () => new DataView(ex.memory.buffer);
const getFn = (i) => { try { return ex.__indirect_function_table.get(Number(i)); } catch (e) { return ex.__indirect_function_table.get(BigInt(i)); } };
const instance = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(wasmPath)), { env: {
  pg6_guest_register_lifted_functions(entries, count) {
    const v = view();
    for (let i = 0; i < Number(count); i++) {
      const va = v.getBigUint64(Number(entries) + 16 * i, true);
      if (vas.length && va <= BigInt(vas[vas.length - 1])) return 0;
      vas.push('0x' + va.toString(16));
      byVa.set(va, getFn(v.getBigUint64(Number(entries) + 16 * i + 8, true)));
    }
    return 1;
  },
  __playgta6_indirect_call(state, target) {
    const fn = byVa.get(BigInt(target));
    if (!fn) fail('no registered function for', target);
    fn(state);
  },
  __playgta6_unimplemented(state, va) { fail('unimplemented', va); },
  __playgta6_divide_error(state, va) { fail('divide error', va); },
  __playgta6_trap(state, va) { fail('trap', va); },
} });
ex = instance.exports;
if (ex[registerSymbol]() !== 1) throw new Error('registration rejected');
const countGlobal = ex[countSymbol];
let v = view();
const count = Number(v.getBigUint64(Number(typeof countGlobal === 'object' ? countGlobal.value : countGlobal), true));
if (count !== vas.length) throw new Error('count mismatch');
for (let i = 0; i < 16; i++) v.setBigUint64(STATE + 8 * i, 0n, true);
v.setBigUint64(STACK_TOP - 8, 0xC0DEC0DEn, true);
v.setBigUint64(STATE + 8 * RSP, BigInt(STACK_TOP - 8), true);
ex[entry](BigInt(STATE));
v = view();
process.stdout.write(`${count} ${vas.join(',')} ${v.getBigUint64(STATE, true)} ${v.getBigUint64(STATE + 8 * RSP, true) - BigInt(STACK_TOP)}\n`);
"#;

#[test]
fn every_shard_compiles_to_a_wasm64_object() {
    let tool = match llvm_tool() {
        Ok(tool) => tool,
        Err(reason) => {
            println!("SKIP every_shard_compiles_to_a_wasm64_object: {reason}");
            return;
        }
    };
    let tmp = TempDir::new("objects");
    let (pe, _) = basic_fixture(false);
    let mut options = options_for(tmp.path(), &pe);
    options.emit_obj = true;
    options.shard_size = 2;
    options.jobs = 3;
    let report = run(&options).expect("build with objects");
    assert_eq!(report.lift.shards.len(), 4);
    for shard in &report.lift.shards {
        let obj = shard.obj.as_ref().expect("object recorded");
        assert!(options.module_dir().join(&obj.path).is_file());
    }
    assert!(report.lift.registration_obj.is_some());
    let llvm = report.llvm.as_ref().expect("tool recorded");
    assert!(
        report
            .manifest
            .contains(&format!("\"name\": \"{}\"", llvm.name))
    );
    assert!(
        !report.manifest.contains(&tool.path().display().to_string()),
        "no tool path"
    );
    let leftovers: Vec<_> = read_tree(&options.module_dir().join(LIFT_DIR))
        .into_keys()
        .filter(|p| p.ends_with(".bc"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "intermediate files removed: {leftovers:?}"
    );

    // Objects are deterministic too.
    options.out_dir = tmp.path().join("out2");
    options.jobs = 1;
    run(&options).expect("second build");
    assert_eq!(
        read_tree(&tmp.path().join("out").join(DEFAULT_MODULE)),
        read_tree(&options.module_dir())
    );
}

#[test]
fn cross_shard_chain_links_and_dispatches_through_the_registration_table() {
    let skip = |reason: String| println!("SKIP cross_shard_chain_links_and_dispatches: {reason}");
    if let Err(reason) = llvm_tool() {
        return skip(reason);
    }
    let Some(node) = find_on_path("node") else {
        return skip("node not found on PATH".into());
    };
    // f_a -> f_b -> f_c (direct calls across shards); f_c reaches f_d, which only a `lea` names,
    // through `call rax` and the dispatcher's registration table.
    let (fa, fb, fc, fd) = (T, T + 0x20, T + 0x40, T + 0x60);
    let mut a = Asm::new(T);
    a.call(fb).add_rax(1).ret().pad_to(0x20);
    a.call(fc).add_rax(10).ret().pad_to(0x40);
    a.lea_rax(fd)
        .raw(&[0xFF, 0xD0])
        .add_rax(100)
        .ret()
        .pad_to(0x60);
    a.mov_eax(1000).ret().pad_to(0x80);
    let pe = TestPe {
        text: a.bytes,
        ..TestPe::default()
    }
    .build();

    let tmp = TempDir::new("exec");
    let mut options = options_for(tmp.path(), &pe);
    options.emit_obj = true;
    options.shard_size = 1;
    options.module = "chain".to_string();
    let report = run(&options).expect("build");
    assert_eq!(report.lift.shards.len(), 4, "one function per shard");
    let mut objects: Vec<PathBuf> = report
        .lift
        .shards
        .iter()
        .map(|s| options.module_dir().join(&s.obj.as_ref().unwrap().path))
        .collect();
    objects.push(
        options
            .module_dir()
            .join(&report.lift.registration_obj.as_ref().unwrap().path),
    );

    let entry = function_symbol(fa);
    let (_, count) = registration_symbols("chain");
    let register = register_function_symbol("chain");
    let exports = [entry.as_str(), register.as_str(), count.as_str()];
    let wasm = tmp.path().join("chain.wasm");
    let mut failures = Vec::new();
    let linked = linkers().into_iter().any(|linker| {
        link(&linker, &objects, &exports, &wasm)
            .map_err(|e| {
                failures.push(format!(
                    "{}: {}",
                    linker.0.display(),
                    e.lines().next().unwrap_or("")
                ))
            })
            .is_ok()
    });
    if !linked {
        return skip(format!(
            "no wasm linker could link the shards ({})",
            failures.join("; ")
        ));
    }
    let script = tmp.path().join("runner.js");
    fs::write(&script, RUNNER_JS).expect("write runner");
    let mut last_error = String::new();
    for flags in [&[][..], &["--experimental-wasm-memory64"][..]] {
        let output = Command::new(&node)
            .args(flags)
            .arg(&script)
            .arg(&wasm)
            .arg(&entry)
            .arg(&register)
            .arg(&count)
            .output()
            .expect("run node");
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert_eq!(
                stdout.trim(),
                format!("4 0x{fa:x},0x{fb:x},0x{fc:x},0x{fd:x} 1111 0"),
                "registration table, cross-shard calls and dispatch"
            );
            return;
        }
        last_error = String::from_utf8_lossy(&output.stderr).into_owned();
        if !last_error.contains("memory64") && !last_error.contains("Memory64") {
            break;
        }
    }
    if last_error.contains("memory64") || last_error.contains("Memory64") {
        return skip(format!(
            "node cannot run wasm64: {}",
            last_error.lines().next().unwrap_or("")
        ));
    }
    panic!("running the linked shards failed:\n{last_error}");
}

// ----- scale -------------------------------------------------------------------------------------

/// Thousands of tiny functions through the whole pipeline (sharding + parallel lifting).
#[test]
fn many_small_functions() {
    let tmp = TempDir::new("many");
    let count = 4000;
    let mut options = options_for(tmp.path(), &chain_fixture(count, true));
    options.shard_size = 256;
    options.jobs = 8;
    let report = run(&options).expect("build");
    assert_eq!(report.discovery.functions, count);
    assert_eq!(report.pdata.functions, count);
    assert_eq!(report.lift.shards.len(), count.div_ceil(256));
    assert_eq!(report.lift.unsupported_total(), 0);
    let table =
        fs::read_to_string(options.module_dir().join(LIFT_DIR).join("functions.ll")).unwrap();
    assert_eq!(
        table.matches("%Pg6LiftedFunctionEntry { i64 ").count(),
        count
    );
}

/// Benchmark on a large generated binary: `cargo test --release -p playgta6-recomp
/// bench_large_fixture -- --ignored --nocapture` (size via `PLAYGTA6_BENCH_FUNCTIONS`).
#[test]
#[ignore]
fn bench_large_fixture() {
    let count: usize = env::var("PLAYGTA6_BENCH_FUNCTIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200_000);
    let tmp = TempDir::new("bench");
    let pe = chain_fixture(count, true);
    let mut options = options_for(tmp.path(), &pe);
    options.jobs = thread::available_parallelism().map_or(1, |n| n.get());
    let start = Instant::now();
    let report = run(&options).expect("build");
    let elapsed = start.elapsed();
    let ir_bytes: u64 = report.lift.shards.iter().map(|s| s.ll.size).sum();
    println!(
        "bench: {count} functions, {} bytes PE, {} shards, {} MiB IR, {} jobs: {:.2?}",
        pe.len(),
        report.lift.shards.len(),
        ir_bytes >> 20,
        options.jobs,
        elapsed
    );
}
