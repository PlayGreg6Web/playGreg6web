//! Program build tests on synthetic PE32+ executables and DLLs written in-test (imports by name
//! and ordinal, exports with ordinals, forwarders and data, base relocations, TLS). Outputs go to
//! temp dirs. The execution test needs `llc`/`clang`, a wasm linker and `node` and skips with a
//! printed note when one is missing.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use super::link::{ExportIndex, ModuleExports, parse_forwarder};
use super::rebase::{Request, assign_bases, rebase_pe};
use super::*;
use crate::image;
use crate::lift::compile::{
    LlvmTool, compile_to_wasm64_object, find_llvm_tool, find_on_path, rustc_tool_dir,
};
use crate::lift::function_symbol;
use crate::pipeline::{LIFT_DIR, MANIFEST_FILE};

const TEXT: u32 = 0x1000;
const DATA: u32 = 0x2000;
const RDATA: u32 = 0x3000;
const SPAN: u32 = 0x1000;

// ----- synthetic PE writer -----------------------------------------------------------------------

fn put(buf: &mut [u8], off: usize, bytes: &[u8]) {
    buf[off..off + bytes.len()].copy_from_slice(bytes);
}

#[derive(Clone, Debug)]
enum Imp {
    Name(&'static str),
    Ord(u16),
}

#[derive(Clone, Debug)]
enum Target {
    Rva(u32),
    Forward(&'static str),
}

#[derive(Clone, Debug)]
struct Export {
    name: Option<&'static str>,
    ordinal: u32,
    target: Target,
}

fn export(name: &'static str, ordinal: u32, target: Target) -> Export {
    Export {
        name: Some(name),
        ordinal,
        target,
    }
}

#[derive(Clone, Debug)]
struct Tls {
    /// RVA and length of the template data (in `.data`).
    data: u32,
    len: u32,
    /// RVA of the index variable (in `.data`).
    index: u32,
    callbacks: Vec<u32>,
}

/// `.text` at 0x1000, `.data` at 0x2000, `.rdata` (imports, exports, TLS) at 0x3000, `.reloc`
/// at 0x4000 when there are relocations.
#[derive(Clone, Debug)]
struct Synth {
    name: &'static str,
    base: u64,
    dll: bool,
    relocs_stripped: bool,
    text: Vec<u8>,
    data: Vec<u8>,
    imports: Vec<(&'static str, Vec<Imp>)>,
    exports: Vec<Export>,
    entry: u32,
    /// DIR64 sites (RVAs); the TLS directory adds its own.
    relocs: Vec<u32>,
    tls: Option<Tls>,
}

struct RdataLayout {
    bytes: Vec<u8>,
    /// IAT slot RVAs per import module.
    slots: Vec<Vec<u32>>,
    export_dir: Option<(u32, u32)>,
    tls_dir: Option<u32>,
    /// DIR64 sites inside `.rdata` (TLS directory fields and callbacks).
    relocs: Vec<u32>,
}

impl Synth {
    fn new(name: &'static str, base: u64, dll: bool) -> Self {
        Self {
            name,
            base,
            dll,
            relocs_stripped: false,
            text: vec![0xC3],
            data: vec![0; 16],
            imports: Vec::new(),
            exports: Vec::new(),
            entry: 0,
            relocs: Vec::new(),
            tls: None,
        }
    }

    fn va(&self, rva: u32) -> u64 {
        self.base + u64::from(rva)
    }

    /// IAT slot VA of import `symbol` of module `module` (indices into `imports`).
    fn slot(&self, module: usize, symbol: usize) -> u64 {
        self.va(self.rdata().slots[module][symbol])
    }

    fn rdata(&self) -> RdataLayout {
        let mut buf = vec![0u8; SPAN as usize];
        let mut cur = ((self.imports.len() + 1) * 20).next_multiple_of(8);
        let mut slots = Vec::new();
        for (m, (dll, names)) in self.imports.iter().enumerate() {
            let ilt = cur;
            let iat = ilt + 8 * (names.len() + 1);
            cur = iat + 8 * (names.len() + 1);
            let name = cur;
            put(&mut buf, name, dll.as_bytes());
            cur = (cur + dll.len() + 1).next_multiple_of(2);
            let mut module_slots = Vec::new();
            for (i, symbol) in names.iter().enumerate() {
                let value = match symbol {
                    Imp::Name(s) => {
                        let hint_name = RDATA + cur as u32;
                        put(&mut buf, cur + 2, s.as_bytes());
                        cur = (cur + 2 + s.len() + 1).next_multiple_of(2);
                        u64::from(hint_name)
                    }
                    Imp::Ord(o) => (1u64 << 63) | u64::from(*o),
                };
                put(&mut buf, ilt + 8 * i, &value.to_le_bytes());
                put(&mut buf, iat + 8 * i, &value.to_le_bytes());
                module_slots.push(RDATA + (iat + 8 * i) as u32);
            }
            let desc = 20 * m;
            put(&mut buf, desc, &(RDATA + ilt as u32).to_le_bytes());
            put(&mut buf, desc + 12, &(RDATA + name as u32).to_le_bytes());
            put(&mut buf, desc + 16, &(RDATA + iat as u32).to_le_bytes());
            slots.push(module_slots);
            cur = cur.next_multiple_of(8);
        }

        let mut export_dir = None;
        if !self.exports.is_empty() {
            let dir = cur.next_multiple_of(8);
            let ordinal_base = self.exports.iter().map(|e| e.ordinal).min().unwrap();
            let count =
                (self.exports.iter().map(|e| e.ordinal).max().unwrap() - ordinal_base + 1) as usize;
            let mut named: Vec<&Export> =
                self.exports.iter().filter(|e| e.name.is_some()).collect();
            named.sort_by_key(|e| e.name);
            let eat = dir + 40;
            let names = eat + 4 * count;
            let ords = names + 4 * named.len();
            let mut strings = ords + 2 * named.len();
            let module_name = strings;
            put(&mut buf, strings, self.name.as_bytes());
            strings += self.name.len() + 1;
            for (i, e) in named.iter().enumerate() {
                put(
                    &mut buf,
                    names + 4 * i,
                    &(RDATA + strings as u32).to_le_bytes(),
                );
                put(
                    &mut buf,
                    ords + 2 * i,
                    &((e.ordinal - ordinal_base) as u16).to_le_bytes(),
                );
                let name = e.name.unwrap();
                put(&mut buf, strings, name.as_bytes());
                strings += name.len() + 1;
            }
            for e in &self.exports {
                let value = match &e.target {
                    Target::Rva(rva) => *rva,
                    Target::Forward(text) => {
                        let at = RDATA + strings as u32;
                        put(&mut buf, strings, text.as_bytes());
                        strings += text.len() + 1;
                        at
                    }
                };
                put(
                    &mut buf,
                    eat + 4 * (e.ordinal - ordinal_base) as usize,
                    &value.to_le_bytes(),
                );
            }
            put(
                &mut buf,
                dir + 12,
                &(RDATA + module_name as u32).to_le_bytes(),
            );
            put(&mut buf, dir + 16, &ordinal_base.to_le_bytes());
            put(&mut buf, dir + 20, &(count as u32).to_le_bytes());
            put(&mut buf, dir + 24, &(named.len() as u32).to_le_bytes());
            put(&mut buf, dir + 28, &(RDATA + eat as u32).to_le_bytes());
            put(&mut buf, dir + 32, &(RDATA + names as u32).to_le_bytes());
            put(&mut buf, dir + 36, &(RDATA + ords as u32).to_le_bytes());
            export_dir = Some((RDATA + dir as u32, (strings - dir) as u32));
            cur = strings;
        }

        let mut tls_dir = None;
        let mut relocs = Vec::new();
        if let Some(tls) = &self.tls {
            let dir = cur.next_multiple_of(8);
            let callbacks = dir + 40;
            let fields = [
                self.va(tls.data),
                self.va(tls.data + tls.len),
                self.va(tls.index),
                self.va(RDATA + callbacks as u32),
            ];
            for (i, value) in fields.iter().enumerate() {
                put(&mut buf, dir + 8 * i, &value.to_le_bytes());
                relocs.push(RDATA + (dir + 8 * i) as u32);
            }
            for (i, &cb) in tls.callbacks.iter().enumerate() {
                put(&mut buf, callbacks + 8 * i, &self.va(cb).to_le_bytes());
                relocs.push(RDATA + (callbacks + 8 * i) as u32);
            }
            tls_dir = Some(RDATA + dir as u32);
            cur = callbacks + 8 * (tls.callbacks.len() + 1);
        }
        assert!(cur <= SPAN as usize, ".rdata overflow");
        RdataLayout {
            bytes: buf,
            slots,
            export_dir,
            tls_dir,
            relocs,
        }
    }

    fn build(&self) -> Vec<u8> {
        let rdata = self.rdata();
        let mut relocs: Vec<u32> = self.relocs.iter().chain(&rdata.relocs).copied().collect();
        relocs.sort_unstable();
        let mut sections: Vec<(&str, u32, Vec<u8>, u32)> = vec![
            (".text", TEXT, self.text.clone(), 0x6000_0020),
            (".data", DATA, self.data.clone(), 0xC000_0040),
            (".rdata", RDATA, rdata.bytes.clone(), 0x4000_0040),
        ];
        let mut reloc_dir = None;
        if !relocs.is_empty() {
            let mut by_page: BTreeMap<u32, Vec<u16>> = BTreeMap::new();
            for &r in &relocs {
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
            reloc_dir = Some((0x4000, raw.len() as u32));
            sections.push((".reloc", 0x4000, raw, 0x4200_0040));
        }
        let size_of_image = 0x5000u32;

        let e_lfanew = 0x80usize;
        let opt = e_lfanew + 24;
        let table = opt + 240;
        let mut file_len = 0x400usize;
        let mut raw_offsets = Vec::new();
        for (_, _, raw, _) in &sections {
            raw_offsets.push(file_len);
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
        let mut characteristics = 0x0022u16;
        if self.dll {
            characteristics |= 0x2000;
        }
        if self.relocs_stripped {
            characteristics |= 0x0001;
        }
        put(&mut buf, e_lfanew + 22, &characteristics.to_le_bytes());
        put(&mut buf, opt, &0x020Bu16.to_le_bytes());
        put(&mut buf, opt + 16, &self.entry.to_le_bytes());
        put(&mut buf, opt + 20, &TEXT.to_le_bytes());
        put(&mut buf, opt + 24, &self.base.to_le_bytes());
        put(&mut buf, opt + 32, &SPAN.to_le_bytes());
        put(&mut buf, opt + 36, &0x200u32.to_le_bytes());
        put(&mut buf, opt + 56, &size_of_image.to_le_bytes());
        put(&mut buf, opt + 60, &0x400u32.to_le_bytes());
        put(&mut buf, opt + 68, &3u16.to_le_bytes());
        put(&mut buf, opt + 108, &16u32.to_le_bytes());
        let dir = |buf: &mut Vec<u8>, index: usize, (rva, size): (u32, u32)| {
            put(buf, opt + 112 + index * 8, &rva.to_le_bytes());
            put(buf, opt + 116 + index * 8, &size.to_le_bytes());
        };
        if let Some(d) = rdata.export_dir {
            dir(&mut buf, 0, d);
        }
        if !self.imports.is_empty() {
            dir(&mut buf, 1, (RDATA, 20 * (self.imports.len() as u32 + 1)));
        }
        if let Some(d) = reloc_dir {
            dir(&mut buf, 5, d);
        }
        if let Some(t) = rdata.tls_dir {
            dir(&mut buf, 9, (t, 40));
        }
        for (i, ((name, rva, raw, chars), &off)) in sections.iter().zip(&raw_offsets).enumerate() {
            let h = table + i * 40;
            put(&mut buf, h, name.as_bytes());
            put(&mut buf, h + 8, &(raw.len().max(1) as u32).to_le_bytes());
            put(&mut buf, h + 12, &rva.to_le_bytes());
            put(
                &mut buf,
                h + 16,
                &(raw.len().next_multiple_of(0x200) as u32).to_le_bytes(),
            );
            put(&mut buf, h + 20, &(off as u32).to_le_bytes());
            put(&mut buf, h + 36, &chars.to_le_bytes());
            put(&mut buf, off, raw);
        }
        buf
    }
}

/// Tiny x86-64 emitter.
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

    fn rel32(&mut self, opcode: &[u8], target: u64) -> &mut Self {
        let end = self.here() + opcode.len() as u64 + 4;
        let rel = i32::try_from(target.wrapping_sub(end) as i64).expect("rel32");
        self.raw(opcode).raw(&rel.to_le_bytes())
    }

    fn call_rip(&mut self, slot: u64) -> &mut Self {
        self.rel32(&[0xFF, 0x15], slot)
    }

    fn mov_rcx_mem(&mut self, target: u64) -> &mut Self {
        self.rel32(&[0x48, 0x8B, 0x0D], target)
    }

    fn mov_eax(&mut self, imm: u32) -> &mut Self {
        self.raw(&[0xB8]).raw(&imm.to_le_bytes())
    }

    /// `movabs rcx, imm64`; returns the RVA offset of the immediate within the emitter.
    fn movabs_rcx(&mut self, imm: u64) -> u32 {
        self.raw(&[0x48, 0xB9]);
        let at = self.bytes.len() as u32;
        self.raw(&imm.to_le_bytes());
        at
    }

    fn sub_rsp_28(&mut self) -> &mut Self {
        self.raw(&[0x48, 0x83, 0xEC, 0x28])
    }

    fn add_rsp_28(&mut self) -> &mut Self {
        self.raw(&[0x48, 0x83, 0xC4, 0x28])
    }

    fn ret(&mut self) -> &mut Self {
        self.raw(&[0xC3])
    }

    fn pad_to(&mut self, offset: u64) -> &mut Self {
        assert!(self.here() <= self.base + offset, "code overflows its slot");
        while self.here() < self.base + offset {
            self.bytes.push(0xCC);
        }
        self
    }
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
            "playgta6_program_{label}_{}_{nanos}",
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

const MINI_SHIMS: &str = "\
extern \"C\" uint64_t pg6_win32_virtual_alloc(uint64_t a, uint64_t s, uint32_t t, uint32_t p);
extern \"C\" void pg6_win32_acquire_srw_lock_exclusive(void* lock);
";

/// Writes the PEs (`name` -> bytes) into `dir/in/` and returns options building `exe` with all
/// other files supplied as `--dll`s, output in `dir/out`.
fn program_options(dir: &Path, exe: &Synth, dlls: &[&Synth]) -> ProgramOptions {
    let input = dir.join("in");
    fs::create_dir_all(input.join("rt").join("os")).unwrap();
    let header = input.join("rt").join("os").join("mini.h");
    fs::write(&header, MINI_SHIMS).unwrap();
    let exe_path = input.join(exe.name);
    fs::write(&exe_path, exe.build()).unwrap();
    let mut options = ProgramOptions::new(&exe_path, dir.join("out"));
    for dll in dlls {
        let path = input.join(dll.name);
        fs::write(&path, dll.build()).unwrap();
        options.dlls.push(path);
    }
    options.build.shim_headers = vec![header];
    options.build.jobs = 2;
    options
}

fn build(options: &ProgramOptions) -> ProgramReport {
    build_program(options, &mut |_, _| {}).unwrap_or_else(|e| panic!("program build failed: {e}"))
}

fn read_tree(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
        for entry in fs::read_dir(dir).expect("list dir") {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                walk(root, &path, out);
            } else {
                let rel = path
                    .strip_prefix(root)
                    .unwrap()
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
                    .join("/");
                out.insert(rel, fs::read(&path).unwrap());
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(dir, dir, &mut out);
    out
}

fn import<'a>(r: &'a ProgramReport, importer: &str, label: &str) -> &'a ImportRecord {
    r.imports
        .iter()
        .find(|i| i.importer == importer && i.label() == label)
        .unwrap_or_else(|| panic!("no import {importer}: {label}"))
}

fn guest_va(record: &ImportRecord) -> u64 {
    match &record.binding {
        Binding::Guest { va, .. } => *va,
        other => panic!("{} not guest-bound: {other:?}", record.label()),
    }
}

fn has_diag(r: &ProgramReport, kind: &str) -> bool {
    r.diagnostics.iter().any(|d| d.kind == kind)
}

/// The guest image of `module` in `out`, flat at its base.
fn image_bytes(out: &Path, module: &str) -> (u64, Vec<u8>) {
    let pgim = fs::read(out.join(module).join("image").join("guest.pgim")).unwrap();
    let image = image::read_image(&pgim).unwrap();
    (image.image_base, image.materialize())
}

fn read_u64(image: &(u64, Vec<u8>), va: u64) -> u64 {
    let off = (va - image.0) as usize;
    u64::from_le_bytes(image.1[off..off + 8].try_into().unwrap())
}

// ----- fixture: exe -> a.dll -> b.dll --------------------------------------------------------------

const EXE_BASE: u64 = 0x40_0000;
const DLL_BASE: u64 = 0x60_0000;

/// `b.dll`: `fb` returns 1, `bdata` (data export) holds 1000, `#3` ordinal-only function.
fn dll_b(base: u64) -> Synth {
    let mut b = Synth::new("b.dll", base, true);
    let mut a = Asm::new(base + u64::from(TEXT));
    a.mov_eax(1).ret().pad_to(0x10);
    a.mov_eax(3).ret();
    b.text = a.bytes;
    b.data = 1000u64.to_le_bytes().to_vec();
    b.exports = vec![
        export("fb", 1, Target::Rva(TEXT)),
        export("bdata", 2, Target::Rva(DATA)),
        Export {
            name: None,
            ordinal: 3,
            target: Target::Rva(TEXT + 0x10),
        },
    ];
    b
}

/// `a.dll`: `fa` = `b!fb() + [own .data]` (10, via an absolute, relocated address), `fwd`
/// forwards to `b.fb`, `fwd3` to `b.#3`; imports `b.dll!fb`.
fn dll_a(base: u64) -> Synth {
    let mut s = Synth::new("a.dll", base, true);
    s.imports = vec![("b.dll", vec![Imp::Name("fb")])];
    let slot = s.slot(0, 0);
    let mut a = Asm::new(base + u64::from(TEXT));
    a.sub_rsp_28().call_rip(slot);
    let imm = a.movabs_rcx(base + u64::from(DATA));
    a.raw(&[0x48, 0x03, 0x01]) // add rax, [rcx]
        .add_rsp_28()
        .ret();
    s.relocs.push(TEXT + imm);
    s.text = a.bytes;
    s.data = 10u64.to_le_bytes().to_vec();
    s.exports = vec![
        export("fa", 1, Target::Rva(TEXT)),
        export("fwd", 2, Target::Forward("b.fb")),
        export("fwd3", 3, Target::Forward("B.#3")),
    ];
    s
}

/// The executable: `main` = `a!#1() + a!fwd() + *b!bdata` = 11 + 1 + 1000.
fn exe_main() -> Synth {
    let mut s = Synth::new("game.exe", EXE_BASE, false);
    s.imports = vec![
        (
            "a.dll",
            vec![Imp::Ord(1), Imp::Name("fwd"), Imp::Name("fwd3")],
        ),
        ("B.DLL", vec![Imp::Name("bdata")]),
        ("KERNEL32.dll", vec![Imp::Name("VirtualAlloc")]),
        ("nosuch.dll", vec![Imp::Name("Missing")]),
    ];
    let (fa, fwd, bdata) = (s.slot(0, 0), s.slot(0, 1), s.slot(1, 0));
    let mut a = Asm::new(EXE_BASE + u64::from(TEXT));
    a.sub_rsp_28()
        .call_rip(fa)
        .raw(&[0x48, 0x89, 0xC3]) // mov rbx, rax
        .call_rip(fwd)
        .raw(&[0x48, 0x01, 0xD8]) // add rax, rbx
        .mov_rcx_mem(bdata)
        .raw(&[0x48, 0x03, 0x01]) // add rax, [rcx]
        .add_rsp_28()
        .ret();
    s.text = a.bytes;
    s.entry = TEXT;
    s
}

// ----- tests -------------------------------------------------------------------------------------

#[test]
fn names_api_sets_and_forwarders() {
    assert_eq!(canonical_dll("C:\\x\\MSVCP140.DLL"), "msvcp140.dll");
    assert_eq!(canonical_dll("vcruntime140_1"), "vcruntime140_1.dll");
    assert_eq!(canonical_dll("game.exe"), "game.exe");
    assert_eq!(canonical_dll("noext."), "noext");
    assert_eq!(module_symbol("libhttpclient.gdk.dll"), "libhttpclient_gdk");
    assert_eq!(module_symbol("vcruntime140_1.dll"), "vcruntime140_1");
    assert_eq!(
        api_set_host("api-ms-win-crt-heap-l1-1-0.dll"),
        Some("ucrtbase.dll")
    );
    assert_eq!(
        api_set_host("api-ms-win-core-com-l1-1-0.dll"),
        Some("combase.dll")
    );
    assert_eq!(
        api_set_host("api-ms-win-core-synch-l1-2-0.dll"),
        Some("kernel32.dll")
    );
    assert_eq!(
        api_set_host("ext-ms-win-rtcore-ntuser-window-l1-1-0.dll"),
        None
    );
    assert_eq!(api_set_host("kernel32.dll"), None);
    assert_eq!(
        parse_forwarder("api-ms-win-core-synch-l1-2-0.AcquireSRWLockExclusive"),
        Some((
            "api-ms-win-core-synch-l1-2-0".to_string(),
            ImportedSymbol::ByName {
                hint: 0,
                name: "AcquireSRWLockExclusive".to_string()
            }
        ))
    );
    assert_eq!(
        parse_forwarder("B.#3"),
        Some(("B".to_string(), ImportedSymbol::ByOrdinal { ordinal: 3 }))
    );
    assert_eq!(parse_forwarder("nodot"), None);
    assert_eq!(parse_forwarder("X.#bad"), None);
    assert!(policy::glob("vcruntime140*.dll", "vcruntime140_1.dll"));
    assert!(policy::glob("??2@*", "??2@YAPEAX_K@Z"));
    assert!(!policy::glob("__CxxFrameHandler*", "_CxxThrowException"));
    assert!(policy::glob("*", ""));
    assert!(host_known("KERNEL32.dll"));
    assert!(host_known("api-ms-win-crt-stdio-l1-1-0.dll"));
    assert!(host_known("xmem"));
    assert!(!host_known("nosuch.dll"));
    assert!(!host_known("msvcp140.dll"));
}

#[test]
fn api_set_rules_match_the_runtime_loader() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../runtime/os/win32_loader.cpp");
    let source = fs::read_to_string(&path).expect("read win32_loader.cpp");
    let ctor = source
        .split("ModuleRegistry::ModuleRegistry()")
        .nth(1)
        .expect("registry constructor");
    let rules: Vec<(String, String)> = ctor
        .split("add_api_set_rule_locked(\"")
        .skip(1)
        .take_while(|_| true)
        .filter_map(|rest| {
            let (prefix, rest) = rest.split_once('"')?;
            let host = rest.split('"').nth(1)?;
            Some((prefix.to_string(), host.to_string()))
        })
        .collect();
    let expected: Vec<(String, String)> = API_SET_RULES
        .iter()
        .map(|(p, h)| (p.to_string(), h.to_string()))
        .collect();
    assert_eq!(
        rules, expected,
        "keep API_SET_RULES in sync with win32_loader.cpp"
    );
}

#[test]
fn host_modules_cover_the_runtime_registrations() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../runtime");
    let mut missing = Vec::new();
    for dir in ["os", "gpu"] {
        let Ok(entries) = fs::read_dir(root.join(dir)) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_none_or(|e| e != "cpp") {
                continue;
            }
            let text = fs::read_to_string(&path).unwrap();
            for literal in text.split('"').skip(1).step_by(2) {
                let lower = literal.to_ascii_lowercase();
                if lower.ends_with(".dll")
                    && lower.len() > 4
                    && !lower.contains(['\\', '/', ' ', '%'])
                    && !host_known(&lower)
                {
                    missing.push(format!("{} ({})", lower, path.display()));
                }
            }
        }
    }
    assert!(
        missing.is_empty(),
        "runtime modules not in HOST_MODULES: {missing:?}"
    );
}

#[test]
fn assign_bases_keeps_free_bases_and_first_fits_collisions() {
    let req = |name: &str, base: u64, size: u32, relocatable: bool| Request {
        name: name.to_string(),
        preferred_base: base,
        size_of_image: size,
        relocatable,
    };
    let bases = assign_bases(&[
        req("exe", 0x1_4000_0000, 0x5000, false),
        req("a", 0x1_8000_0000, 0x12000, true),
        req("b", 0x1_8000_0000, 0x5000, true),
        req("c", 0x1_8000_0000, 0x5000, true),
        req("d", 0x1_4000_0000, 0x1000, true),
        req("e", 0x2_0000_0000, 0x1000, true),
    ])
    .unwrap();
    assert_eq!(
        bases,
        vec![
            0x1_4000_0000,
            0x1_8000_0000,
            0x1_8002_0000,
            0x1_8003_0000,
            0x1_4001_0000,
            0x2_0000_0000
        ]
    );
    let err = assign_bases(&[
        req("exe", 0x1_8000_0000, 0x5000, true),
        req("fixed", 0x1_8000_0000, 0x5000, false),
    ])
    .unwrap_err();
    assert!(err.contains("relocations stripped"), "{err}");
    let err = assign_bases(&[
        req("exe", ADDRESS_LIMIT - 0x10000, 0x10000, false),
        req("x", ADDRESS_LIMIT - 0x10000, 0x10000, true),
    ])
    .unwrap_err();
    assert!(err.contains("no free range"), "{err}");
}

#[test]
fn rebasing_at_build_time_equals_building_at_the_new_base() {
    let at_a = dll_a(DLL_BASE);
    let new_base = DLL_BASE + 0x7_0000;
    let at_b = dll_a(new_base);
    let file = at_a.build();
    let pe = Pe64::parse(&file).unwrap();
    let rebased = rebase_pe(&file, &pe, new_base).unwrap();
    assert_eq!(
        rebased,
        at_b.build(),
        "rebased bytes equal the direct build"
    );
    assert_eq!(rebase_pe(&file, &pe, DLL_BASE).unwrap(), file);

    // TLS directory and callbacks are relocated too.
    let mut tls = dll_b(DLL_BASE);
    tls.tls = Some(Tls {
        data: DATA + 8,
        len: 8,
        index: DATA + 0x10,
        callbacks: vec![TEXT + 0x10],
    });
    let file = tls.build();
    let pe = Pe64::parse(&file).unwrap();
    tls.base = new_base;
    assert_eq!(rebase_pe(&file, &pe, new_base).unwrap(), tls.build());

    // The whole module build: A colliding with B (placed first) is rebased; its outputs equal
    // those of a program where A already sits at that base.
    let t1 = TempDir::new("rebase1");
    let exe = exe_main();
    let r1 = build(&program_options(
        t1.path(),
        &exe,
        &[&dll_a(DLL_BASE), &dll_b(DLL_BASE)],
    ));
    let a1 = r1.module("a.dll").unwrap();
    assert_eq!(a1.preferred_base, DLL_BASE);
    assert_eq!(a1.base, DLL_BASE + 0x1_0000, "first fit above b.dll");
    assert_eq!(r1.module("b.dll").unwrap().base, DLL_BASE);
    let t2 = TempDir::new("rebase2");
    let r2 = build(&program_options(
        t2.path(),
        &exe,
        &[&dll_a(DLL_BASE + 0x1_0000), &dll_b(DLL_BASE)],
    ));
    assert_eq!(r2.module("a.dll").unwrap().preferred_base, a1.base);
    for module in ["a", "b", "main"] {
        assert_eq!(
            read_tree(&t1.path().join("out").join(module)),
            read_tree(&t2.path().join("out").join(module)),
            "module {module}"
        );
    }
}

#[test]
fn relocs_stripped_collision_is_a_clean_error() {
    let tmp = TempDir::new("stripped");
    let exe = exe_main();
    let mut b = dll_b(DLL_BASE);
    let mut a = dll_a(DLL_BASE);
    a.relocs_stripped = true;
    b.relocs_stripped = false;
    let err = build_program(
        &program_options(tmp.path(), &exe, &[&a, &b]),
        &mut |_, _| {},
    )
    .unwrap_err();
    assert!(
        matches!(&err, ProgramError::Layout(m) if m.contains("a.dll") && m.contains("stripped")),
        "{err}"
    );
}

#[test]
fn two_dll_program_binds_ordinals_forwarders_and_data() {
    let tmp = TempDir::new("bind");
    let (exe, a, b) = (exe_main(), dll_a(DLL_BASE), dll_b(DLL_BASE));
    let options = program_options(tmp.path(), &exe, &[&a, &b]);
    let r = build(&options);
    let out = tmp.path().join("out");

    // Order and layout.
    let order: Vec<&str> = r.modules.iter().map(|m| m.dll.as_str()).collect();
    assert_eq!(order, ["b.dll", "a.dll", "game.exe"]);
    let a_base = r.module("a.dll").unwrap().base;
    assert_eq!(a_base, DLL_BASE + 0x1_0000);
    assert!(r.cycles.is_empty());

    // Bindings of the executable.
    let fa = import(&r, "game.exe", "a.dll!#1");
    assert_eq!(guest_va(fa), a_base + u64::from(TEXT));
    let fwd = import(&r, "game.exe", "a.dll!fwd");
    assert_eq!(guest_va(fwd), DLL_BASE + u64::from(TEXT));
    assert_eq!(fwd.via, ["b.fb"]);
    let fwd3 = import(&r, "game.exe", "a.dll!fwd3");
    assert_eq!(guest_va(fwd3), DLL_BASE + u64::from(TEXT) + 0x10);
    let bdata = import(&r, "game.exe", "B.DLL!bdata");
    assert_eq!(guest_va(bdata), DLL_BASE + u64::from(DATA));
    assert!(matches!(
        bdata.binding,
        Binding::Guest {
            kind: ExportKind::Data,
            ..
        }
    ));
    assert_eq!(
        import(&r, "game.exe", "KERNEL32.dll!VirtualAlloc").status(),
        "host"
    );
    assert_eq!(
        import(&r, "game.exe", "nosuch.dll!Missing").status(),
        "unresolved"
    );
    let fb = import(&r, "a.dll", "b.dll!fb");
    assert_eq!(guest_va(fb), DLL_BASE + u64::from(TEXT));

    // External DLLs.
    let status = |dll: &str| {
        r.external
            .iter()
            .find(|e| e.dll == dll)
            .map(|e| e.status)
            .unwrap()
    };
    assert_eq!(status("kernel32.dll"), "host");
    assert_eq!(status("nosuch.dll"), "missing");

    // Pre-filled IAT slots in the packed image; the rest keep their import table value.
    let exe_image = image_bytes(&out, "main");
    assert_eq!(read_u64(&exe_image, exe.slot(0, 0)), guest_va(fa));
    assert_eq!(read_u64(&exe_image, exe.slot(0, 1)), guest_va(fwd));
    assert_eq!(read_u64(&exe_image, exe.slot(1, 0)), guest_va(bdata));
    let kernel32 = read_u64(&exe_image, exe.slot(2, 0));
    assert!(
        (u64::from(RDATA)..u64::from(RDATA + SPAN)).contains(&kernel32),
        "host-bound slot keeps its hint/name RVA until load: {kernel32:#x}"
    );
    let a_image = image_bytes(&out, "a");
    assert_eq!(
        a_image.0, a_base,
        "a.dll's image is emitted at its new base"
    );
    assert_eq!(
        read_u64(&a_image, a_base + (dll_a(a_base).slot(0, 0) - a_base)),
        guest_va(fb)
    );
    let main = r.module("game.exe").unwrap();
    assert_eq!(main.iat_prefill.len(), 4);
    assert_eq!(main.imports.guest, 4);
    assert_eq!(main.imports.host_bound, 1);
    assert_eq!(main.imports.host_unresolved, 1);
    assert_eq!(main.imports.before_bound, 1);
    assert_eq!(main.imports.before_unresolved, 5);

    // Namespaced import bridges without the guest-bound slots.
    let stubs_src = fs::read_to_string(out.join("main/stubs/pg6_import_thunks.cpp")).unwrap();
    assert!(stubs_src.contains("const Pg6ImportThunk* const pg6_import_thunks_main = kThunks;"));
    assert!(stubs_src.contains("const std::size_t pg6_import_thunk_count_main = 2;"));
    assert!(!stubs_src.contains(&format!("0x{:08x}u", exe.slot(0, 0) - EXE_BASE)));
    let a_src = fs::read_to_string(out.join("a/stubs/pg6_import_thunks.cpp")).unwrap();
    assert!(a_src.contains("pg6_import_thunks_a = nullptr;"));
    let header = fs::read_to_string(out.join("a/stubs/pg6_import_thunks.h")).unwrap();
    assert!(header.contains("#ifndef PG6_IMPORT_THUNK_TYPES_DEFINED"));
    assert!(header.contains("pg6_find_import_thunk_a(uint32_t slot_rva);"));
    let bridge = fs::read_to_string(out.join("main/stubs/pg6_import_manifest.json")).unwrap();
    assert!(bridge.contains("\"namespace\": \"main\""));
    assert!(bridge.contains("\"guest\": 4}"));
    assert!(bridge.contains("\"status\": \"guest\", \"module\": \"a\""));
    let build_manifest = fs::read_to_string(out.join("main").join(MANIFEST_FILE)).unwrap();
    assert!(build_manifest.contains("\"guest\": 4"));

    // Registration tables per module, disjoint.
    let a_table = fs::read_to_string(out.join("a").join(LIFT_DIR).join("functions.ll")).unwrap();
    assert!(a_table.contains("@__playgta6_lifted_functions_a = constant"));
    let ranges: Vec<(u64, u64)> = r
        .modules
        .iter()
        .map(|m| (m.first_function, m.last_function))
        .collect();
    for (i, x) in ranges.iter().enumerate() {
        for y in &ranges[i + 1..] {
            assert!(x.1 < y.0 || y.1 < x.0, "{ranges:x?}");
        }
    }

    // Exports summary and the manifest/descriptor.
    let bm = r.module("b.dll").unwrap();
    assert_eq!(
        (
            bm.exports.functions,
            bm.exports.data,
            bm.exports.ordinal_only
        ),
        (2, 1, 1)
    );
    let am = r.module("a.dll").unwrap();
    assert_eq!((am.exports.functions, am.exports.forwarders), (1, 2));
    assert!(
        r.manifest
            .contains("\"schema\": \"playgta6-recomp-program\"")
    );
    assert!(r.manifest.contains(&format!("\"base\": \"0x{a_base:x}\"")));
    assert!(r.manifest.contains("\"via\": [\"b.fb\"]"));
    let descriptor = fs::read_to_string(out.join(PROGRAM_DESCRIPTOR_FILE)).unwrap();
    assert!(descriptor.contains("#include \"guest/program_loader.h\""));
    assert!(descriptor.contains(&format!(
        "{{\"a.dll\", \"a\", 0x{a_base:x}ull, 0x5000ull, 0x0ull, kPg6ProgramModuleDll, 0u, \
         __playgta6_lifted_functions_a, &__playgta6_lifted_function_count_a,\n     \
         reinterpret_cast<const Pg6BootImport* const*>(&pg6_import_thunks_a), \
         &pg6_import_thunk_count_a}},"
    )));
    assert!(descriptor.contains("#include \"a/stubs/pg6_import_thunks.h\""));
    assert!(descriptor.contains("kPg6ProgramModuleMain"));
    assert!(
        fs::read_to_string(out.join(PROGRAM_REPORT_FILE))
            .unwrap()
            .contains("a.dll rebased 0x600000 -> 0x610000")
    );
}

#[test]
fn outputs_are_independent_of_dll_order_and_jobs() {
    let (exe, a, b) = (exe_main(), dll_a(DLL_BASE), dll_b(DLL_BASE));
    let t1 = TempDir::new("det1");
    build(&program_options(t1.path(), &exe, &[&a, &b]));
    let t2 = TempDir::new("det2");
    let mut options = program_options(t2.path(), &exe, &[&b, &a]);
    options.build.jobs = 1;
    options.build.shard_size = 1;
    build(&options);
    let t3 = TempDir::new("det3");
    let mut options = program_options(t3.path(), &exe, &[]);
    options.dll_dirs.push(t1.path().join("in"));
    options.build.jobs = 7;
    options.build.shard_size = 1;
    build(&options);
    // shard_size changes the shard files, so compare like with like: runs 2 and 3.
    assert_eq!(
        read_tree(&t2.path().join("out")),
        read_tree(&t3.path().join("out"))
    );
    let again = TempDir::new("det4");
    build(&program_options(again.path(), &exe, &[&b, &a]));
    assert_eq!(
        read_tree(&t1.path().join("out")),
        read_tree(&again.path().join("out"))
    );
}

#[test]
fn import_cycles_between_dlls_are_ordered_and_reported() {
    let tmp = TempDir::new("cycle");
    let mut c = Synth::new("c.dll", DLL_BASE, true);
    c.imports = vec![("d.dll", vec![Imp::Name("fd")])];
    c.exports = vec![export("fc", 1, Target::Rva(TEXT))];
    let mut d = Synth::new("d.dll", DLL_BASE + 0x10_0000, true);
    d.imports = vec![("c.dll", vec![Imp::Name("fc")])];
    d.exports = vec![export("fd", 1, Target::Rva(TEXT))];
    let mut exe = Synth::new("game.exe", EXE_BASE, false);
    exe.imports = vec![("c.dll", vec![Imp::Name("fc")])];
    let r = build(&program_options(tmp.path(), &exe, &[&c, &d]));
    let order: Vec<&str> = r.modules.iter().map(|m| m.dll.as_str()).collect();
    assert_eq!(order, ["d.dll", "c.dll", "game.exe"]);
    assert_eq!(r.cycles, vec![("d.dll".to_string(), "c.dll".to_string())]);
    assert_eq!(
        guest_va(import(&r, "d.dll", "c.dll!fc")),
        DLL_BASE + u64::from(TEXT)
    );
    assert_eq!(
        guest_va(import(&r, "c.dll", "d.dll!fd")),
        DLL_BASE + 0x10_0000 + u64::from(TEXT)
    );
    assert!(
        r.manifest
            .contains("\"cycles\": [\n    {\"from\": \"d.dll\", \"to\": \"c.dll\"}")
    );
}

#[test]
fn forwarder_chains_cycles_and_api_set_targets() {
    let tmp = TempDir::new("forward");
    // e!f1 -> f.f2 -> g.f3 (a chain over three DLLs); e!lock -> an API set (host);
    // e!loop1 <-> f!loop2 (a cycle); e!gone -> f.nothing (missing export).
    let mut e = Synth::new("e.dll", DLL_BASE, true);
    e.exports = vec![
        export("f1", 1, Target::Forward("f.f2")),
        export(
            "lock",
            2,
            Target::Forward("api-ms-win-core-synch-l1-2-0.AcquireSRWLockExclusive"),
        ),
        export("loop1", 3, Target::Forward("f.loop2")),
        export("gone", 4, Target::Forward("f.nothing")),
    ];
    let mut f = Synth::new("f.dll", DLL_BASE, true);
    f.exports = vec![
        export("f2", 1, Target::Forward("g.f3")),
        export("loop2", 2, Target::Forward("e.loop1")),
    ];
    let mut g = Synth::new("g.dll", DLL_BASE, true);
    g.exports = vec![export("f3", 1, Target::Rva(TEXT))];
    let mut exe = Synth::new("game.exe", EXE_BASE, false);
    exe.imports = vec![(
        "e.dll",
        vec![
            Imp::Name("f1"),
            Imp::Name("lock"),
            Imp::Name("loop1"),
            Imp::Name("gone"),
        ],
    )];
    let r = build(&program_options(tmp.path(), &exe, &[&e, &f, &g]));
    // Forwarder targets become modules (reason "forwarder").
    assert_eq!(r.module("f.dll").unwrap().reason, "forwarder");
    assert_eq!(r.module("g.dll").unwrap().reason, "forwarder");
    let g_base = r.module("g.dll").unwrap().base;
    let f1 = import(&r, "game.exe", "e.dll!f1");
    assert_eq!(guest_va(f1), g_base + u64::from(TEXT));
    assert_eq!(f1.via, ["f.f2", "g.f3"]);
    let lock = import(&r, "game.exe", "e.dll!lock");
    match &lock.binding {
        Binding::Host {
            dll,
            symbol,
            resolution,
        } => {
            assert_eq!(dll, "api-ms-win-core-synch-l1-2-0");
            assert_eq!(symbol_text(symbol), "AcquireSRWLockExclusive");
            assert_eq!(
                resolution,
                &Resolution::Bound {
                    shim: "pg6_win32_acquire_srw_lock_exclusive".into()
                }
            );
        }
        other => panic!("{other:?}"),
    }
    assert!(has_diag(&r, "host_redirect"));
    assert_eq!(import(&r, "game.exe", "e.dll!loop1").status(), "unresolved");
    assert!(has_diag(&r, "forwarder_cycle"));
    assert_eq!(import(&r, "game.exe", "e.dll!gone").status(), "unresolved");
    assert!(has_diag(&r, "missing_export"));
    // The host redirect reaches the bridge: the executable's thunk calls the shim.
    let src = fs::read_to_string(tmp.path().join("out/main/stubs/pg6_import_thunks.cpp")).unwrap();
    assert!(src.contains("e.dll!lock -> pg6_win32_acquire_srw_lock_exclusive"));
}

#[test]
fn policy_precedence_roles_and_overrides() {
    // A guest "kernel32.dll" exporting functions the bridge also binds.
    let mut k = Synth::new("KERNEL32.dll", DLL_BASE, true);
    k.text = vec![0xC3; 0x20];
    k.data = vec![0; 16];
    k.exports = vec![
        export("VirtualAlloc", 1, Target::Rva(TEXT)),
        export("GetTickCount64", 2, Target::Rva(TEXT + 0x10)),
        export("Counter", 3, Target::Rva(DATA)),
    ];
    let mut x = Synth::new("XCurl.dll", DLL_BASE + 0x10_0000, true);
    x.exports = vec![export("curl_easy_init", 1, Target::Rva(TEXT))];
    let mut exe = Synth::new("game.exe", EXE_BASE, false);
    exe.imports = vec![
        (
            "kernel32",
            vec![
                Imp::Name("VirtualAlloc"),
                Imp::Name("GetTickCount64"),
                Imp::Name("Counter"),
            ],
        ),
        ("XCurl.dll", vec![Imp::Name("curl_easy_init")]),
    ];
    let run = |label: &str, overrides: &[&str]| {
        let tmp = TempDir::new(label);
        let mut options = program_options(tmp.path(), &exe, &[&k, &x]);
        options
            .overrides
            .add_file(&overrides.join("\n"))
            .expect("policy");
        (build(&options), tmp)
    };

    // Defaults: *!*=guest; XCurl is host-only.
    let (r, _t) = run("pol_default", &[]);
    assert_eq!(
        import(&r, "game.exe", "kernel32!VirtualAlloc").status(),
        "guest"
    );
    assert_eq!(
        import(&r, "game.exe", "kernel32!VirtualAlloc").rule,
        "*!*=guest"
    );
    assert!(r.module("xcurl.dll").is_none());
    assert_eq!(
        import(&r, "game.exe", "XCurl.dll!curl_easy_init").status(),
        "unresolved"
    );
    assert_eq!(
        r.external
            .iter()
            .find(|e| e.dll == "xcurl.dll")
            .unwrap()
            .status,
        "host_only"
    );

    // prefer-host: host where the bridge binds a shim, guest otherwise; data stays guest.
    let (r, _t) = run(
        "pol_prefer",
        &["kernel32.dll!*=prefer-host", "xcurl.dll=guest"],
    );
    assert_eq!(
        import(&r, "game.exe", "kernel32!VirtualAlloc").status(),
        "host"
    );
    assert_eq!(
        import(&r, "game.exe", "kernel32!GetTickCount64").status(),
        "guest"
    );
    assert_eq!(import(&r, "game.exe", "kernel32!Counter").status(), "guest");
    assert_eq!(
        import(&r, "game.exe", "kernel32!VirtualAlloc").rule,
        "kernel32.dll!*=prefer-host (override)"
    );
    assert_eq!(r.module("xcurl.dll").unwrap().reason, "import");
    assert_eq!(
        import(&r, "game.exe", "XCurl.dll!curl_easy_init").status(),
        "guest"
    );

    // The first matching rule wins; host on data is forced back to guest.
    let (r, _t) = run(
        "pol_first",
        &["kernel32.dll!GetTickCount64=guest", "kernel32.dll!*=host"],
    );
    assert_eq!(
        import(&r, "game.exe", "kernel32!GetTickCount64").status(),
        "guest"
    );
    assert_eq!(
        import(&r, "game.exe", "kernel32!VirtualAlloc").status(),
        "host"
    );
    assert_eq!(import(&r, "game.exe", "kernel32!Counter").status(), "guest");
    assert!(has_diag(&r, "data_export_forced_guest"));

    // --host-dll keeps a supplied DLL off the guest side entirely.
    let (r, _t) = run("pol_hostdll", &["kernel32=host"]);
    assert!(r.module("kernel32.dll").is_none());
    assert_eq!(
        import(&r, "game.exe", "kernel32!VirtualAlloc").status(),
        "host"
    );
    assert!(r.unreferenced.is_empty(), "{:?}", r.unreferenced);
    assert_eq!(
        r.external
            .iter()
            .find(|e| e.dll == "kernel32.dll")
            .unwrap()
            .status,
        "host_only"
    );
}

#[test]
fn required_rules_and_forced_modules() {
    let mut vcr = Synth::new("VCRUNTIME140.dll", DLL_BASE, true);
    vcr.text = vec![0xC3; 0x20];
    vcr.exports = vec![
        export("_CxxThrowException", 1, Target::Rva(TEXT)),
        export("memcpy", 2, Target::Rva(TEXT + 0x10)),
    ];
    let unused = Synth::new("extra.dll", DLL_BASE, true);
    let mut exe = Synth::new("game.exe", EXE_BASE, false);
    exe.imports = vec![(
        "VCRUNTIME140.dll",
        vec![Imp::Name("_CxxThrowException"), Imp::Name("memcpy")],
    )];
    let tmp = TempDir::new("required");
    let mut options = program_options(tmp.path(), &exe, &[&vcr, &unused]);
    options.overrides.add_guest_dll("extra").unwrap();
    let r = build(&options);
    let throw = import(&r, "game.exe", "VCRUNTIME140.dll!_CxxThrowException");
    assert_eq!(throw.status(), "guest");
    assert_eq!(throw.rule, "vcruntime140*.dll!_CxxThrowException=guest");
    // memcpy: prefer-host, but the bridge has no VCRUNTIME140 shim family, so guest.
    let memcpy = import(&r, "game.exe", "VCRUNTIME140.dll!memcpy");
    assert_eq!(memcpy.status(), "guest");
    assert_eq!(memcpy.rule, "vcruntime140.dll!memcpy=prefer-host");
    let extra = r.module("extra.dll").unwrap();
    assert_eq!(extra.reason, "forced");
    let order: Vec<&str> = r.modules.iter().map(|m| m.dll.as_str()).collect();
    assert_eq!(order, ["vcruntime140.dll", "extra.dll", "game.exe"]);

    // Overriding a required rule is reported; without the DLL the rule cannot be honoured.
    let tmp = TempDir::new("required2");
    let mut options = program_options(tmp.path(), &exe, &[&vcr]);
    options
        .overrides
        .add_bind("vcruntime140.dll!_CxxThrowException=host")
        .unwrap();
    let r = build(&options);
    assert_eq!(
        import(&r, "game.exe", "VCRUNTIME140.dll!_CxxThrowException").status(),
        "unresolved"
    );
    assert!(has_diag(&r, "required_rule_overridden"));
    let tmp = TempDir::new("required3");
    let r = build(&program_options(tmp.path(), &exe, &[]));
    assert!(has_diag(&r, "required_guest_unavailable"));
    assert_eq!(r.modules.len(), 1);

    let tmp = TempDir::new("required4");
    let mut options = program_options(tmp.path(), &exe, &[]);
    options.overrides.add_guest_dll("absent.dll").unwrap();
    assert!(matches!(
        build_program(&options, &mut |_, _| {}),
        Err(ProgramError::Options(_))
    ));
}

#[test]
fn allocator_family_must_not_split() {
    // A guest ucrtbase: malloc/free through the API set from the executable, operator new from
    // msvcp140.
    let mut ucrt = Synth::new("ucrtbase.dll", DLL_BASE, true);
    ucrt.text = vec![0xC3; 0x20];
    ucrt.exports = vec![
        export("malloc", 1, Target::Rva(TEXT)),
        export("free", 2, Target::Rva(TEXT + 0x10)),
    ];
    let mut msvcp = Synth::new("msvcp140.dll", DLL_BASE, true);
    msvcp.imports = vec![("api-ms-win-crt-heap-l1-1-0.dll", vec![Imp::Name("free")])];
    msvcp.exports = vec![export("?x@@YAXXZ", 1, Target::Rva(TEXT))];
    let mut exe = Synth::new("game.exe", EXE_BASE, false);
    exe.imports = vec![
        (
            "api-ms-win-crt-heap-l1-1-0.dll",
            vec![Imp::Name("malloc"), Imp::Name("free")],
        ),
        ("MSVCP140.dll", vec![Imp::Name("?x@@YAXXZ")]),
    ];
    let tmp = TempDir::new("alloc_ok");
    let r = build(&program_options(tmp.path(), &exe, &[&ucrt, &msvcp]));
    assert_eq!(r.allocator, Some(("ucrtbase.dll".to_string(), 3)));
    assert_eq!(r.module("ucrtbase.dll").unwrap().reason, "import");
    assert_eq!(
        guest_va(import(
            &r,
            "msvcp140.dll",
            "api-ms-win-crt-heap-l1-1-0.dll!free"
        )),
        guest_va(import(
            &r,
            "game.exe",
            "api-ms-win-crt-heap-l1-1-0.dll!free"
        ))
    );

    let tmp = TempDir::new("alloc_split");
    let mut options = program_options(tmp.path(), &exe, &[&ucrt, &msvcp]);
    options
        .overrides
        .add_bind("ucrtbase.dll!free=host")
        .unwrap();
    let err = build_program(&options, &mut |_, _| {}).unwrap_err();
    assert!(
        matches!(&err, ProgramError::Link(m) if m.contains("allocator split")),
        "{err}"
    );
    options.allow_allocator_split = true;
    options.build.out_dir = tmp.path().join("out2");
    let r = build(&options);
    assert!(has_diag(&r, "allocator_split"));
}

/// Regression (real DLLs): one importer names `KERNEL32.dll`, another the API set that maps to
/// it; both are the same host provider, not an inconsistent binding.
#[test]
fn api_set_and_host_spellings_of_one_import_are_consistent() {
    let mut d = Synth::new("d.dll", DLL_BASE, true);
    d.imports = vec![("KERNEL32.dll", vec![Imp::Name("AcquireSRWLockExclusive")])];
    d.exports = vec![export("fd", 1, Target::Rva(TEXT))];
    let mut exe = Synth::new("game.exe", EXE_BASE, false);
    exe.imports = vec![
        (
            "api-ms-win-core-synch-l1-2-0.dll",
            vec![Imp::Name("AcquireSRWLockExclusive")],
        ),
        ("d.dll", vec![Imp::Name("fd")]),
    ];
    let tmp = TempDir::new("spellings");
    let r = build(&program_options(tmp.path(), &exe, &[&d]));
    for (importer, label) in [
        (
            "game.exe",
            "api-ms-win-core-synch-l1-2-0.dll!AcquireSRWLockExclusive",
        ),
        ("d.dll", "KERNEL32.dll!AcquireSRWLockExclusive"),
    ] {
        assert_eq!(import(&r, importer, label).status(), "host");
    }
}

#[test]
fn tls_and_entry_points_are_recorded_at_the_assigned_base() {
    let tmp = TempDir::new("tls");
    let mut t = dll_b(DLL_BASE);
    t.entry = TEXT + 0x10;
    t.data = vec![0; 0x20];
    t.tls = Some(Tls {
        data: DATA + 8,
        len: 8,
        index: DATA + 0x10,
        callbacks: vec![TEXT],
    });
    let mut blocker = Synth::new("blocker.dll", DLL_BASE, true);
    blocker.exports = vec![export("x", 1, Target::Rva(TEXT))];
    let mut exe = Synth::new("game.exe", EXE_BASE, false);
    exe.imports = vec![
        ("blocker.dll", vec![Imp::Name("x")]),
        ("b.dll", vec![Imp::Name("fb")]),
    ];
    let r = build(&program_options(tmp.path(), &exe, &[&t, &blocker]));
    let m = r.module("b.dll").unwrap();
    let base = DLL_BASE + 0x1_0000;
    assert_eq!(m.base, base, "rebased behind blocker.dll");
    assert_eq!(m.entry_point, base + u64::from(TEXT) + 0x10);
    let tls = m.tls.as_ref().unwrap();
    assert_eq!(tls.index, 0);
    assert_eq!((tls.start, tls.end), (base + 0x2008, base + 0x2010));
    assert_eq!(tls.index_va, base + 0x2010);
    assert_eq!(tls.callbacks, vec![base + u64::from(TEXT)]);
    assert!(
        r.report
            .contains(&format!("DllMain b.dll: 0x{:x}", m.entry_point))
    );
    let descriptor =
        fs::read_to_string(tmp.path().join("out").join(PROGRAM_DESCRIPTOR_FILE)).unwrap();
    assert!(descriptor.contains("kPg6ProgramModuleDll | kPg6ProgramModuleTls, 0u"));
}

#[test]
fn export_index_follows_api_sets_to_guest_modules() {
    let mut ucrt = Synth::new("ucrtbase.dll", DLL_BASE, true);
    ucrt.exports = vec![export("strlen", 1, Target::Rva(TEXT))];
    let pe = Pe64::parse(&ucrt.build()).unwrap();
    let mut index = ExportIndex::default();
    index
        .modules
        .insert("ucrtbase.dll".into(), ModuleExports::from_pe(&pe));
    let mut via = Vec::new();
    let name = |n: &str| ImportedSymbol::ByName {
        hint: 0,
        name: n.to_string(),
    };
    assert_eq!(
        index.resolve(
            "api-ms-win-crt-string-l1-1-0.dll",
            &name("strlen"),
            &mut via
        ),
        Ok(Provider::Guest {
            dll: "ucrtbase.dll".into(),
            va: DLL_BASE + u64::from(TEXT),
            kind: ExportKind::Function
        })
    );
    assert_eq!(
        index.resolve("kernel32.dll", &name("Sleep"), &mut via),
        Ok(Provider::Host {
            dll: "kernel32.dll".into(),
            symbol: name("Sleep")
        })
    );
    assert!(matches!(
        index.resolve("ucrtbase", &name("nope"), &mut via),
        Err(link::LookupError::MissingExport { .. })
    ));
}

#[test]
fn symbol_checks_reject_duplicates_and_strays() {
    assert!(
        check_symbols(
            "a.dll",
            0x1000,
            0x2000,
            &[0x1000, 0x1100],
            &mut HashMap::new()
        )
        .is_ok()
    );
    let mut owner = HashMap::new();
    check_symbols("a.dll", 0x1000, 0x2000, &[0x1000], &mut owner).unwrap();
    let err = check_symbols("b.dll", 0x0, 0x4000, &[0x1000], &mut owner).unwrap_err();
    assert!(
        err.to_string()
            .contains("duplicate symbol sub_1000 in a.dll and b.dll"),
        "{err}"
    );
    let err = check_symbols("c.dll", 0x3000, 0x4000, &[0x2000], &mut owner).unwrap_err();
    assert!(err.to_string().contains("outside its image"), "{err}");
}

// ----- tool-gated: the generated C++ compiles ----------------------------------------------------

/// `$CXX`, else the first of `c++`, `g++`, `clang++` that runs (GNU-style flags only).
fn find_cxx() -> Option<String> {
    let candidates: Vec<String> = match env::var("CXX") {
        Ok(cxx) if !cxx.trim().is_empty() => vec![cxx],
        _ => ["c++", "g++", "clang++"].map(String::from).to_vec(),
    };
    candidates.into_iter().find(|cxx| {
        Command::new(cxx)
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
    })
}

#[test]
fn namespaced_bridges_and_descriptor_compile_together() {
    let Some(cxx) = find_cxx() else {
        println!("SKIP namespaced_bridges_and_descriptor_compile_together: no C++ compiler");
        return;
    };
    let tmp = TempDir::new("cxx");
    let r = build(&program_options(
        tmp.path(),
        &exe_main(),
        &[&dll_a(DLL_BASE), &dll_b(DLL_BASE)],
    ));
    let out = tmp.path().join("out");
    let runtime = Path::new(env!("CARGO_MANIFEST_DIR")).join("../runtime");
    let both = out.join("both.cpp");
    let mut text = String::new();
    for m in &r.modules {
        text.push_str(&format!(
            "#include \"{}/stubs/pg6_import_thunks.h\"\n",
            m.module
        ));
    }
    text.push_str("const void* pg6_probe = &pg6_import_thunks_a;\n");
    fs::write(&both, text).unwrap();
    let mut sources = vec![both];
    sources.extend(r.modules.iter().map(|m| {
        out.join(&m.module)
            .join("stubs")
            .join("pg6_import_thunks.cpp")
    }));
    if runtime.join(PROGRAM_LOADER_INCLUDE).is_file() {
        sources.push(out.join(PROGRAM_DESCRIPTOR_FILE));
    }
    for source in &sources {
        let output = Command::new(&cxx)
            .args(["-std=c++20", "-fsyntax-only", "-Wall", "-Wextra", "-Werror"])
            .arg(format!("-I{}", runtime.display()))
            .arg(format!("-I{}", tmp.path().join("in").join("rt").display()))
            .arg(format!("-I{}", out.display()))
            .arg(source)
            .output()
            .expect("run C++ compiler");
        assert!(
            output.status.success(),
            "{} does not compile:\n{}",
            source.display(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

// ----- tool-gated: link the modules and run them under node --------------------------------------

const RUNNER_JS: &str = r#"'use strict';
// Loads the guest images, registers every module's lifted-function table through its generated
// __playgta6_register_lifted_functions_<module>, then runs the entry; prints rax.
const fs = require('fs');
const [wasmPath, imagesPath, entry, ...registers] = process.argv.slice(2);
const STATE = 0x100000, STACK_TOP = 0x1F0000, RSP = 4;
const byVa = new Map();
let ex;
const fail = (what, va) => { throw new Error(what + ' 0x' + BigInt(va).toString(16)); };
const view = () => new DataView(ex.memory.buffer);
const getFn = (i) => { try { return ex.__indirect_function_table.get(Number(i)); } catch (e) { return ex.__indirect_function_table.get(BigInt(i)); } };
const instance = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(wasmPath)), { env: {
  pg6_guest_register_lifted_functions(entries, count) {
    const v = view();
    for (let i = 0; i < Number(count); i++) {
      const va = v.getBigUint64(Number(entries) + 16 * i, true);
      if (byVa.has(va)) return 0;
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
const images = fs.readFileSync(imagesPath);
const mem = new Uint8Array(ex.memory.buffer);
for (let off = 0; off < images.length;) {
  const base = Number(images.readBigUint64LE(off)), len = Number(images.readBigUint64LE(off + 8));
  mem.set(images.subarray(off + 16, off + 16 + len), base);
  off += 16 + len;
}
for (const r of registers) if (ex[r]() !== 1) throw new Error('registration rejected: ' + r);
let v = view();
for (let i = 0; i < 16; i++) v.setBigUint64(STATE + 8 * i, 0n, true);
v.setBigUint64(STACK_TOP - 8, 0xC0DEC0DEn, true);
v.setBigUint64(STATE + 8 * RSP, BigInt(STACK_TOP - 8), true);
ex[entry](BigInt(STATE));
v = view();
process.stdout.write(`${v.getBigUint64(STATE, true)} ${v.getBigUint64(STATE + 8 * RSP, true) - BigInt(STACK_TOP)}\n`);
"#;

const PROBE_LL: &str = "\
target triple = \"wasm64-unknown-unknown\"

define void @probe(ptr %p) {
entry:
  %v = load i64, ptr %p, align 1
  store i64 %v, ptr %p, align 1
  ret void
}
";

fn llvm_tool() -> Result<LlvmTool, String> {
    let tool = find_llvm_tool().ok_or("no LLVM tool (llc/clang/llvm-tools)")?;
    let dir = TempDir::new("probe_llvm");
    let ll = dir.path().join("probe.ll");
    fs::write(&ll, PROBE_LL).map_err(|e| e.to_string())?;
    compile_to_wasm64_object(&tool, &ll, &dir.path().join("probe.o"))
        .map_err(|e| format!("{} cannot compile wasm64: {e}", tool.path().display()))?;
    Ok(tool)
}

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
    exports: &[String],
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
    .arg(format!("--initial-memory={}", 16u64 << 20));
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

#[test]
fn linked_program_calls_across_modules_through_prefilled_iat_slots() {
    let skip = |reason: String| println!("SKIP linked_program_calls_across_modules: {reason}");
    if let Err(reason) = llvm_tool() {
        return skip(reason);
    }
    let Some(node) = find_on_path("node") else {
        return skip("node not found on PATH".into());
    };
    let tmp = TempDir::new("exec");
    let (exe, a, b) = (exe_main(), dll_a(DLL_BASE), dll_b(DLL_BASE));
    let mut options = program_options(tmp.path(), &exe, &[&a, &b]);
    options.build.emit_obj = true;
    let r = build(&options);
    let out = tmp.path().join("out");
    assert_eq!(
        r.module("a.dll").unwrap().base,
        DLL_BASE + 0x1_0000,
        "a.dll is rebased"
    );

    let mut objects = Vec::new();
    let mut registers = Vec::new();
    let mut images = Vec::new();
    for m in &r.modules {
        let dir = out.join(&m.module);
        for (path, _) in read_tree(&dir.join(LIFT_DIR)) {
            if path.ends_with(".o") {
                objects.push(dir.join(LIFT_DIR).join(path));
            }
        }
        registers.push(register_function_symbol(&m.module));
        let (base, bytes) = image_bytes(&out, &m.module);
        images.extend_from_slice(&base.to_le_bytes());
        images.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        images.extend_from_slice(&bytes);
    }
    // The runtime's guest base (host = base + guest VA): 0, the identity mapping, so the images
    // are copied to their guest VAs in linear memory.
    let base_ll = tmp.path().join("guest_base.ll");
    fs::write(
        &base_ll,
        format!(
            "target triple = \"wasm64-unknown-unknown\"\n@{} = global i64 0, align 8\n",
            crate::lift::GUEST_BASE_SYMBOL
        ),
    )
    .unwrap();
    let base_obj = tmp.path().join("guest_base.o");
    compile_to_wasm64_object(&llvm_tool().unwrap(), &base_ll, &base_obj).unwrap();
    objects.push(base_obj);
    let images_path = tmp.path().join("images.bin");
    fs::write(&images_path, images).unwrap();
    let entry = function_symbol(EXE_BASE + u64::from(TEXT));
    let mut exports = registers.clone();
    exports.push(entry.clone());
    let wasm = tmp.path().join("program.wasm");
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
            "no wasm linker could link the modules ({})",
            failures.join("; ")
        ));
    }
    let script = tmp.path().join("runner.js");
    fs::write(&script, RUNNER_JS).unwrap();
    let mut last_error = String::new();
    for flags in [&[][..], &["--experimental-wasm-memory64"][..]] {
        let output = Command::new(&node)
            .args(flags)
            .arg(&script)
            .arg(&wasm)
            .arg(&images_path)
            .arg(&entry)
            .args(&registers)
            .output()
            .expect("run node");
        if output.status.success() {
            // a!#1 = b!fb + a's relocated data (1 + 10), a!fwd = b!fb (1), *b!bdata = 1000.
            assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "1012 0");
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
    panic!("running the linked program failed:\n{last_error}");
}
