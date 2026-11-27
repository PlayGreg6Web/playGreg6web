//! Discovery cache: the result of the discovery stage (the [`refine::discover_refined`] fixpoint
//! and its trace-seed re-run) stored on disk under a content key, so a build after a recompiler
//! change that cannot affect discovery (a lifter edit, a shim header, a report) skips it.
//!
//! # Why one entry per stage and not per function
//!
//! Discovery is one global fixpoint: block leaders, function entries, jump-table resolutions
//! (sliced across predecessor blocks of other functions), tail-call classification, pointer-slot
//! and vtable seeds and the code-pointer refinement rounds all feed each other, and every rule's
//! result depends on the whole block map of the previous round. A per-function entry would be
//! sound only with the full set of cross-function facts that produced it in its key, which is the
//! whole analysis. And any change to the discovery code (`jump_table.rs`, `discover.rs`,
//! `reach.rs`, ...) can change any function, so no fact computed by the old code can be reused
//! without re-deriving it. The cache is therefore keyed by everything discovery reads and is all
//! or nothing:
//!
//! - [`DISCOVERY_CODE`]: a hash of every non-test source file of the crate except the lifter and
//!   the shard emission (`recomp/build.rs`), plus the manifest and `Cargo.lock` (iced-x86);
//! - the module's input bytes (rebased, as `seed` reads them) and the guest image bytes after the
//!   IAT pre-fill (the refinement reads base relocations and pointers from them), the image
//!   options, and the seeded entry points and no-return slots;
//! - for the trace-seed re-run: the first stage's key plus the accepted seed entries (the only
//!   new input of that run), so a seed file that accepts the same entries still hits.
//!
//! # Format
//!
//! `<module>-<key>.disc`: a header (magic, format version, key, payload length, FNV-1a 64 of the
//! payload) and the payload: the final entry points and no-return slots, the code-pointer
//! outcome, the jump tables, unresolved jumps, pointer slots, relocations and every function's
//! CFG. Instructions are stored as their lengths and decoded again from the image bytes on load
//! (the key pins those bytes; a decode that does not reproduce a stored length is a miss).
//! Entries are written through a temporary and renamed; a damaged, truncated or mismatched entry
//! is a miss. The newest [`KEEP_PER_MODULE`] entries per module are kept (a hit touches its
//! entry).
//!
//! `--verify-discovery-cache` always runs discovery, compares it with the stored entry when there
//! is one (every field, instructions bit for bit) and with its own encode/decode round trip, and
//! fails the build on a difference; the outputs are those of the uncached run.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::SystemTime;

use iced_x86::{Decoder, DecoderOptions, Instruction};

use super::refine::{self, CodePointer, CodePointerOutcome, PointerSource, PointerStatus};
use super::shard_lift::Fnv128;
use crate::cfg::{
    BasicBlock, CallSite, CallTarget, Diagnostic, DiagnosticKind, Edge, Function, JumpTable,
    JumpTableIndex, Terminator, UnresolvedJump,
};
use crate::code_image::{CodeImage, Region};
use crate::discover::{Analysis, DiscoveryTimings};
use crate::image::GuestImage;
use crate::reloc::{RelocKind, Relocation, RelocationMap};

/// Content hash of the sources discovery can depend on (see `recomp/build.rs`).
pub const DISCOVERY_CODE: &str = env!("PLAYGTA6_DISCOVERY_CODE");

/// Entry file format; part of the key and checked in the header.
const FORMAT: u32 = 1;
const MAGIC: &[u8; 8] = b"PG6DISC\0";
const HEADER_LEN: usize = 8 + 4 + 16 + 8 + 8;
const EXT: &str = "disc";
/// Extension of an entry's metadata file ([`DiscoveryCache::stored_timings`]).
const META_EXT: &str = "disc-meta.json";

/// Entries kept per module after a store (newest first by modification time).
pub const KEEP_PER_MODULE: usize = 3;

/// What the discovery stage leaves behind: the analysis, the refinement outcome and the two
/// fields of the code image it extends (entry points and no-return slots).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Discovered {
    pub analysis: Analysis,
    pub code_pointers: CodePointerOutcome,
    pub entry_points: Vec<u64>,
    pub no_return: Vec<u64>,
}

/// How a lookup went (for the progress line).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Hit,
    /// Computed and stored.
    Miss,
    /// Computed and compared with the stored entry (which matched).
    Verified,
    /// Computed and stored; there was no entry to compare with (the round trip matched).
    VerifiedMiss,
}

impl Outcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hit => "hit",
            Self::Miss => "miss",
            Self::Verified => "verified against the stored entry",
            Self::VerifiedMiss => "verified round trip, stored",
        }
    }
}

/// A discovery cache directory.
#[derive(Debug, Clone)]
pub struct DiscoveryCache {
    dir: PathBuf,
    module: String,
    verify: bool,
}

/// Key of the first discovery stage of a module (see the module docs).
pub fn stage_key(
    input: &[u8],
    guest_bytes: Option<&[u8]>,
    image_options: &impl std::fmt::Debug,
    image: &CodeImage<'_>,
) -> u128 {
    let mut h = Fnv128::new();
    h.update(b"playgta6 discovery\0");
    h.update(DISCOVERY_CODE.as_bytes());
    h.update(&FORMAT.to_le_bytes());
    h.update(&(input.len() as u64).to_le_bytes());
    h.update(input);
    match guest_bytes {
        Some(bytes) => {
            h.update(&[1]);
            h.update(&(bytes.len() as u64).to_le_bytes());
            h.update(bytes);
        }
        None => h.update(&[0]),
    }
    h.update(format!("{image_options:?}").as_bytes());
    for list in [&image.entry_points, &image.no_return] {
        h.update(&(list.len() as u64).to_le_bytes());
        for va in list.iter() {
            h.update(&va.to_le_bytes());
        }
    }
    h.finish()
}

/// Key of the trace-seed re-run: the first stage's key and the accepted entries.
pub fn seeded_key(first: u128, accepted: &[u64]) -> u128 {
    let mut h = Fnv128::new();
    h.update(b"playgta6 discovery seeds\0");
    h.update(&first.to_le_bytes());
    h.update(&(accepted.len() as u64).to_le_bytes());
    for va in accepted {
        h.update(&va.to_le_bytes());
    }
    h.finish()
}

impl DiscoveryCache {
    pub fn new(dir: impl Into<PathBuf>, module: &str, verify: bool) -> Self {
        Self {
            dir: dir.into(),
            module: module.to_string(),
            verify,
        }
    }

    fn path(&self, key: u128) -> PathBuf {
        self.dir.join(format!("{}-{key:032x}.{EXT}", self.module))
    }

    /// Path of the metadata next to the entry under `key`: the [`DiscoveryTimings::to_json`] of
    /// the run that computed it (diagnostic only, never part of the entry or its check).
    fn meta_path(&self, key: u128) -> PathBuf {
        self.dir.join(format!("{}-{key:032x}.{META_EXT}", self.module))
    }

    /// The stored timings of the run that computed the entry under `key`, if recorded.
    pub fn stored_timings(&self, key: u128) -> Option<String> {
        let text = fs::read_to_string(self.meta_path(key)).ok()?;
        let text = text.trim();
        (text.starts_with('{') && text.ends_with('}')).then(|| text.to_string())
    }

    /// [`refine::discover_refined`] on `image` through the cache under `key`: on a hit the
    /// stored result is applied to `image` (entry points, no-return slots) and returned without
    /// running discovery. Store failures are ignored (the cache never changes the outputs); a
    /// verification difference is an error. A computed run adds its phase times to `timings`
    /// and stores them as the entry's metadata ([`Self::stored_timings`]).
    pub fn discover_refined(
        &self,
        key: u128,
        image: &mut CodeImage<'_>,
        guest: &GuestImage<'_>,
        timings: &mut DiscoveryTimings,
    ) -> Result<(Analysis, CodePointerOutcome, Outcome), String> {
        let path = self.path(key);
        if !self.verify
            && let Some(found) = load(&path, key, image)
        {
            touch(&path);
            image.entry_points = found.entry_points;
            image.no_return = found.no_return;
            return Ok((found.analysis, found.code_pointers, Outcome::Hit));
        }
        let stored = if self.verify {
            load(&path, key, image)
        } else {
            None
        };
        let mut run = DiscoveryTimings::default();
        let (analysis, code_pointers) =
            refine::discover_refined_timed(image, guest, refine::MAX_ROUNDS, &mut run);
        timings.merge(&run);
        let fresh = Discovered {
            analysis,
            code_pointers,
            entry_points: image.entry_points.clone(),
            no_return: image.no_return.clone(),
        };
        let payload = encode(&fresh);
        let mut outcome = Outcome::Miss;
        if self.verify {
            let round_trip = decode(&payload, image)
                .ok_or("discovery cache: the encoded analysis does not decode")?;
            same(&fresh, &round_trip)
                .map_err(|d| format!("discovery cache: encode/decode round trip differs: {d}"))?;
            outcome = Outcome::VerifiedMiss;
            if let Some(stored) = &stored {
                same(&fresh, stored).map_err(|d| {
                    format!(
                        "discovery cache: stored entry {} differs from a fresh discovery: {d}",
                        path.display()
                    )
                })?;
                outcome = Outcome::Verified;
            }
        }
        if stored.is_none() {
            if self.store(key, &payload).is_ok() {
                let _ = fs::write(self.meta_path(key), run.to_json());
            }
        } else {
            touch(&path);
        }
        Ok((fresh.analysis, fresh.code_pointers, outcome))
    }

    fn store(&self, key: u128, payload: &[u8]) -> io::Result<()> {
        fs::create_dir_all(&self.dir)?;
        let path = self.path(key);
        let tmp = self.dir.join(format!(
            "{}-{key:032x}.{}.tmp",
            self.module,
            std::process::id()
        ));
        {
            let mut file = io::BufWriter::new(fs::File::create(&tmp)?);
            file.write_all(&header(key, payload))?;
            file.write_all(payload)?;
            file.flush()?;
        }
        if let Err(e) = fs::rename(&tmp, &path) {
            let _ = fs::remove_file(&tmp);
            return Err(e);
        }
        self.trim(&path);
        Ok(())
    }

    /// Keeps the newest [`KEEP_PER_MODULE`] entries of this module (and always `current`).
    fn trim(&self, current: &Path) {
        let Ok(dir) = fs::read_dir(&self.dir) else {
            return;
        };
        let prefix = format!("{}-", self.module);
        let mut entries: Vec<(SystemTime, PathBuf)> = dir
            .flatten()
            .filter_map(|e| {
                let name = e.file_name().into_string().ok()?;
                let key = name
                    .strip_prefix(&prefix)?
                    .strip_suffix(&format!(".{EXT}"))?;
                // Another module whose name extends this one's has a non-hex part here.
                if key.len() != 32 || !key.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return None;
                }
                let modified = e.metadata().ok()?.modified().ok()?;
                Some((modified, e.path()))
            })
            .collect();
        entries.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        for (_, path) in entries
            .into_iter()
            .filter(|(_, p)| p != current)
            .skip(KEEP_PER_MODULE.saturating_sub(1))
        {
            let _ = fs::remove_file(path.with_extension(META_EXT));
            let _ = fs::remove_file(path);
        }
    }
}

fn touch(path: &Path) {
    if let Ok(file) = fs::OpenOptions::new().append(true).open(path) {
        let _ = file.set_modified(SystemTime::now());
    }
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn header(key: u128, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&FORMAT.to_le_bytes());
    out.extend_from_slice(&key.to_le_bytes());
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(&fnv64(payload).to_le_bytes());
    out
}

/// The entry at `path` if it is complete, intact, has `key` and decodes against `image`.
fn load(path: &Path, key: u128, image: &CodeImage<'_>) -> Option<Discovered> {
    let bytes = fs::read(path).ok()?;
    if bytes.len() < HEADER_LEN {
        return None;
    }
    let (head, payload) = bytes.split_at(HEADER_LEN);
    let expected = header(key, payload);
    if head != expected.as_slice() {
        return None;
    }
    decode(payload, image)
}

// ---------------------------------------------------------------------------------------------
// Comparison (verification mode)
// ---------------------------------------------------------------------------------------------

/// `Ok` when `a` and `b` are identical, including every bit of every decoded instruction
/// (`Instruction`'s `PartialEq` ignores the IP and length); else the first difference.
pub fn same(a: &Discovered, b: &Discovered) -> Result<(), String> {
    if a.entry_points != b.entry_points {
        return Err("entry points".into());
    }
    if a.no_return != b.no_return {
        return Err("no-return slots".into());
    }
    if a.code_pointers != b.code_pointers {
        return Err("code-pointer outcome".into());
    }
    let (x, y) = (&a.analysis, &b.analysis);
    if x.relocations != y.relocations {
        return Err("relocations".into());
    }
    if x.jump_tables != y.jump_tables {
        return Err("jump tables".into());
    }
    if x.unresolved_jumps != y.unresolved_jumps {
        return Err("unresolved jumps".into());
    }
    if x.pointer_slots != y.pointer_slots {
        return Err("pointer slots".into());
    }
    if x.functions.len() != y.functions.len() {
        return Err(format!(
            "function count {} vs {}",
            x.functions.len(),
            y.functions.len()
        ));
    }
    for ((ka, fa), (kb, fb)) in x.functions.iter().zip(&y.functions) {
        if ka != kb || fa != fb {
            return Err(format!("function {ka:#x}"));
        }
        for (ba, bb) in fa.blocks.values().zip(fb.blocks.values()) {
            let bits = ba.instructions.len() == bb.instructions.len()
                && ba
                    .instructions
                    .iter()
                    .zip(&bb.instructions)
                    .all(|(i, j)| i.eq_all_bits(j));
            if !bits {
                return Err(format!(
                    "instructions of block {:#x} in function {ka:#x}",
                    ba.start
                ));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------------------------

#[derive(Default)]
struct Out(Vec<u8>);

impl Out {
    fn u(&mut self, mut v: u64) {
        loop {
            let byte = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                self.0.push(byte);
                return;
            }
            self.0.push(byte | 0x80);
        }
    }
    fn n(&mut self, n: usize) {
        self.u(n as u64);
    }
    fn b(&mut self, v: bool) {
        self.0.push(u8::from(v));
    }
    fn byte(&mut self, v: u8) {
        self.0.push(v);
    }
    /// `va` relative to `base` (zigzag of the wrapping difference): small for nearby addresses.
    fn rel(&mut self, base: u64, va: u64) {
        let d = va.wrapping_sub(base) as i64;
        self.u(((d << 1) ^ (d >> 63)) as u64);
    }
    fn opt(&mut self, v: Option<u64>) {
        match v {
            Some(v) => {
                self.byte(1);
                self.u(v);
            }
            None => self.byte(0),
        }
    }
    /// Ascending addresses as deltas.
    fn set<'a>(&mut self, items: impl ExactSizeIterator<Item = &'a u64>) {
        self.n(items.len());
        let mut prev = 0;
        for &va in items {
            self.rel(prev, va);
            prev = va;
        }
    }
}

struct In<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> In<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    fn byte(&mut self) -> Option<u8> {
        let b = *self.bytes.get(self.pos)?;
        self.pos += 1;
        Some(b)
    }
    fn u(&mut self) -> Option<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let b = self.byte()?;
            v |= u64::from(b & 0x7f).checked_shl(shift)?;
            if b & 0x80 == 0 {
                return Some(v);
            }
        }
        None
    }
    /// A count, bounded by the bytes left (every item takes at least one byte).
    fn n(&mut self) -> Option<usize> {
        let n = usize::try_from(self.u()?).ok()?;
        (n <= self.bytes.len() - self.pos).then_some(n)
    }
    fn b(&mut self) -> Option<bool> {
        match self.byte()? {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }
    fn rel(&mut self, base: u64) -> Option<u64> {
        let z = self.u()?;
        let d = ((z >> 1) as i64) ^ -((z & 1) as i64);
        Some(base.wrapping_add(d as u64))
    }
    fn opt(&mut self) -> Option<Option<u64>> {
        match self.byte()? {
            0 => Some(None),
            1 => Some(Some(self.u()?)),
            _ => None,
        }
    }
    fn set(&mut self) -> Option<Vec<u64>> {
        let n = self.n()?;
        let mut out = Vec::with_capacity(n);
        let mut prev = 0;
        for _ in 0..n {
            let va = self.rel(prev)?;
            out.push(va);
            prev = va;
        }
        Some(out)
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        let s = self.bytes.get(self.pos..end)?;
        self.pos = end;
        Some(s)
    }
    fn done(&self) -> bool {
        self.pos == self.bytes.len()
    }
}

fn terminator_code(t: Terminator) -> u8 {
    match t {
        Terminator::Fallthrough => 0,
        Terminator::Branch => 1,
        Terminator::Jump => 2,
        Terminator::IndirectJump => 3,
        Terminator::Return => 4,
        Terminator::NoReturnCall => 5,
        Terminator::Trap => 6,
        Terminator::Invalid => 7,
    }
}

fn terminator_from(c: u8) -> Option<Terminator> {
    Some(match c {
        0 => Terminator::Fallthrough,
        1 => Terminator::Branch,
        2 => Terminator::Jump,
        3 => Terminator::IndirectJump,
        4 => Terminator::Return,
        5 => Terminator::NoReturnCall,
        6 => Terminator::Trap,
        7 => Terminator::Invalid,
        _ => return None,
    })
}

fn diagnostic_code(k: DiagnosticKind) -> u8 {
    match k {
        DiagnosticKind::InvalidInstruction => 0,
        DiagnosticKind::TruncatedInstruction => 1,
        DiagnosticKind::NotExecutable => 2,
        DiagnosticKind::Unmapped => 3,
        DiagnosticKind::OverlappingInstruction => 4,
    }
}

fn diagnostic_from(c: u8) -> Option<DiagnosticKind> {
    Some(match c {
        0 => DiagnosticKind::InvalidInstruction,
        1 => DiagnosticKind::TruncatedInstruction,
        2 => DiagnosticKind::NotExecutable,
        3 => DiagnosticKind::Unmapped,
        4 => DiagnosticKind::OverlappingInstruction,
        _ => return None,
    })
}

fn unresolved_code(u: UnresolvedJump) -> u8 {
    match u {
        UnresolvedJump::MemoryOperand => 0,
        UnresolvedJump::LoadedPointer => 1,
        UnresolvedJump::TableBaseUnknown => 2,
        UnresolvedJump::TableUnbounded => 3,
        UnresolvedJump::RegisterLiveIn => 4,
        UnresolvedJump::Computed => 5,
    }
}

fn unresolved_from(c: u8) -> Option<UnresolvedJump> {
    Some(match c {
        0 => UnresolvedJump::MemoryOperand,
        1 => UnresolvedJump::LoadedPointer,
        2 => UnresolvedJump::TableBaseUnknown,
        3 => UnresolvedJump::TableUnbounded,
        4 => UnresolvedJump::RegisterLiveIn,
        5 => UnresolvedJump::Computed,
        _ => return None,
    })
}

fn reloc_code(k: RelocKind) -> u8 {
    match k {
        RelocKind::CallTarget => 0,
        RelocKind::BranchTarget => 1,
        RelocKind::RipData => 2,
        RelocKind::RipAddress => 3,
        RelocKind::CallSlot => 4,
        RelocKind::JumpSlot => 5,
        RelocKind::AbsoluteImmediate => 6,
        RelocKind::AbsoluteDisplacement => 7,
        RelocKind::JumpTable => 8,
        RelocKind::JumpTableIndex => 9,
        RelocKind::JumpTableEntry => 10,
    }
}

fn reloc_from(c: u8) -> Option<RelocKind> {
    Some(match c {
        0 => RelocKind::CallTarget,
        1 => RelocKind::BranchTarget,
        2 => RelocKind::RipData,
        3 => RelocKind::RipAddress,
        4 => RelocKind::CallSlot,
        5 => RelocKind::JumpSlot,
        6 => RelocKind::AbsoluteImmediate,
        7 => RelocKind::AbsoluteDisplacement,
        8 => RelocKind::JumpTable,
        9 => RelocKind::JumpTableIndex,
        10 => RelocKind::JumpTableEntry,
        _ => return None,
    })
}

fn region_code(r: Region) -> u8 {
    match r {
        Region::Code => 0,
        Region::Data => 1,
        Region::Image => 2,
        Region::Outside => 3,
    }
}

fn region_from(c: u8) -> Option<Region> {
    Some(match c {
        0 => Region::Code,
        1 => Region::Data,
        2 => Region::Image,
        3 => Region::Outside,
        _ => return None,
    })
}

fn status_code(s: PointerStatus) -> u8 {
    match s {
        PointerStatus::KnownEntry => 0,
        PointerStatus::Accepted => 1,
        PointerStatus::PdataInterior => 2,
        PointerStatus::JumpTableData => 3,
        PointerStatus::InsideInstruction => 4,
        PointerStatus::NotDecodable => 5,
        PointerStatus::Implausible => 6,
    }
}

fn status_from(c: u8) -> Option<PointerStatus> {
    Some(match c {
        0 => PointerStatus::KnownEntry,
        1 => PointerStatus::Accepted,
        2 => PointerStatus::PdataInterior,
        3 => PointerStatus::JumpTableData,
        4 => PointerStatus::InsideInstruction,
        5 => PointerStatus::NotDecodable,
        6 => PointerStatus::Implausible,
        _ => return None,
    })
}

fn source_bit(s: PointerSource) -> u8 {
    match s {
        PointerSource::Lea => 1,
        PointerSource::BaseReloc => 2,
    }
}

fn put_edge(o: &mut Out, base: u64, e: &Edge) {
    match *e {
        Edge::Fallthrough(t) => {
            o.byte(0);
            o.rel(base, t);
        }
        Edge::Taken(t) => {
            o.byte(1);
            o.rel(base, t);
        }
        Edge::Jump(t) => {
            o.byte(2);
            o.rel(base, t);
        }
        Edge::TailCall(t) => {
            o.byte(3);
            o.rel(base, t);
        }
        Edge::IndirectJump { slot } => {
            o.byte(4);
            o.opt(slot);
        }
        Edge::Return => o.byte(5),
        Edge::JumpTable(t) => {
            o.byte(6);
            o.rel(base, t);
        }
    }
}

fn get_edge(i: &mut In<'_>, base: u64) -> Option<Edge> {
    Some(match i.byte()? {
        0 => Edge::Fallthrough(i.rel(base)?),
        1 => Edge::Taken(i.rel(base)?),
        2 => Edge::Jump(i.rel(base)?),
        3 => Edge::TailCall(i.rel(base)?),
        4 => Edge::IndirectJump { slot: i.opt()? },
        5 => Edge::Return,
        6 => Edge::JumpTable(i.rel(base)?),
        _ => return None,
    })
}

fn put_call(o: &mut Out, base: u64, c: &CallSite) {
    o.rel(base, c.site);
    match c.target {
        CallTarget::Direct(t) => {
            o.byte(0);
            o.rel(c.site, t);
        }
        CallTarget::Indirect { slot } => {
            o.byte(1);
            o.opt(slot);
        }
    }
}

fn get_call(i: &mut In<'_>, base: u64) -> Option<CallSite> {
    let site = i.rel(base)?;
    let target = match i.byte()? {
        0 => CallTarget::Direct(i.rel(site)?),
        1 => CallTarget::Indirect { slot: i.opt()? },
        _ => return None,
    };
    Some(CallSite { site, target })
}

fn put_table(o: &mut Out, t: &JumpTable) {
    o.u(t.site);
    o.rel(t.site, t.load_site);
    o.rel(t.site, t.table);
    o.n(t.targets.len());
    for &target in &t.targets {
        o.rel(t.site, target);
    }
    match t.index {
        Some(index) => {
            o.byte(1);
            o.rel(t.site, index.load_site);
            o.rel(t.site, index.table);
            o.u(index.len);
        }
        None => o.byte(0),
    }
    o.b(t.inferred);
    o.n(t.alternates.len());
    for alt in &t.alternates {
        put_table(o, alt);
    }
}

fn get_table(i: &mut In<'_>, depth: u32) -> Option<JumpTable> {
    if depth > 64 {
        return None;
    }
    let site = i.u()?;
    let load_site = i.rel(site)?;
    let table = i.rel(site)?;
    let n = i.n()?;
    let mut targets = Vec::with_capacity(n);
    for _ in 0..n {
        targets.push(i.rel(site)?);
    }
    let index = match i.byte()? {
        0 => None,
        1 => Some(JumpTableIndex {
            load_site: i.rel(site)?,
            table: i.rel(site)?,
            len: i.u()?,
        }),
        _ => return None,
    };
    let inferred = i.b()?;
    let n = i.n()?;
    let mut alternates = Vec::with_capacity(n);
    for _ in 0..n {
        alternates.push(get_table(i, depth + 1)?);
    }
    Some(JumpTable {
        site,
        load_site,
        table,
        targets,
        index,
        inferred,
        alternates,
    })
}

fn put_block(o: &mut Out, entry: u64, key: u64, b: &BasicBlock) {
    o.rel(entry, key);
    o.rel(key, b.start);
    o.rel(b.start, b.end);
    // Instructions: their lengths when they run back to back from `start` (always, for blocks
    // decoded by discovery), else each one's IP as well.
    let mut ip = b.start;
    let contiguous = b.instructions.iter().all(|instr| {
        let ok = instr.ip() == ip;
        ip = instr.next_ip();
        ok
    });
    o.b(!contiguous);
    o.n(b.instructions.len());
    for instr in &b.instructions {
        if !contiguous {
            o.rel(b.start, instr.ip());
        }
        o.byte(instr.len() as u8);
    }
    o.byte(terminator_code(b.terminator));
    o.n(b.successors.len());
    for e in &b.successors {
        put_edge(o, b.start, e);
    }
    o.n(b.calls.len());
    for c in &b.calls {
        put_call(o, b.start, c);
    }
    o.n(b.diagnostics.len());
    for d in &b.diagnostics {
        o.rel(b.start, d.site);
        o.rel(d.site, d.address);
        o.byte(diagnostic_code(d.kind));
    }
}

/// Decodes the instruction at `ip` (bytes from its section, as discovery's decoder saw them).
fn decode_at(image: &CodeImage<'_>, ip: u64, len: u8) -> Option<Instruction> {
    let bytes = image.code_bytes(ip).ok()?;
    let max = usize::try_from(u64::MAX - ip).unwrap_or(usize::MAX);
    let bytes = &bytes[..bytes.len().min(max)];
    let mut decoder = Decoder::with_ip(64, bytes, ip, DecoderOptions::NONE);
    let instr = decoder.decode();
    (!instr.is_invalid() && instr.len() == usize::from(len)).then_some(instr)
}

fn get_block(i: &mut In<'_>, entry: u64, image: &CodeImage<'_>) -> Option<(u64, BasicBlock)> {
    let key = i.rel(entry)?;
    let start = i.rel(key)?;
    let end = i.rel(start)?;
    let explicit = i.b()?;
    let n = i.n()?;
    let mut instructions = Vec::with_capacity(n);
    if explicit {
        for _ in 0..n {
            let ip = i.rel(start)?;
            let len = i.byte()?;
            instructions.push(decode_at(image, ip, len)?);
        }
    } else if n > 0 {
        let bytes = image.code_bytes(start).ok()?;
        let max = usize::try_from(u64::MAX - start).unwrap_or(usize::MAX);
        let bytes = &bytes[..bytes.len().min(max)];
        let mut decoder = Decoder::with_ip(64, bytes, start, DecoderOptions::NONE);
        let mut instr = Instruction::default();
        for _ in 0..n {
            let len = i.byte()?;
            if !decoder.can_decode() {
                return None;
            }
            decoder.decode_out(&mut instr);
            if instr.is_invalid() || instr.len() != usize::from(len) {
                return None;
            }
            instructions.push(instr);
        }
    }
    let terminator = terminator_from(i.byte()?)?;
    let n = i.n()?;
    let mut successors = Vec::with_capacity(n);
    for _ in 0..n {
        successors.push(get_edge(i, start)?);
    }
    let n = i.n()?;
    let mut calls = Vec::with_capacity(n);
    for _ in 0..n {
        calls.push(get_call(i, start)?);
    }
    let n = i.n()?;
    let mut diagnostics = Vec::with_capacity(n);
    for _ in 0..n {
        let site = i.rel(start)?;
        let address = i.rel(site)?;
        let kind = diagnostic_from(i.byte()?)?;
        diagnostics.push(Diagnostic {
            site,
            address,
            kind,
        });
    }
    Some((
        key,
        BasicBlock {
            start,
            end,
            instructions,
            terminator,
            successors,
            calls,
            diagnostics,
        },
    ))
}

fn put_function(o: &mut Out, f: &Function) {
    o.u(f.entry);
    o.n(f.blocks.len());
    for (&key, block) in &f.blocks {
        put_block(o, f.entry, key, block);
    }
    o.n(f.callees.len());
    for &c in &f.callees {
        o.rel(f.entry, c);
    }
    o.n(f.resume_targets.len());
    for &r in &f.resume_targets {
        o.rel(f.entry, r);
    }
}

fn get_function(bytes: &[u8], image: &CodeImage<'_>) -> Option<Function> {
    let mut i = In::new(bytes);
    let entry = i.u()?;
    let n = i.n()?;
    let mut blocks = BTreeMap::new();
    for _ in 0..n {
        let (key, block) = get_block(&mut i, entry, image)?;
        if blocks.insert(key, block).is_some() {
            return None;
        }
    }
    let mut callees = BTreeSet::new();
    for _ in 0..i.n()? {
        callees.insert(i.rel(entry)?);
    }
    let mut resume_targets = BTreeSet::new();
    for _ in 0..i.n()? {
        resume_targets.insert(i.rel(entry)?);
    }
    i.done().then_some(Function {
        entry,
        blocks,
        callees,
        resume_targets,
    })
}

/// The payload of an entry (see the module docs).
pub fn encode(d: &Discovered) -> Vec<u8> {
    let mut o = Out::default();
    o.set(d.entry_points.iter());
    o.set(d.no_return.iter());

    let cp = &d.code_pointers;
    o.n(cp.rounds);
    o.b(cp.round_limit_hit);
    o.set(cp.no_return_thunks.iter());
    o.n(cp.pointers.len());
    let mut prev = 0;
    for p in &cp.pointers {
        o.rel(prev, p.va);
        prev = p.va;
        o.byte(p.sources.iter().fold(0, |m, &s| m | source_bit(s)));
        o.byte(status_code(p.status));
        o.b(p.lifted);
    }

    let a = &d.analysis;
    o.n(a.jump_tables.len());
    for (&va, table) in &a.jump_tables {
        o.u(va);
        put_table(&mut o, table);
    }
    o.n(a.unresolved_jumps.len());
    let mut prev = 0;
    for (&va, &why) in &a.unresolved_jumps {
        o.rel(prev, va);
        prev = va;
        o.byte(unresolved_code(why));
    }
    o.n(a.pointer_slots.len());
    for (&va, &(slot, target)) in &a.pointer_slots {
        o.u(va);
        o.rel(va, slot);
        o.rel(va, target);
    }
    let relocs = a.relocations.entries();
    o.n(relocs.len());
    let mut prev = 0;
    for r in relocs {
        o.rel(prev, r.site);
        prev = r.site;
        o.byte(reloc_code(r.kind));
        o.rel(r.site, r.target);
        o.byte(region_code(r.region));
    }

    o.n(a.functions.len());
    let mut body = Out::default();
    for f in a.functions.values() {
        body.0.clear();
        put_function(&mut body, f);
        o.n(body.0.len());
        o.0.extend_from_slice(&body.0);
    }
    o.0
}

/// The inverse of [`encode`], decoding instructions from `image`; `None` on any inconsistency.
pub fn decode(payload: &[u8], image: &CodeImage<'_>) -> Option<Discovered> {
    let mut i = In::new(payload);
    let entry_points = i.set()?;
    let no_return = i.set()?;

    let rounds = usize::try_from(i.u()?).ok()?;
    let round_limit_hit = i.b()?;
    let no_return_thunks = i.set()?;
    let n = i.n()?;
    let mut pointers = Vec::with_capacity(n);
    let mut prev = 0;
    for _ in 0..n {
        let va = i.rel(prev)?;
        prev = va;
        let bits = i.byte()?;
        if bits & !3 != 0 {
            return None;
        }
        let sources: BTreeSet<PointerSource> = PointerSource::ALL
            .into_iter()
            .filter(|&s| bits & source_bit(s) != 0)
            .collect();
        let status = status_from(i.byte()?)?;
        let lifted = i.b()?;
        pointers.push(CodePointer {
            va,
            sources,
            status,
            lifted,
        });
    }
    let code_pointers = CodePointerOutcome {
        rounds,
        round_limit_hit,
        no_return_thunks,
        pointers,
    };

    let mut jump_tables = BTreeMap::new();
    for _ in 0..i.n()? {
        let va = i.u()?;
        jump_tables.insert(va, get_table(&mut i, 0)?);
    }
    let mut unresolved_jumps = BTreeMap::new();
    let mut prev = 0;
    for _ in 0..i.n()? {
        let va = i.rel(prev)?;
        prev = va;
        unresolved_jumps.insert(va, unresolved_from(i.byte()?)?);
    }
    let mut pointer_slots = BTreeMap::new();
    for _ in 0..i.n()? {
        let va = i.u()?;
        let slot = i.rel(va)?;
        let target = i.rel(va)?;
        pointer_slots.insert(va, (slot, target));
    }
    let n = i.n()?;
    let mut relocs = Vec::with_capacity(n);
    let mut prev = 0;
    for _ in 0..n {
        let site = i.rel(prev)?;
        prev = site;
        let kind = reloc_from(i.byte()?)?;
        let target = i.rel(site)?;
        let region = region_from(i.byte()?)?;
        relocs.push(Relocation {
            site,
            kind,
            target,
            region,
        });
    }
    // Stored sorted and deduplicated; anything else is not an entry this code wrote.
    if !relocs.windows(2).all(|w| w[0] < w[1]) {
        return None;
    }
    let relocations = RelocationMap::from_entries(relocs);

    let n = i.n()?;
    let mut bodies = Vec::with_capacity(n);
    for _ in 0..n {
        let len = i.n()?;
        bodies.push(i.take(len)?);
    }
    if !i.done() {
        return None;
    }
    let functions = decode_functions(&bodies, image)?;
    let mut map = BTreeMap::new();
    for f in functions {
        if map.insert(f.entry, f).is_some() {
            return None;
        }
    }
    Some(Discovered {
        analysis: Analysis {
            functions: map,
            relocations,
            jump_tables,
            unresolved_jumps,
            pointer_slots,
        },
        code_pointers,
        entry_points,
        no_return,
    })
}

/// Decodes the function bodies on scoped threads (instruction decoding is most of the load).
/// Workers claim small chunks from a shared counter; the result is in input order.
fn decode_functions(bodies: &[&[u8]], image: &CodeImage<'_>) -> Option<Vec<Function>> {
    let threads = crate::discover::default_threads().min(bodies.len()).max(1);
    if threads == 1 {
        return bodies.iter().map(|b| get_function(b, image)).collect();
    }
    let chunk = (bodies.len() / (threads * 8)).max(1);
    let next = AtomicUsize::new(0);
    let parts: Mutex<Vec<(usize, Option<Vec<Function>>)>> = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let mut local = Vec::new();
                loop {
                    let start = next.fetch_add(chunk, Ordering::Relaxed);
                    if start >= bodies.len() {
                        break;
                    }
                    let end = (start + chunk).min(bodies.len());
                    let part: Option<Vec<Function>> = bodies[start..end]
                        .iter()
                        .map(|b| get_function(b, image))
                        .collect();
                    local.push((start, part));
                }
                parts
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .extend(local);
            });
        }
    });
    let mut parts = parts
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    parts.sort_unstable_by_key(|(start, _)| *start);
    let mut out = Vec::with_capacity(bodies.len());
    for (_, part) in parts {
        out.extend(part?);
    }
    Some(out)
}
