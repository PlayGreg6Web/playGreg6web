//! Incremental lifting: a stable shard plan and a content-keyed cache of lifted shard IR.
//!
//! A recompiler change invalidates the whole program lift (the lift cache in
//! `tools/lift_cache.py` is keyed by every recompiler source), and the greedy shard plan
//! ([`super::plan_shards`]) moved every boundary after the first function whose size changed, so
//! most shards' IR changed and their objects were compiled again. Two pieces fix that (see
//! `docs/BUILD_CACHING.md` for measurements):
//!
//! - [`plan_shards_stable`] cuts shards before *anchor* functions chosen by a hash of their own
//!   entry VA and size (content-defined chunking), so a function that grows, shrinks, appears or
//!   disappears only moves the boundaries next to it; every other shard keeps the same functions
//!   and, when their code is unchanged, byte-identical IR (an object cache hit).
//! - [`ShardCache`] (`build-program --lift-cache <dir>`) keeps each shard's IR under a key built
//!   from the [`function_key`]s of its functions plus everything else its emission reads; a shard
//!   whose key is cached is linked in without lifting.
//!
//! # What a key covers
//! [`function_key`] hashes the function's whole CFG (`Debug` of [`Function`]: every block, every
//! decoded instruction with its address, length and operands, every edge including resolved
//! jump-table cases and tail calls, the call sites, the resume targets) plus the
//! [`FlagSummaries`] entries of the function and of every address it calls or jumps to (the only
//! cross-function facts the lifter reads). A discovery or jump-table change therefore re-lifts
//! exactly the functions whose CFG or callee summaries it changed.
//!
//! [`shard_key`] adds the lifter's code version [`LIFT_CODEGEN`] (a build-time hash of
//! `src/lift/**` without tests, `src/cfg.rs`, this file, the crate manifest and `Cargo.lock`; see
//! `recomp/build.rs`), the target, the memory model, the shard dispatch symbol, and per function
//! whether it is probed and whether it keeps external linkage. Any change to the lifter itself
//! re-lifts every shard (correctness first); the shard objects still hit the object cache for
//! every shard whose IR comes out byte-identical.
//!
//! # Layout
//! `<dir>/<key>.ll` (the shard IR exactly as written to `lift/shard-*.ll`) and `<dir>/<key>.meta`
//! (size and FNV-1a 64 of the IR, unsupported instructions by mnemonic), written through
//! temporaries and renamed, the `.meta` last; an entry counts only when both exist and the sizes
//! match. Hits are hard-linked into the output (copied across volumes) and touched, so the
//! least recently used entries can be trimmed by size (`tools/lift_cache.py`). The cache holds
//! generated IR only and must live outside every git work tree like any build output.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Write as _};
use std::fs;
use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use super::{LIFT_DIR, OutputFile, PipelineError, ShardRecord, compile, io_error, shard_file_name};
use crate::cfg::{CallTarget, Function};
use crate::image;
use crate::lift::compile::LlvmTool;
use crate::lift::target::LiftTarget;
use crate::lift::{self, FlagSummaries, MemoryModel, ShardDispatch};

/// Content hash of the lifter sources (see the module docs and `recomp/build.rs`).
pub const LIFT_CODEGEN: &str = env!("PLAYGTA6_LIFT_CODEGEN");

/// Format version of the cache keys and entries; bump when the key derivation changes.
pub const SHARD_CACHE_VERSION: u32 = 1;

/// 128-bit FNV-1a (also usable as a [`fmt::Write`] sink, so `Debug` output is hashed without
/// being materialised).
#[derive(Debug, Clone, Copy)]
pub struct Fnv128(u128);

impl Fnv128 {
    const OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
    const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;

    pub fn new() -> Self {
        Self(Self::OFFSET)
    }

    pub fn update(&mut self, bytes: &[u8]) {
        let mut h = self.0;
        for &b in bytes {
            h ^= u128::from(b);
            h = h.wrapping_mul(Self::PRIME);
        }
        self.0 = h;
    }

    pub fn finish(&self) -> u128 {
        self.0
    }
}

impl Default for Fnv128 {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Write for Fnv128 {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.update(s.as_bytes());
        Ok(())
    }
}

/// Addresses whose [`FlagSummaries`] entries the lifter may read for `f`: its entry, its direct
/// callees and call sites, and every edge target (tail calls; intra-function targets too, which
/// costs nothing and stays correct if the lifter ever looks them up).
pub fn summary_targets(f: &Function) -> BTreeSet<u64> {
    let mut out = BTreeSet::from([f.entry]);
    out.extend(f.callees.iter().copied());
    for block in f.blocks.values() {
        for call in &block.calls {
            if let CallTarget::Direct(t) = call.target {
                out.insert(t);
            }
        }
        out.extend(block.successors.iter().filter_map(|e| e.target()));
    }
    out
}

/// Cache key of one function's lifted code apart from the lifter version and the shard context
/// (see the module docs).
pub fn function_key(f: &Function, summaries: &FlagSummaries) -> u128 {
    let mut h = Fnv128::new();
    let _ = write!(h, "{f:?}");
    for t in summary_targets(f) {
        let _ = write!(
            h,
            "\n{t:x} {:?} {:?} {:?} {:?}",
            summaries.live_in(t),
            summaries.touched(t),
            summaries.written(t),
            summaries.live_in_state(t)
        );
    }
    h.finish()
}

/// Everything besides the functions that the IR of one shard depends on.
#[derive(Debug, Clone, Copy)]
pub struct ShardContext<'a> {
    pub target: &'a LiftTarget,
    pub memory: &'a MemoryModel,
    pub dispatch: Option<ShardDispatch<'a>>,
    pub probes: &'a BTreeSet<u64>,
}

/// Cache key of a shard: [`LIFT_CODEGEN`], the context, and per function (in order) its
/// [`function_key`], whether it is probed and whether it keeps external linkage.
pub fn shard_key(context: &ShardContext<'_>, functions: &[&Function], keys: &[u128]) -> u128 {
    debug_assert_eq!(functions.len(), keys.len());
    let mut h = Fnv128::new();
    let _ = write!(
        h,
        "playgta6-shard-lift v{SHARD_CACHE_VERSION}\ncodegen {LIFT_CODEGEN}\ntarget {:?}\nmemory {:?}\ndispatch {:?}\n",
        context.target,
        context.memory,
        context.dispatch.map(|d| d.dispatcher_symbol)
    );
    for (f, key) in functions.iter().zip(keys) {
        let probe = context.probes.contains(&f.entry);
        let external = context
            .dispatch
            .map(|d| d.external_entries.contains(&f.entry));
        let _ = writeln!(h, "{:x} {key:032x} {probe} {external:?}", f.entry);
    }
    h.finish()
}

/// SplitMix64 finalizer of a function entry: the anchor choice of [`plan_shards_stable`].
pub fn anchor_hash(entry: u64) -> u64 {
    let mut z = entry.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Whether a function at `entry` weighing `weight` instructions starts a shard in a plan whose
/// anchors are on average `spacing` instructions apart: each instruction is an anchor with
/// probability `1 / spacing`, decided by [`anchor_hash`] of the entry, so the choice does not
/// depend on the function size distribution or on any other function.
pub fn is_anchor(entry: u64, weight: usize, spacing: usize) -> bool {
    let spacing = spacing.max(1) as u64;
    let weight = (weight as u64).min(spacing);
    (anchor_hash(entry) & 0xffff_ffff) < (weight << 32) / spacing
}

/// Cuts functions (sorted by entry, `entries[i]` weighing `weights[i]` instructions) into
/// consecutive shards of at most `max_functions` functions and `max_instructions` instructions
/// (a heavier function gets a shard of its own), like [`super::plan_shards`], but also cuts before
/// every anchor ([`is_anchor`], on average every `max_instructions / 2` instructions). An anchor
/// depends only on its own function, and a cut before it does not depend on where the shard
/// started, so the plans before and after a change agree again from the first anchor past it:
/// changing, adding or removing a function moves only the boundaries between the anchors around
/// it (one or two shards, a few when the gap is longer than the budget and split by it). Shards
/// average about half the budget (more, shorter shards than the greedy plan's).
pub fn plan_shards_stable(
    entries: &[u64],
    weights: &[usize],
    max_functions: usize,
    max_instructions: usize,
) -> Vec<Range<usize>> {
    assert_eq!(entries.len(), weights.len());
    let spacing = (max_instructions / 2).max(1);
    let mut shards = Vec::new();
    let (mut start, mut total) = (0, 0usize);
    for (i, (&entry, &weight)) in entries.iter().zip(weights).enumerate() {
        let len = i - start;
        let forced = len >= max_functions || total + weight > max_instructions;
        if len > 0 && (forced || is_anchor(entry, weight, spacing)) {
            shards.push(start..i);
            (start, total) = (i, 0);
        }
        total += weight;
    }
    if start < entries.len() {
        shards.push(start..entries.len());
    }
    shards
}

/// What a cache entry records besides the IR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedShard {
    pub size: u64,
    pub hash: u64,
    pub unsupported: BTreeMap<String, usize>,
}

impl CachedShard {
    fn render(&self) -> String {
        let mut out = format!(
            "playgta6-shard-lift-meta {SHARD_CACHE_VERSION}\nsize {}\nfnv1a64 {:016x}\n",
            self.size, self.hash
        );
        for (mnemonic, count) in &self.unsupported {
            let _ = writeln!(out, "unsupported {mnemonic} {count}");
        }
        out
    }

    fn parse(text: &str) -> Option<Self> {
        let mut lines = text.lines();
        if lines.next()? != format!("playgta6-shard-lift-meta {SHARD_CACHE_VERSION}") {
            return None;
        }
        let size = lines.next()?.strip_prefix("size ")?.parse().ok()?;
        let hash = u64::from_str_radix(lines.next()?.strip_prefix("fnv1a64 ")?, 16).ok()?;
        let mut unsupported = BTreeMap::new();
        for line in lines {
            let mut parts = line.strip_prefix("unsupported ")?.rsplitn(2, ' ');
            let count = parts.next()?.parse().ok()?;
            unsupported.insert(parts.next()?.to_string(), count);
        }
        Some(Self {
            size,
            hash,
            unsupported,
        })
    }
}

/// The shard lift cache directory (see the module docs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardCache {
    pub dir: PathBuf,
}

fn key_name(key: u128) -> String {
    format!("{key:032x}")
}

fn link_or_copy(src: &Path, dst: &Path) -> io::Result<()> {
    if fs::hard_link(src, dst).is_err() {
        fs::copy(src, dst)?;
    }
    Ok(())
}

impl ShardCache {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    fn paths(&self, key: u128) -> (PathBuf, PathBuf) {
        let name = key_name(key);
        (
            self.dir.join(format!("{name}.ll")),
            self.dir.join(format!("{name}.meta")),
        )
    }

    /// Links the cached IR of `key` to `dest` (replacing it) and returns its record, or `None` on
    /// a miss or an incomplete entry.
    pub fn restore(&self, key: u128, dest: &Path) -> Option<CachedShard> {
        let (ll, meta) = self.paths(key);
        let record = CachedShard::parse(&fs::read_to_string(&meta).ok()?)?;
        if fs::metadata(&ll).ok()?.len() != record.size {
            return None;
        }
        match fs::remove_file(dest) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(_) => return None,
        }
        link_or_copy(&ll, dest).ok()?;
        if fs::metadata(dest).ok()?.len() != record.size {
            let _ = fs::remove_file(dest);
            return None;
        }
        // Most recent use first when trimming.
        let now = SystemTime::now();
        for path in [&ll, &meta] {
            if let Ok(file) = fs::File::options().append(true).open(path) {
                let _ = file.set_modified(now);
            }
        }
        Some(record)
    }

    /// Stores the freshly written shard IR `src` under `key`. Failures are ignored (the cache
    /// is an optimisation; the build output is already complete).
    pub fn store(&self, key: u128, src: &Path, record: &CachedShard) {
        let _ = self.try_store(key, src, record);
    }

    fn try_store(&self, key: u128, src: &Path, record: &CachedShard) -> io::Result<()> {
        fs::create_dir_all(&self.dir)?;
        let (ll, meta) = self.paths(key);
        let name = key_name(key);
        let tag = format!(
            "tmp-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        )
        .replace(|c: char| !c.is_ascii_alphanumeric() && c != '-', "");
        let tmp_ll = self.dir.join(format!("{name}.{tag}.ll.tmp"));
        let tmp_meta = self.dir.join(format!("{name}.{tag}.meta.tmp"));
        link_or_copy(src, &tmp_ll)?;
        fs::rename(&tmp_ll, &ll).inspect_err(|_| {
            let _ = fs::remove_file(&tmp_ll);
        })?;
        fs::write(&tmp_meta, record.render())?;
        fs::rename(&tmp_meta, &meta).inspect_err(|_| {
            let _ = fs::remove_file(&tmp_meta);
        })?;
        Ok(())
    }
}

/// Lifts (or, with a cache hit, links in) shard `index` holding `functions` and writes it to
/// `lift_dir`; with `tool`, compiles it. The second value is whether the IR came from `cache`.
#[allow(clippy::too_many_arguments)]
pub(super) fn lift_shard(
    context: &ShardContext<'_>,
    flag_summaries: &FlagSummaries,
    cache: Option<&ShardCache>,
    index: usize,
    functions: &[&Function],
    lift_dir: &Path,
    tool: Option<&LlvmTool>,
) -> Result<(ShardRecord, bool), PipelineError> {
    let ll_name = shard_file_name(index, "ll");
    let ll_path = lift_dir.join(&ll_name);
    let instructions = functions.iter().map(|f| f.instruction_count()).sum();
    let key = cache.map(|_| {
        let keys: Vec<u128> = functions
            .iter()
            .map(|f| function_key(f, flag_summaries))
            .collect();
        shard_key(context, functions, &keys)
    });
    let cached = match (cache, key) {
        (Some(cache), Some(key)) => cache.restore(key, &ll_path),
        _ => None,
    };
    let hit = cached.is_some();
    let (ll, unsupported) = if let Some(record) = cached {
        let ll = OutputFile {
            path: format!("{LIFT_DIR}/{ll_name}"),
            size: record.size,
            hash: record.hash,
        };
        (ll, record.unsupported)
    } else {
        let (ll, unsupported) =
            lift_and_write(context, flag_summaries, functions, &ll_name, &ll_path)?;
        if let (Some(cache), Some(key)) = (cache, key) {
            let record = CachedShard {
                size: ll.size,
                hash: ll.hash,
                unsupported: unsupported.clone(),
            };
            cache.store(key, &ll_path, &record);
        }
        (ll, unsupported)
    };
    let obj = tool
        .map(|tool| {
            compile(
                tool,
                context.target,
                lift_dir,
                &ll_name,
                &shard_file_name(index, "o"),
            )
        })
        .transpose()?;
    Ok((
        ShardRecord {
            index,
            first: functions.first().map_or(0, |f| f.entry),
            last: functions.last().map_or(0, |f| f.entry),
            functions: functions.len(),
            instructions,
            unsupported,
            ll,
            obj,
        },
        hit,
    ))
}

fn lift_and_write(
    context: &ShardContext<'_>,
    flag_summaries: &FlagSummaries,
    functions: &[&Function],
    ll_name: &str,
    ll_path: &Path,
) -> Result<(OutputFile, BTreeMap<String, usize>), PipelineError> {
    let mut lifted = Vec::with_capacity(functions.len());
    for function in functions {
        let mut one =
            lift::lift_function_in(function, *context.memory, flag_summaries).map_err(|error| {
                PipelineError::Lift {
                    entry: function.entry,
                    error,
                }
            })?;
        if context.probes.contains(&function.entry) {
            one.insert_entry_probe();
        }
        lifted.push(one);
    }
    let mut unsupported = BTreeMap::new();
    for u in lifted.iter().flat_map(|f| f.unsupported.iter()) {
        *unsupported.entry(u.mnemonic.clone()).or_insert(0) += 1;
    }
    let ir = lift::emit_module_with_dispatch(&lifted, context.dispatch)
        .map(|ir| context.target.retarget(ir))
        .map_err(|error| PipelineError::Lift {
            entry: lifted.first().map_or(0, |f| f.entry),
            error,
        })?;
    drop(lifted);
    // Never write through a hard link into the cache.
    match fs::remove_file(ll_path) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => {
            return Err(io_error(format!(
                "failed to replace '{}'",
                ll_path.display()
            ))(e));
        }
    }
    fs::write(ll_path, &ir)
        .map_err(io_error(format!("failed to write '{}'", ll_path.display())))?;
    let ll = OutputFile {
        path: format!("{LIFT_DIR}/{ll_name}"),
        size: ir.len() as u64,
        hash: image::fnv1a_64(ir.as_bytes()),
    };
    Ok((ll, unsupported))
}
