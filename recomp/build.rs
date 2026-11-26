//! Exposes two source content hashes, each part of a build cache key:
//!
//! - `PLAYGTA6_LIFT_CODEGEN`: every source file that decides how a discovered function is turned
//!   into LLVM IR (the lifter `src/lift/**` without its tests, the CFG types `src/cfg.rs`, the
//!   shard emission `src/pipeline/shard_lift.rs`, this script, the crate manifest and the
//!   workspace `Cargo.lock` that pins iced-x86). It is part of every key of the shard lift cache
//!   (`pipeline::shard_lift`), so a change to any of these files re-lifts every shard, while a
//!   change anywhere else (discovery, jump tables, reports, the CLI) only re-lifts the shards
//!   whose functions' CFGs changed.
//! - `PLAYGTA6_DISCOVERY_CODE`: every non-test source file under `src/` except the lifter
//!   (`src/lift/**`) and the shard emission (`src/pipeline/shard_lift.rs`), which run after
//!   discovery and cannot change it, plus this script, the manifest and `Cargo.lock`. It keys the
//!   discovery cache (`pipeline::discovery_cache`): deliberately coarse (any change that might
//!   reach discovery, the jump-table rules, the seeding or the refinement re-runs it), so a lifter
//!   edit skips discovery and nothing else does. The `rustc -vV` output is mixed in as well.

use std::fs;
use std::path::{Path, PathBuf};

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // Test modules never change generated code.
            if path.file_name().is_some_and(|n| n == "tests") {
                continue;
            }
            walk(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs")
            // `tests.rs` files are `#[cfg(test)] mod tests;` modules.
            && path.file_name().is_none_or(|n| n != "tests.rs")
        {
            out.push(path);
        }
    }
}

/// FNV-1a 128 (the same parameters as `shard_lift::Fnv128`) over `files` sorted by their path
/// relative to `root`; each file contributes its relative path, its length and its text without
/// `\r` (line endings depend on the checkout, autocrlf, not on the code).
fn content_hash(root: &Path, files: &[PathBuf]) -> u128 {
    let mut named: Vec<(String, &PathBuf)> = files
        .iter()
        .map(|p| {
            let rel = p
                .strip_prefix(root)
                .unwrap_or(p)
                .to_string_lossy()
                .replace('\\', "/");
            (rel, p)
        })
        .collect();
    named.sort();
    named.dedup_by(|a, b| a.0 == b.0);
    let mut h: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
    let mut feed = |bytes: &[u8]| {
        for &b in bytes {
            h ^= u128::from(b);
            h = h.wrapping_mul(0x0000_0000_0100_0000_0000_0000_0000_013b);
        }
    };
    for (rel, path) in &named {
        let bytes = fs::read(path).unwrap_or_default();
        let text: Vec<u8> = bytes.into_iter().filter(|&b| b != b'\r').collect();
        feed(format!("{rel}\0{}\0", text.len()).as_bytes());
        feed(&text);
    }
    h
}

/// FNV-1a 128 of `bytes`.
fn content_hash_bytes(bytes: &[u8]) -> u128 {
    let mut h: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
    for &b in bytes {
        h ^= u128::from(b);
        h = h.wrapping_mul(0x0000_0000_0100_0000_0000_0000_0000_013b);
    }
    h
}

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let src = root.join("src");
    let common: Vec<PathBuf> = [
        root.join("build.rs"),
        root.join("Cargo.toml"),
        root.join("..").join("Cargo.lock"),
    ]
    .into_iter()
    .filter(|p| p.is_file())
    .collect();

    let mut lift = Vec::new();
    walk(&src.join("lift"), &mut lift);
    lift.extend(
        [
            src.join("cfg.rs"),
            src.join("pipeline").join("shard_lift.rs"),
        ]
        .into_iter()
        .filter(|p| p.is_file()),
    );
    lift.extend(common.iter().cloned());

    let mut all = Vec::new();
    walk(&src, &mut all);
    let lift_dir = src.join("lift");
    let shard_lift = src.join("pipeline").join("shard_lift.rs");
    let mut discovery: Vec<PathBuf> = all
        .into_iter()
        .filter(|p| !p.starts_with(&lift_dir) && *p != shard_lift)
        .collect();
    discovery.extend(common.iter().cloned());

    for path in lift.iter().chain(&discovery) {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    // A directory makes cargo rescan it, so added files are seen too.
    println!("cargo:rerun-if-changed={}", src.display());
    println!(
        "cargo:rustc-env=PLAYGTA6_LIFT_CODEGEN={:032x}",
        content_hash(&root, &lift)
    );
    // The compiler is an input too (the discovery cache outlives a toolchain update).
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let version = std::process::Command::new(rustc)
        .arg("-vV")
        .output()
        .map(|o| o.stdout)
        .unwrap_or_default();
    println!("cargo:rerun-if-env-changed=RUSTC");
    let discovery_code = content_hash(&root, &discovery) ^ content_hash_bytes(&version);
    println!("cargo:rustc-env=PLAYGTA6_DISCOVERY_CODE={discovery_code:032x}");
}
