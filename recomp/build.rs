//! Exposes `PLAYGTA6_LIFT_CODEGEN`: a content hash of every source file that decides how a
//! discovered function is turned into LLVM IR (the lifter `src/lift/**` without its tests, the
//! CFG types `src/cfg.rs`, the shard emission `src/pipeline/shard_lift.rs`, this script, the crate
//! manifest and the workspace `Cargo.lock` that pins iced-x86). It is part of every key of the
//! shard lift cache (`pipeline::shard_lift`), so a change to any of these files re-lifts every
//! shard, while a change anywhere else (discovery, jump tables, reports, the CLI) only re-lifts the
//! shards whose functions' CFGs changed.

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
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    walk(&root.join("src").join("lift"), &mut files);
    for extra in [
        root.join("src").join("cfg.rs"),
        root.join("src").join("pipeline").join("shard_lift.rs"),
        root.join("build.rs"),
        root.join("Cargo.toml"),
        root.join("..").join("Cargo.lock"),
    ] {
        if extra.is_file() {
            files.push(extra);
        }
    }
    let mut named: Vec<(String, PathBuf)> = files
        .into_iter()
        .map(|p| {
            let rel = p
                .strip_prefix(&root)
                .unwrap_or(&p)
                .to_string_lossy()
                .replace('\\', "/");
            (rel, p)
        })
        .collect();
    named.sort();
    // FNV-1a 128 (the same parameters as `shard_lift::Fnv128`).
    let mut h: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
    let mut feed = |bytes: &[u8]| {
        for &b in bytes {
            h ^= u128::from(b);
            h = h.wrapping_mul(0x0000_0000_0100_0000_0000_0000_0000_013b);
        }
    };
    for (rel, path) in &named {
        let bytes = fs::read(path).unwrap_or_default();
        // Line endings depend on the checkout (autocrlf), not on the code.
        let text: Vec<u8> = bytes.into_iter().filter(|&b| b != b'\r').collect();
        feed(format!("{rel}\0{}\0", text.len()).as_bytes());
        feed(&text);
        println!("cargo:rerun-if-changed={}", path.display());
    }
    println!(
        "cargo:rerun-if-changed={}",
        root.join("src").join("lift").display()
    );
    println!("cargo:rustc-env=PLAYGTA6_LIFT_CODEGEN={h:032x}");
}
