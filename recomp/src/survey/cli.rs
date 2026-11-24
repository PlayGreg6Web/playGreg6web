//! `playgta6-recomp survey-interfaces`: argument parsing and the end-to-end run.
//!
//! The output contains addresses, GUID constants, source labels, import names and counts only, so
//! it may be printed or written anywhere.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use super::report::{render_json, render_text};
use super::{Config, Env, Source, parse_hex, parse_source_spec, survey};
use crate::discover::discover;
use crate::image::{self, ImageOptions};
use crate::pe::Pe64;
use crate::pipeline::{self, refine};

pub const USAGE: &str = "survey-interfaces <pe-path> --source <spec>... [--guard-slot <va>]... \
[--max-depth N] [--examples N] [--no-refine] [--json] [--out <file>]";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    pub pe_path: PathBuf,
    pub sources: Vec<Source>,
    pub guard_slots: Vec<u64>,
    pub max_depth: u8,
    pub examples: usize,
    /// Plain discovery instead of the code-pointer refinement of `build`.
    pub no_refine: bool,
    pub json: bool,
    pub out: Option<PathBuf>,
}

/// Parses the arguments after `survey-interfaces`. `Ok(None)` means help was requested.
pub fn parse_args(args: &[String]) -> Result<Option<Args>, String> {
    let defaults = Config::default();
    let mut out = Args {
        pe_path: PathBuf::new(),
        sources: Vec::new(),
        guard_slots: Vec::new(),
        max_depth: defaults.max_depth,
        examples: defaults.examples,
        no_refine: false,
        json: false,
        out: None,
    };
    let mut pe_path = None;
    let mut i = 0;
    let value = |i: &mut usize, flag: &str| -> Result<String, String> {
        *i += 1;
        match args.get(*i) {
            Some(v) if !v.starts_with("--") => Ok(v.clone()),
            _ => Err(format!("{flag} requires a value")),
        }
    };
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => return Ok(None),
            "--json" => out.json = true,
            "--no-refine" => out.no_refine = true,
            "--source" => {
                let spec = value(&mut i, "--source")?;
                out.sources.push(parse_source_spec(&spec)?);
            }
            "--guard-slot" => {
                let v = value(&mut i, "--guard-slot")?;
                out.guard_slots.push(parse_hex(&v)?);
            }
            "--max-depth" => {
                let v = value(&mut i, "--max-depth")?;
                out.max_depth = v
                    .parse()
                    .map_err(|_| format!("--max-depth expects a number, got '{v}'"))?;
            }
            "--examples" => {
                let v = value(&mut i, "--examples")?;
                out.examples = v
                    .parse()
                    .map_err(|_| format!("--examples expects a number, got '{v}'"))?;
            }
            "--out" => out.out = Some(PathBuf::from(value(&mut i, "--out")?)),
            flag if flag.starts_with('-') => {
                return Err(format!("unknown option '{flag}' for 'survey-interfaces'"));
            }
            positional => {
                if pe_path.is_some() {
                    return Err(format!(
                        "unexpected extra positional argument '{positional}'"
                    ));
                }
                pe_path = Some(PathBuf::from(positional));
            }
        }
        i += 1;
    }
    out.pe_path = pe_path.ok_or_else(|| "missing input PE path <pe-path>".to_string())?;
    if out.sources.is_empty() {
        return Err("at least one --source is required".to_string());
    }
    Ok(Some(out))
}

/// Runs discovery and the survey; returns the rendered report. Timing goes to stderr.
pub fn run(args: &Args) -> Result<String, String> {
    let path = &args.pe_path;
    let bytes =
        fs::read(path).map_err(|e| format!("failed to read PE file '{}': {e}", path.display()))?;
    let pe = Pe64::parse(&bytes)
        .map_err(|e| format!("failed to parse PE64 '{}': {e}", path.display()))?;
    let started = Instant::now();
    let mut seeded = pipeline::seed(&pe, &bytes);
    let analysis = if args.no_refine {
        discover(&seeded.image)
    } else {
        let guest = image::build_guest_image(&pe, &bytes, ImageOptions::default())
            .map_err(|e| format!("failed to build the guest image: {e:?}"))?;
        refine::discover_refined(&mut seeded.image, &guest.image).0
    };
    eprintln!(
        "discovered {} functions in {:.1}s",
        analysis.functions.len(),
        started.elapsed().as_secs_f64()
    );
    let config = Config {
        sources: args.sources.clone(),
        guard_slots: args.guard_slots.clone(),
        max_depth: args.max_depth,
        examples: args.examples,
    };
    let env = Env::from_pe(&pe);
    let t = Instant::now();
    let result = survey(&env, &seeded.image, &analysis, &config).map_err(|e| e.to_string())?;
    eprintln!("survey took {:.1}s", t.elapsed().as_secs_f64());
    let text = if args.json {
        render_json(&result)
    } else {
        render_text(&result)
    };
    match &args.out {
        Some(file) => {
            if let Some(parent) = file.parent()
                && !parent.as_os_str().is_empty()
            {
                fs::create_dir_all(parent).map_err(|e| {
                    format!(
                        "failed to create output directory '{}': {e}",
                        parent.display()
                    )
                })?;
            }
            fs::write(file, &text)
                .map_err(|e| format!("failed to write '{}': {e}", file.display()))?;
            Ok(format!("Wrote interface survey to {}\n", file.display()))
        }
        None => Ok(text),
    }
}
