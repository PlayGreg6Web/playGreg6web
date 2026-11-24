//! Catalogue-wide export thunks: guest-callable entries for every shim the runtime exports.
//!
//! The import bridge ([`super::render_bridge`]) emits a marshaling thunk only for the IAT slots
//! of one module. Guest code that resolves an API dynamically (`GetModuleHandle` +
//! `GetProcAddress`) needs a guest-callable entry for exports nobody imports statically. This
//! module plans and renders that table for **every export the runtime registers in its module
//! registry** (the `(DLL, export name, ordinal, symbol)` tuples [`super::tables::scan_tables`]
//! reads from the runtime's export tables), binding each one under exactly the rules a static
//! import of the same DLL and name follows ([`ShimIndex::resolve`]): the family table, the
//! authoritative export tables, name keys, the shim audit of audited families (only `ok` binds)
//! and marshalability.
//!
//! # Per-export outcome ([`ExportBinding`])
//! * `generated`: a marshaling thunk is emitted (the same code a static import gets,
//!   [`super::render_bound_thunk`]); one thunk per distinct shim, shared by every DLL spelling of
//!   the export (`KERNEL32` / `KERNELBASE` / API sets that resolve to the same shim).
//! * `gthunk`: a `pg6_gthunk_<name>` guest-state thunk binds first (the same precedence as for
//!   static imports) and is listed as is, without a generated wrapper.
//! * `none`: no guest-callable entry; the row carries a stable reason code
//!   ([`super::Unresolved::code`]: `denied`, `pending_audit`, `unmarshalable`, `ambiguous`, ...,
//!   plus `shim_mismatch` and `undeclared` for table symbols the bridge cannot call) and the
//!   human-readable detail the runtime shows when the guest asks for such an export.
//!
//! # Output (`pg6_export_thunks.{h,cpp}`, `pg6_export_manifest.json`, `pg6_guest_seam.h`)
//! The source defines `extern const Pg6ExportThunk* const pg6_export_thunks` and
//! `extern const std::size_t pg6_export_thunk_count`: one row per export, sorted by DLL (key:
//! lowercase, no `.dll`), then name, then ordinal: `{dll, name, ordinal (0 = none), flags
//! (kPg6ExportGenerated / kPg6ExportGuestState), shim, thunk (null = no guest-callable entry),
//! reason (null unless thunk is null)}`. Thunks are internal to the translation unit, named
//! `pg6_export_<shim>` and never exported, so the table links next to any number of per-module
//! import bridges (`pg6_import_<module>_<n>`, also internal) without a symbol clash, whatever
//! they import. The table has `Pg6BootExport`'s layout (`guest/export_thunks.h`); the runtime
//! registers each row's thunk for the proc address its module registry holds for that export
//! (`guest::register_export_thunks`), so `GetProcAddress` can hand out a callable address.
//! Output is deterministic: identical inputs give byte-identical files.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::lift::MemoryModel;
use crate::pe::ImportedSymbol;

use super::{
    Catalogue, Resolution, SEAM_FILE, ShimDecl, ShimIndex, Unresolved, comment_text, cpp_str,
    dll_key, is_guest_state_thunk, json_str, render_bound_thunk, render_seam_for,
};

/// Generated export table header.
pub const EXPORT_HEADER_FILE: &str = "pg6_export_thunks.h";
/// Generated export table source.
pub const EXPORT_SOURCE_FILE: &str = "pg6_export_thunks.cpp";
/// Export manifest.
pub const EXPORT_MANIFEST_FILE: &str = "pg6_export_manifest.json";

/// How one export is served to a guest caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportBinding {
    /// A generated marshaling thunk calls `shim`.
    Generated { shim: String },
    /// The guest-state thunk `thunk` serves it directly.
    GuestState { thunk: String },
    /// No guest-callable entry.
    None {
        /// Stable code (see the module docs).
        code: String,
        /// Human-readable reason.
        detail: String,
    },
}

/// One registered export.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportRow {
    /// DLL as spelled by the first table entry of this export.
    pub dll: String,
    pub name: String,
    pub ordinal: Option<u16>,
    /// The symbol(s) the runtime tables register for it (sorted, distinct).
    pub symbols: Vec<String>,
    pub binding: ExportBinding,
}

impl ExportRow {
    /// `DLL!name`.
    pub fn label(&self) -> String {
        format!("{}!{}", self.dll, self.name)
    }

    /// The shim or guest-state thunk the row binds, if any.
    pub fn callee(&self) -> Option<&str> {
        match &self.binding {
            ExportBinding::Generated { shim } => Some(shim),
            ExportBinding::GuestState { thunk } => Some(thunk),
            ExportBinding::None { .. } => None,
        }
    }

    fn kind(&self) -> &'static str {
        match self.binding {
            ExportBinding::Generated { .. } => "generated",
            ExportBinding::GuestState { .. } => "gthunk",
            ExportBinding::None { .. } => "none",
        }
    }
}

/// Every registered export with its binding, sorted by `(dll key, name, ordinal)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportPlan {
    pub rows: Vec<ExportRow>,
}

/// Per-DLL tallies of an [`ExportPlan`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExportCounts {
    pub generated: usize,
    pub gthunk: usize,
    pub none: usize,
}

impl ExportPlan {
    /// The plan restricted to the exports `selectors` name (`DLL!export`, DLL compared like an
    /// import's: case-insensitive, `.dll` optional, export name exact), in plan order. For slim
    /// tables and checked-in test fixtures; a selector that matches nothing is an error.
    pub fn only(&self, selectors: &[String]) -> Result<ExportPlan, String> {
        let mut wanted: Vec<(String, &str, bool)> = Vec::new();
        for selector in selectors {
            let (dll, name) = selector
                .split_once('!')
                .filter(|(dll, name)| !dll.is_empty() && !name.is_empty())
                .ok_or_else(|| format!("export selector '{selector}' is not <DLL>!<name>"))?;
            wanted.push((dll_key(dll), name, false));
        }
        let rows: Vec<ExportRow> = self
            .rows
            .iter()
            .filter(|row| {
                let mut keep = false;
                for (dll, name, used) in &mut wanted {
                    if *dll == dll_key(&row.dll) && *name == row.name {
                        *used = true;
                        keep = true;
                    }
                }
                keep
            })
            .cloned()
            .collect();
        if let Some((dll, name, _)) = wanted.iter().find(|(_, _, used)| !used) {
            return Err(format!("no registered export {dll}!{name}"));
        }
        Ok(ExportPlan { rows })
    }

    /// Rows with a guest-callable entry (generated or gthunk).
    pub fn callable_count(&self) -> usize {
        self.rows.iter().filter(|r| r.callee().is_some()).count()
    }

    /// Distinct shims that get a generated thunk, sorted.
    pub fn generated_shims(&self) -> BTreeSet<&str> {
        self.rows
            .iter()
            .filter_map(|r| match &r.binding {
                ExportBinding::Generated { shim } => Some(shim.as_str()),
                _ => None,
            })
            .collect()
    }

    /// Distinct callees (generated shims and guest-state thunks): the number of guest-callable
    /// entries (synthetic guest addresses) the table needs at most.
    pub fn distinct_callees(&self) -> usize {
        self.rows
            .iter()
            .filter_map(ExportRow::callee)
            .collect::<BTreeSet<_>>()
            .len()
    }

    /// `(dll, counts)` per DLL (spellings merged under the first seen), sorted by DLL key.
    pub fn counts_by_dll(&self) -> Vec<(String, ExportCounts)> {
        let mut by_dll: BTreeMap<String, (String, ExportCounts)> = BTreeMap::new();
        for row in &self.rows {
            let (_, counts) = by_dll
                .entry(dll_key(&row.dll))
                .or_insert_with(|| (row.dll.clone(), ExportCounts::default()));
            match row.binding {
                ExportBinding::Generated { .. } => counts.generated += 1,
                ExportBinding::GuestState { .. } => counts.gthunk += 1,
                ExportBinding::None { .. } => counts.none += 1,
            }
        }
        by_dll.into_values().collect()
    }

    /// Rows without a guest-callable entry per reason code, sorted.
    pub fn none_by_reason(&self) -> BTreeMap<&str, usize> {
        let mut out = BTreeMap::new();
        for row in &self.rows {
            if let ExportBinding::None { code, .. } = &row.binding {
                *out.entry(code.as_str()).or_default() += 1;
            }
        }
        out
    }
}

/// Plans a thunk for every export the catalogue's export tables register.
pub fn plan_exports(catalogue: &Catalogue) -> ExportPlan {
    let index = ShimIndex::new(catalogue);
    // One row per (DLL key, export name); ordinal and symbols merged across table entries.
    struct Group {
        dll: String,
        ordinal: Option<u16>,
        symbols: BTreeSet<String>,
    }
    let mut groups: BTreeMap<(String, String), Group> = BTreeMap::new();
    for entry in &catalogue.exports {
        if entry.name.is_empty() {
            continue;
        }
        let group = groups
            .entry((dll_key(&entry.dll), entry.name.clone()))
            .or_insert_with(|| Group {
                dll: entry.dll.clone(),
                ordinal: None,
                symbols: BTreeSet::new(),
            });
        group.ordinal = group.ordinal.or(entry.ordinal);
        group.symbols.insert(entry.symbol.clone());
    }

    let rows = groups
        .into_iter()
        .map(|((_, name), group)| {
            let symbols: Vec<String> = group.symbols.into_iter().collect();
            let binding = bind_export(catalogue, &index, &group.dll, &name, &symbols);
            ExportRow {
                dll: group.dll,
                name,
                ordinal: group.ordinal,
                symbols,
                binding,
            }
        })
        .collect();
    ExportPlan { rows }
}

fn bind_export(
    catalogue: &Catalogue,
    index: &ShimIndex<'_>,
    dll: &str,
    name: &str,
    symbols: &[String],
) -> ExportBinding {
    let none = |why: &Unresolved| ExportBinding::None {
        code: why.code().to_string(),
        detail: why.describe(),
    };
    if let Some(missing) = symbols.iter().find(|s| catalogue.get(s).is_none()) {
        return ExportBinding::None {
            code: "undeclared".to_string(),
            detail: format!("{missing} is registered but not declared in the scanned headers"),
        };
    }
    let symbol = ImportedSymbol::ByName {
        hint: 0,
        name: name.to_string(),
    };
    match index.resolve(dll, &symbol) {
        Resolution::Unresolved(why) => none(&why),
        Resolution::Bound { shim } => match catalogue.get(&shim) {
            None => ExportBinding::None {
                code: "undeclared".to_string(),
                detail: format!("{shim} missing from catalogue"),
            },
            Some(decl) if is_guest_state_thunk(decl) => ExportBinding::GuestState { thunk: shim },
            // The runtime registers `symbols` under this export: a thunk for any other shim would
            // not be what the registry's host proc address stands for.
            Some(_) if !symbols.contains(&shim) => ExportBinding::None {
                code: "shim_mismatch".to_string(),
                detail: format!(
                    "the bridge binds {shim} but the runtime registers {}",
                    symbols.join(", ")
                ),
            },
            Some(_) => ExportBinding::Generated { shim },
        },
    }
}

/// The generated files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportOutput {
    pub header: String,
    pub source: String,
    pub seam: String,
    pub manifest: String,
}

impl ExportOutput {
    /// Write all files into `dir` (created if missing). Returns the written paths.
    pub fn write_to(&self, dir: &Path) -> io::Result<Vec<PathBuf>> {
        fs::create_dir_all(dir)?;
        let files = [
            (EXPORT_HEADER_FILE, &self.header),
            (EXPORT_SOURCE_FILE, &self.source),
            (SEAM_FILE, &self.seam),
            (EXPORT_MANIFEST_FILE, &self.manifest),
        ];
        let mut written = Vec::with_capacity(files.len());
        for (name, text) in files {
            let path = dir.join(name);
            fs::write(&path, text)?;
            written.push(path);
        }
        Ok(written)
    }
}

/// Renders the table source, header, default seam and manifest for `plan`.
pub fn render_exports(plan: &ExportPlan, catalogue: &Catalogue) -> ExportOutput {
    render_exports_for(plan, catalogue, &MemoryModel::default())
}

/// [`render_exports`] for code lifted with `model` (see [`super::render_bridge_for`]).
pub fn render_exports_for(
    plan: &ExportPlan,
    catalogue: &Catalogue,
    model: &MemoryModel,
) -> ExportOutput {
    let window = matches!(model, MemoryModel::Windowed(_));
    ExportOutput {
        header: render_header(),
        source: render_source(plan, catalogue, window),
        seam: render_seam_for(model),
        manifest: render_manifest(plan, catalogue),
    }
}

/// Internal name of the generated thunk of `shim`.
fn thunk_name(shim: &str) -> String {
    format!("pg6_export_{}", shim.strip_prefix("pg6_").unwrap_or(shim))
}

const HEADER_TYPES: &str = "\
using Pg6ExportThunkFn = void (*)(void* state);

inline constexpr uint32_t kPg6ExportGenerated = 1u;   // thunk is a generated marshaling thunk
inline constexpr uint32_t kPg6ExportGuestState = 2u;  // thunk is a pg6_gthunk_* guest-state thunk

struct Pg6ExportThunk {
    const char* dll;
    const char* name;
    uint32_t ordinal;  // 0 = none
    uint32_t flags;
    const char* shim;  // shim or guest-state thunk symbol; null when there is none
    Pg6ExportThunkFn thunk;  // null: no guest-callable entry
    const char* reason;      // set when thunk is null
};
";

fn render_header() -> String {
    let mut out = String::new();
    out.push_str("// Generated by playgta6-recomp export-thunks. Do not edit.\n");
    out.push_str(
        "//
// Guest-callable thunks for every shim the runtime exports, for guest code that resolves an API
// with GetProcAddress (see pg6_guest_seam.h; the table has Pg6BootExport's layout).
#pragma once

#include <cstddef>
#include <cstdint>

#ifndef PG6_EXPORT_THUNK_TYPES_DEFINED
#define PG6_EXPORT_THUNK_TYPES_DEFINED
",
    );
    out.push_str(HEADER_TYPES);
    out.push_str(
        "#endif

// One row per export, sorted by DLL (lowercase, no .dll), name and ordinal.
extern const Pg6ExportThunk* const pg6_export_thunks;
extern const std::size_t pg6_export_thunk_count;
",
    );
    out
}

fn render_source(plan: &ExportPlan, catalogue: &Catalogue, window: bool) -> String {
    let mut out = String::new();
    out.push_str("// Generated by playgta6-recomp export-thunks. Do not edit.\n");
    let counts = plan.counts_by_dll();
    let (generated, gthunk, none) = counts.iter().fold((0, 0, 0), |a, (_, c)| {
        (a.0 + c.generated, a.1 + c.gthunk, a.2 + c.none)
    });
    let _ = writeln!(
        out,
        "// {} exports: {generated} generated, {gthunk} guest-state, {none} without a guest-callable \
         entry; {} distinct thunks.",
        plan.rows.len(),
        plan.distinct_callees()
    );
    let _ = writeln!(out, "#include \"{EXPORT_HEADER_FILE}\"");
    out.push('\n');
    for system in ["atomic", "cstddef", "cstdint", "cstring", "type_traits"] {
        let _ = writeln!(out, "#include <{system}>");
    }
    out.push('\n');
    out.push_str("#ifdef PG6_GUEST_SEAM_HEADER\n#include PG6_GUEST_SEAM_HEADER\n#else\n");
    let _ = writeln!(out, "#include \"{SEAM_FILE}\"\n#endif");

    let includes: BTreeSet<&str> = plan
        .rows
        .iter()
        .filter_map(|r| catalogue.get(r.callee()?))
        .map(|s| s.include.as_str())
        .collect();
    if !includes.is_empty() {
        out.push('\n');
        for include in includes {
            let _ = writeln!(out, "#include {}", cpp_str(include));
        }
    }
    out.push('\n');
    out.push_str(super::SOURCE_HELPERS);

    let shims: Vec<(&str, &ShimDecl)> = plan
        .generated_shims()
        .into_iter()
        .filter_map(|s| catalogue.get(s).map(|d| (s, d)))
        .collect();
    for (shim, decl) in &shims {
        let _ = writeln!(out, "// -> {shim}");
        render_bound_thunk(&mut out, &thunk_name(shim), decl, window);
        out.push('\n');
    }

    if plan.rows.is_empty() {
        let _ = writeln!(
            out,
            "}}  // namespace\n\nconst Pg6ExportThunk* const pg6_export_thunks = nullptr;"
        );
    } else {
        out.push_str("const Pg6ExportThunk kExportThunks[] = {\n");
        for row in &plan.rows {
            let (flags, shim, thunk, reason) = match &row.binding {
                ExportBinding::Generated { shim } => (
                    "kPg6ExportGenerated",
                    cpp_str(shim),
                    format!("&{}", thunk_name(shim)),
                    "nullptr".to_string(),
                ),
                ExportBinding::GuestState { thunk } => (
                    "kPg6ExportGuestState",
                    cpp_str(thunk),
                    format!("&{thunk}"),
                    "nullptr".to_string(),
                ),
                ExportBinding::None { code, detail } => (
                    "0u",
                    "nullptr".to_string(),
                    "nullptr".to_string(),
                    cpp_str(&format!("{code}: {detail}")),
                ),
            };
            let _ = writeln!(
                out,
                "    {{{}, {}, {}u, {flags}, {shim}, {thunk}, {reason}}},",
                cpp_str(&row.dll),
                cpp_str(&row.name),
                row.ordinal.unwrap_or(0)
            );
        }
        let _ = writeln!(
            out,
            "}};\n\n}}  // namespace\n\nconst Pg6ExportThunk* const pg6_export_thunks = kExportThunks;"
        );
    }
    let _ = writeln!(
        out,
        "const std::size_t pg6_export_thunk_count = {};",
        plan.rows.len()
    );
    out
}

fn render_manifest(plan: &ExportPlan, catalogue: &Catalogue) -> String {
    let mut out = String::new();
    out.push_str("{\n  \"format\": \"playgta6-export-thunks\",\n  \"version\": 1,\n");
    let _ = writeln!(
        out,
        "  \"catalogue\": {{\"headers\": {}, \"shims\": {}}},",
        catalogue.header_count,
        catalogue.len()
    );
    let counts = plan.counts_by_dll();
    let (generated, gthunk, none) = counts.iter().fold((0, 0, 0), |a, (_, c)| {
        (a.0 + c.generated, a.1 + c.gthunk, a.2 + c.none)
    });
    let _ = writeln!(
        out,
        "  \"summary\": {{\"exports\": {}, \"generated\": {generated}, \"gthunk\": {gthunk}, \
         \"none\": {none}, \"distinct_thunks\": {}}},",
        plan.rows.len(),
        plan.distinct_callees()
    );
    out.push_str("  \"reasons\": {");
    let reasons = plan.none_by_reason();
    for (i, (code, n)) in reasons.iter().enumerate() {
        let sep = if i == 0 { "" } else { "," };
        let _ = write!(out, "{sep}\n    {}: {n}", json_str(code));
    }
    out.push_str(if reasons.is_empty() {
        "},\n"
    } else {
        "\n  },\n"
    });
    out.push_str("  \"dlls\": [");
    for (i, (dll, c)) in counts.iter().enumerate() {
        let sep = if i == 0 { "" } else { "," };
        let _ = write!(
            out,
            "{sep}\n    {{\"dll\": {}, \"generated\": {}, \"gthunk\": {}, \"none\": {}}}",
            json_str(dll),
            c.generated,
            c.gthunk,
            c.none
        );
    }
    out.push_str(if counts.is_empty() {
        "],\n"
    } else {
        "\n  ],\n"
    });
    out.push_str("  \"exports\": [");
    for (i, row) in plan.rows.iter().enumerate() {
        let sep = if i == 0 { "" } else { "," };
        let ordinal = row
            .ordinal
            .map_or_else(|| "null".to_string(), |o| o.to_string());
        let symbols: Vec<String> = row.symbols.iter().map(|s| json_str(s)).collect();
        let _ = write!(
            out,
            "{sep}\n    {{\"dll\": {}, \"name\": {}, \"ordinal\": {ordinal}, \"symbols\": [{}], \
             \"kind\": \"{}\", ",
            json_str(&row.dll),
            json_str(&row.name),
            symbols.join(", "),
            row.kind()
        );
        match &row.binding {
            ExportBinding::Generated { shim } => {
                let _ = write!(
                    out,
                    "\"shim\": {}, \"thunk\": {}}}",
                    json_str(shim),
                    json_str(&thunk_name(shim))
                );
            }
            ExportBinding::GuestState { thunk } => {
                let _ = write!(
                    out,
                    "\"shim\": {}, \"thunk\": {}}}",
                    json_str(thunk),
                    json_str(thunk)
                );
            }
            ExportBinding::None { code, detail } => {
                let _ = write!(
                    out,
                    "\"shim\": null, \"thunk\": null, \"reason\": {}, \"detail\": {}}}",
                    json_str(code),
                    json_str(detail)
                );
            }
        }
    }
    out.push_str(if plan.rows.is_empty() {
        "]\n}\n"
    } else {
        "\n  ]\n}\n"
    });
    out
}

/// Plain-text summary: per-DLL counts and the reasons exports have no guest-callable entry.
pub fn render_export_coverage(plan: &ExportPlan, catalogue: &Catalogue) -> String {
    let mut out = String::new();
    let counts = plan.counts_by_dll();
    let _ = writeln!(
        out,
        "Export thunks: {} exports, {} guest-callable ({} distinct thunks), {} without ({} shims \
         from {} headers)",
        plan.rows.len(),
        plan.callable_count(),
        plan.distinct_callees(),
        plan.rows.len() - plan.callable_count(),
        catalogue.len(),
        catalogue.header_count
    );
    let width = counts
        .iter()
        .map(|(d, _)| d.len())
        .max()
        .unwrap_or(3)
        .max(3);
    let _ = writeln!(
        out,
        "  {:<width$}  {:>9}  {:>6}  {:>5}",
        "DLL", "generated", "gthunk", "none"
    );
    for (dll, c) in &counts {
        let _ = writeln!(
            out,
            "  {dll:<width$}  {:>9}  {:>6}  {:>5}",
            c.generated, c.gthunk, c.none
        );
    }
    let reasons = plan.none_by_reason();
    if !reasons.is_empty() {
        out.push_str("Without a guest-callable entry:\n");
        for (code, n) in reasons {
            let _ = writeln!(out, "  {code}: {n}");
        }
    }
    for note in &catalogue.diagnostics {
        let _ = writeln!(out, "Note: {}", comment_text(note));
    }
    out
}

#[cfg(test)]
mod tests;
