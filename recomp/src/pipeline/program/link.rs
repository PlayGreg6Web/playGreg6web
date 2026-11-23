//! Module names, API sets, export tables and forwarder resolution of a program build.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::pe::{IMAGE_SCN_MEM_EXECUTE, ImportedSymbol, Pe64};

/// API-set contract prefix -> host module, longest prefix wins. Must stay identical to the rules
/// `ModuleRegistry`'s constructor installs in `runtime/os/win32_loader.cpp` (a test compares
/// them).
pub const API_SET_RULES: &[(&str, &str)] = &[
    ("api-ms-win-core-", "kernel32.dll"),
    ("api-ms-win-core-com-", "combase.dll"),
    ("api-ms-win-core-winrt-", "combase.dll"),
    ("api-ms-win-crt-", "ucrtbase.dll"),
];

/// Windows-style canonical module name: basename, ASCII-lowercased, `.dll` appended when the
/// name has no extension (a trailing `.` means "no extension"); the same rule as the runtime's
/// `canonical_module_name`.
pub fn canonical_dll(name: &str) -> String {
    let base = name.rsplit(['\\', '/']).next().unwrap_or(name);
    if base.is_empty() {
        return String::new();
    }
    let mut out = base.to_ascii_lowercase();
    if out.ends_with('.') {
        out.pop();
    } else if !out.contains('.') {
        out.push_str(".dll");
    }
    out
}

/// Whether a canonical name is an API-set contract (`api-*` / `ext-*`).
pub fn is_api_set(canonical: &str) -> bool {
    canonical.len() > 4 && (canonical.starts_with("api-") || canonical.starts_with("ext-"))
}

/// Host module of an API-set contract ([`API_SET_RULES`]), `None` for other names and for
/// contracts no rule covers.
pub fn api_set_host(canonical: &str) -> Option<&'static str> {
    if !is_api_set(canonical) {
        return None;
    }
    API_SET_RULES
        .iter()
        .filter(|(prefix, _)| canonical.starts_with(prefix))
        .max_by_key(|(prefix, _)| prefix.len())
        .map(|&(_, host)| host)
}

/// Build module name for a DLL file: the lower-cased file stem with every character outside
/// `[a-z0-9_]` replaced by `_` (`libHttpClient.GDK.dll` -> `libhttpclient_gdk`).
pub fn module_symbol(canonical: &str) -> String {
    let stem = canonical.strip_suffix(".dll").unwrap_or(canonical);
    let mut out: String = stem
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    out.truncate(64);
    out
}

/// `#<ordinal>` or the name.
pub fn symbol_text(symbol: &ImportedSymbol) -> String {
    match symbol {
        ImportedSymbol::ByName { name, .. } => name.clone(),
        ImportedSymbol::ByOrdinal { ordinal } => format!("#{ordinal}"),
    }
}

/// Function or data export.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExportKind {
    /// In an executable section: a lifted function entry.
    Function,
    /// Anywhere else: a variable, vtable or other object (never lifted).
    Data,
}

impl ExportKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Function => "function",
            Self::Data => "data",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ExportDef {
    Address { rva: u32, kind: ExportKind },
    Forward(String),
}

/// Export counts of one module.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExportSummary {
    /// Export address table entries (distinct ordinals).
    pub total: usize,
    pub functions: usize,
    pub data: usize,
    pub forwarders: usize,
    /// Entries with no name.
    pub ordinal_only: usize,
}

/// Export table of one guest module, at its assigned base.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleExports {
    pub base: u64,
    by_name: HashMap<String, ExportDef>,
    by_ordinal: BTreeMap<u32, ExportDef>,
    pub summary: ExportSummary,
}

impl ModuleExports {
    pub fn from_pe(pe: &Pe64) -> Self {
        let base = pe.optional.image_base;
        let kind_of = |rva: u32| {
            let executable = pe.sections.iter().any(|s| {
                let size = if s.virtual_size == 0 {
                    s.raw_size
                } else {
                    s.virtual_size
                };
                rva >= s.virtual_address
                    && rva - s.virtual_address < size
                    && s.characteristics & IMAGE_SCN_MEM_EXECUTE != 0
            });
            if executable {
                ExportKind::Function
            } else {
                ExportKind::Data
            }
        };
        let mut by_name = HashMap::new();
        let mut by_ordinal = BTreeMap::new();
        let mut summary = ExportSummary::default();
        let mut named_ordinals = BTreeSet::new();
        for export in pe.exports.iter().flat_map(|t| t.symbols.iter()) {
            let def = match &export.forwarder {
                Some(target) => ExportDef::Forward(target.clone()),
                None => ExportDef::Address {
                    rva: export.rva,
                    kind: kind_of(export.rva),
                },
            };
            if let Some(name) = &export.name {
                by_name.insert(name.clone(), def.clone());
                named_ordinals.insert(export.ordinal);
            }
            if by_ordinal.insert(export.ordinal, def.clone()).is_none() {
                summary.total += 1;
                match def {
                    ExportDef::Forward(_) => summary.forwarders += 1,
                    ExportDef::Address {
                        kind: ExportKind::Function,
                        ..
                    } => summary.functions += 1,
                    ExportDef::Address {
                        kind: ExportKind::Data,
                        ..
                    } => summary.data += 1,
                }
            }
        }
        summary.ordinal_only = by_ordinal
            .keys()
            .filter(|o| !named_ordinals.contains(o))
            .count();
        Self {
            base,
            by_name,
            by_ordinal,
            summary,
        }
    }

    fn lookup(&self, symbol: &ImportedSymbol) -> Option<&ExportDef> {
        match symbol {
            ImportedSymbol::ByName { name, .. } => self.by_name.get(name),
            ImportedSymbol::ByOrdinal { ordinal } => self.by_ordinal.get(&u32::from(*ordinal)),
        }
    }
}

/// Where an import ends up after export lookup and forwarders.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Provider {
    /// An export of a guest module (`dll` canonical).
    Guest {
        dll: String,
        va: u64,
        kind: ExportKind,
    },
    /// A DLL that is not a guest module of the program (`dll` as spelled by the importer or the
    /// forwarder, so the host bridge sees the usual name).
    Host { dll: String, symbol: ImportedSymbol },
}

/// Why an export lookup failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LookupError {
    /// The guest module has no such export.
    MissingExport { dll: String, symbol: String },
    /// A forwarder chain returns to an export it already visited.
    ForwarderCycle(Vec<String>),
    /// A forwarder string is not `DLL.Name` / `DLL.#ordinal`.
    BadForwarder(String),
}

impl LookupError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::MissingExport { .. } => "missing_export",
            Self::ForwarderCycle(_) => "forwarder_cycle",
            Self::BadForwarder(_) => "bad_forwarder",
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Self::MissingExport { dll, symbol } => format!("{dll} does not export {symbol}"),
            Self::ForwarderCycle(chain) => format!("forwarder cycle {}", chain.join(" -> ")),
            Self::BadForwarder(text) => format!("malformed forwarder '{text}'"),
        }
    }
}

/// Longest forwarder chain followed before giving up (a cycle is detected before that).
const MAX_FORWARDS: usize = 32;

/// Splits a forwarder string `DLL.Name` / `DLL.#ordinal` at its last `.`.
pub fn parse_forwarder(text: &str) -> Option<(String, ImportedSymbol)> {
    let (dll, name) = text.rsplit_once('.')?;
    if dll.is_empty() || name.is_empty() {
        return None;
    }
    let symbol = match name.strip_prefix('#') {
        Some(digits) => ImportedSymbol::ByOrdinal {
            ordinal: digits.parse().ok()?,
        },
        None => ImportedSymbol::ByName {
            hint: 0,
            name: name.to_string(),
        },
    };
    Some((dll.to_string(), symbol))
}

/// The export tables of every guest module, keyed by canonical DLL name.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExportIndex {
    pub modules: BTreeMap<String, ModuleExports>,
}

impl ExportIndex {
    /// The guest module a DLL name refers to: itself, or the host module of its API set.
    pub fn guest_module(&self, dll: &str) -> Option<String> {
        let canonical = canonical_dll(dll);
        if self.modules.contains_key(&canonical) {
            return Some(canonical);
        }
        api_set_host(&canonical)
            .filter(|host| self.modules.contains_key(*host))
            .map(str::to_string)
    }

    /// Resolves `symbol` of `dll`, following forwarders across guest modules. `via` receives
    /// every forwarder followed (`DLL.Name` strings).
    pub fn resolve(
        &self,
        dll: &str,
        symbol: &ImportedSymbol,
        via: &mut Vec<String>,
    ) -> Result<Provider, LookupError> {
        let mut dll = dll.to_string();
        let mut symbol = symbol.clone();
        let mut seen: Vec<String> = Vec::new();
        loop {
            let Some(module) = self.guest_module(&dll) else {
                return Ok(Provider::Host { dll, symbol });
            };
            let key = format!("{module}!{}", symbol_text(&symbol));
            if seen.contains(&key) || seen.len() > MAX_FORWARDS {
                seen.push(key);
                return Err(LookupError::ForwarderCycle(seen));
            }
            seen.push(key);
            let exports = &self.modules[&module];
            match exports.lookup(&symbol) {
                None => {
                    return Err(LookupError::MissingExport {
                        dll: module,
                        symbol: symbol_text(&symbol),
                    });
                }
                Some(ExportDef::Address { rva, kind }) => {
                    return Ok(Provider::Guest {
                        dll: module,
                        va: exports.base + u64::from(*rva),
                        kind: *kind,
                    });
                }
                Some(ExportDef::Forward(text)) => {
                    let (target, next) = parse_forwarder(text)
                        .ok_or_else(|| LookupError::BadForwarder(text.clone()))?;
                    via.push(text.clone());
                    dll = target;
                    symbol = next;
                }
            }
        }
    }
}

/// DLL names a PE depends on: normal and delay-load imports in table order, then forwarder
/// targets of its exports (sorted), each canonical and listed once.
pub fn dependencies(pe: &Pe64) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let imports = pe
        .imports
        .iter()
        .map(|m| m.dll_name.as_str())
        .chain(pe.delay_imports.iter().map(|m| m.dll_name.as_str()));
    for dll in imports {
        let canonical = canonical_dll(dll);
        if !out.contains(&canonical) {
            out.push(canonical);
        }
    }
    let forwards: BTreeSet<String> = pe
        .exports
        .iter()
        .flat_map(|t| t.symbols.iter())
        .filter_map(|e| e.forwarder.as_deref())
        .filter_map(parse_forwarder)
        .map(|(dll, _)| canonical_dll(&dll))
        .collect();
    for dll in forwards {
        if !out.contains(&dll) {
            out.push(dll);
        }
    }
    out
}
