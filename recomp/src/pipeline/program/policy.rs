//! Import binding policy of a program build: which side satisfies an import when both a guest
//! module and the runtime's host shims could. See the parent module docs for the table.

use std::fmt;

use super::link::canonical_dll;

/// What to bind an import of a matching DLL symbol to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Action {
    /// The export of the guest module (when the DLL is a guest module of the program).
    Guest,
    /// The host shim bridge, even when a guest module exports the symbol.
    Host,
    /// The host shim when the bridge binds it to a shim, otherwise the guest export.
    PreferHost,
}

impl Action {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Guest => "guest",
            Self::Host => "host",
            Self::PreferHost => "prefer-host",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "guest" => Some(Self::Guest),
            "host" => Some(Self::Host),
            "prefer-host" => Some(Self::PreferHost),
            _ => None,
        }
    }
}

/// Whether a supplied DLL may become a guest module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Role {
    /// Built and linked as a guest module when the program references it.
    Guest,
    /// Never built: every import of it goes to the host shims.
    HostOnly,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Guest => "guest",
            Self::HostOnly => "host",
        }
    }
}

/// Where a rule comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Source {
    /// The built-in table ([`DEFAULT_RULES`], [`DEFAULT_ROLES`]).
    Default,
    /// `--bind`, `--guest-dll`, `--host-dll` or a `--policy` file.
    Override,
}

/// One symbol rule: `dll!symbol=action`. Patterns use `*` as a wildcard; DLL patterns match
/// the canonical (lower-case, `.dll`-suffixed) name, symbol patterns match the export name
/// case-sensitively (`#<n>` matches an ordinal import).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rule {
    pub dll: String,
    pub symbol: String,
    pub action: Action,
    pub source: Source,
    /// Must stay on the guest side for correctness (overriding it is reported).
    pub required: bool,
}

impl fmt::Display for Rule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}!{}={}", self.dll, self.symbol, self.action.as_str())
    }
}

/// One DLL role rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoleRule {
    pub dll: String,
    pub role: Role,
    pub source: Source,
}

/// Stateless memory/string primitives `VCRUNTIME140` exports: safe on either side.
const VCRUNTIME_PRIMITIVES: &[&str] = &[
    "memcpy", "memmove", "memset", "memcmp", "memchr", "strchr", "strrchr", "strstr", "wcschr",
    "wcsrchr", "wcsstr",
];

/// C++ exception handling, `setjmp`/`longjmp`, RTTI and the other `VCRUNTIME140*` exports whose
/// behaviour depends on the guest's own data layouts (`ThrowInfo`, `FuncInfo`, `type_info`,
/// `jmp_buf`, `std::exception`) and on R11's guest unwinding design.
const VCRUNTIME_REQUIRED: &[&str] = &[
    "_CxxThrowException",
    "__CxxFrameHandler*",
    "__C_specific_handler*",
    "__intrinsic_setjmp*",
    "_setjmp*",
    "longjmp",
    "__RTDynamicCast",
    "__RTtypeid",
    "__RTCastToVoid",
    "__std_exception_*",
    "__std_type_info_*",
    "__current_exception*",
    "__uncaught_exception*",
    "__processing_throw",
    "__std_terminate",
    "_purecall",
    "_set_purecall_handler",
    "__DestructExceptionObject",
    "__BuildCatchObject*",
    "__TypeMatch",
    "__AdjustPointer",
    "__FrameUnwindFilter",
    "_CreateFrameInfo",
    "_FindAndUnlinkFrame",
    "_IsExceptionObjectToBeDestroyed",
    "__NLG_*",
    "_local_unwind",
];

/// DLLs whose exports default to the guest side: the C++ runtime redistributables and the UCRT.
const GUEST_FAMILIES: &[&str] = &[
    "msvcp140*.dll",
    "vcruntime140*.dll",
    "concrt140.dll",
    "vccorlib140.dll",
    "vcomp140.dll",
    "ucrtbase.dll",
];

/// Middleware the runtime deliberately replaces with host shims (O14 offline/WebGPU stubs):
/// networking (`XCurl`, `libHttpClient.GDK`), video (`Bink2Scarlett`), and the GPU paths
/// (`XboxBvhBuilder`, AMD FidelityFX).
pub const DEFAULT_HOST_ONLY: &[&str] = &[
    "bink2scarlett.dll",
    "xcurl.dll",
    "libhttpclient.gdk.dll",
    "xboxbvhbuilder.dll",
    "amd_fidelityfx_loader_xbox.dll",
    "amd_fidelityfx_upscaler_xbox.dll",
];

/// The built-in symbol rules, most specific first (the first match wins).
pub fn default_rules() -> Vec<Rule> {
    let mut rules = Vec::new();
    for &symbol in VCRUNTIME_PRIMITIVES {
        rules.push(rule("vcruntime140.dll", symbol, Action::PreferHost, false));
    }
    for &symbol in VCRUNTIME_REQUIRED {
        rules.push(rule("vcruntime140*.dll", symbol, Action::Guest, true));
    }
    for &dll in GUEST_FAMILIES {
        rules.push(rule(dll, "*", Action::Guest, false));
    }
    rules.push(rule("*", "*", Action::Guest, false));
    rules
}

fn rule(dll: &str, symbol: &str, action: Action, required: bool) -> Rule {
    Rule {
        dll: dll.to_string(),
        symbol: symbol.to_string(),
        action,
        source: Source::Default,
        required,
    }
}

/// The built-in DLL roles (every DLL not listed is [`Role::Guest`]).
pub fn default_roles() -> Vec<RoleRule> {
    DEFAULT_HOST_ONLY
        .iter()
        .map(|&dll| RoleRule {
            dll: dll.to_string(),
            role: Role::HostOnly,
            source: Source::Default,
        })
        .collect()
}

/// Heap functions that must resolve to one allocator across all modules: a block allocated by
/// one side and freed by the other corrupts both heaps.
const ALLOCATOR_SYMBOLS: &[&str] = &[
    "malloc",
    "free",
    "calloc",
    "realloc",
    "_recalloc",
    "_msize",
    "_expand",
    "_aligned_*",
    "_malloc_base",
    "_free_base",
    "_calloc_base",
    "_realloc_base",
    "_recalloc_base",
    "_msize_base",
    "_expand_base",
    "??2@*",
    "??3@*",
    "??_U@*",
    "??_V@*",
];

/// DLLs (canonical names, after API-set mapping) whose allocator symbols form the family.
const ALLOCATOR_DLLS: &[&str] = &[
    "ucrtbase.dll",
    "msvcrt.dll",
    "vcruntime140*.dll",
    "msvcp140*.dll",
    "concrt140.dll",
];

/// Whether an import of `symbol` from `dll` (canonical, API sets mapped) belongs to the
/// allocator family.
pub fn is_allocator_symbol(dll: &str, symbol: &str) -> bool {
    ALLOCATOR_DLLS.iter().any(|p| glob(p, dll)) && ALLOCATOR_SYMBOLS.iter().any(|p| glob(p, symbol))
}

/// The policy of one program build: overrides first, then the defaults.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Policy {
    pub rules: Vec<Rule>,
    pub roles: Vec<RoleRule>,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            rules: default_rules(),
            roles: default_roles(),
        }
    }
}

impl Policy {
    /// The defaults with `overrides` taking precedence.
    pub fn with_overrides(overrides: &Overrides) -> Self {
        let mut rules = overrides.rules.clone();
        rules.extend(default_rules());
        let mut roles = overrides.roles.clone();
        roles.extend(default_roles());
        Self { rules, roles }
    }

    /// The first rule matching `symbol` (`#<n>` for an ordinal import) of `dll` (canonical).
    pub fn rule_for(&self, dll: &str, symbol: &str) -> &Rule {
        self.rules
            .iter()
            .find(|r| glob(&r.dll, dll) && glob(&r.symbol, symbol))
            .expect("the default table ends with *!*")
    }

    /// The default rule that would apply without overrides, when it is a required one.
    pub fn required_default(&self, dll: &str, symbol: &str) -> Option<&Rule> {
        self.rules
            .iter()
            .filter(|r| r.source == Source::Default)
            .find(|r| glob(&r.dll, dll) && glob(&r.symbol, symbol))
            .filter(|r| r.required)
    }

    /// Role of a supplied DLL (canonical name) and the rule source that decided it.
    pub fn role_of(&self, dll: &str) -> (Role, Source) {
        self.roles
            .iter()
            .find(|r| glob(&r.dll, dll))
            .map_or((Role::Guest, Source::Default), |r| (r.role, r.source))
    }
}

/// Policy overrides from the command line or a policy file.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Overrides {
    pub rules: Vec<Rule>,
    pub roles: Vec<RoleRule>,
    /// DLLs to build as guest modules even when nothing imports them (`--guest-dll`).
    pub force_guest: Vec<String>,
}

impl Overrides {
    /// Adds `dll!symbol=action`.
    pub fn add_bind(&mut self, spec: &str) -> Result<(), String> {
        let (target, action) = spec.rsplit_once('=').ok_or_else(|| {
            format!("bad binding rule '{spec}' (expected <dll>!<symbol>=<action>)")
        })?;
        let (dll, symbol) = target.split_once('!').ok_or_else(|| {
            format!("bad binding rule '{spec}' (expected <dll>!<symbol>=<action>)")
        })?;
        let action = Action::parse(action.trim()).ok_or_else(|| {
            format!("bad action in '{spec}' (expected guest, host or prefer-host)")
        })?;
        let (dll, symbol) = (dll.trim(), symbol.trim());
        if dll.is_empty() || symbol.is_empty() {
            return Err(format!(
                "bad binding rule '{spec}': empty DLL or symbol pattern"
            ));
        }
        self.rules.push(Rule {
            dll: canonical_pattern(dll),
            symbol: symbol.to_string(),
            action,
            source: Source::Override,
            required: false,
        });
        Ok(())
    }

    /// `--guest-dll <name>`: build it as a guest module (even when unreferenced).
    pub fn add_guest_dll(&mut self, dll: &str) -> Result<(), String> {
        let name = checked_dll(dll)?;
        self.roles.push(RoleRule {
            dll: name.clone(),
            role: Role::Guest,
            source: Source::Override,
        });
        self.force_guest.push(name);
        Ok(())
    }

    /// `--host-dll <name>`: never build it; bind its imports to host shims.
    pub fn add_host_dll(&mut self, dll: &str) -> Result<(), String> {
        let name = checked_dll(dll)?;
        self.roles.push(RoleRule {
            dll: name,
            role: Role::HostOnly,
            source: Source::Override,
        });
        Ok(())
    }

    /// Reads a policy file: one directive per line, `#` starts a comment.
    /// `<dll>!<symbol>=<action>` adds a binding rule, `<dll>=guest` / `<dll>=host` sets a role.
    pub fn add_file(&mut self, text: &str) -> Result<(), String> {
        for (number, line) in text.lines().enumerate() {
            let line = line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let result = if line.contains('!') {
                self.add_bind(line)
            } else {
                match line.split_once('=').map(|(d, r)| (d.trim(), r.trim())) {
                    Some((dll, "guest")) => self.add_guest_dll(dll),
                    Some((dll, "host")) => self.add_host_dll(dll),
                    _ => Err(format!("unknown directive '{line}'")),
                }
            };
            result.map_err(|e| format!("policy line {}: {e}", number + 1))?;
        }
        Ok(())
    }
}

fn checked_dll(dll: &str) -> Result<String, String> {
    let name = canonical_dll(dll.trim());
    if name.is_empty() || name.contains('*') {
        return Err(format!("bad DLL name '{dll}'"));
    }
    Ok(name)
}

/// Canonical form of a DLL pattern (wildcards kept).
fn canonical_pattern(dll: &str) -> String {
    if dll.contains('*') {
        dll.to_ascii_lowercase()
    } else {
        canonical_dll(dll)
    }
}

/// `*`-wildcard match (no other metacharacters).
pub fn glob(pattern: &str, text: &str) -> bool {
    let p = pattern.as_bytes();
    let t = text.as_bytes();
    let (mut pi, mut ti) = (0, 0);
    let mut star: Option<(usize, usize)> = None;
    while ti < t.len() {
        if pi < p.len() && p[pi] == b'*' {
            star = Some((pi, ti));
            pi += 1;
        } else if pi < p.len() && p[pi] == t[ti] {
            pi += 1;
            ti += 1;
        } else if let Some((sp, st)) = star {
            pi = sp + 1;
            ti = st + 1;
            star = Some((sp, st + 1));
        } else {
            return false;
        }
    }
    p[pi..].iter().all(|&c| c == b'*')
}
