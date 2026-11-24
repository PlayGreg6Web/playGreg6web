//! Code-generation target of lifted modules.
//!
//! The lifter always emits the same IR (see the module docs of [`super`]); only three lines depend
//! on where the code will run, and [`LiftTarget::retarget`] rewrites exactly those:
//!
//! | Line | [`LiftTarget::Wasm64`] (default) | [`LiftTarget::Native`] |
//! | :--- | :--- | :--- |
//! | `target datalayout` | [`WASM64_DATALAYOUT`](super::WASM64_DATALAYOUT) | removed: LLVM derives the layout from the triple |
//! | `target triple` | [`WASM64_TRIPLE`](super::WASM64_TRIPLE) | the native triple (e.g. `x86_64-pc-windows-msvc`) |
//! | `attributes #0` | wasm `target-features` (`+simd128`, atomics, bulk memory) | `uwtable "target-cpu"="<cpu>"` |
//!
//! The only wasm-specific construct in lifted IR besides those lines is the `llvm.wasm.swizzle`
//! intrinsic (lifted `pshufb`); a native module defines an internal `@__playgta6_swizzle` with the
//! same semantics on top of `llvm.x86.ssse3.pshuf.b.128` and calls that instead.
//!
//! The wasm feature strings mean nothing to the x86 backend, so a native module drops them and
//! names a CPU instead ([`NATIVE_DEFAULT_CPU`], `x86-64-v3` = AVX2/FMA/BMI2, which the lifted
//! 256-bit vector IR maps onto directly; any x86-64 CPU name LLVM knows can be given instead).
//! `uwtable` keeps unwind tables on every lifted function: the runtime stops a faulting guest by
//! unwinding through lifted frames, and the native harness walks them for host backtraces.
//!
//! The native target exists to run lifted code in an ordinary host process (the native first-boot
//! harness, `tools/build_native_boot.py`, and fast lifter tests). Like the browser build it relies
//! on the guest address space translation of `docs/wasm_abi.md`; the runtime places guest memory
//! so that lifted code's addresses resolve.

use std::fmt;

/// CPU of native modules unless another one is requested.
pub const NATIVE_DEFAULT_CPU: &str = "x86-64-v3";

/// Call/declaration marker of the one wasm-only intrinsic lifted SIMD code uses
/// (`i8x16.swizzle`, for `pshufb`).
const WASM_SWIZZLE_CALL: &str = "@llvm.wasm.swizzle(";
const NATIVE_SWIZZLE_CALL: &str = "@__playgta6_swizzle(";
/// Native replacement: wasm swizzle zeroes lanes whose index is >= 16, x86 `pshufb` lanes whose
/// index has bit 7 set; map the former onto the latter (needs SSSE3, part of every x86-64-v2+ CPU).
const NATIVE_SWIZZLE_IR: &str = "\
define internal <16 x i8> @__playgta6_swizzle(<16 x i8> %a, <16 x i8> %idx) #0 {
  %big = icmp uge <16 x i8> %idx, <i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16>
  %sel = select <16 x i1> %big, <16 x i8> <i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128>, <16 x i8> %idx
  %r = call <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8> %a, <16 x i8> %sel)
  ret <16 x i8> %r
}
declare <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8>, <16 x i8>)
";

/// Where a lifted module will be compiled for.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum LiftTarget {
    /// `wasm64-unknown-unknown` (the browser build). The emitted IR is used unchanged.
    #[default]
    Wasm64,
    /// An x86-64 host triple, for running lifted code natively.
    Native {
        /// LLVM target triple (`x86_64-pc-windows-msvc`, `x86_64-unknown-linux-gnu`, ...).
        triple: String,
        /// LLVM CPU name (`target-cpu` attribute).
        cpu: String,
    },
}

impl LiftTarget {
    /// The native target of the machine running the recompiler, with [`NATIVE_DEFAULT_CPU`].
    pub fn native_host() -> Self {
        Self::Native {
            triple: host_triple().to_string(),
            cpu: NATIVE_DEFAULT_CPU.to_string(),
        }
    }

    /// Parses `wasm64`, `native` (host triple) or `native:<triple>`; `cpu` overrides the CPU of
    /// a native target (ignored for wasm64).
    pub fn parse(spec: &str, cpu: Option<&str>) -> Result<Self, String> {
        let target = match spec {
            "wasm64" => return Ok(Self::Wasm64),
            "native" => Self::native_host(),
            other => match other.strip_prefix("native:") {
                Some(triple) if is_x86_64_triple(triple) => Self::Native {
                    triple: triple.to_string(),
                    cpu: NATIVE_DEFAULT_CPU.to_string(),
                },
                Some(triple) => {
                    return Err(format!(
                        "native triple '{triple}' is not an x86_64 triple (lifted code assumes 64-bit pointers and x86 vector semantics)"
                    ));
                }
                None => {
                    return Err(format!(
                        "unknown target '{other}' (expected wasm64, native or native:<triple>)"
                    ));
                }
            },
        };
        match (target, cpu) {
            (Self::Native { triple, .. }, Some(cpu)) => {
                if cpu.is_empty()
                    || !cpu
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.')
                {
                    return Err(format!("invalid CPU name '{cpu}'"));
                }
                Ok(Self::Native {
                    triple,
                    cpu: cpu.to_string(),
                })
            }
            (target, _) => Ok(target),
        }
    }

    /// Whether this is a native (host) target.
    pub fn is_native(&self) -> bool {
        matches!(self, Self::Native { .. })
    }

    /// The LLVM target triple.
    pub fn triple(&self) -> &str {
        match self {
            Self::Wasm64 => super::WASM64_TRIPLE,
            Self::Native { triple, .. } => triple,
        }
    }

    /// Rewrites the target-dependent lines of an emitted module (see the module docs). For
    /// [`LiftTarget::Wasm64`] the text is returned unchanged.
    pub fn retarget(&self, ir: String) -> String {
        let Self::Native { triple, cpu } = self else {
            return ir;
        };
        let swizzle = ir.contains(WASM_SWIZZLE_CALL);
        let mut out = String::with_capacity(ir.len());
        for line in ir.split_inclusive('\n') {
            let body = line.trim_end_matches(['\n', '\r']);
            if body.starts_with("target datalayout = ") {
                continue;
            }
            if swizzle && body.contains(WASM_SWIZZLE_CALL) {
                if body.starts_with("declare ") {
                    out.push_str(NATIVE_SWIZZLE_IR);
                } else {
                    out.push_str(&line.replace(WASM_SWIZZLE_CALL, NATIVE_SWIZZLE_CALL));
                }
                continue;
            }
            if let Some(with_comdat) = linkonce_global_with_comdat(body) {
                out.push_str(&with_comdat);
                continue;
            }
            if body.starts_with("target triple = ") {
                out.push_str("target triple = \"");
                out.push_str(triple);
                out.push_str("\"\n");
            } else if body.starts_with("attributes #0 = ") {
                out.push_str("attributes #0 = { uwtable \"target-cpu\"=\"");
                out.push_str(cpu);
                out.push_str("\" }\n");
            } else {
                out.push_str(line);
            }
        }
        out
    }
}

impl fmt::Display for LiftTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wasm64 => f.write_str("wasm64"),
            Self::Native { triple, cpu } => write!(f, "native:{triple} ({cpu})"),
        }
    }
}

/// `@name = linkonce_odr ...` global definitions (the shared `cmpxchg16b` lock table) get a COMDAT
/// of their own: without one, COFF has no way to merge `linkonce_odr` data across objects (LLVM
/// emits a weak external with a per-object default, which `link.exe` rejects as conflicting), and
/// on ELF the COMDAT is what lets the linker keep one copy.
fn linkonce_global_with_comdat(body: &str) -> Option<String> {
    let rest = body.strip_prefix('@')?;
    let (name, definition) = rest.split_once(" = ")?;
    if !definition.starts_with("linkonce_odr ") || definition.contains("comdat") {
        return None;
    }
    let definition = match definition.rfind(", align ") {
        Some(at) => format!("{}, comdat{}", &definition[..at], &definition[at..]),
        None => format!("{definition}, comdat"),
    };
    Some(format!("${name} = comdat any\n@{name} = {definition}\n"))
}

/// LLVM triple of the host this binary was built for (x86_64 only; other hosts fall back to the
/// Linux triple, which is what a cross build would ask for).
pub fn host_triple() -> &'static str {
    if cfg!(all(windows, target_env = "msvc")) {
        "x86_64-pc-windows-msvc"
    } else if cfg!(all(windows, target_env = "gnu")) {
        "x86_64-pc-windows-gnu"
    } else if cfg!(target_os = "macos") {
        "x86_64-apple-darwin"
    } else {
        "x86_64-unknown-linux-gnu"
    }
}

fn is_x86_64_triple(triple: &str) -> bool {
    triple.starts_with("x86_64-") && triple.split('-').count() >= 3
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "; header\ntarget datalayout = \"e-m:e\"\ntarget triple = \"wasm64-unknown-unknown\"\n\ndefine void @sub_1(ptr %state) #0 {\n  ret void\n}\n\nattributes #0 = { \"target-features\"=\"+simd128\" }\n";

    #[test]
    fn wasm64_is_identity() {
        assert_eq!(LiftTarget::Wasm64.retarget(SAMPLE.to_string()), SAMPLE);
        assert_eq!(LiftTarget::default(), LiftTarget::Wasm64);
    }

    #[test]
    fn native_rewrites_only_target_lines() {
        let target = LiftTarget::parse("native:x86_64-unknown-linux-gnu", Some("znver2")).unwrap();
        let out = target.retarget(SAMPLE.to_string());
        assert_eq!(
            out,
            "; header\ntarget triple = \"x86_64-unknown-linux-gnu\"\n\ndefine void @sub_1(ptr %state) #0 {\n  ret void\n}\n\nattributes #0 = { uwtable \"target-cpu\"=\"znver2\" }\n"
        );
    }

    #[test]
    fn native_maps_the_wasm_swizzle() {
        let ir = format!(
            "{SAMPLE}declare <16 x i8> @llvm.wasm.swizzle(<16 x i8>, <16 x i8>)\n  %r = call <16 x i8> @llvm.wasm.swizzle(<16 x i8> %a, <16 x i8> %b)\n"
        );
        let out = LiftTarget::native_host().retarget(ir);
        assert!(!out.contains("llvm.wasm."));
        assert!(out.contains("define internal <16 x i8> @__playgta6_swizzle("));
        assert!(out.contains("call <16 x i8> @__playgta6_swizzle(<16 x i8> %a, <16 x i8> %b)"));
        assert!(out.contains("@llvm.x86.ssse3.pshuf.b.128"));
    }

    #[test]
    fn native_gives_linkonce_globals_a_comdat() {
        let ir = format!(
            "{SAMPLE}@__playgta6_cmpxchg16b_locks = linkonce_odr hidden global [64 x i32] zeroinitializer, align 4\n"
        );
        let out = LiftTarget::native_host().retarget(ir.clone());
        assert!(out.contains(
            "$__playgta6_cmpxchg16b_locks = comdat any\n@__playgta6_cmpxchg16b_locks = linkonce_odr hidden global [64 x i32] zeroinitializer, comdat, align 4\n"
        ));
        assert_eq!(LiftTarget::Wasm64.retarget(ir.clone()), ir);
        assert_eq!(
            LiftTarget::native_host().retarget(out.clone()),
            out,
            "idempotent"
        );
    }

    #[test]
    fn parse_targets() {
        assert_eq!(
            LiftTarget::parse("wasm64", Some("znver2")).unwrap(),
            LiftTarget::Wasm64
        );
        let host = LiftTarget::parse("native", None).unwrap();
        assert_eq!(host.triple(), host_triple());
        assert!(host.is_native());
        assert!(LiftTarget::parse("native:aarch64-unknown-linux-gnu", None).is_err());
        assert!(LiftTarget::parse("x86", None).is_err());
        assert!(LiftTarget::parse("native", Some("bad cpu")).is_err());
        assert_eq!(
            host.to_string(),
            format!("native:{} (x86-64-v3)", host_triple())
        );
    }
}
