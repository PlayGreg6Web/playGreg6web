//! Win64 exception-handling metadata: the language-specific data behind `UNWIND_INFO`.
//!
//! A `.pdata` record whose `UNWIND_INFO` has `UNW_FLAG_EHANDLER`/`UNW_FLAG_UHANDLER` names a
//! language handler (an RVA of code) followed by handler-specific data (see [`crate::pdata`]).
//! The handler RVA never points at the handler itself but at code inside the image:
//!
//! - an import thunk `jmp [rip+slot]` whose IAT slot is a known language handler import
//!   (`VCRUNTIME140.dll!__C_specific_handler`, `__CxxFrameHandler3`,
//!   `VCRUNTIME140_1.dll!__CxxFrameHandler4`), or
//! - a static wrapper linked into the image (MSVC's `__GSHandlerCheck_SEH`/`_EH`/`_EH4`) that
//!   checks the stack cookie and then calls such a thunk with the same dispatcher context, so the
//!   data has the inner handler's format; or
//! - `__GSHandlerCheck`, a wrapper with no inner handler (cookie check only, no language data).
//!   It is recognised because it calls the same cookie-check helper as the identified wrappers.
//!
//! [`EhMetadata::from_pe`] identifies every handler that way (bounded decoding of the handler
//! code, no discovery needed) and parses the data of the three supported formats:
//!
//! - `__C_specific_handler`: an inline SCOPE_TABLE, `u32 Count` then `Count` records
//!   `{BeginAddress, EndAddress, HandlerAddress, JumpTarget}` (RVAs). `JumpTarget == 0` is a
//!   `__finally` whose termination handler is `HandlerAddress`; otherwise `JumpTarget` is the
//!   `__except` body inside the function and `HandlerAddress` is the filter funclet, or the
//!   constant `1` (`EXCEPTION_EXECUTE_HANDLER`).
//! - `__CxxFrameHandler3`: the data is the RVA of a `FuncInfo` (`magicNumber:29, bbtFlags:3` with
//!   magic `0x19930520..=0x19930522`, `maxState`, unwind map, try-block map with handler arrays,
//!   IP-to-state map, unwind-help offset, ES type list (>= `..21`), EH flags (>= `..22`); all
//!   tables are RVAs of fixed-size records). Catch funclets return the continuation address in
//!   `rax` (`lea rax, [continuation]`), so FH3 has no continuation addresses in metadata: discovery
//!   finds them as address-taken interior code (see [`crate::discover`]).
//! - `__CxxFrameHandler4`: the data is the RVA of a compressed `FuncInfo4`. A header byte
//!   (`isCatch`, `isSeparated`, `BBT`, `UnwindMap`, `TryBlockMap`, `EHs`, `NoExcept`) says which
//!   fields follow: `bbtFlags` (compressed), `dispUnwindMap`/`dispTryBlockMap` (raw `i32` RVAs),
//!   `dispIPtoStateMap` (raw; for separated code the RVA of a `{segment RVA, map RVA}` table) and,
//!   for catch funclets, `dispFrame` (compressed). The unwind map holds
//!   `nextOffset:30, type:2` entries (`type` 1/2 = destructor RVA + compressed object offset,
//!   3 = cleanup funclet RVA); try blocks hold compressed `tryLow, tryHigh, catchHigh` plus the RVA
//!   of a handler array; each handler has a header byte (`adjectives`, `dispType`,
//!   `dispCatchObj`, `contIsRVA`, `contAddr:2`), optional fields, the catch funclet RVA and 0..2
//!   continuation addresses (function-relative compressed values, or raw RVAs with `contIsRVA`).
//!   IP-to-state entries are compressed `(ip delta, state + 1)` pairs relative to the function
//!   start. Compressed unsigned integers store their length in the low bits of the first byte
//!   (`...0` 1 byte / 7 bits, `..01` 2 / 14, `.011` 3 / 21, `0111` 4 / 28, `1111` 5 bytes, value
//!   in the last four).
//!
//! Every read is bounds-checked against the PE's section mapping and every count is capped;
//! malformed or unrecognised data becomes an [`EhDiagnostic`], never a panic.
//!
//! # What discovery gets
//! [`EhMetadata::seed`] feeds [`CodeImage::eh`]: funclets (catch handlers, cleanup funclets,
//! destructors, filters, termination handlers) become entry points (most already are, through
//! `.pdata`), and **resume targets** — FH4 catch continuations and `__except` bodies, the places
//! where `RtlUnwindEx`/`RtlRestoreContext` resume execution inside a function — are recorded with
//! their owning function (the `.pdata` function containing them) so discovery makes them block
//! leaders and lists them in [`crate::cfg::Function::resume_targets`]. The IAT slots of the
//! `setjmp` family are recorded as well: the instruction after a call through them is a resume
//! target of the caller (`longjmp` returns there).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use iced_x86::{Decoder, DecoderOptions, FlowControl, Instruction, OpKind};

use crate::code_image::{CodeImage, Region};
use crate::pdata::ExceptionDirectory;
use crate::pe::{ImportedSymbol, Pe64};

/// Language-handler imports and the data format they consume.
const HANDLER_IMPORTS: [(&str, HandlerKind); 6] = [
    ("__C_specific_handler", HandlerKind::CSpecific),
    ("__C_specific_handler_noexcept", HandlerKind::CSpecific),
    ("__CxxFrameHandler", HandlerKind::CxxFrameHandler3),
    ("__CxxFrameHandler2", HandlerKind::CxxFrameHandler3),
    ("__CxxFrameHandler3", HandlerKind::CxxFrameHandler3),
    ("__CxxFrameHandler4", HandlerKind::CxxFrameHandler4),
];

/// Imports whose call returns a second time when `longjmp` restores the saved context.
pub const SETJMP_IMPORTS: [&str; 5] = [
    "_setjmp",
    "_setjmpex",
    "setjmp",
    "__intrinsic_setjmp",
    "__intrinsic_setjmpex",
];

/// `FuncInfo` magic numbers of `__CxxFrameHandler3` (`magicNumber:29`).
const FH3_MAGIC_MIN: u32 = 0x1993_0520;
const FH3_MAGIC_MAX: u32 = 0x1993_0522;
const FH3_MAGIC_MASK: u32 = 0x1FFF_FFFF;
/// `HandlerAddress` of an `__except (EXCEPTION_EXECUTE_HANDLER)` scope.
const EXCEPTION_EXECUTE_HANDLER: u32 = 1;

/// Upper bound for any table count (scope records, states, try blocks, handlers, IP map entries,
/// separated segments). Real functions stay far below; garbage counts are rejected.
const MAX_TABLE_ENTRIES: u32 = 1 << 16;
/// Instructions decoded per handler when identifying it.
const HANDLER_SCAN_LIMIT: usize = 96;

// FH4 header bits.
const FH4_IS_CATCH: u8 = 0x01;
const FH4_IS_SEPARATED: u8 = 0x02;
const FH4_BBT: u8 = 0x04;
const FH4_UNWIND_MAP: u8 = 0x08;
const FH4_TRY_BLOCK_MAP: u8 = 0x10;
// FH4 handler header bits.
const FH4_HANDLER_ADJECTIVES: u8 = 0x01;
const FH4_HANDLER_DISP_TYPE: u8 = 0x02;
const FH4_HANDLER_DISP_CATCH_OBJ: u8 = 0x04;
const FH4_HANDLER_CONT_IS_RVA: u8 = 0x08;
const FH4_HANDLER_CONT_SHIFT: u8 = 4;

/// Data format of a language handler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HandlerKind {
    /// `__C_specific_handler` (`_noexcept`): SEH scope table.
    CSpecific,
    /// `__CxxFrameHandler3` (and the older `__CxxFrameHandler`/`2`): `FuncInfo` RVA.
    CxxFrameHandler3,
    /// `__CxxFrameHandler4`: compressed `FuncInfo4` RVA.
    CxxFrameHandler4,
    /// `__GSHandlerCheck`: stack-cookie check only, no language data.
    GsCheck,
}

impl HandlerKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CSpecific => "c_specific",
            Self::CxxFrameHandler3 => "cxx3",
            Self::CxxFrameHandler4 => "cxx4",
            Self::GsCheck => "gs_check",
        }
    }

    /// Kind of the language handler import `name` (`None` for other imports).
    pub fn from_import(name: &str) -> Option<Self> {
        HANDLER_IMPORTS
            .iter()
            .find(|(import, _)| *import == name)
            .map(|(_, kind)| *kind)
    }
}

/// An identified handler routine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Handler {
    pub kind: HandlerKind,
    /// The routine is a static wrapper in the image (MSVC `__GSHandlerCheck*`) rather than an
    /// import thunk; the data keeps the inner handler's format.
    pub wrapper: bool,
}

/// Category of an EH metadata problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EhDiagnosticKind {
    /// The handler routine is not a recognised language handler (custom handler or unknown
    /// wrapper); its data is not parsed. Reported once per handler RVA.
    UnknownHandler,
    /// The handler routine reaches more than one language handler. Reported once per handler.
    AmbiguousHandler,
    /// The data, or a table it points to, is unmapped or runs past the mapped bytes.
    Truncated,
    /// `FuncInfo` magic number outside `0x19930520..=0x19930522`.
    UnknownMagic,
    /// A table count above the sanity limit (or negative).
    TooLarge,
    /// A funclet, handler or resume-target RVA that is not inside executable code.
    NotCode,
    /// A reserved encoding (FH4 `contAddr == 3`).
    Reserved,
    /// FH4 separated code: no IP-to-state map for this function's segment.
    MissingSegment,
    /// An FH4 unwind-map `nextOffset` that does not land on an entry.
    BadUnwindLink,
}

impl EhDiagnosticKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::UnknownHandler => "unknown_handler",
            Self::AmbiguousHandler => "ambiguous_handler",
            Self::Truncated => "truncated",
            Self::UnknownMagic => "unknown_magic",
            Self::TooLarge => "too_large",
            Self::NotCode => "not_code",
            Self::Reserved => "reserved",
            Self::MissingSegment => "missing_segment",
            Self::BadUnwindLink => "bad_unwind_link",
        }
    }
}

/// A non-fatal EH metadata problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EhDiagnostic {
    /// `BeginAddress` of the function, or the handler RVA for handler-level problems.
    pub rva: u32,
    /// The offending RVA (table, funclet, target); equal to `rva` when there is none.
    pub address: u32,
    pub kind: EhDiagnosticKind,
}

impl fmt::Display for EhDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} at RVA 0x{:08X} (address 0x{:08X})",
            self.kind.as_str(),
            self.rva,
            self.address
        )
    }
}

/// Handler of one SEH scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScopeHandler {
    /// `__except (filter)`: RVA of the filter funclet.
    Filter(u32),
    /// `__except (constant)`: the filter is a constant (`1` = `EXCEPTION_EXECUTE_HANDLER`).
    Constant(u32),
    /// `__finally`: RVA of the termination handler.
    Finally(u32),
}

/// One `C_SCOPE_TABLE` record (RVAs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScopeRecord {
    pub begin: u32,
    pub end: u32,
    pub handler: ScopeHandler,
    /// `__except` body (0 for `__finally`).
    pub jump_target: u32,
}

/// `__CxxFrameHandler3` unwind map entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnwindEntry3 {
    pub to_state: i32,
    /// RVA of the cleanup funclet (0 = none).
    pub action: u32,
}

/// `__CxxFrameHandler3` catch handler (`HandlerType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CatchHandler3 {
    pub adjectives: u32,
    /// RVA of the type descriptor (0 = `catch (...)`).
    pub type_descriptor: u32,
    /// Frame offset of the catch object.
    pub catch_object: i32,
    /// RVA of the catch funclet.
    pub handler: u32,
    /// Establisher frame offset.
    pub frame: u32,
}

/// `__CxxFrameHandler3` try block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TryBlock3 {
    pub try_low: i32,
    pub try_high: i32,
    pub catch_high: i32,
    pub handlers: Vec<CatchHandler3>,
}

/// Decoded `__CxxFrameHandler3` `FuncInfo`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuncInfo3 {
    /// RVA of the structure.
    pub rva: u32,
    pub magic: u32,
    pub bbt_flags: u8,
    pub unwind_map: Vec<UnwindEntry3>,
    pub try_blocks: Vec<TryBlock3>,
    /// `(ip RVA, state)` pairs.
    pub ip_to_state: Vec<(u32, i32)>,
    pub unwind_help: i32,
    /// RVA of the exception-specification type list (0 = none or pre-`..21` magic).
    pub es_type_list: u32,
    pub eh_flags: u32,
}

/// `__CxxFrameHandler4` unwind map entry type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UnwindKind4 {
    /// No unwind action.
    None,
    /// Destructor RVA + frame offset of the object.
    DtorWithObj,
    /// Destructor RVA + frame offset of a pointer to the object.
    DtorWithPtrToObj,
    /// RVA of a cleanup funclet.
    Funclet,
}

/// `__CxxFrameHandler4` unwind map entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnwindEntry4 {
    pub kind: UnwindKind4,
    /// State reached after this entry's action (-1 = none).
    pub next_state: i32,
    /// Destructor or cleanup funclet RVA (0 for [`UnwindKind4::None`]).
    pub action: u32,
    /// Frame offset of the object (destructor entries).
    pub object: u32,
}

/// `__CxxFrameHandler4` catch handler (`HandlerType4`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatchHandler4 {
    pub header: u8,
    pub adjectives: u32,
    pub type_descriptor: u32,
    pub catch_object: u32,
    /// RVA of the catch funclet.
    pub handler: u32,
    /// Continuation addresses as RVAs (empty: the funclet returns the continuation in `rax`).
    pub continuations: Vec<u32>,
}

/// `__CxxFrameHandler4` try block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TryBlock4 {
    pub try_low: u32,
    pub try_high: u32,
    pub catch_high: u32,
    /// RVA of the handler array.
    pub handler_array: u32,
    pub handlers: Vec<CatchHandler4>,
}

/// Decoded `__CxxFrameHandler4` `FuncInfo4`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuncInfo4 {
    /// RVA of the compressed structure.
    pub rva: u32,
    pub header: u8,
    pub bbt_flags: u32,
    /// RVAs of the unwind and try-block maps (0 = absent).
    pub unwind_map_rva: u32,
    pub try_block_map_rva: u32,
    /// RVA of the IP-to-state map used for this function (after segment selection).
    pub ip_map_rva: u32,
    /// Establisher frame offset (catch funclets only).
    pub frame: Option<u32>,
    pub unwind_map: Vec<UnwindEntry4>,
    pub try_blocks: Vec<TryBlock4>,
    /// `(ip RVA, state)` pairs.
    pub ip_to_state: Vec<(u32, i32)>,
}

impl FuncInfo4 {
    pub fn is_catch(&self) -> bool {
        self.header & FH4_IS_CATCH != 0
    }

    pub fn is_separated(&self) -> bool {
        self.header & FH4_IS_SEPARATED != 0
    }
}

/// Parsed language-specific data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LanguageData {
    /// No data (`__GSHandlerCheck`), or the data could not be parsed (see diagnostics).
    None,
    ScopeTable(Vec<ScopeRecord>),
    Cxx3(FuncInfo3),
    Cxx4(FuncInfo4),
}

/// Role of a funclet (a separately called piece of code the language handler runs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FuncletKind {
    /// C++ catch block; returns the continuation (address in `rax`, or an index for FH4
    /// handlers with metadata continuations).
    Catch,
    /// C++ unwind action (state cleanup funclet).
    Cleanup,
    /// Destructor called by FH4 for an unwind-map object.
    Destructor,
    /// SEH `__except` filter.
    Filter,
    /// SEH `__finally` termination handler.
    Finally,
}

impl FuncletKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Catch => "catch",
            Self::Cleanup => "cleanup",
            Self::Destructor => "destructor",
            Self::Filter => "filter",
            Self::Finally => "finally",
        }
    }
}

/// Why a resume target exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResumeKind {
    /// C++ catch continuation (FH4 metadata).
    CatchContinuation,
    /// SEH `__except` body.
    ExceptBody,
}

/// One function with language-specific EH data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EhFunction {
    /// `BeginAddress` / `EndAddress` of the `.pdata` record carrying the handler.
    pub begin: u32,
    pub end: u32,
    /// Handler routine RVA.
    pub handler_rva: u32,
    pub handler: Handler,
    /// RVA of the language-specific data (right after the handler RVA in `UNWIND_INFO`).
    pub data_rva: u32,
    pub data: LanguageData,
}

impl EhFunction {
    /// Funclets referenced by the data, `(rva, kind)`, sorted and deduplicated.
    pub fn funclets(&self) -> Vec<(u32, FuncletKind)> {
        let mut out = BTreeSet::new();
        match &self.data {
            LanguageData::None => {}
            LanguageData::ScopeTable(records) => {
                for record in records {
                    match record.handler {
                        ScopeHandler::Filter(rva) => out.insert((rva, FuncletKind::Filter)),
                        ScopeHandler::Finally(rva) => out.insert((rva, FuncletKind::Finally)),
                        ScopeHandler::Constant(_) => false,
                    };
                }
            }
            LanguageData::Cxx3(info) => {
                for entry in info.unwind_map.iter().filter(|e| e.action != 0) {
                    out.insert((entry.action, FuncletKind::Cleanup));
                }
                for handler in info.try_blocks.iter().flat_map(|t| &t.handlers) {
                    out.insert((handler.handler, FuncletKind::Catch));
                }
            }
            LanguageData::Cxx4(info) => {
                for entry in &info.unwind_map {
                    let kind = match entry.kind {
                        UnwindKind4::None => continue,
                        UnwindKind4::Funclet => FuncletKind::Cleanup,
                        UnwindKind4::DtorWithObj | UnwindKind4::DtorWithPtrToObj => {
                            FuncletKind::Destructor
                        }
                    };
                    out.insert((entry.action, kind));
                }
                for handler in info.try_blocks.iter().flat_map(|t| &t.handlers) {
                    out.insert((handler.handler, FuncletKind::Catch));
                }
            }
        }
        out.into_iter().collect()
    }

    /// Resume targets named by the data, `(rva, kind)`, sorted and deduplicated.
    pub fn resume_targets(&self) -> Vec<(u32, ResumeKind)> {
        let mut out = BTreeSet::new();
        match &self.data {
            LanguageData::ScopeTable(records) => {
                for record in records.iter().filter(|r| r.jump_target != 0) {
                    out.insert((record.jump_target, ResumeKind::ExceptBody));
                }
            }
            LanguageData::Cxx4(info) => {
                for handler in info.try_blocks.iter().flat_map(|t| &t.handlers) {
                    for &rva in &handler.continuations {
                        out.insert((rva, ResumeKind::CatchContinuation));
                    }
                }
            }
            LanguageData::None | LanguageData::Cxx3(_) => {}
        }
        out.into_iter().collect()
    }
}

/// Aggregate counts of an [`EhMetadata`] (for the discovery summary).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EhStats {
    /// Functions per handler kind (`as_str` order of [`HandlerKind`]), and with an unrecognised
    /// handler.
    pub c_specific: usize,
    pub cxx3: usize,
    pub cxx4: usize,
    pub gs_check: usize,
    pub unknown: usize,
    /// Functions whose data parsed without diagnostics.
    pub parsed: usize,
    /// Distinct funclets and metadata resume targets.
    pub funclets: usize,
    pub resume_targets: usize,
    pub diagnostics: usize,
}

/// What [`EhMetadata::seed`] hands to discovery (see [`CodeImage::eh`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EhSeeds {
    /// `(owner function entry VA, resume target VA)`, sorted and deduplicated. The owner is the
    /// `.pdata` function containing the target.
    pub resume_targets: Vec<(u64, u64)>,
    /// IAT slot VAs of the `setjmp` family ([`SETJMP_IMPORTS`]), sorted.
    pub setjmp_slots: Vec<u64>,
    pub stats: EhStats,
}

/// EH metadata of an image.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EhMetadata {
    /// Identified handler routines by RVA (unrecognised ones are absent, see diagnostics).
    pub handlers: BTreeMap<u32, Handler>,
    /// Every `.pdata` record with a handler, sorted by `begin`. Records whose handler is
    /// unrecognised are listed with [`LanguageData::None`] in [`Self::unrecognised`] instead.
    pub functions: Vec<EhFunction>,
    /// `BeginAddress` of records whose handler could not be identified.
    pub unrecognised: Vec<u32>,
    /// IAT slot RVAs of `setjmp`-family imports.
    pub setjmp_slots: Vec<u32>,
    pub diagnostics: Vec<EhDiagnostic>,
}

/// Bounds-checked RVA reads.
struct Reader<'p, 'f> {
    pe: &'p Pe64,
    file: &'f [u8],
}

impl<'f> Reader<'_, 'f> {
    /// Bytes from `rva` to the end of its section's file data.
    fn tail(&self, rva: u32) -> Option<&'f [u8]> {
        let offset = self.pe.rva_to_file_offset(rva)?;
        let section = self.pe.sections.iter().find(|s| {
            rva.checked_sub(s.virtual_address)
                .is_some_and(|delta| delta < s.raw_size)
        });
        let end = match section {
            Some(s) => {
                let extent = if s.virtual_size == 0 {
                    s.raw_size
                } else {
                    s.virtual_size.min(s.raw_size)
                };
                (s.raw_offset as usize).checked_add(extent as usize)?
            }
            // Headers.
            None => self.pe.optional.size_of_headers as usize,
        };
        self.file.get(offset..end.min(self.file.len()))
    }

    fn read(&self, rva: u32, len: usize) -> Option<&'f [u8]> {
        self.tail(rva)?.get(..len)
    }

    fn u32(&self, rva: u32) -> Option<u32> {
        self.read(rva, 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

/// Sequential reader over the compressed FH4 encoding.
struct Stream<'f> {
    bytes: &'f [u8],
    pos: usize,
}

impl Stream<'_> {
    fn u8(&mut self) -> Option<u8> {
        let b = *self.bytes.get(self.pos)?;
        self.pos += 1;
        Some(b)
    }

    /// Raw little-endian `i32` (`ReadInt`).
    fn int(&mut self) -> Option<u32> {
        let b = self.bytes.get(self.pos..self.pos.checked_add(4)?)?;
        self.pos += 4;
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Compressed unsigned integer (`ReadUnsigned`).
    fn unsigned(&mut self) -> Option<u32> {
        let (value, len) = decode_compressed(self.bytes.get(self.pos..)?)?;
        self.pos += len;
        Some(value)
    }
}

/// Decodes one FH4 compressed unsigned integer at the start of `bytes`: `(value, length)`.
pub fn decode_compressed(bytes: &[u8]) -> Option<(u32, usize)> {
    let first = *bytes.first()?;
    let len = if first & 0x1 == 0 {
        1
    } else if first & 0x3 == 0x1 {
        2
    } else if first & 0x7 == 0x3 {
        3
    } else if first & 0xF == 0x7 {
        4
    } else {
        5
    };
    let raw = bytes.get(..len)?;
    let value = if len == 5 {
        u32::from_le_bytes([raw[1], raw[2], raw[3], raw[4]])
    } else {
        let mut v = 0u32;
        for (i, b) in raw.iter().enumerate() {
            v |= u32::from(*b) << (8 * i);
        }
        v >> len
    };
    Some((value, len))
}

/// Encodes `value` in the FH4 compressed form (shortest encoding). Inverse of
/// [`decode_compressed`]; used to build test fixtures.
pub fn encode_compressed(value: u32) -> Vec<u8> {
    let len = match value {
        0..0x80 => 1,
        0x80..0x4000 => 2,
        0x4000..0x20_0000 => 3,
        0x20_0000..0x1000_0000 => 4,
        _ => 5,
    };
    if len == 5 {
        let mut out = vec![0x0F];
        out.extend_from_slice(&value.to_le_bytes());
        return out;
    }
    let tag = (1u32 << (len - 1)) - 1;
    let packed = (value << len) | tag;
    packed.to_le_bytes()[..len].to_vec()
}

/// Diagnostics collector for one function.
struct Diags<'a> {
    out: &'a mut Vec<EhDiagnostic>,
    rva: u32,
    count: usize,
}

impl Diags<'_> {
    fn push(&mut self, address: u32, kind: EhDiagnosticKind) {
        self.count += 1;
        self.out.push(EhDiagnostic {
            rva: self.rva,
            address,
            kind,
        });
    }
}

/// Validates a table count read from the data.
fn count(value: u32) -> Option<usize> {
    (value <= MAX_TABLE_ENTRIES).then_some(value as usize)
}

impl EhMetadata {
    /// Parse the language-specific data of every `.pdata` record of `dir` with a handler. `file`
    /// must be the bytes `pe` was parsed from. Never fails; problems become diagnostics.
    pub fn from_pe(pe: &Pe64, file: &[u8], dir: &ExceptionDirectory) -> Self {
        let image = CodeImage::from_pe(pe, file);
        let reader = Reader { pe, file };
        let base = pe.optional.image_base;
        let slots = import_slots(pe);
        let mut out = Self {
            setjmp_slots: slots
                .iter()
                .filter(|(_, name)| SETJMP_IMPORTS.contains(&name.as_str()))
                .map(|(va, _)| (va - base) as u32)
                .collect(),
            ..Self::default()
        };

        // Identify every handler routine once.
        let handler_rvas = dir.handler_rvas();
        let scans: Vec<(u32, HandlerScan)> = handler_rvas
            .iter()
            .map(|&rva| (rva, scan_handler(&image, &slots, base + u64::from(rva))))
            .collect();
        // Cookie-check helpers called first by recognised wrappers identify `__GSHandlerCheck`.
        let gs_helpers: BTreeSet<u64> = scans
            .iter()
            .filter(|(_, scan)| scan.thunk.is_none() && scan.kinds.len() == 1)
            .filter_map(|(_, scan)| scan.direct_calls.first().copied())
            .collect();
        for (rva, scan) in &scans {
            let handler = if let Some(kind) = scan.thunk {
                Ok(Handler {
                    kind,
                    wrapper: false,
                })
            } else {
                match scan.kinds.len() {
                    1 => Ok(Handler {
                        kind: scan.kinds[0],
                        wrapper: true,
                    }),
                    0 if !scan.direct_calls.is_empty()
                        && scan.direct_calls.iter().all(|c| gs_helpers.contains(c)) =>
                    {
                        Ok(Handler {
                            kind: HandlerKind::GsCheck,
                            wrapper: true,
                        })
                    }
                    0 => Err(EhDiagnosticKind::UnknownHandler),
                    _ => Err(EhDiagnosticKind::AmbiguousHandler),
                }
            };
            match handler {
                Ok(handler) => {
                    out.handlers.insert(*rva, handler);
                }
                Err(kind) => out.diagnostics.push(EhDiagnostic {
                    rva: *rva,
                    address: *rva,
                    kind,
                }),
            }
        }

        for entry in &dir.entries {
            let Some(eh) = entry.unwind.as_ref().and_then(|u| u.handler) else {
                continue;
            };
            let Some(&handler) = out.handlers.get(&eh.rva) else {
                out.unrecognised.push(entry.function.begin);
                continue;
            };
            let mut diags = Diags {
                out: &mut out.diagnostics,
                rva: entry.function.begin,
                count: 0,
            };
            let data = match handler.kind {
                HandlerKind::GsCheck => LanguageData::None,
                HandlerKind::CSpecific => {
                    parse_scope_table(&reader, &image, eh.data_rva, &mut diags)
                }
                HandlerKind::CxxFrameHandler3 => {
                    parse_fh3(&reader, &image, eh.data_rva, &mut diags)
                }
                HandlerKind::CxxFrameHandler4 => parse_fh4(
                    &reader,
                    &image,
                    eh.data_rva,
                    entry.function.begin,
                    &mut diags,
                ),
            };
            out.functions.push(EhFunction {
                begin: entry.function.begin,
                end: entry.function.end,
                handler_rva: eh.rva,
                handler,
                data_rva: eh.data_rva,
                data,
            });
        }
        out
    }

    /// Distinct funclets of all functions, `(rva, kind)`.
    pub fn funclets(&self) -> BTreeSet<(u32, FuncletKind)> {
        self.functions
            .iter()
            .flat_map(EhFunction::funclets)
            .collect()
    }

    /// Aggregate counts.
    pub fn stats(&self) -> EhStats {
        let mut stats = EhStats {
            unknown: self.unrecognised.len(),
            diagnostics: self.diagnostics.len(),
            ..EhStats::default()
        };
        let failed: BTreeSet<u32> = self.diagnostics.iter().map(|d| d.rva).collect();
        for function in &self.functions {
            match function.handler.kind {
                HandlerKind::CSpecific => stats.c_specific += 1,
                HandlerKind::CxxFrameHandler3 => stats.cxx3 += 1,
                HandlerKind::CxxFrameHandler4 => stats.cxx4 += 1,
                HandlerKind::GsCheck => stats.gs_check += 1,
            }
            if !failed.contains(&function.begin) {
                stats.parsed += 1;
            }
        }
        stats.funclets = self.funclets().len();
        stats.resume_targets = self
            .functions
            .iter()
            .flat_map(EhFunction::resume_targets)
            .map(|(rva, _)| rva)
            .collect::<BTreeSet<_>>()
            .len();
        stats
    }

    /// Feed the metadata into discovery: funclets in executable code become entry points (kept
    /// sorted and deduplicated), and [`CodeImage::eh`] receives the resume targets with their
    /// owner (the `.pdata` function containing the target; targets outside every `.pdata`
    /// function are owned by the function whose data names them) plus the `setjmp` slots.
    /// Install `.pdata` first ([`ExceptionDirectory::seed`]).
    pub fn seed(&self, image: &mut CodeImage<'_>) {
        let base = image.base();
        let va = |rva: u32| base.saturating_add(u64::from(rva));
        let funclets: Vec<u64> = self
            .funclets()
            .into_iter()
            .map(|(rva, _)| va(rva))
            .filter(|&va| image.region_of(va) == Region::Code)
            .collect();
        let mut entries = std::mem::take(&mut image.entry_points);
        entries.extend(funclets);
        entries.sort_unstable();
        entries.dedup();
        image.entry_points = entries;

        let mut targets = BTreeSet::new();
        for function in &self.functions {
            for (rva, _) in function.resume_targets() {
                let target = va(rva);
                if image.region_of(target) != Region::Code {
                    continue;
                }
                let owner = image
                    .function_table
                    .function_at(target)
                    .unwrap_or_else(|| va(function.begin));
                targets.insert((owner, target));
            }
        }
        image.eh = EhSeeds {
            resume_targets: targets.into_iter().collect(),
            setjmp_slots: self.setjmp_slots.iter().map(|&rva| va(rva)).collect(),
            stats: self.stats(),
        };
    }
}

/// IAT slot VA -> imported symbol name, for normal and delay-load imports.
fn import_slots(pe: &Pe64) -> BTreeMap<u64, String> {
    let base = pe.optional.image_base;
    let mut slots = BTreeMap::new();
    let modules = pe
        .imports
        .iter()
        .map(|m| (m.iat_rva, &m.symbols))
        .chain(pe.delay_imports.iter().map(|m| (m.iat_rva, &m.symbols)));
    for (iat, symbols) in modules {
        if iat == 0 {
            continue;
        }
        for (i, symbol) in symbols.iter().enumerate() {
            if let ImportedSymbol::ByName { name, .. } = symbol {
                let slot = base + u64::from(iat) + 8 * i as u64;
                slots.insert(slot, name.clone());
            }
        }
    }
    slots
}

/// What a bounded walk over a handler routine found.
#[derive(Default)]
struct HandlerScan {
    /// Set when the routine itself is `jmp [rip+slot]` into a language-handler import.
    thunk: Option<HandlerKind>,
    /// Language handlers reached through IAT slots or import thunks (sorted, distinct).
    kinds: Vec<HandlerKind>,
    /// Direct call targets that are not import thunks, in address order.
    direct_calls: Vec<u64>,
}

/// Language-handler kind of the IAT slot read by a RIP-relative `call`/`jmp`.
fn slot_kind(ins: &Instruction, slots: &BTreeMap<u64, String>) -> Option<HandlerKind> {
    (ins.op_count() > 0 && ins.op_kind(0) == OpKind::Memory && ins.is_ip_rel_memory_operand())
        .then(|| slots.get(&ins.ip_rel_memory_address()))
        .flatten()
        .and_then(|name| HandlerKind::from_import(name))
}

/// First instruction at `va`, if it decodes.
fn first_instruction(image: &CodeImage<'_>, va: u64) -> Option<Instruction> {
    let bytes = image.code_bytes(va).ok()?;
    let mut decoder = Decoder::with_ip(64, bytes, va, DecoderOptions::NONE);
    let ins = decoder.decode();
    (!ins.is_invalid()).then_some(ins)
}

/// Whether `va` is an import thunk `jmp [rip+slot]`; returns the handler kind of its slot.
fn thunk_kind(
    image: &CodeImage<'_>,
    slots: &BTreeMap<u64, String>,
    va: u64,
) -> Option<Option<HandlerKind>> {
    let ins = first_instruction(image, va)?;
    (ins.flow_control() == FlowControl::IndirectBranch && ins.is_ip_rel_memory_operand())
        .then(|| slot_kind(&ins, slots))
}

/// Walks the handler routine at `va` (both sides of branches, at most
/// [`HANDLER_SCAN_LIMIT`] instructions) and records what it calls.
fn scan_handler(image: &CodeImage<'_>, slots: &BTreeMap<u64, String>, va: u64) -> HandlerScan {
    let mut scan = HandlerScan::default();
    if let Some(Some(kind)) = thunk_kind(image, slots, va) {
        scan.thunk = Some(kind);
        return scan;
    }
    let mut kinds = BTreeSet::new();
    let mut calls = BTreeSet::new();
    let mut seen = BTreeSet::new();
    let mut work = vec![va];
    let mut budget = HANDLER_SCAN_LIMIT;
    while let Some(start) = work.pop() {
        if !seen.insert(start) {
            continue;
        }
        let Ok(bytes) = image.code_bytes(start) else {
            continue;
        };
        let mut decoder = Decoder::with_ip(64, bytes, start, DecoderOptions::NONE);
        while budget > 0 && decoder.can_decode() {
            budget -= 1;
            let ins = decoder.decode();
            if ins.is_invalid() {
                break;
            }
            let direct = (ins.op_count() > 0 && ins.op_kind(0) == OpKind::NearBranch64)
                .then(|| ins.near_branch64());
            match ins.flow_control() {
                FlowControl::Call | FlowControl::UnconditionalBranch => {
                    if let Some(target) = direct {
                        match thunk_kind(image, slots, target) {
                            Some(Some(kind)) => {
                                kinds.insert(kind);
                            }
                            Some(None) => {}
                            None if ins.flow_control() == FlowControl::Call => {
                                calls.insert(target);
                            }
                            None => work.push(target),
                        }
                    }
                    if ins.flow_control() == FlowControl::UnconditionalBranch {
                        break;
                    }
                }
                FlowControl::IndirectCall => {
                    kinds.extend(slot_kind(&ins, slots));
                }
                FlowControl::IndirectBranch => {
                    kinds.extend(slot_kind(&ins, slots));
                    break;
                }
                FlowControl::ConditionalBranch => {
                    if let Some(target) = direct {
                        work.push(target);
                    }
                }
                FlowControl::Return | FlowControl::Exception | FlowControl::Interrupt => break,
                _ => {}
            }
        }
    }
    scan.kinds = kinds.into_iter().collect();
    scan.direct_calls = calls.into_iter().collect();
    scan
}

/// Records `rva` as code-or-diagnostic.
fn check_code(image: &CodeImage<'_>, rva: u32, diags: &mut Diags<'_>) {
    let va = image.base().saturating_add(u64::from(rva));
    if image.region_of(va) != Region::Code {
        diags.push(rva, EhDiagnosticKind::NotCode);
    }
}

fn is_code(image: &CodeImage<'_>, rva: u32) -> bool {
    image.region_of(image.base().saturating_add(u64::from(rva))) == Region::Code
}

/// `__C_specific_handler` scope table at `rva`.
fn parse_scope_table(
    reader: &Reader<'_, '_>,
    image: &CodeImage<'_>,
    rva: u32,
    diags: &mut Diags<'_>,
) -> LanguageData {
    let Some(raw_count) = reader.u32(rva) else {
        diags.push(rva, EhDiagnosticKind::Truncated);
        return LanguageData::None;
    };
    let Some(n) = count(raw_count) else {
        diags.push(rva, EhDiagnosticKind::TooLarge);
        return LanguageData::None;
    };
    let Some(bytes) = rva.checked_add(4).and_then(|at| reader.read(at, n * 16)) else {
        diags.push(rva, EhDiagnosticKind::Truncated);
        return LanguageData::None;
    };
    let field = |i: usize, f: usize| {
        let at = i * 16 + f * 4;
        u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
    };
    let mut records = Vec::with_capacity(n);
    for i in 0..n {
        let (begin, end, handler, jump_target) =
            (field(i, 0), field(i, 1), field(i, 2), field(i, 3));
        let handler = if jump_target == 0 {
            check_code(image, handler, diags);
            ScopeHandler::Finally(handler)
        } else {
            check_code(image, jump_target, diags);
            if is_code(image, handler) {
                ScopeHandler::Filter(handler)
            } else {
                if handler != EXCEPTION_EXECUTE_HANDLER {
                    diags.push(handler, EhDiagnosticKind::NotCode);
                }
                ScopeHandler::Constant(handler)
            }
        };
        records.push(ScopeRecord {
            begin,
            end,
            handler,
            jump_target,
        });
    }
    LanguageData::ScopeTable(records)
}

/// `n` records of `size` bytes at `rva`, as little-endian `u32` fields.
fn read_records(
    reader: &Reader<'_, '_>,
    rva: u32,
    raw_count: u32,
    size: usize,
    diags: &mut Diags<'_>,
) -> Option<Vec<Vec<u32>>> {
    let Some(n) = count(raw_count) else {
        diags.push(rva, EhDiagnosticKind::TooLarge);
        return None;
    };
    if n == 0 {
        return Some(Vec::new());
    }
    let Some(bytes) = reader.read(rva, n * size) else {
        diags.push(rva, EhDiagnosticKind::Truncated);
        return None;
    };
    Some(
        bytes
            .chunks_exact(size)
            .map(|chunk| {
                (0..size / 4)
                    .map(|i| {
                        let b = &chunk[4 * i..4 * i + 4];
                        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
                    })
                    .collect()
            })
            .collect(),
    )
}

/// `__CxxFrameHandler3`: the data is the RVA of a `FuncInfo`.
fn parse_fh3(
    reader: &Reader<'_, '_>,
    image: &CodeImage<'_>,
    data_rva: u32,
    diags: &mut Diags<'_>,
) -> LanguageData {
    let Some(rva) = reader.u32(data_rva) else {
        diags.push(data_rva, EhDiagnosticKind::Truncated);
        return LanguageData::None;
    };
    let Some(raw_magic) = reader.u32(rva) else {
        diags.push(rva, EhDiagnosticKind::Truncated);
        return LanguageData::None;
    };
    let magic = raw_magic & FH3_MAGIC_MASK;
    if !(FH3_MAGIC_MIN..=FH3_MAGIC_MAX).contains(&magic) {
        diags.push(rva, EhDiagnosticKind::UnknownMagic);
        return LanguageData::None;
    }
    let fields = 8 + (magic - FH3_MAGIC_MIN) as usize;
    let Some(header) = reader.read(rva, fields * 4) else {
        diags.push(rva, EhDiagnosticKind::Truncated);
        return LanguageData::None;
    };
    let f = |i: usize| {
        u32::from_le_bytes([
            header[4 * i],
            header[4 * i + 1],
            header[4 * i + 2],
            header[4 * i + 3],
        ])
    };
    let mut info = FuncInfo3 {
        rva,
        magic,
        bbt_flags: (raw_magic >> 29) as u8,
        unwind_map: Vec::new(),
        try_blocks: Vec::new(),
        ip_to_state: Vec::new(),
        unwind_help: f(7) as i32,
        es_type_list: if fields > 8 { f(8) } else { 0 },
        eh_flags: if fields > 9 { f(9) } else { 0 },
    };
    let (max_state, unwind_rva, n_try, try_rva, n_ip, ip_rva) =
        (f(1), f(2), f(3), f(4), f(5), f(6));

    if let Some(records) = read_records(reader, unwind_rva, max_state, 8, diags) {
        info.unwind_map = records
            .iter()
            .map(|r| UnwindEntry3 {
                to_state: r[0] as i32,
                action: r[1],
            })
            .collect();
        for entry in info.unwind_map.iter().filter(|e| e.action != 0) {
            check_code(image, entry.action, diags);
        }
    }
    if let Some(records) = read_records(reader, try_rva, n_try, 20, diags) {
        for r in records {
            let mut block = TryBlock3 {
                try_low: r[0] as i32,
                try_high: r[1] as i32,
                catch_high: r[2] as i32,
                handlers: Vec::new(),
            };
            if let Some(handlers) = read_records(reader, r[4], r[3], 20, diags) {
                for h in handlers {
                    check_code(image, h[3], diags);
                    block.handlers.push(CatchHandler3 {
                        adjectives: h[0],
                        type_descriptor: h[1],
                        catch_object: h[2] as i32,
                        handler: h[3],
                        frame: h[4],
                    });
                }
            }
            info.try_blocks.push(block);
        }
    }
    if let Some(records) = read_records(reader, ip_rva, n_ip, 8, diags) {
        info.ip_to_state = records.iter().map(|r| (r[0], r[1] as i32)).collect();
    }
    LanguageData::Cxx3(info)
}

/// Compressed stream at `rva`, or a `Truncated` diagnostic.
fn stream<'f>(reader: &Reader<'_, 'f>, rva: u32, diags: &mut Diags<'_>) -> Option<Stream<'f>> {
    match reader.tail(rva) {
        Some(bytes) if !bytes.is_empty() => Some(Stream { bytes, pos: 0 }),
        _ => {
            diags.push(rva, EhDiagnosticKind::Truncated);
            None
        }
    }
}

/// `__CxxFrameHandler4`: the data is the RVA of a compressed `FuncInfo4`. `function` is the RVA
/// of the function the data belongs to (base of function-relative offsets).
fn parse_fh4(
    reader: &Reader<'_, '_>,
    image: &CodeImage<'_>,
    data_rva: u32,
    function: u32,
    diags: &mut Diags<'_>,
) -> LanguageData {
    let Some(rva) = reader.u32(data_rva) else {
        diags.push(data_rva, EhDiagnosticKind::Truncated);
        return LanguageData::None;
    };
    let Some(mut s) = stream(reader, rva, diags) else {
        return LanguageData::None;
    };
    let truncated = |diags: &mut Diags<'_>| {
        diags.push(rva, EhDiagnosticKind::Truncated);
        LanguageData::None
    };
    let Some(header) = s.u8() else {
        return truncated(diags);
    };
    let mut info = FuncInfo4 {
        rva,
        header,
        bbt_flags: 0,
        unwind_map_rva: 0,
        try_block_map_rva: 0,
        ip_map_rva: 0,
        frame: None,
        unwind_map: Vec::new(),
        try_blocks: Vec::new(),
        ip_to_state: Vec::new(),
    };
    let fields = (|| {
        if header & FH4_BBT != 0 {
            info.bbt_flags = s.unsigned()?;
        }
        if header & FH4_UNWIND_MAP != 0 {
            info.unwind_map_rva = s.int()?;
        }
        if header & FH4_TRY_BLOCK_MAP != 0 {
            info.try_block_map_rva = s.int()?;
        }
        let ip_map = s.int()?;
        if header & FH4_IS_CATCH != 0 {
            info.frame = Some(s.unsigned()?);
        }
        Some(ip_map)
    })();
    let Some(ip_map) = fields else {
        return truncated(diags);
    };
    info.ip_map_rva = if info.is_separated() {
        separated_ip_map(reader, ip_map, function, diags).unwrap_or_default()
    } else {
        ip_map
    };

    if info.unwind_map_rva != 0 {
        info.unwind_map = parse_unwind_map4(reader, image, info.unwind_map_rva, diags);
    }
    if info.try_block_map_rva != 0 {
        info.try_blocks = parse_try_map4(reader, image, info.try_block_map_rva, function, diags);
    }
    if info.ip_map_rva != 0 {
        info.ip_to_state = parse_ip_map4(reader, info.ip_map_rva, function, diags);
    }
    LanguageData::Cxx4(info)
}

/// Separated code: `{count, [segment RVA (raw), map RVA (raw)]...}`; the map of `function`.
fn separated_ip_map(
    reader: &Reader<'_, '_>,
    rva: u32,
    function: u32,
    diags: &mut Diags<'_>,
) -> Option<u32> {
    let mut s = stream(reader, rva, diags)?;
    let Some(n) = s.unsigned() else {
        diags.push(rva, EhDiagnosticKind::Truncated);
        return None;
    };
    let Some(n) = count(n) else {
        diags.push(rva, EhDiagnosticKind::TooLarge);
        return None;
    };
    for _ in 0..n {
        let (Some(segment), Some(map)) = (s.int(), s.int()) else {
            diags.push(rva, EhDiagnosticKind::Truncated);
            return None;
        };
        if segment == function {
            return Some(map);
        }
    }
    diags.push(rva, EhDiagnosticKind::MissingSegment);
    None
}

fn parse_unwind_map4(
    reader: &Reader<'_, '_>,
    image: &CodeImage<'_>,
    rva: u32,
    diags: &mut Diags<'_>,
) -> Vec<UnwindEntry4> {
    let Some(mut s) = stream(reader, rva, diags) else {
        return Vec::new();
    };
    let Some(n) = s.unsigned() else {
        diags.push(rva, EhDiagnosticKind::Truncated);
        return Vec::new();
    };
    let Some(n) = count(n) else {
        diags.push(rva, EhDiagnosticKind::TooLarge);
        return Vec::new();
    };
    // (entry start offset, next offset, entry)
    let mut raw = Vec::with_capacity(n);
    for _ in 0..n {
        let start = s.pos;
        let parsed = (|| {
            let next_and_type = s.unsigned()?;
            let kind = match next_and_type & 0x3 {
                0 => UnwindKind4::None,
                1 => UnwindKind4::DtorWithObj,
                2 => UnwindKind4::DtorWithPtrToObj,
                _ => UnwindKind4::Funclet,
            };
            let (mut action, mut object) = (0, 0);
            match kind {
                UnwindKind4::DtorWithObj | UnwindKind4::DtorWithPtrToObj => {
                    action = s.int()?;
                    object = s.unsigned()?;
                }
                UnwindKind4::Funclet => action = s.int()?,
                UnwindKind4::None => {}
            }
            Some((
                next_and_type >> 2,
                UnwindEntry4 {
                    kind,
                    next_state: -1,
                    action,
                    object,
                },
            ))
        })();
        let Some((next_offset, entry)) = parsed else {
            diags.push(rva, EhDiagnosticKind::Truncated);
            return Vec::new();
        };
        if entry.kind != UnwindKind4::None {
            check_code(image, entry.action, diags);
        }
        raw.push((start, next_offset as usize, entry));
    }
    // `nextOffset` counts bytes back from the start of the entry to the start of the entry of
    // the next state; landing on the start of the map (the entry count) ends the chain
    // (state -1).
    let starts: Vec<usize> = raw.iter().map(|(start, _, _)| *start).collect();
    raw.into_iter()
        .map(|(start, next_offset, mut entry)| {
            let link = match start.checked_sub(next_offset) {
                Some(0) => Some(-1),
                Some(at) if next_offset != 0 => starts.binary_search(&at).ok().map(|s| s as i32),
                _ => None,
            };
            match link {
                Some(state) => entry.next_state = state,
                None => diags.push(rva, EhDiagnosticKind::BadUnwindLink),
            }
            entry
        })
        .collect()
}

fn parse_try_map4(
    reader: &Reader<'_, '_>,
    image: &CodeImage<'_>,
    rva: u32,
    function: u32,
    diags: &mut Diags<'_>,
) -> Vec<TryBlock4> {
    let Some(mut s) = stream(reader, rva, diags) else {
        return Vec::new();
    };
    let Some(n) = s.unsigned() else {
        diags.push(rva, EhDiagnosticKind::Truncated);
        return Vec::new();
    };
    let Some(n) = count(n) else {
        diags.push(rva, EhDiagnosticKind::TooLarge);
        return Vec::new();
    };
    let mut blocks = Vec::with_capacity(n);
    for _ in 0..n {
        let parsed = (|| {
            Some(TryBlock4 {
                try_low: s.unsigned()?,
                try_high: s.unsigned()?,
                catch_high: s.unsigned()?,
                handler_array: s.int()?,
                handlers: Vec::new(),
            })
        })();
        let Some(mut block) = parsed else {
            diags.push(rva, EhDiagnosticKind::Truncated);
            return blocks;
        };
        block.handlers = parse_handlers4(reader, image, block.handler_array, function, diags);
        blocks.push(block);
    }
    blocks
}

fn parse_handlers4(
    reader: &Reader<'_, '_>,
    image: &CodeImage<'_>,
    rva: u32,
    function: u32,
    diags: &mut Diags<'_>,
) -> Vec<CatchHandler4> {
    let Some(mut s) = stream(reader, rva, diags) else {
        return Vec::new();
    };
    let Some(n) = s.unsigned() else {
        diags.push(rva, EhDiagnosticKind::Truncated);
        return Vec::new();
    };
    let Some(n) = count(n) else {
        diags.push(rva, EhDiagnosticKind::TooLarge);
        return Vec::new();
    };
    let mut handlers = Vec::with_capacity(n);
    for _ in 0..n {
        let mut reserved = false;
        let parsed = (|| {
            let header = s.u8()?;
            let mut h = CatchHandler4 {
                header,
                adjectives: 0,
                type_descriptor: 0,
                catch_object: 0,
                handler: 0,
                continuations: Vec::new(),
            };
            if header & FH4_HANDLER_ADJECTIVES != 0 {
                h.adjectives = s.unsigned()?;
            }
            if header & FH4_HANDLER_DISP_TYPE != 0 {
                h.type_descriptor = s.int()?;
            }
            if header & FH4_HANDLER_DISP_CATCH_OBJ != 0 {
                h.catch_object = s.unsigned()?;
            }
            h.handler = s.int()?;
            let cont_count = (header >> FH4_HANDLER_CONT_SHIFT) & 0x3;
            reserved = cont_count == 3;
            for _ in 0..cont_count.min(2) {
                let value = if header & FH4_HANDLER_CONT_IS_RVA != 0 {
                    s.int()?
                } else {
                    function.checked_add(s.unsigned()?)?
                };
                h.continuations.push(value);
            }
            Some(h)
        })();
        let Some(handler) = parsed else {
            diags.push(rva, EhDiagnosticKind::Truncated);
            return handlers;
        };
        if reserved {
            diags.push(rva, EhDiagnosticKind::Reserved);
        }
        check_code(image, handler.handler, diags);
        for &cont in &handler.continuations {
            check_code(image, cont, diags);
        }
        handlers.push(handler);
    }
    handlers
}

fn parse_ip_map4(
    reader: &Reader<'_, '_>,
    rva: u32,
    function: u32,
    diags: &mut Diags<'_>,
) -> Vec<(u32, i32)> {
    let Some(mut s) = stream(reader, rva, diags) else {
        return Vec::new();
    };
    let Some(n) = s.unsigned() else {
        diags.push(rva, EhDiagnosticKind::Truncated);
        return Vec::new();
    };
    let Some(n) = count(n) else {
        diags.push(rva, EhDiagnosticKind::TooLarge);
        return Vec::new();
    };
    let mut out = Vec::with_capacity(n);
    let mut ip = function;
    for _ in 0..n {
        let (Some(delta), Some(state)) = (s.unsigned(), s.unsigned()) else {
            diags.push(rva, EhDiagnosticKind::Truncated);
            return out;
        };
        ip = ip.wrapping_add(delta);
        out.push((ip, state.wrapping_sub(1) as i32));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pdata::tests::{BASE, pe_with_pdata, put, unwind_bytes};
    use crate::pdata::{UNW_FLAG_EHANDLER, UNW_FLAG_UHANDLER};

    // ----- fixture layout (RVAs) -------------------------------------------------------------
    //
    // .text 0x1000..0x1200, .rdata 0x2000..0x2800 (see `pdata::tests::pe_with_pdata`).
    const F_SEH: u32 = 0x1080;
    const F_FH3: u32 = 0x10a0;
    const F_FH4: u32 = 0x10c0;
    const F_CATCH: u32 = 0x1100;
    const F_GS_FH3: u32 = 0x1120;
    const F_GS: u32 = 0x1130;
    const F_UNKNOWN: u32 = 0x1140;
    const F_CLEANUP: u32 = 0x1150;

    const GS_HELPER: u32 = 0x1160;
    const THUNK_SEH: u32 = 0x1168;
    const THUNK_FH3: u32 = 0x1170;
    const THUNK_FH4: u32 = 0x1178;
    const THUNK_SETJMP: u32 = 0x1180;
    const GS_WRAPPER_FH3: u32 = 0x1188;
    const GS_CHECK: u32 = 0x1198;
    const UNKNOWN_HANDLER: u32 = 0x11a8;

    const IMPORTS: u32 = 0x2460;
    const UNWIND_SEH: u32 = 0x25c0;
    const UNWIND_FH3: u32 = 0x2600;
    const UNWIND_FH4: u32 = 0x2610;
    const UNWIND_CATCH: u32 = 0x2620;
    const UNWIND_PLAIN: u32 = 0x2630;
    const UNWIND_GS_FH3: u32 = 0x2640;
    const UNWIND_GS: u32 = 0x2658;
    const UNWIND_UNKNOWN: u32 = 0x2668;
    const FH3_INFO: u32 = 0x2680;
    const FH3_UNWIND: u32 = 0x26a8;
    const FH3_TRY: u32 = 0x26b8;
    const FH3_HANDLERS: u32 = 0x26cc;
    const FH3_IP: u32 = 0x26e0;
    const FH4_INFO: u32 = 0x26f0;
    const FH4_UNWIND: u32 = 0x2700;
    const FH4_TRY: u32 = 0x2710;
    const FH4_HANDLERS: u32 = 0x2718;
    const FH4_IP: u32 = 0x2738;
    const FH4_CATCH_INFO: u32 = 0x2740;
    const FH4_SEGMENTS: u32 = 0x2750;
    const FH4_CATCH_IP: u32 = 0x2768;

    // Funclets and targets referenced by the metadata.
    const SEH_FILTER: u32 = 0x1154;
    const SEH_FINALLY: u32 = 0x1158;
    const FH3_CLEANUP: u32 = 0x115c;
    const FH4_CLEANUP: u32 = F_CLEANUP;
    const FH4_DTOR: u32 = 0x115e;
    const FH4_CATCH2: u32 = 0x1110;

    fn c(value: u32) -> Vec<u8> {
        encode_compressed(value)
    }

    fn le(value: u32) -> [u8; 4] {
        value.to_le_bytes()
    }

    /// `jmp [rip+slot]` at `at`.
    fn jmp_slot(file: &mut [u8], at: u32, slot: u32) {
        let disp = slot.wrapping_sub(at + 6);
        put(file, at, &[0xff, 0x25]);
        put(file, at + 2, &le(disp));
    }

    /// `call rel32` at `at`.
    fn call(file: &mut [u8], at: u32, target: u32) {
        put(file, at, &[0xe8]);
        put(file, at + 1, &le(target.wrapping_sub(at + 5)));
    }

    /// Replaces the import directory with `modules` (built at [`IMPORTS`]); returns the IAT
    /// slot RVA of every symbol, per module.
    fn set_imports(file: &mut [u8], modules: &[(&str, &[&str])]) -> Vec<Vec<u32>> {
        let mut at = IMPORTS + 20 * (modules.len() as u32 + 1);
        let mut slots = Vec::new();
        let mut thunks = Vec::new();
        for (_, names) in modules {
            thunks.push(at);
            slots.push((0..names.len() as u32).map(|i| at + 8 * i).collect());
            at += 8 * (names.len() as u32 + 1);
        }
        for (i, (dll, names)) in modules.iter().enumerate() {
            let dll_rva = at;
            put(file, at, dll.as_bytes());
            put(file, at + dll.len() as u32, &[0]);
            at += dll.len() as u32 + 1;
            for (j, name) in names.iter().enumerate() {
                at += at & 1;
                put(file, thunks[i] + 8 * j as u32, &u64::from(at).to_le_bytes());
                put(file, at, &[0, 0]);
                put(file, at + 2, name.as_bytes());
                put(file, at + 2 + name.len() as u32, &[0]);
                at += 3 + name.len() as u32;
            }
            put(file, thunks[i] + 8 * names.len() as u32, &[0; 8]);
            let desc = IMPORTS + 20 * i as u32;
            put(file, desc, &le(thunks[i]));
            put(file, desc + 12, &le(dll_rva));
            put(file, desc + 16, &le(thunks[i]));
        }
        assert!(
            at <= 0x2560,
            "import fixture overflows into .pdata ({at:#x})"
        );
        put(file, IMPORTS + 20 * modules.len() as u32, &[0; 20]);
        // Import data directory (index 1).
        file[0x108 + 8..0x108 + 12].copy_from_slice(&le(IMPORTS));
        file[0x108 + 12..0x108 + 16].copy_from_slice(&le(20 * (modules.len() as u32 + 1)));
        slots
    }

    /// Handler RVA + data, as the `UNWIND_INFO` trailer.
    fn trailer(handler: u32, data: &[u8]) -> Vec<u8> {
        [&le(handler)[..], data].concat()
    }

    struct Fixture {
        file: Vec<u8>,
        setjmp_slot: u32,
    }

    /// Synthetic PE with one function per handler kind (see the layout constants).
    fn fixture() -> Fixture {
        let mut file = pe_with_pdata(&[
            (F_SEH, F_SEH + 0x20, UNWIND_SEH),
            (F_FH3, F_FH3 + 0x20, UNWIND_FH3),
            (F_FH4, F_FH4 + 0x40, UNWIND_FH4),
            (F_CATCH, F_CATCH + 0x20, UNWIND_CATCH),
            (F_GS_FH3, F_GS_FH3 + 0x10, UNWIND_GS_FH3),
            (F_GS, F_GS + 0x10, UNWIND_GS),
            (F_UNKNOWN, F_UNKNOWN + 0x10, UNWIND_UNKNOWN),
            (F_CLEANUP, F_CLEANUP + 0x10, UNWIND_PLAIN),
        ]);
        let slots = set_imports(
            &mut file,
            &[
                (
                    "VCRUNTIME140.dll",
                    &[
                        "__C_specific_handler",
                        "__CxxFrameHandler3",
                        "__intrinsic_setjmp",
                    ],
                ),
                ("VCRUNTIME140_1.dll", &["__CxxFrameHandler4"]),
            ],
        );
        let (seh, fh3, setjmp, fh4) = (slots[0][0], slots[0][1], slots[0][2], slots[1][0]);

        // Handler routines.
        put(&mut file, GS_HELPER, &[0xc3]);
        jmp_slot(&mut file, THUNK_SEH, seh);
        jmp_slot(&mut file, THUNK_FH3, fh3);
        jmp_slot(&mut file, THUNK_FH4, fh4);
        jmp_slot(&mut file, THUNK_SETJMP, setjmp);
        call(&mut file, GS_WRAPPER_FH3, GS_HELPER);
        call(&mut file, GS_WRAPPER_FH3 + 5, THUNK_FH3);
        put(&mut file, GS_WRAPPER_FH3 + 10, &[0xc3]);
        call(&mut file, GS_CHECK, GS_HELPER);
        put(&mut file, GS_CHECK + 5, &[0xb8, 1, 0, 0, 0, 0xc3]);
        put(&mut file, UNKNOWN_HANDLER, &[0x31, 0xc0, 0xc3]);

        // __C_specific_handler: __except(filter), __except(1), __finally.
        let scopes: Vec<u8> = [
            le(3),
            le(F_SEH + 2),
            le(F_SEH + 8),
            le(SEH_FILTER),
            le(F_SEH + 0x10),
            le(F_SEH + 2),
            le(F_SEH + 8),
            le(EXCEPTION_EXECUTE_HANDLER),
            le(F_SEH + 0x14),
            le(F_SEH + 8),
            le(F_SEH + 0xc),
            le(SEH_FINALLY),
            le(0),
        ]
        .concat();
        let flags = UNW_FLAG_EHANDLER | UNW_FLAG_UHANDLER;
        put(
            &mut file,
            UNWIND_SEH,
            &unwind_bytes(flags, &trailer(THUNK_SEH, &scopes)),
        );

        // __CxxFrameHandler3 FuncInfo (magic ..22): 2 states, 1 try block, 2 IP map entries.
        put(
            &mut file,
            UNWIND_FH3,
            &unwind_bytes(flags, &trailer(THUNK_FH3, &le(FH3_INFO))),
        );
        let info3: Vec<u8> = [
            le(0x1993_0522),
            le(2),
            le(FH3_UNWIND),
            le(1),
            le(FH3_TRY),
            le(2),
            le(FH3_IP),
            le(0x28),
            le(0),
            le(1),
        ]
        .concat();
        put(&mut file, FH3_INFO, &info3);
        put(
            &mut file,
            FH3_UNWIND,
            &[le(u32::MAX), le(FH3_CLEANUP), le(0), le(0)].concat(),
        );
        put(
            &mut file,
            FH3_TRY,
            &[le(1), le(1), le(1), le(1), le(FH3_HANDLERS)].concat(),
        );
        put(
            &mut file,
            FH3_HANDLERS,
            &[le(0x8), le(0x2700), le(0x30), le(F_CATCH), le(0x38)].concat(),
        );
        put(
            &mut file,
            FH3_IP,
            &[le(F_FH3), le(u32::MAX), le(F_FH3 + 4), le(0)].concat(),
        );

        // __CxxFrameHandler4 parent: unwind map, try map with two handlers, IP map.
        put(
            &mut file,
            UNWIND_FH4,
            &unwind_bytes(flags, &trailer(THUNK_FH4, &le(FH4_INFO))),
        );
        let info4 = [
            &[FH4_UNWIND_MAP | FH4_TRY_BLOCK_MAP][..],
            &le(FH4_UNWIND),
            &le(FH4_TRY),
            &le(FH4_IP),
        ]
        .concat();
        put(&mut file, FH4_INFO, &info4);
        // Entry starts: 1, 6, 12; next states -1, 0, 1.
        let unwind4 = [
            c(3),
            c((1 << 2) | 3),
            le(FH4_CLEANUP).to_vec(),
            c((5 << 2) | 1),
            le(FH4_DTOR).to_vec(),
            c(0x40),
            c(6 << 2),
        ]
        .concat();
        put(&mut file, FH4_UNWIND, &unwind4);
        let try4 = [c(1), c(1), c(1), c(2), le(FH4_HANDLERS).to_vec()].concat();
        put(&mut file, FH4_TRY, &try4);
        let handlers4 = [
            c(2),
            // adjectives + dispType + one function-relative continuation.
            vec![FH4_HANDLER_ADJECTIVES | FH4_HANDLER_DISP_TYPE | (1 << FH4_HANDLER_CONT_SHIFT)],
            c(8),
            le(0x2700).to_vec(),
            le(F_CATCH).to_vec(),
            c(0x30),
            // catch(...) with two RVA continuations.
            vec![FH4_HANDLER_CONT_IS_RVA | (2 << FH4_HANDLER_CONT_SHIFT)],
            le(FH4_CATCH2).to_vec(),
            le(F_FH4 + 0x34).to_vec(),
            le(F_FH4 + 0x38).to_vec(),
        ]
        .concat();
        put(&mut file, FH4_HANDLERS, &handlers4);
        let ip4 = [c(3), c(0), c(0), c(0x10), c(2), c(0x20), c(1)].concat();
        put(&mut file, FH4_IP, &ip4);

        // FH4 catch funclet: separated code, establisher frame offset, its own IP map, the
        // parent's unwind map and no try blocks of its own.
        put(
            &mut file,
            UNWIND_CATCH,
            &unwind_bytes(flags, &trailer(THUNK_FH4, &le(FH4_CATCH_INFO))),
        );
        let catch_info = [
            &[FH4_IS_CATCH | FH4_IS_SEPARATED | FH4_UNWIND_MAP][..],
            &le(FH4_UNWIND),
            &le(FH4_SEGMENTS),
            &c(0x38),
        ]
        .concat();
        put(&mut file, FH4_CATCH_INFO, &catch_info);
        let segments = [
            c(2),
            le(F_FH4).to_vec(),
            le(FH4_IP).to_vec(),
            le(F_CATCH).to_vec(),
            le(FH4_CATCH_IP).to_vec(),
        ]
        .concat();
        put(&mut file, FH4_SEGMENTS, &segments);
        put(&mut file, FH4_CATCH_IP, &[c(1), c(4), c(2)].concat());

        // GS wrapper around FH3 (data: FuncInfo RVA + GS data), GS check, unknown handler.
        put(
            &mut file,
            UNWIND_GS_FH3,
            &unwind_bytes(
                flags,
                &trailer(GS_WRAPPER_FH3, &[le(FH3_INFO), le(0x20)].concat()),
            ),
        );
        put(
            &mut file,
            UNWIND_GS,
            &unwind_bytes(flags, &trailer(GS_CHECK, &le(0x20))),
        );
        put(
            &mut file,
            UNWIND_UNKNOWN,
            &unwind_bytes(UNW_FLAG_EHANDLER, &trailer(UNKNOWN_HANDLER, &le(0))),
        );
        put(&mut file, UNWIND_PLAIN, &unwind_bytes(0, &[]));
        Fixture {
            file,
            setjmp_slot: setjmp,
        }
    }

    fn parse(file: &[u8]) -> EhMetadata {
        let pe = Pe64::parse(file).expect("synthetic PE parses");
        let dir = ExceptionDirectory::from_pe(&pe, file);
        assert!(dir.diagnostics.is_empty(), "{:?}", dir.diagnostics);
        EhMetadata::from_pe(&pe, file, &dir)
    }

    fn function(eh: &EhMetadata, begin: u32) -> &EhFunction {
        eh.functions
            .iter()
            .find(|f| f.begin == begin)
            .unwrap_or_else(|| panic!("no EH function at {begin:#x}"))
    }

    fn diag_kinds(eh: &EhMetadata) -> Vec<(u32, EhDiagnosticKind)> {
        eh.diagnostics.iter().map(|d| (d.rva, d.kind)).collect()
    }

    // ----- compressed integers ---------------------------------------------------------------

    #[test]
    fn compressed_integers_decode_every_length() {
        assert_eq!(decode_compressed(&[0x00]), Some((0, 1)));
        assert_eq!(decode_compressed(&[0xfe]), Some((0x7f, 1)));
        // 2 bytes: low bits 01, value in the upper 14 bits.
        assert_eq!(decode_compressed(&[0x01, 0x02]), Some((0x80, 2)));
        assert_eq!(decode_compressed(&[0xfd, 0xff]), Some((0x3fff, 2)));
        // 3 bytes: low bits 011.
        assert_eq!(decode_compressed(&[0x03, 0x00, 0x02]), Some((0x4000, 3)));
        // 4 bytes: low bits 0111.
        assert_eq!(
            decode_compressed(&[0xf7, 0xff, 0xff, 0xff]),
            Some((0x0fff_ffff, 4))
        );
        // 5 bytes: low nibble 1111, the high nibble is ignored and the value is the next u32.
        assert_eq!(
            decode_compressed(&[0x0f, 0x78, 0x56, 0x34, 0x12]),
            Some((0x1234_5678, 5))
        );
        assert_eq!(
            decode_compressed(&[0xff, 0xff, 0xff, 0xff, 0xff]),
            Some((u32::MAX, 5))
        );
        // Trailing bytes are not consumed.
        assert_eq!(decode_compressed(&[0x04, 0xaa]), Some((2, 1)));
    }

    #[test]
    fn compressed_integers_reject_truncation_and_round_trip() {
        assert_eq!(decode_compressed(&[]), None);
        assert_eq!(decode_compressed(&[0x01]), None);
        assert_eq!(decode_compressed(&[0x03, 0x00]), None);
        assert_eq!(decode_compressed(&[0x07, 0, 0]), None);
        assert_eq!(decode_compressed(&[0x0f, 0, 0, 0]), None);
        let edges = [
            0,
            1,
            0x7f,
            0x80,
            0x3fff,
            0x4000,
            0x1f_ffff,
            0x20_0000,
            0x0fff_ffff,
            0x1000_0000,
            u32::MAX,
        ];
        for value in edges {
            let bytes = encode_compressed(value);
            assert_eq!(
                decode_compressed(&bytes),
                Some((value, bytes.len())),
                "{value:#x}"
            );
        }
        let lengths: Vec<usize> = edges.iter().map(|&v| encode_compressed(v).len()).collect();
        assert_eq!(lengths, [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);
    }

    // ----- handlers ----------------------------------------------------------------------------

    #[test]
    fn identifies_handlers_through_thunks_and_wrappers() {
        let fx = fixture();
        let eh = parse(&fx.file);
        let thunk = |kind| Handler {
            kind,
            wrapper: false,
        };
        let wrapper = |kind| Handler {
            kind,
            wrapper: true,
        };
        assert_eq!(
            eh.handlers
                .iter()
                .map(|(r, h)| (*r, *h))
                .collect::<Vec<_>>(),
            [
                (THUNK_SEH, thunk(HandlerKind::CSpecific)),
                (THUNK_FH3, thunk(HandlerKind::CxxFrameHandler3)),
                (THUNK_FH4, thunk(HandlerKind::CxxFrameHandler4)),
                (GS_WRAPPER_FH3, wrapper(HandlerKind::CxxFrameHandler3)),
                (GS_CHECK, wrapper(HandlerKind::GsCheck)),
            ]
        );
        assert_eq!(
            diag_kinds(&eh),
            [(UNKNOWN_HANDLER, EhDiagnosticKind::UnknownHandler)]
        );
        assert_eq!(eh.unrecognised, [F_UNKNOWN]);
        assert_eq!(eh.setjmp_slots, [fx.setjmp_slot]);
        assert_eq!(function(&eh, F_GS).data, LanguageData::None);
        // The GS wrapper passes its data through: it is the FH3 FuncInfo.
        assert!(
            matches!(&function(&eh, F_GS_FH3).data, LanguageData::Cxx3(info) if info.rva == FH3_INFO)
        );
        let stats = eh.stats();
        assert_eq!(
            (
                stats.c_specific,
                stats.cxx3,
                stats.cxx4,
                stats.gs_check,
                stats.unknown
            ),
            (1, 2, 2, 1, 1)
        );
        assert_eq!((stats.parsed, stats.diagnostics), (6, 1));
    }

    #[test]
    fn handler_reaching_two_language_handlers_is_ambiguous() {
        let mut fx = fixture();
        // The unknown handler now calls both the FH3 and the FH4 thunk.
        call(&mut fx.file, UNKNOWN_HANDLER, THUNK_FH3);
        call(&mut fx.file, UNKNOWN_HANDLER + 5, THUNK_FH4);
        put(&mut fx.file, UNKNOWN_HANDLER + 10, &[0xc3]);
        let eh = parse(&fx.file);
        assert_eq!(
            diag_kinds(&eh),
            [(UNKNOWN_HANDLER, EhDiagnosticKind::AmbiguousHandler)]
        );
    }

    // ----- formats -----------------------------------------------------------------------------

    #[test]
    fn parses_scope_tables() {
        let eh = parse(&fixture().file);
        let f = function(&eh, F_SEH);
        assert_eq!(f.data_rva, UNWIND_SEH + 12);
        assert_eq!(
            f.data,
            LanguageData::ScopeTable(vec![
                ScopeRecord {
                    begin: F_SEH + 2,
                    end: F_SEH + 8,
                    handler: ScopeHandler::Filter(SEH_FILTER),
                    jump_target: F_SEH + 0x10,
                },
                ScopeRecord {
                    begin: F_SEH + 2,
                    end: F_SEH + 8,
                    handler: ScopeHandler::Constant(EXCEPTION_EXECUTE_HANDLER),
                    jump_target: F_SEH + 0x14,
                },
                ScopeRecord {
                    begin: F_SEH + 8,
                    end: F_SEH + 0xc,
                    handler: ScopeHandler::Finally(SEH_FINALLY),
                    jump_target: 0,
                },
            ])
        );
        assert_eq!(
            f.funclets(),
            [
                (SEH_FILTER, FuncletKind::Filter),
                (SEH_FINALLY, FuncletKind::Finally)
            ]
        );
        assert_eq!(
            f.resume_targets(),
            [
                (F_SEH + 0x10, ResumeKind::ExceptBody),
                (F_SEH + 0x14, ResumeKind::ExceptBody)
            ]
        );
    }

    #[test]
    fn parses_fh3_func_info() {
        let eh = parse(&fixture().file);
        let LanguageData::Cxx3(info) = &function(&eh, F_FH3).data else {
            panic!("FH3 data");
        };
        assert_eq!(
            (
                info.magic,
                info.bbt_flags,
                info.unwind_help,
                info.es_type_list,
                info.eh_flags
            ),
            (0x1993_0522, 0, 0x28, 0, 1)
        );
        assert_eq!(
            info.unwind_map,
            [
                UnwindEntry3 {
                    to_state: -1,
                    action: FH3_CLEANUP
                },
                UnwindEntry3 {
                    to_state: 0,
                    action: 0
                }
            ]
        );
        assert_eq!(
            info.try_blocks,
            [TryBlock3 {
                try_low: 1,
                try_high: 1,
                catch_high: 1,
                handlers: vec![CatchHandler3 {
                    adjectives: 8,
                    type_descriptor: 0x2700,
                    catch_object: 0x30,
                    handler: F_CATCH,
                    frame: 0x38,
                }],
            }]
        );
        assert_eq!(info.ip_to_state, [(F_FH3, -1), (F_FH3 + 4, 0)]);
        let f = function(&eh, F_FH3);
        assert_eq!(
            f.funclets(),
            [
                (F_CATCH, FuncletKind::Catch),
                (FH3_CLEANUP, FuncletKind::Cleanup)
            ]
        );
        // FH3 catch funclets return the continuation in rax: nothing in the metadata.
        assert!(f.resume_targets().is_empty());
    }

    #[test]
    fn parses_fh4_func_info() {
        let eh = parse(&fixture().file);
        let LanguageData::Cxx4(info) = &function(&eh, F_FH4).data else {
            panic!("FH4 data");
        };
        assert!(!info.is_catch() && !info.is_separated());
        assert_eq!(
            (
                info.unwind_map_rva,
                info.try_block_map_rva,
                info.ip_map_rva,
                info.frame
            ),
            (FH4_UNWIND, FH4_TRY, FH4_IP, None)
        );
        assert_eq!(
            info.unwind_map,
            [
                UnwindEntry4 {
                    kind: UnwindKind4::Funclet,
                    next_state: -1,
                    action: FH4_CLEANUP,
                    object: 0
                },
                UnwindEntry4 {
                    kind: UnwindKind4::DtorWithObj,
                    next_state: 0,
                    action: FH4_DTOR,
                    object: 0x40
                },
                UnwindEntry4 {
                    kind: UnwindKind4::None,
                    next_state: 1,
                    action: 0,
                    object: 0
                },
            ]
        );
        assert_eq!(info.try_blocks.len(), 1);
        let block = &info.try_blocks[0];
        assert_eq!(
            (
                block.try_low,
                block.try_high,
                block.catch_high,
                block.handler_array
            ),
            (1, 1, 2, FH4_HANDLERS)
        );
        assert_eq!(
            block.handlers,
            [
                CatchHandler4 {
                    header: 0x13,
                    adjectives: 8,
                    type_descriptor: 0x2700,
                    catch_object: 0,
                    handler: F_CATCH,
                    continuations: vec![F_FH4 + 0x30],
                },
                CatchHandler4 {
                    header: 0x28,
                    adjectives: 0,
                    type_descriptor: 0,
                    catch_object: 0,
                    handler: FH4_CATCH2,
                    continuations: vec![F_FH4 + 0x34, F_FH4 + 0x38],
                },
            ]
        );
        assert_eq!(
            info.ip_to_state,
            [(F_FH4, -1), (F_FH4 + 0x10, 1), (F_FH4 + 0x30, 0)]
        );
        let f = function(&eh, F_FH4);
        assert_eq!(
            f.funclets(),
            [
                (F_CATCH, FuncletKind::Catch),
                (FH4_CATCH2, FuncletKind::Catch),
                (FH4_CLEANUP, FuncletKind::Cleanup),
                (FH4_DTOR, FuncletKind::Destructor),
            ]
        );
        assert_eq!(
            f.resume_targets(),
            [0x30, 0x34, 0x38].map(|off| (F_FH4 + off, ResumeKind::CatchContinuation))
        );
    }

    #[test]
    fn parses_fh4_separated_catch_funclet() {
        let eh = parse(&fixture().file);
        let LanguageData::Cxx4(info) = &function(&eh, F_CATCH).data else {
            panic!("FH4 data");
        };
        assert!(info.is_catch() && info.is_separated());
        assert_eq!(info.frame, Some(0x38));
        // The segment table entry for this funclet selects its own IP map.
        assert_eq!(info.ip_map_rva, FH4_CATCH_IP);
        assert_eq!(info.ip_to_state, [(F_CATCH + 4, 1)]);
        assert_eq!(info.unwind_map.len(), 3);
        assert!(info.try_blocks.is_empty());
        assert!(function(&eh, F_CATCH).resume_targets().is_empty());
    }

    // ----- malformed input ---------------------------------------------------------------------

    fn single_diag(eh: &EhMetadata, begin: u32) -> Vec<(u32, EhDiagnosticKind)> {
        eh.diagnostics
            .iter()
            .filter(|d| d.rva == begin)
            .map(|d| (d.address, d.kind))
            .collect()
    }

    #[test]
    fn malformed_fh3_is_diagnosed() {
        // Unknown magic.
        let mut fx = fixture();
        put(&mut fx.file, FH3_INFO, &le(0x1993_0523));
        let eh = parse(&fx.file);
        assert_eq!(
            single_diag(&eh, F_FH3),
            [(FH3_INFO, EhDiagnosticKind::UnknownMagic)]
        );
        assert_eq!(function(&eh, F_FH3).data, LanguageData::None);

        // Huge state count and a try map running off the end of .rdata.
        let mut fx = fixture();
        put(&mut fx.file, FH3_INFO + 4, &le(0x8000_0000));
        put(&mut fx.file, FH3_INFO + 16, &le(0x27f0));
        let eh = parse(&fx.file);
        assert_eq!(
            single_diag(&eh, F_FH3),
            [
                (FH3_UNWIND, EhDiagnosticKind::TooLarge),
                (0x27f0, EhDiagnosticKind::Truncated)
            ]
        );

        // FuncInfo RVA outside the image.
        let mut fx = fixture();
        put(&mut fx.file, UNWIND_FH3 + 12, &le(0x9000));
        let eh = parse(&fx.file);
        assert_eq!(
            single_diag(&eh, F_FH3),
            [(0x9000, EhDiagnosticKind::Truncated)]
        );

        // A catch funclet that is not code.
        let mut fx = fixture();
        put(&mut fx.file, FH3_HANDLERS + 12, &le(0x2010));
        let eh = parse(&fx.file);
        assert_eq!(
            single_diag(&eh, F_FH3),
            [(0x2010, EhDiagnosticKind::NotCode)]
        );
    }

    #[test]
    fn malformed_fh4_is_diagnosed() {
        // Reserved continuation count 3 (two values are still read).
        let mut fx = fixture();
        put(
            &mut fx.file,
            FH4_HANDLERS + 12,
            &[FH4_HANDLER_CONT_IS_RVA | (3 << FH4_HANDLER_CONT_SHIFT)],
        );
        let eh = parse(&fx.file);
        assert_eq!(
            single_diag(&eh, F_FH4),
            [(FH4_HANDLERS, EhDiagnosticKind::Reserved)]
        );

        // An unwind link that lands inside an entry.
        let mut fx = fixture();
        put(&mut fx.file, FH4_UNWIND + 6, &c((3 << 2) | 1));
        let eh = parse(&fx.file);
        assert_eq!(
            single_diag(&eh, F_FH4),
            [(FH4_UNWIND, EhDiagnosticKind::BadUnwindLink)]
        );

        // Separated funclet without a segment entry.
        let mut fx = fixture();
        put(&mut fx.file, FH4_SEGMENTS + 9, &le(0x1111));
        let eh = parse(&fx.file);
        assert_eq!(
            single_diag(&eh, F_CATCH),
            [(FH4_SEGMENTS, EhDiagnosticKind::MissingSegment)]
        );

        // Counts above the limit (5-byte compressed integer) and a map at the end of .rdata.
        let mut fx = fixture();
        put(&mut fx.file, FH4_TRY, &[0x0f, 1, 0, 1, 0]);
        put(&mut fx.file, FH4_INFO + 9, &le(0x27ff));
        put(&mut fx.file, 0x27ff, &[0x01]);
        let eh = parse(&fx.file);
        assert_eq!(
            single_diag(&eh, F_FH4),
            [
                (FH4_TRY, EhDiagnosticKind::TooLarge),
                (0x27ff, EhDiagnosticKind::Truncated)
            ]
        );

        // FuncInfo4 cut off by the end of the section.
        let mut fx = fixture();
        put(&mut fx.file, UNWIND_FH4 + 12, &le(0x27fc));
        put(
            &mut fx.file,
            0x27fc,
            &[FH4_UNWIND_MAP | FH4_TRY_BLOCK_MAP, 0, 0, 0],
        );
        let eh = parse(&fx.file);
        assert_eq!(
            single_diag(&eh, F_FH4),
            [(0x27fc, EhDiagnosticKind::Truncated)]
        );
    }

    #[test]
    fn malformed_scope_tables_are_diagnosed() {
        let mut fx = fixture();
        // Count 0x10001 exceeds the limit.
        put(&mut fx.file, UNWIND_SEH + 12, &le(MAX_TABLE_ENTRIES + 1));
        let eh = parse(&fx.file);
        assert_eq!(
            single_diag(&eh, F_SEH),
            [(UNWIND_SEH + 12, EhDiagnosticKind::TooLarge)]
        );

        let mut fx = fixture();
        // A filter "constant" that is neither code nor EXCEPTION_EXECUTE_HANDLER, and an
        // __except body in data.
        put(&mut fx.file, UNWIND_SEH + 16 + 8, &le(7));
        put(&mut fx.file, UNWIND_SEH + 16 + 12, &le(0x2004));
        let eh = parse(&fx.file);
        assert_eq!(
            single_diag(&eh, F_SEH),
            [
                (0x2004, EhDiagnosticKind::NotCode),
                (7, EhDiagnosticKind::NotCode)
            ]
        );
    }

    #[test]
    fn garbage_data_never_panics() {
        let fx = fixture();
        let mut seed = 0x9e37_79b9u32;
        for round in 0..64 {
            let mut file = fx.file.clone();
            // Scramble the EH data area (unwind infos stay intact for the first rounds).
            let from = if round < 32 { FH3_INFO } else { UNWIND_SEH };
            for rva in from..0x2800 {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                put(&mut file, rva, &[(seed >> 16) as u8]);
            }
            let pe = Pe64::parse(&file).expect("parses");
            let dir = ExceptionDirectory::from_pe(&pe, &file);
            let eh = EhMetadata::from_pe(&pe, &file, &dir);
            let _ = eh.stats();
        }
    }

    #[test]
    fn parse_is_deterministic() {
        let fx = fixture();
        assert_eq!(parse(&fx.file), parse(&fx.file));
    }

    // ----- discovery seeds ---------------------------------------------------------------------

    #[test]
    fn seed_adds_funclets_resume_targets_and_setjmp_slots() {
        let fx = fixture();
        let pe = Pe64::parse(&fx.file).expect("parses");
        let dir = ExceptionDirectory::from_pe(&pe, &fx.file);
        let eh = EhMetadata::from_pe(&pe, &fx.file, &dir);
        let mut image = CodeImage::from_pe(&pe, &fx.file);
        dir.seed(&mut image);
        let before: BTreeSet<u64> = image.entry_points.iter().copied().collect();
        eh.seed(&mut image);
        let after: BTreeSet<u64> = image.entry_points.iter().copied().collect();
        let added: Vec<u32> = after
            .difference(&before)
            .map(|va| (va - BASE) as u32)
            .collect();
        // Funclets without their own .pdata record (F_CATCH and F_CLEANUP already have one).
        assert_eq!(
            added,
            [FH4_CATCH2, SEH_FILTER, SEH_FINALLY, FH3_CLEANUP, FH4_DTOR]
        );
        let va = |rva: u32| BASE + u64::from(rva);
        assert_eq!(
            image.eh.resume_targets,
            [
                (va(F_SEH), va(F_SEH + 0x10)),
                (va(F_SEH), va(F_SEH + 0x14)),
                (va(F_FH4), va(F_FH4 + 0x30)),
                (va(F_FH4), va(F_FH4 + 0x34)),
                (va(F_FH4), va(F_FH4 + 0x38)),
            ]
        );
        assert_eq!(image.eh.setjmp_slots, [va(fx.setjmp_slot)]);
        assert_eq!(image.eh.stats, eh.stats());
        assert_eq!(image.eh.stats.resume_targets, 5);
    }
}
