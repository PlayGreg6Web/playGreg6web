#!/usr/bin/env python3
"""Real-binary PE64 import coverage auditor for playgta6web (Track 5 / W37).

Parses all required imports from `docs/imports_required.md` (68 DLLs, 587 symbols
from `game_scarlett_master.exe`) and audits live implementation coverage across
`runtime/os/*.{h,cpp}`, `runtime/gpu/*.{h,cpp}`, and `runtime/guest/*.{h,cpp}`:

- `Win32ModuleRegistry` registrations (`register_export`, `register_exports`,
  `register_symbol`, `register_alias`, `register_ordinal`, `register_api_set_rule`,
  `add_export_locked`, `add_api_set_rule_locked`, `REGISTER_*` macros, `E(...)`
  X-macro export tables, and `{ "dll", "symbol", ... }` struct arrays).
- `extern "C"` C-ABI shim declarations (`pg6_win32_*`, `pg6_gdk_*`, `pg6_xbox_*`,
  `pg6_crt_*`, `pg6_mw_*`, `pg6_d3d12_*`) in runtime headers, matched via
  normalized symbol names and Win32 API-set forwarding (`api-ms-win-core-*` ->
  `kernel32.dll` / `kernelbase.dll`, etc.).

Classifies every import as:
- `shimmed`: implemented/registered in `runtime/`
- `guest_recompiled`: redistributable DLL handled by guest recompilation (default:
  `msvcp140.dll` per R12 #161)
- `missing`: remaining unimplemented import required for 100% real-binary boot

Security & privacy invariant:
- Machine-readable `--json` output NEVER includes raw host filesystem paths or usernames.
- Strictly uses Python 3.12 standard library modules only.

Usage:
    python tools/check_import_coverage.py [--repo-root <path>] [--json] [--missing-only] [--min-pct <float>]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
IMPORTS_REQUIRED_REL = "docs/imports_required.md"
RUNTIME_SCAN_DIRS: tuple[str, ...] = ("runtime/os", "runtime/gpu", "runtime/guest")
DEFAULT_GUEST_RECOMPILED_DLLS: frozenset[str] = frozenset({"msvcp140.dll"})

# Default Win32 API-set contract forwarding rules (merged with rules parsed from runtime/).
# Longest matching prefix wins; each prefix maps to an ordered list of candidate host DLLs.
DEFAULT_API_SET_RULES: dict[str, tuple[str, ...]] = {
    "api-ms-win-core-": ("kernel32.dll", "kernelbase.dll"),
    "api-ms-win-core-com-": ("combase.dll", "ole32.dll"),
    "api-ms-win-core-winrt-": ("combase.dll",),
    "api-ms-win-crt-": ("ucrtbase.dll",),
    "api-ms-win-eventing-": ("advapi32.dll",),
    "api-ms-win-security-": ("advapi32.dll", "kernelbase.dll"),
    "ext-ms-win-rtcore-ntuser-": ("user32.dll",),
}

# Equivalent Win32 host DLL pairs (e.g. api-ms-win-core-* functions live in kernel32 or kernelbase).
_EQUIVALENT_HOST_DLLS: dict[str, tuple[str, ...]] = {
    "kernel32.dll": ("kernelbase.dll",),
    "kernelbase.dll": ("kernel32.dll",),
}

# Mapping of `pg6_<family>_*` header declaration prefixes to canonical DLLs / prefixes.
_HEADER_PREFIX_EXACT_DLLS: dict[str, frozenset[str]] = {
    "pg6_win32_": frozenset(
        {
            "kernel32.dll",
            "kernelbase.dll",
            "advapi32.dll",
            "ws2_32.dll",
            "winhttp.dll",
            "bcrypt.dll",
            "user32.dll",
            "combase.dll",
            "ole32.dll",
            "ntdll.dll",
            "wer.dll",
        }
    ),
    "pg6_gdk_": frozenset(
        {
            "xgameruntime.dll",
            "microsoft.xbox.services.141.gdk.c.thunks.dll",
            "microsoft.xbox.services.gdk.c.thunks.dll",
        }
    ),
    "pg6_xbox_": frozenset(
        {
            "xmem.dll",
            "xcrdapi.dll",
            "xg_xs.dll",
            "gameinput.dll",
        }
    ),
    "pg6_crt_": frozenset(
        {
            "ucrtbase.dll",
            "vcruntime140.dll",
            "vcruntime140_1.dll",
        }
    ),
    "pg6_mw_": frozenset(
        {
            "libhttpclient.gdk.dll",
            "bink2scarlett.dll",
            "acphal.dll",
            "xapu.dll",
            "chad.dll",
            "amd_fidelityfx_loader_xbox.dll",
            "xboxbvhbuilder.dll",
            "mfplat.dll",
            "mfreadwrite.dll",
        }
    ),
    "pg6_d3d12_": frozenset(
        {
            "d3d12_xs.dll",
            "d3d12.dll",
        }
    ),
}

_HEADER_PREFIX_DLL_PREFIXES: dict[str, tuple[str, ...]] = {
    "pg6_win32_": (
        "api-ms-win-core-",
        "api-ms-win-eventing-",
        "api-ms-win-security-",
        "ext-ms-win-rtcore-ntuser-",
    ),
    "pg6_gdk_": ("microsoft.xbox.services.",),
    "pg6_crt_": ("api-ms-win-crt-",),
}

# Regexes for path/username sanitization (mirrors tools/repo_health.py).
_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"'<>|,;()\[\]]+")
_UNC_PATH_RE = re.compile(r"\\\\[^\s\"'<>|,;()\[\]]+")
_POSIX_PATH_RE = re.compile(
    r"/(?:Users|home|tmp|var|opt|root|mnt|private|etc|usr|Library|Volumes)/[^\s\"'<>|,;()\[\]]*"
)

# Markdown parsing regexes for docs/imports_required.md.
_H2_RE = re.compile(r"^##\s+(.+?)\s*$")
_DLL_SECTION_RE = re.compile(r"^###\s+`([^`]+)`(?:\s*\((\d+)\s+symbols?\))?\s*$")
_ORDINAL_TEXT_RE = re.compile(r"(?:#|ordinal\s+`?)(\d+)`?", re.IGNORECASE)
_HINT_HEX_RE = re.compile(r"hint\s+`?0[xX]([0-9a-fA-F]+)`?", re.IGNORECASE)

# C/C++ extraction regexes.
_DEFINE_STR_RE = re.compile(r'^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)\s+"([^"]+)"', re.MULTILINE)
_MOD_TAG_DEFINE_RE = re.compile(r"^(?:PG6_)?(?:[A-Z0-9_]*MOD)_([A-Z0-9_]+)$")
_CONST_STR_RE = re.compile(
    r'\b(?:constexpr\s+)?(?:static\s+)?(?:const\s+)?'
    r'(?:char\s*\*\s*(?:const\s+)?|char\s+|(?:std::)?string_view\s+)'
    r'([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[\s*\])?\s*=\s*"([^"]+)"\s*;'
)
_STRING_LITERAL_RE = re.compile(r'^"((?:[^"\\]|\\.)*)"$')
_C_INT_RE = re.compile(r"^(0[xX][0-9a-fA-F]+|\d+)[uUlL]*$")
_EXTERN_C_BLOCK_RE = re.compile(r'\bextern\s+"C"\s*\{')
_EXTERN_C_SINGLE_RE = re.compile(
    r'\bextern\s+"C"\s+[^{;\n]*?\b(pg6_(?:win32|gdk|xbox|crt|mw|d3d12)_[A-Za-z0-9_]+)\s*\('
)
_PG6_HEADER_FN_RE = re.compile(r"\b(pg6_(?:win32|gdk|xbox|crt|mw|d3d12)_[A-Za-z0-9_]+)\s*\(")
_PG6_PREFIX_RE = re.compile(r"^(pg6_(?:win32|gdk|xbox|crt|mw|d3d12)_)(.+)$")
_REGISTER_CALL_RE = re.compile(
    r"\b(?:pg6_loader_)?"
    r"(register_export|add_export_locked|register_symbol|register_ordinal|register_alias|"
    r"register_api_set_rule|add_api_set_rule_locked)\s*\("
)
_REGISTER_EXPORTS_LIST_RE = re.compile(
    r"\b(?:pg6_loader_)?register_exports\s*\(\s*([^,]+?)\s*,\s*\{"
)
_REGISTER_EXPORTS_VAR_RE = re.compile(
    r"\b(?:pg6_loader_)?register_exports\s*\(\s*([^,]+?)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*,"
)
_EXPORT_ENTRY_ARRAY_RE = re.compile(
    r"\b(?:const\s+)?(?:Pg6)?ExportEntry\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*\]\s*=\s*\{"
)
_EXPORT_ENTRY_CALL_RE = re.compile(r"\bexport_entry\s*\(")
_XMACRO_E_RE = re.compile(r"(?<![A-Za-z0-9_])E\s*\(")
_REGISTER_MACRO_RE = re.compile(r"\b(REGISTER_[A-Z0-9_]+)\s*\(")
_TABLE_STRUCT_ROW_RE = re.compile(
    r'\{\s*("(?:[^"\\]|\\.)*"|[A-Za-z_][A-Za-z0-9_]*)\s*,\s*'
    r'("(?:[^"\\]|\\.)*"|nullptr|NULL|0)\s*,\s*'
    r"([^{};]+?)\s*\}"
)
_ALIAS_MIRROR_IF_RE = re.compile(
    r'if\s*\([^)]*==\s*("(?:[^"\\]|\\.)*"|[A-Za-z_][A-Za-z0-9_]*)\s*\)\s*\{[^}]*'
    r'register_export\s*\(\s*("(?:[^"\\]|\\.)*"|[A-Za-z_][A-Za-z0-9_]*)\s*,'
)


# ---------------------------------------------------------------------------
# Generic helpers & sanitization
# ---------------------------------------------------------------------------


def _current_usernames() -> list[str]:
    names: list[str] = []
    for candidate in (
        os.environ.get("USERNAME", ""),
        os.environ.get("USER", ""),
        os.environ.get("LOGNAME", ""),
    ):
        cleaned = candidate.strip()
        if len(cleaned) >= 3 and cleaned not in names:
            names.append(cleaned)
    try:
        home_name = Path.home().name.strip()
        if len(home_name) >= 3 and home_name not in names:
            names.append(home_name)
    except (OSError, RuntimeError):
        pass
    return names


def sanitize_message(text: str) -> str:
    """Strip any accidental host paths or usernames from diagnostic messages."""
    sanitized = _WINDOWS_PATH_RE.sub("<path>", text)
    sanitized = _UNC_PATH_RE.sub("<path>", sanitized)
    sanitized = _POSIX_PATH_RE.sub("<path>", sanitized)
    for username in _current_usernames():
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(username)}(?![A-Za-z0-9_])",
            re.IGNORECASE,
        )
        sanitized = pattern.sub("<user>", sanitized)
    return sanitized


def canonical_dll_name(name: str) -> str:
    """Normalize a Win32/Xbox module name (basename, ASCII lowercase, default `.dll`)."""
    cleaned = name.strip()
    if not cleaned:
        return ""
    sep_idx = max(cleaned.rfind("\\"), cleaned.rfind("/"))
    base = cleaned[sep_idx + 1 :] if sep_idx >= 0 else cleaned
    if not base:
        return ""
    if base.endswith("."):
        base = base[:-1]
    elif "." not in base:
        base = f"{base}.dll"
    return base.lower()


def normalize_symbol_key(name: str) -> str:
    """R5-compatible normalized symbol key: ASCII lowercase with underscores removed."""
    return name.replace("_", "").lower()


def _is_dll_like(value: str) -> bool:
    return value.lower().endswith(".dll") and len(value) > 4


def _strip_c_comments(text: str) -> str:
    """Remove `//` and `/* ... */` comments while preserving string literals and newlines."""
    out: list[str] = []
    i = 0
    n = len(text)
    in_string: str | None = None
    while i < n:
        ch = text[i]
        if in_string is not None:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == in_string:
                in_string = None
            i += 1
            continue

        if ch in ('"', "'"):
            in_string = ch
            out.append(ch)
            i += 1
            continue

        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            i += 2
            while i < n and text[i] != "\n":
                i += 1
            continue

        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                if text[i] == "\n":
                    out.append("\n")
                i += 1
            i += 2
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def _join_line_continuations(text: str) -> str:
    """Collapse C preprocessor backslash-newline continuations into single logical lines."""
    return re.sub(r"\\[ \t]*\r?\n", " ", text)


def _extract_balanced(
    text: str,
    open_idx: int,
    open_ch: str,
    close_ch: str,
) -> tuple[str | None, int]:
    """Return `(inner_text, close_idx)` for balanced delimiters starting at `open_idx`."""
    if open_idx < 0 or open_idx >= len(text) or text[open_idx] != open_ch:
        return None, open_idx
    depth = 0
    in_string: str | None = None
    i = open_idx
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string is not None:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == in_string:
                in_string = None
            i += 1
            continue
        if ch in ('"', "'"):
            in_string = ch
            i += 1
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i], i
        i += 1
    return None, n


def _split_top_level_args(arg_text: str) -> list[str]:
    """Split a comma-separated C/C++ argument list at nesting depth 0."""
    args: list[str] = []
    buf: list[str] = []
    paren = 0
    brace = 0
    bracket = 0
    in_string: str | None = None
    i = 0
    n = len(arg_text)
    while i < n:
        ch = arg_text[i]
        if in_string is not None:
            buf.append(ch)
            if ch == "\\" and i + 1 < n:
                buf.append(arg_text[i + 1])
                i += 2
                continue
            if ch == in_string:
                in_string = None
            i += 1
            continue

        if ch in ('"', "'"):
            in_string = ch
            buf.append(ch)
        elif ch == "(":
            paren += 1
            buf.append(ch)
        elif ch == ")":
            paren = max(0, paren - 1)
            buf.append(ch)
        elif ch == "{":
            brace += 1
            buf.append(ch)
        elif ch == "}":
            brace = max(0, brace - 1)
            buf.append(ch)
        elif ch == "[":
            bracket += 1
            buf.append(ch)
        elif ch == "]":
            bracket = max(0, bracket - 1)
            buf.append(ch)
        elif ch == "," and paren == 0 and brace == 0 and bracket == 0:
            args.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        i += 1

    tail = "".join(buf).strip()
    if tail or args:
        args.append(tail)
    return args


def _parse_c_int(token: str) -> int | None:
    cleaned = token.strip()
    # Unwrap simple static_cast<...>(expr)
    cast_match = re.fullmatch(r"static_cast\s*<[^>]+>\s*\(([^()]+)\)", cleaned)
    if cast_match:
        cleaned = cast_match.group(1).strip()
    match = _C_INT_RE.fullmatch(cleaned)
    if not match:
        return None
    raw = match.group(1)
    try:
        return int(raw, 16) if raw.lower().startswith("0x") else int(raw, 10)
    except ValueError:
        return None


def _unquote_c_string(token: str) -> str | None:
    """Parse one or more adjacent C string literals into a Python string."""
    cleaned = token.strip()
    if not cleaned:
        return None
    # Handle adjacent C string literal concatenation: "foo" "bar"
    parts = re.findall(r'"((?:[^"\\]|\\.)*)"', cleaned)
    if not parts:
        return None
    remainder = re.sub(r'"(?:[^"\\]|\\.)*"', "", cleaned).strip()
    if remainder:
        return None
    joined = "".join(parts)
    # Minimal C escape unescaping for symbol/DLL names
    return joined.replace('\\"', '"').replace("\\\\", "\\")


def _resolve_string_token(token: str, const_map: Mapping[str, str]) -> str | None:
    cleaned = token.strip()
    if not cleaned or cleaned in ("nullptr", "NULL", "0"):
        return None
    unquoted = _unquote_c_string(cleaned)
    if unquoted is not None:
        return unquoted
    if cleaned in const_map:
        return const_map[cleaned]
    return None


# ---------------------------------------------------------------------------
# 1. Markdown parser: docs/imports_required.md
# ---------------------------------------------------------------------------


def parse_imports_required(markdown_text: str) -> dict[str, list[dict[str, Any]]]:
    """Parse every `### `<dll>` (...)` import section and table row from markdown.

    Returns an ordered dict mapping exact DLL names (as written in the report) to
    lists of import entry dicts:
        {
            "kind": "name" | "ordinal",
            "symbol": str,          # e.g. "CreateFileW" or "#1"
            "ordinal": int | None,  # integer ordinal when kind == "ordinal"
            "hint": int | None,     # optional PE hint integer when present
        }
    """
    imports_by_dll: dict[str, list[dict[str, Any]]] = {}
    current_h2: str | None = None
    current_dll: str | None = None

    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        h2_match = _H2_RE.match(line)
        if h2_match:
            current_h2 = h2_match.group(1).strip().lower()
            current_dll = None
            continue

        # Only parse `### <dll>` sections before any H2 or inside Imports / Delay-Load Imports.
        if current_h2 is not None and "import" not in current_h2:
            continue

        dll_match = _DLL_SECTION_RE.match(line)
        if dll_match:
            current_dll = dll_match.group(1).strip()
            imports_by_dll.setdefault(current_dll, [])
            continue

        if current_dll is None or not line.startswith("|"):
            continue

        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 2:
            continue

        kind_raw = cells[0].strip().strip("`").lower()
        # Skip table headers (`Kind | Symbol | ...`) and separator rows (`--- | --- | ...`).
        if kind_raw not in ("name", "ordinal"):
            continue

        symbol_raw = cells[1].strip()
        if symbol_raw.startswith("`") and symbol_raw.endswith("`") and len(symbol_raw) >= 2:
            symbol = symbol_raw[1:-1]
        else:
            symbol = symbol_raw
        if not symbol:
            continue

        meta_cell = cells[2].strip() if len(cells) >= 3 else ""
        ordinal: int | None = None
        hint: int | None = None

        if kind_raw == "ordinal":
            ord_match = _ORDINAL_TEXT_RE.search(symbol) or _ORDINAL_TEXT_RE.search(meta_cell)
            if ord_match:
                ordinal = int(ord_match.group(1))
                if not symbol.startswith("#"):
                    symbol = f"#{ordinal}"
        else:
            hint_match = _HINT_HEX_RE.search(meta_cell)
            if hint_match:
                hint = int(hint_match.group(1), 16)

        imports_by_dll[current_dll].append(
            {
                "kind": kind_raw,
                "symbol": symbol,
                "ordinal": ordinal,
                "hint": hint,
            }
        )

    return imports_by_dll


# ---------------------------------------------------------------------------
# 2. Runtime shim scanner: runtime/{os,gpu,guest}/*.{h,cpp}
# ---------------------------------------------------------------------------


def _build_file_const_map(raw_text: str) -> dict[str, str]:
    """Extract `#define` and `constexpr const char` string constants from one file."""
    const_map: dict[str, str] = {}
    for macro_name, value in _DEFINE_STR_RE.findall(raw_text):
        const_map[macro_name] = value
        tag_match = _MOD_TAG_DEFINE_RE.match(macro_name)
        if tag_match and _is_dll_like(value):
            const_map[tag_match.group(1)] = value
    for ident, value in _CONST_STR_RE.findall(raw_text):
        const_map[ident] = value
    return const_map


def _record_export(
    exports_by_dll: dict[str, dict[str, str]],
    ordinals_by_dll: dict[str, dict[int, str]],
    dll_name: str,
    symbol: str | None,
    ordinal: int | None,
    source_rel: str,
) -> None:
    canon = canonical_dll_name(dll_name)
    if not canon:
        return
    if symbol:
        exports_by_dll.setdefault(canon, {}).setdefault(symbol, source_rel)
    if ordinal is not None and ordinal > 0:
        ordinals_by_dll.setdefault(canon, {}).setdefault(ordinal, source_rel)


def _extract_export_entries_from_block(
    block: str,
    const_map: Mapping[str, str],
) -> list[tuple[str | None, int | None]]:
    """Extract `(symbol_name, ordinal)` pairs from a `register_exports` or `ExportEntry[]` block."""
    found: list[tuple[str | None, int | None]] = []

    # 1. `export_entry("Name", &fn[, ordinal])`
    for match in _EXPORT_ENTRY_CALL_RE.finditer(block):
        inner, _ = _extract_balanced(block, match.end() - 1, "(", ")")
        if inner is None:
            continue
        args = _split_top_level_args(inner)
        if not args:
            continue
        sym = _resolve_string_token(args[0], const_map)
        ordinal: int | None = None
        if len(args) >= 3:
            ord_val = _parse_c_int(args[2])
            if ord_val is not None and ord_val > 0:
                ordinal = ord_val
        if sym or ordinal:
            found.append((sym, ordinal))

    # 2. Raw struct initializers `{"Name", ordinal, addr}` or `{nullptr, ordinal, addr}`
    for brace_match in re.finditer(r"\{", block):
        inner, _ = _extract_balanced(block, brace_match.start(), "{", "}")
        if inner is None:
            continue
        args = _split_top_level_args(inner)
        if len(args) < 2:
            continue
        sym = _resolve_string_token(args[0], const_map)
        ordinal = _parse_c_int(args[1]) if len(args) >= 2 else None
        if sym or (ordinal is not None and ordinal > 0):
            found.append((sym, ordinal if (ordinal is not None and ordinal > 0) else None))

    return found


def _scan_cpp_registrations(
    raw_text: str,
    source_rel: str,
    exports_by_dll: dict[str, dict[str, str]],
    ordinals_by_dll: dict[str, dict[int, str]],
    api_set_rules: dict[str, list[str]],
    module_aliases: dict[str, list[str]],
) -> None:
    """Scan one C/C++ file for all Win32ModuleRegistry export/alias/API-set registrations."""
    const_map = _build_file_const_map(raw_text)
    cleaned = _join_line_continuations(_strip_c_comments(raw_text))

    # Track file-local `const ExportEntry <name>[] = { ... };` tables (e.g. win32_loader.cpp).
    export_arrays: dict[str, list[tuple[str | None, int | None]]] = {}
    for arr_match in _EXPORT_ENTRY_ARRAY_RE.finditer(cleaned):
        arr_name = arr_match.group(1)
        block, _ = _extract_balanced(cleaned, arr_match.end() - 1, "{", "}")
        if block is not None:
            entries = _extract_export_entries_from_block(block, const_map)
            if entries:
                export_arrays[arr_name] = entries

    # 1. X-macro `E(MOD_TAG_OR_DLL, "Symbol", fn[, ordinal])` (shim_exports.cpp, ucrt_bridge.cpp).
    for match in _XMACRO_E_RE.finditer(cleaned):
        inner, _ = _extract_balanced(cleaned, match.end() - 1, "(", ")")
        if inner is None:
            continue
        args = _split_top_level_args(inner)
        if len(args) < 2:
            continue
        dll = _resolve_string_token(args[0], const_map)
        sym = _resolve_string_token(args[1], const_map)
        if not dll or not _is_dll_like(dll):
            continue
        ordinal = _parse_c_int(args[3]) if len(args) >= 4 else None
        if sym or (ordinal is not None and ordinal > 0):
            _record_export(exports_by_dll, ordinals_by_dll, dll, sym, ordinal, source_rel)

    # 2. `REGISTER_*` macros (`REGISTER_EXPORT`, `REGISTER_SHIM`, `REGISTER_ORDINAL`, etc.).
    for match in _REGISTER_MACRO_RE.finditer(cleaned):
        macro_name = match.group(1)
        inner, _ = _extract_balanced(cleaned, match.end() - 1, "(", ")")
        if inner is None:
            continue
        args = _split_top_level_args(inner)
        if len(args) < 2:
            continue
        dll = _resolve_string_token(args[0], const_map)
        if not dll or not _is_dll_like(dll):
            continue
        if "ORDINAL" in macro_name:
            ord_arg1 = _parse_c_int(args[1])
            if ord_arg1 is not None and ord_arg1 > 0:
                _record_export(exports_by_dll, ordinals_by_dll, dll, None, ord_arg1, source_rel)
            elif len(args) >= 3:
                sym = _resolve_string_token(args[1], const_map)
                ord_arg2 = _parse_c_int(args[2])
                _record_export(exports_by_dll, ordinals_by_dll, dll, sym, ord_arg2, source_rel)
        else:
            sym = _resolve_string_token(args[1], const_map)
            ordinal = _parse_c_int(args[2]) if len(args) >= 3 else None
            if sym or (ordinal is not None and ordinal > 0):
                _record_export(exports_by_dll, ordinals_by_dll, dll, sym, ordinal, source_rel)

    # 3. Direct registry calls (`register_export`, `register_symbol`, `register_ordinal`,
    #    `register_alias`, `register_api_set_rule`, `add_export_locked`, `add_api_set_rule_locked`).
    for match in _REGISTER_CALL_RE.finditer(cleaned):
        fn_name = match.group(1)
        inner, _ = _extract_balanced(cleaned, match.end() - 1, "(", ")")
        if inner is None:
            continue
        args = _split_top_level_args(inner)
        if len(args) < 2:
            continue

        if fn_name in ("register_api_set_rule", "add_api_set_rule_locked"):
            prefix = _resolve_string_token(args[0], const_map)
            host = _resolve_string_token(args[1], const_map)
            if prefix and host:
                norm_prefix = prefix.lower()
                norm_host = canonical_dll_name(host)
                hosts = api_set_rules.setdefault(norm_prefix, [])
                if norm_host and norm_host not in hosts:
                    hosts.insert(0, norm_host)
            continue

        if fn_name == "register_alias":
            alias_dll = _resolve_string_token(args[0], const_map)
            target_dll = _resolve_string_token(args[1], const_map)
            if alias_dll and target_dll:
                a_canon = canonical_dll_name(alias_dll)
                t_canon = canonical_dll_name(target_dll)
                if a_canon and t_canon and a_canon != t_canon:
                    targets = module_aliases.setdefault(a_canon, [])
                    if t_canon not in targets:
                        targets.append(t_canon)
            continue

        dll = _resolve_string_token(args[0], const_map)
        if not dll or not _is_dll_like(dll):
            continue

        if fn_name == "register_ordinal":
            ord_val = _parse_c_int(args[1])
            if ord_val is not None and ord_val > 0:
                _record_export(exports_by_dll, ordinals_by_dll, dll, None, ord_val, source_rel)
            elif len(args) >= 3:
                sym = _resolve_string_token(args[1], const_map)
                ord_val2 = _parse_c_int(args[2])
                _record_export(exports_by_dll, ordinals_by_dll, dll, sym, ord_val2, source_rel)
            continue

        # `register_export(dll, name, ordinal, addr)` / `add_export_locked` / `register_symbol`
        sym = _resolve_string_token(args[1], const_map)
        ordinal = _parse_c_int(args[2]) if len(args) >= 3 else None
        if sym or (ordinal is not None and ordinal > 0):
            _record_export(exports_by_dll, ordinals_by_dll, dll, sym, ordinal, source_rel)

    # 4. `register_exports("dll", { export_entry(...), ... })` (xbox_scarlett.cpp).
    for match in _REGISTER_EXPORTS_LIST_RE.finditer(cleaned):
        dll = _resolve_string_token(match.group(1), const_map)
        block, _ = _extract_balanced(cleaned, match.end() - 1, "{", "}")
        if not dll or not _is_dll_like(dll) or block is None:
            continue
        for sym, ordinal in _extract_export_entries_from_block(block, const_map):
            _record_export(exports_by_dll, ordinals_by_dll, dll, sym, ordinal, source_rel)

    # 5. `register_exports("dll", array_var, count)` or loop calling `add_export_locked("dll", ...)`.
    for match in _REGISTER_EXPORTS_VAR_RE.finditer(cleaned):
        dll = _resolve_string_token(match.group(1), const_map)
        arr_var = match.group(2)
        if dll and _is_dll_like(dll) and arr_var in export_arrays:
            for sym, ordinal in export_arrays[arr_var]:
                _record_export(exports_by_dll, ordinals_by_dll, dll, sym, ordinal, source_rel)

    for arr_var, entries in export_arrays.items():
        loop_match = re.search(
            rf"for\s*\([^)]*:\s*{re.escape(arr_var)}\s*\)\s*\{{[^}}]*?"
            r'(?:add_export_locked|register_export)\s*\(\s*("(?:[^"\\]|\\.)*"|[A-Za-z_][A-Za-z0-9_]*)\s*,',
            cleaned,
        )
        if loop_match:
            dll = _resolve_string_token(loop_match.group(1), const_map)
            if dll and _is_dll_like(dll):
                for sym, ordinal in entries:
                    _record_export(exports_by_dll, ordinals_by_dll, dll, sym, ordinal, source_rel)

    # 6. Struct-array export tables `{ kDllConst_or_"dll", "Symbol", ... }` (xbox_middleware.cpp).
    for match in _TABLE_STRUCT_ROW_RE.finditer(cleaned):
        dll_tok, sym_tok, rest_tok = match.groups()
        dll = _resolve_string_token(dll_tok, const_map)
        if not dll or not _is_dll_like(dll):
            continue
        sym = _resolve_string_token(sym_tok, const_map)
        rest_args = _split_top_level_args(rest_tok)
        ordinal: int | None = None
        if rest_args:
            cand_ord = _parse_c_int(rest_args[0])
            if cand_ord is not None and 0 < cand_ord <= 0xFFFF:
                ordinal = cand_ord
        if sym or ordinal:
            _record_export(exports_by_dll, ordinals_by_dll, dll, sym, ordinal, source_rel)

    # 7. Conditional alias mirroring (e.g. shim_exports.cpp `kXblAliasModule`).
    for match in _ALIAS_MIRROR_IF_RE.finditer(cleaned):
        source_dll = _resolve_string_token(match.group(1), const_map)
        alias_dll = _resolve_string_token(match.group(2), const_map)
        if source_dll and alias_dll and _is_dll_like(source_dll) and _is_dll_like(alias_dll):
            s_canon = canonical_dll_name(source_dll)
            a_canon = canonical_dll_name(alias_dll)
            if s_canon and a_canon and s_canon != a_canon:
                targets = module_aliases.setdefault(a_canon, [])
                if s_canon not in targets:
                    targets.append(s_canon)

    # 8. `ucrt_bridge.cpp` mirroring `api-ms-win-crt-*` into `ucrtbase.dll`.
    if re.search(r'register_export\s*\(\s*"ucrtbase\.dll"', cleaned, re.IGNORECASE):
        for dll_key, syms in list(exports_by_dll.items()):
            if dll_key.startswith("api-ms-win-crt-"):
                target_map = exports_by_dll.setdefault("ucrtbase.dll", {})
                for sym, src in syms.items():
                    target_map.setdefault(sym, src)


def _scan_header_extern_c(
    raw_text: str,
    source_rel: str,
    header_exports: dict[str, dict[str, Any]],
) -> None:
    """Scan a C/C++ header for `extern "C"` `pg6_*` shim declarations."""
    cleaned = _strip_c_comments(raw_text)
    found_symbols: set[str] = set()

    for block_match in _EXTERN_C_BLOCK_RE.finditer(cleaned):
        block, _ = _extract_balanced(cleaned, block_match.end() - 1, "{", "}")
        if block is None:
            continue
        for sym in _PG6_HEADER_FN_RE.findall(block):
            found_symbols.add(sym)

    for sym in _EXTERN_C_SINGLE_RE.findall(cleaned):
        found_symbols.add(sym)

    for sym in sorted(found_symbols):
        prefix_match = _PG6_PREFIX_RE.match(sym)
        if not prefix_match:
            continue
        prefix, tail = prefix_match.groups()
        # Skip registration/table/test-reset helpers that are not Win32/Xbox APIs.
        if tail.endswith(("_register_all", "_export_table", "_reset_runtime_state")):
            continue
        keys = [normalize_symbol_key(tail)]
        if prefix == "pg6_d3d12_" and not keys[0].startswith("d3d12"):
            keys.append(f"d3d12{keys[0]}")
        header_exports[sym] = {
            "symbol": sym,
            "prefix": prefix,
            "normalized_keys": keys,
            "source": source_rel,
        }


def scan_runtime_shims(repo_root: Path | str) -> dict[str, Any]:
    """Scan `runtime/{os,gpu,guest}/*.{h,cpp}` for all registered & exported shims.

    Returns a sanitized dict (repo-relative paths only) containing:
    - `scanned_files`: sorted list of repo-relative paths scanned
    - `registered_exports`: `{canonical_dll: {symbol_name: source_rel}}`
    - `registered_ordinals`: `{canonical_dll: {ordinal_int: source_rel}}`
    - `api_set_rules`: `{prefix_lower: [host_dll_canonical, ...]}`
    - `module_aliases`: `{alias_dll_canonical: [target_dll_canonical, ...]}`
    - `header_exports`: `{pg6_symbol: {"symbol", "prefix", "normalized_keys", "source"}}`
    """
    root = Path(repo_root)
    scanned_files: list[str] = []
    exports_by_dll: dict[str, dict[str, str]] = {}
    ordinals_by_dll: dict[str, dict[int, str]] = {}
    api_set_rules: dict[str, list[str]] = {
        prefix: list(hosts) for prefix, hosts in DEFAULT_API_SET_RULES.items()
    }
    module_aliases: dict[str, list[str]] = {}
    header_exports: dict[str, dict[str, Any]] = {}

    for rel_dir in RUNTIME_SCAN_DIRS:
        scan_dir = root / rel_dir
        if not scan_dir.is_dir():
            continue
        files = sorted(
            p for p in scan_dir.iterdir() if p.is_file() and p.suffix.lower() in (".h", ".cpp")
        )
        for path in files:
            source_rel = f"{rel_dir}/{path.name}"
            try:
                raw_text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            scanned_files.append(source_rel)
            _scan_cpp_registrations(
                raw_text,
                source_rel,
                exports_by_dll,
                ordinals_by_dll,
                api_set_rules,
                module_aliases,
            )
            if path.suffix.lower() == ".h":
                _scan_header_extern_c(raw_text, source_rel, header_exports)

    return {
        "scanned_files": scanned_files,
        "registered_exports": {
            dll: dict(sorted(syms.items())) for dll, syms in sorted(exports_by_dll.items())
        },
        "registered_ordinals": {
            dll: dict(sorted(ords.items())) for dll, ords in sorted(ordinals_by_dll.items())
        },
        "api_set_rules": dict(sorted(api_set_rules.items())),
        "module_aliases": dict(sorted(module_aliases.items())),
        "header_exports": dict(sorted(header_exports.items())),
    }


# ---------------------------------------------------------------------------
# 3. Coverage auditing & classification
# ---------------------------------------------------------------------------


def _candidate_host_dlls(
    canonical_dll: str,
    api_set_rules: Mapping[str, Sequence[str]],
    module_aliases: Mapping[str, Sequence[str]],
) -> list[str]:
    """Return ordered list of canonical host DLLs to search for an imported DLL."""
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(dll: str) -> None:
        if dll and dll not in seen:
            seen.add(dll)
            candidates.append(dll)

    _add(canonical_dll)

    # Explicit module aliases (BFS with cycle guard).
    queue = [canonical_dll]
    while queue:
        cur = queue.pop(0)
        for target in module_aliases.get(cur, ()):
            if target not in seen:
                _add(target)
                queue.append(target)

    # Win32 API-set contract forwarding (longest matching prefix wins, matching win32_loader.cpp).
    if canonical_dll.startswith(("api-", "ext-")):
        matching_prefixes = [p for p in api_set_rules if canonical_dll.startswith(p)]
        if matching_prefixes:
            best_len = max(len(p) for p in matching_prefixes)
            for prefix in sorted(p for p in matching_prefixes if len(p) == best_len):
                for host in api_set_rules[prefix]:
                    _add(host)
                    for eq_host in _EQUIVALENT_HOST_DLLS.get(host, ()):
                        _add(eq_host)

    # Equivalent host DLLs for direct imports too (e.g. kernelbase.dll <-> kernel32.dll).
    for existing in list(candidates):
        for eq_host in _EQUIVALENT_HOST_DLLS.get(existing, ()):
            _add(eq_host)

    return candidates


def _prefix_matches_dll(prefix: str, candidate_dlls: Sequence[str]) -> bool:
    exact_set = _HEADER_PREFIX_EXACT_DLLS.get(prefix, frozenset())
    dll_prefixes = _HEADER_PREFIX_DLL_PREFIXES.get(prefix, ())
    for dll in candidate_dlls:
        if dll in exact_set:
            return True
        if any(dll.startswith(p) for p in dll_prefixes):
            return True
    return False


def _build_header_key_index(
    header_exports: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Index header declarations as `{prefix: {normalized_key: entry}}`."""
    index: dict[str, dict[str, dict[str, Any]]] = {}
    for entry in header_exports.values():
        prefix = str(entry["prefix"])
        prefix_map = index.setdefault(prefix, {})
        for key in entry["normalized_keys"]:
            prefix_map.setdefault(str(key), dict(entry))
    return index


def _match_import_against_shims(
    canonical_dll: str,
    entry: Mapping[str, Any],
    shims: Mapping[str, Any],
    header_index: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any] | None:
    """Return match metadata dict if an import is covered by `runtime/`, else `None`."""
    candidates = _candidate_host_dlls(
        canonical_dll,
        shims["api_set_rules"],
        shims["module_aliases"],
    )
    registered_exports = shims["registered_exports"]
    registered_ordinals = shims["registered_ordinals"]

    kind = entry["kind"]
    symbol = str(entry["symbol"])
    ordinal = entry.get("ordinal")

    # 1. Exact registry match (direct DLL, alias, or API-set forwarded host DLL).
    for host_dll in candidates:
        if kind == "ordinal":
            if ordinal is not None and ordinal in registered_ordinals.get(host_dll, {}):
                return {
                    "via": "registry_ordinal",
                    "host_dll": host_dll,
                    "source": registered_ordinals[host_dll][ordinal],
                }
        else:
            if symbol in registered_exports.get(host_dll, {}):
                return {
                    "via": "registry",
                    "host_dll": host_dll,
                    "source": registered_exports[host_dll][symbol],
                }

    # 2. Fallback: `extern "C"` header declaration matched via normalized symbol key + DLL family.
    if kind == "name":
        sym_key = normalize_symbol_key(symbol)
        for prefix, key_map in header_index.items():
            if sym_key in key_map and _prefix_matches_dll(prefix, candidates):
                hdr_entry = key_map[sym_key]
                return {
                    "via": "header_export",
                    "host_dll": candidates[-1] if len(candidates) > 1 else canonical_dll,
                    "shim_symbol": hdr_entry["symbol"],
                    "source": hdr_entry["source"],
                }

    return None


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 2)


def audit_import_coverage(
    repo_root: Path | str | None = None,
    guest_recompiled_dlls: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Audit real-binary import coverage across `docs/imports_required.md` and `runtime/`."""
    root = Path(repo_root) if repo_root is not None else DEFAULT_REPO_ROOT
    guest_dll_set: frozenset[str] = (
        DEFAULT_GUEST_RECOMPILED_DLLS
        if guest_recompiled_dlls is None
        else frozenset(canonical_dll_name(d) for d in guest_recompiled_dlls if d.strip())
    )

    imports_path = root / IMPORTS_REQUIRED_REL
    errors: list[str] = []
    if not imports_path.is_file():
        errors.append(f"missing required file: {IMPORTS_REQUIRED_REL}")
        imports_by_dll: dict[str, list[dict[str, Any]]] = {}
    else:
        try:
            markdown_text = imports_path.read_text(encoding="utf-8")
            imports_by_dll = parse_imports_required(markdown_text)
        except OSError as exc:
            errors.append(sanitize_message(f"failed to read {IMPORTS_REQUIRED_REL}: {exc}"))
            imports_by_dll = {}

    shims = scan_runtime_shims(root)
    header_index = _build_header_key_index(shims["header_exports"])

    dll_reports: list[dict[str, Any]] = []
    total_symbols = 0
    total_shimmed = 0
    total_guest_recompiled = 0
    total_missing = 0

    for dll_name, entries in imports_by_dll.items():
        canon_dll = canonical_dll_name(dll_name)
        is_guest_recompiled = canon_dll in guest_dll_set

        shimmed_syms: list[str] = []
        guest_syms: list[str] = []
        missing_syms: list[str] = []
        via_counts: dict[str, int] = {}

        for entry in entries:
            symbol = str(entry["symbol"])
            match = _match_import_against_shims(canon_dll, entry, shims, header_index)
            if match is not None:
                shimmed_syms.append(symbol)
                via_key = str(match["via"])
                via_counts[via_key] = via_counts.get(via_key, 0) + 1
            elif is_guest_recompiled:
                guest_syms.append(symbol)
            else:
                missing_syms.append(symbol)

        dll_total = len(entries)
        dll_shimmed = len(shimmed_syms)
        dll_guest = len(guest_syms)
        dll_missing = len(missing_syms)

        total_symbols += dll_total
        total_shimmed += dll_shimmed
        total_guest_recompiled += dll_guest
        total_missing += dll_missing

        if dll_missing == 0 and dll_total > 0:
            status = "guest_recompiled" if dll_guest == dll_total else "complete"
        elif dll_shimmed > 0 or dll_guest > 0:
            status = "partial"
        else:
            status = "missing"

        dll_reports.append(
            {
                "dll": dll_name,
                "canonical_dll": canon_dll,
                "status": status,
                "total": dll_total,
                "shimmed": dll_shimmed,
                "guest_recompiled": dll_guest,
                "missing": dll_missing,
                "shimmed_pct": _pct(dll_shimmed, dll_total),
                "effective_boot_pct": _pct(dll_shimmed + dll_guest, dll_total),
                "shimmed_symbols": shimmed_syms,
                "guest_recompiled_symbols": guest_syms,
                "missing_symbols": missing_syms,
                "resolution_breakdown": dict(sorted(via_counts.items())),
            }
        )

    if not errors and total_symbols == 0:
        errors.append(f"{IMPORTS_REQUIRED_REL}: no import sections/symbols found")

    registered_export_count = sum(len(v) for v in shims["registered_exports"].values())
    registered_ordinal_count = sum(len(v) for v in shims["registered_ordinals"].values())

    return {
        "ok": not errors,
        "source": IMPORTS_REQUIRED_REL,
        "guest_recompiled_dlls": sorted(guest_dll_set),
        "totals": {
            "dll_count": len(dll_reports),
            "complete_dlls": sum(1 for d in dll_reports if d["missing"] == 0),
            "partial_dlls": sum(
                1 for d in dll_reports if d["missing"] > 0 and (d["shimmed"] > 0 or d["guest_recompiled"] > 0)
            ),
            "missing_dlls": sum(
                1 for d in dll_reports if d["missing"] == d["total"] and d["total"] > 0
            ),
            "total": total_symbols,
            "shimmed": total_shimmed,
            "guest_recompiled": total_guest_recompiled,
            "missing": total_missing,
            "shimmed_pct": _pct(total_shimmed, total_symbols),
            "guest_recompiled_pct": _pct(total_guest_recompiled, total_symbols),
            "effective_boot_pct": _pct(total_shimmed + total_guest_recompiled, total_symbols),
            "missing_pct": _pct(total_missing, total_symbols),
        },
        "runtime_scan": {
            "scanned_file_count": len(shims["scanned_files"]),
            "registered_module_count": len(shims["registered_exports"]),
            "registered_export_count": registered_export_count,
            "registered_ordinal_count": registered_ordinal_count,
            "header_export_count": len(shims["header_exports"]),
            "api_set_rule_count": len(shims["api_set_rules"]),
        },
        "dlls": dll_reports,
        "errors": errors,
    }


def filter_report_missing_only(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of an audit report filtered to DLLs with missing symbols."""
    filtered_dlls = [
        {
            "dll": d["dll"],
            "canonical_dll": d["canonical_dll"],
            "status": d["status"],
            "total": d["total"],
            "shimmed": d["shimmed"],
            "guest_recompiled": d["guest_recompiled"],
            "missing": d["missing"],
            "shimmed_pct": d["shimmed_pct"],
            "effective_boot_pct": d["effective_boot_pct"],
            "missing_symbols": list(d["missing_symbols"]),
        }
        for d in report["dlls"]
        if d["missing"] > 0
    ]
    out = dict(report)
    out["missing_only"] = True
    out["dlls"] = filtered_dlls
    return out


# ---------------------------------------------------------------------------
# 3b. Import bridge binding audit (O17)
# ---------------------------------------------------------------------------
#
# Registry coverage (above) says whether the runtime *has* something registered under an import's
# name. What boot needs is how the recompiled image actually reaches each import: the import
# bridge (`playgta6-recomp stubs`, recomp/src/stubs.rs) decides that and writes it into its
# manifest, after the shim audit (runtime/os/shim_audit.h). This section reads such a manifest:
#   - `pg6_import_manifest.json` of `playgta6-recomp stubs <pe> | --import-list <file>`
#     (`"format": "playgta6-import-bridge"`), or
#   - `program-manifest.json` of `playgta6-recomp build-program` (`"schema":
#     "playgta6-recomp-program"`), whose imports already carry guest/runtime bindings.
# Every import is classified as `guest` (an export of a guest module), `runtime` (R11 exception
# thunk bound by the program loader), `gthunk` (a pg6_gthunk_ guest-state thunk), `host` (a
# marshaled, audited host shim) or `unresolved`; an unresolved import must carry the bridge's
# `recorded` reason (an audit verdict, an unmarshalable signature or a PG6_IMPORT_GAPS row).

BRIDGE_FORMAT = "playgta6-import-bridge"
PROGRAM_SCHEMA = "playgta6-recomp-program"
BINDING_CLASSES: tuple[str, ...] = ("guest", "runtime", "gthunk", "host", "unresolved")

# DLLs a program build loads as guest modules (the redistributable C++ runtime shipped with the
# game, R12): imports of a single-module bridge manifest from these count as `guest`.
DEFAULT_BRIDGE_GUEST_DLLS: frozenset[str] = frozenset(
    {"msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"}
)

# Exception APIs the runtime program loader binds to R11's guest-state thunks whatever the
# bridge generated (recomp/src/pipeline/program.rs RUNTIME_EXCEPTION_APIS; kept identical by
# tests/test_import_coverage.py).
RUNTIME_EXCEPTION_APIS: frozenset[tuple[str, str]] = frozenset(
    {
        ("kernel32.dll", name)
        for name in (
            "RaiseException",
            "RtlCaptureContext",
            "RtlLookupFunctionEntry",
            "RtlVirtualUnwind",
            "RtlUnwindEx",
            "RtlUnwind",
            "RtlRestoreContext",
            "RtlPcToFileHeader",
            "RtlAddFunctionTable",
            "RtlDeleteFunctionTable",
            "SetUnhandledExceptionFilter",
            "UnhandledExceptionFilter",
        )
    }
    | {
        ("ntdll.dll", name)
        for name in (
            "RtlRaiseException",
            "RtlCaptureContext",
            "RtlLookupFunctionEntry",
            "RtlVirtualUnwind",
            "RtlUnwindEx",
            "RtlUnwind",
            "RtlRestoreContext",
            "RtlPcToFileHeader",
            "RtlAddFunctionTable",
            "RtlDeleteFunctionTable",
        )
    }
)


def _runtime_exception_module(dll: str) -> str | None:
    """`kernel32.dll`/`ntdll.dll` when imports of `dll` reach the runtime exception thunks."""
    canon = canonical_dll_name(dll)
    if canon in ("kernel32.dll", "kernelbase.dll") or canon.startswith("api-ms-win-core-"):
        return "kernel32.dll"
    if canon == "ntdll.dll":
        return "ntdll.dll"
    return None


def classify_bridge_import(
    entry: Mapping[str, Any], guest_dlls: frozenset[str] = DEFAULT_BRIDGE_GUEST_DLLS
) -> dict[str, Any]:
    """One manifest import -> {dll, symbol, class, shim, reason, recorded}."""
    dll = str(entry.get("dll", ""))
    name = entry.get("name")
    ordinal = entry.get("ordinal")
    symbol = str(name) if name is not None else f"#{ordinal}"
    shim = entry.get("shim")
    reason = entry.get("reason")
    recorded = entry.get("recorded")
    binding = entry.get("binding")  # program manifest
    status = entry.get("status")  # bridge manifest
    if binding is not None:
        cls = {"guest": "guest", "runtime": "runtime", "host": "host"}.get(str(binding), "unresolved")
    elif status == "guest" or canonical_dll_name(dll) in guest_dlls:
        cls = "guest"
    elif (
        name is not None
        and (module := _runtime_exception_module(dll)) is not None
        and (module, str(name)) in RUNTIME_EXCEPTION_APIS
    ):
        cls = "runtime"
    elif status == "bound":
        cls = "host"
    else:
        cls = "unresolved"
    if cls == "host" and isinstance(shim, str) and shim.startswith("pg6_gthunk_"):
        cls = "gthunk"
    if cls != "unresolved":
        reason = None
        recorded = None
    return {
        "importer": entry.get("importer"),
        "dll": dll,
        "symbol": symbol,
        "class": cls,
        "shim": shim if cls in ("host", "gthunk") else None,
        "reason": reason,
        "recorded": recorded,
    }


def audit_bridge_manifest(
    manifest: Mapping[str, Any], guest_dlls: Iterable[str] | None = None
) -> dict[str, Any]:
    """Binding report for an import bridge or program manifest (see the section comment)."""
    guest_set = (
        DEFAULT_BRIDGE_GUEST_DLLS
        if guest_dlls is None
        else frozenset(canonical_dll_name(d) for d in guest_dlls if d.strip())
    )
    errors: list[str] = []
    kind = (
        "program"
        if manifest.get("schema") == PROGRAM_SCHEMA
        else "bridge"
        if manifest.get("format") == BRIDGE_FORMAT
        else None
    )
    if kind is None:
        errors.append("not an import bridge or program manifest")
    imports = manifest.get("imports")
    if not isinstance(imports, list):
        errors.append("manifest has no imports list")
        imports = []
    rows = [classify_bridge_import(e, guest_set) for e in imports if isinstance(e, Mapping)]
    totals = {cls: 0 for cls in BINDING_CLASSES}
    per_importer: dict[str, dict[str, int]] = {}
    per_dll: dict[str, dict[str, int]] = {}
    unresolved: list[dict[str, Any]] = []
    unrecorded: list[str] = []
    for row in rows:
        totals[row["class"]] += 1
        importer = str(row["importer"] or "")
        per_importer.setdefault(importer, {cls: 0 for cls in BINDING_CLASSES})[row["class"]] += 1
        per_dll.setdefault(row["dll"], {cls: 0 for cls in BINDING_CLASSES})[row["class"]] += 1
        if row["class"] == "unresolved":
            label = f"{row['dll']}!{row['symbol']}"
            if importer:
                label = f"{importer}: {label}"
            unresolved.append(
                {"import": label, "reason": row["reason"], "recorded": row["recorded"]}
            )
            if not row["recorded"]:
                unrecorded.append(label)
    totals["total"] = len(rows)
    return {
        "ok": not errors and not unrecorded,
        "kind": kind,
        "guest_dlls": sorted(guest_set) if kind == "bridge" else [],
        "totals": totals,
        "importers": dict(sorted(per_importer.items())),
        "dlls": dict(sorted(per_dll.items(), key=lambda kv: kv[0].lower())),
        "unresolved": unresolved,
        "unrecorded": unrecorded,
        "errors": errors,
    }


def format_bridge_report(report: Mapping[str, Any]) -> str:
    lines = ["playgta6web Import Bridge Binding Audit", "=" * 88]
    for err in report["errors"]:
        lines.append(f"ERROR: {err}")
    header = f"{'DLL':<52}" + "".join(f"{cls:>11}" for cls in BINDING_CLASSES)
    lines.append(header)
    lines.append("-" * len(header))
    for dll, counts in report["dlls"].items():
        lines.append(f"{dll:<52}" + "".join(f"{counts[cls]:>11}" for cls in BINDING_CLASSES))
    lines.append("=" * 88)
    totals = report["totals"]
    lines.append(
        f"Totals ({totals['total']} imports): "
        + ", ".join(f"{totals[cls]} {cls}" for cls in BINDING_CLASSES)
    )
    if report["unresolved"]:
        lines.append("Unresolved:")
        for u in report["unresolved"]:
            lines.append(f"  {u['import']}: {u['recorded'] or 'NO RECORDED REASON (' + str(u['reason']) + ')'}")
    if report["unrecorded"]:
        lines.append(f"{len(report['unrecorded'])} unresolved import(s) without a recorded reason")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Human-readable formatter & CLI
# ---------------------------------------------------------------------------


def format_human_report(report: Mapping[str, Any], *, missing_only: bool = False) -> str:
    """Format a crisp human-readable import coverage table and summary."""
    totals = report["totals"]
    lines: list[str] = []
    title = "playgta6web Real-Binary Import Coverage Audit"
    if missing_only:
        title += " (missing only)"
    lines.append(title)
    lines.append("=" * 88)

    if report.get("errors"):
        for err in report["errors"]:
            lines.append(f"ERROR: {err}")
        lines.append("=" * 88)

    dlls = [d for d in report["dlls"] if (d["missing"] > 0 if missing_only else True)]
    name_width = max((len(str(d["dll"])) for d in dlls), default=12)
    name_width = max(name_width, len("DLL"))

    header = (
        f"{'DLL':<{name_width}}  {'Total':>5}  {'Shim':>5}  {'Guest':>5}  "
        f"{'Miss':>5}  {'Boot %':>7}  Status"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for d in dlls:
        lines.append(
            f"{d['dll']:<{name_width}}  {d['total']:>5}  {d['shimmed']:>5}  "
            f"{d['guest_recompiled']:>5}  {d['missing']:>5}  "
            f"{d['effective_boot_pct']:>6.1f}%  {d['status']}"
        )
        if missing_only and d["missing_symbols"]:
            syms_preview = ", ".join(d["missing_symbols"])
            lines.append(f"  missing: {syms_preview}")

    lines.append("=" * 88)
    lines.append(
        f"Totals ({totals['dll_count']} DLLs): "
        f"{totals['shimmed']}/{totals['total']} shimmed ({totals['shimmed_pct']:.2f}%), "
        f"{totals['guest_recompiled']}/{totals['total']} guest-recompiled "
        f"({totals['guest_recompiled_pct']:.2f}%), "
        f"{totals['missing']}/{totals['total']} missing ({totals['missing_pct']:.2f}%)"
    )
    lines.append(
        f"Effective Boot Coverage: {totals['shimmed'] + totals['guest_recompiled']}/"
        f"{totals['total']} ({totals['effective_boot_pct']:.2f}%) | "
        f"DLLs: {totals['complete_dlls']} complete, {totals['partial_dlls']} partial, "
        f"{totals['missing_dlls']} missing"
    )
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit real-binary PE64 import coverage (docs/imports_required.md) against "
            "registered and exported shims in runtime/."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Path to the playgta6web repository root (default: auto-detected).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit strictly sanitized machine-readable JSON (never includes host paths).",
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Show only DLLs with remaining missing symbols.",
    )
    parser.add_argument(
        "--min-pct",
        type=float,
        default=0.0,
        help="Minimum required effective boot coverage percentage (exit 1 if below; default: 0).",
    )
    parser.add_argument(
        "--bridge-manifest",
        type=Path,
        default=None,
        help=(
            "Audit the bindings in an import bridge manifest (pg6_import_manifest.json of "
            "`playgta6-recomp stubs <pe>|--import-list docs/imports_required.md`) or a "
            "program-manifest.json instead of registry coverage; exit 1 if an import is "
            "unresolved without a recorded reason."
        ),
    )
    parser.add_argument(
        "--guest-dll",
        action="append",
        default=None,
        help=(
            "With --bridge-manifest on a single-module manifest: a DLL loaded as a guest module "
            "(repeatable; default: msvcp140.dll, vcruntime140.dll, vcruntime140_1.dll)."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.bridge_manifest is not None:
        try:
            manifest = json.loads(args.bridge_manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(sanitize_message(f"failed to read bridge manifest: {exc}"), file=sys.stderr)
            return 1
        bridge_report = audit_bridge_manifest(manifest, args.guest_dll)
        if args.json:
            print(json.dumps(bridge_report, indent=2))
        else:
            print(format_bridge_report(bridge_report))
        return 0 if bridge_report["ok"] else 1

    report = audit_import_coverage(args.repo_root)
    output_report = filter_report_missing_only(report) if args.missing_only else report

    if args.json:
        print(json.dumps(output_report, indent=2))
    else:
        print(format_human_report(report, missing_only=args.missing_only))

    if not report["ok"]:
        return 1
    return 0 if report["totals"]["effective_boot_pct"] >= args.min_pct else 1


if __name__ == "__main__":
    sys.exit(main())
