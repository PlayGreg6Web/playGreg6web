#!/usr/bin/env python3
"""Recompilation coverage and boot-path bottleneck profiler (Track 1 / R15).

Parses exact JSON produced by `playgta6-recomp build` (`build-manifest.json`,
schema `playgta6-recomp-build`) and `playgta6-recomp discover --json`
(`render_summary_json` in `recomp/src/summary.rs`). Also accepts a build
output directory (module dir or `--out` root containing `<module>/`).

Computes:
- Function / block / instruction lift coverage and shard size distribution.
- Top-N unlifted x86-64 mnemonics and lift-failure categories.
- Entry-point + TLS-callback boot-path BFS, including shortest-path traces
  to `D3D12XboxCreateDevice` / `PresentX` when import/call metadata is present.
- Jump-table resolution summary.
- Prioritized recommendations to unblock splash-screen boot.

Security invariant: stdout, JSON, and Markdown never include host absolute
paths, usernames, or raw binary bytes. Pure Python 3.12 stdlib.

Usage:
    python tools/audit_recomp_build.py <recomp-build-dir-or-json> [--top N] [--boot-depth D] [--markdown] [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_BUILD = "playgta6-recomp-build"
SCHEMA_AUDIT = "playgta6-recomp-audit"
SCHEMA_AUDIT_VERSION = 1
MANIFEST_FILE = "build-manifest.json"
REPORT_FILE = "report.txt"
DEFAULT_TOP = 15
BOOT_TARGETS = ("D3D12XboxCreateDevice", "PresentX")

DISCOVER_JSON_NAMES = (
    "discover.json",
    "discovery.json",
    "cfg.json",
    "summary.json",
    "lift-summary.json",
    "coverage.json",
)

_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"'<>|,;()\[\]]+")
_UNC_PATH_RE = re.compile(r"\\\\[^\s\"'<>|,;()\[\]]+")
_POSIX_PATH_RE = re.compile(
    r"/(?:Users|home|tmp|var|opt|root|mnt|private|etc|usr|Library|Volumes)/[^\s\"'<>|,;()\[\]]*"
)
_REPORT_MNEMONIC_RE = re.compile(r"^\s{2,}([A-Za-z][A-Za-z0-9./]+)\s+(\d+)\s*$")


class AuditError(ValueError):
    """Raised when inputs cannot be loaded or classified."""


# ---------------------------------------------------------------------------
# Sanitization
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


def sanitize_text(raw: str | None) -> str:
    """Strip host absolute paths (both `\\` and `/`) and active usernames."""
    if raw is None:
        return ""
    text = raw.replace("\\", "/")
    sanitized = _WINDOWS_PATH_RE.sub("<path>", text)
    sanitized = _UNC_PATH_RE.sub("<path>", sanitized)
    sanitized = _POSIX_PATH_RE.sub("<path>", sanitized)
    # Catch remaining drive-letter paths after slash normalization.
    sanitized = re.sub(r"[A-Za-z]:/(?:[^\s\"'<>|,;()\[\]]+)", "<path>", sanitized)
    for username in _current_usernames():
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(username)}(?![A-Za-z0-9_])",
            re.IGNORECASE,
        )
        sanitized = pattern.sub("<user>", sanitized)
    return sanitized


def basename_label(source: str | Path | None) -> str:
    """Basename-only label; splits on both `/` and `\\` regardless of host OS."""
    if source is None:
        return "<unknown>"
    raw = str(source).strip()
    if not raw:
        return "<unknown>"
    stripped = raw.rstrip("/\\")
    if not stripped:
        return "<unknown>"
    parts = re.split(r"[/\\]+", stripped)
    base = parts[-1]
    if not base or (len(base) == 2 and base[1] == ":"):
        return "<unknown>"
    return sanitize_text(base)


def sanitize_value(value: Any) -> Any:
    """Recursively sanitize strings; drop raw bytes from reports."""
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, bytearray):
        return f"<bytes:{len(value)}>"
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, Mapping):
        return {sanitize_text(str(k)): sanitize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_value(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_value(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Parsing helpers matching recomp JSON field names
# ---------------------------------------------------------------------------


def parse_va(value: Any) -> int | None:
    """Parse a guest VA from an `0x...` string or integer (summary.rs / pipeline.rs)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text, 16) if text.lower().startswith("0x") else int(text, 0)
        except ValueError:
            try:
                return int(text, 16)
            except ValueError:
                return None
    return None


def format_va(value: Any) -> str:
    parsed = parse_va(value)
    if parsed is not None:
        return f"0x{parsed:x}"
    if isinstance(value, str) and value.strip():
        return sanitize_text(value.strip())
    return sanitize_text(str(value))


def as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return int(text, 0)
        except ValueError:
            parsed = parse_va(text)
            return parsed if parsed is not None else default
    return default


def as_float_pct(part: int, whole: int) -> float:
    if whole <= 0:
        return 0.0
    return round(part * 100.0 / whole, 2)


def median(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _looks_like_bytes(value: Any) -> bool:
    return isinstance(value, (bytes, bytearray))


def is_build_manifest(doc: Mapping[str, Any]) -> bool:
    if doc.get("schema") == SCHEMA_BUILD:
        return True
    return isinstance(doc.get("lift"), Mapping) and isinstance(doc.get("discovery"), Mapping)


def is_discover_json(doc: Mapping[str, Any]) -> bool:
    if is_build_manifest(doc):
        return False
    functions = doc.get("functions")
    return (
        "function_count" in doc
        and "entry_points" in doc
        and isinstance(functions, list)
    )


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def _read_json_file(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AuditError(f"failed to read {basename_label(path)}: {exc.strerror or type(exc).__name__}") from exc
    except UnicodeDecodeError as exc:
        raise AuditError(f"failed to decode {basename_label(path)} as UTF-8 JSON") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise AuditError(f"invalid JSON in {basename_label(path)}: {exc.msg}") from exc


def _iter_json_files_in_dir(directory: Path) -> list[Path]:
    found: list[Path] = []
    direct = directory / MANIFEST_FILE
    if direct.is_file():
        found.append(direct)
    else:
        try:
            for child in sorted(directory.iterdir()):
                if child.is_dir() and (child / MANIFEST_FILE).is_file():
                    found.append(child / MANIFEST_FILE)
        except OSError:
            pass
    for name in DISCOVER_JSON_NAMES:
        candidate = directory / name
        if candidate.is_file():
            found.append(candidate)
        try:
            for child in sorted(directory.iterdir()):
                nested = child / name
                if child.is_dir() and nested.is_file() and nested not in found:
                    found.append(nested)
        except OSError:
            pass
    # Deduplicate while preserving order.
    unique: list[Path] = []
    seen: set[str] = set()
    for path in found:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def load_inputs(path: str | Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str, str | None]:
    """Load a build-manifest and/or discover JSON from a file or directory.

    Returns `(manifest, discover, source_label, report_text)`.
    """
    target = Path(path)
    manifest: dict[str, Any] | None = None
    discover: dict[str, Any] | None = None
    report_text: str | None = None
    labels: list[str] = []

    files: list[Path]
    if target.is_dir():
        files = _iter_json_files_in_dir(target)
        report_path = target / REPORT_FILE
        if report_path.is_file():
            try:
                report_text = report_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                report_text = None
        if not files and report_text is None:
            try:
                for child in sorted(target.iterdir()):
                    nested = child / REPORT_FILE
                    if child.is_dir() and nested.is_file():
                        report_text = nested.read_text(encoding="utf-8", errors="replace")
                        files = _iter_json_files_in_dir(child)
                        break
            except OSError:
                pass
        labels.append(basename_label(target))
    elif target.is_file():
        files = [target]
        labels.append(basename_label(target))
        sibling_dir = target.parent
        if target.name != MANIFEST_FILE:
            sibling_manifest = sibling_dir / MANIFEST_FILE
            if sibling_manifest.is_file() and sibling_manifest != target:
                files.append(sibling_manifest)
        for name in DISCOVER_JSON_NAMES:
            sibling = sibling_dir / name
            if sibling.is_file() and sibling not in files:
                files.append(sibling)
        report_path = sibling_dir / REPORT_FILE
        if report_path.is_file():
            try:
                report_text = report_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                report_text = None
    else:
        raise AuditError(f"input not found: {basename_label(path)}")

    for file_path in files:
        doc = _read_json_file(file_path)
        if _looks_like_bytes(doc):
            continue
        if not isinstance(doc, dict):
            raise AuditError(f"{basename_label(file_path)} is not a JSON object")
        if is_build_manifest(doc):
            manifest = doc
            labels.append(basename_label(file_path))
        elif is_discover_json(doc):
            discover = doc
            labels.append(basename_label(file_path))
        elif isinstance(doc.get("lift"), Mapping) and not manifest:
            manifest = doc
            labels.append(basename_label(file_path))
        elif isinstance(doc.get("functions"), list) and not discover:
            discover = doc
            labels.append(basename_label(file_path))

    if manifest is None and discover is None:
        raise AuditError(
            "no playgta6-recomp build-manifest.json or discover JSON found"
        )
    source = labels[0] if len(labels) == 1 else "+".join(dict.fromkeys(labels))
    return manifest, discover, sanitize_text(source), report_text


def parse_report_mnemonics(report_text: str | None) -> dict[str, int]:
    """Fallback: parse `report.txt` top unsupported mnemonics (pipeline.rs)."""
    counts: dict[str, int] = {}
    if not report_text:
        return counts
    in_section = False
    for line in report_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("top unsupported mnemonics"):
            in_section = True
            continue
        if in_section:
            if not stripped:
                break
            match = _REPORT_MNEMONIC_RE.match(line)
            if not match:
                if stripped.startswith("imports:") or stripped.endswith(":"):
                    break
                continue
            mnemonic, count = match.group(1), int(match.group(2))
            counts[mnemonic.lower()] = counts.get(mnemonic.lower(), 0) + count
    return counts


# ---------------------------------------------------------------------------
# Function graph (discover --json + optional lift overlays)
# ---------------------------------------------------------------------------


def _name_matches_target(name: str, target: str) -> bool:
    left = name.strip()
    right = target.strip()
    if not left or not right:
        return False
    if left.lower() == right.lower():
        return True
    if left.lower().endswith("!" + right.lower()):
        return True
    return right.lower() in left.lower().split("::")[-1].lower()


class FunctionNode:
    __slots__ = (
        "entry",
        "blocks",
        "instructions",
        "callees",
        "named_callees",
        "indirect_calls",
        "unsupported",
        "lifted",
        "name",
    )

    def __init__(self, entry: str) -> None:
        self.entry = entry
        self.blocks = 0
        self.instructions = 0
        self.callees: list[str] = []
        self.named_callees: list[str] = []
        self.indirect_calls = 0
        self.unsupported: dict[str, int] = {}
        self.lifted = True
        self.name: str | None = None


def _add_callee(node: FunctionNode, raw: Any) -> None:
    if raw is None:
        return
    va = parse_va(raw)
    if va is not None:
        label = format_va(va)
        if label not in node.callees:
            node.callees.append(label)
        return
    if isinstance(raw, str) and raw.strip():
        name = sanitize_text(raw.strip())
        if name not in node.named_callees:
            node.named_callees.append(name)


def _parse_unsupported(raw: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            mnemonic = str(key).strip().lower()
            if mnemonic:
                counts[mnemonic] = counts.get(mnemonic, 0) + as_int(value, 0)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                mnemonic = item.strip().lower()
                if mnemonic:
                    counts[mnemonic] = counts.get(mnemonic, 0) + 1
            elif isinstance(item, Mapping):
                mnemonic = str(item.get("mnemonic") or item.get("kind") or "").strip().lower()
                if mnemonic:
                    counts[mnemonic] = counts.get(mnemonic, 0) + max(as_int(item.get("count"), 1), 1)
    return counts


def ingest_discover_functions(discover: Mapping[str, Any] | None) -> dict[str, FunctionNode]:
    nodes: dict[str, FunctionNode] = {}
    if not discover:
        return nodes
    for item in discover.get("functions") or []:
        if not isinstance(item, Mapping):
            continue
        entry_raw = item.get("entry") or item.get("va") or item.get("address")
        va = parse_va(entry_raw)
        if va is None:
            continue
        entry = format_va(va)
        node = nodes.get(entry) or FunctionNode(entry)
        node.blocks = as_int(item.get("blocks"), node.blocks)
        node.instructions = as_int(item.get("instructions"), node.instructions)
        node.indirect_calls = as_int(item.get("indirect_calls"), node.indirect_calls)
        if item.get("name"):
            node.name = sanitize_text(str(item["name"]))
        for field in ("callees", "tail_calls", "calls"):
            values = item.get(field) or []
            if isinstance(values, list):
                for callee in values:
                    _add_callee(node, callee)
        imports = item.get("imports") or item.get("import_callees") or []
        if isinstance(imports, list):
            for imported in imports:
                _add_callee(node, imported)
        unsupported = _parse_unsupported(
            item.get("unsupported_mnemonics")
            or item.get("unsupported_by_mnemonic")
            or item.get("unsupported")
        )
        for mnemonic, count in unsupported.items():
            node.unsupported[mnemonic] = node.unsupported.get(mnemonic, 0) + count
        if "lifted" in item:
            node.lifted = bool(item["lifted"])
        elif node.unsupported:
            node.lifted = False
        nodes[entry] = node
    return nodes


def ingest_tls_and_roots(
    manifest: Mapping[str, Any] | None,
    discover: Mapping[str, Any] | None,
) -> list[str]:
    roots: list[str] = []

    def push(raw: Any) -> None:
        va = parse_va(raw)
        if va is None:
            return
        label = format_va(va)
        if label not in roots:
            roots.append(label)

    if manifest:
        image = manifest.get("image") or {}
        if isinstance(image, Mapping):
            push(image.get("entry_point"))
        options = manifest.get("options") or {}
        if isinstance(options, Mapping):
            for extra in options.get("extra_entries") or []:
                push(extra)
        refinements = manifest.get("refinements") or {}
        if isinstance(refinements, Mapping):
            code_pointers = refinements.get("code_pointers") or {}
            if isinstance(code_pointers, Mapping):
                for accepted in code_pointers.get("accepted") or []:
                    push(accepted)

    if discover:
        tls = discover.get("tls_callbacks") or discover.get("tls") or []
        if isinstance(tls, Mapping):
            tls = tls.get("callbacks") or tls.get("entry_points") or []
        if isinstance(tls, list):
            for cb in tls:
                if isinstance(cb, Mapping):
                    push(cb.get("va") or cb.get("entry") or cb.get("address"))
                else:
                    push(cb)
        entries = discover.get("entry_points") or []
        if isinstance(entries, list):
            # Discover lists PE entry, exports, TLS callbacks together; keep order.
            for entry in entries:
                push(entry)

    return roots


def _count_function_coverage(
    nodes: Mapping[str, FunctionNode],
    discovered_functions: int,
    lifted_fn_count: int,
    address_taken: int,
    has_lift_data: bool,
) -> tuple[int, int]:
    """Return `(lifted_functions, unlifted_functions)`.

    Per-function `unsupported_mnemonics` mark a function as unlifted even when
    R9 emitted IR for every discovered CFG (traps at run time). Discover-only
    inputs with no lift section count as fully unlifted.
    """
    unlifted_from_nodes = sum(1 for n in nodes.values() if n.unsupported or not n.lifted)
    if nodes and unlifted_from_nodes:
        lifted = max(discovered_functions - unlifted_from_nodes, 0)
        return lifted, unlifted_from_nodes + address_taken
    if has_lift_data:
        lifted = lifted_fn_count
        unlifted = max(discovered_functions - lifted, 0) + address_taken
        return max(lifted, 0), max(unlifted, 0)
    return 0, max(discovered_functions, len(nodes)) + address_taken


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _mnemonic_rows(
    counts: Mapping[str, int],
    affected: Mapping[str, set[str]],
    top: int,
) -> list[dict[str, Any]]:
    rows: list[tuple[str, int, int]] = []
    keys = set(counts) | set(affected)
    for mnemonic in keys:
        count = int(counts.get(mnemonic, 0))
        funcs = len(affected.get(mnemonic, ()))
        if count <= 0 and funcs <= 0:
            continue
        if count <= 0:
            count = funcs
        rows.append((mnemonic, count, funcs))
    rows.sort(key=lambda row: (-row[1], -row[2], row[0]))
    limited = rows[: max(top, 0)]
    return [
        {
            "mnemonic": name,
            "count": count,
            "affected_functions": funcs,
        }
        for name, count, funcs in limited
    ]


def _shard_stats(manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    empty = {"count": 0, "min": 0, "median": 0.0, "max": 0, "unit": "instructions"}
    if not manifest:
        return empty
    lift = manifest.get("lift") or {}
    if not isinstance(lift, Mapping):
        return empty
    shards = lift.get("shards") or []
    if not isinstance(shards, list) or not shards:
        return empty
    sizes: list[int] = []
    for shard in shards:
        if isinstance(shard, Mapping):
            sizes.append(as_int(shard.get("instructions"), 0))
        elif isinstance(shard, (int, float)):
            sizes.append(int(shard))
    if not sizes:
        return empty
    return {
        "count": len(sizes),
        "min": min(sizes),
        "median": median(sizes),
        "max": max(sizes),
        "unit": "instructions",
    }


def _jump_tables(
    manifest: Mapping[str, Any] | None,
    discover: Mapping[str, Any] | None,
) -> dict[str, Any]:
    resolved = 0
    unresolved = 0
    source = None
    if discover and isinstance(discover.get("jump_tables"), Mapping):
        source = discover["jump_tables"]
    elif manifest:
        discovery = manifest.get("discovery") or {}
        if isinstance(discovery, Mapping) and isinstance(discovery.get("jump_tables"), Mapping):
            source = discovery["jump_tables"]
    if isinstance(source, Mapping):
        resolved = as_int(source.get("resolved"), 0)
        unresolved = as_int(source.get("unresolved"), 0)
    total = resolved + unresolved
    return {
        "resolved_jump_tables": resolved,
        "unresolved_indirect_jumps": unresolved,
        "resolution_pct": as_float_pct(resolved, total),
    }


def boot_bfs(
    nodes: Mapping[str, FunctionNode],
    roots: Sequence[str],
    boot_depth: int | None,
) -> tuple[list[str], dict[str, list[str]], int]:
    """BFS over direct/tail callees. `boot_depth` is max edge distance from a root."""
    reachable: list[str] = []
    parents: dict[str, str | None] = {}
    paths: dict[str, list[str]] = {}
    indirect = 0
    queue: deque[tuple[str, int]] = deque()
    seen: set[str] = set()

    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        queue.append((root, 0))
        parents[root] = None
        paths[root] = [root]

    while queue:
        current, depth = queue.popleft()
        reachable.append(current)
        node = nodes.get(current)
        if node is None:
            continue
        indirect += node.indirect_calls
        if boot_depth is not None and depth >= boot_depth:
            continue
        for callee in node.callees:
            if callee in seen:
                continue
            seen.add(callee)
            parents[callee] = current
            paths[callee] = paths[current] + [callee]
            queue.append((callee, depth + 1))
        for name in node.named_callees:
            if name in seen:
                continue
            seen.add(name)
            parents[name] = current
            paths[name] = paths[current] + [name]
            queue.append((name, depth + 1))

    return reachable, paths, indirect


def _shortest_boot_targets(
    nodes: Mapping[str, FunctionNode],
    paths: Mapping[str, list[str]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    named_nodes = list(paths.keys())
    for target in BOOT_TARGETS:
        match: str | None = None
        if target in paths:
            match = target
        else:
            for key in named_nodes:
                if _name_matches_target(key, target):
                    match = key
                    break
            if match is None:
                for node in nodes.values():
                    for name in node.named_callees:
                        if _name_matches_target(name, target) and node.entry in paths:
                            match = name
                            path = list(paths[node.entry]) + [name]
                            result[target] = {
                                "reachable": True,
                                "via": name,
                                "shortest_path": path,
                            }
                            break
                    if target in result:
                        break
                if target in result:
                    continue
        if match is not None and match in paths:
            result[target] = {
                "reachable": True,
                "via": match,
                "shortest_path": list(paths[match]),
            }
        else:
            result[target] = {"reachable": False, "via": None, "shortest_path": []}
    return result


def _recommendations(
    coverage: Mapping[str, Any],
    boot: Mapping[str, Any],
    jump: Mapping[str, Any],
    mnemonics: Sequence[Mapping[str, Any]],
    unresolved_imports: Sequence[str],
    address_taken_not_lifted: int,
) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []

    def add(category: str, detail: str) -> None:
        recs.append({"priority": len(recs) + 1, "category": category, "detail": detail})

    boot_pct = float(boot.get("boot_lifted_pct") or 0.0)
    if boot.get("boot_reachable_functions") and boot_pct < 100.0:
        add(
            "boot_lift",
            f"Boot-path lift coverage is {boot_pct:.2f}%; lift remaining reachable functions before splash.",
        )
    boot_mnems = boot.get("boot_unlifted_mnemonics") or []
    if boot_mnems:
        top = boot_mnems[0]
        add(
            "boot_mnemonic",
            f"Implement lifter support for `{top['mnemonic']}` "
            f"({top['count']} hits on the boot path, {top['affected_functions']} functions).",
        )
    missing_targets = [
        name
        for name, info in (boot.get("boot_targets") or {}).items()
        if not (info or {}).get("reachable")
    ]
    if missing_targets:
        add(
            "boot_gpu_imports",
            "No call-graph path to "
            + " / ".join(missing_targets)
            + "; wire import thunks and confirm IAT calls from the boot chain.",
        )
    else:
        reachable_targets = [
            name
            for name, info in (boot.get("boot_targets") or {}).items()
            if (info or {}).get("reachable")
        ]
        if reachable_targets:
            add(
                "boot_gpu_imports",
                "Boot path reaches "
                + " / ".join(reachable_targets)
                + "; keep those d3d12_xs shims complete for PresentX splash.",
            )
    if as_int(boot.get("boot_unresolved_indirect_calls"), 0) > 0:
        add(
            "boot_indirect",
            f"{boot['boot_unresolved_indirect_calls']} unresolved indirect calls on the boot path "
            "(IAT/vcall); resolve slots or add dispatcher shims.",
        )
    if float(jump.get("resolution_pct") or 0.0) < 100.0 and (
        as_int(jump.get("unresolved_indirect_jumps"), 0) > 0
    ):
        add(
            "jump_tables",
            f"{jump['unresolved_indirect_jumps']} unresolved indirect jumps remain "
            f"({jump['resolution_pct']:.2f}% tables resolved); improve switch-table recovery.",
        )
    if mnemonics and not boot_mnems:
        top = mnemonics[0]
        add(
            "lifter_mnemonic",
            f"Highest remaining unlifted mnemonic is `{top['mnemonic']}` "
            f"({top['count']} occurrences, {top['affected_functions']} functions).",
        )
    if unresolved_imports:
        preview = ", ".join(unresolved_imports[:6])
        extra = "" if len(unresolved_imports) <= 6 else f" (+{len(unresolved_imports) - 6} more)"
        add(
            "unresolved_imports",
            f"Bind remaining unresolved imports referenced by discovered code: {preview}{extra}.",
        )
    if address_taken_not_lifted > 0:
        add(
            "address_taken",
            f"{address_taken_not_lifted} address-taken code pointers are not lifted; "
            "seed them as extra `--entry` points.",
        )
    if float(coverage.get("lifted_pct") or 0.0) < 100.0 and not recs:
        add(
            "coverage",
            f"Overall function lift coverage is {coverage['lifted_pct']:.2f}%; "
            "expand the lifter for remaining unlifted functions.",
        )
    if not recs:
        add(
            "ready",
            "No remaining lift/boot blockers in this report; run the guest image against the splash path.",
        )
    return recs


def _unresolved_import_names(manifest: Mapping[str, Any] | None) -> list[str]:
    names: list[str] = []
    if not manifest:
        return names
    refinements = manifest.get("refinements") or {}
    if not isinstance(refinements, Mapping):
        return names
    slots = refinements.get("import_slots") or {}
    if not isinstance(slots, Mapping):
        return names
    for item in slots.get("referenced_unresolved") or []:
        if isinstance(item, Mapping):
            imported = str(item.get("import") or "").strip()
            dll = str(item.get("dll") or "").strip()
            if imported and dll:
                label = f"{dll}!{imported}"
            else:
                label = imported or dll
            if label and label not in names:
                names.append(sanitize_text(label))
    return names


def audit_recomp_build(
    path: str | Path,
    *,
    top: int = DEFAULT_TOP,
    boot_depth: int | None = None,
) -> dict[str, Any]:
    """Audit a recomp build directory, `build-manifest.json`, or discover JSON file."""
    if top < 0:
        raise AuditError("--top must be >= 0")
    if boot_depth is not None and boot_depth < 0:
        raise AuditError("--boot-depth must be >= 0")

    manifest, discover, source, report_text = load_inputs(path)
    nodes = ingest_discover_functions(discover)
    roots = ingest_tls_and_roots(manifest, discover)

    discovery = (manifest or {}).get("discovery") if manifest else None
    lift = (manifest or {}).get("lift") if manifest else None
    discovery = discovery if isinstance(discovery, Mapping) else {}
    lift = lift if isinstance(lift, Mapping) else {}

    discovered_functions = as_int(
        discovery.get("functions"),
        as_int((discover or {}).get("function_count"), len(nodes)),
    )
    if not discovered_functions:
        discovered_functions = len(nodes)

    lifted_fn_count = as_int(lift.get("functions"), 0)
    address_taken = 0
    refinements = (manifest or {}).get("refinements") if manifest else None
    if isinstance(refinements, Mapping):
        dispatch = refinements.get("dispatch") or {}
        if isinstance(dispatch, Mapping):
            not_lifted = dispatch.get("address_taken_not_lifted") or []
            if isinstance(not_lifted, list):
                address_taken = len(not_lifted)

    for node in nodes.values():
        if node.unsupported:
            node.lifted = False

    lifted_functions, unlifted_functions = _count_function_coverage(
        nodes,
        discovered_functions,
        lifted_fn_count,
        address_taken,
        has_lift_data=bool(lift),
    )

    total_blocks = as_int(
        discovery.get("blocks"),
        as_int((discover or {}).get("block_count"), sum(n.blocks for n in nodes.values())),
    )
    total_instructions = as_int(
        lift.get("instructions"),
        as_int(
            discovery.get("instructions"),
            as_int((discover or {}).get("instruction_count"), sum(n.instructions for n in nodes.values())),
        ),
    )
    unsupported_instructions = as_int(lift.get("unsupported_instructions"), 0)
    if not unsupported_instructions:
        unsupported_instructions = sum(sum(n.unsupported.values()) for n in nodes.values())
    lifted_instructions = max(total_instructions - unsupported_instructions, 0)
    # Blocks: without per-block lift data, treat unsupported functions' blocks as unlifted.
    unlifted_blocks = sum(n.blocks for n in nodes.values() if not n.lifted)
    lifted_blocks = max(total_blocks - unlifted_blocks, 0) if total_blocks else 0

    coverage = {
        "lifted_functions": lifted_functions,
        "unlifted_functions": unlifted_functions,
        "discovered_functions": discovered_functions,
        "lifted_pct": as_float_pct(lifted_functions, discovered_functions),
        "lifted_blocks": lifted_blocks,
        "unlifted_blocks": unlifted_blocks,
        "block_lifted_pct": as_float_pct(lifted_blocks, total_blocks),
        "lifted_instructions": lifted_instructions,
        "unlifted_instructions": unsupported_instructions,
        "instruction_lifted_pct": as_float_pct(lifted_instructions, total_instructions),
        "unsupported_instructions": unsupported_instructions,
    }

    mnemonic_counts: dict[str, int] = {}
    mnemonic_funcs: dict[str, set[str]] = defaultdict(set)
    for node in nodes.values():
        for mnemonic, count in node.unsupported.items():
            mnemonic_counts[mnemonic] = mnemonic_counts.get(mnemonic, 0) + count
            mnemonic_funcs[mnemonic].add(node.entry)
    if not mnemonic_counts:
        by_mnemonic = lift.get("unsupported_by_mnemonic") if lift else None
        if isinstance(by_mnemonic, Mapping):
            for key, value in by_mnemonic.items():
                mnemonic = str(key).strip().lower()
                if mnemonic:
                    mnemonic_counts[mnemonic] = mnemonic_counts.get(mnemonic, 0) + as_int(value, 0)
    if not mnemonic_counts:
        for mnemonic, count in parse_report_mnemonics(report_text).items():
            mnemonic_counts[mnemonic] = mnemonic_counts.get(mnemonic, 0) + count

    mnemonics = _mnemonic_rows(mnemonic_counts, mnemonic_funcs, top)
    shards = _shard_stats(manifest)
    jump = _jump_tables(manifest, discover)

    if not roots and nodes:
        roots = list(nodes.keys())[:1]

    reachable, paths, boot_indirect = boot_bfs(nodes, roots, boot_depth)
    boot_nodes = [nodes[e] for e in reachable if e in nodes]
    if boot_nodes:
        boot_unlifted = sum(1 for n in boot_nodes if not n.lifted)
        boot_lifted = len(boot_nodes) - boot_unlifted
        boot_lifted_pct = as_float_pct(boot_lifted, len(boot_nodes))
    elif reachable:
        boot_lifted_pct = coverage["lifted_pct"]
        boot_lifted = 0
        boot_unlifted = 0
    else:
        boot_lifted_pct = coverage["lifted_pct"]
        boot_lifted = 0
        boot_unlifted = 0

    boot_mnem_counts: dict[str, int] = {}
    boot_mnem_funcs: dict[str, set[str]] = defaultdict(set)
    for node in boot_nodes:
        for mnemonic, count in node.unsupported.items():
            boot_mnem_counts[mnemonic] = boot_mnem_counts.get(mnemonic, 0) + count
            boot_mnem_funcs[mnemonic].add(node.entry)

    boot_targets = _shortest_boot_targets(nodes, paths)
    boot = {
        "roots": roots,
        "boot_depth": boot_depth,
        "boot_reachable_functions": len(reachable) if (nodes or roots) else discovered_functions,
        "boot_lifted_functions": boot_lifted if boot_nodes else lifted_functions,
        "boot_unlifted_functions": boot_unlifted if boot_nodes else unlifted_functions,
        "boot_lifted_pct": boot_lifted_pct if boot_nodes else coverage["lifted_pct"],
        "boot_unlifted_mnemonics": _mnemonic_rows(boot_mnem_counts, boot_mnem_funcs, top),
        "boot_unresolved_indirect_calls": boot_indirect,
        "boot_targets": boot_targets,
        "reachable_entries": reachable,
    }

    unresolved_imports = _unresolved_import_names(manifest)
    recs = _recommendations(
        coverage,
        boot,
        jump,
        mnemonics,
        unresolved_imports,
        address_taken,
    )

    module = ""
    if manifest and isinstance(manifest.get("module"), str):
        module = sanitize_text(manifest["module"])

    report = {
        "schema": SCHEMA_AUDIT,
        "schema_version": SCHEMA_AUDIT_VERSION,
        "source": source,
        "module": module,
        "coverage": coverage,
        "shards": shards,
        "unlifted_mnemonics": mnemonics,
        "jump_tables": jump,
        "boot": boot,
        "unresolved_imports": unresolved_imports,
        "address_taken_not_lifted": address_taken,
        "recommendations": recs,
    }
    return sanitize_value(report)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def format_text_report(report: Mapping[str, Any]) -> str:
    cov = report["coverage"]
    shards = report["shards"]
    jump = report["jump_tables"]
    boot = report["boot"]
    lines = [
        f"playgta6-recomp coverage audit: {report.get('source') or '<unknown>'}",
        f"module: {report.get('module') or '-'}",
        "",
        "coverage:",
        f"  functions:    {cov['lifted_functions']} lifted / {cov['unlifted_functions']} unlifted "
        f"of {cov['discovered_functions']} ({cov['lifted_pct']:.2f}%)",
        f"  blocks:       {cov['lifted_blocks']} lifted / {cov['unlifted_blocks']} unlifted "
        f"({cov['block_lifted_pct']:.2f}%)",
        f"  instructions: {cov['lifted_instructions']} lifted / {cov['unlifted_instructions']} unlifted "
        f"({cov['instruction_lifted_pct']:.2f}%)",
        "",
        "shards:",
        f"  count {shards['count']}, size ({shards['unit']}) "
        f"min {shards['min']} / median {shards['median']} / max {shards['max']}",
        "",
        "jump tables:",
        f"  resolved {jump['resolved_jump_tables']}, unresolved {jump['unresolved_indirect_jumps']} "
        f"({jump['resolution_pct']:.2f}%)",
        "",
        "boot path:",
        f"  roots: {', '.join(boot.get('roots') or []) or '-'}",
        f"  reachable {boot['boot_reachable_functions']} functions, "
        f"lifted {boot['boot_lifted_pct']:.2f}%, "
        f"unresolved indirect calls {boot['boot_unresolved_indirect_calls']}",
    ]
    depth = boot.get("boot_depth")
    if depth is not None:
        lines.append(f"  depth limit: {depth}")
    for name, info in (boot.get("boot_targets") or {}).items():
        if info.get("reachable"):
            path = " -> ".join(info.get("shortest_path") or [])
            lines.append(f"  {name}: reachable via {path}")
        else:
            lines.append(f"  {name}: not reachable on discovered boot path")
    lines.append("")
    lines.append("top unlifted mnemonics:")
    mnems = report.get("unlifted_mnemonics") or []
    if not mnems:
        lines.append("  (none)")
    else:
        for row in mnems:
            lines.append(
                f"  {row['mnemonic']:<16} {row['count']:>8}  functions {row['affected_functions']}"
            )
    boot_mnems = boot.get("boot_unlifted_mnemonics") or []
    lines.append("boot unlifted mnemonics:")
    if not boot_mnems:
        lines.append("  (none)")
    else:
        for row in boot_mnems:
            lines.append(
                f"  {row['mnemonic']:<16} {row['count']:>8}  functions {row['affected_functions']}"
            )
    lines.append("")
    lines.append("recommendations:")
    for rec in report.get("recommendations") or []:
        lines.append(f"  {rec['priority']}. [{rec['category']}] {rec['detail']}")
    return "\n".join(lines) + "\n"


def format_markdown_report(report: Mapping[str, Any]) -> str:
    cov = report["coverage"]
    shards = report["shards"]
    jump = report["jump_tables"]
    boot = report["boot"]
    lines = [
        f"# Recomp coverage audit (`{report.get('source') or 'unknown'}`)",
        "",
        f"- Module: `{report.get('module') or '-'}`",
        f"- Functions: **{cov['lifted_functions']}** lifted / **{cov['unlifted_functions']}** unlifted "
        f"of {cov['discovered_functions']} (**{cov['lifted_pct']:.2f}%**)",
        f"- Blocks: {cov['lifted_blocks']} / {cov['unlifted_blocks']} ({cov['block_lifted_pct']:.2f}%)",
        f"- Instructions: {cov['lifted_instructions']} / {cov['unlifted_instructions']} "
        f"({cov['instruction_lifted_pct']:.2f}%)",
        f"- Shards: {shards['count']} "
        f"(min {shards['min']}, median {shards['median']}, max {shards['max']} {shards['unit']})",
        f"- Jump tables: {jump['resolved_jump_tables']} resolved, "
        f"{jump['unresolved_indirect_jumps']} unresolved ({jump['resolution_pct']:.2f}%)",
        "",
        "## Boot path",
        "",
        f"- Reachable functions: {boot['boot_reachable_functions']}",
        f"- Boot lifted: {boot['boot_lifted_pct']:.2f}%",
        f"- Unresolved indirect calls: {boot['boot_unresolved_indirect_calls']}",
        f"- Roots: {', '.join(f'`{r}`' for r in (boot.get('roots') or [])) or '_'}",
        "",
    ]
    for name, info in (boot.get("boot_targets") or {}).items():
        if info.get("reachable"):
            path = " → ".join(f"`{p}`" for p in (info.get("shortest_path") or []))
            lines.append(f"- `{name}`: reachable {path}")
        else:
            lines.append(f"- `{name}`: **not reachable**")
    lines.extend(["", "## Top unlifted mnemonics", ""])
    mnems = report.get("unlifted_mnemonics") or []
    if not mnems:
        lines.append("_None._")
    else:
        lines.append("| Mnemonic | Count | Functions |")
        lines.append("| --- | ---: | ---: |")
        for row in mnems:
            lines.append(f"| `{row['mnemonic']}` | {row['count']} | {row['affected_functions']} |")
    lines.extend(["", "## Recommendations", ""])
    for rec in report.get("recommendations") or []:
        lines.append(f"{rec['priority']}. **{rec['category']}** — {rec['detail']}")
    return "\n".join(lines) + "\n"


def format_json_report(report: Mapping[str, Any]) -> str:
    return json.dumps(sanitize_value(report), indent=2, sort_keys=False) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit playgta6-recomp build coverage and boot-path lift bottlenecks "
            "from build-manifest.json and/or discover --json."
        )
    )
    parser.add_argument(
        "input",
        help="Path to a recomp build directory, build-manifest.json, or discover JSON file.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP,
        metavar="N",
        help=f"Top-N unlifted mnemonics to rank (default {DEFAULT_TOP}).",
    )
    parser.add_argument(
        "--boot-depth",
        type=int,
        default=None,
        metavar="D",
        help="Maximum BFS depth from entry/TLS roots (default: unlimited).",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Emit a sanitized Markdown report.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a sanitized machine-readable JSON report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        report = audit_recomp_build(
            args.input,
            top=args.top,
            boot_depth=args.boot_depth,
        )
    except AuditError as exc:
        print(f"Error: {sanitize_text(str(exc))}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(
            f"Error: {sanitize_text(exc.strerror or type(exc).__name__)}",
            file=sys.stderr,
        )
        return 1

    if args.json:
        sys.stdout.write(format_json_report(report))
    elif args.markdown:
        sys.stdout.write(format_markdown_report(report))
    else:
        sys.stdout.write(format_text_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
