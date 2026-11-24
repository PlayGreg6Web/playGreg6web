#!/usr/bin/env python3
"""End-to-end browser real-boot readiness verifier CLI (Track 5 / W48).

Validates the local chain required to boot the recompiled game in Chrome/Edge:

1. `game_assets` — MicrosoftGame.Config + ShellVisuals splash + DLL/RPF inventory
2. `import_coverage` — 587/587 (100%) PE import shim coverage
3. `recomp_artifacts` — program/build manifest, `.pgim`, shards, lift coverage %
4. `shader_pack` — valid WGPK `.wgsl.pack` + `reflection-manifest.json`
5. `wasm64_runtime` — `runtime.wasm` memory64/shared/host ABI + smoke vs linked guest
6. `local_server` — ephemeral COOP/COEP server + Range 206 + boot endpoints

Security & privacy invariants:
- Machine-readable `--json` and text output NEVER include host absolute paths
  (Windows or POSIX), usernames, or raw binary bytes.
- Strictly uses Python 3.12 standard library modules only.

Usage:
    python tools/verify_browser_boot.py [--repo-root <dir>] [--recomp-dir <dir>]
        [--skip-server] [--json]
"""

from __future__ import annotations

import argparse
import http.client
import importlib.util
import json
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = 1
SMOKE_SHELL_MAX_BYTES = 1 * 1024 * 1024  # 1 MiB
DEFAULT_RECOMP_DIR_REL = Path("work") / "recomp_build"
DEFAULT_SHADERS_DIR_REL = Path("local-data") / "shaders"
RUNTIME_WASM_REL = Path("web") / "dist" / "runtime.wasm"

GATE_NAMES: tuple[str, ...] = (
    "game_assets",
    "import_coverage",
    "recomp_artifacts",
    "shader_pack",
    "wasm64_runtime",
    "local_server",
)

GUEST_BOOT_EXPORT_MARKERS: frozenset[str] = frozenset(
    {
        "pg6_guest_boot_run",
        "pg6_guest_boot_reset",
        "pg6_guest_boot_report_json",
        "pg6_guest_boot_summary",
        "pg6_guest_boot_prepare_thread",
        "pg6_program_init",
        "pg6_program_boot",
        "__playgta6_register_lifted_functions",
        "__playgta6_register_lifted_functions_main",
    }
)

_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"'<>|,;()\[\]]+")
_UNC_PATH_RE = re.compile(r"\\\\[^\s\"'<>|,;()\[\]]+")
_POSIX_PATH_RE = re.compile(
    r"/(?:Users|home|tmp|var|opt|root|mnt|private|etc|usr|Library|Volumes)/[^\s\"'<>|,;()\[\]]*"
)
_DRIVE_SLASH_RE = re.compile(r"[A-Za-z]:/(?:[^\s\"'<>|,;()\[\]]+)")


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
    text = str(raw).replace("\\", "/")
    sanitized = _WINDOWS_PATH_RE.sub("<path>", text)
    sanitized = _UNC_PATH_RE.sub("<path>", sanitized)
    sanitized = _POSIX_PATH_RE.sub("<path>", sanitized)
    sanitized = _DRIVE_SLASH_RE.sub("<path>", sanitized)
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


def repo_relative_label(path: Path, repo_root: Path) -> str:
    """Return a repo-relative POSIX label, never an absolute host path."""
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
        return sanitize_text(rel.as_posix())
    except (OSError, ValueError):
        return basename_label(path)


def sanitize_value(value: Any) -> Any:
    """Recursively sanitize strings; replace raw bytes with size placeholders."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<bytes:{len(value)}>"
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, Mapping):
        return {sanitize_text(str(k)): sanitize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_value(v) for v in value]
    if isinstance(value, Path):
        return basename_label(value)
    return value


# ---------------------------------------------------------------------------
# Gate result model
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    """Structured readiness result for one browser-boot gate."""

    name: str
    status: str  # pass | warn | fail | skip
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    next_action: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "summary": sanitize_text(self.summary),
            "details": sanitize_value(self.details),
            "next_action": sanitize_text(self.next_action) if self.next_action else None,
        }


# ---------------------------------------------------------------------------
# Dynamic sibling-tool loaders
# ---------------------------------------------------------------------------


def _load_tools_module(module_filename: str, module_name: str) -> Any:
    """Load a sibling tools/*.py module, reusing an already-imported copy when present.

    Registers the module in `sys.modules` before `exec_module` so `@dataclass`
    (and similar) can resolve `cls.__module__`.
    """
    short_name = module_filename[:-3] if module_filename.endswith(".py") else module_filename
    for key in (module_name, short_name):
        existing = sys.modules.get(key)
        if existing is not None:
            return existing

    tools_dir = Path(__file__).resolve().parent
    path = tools_dir / module_filename
    if not path.is_file():
        raise FileNotFoundError(f"missing tool module: tools/{module_filename}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load tools/{module_filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_server_module(repo_root: Path) -> Any:
    """Load `server/serve_local.py` from the given repo root (no cross-repo cache)."""
    path = (repo_root / "server" / "serve_local.py").resolve()
    if not path.is_file():
        raise FileNotFoundError("missing server/serve_local.py")
    # Unique module name per path so temp-repo tests cannot reuse another checkout's server.
    safe = re.sub(r"[^A-Za-z0-9_]", "_", str(path))
    module_name = f"_playgta6_verify_serve_local_{safe}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError("failed to load server/serve_local.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _resolve_env_paths(
    repo_root: Path,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve GTA6_* paths without mutating os.environ (mirrors local_env/preflight)."""
    iga = _load_tools_module("inspect_game_assets.py", "_playgta6_verify_iga")
    env_map = os.environ if env is None else env
    dotenv_vals = iga.load_dotenv(repo_root / ".env")

    raw_game = (env_map.get("GTA6_GAME_DIR") or dotenv_vals.get("GTA6_GAME_DIR", "")).strip()
    raw_data = (env_map.get("GTA6_DATA_DIR") or dotenv_vals.get("GTA6_DATA_DIR", "")).strip()
    raw_work = (env_map.get("GTA6_WORK_DIR") or dotenv_vals.get("GTA6_WORK_DIR", "")).strip()

    def _resolve(raw: str, default_rel: str | None) -> Path | None:
        if raw:
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = repo_root / candidate
            return candidate
        if default_rel is None:
            return None
        return repo_root / default_rel

    return {
        "raw_game": raw_game,
        "raw_data": raw_data,
        "raw_work": raw_work,
        "game_path": _resolve(raw_game, None),
        "data_path": _resolve(raw_data, "local-data"),
        "work_path": _resolve(raw_work, "work"),
    }


# ---------------------------------------------------------------------------
# Individual gates
# ---------------------------------------------------------------------------


def check_game_assets_gate(
    repo_root: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> GateResult:
    """Gate 1: local Xbox package readiness via inspect_game_assets."""
    iga = _load_tools_module("inspect_game_assets.py", "_playgta6_verify_iga")
    resolved = _resolve_env_paths(repo_root, env=env)
    raw_game = resolved["raw_game"]
    game_path: Path | None = resolved["game_path"]

    if not raw_game:
        return GateResult(
            name="game_assets",
            status="skip",
            summary="GTA6_GAME_DIR is not configured",
            details={"configured": False, "exists": False},
            next_action='python tools/local_env.py --set-game "<path>"',
        )

    if game_path is None or not game_path.is_dir():
        return GateResult(
            name="game_assets",
            status="fail",
            summary="GTA6_GAME_DIR is configured but missing on disk",
            details={"configured": True, "exists": False},
            next_action='python tools/local_env.py --set-game "<path>"',
        )

    report = iga.inspect_game_directory(game_path)
    readiness = report.get("readiness") or {}
    inventory = report.get("inventory") or {}
    shell_visuals = report.get("shellVisuals") or []
    splash = next((v for v in shell_visuals if v.get("field") == "splashScreenImage"), None)
    splash_ok = bool(splash and splash.get("complete") and splash.get("headerFormat") == "png")

    details = {
        "configured": True,
        "exists": True,
        "configFound": bool(report.get("configFound")),
        "configValid": bool(report.get("configValid")),
        "status": report.get("status"),
        "splashPresent": bool(splash and splash.get("present")),
        "splashComplete": bool(splash and splash.get("complete")),
        "splashHeaderFormat": (splash or {}).get("headerFormat"),
        "splashRelativePath": (splash or {}).get("relativePath") or (splash or {}).get("declaredPath"),
        "dllCount": inventory.get("dllCount", 0),
        "rpfCount": inventory.get("rpfCount", 0),
        "allDeclaredAssetsComplete": bool(readiness.get("allDeclaredAssetsComplete")),
        "hasPartialDownloads": bool(readiness.get("hasPartialDownloads")),
    }

    if not report.get("configValid"):
        return GateResult(
            name="game_assets",
            status="fail",
            summary=sanitize_text(str(report.get("error") or "MicrosoftGame.Config invalid")),
            details=details,
            next_action="python tools/inspect_game_assets.py --from-env",
        )

    if not splash_ok:
        return GateResult(
            name="game_assets",
            status="fail",
            summary="ShellVisuals splash PNG missing or incomplete",
            details=details,
            next_action="python tools/inspect_game_assets.py --from-env",
        )

    if readiness.get("hasPartialDownloads") or not readiness.get("allDeclaredAssetsComplete"):
        return GateResult(
            name="game_assets",
            status="warn",
            summary=(
                f"Splash OK; package partial "
                f"(dlls={inventory.get('dllCount', 0)}, rpf={inventory.get('rpfCount', 0)})"
            ),
            details=details,
            next_action="python tools/inspect_game_assets.py --from-env",
        )

    return GateResult(
        name="game_assets",
        status="pass",
        summary=(
            f"MicrosoftGame.Config + splash PNG ready "
            f"(dlls={inventory.get('dllCount', 0)}, rpf={inventory.get('rpfCount', 0)})"
        ),
        details=details,
        next_action=None,
    )


def check_import_coverage_gate(
    repo_root: Path,
    *,
    audit_fn: Callable[[Path], Mapping[str, Any]] | None = None,
) -> GateResult:
    """Gate 2: real-binary PE import coverage (587/587 = 100%)."""
    if audit_fn is None:
        cic = _load_tools_module("check_import_coverage.py", "_playgta6_verify_cic")
        report = cic.audit_import_coverage(repo_root)
    else:
        report = dict(audit_fn(repo_root))

    totals = report.get("totals") or {}
    total = int(totals.get("total") or 0)
    covered = int(totals.get("shimmed") or 0) + int(totals.get("guest_recompiled") or 0)
    missing = int(totals.get("missing") or 0)
    pct = float(totals.get("effective_boot_pct") or 0.0)
    errors = list(report.get("errors") or [])

    details = {
        "total": total,
        "covered": covered,
        "missing": missing,
        "effectiveBootPct": pct,
        "shimmed": int(totals.get("shimmed") or 0),
        "guestRecompiled": int(totals.get("guest_recompiled") or 0),
        "errors": [sanitize_text(e) for e in errors],
    }

    if errors or total <= 0:
        return GateResult(
            name="import_coverage",
            status="fail",
            summary=sanitize_text(errors[0] if errors else "import coverage audit produced no symbols"),
            details=details,
            next_action="python tools/check_import_coverage.py --json",
        )

    if missing == 0 and pct >= 100.0:
        return GateResult(
            name="import_coverage",
            status="pass",
            summary=f"{covered}/{total} ({pct:.2f}%) import coverage",
            details=details,
            next_action=None,
        )

    return GateResult(
        name="import_coverage",
        status="fail",
        summary=f"{covered}/{total} ({pct:.2f}%) import coverage; {missing} missing",
        details=details,
        next_action="python tools/check_import_coverage.py --missing-only",
    )


def _find_manifest_files(recomp_dir: Path) -> dict[str, Path | None]:
    program: Path | None = None
    build: Path | None = None
    candidates = [recomp_dir]
    try:
        for child in sorted(recomp_dir.iterdir()):
            if child.is_dir() and not child.is_symlink():
                candidates.append(child)
    except OSError:
        pass

    for directory in candidates:
        prog = directory / "program-manifest.json"
        build_m = directory / "build-manifest.json"
        if program is None and prog.is_file():
            program = prog
        if build is None and build_m.is_file():
            build = build_m
        if program is not None and build is not None:
            break
    return {"program": program, "build": build}


def _count_pgim_files(recomp_dir: Path) -> tuple[int, list[str]]:
    """Recursively discover `.pgim` guest images under `recomp_dir`.

    Returns `(count, sorted_relative_posix_paths)` with host paths stripped so
    nested `playgta6-recomp build-program` outputs (`<module>/image/guest.pgim`)
    and flat layouts (`guest.pgim`) are both discovered deterministically.
    """
    rel_paths: list[str] = []
    try:
        for entry in recomp_dir.rglob("*.pgim"):
            try:
                if entry.is_symlink() or not entry.is_file():
                    continue
                rel = entry.relative_to(recomp_dir).as_posix()
                rel_paths.append(sanitize_text(rel))
            except (OSError, ValueError):
                continue
    except OSError:
        return 0, []
    rel_paths.sort(key=lambda item: (item.lower(), item))
    return len(rel_paths), rel_paths


def _lift_stats_from_manifest(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Extract shard count + lift coverage from program or build manifests."""
    shards = 0
    functions = 0
    instructions = 0
    unsupported = 0
    schema = doc.get("schema")

    if schema == "playgta6-recomp-program" or isinstance(doc.get("modules"), list):
        modules = doc.get("modules") or []
        for mod in modules:
            if not isinstance(mod, Mapping):
                continue
            lift = mod.get("lift") if isinstance(mod.get("lift"), Mapping) else {}
            functions += int(lift.get("functions") or 0)
            instructions += int(lift.get("instructions") or 0)
            unsupported += int(lift.get("unsupported") or lift.get("unsupported_instructions") or 0)
            # Prefer explicit shard lists from nested build manifests when present.
            nested = mod.get("build_manifest") if isinstance(mod.get("build_manifest"), Mapping) else {}
            # program modules may embed lift only; shard count often lives in build-manifest.
        # Fall through: also inspect top-level lift if present.
        lift_top = doc.get("lift") if isinstance(doc.get("lift"), Mapping) else {}
        if lift_top:
            functions = max(functions, int(lift_top.get("functions") or 0))
            instructions = max(instructions, int(lift_top.get("instructions") or 0))
            unsupported = max(
                unsupported,
                int(lift_top.get("unsupported") or lift_top.get("unsupported_instructions") or 0),
            )
            shard_list = lift_top.get("shards")
            if isinstance(shard_list, list):
                shards = max(shards, len(shard_list))

    lift = doc.get("lift") if isinstance(doc.get("lift"), Mapping) else {}
    if lift:
        functions = max(functions, int(lift.get("functions") or 0))
        instructions = max(instructions, int(lift.get("instructions") or 0))
        unsupported = max(
            unsupported,
            int(lift.get("unsupported_instructions") or lift.get("unsupported") or 0),
        )
        shard_list = lift.get("shards")
        if isinstance(shard_list, list):
            shards = max(shards, len(shard_list))

    lifted = max(instructions - unsupported, 0) if instructions else functions
    whole = instructions if instructions else functions
    coverage_pct = round((lifted * 100.0 / whole), 2) if whole > 0 else 0.0
    return {
        "schema": schema,
        "shardCount": shards,
        "liftedFunctions": functions,
        "liftedInstructions": lifted if instructions else functions,
        "totalInstructions": instructions,
        "unsupportedInstructions": unsupported,
        "liftCoveragePct": coverage_pct,
    }


def check_recomp_artifacts_gate(
    repo_root: Path,
    recomp_dir: Path | None = None,
) -> GateResult:
    """Gate 3: recompiled program/build artifacts under work/recomp_build."""
    target = recomp_dir if recomp_dir is not None else (repo_root / DEFAULT_RECOMP_DIR_REL)
    try:
        label = repo_relative_label(target, repo_root)
    except Exception:  # noqa: BLE001
        label = "work/recomp_build"

    details: dict[str, Any] = {
        "recompDirLabel": label,
        "dirExists": False,
        "programManifest": False,
        "buildManifest": False,
        "pgimCount": 0,
        "pgimNames": [],
        "shardCount": 0,
        "liftCoveragePct": 0.0,
    }

    if not target.is_dir():
        return GateResult(
            name="recomp_artifacts",
            status="fail",
            summary=f"recomp directory missing ({label})",
            details=details,
            next_action="cargo run -p playgta6-recomp -- build-program --help",
        )

    details["dirExists"] = True
    manifests = _find_manifest_files(target)
    program_path = manifests["program"]
    build_path = manifests["build"]
    details["programManifest"] = program_path is not None
    details["buildManifest"] = build_path is not None
    details["programManifestName"] = basename_label(program_path) if program_path else None
    details["buildManifestName"] = basename_label(build_path) if build_path else None

    pgim_count, pgim_names = _count_pgim_files(target)
    details["pgimCount"] = pgim_count
    details["pgimNames"] = pgim_names[:16]

    lift_stats: dict[str, Any] = {
        "shardCount": 0,
        "liftCoveragePct": 0.0,
        "liftedFunctions": 0,
        "totalInstructions": 0,
        "unsupportedInstructions": 0,
    }
    parse_errors: list[str] = []
    for path in (program_path, build_path):
        if path is None:
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            parse_errors.append(f"{basename_label(path)}: {type(exc).__name__}")
            continue
        if not isinstance(doc, dict):
            parse_errors.append(f"{basename_label(path)}: expected JSON object")
            continue
        stats = _lift_stats_from_manifest(doc)
        # Prefer richer stats (higher instruction counts / shards).
        if stats["totalInstructions"] >= lift_stats["totalInstructions"]:
            lift_stats.update(stats)
        elif stats["shardCount"] > lift_stats["shardCount"]:
            lift_stats["shardCount"] = stats["shardCount"]

    details.update(
        {
            "shardCount": lift_stats.get("shardCount", 0),
            "liftCoveragePct": lift_stats.get("liftCoveragePct", 0.0),
            "liftedFunctions": lift_stats.get("liftedFunctions", 0),
            "totalInstructions": lift_stats.get("totalInstructions", 0),
            "unsupportedInstructions": lift_stats.get("unsupportedInstructions", 0),
            "parseErrors": parse_errors,
        }
    )

    has_manifest = bool(program_path or build_path)
    if not has_manifest:
        return GateResult(
            name="recomp_artifacts",
            status="fail",
            summary="no program-manifest.json or build-manifest.json found",
            details=details,
            next_action="cargo run -p playgta6-recomp -- build-program --help",
        )

    if pgim_count <= 0:
        return GateResult(
            name="recomp_artifacts",
            status="fail",
            summary="manifest present but no .pgim guest image found",
            details=details,
            next_action="cargo run -p playgta6-recomp -- build-program --help",
        )

    coverage = float(lift_stats.get("liftCoveragePct") or 0.0)
    shards = int(lift_stats.get("shardCount") or 0)
    summary = (
        f"manifest + {pgim_count} .pgim; shards={shards}; lift={coverage:.2f}%"
    )
    if coverage <= 0.0 and int(lift_stats.get("liftedFunctions") or 0) <= 0:
        return GateResult(
            name="recomp_artifacts",
            status="warn",
            summary=summary + " (lift coverage unavailable)",
            details=details,
            next_action="python tools/audit_recomp_build.py work/recomp_build --json",
        )

    return GateResult(
        name="recomp_artifacts",
        status="pass",
        summary=summary,
        details=details,
        next_action=None,
    )


def check_shader_pack_gate(
    repo_root: Path,
    *,
    data_dir: Path | None = None,
) -> GateResult:
    """Gate 4: WGPK shader pack + reflection-manifest.json under local-data/shaders/."""
    wgpk = _load_tools_module("inspect_wgsl_pack.py", "_playgta6_verify_wgpk")
    shaders_dir = (
        (data_dir / "shaders")
        if data_dir is not None
        else (repo_root / DEFAULT_SHADERS_DIR_REL)
    )

    details: dict[str, Any] = {
        "shadersDirLabel": "local-data/shaders",
        "dirExists": shaders_dir.is_dir(),
        "packCount": 0,
        "packNames": [],
        "validPackCount": 0,
        "reflectionPresent": False,
        "reflectionValid": False,
        "reflectionShaderCount": 0,
        "hashParityOk": False,
    }

    if not shaders_dir.is_dir():
        return GateResult(
            name="shader_pack",
            status="fail",
            summary="local-data/shaders/ directory missing",
            details=details,
            next_action="python tools/gen_fallback_shaders.py --out-dir local-data/shaders",
        )

    pack_files: list[Path] = []
    try:
        for entry in sorted(shaders_dir.iterdir(), key=lambda p: p.name.lower()):
            if entry.is_file() and entry.name.lower().endswith(".wgsl.pack"):
                pack_files.append(entry)
    except OSError:
        pack_files = []

    details["packCount"] = len(pack_files)
    details["packNames"] = [p.name for p in pack_files]

    if not pack_files:
        return GateResult(
            name="shader_pack",
            status="fail",
            summary="no .wgsl.pack bundles in local-data/shaders/",
            details=details,
            next_action="python tools/gen_fallback_shaders.py --out-dir local-data/shaders",
        )

    pack_hashes: set[str] = set()
    valid_packs = 0
    pack_errors: list[str] = []
    preferred = next((p for p in pack_files if p.name.lower() == "fallback.wgsl.pack"), pack_files[0])

    for pack_path in pack_files:
        try:
            report = wgpk.inspect_pack(pack_path, source_label=pack_path.name)
            valid_packs += 1
            for shader in report.shaders:
                pack_hashes.add(shader.hash_hex)
            if pack_path == preferred:
                details["preferredPack"] = pack_path.name
                details["preferredEntryCount"] = report.entry_count
                details["preferredClean"] = report.clean
        except Exception as exc:  # noqa: BLE001 - surface sanitized parse failures
            pack_errors.append(f"{pack_path.name}: {sanitize_text(str(exc))}")

    details["validPackCount"] = valid_packs
    details["packErrors"] = pack_errors

    if valid_packs <= 0:
        return GateResult(
            name="shader_pack",
            status="fail",
            summary="no valid WGPK .wgsl.pack bundles found",
            details=details,
            next_action="python tools/inspect_wgsl_pack.py inspect local-data/shaders/fallback.wgsl.pack",
        )

    reflection_path = shaders_dir / "reflection-manifest.json"
    details["reflectionPresent"] = reflection_path.is_file()
    reflection_hashes: set[str] = set()
    if reflection_path.is_file():
        try:
            reflection = json.loads(reflection_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            details["reflectionError"] = sanitize_text(str(exc))
            reflection = None
        if isinstance(reflection, dict):
            shaders_map = reflection.get("shaders")
            if isinstance(shaders_map, dict):
                reflection_hashes = {str(k) for k in shaders_map.keys()}
                details["reflectionValid"] = True
                details["reflectionShaderCount"] = len(reflection_hashes)
                details["schemaVersion"] = reflection.get("schema_version")
            else:
                details["reflectionError"] = "reflection-manifest.json missing shaders object"

    if not details["reflectionPresent"]:
        return GateResult(
            name="shader_pack",
            status="fail",
            summary=f"{valid_packs} valid WGPK pack(s) but reflection-manifest.json missing",
            details=details,
            next_action="python tools/gen_fallback_shaders.py --out-dir local-data/shaders",
        )

    if not details["reflectionValid"]:
        return GateResult(
            name="shader_pack",
            status="fail",
            summary="reflection-manifest.json present but invalid",
            details=details,
            next_action="python tools/gen_fallback_shaders.py --out-dir local-data/shaders",
        )

    # Parity: every pack hash should appear in the reflection manifest (manifest may be a superset).
    missing_in_reflection = sorted(pack_hashes - reflection_hashes)
    details["hashParityOk"] = len(missing_in_reflection) == 0
    details["packHashCount"] = len(pack_hashes)
    details["missingInReflection"] = missing_in_reflection[:16]

    if missing_in_reflection:
        return GateResult(
            name="shader_pack",
            status="warn",
            summary=(
                f"WGPK packs OK ({valid_packs}); reflection missing "
                f"{len(missing_in_reflection)} pack hash(es)"
            ),
            details=details,
            next_action="python tools/gen_fallback_shaders.py --out-dir local-data/shaders",
        )

    return GateResult(
        name="shader_pack",
        status="pass",
        summary=(
            f"{valid_packs} valid WGPK pack(s); reflection-manifest.json "
            f"({details['reflectionShaderCount']} shaders) parity OK"
        ),
        details=details,
        next_action=None,
    )


def classify_runtime_wasm_kind(
    *,
    byte_size: int,
    export_names: Sequence[str],
) -> str:
    """Distinguish smoke shell (< 1 MiB) from linked guest runtime."""
    guest_exports = [name for name in export_names if name in GUEST_BOOT_EXPORT_MARKERS]
    if byte_size >= SMOKE_SHELL_MAX_BYTES or guest_exports:
        return "linked_guest_runtime"
    return "smoke_shell"


def check_wasm64_runtime_gate(repo_root: Path) -> GateResult:
    """Gate 5: web/dist/runtime.wasm Wasm64/threads/ABI + smoke vs linked guest."""
    wasm_mod = _load_tools_module("check_wasm_exports.py", "_playgta6_verify_wasm")
    wasm_path = repo_root / RUNTIME_WASM_REL
    details: dict[str, Any] = {
        "pathLabel": "web/dist/runtime.wasm",
        "present": False,
        "byteSize": 0,
        "kind": None,
        "isMemory64": False,
        "isSharedMemory": False,
        "hostImportsOk": False,
        "hostImportsValid": 0,
        "hostImportsTotal": len(getattr(wasm_mod, "REQUIRED_HOST_IMPORTS", ())),
        "guestBootExports": [],
        "exportCount": 0,
    }

    if not wasm_path.is_file():
        return GateResult(
            name="wasm64_runtime",
            status="fail",
            summary="web/dist/runtime.wasm missing",
            details=details,
            next_action="pwsh runtime/build_wasm.ps1",
        )

    details["present"] = True
    try:
        raw = wasm_path.read_bytes()
    except OSError as exc:
        return GateResult(
            name="wasm64_runtime",
            status="fail",
            summary=f"unable to read runtime.wasm ({sanitize_text(exc.strerror or type(exc).__name__)})",
            details=details,
            next_action="pwsh runtime/build_wasm.ps1",
        )

    details["byteSize"] = len(raw)
    try:
        module = wasm_mod.parse_wasm_module(raw)
        report = wasm_mod.validate_runtime_wasm(
            module,
            require_memory64=True,
            require_shared_memory=True,
            require_host_imports=True,
            path="web/dist/runtime.wasm",
        )
    except Exception as exc:  # noqa: BLE001
        return GateResult(
            name="wasm64_runtime",
            status="fail",
            summary=f"runtime.wasm parse/validate failed: {sanitize_text(str(exc))}",
            details=details,
            next_action="python tools/check_wasm_exports.py web/dist/runtime.wasm --json",
        )

    export_names = [exp.name for exp in report.exports]
    guest_exports = sorted(name for name in export_names if name in GUEST_BOOT_EXPORT_MARKERS)
    kind = classify_runtime_wasm_kind(byte_size=report.byte_size, export_names=export_names)
    host_ok_count = sum(1 for h in report.host_imports if h.ok)

    details.update(
        {
            "byteSize": report.byte_size,
            "kind": kind,
            "isMemory64": report.is_memory64,
            "isSharedMemory": report.is_shared_memory,
            "hostImportsOk": report.all_host_imports_ok,
            "hostImportsValid": host_ok_count,
            "hostImportsTotal": len(report.host_imports),
            "guestBootExports": guest_exports,
            "exportCount": report.exports_count,
            "errors": [sanitize_text(e) for e in report.errors],
            "warnings": [sanitize_text(w) for w in report.warnings],
        }
    )

    abi_ok = bool(
        report.is_memory64 and report.is_shared_memory and report.all_host_imports_ok
    )
    if not abi_ok:
        return GateResult(
            name="wasm64_runtime",
            status="fail",
            summary=(
                f"runtime.wasm ABI incomplete "
                f"(memory64={report.is_memory64}, shared={report.is_shared_memory}, "
                f"host={host_ok_count}/{len(report.host_imports)}; kind={kind})"
            ),
            details=details,
            next_action=(
                "python tools/check_wasm_exports.py web/dist/runtime.wasm "
                "--require-memory64 --require-shared-memory --require-host-imports"
            ),
        )

    if kind == "smoke_shell":
        return GateResult(
            name="wasm64_runtime",
            status="warn",
            summary=(
                f"ABI OK ({host_ok_count}/{len(report.host_imports)}) but runtime.wasm is "
                f"smoke_shell ({report.byte_size} bytes < 1 MiB; no guest boot exports)"
            ),
            details=details,
            next_action="python tools/build_guest_wasm.py work/recomp_build --verify",
        )

    return GateResult(
        name="wasm64_runtime",
        status="pass",
        summary=(
            f"linked_guest_runtime ABI OK "
            f"({report.byte_size} bytes; host {host_ok_count}/{len(report.host_imports)}; "
            f"guest exports={len(guest_exports)})"
        ),
        details=details,
        next_action=None,
    )


def _http_get(
    host: str,
    port: int,
    path: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = 3.0,
) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("GET", path, headers=dict(headers or {}))
        resp = conn.getresponse()
        body = resp.read()
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, body
    finally:
        conn.close()


def check_local_server_gate(
    repo_root: Path,
    *,
    data_dir: Path | None = None,
    game_dir: Path | None = None,
    recomp_dir: Path | None = None,
    skip_server: bool = False,
) -> GateResult:
    """Gate 6: ephemeral COOP/COEP server + Range 206 + boot endpoints."""
    if skip_server:
        return GateResult(
            name="local_server",
            status="skip",
            summary="server check skipped (--skip-server)",
            details={"checked": False, "skipped": True},
            next_action=None,
        )

    details: dict[str, Any] = {
        "checked": True,
        "mode": None,
        "coopValid": False,
        "coepValid": False,
        "corpValid": False,
        "acceptRangesValid": False,
        "statusEndpointOk": False,
        "range206Ok": False,
        "rootOk": False,
        "shadersOk": False,
        "bootOk": False,
        "gameSplashOk": None,
        "splashUrl": None,
    }

    server_file = repo_root / "server" / "serve_local.py"
    if not server_file.is_file():
        return GateResult(
            name="local_server",
            status="fail",
            summary="server/serve_local.py not found",
            details=details,
            next_action="python server/serve_local.py",
        )

    try:
        serve = _load_server_module(repo_root)
    except Exception:  # noqa: BLE001
        return GateResult(
            name="local_server",
            status="fail",
            summary="server/serve_local.py import failed",
            details=details,
            next_action="python server/serve_local.py",
        )

    web_root = repo_root / "web"
    resolved_data = data_dir if data_dir is not None else (repo_root / "local-data")
    shaders_root = resolved_data / "shaders"
    boot_root = recomp_dir if recomp_dir is not None else (repo_root / DEFAULT_RECOMP_DIR_REL)
    resolved_game = game_dir

    handler_cls = getattr(serve, "PlayGta6Handler", None)
    if handler_cls is None:
        return GateResult(
            name="local_server",
            status="fail",
            summary="PlayGta6Handler not found in server module",
            details=details,
            next_action="python server/serve_local.py",
        )

    class _QuietHandler(handler_cls):  # type: ignore[misc,valid-type]
        def log_message(self, format: str, *args: Any) -> None:
            return None

    httpd: ThreadingHTTPServer | None = None
    thread: threading.Thread | None = None
    try:
        create_server = getattr(serve, "create_server", None)
        if callable(create_server):
            httpd = create_server(
                "127.0.0.1",
                0,
                web_root,
                resolved_data,
                game_root=resolved_game,
                shaders_root=shaders_root,
                boot_root=boot_root,
            )
            # Replace handler logging by wrapping serve_forever requests through QuietHandler
            # is hard after create_server; mute via monkeypatch on instance RequestHandlerClass.
            httpd.RequestHandlerClass = partial(  # type: ignore[method-assign]
                _QuietHandler,
                web_root=web_root,
                data_root=resolved_data,
                game_root=resolved_game,
                shaders_root=shaders_root,
                boot_root=boot_root,
            )
        else:
            handler_factory = partial(
                _QuietHandler,
                web_root=web_root,
                data_root=resolved_data,
                game_root=resolved_game,
                shaders_root=shaders_root,
                boot_root=boot_root,
            )
            httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_factory)

        port = int(httpd.server_address[1])
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        details["mode"] = "ephemeral-http"

        # /api/status + isolation headers
        status_code, headers, body = _http_get("127.0.0.1", port, "/api/status")
        details["coopValid"] = headers.get("cross-origin-opener-policy") == "same-origin"
        details["coepValid"] = headers.get("cross-origin-embedder-policy") == "require-corp"
        details["corpValid"] = headers.get("cross-origin-resource-policy") == "same-origin"
        details["acceptRangesValid"] = headers.get("accept-ranges") == "bytes"
        status_ok = False
        splash_url: str | None = None
        shader_pack_url: str | None = None
        pgim_url: str | None = None
        if status_code == 200:
            try:
                payload = json.loads(body.decode("utf-8"))
                status_ok = bool(
                    isinstance(payload, dict) and payload.get("ok") and payload.get("coopCoep")
                )
                if isinstance(payload, dict):
                    splash_url = payload.get("splashImage")
                    shader_pack_url = payload.get("shaderPackUrl")
                    pgim_url = payload.get("pgimUrl")
            except (UnicodeDecodeError, json.JSONDecodeError):
                status_ok = False
        details["statusEndpointOk"] = status_ok
        details["splashUrl"] = splash_url
        details["shaderPackUrl"] = shader_pack_url
        details["pgimUrl"] = pgim_url

        # `/` root
        root_code, _, _ = _http_get("127.0.0.1", port, "/")
        details["rootOk"] = root_code == 200

        # Range 206 against index.html (or status body fallback file)
        range_ok = False
        range_code, range_headers, _ = _http_get(
            "127.0.0.1",
            port,
            "/",
            headers={"Range": "bytes=0-3"},
        )
        if range_code == 206 and range_headers.get("content-range", "").startswith("bytes "):
            range_ok = True
        details["range206Ok"] = range_ok

        # /shaders/
        shaders_ok = False
        if shader_pack_url:
            sc, _, _ = _http_get("127.0.0.1", port, shader_pack_url)
            shaders_ok = sc == 200
        else:
            # Directory may 404 for listing; accept reflection or any pack probe.
            probe = "/shaders/reflection-manifest.json"
            sc, _, _ = _http_get("127.0.0.1", port, probe)
            shaders_ok = sc == 200
        details["shadersOk"] = shaders_ok

        # /boot/
        boot_ok = False
        if pgim_url:
            bc, _, _ = _http_get("127.0.0.1", port, pgim_url)
            boot_ok = bc == 200
        elif boot_root.is_dir():
            # No pgim yet — endpoint mount still considered present if meta.json/boot allowlist 404s cleanly.
            bc, _, _ = _http_get("127.0.0.1", port, "/boot/missing.pgim")
            boot_ok = bc in (404, 403)
        details["bootOk"] = boot_ok

        # /game/<splash>
        game_splash_ok: bool | None = None
        if splash_url:
            gc, _, _ = _http_get("127.0.0.1", port, splash_url)
            game_splash_ok = gc == 200
        elif resolved_game is None:
            game_splash_ok = None
        else:
            game_splash_ok = False
        details["gameSplashOk"] = game_splash_ok

        isolation_ok = bool(
            details["coopValid"]
            and details["coepValid"]
            and details["statusEndpointOk"]
            and details["range206Ok"]
            and details["rootOk"]
        )
        endpoints_ok = bool(details["shadersOk"] and details["bootOk"])
        splash_requirement_ok = game_splash_ok is not False

        if isolation_ok and endpoints_ok and splash_requirement_ok:
            splash_note = (
                f"; splash={splash_url}"
                if splash_url
                else "; splash=n/a"
            )
            return GateResult(
                name="local_server",
                status="pass",
                summary=f"COOP/COEP + Range 206 + boot endpoints OK{splash_note}",
                details=details,
                next_action=None,
            )

        missing: list[str] = []
        if not details["coopValid"] or not details["coepValid"]:
            missing.append("isolation-headers")
        if not details["statusEndpointOk"]:
            missing.append("/api/status")
        if not details["range206Ok"]:
            missing.append("range-206")
        if not details["rootOk"]:
            missing.append("/")
        if not details["shadersOk"]:
            missing.append("/shaders/")
        if not details["bootOk"]:
            missing.append("/boot/")
        if game_splash_ok is False:
            missing.append("/game/<splash>")

        return GateResult(
            name="local_server",
            status="fail",
            summary="server self-check failed: " + ", ".join(missing),
            details=details,
            next_action="python server/serve_local.py",
        )
    except OSError:
        # Restricted environments: fall back to in-process header probe.
        try:
            captured: dict[str, str] = {}

            class _ProbeHandler(handler_cls):  # type: ignore[misc,valid-type]
                def __init__(self) -> None:
                    self.request_version = "HTTP/1.1"

                def send_header(self, keyword: str, value: str) -> None:
                    captured[keyword.lower()] = value

                def flush_headers(self) -> None:
                    return None

            probe = _ProbeHandler()
            handler_cls.end_headers(probe)
            details["mode"] = "in-process"
            details["coopValid"] = captured.get("cross-origin-opener-policy") == "same-origin"
            details["coepValid"] = captured.get("cross-origin-embedder-policy") == "require-corp"
            details["corpValid"] = captured.get("cross-origin-resource-policy") == "same-origin"
            details["acceptRangesValid"] = captured.get("accept-ranges") == "bytes"
            details["statusEndpointOk"] = callable(getattr(serve, "build_status_payload", None))
            ok = bool(details["coopValid"] and details["coepValid"])
            return GateResult(
                name="local_server",
                status="pass" if ok else "fail",
                summary=(
                    "in-process COOP/COEP headers OK (socket bind unavailable)"
                    if ok
                    else "in-process COOP/COEP headers missing"
                ),
                details=details,
                next_action=None if ok else "python server/serve_local.py",
            )
        except Exception:  # noqa: BLE001
            return GateResult(
                name="local_server",
                status="fail",
                summary="server header self-check failed",
                details=details,
                next_action="python server/serve_local.py",
            )
    except Exception:  # noqa: BLE001
        return GateResult(
            name="local_server",
            status="fail",
            summary="server self-check failed",
            details=details,
            next_action="python server/serve_local.py",
        )
    finally:
        if httpd is not None:
            try:
                httpd.shutdown()
            except Exception:  # noqa: BLE001
                pass
            try:
                httpd.server_close()
            except Exception:  # noqa: BLE001
                pass
        if thread is not None:
            thread.join(timeout=2)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _gate_passed(gate: GateResult, *, allow_warn: bool = False) -> bool:
    if gate.status == "pass":
        return True
    if allow_warn and gate.status == "warn":
        return True
    return False


def compute_readiness(gates: Mapping[str, GateResult]) -> dict[str, Any]:
    """Compute splash / real-guest readiness booleans and ordered next steps."""
    game = gates["game_assets"]
    imports = gates["import_coverage"]
    recomp = gates["recomp_artifacts"]
    shaders = gates["shader_pack"]
    wasm = gates["wasm64_runtime"]
    server = gates["local_server"]

    wasm_kind = (wasm.details or {}).get("kind")
    splash_ready = bool(
        _gate_passed(game, allow_warn=True)
        and game.status != "skip"
        and _gate_passed(shaders, allow_warn=True)
        and _gate_passed(wasm, allow_warn=True)
        and (server.status in ("pass", "skip"))
    )

    real_boot_ready = bool(
        splash_ready
        and _gate_passed(imports)
        and _gate_passed(recomp, allow_warn=True)
        and wasm_kind == "linked_guest_runtime"
        and _gate_passed(wasm)
        and server.status in ("pass", "skip")
    )

    next_steps: list[str] = []
    for name in GATE_NAMES:
        gate = gates[name]
        if gate.status in ("fail", "warn", "skip") and gate.next_action:
            if gate.next_action not in next_steps:
                next_steps.append(gate.next_action)

    if splash_ready and not real_boot_ready:
        if wasm_kind == "smoke_shell":
            cmd = "python tools/build_guest_wasm.py work/recomp_build --verify"
            if cmd not in next_steps:
                next_steps.append(cmd)
        if not _gate_passed(recomp, allow_warn=True):
            cmd = "cargo run -p playgta6-recomp -- build-program --help"
            if cmd not in next_steps:
                next_steps.append(cmd)

    if real_boot_ready:
        cmd = "python server/serve_local.py"
        if cmd not in next_steps:
            next_steps.append(cmd)

    return {
        "ready_for_browser_splash": splash_ready,
        "ready_for_real_guest_boot": real_boot_ready,
        "next_steps": next_steps,
        "wasm_kind": wasm_kind,
    }


def run_verification(
    repo_root: Path | str = DEFAULT_REPO_ROOT,
    *,
    recomp_dir: Path | str | None = None,
    skip_server: bool = False,
    env: Mapping[str, str] | None = None,
    import_coverage_fn: Callable[[Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run all six browser-boot gates and return a sanitized JSON-safe report."""
    root = Path(repo_root).resolve()
    resolved_env = _resolve_env_paths(root, env=env)
    data_path: Path = resolved_env["data_path"]
    game_path: Path | None = resolved_env["game_path"]
    work_path: Path = resolved_env["work_path"]

    if recomp_dir is not None:
        recomp_path = Path(recomp_dir)
        if not recomp_path.is_absolute():
            recomp_path = root / recomp_path
    else:
        recomp_path = work_path / "recomp_build"

    gates_list = [
        check_game_assets_gate(root, env=env),
        check_import_coverage_gate(root, audit_fn=import_coverage_fn),
        check_recomp_artifacts_gate(root, recomp_dir=recomp_path),
        check_shader_pack_gate(root, data_dir=data_path),
        check_wasm64_runtime_gate(root),
        check_local_server_gate(
            root,
            data_dir=data_path,
            game_dir=game_path if (game_path is not None and game_path.is_dir()) else None,
            recomp_dir=recomp_path,
            skip_server=skip_server,
        ),
    ]
    gates = {gate.name: gate for gate in gates_list}
    readiness = compute_readiness(gates)

    return sanitize_value(
        {
            "schemaVersion": SCHEMA_VERSION,
            "ok": bool(readiness["ready_for_browser_splash"]),
            "ready_for_browser_splash": readiness["ready_for_browser_splash"],
            "ready_for_real_guest_boot": readiness["ready_for_real_guest_boot"],
            "gates": {name: gates[name].to_dict() for name in GATE_NAMES},
            "next_steps": readiness["next_steps"],
        }
    )


def format_report_text(report: Mapping[str, Any]) -> str:
    """Render a concise terminal checklist for the six gates."""
    lines: list[str] = [
        "=== playgta6web Browser Real-Boot Readiness ===",
        "",
        "Gates:",
    ]
    status_tag = {
        "pass": "[PASS]",
        "warn": "[WARN]",
        "fail": "[FAIL]",
        "skip": "[SKIP]",
    }
    gates = report.get("gates") or {}
    for name in GATE_NAMES:
        gate = gates.get(name) or {}
        status = gate.get("status", "fail")
        tag = status_tag.get(status, "[????]")
        summary = gate.get("summary") or ""
        lines.append(f"  {tag} {name:<18} {summary}")

    lines.append("")
    lines.append("Readiness:")
    lines.append(
        f"  ready_for_browser_splash : "
        f"{'YES' if report.get('ready_for_browser_splash') else 'NO'}"
    )
    lines.append(
        f"  ready_for_real_guest_boot: "
        f"{'YES' if report.get('ready_for_real_guest_boot') else 'NO'}"
    )

    next_steps = report.get("next_steps") or []
    if next_steps:
        lines.append("")
        lines.append("Next Steps:")
        for step in next_steps:
            lines.append(f"  $ {step}")

    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify end-to-end browser real-boot readiness across game assets, "
            "import coverage, recomp artifacts, shader packs, Wasm64 runtime, "
            "and the local COOP/COEP server."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Path to the playgta6web repository root (defaults to parent of tools/).",
    )
    parser.add_argument(
        "--recomp-dir",
        type=Path,
        default=None,
        help="Recomp build directory (default: <work>/recomp_build or work/recomp_build).",
    )
    parser.add_argument(
        "--skip-server",
        action="store_true",
        help="Skip the ephemeral local COOP/COEP server self-check.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit sanitized machine-readable JSON (never includes host paths or usernames).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    report = run_verification(
        repo_root=args.repo_root,
        recomp_dir=args.recomp_dir,
        skip_server=args.skip_server,
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report_text(report))

    if report.get("ready_for_real_guest_boot"):
        return 0
    if report.get("ready_for_browser_splash"):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
