#!/usr/bin/env python3
"""Safe local browser real-boot artifact staging CLI (Track 5 / W51).

Stages allowlisted local recompilation boot artifacts (`program-manifest.json`,
`build-manifest.json`, nested `*.pgim`) from `--program-dir` into `--boot-dir`
(served at `/boot/` by `server/serve_local.py`), and validates + atomically
stages `runtime.wasm` + `runtime.js` from `--wasm-dir` into `--web-dist`.

Safety & privacy invariants:
- Refuses to write boot artifacts into tracked repository directories
  (`--boot-dir` must be under `work/`, `local-data/`, or outside the repo).
- Refuses to write Wasm64 runtime artifacts into tracked repository directories
  (`--web-dist` must be `web/dist` or outside the repo).
- Skips any symlinks escaping `--program-dir` / `--wasm-dir` and never copies
  non-allowlisted build artifacts (`.ll`, `.o`, `.cpp`, `.c`, `.h`, `.rsp`, etc.).
- Validates `runtime.wasm` via `tools/check_wasm_exports.py` (`memory64`,
  shared memory, 5/5 `env.host_*` ABI imports) BEFORE atomically replacing
  files in `--web-dist` (`os.replace` via same-directory temp file).
- Strictly sanitized JSON and text output: zero Windows/POSIX/UNC host paths,
  zero usernames, and zero raw bytes.
- Pure Python 3.12 standard library only.

Usage:
    python tools/stage_browser_boot.py --program-dir work/recomp_program
        [--wasm-dir work/wasm_guest_dist] [--boot-dir work/recomp_build]
        [--web-dist web/dist] [--dry-run] [--json]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROGRAM_DIR_REL = Path("work") / "recomp_program"
DEFAULT_BOOT_DIR_REL = Path("work") / "recomp_build"
DEFAULT_WEB_DIST_REL = Path("web") / "dist"

RUNTIME_WASM_NAME = "runtime.wasm"
RUNTIME_JS_NAME = "runtime.js"

ALLOWLISTED_MANIFEST_NAMES: frozenset[str] = frozenset(
    {
        "program-manifest.json",
        "build-manifest.json",
    }
)
ALLOWLISTED_BOOT_SUFFIXES: tuple[str, ...] = (".pgim",)
COPY_CHUNK_BYTES = 256 * 1024

_UNC_BS_PATH_RE = re.compile(r"\\\\[^\s\"'<>|,;()\[\]]+")
_UNC_FS_PATH_RE = re.compile(r"(?<!:)//[^\s\"'<>|,;()\[\]]+")
_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"'<>|,;()\[\]]*")
_POSIX_PATH_RE = re.compile(
    r"/(?:Users|home|tmp|var|opt|root|mnt|private|etc|usr|Library|Volumes)/[^\s\"'<>|,;()\[\]]*"
)


class StageBrowserBootError(Exception):
    """Raised for internal staging errors caught and reported cleanly."""


# ---------------------------------------------------------------------------
# Cross-platform sanitization
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
    """Strip Windows/POSIX/UNC host paths (both `\\` and `/`) and active usernames."""
    if raw is None:
        return ""
    text = str(raw)
    # Strip Windows UNC backslash paths before normalizing separators so
    # `\\server\share\user\file` never survives as `//server/share/...`.
    text = _UNC_BS_PATH_RE.sub("<path>", text)
    text = text.replace("\\", "/")
    text = _WINDOWS_PATH_RE.sub("<path>", text)
    text = _UNC_FS_PATH_RE.sub("<path>", text)
    text = _POSIX_PATH_RE.sub("<path>", text)
    for username in _current_usernames():
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(username)}(?![A-Za-z0-9_])",
            re.IGNORECASE,
        )
        text = pattern.sub("<user>", text)
    return text


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


def repo_relative_label(path: Path | None, repo_root: Path) -> str | None:
    """Return a repo-relative POSIX label or a sanitized basename, never a host path."""
    if path is None:
        return None
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
        posix = rel.as_posix()
        return sanitize_text("." if not posix else posix)
    except (OSError, ValueError):
        return basename_label(path)


def sanitize_value(value: Any) -> Any:
    """Recursively sanitize strings/paths and replace raw bytes with size placeholders."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<bytes:{len(value)}>"
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, Path):
        return basename_label(value)
    if isinstance(value, Mapping):
        return {sanitize_text(str(k)): sanitize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_value(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Sibling tool loader (tools/check_wasm_exports.py)
# ---------------------------------------------------------------------------


def _load_tools_module(module_filename: str, module_name: str) -> Any:
    """Load a sibling tools/*.py module, reusing an already-imported copy when present."""
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


# ---------------------------------------------------------------------------
# Tracked-directory safety guards
# ---------------------------------------------------------------------------


def _resolve_against_repo(raw_path: Path | str, repo_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        return repo_root / path
    return path


def validate_boot_dir_safety(boot_dir: Path, repo_root: Path) -> tuple[bool, str | None]:
    """Ensure `--boot-dir` is under `work/`, `local-data/`, or outside the repository."""
    resolved_repo = repo_root.resolve()
    resolved_boot = boot_dir.resolve()
    if not resolved_boot.is_relative_to(resolved_repo):
        return True, None

    # Compare against unresolved repo children so a symlink inside `work/` or
    # `local-data/` pointing into a tracked directory cannot bypass the guard.
    allowed_roots = (
        resolved_repo / "work",
        resolved_repo / "local-data",
    )
    if any(resolved_boot.is_relative_to(allowed) for allowed in allowed_roots):
        return True, None

    label = repo_relative_label(resolved_boot, resolved_repo) or "<repo>"
    return (
        False,
        f"refusing to write boot artifacts into tracked repository directory '{label}' "
        "(must be under work/, local-data/, or outside the repository)",
    )


def validate_web_dist_safety(web_dist: Path, repo_root: Path) -> tuple[bool, str | None]:
    """Ensure `--web-dist` is `web/dist` (gitignored) or outside the repository."""
    resolved_repo = repo_root.resolve()
    resolved_dist = web_dist.resolve()
    if not resolved_dist.is_relative_to(resolved_repo):
        return True, None

    allowed_dist = resolved_repo / "web" / "dist"
    if resolved_dist.is_relative_to(allowed_dist):
        return True, None

    label = repo_relative_label(resolved_dist, resolved_repo) or "<repo>"
    return (
        False,
        f"refusing to write Wasm64 runtime into tracked repository directory '{label}' "
        "(must be web/dist or outside the repository)",
    )


# ---------------------------------------------------------------------------
# Allowlisted boot artifact discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootArtifact:
    """One allowlisted boot artifact discovered under `--program-dir`."""

    rel_posix: str
    source_path: Path
    kind: str  # "program_manifest" | "build_manifest" | "pgim"


@dataclass
class BootDiscovery:
    """Result of scanning `--program-dir` for allowlisted boot artifacts."""

    artifacts: list[BootArtifact] = field(default_factory=list)
    skipped_escaping_symlinks: int = 0
    skipped_non_allowlisted: int = 0


def is_allowlisted_boot_file(filename: str) -> bool:
    """Return True only for program/build manifests and `.pgim` guest images."""
    lowered = filename.lower()
    if lowered in ALLOWLISTED_MANIFEST_NAMES:
        return True
    return any(lowered.endswith(suffix) for suffix in ALLOWLISTED_BOOT_SUFFIXES)


def _boot_artifact_kind(filename: str) -> str:
    lowered = filename.lower()
    if lowered == "program-manifest.json":
        return "program_manifest"
    if lowered == "build-manifest.json":
        return "build_manifest"
    return "pgim"


def discover_boot_artifacts(program_dir: Path) -> BootDiscovery:
    """Recursively discover allowlisted manifests and `*.pgim` files inside `program_dir`.

    Skips any symlinks escaping `program_dir` (file or directory symlinks) as well
    as non-allowlisted build artifacts such as `.ll`, `.o`, `.cpp`, `.c`, `.h`,
    `.rsp`, or arbitrary `.json` files.
    """
    resolved_root = program_dir.resolve()
    discovery = BootDiscovery()
    seen_rel: set[str] = set()

    for entry in program_dir.rglob("*"):
        if entry.is_symlink():
            try:
                resolved = entry.resolve(strict=True)
            except OSError:
                discovery.skipped_escaping_symlinks += 1
                continue
            if not resolved.is_relative_to(resolved_root):
                discovery.skipped_escaping_symlinks += 1
                continue
            if resolved.is_dir():
                continue
            if not resolved.is_file():
                continue
        else:
            if not entry.is_file():
                continue
            try:
                resolved = entry.resolve(strict=True)
            except OSError:
                continue
            if not resolved.is_relative_to(resolved_root):
                discovery.skipped_escaping_symlinks += 1
                continue

        if not is_allowlisted_boot_file(entry.name):
            discovery.skipped_non_allowlisted += 1
            continue

        try:
            rel_posix = entry.relative_to(program_dir).as_posix()
        except ValueError:
            continue
        if not rel_posix or ".." in rel_posix.split("/"):
            continue
        if rel_posix in seen_rel:
            continue
        seen_rel.add(rel_posix)
        discovery.artifacts.append(
            BootArtifact(
                rel_posix=rel_posix,
                source_path=resolved,
                kind=_boot_artifact_kind(entry.name),
            )
        )

    discovery.artifacts.sort(key=lambda item: (item.rel_posix.lower(), item.rel_posix))
    return discovery


# ---------------------------------------------------------------------------
# Content comparison & atomic file staging
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 digest of a file using streaming I/O."""
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


@dataclass(frozen=True)
class FileStagePlan:
    """Incremental copy decision for one source -> destination file pair."""

    needs_copy: bool
    reason: str  # "missing" | "replace_symlink" | "size_changed" | "sha256_changed" | "unchanged"
    size_bytes: int
    sha256: str


def plan_file_stage(src: Path, dst: Path) -> FileStagePlan:
    """Determine whether `src` must be copied to `dst` (missing, size, or SHA-256 changed)."""
    src_stat = src.stat()
    src_size = int(src_stat.st_size)
    src_digest = sha256_file(src)

    if dst.is_symlink():
        return FileStagePlan(
            needs_copy=True,
            reason="replace_symlink",
            size_bytes=src_size,
            sha256=src_digest,
        )

    if not dst.exists():
        return FileStagePlan(
            needs_copy=True,
            reason="missing",
            size_bytes=src_size,
            sha256=src_digest,
        )

    if not dst.is_file():
        raise StageBrowserBootError(
            f"destination path is not a regular file: {basename_label(dst)}"
        )

    try:
        if src.resolve() == dst.resolve():
            return FileStagePlan(
                needs_copy=False,
                reason="unchanged",
                size_bytes=src_size,
                sha256=src_digest,
            )
    except OSError:
        pass

    dst_stat = dst.stat()
    if int(dst_stat.st_size) != src_size:
        return FileStagePlan(
            needs_copy=True,
            reason="size_changed",
            size_bytes=src_size,
            sha256=src_digest,
        )

    dst_digest = sha256_file(dst)
    if dst_digest != src_digest:
        return FileStagePlan(
            needs_copy=True,
            reason="sha256_changed",
            size_bytes=src_size,
            sha256=src_digest,
        )

    return FileStagePlan(
        needs_copy=False,
        reason="unchanged",
        size_bytes=src_size,
        sha256=src_digest,
    )


def _write_temp_copy(src: Path, dst_dir: Path, final_name: str) -> Path:
    """Stream-copy `src` into a new temp file in `dst_dir`, preserving timestamps."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    src_stat = src.stat()
    fd, tmp_str = tempfile.mkstemp(
        prefix=f".{final_name}.",
        suffix=".tmp",
        dir=dst_dir,
    )
    tmp_path = Path(tmp_str)
    try:
        with os.fdopen(fd, "wb") as dst_fh, src.open("rb") as src_fh:
            while True:
                chunk = src_fh.read(COPY_CHUNK_BYTES)
                if not chunk:
                    break
                dst_fh.write(chunk)
            dst_fh.flush()
            try:
                os.fsync(dst_fh.fileno())
            except OSError:
                pass
        try:
            os.utime(tmp_path, ns=(src_stat.st_atime_ns, src_stat.st_mtime_ns))
        except OSError:
            pass
        return tmp_path
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_copy_file(src: Path, dst: Path) -> None:
    """Atomically copy `src` to `dst` via a same-directory temp file and `os.replace`."""
    tmp_path = _write_temp_copy(src, dst.parent, dst.name)
    try:
        os.replace(tmp_path, dst)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_copy_files(pairs: Sequence[tuple[Path, Path]]) -> None:
    """Prepare all temp files first, then atomically replace each destination."""
    staged: list[tuple[Path, Path]] = []
    replaced: set[Path] = set()
    try:
        for src, dst in pairs:
            tmp_path = _write_temp_copy(src, dst.parent, dst.name)
            staged.append((tmp_path, dst))
        for tmp_path, dst in staged:
            os.replace(tmp_path, dst)
            replaced.add(tmp_path)
    except Exception:
        for tmp_path, _ in staged:
            if tmp_path not in replaced:
                tmp_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Wasm64 validation via tools/check_wasm_exports.py
# ---------------------------------------------------------------------------


def validate_wasm_candidate(wasm_path: Path) -> dict[str, Any]:
    """Validate `runtime.wasm` for memory64, shared memory, and 5/5 `env.host_*` ABI imports."""
    try:
        wasm_mod = _load_tools_module("check_wasm_exports.py", "_playgta6_stage_wasm")
    except Exception as exc:  # noqa: BLE001
        return {
            "checked": True,
            "ok": False,
            "byte_size": 0,
            "is_memory64": False,
            "is_shared_memory": False,
            "all_host_imports_ok": False,
            "host_imports_valid": 0,
            "host_imports_total": 0,
            "missing_host_imports": [],
            "mismatched_host_imports": [],
            "exports_count": 0,
            "errors": [sanitize_text(f"unable to load check_wasm_exports.py: {exc}")],
            "warnings": [],
        }

    expected_total = len(getattr(wasm_mod, "REQUIRED_HOST_IMPORTS", ()))
    raw: bytes | None = None
    try:
        raw = wasm_path.read_bytes()
        module = wasm_mod.parse_wasm_module(raw)
        report = wasm_mod.validate_runtime_wasm(
            module,
            require_memory64=True,
            require_shared_memory=True,
            require_host_imports=True,
            path=RUNTIME_WASM_NAME,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "checked": True,
            "ok": False,
            "byte_size": len(raw) if raw is not None else 0,
            "is_memory64": False,
            "is_shared_memory": False,
            "all_host_imports_ok": False,
            "host_imports_valid": 0,
            "host_imports_total": expected_total,
            "missing_host_imports": [],
            "mismatched_host_imports": [],
            "exports_count": 0,
            "errors": [sanitize_text(str(exc))],
            "warnings": [],
        }

    host_valid = sum(1 for item in report.host_imports if item.ok)
    host_total = len(report.host_imports)
    abi_ok = bool(
        report.ok
        and report.is_memory64
        and report.is_shared_memory
        and report.all_host_imports_ok
    )
    return {
        "checked": True,
        "ok": abi_ok,
        "byte_size": report.byte_size,
        "wasm_version": report.wasm_version,
        "is_memory64": report.is_memory64,
        "is_shared_memory": report.is_shared_memory,
        "all_host_imports_ok": report.all_host_imports_ok,
        "host_imports_valid": host_valid,
        "host_imports_total": host_total,
        "missing_host_imports": list(report.missing_host_imports),
        "mismatched_host_imports": list(report.mismatched_host_imports),
        "exports_count": report.exports_count,
        "errors": [sanitize_text(err) for err in report.errors],
        "warnings": [sanitize_text(warn) for warn in report.warnings],
    }


# ---------------------------------------------------------------------------
# End-to-end staging pipeline
# ---------------------------------------------------------------------------


def _check_file_within_root(path: Path, root: Path) -> tuple[bool, Path | None]:
    """Verify `path` exists as a regular file whose resolved target stays inside `root`."""
    resolved_root = root.resolve()
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return False, None
    if not resolved.is_relative_to(resolved_root):
        return False, None
    if not resolved.is_file():
        return False, None
    return True, resolved


def stage_browser_boot(
    *,
    program_dir: Path | str,
    wasm_dir: Path | str | None = None,
    boot_dir: Path | str = DEFAULT_BOOT_DIR_REL,
    web_dist: Path | str = DEFAULT_WEB_DIST_REL,
    repo_root: Path | str = DEFAULT_REPO_ROOT,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Stage allowlisted boot artifacts and (optionally) validated Wasm64 runtime files."""
    resolved_repo = Path(repo_root).resolve()
    program_path = _resolve_against_repo(program_dir, resolved_repo)
    boot_path = _resolve_against_repo(boot_dir, resolved_repo)
    web_dist_path = _resolve_against_repo(web_dist, resolved_repo)
    wasm_path = (
        _resolve_against_repo(wasm_dir, resolved_repo) if wasm_dir is not None else None
    )

    errors: list[str] = []
    warnings: list[str] = []

    boot_safe, boot_safety_err = validate_boot_dir_safety(boot_path, resolved_repo)
    web_dist_safe, web_dist_safety_err = validate_web_dist_safety(
        web_dist_path, resolved_repo
    )

    boot_section: dict[str, Any] = {
        "status": "pending",
        "discovered_count": 0,
        "copied_count": 0,
        "planned_count": 0,
        "unchanged_count": 0,
        "skipped_non_allowlisted": 0,
        "skipped_escaping_symlinks": 0,
        "total_bytes": 0,
        "artifacts": [],
    }
    wasm_section: dict[str, Any] = {
        "status": "skipped",
        "validation": None,
        "copied_count": 0,
        "planned_count": 0,
        "unchanged_count": 0,
        "files": [],
    }

    if not boot_safe:
        boot_section["status"] = "refused"
        errors.append(boot_safety_err or "refused unsafe --boot-dir")
    if not web_dist_safe:
        wasm_section["status"] = "refused"
        errors.append(web_dist_safety_err or "refused unsafe --web-dist")

    # Early exit if any safety guard tripped — never touch filesystem.
    if not boot_safe or not web_dist_safe:
        return sanitize_value(
            {
                "schema_version": SCHEMA_VERSION,
                "ok": False,
                "dry_run": dry_run,
                "paths": {
                    "program_dir": repo_relative_label(program_path, resolved_repo),
                    "boot_dir": repo_relative_label(boot_path, resolved_repo),
                    "wasm_dir": repo_relative_label(wasm_path, resolved_repo),
                    "web_dist": repo_relative_label(web_dist_path, resolved_repo),
                },
                "safety": {
                    "boot_dir_safe": boot_safe,
                    "web_dist_safe": web_dist_safe,
                },
                "boot": boot_section,
                "wasm": wasm_section,
                "errors": errors,
                "warnings": warnings,
            }
        )

    # ---- Stage 1: Allowlisted boot artifacts (--program-dir -> --boot-dir) ----
    if not program_path.is_dir():
        boot_section["status"] = "error"
        errors.append(
            f"program-dir does not exist or is not a directory: "
            f"{repo_relative_label(program_path, resolved_repo)}"
        )
    else:
        resolved_program = program_path.resolve()
        resolved_boot = boot_path.resolve()
        if resolved_boot == resolved_program or resolved_boot.is_relative_to(resolved_program):
            boot_section["status"] = "refused"
            errors.append(
                "refusing to stage boot artifacts into program-dir itself or a subdirectory of it"
            )
        else:
            discovery = discover_boot_artifacts(program_path)
            boot_section["discovered_count"] = len(discovery.artifacts)
            boot_section["skipped_non_allowlisted"] = discovery.skipped_non_allowlisted
            boot_section["skipped_escaping_symlinks"] = discovery.skipped_escaping_symlinks

            if discovery.skipped_escaping_symlinks > 0:
                warnings.append(
                    f"skipped {discovery.skipped_escaping_symlinks} symlink(s) escaping program-dir"
                )

            if not discovery.artifacts:
                boot_section["status"] = "empty"
                warnings.append(
                    "no allowlisted boot artifacts (program-manifest.json, build-manifest.json, *.pgim) "
                    "found in program-dir"
                )
            else:
                copied = 0
                planned = 0
                unchanged = 0
                total_bytes = 0
                artifact_entries: list[dict[str, Any]] = []
                boot_failed = False

                for artifact in discovery.artifacts:
                    dst_file = (boot_path / artifact.rel_posix)
                    # Guard against destination path traversal even if rel_posix were strange.
                    if not dst_file.resolve().is_relative_to(resolved_boot):
                        boot_failed = True
                        errors.append(
                            f"refused boot artifact escaping boot-dir: {artifact.rel_posix}"
                        )
                        continue
                    try:
                        plan = plan_file_stage(artifact.source_path, dst_file)
                    except (OSError, StageBrowserBootError) as exc:
                        boot_failed = True
                        errors.append(
                            f"failed to inspect boot artifact {artifact.rel_posix}: "
                            f"{sanitize_text(str(exc))}"
                        )
                        continue

                    total_bytes += plan.size_bytes
                    if not plan.needs_copy:
                        unchanged += 1
                        action = "unchanged"
                    elif dry_run:
                        planned += 1
                        action = "plan_copy"
                    else:
                        try:
                            atomic_copy_file(artifact.source_path, dst_file)
                        except OSError as exc:
                            boot_failed = True
                            errors.append(
                                f"failed to stage boot artifact {artifact.rel_posix}: "
                                f"{sanitize_text(str(exc))}"
                            )
                            continue
                        copied += 1
                        action = "copied"

                    artifact_entries.append(
                        {
                            "rel_path": artifact.rel_posix,
                            "kind": artifact.kind,
                            "action": action,
                            "reason": plan.reason,
                            "size_bytes": plan.size_bytes,
                            "sha256": plan.sha256,
                        }
                    )

                boot_section["copied_count"] = copied
                boot_section["planned_count"] = planned
                boot_section["unchanged_count"] = unchanged
                boot_section["total_bytes"] = total_bytes
                boot_section["artifacts"] = artifact_entries

                if boot_failed:
                    boot_section["status"] = "error"
                elif dry_run:
                    boot_section["status"] = "planned" if planned > 0 else "unchanged"
                elif copied > 0:
                    boot_section["status"] = "staged"
                else:
                    boot_section["status"] = "unchanged"

    # ---- Stage 2: Validated Wasm64 runtime (--wasm-dir -> --web-dist) ----
    if wasm_path is not None:
        if not wasm_path.is_dir():
            wasm_section["status"] = "skipped"
            warnings.append("wasm-dir does not exist; skipping Wasm64 staging")
        else:
            src_wasm_raw = wasm_path / RUNTIME_WASM_NAME
            src_js_raw = wasm_path / RUNTIME_JS_NAME

            wasm_exists_raw = src_wasm_raw.exists() or src_wasm_raw.is_symlink()
            js_exists_raw = src_js_raw.exists() or src_js_raw.is_symlink()

            if not wasm_exists_raw and not js_exists_raw:
                wasm_section["status"] = "skipped"
                warnings.append(
                    "wasm-dir does not contain runtime.wasm + runtime.js; skipping Wasm64 staging"
                )
            else:
                wasm_safe_file, resolved_wasm = _check_file_within_root(
                    src_wasm_raw, wasm_path
                )
                js_safe_file, resolved_js = _check_file_within_root(
                    src_js_raw, wasm_path
                )

                if wasm_exists_raw and not wasm_safe_file:
                    wasm_section["status"] = "refused"
                    errors.append(
                        "refusing runtime.wasm from wasm-dir (escaping symlink or not a regular file)"
                    )
                elif js_exists_raw and not js_safe_file:
                    wasm_section["status"] = "refused"
                    errors.append(
                        "refusing runtime.js from wasm-dir (escaping symlink or not a regular file)"
                    )
                elif not wasm_safe_file:
                    wasm_section["status"] = "error"
                    errors.append("runtime.wasm missing in wasm-dir alongside runtime.js")
                elif not js_safe_file:
                    wasm_section["status"] = "error"
                    errors.append("runtime.js missing in wasm-dir alongside runtime.wasm")
                else:
                    assert resolved_wasm is not None and resolved_js is not None
                    validation = validate_wasm_candidate(resolved_wasm)
                    wasm_section["validation"] = validation

                    if not validation.get("ok"):
                        wasm_section["status"] = "refused"
                        val_errors = validation.get("errors") or [
                            "runtime.wasm failed Wasm64/shared/host-ABI validation"
                        ]
                        errors.append(
                            "refusing to stage runtime.wasm into web-dist (validation failed): "
                            + "; ".join(val_errors)
                        )
                    else:
                        file_pairs = (
                            (RUNTIME_WASM_NAME, resolved_wasm, web_dist_path / RUNTIME_WASM_NAME),
                            (RUNTIME_JS_NAME, resolved_js, web_dist_path / RUNTIME_JS_NAME),
                        )
                        plans: list[tuple[str, Path, Path, FileStagePlan]] = []
                        plan_failed = False
                        for name, src_file, dst_file in file_pairs:
                            try:
                                plan = plan_file_stage(src_file, dst_file)
                            except (OSError, StageBrowserBootError) as exc:
                                plan_failed = True
                                errors.append(
                                    f"failed to inspect {name}: {sanitize_text(str(exc))}"
                                )
                                break
                            plans.append((name, src_file, dst_file, plan))

                        if plan_failed:
                            wasm_section["status"] = "error"
                        else:
                            to_copy = [
                                (src_file, dst_file)
                                for _, src_file, dst_file, plan in plans
                                if plan.needs_copy
                            ]
                            if to_copy and not dry_run:
                                try:
                                    atomic_copy_files(to_copy)
                                except OSError as exc:
                                    wasm_section["status"] = "error"
                                    errors.append(
                                        f"failed to atomically stage Wasm64 runtime: "
                                        f"{sanitize_text(str(exc))}"
                                    )
                                    to_copy = []
                                    plan_failed = True

                            if not plan_failed:
                                copied = 0
                                planned = 0
                                unchanged = 0
                                file_entries: list[dict[str, Any]] = []
                                for name, _, _, plan in plans:
                                    if not plan.needs_copy:
                                        unchanged += 1
                                        action = "unchanged"
                                    elif dry_run:
                                        planned += 1
                                        action = "plan_copy"
                                    else:
                                        copied += 1
                                        action = "copied"
                                    file_entries.append(
                                        {
                                            "name": name,
                                            "action": action,
                                            "reason": plan.reason,
                                            "size_bytes": plan.size_bytes,
                                            "sha256": plan.sha256,
                                        }
                                    )
                                wasm_section["copied_count"] = copied
                                wasm_section["planned_count"] = planned
                                wasm_section["unchanged_count"] = unchanged
                                wasm_section["files"] = file_entries
                                if dry_run:
                                    wasm_section["status"] = (
                                        "planned" if planned > 0 else "unchanged"
                                    )
                                elif copied > 0:
                                    wasm_section["status"] = "staged"
                                else:
                                    wasm_section["status"] = "unchanged"

    ok = len(errors) == 0
    return sanitize_value(
        {
            "schema_version": SCHEMA_VERSION,
            "ok": ok,
            "dry_run": dry_run,
            "paths": {
                "program_dir": repo_relative_label(program_path, resolved_repo),
                "boot_dir": repo_relative_label(boot_path, resolved_repo),
                "wasm_dir": repo_relative_label(wasm_path, resolved_repo),
                "web_dist": repo_relative_label(web_dist_path, resolved_repo),
            },
            "safety": {
                "boot_dir_safe": boot_safe,
                "web_dist_safe": web_dist_safe,
            },
            "boot": boot_section,
            "wasm": wasm_section,
            "errors": errors,
            "warnings": warnings,
        }
    )


# ---------------------------------------------------------------------------
# Output formatting & CLI
# ---------------------------------------------------------------------------


def format_report_text(report: Mapping[str, Any]) -> str:
    """Render a concise, strictly sanitized human-readable staging summary."""
    status_label = "PASS" if report.get("ok") else "FAIL"
    mode = "dry-run" if report.get("dry_run") else "stage"
    paths = report.get("paths") or {}
    boot = report.get("boot") or {}
    wasm = report.get("wasm") or {}

    lines: list[str] = [
        f"Browser Real-Boot Staging [{status_label}] mode={mode}",
        f"  Boot: {paths.get('program_dir')} -> {paths.get('boot_dir')} "
        f"[{str(boot.get('status', 'unknown')).upper()}] "
        f"(discovered={boot.get('discovered_count', 0)}, "
        f"copied={boot.get('copied_count', 0)}, "
        f"planned={boot.get('planned_count', 0)}, "
        f"unchanged={boot.get('unchanged_count', 0)}, "
        f"skipped_other={boot.get('skipped_non_allowlisted', 0)}, "
        f"skipped_symlinks={boot.get('skipped_escaping_symlinks', 0)})",
    ]

    artifacts = boot.get("artifacts") or []
    for item in artifacts:
        lines.append(
            f"    [{str(item.get('action', '')).upper()}] "
            f"{item.get('rel_path')} ({item.get('kind')}, {item.get('size_bytes')} bytes, "
            f"{item.get('reason')})"
        )

    wasm_status = str(wasm.get("status", "skipped")).upper()
    wasm_src = paths.get("wasm_dir") or "<none>"
    lines.append(
        f"  Wasm64: {wasm_src} -> {paths.get('web_dist')} [{wasm_status}]"
    )
    validation = wasm.get("validation")
    if isinstance(validation, Mapping):
        val_tag = "PASS" if validation.get("ok") else "FAIL"
        lines.append(
            f"    Validation [{val_tag}]: "
            f"memory64={'YES' if validation.get('is_memory64') else 'NO'}, "
            f"shared={'YES' if validation.get('is_shared_memory') else 'NO'}, "
            f"host_imports={validation.get('host_imports_valid', 0)}/"
            f"{validation.get('host_imports_total', 0)}"
        )
    for item in wasm.get("files") or []:
        lines.append(
            f"    [{str(item.get('action', '')).upper()}] "
            f"{item.get('name')} ({item.get('size_bytes')} bytes, {item.get('reason')})"
        )

    warnings = report.get("warnings") or []
    if warnings:
        lines.append("  Warnings:")
        for warn in warnings:
            lines.append(f"    - {warn}")

    errors = report.get("errors") or []
    if errors:
        lines.append("  Errors:")
        for err in errors:
            lines.append(f"    - {err}")

    return sanitize_text("\n".join(lines))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Safely stage allowlisted local recomp boot artifacts (.pgim + manifests) "
            "and validated Wasm64 runtime.{wasm,js} for server/serve_local.py."
        )
    )
    parser.add_argument(
        "--program-dir",
        type=Path,
        default=DEFAULT_PROGRAM_DIR_REL,
        help="Source recomp program directory (default: work/recomp_program).",
    )
    parser.add_argument(
        "--wasm-dir",
        type=Path,
        default=None,
        help="Optional directory containing runtime.wasm + runtime.js (e.g. work/wasm_guest_dist).",
    )
    parser.add_argument(
        "--boot-dir",
        type=Path,
        default=DEFAULT_BOOT_DIR_REL,
        help="Destination boot directory served at /boot/ (default: work/recomp_build).",
    )
    parser.add_argument(
        "--web-dist",
        type=Path,
        default=DEFAULT_WEB_DIST_REL,
        help="Destination Wasm64 web dist directory (default: web/dist).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Repository root used for tracked-directory safety checks (default: parent of tools/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report planned copies without creating or modifying files.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit strictly sanitized machine-readable JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    report = stage_browser_boot(
        program_dir=args.program_dir,
        wasm_dir=args.wasm_dir,
        boot_dir=args.boot_dir,
        web_dist=args.web_dist,
        repo_root=args.repo_root,
        dry_run=args.dry_run,
    )

    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report_text(report))

    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
