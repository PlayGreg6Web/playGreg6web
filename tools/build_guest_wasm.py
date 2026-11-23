#!/usr/bin/env python3
"""Parallel sharded LLVM IR / C++ → Wasm64 Emscripten build orchestrator (Track 1 / R13).

Takes a `playgta6-recomp build` (R9) output directory of sharded `.ll` / `.c` / `.cpp`
translation units, compiles dirty shards in parallel with `emcc`/`em++`, caches `.o`
files by SHA-256 of content + compiler flags, and links with `libplaygta6_core.a`
into `web/dist/runtime.js` + `web/dist/runtime.wasm`.

Flags match `runtime/CMakeLists.txt` and `runtime/build_wasm.{sh,ps1}`:

    -sMEMORY64=1 -pthread -sMODULARIZE=1 -lexports.js ...

Pure Python 3.12 standard library only. Usable as a library and as a CLI:

    python tools/build_guest_wasm.py <recomp-out-dir>
        [--jobs N] [--cache-dir work/wasm_obj_cache] [--out-dir web/dist]
        [--opt O2|O3] [--runtime-lib path/to/libplaygta6_core.a]
        [--dry-run] [--check-only] [--verify] [--json] [--sanitize]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

SCHEMA_VERSION = 1
DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR_REL = Path("work") / "wasm_obj_cache"
DEFAULT_OUT_DIR_REL = Path("web") / "dist"
DEFAULT_RUNTIME_LIB_REL = Path("runtime") / "build-wasm64" / "libplaygta6_core.a"
EMSDK_UPSTREAM_REL = Path("work") / "emsdk" / "upstream" / "emscripten"
RUNTIME_JS = "runtime.js"
RUNTIME_WASM = "runtime.wasm"

SOURCE_SUFFIXES: frozenset[str] = frozenset({".ll", ".c", ".cc", ".cpp", ".cxx"})
IR_SUFFIXES: frozenset[str] = frozenset({".ll"})
CXX_SUFFIXES: frozenset[str] = frozenset({".cc", ".cpp", ".cxx"})
SKIP_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        "node_modules",
        "target",
        "local-data",
    }
)

# Compile flags applied to every translation unit (plus `-O*` from --opt).
COMPILE_WASM_FLAGS: tuple[str, ...] = ("-sMEMORY64=1", "-pthread")

# Link flags matching runtime/CMakeLists.txt EMSCRIPTEN target_link_options.
LINK_FLAGS: tuple[str, ...] = (
    "-pthread",
    "-sMEMORY64=1",
    "-sPTHREAD_POOL_SIZE=4",
    "-sMODULARIZE=1",
    "-sEXPORT_NAME=createRuntimeModule",
    "-sALLOW_MEMORY_GROWTH=1",
    # Guest address space (runtime/os/vmem.h): identity-mapped guest pool in linear memory
    # [4 GiB, 16 GiB), runtime heap fenced below it.
    "-sMAXIMUM_MEMORY=16GB",
    "-Wl,--wrap=sbrk",
    "-sENVIRONMENT=worker",
    "-sERROR_ON_UNDEFINED_SYMBOLS=0",
    "-lexports.js",
    "-Wl,--undefined=__playgta6_indirect_call",
    "-Wl,--undefined=__playgta6_trap",
    "-Wl,--undefined=__playgta6_unimplemented",
    "-Wl,--undefined=__playgta6_divide_error",
)

VERIFY_FLAGS: tuple[str, ...] = (
    "--require-memory64",
    "--require-shared-memory",
    "--require-host-imports",
)

OPT_LEVELS: frozenset[str] = frozenset({"O0", "O1", "O2", "O3"})

_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"'<>|,;()\[\]]+")
_UNC_PATH_RE = re.compile(r"\\\\[^\s\"'<>|,;()\[\]]+")
_POSIX_PATH_RE = re.compile(
    r"/(?:Users|home|tmp|var|opt|root|mnt|private|etc|usr|Library|Volumes)/[^\s\"'<>|,;()\[\]]*"
)

CommandRunner = Callable[[Sequence[str]], "CommandResult"]
WhichFn = Callable[[str], str | None]


# ---------------------------------------------------------------------------
# Errors / results
# ---------------------------------------------------------------------------


class BuildGuestWasmError(Exception):
    """Raised for user-facing configuration or discovery failures."""


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class SourceUnit:
    """One translation unit discovered from a recomp output directory."""

    unit_id: str
    path: Path
    kind: str  # "ll" | "c" | "cpp"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.unit_id, "kind": self.kind}


@dataclass(frozen=True)
class Toolchain:
    emcc: Path | None
    emxx: Path | None
    origin: str

    @property
    def ok(self) -> bool:
        return self.emcc is not None

    def compiler_for(self, kind: str) -> Path | None:
        if kind == "cpp" and self.emxx is not None:
            return self.emxx
        return self.emcc

    def linker(self) -> Path | None:
        if self.emxx is not None:
            return self.emxx
        return self.emcc

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "emcc_found": self.emcc is not None,
            "emxx_found": self.emxx is not None,
            "origin": self.origin,
        }


@dataclass
class UnitBuildResult:
    unit: SourceUnit
    cache_key: str
    cache_hit: bool
    object_path: Path | None
    argv: tuple[str, ...]
    status: str  # "hit" | "compiled" | "planned" | "error"
    returncode: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.unit.unit_id,
            "kind": self.unit.kind,
            "cache_key": self.cache_key,
            "cache_hit": self.cache_hit,
            "status": self.status,
            "returncode": self.returncode,
            "argv": list(self.argv),
        }
        if self.error:
            out["error"] = self.error
        return out


@dataclass
class BuildReport:
    ok: bool
    mode: str
    schema_version: int = SCHEMA_VERSION
    toolchain: Toolchain | None = None
    jobs: int = 1
    opt: str = "O2"
    compile_flags: tuple[str, ...] = ()
    link_flags: tuple[str, ...] = LINK_FLAGS
    sources: list[UnitBuildResult] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    link_argv: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    verify: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "mode": self.mode,
            "schema_version": self.schema_version,
            "toolchain": self.toolchain.to_dict() if self.toolchain else None,
            "jobs": self.jobs,
            "opt": self.opt,
            "compile_flags": list(self.compile_flags),
            "link_flags": list(self.link_flags),
            "sources": [u.to_dict() for u in self.sources],
            "source_count": len(self.sources),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "link_argv": list(self.link_argv),
            "outputs": list(self.outputs),
            "verify": self.verify,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Sanitization (never leak host paths / usernames in JSON or --sanitize text)
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


def sanitize_text(text: str) -> str:
    """Replace host paths and usernames in a free-form string."""
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


def sanitize_argv(argv: Sequence[str]) -> list[str]:
    """Keep flag tokens; replace filesystem operands with basenames / labels."""
    out: list[str] = []
    skip_next_as_file = False
    file_flags = {"-o", "-I", "--cache-dir", "--out-dir", "--runtime-lib"}
    for idx, token in enumerate(argv):
        if skip_next_as_file:
            out.append(_sanitize_fs_token(token))
            skip_next_as_file = False
            continue
        if token in file_flags:
            out.append(token)
            skip_next_as_file = True
            continue
        if idx == 0:
            out.append(_tool_label(token))
            continue
        if token.startswith("-"):
            out.append(token)
            continue
        out.append(_sanitize_fs_token(token))
    return out


def _tool_label(token: str) -> str:
    normalized = token.replace("\\", "/")
    basename = normalized.rsplit("/", 1)[-1]
    name = basename.lower()
    if name.startswith("em++") or name.startswith("emxx"):
        return "em++"
    if name.startswith("emcc"):
        return "emcc"
    if name.startswith("python"):
        return "python"
    stem = Path(basename).stem
    return stem or "tool"


def _sanitize_fs_token(token: str) -> str:
    normalized = token.replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1]
    if name in {RUNTIME_JS, RUNTIME_WASM, "libplaygta6_core.a"}:
        return name
    suffix = Path(name).suffix.lower()
    if suffix in {".o", ".a", ".ll", ".c", ".cc", ".cpp", ".cxx", ".js", ".wasm", ".h"}:
        return name
    if suffix or "/" in normalized or (len(token) >= 2 and token[1] == ":"):
        return name or "<path>"
    return token


def posix_rel(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
        return rel.as_posix()
    except ValueError:
        return path.name


# ---------------------------------------------------------------------------
# Toolchain discovery
# ---------------------------------------------------------------------------


_EMCC_NAMES: tuple[str, ...] = ("emcc", "emcc.bat", "emcc.cmd", "emcc.exe", "emcc.py")
_EMXX_NAMES: tuple[str, ...] = (
    "em++",
    "em++.bat",
    "em++.cmd",
    "em++.exe",
    "em++.py",
    "emxx",
    "emxx.bat",
    "emxx.cmd",
)


def _first_existing(directory: Path, names: Sequence[str]) -> Path | None:
    if not directory.is_dir():
        return None
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def discover_emscripten(
    *,
    repo_root: Path,
    env: Mapping[str, str] | None = None,
    which_fn: WhichFn | None = None,
) -> Toolchain:
    """Locate emcc/em++ on PATH, EMSDK, or `work/emsdk/upstream/emscripten`."""
    env_map = dict(os.environ if env is None else env)
    which = shutil.which if which_fn is None else which_fn

    emcc_path: Path | None = None
    emxx_path: Path | None = None
    origin = "missing"

    which_emcc = which("emcc")
    which_emxx = which("em++") or which("emxx")
    if which_emcc:
        emcc_path = Path(which_emcc)
        origin = "path"
    if which_emxx:
        emxx_path = Path(which_emxx)
        if origin == "missing":
            origin = "path"

    search_dirs: list[tuple[str, Path]] = []
    emsdk = (env_map.get("EMSDK") or "").strip()
    if emsdk:
        search_dirs.append(("emsdk", Path(emsdk) / "upstream" / "emscripten"))
    search_dirs.append(("work/emsdk", repo_root / EMSDK_UPSTREAM_REL))

    for label, directory in search_dirs:
        found_cc = _first_existing(directory, _EMCC_NAMES)
        found_xx = _first_existing(directory, _EMXX_NAMES)
        if emcc_path is None and found_cc is not None:
            emcc_path = found_cc
            origin = label
        if emxx_path is None and found_xx is not None:
            emxx_path = found_xx
            if origin == "missing":
                origin = label

    return Toolchain(emcc=emcc_path, emxx=emxx_path, origin=origin)


def default_run_command(argv: Sequence[str]) -> CommandResult:
    if not argv:
        return CommandResult(returncode=1, stderr="empty command")
    try:
        proc = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except OSError as exc:
        return CommandResult(returncode=127, stderr=str(exc))
    return CommandResult(
        returncode=int(proc.returncode),
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------


def parse_opt_level(raw: str) -> str:
    text = raw.strip()
    if text.startswith("-"):
        text = text[1:]
    if text in OPT_LEVELS:
        return text
    if text in {"0", "1", "2", "3"}:
        return f"O{text}"
    raise BuildGuestWasmError(f"invalid --opt {raw!r}; expected O0, O1, O2, or O3")


def compile_flags_for(*, opt: str, kind: str, include_dirs: Sequence[Path] = ()) -> tuple[str, ...]:
    flags: list[str] = [*COMPILE_WASM_FLAGS, f"-{opt}"]
    if kind == "ll":
        flags.extend(["-x", "ir"])
    elif kind == "cpp":
        flags.append("-std=c++20")
    flags.append("-c")
    for include in include_dirs:
        flags.extend(["-I", str(include)])
    return tuple(flags)


def cache_key_for(content: bytes, flags: Sequence[str], *, kind: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"playgta6-guest-wasm-v1\0")
    digest.update(kind.encode("utf-8"))
    digest.update(b"\0")
    for flag in flags:
        digest.update(flag.encode("utf-8"))
        digest.update(b"\0")
    digest.update(content)
    return digest.hexdigest()


def _kind_for_suffix(suffix: str) -> str | None:
    lowered = suffix.lower()
    if lowered in IR_SUFFIXES:
        return "ll"
    if lowered == ".c":
        return "c"
    if lowered in CXX_SUFFIXES:
        return "cpp"
    return None


def _collect_manifest_paths(data: Any) -> list[str]:
    paths: list[str] = []

    def take(item: Any) -> None:
        if isinstance(item, str) and item.strip():
            paths.append(item.strip())
        elif isinstance(item, Mapping):
            for key in ("path", "file", "source", "ll", "cpp", "c"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    paths.append(value.strip())
                    return

    if isinstance(data, list):
        for item in data:
            take(item)
        return paths
    if not isinstance(data, Mapping):
        return paths

    for key in (
        "sources",
        "files",
        "shards",
        "units",
        "translation_units",
        "ll",
        "c",
        "cpp",
        "cxx",
    ):
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                take(item)
        else:
            take(value)
    # R9 `playgta6-recomp build` nests shard paths under lift.shards[].ll and files[].path.
    for nested_key in ("build", "outputs", "lift", "registration"):
        nested = data.get(nested_key)
        if isinstance(nested, Mapping):
            paths.extend(_collect_manifest_paths(nested))
        elif isinstance(nested, list):
            paths.extend(_collect_manifest_paths(nested))
    return paths


def load_build_manifest(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_sources(source_dir: Path, *, manifest_name: str = "build-manifest.json") -> list[SourceUnit]:
    """Discover sharded `.ll` / `.c` / `.cpp` units from a recomp output directory."""
    if not source_dir.is_dir():
        raise BuildGuestWasmError(f"source directory not found: {source_dir.name}")

    root = source_dir.resolve()
    found: dict[str, SourceUnit] = {}

    def add_file(path: Path) -> None:
        if not path.is_file():
            return
        kind = _kind_for_suffix(path.suffix)
        if kind is None:
            return
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            return
        unit_id = posix_rel(resolved, root)
        found[unit_id] = SourceUnit(unit_id=unit_id, path=resolved, kind=kind)

    manifest_path = root / manifest_name
    if manifest_path.is_file():
        try:
            data = load_build_manifest(manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            raise BuildGuestWasmError(f"invalid {manifest_name}: {exc}") from exc
        for rel in _collect_manifest_paths(data):
            add_file((root / rel).resolve() if not Path(rel).is_absolute() else Path(rel))

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for name in filenames:
            add_file(Path(dirpath) / name)

    units = list(found.values())
    units.sort(key=lambda u: u.unit_id)
    return units


def cache_object_path(cache_dir: Path, key: str, unit_id: str) -> Path:
    stem = Path(unit_id.replace("/", "__")).stem
    return cache_dir / key[:2] / f"{stem}-{key[:16]}.o"


def build_compile_argv(
    compiler: Path | str,
    source: Path,
    output: Path,
    flags: Sequence[str],
) -> tuple[str, ...]:
    return (str(compiler), *flags, str(source), "-o", str(output))


def build_link_argv(
    linker: Path | str,
    objects: Sequence[Path],
    *,
    output_js: Path,
    runtime_libs: Sequence[Path] = (),
    extra_flags: Sequence[str] = (),
) -> tuple[str, ...]:
    argv: list[str] = [str(linker), *LINK_FLAGS, *extra_flags]
    argv.extend(str(path) for path in objects)
    argv.extend(str(path) for path in runtime_libs)
    argv.extend(["-o", str(output_js)])
    return tuple(argv)


# ---------------------------------------------------------------------------
# Build orchestration
# ---------------------------------------------------------------------------


def compile_unit(
    unit: SourceUnit,
    *,
    compiler: Path,
    flags: Sequence[str],
    cache_dir: Path,
    dry_run: bool,
    run_command: CommandRunner,
) -> UnitBuildResult:
    content = unit.path.read_bytes()
    key = cache_key_for(content, flags, kind=unit.kind)
    object_path = cache_object_path(cache_dir, key, unit.unit_id)
    argv = build_compile_argv(compiler, unit.path, object_path, flags)
    display_argv = tuple(sanitize_argv(argv))

    if object_path.is_file():
        return UnitBuildResult(
            unit=unit,
            cache_key=key,
            cache_hit=True,
            object_path=object_path,
            argv=display_argv,
            status="hit",
        )

    if dry_run:
        return UnitBuildResult(
            unit=unit,
            cache_key=key,
            cache_hit=False,
            object_path=object_path,
            argv=display_argv,
            status="planned",
        )

    object_path.parent.mkdir(parents=True, exist_ok=True)
    result = run_command(argv)
    if result.returncode != 0:
        err = sanitize_text((result.stderr or result.stdout or "compile failed").strip())
        return UnitBuildResult(
            unit=unit,
            cache_key=key,
            cache_hit=False,
            object_path=None,
            argv=display_argv,
            status="error",
            returncode=result.returncode,
            error=f"{unit.unit_id}: {err}",
        )
    if not object_path.is_file():
        return UnitBuildResult(
            unit=unit,
            cache_key=key,
            cache_hit=False,
            object_path=None,
            argv=display_argv,
            status="error",
            returncode=1,
            error=f"{unit.unit_id}: compiler exited 0 but did not write object file",
        )
    return UnitBuildResult(
        unit=unit,
        cache_key=key,
        cache_hit=False,
        object_path=object_path,
        argv=display_argv,
        status="compiled",
    )


def run_verify(
    wasm_path: Path,
    *,
    run_command: CommandRunner | None = None,
    python_executable: str | None = None,
    tools_dir: Path | None = None,
) -> dict[str, Any]:
    """Run tools/check_wasm_exports.py with the required Wasm64/ABI flags."""
    checker = (tools_dir or DEFAULT_REPO_ROOT / "tools") / "check_wasm_exports.py"
    argv = [
        python_executable or sys.executable,
        str(checker),
        str(wasm_path),
        *VERIFY_FLAGS,
        "--json",
    ]
    if run_command is not None:
        result = run_command(argv)
        payload: dict[str, Any]
        try:
            payload = json.loads(result.stdout) if result.stdout.strip() else {}
        except json.JSONDecodeError:
            payload = {}
        ok = result.returncode == 0 and bool(payload.get("ok", False))
        errors = payload.get("errors") if isinstance(payload.get("errors"), list) else []
        if result.returncode != 0 and not errors:
            errors = [sanitize_text((result.stderr or "verify failed").strip())]
        return {
            "ok": ok,
            "returncode": result.returncode,
            "argv": sanitize_argv(argv),
            "errors": [sanitize_text(str(e)) for e in errors],
        }

    tools_parent = str((tools_dir or DEFAULT_REPO_ROOT / "tools"))
    if tools_parent not in sys.path:
        sys.path.insert(0, tools_parent)
    import check_wasm_exports  # noqa: WPS433 — local tools/ import

    report = check_wasm_exports.validate_wasm_file(
        wasm_path,
        require_memory64=True,
        require_shared_memory=True,
        require_host_imports=True,
    )
    data = report.to_dict()
    data["path"] = RUNTIME_WASM
    data["argv"] = sanitize_argv(argv)
    return {
        "ok": report.ok,
        "returncode": 0 if report.ok else 1,
        "argv": sanitize_argv(argv),
        "errors": [sanitize_text(e) for e in report.errors],
        "is_memory64": report.is_memory64,
        "is_shared_memory": report.is_shared_memory,
        "all_host_imports_ok": report.all_host_imports_ok,
    }


def resolve_runtime_libs(
    *,
    repo_root: Path,
    runtime_lib: Path | None,
    extra_objects: Sequence[Path],
) -> list[Path]:
    libs: list[Path] = []
    if runtime_lib is not None:
        libs.append(runtime_lib)
    else:
        default_lib = repo_root / DEFAULT_RUNTIME_LIB_REL
        if default_lib.is_file():
            libs.append(default_lib)
        else:
            alt = repo_root / "runtime" / "libplaygta6_core.a"
            if alt.is_file():
                libs.append(alt)
    libs.extend(extra_objects)
    return libs


def run_build(
    *,
    source_dir: Path | None,
    repo_root: Path = DEFAULT_REPO_ROOT,
    cache_dir: Path | None = None,
    out_dir: Path | None = None,
    jobs: int | None = None,
    opt: str = "O2",
    runtime_lib: Path | None = None,
    extra_objects: Sequence[Path] = (),
    extra_link_flags: Sequence[str] = (),
    dry_run: bool = False,
    check_only: bool = False,
    verify: bool = False,
    run_command: CommandRunner | None = None,
    which_fn: WhichFn | None = None,
    env: Mapping[str, str] | None = None,
) -> BuildReport:
    """Discover, compile (in parallel), and link guest shards into runtime.{js,wasm}."""
    opt_level = parse_opt_level(opt)
    worker_count = jobs if jobs is not None else (os.cpu_count() or 1)
    if worker_count < 1:
        raise BuildGuestWasmError("--jobs must be >= 1")

    runner = run_command or default_run_command
    toolchain = discover_emscripten(repo_root=repo_root, env=env, which_fn=which_fn)
    mode = "check-only" if check_only else ("dry-run" if dry_run else "build")
    report = BuildReport(
        ok=toolchain.ok,
        mode=mode,
        toolchain=toolchain,
        jobs=worker_count,
        opt=opt_level,
        compile_flags=(*COMPILE_WASM_FLAGS, f"-{opt_level}"),
        link_flags=LINK_FLAGS,
    )

    if not toolchain.ok:
        report.errors.append(
            "emcc not found on PATH, EMSDK, or work/emsdk/upstream/emscripten"
        )
        return report

    if check_only:
        if toolchain.emxx is None:
            report.warnings.append("em++ not found; C++ shards would use emcc")
        return report

    if source_dir is None:
        report.ok = False
        report.errors.append("missing source directory (recomp build output)")
        return report

    units = discover_sources(source_dir)
    if not units:
        report.ok = False
        report.errors.append("no .ll/.c/.cpp translation units found")
        return report

    cache_root = (cache_dir or (repo_root / DEFAULT_CACHE_DIR_REL)).resolve()
    output_dir = (out_dir or (repo_root / DEFAULT_OUT_DIR_REL)).resolve()
    include_dirs = [repo_root / "runtime", source_dir.resolve()]
    include_dirs = [p for p in include_dirs if p.is_dir()]

    planned: list[UnitBuildResult] = []
    compile_tasks: list[tuple[SourceUnit, Path, tuple[str, ...]]] = []
    for unit in units:
        compiler = toolchain.compiler_for(unit.kind)
        if compiler is None:
            report.ok = False
            report.errors.append(f"no compiler for {unit.unit_id} ({unit.kind})")
            continue
        flags = compile_flags_for(opt=opt_level, kind=unit.kind, include_dirs=include_dirs)
        content = unit.path.read_bytes()
        key = cache_key_for(content, flags, kind=unit.kind)
        object_path = cache_object_path(cache_root, key, unit.unit_id)
        argv = tuple(sanitize_argv(build_compile_argv(compiler, unit.path, object_path, flags)))
        if object_path.is_file():
            planned.append(
                UnitBuildResult(
                    unit=unit,
                    cache_key=key,
                    cache_hit=True,
                    object_path=object_path,
                    argv=argv,
                    status="hit",
                )
            )
        else:
            compile_tasks.append((unit, compiler, flags))
            planned.append(
                UnitBuildResult(
                    unit=unit,
                    cache_key=key,
                    cache_hit=False,
                    object_path=object_path,
                    argv=argv,
                    status="planned" if dry_run else "pending",
                )
            )

    compiled_by_id: dict[str, UnitBuildResult] = {item.unit.unit_id: item for item in planned}

    if not dry_run and compile_tasks:
        lock = threading.Lock()

        def _job(task: tuple[SourceUnit, Path, tuple[str, ...]]) -> UnitBuildResult:
            unit, compiler, flags = task
            return compile_unit(
                unit,
                compiler=compiler,
                flags=flags,
                cache_dir=cache_root,
                dry_run=False,
                run_command=runner,
            )

        workers = min(worker_count, len(compile_tasks))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_job, task) for task in compile_tasks]
            for fut in as_completed(futures):
                result = fut.result()
                with lock:
                    compiled_by_id[result.unit.unit_id] = result

    ordered = [compiled_by_id[u.unit_id] for u in units]
    report.sources = ordered
    report.cache_hits = sum(1 for item in ordered if item.cache_hit)
    report.cache_misses = sum(1 for item in ordered if not item.cache_hit)

    compile_errors = [item.error for item in ordered if item.status == "error" and item.error]
    report.errors.extend(compile_errors)

    objects = [item.object_path for item in ordered if item.object_path is not None]
    runtime_libs = resolve_runtime_libs(
        repo_root=repo_root,
        runtime_lib=runtime_lib,
        extra_objects=extra_objects,
    )
    if not runtime_libs:
        report.warnings.append("libplaygta6_core.a not found; link line omits the runtime archive")

    linker = toolchain.linker()
    assert linker is not None
    output_js = output_dir / RUNTIME_JS
    output_wasm = output_dir / RUNTIME_WASM
    raw_link = build_link_argv(
        linker,
        objects if objects else [Path(item.unit.unit_id).with_suffix(".o") for item in ordered],
        output_js=output_js,
        runtime_libs=runtime_libs,
        extra_flags=extra_link_flags,
    )
    report.link_argv = tuple(sanitize_argv(raw_link))
    report.outputs = (RUNTIME_JS, RUNTIME_WASM)

    if compile_errors:
        report.ok = False
        return report

    if dry_run:
        report.ok = True
        return report

    output_dir.mkdir(parents=True, exist_ok=True)
    link_result = runner(raw_link)
    if link_result.returncode != 0:
        report.ok = False
        report.errors.append(
            sanitize_text((link_result.stderr or link_result.stdout or "link failed").strip())
        )
        return report

    missing_outputs = [name for name, path in ((RUNTIME_JS, output_js), (RUNTIME_WASM, output_wasm)) if not path.is_file()]
    if missing_outputs:
        report.ok = False
        report.errors.append("missing link outputs: " + ", ".join(missing_outputs))
        return report

    if verify:
        report.verify = run_verify(
            output_wasm,
            run_command=run_command,
            tools_dir=repo_root / "tools",
        )
        if not report.verify.get("ok"):
            report.ok = False
            report.errors.extend(report.verify.get("errors") or ["wasm export verification failed"])
            return report

    report.ok = True
    return report


def format_report_text(report: BuildReport, *, sanitize: bool = False) -> str:
    status = "PASS" if report.ok else "FAIL"
    lines = [
        f"Guest Wasm64 build [{status}] mode={report.mode}",
        f"  Toolchain: emcc={'yes' if report.toolchain and report.toolchain.emcc else 'no'} "
        f"em++={'yes' if report.toolchain and report.toolchain.emxx else 'no'} "
        f"origin={report.toolchain.origin if report.toolchain else 'n/a'}",
        f"  Jobs: {report.jobs}  Opt: -{report.opt}",
        f"  Compile flags: {' '.join(report.compile_flags)}",
        f"  Link flags: {' '.join(report.link_flags)}",
        f"  Sources: {len(report.sources)} (cache hits {report.cache_hits}, misses {report.cache_misses})",
    ]
    if report.sources:
        lines.append("  Units:")
        for item in report.sources:
            tag = item.status.upper()
            lines.append(f"    [{tag}] {item.unit.unit_id} ({item.unit.kind})")
    if report.link_argv:
        lines.append("  Link: " + " ".join(report.link_argv))
    if report.outputs:
        lines.append("  Outputs: " + ", ".join(report.outputs))
    if report.verify is not None:
        vstat = "PASS" if report.verify.get("ok") else "FAIL"
        lines.append(f"  Verify: {vstat}")
    if report.errors:
        lines.append("  Errors:")
        for err in report.errors:
            lines.append(f"    - {err}")
    if report.warnings:
        lines.append("  Warnings:")
        for warn in report.warnings:
            lines.append(f"    - {warn}")
    text = "\n".join(lines)
    return sanitize_text(text) if sanitize else text


def report_to_json(report: BuildReport) -> str:
    data = report.to_dict()
    encoded = json.dumps(data, indent=2)
    return sanitize_text(encoded)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile sharded playgta6-recomp LLVM IR / C++ sources to Wasm64 with "
            "Emscripten and link web/dist/runtime.js + runtime.wasm."
        )
    )
    parser.add_argument(
        "source_dir",
        nargs="?",
        default=None,
        help="R9 recomp build output directory (sharded .ll/.c/.cpp). Optional with --check-only.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root (default: parent of tools/).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Parallel compile workers (default: CPU count).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=f"Object cache directory (default: {DEFAULT_CACHE_DIR_REL.as_posix()}).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=f"Link output directory for runtime.js/wasm (default: {DEFAULT_OUT_DIR_REL.as_posix()}).",
    )
    parser.add_argument(
        "--opt",
        default="O2",
        help="Optimization level: O0, O1, O2 (default), or O3.",
    )
    parser.add_argument(
        "--runtime-lib",
        default=None,
        help="Path to libplaygta6_core.a (default: runtime/build-wasm64/libplaygta6_core.a).",
    )
    parser.add_argument(
        "--extra-object",
        action="append",
        default=[],
        dest="extra_objects",
        help="Additional object/archive to pass to the linker (repeatable).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover sources and print compile/link commands without invoking emcc.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify emcc/em++ discovery (PATH, EMSDK, work/emsdk/upstream/emscripten) and exit.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After linking, run tools/check_wasm_exports.py with Wasm64/shared/host-import checks.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit a sanitized machine-readable JSON report.",
    )
    parser.add_argument(
        "--sanitize",
        action="store_true",
        help="Sanitize host paths and usernames in text output (JSON is always sanitized).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else DEFAULT_REPO_ROOT

    source = Path(args.source_dir) if args.source_dir else None
    cache_dir = Path(args.cache_dir) if args.cache_dir else repo_root / DEFAULT_CACHE_DIR_REL
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / DEFAULT_OUT_DIR_REL
    runtime_lib = Path(args.runtime_lib) if args.runtime_lib else None
    extra_objects = [Path(p) for p in args.extra_objects]

    try:
        report = run_build(
            source_dir=source,
            repo_root=repo_root,
            cache_dir=cache_dir,
            out_dir=out_dir,
            jobs=args.jobs,
            opt=args.opt,
            runtime_lib=runtime_lib,
            extra_objects=extra_objects,
            dry_run=args.dry_run,
            check_only=args.check_only,
            verify=args.verify,
        )
    except BuildGuestWasmError as exc:
        err = sanitize_text(str(exc)) if (args.as_json or args.sanitize) else str(exc)
        if args.as_json:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "schema_version": SCHEMA_VERSION,
                        "errors": [err],
                    },
                    indent=2,
                )
            )
        else:
            print(f"Error: {err}", file=sys.stderr)
        return 2

    if args.as_json:
        print(report_to_json(report))
    else:
        print(format_report_text(report, sanitize=args.sanitize))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
