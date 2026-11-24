#!/usr/bin/env python3
"""Parallel sharded LLVM IR / C++ → Wasm64 Emscripten build orchestrator (Track 1 / R13, R21, R24, R25).

Takes a `playgta6-recomp build` (R9) output directory of sharded `.ll` / `.c` / `.cpp`
translation units, or a multi-module `playgta6-recomp build-program` (R12) tree with
`program-manifest.json`, compiles dirty shards in parallel with `emcc`/`em++`, caches
`.o` files by SHA-256 of content + compiler flags, and links with `libplaygta6_core.a`
into `web/dist/runtime.js` + `web/dist/runtime.wasm`.

Flags match `runtime/CMakeLists.txt` and `runtime/build_wasm.{sh,ps1}`:

    -sMEMORY64=1 -pthread -sMODULARIZE=1 -lexports.js ...

Pure Python 3.12 standard library only. Usable as a library and as a CLI:

    python tools/build_guest_wasm.py <recomp-out-dir>
        [--jobs N] [--cache-dir work/wasm_obj_cache] [--out-dir web/dist]
        [--opt O0|O1|O2|O3|Os|Oz] [--runtime-lib path/to/libplaygta6_core.a]
        [--strip] [--pthread-pool-size N] [--extra-link-flag FLAG]
        [--response-file] [--sample N] [--link-guest-boot]
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
import time
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
BUILD_MANIFEST_FILE = "build-manifest.json"
PROGRAM_MANIFEST_FILE = "program-manifest.json"
RESPONSE_FILE_NAME = "link_objects.rsp"
RESPONSE_FILE_THRESHOLD = 32
BOOT_LINKED_GUEST_DEFINE = "-DPG6_BOOT_LINKED_GUEST=1"
BOOT_PROGRAM_DEFINE = "-DPG6_BOOT_PROGRAM=1"
DEFAULT_GUEST_BOOT_ENTRY_REL = Path("runtime") / "main.cpp"
DEFAULT_WASM_BOOT_REL = Path("runtime") / "wasm_boot.cpp"
PROGRAM_LEVEL_UNITS: tuple[str, ...] = (
    "exports/pg6_export_thunks.cpp",
    "program_modules.cpp",
    "pg6_program.cpp",
)

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

# Bump whenever retarget_wasm64_ir changes its output (salted into .ll cache keys).
WASM64_IR_RETARGET_VERSION = 1
WASM64_SOURCE_TRIPLE = "wasm64-unknown-unknown"
WASM64_EMSCRIPTEN_TRIPLE = "wasm64-unknown-emscripten"

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

STRIP_LINK_FLAGS: tuple[str, ...] = (
    "-Wl,--strip-all",
    "-sASSERTIONS=0",
)
PTHREAD_POOL_SIZE_PREFIX = "-sPTHREAD_POOL_SIZE="
LINK_OPT_LEVELS: frozenset[str] = frozenset({"Os", "Oz"})

# Chrome/V8 enforces a hard 1 GiB WebAssembly.instantiate() module size limit.
V8_MAX_WASM_MODULE_BYTES = 1024 * 1024 * 1024

VERIFY_FLAGS: tuple[str, ...] = (
    "--require-memory64",
    "--require-shared-memory",
    "--require-host-imports",
)

OPT_LEVELS: frozenset[str] = frozenset({"O0", "O1", "O2", "O3", "Os", "Oz"})

_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"'<>|,;()\[\]]+")
_UNC_PATH_RE = re.compile(r"\\\\[^\s\"'<>|,;()\[\]]+")
_POSIX_PATH_RE = re.compile(
    r"/(?:Users|home|tmp|var|opt|root|mnt|private|etc|usr|Library|Volumes)/[^\s\"'<>|,;()\[\]]*"
)
_TARGET_DATALAYOUT_RE = re.compile(r"^[ \t]*target[ \t]+datalayout[ \t]*=")
_TARGET_TRIPLE_WASM64_RE = re.compile(
    r'^([ \t]*target[ \t]+triple[ \t]*=[ \t]*)"wasm64-unknown-unknown"([ \t]*(?:;.*)?)$'
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
    module: str | None = None
    instructions: int = 0

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.unit_id, "kind": self.kind}
        if self.module:
            out["module"] = self.module
        if self.instructions:
            out["instructions"] = self.instructions
        return out


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
    seconds: float = 0.0
    object_bytes: int = 0

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
        if self.unit.module:
            out["module"] = self.unit.module
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
    linked_guest_boot: bool = False
    response_file: str | None = None
    sample: int = 0
    extrapolated: dict[str, Any] | None = None
    compile_seconds: float = 0.0
    object_bytes: int = 0
    wasm_bytes: int = 0
    exceeds_v8_1gib_limit: bool = False

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
            "linked_guest_boot": self.linked_guest_boot,
            "response_file": self.response_file,
            "sample": self.sample,
            "extrapolated": self.extrapolated,
            "compile_seconds": self.compile_seconds,
            "object_bytes": self.object_bytes,
            "wasm_bytes": self.wasm_bytes,
            "exceeds_v8_1gib_limit": self.exceeds_v8_1gib_limit,
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
    file_flags = {
        "-o",
        "-I",
        "--cache-dir",
        "--out-dir",
        "--runtime-lib",
        "--guest-boot-entry",
    }
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
    rsp_prefix = token.startswith("@")
    body = token[1:] if rsp_prefix else token
    normalized = body.replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1]
    if name in {RUNTIME_JS, RUNTIME_WASM, "libplaygta6_core.a", RESPONSE_FILE_NAME}:
        return f"@{name}" if rsp_prefix else name
    suffix = Path(name).suffix.lower()
    if suffix in {".o", ".a", ".ll", ".c", ".cc", ".cpp", ".cxx", ".js", ".wasm", ".h", ".rsp"}:
        return f"@{name}" if rsp_prefix else name
    drive = len(body) >= 2 and body[1] == ":"
    if suffix or "/" in normalized or drive:
        label = name or "<path>"
        return f"@{label}" if rsp_prefix else label
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
    if text in {"0", "1", "2", "3", "s", "z"}:
        return f"O{text}"
    raise BuildGuestWasmError(f"invalid --opt {raw!r}; expected O0, O1, O2, O3, Os, or Oz")


def retarget_wasm64_ir(text: str) -> str:
    """Normalize lifted wasm64 LLVM IR for Emscripten 7 / LLVM 24 `emcc`.

    Drops any hardcoded `target datalayout = ...` line so `emcc` supplies the
    active LLVM backend's DataLayout, and rewrites
    `target triple = "wasm64-unknown-unknown"` to `"wasm64-unknown-emscripten"`.
    """
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.endswith("\r\n"):
            body, eol = line[:-2], "\r\n"
        elif line.endswith("\n"):
            body, eol = line[:-1], "\n"
        else:
            body, eol = line, ""
        if _TARGET_DATALAYOUT_RE.match(body):
            continue
        match = _TARGET_TRIPLE_WASM64_RE.match(body)
        if match is not None:
            out.append(f'{match.group(1)}"{WASM64_EMSCRIPTEN_TRIPLE}"{match.group(2)}{eol}')
        else:
            out.append(line)
    return "".join(out)


def compile_flags_for(
    *,
    opt: str,
    kind: str,
    include_dirs: Sequence[Path] = (),
    extra_flags: Sequence[str] = (),
) -> tuple[str, ...]:
    flags: list[str] = [*COMPILE_WASM_FLAGS, f"-{opt}"]
    flags.extend(extra_flags)
    if kind == "ll":
        if "-msimd128" not in flags:
            flags.append("-msimd128")
        flags.extend(["-x", "ir"])
    elif kind == "cpp":
        flags.append("-std=c++20")
    flags.append("-c")
    if kind != "ll":
        for include in include_dirs:
            flags.extend(["-I", str(include)])
    return tuple(flags)


def cache_key_for(
    content: bytes,
    flags: Sequence[str],
    *,
    kind: str,
    retarget_version: int | str | None = None,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"playgta6-guest-wasm-v1\0")
    digest.update(kind.encode("utf-8"))
    digest.update(b"\0")
    if kind == "ll":
        version = WASM64_IR_RETARGET_VERSION if retarget_version is None else retarget_version
        digest.update(b"retarget-v")
        digest.update(str(version).encode("utf-8"))
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
            paths.append(item.strip().replace("\\", "/"))
        elif isinstance(item, Mapping):
            for key in ("path", "file", "source", "ll", "cpp", "c"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    paths.append(value.strip().replace("\\", "/"))
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


def resolve_source_root(source: Path) -> Path:
    """Accept a recomp output directory or a direct path to a manifest JSON file."""
    if source.is_file() and source.name in {BUILD_MANIFEST_FILE, PROGRAM_MANIFEST_FILE}:
        return source.parent
    return source


def _instruction_counts(data: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not isinstance(data, Mapping):
        return counts
    lift = data.get("lift")
    if not isinstance(lift, Mapping):
        return counts
    shards = lift.get("shards")
    if not isinstance(shards, list):
        return counts
    for shard in shards:
        if not isinstance(shard, Mapping):
            continue
        rel = shard.get("ll") or shard.get("path")
        instr = shard.get("instructions")
        if isinstance(rel, str) and isinstance(instr, int):
            counts[rel.replace("\\", "/")] = instr
    return counts


def _module_name_from_entry(item: Any) -> str | None:
    if isinstance(item, str) and item.strip():
        return item.strip().replace("\\", "/").rstrip("/")
    if not isinstance(item, Mapping):
        return None
    for key in ("module", "name", "dir"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            text = value.strip().replace("\\", "/").rstrip("/")
            if text.endswith(f"/{BUILD_MANIFEST_FILE}"):
                text = text[: -len(BUILD_MANIFEST_FILE)].rstrip("/")
            return text
    return None


def is_shard_unit(unit: SourceUnit) -> bool:
    name = Path(unit.unit_id.replace("\\", "/")).name
    return unit.kind == "ll" and name.startswith("shard-")


def sample_units(units: Sequence[SourceUnit], count: int) -> list[SourceUnit]:
    """`count` shards spread evenly over the set (registration / thunks are not sampled)."""
    shards = [unit for unit in units if is_shard_unit(unit)]
    if count <= 0:
        return list(units)
    if not shards:
        return []
    if count >= len(shards):
        return list(shards)
    step = len(shards) / count
    picked: list[SourceUnit] = []
    seen: set[str] = set()
    for index in range(count):
        unit = shards[int(index * step)]
        if unit.unit_id in seen:
            continue
        seen.add(unit.unit_id)
        picked.append(unit)
    return picked


def _discover_tree_sources(
    root: Path,
    *,
    manifest_name: str = BUILD_MANIFEST_FILE,
    module: str | None = None,
) -> list[SourceUnit]:
    found: dict[str, SourceUnit] = {}
    instructions: dict[str, int] = {}

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
        instr = instructions.get(unit_id, 0)
        found[unit_id] = SourceUnit(
            unit_id=unit_id,
            path=resolved,
            kind=kind,
            module=module,
            instructions=instr,
        )

    manifest_path = root / manifest_name
    if manifest_path.is_file():
        try:
            data = load_build_manifest(manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            raise BuildGuestWasmError(f"invalid {manifest_name}: {exc}") from exc
        instructions.update(_instruction_counts(data))
        for rel in _collect_manifest_paths(data):
            add_file((root / rel).resolve() if not Path(rel).is_absolute() else Path(rel))

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for name in filenames:
            add_file(Path(dirpath) / name)

    units = list(found.values())
    units.sort(key=lambda u: u.unit_id)
    return units


def discover_program_sources(root: Path, data: Any) -> list[SourceUnit]:
    """Discover every module plus program-level export / descriptor sources."""
    found: dict[str, SourceUnit] = {}
    modules: list[str] = []
    if isinstance(data, Mapping):
        raw_modules = data.get("modules")
        if isinstance(raw_modules, list):
            for item in raw_modules:
                name = _module_name_from_entry(item)
                if name and name not in modules:
                    modules.append(name)

    for name in modules:
        module_dir = root / name
        if not module_dir.is_dir():
            continue
        for unit in _discover_tree_sources(module_dir.resolve(), module=name):
            unit_id = f"{name}/{unit.unit_id}"
            found[unit_id] = SourceUnit(
                unit_id=unit_id,
                path=unit.path,
                kind=unit.kind,
                module=name,
                instructions=unit.instructions,
            )

    program_rels: list[str] = []
    for rel in PROGRAM_LEVEL_UNITS:
        if rel not in program_rels:
            program_rels.append(rel)
    if isinstance(data, Mapping):
        module_prefixes = tuple(f"{name}/" for name in modules)
        for rel in _collect_manifest_paths(data):
            if rel in modules or rel.startswith(module_prefixes):
                continue
            if rel not in program_rels:
                program_rels.append(rel)

    resolved_root = root.resolve()
    for rel in program_rels:
        path = (root / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
        if not path.is_file():
            continue
        kind = _kind_for_suffix(path.suffix)
        if kind is None:
            continue
        try:
            path.relative_to(resolved_root)
        except ValueError:
            continue
        unit_id = posix_rel(path, resolved_root)
        found[unit_id] = SourceUnit(
            unit_id=unit_id,
            path=path,
            kind=kind,
            module="program",
        )

    units = list(found.values())
    units.sort(key=lambda u: u.unit_id)
    return units


def discover_sources(source_dir: Path, *, manifest_name: str = BUILD_MANIFEST_FILE) -> list[SourceUnit]:
    """Discover sharded `.ll` / `.c` / `.cpp` units from a recomp output directory."""
    root = resolve_source_root(source_dir)
    if not root.is_dir():
        raise BuildGuestWasmError(f"source directory not found: {root.name}")

    resolved = root.resolve()
    program_path = resolved / PROGRAM_MANIFEST_FILE
    if source_dir.is_file() and source_dir.name == PROGRAM_MANIFEST_FILE:
        program_path = source_dir.resolve()
    if program_path.is_file():
        try:
            data = load_build_manifest(program_path)
        except (OSError, json.JSONDecodeError) as exc:
            raise BuildGuestWasmError(f"invalid {PROGRAM_MANIFEST_FILE}: {exc}") from exc
        return discover_program_sources(resolved, data)

    return _discover_tree_sources(resolved, manifest_name=manifest_name)


def cache_object_path(cache_dir: Path, key: str, unit_id: str) -> Path:
    stem = Path(unit_id.replace("/", "__")).stem
    return cache_dir / key[:2] / f"{stem}-{key[:16]}.o"


def cache_retargeted_ir_path(cache_dir: Path, key: str, unit_id: str | None = None) -> Path:
    if unit_id:
        stem = Path(unit_id.replace("/", "__")).stem
        return cache_dir / key[:2] / f"{stem}-{key[:16]}.retargeted.ll"
    return cache_dir / f"{key}.retargeted.ll"


def build_compile_argv(
    compiler: Path | str,
    source: Path,
    output: Path,
    flags: Sequence[str],
) -> tuple[str, ...]:
    return (str(compiler), *flags, str(source), "-o", str(output))


def should_use_response_file(object_count: int, requested: bool | None) -> bool:
    if requested is False:
        return False
    if requested is True:
        return True
    return object_count > RESPONSE_FILE_THRESHOLD


def is_program_build(source_root: Path, units: Sequence[SourceUnit]) -> bool:
    """Return True when building a multi-module R12 program tree."""
    if (source_root / PROGRAM_MANIFEST_FILE).is_file():
        return True
    for unit in units:
        if Path(unit.unit_id.replace("\\", "/")).name == "pg6_program.cpp":
            return True
    return False


def _format_rsp_path(item: Path | str) -> str:
    """Normalize a response-file object path to POSIX slashes; quote if it contains whitespace."""
    normalized = str(item).replace("\\", "/")
    if any(ch.isspace() for ch in normalized):
        escaped = normalized.replace('"', '\\"')
        return f'"{escaped}"'
    return normalized


def write_link_response_file(path: Path, objects: Sequence[Path | str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_format_rsp_path(item) for item in objects]
    text = "\n".join(lines)
    if lines:
        text += "\n"
    path.write_text(text, encoding="utf-8")
    return path


def resolve_link_flags(
    *,
    opt: str = "O2",
    strip: bool = False,
    pthread_pool_size: int | None = None,
    extra_link_flags: Sequence[str] = (),
) -> tuple[str, ...]:
    """Compute effective linker flags for em++ from defaults and build options."""
    if pthread_pool_size is not None:
        if isinstance(pthread_pool_size, bool) or pthread_pool_size < 1:
            raise BuildGuestWasmError("--pthread-pool-size must be >= 1")

    flags: list[str] = []
    replaced_pool = False
    for flag in LINK_FLAGS:
        if pthread_pool_size is not None and flag.startswith(PTHREAD_POOL_SIZE_PREFIX):
            flags.append(f"{PTHREAD_POOL_SIZE_PREFIX}{pthread_pool_size}")
            replaced_pool = True
        else:
            flags.append(flag)
    if pthread_pool_size is not None and not replaced_pool:
        flags.append(f"{PTHREAD_POOL_SIZE_PREFIX}{pthread_pool_size}")

    if opt in LINK_OPT_LEVELS:
        opt_flag = f"-{opt}"
        if opt_flag not in flags:
            flags.append(opt_flag)

    if strip:
        for strip_flag in STRIP_LINK_FLAGS:
            if strip_flag not in flags:
                flags.append(strip_flag)

    for extra in extra_link_flags:
        if extra:
            flags.append(extra)

    return tuple(flags)


def check_v8_wasm_size(
    wasm_bytes: int,
    *,
    limit: int | None = None,
) -> tuple[bool, str | None]:
    """Check whether a linked runtime.wasm exceeds Chrome/V8's 1 GiB module cap."""
    max_bytes = V8_MAX_WASM_MODULE_BYTES if limit is None else limit
    if wasm_bytes <= max_bytes:
        return False, None
    mib = wasm_bytes / (1024 * 1024)
    over_mib = (wasm_bytes - max_bytes) / (1024 * 1024)
    warning = (
        f"{RUNTIME_WASM} size is {wasm_bytes} bytes ({mib:.2f} MiB), exceeding "
        f"Chrome/V8's 1 GiB ({max_bytes} bytes) WebAssembly.instantiate() module size limit "
        f"by {over_mib:.2f} MiB; rebuild with --strip and/or --opt Oz (or --opt Os) "
        f"to keep the module under 1 GiB"
    )
    return True, warning


def build_link_argv(
    linker: Path | str,
    objects: Sequence[Path],
    *,
    output_js: Path,
    runtime_libs: Sequence[Path] = (),
    link_flags: Sequence[str] | None = None,
    extra_flags: Sequence[str] = (),
    response_file: Path | None = None,
    opt: str = "O2",
    strip: bool = False,
    pthread_pool_size: int | None = None,
) -> tuple[str, ...]:
    if link_flags is None:
        effective_flags = resolve_link_flags(
            opt=opt,
            strip=strip,
            pthread_pool_size=pthread_pool_size,
            extra_link_flags=extra_flags,
        )
    else:
        effective_flags = (*link_flags, *extra_flags)
    argv: list[str] = [str(linker), *effective_flags]
    if response_file is not None:
        rsp_arg = str(response_file).replace("\\", "/")
        argv.append(f"@{rsp_arg}")
    else:
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
    started = time.monotonic()
    content = unit.path.read_bytes()
    key = cache_key_for(content, flags, kind=unit.kind)
    object_path = cache_object_path(cache_dir, key, unit.unit_id)
    argv = build_compile_argv(compiler, unit.path, object_path, flags)
    display_argv = tuple(sanitize_argv(argv))

    def _bytes(path: Path | None) -> int:
        if path is None or not path.is_file():
            return 0
        try:
            return path.stat().st_size
        except OSError:
            return 0

    if object_path.is_file():
        return UnitBuildResult(
            unit=unit,
            cache_key=key,
            cache_hit=True,
            object_path=object_path,
            argv=display_argv,
            status="hit",
            seconds=time.monotonic() - started,
            object_bytes=_bytes(object_path),
        )

    if dry_run:
        return UnitBuildResult(
            unit=unit,
            cache_key=key,
            cache_hit=False,
            object_path=object_path,
            argv=display_argv,
            status="planned",
            seconds=time.monotonic() - started,
        )

    object_path.parent.mkdir(parents=True, exist_ok=True)
    retarget_path: Path | None = None
    if unit.kind == "ll":
        ir_text = content.decode("utf-8", errors="surrogateescape")
        retargeted_text = retarget_wasm64_ir(ir_text)
        if retargeted_text != ir_text:
            retarget_path = cache_retargeted_ir_path(cache_dir, key, unit.unit_id)
            retarget_path.parent.mkdir(parents=True, exist_ok=True)
            retarget_path.write_bytes(retargeted_text.encode("utf-8", errors="surrogateescape"))
            argv = build_compile_argv(compiler, retarget_path, object_path, flags)
            display_argv = tuple(sanitize_argv(argv))
    try:
        result = run_command(argv)
    finally:
        if retarget_path is not None:
            try:
                retarget_path.unlink(missing_ok=True)
            except OSError:
                pass
    elapsed = time.monotonic() - started
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
            seconds=elapsed,
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
            seconds=elapsed,
        )
    return UnitBuildResult(
        unit=unit,
        cache_key=key,
        cache_hit=False,
        object_path=object_path,
        argv=display_argv,
        status="compiled",
        seconds=elapsed,
        object_bytes=_bytes(object_path),
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


def _object_bytes(path: Path | None) -> int:
    if path is None or not path.is_file():
        return 0
    try:
        return path.stat().st_size
    except OSError:
        return 0


def extrapolate_sample(
    all_units: Sequence[SourceUnit],
    sampled: Sequence[UnitBuildResult],
    *,
    jobs: int,
) -> dict[str, Any]:
    all_shards = [unit for unit in all_units if is_shard_unit(unit)]
    sampled_shards = [item for item in sampled if is_shard_unit(item.unit)]
    sample_count = len(sampled_shards) or 1
    total_shards = len(all_shards)
    sample_bytes = sum(item.object_bytes or _object_bytes(item.object_path) for item in sampled_shards)
    sample_cpu = sum(item.seconds for item in sampled_shards)
    all_instr = sum(unit.instructions for unit in all_shards)
    sample_instr = sum(item.unit.instructions for item in sampled_shards)
    if all_instr > 0 and sample_instr > 0:
        factor = all_instr / sample_instr
    else:
        factor = total_shards / sample_count if total_shards else 1.0
    cpu = sample_cpu * factor
    return {
        "sample_units": len(sampled_shards),
        "total_shards": total_shards,
        "object_bytes": int(sample_bytes * factor),
        "compile_seconds": round(cpu, 3),
        "wall_seconds_at_jobs": round(cpu / max(1, jobs), 3),
    }


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
    strip: bool = False,
    pthread_pool_size: int | None = None,
    dry_run: bool = False,
    check_only: bool = False,
    verify: bool = False,
    run_command: CommandRunner | None = None,
    which_fn: WhichFn | None = None,
    env: Mapping[str, str] | None = None,
    sample: int = 0,
    response_file: bool | None = None,
    link_guest_boot: bool = False,
    guest_boot_entry: Path | None = None,
) -> BuildReport:
    """Discover, compile (in parallel), and link guest shards into runtime.{js,wasm}."""
    opt_level = parse_opt_level(opt)
    worker_count = jobs if jobs is not None else (os.cpu_count() or 1)
    if worker_count < 1:
        raise BuildGuestWasmError("--jobs must be >= 1")
    if sample < 0:
        raise BuildGuestWasmError("--sample must be >= 0")
    effective_link_flags = resolve_link_flags(
        opt=opt_level,
        strip=strip,
        pthread_pool_size=pthread_pool_size,
        extra_link_flags=extra_link_flags,
    )

    runner = run_command or default_run_command
    toolchain = discover_emscripten(repo_root=repo_root, env=env, which_fn=which_fn)
    if check_only:
        mode = "check-only"
    elif sample > 0:
        mode = "sample"
    elif dry_run:
        mode = "dry-run"
    else:
        mode = "build"
    report = BuildReport(
        ok=toolchain.ok,
        mode=mode,
        toolchain=toolchain,
        jobs=worker_count,
        opt=opt_level,
        compile_flags=(*COMPILE_WASM_FLAGS, f"-{opt_level}"),
        link_flags=effective_link_flags,
        linked_guest_boot=link_guest_boot,
        sample=sample,
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

    source_root = resolve_source_root(source_dir)
    all_units = discover_sources(source_dir)
    program_build = is_program_build(source_root, all_units)
    if link_guest_boot:
        entry = guest_boot_entry or (repo_root / DEFAULT_GUEST_BOOT_ENTRY_REL)
        if not entry.is_file():
            report.ok = False
            report.errors.append(f"guest boot entrypoint not found: {entry.name}")
            return report

        def _make_boot_unit(path: Path) -> SourceUnit:
            resolved_path = path.resolve()
            resolved_repo = repo_root.resolve()
            boot_id = posix_rel(resolved_path, resolved_repo)
            if boot_id == resolved_path.name and resolved_path != (resolved_repo / resolved_path.name):
                boot_id = "runtime/" + resolved_path.name
            return SourceUnit(
                unit_id=boot_id.replace("\\", "/"),
                path=resolved_path,
                kind="cpp",
                module="boot",
            )

        entry_unit = _make_boot_unit(entry)
        all_units.append(entry_unit)

        wasm_boot = repo_root / DEFAULT_WASM_BOOT_REL
        if wasm_boot.is_file() and wasm_boot.resolve() != entry_unit.path:
            all_units.append(_make_boot_unit(wasm_boot))

        all_units.sort(key=lambda u: u.unit_id)

    if not all_units:
        report.ok = False
        report.errors.append("no .ll/.c/.cpp translation units found")
        return report

    units = sample_units(all_units, sample) if sample > 0 else all_units
    if sample > 0 and not units:
        report.ok = False
        report.errors.append("no shard-*.ll units available to sample")
        return report

    cache_root = (cache_dir or (repo_root / DEFAULT_CACHE_DIR_REL)).resolve()
    output_dir = (out_dir or (repo_root / DEFAULT_OUT_DIR_REL)).resolve()
    include_dirs = [repo_root / "runtime", source_root.resolve()]
    include_dirs = [p for p in include_dirs if p.is_dir()]

    boot_defines: tuple[str, ...] = (
        (BOOT_LINKED_GUEST_DEFINE, BOOT_PROGRAM_DEFINE)
        if program_build
        else (BOOT_LINKED_GUEST_DEFINE,)
    )
    planned: list[UnitBuildResult] = []
    compile_tasks: list[tuple[SourceUnit, Path, tuple[str, ...]]] = []
    for unit in units:
        compiler = toolchain.compiler_for(unit.kind)
        if compiler is None:
            report.ok = False
            report.errors.append(f"no compiler for {unit.unit_id} ({unit.kind})")
            continue
        extra = boot_defines if link_guest_boot and unit.module == "boot" else ()
        flags = compile_flags_for(
            opt=opt_level,
            kind=unit.kind,
            include_dirs=include_dirs,
            extra_flags=extra,
        )
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
                    object_bytes=_object_bytes(object_path),
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

    compile_started = time.monotonic()
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

    ordered = [compiled_by_id[u.unit_id] for u in units if u.unit_id in compiled_by_id]
    report.sources = ordered
    report.cache_hits = sum(1 for item in ordered if item.cache_hit)
    report.cache_misses = sum(1 for item in ordered if not item.cache_hit)
    report.compile_seconds = round(time.monotonic() - compile_started, 3)
    report.object_bytes = sum(
        item.object_bytes or _object_bytes(item.object_path) for item in ordered
    )

    compile_errors = [item.error for item in ordered if item.status == "error" and item.error]
    report.errors.extend(compile_errors)

    if sample > 0:
        report.extrapolated = extrapolate_sample(all_units, ordered, jobs=worker_count)
        if compile_errors:
            report.ok = False
            return report
        report.ok = True
        return report

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
    planned_objects = objects if objects else [Path(item.unit.unit_id).with_suffix(".o") for item in ordered]
    rsp_path: Path | None = None
    if should_use_response_file(len(planned_objects), response_file):
        rsp_path = cache_root / RESPONSE_FILE_NAME
        write_link_response_file(rsp_path, planned_objects)
        report.response_file = RESPONSE_FILE_NAME
    raw_link = build_link_argv(
        linker,
        planned_objects,
        output_js=output_js,
        runtime_libs=runtime_libs,
        link_flags=effective_link_flags,
        response_file=rsp_path,
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

    report.wasm_bytes = _object_bytes(output_wasm)
    exceeds_limit, size_warning = check_v8_wasm_size(report.wasm_bytes)
    report.exceeds_v8_1gib_limit = exceeds_limit
    if size_warning is not None:
        report.warnings.append(size_warning)

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
    if report.linked_guest_boot:
        lines.append("  Linked guest boot: yes")
    if report.response_file:
        lines.append(f"  Response file: @{report.response_file}")
    if report.sample:
        lines.append(f"  Sample: {report.sample} shards")
    if report.extrapolated:
        extra = report.extrapolated
        lines.append(
            "  Extrapolated: "
            f"{extra.get('compile_seconds', 0)}s compile, "
            f"{extra.get('object_bytes', 0)} bytes .o "
            f"(from {extra.get('sample_units', 0)} of {extra.get('total_shards', 0)} shards)"
        )
    if report.sources:
        lines.append("  Units:")
        for item in report.sources:
            tag = item.status.upper()
            module = f" [{item.unit.module}]" if item.unit.module else ""
            lines.append(f"    [{tag}] {item.unit.unit_id} ({item.unit.kind}){module}")
    if report.link_argv:
        lines.append("  Link: " + " ".join(report.link_argv))
    if report.outputs:
        lines.append("  Outputs: " + ", ".join(report.outputs))
    if report.wasm_bytes:
        mib = report.wasm_bytes / (1024 * 1024)
        limit_note = " [EXCEEDS V8 1 GiB LIMIT]" if report.exceeds_v8_1gib_limit else ""
        lines.append(f"  Wasm size: {report.wasm_bytes} bytes ({mib:.2f} MiB){limit_note}")
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
        help="Optimization level: O0, O1, O2 (default), O3, Os, or Oz.",
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
        "--strip",
        action="store_true",
        help=(
            "Strip debug/name sections at link time (-Wl,--strip-all -sASSERTIONS=0) "
            "to reduce runtime.wasm size."
        ),
    )
    parser.add_argument(
        "--extra-link-flag",
        action="append",
        default=[],
        dest="extra_link_flags",
        metavar="FLAG",
        help=(
            "Additional linker flag passed to em++ (repeatable; e.g. "
            "--extra-link-flag=-sINITIAL_MEMORY=2GB)."
        ),
    )
    parser.add_argument(
        "--pthread-pool-size",
        type=int,
        default=None,
        metavar="N",
        help="Override -sPTHREAD_POOL_SIZE=N on the link line (must be >= 1; default: 4).",
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
    parser.add_argument(
        "--response-file",
        action="store_true",
        help=(
            f"Pass objects to the linker via @{RESPONSE_FILE_NAME} "
            f"(auto-enabled when linking more than {RESPONSE_FILE_THRESHOLD} objects)."
        ),
    )
    parser.add_argument(
        "--no-response-file",
        action="store_true",
        help="Put every object path on the em++ command line even for large links.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        metavar="N",
        help="Compile N evenly spaced shard-*.ll units, skip linking, and extrapolate time/size.",
    )
    parser.add_argument(
        "--link-guest-boot",
        action="store_true",
        help=(
            f"Compile runtime/main.cpp and runtime/wasm_boot.cpp with {BOOT_LINKED_GUEST_DEFINE} "
            f"(plus {BOOT_PROGRAM_DEFINE} for multi-module program builds) and include them in the wasm64 link."
        ),
    )
    parser.add_argument(
        "--guest-boot-entry",
        default=None,
        help="Entrypoint source for --link-guest-boot (default: runtime/main.cpp).",
    )
    return parser


def _normalize_cli_argv(
    argv: Sequence[str],
    parser: argparse.ArgumentParser,
) -> list[str]:
    """Allow `--extra-link-flag -Wl,...` in addition to `--extra-link-flag=-Wl,...`."""
    known_options: set[str] = set()
    for action in parser._actions:
        known_options.update(action.option_strings)

    out: list[str] = []
    idx = 0
    items = list(argv)
    while idx < len(items):
        token = items[idx]
        if token == "--":
            out.extend(items[idx:])
            break
        if (
            token == "--extra-link-flag"
            and idx + 1 < len(items)
            and items[idx + 1] not in known_options
            and items[idx + 1] != "--"
        ):
            out.append(f"--extra-link-flag={items[idx + 1]}")
            idx += 2
            continue
        out.append(token)
        idx += 1
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(_normalize_cli_argv(raw_argv, parser))
    repo_root = Path(args.repo_root).resolve() if args.repo_root else DEFAULT_REPO_ROOT

    source = Path(args.source_dir) if args.source_dir else None
    cache_dir = Path(args.cache_dir) if args.cache_dir else repo_root / DEFAULT_CACHE_DIR_REL
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / DEFAULT_OUT_DIR_REL
    runtime_lib = Path(args.runtime_lib) if args.runtime_lib else None
    extra_objects = [Path(p) for p in args.extra_objects]
    if args.response_file and args.no_response_file:
        parser.error("use either --response-file or --no-response-file, not both")
    response_file: bool | None
    if args.response_file:
        response_file = True
    elif args.no_response_file:
        response_file = False
    else:
        response_file = None
    guest_boot_entry = Path(args.guest_boot_entry) if args.guest_boot_entry else None

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
            extra_link_flags=args.extra_link_flags,
            strip=args.strip,
            pthread_pool_size=args.pthread_pool_size,
            dry_run=args.dry_run,
            check_only=args.check_only,
            verify=args.verify,
            sample=args.sample,
            response_file=response_file,
            link_guest_boot=args.link_guest_boot,
            guest_boot_entry=guest_boot_entry,
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
