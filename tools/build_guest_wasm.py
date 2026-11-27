#!/usr/bin/env python3
"""Parallel sharded LLVM IR / C++ → Wasm64 Emscripten build orchestrator (Track 1 / R13, R21, R24, R25, R26, R27, R28, R30, R31, R33, R35, R37d, R37e, R37f; Track 5 / W110; Track 4 / T44).

Takes a `playgta6-recomp build` (R9) output directory of sharded `.ll` / `.c` / `.cpp`
translation units, or a multi-module `playgta6-recomp build-program` (R12) tree with
`program-manifest.json`, compiles dirty shards in parallel with `emcc`/`em++`, caches
`.o` files by SHA-256 of content + compiler flags (plus `runtime/` and generated/sibling
`.h`/`.hpp`/`.inl` header fingerprints for C/C++ units, never `.ll` shards), and links with
`libplaygta6_core.a` into `web/dist/runtime.js` +
`web/dist/runtime.wasm` (optionally splitting `.ll` shards and `functions.ll`
registration tables across `guest_part_*.wasm` `-sSIDE_MODULE=1` modules with `-fPIC`,
`-sMAIN_MODULE=1`, filtered whole-archive, runtime `Module.dynamicLibraries` loading,
post-link V8-cage GC + throttled pthread `CMD_LOAD` JS optimization, auto-detected
`Pg6ShardDispatchTable` shard-dispatch support, pure-Python Wasm section telemetry,
and `guest_manifest.json` emission).

Opt-in browser-proven split (R37e/R37f): `--split-strategy asymmetric` packs contiguous
shards so the largest side module stays under V8's 1024 MiB cap (default part-0 ceiling
994 MiB, reproducing the live ~994/171 MiB two-part shape) and links side modules with
`-sSIDE_MODULE=2` plus `llvm-nm`-computed cross-module `-Wl,--export=<sym>` response
files (`side_part_<i>_exports.rsp`) and `-Wl,--export-if-defined=__wasm_call_ctors` so
unused internal `sub_*` exports are pruned without stripping live cross-part or entry
functions. Existing callers that omit the flag keep `-sSIDE_MODULE=1` and the prior
balanced/contiguous strategies.

Flags match `runtime/CMakeLists.txt` and `runtime/build_wasm.{sh,ps1}`:

    -sMEMORY64=1 -pthread -sMODULARIZE=1 -lexports.js ...

Pure Python 3.12 standard library only. Usable as a library and as a CLI:

    python tools/build_guest_wasm.py <recomp-out-dir>
        [--jobs N] [--cache-dir work/wasm_obj_cache] [--out-dir web/dist]
        [--opt O0|O1|O2|O3|Os|Oz] [--runtime-lib path/to/libplaygta6_core.a]
        [--strip] [--no-link-opt] [--shrink-largest-shards N]
        [--wasm-exceptions] [--initial-memory SIZE] [--stack-size SIZE]
        [--assertions 0|1] [--browser-boot-exports] [--split-modules N]
        [--split-strategy balanced|contiguous|asymmetric|capped]
        [--max-side-module-mib MIB] [--proxy-to-pthread]
        [--max-concurrent-pthread-loads N] [--node-max-old-space-mb MB]
        [--pthread-pool-size N] [--extra-link-flag FLAG]
        [--verify-max-table-imports N] [--response-file] [--sample N]
        [--link-guest-boot] [--dry-run] [--check-only] [--verify]
        [--json] [--sanitize]
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import hashlib
import inspect
import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import emsdk_locator  # noqa: E402

SCHEMA_VERSION = 1
DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR_REL = Path("work") / "wasm_obj_cache"
DEFAULT_OUT_DIR_REL = Path("web") / "dist"
DEFAULT_RUNTIME_LIB_REL = Path("runtime") / "build-wasm64" / "libplaygta6_core.a"
EMSDK_UPSTREAM_REL = Path("work") / "emsdk" / "upstream" / "emscripten"
RUNTIME_JS = "runtime.js"
RUNTIME_WASM = "runtime.wasm"
GUEST_MANIFEST_FILE = "guest_manifest.json"
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
HEADER_SUFFIXES: frozenset[str] = frozenset({".h", ".hpp", ".inl"})
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
WASM64_IR_RETARGET_VERSION = 2
WASM64_SOURCE_TRIPLE = "wasm64-unknown-unknown"
WASM64_EMSCRIPTEN_TRIPLE = "wasm64-unknown-emscripten"

# Compile flags applied to every translation unit (plus `-O*` from --opt).
COMPILE_WASM_FLAGS: tuple[str, ...] = ("-sMEMORY64=1", "-pthread")

# Target features every object of the shared-memory pthread build must carry (what `emcc -pthread` implies).
# Passed explicitly to every compile, and merged into the `"target-features"` attribute of lifted IR (function
# attributes REPLACE the command-line features), so wasm-ld's feature check passes without --no-check-features.
SHARED_MEMORY_TARGET_FEATURES: tuple[str, ...] = ("+atomics", "+bulk-memory", "+mutable-globals")
SHARED_MEMORY_COMPILE_FLAGS: tuple[str, ...] = ("-matomics", "-mbulk-memory", "-mmutable-globals")
_IR_TARGET_FEATURES_RE = re.compile(r'"target-features"="([^"]*)"')

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
WASM_EXCEPTIONS_FLAG = "-fwasm-exceptions"
INITIAL_MEMORY_PREFIX = "-sINITIAL_MEMORY="
STACK_SIZE_PREFIX = "-sSTACK_SIZE="
DEFAULT_PTHREAD_STACK_SIZE_PREFIX = "-sDEFAULT_PTHREAD_STACK_SIZE="
ASSERTIONS_PREFIX = "-sASSERTIONS="
BROWSER_BOOT_INCOMING_MODULE_JS_API = (
    "locateFile,instantiateWasm,mainScriptUrlOrBlob,print,printErr,"
    "preRun,onRuntimeInitialized,noInitialRun,setStatus,onExit"
)
SPLIT_INCOMING_MODULE_JS_API = (
    f"{BROWSER_BOOT_INCOMING_MODULE_JS_API},dynamicLibraries"
)
BROWSER_BOOT_EXPORTED_RUNTIME_METHODS = (
    "wasmMemory,FS,HEAPU8,addRunDependency,removeRunDependency"
)
BROWSER_BOOT_EXPORTED_FUNCTIONS = (
    "_main,_playgta6_get_input_state_ptr,_playgta6_audio_ring"
)
BROWSER_BOOT_EXPORT_LINK_FLAGS: tuple[str, ...] = (
    f"-sINCOMING_MODULE_JS_API={BROWSER_BOOT_INCOMING_MODULE_JS_API}",
    f"-sEXPORTED_RUNTIME_METHODS={BROWSER_BOOT_EXPORTED_RUNTIME_METHODS}",
    f"-sEXPORTED_FUNCTIONS={BROWSER_BOOT_EXPORTED_FUNCTIONS}",
)

# Chrome/V8 enforces hard WebAssembly instantiation and per-function limits:
#   - kV8MaxWasmModuleSize = 1 GiB (1,073,741,824 bytes)
#   - kV8MaxWasmFunctionSize = 7,654,321 bytes (~7.65 MB / ~7.30 MiB)
#   - kV8MaxWasmFunctionLocals = 50,000 locals per function
V8_MAX_WASM_MODULE_BYTES = 1024 * 1024 * 1024
V8_MAX_WASM_FUNCTION_BYTES = 7_654_321
V8_MAX_WASM_FUNCTION_LOCALS = 50_000
LARGE_FUNC_BODY_THRESHOLD_BYTES = 1024 * 1024
V8_MODULE_WARN_FRACTION = 0.90
V8_FUNC_BODY_WARN_FRACTION = 0.75
V8_FUNC_LOCALS_WARN_FRACTION = 0.80

# Split-module dynamic linking (R28/R31/R37e): -fPIC + -sMAIN_MODULE=1 + -sSIDE_MODULE=1
# (default) or -sSIDE_MODULE=2 (asymmetric / pruned), with registration tables in an extra
# side module, filtered whole-archive runtime lib, growable 16 GiB linear memory, and
# Module.dynamicLibraries loaded by web/loader.js.
PIC_FLAG = "-fPIC"
MAIN_MODULE_FLAG = "-sMAIN_MODULE=1"
SIDE_MODULE_FLAG = "-sSIDE_MODULE=1"
SIDE_MODULE_PRUNED_FLAG = "-sSIDE_MODULE=2"
SPLIT_DEFAULT_INITIAL_MEMORY = "64MB"
# Pre-warm enough em-pthread workers for the game's 43+ guest threads so split-module
# browser builds avoid cold worker spawns and side-module re-instantiation boot timeouts.
SPLIT_DEFAULT_PTHREAD_POOL_SIZE = 48
SPLIT_RUNTIME_ARCHIVE_EXCLUDES: tuple[str, ...] = ("wasm_boot.cpp.o", "main.cpp.o")
SIDE_MODULE_CTOR_EXPORT_FLAG = "-Wl,--export-if-defined=__wasm_call_ctors"
SIDE_MODULE_LINK_FLAGS: tuple[str, ...] = (
    "-sMEMORY64=1",
    "-pthread",
    SIDE_MODULE_FLAG,
)
SIDE_MODULE_PRUNED_LINK_FLAGS: tuple[str, ...] = (
    "-sMEMORY64=1",
    "-pthread",
    SIDE_MODULE_PRUNED_FLAG,
    SIDE_MODULE_CTOR_EXPORT_FLAG,
)
SIDE_MODULE_NAME_TEMPLATE = "guest_part_{index}.wasm"
_SIDE_MODULE_FILENAME_RE = re.compile(r"^guest_part_(\d+)\.wasm$")
SIDE_RESPONSE_FILE_TEMPLATE = "side_part_{index}.rsp"
SIDE_EXPORTS_RESPONSE_FILE_TEMPLATE = "side_part_{index}_exports.rsp"
LLVM_NM_DEFINED_GLOBAL_TYPES: frozenset[str] = frozenset({"T", "W", "D", "B", "R", "C", "V"})
LLVM_NM_UNDEFINED_TYPES: frozenset[str] = frozenset({"U", "w"})
LLVM_NM_MAX_CMD_CHARS = 16_384
SIDE_MODULE_MIN_INPUT_BYTES_FOR_SIZE_GUARD = 64 * 1024  # 64 KiB
SIDE_MODULE_MIN_OUTPUT_BYTES = 4096  # 4 KiB (catches ~676 B stripped side modules)
SIDE_MODULE_REQUIRED_EXPORT_NAMES: frozenset[str] = frozenset(
    {
        "__wasm_call_ctors",
        "pg6_import_thunks",
        "pg6_import_thunk_count",
        "pg6_export_thunks",
        "pg6_export_thunk_count",
    }
)
SPLIT_STRATEGIES: frozenset[str] = frozenset({"balanced", "contiguous", "asymmetric", "capped"})
DEFAULT_SPLIT_STRATEGY = "balanced"
# Proven Chrome two-part ceiling: pack part_0 up to ~994 MiB so part_1 lands near ~171 MiB.
DEFAULT_MAX_SIDE_MODULE_MIB = 994
# `capped` strategy: every part stays <= this (leaves ~124 MiB margin under the loader/V8 1024 MiB cap).
DEFAULT_CAPPED_SIDE_MODULE_MIB = 900
MAX_SIDE_MODULE_MIB_LIMIT = V8_MAX_WASM_MODULE_BYTES // (1024 * 1024)
DEFAULT_SPLIT_NODE_MAX_OLD_SPACE_MB = 16384
PROXY_TO_PTHREAD_FLAG = "-sPROXY_TO_PTHREAD=1"
DEFAULT_MAX_CONCURRENT_PTHREAD_LOADS = 2
SPLIT_PTHREAD_OPT_MARKER = "/* __playgta6_split_pthread_opt_v1__ */"

# Shard-level indirect-call dispatch (R37: Pg6ShardDispatchTable).
INDIRECT_DISPATCH_MODES: frozenset[str] = frozenset({"table", "shard"})
DEFAULT_INDIRECT_DISPATCH = "table"
SHARD_DISPATCH_FN_PREFIX = "__playgta6_dispatch_shard_"
SHARD_DISPATCH_TABLE_PREFIX = "__playgta6_shard_dispatch_"
SIDE_MODULE_REQUIRED_EXPORT_PREFIXES: tuple[str, ...] = (
    "pg6_register_",
    "pg6_dispatch_",
    "pg6_entry_",
    "pg6_import_thunks_",
    "pg6_import_thunk_count_",
    "pg6_export_thunks_",
    "pg6_export_thunk_count_",
    "__playgta6_register_lifted_functions_",
    "__playgta6_lifted_functions_",
    "__playgta6_lifted_function_count_",
    SHARD_DISPATCH_FN_PREFIX,
    SHARD_DISPATCH_TABLE_PREFIX,
)
LEGACY_TABLE_BASELINE_ENTRIES = 437_813

# Pure-Python WebAssembly binary inspection constants (Wasm Core + Memory64 + dylink.0).
WASM_MAGIC = b"\x00asm"
WASM_VERSION = 1
WASM_SECTION_CUSTOM = 0
WASM_SECTION_IMPORT = 2
WASM_SECTION_TABLE = 4
WASM_SECTION_EXPORT = 7
WASM_SECTION_CODE = 10
WASM_KNOWN_SECTION_IDS: frozenset[int] = frozenset(range(0, 14))
WASM_DYLINK_MEM_INFO = 1
WASM_REFTYPE_FUNCREF = 0x70
WASM_LIMITS_FLAG_HAS_MAX = 0x01
WASM_LIMITS_FLAG_MEMORY64 = 0x04
WASM_TABLE_EXTENDED_PREFIX = 0x40

VERIFY_FLAGS: tuple[str, ...] = (
    "--require-memory64",
    "--require-shared-memory",
    "--require-host-imports",
)

OPT_LEVELS: frozenset[str] = frozenset({"O0", "O1", "O2", "O3", "Os", "Oz"})

_SIZE_TOKEN_RE = re.compile(r"^([0-9]+)([A-Za-z]*)$")
_NODE_MAX_OLD_SPACE_RE = re.compile(
    r"(?:^|\s+)--max[-_]old[-_]space[-_]size(?:=\S+|\s+\S+)(?=\s|$)"
)

_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"'<>|,;()\[\]]+")
_UNC_PATH_RE = re.compile(r"\\\\[^\s\"'<>|,;()\[\]]+")
_POSIX_PATH_RE = re.compile(
    r"/(?:Users|home|tmp|var|opt|root|mnt|private|etc|usr|Library|Volumes)/[^\s\"'<>|,;()\[\]]*"
)
_TARGET_DATALAYOUT_RE = re.compile(r"^[ \t]*target[ \t]+datalayout[ \t]*=")
_TARGET_TRIPLE_WASM64_RE = re.compile(
    r'^([ \t]*target[ \t]+triple[ \t]*=[ \t]*)"wasm64-unknown-unknown"([ \t]*(?:;.*)?)$'
)
_LOCAL_INCLUDE_RE = re.compile(rb'^[ \t]*#[ \t]*include[ \t]*"([^"\r\n]+)"', re.MULTILINE)

CommandRunner = Callable[..., "CommandResult"]
WhichFn = Callable[[str], str | None]


# ---------------------------------------------------------------------------
# Errors / results
# ---------------------------------------------------------------------------


class BuildGuestWasmError(Exception):
    """Raised for user-facing configuration or discovery failures."""


class WasmInspectError(ValueError):
    """Raised when a linked WebAssembly binary is truncated or malformed."""


@dataclass(frozen=True)
class IndirectDispatchDetection:
    """Result of auto-detecting `table` vs `shard` indirect-call dispatch mode."""

    mode: str  # "table" | "shard"
    source: str

    def to_dict(self) -> dict[str, str]:
        return {
            "mode": self.mode,
            "source": self.source,
        }


@dataclass(frozen=True)
class WasmModuleMetrics:
    """Numeric-only post-link section metrics for one built `.wasm` module."""

    name: str
    role: str  # "main" | "side"
    wasm_bytes: int
    func_imports: int = 0
    got_func_imports: int = 0
    got_mem_imports: int = 0
    func_exports: int = 0
    table_size: int = 0
    defined_table_size: int = 0
    dylink_table_size: int | None = None
    dylink_memory_size: int | None = None
    has_dylink: bool = False
    shard_dispatch_exports: int = 0
    shard_table_exports: int = 0
    sub_func_exports: int = 0
    sub_func_imports: int = 0
    got_func_sub_imports: int = 0
    got_func_shard_dispatch_imports: int = 0
    code_func_count: int = 0
    max_func_body_bytes: int = 0
    max_func_locals: int = 0
    large_func_bodies_over_1mib: int = 0

    @property
    def size_mib(self) -> float:
        return round(self.wasm_bytes / (1024 * 1024), 4)

    @property
    def v8_module_headroom_bytes(self) -> int:
        return V8_MAX_WASM_MODULE_BYTES - self.wasm_bytes

    @property
    def v8_module_headroom_mib(self) -> float:
        return round(self.v8_module_headroom_bytes / (1024 * 1024), 4)

    @property
    def v8_module_used_pct(self) -> float:
        return round((self.wasm_bytes / V8_MAX_WASM_MODULE_BYTES) * 100.0, 2)

    @property
    def max_func_body_mib(self) -> float:
        return round(self.max_func_body_bytes / (1024 * 1024), 4)

    @property
    def v8_func_body_headroom_bytes(self) -> int:
        return V8_MAX_WASM_FUNCTION_BYTES - self.max_func_body_bytes

    @property
    def v8_func_locals_headroom(self) -> int:
        return V8_MAX_WASM_FUNCTION_LOCALS - self.max_func_locals

    @property
    def exceeds_v8_module_limit(self) -> bool:
        return self.wasm_bytes > V8_MAX_WASM_MODULE_BYTES

    @property
    def exceeds_v8_func_size_limit(self) -> bool:
        return self.max_func_body_bytes > V8_MAX_WASM_FUNCTION_BYTES

    @property
    def exceeds_v8_locals_limit(self) -> bool:
        return self.max_func_locals > V8_MAX_WASM_FUNCTION_LOCALS

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "wasm_bytes": self.wasm_bytes,
            "size_mib": self.size_mib,
            "func_imports": self.func_imports,
            "got_func_imports": self.got_func_imports,
            "got_mem_imports": self.got_mem_imports,
            "func_exports": self.func_exports,
            "table_size": self.table_size,
            "defined_table_size": self.defined_table_size,
            "dylink_table_size": self.dylink_table_size,
            "dylink_memory_size": self.dylink_memory_size,
            "has_dylink": self.has_dylink,
            "shard_dispatch_exports": self.shard_dispatch_exports,
            "shard_table_exports": self.shard_table_exports,
            "sub_func_exports": self.sub_func_exports,
            "sub_func_imports": self.sub_func_imports,
            "got_func_sub_imports": self.got_func_sub_imports,
            "got_func_shard_dispatch_imports": self.got_func_shard_dispatch_imports,
            "code_func_count": self.code_func_count,
            "max_func_body_bytes": self.max_func_body_bytes,
            "max_func_body_mib": self.max_func_body_mib,
            "max_func_locals": self.max_func_locals,
            "large_func_bodies_over_1mib": self.large_func_bodies_over_1mib,
            "v8_module_headroom_bytes": self.v8_module_headroom_bytes,
            "v8_module_headroom_mib": self.v8_module_headroom_mib,
            "v8_module_used_pct": self.v8_module_used_pct,
            "v8_func_body_headroom_bytes": self.v8_func_body_headroom_bytes,
            "v8_func_locals_headroom": self.v8_func_locals_headroom,
            "exceeds_v8_module_limit": self.exceeds_v8_module_limit,
            "exceeds_v8_func_size_limit": self.exceeds_v8_func_size_limit,
            "exceeds_v8_locals_limit": self.exceeds_v8_locals_limit,
        }

    def to_manifest_dict(self) -> dict[str, Any]:
        """CamelCase numeric-only module metrics for `guest_manifest.json`."""
        return {
            "name": self.name,
            "role": self.role,
            "wasmBytes": self.wasm_bytes,
            "sizeMiB": self.size_mib,
            "funcImports": self.func_imports,
            "gotFuncImports": self.got_func_imports,
            "gotMemImports": self.got_mem_imports,
            "funcExports": self.func_exports,
            "tableSize": self.table_size,
            "definedTableSize": self.defined_table_size,
            "dylinkTableSize": self.dylink_table_size,
            "dylinkMemorySize": self.dylink_memory_size,
            "hasDylink": self.has_dylink,
            "shardDispatchExports": self.shard_dispatch_exports,
            "shardTableExports": self.shard_table_exports,
            "subFuncExports": self.sub_func_exports,
            "subFuncImports": self.sub_func_imports,
            "gotFuncSubImports": self.got_func_sub_imports,
            "gotFuncShardDispatchImports": self.got_func_shard_dispatch_imports,
            "codeFuncCount": self.code_func_count,
            "maxFuncBodyBytes": self.max_func_body_bytes,
            "maxFuncBodyMiB": self.max_func_body_mib,
            "maxFuncLocals": self.max_func_locals,
            "largeFuncBodiesOver1MiB": self.large_func_bodies_over_1mib,
            "v8ModuleHeadroomBytes": self.v8_module_headroom_bytes,
            "v8ModuleHeadroomMiB": self.v8_module_headroom_mib,
            "v8ModuleUsedPct": self.v8_module_used_pct,
            "v8FuncBodyHeadroomBytes": self.v8_func_body_headroom_bytes,
            "v8FuncLocalsHeadroom": self.v8_func_locals_headroom,
            "exceedsV8ModuleLimit": self.exceeds_v8_module_limit,
            "exceedsV8FuncSizeLimit": self.exceeds_v8_func_size_limit,
            "exceedsV8LocalsLimit": self.exceeds_v8_locals_limit,
        }


@dataclass(frozen=True)
class WasmBuildMetrics:
    """Aggregate post-link Wasm section metrics across main + all side modules."""

    indirect_dispatch: str
    modules: tuple[WasmModuleMetrics, ...]

    @property
    def total_wasm_bytes(self) -> int:
        return sum(m.wasm_bytes for m in self.modules)

    @property
    def total_size_mib(self) -> float:
        return round(self.total_wasm_bytes / (1024 * 1024), 4)

    @property
    def total_func_imports(self) -> int:
        return sum(m.func_imports for m in self.modules)

    @property
    def total_got_func_imports(self) -> int:
        return sum(m.got_func_imports for m in self.modules)

    @property
    def total_got_mem_imports(self) -> int:
        return sum(m.got_mem_imports for m in self.modules)

    @property
    def total_func_exports(self) -> int:
        return sum(m.func_exports for m in self.modules)

    @property
    def main_table_size(self) -> int:
        return sum(m.table_size for m in self.modules if m.role == "main")

    @property
    def side_table_size(self) -> int:
        return sum(m.table_size for m in self.modules if m.role == "side")

    @property
    def total_table_size(self) -> int:
        return self.main_table_size + self.side_table_size

    @property
    def side_module_count(self) -> int:
        return sum(1 for m in self.modules if m.role == "side")

    @property
    def shard_dispatch_exports(self) -> int:
        return sum(m.shard_dispatch_exports for m in self.modules)

    @property
    def shard_table_exports(self) -> int:
        return sum(m.shard_table_exports for m in self.modules)

    @property
    def got_func_sub_imports(self) -> int:
        return sum(m.got_func_sub_imports for m in self.modules)

    @property
    def got_func_shard_dispatch_imports(self) -> int:
        return sum(m.got_func_shard_dispatch_imports for m in self.modules)

    @property
    def max_module_bytes(self) -> int:
        return max((m.wasm_bytes for m in self.modules), default=0)

    @property
    def max_module_mib(self) -> float:
        return round(self.max_module_bytes / (1024 * 1024), 4)

    @property
    def max_module_name(self) -> str | None:
        if not self.modules:
            return None
        return max(self.modules, key=lambda m: m.wasm_bytes).name

    @property
    def min_module_headroom_bytes(self) -> int:
        return V8_MAX_WASM_MODULE_BYTES - self.max_module_bytes

    @property
    def min_module_headroom_mib(self) -> float:
        return round(self.min_module_headroom_bytes / (1024 * 1024), 4)

    @property
    def max_func_body_bytes(self) -> int:
        return max((m.max_func_body_bytes for m in self.modules), default=0)

    @property
    def max_func_body_mib(self) -> float:
        return round(self.max_func_body_bytes / (1024 * 1024), 4)

    @property
    def func_body_headroom_bytes(self) -> int:
        return V8_MAX_WASM_FUNCTION_BYTES - self.max_func_body_bytes

    @property
    def max_func_locals(self) -> int:
        return max((m.max_func_locals for m in self.modules), default=0)

    @property
    def func_locals_headroom(self) -> int:
        return V8_MAX_WASM_FUNCTION_LOCALS - self.max_func_locals

    @property
    def large_func_bodies_over_1mib(self) -> int:
        return sum(m.large_func_bodies_over_1mib for m in self.modules)

    @property
    def total_code_func_count(self) -> int:
        return sum(m.code_func_count for m in self.modules)

    @property
    def within_all_limits(self) -> bool:
        return all(
            not (
                m.exceeds_v8_module_limit
                or m.exceeds_v8_func_size_limit
                or m.exceeds_v8_locals_limit
            )
            for m in self.modules
        )

    def v8_headroom_dict(self) -> dict[str, Any]:
        return {
            "max_module_bytes": self.max_module_bytes,
            "max_module_mib": self.max_module_mib,
            "max_module_name": self.max_module_name,
            "min_module_headroom_bytes": self.min_module_headroom_bytes,
            "min_module_headroom_mib": self.min_module_headroom_mib,
            "v8_module_limit_bytes": V8_MAX_WASM_MODULE_BYTES,
            "v8_module_limit_mib": round(V8_MAX_WASM_MODULE_BYTES / (1024 * 1024), 4),
            "max_func_body_bytes": self.max_func_body_bytes,
            "max_func_body_mib": self.max_func_body_mib,
            "v8_func_body_limit_bytes": V8_MAX_WASM_FUNCTION_BYTES,
            "func_body_headroom_bytes": self.func_body_headroom_bytes,
            "max_func_locals": self.max_func_locals,
            "v8_func_locals_limit": V8_MAX_WASM_FUNCTION_LOCALS,
            "func_locals_headroom": self.func_locals_headroom,
            "large_func_bodies_over_1mib": self.large_func_bodies_over_1mib,
            "total_code_func_count": self.total_code_func_count,
            "total_got_func_imports": self.total_got_func_imports,
            "total_table_size": self.total_table_size,
            "main_table_size": self.main_table_size,
            "side_table_size": self.side_table_size,
            "within_all_limits": self.within_all_limits,
        }

    def v8_headroom_manifest_dict(self) -> dict[str, Any]:
        return {
            "maxModuleBytes": self.max_module_bytes,
            "maxModuleMiB": self.max_module_mib,
            "maxModuleName": self.max_module_name,
            "minModuleHeadroomBytes": self.min_module_headroom_bytes,
            "minModuleHeadroomMiB": self.min_module_headroom_mib,
            "v8ModuleLimitBytes": V8_MAX_WASM_MODULE_BYTES,
            "v8ModuleLimitMiB": round(V8_MAX_WASM_MODULE_BYTES / (1024 * 1024), 4),
            "maxFuncBodyBytes": self.max_func_body_bytes,
            "maxFuncBodyMiB": self.max_func_body_mib,
            "v8FuncBodyLimitBytes": V8_MAX_WASM_FUNCTION_BYTES,
            "funcBodyHeadroomBytes": self.func_body_headroom_bytes,
            "maxFuncLocals": self.max_func_locals,
            "v8FuncLocalsLimit": V8_MAX_WASM_FUNCTION_LOCALS,
            "funcLocalsHeadroom": self.func_locals_headroom,
            "largeFuncBodiesOver1MiB": self.large_func_bodies_over_1mib,
            "totalCodeFuncCount": self.total_code_func_count,
            "totalGotFuncImports": self.total_got_func_imports,
            "totalTableSize": self.total_table_size,
            "mainTableSize": self.main_table_size,
            "sideTableSize": self.side_table_size,
            "withinAllLimits": self.within_all_limits,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "indirect_dispatch": self.indirect_dispatch,
            "modules": [m.to_dict() for m in self.modules],
            "totals": {
                "module_count": len(self.modules),
                "side_module_count": self.side_module_count,
                "wasm_bytes": self.total_wasm_bytes,
                "size_mib": self.total_size_mib,
                "func_imports": self.total_func_imports,
                "got_func_imports": self.total_got_func_imports,
                "got_mem_imports": self.total_got_mem_imports,
                "func_exports": self.total_func_exports,
                "table_size": self.total_table_size,
                "main_table_size": self.main_table_size,
                "side_table_size": self.side_table_size,
                "shard_dispatch_exports": self.shard_dispatch_exports,
                "shard_table_exports": self.shard_table_exports,
                "got_func_sub_imports": self.got_func_sub_imports,
                "got_func_shard_dispatch_imports": self.got_func_shard_dispatch_imports,
            },
            "v8_headroom": self.v8_headroom_dict(),
        }

    def to_manifest_dict(self) -> dict[str, Any]:
        """CamelCase numeric-only summary for `guest_manifest.json`."""
        return {
            "indirectDispatch": self.indirect_dispatch,
            "modules": [m.to_manifest_dict() for m in self.modules],
            "totals": {
                "moduleCount": len(self.modules),
                "sideModuleCount": self.side_module_count,
                "wasmBytes": self.total_wasm_bytes,
                "sizeMiB": self.total_size_mib,
                "funcImports": self.total_func_imports,
                "gotFuncImports": self.total_got_func_imports,
                "gotMemImports": self.total_got_mem_imports,
                "funcExports": self.total_func_exports,
                "tableSize": self.total_table_size,
                "mainTableSize": self.main_table_size,
                "sideTableSize": self.side_table_size,
                "shardDispatchExports": self.shard_dispatch_exports,
                "shardTableExports": self.shard_table_exports,
                "gotFuncSubImports": self.got_func_sub_imports,
                "gotFuncShardDispatchImports": self.got_func_shard_dispatch_imports,
            },
            "v8Headroom": self.v8_headroom_manifest_dict(),
        }


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
    emar: Path | None = None
    llvm_nm: Path | None = None

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

    def archiver(self) -> Path | None:
        if self.emar is not None:
            return self.emar
        base = self.emcc or self.emxx
        if base is not None:
            sibling = _first_existing(base.parent, _EMAR_NAMES)
            if sibling is not None:
                return sibling
            suffix = base.suffix if base.suffix.lower() in {".bat", ".cmd", ".exe", ".py"} else ""
            return base.parent / f"emar{suffix}"
        return None

    def nm(self) -> Path | None:
        if self.llvm_nm is not None:
            return self.llvm_nm
        base = self.emxx or self.emcc
        if base is not None:
            for directory in (base.parent.parent / "bin", base.parent):
                found = _first_existing(directory, _LLVM_NM_NAMES)
                if found is not None:
                    return found
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "emcc_found": self.emcc is not None,
            "emxx_found": self.emxx is not None,
            "llvm_nm_found": self.nm() is not None,
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
class SideModuleResult:
    index: int
    name: str
    shards: tuple[str, ...] = ()
    response_file: str | None = None
    exports_response_file: str | None = None
    exported_symbol_count: int | None = None
    link_argv: tuple[str, ...] = ()
    wasm_bytes: int = 0
    exceeds_v8_1gib_limit: bool = False
    role: str = "shards"  # "shards" | "registration"
    wasm_metrics: WasmModuleMetrics | None = None
    estimated_bytes: int = 0
    side_module_mode: int = 1
    link_cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "role": self.role,
            "shard_count": len(self.shards),
            "shards": list(self.shards),
            "response_file": self.response_file,
            "exports_response_file": self.exports_response_file,
            "exported_symbol_count": self.exported_symbol_count,
            "link_argv": list(self.link_argv),
            "estimated_bytes": self.estimated_bytes,
            "estimated_size_mib": round(self.estimated_bytes / (1024 * 1024), 4)
            if self.estimated_bytes
            else 0.0,
            "wasm_bytes": self.wasm_bytes,
            "exceeds_v8_1gib_limit": self.exceeds_v8_1gib_limit,
            "side_module_mode": self.side_module_mode,
            "wasm_metrics": self.wasm_metrics.to_dict() if self.wasm_metrics is not None else None,
        }

    def to_manifest_dict(self) -> dict[str, Any]:
        """CamelCase side-module entry for `guest_manifest.json` telemetry."""
        out: dict[str, Any] = {
            "index": self.index,
            "name": self.name,
            "role": self.role,
            "shardCount": len(self.shards),
            "shards": list(self.shards),
            "estimatedBytes": self.estimated_bytes,
            "estimatedSizeMiB": round(self.estimated_bytes / (1024 * 1024), 4)
            if self.estimated_bytes
            else 0.0,
            "wasmBytes": self.wasm_bytes,
            "sizeMiB": round(self.wasm_bytes / (1024 * 1024), 4) if self.wasm_bytes else 0.0,
            "exceedsV8_1GiBLimit": self.exceeds_v8_1gib_limit,
            "sideModuleMode": self.side_module_mode,
        }
        if self.exported_symbol_count is not None:
            out["exportedSymbolCount"] = self.exported_symbol_count
        return out


@dataclass
class BuildReport:
    ok: bool
    mode: str
    schema_version: int = SCHEMA_VERSION
    toolchain: Toolchain | None = None
    jobs: int = 1
    opt: str = "O2"
    opt_modules: dict[str, str] = field(default_factory=dict)
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
    stage_seconds: dict[str, float] = field(default_factory=dict)
    object_bytes: int = 0
    wasm_bytes: int = 0
    exceeds_v8_1gib_limit: bool = False
    no_link_opt: bool = False
    shrink_largest_shards: int = 0
    wasm_exceptions: bool = False
    initial_memory: str | None = None
    stack_size: str | None = None
    assertions: int | None = None
    browser_boot_exports: bool = False
    split_modules: int = 1
    split_strategy: str = DEFAULT_SPLIT_STRATEGY
    max_side_module_mib: int = DEFAULT_MAX_SIDE_MODULE_MIB
    side_module_mode: int = 1
    side_modules: list[SideModuleResult] = field(default_factory=list)
    node_max_old_space_mb: int | None = None
    proxy_to_pthread: bool = False
    max_concurrent_pthread_loads: int = DEFAULT_MAX_CONCURRENT_PTHREAD_LOADS
    split_runtime_js_patches: dict[str, bool] | None = None
    indirect_dispatch: str = DEFAULT_INDIRECT_DISPATCH
    indirect_dispatch_source: str = "default"
    verify_max_table_imports: int | None = None
    guest_manifest: str | None = None
    guest_dll_modules: tuple[str, ...] = ()
    wasm_metrics: WasmBuildMetrics | None = None
    removed_stale_side_modules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "mode": self.mode,
            "schema_version": self.schema_version,
            "toolchain": self.toolchain.to_dict() if self.toolchain else None,
            "jobs": self.jobs,
            "opt": self.opt,
            "opt_modules": dict(self.opt_modules),
            "no_link_opt": self.no_link_opt,
            "shrink_largest_shards": self.shrink_largest_shards,
            "wasm_exceptions": self.wasm_exceptions,
            "initial_memory": self.initial_memory,
            "stack_size": self.stack_size,
            "assertions": self.assertions,
            "browser_boot_exports": self.browser_boot_exports,
            "split_modules": self.split_modules,
            "split_strategy": self.split_strategy,
            "max_side_module_mib": self.max_side_module_mib,
            "side_module_mode": self.side_module_mode,
            "side_modules": [m.to_dict() for m in self.side_modules],
            "removed_stale_side_modules": list(self.removed_stale_side_modules),
            "node_max_old_space_mb": self.node_max_old_space_mb,
            "proxy_to_pthread": self.proxy_to_pthread,
            "max_concurrent_pthread_loads": self.max_concurrent_pthread_loads,
            "split_runtime_js_patches": (
                dict(self.split_runtime_js_patches)
                if self.split_runtime_js_patches is not None
                else None
            ),
            "indirect_dispatch": self.indirect_dispatch,
            "indirect_dispatch_source": self.indirect_dispatch_source,
            "verify_max_table_imports": self.verify_max_table_imports,
            "guest_manifest": self.guest_manifest,
            "guest_dll_modules": list(self.guest_dll_modules),
            "wasm_metrics": self.wasm_metrics.to_dict() if self.wasm_metrics is not None else None,
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
            "stage_seconds": dict(self.stage_seconds),
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
    if name.startswith("emar"):
        return "emar"
    if name.startswith("llvm-nm"):
        return "llvm-nm"
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
_EMAR_NAMES: tuple[str, ...] = ("emar", "emar.bat", "emar.cmd", "emar.exe", "emar.py")
_LLVM_NM_NAMES: tuple[str, ...] = (
    ("llvm-nm.exe", "llvm-nm", "llvm-nm.bat", "llvm-nm.cmd")
    if os.name == "nt"
    else ("llvm-nm", "llvm-nm.exe", "llvm-nm.bat", "llvm-nm.cmd")
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
    """Locate emcc/em++/emar/llvm-nm on PATH, EMSDK, or `work/emsdk/upstream`."""
    env_map = dict(os.environ if env is None else env)
    which = shutil.which if which_fn is None else which_fn

    emcc_path: Path | None = None
    emxx_path: Path | None = None
    emar_path: Path | None = None
    llvm_nm_path: Path | None = None
    origin = "missing"

    which_emcc = which("emcc")
    which_emxx = which("em++") or which("emxx")
    which_emar = which("emar")
    if which_emcc:
        emcc_path = Path(which_emcc)
        origin = "path"
    if which_emxx:
        emxx_path = Path(which_emxx)
        if origin == "missing":
            origin = "path"
    if which_emar:
        emar_path = Path(which_emar)

    search_dirs: list[tuple[str, Path]] = []
    emsdk = (env_map.get("EMSDK") or "").strip()
    if emsdk:
        search_dirs.append(("emsdk", Path(emsdk) / "upstream" / "emscripten"))
    search_dirs.append(("work/emsdk", repo_root / EMSDK_UPSTREAM_REL))
    if not _first_existing(repo_root / EMSDK_UPSTREAM_REL, _EMCC_NAMES):
        picked = emsdk_locator.select_work_emsdk(repo_root)
        if picked is not None:
            search_dirs.append(("work/**/emsdk", picked[0]))
            if which("emcc") is None and not (env_map.get("EMSDK") or "").strip():
                print(emsdk_locator.describe(repo_root, picked), file=sys.stderr)

    for label, directory in search_dirs:
        found_cc = _first_existing(directory, _EMCC_NAMES)
        found_xx = _first_existing(directory, _EMXX_NAMES)
        found_ar = _first_existing(directory, _EMAR_NAMES)
        if emcc_path is None and found_cc is not None:
            emcc_path = found_cc
            origin = label
        if emxx_path is None and found_xx is not None:
            emxx_path = found_xx
            if origin == "missing":
                origin = label
        if emar_path is None and found_ar is not None:
            emar_path = found_ar

    # Prefer llvm-nm from the discovered Emscripten toolchain's sibling LLVM bin/ dir,
    # then EMSDK / work/emsdk upstream/bin, then PATH.
    base_tool = emxx_path or emcc_path
    nm_dirs: list[Path] = []
    if base_tool is not None:
        nm_dirs.append(base_tool.parent.parent / "bin")
        nm_dirs.append(base_tool.parent)
    for _, directory in search_dirs:
        nm_dirs.append(directory.parent / "bin")
    for nm_dir in nm_dirs:
        found_nm = _first_existing(nm_dir, _LLVM_NM_NAMES)
        if found_nm is not None:
            llvm_nm_path = found_nm
            break
    if llvm_nm_path is None:
        which_nm = which("llvm-nm")
        if which_nm:
            llvm_nm_path = Path(which_nm)

    return Toolchain(
        emcc=emcc_path,
        emxx=emxx_path,
        origin=origin,
        emar=emar_path,
        llvm_nm=llvm_nm_path,
    )


def default_run_command(
    argv: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
) -> CommandResult:
    if not argv:
        return CommandResult(returncode=1, stderr="empty command")
    try:
        proc = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            env=dict(env) if env is not None else None,
            check=False,
        )
    except OSError as exc:
        return CommandResult(returncode=127, stderr=str(exc))
    return CommandResult(
        returncode=int(proc.returncode),
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )


def _runner_accepts_env(runner: CommandRunner) -> bool:
    try:
        sig = inspect.signature(runner)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == "env" and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


def _invoke_runner(
    runner: CommandRunner,
    argv: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
) -> CommandResult:
    if env is not None and _runner_accepts_env(runner):
        return runner(argv, env=env)
    return runner(argv)


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


def parse_opt_modules(
    raw: Sequence[str] | Mapping[str, str] | None,
) -> dict[str, str]:
    """Parse repeatable `--opt-module MODULE=LEVEL` overrides into `{module: opt_level}`."""
    if raw is None:
        return {}
    out: dict[str, str] = {}
    if isinstance(raw, Mapping):
        for mod_raw, level_raw in raw.items():
            if not isinstance(mod_raw, str) or not isinstance(level_raw, str):
                raise BuildGuestWasmError(
                    f"invalid --opt-module entry ({mod_raw!r}={level_raw!r}); expected MODULE=LEVEL"
                )
            mod = mod_raw.strip()
            if not mod or any(ch.isspace() for ch in mod):
                raise BuildGuestWasmError(
                    f"invalid --opt-module {mod_raw!r}={level_raw!r}; module name must be non-empty"
                )
            try:
                level = parse_opt_level(level_raw)
            except BuildGuestWasmError as exc:
                raise BuildGuestWasmError(
                    f"invalid --opt-module {mod_raw}={level_raw!r}; "
                    "expected level O0, O1, O2, O3, Os, or Oz"
                ) from exc
            out[mod] = level
        return out
    items: Sequence[str] = (raw,) if isinstance(raw, str) else raw
    for entry in items:
        if not isinstance(entry, str) or "=" not in entry:
            raise BuildGuestWasmError(
                f"invalid --opt-module {entry!r}; expected MODULE=LEVEL (e.g. bink2scarlett=O3)"
            )
        mod_raw, level_raw = entry.split("=", 1)
        mod = mod_raw.strip()
        if not mod or any(ch.isspace() for ch in mod):
            raise BuildGuestWasmError(
                f"invalid --opt-module {entry!r}; module name must be non-empty (expected MODULE=LEVEL)"
            )
        try:
            level = parse_opt_level(level_raw)
        except BuildGuestWasmError as exc:
            raise BuildGuestWasmError(
                f"invalid --opt-module {entry!r}; expected level O0, O1, O2, O3, Os, or Oz"
            ) from exc
        out[mod] = level
    return out


def _unit_module_name(unit: SourceUnit) -> str | None:
    """Return the logical module name for `unit` (e.g. `bink2scarlett`, `main`)."""
    if unit.module:
        return unit.module
    uid = unit.unit_id.replace("\\", "/")
    if "/" in uid:
        head = uid.split("/", 1)[0]
        if head and head not in {"lift", "stubs", "exports"}:
            return head
    stem = Path(uid).stem
    if "__" in stem:
        prefix = stem.split("__", 1)[0]
        if prefix:
            return prefix
    return None


def opt_level_for_unit(
    unit: SourceUnit,
    *,
    default_opt: str,
    opt_modules: Mapping[str, str] | None = None,
    shrink_ids: frozenset[str] = frozenset(),
) -> str:
    """Resolve the optimization level for a single `SourceUnit`.

    Precedence: explicit `--opt-module` override > `--shrink-largest-shards` (`-Oz`) > `--opt`.
    """
    if opt_modules:
        mod = _unit_module_name(unit)
        if mod is not None and mod in opt_modules:
            return opt_modules[mod]
    if unit.unit_id in shrink_ids:
        return "Oz"
    return default_opt


def parse_size_token(raw: str, *, flag_name: str) -> str:
    if not isinstance(raw, str):
        raise BuildGuestWasmError(
            f"invalid {flag_name} {raw!r}; expected non-empty positive alphanumeric size token (e.g. 64MB)"
        )
    token = raw.strip()
    match = _SIZE_TOKEN_RE.match(token)
    if match is None or int(match.group(1)) <= 0:
        raise BuildGuestWasmError(
            f"invalid {flag_name} {raw!r}; expected non-empty positive alphanumeric size token (e.g. 64MB, 8MB, 8388608)"
        )
    return token


def parse_assertions(raw: int | None) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int) or raw not in (0, 1):
        raise BuildGuestWasmError(f"invalid --assertions {raw!r}; expected 0 or 1")
    return raw


def parse_split_modules(raw: int) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise BuildGuestWasmError(f"invalid --split-modules {raw!r}; expected an integer >= 1")
    return raw


def parse_split_strategy(raw: str) -> str:
    if not isinstance(raw, str):
        raise BuildGuestWasmError(
            f"invalid --split-strategy {raw!r}; expected 'balanced', 'contiguous', 'asymmetric', or 'capped'"
        )
    token = raw.strip().lower()
    if token not in SPLIT_STRATEGIES:
        raise BuildGuestWasmError(
            f"invalid --split-strategy {raw!r}; expected 'balanced', 'contiguous', 'asymmetric', or 'capped'"
        )
    return token


def parse_max_side_module_mib(raw: int) -> int:
    if (
        isinstance(raw, bool)
        or not isinstance(raw, int)
        or raw < 1
        or raw > MAX_SIDE_MODULE_MIB_LIMIT
    ):
        raise BuildGuestWasmError(
            f"invalid --max-side-module-mib {raw!r}; expected an integer between 1 and "
            f"{MAX_SIDE_MODULE_MIB_LIMIT} (V8 1 GiB module limit)"
        )
    return raw


def side_module_mode_for_strategy(strategy: str) -> int:
    """Return 2 for asymmetric/capped (pruned) side modules, else 1 (legacy default)."""
    return 2 if parse_split_strategy(strategy) in ("asymmetric", "capped") else 1


def parse_node_max_old_space_mb(raw: int | None) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise BuildGuestWasmError(
            f"invalid --node-max-old-space-mb {raw!r}; expected an integer >= 1"
        )
    return raw


def parse_max_concurrent_pthread_loads(raw: int) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise BuildGuestWasmError(
            f"invalid --max-concurrent-pthread-loads {raw!r}; "
            "expected an integer >= 0 (0 disables throttling)"
        )
    return raw


def parse_verify_max_table_imports(raw: int | None) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise BuildGuestWasmError(
            f"invalid --verify-max-table-imports {raw!r}; expected an integer >= 0"
        )
    return raw


def _has_node_max_old_space_size(node_options: str) -> bool:
    return _NODE_MAX_OLD_SPACE_RE.search(node_options) is not None


def _strip_node_max_old_space_size(node_options: str) -> str:
    return _NODE_MAX_OLD_SPACE_RE.sub("", node_options).strip()


def build_link_env(
    base_env: Mapping[str, str] | None,
    *,
    split_modules: int = 1,
    node_max_old_space_mb: int | None = None,
) -> dict[str, str]:
    """Return a copy of `base_env` with `NODE_OPTIONS` scaled for large Emscripten links.

    - When `node_max_old_space_mb` is explicitly set, merges/replaces
      `--max-old-space-size=<N>` into `NODE_OPTIONS` while preserving other flags.
    - Otherwise, when `split_modules > 1` and `NODE_OPTIONS` does not already set
      `--max-old-space-size`, merges `--max-old-space-size=16384` without clobbering
      existing flags.
    - When `split_modules == 1` and `node_max_old_space_mb` is None, leaves
      `NODE_OPTIONS` completely untouched/unset.
    """
    split_count = parse_split_modules(split_modules)
    explicit_mb = parse_node_max_old_space_mb(node_max_old_space_mb)
    env_out = dict(os.environ if base_env is None else base_env)
    existing = env_out.get("NODE_OPTIONS", "")

    if explicit_mb is not None:
        cleaned = _strip_node_max_old_space_size(existing)
        flag = f"--max-old-space-size={explicit_mb}"
        env_out["NODE_OPTIONS"] = f"{cleaned} {flag}" if cleaned else flag
        return env_out

    if split_count > 1 and not _has_node_max_old_space_size(existing):
        flag = f"--max-old-space-size={DEFAULT_SPLIT_NODE_MAX_OLD_SPACE_MB}"
        stripped = existing.strip()
        env_out["NODE_OPTIONS"] = f"{stripped} {flag}" if stripped else flag
    return env_out


def _merge_ir_target_features(match: "re.Match[str]") -> str:
    present = [f for f in match.group(1).split(",") if f]
    for feature in SHARED_MEMORY_TARGET_FEATURES:
        if feature not in present:
            present.append(feature)
    return '"target-features"="' + ",".join(present) + '"'


def retarget_wasm64_ir(text: str) -> str:
    """Normalize lifted wasm64 LLVM IR for Emscripten 7 / LLVM 24 `emcc`.

    Drops any hardcoded `target datalayout = ...` line so `emcc` supplies the
    active LLVM backend's DataLayout, and rewrites
    `target triple = "wasm64-unknown-unknown"` to `"wasm64-unknown-emscripten"`.
    Every `"target-features"` attribute also gains the shared-memory pthread features
    (`+atomics,+bulk-memory,+mutable-globals`) so shards without atomic ops still link.
    """
    text = _IR_TARGET_FEATURES_RE.sub(_merge_ir_target_features, text)
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
    wasm_exceptions: bool = False,
    pic: bool = False,
) -> tuple[str, ...]:
    flags: list[str] = [*COMPILE_WASM_FLAGS, *SHARED_MEMORY_COMPILE_FLAGS, f"-{opt}"]
    if pic and PIC_FLAG not in flags:
        flags.append(PIC_FLAG)
    flags.extend(extra_flags)
    if kind == "ll":
        if "-msimd128" not in flags:
            flags.append("-msimd128")
        flags.extend(["-x", "ir"])
    elif kind == "cpp":
        flags.append("-std=c++20")
        if wasm_exceptions and WASM_EXCEPTIONS_FLAG not in flags:
            flags.append(WASM_EXCEPTIONS_FLAG)
    flags.append("-c")
    if kind != "ll":
        for include in include_dirs:
            flags.extend(["-I", str(include)])
    return tuple(flags)


def _should_skip_header_dir(dir_name: str) -> bool:
    """Return True for build, target, hidden, or VCS directories excluded from header hashing."""
    if not dir_name or dir_name.startswith("."):
        return True
    lowered = dir_name.lower()
    if lowered.startswith("build"):
        return True
    return dir_name in SKIP_DIR_NAMES or lowered in SKIP_DIR_NAMES


def _iter_header_files(root: Path) -> list[tuple[str, Path]]:
    """Collect `(posix_rel, path)` for all headers under `root`, sorted deterministically."""
    if not root.is_dir():
        return []
    entries: list[tuple[str, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not _should_skip_header_dir(d))
        current = Path(dirpath)
        for filename in sorted(filenames):
            candidate = current / filename
            if candidate.suffix.lower() not in HEADER_SUFFIXES:
                continue
            if not candidate.is_file():
                continue
            rel = candidate.relative_to(root).as_posix()
            entries.append((rel, candidate))
    entries.sort(key=lambda item: item[0])
    return entries


def compute_runtime_headers_fingerprint(
    repo_root: Path | None = None,
    *,
    header_roots: Sequence[Path] | None = None,
) -> str:
    """Return a deterministic SHA-256 hex digest of `runtime/` headers (or explicit `header_roots`).

    Walks `repo_root / "runtime"` (or each directory in `header_roots`) in sorted POSIX
    relative path order, skipping `build*`, `target`, hidden directories, and `SKIP_DIR_NAMES`.
    Hashes every `*.h`, `*.hpp`, and `*.inl` file's relative path and raw bytes into one
    SHA-256 hex digest. Computed once per build and salted into C/C++ `.o` cache keys only
    (never `.ll` LLVM IR shards).
    """
    if header_roots is not None:
        roots = [Path(p) for p in header_roots]
    elif repo_root is not None:
        roots = [Path(repo_root) / "runtime"]
    else:
        roots = [DEFAULT_REPO_ROOT / "runtime"]

    digest = hashlib.sha256()
    digest.update(b"playgta6-runtime-headers-v1\0")
    multi_root = len(roots) > 1
    for root_index, root in enumerate(roots):
        if multi_root:
            digest.update(f"root:{root_index}\0".encode("ascii"))
        for rel, header_path in _iter_header_files(root):
            data = header_path.read_bytes()
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(len(data)).encode("ascii"))
            digest.update(b"\0")
            digest.update(data)
            digest.update(b"\0")
    return digest.hexdigest()


def _dir_headers_fingerprint(directory: Path, *, tag: bytes) -> bytes:
    """Return a deterministic SHA-256 digest of direct-child headers in `directory`, or `b""`."""
    if not directory.is_dir():
        return b""
    headers = sorted(
        (
            entry
            for entry in directory.iterdir()
            if entry.is_file() and entry.suffix.lower() in HEADER_SUFFIXES
        ),
        key=lambda p: p.name,
    )
    if not headers:
        return b""
    digest = hashlib.sha256()
    digest.update(tag)
    for header in headers:
        data = header.read_bytes()
        digest.update(header.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(data)).encode("ascii"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.digest()


def sibling_headers_fingerprint(source_path: Path) -> bytes:
    """Return a deterministic SHA-256 digest of same-directory `.h`/`.hpp`/`.inl` headers, or `b""`."""
    return _dir_headers_fingerprint(source_path.parent, tag=b"playgta6-sibling-headers-v1\0")


def source_root_headers_fingerprint(source_root: Path) -> bytes:
    """Return a deterministic SHA-256 digest of top-level generated headers in `source_root`, or `b""`."""
    return _dir_headers_fingerprint(source_root, tag=b"playgta6-source-root-headers-v1\0")


def _included_source_root_headers_fingerprint(
    content: bytes,
    source_root: Path | None,
) -> bytes:
    """Hash any local `#include "..."` headers resolved under `source_root` (transitively)."""
    if source_root is None or not source_root.is_dir():
        return b""
    resolved_root = source_root.resolve()
    matched: dict[str, bytes] = {}
    queue: list[tuple[Path, bytes]] = [(resolved_root, content)]
    while queue:
        base_dir, text = queue.pop()
        for match in _LOCAL_INCLUDE_RE.finditer(text):
            raw_inc = match.group(1).decode("utf-8", errors="ignore").replace("\\", "/").strip()
            if not raw_inc or raw_inc.startswith("/"):
                continue
            for search_dir in (base_dir, resolved_root):
                candidate = (search_dir / raw_inc).resolve()
                try:
                    rel = candidate.relative_to(resolved_root).as_posix()
                except ValueError:
                    continue
                if rel in matched:
                    break
                if candidate.suffix.lower() in HEADER_SUFFIXES and candidate.is_file():
                    data = candidate.read_bytes()
                    matched[rel] = data
                    queue.append((candidate.parent, data))
                    break
    if not matched:
        return b""
    digest = hashlib.sha256()
    digest.update(b"playgta6-included-source-headers-v1\0")
    for rel in sorted(matched):
        data = matched[rel]
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(data)).encode("ascii"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.digest()


def _extra_fingerprint_for_unit(
    unit: SourceUnit,
    *,
    content: bytes | None = None,
    source_root: Path | None = None,
    source_root_headers_fp: bytes = b"",
) -> bytes:
    if unit.kind not in {"c", "cpp"}:
        return b""
    sibling_fp = sibling_headers_fingerprint(unit.path)
    included_fp = (
        _included_source_root_headers_fingerprint(content, source_root)
        if content is not None and source_root is not None
        else b""
    )
    if not source_root_headers_fp and not included_fp:
        return sibling_fp
    digest = hashlib.sha256()
    digest.update(b"playgta6-unit-extra-headers-v1\0")
    for label, part in (
        (b"sibling", sibling_fp),
        (b"source-root", source_root_headers_fp),
        (b"included", included_fp),
    ):
        if part:
            digest.update(label)
            digest.update(b"\0")
            digest.update(str(len(part)).encode("ascii"))
            digest.update(b"\0")
            digest.update(part)
            digest.update(b"\0")
    return digest.digest()


def _cache_key_prefix(
    flags: Sequence[str],
    *,
    kind: str,
    retarget_version: int | str | None = None,
    extra_fingerprint: bytes = b"",
    runtime_headers_fingerprint: str = "",
) -> "hashlib._Hash":
    """Everything in the cache key except the unit content (see `cache_key_for`)."""
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
    if kind in {"c", "cpp"} and runtime_headers_fingerprint:
        rt_fp_bytes = runtime_headers_fingerprint.encode("utf-8")
        digest.update(b"runtime-headers\0")
        digest.update(str(len(rt_fp_bytes)).encode("ascii"))
        digest.update(b"\0")
        digest.update(rt_fp_bytes)
        digest.update(b"\0")
    if extra_fingerprint:
        digest.update(b"extra\0")
        digest.update(str(len(extra_fingerprint)).encode("ascii"))
        digest.update(b"\0")
        digest.update(extra_fingerprint)
        digest.update(b"@0")
    return digest


def cache_key_for(
    content: bytes,
    flags: Sequence[str],
    *,
    kind: str,
    retarget_version: int | str | None = None,
    extra_fingerprint: bytes = b"",
    runtime_headers_fingerprint: str = "",
) -> str:
    digest = _cache_key_prefix(
        flags,
        kind=kind,
        retarget_version=retarget_version,
        extra_fingerprint=extra_fingerprint,
        runtime_headers_fingerprint=runtime_headers_fingerprint,
    )
    digest.update(content)
    return digest.hexdigest()


KEY_STAMP_FILE_NAME = "key-stamps-v1.json"


class KeyStampCache:
    """Remembers `cache_key_for` results for big `.ll` units by (path, size, mtime_ns, key prefix).

    Keys stay byte-identical to `cache_key_for`; the stamp only skips re-reading and re-hashing
    multi-hundred-MiB shards that did not change since the last build (make-style heuristic).
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: dict[str, list[Any]] = {}
        self._dirty = False
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._entries = {k: v for k, v in raw.items() if isinstance(v, list) and len(v) == 3}
        except (OSError, ValueError):
            self._entries = {}

    def key_for_file(
        self,
        file_path: Path,
        flags: Sequence[str],
        *,
        kind: str,
        runtime_headers_fingerprint: str = "",
    ) -> str:
        prefix = _cache_key_prefix(
            flags, kind=kind, runtime_headers_fingerprint=runtime_headers_fingerprint
        )
        st = file_path.stat()
        stamp_id = f"{file_path}|{prefix.hexdigest()}"
        entry = self._entries.get(stamp_id)
        if entry is not None and entry[0] == st.st_size and entry[1] == st.st_mtime_ns:
            return str(entry[2])
        digest = prefix
        with file_path.open("rb") as fh:
            while True:
                chunk = fh.read(8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        key = digest.hexdigest()
        self._entries[stamp_id] = [st.st_size, st.st_mtime_ns, key]
        self._dirty = True
        return key

    def save(self) -> None:
        if not self._dirty:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._entries), encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError:
            pass


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


def is_registration_unit(unit: SourceUnit) -> bool:
    """Return True for lifted function registration tables (`functions.ll`)."""
    name = Path(unit.unit_id.replace("\\", "/")).name
    return unit.kind == "ll" and name == "functions.ll"


def is_side_shard_unit(unit: SourceUnit) -> bool:
    """Return True for any lifted LLVM IR `.ll` unit that belongs in a shard side module.

    Includes standard `shard-*.ll` units and any non-registration guest DLL `.ll` file
    (e.g. `bink2scarlett/*.ll` or `bink2scarlett/lift/*.ll` other than `functions.ll`)
    so lifted guest DLL IR is never dropped or misclassified into `runtime.wasm`.
    """
    return unit.kind == "ll" and not is_registration_unit(unit)


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


def select_shrink_shards(units: Sequence[SourceUnit], count: int) -> frozenset[str]:
    """Return unit_ids of the top-`count` largest `.ll` shard units to compile at `-Oz`.

    Prefers `shard-*.ll` units when present (excluding registration/thunks) and falls back
    to all `.ll` units otherwise. Ranks by manifest instruction count when available,
    otherwise by source file byte size, deterministically tie-broken by `unit_id`.
    """
    if count <= 0:
        return frozenset()
    candidates = [unit for unit in units if is_shard_unit(unit)]
    if not candidates:
        candidates = [unit for unit in units if unit.kind == "ll"]
    if not candidates:
        return frozenset()
    use_instructions = any(unit.instructions > 0 for unit in candidates)
    if use_instructions:
        ranked = sorted(
            candidates,
            key=lambda u: (-u.instructions, u.unit_id),
        )
    else:
        ranked = sorted(
            candidates,
            key=lambda u: (-_object_bytes(u.path), u.unit_id),
        )
    return frozenset(unit.unit_id for unit in ranked[:count])


def _partition_weight(item: UnitBuildResult) -> int:
    obj_size = item.object_bytes or _object_bytes(item.object_path)
    if obj_size > 0:
        return obj_size
    return _object_bytes(item.unit.path)


def _minimax_contiguous_bound(weights: Sequence[int], bucket_count: int) -> int:
    """Return the exact minimum possible max bucket weight across `bucket_count` contiguous slices."""
    lo = max(weights)
    hi = sum(weights)
    if lo == hi or bucket_count == 1:
        return hi
    while lo < hi:
        mid = (lo + hi) // 2
        buckets_used = 1
        current = 0
        for weight in weights:
            if current + weight > mid:
                buckets_used += 1
                if buckets_used > bucket_count:
                    break
                current = weight
            else:
                current += weight
        if buckets_used <= bucket_count:
            hi = mid
        else:
            lo = mid + 1
    return lo


def partition_shards_contiguous(
    shard_items: Sequence[UnitBuildResult],
    bucket_count: int,
) -> list[list[UnitBuildResult]]:
    """Partition shards into up to `bucket_count` contiguous slices in ascending `unit_id` order.

    Uses exact minimax contiguous partitioning (binary search for the tightest possible max
    bucket byte size, followed by exact minimum-variance contiguous DP under that cap) so every
    bucket is a strictly contiguous slice of sorted `unit_id`s with optimal byte balance and
    `min(bucket_count, len(shard_items))` non-empty buckets.
    """
    bucket_limit = parse_split_modules(bucket_count)
    if not shard_items:
        return []
    ordered = sorted(shard_items, key=lambda item: item.unit.unit_id)
    n = len(ordered)
    k = min(bucket_limit, n)
    if k == 1:
        return [ordered]
    if k == n:
        return [[item] for item in ordered]

    weights = [_partition_weight(item) for item in ordered]
    prefix = [0] * (n + 1)
    for idx, weight in enumerate(weights):
        prefix[idx + 1] = prefix[idx] + weight

    max_cap = _minimax_contiguous_bound(weights, k)

    # Among all k-way contiguous partitions where every bucket has sum <= max_cap, minimize
    # (sum of squared bucket weights, sum of squared bucket shard counts). Both quadratic
    # terms satisfy the quadrangle inequality under the per-bucket cap, so divide-and-conquer
    # DP computes the exact global optimum in O(K * N * log N).
    max_end = [0] * k
    max_end[0] = min(n - (k - 1), bisect_right(prefix, max_cap) - 1)
    for b in range(1, k):
        prev_max = max_end[b - 1]
        limit_i = n - (k - 1 - b)
        reach = bisect_right(prefix, prefix[prev_max] + max_cap) - 1
        max_end[b] = min(limit_i, reach)

    prev_w = [0] * (n + 1)
    prev_c = [0] * (n + 1)
    for i in range(1, max_end[0] + 1):
        seg = prefix[i]
        prev_w[i] = seg * seg
        prev_c[i] = i * i

    split_at: list[list[int]] = [[0] * (n + 1) for _ in range(k)]

    for b in range(1, k):
        prev_max = max_end[b - 1]
        if b == k - 1:
            p_n = prefix[n]
            j_lo = max(b, bisect_left(prefix, p_n - max_cap))
            j_hi = min(n - 1, prev_max)
            best_j = -1
            best_w = 0
            best_c = 0
            for j in range(j_lo, j_hi + 1):
                seg = p_n - prefix[j]
                if seg > max_cap:
                    continue
                cnt = n - j
                cand_w = prev_w[j] + seg * seg
                cand_c = prev_c[j] + cnt * cnt
                if best_j == -1 or cand_w < best_w or (cand_w == best_w and cand_c < best_c):
                    best_j = j
                    best_w = cand_w
                    best_c = cand_c
            split_at[b][n] = best_j
            break

        curr_w = [0] * (n + 1)
        curr_c = [0] * (n + 1)
        row_split = split_at[b]

        def _solve(i_lo: int, i_hi: int, opt_lo: int, opt_hi: int) -> None:
            if i_lo > i_hi:
                return
            mid = (i_lo + i_hi) // 2
            p_mid = prefix[mid]
            j_start = max(opt_lo, bisect_left(prefix, p_mid - max_cap))
            j_end = min(opt_hi, mid - 1, prev_max)
            best_j = -1
            best_w = 0
            best_c = 0
            for j in range(j_start, j_end + 1):
                seg = p_mid - prefix[j]
                if seg > max_cap:
                    continue
                cnt = mid - j
                cand_w = prev_w[j] + seg * seg
                cand_c = prev_c[j] + cnt * cnt
                if best_j == -1 or cand_w < best_w or (cand_w == best_w and cand_c < best_c):
                    best_j = j
                    best_w = cand_w
                    best_c = cand_c
            curr_w[mid] = best_w
            curr_c[mid] = best_c
            row_split[mid] = best_j
            _solve(i_lo, mid - 1, opt_lo, best_j)
            _solve(mid + 1, i_hi, best_j, opt_hi)

        _solve(b + 1, max_end[b], b, prev_max)
        prev_w = curr_w
        prev_c = curr_c

    cuts = [0] * (k + 1)
    cuts[k] = n
    end = n
    for b in range(k - 1, 0, -1):
        start = split_at[b][end]
        cuts[b] = start
        end = start
    return [ordered[cuts[b] : cuts[b + 1]] for b in range(k)]


def partition_shards_asymmetric(
    shard_items: Sequence[UnitBuildResult],
    bucket_count: int,
    *,
    max_bucket_bytes: int | None = None,
    collapse_if_fits: bool = False,
) -> list[list[UnitBuildResult]]:
    """Partition shards into up to `bucket_count` asymmetric contiguous slices in VA order.

    Packs `part_0` greedily up to `max_bucket_bytes` (default `DEFAULT_MAX_SIDE_MODULE_MIB` MiB,
    the proven ~994 MiB Chrome ceiling under V8's 1,024 MiB module cap), then packs remaining
    shards into `part_1..N-1` in strictly contiguous `unit_id` order while keeping every bucket
    non-empty. When total weight exceeds `bucket_count * max_bucket_bytes`, relaxes the per-bucket
    ceiling only to the exact minimax contiguous bound so remaining buckets stay as small as
    possible under V8's 1 GiB limit (two-part shape ≈ 994/171 MiB).

    When `collapse_if_fits=True` and the combined weight of all shards is `<= max_bucket_bytes`,
    collapses all code shards into a single bucket (`[ordered]`) so there are zero cross-part
    `sub_*` imports/exports between code side modules.
    """
    bucket_limit = parse_split_modules(bucket_count)
    if max_bucket_bytes is not None:
        if (
            isinstance(max_bucket_bytes, bool)
            or not isinstance(max_bucket_bytes, int)
            or max_bucket_bytes < 1
        ):
            raise BuildGuestWasmError(
                f"invalid max_bucket_bytes {max_bucket_bytes!r}; expected a positive integer"
            )
        ceiling = max_bucket_bytes
    else:
        ceiling = DEFAULT_MAX_SIDE_MODULE_MIB * 1024 * 1024

    if not shard_items:
        return []
    ordered = sorted(shard_items, key=lambda item: item.unit.unit_id)
    n = len(ordered)
    k = min(bucket_limit, n)
    if k == 1:
        return [ordered]

    weights = [_partition_weight(item) for item in ordered]
    if collapse_if_fits and sum(weights) <= ceiling:
        return [ordered]
    if k == n:
        return [[item] for item in ordered]

    minimax_cap = _minimax_contiguous_bound(weights, k)
    effective_cap = max(ceiling, minimax_cap)

    buckets: list[list[UnitBuildResult]] = []
    pos = 0
    for b in range(k):
        remaining_buckets = k - 1 - b
        if remaining_buckets == 0:
            buckets.append(ordered[pos:])
            break
        max_end_idx = n - remaining_buckets
        bucket: list[UnitBuildResult] = [ordered[pos]]
        acc = weights[pos]
        pos += 1
        while pos < max_end_idx and acc + weights[pos] <= effective_cap:
            bucket.append(ordered[pos])
            acc += weights[pos]
            pos += 1
        buckets.append(bucket)
    return buckets


def partition_shards_capped(
    shard_items: Sequence[UnitBuildResult],
    max_parts: int,
    *,
    max_bucket_bytes: int,
) -> list[list[UnitBuildResult]]:
    """Size-driven contiguous partition: the fewest parts whose every part is <= `max_bucket_bytes`.

    N is auto-chosen (smallest k with exact minimax contiguous bound <= cap), then the shards are
    split into k contiguous `unit_id`-ordered slices with minimum size variance
    (`partition_shards_contiguous`), so parts are balanced rather than front-loaded. Fully
    deterministic. Raises `BuildGuestWasmError` instead of ever returning an over-cap part.
    """
    limit = parse_split_modules(max_parts)
    if isinstance(max_bucket_bytes, bool) or not isinstance(max_bucket_bytes, int) or max_bucket_bytes < 1:
        raise BuildGuestWasmError(
            f"invalid max_bucket_bytes {max_bucket_bytes!r}; expected a positive integer"
        )
    if not shard_items:
        return []
    ordered = sorted(shard_items, key=lambda item: item.unit.unit_id)
    weights = [_partition_weight(item) for item in ordered]
    mib = 1024 * 1024
    biggest = max(range(len(weights)), key=lambda i: weights[i])
    if weights[biggest] > max_bucket_bytes:
        raise BuildGuestWasmError(
            f"shard {ordered[biggest].unit.unit_id} alone is {weights[biggest] / mib:.1f} MiB, over the "
            f"{max_bucket_bytes / mib:.1f} MiB per-part cap; shrink it (--shrink-largest-shards) or "
            f"raise --max-side-module-mib (hard limit {MAX_SIDE_MODULE_MIB_LIMIT})"
        )
    total = sum(weights)
    k = max(1, -(-total // max_bucket_bytes))
    k = min(k, len(ordered))
    while k < len(ordered) and _minimax_contiguous_bound(weights, k) > max_bucket_bytes:
        k += 1
    if k > limit:
        raise BuildGuestWasmError(
            f"guest code ({total / mib:.1f} MiB of objects) needs {k} parts of <= "
            f"{max_bucket_bytes / mib:.1f} MiB but --split-modules allows only {limit}; "
            f"raise --split-modules (parts are auto-sized, it is only an upper bound)"
        )
    buckets = partition_shards_contiguous(ordered, k)
    worst = max(sum(_partition_weight(it) for it in b) for b in buckets)
    if worst > max_bucket_bytes:
        raise BuildGuestWasmError(
            f"capped partition produced a {worst / mib:.1f} MiB part over the "
            f"{max_bucket_bytes / mib:.1f} MiB cap"
        )
    return buckets


def partition_shard_units(
    items: Sequence[UnitBuildResult],
    split_modules: int,
    *,
    strategy: str = DEFAULT_SPLIT_STRATEGY,
    max_bucket_bytes: int | None = None,
    collapse_if_fits: bool = False,
) -> list[list[UnitBuildResult]]:
    """Deterministically partition `.ll` shard units into up to `split_modules` buckets.

    Supports three strategies:
      - `"balanced"` (default): greedy LPT bin-packing by compiled `.o` byte size (falling
        back to source file byte size for dry-runs), tie-broken deterministically by `unit_id`.
      - `"contiguous"`: exact minimax contiguous partition in ascending `unit_id` (guest VA /
        `.text`) order via `partition_shards_contiguous`, keeping adjacent shards together to
        minimize cross-module imports and JS stub closures.
      - `"asymmetric"`: contiguous min-cut via `partition_shards_asymmetric`, packing `part_0`
        up to `max_bucket_bytes` (default 994 MiB) under V8's 1 GiB cap so a two-part guest
        reproduces the proven ~994/171 MiB browser split; pairs with `-sSIDE_MODULE=2`.
        Pass `collapse_if_fits=True` to collapse all code shards into 1 side module whenever
        their combined weight fits within `max_bucket_bytes`.

    Emits only non-empty buckets when fewer shards exist than `split_modules`.
    """
    bucket_limit = parse_split_modules(split_modules)
    split_strat = parse_split_strategy(strategy)
    shards = [item for item in items if is_side_shard_unit(item.unit)]
    if not shards:
        return []
    if split_strat == "contiguous":
        return partition_shards_contiguous(shards, bucket_limit)
    if split_strat == "capped":
        return partition_shards_capped(
            shards,
            bucket_limit,
            max_bucket_bytes=(
                max_bucket_bytes
                if max_bucket_bytes is not None
                else DEFAULT_CAPPED_SIDE_MODULE_MIB * 1024 * 1024
            ),
        )
    if split_strat == "asymmetric":
        return partition_shards_asymmetric(
            shards,
            bucket_limit,
            max_bucket_bytes=max_bucket_bytes,
            collapse_if_fits=collapse_if_fits,
        )
    bucket_count = min(bucket_limit, len(shards))
    ranked = sorted(shards, key=lambda item: (-_partition_weight(item), item.unit.unit_id))
    buckets: list[list[UnitBuildResult]] = [[] for _ in range(bucket_count)]
    totals: list[int] = [0] * bucket_count
    for item in ranked:
        target_idx = min(range(bucket_count), key=lambda idx: (totals[idx], len(buckets[idx]), idx))
        buckets[target_idx].append(item)
        totals[target_idx] += _partition_weight(item)
    for bucket in buckets:
        bucket.sort(key=lambda item: item.unit.unit_id)
    return [bucket for bucket in buckets if bucket]


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


_SAFE_MODULE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _load_program_manifest_data(
    source_root: Path,
    *,
    manifest_path: Path | None = None,
) -> Mapping[str, Any] | None:
    """Load `program-manifest.json` as a mapping if present and valid JSON."""
    prog_manifest = (
        manifest_path
        if manifest_path is not None and manifest_path.name == PROGRAM_MANIFEST_FILE
        else (source_root / PROGRAM_MANIFEST_FILE)
    )
    if not prog_manifest.is_file():
        return None
    try:
        data = load_build_manifest(prog_manifest)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, Mapping) else None


def _sanitize_guest_module_name(raw: Any) -> str | None:
    """Sanitize a guest module name so host paths, usernames, or traversal never leak."""
    if not isinstance(raw, str):
        return None
    text = raw.strip().replace("\\", "/").rstrip("/")
    if not text:
        return None
    if text.endswith(f"/{BUILD_MANIFEST_FILE}"):
        text = text[: -len(BUILD_MANIFEST_FILE)].rstrip("/")
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    if ":" in text:
        text = text.rsplit(":", 1)[-1]
    text = _SAFE_MODULE_NAME_RE.sub("", text).strip(".")
    if not text or text in {".", ".."}:
        return None
    return text


def _is_guest_dll_entry(item: Any, module_name: str) -> bool:
    """Return True when a `program-manifest.json` module entry is a lifted guest DLL."""
    if isinstance(item, Mapping):
        role = item.get("role")
        if isinstance(role, str) and role.strip():
            role_lower = role.strip().lower()
            if role_lower in {"exe", "main"}:
                return False
            if role_lower == "dll":
                return True
        for key in ("file", "dll"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                lowered = val.strip().lower()
                if lowered.endswith(".exe"):
                    return False
                if lowered.endswith(".dll"):
                    return True
    lowered_name = module_name.lower()
    if lowered_name in {"main", "program", "boot"} or lowered_name.endswith(".exe"):
        return False
    return True


def discover_guest_dll_modules(
    source_root: Path,
    *,
    manifest_path: Path | None = None,
) -> tuple[str, ...]:
    """Return the sanitized, manifest-ordered tuple of lifted guest DLL module names.

    Reads `program-manifest.json` (`modules[]`), excludes the main executable (`role == "exe"`,
    `.exe` file, or `main`), strips any path components, and preserves deterministic order
    (e.g. `("vcruntime140", "bink2scarlett")`). Returns `()` for single-module builds.
    """
    data = _load_program_manifest_data(source_root, manifest_path=manifest_path)
    if data is None:
        return ()
    raw_modules = data.get("modules")
    if not isinstance(raw_modules, list):
        return ()
    dll_modules: list[str] = []
    seen: set[str] = set()
    for item in raw_modules:
        raw_name = _module_name_from_entry(item)
        if not raw_name:
            continue
        clean_name = _sanitize_guest_module_name(raw_name)
        if not clean_name or not _is_guest_dll_entry(item, clean_name):
            continue
        if clean_name not in seen:
            seen.add(clean_name)
            dll_modules.append(clean_name)
    return tuple(dll_modules)


def _parse_manifest_va(raw: Any) -> int | None:
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, int):
        return raw if raw > 0 else None
    if isinstance(raw, str):
        token = raw.strip()
        if not token:
            return None
        try:
            value = int(token, 16) if token.lower().startswith("0x") else int(token, 10)
        except ValueError:
            return None
        return value if value > 0 else None
    return None


def _add_va_function_symbols(target: set[str], raw_va: Any) -> None:
    va = _parse_manifest_va(raw_va)
    if va is None:
        return
    target.add(f"sub_{va:X}")
    target.add(f"sub_{va:x}")


def extract_program_required_exports(
    source_root: Path,
    *,
    manifest_path: Path | None = None,
) -> frozenset[str]:
    """Extract required cross-module / guest-DLL export symbols from `program-manifest.json`.

    Retains every lifted module's registration/dispatch/import-bridge symbols, explicit
    `required_exports` / `exports.symbols` names, and lifted entry-point / TLS / guest-bound
    IAT target `sub_<VA>` symbols so `-sSIDE_MODULE=2` pruning never drops live guest DLL
    entries (e.g. `bink2scarlett`, `vcruntime140`).
    """
    data = _load_program_manifest_data(source_root, manifest_path=manifest_path)
    if data is None:
        return frozenset()

    symbols: set[str] = set()

    def _take_sym(val: Any) -> None:
        if isinstance(val, str):
            tok = val.strip()
            if tok:
                symbols.add(tok)

    raw_top_req = data.get("required_exports")
    if isinstance(raw_top_req, list):
        for sym in raw_top_req:
            _take_sym(sym)

    raw_modules = data.get("modules")
    if isinstance(raw_modules, list):
        for mod in raw_modules:
            if not isinstance(mod, Mapping):
                continue
            mod_name = _sanitize_guest_module_name(_module_name_from_entry(mod))
            if mod_name:
                symbols.update(
                    {
                        f"__playgta6_register_lifted_functions_{mod_name}",
                        f"__playgta6_lifted_functions_{mod_name}",
                        f"__playgta6_lifted_function_count_{mod_name}",
                        f"{SHARD_DISPATCH_TABLE_PREFIX}{mod_name}",
                        f"pg6_import_thunks_{mod_name}",
                        f"pg6_import_thunk_count_{mod_name}",
                    }
                )
            for key in (
                "shard_dispatch",
                "shard_dispatch_symbol",
                "shard_dispatch_table",
                "registration_symbol",
            ):
                _take_sym(mod.get(key))

            reg = mod.get("registration")
            if isinstance(reg, Mapping):
                for key in (
                    "register_function",
                    "table_symbol",
                    "count_symbol",
                    "shard_dispatch_symbol",
                    "shard_dispatch",
                    "shard_dispatch_table",
                ):
                    _take_sym(reg.get(key))

            bridge = mod.get("bridge")
            if isinstance(bridge, Mapping):
                for key in ("thunks_symbol", "thunk_count_symbol"):
                    _take_sym(bridge.get(key))

            _add_va_function_symbols(symbols, mod.get("entry_point"))

            tls = mod.get("tls")
            if isinstance(tls, Mapping):
                callbacks = tls.get("callbacks")
                if isinstance(callbacks, list):
                    for cb in callbacks:
                        _add_va_function_symbols(symbols, cb)

            iat_prefill = mod.get("iat_prefill")
            if isinstance(iat_prefill, list):
                for entry in iat_prefill:
                    if isinstance(entry, Mapping):
                        _add_va_function_symbols(symbols, entry.get("value"))

            mod_exports = mod.get("exports")
            if isinstance(mod_exports, Mapping):
                for list_key in ("symbols", "functions"):
                    sym_list = mod_exports.get(list_key)
                    if isinstance(sym_list, list):
                        for item in sym_list:
                            if isinstance(item, str):
                                if item.strip().lower().startswith("0x"):
                                    _add_va_function_symbols(symbols, item)
                                else:
                                    _take_sym(item)
                            elif isinstance(item, Mapping):
                                _take_sym(item.get("symbol"))
                                _add_va_function_symbols(symbols, item.get("va") or item.get("target"))

            mod_req = mod.get("required_exports")
            if isinstance(mod_req, list):
                for sym in mod_req:
                    _take_sym(sym)

    raw_imports = data.get("imports")
    if isinstance(raw_imports, list):
        for imp in raw_imports:
            if not isinstance(imp, Mapping):
                continue
            if imp.get("binding") == "guest" and imp.get("kind") != "data":
                _add_va_function_symbols(symbols, imp.get("target"))

    return frozenset(symbols)


def _normalize_dispatch_token(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    token = raw.strip().lower()
    if token in INDIRECT_DISPATCH_MODES:
        return token
    return None


def _extract_manifest_indirect_dispatch(data: Any) -> str | None:
    """Inspect a `program-manifest.json` or `build-manifest.json` object for dispatch mode."""
    if not isinstance(data, Mapping):
        return None

    direct = _normalize_dispatch_token(data.get("indirect_dispatch"))
    if direct is not None:
        return direct

    options = data.get("options")
    if isinstance(options, Mapping):
        opt_mode = _normalize_dispatch_token(options.get("indirect_dispatch"))
        if opt_mode is not None:
            return opt_mode
        opt_build = options.get("build")
        if isinstance(opt_build, Mapping):
            opt_build_mode = _normalize_dispatch_token(opt_build.get("indirect_dispatch"))
            if opt_build_mode is not None:
                return opt_build_mode

    lift = data.get("lift")
    if isinstance(lift, Mapping):
        lift_mode = _normalize_dispatch_token(lift.get("indirect_dispatch"))
        if lift_mode is not None:
            return lift_mode
        reg = lift.get("registration")
        if isinstance(reg, Mapping):
            for key in ("indirect_dispatch", "mode"):
                reg_mode = _normalize_dispatch_token(reg.get(key))
                if reg_mode is not None:
                    return reg_mode
            for key in ("symbol", "table", "shard_dispatch", "shard_dispatch_symbol"):
                sym = reg.get(key)
                if isinstance(sym, str) and sym.startswith(SHARD_DISPATCH_TABLE_PREFIX):
                    return "shard"

    modules = data.get("modules")
    if isinstance(modules, list):
        seen_table = False
        for mod in modules:
            if not isinstance(mod, Mapping):
                continue
            mod_mode = _normalize_dispatch_token(mod.get("indirect_dispatch"))
            if mod_mode == "shard":
                return "shard"
            if mod_mode == "table":
                seen_table = True
            for key in (
                "shard_dispatch",
                "shard_dispatch_symbol",
                "shard_dispatch_table",
                "registration_symbol",
            ):
                sym = mod.get(key)
                if isinstance(sym, str) and sym.startswith(SHARD_DISPATCH_TABLE_PREFIX):
                    return "shard"
            reg = mod.get("registration")
            if isinstance(reg, Mapping):
                for key in (
                    "shard_dispatch",
                    "shard_dispatch_symbol",
                    "shard_dispatch_table",
                ):
                    sym = reg.get(key)
                    if isinstance(sym, str) and sym.startswith(SHARD_DISPATCH_TABLE_PREFIX):
                        return "shard"
        if seen_table:
            return "table"

    return None


def _file_head_tail_contains(
    path: Path,
    needles: Sequence[bytes],
    *,
    window_bytes: int = 262_144,
) -> bool:
    """Check head and tail slices of a file for any marker without loading huge IR files."""
    if not path.is_file():
        return False
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size <= window_bytes * 2:
                blob = fh.read()
                return any(needle in blob for needle in needles)
            head = fh.read(window_bytes)
            if any(needle in head for needle in needles):
                return True
            fh.seek(max(0, size - window_bytes))
            tail = fh.read(window_bytes)
            return any(needle in tail for needle in needles)
    except OSError:
        return False


def detect_indirect_dispatch(
    source_root: Path,
    units: Sequence[SourceUnit],
    *,
    manifest_path: Path | None = None,
) -> IndirectDispatchDetection:
    """Auto-detect `shard` (`Pg6ShardDispatchTable`) vs legacy `table` indirect dispatch.

    Checks in deterministic order:
      1. `program-manifest.json` (`indirect_dispatch`, `options.indirect_dispatch`, or module entries).
      2. Root and per-module `build-manifest.json`.
      3. `pg6_program.cpp` (`__playgta6_shard_dispatch_` or `Pg6ShardDispatchTable`).
      4. `functions.ll` registration units (`__playgta6_shard_dispatch_` / `__playgta6_dispatch_shard_`).
      5. Sampled `shard-*.ll` units (`__playgta6_dispatch_shard_`).
      6. Fallback: `"table"` (legacy).
    """
    prog_manifest = (
        manifest_path
        if manifest_path is not None and manifest_path.name == PROGRAM_MANIFEST_FILE
        else (source_root / PROGRAM_MANIFEST_FILE)
    )
    if prog_manifest.is_file():
        try:
            prog_data = load_build_manifest(prog_manifest)
        except (OSError, json.JSONDecodeError):
            prog_data = None
        prog_mode = _extract_manifest_indirect_dispatch(prog_data)
        if prog_mode is not None:
            return IndirectDispatchDetection(mode=prog_mode, source=PROGRAM_MANIFEST_FILE)

    build_manifests: list[Path] = []
    if manifest_path is not None and manifest_path.name == BUILD_MANIFEST_FILE:
        build_manifests.append(manifest_path)
    root_build_manifest = source_root / BUILD_MANIFEST_FILE
    if root_build_manifest.is_file() and root_build_manifest not in build_manifests:
        build_manifests.append(root_build_manifest)
    seen_mod_dirs: set[str] = set()
    for unit in units:
        mod = unit.module
        if not mod or mod in {"program", "boot"} or mod in seen_mod_dirs:
            continue
        seen_mod_dirs.add(mod)
        mod_manifest = source_root / mod / BUILD_MANIFEST_FILE
        if mod_manifest.is_file() and mod_manifest not in build_manifests:
            build_manifests.append(mod_manifest)

    saw_build_table = False
    for bm_path in build_manifests:
        try:
            bm_data = load_build_manifest(bm_path)
        except (OSError, json.JSONDecodeError):
            continue
        bm_mode = _extract_manifest_indirect_dispatch(bm_data)
        if bm_mode == "shard":
            return IndirectDispatchDetection(mode="shard", source=BUILD_MANIFEST_FILE)
        if bm_mode == "table":
            saw_build_table = True
    if saw_build_table:
        return IndirectDispatchDetection(mode="table", source=BUILD_MANIFEST_FILE)

    program_cpp_paths: list[Path] = [
        unit.path
        for unit in units
        if Path(unit.unit_id.replace("\\", "/")).name == "pg6_program.cpp"
    ]
    root_prog_cpp = source_root / "pg6_program.cpp"
    if root_prog_cpp.is_file() and root_prog_cpp.resolve() not in {p.resolve() for p in program_cpp_paths}:
        program_cpp_paths.append(root_prog_cpp)
    for cpp_path in program_cpp_paths:
        if _file_head_tail_contains(
            cpp_path,
            (SHARD_DISPATCH_TABLE_PREFIX.encode("ascii"), b"Pg6ShardDispatchTable"),
        ):
            return IndirectDispatchDetection(mode="shard", source="pg6_program.cpp")

    reg_needles = (
        SHARD_DISPATCH_TABLE_PREFIX.encode("ascii"),
        SHARD_DISPATCH_FN_PREFIX.encode("ascii"),
    )
    for unit in units:
        if is_registration_unit(unit) and _file_head_tail_contains(unit.path, reg_needles):
            return IndirectDispatchDetection(mode="shard", source="functions.ll")

    shard_needles = (SHARD_DISPATCH_FN_PREFIX.encode("ascii"),)
    checked_shards = 0
    for unit in units:
        if not is_shard_unit(unit):
            continue
        if _file_head_tail_contains(unit.path, shard_needles):
            return IndirectDispatchDetection(mode="shard", source="shard-ir")
        checked_shards += 1
        if checked_shards >= 4:
            break

    return IndirectDispatchDetection(mode=DEFAULT_INDIRECT_DISPATCH, source="default")


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


_HEX_OR_DASH_RE = re.compile(r"^[0-9a-fA-F-]+$")


def is_required_side_export_symbol(
    symbol: str,
    *,
    required_prefixes: Sequence[str] = SIDE_MODULE_REQUIRED_EXPORT_PREFIXES,
    required_names: frozenset[str] = SIDE_MODULE_REQUIRED_EXPORT_NAMES,
) -> bool:
    """Return True for dynamic-init/registration/dispatch roots always kept exported."""
    if symbol in required_names:
        return True
    prefixes = tuple(required_prefixes)
    return bool(prefixes) and symbol.startswith(prefixes)


def parse_llvm_nm_output(stdout: str) -> tuple[set[str], set[str]]:
    """Parse BSD (default) or POSIX `llvm-nm` output into `(defined_globals, undefined)`."""
    defined: set[str] = set()
    undefined: set[str] = set()
    for raw_line in stdout.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.endswith(":"):
            continue
        parts = stripped.split()
        sym_type: str | None = None
        sym_name: str | None = None
        if (
            len(parts) >= 3
            and len(parts[1]) == 1
            and parts[1].isalpha()
            and _HEX_OR_DASH_RE.match(parts[0]) is not None
        ):
            # BSD format with address or dashes: `00000000 T sub_140001000` / `-------- D foo`
            sym_type = parts[1]
            sym_name = parts[2]
        elif len(parts) == 2 and len(parts[0]) == 1 and parts[0].isalpha():
            # BSD format without address (common on wasm objects / `U`): `T sub_...` / `U sub_...`
            sym_type = parts[0]
            sym_name = parts[1]
        elif len(parts) >= 2 and len(parts[1]) == 1 and parts[1].isalpha():
            # POSIX format (`-P`): `sub_140001000 T 0 0` / `sub_140009000 U`
            sym_type = parts[1]
            sym_name = parts[0]
        else:
            continue
        if not sym_name:
            continue
        if sym_type in LLVM_NM_DEFINED_GLOBAL_TYPES:
            defined.add(sym_name)
        elif sym_type in LLVM_NM_UNDEFINED_TYPES:
            undefined.add(sym_name)
    return defined, undefined


def _chunk_object_paths(
    objects: Sequence[Path | str],
    *,
    max_cmd_chars: int = LLVM_NM_MAX_CMD_CHARS,
) -> list[list[str]]:
    """Split object paths into command-line-safe batches below Windows' 32 KiB CreateProcess cap."""
    batches: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for obj in objects:
        token = str(obj)
        token_len = len(token) + 3
        if current and current_chars + token_len > max_cmd_chars:
            batches.append(current)
            current = [token]
            current_chars = token_len
        else:
            current.append(token)
            current_chars += token_len
    if current:
        batches.append(current)
    return batches


def _inspect_objects_with_llvm_nm(
    objects: Sequence[Path | str],
    *,
    llvm_nm_path: Path | str,
    run_command: CommandRunner,
    max_cmd_chars: int = LLVM_NM_MAX_CMD_CHARS,
) -> tuple[set[str], set[str]]:
    defined: set[str] = set()
    undefined: set[str] = set()
    if not objects:
        return defined, undefined
    for batch in _chunk_object_paths(objects, max_cmd_chars=max_cmd_chars):
        argv = (str(llvm_nm_path), "--extern-only", "--no-sort", *batch)
        result = run_command(argv)
        if result.returncode != 0:
            err = sanitize_text((result.stderr or result.stdout or "llvm-nm failed").strip())
            raise BuildGuestWasmError(f"llvm-nm failed (exit {result.returncode}): {err}")
        batch_def, batch_undef = parse_llvm_nm_output(result.stdout)
        defined.update(batch_def)
        undefined.update(batch_undef)
    return defined, undefined


def compute_side_module_cross_exports(
    side_buckets: Sequence[Sequence[Path | str]],
    main_objects: Sequence[Path | str],
    llvm_nm_path: Path | str,
    *,
    run_command: CommandRunner | None = None,
    required_prefixes: Sequence[str] = SIDE_MODULE_REQUIRED_EXPORT_PREFIXES,
    required_names: frozenset[str] = SIDE_MODULE_REQUIRED_EXPORT_NAMES,
    max_cmd_chars: int = LLVM_NM_MAX_CMD_CHARS,
) -> list[list[str]]:
    """Compute per-bucket required cross-module exports for `-sSIDE_MODULE=2` via `llvm-nm`.

    For each side bucket `i`, returns the deterministically sorted list of global symbols
    defined (`T`, `W`, `D`, `B`, `R`, `C`, `V`) in bucket `i` that are either:
      - referenced as undefined (`U`, `w`) by ANY other side bucket `j != i`,
      - referenced as undefined (`U`, `w`) by `main_objects`, or
      - matched by required dynamic-init / registration / dispatch / entry prefixes or names.

    Intra-bucket symbols that are never referenced outside bucket `i` are omitted so
    `wasm-ld` under `-sSIDE_MODULE=2` prunes them from the Wasm export section.
    """
    runner = run_command or default_run_command
    bucket_defined: list[set[str]] = []
    bucket_undefined: list[set[str]] = []
    for bucket in side_buckets:
        b_def, b_undef = _inspect_objects_with_llvm_nm(
            bucket,
            llvm_nm_path=llvm_nm_path,
            run_command=runner,
            max_cmd_chars=max_cmd_chars,
        )
        bucket_defined.append(b_def)
        bucket_undefined.append(b_undef)

    _, main_undefined = _inspect_objects_with_llvm_nm(
        main_objects,
        llvm_nm_path=llvm_nm_path,
        run_command=runner,
        max_cmd_chars=max_cmd_chars,
    )

    undef_ref_counts: dict[str, int] = {}
    for b_undef in bucket_undefined:
        for sym in b_undef:
            undef_ref_counts[sym] = undef_ref_counts.get(sym, 0) + 1

    req_prefixes_tuple = tuple(required_prefixes)
    exports_per_bucket: list[list[str]] = []
    for idx, b_def in enumerate(bucket_defined):
        own_undef = bucket_undefined[idx]
        needed: list[str] = []
        for sym in b_def:
            in_main = sym in main_undefined
            in_other_bucket = undef_ref_counts.get(sym, 0) > (1 if sym in own_undef else 0)
            is_required = is_required_side_export_symbol(
                sym,
                required_prefixes=req_prefixes_tuple,
                required_names=required_names,
            )
            if in_main or in_other_bucket or is_required:
                needed.append(sym)
        needed.sort()
        exports_per_bucket.append(needed)
    return exports_per_bucket


def write_side_exports_response_file(
    path: Path,
    symbols: Sequence[str],
    *,
    include_ctors: bool = True,
) -> Path:
    """Write `-Wl,--export=<sym>` flags (plus `__wasm_call_ctors` root) to a linker `.rsp` file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if include_ctors:
        lines.append(SIDE_MODULE_CTOR_EXPORT_FLAG)
    seen: set[str] = set()
    for sym in symbols:
        if not sym or sym in seen:
            continue
        seen.add(sym)
        if sym == "__wasm_call_ctors" and include_ctors:
            continue
        flag = f"-Wl,--export={sym}"
        if any(ch.isspace() for ch in flag):
            escaped = flag.replace('"', '\\"')
            flag = f'"{escaped}"'
        lines.append(flag)
    text = "\n".join(lines) + ("\n" if lines else "")
    path.write_text(text, encoding="utf-8")
    return path


def check_side_module_not_empty(
    wasm_bytes: int,
    *,
    input_object_bytes: int,
    module_name: str,
    side_module_mode: int = 1,
    min_output_bytes: int = SIDE_MODULE_MIN_OUTPUT_BYTES,
    min_input_bytes: int = SIDE_MODULE_MIN_INPUT_BYTES_FOR_SIZE_GUARD,
) -> str | None:
    """Return an error string if a linked side module looks stripped/empty relative to its inputs."""
    if input_object_bytes <= min_input_bytes:
        return None
    if wasm_bytes >= min_output_bytes:
        return None
    mode_hint = (
        "under -sSIDE_MODULE=2 (check cross-module llvm-nm export response file and root symbols)"
        if side_module_mode == 2
        else "during side-module linking"
    )
    return (
        f"{module_name} linked to only {wasm_bytes} bytes from {input_object_bytes} bytes of "
        f"input objects (threshold: output >= {min_output_bytes} bytes when inputs > "
        f"{min_input_bytes} bytes); all guest functions appear to have been stripped {mode_hint}"
    )


def resolve_link_flags(
    *,
    opt: str = "O2",
    strip: bool = False,
    pthread_pool_size: int | None = None,
    extra_link_flags: Sequence[str] = (),
    no_link_opt: bool = False,
    wasm_exceptions: bool = False,
    initial_memory: str | None = None,
    stack_size: str | None = None,
    assertions: int | None = None,
    browser_boot_exports: bool = False,
    main_module: bool = False,
    proxy_to_pthread: bool = False,
) -> tuple[str, ...]:
    """Compute effective linker flags for em++ from defaults and build options."""
    if pthread_pool_size is not None:
        if isinstance(pthread_pool_size, bool) or pthread_pool_size < 1:
            raise BuildGuestWasmError("--pthread-pool-size must be >= 1")
    effective_pool_size = (
        pthread_pool_size
        if pthread_pool_size is not None
        else (SPLIT_DEFAULT_PTHREAD_POOL_SIZE if main_module else None)
    )
    initial_memory_token = (
        parse_size_token(initial_memory, flag_name="--initial-memory")
        if initial_memory is not None
        else (SPLIT_DEFAULT_INITIAL_MEMORY if main_module else None)
    )
    stack_size_token = (
        parse_size_token(stack_size, flag_name="--stack-size")
        if stack_size is not None
        else None
    )
    assertions_val = parse_assertions(assertions)

    flags: list[str] = []
    replaced_pool = False
    for flag in LINK_FLAGS:
        if effective_pool_size is not None and flag.startswith(PTHREAD_POOL_SIZE_PREFIX):
            flags.append(f"{PTHREAD_POOL_SIZE_PREFIX}{effective_pool_size}")
            replaced_pool = True
        else:
            flags.append(flag)
    if effective_pool_size is not None and not replaced_pool:
        flags.append(f"{PTHREAD_POOL_SIZE_PREFIX}{effective_pool_size}")

    if main_module:
        for growth_flag in ("-sALLOW_MEMORY_GROWTH=1", "-sMAXIMUM_MEMORY=16GB"):
            if growth_flag not in flags:
                flags.append(growth_flag)
        if MAIN_MODULE_FLAG not in flags:
            flags.append(MAIN_MODULE_FLAG)

    if not no_link_opt and opt in LINK_OPT_LEVELS:
        opt_flag = f"-{opt}"
        if opt_flag not in flags:
            flags.append(opt_flag)

    if wasm_exceptions and WASM_EXCEPTIONS_FLAG not in flags:
        flags.append(WASM_EXCEPTIONS_FLAG)

    if initial_memory_token is not None:
        init_mem_flag = f"{INITIAL_MEMORY_PREFIX}{initial_memory_token}"
        if init_mem_flag not in flags:
            flags.append(init_mem_flag)

    if stack_size_token is not None:
        for stack_flag in (
            f"{STACK_SIZE_PREFIX}{stack_size_token}",
            f"{DEFAULT_PTHREAD_STACK_SIZE_PREFIX}{stack_size_token}",
        ):
            if stack_flag not in flags:
                flags.append(stack_flag)

    if main_module:
        split_incoming_flag = f"-sINCOMING_MODULE_JS_API={SPLIT_INCOMING_MODULE_JS_API}"
        if split_incoming_flag not in flags:
            flags.append(split_incoming_flag)
        if browser_boot_exports:
            for export_flag in (
                f"-sEXPORTED_RUNTIME_METHODS={BROWSER_BOOT_EXPORTED_RUNTIME_METHODS}",
                f"-sEXPORTED_FUNCTIONS={BROWSER_BOOT_EXPORTED_FUNCTIONS}",
            ):
                if export_flag not in flags:
                    flags.append(export_flag)
    elif browser_boot_exports:
        for export_flag in BROWSER_BOOT_EXPORT_LINK_FLAGS:
            if export_flag not in flags:
                flags.append(export_flag)

    if strip:
        for strip_flag in STRIP_LINK_FLAGS:
            if assertions_val is not None and strip_flag.startswith(ASSERTIONS_PREFIX):
                continue
            if strip_flag not in flags:
                flags.append(strip_flag)

    if assertions_val is None and not any(f.startswith(ASSERTIONS_PREFIX) for f in flags):
        # Release default: no runtime checks (pass --assertions 1 for a debug build).
        flags.extend(["-sASSERTIONS=0", "-sSAFE_HEAP=0"])
    if assertions_val is not None:
        flags = [f for f in flags if not f.startswith(ASSERTIONS_PREFIX)]
        assertions_flag = f"{ASSERTIONS_PREFIX}{assertions_val}"
        if assertions_flag not in flags:
            flags.append(assertions_flag)

    if proxy_to_pthread and PROXY_TO_PTHREAD_FLAG not in flags:
        flags.append(PROXY_TO_PTHREAD_FLAG)

    for extra in extra_link_flags:
        if extra:
            flags.append(extra)

    return tuple(flags)


def resolve_side_module_link_flags(
    *,
    opt: str = "O2",
    strip: bool = False,
    no_link_opt: bool = False,
    wasm_exceptions: bool = False,
    pruned: bool = False,
) -> tuple[str, ...]:
    """Compute em++ linker flags for a guest shard side module.

    Uses `-sSIDE_MODULE=1` by default. When `pruned=True` (asymmetric browser split), uses
    `-sSIDE_MODULE=2` so Emscripten drops unused internal `sub_*` exports.
    """
    flags: list[str] = list(
        SIDE_MODULE_PRUNED_LINK_FLAGS if pruned else SIDE_MODULE_LINK_FLAGS
    )
    if not no_link_opt and opt in LINK_OPT_LEVELS:
        opt_flag = f"-{opt}"
        if opt_flag not in flags:
            flags.append(opt_flag)
    if wasm_exceptions and WASM_EXCEPTIONS_FLAG not in flags:
        flags.append(WASM_EXCEPTIONS_FLAG)
    if strip and "-Wl,--strip-all" not in flags:
        flags.append("-Wl,--strip-all")
    return tuple(flags)


def check_v8_wasm_size(
    wasm_bytes: int,
    *,
    limit: int | None = None,
    module_name: str = RUNTIME_WASM,
) -> tuple[bool, str | None]:
    """Check whether a linked .wasm module exceeds Chrome/V8's 1 GiB module cap."""
    max_bytes = V8_MAX_WASM_MODULE_BYTES if limit is None else limit
    if wasm_bytes <= max_bytes:
        return False, None
    mib = wasm_bytes / (1024 * 1024)
    over_mib = (wasm_bytes - max_bytes) / (1024 * 1024)
    warning = (
        f"{module_name} size is {wasm_bytes} bytes ({mib:.2f} MiB), exceeding "
        f"Chrome/V8's 1 GiB ({max_bytes} bytes) WebAssembly.instantiate() module size limit "
        f"by {over_mib:.2f} MiB; rebuild with --strip, --opt Oz (or --opt Os), "
        f"and/or --split-modules N to keep every module under 1 GiB"
    )
    return True, warning


def build_side_module_link_argv(
    linker: Path | str,
    objects: Sequence[Path],
    *,
    output_wasm: Path,
    side_link_flags: Sequence[str] | None = None,
    response_file: Path | None = None,
    exports_response_file: Path | None = None,
    opt: str = "O2",
    strip: bool = False,
    no_link_opt: bool = False,
    wasm_exceptions: bool = False,
    pruned: bool = False,
) -> tuple[str, ...]:
    if side_link_flags is None:
        effective_flags = resolve_side_module_link_flags(
            opt=opt,
            strip=strip,
            no_link_opt=no_link_opt,
            wasm_exceptions=wasm_exceptions,
            pruned=pruned,
        )
    else:
        effective_flags = tuple(side_link_flags)
    argv: list[str] = [str(linker), *effective_flags]
    if exports_response_file is not None:
        exp_rsp_arg = str(exports_response_file).replace("\\", "/")
        argv.append(f"@{exp_rsp_arg}")
    if response_file is not None:
        rsp_arg = str(response_file).replace("\\", "/")
        argv.append(f"@{rsp_arg}")
    else:
        argv.extend(str(path) for path in objects)
    argv.extend(["-o", str(output_wasm)])
    return tuple(argv)


def split_filtered_archive_path(cache_dir: Path, archive: Path) -> Path:
    """Return the cache-dir copy path for a split-module filtered runtime archive."""
    return cache_dir / f"{archive.stem}_split{archive.suffix}"


def prepare_split_runtime_archive(
    archive: Path,
    *,
    cache_dir: Path,
    archiver: Path | str,
    excludes: Sequence[str] = SPLIT_RUNTIME_ARCHIVE_EXCLUDES,
    dry_run: bool = False,
    run_command: CommandRunner | None = None,
) -> tuple[Path, tuple[str, ...], CommandResult]:
    """Copy `archive` into `cache_dir` and strip boot members via `emar d`.

    Never mutates the original archive. Under `dry_run=True`, returns the planned
    cache copy path and `emar d` command without touching disk or running `emar`.
    """
    filtered_path = split_filtered_archive_path(cache_dir, archive)
    argv: tuple[str, ...] = (str(archiver), "d", str(filtered_path), *excludes)
    if dry_run:
        return filtered_path, argv, CommandResult(returncode=0)
    if not archive.is_file():
        return (
            filtered_path,
            argv,
            CommandResult(returncode=1, stderr=f"runtime archive not found: {archive.name}"),
        )
    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    if filtered_path.exists():
        filtered_path.unlink()
    shutil.copyfile(archive, filtered_path)
    runner = run_command or default_run_command
    result = runner(argv)
    return filtered_path, argv, result


def build_link_argv(
    linker: Path | str,
    objects: Sequence[Path],
    *,
    output_js: Path,
    runtime_libs: Sequence[Path] = (),
    side_modules: Sequence[Path] = (),
    link_flags: Sequence[str] | None = None,
    extra_flags: Sequence[str] = (),
    response_file: Path | None = None,
    opt: str = "O2",
    strip: bool = False,
    pthread_pool_size: int | None = None,
    no_link_opt: bool = False,
    wasm_exceptions: bool = False,
    initial_memory: str | None = None,
    stack_size: str | None = None,
    assertions: int | None = None,
    browser_boot_exports: bool = False,
    main_module: bool = False,
    whole_archive_libs: bool = False,
    proxy_to_pthread: bool = False,
) -> tuple[str, ...]:
    if link_flags is None:
        effective_flags = resolve_link_flags(
            opt=opt,
            strip=strip,
            pthread_pool_size=pthread_pool_size,
            extra_link_flags=extra_flags,
            no_link_opt=no_link_opt,
            wasm_exceptions=wasm_exceptions,
            initial_memory=initial_memory,
            stack_size=stack_size,
            assertions=assertions,
            browser_boot_exports=browser_boot_exports,
            main_module=main_module,
            proxy_to_pthread=proxy_to_pthread,
        )
    else:
        effective_flags = (*link_flags, *extra_flags)
    use_whole_archive = whole_archive_libs or (link_flags is None and main_module)
    argv: list[str] = [str(linker), *effective_flags]
    if response_file is not None:
        rsp_arg = str(response_file).replace("\\", "/")
        argv.append(f"@{rsp_arg}")
    else:
        argv.extend(str(path) for path in objects)
    for lib in runtime_libs:
        if use_whole_archive and lib.suffix.lower() == ".a" and not _is_companion_static_lib(lib):
            argv.extend(["-Wl,--whole-archive", str(lib), "-Wl,--no-whole-archive"])
        else:
            argv.append(str(lib))
    argv.extend(str(path) for path in side_modules)
    argv.extend(["-o", str(output_js)])
    return tuple(argv)


# ---------------------------------------------------------------------------
# Split-module JS post-link optimizer (R33)
# ---------------------------------------------------------------------------

_STUB_CREATOR_RE = re.compile(
    r"(?P<prefix>if\s*\(\s*!\s*\(\s*(?P<prop>[A-Za-z_$][\w$]*)\s+in\s+(?P<stubs>[A-Za-z_$][\w$]*)\s*\)\s*\)\s*\{\s*"
    r"(?:var|let)\s+(?P<resolved>[A-Za-z_$][\w$]*)\s*;\s*)"
    r"(?P<stub_assign>(?P=stubs)\s*\[\s*(?P=prop)\s*\]\s*=\s*\(\s*\.\.\.(?P<args>[A-Za-z_$][\w$]*)\s*\)\s*=>\s*\{\s*"
    r"(?P=resolved)\s*\|\|=\s*(?P<resolve_fn>[A-Za-z_$][\w$]*)\s*\(\s*(?P=prop)\s*\)\s*;\s*"
    r"return\s+(?P=resolved)\s*\(\s*\.\.\.(?P=args)\s*\)\s*;?\s*\}\s*;?)"
)

_MODULE_EXPORTS_DECL_RE = re.compile(r"\b((?:var|let)\s+moduleExports\s*;)")

_AWAIT_LOAD_DYLIBS_RE = re.compile(r"\bawait\s+loadDylibs\s*\(\s*\)\s*;")

_WORKER_POST_MESSAGE_OBJ_RE = re.compile(
    r"(?P<worker>[A-Za-z_$][\w$]*)\s*\.\s*postMessage\s*\(\s*\{"
)

_CMD_LOAD_PROP_RE = re.compile(
    r"(?:^|[{,\s])(?:['\"])?cmd(?:['\"])?\s*:\s*(?:CMD_LOAD|['\"]load['\"])(?=\s*[,}])"
)

_CASE_CMD_LOADED_RE = re.compile(r"(\bcase\s+(?:CMD_LOADED|['\"]loaded['\"])\s*:)")

_IF_CMD_LOADED_RE = re.compile(
    r"(\bif\s*\(\s*(?:[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*===?\s*"
    r"(?:CMD_LOADED|['\"]loaded['\"])\s*\)\s*\{)"
)


def _find_balanced_call_end(js_text: str, open_paren_idx: int) -> int | None:
    """Return index of the matching `)` for `js_text[open_paren_idx] == '('`."""
    n = len(js_text)
    if open_paren_idx < 0 or open_paren_idx >= n or js_text[open_paren_idx] != "(":
        return None
    depth = 0
    idx = open_paren_idx
    quote: str | None = None
    while idx < n:
        ch = js_text[idx]
        if quote is not None:
            if ch == "\\" and idx + 1 < n:
                idx += 2
                continue
            if ch == quote:
                quote = None
            idx += 1
            continue
        if ch == "/" and idx + 1 < n:
            nxt = js_text[idx + 1]
            if nxt == "/":
                nl = js_text.find("\n", idx + 2)
                if nl == -1:
                    return None
                idx = nl + 1
                continue
            if nxt == "*":
                end_comment = js_text.find("*/", idx + 2)
                if end_comment == -1:
                    return None
                idx = end_comment + 2
                continue
        if ch in ("'", '"', "`"):
            quote = ch
            idx += 1
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return idx
        idx += 1
    return None


def _patch_pthread_cmd_load_throttle(
    js_text: str,
    *,
    max_concurrent: int,
) -> tuple[str, bool, bool]:
    """Atomically patch `CMD_LOAD` throttling + `CMD_LOADED` queue dispatch.

    Returns `(updated_js, throttle_applied, dispatch_applied)`. If either site is
    missing, both edits are skipped so queued workers can never deadlock.
    """
    load_span: tuple[int, int, str, str] | None = None
    for match in _WORKER_POST_MESSAGE_OBJ_RE.finditer(js_text):
        open_paren = js_text.find("(", match.start())
        close_paren = _find_balanced_call_end(js_text, open_paren)
        if close_paren is None:
            continue
        args_text = js_text[open_paren + 1 : close_paren]
        if not _CMD_LOAD_PROP_RE.search(args_text):
            continue
        end_idx = close_paren + 1
        while end_idx < len(js_text) and js_text[end_idx] in " \t":
            end_idx += 1
        if end_idx < len(js_text) and js_text[end_idx] == ";":
            end_idx += 1
        else:
            end_idx = close_paren + 1
        load_span = (match.start(), end_idx, match.group("worker"), args_text)
        break

    if load_span is None:
        return js_text, False, False

    loaded_re: re.Pattern[str] | None = None
    if _CASE_CMD_LOADED_RE.search(js_text) is not None:
        loaded_re = _CASE_CMD_LOADED_RE
    elif _IF_CMD_LOADED_RE.search(js_text) is not None:
        loaded_re = _IF_CMD_LOADED_RE
    if loaded_re is None:
        return js_text, False, False

    start_idx, end_idx, worker_var, args_text = load_span
    throttled_block = (
        "{ (() => {"
        f" var _w = {worker_var};"
        f" var _loadArgs = [{args_text}];"
        f" var _maxLoads = {max_concurrent};"
        " PThread._loadQueue ||= [];"
        " PThread._activeLoads ||= 0;"
        " PThread._dispatchCmdLoad ||= () => {"
        " while (PThread._activeLoads < _maxLoads && PThread._loadQueue.length > 0) {"
        " var _next = PThread._loadQueue.shift();"
        " PThread._activeLoads++;"
        " _next();"
        " }"
        " };"
        " var _hasOwnPost = Object.prototype.hasOwnProperty.call(_w, 'postMessage');"
        " var _origPost = _w.postMessage;"
        " var _pending = [];"
        " var _dispatched = false;"
        " var _sendLoad = () => {"
        " if (_dispatched) return;"
        " _dispatched = true;"
        " if (_hasOwnPost) { _w.postMessage = _origPost; } else { delete _w.postMessage; }"
        " _origPost.apply(_w, _loadArgs);"
        " for (var _i = 0; _i < _pending.length; _i++) { _origPost.apply(_w, _pending[_i]); }"
        " _pending.length = 0;"
        " };"
        " _w.postMessage = (..._a) => {"
        " if (_dispatched) return _origPost.apply(_w, _a);"
        " _pending.push(_a);"
        " };"
        " PThread._loadQueue.push(_sendLoad);"
        " PThread._dispatchCmdLoad();"
        " })(); }"
    )
    updated = js_text[:start_idx] + throttled_block + js_text[end_idx:]

    loaded_hook = (
        " if (typeof PThread !== 'undefined' && PThread) {"
        " if (PThread._activeLoads > 0) PThread._activeLoads--;"
        " if (typeof PThread._dispatchCmdLoad === 'function') PThread._dispatchCmdLoad();"
        " }"
    )
    updated, dispatch_count = loaded_re.subn(
        lambda m: f"{m.group(1)}{loaded_hook}",
        updated,
        count=1,
    )
    if dispatch_count == 0:
        return js_text, False, False
    return updated, True, True


def optimize_split_runtime_js(
    js_text: str,
    *,
    max_concurrent_pthread_loads: int = DEFAULT_MAX_CONCURRENT_PTHREAD_LOADS,
) -> tuple[str, dict[str, bool]]:
    """Apply idempotent R33 V8-cage GC & throttled pthread `CMD_LOAD` patches to split `runtime.js`.

    Patch 1 (V8 4 GiB pointer-compression cage cleanup + T97 zero-alloc fixed-arity stubs):
      - Registers eager cross-shard stub binders inside `instantiateModule`'s `proxyHandler`
        (`globalThis.__pg6StubBinders`) and replaces Emscripten's rest-parameter `(...args)`
        stub wrapper with fixed-arity zero-allocation wrappers chosen once at stub creation time
        (`(a) => ...` for `sub_*`, `(a, b) => ...` for `__playgta6_dispatch_shard_*`, and
        `(...args) => ...` fallback for all other symbols).
      - Registers `moduleExports`/`localScope` clearers right after `var moduleExports;`
        (`globalThis.__pg6ExportClearers`).
      - Immediately after `await loadDylibs();`, binds all lazy stubs, clears `moduleExports`
        and `localScope`, empties `LDSO.loadedLibsByName[lib].exports` for
        `guest_part_\\d+\\.wasm`, deletes `sub_*` keys from `wasmImports` and `GOT`, and clears
        `functionsInTableMap`.

    Patch 2 (Ordered throttled pthread `CMD_LOAD`, when `max_concurrent_pthread_loads > 0`):
      - Limits concurrent `worker.postMessage({ cmd: CMD_LOAD, ... })` calls to
        `max_concurrent_pthread_loads` via `PThread._loadQueue` / `PThread._activeLoads`.
      - Wraps `worker.postMessage` until `_dispatchCmdLoad` posts `CMD_LOAD`, buffering any
        early `CMD_RUN` or other messages and flushing them strictly AFTER `CMD_LOAD`.
      - Decrements `PThread._activeLoads` and dispatches the next queued load on `CMD_LOADED`.
    """
    max_loads = parse_max_concurrent_pthread_loads(max_concurrent_pthread_loads)
    status: dict[str, bool] = {
        "already_patched": False,
        "stub_binders": False,
        "export_clearers": False,
        "post_load_dylibs_cleanup": False,
        "pthread_cmd_load_throttle": False,
        "pthread_cmd_loaded_dispatch": False,
    }
    if SPLIT_PTHREAD_OPT_MARKER in js_text:
        status["already_patched"] = True
        return js_text, status

    updated = js_text

    # Patch 1: apply atomically only when both lazy stub creator and `await loadDylibs();` exist
    # so we never clear `wasmImports`/`moduleExports` without eager stub binding first.
    if (
        _STUB_CREATOR_RE.search(updated) is not None
        and _AWAIT_LOAD_DYLIBS_RE.search(updated) is not None
    ):
        def _inject_stub_binder(match: re.Match[str]) -> str:
            prop = match.group("prop")
            stubs = match.group("stubs")
            resolved = match.group("resolved")
            resolve_fn = match.group("resolve_fn")
            args_name = match.group("args")
            used = {prop, stubs, resolved, resolve_fn, args_name}
            p_var = "_p" if "_p" not in used else "_pg6p"
            used.add(p_var)
            e_var = "_e" if "_e" not in used else "_pg6e"
            arg1 = "a" if "a" not in used else "_a0"
            arg2 = "b" if "b" not in used and "b" != arg1 else "_a1"
            binder = (
                f"const {p_var} = {prop}; "
                "(globalThis.__pg6StubBinders ||= []).push(() => { "
                f"try {{ {resolved} ||= {resolve_fn}({p_var}); "
                f"if ({resolved}) {stubs}[{p_var}] = {resolved}; }} catch ({e_var}) {{}} "
                "}); "
            )
            # T97: specialize hot guest cross-part stubs at creation time to avoid per-call
            # `...args` rest-array allocation and spread overhead on every invocation.
            fast_stub_assign = (
                f"{stubs}[{p_var}] = "
                f"(typeof {p_var} === 'string' && {p_var}.startsWith('sub_')) "
                f"? ({arg1}) => ({resolved} || ({resolved} = {resolve_fn}({p_var})))({arg1}) "
                f": (typeof {p_var} === 'string' && {p_var}.startsWith('{SHARD_DISPATCH_FN_PREFIX}')) "
                f"? ({arg1}, {arg2}) => ({resolved} || ({resolved} = {resolve_fn}({p_var})))({arg1}, {arg2}) "
                f": (...{args_name}) => ({resolved} || ({resolved} = {resolve_fn}({p_var})))(...{args_name});"
            )
            return f"{match.group('prefix')}{binder}{fast_stub_assign}"

        updated, stub_count = _STUB_CREATOR_RE.subn(_inject_stub_binder, updated)
        status["stub_binders"] = stub_count > 0

        clearer_snippet = (
            " (globalThis.__pg6ExportClearers ||= []).push(() => {"
            " moduleExports = null;"
            " if (typeof localScope !== 'undefined') localScope = null;"
            " });"
        )
        updated, clearer_count = _MODULE_EXPORTS_DECL_RE.subn(
            lambda m: f"{m.group(1)}{clearer_snippet}",
            updated,
        )
        status["export_clearers"] = clearer_count > 0

        post_cleanup = (
            "{ await loadDylibs(); (() => {"
            " var _binders = globalThis.__pg6StubBinders;"
            " if (_binders) {"
            " for (var _i = 0; _i < _binders.length; _i++) {"
            " try { _binders[_i](); } catch (_e) {}"
            " }"
            " _binders.length = 0;"
            " globalThis.__pg6StubBinders = null;"
            " }"
            " var _clearers = globalThis.__pg6ExportClearers;"
            " if (_clearers) {"
            " for (var _j = 0; _j < _clearers.length; _j++) {"
            " try { _clearers[_j](); } catch (_e) {}"
            " }"
            " _clearers.length = 0;"
            " globalThis.__pg6ExportClearers = null;"
            " }"
            " if (typeof LDSO !== 'undefined' && LDSO && LDSO.loadedLibsByName) {"
            " for (var _lib in LDSO.loadedLibsByName) {"
            " if (/guest_part_\\d+\\.wasm$/.test(_lib) && LDSO.loadedLibsByName[_lib]) {"
            " LDSO.loadedLibsByName[_lib].exports = {};"
            " }"
            " }"
            " }"
            " if (typeof wasmImports !== 'undefined' && wasmImports) {"
            " for (var _k in wasmImports) {"
            " if (_k.startsWith('sub_')) delete wasmImports[_k];"
            " }"
            " }"
            " if (typeof GOT !== 'undefined' && GOT) {"
            " for (var _g in GOT) {"
            " if (_g.startsWith('sub_')) delete GOT[_g];"
            " }"
            " }"
            " if (typeof functionsInTableMap !== 'undefined' && functionsInTableMap) {"
            " if (typeof functionsInTableMap.clear === 'function') {"
            " functionsInTableMap.clear();"
            " } else {"
            " try { functionsInTableMap = undefined; } catch (_e) {}"
            " }"
            " }"
            " })(); }"
        )
        updated, cleanup_count = _AWAIT_LOAD_DYLIBS_RE.subn(
            lambda _m: post_cleanup,
            updated,
        )
        status["post_load_dylibs_cleanup"] = cleanup_count > 0

    # Patch 2: ordered throttled pthread CMD_LOAD (disabled when max_loads == 0).
    if max_loads > 0:
        updated, throttle_ok, dispatch_ok = _patch_pthread_cmd_load_throttle(
            updated,
            max_concurrent=max_loads,
        )
        status["pthread_cmd_load_throttle"] = throttle_ok
        status["pthread_cmd_loaded_dispatch"] = dispatch_ok

    any_applied = any(v for k, v in status.items() if k != "already_patched")
    if any_applied:
        updated = f"{SPLIT_PTHREAD_OPT_MARKER}\n{updated}"
    return updated, status


def patch_split_runtime_js_file(
    runtime_js_path: Path,
    *,
    max_concurrent_pthread_loads: int = DEFAULT_MAX_CONCURRENT_PTHREAD_LOADS,
) -> dict[str, bool]:
    """Read, optimize in-place, and return patch status for a split-module `runtime.js` file."""
    max_loads = parse_max_concurrent_pthread_loads(max_concurrent_pthread_loads)
    if not runtime_js_path.is_file():
        return {
            "already_patched": False,
            "stub_binders": False,
            "export_clearers": False,
            "post_load_dylibs_cleanup": False,
            "pthread_cmd_load_throttle": False,
            "pthread_cmd_loaded_dispatch": False,
        }
    original = runtime_js_path.read_text(encoding="utf-8")
    patched, status = optimize_split_runtime_js(
        original,
        max_concurrent_pthread_loads=max_loads,
    )
    if patched != original:
        runtime_js_path.write_text(patched, encoding="utf-8")
    return status


# ---------------------------------------------------------------------------
# Pure-Python Wasm binary section inspector & guest_manifest.json writer (R37d)
# ---------------------------------------------------------------------------


class _WasmStreamCursor:
    """Bounds-checked binary cursor over a stream slice (`BinaryIO` or `BytesIO`)."""

    __slots__ = ("_fh", "_pos", "_end")

    def __init__(self, fh: BinaryIO, start: int, end: int) -> None:
        self._fh = fh
        self._pos = start
        self._end = end
        self._fh.seek(start)

    @property
    def pos(self) -> int:
        return self._pos

    @property
    def remaining(self) -> int:
        return self._end - self._pos

    def skip(self, length: int, *, context: str = "bytes") -> None:
        if length < 0 or self._pos + length > self._end:
            raise WasmInspectError(
                f"truncated {context}: need {length} bytes, only {self.remaining} available"
            )
        self._pos += length
        self._fh.seek(self._pos)

    def read_bytes(self, length: int, *, context: str = "bytes") -> bytes:
        if length < 0 or self._pos + length > self._end:
            raise WasmInspectError(
                f"unexpected EOF reading {context}: need {length} bytes, only {self.remaining} available"
            )
        data = self._fh.read(length)
        if len(data) != length:
            raise WasmInspectError(
                f"short read for {context}: expected {length} bytes, got {len(data)}"
            )
        self._pos += length
        return data

    def read_u8(self, *, context: str = "u8") -> int:
        return self.read_bytes(1, context=context)[0]

    def read_uleb128(self, *, max_bits: int, context: str = "u32") -> int:
        result = 0
        shift = 0
        max_bytes = (max_bits + 6) // 7
        for idx in range(max_bytes):
            byte = self.read_u8(context=f"ULEB128 {context}")
            if idx == max_bytes - 1:
                if byte & 0x80:
                    raise WasmInspectError(f"overlong ULEB128 {context}")
                remaining_bits = max_bits - shift
                overflow_mask = ~((1 << remaining_bits) - 1) & 0x7F
                if byte & overflow_mask:
                    raise WasmInspectError(f"ULEB128 {context} overflows {max_bits} bits")
            result |= (byte & 0x7F) << shift
            if (byte & 0x80) == 0:
                return result
            shift += 7
        raise WasmInspectError(f"overlong ULEB128 {context}")

    def read_sleb128(self, *, max_bits: int, context: str = "s33") -> int:
        result = 0
        shift = 0
        max_bytes = (max_bits + 6) // 7
        for idx in range(max_bytes):
            byte = self.read_u8(context=f"SLEB128 {context}")
            if idx == max_bytes - 1 and (byte & 0x80):
                raise WasmInspectError(f"overlong SLEB128 {context}")
            result |= (byte & 0x7F) << shift
            shift += 7
            if (byte & 0x80) == 0:
                if shift < max_bits and (byte & 0x40):
                    result |= -(1 << shift)
                elif byte & 0x40:
                    result -= 1 << shift
                return result
        raise WasmInspectError(f"overlong SLEB128 {context}")

    def read_u32(self, *, context: str = "u32") -> int:
        return self.read_uleb128(max_bits=32, context=context)

    def read_u64(self, *, context: str = "u64") -> int:
        return self.read_uleb128(max_bits=64, context=context)

    def read_name(self, *, context: str = "name") -> str:
        length = self.read_u32(context=f"{context} length")
        raw = self.read_bytes(length, context=context)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WasmInspectError(f"invalid UTF-8 in {context}: {exc}") from exc

    def read_valtype_code(self, *, context: str = "valtype") -> int:
        code = self.read_u8(context=context)
        if code in (0x63, 0x64):
            self.read_sleb128(max_bits=33, context=f"{context} heaptype")
        return code

    def read_limits(self, *, context: str = "limits") -> tuple[int, int, int | None]:
        flags = self.read_u8(context=f"{context} flags")
        has_max = bool(flags & WASM_LIMITS_FLAG_HAS_MAX)
        is_64 = bool(flags & WASM_LIMITS_FLAG_MEMORY64)
        if is_64:
            min_val = self.read_u64(context=f"{context} min64")
            max_val = self.read_u64(context=f"{context} max64") if has_max else None
        else:
            min_val = self.read_u32(context=f"{context} min32")
            max_val = self.read_u32(context=f"{context} max32") if has_max else None
        return flags, min_val, max_val

    def skip_const_expr(self, *, context: str = "const expr") -> None:
        """Advance past a Wasm constant expression ending with `0x0B` (end)."""
        while self.remaining > 0:
            op = self.read_u8(context=f"{context} opcode")
            if op == 0x0B:
                return
            if op == 0x41:  # i32.const
                self.read_sleb128(max_bits=32, context="i32.const")
            elif op == 0x42:  # i64.const
                self.read_sleb128(max_bits=64, context="i64.const")
            elif op == 0x43:  # f32.const
                self.skip(4, context="f32.const")
            elif op == 0x44:  # f64.const
                self.skip(8, context="f64.const")
            elif op == 0x23:  # global.get
                self.read_u32(context="global.get")
            elif op == 0xD0:  # ref.null
                self.read_sleb128(max_bits=33, context="ref.null")
            elif op == 0xD2:  # ref.func
                self.read_u32(context="ref.func")
            elif op == 0xFD:  # SIMD v128.const
                sub = self.read_u32(context="simd opcode")
                if sub == 12:
                    self.skip(16, context="v128.const")
            elif op in (0x6A, 0x6B, 0x6C, 0x7C, 0x7D, 0x7E):
                # Extended const expressions (i32/i64 add/sub/mul): no immediates.
                continue
            else:
                raise WasmInspectError(f"unsupported opcode 0x{op:02x} in {context}")
        raise WasmInspectError(f"unterminated {context}")


def _parse_dylink0_mem_info(cursor: _WasmStreamCursor) -> tuple[int, int] | None:
    """Parse `dylink.0` custom section subsections and return `(memory_size, table_size)`."""
    while cursor.remaining > 0:
        sub_type = cursor.read_u8(context="dylink.0 subsection type")
        sub_size = cursor.read_u32(context="dylink.0 subsection size")
        sub_start = cursor.pos
        sub_end = sub_start + sub_size
        if sub_end > cursor._end:
            raise WasmInspectError("truncated dylink.0 subsection")
        if sub_type == WASM_DYLINK_MEM_INFO:
            sub_cursor = _WasmStreamCursor(cursor._fh, sub_start, sub_end)
            mem_size = sub_cursor.read_u32(context="dylink.0 memory_size")
            _mem_align = sub_cursor.read_u32(context="dylink.0 memory_alignment")
            table_size = sub_cursor.read_u32(context="dylink.0 table_size")
            _table_align = sub_cursor.read_u32(context="dylink.0 table_alignment")
            return mem_size, table_size
        cursor.skip(sub_size, context="dylink.0 subsection")
    return None


def _inspect_wasm_stream(
    fh: BinaryIO,
    total_bytes: int,
    *,
    name: str,
    role: str,
) -> WasmModuleMetrics:
    if total_bytes < 8:
        raise WasmInspectError(
            f"{name}: Wasm binary too short ({total_bytes} bytes, need at least 8)"
        )
    cursor = _WasmStreamCursor(fh, 0, total_bytes)
    magic = cursor.read_bytes(4, context="Wasm magic")
    if magic != WASM_MAGIC:
        raise WasmInspectError(f"{name}: invalid Wasm magic {magic!r}")
    (version,) = struct.unpack("<I", cursor.read_bytes(4, context="Wasm version"))
    if version != WASM_VERSION:
        raise WasmInspectError(f"{name}: unsupported Wasm version {version}")

    func_imports = 0
    got_func_imports = 0
    got_mem_imports = 0
    func_exports = 0
    defined_table_size = 0
    imported_table_size = 0
    dylink_table_size: int | None = None
    dylink_memory_size: int | None = None
    has_dylink = False
    shard_dispatch_exports = 0
    shard_table_exports = 0
    sub_func_exports = 0
    sub_func_imports = 0
    got_func_sub_imports = 0
    got_func_shard_dispatch_imports = 0
    code_func_count = 0
    max_func_body_bytes = 0
    max_func_locals = 0
    large_func_bodies_over_1mib = 0

    while cursor.remaining > 0:
        sec_id = cursor.read_u8(context="section id")
        if sec_id not in WASM_KNOWN_SECTION_IDS:
            raise WasmInspectError(f"{name}: unknown Wasm section id {sec_id}")
        sec_size = cursor.read_u32(context=f"section {sec_id} size")
        sec_start = cursor.pos
        sec_end = sec_start + sec_size
        if sec_end > total_bytes:
            raise WasmInspectError(
                f"{name}: truncated section {sec_id} (size {sec_size} exceeds EOF)"
            )

        if sec_id == WASM_SECTION_CUSTOM:
            sec_cur = _WasmStreamCursor(fh, sec_start, sec_end)
            custom_name = sec_cur.read_name(context="custom section name")
            if custom_name == "dylink.0":
                has_dylink = True
                mem_info = _parse_dylink0_mem_info(sec_cur)
                if mem_info is not None:
                    dylink_memory_size, dylink_table_size = mem_info
            elif custom_name == "dylink" and not has_dylink:
                has_dylink = True
                dylink_memory_size = sec_cur.read_u32(context="dylink memory_size")
                _mem_align = sec_cur.read_u32(context="dylink memory_alignment")
                dylink_table_size = sec_cur.read_u32(context="dylink table_size")
                _table_align = sec_cur.read_u32(context="dylink table_alignment")
        elif sec_id == WASM_SECTION_IMPORT:
            sec_cur = _WasmStreamCursor(fh, sec_start, sec_end)
            count = sec_cur.read_u32(context="import count")
            for idx in range(count):
                mod_name = sec_cur.read_name(context=f"import[{idx}] module")
                field_name = sec_cur.read_name(context=f"import[{idx}] field")
                kind = sec_cur.read_u8(context=f"import[{idx}] kind")
                if kind == 0x00:  # func
                    sec_cur.read_u32(context=f"import[{idx}] func type_idx")
                    func_imports += 1
                    if field_name.startswith("sub_"):
                        sub_func_imports += 1
                elif kind == 0x01:  # table
                    reftype = sec_cur.read_valtype_code(context=f"import[{idx}] table reftype")
                    _, min_sz, _ = sec_cur.read_limits(context=f"import[{idx}] table limits")
                    if reftype == WASM_REFTYPE_FUNCREF:
                        imported_table_size += min_sz
                elif kind == 0x02:  # memory
                    sec_cur.read_limits(context=f"import[{idx}] memory limits")
                elif kind == 0x03:  # global
                    sec_cur.read_valtype_code(context=f"import[{idx}] global valtype")
                    _mut = sec_cur.read_u8(context=f"import[{idx}] global mutability")
                    if mod_name == "GOT.func":
                        got_func_imports += 1
                        if field_name.startswith("sub_"):
                            got_func_sub_imports += 1
                        elif field_name.startswith(SHARD_DISPATCH_FN_PREFIX):
                            got_func_shard_dispatch_imports += 1
                    elif mod_name == "GOT.mem":
                        got_mem_imports += 1
                elif kind == 0x04:  # tag
                    _attr = sec_cur.read_u8(context=f"import[{idx}] tag attr")
                    sec_cur.read_u32(context=f"import[{idx}] tag type_idx")
                else:
                    raise WasmInspectError(f"{name}: unknown import kind 0x{kind:02x}")
        elif sec_id == WASM_SECTION_TABLE:
            sec_cur = _WasmStreamCursor(fh, sec_start, sec_end)
            count = sec_cur.read_u32(context="table count")
            for idx in range(count):
                first = sec_cur.read_u8(context=f"table[{idx}] prefix/reftype")
                has_init = False
                if first == WASM_TABLE_EXTENDED_PREFIX:
                    reserved = sec_cur.read_u8(context=f"table[{idx}] reserved")
                    if reserved != 0x00:
                        raise WasmInspectError(
                            f"{name}: invalid extended table reserved byte 0x{reserved:02x}"
                        )
                    reftype = sec_cur.read_valtype_code(context=f"table[{idx}] reftype")
                    has_init = True
                else:
                    reftype = first
                    if reftype in (0x63, 0x64):
                        sec_cur.read_sleb128(max_bits=33, context=f"table[{idx}] heaptype")
                _, min_sz, _ = sec_cur.read_limits(context=f"table[{idx}] limits")
                if has_init:
                    sec_cur.skip_const_expr(context=f"table[{idx}] init expr")
                if reftype == WASM_REFTYPE_FUNCREF:
                    defined_table_size += min_sz
        elif sec_id == WASM_SECTION_EXPORT:
            sec_cur = _WasmStreamCursor(fh, sec_start, sec_end)
            count = sec_cur.read_u32(context="export count")
            for idx in range(count):
                exp_name = sec_cur.read_name(context=f"export[{idx}] name")
                kind = sec_cur.read_u8(context=f"export[{idx}] kind")
                _exp_idx = sec_cur.read_u32(context=f"export[{idx}] index")
                if kind == 0x00:  # func
                    func_exports += 1
                    if exp_name.startswith(SHARD_DISPATCH_FN_PREFIX):
                        shard_dispatch_exports += 1
                    elif exp_name.startswith("sub_"):
                        sub_func_exports += 1
                elif kind == 0x03:  # global
                    if exp_name.startswith(SHARD_DISPATCH_TABLE_PREFIX):
                        shard_table_exports += 1
        elif sec_id == WASM_SECTION_CODE:
            sec_cur = _WasmStreamCursor(fh, sec_start, sec_end)
            if sec_cur.remaining > 0:
                func_count = sec_cur.read_u32(context="code function count")
                code_func_count += func_count
                for idx in range(func_count):
                    body_size = sec_cur.read_u32(context=f"code[{idx}] body size")
                    body_start = sec_cur.pos
                    body_end = body_start + body_size
                    if body_end > sec_end:
                        raise WasmInspectError(
                            f"{name}: truncated code[{idx}] body "
                            f"(size {body_size} exceeds code section)"
                        )
                    saved_end = sec_cur._end
                    sec_cur._end = body_end
                    try:
                        local_decl_count = sec_cur.read_u32(
                            context=f"code[{idx}] local decl count"
                        )
                        total_locals = 0
                        for loc_idx in range(local_decl_count):
                            local_n = sec_cur.read_u32(
                                context=f"code[{idx}] local[{loc_idx}] count"
                            )
                            sec_cur.read_valtype_code(
                                context=f"code[{idx}] local[{loc_idx}] valtype"
                            )
                            total_locals += local_n
                    finally:
                        sec_cur._end = saved_end
                    if body_size > max_func_body_bytes:
                        max_func_body_bytes = body_size
                    if total_locals > max_func_locals:
                        max_func_locals = total_locals
                    if body_size > LARGE_FUNC_BODY_THRESHOLD_BYTES:
                        large_func_bodies_over_1mib += 1
                    remaining_body = body_end - sec_cur._pos
                    if remaining_body > 0:
                        sec_cur.skip(remaining_body, context=f"code[{idx}] bytecode")

        # Advance cleanly to the end of the section (O(1) seek over data/debug sections).
        cursor._pos = sec_end
        fh.seek(sec_end)

    effective_table_size = (
        dylink_table_size
        if has_dylink and dylink_table_size is not None
        else (defined_table_size if defined_table_size > 0 else imported_table_size)
    )
    return WasmModuleMetrics(
        name=name,
        role=role,
        wasm_bytes=total_bytes,
        func_imports=func_imports,
        got_func_imports=got_func_imports,
        got_mem_imports=got_mem_imports,
        func_exports=func_exports,
        table_size=effective_table_size,
        defined_table_size=defined_table_size,
        dylink_table_size=dylink_table_size,
        dylink_memory_size=dylink_memory_size,
        has_dylink=has_dylink,
        shard_dispatch_exports=shard_dispatch_exports,
        shard_table_exports=shard_table_exports,
        sub_func_exports=sub_func_exports,
        sub_func_imports=sub_func_imports,
        got_func_sub_imports=got_func_sub_imports,
        got_func_shard_dispatch_imports=got_func_shard_dispatch_imports,
        code_func_count=code_func_count,
        max_func_body_bytes=max_func_body_bytes,
        max_func_locals=max_func_locals,
        large_func_bodies_over_1mib=large_func_bodies_over_1mib,
    )


def inspect_wasm_bytes(
    data: bytes | bytearray | memoryview,
    *,
    name: str = RUNTIME_WASM,
    role: str = "main",
) -> WasmModuleMetrics:
    """Parse `import`, `export`, `table`, `code` headers, and `dylink.0` sections from Wasm bytes."""
    raw = bytes(data)
    return _inspect_wasm_stream(io.BytesIO(raw), len(raw), name=name, role=role)


def inspect_wasm_file(
    wasm_path: Path,
    *,
    name: str | None = None,
    role: str = "main",
) -> WasmModuleMetrics:
    """Stream-inspect a built `.wasm` file without loading function bytecode or `data` into RAM."""
    label = name or wasm_path.name
    try:
        total_bytes = wasm_path.stat().st_size
        with wasm_path.open("rb") as fh:
            return _inspect_wasm_stream(fh, total_bytes, name=label, role=role)
    except OSError as exc:
        raise WasmInspectError(f"{label}: unable to read Wasm binary: {exc}") from exc


def collect_wasm_metrics(
    *,
    main_wasm: Path,
    side_wasms: Sequence[tuple[str, Path]] = (),
    indirect_dispatch: str = DEFAULT_INDIRECT_DISPATCH,
    warnings: list[str] | None = None,
    stamp_cache_path: Path | None = None,
) -> WasmBuildMetrics:
    """Inspect `runtime.wasm` and every `guest_part_*.wasm` side module after linking.

    With `stamp_cache_path`, a module whose (path, size, mtime_ns) is unchanged since the last
    inspection reuses the stored metrics instead of re-parsing hundreds of MiB.
    """
    stamps: dict[str, Any] = {}
    if stamp_cache_path is not None:
        try:
            loaded = json.loads(stamp_cache_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                stamps = loaded
        except (OSError, ValueError):
            stamps = {}
    stamps_dirty = False
    modules: list[WasmModuleMetrics] = []
    targets: list[tuple[str, str, Path]] = [(RUNTIME_WASM, "main", main_wasm)]
    for side_name, side_path in side_wasms:
        targets.append((side_name, "side", side_path))

    for mod_name, role, path in targets:
        stamp_id = f"{path}|{mod_name}|{role}"
        try:
            st = path.stat()
            stamp = [st.st_size, st.st_mtime_ns]
        except OSError:
            stamp = None
        cached_entry = stamps.get(stamp_id)
        if stamp is not None and isinstance(cached_entry, dict) and cached_entry.get("stamp") == stamp:
            try:
                modules.append(WasmModuleMetrics(**cached_entry["metrics"]))
                continue
            except (TypeError, KeyError):
                pass
        try:
            inspected = inspect_wasm_file(path, name=mod_name, role=role)
            modules.append(inspected)
            if stamp is not None and stamp_cache_path is not None:
                stamps[stamp_id] = {"stamp": stamp, "metrics": dataclasses.asdict(inspected)}
                stamps_dirty = True
        except WasmInspectError as exc:
            if warnings is not None:
                warnings.append(f"wasm telemetry skipped for {mod_name}: {sanitize_text(str(exc))}")
            modules.append(
                WasmModuleMetrics(
                    name=mod_name,
                    role=role,
                    wasm_bytes=_object_bytes(path),
                )
            )
    if stamps_dirty and stamp_cache_path is not None:
        try:
            stamp_cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = stamp_cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(stamps), encoding="utf-8")
            os.replace(tmp, stamp_cache_path)
        except OSError:
            pass
    return WasmBuildMetrics(indirect_dispatch=indirect_dispatch, modules=tuple(modules))


def verify_table_import_limits(
    metrics: WasmBuildMetrics,
    max_table_imports: int,
) -> list[str]:
    """Verify total `GOT.func` imports and table footprint against `--verify-max-table-imports`."""
    limit = parse_verify_max_table_imports(max_table_imports)
    assert limit is not None
    errors: list[str] = []
    if metrics.total_got_func_imports > limit:
        errors.append(
            f"total GOT.func imports ({metrics.total_got_func_imports}) exceeds "
            f"--verify-max-table-imports limit ({limit})"
        )
    if metrics.side_module_count > 0:
        if metrics.side_table_size > limit:
            errors.append(
                f"total side-module dylink.0 tableSize ({metrics.side_table_size}) exceeds "
                f"--verify-max-table-imports limit ({limit})"
            )
    elif metrics.total_table_size > limit:
        errors.append(
            f"total wasmTable size ({metrics.total_table_size}) exceeds "
            f"--verify-max-table-imports limit ({limit})"
        )
    return errors


def check_v8_headroom_warnings(
    metrics: WasmBuildMetrics,
    *,
    include_module_exceeded: bool = True,
) -> list[str]:
    """Return sanitized warnings when any module approaches or exceeds V8 Wasm64 limits.

    Checks per-module:
      - Module size against `V8_MAX_WASM_MODULE_BYTES` (1024 MiB; warns >= `V8_MODULE_WARN_FRACTION`).
        Pass `include_module_exceeded=False` when `check_v8_wasm_size()` already reported >1 GiB.
      - Largest function body size against `V8_MAX_WASM_FUNCTION_BYTES` (`7,654,321` B; warns >= 75%).
      - Max function locals against `V8_MAX_WASM_FUNCTION_LOCALS` (`50,000`; warns >= 80%).
    """
    module_warn_bytes = int(V8_MAX_WASM_MODULE_BYTES * V8_MODULE_WARN_FRACTION)
    func_body_warn_bytes = int(V8_MAX_WASM_FUNCTION_BYTES * V8_FUNC_BODY_WARN_FRACTION)
    func_locals_warn_count = int(V8_MAX_WASM_FUNCTION_LOCALS * V8_FUNC_LOCALS_WARN_FRACTION)
    warnings: list[str] = []

    for mod in metrics.modules:
        if mod.exceeds_v8_module_limit:
            if include_module_exceeded:
                over_mib = (mod.wasm_bytes - V8_MAX_WASM_MODULE_BYTES) / (1024 * 1024)
                warnings.append(
                    sanitize_text(
                        f"{mod.name} size is {mod.wasm_bytes} bytes ({mod.size_mib:.2f} MiB, "
                        f"{mod.v8_module_used_pct:.1f}% of V8 1024 MiB limit), exceeding "
                        f"Chrome/V8's {V8_MAX_WASM_MODULE_BYTES} byte module cap by {over_mib:.2f} MiB"
                    )
                )
        elif mod.wasm_bytes >= module_warn_bytes:
            warnings.append(
                sanitize_text(
                    f"{mod.name} size is {mod.wasm_bytes} bytes ({mod.size_mib:.2f} MiB, "
                    f"{mod.v8_module_used_pct:.1f}% of V8 1024 MiB limit), approaching "
                    f"Chrome/V8's 1024 MiB module cap ({mod.v8_module_headroom_mib:.2f} MiB headroom remaining)"
                )
            )

        if mod.exceeds_v8_func_size_limit:
            over_bytes = mod.max_func_body_bytes - V8_MAX_WASM_FUNCTION_BYTES
            warnings.append(
                sanitize_text(
                    f"{mod.name} largest function body is {mod.max_func_body_bytes:,} bytes "
                    f"({mod.max_func_body_bytes / 1_000_000:.2f} MB), exceeding V8's "
                    f"kV8MaxWasmFunctionSize ({V8_MAX_WASM_FUNCTION_BYTES:,} bytes) by "
                    f"{over_bytes:,} bytes"
                )
            )
        elif mod.max_func_body_bytes >= func_body_warn_bytes:
            pct = (mod.max_func_body_bytes / V8_MAX_WASM_FUNCTION_BYTES) * 100.0
            warnings.append(
                sanitize_text(
                    f"{mod.name} largest function body is {mod.max_func_body_bytes:,} bytes "
                    f"({mod.max_func_body_bytes / 1_000_000:.2f} MB, {pct:.1f}% of V8's "
                    f"{V8_MAX_WASM_FUNCTION_BYTES:,} byte kV8MaxWasmFunctionSize; "
                    f"{mod.v8_func_body_headroom_bytes:,} bytes headroom remaining)"
                )
            )

        if mod.exceeds_v8_locals_limit:
            over_locals = mod.max_func_locals - V8_MAX_WASM_FUNCTION_LOCALS
            warnings.append(
                sanitize_text(
                    f"{mod.name} max function locals is {mod.max_func_locals:,}, exceeding V8's "
                    f"kV8MaxWasmFunctionLocals ({V8_MAX_WASM_FUNCTION_LOCALS:,}) by "
                    f"{over_locals:,}"
                )
            )
        elif mod.max_func_locals >= func_locals_warn_count:
            pct = (mod.max_func_locals / V8_MAX_WASM_FUNCTION_LOCALS) * 100.0
            warnings.append(
                sanitize_text(
                    f"{mod.name} max function locals is {mod.max_func_locals:,} "
                    f"({pct:.1f}% of V8's {V8_MAX_WASM_FUNCTION_LOCALS:,} "
                    f"kV8MaxWasmFunctionLocals; {mod.v8_func_locals_headroom:,} headroom remaining)"
                )
            )

    return warnings


def cleanup_stale_side_modules(
    output_dir: Path,
    active_side_names: Sequence[str],
    *,
    dry_run: bool = False,
) -> list[str]:
    """Remove (or report in dry-run) stale `guest_part_<N>.wasm` files not in `active_side_names`.

    Scans top-level entries in `output_dir` matching `_SIDE_MODULE_FILENAME_RE` (only files
    or symlinks, never directories, never `runtime.wasm` or unrelated `.wasm` files).
    Returns stale filenames sorted naturally by part index.
    """
    if not output_dir.is_dir():
        return []
    active = set(active_side_names)
    stale: list[tuple[int, str, Path]] = []
    try:
        entries = list(output_dir.iterdir())
    except OSError:
        return []
    for entry in entries:
        match = _SIDE_MODULE_FILENAME_RE.match(entry.name)
        if match is None or entry.name in active:
            continue
        is_link = entry.is_symlink()
        if not is_link and not entry.is_file():
            continue
        if entry.is_dir() and not is_link:
            continue
        stale.append((int(match.group(1)), entry.name, entry))
    stale.sort(key=lambda item: (item[0], item[1]))
    removed: list[str] = []
    for _, name, path in stale:
        if not dry_run:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue
        removed.append(name)
    return removed


def write_guest_manifest(
    output_dir: Path,
    *,
    indirect_dispatch: str,
    split_modules: int,
    split_strategy: str,
    side_modules: Sequence[SideModuleResult],
    wasm_metrics: WasmBuildMetrics | None,
    max_side_module_mib: int | None = None,
    side_module_mode: int | None = None,
    guest_dll_modules: Sequence[str] = (),
) -> Path:
    """Write or update `guest_manifest.json` with backwards-compatible fields + `wasmMetrics`."""
    manifest_path = output_dir / GUEST_MANIFEST_FILE
    payload: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                payload.update(existing)
        except (OSError, json.JSONDecodeError):
            payload = {}

    sanitized_dlls: list[str] = []
    seen_dlls: set[str] = set()
    for raw_mod in guest_dll_modules:
        clean_mod = _sanitize_guest_module_name(raw_mod)
        if clean_mod and clean_mod not in seen_dlls:
            seen_dlls.add(clean_mod)
            sanitized_dlls.append(clean_mod)

    payload["schemaVersion"] = SCHEMA_VERSION
    payload["runtimeJs"] = RUNTIME_JS
    payload["runtimeWasm"] = RUNTIME_WASM
    payload["indirectDispatch"] = indirect_dispatch
    payload["splitModules"] = split_modules
    payload["splitStrategy"] = split_strategy
    if max_side_module_mib is not None:
        payload["maxSideModuleMiB"] = max_side_module_mib
    if side_module_mode is not None:
        payload["sideModuleMode"] = side_module_mode
    payload["guest_dll_modules"] = sanitized_dlls
    # Keep the legacy name list for loaders; also emit sized telemetry for each part.
    payload["sideModules"] = [side.name for side in side_modules]
    payload["sideModuleSizes"] = [side.to_manifest_dict() for side in side_modules]
    if wasm_metrics is not None:
        payload["wasmMetrics"] = wasm_metrics.to_manifest_dict()

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def format_wasm_metrics_lines(metrics: WasmBuildMetrics) -> list[str]:
    """Render a crisp aligned Wasm module telemetry table and shard-dispatch banner."""
    lines: list[str] = ["  Wasm module telemetry:"]
    name_width = max(
        len("Module"),
        len("TOTAL"),
        *(len(m.name) for m in metrics.modules),
    )
    header = (
        f"    {'Module':<{name_width}}  {'Size (MiB)':>10}  {'Func Imp':>8}  "
        f"{'GOT.func':>8}  {'Func Exp':>8}  {'TableSize':>9}"
    )
    lines.append(header)
    for mod in metrics.modules:
        mib = mod.wasm_bytes / (1024 * 1024)
        lines.append(
            f"    {mod.name:<{name_width}}  {mib:>10.2f}  {mod.func_imports:>8}  "
            f"{mod.got_func_imports:>8}  {mod.func_exports:>8}  {mod.table_size:>9}"
        )
    if len(metrics.modules) > 1:
        total_mib = metrics.total_wasm_bytes / (1024 * 1024)
        lines.append(
            f"    {'TOTAL':<{name_width}}  {total_mib:>10.2f}  {metrics.total_func_imports:>8}  "
            f"{metrics.total_got_func_imports:>8}  {metrics.total_func_exports:>8}  "
            f"{metrics.total_table_size:>9}"
        )
    lines.append(
        f"  Table / GOT.func footprint: wasmTable={metrics.total_table_size} "
        f"(main={metrics.main_table_size}, side={metrics.side_table_size}), "
        f"GOT.func={metrics.total_got_func_imports}"
    )
    max_mod_label = metrics.max_module_name or "n/a"
    headroom_status = "OK" if metrics.within_all_limits else "OVER LIMIT"
    max_body_mb = metrics.max_func_body_bytes / 1_000_000
    v8_body_mb = V8_MAX_WASM_FUNCTION_BYTES / 1_000_000
    lines.append(
        f"  V8 headroom [{headroom_status}]: "
        f"max module {max_mod_label}={metrics.max_module_mib:.2f} / 1024.00 MiB "
        f"(headroom {metrics.min_module_headroom_mib:.2f} MiB), "
        f"largest func={metrics.max_func_body_bytes:,} B ({max_body_mb:.2f} / {v8_body_mb:.2f} MB), "
        f"max locals={metrics.max_func_locals:,} / {V8_MAX_WASM_FUNCTION_LOCALS:,}, "
        f"bodies >1 MiB={metrics.large_func_bodies_over_1mib} "
        f"(code funcs={metrics.total_code_func_count:,}), "
        f"GOT.func={metrics.total_got_func_imports}, wasmTable={metrics.total_table_size}"
    )
    if metrics.indirect_dispatch == "shard":
        effective = (
            max(metrics.total_got_func_imports, metrics.side_table_size)
            if metrics.side_module_count > 0
            else metrics.total_table_size
        )
        if 0 < effective < LEGACY_TABLE_BASELINE_ENTRIES:
            ratio = LEGACY_TABLE_BASELINE_ENTRIES / effective
            reduction_note = (
                f"vs {LEGACY_TABLE_BASELINE_ENTRIES:,} legacy per-function entries; "
                f"~{ratio:,.1f}x reduction"
            )
        elif effective == 0:
            reduction_note = (
                f"0 guest table/GOT.func entries vs {LEGACY_TABLE_BASELINE_ENTRIES:,} legacy"
            )
        else:
            reduction_note = f"vs {LEGACY_TABLE_BASELINE_ENTRIES:,} legacy baseline"
        lines.append(
            f"  [shard-dispatch] Active: GOT.func={metrics.total_got_func_imports}, "
            f"side tableSize={metrics.side_table_size}, "
            f"dispatchers={metrics.shard_dispatch_exports} ({reduction_note})"
        )
    return lines


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
    runtime_headers_fingerprint: str = "",
    source_root: Path | None = None,
    source_root_headers_fp: bytes = b"",
) -> UnitBuildResult:
    started = time.monotonic()
    content = unit.path.read_bytes()
    extra_fp = _extra_fingerprint_for_unit(
        unit,
        content=content,
        source_root=source_root,
        source_root_headers_fp=source_root_headers_fp,
    )
    key = cache_key_for(
        content,
        flags,
        kind=unit.kind,
        extra_fingerprint=extra_fp,
        runtime_headers_fingerprint=runtime_headers_fingerprint,
    )
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


def _resolved_path_key(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _is_companion_static_lib(path: Path) -> bool:
    """Return True for companion third-party static archives (e.g. `libopus.a`).

    Companion archives do not bundle playgta6 boot objects (`wasm_boot.cpp.o`,
    `main.cpp.o`), so split-module builds must preserve them directly without
    `emar d` filtering or `--whole-archive` wrapping.
    """
    return path.name.lower() == "libopus.a"


def _find_sibling_opus_lib(runtime_lib: Path) -> Path | None:
    """Locate CMake FetchContent `libopus.a` beside a built `libplaygta6_core.a`."""
    parent = runtime_lib.parent
    for candidate in (
        parent / "_deps" / "opus-build" / "libopus.a",
        parent / "libopus.a",
    ):
        if candidate.is_file():
            return candidate
    return None


def resolve_runtime_libs(
    *,
    repo_root: Path,
    runtime_lib: Path | None,
    extra_objects: Sequence[Path],
) -> list[Path]:
    libs: list[Path] = []
    seen: set[Path] = set()

    def _append_unique(path: Path) -> None:
        key = _resolved_path_key(path)
        if key not in seen:
            seen.add(key)
            libs.append(path)

    primary: Path | None = None
    if runtime_lib is not None:
        primary = runtime_lib
    else:
        default_lib = repo_root / DEFAULT_RUNTIME_LIB_REL
        if default_lib.is_file():
            primary = default_lib
        else:
            alt = repo_root / "runtime" / "libplaygta6_core.a"
            if alt.is_file():
                primary = alt

    if primary is not None:
        _append_unique(primary)
        if primary.is_file():
            sibling_opus = _find_sibling_opus_lib(primary)
            if sibling_opus is not None:
                _append_unique(sibling_opus)

    for extra in extra_objects:
        _append_unique(extra)
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


LINK_CACHE_DIR_NAME = "linked"
LINK_CACHE_KEEP_PER_LABEL = 2
DEFAULT_LINK_JOBS = 2
WASM_METRICS_STAMP_FILE_NAME = "wasm-metrics-v1.json"


def _file_sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def link_cache_key(
    label: str,
    *,
    linker: Path | None,
    flags: Sequence[str],
    objects: Sequence[Path],
    content_hashed_files: Sequence[Path] = (),
    extra_text: str = "",
) -> str:
    """Content key for one link step.

    Compiled objects are named `<stem>-<key16>.o` after their own content key, so name + size
    identify them without re-reading hundreds of MiB; archives and loose inputs are hashed.
    The toolchain is identified by the linker's size/mtime plus its `emscripten-version.txt`.
    """
    digest = hashlib.sha256()
    digest.update(f"playgta6-link-cache-v1:{label}".encode("utf-8"))
    digest.update(b"\0")
    if linker is not None:
        try:
            st = linker.stat()
            digest.update(f"{linker.name}:{st.st_size}:{st.st_mtime_ns}".encode("utf-8"))
        except OSError:
            digest.update(b"linker-missing")
        for version_file in (
            linker.parent / "emscripten-version.txt",
            linker.parent.parent / "emscripten" / "emscripten-version.txt",
        ):
            try:
                digest.update(version_file.read_bytes())
                break
            except OSError:
                continue
    digest.update(b"\0")
    for flag in flags:
        digest.update(flag.encode("utf-8"))
        digest.update(b"\0")
    digest.update(b"objects\0")
    for obj in objects:
        try:
            size = obj.stat().st_size
        except OSError:
            size = -1
        digest.update(f"{obj.name}:{size}".encode("utf-8"))
        digest.update(b"\0")
    digest.update(b"files\0")
    for path in content_hashed_files:
        try:
            digest.update(f"{path.name}:{_file_sha256_hex(path)}".encode("utf-8"))
        except OSError:
            digest.update(f"{path.name}:missing".encode("utf-8"))
        digest.update(b"\0")
    digest.update(extra_text.encode("utf-8"))
    return digest.hexdigest()


def _link_cache_entry(cache_root: Path, label: str, key: str, suffix: str) -> Path:
    return cache_root / LINK_CACHE_DIR_NAME / f"{label}-{key[:32]}{suffix}"


def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def restore_cached_link_output(cached: Path, dest: Path, *, hardlink: bool) -> bool:
    """Make `dest` identical to `cached` (hard link when allowed, else copy). False on miss."""
    if not cached.is_file():
        return False
    if dest.is_file() and hardlink and _same_file(cached, dest):
        return True
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest.unlink()
        if hardlink:
            try:
                os.link(cached, dest)
                return True
            except OSError:
                pass
        shutil.copyfile(cached, dest)
        return True
    except OSError:
        return False


def store_link_output(cached: Path, produced: Path, *, hardlink: bool) -> None:
    """Publish a freshly linked output into the link cache (best effort)."""
    try:
        cached.parent.mkdir(parents=True, exist_ok=True)
        if cached.exists():
            cached.unlink()
        if hardlink:
            try:
                os.link(produced, cached)
                return
            except OSError:
                pass
        shutil.copyfile(produced, cached)
    except OSError:
        pass


def prune_link_cache(cache_root: Path, label: str, *, keep: int = LINK_CACHE_KEEP_PER_LABEL) -> None:
    """Keep only the newest `keep` cached outputs for `label` (parts are hundreds of MiB each)."""
    directory = cache_root / LINK_CACHE_DIR_NAME
    try:
        entries = [
            p for p in directory.iterdir() if p.is_file() and p.name.startswith(f"{label}-")
        ]
        entries.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
        for old in entries[keep:]:
            old.unlink()
    except OSError:
        pass


def run_build(
    *,
    source_dir: Path | None,
    repo_root: Path = DEFAULT_REPO_ROOT,
    cache_dir: Path | None = None,
    out_dir: Path | None = None,
    jobs: int | None = None,
    opt: str = "O2",
    opt_modules: Sequence[str] | Mapping[str, str] | None = None,
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
    no_link_opt: bool = False,
    shrink_largest_shards: int = 0,
    wasm_exceptions: bool = False,
    initial_memory: str | None = None,
    stack_size: str | None = None,
    assertions: int | None = None,
    browser_boot_exports: bool = False,
    split_modules: int = 1,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
    max_side_module_mib: int = DEFAULT_MAX_SIDE_MODULE_MIB,
    collapse_single_side_module: bool = False,
    node_max_old_space_mb: int | None = None,
    proxy_to_pthread: bool = False,
    max_concurrent_pthread_loads: int = DEFAULT_MAX_CONCURRENT_PTHREAD_LOADS,
    verify_max_table_imports: int | None = None,
    link_jobs: int = DEFAULT_LINK_JOBS,
    link_cache: bool = True,
) -> BuildReport:
    """Discover, compile (in parallel), and link guest shards into runtime.{js,wasm}."""
    opt_level = parse_opt_level(opt)
    opt_modules_map = parse_opt_modules(opt_modules)
    worker_count = jobs if jobs is not None else (os.cpu_count() or 1)
    if worker_count < 1:
        raise BuildGuestWasmError("--jobs must be >= 1")
    if sample < 0:
        raise BuildGuestWasmError("--sample must be >= 0")
    if isinstance(shrink_largest_shards, bool) or shrink_largest_shards < 0:
        raise BuildGuestWasmError("--shrink-largest-shards must be >= 0")
    split_count = parse_split_modules(split_modules)
    split_strat = parse_split_strategy(split_strategy)
    max_side_mib = parse_max_side_module_mib(max_side_module_mib)
    side_mode = side_module_mode_for_strategy(split_strat) if split_count > 1 else 1
    use_split = split_count > 1
    max_pthread_loads = parse_max_concurrent_pthread_loads(max_concurrent_pthread_loads)
    max_table_imports_limit = parse_verify_max_table_imports(verify_max_table_imports)
    explicit_node_mb = parse_node_max_old_space_mb(node_max_old_space_mb)
    base_link_env: Mapping[str, str] = os.environ if env is None else env
    link_env = build_link_env(
        base_link_env,
        split_modules=split_count,
        node_max_old_space_mb=explicit_node_mb,
    )
    effective_node_mb: int | None
    if explicit_node_mb is not None:
        effective_node_mb = explicit_node_mb
    elif use_split and not _has_node_max_old_space_size(base_link_env.get("NODE_OPTIONS", "")):
        effective_node_mb = DEFAULT_SPLIT_NODE_MAX_OLD_SPACE_MB
    else:
        effective_node_mb = None
    initial_memory_token = (
        parse_size_token(initial_memory, flag_name="--initial-memory")
        if initial_memory is not None
        else (SPLIT_DEFAULT_INITIAL_MEMORY if use_split else None)
    )
    stack_size_token = (
        parse_size_token(stack_size, flag_name="--stack-size")
        if stack_size is not None
        else None
    )
    assertions_val = parse_assertions(assertions)
    effective_link_flags = resolve_link_flags(
        opt=opt_level,
        strip=strip,
        pthread_pool_size=pthread_pool_size,
        extra_link_flags=extra_link_flags,
        no_link_opt=no_link_opt,
        wasm_exceptions=wasm_exceptions,
        initial_memory=initial_memory_token,
        stack_size=stack_size_token,
        assertions=assertions_val,
        browser_boot_exports=browser_boot_exports,
        main_module=use_split,
        proxy_to_pthread=proxy_to_pthread,
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
    base_compile_flags = (
        (*COMPILE_WASM_FLAGS, *SHARED_MEMORY_COMPILE_FLAGS, f"-{opt_level}", PIC_FLAG)
        if use_split
        else (*COMPILE_WASM_FLAGS, *SHARED_MEMORY_COMPILE_FLAGS, f"-{opt_level}")
    )
    report = BuildReport(
        ok=toolchain.ok,
        mode=mode,
        toolchain=toolchain,
        jobs=worker_count,
        opt=opt_level,
        opt_modules=dict(opt_modules_map),
        compile_flags=base_compile_flags,
        link_flags=effective_link_flags,
        linked_guest_boot=link_guest_boot,
        sample=sample,
        no_link_opt=no_link_opt,
        shrink_largest_shards=shrink_largest_shards,
        wasm_exceptions=wasm_exceptions,
        initial_memory=initial_memory_token,
        stack_size=stack_size_token,
        assertions=assertions_val,
        browser_boot_exports=browser_boot_exports,
        split_modules=split_count,
        split_strategy=split_strat,
        max_side_module_mib=max_side_mib,
        side_module_mode=side_mode,
        node_max_old_space_mb=effective_node_mb,
        proxy_to_pthread=proxy_to_pthread,
        max_concurrent_pthread_loads=max_pthread_loads,
        verify_max_table_imports=max_table_imports_limit,
    )

    if not toolchain.ok:
        report.errors.append(
            "emcc not found on PATH, EMSDK, or work/**/emsdk/upstream/emscripten"
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
    manifest_arg = source_dir if source_dir.is_file() else None
    all_units = discover_sources(source_dir)
    program_build = is_program_build(source_root, all_units)
    report.guest_dll_modules = discover_guest_dll_modules(
        source_root,
        manifest_path=manifest_arg,
    )
    dispatch_info = detect_indirect_dispatch(
        source_root,
        all_units,
        manifest_path=manifest_arg,
    )
    report.indirect_dispatch = dispatch_info.mode
    report.indirect_dispatch_source = dispatch_info.source
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
    shrink_ids: frozenset[str] = (
        select_shrink_shards(units, shrink_largest_shards)
        if shrink_largest_shards > 0 and opt_level != "Oz"
        else frozenset()
    )
    runtime_headers_fp = compute_runtime_headers_fingerprint(repo_root)
    source_root_hdr_fp = source_root_headers_fingerprint(source_root)
    key_stamps = KeyStampCache(cache_root / KEY_STAMP_FILE_NAME)
    plan_started = time.monotonic()
    planned: list[UnitBuildResult] = []
    compile_tasks: list[tuple[SourceUnit, Path, tuple[str, ...]]] = []
    for unit in units:
        compiler = toolchain.compiler_for(unit.kind)
        if compiler is None:
            report.ok = False
            report.errors.append(f"no compiler for {unit.unit_id} ({unit.kind})")
            continue
        unit_opt = opt_level_for_unit(
            unit,
            default_opt=opt_level,
            opt_modules=opt_modules_map,
            shrink_ids=shrink_ids,
        )
        extra = boot_defines if link_guest_boot and unit.module == "boot" else ()
        flags = compile_flags_for(
            opt=unit_opt,
            kind=unit.kind,
            include_dirs=include_dirs,
            extra_flags=extra,
            wasm_exceptions=wasm_exceptions,
            pic=use_split,
        )
        if unit.kind == "ll":
            key = key_stamps.key_for_file(
                unit.path,
                flags,
                kind=unit.kind,
                runtime_headers_fingerprint=runtime_headers_fp,
            )
        else:
            content = unit.path.read_bytes()
            extra_fp = _extra_fingerprint_for_unit(
                unit,
                content=content,
                source_root=source_root,
                source_root_headers_fp=source_root_hdr_fp,
            )
            key = cache_key_for(
                content,
                flags,
                kind=unit.kind,
                extra_fingerprint=extra_fp,
                runtime_headers_fingerprint=runtime_headers_fp,
            )
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

    if not dry_run:
        key_stamps.save()
    report.stage_seconds["plan_cache_keys"] = round(time.monotonic() - plan_started, 3)
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
                runtime_headers_fingerprint=runtime_headers_fp,
                source_root=source_root,
                source_root_headers_fp=source_root_hdr_fp,
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
    report.stage_seconds["wasm_shard_compile"] = report.compile_seconds
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

    if use_split:
        capped_mib = (
            DEFAULT_CAPPED_SIDE_MODULE_MIB
            if split_strat == "capped" and max_side_mib == DEFAULT_MAX_SIDE_MODULE_MIB
            else max_side_mib
        )
        max_bucket_bytes = (
            capped_mib * 1024 * 1024 if split_strat in ("asymmetric", "capped") else None
        )
        buckets = partition_shard_units(
            ordered,
            split_count,
            strategy=split_strat,
            max_bucket_bytes=max_bucket_bytes,
            collapse_if_fits=collapse_single_side_module,
        )
        registration_items = [item for item in ordered if is_registration_unit(item.unit)]
        non_shard_items = [
            item
            for item in ordered
            if not is_side_shard_unit(item.unit) and not is_registration_unit(item.unit)
        ]
        all_side_buckets: list[tuple[str, list[UnitBuildResult]]] = [
            ("shards", bucket) for bucket in buckets
        ]
        if registration_items:
            # Place `functions.ll` registration tables in the final side module (`guest_part_N.wasm`)
            # for both legacy `table` mode and `shard` mode: all shard side modules load first via
            # `Module.dynamicLibraries` and register their `__playgta6_dispatch_shard_*` exports in
            # `GOT.func` before `functions.o` applies its data relocations and exports
            # `__playgta6_shard_dispatch_*` to `runtime.wasm`'s `pg6_program.o`.
            all_side_buckets.append(("registration", registration_items))
        use_pruned_side = side_mode == 2
        side_link_flags = resolve_side_module_link_flags(
            opt=opt_level,
            strip=strip,
            no_link_opt=no_link_opt,
            wasm_exceptions=wasm_exceptions,
            pruned=use_pruned_side,
        )
        side_plans: list[
            tuple[SideModuleResult, Path, list[Path], Path | None, Path | None, tuple[str, ...]]
        ] = []
        for idx, (bucket_role, bucket) in enumerate(all_side_buckets):
            side_name = SIDE_MODULE_NAME_TEMPLATE.format(index=idx)
            side_wasm_path = output_dir / side_name
            bucket_objects = [
                item.object_path
                if item.object_path is not None
                else Path(item.unit.unit_id).with_suffix(".o")
                for item in bucket
            ]
            estimated_bytes = sum(_partition_weight(item) for item in bucket)
            side_rsp_name = SIDE_RESPONSE_FILE_TEMPLATE.format(index=idx)
            side_rsp_path = None if response_file is False else (cache_root / side_rsp_name)
            side_exports_rsp_name = (
                SIDE_EXPORTS_RESPONSE_FILE_TEMPLATE.format(index=idx) if use_pruned_side else None
            )
            side_exports_rsp_path = (
                (cache_root / side_exports_rsp_name) if side_exports_rsp_name is not None else None
            )
            raw_side_link = build_side_module_link_argv(
                linker,
                bucket_objects,
                output_wasm=side_wasm_path,
                side_link_flags=side_link_flags,
                response_file=side_rsp_path,
                exports_response_file=side_exports_rsp_path,
                pruned=use_pruned_side,
            )
            side_result = SideModuleResult(
                index=idx,
                name=side_name,
                shards=tuple(item.unit.unit_id for item in bucket),
                response_file=side_rsp_name if side_rsp_path is not None else None,
                exports_response_file=side_exports_rsp_name,
                link_argv=tuple(sanitize_argv(raw_side_link)),
                role=bucket_role,
                estimated_bytes=estimated_bytes,
                side_module_mode=side_mode,
            )
            side_plans.append(
                (
                    side_result,
                    side_wasm_path,
                    bucket_objects,
                    side_rsp_path,
                    side_exports_rsp_path,
                    raw_side_link,
                )
            )

        archiver = toolchain.archiver()
        split_runtime_libs: list[Path] = []
        archive_filter_plans: list[tuple[Path, Path]] = []
        for lib in runtime_libs:
            if lib.suffix.lower() == ".a" and not _is_companion_static_lib(lib):
                filtered_lib = split_filtered_archive_path(cache_root, lib)
                split_runtime_libs.append(filtered_lib)
                archive_filter_plans.append((lib, filtered_lib))
            else:
                split_runtime_libs.append(lib)

        main_objects = [
            item.object_path
            if item.object_path is not None
            else Path(item.unit.unit_id).with_suffix(".o")
            for item in non_shard_items
        ]
        rsp_path: Path | None = None
        if should_use_response_file(len(main_objects), response_file):
            rsp_path = cache_root / RESPONSE_FILE_NAME
            report.response_file = RESPONSE_FILE_NAME
        raw_link = build_link_argv(
            linker,
            main_objects,
            output_js=output_js,
            runtime_libs=split_runtime_libs,
            link_flags=effective_link_flags,
            response_file=rsp_path,
            whole_archive_libs=True,
        )
        report.side_modules = [plan[0] for plan in side_plans]
        report.link_argv = tuple(sanitize_argv(raw_link))
        report.outputs = (RUNTIME_JS, RUNTIME_WASM, *(plan[0].name for plan in side_plans))

        if compile_errors:
            report.ok = False
            return report

        if dry_run:
            report.removed_stale_side_modules = cleanup_stale_side_modules(
                output_dir,
                [plan[0].name for plan in side_plans],
                dry_run=True,
            )
            report.ok = True
            return report

        cross_exports_by_part: list[list[str]] = []
        cross_exports_started = time.monotonic()
        if use_pruned_side and side_plans:
            llvm_nm_path = toolchain.nm()
            if llvm_nm_path is None:
                report.ok = False
                report.errors.append(
                    "llvm-nm not found alongside em++/emcc (upstream/bin) or on PATH; "
                    "required to compute cross-module exports for --split-strategy asymmetric "
                    "(-sSIDE_MODULE=2)"
                )
                return report
            nm_main_objects = list(main_objects) + [
                p for p in extra_objects if p.suffix.lower() == ".o" and p.is_file()
            ]
            manifest_required_exports = extract_program_required_exports(
                source_root,
                manifest_path=manifest_arg,
            )
            effective_required_names = (
                SIDE_MODULE_REQUIRED_EXPORT_NAMES | manifest_required_exports
            )
            try:
                cross_exports_by_part = compute_side_module_cross_exports(
                    [plan[2] for plan in side_plans],
                    nm_main_objects,
                    llvm_nm_path,
                    run_command=runner,
                    required_names=effective_required_names,
                )
            except BuildGuestWasmError as exc:
                report.ok = False
                report.errors.append(sanitize_text(str(exc)))
                return report

        report.stage_seconds["cross_exports_nm"] = round(time.monotonic() - cross_exports_started, 3)
        output_dir.mkdir(parents=True, exist_ok=True)
        side_link_started = time.monotonic()

        def _link_side_part(idx: int) -> str | None:
            """Link (or restore from the link cache) one part; returns an error string or None."""
            (
                side_result,
                side_wasm_path,
                bucket_objects,
                side_rsp_path,
                side_exports_rsp_path,
                raw_side_link,
            ) = side_plans[idx]
            part_exports: list[str] = []
            if side_exports_rsp_path is not None:
                part_exports = cross_exports_by_part[idx] if idx < len(cross_exports_by_part) else []
                side_result.exported_symbol_count = len(part_exports)
            label = Path(side_result.name).stem
            cache_key = link_cache_key(
                label,
                linker=linker,
                flags=side_link_flags,
                objects=bucket_objects,
                extra_text="\n".join(part_exports),
            )
            cached_wasm = _link_cache_entry(cache_root, label, cache_key, ".wasm")
            restored = link_cache and restore_cached_link_output(
                cached_wasm, side_wasm_path, hardlink=True
            )
            if restored:
                side_result.link_cache_hit = True
            else:
                if side_rsp_path is not None:
                    write_link_response_file(side_rsp_path, bucket_objects)
                if side_exports_rsp_path is not None:
                    write_side_exports_response_file(side_exports_rsp_path, part_exports)
                if side_wasm_path.exists():
                    # Never relink in place: the old file may be hard-linked into the link cache.
                    side_wasm_path.unlink()
                side_link_result = runner(raw_side_link)
                if side_link_result.returncode != 0:
                    err = sanitize_text(
                        (side_link_result.stderr or side_link_result.stdout or "side module link failed").strip()
                    )
                    return f"{side_result.name}: {err}"
                if not side_wasm_path.is_file():
                    return f"missing side module link output: {side_result.name}"
            side_result.wasm_bytes = _object_bytes(side_wasm_path)
            input_obj_bytes = sum(_object_bytes(obj_path) for obj_path in bucket_objects)
            empty_err = check_side_module_not_empty(
                side_result.wasm_bytes,
                input_object_bytes=input_obj_bytes,
                module_name=side_result.name,
                side_module_mode=side_result.side_module_mode,
            )
            if empty_err is not None:
                return empty_err
            if link_cache and not restored:
                store_link_output(cached_wasm, side_wasm_path, hardlink=True)
                prune_link_cache(cache_root, label)
            return None

        side_workers = max(1, min(link_jobs, len(side_plans)))
        if side_workers == 1:
            side_errors = [_link_side_part(i) for i in range(len(side_plans))]
        else:
            with ThreadPoolExecutor(max_workers=side_workers) as link_pool:
                side_errors = list(link_pool.map(_link_side_part, range(len(side_plans))))
        first_side_error = next((e for e in side_errors if e), None)
        if first_side_error is not None:
            report.ok = False
            report.errors.append(first_side_error)
            return report

        report.stage_seconds["side_module_links"] = round(time.monotonic() - side_link_started, 3)
        if archive_filter_plans:
            assert archiver is not None
            for orig_archive, filtered_archive in archive_filter_plans:
                _, _, ar_result = prepare_split_runtime_archive(
                    orig_archive,
                    cache_dir=cache_root,
                    archiver=archiver,
                    dry_run=False,
                    run_command=runner,
                )
                if ar_result.returncode != 0:
                    report.ok = False
                    err = sanitize_text(
                        (ar_result.stderr or ar_result.stdout or "emar d failed").strip()
                    )
                    report.errors.append(f"{filtered_archive.name}: {err}")
                    return report

        if rsp_path is not None:
            write_link_response_file(rsp_path, main_objects)
        runtime_link_started = time.monotonic()
        rt_key = link_cache_key(
            "runtime",
            linker=linker,
            flags=effective_link_flags,
            objects=main_objects,
            content_hashed_files=[p for p in runtime_libs if p.is_file()],
        )
        cached_rt_js = _link_cache_entry(cache_root, "runtime", rt_key, ".js")
        cached_rt_wasm = _link_cache_entry(cache_root, "runtime", rt_key, ".wasm")
        rt_restored = (
            link_cache
            and cached_rt_js.is_file()
            and cached_rt_wasm.is_file()
            and restore_cached_link_output(cached_rt_js, output_js, hardlink=False)
            and restore_cached_link_output(cached_rt_wasm, output_wasm, hardlink=False)
        )
        if rt_restored:
            link_result = CommandResult(returncode=0, stdout="", stderr="")
        else:
            link_result = _invoke_runner(runner, raw_link, env=link_env)
            if link_cache and link_result.returncode == 0:
                store_link_output(cached_rt_js, output_js, hardlink=False)
                store_link_output(cached_rt_wasm, output_wasm, hardlink=False)
                prune_link_cache(cache_root, "runtime", keep=4)
        report.stage_seconds["runtime_link"] = round(time.monotonic() - runtime_link_started, 3)
        if link_result.returncode != 0:
            report.ok = False
            report.errors.append(
                sanitize_text((link_result.stderr or link_result.stdout or "link failed").strip())
            )
            return report

        missing_outputs = [
            name for name, path in ((RUNTIME_JS, output_js), (RUNTIME_WASM, output_wasm)) if not path.is_file()
        ]
        if missing_outputs:
            report.ok = False
            report.errors.append("missing link outputs: " + ", ".join(missing_outputs))
            return report

        report.split_runtime_js_patches = patch_split_runtime_js_file(
            output_js,
            max_concurrent_pthread_loads=max_pthread_loads,
        )

        report.wasm_bytes = _object_bytes(output_wasm)
        runtime_exceeds, runtime_warning = check_v8_wasm_size(
            report.wasm_bytes,
            module_name=RUNTIME_WASM,
        )
        if runtime_warning is not None:
            report.warnings.append(runtime_warning)
        any_exceeds = runtime_exceeds
        for side_result in report.side_modules:
            side_exceeds, side_warning = check_v8_wasm_size(
                side_result.wasm_bytes,
                module_name=side_result.name,
            )
            side_result.exceeds_v8_1gib_limit = side_exceeds
            if (
                split_strat == "capped"
                and max_bucket_bytes is not None
                and side_result.wasm_bytes > max_bucket_bytes
            ):
                report.ok = False
                report.errors.append(
                    f"{side_result.name} is {side_result.wasm_bytes / (1024 * 1024):.1f} MiB, over the "
                    f"{max_bucket_bytes // (1024 * 1024)} MiB per-part cap (linked size exceeds the "
                    f"object estimate); lower --max-side-module-mib to split into more parts"
                )
            if side_warning is not None:
                report.warnings.append(side_warning)
            any_exceeds = any_exceeds or side_exceeds
        report.exceeds_v8_1gib_limit = any_exceeds
        report.removed_stale_side_modules = cleanup_stale_side_modules(
            output_dir,
            [side_result.name for side_result in report.side_modules],
            dry_run=False,
        )
    else:
        planned_objects = objects if objects else [Path(item.unit.unit_id).with_suffix(".o") for item in ordered]
        rsp_path = None
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
            report.removed_stale_side_modules = cleanup_stale_side_modules(
                output_dir,
                (),
                dry_run=True,
            )
            report.ok = True
            return report

        output_dir.mkdir(parents=True, exist_ok=True)
        link_result = _invoke_runner(runner, raw_link, env=link_env)
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
        report.removed_stale_side_modules = cleanup_stale_side_modules(
            output_dir,
            (),
            dry_run=False,
        )

    metrics_started = time.monotonic()
    side_wasm_pairs = [
        (side_result.name, output_dir / side_result.name) for side_result in report.side_modules
    ]
    report.wasm_metrics = collect_wasm_metrics(
        main_wasm=output_wasm,
        side_wasms=side_wasm_pairs,
        indirect_dispatch=report.indirect_dispatch,
        warnings=report.warnings,
        stamp_cache_path=cache_root / WASM_METRICS_STAMP_FILE_NAME if link_cache else None,
    )
    metrics_by_name = {m.name: m for m in report.wasm_metrics.modules}
    for side_result in report.side_modules:
        side_result.wasm_metrics = metrics_by_name.get(side_result.name)

    report.stage_seconds["wasm_inspect"] = round(time.monotonic() - metrics_started, 3)
    report.warnings.extend(
        check_v8_headroom_warnings(report.wasm_metrics, include_module_exceeded=False)
    )

    if report.indirect_dispatch == "shard" and report.wasm_metrics.got_func_sub_imports > 0:
        report.warnings.append(
            f"shard-dispatch build still has {report.wasm_metrics.got_func_sub_imports} "
            "GOT.func sub_* import(s); verify functions.ll and shard objects were all built "
            "with --indirect-dispatch shard"
        )

    write_guest_manifest(
        output_dir,
        indirect_dispatch=report.indirect_dispatch,
        split_modules=report.split_modules,
        split_strategy=report.split_strategy,
        side_modules=report.side_modules,
        wasm_metrics=report.wasm_metrics,
        max_side_module_mib=report.max_side_module_mib if report.split_modules > 1 else None,
        side_module_mode=report.side_module_mode if report.split_modules > 1 else None,
        guest_dll_modules=report.guest_dll_modules,
    )
    report.guest_manifest = GUEST_MANIFEST_FILE

    if max_table_imports_limit is not None:
        limit_errors = verify_table_import_limits(report.wasm_metrics, max_table_imports_limit)
        if limit_errors:
            report.ok = False
            report.errors.extend(limit_errors)
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
    if report.mode != "check-only":
        lines.append(
            f"  Indirect dispatch: {report.indirect_dispatch} "
            f"(detected from {report.indirect_dispatch_source})"
        )
    if report.guest_dll_modules:
        lines.append("  Guest DLL modules: " + ", ".join(report.guest_dll_modules))
    if report.verify_max_table_imports is not None:
        lines.append(f"  Max table imports limit: {report.verify_max_table_imports}")
    if report.linked_guest_boot:
        lines.append("  Linked guest boot: yes")
    if report.no_link_opt:
        lines.append("  No link opt: yes (skipping link-time -Os/-Oz wasm-opt)")
    if report.opt_modules:
        pairs = ", ".join(f"{mod}=-{lvl}" for mod, lvl in sorted(report.opt_modules.items()))
        lines.append(f"  Module opt overrides: {pairs}")
    if report.shrink_largest_shards:
        lines.append(f"  Shrink largest shards: {report.shrink_largest_shards} (compiled at -Oz)")
    if report.wasm_exceptions:
        lines.append("  Wasm exceptions: yes (-fwasm-exceptions on C++ units and link)")
    if report.initial_memory:
        lines.append(f"  Initial memory: {report.initial_memory}")
    if report.stack_size:
        lines.append(
            f"  Stack size: {report.stack_size} "
            f"(-sSTACK_SIZE={report.stack_size} -sDEFAULT_PTHREAD_STACK_SIZE={report.stack_size})"
        )
    if report.assertions is not None:
        lines.append(f"  Assertions: {report.assertions} (-sASSERTIONS={report.assertions})")
    if report.browser_boot_exports:
        lines.append("  Browser boot exports: yes")
    if report.split_modules > 1:
        side_flag = (
            SIDE_MODULE_PRUNED_FLAG if report.side_module_mode == 2 else SIDE_MODULE_FLAG
        )
        split_extra = (
            f", max_side={report.max_side_module_mib}MiB"
            if report.split_strategy in ("asymmetric", "capped")
            else ""
        )
        lines.append(
            f"  Split modules: {report.split_modules} "
            f"(strategy={report.split_strategy}{split_extra}, {PIC_FLAG}, "
            f"{MAIN_MODULE_FLAG}, {side_flag})"
        )
    if report.proxy_to_pthread:
        lines.append(f"  Proxy to pthread: yes ({PROXY_TO_PTHREAD_FLAG})")
    if report.split_runtime_js_patches is not None:
        applied = [
            k for k, v in report.split_runtime_js_patches.items() if v and k != "already_patched"
        ]
        if applied:
            lines.append(f"  Split runtime JS opt: {', '.join(applied)}")
    if report.node_max_old_space_mb is not None:
        lines.append(
            f"  Node max old space: {report.node_max_old_space_mb} MB "
            f"(NODE_OPTIONS=--max-old-space-size={report.node_max_old_space_mb})"
        )
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
    if report.side_modules:
        lines.append("  Side modules:")
        for side in report.side_modules:
            size_part = ""
            if side.wasm_bytes:
                smib = side.wasm_bytes / (1024 * 1024)
                slimit = " [EXCEEDS V8 1 GiB LIMIT]" if side.exceeds_v8_1gib_limit else ""
                size_part = f" — {side.wasm_bytes} bytes ({smib:.2f} MiB){slimit}"
            elif side.estimated_bytes:
                emib = side.estimated_bytes / (1024 * 1024)
                size_part = f" — est. {side.estimated_bytes} bytes ({emib:.2f} MiB)"
            exports_part = (
                f", {side.exported_symbol_count} cross-exports"
                if side.exported_symbol_count is not None
                else ""
            )
            lines.append(
                f"    {side.name} ({len(side.shards)} shards, "
                f"SIDE_MODULE={side.side_module_mode}{exports_part}){size_part}"
            )
            if side.link_argv:
                lines.append("      Link: " + " ".join(side.link_argv))
    if report.link_argv:
        lines.append("  Link: " + " ".join(report.link_argv))
    if report.outputs:
        lines.append("  Outputs: " + ", ".join(report.outputs))
    if report.removed_stale_side_modules:
        stale_label = (
            "Stale side modules (would remove)"
            if report.mode == "dry-run"
            else "Removed stale side modules"
        )
        lines.append(f"  {stale_label}: " + ", ".join(report.removed_stale_side_modules))
    if report.wasm_bytes:
        mib = report.wasm_bytes / (1024 * 1024)
        limit_note = " [EXCEEDS V8 1 GiB LIMIT]" if report.exceeds_v8_1gib_limit else ""
        lines.append(f"  Wasm size: {report.wasm_bytes} bytes ({mib:.2f} MiB){limit_note}")
    if report.wasm_metrics is not None:
        lines.extend(format_wasm_metrics_lines(report.wasm_metrics))
    if report.guest_manifest:
        if report.stage_seconds:
            lines.append(
                "  Stage seconds: "
                + ", ".join(f"{k}={v}" for k, v in report.stage_seconds.items())
            )
        lines.append(f"  Guest manifest: {report.guest_manifest}")
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
        "--link-jobs",
        type=int,
        default=DEFAULT_LINK_JOBS,
        help=(
            "Side-module parts linked in parallel (default: %(default)s; each link can use "
            "several GiB of node heap, so keep this low on small machines)."
        ),
    )
    parser.add_argument(
        "--no-link-cache",
        action="store_true",
        help="Always relink parts/runtime and re-inspect wasm instead of reusing cached results.",
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
        "--opt-module",
        action="append",
        default=[],
        dest="opt_modules",
        metavar="MODULE=LEVEL",
        help=(
            "Per-module optimization level override (repeatable; e.g. "
            "--opt-module bink2scarlett=O3). Valid levels: O0, O1, O2, O3, Os, Oz."
        ),
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
        "--no-link-opt",
        action="store_true",
        help=(
            "Do not pass -Os/-Oz (or any -O* opt level) on the em++ link line, skipping "
            "full-module Binaryen wasm-opt while still compiling shards with --opt."
        ),
    )
    parser.add_argument(
        "--shrink-largest-shards",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Compile the top-N largest .ll shards at -Oz instead of --opt (must be >= 0; "
            "no-op when --opt is already Oz)."
        ),
    )
    parser.add_argument(
        "--wasm-exceptions",
        action="store_true",
        help=(
            "Enable native Wasm C++ exceptions (-fwasm-exceptions) on C++ translation units "
            "and on the em++ link line (never applied to .ll IR shards)."
        ),
    )
    parser.add_argument(
        "--initial-memory",
        default=None,
        metavar="SIZE",
        help="Pass -sINITIAL_MEMORY=SIZE on the link line (e.g. 64MB).",
    )
    parser.add_argument(
        "--stack-size",
        default=None,
        metavar="SIZE",
        help=(
            "Pass both -sSTACK_SIZE=SIZE and -sDEFAULT_PTHREAD_STACK_SIZE=SIZE on the link "
            "line (e.g. 8MB)."
        ),
    )
    parser.add_argument(
        "--assertions",
        type=int,
        default=None,
        metavar="0|1",
        help=(
            "Explicitly set -sASSERTIONS=0 or -sASSERTIONS=1 on the link line "
            "(overrides --strip's -sASSERTIONS=0 when set)."
        ),
    )
    parser.add_argument(
        "--browser-boot-exports",
        action="store_true",
        help=(
            "Export Emscripten 7.0.1 incoming Module JS API and runtime methods required by "
            "web/loader.js for browser boot."
        ),
    )
    parser.add_argument(
        "--split-modules",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Split .ll shards into N Emscripten side modules (plus an extra side module for "
            "functions.ll registration tables) paired with a -sMAIN_MODULE=1 runtime loaded via "
            "Module.dynamicLibraries (must be >= 1; default: 1 = monolithic). Uses "
            f"{SIDE_MODULE_FLAG} unless --split-strategy asymmetric selects {SIDE_MODULE_PRUNED_FLAG}."
        ),
    )
    parser.add_argument(
        "--split-strategy",
        default=DEFAULT_SPLIT_STRATEGY,
        metavar="balanced|contiguous|asymmetric|capped",
        help=(
            "Shard bucketing strategy when --split-modules > 1: 'balanced' (default, greedy LPT "
            "size bin-packing), 'contiguous' (exact minimax contiguous partition in sorted "
            "unit_id / guest VA order), or 'asymmetric' (opt-in browser-proven contiguous pack "
            f"with part_0 capped at --max-side-module-mib / default {DEFAULT_MAX_SIDE_MODULE_MIB} "
            f"MiB and {SIDE_MODULE_PRUNED_FLAG} export pruning), or 'capped' (contiguous, parts "
            f"auto-sized to <= {DEFAULT_CAPPED_SIDE_MODULE_MIB} MiB each and balanced; --split-modules "
            "is only the upper bound; errors instead of emitting an over-cap part)."
        ),
    )
    parser.add_argument(
        "--max-side-module-mib",
        type=int,
        default=DEFAULT_MAX_SIDE_MODULE_MIB,
        metavar="MIB",
        help=(
            "Asymmetric part_0 ceiling in MiB when --split-strategy asymmetric "
            f"(default: {DEFAULT_MAX_SIDE_MODULE_MIB}; must be 1..{MAX_SIDE_MODULE_MIB_LIMIT}). "
            "Uses the same .o/source size estimate as balanced/contiguous partitioning."
        ),
    )
    parser.add_argument(
        "--collapse-single-side-module",
        action="store_true",
        help=(
            "When --split-strategy asymmetric is used and all code shards fit within "
            "--max-side-module-mib, collapse them into a single code side module "
            "(guest_part_0.wasm + registration guest_part_1.wasm) to eliminate cross-part "
            "sub_* JS stub bounces."
        ),
    )
    parser.add_argument(
        "--proxy-to-pthread",
        action="store_true",
        help=(
            f"Pass {PROXY_TO_PTHREAD_FLAG} on the link line so main() runs on a dedicated "
            "pthread instead of blocking the coordinator worker JS event loop."
        ),
    )
    parser.add_argument(
        "--max-concurrent-pthread-loads",
        type=int,
        default=DEFAULT_MAX_CONCURRENT_PTHREAD_LOADS,
        metavar="N",
        help=(
            "Max concurrent pthread CMD_LOAD instantiations in split-module runtime.js "
            f"post-link optimization (default: {DEFAULT_MAX_CONCURRENT_PTHREAD_LOADS}; "
            "0 disables CMD_LOAD throttling)."
        ),
    )
    parser.add_argument(
        "--node-max-old-space-mb",
        type=int,
        default=None,
        metavar="MB",
        help=(
            f"Set NODE_OPTIONS=--max-old-space-size=MB for the main em++ link "
            f"(defaults to {DEFAULT_SPLIT_NODE_MAX_OLD_SPACE_MB} automatically when "
            f"--split-modules > 1 and NODE_OPTIONS does not already set "
            f"--max-old-space-size; untouched for --split-modules 1 unless set)."
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
        help=(
            "Override -sPTHREAD_POOL_SIZE=N on the link line (must be >= 1; "
            f"default: 4 monolithic, {SPLIT_DEFAULT_PTHREAD_POOL_SIZE} when --split-modules > 1)."
        ),
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
        "--verify-max-table-imports",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Fail the build if total GOT.func imports or wasmTable footprint exceeds N "
            "(useful for enforcing shard-dispatch bounds in CI/local builds)."
        ),
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
            opt_modules=args.opt_modules,
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
            no_link_opt=args.no_link_opt,
            shrink_largest_shards=args.shrink_largest_shards,
            wasm_exceptions=args.wasm_exceptions,
            initial_memory=args.initial_memory,
            stack_size=args.stack_size,
            assertions=args.assertions,
            browser_boot_exports=args.browser_boot_exports,
            split_modules=args.split_modules,
            split_strategy=args.split_strategy,
            max_side_module_mib=args.max_side_module_mib,
            collapse_single_side_module=args.collapse_single_side_module,
            node_max_old_space_mb=args.node_max_old_space_mb,
            proxy_to_pthread=args.proxy_to_pthread,
            max_concurrent_pthread_loads=args.max_concurrent_pthread_loads,
            verify_max_table_imports=args.verify_max_table_imports,
            link_jobs=max(1, args.link_jobs),
            link_cache=not args.no_link_cache,
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
