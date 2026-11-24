#!/usr/bin/env python3
"""Native first-boot harness builder (Track 1 / R14).

Turns one module of a `playgta6-recomp build --target native` output directory into a host
executable that boots the guest in an ordinary, debuggable process
(`runtime/guest/boot_main.cpp`, boot sequence in `runtime/guest/boot.h`):

1. every lifted shard (`lift/shard-NNNNN.ll`) and the registration table (`lift/functions.ll`) is
   compiled to a host object with the LLVM tools (`opt -O<n>` then `llc -O<n>`), in parallel,
   cached by content: the cache key is SHA-256 over the IR (or the build manifest's FNV-1a hash
   of it, which lets `--delete-ir` drop compiled IR and still rebuild nothing), the flags and
   the tool versions. A wasm64 build is accepted too: its IR is retargeted on the fly exactly like
   `playgta6-recomp build --target native` would emit it (`recomp/src/lift/target.rs`: the
   datalayout line is dropped, the triple replaced, `attributes #0` set to
   `uwtable "target-cpu"="<cpu>"`).
2. a small CMake project (`<out>/harness/`) builds the host runtime library from `runtime/`
   (`add_subdirectory`, only the needed targets), the generated import bridge
   (`stubs/pg6_import_thunks.cpp`) and `boot_main.cpp` (`PG6_BOOT_LINKED_GUEST`), and links them
   with all objects into `<out>/bin/playgta6_boot[.exe]`.

Toolchains: LLVM `opt`/`llc` from `--llvm-bin`, `PLAYGTA6_LLVM_BIN`, `PATH` or the rustup
`llvm-tools` component (`rustup component add llvm-tools`). C++ compiler and linker are whatever
CMake picks for the host: on Windows the Visual Studio generator (MSVC `cl`/`link`, no vcvars
needed), elsewhere the default C++ compiler (`-relocation-model=pic` objects, PIE link).
Linker notes for multi-GB object sets: the MSVC link uses `/INCREMENTAL:NO`, `/OPT:NOREF,NOICF`
(reference elimination over a whole game's code costs minutes and saves nothing here) and
`/DEBUG:NONE` unless `--debug-info`. The harness image is linked at a fixed high base
(`/BASE:0x7E0000000000 /DYNAMICBASE:NO`) so it never sits in the guest address space
(the game wants 0x1'4000'0000, MSVC's default EXE base; the runtime's guest space starts at
4 GiB). Objects are passed as CMake external-object sources (MSBuild hands them to `link`
through its own response file).

Pure Python 3 standard library. Usage:

    python tools/build_native_boot.py <recomp-out>/<module> [--out <dir>] [-O 1] [--jobs N]
        [--cpu x86-64-v3] [--triple <t>] [--llvm-bin <dir>] [--cache-dir <dir>] [--sample N]
        [--delete-ir] [--no-link] [--debug-info] [--keep-going] [--json]

Outputs go to `<module-dir>/native/` by default (inside the gitignored build output). Prints
progress with per-shard timings and an ETA, a summary with object bytes and times, and the
command to run the harness. `--sample N` compiles only N evenly spread shards and extrapolates
time and size for the whole set (nothing is linked).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

SCHEMA_VERSION = 1
# Bump whenever retarget_ir changes its output (part of the object cache key).
RETARGET_VERSION = 2
REPO_ROOT = Path(__file__).resolve().parent.parent
NATIVE_DEFAULT_CPU = "x86-64-v3"
DEFAULT_OPT_LEVEL = "1"
WINDOWS_HARNESS_BASE = "0x7E0000000000"
HARNESS_TARGET = "playgta6_boot_guest"
HARNESS_OUTPUT = "playgta6_boot"
MANIFEST_FILE = "build-manifest.json"
PROGRAM_MANIFEST_FILE = "program-manifest.json"
WASM64_TRIPLE = "wasm64-unknown-unknown"
CMAKE_FALLBACKS = (Path(r"C:\Program Files\CMake\bin\cmake.exe"),)


class NativeBootError(Exception):
    """User-facing configuration or build failure."""


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


# (argv, stdin bytes or None) -> result
CommandRunner = Callable[[Sequence[str], "bytes | None"], CommandResult]


def run_command(argv: Sequence[str], stdin: bytes | None = None) -> CommandResult:
    proc = subprocess.run(list(argv), input=stdin, capture_output=True, check=False)
    return CommandResult(
        proc.returncode,
        proc.stdout.decode("utf-8", "replace"),
        proc.stderr.decode("utf-8", "replace"),
    )


# ---------------------------------------------------------------------------
# Targets and IR
# ---------------------------------------------------------------------------


def host_triple() -> str:
    if sys.platform == "win32":
        return "x86_64-pc-windows-msvc"
    if sys.platform == "darwin":
        return "x86_64-apple-darwin"
    return "x86_64-unknown-linux-gnu"


WASM_SWIZZLE_CALL = "@llvm.wasm.swizzle("
NATIVE_SWIZZLE_CALL = "@__playgta6_swizzle("
_V16 = "<" + ", ".join(["i8 16"] * 16) + ">"
_VNEG = "<" + ", ".join(["i8 -128"] * 16) + ">"
NATIVE_SWIZZLE_IR = (
    "define internal <16 x i8> @__playgta6_swizzle(<16 x i8> %a, <16 x i8> %idx) #0 {\n"
    f"  %big = icmp uge <16 x i8> %idx, {_V16}\n"
    f"  %sel = select <16 x i1> %big, <16 x i8> {_VNEG}, <16 x i8> %idx\n"
    "  %r = call <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8> %a, <16 x i8> %sel)\n"
    "  ret <16 x i8> %r\n"
    "}\n"
    "declare <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8>, <16 x i8>)\n"
)


def _linkonce_global_with_comdat(body: str) -> str | None:
    """`@x = linkonce_odr ...` globals get their own COMDAT (COFF cannot merge them otherwise)."""
    if not body.startswith("@") or " = " not in body:
        return None
    name, definition = body[1:].split(" = ", 1)
    if not definition.startswith("linkonce_odr ") or "comdat" in definition:
        return None
    at = definition.rfind(", align ")
    definition = definition[:at] + ", comdat" + definition[at:] if at >= 0 else definition + ", comdat"
    return f"${name} = comdat any\n@{name} = {definition}\n"


def retarget_ir(text: str, triple: str, cpu: str) -> str:
    """Rewrites the target lines of a lifted module (mirrors `LiftTarget::retarget`, including
    the native replacement of the wasm-only `llvm.wasm.swizzle` intrinsic)."""
    swizzle = WASM_SWIZZLE_CALL in text
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        if body.startswith("target datalayout = "):
            continue
        if swizzle and WASM_SWIZZLE_CALL in body:
            if body.startswith("declare "):
                out.append(NATIVE_SWIZZLE_IR)
            else:
                out.append(line.replace(WASM_SWIZZLE_CALL, NATIVE_SWIZZLE_CALL))
            continue
        comdat = _linkonce_global_with_comdat(body)
        if comdat is not None:
            out.append(comdat)
            continue
        if body.startswith("target triple = "):
            out.append(f'target triple = "{triple}"\n')
        elif body.startswith("attributes #0 = "):
            out.append(f'attributes #0 = {{ uwtable "target-cpu"="{cpu}" }}\n')
        else:
            out.append(line)
    return "".join(out)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Unit:
    """One IR module to compile."""

    name: str  # e.g. shard-00012 / functions
    ll: Path
    manifest_hash: str | None  # FNV-1a 64 of the .ll from the build manifest
    instructions: int = 0


@dataclass
class ModuleInputs:
    module_dir: Path
    module: str
    target: str  # "wasm64" or "native:<triple>"
    units: list[Unit]
    stubs_dir: Path
    pgim: Path
    manifest: dict = field(default_factory=dict)


def load_module(module_dir: Path) -> ModuleInputs:
    module_dir = module_dir.resolve()
    manifest_path = module_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        raise NativeBootError(
            f"{manifest_path.name} not found in {module_dir} (pass the module directory of a "
            "`playgta6-recomp build` output, e.g. <out>/main)"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise NativeBootError(f"cannot read {manifest_path.name}: {err}") from err
    if manifest.get("schema") != "playgta6-recomp-build":
        raise NativeBootError(f"{manifest_path.name} is not a playgta6-recomp build manifest")
    hashes = {f["path"]: f.get("fnv1a64") for f in manifest.get("files", []) if "path" in f}
    lift = manifest.get("lift", {})
    units: list[Unit] = []
    for shard in lift.get("shards", []):
        rel = shard["ll"]
        units.append(
            Unit(
                name=Path(rel).stem,
                ll=module_dir / rel,
                manifest_hash=hashes.get(rel),
                instructions=int(shard.get("instructions", 0)),
            )
        )
    registration = lift.get("registration", {})
    if registration.get("ll"):
        rel = registration["ll"]
        units.append(Unit(Path(rel).stem, module_dir / rel, hashes.get(rel)))
    if not units:
        raise NativeBootError("the manifest lists no lifted shards")
    target = manifest.get("options", {}).get("target", "wasm64")
    return ModuleInputs(
        module_dir=module_dir,
        module=str(manifest.get("module", "main")),
        target=target,
        units=units,
        stubs_dir=module_dir / "stubs",
        pgim=module_dir / "image" / "guest.pgim",
        manifest=manifest,
    )


# ---------------------------------------------------------------------------
# LLVM tools
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LlvmTools:
    opt: Path
    llc: Path
    version: str


def _exe(name: str) -> str:
    return name + (".exe" if sys.platform == "win32" else "")


def find_llvm_tools(llvm_bin: Path | None, runner: CommandRunner = run_command) -> LlvmTools:
    candidates: list[Path] = []
    if llvm_bin is not None:
        candidates.append(llvm_bin)
    if os.environ.get("PLAYGTA6_LLVM_BIN"):
        candidates.append(Path(os.environ["PLAYGTA6_LLVM_BIN"]))
    on_path = shutil.which("llc")
    if on_path:
        candidates.append(Path(on_path).parent)
    try:
        sysroot = subprocess.run(
            ["rustc", "--print", "sysroot"], capture_output=True, text=True, check=False
        ).stdout.strip()
        if sysroot:
            candidates.append(Path(sysroot) / "lib" / "rustlib" / host_triple() / "bin")
    except OSError:
        pass
    for directory in candidates:
        opt = directory / _exe("opt")
        llc = directory / _exe("llc")
        if opt.is_file() and llc.is_file():
            res = runner([str(llc), "--version"], None)
            version = next(
                (l.strip() for l in res.stdout.splitlines() if "version" in l.lower()), "unknown"
            )
            return LlvmTools(opt, llc, version)
    raise NativeBootError(
        "no LLVM opt/llc found (pass --llvm-bin, set PLAYGTA6_LLVM_BIN, put llc on PATH or run "
        "`rustup component add llvm-tools`)"
    )


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompileConfig:
    triple: str
    cpu: str
    opt_level: str
    cache_dir: Path
    log_dir: Path
    tools: LlvmTools
    delete_ir: bool = False

    @property
    def obj_suffix(self) -> str:
        return ".obj" if "windows" in self.triple else ".o"

    @property
    def pic(self) -> bool:
        return "windows" not in self.triple

    def flags(self) -> list[str]:
        return [self.triple, self.cpu, f"O{self.opt_level}", "pic" if self.pic else "static"]


@dataclass
class UnitResult:
    unit: Unit
    obj: Path | None
    cached: bool
    seconds: float
    error: str | None = None

    @property
    def obj_bytes(self) -> int:
        return self.obj.stat().st_size if self.obj is not None and self.obj.is_file() else 0


def cache_key(unit: Unit, cfg: CompileConfig, content: bytes | None) -> str:
    h = hashlib.sha256()
    h.update(f"playgta6-native-obj v{SCHEMA_VERSION} retarget {RETARGET_VERSION}\n".encode())
    h.update(("\n".join(cfg.flags()) + "\n" + cfg.tools.version + "\n").encode())
    if content is not None:
        h.update(hashlib.sha256(content).digest())
    elif unit.manifest_hash is not None:
        h.update(f"fnv1a64:{unit.manifest_hash}".encode())
    else:
        raise NativeBootError(f"{unit.ll} is missing and the manifest has no hash for it")
    return h.hexdigest()


def compile_unit(unit: Unit, cfg: CompileConfig, runner: CommandRunner = run_command) -> UnitResult:
    start = time.monotonic()
    # The manifest hash is the content identity when present (cheap, and survives --delete-ir).
    content = None if unit.manifest_hash is not None else _read(unit.ll)
    key = cache_key(unit, cfg, content)
    obj = cfg.cache_dir / f"{key}{cfg.obj_suffix}"
    if obj.is_file():
        return UnitResult(unit, obj, True, time.monotonic() - start)
    if not unit.ll.is_file():
        return UnitResult(unit, None, False, 0.0, f"{unit.ll.name} is missing (deleted IR?)")
    # Idempotent: wasm64 IR is retargeted, native IR gets the requested CPU.
    text = retarget_ir(_read(unit.ll).decode("utf-8"), cfg.triple, cfg.cpu)
    bc = cfg.cache_dir / f"{key}.tmp.bc"
    tmp_obj = cfg.cache_dir / f"{key}.tmp{cfg.obj_suffix}"
    opt = runner([str(cfg.tools.opt), f"-O{cfg.opt_level}", "-", "-o", str(bc)], text.encode("utf-8"))
    del text
    if opt.returncode != 0:
        return _failed(unit, cfg, start, "opt", opt, bc, tmp_obj)
    llc_argv = [
        str(cfg.tools.llc),
        f"-mtriple={cfg.triple}",
        f"-mcpu={cfg.cpu}",
        f"-O{cfg.opt_level}",
        "-filetype=obj",
    ]
    if cfg.pic:
        llc_argv.append("-relocation-model=pic")
    llc = runner(llc_argv + [str(bc), "-o", str(tmp_obj)], None)
    if llc.returncode != 0:
        return _failed(unit, cfg, start, "llc", llc, bc, tmp_obj)
    bc.unlink(missing_ok=True)
    os.replace(tmp_obj, obj)
    if cfg.delete_ir and unit.manifest_hash is not None:
        unit.ll.unlink(missing_ok=True)
    return UnitResult(unit, obj, False, time.monotonic() - start)


def _read(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as err:
        raise NativeBootError(f"cannot read {path}: {err}") from err


def _failed(
    unit: Unit, cfg: CompileConfig, start: float, tool: str, res: CommandResult, *tmp: Path
) -> UnitResult:
    for path in tmp:
        path.unlink(missing_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    log = cfg.log_dir / f"{unit.name}.{tool}.log"
    log.write_text(res.stdout + res.stderr, encoding="utf-8")
    first = next((l for l in (res.stderr or res.stdout).splitlines() if l.strip()), "")
    return UnitResult(
        unit, None, False, time.monotonic() - start,
        f"{tool} failed (exit {res.returncode}): {first[:300]} [log: {log}]",
    )


def compile_all(
    units: Sequence[Unit],
    cfg: CompileConfig,
    jobs: int,
    runner: CommandRunner = run_command,
    keep_going: bool = False,
    progress: Callable[[str], None] | None = None,
) -> list[UnitResult]:
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    results: list[UnitResult] = []
    lock = threading.Lock()
    stop = threading.Event()
    start = time.monotonic()
    total = len(units)

    def work(unit: Unit) -> UnitResult:
        if stop.is_set():
            return UnitResult(unit, None, False, 0.0, "skipped after an earlier failure")
        res = compile_unit(unit, cfg, runner)
        if res.error is not None and not keep_going:
            stop.set()
        return res

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = [pool.submit(work, u) for u in units]
        for fut in as_completed(futures):
            res = fut.result()
            with lock:
                results.append(res)
                done = len(results)
            if progress is not None:
                elapsed = time.monotonic() - start
                eta = elapsed / done * (total - done) if done else 0.0
                state = "cached" if res.cached else ("FAILED" if res.error else f"{res.seconds:.1f}s")
                progress(
                    f"[{done}/{total}] {res.unit.name} {state} "
                    f"(elapsed {elapsed:.0f}s, eta {eta:.0f}s)"
                )
    order = {u.name: i for i, u in enumerate(units)}
    results.sort(key=lambda r: order[r.unit.name])
    return results


def sample_units(units: Sequence[Unit], count: int) -> list[Unit]:
    """`count` shards spread evenly over the set (the registration table is not sampled)."""
    shards = [u for u in units if Path(u.name).name.startswith("shard-")]
    if count >= len(shards):
        return list(shards)
    step = len(shards) / count
    return [shards[int(i * step)] for i in range(count)]


# ---------------------------------------------------------------------------
# Harness project
# ---------------------------------------------------------------------------


def _cmake_path(path: Path) -> str:
    return path.resolve().as_posix()


@dataclass
class BuildInputs:
    """What to compile and link: one module build, or a program build (several modules)."""

    root: Path  # module directory or program directory
    label: str
    target: str  # "wasm64" or "native:<triple>" (of the first module)
    units: list[Unit]
    sources: list[Path]  # C++ sources of the harness besides boot_main.cpp
    include_dirs: list[Path]
    defines: list[str]
    run_args: str  # how the harness finds its images


def module_inputs(inputs: ModuleInputs) -> BuildInputs:
    return BuildInputs(
        root=inputs.module_dir,
        label=f"module {inputs.module}",
        target=inputs.target,
        units=list(inputs.units),
        sources=[inputs.stubs_dir / "pg6_import_thunks.cpp"],
        include_dirs=[inputs.stubs_dir],
        defines=["PG6_BOOT_LINKED_GUEST", f"PG6_BOOT_MODULE={inputs.module}"],
        run_args=f"--pgim {inputs.pgim}",
    )


def load_program(program_dir: Path) -> BuildInputs:
    """A `playgta6-recomp build-program` output: every module's shards, the generated program
    descriptor (`pg6_program.cpp`) and every module's import bridge."""
    program_dir = program_dir.resolve()
    path = program_dir / PROGRAM_MANIFEST_FILE
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise NativeBootError(f"cannot read {path.name}: {err}") from err
    if manifest.get("schema") != "playgta6-recomp-program":
        raise NativeBootError(f"{path.name} is not a playgta6-recomp program manifest")
    modules = [load_module(program_dir / str(m["module"])) for m in manifest.get("modules", [])]
    if not modules:
        raise NativeBootError("the program manifest lists no modules")
    units = [
        dataclasses.replace(u, name=f"{m.module}/{u.name}") for m in modules for u in m.units
    ]
    targets = {m.target for m in modules}
    if len(targets) != 1:
        raise NativeBootError(f"modules were built for different targets: {sorted(targets)}")
    return BuildInputs(
        root=program_dir,
        label=f"program ({len(modules)} modules)",
        target=modules[0].target,
        units=units,
        sources=[program_dir / "pg6_program.cpp"]
        + [m.stubs_dir / "pg6_import_thunks.cpp" for m in modules],
        include_dirs=[program_dir],
        defines=["PG6_BOOT_PROGRAM"],
        run_args=f"--program-dir {program_dir}",
    )


def load_inputs(path: Path) -> BuildInputs:
    if (path / PROGRAM_MANIFEST_FILE).is_file():
        return load_program(path)
    return module_inputs(load_module(path))


def harness_cmakelists(
    repo_root: Path,
    inputs: BuildInputs,
    objects: Sequence[Path],
    bin_dir: Path,
    debug_info: bool,
) -> str:
    objs = "\n".join(f'    "{_cmake_path(o)}"' for o in objects)
    sources = "\n".join(f'    "{_cmake_path(s)}"' for s in inputs.sources)
    includes = " ".join(f'"{_cmake_path(d)}"' for d in inputs.include_dirs)
    defines = " ".join(inputs.defines)
    msvc_debug = "/DEBUG" if debug_info else "/DEBUG:NONE"
    return f"""# Generated by tools/build_native_boot.py. Do not edit.
cmake_minimum_required(VERSION 3.24)
project(playgta6_native_boot LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory("{_cmake_path(repo_root / 'runtime')}" runtime EXCLUDE_FROM_ALL)

add_executable({HARNESS_TARGET}
    "{_cmake_path(repo_root / 'runtime' / 'guest' / 'boot_main.cpp')}"
{sources}
{objs}
)
set_source_files_properties(
{objs}
    PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
target_compile_definitions({HARNESS_TARGET} PRIVATE {defines})
target_include_directories({HARNESS_TARGET} PRIVATE {includes})
target_link_libraries({HARNESS_TARGET} PRIVATE playgta6_core)
set_target_properties({HARNESS_TARGET} PROPERTIES
    OUTPUT_NAME {HARNESS_OUTPUT}
    RUNTIME_OUTPUT_DIRECTORY "{_cmake_path(bin_dir)}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "{_cmake_path(bin_dir)}"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "{_cmake_path(bin_dir)}")
if(MSVC)
    target_compile_options({HARNESS_TARGET} PRIVATE /bigobj)
    target_link_options({HARNESS_TARGET} PRIVATE
        /INCREMENTAL:NO /OPT:NOREF /OPT:NOICF {msvc_debug}
        /DYNAMICBASE:NO /BASE:{WINDOWS_HARNESS_BASE})
endif()
"""


def find_cmake() -> str:
    found = shutil.which("cmake")
    if found:
        return found
    for candidate in CMAKE_FALLBACKS:
        if candidate.is_file():
            return str(candidate)
    raise NativeBootError("cmake not found (put it on PATH)")


def build_harness(
    out_dir: Path,
    repo_root: Path,
    inputs: BuildInputs,
    objects: Sequence[Path],
    debug_info: bool,
    jobs: int,
    runner: CommandRunner = run_command,
    progress: Callable[[str], None] | None = None,
) -> Path:
    project = out_dir / "harness"
    build = out_dir / "harness-build"
    bin_dir = out_dir / "bin"
    project.mkdir(parents=True, exist_ok=True)
    text = harness_cmakelists(repo_root, inputs, objects, bin_dir, debug_info)
    cml = project / "CMakeLists.txt"
    if not cml.is_file() or cml.read_text(encoding="utf-8") != text:
        cml.write_text(text, encoding="utf-8")
    cmake = find_cmake()
    config = "RelWithDebInfo" if debug_info else "Release"
    steps = [
        [cmake, "-S", str(project), "-B", str(build), f"-DCMAKE_BUILD_TYPE={config}"],
        [cmake, "--build", str(build), "--config", config, "--target", HARNESS_TARGET,
         "--parallel", str(max(1, jobs))],
    ]
    for argv in steps:
        if progress is not None:
            progress("$ " + " ".join(argv[:4]) + " ...")
        res = runner(argv, None)
        if res.returncode != 0:
            log = out_dir / "harness-build.log"
            log.write_text(res.stdout + res.stderr, encoding="utf-8")
            tail = "\n".join((res.stdout + res.stderr).splitlines()[-25:])
            raise NativeBootError(f"harness build failed (see {log}):\n{tail}")
    exe = bin_dir / (_exe(HARNESS_OUTPUT))
    if not exe.is_file():
        raise NativeBootError(f"harness build reported success but {exe} is missing")
    return exe


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _mib(n: int) -> str:
    return f"{n / (1 << 20):.1f} MiB"


def main(argv: Sequence[str] | None = None, runner: CommandRunner = run_command) -> int:
    parser = argparse.ArgumentParser(
        description="Compile a native-target recomp build and link the first-boot harness."
    )
    parser.add_argument(
        "module_dir", type=Path,
        help="<recomp-out>/<module> of `build`, or the <out> directory of `build-program`",
    )
    parser.add_argument("--out", type=Path, help="output directory (default <input-dir>/native)")
    parser.add_argument("-O", dest="opt_level", default=DEFAULT_OPT_LEVEL, choices=["0", "1", "2", "3"])
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--cpu", default=NATIVE_DEFAULT_CPU)
    parser.add_argument("--triple", default=None, help="default: the build's native triple or the host's")
    parser.add_argument("--llvm-bin", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None, help="default <out>/obj-cache")
    parser.add_argument("--sample", type=int, default=0, help="compile N spread shards and extrapolate")
    parser.add_argument("--delete-ir", action="store_true", help="delete each .ll once compiled")
    parser.add_argument("--no-link", action="store_true")
    parser.add_argument("--debug-info", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--json", action="store_true", help="print a JSON summary")
    args = parser.parse_args(argv)

    def say(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    try:
        inputs = load_inputs(args.module_dir)
        triple = args.triple
        if triple is None:
            triple = inputs.target.split(":", 1)[1] if inputs.target.startswith("native:") else host_triple()
        if not triple.startswith("x86_64-"):
            raise NativeBootError(f"triple {triple} is not x86_64")
        out_dir = (args.out or inputs.root / "native").resolve()
        tools = find_llvm_tools(args.llvm_bin, runner)
        cfg = CompileConfig(
            triple=triple,
            cpu=args.cpu,
            opt_level=args.opt_level,
            cache_dir=(args.cache_dir or out_dir / "obj-cache").resolve(),
            log_dir=out_dir / "logs",
            tools=tools,
            delete_ir=args.delete_ir,
        )
        units = sample_units(inputs.units, args.sample) if args.sample > 0 else inputs.units
        say(
            f"{inputs.label} ({inputs.target}): {len(units)} unit(s) -> {triple} "
            f"cpu {args.cpu} -O{args.opt_level}, {args.jobs} job(s); {tools.version}"
        )
        t0 = time.monotonic()
        results = compile_all(units, cfg, args.jobs, runner, args.keep_going, say)
        compile_s = time.monotonic() - t0
        failed = [r for r in results if r.error]
        built = [r for r in results if not r.error and not r.cached]
        obj_bytes = sum(r.obj_bytes for r in results)
        summary: dict = {
            "schema": "playgta6-native-boot-build",
            "schema_version": SCHEMA_VERSION,
            "inputs": inputs.label,
            "units": len(units),
            "compiled": len(built),
            "cached": sum(1 for r in results if r.cached),
            "failed": [{"unit": r.unit.name, "error": r.error} for r in failed],
            "compile_seconds": round(compile_s, 1),
            "cpu_seconds": round(sum(r.seconds for r in results), 1),
            "object_bytes": obj_bytes,
        }
        say(
            f"compiled {len(built)}, cached {summary['cached']}, failed {len(failed)} in "
            f"{compile_s:.0f}s wall ({summary['cpu_seconds']:.0f}s summed); objects {_mib(obj_bytes)}"
        )
        for r in failed[:20]:
            say(f"  {r.unit.name}: {r.error}")
        if args.sample > 0:
            all_instr = sum(u.instructions for u in inputs.units)
            sample_instr = sum(r.unit.instructions for r in results) or 1
            # Time only from shards compiled now (cached ones took no time).
            built_instr = sum(r.unit.instructions for r in built) or 1
            cpu = sum(r.seconds for r in built) * all_instr / built_instr if built else 0.0
            summary["extrapolated"] = {
                "object_bytes": int(obj_bytes * all_instr / sample_instr),
                "cpu_seconds": round(cpu, 1),
                "wall_seconds_at_jobs": round(cpu / max(1, args.jobs), 1),
            }
            say(
                f"sample covers {sample_instr} of {all_instr} instructions: full build ~"
                f"{_mib(summary['extrapolated']['object_bytes'])} of objects, ~"
                f"{summary['extrapolated']['wall_seconds_at_jobs']:.0f}s at {args.jobs} jobs"
            )
        elif failed:
            raise NativeBootError(f"{len(failed)} unit(s) failed to compile (logs in {cfg.log_dir})")
        elif not args.no_link:
            t1 = time.monotonic()
            objects = [r.obj for r in results if r.obj is not None]
            exe = build_harness(out_dir, REPO_ROOT, inputs, objects, args.debug_info, args.jobs, runner, say)
            summary["link_seconds"] = round(time.monotonic() - t1, 1)
            summary["harness"] = str(exe)
            summary["harness_bytes"] = exe.stat().st_size
            say(f"linked {exe} ({_mib(exe.stat().st_size)}) in {summary['link_seconds']:.0f}s")
            say(f"run: {exe} {inputs.run_args} --name <exe name> --game-dir <game dir> --report boot-report.json")
        if args.json:
            print(json.dumps(summary, indent=2))
        return 1 if failed and args.sample == 0 else 0
    except NativeBootError as err:
        say(f"error: {err}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
