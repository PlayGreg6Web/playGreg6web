#!/usr/bin/env python3
"""One-command native boot runner: rebuild what is stale, run `playgta6_boot`, keep the run.

    python tools/run_native_boot.py [--preset smoke|start-screen|menu-flow|menu-flow-2|menu-flow-3|menu-flow-4|menu-to-ingame|long]
        [--name NAME]
        [--exe PATH] [--game-dir DIR] [options] [-- extra playgta6_boot options]

What it does, in order:

1. Staleness (mtime based, cheap): the recompiler (`target/release/playgta6-recomp`) is rebuilt
   with `cargo build --release -p playgta6-recomp` when missing or older than the sources of the
   workspace crates; the program (`<build-dir>/program`, `playgta6-recomp build-program <exe>
   --out ... --dll-dir <game-dir> --target native`) when its `program-manifest.json` is missing
   or older than the recompiler, the exe or the runtime files it binds imports from
   (`runtime/os/*.h|*.cpp`, `runtime/gpu/d3d12_xs.h`, `d3d12_xs_exports.*`,
   `runtime/guest/guest_callbacks.h`). A stale program is first looked up in the lift cache
   (`<object cache>-lift`, or `PLAYGTA6_LIFT_CACHE`; `off` disables) by a hash of the recompiler
   sources, the exe and DLL bytes, the shim headers and the lift options: a hit hard-links the
   cached shard IR and metadata in and skips the recompiler build, discovery and the lift; a miss
   lifts and stores the result (the 2 most recent entries are kept, `PLAYGTA6_LIFT_CACHE_KEEP`);
   the harness (`python tools/build_native_boot.py
   <build-dir>/program --keep-going --delete-ir --debug-info`, objects from the machine-wide cache
   shared by all checkouts, or `--cache-dir DIR` / `PLAYGTA6_OBJ_CACHE`;
   the PDB lets `--profile` name runtime functions) when the binary is missing or older than the
   program manifest, any `runtime/` source or `tools/build_native_boot.py`. Only stale stages run, in dependency order (a rebuilt stage makes
   the later ones stale). `--rebuild` forces all, `--no-build` fails instead of building,
   `--dry-run` prints the plan and stops, `--trust-build` skips the checks and uses the existing
   program and harness as they are (for a build made elsewhere, e.g. from another checkout).
2. Run `playgta6_boot --program-dir <program> --name <exe name> --game-dir <game-dir>` with a
   preset or explicit options. Presets: `smoke` (60 s limit, 90 s watchdog), `start-screen`
   (120 s limit; an input script pressing A for 100 ms once a second from 5 s on), `menu-flow`
   (240 s limit; A every 2.5 s from 20 s, then Up x6 + Down, Down, A every 14 s through LEGAL
   NOTICES to the list screen, then A every 3 s), `menu-flow-2` (360 s limit; the same, then the
   Up x6 + Down, Down, A cycle on), `menu-flow-3` (420 s limit; the same to the list screen, then
   scrolls it with RDown + Down and presses A), `menu-flow-4` (420 s limit;
   Up + A on the list screen, which leads to the 34 draws/frame screen, then the menu cycle),
   `menu-to-ingame` (900 s limit; to the legal-document viewer, B / open, scroll, close each
   entry until the viewer is dismissed, then the browser
   preset's later phases, web/input/input_bridge.js: hold-A / Menu skip until the screen changes,
   A confirmations, then left/right stick and trigger movement with A every 12 s), `long`
   (600 s limit, 630 s watchdog). The menu-flow presets are screen-synchronised: they repeat the
   menu cycle until the list screen after LEGAL NOTICES holds (draws/frame 33 or 50, read live from
   the present counters; `wait` / `repeat` steps in runtime/guest/input_script.h), so a slower run
   (a payload capture, a busy machine) does not desync them. `--max-ms`, `--watchdog-ms`, `--input-script FILE`,
   `--shader-pack FILE`, `--gpu-capture-frames N` (captures into the run directory),
   `--gpu-capture-payloads`, `--gpu-capture-thin-uploads N`, `--gpu-capture-start-ms N` (frames start
   late; objects and the newest writes are kept), `--gpu-capture-start-at-script` (frames start at
   the script's `capture` step: the presets' list screen), `--dump-pipelines DIR` and everything
   after `--` are passed through.
3. The run lands in `<runs-dir>/<name>/`: `boot-report.json`, `summary.txt` (the harness text
   summary), `boot.log` (stdout and stderr), `capture.pg6trace` (+ `.json`, with a GPU capture),
   `input-script.txt` (with an input script), `import-trace.txt` (`--import-trace`),
   `command.json` (the command line, the staleness plan and the exit code) and `digest.txt`.
   Every run also appends the indirect-call targets it dispatched to a per-title file outside the
   repository (`<object cache>-targets/<exe>-<hash>/indirect-targets.txt`, `PLAYGTA6_INDIRECT_TARGETS`
   overrides, `off` or `--no-record-targets` disables); `--seed-targets [FILE]` lifts the program
   with those targets as function entries (validated by the recompiler; the lift report lands in
   `<build-dir>/seed-report.jsonl`: rejected targets and targets inside functions with unresolved
   jump tables, for `survey-jumps`). A stop on an unresolved indirect call: rerun, then
   `--seed-targets`, and the next boot goes past it if the code was reached.
4. `--baseline NAME|DIR` compares the run with an earlier run through
   `tools/compare_boot_reports.py` (capture and import trace are included when both runs have
   them; `--ignore`, `--time-tolerance` and `--count-tolerance` are passed on).
5. A short digest is printed: milestones, presents, stop, draw counters.

Paths never come from the repository: the exe from `--exe` or `PLAYGTA6_EXE`, the package
directory from `--game-dir` or `PLAYGTA6_GAME_DIR`. Build outputs default to `work/native-boot`
(gitignored); runs to `<build-dir>/runs`.

Exit code: 0 the run produced a report (and matches the baseline, if given), 1 the baseline
comparison found differences, 2 usage or build error, 3 the harness produced no report.
The harness' own exit code (fault, watchdog, guest exit code) is in command.json and the digest.
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
from pathlib import Path
from typing import Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import boot_summary  # noqa: E402
import compare_boot_reports  # noqa: E402
import find_chrome  # noqa: E402
import guest_symbols  # noqa: E402
import screen_flow  # noqa: E402
import screen_timeline  # noqa: E402
import stop_message  # noqa: E402
import clean_build_dirs  # noqa: E402
import build_guard  # noqa: E402
import prebuilt_guard  # noqa: E402
from lift_cache import (  # noqa: E402,F401 (re-exported for the tests)
    LIFT_CACHE_ENV, LIFT_CACHE_KEEP, LIFT_CACHE_KEEP_ENV, LIFT_CACHE_VERSION, LIFT_STAMP, PROGRAM_KEEP,
    clear_program, file_digest, lift_cache_dir, lift_cache_keep, lift_cache_restore, lift_cache_store,
    lift_inputs_key, lift_key, read_stamp, recomp_source_files, shim_inputs, workspace_dirs,
    SHARD_CACHE_ENV, shard_cache_args, shard_cache_dir, shard_cache_max_bytes, trim_shard_cache,
)

TARGETS_ENV = "PLAYGTA6_INDIRECT_TARGETS"
TARGETS_FILE = "indirect-targets.txt"

EXE_SUFFIX = ".exe" if sys.platform == "win32" else ""
SOURCE_SUFFIXES = {".cpp", ".h", ".hpp", ".c", ".cmake", ".txt", ".rs", ".toml"}

PRESETS: dict[str, dict] = {
    "smoke": {"max_ms": 60_000, "watchdog_ms": 90_000, "input": None},
    "start-screen": {"max_ms": 120_000, "watchdog_ms": 150_000, "input": "start-screen"},
    "menu-flow": {"max_ms": 240_000, "watchdog_ms": 270_000, "input": "menu-flow"},
    "menu-flow-2": {"max_ms": 360_000, "watchdog_ms": 390_000, "input": "menu-flow-2"},
    "menu-flow-3": {"max_ms": 420_000, "watchdog_ms": 450_000, "input": "menu-flow-3"},
    "menu-flow-4": {"max_ms": 420_000, "watchdog_ms": 450_000, "input": "menu-flow-4"},
    "menu-to-ingame": {"max_ms": 900_000, "watchdog_ms": 930_000, "input": "menu-to-ingame"},
    "menu-to-ingame-setup": {"max_ms": 900_000, "watchdog_ms": 930_000, "input": "menu-to-ingame-setup"},
    "long": {"max_ms": 600_000, "watchdog_ms": 630_000, "input": None},
}
INPUT_START_MS = 5_000
INPUT_PERIOD_MS = 1_000
INPUT_HOLD_MS = 100

Runner = Callable[[list[str], Path, Path | None, float | None], int]


class RunnerError(Exception):
    """Usage or build failure (exit code 2)."""


def run_command(cmd: list[str], cwd: Path, log: Path | None = None, timeout: float | None = None) -> int:
    """Runs `cmd`; with `log`, stdout and stderr go there, otherwise to this console."""
    try:
        if log is None:
            return subprocess.run(cmd, cwd=cwd, timeout=timeout).returncode
        with log.open("wb") as fh:
            return subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        return 124
    except OSError as exc:
        raise RunnerError(f"cannot run {cmd[0]}: {exc}") from exc


# ---------------------------------------------------------------------------
# Native play (--play) conveniences
# ---------------------------------------------------------------------------

SHADER_PACK_ENV = "PLAYGTA6_NATIVE_SHADER_PACK"
SHADER_PACK_DIR_ENV = "PLAYGTA6_SHADER_PACK_DIR"  # extra directory searched for *.wgsl.pack when no pack is given
SAVE_DIR_ENV = "PLAYGTA6_SAVE_DIR"


def find_shader_pack(repo: Path, env: Mapping[str, str]) -> tuple[Path | None, str]:
    """The shader pack for --play when --shader-pack is absent: $PLAYGTA6_NATIVE_SHADER_PACK if it names a
    file, else the newest `*.wgsl.pack` below $PLAYGTA6_SHADER_PACK_DIR and `<repo>/work`. Returns
    (path or None, a one-line description of where it came from)."""
    named = env.get(SHADER_PACK_ENV)
    if named:
        if Path(named).is_file():
            return Path(named), f"${SHADER_PACK_ENV}"
        return None, f"${SHADER_PACK_ENV} names a missing file: {named}"
    roots = []
    if env.get(SHADER_PACK_DIR_ENV):
        roots.append(Path(env[SHADER_PACK_DIR_ENV]))
    roots.append(repo / "work")
    best: tuple[float, Path] | None = None
    for root in roots:
        if not root.is_dir():
            continue
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if f.endswith(".wgsl.pack"):
                    path = Path(dirpath) / f
                    try:
                        mtime = path.stat().st_mtime
                    except OSError:
                        continue
                    if best is None or mtime > best[0]:
                        best = (mtime, path)
    if best is None:
        return None, "no *.wgsl.pack found"
    return best[1], "newest *.wgsl.pack under " + " / ".join(str(r) for r in roots)


def default_save_root(env: Mapping[str, str]) -> Path:
    """Persistent save directory for --play: $PLAYGTA6_SAVE_DIR, else %LOCALAPPDATA%/playgta6/saves
    (~/.local/share/playgta6/saves elsewhere)."""
    if env.get(SAVE_DIR_ENV):
        return Path(env[SAVE_DIR_ENV])
    local = env.get("LOCALAPPDATA")
    base = Path(local) if local else Path.home() / ".local" / "share"
    return base / "playgta6" / "saves"


def default_recordings_dir(env: Mapping[str, str]) -> Path:
    """Where --record-input recordings go: %LOCALAPPDATA%/playgta6/recordings."""
    local = env.get("LOCALAPPDATA")
    base = Path(local) if local else Path.home() / ".local" / "share"
    return base / "playgta6" / "recordings"


def play_save_dir(fresh: bool, explicit: Path | None, env: Mapping[str, str], now: str) -> Path:
    """--save-dir wins; --fresh-setup gives a new empty directory next to the persistent one (so the
    setup screens appear); otherwise the persistent default."""
    if explicit is not None:
        return explicit
    root = default_save_root(env)
    return root.parent / f"fresh-{now}" if fresh else root


def hard_stops_text() -> str | None:
    try:
        return HARD_STOPS_MD.read_text(encoding="utf-8") if HARD_STOPS_MD.is_file() else None
    except OSError:
        return None


def watch_for_fault(run_dir: Path, done, out=None, poll_s: float = 0.5) -> None:
    """While the window is held open on a guest fault the harness has already written the report:
    print tools/fault_report.py's output to the console once it shows up."""
    out = out or sys.stdout
    report = run_dir / "boot-report.json"
    while not done.wait(poll_s):
        data = load_report(report)
        if data and isinstance(data.get("fault_record"), dict):
            print("[play] the guest faulted; the window stays open on the last frame (close it to exit)",
                  file=out, flush=True)
            stop_message.emit(stop_message.select(data, hard_stops_text(), run_dir) or "", out)
            try:
                proc = subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "fault_report.py"), str(report)],
                                      capture_output=True, text=True, timeout=60)
                print(proc.stdout or proc.stderr, file=out, flush=True)
            except (OSError, subprocess.SubprocessError) as exc:
                print(f"[play] fault_report failed: {exc}", file=out, flush=True)
            return


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------


def newest_mtime(root: Path, files: Sequence[Path] = ()) -> float:
    """Newest mtime among the build-relevant sources below `root` (and `files`); 0 if none."""
    newest = 0.0
    for path in files:
        if path.is_file():
            newest = max(newest, path.stat().st_mtime)
    if root.is_dir():
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(("build", "target", ".")) and d != "node_modules"]
            for name in filenames:
                if Path(name).suffix in SOURCE_SUFFIXES:
                    newest = max(newest, os.stat(os.path.join(dirpath, name)).st_mtime)
    return newest


def mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


class Plan:
    def __init__(self) -> None:
        self.steps: list[dict] = []
        # stage -> {"run": fn() (instead of the command), "before": fn(), "after": fn()}; not in the JSON.
        self.hooks: dict[str, dict[str, Callable[[], None]]] = {}

    def add(self, stage: str, reason: str, cmd: list[str], **hooks: Callable[[], None]) -> None:
        self.steps.append({"stage": stage, "reason": reason, "cmd": cmd})
        if hooks:
            self.hooks[stage] = hooks

    def drop(self, stage: str) -> None:
        self.steps = [s for s in self.steps if s["stage"] != stage]
        self.hooks.pop(stage, None)

    def has(self, stage: str) -> bool:
        return any(s["stage"] == stage for s in self.steps)


# The native lift; part of the lift cache key (see tools/lift_cache.py).
NATIVE_LIFT_OPTIONS = ("--target", "native")


# Longest path below the build dir that the native build produces (MSBuild/CMake nesting under
# harness-build plus a hash-named cache object); the object names themselves are already short hashes.
BUILD_SUBPATH_BUDGET = 130
WINDOWS_PATH_LIMIT = 200


def choose_build_dir(repo: Path, requested: Path | None, env: Mapping[str, str], windows: bool | None = None,
                     limit: int = WINDOWS_PATH_LIMIT) -> tuple[Path, str | None]:
    """The build directory, and a note when the default was replaced by a short per-checkout one.

    An explicit --build-dir is always honored. On Windows the default `<repo>/work/native-boot` is
    replaced by `%LOCALAPPDATA%/playgta6/nb-<hash of repo path>` (or `<drive>:/pg6b/<hash>`) when
    its deepest generated path would exceed `limit` (MAX_PATH trouble in deep agent worktrees).
    """
    if requested is not None:
        return requested.resolve(), None
    default = (repo / "work" / "native-boot").resolve()
    windows = (os.name == "nt") if windows is None else windows
    if not windows or len(str(default)) + BUILD_SUBPATH_BUDGET <= limit:
        return default, None
    digest = hashlib.sha256(str(repo.resolve()).lower().encode("utf-8")).hexdigest()[:10]
    local = env.get("LOCALAPPDATA")
    short = Path(local) / "playgta6" / f"nb-{digest}" if local else Path(f"{default.drive or 'C:'}/pg6b") / digest
    note = (f"note: default build dir {default} is too long for Windows "
            f"({len(str(default)) + BUILD_SUBPATH_BUDGET} > {limit} chars); using {short}")
    return short, note


def indirect_targets_path(cache_dir: Path | None, env: Mapping[str, str], exe: Path | None) -> Path | None:
    """Where a title's recorded indirect-call targets live: `$PLAYGTA6_INDIRECT_TARGETS` (`off`
    disables), else `<object cache>-targets/<exe stem>-<exe hash>/indirect-targets.txt`. Never in
    the repository (the addresses are derived from the game)."""
    value = env.get(TARGETS_ENV)
    if value:
        return None if value.lower() in ("0", "off", "no") else Path(value).expanduser().resolve()
    if exe is None or not exe.is_file():
        return None
    import build_native_boot  # noqa: PLC0415 (only for the default object cache location)
    obj = (cache_dir if cache_dir is not None else build_native_boot.default_cache_dir(env)).resolve()
    return obj.with_name(obj.name + "-targets") / f"{exe.stem}-{file_digest(exe)[:12]}" / TARGETS_FILE


def _set_marker(marker: Path, on: bool) -> None:
    if on:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("native executor build")
    elif marker.is_file():
        marker.unlink()


def make_plan(repo: Path, build_dir: Path, exe: Path | None, game_dir: Path | None, jobs: int,
              force: bool, harness_override: Path | None,
              rebuild_program: bool = False, cache_dir: Path | None = None,
              lift_cache: Path | None = None, keep: int = LIFT_CACHE_KEEP,
              recomp_cache: Path | None = None, toolchain: str | None = None,
              seed_targets: Path | None = None, native_executor: bool = False,
              probe_functions: Path | None = None, shard_cache: Path | None = None,
              shard_cache_bytes: int = 0) -> tuple[Plan, Path, Path]:
    """Returns the build plan, the program directory and the harness path. Objects go to
    `cache_dir`, else build_native_boot's machine-wide cache (shared by all checkouts). With
    `lift_cache`, a stale program is first looked up there by the hash of its inputs: a hit is
    linked in instead of building the recompiler and running discovery and the lift. With
    `shard_cache`, a lift reuses every unchanged shard from it (`build-program --lift-cache`) and
    trims it to `shard_cache_bytes` (<= 0: never) afterwards."""
    program = build_dir / "program"
    recomp = repo / "target" / "release" / ("playgta6-recomp" + EXE_SUFFIX)
    harness = harness_override or program / "native" / "bin" / ("playgta6_boot" + EXE_SUFFIX)
    plan = Plan()

    def stale(stage: str, target: Path, reasons: list[tuple[str, float]]) -> str | None:
        if force:
            return "forced (--rebuild)"
        t = mtime(target)
        if t is None:
            return f"{target.name} is missing"
        for what, newest in reasons:
            if newest > t:
                return f"{what} is newer than {target.name}"
        return None

    sources = [(d / "Cargo.toml") for d in workspace_dirs(repo)] + [repo / "Cargo.toml", repo / "Cargo.lock"]
    newest_rs = max([newest_mtime(d / "src") for d in workspace_dirs(repo)] + [0.0] + [mtime(p) or 0.0 for p in sources])
    reason = stale("recomp", recomp, [("the recompiler sources", newest_rs)])
    if reason:
        cargo = ["cargo", "build", "--release", "-p", "playgta6-recomp"]
        rkey = None
        if recomp_cache is not None and not force:
            rkey = build_guard.recomp_inputs_key(repo, toolchain or build_guard.rustc_version())
            if mtime(recomp) is not None and build_guard.read_recomp_stamp(recomp) == rkey:
                reason = None  # this exact binary is there (only mtimes moved)
            elif (recomp_cache / rkey / recomp.name).is_file():
                plan.add("recomp", f"prebuilt recompiler cache hit {rkey[:12]} (no cargo build)",
                         ["recomp-cache", "restore", str(recomp_cache / rkey), str(recomp)],
                         run=lambda: build_guard.recomp_cache_restore(recomp_cache, rkey, recomp))
                reason = None
        if reason and rkey is not None:
            plan.add("recomp", reason, cargo,
                     after=lambda: build_guard.recomp_cache_store(recomp_cache, rkey, recomp))
        elif reason:
            plan.add("recomp", reason, cargo)

    manifest = program / "program-manifest.json"
    exe_time = mtime(exe) if exe else 0.0
    # The import bindings come from the runtime's shim headers and export tables: a new guest
    # thunk (e.g. combase!CoCreateInstance) is only bound once the program is generated again.
    shim_time = max([mtime(p) or 0.0 for p in shim_inputs(repo)] + [0.0])
    reason = "forced (--rebuild-program)" if rebuild_program else None
    seed_time = (mtime(seed_targets) or 0.0) if seed_targets else 0.0
    probe_time = (mtime(probe_functions) or 0.0) if probe_functions else 0.0
    reason = reason or stale("program", manifest, [("the recompiler", mtime(recomp) or 0.0), ("the exe", exe_time or 0.0),
                                         ("the runtime shim headers or export tables", shim_time),
                                         ("the recorded targets", seed_time),
                                         ("the probe function list", probe_time)])
    if reason is None and plan.has("recomp"):
        reason = "the recompiler is rebuilt"
    lift_options = list(NATIVE_LIFT_OPTIONS)
    key_options = list(lift_options)
    if seed_targets is not None:
        # The lift depends on the file's content: it is part of the cache key (the report path is not).
        key_options += ["seed-targets", file_digest(seed_targets)]
        lift_options += ["--seed-targets", str(seed_targets), "--seed-report", str(build_dir / "seed-report.jsonl")]
    if probe_functions is not None:
        # Entry counters change the lifted code: the list's content is part of the cache key.
        key_options += ["probe-functions", file_digest(probe_functions)]
        lift_options += ["--probe-functions", str(probe_functions)]
    key = None
    if reason and lift_cache is not None and exe is not None and exe.is_file() and game_dir is not None:
        key = lift_inputs_key(repo, exe, game_dir, key_options)
        if not (force or rebuild_program):
            if read_stamp(program) == key and manifest.is_file():
                reason = None  # the program there is this exact lift (only mtimes moved)
                plan.drop("recomp")
            elif (lift_cache / key / "program-manifest.json").is_file():
                reason = None
                plan.drop("recomp")
                plan.add("program", f"lift cache hit {key[:12]} (no recomp build, discovery or lift)",
                         ["lift-cache", "restore", str(lift_cache / key), str(program)],
                         run=lambda: lift_cache_restore(lift_cache, key, program))
    if reason:
        hooks = {}
        if key is not None:
            hooks = {"before": lambda: clear_program(program)}

        def after() -> None:
            if key is not None:
                lift_cache_store(lift_cache, key, program, keep)
            if shard_cache is not None and shard_cache_bytes > 0:
                trim_shard_cache(shard_cache, shard_cache_bytes)

        if key is not None or shard_cache is not None:
            hooks["after"] = after
        plan.add("program", reason, [str(recomp), "build-program", str(exe), "--out", str(program),
                                     "--dll-dir", str(game_dir), *lift_options, *shard_cache_args(shard_cache),
                                     "--jobs", str(jobs)], **hooks)

    if harness_override is None:
        runtime_newest = newest_mtime(repo / "runtime", [repo / "tools" / "build_native_boot.py"])
        reason = stale("harness", harness, [("the program", mtime(manifest) or 0.0),
                                            ("the runtime sources or build_native_boot.py", runtime_newest)])
        if reason is None and plan.has("program"):
            reason = "the program is rebuilt"
        marker = harness.parent / ".native-executor"
        if reason is None and native_executor and not marker.is_file():
            reason = "the harness was built without --native-executor"
        if reason is None and not native_executor and marker.is_file():
            reason = "the harness was built with --native-executor (rebuilding without)"
        if reason:
            cache = ["--cache-dir", str(cache_dir.resolve())] if cache_dir is not None else []
            plan.add("harness", reason, [sys.executable, str(repo / "tools" / "build_native_boot.py"), str(program),
                                         *cache, "--jobs", str(jobs), "--keep-going", "--delete-ir", "--debug-info",
                                         *(["--native-executor"] if native_executor else [])],
                     after=lambda: _set_marker(marker, native_executor))
    return plan, program, harness


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


# Screens of the menu flow by draws per frame (the live present counters, #663): start screen ~21,
# loading 4, UI TEXT SIZE 20, FINISH SETUP? 30, 23, LEGAL NOTICES 37 (44 with the cursor moved),
# then the list screen reached through LEGAL NOTICES' "Next >": 50 with the cursor up, 33 after Down.
# Replayed (R40 menu-to-ingame run): that list screen is a legal-document viewer (a "PRIVACY POLICY"
# heading, an entry list on the left, hints SCROLL / SELECT (A) / BACK (B)); 51 with a document open.
LIST_SCREEN = "draws_per_frame==33|50"


def _menu_cycle(t: int = 0) -> list[tuple[int, str, int]]:
    """Up x6 (cursor reset), Down, Down, A: LEGAL NOTICES' "Next >" (and each earlier screen's A)."""
    return ([(t + i * 400, "DUp", 100) for i in range(6)]
            + [(t + 3_000, "DDown", 150), (t + 4_500, "DDown", 150), (t + 6_500, "A", INPUT_HOLD_MS)])


def _lines(events: Sequence[tuple[int, str, int]], limit: int | None = None) -> list[str]:
    return [f"{t} {b} {h}" for t, b, h in events if limit is None or t < limit]


def _to_list_screen() -> list[str]:
    """A every 2.5 s from 20 s (intro, start screen), then the menu cycle every 14 s until the list
    screen after LEGAL NOTICES is up and steady (screen-synchronised: a slow run just takes more
    rounds; no press goes out once the list screen is on), then the capture step (only acts with
    --gpu-capture-start-at-script). Later press times count from the moment the list screen held."""
    lines = _lines([(t, "A", INPUT_HOLD_MS) for t in range(20_000, 60_000, 2_500)])
    lines += ["# step list-screen", f"repeat every 14s until {LIST_SCREEN} for 1500ms"]
    lines += ["  " + line for line in _lines(_menu_cycle())]
    lines += ["end", "capture"]
    return lines


def menu_flow_script_text(duration_ms: int) -> str:
    """The menu-flow preset: to the list screen after LEGAL NOTICES (_to_list_screen), then A every 3 s."""
    lines = ["# menu-flow (tools/run_native_boot.py): A every 2.5 s from 20 s; Up x6, Down, Down, A every 14 s",
             "# until the list screen after LEGAL NOTICES (" + LIST_SCREEN + "); then A every 3 s"]
    lines += _to_list_screen()
    lines += _lines([(t, "A", INPUT_HOLD_MS) for t in range(3_000, duration_ms, 3_000)])
    return "\n".join(lines) + "\n"


def menu_flow_2_script_text(duration_ms: int) -> str:
    """The menu-flow-2 preset: to the list screen (_to_list_screen), then the menu cycle every 14 s
    (Up x6 + Down, Down, A: the list screen's third entry; later cycles keep the cursor
    deterministic). Measured (#663): only that cycle left the list screens; Down x0/x1 + A and
    RDown + A did nothing."""
    lines = ["# menu-flow-2 (tools/run_native_boot.py): to the list screen after LEGAL NOTICES", "# ("
             + LIST_SCREEN + "), then every 14 s: Up x6 (cursor reset), Down, Down, A"]
    lines += _to_list_screen()
    events: list[tuple[int, str, int]] = []
    for t in range(2_000, duration_ms, 14_000):
        events += _menu_cycle(t)
    lines += _lines(events, duration_ms)
    return "\n".join(lines) + "\n"


def menu_flow_3_script_text(duration_ms: int) -> str:
    """The menu-flow-3 preset: to the list screen (_to_list_screen), then scroll it (RDown held 1.2 s
    + D-pad Down every 2 s for 100 s), then A every 4 s. Measured (#663): the list screen (50/33
    draws/frame) is the screen that stayed up to the 420 s limit; A only ever reached it in its
    Down (33) state."""
    lines = ["# menu-flow-3 (tools/run_native_boot.py): to the list screen after LEGAL NOTICES (" + LIST_SCREEN + ");",
             "# then RDown + Down every 2 s for 100 s; then A every 4 s"]
    lines += _to_list_screen()
    events: list[tuple[int, str, int]] = []
    for t in range(0, 100_000, 2_000):
        events += [(t, "RDown", 1_200), (t + 1_500, "DDown", 150)]
    events += [(t, "A", INPUT_HOLD_MS) for t in range(102_000, duration_ms, 4_000)]
    lines += _lines(events, duration_ms)
    return "\n".join(lines) + "\n"


# The screen behind the list screen's Up + A (#663: 34 draws/frame after a ~3 s transition).
NEXT_SCREEN = "draws_per_frame==34"


def menu_flow_4_script_text(duration_ms: int) -> str:
    """The menu-flow-4 preset: to the list screen (_to_list_screen), D-pad Up (the cursor's top
    state, 50 draws/frame) and A, then wait for the next screen (34 draws/frame, steady 2 s), then
    the menu cycle every 14 s (untested on that screen). Measured (#663, native, menu-flow-3 with
    waits): A in the list screen's Down state (33) only blipped; Up + A animated for ~3 s into a
    34 draws/frame screen that stayed up through 250 s of RDown + Down and A every 4 s."""
    lines = ["# menu-flow-4 (tools/run_native_boot.py): to the list screen after LEGAL NOTICES (" + LIST_SCREEN + ");",
             "# Up, A; wait for " + NEXT_SCREEN + "; then every 14 s: Up x6, Down, Down, A"]
    lines += _to_list_screen()
    lines += ["1000 DUp 150", "3000 A 100", f"wait {NEXT_SCREEN} for 2s timeout 30s"]
    events: list[tuple[int, str, int]] = []
    for t in range(2_000, duration_ms, 14_000):
        events += _menu_cycle(t)
    lines += _lines(events, duration_ms)
    return "\n".join(lines) + "\n"


# LEGAL NOTICES (Copyright, Photosensitivity Health Warning, "Next >", BACK): 37 draws/frame; A on
# "Copyright" opens the document viewer (the old preset looped there). Down x2 reaches "Next >".
LEGAL_SCREEN = "37"

# The legal-document viewer (the list screen): 50 / 33 draws/frame on its list, 51 with a document open.
VIEWER_SCREEN = "33|50|51"


def _viewer_round() -> list[tuple[int, str, int]]:
    """One round on the legal-document viewer: B (back: leaves the viewer, or closes an open
    document), A (open the entry under the cursor), RDown x3 (scroll the document to its end),
    B (back to the list), D-pad Down (next entry)."""
    return [(0, "B", 150), (3_000, "A", INPUT_HOLD_MS), (5_000, "RDown", 1_200), (7_000, "RDown", 1_200),
            (9_000, "RDown", 1_200), (12_000, "B", 150), (15_000, "DDown", 150)]


# The current boot flow (existing save, with or without a shader pack; tools/screen_flow.py): intro video (13-15
# draws), loading artwork (2 / 4 / 6), UI screens (~9-10), then the 3D scene (38 up to ~520 draws) from ~55-95 s.
# The gate is a RANGE (scene: draws/frame >= SCENE_DRAWS), never one exact count.
SCENE_DRAWS = 38


def menu_to_ingame_script_text(duration_ms: int) -> str:
    """The menu-to-ingame preset (current flow): A at 20 s, then A every 2.5 s only until the 3D scene holds
    (draws/frame >= SCENE_DRAWS for 2 s; the intro videos, loading artwork and UI screens before it need no exact
    count), a settle wait, then the movement round every 12 s until the time limit. Every wait / repeat step is
    named (`# step`), so the digest says which one missed. The fresh-setup flow with its LEGAL NOTICES screens
    is menu-to-ingame-setup (picked automatically by --fresh-setup)."""
    lines = ["# menu-to-ingame (tools/run_native_boot.py, current flow): A from 20 s until the scene (draws/frame >= "
             f"{SCENE_DRAWS}), then the movement round every 12 s",
             "20000 A 100",
             "# step intro-to-scene",
             f"repeat every 2500ms until draws_per_frame>={SCENE_DRAWS} for 2s timeout 180s", "  0 A 100", "end",
             "# step scene-settle",
             f"wait draws_per_frame>={SCENE_DRAWS} for 3s timeout 60s",
             "# step movement-rounds*",
             "repeat every 12s until presents>=999999999 timeout 800s",
             "  0 A 100", "  1500 Up 1500", "  3500 RRight 1200", "  5500 LT 1000", "  7000 Up+RT 1000", "end"]
    return "\n".join(lines) + "\n"


def menu_to_ingame_setup_script_text(duration_ms: int) -> str:
    """The menu-to-ingame-setup preset (fresh setup flow): A (every 14 s) to LEGAL NOTICES (37 draws/frame), then a plain wait until it
    is steady (no A may go out there: a round-start A opened Copyright and the R40 run stayed in the
    viewer to 900 s); B every 3 s only while the screen is not 37 (a document viewer that got opened
    closes); then the menu cycle (Up x6, Down, Down, A on "Next >") every 14 s until the screen after
    it (VIEWER_SCREEN) holds. Measured natively, "Next >" did not react early on LEGAL NOTICES (one
    Down, Down, A ~9 s after it appeared did nothing; the first cycle ~18 s in did nothing, the
    second ~32 s in passed), and 44 draws/frame is LEGAL NOTICES with the cursor moved, so neither a
    single press nor "left 37" counts as having passed it. After that everything is screen-synced: the browser
    preset's skip / confirm round (A hold, Menu, A) until the screen leaves 34 draws/frame, then the
    movement round (A, left stick forward, right stick, LT, RT) every 12 s until the time limit."""
    lines = ["# menu-to-ingame (tools/run_native_boot.py): A to LEGAL NOTICES (" + LEGAL_SCREEN + " draws/frame),",
             "# wait (no A), B if a viewer is open, then Up x6, Down, Down, A on \"Next >\" every 14 s until "
             + VIEWER_SCREEN + ", then screen-synced rounds"]
    lines += _lines([(t, "A", INPUT_HOLD_MS) for t in range(20_000, 60_000, 2_500)])
    lines += ["# step legal-notices", f"repeat every 14s until draws_per_frame=={LEGAL_SCREEN} for 500ms timeout 240s",
              "  0 A 100", "end",
              "# step legal-notices-steady", f"wait draws_per_frame=={LEGAL_SCREEN} for 2s timeout 30s",
              "# step close-viewer", f"repeat every 3s until draws_per_frame=={LEGAL_SCREEN} for 1s timeout 30s",
              "  0 B 150", "end",
              "# step legal-terms-screen", f"repeat every 14s until draws_per_frame=={VIEWER_SCREEN} for 1500ms timeout 240s"]
    lines += ["  " + line for line in _lines(_menu_cycle())]
    lines += ["end", "capture"]
    # Terms screen: one round per 16 s until Confirm has taken (the loading screen after it, TERMS_PASSED, held 3 s).
    # Not "until draws/frame leaves VIEWER_SCREEN": the terms screen's draws/frame jump between ~17 and ~86 while
    # its cursor and document panel animate, and a `repeat` sends no press while its condition holds even for
    # one frame, so that gate dropped the round's checkbox A / Down / Confirm presses (#663).
    lines += ["# step terms-confirm", f"repeat every 16s until {TERMS_PASSED} for 3s timeout 200s"]
    lines += ["  " + line for line in _lines(_terms_round())]
    lines += ["end"]
    lines += ["# step loading-skip", "repeat every 15s until draws_per_frame!=34 for 3s timeout 120s",
              "  0 A 600", "  2500 Menu 150", "  4500 A 100", "end",
              "# step movement-rounds*", "repeat every 12s until presents>=999999999 timeout 800s",
              "  0 A 100", "  1500 Up 1500", "  3500 RRight 1200", "  5500 LT 1000", "  7000 Up+RT 1000", "end"]
    return "\n".join(lines) + "\n"


# After "Confirm >" on the terms screen: a black loading screen (2-3 draws/frame). The terms screen itself never
# drops this low (measured natively: 17 or more draws/frame through its rounds).
TERMS_PASSED = "draws_per_frame<=4"


def _terms_round() -> list[tuple[int, str, int]]:
    """ROCKSTAR GAMES LEGAL TERMS & POLICIES, same timing as the browser chrome-to-ingame round: Up x6 (cursor to
    the top), right stick down (scroll), A (Privacy Policy checkbox), Down, right stick down, A (Terms of Service
    checkbox), Down, A ("Confirm >", enabled once both are checked)."""
    return ([(i * 450, "DUp", 150) for i in range(6)]
            + [(2_700, "RDown", 900), (4_000, "A", 220), (5_200, "DDown", 150), (5_800, "RDown", 900),
               (7_100, "A", 220), (8_300, "DDown", 150), (9_500, "A", 220)])


def input_script_text(duration_ms: int, kind: str = "start-screen") -> str:
    if kind == "menu-to-ingame-setup":
        return menu_to_ingame_setup_script_text(duration_ms)
    if kind == "menu-to-ingame":
        return menu_to_ingame_script_text(duration_ms)
    if kind == "menu-flow-4":
        return menu_flow_4_script_text(duration_ms)
    if kind == "menu-flow-3":
        return menu_flow_3_script_text(duration_ms)
    if kind == "menu-flow-2":
        return menu_flow_2_script_text(duration_ms)
    if kind == "menu-flow":
        return menu_flow_script_text(duration_ms)
    lines = ["# A pressed once a second from 5 s (tools/run_native_boot.py, start-screen preset)"]
    t = INPUT_START_MS
    while t < duration_ms:
        lines.append(f"{t} A {INPUT_HOLD_MS}")
        t += INPUT_PERIOD_MS
    return "\n".join(lines) + "\n"


def harness_command(harness: Path) -> list[str]:
    return [sys.executable, str(harness)] if harness.suffix == ".py" else [str(harness)]


HARD_STOPS_MD = REPO_ROOT / "docs" / "HARD_STOPS.md"


def annotate_report(report: dict, preset: str, name: str, args: Sequence[str], harness: Path,
                    hard_stops: str | None = None) -> bool:
    """Add what the harness cannot know: the `run` block (preset, name, args), the build block from
    build-info.json beside the exe when the harness predates the embedded one, and the HARD_STOPS.md
    status of an unimplemented-slot fault. Returns whether the report changed."""
    before = json.dumps(report, sort_keys=True)
    report["run"] = {"preset": preset, "name": name, "args": list(args)}
    if "build" not in report:
        try:
            report["build"] = json.loads((harness.parent / "build-info.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    slot = guest_symbols.find_unimplemented(report)
    rec = report.get("fault_record")
    if slot and isinstance(rec, dict) and isinstance(rec.get("unimplemented_slot"), dict) and hard_stops:
        status = guest_symbols.hard_stop_status(hard_stops, slot["iface"], slot["slot"], slot.get("name"))
        if status:
            rec["unimplemented_slot"]["hard_stop"] = status
    return json.dumps(report, sort_keys=True) != before


def load_report(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def timeline_presents(report: dict) -> list:
    """Capture samples, else the runtime's own per-present samples (R40: no GPU capture needed)."""
    return report.get("presents") or screen_timeline.presents_from_samples(report.get("present_samples") or [])


def audio_line(report: dict) -> str | None:
    """One-line summary of the report's `audio` section (None when the report has none)."""
    a = report.get("audio")
    if not isinstance(a, dict):
        return None
    windows = a.get("non_silent_frames_per_window") or []
    live = sum(1 for w in windows if w)
    fraction = a.get("non_silent_fraction") or 0
    return (f"audio: {a.get('render_frames', 0)} frames, {fraction * 100:.1f}% non-silent "
            f"({live}/{len(windows)} windows), spatial passes {a.get('spatial_passes', 0)}, "
            f"xapu decoded {a.get('xapu_decoded_packets', 0)} / failed {a.get('xapu_failed_packets', 0)} "
            f"/ results {a.get('xapu_results_posted', 0)}"
            + (f" / unhandled commands {a['xapu_unhandled_commands']}" if a.get('xapu_unhandled_commands') else ""))


def digest(report: dict, exit_code: int, max_counters: int = 12) -> str:
    lines = [f"status: {report.get('status')} (phase {report.get('phase')}, exit_code {report.get('exit_code')}, "
             f"harness exit {exit_code}), {report.get('elapsed_ms')} ms"]
    if report.get("message"):
        lines.append(f"message: {report['message']}")
    lines.append("milestones:")
    for m in report.get("milestones") or []:
        seq = f", import call {m['import_seq']}" if "import_seq" in m else ""
        lines.append(f"  {m.get('milestone')}: {m.get('elapsed_ms')} ms{seq}")
    presents = report.get("presents") or []
    fp = report.get("first_present")
    first = f", first at import call {fp.get('import_seq')} ({fp.get('elapsed_ms')} ms)" if isinstance(fp, dict) else ""
    if presents:
        lines.append(f"presents: {len(presents)}{first}")
    else:
        # No capture, so no per-Present samples: the present pacer's count (not "0 presents").
        paced = (report.get("present_pacing") or {}).get("presents")
        count = f"{paced} (paced, no capture samples)" if paced else "0"
        lines.append(f"presents: {count}{first}")
    if presents:
        last = presents[-1]
        lines.append(f"  last: wall {last.get('wall_ms')} ms, dispatches {last.get('dispatches')}, "
                     f"import calls {last.get('import_calls')}")
        counters = {k: v for k, v in (last.get("counters") or {}).items()
                    if re.search(r"draw|dispatch|upload|copy|clear", k) and v}
        for k, v in list(counters.items())[:max_counters]:
            lines.append(f"  {k}: {v}")
    audio = audio_line(report)
    if audio:
        lines.append(audio)
    fault = report.get("fault")
    if isinstance(fault, dict):
        func = (fault.get("function") or {}).get("entry")
        lines.append(f"stop: {fault.get('kind')} at {fault.get('va')} (function {func})")
        for fr in ((report.get("backtrace") or {}).get("frames") or [])[:4]:
            lines.append(f"  from {fr.get('function')} +{fr.get('offset')}")
    else:
        lines.append("stop: no fault")
    lines.append(f"import calls: {(report.get('counters') or {}).get('import_calls')}")
    if report.get("screen_timeline"):
        lines += screen_timeline.format_table(report["screen_timeline"])
    return "\n".join(lines)


NL = chr(10)
THUMB_MIN_SEGMENT_MS = 1000.0


def thumbnail_frames(segs: Sequence[dict], min_ms: float = THUMB_MIN_SEGMENT_MS,
                     total_presents: int | None = None, first_present: int = 0) -> list[tuple[int, int, int, float]]:
    """(segment number, trace frame, draws/frame, start seconds) of every segment spanning more than
    `min_ms` (first to last sampled present), at its middle present. A late-started capture's frame
    i is present `first_present` + i (the sidecar's first_present); `total_presents` bounds the
    run's present index."""
    out = []
    for n, s in enumerate(segs):
        if float(s["end_time"]) - float(s["start_time"]) <= min_ms:
            continue
        lo = max(int(s["first_present"]), first_present)
        hi = int(s["first_present"]) + int(s["presents"]) - 1
        if lo > hi:
            continue
        present = lo + (hi - lo) // 2
        if total_presents is not None and present >= total_presents:
            continue
        out.append((n, present - first_present, int(s["draws_per_frame"]), float(s["start_time"]) / 1000.0))
    return out


def capture_first_present(run_dir: Path) -> int:
    """The sidecar's first_present (0 for a capture from boot or an older sidecar)."""
    try:
        side = json.loads((run_dir / "capture.pg6trace.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    return int(side.get("first_present") or 0) if isinstance(side, dict) else 0


INPUT_LOG = re.compile(r"^\[input\] (\d+) ms: (.*)$", re.M)


def logged_input_times(log_text: str) -> list[float]:
    """Press times (ms) the harness logged ("[input] <ms> ms: press ..."): the real times of a
    screen-synchronised script, whose own times are relative to its waits."""
    return [float(m.group(1)) for m in INPUT_LOG.finditer(log_text) if m.group(2).startswith("press ")]


def input_wait_line(log_text: str) -> str | None:
    """One digest line on the script's wait / repeat steps, or None without any."""
    ends = [(int(m.group(1)), m.group(2)) for m in INPUT_LOG.finditer(log_text)
            if m.group(2).endswith(")") and (" met (" in m.group(2) or " timed out (" in m.group(2))]
    if not ends:
        return None
    parts = [f"{what.split(' (presents')[0]} @ {ms / 1000:.1f} s" for ms, what in ends]
    return "input waits: " + "; ".join(parts)


def thumbnail_name(seg: int, start_s: float, draws: int) -> str:
    return f"seg{seg:02d}_{start_s:.0f}s_{draws}d.png"


def replay_command(chrome: str, run_dir: Path, pack: str, frames: Sequence[int], out_prefix: Path,
                   timeout: int = 900) -> list[str]:
    return [sys.executable, str(REPO_ROOT / "tools" / "replay_capture.py"), "--chrome", chrome,
            "--replay-dir", str(run_dir), "--trace", "capture.pg6trace", "--pack", pack,
            "--thumbat", ",".join(str(f) for f in frames), "--out", str(out_prefix),
            "--timeout", str(timeout)]


def make_thumbnails(args, run_dir: Path, report: dict, env: Mapping[str, str],
                    call: Callable[[list[str]], int] | None = None) -> list[str]:
    """Replay the capture and save one image per screen segment into <run>/thumbs; returns digest lines."""
    capture = run_dir / "capture.pg6trace"
    if not capture.is_file() or not args.gpu_capture_payloads:
        return ["thumbnails: need a GPU capture with payloads (--gpu-capture-frames N --gpu-capture-payloads, "
                "ideally --gpu-capture-thin-uploads N); none made"]
    chrome = find_chrome.resolve_chrome(args.chrome, env)
    pack = args.replay_pack
    if not chrome or not pack:
        return ["thumbnails: no Chrome found (pass --chrome or set $PLAYGTA6_CHROME) and pass --replay-pack (the wgsl pack)"]
    picks = thumbnail_frames(report.get("screen_timeline") or [], total_presents=len(report.get("presents") or []) or None,
                             first_present=capture_first_present(run_dir))
    if not picks:
        return ["thumbnails: no screen segment longer than 1 s within the captured frames"]
    thumbs = run_dir / "thumbs"
    thumbs.mkdir(exist_ok=True)
    pack_path = Path(pack)
    if pack_path.parent.resolve() != run_dir.resolve():
        shutil.copyfile(pack_path, run_dir / pack_path.name)
    cmd = replay_command(chrome, run_dir, pack_path.name, [p[1] for p in picks], thumbs / "replay")
    code = (call or (lambda c: subprocess.call(c)))(cmd)  # replay_capture always closes Chrome
    lines = [f"thumbnails: {len(picks)} segment(s), replay exit {code}"]
    for seg, frame, draws, start_s in picks:
        src = thumbs / f"replay-swap{frame}.png"
        dst = thumbs / thumbnail_name(seg, start_s, draws)
        if src.is_file():
            src.replace(dst)
            lines.append(f"  {dst}")
        else:
            lines.append(f"  segment {seg}: no image for frame {frame}")
    return lines


def shader_pack_hint(report: dict, shader_pack: Path | None) -> str | None:
    """One line when no shader pack was given and most draws were dropped for want of a shader module."""
    if shader_pack:
        return None
    work = report.get("gpu_work")
    if not isinstance(work, dict):
        return None
    dropped = work.get("draws_no_shader_module")
    if not isinstance(dropped, int) or dropped <= 0:
        return None
    total = work.get("draws_issued")
    if not isinstance(total, int) or total < dropped:
        total = dropped + (work.get("draws_emitted") if isinstance(work.get("draws_emitted"), int) else 0)
    if dropped * 2 <= total:
        return None
    return (f"hint: no shader pack: {dropped} of {total} draws dropped (draws_no_shader_module); build one with "
            "`cargo run --release -p playgta6-shaders -- pack-rdna2 <input-dir> --out <out.wgsl.pack>` "
            "and pass --shader-pack")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rebuild what is stale, run playgta6_boot with a preset or explicit options and keep the run. "
        "Options after `--` go to playgta6_boot unchanged. See the module docstring.",
        epilog="Exit code: 0 ok, 1 baseline differences, 2 usage or build error, 3 no report.",
    )
    p.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    p.add_argument("--name", help="run name (default <preset>-<timestamp>)")
    p.add_argument("--exe", type=Path, help="game executable (default $PLAYGTA6_EXE)")
    p.add_argument("--game-dir", type=Path, help="package directory (default $PLAYGTA6_GAME_DIR)")
    p.add_argument("--build-dir", type=Path, help="build outputs (default work/native-boot)")
    p.add_argument("--runs-dir", type=Path, help="run directories (default <build-dir>/runs)")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    p.add_argument("--cache-dir", type=Path,
                   help="object cache (default: build_native_boot's machine-wide cache, $PLAYGTA6_OBJ_CACHE)")
    p.add_argument("--rebuild", action="store_true", help="rebuild every stage")
    p.add_argument("--rebuild-program", action="store_true",
                   help="regenerate the lifted program even when the stamps say it is current")
    p.add_argument("--build-only", action="store_true", help="run the (timed) build stages, then stop before the run")
    p.add_argument("--no-build", action="store_true", help="fail instead of building a stale stage")
    p.add_argument("--trust-build", action="store_true",
                   help="skip the staleness checks and use the existing program and harness as they are")
    p.add_argument("--dry-run", action="store_true", help="print the build plan and the command, then stop")
    p.add_argument("--harness", type=Path, help="use this harness instead of the built one (no harness build)")
    p.add_argument("--max-ms", type=int, help="harness --max-ms (default from the preset)")
    p.add_argument("--watchdog-ms", type=int, help="harness --watchdog-ms (default from the preset)")
    p.add_argument("--input-script", type=Path, help="input script file (overrides the preset's)")
    p.add_argument("--shader-pack", type=Path, help="WGPK shader pack")
    p.add_argument("--gpu-capture-frames", type=int, metavar="N", help="capture N frames into the run directory")
    p.add_argument("--gpu-capture-payloads", action="store_true", help="keep payloads in the capture")
    p.add_argument("--gpu-capture-thin-uploads", type=int, metavar="N",
                   help="with payloads: keep only every Nth write of a texture/buffer > 4 MiB")
    p.add_argument("--gpu-capture-start-ms", type=int, metavar="N",
                   help="start the captured frames N ms after boot (objects and the newest writes are kept)")
    p.add_argument("--gpu-capture-start-at-script", action="store_true",
                   help="start the captured frames at the input script's `capture` step (the menu-flow presets: the list screen)")
    p.add_argument("--thumbnails", action="store_true",
                   help="after a payload GPU capture, replay it in Chrome and save one image per screen segment")
    p.add_argument("--chrome", help="Chrome executable for --thumbnails (default $PLAYGTA6_CHROME, then auto-detected)")
    p.add_argument("--replay-pack", help="wgsl pack file for --thumbnails")
    p.add_argument("--import-trace", action="store_true", help="write import-trace.txt")
    p.add_argument("--seed-targets", nargs="?", const="auto", metavar="FILE",
                   help="lift with the recorded indirect-call targets as function entries (default file: this "
                        "title's recorded targets, see PLAYGTA6_INDIRECT_TARGETS)")
    p.add_argument("--probe-functions", type=Path, metavar="FILE",
                   help="lift with an entry counter in each function whose hex VA is listed in FILE (one per line; "
                        "keep the file outside git), so direct calls show in the boot report's audio_probes")
    p.add_argument("--no-record-targets", action="store_true",
                   help="do not record the indirect-call targets this run dispatches")
    p.add_argument("--dump-pipelines", help="harness --dump-pipelines (a directory outside the repo, or under work/)")
    p.add_argument("--audio-dump", help="harness --audio-dump: WAV file of the stereo mix")
    p.add_argument("--baseline", help="earlier run (name in the runs directory, or a directory) to compare with")
    p.add_argument("--ignore", action="append", default=[], metavar="PATTERN", help="compare_boot_reports --ignore")
    p.add_argument("--time-tolerance", type=float, metavar="REL")
    p.add_argument("--count-tolerance", type=float, metavar="REL")
    p.add_argument("--native-executor", action="store_true",
                   help="build the harness with --window support (Dawn executor; needs a Dawn checkout/cache)")
    p.add_argument("--live-executor", action="store_true",
                   help="headless live run: --native-executor build, then run with --live-executor (the live Dawn "
                        "executor presenting into an offscreen swapchain texture: --live-digest, --screenshot-at, "
                        "--gpu-validation, gpu_time all work, no window opens). A `--window` in the extra options is "
                        "turned into --live-executor unless --show-window is given")
    p.add_argument("--show-window", action="store_true",
                   help="keep an extra-option `--window` as a real window (agents and diagnostics default to headless)")
    p.add_argument("--play", action="store_true",
                   help="live native play: --native-executor build, then run with --window --audio-out --host-input")
    p.add_argument("--record-input", nargs="?", const="auto", metavar="FILE",
                   help="with --play: record the real keyboard/pad input as an --input-script file (replay it with "
                        "--input-script; deterministic by present index). Without FILE: "
                        "%%LOCALAPPDATA%%/playgta6/recordings/<name>.txt")
    p.add_argument("--keymap", metavar="FILE",
                   help="with --play: key/mouse rebinding file (default: %%LOCALAPPDATA%%/playgta6/keymap.ini, "
                        "written with the defaults on first run)")
    p.add_argument("--fresh-setup", action="store_true",
                   help="with --play: a new empty save directory, so the setup screens appear again")
    p.add_argument("--save-dir", type=Path, help="with --play: save directory (default: persistent, see docs)")
    p.add_argument("--present-mode", choices=["auto", "fifo", "mailbox", "immediate"],
                   help="with --play: window present mode (default auto: mailbox, immediate, fifo; unsupported falls back)")
    p.add_argument("--present-interval", choices=["honor", "min1"], help="title-driven present gap: honor = the title's sync interval (console timing), min1 = 1 vblank (default)")
    p.add_argument("--audio-mmcss", choices=["pro-audio", "audio", "off"],
                   help="with --play: WASAPI render-thread MMCSS class (default: harness default pro-audio; env PLAYGTA6_AUDIO_MMCSS)")
    p.add_argument("--max-fps", type=int, help="with --play: host-side present cap (0 = off); guest pacing is unchanged")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="after `--`: further playgta6_boot options")
    return p


# kRelaunchSelfExitCode (runtime/os/win32_compat.h, ASCII "RELN"): the title called XLaunchNewGame on
# itself. The guest stopped; the harness starts it again with the same command line.
RELAUNCH_SELF_EXIT_CODE = 0x52454C4E
MAX_RELAUNCHES = 3


def is_relaunch_request(report: dict | None) -> bool:
    return bool(report) and report.get("exit_code") == RELAUNCH_SELF_EXIT_CODE


def run_with_relaunch(cmd: list[str], repo: Path, run_dir: Path, runner: Runner,
                      timeout: float) -> tuple[int, dict | None, int]:
    """Runs the harness; when the title relaunched itself, archives that launch's report and log as
    boot-report.launchN.json / boot.launchN.log and runs it again (at most MAX_RELAUNCHES times)."""
    relaunches = 0
    while True:
        code = runner(cmd, repo, run_dir / "boot.log", timeout)
        report = load_report(run_dir / "boot-report.json")
        if not is_relaunch_request(report) or relaunches >= MAX_RELAUNCHES:
            return code, report, relaunches
        relaunches += 1
        for name, archived in (("boot-report.json", f"boot-report.launch{relaunches}.json"),
                               ("boot.log", f"boot.launch{relaunches}.log"),
                               ("summary.txt", f"summary.launch{relaunches}.txt")):
            if (run_dir / name).is_file():
                (run_dir / name).replace(run_dir / archived)
        print(f"[run] the title relaunched itself (XLaunchNewGame); relaunch {relaunches} of at most "
              f"{MAX_RELAUNCHES} with the same arguments", file=sys.stderr, flush=True)


def resolve_watchdog_ms(preset, max_ms_override, watchdog_ms_override):
    """Hard watchdog: explicit value wins; otherwise keep the preset's margin above the effective max-ms so a
    longer --max-ms is not cut short by the preset's (smaller) watchdog."""
    if watchdog_ms_override is not None:
        return watchdog_ms_override
    if max_ms_override is None:
        return preset["watchdog_ms"]
    return max(preset["watchdog_ms"], max_ms_override + (preset["watchdog_ms"] - preset["max_ms"]))


def main(argv: Sequence[str] | None = None, runner: Runner = run_command,
         env: Mapping[str, str] | None = None, repo_root: Path = REPO_ROOT) -> int:
    env = os.environ if env is None else env
    try:
        args = build_arg_parser().parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code in (0, None) else 2
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    if env is os.environ and runner is run_command:  # real runs only: free disk under PLAYGTA6_MIN_FREE_GB trims the caches first
        import trim_caches  # noqa: PLC0415
        import cache_root  # noqa: PLC0415
        cache_root.legacy_hint(env, lambda m: print(m, file=sys.stderr, flush=True),
                               explicit=getattr(args, "cache_dir", None) is not None)
        trim_caches.auto_trim_if_low(env, repo_root, extra_objs=[args.cache_dir] if getattr(args, "cache_dir", None) else [])
    try:
        return _run(args, extra, runner, env, repo_root)
    except RunnerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run(args, extra: list[str], runner: Runner, env: Mapping[str, str], repo: Path) -> int:
    exe = args.exe or (Path(env["PLAYGTA6_EXE"]) if env.get("PLAYGTA6_EXE") else None)
    game_dir = args.game_dir or (Path(env["PLAYGTA6_GAME_DIR"]) if env.get("PLAYGTA6_GAME_DIR") else None)
    if exe is None or game_dir is None:
        missing = ["--exe (or $PLAYGTA6_EXE)" if exe is None else "",
                   "--game-dir (or $PLAYGTA6_GAME_DIR)" if game_dir is None else ""]
        raise RunnerError("missing game paths: pass " + " and ".join(m for m in missing if m)
                          + " (they are never read from the repository)")
    if not args.dry_run and not exe.is_file():
        raise RunnerError(f"exe not found: {exe}")
    if not args.dry_run and not game_dir.is_dir():
        raise RunnerError(f"game directory not found: {game_dir}")
    build_dir, note = choose_build_dir(repo, args.build_dir, env)
    if note:
        print(note, file=sys.stderr)
    runs_dir = (args.runs_dir or prebuilt_guard.default_runs_dir(build_dir, env) or build_dir / "runs").resolve()
    if prebuilt_guard.containing_prebuilt(build_dir) is not None:
        if prebuilt_guard.containing_prebuilt(runs_dir) is not None:
            raise RunnerError(f"--runs-dir {runs_dir} is inside the prebuilt build "
                              f"{prebuilt_guard.containing_prebuilt(build_dir)}; a prebuilt stays read-only")
        print(f"[prebuilt] runs go to {runs_dir} (the prebuilt build is never written)", file=sys.stderr)
    preset = PRESETS[args.preset]
    name = args.name or f"{args.preset}-{time.strftime('%Y%m%d-%H%M%S')}"
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise RunnerError(f"invalid run name {name!r}")
    run_dir = runs_dir / name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RunnerError(f"run directory {run_dir} already exists (pick another --name)")

    lift_cache = None if (args.dry_run or args.trust_build) else lift_cache_dir(args.cache_dir, env)
    recomp_cache = None if (args.dry_run or args.trust_build) else build_guard.recomp_cache_dir(args.cache_dir, env)
    if args.dry_run:
        return _run_locked(args, extra, runner, env, repo, exe, game_dir, build_dir, runs_dir, preset, run_dir,
                           lift_cache, recomp_cache)
    try:
        with build_guard.build_dir_lock(build_dir, lock_dir=prebuilt_guard.lock_dir(build_dir, env)):
            return _run_locked(args, extra, runner, env, repo, exe, game_dir, build_dir, runs_dir, preset,
                               run_dir, lift_cache, recomp_cache)
    except build_guard.BuildDirBusy as exc:
        raise RunnerError(str(exc)) from exc


def resolve_live_mode(args, extra: list[str]) -> tuple[list[str], bool]:
    """(extra options, needs the native-executor build). Only --play (a person playing) and --show-window
    open a window; every other live run (--live-executor, or a `--window` in the extra options) is headless:
    the same LivePump + executor + present, into the executor's offscreen swapchain texture."""
    extra = list(extra)
    if not args.play and not args.show_window and "--window" in extra:
        extra = ["--live-executor" if a == "--window" else a for a in extra]
        print("[play] --window in the extra options runs headless as --live-executor (--show-window keeps the window)",
              file=sys.stderr)
    if args.live_executor and "--live-executor" not in extra and "--window" not in extra and not args.play:
        extra.append("--live-executor")
    live = args.play or args.live_executor or any(a in ("--window", "--live-executor") for a in extra)
    return extra, live


def _run_locked(args, extra, runner, env, repo, exe, game_dir, build_dir, runs_dir, preset, run_dir,
                lift_cache, recomp_cache) -> int:
    extra, live_run = resolve_live_mode(args, extra)
    targets_file = None if args.no_record_targets and not args.seed_targets else         indirect_targets_path(args.cache_dir, env, exe)
    seed_file = None
    if args.seed_targets:
        seed_file = targets_file if args.seed_targets == "auto" else Path(args.seed_targets)
        if seed_file is None or not seed_file.is_file():
            raise RunnerError(f"--seed-targets: no recorded targets at {seed_file} (run the title once first)")
    plan, program, harness = make_plan(repo, build_dir, exe, game_dir, args.jobs, args.rebuild, args.harness,
                                      args.rebuild_program, args.cache_dir, lift_cache,
                                      lift_cache_keep(env), recomp_cache, seed_targets=seed_file,
                                      native_executor=args.native_executor or args.play or live_run,
                                      probe_functions=args.probe_functions,
                                      shard_cache=None if args.dry_run else shard_cache_dir(lift_cache, env),
                                      shard_cache_bytes=shard_cache_max_bytes(env))
    if args.trust_build:
        plan.steps = []
    if args.play:
        if not args.shader_pack and not args.build_only:
            found, why = find_shader_pack(repo, env)
            if found is None:
                raise RunnerError(f"no shader pack ({why}): pass --shader-pack <std.wgsl.pack>, set "
                                  f"{SHADER_PACK_ENV}, or put one under work/ (or ${SHADER_PACK_DIR_ENV}); "
                                  "without one nothing is drawn")
            args.shader_pack = found
            print(f"[play] shader pack: {found} ({why})", file=sys.stderr)
        if args.record_input:
            rec = (default_recordings_dir(env) / f"{args.name or 'play'}.txt") if args.record_input == "auto" else Path(args.record_input)
            extra = ["--record-input", str(rec)] + list(extra)
            print(f"[play] recording input to {rec} (replay: --input-script)", file=sys.stderr)
        if args.keymap:
            extra = ["--keymap", args.keymap] + list(extra)
        if "--save-dir" not in extra:
            save = play_save_dir(args.fresh_setup, args.save_dir, env, time.strftime("%Y%m%d-%H%M%S"))
            extra = ["--save-dir", str(save)] + list(extra)
            print(f"[play] save dir: {save}" + (" (fresh: setup screens appear)" if args.fresh_setup else ""),
                  file=sys.stderr)
    max_ms = args.max_ms if args.max_ms is not None else preset["max_ms"]
    watchdog_ms = resolve_watchdog_ms(preset, args.max_ms, args.watchdog_ms)

    cmd = harness_command(harness)
    cmd += ["--program-dir", str(program), "--name", exe.name, "--game-dir", str(game_dir),
            "--max-ms", str(max_ms), "--watchdog-ms", str(watchdog_ms),
            "--report", str(run_dir / "boot-report.json"), "--summary", str(run_dir / "summary.txt")]
    script_path = run_dir / "input-script.txt"
    if args.input_script:
        cmd += ["--input-script", str(args.input_script)]
    elif preset["input"]:
        cmd += ["--input-script", str(script_path)]
    if args.shader_pack:
        cmd += ["--shader-pack", str(args.shader_pack)]
    if args.gpu_capture_frames is not None or args.gpu_capture_payloads:
        cmd += ["--gpu-capture", str(run_dir / "capture.pg6trace"),
                "--gpu-capture-frames", str(args.gpu_capture_frames or 10)]
        if args.gpu_capture_payloads:
            cmd += ["--gpu-capture-payloads"]
            if args.gpu_capture_thin_uploads:
                cmd += ["--gpu-capture-thin-uploads", str(args.gpu_capture_thin_uploads)]
        if args.gpu_capture_start_ms:
            cmd += ["--gpu-capture-start-ms", str(args.gpu_capture_start_ms)]
        if args.gpu_capture_start_at_script:
            cmd += ["--gpu-capture-start-at-script"]
    if args.import_trace:
        cmd += ["--import-trace", str(run_dir / "import-trace.txt")]
    if targets_file is not None and not args.no_record_targets:
        cmd += ["--record-indirect-targets", str(targets_file)]
    if args.dump_pipelines:
        cmd += ["--dump-pipelines", args.dump_pipelines]
    if args.audio_dump:
        cmd += ["--audio-dump", args.audio_dump]
    if args.play:
        cmd += ["--window", "--audio-out", "--host-input"]
        if args.present_mode:
            cmd += ["--present-mode", args.present_mode]
        if args.audio_mmcss:
            cmd += ["--audio-mmcss", args.audio_mmcss]
        if args.present_interval:
            cmd += ["--present-interval", args.present_interval]
        if args.max_fps:
            cmd += ["--max-fps", str(args.max_fps)]
    cmd += extra

    for step in plan.steps:
        print(f"[plan] {step['stage']}: {step['reason']}", file=sys.stderr)
    if args.dry_run:
        print(json.dumps({"plan": plan.steps, "command": cmd}, indent=2))
        return 0
    if targets_file is not None and not args.no_record_targets:
        targets_file.parent.mkdir(parents=True, exist_ok=True)
    if plan.steps and args.no_build:
        raise RunnerError("stale stages with --no-build: " + ", ".join(s["stage"] for s in plan.steps))
    if args.harness is not None and not args.harness.is_file():
        raise RunnerError(f"harness not found: {args.harness}")
    for step in plan.steps:
        print(f"[build] {step['stage']}: {' '.join(step['cmd'])}", file=sys.stderr, flush=True)
        hooks = plan.hooks.get(step["stage"], {})
        started = time.monotonic()
        try:
            if "run" in hooks:
                hooks["run"]()
            else:
                if "before" in hooks:
                    hooks["before"]()
                if runner(step["cmd"], repo, None, None) != 0:
                    raise RunnerError(f"{step['stage']} failed")
                if "after" in hooks:
                    hooks["after"]()
        except OSError as exc:
            raise RunnerError(f"{step['stage']} failed: {exc}") from exc
        print(f"[build] {step['stage']}: done in {time.monotonic() - started:.0f} s", file=sys.stderr, flush=True)
    if not harness.is_file():
        raise RunnerError(f"harness not found after the build: {harness}")

    if args.build_only:
        print(f"[build] done; harness {harness}", file=sys.stderr, flush=True)
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    if prebuilt_guard.containing_prebuilt(build_dir) is None:
        clean_build_dirs.write_marker(build_dir, "run_native_boot.py", repo)
    if preset["input"] and not args.input_script:
        kind = preset["input"]
        if kind == "menu-to-ingame" and args.fresh_setup:
            kind = "menu-to-ingame-setup"  # the setup screens (legal notices) only appear on a fresh save
        script_path.write_text(input_script_text(max_ms, kind), encoding="utf-8")
    print(f"[run] {run_dir}", file=sys.stderr, flush=True)
    if args.play and not args.dry_run:
        # The window banner (runtime/guest/stop_message.h) tells a pending owner decision from a crash with this list.
        os.environ["PLAYGTA6_HARD_STOP_SLOTS"] = stop_message.pending_arg(hard_stops_text())
    watch_done = threading.Event()
    if args.play and not args.dry_run:
        threading.Thread(target=watch_for_fault, args=(run_dir, watch_done), daemon=True).start()
    try:
        code, report, relaunches = run_with_relaunch(cmd, repo, run_dir, runner, watchdog_ms / 1000 + 60)
    finally:
        watch_done.set()
    (run_dir / "command.json").write_text(json.dumps({
        "preset": args.preset, "command": cmd, "plan": plan.steps, "harness_exit": code,
        "relaunches": relaunches}, indent=2), encoding="utf-8")
    if report is None:
        print(f"error: the harness (exit {code}) wrote no report; see {run_dir / 'boot.log'}", file=sys.stderr)
        return 3
    try:
        hs_md = HARD_STOPS_MD.read_text(encoding="utf-8") if HARD_STOPS_MD.is_file() else None
    except OSError:
        hs_md = None
    if annotate_report(report, args.preset, run_dir.name, cmd[len(harness_command(harness)):], harness, hs_md):
        (run_dir / "boot-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    presents = timeline_presents(report)
    log_text = ""
    try:
        log_text = (run_dir / "boot.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    if presents and "screen_timeline" not in report:
        times: list[float] = logged_input_times(log_text)
        ipath = Path(args.input_script) if args.input_script else script_path
        if not times and ipath.is_file():
            times = screen_timeline.parse_input_times(ipath.read_text(encoding="utf-8"))
        report["screen_timeline"] = screen_timeline.build_timeline(presents, times)
        (run_dir / "boot-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    text = digest(report, code)
    waits = input_wait_line(log_text)
    if waits:
        text += NL + waits
    ipath = Path(args.input_script) if args.input_script else script_path
    if ipath.is_file():
        samples = report.get("present_samples") or []
        missed = screen_flow.timeout_lines(
            log_text, ipath.read_text(encoding="utf-8"),
            last_draws=int(samples[-1].get("draws") or 0) if samples else None,
            end_ms=report.get("elapsed_ms"))
        if missed:
            text += NL + NL.join(missed)
    if report.get("present_samples"):
        text += NL + screen_flow.render(screen_flow.detect_from_report(report)).replace("\n", NL)
    (run_dir / "digest.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    try:
        print(NL + "summary (tools/boot_summary.py):" + NL + boot_summary.render_path(run_dir))
    except (OSError, ValueError) as exc:
        print(f"[run] boot_summary failed: {exc}", file=sys.stderr)
    stop_line = stop_message.select(report, hs_md, run_dir)
    if stop_line:
        stop_message.emit(NL + stop_line)
    hint = shader_pack_hint(report, args.shader_pack)
    if hint:
        print(hint)
    if args.thumbnails:
        tlines = make_thumbnails(args, run_dir, report, env)
        with (run_dir / "digest.txt").open("a", encoding="utf-8") as f:
            f.write(NL.join(tlines) + NL)
        print(NL.join(tlines))
    if args.baseline:
        return compare_with_baseline(args, run_dir, runs_dir)
    return 0


def compare_with_baseline(args, run_dir: Path, runs_dir: Path) -> int:
    base = Path(args.baseline)
    if not base.is_dir():
        base = runs_dir / args.baseline
    if not (base / "boot-report.json").is_file():
        raise RunnerError(f"baseline run not found: {args.baseline}")
    argv = [str(base / "boot-report.json"), str(run_dir / "boot-report.json")]
    for sub, flag in (("capture.pg6trace", "--capture"), ("import-trace.txt", "--import-trace")):
        if (base / sub).is_file() and (run_dir / sub).is_file():
            argv += [flag, str(base / sub), str(run_dir / sub)]
    if "--import-trace" in argv and "--capture" not in argv:
        argv = argv[:argv.index("--import-trace")]
    for pattern in args.ignore:
        argv += ["--ignore", pattern]
    if args.time_tolerance is not None:
        argv += ["--time-tolerance", str(args.time_tolerance)]
    if args.count_tolerance is not None:
        argv += ["--count-tolerance", str(args.count_tolerance)]
    print(f"[compare] against {base}")
    code = compare_boot_reports.main(argv)
    return 1 if code == 1 else (2 if code == 2 else 0)


if __name__ == "__main__":
    raise SystemExit(main())
