#!/usr/bin/env python3
"""One-command native boot runner: rebuild what is stale, run `playgta6_boot`, keep the run.

    python tools/run_native_boot.py [--preset smoke|start-screen|menu-flow|long] [--name NAME]
        [--exe PATH] [--game-dir DIR] [options] [-- extra playgta6_boot options]

What it does, in order:

1. Staleness (mtime based, cheap): the recompiler (`target/release/playgta6-recomp`) is rebuilt
   with `cargo build --release -p playgta6-recomp` when missing or older than the sources of the
   workspace crates; the program (`<build-dir>/program`, `playgta6-recomp build-program <exe>
   --out ... --dll-dir <game-dir> --target native`) when its `program-manifest.json` is missing
   or older than the recompiler, the exe or the runtime files it binds imports from
   (`runtime/os/*.h|*.cpp`, `runtime/gpu/d3d12_xs.h`, `d3d12_xs_exports.*`,
   `runtime/guest/guest_callbacks.h`); the harness (`python tools/build_native_boot.py
   <build-dir>/program --cache-dir <build-dir>/obj-cache --keep-going --delete-ir --debug-info`;
   the PDB lets `--profile` name runtime functions) when the binary is missing or older than the
   program manifest, any `runtime/` source or `tools/build_native_boot.py`. Only stale stages run, in dependency order (a rebuilt stage makes
   the later ones stale). `--rebuild` forces all, `--no-build` fails instead of building,
   `--dry-run` prints the plan and stops, `--trust-build` skips the checks and uses the existing
   program and harness as they are (for a build made elsewhere, e.g. from another checkout).
2. Run `playgta6_boot --program-dir <program> --name <exe name> --game-dir <game-dir>` with a
   preset or explicit options. Presets: `smoke` (60 s limit, 90 s watchdog), `start-screen`
   (120 s limit; an input script pressing A for 100 ms once a second from 5 s on), `menu-flow`
   (240 s limit; A every 2.5 s from 20 s, then D-pad Down, Down, A through LEGAL NOTICES), `long`
   (600 s limit, 630 s watchdog). `--max-ms`, `--watchdog-ms`, `--input-script FILE`,
   `--shader-pack FILE`, `--gpu-capture-frames N` (captures into the run directory),
   `--gpu-capture-payloads`, `--gpu-capture-thin-uploads N`, `--dump-pipelines DIR` and everything after `--` are passed through.
3. The run lands in `<runs-dir>/<name>/`: `boot-report.json`, `summary.txt` (the harness text
   summary), `boot.log` (stdout and stderr), `capture.pg6trace` (+ `.json`, with a GPU capture),
   `input-script.txt` (with an input script), `import-trace.txt` (`--import-trace`),
   `command.json` (the command line, the staleness plan and the exit code) and `digest.txt`.
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
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_boot_reports  # noqa: E402
import screen_timeline  # noqa: E402

EXE_SUFFIX = ".exe" if sys.platform == "win32" else ""
SOURCE_SUFFIXES = {".cpp", ".h", ".hpp", ".c", ".cmake", ".txt", ".rs", ".toml"}

PRESETS: dict[str, dict] = {
    "smoke": {"max_ms": 60_000, "watchdog_ms": 90_000, "input": None},
    "start-screen": {"max_ms": 120_000, "watchdog_ms": 150_000, "input": "start-screen"},
    "menu-flow": {"max_ms": 240_000, "watchdog_ms": 270_000, "input": "menu-flow"},
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


def workspace_dirs(repo: Path) -> list[Path]:
    """Crate directories of the cargo workspace (the root Cargo.toml `members`), else `recomp`."""
    manifest = repo / "Cargo.toml"
    members: list[str] = []
    if manifest.is_file():
        text = manifest.read_text(encoding="utf-8", errors="replace")
        block = re.search(r"members\s*=\s*\[(.*?)\]", text, re.S)
        if block:
            members = re.findall(r'"([^"]+)"', block.group(1))
    return [repo / m for m in members if "*" not in m] or [repo / "recomp"]


def shim_inputs(repo: Path) -> list[Path]:
    """The runtime files `build-program` scans to bind imports (recomp `checkout_shim_headers`):
    `runtime/os/*.h` and `*.cpp` (export tables, `pg6_gthunk_*` registrations),
    `runtime/gpu/d3d12_xs.h`, `runtime/gpu/d3d12_xs_exports.*`, `runtime/guest/guest_callbacks.h`."""
    runtime = repo / "runtime"
    files = sorted((runtime / "os").glob("*.h")) + sorted((runtime / "os").glob("*.cpp"))
    files += [runtime / "gpu" / "d3d12_xs.h", runtime / "guest" / "guest_callbacks.h"]
    files += sorted((runtime / "gpu").glob("d3d12_xs_exports.*"))
    return files


class Plan:
    def __init__(self) -> None:
        self.steps: list[dict] = []

    def add(self, stage: str, reason: str, cmd: list[str]) -> None:
        self.steps.append({"stage": stage, "reason": reason, "cmd": cmd})

    def has(self, stage: str) -> bool:
        return any(s["stage"] == stage for s in self.steps)


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


def make_plan(repo: Path, build_dir: Path, exe: Path | None, game_dir: Path | None, jobs: int,
              force: bool, harness_override: Path | None) -> tuple[Plan, Path, Path]:
    """Returns the build plan, the program directory and the harness path."""
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
        plan.add("recomp", reason, ["cargo", "build", "--release", "-p", "playgta6-recomp"])

    manifest = program / "program-manifest.json"
    exe_time = mtime(exe) if exe else 0.0
    # The import bindings come from the runtime's shim headers and export tables: a new guest
    # thunk (e.g. combase!CoCreateInstance) is only bound once the program is generated again.
    shim_time = max([mtime(p) or 0.0 for p in shim_inputs(repo)] + [0.0])
    reason = stale("program", manifest, [("the recompiler", mtime(recomp) or 0.0), ("the exe", exe_time or 0.0),
                                         ("the runtime shim headers or export tables", shim_time)])
    if reason is None and plan.has("recomp"):
        reason = "the recompiler is rebuilt"
    if reason:
        plan.add("program", reason, [str(recomp), "build-program", str(exe), "--out", str(program),
                                     "--dll-dir", str(game_dir), "--target", "native", "--jobs", str(jobs)])

    if harness_override is None:
        runtime_newest = newest_mtime(repo / "runtime", [repo / "tools" / "build_native_boot.py"])
        reason = stale("harness", harness, [("the program", mtime(manifest) or 0.0),
                                            ("the runtime sources or build_native_boot.py", runtime_newest)])
        if reason is None and plan.has("program"):
            reason = "the program is rebuilt"
        if reason:
            plan.add("harness", reason, [sys.executable, str(repo / "tools" / "build_native_boot.py"), str(program),
                                         "--cache-dir", str(build_dir / "obj-cache"), "--jobs", str(jobs),
                                         "--keep-going", "--delete-ir", "--debug-info"])
    return plan, program, harness


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def menu_flow_script_text(duration_ms: int) -> str:
    """The menu-flow preset: past the intro movie (about 20 s) A every 2.5 s through the start
    screen, loading and UI TEXT SIZE; then D-pad Down, Down, A (the "Next >" entry of LEGAL
    NOTICES) every 12 s, which lands whatever the exact arrival time; then A every 3 s."""
    lines = ["# menu-flow (tools/run_native_boot.py): A every 2.5 s from 20 s; Down, Down, A every 12 s",
             "# from 64 s (LEGAL NOTICES -> Next >); then A every 3 s"]
    events = [(t, "A", INPUT_HOLD_MS) for t in range(20_000, 60_000, 2_500)]
    for t in range(64_000, 124_000, 12_000):
        events += [(t, "DDown", 150), (t + 2_000, "DDown", 150), (t + 4_500, "A", INPUT_HOLD_MS)]
    events += [(t, "A", INPUT_HOLD_MS) for t in range(130_000, duration_ms, 3_000)]
    lines += [f"{t} {b} {h}" for t, b, h in events if t < duration_ms]
    return "\n".join(lines) + "\n"


def input_script_text(duration_ms: int, kind: str = "start-screen") -> str:
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


def load_report(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def timeline_presents(report: dict) -> list:
    """Capture samples, else the runtime's own per-present samples (R40: no GPU capture needed)."""
    return report.get("presents") or screen_timeline.presents_from_samples(report.get("present_samples") or [])


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
    p.add_argument("--rebuild", action="store_true", help="rebuild every stage")
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
    p.add_argument("--import-trace", action="store_true", help="write import-trace.txt")
    p.add_argument("--dump-pipelines", help="harness --dump-pipelines (a directory under work/)")
    p.add_argument("--audio-dump", help="harness --audio-dump: WAV file of the stereo mix")
    p.add_argument("--baseline", help="earlier run (name in the runs directory, or a directory) to compare with")
    p.add_argument("--ignore", action="append", default=[], metavar="PATTERN", help="compare_boot_reports --ignore")
    p.add_argument("--time-tolerance", type=float, metavar="REL")
    p.add_argument("--count-tolerance", type=float, metavar="REL")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="after `--`: further playgta6_boot options")
    return p


def main(argv: Sequence[str] | None = None, runner: Runner = run_command,
         env: Mapping[str, str] | None = None, repo_root: Path = REPO_ROOT) -> int:
    env = os.environ if env is None else env
    try:
        args = build_arg_parser().parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code in (0, None) else 2
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    try:
        return _run(args, extra, runner, env, repo_root)
    except RunnerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run(args, extra: list[str], runner: Runner, env: Mapping[str, str], repo: Path) -> int:
    exe = args.exe or (Path(env["PLAYGTA6_EXE"]) if env.get("PLAYGTA6_EXE") else None)
    game_dir = args.game_dir or (Path(env["PLAYGTA6_GAME_DIR"]) if env.get("PLAYGTA6_GAME_DIR") else None)
    if exe is None or game_dir is None:
        raise RunnerError("the game exe and package directory are required: --exe / --game-dir "
                          "or PLAYGTA6_EXE / PLAYGTA6_GAME_DIR (they are never read from the repository)")
    if not args.dry_run and not exe.is_file():
        raise RunnerError(f"exe not found: {exe}")
    if not args.dry_run and not game_dir.is_dir():
        raise RunnerError(f"game directory not found: {game_dir}")
    build_dir, note = choose_build_dir(repo, args.build_dir, env)
    if note:
        print(note, file=sys.stderr)
    runs_dir = (args.runs_dir or build_dir / "runs").resolve()
    preset = PRESETS[args.preset]
    name = args.name or f"{args.preset}-{time.strftime('%Y%m%d-%H%M%S')}"
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise RunnerError(f"invalid run name {name!r}")
    run_dir = runs_dir / name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RunnerError(f"run directory {run_dir} already exists (pick another --name)")

    plan, program, harness = make_plan(repo, build_dir, exe, game_dir, args.jobs, args.rebuild, args.harness)
    if args.trust_build:
        plan.steps = []
    max_ms = args.max_ms if args.max_ms is not None else preset["max_ms"]
    watchdog_ms = args.watchdog_ms if args.watchdog_ms is not None else preset["watchdog_ms"]

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
    if args.import_trace:
        cmd += ["--import-trace", str(run_dir / "import-trace.txt")]
    if args.dump_pipelines:
        cmd += ["--dump-pipelines", args.dump_pipelines]
    if args.audio_dump:
        cmd += ["--audio-dump", args.audio_dump]
    cmd += extra

    for step in plan.steps:
        print(f"[plan] {step['stage']}: {step['reason']}", file=sys.stderr)
    if args.dry_run:
        print(json.dumps({"plan": plan.steps, "command": cmd}, indent=2))
        return 0
    if plan.steps and args.no_build:
        raise RunnerError("stale stages with --no-build: " + ", ".join(s["stage"] for s in plan.steps))
    if args.harness is not None and not args.harness.is_file():
        raise RunnerError(f"harness not found: {args.harness}")
    for step in plan.steps:
        print(f"[build] {step['stage']}: {' '.join(step['cmd'])}", file=sys.stderr, flush=True)
        if runner(step["cmd"], repo, None, None) != 0:
            raise RunnerError(f"{step['stage']} failed")
    if not harness.is_file():
        raise RunnerError(f"harness not found after the build: {harness}")

    run_dir.mkdir(parents=True, exist_ok=True)
    if preset["input"] and not args.input_script:
        script_path.write_text(input_script_text(max_ms, preset["input"]), encoding="utf-8")
    print(f"[run] {run_dir}", file=sys.stderr, flush=True)
    code = runner(cmd, repo, run_dir / "boot.log", watchdog_ms / 1000 + 60)
    report = load_report(run_dir / "boot-report.json")
    (run_dir / "command.json").write_text(json.dumps({
        "preset": args.preset, "command": cmd, "plan": plan.steps, "harness_exit": code}, indent=2), encoding="utf-8")
    if report is None:
        print(f"error: the harness (exit {code}) wrote no report; see {run_dir / 'boot.log'}", file=sys.stderr)
        return 3
    presents = timeline_presents(report)
    if presents and "screen_timeline" not in report:
        times: list[float] = []
        ipath = Path(args.input_script) if args.input_script else script_path
        if ipath.is_file():
            times = screen_timeline.parse_input_times(ipath.read_text(encoding="utf-8"))
        report["screen_timeline"] = screen_timeline.build_timeline(presents, times)
        (run_dir / "boot-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    text = digest(report, code)
    (run_dir / "digest.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
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
