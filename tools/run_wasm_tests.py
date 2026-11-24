#!/usr/bin/env python3
"""Local wasm64 runtime test runner (Track 2 / O25).

CI builds and runs the ~28 C++ host tests in ``runtime/CMakeLists.txt`` natively only, but the
browser runtime is Wasm64 (``-sMEMORY64=1 -pthread``, function pointers are table indices, string
literals live below 64 KiB, the guest pool is 12 GiB of identity-mapped linear memory). This tool
answers "do all runtime tests pass as wasm64?" with one command. It is local-only by design (the
build is heavy and CI minutes are metered); it is not wired into any workflow.

What it does:

1. Configures ``runtime/`` with the Emscripten toolchain (``emcmake cmake -DPLAYGTA6_WASM_TESTS=ON``)
   inside the ``emscripten/emsdk`` Docker image (repo mounted read-only, gitignored persistent
   build dir ``runtime/build-wasm64-tests/``, named Docker volume for the Emscripten cache so the
   second run is incremental). ``--no-docker`` uses an activated local emsdk instead.
2. Builds every CTest-registered test executable (discovered via ``ctest --show-only=json-v1``,
   so tests other tasks add are picked up automatically) with the flags the real runtime uses plus
   ``-fexceptions`` and the Node host-import library ``tools/wasm_test_support/host_imports.js``.
3. Runs each produced ``.js`` under a memory64-capable Node (``node:24`` image by default),
   in parallel, one Node process per test, with a per-test timeout. A tiny memory64 probe module
   is instantiated first; if Node cannot run memory64 the run aborts with exit code 2 instead of
   reporting every test as failed.
4. Prints a table (test, built, result, duration, first failing check lines) and optionally
   writes a JSON report.

Usage::

    python tools/run_wasm_tests.py [--tests GLOB ...] [--jobs N] [--build-dir DIR]
        [--keep-going] [--json REPORT] [--emsdk-image TAG] [--node-image TAG]
        [--no-docker] [--timeout SECONDS] [--build-type Release|Debug|RelWithDebInfo]
        [--list]

``--tests`` globs match the CTest name or the executable target name (``test_xbox_scarlett``).
The build always reports every failing target (``ninja -k 0``). Without ``--keep-going`` no test
is started after the first build or test failure (reported ``not-run``); with it every built
test runs.

Exit codes: 0 all selected tests passed (skips allowed), 1 a test failed / timed out / did not
build, 2 environment problem (no Docker/emsdk, Node without memory64, nothing discovered).

JSON report schema (``schema_version`` 1)::

    {
      "schema_version": 1,
      "tool": "tools/run_wasm_tests.py",
      "started_at": "<UTC ISO-8601>",
      "duration_s": <float>,
      "config": {"mode": "docker"|"local", "emsdk_image": str|null, "node_image": str|null,
                  "node_version": str, "node_flags": [str], "build_dir": str, "build_type": str,
                  "jobs": int, "timeout_s": float, "filters": [str]},
      "build": {"ok": bool, "duration_s": float, "log": str},
      "summary": {"total": int, "passed": int, "failed": int, "timeout": int, "skipped": int,
                   "not_built": int, "not_run": int},
      "tests": [{"name": str, "target": str, "built": bool,
                  "status": "pass"|"fail"|"timeout"|"skipped"|"not-built"|"not-run",
                  "returncode": int|null, "duration_s": float, "skip_reason": str|null,
                  "failures": [str]}]
    }
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUILD_DIR_REL = Path("runtime") / "build-wasm64-tests"
DEFAULT_EMSDK_IMAGE = "emscripten/emsdk:4.0.10"
DEFAULT_NODE_IMAGE = "node:24"
EMCACHE_VOLUME = "playgta6-wasm-emcache"
DEFAULT_TIMEOUT_S = 180.0
EMSDK_UPSTREAM_REL = Path("work") / "emsdk" / "upstream" / "emscripten"

# The emsdk image ships CMake 3.22 but runtime/CMakeLists.txt needs 3.24; the derived image adds a
# current CMake + Ninja once and is cached by Docker afterwards.
DERIVED_IMAGE_REPO = "playgta6-wasm-tests"
DERIVED_DOCKERFILE = (
    "FROM {base}\n"
    'RUN pip install --no-cache-dir "cmake>=3.24" ninja\n'
)

# Environment-level skips only (toolchain limits, never "the test fails on wasm"): CTest name ->
# reason. Real wasm-only failures are fixed in the runtime/tests, not listed here.
KNOWN_SKIPS: dict[str, str] = {}

# A shared-memory memory64 module (the combination pthreads + MEMORY64 needs). If Node cannot
# compile this, none of the test builds can run.
MEMORY64_PROBE_JS = (
    "const b=new Uint8Array([0,97,115,109,1,0,0,0,5,4,1,7,1,1]);"
    "try{new WebAssembly.Module(b);console.log('memory64:ok '+process.version)}"
    "catch(e){console.log('memory64:unsupported '+process.version+' '+e.message);process.exit(3)}"
)
NODE_MEMORY64_FLAG = "--experimental-wasm-memory64"

STATUS_PASS, STATUS_FAIL, STATUS_TIMEOUT = "pass", "fail", "timeout"
STATUS_SKIPPED, STATUS_NOT_BUILT, STATUS_NOT_RUN = "skipped", "not-built", "not-run"

_FAIL_LINE_RE = re.compile(
    r"(^FAIL\b|\bFAIL\b\s*\[|assert|abort|Aborted|RuntimeError|Uncaught|\bunreachable\b|"
    r"failure\(s\)|check\(s\) failed|\bTrap\b|CompileError|out of memory)",
    re.IGNORECASE,
)


class EnvironmentProblem(RuntimeError):
    """The toolchain/Node cannot do the job (reported with exit code 2)."""


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    output: str
    timed_out: bool = False
    duration_s: float = 0.0


Executor = Callable[..., CommandResult]
"""``executor(argv, *, timeout=None, on_line=None, env=None, cwd=None, name=None, stdin_text=None)``"""


@dataclass(frozen=True)
class CTestEntry:
    name: str
    target: str  # executable target / output basename (without .js)
    command: tuple[str, ...]


@dataclass
class TestResult:
    name: str
    target: str
    built: bool = False
    status: str = STATUS_NOT_RUN
    returncode: int | None = None
    duration_s: float = 0.0
    skip_reason: str | None = None
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "target": self.target,
            "built": self.built,
            "status": self.status,
            "returncode": self.returncode,
            "duration_s": round(self.duration_s, 3),
            "skip_reason": self.skip_reason,
            "failures": list(self.failures),
        }


@dataclass
class Config:
    repo_root: Path = REPO_ROOT
    build_dir: Path = REPO_ROOT / DEFAULT_BUILD_DIR_REL
    filters: list[str] = field(default_factory=list)
    jobs: int = 4
    keep_going: bool = False
    json_path: Path | None = None
    emsdk_image: str = DEFAULT_EMSDK_IMAGE
    node_image: str = DEFAULT_NODE_IMAGE
    docker: bool = True
    timeout_s: float = DEFAULT_TIMEOUT_S
    build_type: str = "Release"
    list_only: bool = False


# ---------------------------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_wasm_tests.py",
        description="Build every CTest-registered runtime test as wasm64 + pthreads and run it "
        "under a memory64-capable Node. Local-only; see the module docstring for the JSON schema.",
    )
    parser.add_argument("--tests", nargs="+", default=[], metavar="GLOB",
                        help="only tests whose CTest name or target matches (default: all)")
    parser.add_argument("--jobs", "-j", type=int, default=0,
                        help="parallel build/test jobs (default: min(4, CPUs) for tests, CPUs for the build)")
    parser.add_argument("--build-dir", type=Path, default=None,
                        help=f"CMake build directory (default: {DEFAULT_BUILD_DIR_REL.as_posix()}, gitignored)")
    parser.add_argument("--keep-going", action="store_true",
                        help="continue building/running after a failure (default: stop at the first one)")
    parser.add_argument("--json", type=Path, default=None, dest="json_path", metavar="REPORT",
                        help="write a machine-readable report to this path")
    parser.add_argument("--emsdk-image", default=DEFAULT_EMSDK_IMAGE, help="Docker image used to build")
    parser.add_argument("--node-image", default=DEFAULT_NODE_IMAGE,
                        help="Docker image used to run the tests (must support memory64, Node >= 24)")
    parser.add_argument("--no-docker", action="store_true",
                        help="use an activated local emsdk and the local node instead of Docker")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, dest="timeout_s",
                        help="per-test timeout in seconds (default: %(default)s)")
    parser.add_argument("--build-type", default="Release", choices=("Release", "Debug", "RelWithDebInfo"))
    parser.add_argument("--list", action="store_true", dest="list_only",
                        help="configure and list the discovered tests, then exit")
    return parser


def config_from_args(argv: Sequence[str] | None, repo_root: Path = REPO_ROOT) -> Config:
    args = build_arg_parser().parse_args(argv)
    build_dir = args.build_dir if args.build_dir is not None else repo_root / DEFAULT_BUILD_DIR_REL
    if not build_dir.is_absolute():
        build_dir = Path.cwd() / build_dir
    if args.jobs < 0:
        raise SystemExit("--jobs must be >= 0")
    return Config(
        repo_root=repo_root,
        build_dir=build_dir,
        filters=list(args.tests),
        jobs=args.jobs,
        keep_going=args.keep_going,
        json_path=args.json_path,
        emsdk_image=args.emsdk_image,
        node_image=args.node_image,
        docker=not args.no_docker,
        timeout_s=args.timeout_s,
        build_type=args.build_type,
        list_only=args.list_only,
    )


def parse_ctest_json(text: str) -> list[CTestEntry]:
    """Parse ``ctest --show-only=json-v1``; the target is the ``.js`` (or bare exe) in the command."""
    data = json.loads(text)
    if data.get("kind") != "ctestInfo":
        raise ValueError("not a ctest json-v1 document")
    entries: list[CTestEntry] = []
    for test in data.get("tests", []):
        command = [str(part) for part in test.get("command", [])]
        exe = next((p for p in reversed(command) if not p.startswith("-")), None)
        if exe is None:
            continue
        base = re.split(r"[\\/]", exe)[-1]
        target = base[:-3] if base.endswith(".js") else re.sub(r"\.(exe|wasm)$", "", base)
        entries.append(CTestEntry(name=str(test["name"]), target=target, command=tuple(command)))
    return entries


def select_tests(entries: Sequence[CTestEntry], filters: Sequence[str]) -> list[CTestEntry]:
    if not filters:
        return list(entries)
    return [e for e in entries
            if any(fnmatch.fnmatchcase(e.name, g) or fnmatch.fnmatchcase(e.target, g) for g in filters)]


def extract_failure_lines(output: str, limit: int = 3) -> list[str]:
    """First lines that look like failed checks/aborts; falls back to the last output lines."""
    lines = [ln.rstrip() for ln in output.splitlines() if ln.strip()]
    hits = [ln for ln in lines if _FAIL_LINE_RE.search(ln)]
    picked = hits[:limit] if hits else lines[-limit:]
    return [ln[:200] for ln in picked]


def classify_result(returncode: int | None, timed_out: bool) -> str:
    if timed_out:
        return STATUS_TIMEOUT
    return STATUS_PASS if returncode == 0 else STATUS_FAIL


def parse_probe_output(output: str) -> tuple[bool, str]:
    """-> (memory64 supported, node version) from the probe script's output."""
    match = re.search(r"memory64:(ok|unsupported)\s+(v\d+\.\d+\.\d+)", output)
    if not match:
        return False, "unknown"
    return match.group(1) == "ok", match.group(2)


def summarize(results: Sequence[TestResult]) -> dict[str, int]:
    count = lambda s: sum(1 for r in results if r.status == s)  # noqa: E731
    return {
        "total": len(results),
        "passed": count(STATUS_PASS),
        "failed": count(STATUS_FAIL),
        "timeout": count(STATUS_TIMEOUT),
        "skipped": count(STATUS_SKIPPED),
        "not_built": count(STATUS_NOT_BUILT),
        "not_run": count(STATUS_NOT_RUN),
    }


def exit_code_for(results: Sequence[TestResult]) -> int:
    bad = {STATUS_FAIL, STATUS_TIMEOUT, STATUS_NOT_BUILT, STATUS_NOT_RUN}
    return 1 if any(r.status in bad for r in results) else 0


def build_report(
    *,
    config: Config,
    results: Sequence[TestResult],
    node_version: str,
    node_flags: Sequence[str],
    build_ok: bool,
    build_duration_s: float,
    build_log: str,
    started_at: str,
    duration_s: float,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "tools/run_wasm_tests.py",
        "started_at": started_at,
        "duration_s": round(duration_s, 3),
        "config": {
            "mode": "docker" if config.docker else "local",
            "emsdk_image": config.emsdk_image if config.docker else None,
            "node_image": config.node_image if config.docker else None,
            "node_version": node_version,
            "node_flags": list(node_flags),
            "build_dir": str(config.build_dir),
            "build_type": config.build_type,
            "jobs": config.jobs,
            "timeout_s": config.timeout_s,
            "filters": list(config.filters),
        },
        "build": {"ok": build_ok, "duration_s": round(build_duration_s, 3), "log": build_log},
        "summary": summarize(results),
        "tests": [r.to_dict() for r in results],
    }


def format_table(results: Sequence[TestResult]) -> str:
    rows = [("test", "built", "result", "time", "details")]
    for r in results:
        detail = r.skip_reason if r.status == STATUS_SKIPPED else (r.failures[0] if r.failures else "")
        rows.append((r.name, "yes" if r.built else "no", r.status.upper(),
                     f"{r.duration_s:.1f}s" if r.duration_s else "-", detail))
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    out = []
    for i, row in enumerate(rows):
        out.append("  ".join(row[j].ljust(widths[j]) for j in range(4)) + "  " + row[4])
        if i == 0:
            out.append("  ".join("-" * w for w in widths))
    for r in results:
        if r.status in (STATUS_FAIL, STATUS_TIMEOUT) and len(r.failures) > 1:
            out.append(f"  {r.name}:")
            out.extend(f"    {ln}" for ln in r.failures)
    s = summarize(results)
    out.append("")
    out.append(
        f"{s['passed']}/{s['total']} passed, {s['failed']} failed, {s['timeout']} timed out, "
        f"{s['skipped']} skipped, {s['not_built']} not built, {s['not_run']} not run"
    )
    return "\n".join(out)


def derived_image_tag(emsdk_image: str) -> str:
    return f"{DERIVED_IMAGE_REPO}:" + re.sub(r"[^A-Za-z0-9_.-]+", "-", emsdk_image.split("/")[-1].replace(":", "-"))


def cmake_configure_args(source: str, build: str, build_type: str) -> list[str]:
    return ["emcmake", "cmake", "-S", source, "-B", build, "-G", "Ninja",
            f"-DCMAKE_BUILD_TYPE={build_type}", "-DPLAYGTA6_WASM_TESTS=ON"]


def cmake_build_args(build: str, targets: Sequence[str], jobs: int) -> list[str]:
    """Always `ninja -k 0`: one build reports every failing target, not just the first."""
    args = ["cmake", "--build", build]
    if jobs:
        args += ["--parallel", str(jobs)]
    if targets:
        args += ["--target", *targets]
    args += ["--", "-k", "0"]
    return args


# ---------------------------------------------------------------------------------------------
# Process execution
# ---------------------------------------------------------------------------------------------


def run_command(argv: Sequence[str], *, timeout: float | None = None,
                on_line: Callable[[str], None] | None = None, env: dict | None = None,
                cwd: Path | str | None = None, name: str | None = None,
                stdin_text: str | None = None) -> CommandResult:
    """Run a process, capturing merged output; ``name`` is a Docker container to kill on timeout."""
    start = time.monotonic()
    full_env = dict(os.environ)
    full_env["MSYS_NO_PATHCONV"] = "1"  # Git Bash must not rewrite /src style container paths
    full_env["MSYS2_ARG_CONV_EXCL"] = "*"
    if env:
        full_env.update(env)
    try:
        proc = subprocess.Popen(list(argv), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", env=full_env,
                                cwd=None if cwd is None else str(cwd))
    except FileNotFoundError as exc:
        return CommandResult(127, f"{argv[0]}: {exc}", False, 0.0)
    if stdin_text is not None:
        assert proc.stdin is not None
        proc.stdin.write(stdin_text)
        proc.stdin.close()
    chunks: list[str] = []

    def pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            chunks.append(line)
            if on_line:
                on_line(line.rstrip("\n"))

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        if name:
            subprocess.run(["docker", "kill", name], capture_output=True, check=False)
        proc.kill()
        proc.wait()
    reader.join(timeout=10)
    return CommandResult(proc.returncode, "".join(chunks), timed_out, time.monotonic() - start)


# ---------------------------------------------------------------------------------------------
# Toolchains
# ---------------------------------------------------------------------------------------------


class DockerToolchain:
    """Builds in a long-lived emsdk container (docker exec per step); runs tests in node:N."""

    def __init__(self, config: Config, executor: Executor):
        self.config = config
        self.run = executor
        self.container = f"pg6-wasm-build-{uuid.uuid4().hex[:8]}"
        self.src = "/src"
        self.build = "/build"

    def check_available(self) -> None:
        res = self.run(["docker", "version", "--format", "{{.Server.Version}}"], timeout=60)
        if res.returncode != 0:
            raise EnvironmentProblem("Docker is not available (is Docker Desktop running?): "
                                     + res.output.strip()[:200] + " -- use --no-docker for a local emsdk")

    def ensure_image(self) -> str:
        tag = derived_image_tag(self.config.emsdk_image)
        if self.run(["docker", "image", "inspect", tag], timeout=60).returncode == 0:
            return tag
        print(f"Docker: building helper image {tag} (one-time: CMake >= 3.24 + Ninja on {self.config.emsdk_image})")
        dockerfile = DERIVED_DOCKERFILE.format(base=self.config.emsdk_image)
        res = self.run(["docker", "build", "-t", tag, "-"], timeout=900, stdin_text=dockerfile)
        if res.returncode != 0:
            raise EnvironmentProblem("docker build of the helper image failed:\n" + res.output[-800:])
        return tag

    def start(self) -> None:
        tag = self.ensure_image()
        self.config.build_dir.mkdir(parents=True, exist_ok=True)
        res = self.run([
            "docker", "run", "-d", "--rm", "--name", self.container,
            "-v", f"{self.config.repo_root}:{self.src}:ro",
            "-v", f"{self.config.build_dir}:{self.build}",
            "-v", f"{EMCACHE_VOLUME}:/emcache", "-e", "EM_CACHE=/emcache",
            "-w", self.build, tag, "sleep", "infinity",
        ], timeout=120)
        if res.returncode != 0:
            raise EnvironmentProblem("could not start the build container:\n" + res.output[-500:])

    def stop(self) -> None:
        self.run(["docker", "rm", "-f", self.container], timeout=60)

    def exec(self, argv: Sequence[str], *, on_line=None, timeout=None) -> CommandResult:
        return self.run(["docker", "exec", self.container, *argv], on_line=on_line, timeout=timeout)

    # Paths as seen by the build tools.
    @property
    def source_arg(self) -> str:
        return f"{self.src}/runtime"

    def node_prefix(self, name: str, timeout_s: float) -> list[str]:
        return ["docker", "run", "--rm", "--name", name,
                "-v", f"{self.config.build_dir}:/build:ro", "-v", f"{self.config.repo_root}:/src:ro",
                "-w", "/tmp",  # tests (NODERAWFS) write scratch files to the cwd
                self.config.node_image, "timeout", "-s", "KILL", str(int(timeout_s) + 5), "node"]

    def probe_prefix(self) -> list[str]:
        return ["docker", "run", "--rm", self.config.node_image, "node"]

    def test_script_path(self, target: str) -> str:
        return f"/build/{target}.js"


class LocalToolchain:
    """Uses an activated emsdk (PATH, $EMSDK, or work/emsdk) and the local node."""

    def __init__(self, config: Config, executor: Executor):
        self.config = config
        self.run = executor
        self.env: dict[str, str] = {}

    def check_available(self) -> None:
        dirs = [os.environ.get("PATH", "")]
        emsdk = os.environ.get("EMSDK")
        candidates = []
        if emsdk:
            candidates.append(Path(emsdk) / "upstream" / "emscripten")
        candidates.append(self.config.repo_root / EMSDK_UPSTREAM_REL)
        for cand in candidates:
            if cand.is_dir():
                dirs.insert(0, str(cand))
        self.env = {"PATH": os.pathsep.join(dirs)}
        if shutil.which("emcmake", path=self.env["PATH"]) is None:
            raise EnvironmentProblem("emcmake not found on PATH, $EMSDK or work/emsdk -- activate an emsdk "
                                     "or drop --no-docker")
        for tool in ("cmake", "ninja"):
            if shutil.which(tool, path=self.env["PATH"]) is None:
                raise EnvironmentProblem(f"{tool} not found on PATH (needed for --no-docker)")

    def start(self) -> None:
        self.config.build_dir.mkdir(parents=True, exist_ok=True)

    def stop(self) -> None:
        pass

    def exec(self, argv: Sequence[str], *, on_line=None, timeout=None) -> CommandResult:
        exe = shutil.which(argv[0], path=self.env.get("PATH")) or argv[0]
        return self.run([exe, *argv[1:]], on_line=on_line, timeout=timeout, env=self.env,
                        cwd=self.config.build_dir)

    @property
    def source_arg(self) -> str:
        return str(self.config.repo_root / "runtime")

    @property
    def build(self) -> str:
        return str(self.config.build_dir)

    def node_prefix(self, name: str, timeout_s: float) -> list[str]:
        return ["node"]

    def probe_prefix(self) -> list[str]:
        return ["node"]

    def test_script_path(self, target: str) -> str:
        return str(self.config.build_dir / f"{target}.js")


# ---------------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------------


def probe_node(tc, executor: Executor) -> tuple[str, list[str]]:
    """-> (node version, extra node flags). Raises EnvironmentProblem if memory64 cannot run."""
    for flags in ([], [NODE_MEMORY64_FLAG]):
        res = executor([*tc.probe_prefix(), *flags, "-e", MEMORY64_PROBE_JS], timeout=180)
        ok, version = parse_probe_output(res.output)
        if res.returncode == 0 and ok:
            return version, flags
        if res.returncode == 127 or (not res.output.strip() and res.returncode != 0):
            raise EnvironmentProblem("could not run node for the memory64 probe: " + res.output.strip()[:200])
    raise EnvironmentProblem(
        f"Node {version} cannot instantiate memory64 modules (needs Node >= 24 or "
        f"{NODE_MEMORY64_FLAG}); use --node-image node:24. Not running any tests.")


def _built_markers(tc, targets: Sequence[str]) -> dict[str, bool]:
    """A target is built when `ninja -n <target>` has no work left (stale outputs do not count)."""
    built: dict[str, bool] = {}
    for target in targets:
        res = tc.exec(["ninja", "-C", tc.build, "-n", target], timeout=120)
        built[target] = res.returncode == 0 and "no work to do" in res.output
    return built


def rwt_merge(first: CommandResult, second: CommandResult) -> CommandResult:
    return CommandResult(second.returncode, first.output + second.output)


def run_suite(config: Config, executor: Executor = run_command, out=print) -> tuple[int, dict | None]:
    started = time.monotonic()
    started_at = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    tc = DockerToolchain(config, executor) if config.docker else LocalToolchain(config, executor)
    tc.check_available()
    node_version, node_flags = probe_node(tc, executor)
    out(f"node {node_version}{' ' + ' '.join(node_flags) if node_flags else ''}: memory64 ok "
        f"({'docker ' + config.node_image if config.docker else 'local'})")
    cpus = os.cpu_count() or 4
    build_jobs = config.jobs or cpus
    test_jobs = config.jobs or min(4, cpus)
    config.jobs = test_jobs

    tc.start()
    try:
        out("configuring (emcmake cmake -DPLAYGTA6_WASM_TESTS=ON) ...")
        res = tc.exec(cmake_configure_args(tc.source_arg, tc.build, config.build_type))
        if res.returncode != 0:
            raise EnvironmentProblem("CMake configure failed:\n" + res.output[-1500:])
        listing = tc.exec(["ctest", "--test-dir", tc.build, "--show-only=json-v1"])
        try:
            entries = select_tests(parse_ctest_json(listing.output), config.filters)
        except (ValueError, KeyError) as exc:
            raise EnvironmentProblem(f"could not read the CTest test list: {exc}\n{listing.output[-500:]}")
        if not entries:
            raise EnvironmentProblem("no tests discovered" + (f" matching {config.filters}" if config.filters else ""))
        if config.list_only:
            for e in entries:
                out(f"{e.name}\t{e.target}")
            return 0, None

        targets = sorted({e.target for e in entries})
        out(f"building {len(targets)} test executable(s) as wasm64 + pthreads (jobs={build_jobs}) ...")
        log_path = config.build_dir / "build.log"
        progress = {"last": -1}

        def on_build_line(line: str) -> None:
            match = re.match(r"\[(\d+)/(\d+)\]", line)
            if match and int(match.group(1)) // 10 != progress["last"]:
                progress["last"] = int(match.group(1)) // 10
                out(f"  build {line[:100]}")

        t0 = time.monotonic()
        build_res = tc.exec(cmake_build_args(tc.build, targets if config.filters else [], build_jobs),
                            on_line=on_build_line)
        build_duration = time.monotonic() - t0
        try:
            log_path.write_text(build_res.output, encoding="utf-8")
        except OSError:
            pass
        if build_res.returncode == 0:
            built = {t: True for t in targets}  # ninja only exits 0 when every requested edge is done
        else:
            # Re-run once: the first run may have populated the Emscripten cache (touching sysroot
            # headers), which makes ninja's dry run report everything dirty. The second run is
            # incremental and leaves only the genuinely failing targets pending.
            retry = tc.exec(cmake_build_args(tc.build, targets if config.filters else [], build_jobs))
            build_res = rwt_merge(build_res, retry)
            built = _built_markers(tc, targets)
    finally:
        tc.stop()

    results = [TestResult(name=e.name, target=e.target, built=built.get(e.target, False)) for e in entries]
    build_ok = all(r.built for r in results)
    for r in results:
        if not r.built:
            r.status = STATUS_NOT_BUILT
            r.failures = _build_errors_for(build_res.output, r.target)
    if not build_ok:
        out(f"build: {sum(not r.built for r in results)} target(s) failed to build (log: {log_path})")

    stop_flag = threading.Event()
    skipped = {r.name: KNOWN_SKIPS[r.name] for r in results if r.name in KNOWN_SKIPS}
    if not build_ok and not config.keep_going:
        stop_flag.set()

    def run_one(r: TestResult) -> None:
        if r.name in skipped:
            r.status, r.skip_reason = STATUS_SKIPPED, skipped[r.name]
            return
        if not r.built:
            return
        if stop_flag.is_set():
            r.status = STATUS_NOT_RUN
            return
        container = f"pg6-wasm-test-{uuid.uuid4().hex[:8]}"
        argv = [*tc.node_prefix(container, config.timeout_s), *node_flags, tc.test_script_path(r.target)]
        res = executor(argv, timeout=config.timeout_s, name=container)
        timed_out = res.timed_out or (res.returncode == 137 and config.docker)
        r.returncode, r.duration_s = res.returncode, res.duration_s
        r.status = classify_result(res.returncode, timed_out)
        if r.status != STATUS_PASS:
            r.failures = extract_failure_lines(res.output)
            if not config.keep_going:
                stop_flag.set()
        out(f"  {r.status.upper():8} {r.name} ({r.duration_s:.1f}s)")

    runnable = [r for r in results if r.built or r.name in skipped]
    with ThreadPoolExecutor(max_workers=max(1, test_jobs)) as pool:
        list(pool.map(run_one, runnable))
    for r in results:
        if r.status == STATUS_NOT_RUN:
            r.failures = ["not run: stopped after an earlier failure (use --keep-going)"]

    out("")
    out(format_table(results))
    report = build_report(config=config, results=results, node_version=node_version, node_flags=node_flags,
                          build_ok=build_ok, build_duration_s=build_duration, build_log=str(log_path),
                          started_at=started_at, duration_s=time.monotonic() - started)
    return exit_code_for(results), report


def _build_errors_for(build_output: str, target: str) -> list[str]:
    """Compiler/linker error lines from the build log that mention the target (best effort)."""
    lines = build_output.splitlines()
    related: list[str] = []
    for i, line in enumerate(lines):
        if line.startswith("FAILED:") and (target in line or f"/{target}.dir/" in line):
            related.extend(ln for ln in lines[i + 1:i + 40] if re.search(r"error|undefined", ln, re.I))
    if not related:
        related = [ln for ln in lines if re.search(r"error:|undefined symbol", ln)]
    return [ln.strip()[:200] for ln in related[:3]] or ["build failed (see build.log)"]


def main(argv: Sequence[str] | None = None) -> int:
    config = config_from_args(argv)
    try:
        code, report = run_suite(config)
    except EnvironmentProblem as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if report is not None and config.json_path is not None:
        config.json_path.parent.mkdir(parents=True, exist_ok=True)
        config.json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"report written to {config.json_path}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
