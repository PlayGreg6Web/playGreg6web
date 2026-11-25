#!/usr/bin/env python3
"""One-command browser boot & test runner: rebuild what is stale, serve locally, drive Chrome via CDP, keep the run.

    python tools/run_browser_boot.py [--preset smoke|start-screen|long] [--name NAME]
        [--smoke] [--boot-dir DIR] [--game-dir DIR] [--shader-pack FILE] [--out-dir DIR]
        [--runs-dir DIR] [--port PORT] [--duration-s N] [--headless | --headed]
        [--chrome PATH] [--trust-build] [--rebuild] [--no-build] [--dry-run]
        [--capture-frames [DIR]] [--capture-interval-sec SEC] [--mp4] [--mp4-fps FPS]
        [--mp4-skip-black] [--hide-hud] [--baseline NAME|DIR]
        [--input-script PRESET] [--stream-shaders] [--stream-threshold-mb MB]
        [--boot-options JSON|FILE] [--report-to-log] [--max-wall-ms MS] [--dump-pipelines]
        [--quiet-server]

What it does, in order:

1. Staleness & build plan (real-game mode; skipped for `--smoke` and `--trust-build`):
   - `recomp`: `cargo build --release -p playgta6-recomp` when missing or older than
     workspace sources (only when `program` needs to be built or `--exe` is provided).
   - `program`: `playgta6-recomp build-program <exe> --out <boot-dir> --target wasm64
     --memory-model window-runs --indirect-dispatch shard --jobs N` (+ `--dll-dir <game-dir>`)
     when `<boot-dir>/program-manifest.json` is missing or stale.
   - `runtime_configure` / `runtime_lib`: Release PIC `libplaygta6_core.a` built via
     `emcmake cmake` (`-sMEMORY64=1 -pthread -fPIC`) when missing or older than `runtime/`.
   - `guest_wasm`: `python tools/build_guest_wasm.py <boot-dir> --out-dir <out-dir> ...`
     with the proven asymmetric split flags (`--split-modules 2 --split-strategy asymmetric
     --link-guest-boot --browser-boot-exports --proxy-to-pthread --strip --no-link-opt`)
     when `runtime.wasm`/`runtime.js` is missing or stale.
   - Optional `--shader-pack FILE`: atomically stages `<pack>` and any sibling
     `<pack>.reflect.json` / `reflection-manifest.json` sidecars into a gitignored
     staging directory served at `/shaders/`.
2. Local server + Chrome CDP session:
   - Picks a free localhost port (or `--port`), starts `server/serve_local.py` with
     `COOP`/`COEP` headers, and waits until `/api/status` responds.
   - Discovers Chrome/Chromium/Edge automatically (`--chrome`, `CHROME_PATH`, PATH,
     standard OS install paths) and launches it with a clean temporary `--user-data-dir`,
     ephemeral CDP debugging port, `--enable-unsafe-webgpu`, and `--headless=new` (unless
     `--headed`).
   - Connects over pure-Python stdlib CDP WebSocket, collects console logs, page errors,
     HUD/WebGPU/boot telemetry, optional screenshots (`--capture-frames`), optional H.264
     MP4 via `ffmpeg` (`--mp4`), and scripted input on `start-screen`.
   - Always tears down Chrome, temp user-data-dir, and `serve_local.py` in `finally:`.
3. Uniform artifacts under `<runs-dir>/<name>/` (default `work/browser-boot/runs/<name>/`,
   gitignored, zero host paths in shared outputs):
   - `boot-report.json` (`playgta6-boot-report` v2, compatible with `compare_boot_reports.py`)
   - `console.log`
   - `telemetry.json`
   - `command.json`
   - `digest.txt`
   - optional `frames/frame_0001.jpg`... and `capture.mp4`
4. Optional `--baseline NAME|DIR` invokes `tools/compare_boot_reports.py`.

Exit code: 0 ok (and matches baseline if given), 1 baseline differences, 2 usage or build error,
3 the browser session produced no report.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import io
import json
import math
import os
import re
import select
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_boot_reports  # noqa: E402
import clean_build_dirs  # noqa: E402

EXE_SUFFIX = ".exe" if sys.platform == "win32" else ""
SOURCE_SUFFIXES = {".cpp", ".h", ".hpp", ".c", ".cmake", ".txt", ".rs", ".toml"}
REPORT_SCHEMA = "playgta6-boot-report"
REPORT_SCHEMA_VERSION = 2
WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

DEFAULT_BUILD_DIR_REL = Path("work") / "browser-boot"
DEFAULT_RUNS_DIR_REL = DEFAULT_BUILD_DIR_REL / "runs"
DEFAULT_BOOT_DIR_REL = Path("work") / "recomp_build"
DEFAULT_OUT_DIR_REL = Path("web") / "dist"
DEFAULT_RUNTIME_PIC_REL = DEFAULT_BUILD_DIR_REL / "runtime-pic" / "libplaygta6_core.a"
DEFAULT_SHADER_STAGE_REL = DEFAULT_BUILD_DIR_REL / "staged-shaders"

PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {"duration_s": 60.0, "input": False},
    "start-screen": {"duration_s": 120.0, "input": True},
    "long": {"duration_s": 600.0, "input": False},
}
SMOKE_MODE_DEFAULT_DURATION_S = 10.0
INPUT_START_S = 5.0
INPUT_PERIOD_S = 1.0
INPUT_HOLD_S = 0.1
DEFAULT_FRAME_INTERVAL_S = 1.0
DEFAULT_CAPTURE_INTERVAL_SEC = 2.0
CANVAS_MIN_X = 410
DEFAULT_POLL_INTERVAL_S = 0.5
SERVER_READY_TIMEOUT_S = 15.0
CDP_READY_TIMEOUT_S = 15.0

COPY_CHUNK_BYTES = 256 * 1024

ALLOWED_INPUT_SCRIPT_PRESETS: dict[str, str] = {
    "start-screen": "start-screen",
    "menu-flow": "menu-flow",
    "calibration-skip": "calibration-skip",
    "intro-watch": "intro-watch",
    "legal-notices": "legal-notices",
    "chrome-menu-flow": "chrome-menu-flow",
    "chrome-full-boot": "chrome-full-boot",
    "menu-to-ingame": "menu-to-ingame",
}
INPUT_SCRIPT_ALIASES: dict[str, str] = ALLOWED_INPUT_SCRIPT_PRESETS
_SAFE_INPUT_SCRIPT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
MAX_GPU_ADAPTER_LEN = 96
_WGPU_ADAPTER_LOG_RE = re.compile(r"\[wgpu_worker\]\s+WebGPU adapter:\s*([^\r\n]+)", re.IGNORECASE)
_LOG_ELAPSED_PREFIX_RE = re.compile(r"^\s*\[\s*(\d+(?:\.\d+)?)\s*s\s*\]")
_FIRST_VISIBLE_FRAME_LOG_RE = re.compile(
    r"\bFIRST_VISIBLE_FRAME\b|\bfirst[\s_-]*(?:visible|non[\s_-]*black)[\s_-]*frame\b",
    re.IGNORECASE,
)
_FIRST_VISIBLE_FRAME_MS_RE = re.compile(
    r"\b(?:first_visible_frame_ms|firstVisibleFrameMs)\s*[=:]\s*(\d+(?:\.\d+)?)\b"
)
_FIRST_VISIBLE_FRAME_S_RE = re.compile(
    r"\b(?:first_visible_frame_s|firstVisibleFrameS)\s*[=:]\s*(\d+(?:\.\d+)?)\b"
)
_MILESTONE_FIRST_VISIBLE_AT_RE = re.compile(
    r"\bFIRST_VISIBLE_FRAME\b[^\r\n]{0,64}?\bat\s+(\d+(?:\.\d+)?)\s*(ms|s)\b",
    re.IGNORECASE,
)
_BLACK_LOADING_MS_RE = re.compile(
    r"\b(?:black_loading_ms|blackLoadingMs)\s*[=:]\s*(\d+(?:\.\d+)?)\b"
)
_BLACK_LOADING_S_RE = re.compile(
    r"\b(?:black_loading_s|blackLoadingS)\s*[=:]\s*(\d+(?:\.\d+)?)\b"
)
_AUDIO_RING_CONTEXT_RE = re.compile(
    r"audio_worker|audio_ring|guest[\s_-]*pcm[\s_-]*ring|guestRing|guestFramesDrained|playgta6_audio_ring",
    re.IGNORECASE,
)
_AUDIO_RING_ATTACHED_KV_RE = re.compile(
    r"\b(?:audio_ring_attached|audioRingAttached|guest_ring_attached|guestRingAttached|pcm_ring_attached|pcmRingAttached)\s*[=:]\s*(true|false|yes|no|1|0)\b",
    re.IGNORECASE,
)
_AUDIO_RING_ATTACHED_EVENT_RE = re.compile(
    r"\b(?:attach-guest-pcm-ring|guest[\s_-]*pcm[\s_-]*ring[\s_-]*attached|playgta6_audio_ring[^\r\n]{0,40}\battached)\b",
    re.IGNORECASE,
)
_AUDIO_RING_ATTACH_FAIL_RE = re.compile(
    r"\b(?:failed|error|detached|not\s+attached|false)\b",
    re.IGNORECASE,
)
_AUDIO_RING_DETACHED_EVENT_RE = re.compile(
    r"\b(?:detach-guest-pcm-ring|guest[\s_-]*pcm[\s_-]*ring[\s_-]*detached)\b",
    re.IGNORECASE,
)
_AUDIO_GUEST_FRAMES_DRAINED_KV_RE = re.compile(
    r"\b(?:audio_frames_drained|audioFramesDrained|guest_frames_drained|guestFramesDrained)\s*[=:]\s*(\d+)\b"
)
_AUDIO_GENERIC_FRAMES_DRAINED_KV_RE = re.compile(
    r"\b(?:frames_drained|framesDrained)\s*[=:]\s*(\d+)\b"
)
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_IEND_CHUNK = b"\x00\x00\x00\x00IEND\xaeB`\x82"

BOOT_OPTIONS_NAME = "boot-options.json"
DEFAULT_DUMP_PIPELINES_DIR = "/pg6_dump"
BOOT_OPTIONS_UINT_KEYS: frozenset[str] = frozenset(
    {
        "max_wall_ms",
        "max_dispatches",
        "arena_log_limit",
        "import_trace_limit",
    }
)
BOOT_OPTIONS_BOOL_KEYS: frozenset[str] = frozenset(
    {
        "report_to_log",
        "verify_window_marks",
    }
)
BOOT_OPTIONS_STR_KEYS: frozenset[str] = frozenset(
    {
        "dump_pipelines_dir",
    }
)
BOOT_OPTIONS_ALLOWED_KEYS: frozenset[str] = (
    BOOT_OPTIONS_UINT_KEYS | BOOT_OPTIONS_BOOL_KEYS | BOOT_OPTIONS_STR_KEYS
)
_UINT64_MAX = (1 << 64) - 1

_UNC_BS_PATH_RE = re.compile(r"\\\\[^\s\"'<>|,;()\[\]]+")
_UNC_FS_PATH_RE = re.compile(r"(?<!:)//[^\s\"'<>|,;()\[\]]+")
_WINDOWS_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s\"'<>|,;()\[\]]*")
_POSIX_PATH_RE = re.compile(
    r"/(?:Users|home|tmp|var|opt|root|mnt|private|etc|usr|Library|Volumes)/[^\s\"'<>|,;()\[\]]*"
)

CommandRunner = Callable[[list[str], Path, Path | None, float | None], int]
WhichFn = Callable[[str], str | None]


class RunnerError(Exception):
    """Usage, environment, or build failure (exit code 2)."""


class CdpError(Exception):
    """Chrome DevTools Protocol transport or command failure."""


# ---------------------------------------------------------------------------
# Sanitization (zero host paths / usernames in artifacts and dry-run output)
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
    """Strip Windows/POSIX/UNC host paths and active usernames."""
    if raw is None:
        return ""
    text = str(raw)
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


def sanitize_cmd(cmd: Sequence[str], repo_root: Path) -> list[str]:
    """Sanitize a command argv list so host paths become repo-relative labels or basenames."""
    out: list[str] = []
    resolved_repo = repo_root.resolve()
    for idx, token in enumerate(cmd):
        text = str(token)
        if idx == 0 and (os.sep in text or "/" in text or "\\" in text):
            out.append(basename_label(text))
            continue
        if text.startswith("-"):
            if "=" in text:
                flag, val = text.split("=", 1)
                if _looks_like_host_path(val):
                    out.append(f"{flag}={repo_relative_label(Path(val), resolved_repo) or basename_label(val)}")
                else:
                    out.append(sanitize_text(text))
            else:
                out.append(sanitize_text(text))
            continue
        if _looks_like_host_path(text):
            out.append(repo_relative_label(Path(text), resolved_repo) or basename_label(text))
        else:
            out.append(sanitize_text(text))
    return out


def _looks_like_host_path(text: str) -> bool:
    if not text:
        return False
    if text.startswith(("http://", "https://", "ws://", "wss://")):
        return False
    if re.match(r"^[A-Za-z]:[\\/]", text) or text.startswith("\\\\"):
        return True
    if _POSIX_PATH_RE.match(text):
        return True
    return False


# ---------------------------------------------------------------------------
# Subprocess & filesystem helpers
# ---------------------------------------------------------------------------


def run_command(cmd: list[str], cwd: Path, log: Path | None = None, timeout: float | None = None) -> int:
    """Run `cmd`; with `log`, stdout and stderr go there, otherwise to this console."""
    try:
        if log is None:
            return subprocess.run(cmd, cwd=cwd, timeout=timeout).returncode
        with log.open("wb") as fh:
            return subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        return 124
    except OSError as exc:
        raise RunnerError(f"cannot run {cmd[0]}: {exc}") from exc


def load_dotenv_value(repo_root: Path, key: str) -> str | None:
    """Read one key from `<repo_root>/.env` if present."""
    env_file = repo_root / ".env"
    if not env_file.is_file():
        return None
    try:
        content = env_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key:
            val = v.strip().strip('"').strip("'")
            return val or None
    return None


def resolve_against_repo(raw_path: Path | str, repo_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        return (repo_root / path).resolve()
    return path.resolve()


def newest_mtime(root: Path, files: Sequence[Path] = ()) -> float:
    """Newest mtime among build-relevant sources below `root` (and `files`); 0 if none."""
    newest = 0.0
    for path in files:
        if path.is_file():
            try:
                newest = max(newest, path.stat().st_mtime)
            except OSError:
                pass
    if root.is_dir():
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith(("build", "target", ".")) and d != "node_modules"
            ]
            for name in filenames:
                if Path(name).suffix in SOURCE_SUFFIXES:
                    try:
                        newest = max(newest, os.stat(os.path.join(dirpath, name)).st_mtime)
                    except OSError:
                        pass
    return newest


def mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def workspace_dirs(repo: Path) -> list[Path]:
    """Crate directories of the cargo workspace (root Cargo.toml `members`), else `recomp`."""
    manifest = repo / "Cargo.toml"
    members: list[str] = []
    if manifest.is_file():
        text = manifest.read_text(encoding="utf-8", errors="replace")
        block = re.search(r"members\s*=\s*\[(.*?)\]", text, re.S)
        if block:
            members = re.findall(r'"([^"]+)"', block.group(1))
    return [repo / m for m in members if "*" not in m] or [repo / "recomp"]


def atomic_copy_file(src: Path, dst: Path) -> None:
    """Atomically copy `src` to `dst` via a same-directory temp file and `os.replace`."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_stat = src.stat()
    fd, tmp_str = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=dst.parent)
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
            os.utime(tmp_path, ns=(src_stat.st_atime_ns, src_stat.st_mtime_ns))
        except OSError:
            pass
        os.replace(tmp_path, dst)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def find_free_port(host: str = "127.0.0.1") -> int:
    """Allocate an ephemeral free TCP port on `host`."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def terminate_process(proc: Any, wait_s: float = 5.0) -> int | None:
    """Gracefully terminate a subprocess, escalating to kill() if needed."""
    if proc is None:
        return None
    try:
        if proc.poll() is not None:
            return proc.returncode
        proc.terminate()
        try:
            return proc.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                return proc.wait(timeout=wait_s)
            except subprocess.TimeoutExpired:
                return proc.poll()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Staleness & build planning
# ---------------------------------------------------------------------------


class Plan:
    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []

    def add(self, stage: str, reason: str, cmd: list[str]) -> None:
        self.steps.append({"stage": stage, "reason": reason, "cmd": cmd})

    def has(self, stage: str) -> bool:
        return any(s["stage"] == stage for s in self.steps)

    def sanitized(self, repo_root: Path) -> list[dict[str, Any]]:
        return [
            {
                "stage": s["stage"],
                "reason": sanitize_text(s["reason"]),
                "cmd": sanitize_cmd(s["cmd"], repo_root),
            }
            for s in self.steps
        ]


def make_build_plan(
    *,
    repo: Path,
    build_dir: Path,
    boot_dir: Path,
    out_dir: Path,
    exe: Path | None,
    game_dir: Path | None,
    runtime_lib_override: Path | None,
    jobs: int,
    force: bool,
    smoke: bool,
) -> tuple[Plan, Path]:
    """Compute the staleness build plan for browser Wasm64 artifacts.

    Returns `(plan, runtime_lib_path)`.
    """
    runtime_pic_dir = build_dir / "runtime-pic"
    runtime_lib = runtime_lib_override if runtime_lib_override is not None else (runtime_pic_dir / "libplaygta6_core.a")
    plan = Plan()
    if smoke:
        return plan, runtime_lib

    def stale(target: Path, reasons: list[tuple[str, float]]) -> str | None:
        if force:
            return "forced (--rebuild)"
        t = mtime(target)
        if t is None:
            return f"{target.name} is missing"
        for what, newest in reasons:
            if newest > t:
                return f"{what} is newer than {target.name}"
        return None

    recomp = repo / "target" / "release" / ("playgta6-recomp" + EXE_SUFFIX)
    manifest = boot_dir / "program-manifest.json"
    manifest_time = mtime(manifest)

    # Check recomp + program only when exe is given, manifest is missing, or --rebuild is forced.
    check_program = force or manifest_time is None or exe is not None
    if check_program:
        sources = [(d / "Cargo.toml") for d in workspace_dirs(repo)] + [repo / "Cargo.toml", repo / "Cargo.lock"]
        newest_rs = max(
            [newest_mtime(d / "src") for d in workspace_dirs(repo)]
            + [0.0]
            + [mtime(p) or 0.0 for p in sources]
        )
        recomp_reason = stale(recomp, [("the recompiler sources", newest_rs)])
        if recomp_reason:
            plan.add("recomp", recomp_reason, ["cargo", "build", "--release", "-p", "playgta6-recomp"])

        exe_time = mtime(exe) if exe else 0.0
        prog_reason = stale(
            manifest,
            [("the recompiler", mtime(recomp) or 0.0), ("the exe", exe_time or 0.0)],
        )
        if prog_reason is None and plan.has("recomp"):
            prog_reason = "the recompiler is rebuilt"
        if prog_reason:
            if exe is None:
                raise RunnerError(
                    "program-manifest.json is missing or forced to rebuild, so --exe or PLAYGTA6_EXE "
                    "is required (or pass --smoke / --trust-build)"
                )
            prog_cmd = [
                str(recomp),
                "build-program",
                str(exe),
                "--out",
                str(boot_dir),
                "--target",
                "wasm64",
                "--memory-model",
                "window-runs",
                "--indirect-dispatch",
                "shard",
                "--jobs",
                str(jobs),
            ]
            if game_dir is not None:
                prog_cmd += ["--dll-dir", str(game_dir)]
            plan.add("program", prog_reason, prog_cmd)

    # Stage 2: Release PIC libplaygta6_core.a
    runtime_newest = newest_mtime(repo / "runtime")
    if runtime_lib_override is None:
        cmake_cache = runtime_pic_dir / "CMakeCache.txt"
        cfg_reason = "forced (--rebuild)" if force else (
            "CMakeCache.txt is missing" if mtime(cmake_cache) is None else None
        )
        if cfg_reason:
            plan.add(
                "runtime_configure",
                cfg_reason,
                [
                    "emcmake",
                    "cmake",
                    "-S",
                    str(repo / "runtime"),
                    "-B",
                    str(runtime_pic_dir),
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_CXX_FLAGS=-sMEMORY64=1 -pthread -fPIC",
                    "-DCMAKE_C_FLAGS=-sMEMORY64=1 -pthread -fPIC",
                ],
            )
        lib_reason = stale(runtime_lib, [("the runtime sources", runtime_newest)])
        if lib_reason is None and plan.has("runtime_configure"):
            lib_reason = "CMake was reconfigured"
        if lib_reason:
            plan.add(
                "runtime_lib",
                lib_reason,
                [
                    "cmake",
                    "--build",
                    str(runtime_pic_dir),
                    "--target",
                    "playgta6_core",
                    "--parallel",
                    str(jobs),
                ],
            )

    # Stage 3: Guest Wasm64 + asymmetric split side modules via build_guest_wasm.py
    runtime_wasm = out_dir / "runtime.wasm"
    runtime_js = out_dir / "runtime.js"
    build_guest_script = repo / "tools" / "build_guest_wasm.py"
    wasm_inputs = [
        ("program-manifest.json", mtime(manifest) or 0.0),
        ("libplaygta6_core.a", mtime(runtime_lib) or 0.0),
        ("runtime sources or build_guest_wasm.py", newest_mtime(repo / "runtime", [build_guest_script])),
    ]
    wasm_reason = stale(runtime_wasm, wasm_inputs) or stale(runtime_js, wasm_inputs)
    if wasm_reason is None and (plan.has("program") or plan.has("runtime_lib")):
        wasm_reason = "an upstream stage is rebuilt"
    if wasm_reason:
        wasm_cmd = [
            sys.executable,
            str(build_guest_script),
            str(boot_dir),
            "--out-dir",
            str(out_dir),
            "--cache-dir",
            str(build_dir / "wasm_obj_cache"),
            "--runtime-lib",
            str(runtime_lib),
            "--jobs",
            str(jobs),
            "--opt",
            "O2",
            "--link-guest-boot",
            "--browser-boot-exports",
            "--split-modules",
            "2",
            "--split-strategy",
            "asymmetric",
            "--proxy-to-pthread",
            "--strip",
            "--no-link-opt",
        ]
        plan.add("guest_wasm", wasm_reason, wasm_cmd)

    return plan, runtime_lib


# ---------------------------------------------------------------------------
# Shader pack + reflection sidecar staging
# ---------------------------------------------------------------------------


def plan_shader_staging(shader_pack: Path | None, stage_dir: Path) -> list[dict[str, str]]:
    """Plan atomic staging of `--shader-pack` plus `<pack>.reflect.json` and `reflection-manifest.json`."""
    if shader_pack is None:
        return []
    entries: list[dict[str, str]] = [
        {"kind": "shader_pack", "src": str(shader_pack), "dst": str(stage_dir / shader_pack.name), "name": shader_pack.name}
    ]
    # Primary sidecar convention (tools/stage_reflection_sidecar.py): `<pack>.reflect.json`
    sidecar_primary = Path(str(shader_pack) + ".reflect.json")
    if sidecar_primary.is_file():
        entries.append(
            {
                "kind": "stage_reflection",
                "src": str(sidecar_primary),
                "dst": str(stage_dir / sidecar_primary.name),
                "name": sidecar_primary.name,
            }
        )
    else:
        alt = shader_pack.with_suffix(".reflect.json")
        if alt != sidecar_primary and alt.is_file():
            entries.append(
                {
                    "kind": "stage_reflection",
                    "src": str(alt),
                    "dst": str(stage_dir / alt.name),
                    "name": alt.name,
                }
            )
    manifest = shader_pack.parent / "reflection-manifest.json"
    if manifest.is_file():
        entries.append(
            {
                "kind": "reflection_manifest",
                "src": str(manifest),
                "dst": str(stage_dir / manifest.name),
                "name": manifest.name,
            }
        )
    return entries


def stage_shader_pack(shader_pack: Path, stage_dir: Path) -> list[dict[str, Any]]:
    """Stage shader pack and sibling reflection sidecars into `stage_dir`."""
    if not shader_pack.is_file():
        raise RunnerError(f"shader pack not found: {basename_label(shader_pack)}")
    stage_dir.mkdir(parents=True, exist_ok=True)
    planned = plan_shader_staging(shader_pack, stage_dir)
    staged: list[dict[str, Any]] = []
    for item in planned:
        src = Path(item["src"])
        dst = Path(item["dst"])
        needs_copy = True
        if dst.is_file():
            try:
                if src.resolve() == dst.resolve():
                    needs_copy = False
                else:
                    s_st, d_st = src.stat(), dst.stat()
                    if s_st.st_size == d_st.st_size and int(s_st.st_mtime) == int(d_st.st_mtime):
                        needs_copy = False
            except OSError:
                needs_copy = True
        if needs_copy:
            atomic_copy_file(src, dst)
        staged.append({"kind": item["kind"], "name": item["name"], "copied": needs_copy})
    return staged


def atomic_write_bytes(payload: bytes, dst: Path) -> None:
    """Atomically write `payload` to `dst` via a same-directory temp file and `os.replace`."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=dst.parent)
    tmp_path = Path(tmp_str)
    try:
        with os.fdopen(fd, "wb") as dst_fh:
            dst_fh.write(payload)
            dst_fh.flush()
        os.replace(tmp_path, dst)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# URL query options (--input-script, --stream-shaders, --stream-threshold-mb)
# & validated boot-options.json staging
# ---------------------------------------------------------------------------


def sanitize_input_script_preset(raw: Any) -> str | None:
    """Sanitize and canonicalize an input-script preset against the allowlist.

    Matches `sanitizeInputScriptPreset` in `web/app_bootstrap.js`.
    """
    if not isinstance(raw, str):
        return None
    trimmed = raw.strip()
    if not trimmed or len(trimmed) > 64 or any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in trimmed):
        return None
    lower = trimmed.lower()
    if not _SAFE_INPUT_SCRIPT_RE.match(lower):
        return None
    return ALLOWED_INPUT_SCRIPT_PRESETS.get(lower)


def validate_input_script_preset(raw: str | None) -> str | None:
    """Validate `--input-script`; returns None if unset or raises RunnerError if disallowed."""
    if raw is None:
        return None
    cleaned = sanitize_input_script_preset(raw)
    if cleaned is None:
        allowed = ", ".join(sorted(ALLOWED_INPUT_SCRIPT_PRESETS.values()))
        raise RunnerError(f"invalid --input-script {sanitize_text(str(raw))!r} (allowed: {allowed})")
    return cleaned


def validate_stream_threshold_mb(raw: float | int | None) -> float | None:
    """Validate `--stream-threshold-mb` as a positive finite number."""
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise RunnerError("--stream-threshold-mb must be a positive finite number")
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise RunnerError("--stream-threshold-mb must be a positive finite number")
    return value


def format_stream_threshold_mb(value: float) -> str:
    """Format a positive finite MiB threshold as a plain decimal string (no exponent)."""
    if value == int(value):
        return str(int(value))
    text = f"{value:.10f}".rstrip("0").rstrip(".")
    if not text or text == "0":
        text = f"{value:.16f}".rstrip("0").rstrip(".")
    return text


def build_target_url(
    host: str,
    port: int | str,
    *,
    smoke: bool = False,
    input_script: str | None = None,
    stream_shaders: bool = False,
    stream_threshold_mb: float | None = None,
    hide_hud: bool = False,
) -> str:
    """Construct the local browser boot URL with sanitized query parameters."""
    params: list[tuple[str, str]] = []
    if smoke:
        params.append(("smoke", "1"))
    clean_script = validate_input_script_preset(input_script)
    if clean_script is not None:
        params.append(("inputScript", clean_script))
    if stream_shaders:
        params.append(("streamShaders", "1"))
    clean_threshold = validate_stream_threshold_mb(stream_threshold_mb)
    if clean_threshold is not None:
        params.append(("streamThresholdMb", format_stream_threshold_mb(clean_threshold)))
    if hide_hud:
        params.append(("hud", "0"))
    base = f"http://{host}:{port}/"
    if not params:
        return base
    return f"{base}?{urlencode(params)}"


def validate_mp4_fps(raw: float | int | None) -> float | None:
    """Validate `--mp4-fps` as an optional positive finite number."""
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise RunnerError("--mp4-fps must be a positive finite number")
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise RunnerError("--mp4-fps must be a positive finite number")
    return value


def validate_capture_interval_sec(raw: float | int) -> float:
    """Validate `--capture-interval-sec` as a positive finite number."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise RunnerError("--capture-interval-sec must be a positive finite number")
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise RunnerError("--capture-interval-sec must be a positive finite number")
    return value


def _validate_untracked_output_dir(target_dir: Path, repo_root: Path, artifact_name: str) -> None:
    """Refuse writing artifacts into tracked repository directories."""
    resolved_repo = repo_root.resolve()
    resolved_target = target_dir.resolve()
    if not resolved_target.is_relative_to(resolved_repo):
        return
    allowed_roots = (
        resolved_repo / "work",
        resolved_repo / "local-data",
    )
    if any(resolved_target.is_relative_to(allowed) for allowed in allowed_roots):
        return
    label = repo_relative_label(resolved_target, resolved_repo) or "<repo>"
    raise RunnerError(
        f"refusing to write {artifact_name} into tracked repository directory '{label}' "
        "(must be under work/, local-data/, or outside the repository)"
    )


def validate_boot_dir_safety(boot_dir: Path, repo_root: Path) -> None:
    """Refuse writing `boot-options.json` into tracked repository directories."""
    _validate_untracked_output_dir(boot_dir, repo_root, BOOT_OPTIONS_NAME)


def validate_capture_frames_dir_safety(capture_dir: Path, repo_root: Path) -> None:
    """Refuse writing captured frames into tracked repository directories."""
    _validate_untracked_output_dir(capture_dir, repo_root, "captured frames")


def validate_capture_frames_arg(raw: Any, repo_root: Path) -> tuple[bool, Path | None]:
    """Validate `--capture-frames [DIR]` and return `(enabled, capture_dir_or_none)`."""
    if raw is None or raw is False:
        return False, None
    if raw is True:
        return True, None
    if isinstance(raw, (str, Path)):
        text = str(raw).strip()
        if not text:
            raise RunnerError("--capture-frames directory path must not be empty")
        capture_dir = resolve_against_repo(text, repo_root)
        validate_capture_frames_dir_safety(capture_dir, repo_root)
        if capture_dir.exists() and not capture_dir.is_dir():
            raise RunnerError(f"--capture-frames path is not a directory: {basename_label(capture_dir)}")
        return True, capture_dir
    raise RunnerError("invalid --capture-frames value")


def _validate_dump_pipelines_dir(value: Any) -> str:
    if not isinstance(value, str):
        raise RunnerError("invalid boot-options: 'dump_pipelines_dir' must be a non-empty string")
    cleaned = value.strip()
    if (
        not cleaned
        or len(cleaned) > 256
        or any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in cleaned)
        or "\\" in cleaned
        or re.match(r"^[A-Za-z]:", cleaned)
        or ".." in cleaned.split("/")
    ):
        raise RunnerError(
            "invalid boot-options: 'dump_pipelines_dir' must be a safe non-empty virtual path (e.g. '/pg6_dump')"
        )
    return cleaned


def parse_and_validate_boot_options(
    *,
    boot_options: Mapping[str, Any] | str | Path | None = None,
    report_to_log: bool = False,
    max_wall_ms: int | None = None,
    dump_pipelines: bool = False,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any] | None:
    """Parse and schema-validate `--boot-options`, `--report-to-log`, `--max-wall-ms`, and `--dump-pipelines`."""
    if boot_options is None and not report_to_log and max_wall_ms is None and not dump_pipelines:
        return None

    resolved_repo = repo_root.resolve()
    raw_obj: Any = {}

    if boot_options is not None:
        if isinstance(boot_options, Mapping):
            raw_obj = dict(boot_options)
        elif isinstance(boot_options, Path):
            try:
                raw_text = boot_options.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeDecodeError) as exc:
                raise RunnerError(
                    f"failed to read boot-options file {basename_label(boot_options)}: {sanitize_text(str(exc))}"
                ) from exc
            try:
                raw_obj = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                raise RunnerError(
                    f"invalid JSON in boot-options file {basename_label(boot_options)} "
                    f"(line {exc.lineno}, col {exc.colno}): {exc.msg}"
                ) from exc
        elif isinstance(boot_options, str):
            stripped = boot_options.strip()
            if not stripped:
                raise RunnerError("invalid --boot-options: value must be a JSON object or file path")
            candidate_path = (
                None
                if stripped.startswith(("{", "[", '"'))
                else resolve_against_repo(boot_options, resolved_repo)
            )
            looks_like_json = (
                stripped.startswith(("{", "[", '"'))
                or stripped in ("true", "false", "null")
                or (
                    stripped[0] in "-0123456789"
                    and (candidate_path is None or not candidate_path.is_file())
                )
            )
            if looks_like_json:
                try:
                    raw_obj = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise RunnerError(
                        f"invalid inline --boot-options JSON (line {exc.lineno}, col {exc.colno}): {exc.msg}"
                    ) from exc
            else:
                options_path = candidate_path if candidate_path is not None else resolve_against_repo(boot_options, resolved_repo)
                if not options_path.is_file():
                    raise RunnerError(
                        f"boot-options JSON file not found: {repo_relative_label(options_path, resolved_repo)}"
                    )
                try:
                    raw_text = options_path.read_text(encoding="utf-8-sig")
                except (OSError, UnicodeDecodeError) as exc:
                    raise RunnerError(
                        f"failed to read boot-options file {basename_label(options_path)}: {sanitize_text(str(exc))}"
                    ) from exc
                try:
                    raw_obj = json.loads(raw_text)
                except json.JSONDecodeError as exc:
                    raise RunnerError(
                        f"invalid JSON in boot-options file {basename_label(options_path)} "
                        f"(line {exc.lineno}, col {exc.colno}): {exc.msg}"
                    ) from exc
        else:
            raise RunnerError("invalid --boot-options: expected JSON object, JSON string, or file path")

    if not isinstance(raw_obj, dict):
        raise RunnerError("invalid boot-options: top-level JSON value must be an object")

    validated: dict[str, Any] = {}
    for raw_key, value in raw_obj.items():
        key = str(raw_key)
        if key not in BOOT_OPTIONS_ALLOWED_KEYS:
            allowed = ", ".join(sorted(BOOT_OPTIONS_ALLOWED_KEYS))
            raise RunnerError(
                f"invalid boot-options: unsupported key '{sanitize_text(key)}' (allowed: {allowed})"
            )
        if key in BOOT_OPTIONS_UINT_KEYS:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > _UINT64_MAX:
                raise RunnerError(f"invalid boot-options: '{key}' must be a non-negative uint64 integer")
            validated[key] = int(value)
        elif key in BOOT_OPTIONS_BOOL_KEYS:
            if not isinstance(value, bool):
                raise RunnerError(f"invalid boot-options: '{key}' must be a boolean")
            validated[key] = value
        elif key in BOOT_OPTIONS_STR_KEYS:
            validated[key] = _validate_dump_pipelines_dir(value)

    if report_to_log:
        validated["report_to_log"] = True

    if max_wall_ms is not None:
        if isinstance(max_wall_ms, bool) or not isinstance(max_wall_ms, int) or max_wall_ms < 0 or max_wall_ms > _UINT64_MAX:
            raise RunnerError("--max-wall-ms must be a non-negative uint64 integer")
        validated["max_wall_ms"] = int(max_wall_ms)

    if dump_pipelines:
        validated["dump_pipelines_dir"] = DEFAULT_DUMP_PIPELINES_DIR

    return {key: validated[key] for key in sorted(validated)}


def serialize_boot_options(options: Mapping[str, Any]) -> bytes:
    """Serialize validated boot options to deterministic UTF-8 JSON bytes."""
    return (json.dumps(dict(options), indent=2, sort_keys=True) + "\n").encode("utf-8")


def stage_boot_options_file(
    options: Mapping[str, Any],
    boot_dir: Path,
    *,
    repo_root: Path = REPO_ROOT,
    dry_run: bool = False,
    check_safety: bool = True,
) -> dict[str, Any]:
    """Validate target directory safety and atomically stage `boot-options.json`."""
    if check_safety:
        validate_boot_dir_safety(boot_dir, repo_root)

    ordered = {key: options[key] for key in sorted(options)}
    payload = serialize_boot_options(ordered)
    digest_hex = hashlib.sha256(payload).hexdigest()
    dst_file = boot_dir / BOOT_OPTIONS_NAME

    needs_write = True
    if dst_file.exists() and not dst_file.is_symlink() and dst_file.is_file():
        try:
            if dst_file.read_bytes() == payload:
                needs_write = False
        except OSError:
            needs_write = True

    if needs_write and not dry_run:
        try:
            atomic_write_bytes(payload, dst_file)
        except OSError as exc:
            raise RunnerError(f"failed to stage {BOOT_OPTIONS_NAME}: {sanitize_text(str(exc))}") from exc

    if not needs_write:
        action = "unchanged"
        staged = True
    elif dry_run:
        action = "plan_write"
        staged = False
    else:
        action = "written"
        staged = True

    return sanitize_value(
        {
            "filename": BOOT_OPTIONS_NAME,
            "staged": staged,
            "action": action,
            "size_bytes": len(payload),
            "sha256": digest_hex,
            "options": ordered,
        }
    )


# ---------------------------------------------------------------------------
# Chrome / Chromium / Edge discovery
# ---------------------------------------------------------------------------


def discover_chrome(
    explicit_path: Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    which_fn: WhichFn = shutil.which,
    platform: str | None = None,
    exists_fn: Callable[[Path], bool] | None = None,
) -> Path | None:
    """Locate Chrome, Chromium, or Edge across Windows, Linux, and macOS."""
    env_map = os.environ if env is None else env
    plat = sys.platform if platform is None else platform
    is_file = (lambda p: p.is_file()) if exists_fn is None else exists_fn

    if explicit_path is not None:
        p = Path(explicit_path).expanduser()
        return p if is_file(p) else None

    for env_key in ("CHROME_PATH", "PLAYGTA6_CHROME"):
        raw = (env_map.get(env_key) or "").strip()
        if raw:
            p = Path(raw).expanduser()
            if is_file(p):
                return p

    if plat == "win32":
        which_names = ("chrome", "chrome.exe", "msedge", "msedge.exe", "chromium", "chromium.exe")
    elif plat == "darwin":
        which_names = ("google-chrome", "chromium", "microsoft-edge")
    else:
        which_names = (
            "google-chrome",
            "google-chrome-stable",
            "chromium",
            "chromium-browser",
            "microsoft-edge",
            "microsoft-edge-stable",
        )

    for name in which_names:
        found = which_fn(name)
        if found:
            p = Path(found)
            if is_file(p):
                return p

    candidates: list[Path] = []
    if plat == "win32":
        roots = [
            env_map.get("PROGRAMFILES", r"C:\Program Files"),
            env_map.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
            env_map.get("LOCALAPPDATA", ""),
        ]
        rel_paths = (
            Path("Google") / "Chrome" / "Application" / "chrome.exe",
            Path("Microsoft") / "Edge" / "Application" / "msedge.exe",
            Path("Chromium") / "Application" / "chrome.exe",
        )
        for root in roots:
            if not root:
                continue
            for rel in rel_paths:
                candidates.append(Path(root) / rel)
    elif plat == "darwin":
        candidates.extend(
            [
                Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
                Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
                Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
            ]
        )
    else:
        candidates.extend(
            [
                Path("/usr/bin/google-chrome"),
                Path("/usr/bin/google-chrome-stable"),
                Path("/usr/bin/chromium"),
                Path("/usr/bin/chromium-browser"),
                Path("/usr/bin/microsoft-edge"),
                Path("/snap/bin/chromium"),
            ]
        )

    for candidate in candidates:
        if is_file(candidate):
            return candidate
    return None


def build_chrome_args(
    chrome_path: Path,
    *,
    user_data_dir: Path,
    cdp_port: int,
    headless: bool,
    initial_url: str = "about:blank",
) -> list[str]:
    """Construct deterministic Chrome launch flags for WebGPU + CDP testing."""
    cmd = [
        str(chrome_path),
        f"--user-data-dir={user_data_dir}",
        f"--remote-debugging-port={cdp_port}",
        "--enable-unsafe-webgpu",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-timer-throttling",
        "--disable-backgrounding-occluded-windows",
        "--disable-renderer-backgrounding",
        "--autoplay-policy=no-user-gesture-required",
        "--window-size=1280,720",
    ]
    if headless:
        cmd.append("--headless=new")
    cmd.append(initial_url)
    return cmd


# ---------------------------------------------------------------------------
# Local server readiness helper
# ---------------------------------------------------------------------------


def build_server_cmd(
    *,
    repo: Path,
    host: str,
    port: int,
    web_dir: Path,
    data_dir: Path,
    boot_dir: Path | None,
    game_dir: Path | None,
    shaders_dir: Path | None,
    quiet: bool = False,
) -> list[str]:
    """Build the `server/serve_local.py` launch command."""
    cmd = [
        sys.executable,
        str(repo / "server" / "serve_local.py"),
        "--host",
        host,
        "--port",
        str(port),
        "--web",
        str(web_dir),
        "--data",
        str(data_dir),
    ]
    if boot_dir is not None:
        cmd += ["--boot", str(boot_dir)]
    if game_dir is not None:
        cmd += ["--game", str(game_dir)]
    if shaders_dir is not None:
        cmd += ["--shaders", str(shaders_dir)]
    if quiet:
        cmd.append("--quiet")
    return cmd


def wait_for_server_ready(
    host: str,
    port: int,
    *,
    timeout_s: float = SERVER_READY_TIMEOUT_S,
    proc: Any = None,
) -> dict[str, Any]:
    """Poll `/api/status` (or `/`) until HTTP 200 with COOP/COEP headers is observed."""
    deadline = time.monotonic() + timeout_s
    last_error = "no response"
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RunnerError(f"serve_local.py exited prematurely with code {proc.returncode}")
        for path in ("/api/status", "/"):
            conn: http.client.HTTPConnection | None = None
            try:
                conn = http.client.HTTPConnection(host, port, timeout=1.5)
                conn.request("GET", path)
                resp = conn.getresponse()
                body = resp.read()
                coop = (resp.getheader("Cross-Origin-Opener-Policy") or "").strip().lower()
                coep = (resp.getheader("Cross-Origin-Embedder-Policy") or "").strip().lower()
                if resp.status == 200:
                    if coop != "same-origin" or coep != "require-corp":
                        last_error = f"missing COOP/COEP headers (coop={coop!r}, coep={coep!r})"
                        continue
                    api_status: dict[str, Any] = {}
                    if path == "/api/status":
                        try:
                            parsed = json.loads(body.decode("utf-8", errors="replace"))
                            if isinstance(parsed, dict):
                                api_status = parsed
                        except json.JSONDecodeError:
                            pass
                    return {
                        "status_code": resp.status,
                        "path": path,
                        "coop": coop,
                        "coep": coep,
                        "api_status": api_status,
                    }
                last_error = f"GET {path} returned HTTP {resp.status}"
            except OSError as exc:
                last_error = str(exc)
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except OSError:
                        pass
        time.sleep(0.1)
    raise RunnerError(f"local server did not become ready within {timeout_s:.1f}s ({sanitize_text(last_error)})")


# ---------------------------------------------------------------------------
# Pure-Python stdlib RFC 6455 WebSocket + CDP client
# ---------------------------------------------------------------------------


def _ws_encode_frame(opcode: int, payload: bytes, mask: bool = True) -> bytes:
    header = bytearray()
    header.append(0x80 | (opcode & 0x0F))
    length = len(payload)
    mask_bit = 0x80 if mask else 0x00
    if length < 126:
        header.append(mask_bit | length)
    elif length <= 0xFFFF:
        header.append(mask_bit | 126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(mask_bit | 127)
        header.extend(struct.pack("!Q", length))
    if mask:
        mask_key = os.urandom(4)
        header.extend(mask_key)
        masked = bytes(b ^ mask_key[i & 3] for i, b in enumerate(payload))
        return bytes(header) + masked
    return bytes(header) + payload


class CdpClient:
    """Minimal pure-Python stdlib WebSocket client for Chrome DevTools Protocol."""

    def __init__(self, ws_url: str, timeout: float = 15.0) -> None:
        self.ws_url = ws_url
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._buf = bytearray()
        self._next_id = 1
        self._events: list[dict[str, Any]] = []
        self._closed = False

    def connect(self) -> None:
        parsed = urlsplit(self.ws_url)
        if parsed.scheme != "ws":
            raise CdpError(f"unsupported CDP WebSocket scheme: {parsed.scheme!r}")
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 80
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        sock = socket.create_connection((host, port), timeout=self.timeout)
        sock.settimeout(self.timeout)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        req = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            f"\r\n"
        ).encode("ascii")
        try:
            sock.sendall(req)
            raw = bytearray()
            while b"\r\n\r\n" not in raw:
                chunk = sock.recv(4096)
                if not chunk:
                    raise CdpError("CDP WebSocket closed during handshake")
                raw.extend(chunk)
                if len(raw) > 65536:
                    raise CdpError("CDP WebSocket handshake header too large")
            header_end = raw.index(b"\r\n\r\n") + 4
            header_text = raw[:header_end].decode("latin1")
            leftover = raw[header_end:]
            lines = header_text.split("\r\n")
            if not lines or " 101 " not in lines[0]:
                raise CdpError(f"CDP WebSocket handshake failed: {lines[0] if lines else 'empty'}")
            headers: dict[str, str] = {}
            for line in lines[1:]:
                if ":" in line:
                    k, v = line.split(":", 1)
                    headers[k.strip().lower()] = v.strip()
            expected_accept = base64.b64encode(
                hashlib.sha1((key + WEBSOCKET_GUID).encode("ascii")).digest()
            ).decode("ascii")
            if headers.get("sec-websocket-accept") != expected_accept:
                raise CdpError("CDP WebSocket handshake returned invalid Sec-WebSocket-Accept")
            self._sock = sock
            self._buf = bytearray(leftover)
            self._closed = False
        except Exception:
            try:
                sock.close()
            except OSError:
                pass
            raise

    def close(self) -> None:
        sock = self._sock
        self._sock = None
        if sock is None:
            return
        if not self._closed:
            try:
                sock.sendall(_ws_encode_frame(0x8, b"", mask=True))
            except OSError:
                pass
        self._closed = True
        try:
            sock.close()
        except OSError:
            pass

    def _read_exact(self, n: int, deadline: float | None) -> bytes:
        assert self._sock is not None
        while len(self._buf) < n:
            if deadline is not None:
                rem = deadline - time.monotonic()
                if rem <= 0:
                    raise TimeoutError("CDP WebSocket read timed out")
                self._sock.settimeout(rem)
            else:
                self._sock.settimeout(self.timeout)
            chunk = self._sock.recv(max(4096, n - len(self._buf)))
            if not chunk:
                self._closed = True
                raise CdpError("CDP WebSocket closed unexpectedly")
            self._buf.extend(chunk)
        data = bytes(self._buf[:n])
        del self._buf[:n]
        return data

    def _recv_message(self, timeout: float | None = None) -> str | None:
        if self._sock is None or self._closed:
            return None
        deadline = None if timeout is None else (time.monotonic() + timeout)
        fragments: list[bytes] = []
        message_opcode: int | None = None
        while True:
            if not self._buf and timeout is not None:
                rem = (deadline - time.monotonic()) if deadline is not None else timeout
                if rem <= 0:
                    return None
                ready, _, _ = select.select([self._sock], [], [], rem)
                if not ready:
                    return None
            head = self._read_exact(2, deadline)
            b0, b1 = head[0], head[1]
            fin = bool(b0 & 0x80)
            opcode = b0 & 0x0F
            masked = bool(b1 & 0x80)
            length = b1 & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._read_exact(2, deadline))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._read_exact(8, deadline))[0]
            mask_key = self._read_exact(4, deadline) if masked else b""
            payload = self._read_exact(length, deadline) if length else b""
            if masked and payload:
                payload = bytes(b ^ mask_key[i & 3] for i, b in enumerate(payload))

            if opcode == 0x8:  # close
                self._closed = True
                return None
            if opcode == 0x9:  # ping -> pong
                try:
                    self._sock.sendall(_ws_encode_frame(0xA, payload, mask=True))
                except OSError:
                    pass
                continue
            if opcode == 0xA:  # pong
                continue
            if opcode in (0x1, 0x2):
                message_opcode = opcode
                fragments = [payload]
            elif opcode == 0x0:
                fragments.append(payload)
            if fin and message_opcode is not None:
                full = b"".join(fragments)
                return full.decode("utf-8", errors="replace")

    def send(self, method: str, params: dict[str, Any] | None = None, timeout: float | None = None) -> dict[str, Any]:
        if self._sock is None or self._closed:
            raise CdpError("CDP WebSocket is not connected")
        msg_id = self._next_id
        self._next_id += 1
        payload = json.dumps({"id": msg_id, "method": method, "params": params or {}}).encode("utf-8")
        self._sock.sendall(_ws_encode_frame(0x1, payload, mask=True))
        eff_timeout = self.timeout if timeout is None else timeout
        deadline = time.monotonic() + eff_timeout
        while True:
            rem = deadline - time.monotonic()
            if rem <= 0:
                raise CdpError(f"CDP command {method} timed out")
            text = self._recv_message(timeout=rem)
            if text is None:
                raise CdpError(f"CDP WebSocket closed while awaiting {method}")
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(msg, dict):
                continue
            if msg.get("id") == msg_id:
                if "error" in msg:
                    raise CdpError(f"CDP {method} error: {msg['error']}")
                result = msg.get("result")
                return result if isinstance(result, dict) else {}
            if "method" in msg:
                self._events.append(msg)

    def drain_events(self, timeout: float = 0.0) -> list[dict[str, Any]]:
        """Return queued CDP events and optionally read new events for up to `timeout` seconds."""
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            rem = deadline - time.monotonic()
            if rem <= 0 and not self._buf:
                break
            try:
                text = self._recv_message(timeout=max(0.0, rem))
            except (TimeoutError, OSError, CdpError):
                break
            if text is None:
                break
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(msg, dict) and "method" in msg:
                self._events.append(msg)
            if timeout <= 0 and not self._buf:
                break
        drained = self._events
        self._events = []
        return drained


def wait_for_cdp_page_ws(
    cdp_port: int,
    *,
    timeout_s: float = CDP_READY_TIMEOUT_S,
    proc: Any = None,
) -> str:
    """Wait until Chrome exposes a page target on `http://127.0.0.1:<cdp_port>/json/list`."""
    deadline = time.monotonic() + timeout_s
    last_error = "no CDP response"
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RunnerError(f"Chrome exited prematurely with code {proc.returncode}")
        for endpoint in (f"http://127.0.0.1:{cdp_port}/json/list", f"http://127.0.0.1:{cdp_port}/json"):
            try:
                with urllib.request.urlopen(endpoint, timeout=1.5) as resp:
                    data = json.loads(resp.read().decode("utf-8", errors="replace"))
                if isinstance(data, list):
                    for target in data:
                        if (
                            isinstance(target, dict)
                            and target.get("type") == "page"
                            and isinstance(target.get("webSocketDebuggerUrl"), str)
                        ):
                            return target["webSocketDebuggerUrl"]
                last_error = "no page target yet"
            except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
                last_error = str(exc)
        time.sleep(0.15)
    raise RunnerError(f"Chrome CDP endpoint did not become ready within {timeout_s:.1f}s ({sanitize_text(last_error)})")


# ---------------------------------------------------------------------------
# Browser telemetry probe & CDP session execution
# ---------------------------------------------------------------------------

TELEMETRY_PROBE_JS = r"""(() => {
  const safeText = (id) => {
    const el = document.getElementById(id);
    return el ? String(el.textContent || '').trim() : null;
  };
  let bootState = null;
  try {
    if (window.__playgta6BootState && typeof window.__playgta6BootState === 'object') {
      bootState = JSON.parse(JSON.stringify(window.__playgta6BootState));
    } else if (window.playgta6App && typeof window.playgta6App.getBootTelemetry === 'function') {
      bootState = JSON.parse(JSON.stringify(window.playgta6App.getBootTelemetry()));
    }
  } catch {}
  let diagnostics = null;
  try {
    if (window.playgta6App && typeof window.playgta6App.getDiagnosticsSnapshot === 'function') {
      diagnostics = JSON.parse(JSON.stringify(window.playgta6App.getDiagnosticsSnapshot()));
    }
  } catch {}
  let tracerSummary = null;
  try {
    if (window.playgta6App && window.playgta6App.bootTracer && typeof window.playgta6App.bootTracer.getSummary === 'function') {
      tracerSummary = JSON.parse(JSON.stringify(window.playgta6App.bootTracer.getSummary()));
    }
  } catch {}
  let bootUrlOptions = null;
  try {
    if (window.playgta6App && window.playgta6App.bootUrlOptions && typeof window.playgta6App.bootUrlOptions === 'object') {
      bootUrlOptions = JSON.parse(JSON.stringify(window.playgta6App.bootUrlOptions));
    }
  } catch {}
  let inputScriptState = null;
  try {
    if (window.playgta6App && window.playgta6App.inputBridge && typeof window.playgta6App.inputBridge.getScriptState === 'function') {
      inputScriptState = JSON.parse(JSON.stringify(window.playgta6App.inputBridge.getScriptState()));
    }
  } catch {}
  let memory = null;
  try {
    const pm = typeof performance !== 'undefined' && performance && performance.memory;
    if (pm && typeof pm === 'object') {
      const used = Number(pm.usedJSHeapSize);
      const total = Number(pm.totalJSHeapSize);
      const limit = Number(pm.jsHeapSizeLimit);
      memory = {
        usedJSHeapSize: Number.isFinite(used) && used >= 0 ? used : null,
        totalJSHeapSize: Number.isFinite(total) && total >= 0 ? total : null,
        jsHeapSizeLimit: Number.isFinite(limit) && limit >= 0 ? limit : null,
      };
    }
  } catch {}
  let audioStats = null;
  try {
    const ae = window.playgta6App && window.playgta6App.audioEngine;
    if (ae && typeof ae.getStats === 'function') {
      audioStats = JSON.parse(JSON.stringify(ae.getStats()));
    } else if (ae && typeof ae.getTelemetry === 'function') {
      audioStats = JSON.parse(JSON.stringify(ae.getTelemetry()));
    }
  } catch {}
  const hudLog = [];
  try {
    const logRoot = document.getElementById('hud-log');
    if (logRoot) {
      for (const child of Array.from(logRoot.children).slice(-60)) {
        hudLog.push(String(child.textContent || '').trim());
      }
    }
  } catch {}
  return {
    href: String(location.href || ''),
    readyState: String(document.readyState || ''),
    capabilities: {
      crossOriginIsolated: self.crossOriginIsolated === true,
      sharedArrayBuffer: typeof SharedArrayBuffer === 'function',
      webgpu: Boolean(typeof navigator !== 'undefined' && 'gpu' in navigator && navigator.gpu),
    },
    bootState,
    diagnostics,
    tracerSummary,
    bootUrlOptions,
    inputScriptState,
    memory,
    audioStats,
    hud: {
      stage: safeText('hud-boot-stage'),
      status: safeText('hud-status'),
      frames: safeText('metric-frames'),
      fps: safeText('metric-fps'),
      gpuErrors: safeText('gpu-validation-errors'),
      modeBadge: safeText('boot-mode-badge'),
      sideModules: safeText('side-module-status'),
      logLines: hudLog,
    },
  };
})()"""

FORCE_SMOKE_JS = r"""(async () => {
  try {
    const app = window.playgta6App;
    if (!app || !app.session) return 'no-app';
    const state = typeof app.session.getState === 'function' ? app.session.getState() : '';
    if (state === 'running_smoke') return 'already-smoke';
    if (typeof app.session.start === 'function') {
      await app.session.start({ preferSmoke: true });
      return 'started-smoke';
    }
    const btn = document.getElementById('btn-smoke-demo');
    if (btn && btn.dataset.active !== 'true') {
      btn.click();
      return 'clicked-smoke';
    }
    return state || 'unknown';
  } catch (err) {
    return 'error:' + String(err && err.message ? err.message : err);
  }
})()"""


@dataclass
class BrowserSessionResult:
    """Raw output of a live (or mocked) browser boot run."""

    ok: bool = True
    exit_code: int = 0
    elapsed_ms: float = 0.0
    status: str = "ok"  # ok | fault | error | timeout
    message: str = ""
    crashed: bool = False
    server_ready: dict[str, Any] = field(default_factory=dict)
    final_probe: dict[str, Any] = field(default_factory=dict)
    console_lines: list[str] = field(default_factory=list)
    page_errors: list[str] = field(default_factory=list)
    presents: list[dict[str, Any]] = field(default_factory=list)
    first_present_ms: float | None = None
    captured_frames: list[str] = field(default_factory=list)
    capture_frames_enabled: bool = False
    capture_frames_dir: str | None = None
    non_black_canvas_frames: int | None = None
    scripted_inputs_sent: int = 0
    mp4_path: str | None = None
    mp4_fps: float | None = None
    mp4_skip_black: bool = False
    hide_hud: bool = False
    warnings: list[str] = field(default_factory=list)
    target_url: str = ""
    input_script: str | None = None
    stream_shaders: bool = False
    stream_threshold_mb: float | None = None
    boot_options: dict[str, Any] | None = None
    gpu_adapter: str | None = None
    first_visible_frame_s: float | None = None
    black_loading_s: float | None = None
    js_heap_used_mb: float | None = None
    js_heap_total_mb: float | None = None
    audio_ring_attached: bool | None = None
    audio_frames_drained: int | None = None
    present_fps_avg: float | None = None


def _format_console_event(params: Mapping[str, Any], elapsed_s: float) -> tuple[str, bool]:
    kind = str(params.get("type") or "log")
    parts: list[str] = []
    for arg in params.get("args") or []:
        if not isinstance(arg, Mapping):
            continue
        if "value" in arg and arg["value"] is not None:
            parts.append(str(arg["value"]))
        elif "description" in arg and arg["description"]:
            parts.append(str(arg["description"]))
        elif "type" in arg:
            parts.append(f"<{arg['type']}>")
    text = sanitize_text(" ".join(parts).strip())
    return f"[{elapsed_s:7.2f}s] [console.{kind}] {text}", (kind == "error")


def _format_exception_event(params: Mapping[str, Any], elapsed_s: float) -> str:
    details = params.get("exceptionDetails") or {}
    text = details.get("text") or "Uncaught exception"
    exc_obj = details.get("exception") or {}
    desc = exc_obj.get("description") or exc_obj.get("value") or ""
    msg = f"{text}: {desc}" if desc and str(desc) not in str(text) else str(text)
    return f"[{elapsed_s:7.2f}s] [pageerror] {sanitize_text(msg)}"


def _extract_present_count(probe: Mapping[str, Any]) -> int:
    candidates: list[int] = []
    boot_state = probe.get("bootState")
    if isinstance(boot_state, Mapping):
        for key in ("presentCount",):
            val = boot_state.get(key)
            if isinstance(val, (int, float)) and val >= 0:
                candidates.append(int(val))
        gpu = boot_state.get("gpu")
        if isinstance(gpu, Mapping) and isinstance(gpu.get("presentCount"), (int, float)):
            candidates.append(int(gpu["presentCount"]))
    diag = probe.get("diagnostics")
    if isinstance(diag, Mapping) and isinstance(diag.get("framesPresented"), (int, float)):
        candidates.append(int(diag["framesPresented"]))
    hud = probe.get("hud")
    if isinstance(hud, Mapping) and isinstance(hud.get("frames"), str) and hud["frames"].isdigit():
        candidates.append(int(hud["frames"]))
    return max(candidates) if candidates else 0


def _extract_non_negative_int(*candidates: Any) -> int:
    for val in candidates:
        if isinstance(val, bool) or val is None:
            continue
        if isinstance(val, (int, float)) and math.isfinite(val) and val >= 0:
            return int(val)
    return 0


def _extract_gpu_counts(probe: Mapping[str, Any]) -> dict[str, int]:
    boot_state = probe.get("bootState") if isinstance(probe.get("bootState"), Mapping) else {}
    gpu = boot_state.get("gpu") if isinstance(boot_state.get("gpu"), Mapping) else {}
    diag = probe.get("diagnostics") if isinstance(probe.get("diagnostics"), Mapping) else {}
    return {
        "realPackDrawCount": _extract_non_negative_int(
            gpu.get("realPackDrawCount"),
            boot_state.get("realPackDrawCount"),
        ),
        "fallbackDrawCount": _extract_non_negative_int(
            gpu.get("fallbackDrawCount"),
            boot_state.get("fallbackDrawCount"),
        ),
        "unboundBindGroupCount": _extract_non_negative_int(
            gpu.get("unboundBindGroupCount"),
            boot_state.get("unboundBindGroupCount"),
        ),
        "skippedDrawCount": _extract_non_negative_int(
            gpu.get("skippedDrawCount"),
            boot_state.get("skippedDrawCount"),
        ),
        "validationErrorCount": _extract_non_negative_int(
            gpu.get("validationErrorCount"),
            boot_state.get("validationErrorCount"),
            diag.get("validationErrorCount"),
        ),
    }


def _extract_input_script(
    probe: Mapping[str, Any],
    fallback_input_script: str | None = None,
) -> str | None:
    """Extract and sanitize active input-script preset from probe or CLI fallback."""
    candidates: list[Any] = []
    script_state = probe.get("inputScriptState")
    if isinstance(script_state, Mapping) and script_state.get("active") is not False:
        candidates.append(script_state.get("preset"))
        candidates.append(script_state.get("name"))
    boot_url_opts = probe.get("bootUrlOptions")
    if isinstance(boot_url_opts, Mapping):
        candidates.append(boot_url_opts.get("inputScript"))
    boot_state = probe.get("bootState")
    if isinstance(boot_state, Mapping):
        candidates.append(boot_state.get("inputScript"))
    diag = probe.get("diagnostics")
    if isinstance(diag, Mapping):
        candidates.append(diag.get("inputScript"))
    href = probe.get("href")
    if isinstance(href, str) and "?" in href:
        parsed = urlsplit(href)
        for key, value in parse_qsl(parsed.query, keep_blank_values=False):
            if key.lower() == "inputscript":
                candidates.append(value)
    candidates.append(fallback_input_script)
    for raw in candidates:
        cleaned = sanitize_input_script_preset(raw)
        if cleaned is not None:
            return cleaned
    return None


def sanitize_gpu_adapter(raw: Any) -> str | None:
    """Sanitize a WebGPU adapter string to lowercase safe vendor/arch tokens (max 96 chars, no host paths)."""
    if not isinstance(raw, str):
        return None
    lines = raw.splitlines()
    first_line = lines[0] if lines else ""
    if not first_line.strip():
        return None
    cleaned = sanitize_text(first_line)
    cleaned = re.sub(r"<(?:path|user)>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.lower()
    segments: list[str] = []
    for raw_seg in cleaned.split("/"):
        seg = re.sub(r"\.\.+", " ", raw_seg)
        seg = re.sub(r"[^a-z0-9._+\- ]+", " ", seg)
        seg = re.sub(r"\s+", " ", seg).strip(" ._-+")
        if seg and re.search(r"[a-z0-9]", seg):
            segments.append(seg)
    if not segments:
        return None
    joined = " / ".join(segments)
    if len(joined) > MAX_GPU_ADAPTER_LEN:
        joined = joined[:MAX_GPU_ADAPTER_LEN].rstrip(" /._-+")
    return joined or None


def extract_gpu_adapter_from_line(line: Any) -> str | None:
    """Extract and sanitize a WebGPU adapter string from one console/log line."""
    if not isinstance(line, str):
        return None
    match = _WGPU_ADAPTER_LOG_RE.search(line)
    if match is None:
        return None
    return sanitize_gpu_adapter(match.group(1))


def extract_gpu_adapter_from_console_lines(lines: Sequence[str] | None) -> str | None:
    """Scan console log lines for `[wgpu_worker] WebGPU adapter: ...` and return the latest sanitized value."""
    if not lines:
        return None
    detected: str | None = None
    for line in lines:
        parsed = extract_gpu_adapter_from_line(line)
        if parsed is not None:
            detected = parsed
    return detected


def _extract_gpu_adapter(
    session: BrowserSessionResult,
    probe: Mapping[str, Any] | None = None,
) -> str | None:
    """Resolve sanitized `gpu_adapter` from session, console logs, or probe telemetry."""
    direct = sanitize_gpu_adapter(session.gpu_adapter)
    if direct is not None:
        return direct
    from_console = extract_gpu_adapter_from_console_lines(session.console_lines)
    if from_console is not None:
        return from_console
    probe_map = probe if isinstance(probe, Mapping) else (
        session.final_probe if isinstance(session.final_probe, Mapping) else {}
    )
    hud = probe_map.get("hud")
    if isinstance(hud, Mapping):
        hud_lines = hud.get("logLines")
        if isinstance(hud_lines, list):
            from_hud = extract_gpu_adapter_from_console_lines([str(x) for x in hud_lines])
            if from_hud is not None:
                return from_hud
    boot_state = probe_map.get("bootState")
    if isinstance(boot_state, Mapping):
        gpu = boot_state.get("gpu")
        if isinstance(gpu, Mapping):
            for key in ("adapter", "gpuAdapter", "gpu_adapter"):
                val = sanitize_gpu_adapter(gpu.get(key))
                if val is not None:
                    return val
    diag = probe_map.get("diagnostics")
    if isinstance(diag, Mapping):
        for key in ("gpuAdapter", "gpu_adapter"):
            val = sanitize_gpu_adapter(diag.get(key))
            if val is not None:
                return val
    return None


def _sanitize_non_negative_float(value: Any, *, digits: int | None = None) -> float | None:
    """Return a finite non-negative float (optionally rounded), rejecting bools/strings/NaN/Inf."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    fval = float(value)
    if not math.isfinite(fval) or fval < 0.0:
        return None
    if digits is not None:
        fval = round(fval, digits)
    return 0.0 if fval == 0.0 else fval


def _sanitize_optional_non_negative_int(value: Any) -> int | None:
    """Return a finite non-negative integer, rejecting bools/strings/NaN/Inf/negatives."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        if math.isfinite(value) and value >= 0.0 and value.is_integer():
            return int(value)
        return None
    return None


def _sanitize_optional_bool(value: Any) -> bool | None:
    """Return a boolean only when the input is an actual bool."""
    return value if isinstance(value, bool) else None


def sanitize_js_heap_mb_from_bytes(raw_bytes: Any) -> float | None:
    """Convert non-negative finite JS heap byte counts into MiB rounded to 1 decimal place."""
    val = _sanitize_non_negative_float(raw_bytes)
    if val is None:
        return None
    mb = round(val / (1024.0 * 1024.0), 1)
    return 0.0 if mb == 0.0 else mb


def extract_js_heap_mb_from_probe(probe: Mapping[str, Any] | None) -> tuple[float | None, float | None]:
    """Extract sanitized `(js_heap_used_mb, js_heap_total_mb)` from a browser telemetry probe."""
    if not isinstance(probe, Mapping):
        return None, None
    used_mb: float | None = None
    total_mb: float | None = None

    boot_state = probe.get("bootState") if isinstance(probe.get("bootState"), Mapping) else {}
    diag = probe.get("diagnostics") if isinstance(probe.get("diagnostics"), Mapping) else {}
    memory_sources: list[Mapping[str, Any]] = []
    for candidate in (
        probe.get("memory"),
        boot_state.get("memory"),
        diag.get("memory"),
        probe,
        boot_state,
        diag,
    ):
        if isinstance(candidate, Mapping):
            memory_sources.append(candidate)

    for src in memory_sources:
        if used_mb is None:
            for key in ("js_heap_used_mb", "jsHeapUsedMb"):
                val = _sanitize_non_negative_float(src.get(key), digits=1)
                if val is not None:
                    used_mb = val
                    break
        if used_mb is None:
            for key in ("usedJSHeapSize", "used_js_heap_size"):
                val = sanitize_js_heap_mb_from_bytes(src.get(key))
                if val is not None:
                    used_mb = val
                    break
        if total_mb is None:
            for key in ("js_heap_total_mb", "jsHeapTotalMb"):
                val = _sanitize_non_negative_float(src.get(key), digits=1)
                if val is not None:
                    total_mb = val
                    break
        if total_mb is None:
            for key in ("totalJSHeapSize", "total_js_heap_size"):
                val = sanitize_js_heap_mb_from_bytes(src.get(key))
                if val is not None:
                    total_mb = val
                    break
        if used_mb is not None and total_mb is not None:
            break
    return used_mb, total_mb


def _extract_js_heap_mb(
    session: BrowserSessionResult,
    probe: Mapping[str, Any] | None = None,
) -> tuple[float | None, float | None]:
    """Resolve sanitized `(js_heap_used_mb, js_heap_total_mb)` from session or telemetry probe."""
    used_mb = _sanitize_non_negative_float(session.js_heap_used_mb, digits=1)
    total_mb = _sanitize_non_negative_float(session.js_heap_total_mb, digits=1)
    if used_mb is not None and total_mb is not None:
        return used_mb, total_mb
    probe_map = probe if isinstance(probe, Mapping) else (
        session.final_probe if isinstance(session.final_probe, Mapping) else {}
    )
    p_used, p_total = extract_js_heap_mb_from_probe(probe_map)
    return (
        used_mb if used_mb is not None else p_used,
        total_mb if total_mb is not None else p_total,
    )


def extract_visible_frame_timing_from_console_lines(
    lines: Sequence[str] | None,
) -> tuple[float | None, float | None]:
    """Scan console/HUD log lines for `FIRST_VISIBLE_FRAME` and `black_loading` timing in seconds."""
    if not lines:
        return None, None
    first_visible_s: float | None = None
    black_loading_s: float | None = None
    for raw_line in lines:
        if not isinstance(raw_line, str):
            continue
        if first_visible_s is None:
            m_ms = _FIRST_VISIBLE_FRAME_MS_RE.search(raw_line)
            if m_ms is not None:
                val = _sanitize_non_negative_float(float(m_ms.group(1)) / 1000.0, digits=3)
                if val is not None:
                    first_visible_s = val
        if first_visible_s is None:
            m_s = _FIRST_VISIBLE_FRAME_S_RE.search(raw_line)
            if m_s is not None:
                val = _sanitize_non_negative_float(float(m_s.group(1)), digits=3)
                if val is not None:
                    first_visible_s = val
        if first_visible_s is None:
            m_at = _MILESTONE_FIRST_VISIBLE_AT_RE.search(raw_line)
            if m_at is not None:
                raw_num = float(m_at.group(1))
                unit = m_at.group(2).lower()
                secs = raw_num / 1000.0 if unit == "ms" else raw_num
                val = _sanitize_non_negative_float(secs, digits=3)
                if val is not None:
                    first_visible_s = val
        if first_visible_s is None and _FIRST_VISIBLE_FRAME_LOG_RE.search(raw_line):
            m_prefix = _LOG_ELAPSED_PREFIX_RE.search(raw_line)
            if m_prefix is not None:
                val = _sanitize_non_negative_float(float(m_prefix.group(1)), digits=3)
                if val is not None:
                    first_visible_s = val
        if black_loading_s is None:
            m_bl_ms = _BLACK_LOADING_MS_RE.search(raw_line)
            if m_bl_ms is not None:
                val = _sanitize_non_negative_float(float(m_bl_ms.group(1)) / 1000.0, digits=3)
                if val is not None:
                    black_loading_s = val
        if black_loading_s is None:
            m_bl_s = _BLACK_LOADING_S_RE.search(raw_line)
            if m_bl_s is not None:
                val = _sanitize_non_negative_float(float(m_bl_s.group(1)), digits=3)
                if val is not None:
                    black_loading_s = val
        if first_visible_s is not None and black_loading_s is not None:
            break
    return first_visible_s, black_loading_s


def _extract_visible_frame_timing(
    session: BrowserSessionResult,
    probe: Mapping[str, Any] | None = None,
) -> tuple[float | None, float | None]:
    """Resolve `(first_visible_frame_s, black_loading_s)` from session, probe telemetry, or logs."""
    first_visible_ms: float | None = None
    black_loading_ms: float | None = None

    direct_fvf_s = _sanitize_non_negative_float(session.first_visible_frame_s, digits=3)
    if direct_fvf_s is not None:
        first_visible_ms = direct_fvf_s * 1000.0
    direct_bl_s = _sanitize_non_negative_float(session.black_loading_s, digits=3)
    if direct_bl_s is not None:
        black_loading_ms = direct_bl_s * 1000.0

    probe_map = probe if isinstance(probe, Mapping) else (
        session.final_probe if isinstance(session.final_probe, Mapping) else {}
    )
    boot_state = probe_map.get("bootState") if isinstance(probe_map.get("bootState"), Mapping) else {}
    gpu = boot_state.get("gpu") if isinstance(boot_state.get("gpu"), Mapping) else {}
    diag = probe_map.get("diagnostics") if isinstance(probe_map.get("diagnostics"), Mapping) else {}
    tracer = probe_map.get("tracerSummary") if isinstance(probe_map.get("tracerSummary"), Mapping) else {}

    timing_sources = (gpu, boot_state, diag, tracer, probe_map)
    for src in timing_sources:
        if first_visible_ms is None:
            for key in ("firstVisibleFrameMs", "first_visible_frame_ms"):
                val = _sanitize_non_negative_float(src.get(key))
                if val is not None:
                    first_visible_ms = val
                    break
        if first_visible_ms is None:
            for key in ("firstVisibleFrameS", "first_visible_frame_s"):
                val = _sanitize_non_negative_float(src.get(key))
                if val is not None:
                    first_visible_ms = val * 1000.0
                    break
        if black_loading_ms is None:
            for key in ("blackLoadingMs", "black_loading_ms"):
                val = _sanitize_non_negative_float(src.get(key))
                if val is not None:
                    black_loading_ms = val
                    break
        if black_loading_ms is None:
            for key in ("blackLoadingS", "black_loading_s"):
                val = _sanitize_non_negative_float(src.get(key))
                if val is not None:
                    black_loading_ms = val * 1000.0
                    break

    first_present_ms: float | None = None
    for src in timing_sources:
        for key in ("firstPresentMs", "first_present_ms"):
            val = _sanitize_non_negative_float(src.get(key))
            if val is not None:
                first_present_ms = val
                break
        if first_present_ms is not None:
            break

    raw_milestones = tracer.get("milestones")
    if isinstance(raw_milestones, list):
        for m in raw_milestones:
            if not isinstance(m, Mapping):
                continue
            if "reached" in m and m.get("reached") is not True:
                continue
            name = m.get("milestone")
            elapsed = _sanitize_non_negative_float(
                m.get("elapsedMs") if "elapsedMs" in m else m.get("elapsed_ms")
            )
            if elapsed is None:
                continue
            if name == "FIRST_VISIBLE_FRAME" and first_visible_ms is None:
                first_visible_ms = elapsed
            elif name == "FIRST_PRESENT" and first_present_ms is None:
                first_present_ms = elapsed

    if first_visible_ms is None or black_loading_ms is None:
        log_fvf_s, log_bl_s = extract_visible_frame_timing_from_console_lines(session.console_lines)
        if first_visible_ms is None and log_fvf_s is not None:
            first_visible_ms = log_fvf_s * 1000.0
        if black_loading_ms is None and log_bl_s is not None:
            black_loading_ms = log_bl_s * 1000.0

    if first_visible_ms is None or black_loading_ms is None:
        hud = probe_map.get("hud")
        if isinstance(hud, Mapping) and isinstance(hud.get("logLines"), list):
            hud_fvf_s, hud_bl_s = extract_visible_frame_timing_from_console_lines(
                [str(x) for x in hud["logLines"]]
            )
            if first_visible_ms is None and hud_fvf_s is not None:
                first_visible_ms = hud_fvf_s * 1000.0
            if black_loading_ms is None and hud_bl_s is not None:
                black_loading_ms = hud_bl_s * 1000.0

    if first_present_ms is None:
        first_present_ms = _sanitize_non_negative_float(session.first_present_ms)

    if black_loading_ms is None and first_visible_ms is not None and first_present_ms is not None:
        if first_visible_ms >= first_present_ms:
            black_loading_ms = first_visible_ms - first_present_ms

    first_visible_s = (
        direct_fvf_s
        if direct_fvf_s is not None
        else (
            _sanitize_non_negative_float(first_visible_ms / 1000.0, digits=3)
            if first_visible_ms is not None
            else None
        )
    )
    black_loading_s = (
        direct_bl_s
        if direct_bl_s is not None
        else (
            _sanitize_non_negative_float(black_loading_ms / 1000.0, digits=3)
            if black_loading_ms is not None
            else None
        )
    )
    return first_visible_s, black_loading_s


def extract_audio_ring_summary_from_console_lines(
    lines: Sequence[str] | None,
) -> tuple[bool | None, int | None]:
    """Scan console/HUD log lines for `#1028` guest audio PCM ring attachment and drained frame counts."""
    if not lines:
        return None, None
    attached: bool | None = None
    frames_drained: int | None = None
    for raw_line in lines:
        if not isinstance(raw_line, str):
            continue
        kv_att = _AUDIO_RING_ATTACHED_KV_RE.search(raw_line)
        if kv_att is not None:
            token = kv_att.group(1).lower()
            attached = token in ("true", "yes", "1")
        elif _AUDIO_RING_DETACHED_EVENT_RE.search(raw_line):
            attached = False
        elif _AUDIO_RING_ATTACHED_EVENT_RE.search(raw_line) and not _AUDIO_RING_ATTACH_FAIL_RE.search(raw_line):
            attached = True

        kv_guest = _AUDIO_GUEST_FRAMES_DRAINED_KV_RE.search(raw_line)
        if kv_guest is not None:
            frames_drained = int(kv_guest.group(1))
        elif _AUDIO_RING_CONTEXT_RE.search(raw_line):
            kv_gen = _AUDIO_GENERIC_FRAMES_DRAINED_KV_RE.search(raw_line)
            if kv_gen is not None:
                frames_drained = int(kv_gen.group(1))
    return attached, frames_drained


def _extract_audio_ring_summary(
    session: BrowserSessionResult,
    probe: Mapping[str, Any] | None = None,
) -> tuple[bool | None, int | None]:
    """Resolve `#1028` `(audio_ring_attached, audio_frames_drained)` from session, probe, or logs."""
    attached = _sanitize_optional_bool(session.audio_ring_attached)
    frames_drained = _sanitize_optional_non_negative_int(session.audio_frames_drained)
    if attached is not None and frames_drained is not None:
        return attached, frames_drained

    probe_map = probe if isinstance(probe, Mapping) else (
        session.final_probe if isinstance(session.final_probe, Mapping) else {}
    )
    boot_state = probe_map.get("bootState") if isinstance(probe_map.get("bootState"), Mapping) else {}
    diag = probe_map.get("diagnostics") if isinstance(probe_map.get("diagnostics"), Mapping) else {}

    audio_sources: list[Mapping[str, Any]] = []
    for candidate in (
        probe_map.get("audioStats"),
        probe_map.get("audio"),
        boot_state.get("audio"),
        diag.get("audio"),
        boot_state,
        diag,
    ):
        if isinstance(candidate, Mapping):
            audio_sources.append(candidate)

    for src in audio_sources:
        if attached is None:
            for key in (
                "audio_ring_attached",
                "audioRingAttached",
                "guestRingAttached",
                "guest_ring_attached",
                "pcmRingAttached",
                "ringAttached",
            ):
                val = _sanitize_optional_bool(src.get(key))
                if val is not None:
                    attached = val
                    break
        if frames_drained is None:
            for key in (
                "audio_frames_drained",
                "audioFramesDrained",
                "guestFramesDrained",
                "guest_frames_drained",
                "guestDrainedFrames",
            ):
                val = _sanitize_optional_non_negative_int(src.get(key))
                if val is not None:
                    frames_drained = val
                    break

    if frames_drained is None:
        for src in audio_sources:
            for key in ("framesDrained", "frames_drained"):
                val = _sanitize_optional_non_negative_int(src.get(key))
                if val is not None:
                    frames_drained = val
                    break
            if frames_drained is not None:
                break

    if attached is None or frames_drained is None:
        log_att, log_drained = extract_audio_ring_summary_from_console_lines(session.console_lines)
        if attached is None:
            attached = log_att
        if frames_drained is None:
            frames_drained = log_drained

    if attached is None or frames_drained is None:
        hud = probe_map.get("hud")
        if isinstance(hud, Mapping) and isinstance(hud.get("logLines"), list):
            hud_att, hud_drained = extract_audio_ring_summary_from_console_lines(
                [str(x) for x in hud["logLines"]]
            )
            if attached is None:
                attached = hud_att
            if frames_drained is None:
                frames_drained = hud_drained

    return attached, frames_drained


def compute_present_fps_avg(
    presents: Sequence[Mapping[str, Any]] | None,
    *,
    present_count: int | None = None,
    first_present_ms: float | None = None,
) -> float | None:
    """Compute average presents/sec from FIRST_PRESENT to last present wall time.

    Returns ``None`` when fewer than 2 presents were observed, when no valid last
    present wall timestamp exists, or when the elapsed duration is <= 0 ms.
    """
    walls: list[float] = []
    if presents:
        for sample in presents:
            if isinstance(sample, Mapping):
                wall_ms = _sanitize_non_negative_float(sample.get("wall_ms"))
                if wall_ms is not None:
                    walls.append(wall_ms)

    clean_count = _sanitize_optional_non_negative_int(present_count)
    total_presents = max(clean_count, len(walls)) if clean_count is not None else len(walls)
    if total_presents < 2 or not walls:
        return None

    start_ms = _sanitize_non_negative_float(first_present_ms)
    if start_ms is None:
        start_ms = walls[0]
    end_ms = walls[-1]
    delta_ms = end_ms - start_ms
    if not math.isfinite(delta_ms) or delta_ms <= 0.0:
        return None

    fps = (float(total_presents) - 1.0) * 1000.0 / delta_ms
    return _sanitize_non_negative_float(fps, digits=2)


def _extract_present_fps_avg(
    session: BrowserSessionResult,
    probe: Mapping[str, Any] | None = None,
) -> float | None:
    """Resolve `present_fps_avg` from session override or recorded present timestamps."""
    direct = _sanitize_non_negative_float(session.present_fps_avg, digits=2)
    if direct is not None:
        return direct
    probe_map = probe if isinstance(probe, Mapping) else (
        session.final_probe if isinstance(session.final_probe, Mapping) else {}
    )
    present_count = max(len(session.presents), _extract_present_count(probe_map))
    return compute_present_fps_avg(
        session.presents,
        present_count=present_count,
        first_present_ms=session.first_present_ms,
    )


def _sample_present_record(probe: Mapping[str, Any], index: int, wall_ms: float) -> dict[str, Any]:
    counts = _extract_gpu_counts(probe)
    return {
        "index": index,
        "wall_ms": round(wall_ms, 2),
        "dispatches": 0,
        "import_calls": 0,
        "counters": {
            "draws_real_pack": counts["realPackDrawCount"],
            "draws_fallback": counts["fallbackDrawCount"],
            "draws_skipped": counts["skippedDrawCount"],
            "gpu_validation_errors": counts["validationErrorCount"],
        },
    }


def _import_pil_image() -> Any:
    """Return `PIL.Image` module if Pillow is installed, else `None`."""
    try:
        from PIL import Image  # type: ignore[import-not-found]

        return Image
    except ImportError:
        return None


def check_canvas_non_black(
    image_bytes: bytes,
    *,
    canvas_min_x: int = CANVAS_MIN_X,
    pil_image_mod: Any | None = None,
    _auto_import_pil: bool = True,
) -> bool | None:
    """Check whether RGB pixels at `x >= canvas_min_x` have any non-zero channel.

    Returns `True` when at least one pixel in `x >= canvas_min_x` has a non-zero
    RGB value, `False` when all pixels in that region are pure black (or the image
    width is `<= canvas_min_x`), and `None` when Pillow is unavailable or the
    image bytes cannot be decoded.
    """
    img_mod = pil_image_mod
    if img_mod is None and _auto_import_pil:
        img_mod = _import_pil_image()
    if img_mod is None:
        return None
    try:
        with img_mod.open(io.BytesIO(image_bytes)) as img:
            rgb = img.convert("RGB")
            width, height = rgb.size
            if width <= canvas_min_x or height <= 0:
                return False
            cropped = rgb.crop((canvas_min_x, 0, width, height))
            extrema = cropped.getextrema()
    except Exception:
        return None
    if not extrema:
        return False
    for channel in extrema:
        if isinstance(channel, (tuple, list)) and len(channel) >= 2:
            if channel[1] > 0:
                return True
        elif isinstance(channel, (int, float)) and channel > 0:
            return True
    return False


_FRAME_FILE_RE = re.compile(r"^frame_(\d+)\.(?:jpg|png)$")


def _list_numbered_frames(frames_dir: Path, ext: str) -> list[tuple[int, Path]]:
    numbered: list[tuple[int, Path]] = []
    suffix = f".{ext}"
    for candidate in frames_dir.glob(f"frame_*{suffix}"):
        if not candidate.is_file():
            continue
        match = _FRAME_FILE_RE.match(candidate.name)
        if match is not None:
            numbered.append((int(match.group(1)), candidate))
    numbered.sort(key=lambda item: item[0])
    return numbered


def find_first_non_black_frame_number(
    frames: Sequence[tuple[int, Path]],
    *,
    canvas_min_x: int = CANVAS_MIN_X,
    pil_image_mod: Any | None = None,
    _auto_import_pil: bool = True,
) -> tuple[int | None, int]:
    """Return `(first_non_black_frame_number, skipped_leading_black_count)`.

    Returns `(None, 0)` when Pillow is unavailable, no frames exist, or every
    captured frame is black so callers fall back to encoding all frames cleanly.
    """
    if not frames:
        return None, 0
    img_mod = pil_image_mod
    if img_mod is None and _auto_import_pil:
        img_mod = _import_pil_image()
    if img_mod is None:
        return None, 0

    skipped = 0
    for frame_num, frame_path in frames:
        try:
            raw_bytes = frame_path.read_bytes()
        except OSError:
            return (frame_num, skipped) if skipped > 0 else (None, 0)
        non_black = check_canvas_non_black(
            raw_bytes,
            canvas_min_x=canvas_min_x,
            pil_image_mod=img_mod,
            _auto_import_pil=False,
        )
        if non_black is True:
            return frame_num, skipped
        if non_black is None:
            return (frame_num, skipped) if skipped > 0 else (None, 0)
        skipped += 1
    return None, 0


def _format_ffmpeg_fps(fps: float) -> str:
    """Format a validated positive finite FPS value for ffmpeg `-framerate`."""
    rounded = round(fps, 3)
    if rounded > 0.0:
        return str(rounded)
    text = f"{fps:.10f}".rstrip("0").rstrip(".")
    return text if text and text != "0" else f"{fps:.16f}".rstrip("0").rstrip(".")


def _is_valid_frame_file(path: Path, ext: str) -> bool:
    """Return True when `path` is a readable, non-empty, structurally complete PNG/JPEG frame."""
    try:
        st = path.stat()
    except OSError:
        return False
    if st.st_size <= 0:
        return False
    try:
        with path.open("rb") as fh:
            if ext == "png":
                if st.st_size < len(_PNG_SIGNATURE) + len(_PNG_IEND_CHUNK):
                    return False
                if fh.read(len(_PNG_SIGNATURE)) != _PNG_SIGNATURE:
                    return False
                tail_len = min(st.st_size, 32)
                fh.seek(st.st_size - tail_len)
                return _PNG_IEND_CHUNK in fh.read(tail_len)
            if ext == "jpg":
                if st.st_size < 4:
                    return False
                if fh.read(2) != b"\xff\xd8":
                    return False
                tail_len = min(st.st_size, 16)
                fh.seek(st.st_size - tail_len)
                return fh.read(tail_len).rstrip(b"\x00").endswith(b"\xff\xd9")
    except OSError:
        return False
    return True


def encode_frames_to_mp4(
    frames_dir: Path,
    out_mp4: Path,
    *,
    fps: float = 1.0,
    skip_black: bool = False,
    canvas_min_x: int = CANVAS_MIN_X,
    pil_image_mod: Any | None = None,
    _auto_import_pil: bool = True,
    runner: CommandRunner = run_command,
    which_fn: WhichFn = shutil.which,
) -> tuple[bool, str | None]:
    """Encode captured `frame_0001.jpg` / `frame_0001.png` screenshots into an H.264 MP4 using `ffmpeg`."""
    if fps is None:
        raise RunnerError("--mp4-fps must be a positive finite number")
    clean_fps = validate_mp4_fps(fps)
    assert clean_fps is not None
    ffmpeg = which_fn("ffmpeg")
    if not ffmpeg:
        return False, "ffmpeg not found on PATH; skipping MP4 encoding"
    if not frames_dir.is_dir():
        return False, "no captured JPEG frames found for MP4 encoding"
    jpg_frames = _list_numbered_frames(frames_dir, "jpg")
    png_frames = _list_numbered_frames(frames_dir, "png") if not jpg_frames else []
    if jpg_frames:
        ext = "jpg"
        pattern = "frame_%04d.jpg"
        numbered_frames = jpg_frames
    elif png_frames:
        ext = "png"
        pattern = "frame_%04d.png"
        numbered_frames = png_frames
    else:
        return False, "no captured JPEG frames found for MP4 encoding"

    valid_frames = [(num, path) for num, path in numbered_frames if _is_valid_frame_file(path, ext)]
    if not valid_frames:
        return False, "no valid captured frames found for MP4 encoding"

    needs_staging = len(valid_frames) < len(numbered_frames)
    stage_ctx: tempfile.TemporaryDirectory[str] | None = None
    try:
        if needs_staging:
            stage_ctx = tempfile.TemporaryDirectory(prefix="playgta6-mp4-frames-")
            stage_dir = Path(stage_ctx.name)
            staged_frames: list[tuple[int, Path]] = []
            for idx, (_orig_num, src_path) in enumerate(valid_frames, start=1):
                dst_path = stage_dir / f"frame_{idx:04d}.{ext}"
                try:
                    os.link(src_path, dst_path)
                except OSError:
                    shutil.copy2(src_path, dst_path)
                staged_frames.append((idx, dst_path))
            effective_frames_dir = stage_dir
            effective_frames = staged_frames
        else:
            effective_frames_dir = frames_dir
            effective_frames = valid_frames

        start_number: int | None = None
        if skip_black:
            first_non_black, skipped_count = find_first_non_black_frame_number(
                effective_frames,
                canvas_min_x=canvas_min_x,
                pil_image_mod=pil_image_mod,
                _auto_import_pil=_auto_import_pil,
            )
            if first_non_black is not None and skipped_count > 0:
                start_number = first_non_black

        rate_str = _format_ffmpeg_fps(clean_fps)
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            rate_str,
        ]
        if start_number is not None:
            cmd += ["-start_number", str(start_number)]
        cmd += [
            "-i",
            str(effective_frames_dir / pattern),
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_mp4),
        ]
        rc = runner(cmd, frames_dir.parent, frames_dir.parent / "ffmpeg.log", 120.0)
        if rc != 0 or not out_mp4.is_file():
            return False, f"ffmpeg exited with code {rc}"
        return True, None
    finally:
        if stage_ctx is not None:
            stage_ctx.cleanup()


def run_browser_session(
    *,
    repo: Path,
    run_dir: Path,
    chrome_path: Path,
    port: int,
    boot_dir: Path | None,
    game_dir: Path | None,
    shaders_dir: Path | None,
    duration_s: float,
    headless: bool,
    smoke: bool,
    send_input: bool,
    capture_frames: bool | Path | str | None,
    frame_interval_s: float,
    encode_mp4: bool,
    mp4_fps: float | None = None,
    mp4_skip_black: bool = False,
    hide_hud: bool = False,
    capture_frames_dir: Path | None = None,
    capture_interval_sec: float | None = None,
    input_script: str | None = None,
    stream_shaders: bool = False,
    stream_threshold_mb: float | None = None,
    boot_options: Mapping[str, Any] | None = None,
    quiet_server: bool = False,
    runner: CommandRunner = run_command,
    popen_fn: Callable[..., Any] = subprocess.Popen,
    cdp_factory: Callable[[str], CdpClient] = CdpClient,
    wait_server_fn: Callable[..., dict[str, Any]] = wait_for_server_ready,
    wait_cdp_fn: Callable[..., str] = wait_for_cdp_page_ws,
    which_fn: WhichFn = shutil.which,
) -> BrowserSessionResult:
    """Start `serve_local.py` + Chrome, drive CDP for `duration_s`, and always clean up."""
    host = "127.0.0.1"
    server_port = port if port > 0 else find_free_port(host)
    cdp_port = find_free_port(host)
    clean_input_script = validate_input_script_preset(input_script)
    clean_threshold_mb = validate_stream_threshold_mb(stream_threshold_mb)
    clean_mp4_fps = validate_mp4_fps(mp4_fps)
    effective_hide_hud = bool(hide_hud)
    effective_mp4_skip_black = bool(mp4_skip_black)
    effective_canvas_min_x = 0 if effective_hide_hud else CANVAS_MIN_X
    target_url = build_target_url(
        host,
        server_port,
        smoke=smoke,
        input_script=clean_input_script,
        stream_shaders=stream_shaders,
        stream_threshold_mb=clean_threshold_mb,
        hide_hud=effective_hide_hud,
    )

    if capture_frames_dir is not None:
        resolved_capture_dir: Path | None = resolve_against_repo(capture_frames_dir, repo)
        validate_capture_frames_dir_safety(resolved_capture_dir, repo)
        capture_enabled = True
    else:
        capture_enabled, resolved_capture_dir = validate_capture_frames_arg(capture_frames, repo)

    use_png_capture = resolved_capture_dir is not None
    if use_png_capture:
        assert resolved_capture_dir is not None
        frames_dir = resolved_capture_dir
        effective_frame_interval = validate_capture_interval_sec(
            DEFAULT_CAPTURE_INTERVAL_SEC if capture_interval_sec is None else capture_interval_sec
        )
    else:
        frames_dir = run_dir / "frames"
        effective_frame_interval = float(frame_interval_s)

    want_frames = capture_enabled or encode_mp4

    profile_dir = Path(tempfile.mkdtemp(prefix="playgta6-chrome-profile-"))
    smoke_boot_dir: Path | None = None
    if smoke:
        smoke_boot_dir = Path(tempfile.mkdtemp(prefix="playgta6-smoke-boot-"))
        if boot_options is not None:
            stage_boot_options_file(
                boot_options,
                smoke_boot_dir,
                repo_root=repo,
                dry_run=False,
                check_safety=False,
            )
        effective_boot_dir: Path | None = smoke_boot_dir
        effective_game_dir: Path | None = None
    else:
        if boot_options is not None and boot_dir is not None:
            stage_boot_options_file(
                boot_options,
                boot_dir,
                repo_root=repo,
                dry_run=False,
                check_safety=True,
            )
        effective_boot_dir = boot_dir
        effective_game_dir = game_dir

    data_dir = repo / "local-data"
    web_dir = repo / "web"
    server_cmd = build_server_cmd(
        repo=repo,
        host=host,
        port=server_port,
        web_dir=web_dir,
        data_dir=data_dir,
        boot_dir=effective_boot_dir,
        game_dir=effective_game_dir,
        shaders_dir=shaders_dir,
        quiet=bool(quiet_server),
    )
    chrome_cmd = build_chrome_args(
        chrome_path,
        user_data_dir=profile_dir,
        cdp_port=cdp_port,
        headless=headless,
        initial_url="about:blank",
    )

    result = BrowserSessionResult(
        target_url=target_url,
        input_script=clean_input_script,
        stream_shaders=bool(stream_shaders or clean_threshold_mb is not None),
        stream_threshold_mb=clean_threshold_mb,
        boot_options=dict(boot_options) if boot_options is not None else None,
        capture_frames_enabled=want_frames,
        capture_frames_dir=repo_relative_label(frames_dir, repo) if want_frames else None,
        hide_hud=effective_hide_hud,
        mp4_fps=clean_mp4_fps,
        mp4_skip_black=effective_mp4_skip_black,
    )
    server_proc: Any = None
    chrome_proc: Any = None
    cdp: CdpClient | None = None
    server_log_fh = None
    chrome_log_fh = None

    pil_image_mod: Any = None
    if want_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
        pil_image_mod = _import_pil_image()
        if pil_image_mod is not None:
            result.non_black_canvas_frames = 0
        else:
            result.non_black_canvas_frames = None
            result.warnings.append("Pillow (PIL) not installed; skipping non-black canvas extrema check")

    start_mono = time.monotonic()
    try:
        server_log_fh = (run_dir / "server.log").open("wb")
        server_proc = popen_fn(
            server_cmd,
            cwd=repo,
            stdout=server_log_fh,
            stderr=subprocess.STDOUT,
        )
        result.server_ready = wait_server_fn(host, server_port, timeout_s=SERVER_READY_TIMEOUT_S, proc=server_proc)

        chrome_log_fh = (run_dir / "chrome.log").open("wb")
        chrome_proc = popen_fn(
            chrome_cmd,
            cwd=repo,
            stdout=chrome_log_fh,
            stderr=subprocess.STDOUT,
        )
        ws_url = wait_cdp_fn(cdp_port, timeout_s=CDP_READY_TIMEOUT_S, proc=chrome_proc)

        cdp = cdp_factory(ws_url)
        try:
            cdp.connect()
            cdp.send("Runtime.enable")
            cdp.send("Log.enable")
            cdp.send("Page.enable")
            cdp.send("Page.navigate", {"url": target_url})
        except CdpError as exc:
            result.ok = False
            result.crashed = True
            result.status = "error"
            result.exit_code = 1
            result.message = sanitize_text(str(exc))
            result.elapsed_ms = round((time.monotonic() - start_mono) * 1000.0, 2)
            return result

        nav_start = time.monotonic()
        end_time = nav_start + duration_s
        next_poll = nav_start + min(DEFAULT_POLL_INTERVAL_S, duration_s)
        next_frame = nav_start + min(effective_frame_interval, duration_s) if want_frames else float("inf")
        next_input = nav_start + INPUT_START_S if send_input else float("inf")
        last_present_count = 0
        smoke_triggered = False

        def capture_one_frame() -> None:
            assert cdp is not None
            try:
                cdp_params: dict[str, Any] = (
                    {"format": "png"} if use_png_capture else {"format": "jpeg", "quality": 85}
                )
                shot = cdp.send("Page.captureScreenshot", cdp_params, timeout=8.0)
                b64_data = shot.get("data")
                if isinstance(b64_data, str) and b64_data:
                    raw_bytes = base64.b64decode(b64_data)
                    if not raw_bytes:
                        return
                    idx = len(result.captured_frames) + 1
                    ext = "png" if use_png_capture else "jpg"
                    fname = f"frame_{idx:04d}.{ext}"
                    atomic_write_bytes(raw_bytes, frames_dir / fname)
                    result.captured_frames.append(fname)
                    if pil_image_mod is not None:
                        non_black = check_canvas_non_black(
                            raw_bytes,
                            canvas_min_x=effective_canvas_min_x,
                            pil_image_mod=pil_image_mod,
                            _auto_import_pil=False,
                        )
                        if non_black is True:
                            result.non_black_canvas_frames = (result.non_black_canvas_frames or 0) + 1
            except (CdpError, ValueError, OSError):
                pass

        def process_events(events: Sequence[dict[str, Any]]) -> None:
            now_s = max(0.0, time.monotonic() - nav_start)
            for ev in events:
                method = ev.get("method")
                params = ev.get("params") or {}
                if method == "Runtime.consoleAPICalled":
                    line, _is_err = _format_console_event(params, now_s)
                    result.console_lines.append(line)
                    adapter = extract_gpu_adapter_from_line(line)
                    if adapter is not None:
                        result.gpu_adapter = adapter
                elif method == "Runtime.exceptionThrown":
                    line = _format_exception_event(params, now_s)
                    result.console_lines.append(line)
                    result.page_errors.append(line)
                elif method == "Log.entryAdded":
                    entry = params.get("entry") or {}
                    lvl = str(entry.get("level") or "info")
                    txt = sanitize_text(str(entry.get("text") or ""))
                    entry_line = f"[{now_s:7.2f}s] [log.{lvl}] {txt}"
                    result.console_lines.append(entry_line)
                    adapter = extract_gpu_adapter_from_line(entry_line)
                    if adapter is not None:
                        result.gpu_adapter = adapter
                elif method == "Inspector.targetCrashed":
                    result.crashed = True
                    result.status = "error"
                    result.message = "browser target crashed"

        def evaluate_probe() -> dict[str, Any]:
            assert cdp is not None
            try:
                resp = cdp.send(
                    "Runtime.evaluate",
                    {"expression": TELEMETRY_PROBE_JS, "returnByValue": True},
                    timeout=5.0,
                )
                val = (resp.get("result") or {}).get("value")
                return val if isinstance(val, dict) else {}
            except CdpError:
                return {}

        while time.monotonic() < end_time and not result.crashed:
            if chrome_proc.poll() is not None:
                result.crashed = True
                result.status = "error"
                result.message = f"Chrome exited early with code {chrome_proc.returncode}"
                break

            now = time.monotonic()
            sleep_for = min(0.15, max(0.0, end_time - now))
            process_events(cdp.drain_events(timeout=sleep_for))
            now = time.monotonic()

            if now >= next_poll:
                probe = evaluate_probe()
                if probe:
                    result.final_probe = probe
                    if smoke and not smoke_triggered and probe.get("readyState") in ("interactive", "complete"):
                        state = (probe.get("bootState") or {}).get("sessionState")
                        if state and state != "running_smoke":
                            try:
                                cdp.send(
                                    "Runtime.evaluate",
                                    {"expression": FORCE_SMOKE_JS, "awaitPromise": True, "returnByValue": True},
                                    timeout=5.0,
                                )
                            except CdpError:
                                pass
                        smoke_triggered = True
                    present_count = _extract_present_count(probe)
                    wall_ms = (now - nav_start) * 1000.0
                    if present_count > last_present_count:
                        if result.first_present_ms is None:
                            result.first_present_ms = round(wall_ms, 2)
                        result.presents.append(_sample_present_record(probe, len(result.presents), wall_ms))
                        last_present_count = present_count
                    tracer = probe.get("tracerSummary") or {}
                    last_trap = tracer.get("lastTrap") if isinstance(tracer, Mapping) else None
                    if isinstance(last_trap, Mapping) and last_trap.get("fatal") is True:
                        result.status = "fault"
                        result.message = sanitize_text(str(last_trap.get("message") or last_trap.get("kind") or "fatal trap"))
                        break
                next_poll = now + DEFAULT_POLL_INTERVAL_S

            if send_input and now >= next_input:
                try:
                    cdp.send(
                        "Input.dispatchKeyEvent",
                        {"type": "rawKeyDown", "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13},
                        timeout=3.0,
                    )
                    time.sleep(min(INPUT_HOLD_S, max(0.0, end_time - time.monotonic())))
                    cdp.send(
                        "Input.dispatchKeyEvent",
                        {"type": "keyUp", "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13},
                        timeout=3.0,
                    )
                    result.scripted_inputs_sent += 1
                except CdpError:
                    pass
                next_input = now + INPUT_PERIOD_S

            if want_frames and now >= next_frame:
                capture_one_frame()
                next_frame = now + effective_frame_interval

        # Final telemetry sample + drain
        if not result.crashed:
            process_events(cdp.drain_events(timeout=0.05))
            if want_frames and not result.captured_frames:
                capture_one_frame()
            final_probe = evaluate_probe()
            if final_probe:
                result.final_probe = final_probe
                present_count = _extract_present_count(final_probe)
                wall_ms = (time.monotonic() - nav_start) * 1000.0
                if present_count > last_present_count:
                    if result.first_present_ms is None:
                        result.first_present_ms = round(wall_ms, 2)
                    result.presents.append(_sample_present_record(final_probe, len(result.presents), wall_ms))

        result.elapsed_ms = round((time.monotonic() - start_mono) * 1000.0, 2)
        if result.gpu_adapter is None:
            result.gpu_adapter = _extract_gpu_adapter(result)
        if result.gpu_adapter is None and chrome_log_fh is not None:
            try:
                chrome_log_fh.flush()
                chrome_log_path = run_dir / "chrome.log"
                if chrome_log_path.is_file():
                    chrome_log_lines = chrome_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    result.gpu_adapter = extract_gpu_adapter_from_console_lines(chrome_log_lines)
            except OSError:
                pass
        result.first_visible_frame_s, result.black_loading_s = _extract_visible_frame_timing(result)
        result.js_heap_used_mb, result.js_heap_total_mb = _extract_js_heap_mb(result)
        result.audio_ring_attached, result.audio_frames_drained = _extract_audio_ring_summary(result)
        result.present_fps_avg = _extract_present_fps_avg(result)
        if result.crashed:
            result.ok = False
            result.exit_code = 1
        elif result.status == "fault":
            result.exit_code = 0

        if encode_mp4:
            effective_mp4_fps = (
                clean_mp4_fps
                if clean_mp4_fps is not None
                else max(1.0, round(1.0 / max(effective_frame_interval, 0.1), 3))
            )
            ok_mp4, mp4_warn = encode_frames_to_mp4(
                frames_dir,
                run_dir / "capture.mp4",
                fps=effective_mp4_fps,
                skip_black=effective_mp4_skip_black,
                canvas_min_x=effective_canvas_min_x,
                pil_image_mod=pil_image_mod,
                _auto_import_pil=False,
                runner=runner,
                which_fn=which_fn,
            )
            if ok_mp4:
                result.mp4_path = "capture.mp4"
            elif mp4_warn:
                result.warnings.append(mp4_warn)
                print(f"[warn] {mp4_warn}", file=sys.stderr)

        return result
    finally:
        if cdp is not None:
            try:
                cdp.close()
            except Exception:
                pass
        terminate_process(chrome_proc)
        terminate_process(server_proc)
        for fh in (chrome_log_fh, server_log_fh):
            if fh is not None:
                try:
                    fh.close()
                except OSError:
                    pass
        shutil.rmtree(profile_dir, ignore_errors=True)
        if smoke_boot_dir is not None:
            shutil.rmtree(smoke_boot_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Report synthesis (playgta6-boot-report v2) & digest
# ---------------------------------------------------------------------------


def build_boot_report(
    session: BrowserSessionResult,
    *,
    smoke: bool,
    preset: str,
    input_script: str | None = None,
    stream_shaders: bool = False,
    stream_threshold_mb: float | None = None,
    boot_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a sanitized `playgta6-boot-report` v2 dict compatible with `compare_boot_reports.py`."""
    probe = session.final_probe if isinstance(session.final_probe, Mapping) else {}
    boot_state = probe.get("bootState") if isinstance(probe.get("bootState"), Mapping) else {}
    gpu_counts = _extract_gpu_counts(probe)
    tracer = probe.get("tracerSummary") if isinstance(probe.get("tracerSummary"), Mapping) else {}
    hud = probe.get("hud") if isinstance(probe.get("hud"), Mapping) else {}
    effective_input_script = _extract_input_script(
        probe,
        input_script if input_script is not None else session.input_script,
    )
    effective_stream_threshold_mb = (
        stream_threshold_mb if stream_threshold_mb is not None else session.stream_threshold_mb
    )
    effective_stream_shaders = bool(
        stream_shaders or session.stream_shaders or effective_stream_threshold_mb is not None
    )
    effective_boot_options = boot_options if boot_options is not None else session.boot_options
    effective_gpu_adapter = _extract_gpu_adapter(session, probe)
    effective_first_visible_frame_s, effective_black_loading_s = _extract_visible_frame_timing(session, probe)
    effective_js_heap_used_mb, effective_js_heap_total_mb = _extract_js_heap_mb(session, probe)
    effective_audio_ring_attached, effective_audio_frames_drained = _extract_audio_ring_summary(session, probe)
    effective_present_fps_avg = _extract_present_fps_avg(session, probe)

    milestones: list[dict[str, Any]] = []
    raw_milestones = tracer.get("milestones")
    if isinstance(raw_milestones, list):
        reached = [
            m for m in raw_milestones
            if isinstance(m, Mapping) and m.get("reached") is True and m.get("milestone")
        ]
        reached.sort(key=lambda m: (float(m.get("elapsedMs") or 0.0), int(m.get("index") or 0)))
        for m in reached:
            milestones.append(
                {
                    "milestone": str(m["milestone"]),
                    "elapsed_ms": round(float(m.get("elapsedMs") or 0.0), 2),
                }
            )
    if session.first_present_ms is not None and not any(m["milestone"] == "FIRST_PRESENT" for m in milestones):
        milestones.append({"milestone": "FIRST_PRESENT", "elapsed_ms": round(session.first_present_ms, 2)})

    last_trap = tracer.get("lastTrap") if isinstance(tracer.get("lastTrap"), Mapping) else None
    fault: dict[str, Any] | None = None
    if isinstance(last_trap, Mapping) and last_trap.get("fatal") is True:
        fault = {
            "kind": str(last_trap.get("kind") or "GUEST_TRAP"),
            "trap_kind": str(last_trap.get("kind") or "GUEST_TRAP"),
            "va": last_trap.get("guestVa"),
            "detail": sanitize_text(str(last_trap.get("message") or "")),
            "exception_code": None,
            "function": None,
            "import": last_trap.get("import"),
        }

    boot_mode = str(boot_state.get("bootMode") or ("smoke-demo" if smoke else "idle"))
    phase = str(
        tracer.get("stage")
        or boot_state.get("milestone")
        or hud.get("stage")
        or ("SMOKE_DEMO" if smoke else "NOT_STARTED")
    )
    status = session.status
    if fault is not None:
        status = "fault"
    elif session.crashed:
        status = "error"

    present_count = max(len(session.presents), _extract_present_count(probe))
    first_present: dict[str, Any] | None = None
    if session.first_present_ms is not None or present_count > 0:
        first_present = {
            "import_seq": 0,
            "elapsed_ms": round(session.first_present_ms if session.first_present_ms is not None else session.elapsed_ms, 2),
            "imports": [],
        }

    console_errors = sum(1 for line in session.console_lines if "[console.error]" in line)
    summary: dict[str, Any] = {
        "realPackDrawCount": gpu_counts["realPackDrawCount"],
        "fallbackDrawCount": gpu_counts["fallbackDrawCount"],
        "validationErrorCount": gpu_counts["validationErrorCount"],
        "inputScript": effective_input_script,
    }
    if effective_stream_shaders:
        summary["streamShaders"] = True
    if effective_stream_threshold_mb is not None:
        summary["streamThresholdMb"] = effective_stream_threshold_mb
    if session.capture_frames_enabled or session.captured_frames or session.non_black_canvas_frames is not None:
        summary["capturedFrames"] = len(session.captured_frames)
        summary["nonBlackCanvasFrames"] = session.non_black_canvas_frames
    if effective_gpu_adapter is not None:
        summary["gpu_adapter"] = effective_gpu_adapter
    if effective_first_visible_frame_s is not None:
        summary["first_visible_frame_s"] = effective_first_visible_frame_s
        summary["first_visible_frame_ms"] = round(effective_first_visible_frame_s * 1000.0, 2)
    if effective_black_loading_s is not None:
        summary["black_loading_s"] = effective_black_loading_s
        summary["black_loading_ms"] = round(effective_black_loading_s * 1000.0, 2)
    if effective_js_heap_used_mb is not None:
        summary["js_heap_used_mb"] = effective_js_heap_used_mb
    if effective_js_heap_total_mb is not None:
        summary["js_heap_total_mb"] = effective_js_heap_total_mb
    if effective_audio_ring_attached is not None:
        summary["audio_ring_attached"] = effective_audio_ring_attached
    if effective_audio_frames_drained is not None:
        summary["audio_frames_drained"] = effective_audio_frames_drained
    if effective_present_fps_avg is not None:
        summary["present_fps_avg"] = effective_present_fps_avg

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "runner": "browser",
        "preset": preset,
        "mode": boot_mode,
        "smoke": bool(smoke),
        "inputScript": effective_input_script,
        "gpu_adapter": effective_gpu_adapter,
        "first_visible_frame_s": effective_first_visible_frame_s,
        "black_loading_s": effective_black_loading_s,
        "js_heap_used_mb": effective_js_heap_used_mb,
        "js_heap_total_mb": effective_js_heap_total_mb,
        "audio_ring_attached": effective_audio_ring_attached,
        "audio_frames_drained": effective_audio_frames_drained,
        "present_fps_avg": effective_present_fps_avg,
        "status": status,
        "phase": phase,
        "exit_code": session.exit_code,
        "message": sanitize_text(session.message or str(hud.get("status") or "")),
        "elapsed_ms": round(float(session.elapsed_ms), 2),
        "milestones": milestones,
        "first_calls": [],
        "fault": fault,
        "backtrace": {"method": "browser", "frames": []},
        "counters": {
            "import_calls": 0,
            "imports": [],
            "presents": present_count,
            "real_pack_draws": gpu_counts["realPackDrawCount"],
            "fallback_draws": gpu_counts["fallbackDrawCount"],
            "unbound_bind_groups": gpu_counts["unboundBindGroupCount"],
            "skipped_draws": gpu_counts["skippedDrawCount"],
            "gpu_validation_errors": gpu_counts["validationErrorCount"],
            "page_errors": len(session.page_errors),
            "console_errors": console_errors,
        },
        "summary": summary,
        "address_space": {
            "failures": 0,
            "reserved_peak": 0,
            "events": 0,
            "log": [],
        },
        "presents": list(session.presents),
        "first_present": first_present,
    }
    if effective_boot_options is not None:
        report["boot_options"] = dict(effective_boot_options)
    return sanitize_value(report)


def build_telemetry_payload(
    session: BrowserSessionResult,
    *,
    smoke: bool,
    preset: str,
    headless: bool,
    input_script: str | None = None,
    stream_shaders: bool = False,
    stream_threshold_mb: float | None = None,
    boot_options: Mapping[str, Any] | None = None,
    hide_hud: bool = False,
    mp4_fps: float | None = None,
    mp4_skip_black: bool = False,
) -> dict[str, Any]:
    """Build sanitized `telemetry.json` artifact."""
    probe = session.final_probe if isinstance(session.final_probe, Mapping) else {}
    gpu_counts = _extract_gpu_counts(probe)
    effective_input_script = _extract_input_script(
        probe,
        input_script if input_script is not None else session.input_script,
    )
    effective_stream_threshold_mb = (
        stream_threshold_mb if stream_threshold_mb is not None else session.stream_threshold_mb
    )
    effective_stream_shaders = bool(
        stream_shaders or session.stream_shaders or effective_stream_threshold_mb is not None
    )
    effective_boot_options = boot_options if boot_options is not None else session.boot_options
    effective_gpu_adapter = _extract_gpu_adapter(session, probe)
    effective_first_visible_frame_s, effective_black_loading_s = _extract_visible_frame_timing(session, probe)
    effective_js_heap_used_mb, effective_js_heap_total_mb = _extract_js_heap_mb(session, probe)
    effective_audio_ring_attached, effective_audio_frames_drained = _extract_audio_ring_summary(session, probe)
    effective_present_fps_avg = _extract_present_fps_avg(session, probe)
    effective_hide_hud = bool(hide_hud or session.hide_hud)
    effective_mp4_fps = mp4_fps if mp4_fps is not None else session.mp4_fps
    effective_mp4_skip_black = bool(mp4_skip_black or session.mp4_skip_black)
    summary: dict[str, Any] = {
        "realPackDrawCount": gpu_counts["realPackDrawCount"],
        "fallbackDrawCount": gpu_counts["fallbackDrawCount"],
        "validationErrorCount": gpu_counts["validationErrorCount"],
        "inputScript": effective_input_script,
    }
    if effective_stream_shaders:
        summary["streamShaders"] = True
    if effective_stream_threshold_mb is not None:
        summary["streamThresholdMb"] = effective_stream_threshold_mb
    if session.capture_frames_enabled or session.captured_frames or session.non_black_canvas_frames is not None:
        summary["capturedFrames"] = len(session.captured_frames)
        summary["nonBlackCanvasFrames"] = session.non_black_canvas_frames
    if effective_gpu_adapter is not None:
        summary["gpu_adapter"] = effective_gpu_adapter
    if effective_first_visible_frame_s is not None:
        summary["first_visible_frame_s"] = effective_first_visible_frame_s
        summary["first_visible_frame_ms"] = round(effective_first_visible_frame_s * 1000.0, 2)
    if effective_black_loading_s is not None:
        summary["black_loading_s"] = effective_black_loading_s
        summary["black_loading_ms"] = round(effective_black_loading_s * 1000.0, 2)
    if effective_js_heap_used_mb is not None:
        summary["js_heap_used_mb"] = effective_js_heap_used_mb
    if effective_js_heap_total_mb is not None:
        summary["js_heap_total_mb"] = effective_js_heap_total_mb
    if effective_audio_ring_attached is not None:
        summary["audio_ring_attached"] = effective_audio_ring_attached
    if effective_audio_frames_drained is not None:
        summary["audio_frames_drained"] = effective_audio_frames_drained
    if effective_present_fps_avg is not None:
        summary["present_fps_avg"] = effective_present_fps_avg
    return sanitize_value(
        {
            "schema": "playgta6-browser-telemetry",
            "schema_version": 1,
            "preset": preset,
            "smoke": smoke,
            "headless": headless,
            "hide_hud": effective_hide_hud,
            "inputScript": effective_input_script,
            "gpu_adapter": effective_gpu_adapter,
            "first_visible_frame_s": effective_first_visible_frame_s,
            "black_loading_s": effective_black_loading_s,
            "js_heap_used_mb": effective_js_heap_used_mb,
            "js_heap_total_mb": effective_js_heap_total_mb,
            "audio_ring_attached": effective_audio_ring_attached,
            "audio_frames_drained": effective_audio_frames_drained,
            "present_fps_avg": effective_present_fps_avg,
            "stream_shaders": effective_stream_shaders,
            "stream_threshold_mb": effective_stream_threshold_mb,
            "boot_options": dict(effective_boot_options) if effective_boot_options is not None else None,
            "elapsed_ms": round(float(session.elapsed_ms), 2),
            "status": session.status,
            "crashed": session.crashed,
            "server": session.server_ready,
            "capabilities": probe.get("capabilities") or {},
            "boot_state": probe.get("bootState"),
            "diagnostics": probe.get("diagnostics"),
            "boot_tracer": probe.get("tracerSummary"),
            "hud": probe.get("hud"),
            "summary": summary,
            "presents_sampled": len(session.presents),
            "first_present_ms": session.first_present_ms,
            "scripted_inputs_sent": session.scripted_inputs_sent,
            "captured_frames": list(session.captured_frames),
            "capturedFrames": len(session.captured_frames),
            "nonBlackCanvasFrames": session.non_black_canvas_frames,
            "capture_frames_dir": session.capture_frames_dir,
            "mp4": session.mp4_path,
            "mp4_fps": effective_mp4_fps,
            "mp4_skip_black": effective_mp4_skip_black,
            "page_error_count": len(session.page_errors),
            "warnings": list(session.warnings),
        }
    )


def load_report(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def digest(report: dict[str, Any], exit_code: int, max_counters: int = 12) -> str:
    """Format a compact human-readable run digest (mirroring `run_native_boot.digest`)."""
    lines = [
        f"status: {report.get('status')} (phase {report.get('phase')}, mode {report.get('mode')}, "
        f"exit_code {report.get('exit_code')}, runner exit {exit_code}), {report.get('elapsed_ms')} ms"
    ]
    if report.get("message"):
        lines.append(f"message: {report['message']}")
    lines.append("milestones:")
    milestones = report.get("milestones") or []
    if not milestones:
        lines.append("  (none)")
    for m in milestones:
        seq = f", import call {m['import_seq']}" if "import_seq" in m else ""
        lines.append(f"  {m.get('milestone')}: {m.get('elapsed_ms')} ms{seq}")
    presents = report.get("presents") or []
    fp = report.get("first_present")
    first = f", first at {fp.get('elapsed_ms')} ms" if isinstance(fp, dict) else ""
    counters = report.get("counters") or {}
    total_presents = counters.get("presents", len(presents))
    lines.append(f"presents: {total_presents} ({len(presents)} samples{first})")
    if presents:
        last = presents[-1]
        lines.append(f"  last: wall {last.get('wall_ms')} ms")
        draw_counters = {
            k: v for k, v in (last.get("counters") or {}).items()
            if re.search(r"draw|dispatch|upload|copy|clear|error", k) and v
        }
        for k, v in list(draw_counters.items())[:max_counters]:
            lines.append(f"  {k}: {v}")
    lines.append(
        "counters: "
        f"real_pack_draws={counters.get('real_pack_draws', 0)}, "
        f"fallback_draws={counters.get('fallback_draws', 0)}, "
        f"gpu_validation_errors={counters.get('gpu_validation_errors', 0)}, "
        f"page_errors={counters.get('page_errors', 0)}"
    )
    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    present_fps_avg = _sanitize_non_negative_float(report.get("present_fps_avg"), digits=2)
    if present_fps_avg is None:
        present_fps_avg = _sanitize_non_negative_float(summary.get("present_fps_avg"), digits=2)
    if present_fps_avg is not None:
        lines.append(f"present_fps_avg: {present_fps_avg:.2f}")
    gpu_adapter = sanitize_gpu_adapter(report.get("gpu_adapter") or summary.get("gpu_adapter"))
    if gpu_adapter:
        lines.append(f"gpu_adapter: {gpu_adapter}")
    first_visible_frame_s = _sanitize_non_negative_float(
        report.get("first_visible_frame_s"), digits=3
    )
    if first_visible_frame_s is None:
        first_visible_frame_s = _sanitize_non_negative_float(
            summary.get("first_visible_frame_s"), digits=3
        )
    if first_visible_frame_s is None:
        for ms_val in (
            report.get("first_visible_frame_ms"),
            summary.get("first_visible_frame_ms"),
            summary.get("firstVisibleFrameMs"),
        ):
            clean_ms = _sanitize_non_negative_float(ms_val)
            if clean_ms is not None:
                first_visible_frame_s = _sanitize_non_negative_float(clean_ms / 1000.0, digits=3)
                break
    black_loading_s = _sanitize_non_negative_float(report.get("black_loading_s"), digits=3)
    if black_loading_s is None:
        black_loading_s = _sanitize_non_negative_float(summary.get("black_loading_s"), digits=3)
    if black_loading_s is None:
        for ms_val in (
            report.get("black_loading_ms"),
            summary.get("black_loading_ms"),
            summary.get("blackLoadingMs"),
        ):
            clean_ms = _sanitize_non_negative_float(ms_val)
            if clean_ms is not None:
                black_loading_s = _sanitize_non_negative_float(clean_ms / 1000.0, digits=3)
                break
    if first_visible_frame_s is not None:
        if black_loading_s is not None:
            lines.append(
                f"first_visible_frame_s: {first_visible_frame_s} (black_loading_s={black_loading_s})"
            )
        else:
            lines.append(f"first_visible_frame_s: {first_visible_frame_s}")
    elif black_loading_s is not None:
        lines.append(f"black_loading_s: {black_loading_s}")
    js_heap_used_mb = _sanitize_non_negative_float(report.get("js_heap_used_mb"), digits=1)
    if js_heap_used_mb is None:
        js_heap_used_mb = _sanitize_non_negative_float(summary.get("js_heap_used_mb"), digits=1)
    js_heap_total_mb = _sanitize_non_negative_float(report.get("js_heap_total_mb"), digits=1)
    if js_heap_total_mb is None:
        js_heap_total_mb = _sanitize_non_negative_float(summary.get("js_heap_total_mb"), digits=1)
    heap_parts: list[str] = []
    if js_heap_used_mb is not None:
        heap_parts.append(f"used={js_heap_used_mb:.1f}")
    if js_heap_total_mb is not None:
        heap_parts.append(f"total={js_heap_total_mb:.1f}")
    if heap_parts:
        lines.append(f"js_heap_mb: {', '.join(heap_parts)}")
    audio_ring_attached = _sanitize_optional_bool(report.get("audio_ring_attached"))
    if audio_ring_attached is None:
        audio_ring_attached = _sanitize_optional_bool(summary.get("audio_ring_attached"))
    audio_frames_drained = _sanitize_optional_non_negative_int(report.get("audio_frames_drained"))
    if audio_frames_drained is None:
        audio_frames_drained = _sanitize_optional_non_negative_int(summary.get("audio_frames_drained"))
    audio_parts: list[str] = []
    if audio_ring_attached is not None:
        audio_parts.append(f"attached={'true' if audio_ring_attached else 'false'}")
    if audio_frames_drained is not None:
        audio_parts.append(f"frames_drained={audio_frames_drained}")
    if audio_parts:
        lines.append(f"audio_ring: {', '.join(audio_parts)}")
    input_script = report.get("inputScript") or summary.get("inputScript")
    if input_script:
        lines.append(f"input_script: {input_script}")
    if "capturedFrames" in summary:
        non_black = summary.get("nonBlackCanvasFrames")
        nb_label = "skipped" if non_black is None else str(non_black)
        lines.append(f"captured_frames: {summary['capturedFrames']} (non_black_canvas={nb_label})")
    fault = report.get("fault")
    if isinstance(fault, dict):
        lines.append(f"stop: {fault.get('kind')} at {fault.get('va')} ({fault.get('detail')})")
    else:
        lines.append("stop: no fault")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------


def compare_with_baseline(args: argparse.Namespace, run_dir: Path, runs_dir: Path) -> int:
    """Compare `<run_dir>/boot-report.json` against `--baseline` via `compare_boot_reports`."""
    base = Path(args.baseline)
    if not base.is_dir():
        base = runs_dir / args.baseline
    if not (base / "boot-report.json").is_file():
        raise RunnerError(f"baseline run not found: {sanitize_text(str(args.baseline))}")
    argv = [str(base / "boot-report.json"), str(run_dir / "boot-report.json")]
    for pattern in args.ignore:
        argv += ["--ignore", pattern]
    if args.time_tolerance is not None:
        argv += ["--time-tolerance", str(args.time_tolerance)]
    if args.count_tolerance is not None:
        argv += ["--count-tolerance", str(args.count_tolerance)]
    print(f"[compare] against {basename_label(base)}")
    code = compare_boot_reports.main(argv)
    return 1 if code == 1 else (2 if code == 2 else 0)


# ---------------------------------------------------------------------------
# CLI & main entrypoint
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Rebuild stale browser Wasm64 artifacts, start local COOP/COEP server + Chrome CDP, "
            "collect logs/telemetry/boot-report.json, and optionally compare against a baseline."
        ),
        epilog="Exit code: 0 ok, 1 baseline differences, 2 usage or build error, 3 no report.",
    )
    p.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    p.add_argument("--name", help="run name (default <preset>-<timestamp>)")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="run synthetic browser smoke boot with zero game files (skips recomp/Wasm64 build stages)",
    )
    p.add_argument("--exe", type=Path, help="game executable for recompilation (default $PLAYGTA6_EXE)")
    p.add_argument("--boot-dir", type=Path, help="recompiled boot directory served at /boot/ (default work/recomp_build)")
    p.add_argument("--game-dir", type=Path, help="local game directory served at /game/ (default $GTA6_GAME_DIR)")
    p.add_argument("--shader-pack", type=Path, help="WGPK shader pack (+ optional <pack>.reflect.json sidecar)")
    p.add_argument("--out-dir", type=Path, help="Wasm64 output directory (default web/dist)")
    p.add_argument("--build-dir", type=Path, help="browser build cache directory (default work/browser-boot)")
    p.add_argument("--runs-dir", type=Path, help="run directories (default work/browser-boot/runs)")
    p.add_argument("--runtime-lib", type=Path, help="override PIC libplaygta6_core.a (skips runtime CMake build)")
    p.add_argument("--port", type=int, default=0, help="local server port (default 0 = ephemeral free port)")
    p.add_argument("--duration-s", type=float, default=None, help="run duration in seconds (default from preset)")
    p.add_argument(
        "--frame-interval-s",
        type=float,
        default=DEFAULT_FRAME_INTERVAL_S,
        help="seconds between screenshots when --capture-frames or --mp4 is set (default 1.0)",
    )
    head_group = p.add_mutually_exclusive_group()
    head_group.add_argument("--headless", dest="headless", action="store_true", default=True, help="launch Chrome headless (default)")
    head_group.add_argument("--headed", dest="headless", action="store_false", help="launch Chrome with a visible window")
    p.add_argument("--chrome", type=Path, help="Chrome/Chromium/Edge binary (default $CHROME_PATH or auto-discover)")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="parallel build jobs")
    p.add_argument("--trust-build", action="store_true", help="skip build staleness checks and use existing artifacts")
    p.add_argument("--rebuild", action="store_true", help="force rebuilding all real-game stages")
    p.add_argument("--no-build", action="store_true", help="fail instead of building any stale stage")
    p.add_argument("--dry-run", action="store_true", help="print sanitized build/launch plan as JSON and exit")
    p.add_argument(
        "--capture-frames",
        nargs="?",
        const=True,
        default=None,
        metavar="DIR",
        help=(
            "capture periodic CDP screenshots: pass DIR to write PNGs into DIR "
            "(at --capture-interval-sec), or omit DIR to write JPEGs into <run-dir>/frames/"
        ),
    )
    p.add_argument(
        "--capture-interval-sec",
        type=float,
        default=DEFAULT_CAPTURE_INTERVAL_SEC,
        metavar="SEC",
        help="positive finite seconds between PNG screenshots when --capture-frames <dir> is set (default 2.0)",
    )
    p.add_argument("--mp4", action="store_true", help="encode captured frames to capture.mp4 via ffmpeg")
    p.add_argument(
        "--mp4-fps",
        type=float,
        default=None,
        metavar="FPS",
        help="positive finite framerate for capture.mp4 via ffmpeg (default derived from frame capture interval)",
    )
    p.add_argument(
        "--mp4-skip-black",
        action="store_true",
        help="omit leading black-only boot frames when building capture.mp4 via ffmpeg",
    )
    p.add_argument(
        "--hide-hud",
        action="store_true",
        help="append hud=0 to the target URL for clean fullscreen captures without the HUD card",
    )
    p.add_argument("--baseline", help="earlier run (name in runs-dir or directory path) to compare against")
    p.add_argument("--ignore", action="append", default=[], metavar="PATTERN", help="compare_boot_reports --ignore")
    p.add_argument("--time-tolerance", type=float, metavar="REL", help="compare_boot_reports --time-tolerance")
    p.add_argument("--count-tolerance", type=float, metavar="REL", help="compare_boot_reports --count-tolerance")
    p.add_argument(
        "--input-script",
        default=None,
        metavar="PRESET",
        help=(
            "sanitized scripted-input preset appended as ?inputScript=<preset> "
            "(allowed: start-screen, menu-flow, calibration-skip, intro-watch, "
            "legal-notices, chrome-menu-flow, chrome-full-boot, menu-to-ingame)"
        ),
    )
    p.add_argument(
        "--stream-shaders",
        action="store_true",
        help="append streamShaders=1 to the target URL to enable ranged WGSL pack streaming",
    )
    p.add_argument(
        "--stream-threshold-mb",
        type=float,
        default=None,
        metavar="MB",
        help="positive finite MiB threshold appended as streamThresholdMb=<N> to the target URL",
    )
    p.add_argument(
        "--boot-options",
        default=None,
        metavar="JSON_OR_PATH",
        help="inline JSON object or path to JSON file validated and staged as boot-options.json",
    )
    p.add_argument(
        "--report-to-log",
        action="store_true",
        help="set report_to_log=true in validated boot-options.json",
    )
    p.add_argument(
        "--max-wall-ms",
        type=int,
        default=None,
        metavar="MS",
        help="set max_wall_ms=<uint64> in validated boot-options.json",
    )
    p.add_argument(
        "--dump-pipelines",
        action="store_true",
        help=f'set dump_pipelines_dir="{DEFAULT_DUMP_PIPELINES_DIR}" in validated boot-options.json',
    )
    p.add_argument(
        "--quiet-server",
        action="store_true",
        help="pass --quiet to server/serve_local.py to suppress per-request access logs during long captures",
    )
    return p


def _resolve_game_dir(args: argparse.Namespace, env: Mapping[str, str], repo: Path) -> Path | None:
    if args.game_dir is not None:
        return resolve_against_repo(args.game_dir, repo)
    for key in ("GTA6_GAME_DIR", "PLAYGTA6_GAME_DIR"):
        raw = (env.get(key) or "").strip()
        if raw:
            return resolve_against_repo(raw, repo)
    dotenv_val = load_dotenv_value(repo, "GTA6_GAME_DIR")
    if dotenv_val:
        return resolve_against_repo(dotenv_val, repo)
    return None


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: CommandRunner = run_command,
    session_runner: Callable[..., BrowserSessionResult] | None = None,
    chrome_discoverer: Callable[..., Path | None] = discover_chrome,
    env: Mapping[str, str] | None = None,
    repo_root: Path = REPO_ROOT,
) -> int:
    env_map = os.environ if env is None else env
    try:
        args = build_arg_parser().parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code in (0, None) else 2
    try:
        return _run(
            args,
            runner=runner,
            session_runner=session_runner or run_browser_session,
            chrome_discoverer=chrome_discoverer,
            env=env_map,
            repo=repo_root.resolve(),
        )
    except RunnerError as exc:
        print(f"error: {sanitize_text(str(exc))}", file=sys.stderr)
        return 2


def _run(
    args: argparse.Namespace,
    *,
    runner: CommandRunner,
    session_runner: Callable[..., BrowserSessionResult],
    chrome_discoverer: Callable[..., Path | None],
    env: Mapping[str, str],
    repo: Path,
) -> int:
    if args.rebuild and args.trust_build:
        raise RunnerError("cannot combine --rebuild and --trust-build")
    if args.rebuild and args.no_build:
        raise RunnerError("cannot combine --rebuild and --no-build")
    if args.port < 0 or args.port > 65535:
        raise RunnerError(f"invalid --port {args.port} (expected 0..65535)")
    if args.duration_s is not None and args.duration_s <= 0:
        raise RunnerError("--duration-s must be > 0")
    if not math.isfinite(args.frame_interval_s) or args.frame_interval_s <= 0:
        raise RunnerError("--frame-interval-s must be > 0")
    capture_interval_sec = validate_capture_interval_sec(args.capture_interval_sec)
    capture_frames_enabled, capture_frames_dir = validate_capture_frames_arg(args.capture_frames, repo)
    mp4_fps = validate_mp4_fps(args.mp4_fps)
    mp4_skip_black = bool(args.mp4_skip_black)
    hide_hud = bool(args.hide_hud)
    quiet_server = bool(args.quiet_server)
    if args.jobs < 1:
        raise RunnerError("--jobs must be >= 1")

    input_script = validate_input_script_preset(args.input_script)
    stream_shaders = bool(args.stream_shaders)
    stream_threshold_mb = validate_stream_threshold_mb(args.stream_threshold_mb)
    dump_pipelines = bool(args.dump_pipelines)
    resolved_boot_options = parse_and_validate_boot_options(
        boot_options=args.boot_options,
        report_to_log=bool(args.report_to_log),
        max_wall_ms=args.max_wall_ms,
        dump_pipelines=dump_pipelines,
        repo_root=repo,
    )

    preset_cfg = PRESETS[args.preset]
    if args.duration_s is not None:
        duration_s = float(args.duration_s)
    elif args.smoke and args.preset == "smoke":
        duration_s = SMOKE_MODE_DEFAULT_DURATION_S
    else:
        duration_s = float(preset_cfg["duration_s"])
    send_input = bool(preset_cfg["input"])

    build_dir = resolve_against_repo(args.build_dir or DEFAULT_BUILD_DIR_REL, repo)
    runs_dir = resolve_against_repo(args.runs_dir or (build_dir / "runs"), repo)
    boot_dir = resolve_against_repo(args.boot_dir or DEFAULT_BOOT_DIR_REL, repo)
    out_dir = resolve_against_repo(args.out_dir or DEFAULT_OUT_DIR_REL, repo)
    runtime_lib_override = resolve_against_repo(args.runtime_lib, repo) if args.runtime_lib else None

    exe: Path | None = None
    if args.exe is not None:
        exe = resolve_against_repo(args.exe, repo)
    elif env.get("PLAYGTA6_EXE"):
        exe = resolve_against_repo(env["PLAYGTA6_EXE"], repo)
    game_dir = None if args.smoke else _resolve_game_dir(args, env, repo)
    shader_pack = resolve_against_repo(args.shader_pack, repo) if args.shader_pack else None
    shader_stage_dir = build_dir / "staged-shaders"

    name = args.name or f"{'smoke' if args.smoke and args.preset == 'smoke' else args.preset}-{time.strftime('%Y%m%d-%H%M%S')}"
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise RunnerError(f"invalid run name {name!r}")
    run_dir = runs_dir / name
    if not args.dry_run and run_dir.exists() and any(run_dir.iterdir()):
        raise RunnerError(f"run directory {basename_label(run_dir)} already exists (pick another --name)")

    if not args.smoke and not args.dry_run:
        if exe is not None and not exe.is_file():
            raise RunnerError(f"exe not found: {basename_label(exe)}")
        if args.game_dir is not None and (game_dir is None or not game_dir.is_dir()):
            raise RunnerError(f"game directory not found: {basename_label(args.game_dir)}")
    if shader_pack is not None and not args.dry_run and not shader_pack.is_file():
        raise RunnerError(f"shader pack not found: {basename_label(shader_pack)}")

    if args.trust_build or args.smoke:
        plan = Plan()
    else:
        plan, _runtime_lib = make_build_plan(
            repo=repo,
            build_dir=build_dir,
            boot_dir=boot_dir,
            out_dir=out_dir,
            exe=exe,
            game_dir=game_dir,
            runtime_lib_override=runtime_lib_override,
            jobs=args.jobs,
            force=args.rebuild,
            smoke=args.smoke,
        )

    shader_plan = plan_shader_staging(shader_pack, shader_stage_dir) if shader_pack is not None else []
    chrome_path = chrome_discoverer(args.chrome, env=env)
    sanitized_plan = plan.sanitized(repo)

    for step in sanitized_plan:
        print(f"[plan] {step['stage']}: {step['reason']}", file=sys.stderr)

    effective_shaders_dir = shader_stage_dir if shader_pack is not None else None
    if resolved_boot_options is not None and not args.smoke:
        validate_boot_dir_safety(boot_dir, repo)

    if args.dry_run:
        preview_port = args.port if args.port > 0 else 0
        server_preview = build_server_cmd(
            repo=repo,
            host="127.0.0.1",
            port=preview_port,
            web_dir=repo / "web",
            data_dir=repo / "local-data",
            boot_dir=None if args.smoke else boot_dir,
            game_dir=game_dir,
            shaders_dir=effective_shaders_dir,
            quiet=quiet_server,
        )
        boot_options_staging_preview: dict[str, Any] | None = None
        if resolved_boot_options is not None:
            if args.smoke:
                payload = serialize_boot_options(resolved_boot_options)
                boot_options_staging_preview = sanitize_value(
                    {
                        "filename": BOOT_OPTIONS_NAME,
                        "staged": False,
                        "action": "plan_ephemeral",
                        "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "options": resolved_boot_options,
                    }
                )
            else:
                boot_options_staging_preview = stage_boot_options_file(
                    resolved_boot_options,
                    boot_dir,
                    repo_root=repo,
                    dry_run=True,
                    check_safety=True,
                )
        preview_url = build_target_url(
            "127.0.0.1",
            preview_port or "<ephemeral>",
            smoke=args.smoke,
            input_script=input_script,
            stream_shaders=stream_shaders,
            stream_threshold_mb=stream_threshold_mb,
            hide_hud=hide_hud,
        )
        dry_payload = sanitize_value(
            {
                "mode": "smoke" if args.smoke else "real-game",
                "preset": args.preset,
                "run_name": name,
                "run_dir": repo_relative_label(run_dir, repo),
                "duration_s": duration_s,
                "headless": args.headless,
                "hide_hud": hide_hud,
                "send_input": send_input,
                "input_script": input_script,
                "inputScript": input_script,
                "stream_shaders": stream_shaders,
                "stream_threshold_mb": stream_threshold_mb,
                "boot_options": resolved_boot_options,
                "boot_options_staging": boot_options_staging_preview,
                "dump_pipelines": dump_pipelines,
                "dumpPipelines": dump_pipelines,
                "capture_frames": bool(capture_frames_enabled or args.mp4),
                "capture_frames_dir": repo_relative_label(capture_frames_dir, repo) if capture_frames_dir is not None else None,
                "capture_interval_sec": capture_interval_sec,
                "mp4": bool(args.mp4),
                "mp4_fps": mp4_fps,
                "mp4_skip_black": mp4_skip_black,
                "quiet_server": quiet_server,
                "trust_build": bool(args.trust_build),
                "port": preview_port,
                "chrome_found": chrome_path is not None,
                "chrome_label": basename_label(chrome_path) if chrome_path else None,
                "plan": sanitized_plan,
                "shader_staging": [
                    {"kind": s["kind"], "name": s["name"]} for s in shader_plan
                ],
                "server_command": sanitize_cmd(server_preview, repo),
                "url": preview_url,
            }
        )
        print(json.dumps(dry_payload, indent=2))
        return 0

    if plan.steps and args.no_build:
        raise RunnerError("stale stages with --no-build: " + ", ".join(s["stage"] for s in plan.steps))

    if chrome_path is None:
        raise RunnerError(
            "Chrome/Chromium/Edge executable not found (pass --chrome PATH or set CHROME_PATH)"
        )

    for step in plan.steps:
        sanitized_step_cmd = " ".join(sanitize_cmd(step["cmd"], repo))
        print(f"[build] {step['stage']}: {sanitized_step_cmd}", file=sys.stderr, flush=True)
        if runner(step["cmd"], repo, None, None) != 0:
            raise RunnerError(f"{step['stage']} failed")

    staged_shaders: list[dict[str, Any]] = []
    if shader_pack is not None:
        staged_shaders = stage_shader_pack(shader_pack, shader_stage_dir)

    staged_boot_options: dict[str, Any] | None = None
    if resolved_boot_options is not None and not args.smoke:
        staged_boot_options = stage_boot_options_file(
            resolved_boot_options,
            boot_dir,
            repo_root=repo,
            dry_run=False,
            check_safety=True,
        )

    if not args.smoke and not (out_dir / "runtime.wasm").is_file():
        raise RunnerError(
            f"runtime.wasm not found in {repo_relative_label(out_dir, repo)} after build "
            "(build Wasm64 runtime or pass --smoke)"
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    clean_build_dirs.write_marker(build_dir, "run_browser_boot.py", repo)
    print(f"[run] {repo_relative_label(run_dir, repo)}", file=sys.stderr, flush=True)

    session = session_runner(
        repo=repo,
        run_dir=run_dir,
        chrome_path=chrome_path,
        port=args.port,
        boot_dir=None if args.smoke else boot_dir,
        game_dir=game_dir,
        shaders_dir=effective_shaders_dir,
        duration_s=duration_s,
        headless=args.headless,
        smoke=args.smoke,
        send_input=send_input,
        capture_frames=capture_frames_enabled,
        capture_frames_dir=capture_frames_dir,
        capture_interval_sec=capture_interval_sec,
        frame_interval_s=float(args.frame_interval_s),
        encode_mp4=bool(args.mp4),
        mp4_fps=mp4_fps,
        mp4_skip_black=mp4_skip_black,
        hide_hud=hide_hud,
        input_script=input_script,
        stream_shaders=stream_shaders,
        stream_threshold_mb=stream_threshold_mb,
        boot_options=resolved_boot_options,
        quiet_server=quiet_server,
        runner=runner,
    )

    # Always write console.log, telemetry.json, and command.json
    console_lines = list(session.console_lines)
    hud_lines = ((session.final_probe or {}).get("hud") or {}).get("logLines")
    if isinstance(hud_lines, list) and hud_lines:
        console_lines.append("--- HUD LOG ---")
        console_lines.extend(sanitize_text(str(line)) for line in hud_lines)
    (run_dir / "console.log").write_text(
        ("\n".join(sanitize_text(line) for line in console_lines) + "\n") if console_lines else "",
        encoding="utf-8",
    )

    telemetry_payload = build_telemetry_payload(
        session,
        smoke=args.smoke,
        preset=args.preset,
        headless=args.headless,
        input_script=input_script,
        stream_shaders=stream_shaders,
        stream_threshold_mb=stream_threshold_mb,
        boot_options=resolved_boot_options,
        hide_hud=hide_hud,
        mp4_fps=mp4_fps,
        mp4_skip_black=mp4_skip_black,
    )
    (run_dir / "telemetry.json").write_text(json.dumps(telemetry_payload, indent=2) + "\n", encoding="utf-8")

    command_meta = sanitize_value(
        {
            "preset": args.preset,
            "smoke": bool(args.smoke),
            "name": name,
            "duration_s": duration_s,
            "headless": bool(args.headless),
            "hide_hud": hide_hud,
            "send_input": send_input,
            "input_script": input_script,
            "inputScript": input_script,
            "stream_shaders": stream_shaders,
            "stream_threshold_mb": stream_threshold_mb,
            "boot_options": resolved_boot_options,
            "staged_boot_options": staged_boot_options,
            "dump_pipelines": dump_pipelines,
            "dumpPipelines": dump_pipelines,
            "capture_frames": bool(capture_frames_enabled or args.mp4),
            "capture_frames_dir": repo_relative_label(capture_frames_dir, repo) if capture_frames_dir is not None else None,
            "capture_interval_sec": capture_interval_sec,
            "capturedFrames": len(session.captured_frames),
            "nonBlackCanvasFrames": session.non_black_canvas_frames,
            "mp4": bool(args.mp4),
            "mp4_fps": mp4_fps,
            "mp4_skip_black": mp4_skip_black,
            "quiet_server": quiet_server,
            "trust_build": bool(args.trust_build),
            "chrome": basename_label(chrome_path),
            "plan": sanitized_plan,
            "staged_shaders": staged_shaders,
            "exit_code": session.exit_code,
        }
    )
    (run_dir / "command.json").write_text(json.dumps(command_meta, indent=2) + "\n", encoding="utf-8")

    report_path = run_dir / "boot-report.json"
    if session.ok or session.final_probe or session.presents or session.status in ("ok", "fault"):
        report = build_boot_report(
            session,
            smoke=args.smoke,
            preset=args.preset,
            input_script=input_script,
            stream_shaders=stream_shaders,
            stream_threshold_mb=stream_threshold_mb,
            boot_options=resolved_boot_options,
        )
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    loaded_report = load_report(report_path)
    if loaded_report is None:
        print(
            f"error: browser session (exit {session.exit_code}) wrote no report; "
            f"see {repo_relative_label(run_dir / 'console.log', repo)}",
            file=sys.stderr,
        )
        return 3

    digest_text = digest(loaded_report, session.exit_code)
    (run_dir / "digest.txt").write_text(digest_text + "\n", encoding="utf-8")
    print(digest_text)

    if args.baseline:
        return compare_with_baseline(args, run_dir, runs_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
