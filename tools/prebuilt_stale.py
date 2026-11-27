#!/usr/bin/env python3
"""Staleness hint for `play_native.ps1 -Prebuilt` / `play_local.ps1 -Prebuilt` (never blocks, never rebuilds).

    prebuilt_stale.py --prebuilt DIR [--repo R] [--web]

Prints one line plus up to 3 subjects when commits between the prebuilt's source commit and HEAD touch
runtime/recomp/harness/shader-pack paths. Prints nothing when git or the commit is unavailable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

# Paths that change what a prebuilt build contains (runtime, recompiler, boot harness, shader packs).
RELEVANT_PATHS = (
    "runtime/",
    "recomp/",
    "shaders/",
    "tools/run_native_boot.py",
    "tools/build_native_boot.py",
    "tools/run_browser_boot.py",
    "tools/stage_browser_boot.py",
    "tools/pack_rdna2_fallback_shaders.py",
    "tools/gen_fallback_shaders.py",
    "tools/prebuilt_web.py",
    "tools/update_prebuilt.ps1",
)


def is_relevant(path: str) -> bool:
    path = path.replace("\\", "/")
    return any(path == p or (p.endswith("/") and path.startswith(p)) for p in RELEVANT_PATHS)


def format_message(count: int, sha: str, date: str, subjects: list[str], web: bool = False) -> str:
    refresh = "tools/update_prebuilt.ps1" + (" -Web" if web else "")
    noun = "commit" if count == 1 else "commits"
    lines = [f"Prebuilt is {count} runtime {noun} behind (built from {sha[:9]}, {date}). It still runs; refresh with {refresh}."]
    lines += [f"  {s}" for s in subjects[:3]]
    return "\n".join(lines)


def _git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def check(prebuilt: Path, repo: Path, web: bool = False) -> str | None:
    try:
        commit = str(json.loads((prebuilt / "prebuilt.json").read_text(encoding="utf-8-sig")).get("commit", ""))
    except (OSError, ValueError):
        return None
    if not commit or _git(repo, "rev-parse", "--verify", "--quiet", commit + "^{commit}") is None:
        return None
    log = _git(repo, "log", "--format=%s", f"{commit}..HEAD", "--", *RELEVANT_PATHS)
    if not log:
        return None
    subjects = log.splitlines()
    date = _git(repo, "log", "-1", "--format=%cs", commit) or "?"
    return format_message(len(subjects), commit, date, subjects, web)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prebuilt", required=True)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--web", action="store_true")
    a = ap.parse_args()
    try:
        msg = check(Path(a.prebuilt), Path(a.repo), a.web)
    except Exception:
        msg = None
    if msg:
        print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
