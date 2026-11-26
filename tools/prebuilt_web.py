#!/usr/bin/env python3
"""Helpers for the prebuilt browser build (tools/update_prebuilt.ps1, tools/play_local.ps1 -Prebuilt).

A prebuilt directory is self-contained (no game bytes in git, it lives outside the repo):
    web/            copy of the repo web root plus web/dist (runtime.wasm, guest parts)
    boot/           the recompiled program served at /boot/
    build/          run_browser_boot.py build dir (staged-shaders/, runtime-pic/, wasm_obj_cache/)
    prebuilt.json   {"commit": ..., "built_at": ...}

    prebuilt_web.py prepare --repo R --dest D [--seed-from OLD]   copy web/ (without dist), hard-link the object cache from OLD
    prebuilt_web.py stamp   --repo R --dest D                     write prebuilt.json (commit of R's HEAD)
    prebuilt_web.py info    --prebuilt D [--repo R]               print the source commit and how far behind origin/main
    prebuilt_web.py swap    --new N --target T                    replace T by N (the old one is removed after the rename)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

MANIFEST = "prebuilt.json"


def _git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _link_or_copy(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare(repo: Path, dest: Path, seed_from: Path | None) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    web = dest / "web"
    if web.exists():
        shutil.rmtree(web)
    shutil.copytree(repo / "web", web, ignore=shutil.ignore_patterns("dist"))
    (web / "dist").mkdir(exist_ok=True)
    if seed_from is not None:
        old = seed_from / "build" / "wasm_obj_cache"
        if old.is_dir():
            shutil.copytree(old, dest / "build" / "wasm_obj_cache", copy_function=_link_or_copy, dirs_exist_ok=True)
            print(f"seeded wasm_obj_cache from {old}")


def stamp(repo: Path, dest: Path) -> None:
    commit = _git(repo, "rev-parse", "HEAD")
    if not commit:
        raise SystemExit("cannot read the source commit")
    data = {
        "commit": commit,
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    (dest / MANIFEST).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def info(prebuilt: Path, repo: Path | None) -> str:
    manifest = prebuilt / MANIFEST
    if not manifest.is_file():
        return f"prebuilt: {prebuilt} has no {MANIFEST} (run tools/update_prebuilt.ps1 -Web)"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    commit = str(data.get("commit", ""))
    line = f"prebuilt: commit {commit[:9]} built {data.get('built_at', '?')}"
    if repo is not None:
        _git(repo, "fetch", "--quiet", "origin", "main")
        behind = _git(repo, "rev-list", "--count", f"{commit}..origin/main")
        if behind is None:
            line += " (not comparable to origin/main in this checkout)"
        elif behind == "0":
            line += " (up to date with origin/main)"
        else:
            line += f" ({behind} commits behind origin/main; refresh with tools/update_prebuilt.ps1 -Web)"
    return line


def swap(new: Path, target: Path) -> None:
    old = target.with_name(target.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if target.exists():
        os.replace(target, old)
    try:
        os.replace(new, target)
    except OSError:
        if old.exists():
            os.replace(old, target)
        raise
    shutil.rmtree(old, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--repo", type=Path, required=True)
    a.add_argument("--dest", type=Path, required=True)
    a.add_argument("--seed-from", type=Path)
    a = sub.add_parser("stamp")
    a.add_argument("--repo", type=Path, required=True)
    a.add_argument("--dest", type=Path, required=True)
    a = sub.add_parser("info")
    a.add_argument("--prebuilt", type=Path, required=True)
    a.add_argument("--repo", type=Path)
    a = sub.add_parser("swap")
    a.add_argument("--new", type=Path, required=True)
    a.add_argument("--target", type=Path, required=True)
    args = p.parse_args(argv)
    if args.cmd == "prepare":
        prepare(args.repo, args.dest, args.seed_from)
    elif args.cmd == "stamp":
        stamp(args.repo, args.dest)
    elif args.cmd == "info":
        print(info(args.prebuilt, args.repo))
    else:
        swap(args.new, args.target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
