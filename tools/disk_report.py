#!/usr/bin/env python3
"""Read-only disk usage report for the pg6* build/run/worktree directories.

Scans candidate directories (default: top-level `pg6*` dirs on the drive the repo is on, plus the
children of the repo's `work/`), sizes them in parallel (os.scandir, reparse points/symlinks are
never followed), classifies them, marks which are still referenced (PLAYGTA6_* env vars, git
worktrees) and prints a table sorted by size with a "safe to delete?" hint and the command you can
run yourself. This tool NEVER deletes anything.

  python tools/disk_report.py                 # default roots
  python tools/disk_report.py --root H:/ --pattern "pg6*" --top 20
  python tools/disk_report.py --root H:/dev/x --all   # every child of a root, not only pg6*
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parent.parent
GIB = 1024 ** 3


def _is_link(entry: "os.DirEntry[str]") -> bool:
    try:
        if entry.is_symlink():
            return True
        st = entry.stat(follow_symlinks=False)
        return bool(getattr(st, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    except OSError:
        return True


def dir_size(path: str) -> tuple[int, int]:
    """(bytes, file count); iterative, skips reparse points and unreadable dirs."""
    total = files = 0
    stack = [path]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for e in it:
                    try:
                        if _is_link(e):
                            continue
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        else:
                            total += e.stat(follow_symlinks=False).st_size
                            files += 1
                    except OSError:
                        pass
        except OSError:
            pass
    return total, files


def _norm(p: "str | os.PathLike") -> str:
    return os.path.normcase(os.path.abspath(str(p)))


def _inside(child: str, parent: str) -> bool:
    c, p = _norm(child), _norm(parent)
    return c == p or c.startswith(p.rstrip("\\/") + os.sep)


def git_worktrees(repo: Path) -> list[str]:
    try:
        out = subprocess.run(["git", "-C", str(repo), "worktree", "list", "--porcelain"],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line[9:].strip() for line in out.splitlines() if line.startswith("worktree ")]


def env_references(environ: dict) -> dict:
    """path -> env var name, for every PLAYGTA6_* / GTA6_* variable whose value is an existing path."""
    refs: dict = {}
    for k, v in environ.items():
        if k.startswith(("PLAYGTA6_", "GTA6_")) and v and os.path.exists(v):
            refs[_norm(v)] = k
    return refs


def _listing(path: str, limit: int = 400) -> list:
    try:
        with os.scandir(path) as it:
            return [e.name for _, e in zip(range(limit), it)]
    except OSError:
        return []


def classify(path: str, worktrees: Iterable[str] = ()) -> str:
    name = os.path.basename(path.rstrip("\\/")).lower()
    if any(_norm(w) == _norm(path) for w in worktrees):
        return "git worktree"
    low = [n.lower() for n in _listing(path)]
    gitp = os.path.join(path, ".git")
    if os.path.exists(gitp):
        return "git worktree" if os.path.isfile(gitp) else "git clone"
    if any(n.endswith((".wgsl.pack", ".reflect")) for n in low):
        return "shader pack"
    if "boot-report.json" in low or name == "runs":
        return "run dir"
    runs = os.path.join(path, "runs")
    if os.path.isdir(runs):
        for r in _listing(runs, 50):
            if os.path.exists(os.path.join(runs, r, "boot-report.json")):
                return "run dir"
    if name in ("program", "build") or "harness-build" in name or any(
            n in low for n in ("program", "harness-build", "cmakecache.txt", ".playgta6-build-dir")):
        return "build output"
    if "save" in name or any("save" in n for n in low[:50]):
        return "save copy"
    if "log" in name or (low and all(n.endswith((".log", ".txt")) for n in low)):
        return "logs"
    return "other"


def quote(p: str) -> str:
    return '"' + p.replace('"', '`"') + '"'


def hint(kind: str, path: str, refs: list) -> tuple:
    """(safe label, command to run yourself)."""
    if kind == "git worktree":
        return "NO (worktree)", f"git worktree remove {quote(path)}   # only after its branch is merged/abandoned"
    if kind == "git clone":
        return "NO (git clone)", ""
    if refs:
        return "NO (in use)", ""
    rm = f"Remove-Item -Recurse -Force -LiteralPath {quote(path)}"
    if kind in ("build output", "run dir", "logs"):
        return "yes", rm
    if kind == "shader pack":
        return "maybe (regen is slow)", rm
    if kind == "save copy":
        return "maybe (check saves)", rm
    return "unknown", rm


def candidates(roots: list, pattern: str, every: bool) -> list:
    out: list = []
    for r in roots:
        try:
            with os.scandir(r) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False) and not _is_link(e) and (
                                every or fnmatch.fnmatch(e.name.lower(), pattern.lower())):
                            out.append(e.path)
                    except OSError:
                        pass
        except OSError:
            pass
    return out


def build_report(cands: list, worktrees: list, envrefs: dict, workers: int = 16) -> list:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        sizes = list(ex.map(dir_size, cands))
    rows = []
    for path, (size, files) in zip(cands, sizes):
        kind = classify(path, worktrees)
        refs = [f"env:{k}" for p, k in envrefs.items() if _inside(p, path)]
        refs += ["worktree:" + os.path.basename(w) for w in worktrees
                 if _inside(w, path) and _norm(w) != _norm(path) and _norm(w) != _norm(REPO)]
        safe, cmd = hint(kind, path, refs)
        if kind == "git worktree" and refs:
            safe = "NO (worktree, in use)"
        rows.append(dict(path=path, size=size, files=files, kind=kind, refs=refs, safe=safe, cmd=cmd))
    rows.sort(key=lambda r: r["size"], reverse=True)
    return rows


def fmt_size(n: int) -> str:
    return f"{n / GIB:.2f} GiB" if n >= GIB / 10 else f"{n / 1024 ** 2:.1f} MiB"


def render(rows: list, top: int) -> str:
    lines = [f"{'size':>11}  {'kind':<13} {'safe to delete?':<22} {'referenced by':<24} path / command"]
    for r in rows[:top]:
        ref = ",".join(r["refs"]) or "-"
        lines.append(f"{fmt_size(r['size']):>11}  {r['kind']:<13} {r['safe']:<22} {ref[:24]:<24} {r['path']}")
        if r["cmd"]:
            lines.append(f"{'':>11}  -> {r['cmd']}")
    lines.append(f"total {fmt_size(sum(r['size'] for r in rows))} in {len(rows)} dirs "
                 "(read-only report; nothing was deleted)")
    return "\n".join(lines)


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", help="directory whose children are scanned (repeatable)")
    ap.add_argument("--pattern", default="pg6*", help="child-name glob for --root dirs (default pg6*)")
    ap.add_argument("--all", action="store_true", help="scan every child of the roots, ignore --pattern")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    if a.root:
        cands = candidates(a.root, a.pattern, a.all)
    else:
        drive = os.path.splitdrive(str(REPO))[0]
        cands = candidates([drive + os.sep if drive else "/"], a.pattern, a.all) \
            + candidates([str(REPO / "work")], "*", True)
    seen: set = set()
    cands = [c for c in cands if not (_norm(c) in seen or seen.add(_norm(c)))]
    cands = [c for c in cands if _norm(c) != _norm(REPO)]
    rows = build_report(cands, git_worktrees(REPO), env_references(dict(os.environ)), a.workers)
    print(render(rows, a.top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
