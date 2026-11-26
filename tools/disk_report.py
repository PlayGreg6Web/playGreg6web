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


def registry_user_env() -> dict:
    """User-level environment variables (HKCU\Environment); empty off Windows or on error.
    The process env of a tool/agent often lacks vars set via SetEnvironmentVariable(..., "User")."""
    try:
        import winreg  # type: ignore
    except ImportError:
        return {}
    out: dict = {}
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            i = 0
            while True:
                try:
                    name, val, _t = winreg.EnumValue(key, i)
                except OSError:
                    break
                i += 1
                if isinstance(val, str):
                    out[name] = os.path.expandvars(val)
    except OSError:
        pass
    return out


def merged_env(environ: dict, registry: "dict | None" = None) -> dict:
    """Registry user vars first, the process env wins on conflicts."""
    reg = registry_user_env() if registry is None else registry
    return {**reg, **environ}


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


def _is_hex64(stem: str) -> bool:
    return len(stem) == 64 and all(c in "0123456789abcdef" for c in stem)


def _looks_like_objcache(path: str, low: list) -> bool:
    """Layout of tools/build_native_boot.py's --cache-dir: flat `<sha256>.obj|.o` (+ .lock/.tmp)."""
    name = os.path.basename(path.rstrip("\/")).lower()
    if "objcache" in name:
        return True
    hits = sum(1 for n in low if _is_hex64(n.split(".", 1)[0]) and n.endswith((".obj", ".o")))
    return hits >= 1 and hits * 2 >= len(low)


def shader_pack_files(path: str) -> list:
    """*.wgsl.pack files in the top two levels of `path` (dir itself and its child dirs)."""
    found: list = []
    for n in _listing(path):
        full = os.path.join(path, n)
        if n.lower().endswith(".wgsl.pack"):
            found.append(full)
        elif os.path.isdir(full) and not os.path.islink(full):
            found += [os.path.join(full, m) for m in _listing(full) if m.lower().endswith(".wgsl.pack")]
    return found


def _newest_mtime(files: list) -> float:
    best = 0.0
    for f in files:
        try:
            best = max(best, os.stat(f).st_mtime)
        except OSError:
            pass
    return best


def classify(path: str, worktrees: Iterable[str] = ()) -> str:
    name = os.path.basename(path.rstrip("\\/")).lower()
    if any(_norm(w) == _norm(path) for w in worktrees):
        return "git worktree"
    low = [n.lower() for n in _listing(path)]
    gitp = os.path.join(path, ".git")
    if os.path.exists(gitp):
        return "git worktree" if os.path.isfile(gitp) else "git clone"
    if _looks_like_objcache(path, low):
        return "object cache"
    if shader_pack_files(path) or any(n.endswith((".wgsl.pack", ".reflect")) for n in low):
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
    if kind == "object cache":
        return "NO (in use, keep)", ""
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


def build_report(cands: list, worktrees: list, envrefs: dict, workers: int = 16,
                 work_dir: "str | None" = None) -> list:
    """`work_dir`: the dir play_native.ps1 auto-picks the newest *.wgsl.pack from (repo work/)."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        sizes = list(ex.map(dir_size, cands))
    kinds = [classify(c, worktrees) for c in cands]
    std = [(_newest_mtime(shader_pack_files(c)), c) for c, k in zip(cands, kinds)
           if k == "shader pack" and os.path.basename(c.rstrip("\/")).lower().startswith("pg6nb-std")]
    newest_std = _norm(max(std)[1]) if std else None
    auto = None
    if work_dir and os.path.isdir(work_dir):
        packs = shader_pack_files(work_dir)
        if packs:
            auto = _norm(max(packs, key=lambda f: _newest_mtime([f])))
    rows = []
    for path, (size, files), kind in zip(cands, sizes, kinds):
        refs = [f"env:{k}" for p, k in envrefs.items() if _inside(p, path)]
        if kind == "shader pack" and auto and _inside(auto, path):
            refs.append("auto-pick")
        refs += ["worktree:" + os.path.basename(w) for w in worktrees
                 if _inside(w, path) and _norm(w) != _norm(path) and _norm(w) != _norm(REPO)]
        safe, cmd = hint(kind, path, refs)
        if kind == "shader pack" and not refs and newest_std == _norm(path):
            safe = "keep (standard pack)"
            cmd = ""
            refs = ["std pack"]
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
    rows = build_report(cands, git_worktrees(REPO), env_references(merged_env(dict(os.environ))), a.workers,
                        str(REPO / "work"))
    print(render(rows, a.top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
