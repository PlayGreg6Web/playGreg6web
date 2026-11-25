#!/usr/bin/env python3
"""Safely delete build/run directories created by the boot runners.

run_native_boot.py and run_browser_boot.py drop a `.playgta6-build-dir` marker
(JSON) at the root of every build dir they create. This tool deletes ONLY
directories that carry that marker at their root, are not the repo (or an
ancestor of it, or contain tracked files), and have not been modified for
--min-age-min minutes. Symlinks and junctions are never followed: they are
removed as links, their targets are left alone.

  python tools/clean_build_dirs.py --scan H:/            # dry run (default)
  python tools/clean_build_dirs.py --scan H:/ --yes      # really delete
  python tools/clean_build_dirs.py H:/nb4 --yes --min-age-min 0

--scan only looks at the direct children of each root (not recursive).
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

MARKER = ".playgta6-build-dir"
REPO = Path(__file__).resolve().parent.parent


def write_marker(build_dir: Path, tool: str, repo: Path = REPO) -> None:
    """Mark build_dir as safe-to-clean (idempotent; never marks the repo itself)."""
    try:
        build_dir = Path(build_dir)
        if build_dir.resolve() == Path(repo).resolve():
            return
        build_dir.mkdir(parents=True, exist_ok=True)
        marker = build_dir / MARKER
        if marker.exists() or marker.is_symlink():
            return
        marker.write_text(json.dumps({
            "tool": tool,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "repo": str(Path(repo).resolve()),
        }, indent=2), encoding="utf-8")
    except OSError:
        pass  # a marker is best effort; never break a run over it


def _is_link(path: Path) -> bool:
    """Symlink or Windows junction / other reparse point."""
    try:
        st = os.lstat(path)
    except OSError:
        return False
    if stat.S_ISLNK(st.st_mode):
        return True
    return bool(getattr(st, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _same_or_inside(path: Path, other: Path) -> bool:
    """True if `path` is `other` or lies inside it."""
    return path == other or other in path.parents


def _has_tracked_files(path: Path, repo: Path) -> bool | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), "ls-files", "--", str(path)],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return bool(out.stdout.strip())


def newest_mtime(root: Path) -> float:
    """Newest mtime in the tree, via lstat, never descending into links."""
    newest = os.lstat(root).st_mtime
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    p = Path(e.path)
                    try:
                        st = e.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    newest = max(newest, st.st_mtime)
                    if e.is_dir(follow_symlinks=False) and not _is_link(p):
                        stack.append(p)
        except OSError:
            continue
    return newest


def tree_size(root: Path) -> int:
    total = 0
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        st = e.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    p = Path(e.path)
                    if _is_link(p):
                        continue
                    if e.is_dir(follow_symlinks=False):
                        stack.append(p)
                    else:
                        total += st.st_size
        except OSError:
            continue
    return total


def check_candidate(path: Path, repo: Path, min_age_min: float, now: float | None = None) -> tuple[bool, str]:
    """Return (deletable, reason)."""
    now = time.time() if now is None else now
    repo = repo.resolve()
    if _is_link(path):
        return False, "is a symlink/junction"
    if not path.is_dir():
        return False, "not a directory"
    real = path.resolve()
    if _same_or_inside(repo, real):
        return False, "is the repo or contains it"
    marker = path / MARKER
    if not marker.is_file() or _is_link(marker):
        return False, "no marker"
    if _same_or_inside(real, repo):
        tracked = _has_tracked_files(real, repo)
        if tracked is None or tracked:
            return False, "inside the repo's tracked tree"
    age_min = (now - newest_mtime(path)) / 60
    if age_min < min_age_min:
        return False, f"modified {age_min:.0f} min ago (< {min_age_min:g})"
    return True, "ok"


def _retry_writable(fn, p: Path) -> None:
    try:
        fn(p)
    except PermissionError:
        os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
        fn(p)


def _remove_link(p: Path) -> None:
    """Remove a symlink/junction itself, never its target."""
    try:
        _retry_writable(os.unlink, p)
    except OSError:
        _retry_writable(os.rmdir, p)  # directory junction on Windows


def remove_tree(root: Path) -> None:
    """Delete root without ever following links out of it."""
    stack: list[tuple[Path, bool]] = [(root, False)]
    while stack:
        d, visited = stack.pop()
        if visited:
            _retry_writable(os.rmdir, d)
            continue
        stack.append((d, True))
        with os.scandir(d) as it:
            entries = list(it)
        for e in entries:
            p = Path(e.path)
            if _is_link(p):
                _remove_link(p)
            elif e.is_dir(follow_symlinks=False):
                stack.append((p, False))
            else:
                _retry_writable(os.unlink, p)


def collect(paths: Sequence[Path], scan: Sequence[Path]) -> list[Path]:
    out: list[Path] = list(paths)
    for root in scan:
        try:
            with os.scandir(root) as it:
                out.extend(Path(e.path) for e in it if e.is_dir(follow_symlinks=False))
        except OSError as exc:
            print(f"warn: cannot scan {root}: {exc}", file=sys.stderr)
    return out


def fmt_size(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GiB"


def main(argv: Sequence[str] | None = None, repo: Path = REPO) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*", type=Path, help="build dirs to clean")
    p.add_argument("--scan", action="append", default=[], type=Path, metavar="ROOT",
                   help="also consider the direct children of ROOT (not recursive)")
    p.add_argument("--min-age-min", type=float, default=60.0, help="skip dirs modified more recently (default 60)")
    p.add_argument("--yes", action="store_true", help="actually delete (default is a dry run)")
    p.add_argument("--dry-run", action="store_true", help="force a dry run even with --yes")
    args = p.parse_args(argv)
    if not args.paths and not args.scan:
        p.error("give paths or --scan ROOT")
    delete = args.yes and not args.dry_run
    freed = 0
    n = 0
    explicit = set(args.paths)
    for path in collect(args.paths, args.scan):
        ok, why = check_candidate(path, repo, args.min_age_min)
        if not ok:
            if path in explicit:
                print(f"skip   {path}: {why}")
            continue
        size = tree_size(path)
        if delete:
            try:
                remove_tree(path)
            except OSError as exc:
                print(f"fail   {path}: {exc}", file=sys.stderr)
                continue
            print(f"delete {path} ({fmt_size(size)})")
        else:
            print(f"would delete {path} ({fmt_size(size)})")
        freed += size
        n += 1
    verb = "freed" if delete else "would free"
    print(f"{n} dir(s), {verb} {fmt_size(freed)}" + ("" if delete else " (dry run; pass --yes to delete)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
