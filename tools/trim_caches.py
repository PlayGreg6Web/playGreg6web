#!/usr/bin/env python3
"""Reports and trims every build cache of the project to a size budget (least recently used first).

DRY RUN by default: nothing is deleted without `--apply`. Only the known cache roots below are ever
looked at; entries are removed one whole cache entry at a time (a file, an `.ll`+`.meta` pair, or a key
directory), never the root itself and never anything in a protected set:

  - entries used within the grace period (default 6 h: a build may be linking them),
  - the newest entry of every cache (the generation in use), and
  - anything holding a fresh lock (`*.lock`, `.build.lock`, #1639) or being written (`.tmp*`).

Caches (kind: root, entry, default budget):
  obj      object cache (<cache root>/objcache, PLAYGTA6_OBJ_CACHE, --cache-dir, <drive>:/pg6nb-objcache*)   file      40 GB
  lift     <obj>-lift, PLAYGTA6_LIFT_CACHE (#1475)                                    key dir   20 GB
  shard    <obj>-lift-shards, PLAYGTA6_SHARD_LIFT_CACHE (#1885)                       .ll pair  24 GB
  recomp   <obj>-recomp, PLAYGTA6_RECOMP_CACHE (#1639)                                key dir    5 GB
  targets  <obj>-targets (indirect-target records, #1654)                             key dir    1 GB
  wasm     work/(browser-boot/)wasm_obj_cache of the repo, prebuilt-web builds        file      20 GB
  dawn     %LOCALAPPDATA%/playgta6/dawn-cache (#1750)                                 file       2 GB

Usage:
    python tools/trim_caches.py [--apply] [--budget kind=GB ...] [--grace-h H] [--root DIR ...] [--json]

The builders call `auto_trim_if_low()` when the free disk space drops under `PLAYGTA6_MIN_FREE_GB`
(default 20; 0 disables).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

GB = 1 << 30
DEFAULT_BUDGET_GB = {"obj": 40.0, "lift": 20.0, "shard": 24.0, "recomp": 5.0, "targets": 1.0,
                     "wasm": 20.0, "dawn": 2.0}
GRACE_S = 6 * 3600.0
LOCK_FRESH_S = 24 * 3600.0
MIN_FREE_ENV = "PLAYGTA6_MIN_FREE_GB"
DEFAULT_MIN_FREE_GB = 20.0
OBJ_SUFFIXES = (".obj", ".o")
LOCK_NAMES = (".build.lock",)
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class CacheRoot:
    kind: str
    path: Path


@dataclass
class Entry:
    paths: list
    size: int
    used: float  # newest atime/mtime
    locked: bool = False


@dataclass
class Report:
    root: CacheRoot
    budget: int
    entries: int = 0
    size: int = 0
    victims: list = field(default_factory=list)
    freed: int = 0
    kept_protected: int = 0


# ----------------------------------------------------------------------------------------------
# Discovery: only these roots are ever touched
# ----------------------------------------------------------------------------------------------

def _off(value: str) -> bool:
    return value.lower() in ("0", "off", "no")


def _siblings(obj: Path) -> list:
    name = obj.name
    return [CacheRoot("lift", obj.with_name(name + "-lift")),
            CacheRoot("shard", obj.with_name(name + "-lift-shards")),
            CacheRoot("recomp", obj.with_name(name + "-recomp")),
            CacheRoot("targets", obj.with_name(name + "-targets"))]


def _is_obj_name(name: str) -> bool:
    return name.startswith("pg6nb-objcache") and not name.endswith(("-lift", "-lift-shards", "-recomp", "-targets"))


def discover(env: Mapping[str, str], repo: Path = REPO_ROOT, extra_objs: Sequence[Path] = (),
             scan_dirs: Sequence[Path] | None = None) -> list:
    """The known cache roots that exist on disk (deduplicated)."""
    objs = [Path(p) for p in extra_objs]
    if env.get("PLAYGTA6_OBJ_CACHE"):
        objs.append(Path(env["PLAYGTA6_OBJ_CACHE"]).expanduser())
    else:
        import build_native_boot  # noqa: PLC0415
        objs.append(build_native_boot.default_cache_dir(env))
    for scan in (scan_dirs if scan_dirs is not None else [Path(repo.anchor or "/")]):
        try:
            objs += [p for p in scan.iterdir() if p.is_dir() and _is_obj_name(p.name)]
        except OSError:
            pass
    roots = []
    for obj in objs:
        roots.append(CacheRoot("obj", obj))
        roots += _siblings(obj)
    for kind, var in (("lift", "PLAYGTA6_LIFT_CACHE"), ("shard", "PLAYGTA6_SHARD_LIFT_CACHE"),
                      ("recomp", "PLAYGTA6_RECOMP_CACHE")):
        if env.get(var) and not _off(env[var]):
            roots.append(CacheRoot(kind, Path(env[var]).expanduser()))
    if env.get("PLAYGTA6_INDIRECT_TARGETS") and not _off(env["PLAYGTA6_INDIRECT_TARGETS"]):
        roots.append(CacheRoot("targets", Path(env["PLAYGTA6_INDIRECT_TARGETS"]).expanduser().parent.parent))
    for rel in ("work/browser-boot/wasm_obj_cache", "work/wasm_obj_cache"):
        roots.append(CacheRoot("wasm", repo / rel))
    import cache_root  # noqa: PLC0415
    roots.append(CacheRoot("wasm", cache_root.wasm_cache(env)))
    for legacy in cache_root.legacy_roots(env):  # earlier default locations (deduplicated by path below)
        if legacy.name == "objcache":
            roots.append(CacheRoot("obj", legacy))
            roots += _siblings(legacy)
    local = env.get("LOCALAPPDATA")
    base = Path(local) if local else (Path(env["XDG_CACHE_HOME"]) if env.get("XDG_CACHE_HOME")
                                      else Path.home() / ".cache")
    roots.append(CacheRoot("dawn", base / "playgta6" / "dawn-cache"))
    pw = base / "playgta6" / "prebuilt-web"
    if pw.is_dir():
        roots.append(CacheRoot("wasm", pw / "build" / "wasm_obj_cache"))
        roots += [CacheRoot("wasm", p / "build" / "wasm_obj_cache") for p in pw.glob("*") if p.is_dir()]
    seen = set()
    out = []
    for r in roots:
        try:
            key = str(r.path.resolve()).lower()
            if key in seen or not r.path.is_dir():
                continue
        except OSError:
            continue
        seen.add(key)
        out.append(CacheRoot(r.kind, r.path.resolve()))
    return out


# ----------------------------------------------------------------------------------------------
# Scanning
# ----------------------------------------------------------------------------------------------

def _used(st: os.stat_result) -> float:
    return max(st.st_atime, st.st_mtime)


def _is_lock(name: str) -> bool:
    return name.endswith(".lock") or name in LOCK_NAMES


def _tree(path: Path):
    """(bytes, newest use, holds a fresh lock) of a directory tree; symlinks are never followed."""
    size, used, lock, now = 0, 0.0, False, time.time()
    for dirpath, _, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                st = os.lstat(os.path.join(dirpath, name))
            except OSError:
                continue
            size += st.st_size
            used = max(used, _used(st))
            if _is_lock(name) and now - st.st_mtime < LOCK_FRESH_S:
                lock = True
    return size, used, lock


def scan(root: CacheRoot) -> list:
    """The trimmable entries of a root, in no particular order."""
    entries = []
    now = time.time()
    try:
        children = list(os.scandir(root.path))
    except OSError:
        return entries
    if root.kind in ("lift", "recomp", "targets"):
        for c in children:
            if c.is_symlink() or not c.is_dir(follow_symlinks=False):
                continue
            size, used, lock = _tree(Path(c.path))
            try:
                used = max(used, c.stat(follow_symlinks=False).st_mtime)  # not atime: listing a dir touches it
            except OSError:
                pass
            entries.append(Entry([Path(c.path)], size, used, lock or ".tmp" in c.name))
    elif root.kind == "shard":
        groups = {}
        for c in children:
            if not c.is_file(follow_symlinks=False):
                continue
            stem, _, ext = c.name.rpartition(".")
            if ext not in ("ll", "meta", "tmp"):
                continue
            try:
                st = c.stat(follow_symlinks=False)
            except OSError:
                continue
            e = groups.setdefault(stem if ext != "tmp" else c.name, Entry([], 0, 0.0))
            e.paths.append(Path(c.path))
            e.size += st.st_size
            e.used = max(e.used, st.st_mtime)
            e.locked = e.locked or ext == "tmp"
        entries = list(groups.values())
    else:  # obj, wasm, dawn: single files (dawn and wasm may nest)
        if root.kind == "obj":
            walker = [(str(root.path), [], [c.name for c in children if c.is_file(follow_symlinks=False)])]
        else:
            walker = os.walk(root.path, followlinks=False)
        for dirpath, _, files in walker:
            for name in files:
                p = Path(dirpath) / name
                try:
                    st = os.lstat(p)
                except OSError:
                    continue
                busy = _is_lock(name) or ".tmp" in name
                if root.kind == "obj" and not (busy or name.endswith(OBJ_SUFFIXES)):
                    continue
                entries.append(Entry([p], st.st_size, _used(st), busy and now - st.st_mtime < LOCK_FRESH_S))
    return entries


def plan(root: CacheRoot, budget_bytes: int, grace_s: float = GRACE_S, now: float | None = None) -> Report:
    now = time.time() if now is None else now
    entries = scan(root)
    rep = Report(root, budget_bytes, len(entries), sum(e.size for e in entries))
    if budget_bytes <= 0:
        return rep
    order = sorted(entries, key=lambda e: e.used)
    live = [e for e in order if not e.locked]
    newest = live[-1] if live else None
    total = rep.size
    for e in order:
        if total <= budget_bytes:
            break
        if e is newest or e.locked or now - e.used < grace_s:
            rep.kept_protected += 1
            continue
        rep.victims.append(e)
        total -= e.size
    rep.freed = rep.size - total
    return rep


def _inside(path: Path, root: Path) -> bool:
    try:
        return path.resolve() != root.resolve() and path.resolve().relative_to(root.resolve()) is not None
    except (ValueError, OSError):
        return False


def apply_plan(rep: Report) -> int:
    """Deletes the planned victims (only paths strictly below the cache root). Returns bytes freed."""
    freed = 0
    for e in rep.victims:
        ok = True
        for p in e.paths:
            if not _inside(p, rep.root.path):
                ok = False
                continue
            try:
                if p.is_dir() and not p.is_symlink():
                    shutil.rmtree(p)
                else:
                    p.unlink()
            except OSError:
                ok = False  # in use or already gone
        if ok:
            freed += e.size
    rep.freed = freed
    return freed


def run(roots: Sequence[CacheRoot], budgets_gb: Mapping[str, float], apply: bool, grace_s: float = GRACE_S,
        scale: float = 1.0) -> list:
    reps = []
    for r in roots:
        rep = plan(r, int(budgets_gb.get(r.kind, 0) * scale * GB), grace_s)
        if apply and rep.victims:
            apply_plan(rep)
        reps.append(rep)
    return reps


def _size(n: int) -> str:
    return f"{n / GB:.2f} GB" if n >= GB else f"{n / (1 << 20):.1f} MiB"


def format_reports(reps: Sequence[Report], apply: bool) -> str:
    lines = [f"{'kind':8} {'entries':>7} {'size':>10} {'budget':>8} {'freed' if apply else 'would free':>11}  root"]
    for r in reps:
        lines.append(f"{r.root.kind:8} {r.entries:>7} {_size(r.size):>10} {r.budget / GB:>6.0f}GB "
                     f"{_size(r.freed):>11}  {r.root.path}"
                     + (f"  ({r.kept_protected} protected)" if r.kept_protected else ""))
    lines.append(f"total {_size(sum(r.size for r in reps))}; {'freed' if apply else 'would free'} "
                 f"{_size(sum(r.freed for r in reps))}" + ("" if apply else " (dry run: --apply to delete)"))
    return "\n".join(lines)


def budgets_from(env: Mapping[str, str], overrides: Mapping[str, float] | None = None) -> dict:
    b = dict(DEFAULT_BUDGET_GB)
    for kind, var in (("obj", "PLAYGTA6_OBJ_CACHE_MAX_GB"), ("shard", "PLAYGTA6_SHARD_LIFT_CACHE_GB")):
        if env.get(var):
            try:
                b[kind] = float(env[var])
            except ValueError:
                pass
    b.update(overrides or {})
    return b


# ----------------------------------------------------------------------------------------------
# Builders: trim when the disk runs low
# ----------------------------------------------------------------------------------------------

def min_free_bytes(env: Mapping[str, str]) -> int:
    try:
        return int(float(env.get(MIN_FREE_ENV) or DEFAULT_MIN_FREE_GB) * GB)
    except ValueError:
        return int(DEFAULT_MIN_FREE_GB * GB)


def auto_trim_if_low(env: Mapping[str, str], probe: Path, say: Callable[[str], None] | None = None,
                     extra_objs: Sequence[Path] = (), repo: Path = REPO_ROOT,
                     scan_dirs: Sequence[Path] | None = None,
                     free_fn: Callable[[Path], int] | None = None) -> int:
    """When the free space of `probe`'s volume is below `PLAYGTA6_MIN_FREE_GB` (default 20), trims the
    known caches to their budgets (then to half of them). Logs one line. Returns bytes freed."""
    say = say or (lambda m: print(m, file=sys.stderr, flush=True))
    limit = min_free_bytes(env)
    if limit <= 0:
        return 0
    free_fn = free_fn or (lambda p: shutil.disk_usage(p).free)
    try:
        p = probe if probe.exists() else next(a for a in probe.parents if a.exists())
        free = free_fn(p)
    except (OSError, StopIteration):
        return 0
    if free >= limit:
        return 0
    roots = discover(env, repo, extra_objs, scan_dirs)
    budgets = budgets_from(env)
    freed = sum(r.freed for r in run(roots, budgets, True))
    if free + freed < limit:
        freed += sum(r.freed for r in run(roots, budgets, True, scale=0.5))
    say(f"disk low: {free / GB:.1f} GB free < {limit / GB:.0f} GB ({MIN_FREE_ENV}); trimmed the build caches "
        f"to their budgets, freed {_size(freed)} (python tools/trim_caches.py to inspect)")
    return freed


def main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    env = os.environ if env is None else env
    ap = argparse.ArgumentParser(description="Reports and trims the project's build caches (dry run unless --apply).")
    ap.add_argument("--apply", action="store_true", help="delete (default: dry run)")
    ap.add_argument("--budget", action="append", default=[], metavar="KIND=GB")
    ap.add_argument("--grace-h", type=float, default=GRACE_S / 3600)
    ap.add_argument("--root", action="append", default=[], type=Path, metavar="OBJ_DIR",
                    help="an additional object cache (its -lift/-lift-shards/-recomp/-targets siblings follow)")
    ap.add_argument("--no-scan", action="store_true", help="do not look for <drive>:/pg6nb-objcache*")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    over = {}
    for item in args.budget:
        kind, _, gb = item.partition("=")
        if kind not in DEFAULT_BUDGET_GB:
            ap.error(f"unknown cache kind {kind!r} (one of {', '.join(DEFAULT_BUDGET_GB)})")
        over[kind] = float(gb)
    roots = discover(env, REPO_ROOT, args.root, [] if args.no_scan else None)
    reps = run(roots, budgets_from(env, over), args.apply, args.grace_h * 3600)
    if args.json:
        print(json.dumps([{"kind": r.root.kind, "root": str(r.root.path), "entries": r.entries, "bytes": r.size,
                           "budget": r.budget, "freed": r.freed, "protected": r.kept_protected} for r in reps],
                         indent=2))
    else:
        print(format_reports(reps, args.apply))
    return 0


if __name__ == "__main__":
    sys.exit(main())
