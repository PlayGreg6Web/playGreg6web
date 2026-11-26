"""One canonical root for every build cache (object, lift, shard lift, recomp, targets, wasm objects).

`PLAYGTA6_CACHE_ROOT` (a directory, or `auto` = `<fixed drive with the most free space>:/playgta6-cache`
on Windows), else `%LOCALAPPDATA%/playgta6/cache` (`$XDG_CACHE_HOME`/`~/.cache` elsewhere). Below it:

    objcache  objcache-lift  objcache-lift-shards  objcache-recomp  objcache-targets  wasm_obj_cache

(the siblings are derived from the object cache path by the existing helpers, so `--cache-dir` and the
per-cache env overrides keep working). Nothing here moves or deletes anything: `legacy_hint()` only
prints, once, when an older cache location still exists.
"""

from __future__ import annotations

import os
import shutil
import string
import sys
from pathlib import Path
from typing import Callable, Mapping

ROOT_ENV = "PLAYGTA6_CACHE_ROOT"
OBJ_ENV = "PLAYGTA6_OBJ_CACHE"
WASM_ENV = "PLAYGTA6_WASM_CACHE"
HINT_ENV = "PLAYGTA6_NO_CACHE_HINT"
OBJ_NAME = "objcache"
SUFFIXES = ("", "-lift", "-lift-shards", "-recomp", "-targets")
HINT_MARKER = ".legacy-hint-shown"


def _local_base(env: Mapping[str, str], windows: bool) -> Path:
    if windows:
        return Path(env["LOCALAPPDATA"]) if env.get("LOCALAPPDATA") else Path.home() / "AppData" / "Local"
    return Path(env["XDG_CACHE_HOME"]) if env.get("XDG_CACHE_HOME") else Path.home() / ".cache"


def _most_free_drive() -> Path | None:
    best, best_free = None, -1
    for letter in string.ascii_uppercase:
        drive = Path(f"{letter}:/")
        try:
            if drive.exists():
                free = shutil.disk_usage(drive).free
                if free > best_free:
                    best, best_free = drive, free
        except OSError:
            continue
    return best / "playgta6-cache" if best else None


def cache_root(env: Mapping[str, str] | None = None, windows: bool | None = None) -> Path:
    env = os.environ if env is None else env
    windows = (os.name == "nt") if windows is None else windows
    value = env.get(ROOT_ENV)
    if value and value.lower() == "auto":
        auto = _most_free_drive() if windows else None
        if auto is not None:
            return auto
    elif value:
        return Path(value).expanduser()
    return _local_base(env, windows) / "playgta6" / "cache"


def obj_cache(env: Mapping[str, str] | None = None, windows: bool | None = None) -> Path:
    env = os.environ if env is None else env
    if env.get(OBJ_ENV):
        return Path(env[OBJ_ENV]).expanduser()
    return cache_root(env, windows) / OBJ_NAME


def wasm_cache(env: Mapping[str, str] | None = None, windows: bool | None = None) -> Path:
    env = os.environ if env is None else env
    if env.get(WASM_ENV):
        return Path(env[WASM_ENV]).expanduser()
    return cache_root(env, windows) / "wasm_obj_cache"


def legacy_roots(env: Mapping[str, str] | None = None, windows: bool | None = None) -> list[Path]:
    """Existing cache dirs of the earlier default locations that are not below the canonical root."""
    env = os.environ if env is None else env
    windows = (os.name == "nt") if windows is None else windows
    canon = str(cache_root(env, windows).resolve()).lower()
    out: list[Path] = []
    old = _local_base(env, windows) / "playgta6"
    cands = [old / (OBJ_NAME + s) for s in SUFFIXES]
    if windows:
        for letter in string.ascii_uppercase:
            try:
                for p in Path(f"{letter}:/").iterdir():
                    if p.name.startswith("pg6nb-objcache"):
                        cands.append(p)
            except OSError:
                continue
    for p in cands:
        try:
            if p.is_dir() and not str(p.resolve()).lower().startswith(canon):
                out.append(p)
        except OSError:
            pass
    return out


def legacy_hint(env: Mapping[str, str] | None = None, say: Callable[[str], None] | None = None,
                explicit: bool = False, windows: bool | None = None) -> bool:
    """Prints a one-time hint (marker file in the canonical root) when legacy cache dirs exist."""
    env = os.environ if env is None else env
    if explicit or env.get(OBJ_ENV) or env.get(HINT_ENV):
        return False
    old = legacy_roots(env, windows)
    root = cache_root(env, windows)
    marker = root / HINT_MARKER
    if not old or marker.exists():
        return False
    say = say or (lambda m: print(m, file=sys.stderr, flush=True))
    say(f"note: build caches now live under {root} (PLAYGTA6_CACHE_ROOT). Older cache dirs still exist: "
        + ", ".join(str(p) for p in old[:6]) + (" ..." if len(old) > 6 else ""))
    say("      size/trim them: python tools/trim_caches.py [--apply]; reuse the %LOCALAPPDATA% ones by moving them "
        f"(same names) into {root}, or delete them. Nothing is moved automatically. Silence: {HINT_ENV}=1")
    try:
        root.mkdir(parents=True, exist_ok=True)
        marker.write_text("shown")
    except OSError:
        pass
    return True
