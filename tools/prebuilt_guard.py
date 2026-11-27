"""Keeps a prebuilt build (a directory holding prebuilt.json) read-only for the runners.

A boot run from a prebuilt used to put runs/, program state, the build lock and the
.playgta6-build-dir marker inside it, and a later "delete program/" cleanup emptied the prebuilt.
Now: runs and the build lock live outside (default %LOCALAPPDATA%\playgta6\runs\<prebuilt-name>\,
else the sibling `<dir>-runs`), no marker is written inside, and every deleting code path calls
refuse_inside_prebuilt().
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

MANIFEST = "prebuilt.json"


class PrebuiltProtected(RuntimeError):
    pass


def containing_prebuilt(path: Path | str) -> Path | None:
    """The prebuilt directory that is `path` or contains it, else None."""
    p = Path(path).expanduser()
    try:
        p = p.resolve()
    except OSError:
        p = p.absolute()
    for cand in (p, *p.parents):
        if (cand / MANIFEST).is_file():
            return cand
    return None


def refuse_inside_prebuilt(path: Path | str, what: str = "delete") -> None:
    pb = containing_prebuilt(path)
    if pb is not None:
        raise PrebuiltProtected(
            f"refusing to {what} {path}: it is inside the prebuilt build {pb} (prebuilt.json present). "
            f"A prebuilt is read-only for the runners; refresh it with tools/update_prebuilt.ps1.")


def runs_root(prebuilt: Path, env: Mapping[str, str] | None = None) -> Path:
    """Where runs of `prebuilt` go: <LOCALAPPDATA>/playgta6/runs/<name>, else the sibling <dir>-runs."""
    env = os.environ if env is None else env
    prebuilt = Path(prebuilt).resolve()
    la = env.get("LOCALAPPDATA")
    if la:
        return Path(la) / "playgta6" / "runs" / prebuilt.name
    return prebuilt.with_name(prebuilt.name + "-runs")


def default_runs_dir(build_dir: Path, env: Mapping[str, str] | None = None) -> Path | None:
    """The outside runs dir when build_dir is (inside) a prebuilt, else None (use <build>/runs)."""
    pb = containing_prebuilt(build_dir)
    return runs_root(pb, env) if pb is not None else None


def lock_dir(build_dir: Path, env: Mapping[str, str] | None = None) -> Path:
    """Directory that holds the build lock: the build dir, or outside a prebuilt."""
    out = default_runs_dir(build_dir, env)
    return out if out is not None else Path(build_dir)
