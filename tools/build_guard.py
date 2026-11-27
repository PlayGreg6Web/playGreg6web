"""Build-speed and build-safety helpers for `tools/run_native_boot.py`.

* Prebuilt recompiler cache: `playgta6-recomp` is stored under a key made of the content hashes
  of everything it is built from (see `lift_cache.recomp_source_files`) plus the rustc version, so
  a fresh worktree or checkout with identical recompiler sources copies the binary in instead of
  running a cold `cargo build --release` (minutes of CPU).
* Build-directory lock: one run owns a build dir at a time. A second run on the same directory
  fails at once with a message naming the holder, instead of two builds writing the same
  program/ and native/ trees. The lock is the heartbeat-kept lock file of build_native_boot.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping

RECOMP_CACHE_ENV = "PLAYGTA6_RECOMP_CACHE"  # directory, or "off"
RECOMP_CACHE_KEEP = 3
RECOMP_CACHE_VERSION = 1
RECOMP_KEY_SUFFIX = ".key"  # stamp beside the binary: the key it was built or restored from
BUILD_LOCK_NAME = ".build.lock"


class BuildDirBusy(Exception):
    """The build directory is in use by another run."""


def recomp_key(source_hashes: Mapping[str, str], toolchain: str) -> str:
    """Key of a recompiler binary: source file name -> sha256, plus the toolchain string."""
    h = hashlib.sha256(f"playgta6-recomp v{RECOMP_CACHE_VERSION}\n".encode())
    for name in sorted(source_hashes):
        h.update(f"src {name} {source_hashes[name]}\n".encode())
    h.update(f"toolchain {toolchain}\n".encode())
    return h.hexdigest()[:32]


def recomp_inputs_key(repo: Path, toolchain: str) -> str:
    from lift_cache import file_digest, recomp_source_files  # noqa: PLC0415
    return recomp_key({f.relative_to(repo).as_posix(): file_digest(f) for f in recomp_source_files(repo)}, toolchain)


def rustc_version() -> str:
    try:
        out = subprocess.run(["rustc", "-vV"], capture_output=True, text=True, timeout=30)
        return " ".join(out.stdout.split()) if out.returncode == 0 else "rustc-unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "rustc-unknown"


def recomp_cache_dir(cache_dir: Path | None, env: Mapping[str, str]) -> Path | None:
    """`PLAYGTA6_RECOMP_CACHE` ("off" disables), else `<object cache>-recomp` beside the object cache."""
    value = env.get(RECOMP_CACHE_ENV)
    if value:
        return None if value.lower() in ("0", "off", "no") else Path(value).expanduser().resolve()
    import build_native_boot  # noqa: PLC0415
    obj = (cache_dir if cache_dir is not None else build_native_boot.default_cache_dir(env)).resolve()
    return obj.with_name(obj.name + "-recomp")


def stamp_path(binary: Path) -> Path:
    return binary.with_name(binary.name + RECOMP_KEY_SUFFIX)


def read_recomp_stamp(binary: Path) -> str | None:
    try:
        return stamp_path(binary).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def recomp_cache_restore(cache: Path, key: str, binary: Path) -> None:
    binary.parent.mkdir(parents=True, exist_ok=True)
    src = cache / key / binary.name
    tmp = binary.with_name(binary.name + f".tmp-{os.getpid()}")
    shutil.copy2(src, tmp)
    os.replace(tmp, binary)
    os.utime(binary)  # newer than the sources: cargo and the staleness check see it as fresh
    os.utime(cache / key)
    stamp_path(binary).write_text(key + "\n", encoding="utf-8")


def recomp_cache_store(cache: Path, key: str, binary: Path, keep: int = RECOMP_CACHE_KEEP) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    entry = cache / key
    if not (entry / binary.name).is_file():
        tmp = cache / f"{key}.tmp-{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        shutil.copy2(binary, tmp / binary.name)
        try:
            tmp.rename(entry)
        except OSError:  # another checkout stored the same key meanwhile
            shutil.rmtree(tmp, ignore_errors=True)
    stamp_path(binary).write_text(key + "\n", encoding="utf-8")
    entries = sorted((e for e in cache.iterdir() if e.is_dir() and ".tmp-" not in e.name),
                     key=lambda e: e.stat().st_mtime, reverse=True)
    for old in entries[max(keep, 1):]:
        if old.name != key:
            shutil.rmtree(old, ignore_errors=True)


def lock_holder(lock: Path) -> str:
    try:
        return lock.read_text(encoding="utf-8").strip() or "unknown process"
    except OSError:
        return "unknown process"


@contextmanager
def build_dir_lock(build_dir: Path, stale_s: float | None = None, lock_dir: Path | None = None) -> Iterator[Path]:
    """Holds `<build_dir>/.build.lock` for the duration; raises BuildDirBusy when another live run has it."""
    import build_native_boot  # noqa: PLC0415
    lock_dir = build_dir if lock_dir is None else lock_dir  # outside a prebuilt (see prebuilt_guard.lock_dir)
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = lock_dir / BUILD_LOCK_NAME
    for _ in range(2):  # the second try follows the removal of a stale lock
        if build_native_boot.try_lock(lock, stale_s):
            break
    else:
        raise BuildDirBusy(
            f"build directory {build_dir} is in use by another run (lock {lock}, held by "
            f"'{lock_holder(lock)}' = pid and host). Pass a different --build-dir for a parallel run, "
            f"or wait; a lock whose process died is taken over after "
            f"{int(build_native_boot.LOCK_STALE_S)} s.")
    try:
        yield lock
    finally:
        build_native_boot.release_lock(lock)
