#!/usr/bin/env python3
"""Machine-wide boot lock: one game boot at a time per PC, so parallel agents do not skew timings or trip watchdogs.

A lock file `<dir>/boot.lock` (default %LOCALAPPDATA%\\playgta6, else ~/.cache/playgta6) is created atomically
(O_CREAT|O_EXCL) and holds JSON {pid, started, name}. A lock whose PID is dead is stale and is broken.
Opt out with --no-boot-lock or PG6_NO_BOOT_LOCK=1 (intentional parallel runs).
"""
from __future__ import annotations

import atexit
import json
import os
import sys
import time
from pathlib import Path
from typing import Mapping

DEFAULT_TIMEOUT_MIN = 60.0


def lock_dir(env: Mapping[str, str]) -> Path:
    if env.get("PG6_BOOT_LOCK_DIR"):
        return Path(env["PG6_BOOT_LOCK_DIR"])
    base = env.get("LOCALAPPDATA")
    return Path(base) / "playgta6" if base else Path.home() / ".cache" / "playgta6"


def disabled(env: Mapping[str, str], flag: bool = False) -> bool:
    return flag or env.get("PG6_NO_BOOT_LOCK", "").strip().lower() not in ("", "0", "false", "no")


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes  # noqa: PLC0415
        k32 = ctypes.windll.kernel32
        k32.OpenProcess.restype = ctypes.c_void_p
        k32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return k32.GetLastError() == 5  # access denied: it exists
        try:
            code = ctypes.c_ulong()
            return bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_owner(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


class BootLock:
    """Context manager. `waited_s` is the time spent waiting; `acquired` is True while held."""

    def __init__(self, directory: Path, name: str = "", timeout_s: float = DEFAULT_TIMEOUT_MIN * 60,
                 poll_s: float = 2.0, out=None, pid: int | None = None):
        self.path = Path(directory) / "boot.lock"
        self.name, self.timeout_s, self.poll_s = name, timeout_s, poll_s
        self.out = out if out is not None else sys.stderr
        self.pid = os.getpid() if pid is None else pid
        self.waited_s = 0.0
        self.broke_stale = False
        self.acquired = False

    def _try_create(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"pid": self.pid, "started": time.time(), "name": self.name}, f)
        return True

    def acquire(self) -> None:
        t0 = time.monotonic()
        announced = False
        while True:
            if self._try_create():
                break
            owner = read_owner(self.path)
            if owner is None:  # unreadable: the creator may be mid-write, so only break it when old
                try:
                    stale = time.time() - self.path.stat().st_mtime > 10
                except OSError:
                    continue
            else:
                stale = not pid_alive(int(owner.get("pid") or 0))
            if stale:
                print(f"[boot-lock] breaking stale lock (owner {owner})", file=self.out, flush=True)
                self.broke_stale = True
                try:
                    self.path.unlink()
                except OSError:
                    pass
                continue
            waited = time.monotonic() - t0
            who = (owner or {}).get("pid", "?")
            run = (owner or {}).get("name", "?")
            if waited >= self.timeout_s:
                raise TimeoutError(f"boot lock {self.path} still held by pid {who} (run {run}) after {waited:.0f} s; "
                                   "use --no-boot-lock / PG6_NO_BOOT_LOCK=1 to run in parallel")
            if not announced:
                started = (owner or {}).get("started")
                age = f", running {time.time() - started:.0f} s" if isinstance(started, (int, float)) else ""
                print(f"[boot-lock] waiting: another boot holds the lock (pid {who}, run {run}{age}); "
                      f"timeout {self.timeout_s:.0f} s", file=self.out, flush=True)
                announced = True
            time.sleep(self.poll_s)
        self.waited_s = time.monotonic() - t0
        self.acquired = True
        atexit.register(self.release)
        if announced:
            print(f"[boot-lock] acquired after {self.waited_s:.0f} s", file=self.out, flush=True)

    def release(self) -> None:
        if not self.acquired:
            return
        self.acquired = False
        owner = read_owner(self.path)
        if owner is not None and owner.get("pid") != self.pid:
            return  # someone broke our lock and took it
        try:
            self.path.unlink()
        except OSError:
            pass

    def __enter__(self) -> "BootLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
