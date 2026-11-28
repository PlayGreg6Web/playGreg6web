#!/usr/bin/env python3
"""Local dev & asset-streaming HTTP server for playgta6web.

- Serves `web/` at `/` and `--data` (default `local-data/`) at `/data/`.
- Serves `GTA6_GAME_DIR` (when configured) at `/game/` for local game & splash assets.
- Serves `local-data/shaders/` at `/shaders/` and `work/recomp_build/` at `/boot/`
  (boot allowlisted to `.pgim`/`.json`/`.wasm`/`.js` only).
- Serves sanitized JSON readiness diagnostics at `/api/status` (including staged
  `/dist/*.wasm` side-module discovery).
- Optionally enables `POST /__pg6_dump/<relpath>` (with `?append=1`) when started with
  `--allow-dump-dir <dir>` to capture live Chrome Wasm64 pipeline dumps strictly inside
  the configured directory with traversal protection and per-file size caps.
- Sends Cross-Origin Isolation headers (COOP/COEP/CORP) required for SharedArrayBuffer.
- Uses HTTP/1.1 persistent keep-alive connections with large streaming buffers (1 MiB)
  and a bounded thread-safe read-only file-handle cache for `.rpf`/`.pack`/`.pgim`/`.wasm`.
- Supports HTTP `Range` (`bytes=start-end`, `bytes=start-`, `bytes=-suffix`), `HEAD`,
  `OPTIONS`, `Cache-Control: no-cache`, deterministic strong `ETag`/`Last-Modified`,
  `If-Match` (`412`), `If-None-Match` (`304`), and `If-Range`.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import contextlib
import ctypes
import errno
import importlib.util
import io
import json
import mimetypes
import ntpath
import os
import posixpath
import re
import socket
import stat
import sys
import tempfile
import threading
import time
from datetime import timezone
from email.utils import formatdate, parsedate_to_datetime
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import parse_qs, unquote, urlsplit

if os.name == "nt":
    import msvcrt

RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$", re.IGNORECASE)

BOOT_ALLOWED_SUFFIXES = (".pgim", ".json", ".wasm", ".js")
CACHEABLE_IMMUTABLE_SUFFIXES = (
    ".rpf",
    ".rpf9",
    ".pack",
    ".pgim",
    ".wasm",
    ".bk2",
    ".bik",
    ".awc",
    ".ttpf",
    ".reflect.json",
)
DEFAULT_FILE_HANDLE_CACHE_SIZE = 32
DEFAULT_CONCURRENT_FD_POOL_SIZE = 8  # Concurrent read-only FDs per cached file on non-pread OSes
DEFAULT_MAX_TOTAL_OPEN_FDS = 256  # Global open FD budget across all cached entries
DEFAULT_RESOLVED_PATH_CACHE_SIZE = 512  # Bounded LRU cache for verified non-symlink resolved paths
DEFAULT_INTERSTAGE_CACHE_SIZE = 16  # Bounded LRU cache for dynamically extracted .interstage.json bytes
INTERSTAGE_SIDECAR_SUFFIX = ".interstage.json"
REFLECT_SIDECAR_SUFFIX = ".reflect.json"
STAGE_REFLECTION_SCHEMA = "playgta6-stage-reflection"
STAGE_REFLECTION_INTERSTAGE_VERSION = 3
RUNTIME_WASM_NAME = "runtime.wasm"
PREFERRED_SHADER_PACK_NAMES = ("splash.wgsl.pack", "fallback.wgsl.pack")
SAFE_SIDE_MODULE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.wasm$", re.IGNORECASE)

DUMP_URL_PREFIX = "/__pg6_dump/"
DEFAULT_MAX_DUMP_FILE_BYTES = 256 * 1024 * 1024  # 256 MiB default cap per dumped file
_DUMP_WRITE_LOCK = threading.Lock()
_LOG_LOCK = threading.Lock()

STREAM_CHUNK_BYTES = 1024 * 1024  # 1 MiB streaming buffer for large full-file 200 OK reads
STREAM_CHUNK_SIZE = STREAM_CHUNK_BYTES  # Backwards-compatible alias
RANGE_STREAM_CHUNK_BYTES = 256 * 1024  # 256 KiB streaming buffer for 206 Partial Content (W368)
SOCKET_SNDBUF_BYTES = 1024 * 1024  # 1 MiB TCP send buffer hint for loopback Range throughput
MAX_METHOD_NOT_ALLOWED_DRAIN_BYTES = 65536  # Drain small bodies on 405 to prevent Windows RST races
_STREAM_BUFFERS = threading.local()
KEEP_ALIVE_TIMEOUT = 30.0  # Idle socket timeout (seconds) for HTTP/1.1 keep-alive connections
ALLOWED_METHODS = "GET, HEAD, OPTIONS"
DUMP_ALLOWED_METHODS = "POST, OPTIONS"

CLIENT_DISCONNECT_ERRORS = (
    BrokenPipeError,
    ConnectionResetError,
    ConnectionAbortedError,
    TimeoutError,
)
_DISCONNECT_WINERRORS = frozenset({10038, 10053, 10054, 10057, 10058})
_DISCONNECT_ERRNOS = frozenset(
    code
    for code in (
        getattr(errno, "EPIPE", None),
        getattr(errno, "ECONNRESET", None),
        getattr(errno, "ECONNABORTED", None),
        getattr(errno, "ESHUTDOWN", None),
        getattr(errno, "ENOTCONN", None),
    )
    if code is not None
)


def _is_client_disconnect_error(exc: BaseException | None) -> bool:
    """Return True when `exc` represents a benign client socket disconnect/abort."""
    if exc is None:
        return False
    if isinstance(exc, CLIENT_DISCONNECT_ERRORS):
        return True
    if isinstance(exc, OSError):
        winerr = getattr(exc, "winerror", None)
        if winerr in _DISCONNECT_WINERRORS:
            return True
        if exc.errno in _DISCONNECT_ERRNOS:
            return True
    return False


_DATE_CACHE_LOCK = threading.Lock()
_DATE_CACHE_SEC: int = -1
_DATE_CACHE_STR: str = ""


def _cached_http_date_now() -> str:
    """Return RFC 9110 Date header string cached at 1-second resolution."""
    global _DATE_CACHE_SEC, _DATE_CACHE_STR
    now_sec = int(time.time())
    if now_sec == _DATE_CACHE_SEC and _DATE_CACHE_STR:
        return _DATE_CACHE_STR
    formatted = formatdate(now_sec, usegmt=True)
    with _DATE_CACHE_LOCK:
        _DATE_CACHE_SEC = now_sec
        _DATE_CACHE_STR = formatted
    return formatted

mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/json", ".json")


def _try_resolve_root(root: Path | None) -> Path | None:
    """Resolve an optional root directory path once, returning None on failure."""
    if root is None:
        return None
    try:
        return root.resolve()
    except OSError:
        return None


def _normalize_safe_relpath(rel_url_path: str) -> str | None:
    """Pure-Python lexical validation and normalization of a relative URL path."""
    decoded = unquote(rel_url_path)
    if "\x00" in decoded:
        return None
    unslashed = decoded.replace("\\", "/")
    if ".." in unslashed.split("/"):
        return None
    normalized = posixpath.normpath("/" + unslashed).lstrip("/")
    if not normalized or normalized.startswith(".."):
        return None
    return normalized


def safe_resolve(
    root: Path,
    rel_url_path: str,
    *,
    root_resolved: Path | None = None,
) -> Path | None:
    """Resolve a URL path strictly inside `root`, rejecting traversal and external symlinks."""
    normalized = _normalize_safe_relpath(rel_url_path)
    if normalized is None:
        return None
    try:
        resolved_root = root_resolved if root_resolved is not None else root.resolve()
        candidate = (resolved_root / normalized).resolve()
        candidate.relative_to(resolved_root)
    except (OSError, ValueError):
        return None
    return candidate


def _is_u32_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 0xFFFFFFFF


def _normalize_interstage_params(raw_params: Any) -> list[dict[str, int]] | None:
    """Validate and normalize FSSM `vs_outputs` / `ps_inputs` arrays (#2085)."""
    if not isinstance(raw_params, list):
        return None
    normalized: list[dict[str, int]] = []
    for item in raw_params:
        if not isinstance(item, dict):
            return None
        semantic_reg = item.get("semantic_reg")
        location = item.get("location")
        if not (_is_u32_int(semantic_reg) and semantic_reg <= 0xFF and _is_u32_int(location)):
            return None
        rec: dict[str, int] = {
            "semantic_reg": semantic_reg,
            "location": location,
        }
        if "mask" in item and item["mask"] is not None:
            mask = item["mask"]
            if not (_is_u32_int(mask) and mask <= 0x0F):
                return None
            rec["mask"] = mask
        normalized.append(rec)
    return normalized


def extract_interstage_sidecar_bytes(reflect_path: Path) -> bytes:
    """Extract a compact `playgta6-stage-reflection` v3 interstage JSON from a `.reflect.json` file.

    Retains only stages with non-empty `vs_outputs` (vertex) or `ps_inputs` (fragment) so the
    browser GPU worker can fetch ~20-40 KiB instead of parsing an 80+ MiB full sidecar.
    """
    with reflect_path.open(encoding="utf-8-sig") as f:
        doc = json.load(f)
    if not isinstance(doc, dict) or doc.get("schema") != STAGE_REFLECTION_SCHEMA:
        raise ValueError("invalid stage reflection sidecar schema")
    stages = doc.get("stages")
    if not isinstance(stages, dict):
        raise ValueError("stage reflection sidecar 'stages' must be an object")

    compact_stages: dict[str, Any] = {}
    for key, stage_obj in stages.items():
        if not isinstance(key, str) or not key or not isinstance(stage_obj, dict):
            continue
        raw_stage = stage_obj.get("stage")
        stage = {"vertex": "vertex", "fragment": "fragment", "pixel": "fragment"}.get(
            str(raw_stage)
        )
        if stage == "vertex":
            raw_vs = stage_obj.get("vs_outputs")
            if raw_vs is None:
                continue
            vs_outputs = _normalize_interstage_params(raw_vs)
            if vs_outputs:
                compact_stages[key] = {
                    "stage": "vertex",
                    "bindings": [],
                    "vs_outputs": vs_outputs,
                }
        elif stage == "fragment":
            raw_ps = stage_obj.get("ps_inputs")
            if raw_ps is None:
                continue
            ps_inputs = _normalize_interstage_params(raw_ps)
            if ps_inputs:
                compact_stages[key] = {
                    "stage": "fragment",
                    "bindings": [],
                    "ps_inputs": ps_inputs,
                }

    payload = {
        "schema": STAGE_REFLECTION_SCHEMA,
        "version": STAGE_REFLECTION_INTERSTAGE_VERSION,
        "stages": compact_stages,
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


class _InterstageSidecarCache:
    """Thread-safe bounded LRU cache for dynamically extracted `.interstage.json` UTF-8 bytes.

    Keyed by `(realpath, mtime_ns, size)` so edits to `<name>.reflect.json` automatically
    invalidate cached bytes, with per-key coalescing to avoid redundant parses of 80+ MiB files.
    """

    def __init__(self, max_entries: int = DEFAULT_INTERSTAGE_CACHE_SIZE) -> None:
        self._max_entries = max(0, max_entries)
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple[str, int, int], bytes] = OrderedDict()
        self._inflight: dict[tuple[str, int, int], threading.Event] = {}
        self._hits = 0
        self._misses = 0

    def get_or_build(
        self,
        reflect_path: Path,
        st: os.stat_result,
        *,
        realpath: str | None = None,
    ) -> bytes:
        resolved_key_path = realpath if realpath is not None else os.path.realpath(reflect_path)
        cache_key = (resolved_key_path, st.st_mtime_ns, st.st_size)
        if self._max_entries <= 0:
            return extract_interstage_sidecar_bytes(reflect_path)

        while True:
            with self._lock:
                cached = self._entries.get(cache_key)
                if cached is not None:
                    self._entries.move_to_end(cache_key)
                    self._hits += 1
                    return cached
                inflight_ev = self._inflight.get(cache_key)
                if inflight_ev is None:
                    inflight_ev = threading.Event()
                    self._inflight[cache_key] = inflight_ev
                    is_builder = True
                    break
                is_builder = False
            if not is_builder:
                inflight_ev.wait()

        try:
            data = extract_interstage_sidecar_bytes(reflect_path)
        except Exception:
            with self._lock:
                self._inflight.pop(cache_key, None)
                inflight_ev.set()
            raise

        with self._lock:
            self._misses += 1
            self._entries[cache_key] = data
            self._entries.move_to_end(cache_key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
            self._inflight.pop(cache_key, None)
            inflight_ev.set()
        return data

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "hits": self._hits,
                "misses": self._misses,
            }


_DEFAULT_INTERSTAGE_SIDECAR_CACHE = _InterstageSidecarCache()


class _ResolvedPathCache:
    """Thread-safe bounded LRU cache for verified non-symlink in-root file paths.

    Eliminates redundant per-request `Path.resolve()` (`nt._getfinalpathname`) and
    `mimetypes.guess_type()` calls on Windows while remaining zero-staleness: callers
    still `os.lstat()` the cached path on every request and immediately evict on
    missing/non-regular/symlink changes.
    """

    __slots__ = ("_max_entries", "_lock", "_entries")

    def __init__(self, max_entries: int = DEFAULT_RESOLVED_PATH_CACHE_SIZE) -> None:
        self._max_entries = max(0, max_entries)
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple[str, str], tuple[Path, str, str]] = OrderedDict()

    def get(self, key: tuple[str, str]) -> tuple[Path, str, str] | None:
        if self._max_entries <= 0:
            return None
        with self._lock:
            val = self._entries.get(key)
            if val is not None:
                self._entries.move_to_end(key)
            return val

    def put(self, key: tuple[str, str], target: Path, realpath: str, content_type: str) -> None:
        if self._max_entries <= 0:
            return
        with self._lock:
            self._entries[key] = (target, realpath, content_type)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def invalidate(self, key: tuple[str, str]) -> None:
        if self._max_entries <= 0:
            return
        with self._lock:
            self._entries.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def resolve_dump_target(
    dump_root: Path, rel_url_path: str
) -> tuple[tuple[Path, str] | None, HTTPStatus | None, str | None]:
    """Validate `<relpath>` for `POST /__pg6_dump/<relpath>` strictly inside `dump_root`.

    Rejects empty paths, absolute paths, drive letters/colons, backslashes, `.`/`..`
    traversal segments, NUL/control characters, directory targets, and symlinks
    escaping the resolved `dump_root`.

    Returns:
        `((target_path, sanitized_relpath), None, None)` on success, or
        `(None, HTTPStatus, static_error_message)` on rejection (zero host paths).
    """
    if not rel_url_path:
        return None, HTTPStatus.BAD_REQUEST, "empty dump path"

    decoded = unquote(rel_url_path)
    if not decoded:
        return None, HTTPStatus.BAD_REQUEST, "empty dump path"
    if "\x00" in rel_url_path or any(ord(ch) < 32 or ord(ch) == 127 for ch in decoded):
        return None, HTTPStatus.BAD_REQUEST, "invalid control character in dump path"
    if "\\" in decoded:
        return None, HTTPStatus.BAD_REQUEST, "backslashes are not permitted in dump path"
    if decoded.startswith("/") or posixpath.isabs(decoded) or ntpath.isabs(decoded):
        return None, HTTPStatus.BAD_REQUEST, "absolute paths are not permitted"
    if ntpath.splitdrive(decoded)[0] or ":" in decoded:
        return None, HTTPStatus.BAD_REQUEST, "drive letters and colons are not permitted"

    segments = decoded.split("/")
    if not segments or any(seg in ("", ".", "..") for seg in segments):
        return None, HTTPStatus.BAD_REQUEST, "invalid dump path segment"

    try:
        root_resolved = dump_root.resolve()
    except (OSError, RuntimeError):
        return None, HTTPStatus.FORBIDDEN, "invalid dump directory"
    if root_resolved.exists() and not root_resolved.is_dir():
        return None, HTTPStatus.FORBIDDEN, "invalid dump directory"

    current = root_resolved
    for idx, seg in enumerate(segments):
        current = current / seg
        try:
            if current.is_symlink():
                resolved_link = current.resolve()
                if not resolved_link.is_relative_to(root_resolved):
                    return None, HTTPStatus.FORBIDDEN, "symlink escapes dump directory"
                if idx == len(segments) - 1 and resolved_link == root_resolved:
                    return None, HTTPStatus.FORBIDDEN, "symlink escapes dump directory"
            if idx < len(segments) - 1 and current.exists() and not current.is_dir():
                return None, HTTPStatus.BAD_REQUEST, "parent path is not a directory"
        except (OSError, ValueError, RuntimeError):
            return None, HTTPStatus.FORBIDDEN, "invalid dump target"

    try:
        candidate_resolved = current.resolve()
        if not candidate_resolved.is_relative_to(root_resolved) or candidate_resolved == root_resolved:
            return None, HTTPStatus.FORBIDDEN, "path escapes dump directory"
        if current.exists() and current.is_dir():
            return None, HTTPStatus.BAD_REQUEST, "target path is a directory"
    except (OSError, ValueError, RuntimeError):
        return None, HTTPStatus.FORBIDDEN, "invalid dump target"

    sanitized_relpath = "/".join(segments)
    return (current, sanitized_relpath), None, None


def parse_byte_range_status(header: str, file_size: int) -> tuple[str, tuple[int, int] | None]:
    """Classify an HTTP `Range` header per RFC 9110 §14.1.2.

    Returns:
        - `("ok", (start, end))` for a valid satisfiable single `bytes=` range.
        - `("unsatisfiable", None)` for a single `bytes=` range that is malformed or
          cannot be satisfied against `file_size` (server responds `416`).
        - `("ignore", None)` for unsupported range units (e.g. `items=0-5`) or
          multi-range specs (`bytes=0-10, 20-30`) (server ignores `Range` and
          responds `200 OK`).
    """
    stripped = header.strip()
    if not stripped:
        return "ignore", None
    if not stripped.lower().startswith("bytes=") or "," in stripped:
        return "ignore", None
    match = RANGE_RE.match(stripped)
    if not match:
        return "unsatisfiable", None
    start_s, end_s = match.groups()
    if not start_s and not end_s:
        return "unsatisfiable", None
    if file_size <= 0:
        return "unsatisfiable", None
    if start_s and end_s:
        start, end = int(start_s), min(int(end_s), file_size - 1)
    elif start_s:
        start, end = int(start_s), file_size - 1
    else:
        suffix = int(end_s)
        if suffix <= 0:
            return "unsatisfiable", None
        start, end = max(file_size - suffix, 0), file_size - 1
    if start > end or start >= file_size:
        return "unsatisfiable", None
    return "ok", (start, end)


def parse_byte_range(header: str, file_size: int) -> tuple[int, int] | None:
    """Parse a single HTTP byte range against `file_size`."""
    status, bounds = parse_byte_range_status(header, file_size)
    return bounds if status == "ok" else None


def compute_etag(stat_result: os.stat_result) -> str:
    """Return a deterministic strong ETag derived from file size and mtime_ns."""
    return f'"{stat_result.st_size:x}-{stat_result.st_mtime_ns:x}"'


def _strip_weak_prefix(tag: str) -> str:
    cleaned = tag.strip()
    if cleaned.startswith(("W/", "w/")):
        return cleaned[2:].strip()
    return cleaned


def etag_matches_if_none_match(header_value: str, etag: str) -> bool:
    """Evaluate `If-None-Match` using RFC 9110 weak comparison."""
    target = _strip_weak_prefix(etag)
    for raw_token in header_value.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if token == "*":
            return True
        if _strip_weak_prefix(token) == target:
            return True
    return False


def etag_matches_if_match(header_value: str, etag: str) -> bool:
    """Evaluate `If-Match` using RFC 9110 Section 13.1.1 strong comparison."""
    if not etag or etag.startswith(("W/", "w/")):
        return False
    for raw_token in header_value.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if token == "*":
            return True
        if token.startswith(("W/", "w/")):
            continue
        if token == etag:
            return True
    return False


def if_range_matches(header_value: str, etag: str, last_modified: str, mtime: float) -> bool:
    """Evaluate RFC 9110 Section 13.1.5 `If-Range` condition (strong ETag or HTTP-date)."""
    value = header_value.strip()
    if not value:
        return False
    if value.startswith('"') or value.startswith(("W/", "w/")):
        # Strong comparison only: weak entity-tags never match If-Range.
        if value.startswith(("W/", "w/")) or etag.startswith(("W/", "w/")):
            return False
        return value == etag
    if value == last_modified:
        return True
    try:
        parsed_dt = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return False
    if parsed_dt is None:
        return False
    if parsed_dt.tzinfo is None:
        parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
    return int(parsed_dt.timestamp()) == int(mtime)


def if_modified_since_not_modified(header_value: str, mtime: float) -> bool:
    """Evaluate RFC 9110 Section 13.1.3 `If-Modified-Since` (1-second HTTP-date granularity)."""
    value = header_value.strip()
    if not value:
        return False
    try:
        parsed_dt = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return False
    if parsed_dt is None:
        return False
    if parsed_dt.tzinfo is None:
        parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
    return int(mtime) <= int(parsed_dt.timestamp())


def _get_thread_stream_buffer(chunk_bytes: int) -> bytearray:
    """Return a reusable per-thread `bytearray` of at least `chunk_bytes` for zero-alloc streaming."""
    buf: bytearray | None = getattr(_STREAM_BUFFERS, "buf", None)
    if buf is None or len(buf) < chunk_bytes:
        buf = bytearray(chunk_bytes)
        _STREAM_BUFFERS.buf = buf
    return buf


_HAS_PREAD = hasattr(os, "pread")
_HAS_PREADV = hasattr(os, "preadv")
_ORIG_OS_READ = os.read
_MTIME_HTTP_DATE_LOCK = threading.Lock()
_MTIME_HTTP_DATE_CACHE: OrderedDict[int, str] = OrderedDict()
_MTIME_HTTP_DATE_CACHE_MAX = 256


def _format_mtime_http_date(mtime: float) -> str:
    """Return RFC 9110 Last-Modified HTTP-date cached by integer mtime second."""
    sec = int(mtime)
    with _MTIME_HTTP_DATE_LOCK:
        cached = _MTIME_HTTP_DATE_CACHE.get(sec)
        if cached is not None:
            _MTIME_HTTP_DATE_CACHE.move_to_end(sec)
            return cached
    formatted = formatdate(sec, usegmt=True)
    with _MTIME_HTTP_DATE_LOCK:
        _MTIME_HTTP_DATE_CACHE[sec] = formatted
        _MTIME_HTTP_DATE_CACHE.move_to_end(sec)
        while len(_MTIME_HTTP_DATE_CACHE) > _MTIME_HTTP_DATE_CACHE_MAX:
            _MTIME_HTTP_DATE_CACHE.popitem(last=False)
    return formatted


def _make_unowned_fileio(fd: int) -> io.FileIO | None:
    try:
        return io.FileIO(fd, "rb", closefd=False)
    except OSError:
        return None


def _close_fileio_safely(fio: io.FileIO | None) -> None:
    if fio is None:
        return
    try:
        fio.close()
    except OSError:
        pass
_WIN32_KERNEL32: Any | None = None
_WIN32_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
_WIN32_GENERIC_READ = 0x80000000
_WIN32_FILE_SHARE_ALL = 0x01 | 0x02 | 0x04  # READ | WRITE | DELETE
_WIN32_OPEN_EXISTING = 3
_WIN32_FILE_ATTRIBUTE_NORMAL = 0x80
_WIN32_FILE_FLAG_RANDOM_ACCESS = 0x10000000


def _get_win32_kernel32() -> Any | None:
    global _WIN32_KERNEL32
    if os.name != "nt":
        return None
    if _WIN32_KERNEL32 is None:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateFileW.restype = ctypes.c_void_p
        k32.CreateFileW.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        k32.CloseHandle.restype = ctypes.c_int
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        _WIN32_KERNEL32 = k32
    return _WIN32_KERNEL32


def is_cacheable_file_path(path: Path) -> bool:
    """Return True for large immutable asset suffixes eligible for FD caching."""
    return path.name.lower().endswith(CACHEABLE_IMMUTABLE_SUFFIXES)


def _open_readonly_fd(path: Path) -> int:
    """Open `path` read-only with delete/write sharing on Windows for safe FD caching."""
    path_str = str(path)
    if os.name == "nt":
        k32 = _get_win32_kernel32()
        if k32 is not None:
            handle = k32.CreateFileW(
                path_str,
                _WIN32_GENERIC_READ,
                _WIN32_FILE_SHARE_ALL,
                None,
                _WIN32_OPEN_EXISTING,
                _WIN32_FILE_ATTRIBUTE_NORMAL | _WIN32_FILE_FLAG_RANDOM_ACCESS,
                None,
            )
            if handle is not None and handle != _WIN32_INVALID_HANDLE_VALUE:
                try:
                    return msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
                except OSError:
                    k32.CloseHandle(handle)
    return os.open(path_str, os.O_RDONLY | getattr(os, "O_BINARY", 0))


def _close_fd_safely(fd: int | None) -> None:
    if fd is None or fd < 0:
        return
    try:
        os.close(fd)
    except OSError:
        pass


class _BorrowedFileReader:
    """Per-transfer reader over a borrowed read-only file descriptor."""

    __slots__ = ("_fd", "_io_lock", "_use_pread", "_cursor", "_fileio")

    def __init__(
        self,
        fd: int,
        *,
        io_lock: threading.Lock | None,
        use_pread: bool,
        fileio: io.FileIO | None = None,
    ) -> None:
        self._fd = fd
        self._io_lock = io_lock
        self._use_pread = use_pread
        self._cursor: int | None = None
        self._fileio = fileio

    @property
    def fd(self) -> int:
        return self._fd

    def can_readinto(self) -> bool:
        if self._use_pread:
            return _HAS_PREADV
        return self._fileio is not None

    def readinto_at(self, offset: int, mv: memoryview) -> int:
        if self._use_pread and _HAS_PREADV:
            return int(os.preadv(self._fd, (mv,), offset))
        fio = self._fileio
        if fio is None:
            chunk = self.read_at(offset, len(mv))
            n = len(chunk)
            if n:
                mv[:n] = chunk
            return n
        if self._io_lock is not None:
            with self._io_lock:
                os.lseek(self._fd, offset, os.SEEK_SET)
                n = fio.readinto(mv)
                return int(n or 0)
        if self._cursor != offset:
            os.lseek(self._fd, offset, os.SEEK_SET)
        n = fio.readinto(mv)
        n_int = int(n or 0)
        self._cursor = offset + n_int
        return n_int

    def read_at(self, offset: int, size: int) -> bytes:
        if self._use_pread:
            return os.pread(self._fd, size, offset)
        if self._io_lock is not None:
            with self._io_lock:
                os.lseek(self._fd, offset, os.SEEK_SET)
                return os.read(self._fd, size)
        if self._cursor != offset:
            os.lseek(self._fd, offset, os.SEEK_SET)
        chunk = os.read(self._fd, size)
        self._cursor = offset + len(chunk)
        return chunk


_ORIG_BORROWED_READ_AT = _BorrowedFileReader.read_at


class _CachedFileHandle:
    """Refcounted read-only file descriptor entry guarded by `(realpath, size, mtime_ns)`.

    On platforms with `os.pread`, all concurrent readers share the single primary FD
    lock-free. On non-`pread` platforms (notably Windows), a bounded lazy per-entry pool
    grants each concurrent reader its own dedicated read-only FD (up to
    `max_concurrent_fds`) so 8-lane HTTPFS Range transfers do not serialize on `io_lock`.
    """

    __slots__ = (
        "realpath",
        "size",
        "mtime_ns",
        "key",
        "fd",
        "io_lock",
        "refcount",
        "retired",
        "max_concurrent_fds",
        "_pool_lock",
        "_primary_users",
        "_idle_pool_fds",
        "_active_pool_fds",
        "_pending_pool_opens",
        "_fd_fileios",
        "_on_before_pool_open",
        "_on_pool_fd_opened",
        "_on_pool_open_aborted",
        "_on_pool_fd_closed",
    )

    def __init__(
        self,
        realpath: str,
        size: int,
        mtime_ns: int,
        fd: int,
        *,
        max_concurrent_fds: int = DEFAULT_CONCURRENT_FD_POOL_SIZE,
        on_before_pool_open: Callable[[], bool] | None = None,
        on_pool_fd_opened: Callable[[], None] | None = None,
        on_pool_open_aborted: Callable[[], None] | None = None,
        on_pool_fd_closed: Callable[[int], None] | None = None,
    ) -> None:
        self.realpath = realpath
        self.size = size
        self.mtime_ns = mtime_ns
        self.key = (realpath, size, mtime_ns)
        self.fd = fd
        self.io_lock = threading.Lock()
        self.refcount = 1
        self.retired = False
        self.max_concurrent_fds = max(1, max_concurrent_fds)
        self._pool_lock = threading.Lock()
        self._primary_users = 0
        self._idle_pool_fds: list[int] = []
        self._active_pool_fds = 0
        self._pending_pool_opens = 0
        self._fd_fileios: dict[int, io.FileIO] = {}
        primary_fio = _make_unowned_fileio(fd)
        if primary_fio is not None:
            self._fd_fileios[fd] = primary_fio
        self._on_before_pool_open = on_before_pool_open
        self._on_pool_fd_opened = on_pool_fd_opened
        self._on_pool_open_aborted = on_pool_open_aborted
        self._on_pool_fd_closed = on_pool_fd_closed

    def _register_fileio_locked(self, fd: int) -> io.FileIO | None:
        fio = self._fd_fileios.get(fd)
        if fio is None and fd >= 0:
            fio = _make_unowned_fileio(fd)
            if fio is not None:
                self._fd_fileios[fd] = fio
        return fio

    def _pop_fileio_locked(self, fd: int) -> io.FileIO | None:
        return self._fd_fileios.pop(fd, None)

    def try_donate_pooled_fd(self, fd: int) -> bool:
        """Accept an already-opened, validated FD into the idle pool during cold-miss races."""
        with self._pool_lock:
            if self.retired or self.fd < 0:
                return False
            total_fds = (
                1
                + len(self._idle_pool_fds)
                + self._active_pool_fds
                + self._pending_pool_opens
            )
            if total_fds >= self.max_concurrent_fds:
                return False
            self._register_fileio_locked(fd)
            self._idle_pool_fds.append(fd)
            return True

    def evict_one_idle_pool_fd(self) -> int | None:
        """Pop one idle pooled FD for global FD budget reclamation, or return None."""
        fio: io.FileIO | None = None
        fd: int | None = None
        with self._pool_lock:
            if not self._idle_pool_fds:
                return None
            fd = self._idle_pool_fds.pop(0)
            fio = self._pop_fileio_locked(fd)
        _close_fileio_safely(fio)
        return fd

    def mark_retired(self) -> list[int]:
        """Mark entry retired and detach any idle pooled FDs for immediate closure."""
        fios: list[io.FileIO] = []
        with self._pool_lock:
            self.retired = True
            if not self._idle_pool_fds:
                return []
            idle = list(self._idle_pool_fds)
            self._idle_pool_fds.clear()
            for idle_fd in idle:
                fio = self._pop_fileio_locked(idle_fd)
                if fio is not None:
                    fios.append(fio)
        for fio in fios:
            _close_fileio_safely(fio)
        return idle

    def detach_fd(self) -> int | None:
        fio: io.FileIO | None = None
        with self._pool_lock:
            fd = self.fd
            if fd < 0:
                return None
            self.fd = -1
            fio = self._pop_fileio_locked(fd)
        _close_fileio_safely(fio)
        return fd

    def detach_all_fds(self) -> tuple[int | None, list[int]]:
        """Detach primary FD and any remaining idle pooled FDs."""
        fios: list[io.FileIO] = []
        with self._pool_lock:
            primary_fd: int | None = None
            if self.fd >= 0:
                primary_fd = self.fd
                self.fd = -1
                fio = self._pop_fileio_locked(primary_fd)
                if fio is not None:
                    fios.append(fio)
            if self._idle_pool_fds:
                idle = list(self._idle_pool_fds)
                self._idle_pool_fds.clear()
                for idle_fd in idle:
                    fio = self._pop_fileio_locked(idle_fd)
                    if fio is not None:
                        fios.append(fio)
            else:
                idle = []
        for fio in fios:
            _close_fileio_safely(fio)
        return primary_fd, idle

    @contextlib.contextmanager
    def borrow_fd(self) -> Iterator[_BorrowedFileReader]:
        """Borrow a dedicated reader for an entire Range transfer.

        - When `_HAS_PREAD` is True, uses lock-free `os.pread(self.fd, ...)` with zero
          pool expansion.
        - When `_HAS_PREAD` is False, uses the primary FD when uncontended (sequential
          reads open only 1 FD), pops an idle pooled FD, or lazily opens a new read-only
          FD up to `max_concurrent_fds`. If all pooled FDs are busy, cleanly falls back
          to the primary FD guarded by `self.io_lock`.
        """
        if _HAS_PREAD:
            fd = self.fd
            if fd < 0:
                raise OSError(errno.EBADF, "cached file handle is closed")
            yield _BorrowedFileReader(fd, io_lock=None, use_pread=True)
            return

        is_pooled = False
        needs_open = False
        fd = -1
        fio: io.FileIO | None = None
        with self._pool_lock:
            if self.fd < 0:
                raise OSError(errno.EBADF, "cached file handle is closed")
            if self._primary_users == 0:
                self._primary_users = 1
                fd = self.fd
                fio = self._register_fileio_locked(fd)
            elif self._idle_pool_fds:
                fd = self._idle_pool_fds.pop()
                self._active_pool_fds += 1
                is_pooled = True
                fio = self._register_fileio_locked(fd)
            elif (
                not self.retired
                and (
                    1
                    + len(self._idle_pool_fds)
                    + self._active_pool_fds
                    + self._pending_pool_opens
                )
                < self.max_concurrent_fds
            ):
                self._pending_pool_opens += 1
                needs_open = True
            else:
                self._primary_users += 1
                fd = self.fd
                fio = self._register_fileio_locked(fd)

        if needs_open:
            new_fd: int | None = None
            opened_ok = False
            budget_ok = True
            if self._on_before_pool_open is not None:
                budget_ok = bool(self._on_before_pool_open())
            if budget_ok:
                try:
                    candidate_fd = _open_readonly_fd(Path(self.realpath))
                    try:
                        fstat = os.fstat(candidate_fd)
                        if (
                            stat.S_ISREG(fstat.st_mode)
                            and fstat.st_size == self.size
                            and fstat.st_mtime_ns == self.mtime_ns
                        ):
                            new_fd = candidate_fd
                            opened_ok = True
                        else:
                            _close_fd_safely(candidate_fd)
                    except OSError:
                        _close_fd_safely(candidate_fd)
                except OSError:
                    opened_ok = False

            discard_fd: int | None = None
            notify_opened = False
            with self._pool_lock:
                self._pending_pool_opens -= 1
                if opened_ok and new_fd is not None and not self.retired:
                    self._active_pool_fds += 1
                    fd = new_fd
                    fio = self._register_fileio_locked(fd)
                    is_pooled = True
                    notify_opened = True
                else:
                    if new_fd is not None:
                        discard_fd = new_fd
                    self._primary_users += 1
                    fd = self.fd
                    fio = self._register_fileio_locked(fd)

            if discard_fd is not None:
                _close_fd_safely(discard_fd)
            if notify_opened:
                if self._on_pool_fd_opened is not None:
                    self._on_pool_fd_opened()
            elif budget_ok and self._on_pool_open_aborted is not None:
                self._on_pool_open_aborted()

        try:
            yield _BorrowedFileReader(
                fd,
                io_lock=None if is_pooled else self.io_lock,
                use_pread=False,
                fileio=fio,
            )
        finally:
            fd_to_close: int | None = None
            fio_to_close: io.FileIO | None = None
            with self._pool_lock:
                if is_pooled:
                    self._active_pool_fds -= 1
                    if self.retired:
                        fd_to_close = fd
                        fio_to_close = self._pop_fileio_locked(fd)
                    else:
                        self._idle_pool_fds.append(fd)
                else:
                    self._primary_users -= 1
            if fio_to_close is not None:
                _close_fileio_safely(fio_to_close)
            if fd_to_close is not None:
                _close_fd_safely(fd_to_close)
                if self._on_pool_fd_closed is not None:
                    self._on_pool_fd_closed(1)

    def read_at(self, offset: int, size: int) -> bytes:
        if _HAS_PREAD:
            return os.pread(self.fd, size, offset)
        with self.borrow_fd() as reader:
            return reader.read_at(offset, size)


class ReadOnlyFileHandleCache:
    """Thread-safe bounded LRU cache of read-only file descriptors for large Range assets.

    Entries are validated against `(realpath, st_size, st_mtime_ns)` on every acquire;
    stale entries are immediately retired and closed once in-flight reads release them.
    """

    def __init__(
        self,
        max_entries: int = DEFAULT_FILE_HANDLE_CACHE_SIZE,
        *,
        max_concurrent_fds_per_file: int = DEFAULT_CONCURRENT_FD_POOL_SIZE,
        max_total_open_fds: int | None = DEFAULT_MAX_TOTAL_OPEN_FDS,
    ) -> None:
        if max_entries < 0:
            raise ValueError("max_entries must be >= 0")
        if max_concurrent_fds_per_file < 1:
            raise ValueError("max_concurrent_fds_per_file must be >= 1")
        if max_total_open_fds is not None and max_total_open_fds < 1:
            raise ValueError("max_total_open_fds must be >= 1")
        self._max_entries = max_entries
        self._max_concurrent_fds_per_file = max_concurrent_fds_per_file
        self._max_total_open_fds = max_total_open_fds
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, _CachedFileHandle] = OrderedDict()
        self._closed = False
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._invalidations = 0
        self._open_handles = 0
        self._pooled_handles = 0
        self._pool_opens = 0
        self._reserved_pool_opens = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def _trim_idle_pooled_fds_locked(self, target_max_open: int, fds_to_close: list[int]) -> None:
        """Reclaim idle pooled FDs from least-recently-used entries under `self._lock`."""
        if self._open_handles <= target_max_open:
            return
        for entry in self._entries.values():
            while self._open_handles > target_max_open:
                evicted_fd = entry.evict_one_idle_pool_fd()
                if evicted_fd is None:
                    break
                self._open_handles -= 1
                self._pooled_handles -= 1
                fds_to_close.append(evicted_fd)
            if self._open_handles <= target_max_open:
                break

    def _reserve_global_pool_fd_budget(self) -> bool:
        """Atomically reserve one pooled FD slot within `max_total_open_fds`."""
        limit = self._max_total_open_fds
        if limit is None:
            return True
        fds_to_close: list[int] = []
        allowed = False
        with self._lock:
            if self._closed:
                return False
            if self._open_handles + self._reserved_pool_opens >= limit:
                self._trim_idle_pooled_fds_locked(
                    max(0, limit - 1 - self._reserved_pool_opens), fds_to_close
                )
            allowed = (self._open_handles + self._reserved_pool_opens) < limit
            if allowed:
                self._reserved_pool_opens += 1
        for fd in fds_to_close:
            _close_fd_safely(fd)
        return allowed

    def _record_pool_fd_opened(self) -> None:
        with self._lock:
            if self._reserved_pool_opens > 0:
                self._reserved_pool_opens -= 1
            self._open_handles += 1
            self._pooled_handles += 1
            self._pool_opens += 1

    def _release_global_pool_fd_reservation(self) -> None:
        with self._lock:
            if self._reserved_pool_opens > 0:
                self._reserved_pool_opens -= 1

    def _record_pool_fd_closed(self, count: int) -> None:
        with self._lock:
            self._open_handles -= count
            self._pooled_handles -= count

    def _retire_entry_locked(
        self, entry: _CachedFileHandle, fds_to_close: list[int]
    ) -> None:
        """Retire `entry` under `self._lock`, collecting any FDs ready to close."""
        idle_fds = entry.mark_retired()
        if idle_fds:
            self._open_handles -= len(idle_fds)
            self._pooled_handles -= len(idle_fds)
            fds_to_close.extend(idle_fds)
        if entry.refcount <= 0:
            primary_fd, leftover_idle = entry.detach_all_fds()
            if primary_fd is not None:
                self._open_handles -= 1
                fds_to_close.append(primary_fd)
            if leftover_idle:
                self._open_handles -= len(leftover_idle)
                self._pooled_handles -= len(leftover_idle)
                fds_to_close.extend(leftover_idle)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "max_entries": self._max_entries,
                "max_concurrent_fds_per_file": self._max_concurrent_fds_per_file,
                "max_total_open_fds": (
                    self._max_total_open_fds if self._max_total_open_fds is not None else 0
                ),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "invalidations": self._invalidations,
                "open_handles": self._open_handles,
                "pooled_handles": self._pooled_handles,
                "pool_opens": self._pool_opens,
            }

    def acquire(
        self,
        path: Path,
        st: os.stat_result,
        *,
        realpath: str | None = None,
    ) -> _CachedFileHandle | None:
        """Borrow a refcounted handle matching `(realpath, st.st_size, st.st_mtime_ns)`."""
        if self._max_entries <= 0:
            return None
        if realpath is None:
            try:
                realpath = os.path.realpath(path)
            except OSError:
                return None
        expected_key = (realpath, st.st_size, st.st_mtime_ns)
        stale_fds: list[int] = []

        with self._lock:
            if self._closed:
                return None
            existing = self._entries.get(realpath)
            if existing is not None:
                if existing.key == expected_key:
                    self._entries.move_to_end(realpath)
                    existing.refcount += 1
                    self._hits += 1
                    return existing
                self._entries.pop(realpath, None)
                self._invalidations += 1
                self._retire_entry_locked(existing, stale_fds)

        for stale_fd in stale_fds:
            _close_fd_safely(stale_fd)

        try:
            fd = _open_readonly_fd(path)
        except OSError:
            return None

        try:
            fstat = os.fstat(fd)
        except OSError:
            _close_fd_safely(fd)
            return None

        if (
            not stat.S_ISREG(fstat.st_mode)
            or fstat.st_size != st.st_size
            or fstat.st_mtime_ns != st.st_mtime_ns
        ):
            _close_fd_safely(fd)
            return None

        fds_to_close: list[int] = []
        with self._lock:
            if self._closed:
                fds_to_close.append(fd)
                entry: _CachedFileHandle | None = None
            else:
                existing = self._entries.get(realpath)
                if existing is not None and existing.key == expected_key:
                    self._entries.move_to_end(realpath)
                    existing.refcount += 1
                    self._hits += 1
                    donated = False
                    if not _HAS_PREAD:
                        limit = self._max_total_open_fds
                        if (
                            limit is not None
                            and self._open_handles + self._reserved_pool_opens >= limit
                        ):
                            self._trim_idle_pooled_fds_locked(
                                max(0, limit - 1 - self._reserved_pool_opens), fds_to_close
                            )
                        if (
                            limit is None
                            or (self._open_handles + self._reserved_pool_opens) < limit
                        ):
                            donated = existing.try_donate_pooled_fd(fd)
                            if donated:
                                self._open_handles += 1
                                self._pooled_handles += 1
                                self._pool_opens += 1
                    if not donated:
                        fds_to_close.append(fd)
                    entry = existing
                else:
                    if existing is not None:
                        self._entries.pop(realpath, None)
                        self._invalidations += 1
                        self._retire_entry_locked(existing, fds_to_close)
                    if (
                        self._max_total_open_fds is not None
                        and self._open_handles + self._reserved_pool_opens
                        >= self._max_total_open_fds
                    ):
                        self._trim_idle_pooled_fds_locked(
                            max(0, self._max_total_open_fds - 1 - self._reserved_pool_opens),
                            fds_to_close,
                        )
                    entry = _CachedFileHandle(
                        realpath,
                        st.st_size,
                        st.st_mtime_ns,
                        fd,
                        max_concurrent_fds=self._max_concurrent_fds_per_file,
                        on_before_pool_open=self._reserve_global_pool_fd_budget,
                        on_pool_fd_opened=self._record_pool_fd_opened,
                        on_pool_open_aborted=self._release_global_pool_fd_reservation,
                        on_pool_fd_closed=self._record_pool_fd_closed,
                    )
                    self._entries[realpath] = entry
                    self._open_handles += 1
                    self._misses += 1
                    while len(self._entries) > self._max_entries:
                        _, evicted = self._entries.popitem(last=False)
                        self._evictions += 1
                        self._retire_entry_locked(evicted, fds_to_close)
                    while (
                        self._max_total_open_fds is not None
                        and self._open_handles > self._max_total_open_fds
                        and len(self._entries) > 1
                    ):
                        self._trim_idle_pooled_fds_locked(
                            self._max_total_open_fds, fds_to_close
                        )
                        if self._open_handles <= self._max_total_open_fds:
                            break
                        _, evicted = self._entries.popitem(last=False)
                        self._evictions += 1
                        self._retire_entry_locked(evicted, fds_to_close)

        for dead_fd in fds_to_close:
            _close_fd_safely(dead_fd)
        return entry

    def release(self, entry: _CachedFileHandle) -> None:
        """Release a previously acquired handle; closes all remaining FDs if retired."""
        fds_to_close: list[int] = []
        with self._lock:
            entry.refcount -= 1
            if entry.retired and entry.refcount <= 0:
                primary_fd, idle_fds = entry.detach_all_fds()
                if primary_fd is not None:
                    self._open_handles -= 1
                    fds_to_close.append(primary_fd)
                if idle_fds:
                    self._open_handles -= len(idle_fds)
                    self._pooled_handles -= len(idle_fds)
                    fds_to_close.extend(idle_fds)
        for fd in fds_to_close:
            _close_fd_safely(fd)

    def invalidate(self, path: Path, *, realpath: str | None = None) -> None:
        """Evict and retire any cached handle for `path`."""
        if realpath is None:
            try:
                realpath = os.path.realpath(path)
            except OSError:
                return
        fds_to_close: list[int] = []
        with self._lock:
            existing = self._entries.pop(realpath, None)
            if existing is not None:
                self._invalidations += 1
                self._retire_entry_locked(existing, fds_to_close)
        for fd in fds_to_close:
            _close_fd_safely(fd)

    def invalidate_entry(self, entry: _CachedFileHandle) -> None:
        """Retire a specific cached entry (e.g., on mid-stream truncation)."""
        fds_to_close: list[int] = []
        with self._lock:
            if self._entries.get(entry.realpath) is entry:
                self._entries.pop(entry.realpath, None)
                self._invalidations += 1
                self._retire_entry_locked(entry, fds_to_close)
        for fd in fds_to_close:
            _close_fd_safely(fd)

    def close_all(self) -> None:
        """Close all idle cached handles and prevent further caching after shutdown."""
        fds_to_close: list[int] = []
        with self._lock:
            self._closed = True
            while self._entries:
                _, entry = self._entries.popitem(last=False)
                self._retire_entry_locked(entry, fds_to_close)
        for fd in fds_to_close:
            _close_fd_safely(fd)


def _count_shader_packs(web_root: Path, data_root: Path, shaders_root: Path | None = None) -> int:
    """Count `.wgsl.pack` bundles across expected shallow directories."""
    candidate_dirs = [
        data_root,
        data_root / "shaders",
        web_root / "dist",
        web_root / "shaders",
    ]
    if shaders_root is not None:
        candidate_dirs.append(shaders_root)
    seen: set[Path] = set()
    count = 0
    for directory in candidate_dirs:
        if not directory.is_dir():
            continue
        try:
            resolved = directory.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            for entry in directory.iterdir():
                if entry.is_file() and entry.name.lower().endswith(".wgsl.pack"):
                    count += 1
        except OSError:
            continue
    return count


def _load_inspect_game_assets() -> Any | None:
    """Lazily load tools/inspect_game_assets.py for safe MicrosoftGame.Config parsing."""
    tools_path = Path(__file__).resolve().parent.parent / "tools" / "inspect_game_assets.py"
    if not tools_path.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_playgta6_inspect_game_assets", tools_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


def _discover_splash_image_url(game_root: Path | None) -> str | None:
    """Return `/game/<rel>` when ShellVisuals.SplashScreenImage exists inside game_root."""
    if game_root is None:
        return None
    try:
        if not game_root.is_dir():
            return None
        game_resolved = game_root.resolve()
    except OSError:
        return None

    iga = _load_inspect_game_assets()
    if iga is None:
        return None

    try:
        config_path = iga._locate_microsoft_game_config(game_root)
        if config_path is None:
            return None
        parsed = iga.parse_microsoft_game_config(config_path)
        declared = (parsed.get("shellVisuals") or {}).get("splashScreenImage")
        parts, err = iga._validate_package_relative_path(declared)
        if err is not None or parts is None:
            return None
        package_root = config_path.parent
        package_root_resolved = package_root.resolve()
        resolved_file = iga._resolve_case_insensitive_in_root(
            package_root, package_root_resolved, parts
        )
        if resolved_file is None or not resolved_file.is_file():
            return None
        resolved = resolved_file.resolve()
        if not resolved.is_relative_to(game_resolved):
            return None
        rel = resolved.relative_to(game_resolved).as_posix()
        if not rel or ".." in rel.split("/"):
            return None
        return f"/game/{rel}"
    except Exception:
        return None


def _list_shader_pack_names(shaders_root: Path | None) -> list[str]:
    if shaders_root is None or not shaders_root.is_dir():
        return []
    try:
        shaders_root_resolved = shaders_root.resolve()
    except OSError:
        return []
    names: list[str] = []
    try:
        for entry in shaders_root.iterdir():
            try:
                if not entry.name.lower().endswith(".wgsl.pack"):
                    continue
                if not entry.is_file():
                    continue
                # Ensure the file stays inside shaders_root (reject external symlinks).
                resolved = entry.resolve(strict=True)
                if not resolved.is_relative_to(shaders_root_resolved):
                    continue
                if not resolved.is_file() or not resolved.name.lower().endswith(".wgsl.pack"):
                    continue
                names.append(entry.name)
            except OSError:
                continue
    except OSError:
        return []
    return sorted(names, key=lambda name: (name.lower(), name))


def _discover_shader_pack_url(shaders_root: Path | None) -> str | None:
    """Return the preferred `/shaders/<name>` URL for local `.wgsl.pack` bundles.

    Preference order matches `runtime/wasm_boot.cpp`:
    1. `splash.wgsl.pack` when present
    2. `fallback.wgsl.pack` when present
    3. First sorted `*.wgsl.pack` file if any
    4. `None` when no `.wgsl.pack` files exist
    """
    names = _list_shader_pack_names(shaders_root)
    if not names:
        return None
    for preferred_name in PREFERRED_SHADER_PACK_NAMES:
        match = next((name for name in names if name.lower() == preferred_name), None)
        if match is not None:
            return f"/shaders/{match}"
    return f"/shaders/{names[0]}"


def _discover_reflection_manifest_url(shaders_root: Path | None) -> str | None:
    if shaders_root is None or not shaders_root.is_dir():
        return None
    candidate = shaders_root / "reflection-manifest.json"
    try:
        if not candidate.is_file():
            return None
        resolved = candidate.resolve()
        if not resolved.is_relative_to(shaders_root.resolve()):
            return None
    except OSError:
        return None
    return "/shaders/reflection-manifest.json"


def _pgim_sort_key(rel_posix: str) -> tuple[int, str, str]:
    """Rank discovered `.pgim` paths: `main/image/guest.pgim`, other `main/*`, top-level, nested."""
    lowered = rel_posix.lower()
    if lowered == "main/image/guest.pgim":
        rank = 0
    elif lowered.startswith("main/"):
        rank = 1
    elif "/" not in lowered:
        rank = 2
    else:
        rank = 3
    return (rank, lowered, rel_posix)


def _discover_pgim_url(boot_root: Path | None) -> str | None:
    if boot_root is None or not boot_root.is_dir():
        return None
    try:
        boot_root_resolved = boot_root.resolve()
    except OSError:
        return None

    candidates: list[str] = []
    seen: set[str] = set()
    try:
        for dirpath, _dirnames, filenames in os.walk(boot_root, followlinks=False):
            dir_path = Path(dirpath)
            for filename in filenames:
                if not filename.lower().endswith(".pgim"):
                    continue
                entry = dir_path / filename
                try:
                    if not entry.is_file():
                        continue
                    resolved = entry.resolve(strict=True)
                    if not resolved.is_relative_to(boot_root_resolved):
                        continue
                    if not resolved.name.lower().endswith(".pgim"):
                        continue
                    rel_posix = entry.relative_to(boot_root).as_posix()
                except (OSError, ValueError):
                    continue
                if not rel_posix or ".." in rel_posix.split("/") or rel_posix in seen:
                    continue
                seen.add(rel_posix)
                candidates.append(rel_posix)
    except OSError:
        return None

    if not candidates:
        return None
    candidates.sort(key=_pgim_sort_key)
    return f"/boot/{candidates[0]}"


def _discover_program_manifest_url(boot_root: Path | None) -> str | None:
    if boot_root is None or not boot_root.is_dir():
        return None
    candidate = boot_root / "program-manifest.json"
    try:
        if not candidate.is_file():
            return None
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(boot_root.resolve()):
            return None
        if not _boot_extension_allowed(resolved):
            return None
    except OSError:
        return None
    return "/boot/program-manifest.json"


def _boot_extension_allowed(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in BOOT_ALLOWED_SUFFIXES)


def _natural_sort_key(name: str) -> tuple[tuple[int, int | str], ...]:
    """Return a case-insensitive natural sort key (e.g., part_2 before part_10)."""
    parts = re.split(r"(\d+)", name.lower())
    return tuple((0, int(part)) if part.isdigit() else (1, part) for part in parts if part)


def is_safe_side_module_name(filename: str) -> bool:
    """Return True only for safe top-level sibling `*.wasm` side-module filenames."""
    if not filename or filename.lower() == RUNTIME_WASM_NAME.lower():
        return False
    if "/" in filename or "\\" in filename or "\x00" in filename or ".." in filename:
        return False
    return SAFE_SIDE_MODULE_NAME_RE.match(filename) is not None


def _discover_side_module_urls(web_root: Path) -> list[str]:
    """Discover top-level `*.wasm` side modules in `web_root/dist` as `/dist/<name>` URLs.

    Excludes `runtime.wasm`, skips directories, nested files, unsafe filenames, and
    broken or escaping symlinks, and returns URLs in natural sort order with zero host paths.
    """
    dist_dir = web_root / "dist"
    if not dist_dir.is_dir():
        return []
    try:
        web_root_resolved = web_root.resolve()
        dist_resolved = dist_dir.resolve(strict=True)
        if not dist_resolved.is_relative_to(web_root_resolved):
            return []
        entries = list(dist_dir.iterdir())
    except OSError:
        return []

    names: list[str] = []
    for entry in entries:
        name = entry.name
        if not is_safe_side_module_name(name):
            continue
        try:
            resolved = entry.resolve(strict=True)
            if not resolved.is_relative_to(dist_resolved):
                continue
            if not resolved.is_file():
                continue
            if not resolved.name.lower().endswith(".wasm"):
                continue
        except OSError:
            continue
        names.append(name)

    names.sort(key=lambda item: (_natural_sort_key(item), item))
    urls: list[str] = []
    seen_lower: set[str] = set()
    for name in names:
        lowered = name.lower()
        if lowered in seen_lower:
            continue
        seen_lower.add(lowered)
        urls.append(f"/dist/{name}")
    return urls


def build_status_payload(
    web_root: Path,
    data_root: Path,
    *,
    game_root: Path | None = None,
    shaders_root: Path | None = None,
    boot_root: Path | None = None,
) -> dict[str, bool | int | str | list[str] | None]:
    """Build sanitized readiness metadata with zero host filesystem paths."""
    data_configured = data_root.is_dir()
    has_manifest = data_configured and (data_root / "data-manifest.json").is_file()
    has_bootset = data_configured and (data_root / "bootset.json").is_file()

    dist_dir = web_root / "dist"
    side_module_urls = _discover_side_module_urls(web_root)
    has_wasm = False
    runtime_wasm = dist_dir / RUNTIME_WASM_NAME
    if runtime_wasm.is_file():
        try:
            web_root_resolved = web_root.resolve()
            dist_resolved = dist_dir.resolve(strict=True)
            runtime_resolved = runtime_wasm.resolve(strict=True)
            has_wasm = (
                dist_resolved.is_relative_to(web_root_resolved)
                and runtime_resolved.is_relative_to(dist_resolved)
                and runtime_resolved.is_file()
            )
        except OSError:
            has_wasm = False
    if not has_wasm:
        has_wasm = bool(side_module_urls)

    effective_shaders = shaders_root if shaders_root is not None else (data_root / "shaders")
    preferred_shader_pack_url = _discover_shader_pack_url(effective_shaders)

    return {
        "ok": True,
        "coopCoep": True,
        "dataDirConfigured": data_configured,
        "hasManifest": has_manifest,
        "hasBootset": has_bootset,
        "hasWasmBinary": has_wasm,
        "sideModuleCount": len(side_module_urls),
        "sideModuleUrls": side_module_urls,
        "shaderPackCount": _count_shader_packs(web_root, data_root, effective_shaders),
        "splashImage": _discover_splash_image_url(game_root),
        "shaderPackUrl": preferred_shader_pack_url,
        "preferredShaderPackUrl": preferred_shader_pack_url,
        "reflectionManifestUrl": _discover_reflection_manifest_url(effective_shaders),
        "pgimUrl": _discover_pgim_url(boot_root),
        "programManifestUrl": _discover_program_manifest_url(boot_root),
        "gameBase": "/game/" if (game_root is not None and game_root.is_dir()) else None,
    }


class PlayGta6Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    default_request_version = "HTTP/1.1"
    timeout = KEEP_ALIVE_TIMEOUT
    disable_nagle_algorithm = True

    def __init__(
        self,
        *args,
        web_root: Path,
        data_root: Path,
        game_root: Path | None = None,
        shaders_root: Path | None = None,
        boot_root: Path | None = None,
        dump_root: Path | None = None,
        max_dump_file_bytes: int = DEFAULT_MAX_DUMP_FILE_BYTES,
        keep_alive_timeout: float | None = None,
        stream_chunk_bytes: int | None = None,
        range_stream_chunk_bytes: int | None = None,
        quiet: bool = False,
        _resolved_roots: dict[str, Path | None] | None = None,
        **kwargs,
    ):
        self.web_root = web_root
        self.data_root = data_root
        self.game_root = game_root
        self.shaders_root = shaders_root if shaders_root is not None else (data_root / "shaders")
        self.boot_root = boot_root
        self.dump_root = dump_root
        self.max_dump_file_bytes = max_dump_file_bytes
        if keep_alive_timeout is not None:
            self.timeout = keep_alive_timeout
        self._stream_chunk_bytes = stream_chunk_bytes
        self._range_stream_chunk_bytes = range_stream_chunk_bytes
        self.quiet = bool(quiet)
        if _resolved_roots is not None:
            self._web_root_resolved = _resolved_roots.get("web")
            self._data_root_resolved = _resolved_roots.get("data")
            self._game_root_resolved = _resolved_roots.get("game")
            self._shaders_root_resolved = _resolved_roots.get("shaders")
            self._boot_root_resolved = _resolved_roots.get("boot")
        else:
            self._web_root_resolved = _try_resolve_root(self.web_root)
            self._data_root_resolved = _try_resolve_root(self.data_root)
            self._game_root_resolved = _try_resolve_root(self.game_root)
            self._shaders_root_resolved = _try_resolve_root(self.shaders_root)
            self._boot_root_resolved = _try_resolve_root(self.boot_root)
        self._headers_sent = False
        self._connection_header_sent = False
        self._client_requested_close = False
        super().__init__(*args, **kwargs)

    def _is_quiet(self) -> bool:
        if getattr(self, "quiet", False):
            return True
        server = getattr(self, "server", None)
        return bool(getattr(server, "quiet", False)) if server is not None else False

    def log_message(self, format: str, *args: Any) -> None:
        """Write a lock-serialized log line to stderr without aborting responses."""
        if self._is_quiet():
            return
        stderr = sys.stderr
        if stderr is None:
            return
        try:
            message = format % args if args else str(format)
            table = getattr(self, "_control_char_table", None)
            if table is not None:
                message = message.translate(table)
            line = (
                f"{self.address_string()} - - "
                f"[{self.log_date_time_string()}] {message}\n"
            )
        except (OSError, ValueError):
            return
        with _LOG_LOCK:
            try:
                stderr.write(line)
            except (OSError, ValueError):
                return

    def setup(self) -> None:
        super().setup()
        self._configure_client_socket()

    def _configure_client_socket(self) -> None:
        conn = getattr(self, "connection", None)
        if conn is None or not hasattr(conn, "setsockopt"):
            return
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        if SOCKET_SNDBUF_BYTES > 0:
            try:
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SOCKET_SNDBUF_BYTES)
            except OSError:
                pass

    def _get_stream_chunk_bytes(self) -> int:
        configured = getattr(self, "_stream_chunk_bytes", None)
        if configured is not None:
            return configured
        if STREAM_CHUNK_BYTES != 1024 * 1024:
            return STREAM_CHUNK_BYTES
        return STREAM_CHUNK_SIZE

    def _get_range_stream_chunk_bytes(self) -> int:
        configured_range = getattr(self, "_range_stream_chunk_bytes", None)
        if configured_range is not None:
            return configured_range
        configured_stream = getattr(self, "_stream_chunk_bytes", None)
        if configured_stream is not None:
            return configured_stream
        return RANGE_STREAM_CHUNK_BYTES

    def _request_has_unconsumed_body(self) -> bool:
        if not hasattr(self, "headers") or self.headers is None:
            return False
        if self.headers.get("Transfer-Encoding"):
            return True
        raw_length = self.headers.get("Content-Length")
        if not raw_length:
            return False
        try:
            return int(raw_length.strip()) > 0
        except ValueError:
            return True

    def parse_request(self) -> bool:
        ok = super().parse_request()
        if not ok:
            self.close_connection = True
            return False

        if hasattr(self.headers, "get_all"):
            raw_conn = self.headers.get_all("Connection") or []
        else:
            single = self.headers.get("Connection")
            raw_conn = [single] if single else []

        tokens = {
            tok.strip().lower()
            for value in raw_conn
            for tok in value.split(",")
            if tok.strip()
        }
        self._client_requested_close = "close" in tokens
        if self._client_requested_close or self._request_has_unconsumed_body():
            self.close_connection = True
        elif "keep-alive" in tokens and self.protocol_version >= "HTTP/1.1":
            self.close_connection = False
        return True

    def send_header(self, keyword: str, value: str) -> None:
        if keyword.lower() == "connection":
            self._connection_header_sent = True
        super().send_header(keyword, value)

    def date_time_string(self, timestamp: float | None = None) -> str:
        if timestamp is None:
            return _cached_http_date_now()
        return super().date_time_string(timestamp)

    def _append_standard_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Accept-Ranges", "bytes")
        if not getattr(self, "_connection_header_sent", False):
            if getattr(self, "close_connection", False):
                self.send_header("Connection", "close")
            else:
                self.send_header("Connection", "keep-alive")
                if isinstance(self.timeout, (int, float)) and self.timeout > 0:
                    self.send_header("Keep-Alive", f"timeout={max(1, int(self.timeout))}")

    def end_headers(self) -> None:
        self._append_standard_headers()
        self._headers_sent = True
        super().end_headers()

    def _finish_headers_buffered(self) -> bytes:
        """Finalize headers into bytes without writing to the socket yet (for header+body coalescing)."""
        self._append_standard_headers()
        if self.request_version != "HTTP/0.9":
            self._headers_buffer.append(b"\r\n")
        header_bytes = b"".join(self._headers_buffer)
        self._headers_buffer = []
        return header_bytes

    def send_error(
        self,
        code: int,
        message: str | None = None,
        explain: str | None = None,
    ) -> None:
        self.close_connection = True
        try:
            super().send_error(code, message, explain)
        except OSError as exc:
            if not _is_client_disconnect_error(exc):
                raise
            self.close_connection = True

    def _send_empty(self, status: HTTPStatus) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _send_json(
        self,
        status: HTTPStatus,
        payload: dict[str, Any],
        *,
        close: bool = False,
    ) -> None:
        if close:
            if self._request_has_unconsumed_body():
                self._drain_small_request_body()
            self.close_connection = True
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if hasattr(self.wfile, "_sock"):
            header_bytes = self._finish_headers_buffered()
            self._headers_sent = True
            try:
                self.wfile.write(header_bytes + body)
            except OSError as exc:
                self.close_connection = True
                if _is_client_disconnect_error(exc):
                    return
                raise
            except Exception:
                self.close_connection = True
                raise
            return
        self.end_headers()
        try:
            self.wfile.write(body)
        except OSError as exc:
            self.close_connection = True
            if _is_client_disconnect_error(exc):
                return
            raise
        except Exception:
            self.close_connection = True
            raise

    def _drain_small_request_body(
        self, max_bytes: int = MAX_METHOD_NOT_ALLOWED_DRAIN_BYTES
    ) -> None:
        """Drain a small unread request body once before closing to avoid Windows TCP RST races."""
        if getattr(self, "_request_body_consumed", False):
            return
        self._request_body_consumed = True
        headers = getattr(self, "headers", None)
        rfile = getattr(self, "rfile", None)
        if headers is None or rfile is None:
            return
        if headers.get("Transfer-Encoding"):
            return
        raw_length = headers.get("Content-Length")
        if not raw_length:
            return
        try:
            content_length = int(raw_length.strip())
        except ValueError:
            return
        if content_length <= 0 or content_length > max_bytes:
            return
        remaining = content_length
        try:
            while remaining > 0:
                chunk = rfile.read(min(remaining, 8192))
                if not chunk:
                    break
                remaining -= len(chunk)
        except (CLIENT_DISCONNECT_ERRORS, OSError):
            pass

    def _send_method_not_allowed(self) -> None:
        # Drain small unread bodies before closing so Windows loopback clients do
        # not hit WinError 10053 TCP RST races while reading the 405 response.
        self._drain_small_request_body()
        # Close the connection because unsupported methods (e.g. POST/PUT) may
        # carry an unread request body that would desynchronize keep-alive.
        self.close_connection = True
        self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
        self.send_header("Allow", ALLOWED_METHODS)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        wfile = getattr(self, "wfile", None)
        if wfile is not None and hasattr(wfile, "flush"):
            try:
                wfile.flush()
            except CLIENT_DISCONNECT_ERRORS:
                pass

    def handle_one_request(self) -> None:
        """Handle a single HTTP request with quiet disconnect/timeout handling."""
        self._headers_sent = False
        self._connection_header_sent = False
        self._request_body_consumed = False
        try:
            self.raw_requestline = self.rfile.readline(65537)
            if len(self.raw_requestline) > 65536:
                self.requestline = ""
                self.request_version = self.default_request_version
                self.command = ""
                self.close_connection = True
                self.send_error(HTTPStatus.REQUEST_URI_TOO_LONG)
                return
            if not self.raw_requestline:
                self.close_connection = True
                return
            if not self.parse_request():
                self.close_connection = True
                return
            mname = "do_" + self.command
            if not hasattr(self, mname):
                self._send_method_not_allowed()
                return
            method = getattr(self, mname)
            method()
            self.wfile.flush()
        except OSError as exc:
            self.close_connection = True
            if _is_client_disconnect_error(exc):
                return
            if not self._headers_sent:
                try:
                    self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR)
                    self.wfile.flush()
                except OSError:
                    pass
            raise
        except Exception:
            self.close_connection = True
            if not self._headers_sent:
                try:
                    self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR)
                    self.wfile.flush()
                except OSError:
                    pass
            raise

    def _get_resolved_path_cache(self) -> _ResolvedPathCache | None:
        cache = getattr(self, "resolved_path_cache", None)
        if cache is not None:
            return cache
        server = getattr(self, "server", None)
        return getattr(server, "resolved_path_cache", None) if server is not None else None

    def _get_interstage_sidecar_cache(self) -> _InterstageSidecarCache:
        cache = getattr(self, "interstage_sidecar_cache", None)
        if cache is not None:
            return cache
        server = getattr(self, "server", None)
        server_cache = getattr(server, "interstage_sidecar_cache", None) if server is not None else None
        return server_cache if server_cache is not None else _DEFAULT_INTERSTAGE_SIDECAR_CACHE

    def _try_serve_dynamic_interstage(self, url_path: str, send_body: bool) -> bool:
        """Dynamically extract and serve `/shaders/<name>.interstage.json` from `<name>.reflect.json`.

        Called only when no static `<name>.interstage.json` exists on disk. Returns True if a
        response was sent, False to fall through to a normal 404.
        """
        if not url_path.startswith("/shaders/") or self.shaders_root is None:
            return False
        rel = url_path[len("/shaders/") :]
        normalized = _normalize_safe_relpath(rel)
        if normalized is None or not normalized.lower().endswith(INTERSTAGE_SIDECAR_SUFFIX):
            return False
        stem = normalized[: -len(INTERSTAGE_SIDECAR_SUFFIX)]
        if not stem or stem.endswith("/"):
            return False
        reflect_rel = f"{stem}{REFLECT_SIDECAR_SUFFIX}"
        reflect_path = safe_resolve(
            self.shaders_root,
            reflect_rel,
            root_resolved=getattr(self, "_shaders_root_resolved", None),
        )
        if reflect_path is None:
            return False
        try:
            st = reflect_path.stat()
        except OSError:
            return False
        if not stat.S_ISREG(st.st_mode):
            return False

        cache = self._get_interstage_sidecar_cache()
        try:
            body = cache.get_or_build(reflect_path, st, realpath=str(reflect_path))
        except (OSError, ValueError):
            return False

        etag = f'"interstage-{st.st_size:x}-{st.st_mtime_ns:x}"'
        last_modified = _format_mtime_http_date(st.st_mtime)
        if_match = self._get_combined_header("If-Match")
        if if_match is not None and not etag_matches_if_match(if_match, etag):
            self._send_precondition_failed(etag, last_modified)
            return True
        if_none_match = self._get_combined_header("If-None-Match")
        if if_none_match and etag_matches_if_none_match(if_none_match, etag):
            self._send_not_modified(etag, last_modified)
            return True

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("ETag", etag)
        self.send_header("Last-Modified", last_modified)
        self.send_header("Cache-Control", "no-cache")
        if not send_body:
            self.end_headers()
            return True
        if hasattr(self.wfile, "_sock"):
            header_bytes = self._finish_headers_buffered()
            self._headers_sent = True
            try:
                self.wfile.write(header_bytes + body)
            except OSError as exc:
                self.close_connection = True
                if _is_client_disconnect_error(exc):
                    return True
                raise
            except Exception:
                self.close_connection = True
                raise
            return True
        self.end_headers()
        try:
            self.wfile.write(body)
        except OSError as exc:
            self.close_connection = True
            if _is_client_disconnect_error(exc):
                return True
            raise
        except Exception:
            self.close_connection = True
            raise
        return True

    def _classify_route(
        self, url_path: str
    ) -> tuple[str, Path, Path | None, str, bool] | None:
        """Classify `url_path` into `(mount, root, root_resolved, rel_url, is_boot)`."""
        if url_path == "/":
            return ("web", self.web_root, getattr(self, "_web_root_resolved", None), "index.html", False)
        if url_path.startswith("/data/"):
            return (
                "data",
                self.data_root,
                getattr(self, "_data_root_resolved", None),
                url_path[len("/data/") :],
                False,
            )
        if url_path.startswith("/game/"):
            if self.game_root is None:
                return None
            return (
                "game",
                self.game_root,
                getattr(self, "_game_root_resolved", None),
                url_path[len("/game/") :],
                False,
            )
        if url_path.startswith("/shaders/"):
            if self.shaders_root is None:
                return None
            return (
                "shaders",
                self.shaders_root,
                getattr(self, "_shaders_root_resolved", None),
                url_path[len("/shaders/") :],
                False,
            )
        if url_path.startswith("/boot/"):
            if self.boot_root is None:
                return None
            return (
                "boot",
                self.boot_root,
                getattr(self, "_boot_root_resolved", None),
                url_path[len("/boot/") :],
                True,
            )
        return (
            "web",
            self.web_root,
            getattr(self, "_web_root_resolved", None),
            url_path.lstrip("/"),
            False,
        )

    def _resolve_target(self, url_path: str) -> Path | None | tuple[None, HTTPStatus]:
        """Resolve a URL path to a filesystem target.

        Returns:
            Path on success, None for a generic 404, or (None, status) for an
            explicit error status (e.g. 403 for disallowed `/boot/` extensions).
        """
        route = self._classify_route(url_path)
        if route is None:
            return None
        _mount, root, root_resolved, rel, is_boot = route
        target = safe_resolve(root, rel, root_resolved=root_resolved)
        if target is None:
            return None
        if is_boot and not _boot_extension_allowed(target):
            return (None, HTTPStatus.FORBIDDEN)
        return target

    def _resolve_target_fast(
        self, url_path: str
    ) -> tuple[Path | None, str | None, str | None, tuple[str, str] | None, bool, HTTPStatus | None]:
        """Resolve `url_path` using the bounded LRU cache for verified non-symlink paths.

        Returns `(target, realpath, content_type, cache_key, from_cache, err_status)`.
        """
        route = self._classify_route(url_path)
        if route is None:
            return (None, None, None, None, False, None)
        mount, root, root_resolved, rel, is_boot = route
        normalized = _normalize_safe_relpath(rel)
        if normalized is None:
            return (None, None, None, None, False, None)

        path_cache = self._get_resolved_path_cache()
        cache_key: tuple[str, str] | None = None
        if path_cache is not None and root_resolved is not None:
            cache_key = (f"{mount}:{root_resolved}", normalized)
            cached = path_cache.get(cache_key)
            if cached is not None:
                c_target, c_realpath, c_ctype = cached
                if is_boot and not _boot_extension_allowed(c_target):
                    return (None, None, None, None, False, HTTPStatus.FORBIDDEN)
                return (c_target, c_realpath, c_ctype, cache_key, True, None)

        try:
            eff_root = root_resolved if root_resolved is not None else root.resolve()
            candidate_raw = eff_root / normalized
            target = candidate_raw.resolve()
            target.relative_to(eff_root)
        except (OSError, ValueError):
            return (None, None, None, None, False, None)

        if is_boot and not _boot_extension_allowed(target):
            return (None, None, None, None, False, HTTPStatus.FORBIDDEN)

        realpath_str = str(target) if target.is_absolute() else None
        content_type, _ = mimetypes.guess_type(realpath_str or str(target))
        content_type = content_type or "application/octet-stream"

        # Only allow caching when candidate_raw was not a symlink/reparse redirect so
        # subsequent hits can verify non-symlink regular file identity in a single os.lstat().
        can_cache = (
            path_cache is not None
            and cache_key is not None
            and realpath_str is not None
            and target == candidate_raw
        )
        return (target, realpath_str, content_type, cache_key if can_cache else None, False, None)

    def _serve_status(self, send_body: bool) -> None:
        payload = build_status_payload(
            self.web_root,
            self.data_root,
            game_root=self.game_root,
            shaders_root=self.shaders_root,
            boot_root=self.boot_root,
        )
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if send_body:
            try:
                self.wfile.write(body)
            except CLIENT_DISCONNECT_ERRORS:
                self.close_connection = True
                return
            except Exception:
                self.close_connection = True
                raise

    def _get_file_handle_cache(self) -> ReadOnlyFileHandleCache | None:
        cache = getattr(self, "file_handle_cache", None)
        if cache is not None:
            return cache
        server = getattr(self, "server", None)
        return getattr(server, "file_handle_cache", None) if server is not None else None

    def _get_combined_header(self, name: str) -> str | None:
        if hasattr(self.headers, "get_all"):
            values = self.headers.get_all(name)
            return ", ".join(values) if values else None
        return self.headers.get(name)

    def _send_not_modified(self, etag: str, last_modified: str) -> None:
        self.send_response(HTTPStatus.NOT_MODIFIED)
        self.send_header("ETag", etag)
        self.send_header("Last-Modified", last_modified)
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

    def _send_precondition_failed(self, etag: str, last_modified: str) -> None:
        self.send_response(HTTPStatus.PRECONDITION_FAILED)
        self.send_header("Content-Length", "0")
        self.send_header("ETag", etag)
        self.send_header("Last-Modified", last_modified)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _stream_cached_file_range(
        self,
        entry: _CachedFileHandle,
        start: int,
        length: int,
        cache: ReadOnlyFileHandleCache,
        *,
        chunk_bytes: int | None = None,
        pending_headers: bytes | None = None,
    ) -> None:
        effective_chunk = max(
            1,
            chunk_bytes if chunk_bytes is not None else self._get_range_stream_chunk_bytes(),
        )
        offset = start
        remaining = length
        write = self.wfile.write
        headers_to_send = pending_headers
        with entry.borrow_fd() as reader:
            use_readinto = (
                type(reader).read_at is _ORIG_BORROWED_READ_AT
                and os.read is _ORIG_OS_READ
                and reader.can_readinto()
                and hasattr(self.wfile, "_sock")
            )
            if use_readinto:
                header_len = len(headers_to_send) if headers_to_send else 0
                buf = _get_thread_stream_buffer(header_len + effective_chunk)
                mv = memoryview(buf)
                while remaining > 0:
                    to_read = effective_chunk if remaining > effective_chunk else remaining
                    if headers_to_send is not None:
                        hdr_len = len(headers_to_send)
                        buf[:hdr_len] = headers_to_send
                        try:
                            n = reader.readinto_at(offset, mv[hdr_len : hdr_len + to_read])
                        except OSError:
                            cache.invalidate_entry(entry)
                            raise
                        self._headers_sent = True
                        headers_to_send = None
                        if not n:
                            cache.invalidate_entry(entry)
                            self.close_connection = True
                            write(mv[:hdr_len])
                            break
                        offset += n
                        remaining -= n
                        write(mv[: hdr_len + n])
                    else:
                        try:
                            n = reader.readinto_at(offset, mv[:to_read])
                        except OSError:
                            cache.invalidate_entry(entry)
                            raise
                        if not n:
                            cache.invalidate_entry(entry)
                            self.close_connection = True
                            break
                        offset += n
                        remaining -= n
                        write(mv[:n])
                return

            while remaining > 0:
                to_read = effective_chunk if remaining > effective_chunk else remaining
                try:
                    chunk = reader.read_at(offset, to_read)
                except OSError:
                    cache.invalidate_entry(entry)
                    raise
                if not chunk:
                    # File shrank mid-stream after Content-Length was emitted:
                    # invalidate handle and force socket close so HTTP/1.1 framing stays synced.
                    cache.invalidate_entry(entry)
                    self.close_connection = True
                    if headers_to_send is not None:
                        self._headers_sent = True
                        write(headers_to_send)
                        headers_to_send = None
                    break
                chunk_len = len(chunk)
                offset += chunk_len
                remaining -= chunk_len
                if headers_to_send is not None:
                    self._headers_sent = True
                    write(headers_to_send + chunk)
                    headers_to_send = None
                else:
                    write(chunk)

    def _stream_file_range(
        self,
        f: Any,
        length: int,
        *,
        chunk_bytes: int | None = None,
        pending_headers: bytes | None = None,
    ) -> None:
        effective_chunk = max(
            1,
            chunk_bytes if chunk_bytes is not None else self._get_stream_chunk_bytes(),
        )
        readinto = getattr(f, "readinto", None)
        use_fast_buffer = callable(readinto) and hasattr(self.wfile, "_sock")
        headers_to_send = pending_headers

        remaining = length
        if use_fast_buffer:
            header_len = len(headers_to_send) if headers_to_send else 0
            buf = _get_thread_stream_buffer(header_len + effective_chunk)
            mv = memoryview(buf)
            while remaining > 0:
                to_read = effective_chunk if remaining > effective_chunk else remaining
                if headers_to_send is not None:
                    hdr_len = len(headers_to_send)
                    buf[:hdr_len] = headers_to_send
                    n = readinto(mv[hdr_len : hdr_len + to_read])
                    self._headers_sent = True
                    headers_to_send = None
                    if not n:
                        self.close_connection = True
                        self.wfile.write(mv[:hdr_len])
                        break
                    remaining -= n
                    if remaining == 0:
                        f.close()
                    self.wfile.write(mv[: hdr_len + n])
                else:
                    n = readinto(mv[:to_read])
                    if not n:
                        # File shrank mid-stream after Content-Length was emitted:
                        # force socket close so HTTP/1.1 framing cannot desynchronize.
                        self.close_connection = True
                        break
                    remaining -= n
                    if remaining == 0:
                        # Release the file before the last bytes reach the client: once the
                        # response is complete the file can be replaced or deleted (Windows
                        # refuses to delete a file that is still open).
                        f.close()
                    self.wfile.write(mv[:n])
            return

        while remaining > 0:
            chunk = f.read(min(effective_chunk, remaining))
            if not chunk:
                # File shrank mid-stream after Content-Length was emitted:
                # force socket close so HTTP/1.1 framing cannot desynchronize.
                self.close_connection = True
                if headers_to_send is not None:
                    self._headers_sent = True
                    self.wfile.write(headers_to_send)
                    headers_to_send = None
                break
            remaining -= len(chunk)
            if remaining == 0:
                # Release the file before the last bytes reach the client: once the
                # response is complete the file can be replaced or deleted (Windows
                # refuses to delete a file that is still open).
                f.close()
            if headers_to_send is not None:
                self._headers_sent = True
                self.wfile.write(headers_to_send + chunk)
                headers_to_send = None
            else:
                self.wfile.write(chunk)

    def _serve_file(self, url_path: str, send_body: bool) -> None:
        path_cache = self._get_resolved_path_cache()
        use_fast_resolve = (
            type(self)._resolve_target is PlayGta6Handler._resolve_target
            and "_resolve_target" not in self.__dict__
        )
        if use_fast_resolve:
            target, target_realpath, content_type, path_cache_key, from_path_cache, err_status = (
                self._resolve_target_fast(url_path)
            )
        else:
            resolved = self._resolve_target(url_path)
            if isinstance(resolved, tuple):
                target, target_realpath, content_type, path_cache_key, from_path_cache, err_status = (
                    None,
                    None,
                    None,
                    None,
                    False,
                    resolved[1],
                )
            else:
                target = resolved
                target_realpath = (
                    str(target) if (isinstance(target, Path) and target.is_absolute()) else None
                )
                content_type = None
                path_cache_key = None
                from_path_cache = False
                err_status = None
        if err_status is not None:
            self._send_empty(err_status)
            return
        if target is None:
            self._send_empty(HTTPStatus.NOT_FOUND)
            return

        cache = self._get_file_handle_cache()
        st: os.stat_result | None = None
        if from_path_cache:
            try:
                st = os.lstat(target)
                if stat.S_ISLNK(st.st_mode):
                    # File was replaced by a symlink; evict cached path and re-resolve fully.
                    if path_cache is not None and path_cache_key is not None:
                        path_cache.invalidate(path_cache_key)
                    if cache is not None and isinstance(target, Path):
                        cache.invalidate(target, realpath=target_realpath)
                    st = None
                    from_path_cache = False
                    resolved = self._resolve_target(url_path)
                    if isinstance(resolved, tuple):
                        self._send_empty(resolved[1])
                        return
                    target = resolved
                    if target is None:
                        self._send_empty(HTTPStatus.NOT_FOUND)
                        return
                    target_realpath = (
                        str(target) if (isinstance(target, Path) and target.is_absolute()) else None
                    )
                    content_type, _ = mimetypes.guess_type(target_realpath or str(target))
                    content_type = content_type or "application/octet-stream"
                    path_cache_key = None
            except OSError:
                if path_cache is not None and path_cache_key is not None:
                    path_cache.invalidate(path_cache_key)
                if cache is not None and isinstance(target, Path):
                    cache.invalidate(target, realpath=target_realpath)
                if self._try_serve_dynamic_interstage(url_path, send_body):
                    return
                self._send_empty(HTTPStatus.NOT_FOUND)
                return

        if st is None:
            try:
                st = target.stat()
            except OSError:
                if cache is not None and isinstance(target, Path):
                    cache.invalidate(target, realpath=target_realpath)
                if self._try_serve_dynamic_interstage(url_path, send_body):
                    return
                self._send_empty(HTTPStatus.NOT_FOUND)
                return

        if not stat.S_ISREG(st.st_mode):
            if from_path_cache and path_cache is not None and path_cache_key is not None:
                path_cache.invalidate(path_cache_key)
            if cache is not None and isinstance(target, Path):
                cache.invalidate(target, realpath=target_realpath)
            self._send_empty(HTTPStatus.NOT_FOUND)
            return

        if (
            not from_path_cache
            and path_cache is not None
            and path_cache_key is not None
            and target_realpath is not None
            and content_type is not None
        ):
            path_cache.put(path_cache_key, target, target_realpath, content_type)

        if content_type is None:
            content_type, _ = mimetypes.guess_type(target_realpath or str(target))
            content_type = content_type or "application/octet-stream"

        if_match = self._get_combined_header("If-Match")
        range_header = self.headers.get("Range")
        if_none_match = self._get_combined_header("If-None-Match") if not range_header else None
        if_modified_since = (
            self.headers.get("If-Modified-Since")
            if (not range_header and not if_none_match)
            else None
        )
        if_range_header = self.headers.get("If-Range")

        def evaluate_request_against_stat(
            cur_st: os.stat_result,
        ) -> tuple[str, int, str, str, int, int, int, HTTPStatus]:
            cur_size = cur_st.st_size
            cur_etag = compute_etag(cur_st)
            cur_last_modified = _format_mtime_http_date(cur_st.st_mtime)

            # RFC 9110 §13.2.2 Step 1: evaluate If-Match (strong comparison) before Range / If-None-Match.
            if if_match is not None and not etag_matches_if_match(if_match, cur_etag):
                return ("412", cur_size, cur_etag, cur_last_modified, 0, 0, 0, HTTPStatus.PRECONDITION_FAILED)

            # Never short-circuit HTTP Range requests to 304 Not Modified: browsers
            # attach cached If-None-Match / If-Modified-Since headers when fetching
            # subsequent slices of large .rpf / .wgsl.pack files, and returning 304
            # starves or corrupts ranged streaming (RFC 9110 §13.1.5 uses If-Range).
            if not range_header:
                if if_none_match:
                    if etag_matches_if_none_match(if_none_match, cur_etag):
                        return ("304", cur_size, cur_etag, cur_last_modified, 0, 0, 0, HTTPStatus.NOT_MODIFIED)
                elif if_modified_since and if_modified_since_not_modified(if_modified_since, cur_st.st_mtime):
                    return ("304", cur_size, cur_etag, cur_last_modified, 0, 0, 0, HTTPStatus.NOT_MODIFIED)

            apply_range = bool(range_header)
            if apply_range and if_range_header is not None:
                if not if_range_matches(if_range_header, cur_etag, cur_last_modified, cur_st.st_mtime):
                    apply_range = False

            cur_start, cur_end = 0, max(cur_size - 1, 0)
            cur_status = HTTPStatus.OK
            if apply_range and range_header:
                range_kind, parsed = parse_byte_range_status(range_header, cur_size)
                if range_kind == "unsatisfiable":
                    return (
                        "416",
                        cur_size,
                        cur_etag,
                        cur_last_modified,
                        0,
                        0,
                        0,
                        HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                    )
                if range_kind == "ok" and parsed is not None:
                    cur_start, cur_end = parsed
                    cur_status = HTTPStatus.PARTIAL_CONTENT

            cur_length = (cur_end - cur_start + 1) if cur_size > 0 else 0
            return ("send", cur_size, cur_etag, cur_last_modified, cur_start, cur_end, cur_length, cur_status)

        action, file_size, etag, last_modified, start, end, length, status = (
            evaluate_request_against_stat(st)
        )
        if action == "412":
            self._send_precondition_failed(etag, last_modified)
            return
        if action == "304":
            self._send_not_modified(etag, last_modified)
            return
        if action == "416":
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{file_size}")
            self.send_header("Content-Length", "0")
            self.send_header("ETag", etag)
            self.send_header("Last-Modified", last_modified)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return

        def emit_file_headers(
            cur_status: HTTPStatus,
            cur_length: int,
            cur_etag: str,
            cur_last_modified: str,
            cur_start: int,
            cur_end: int,
            cur_size: int,
            *,
            buffered: bool = False,
        ) -> bytes | None:
            self.send_response(cur_status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(cur_length))
            self.send_header("ETag", cur_etag)
            self.send_header("Last-Modified", cur_last_modified)
            self.send_header("Cache-Control", "no-cache")
            if cur_status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {cur_start}-{cur_end}/{cur_size}")
            if buffered:
                return self._finish_headers_buffered()
            self.end_headers()
            return None

        if not send_body or length == 0:
            emit_file_headers(status, length, etag, last_modified, start, end, file_size)
            return

        can_coalesce_headers = hasattr(self.wfile, "_sock")
        chunk_bytes = (
            self._get_range_stream_chunk_bytes()
            if status == HTTPStatus.PARTIAL_CONTENT
            else self._get_stream_chunk_bytes()
        )

        if cache is not None and isinstance(target, Path) and is_cacheable_file_path(target):
            cached_entry = cache.acquire(target, st, realpath=target_realpath)
            if cached_entry is not None:
                try:
                    pending_hdrs = emit_file_headers(
                        status,
                        length,
                        etag,
                        last_modified,
                        start,
                        end,
                        file_size,
                        buffered=can_coalesce_headers,
                    )
                    self._stream_cached_file_range(
                        cached_entry,
                        start,
                        length,
                        cache,
                        chunk_bytes=chunk_bytes,
                        pending_headers=pending_hdrs,
                    )
                except OSError as exc:
                    self.close_connection = True
                    if _is_client_disconnect_error(exc):
                        return
                    raise
                except Exception:
                    self.close_connection = True
                    raise
                finally:
                    cache.release(cached_entry)
                return

        try:
            f = target.open("rb")
        except OSError:
            self._send_empty(HTTPStatus.NOT_FOUND)
            return

        try:
            with f:
                try:
                    fstat = os.fstat(f.fileno())
                except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
                    fstat = None
                if fstat is not None and (
                    fstat.st_size != st.st_size or fstat.st_mtime_ns != st.st_mtime_ns
                ):
                    if not stat.S_ISREG(fstat.st_mode):
                        self._send_empty(HTTPStatus.NOT_FOUND)
                        return
                    action, file_size, etag, last_modified, start, end, length, status = (
                        evaluate_request_against_stat(fstat)
                    )
                    if action == "412":
                        self._send_precondition_failed(etag, last_modified)
                        return
                    if action == "304":
                        self._send_not_modified(etag, last_modified)
                        return
                    if action == "416":
                        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                        self.send_header("Content-Range", f"bytes */{file_size}")
                        self.send_header("Content-Length", "0")
                        self.send_header("ETag", etag)
                        self.send_header("Last-Modified", last_modified)
                        self.send_header("Cache-Control", "no-store")
                        self.end_headers()
                        return
                    if length == 0:
                        emit_file_headers(
                            status, length, etag, last_modified, start, end, file_size
                        )
                        return
                    chunk_bytes = (
                        self._get_range_stream_chunk_bytes()
                        if status == HTTPStatus.PARTIAL_CONTENT
                        else self._get_stream_chunk_bytes()
                    )

                if start > 0:
                    f.seek(start)
                pending_hdrs = emit_file_headers(
                    status,
                    length,
                    etag,
                    last_modified,
                    start,
                    end,
                    file_size,
                    buffered=can_coalesce_headers,
                )
                self._stream_file_range(
                    f,
                    length,
                    chunk_bytes=chunk_bytes,
                    pending_headers=pending_hdrs,
                )
        except OSError as exc:
            self.close_connection = True
            if _is_client_disconnect_error(exc):
                return
            raise
        except Exception:
            self.close_connection = True
            raise

    def _dispatch(self, send_body: bool) -> None:
        url_path = urlsplit(self.path).path
        if url_path == "/api/status":
            self._serve_status(send_body=send_body)
            return
        self._serve_file(url_path, send_body=send_body)

    def _parse_content_length(self) -> tuple[int | None, HTTPStatus | None, str | None]:
        if self.headers.get("Transfer-Encoding"):
            return None, HTTPStatus.LENGTH_REQUIRED, "Content-Length required"
        if hasattr(self.headers, "get_all"):
            raw_values = self.headers.get_all("Content-Length") or []
        else:
            single = self.headers.get("Content-Length")
            raw_values = [single] if single else []
        if not raw_values:
            return None, HTTPStatus.LENGTH_REQUIRED, "Content-Length header required"

        parsed_values: set[int] = set()
        for raw in raw_values:
            cleaned = raw.strip()
            if not cleaned or not cleaned.isdigit():
                return None, HTTPStatus.BAD_REQUEST, "invalid Content-Length header"
            parsed_values.add(int(cleaned))
        if len(parsed_values) != 1:
            return None, HTTPStatus.BAD_REQUEST, "conflicting Content-Length headers"
        return next(iter(parsed_values)), None, None

    def _handle_dump_post(self, parsed_url: Any) -> None:
        if self.dump_root is None:
            self._drain_small_request_body()  # avoid Windows RST (WinError 10053) on unread body
            self._send_json(
                HTTPStatus.FORBIDDEN,
                {"ok": False, "error": "dump endpoint disabled"},
                close=True,
            )
            return

        rel_url_path = parsed_url.path[len(DUMP_URL_PREFIX) :]
        resolved, err_status, err_msg = resolve_dump_target(self.dump_root, rel_url_path)
        if resolved is None:
            self._drain_small_request_body()  # avoid Windows RST (WinError 10053) on unread body
            self._send_json(
                err_status or HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": err_msg or "invalid dump path"},
                close=True,
            )
            return
        target, sanitized_relpath = resolved

        content_length, len_status, len_msg = self._parse_content_length()
        if content_length is None:
            self._send_json(
                len_status or HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": len_msg or "invalid Content-Length"},
                close=True,
            )
            return

        max_bytes = self.max_dump_file_bytes
        if content_length > max_bytes:
            self._send_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"ok": False, "error": "payload exceeds max dump file size"},
                close=True,
            )
            return

        query = parse_qs(parsed_url.query, keep_blank_values=True)
        append_vals = query.get("append", [])
        append_mode = any(v.strip().lower() in ("1", "true", "yes") for v in append_vals)

        if append_mode:
            try:
                existing_size = target.stat().st_size if target.exists() else 0
            except OSError:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": "dump write failed"},
                    close=True,
                )
                return
            if existing_size + content_length > max_bytes:
                self._send_json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"ok": False, "error": "payload exceeds max dump file size"},
                    close=True,
                )
                return

        try:
            root_resolved = self.dump_root.resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            parent_resolved = target.parent.resolve(strict=True)
            if not parent_resolved.is_relative_to(root_resolved):
                self._send_json(
                    HTTPStatus.FORBIDDEN,
                    {"ok": False, "error": "path escapes dump directory"},
                    close=True,
                )
                return
        except OSError:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error": "dump write failed"},
                close=True,
            )
            return

        tmp_path: Path | None = None
        bytes_written = 0
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=target.parent,
                prefix=".pg6_dump_",
                suffix=".tmp",
                delete=False,
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                remaining = content_length
                chunk_bytes = max(1, self._get_stream_chunk_bytes())
                while remaining > 0:
                    chunk = self.rfile.read(min(chunk_bytes, remaining))
                    if not chunk:
                        self._request_body_consumed = True
                        self._send_json(
                            HTTPStatus.BAD_REQUEST,
                            {"ok": False, "error": "incomplete request body"},
                            close=True,
                        )
                        return
                    bytes_written += len(chunk)
                    if bytes_written > max_bytes:
                        self._request_body_consumed = True
                        self._send_json(
                            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                            {"ok": False, "error": "payload exceeds max dump file size"},
                            close=True,
                        )
                        return
                    tmp_file.write(chunk)
                    remaining -= len(chunk)

            # Request body is completely consumed; keep-alive is safe if the client wants it.
            self._request_body_consumed = True
            if self.protocol_version >= "HTTP/1.1" and not getattr(
                self, "_client_requested_close", False
            ):
                self.close_connection = False

            with _DUMP_WRITE_LOCK:
                # Re-validate target containment under lock before committing bytes.
                recheck, re_status, re_msg = resolve_dump_target(self.dump_root, rel_url_path)
                if recheck is None:
                    self._send_json(
                        re_status or HTTPStatus.FORBIDDEN,
                        {"ok": False, "error": re_msg or "invalid dump target"},
                        close=True,
                    )
                    return

                cache = self._get_file_handle_cache()
                if cache is not None:
                    cache.invalidate(target)
                path_cache = self._get_resolved_path_cache()
                if path_cache is not None:
                    path_cache.clear()

                if append_mode:
                    current_size = target.stat().st_size if target.exists() else 0
                    if current_size + bytes_written > max_bytes:
                        self._send_json(
                            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                            {"ok": False, "error": "payload exceeds max dump file size"},
                            close=True,
                        )
                        return
                    with target.open("ab") as dest, tmp_path.open("rb") as src:
                        chunk_bytes = max(1, self._get_stream_chunk_bytes())
                        while True:
                            buf = src.read(chunk_bytes)
                            if not buf:
                                break
                            dest.write(buf)
                    total_bytes = target.stat().st_size
                else:
                    os.replace(tmp_path, target)
                    tmp_path = None
                    total_bytes = bytes_written
        except CLIENT_DISCONNECT_ERRORS:
            self.close_connection = True
            return
        except OSError:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error": "dump write failed"},
                close=True,
            )
            return
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass

        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "path": sanitized_relpath,
                "bytes_written": bytes_written,
                "total_bytes": total_bytes,
            },
        )

    def do_GET(self) -> None:
        self._dispatch(send_body=True)

    def do_HEAD(self) -> None:
        self._dispatch(send_body=False)

    def do_POST(self) -> None:
        parsed_url = urlsplit(self.path)
        if parsed_url.path.startswith(DUMP_URL_PREFIX):
            self._handle_dump_post(parsed_url)
            return
        self._send_method_not_allowed()

    def do_OPTIONS(self) -> None:
        url_path = urlsplit(self.path).path
        allow = (
            DUMP_ALLOWED_METHODS
            if (self.dump_root is not None and url_path.startswith(DUMP_URL_PREFIX))
            else ALLOWED_METHODS
        )
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Allow", allow)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that suppresses noisy tracebacks on client disconnects."""

    request_queue_size = 256
    daemon_threads = True
    block_on_close = False

    def __init__(
        self,
        server_address: tuple[str, int],
        RequestHandlerClass: Any,
        bind_and_activate: bool = True,
        *,
        file_handle_cache_size: int = DEFAULT_FILE_HANDLE_CACHE_SIZE,
        max_concurrent_fds_per_file: int = DEFAULT_CONCURRENT_FD_POOL_SIZE,
        max_total_open_fds: int | None = DEFAULT_MAX_TOTAL_OPEN_FDS,
        resolved_path_cache_size: int = DEFAULT_RESOLVED_PATH_CACHE_SIZE,
        quiet: bool = False,
    ) -> None:
        if file_handle_cache_size < 0:
            raise ValueError("file_handle_cache_size must be >= 0")
        if max_concurrent_fds_per_file < 1:
            raise ValueError("max_concurrent_fds_per_file must be >= 1")
        if max_total_open_fds is not None and max_total_open_fds < 1:
            raise ValueError("max_total_open_fds must be >= 1")
        if resolved_path_cache_size < 0:
            raise ValueError("resolved_path_cache_size must be >= 0")
        self.quiet = bool(quiet)
        self.file_handle_cache: ReadOnlyFileHandleCache | None = (
            ReadOnlyFileHandleCache(
                max_entries=file_handle_cache_size,
                max_concurrent_fds_per_file=max_concurrent_fds_per_file,
                max_total_open_fds=max_total_open_fds,
            )
            if file_handle_cache_size > 0
            else None
        )
        self.resolved_path_cache: _ResolvedPathCache | None = (
            _ResolvedPathCache(max_entries=resolved_path_cache_size)
            if resolved_path_cache_size > 0
            else None
        )
        self.interstage_sidecar_cache: _InterstageSidecarCache = _InterstageSidecarCache(
            max_entries=DEFAULT_INTERSTAGE_CACHE_SIZE
        )
        super().__init__(
            server_address,
            RequestHandlerClass,
            bind_and_activate=bind_and_activate,
        )

    def server_close(self) -> None:
        cache = getattr(self, "file_handle_cache", None)
        if cache is not None:
            cache.close_all()
        path_cache = getattr(self, "resolved_path_cache", None)
        if path_cache is not None:
            path_cache.clear()
        interstage_cache = getattr(self, "interstage_sidecar_cache", None)
        if interstage_cache is not None:
            interstage_cache.clear()
        super().server_close()

    def handle_error(self, request: Any, client_address: Any) -> None:
        exc = sys.exc_info()[1]
        if _is_client_disconnect_error(exc):
            return
        with _LOG_LOCK:
            try:
                super().handle_error(request, client_address)
            except (OSError, ValueError):
                return


def create_server(
    host: str,
    port: int,
    web_root: Path,
    data_root: Path,
    *,
    game_root: Path | None = None,
    shaders_root: Path | None = None,
    boot_root: Path | None = None,
    dump_root: Path | None = None,
    max_dump_file_bytes: int = DEFAULT_MAX_DUMP_FILE_BYTES,
    keep_alive_timeout: float | None = None,
    stream_chunk_bytes: int | None = None,
    range_stream_chunk_bytes: int | None = None,
    file_handle_cache_size: int = DEFAULT_FILE_HANDLE_CACHE_SIZE,
    max_concurrent_fds_per_file: int = DEFAULT_CONCURRENT_FD_POOL_SIZE,
    max_total_open_fds: int | None = DEFAULT_MAX_TOTAL_OPEN_FDS,
    resolved_path_cache_size: int = DEFAULT_RESOLVED_PATH_CACHE_SIZE,
    quiet: bool = False,
) -> QuietThreadingHTTPServer:
    if max_dump_file_bytes < 0:
        raise ValueError("max_dump_file_bytes must be >= 0")
    if stream_chunk_bytes is not None and stream_chunk_bytes <= 0:
        raise ValueError("stream_chunk_bytes must be > 0")
    if range_stream_chunk_bytes is not None and range_stream_chunk_bytes <= 0:
        raise ValueError("range_stream_chunk_bytes must be > 0")
    if file_handle_cache_size < 0:
        raise ValueError("file_handle_cache_size must be >= 0")
    if max_concurrent_fds_per_file < 1:
        raise ValueError("max_concurrent_fds_per_file must be >= 1")
    if max_total_open_fds is not None and max_total_open_fds < 1:
        raise ValueError("max_total_open_fds must be >= 1")
    if resolved_path_cache_size < 0:
        raise ValueError("resolved_path_cache_size must be >= 0")
    effective_shaders_root = shaders_root if shaders_root is not None else (data_root / "shaders")
    resolved_roots = {
        "web": _try_resolve_root(web_root),
        "data": _try_resolve_root(data_root),
        "game": _try_resolve_root(game_root),
        "shaders": _try_resolve_root(effective_shaders_root),
        "boot": _try_resolve_root(boot_root),
    }
    handler = partial(
        PlayGta6Handler,
        web_root=web_root,
        data_root=data_root,
        game_root=game_root,
        shaders_root=shaders_root,
        boot_root=boot_root,
        dump_root=dump_root,
        max_dump_file_bytes=max_dump_file_bytes,
        keep_alive_timeout=keep_alive_timeout,
        stream_chunk_bytes=stream_chunk_bytes,
        range_stream_chunk_bytes=range_stream_chunk_bytes,
        quiet=quiet,
        _resolved_roots=resolved_roots,
    )
    return QuietThreadingHTTPServer(
        (host, port),
        handler,
        file_handle_cache_size=file_handle_cache_size,
        max_concurrent_fds_per_file=max_concurrent_fds_per_file,
        max_total_open_fds=max_total_open_fds,
        resolved_path_cache_size=resolved_path_cache_size,
        quiet=quiet,
    )


def _load_dotenv_value(repo_root: Path, key: str) -> str | None:
    env_file = repo_root / ".env"
    if not env_file.is_file():
        return None
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key:
            val = v.strip().strip('"').strip("'")
            return val or None
    return None


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    default_data_raw = os.environ.get("GTA6_DATA_DIR") or _load_dotenv_value(repo_root, "GTA6_DATA_DIR")
    default_data = Path(default_data_raw) if default_data_raw else (repo_root / "local-data")
    if not default_data.is_absolute():
        default_data = (repo_root / default_data).resolve()

    default_game_raw = os.environ.get("GTA6_GAME_DIR") or _load_dotenv_value(repo_root, "GTA6_GAME_DIR")
    default_game = Path(default_game_raw) if default_game_raw else None
    if default_game is not None and not default_game.is_absolute():
        default_game = (repo_root / default_game).resolve()

    default_work_raw = os.environ.get("GTA6_WORK_DIR") or _load_dotenv_value(repo_root, "GTA6_WORK_DIR")
    default_work = Path(default_work_raw) if default_work_raw else (repo_root / "work")
    if not default_work.is_absolute():
        default_work = (repo_root / default_work).resolve()

    parser = argparse.ArgumentParser(description="Local COOP/COEP + HTTP Range server for playgta6web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--web", type=Path, default=repo_root / "web")
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument(
        "--game",
        type=Path,
        default=default_game,
        help="Local Xbox Series game directory served at /game/ (from GTA6_GAME_DIR when set).",
    )
    parser.add_argument(
        "--shaders",
        type=Path,
        default=None,
        help="Shader pack directory served at /shaders/ (default: <data>/shaders).",
    )
    parser.add_argument(
        "--boot",
        type=Path,
        default=None,
        help="Boot-bundle directory served at /boot/ (default: <work>/recomp_build).",
    )
    parser.add_argument(
        "--work",
        type=Path,
        default=default_work,
        help="Work directory used to derive the default --boot path.",
    )
    parser.add_argument(
        "--allow-dump-dir",
        type=Path,
        default=None,
        help="Opt-in local directory enabling POST /__pg6_dump/<relpath> (disabled by default).",
    )
    parser.add_argument(
        "--max-dump-file-bytes",
        type=int,
        default=DEFAULT_MAX_DUMP_FILE_BYTES,
        help=f"Maximum allowed byte size per file written via POST /__pg6_dump/ (default: {DEFAULT_MAX_DUMP_FILE_BYTES}).",
    )
    parser.add_argument(
        "--range-chunk-bytes",
        type=int,
        default=None,
        help=(
            "Streaming chunk size in bytes for HTTP 206 Partial Content responses "
            f"(default: {RANGE_STREAM_CHUNK_BYTES})."
        ),
    )
    parser.add_argument(
        "--max-concurrent-fds",
        type=int,
        default=DEFAULT_CONCURRENT_FD_POOL_SIZE,
        help=(
            "Maximum concurrent read-only file descriptors pooled per cached file on "
            f"non-pread platforms (default: {DEFAULT_CONCURRENT_FD_POOL_SIZE})."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-request HTTP access logs on stderr.",
    )
    args = parser.parse_args()

    if args.max_dump_file_bytes < 0:
        parser.error("--max-dump-file-bytes must be >= 0")
    if args.range_chunk_bytes is not None and args.range_chunk_bytes <= 0:
        parser.error("--range-chunk-bytes must be > 0")
    if args.max_concurrent_fds < 1:
        parser.error("--max-concurrent-fds must be >= 1")

    args.data.mkdir(parents=True, exist_ok=True)
    shaders_root = args.shaders if args.shaders is not None else (args.data / "shaders")
    boot_root = args.boot if args.boot is not None else (args.work / "recomp_build")
    game_root = args.game if args.game is not None else None
    dump_root = args.allow_dump_dir.resolve() if args.allow_dump_dir is not None else None
    if dump_root is not None:
        dump_root.mkdir(parents=True, exist_ok=True)

    httpd = create_server(
        args.host,
        args.port,
        args.web,
        args.data,
        game_root=game_root,
        shaders_root=shaders_root,
        boot_root=boot_root,
        dump_root=dump_root,
        max_dump_file_bytes=args.max_dump_file_bytes,
        range_stream_chunk_bytes=args.range_chunk_bytes,
        max_concurrent_fds_per_file=args.max_concurrent_fds,
        quiet=args.quiet,
    )
    dump_summary = (
        f", /__pg6_dump={dump_root} (max={args.max_dump_file_bytes}B)"
        if dump_root is not None
        else ""
    )
    print(
        f"Serving web={args.web}, /data={args.data}, /shaders={shaders_root}, "
        f"/boot={boot_root}, /game={game_root}{dump_summary} at http://{args.host}:{args.port}/"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
