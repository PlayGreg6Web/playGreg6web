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
- Uses HTTP/1.1 persistent keep-alive connections with large streaming buffers (1 MiB).
- Supports HTTP `Range` (`bytes=start-end`, `bytes=start-`, `bytes=-suffix`), `HEAD`,
  `OPTIONS`, `Cache-Control: no-cache`, deterministic `ETag`/`Last-Modified`,
  `If-None-Match` (`304`), and `If-Range`.
"""

from __future__ import annotations

import argparse
import importlib.util
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
from datetime import timezone
from email.utils import formatdate, parsedate_to_datetime
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$", re.IGNORECASE)

BOOT_ALLOWED_SUFFIXES = (".pgim", ".json", ".wasm", ".js")
RUNTIME_WASM_NAME = "runtime.wasm"
PREFERRED_SHADER_PACK_NAMES = ("splash.wgsl.pack", "fallback.wgsl.pack")
SAFE_SIDE_MODULE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.wasm$", re.IGNORECASE)

DUMP_URL_PREFIX = "/__pg6_dump/"
DEFAULT_MAX_DUMP_FILE_BYTES = 64 * 1024 * 1024  # 64 MiB default cap per dumped file
_DUMP_WRITE_LOCK = threading.Lock()

STREAM_CHUNK_BYTES = 1024 * 1024  # 1 MiB streaming buffer for large .pgim/.wasm/Range reads
STREAM_CHUNK_SIZE = STREAM_CHUNK_BYTES  # Backwards-compatible alias
SOCKET_SNDBUF_BYTES = 1024 * 1024  # 1 MiB TCP send buffer hint for loopback Range throughput
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

mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/json", ".json")


def safe_resolve(root: Path, rel_url_path: str) -> Path | None:
    """Resolve a URL path strictly inside `root`, rejecting traversal and external symlinks."""
    decoded = unquote(rel_url_path)
    if "\x00" in decoded:
        return None
    unslashed = decoded.replace("\\", "/")
    if ".." in unslashed.split("/"):
        return None
    normalized = posixpath.normpath("/" + unslashed).lstrip("/")
    if not normalized or normalized.startswith(".."):
        return None
    try:
        root_resolved = root.resolve()
        candidate = (root / normalized).resolve()
        candidate.relative_to(root_resolved)
    except (OSError, ValueError):
        return None
    return candidate


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


def parse_byte_range(header: str, file_size: int) -> tuple[int, int] | None:
    """Parse a single HTTP byte range against `file_size`."""
    match = RANGE_RE.match(header.strip())
    if not match or file_size <= 0:
        return None
    start_s, end_s = match.groups()
    if not start_s and not end_s:
        return None
    if start_s and end_s:
        start, end = int(start_s), min(int(end_s), file_size - 1)
    elif start_s:
        start, end = int(start_s), file_size - 1
    else:
        suffix = int(end_s)
        if suffix <= 0:
            return None
        start, end = max(file_size - suffix, 0), file_size - 1
    if start > end or start >= file_size:
        return None
    return start, end


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
    """Return a reusable per-thread `bytearray` of `chunk_bytes` for zero-alloc streaming."""
    buf: bytearray | None = getattr(_STREAM_BUFFERS, "buf", None)
    if buf is None or len(buf) != chunk_bytes:
        buf = bytearray(chunk_bytes)
        _STREAM_BUFFERS.buf = buf
    return buf


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
        self._headers_sent = False
        self._connection_header_sent = False
        self._client_requested_close = False
        super().__init__(*args, **kwargs)

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

    def end_headers(self) -> None:
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
        self._headers_sent = True
        super().end_headers()

    def send_error(
        self,
        code: int,
        message: str | None = None,
        explain: str | None = None,
    ) -> None:
        self.close_connection = True
        try:
            super().send_error(code, message, explain)
        except CLIENT_DISCONNECT_ERRORS:
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
            self.close_connection = True
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except CLIENT_DISCONNECT_ERRORS:
            self.close_connection = True
            return
        except Exception:
            self.close_connection = True
            raise

    def _send_method_not_allowed(self) -> None:
        # Close the connection because unsupported methods (e.g. POST/PUT) may
        # carry an unread request body that would desynchronize keep-alive.
        self.close_connection = True
        self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
        self.send_header("Allow", ALLOWED_METHODS)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

    def handle_one_request(self) -> None:
        """Handle a single HTTP request with quiet disconnect/timeout handling."""
        self._headers_sent = False
        self._connection_header_sent = False
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
        except CLIENT_DISCONNECT_ERRORS:
            self.close_connection = True
            return
        except Exception:
            self.close_connection = True
            if not self._headers_sent:
                try:
                    self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR)
                    self.wfile.flush()
                except CLIENT_DISCONNECT_ERRORS:
                    pass
            raise

    def _resolve_target(self, url_path: str) -> Path | None | tuple[None, HTTPStatus]:
        """Resolve a URL path to a filesystem target.

        Returns:
            Path on success, None for a generic 404, or (None, status) for an
            explicit error status (e.g. 403 for disallowed `/boot/` extensions).
        """
        if url_path == "/":
            return (self.web_root / "index.html").resolve()
        if url_path.startswith("/data/"):
            return safe_resolve(self.data_root, url_path[len("/data/") :])
        if url_path.startswith("/game/"):
            if self.game_root is None:
                return None
            return safe_resolve(self.game_root, url_path[len("/game/") :])
        if url_path.startswith("/shaders/"):
            if self.shaders_root is None:
                return None
            return safe_resolve(self.shaders_root, url_path[len("/shaders/") :])
        if url_path.startswith("/boot/"):
            if self.boot_root is None:
                return None
            rel = url_path[len("/boot/") :]
            # Reject traversal before extension checks so `../secret.exe` cannot
            # short-circuit the allowlist via a lucky extension after normpath.
            if ".." in unquote(rel).replace("\\", "/").split("/"):
                return None
            target = safe_resolve(self.boot_root, rel)
            if target is None:
                return None
            if not _boot_extension_allowed(target):
                return (None, HTTPStatus.FORBIDDEN)
            return target
        return safe_resolve(self.web_root, url_path.lstrip("/"))

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

    def _send_not_modified(self, etag: str, last_modified: str) -> None:
        self.send_response(HTTPStatus.NOT_MODIFIED)
        self.send_header("ETag", etag)
        self.send_header("Last-Modified", last_modified)
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

    def _stream_file_range(self, f: Any, length: int) -> None:
        chunk_bytes = max(1, self._get_stream_chunk_bytes())
        readinto = getattr(f, "readinto", None)
        use_fast_buffer = callable(readinto) and hasattr(self.wfile, "_sock")

        remaining = length
        if use_fast_buffer:
            buf = _get_thread_stream_buffer(chunk_bytes)
            mv = memoryview(buf)
            while remaining > 0:
                to_read = chunk_bytes if remaining > chunk_bytes else remaining
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
            chunk = f.read(min(chunk_bytes, remaining))
            if not chunk:
                # File shrank mid-stream after Content-Length was emitted:
                # force socket close so HTTP/1.1 framing cannot desynchronize.
                self.close_connection = True
                break
            remaining -= len(chunk)
            if remaining == 0:
                # Release the file before the last bytes reach the client: once the
                # response is complete the file can be replaced or deleted (Windows
                # refuses to delete a file that is still open).
                f.close()
            self.wfile.write(chunk)

    def _serve_file(self, url_path: str, send_body: bool) -> None:
        resolved = self._resolve_target(url_path)
        if isinstance(resolved, tuple):
            self._send_empty(resolved[1])
            return
        target = resolved
        if target is None:
            self._send_empty(HTTPStatus.NOT_FOUND)
            return

        try:
            st = target.stat()
        except OSError:
            self._send_empty(HTTPStatus.NOT_FOUND)
            return
        if not stat.S_ISREG(st.st_mode):
            self._send_empty(HTTPStatus.NOT_FOUND)
            return

        file_size = st.st_size
        etag = compute_etag(st)
        last_modified = formatdate(st.st_mtime, usegmt=True)
        range_header = self.headers.get("Range")

        # Never short-circuit HTTP Range requests to 304 Not Modified: browsers
        # attach cached If-None-Match / If-Modified-Since headers when fetching
        # subsequent slices of large .rpf / .wgsl.pack files, and returning 304
        # starves or corrupts ranged streaming (RFC 9110 §13.1.5 uses If-Range).
        if not range_header:
            if hasattr(self.headers, "get_all"):
                raw_if_none_match = self.headers.get_all("If-None-Match")
                if_none_match = ", ".join(raw_if_none_match) if raw_if_none_match else None
            else:
                if_none_match = self.headers.get("If-None-Match")
            if if_none_match:
                if etag_matches_if_none_match(if_none_match, etag):
                    self._send_not_modified(etag, last_modified)
                    return
            else:
                if_modified_since = self.headers.get("If-Modified-Since")
                if if_modified_since and if_modified_since_not_modified(if_modified_since, st.st_mtime):
                    self._send_not_modified(etag, last_modified)
                    return

        content_type, _ = mimetypes.guess_type(str(target))
        content_type = content_type or "application/octet-stream"

        if_range_header = self.headers.get("If-Range")
        apply_range = bool(range_header)
        if apply_range and if_range_header:
            if not if_range_matches(if_range_header, etag, last_modified, st.st_mtime):
                apply_range = False

        start, end = 0, max(file_size - 1, 0)
        status = HTTPStatus.OK

        if apply_range and range_header:
            parsed = parse_byte_range(range_header, file_size)
            if parsed is None:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.send_header("Content-Length", "0")
                self.send_header("ETag", etag)
                self.send_header("Last-Modified", last_modified)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            start, end = parsed
            status = HTTPStatus.PARTIAL_CONTENT

        length = (end - start + 1) if file_size > 0 else 0

        def send_file_headers() -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("ETag", etag)
            self.send_header("Last-Modified", last_modified)
            self.send_header("Cache-Control", "no-cache")
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.end_headers()

        if not send_body or length == 0:
            send_file_headers()
            return

        try:
            f = target.open("rb")
        except OSError:
            self._send_empty(HTTPStatus.NOT_FOUND)
            return

        try:
            with f:
                if start > 0:
                    f.seek(start)
                send_file_headers()
                self._stream_file_range(f, length)
        except CLIENT_DISCONNECT_ERRORS:
            self.close_connection = True
            return
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
            self._send_json(
                HTTPStatus.FORBIDDEN,
                {"ok": False, "error": "dump endpoint disabled"},
                close=True,
            )
            return

        rel_url_path = parsed_url.path[len(DUMP_URL_PREFIX) :]
        resolved, err_status, err_msg = resolve_dump_target(self.dump_root, rel_url_path)
        if resolved is None:
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
                        self._send_json(
                            HTTPStatus.BAD_REQUEST,
                            {"ok": False, "error": "incomplete request body"},
                            close=True,
                        )
                        return
                    bytes_written += len(chunk)
                    if bytes_written > max_bytes:
                        self._send_json(
                            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                            {"ok": False, "error": "payload exceeds max dump file size"},
                            close=True,
                        )
                        return
                    tmp_file.write(chunk)
                    remaining -= len(chunk)

            # Request body is completely consumed; keep-alive is safe if the client wants it.
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

    block_on_close = False

    def handle_error(self, request: Any, client_address: Any) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, CLIENT_DISCONNECT_ERRORS):
            return
        super().handle_error(request, client_address)


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
) -> QuietThreadingHTTPServer:
    if max_dump_file_bytes < 0:
        raise ValueError("max_dump_file_bytes must be >= 0")
    if stream_chunk_bytes is not None and stream_chunk_bytes <= 0:
        raise ValueError("stream_chunk_bytes must be > 0")
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
    )
    return QuietThreadingHTTPServer((host, port), handler)


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
    args = parser.parse_args()

    if args.max_dump_file_bytes < 0:
        parser.error("--max-dump-file-bytes must be >= 0")

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
