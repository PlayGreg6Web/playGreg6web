#!/usr/bin/env python3
"""Local dev & asset-streaming HTTP server for playgta6web.

- Serves `web/` at `/` and `--data` (default `local-data/`) at `/data/`.
- Serves `GTA6_GAME_DIR` (when configured) at `/game/` for local game & splash assets.
- Serves `local-data/shaders/` at `/shaders/` and `work/recomp_build/` at `/boot/`
  (boot allowlisted to `.pgim`/`.json`/`.wasm`/`.js` only).
- Serves sanitized JSON readiness diagnostics at `/api/status`.
- Sends Cross-Origin Isolation headers (COOP/COEP/CORP) required for SharedArrayBuffer.
- Supports HTTP `Range` (`bytes=start-end`, `bytes=start-`, `bytes=-suffix`), `HEAD`,
  deterministic `ETag`/`Last-Modified`, `If-None-Match` (`304`), and `If-Range`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import mimetypes
import os
import posixpath
import re
import sys
from datetime import timezone
from email.utils import formatdate, parsedate_to_datetime
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$", re.IGNORECASE)

BOOT_ALLOWED_SUFFIXES = (".pgim", ".json", ".wasm", ".js")

CLIENT_DISCONNECT_ERRORS = (
    BrokenPipeError,
    ConnectionResetError,
    ConnectionAbortedError,
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
    names: list[str] = []
    try:
        for entry in shaders_root.iterdir():
            try:
                if entry.is_file() and entry.name.lower().endswith(".wgsl.pack"):
                    # Ensure the file stays inside shaders_root (reject external symlinks).
                    resolved = entry.resolve()
                    if resolved.is_relative_to(shaders_root.resolve()):
                        names.append(entry.name)
            except OSError:
                continue
    except OSError:
        return []
    return sorted(names, key=lambda name: name.lower())


def _discover_shader_pack_url(shaders_root: Path | None) -> str | None:
    names = _list_shader_pack_names(shaders_root)
    if not names:
        return None
    preferred = next((name for name in names if name.lower() == "fallback.wgsl.pack"), None)
    chosen = preferred or names[0]
    return f"/shaders/{chosen}"


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


def build_status_payload(
    web_root: Path,
    data_root: Path,
    *,
    game_root: Path | None = None,
    shaders_root: Path | None = None,
    boot_root: Path | None = None,
) -> dict[str, bool | int | str | None]:
    """Build sanitized readiness metadata with zero host filesystem paths."""
    data_configured = data_root.is_dir()
    has_manifest = data_configured and (data_root / "data-manifest.json").is_file()
    has_bootset = data_configured and (data_root / "bootset.json").is_file()

    dist_dir = web_root / "dist"
    has_wasm = (dist_dir / "runtime.wasm").is_file()
    if not has_wasm and dist_dir.is_dir():
        try:
            has_wasm = any(
                entry.is_file() and entry.suffix.lower() == ".wasm"
                for entry in dist_dir.iterdir()
            )
        except OSError:
            has_wasm = False

    effective_shaders = shaders_root if shaders_root is not None else (data_root / "shaders")

    return {
        "ok": True,
        "coopCoep": True,
        "dataDirConfigured": data_configured,
        "hasManifest": has_manifest,
        "hasBootset": has_bootset,
        "hasWasmBinary": has_wasm,
        "shaderPackCount": _count_shader_packs(web_root, data_root, effective_shaders),
        "splashImage": _discover_splash_image_url(game_root),
        "shaderPackUrl": _discover_shader_pack_url(effective_shaders),
        "reflectionManifestUrl": _discover_reflection_manifest_url(effective_shaders),
        "pgimUrl": _discover_pgim_url(boot_root),
        "programManifestUrl": _discover_program_manifest_url(boot_root),
        "gameBase": "/game/" if (game_root is not None and game_root.is_dir()) else None,
    }


class PlayGta6Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(
        self,
        *args,
        web_root: Path,
        data_root: Path,
        game_root: Path | None = None,
        shaders_root: Path | None = None,
        boot_root: Path | None = None,
        **kwargs,
    ):
        self.web_root = web_root
        self.data_root = data_root
        self.game_root = game_root
        self.shaders_root = shaders_root if shaders_root is not None else (data_root / "shaders")
        self.boot_root = boot_root
        super().__init__(*args, **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def _send_empty(self, status: HTTPStatus) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

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
                return

    def _serve_file(self, url_path: str, send_body: bool) -> None:
        resolved = self._resolve_target(url_path)
        if isinstance(resolved, tuple):
            self._send_empty(resolved[1])
            return
        target = resolved
        if target is None or not target.is_file():
            self._send_empty(HTTPStatus.NOT_FOUND)
            return

        try:
            st = target.stat()
        except OSError:
            self._send_empty(HTTPStatus.NOT_FOUND)
            return

        file_size = st.st_size
        etag = compute_etag(st)
        last_modified = formatdate(st.st_mtime, usegmt=True)

        if_none_match = self.headers.get("If-None-Match")
        if if_none_match and etag_matches_if_none_match(if_none_match, etag):
            self.send_response(HTTPStatus.NOT_MODIFIED)
            self.send_header("ETag", etag)
            self.send_header("Last-Modified", last_modified)
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            return

        content_type, _ = mimetypes.guess_type(str(target))
        content_type = content_type or "application/octet-stream"

        range_header = self.headers.get("Range")
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
            return

        try:
            with target.open("rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(256 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except CLIENT_DISCONNECT_ERRORS:
            return

    def _dispatch(self, send_body: bool) -> None:
        url_path = urlsplit(self.path).path
        if url_path == "/api/status":
            self._serve_status(send_body=send_body)
            return
        self._serve_file(url_path, send_body=send_body)

    def do_GET(self) -> None:
        self._dispatch(send_body=True)

    def do_HEAD(self) -> None:
        self._dispatch(send_body=False)


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that suppresses noisy tracebacks on client disconnects."""

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
) -> QuietThreadingHTTPServer:
    handler = partial(
        PlayGta6Handler,
        web_root=web_root,
        data_root=data_root,
        game_root=game_root,
        shaders_root=shaders_root,
        boot_root=boot_root,
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
    args = parser.parse_args()

    args.data.mkdir(parents=True, exist_ok=True)
    shaders_root = args.shaders if args.shaders is not None else (args.data / "shaders")
    boot_root = args.boot if args.boot is not None else (args.work / "recomp_build")
    game_root = args.game if args.game is not None else None

    httpd = create_server(
        args.host,
        args.port,
        args.web,
        args.data,
        game_root=game_root,
        shaders_root=shaders_root,
        boot_root=boot_root,
    )
    print(
        f"Serving web={args.web}, /data={args.data}, /shaders={shaders_root}, "
        f"/boot={boot_root}, /game={game_root} at http://{args.host}:{args.port}/"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
