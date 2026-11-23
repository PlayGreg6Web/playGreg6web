#!/usr/bin/env python3
"""Local dev & asset-streaming HTTP server for playgta6web.

- Serves `web/` at `/` and `--data` (default `local-data/`) at `/data/`.
- Serves sanitized JSON readiness diagnostics at `/api/status`.
- Sends Cross-Origin Isolation headers (COOP/COEP/CORP) required for SharedArrayBuffer.
- Supports HTTP `Range` (`bytes=start-end`, `bytes=start-`, `bytes=-suffix`), `HEAD`,
  deterministic `ETag`/`Last-Modified`, `If-None-Match` (`304`), and `If-Range`.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import posixpath
import re
from datetime import timezone
from email.utils import formatdate, parsedate_to_datetime
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$", re.IGNORECASE)

mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/json", ".json")


def safe_resolve(root: Path, rel_url_path: str) -> Path | None:
    """Resolve a URL path strictly inside `root`, rejecting traversal."""
    normalized = posixpath.normpath("/" + unquote(rel_url_path)).lstrip("/")
    if not normalized or normalized.startswith(".."):
        return None
    candidate = (root / normalized).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
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


def _count_shader_packs(web_root: Path, data_root: Path) -> int:
    """Count `.wgsl.pack` bundles across expected shallow directories."""
    candidate_dirs = [
        data_root,
        data_root / "shaders",
        web_root / "dist",
        web_root / "shaders",
    ]
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


def build_status_payload(web_root: Path, data_root: Path) -> dict[str, bool | int]:
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

    return {
        "ok": True,
        "coopCoep": True,
        "dataDirConfigured": data_configured,
        "hasManifest": has_manifest,
        "hasBootset": has_bootset,
        "hasWasmBinary": has_wasm,
        "shaderPackCount": _count_shader_packs(web_root, data_root),
    }


class PlayGta6Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, web_root: Path, data_root: Path, **kwargs):
        self.web_root = web_root
        self.data_root = data_root
        super().__init__(*args, **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def _resolve_target(self, url_path: str) -> Path | None:
        if url_path == "/":
            return (self.web_root / "index.html").resolve()
        if url_path.startswith("/data/"):
            return safe_resolve(self.data_root, url_path[len("/data/"):])
        return safe_resolve(self.web_root, url_path.lstrip("/"))

    def _serve_status(self, send_body: bool) -> None:
        payload = build_status_payload(self.web_root, self.data_root)
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if send_body:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _serve_file(self, url_path: str, send_body: bool) -> None:
        target = self._resolve_target(url_path)
        if target is None or not target.is_file():
            self.send_response(HTTPStatus.NOT_FOUND)
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return

        try:
            st = target.stat()
        except OSError:
            self.send_response(HTTPStatus.NOT_FOUND)
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
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
        except (BrokenPipeError, ConnectionResetError):
            pass

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


def create_server(host: str, port: int, web_root: Path, data_root: Path) -> ThreadingHTTPServer:
    handler = partial(PlayGta6Handler, web_root=web_root, data_root=data_root)
    return ThreadingHTTPServer((host, port), handler)


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

    parser = argparse.ArgumentParser(description="Local COOP/COEP + HTTP Range server for playgta6web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--web", type=Path, default=repo_root / "web")
    parser.add_argument("--data", type=Path, default=default_data)
    args = parser.parse_args()

    args.data.mkdir(parents=True, exist_ok=True)
    httpd = create_server(args.host, args.port, args.web, args.data)
    print(f"Serving web={args.web} and /data={args.data} at http://{args.host}:{args.port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
