import concurrent.futures
import errno
from functools import partial
import hashlib
import http.client
import io
import json
import math
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "server"))

import serve_local  # noqa: E402


def _make_deterministic_bytes(size: int) -> bytes:
    """Create a deterministic multi-MiB byte pattern with unique 4-byte page indices."""
    block = bytearray(size)
    page = 4096
    for offset in range(0, size, page):
        end = min(offset + page, size)
        tag = (offset // page).to_bytes(4, "little")
        chunk_len = end - offset
        repeats = (chunk_len // 4) + 1
        block[offset:end] = (tag * repeats)[:chunk_len]
    return bytes(block)


class TestServeLocalRangeFastpath(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.web_root = base / "web"
        self.data_root = base / "data"
        self.dump_root = base / "dumps"
        self.web_root.mkdir()
        self.data_root.mkdir()
        self.dump_root.mkdir()

        (self.web_root / "index.html").write_text("<h1>playgta6web</h1>", encoding="utf-8")
        (self.web_root / "runtime.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00")

        # 3.5 MiB deterministic payload spanning >3x 1 MiB STREAM_CHUNK_BYTES buffers.
        self.large_size = 3 * 1024 * 1024 + 512 * 1024
        self.large_payload = _make_deterministic_bytes(self.large_size)
        self.large_file = self.data_root / "splash.wgsl.pack"
        self.large_file.write_bytes(self.large_payload)

        self.httpd = serve_local.create_server(
            "127.0.0.1",
            0,
            self.web_root,
            self.data_root,
        )
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)
        self.tmp.cleanup()

    def _request(
        self,
        method: str,
        path: str,
        headers: dict[str, str] | None = None,
        *,
        port: int | None = None,
        body: bytes | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", port or self.port, timeout=10)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            resp = conn.getresponse()
            data = resp.read()
            return resp.status, dict(resp.getheaders()), data
        finally:
            conn.close()

    def _assert_coop_coep(self, headers: dict[str, str]) -> None:
        self.assertEqual(headers.get("Cross-Origin-Opener-Policy"), "same-origin")
        self.assertEqual(headers.get("Cross-Origin-Embedder-Policy"), "require-corp")
        self.assertEqual(headers.get("Cross-Origin-Resource-Policy"), "same-origin")
        self.assertEqual(headers.get("Accept-Ranges"), "bytes")

    def test_stream_chunk_bytes_constant_is_1mib_and_validated(self) -> None:
        self.assertEqual(serve_local.STREAM_CHUNK_BYTES, 1024 * 1024)
        self.assertEqual(serve_local.STREAM_CHUNK_SIZE, serve_local.STREAM_CHUNK_BYTES)
        self.assertGreaterEqual(serve_local.SOCKET_SNDBUF_BYTES, 1024 * 1024)
        self.assertTrue(serve_local.PlayGta6Handler.disable_nagle_algorithm)

        with self.assertRaises(ValueError):
            serve_local.create_server(
                "127.0.0.1",
                0,
                self.web_root,
                self.data_root,
                stream_chunk_bytes=0,
            )
        with self.assertRaises(ValueError):
            serve_local.create_server(
                "127.0.0.1",
                0,
                self.web_root,
                self.data_root,
                stream_chunk_bytes=-4096,
            )

    def test_client_socket_sets_tcp_nodelay(self) -> None:
        observed: list[tuple[int, int]] = []

        class _SocketProbeHandler(serve_local.PlayGta6Handler):
            def log_message(self, format: str, *args) -> None:
                return None

            def do_GET(self) -> None:
                nodelay = self.connection.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
                sndbuf = self.connection.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
                observed.append((nodelay, sndbuf))
                super().do_GET()

        handler = partial(
            _SocketProbeHandler,
            web_root=self.web_root,
            data_root=self.data_root,
        )
        httpd = serve_local.QuietThreadingHTTPServer(("127.0.0.1", 0), handler)
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            status, headers, body = self._request("GET", "/runtime.wasm", port=port)
            self.assertEqual(status, 200)
            self._assert_coop_coep(headers)
            self.assertEqual(body, b"\x00asm\x01\x00\x00\x00")
            self.assertEqual(len(observed), 1)
            nodelay, sndbuf = observed[0]
            self.assertNotEqual(nodelay, 0, "TCP_NODELAY must be enabled on client sockets")
            self.assertGreater(sndbuf, 0)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=2)

    def test_etag_if_none_match_and_if_modified_since_304_preserve_coop_coep(self) -> None:
        st, hdr, body = self._request("GET", "/runtime.wasm")
        self.assertEqual(st, 200)
        self._assert_coop_coep(hdr)
        self.assertEqual(hdr.get("Cache-Control"), "no-cache")
        etag = hdr.get("ETag")
        last_modified = hdr.get("Last-Modified")
        self.assertTrue(etag)
        self.assertTrue(last_modified)
        self.assertEqual(body, b"\x00asm\x01\x00\x00\x00")

        # GET and HEAD with exact, weak-prefixed, list, and wildcard If-None-Match -> 304 + COOP/COEP
        for method in ("GET", "HEAD"):
            for inm in (etag, f"W/{etag}", f'"other", {etag}', "*"):
                with self.subTest(method=method, if_none_match=inm):
                    st304, hdr304, body304 = self._request(
                        method, "/runtime.wasm", {"If-None-Match": inm}
                    )
                    self.assertEqual(st304, 304)
                    self._assert_coop_coep(hdr304)
                    self.assertEqual(hdr304.get("ETag"), etag)
                    self.assertEqual(hdr304.get("Last-Modified"), last_modified)
                    self.assertEqual(hdr304.get("Cache-Control"), "no-cache")
                    self.assertEqual(body304, b"")

        # If-Modified-Since without If-None-Match -> 304 with COOP/COEP
        for method in ("GET", "HEAD"):
            with self.subTest(method=method, conditional="If-Modified-Since"):
                st_ims, hdr_ims, body_ims = self._request(
                    method, "/runtime.wasm", {"If-Modified-Since": last_modified}
                )
                self.assertEqual(st_ims, 304)
                self._assert_coop_coep(hdr_ims)
                self.assertEqual(hdr_ims.get("ETag"), etag)
                self.assertEqual(hdr_ims.get("Cache-Control"), "no-cache")
                self.assertEqual(body_ims, b"")

        # Per RFC 9110, If-None-Match takes precedence over If-Modified-Since:
        # stale If-None-Match must return 200 even if If-Modified-Since matches.
        st_prec, hdr_prec, body_prec = self._request(
            "GET",
            "/runtime.wasm",
            {"If-None-Match": '"stale-etag"', "If-Modified-Since": last_modified},
        )
        self.assertEqual(st_prec, 200)
        self._assert_coop_coep(hdr_prec)
        self.assertEqual(body_prec, b"\x00asm\x01\x00\x00\x00")

        # Same byte length with bumped mtime_ns invalidates ETag -> 200 OK.
        wasm_path = self.web_root / "runtime.wasm"
        prev_stat = wasm_path.stat()
        updated_bytes = b"\x00asm\x01\x00\x00\x01"
        wasm_path.write_bytes(updated_bytes)
        os.utime(wasm_path, ns=(prev_stat.st_atime_ns, prev_stat.st_mtime_ns + 2_000_000_000))
        st_new, hdr_new, body_new = self._request(
            "GET", "/runtime.wasm", {"If-None-Match": etag}
        )
        self.assertEqual(st_new, 200)
        self._assert_coop_coep(hdr_new)
        self.assertNotEqual(hdr_new.get("ETag"), etag)
        self.assertEqual(body_new, updated_bytes)

    def test_multi_mib_206_partial_content_streaming_fidelity_and_keep_alive(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        try:
            # 1. Multi-MiB interior Range spanning >2x STREAM_CHUNK_BYTES (2.5 MiB)
            start = 131_072
            length = 2 * 1024 * 1024 + 512 * 1024
            end = start + length - 1
            conn.request("GET", "/data/splash.wgsl.pack", headers={"Range": f"bytes={start}-{end}"})
            resp1 = conn.getresponse()
            body1 = resp1.read()
            hdr1 = dict(resp1.getheaders())
            first_sock = conn.sock

            self.assertEqual(resp1.status, 206)
            self.assertFalse(resp1.will_close)
            self._assert_coop_coep(hdr1)
            self.assertEqual(hdr1.get("Cache-Control"), "no-cache")
            self.assertTrue(hdr1.get("ETag"))
            self.assertEqual(hdr1.get("Content-Length"), str(length))
            self.assertEqual(hdr1.get("Content-Range"), f"bytes {start}-{end}/{self.large_size}")
            self.assertEqual(
                hashlib.sha256(body1).hexdigest(),
                hashlib.sha256(self.large_payload[start : end + 1]).hexdigest(),
            )
            self.assertEqual(body1, self.large_payload[start : end + 1])

            # 2. Suffix Range spanning >1 MiB (1.5 MiB) on same keep-alive socket
            suffix_len = 1 * 1024 * 1024 + 512 * 1024
            conn.request("GET", "/data/splash.wgsl.pack", headers={"Range": f"bytes=-{suffix_len}"})
            resp2 = conn.getresponse()
            body2 = resp2.read()
            hdr2 = dict(resp2.getheaders())
            self.assertIs(conn.sock, first_sock)
            self.assertEqual(resp2.status, 206)
            self._assert_coop_coep(hdr2)
            self.assertEqual(
                hdr2.get("Content-Range"),
                f"bytes {self.large_size - suffix_len}-{self.large_size - 1}/{self.large_size}",
            )
            self.assertEqual(body2, self.large_payload[-suffix_len:])

            # 3. Open-ended Range spanning >2 MiB on same keep-alive socket
            open_start = 1024 * 1024
            conn.request("GET", "/data/splash.wgsl.pack", headers={"Range": f"bytes={open_start}-"})
            resp3 = conn.getresponse()
            body3 = resp3.read()
            hdr3 = dict(resp3.getheaders())
            self.assertIs(conn.sock, first_sock)
            self.assertEqual(resp3.status, 206)
            self._assert_coop_coep(hdr3)
            self.assertEqual(
                hdr3.get("Content-Range"),
                f"bytes {open_start}-{self.large_size - 1}/{self.large_size}",
            )
            self.assertEqual(body3, self.large_payload[open_start:])

            # 4. Full 3.5 MiB 200 OK GET on the same socket
            conn.request("GET", "/data/splash.wgsl.pack")
            resp4 = conn.getresponse()
            body4 = resp4.read()
            hdr4 = dict(resp4.getheaders())
            self.assertIs(conn.sock, first_sock)
            self.assertEqual(resp4.status, 200)
            self._assert_coop_coep(hdr4)
            self.assertEqual(hdr4.get("Content-Length"), str(self.large_size))
            self.assertEqual(
                hashlib.sha256(body4).hexdigest(),
                hashlib.sha256(self.large_payload).hexdigest(),
            )
        finally:
            conn.close()

    def test_concurrent_multi_mib_range_reads_thread_local_buffer_isolation(self) -> None:
        ranges = [
            (0, 1_572_863),
            (524_288, 2_097_151),
            (1_048_576, 2_621_439),
            (1_572_864, 3_145_727),
            (2_097_152, self.large_size - 1),
            (65_536, 2_228_223),
        ]

        def fetch_range(bounds: tuple[int, int]) -> tuple[int, int, str, str]:
            lo, hi = bounds
            st, hdr, body = self._request(
                "GET", "/data/splash.wgsl.pack", {"Range": f"bytes={lo}-{hi}"}
            )
            self.assertEqual(st, 206)
            self._assert_coop_coep(hdr)
            self.assertEqual(hdr.get("Content-Range"), f"bytes {lo}-{hi}/{self.large_size}")
            return (
                lo,
                hi,
                hashlib.sha256(body).hexdigest(),
                hashlib.sha256(self.large_payload[lo : hi + 1]).hexdigest(),
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(fetch_range, ranges * 2))

        for lo, hi, actual_sha, expected_sha in results:
            self.assertEqual(actual_sha, expected_sha, msg=f"Range {lo}-{hi} hash mismatch")

    def test_custom_stream_chunk_bytes_and_fastpath_truncation(self) -> None:
        httpd = serve_local.create_server(
            "127.0.0.1",
            0,
            self.web_root,
            self.data_root,
            stream_chunk_bytes=64 * 1024,
        )
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            lo, hi = 10_000, 300_000
            st, hdr, body = self._request(
                "GET",
                "/data/splash.wgsl.pack",
                {"Range": f"bytes={lo}-{hi}"},
                port=port,
            )
            self.assertEqual(st, 206)
            self._assert_coop_coep(hdr)
            self.assertEqual(body, self.large_payload[lo : hi + 1])
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=2)

        # Verify readinto fastpath marks close_connection on mid-stream truncation.
        class _TruncatingReadintoFile:
            def __init__(self) -> None:
                self._calls = 0

            def readinto(self, mv: memoryview) -> int:
                self._calls += 1
                if self._calls == 1:
                    data = b"ABCD"
                    mv[: len(data)] = data
                    return len(data)
                return 0

            def close(self) -> None:
                return None

        sent = bytearray()

        class _FakeSocketWriter:
            _sock = object()

            def write(self, data) -> int:
                sent.extend(data)
                return len(data)

        handler = object.__new__(serve_local.PlayGta6Handler)
        handler._stream_chunk_bytes = 16
        handler.close_connection = False
        handler.wfile = _FakeSocketWriter()
        handler._stream_file_range(_TruncatingReadintoFile(), length=32)
        self.assertEqual(bytes(sent), b"ABCD")
        self.assertTrue(handler.close_connection)

    def test_traversal_and_dump_endpoint_preserved(self) -> None:
        st_trav, hdr_trav, _ = self._request("GET", "/data/../../README.md")
        self.assertEqual(st_trav, 404)
        self._assert_coop_coep(hdr_trav)

        # Disabled dump endpoint still returns 403 JSON with COOP/COEP intact.
        st_dis, hdr_dis, body_dis = self._request(
            "POST", "/__pg6_dump/test.bin", body=b"abc"
        )
        self.assertEqual(st_dis, 403)
        self._assert_coop_coep(hdr_dis)
        self.assertFalse(json.loads(body_dis.decode("utf-8"))["ok"])

        # Enabled dump endpoint writes cleanly with COOP/COEP intact.
        httpd = serve_local.create_server(
            "127.0.0.1",
            0,
            self.web_root,
            self.data_root,
            dump_root=self.dump_root,
        )
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            payload = b"DXBC_FASTPATH_CHECK"
            st_ok, hdr_ok, body_ok = self._request(
                "POST", "/__pg6_dump/sub/pso.bin", port=port, body=payload
            )
            self.assertEqual(st_ok, 200)
            self._assert_coop_coep(hdr_ok)
            self.assertEqual(json.loads(body_ok.decode("utf-8"))["bytes_written"], len(payload))
            self.assertEqual((self.dump_root / "sub" / "pso.bin").read_bytes(), payload)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=2)

    def test_range_request_with_matching_if_none_match_or_ims_returns_206_not_304(self) -> None:
        st0, hdr0, _ = self._request("HEAD", "/data/splash.wgsl.pack")
        self.assertEqual(st0, 200)
        etag = hdr0.get("ETag")
        last_modified = hdr0.get("Last-Modified")
        self.assertTrue(etag)
        self.assertTrue(last_modified)

        start, end = 4096, 8191
        expected = self.large_payload[start : end + 1]
        for extra_headers in (
            {"Range": f"bytes={start}-{end}", "If-None-Match": etag},
            {"Range": f"bytes={start}-{end}", "If-Modified-Since": last_modified},
            {
                "Range": f"bytes={start}-{end}",
                "If-None-Match": etag,
                "If-Modified-Since": last_modified,
            },
        ):
            with self.subTest(headers=sorted(extra_headers.keys())):
                st_r, hdr_r, body_r = self._request("GET", "/data/splash.wgsl.pack", extra_headers)
                self.assertEqual(st_r, 206)
                self._assert_coop_coep(hdr_r)
                self.assertEqual(hdr_r.get("Content-Range"), f"bytes {start}-{end}/{self.large_size}")
                self.assertEqual(body_r, expected)

    def test_http11_keep_alive_sequential_range_get_head_404_status_and_conditionals(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            def send(method: str, path: str, headers: dict[str, str] | None = None):
                conn.request(method, path, headers=headers or {})
                resp = conn.getresponse()
                data = resp.read()
                return resp, dict(resp.getheaders()), data

            # 1. Range GET (206)
            resp1, hdr1, body1 = send("GET", "/data/splash.wgsl.pack", {"Range": "bytes=0-4095"})
            self.assertEqual(resp1.version, 11)
            self.assertEqual(resp1.status, 206)
            self.assertFalse(resp1.will_close)
            self._assert_coop_coep(hdr1)
            self.assertEqual(hdr1.get("Content-Length"), "4096")
            self.assertEqual(body1, self.large_payload[:4096])
            sock = conn.sock
            self.assertIsNotNone(sock)
            etag = hdr1["ETag"]

            # 2. Full GET (200) on the same socket
            resp2, hdr2, body2 = send("GET", "/index.html")
            self.assertIs(conn.sock, sock)
            self.assertEqual(resp2.version, 11)
            self.assertEqual(resp2.status, 200)
            self.assertFalse(resp2.will_close)
            self._assert_coop_coep(hdr2)
            self.assertEqual(hdr2.get("Content-Length"), str(len(body2)))
            self.assertEqual(body2, b"<h1>playgta6web</h1>")

            # 3. HEAD (200, 0 body bytes) on the same socket
            resp3, hdr3, body3 = send("HEAD", "/runtime.wasm")
            self.assertIs(conn.sock, sock)
            self.assertEqual(resp3.version, 11)
            self.assertEqual(resp3.status, 200)
            self.assertFalse(resp3.will_close)
            self._assert_coop_coep(hdr3)
            self.assertEqual(hdr3.get("Content-Length"), "8")
            self.assertEqual(body3, b"")

            # 4. 404 Not Found (Content-Length: 0) on the same socket
            resp4, hdr4, body4 = send("GET", "/data/missing-archive.rpf")
            self.assertIs(conn.sock, sock)
            self.assertEqual(resp4.version, 11)
            self.assertEqual(resp4.status, 404)
            self.assertFalse(resp4.will_close)
            self._assert_coop_coep(hdr4)
            self.assertEqual(hdr4.get("Content-Length"), "0")
            self.assertEqual(body4, b"")

            # 5. /api/status JSON (200) on the same socket
            resp5, hdr5, body5 = send("GET", "/api/status")
            self.assertIs(conn.sock, sock)
            self.assertEqual(resp5.version, 11)
            self.assertEqual(resp5.status, 200)
            self.assertFalse(resp5.will_close)
            self._assert_coop_coep(hdr5)
            self.assertEqual(hdr5.get("Content-Length"), str(len(body5)))
            self.assertTrue(json.loads(body5.decode("utf-8"))["ok"])

            # 6. 412 Precondition Failed + 304 Not Modified + 416 + OPTIONS still keep socket alive
            resp6, hdr6, body6 = send(
                "GET",
                "/data/splash.wgsl.pack",
                {"Range": "bytes=0-15", "If-Match": '"stale-etag"'},
            )
            self.assertIs(conn.sock, sock)
            self.assertEqual(resp6.status, 412)
            self.assertFalse(resp6.will_close)
            self._assert_coop_coep(hdr6)
            self.assertEqual(hdr6.get("Content-Length"), "0")
            self.assertEqual(hdr6.get("ETag"), etag)
            self.assertEqual(body6, b"")

            resp7, hdr7, body7 = send(
                "GET",
                "/data/splash.wgsl.pack",
                {"If-None-Match": etag},
            )
            self.assertIs(conn.sock, sock)
            self.assertEqual(resp7.status, 304)
            self.assertFalse(resp7.will_close)
            self._assert_coop_coep(hdr7)
            self.assertEqual(body7, b"")

            resp8, hdr8, body8 = send(
                "GET",
                "/data/splash.wgsl.pack",
                {"Range": f"bytes={self.large_size + 100}-"},
            )
            self.assertIs(conn.sock, sock)
            self.assertEqual(resp8.status, 416)
            self.assertFalse(resp8.will_close)
            self.assertEqual(hdr8.get("Content-Length"), "0")
            self.assertEqual(body8, b"")

            resp9, hdr9, body9 = send("OPTIONS", "/data/splash.wgsl.pack")
            self.assertIs(conn.sock, sock)
            self.assertEqual(resp9.status, 204)
            self.assertFalse(resp9.will_close)
            self._assert_coop_coep(hdr9)
            self.assertEqual(body9, b"")

            # Final Range GET proves keep-alive framing stayed synchronized throughout.
            resp10, hdr10, body10 = send(
                "GET",
                "/data/splash.wgsl.pack",
                {"Range": "bytes=4096-8191", "If-Match": etag},
            )
            self.assertIs(conn.sock, sock)
            self.assertEqual(resp10.status, 206)
            self.assertFalse(resp10.will_close)
            self.assertEqual(body10, self.large_payload[4096:8192])
        finally:
            conn.close()

    def test_etag_if_match_and_if_range_validation(self) -> None:
        pack_path = self.data_root / "tiny.wgsl.pack"
        v1 = b"WGPK0123456789abcdef"
        pack_path.write_bytes(v1)
        expected_etag_v1 = serve_local.compute_etag(pack_path.stat())

        st_head, hdr_head, body_head = self._request("HEAD", "/data/tiny.wgsl.pack")
        self.assertEqual(st_head, 200)
        self._assert_coop_coep(hdr_head)
        self.assertEqual(hdr_head.get("ETag"), expected_etag_v1)
        self.assertEqual(body_head, b"")

        # If-Match matching ETag / wildcard / list -> 206 Partial Content
        for if_match in (expected_etag_v1, "*", f'"other-tag", {expected_etag_v1}'):
            with self.subTest(if_match=if_match):
                st, hdr, body = self._request(
                    "GET",
                    "/data/tiny.wgsl.pack",
                    {"Range": "bytes=4-11", "If-Match": if_match},
                )
                self.assertEqual(st, 206)
                self._assert_coop_coep(hdr)
                self.assertEqual(hdr.get("ETag"), expected_etag_v1)
                self.assertEqual(hdr.get("Content-Range"), f"bytes 4-11/{len(v1)}")
                self.assertEqual(body, v1[4:12])

        # If-Match mismatch or weak ETag -> 412 Precondition Failed with Content-Length: 0
        for method in ("GET", "HEAD"):
            for bad_if_match in ('"wrong-etag"', f"W/{expected_etag_v1}", ""):
                with self.subTest(method=method, bad_if_match=bad_if_match):
                    st, hdr, body = self._request(
                        method,
                        "/data/tiny.wgsl.pack",
                        {"Range": "bytes=4-11", "If-Match": bad_if_match},
                    )
                    self.assertEqual(st, 412)
                    self._assert_coop_coep(hdr)
                    self.assertEqual(hdr.get("Content-Length"), "0")
                    self.assertEqual(hdr.get("ETag"), expected_etag_v1)
                    self.assertEqual(body, b"")

        # If-Range matching strong ETag -> 206; mismatched or weak ETag -> 200 full file
        st_ir_ok, hdr_ir_ok, body_ir_ok = self._request(
            "GET",
            "/data/tiny.wgsl.pack",
            {"Range": "bytes=4-11", "If-Range": expected_etag_v1},
        )
        self.assertEqual(st_ir_ok, 206)
        self.assertEqual(hdr_ir_ok.get("Content-Range"), f"bytes 4-11/{len(v1)}")
        self.assertEqual(body_ir_ok, v1[4:12])

        for stale_if_range in ('"stale-etag"', f"W/{expected_etag_v1}"):
            with self.subTest(stale_if_range=stale_if_range):
                st_ir_miss, hdr_ir_miss, body_ir_miss = self._request(
                    "GET",
                    "/data/tiny.wgsl.pack",
                    {"Range": "bytes=4-11", "If-Range": stale_if_range},
                )
                self.assertEqual(st_ir_miss, 200)
                self._assert_coop_coep(hdr_ir_miss)
                self.assertIsNone(hdr_ir_miss.get("Content-Range"))
                self.assertEqual(hdr_ir_miss.get("Content-Length"), str(len(v1)))
                self.assertEqual(body_ir_miss, v1)

        # Updating file on disk immediately turns previous ETag stale for both If-Match and If-Range.
        v2 = b"WGPK-UPDATED-PAYLOAD-0123456789"
        pack_path.write_bytes(v2)
        expected_etag_v2 = serve_local.compute_etag(pack_path.stat())
        self.assertNotEqual(expected_etag_v2, expected_etag_v1)

        st_412, hdr_412, body_412 = self._request(
            "GET",
            "/data/tiny.wgsl.pack",
            {"Range": "bytes=0-7", "If-Match": expected_etag_v1},
        )
        self.assertEqual(st_412, 412)
        self.assertEqual(hdr_412.get("ETag"), expected_etag_v2)
        self.assertEqual(hdr_412.get("Content-Length"), "0")
        self.assertEqual(body_412, b"")

        st_200, hdr_200, body_200 = self._request(
            "GET",
            "/data/tiny.wgsl.pack",
            {"Range": "bytes=0-7", "If-Range": expected_etag_v1},
        )
        self.assertEqual(st_200, 200)
        self.assertEqual(hdr_200.get("ETag"), expected_etag_v2)
        self.assertEqual(body_200, v2)

    def test_concurrent_8_thread_rpf_range_reads_and_disk_update_invalidation(self) -> None:
        rpf_path = self.data_root / "common.rpf"
        payload_v1 = _make_deterministic_bytes(256 * 1024)
        rpf_path.write_bytes(payload_v1)

        slices = [(i * 4096, i * 4096 + 8191) for i in range(32)]

        def fetch_slice(bounds: tuple[int, int]) -> tuple[int, int, bytes, str]:
            lo, hi = bounds
            st, hdr, body = self._request(
                "GET",
                "/data/common.rpf",
                {"Range": f"bytes={lo}-{hi}"},
            )
            self.assertEqual(st, 206)
            self._assert_coop_coep(hdr)
            return lo, hi, body, hdr["ETag"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results_v1 = list(pool.map(fetch_slice, slices))

        etag_v1 = serve_local.compute_etag(rpf_path.stat())
        for lo, hi, body, etag in results_v1:
            self.assertEqual(etag, etag_v1)
            self.assertEqual(body, payload_v1[lo : hi + 1])

        cache = self.httpd.file_handle_cache
        self.assertIsNotNone(cache)
        assert cache is not None
        stats_v1 = cache.stats()
        self.assertGreater(stats_v1["hits"], 0)
        self.assertGreaterEqual(stats_v1["entries"], 1)

        # Update file on disk (different size) -> cached handle invalidates cleanly.
        payload_v2 = bytes(((b ^ 0x5A) for b in _make_deterministic_bytes(320 * 1024)))
        rpf_path.write_bytes(payload_v2)
        etag_v2 = serve_local.compute_etag(rpf_path.stat())
        self.assertNotEqual(etag_v2, etag_v1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results_v2 = list(pool.map(fetch_slice, slices))

        for lo, hi, body, etag in results_v2:
            self.assertEqual(etag, etag_v2)
            self.assertEqual(body, payload_v2[lo : hi + 1])

        stats_v2 = cache.stats()
        self.assertGreaterEqual(stats_v2["invalidations"], 1)

        # Same-size overwrite with bumped mtime_ns also invalidates cached handle.
        prev_st = rpf_path.stat()
        payload_v3 = bytes(((b ^ 0xA5) for b in payload_v2))
        self.assertEqual(len(payload_v3), len(payload_v2))
        rpf_path.write_bytes(payload_v3)
        os.utime(rpf_path, ns=(prev_st.st_atime_ns, prev_st.st_mtime_ns + 5_000_000_000))
        etag_v3 = serve_local.compute_etag(rpf_path.stat())
        self.assertNotEqual(etag_v3, etag_v2)

        st_v3, hdr_v3, body_v3 = self._request(
            "GET",
            "/data/common.rpf",
            {"Range": "bytes=0-16383", "If-Match": etag_v3},
        )
        self.assertEqual(st_v3, 206)
        self.assertEqual(hdr_v3.get("ETag"), etag_v3)
        self.assertEqual(body_v3, payload_v3[:16384])
        self.assertGreaterEqual(cache.stats()["invalidations"], stats_v2["invalidations"] + 1)

        # Deleting the cached file works cleanly (including on Windows) and yields 404.
        rpf_path.unlink()
        st_404, _, body_404 = self._request("GET", "/data/common.rpf")
        self.assertEqual(st_404, 404)
        self.assertEqual(body_404, b"")

    def test_file_handle_cache_bounded_lru_eviction_and_shutdown_cleanup(self) -> None:
        with self.assertRaises(ValueError):
            serve_local.create_server(
                "127.0.0.1",
                0,
                self.web_root,
                self.data_root,
                file_handle_cache_size=-1,
            )

        files = {
            "a.rpf": b"RPF-AAAA-0123456789",
            "b.pack": b"PACK-BBBB-0123456789",
            "c.pgim": b"PGIM-CCCC-0123456789",
            "d.wasm": b"\x00asm\x01\x00\x00\x00DDDD",
        }
        for name, data in files.items():
            (self.data_root / name).write_bytes(data)

        httpd = serve_local.create_server(
            "127.0.0.1",
            0,
            self.web_root,
            self.data_root,
            file_handle_cache_size=2,
        )
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        cache = httpd.file_handle_cache
        self.assertIsNotNone(cache)
        assert cache is not None
        try:
            # Non-cacheable .html file should never enter the FD cache.
            st_html, _, body_html = self._request("GET", "/index.html", port=port)
            self.assertEqual(st_html, 200)
            self.assertEqual(body_html, b"<h1>playgta6web</h1>")
            self.assertEqual(len(cache), 0)
            self.assertEqual(cache.stats()["open_handles"], 0)

            for name, expected in files.items():
                st, _, body = self._request(
                    "GET",
                    f"/data/{name}",
                    {"Range": "bytes=0-7"},
                    port=port,
                )
                self.assertEqual(st, 206)
                self.assertEqual(body, expected[:8])
                self.assertLessEqual(len(cache), 2)
                self.assertLessEqual(cache.stats()["open_handles"], 2)

            stats = cache.stats()
            self.assertEqual(stats["entries"], 2)
            self.assertEqual(stats["open_handles"], 2)
            self.assertGreaterEqual(stats["evictions"], 2)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=2)

        # Shutdown / server_close must close every cached handle with zero FD leaks.
        final_stats = cache.stats()
        self.assertEqual(final_stats["entries"], 0)
        self.assertEqual(final_stats["open_handles"], 0)

    def test_server_request_queue_size_and_daemon_threads(self) -> None:
        self.assertGreaterEqual(serve_local.QuietThreadingHTTPServer.request_queue_size, 256)
        self.assertGreaterEqual(self.httpd.request_queue_size, 256)
        self.assertIs(serve_local.QuietThreadingHTTPServer.daemon_threads, True)
        self.assertIs(self.httpd.daemon_threads, True)
        self.assertTrue(hasattr(serve_local._LOG_LOCK, "acquire"))
        self.assertTrue(hasattr(serve_local._LOG_LOCK, "release"))

    def test_log_message_stderr_enospc_and_closed_stream_do_not_break_responses(self) -> None:
        write_calls = 0

        class _FailingStderr:
            def __init__(self, exc: Exception) -> None:
                self._exc = exc

            def write(self, _text: str) -> int:
                nonlocal write_calls
                write_calls += 1
                raise self._exc

        enospc = OSError(errno.ENOSPC, "No space left on device")
        with mock.patch.object(sys, "stderr", _FailingStderr(enospc)):
            st200, hdr200, body200 = self._request("GET", "/runtime.wasm")
            self.assertEqual(st200, 200)
            self._assert_coop_coep(hdr200)
            self.assertEqual(body200, b"\x00asm\x01\x00\x00\x00")

            st206, hdr206, body206 = self._request(
                "GET", "/data/splash.wgsl.pack", {"Range": "bytes=0-4095"}
            )
            self.assertEqual(st206, 206)
            self._assert_coop_coep(hdr206)
            self.assertEqual(hdr206.get("Content-Range"), f"bytes 0-4095/{self.large_size}")
            self.assertEqual(body206, self.large_payload[:4096])

        self.assertGreaterEqual(write_calls, 2)

        closed_err = ValueError("I/O operation on closed file")
        with mock.patch.object(sys, "stderr", _FailingStderr(closed_err)):
            st_val, hdr_val, body_val = self._request("GET", "/index.html")
            self.assertEqual(st_val, 200)
            self._assert_coop_coep(hdr_val)
            self.assertEqual(body_val, b"<h1>playgta6web</h1>")

        with mock.patch.object(sys, "stderr", None):
            st_none, hdr_none, body_none = self._request(
                "GET", "/data/splash.wgsl.pack", {"Range": "bytes=4096-8191"}
            )
            self.assertEqual(st_none, 206)
            self._assert_coop_coep(hdr_none)
            self.assertEqual(body_none, self.large_payload[4096:8192])

    def test_log_message_serializes_concurrent_stderr_writes_under_log_lock(self) -> None:
        in_write = False
        overlap_detected = False
        lines: list[str] = []

        class _OverlapDetectingStderr:
            def write(self, text: str) -> int:
                nonlocal in_write, overlap_detected
                if in_write:
                    overlap_detected = True
                in_write = True
                try:
                    lines.append(text)
                    return len(text)
                finally:
                    in_write = False

        with mock.patch.object(sys, "stderr", _OverlapDetectingStderr()):
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                futures = [
                    pool.submit(self._request, "GET", "/runtime.wasm") for _ in range(16)
                ]
                for fut in futures:
                    st, hdr, body = fut.result()
                    self.assertEqual(st, 200)
                    self._assert_coop_coep(hdr)
                    self.assertEqual(body, b"\x00asm\x01\x00\x00\x00")

        self.assertFalse(overlap_detected, "_LOG_LOCK must serialize concurrent stderr writes")
        self.assertGreaterEqual(len(lines), 16)

    def test_quiet_option_suppresses_access_logs_and_cli_flag(self) -> None:
        noisy_buf = io.StringIO()
        with mock.patch.object(sys, "stderr", noisy_buf):
            st_noisy, hdr_noisy, body_noisy = self._request("GET", "/index.html")
        self.assertEqual(st_noisy, 200)
        self._assert_coop_coep(hdr_noisy)
        self.assertEqual(body_noisy, b"<h1>playgta6web</h1>")
        self.assertIn("/index.html", noisy_buf.getvalue())

        quiet_httpd = serve_local.create_server(
            "127.0.0.1",
            0,
            self.web_root,
            self.data_root,
            quiet=True,
        )
        self.assertTrue(quiet_httpd.quiet)
        quiet_port = quiet_httpd.server_address[1]
        quiet_thread = threading.Thread(target=quiet_httpd.serve_forever, daemon=True)
        quiet_thread.start()
        try:
            quiet_buf = io.StringIO()
            with mock.patch.object(sys, "stderr", quiet_buf):
                st200, hdr200, body200 = self._request("GET", "/index.html", port=quiet_port)
                self.assertEqual(st200, 200)
                self._assert_coop_coep(hdr200)
                self.assertEqual(body200, b"<h1>playgta6web</h1>")

                st206, hdr206, body206 = self._request(
                    "GET",
                    "/data/splash.wgsl.pack",
                    {"Range": "bytes=0-63"},
                    port=quiet_port,
                )
                self.assertEqual(st206, 206)
                self._assert_coop_coep(hdr206)
                self.assertEqual(body206, self.large_payload[:64])

                st404, hdr404, body404 = self._request("GET", "/does-not-exist", port=quiet_port)
                self.assertEqual(st404, 404)
                self._assert_coop_coep(hdr404)
                self.assertEqual(body404, b"")

            self.assertEqual(quiet_buf.getvalue(), "")
        finally:
            quiet_httpd.shutdown()
            quiet_httpd.server_close()
            quiet_thread.join(timeout=2)

        # Verify CLI --quiet flag forwards quiet=True (and defaults to quiet=False).
        for argv_extra, expected_quiet in (([], False), (["--quiet"], True)):
            with self.subTest(argv_extra=argv_extra):
                captured_kwargs: dict[str, object] = {}

                class _FakeServer:
                    def serve_forever(self) -> None:
                        raise KeyboardInterrupt

                    def server_close(self) -> None:
                        return None

                def _fake_create_server(*args, **kwargs):
                    captured_kwargs.update(kwargs)
                    return _FakeServer()

                argv = [
                    "serve_local.py",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "0",
                    "--web",
                    str(self.web_root),
                    "--data",
                    str(self.data_root),
                    *argv_extra,
                ]
                with (
                    mock.patch.object(sys, "argv", argv),
                    mock.patch.object(serve_local, "create_server", side_effect=_fake_create_server),
                    mock.patch("builtins.print"),
                ):
                    serve_local.main()
                self.assertIs(captured_kwargs.get("quiet"), expected_quiet)

    def test_range_chunk_bytes_and_max_concurrent_fds_validation_and_cli_flags(self) -> None:
        self.assertEqual(serve_local.RANGE_STREAM_CHUNK_BYTES, 256 * 1024)
        self.assertEqual(serve_local.DEFAULT_CONCURRENT_FD_POOL_SIZE, 8)
        self.assertEqual(serve_local.STREAM_CHUNK_BYTES, 1024 * 1024)
        self.assertEqual(serve_local.STREAM_CHUNK_SIZE, 1024 * 1024)

        for bad_chunk in (0, -1, -4096):
            with self.subTest(bad_range_chunk=bad_chunk):
                with self.assertRaises(ValueError):
                    serve_local.create_server(
                        "127.0.0.1",
                        0,
                        self.web_root,
                        self.data_root,
                        range_stream_chunk_bytes=bad_chunk,
                    )

        for bad_fds in (0, -1, -8):
            with self.subTest(bad_max_concurrent_fds=bad_fds):
                with self.assertRaises(ValueError):
                    serve_local.ReadOnlyFileHandleCache(max_concurrent_fds_per_file=bad_fds)
                with self.assertRaises(ValueError):
                    serve_local.create_server(
                        "127.0.0.1",
                        0,
                        self.web_root,
                        self.data_root,
                        max_concurrent_fds_per_file=bad_fds,
                    )
                with self.assertRaises(ValueError):
                    serve_local.QuietThreadingHTTPServer(
                        ("127.0.0.1", 0),
                        partial(
                            serve_local.PlayGta6Handler,
                            web_root=self.web_root,
                            data_root=self.data_root,
                        ),
                        bind_and_activate=False,
                        max_concurrent_fds_per_file=bad_fds,
                    )

        # Verify chunk-size precedence: default 206 -> 256 KiB, 200 -> 1 MiB;
        # explicit stream_chunk_bytes alone applies to both; explicit range_stream_chunk_bytes
        # overrides 206 only.
        h_default = object.__new__(serve_local.PlayGta6Handler)
        h_default._stream_chunk_bytes = None
        h_default._range_stream_chunk_bytes = None
        self.assertEqual(h_default._get_stream_chunk_bytes(), 1024 * 1024)
        self.assertEqual(h_default._get_range_stream_chunk_bytes(), 256 * 1024)

        h_stream_only = object.__new__(serve_local.PlayGta6Handler)
        h_stream_only._stream_chunk_bytes = 64 * 1024
        h_stream_only._range_stream_chunk_bytes = None
        self.assertEqual(h_stream_only._get_stream_chunk_bytes(), 64 * 1024)
        self.assertEqual(h_stream_only._get_range_stream_chunk_bytes(), 64 * 1024)

        h_range_override = object.__new__(serve_local.PlayGta6Handler)
        h_range_override._stream_chunk_bytes = None
        h_range_override._range_stream_chunk_bytes = 128 * 1024
        self.assertEqual(h_range_override._get_stream_chunk_bytes(), 1024 * 1024)
        self.assertEqual(h_range_override._get_range_stream_chunk_bytes(), 128 * 1024)

        # Verify live 206 Range transfer streams in 256 KiB chunks while 200 OK streams in 1 MiB chunks.
        observed_read_sizes: list[int] = []
        orig_read_at = serve_local._BorrowedFileReader.read_at

        def _spy_read_at(reader_self, offset: int, size: int) -> bytes:
            observed_read_sizes.append(size)
            return orig_read_at(reader_self, offset, size)

        with mock.patch.object(serve_local._BorrowedFileReader, "read_at", _spy_read_at):
            # 1 MiB Range request -> four 256 KiB chunks.
            st206, _, body206 = self._request(
                "GET",
                "/data/splash.wgsl.pack",
                {"Range": f"bytes=0-{1024 * 1024 - 1}"},
            )
            self.assertEqual(st206, 206)
            self.assertEqual(len(body206), 1024 * 1024)
            range_reads = list(observed_read_sizes)
            observed_read_sizes.clear()

            # Full 3.5 MiB 200 OK request -> three 1 MiB chunks + one 512 KiB chunk.
            st200, _, body200 = self._request("GET", "/data/splash.wgsl.pack")
            self.assertEqual(st200, 200)
            self.assertEqual(len(body200), self.large_size)
            full_reads = list(observed_read_sizes)

        self.assertEqual(range_reads, [256 * 1024] * 4)
        self.assertEqual(full_reads, [1024 * 1024, 1024 * 1024, 1024 * 1024, 512 * 1024])

        # Verify CLI --range-chunk-bytes and --max-concurrent-fds forwarding & validation.
        for argv_extra, expected_range_chunk, expected_max_fds in (
            ([], None, serve_local.DEFAULT_CONCURRENT_FD_POOL_SIZE),
            (["--range-chunk-bytes", "131072", "--max-concurrent-fds", "12"], 131072, 12),
        ):
            with self.subTest(argv_extra=argv_extra):
                captured_kwargs: dict[str, object] = {}

                class _FakeServer:
                    def serve_forever(self) -> None:
                        raise KeyboardInterrupt

                    def server_close(self) -> None:
                        return None

                def _fake_create_server(*args, **kwargs):
                    captured_kwargs.update(kwargs)
                    return _FakeServer()

                argv = [
                    "serve_local.py",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "0",
                    "--web",
                    str(self.web_root),
                    "--data",
                    str(self.data_root),
                    *argv_extra,
                ]
                with (
                    mock.patch.object(sys, "argv", argv),
                    mock.patch.object(serve_local, "create_server", side_effect=_fake_create_server),
                    mock.patch("builtins.print"),
                ):
                    serve_local.main()
                self.assertEqual(captured_kwargs.get("range_stream_chunk_bytes"), expected_range_chunk)
                self.assertEqual(captured_kwargs.get("max_concurrent_fds_per_file"), expected_max_fds)

        for bad_cli in (["--range-chunk-bytes", "0"], ["--max-concurrent-fds", "0"]):
            with self.subTest(bad_cli=bad_cli):
                argv = [
                    "serve_local.py",
                    "--web",
                    str(self.web_root),
                    "--data",
                    str(self.data_root),
                    *bad_cli,
                ]
                with (
                    mock.patch.object(sys, "argv", argv),
                    mock.patch.object(sys, "stderr", io.StringIO()),
                    self.assertRaises(SystemExit) as cm,
                ):
                    serve_local.main()
                self.assertNotEqual(cm.exception.code, 0)

    def test_8lane_httpfs_persistent_keepalive_stress_nodelay_latency_and_zero_fd_leaks(self) -> None:
        levels_size = 2 * 1024 * 1024  # 2 MiB synthetic levels.rpf
        bvh4_size = 1 * 1024 * 1024 + 512 * 1024  # 1.5 MiB synthetic core_bvh4.rpf
        levels_bytes = _make_deterministic_bytes(levels_size)
        bvh4_bytes = bytes((b ^ 0x3C) for b in _make_deterministic_bytes(bvh4_size))

        levels_path = self.data_root / "levels.rpf"
        bvh4_path = self.data_root / "core_bvh4.rpf"
        levels_path.write_bytes(levels_bytes)
        bvh4_path.write_bytes(bvh4_bytes)

        expected_assets = {
            "/data/levels.rpf": (levels_bytes, serve_local.compute_etag(levels_path.stat())),
            "/data/core_bvh4.rpf": (bvh4_bytes, serve_local.compute_etag(bvh4_path.stat())),
        }

        nodelay_lock = threading.Lock()
        observed_sockets: dict[int, int] = {}
        orig_setup = serve_local.PlayGta6Handler.setup

        def _tracking_setup(handler_self) -> None:
            orig_setup(handler_self)
            conn = getattr(handler_self, "connection", None)
            if conn is not None:
                nodelay = conn.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
                with nodelay_lock:
                    observed_sockets[id(conn)] = nodelay

        lanes = 8
        requests_per_lane = 32  # 8 * 32 = 256 total 206 Range requests (>= 200)
        start_barrier = threading.Barrier(lanes, timeout=10.0)

        with mock.patch.object(serve_local.PlayGta6Handler, "setup", _tracking_setup):
            httpd = serve_local.create_server(
                "127.0.0.1",
                0,
                self.web_root,
                self.data_root,
                max_concurrent_fds_per_file=lanes,
                quiet=True,
            )
            port = httpd.server_address[1]
            cache = httpd.file_handle_cache
            self.assertIsNotNone(cache)
            assert cache is not None
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                def run_lane(lane_idx: int) -> tuple[list[float], list[float], int]:
                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
                    all_lat_ms: list[float] = []
                    probe_lat_ms: list[float] = []
                    first_sock = None
                    try:
                        # Synchronize all 8 lanes so connections & Range bursts overlap.
                        start_barrier.wait()
                        for req_idx in range(requests_per_lane):
                            url_path = (
                                "/data/levels.rpf"
                                if ((lane_idx + req_idx) % 2 == 0)
                                else "/data/core_bvh4.rpf"
                            )
                            payload, expected_etag = expected_assets[url_path]
                            file_len = len(payload)
                            is_probe = (req_idx % 2) == 1
                            slice_len = (64 * 1024) if is_probe else (256 * 1024)
                            max_start = file_len - slice_len
                            offset = ((lane_idx * 131_072) + (req_idx * 65_536)) % (max_start + 1)
                            end = offset + slice_len - 1
                            expected_slice = payload[offset : end + 1]
                            expected_sha = hashlib.sha256(expected_slice).hexdigest()

                            t0 = time.perf_counter()
                            conn.request(
                                "GET",
                                url_path,
                                headers={"Range": f"bytes={offset}-{end}"},
                            )
                            resp = conn.getresponse()
                            body = resp.read()
                            elapsed_ms = (time.perf_counter() - t0) * 1000.0

                            hdrs = dict(resp.getheaders())
                            self.assertEqual(resp.status, 206)
                            self.assertFalse(resp.will_close)
                            self._assert_coop_coep(hdrs)
                            etag = hdrs.get("ETag", "")
                            self.assertEqual(etag, expected_etag)
                            self.assertTrue(etag.startswith('"') and etag.endswith('"'))
                            self.assertEqual(
                                hdrs.get("Content-Range"),
                                f"bytes {offset}-{end}/{file_len}",
                            )
                            self.assertEqual(hdrs.get("Content-Length"), str(slice_len))
                            self.assertEqual(len(body), slice_len)
                            self.assertEqual(hashlib.sha256(body).hexdigest(), expected_sha)

                            if first_sock is None:
                                first_sock = conn.sock
                                self.assertIsNotNone(first_sock)
                            else:
                                self.assertIs(
                                    conn.sock,
                                    first_sock,
                                    msg=f"Lane {lane_idx} failed to reuse keep-alive socket at req {req_idx}",
                                )

                            all_lat_ms.append(elapsed_ms)
                            if is_probe:
                                probe_lat_ms.append(elapsed_ms)
                        return all_lat_ms, probe_lat_ms, requests_per_lane
                    finally:
                        conn.close()

                with concurrent.futures.ThreadPoolExecutor(max_workers=lanes) as pool:
                    lane_results = list(pool.map(run_lane, range(lanes)))

                all_latencies: list[float] = []
                probe_latencies: list[float] = []
                total_completed = 0
                for lane_all, lane_probes, count in lane_results:
                    all_latencies.extend(lane_all)
                    probe_latencies.extend(lane_probes)
                    total_completed += count

                self.assertEqual(total_completed, lanes * requests_per_lane)
                self.assertGreaterEqual(total_completed, 200)

                # Every one of the 8 concurrent client sockets must have TCP_NODELAY enabled.
                self.assertEqual(len(observed_sockets), lanes)
                for sock_id, nodelay_val in observed_sockets.items():
                    self.assertNotEqual(
                        nodelay_val,
                        0,
                        msg=f"TCP_NODELAY was disabled on client socket {sock_id}",
                    )

                stats_active = cache.stats()
                self.assertEqual(stats_active["entries"], 2)
                self.assertGreaterEqual(stats_active["hits"], total_completed - 2)
                self.assertLessEqual(
                    stats_active["open_handles"],
                    2 * lanes,
                )

                # Sanity-check p99 loopback Range latency on 64 KiB probes.
                ordered_probes = sorted(probe_latencies)
                p99_idx = min(len(ordered_probes) - 1, math.ceil(0.99 * len(ordered_probes)) - 1)
                probe_p99_ms = ordered_probes[p99_idx]
                self.assertLess(
                    probe_p99_ms,
                    150.0,
                    msg=f"8-lane 64 KiB loopback probe p99 latency unexpectedly high: {probe_p99_ms:.2f} ms",
                )
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join(timeout=2)

        final_stats = cache.stats()
        self.assertEqual(final_stats["entries"], 0)
        self.assertEqual(final_stats["open_handles"], 0)
        self.assertEqual(final_stats["pooled_handles"], 0)

    def test_non_pread_concurrent_fd_pool_parallel_reads_fallback_and_in_flight_invalidation(self) -> None:
        levels_path = self.data_root / "levels_pool.rpf"
        bvh4_path = self.data_root / "core_bvh4_pool.rpf"
        levels_bytes = _make_deterministic_bytes(512 * 1024)
        bvh4_bytes = bytes((b ^ 0x55) for b in _make_deterministic_bytes(256 * 1024))
        levels_path.write_bytes(levels_bytes)
        bvh4_path.write_bytes(bvh4_bytes)
        levels_st = levels_path.stat()
        bvh4_st = bvh4_path.stat()

        with mock.patch.object(serve_local, "_HAS_PREAD", False):
            # Part A: Prove 4 concurrent threads read from the SAME cached .rpf simultaneously
            # without serializing on io_lock (Barrier(4) inside os.read would deadlock if serialized).
            cache = serve_local.ReadOnlyFileHandleCache(
                max_entries=4,
                max_concurrent_fds_per_file=4,
            )
            orig_os_read = os.read
            read_barrier = threading.Barrier(4, timeout=5.0)
            overlap_lock = threading.Lock()
            active_readers = 0
            max_concurrent_readers = 0
            observed_fds: set[int] = set()
            barrier_fds: set[int] = set()

            def _barrier_os_read(fd: int, n: int) -> bytes:
                nonlocal active_readers, max_concurrent_readers
                with overlap_lock:
                    should_barrier = fd in barrier_fds
                if should_barrier:
                    with overlap_lock:
                        active_readers += 1
                        if active_readers > max_concurrent_readers:
                            max_concurrent_readers = active_readers
                        observed_fds.add(fd)
                    try:
                        read_barrier.wait()
                        return orig_os_read(fd, n)
                    finally:
                        with overlap_lock:
                            active_readers -= 1
                return orig_os_read(fd, n)

            try:
                def parallel_worker(idx: int) -> bytes:
                    entry = cache.acquire(levels_path, levels_st)
                    self.assertIsNotNone(entry)
                    assert entry is not None
                    try:
                        with entry.borrow_fd() as reader:
                            with overlap_lock:
                                barrier_fds.add(reader.fd)
                            offset = idx * 65_536
                            return reader.read_at(offset, 65_536)
                    finally:
                        cache.release(entry)

                with mock.patch.object(serve_local.os, "read", side_effect=_barrier_os_read):
                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                        chunks = list(pool.map(parallel_worker, range(4)))

                for idx, data in enumerate(chunks):
                    self.assertEqual(data, levels_bytes[idx * 65_536 : (idx + 1) * 65_536])
                self.assertEqual(
                    max_concurrent_readers,
                    4,
                    "All 4 non-pread readers must execute os.read concurrently without io_lock serialization",
                )
                self.assertEqual(len(observed_fds), 4)

                stats_after_parallel = cache.stats()
                self.assertEqual(stats_after_parallel["entries"], 1)
                self.assertEqual(stats_after_parallel["open_handles"], 4)
                self.assertEqual(stats_after_parallel["pooled_handles"], 3)
                self.assertEqual(stats_after_parallel["pool_opens"], 3)

                # Part B: Oversubscription (> max_concurrent_fds_per_file) cleanly falls back to
                # primary FD guarded by io_lock without failing or leaking FDs.
                oversub_barrier = threading.Barrier(6, timeout=5.0)

                def oversub_worker(idx: int) -> bytes:
                    entry = cache.acquire(levels_path, levels_st)
                    self.assertIsNotNone(entry)
                    assert entry is not None
                    try:
                        with entry.borrow_fd() as reader:
                            oversub_barrier.wait()
                            offset = idx * 32_768
                            part1 = reader.read_at(offset, 16_384)
                            part2 = reader.read_at(offset + 16_384, 16_384)
                            return part1 + part2
                    finally:
                        cache.release(entry)

                with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                    oversub_chunks = list(pool.map(oversub_worker, range(6)))

                for idx, data in enumerate(oversub_chunks):
                    self.assertEqual(data, levels_bytes[idx * 32_768 : (idx + 1) * 32_768])
                self.assertEqual(cache.stats()["open_handles"], 4)
                self.assertEqual(cache.stats()["pooled_handles"], 3)

                # Part C: Invalidation while 4 readers are actively in-flight closes ALL pooled
                # and primary FDs once the in-flight readers finish, with zero FD leaks.
                entered_barrier = threading.Barrier(5, timeout=5.0)
                release_gate = threading.Event()

                def inflight_reader(idx: int) -> bytes:
                    entry = cache.acquire(levels_path, levels_st)
                    self.assertIsNotNone(entry)
                    assert entry is not None
                    try:
                        with entry.borrow_fd() as reader:
                            entered_barrier.wait()
                            self.assertTrue(release_gate.wait(timeout=5.0))
                            return reader.read_at(idx * 4096, 4096)
                    finally:
                        cache.release(entry)

                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    futures = [pool.submit(inflight_reader, i) for i in range(4)]
                    entered_barrier.wait()
                    cache.invalidate(levels_path)
                    self.assertEqual(len(cache), 0)
                    release_gate.set()
                    inflight_results = [fut.result() for fut in futures]

                for idx, data in enumerate(inflight_results):
                    self.assertEqual(data, levels_bytes[idx * 4096 : (idx + 1) * 4096])

                stats_post_inv = cache.stats()
                self.assertEqual(stats_post_inv["entries"], 0)
                self.assertEqual(stats_post_inv["open_handles"], 0)
                self.assertEqual(stats_post_inv["pooled_handles"], 0)
            finally:
                cache.close_all()

            # Part D: LRU eviction while pooled FDs are in-flight closes all evicted FDs on release.
            evict_cache = serve_local.ReadOnlyFileHandleCache(
                max_entries=1,
                max_concurrent_fds_per_file=4,
            )
            evict_barrier = threading.Barrier(4, timeout=5.0)
            evict_gate = threading.Event()
            try:
                def evict_reader(idx: int) -> bytes:
                    entry = evict_cache.acquire(levels_path, levels_st)
                    self.assertIsNotNone(entry)
                    assert entry is not None
                    try:
                        with entry.borrow_fd() as reader:
                            evict_barrier.wait()
                            self.assertTrue(evict_gate.wait(timeout=5.0))
                            return reader.read_at(idx * 8192, 8192)
                    finally:
                        evict_cache.release(entry)

                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                    futures = [pool.submit(evict_reader, i) for i in range(3)]
                    evict_barrier.wait()
                    # Acquire a second file to evict levels_path while its 3 FDs are in-flight.
                    bvh4_entry = evict_cache.acquire(bvh4_path, bvh4_st)
                    self.assertIsNotNone(bvh4_entry)
                    assert bvh4_entry is not None
                    try:
                        with bvh4_entry.borrow_fd() as reader:
                            self.assertEqual(reader.read_at(0, 1024), bvh4_bytes[:1024])
                    finally:
                        evict_cache.release(bvh4_entry)
                    evict_gate.set()
                    evicted_results = [fut.result() for fut in futures]

                for idx, data in enumerate(evicted_results):
                    self.assertEqual(data, levels_bytes[idx * 8192 : (idx + 1) * 8192])

                stats_after_evict = evict_cache.stats()
                self.assertEqual(stats_after_evict["entries"], 1)
                self.assertEqual(stats_after_evict["evictions"], 1)
                self.assertEqual(stats_after_evict["open_handles"], 1)
                self.assertEqual(stats_after_evict["pooled_handles"], 0)
            finally:
                evict_cache.close_all()

            self.assertEqual(evict_cache.stats()["open_handles"], 0)
            self.assertEqual(evict_cache.stats()["pooled_handles"], 0)

    def test_preresolved_roots_avoid_per_request_realpath_and_405_drains_small_bodies(self) -> None:
        # Confirm Range requests on cached assets invoke os.path.realpath only once (for the
        # target candidate inside safe_resolve) rather than 3x (root + candidate + cache.acquire),
        # and forward the already-resolved realpath straight into ReadOnlyFileHandleCache.acquire.
        orig_realpath = os.path.realpath
        realpath_calls = 0
        acquire_realpaths: list[str | None] = []
        orig_acquire = serve_local.ReadOnlyFileHandleCache.acquire

        def _counting_realpath(path, *, strict=False):
            nonlocal realpath_calls
            realpath_calls += 1
            return orig_realpath(path, strict=strict)

        def _spy_acquire(cache_self, path, st, *, realpath=None):
            acquire_realpaths.append(realpath)
            return orig_acquire(cache_self, path, st, realpath=realpath)

        with (
            mock.patch.object(serve_local.os.path, "realpath", side_effect=_counting_realpath),
            mock.patch.object(serve_local.ReadOnlyFileHandleCache, "acquire", _spy_acquire),
        ):
            st, hdr, body = self._request(
                "GET",
                "/data/splash.wgsl.pack",
                {"Range": "bytes=0-4095"},
            )
        self.assertEqual(st, 206)
        self._assert_coop_coep(hdr)
        self.assertEqual(body, self.large_payload[:4096])
        self.assertEqual(
            realpath_calls,
            1,
            f"Expected exactly 1 realpath call (target only, pre-resolved root & cache), got {realpath_calls}",
        )
        self.assertEqual(acquire_realpaths, [str(self.large_file.resolve())])

        # Confirm repeated 405 responses with small unread request bodies drain cleanly on Windows
        # loopback without triggering TCP RST (WinError 10053 / ConnectionAbortedError).
        for method, path, payload in (
            ("POST", "/api/status", b'{"x":1}' * 8),
            ("PUT", "/data/splash.wgsl.pack", b"A" * 4096),
            ("DELETE", "/index.html", b"B" * 256),
        ):
            with self.subTest(method=method, path=path, size=len(payload)):
                st405, hdr405, body405 = self._request(method, path, body=payload)
                self.assertEqual(st405, 405)
                self._assert_coop_coep(hdr405)
                self.assertEqual(hdr405.get("Content-Length"), "0")
                self.assertEqual(hdr405.get("Connection", "").lower(), "close")
                self.assertIn("GET", hdr405.get("Allow", ""))
                self.assertEqual(body405, b"")

    def test_resolved_path_cache_skips_repeat_realpath_invalidates_on_mutate_and_blocks_traversal(self) -> None:
        orig_realpath = os.path.realpath
        realpath_calls = 0

        def _counting_realpath(path, *, strict=False):
            nonlocal realpath_calls
            realpath_calls += 1
            return orig_realpath(path, strict=strict)

        path_cache = self.httpd.resolved_path_cache
        self.assertIsNotNone(path_cache)
        assert path_cache is not None

        with mock.patch.object(serve_local.os.path, "realpath", side_effect=_counting_realpath):
            # 1st request populates resolved_path_cache (1 realpath call).
            st1, hdr1, body1 = self._request(
                "GET", "/data/splash.wgsl.pack", {"Range": "bytes=0-1023"}
            )
            self.assertEqual(st1, 206)
            self.assertEqual(body1, self.large_payload[:1024])
            self.assertEqual(realpath_calls, 1)

            # Subsequent Range requests hit resolved_path_cache with 0 realpath calls.
            for lo in (1024, 4096, 65536, 131072):
                st_n, _, body_n = self._request(
                    "GET", "/data/splash.wgsl.pack", {"Range": f"bytes={lo}-{lo + 1023}"}
                )
                self.assertEqual(st_n, 206)
                self.assertEqual(body_n, self.large_payload[lo : lo + 1024])
            self.assertEqual(realpath_calls, 1)

        # Traversal attempts must never enter or hit resolved_path_cache.
        size_before_trav = len(path_cache)
        for bad_url in ("/data/../web/index.html", "/data/%2e%2e/web/index.html", "/data/a%00b.rpf"):
            st_bad, hdr_bad, _ = self._request("GET", bad_url)
            self.assertEqual(st_bad, 404)
            self._assert_coop_coep(hdr_bad)
        self.assertEqual(len(path_cache), size_before_trav)

        # Updating, truncating, and deleting a cached path must reflect immediately.
        dyn = self.data_root / "dyn.rpf"
        dyn.write_bytes(b"VERSION-ONE-0123456789")
        st_v1, hdr_v1, body_v1 = self._request("GET", "/data/dyn.rpf", {"Range": "bytes=0-10"})
        self.assertEqual(st_v1, 206)
        self.assertEqual(body_v1, b"VERSION-ONE")

        dyn.write_bytes(b"V2-SHORT")
        st_v2, hdr_v2, body_v2 = self._request("GET", "/data/dyn.rpf", {"Range": "bytes=0-7"})
        self.assertEqual(st_v2, 206)
        self.assertEqual(body_v2, b"V2-SHORT")
        self.assertNotEqual(hdr_v2.get("ETag"), hdr_v1.get("ETag"))

        dyn.unlink()
        st_gone, _, body_gone = self._request("GET", "/data/dyn.rpf")
        self.assertEqual(st_gone, 404)
        self.assertEqual(body_gone, b"")

    def test_cached_fd_zero_alloc_readinto_and_single_write_header_body_coalescing(self) -> None:
        write_sizes: list[int] = []
        orig_serve_file = serve_local.PlayGta6Handler._serve_file

        def _spy_serve_file(handler_self, url_path: str, send_body: bool) -> None:
            orig_write = handler_self.wfile.write

            def _counting_write(data):
                write_sizes.append(len(data))
                return orig_write(data)

            handler_self.wfile.write = _counting_write
            try:
                orig_serve_file(handler_self, url_path, send_body)
            finally:
                handler_self.wfile.write = orig_write

        with mock.patch.object(serve_local.PlayGta6Handler, "_serve_file", _spy_serve_file):
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
            try:
                # 64 KiB Range request must coalesce headers + payload into a single socket write.
                conn.request("GET", "/data/splash.wgsl.pack", headers={"Range": "bytes=0-65535"})
                resp1 = conn.getresponse()
                body1 = resp1.read()
                self.assertEqual(resp1.status, 206)
                self.assertFalse(resp1.will_close)
                self.assertEqual(body1, self.large_payload[:65536])
                self.assertEqual(len(write_sizes), 1, f"Expected 1 coalesced write, got {write_sizes}")
                self.assertGreater(write_sizes[0], 65536)

                # Follow-up request on the same keep-alive connection stays cleanly framed.
                write_sizes.clear()
                conn.request(
                    "GET", "/data/splash.wgsl.pack", headers={"Range": "bytes=65536-131071"}
                )
                resp2 = conn.getresponse()
                body2 = resp2.read()
                self.assertEqual(resp2.status, 206)
                self.assertFalse(resp2.will_close)
                self.assertEqual(body2, self.large_payload[65536:131072])
                self.assertEqual(len(write_sizes), 1)
            finally:
                conn.close()

        # Verify mid-stream truncation on cached readinto_at invalidates entry and closes connection.
        trunc_file = self.data_root / "trunc_test.rpf"
        trunc_file.write_bytes(b"ABCDEFGH")
        trunc_cache = serve_local.ReadOnlyFileHandleCache(max_entries=4)
        try:
            entry = trunc_cache.acquire(trunc_file, trunc_file.stat())
            self.assertIsNotNone(entry)
            assert entry is not None

            sent = bytearray()

            class _FakeSockWfile:
                _sock = object()

                def write(self, data) -> int:
                    sent.extend(data)
                    return len(data)

            handler = object.__new__(serve_local.PlayGta6Handler)
            handler.wfile = _FakeSockWfile()
            handler.close_connection = False
            handler._headers_sent = False
            handler._range_stream_chunk_bytes = 16
            handler._stream_chunk_bytes = None
            handler._stream_cached_file_range(
                entry,
                0,
                32,  # longer than the 8-byte file -> triggers mid-stream EOF
                trunc_cache,
                chunk_bytes=16,
                pending_headers=b"HTTP/1.1 206 Partial Content\r\n\r\n",
            )
            trunc_cache.release(entry)
            self.assertTrue(handler.close_connection)
            self.assertEqual(bytes(sent), b"HTTP/1.1 206 Partial Content\r\n\r\nABCDEFGH")
            self.assertEqual(trunc_cache.stats()["entries"], 0)
            self.assertEqual(trunc_cache.stats()["open_handles"], 0)
        finally:
            trunc_cache.close_all()

    def test_expanded_cacheable_suffixes_cold_miss_donation_and_global_fd_cap(self) -> None:
        # 1. Expanded suffixes enter ReadOnlyFileHandleCache; .html/.js still do not.
        for name in (
            "movie.bk2",
            "intro.bik",
            "bank.awc",
            "font.ttpf",
            "world.rpf9",
            "splash.reflect.json",
        ):
            self.assertTrue(serve_local.is_cacheable_file_path(Path(name)), msg=name)
        for name in ("index.html", "loader.js", "style.css", "data-manifest.json"):
            self.assertFalse(serve_local.is_cacheable_file_path(Path(name)), msg=name)

        reflect_file = self.data_root / "splash.reflect.json"
        reflect_file.write_bytes(b'{"version":1,"shaders":[]}')
        st1, hdr1, body1 = self._request(
            "GET", "/data/splash.reflect.json", {"Range": "bytes=0-11"}
        )
        self.assertEqual(st1, 206)
        self.assertEqual(body1, b'{"version":1')
        cache = self.httpd.file_handle_cache
        assert cache is not None
        self.assertGreaterEqual(cache.stats()["entries"], 1)

        reflect_file.write_bytes(b'{"version":2,"shaders":[1,2,3]}')
        st2, hdr2, body2 = self._request(
            "GET", "/data/splash.reflect.json", {"Range": "bytes=0-11"}
        )
        self.assertEqual(st2, 206)
        self.assertEqual(body2, b'{"version":2')
        self.assertNotEqual(hdr2.get("ETag"), hdr1.get("ETag"))

        # 2. Cold-miss FD donation on non-pread avoids opening & discarding redundant FDs.
        cold_file = self.data_root / "cold_herd.rpf"
        cold_payload = _make_deterministic_bytes(128 * 1024)
        cold_file.write_bytes(cold_payload)
        cold_st = cold_file.stat()

        with mock.patch.object(serve_local, "_HAS_PREAD", False):
            herd_cache = serve_local.ReadOnlyFileHandleCache(
                max_entries=4,
                max_concurrent_fds_per_file=4,
                max_total_open_fds=16,
            )
            open_calls = 0
            open_lock = threading.Lock()
            orig_open_fd = serve_local._open_readonly_fd
            open_barrier = threading.Barrier(4, timeout=5.0)
            use_barrier = threading.Barrier(4, timeout=5.0)

            def _counting_open(path: Path) -> int:
                nonlocal open_calls
                with open_lock:
                    open_calls += 1
                    idx = open_calls
                fd = orig_open_fd(path)
                if idx <= 4:
                    open_barrier.wait()
                return fd

            try:
                with mock.patch.object(serve_local, "_open_readonly_fd", side_effect=_counting_open):
                    def worker(i: int) -> bytes:
                        entry = herd_cache.acquire(cold_file, cold_st)
                        self.assertIsNotNone(entry)
                        assert entry is not None
                        try:
                            with entry.borrow_fd() as reader:
                                use_barrier.wait()
                                return reader.read_at(i * 4096, 4096)
                        finally:
                            herd_cache.release(entry)

                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                        results = list(pool.map(worker, range(4)))

                for i, chunk in enumerate(results):
                    self.assertEqual(chunk, cold_payload[i * 4096 : (i + 1) * 4096])
                self.assertEqual(
                    open_calls,
                    4,
                    f"Cold-miss donation should reuse all 4 opened FDs instead of reopening (got {open_calls})",
                )
                self.assertEqual(herd_cache.stats()["open_handles"], 4)
            finally:
                herd_cache.close_all()
            self.assertEqual(herd_cache.stats()["open_handles"], 0)

            # 3. Global max_total_open_fds trims idle pooled FDs across files without exceeding budget.
            budget_cache = serve_local.ReadOnlyFileHandleCache(
                max_entries=4,
                max_concurrent_fds_per_file=4,
                max_total_open_fds=6,
            )
            files = []
            for idx in range(4):
                p = self.data_root / f"budget_{idx}.rpf"
                p.write_bytes(_make_deterministic_bytes(64 * 1024))
                files.append((p, p.stat()))
            try:
                for p, st_f in files:
                    b = threading.Barrier(4, timeout=5.0)

                    def read_f(slot: int, path_obj=p, st_obj=st_f, bar=b) -> int:
                        entry = budget_cache.acquire(path_obj, st_obj)
                        assert entry is not None
                        try:
                            with entry.borrow_fd() as reader:
                                bar.wait()
                                return len(reader.read_at(slot * 1024, 1024))
                        finally:
                            budget_cache.release(entry)

                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                        lens = list(pool.map(read_f, range(4)))
                    self.assertEqual(lens, [1024] * 4)
                    self.assertLessEqual(budget_cache.stats()["open_handles"], 6)
            finally:
                budget_cache.close_all()
            self.assertEqual(budget_cache.stats()["open_handles"], 0)
            self.assertEqual(budget_cache.stats()["pooled_handles"], 0)

    def test_oversubscription_12_lanes_and_rfc_range_dump_disconnect_toctou_edge_cases(self) -> None:
        # 1. 12 concurrent keep-alive lanes oversubscribing an 8-FD pool on one .wgsl.pack.
        httpd = serve_local.create_server(
            "127.0.0.1",
            0,
            self.web_root,
            self.data_root,
            dump_root=self.dump_root,
            max_dump_file_bytes=256,
            max_concurrent_fds_per_file=8,
            quiet=True,
        )
        port = httpd.server_address[1]
        cache = httpd.file_handle_cache
        assert cache is not None
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            lanes = 12
            barrier = threading.Barrier(lanes, timeout=10.0)

            def oversub_lane(lane_idx: int) -> None:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
                try:
                    barrier.wait()
                    for step in range(8):
                        start = ((lane_idx * 65536) + (step * 16384)) % (self.large_size - 65536)
                        end = start + 65535
                        conn.request(
                            "GET", "/data/splash.wgsl.pack", headers={"Range": f"bytes={start}-{end}"}
                        )
                        resp = conn.getresponse()
                        body = resp.read()
                        self.assertEqual(resp.status, 206)
                        self.assertFalse(resp.will_close)
                        self.assertEqual(
                            hashlib.sha256(body).hexdigest(),
                            hashlib.sha256(self.large_payload[start : end + 1]).hexdigest(),
                        )
                finally:
                    conn.close()

            with concurrent.futures.ThreadPoolExecutor(max_workers=lanes) as pool:
                list(pool.map(oversub_lane, range(lanes)))

            # 2. RFC 9110 Range edge cases: unknown unit & multi-range -> 200 OK; 0-byte file -> 416.
            small = self.data_root / "small.rpf"
            small_data = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            small.write_bytes(small_data)
            empty_file = self.data_root / "empty.rpf"
            empty_file.write_bytes(b"")

            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            try:
                for range_hdr in ("items=0-10", "bytes=0-15, 32-47"):
                    conn.request("GET", "/data/small.rpf", headers={"Range": range_hdr})
                    resp = conn.getresponse()
                    body = resp.read()
                    hdrs = dict(resp.getheaders())
                    self.assertEqual(resp.status, 200, msg=range_hdr)
                    self.assertFalse(resp.will_close)
                    self._assert_coop_coep(hdrs)
                    self.assertEqual(body, small_data)

                conn.request("GET", "/data/empty.rpf", headers={"Range": "bytes=0-0"})
                resp_empty = conn.getresponse()
                body_empty = resp_empty.read()
                hdr_empty = dict(resp_empty.getheaders())
                self.assertEqual(resp_empty.status, 416)
                self.assertEqual(hdr_empty.get("Content-Range"), "bytes */0")
                self.assertEqual(body_empty, b"")
            finally:
                conn.close()

            # 3. Early POST /__pg6_dump/ error responses drain small bodies cleanly on loopback.
            for dump_path, body_bytes, expected_status in (
                ("/__pg6_dump/../escape.bin", b"X" * 4096, 400),
                ("/__pg6_dump/too_big.bin", b"Y" * 4096, 413),
            ):
                st_d, hdr_d, raw_d = self._request(
                    "POST", dump_path, port=port, body=body_bytes
                )
                self.assertEqual(st_d, expected_status)
                self._assert_coop_coep(hdr_d)
                self.assertFalse(json.loads(raw_d.decode("utf-8"))["ok"])

            # 4. TOCTOU fallback revalidation when cache.acquire() returns None after file mutates.
            toctou_file = self.data_root / "toctou.rpf"
            toctou_file.write_bytes(b"OLD-DATA-01234567")
            new_bytes = b"NEW-REPLACED-PAYLOAD-9876543210!"

            orig_acquire = cache.acquire

            def _mutate_and_miss(path: Path, st_obj: os.stat_result, *, realpath=None):
                if path.name == "toctou.rpf":
                    toctou_file.write_bytes(new_bytes)
                    return None
                return orig_acquire(path, st_obj, realpath=realpath)

            with mock.patch.object(cache, "acquire", side_effect=_mutate_and_miss):
                st_t, hdr_t, body_t = self._request(
                    "GET", "/data/toctou.rpf", {"Range": "bytes=0-15"}, port=port
                )
            self.assertEqual(st_t, 206)
            self.assertEqual(body_t, new_bytes[:16])
            self.assertEqual(hdr_t.get("Content-Range"), f"bytes 0-15/{len(new_bytes)}")
            self.assertEqual(hdr_t.get("ETag"), serve_local.compute_etag(toctou_file.stat()))
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=2)

        self.assertEqual(cache.stats()["open_handles"], 0)
        self.assertEqual(cache.stats()["pooled_handles"], 0)

        # 5. Extended Winsock/POSIX disconnect codes are classified quietly.
        ws_err = OSError("socket shutdown")
        ws_err.winerror = 10058  # WSAESHUTDOWN
        self.assertTrue(serve_local._is_client_disconnect_error(ws_err))
        if hasattr(errno, "ENOTCONN"):
            self.assertTrue(
                serve_local._is_client_disconnect_error(OSError(errno.ENOTCONN, "not connected"))
            )
        self.assertFalse(
            serve_local._is_client_disconnect_error(OSError(errno.EACCES, "permission denied"))
        )

    def test_shaders_interstage_static_and_dynamic_fallback_with_cache(self) -> None:
        shaders_dir = self.data_root / "shaders"
        shaders_dir.mkdir(exist_ok=True)
        outside_reflect = Path(self.tmp.name) / "escape.wgsl.pack.reflect.json"
        outside_reflect.write_text(
            json.dumps({
                "schema": "playgta6-stage-reflection",
                "version": 1,
                "stages": {
                    "0x0000000000000001": {
                        "stage": "vertex",
                        "bindings": [],
                        "vs_outputs": [{"semantic_reg": 1, "location": 0}],
                    }
                },
            }),
            encoding="utf-8",
        )

        # 1. Missing both .interstage.json and .reflect.json -> 404 + COOP/COEP
        st_miss, hdr_miss, body_miss = self._request(
            "GET", "/shaders/splash.wgsl.pack.interstage.json"
        )
        self.assertEqual(st_miss, 404)
        self._assert_coop_coep(hdr_miss)
        self.assertEqual(body_miss, b"")

        # Traversal must never resolve outside shaders_root
        st_esc, hdr_esc, _ = self._request(
            "GET", "/shaders/../escape.wgsl.pack.interstage.json"
        )
        self.assertEqual(st_esc, 404)
        self._assert_coop_coep(hdr_esc)

        # 2. Only splash.wgsl.pack.reflect.json exists -> dynamically extract compact v3 JSON & cache
        reflect_path = shaders_dir / "splash.wgsl.pack.reflect.json"
        reflect_doc = {
            "schema": "playgta6-stage-reflection",
            "version": 3,
            "stages": {
                "0x5fe7a88d9cb680b3": {
                    "stage": "vertex",
                    "bindings": [
                        {"group": 0, "binding": 0, "kind": "uniform",
                         "source": {"user_sgpr": 8, "table_byte_offset": 0}},
                    ],
                    "vertex_inputs": [
                        {"location": 0, "vb_slot": 0, "byte_offset": 0, "components": 3, "component_type": "f32"},
                    ],
                    "vs_outputs": [
                        {"semantic_reg": 1, "location": 0},
                        {"semantic_reg": 2, "location": 1},
                    ],
                },
                "0x56214b8d3b10a108": {
                    "stage": "fragment",
                    "bindings": [
                        {"group": 0, "binding": 0, "kind": "texture",
                         "source": {"user_sgpr": 8, "table_byte_offset": 0}},
                    ],
                    "ps_inputs": [
                        {"semantic_reg": 1, "location": 0, "mask": 0},
                        {"semantic_reg": 2, "location": 0, "mask": 3},
                    ],
                },
                "0x0000000000009999": {
                    "stage": "compute",
                    "bindings": [
                        {"group": 0, "binding": 0, "kind": "storage",
                         "source": {"user_sgpr": 4, "table_byte_offset": 0}},
                    ],
                },
                "0x0000000000008888": {
                    "stage": "vertex",
                    "bindings": [
                        {"group": 0, "binding": 0, "kind": "uniform",
                         "source": {"user_sgpr": 8, "table_byte_offset": 0}},
                    ],
                },
            },
        }
        reflect_raw = json.dumps(reflect_doc, indent=2)
        reflect_path.write_text(reflect_raw, encoding="utf-8")

        st1, hdr1, body1 = self._request("GET", "/shaders/splash.wgsl.pack.interstage.json")
        self.assertEqual(st1, 200)
        self._assert_coop_coep(hdr1)
        self.assertEqual(hdr1.get("Content-Type"), "application/json")
        self.assertEqual(hdr1.get("Content-Length"), str(len(body1)))
        self.assertLess(len(body1), len(reflect_raw.encode("utf-8")))
        parsed1 = json.loads(body1.decode("utf-8"))
        self.assertEqual(
            parsed1,
            {
                "schema": "playgta6-stage-reflection",
                "version": 3,
                "stages": {
                    "0x5fe7a88d9cb680b3": {
                        "stage": "vertex",
                        "bindings": [],
                        "vs_outputs": [
                            {"semantic_reg": 1, "location": 0},
                            {"semantic_reg": 2, "location": 1},
                        ],
                    },
                    "0x56214b8d3b10a108": {
                        "stage": "fragment",
                        "bindings": [],
                        "ps_inputs": [
                            {"semantic_reg": 1, "location": 0, "mask": 0},
                            {"semantic_reg": 2, "location": 0, "mask": 3},
                        ],
                    },
                },
            },
        )
        etag1 = hdr1.get("ETag")
        self.assertTrue(etag1)
        self.assertEqual(self.httpd.interstage_sidecar_cache.stats()["misses"], 1)
        self.assertEqual(self.httpd.interstage_sidecar_cache.stats()["hits"], 0)

        # Second GET hits in-memory cache without re-parsing .reflect.json; HEAD and If-None-Match 304 work
        with mock.patch.object(
            serve_local,
            "extract_interstage_sidecar_bytes",
            side_effect=AssertionError("expected interstage cache hit"),
        ):
            st2, hdr2, body2 = self._request("GET", "/shaders/splash.wgsl.pack.interstage.json")
            self.assertEqual(st2, 200)
            self.assertEqual(body2, body1)
            self.assertEqual(hdr2.get("ETag"), etag1)

            st_head, hdr_head, body_head = self._request(
                "HEAD", "/shaders/splash.wgsl.pack.interstage.json"
            )
            self.assertEqual(st_head, 200)
            self._assert_coop_coep(hdr_head)
            self.assertEqual(hdr_head.get("Content-Length"), str(len(body1)))
            self.assertEqual(body_head, b"")

            st_304, hdr_304, body_304 = self._request(
                "GET",
                "/shaders/splash.wgsl.pack.interstage.json",
                {"If-None-Match": etag1},
            )
            self.assertEqual(st_304, 304)
            self._assert_coop_coep(hdr_304)
            self.assertEqual(body_304, b"")
        self.assertGreaterEqual(self.httpd.interstage_sidecar_cache.stats()["hits"], 3)

        # Mutating .reflect.json (size/mtime_ns) invalidates cache and recomputes
        prev_st = reflect_path.stat()
        reflect_doc["stages"]["0x000000000000aaaa"] = {
            "stage": "fragment",
            "bindings": [],
            "ps_inputs": [{"semantic_reg": 4, "location": 0, "mask": 15}],
        }
        reflect_path.write_text(json.dumps(reflect_doc), encoding="utf-8")
        os.utime(reflect_path, ns=(prev_st.st_atime_ns, prev_st.st_mtime_ns + 2_000_000_000))
        st_upd, hdr_upd, body_upd = self._request(
            "GET", "/shaders/splash.wgsl.pack.interstage.json"
        )
        self.assertEqual(st_upd, 200)
        self.assertNotEqual(hdr_upd.get("ETag"), etag1)
        parsed_upd = json.loads(body_upd.decode("utf-8"))
        self.assertIn("0x000000000000aaaa", parsed_upd["stages"])
        self.assertEqual(self.httpd.interstage_sidecar_cache.stats()["misses"], 2)

        # 3. Malformed .reflect.json degrades cleanly to 404
        bad_reflect = shaders_dir / "broken.wgsl.pack.reflect.json"
        bad_reflect.write_text("{not valid json", encoding="utf-8")
        st_bad, hdr_bad, _ = self._request("GET", "/shaders/broken.wgsl.pack.interstage.json")
        self.assertEqual(st_bad, 404)
        self._assert_coop_coep(hdr_bad)

        # 4. Static .interstage.json on disk takes precedence over dynamic .reflect.json fallback
        static_path = shaders_dir / "splash.wgsl.pack.interstage.json"
        static_doc = {
            "schema": "playgta6-stage-reflection",
            "version": 3,
            "stages": {
                "0x000000000000feed": {
                    "stage": "vertex",
                    "bindings": [],
                    "vs_outputs": [{"semantic_reg": 7, "location": 2}],
                }
            },
        }
        static_bytes = json.dumps(static_doc, separators=(",", ":")).encode("utf-8")
        static_path.write_bytes(static_bytes)

        st_static, hdr_static, body_static = self._request(
            "GET", "/shaders/splash.wgsl.pack.interstage.json"
        )
        self.assertEqual(st_static, 200)
        self._assert_coop_coep(hdr_static)
        self.assertEqual(hdr_static.get("Content-Type"), "application/json")
        self.assertEqual(body_static, static_bytes)


if __name__ == "__main__":
    unittest.main()
