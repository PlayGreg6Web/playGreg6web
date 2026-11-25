import concurrent.futures
from functools import partial
import hashlib
import http.client
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest

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


if __name__ == "__main__":
    unittest.main()
