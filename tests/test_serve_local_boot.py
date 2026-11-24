"""Tests for W43/W52 local splash / shader / multi-module PGIM / game endpoints on serve_local.py."""

from __future__ import annotations

import contextlib
import http.client
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "server"))

import serve_local  # noqa: E402

SYNTHETIC_CONFIG = """<?xml version="1.0" encoding="utf-8"?>
<Game configVersion="1">
  <Identity Name="Synthetic.BootTitle" Publisher="CN=Synthetic" Version="1.0.0.0" />
  <TitleId>AABBCCDD</TitleId>
  <ExecutableList>
    <Executable Name="Game.exe" Id="Game" TargetDeviceFamily="Scarlett" />
  </ExecutableList>
  <ShellVisuals DefaultDisplayName="Synthetic Boot"
                SplashScreenImage="Assets\\xbs_splash.png"
                BackgroundColor="#000000" />
</Game>
"""

XXE_CONFIG = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<Game configVersion="1">
  <ShellVisuals SplashScreenImage="Assets\\xbs_splash.png" />
</Game>
"""


class TestServeLocalBoot(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.web_root = base / "web"
        self.data_root = base / "data"
        self.game_root = base / "game"
        self.shaders_root = self.data_root / "shaders"
        self.boot_root = base / "work" / "recomp_build"

        self.web_root.mkdir()
        self.data_root.mkdir()
        self.game_root.mkdir()
        self.shaders_root.mkdir(parents=True)
        self.boot_root.mkdir(parents=True)

        (self.web_root / "index.html").write_text("<h1>boot</h1>", encoding="utf-8")
        (self.data_root / "data-manifest.json").write_text("{}", encoding="utf-8")
        (self.data_root / "bootset.json").write_text("[]", encoding="utf-8")

        (self.game_root / "MicrosoftGame.Config").write_text(SYNTHETIC_CONFIG, encoding="utf-8")
        assets = self.game_root / "Assets"
        assets.mkdir()
        self.splash_bytes = b"\x89PNG\r\n\x1a\n" + b"splash-bytes-0123456789"
        (assets / "xbs_splash.png").write_bytes(self.splash_bytes)
        x64_dir = self.game_root / "x64"
        x64_dir.mkdir()
        self.rpf_bytes = b"RPF9" + bytes(range(64)) + b"synthetic-rpf-tail"
        (x64_dir / "common.rpf").write_bytes(self.rpf_bytes)
        self.dll_bytes = b"MZ\x90\x00synthetic-dll-payload-0123456789"
        (self.game_root / "runtime_dep.dll").write_bytes(self.dll_bytes)

        self.pack_bytes = b"WGPK" + b"pack-payload-0123456789abcdef"
        (self.shaders_root / "other.wgsl.pack").write_bytes(b"WGPK" + b"other")
        (self.shaders_root / "fallback.wgsl.pack").write_bytes(self.pack_bytes)
        (self.shaders_root / "reflection-manifest.json").write_text(
            '{"schema_version":6}', encoding="utf-8"
        )

        self.pgim_bytes = b"PGIM" + b"guest-image-0123456789abcdef"
        (self.boot_root / "guest.pgim").write_bytes(self.pgim_bytes)
        (self.boot_root / "meta.json").write_text('{"ok":true}', encoding="utf-8")
        (self.boot_root / "helper.js").write_text("export default 1;\n", encoding="utf-8")
        (self.boot_root / "runtime.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00")
        (self.boot_root / "evil.exe").write_bytes(b"MZ-disallowed")
        (self.boot_root / "lifted.ll").write_text("; disallowed\n", encoding="utf-8")
        (self.boot_root / ".env").write_text("SECRET=1\n", encoding="utf-8")

        # Outside roots used for traversal / symlink escape attempts.
        self.secret_outside = base / "outside_secret.bin"
        self.secret_outside.write_bytes(b"TOP-SECRET-BYTES")

        self.httpd = serve_local.create_server(
            "127.0.0.1",
            0,
            self.web_root,
            self.data_root,
            game_root=self.game_root,
            shaders_root=self.shaders_root,
            boot_root=self.boot_root,
        )
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)
        self.tmp.cleanup()

    def _request(self, method: str, path: str, headers: dict[str, str] | None = None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request(method, path, headers=headers or {})
            resp = conn.getresponse()
            body = resp.read()
            return resp.status, dict(resp.getheaders()), body
        finally:
            conn.close()

    def _assert_isolation_headers(self, headers: dict[str, str]) -> None:
        self.assertEqual(headers.get("Cross-Origin-Opener-Policy"), "same-origin")
        self.assertEqual(headers.get("Cross-Origin-Embedder-Policy"), "require-corp")
        self.assertEqual(headers.get("Cross-Origin-Resource-Policy"), "same-origin")
        self.assertEqual(headers.get("Accept-Ranges"), "bytes")

    def _assert_no_host_leak(self, raw_text: str) -> None:
        sensitive = [
            str(self.web_root),
            str(self.data_root),
            str(self.game_root),
            str(self.boot_root),
            str(Path(self.tmp.name)),
            str(REPO_ROOT),
            os.environ.get("USERNAME", ""),
            os.environ.get("USER", ""),
            os.environ.get("LOGNAME", ""),
        ]
        for token in sensitive:
            if token and len(token) > 2:
                self.assertNotIn(token, raw_text)
                self.assertNotIn(token.replace("\\", "/"), raw_text)
        self.assertNotRegex(raw_text, r"[A-Za-z]:\\")
        self.assertNotRegex(raw_text, r"/(?:Users|home|tmp|var|opt|root)/")

    def test_api_status_discovers_boot_urls_without_host_paths(self) -> None:
        status, headers, body = self._request("GET", "/api/status")
        self.assertEqual(status, 200)
        self._assert_isolation_headers(headers)
        self.assertEqual(headers.get("Content-Type"), "application/json")
        self.assertEqual(headers.get("Cache-Control"), "no-store")

        raw_text = body.decode("utf-8")
        data = json.loads(raw_text)

        # Legacy fields remain intact.
        for key in (
            "ok",
            "coopCoep",
            "dataDirConfigured",
            "hasManifest",
            "hasBootset",
            "hasWasmBinary",
            "shaderPackCount",
        ):
            self.assertIn(key, data)
        self.assertTrue(data["ok"])
        self.assertTrue(data["coopCoep"])
        self.assertTrue(data["dataDirConfigured"])
        self.assertTrue(data["hasManifest"])
        self.assertTrue(data["hasBootset"])
        self.assertGreaterEqual(data["shaderPackCount"], 1)

        # New boot-discovery fields (relative URLs only).
        self.assertEqual(data["splashImage"], "/game/Assets/xbs_splash.png")
        self.assertEqual(data["shaderPackUrl"], "/shaders/fallback.wgsl.pack")
        self.assertEqual(data["reflectionManifestUrl"], "/shaders/reflection-manifest.json")
        self.assertEqual(data["pgimUrl"], "/boot/guest.pgim")
        self.assertIsNone(data["programManifestUrl"])
        self.assertEqual(data["gameBase"], "/game/")

        self._assert_no_host_leak(raw_text)

    def test_api_status_null_when_assets_missing(self) -> None:
        # Empty synthetic dirs -> null discovery fields, legacy keys still present.
        empty = Path(self.tmp.name) / "empty"
        web = empty / "web"
        data = empty / "data"
        shaders = data / "shaders"
        boot = empty / "boot"
        game = empty / "game"
        for path in (web, data, shaders, boot, game):
            path.mkdir(parents=True)
        (web / "index.html").write_text("x", encoding="utf-8")

        httpd = serve_local.create_server(
            "127.0.0.1",
            0,
            web,
            data,
            game_root=game,
            shaders_root=shaders,
            boot_root=boot,
        )
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            try:
                conn.request("GET", "/api/status")
                resp = conn.getresponse()
                raw = resp.read().decode("utf-8")
                self.assertEqual(resp.status, 200)
            finally:
                conn.close()
            data = json.loads(raw)
            self.assertIsNone(data["splashImage"])
            self.assertIsNone(data["shaderPackUrl"])
            self.assertIsNone(data["reflectionManifestUrl"])
            self.assertIsNone(data["pgimUrl"])
            self.assertIsNone(data["programManifestUrl"])
            self.assertEqual(data["gameBase"], "/game/")
            self._assert_no_host_leak(raw)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=2)

    def test_api_status_game_base_null_when_game_root_absent(self) -> None:
        payload_none = serve_local.build_status_payload(
            self.web_root,
            self.data_root,
            game_root=None,
            shaders_root=self.shaders_root,
            boot_root=self.boot_root,
        )
        self.assertIsNone(payload_none["gameBase"])
        self.assertIsNone(payload_none["splashImage"])

        missing_game = Path(self.tmp.name) / "does_not_exist_game"
        payload_missing = serve_local.build_status_payload(
            self.web_root,
            self.data_root,
            game_root=missing_game,
            shaders_root=self.shaders_root,
            boot_root=self.boot_root,
        )
        self.assertIsNone(payload_missing["gameBase"])
        self.assertIsNone(payload_missing["splashImage"])
        self._assert_no_host_leak(json.dumps(payload_missing))

    def test_xxe_config_does_not_set_splash_image(self) -> None:
        (self.game_root / "MicrosoftGame.Config").write_text(XXE_CONFIG, encoding="utf-8")
        status, _, body = self._request("GET", "/api/status")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertIsNone(data["splashImage"])

    def test_shaders_and_boot_serve_200_206_etag_coop_coep(self) -> None:
        for path, expected in (
            ("/shaders/fallback.wgsl.pack", self.pack_bytes),
            ("/boot/guest.pgim", self.pgim_bytes),
            ("/game/Assets/xbs_splash.png", self.splash_bytes),
        ):
            with self.subTest(path=path, mode="full"):
                status, headers, body = self._request("GET", path)
                self.assertEqual(status, 200)
                self._assert_isolation_headers(headers)
                self.assertTrue(headers.get("ETag"))
                self.assertTrue(headers.get("Last-Modified"))
                self.assertEqual(body, expected)

            with self.subTest(path=path, mode="range"):
                status, headers, body = self._request("GET", path, {"Range": "bytes=2-7"})
                self.assertEqual(status, 206)
                self._assert_isolation_headers(headers)
                self.assertTrue(headers.get("ETag"))
                self.assertEqual(headers.get("Content-Range"), f"bytes 2-7/{len(expected)}")
                self.assertEqual(body, expected[2:8])

            with self.subTest(path=path, mode="head"):
                status, headers, body = self._request("HEAD", path)
                self.assertEqual(status, 200)
                self._assert_isolation_headers(headers)
                self.assertTrue(headers.get("ETag"))
                self.assertEqual(body, b"")

    def test_boot_blocks_disallowed_extensions(self) -> None:
        for path in ("/boot/evil.exe", "/boot/lifted.ll", "/boot/.env"):
            with self.subTest(path=path):
                status, headers, body = self._request("GET", path)
                self.assertIn(status, (403, 404))
                self._assert_isolation_headers(headers)
                self.assertEqual(body, b"")
                # Disallowed extension that resolves inside boot root must be 403.
                if path in ("/boot/evil.exe", "/boot/lifted.ll"):
                    self.assertEqual(status, 403)

    def test_boot_and_shaders_block_path_traversal(self) -> None:
        outside_name = self.secret_outside.name
        parent_attempts = [
            f"/boot/../{outside_name}",
            "/boot/../../README.md",
            "/boot/%2e%2e/" + outside_name,
            f"/shaders/../{outside_name}",
            "/shaders/../../README.md",
            f"/game/../{outside_name}",
            "/data/../../README.md",
        ]
        for path in parent_attempts:
            with self.subTest(path=path):
                status, headers, body = self._request("GET", path)
                self.assertEqual(status, 404)
                self._assert_isolation_headers(headers)
                self.assertEqual(body, b"")
                self.assertNotIn(b"TOP-SECRET", body)

    def test_boot_allows_safe_extensions(self) -> None:
        for path in ("/boot/meta.json", "/boot/helper.js", "/boot/runtime.wasm"):
            with self.subTest(path=path):
                status, headers, body = self._request("GET", path)
                self.assertEqual(status, 200)
                self._assert_isolation_headers(headers)
                self.assertTrue(headers.get("ETag"))
                self.assertGreater(len(body), 0)

    def test_legacy_routes_still_work(self) -> None:
        status, headers, body = self._request("GET", "/")
        self.assertEqual(status, 200)
        self._assert_isolation_headers(headers)
        self.assertIn(b"boot", body)

        status, headers, body = self._request("GET", "/data/data-manifest.json")
        self.assertEqual(status, 200)
        self._assert_isolation_headers(headers)
        self.assertEqual(body, b"{}")

    def test_game_rpf_and_assets_range_etag_conditional_and_traversal(self) -> None:
        # Verify GET, HEAD, 206 Range, ETag, If-None-Match (304), If-Range, and COOP/COEP
        # across .rpf, .dll, .png, and MicrosoftGame.Config under /game/.
        assets = (
            ("/game/x64/common.rpf", self.rpf_bytes),
            ("/game/runtime_dep.dll", self.dll_bytes),
            ("/game/Assets/xbs_splash.png", self.splash_bytes),
            ("/game/MicrosoftGame.Config", (self.game_root / "MicrosoftGame.Config").read_bytes()),
        )
        for path, expected in assets:
            with self.subTest(path=path, mode="get"):
                status, headers, body = self._request("GET", path)
                self.assertEqual(status, 200)
                self._assert_isolation_headers(headers)
                etag = headers.get("ETag")
                last_modified = headers.get("Last-Modified")
                self.assertTrue(etag)
                self.assertTrue(last_modified)
                self.assertEqual(headers.get("Content-Length"), str(len(expected)))
                self.assertEqual(body, expected)

            with self.subTest(path=path, mode="head"):
                status, headers, body = self._request("HEAD", path)
                self.assertEqual(status, 200)
                self._assert_isolation_headers(headers)
                self.assertEqual(headers.get("ETag"), etag)
                self.assertEqual(headers.get("Content-Length"), str(len(expected)))
                self.assertEqual(body, b"")

            with self.subTest(path=path, mode="range_206"):
                status, headers, body = self._request("GET", path, {"Range": "bytes=4-15"})
                self.assertEqual(status, 206)
                self._assert_isolation_headers(headers)
                self.assertEqual(headers.get("ETag"), etag)
                self.assertEqual(headers.get("Content-Range"), f"bytes 4-15/{len(expected)}")
                self.assertEqual(headers.get("Content-Length"), "12")
                self.assertEqual(body, expected[4:16])

            with self.subTest(path=path, mode="if_none_match_304"):
                status, headers, body = self._request("GET", path, {"If-None-Match": etag})
                self.assertEqual(status, 304)
                self._assert_isolation_headers(headers)
                self.assertEqual(headers.get("ETag"), etag)
                self.assertEqual(body, b"")

            with self.subTest(path=path, mode="if_range_match_and_mismatch"):
                status_ok, headers_ok, body_ok = self._request(
                    "GET", path, {"Range": "bytes=0-7", "If-Range": etag}
                )
                self.assertEqual(status_ok, 206)
                self._assert_isolation_headers(headers_ok)
                self.assertEqual(headers_ok.get("Content-Range"), f"bytes 0-7/{len(expected)}")
                self.assertEqual(body_ok, expected[:8])

                status_stale, headers_stale, body_stale = self._request(
                    "GET", path, {"Range": "bytes=0-7", "If-Range": '"stale-etag"'}
                )
                self.assertEqual(status_stale, 200)
                self._assert_isolation_headers(headers_stale)
                self.assertIsNone(headers_stale.get("Content-Range"))
                self.assertEqual(body_stale, expected)

        # Suffix, open-ended, and unsatisfiable ranges on synthetic .rpf
        status, headers, body = self._request("GET", "/game/x64/common.rpf", {"Range": "bytes=-8"})
        self.assertEqual(status, 206)
        self._assert_isolation_headers(headers)
        self.assertEqual(body, self.rpf_bytes[-8:])

        status, headers, body = self._request("GET", "/game/x64/common.rpf", {"Range": "bytes=999999-"})
        self.assertEqual(status, 416)
        self._assert_isolation_headers(headers)
        self.assertEqual(headers.get("Content-Range"), f"bytes */{len(self.rpf_bytes)}")
        self.assertEqual(body, b"")

        # Strict traversal rejection on /game/, including .. collapsing into existing files.
        traversal_paths = (
            f"/game/../{self.secret_outside.name}",
            "/game/../MicrosoftGame.Config",
            "/game/Assets/../MicrosoftGame.Config",
            "/game/x64/../../" + self.secret_outside.name,
            "/game/%2e%2e/MicrosoftGame.Config",
            "/game/..%5cMicrosoftGame.Config",
            "/game/x64/%2e%2e/%2e%2e/" + self.secret_outside.name,
        )
        for bad_path in traversal_paths:
            with self.subTest(bad_path=bad_path):
                status, headers, body = self._request("GET", bad_path)
                self.assertIn(status, (403, 404))
                self._assert_isolation_headers(headers)
                self.assertEqual(body, b"")
                self.assertNotIn(b"TOP-SECRET", body)
                self.assertNotIn(b"Synthetic.BootTitle", body)


class TestServeLocalMultiModulePgim(unittest.TestCase):
    """Unit and live HTTP tests for W52 recursive `.pgim` + `programManifestUrl` + `gameBase`."""

    def test_discover_pgim_url_priority_unit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)

            # Empty / missing returns None.
            self.assertIsNone(serve_local._discover_pgim_url(None))
            self.assertIsNone(serve_local._discover_pgim_url(base / "missing"))
            boot = base / "boot"
            boot.mkdir()
            self.assertIsNone(serve_local._discover_pgim_url(boot))

            # Nested non-main modules only -> lexicographical winner.
            (boot / "vcruntime140" / "image").mkdir(parents=True)
            (boot / "msvcp140" / "image").mkdir(parents=True)
            (boot / "vcruntime140" / "image" / "guest.pgim").write_bytes(b"PGIM-vc")
            (boot / "msvcp140" / "image" / "guest.pgim").write_bytes(b"PGIM-msvcp")
            self.assertEqual(
                serve_local._discover_pgim_url(boot),
                "/boot/msvcp140/image/guest.pgim",
            )

            # Top-level flat .pgim beats nested non-main modules (sorted lexicographically).
            (boot / "zeta.pgim").write_bytes(b"PGIM-zeta")
            (boot / "alpha.pgim").write_bytes(b"PGIM-alpha")
            self.assertEqual(serve_local._discover_pgim_url(boot), "/boot/alpha.pgim")

            # Any path under main/ beats top-level and other nested modules.
            (boot / "main").mkdir(parents=True)
            (boot / "main" / "custom.pgim").write_bytes(b"PGIM-main-custom")
            self.assertEqual(serve_local._discover_pgim_url(boot), "/boot/main/custom.pgim")

            # Exact main/image/guest.pgim beats all other main/* entries.
            (boot / "main" / "image").mkdir(parents=True)
            (boot / "main" / "image" / "guest.pgim").write_bytes(b"PGIM-main-image")
            self.assertEqual(
                serve_local._discover_pgim_url(boot),
                "/boot/main/image/guest.pgim",
            )

    def test_discover_program_manifest_url_unit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            boot = base / "boot"
            boot.mkdir()
            self.assertIsNone(serve_local._discover_program_manifest_url(None))
            self.assertIsNone(serve_local._discover_program_manifest_url(base / "missing"))
            self.assertIsNone(serve_local._discover_program_manifest_url(boot))

            (boot / "program-manifest.json").write_text(
                '{"schema_version":1,"modules":[]}', encoding="utf-8"
            )
            self.assertEqual(
                serve_local._discover_program_manifest_url(boot),
                "/boot/program-manifest.json",
            )

    def test_live_server_nested_multi_module_pgim_and_program_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web = base / "web"
            data = base / "data"
            boot = base / "work" / "recomp_build"
            web.mkdir()
            data.mkdir()
            boot.mkdir(parents=True)
            (web / "index.html").write_text("<h1>multi</h1>", encoding="utf-8")

            main_pgim = b"PGIM" + b"main-module-payload-0123456789abcdef"
            msvcp_pgim = b"PGIM" + b"msvcp140-payload-fedcba9876543210"
            manifest_bytes = json.dumps(
                {
                    "schema_version": 1,
                    "modules": [
                        {"name": "main", "pgim": "main/image/guest.pgim"},
                        {"name": "msvcp140", "pgim": "msvcp140/image/guest.pgim"},
                    ],
                }
            ).encode("utf-8")

            (boot / "main" / "image").mkdir(parents=True)
            (boot / "msvcp140" / "image").mkdir(parents=True)
            (boot / "main" / "image" / "guest.pgim").write_bytes(main_pgim)
            (boot / "msvcp140" / "image" / "guest.pgim").write_bytes(msvcp_pgim)
            (boot / "main" / "lifted.ll").write_text("; disallowed nested IR\n", encoding="utf-8")
            (boot / "program-manifest.json").write_bytes(manifest_bytes)

            httpd = serve_local.create_server(
                "127.0.0.1",
                0,
                web,
                data,
                game_root=None,
                boot_root=boot,
            )
            port = httpd.server_address[1]
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                def req(method: str, path: str, headers: dict[str, str] | None = None):
                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                    try:
                        conn.request(method, path, headers=headers or {})
                        resp = conn.getresponse()
                        body = resp.read()
                        return resp.status, dict(resp.getheaders()), body
                    finally:
                        conn.close()

                # /api/status discovers nested main/image/guest.pgim + program-manifest.json, gameBase=None.
                status, headers, body = req("GET", "/api/status")
                self.assertEqual(status, 200)
                self.assertEqual(headers.get("Cross-Origin-Opener-Policy"), "same-origin")
                self.assertEqual(headers.get("Cross-Origin-Embedder-Policy"), "require-corp")
                raw = body.decode("utf-8")
                payload = json.loads(raw)
                self.assertEqual(payload["pgimUrl"], "/boot/main/image/guest.pgim")
                self.assertEqual(payload["programManifestUrl"], "/boot/program-manifest.json")
                self.assertIsNone(payload["gameBase"])
                self.assertNotIn(str(boot), raw)
                self.assertNotIn(str(base), raw)

                # Serve nested main/image/guest.pgim with 200, 206 Range, ETag, and COOP/COEP.
                st_full, hdr_full, body_full = req("GET", "/boot/main/image/guest.pgim")
                self.assertEqual(st_full, 200)
                self.assertEqual(hdr_full.get("Cross-Origin-Opener-Policy"), "same-origin")
                self.assertEqual(hdr_full.get("Cross-Origin-Embedder-Policy"), "require-corp")
                self.assertEqual(hdr_full.get("Cross-Origin-Resource-Policy"), "same-origin")
                self.assertTrue(hdr_full.get("ETag"))
                self.assertEqual(body_full, main_pgim)

                st_rng, hdr_rng, body_rng = req(
                    "GET", "/boot/main/image/guest.pgim", {"Range": "bytes=4-15"}
                )
                self.assertEqual(st_rng, 206)
                self.assertEqual(hdr_rng.get("Cross-Origin-Opener-Policy"), "same-origin")
                self.assertEqual(hdr_rng.get("Cross-Origin-Embedder-Policy"), "require-corp")
                self.assertEqual(
                    hdr_rng.get("Content-Range"), f"bytes 4-15/{len(main_pgim)}"
                )
                self.assertEqual(body_rng, main_pgim[4:16])

                # Serve nested secondary module and program-manifest.json.
                st_dep, _, body_dep = req("GET", "/boot/msvcp140/image/guest.pgim")
                self.assertEqual(st_dep, 200)
                self.assertEqual(body_dep, msvcp_pgim)

                st_man, hdr_man, body_man = req("GET", "/boot/program-manifest.json")
                self.assertEqual(st_man, 200)
                self.assertEqual(hdr_man.get("Content-Type"), "application/json")
                self.assertEqual(body_man, manifest_bytes)

                # Nested disallowed extension still rejected with 403.
                st_bad, _, body_bad = req("GET", "/boot/main/lifted.ll")
                self.assertEqual(st_bad, 403)
                self.assertEqual(body_bad, b"")
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join(timeout=2)


class TestServeLocalBootSymlink(unittest.TestCase):
    def test_boot_rejects_external_symlink(self) -> None:
        if os.name == "nt":
            # Creating symlinks on Windows often requires elevated privileges.
            self.skipTest("symlink escape coverage is POSIX-oriented")

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web = base / "web"
            data = base / "data"
            boot = base / "boot"
            web.mkdir()
            data.mkdir()
            boot.mkdir()
            (web / "index.html").write_text("x", encoding="utf-8")

            outside = base / "outside.env"
            outside.write_text("SECRET=1\n", encoding="utf-8")
            link = boot / "linked.env"
            try:
                link.symlink_to(outside)
            except OSError as exc:
                self.skipTest(f"unable to create symlink: {exc}")

            httpd = serve_local.create_server(
                "127.0.0.1",
                0,
                web,
                data,
                boot_root=boot,
            )
            port = httpd.server_address[1]
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                try:
                    conn.request("GET", "/boot/linked.env")
                    resp = conn.getresponse()
                    body = resp.read()
                    status = resp.status
                finally:
                    conn.close()
                self.assertIn(status, (403, 404))
                self.assertNotIn(b"SECRET", body)
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join(timeout=2)

    def test_external_symlinks_rejected_for_pgim_manifest_and_game(self) -> None:
        if os.name == "nt":
            self.skipTest("symlink escape coverage is POSIX-oriented")

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web = base / "web"
            data = base / "data"
            boot = base / "boot"
            game = base / "game"
            for d in (web, data, boot, game):
                d.mkdir()
            (web / "index.html").write_text("x", encoding="utf-8")

            outside_pgim = base / "outside.pgim"
            outside_pgim.write_bytes(b"PGIM-OUTSIDE-SECRET")
            outside_manifest = base / "outside.json"
            outside_manifest.write_text('{"secret":true}', encoding="utf-8")
            outside_rpf = base / "outside.rpf"
            outside_rpf.write_bytes(b"RPF9-OUTSIDE-SECRET")

            (boot / "main" / "image").mkdir(parents=True)
            try:
                (boot / "main" / "image" / "guest.pgim").symlink_to(outside_pgim)
                (boot / "program-manifest.json").symlink_to(outside_manifest)
                (game / "escaped.rpf").symlink_to(outside_rpf)
            except OSError as exc:
                self.skipTest(f"unable to create symlink: {exc}")

            self.assertIsNone(serve_local._discover_pgim_url(boot))
            self.assertIsNone(serve_local._discover_program_manifest_url(boot))

            httpd = serve_local.create_server(
                "127.0.0.1",
                0,
                web,
                data,
                game_root=game,
                boot_root=boot,
            )
            port = httpd.server_address[1]
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                for path in (
                    "/boot/main/image/guest.pgim",
                    "/boot/program-manifest.json",
                    "/game/escaped.rpf",
                ):
                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                    try:
                        conn.request("GET", path)
                        resp = conn.getresponse()
                        body = resp.read()
                        self.assertIn(resp.status, (403, 404))
                        self.assertNotIn(b"SECRET", body)
                    finally:
                        conn.close()
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join(timeout=2)


class TestServeLocalQuietDisconnects(unittest.TestCase):
    """Tests for W59 quiet handling of client-disconnect socket errors."""

    def test_create_server_uses_quiet_threading_http_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web = base / "web"
            data = base / "data"
            web.mkdir()
            data.mkdir()
            httpd = serve_local.create_server("127.0.0.1", 0, web, data)
            try:
                self.assertIsInstance(httpd, serve_local.QuietThreadingHTTPServer)
            finally:
                httpd.server_close()

    def test_quiet_server_handle_error_suppresses_disconnects_and_reports_others(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web = base / "web"
            data = base / "data"
            web.mkdir()
            data.mkdir()
            httpd = serve_local.create_server("127.0.0.1", 0, web, data)
            try:
                for exc_cls in (
                    ConnectionAbortedError,
                    ConnectionResetError,
                    BrokenPipeError,
                ):
                    with self.subTest(exc=exc_cls.__name__):
                        buf = io.StringIO()
                        with contextlib.redirect_stderr(buf):
                            try:
                                raise exc_cls("client cancelled range request")
                            except Exception:
                                httpd.handle_error(None, ("127.0.0.1", 54321))
                        self.assertEqual(buf.getvalue(), "")

                buf = io.StringIO()
                with contextlib.redirect_stderr(buf):
                    try:
                        raise RuntimeError("unexpected server failure")
                    except Exception:
                        httpd.handle_error(None, ("127.0.0.1", 54321))
                stderr_output = buf.getvalue()
                self.assertIn("RuntimeError", stderr_output)
                self.assertIn("unexpected server failure", stderr_output)
            finally:
                httpd.server_close()

    def test_serve_file_handles_disconnect_during_chunk_write_quietly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web = base / "web"
            data = base / "data"
            web.mkdir()
            data.mkdir()
            payload = b"0123456789abcdef" * 1024
            (data / "chunk.bin").write_bytes(payload)

            for exc_cls in (
                ConnectionAbortedError,
                ConnectionResetError,
                BrokenPipeError,
            ):
                with self.subTest(exc=exc_cls.__name__):
                    class _AbortingWriter:
                        def write(self, _data: bytes) -> int:
                            raise exc_cls("peer closed socket")

                    handler = object.__new__(serve_local.PlayGta6Handler)
                    handler.web_root = web
                    handler.data_root = data
                    handler.game_root = None
                    handler.shaders_root = data / "shaders"
                    handler.boot_root = None
                    handler.headers = {}
                    handler.wfile = _AbortingWriter()
                    handler.send_response = lambda *a, **kw: None
                    handler.send_header = lambda *a, **kw: None
                    handler.end_headers = lambda *a, **kw: None

                    # Should return quietly without propagating socket disconnect errors.
                    handler._serve_file("/data/chunk.bin", send_body=True)


if __name__ == "__main__":
    unittest.main()
