#!/usr/bin/env python3
"""Unit and CLI tests for tools/stage_browser_boot.py (Track 5 / W51)."""

from __future__ import annotations

import http.client
import io
import json
import os
import re
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "server"))

import check_wasm_exports as wasm  # noqa: E402
import serve_local  # noqa: E402
import stage_browser_boot as sbb  # noqa: E402


def _make_repo_skeleton(base: Path) -> Path:
    repo = base / "repo"
    (repo / "web").mkdir(parents=True)
    (repo / "web" / "index.html").write_text("<h1>boot</h1>", encoding="utf-8")
    (repo / "tools").mkdir()
    (repo / "runtime").mkdir()
    (repo / "work").mkdir()
    (repo / "local-data").mkdir()
    return repo


def _populate_program_dir(program_dir: Path) -> None:
    """Create a realistic multi-module build-program tree with allowlisted & excluded files."""
    program_dir.mkdir(parents=True, exist_ok=True)
    (program_dir / "program-manifest.json").write_text(
        json.dumps({"schema": "playgta6-recomp-program", "modules": ["main", "helper"]}),
        encoding="utf-8",
    )
    (program_dir / "pg6_program.cpp").write_text("// excluded cpp\n", encoding="utf-8")
    (program_dir / "notes.txt").write_text("excluded text\n", encoding="utf-8")
    (program_dir / "debug.json").write_text('{"excluded": true}\n', encoding="utf-8")

    main_dir = program_dir / "main"
    (main_dir / "image").mkdir(parents=True)
    (main_dir / "shards").mkdir(parents=True)
    (main_dir / "build-manifest.json").write_text(
        json.dumps({"schema": "playgta6-recomp-build", "module": "main"}),
        encoding="utf-8",
    )
    (main_dir / "image" / "guest.pgim").write_bytes(b"PGIM" + b"\x01" * 48)
    (main_dir / "shards" / "shard-000.ll").write_text("; excluded llvm ir\n", encoding="utf-8")
    (main_dir / "shards" / "shard-000.o").write_bytes(b"\x7fELF" + b"\x00" * 16)
    (main_dir / "registration.cpp").write_text("// excluded registration\n", encoding="utf-8")

    helper_dir = program_dir / "helper" / "image"
    helper_dir.mkdir(parents=True)
    (helper_dir / "helper.pgim").write_bytes(b"PGIM" + b"\x02" * 32)
    (program_dir / "helper" / "build-manifest.json").write_text(
        json.dumps({"schema": "playgta6-recomp-build", "module": "helper"}),
        encoding="utf-8",
    )


def _write_wasm_pair(
    wasm_dir: Path,
    wasm_bytes: bytes,
    js_text: str = "export default function createRuntimeModule() {}\n",
) -> None:
    wasm_dir.mkdir(parents=True, exist_ok=True)
    (wasm_dir / "runtime.wasm").write_bytes(wasm_bytes)
    (wasm_dir / "runtime.js").write_text(js_text, encoding="utf-8")


def _can_create_symlinks(tmp_dir: Path) -> bool:
    target = tmp_dir / "_symlink_probe_target"
    link = tmp_dir / "_symlink_probe_link"
    target.write_text("ok", encoding="utf-8")
    try:
        os.symlink(target, link)
        return True
    except (OSError, NotImplementedError):
        return False
    finally:
        link.unlink(missing_ok=True)
        target.unlink(missing_ok=True)


class TestBootArtifactStaging(unittest.TestCase):
    def test_allowlisted_staging_preserves_relative_paths_and_skips_ir_obj_cpp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            program_dir = repo / "work" / "recomp_program"
            boot_dir = repo / "work" / "recomp_build"
            _populate_program_dir(program_dir)

            report = sbb.stage_browser_boot(
                program_dir=program_dir,
                boot_dir=boot_dir,
                repo_root=repo,
            )
            self.assertTrue(report["ok"])
            self.assertEqual(report["boot"]["status"], "staged")
            self.assertEqual(report["boot"]["discovered_count"], 5)
            self.assertEqual(report["boot"]["copied_count"], 5)
            self.assertEqual(report["boot"]["unchanged_count"], 0)
            self.assertEqual(report["boot"]["skipped_non_allowlisted"], 6)
            self.assertEqual(report["wasm"]["status"], "skipped")

            expected_files = {
                "program-manifest.json",
                "main/build-manifest.json",
                "main/image/guest.pgim",
                "helper/build-manifest.json",
                "helper/image/helper.pgim",
            }
            staged_files = {
                p.relative_to(boot_dir).as_posix()
                for p in boot_dir.rglob("*")
                if p.is_file()
            }
            self.assertEqual(staged_files, expected_files)
            self.assertEqual(
                (boot_dir / "main" / "image" / "guest.pgim").read_bytes(),
                b"PGIM" + b"\x01" * 48,
            )

    def test_incremental_staging_skips_unchanged_and_updates_on_size_or_sha256_change(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            program_dir = repo / "work" / "recomp_program"
            boot_dir = repo / "work" / "recomp_build"
            _populate_program_dir(program_dir)

            first = sbb.stage_browser_boot(
                program_dir=program_dir,
                boot_dir=boot_dir,
                repo_root=repo,
            )
            self.assertTrue(first["ok"])
            self.assertEqual(first["boot"]["copied_count"], 5)

            # Second run: everything unchanged.
            second = sbb.stage_browser_boot(
                program_dir=program_dir,
                boot_dir=boot_dir,
                repo_root=repo,
            )
            self.assertTrue(second["ok"])
            self.assertEqual(second["boot"]["status"], "unchanged")
            self.assertEqual(second["boot"]["copied_count"], 0)
            self.assertEqual(second["boot"]["unchanged_count"], 5)

            # Same-size content modification (SHA-256 changed, even if mtime preserved).
            guest_src = program_dir / "main" / "image" / "guest.pgim"
            orig_stat = guest_src.stat()
            guest_src.write_bytes(b"PGIM" + b"\x99" * 48)
            os.utime(guest_src, ns=(orig_stat.st_atime_ns, orig_stat.st_mtime_ns))

            # Size modification on helper.pgim.
            helper_src = program_dir / "helper" / "image" / "helper.pgim"
            helper_src.write_bytes(b"PGIM" + b"\x02" * 64)

            third = sbb.stage_browser_boot(
                program_dir=program_dir,
                boot_dir=boot_dir,
                repo_root=repo,
            )
            self.assertTrue(third["ok"])
            self.assertEqual(third["boot"]["copied_count"], 2)
            self.assertEqual(third["boot"]["unchanged_count"], 3)
            by_rel = {a["rel_path"]: a for a in third["boot"]["artifacts"]}
            self.assertEqual(by_rel["main/image/guest.pgim"]["reason"], "sha256_changed")
            self.assertEqual(by_rel["helper/image/helper.pgim"]["reason"], "size_changed")
            self.assertEqual(
                (boot_dir / "main" / "image" / "guest.pgim").read_bytes(),
                b"PGIM" + b"\x99" * 48,
            )

    def test_escaping_symlink_in_program_dir_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            if not _can_create_symlinks(base):
                self.skipTest("symlinks require elevated privileges on this host")
            repo = _make_repo_skeleton(base)
            program_dir = repo / "work" / "recomp_program"
            boot_dir = repo / "work" / "recomp_build"
            _populate_program_dir(program_dir)

            outside_secret = base / "outside.pgim"
            outside_secret.write_bytes(b"PGIM-SECRET")
            os.symlink(outside_secret, program_dir / "escape.pgim")

            report = sbb.stage_browser_boot(
                program_dir=program_dir,
                boot_dir=boot_dir,
                repo_root=repo,
            )
            self.assertTrue(report["ok"])
            self.assertGreaterEqual(report["boot"]["skipped_escaping_symlinks"], 1)
            self.assertFalse((boot_dir / "escape.pgim").exists())

    def test_escaping_symlink_guard_via_mock_runs_on_all_hosts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = _make_repo_skeleton(base)
            program_dir = repo / "work" / "recomp_program"
            boot_dir = repo / "work" / "recomp_build"
            _populate_program_dir(program_dir)
            escape_entry = program_dir / "escape.pgim"
            escape_entry.write_bytes(b"PGIM-FAKE")
            outside_target = base / "outside_secret.pgim"
            outside_target.write_bytes(b"PGIM-OUTSIDE")

            orig_is_symlink = Path.is_symlink
            orig_resolve = Path.resolve

            def fake_is_symlink(self_path: Path) -> bool:
                if self_path.name == "escape.pgim":
                    return True
                return orig_is_symlink(self_path)

            def fake_resolve(self_path: Path, strict: bool = False) -> Path:
                if self_path.name == "escape.pgim":
                    return orig_resolve(outside_target, strict=strict)
                return orig_resolve(self_path, strict=strict)

            with (
                mock.patch.object(Path, "is_symlink", fake_is_symlink),
                mock.patch.object(Path, "resolve", fake_resolve),
            ):
                report = sbb.stage_browser_boot(
                    program_dir=program_dir,
                    boot_dir=boot_dir,
                    repo_root=repo,
                )

            self.assertTrue(report["ok"])
            self.assertEqual(report["boot"]["skipped_escaping_symlinks"], 1)
            self.assertFalse((boot_dir / "escape.pgim").exists())
            self.assertTrue((boot_dir / "main" / "image" / "guest.pgim").is_file())

    def test_staged_boot_artifacts_served_cleanly_by_serve_local(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            program_dir = repo / "work" / "recomp_program"
            boot_dir = repo / "work" / "recomp_build"
            _populate_program_dir(program_dir)

            report = sbb.stage_browser_boot(
                program_dir=program_dir,
                boot_dir=boot_dir,
                repo_root=repo,
            )
            self.assertTrue(report["ok"])

            httpd = serve_local.create_server(
                "127.0.0.1",
                0,
                repo / "web",
                repo / "local-data",
                boot_root=boot_dir,
            )
            port = int(httpd.server_address[1])
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3.0)
                try:
                    conn.request("GET", "/boot/program-manifest.json")
                    resp = conn.getresponse()
                    manifest_body = resp.read()
                    self.assertEqual(resp.status, 200)
                    self.assertEqual(
                        resp.getheader("Cross-Origin-Opener-Policy"), "same-origin"
                    )
                    self.assertIn(b"playgta6-recomp-program", manifest_body)

                    conn.request("GET", "/boot/main/image/guest.pgim")
                    resp2 = conn.getresponse()
                    pgim_body = resp2.read()
                    self.assertEqual(resp2.status, 200)
                    self.assertEqual(pgim_body, b"PGIM" + b"\x01" * 48)

                    # Excluded .ll was never staged and is blocked by /boot/ allowlist.
                    conn.request("GET", "/boot/main/shards/shard-000.ll")
                    resp3 = conn.getresponse()
                    resp3.read()
                    self.assertEqual(resp3.status, 403)
                finally:
                    conn.close()
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join(timeout=2)

    def test_missing_program_dir_fails_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            report = sbb.stage_browser_boot(
                program_dir=repo / "work" / "does_not_exist",
                boot_dir=repo / "work" / "recomp_build",
                repo_root=repo,
            )
            self.assertFalse(report["ok"])
            self.assertEqual(report["boot"]["status"], "error")
            self.assertTrue(report["errors"])


class TestTrackedDirectorySafetyGuard(unittest.TestCase):
    def test_refuses_tracked_boot_dirs_and_allows_work_local_data_and_external(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = _make_repo_skeleton(base)
            program_dir = repo / "work" / "recomp_program"
            _populate_program_dir(program_dir)

            for bad_rel in ("web", "tools", "runtime", "web/dist", "."):
                bad_boot = repo / bad_rel
                report = sbb.stage_browser_boot(
                    program_dir=program_dir,
                    boot_dir=bad_boot,
                    repo_root=repo,
                )
                self.assertFalse(report["ok"], f"expected refusal for {bad_rel}")
                self.assertFalse(report["safety"]["boot_dir_safe"])
                self.assertEqual(report["boot"]["status"], "refused")
                self.assertFalse((bad_boot / "program-manifest.json").exists())

            # Allowed inside work/
            ok_work = sbb.stage_browser_boot(
                program_dir=program_dir,
                boot_dir=repo / "work" / "custom_boot",
                repo_root=repo,
            )
            self.assertTrue(ok_work["ok"])
            self.assertTrue((repo / "work" / "custom_boot" / "program-manifest.json").is_file())

            # Allowed inside local-data/
            ok_data = sbb.stage_browser_boot(
                program_dir=program_dir,
                boot_dir=repo / "local-data" / "boot_cache",
                repo_root=repo,
            )
            self.assertTrue(ok_data["ok"])
            self.assertTrue(
                (repo / "local-data" / "boot_cache" / "program-manifest.json").is_file()
            )

            # Allowed outside repo (temp dir)
            ext_boot = base / "external_boot"
            ok_ext = sbb.stage_browser_boot(
                program_dir=program_dir,
                boot_dir=ext_boot,
                repo_root=repo,
            )
            self.assertTrue(ok_ext["ok"])
            self.assertTrue((ext_boot / "program-manifest.json").is_file())

    def test_refuses_tracked_web_dist_and_allows_web_dist_and_external(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = _make_repo_skeleton(base)
            program_dir = repo / "work" / "recomp_program"
            wasm_dir = repo / "work" / "wasm_guest_dist"
            _populate_program_dir(program_dir)
            _write_wasm_pair(wasm_dir, wasm.build_valid_runtime_wasm())

            for bad_rel in ("web", "tools", "runtime", "work/recomp_build", "."):
                bad_dist = repo / bad_rel
                report = sbb.stage_browser_boot(
                    program_dir=program_dir,
                    wasm_dir=wasm_dir,
                    boot_dir=repo / "work" / "recomp_build",
                    web_dist=bad_dist,
                    repo_root=repo,
                )
                self.assertFalse(report["ok"], f"expected refusal for web_dist={bad_rel}")
                self.assertFalse(report["safety"]["web_dist_safe"])
                self.assertEqual(report["wasm"]["status"], "refused")
                self.assertFalse((bad_dist / "runtime.wasm").exists())

            # Allowed: web/dist
            ok_dist = sbb.stage_browser_boot(
                program_dir=program_dir,
                wasm_dir=wasm_dir,
                boot_dir=repo / "work" / "recomp_build",
                web_dist=repo / "web" / "dist",
                repo_root=repo,
            )
            self.assertTrue(ok_dist["ok"])
            self.assertTrue((repo / "web" / "dist" / "runtime.wasm").is_file())

            # Allowed: outside repo
            ext_dist = base / "external_dist"
            ok_ext = sbb.stage_browser_boot(
                program_dir=program_dir,
                wasm_dir=wasm_dir,
                boot_dir=repo / "work" / "recomp_build",
                web_dist=ext_dist,
                repo_root=repo,
            )
            self.assertTrue(ok_ext["ok"])
            self.assertTrue((ext_dist / "runtime.wasm").is_file())

    def test_refuses_boot_dir_equal_to_or_inside_program_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            program_dir = repo / "work" / "recomp_program"
            _populate_program_dir(program_dir)

            for nested in (program_dir, program_dir / "nested_out"):
                report = sbb.stage_browser_boot(
                    program_dir=program_dir,
                    boot_dir=nested,
                    repo_root=repo,
                )
                self.assertFalse(report["ok"])
                self.assertEqual(report["boot"]["status"], "refused")


class TestWasm64ValidatedStaging(unittest.TestCase):
    def test_valid_wasm64_stages_atomically_into_web_dist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            program_dir = repo / "work" / "recomp_program"
            wasm_dir = repo / "work" / "wasm_guest_dist"
            boot_dir = repo / "work" / "recomp_build"
            web_dist = repo / "web" / "dist"
            _populate_program_dir(program_dir)

            valid_wasm = wasm.build_valid_runtime_wasm()
            js_payload = "export default function createRuntimeModule() { return 64; }\n"
            _write_wasm_pair(wasm_dir, valid_wasm, js_payload)

            report = sbb.stage_browser_boot(
                program_dir=program_dir,
                wasm_dir=wasm_dir,
                boot_dir=boot_dir,
                web_dist=web_dist,
                repo_root=repo,
            )
            self.assertTrue(report["ok"])
            self.assertEqual(report["wasm"]["status"], "staged")
            self.assertEqual(report["wasm"]["copied_count"], 2)
            self.assertTrue(report["wasm"]["validation"]["ok"])
            self.assertTrue(report["wasm"]["validation"]["is_memory64"])
            self.assertTrue(report["wasm"]["validation"]["is_shared_memory"])
            self.assertTrue(report["wasm"]["validation"]["all_host_imports_ok"])
            self.assertEqual(report["wasm"]["validation"]["host_imports_valid"], 5)
            self.assertEqual(report["wasm"]["validation"]["host_imports_total"], 5)
            self.assertEqual((web_dist / "runtime.wasm").read_bytes(), valid_wasm)
            self.assertEqual((web_dist / "runtime.js").read_text(encoding="utf-8"), js_payload)
            # Ensure no leftover .tmp files in web/dist.
            self.assertEqual(
                sorted(p.name for p in web_dist.iterdir()),
                ["runtime.js", "runtime.wasm"],
            )

            # Second run with identical valid Wasm64 is a clean no-op.
            second = sbb.stage_browser_boot(
                program_dir=program_dir,
                wasm_dir=wasm_dir,
                boot_dir=boot_dir,
                web_dist=web_dist,
                repo_root=repo,
            )
            self.assertTrue(second["ok"])
            self.assertEqual(second["wasm"]["status"], "unchanged")
            self.assertEqual(second["wasm"]["copied_count"], 0)
            self.assertEqual(second["wasm"]["unchanged_count"], 2)

    def test_atomic_replace_failure_cleans_up_temp_files_and_preserves_existing_wasm(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            program_dir = repo / "work" / "recomp_program"
            wasm_dir = repo / "work" / "wasm_guest_dist"
            boot_dir = repo / "work" / "recomp_build"
            web_dist = repo / "web" / "dist"
            _populate_program_dir(program_dir)

            existing_wasm = wasm.build_valid_runtime_wasm(min_pages=512)
            existing_js = "// existing runtime.js\n"
            _write_wasm_pair(web_dist, existing_wasm, existing_js)

            new_wasm = wasm.build_valid_runtime_wasm(min_pages=1024)
            _write_wasm_pair(wasm_dir, new_wasm, "// new runtime.js\n")

            orig_replace = os.replace

            def fail_on_web_dist(src: Path | str, dst: Path | str) -> None:
                if Path(dst).parent.resolve() == web_dist.resolve():
                    raise OSError("simulated replace failure")
                orig_replace(src, dst)

            with mock.patch("stage_browser_boot.os.replace", side_effect=fail_on_web_dist):
                report = sbb.stage_browser_boot(
                    program_dir=program_dir,
                    wasm_dir=wasm_dir,
                    boot_dir=boot_dir,
                    web_dist=web_dist,
                    repo_root=repo,
                )

            self.assertFalse(report["ok"])
            self.assertEqual(report["wasm"]["status"], "error")
            self.assertEqual((web_dist / "runtime.wasm").read_bytes(), existing_wasm)
            self.assertEqual(
                (web_dist / "runtime.js").read_text(encoding="utf-8"),
                existing_js,
            )
            self.assertEqual(
                sorted(p.name for p in web_dist.iterdir()),
                ["runtime.js", "runtime.wasm"],
            )

    def test_invalid_wasm64_refuses_and_preserves_existing_web_dist(self) -> None:
        cases = [
            ("wasm32_not_memory64", wasm.build_valid_runtime_wasm(is_memory64=False)),
            ("unshared_memory", wasm.build_valid_runtime_wasm(is_shared=False)),
            ("wasm32_host_sigs", wasm.build_valid_runtime_wasm(wasm32_host_sigs=True)),
            (
                "missing_host_import",
                wasm.build_valid_runtime_wasm(omit_host_imports=("host_wgpu_submit",)),
            ),
            ("corrupt_bytes", b"NOT_A_WASM_BINARY"),
        ]
        for label, bad_wasm in cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as tmp:
                repo = _make_repo_skeleton(Path(tmp))
                program_dir = repo / "work" / "recomp_program"
                wasm_dir = repo / "work" / "wasm_guest_dist"
                boot_dir = repo / "work" / "recomp_build"
                web_dist = repo / "web" / "dist"
                _populate_program_dir(program_dir)

                # Pre-populate existing valid runtime in web/dist.
                existing_wasm = wasm.build_valid_runtime_wasm(min_pages=512)
                existing_js = "// known-good runtime.js\n"
                _write_wasm_pair(web_dist, existing_wasm, existing_js)

                # Candidate contains invalid Wasm64 + new JS.
                _write_wasm_pair(wasm_dir, bad_wasm, "// broken runtime.js\n")

                report = sbb.stage_browser_boot(
                    program_dir=program_dir,
                    wasm_dir=wasm_dir,
                    boot_dir=boot_dir,
                    web_dist=web_dist,
                    repo_root=repo,
                )
                self.assertFalse(report["ok"])
                self.assertEqual(report["wasm"]["status"], "refused")
                self.assertFalse(report["wasm"]["validation"]["ok"])
                self.assertTrue(report["errors"])
                # Existing web/dist files MUST remain completely untouched!
                self.assertEqual((web_dist / "runtime.wasm").read_bytes(), existing_wasm)
                self.assertEqual(
                    (web_dist / "runtime.js").read_text(encoding="utf-8"),
                    existing_js,
                )

    def test_partial_wasm_bundle_fails_without_touching_web_dist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            program_dir = repo / "work" / "recomp_program"
            wasm_dir = repo / "work" / "wasm_guest_dist"
            web_dist = repo / "web" / "dist"
            _populate_program_dir(program_dir)
            wasm_dir.mkdir(parents=True)
            (wasm_dir / "runtime.wasm").write_bytes(wasm.build_valid_runtime_wasm())
            # Deliberately omit runtime.js.

            report = sbb.stage_browser_boot(
                program_dir=program_dir,
                wasm_dir=wasm_dir,
                boot_dir=repo / "work" / "recomp_build",
                web_dist=web_dist,
                repo_root=repo,
            )
            self.assertFalse(report["ok"])
            self.assertEqual(report["wasm"]["status"], "error")
            self.assertFalse(web_dist.exists())


class TestDryRunAndCli(unittest.TestCase):
    def test_dry_run_plans_without_writing_any_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            program_dir = repo / "work" / "recomp_program"
            wasm_dir = repo / "work" / "wasm_guest_dist"
            boot_dir = repo / "work" / "recomp_build"
            web_dist = repo / "web" / "dist"
            _populate_program_dir(program_dir)
            _write_wasm_pair(wasm_dir, wasm.build_valid_runtime_wasm())

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = sbb.main(
                    [
                        "--repo-root",
                        str(repo),
                        "--program-dir",
                        "work/recomp_program",
                        "--wasm-dir",
                        "work/wasm_guest_dist",
                        "--boot-dir",
                        "work/recomp_build",
                        "--web-dist",
                        "web/dist",
                        "--dry-run",
                        "--json",
                    ]
                )
            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["boot"]["status"], "planned")
            self.assertEqual(payload["boot"]["planned_count"], 5)
            self.assertEqual(payload["boot"]["copied_count"], 0)
            self.assertEqual(payload["wasm"]["status"], "planned")
            self.assertEqual(payload["wasm"]["planned_count"], 2)
            self.assertEqual(payload["wasm"]["copied_count"], 0)
            self.assertFalse(boot_dir.exists())
            self.assertFalse(web_dist.exists())

    def test_cli_returns_nonzero_when_wasm_validation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            program_dir = repo / "work" / "recomp_program"
            wasm_dir = repo / "work" / "wasm_guest_dist"
            _populate_program_dir(program_dir)
            _write_wasm_pair(wasm_dir, wasm.build_valid_runtime_wasm(is_memory64=False))

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = sbb.main(
                    [
                        "--repo-root",
                        str(repo),
                        "--program-dir",
                        str(program_dir),
                        "--wasm-dir",
                        str(wasm_dir),
                        "--boot-dir",
                        str(repo / "work" / "recomp_build"),
                        "--web-dist",
                        str(repo / "web" / "dist"),
                        "--dry-run",
                    ]
                )
            self.assertEqual(rc, 1)
            output = buf.getvalue()
            self.assertIn("FAIL", output)
            self.assertIn("memory64=NO", output)


class TestCrossPlatformSanitization(unittest.TestCase):
    def test_sanitizes_windows_unc_and_posix_paths_and_usernames(self) -> None:
        samples = [
            r"C:\Users\AliceSecret\work\recomp_program\guest.pgim",
            r"D:/Users/CharlieSecret/local-data/boot/program-manifest.json",
            r"\\fileserver\PrivateShare\BobSecret\runtime.wasm",
            "//fileserver/PrivateShare/BobSecret/runtime.js",
            "/home/bobsecret/src/playgta6web/web/dist/runtime.wasm",
            "/Users/AliceSecret/Library/Caches/guest.pgim",
            "/tmp/AliceSecret-stage/recomp_build",
        ]
        old_username = os.environ.get("USERNAME")
        old_user = os.environ.get("USER")
        try:
            os.environ["USERNAME"] = "AliceSecret"
            os.environ["USER"] = "BobSecret"
            for sample in samples:
                cleaned = sbb.sanitize_text(sample)
                self.assertNotIn("AliceSecret", cleaned)
                self.assertNotIn("BobSecret", cleaned)
                self.assertNotIn("CharlieSecret", cleaned)
                self.assertNotRegex(cleaned, r"[A-Za-z]:\\")
                self.assertNotRegex(cleaned, r"[A-Za-z]:/")
                self.assertNotIn("\\\\", cleaned)
                self.assertNotIn("//fileserver", cleaned)
                self.assertNotIn("/home/", cleaned)
                self.assertNotIn("/Users/", cleaned)
                self.assertNotIn("/tmp/", cleaned)
        finally:
            if old_username is None:
                os.environ.pop("USERNAME", None)
            else:
                os.environ["USERNAME"] = old_username
            if old_user is None:
                os.environ.pop("USER", None)
            else:
                os.environ["USER"] = old_user

    def test_basename_label_handles_both_separators_cross_platform(self) -> None:
        self.assertEqual(
            sbb.basename_label(r"C:\Users\DevUser\work\recomp_program\guest.pgim"),
            "guest.pgim",
        )
        self.assertEqual(
            sbb.basename_label("/home/devuser/work/recomp_program/guest.pgim"),
            "guest.pgim",
        )
        self.assertEqual(sbb.basename_label("C:\\"), "<unknown>")
        self.assertEqual(sbb.basename_label(None), "<unknown>")

    def test_sanitize_value_replaces_raw_bytes(self) -> None:
        sanitized = sbb.sanitize_value(
            {
                "raw": b"\x00asm\x01\x00\x00\x00",
                "nested": [bytearray(b"abc"), memoryview(b"12345")],
                "path": Path(r"C:\Users\SecretName\guest.pgim"),
            }
        )
        self.assertEqual(sanitized["raw"], "<bytes:8>")
        self.assertEqual(sanitized["nested"], ["<bytes:3>", "<bytes:5>"])
        self.assertEqual(sanitized["path"], "guest.pgim")

    def test_json_and_text_reports_contain_zero_host_paths_or_usernames(self) -> None:
        with tempfile.TemporaryDirectory(prefix="StageSecretHost_") as tmp:
            base = Path(tmp)
            repo = _make_repo_skeleton(base)
            program_dir = repo / "work" / "recomp_program"
            wasm_dir = repo / "work" / "wasm_guest_dist"
            ext_boot = base / "ext_boot_out"
            ext_dist = base / "ext_dist_out"
            _populate_program_dir(program_dir)
            _write_wasm_pair(wasm_dir, wasm.build_valid_runtime_wasm())

            report = sbb.stage_browser_boot(
                program_dir=program_dir,
                wasm_dir=wasm_dir,
                boot_dir=ext_boot,
                web_dist=ext_dist,
                repo_root=repo,
            )
            self.assertTrue(report["ok"])
            encoded = json.dumps(report)
            rendered = sbb.format_report_text(report)

            for text in (encoded, rendered):
                self.assertNotIn("StageSecretHost_", text)
                self.assertNotIn(str(base), text)
                self.assertNotIn(str(base).replace("\\", "/"), text)
                self.assertNotRegex(text, r"[A-Za-z]:[\\/]")
                self.assertNotIn("/Users/", text)
                self.assertNotIn("/home/", text)
                self.assertNotIn("/tmp/", text)
                for username in sbb._current_usernames():
                    self.assertNotRegex(
                        text,
                        re.compile(
                            rf"(?<![A-Za-z0-9_]){re.escape(username)}(?![A-Za-z0-9_])",
                            re.IGNORECASE,
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
