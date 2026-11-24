#!/usr/bin/env python3
"""Unit and CLI tests for tools/verify_browser_boot.py (Track 5 / W48)."""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import check_wasm_exports as wasm  # noqa: E402
import inspect_wgsl_pack as wgpk  # noqa: E402
import verify_browser_boot as vbb  # noqa: E402

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

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"splash-bytes-0123456789ABCDEF"


def _full_import_coverage(_repo: Path) -> dict:
    return {
        "ok": True,
        "errors": [],
        "totals": {
            "total": 587,
            "shimmed": 580,
            "guest_recompiled": 7,
            "missing": 0,
            "effective_boot_pct": 100.0,
            "shimmed_pct": 98.81,
            "guest_recompiled_pct": 1.19,
            "missing_pct": 0.0,
        },
    }


def _partial_import_coverage(_repo: Path) -> dict:
    return {
        "ok": True,
        "errors": [],
        "totals": {
            "total": 587,
            "shimmed": 500,
            "guest_recompiled": 7,
            "missing": 80,
            "effective_boot_pct": 86.37,
            "shimmed_pct": 85.18,
            "guest_recompiled_pct": 1.19,
            "missing_pct": 13.63,
        },
    }


def _build_shader_artifacts(shaders_dir: Path) -> None:
    shaders_dir.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "hash": None,
            "stage": "vertex",
            "wgsl": "@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }",
        },
        {
            "hash": None,
            "stage": "fragment",
            "wgsl": "@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }",
        },
    ]
    pack_bytes = wgpk.build_wgpk(entries)
    bundle = wgpk.parse_wgpk(pack_bytes)
    (shaders_dir / "fallback.wgsl.pack").write_bytes(pack_bytes)
    reflection = {
        "schema_version": 1,
        "shaders": {entry.hash_hex: {"stage": entry.stage_name} for entry in bundle.entries},
    }
    (shaders_dir / "reflection-manifest.json").write_text(
        json.dumps(reflection, indent=2),
        encoding="utf-8",
    )


def _build_recomp_artifacts(recomp_dir: Path, *, with_pgim: bool = True) -> None:
    recomp_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "playgta6-recomp-build",
        "schema_version": 1,
        "module": "main",
        "lift": {
            "functions": 10,
            "instructions": 100,
            "unsupported_instructions": 5,
            "shards": [
                {"index": 0, "functions": 4, "instructions": 40},
                {"index": 1, "functions": 6, "instructions": 60},
            ],
        },
        "discovery": {"functions": 10, "instructions": 100},
    }
    (recomp_dir / "build-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    program = {
        "schema": "playgta6-recomp-program",
        "schema_version": 1,
        "modules": [
            {
                "module": "main",
                "dll": "game.exe",
                "lift": {"functions": 10, "instructions": 100, "unsupported": 5},
            }
        ],
    }
    (recomp_dir / "program-manifest.json").write_text(json.dumps(program), encoding="utf-8")
    if with_pgim:
        (recomp_dir / "guest.pgim").write_bytes(b"PGIM" + b"\x00" * 32)


def _write_runtime_wasm(
    dist_dir: Path,
    *,
    kind: str = "smoke_shell",
) -> None:
    dist_dir.mkdir(parents=True, exist_ok=True)
    raw = wasm.build_valid_runtime_wasm()
    if kind == "linked_guest_runtime":
        # Prefer export-based classification, then pad past 1 MiB for honesty.
        module = wasm.parse_wasm_module(raw)
        exports = [(exp.name, exp.kind, exp.index) for exp in module.exports]
        exports.append(("pg6_guest_boot_run", "func", 0))
        # Rebuild with guest export + pad.
        base = wasm.build_valid_runtime_wasm()
        # Append custom section padding via raw bytes after a valid module is awkward;
        # pad file size beyond 1 MiB while keeping a valid prefix parse...
        # Better: build module with guest export using build_wasm_module helpers.
        types = []
        imports = []
        # memory + host imports
        imports.append(
            {
                "module": "env",
                "name": "memory",
                "kind": "memory",
                "min_pages": 256,
                "max_pages": 65536,
                "is_shared": True,
                "is_memory64": True,
            }
        )
        for spec in wasm.REQUIRED_HOST_IMPORTS:
            type_idx = len(types)
            types.append((spec.params_wasm64, spec.results_wasm64))
            imports.append(
                {
                    "module": spec.module,
                    "name": spec.name,
                    "kind": "func",
                    "type_index": type_idx,
                }
            )
        # local func type for guest boot export
        boot_type = len(types)
        types.append((("i64",), ("i32",)))
        raw = wasm.build_wasm_module(
            types=types,
            imports=imports,
            function_types=[boot_type],
            exports=[
                ("memory", "memory", 0),
                ("pg6_guest_boot_run", "func", len([i for i in imports if i["kind"] == "func"])),
            ],
            target_features=["+memory64", "+atomics", "+bulk-memory"],
        )
        # Ensure code section exists? function section without code may fail some validators
        # but check_wasm_exports only parses type/import/function/memory/export.
        if len(raw) < vbb.SMOKE_SHELL_MAX_BYTES:
            # Pad with a custom section filled with zeros (still valid wasm).
            pad_needed = vbb.SMOKE_SHELL_MAX_BYTES - len(raw) + 16
            name = wasm.encode_name("pad")
            payload = name + (b"\x00" * max(pad_needed - len(name), 1))
            raw = raw + wasm.encode_section(wasm.SECTION_CUSTOM, payload)
    (dist_dir / "runtime.wasm").write_bytes(raw)
    (dist_dir / "runtime.js").write_text("export default {};\n", encoding="utf-8")


def _create_repo_skeleton(base: Path, *, include_server: bool = True) -> Path:
    repo = base / "repo"
    (repo / "web").mkdir(parents=True)
    (repo / "web" / "index.html").write_text("<h1>boot</h1>", encoding="utf-8")
    (repo / "local-data").mkdir()
    (repo / "work").mkdir()
    if include_server:
        (repo / "server").mkdir(parents=True)
        shutil.copy2(REPO_ROOT / "server" / "serve_local.py", repo / "server" / "serve_local.py")
        # serve_local lazily loads inspect_game_assets from tools/; copy it too.
        (repo / "tools").mkdir(parents=True)
        shutil.copy2(
            REPO_ROOT / "tools" / "inspect_game_assets.py",
            repo / "tools" / "inspect_game_assets.py",
        )
    return repo


def _populate_complete_env(
    repo: Path,
    base: Path,
    *,
    wasm_kind: str = "linked_guest_runtime",
    with_game: bool = True,
) -> Path | None:
    game_dir = None
    if with_game:
        game_dir = base / "SyntheticGameInstall"
        game_dir.mkdir()
        (game_dir / "MicrosoftGame.Config").write_text(SYNTHETIC_CONFIG, encoding="utf-8")
        assets = game_dir / "Assets"
        assets.mkdir()
        (assets / "xbs_splash.png").write_bytes(PNG_BYTES)
        (game_dir / "Game.exe").write_bytes(b"MZ" + b"\x90" * 64)
        (game_dir / "helper.dll").write_bytes(b"MZ" + b"\x91" * 64)
        (game_dir / "levels.rpf").write_bytes(b"RPF9" + b"\x01" * 32)
        (repo / ".env").write_text(
            f"GTA6_GAME_DIR={game_dir}\nGTA6_DATA_DIR=local-data\nGTA6_WORK_DIR=work\n",
            encoding="utf-8",
        )

    _build_shader_artifacts(repo / "local-data" / "shaders")
    _build_recomp_artifacts(repo / "work" / "recomp_build")
    _write_runtime_wasm(repo / "web" / "dist", kind=wasm_kind)
    return game_dir


class TestVerifyBrowserBoot(unittest.TestCase):
    def test_missing_environment_fails_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _create_repo_skeleton(Path(tmp))
            report = vbb.run_verification(
                repo,
                env={},
                skip_server=True,
                import_coverage_fn=_partial_import_coverage,
            )
            self.assertFalse(report["ready_for_browser_splash"])
            self.assertFalse(report["ready_for_real_guest_boot"])
            self.assertEqual(report["gates"]["game_assets"]["status"], "skip")
            self.assertEqual(report["gates"]["import_coverage"]["status"], "fail")
            self.assertEqual(report["gates"]["recomp_artifacts"]["status"], "fail")
            self.assertEqual(report["gates"]["shader_pack"]["status"], "fail")
            self.assertEqual(report["gates"]["wasm64_runtime"]["status"], "fail")
            self.assertEqual(report["gates"]["local_server"]["status"], "skip")

    def test_partial_environment_splash_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = _create_repo_skeleton(base)
            _build_shader_artifacts(repo / "local-data" / "shaders")
            _write_runtime_wasm(repo / "web" / "dist", kind="smoke_shell")
            report = vbb.run_verification(
                repo,
                env={},
                skip_server=True,
                import_coverage_fn=_partial_import_coverage,
            )
            self.assertEqual(report["gates"]["shader_pack"]["status"], "pass")
            self.assertEqual(report["gates"]["wasm64_runtime"]["status"], "warn")
            self.assertEqual(report["gates"]["wasm64_runtime"]["details"]["kind"], "smoke_shell")
            self.assertFalse(report["ready_for_browser_splash"])
            self.assertFalse(report["ready_for_real_guest_boot"])

    def test_complete_environment_ready_for_real_guest_boot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = _create_repo_skeleton(base)
            _populate_complete_env(repo, base, wasm_kind="linked_guest_runtime")
            report = vbb.run_verification(
                repo,
                env={},
                skip_server=False,
                import_coverage_fn=_full_import_coverage,
            )
            self.assertEqual(report["gates"]["game_assets"]["status"], "pass")
            self.assertEqual(report["gates"]["import_coverage"]["status"], "pass")
            self.assertEqual(report["gates"]["recomp_artifacts"]["status"], "pass")
            self.assertEqual(report["gates"]["shader_pack"]["status"], "pass")
            self.assertEqual(report["gates"]["wasm64_runtime"]["status"], "pass")
            self.assertEqual(report["gates"]["wasm64_runtime"]["details"]["kind"], "linked_guest_runtime")
            self.assertEqual(report["gates"]["local_server"]["status"], "pass")
            self.assertTrue(report["ready_for_browser_splash"])
            self.assertTrue(report["ready_for_real_guest_boot"])
            self.assertIn("python server/serve_local.py", report["next_steps"])

    def test_smoke_shell_vs_linked_guest_distinction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = _create_repo_skeleton(base, include_server=False)
            _populate_complete_env(repo, base, wasm_kind="smoke_shell")
            smoke = vbb.run_verification(
                repo,
                env={},
                skip_server=True,
                import_coverage_fn=_full_import_coverage,
            )
            self.assertEqual(smoke["gates"]["wasm64_runtime"]["details"]["kind"], "smoke_shell")
            self.assertEqual(smoke["gates"]["wasm64_runtime"]["status"], "warn")
            self.assertTrue(smoke["ready_for_browser_splash"])
            self.assertFalse(smoke["ready_for_real_guest_boot"])

            _write_runtime_wasm(repo / "web" / "dist", kind="linked_guest_runtime")
            linked = vbb.run_verification(
                repo,
                env={},
                skip_server=True,
                import_coverage_fn=_full_import_coverage,
            )
            self.assertEqual(
                linked["gates"]["wasm64_runtime"]["details"]["kind"],
                "linked_guest_runtime",
            )
            self.assertEqual(linked["gates"]["wasm64_runtime"]["status"], "pass")
            self.assertTrue(linked["ready_for_real_guest_boot"])

    def test_classify_runtime_wasm_kind_helpers(self) -> None:
        self.assertEqual(
            vbb.classify_runtime_wasm_kind(byte_size=1000, export_names=["memory"]),
            "smoke_shell",
        )
        self.assertEqual(
            vbb.classify_runtime_wasm_kind(
                byte_size=1000,
                export_names=["pg6_guest_boot_run"],
            ),
            "linked_guest_runtime",
        )
        self.assertEqual(
            vbb.classify_runtime_wasm_kind(
                byte_size=vbb.SMOKE_SHELL_MAX_BYTES,
                export_names=["memory"],
            ),
            "linked_guest_runtime",
        )

    def test_ephemeral_server_gate_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = _create_repo_skeleton(base)
            _populate_complete_env(repo, base)
            gate = vbb.check_local_server_gate(
                repo,
                data_dir=repo / "local-data",
                game_dir=base / "SyntheticGameInstall",
                recomp_dir=repo / "work" / "recomp_build",
                skip_server=False,
            )
            self.assertEqual(gate.status, "pass")
            self.assertTrue(gate.details["coopValid"])
            self.assertTrue(gate.details["coepValid"])
            self.assertTrue(gate.details["range206Ok"])
            self.assertTrue(gate.details["statusEndpointOk"])
            self.assertTrue(gate.details["shadersOk"])
            self.assertTrue(gate.details["bootOk"])
            self.assertTrue(gate.details["gameSplashOk"])

    def test_sanitization_strips_windows_and_posix_paths_and_usernames(self) -> None:
        samples = [
            r"C:\Users\AliceSecret\game\MicrosoftGame.Config",
            "/home/bobsecret/work/recomp_build/guest.pgim",
            r"\\nas\share\PrivateUser\dump.rpf",
            "D:/Users/CharlieSecret/local-data/shaders/fallback.wgsl.pack",
        ]
        # Force username detection by temporarily setting env USERNAME.
        old_user = os.environ.get("USERNAME")
        old_unix = os.environ.get("USER")
        try:
            os.environ["USERNAME"] = "AliceSecret"
            os.environ["USER"] = "bobsecret"
            for sample in samples:
                cleaned = vbb.sanitize_text(sample)
                self.assertNotIn("AliceSecret", cleaned)
                self.assertNotIn("bobsecret", cleaned)
                self.assertNotIn("CharlieSecret", cleaned)
                self.assertNotRegex(cleaned, r"[A-Za-z]:\\")
                self.assertNotRegex(cleaned, r"[A-Za-z]:/")
                self.assertNotIn("/home/", cleaned)
                self.assertNotIn("/Users/", cleaned)
        finally:
            if old_user is None:
                os.environ.pop("USERNAME", None)
            else:
                os.environ["USERNAME"] = old_user
            if old_unix is None:
                os.environ.pop("USER", None)
            else:
                os.environ["USER"] = old_unix

        # basename_label must never return absolute paths on either separator style.
        self.assertEqual(
            vbb.basename_label(r"C:\Users\AliceSecret\guest.pgim"),
            "guest.pgim",
        )
        self.assertEqual(
            vbb.basename_label("/home/bobsecret/work/guest.pgim"),
            "guest.pgim",
        )

        # Raw bytes never appear in sanitize_value output.
        self.assertEqual(vbb.sanitize_value(b"\x00asm"), "<bytes:4>")

    def test_json_cli_output_is_sanitized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = _create_repo_skeleton(base)
            secret_game = base / "Users" / "TopSecretUser" / "PrivateGTA6"
            secret_game.mkdir(parents=True)
            (secret_game / "MicrosoftGame.Config").write_text(SYNTHETIC_CONFIG, encoding="utf-8")
            assets = secret_game / "Assets"
            assets.mkdir()
            (assets / "xbs_splash.png").write_bytes(PNG_BYTES)
            (secret_game / "Game.exe").write_bytes(b"MZ" + b"\x90" * 64)
            (repo / ".env").write_text(
                f"GTA6_GAME_DIR={secret_game}\nGTA6_DATA_DIR=local-data\nGTA6_WORK_DIR=work\n",
                encoding="utf-8",
            )
            _build_shader_artifacts(repo / "local-data" / "shaders")
            _build_recomp_artifacts(repo / "work" / "recomp_build")
            _write_runtime_wasm(repo / "web" / "dist", kind="linked_guest_runtime")

            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                # Monkeypatch import coverage inside CLI by calling library with injection,
                # then exercise --json formatting on the report.
                report = vbb.run_verification(
                    repo,
                    env={},
                    skip_server=True,
                    import_coverage_fn=_full_import_coverage,
                )
                print(json.dumps(report, indent=2))

            payload_text = stdout.getvalue()
            self.assertNotIn("TopSecretUser", payload_text)
            self.assertNotIn(str(secret_game), payload_text)
            # Absolute path fragments / usernames must not appear.
            self.assertNotRegex(payload_text, r"[A-Za-z]:\\\\Users")
            self.assertNotRegex(payload_text, r"[A-Za-z]:/Users")
            self.assertNotIn("TopSecretUser", payload_text)
            parsed = json.loads(payload_text)
            self.assertIn("gates", parsed)
            self.assertIn("ready_for_browser_splash", parsed)
            self.assertIn("ready_for_real_guest_boot", parsed)
            self.assertNotIn("\x00", payload_text)

    def test_cli_json_flag_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = _create_repo_skeleton(base)
            _populate_complete_env(repo, base, wasm_kind="linked_guest_runtime")

            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = vbb.main(
                    [
                        "--repo-root",
                        str(repo),
                        "--skip-server",
                        "--json",
                    ]
                )
            # Splash artifacts are present; import coverage against the empty temp repo
            # fails, so real-guest boot is false but splash readiness can still be true.
            payload = json.loads(stdout.getvalue())
            self.assertIn("gates", payload)
            self.assertEqual(payload["gates"]["local_server"]["status"], "skip")
            self.assertEqual(payload["gates"]["import_coverage"]["status"], "fail")
            self.assertFalse(payload["ready_for_real_guest_boot"])
            self.assertEqual(code, 0 if payload["ready_for_browser_splash"] else 1)
            blob = stdout.getvalue()
            self.assertNotRegex(blob, r"[A-Za-z]:\\")
            self.assertNotRegex(blob, r"/Users/")
            self.assertNotRegex(blob, r"/home/")

    def test_gate_result_to_dict_shape(self) -> None:
        gate = vbb.GateResult(
            name="shader_pack",
            status="pass",
            summary="ok",
            details={"packNames": ["fallback.wgsl.pack"]},
            next_action=None,
        )
        as_dict = gate.to_dict()
        self.assertEqual(
            set(as_dict.keys()),
            {"name", "status", "summary", "details", "next_action"},
        )

    def test_recomp_artifacts_without_pgim_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _create_repo_skeleton(Path(tmp), include_server=False)
            _build_recomp_artifacts(repo / "work" / "recomp_build", with_pgim=False)
            gate = vbb.check_recomp_artifacts_gate(repo, recomp_dir=repo / "work" / "recomp_build")
            self.assertEqual(gate.status, "fail")
            self.assertIn(".pgim", gate.summary)

    def test_recomp_artifacts_discovers_nested_program_pgims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _create_repo_skeleton(Path(tmp), include_server=False)
            recomp_dir = repo / "work" / "recomp_program"
            recomp_dir.mkdir(parents=True)
            # Create >= 4 unrelated sibling directories before module dirs to guard
            # against regressions of the old non-recursive directory-count limit.
            for sibling in ("a_exports", "b_stubs", "c_tables", "d_logs", "exports"):
                (recomp_dir / sibling).mkdir()
                (recomp_dir / sibling / "placeholder.txt").write_text("ok\n", encoding="utf-8")

            program_manifest = {
                "schema": "playgta6-recomp-program",
                "schema_version": 1,
                "modules": [
                    {
                        "module": "main",
                        "dll": "game.exe",
                        "lift": {"functions": 120, "instructions": 1000, "unsupported": 10},
                    },
                    {
                        "module": "vcruntime140",
                        "dll": "vcruntime140.dll",
                        "lift": {"functions": 30, "instructions": 200, "unsupported": 0},
                    },
                ],
            }
            (recomp_dir / "program-manifest.json").write_text(
                json.dumps(program_manifest),
                encoding="utf-8",
            )
            for mod_name in ("main", "vcruntime140"):
                img_dir = recomp_dir / mod_name / "image"
                img_dir.mkdir(parents=True)
                (img_dir / "guest.pgim").write_bytes(b"PGIM" + b"\x00" * 32)
                (recomp_dir / mod_name / "build-manifest.json").write_text(
                    json.dumps(
                        {
                            "schema": "playgta6-recomp-build",
                            "schema_version": 1,
                            "module": mod_name,
                            "lift": {
                                "functions": 60,
                                "instructions": 500,
                                "unsupported_instructions": 5,
                                "shards": [{"index": 0}, {"index": 1}],
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            gate = vbb.check_recomp_artifacts_gate(repo, recomp_dir=recomp_dir)
            self.assertEqual(gate.status, "pass")
            self.assertEqual(gate.details["pgimCount"], 2)
            self.assertEqual(
                gate.details["pgimNames"],
                ["main/image/guest.pgim", "vcruntime140/image/guest.pgim"],
            )
            payload = json.dumps(gate.to_dict())
            self.assertNotRegex(payload, r"[A-Za-z]:\\")
            self.assertNotRegex(payload, r"[A-Za-z]:/")
            self.assertNotIn("/Users/", payload)
            self.assertNotIn("/home/", payload)

    def test_count_pgim_files_deterministic_and_sanitized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "recomp_out"
            (root / "zeta" / "image").mkdir(parents=True)
            (root / "alpha" / "nested" / "image").mkdir(parents=True)
            (root / "zeta" / "image" / "guest.pgim").write_bytes(b"PGIM")
            (root / "alpha" / "nested" / "image" / "guest.pgim").write_bytes(b"PGIM")
            (root / "root.pgim").write_bytes(b"PGIM")
            count, rels = vbb._count_pgim_files(root)
            self.assertEqual(count, 3)
            self.assertEqual(
                rels,
                [
                    "alpha/nested/image/guest.pgim",
                    "root.pgim",
                    "zeta/image/guest.pgim",
                ],
            )
            for rel in rels:
                self.assertNotIn("\\", rel)
                self.assertFalse(Path(rel).is_absolute())

    def test_import_coverage_100_percent_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _create_repo_skeleton(Path(tmp), include_server=False)
            gate = vbb.check_import_coverage_gate(repo, audit_fn=_full_import_coverage)
            self.assertEqual(gate.status, "pass")
            self.assertIn("587/587", gate.summary)


if __name__ == "__main__":
    unittest.main()
