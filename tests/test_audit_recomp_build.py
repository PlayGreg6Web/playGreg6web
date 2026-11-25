#!/usr/bin/env python3
"""Unit and CLI tests for tools/audit_recomp_build.py (Track 1 / R15)."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import audit_recomp_build as arb  # noqa: E402


def _manifest(**overrides: object) -> dict:
    """Minimal R9 `build-manifest.json` using exact pipeline.rs field names."""
    data: dict = {
        "schema": "playgta6-recomp-build",
        "schema_version": 1,
        "module": "main",
        "tool": {"name": "playgta6-recomp", "version": "0.1.0", "llvm": None},
        "input": {"size": 4096, "fnv1a64": "0x0123456789abcdef"},
        "options": {
            "emit_obj": False,
            "shard_size": 2048,
            "shard_instructions": 65536,
            "extra_entries": ["0x140002000"],
            "shim_headers": ["os/win32_kernel.h"],
            "map_headers": True,
            "executable_bytes": "store",
        },
        "image": {
            "image_base": "0x140000000",
            "entry_point": "0x140001000",
            "size_of_image": "0x4000",
            "sections": [],
            "base_relocations": {"dir64": 2, "highlow": 0},
            "container_bytes": 128,
        },
        "pdata": {"records": 4, "valid": 4, "functions": 4, "diagnostics": {}},
        "discovery": {
            "rounds": 2,
            "entry_points": 3,
            "functions": 5,
            "blocks": 12,
            "instructions": 80,
            "jump_tables": {"resolved": 3, "unresolved": 1},
            "terminators": {"trap": 1, "invalid": 0, "no_return_call": 0},
            "relocations": {"call_target": 4, "jump_table": 3},
            "diagnostics": {},
        },
        "refinements": {
            "no_return": {"imports": [], "thunks": []},
            "code_pointers": {
                "pdata_present": True,
                "round_limit_hit": False,
                "by_source": {},
                "accepted": [],
            },
            "dispatch": {
                "registered_functions": 5,
                "address_taken_not_lifted": [
                    {"va": "0x140009000", "sources": "lea", "status": "outside"}
                ],
            },
            "import_slots": {
                "slot_reads": 2,
                "iat_reads": 2,
                "non_iat_reads": [],
                "misaligned": [],
                "referenced_unresolved": [
                    {
                        "dll": "d3d12_xs.dll",
                        "import": "D3D12XboxCreateDevice",
                        "reason": "no shim",
                    }
                ],
                "unreferenced": [],
            },
        },
        "lift": {
            "functions": 5,
            "instructions": 80,
            "unsupported_instructions": 9,
            "unsupported_by_mnemonic": {
                "cpuid": 5,
                "vfmadd132ps": 3,
                "xgetbv": 1,
            },
            "shards": [
                {
                    "index": 0,
                    "first": "0x140001000",
                    "last": "0x140001100",
                    "functions": 2,
                    "instructions": 20,
                    "ll": "lift/shard-00000.ll",
                    "obj": None,
                },
                {
                    "index": 1,
                    "first": "0x140001200",
                    "last": "0x140001300",
                    "functions": 3,
                    "instructions": 60,
                    "ll": "lift/shard-00001.ll",
                    "obj": None,
                },
            ],
            "registration": {
                "table_symbol": "__playgta6_lifted_functions_main",
                "count_symbol": "__playgta6_lifted_function_count_main",
                "register_function": "__playgta6_register_lifted_functions_main",
                "entry_type": "%Pg6LiftedFunctionEntry",
                "count": 5,
                "ll": "lift/functions.ll",
                "obj": None,
            },
        },
        "imports": {
            "total": 4,
            "bound": 3,
            "unresolved": 1,
            "by_dll": [{"dll": "d3d12_xs.dll", "bound": 0, "unresolved": 1}],
            "catalogue": {"headers": 1, "shims": 3, "diagnostics": 0},
        },
        "diagnostics": {
            "pdata": 0,
            "discovery": 0,
            "image": 0,
            "catalogue": 0,
            "unsupported_instructions": 9,
        },
        "files": [{"path": "report.txt", "size": 32, "fnv1a64": "0x0000000000000001"}],
    }
    data.update(overrides)
    return data


def _discover(**overrides: object) -> dict:
    """`playgta6-recomp discover --json` shape from summary.rs, plus boot metadata."""
    data: dict = {
        "image_base": "0x140000000",
        "image_end": "0x140004000",
        "entry_points": ["0x140001000", "0x140002000"],
        "function_count": 5,
        "block_count": 12,
        "instruction_count": 80,
        "diagnostic_count": 0,
        "pdata": {"functions": 4, "ranges": 4},
        "jump_tables": {"resolved": 3, "unresolved": 1},
        "eh": {
            "c_specific": 0,
            "cxx3": 0,
            "cxx4": 0,
            "gs_check": 0,
            "unknown_handler": 0,
            "parsed": 0,
            "funclets": 0,
            "metadata_resume_targets": 0,
            "diagnostics": 0,
            "setjmp_slots": 0,
            "resume_targets": 0,
            "functions_with_resume_targets": 0,
        },
        "edges": {
            "fallthrough": 4,
            "taken": 1,
            "jump": 1,
            "tail_call": 1,
            "indirect_jump": 1,
            "return": 5,
            "jump_table": 3,
        },
        "relocations": {"call_target": 4, "jump_table": 3},
        "tls_callbacks": ["0x140002000"],
        "functions": [
            {
                "entry": "0x140001000",
                "blocks": 3,
                "instructions": 20,
                "callees": ["0x140001100"],
                "tail_calls": [],
                "indirect_calls": 1,
                "diagnostics": [],
                "unsupported_mnemonics": {"cpuid": 5},
            },
            {
                "entry": "0x140001100",
                "blocks": 2,
                "instructions": 10,
                "callees": ["0x140001200"],
                "tail_calls": [],
                "indirect_calls": 0,
                "diagnostics": [],
            },
            {
                "entry": "0x140001200",
                "blocks": 3,
                "instructions": 18,
                "callees": ["0x140001400"],
                "tail_calls": [],
                "indirect_calls": 1,
                "imports": ["D3D12XboxCreateDevice", "PresentX"],
                "diagnostics": [],
            },
            {
                "entry": "0x140002000",
                "blocks": 2,
                "instructions": 12,
                "callees": ["0x140001200"],
                "tail_calls": [],
                "indirect_calls": 0,
                "diagnostics": [],
            },
            {
                "entry": "0x140001400",
                "blocks": 2,
                "instructions": 20,
                "callees": [],
                "tail_calls": [],
                "indirect_calls": 0,
                "diagnostics": [],
                "unsupported_mnemonics": {"vfmadd132ps": 3, "xgetbv": 1},
            },
        ],
    }
    data.update(overrides)
    return data


def _write_build_dir(
    root: Path,
    *,
    manifest: dict | None = None,
    discover: dict | None = None,
    report: str | None = None,
    nested_module: str = "main",
) -> Path:
    module_dir = root / nested_module
    module_dir.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        (module_dir / "build-manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
    if discover is not None:
        (module_dir / "discover.json").write_text(json.dumps(discover), encoding="utf-8")
    if report is not None:
        (module_dir / "report.txt").write_text(report, encoding="utf-8")
    return module_dir


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = arb.main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


class TestParseVaAndStats(unittest.TestCase):
    def test_parse_va_hex_and_int(self) -> None:
        self.assertEqual(arb.parse_va("0x140001000"), 0x140001000)
        self.assertEqual(arb.parse_va(0x140001000), 0x140001000)
        self.assertEqual(arb.format_va("0x140001000"), "0x140001000")
        self.assertIsNone(arb.parse_va("PresentX"))
        self.assertEqual(arb.median([20, 60]), 40.0)
        self.assertEqual(arb.median([1, 2, 3]), 2.0)
        self.assertEqual(arb.as_float_pct(1, 4), 25.0)
        self.assertEqual(arb.as_float_pct(0, 0), 0.0)


class TestFullAuditFromR9AndDiscover(unittest.TestCase):
    def test_directory_loads_manifest_and_discover_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = _write_build_dir(Path(tmp), manifest=_manifest(), discover=_discover())
            report = arb.audit_recomp_build(build_dir, top=10)

        self.assertEqual(report["schema"], "playgta6-recomp-audit")
        self.assertEqual(report["module"], "main")
        cov = report["coverage"]
        self.assertEqual(cov["discovered_functions"], 5)
        self.assertEqual(cov["lifted_functions"], 3)
        self.assertEqual(cov["unlifted_functions"], 3)  # 2 unsupported + 1 address-taken
        self.assertEqual(cov["lifted_pct"], 60.0)
        self.assertEqual(cov["unlifted_instructions"], 9)
        self.assertEqual(cov["lifted_instructions"], 71)
        self.assertEqual(cov["instruction_lifted_pct"], 88.75)

        shards = report["shards"]
        self.assertEqual(shards["count"], 2)
        self.assertEqual(shards["min"], 20)
        self.assertEqual(shards["max"], 60)
        self.assertEqual(shards["median"], 40.0)

        jump = report["jump_tables"]
        self.assertEqual(jump["resolved_jump_tables"], 3)
        self.assertEqual(jump["unresolved_indirect_jumps"], 1)
        self.assertEqual(jump["resolution_pct"], 75.0)

        mnemonics = report["unlifted_mnemonics"]
        self.assertEqual(mnemonics[0]["mnemonic"], "cpuid")
        self.assertEqual(mnemonics[0]["count"], 5)
        self.assertGreaterEqual(mnemonics[0]["affected_functions"], 1)

        boot = report["boot"]
        self.assertIn("0x140001000", boot["roots"])
        self.assertIn("0x140002000", boot["roots"])
        self.assertGreaterEqual(boot["boot_reachable_functions"], 5)
        self.assertEqual(boot["boot_unresolved_indirect_calls"], 2)
        self.assertTrue(boot["boot_targets"]["D3D12XboxCreateDevice"]["reachable"])
        self.assertTrue(boot["boot_targets"]["PresentX"]["reachable"])
        d3d_path = boot["boot_targets"]["D3D12XboxCreateDevice"]["shortest_path"]
        self.assertIn(d3d_path[0], boot["roots"])
        self.assertEqual(d3d_path[-1], "D3D12XboxCreateDevice")
        self.assertIn("0x140001200", d3d_path)
        # TLS callback 0x140002000 is a shorter boot root than PE entry -> 0x140001100.
        self.assertEqual(d3d_path, ["0x140002000", "0x140001200", "D3D12XboxCreateDevice"])

        recs = report["recommendations"]
        self.assertTrue(recs)
        self.assertEqual(recs[0]["priority"], 1)
        categories = {r["category"] for r in recs}
        self.assertTrue({"boot_lift", "boot_mnemonic", "jump_tables"} & categories)

    def test_direct_manifest_file_without_cfg_uses_aggregate_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "build-manifest.json"
            path.write_text(json.dumps(_manifest()), encoding="utf-8")
            report = arb.audit_recomp_build(path)

        cov = report["coverage"]
        self.assertEqual(cov["lifted_functions"], 5)
        self.assertEqual(cov["unlifted_functions"], 1)  # address-taken only
        self.assertEqual(cov["lifted_pct"], 100.0)
        self.assertEqual(report["unlifted_mnemonics"][0]["mnemonic"], "cpuid")
        self.assertEqual(report["unlifted_mnemonics"][0]["count"], 5)
        self.assertEqual(report["jump_tables"]["resolved_jump_tables"], 3)
        self.assertEqual(report["boot"]["roots"][0], "0x140001000")
        self.assertIn("d3d12_xs.dll!D3D12XboxCreateDevice", report["unresolved_imports"])

    def test_discover_json_file_only_is_fully_unlifted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            discover = _discover()
            # Strip overlay fields so this is a pure summary.rs document plus TLS.
            for fn in discover["functions"]:
                fn.pop("unsupported_mnemonics", None)
                fn.pop("imports", None)
            path.write_text(json.dumps(discover), encoding="utf-8")
            report = arb.audit_recomp_build(path)

        self.assertEqual(report["coverage"]["lifted_functions"], 0)
        self.assertEqual(report["coverage"]["unlifted_functions"], 5)
        self.assertEqual(report["coverage"]["lifted_pct"], 0.0)
        self.assertEqual(report["jump_tables"]["resolution_pct"], 75.0)
        self.assertGreaterEqual(report["boot"]["boot_reachable_functions"], 2)


class TestBootPathDepthLimit(unittest.TestCase):
    def test_boot_depth_zero_stays_on_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = _write_build_dir(Path(tmp), manifest=_manifest(), discover=_discover())
            limited = arb.audit_recomp_build(build_dir, boot_depth=0)
            deeper = arb.audit_recomp_build(build_dir, boot_depth=1)

        self.assertEqual(set(limited["boot"]["reachable_entries"]), {"0x140001000", "0x140002000"})
        self.assertFalse(limited["boot"]["boot_targets"]["D3D12XboxCreateDevice"]["reachable"])
        self.assertIn("0x140001100", deeper["boot"]["reachable_entries"])
        self.assertIn("0x140001200", deeper["boot"]["reachable_entries"])
        self.assertGreater(
            deeper["boot"]["boot_reachable_functions"],
            limited["boot"]["boot_reachable_functions"],
        )

    def test_boot_depth_one_does_not_reach_leaf_unlifted_function(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = _write_build_dir(Path(tmp), manifest=_manifest(), discover=_discover())
            report = arb.audit_recomp_build(build_dir, boot_depth=1)

        reachable = report["boot"]["reachable_entries"]
        self.assertNotIn("0x140001400", reachable)
        self.assertNotIn("D3D12XboxCreateDevice", reachable)


class TestTopNMnemonicRanking(unittest.TestCase):
    def test_top_n_orders_by_count_then_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "build-manifest.json"
            path.write_text(json.dumps(_manifest()), encoding="utf-8")
            top2 = arb.audit_recomp_build(path, top=2)
            top1 = arb.audit_recomp_build(path, top=1)

        names = [row["mnemonic"] for row in top2["unlifted_mnemonics"]]
        self.assertEqual(names, ["cpuid", "vfmadd132ps"])
        self.assertEqual(len(top1["unlifted_mnemonics"]), 1)
        self.assertEqual(top1["unlifted_mnemonics"][0]["mnemonic"], "cpuid")
        self.assertEqual(top1["unlifted_mnemonics"][0]["count"], 5)


class TestJumpTableAndReportFallback(unittest.TestCase):
    def test_report_txt_mnemonics_used_when_lift_map_missing(self) -> None:
        manifest = _manifest()
        manifest["lift"]["unsupported_by_mnemonic"] = {}
        manifest["lift"]["unsupported_instructions"] = 4
        report_txt = (
            "playgta6-recomp build report: module main\n"
            "coverage:\n"
            "  top unsupported mnemonics (2 distinct):\n"
            "    vpclmulqdq            3\n"
            "    wrfsbase              1\n"
            "\n"
            "imports: 0 total\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = _write_build_dir(Path(tmp), manifest=manifest, report=report_txt)
            report = arb.audit_recomp_build(build_dir, top=5)

        names = [row["mnemonic"] for row in report["unlifted_mnemonics"]]
        self.assertEqual(names, ["vpclmulqdq", "wrfsbase"])
        self.assertEqual(report["unlifted_mnemonics"][0]["count"], 3)


class TestMarkdownAndJsonOutput(unittest.TestCase):
    def test_cli_json_and_markdown_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = _write_build_dir(Path(tmp), manifest=_manifest(), discover=_discover())
            json_code, json_out, json_err = _run_cli([str(build_dir), "--json", "--top", "3"])
            md_code, md_out, md_err = _run_cli([str(build_dir), "--markdown"])
            txt_code, txt_out, txt_err = _run_cli([str(build_dir)])

        self.assertEqual((json_code, json_err), (0, ""))
        payload = json.loads(json_out)
        self.assertEqual(payload["schema"], "playgta6-recomp-audit")
        self.assertEqual(payload["coverage"]["lifted_functions"], 3)
        self.assertIn("boot_reachable_functions", payload["boot"])

        self.assertEqual(md_code, 0)
        self.assertIn("# Recomp coverage audit", md_out)
        self.assertIn("D3D12XboxCreateDevice", md_out)
        self.assertIn("Recommendations", md_out)
        self.assertEqual(md_err, "")

        self.assertEqual(txt_code, 0)
        self.assertTrue(txt_out.startswith("playgta6-recomp coverage audit:"))
        self.assertIn("jump tables:", txt_out)
        self.assertIn("recommendations:", txt_out)
        self.assertEqual(txt_err, "")

    def test_missing_input_exits_one(self) -> None:
        code, out, err = _run_cli([str(Path("missing-build-dir-xyz")), "--json"])
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("Error:", err)
        self.assertNotRegex(err, r"[A-Za-z]:[\\/]")


class TestCrossPlatformSanitization(unittest.TestCase):
    def test_sanitize_normalizes_slashes_and_strips_users(self) -> None:
        windows = r"C:\Users\marcb\work\build-manifest.json"
        posix = "/home/marcb/work/build-manifest.json"
        mixed = r"C:\Users\marcb/work\out"
        self.assertEqual(arb.basename_label(windows), "build-manifest.json")
        self.assertEqual(arb.basename_label(posix), "build-manifest.json")
        self.assertEqual(arb.basename_label(mixed), "out")
        cleaned = arb.sanitize_text(windows + " " + posix)
        self.assertNotIn("Users", cleaned)
        self.assertNotIn("marcb", cleaned)
        self.assertNotIn("C:", cleaned)
        self.assertIn("<path>", cleaned)

    def test_audit_output_has_no_host_paths_usernames_or_bytes(self) -> None:
        username = os.environ.get("USERNAME") or os.environ.get("USER") or "marcb"
        manifest = _manifest()
        manifest["options"]["shim_headers"] = [
            rf"C:\Users\{username}\playgta6web\runtime\os\win32_kernel.h",
            f"/home/{username}/runtime/os/win32_kernel.h",
        ]
        discover = _discover()
        discover["functions"][2]["name"] = rf"C:\Users\{username}\secret.bin"

        with tempfile.TemporaryDirectory() as tmp:
            nasty = Path(tmp) / f"Users" / username / "build"
            nasty.mkdir(parents=True)
            path = nasty / "build-manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            (nasty / "discover.json").write_text(json.dumps(discover), encoding="utf-8")
            with mock.patch.dict(os.environ, {"USERNAME": username, "USER": username}):
                report = arb.audit_recomp_build(path, top=5)
                rendered = arb.format_json_report(report) + arb.format_markdown_report(report)
                code, json_out, err = _run_cli([str(path), "--json"])

        blob = json.dumps(report) + rendered + json_out + err
        self.assertEqual(code, 0)
        self.assertNotIn(username, blob)
        self.assertNotIn("C:\\", blob)
        self.assertNotIn("C:/Users", blob)
        self.assertNotIn("/home/", blob)
        self.assertNotRegex(blob, r"(?<![A-Za-z0-9_])Users(?![A-Za-z0-9_])")
        self.assertNotIsInstance(report.get("source"), bytes)
        self.assertTrue(
            report["source"].startswith("build-manifest.json"),
            msg=report["source"],
        )
        dumped = json.dumps(report)
        self.assertNotIn('"\\u00', dumped)  # no sneaky binary escapes expected in labels

    def test_bytes_values_are_redacted(self) -> None:
        redacted = arb.sanitize_value({"blob": b"\x00\xffMZ"})
        self.assertEqual(redacted["blob"], "<bytes:4>")


class TestOutRootDirectoryScan(unittest.TestCase):
    def test_out_root_finds_nested_module_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "work-build"
            _write_build_dir(root, manifest=_manifest(), discover=_discover(), nested_module="game")
            report = arb.audit_recomp_build(root)

        self.assertEqual(report["module"], "main")
        self.assertEqual(report["coverage"]["discovered_functions"], 5)
        self.assertTrue(report["boot"]["boot_targets"]["PresentX"]["reachable"])


if __name__ == "__main__":
    unittest.main()
