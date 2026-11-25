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
        # Single-module schema stays strictly backwards compatible.
        self.assertNotIn("program", report)
        self.assertNotIn("wasm_footprint", report)


def _program_manifest(
    *,
    indirect_dispatch: str | None = "shard",
    unresolved_imports: list[dict] | None = None,
    cycles: list[dict] | None = None,
) -> dict:
    """Synthetic multi-module `program-manifest.json` matching recomp/src/pipeline/program.rs."""
    build_opts: dict = {
        "emit_obj": False,
        "shard_size": 2048,
        "shard_instructions": 65536,
        "map_headers": True,
        "executable_bytes": "store",
    }
    if indirect_dispatch is not None:
        build_opts["indirect_dispatch"] = indirect_dispatch

    imports: list[dict] = [
        {
            "importer": "guest_app.exe",
            "dll": "video_codec.dll",
            "name": "CodecDecodeFrame",
            "ordinal": None,
            "slot_rva": "0x2000",
            "delay_load": False,
            "binding": "guest",
            "provider": "video_codec.dll",
            "target": "0x180001000",
            "kind": "function",
            "shim": None,
            "reason": None,
            "recorded": None,
            "via": [],
            "rule": "default",
        },
        {
            "importer": "guest_app.exe",
            "dll": "KERNEL32.dll",
            "name": "RaiseException",
            "ordinal": None,
            "slot_rva": "0x2008",
            "delay_load": False,
            "binding": "runtime",
            "provider": "KERNEL32.dll",
            "target": None,
            "kind": None,
            "shim": "guest_exceptions",
            "reason": None,
            "recorded": None,
            "via": [],
            "rule": "default",
        },
        {
            "importer": "video_codec.dll",
            "dll": "kernel32.dll",
            "name": "VirtualAlloc",
            "ordinal": None,
            "slot_rva": "0x3000",
            "delay_load": False,
            "binding": "host",
            "provider": None,
            "target": None,
            "kind": None,
            "shim": "os_VirtualAlloc",
            "reason": None,
            "recorded": None,
            "via": [],
            "rule": "default",
        },
    ]
    if unresolved_imports is None:
        unresolved_imports = [
            {
                "importer": "guest_app.exe",
                "dll": "d3d12_xs.dll",
                "name": "D3D12XboxCreateDevice",
                "ordinal": None,
                "slot_rva": "0x2010",
                "delay_load": False,
                "binding": "unresolved",
                "provider": None,
                "target": None,
                "kind": None,
                "shim": None,
                "reason": "no_shim",
                "recorded": None,
                "via": [],
                "rule": "default",
            },
            {
                "importer": "video_codec.dll",
                "dll": "audio_ext.dll",
                "name": None,
                "ordinal": 42,
                "slot_rva": "0x3008",
                "delay_load": False,
                "binding": "unresolved",
                "provider": None,
                "target": None,
                "kind": None,
                "shim": None,
                "reason": "no_shim",
                "recorded": "optional-codec-probe",
                "via": [],
                "rule": "default",
            },
        ]
    imports.extend(unresolved_imports)

    app_unres = sum(
        1 for i in unresolved_imports if str(i.get("importer", "")).lower() == "guest_app.exe"
    )
    codec_unres = sum(
        1 for i in unresolved_imports if str(i.get("importer", "")).lower() == "video_codec.dll"
    )
    dsp_unres = sum(
        1 for i in unresolved_imports if str(i.get("importer", "")).lower() == "audio_dsp.dll"
    )

    return {
        "schema": "playgta6-recomp-program",
        "schema_version": 1,
        "tool": {"name": "playgta6-recomp", "version": "0.1.0"},
        "options": {
            "allow_allocator_split": False,
            "address_pool": ["0x140000000", "0x200000000"],
            "overrides": [],
            "force_guest": [],
            "build": build_opts,
        },
        "modules": [
            {
                "file": "audio_dsp.dll",
                "dll": "audio_dsp.dll",
                "module": "audio_dsp",
                "role": "dll",
                "reason": "import",
                "input": {"size": 2048, "fnv1a64": "0x0000000000000010"},
                "built_fnv1a64": "0x0000000000000011",
                "preferred_base": "0x180100000",
                "base": "0x180100000",
                "rebased": False,
                "size_of_image": "0x2000",
                "range_end": "0x180110000",
                "entry_point": "0x180101000",
                "tls": None,
                "exports": {
                    "total": 4,
                    "functions": 4,
                    "data": 0,
                    "forwarders": 0,
                    "ordinal_only": 0,
                },
                "imports": {
                    "total": 2 + dsp_unres,
                    "guest": 0,
                    "host_bound": 2,
                    "runtime": 0,
                    "host_unresolved": dsp_unres,
                    "before": {"bound": 2, "unresolved": dsp_unres},
                },
                "depends_on": [],
                "iat_prefill": [],
                "registration": {
                    "register_function": "__playgta6_register_lifted_functions_audio_dsp",
                    "table_symbol": "__playgta6_lifted_functions_audio_dsp",
                    "count": 3,
                    "first": "0x180101000",
                    "last": "0x180101200",
                },
                "bridge": {
                    "thunks_symbol": "pg6_import_thunks_audio_dsp",
                    "thunk_count_symbol": "pg6_import_thunk_count_audio_dsp",
                },
                "lift": {"functions": 3, "instructions": 40, "unsupported": 0},
                "build_manifest": {
                    "path": "audio_dsp/build-manifest.json",
                    "fnv1a64": "0x0000000000000012",
                },
            },
            {
                "file": "video_codec.dll",
                "dll": "video_codec.dll",
                "module": "video_codec",
                "role": "dll",
                "reason": "import",
                "input": {"size": 4096, "fnv1a64": "0x0000000000000020"},
                "built_fnv1a64": "0x0000000000000021",
                "preferred_base": "0x180000000",
                "base": "0x180000000",
                "rebased": False,
                "size_of_image": "0x4000",
                "range_end": "0x180010000",
                "entry_point": "0x180001000",
                "tls": None,
                "exports": {
                    "total": 6,
                    "functions": 6,
                    "data": 0,
                    "forwarders": 0,
                    "ordinal_only": 0,
                },
                "imports": {
                    "total": 3 + codec_unres,
                    "guest": 1,
                    "host_bound": 2,
                    "runtime": 0,
                    "host_unresolved": codec_unres,
                    "before": {"bound": 2, "unresolved": 1 + codec_unres},
                },
                "depends_on": ["audio_dsp.dll"],
                "iat_prefill": [
                    {"slot_rva": "0x3010", "value": "0x180101000", "import": "audio_dsp.dll!Mix"}
                ],
                "registration": {
                    "register_function": "__playgta6_register_lifted_functions_video_codec",
                    "table_symbol": "__playgta6_lifted_functions_video_codec",
                    "count": 4,
                    "first": "0x180001000",
                    "last": "0x180001300",
                },
                "bridge": {
                    "thunks_symbol": "pg6_import_thunks_video_codec",
                    "thunk_count_symbol": "pg6_import_thunk_count_video_codec",
                },
                "lift": {"functions": 4, "instructions": 60, "unsupported": 2},
                "build_manifest": {
                    "path": "video_codec/build-manifest.json",
                    "fnv1a64": "0x0000000000000022",
                },
            },
            {
                "file": "guest_app.exe",
                "dll": "guest_app.exe",
                "module": "main",
                "role": "exe",
                "reason": "executable",
                "input": {"size": 8192, "fnv1a64": "0x0000000000000030"},
                "built_fnv1a64": "0x0000000000000031",
                "preferred_base": "0x140000000",
                "base": "0x140000000",
                "rebased": False,
                "size_of_image": "0x8000",
                "range_end": "0x140010000",
                "entry_point": "0x140001000",
                "tls": {
                    "index": 0,
                    "start": "0x140003000",
                    "end": "0x140003100",
                    "index_va": "0x140003200",
                    "callbacks": ["0x140002000"],
                    "zero_fill": 0,
                    "characteristics": "0x0",
                },
                "exports": {
                    "total": 0,
                    "functions": 0,
                    "data": 0,
                    "forwarders": 0,
                    "ordinal_only": 0,
                },
                "imports": {
                    "total": 4 + app_unres,
                    "guest": 1,
                    "host_bound": 2,
                    "runtime": 1,
                    "host_unresolved": app_unres,
                    "before": {"bound": 2, "unresolved": 2 + app_unres},
                },
                "depends_on": ["video_codec.dll"],
                "iat_prefill": [
                    {
                        "slot_rva": "0x2000",
                        "value": "0x180001000",
                        "import": "video_codec.dll!CodecDecodeFrame",
                    }
                ],
                "registration": {
                    "register_function": "__playgta6_register_lifted_functions_main",
                    "table_symbol": "__playgta6_lifted_functions_main",
                    "count": 5,
                    "first": "0x140001000",
                    "last": "0x140001400",
                    **(
                        {"shard_dispatch_symbol": "__playgta6_shard_dispatch_main"}
                        if indirect_dispatch == "shard"
                        else {}
                    ),
                },
                "bridge": {
                    "thunks_symbol": "pg6_import_thunks_main",
                    "thunk_count_symbol": "pg6_import_thunk_count_main",
                },
                "lift": {"functions": 5, "instructions": 80, "unsupported": 9},
                "build_manifest": {
                    "path": "main/build-manifest.json",
                    "fnv1a64": "0x0000000000000032",
                },
            },
        ],
        "init_order": ["audio_dsp.dll", "video_codec.dll", "guest_app.exe"],
        "cycles": cycles or [],
        "imports": imports,
        "external_dlls": [
            {"dll": "d3d12_xs.dll", "status": "host", "importers": ["guest_app.exe"]},
            {"dll": "audio_ext.dll", "status": "missing", "importers": ["video_codec.dll"]},
        ],
        "unreferenced_dlls": [],
        "allocator": {"provider": "host", "imports": 3},
        "export_thunks": {
            "dir": "exports",
            "exports": 10,
            "generated": 10,
            "gthunk": 0,
            "none": 0,
            "distinct_thunks": 8,
            "none_by_reason": {},
        },
        "diagnostics": [],
        "files": [],
    }


def _guest_manifest(
    *,
    indirect_dispatch: str = "shard",
    got_func_imports: int = 18,
    exceeds_v8: bool = False,
) -> dict:
    """Synthetic `guest_manifest.json` matching tools/build_guest_wasm.py."""
    return {
        "schemaVersion": 1,
        "runtimeJs": "runtime.js",
        "runtimeWasm": "runtime.wasm",
        "indirectDispatch": indirect_dispatch,
        "splitModules": 2,
        "splitStrategy": "contiguous",
        "guest_dll_modules": ["audio_dsp", "video_codec"],
        "sideModules": ["guest_part_00.wasm", "guest_part_01.wasm"],
        "sideModuleSizes": [
            {
                "index": 0,
                "name": "guest_part_00.wasm",
                "role": "shards",
                "shardCount": 3,
                "shards": ["shard-00000.ll", "shard-00001.ll", "shard-00002.ll"],
                "wasmBytes": 2 * 1024 * 1024,
                "sizeMiB": 2.0,
                "exceedsV8_1GiBLimit": exceeds_v8,
                "sideModuleMode": 1,
            },
            {
                "index": 1,
                "name": "guest_part_01.wasm",
                "role": "shards",
                "shardCount": 2,
                "shards": ["shard-00003.ll", "shard-00004.ll"],
                "wasmBytes": 1 * 1024 * 1024,
                "sizeMiB": 1.0,
                "exceedsV8_1GiBLimit": False,
                "sideModuleMode": 1,
            },
        ],
        "wasmMetrics": {
            "indirectDispatch": indirect_dispatch,
            "modules": [
                {
                    "name": "runtime.wasm",
                    "role": "main",
                    "wasmBytes": 4 * 1024 * 1024,
                    "sizeMiB": 4.0,
                    "funcImports": 40,
                    "gotFuncImports": 6,
                    "gotMemImports": 4,
                    "funcExports": 120,
                    "tableSize": 64,
                    "definedTableSize": 64,
                    "dylinkTableSize": None,
                    "dylinkMemorySize": None,
                    "hasDylink": False,
                    "shardDispatchExports": 2,
                    "shardTableExports": 1,
                    "subFuncExports": 0,
                    "subFuncImports": 0,
                    "gotFuncSubImports": 0,
                    "gotFuncShardDispatchImports": 4,
                },
                {
                    "name": "guest_part_00.wasm",
                    "role": "side",
                    "wasmBytes": 2 * 1024 * 1024,
                    "sizeMiB": 2.0,
                    "funcImports": 12,
                    "gotFuncImports": max(got_func_imports - 6, 0),
                    "gotMemImports": 2,
                    "funcExports": 30,
                    "tableSize": 16,
                    "definedTableSize": 0,
                    "dylinkTableSize": 16,
                    "dylinkMemorySize": 4096,
                    "hasDylink": True,
                    "shardDispatchExports": 3,
                    "shardTableExports": 1,
                    "subFuncExports": 0,
                    "subFuncImports": 0,
                    "gotFuncSubImports": 0 if indirect_dispatch == "shard" else got_func_imports,
                    "gotFuncShardDispatchImports": 3 if indirect_dispatch == "shard" else 0,
                },
            ],
            "totals": {
                "moduleCount": 2,
                "sideModuleCount": 1,
                "wasmBytes": 6 * 1024 * 1024,
                "sizeMiB": 6.0,
                "funcImports": 52,
                "gotFuncImports": got_func_imports,
                "gotMemImports": 6,
                "funcExports": 150,
                "tableSize": 80,
                "mainTableSize": 64,
                "sideTableSize": 16,
                "shardDispatchExports": 5,
                "shardTableExports": 2,
                "gotFuncSubImports": 0 if indirect_dispatch == "shard" else got_func_imports,
                "gotFuncShardDispatchImports": 7 if indirect_dispatch == "shard" else 0,
            },
        },
    }


def _write_program_dir(
    root: Path,
    *,
    program_manifest: dict | None = None,
    guest_manifest: dict | None = None,
    include_module_manifests: bool = True,
    guest_in_dist: bool = False,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    pm = program_manifest if program_manifest is not None else _program_manifest()
    (root / "program-manifest.json").write_text(json.dumps(pm), encoding="utf-8")

    if include_module_manifests:
        # Main module: 2 shards (20 + 60 instructions), 5 functions.
        _write_build_dir(root, manifest=_manifest(), discover=_discover(), nested_module="main")

        # Guest DLL 1 (video_codec): 2 shards (25 + 35 instructions), 4 functions.
        vc_manifest = _manifest(module="video_codec")
        vc_manifest["discovery"] = {
            **vc_manifest["discovery"],
            "functions": 4,
            "blocks": 8,
            "instructions": 60,
            "jump_tables": {"resolved": 1, "unresolved": 0},
        }
        vc_manifest["refinements"]["dispatch"]["address_taken_not_lifted"] = []
        vc_manifest["refinements"]["import_slots"]["referenced_unresolved"] = []
        vc_manifest["lift"] = {
            "functions": 4,
            "instructions": 60,
            "unsupported_instructions": 2,
            "unsupported_by_mnemonic": {"rdtsc": 2},
            "shards": [
                {"index": 0, "functions": 2, "instructions": 25, "ll": "lift/shard-00000.ll"},
                {"index": 1, "functions": 2, "instructions": 35, "ll": "lift/shard-00001.ll"},
            ],
        }
        _write_build_dir(root, manifest=vc_manifest, nested_module="video_codec")

        # Guest DLL 2 (audio_dsp): 1 shard (40 instructions), 3 functions, 0 unsupported.
        dsp_manifest = _manifest(module="audio_dsp")
        dsp_manifest["discovery"] = {
            **dsp_manifest["discovery"],
            "functions": 3,
            "blocks": 6,
            "instructions": 40,
            "jump_tables": {"resolved": 2, "unresolved": 0},
        }
        dsp_manifest["refinements"]["dispatch"]["address_taken_not_lifted"] = []
        dsp_manifest["refinements"]["import_slots"]["referenced_unresolved"] = []
        dsp_manifest["lift"] = {
            "functions": 3,
            "instructions": 40,
            "unsupported_instructions": 0,
            "unsupported_by_mnemonic": {},
            "shards": [
                {"index": 0, "functions": 3, "instructions": 40, "ll": "lift/shard-00000.ll"},
            ],
        }
        _write_build_dir(root, manifest=dsp_manifest, nested_module="audio_dsp")

    if guest_manifest is not None:
        target_dir = (root / "dist") if guest_in_dist else root
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "guest_manifest.json").write_text(
            json.dumps(guest_manifest), encoding="utf-8"
        )
    return root


class TestMultiModuleProgramAudit(unittest.TestCase):
    def test_program_directory_summarizes_modules_shards_imports_and_got_func(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prog_dir = _write_program_dir(
                Path(tmp) / "recomp-program",
                guest_manifest=_guest_manifest(),
            )
            report = arb.audit_recomp_build(prog_dir, top=10)
            json_code, json_out, json_err = _run_cli([str(prog_dir), "--json"])
            md_code, md_out, md_err = _run_cli([str(prog_dir), "--markdown"])
            txt_code, txt_out, txt_err = _run_cli([str(prog_dir)])

        self.assertEqual((json_code, json_err), (0, ""))
        self.assertEqual((md_code, md_err), (0, ""))
        self.assertEqual((txt_code, txt_err), (0, ""))
        self.assertEqual(json.loads(json_out)["schema"], "playgta6-recomp-audit")

        self.assertEqual(report["module"], "main")
        prog = report["program"]
        self.assertEqual(prog["module_count"], 3)
        self.assertEqual(prog["exe_count"], 1)
        self.assertEqual(prog["guest_dll_count"], 2)
        self.assertEqual(prog["guest_dlls"], ["audio_dsp.dll", "video_codec.dll"])
        self.assertEqual(prog["init_order"], ["audio_dsp.dll", "video_codec.dll", "guest_app.exe"])
        self.assertEqual(prog["indirect_dispatch"], "shard")
        self.assertEqual(prog["total_shards"], 5)  # 1 + 2 + 2
        self.assertEqual(prog["total_discovered_functions"], 12)  # 3 + 4 + 5
        self.assertEqual(prog["total_lifted_functions"], 10)  # 3 + 4 + 3 (from discover)
        self.assertEqual(prog["imports"]["guest"], 2)
        self.assertEqual(prog["imports"]["runtime"], 1)
        self.assertEqual(prog["imports"]["host_bound"], 6)
        self.assertEqual(prog["imports"]["host_unresolved"], 2)
        self.assertEqual(prog["imports"]["unresolved_recorded"], 1)
        self.assertEqual(prog["imports"]["unresolved_unrecorded"], 1)

        by_mod = {m["module"]: m for m in prog["modules"]}
        self.assertEqual(by_mod["audio_dsp"]["shard_count"], 1)
        self.assertEqual(by_mod["audio_dsp"]["lifted_functions"], 3)
        self.assertEqual(by_mod["video_codec"]["shard_count"], 2)
        self.assertEqual(by_mod["video_codec"]["imports"]["host_unresolved"], 1)
        self.assertEqual(by_mod["video_codec"]["unresolved_imports"], ["audio_ext.dll!#42"])
        self.assertEqual(by_mod["main"]["shard_count"], 2)
        self.assertIn("d3d12_xs.dll!D3D12XboxCreateDevice", by_mod["main"]["unresolved_imports"])

        # Top-level shards & coverage aggregate across all 3 modules.
        self.assertEqual(report["shards"]["count"], 5)
        self.assertEqual(report["shards"]["min"], 20)
        self.assertEqual(report["shards"]["max"], 60)
        self.assertEqual(report["coverage"]["discovered_functions"], 12)
        self.assertEqual(report["coverage"]["unsupported_instructions"], 11)  # 9 + 2 + 0
        self.assertTrue(report["boot"]["boot_targets"]["D3D12XboxCreateDevice"]["reachable"])

        # Wasm & GOT.func footprint telemetry from guest_manifest.json.
        fp = report["wasm_footprint"]
        self.assertTrue(fp["available"])
        self.assertEqual(fp["indirect_dispatch"], "shard")
        self.assertEqual(fp["total_got_func_imports"], 18)
        self.assertEqual(fp["got_func_sub_imports"], 0)
        self.assertEqual(fp["got_func_shard_dispatch_imports"], 7)
        self.assertEqual(fp["total_table_size"], 80)
        self.assertEqual(fp["main_table_size"], 64)
        self.assertEqual(fp["side_table_size"], 16)
        self.assertEqual(fp["side_module_shard_count"], 5)
        self.assertFalse(fp["any_exceeds_v8_1gib_limit"])

        # Text & Markdown include multi-module and GOT.func sections.
        self.assertIn("program modules:", txt_out)
        self.assertIn("video_codec.dll", txt_out)
        self.assertIn("wasm / GOT.func footprint:", txt_out)
        self.assertIn("GOT.func imports: 18", txt_out)
        self.assertIn("## Program modules", md_out)
        self.assertIn("## Wasm & GOT.func footprint", md_out)
        self.assertIn("`video_codec`", md_out)

    def test_program_manifest_file_without_nested_manifests_or_guest_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prog_dir = _write_program_dir(
                Path(tmp) / "prog-only",
                include_module_manifests=False,
                guest_manifest=None,
            )
            report = arb.audit_recomp_build(prog_dir / "program-manifest.json")

        self.assertEqual(report["program"]["module_count"], 3)
        self.assertEqual(report["program"]["total_lifted_functions"], 12)  # 3 + 4 + 5 from lift
        self.assertFalse(any(m["has_build_manifest"] for m in report["program"]["modules"]))
        self.assertFalse(report["wasm_footprint"]["available"])
        self.assertEqual(report["wasm_footprint"]["indirect_dispatch"], "shard")
        self.assertIsNone(report["wasm_footprint"]["total_got_func_imports"])

    def test_program_discovers_guest_manifest_in_dist_subdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prog_dir = _write_program_dir(
                Path(tmp) / "combined",
                guest_manifest=_guest_manifest(got_func_imports=24),
                guest_in_dist=True,
            )
            report = arb.audit_recomp_build(prog_dir)

        self.assertTrue(report["wasm_footprint"]["available"])
        self.assertEqual(report["wasm_footprint"]["total_got_func_imports"], 24)

    def test_guest_manifest_only_directory_and_recommendations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp) / "dist"
            dist_dir.mkdir()
            gm = _guest_manifest(
                indirect_dispatch="table",
                got_func_imports=2500,
                exceeds_v8=True,
            )
            (dist_dir / "guest_manifest.json").write_text(json.dumps(gm), encoding="utf-8")
            report = arb.audit_recomp_build(dist_dir)
            file_report = arb.audit_recomp_build(dist_dir / "guest_manifest.json")

        self.assertEqual(report["program"]["module_count"], 3)  # main + 2 guest DLLs
        self.assertEqual(report["program"]["guest_dll_count"], 2)
        self.assertEqual(report["program"]["guest_dlls"], ["audio_dsp", "video_codec"])
        self.assertEqual(report["shards"]["count"], 5)
        self.assertEqual(report["wasm_footprint"]["total_got_func_imports"], 2500)
        self.assertTrue(report["wasm_footprint"]["any_exceeds_v8_1gib_limit"])
        rec_cats = {r["category"] for r in report["recommendations"]}
        self.assertIn("wasm_size", rec_cats)
        self.assertIn("wasm_got_func", rec_cats)
        self.assertEqual(file_report["wasm_footprint"]["total_got_func_imports"], 2500)

    def test_cycle_recommendation_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pm = _program_manifest(
                cycles=[{"from": "video_codec.dll", "to": "audio_dsp.dll"}],
            )
            prog_dir = _write_program_dir(Path(tmp) / "cyclic", program_manifest=pm)
            report = arb.audit_recomp_build(prog_dir)

        rec_cats = {r["category"] for r in report["recommendations"]}
        self.assertIn("module_cycles", rec_cats)
        self.assertEqual(len(report["program"]["cycles"]), 1)


class TestFailOnUnresolvedThreshold(unittest.TestCase):
    def test_fail_on_unresolved_default_and_numeric_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prog_dir = _write_program_dir(Path(tmp) / "prog-unres")

            # 2 unresolved imports: bare flag (N=0) and N=1 fail with code 2, still emit JSON.
            code0, out0, err0 = _run_cli([str(prog_dir), "--json", "--fail-on-unresolved"])
            code1, out1, err1 = _run_cli([str(prog_dir), "--json", "--fail-on-unresolved", "1"])
            # N=2 and N=5 pass with code 0.
            code2, out2, err2 = _run_cli([str(prog_dir), "--json", "--fail-on-unresolved", "2"])
            code5, out5, err5 = _run_cli([str(prog_dir), "--json", "--fail-on-unresolved", "5"])
            # Negative threshold is an invalid argument error (exit 1).
            codeneg, outneg, errneg = _run_cli(
                [str(prog_dir), "--json", "--fail-on-unresolved", "-1"]
            )

        self.assertEqual(code0, arb.EXIT_THRESHOLD_FAILED)
        self.assertEqual(json.loads(out0)["program"]["imports"]["host_unresolved"], 2)
        self.assertIn("Threshold failed: 2 unresolved import(s)", err0)

        self.assertEqual(code1, arb.EXIT_THRESHOLD_FAILED)
        self.assertEqual(json.loads(out1)["schema"], "playgta6-recomp-audit")
        self.assertIn("--fail-on-unresolved=1", err1)

        self.assertEqual((code2, err2), (0, ""))
        self.assertEqual(json.loads(out2)["program"]["imports"]["host_unresolved"], 2)
        self.assertEqual((code5, err5), (0, ""))
        self.assertEqual(json.loads(out5)["schema"], "playgta6-recomp-audit")

        self.assertEqual(codeneg, 1)
        self.assertEqual(outneg, "")
        self.assertIn("Error:", errneg)

    def test_fail_on_unresolved_zero_passes_when_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pm = _program_manifest(unresolved_imports=[])
            prog_dir = _write_program_dir(
                Path(tmp) / "prog-clean",
                program_manifest=pm,
                include_module_manifests=False,
            )
            code, out, err = _run_cli([str(prog_dir), "--json", "--fail-on-unresolved", "0"])

        self.assertEqual((code, err), (0, ""))
        payload = json.loads(out)
        self.assertEqual(payload["program"]["imports"]["host_unresolved"], 0)
        self.assertEqual(payload["unresolved_imports"], [])

    def test_fail_on_unresolved_works_for_single_module_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "build-manifest.json"
            path.write_text(json.dumps(_manifest()), encoding="utf-8")
            code_fail, out_fail, err_fail = _run_cli([str(path), "--json", "--fail-on-unresolved"])
            code_pass, out_pass, err_pass = _run_cli(
                [str(path), "--json", "--fail-on-unresolved", "1"]
            )

        self.assertEqual(code_fail, arb.EXIT_THRESHOLD_FAILED)
        self.assertIn("Threshold failed: 1 unresolved import(s)", err_fail)
        self.assertEqual(json.loads(out_fail)["module"], "main")
        self.assertEqual((code_pass, err_pass), (0, ""))
        self.assertEqual(json.loads(out_pass)["module"], "main")


class TestMultiModuleSanitizationAndSafety(unittest.TestCase):
    def test_program_and_guest_manifests_strip_host_paths_usernames_and_reject_traversal(
        self,
    ) -> None:
        username = os.environ.get("USERNAME") or os.environ.get("USER") or "marcb"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Place a decoy manifest outside the program directory to verify traversal is blocked.
            outside = tmp_path / "outside"
            outside.mkdir()
            decoy = _manifest(module="secret_outside")
            (outside / "build-manifest.json").write_text(json.dumps(decoy), encoding="utf-8")

            prog_dir = tmp_path / "Users" / username / "recomp_out"
            pm = _program_manifest()
            pm["modules"][0]["file"] = rf"C:\Users\{username}\game\audio_dsp.dll"
            pm["modules"][0]["dll"] = f"/home/{username}/game/audio_dsp.dll"
            pm["modules"][0]["build_manifest"]["path"] = "../../../outside/build-manifest.json"
            pm["imports"].append(
                {
                    "importer": rf"C:\Users\{username}\game\audio_dsp.dll",
                    "dll": f"/Users/{username}/private/custom_ext.dll",
                    "name": "Probe",
                    "ordinal": None,
                    "slot_rva": "0x4000",
                    "delay_load": False,
                    "binding": "unresolved",
                    "provider": None,
                    "target": None,
                    "kind": None,
                    "shim": None,
                    "reason": rf"missing in C:\Users\{username}\shims",
                    "recorded": None,
                    "via": [],
                    "rule": "default",
                }
            )
            gm = _guest_manifest()
            gm["guest_dll_modules"] = [rf"C:\Users\{username}\audio_dsp", f"/home/{username}/video_codec"]
            gm["sideModules"] = [rf"C:\Users\{username}\guest_part_00.wasm"]
            gm["sideModuleSizes"][0]["name"] = rf"C:\Users\{username}\guest_part_00.wasm"
            gm["wasmMetrics"]["modules"][0]["name"] = f"/home/{username}/runtime.wasm"

            _write_program_dir(
                prog_dir,
                program_manifest=pm,
                guest_manifest=gm,
                include_module_manifests=False,
            )

            with mock.patch.dict(os.environ, {"USERNAME": username, "USER": username}):
                report = arb.audit_recomp_build(prog_dir)
                rendered = (
                    arb.format_json_report(report)
                    + arb.format_markdown_report(report)
                    + arb.format_text_report(report)
                )

        blob = json.dumps(report) + rendered
        self.assertNotIn(username, blob)
        self.assertNotIn("C:\\", blob)
        self.assertNotIn("C:/Users", blob)
        self.assertNotIn("/home/", blob)
        self.assertNotIn("/Users/", blob)
        self.assertNotIn("secret_outside", blob)
        # Path traversal module did not load the outside build-manifest.json.
        by_mod = {m["module"]: m for m in report["program"]["modules"]}
        self.assertFalse(by_mod["audio_dsp"]["has_build_manifest"])
        self.assertEqual(by_mod["audio_dsp"]["dll"], "audio_dsp.dll")
        self.assertEqual(report["wasm_footprint"]["guest_dll_modules"], ["audio_dsp", "video_codec"])


if __name__ == "__main__":
    unittest.main()
