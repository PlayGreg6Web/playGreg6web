import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import check_import_coverage as cic  # noqa: E402


class TestLiveRepoImportCoverage(unittest.TestCase):
    """Audit the real repository's `docs/imports_required.md` against live `runtime/` shims."""

    def test_parse_live_imports_required_exact_counts(self) -> None:
        md_text = (REPO_ROOT / cic.IMPORTS_REQUIRED_REL).read_text(encoding="utf-8")
        parsed = cic.parse_imports_required(md_text)

        self.assertEqual(len(parsed), 68, "Expected exactly 68 imported DLLs")
        total_symbols = sum(len(entries) for entries in parsed.values())
        self.assertEqual(total_symbols, 587, "Expected exactly 587 total imported symbols")

        # Spot-check both named and ordinal entries from the real binary report.
        self.assertIn("GameInput.dll", parsed)
        self.assertEqual(len(parsed["GameInput.dll"]), 1)
        self.assertEqual(parsed["GameInput.dll"][0]["kind"], "ordinal")
        self.assertEqual(parsed["GameInput.dll"][0]["symbol"], "#1")
        self.assertEqual(parsed["GameInput.dll"][0]["ordinal"], 1)

        self.assertIn("MSVCP140.dll", parsed)
        self.assertEqual(len(parsed["MSVCP140.dll"]), 113)

        self.assertIn("WS2_32.dll", parsed)
        self.assertEqual(len(parsed["WS2_32.dll"]), 25)
        ws2_ordinals = [e["ordinal"] for e in parsed["WS2_32.dll"] if e["kind"] == "ordinal"]
        self.assertIn(111, ws2_ordinals)
        self.assertIn(115, ws2_ordinals)

    def test_live_audit_coverage_and_dll_classifications(self) -> None:
        report = cic.audit_import_coverage(REPO_ROOT)
        self.assertTrue(report["ok"], msg=f"Audit failed: {report['errors']}")
        self.assertEqual(report["errors"], [])

        totals = report["totals"]
        self.assertEqual(totals["dll_count"], 68)
        self.assertEqual(totals["total"], 587)
        self.assertEqual(
            totals["shimmed"] + totals["guest_recompiled"] + totals["missing"],
            totals["total"],
        )
        self.assertGreater(totals["effective_boot_pct"], 50.0)
        self.assertGreater(totals["shimmed_pct"], 50.0)

        by_dll = {d["dll"]: d for d in report["dlls"]}

        # O13 UCRT + VCRUNTIME140/140_1 should be 100% shimmed.
        for dll in (
            "api-ms-win-crt-math-l1-1-0.dll",
            "api-ms-win-crt-runtime-l1-1-0.dll",
            "api-ms-win-crt-stdio-l1-1-0.dll",
            "api-ms-win-crt-string-l1-1-0.dll",
            "VCRUNTIME140.dll",
            "VCRUNTIME140_1.dll",
        ):
            self.assertEqual(by_dll[dll]["missing"], 0, msg=f"Expected {dll} fully shimmed")

        # O12 Xbox Scarlett (including GameInput.dll ordinal #1) should be 100% shimmed.
        for dll in ("xmem.dll", "XCRDAPI.dll", "xg_xs.dll", "GameInput.dll"):
            self.assertEqual(by_dll[dll]["missing"], 0, msg=f"Expected {dll} fully shimmed")

        # O14 Xbox Middleware should be 100% shimmed (including mangled XboxBvhBuilder.dll names).
        for dll in (
            "libHttpClient.GDK.dll",
            "Bink2Scarlett.dll",
            "AcpHal.dll",
            "xapu.dll",
            "chad.dll",
            "amd_fidelityfx_loader_xbox.dll",
            "XboxBvhBuilder.dll",
            "MFPlat.DLL",
            "MFReadWrite.dll",
        ):
            self.assertEqual(by_dll[dll]["missing"], 0, msg=f"Expected {dll} fully shimmed")

        # R12 guest-recompiled DLL classification.
        self.assertEqual(by_dll["MSVCP140.dll"]["status"], "guest_recompiled")
        self.assertEqual(by_dll["MSVCP140.dll"]["guest_recompiled"], 113)
        self.assertEqual(by_dll["MSVCP140.dll"]["missing"], 0)

        # API-set forwarding (`api-ms-win-core-*` -> `KERNEL32.dll`).
        self.assertEqual(by_dll["api-ms-win-core-handle-l1-1-0.dll"]["shimmed"], 1)
        self.assertEqual(by_dll["api-ms-win-core-libraryloader-l1-2-0.dll"]["missing"], 0)
        self.assertEqual(by_dll["api-ms-win-core-profile-l1-1-0.dll"]["missing"], 0)

    def test_live_cli_json_is_valid_and_sanitized(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "tools" / "check_import_coverage.py"), "--json"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        parsed = json.loads(proc.stdout)
        self.assertTrue(parsed["ok"])
        self.assertEqual(parsed["totals"]["dll_count"], 68)
        self.assertEqual(parsed["totals"]["total"], 587)
        self.assertGreater(parsed["totals"]["effective_boot_pct"], 50.0)

        # Strict sanitization: zero host paths in JSON output.
        raw_json = proc.stdout
        self.assertNotIn(str(REPO_ROOT), raw_json)
        self.assertNotIn(str(REPO_ROOT).replace("\\", "/"), raw_json)
        self.assertNotIn(str(Path.home()), raw_json)
        self.assertNotIn(str(Path.home()).replace("\\", "/"), raw_json)


class TestSyntheticImportCoverage(unittest.TestCase):
    """Test parser, scanner, API-set forwarding, ordinals, guest DLLs, and CLI flags on fixtures."""

    def _write(self, root: Path, rel: str, content: str) -> None:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def _build_synthetic_repo(self, root: Path) -> Path:
        self._write(
            root,
            "docs/imports_required.md",
            """# PE64 Inspection Report — `synthetic.exe`

## Imports

### `api-ms-win-core-file-l1-1-0.dll` (2 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `CreateFileW` | hint `0x000A` |
| name | `MissingFileApi` | hint `0x000B` |

### `api-ms-win-core-memory-l1-1-0.dll` (1 symbol)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `VirtualAlloc` | hint `0x002F` |

### `GameInput.dll` (1 symbol)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| ordinal | `#1` | ordinal `1` |

### `WS2_32.dll` (2 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| ordinal | `#111` | ordinal `111` |
| ordinal | `#115` | ordinal `115` |

### `d3d12_xs.dll` (2 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `D3D12XboxCreateDevice` | hint `0x0009` |
| name | `D3D12ConfigureVirtualMemoryX` | hint `0x0001` |

### `MSVCP140.dll` (2 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `??0ios_base@std@@IEAA@XZ` | hint `0x0077` |
| name | `_Mtx_lock` | hint `0x0585` |

## Exports

| Ordinal | RVA | Name | Forwarder |
| --- | --- | --- | --- |
| 1 | `0x1000` | `IgnoredExport` | - |
""",
        )

        # Kernel32/KernelBase + ordinal + alias + register_symbol + register_ordinal patterns.
        self._write(
            root,
            "runtime/os/synthetic_shims.cpp",
            """#include <cstdint>
void register_all() {
    register_export("KERNEL32.dll", "CreateFileW", 0, 1);
    register_symbol("kernelbase.dll", "VirtualAlloc", 2);
    register_exports("GameInput.dll", {
        export_entry("GameInputCreate", &dummy_fn, 1u),
    });
    register_ordinal("ws2_32.dll", 111, 3);
}
""",
        )

        # Header-only extern "C" pg6_d3d12_* declaration (no cpp registry entry yet).
        self._write(
            root,
            "runtime/gpu/d3d12_bridge.h",
            """#pragma once
extern "C" {
int pg6_d3d12_xbox_create_device(void* params, void** out_dev);
}
""",
        )

        return root

    def test_synthetic_named_ordinal_apiset_header_and_guest_recompiled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._build_synthetic_repo(Path(tmp))
            report = cic.audit_import_coverage(repo)
            self.assertTrue(report["ok"])

            totals = report["totals"]
            # Total: 2 + 1 + 1 + 2 + 2 + 2 = 10
            # Shimmed: CreateFileW (k32 via api-set), VirtualAlloc (kernelbase via api-set),
            #          GameInput #1 (ordinal), WS2_32 #111 (ordinal), D3D12XboxCreateDevice (header) = 5
            # Guest recompiled: MSVCP140.dll = 2
            # Missing: MissingFileApi, WS2_32 #115, D3D12ConfigureVirtualMemoryX = 3
            self.assertEqual(totals["dll_count"], 6)
            self.assertEqual(totals["total"], 10)
            self.assertEqual(totals["shimmed"], 5)
            self.assertEqual(totals["guest_recompiled"], 2)
            self.assertEqual(totals["missing"], 3)
            self.assertEqual(totals["shimmed_pct"], 50.0)
            self.assertEqual(totals["effective_boot_pct"], 70.0)

            by_dll = {d["dll"]: d for d in report["dlls"]}
            self.assertEqual(by_dll["api-ms-win-core-file-l1-1-0.dll"]["shimmed_symbols"], ["CreateFileW"])
            self.assertEqual(by_dll["api-ms-win-core-file-l1-1-0.dll"]["missing_symbols"], ["MissingFileApi"])
            self.assertEqual(by_dll["api-ms-win-core-memory-l1-1-0.dll"]["shimmed_symbols"], ["VirtualAlloc"])
            self.assertEqual(by_dll["GameInput.dll"]["shimmed_symbols"], ["#1"])
            self.assertEqual(by_dll["WS2_32.dll"]["shimmed_symbols"], ["#111"])
            self.assertEqual(by_dll["WS2_32.dll"]["missing_symbols"], ["#115"])
            self.assertEqual(by_dll["d3d12_xs.dll"]["shimmed_symbols"], ["D3D12XboxCreateDevice"])
            self.assertEqual(by_dll["d3d12_xs.dll"]["missing_symbols"], ["D3D12ConfigureVirtualMemoryX"])
            self.assertEqual(by_dll["MSVCP140.dll"]["status"], "guest_recompiled")

            # Passing empty guest_recompiled_dlls marks MSVCP140.dll as missing.
            no_guest = cic.audit_import_coverage(repo, guest_recompiled_dlls=set())
            self.assertEqual(no_guest["totals"]["guest_recompiled"], 0)
            self.assertEqual(no_guest["totals"]["missing"], 5)
            self.assertEqual(no_guest["totals"]["effective_boot_pct"], 50.0)

    def test_synthetic_register_alias_and_macros(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(
                root,
                "docs/imports_required.md",
                """### `CustomAlias.dll` (2 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `AlphaFunc` | hint `0x0001` |
| ordinal | `#7` | ordinal `7` |
""",
            )
            self._write(
                root,
                "runtime/guest/alias_reg.cpp",
                """#define TARGET_DLL "RealTarget.dll"
void init() {
    register_alias("CustomAlias.dll", TARGET_DLL);
    REGISTER_EXPORT(TARGET_DLL, "AlphaFunc", 10);
    REGISTER_ORDINAL(TARGET_DLL, 7, 11);
}
""",
            )
            report = cic.audit_import_coverage(root)
            self.assertTrue(report["ok"])
            self.assertEqual(report["totals"]["shimmed"], 2)
            self.assertEqual(report["totals"]["missing"], 0)
            self.assertEqual(report["totals"]["effective_boot_pct"], 100.0)

    def test_cli_missing_only_min_pct_and_sanitized_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._build_synthetic_repo(Path(tmp))
            script = str(REPO_ROOT / "tools" / "check_import_coverage.py")

            # Coverage is 70.0%: --min-pct 70 passes (exit 0).
            proc_pass = subprocess.run(
                [sys.executable, script, "--repo-root", str(repo), "--min-pct", "70", "--missing-only", "--json"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_pass.returncode, 0, msg=proc_pass.stderr or proc_pass.stdout)
            data = json.loads(proc_pass.stdout)
            self.assertTrue(data["missing_only"])
            self.assertEqual(data["totals"]["effective_boot_pct"], 70.0)
            # Only the 3 DLLs with missing symbols should remain in filtered list.
            self.assertEqual(
                [d["dll"] for d in data["dlls"]],
                ["api-ms-win-core-file-l1-1-0.dll", "WS2_32.dll", "d3d12_xs.dll"],
            )
            # Ensure temp directory / host path never leaks into JSON.
            self.assertNotIn(str(repo), proc_pass.stdout)
            self.assertNotIn(str(repo).replace("\\", "/"), proc_pass.stdout)

            # Coverage is 70.0%: --min-pct 75 fails (exit 1).
            proc_fail = subprocess.run(
                [sys.executable, script, "--repo-root", str(repo), "--min-pct", "75"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_fail.returncode, 1)
            self.assertIn("Effective Boot Coverage: 7/10 (70.00%)", proc_fail.stdout)

            # Human-readable --missing-only lists missing symbol names and omits complete DLLs.
            proc_human = subprocess.run(
                [sys.executable, script, "--repo-root", str(repo), "--missing-only"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_human.returncode, 0)
            self.assertIn("MissingFileApi", proc_human.stdout)
            self.assertIn("#115", proc_human.stdout)
            self.assertNotIn("MSVCP140.dll", proc_human.stdout)
            self.assertNotIn("GameInput.dll", proc_human.stdout)

    def test_missing_imports_doc_exits_nonzero_and_sanitizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty_repo = Path(tmp)
            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "check_import_coverage.py"),
                    "--repo-root",
                    str(empty_repo),
                    "--json",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 1)
            data = json.loads(proc.stdout)
            self.assertFalse(data["ok"])
            self.assertNotIn(str(empty_repo), proc.stdout)


def _bridge_import(dll, name=None, ordinal=None, status="unresolved", **extra):
    entry = {"dll": dll, "name": name, "ordinal": ordinal, "status": status}
    entry.update(extra)
    return entry


class TestBridgeManifestAudit(unittest.TestCase):
    """O17: binding classes from an import bridge / program manifest (recomp/src/stubs.rs)."""

    BRIDGE = {
        "format": "playgta6-import-bridge",
        "version": 1,
        "imports": [
            _bridge_import("KERNEL32.dll", "Sleep", status="bound", shim="pg6_win32_sleep"),
            _bridge_import(
                "api-ms-win-crt-utility-l1-1-0.dll", "qsort", status="bound", shim="pg6_gthunk_qsort"
            ),
            _bridge_import("MSVCP140.dll", "?foo@@YAXXZ", reason="unknown_dll", recorded=None),
            _bridge_import(
                "VCRUNTIME140.dll", "memcpy", status="bound", shim="pg6_crt_memcpy"
            ),
            _bridge_import(
                "api-ms-win-core-rtlsupport-l1-1-0.dll", "RtlVirtualUnwind", reason="no_shim"
            ),
            _bridge_import(
                "MFPlat.DLL",
                "MFCreateAttributes",
                reason="denied",
                recorded="denied: host COM: IUnknown stub with a host vtable",
            ),
            _bridge_import("WS2_32.dll", ordinal=11, reason="unmarshalable", recorded="unmarshalable: x"),
        ],
    }

    def test_classifies_every_binding_class(self) -> None:
        report = cic.audit_bridge_manifest(self.BRIDGE)
        self.assertTrue(report["ok"], report["unrecorded"])
        self.assertEqual(report["kind"], "bridge")
        totals = report["totals"]
        self.assertEqual(
            {k: totals[k] for k in cic.BINDING_CLASSES},
            {"guest": 2, "runtime": 1, "gthunk": 1, "host": 1, "unresolved": 2},
        )
        self.assertEqual(totals["total"], 7)
        labels = [u["import"] for u in report["unresolved"]]
        self.assertEqual(labels, ["MFPlat.DLL!MFCreateAttributes", "WS2_32.dll!#11"])
        self.assertEqual(report["dlls"]["MSVCP140.dll"]["guest"], 1)

    def test_unrecorded_unresolved_import_fails(self) -> None:
        manifest = dict(self.BRIDGE)
        manifest["imports"] = self.BRIDGE["imports"] + [
            _bridge_import("xmem.dll", "XMemNew", reason="no_shim", recorded=None)
        ]
        report = cic.audit_bridge_manifest(manifest)
        self.assertFalse(report["ok"])
        self.assertEqual(report["unrecorded"], ["xmem.dll!XMemNew"])
        # Without the declared guest policy the C++ runtime imports are unresolved too.
        report = cic.audit_bridge_manifest(self.BRIDGE, guest_dlls=[])
        self.assertIn("MSVCP140.dll!?foo@@YAXXZ", report["unrecorded"])
        self.assertEqual(report["totals"]["host"], 2)  # VCRUNTIME140!memcpy on the host now

    def test_program_manifest_bindings(self) -> None:
        program = {
            "schema": "playgta6-recomp-program",
            "imports": [
                {"importer": "game.exe", "dll": "MSVCP140.dll", "name": "f", "ordinal": None,
                 "binding": "guest", "shim": None, "reason": None},
                {"importer": "game.exe", "dll": "KERNEL32.dll", "name": "RtlUnwindEx",
                 "ordinal": None, "binding": "runtime", "shim": None, "reason": None},
                {"importer": "msvcp140.dll", "dll": "api-ms-win-crt-runtime-l1-1-0.dll",
                 "name": "_initterm", "ordinal": None, "binding": "host",
                 "shim": "pg6_gthunk_initterm", "reason": None},
                {"importer": "msvcp140.dll", "dll": "api-ms-win-crt-math-l1-1-0.dll",
                 "name": "floor", "ordinal": None, "binding": "unresolved", "shim": None,
                 "reason": "no_shim", "recorded": None},
            ],
        }
        report = cic.audit_bridge_manifest(program)
        self.assertEqual(report["kind"], "program")
        self.assertFalse(report["ok"])
        self.assertEqual(
            report["unrecorded"], ["msvcp140.dll: api-ms-win-crt-math-l1-1-0.dll!floor"]
        )
        self.assertEqual(report["importers"]["game.exe"]["runtime"], 1)
        self.assertEqual(report["importers"]["msvcp140.dll"]["gthunk"], 1)

    def test_cli_bridge_mode_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.json"
            good.write_text(json.dumps(self.BRIDGE), encoding="utf-8")
            bad_manifest = dict(self.BRIDGE)
            bad_manifest["imports"] = [_bridge_import("chad.dll", "XDspNew", reason="no_shim")]
            bad = Path(tmp) / "bad.json"
            bad.write_text(json.dumps(bad_manifest), encoding="utf-8")
            tool = str(REPO_ROOT / "tools" / "check_import_coverage.py")
            ok = subprocess.run(
                [sys.executable, tool, "--bridge-manifest", str(good), "--json"],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertEqual(json.loads(ok.stdout)["totals"]["gthunk"], 1)
            fail = subprocess.run(
                [sys.executable, tool, "--bridge-manifest", str(bad)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(fail.returncode, 1)
            self.assertIn("NO RECORDED REASON", fail.stdout)

    def test_runtime_exception_apis_match_the_program_builder(self) -> None:
        source = (REPO_ROOT / "recomp" / "src" / "pipeline" / "program.rs").read_text(
            encoding="utf-8"
        )
        block = source.split("pub const RUNTIME_EXCEPTION_APIS", 1)[1].split("];", 1)[0]
        pairs = {
            (dll.lower(), name)
            for dll, name in re.findall(r'\("([^"]+)",\s*"([^"]+)"\)', block)
        }
        self.assertEqual(pairs, set(cic.RUNTIME_EXCEPTION_APIS))


if __name__ == "__main__":
    unittest.main()
