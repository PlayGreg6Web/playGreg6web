import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import repo_health  # noqa: E402

REQUIRED_REPO_FILES: tuple[str, ...] = (
    repo_health.GPU_ABI_HEADER_REL,
    repo_health.GPU_TRACE_PY_REL,
    repo_health.WGPU_WORKER_JS_REL,
    repo_health.SMOKE_SCENE_JS_REL,
    repo_health.INPUT_BRIDGE_JS_REL,
    repo_health.HTTPFS_HEADER_REL,
    repo_health.HTTPFS_JS_REL,
    repo_health.WORKSTREAMS_MD_REL,
)


def _copy_minimal_repo(dest: Path) -> Path:
    """Copy the ABI-checked files from the real repo into an isolated temp repo."""
    for rel in REQUIRED_REPO_FILES:
        src = REPO_ROOT / rel
        dst = dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return dest


class TestRepoHealthRealRepo(unittest.TestCase):
    """Run the full verifier against the real repository to catch live ABI drift in CI."""

    def test_real_repo_passes_all_checks(self) -> None:
        report = repo_health.check_repo_health(REPO_ROOT, strict=True)
        self.assertTrue(
            report["ok"],
            msg=f"Real repo health check failed:\n{repo_health.format_human_report(report)}",
        )
        self.assertEqual(report["error_count"], 0)
        self.assertEqual(report["warning_count"], 0)

        checks = report["checks"]
        gpu = checks["gpu_opcode_parity"]
        self.assertTrue(gpu["ok"])
        self.assertEqual(gpu["opcode_count"], 36)
        self.assertEqual(gpu["min_opcode"], 0)
        self.assertEqual(gpu["max_opcode"], 35)
        for target_rel in (
            repo_health.GPU_TRACE_PY_REL,
            repo_health.WGPU_WORKER_JS_REL,
            repo_health.SMOKE_SCENE_JS_REL,
        ):
            self.assertTrue(gpu["targets"][target_rel]["ok"])
            self.assertEqual(gpu["targets"][target_rel]["count"], 36)

        inp = checks["input_state_parity"]
        self.assertTrue(inp["ok"])
        self.assertEqual(inp["cpp_static_assert_bytes"], 40)
        self.assertEqual(inp["cpp_computed_bytes"], 40)
        self.assertEqual(inp["js_input_state_bytes"], 40)
        self.assertEqual(inp["cpp_field_offsets"], inp["js_field_offsets"])

        httpfs = checks["httpfs_mailbox_parity"]
        self.assertTrue(httpfs["ok"])
        self.assertEqual(httpfs["cpp_constants"]["mailbox_version"], 1)
        self.assertEqual(httpfs["cpp_constants"]["header_bytes"], 40)
        self.assertEqual(httpfs["cpp_constants"]["payload_bytes"], 65536)
        self.assertEqual(httpfs["cpp_offsets"], httpfs["js_offsets"])

        hyg = checks["tracked_hygiene"]
        self.assertTrue(hyg["ok"])
        self.assertEqual(hyg["forbidden_files"], [])
        self.assertEqual(hyg["oversized_binaries"], [])
        self.assertGreater(hyg["tracked_file_count"], 0)

        ws = checks["workstreams_summary"]
        self.assertTrue(ws["ok"])
        self.assertEqual(len(ws["tracks"]), 5)
        self.assertGreater(ws["totals"]["total"], 0)
        self.assertEqual(
            ws["totals"]["done"] + ws["totals"]["in_progress"] + ws["totals"]["unclaimed"],
            ws["totals"]["total"],
        )

    def test_cli_json_is_valid_and_sanitized(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "tools" / "repo_health.py"), "--json"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        parsed = json.loads(proc.stdout)
        self.assertTrue(parsed["ok"])
        self.assertEqual(parsed["error_count"], 0)

        # Ensure zero host filesystem paths leaked into JSON output
        raw_json = proc.stdout
        self.assertNotIn(str(REPO_ROOT), raw_json)
        self.assertNotIn(str(REPO_ROOT).replace("\\", "/"), raw_json)
        self.assertNotIn(str(Path.home()), raw_json)
        self.assertNotIn(str(Path.home()).replace("\\", "/"), raw_json)


class TestRepoHealthSyntheticDrift(unittest.TestCase):
    """Test every check against synthetic drift & hygiene violation scenarios."""

    def test_mismatched_gpu_opcode_fails_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _copy_minimal_repo(Path(tmp))
            worker_path = repo / repo_health.WGPU_WORKER_JS_REL
            original = worker_path.read_text(encoding="utf-8")
            mutated = original.replace("DISPATCH_COMPUTE: 21,", "DISPATCH_COMPUTE: 99,")
            self.assertNotEqual(original, mutated)
            worker_path.write_text(mutated, encoding="utf-8")

            res = repo_health.check_gpu_opcode_parity(repo)
            self.assertFalse(res["ok"])
            self.assertFalse(res["targets"][repo_health.WGPU_WORKER_JS_REL]["ok"])
            self.assertEqual(
                res["targets"][repo_health.WGPU_WORKER_JS_REL]["mismatched"],
                [{"name": "DISPATCH_COMPUTE", "expected": 21, "actual": 99}],
            )
            self.assertTrue(res["targets"][repo_health.GPU_TRACE_PY_REL]["ok"])
            self.assertTrue(res["targets"][repo_health.SMOKE_SCENE_JS_REL]["ok"])

            overall = repo_health.check_repo_health(repo)
            self.assertFalse(overall["ok"])
            self.assertGreaterEqual(overall["error_count"], 1)

    def test_missing_gpu_opcode_in_python_and_smoke_scene_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _copy_minimal_repo(Path(tmp))
            trace_path = repo / repo_health.GPU_TRACE_PY_REL
            trace_path.write_text(
                trace_path.read_text(encoding="utf-8").replace("    DispatchCompute = 21\n", ""),
                encoding="utf-8",
            )
            smoke_path = repo / repo_health.SMOKE_SCENE_JS_REL
            smoke_path.write_text(
                smoke_path.read_text(encoding="utf-8").replace("  DRAW_INDEXED: 18,\n", ""),
                encoding="utf-8",
            )

            res = repo_health.check_gpu_opcode_parity(repo)
            self.assertFalse(res["ok"])
            self.assertEqual(
                res["targets"][repo_health.GPU_TRACE_PY_REL]["missing"],
                ["DispatchCompute"],
            )
            self.assertEqual(
                res["targets"][repo_health.SMOKE_SCENE_JS_REL]["missing"],
                ["DRAW_INDEXED"],
            )

    def test_bad_input_state_size_and_offset_fails_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _copy_minimal_repo(Path(tmp))
            bridge_path = repo / repo_health.INPUT_BRIDGE_JS_REL
            text = bridge_path.read_text(encoding="utf-8")
            mutated = text.replace(
                "export const INPUT_STATE_BYTES = 40;",
                "export const INPUT_STATE_BYTES = 44;",
            ).replace("RIGHT_TRIGGER: 36,", "RIGHT_TRIGGER: 40,")
            bridge_path.write_text(mutated, encoding="utf-8")

            res = repo_health.check_input_state_parity(repo)
            self.assertFalse(res["ok"])
            self.assertEqual(res["cpp_static_assert_bytes"], 40)
            self.assertEqual(res["js_input_state_bytes"], 44)
            self.assertTrue(any("byte size drift" in e for e in res["errors"]))
            self.assertTrue(any("RIGHT_TRIGGER" in e for e in res["errors"]))

    def test_httpfs_mailbox_constant_and_offset_drift_fails_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _copy_minimal_repo(Path(tmp))
            js_path = repo / repo_health.HTTPFS_JS_REL
            text = js_path.read_text(encoding="utf-8")
            mutated = (
                text.replace("const HTTPFS_PAYLOAD_BYTES = 64 * 1024;", "const HTTPFS_PAYLOAD_BYTES = 32 * 1024;")
                .replace("RESULT: 32,", "RESULT: 36,")
                .replace("ResponseReady: 2,", "ResponseReady: 7,")
            )
            js_path.write_text(mutated, encoding="utf-8")

            res = repo_health.check_httpfs_mailbox_parity(repo)
            self.assertFalse(res["ok"])
            self.assertTrue(any("payload_bytes" in e for e in res["errors"]))
            self.assertTrue(any("HttpfsOffsets.RESULT" in e for e in res["errors"]))
            self.assertTrue(any("HttpfsStatus.ResponseReady" in e for e in res["errors"]))

    def test_forbidden_rpf_and_env_tracked_file_fails_hygiene_git_and_walk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _copy_minimal_repo(Path(tmp))
            (repo / "update").mkdir(parents=True, exist_ok=True)
            (repo / "update" / "dlc.rpf").write_bytes(b"RPF9dummy")
            (repo / ".env").write_text("GTA6_GAME_DIR=/secret\n", encoding="utf-8")

            # Fallback walk detection (non-git temp dir)
            walk_res = repo_health.check_tracked_hygiene(repo)
            self.assertFalse(walk_res["ok"])
            self.assertEqual(walk_res["source"], "walk")
            self.assertIn("update/dlc.rpf", walk_res["forbidden_files"])
            self.assertIn(".env", walk_res["forbidden_files"])

            # Git ls-files detection when tracked in a real git repo
            if shutil.which("git"):
                subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
                subprocess.run(["git", "-C", str(repo), "add", "update/dlc.rpf", ".env"], check=True)
                git_res = repo_health.check_tracked_hygiene(repo)
                self.assertFalse(git_res["ok"])
                self.assertEqual(git_res["source"], "git")
                self.assertIn("update/dlc.rpf", git_res["forbidden_files"])
                self.assertIn(".env", git_res["forbidden_files"])

    def test_oversized_binary_blob_fails_hygiene(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _copy_minimal_repo(Path(tmp))
            blob = repo / "assets_blob.dat"
            blob.write_bytes(b"\x00\x01\x02\x03" * 512)  # 2048 bytes binary

            res = repo_health.check_tracked_hygiene(repo, max_binary_bytes=1024)
            self.assertFalse(res["ok"])
            self.assertEqual(res["oversized_binaries"], [{"path": "assets_blob.dat", "bytes": 2048}])

            # Large UTF-8 text file over threshold should NOT be flagged as binary blob
            blob.unlink()
            text_file = repo / "large_source.rs"
            text_file.write_text("a" * 2048, encoding="utf-8")
            clean_res = repo_health.check_tracked_hygiene(repo, max_binary_bytes=1024)
            self.assertTrue(clean_res["ok"])

    def test_workstreams_summary_counting_and_strict_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _copy_minimal_repo(Path(tmp))
            ws_path = repo / repo_health.WORKSTREAMS_MD_REL
            ws_path.write_text(
                "\n".join(
                    [
                        "# Workstreams",
                        "",
                        "## Track 1: Recompiler",
                        "- [x] **R1 — Done task** — `DONE` (PR #1)",
                        "  - Sub-bullet should not be counted",
                        "- [ ] **R2 — Active task** — `IN PROGRESS` (#2, @alice)",
                        "- [ ] **R3 — Open task** *(unclaimed)*",
                        "",
                        "## Track 2: Runtime",
                        "- [x] **O1 — Done task** — `DONE` (PR #3)",
                        "- [x] **O2 — Also done** — `DONE` (PR #4)",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            res = repo_health.summarize_workstreams(repo)
            self.assertTrue(res["ok"])
            self.assertEqual(res["warnings"], [])
            self.assertEqual(len(res["tracks"]), 2)
            self.assertEqual(res["tracks"][0]["done"], 1)
            self.assertEqual(res["tracks"][0]["in_progress"], 1)
            self.assertEqual(res["tracks"][0]["unclaimed"], 1)
            self.assertEqual(res["tracks"][0]["total"], 3)
            self.assertEqual(res["tracks"][1]["done"], 2)
            self.assertEqual(res["tracks"][1]["in_progress"], 0)
            self.assertEqual(res["tracks"][1]["unclaimed"], 0)
            self.assertEqual(res["totals"]["total"], 5)
            self.assertEqual(res["totals"]["done"], 3)
            self.assertEqual(res["totals"]["in_progress"], 1)
            self.assertEqual(res["totals"]["unclaimed"], 1)
            self.assertEqual(res["totals"]["completion_pct"], 60.0)

            # Now introduce a checkbox/status mismatch warning and check --strict behavior
            ws_path.write_text(
                "\n".join(
                    [
                        "## Track 1: Recompiler",
                        "- [x] **R1 — Contradictory** — `IN PROGRESS` (#9)",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            non_strict = repo_health.check_repo_health(repo, strict=False)
            self.assertTrue(non_strict["ok"])
            self.assertEqual(non_strict["warning_count"], 1)

            strict_res = repo_health.check_repo_health(repo, strict=True)
            self.assertFalse(strict_res["ok"])
            self.assertEqual(strict_res["warning_count"], 1)

    def test_cli_returns_nonzero_on_drift_and_sanitizes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _copy_minimal_repo(Path(tmp))
            (repo / "game.xvd").write_bytes(b"msft-xvd")

            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "repo_health.py"),
                    "--repo-root",
                    str(repo),
                    "--json",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 1)
            payload = json.loads(proc.stdout)
            self.assertFalse(payload["ok"])
            self.assertIn("game.xvd", payload["checks"]["tracked_hygiene"]["forbidden_files"])
            # Confirm temp host path is never leaked in JSON
            self.assertNotIn(str(repo), proc.stdout)
            self.assertNotIn(str(repo).replace("\\", "/"), proc.stdout)


if __name__ == "__main__":
    unittest.main()
