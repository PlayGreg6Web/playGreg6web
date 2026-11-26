import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import boot_summary as bs  # noqa: E402


def base():
    return {
        "status": "limit_reached", "phase": "entry", "elapsed_ms": 90000,
        "message": "run bound reached",
        "counters": {"dispatches": 1234567},
        "milestones": [{"milestone": "PGIM_LOAD", "elapsed_ms": 200}, {"milestone": "FIRST_PRESENT", "elapsed_ms": 3400}],
        "present_samples_total": 5,
        "present_samples": [{"wall_ms": t} for t in (0, 33, 66, 100, 133, 400)],
        "screen_timeline": [{"start_time": 0, "end_time": 4000, "presents": 5, "draws_per_frame": 37}],
        "gpu_work": {"draws_no_shader_module": 0,
                     "drops_by_program": {"0x1": 5, "0x2": 50, "0x3": 7, "0x4": 1, "0x5": 9, "0x6": 3, "0x7": 0}},
        "native_executor": {"device_errors": 2, "shader_pack_hits": 10, "shader_pack_misses": 3, "frames_submitted": 5},
        "audio": {"xapu_decoded_packets": 152, "xapu_failed_packets": 0, "xapu_commands": 168, "non_silent_fraction": 0.5},
        "audio_files": {"audio_opens": 4, "audio_open_ok": 3, "audio_open_failed": 1},
        "audio_out": {"device_open": True, "frames_played": 100, "underruns": 7, "error": ""},
        "bink": {"binding": "shim", "total_calls": 9, "exports": [{"name": "BinkOpen", "calls": 9}]},
        "memory_budget": {"host_private_bytes": 1048576, "gpu_visible_bytes": 0,
                          "guest_arena": {"arena_peak_committed": 3 * 1024 ** 3}},
        "pipeline_prewarm": {"prewarmed": 8, "hit_before_use": 6, "wasted": 2, "failed": 0},
    }


class BootSummaryTest(unittest.TestCase):
    def test_limit_run(self):
        out = bs.summarize(base())
        self.assertIn("ended: limit", out)
        self.assertIn("device_errors=2", out)
        self.assertIn("10 hits, 3 misses", out)
        self.assertIn("0x2=50, 0x5=9, 0x3=7, 0x1=5, 0x6=3", out)
        self.assertNotIn("0x7", out)
        self.assertIn("3 ok, 1 failed", out)
        self.assertIn("7 underruns", out)
        self.assertIn("BinkOpen=9", out)
        self.assertIn("arena peak 3.0 GiB", out)
        self.assertIn("6 hit before use", out)
        self.assertIn("fps:", out)

    def test_missing_sections_skipped(self):
        out = bs.summarize({"status": "limit_reached", "elapsed_ms": 1000})
        self.assertIn("gpu errors: none", out)
        self.assertNotIn("bink", out)
        self.assertNotIn("prewarm", out)

    def test_fault_with_hard_stop(self):
        r = base()
        r["status"] = "fault"
        r["fault_record"] = {"kind": "unimplemented", "elapsed_ms": 5000, "rip": "0x140001000", "target": "0x0",
                             "function": {"entry": "0x140001000", "offset": "0x10"}}
        r["message"] = "com bridge: unimplemented xgameruntime.XStore {00000000-0000-0000-0000-000000000000} slot 66 (XStoreRegisterGameLicenseChanged)"
        md = "## 1. XStore (`xgameruntime.XStore`)\n\n| Slot | Function | Today | Set by |\n|---|---|---|---|\n| 66 / 67 | Reg | Trapping stub | #235 |\n"
        out = bs.summarize(r, None, md)
        self.assertIn("FAULT unimplemented", out)
        self.assertIn("slot 66", out)
        self.assertIn("hard stop: Reg -> Trapping stub", out)

    def test_structured_slot_and_identity(self):
        r = base()
        r["status"] = "fault"
        r["build"] = {"source_commit": "abcdef0123456789", "dirty": True, "profile": "Release", "executor": True,
                      "recomp_hash": "0011"}
        r["run"] = {"preset": "menu-flow", "name": "n1"}
        r["fault_record"] = {"kind": "unimplemented_slot", "elapsed_ms": 5000, "rip": "0x1", "target": "0x0",
                             "unimplemented_slot": {"interface": "XSystem", "guid": "g", "slot": 7,
                                                    "name": "XSystemGetAppSpecificDeviceId"}}
        md = "## 1. XSystem\n\n| Slot | Function | Today | Set by |\n|---|---|---|---|\n| 7 | XSystemGetAppSpecificDeviceId | Trapping stub | #1 |\n"
        out = bs.summarize(r, None, md)
        self.assertIn("build: abcdef0123+dirty, Release, executor, recomp 0011", out)
        self.assertIn("launch: preset menu-flow, name n1", out)
        self.assertIn("com slot: XSystem slot 7 XSystemGetAppSpecificDeviceId", out)
        self.assertIn("hard stop: XSystemGetAppSpecificDeviceId -> Trapping stub", out)

    def test_compare_and_cli(self):
        a, b = base(), base()
        b["native_executor"]["device_errors"] = 0
        b["elapsed_ms"] = 100000
        text = bs.compare(a, b)
        self.assertIn("native_executor.device_errors", text)
        self.assertIn("-2", text)
        self.assertIn("+10,000", text)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "boot-report.json").write_text(json.dumps(a), encoding="utf-8")
            self.assertEqual(bs.main([d]), 0)
            self.assertEqual(bs.main([str(Path(d) / "missing.json")]), 2)


if __name__ == "__main__":
    unittest.main()
