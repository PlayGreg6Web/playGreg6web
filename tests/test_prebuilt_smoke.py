import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import prebuilt_smoke as ps  # noqa: E402

HS = "## 1. Slots\n| Slot | Name | Status |\n|---|---|---|\n| XFoo 3 | DoThing | Pending |\n"


class DecideTests(unittest.TestCase):
    def test_no_report_fails(self) -> None:
        ok, line = ps.decide(None, HS, "run1")
        self.assertFalse(ok)
        self.assertIn("run1", line)

    def test_presents_no_fault_passes(self) -> None:
        ok, line = ps.decide({"present_samples_total": 120}, HS, "r")
        self.assertTrue(ok)
        self.assertIn("120 presents", line)

    def test_zero_presents_fails(self) -> None:
        ok, line = ps.decide({"present_samples_total": 0}, HS, "r")
        self.assertFalse(ok)
        self.assertIn("0 presents", line)

    def test_crash_fails_with_stop_line(self) -> None:
        rep = {"present_samples_total": 50, "fault_record": {"kind": "access_violation", "va": "0x10"}}
        ok, line = ps.decide(rep, HS, "rd")
        self.assertFalse(ok)
        self.assertTrue(line.startswith("Crashed:"))
        self.assertIn("rd", line)

    def test_wall_time_limit_passes(self) -> None:
        for rep in ({"present_samples_total": 50, "status": "wall_time_limit"},
                    {"present_samples_total": 50, "fault_record": {"kind": "wall_time_limit"}}):
            self.assertTrue(ps.decide(rep, HS, "r")[0])

    def test_stop_requested_limit_reached_passes(self) -> None:
        rep = {"present_samples_total": 2831, "status": "limit_reached", "fault_record": {"kind": "stop_requested", "va": "0x140004810"}}
        self.assertTrue(ps.decide(rep, HS, "r")[0])
        rep["present_samples_total"] = 0
        self.assertFalse(ps.decide(rep, HS, "r")[0])

    def test_other_limit_kind_still_fails(self) -> None:
        rep = {"present_samples_total": 50, "status": "limit_reached", "fault_record": {"kind": "dispatch_limit", "va": "0x10"}}
        self.assertFalse(ps.decide(rep, HS, "r")[0])

    def test_wall_time_limit_without_presents_fails(self) -> None:
        self.assertFalse(ps.decide({"present_samples_total": 0, "fault_record": {"kind": "wall_time_limit"}}, HS, "r")[0])

    def test_crash_without_presents_fails(self) -> None:
        rep = {"fault_record": {"kind": "access_violation", "va": "0x10"}}
        self.assertFalse(ps.decide(rep, HS, "rd")[0])

    def test_expect_name_mismatch_fails(self) -> None:
        rep = {"present_samples_total": 50, "run": {"name": "prebuilt-smoke-old"}}
        ok, line = ps.decide(rep, HS, "r", expect_name="prebuilt-smoke-new")
        self.assertFalse(ok)
        self.assertIn("stale", line)

    def test_expect_name_match_passes(self) -> None:
        rep = {"present_samples_total": 50, "run": {"name": "n1"}}
        self.assertTrue(ps.decide(rep, HS, "r", expect_name="n1")[0])

    def test_expect_name_missing_run_fails(self) -> None:
        self.assertFalse(ps.decide({"present_samples_total": 50}, HS, "r", expect_name="n1")[0])


class PruneTests(unittest.TestCase):
    def test_keeps_newest_three(self) -> None:
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            for i in range(5):
                d = root / f"run{i}"
                d.mkdir()
                os.utime(d, (1000 + i, 1000 + i))
            ps.prune_runs(root, 3)
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["run2", "run3", "run4"])


if __name__ == "__main__":
    unittest.main()
