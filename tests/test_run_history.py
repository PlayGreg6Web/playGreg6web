import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import run_history as rh  # noqa: E402


def report(fps_step=33, draws=37, misses=3, fault=False, preset="smoke", commit=None, underruns=0):
    r = {"preset": preset, "status": "limit_reached", "elapsed_ms": 90000,
         "milestones": [{"milestone": "FIRST_PRESENT", "elapsed_ms": 3400}],
         "present_samples": [{"wall_ms": i * fps_step} for i in range(20)],
         "screen_timeline": [{"draws_per_frame": draws // 2}, {"draws_per_frame": draws}],
         "native_executor": {"device_errors": 1, "shader_pack_misses": misses},
         "audio_out": {"underruns": underruns},
         "memory_budget": {"host_private_bytes": 2 * 1048576 * 100}}
    if fault:
        r["fault_record"] = {"kind": "access_violation", "elapsed_ms": 5000}
    if commit:
        r["commit"] = commit
    return r


def make_run(build: Path, name: str, rep: dict, mtime: float):
    d = build / "runs" / name
    d.mkdir(parents=True)
    p = d / "boot-report.json"
    p.write_text(json.dumps(rep), encoding="utf-8")
    os.utime(p, (mtime, mtime))


class RunHistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.b = self.root / "pg6a"
        self.b.mkdir()
        (self.b / "prebuilt.json").write_text(json.dumps({"commit": "aaaaaaaaaaaa"}), encoding="utf-8")
        (self.root / "other").mkdir()
        make_run(self.root / "other", "ignored", report(), 1000)
        make_run(self.b, "r2", report(commit="bbbbbbbbbbbb"), 2000)
        make_run(self.b, "r1", report(), 1000)

    def tearDown(self):
        self.tmp.cleanup()

    def rows(self, **kw):
        return rh.collect(rh.discover([self.root], "pg6*", None), **kw)

    def test_rows_sorted_with_commit_and_metrics(self):
        rows = self.rows()
        self.assertEqual([r["run"] for r in rows], ["r1", "r2"])
        self.assertEqual([r["commit"] for r in rows], ["aaaaaaaaaa", "bbbbbbbbbb"])
        r = rows[0]
        self.assertEqual((r["gpu_errors"], r["pack_misses"], r["underruns"], r["private_peak_mib"]), (1, 3, 0, 200.0))
        self.assertAlmostEqual(r["fps_p50"], 30.3, places=1)
        self.assertIn("37 draws/frame", r["max_screen"])
        self.assertTrue(r["ended"].startswith("limit"))

    def test_since_filters(self):
        self.assertEqual([r["run"] for r in self.rows(since=1500)], ["r2"])
        self.assertEqual(rh.parse_since("2d", now=200000), 200000 - 172800)

    def test_fault_ending(self):
        make_run(self.b, "r3", report(fault=True), 3000)
        self.assertTrue(self.rows()[-1]["ended"].startswith("FAULT access_violation"))

    def test_group_by_commit_flags_regression_and_improvement(self):
        make_run(self.b, "r3", report(fps_step=100, commit="cccccccccc", draws=10), 3000)
        g = rh.group_by_commit(self.rows())
        self.assertEqual(g[0]["change"], "")
        self.assertEqual(g[1]["change"], "")
        self.assertTrue(g[2]["change"].startswith("REGRESSION"))
        self.assertIn("fps p50", g[2]["change"])
        self.assertEqual([x["commit"] for x in rh.regressions(g)], ["cccccccccc"])
        make_run(self.b, "r4", report(commit="dddddddddd"), 4000)
        g = rh.group_by_commit(self.rows())
        self.assertTrue(g[3]["change"].startswith("improved"))
        self.assertEqual(rh.regressions(g), [])

    def test_cli_formats_and_exit_code_and_read_only(self):
        make_run(self.b, "r3", report(fault=True, commit="cccccccccc"), 3000)
        before = sorted(str(p) for p in self.root.rglob("*"))
        out = self.root / "t.csv"
        self.assertEqual(rh.main(["--root", str(self.root), "--prebuilt-dir", "", "--format", "csv", "--out", str(out)]), 0)
        self.assertTrue(out.read_text(encoding="utf-8").startswith("time,run,commit"))
        self.assertEqual(rh.main(["--root", str(self.root), "--prebuilt-dir", "", "--fail-on-regression"]), 1)
        after = sorted(str(p) for p in self.root.rglob("*") if p != out)
        self.assertEqual(before, after)
        self.assertEqual(rh.main(["--root", str(self.root), "--prebuilt-dir", "", "--since", "1000000d"]), 0)


if __name__ == "__main__":
    unittest.main()
