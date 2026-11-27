import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import clean_build_dirs  # noqa: E402
import lift_cache  # noqa: E402
import prebuilt_guard as pg  # noqa: E402
import run_history as rh  # noqa: E402


class PrebuiltGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.pb = self.root / "prebuilt"
        (self.pb / "program" / "x").mkdir(parents=True)
        (self.pb / "program" / "x" / "f.bin").write_text("data")
        (self.pb / "prebuilt.json").write_text("{}")

    def tearDown(self):
        self.tmp.cleanup()

    def test_containing(self):
        self.assertEqual(pg.containing_prebuilt(self.pb / "program" / "x"), self.pb)
        self.assertEqual(pg.containing_prebuilt(self.pb), self.pb)
        self.assertIsNone(pg.containing_prebuilt(self.root / "other"))

    def test_runs_root_outside(self):
        r = pg.runs_root(self.pb, {"LOCALAPPDATA": str(self.root / "la")})
        self.assertEqual(r, self.root / "la" / "playgta6" / "runs" / "prebuilt")
        self.assertEqual(pg.runs_root(self.pb, {}), self.root / "prebuilt-runs")
        self.assertIsNone(pg.containing_prebuilt(r))
        # web layout: build dir is <prebuilt>/build
        self.assertEqual(pg.default_runs_dir(self.pb / "build", {}), self.root / "prebuilt-runs")
        self.assertIsNone(pg.default_runs_dir(self.root / "plain", {}))
        self.assertEqual(pg.lock_dir(self.root / "plain", {}), self.root / "plain")

    def test_clear_program_refuses(self):
        with self.assertRaises(pg.PrebuiltProtected):
            lift_cache.clear_program(self.pb / "program")
        self.assertTrue((self.pb / "program" / "x" / "f.bin").is_file())

    def test_clear_program_normal(self):
        prog = self.root / "plain" / "program"
        (prog / "y").mkdir(parents=True)
        lift_cache.clear_program(prog)
        self.assertFalse((prog / "y").exists())

    def test_remove_tree_and_candidate_refuse(self):
        with self.assertRaises(pg.PrebuiltProtected):
            clean_build_dirs.remove_tree(self.pb / "program")
        (self.pb / clean_build_dirs.MARKER).write_text("{}")
        ok, why = clean_build_dirs.check_candidate(self.pb, self.root / "repo", 0)
        self.assertFalse(ok)
        self.assertIn("prebuilt", why)
        self.assertTrue((self.pb / "program" / "x" / "f.bin").is_file())

    def test_run_history_finds_new_location(self):
        la = self.root / "la" / "playgta6"
        run = la / "runs" / "prebuilt" / "r1"
        run.mkdir(parents=True)
        (run / "boot-report.json").write_text("{}")
        pairs = rh.discover([], "pg6*", la)
        self.assertEqual(pairs, [(run, la / "prebuilt")])


if __name__ == "__main__":
    unittest.main()
