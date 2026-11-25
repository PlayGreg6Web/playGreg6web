import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import clean_build_dirs as cbd  # noqa: E402


def age(path: Path, minutes: float) -> None:
    t = time.time() - minutes * 60
    for dp, dns, fns in os.walk(path):
        for n in fns + dns:
            os.utime(os.path.join(dp, n), (t, t))
    os.utime(path, (t, t))


class CleanBuildDirs(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        self.git("init", "-q")
        (self.repo / "tracked.txt").write_text("x")
        self.git("add", ".")
        self.git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "i")
        self.roots = self.tmp / "roots"
        self.roots.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def git(self, *a):
        subprocess.run(["git", "-C", str(self.repo), *a], check=True, capture_output=True)

    def mk(self, name, marker=True, minutes=120):
        d = self.roots / name
        (d / "sub").mkdir(parents=True)
        (d / "sub" / "f.bin").write_bytes(b"0" * 100)
        if marker:
            cbd.write_marker(d, "test", self.repo)
        age(d, minutes)
        return d

    def run_main(self, *args):
        return cbd.main(list(args), repo=self.repo)

    def test_marker_written_with_json(self):
        d = self.roots / "m"
        cbd.write_marker(d, "tool-x", self.repo)
        data = json.loads((d / cbd.MARKER).read_text())
        self.assertEqual(data["tool"], "tool-x")
        self.assertIn("created", data)
        self.assertIn("repo", data)

    def test_marker_not_written_for_repo(self):
        cbd.write_marker(self.repo, "t", self.repo)
        self.assertFalse((self.repo / cbd.MARKER).exists())

    def test_dry_run_default_keeps(self):
        d = self.mk("a")
        self.run_main("--scan", str(self.roots))
        self.assertTrue(d.exists())

    def test_yes_deletes_marked_only(self):
        a = self.mk("a")
        b = self.mk("b", marker=False)
        self.run_main("--scan", str(self.roots), "--yes")
        self.assertFalse(a.exists())
        self.assertTrue(b.exists())

    def test_dry_run_flag_overrides_yes(self):
        a = self.mk("a")
        self.run_main("--scan", str(self.roots), "--yes", "--dry-run")
        self.assertTrue(a.exists())

    def test_min_age(self):
        young = self.mk("young", minutes=5)
        old = self.mk("old", minutes=90)
        self.run_main("--scan", str(self.roots), "--yes")
        self.assertTrue(young.exists())
        self.assertFalse(old.exists())
        self.run_main(str(young), "--yes", "--min-age-min", "0")
        self.assertFalse(young.exists())

    def test_scan_not_recursive(self):
        outer = self.roots / "outer"
        outer.mkdir()
        inner = outer / "inner"
        inner.mkdir()
        cbd.write_marker(inner, "t", self.repo)
        age(inner, 120)
        self.run_main("--scan", str(self.roots), "--yes")
        self.assertTrue(inner.exists())

    def test_symlink_escape(self):
        outside = self.tmp / "outside"
        outside.mkdir()
        (outside / "keep.txt").write_text("keep")
        d = self.mk("link")
        try:
            os.symlink(outside, d / "sub" / "esc", target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        age(d, 120)
        self.run_main("--scan", str(self.roots), "--yes", "--min-age-min", "0")
        self.assertFalse(d.exists())
        self.assertTrue((outside / "keep.txt").exists())

    def test_symlinked_root_refused(self):
        real = self.mk("real")
        link = self.roots / "lnk"
        try:
            os.symlink(real, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        ok, _ = cbd.check_candidate(link, self.repo, 0)
        self.assertFalse(ok)

    def test_repo_protected(self):
        (self.repo / cbd.MARKER).write_text("{}")
        ok, _ = cbd.check_candidate(self.repo, self.repo, 0)
        self.assertFalse(ok)
        ok, _ = cbd.check_candidate(self.tmp, self.repo, 0)  # ancestor of repo
        self.assertFalse(ok)
        tracked = self.repo / "tdir"
        tracked.mkdir()
        (tracked / "t.txt").write_text("t")
        (tracked / cbd.MARKER).write_text("{}")
        self.git("add", "-f", "tdir/t.txt")
        age(tracked, 120)
        ok, why = cbd.check_candidate(tracked, self.repo, 0)
        self.assertFalse(ok, why)
        # untracked marked dir inside the repo (e.g. work/) is allowed
        work = self.repo / "work"
        work.mkdir()
        cbd.write_marker(work, "t", self.repo)
        age(work, 120)
        ok, why = cbd.check_candidate(work, self.repo, 0)
        self.assertTrue(ok, why)


if __name__ == "__main__":
    unittest.main()
