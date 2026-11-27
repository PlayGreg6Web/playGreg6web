import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import prebuilt_stale as ps  # noqa: E402


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


class PathFilterTests(unittest.TestCase):
    def test_relevant(self) -> None:
        for p in ("runtime/os/x.cpp", "recomp/src/a.rs", "shaders/a.wgsl", "tools/run_native_boot.py", "runtime\main.cpp"):
            self.assertTrue(ps.is_relevant(p), p)

    def test_irrelevant(self) -> None:
        for p in ("docs/PLAY_LOCALLY.md", "web/index.html", "tests/a.py", "runtime_notes.md", "tools/screen_timeline.py"):
            self.assertFalse(ps.is_relevant(p), p)


class MessageTests(unittest.TestCase):
    def test_format(self) -> None:
        m = ps.format_message(5, "a" * 40, "2026-11-01", ["s1", "s2", "s3", "s4"], web=True)
        lines = m.splitlines()
        self.assertEqual(
            lines[0],
            "Prebuilt is 5 runtime commits behind (built from aaaaaaaaa, 2026-11-01). It still runs; refresh with tools/update_prebuilt.ps1 -Web.",
        )
        self.assertEqual(lines[1:], ["  s1", "  s2", "  s3"])

    def test_singular_native(self) -> None:
        self.assertIn("1 runtime commit behind", ps.format_message(1, "b" * 40, "d", ["x"]))
        self.assertTrue(ps.format_message(1, "b" * 40, "d", ["x"]).splitlines()[0].endswith("update_prebuilt.ps1."))


class CheckTests(unittest.TestCase):
    def test_check_end_to_end_and_silent_skips(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d) / "r"
            repo.mkdir()
            git(repo, "init", "-q")
            (repo / "docs").mkdir()
            (repo / "docs" / "a.md").write_text("1")
            git(repo, "add", "-A"); git(repo, "commit", "-qm", "base")
            base = git(repo, "rev-parse", "HEAD")
            pre = Path(d) / "p"
            pre.mkdir()
            (pre / "prebuilt.json").write_text(json.dumps({"commit": base}), encoding="utf-8")
            (repo / "docs" / "a.md").write_text("2")
            git(repo, "commit", "-qam", "docs only")
            self.assertIsNone(ps.check(pre, repo))
            (repo / "runtime").mkdir()
            (repo / "runtime" / "m.cpp").write_text("x")
            git(repo, "add", "-A"); git(repo, "commit", "-qm", "runtime change")
            msg = ps.check(pre, repo)
            self.assertIsNotNone(msg)
            self.assertIn("1 runtime commit behind", msg)
            self.assertIn("runtime change", msg)
            (pre / "prebuilt.json").write_text(json.dumps({"commit": "f" * 40}), encoding="utf-8")
            self.assertIsNone(ps.check(pre, repo))
            self.assertIsNone(ps.check(Path(d) / "missing", repo))
            self.assertIsNone(ps.check(pre, Path(d) / "notarepo"))


if __name__ == "__main__":
    unittest.main()
