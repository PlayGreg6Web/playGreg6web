#!/usr/bin/env python3
"""Tests for tools/disk_report.py using a fake tree."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import disk_report as dr  # noqa: E402


def mk(root: Path, rel: str, size: int = 0) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * size)
    return p


class DiskReportTest(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.root = Path(self._t.name)
        mk(self.root, "pg6pack/a.wgsl.pack", 5000)
        mk(self.root, "pg6run/runs/r1/boot-report.json", 100)
        mk(self.root, "pg6build/program/x.bin", 300)
        mk(self.root, "pg6wt/.git", 10)
        mk(self.root, "pg6logs/a.log", 50)
        mk(self.root, "other/big.bin", 99999)

    def tearDown(self):
        self._t.cleanup()

    def test_size_and_pattern(self):
        c = dr.candidates([str(self.root)], "pg6*", False)
        self.assertEqual(sorted(os.path.basename(x) for x in c),
                         ["pg6build", "pg6logs", "pg6pack", "pg6run", "pg6wt"])
        self.assertEqual(dr.dir_size(str(self.root / "pg6pack"))[0], 5000)

    def test_classify(self):
        def k(n):
            return dr.classify(str(self.root / n))
        self.assertEqual(k("pg6pack"), "shader pack")
        self.assertEqual(k("pg6run"), "run dir")
        self.assertEqual(k("pg6build"), "build output")
        self.assertEqual(k("pg6wt"), "git worktree")
        self.assertEqual(k("pg6logs"), "logs")

    def test_report_sorted_refs_and_readonly(self):
        env = {"PLAYGTA6_NATIVE_SHADER_PACK": str(self.root / "pg6pack" / "a.wgsl.pack")}
        wts = [str(self.root / "pg6wt")]
        c = dr.candidates([str(self.root)], "pg6*", False)
        rows = dr.build_report(c, wts, dr.env_references(env), workers=2)
        self.assertEqual(rows[0]["size"], max(r["size"] for r in rows))
        by = {os.path.basename(r["path"]): r for r in rows}
        self.assertTrue(by["pg6pack"]["safe"].startswith("NO"))
        self.assertEqual(by["pg6wt"]["safe"], "NO (worktree)")
        self.assertIn("git worktree remove", by["pg6wt"]["cmd"])
        self.assertEqual(by["pg6build"]["safe"], "yes")
        self.assertIn("Remove-Item", by["pg6build"]["cmd"])
        self.assertTrue((self.root / "pg6pack" / "a.wgsl.pack").exists())
        self.assertIn("nothing was deleted", dr.render(rows, 10))

    def test_objcache_stdpack_registry(self):
        h = "a" * 64
        mk(self.root, f"pg6nb-objcache/{h}.obj", 10)
        mk(self.root, f"pg6nb-objcache-lift/{h}.o", 10)
        mk(self.root, "pg6nb-std/sub/x.wgsl.pack", 100)
        mk(self.root, "pg6nb-std2/y.wgsl.pack", 100)
        mk(self.root, "pg6nb-pk7/runs/r1/boot-report.json", 5)
        mk(self.root, "pg6nb-pk7/z.wgsl.pack", 50)
        mk(self.root, "pg6ab/q.wgsl.pack", 50)
        os.utime(self.root / "pg6nb-std" / "sub" / "x.wgsl.pack", (1, 1))
        for n in ("pg6nb-objcache", "pg6nb-objcache-lift"):
            self.assertEqual(dr.classify(str(self.root / n)), "object cache")
        self.assertEqual(dr.classify(str(self.root / "pg6nb-pk7")), "shader pack")
        reg = {"PLAYGTA6_SHADER_PACK": str(self.root / "pg6ab" / "q.wgsl.pack")}
        env = dr.merged_env({}, reg)
        self.assertIn("PLAYGTA6_SHADER_PACK", env)
        self.assertEqual(dr.merged_env({"A": "1", "B": "p"}, {"B": "r"})["B"], "p")
        c = dr.candidates([str(self.root)], "pg6*", False)
        rows = dr.build_report(c, [], dr.env_references(env), workers=2)
        by = {os.path.basename(r["path"]): r for r in rows}
        self.assertEqual(by["pg6nb-objcache"]["safe"], "NO (in use, keep)")
        self.assertEqual(by["pg6nb-objcache-lift"]["safe"], "NO (in use, keep)")
        self.assertEqual(by["pg6ab"]["safe"], "NO (in use)")
        self.assertEqual(by["pg6nb-std2"]["safe"], "keep (standard pack)")
        self.assertEqual(by["pg6nb-std"]["kind"], "shader pack")
        self.assertTrue(by["pg6nb-std"]["safe"].startswith("maybe"))
        self.assertEqual(by["pg6nb-pk7"]["kind"], "shader pack")

    def test_symlink_skipped(self):
        link = self.root / "pg6pack" / "lnk"
        try:
            os.symlink(self.root / "other", link, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        self.assertEqual(dr.dir_size(str(self.root / "pg6pack"))[0], 5000)


if __name__ == "__main__":
    unittest.main()
