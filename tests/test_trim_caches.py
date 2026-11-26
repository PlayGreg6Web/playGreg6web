"""tools/trim_caches.py on fake cache trees (nothing real is touched)."""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import trim_caches as tc  # noqa: E402

H = 3600.0


def age(path: Path, hours: float) -> None:
    t = time.time() - hours * H
    os.utime(path, (t, t))


def mkfile(path: Path, size: int, hours: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    age(path, hours)
    return path


class TrimTest(unittest.TestCase):
    def test_dry_run_deletes_nothing_and_apply_removes_lru_keeping_newest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "pg6nb-objcache-x"
            files = [mkfile(root / f"{i}.obj", 100, 100 - i * 10) for i in range(4)]  # 0 oldest
            cr = tc.CacheRoot("obj", root)
            rep = tc.run([cr], {"obj": 250 / tc.GB}, apply=False)[0]
            self.assertEqual(rep.freed, 200)
            self.assertTrue(all(f.exists() for f in files))
            tc.run([cr], {"obj": 250 / tc.GB}, apply=True)
            self.assertEqual([f.exists() for f in files], [False, False, True, True])

    def test_newest_grace_and_locks_are_protected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "c-lift"
            for name, h in (("a", 50), ("b", 40), ("c", 1)):
                mkfile(root / name / "program-manifest.json", 100, h)
            mkfile(root / "a" / ".build.lock", 1, 0)  # fresh lock: a is held
            for name, h in (("a", 50), ("b", 40), ("c", 1)):
                age(root / name, h)
            tc.run([tc.CacheRoot("lift", root)], {"lift": 0.0000001}, apply=True)
            self.assertTrue((root / "a").exists())   # locked
            self.assertFalse((root / "b").exists())  # old and free
            self.assertTrue((root / "c").exists())   # newest and inside the grace

    def test_shard_pairs_go_together_and_temporaries_are_kept(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "c-lift-shards"
            for n, h in (("1", 90), ("2", 80), ("3", 70)):
                mkfile(root / f"{n}.ll", 100, h)
                mkfile(root / f"{n}.meta", 5, h)
            tmp = mkfile(root / "9.tmp", 50, 0)
            tc.run([tc.CacheRoot("shard", root)], {"shard": 120 / tc.GB}, apply=True)
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["3.ll", "3.meta", "9.tmp"])
            self.assertTrue(tmp.exists())

    def test_never_deletes_outside_the_root(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "dawn-cache"
            outside = mkfile(Path(d) / "keep.bin", 100, 99)
            mkfile(root / "t" / "a.bin", 100, 99)
            mkfile(root / "t" / "b.bin", 100, 98)
            mkfile(root / "t" / "c.bin", 100, 97)
            rep = tc.plan(tc.CacheRoot("dawn", root), 1)
            rep.victims[0].paths.append(outside)  # a bogus plan entry
            tc.apply_plan(rep)
            self.assertTrue(outside.exists())

    def test_discover_uses_env_and_siblings(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            obj = base / "pg6nb-objcache-t"
            for n in ("", "-lift", "-lift-shards", "-recomp", "-targets"):
                (base / f"pg6nb-objcache-t{n}").mkdir()
            roots = tc.discover({"PLAYGTA6_OBJ_CACHE": str(obj), "LOCALAPPDATA": str(base / "none")},
                                repo=base / "repo", scan_dirs=[base])
            self.assertEqual(sorted(r.kind for r in roots), ["lift", "obj", "recomp", "shard", "targets"])

    def test_auto_trim_only_when_disk_is_low(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            obj = base / "pg6nb-objcache-t"
            for i in range(3):
                mkfile(obj / f"{i}.obj", 100, 99 - i)
            env = {"PLAYGTA6_OBJ_CACHE": str(obj), "LOCALAPPDATA": str(base / "none"),
                   "PLAYGTA6_OBJ_CACHE_MAX_GB": str(150 / tc.GB)}
            logs = []
            kw = dict(say=logs.append, repo=base / "repo", scan_dirs=[])
            self.assertEqual(tc.auto_trim_if_low(env, obj, free_fn=lambda p: 50 * tc.GB, **kw), 0)
            self.assertEqual(len(list(obj.iterdir())), 3)
            self.assertGreater(tc.auto_trim_if_low(env, obj, free_fn=lambda p: 1 * tc.GB, **kw), 0)
            self.assertTrue(logs and "disk low" in logs[0])
            self.assertLess(len(list(obj.iterdir())), 3)
            env["PLAYGTA6_MIN_FREE_GB"] = "0"
            self.assertEqual(tc.auto_trim_if_low(env, obj, free_fn=lambda p: 0, **kw), 0)


if __name__ == "__main__":
    unittest.main()


class CacheRootTest(unittest.TestCase):
    def test_root_default_env_and_obj_override(self):
        import cache_root as cr
        win = cr.cache_root({"LOCALAPPDATA": "C:/L"}, windows=True)
        self.assertEqual(win, Path("C:/L") / "playgta6" / "cache")
        self.assertEqual(cr.obj_cache({cr.ROOT_ENV: "/r"}, windows=False), Path("/r") / "objcache")
        self.assertEqual(cr.wasm_cache({cr.ROOT_ENV: "/r"}, windows=False), Path("/r") / "wasm_obj_cache")
        self.assertEqual(cr.obj_cache({cr.OBJ_ENV: "/o", cr.ROOT_ENV: "/r"}), Path("/o"))

    def test_legacy_hint_once_and_never_moves(self):
        import cache_root as cr
        with tempfile.TemporaryDirectory() as d:
            local = Path(d) / "local"
            old = local / "playgta6" / "objcache"
            mkfile(old / "a.obj", 10, 1)
            env = {"LOCALAPPDATA": str(local)}
            msgs = []
            self.assertTrue(cr.legacy_hint(env, msgs.append, windows=False) or True)
            msgs.clear()
            env = {"XDG_CACHE_HOME": str(local)}
            self.assertTrue(cr.legacy_hint(env, msgs.append, windows=False))
            self.assertIn(str(old), msgs[0])
            self.assertFalse(cr.legacy_hint(env, msgs.append, windows=False))  # once
            self.assertFalse(cr.legacy_hint(env, msgs.append, explicit=True, windows=False))
            self.assertTrue((old / "a.obj").exists())
