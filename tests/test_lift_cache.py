"""Shard lift cache helpers in tools/lift_cache.py (trim order, half entries, temporaries)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import lift_cache  # noqa: E402

NOW = 1_000_000.0


def entry(cache: Path, n: int, used: float, size: int = 100) -> str:
    key = f"{n:032x}"
    for ext, body in (("ll", b"x" * size), ("meta", b"m")):
        path = cache / f"{key}.{ext}"
        path.write_bytes(body)
        os.utime(path, (used, used))
    return key


class TrimShardCacheTest(unittest.TestCase):
    def test_removes_least_recently_used_whole_entries_first(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d)
            old = entry(cache, 1, NOW - 9000)
            mid = entry(cache, 2, NOW - 8000)
            new = entry(cache, 3, NOW - 7000)
            removed, left = lift_cache.trim_shard_cache(cache, 250, grace_s=3600, now=NOW)
            self.assertEqual(removed, 1)
            self.assertEqual(left, 202)
            names = sorted(p.name for p in cache.iterdir())
            self.assertEqual(names, sorted(f"{k}.{e}" for k in (mid, new) for e in ("ll", "meta")))
            self.assertNotIn(old, " ".join(names))

    def test_keeps_recent_entries_and_drops_stale_temporaries(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d)
            entry(cache, 1, NOW - 10)  # in use by a build: kept even over the cap
            stale = cache / ("a" * 32 + ".tmp-1.ll.tmp")
            stale.write_bytes(b"t")
            os.utime(stale, (NOW - 7200, NOW - 7200))
            fresh = cache / ("b" * 32 + ".tmp-2.ll.tmp")
            fresh.write_bytes(b"t")
            os.utime(fresh, (NOW - 5, NOW - 5))
            removed, _ = lift_cache.trim_shard_cache(cache, 0, grace_s=3600, now=NOW)
            self.assertEqual(removed, 0)
            self.assertFalse(stale.exists())
            self.assertTrue(fresh.exists())
            self.assertEqual(len(list(cache.glob("*.ll"))), 1)

    def test_missing_dir_and_settings(self):
        self.assertEqual(lift_cache.trim_shard_cache(Path("/nonexistent/pg6-shards"), 0), (0, 0))
        self.assertEqual(lift_cache.shard_cache_max_bytes({}), int(lift_cache.SHARD_CACHE_GB * (1 << 30)))
        self.assertEqual(lift_cache.shard_cache_max_bytes({lift_cache.SHARD_CACHE_GB_ENV: "0.5"}), 1 << 29)
        self.assertEqual(lift_cache.shard_cache_args(Path("c")), ["--lift-cache", "c"])


if __name__ == "__main__":
    unittest.main()
