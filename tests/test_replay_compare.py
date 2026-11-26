import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import replay_compare  # noqa: E402


def png(w, h, px):
    return replay_compare.encode_png_rgb(w, h, bytes(px))


class ReplayCompareTests(unittest.TestCase):
    def test_parse_digests(self):
        text = "digest frame 3: 00112233aabbccdd\nswapchain frame 3 x\ndigest frame 10: ffffffff00000000"
        self.assertEqual(replay_compare.parse_digests(text),
                         {3: "00112233aabbccdd", 10: "ffffffff00000000"})

    def test_frame_list(self):
        self.assertEqual(replay_compare.frame_list(4, 10), [0, 4, 8, 10])
        self.assertEqual(replay_compare.frame_list(1, 2), [0, 1, 2])
        with self.assertRaises(ValueError):
            replay_compare.frame_list(0, 5)

    def test_diff_digests_first_and_missing(self):
        a = {0: "x", 1: "y", 2: "z"}
        b = {0: "x", 1: "q", 3: "w"}
        self.assertEqual(replay_compare.diff_digests(a, b), [1, 2, 3])
        self.assertEqual(replay_compare.diff_digests(a, dict(a)), [])

    def test_diff_images(self):
        a = png(2, 1, [10, 10, 10, 200, 0, 0])
        b = png(2, 1, [10, 10, 10, 200, 9, 0])
        n, total, mx, out = replay_compare.diff_images(a, b)
        self.assertEqual((n, total, mx), (1, 2, 9))
        self.assertTrue(out.startswith(b"\x89PNG"))
        self.assertEqual(replay_compare.diff_images(a, png(1, 1, [0, 0, 0]))[0], None)

    def test_compare_writes_diff_for_first_bad_frame(self):
        with tempfile.TemporaryDirectory() as d:
            for side, g in (("A", 0), ("B", 50)):
                Path(f"{d}/o-{side}-swap5.png").write_bytes(png(1, 1, [g, 0, 0]))
            rep = replay_compare.compare({4: "a", 5: "a"}, {4: "a", 5: "b"}, f"{d}/o-A", f"{d}/o-B", f"{d}/o")
            self.assertEqual(rep["first_differing_frame"], 5)
            self.assertEqual(rep["thumb_max_delta"], 50)
            self.assertTrue(Path(rep["diff_image"]).exists())
            same = replay_compare.compare({1: "a"}, {1: "a"}, "x", "y", "z")
            self.assertTrue(same["identical"])


if __name__ == "__main__":
    unittest.main()
