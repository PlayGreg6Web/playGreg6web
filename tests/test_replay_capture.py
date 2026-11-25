from pathlib import Path
import struct
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import replay_capture  # noqa: E402


def pkt(op, payload=b""):
    return struct.pack("<HHI", op, 0, len(payload)) + payload


def upload(tex, size):
    return pkt(13, struct.pack("<I", tex) + b"\0" * size)


class TestReplayCapture(unittest.TestCase):
    def test_parse_frames(self) -> None:
        self.assertEqual(replay_capture.parse_frames("0, 100,5"), [0, 100, 5])
        self.assertEqual(replay_capture.parse_frames(""), [])
        with self.assertRaises(ValueError):
            replay_capture.parse_frames("-1")

    def test_trim_keeps_every_nth_upload(self) -> None:
        data = b"".join(upload(1, 100) + upload(2, 4) + pkt(7) for _ in range(4))
        self.assertEqual(replay_capture.largest_textures(data, 1), [1])
        out, frames, dropped = replay_capture.trim_trace(data, 2, [1])
        self.assertEqual(frames, 4)
        self.assertEqual(dropped, 2 * 104)
        self.assertEqual(len(out), len(data) - 2 * (8 + 104))

    def test_parse_replay_log(self) -> None:
        text = "swapchain frame 5 (3840x2160 bgra8unorm): non-black 10/100 (10.00%), distinct 7\n"
        (s,) = replay_capture.parse_replay_log(text)
        self.assertEqual((s["frame"], s["non_black_pct"], s["distinct"]), (5, 10.0, 7))

    def test_args(self) -> None:
        args = replay_capture.build_parser().parse_args(
            ["--chrome", "c", "--replay-dir", "d", "--trace", "t", "--pack", "p", "--out", "o", "--thumbat", "1,2"])
        self.assertEqual(args.thumbat, [1, 2])


if __name__ == "__main__":
    unittest.main()
