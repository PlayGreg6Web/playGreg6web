import contextlib
import io
import json
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import screen_timeline as st  # noqa: E402


def present(t, draws, res):
    return {"wall_ms": t, "counters": {"draws_issued": draws, "resources_created": res}}


def _png_chunk(ctype: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(ctype + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", crc)


def _write_solid_png(path: Path, rgb: tuple[int, int, int], width: int = 16, height: int = 12) -> None:
    """Write a minimal valid 8-bit RGB PNG without external dependencies."""
    r, g, b = rgb
    row = b"\x00" + bytes([r, g, b]) * width
    raw = row * height
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(png)


class ScreenTimeline(unittest.TestCase):
    def test_collapse_and_split(self):
        ps = [present(0, 10, 5), present(10, 20, 5), present(20, 31, 5),  # 10, 10, 11 per frame
              present(30, 41, 5), present(40, 91, 5), present(50, 141, 9)]
        segs = st.build_timeline(ps, [5, 35, 36, 45])
        self.assertEqual([s["first_present"] for s in segs], [0, 4, 5])
        self.assertEqual(segs[0]["presents"], 4)
        self.assertEqual(segs[0]["draws_per_frame"], 10)
        self.assertEqual(segs[0]["end_time"], 30)
        self.assertEqual(segs[2]["new_resources_at_start"], 4)
        self.assertEqual([s["input_events_in_segment"] for s in segs], [3, 1, 0])

    def test_present_samples_fallback(self):
        samples = [{"index": i, "wall_ms": 10 * i, "draws": d, "resources": r}
                   for i, (d, r) in enumerate([(10, 5), (10, 0), (11, 0), (50, 0), (50, 4)])]
        segs = st.build_timeline(st.presents_from_samples(samples))
        self.assertEqual([s["first_present"] for s in segs], [0, 3, 4])
        self.assertEqual(segs[0]["presents"], 3)
        self.assertEqual(segs[0]["new_resources_at_start"], 5)
        self.assertEqual(segs[2]["new_resources_at_start"], 4)
        self.assertEqual(st.presents_from_samples([]), [])

    def test_cap(self):
        ps = [present(i, i * (i + 1) * 5, 0) for i in range(1, 50)]
        segs = st.build_timeline(ps, max_segments=5)
        self.assertEqual(len(segs), 5)
        self.assertEqual(sum(s["presents"] for s in segs), 49)

    def test_empty_and_parse(self):
        self.assertEqual(st.build_timeline([]), [])
        self.assertEqual(st.parse_input_times("# c\n100 A 50\n250 START\n"), [100.0, 250.0])
        table = st.format_table(st.build_timeline([present(0, 1, 0)]))
        self.assertEqual(table[0], "screen timeline: 1 segment(s)")
        self.assertIn("draws/frame", table[1])


class ScreenTimelineCaptureCLI(unittest.TestCase):
    def _run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = st.main(argv)
        return rc, out.getvalue(), err.getvalue()

    def test_transitions_interval_sec_and_json_output(self):
        with tempfile.TemporaryDirectory(prefix="playgta6_secret_host_dir_") as tmp:
            frames_dir = Path(tmp)
            # Leading black (frames 1..2), Splash red (frames 3..4), Black loading (frame 5),
            # Start Screen blue (frames 6..8), Main Menu green (frames 9..10).
            colors = [
                (0, 0, 0),
                (0, 0, 0),
                (200, 40, 40),
                (200, 40, 40),
                (0, 0, 0),
                (30, 80, 220),
                (30, 80, 220),
                (30, 80, 220),
                (40, 200, 80),
                (40, 200, 80),
            ]
            for idx, rgb in enumerate(colors, start=1):
                _write_solid_png(frames_dir / f"frame_{idx:04d}.png", rgb)

            rc, stdout, stderr = self._run_cli([str(frames_dir), "--interval-sec", "1.5", "--json"])
            self.assertEqual(rc, 0, stderr)
            self.assertEqual(stderr, "")
            # Must never leak host filesystem paths.
            self.assertNotIn(str(frames_dir), stdout)
            self.assertNotIn(frames_dir.name, stdout)

            data = json.loads(stdout)
            self.assertEqual(
                set(data.keys()),
                {
                    "total_frames",
                    "unique_segments",
                    "black_frames",
                    "non_black_frames",
                    "first_non_black_frame",
                    "first_non_black_s",
                    "segments",
                },
            )
            self.assertEqual(data["total_frames"], 10)
            self.assertEqual(data["unique_segments"], 5)
            self.assertEqual(data["black_frames"], 3)
            self.assertEqual(data["non_black_frames"], 7)
            self.assertEqual(data["first_non_black_frame"], 3)
            self.assertEqual(data["first_non_black_s"], 3.0)

            segs = data["segments"]
            self.assertEqual(len(segs), 5)
            expected_seg_keys = {
                "index",
                "start_frame",
                "end_frame",
                "frame_count",
                "start_s",
                "end_s",
                "dwell_s",
                "mean_luma",
                "is_black",
            }
            for seg in segs:
                self.assertEqual(set(seg.keys()), expected_seg_keys)

            self.assertEqual([s["index"] for s in segs], [0, 1, 2, 3, 4])
            self.assertEqual([(s["start_frame"], s["end_frame"], s["frame_count"]) for s in segs], [
                (1, 2, 2),
                (3, 4, 2),
                (5, 5, 1),
                (6, 8, 3),
                (9, 10, 2),
            ])
            self.assertEqual([(s["start_s"], s["end_s"], s["dwell_s"]) for s in segs], [
                (0.0, 3.0, 3.0),
                (3.0, 6.0, 3.0),
                (6.0, 7.5, 1.5),
                (7.5, 12.0, 4.5),
                (12.0, 15.0, 3.0),
            ])
            self.assertEqual([s["is_black"] for s in segs], [True, False, True, False, False])
            self.assertEqual(segs[0]["mean_luma"], 0.0)
            self.assertGreater(segs[1]["mean_luma"], 50.0)
            self.assertEqual(segs[2]["mean_luma"], 0.0)
            self.assertGreater(segs[3]["mean_luma"], 50.0)
            self.assertGreater(segs[4]["mean_luma"], 100.0)

    def test_json_without_interval_sec_sets_times_to_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = Path(tmp)
            _write_solid_png(frames_dir / "frame_0001.png", (0, 0, 0))
            _write_solid_png(frames_dir / "frame_0002.png", (180, 180, 180))
            _write_solid_png(frames_dir / "frame_0003.png", (180, 180, 180))

            rc, stdout, stderr = self._run_cli([str(frames_dir), "--json"])
            self.assertEqual(rc, 0, stderr)
            data = json.loads(stdout)
            self.assertEqual(data["total_frames"], 3)
            self.assertEqual(data["unique_segments"], 2)
            self.assertEqual(data["black_frames"], 1)
            self.assertEqual(data["non_black_frames"], 2)
            self.assertEqual(data["first_non_black_frame"], 2)
            self.assertIsNone(data["first_non_black_s"])
            for seg in data["segments"]:
                self.assertIsNone(seg["start_s"])
                self.assertIsNone(seg["end_s"])
                self.assertIsNone(seg["dwell_s"])

    def test_plain_text_output_with_and_without_interval(self):
        with tempfile.TemporaryDirectory(prefix="playgta6_plain_dir_") as tmp:
            frames_dir = Path(tmp)
            _write_solid_png(frames_dir / "frame_0001.png", (0, 0, 0))
            _write_solid_png(frames_dir / "frame_0002.png", (120, 120, 120))
            _write_solid_png(frames_dir / "frame_0003.png", (120, 120, 120))

            rc1, out1, err1 = self._run_cli([str(frames_dir)])
            self.assertEqual(rc1, 0, err1)
            self.assertIn("screen timeline: 3 frames, 2 segments", out1)
            self.assertIn("first_non_black: frame 2", out1)
            self.assertIn("black", out1)
            self.assertIn("visible", out1)
            self.assertNotIn("dwell_s", out1)
            self.assertNotIn(str(frames_dir), out1)
            self.assertNotIn(frames_dir.name, out1)

            rc2, out2, err2 = self._run_cli([str(frames_dir), "--interval-sec", "2.0"])
            self.assertEqual(rc2, 0, err2)
            self.assertIn("first_non_black: frame 2 @ 2.00s", out2)
            self.assertIn("dwell_s", out2)
            self.assertIn("4.00", out2)
            self.assertNotIn(str(frames_dir), out2)

    def test_empty_directory_and_non_image_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = Path(tmp)
            # Also confirm non-image files are ignored cleanly.
            (frames_dir / "notes.txt").write_text("ignore me", encoding="utf-8")

            rc_json, out_json, err_json = self._run_cli([str(frames_dir), "--interval-sec", "1.0", "--json"])
            self.assertEqual(rc_json, 0, err_json)
            data = json.loads(out_json)
            self.assertEqual(data, {
                "total_frames": 0,
                "unique_segments": 0,
                "black_frames": 0,
                "non_black_frames": 0,
                "first_non_black_frame": None,
                "first_non_black_s": None,
                "segments": [],
            })

            rc_txt, out_txt, err_txt = self._run_cli([str(frames_dir)])
            self.assertEqual(rc_txt, 0, err_txt)
            self.assertEqual(out_txt.strip(), "screen timeline: 0 frames, 0 segments (black: 0, non-black: 0)")

    def test_all_black_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = Path(tmp)
            for idx in (1, 2, 3):
                _write_solid_png(frames_dir / f"frame_{idx:04d}.png", (0, 0, 0))
            rc, stdout, stderr = self._run_cli([str(frames_dir), "--interval-sec", "0.5", "--json"])
            self.assertEqual(rc, 0, stderr)
            data = json.loads(stdout)
            self.assertEqual(data["total_frames"], 3)
            self.assertEqual(data["unique_segments"], 1)
            self.assertEqual(data["black_frames"], 3)
            self.assertEqual(data["non_black_frames"], 0)
            self.assertIsNone(data["first_non_black_frame"])
            self.assertIsNone(data["first_non_black_s"])
            self.assertEqual(data["segments"][0]["dwell_s"], 1.5)
            self.assertTrue(data["segments"][0]["is_black"])

    def test_stdlib_png_fallback_matches_pil_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = Path(tmp)
            _write_solid_png(frames_dir / "frame_0001.png", (0, 0, 0))
            _write_solid_png(frames_dir / "frame_0002.png", (255, 0, 0))
            _write_solid_png(frames_dir / "frame_0003.png", (0, 255, 0))

            fallback = st.analyze_frames_dir(frames_dir, interval_sec=1.0, _auto_import_pil=False)
            with_pil = st.analyze_frames_dir(frames_dir, interval_sec=1.0, _auto_import_pil=True)
            self.assertEqual(fallback["total_frames"], with_pil["total_frames"])
            self.assertEqual(fallback["unique_segments"], 3)
            self.assertEqual(with_pil["unique_segments"], 3)
            self.assertEqual(fallback["first_non_black_frame"], 2)
            self.assertEqual(fallback["first_non_black_s"], 1.0)
            for s_fb, s_pil in zip(fallback["segments"], with_pil["segments"]):
                self.assertEqual(s_fb["is_black"], s_pil["is_black"])
                self.assertAlmostEqual(s_fb["mean_luma"], s_pil["mean_luma"], delta=1.0)

    def test_invalid_inputs_return_sanitized_error(self):
        with tempfile.TemporaryDirectory(prefix="playgta6_missing_parent_") as tmp:
            missing = Path(tmp) / "does_not_exist"
            rc, stdout, stderr = self._run_cli([str(missing)])
            self.assertEqual(rc, 2)
            self.assertEqual(stdout, "")
            self.assertNotIn(str(missing), stderr)
            self.assertNotIn("playgta6_missing_parent_", stderr)

            rc_bad, _, stderr_bad = self._run_cli([tmp, "--interval-sec", "0"])
            self.assertEqual(rc_bad, 2)
            self.assertIn("--interval-sec", stderr_bad)

            rc_neg, _, _ = self._run_cli([tmp, "--interval-sec", "-1.5"])
            self.assertEqual(rc_neg, 2)

    def test_script_entrypoint_subprocess(self):
        script = Path(__file__).resolve().parent.parent / "tools" / "screen_timeline.py"
        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = Path(tmp)
            _write_solid_png(frames_dir / "frame_0001.png", (250, 250, 250))
            _write_solid_png(frames_dir / "frame_0002.png", (0, 0, 0))
            proc = subprocess.run(
                [sys.executable, str(script), str(frames_dir), "--interval-sec", "0.25", "--json"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            data = json.loads(proc.stdout)
            self.assertEqual(data["total_frames"], 2)
            self.assertEqual(data["unique_segments"], 2)
            self.assertEqual(data["first_non_black_frame"], 1)
            self.assertEqual(data["first_non_black_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
