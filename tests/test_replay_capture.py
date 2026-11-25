import base64
import contextlib
import io
import json
from pathlib import Path
import re
import struct
import sys
import tempfile
import unittest
from unittest import mock
import zlib

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import replay_capture  # noqa: E402


def pkt(op, payload=b""):
    return struct.pack("<HHI", op, 0, len(payload)) + payload


def upload(tex, size):
    return pkt(13, struct.pack("<I", tex) + b"\0" * size)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I4s", len(data), chunk_type) + data + struct.pack(">I", crc)


def make_png(width: int, height: int, rows: list[bytes], *, color_type: int = 2,
             filter_types: list[int] | None = None, palette: bytes | None = None) -> bytes:
    """Build a valid 8-bit PNG with explicit color type and per-row PNG filter types."""
    bpp = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    stride = width * bpp
    assert len(rows) == height and all(len(r) == stride for r in rows)
    ftypes = filter_types if filter_types is not None else [0] * height
    raw = bytearray()
    prev = bytes(stride)
    for y, row in enumerate(rows):
        ft = ftypes[y]
        raw.append(ft)
        if ft == 0:
            raw += row
        elif ft == 1:
            for i in range(stride):
                a = row[i - bpp] if i >= bpp else 0
                raw.append((row[i] - a) & 0xFF)
        elif ft == 2:
            for i in range(stride):
                raw.append((row[i] - prev[i]) & 0xFF)
        elif ft == 3:
            for i in range(stride):
                a = row[i - bpp] if i >= bpp else 0
                raw.append((row[i] - ((a + prev[i]) >> 1)) & 0xFF)
        elif ft == 4:
            for i in range(stride):
                a = row[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                raw.append((row[i] - replay_capture._paeth_predictor(a, b, c)) & 0xFF)
        else:
            raise ValueError(ft)
        prev = row
    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    chunks = [b"\x89PNG\r\n\x1a\n", _png_chunk(b"IHDR", ihdr)]
    if palette is not None:
        chunks.append(_png_chunk(b"PLTE", palette))
    chunks.append(_png_chunk(b"IDAT", zlib.compress(bytes(raw))))
    chunks.append(_png_chunk(b"IEND", b""))
    return b"".join(chunks)


def make_solid_rgb_png(width: int, height: int, r: int, g: int, b: int) -> bytes:
    row = bytes((r, g, b)) * width
    return make_png(width, height, [row] * height, color_type=2)


def to_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


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
        # Frame 1's upload is dropped; frame 3's is the last of its region (nothing rewrites it): kept.
        self.assertEqual(dropped, 104)
        self.assertEqual(len(out), len(data) - (8 + 104))

    def test_trim_keeps_sub_rect_writes_nothing_rewrites(self) -> None:
        # #691: a glyph atlas gets a whole write, then one band per new glyph. Dropping a band
        # loses its glyphs for the rest of the replay; only a region rewritten later may go.
        def write(tex, y, h, mip=0):
            return pkt(13, struct.pack("<12I", tex, mip, 0, y, 0, 64, h, 1, 256, h, 4, 0) + b"abcd")
        frames = [write(9, 0, 256), write(9, 64, 64), write(9, 128, 64), write(9, 64, 64, mip=1),
                  write(9, 0, 256), write(9, 192, 64)]
        data = b"".join(f + pkt(7) for f in frames)
        step = len(frames[0]) + 8
        self.assertEqual(replay_capture.superseded_writes(data, [9]), {0, step, 2 * step})
        out, n, dropped = replay_capture.trim_trace(data, 100, [9])
        self.assertEqual(n, 6)
        # Frames 1 and 2 (bands the whole write of frame 4 covers) go; frame 0 is a kept frame; the
        # mip-1 band, the whole write and the last band are never rewritten: kept.
        self.assertEqual(dropped, 2 * (48 + 4))
        self.assertEqual(out, frames[0] + pkt(7) + pkt(7) + pkt(7) + b"".join(f + pkt(7) for f in frames[3:]))

    def test_trim_keeps_writes_rewritten_only_much_later(self) -> None:
        # #691: an atlas band written at frame 1 and rewritten (with more glyphs) only at frame 30
        # must survive: dropping it leaves the replay without its glyphs for 29 frames.
        def write(y, h):
            return pkt(13, struct.pack("<12I", 9, 0, 0, y, 0, 64, h, 1, 256, h, 4, 0) + b"abcd")
        data = write(0, 256) + pkt(7) + write(0, 64) + pkt(7) * 29 + write(0, 64) + pkt(7) + write(0, 64) + pkt(7)
        late = len(write(0, 256)) + 8 + len(write(0, 64)) + 8 * 29
        self.assertEqual(replay_capture.superseded_writes(data, [9]), {late})
        self.assertEqual(replay_capture.superseded_writes(data, [9], horizon=40), {len(write(0, 256)) + 8, late})
        out, n, dropped = replay_capture.trim_trace(data, 100, [9])
        self.assertEqual((n, dropped), (32, 48 + 4))

    def test_thin_marker_is_skipped(self) -> None:
        tex_marker = pkt(13, struct.pack("<I", 1) + b"\0" * 36 + struct.pack("<II", 0, 0x6E696874))
        buf_marker = pkt(4, struct.pack("<IIII", 7, 0, 0, 0x6E696874))
        real_buf = pkt(4, struct.pack("<IIII", 7, 0, 2, 0) + b"ab")
        data = upload(1, 100) + tex_marker + buf_marker + real_buf + pkt(7)
        self.assertEqual(replay_capture.strip_thin_markers(data), upload(1, 100) + real_buf + pkt(7))
        self.assertEqual(replay_capture.largest_textures(data, 1), [1])
        out, frames, dropped = replay_capture.trim_trace(tex_marker + pkt(7), 2, [1])
        self.assertEqual((frames, dropped), (1, 0))
        self.assertEqual(out, tex_marker + pkt(7))

    def test_parse_replay_log(self) -> None:
        text = "swapchain frame 5 (3840x2160 bgra8unorm): non-black 10/100 (10.00%), distinct 7\n"
        (s,) = replay_capture.parse_replay_log(text)
        self.assertEqual((s["frame"], s["non_black_pct"], s["distinct"]), (5, 10.0, 7))

    def test_args_preserves_existing_and_adds_new_flags(self) -> None:
        args = replay_capture.build_parser().parse_args(
            ["--chrome", "c", "--replay-dir", "d", "--trace", "t", "--pack", "p", "--out", "o", "--thumbat", "1,2"])
        self.assertEqual(args.thumbat, [1, 2])
        self.assertFalse(args.verify_non_black)
        self.assertFalse(args.json)
        self.assertEqual(args.luma_threshold, 0.0)

        args2 = replay_capture.build_parser().parse_args(
            ["--chrome", "c", "--replay-dir", "d", "--trace", "t", "--pack", "p", "--out", "o",
             "--verify-non-black", "--luma-threshold", "2.5", "--json"])
        self.assertTrue(args2.verify_non_black)
        self.assertTrue(args2.json)
        self.assertEqual(args2.luma_threshold, 2.5)


class TestNonBlackLuminanceVerification(unittest.TestCase):
    def test_pure_python_and_pillow_match_across_filters_and_color_types(self) -> None:
        # 4x5 RGB image exercising all 5 PNG filter types (0..4)
        rgb_rows = [
            bytes([0, 0, 0, 10, 20, 30, 50, 100, 150, 255, 255, 255]),
            bytes([5, 15, 25, 20, 40, 60, 80, 120, 160, 200, 180, 140]),
            bytes([12, 34, 56, 78, 90, 112, 134, 156, 178, 200, 222, 244]),
            bytes([250, 10, 5, 128, 64, 32, 16, 8, 4, 0, 127, 255]),
            bytes([3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]),
        ]
        png_rgb = make_png(4, 5, rgb_rows, color_type=2, filter_types=[0, 1, 2, 3, 4])
        # RGBA, Grayscale, Grayscale+Alpha, Palette
        png_rgba = make_png(2, 2, [
            bytes([0, 0, 0, 255, 0, 127, 255, 255]),
            bytes([255, 0, 0, 128, 0, 255, 0, 0]),
        ], color_type=6, filter_types=[1, 4])
        png_gray = make_png(2, 2, [bytes([0, 64]), bytes([128, 255])], color_type=0, filter_types=[2, 3])
        png_ga = make_png(2, 1, [bytes([0, 255, 200, 255])], color_type=4, filter_types=[1])
        png_pal = make_png(2, 1, [bytes([0, 1])], color_type=3, palette=bytes([0, 0, 0, 10, 200, 50]))

        for sample in (png_rgb, png_rgba, png_gray, png_ga, png_pal):
            py_stats = replay_capture.analyze_png_luminance(sample, use_pillow=False)
            self.assertEqual(py_stats["decoder"], "pure_python")
            pil_stats = replay_capture.analyze_png_luminance(sample, use_pillow=True)
            for key in ("width", "height", "total_pixels", "non_black_pixels",
                        "non_black_pct", "max_luma", "mean_luma", "is_non_black"):
                self.assertEqual(py_stats[key], pil_stats[key], f"mismatch on {key}")

    def test_verify_non_black_detects_all_black_regression(self) -> None:
        black_a = make_solid_rgb_png(8, 6, 0, 0, 0)
        black_b = make_solid_rgb_png(8, 6, 0, 0, 0)
        for use_pillow in (True, False):
            report = replay_capture.verify_non_black_captures(
                {"swap0": black_a, "swap10": to_data_url(black_b)},
                use_pillow=use_pillow,
            )
            self.assertEqual(report["total_frames"], 2)
            self.assertEqual(report["non_black_frames"], 0)
            self.assertEqual(report["black_frames"], 2)
            self.assertEqual(report["max_luma"], 0.0)
            self.assertTrue(report["all_black_regression"])

    def test_verify_non_black_passes_when_any_frame_is_lit(self) -> None:
        black = make_solid_rgb_png(4, 4, 0, 0, 0)
        cyan = make_solid_rgb_png(4, 4, 0, 127, 255)
        expected_luma = round(0.2126 * 0 + 0.7152 * 127 + 0.0722 * 255, 3)
        report = replay_capture.verify_non_black_captures(
            {"swap0": black, "swap2": cyan}, use_pillow=False
        )
        self.assertEqual(report["total_frames"], 2)
        self.assertEqual(report["non_black_frames"], 1)
        self.assertEqual(report["black_frames"], 1)
        self.assertAlmostEqual(report["max_luma"], expected_luma, places=3)
        self.assertFalse(report["all_black_regression"])

    def test_verify_non_black_threshold_and_empty_captures(self) -> None:
        dim = make_solid_rgb_png(4, 4, 2, 2, 2)  # luma = 2.0
        report_default = replay_capture.verify_non_black_captures([dim], luma_threshold=0.0, use_pillow=False)
        self.assertFalse(report_default["all_black_regression"])
        self.assertEqual(report_default["non_black_frames"], 1)

        report_strict = replay_capture.verify_non_black_captures([dim], luma_threshold=5.0, use_pillow=False)
        self.assertTrue(report_strict["all_black_regression"])
        self.assertEqual(report_strict["non_black_frames"], 0)
        self.assertEqual(report_strict["black_frames"], 1)
        self.assertAlmostEqual(report_strict["max_luma"], 2.0, places=3)

        report_empty = replay_capture.verify_non_black_captures({})
        self.assertTrue(report_empty["all_black_regression"])
        self.assertEqual(report_empty["non_black_frames"], 0)
        self.assertEqual(report_empty["black_frames"], 0)
        self.assertEqual(report_empty["max_luma"], 0.0)


class TestPacingMetrics(unittest.TestCase):
    def test_per_frame_30fps_pacing_summary(self) -> None:
        # 3 frames within 33.333ms budget, 2 frames over budget (avg 32.867ms -> 30.426 FPS)
        times_ms = [16.0, 25.0, 33.333, 40.0, 50.0]
        pacing = replay_capture.summarize_pacing(times_ms)
        expected_fps = round(1000.0 / (sum(times_ms) / len(times_ms)), 3)
        self.assertEqual(pacing["timing_source"], "per_frame")
        self.assertEqual(pacing["frame_count"], 5)
        self.assertAlmostEqual(pacing["present_fps_avg"], expected_fps, places=3)
        self.assertEqual(pacing["frames_within_30fps_budget"], 3)
        self.assertEqual(pacing["frames_over_30fps_budget"], 2)
        self.assertEqual(pacing["within_30fps_budget_pct"], 60.0)
        self.assertTrue(pacing["meets_30fps_target"])

        # Below 30 FPS average (avg 40ms -> 25.0 FPS)
        slow_pacing = replay_capture.summarize_pacing([20.0, 30.0, 40.0, 50.0, 60.0])
        self.assertEqual(slow_pacing["present_fps_avg"], 25.0)
        self.assertEqual(slow_pacing["frames_within_30fps_budget"], 2)
        self.assertEqual(slow_pacing["frames_over_30fps_budget"], 3)
        self.assertFalse(slow_pacing["meets_30fps_target"])

    def test_per_frame_all_within_30fps_budget(self) -> None:
        pacing = replay_capture.summarize_pacing([16.667, 20.0, 30.0])
        self.assertEqual(pacing["frames_within_30fps_budget"], 3)
        self.assertEqual(pacing["frames_over_30fps_budget"], 0)
        self.assertEqual(pacing["within_30fps_budget_pct"], 100.0)
        self.assertGreater(pacing["present_fps_avg"], 30.0)
        self.assertTrue(pacing["meets_30fps_target"])

    def test_wall_clock_fallback_pacing(self) -> None:
        # 60 frames in 2.0s => 30.0 FPS average, all within 30 FPS budget
        fast = replay_capture.summarize_pacing([], replayed_frames=60, elapsed_seconds=2.0)
        self.assertEqual(fast["timing_source"], "wall_clock")
        self.assertEqual(fast["present_fps_avg"], 30.0)
        self.assertEqual(fast["frames_within_30fps_budget"], 60)
        self.assertTrue(fast["meets_30fps_target"])

        # 10 frames in 1.0s => 10.0 FPS average, 0 within 30 FPS budget
        slow = replay_capture.summarize_pacing(None, replayed_frames=10, elapsed_seconds=1.0)
        self.assertEqual(slow["present_fps_avg"], 10.0)
        self.assertEqual(slow["frames_within_30fps_budget"], 0)
        self.assertFalse(slow["meets_30fps_target"])

    def test_parse_frame_times_from_log(self) -> None:
        log = "frame 0 present_ms=16.5\nframe 1 frame_ms=42.25\nframe 2 dt_ms: 33.0\n"
        self.assertEqual(replay_capture.parse_frame_times_from_log(log), [16.5, 42.25, 33.0])


class TestSanitizedJsonAndCliExitCodes(unittest.TestCase):
    def _assert_no_host_paths(self, json_str: str) -> None:
        # Zero Windows drive paths, UNC paths, file:// URIs, or Unix user/system paths
        self.assertNotRegex(json_str, r"[A-Za-z]:[\\/]")
        self.assertNotIn("file://", json_str.lower())
        self.assertNotIn("\\\\", json_str)
        self.assertNotRegex(json_str, r"/(?:Users|home|tmp|var|opt|usr|mnt|private|root)/")

    def test_sanitize_text_scrubs_windows_posix_unc_and_file_uris(self) -> None:
        sample = (
            r"Failed reading C:\Users\alice\secret_game\dump.pg6trace and "
            r"C:/Program Files/Google/Chrome/chrome.exe and "
            r"/home/alice/local-data/capture.pg6trace and "
            r"file:///Users/bob/replay.html and "
            r"\\fileserver\share\game\build.bin; non-black 10/100 (10.00%)"
        )
        cleaned = replay_capture.sanitize_text(sample)
        self._assert_no_host_paths(cleaned)
        self.assertIn("non-black 10/100 (10.00%)", cleaned)
        self.assertNotIn("alice", cleaned)
        self.assertNotIn("bob", cleaned)
        self.assertNotIn("fileserver", cleaned)

    def test_build_replay_summary_sanitizes_all_fields(self) -> None:
        lit = make_solid_rgb_png(4, 4, 20, 200, 40)
        summary = replay_capture.build_replay_summary(
            trace_name=r"C:\Users\alice\work\replay\capture.pg6trace",
            pack_name="/home/alice/work/replay/shaders.wgsl.pack",
            log_text=(
                "swapchain frame 0 (960x540 bgra8unorm): non-black 100/100 (100.00%), distinct 1\n"
                "present_ms=20.0\n"
                "present_ms=25.0\n"
                "RESULT: replayed 2 frame(s), 0 validation error(s)\n"
            ),
            thumbs={r"C:\Users\alice\out\swap0.png": lit},
            elapsed_seconds=0.05,
            verify_non_black=True,
            error=r"note: C:\Users\alice\local-data\secret.bin",
            use_pillow=False,
        )
        payload = json.dumps(summary, sort_keys=True)
        self._assert_no_host_paths(payload)
        self.assertEqual(summary["trace"], "capture.pg6trace")
        self.assertEqual(summary["pack"], "shaders.wgsl.pack")
        self.assertEqual(summary["luminance"]["frames"][0]["label"], "swap0.png")
        self.assertEqual(summary["present_fps_avg"], round(1000.0 / 22.5, 3))
        self.assertEqual(summary["frames_within_30fps_budget"], 2)
        self.assertEqual(summary["non_black_frames"], 1)
        self.assertEqual(summary["black_frames"], 0)
        self.assertFalse(summary["all_black_regression"])
        self.assertGreater(summary["max_luma"], 100.0)
        self.assertNotIn("alice", payload)

    def _run_mocked_main(self, *, thumbs_map: dict[str, str], log_text: str,
                         frame_times_ms: list[float] | None, extra_args: list[str]):
        class FakeProc:
            def kill(self):
                pass

            def wait(self):
                return 0

        class FakeCdp:
            def __init__(self, _ws_url):
                pass

            def call(self, _method, **_params):
                return {}

            def evaluate(self, expr):
                if "getElementById('log')" in expr:
                    return log_text
                if "window.__thumbs" in expr:
                    return json.dumps(thumbs_map)
                if "window.__frameTimesMs" in expr:
                    return json.dumps(frame_times_ms or [])
                return ""

        fake_tabs = io.BytesIO(json.dumps([
            {"type": "page", "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/1"}
        ]).encode("utf-8"))

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "cap.pg6trace").write_bytes(pkt(7))
            (d / "shaders.wgsl.pack").write_bytes(b"WGPK" + struct.pack("<3I", 1, 0, 0))
            out_prefix = str(d / "out")
            argv = [
                "--chrome", r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                "--replay-dir", str(d),
                "--trace", "cap.pg6trace",
                "--pack", "shaders.wgsl.pack",
                "--out", out_prefix,
                "--thumbat", "0,1",
                "--web-root", str(d),
                *extra_args,
            ]
            stdout_buf = io.StringIO()
            with (
                mock.patch.object(replay_capture.subprocess, "Popen", return_value=FakeProc()),
                mock.patch.object(replay_capture.urllib.request, "urlopen", return_value=fake_tabs),
                mock.patch.object(replay_capture, "Cdp", FakeCdp),
                mock.patch.object(replay_capture.time, "sleep", lambda _s: None),
                contextlib.redirect_stdout(stdout_buf),
            ):
                rc = replay_capture.main(argv)
            return rc, stdout_buf.getvalue(), str(d)

    def test_main_verify_non_black_exits_nonzero_on_all_black_capture(self) -> None:
        black_url = to_data_url(make_solid_rgb_png(8, 8, 0, 0, 0))
        log_text = (
            "swapchain frame 0 (960x540 bgra8unorm): non-black 0/100 (0.00%), distinct 1\n"
            "swapchain frame 1 (960x540 bgra8unorm): non-black 0/100 (0.00%), distinct 1\n"
            "RESULT: replayed 2 frame(s), 0 validation error(s)\n"
        )
        rc, out, temp_dir = self._run_mocked_main(
            thumbs_map={"swap0": black_url, "swap1": black_url},
            log_text=log_text,
            frame_times_ms=[20.0, 22.0],
            extra_args=["--verify-non-black", "--json"],
        )
        self.assertNotEqual(rc, 0)
        data = json.loads(out)
        self.assertFalse(data["ok"])
        self.assertTrue(data["verify_non_black"])
        self.assertTrue(data["all_black_regression"])
        self.assertEqual(data["non_black_frames"], 0)
        self.assertEqual(data["black_frames"], 2)
        self.assertEqual(data["max_luma"], 0.0)
        self.assertEqual(data["frames_within_30fps_budget"], 2)
        self.assertGreater(data["present_fps_avg"], 30.0)
        self._assert_no_host_paths(out)
        self.assertNotIn(temp_dir, out)

    def test_main_verify_non_black_exits_zero_on_non_black_capture(self) -> None:
        black_url = to_data_url(make_solid_rgb_png(8, 8, 0, 0, 0))
        lit_url = to_data_url(make_solid_rgb_png(8, 8, 0, 127, 255))
        log_text = (
            "swapchain frame 0 (960x540 bgra8unorm): non-black 0/100 (0.00%), distinct 1\n"
            "swapchain frame 1 (960x540 bgra8unorm): non-black 100/100 (100.00%), distinct 1\n"
            "RESULT: replayed 2 frame(s), 0 validation error(s)\n"
        )
        rc, out, temp_dir = self._run_mocked_main(
            thumbs_map={"swap0": black_url, "swap1": lit_url},
            log_text=log_text,
            frame_times_ms=[16.5, 45.0],
            extra_args=["--verify-non-black", "--json"],
        )
        self.assertEqual(rc, 0)
        data = json.loads(out)
        self.assertTrue(data["ok"])
        self.assertFalse(data["all_black_regression"])
        self.assertEqual(data["non_black_frames"], 1)
        self.assertEqual(data["black_frames"], 1)
        self.assertGreater(data["max_luma"], 100.0)
        self.assertEqual(data["frames_within_30fps_budget"], 1)
        self.assertAlmostEqual(data["present_fps_avg"], round(1000.0 / 30.75, 3), places=3)
        self._assert_no_host_paths(out)
        self.assertNotIn(temp_dir, out)

    def test_main_without_verify_non_black_preserves_legacy_exit_code_and_output(self) -> None:
        black_url = to_data_url(make_solid_rgb_png(4, 4, 0, 0, 0))
        log_text = (
            "swapchain frame 0 (960x540 bgra8unorm): non-black 0/100 (0.00%), distinct 1\n"
            "RESULT: replayed 1 frame(s), 0 validation error(s)\n"
        )
        rc, out, _ = self._run_mocked_main(
            thumbs_map={"swap0": black_url},
            log_text=log_text,
            frame_times_ms=[25.0],
            extra_args=[],
        )
        self.assertEqual(rc, 0)
        self.assertIn("frame 0: 960x540 bgra8unorm non-black 0.00% distinct 1", out)
        self.assertIn("RESULT: replayed 1 frame(s), 0 validation error(s)", out)
        self.assertIn("present_fps_avg=40.000", out)
        self.assertIn("frames_within_30fps_budget=1/1", out)


class ResolveTrimTests(unittest.TestCase):
    MB = 1048576

    def test_small_trace_untouched(self):
        self.assertEqual(replay_capture.resolve_trim(100 * self.MB, 0, 3), (0, 3, False))

    def test_large_trace_auto_trims(self):
        self.assertEqual(replay_capture.resolve_trim(2000 * self.MB, 0, 3), (100, 12, True))
        self.assertEqual(replay_capture.resolve_trim(2000 * self.MB, 0, 20), (100, 20, True))

    def test_threshold_boundary_and_override(self):
        self.assertEqual(replay_capture.resolve_trim(1536 * self.MB, 0, 3)[2], False)
        self.assertEqual(replay_capture.resolve_trim(1536 * self.MB + 1, 0, 3)[2], True)
        self.assertEqual(replay_capture.resolve_trim(100 * self.MB, 0, 3, max_trace_mb=50)[2], True)

    def test_disabled_and_explicit(self):
        self.assertEqual(replay_capture.resolve_trim(5000 * self.MB, 0, 3, max_trace_mb=0), (0, 3, False))
        self.assertEqual(replay_capture.resolve_trim(5000 * self.MB, 50, 5), (50, 5, False))


if __name__ == "__main__":
    unittest.main()
