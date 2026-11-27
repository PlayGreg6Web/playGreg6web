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
        self.assertTrue(segs[-1]["folded"])
        self.assertFalse(any(s.get("folded") for s in segs[:-1]))

    def test_cap_fold_keeps_draw_range(self):
        # A 2-draw screen hits the cap, then a heavy scene follows: the folded tail must not read as
        # "2 draws/frame until the end".
        ps, total = [], 0
        for i, d in enumerate([5, 9, 2, 200, 150, 260, 219]):
            total += d
            ps.append(present(i * 1000, total, 0))
        segs = st.build_timeline(ps, max_segments=3)
        tail = segs[-1]
        self.assertEqual((tail["first_present"], tail["presents"]), (2, 5))
        self.assertTrue(tail["folded"])
        self.assertEqual((tail["draws_min"], tail["draws_max"], tail["draws_per_frame"]), (2, 260, 219))
        self.assertIn("folded past the cap: 2-260", st.format_table(segs)[-1])
        # Under the cap nothing is folded.
        self.assertFalse(any(s.get("folded") for s in st.build_timeline(ps)))

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
                    "black_frame_count",
                    "non_black_frame_count",
                    "all_black",
                    "first_non_black_frame",
                    "first_non_black_s",
                    "black_loading_s",
                    "segments",
                },
            )
            self.assertEqual(data["total_frames"], 10)
            self.assertEqual(data["unique_segments"], 5)
            self.assertEqual(data["black_frames"], 3)
            self.assertEqual(data["non_black_frames"], 7)
            self.assertEqual(data["black_frame_count"], 3)
            self.assertEqual(data["non_black_frame_count"], 7)
            self.assertFalse(data["all_black"])
            self.assertEqual(data["first_non_black_frame"], 3)
            self.assertEqual(data["first_non_black_s"], 3.0)
            # Leading black is frames 1..2 (3.0s); mid-capture black frame 5 does not inflate black_loading_s.
            self.assertEqual(data["black_loading_s"], 3.0)

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
                "delta_s",
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
            self.assertEqual([(s["start_s"], s["end_s"], s["dwell_s"], s["delta_s"]) for s in segs], [
                (0.0, 3.0, 3.0, 0.0),
                (3.0, 6.0, 3.0, 3.0),
                (6.0, 7.5, 1.5, 3.0),
                (7.5, 12.0, 4.5, 1.5),
                (12.0, 15.0, 3.0, 4.5),
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
            self.assertEqual(data["black_frame_count"], 1)
            self.assertEqual(data["non_black_frame_count"], 2)
            self.assertFalse(data["all_black"])
            self.assertEqual(data["first_non_black_frame"], 2)
            self.assertIsNone(data["first_non_black_s"])
            self.assertIsNone(data["black_loading_s"])
            for seg in data["segments"]:
                self.assertIsNone(seg["start_s"])
                self.assertIsNone(seg["end_s"])
                self.assertIsNone(seg["dwell_s"])
                self.assertIsNone(seg["delta_s"])

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
            self.assertNotIn("delta_s", out1)
            self.assertNotIn("black_loading_s", out1)
            self.assertNotIn(str(frames_dir), out1)
            self.assertNotIn(frames_dir.name, out1)

            rc2, out2, err2 = self._run_cli([str(frames_dir), "--interval-sec", "2.0"])
            self.assertEqual(rc2, 0, err2)
            self.assertIn("first_non_black: frame 2 @ 2.00s", out2)
            self.assertIn("black_loading_s: 2.00s", out2)
            self.assertIn("dwell_s", out2)
            self.assertIn("delta_s", out2)
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
                "black_frame_count": 0,
                "non_black_frame_count": 0,
                "all_black": False,
                "first_non_black_frame": None,
                "first_non_black_s": None,
                "black_loading_s": 0.0,
                "segments": [],
            })

            rc_no_int, out_no_int, err_no_int = self._run_cli([str(frames_dir), "--json"])
            self.assertEqual(rc_no_int, 0, err_no_int)
            data_no_int = json.loads(out_no_int)
            self.assertIsNone(data_no_int["black_loading_s"])
            self.assertIsNone(data_no_int["first_non_black_s"])
            self.assertFalse(data_no_int["all_black"])

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
            self.assertEqual(data["black_frame_count"], 3)
            self.assertEqual(data["non_black_frame_count"], 0)
            self.assertTrue(data["all_black"])
            self.assertIsNone(data["first_non_black_frame"])
            self.assertIsNone(data["first_non_black_s"])
            self.assertEqual(data["black_loading_s"], 1.5)
            self.assertEqual(data["segments"][0]["dwell_s"], 1.5)
            self.assertEqual(data["segments"][0]["delta_s"], 0.0)
            self.assertTrue(data["segments"][0]["is_black"])

            rc_txt, out_txt, err_txt = self._run_cli([str(frames_dir), "--interval-sec", "0.5"])
            self.assertEqual(rc_txt, 0, err_txt)
            self.assertIn("black_loading_s: 1.50s", out_txt)
            self.assertIn("all_black: true", out_txt)

    def test_immediate_non_black_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = Path(tmp)
            colors = [
                (220, 220, 220),
                (220, 220, 220),
                (40, 180, 90),
                (0, 0, 0),
            ]
            for idx, rgb in enumerate(colors, start=1):
                _write_solid_png(frames_dir / f"frame_{idx:04d}.png", rgb)

            rc, stdout, stderr = self._run_cli([str(frames_dir), "--interval-sec", "0.5", "--json"])
            self.assertEqual(rc, 0, stderr)
            data = json.loads(stdout)
            self.assertEqual(data["total_frames"], 4)
            self.assertEqual(data["unique_segments"], 3)
            self.assertEqual(data["black_frame_count"], 1)
            self.assertEqual(data["non_black_frame_count"], 3)
            self.assertFalse(data["all_black"])
            self.assertEqual(data["first_non_black_frame"], 1)
            self.assertEqual(data["first_non_black_s"], 0.0)
            self.assertEqual(data["black_loading_s"], 0.0)
            self.assertEqual([s["delta_s"] for s in data["segments"]], [0.0, 1.0, 0.5])

    def test_json_host_path_sanitization(self):
        with tempfile.TemporaryDirectory(prefix="playgta6_host_leak_") as tmp:
            nested = Path(tmp) / "Users" / "secret_alice" / "captures"
            nested.mkdir(parents=True)
            _write_solid_png(nested / "frame_0001.png", (0, 0, 0))
            _write_solid_png(nested / "frame_0002.png", (200, 100, 50))

            rc, stdout, stderr = self._run_cli([str(nested), "--interval-sec", "1.0", "--json"])
            self.assertEqual(rc, 0, stderr)
            self.assertEqual(stderr, "")
            for forbidden in (str(nested), str(tmp), "secret_alice", "playgta6_host_leak_", "C:\\", "/home/", "/Users/"):
                self.assertNotIn(forbidden, stdout)
            data = json.loads(stdout)
            self.assertEqual(data["black_loading_s"], 1.0)
            self.assertEqual(data["first_non_black_s"], 1.0)

        # Direct unit checks for host-path & username scrubbing helpers.
        self.assertEqual(
            st.sanitize_path_label(r"C:\Users\alice\Documents\captures\frame_0001.png"),
            "frame_0001.png",
        )
        self.assertEqual(
            st.sanitize_path_label(r"C:\Users\Alice Smith\My Captures\frame_0001.png"),
            "frame_0001.png",
        )
        self.assertEqual(
            st.sanitize_path_label("/home/bob/work/captures/frame_0002.png"),
            "frame_0002.png",
        )
        self.assertEqual(
            st.sanitize_path_label("/Users/carol/Desktop/frame_0003.png"),
            "frame_0003.png",
        )
        self.assertEqual(
            st.sanitize_path_label(r"\\fileserver\share\dave\frame_0004.png"),
            "frame_0004.png",
        )
        self.assertEqual(st.sanitize_path_label(r"C:\Users\alice"), "<redacted>")
        self.assertEqual(st.sanitize_path_label("/home/bob/"), "<redacted>")
        self.assertEqual(st.sanitize_path_label("../../Users/eve/frame_0005.png"), "frame_0005.png")
        self.assertEqual(st.sanitize_path_label("captures/frame_0006.png"), "captures/frame_0006.png")

        scrubbed = st.sanitize_json_payload({
            "win": r"C:\Users\alice\captures\frame_0001.png",
            "unix": "/home/bob/captures/frame_0002.png",
            "nested": [Path(r"C:\Users\alice\run\frame_0003.png"), {"msg": "loaded /Users/carol/out.png"}],
            "count": 3,
            "flag": False,
        })
        serialized = json.dumps(scrubbed)
        for leaked in ("C:\\", "/home/", "/Users/", "alice", "bob", "carol"):
            self.assertNotIn(leaked, serialized)
        self.assertEqual(scrubbed["win"], "frame_0001.png")
        self.assertEqual(scrubbed["unix"], "frame_0002.png")
        self.assertEqual(scrubbed["nested"][0], "frame_0003.png")
        self.assertEqual(scrubbed["nested"][1]["msg"], "loaded out.png")
        self.assertEqual(scrubbed["count"], 3)
        self.assertFalse(scrubbed["flag"])

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


def _build_synthetic_boot_presents(
    *,
    bink2_light_fps: float = 40.0,
    bink2_heavy_fps: float = 20.0,
    menu_fps: float = 25.0,
    spinner_fps: float = 30.0,
    spinner_s: float = 6.0,
) -> list[dict]:
    """Build cumulative-counter presents: Start Screen -> 4K Bink2 -> Menu (2 screens) -> Loading Spinner."""
    presents: list[dict] = []
    t_ms = 0.0
    total_draws = 0
    total_res = 5

    def emit(count: int, dt_ms: float, draws_per_frame: int, res_at_start: int = 0) -> None:
        nonlocal t_ms, total_draws, total_res
        for idx in range(count):
            if idx == 0 and res_at_start:
                total_res += res_at_start
            total_draws += draws_per_frame
            presents.append({
                "wall_ms": round(t_ms, 3),
                "counters": {"draws_issued": total_draws, "resources_created": total_res},
            })
            t_ms += dt_ms

    # Start Screen: 12 draws/p for 2s @ 25 FPS (pre-Bink2 -> must not pollute post-Bink2 menu)
    emit(50, 40.0, 12, res_at_start=3)
    # 4K Bink2 light window: 10s @ bink2_light_fps (4 draws/p)
    light_dt = 1000.0 / bink2_light_fps
    emit(int(round(10.0 * bink2_light_fps)), light_dt, 4, res_at_start=2)
    # 4K Bink2 heavy window: 10s @ bink2_heavy_fps (4 draws/p)
    heavy_dt = 1000.0 / bink2_heavy_fps
    emit(int(round(10.0 * bink2_heavy_fps)) + 1, heavy_dt, 4, res_at_start=0)
    # Menu screen 1 (UI TEXT SIZE): 15 draws/p for 5s @ menu_fps
    menu_dt = 1000.0 / menu_fps
    emit(int(round(5.0 * menu_fps)), menu_dt, 15, res_at_start=4)
    # Menu screen 2 (BRIGHTNESS): 25 draws/p for 5s @ menu_fps
    emit(int(round(5.0 * menu_fps)) + 1, menu_dt, 25, res_at_start=1)
    # Loading Spinner: 1 draw/p sustained for spinner_s @ spinner_fps
    spin_dt = 1000.0 / spinner_fps
    emit(int(round(spinner_s * spinner_fps)) + 1, spin_dt, 1, res_at_start=1)
    return presents


class BootPhaseAnalysisTests(unittest.TestCase):
    def _run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = st.main(argv)
        return rc, out.getvalue(), err.getvalue()

    def test_analyze_boot_phases_full_synthetic_boot(self):
        presents = _build_synthetic_boot_presents(
            bink2_light_fps=40.0,
            bink2_heavy_fps=20.0,
            menu_fps=25.0,
            spinner_fps=30.0,
            spinner_s=6.0,
        )
        phases = st.analyze_boot_phases(presents, window_ms=5000.0, black_loading_s=4.5)
        self.assertEqual(phases["window_ms"], 5000.0)
        self.assertEqual(phases["bink2_draws_range"], [3, 5])
        self.assertEqual(phases["black_loading_s"], 4.5)

        bink2 = phases["bink2_video"]
        self.assertIsNotNone(bink2)
        self.assertEqual(bink2["segments"], 1)
        self.assertAlmostEqual(bink2["duration_s"], 20.0, delta=0.1)
        self.assertAlmostEqual(bink2["fps_min"], 20.0, delta=0.5)
        self.assertAlmostEqual(bink2["fps_max"], 40.0, delta=0.5)
        self.assertAlmostEqual(bink2["fps_median"], 30.0, delta=0.5)
        self.assertAlmostEqual(bink2["fps_avg"], 30.0, delta=0.5)
        self.assertGreater(bink2["presents"], 500)
        self.assertAlmostEqual(bink2["start_s"], 2.0, delta=0.1)
        self.assertAlmostEqual(bink2["end_s"], 22.0, delta=0.1)

        menu = phases["menu"]
        self.assertIsNotNone(menu)
        self.assertEqual(menu["segments"], 2)
        self.assertAlmostEqual(menu["duration_s"], 10.0, delta=0.15)
        self.assertAlmostEqual(menu["fps_avg"], 25.0, delta=0.5)
        self.assertAlmostEqual(menu["fps_median"], 25.0, delta=0.5)
        self.assertAlmostEqual(menu["fps_min"], 25.0, delta=0.5)
        self.assertAlmostEqual(menu["fps_max"], 25.0, delta=0.5)

        spinner = phases["loading_spinner"]
        self.assertIsNotNone(spinner)
        self.assertEqual(spinner["segments"], 1)
        self.assertAlmostEqual(spinner["duration_s"], 6.0, delta=0.1)
        self.assertEqual(phases["loading_spinner_s"], spinner["duration_s"])
        self.assertAlmostEqual(spinner["fps_avg"], 30.0, delta=0.5)

        summary_lines = st.format_phase_summary(phases)
        summary_text = "\n".join(summary_lines)
        self.assertIn("bink2_video:", summary_text)
        self.assertIn("menu:", summary_text)
        self.assertIn("loading_spinner:", summary_text)
        self.assertIn("black_loading_s: 4.50s", summary_text)

    def test_present_samples_polled_browser_and_explicit_phase_tags(self):
        # 1. Per-present `draws` format
        per_present = [
            {"index": i, "wall_ms": i * 50.0, "draws": 4, "resources": 0}
            for i in range(6)
        ] + [
            {"index": 6 + i, "wall_ms": 300.0 + i * 100.0, "draws": 20, "resources": 0}
            for i in range(5)
        ]
        p1 = st.analyze_boot_phases(per_present, window_ms=5000.0)
        self.assertIsNotNone(p1["bink2_video"])
        self.assertEqual(p1["bink2_video"]["fps_avg"], 20.0)
        self.assertIsNotNone(p1["menu"])
        self.assertEqual(p1["menu"]["fps_avg"], 10.0)

        # 2. Polled browser presents with `present_count` + `draws_per_frame`
        polled = [
            {"wall_ms": 10000.0, "present_count": 100, "draws_per_frame": 4},
            {"wall_ms": 15000.0, "present_count": 250, "draws_per_frame": 4},  # 30 fps
            {"wall_ms": 20000.0, "present_count": 350, "draws_per_frame": 4},  # 20 fps
            {"wall_ms": 25000.0, "present_count": 475, "draws_per_frame": 22}, # menu
            {"wall_ms": 30000.0, "present_count": 625, "draws_per_frame": 22}, # 30 fps menu
        ]
        p2 = st.analyze_boot_phases(polled, window_ms=5000.0)
        self.assertEqual(p2["bink2_video"]["fps_min"], 20.0)
        self.assertEqual(p2["bink2_video"]["fps_max"], 30.0)
        self.assertEqual(p2["bink2_video"]["fps_median"], 25.0)
        self.assertEqual(p2["bink2_video"]["fps_avg"], 25.0)
        self.assertEqual(p2["bink2_video"]["duration_s"], 10.0)
        self.assertEqual(p2["menu"]["fps_avg"], 30.0)

        # 3. Polled browser presents with `presents` + cumulative `counters.draws_issued` + explicit phase tags
        tagged = [
            {"wall_ms": 1000.0, "presents": 10, "phase": "start_screen", "counters": {"draws_issued": 120}},
            {"wall_ms": 2000.0, "presents": 30, "phase": "start_screen", "counters": {"draws_issued": 360}},
            {"wall_ms": 3000.0, "presents": 60, "phase": "bink2_video", "counters": {"draws_issued": 480}},
            {"wall_ms": 4000.0, "presents": 90, "phase": "bink2_video", "counters": {"draws_issued": 600}},
            {"wall_ms": 5000.0, "presents": 110, "phase": "loading_spinner", "counters": {"draws_issued": 620}},
            {"wall_ms": 6000.0, "presents": 130, "phase": "loading_spinner", "counters": {"draws_issued": 640}},
        ]
        p3 = st.analyze_boot_phases(tagged, window_ms=5000.0)
        self.assertIsNotNone(p3["bink2_video"])
        self.assertEqual(p3["bink2_video"]["fps_avg"], 30.0)
        self.assertIsNone(p3["menu"])
        self.assertIsNotNone(p3["loading_spinner"])
        self.assertEqual(p3["loading_spinner"]["duration_s"], 1.0)
        self.assertEqual(p3["loading_spinner"]["fps_avg"], 20.0)

    def test_short_phase_fallback_empty_and_validation(self):
        # Short phase (< 5000ms window) falls back to inter-present FPS cleanly
        short = [
            {"wall_ms": 0.0, "draws": 4},
            {"wall_ms": 50.0, "draws": 4},   # 20 fps
            {"wall_ms": 150.0, "draws": 4},  # 10 fps
            {"wall_ms": 250.0, "draws": 4},  # 10 fps
            {"wall_ms": 450.0, "draws": 4},  # 5 fps
        ]
        p = st.analyze_boot_phases(short, window_ms=5000.0)
        self.assertEqual(p["bink2_video"]["fps_min"], 5.0)
        self.assertEqual(p["bink2_video"]["fps_median"], 10.0)
        self.assertEqual(p["bink2_video"]["fps_max"], 20.0)
        self.assertEqual(p["bink2_video"]["fps_avg"], 8.89)
        self.assertEqual(p["bink2_video"]["duration_s"], 0.45)

        # Empty input
        empty = st.analyze_boot_phases([])
        self.assertIsNone(empty["bink2_video"])
        self.assertIsNone(empty["menu"])
        self.assertIsNone(empty["loading_spinner"])
        self.assertIsNone(empty["loading_spinner_s"])
        self.assertEqual(st.format_phase_summary(empty), ["boot phases: none detected"])

        # Explicit loading_spinner_s without presents
        only_spinner = st.analyze_boot_phases([], loading_spinner_s=7.25)
        self.assertEqual(only_spinner["loading_spinner_s"], 7.25)
        self.assertEqual(only_spinner["loading_spinner"]["duration_s"], 7.25)

        # Invalid parameters raise ValueError
        for bad_win in (0, -100, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                st.analyze_boot_phases([], window_ms=bad_win)
        for bad_rng in ((5, 3), (-1, 4), (3,)):
            with self.assertRaises(ValueError):
                st.analyze_boot_phases([], bink2_draws_range=bad_rng)  # type: ignore[arg-type]

    def test_extract_boot_phases_from_report_fallbacks_and_cli(self):
        # 1. screen_timeline fallback + summary overrides
        report_tl = {
            "schema": "playgta6-boot-report",
            "screen_timeline": [
                {"start_time": 1000.0, "end_time": 11000.0, "presents": 301, "draws_per_frame": 4},
                {"start_time": 11000.0, "end_time": 16000.0, "presents": 126, "draws_per_frame": 20},
                {"start_time": 16000.0, "end_time": 20000.0, "presents": 121, "draws_per_frame": 1},
            ],
            "summary": {"black_loading_s": 3.5},
        }
        p_tl = st.extract_boot_phases_from_report(report_tl)
        self.assertEqual(p_tl["bink2_video"]["fps_avg"], 30.0)
        self.assertEqual(p_tl["menu"]["fps_avg"], 25.0)
        self.assertEqual(p_tl["loading_spinner"]["duration_s"], 4.0)
        self.assertEqual(p_tl["loading_spinner_s"], 4.0)
        self.assertEqual(p_tl["black_loading_s"], 3.5)

        # 2. Pre-computed summary/telemetry fields + bad/path values rejected
        report_pre = {
            "schema": "playgta6-boot-report",
            "summary": {
                "bink2_fps_min": 18.5,
                "bink2_fps_median": 28.0,
                "bink2_fps_max": 34.0,
                "bink2_duration_s": 42.0,
                "menu_fps_median": 29.0,
                "menu_fps_avg": 27.5,
                "loading_spinner_s": 5.2,
                "secret": r"C:\Users\secret_alice\dump.bin",
            },
            "telemetry": {
                "bink2_fps_avg": r"/home/secret_bob/bad",
                "black_loading_s": -2.0,
            },
        }
        p_pre = st.extract_boot_phases_from_report(report_pre)
        self.assertEqual(p_pre["bink2_video"]["fps_min"], 18.5)
        self.assertEqual(p_pre["bink2_video"]["fps_median"], 28.0)
        self.assertEqual(p_pre["bink2_video"]["fps_max"], 34.0)
        self.assertEqual(p_pre["bink2_video"]["duration_s"], 42.0)
        self.assertIsNone(p_pre["bink2_video"]["fps_avg"])
        self.assertEqual(p_pre["menu"]["fps_median"], 29.0)
        self.assertEqual(p_pre["menu"]["fps_avg"], 27.5)
        self.assertEqual(p_pre["loading_spinner_s"], 5.2)
        self.assertIsNone(p_pre["black_loading_s"])
        serialized = json.dumps(p_pre)
        self.assertNotIn("secret_alice", serialized)
        self.assertNotIn("secret_bob", serialized)

        # 3. CLI --boot-report JSON, plain text, and sanitized errors
        with tempfile.TemporaryDirectory(prefix="playgta6_secret_report_") as tmp:
            nested = Path(tmp) / "Users" / "secret_alice"
            nested.mkdir(parents=True)
            rep_path = nested / "boot-report.json"
            rep_path.write_text(json.dumps({
                "schema": "playgta6-boot-report",
                "presents": _build_synthetic_boot_presents(),
                "black_loading_s": 2.5,
            }), encoding="utf-8")

            rc_j, out_j, err_j = self._run_cli(["--boot-report", str(rep_path), "--json"])
            self.assertEqual(rc_j, 0, err_j)
            self.assertEqual(err_j, "")
            for forbidden in (str(rep_path), str(tmp), "secret_alice", "playgta6_secret_report_"):
                self.assertNotIn(forbidden, out_j)
            data = json.loads(out_j)
            self.assertIsNotNone(data["bink2_video"])
            self.assertIsNotNone(data["menu"])
            self.assertIsNotNone(data["loading_spinner"])

            rc_t, out_t, err_t = self._run_cli(["--boot-report", str(rep_path)])
            self.assertEqual(rc_t, 0, err_t)
            self.assertIn("bink2_video:", out_t)
            self.assertIn("menu:", out_t)
            self.assertIn("loading_spinner:", out_t)
            self.assertNotIn("secret_alice", out_t)

            # Missing boot report -> exit 2, sanitized stderr
            missing = nested / "missing.json"
            rc_m, _, err_m = self._run_cli(["--boot-report", str(missing)])
            self.assertEqual(rc_m, 2)
            self.assertNotIn("secret_alice", err_m)
            self.assertNotIn(str(missing), err_m)

            # Both frames_dir and --boot-report -> exit 2
            rc_both, _, _ = self._run_cli([str(nested), "--boot-report", str(rep_path)])
            self.assertEqual(rc_both, 2)

            # Neither -> exit 2
            rc_none, _, _ = self._run_cli([])
            self.assertEqual(rc_none, 2)


if __name__ == "__main__":
    unittest.main()
