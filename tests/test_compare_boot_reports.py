import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import compare_boot_reports as cbr  # noqa: E402
import gpu_trace  # noqa: E402


def make_report() -> dict:
    return {
        "schema": "playgta6-boot-report",
        "schema_version": 1,
        "status": "fault",
        "exit_code": None,
        "phase": "entry",
        "milestones": [
            {"milestone": "PGIM_LOAD", "elapsed_ms": 100},
            {"milestone": "D3D12_CREATE_DEVICE", "elapsed_ms": 900},
        ],
        "first_calls": [
            {"name": "D3D12XboxCreateDevice", "import_seq": 500, "elapsed_ms": 900, "thread": 1},
        ],
        "fault": {"kind": "unimplemented", "trap_kind": None, "va": "0x1000", "detail": "0x0",
                  "function": {"entry": "0x0f00", "offset": "0x100"}, "import": None},
        "backtrace": {"method": "scan", "frames": [{"function": "0x0f00"}, {"function": "0x0e00"}]},
        "counters": {"dispatches": 10, "import_calls": 1000,
                     "imports": [{"dll": "k", "symbol": "Foo", "resolved": True, "count": 800}]},
        "address_space": {"events": 2, "failures": 0, "reserved_peak": "0x2000", "log": [
            {"event": "reserve", "ok": True, "va": "0x10000", "size": "0x1000", "protect": "0x4"},
            {"event": "commit", "ok": True, "va": "0x10000", "size": "0x800", "protect": "0x4"},
        ]},
    }


def make_report_v2() -> dict:
    """A schema 2 report: milestone import indices, per-Present samples, first_present counts."""
    r = make_report()
    r["schema_version"] = 2
    r["milestones"][0]["import_seq"] = 0
    r["milestones"][1]["import_seq"] = 500
    r["presents"] = make_sidecar(2)["presents"]
    r["first_present"] = {"import_seq": 100, "elapsed_ms": 950, "imports": [
        {"thread": 1, "dll": "k", "symbol": "A", "count": 2},
        {"thread": 1, "dll": "k", "symbol": "B", "count": 1},
        {"thread": 2, "dll": "k", "symbol": "B", "count": 1},
    ]}
    return r


def make_trace(draws_per_frame: list[int]) -> bytes:
    out = b""
    for n in draws_per_frame:
        out += gpu_trace.encode_begin_frame()
        for _ in range(n):
            out += gpu_trace.encode_draw(3)
        out += gpu_trace.encode_present()
    return out


def make_sidecar(frames: int, wall_ms: float = 10.0) -> dict:
    return {
        "schema": "playgta6-frame-capture", "schema_version": 1, "payloads": "omitted",
        "frames_complete": frames, "omitted": [],
        "presents": [{"index": i, "wall_ms": wall_ms * (i + 1), "dispatches": 5 * (i + 1),
                      "import_calls": 100 * (i + 1), "reserved_bytes": 4096, "committed_bytes": 1024,
                      "counters": {"draws_ok": 2}} for i in range(frames)],
    }


class CompareTests(unittest.TestCase):
    tol = cbr.Tolerances()

    def run_compare(self, a, b, **kw):
        return cbr.compare(a, b, self.tol, **kw)

    def fields(self, c):
        return {e["field"] for e in c.failures()}

    def test_identical_passes(self):
        c = self.run_compare(make_report(), make_report())
        self.assertEqual(c.failures(), [])
        self.assertEqual(cbr.to_result(c)["verdict"], "pass")

    def test_changed_milestone_fails(self):
        b = make_report()
        b["milestones"].pop()
        c = self.run_compare(make_report(), b)
        self.assertIn("milestones.sequence", self.fields(c))

    def test_repeated_milestone_names_are_compared_individually(self):
        a = make_report()
        a["milestones"] = [{"milestone": "PGIM_LOAD", "elapsed_ms": 100}, {"milestone": "PGIM_LOAD", "elapsed_ms": 200}]
        b = copy.deepcopy(a)
        self.assertEqual(self.run_compare(a, b).failures(), [])
        b["milestones"][0]["elapsed_ms"] = 9000  # the first of the two differs, the last one does not
        self.assertIn("milestones.PGIM_LOAD.elapsed_ms", self.fields(self.run_compare(a, b)))
        b = copy.deepcopy(a)
        b["milestones"][1]["elapsed_ms"] = 9000
        self.assertIn("milestones.PGIM_LOAD#2.elapsed_ms", self.fields(self.run_compare(a, b)))

    def test_changed_stop_fails(self):
        b = make_report()
        b["fault"]["va"] = "0x2000"
        b["backtrace"]["frames"][1]["function"] = "0x0d00"
        f = self.fields(self.run_compare(make_report(), b))
        self.assertTrue({"stop.fault.va", "stop.backtrace.functions"} <= f)

    def test_timing_jitter_within_tolerance_passes(self):
        b = make_report()
        b["milestones"][1]["elapsed_ms"] = 1100
        b["first_calls"][0]["elapsed_ms"] = 1100
        b["first_calls"][0]["import_seq"] = 510
        b["counters"]["imports"][0]["count"] = 810
        self.assertEqual(self.run_compare(make_report(), b).failures(), [])
        b["milestones"][1]["elapsed_ms"] = 9000
        self.assertIn("milestones.D3D12_CREATE_DEVICE.elapsed_ms", self.fields(self.run_compare(make_report(), b)))

    def test_ignore_pattern(self):
        b = make_report()
        b["fault"]["va"] = "0x2000"
        c = self.run_compare(make_report(), b, ignore=["stop.fault.*"])
        self.assertEqual(c.failures(), [])

    def test_address_space_difference_located(self):
        b = make_report()
        b["address_space"]["log"][1]["size"] = "0x900"
        c = self.run_compare(make_report(), b)
        entry = next(e for e in c.failures() if e["field"] == "address_space.log")
        self.assertEqual(entry["a"]["first_difference"], 1)

    def test_capture_differences_reported_per_frame(self):
        ca = cbr.Capture(make_sidecar(2), gpu_trace.decode_stream(make_trace([1, 2])))
        cb = cbr.Capture(make_sidecar(2), gpu_trace.decode_stream(make_trace([1, 3])))
        same = self.run_compare(make_report(), make_report(), captures=(ca, ca))
        self.assertEqual(same.failures(), [])
        f = self.fields(self.run_compare(make_report(), make_report(), captures=(ca, cb)))
        self.assertIn("capture.frame[1].packets", f)
        self.assertFalse(any(x.startswith("capture.frame[0]") for x in f))

    def test_present_distribution_and_import_trace(self):
        ca = cbr.Capture(make_sidecar(2), gpu_trace.decode_stream(make_trace([1, 1])))
        cb = cbr.Capture(make_sidecar(2, wall_ms=500.0), gpu_trace.decode_stream(make_trace([1, 1])))
        rows_a = [(1, 1, "k!A"), (2, 1, "k!A"), (3, 2, "k!B"), (150, 1, "k!A")]
        rows_b = [(1, 1, "k!A"), (2, 1, "k!A"), (3, 2, "k!C"), (150, 1, "k!A")]
        c = self.run_compare(make_report(), make_report(), captures=(ca, cb), import_traces=(rows_a, rows_b))
        f = self.fields(c)
        self.assertIn("presents.wall_ms.median", f)
        self.assertIn("import_trace.overall.k!B", f)
        self.assertNotIn("import_trace.busiest_thread.k!A", f)  # 2 calls up to the first present in both


class ProgressCounterTests(unittest.TestCase):
    """R40: gpu_work and memory.cpu_window.alias_* are compared with count tolerance."""

    @staticmethod
    def report(draws_emitted: int, skipped: int, alias_slots: int) -> dict:
        r = make_report_v2()
        r["gpu_work"] = {"draws_emitted": draws_emitted, "draws_no_shader_module": skipped,
                         "dispatches_emitted": 10, "dispatches_no_shader_module": 0}
        r["memory"] = {"cpu_window": {"alias_slots": alias_slots, "alias_conflicts": 0}}
        return r

    def run_compare(self, a: dict, b: dict, ignore=()) -> cbr.Comparison:
        return cbr.compare(a, b, cbr.Tolerances(), ignore)

    def test_equal_and_within_tolerance_pass(self) -> None:
        c = self.run_compare(self.report(1000, 5, 20), self.report(1020, 5, 20))
        self.assertEqual(c.failures(), [])
        self.assertTrue(any(e["field"] == "gpu_work.draws_emitted" for e in c.entries))
        self.assertTrue(any(e["field"] == "memory.alias.alias_slots" for e in c.entries))

    def test_skip_regression_is_flagged_and_ignorable(self) -> None:
        a, b = self.report(1000, 0, 20), self.report(1000, 400, 60)
        fields = {e["field"] for e in self.run_compare(a, b).failures()}
        self.assertEqual(fields, {"gpu_work.draws_no_shader_module", "memory.alias.alias_slots"})
        c = self.run_compare(a, b, ["gpu_work.*", "memory.alias.*"])
        self.assertEqual(c.failures(), [])

    def test_missing_on_one_side_is_not_compared(self) -> None:
        c = self.run_compare(make_report_v2(), self.report(1, 1, 1))
        self.assertFalse(any(e["field"].startswith(("gpu_work", "memory.alias")) for e in c.entries))


class ReportVersionTests(unittest.TestCase):
    tol = cbr.Tolerances()

    def run_compare(self, a, b, **kw):
        return cbr.compare(a, b, self.tol, **kw)

    def fields(self, c):
        return {e["field"] for e in c.failures()}

    def test_v2_reports_compare_from_report_fields_alone(self):
        self.assertEqual(self.run_compare(make_report_v2(), make_report_v2()).failures(), [])
        b = make_report_v2()
        b["milestones"][1]["import_seq"] = 900
        b["presents"][1]["dispatches"] = 500
        b["first_present"]["imports"][0]["count"] = 9
        c = self.run_compare(make_report_v2(), b)
        names = {e["field"] for e in c.entries}
        self.assertIn("presents.dispatches.max", names)
        self.assertIn("import_trace.busiest_thread.k!A", names)
        f = self.fields(c)
        self.assertTrue({"milestones.D3D12_CREATE_DEVICE.import_seq", "presents.dispatches.max",
                         "import_trace.overall.k!A"} <= f)
        self.assertNotIn("import_trace.overall.k!B", f)

    def test_v2_first_present_uses_busiest_thread(self):
        fp = cbr.first_present_from_report(make_report_v2())
        self.assertEqual(fp.import_seq, 100)
        self.assertEqual(fp.overall, {"k!A": 2, "k!B": 2})
        self.assertEqual(fp.busiest, {"k!A": 2, "k!B": 1})

    def test_v1_against_v2_falls_back_to_sidecar_and_trace(self):
        # A is schema 2; B is a schema 1 report whose data comes from the sidecar and import trace.
        c = cbr.compare(make_report_v2(), make_report(), self.tol)
        self.assertTrue(any(e["field"] == "import_trace.first_present_seen" and not e["ok"] for e in c.entries))
        self.assertFalse(any(e["field"].startswith("presents.") for e in c.entries))
        cap = cbr.Capture(make_sidecar(2), gpu_trace.decode_stream(make_trace([1, 1])))
        cap.sidecar["presents"][0]["import_calls"] = 100
        rows_b = [(1, 1, "k!A"), (2, 1, "k!A"), (3, 1, "k!B"), (4, 2, "k!B"), (150, 1, "k!A")]
        c = cbr.compare(make_report_v2(), make_report(), self.tol, captures=(cap, cap), import_traces=([], rows_b))
        self.assertEqual(c.failures(), [])


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cbr.main([str(a) for a in args])
        return code, out.getvalue(), err.getvalue()

    def test_exit_codes_and_json(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d, "a.json"), Path(d, "b.json")
            a.write_text(json.dumps(make_report()))
            rb = copy.deepcopy(make_report())
            b.write_text(json.dumps(rb))
            code, out, _ = self.run_cli(a, b)
            self.assertEqual(code, 0)
            self.assertIn("PASS", out)
            rb["status"] = "exited"
            b.write_text(json.dumps(rb))
            code, out, _ = self.run_cli(a, b, "--json")
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(out)["verdict"], "fail")
            code, _, err = self.run_cli(a, Path(d, "missing.json"))
            self.assertEqual(code, 2)
            self.assertIn("error", err)
            code, _, _ = self.run_cli(a)
            self.assertEqual(code, 2)

    def test_capture_files(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for name in ("a", "b"):
                t = Path(d, f"{name}.pg6trace")
                t.write_bytes(make_trace([1, 2]))
                Path(str(t) + ".json").write_text(json.dumps(make_sidecar(2)))
                r = Path(d, f"{name}.json")
                r.write_text(json.dumps(make_report()))
                paths.append((r, t))
            code, _, _ = self.run_cli(paths[0][0], paths[1][0], "--capture", paths[0][1], paths[1][1])
            self.assertEqual(code, 0)


def make_browser_report(
    *,
    real_pack_draws: int = 100,
    fallback_draws: int = 20,
    skipped_draws: int = 50,
    gpu_validation_errors: int = 2,
    non_black_canvas_frames: int | None = 4,
    first_present_ms: float = 1000.0,
    present_walls_ms: list[float] | None = None,
    present_count: int | None = None,
    elapsed_ms: float = 5000.0,
) -> dict:
    """Browser `playgta6-boot-report` v2 fixture matching `tools/run_browser_boot.py`."""
    walls = present_walls_ms if present_walls_ms is not None else [1000.0, 1500.0, 2000.0, 2500.0]
    total_presents = present_count if present_count is not None else len(walls)
    summary: dict = {
        "realPackDrawCount": real_pack_draws,
        "fallbackDrawCount": fallback_draws,
        "validationErrorCount": gpu_validation_errors,
        "inputScript": None,
    }
    if non_black_canvas_frames is not None:
        summary["capturedFrames"] = max(non_black_canvas_frames, 4)
        summary["nonBlackCanvasFrames"] = non_black_canvas_frames
    presents = [
        {
            "index": i,
            "wall_ms": w,
            "dispatches": 0,
            "import_calls": 0,
            "counters": {
                "draws_real_pack": real_pack_draws,
                "draws_fallback": fallback_draws,
                "draws_skipped": skipped_draws,
                "gpu_validation_errors": gpu_validation_errors,
            },
        }
        for i, w in enumerate(walls)
    ]
    return {
        "schema": "playgta6-boot-report",
        "schema_version": 2,
        "runner": "browser",
        "status": "ok",
        "exit_code": 0,
        "phase": "FIRST_PRESENT",
        "elapsed_ms": elapsed_ms,
        "milestones": [
            {"milestone": "PGIM_LOAD", "elapsed_ms": 200.0},
            {"milestone": "FIRST_PRESENT", "elapsed_ms": first_present_ms},
        ],
        "first_calls": [],
        "fault": None,
        "backtrace": {"method": "browser", "frames": []},
        "counters": {
            "import_calls": 0,
            "imports": [],
            "presents": total_presents,
            "real_pack_draws": real_pack_draws,
            "fallback_draws": fallback_draws,
            "skipped_draws": skipped_draws,
            "gpu_validation_errors": gpu_validation_errors,
            "page_errors": 0,
            "console_errors": 0,
        },
        "summary": summary,
        "address_space": {"failures": 0, "reserved_peak": 0, "events": 0, "log": []},
        "presents": presents,
        "first_present": {"import_seq": 0, "elapsed_ms": first_present_ms, "imports": []},
    }


class GpuDeltaTests(unittest.TestCase):
    tol = cbr.Tolerances()

    def test_legacy_reports_omit_gpu_delta_in_json_and_text(self):
        c = cbr.compare(make_report(), make_report(), self.tol)
        result = cbr.to_result(c)
        self.assertEqual(result["verdict"], "pass")
        self.assertNotIn("gpu_delta", result)
        self.assertNotIn("[gpu_delta]", cbr.format_text(c))
        self.assertEqual(
            set(result.keys()),
            {"verdict", "checked", "differences", "tolerances", "ignore", "entries"},
        )

    def test_browser_reports_surface_gpu_draw_and_frame_pacing_deltas(self):
        t70 = make_browser_report(
            real_pack_draws=100,
            fallback_draws=50,
            skipped_draws=200,
            gpu_validation_errors=3,
            non_black_canvas_frames=2,
            first_present_ms=5000.0,
            present_walls_ms=[5000.0, 5500.0, 6000.0, 6500.0],
        )
        t71 = make_browser_report(
            real_pack_draws=500,
            fallback_draws=10,
            skipped_draws=20,
            gpu_validation_errors=0,
            non_black_canvas_frames=18,
            first_present_ms=3000.0,
            present_walls_ms=[3000.0, 3100.0, 3200.0, 3300.0],
        )
        c = cbr.compare(t70, t71, self.tol)
        result = cbr.to_result(c)
        self.assertIn("gpu_delta", result)
        gd = result["gpu_delta"]

        self.assertEqual(gd["real_pack_draws"], {"a": 100, "b": 500, "delta": 400, "pct": 400.0})
        self.assertEqual(gd["fallback_draws"], {"a": 50, "b": 10, "delta": -40, "pct": -80.0})
        self.assertEqual(gd["skipped_draws"], {"a": 200, "b": 20, "delta": -180, "pct": -90.0})
        self.assertEqual(gd["gpu_validation_errors"], {"a": 3, "b": 0, "delta": -3, "pct": -100.0})
        self.assertEqual(gd["non_black_canvas_frames"], {"a": 2, "b": 18, "delta": 16, "pct": 800.0})
        self.assertEqual(gd["first_present_ms"], {"a": 5000.0, "b": 3000.0, "delta": -2000.0, "pct": -40.0})
        self.assertEqual(gd["presents"], {"a": 4, "b": 4, "delta": 0, "pct": 0.0})
        # 3 intervals over 1500 ms -> 2.0 fps (500 ms/frame) vs 3 intervals over 300 ms -> 10.0 fps (100 ms/frame)
        self.assertEqual(gd["present_fps"], {"a": 2.0, "b": 10.0, "delta": 8.0, "pct": 400.0})
        self.assertEqual(gd["frame_time_ms"], {"a": 500.0, "b": 100.0, "delta": -400.0, "pct": -80.0})
        self.assertEqual(gd["frame_time_median_ms"], {"a": 500.0, "b": 100.0, "delta": -400.0, "pct": -80.0})

        text = cbr.format_text(c)
        self.assertIn("[gpu_delta]", text)
        self.assertIn("real_pack_draws: 100 -> 500 (+400, +400.0%)", text)
        self.assertIn("skipped_draws: 200 -> 20 (-180, -90.0%)", text)
        self.assertIn("gpu_validation_errors: 3 -> 0 (-3, -100.0%)", text)
        self.assertIn("non_black_canvas_frames: 2 -> 18 (+16, +800.0%)", text)
        self.assertIn("first_present_ms: 5000.0 -> 3000.0 (-2000.0 ms, -40.0%)", text)
        self.assertIn("present_fps: 2.0 -> 10.0 (+8.0 fps, +400.0%)", text)
        self.assertIn("frame_time_ms: 500.0 -> 100.0 (-400.0 ms, -80.0%)", text)

    def test_subsampled_presents_compute_true_fps_without_poll_median(self):
        # Browser runs poll ~every 500ms while counters.presents records total presents.
        ra = make_browser_report(present_walls_ms=[1000.0, 2000.0, 3000.0], present_count=21)
        rb = make_browser_report(present_walls_ms=[1000.0, 2000.0, 3000.0], present_count=61)
        gd = cbr.compute_gpu_delta(ra, rb)
        # 20 intervals / 2s = 10.0 fps vs 60 intervals / 2s = 30.0 fps
        self.assertEqual(gd["present_fps"], {"a": 10.0, "b": 30.0, "delta": 20.0, "pct": 200.0})
        self.assertEqual(gd["frame_time_ms"], {"a": 100.0, "b": 33.33, "delta": -66.67, "pct": -66.67})
        self.assertNotIn("frame_time_median_ms", gd)

    def test_fallback_sources_and_zero_baseline_and_partial_fields(self):
        ra = make_report()
        rb = make_report()
        ra["counters"]["real_pack_draws"] = 0  # falsy zero must NOT fall through to summary
        ra["summary"] = {"realPackDrawCount": 999, "fallbackDrawCount": 4, "nonBlackCanvasFrames": 0}
        ra["milestones"].append({"milestone": "FIRST_PRESENT", "elapsed_ms": 1200.0})
        ra["counters"]["presents"] = 11
        ra["elapsed_ms"] = 3200.0  # 10 intervals in 2000ms -> 5.0 fps

        rb["counters"]["real_pack_draws"] = 15
        rb["summary"] = {"fallbackDrawCount": 0, "nonBlackCanvasFrames": 5}
        rb["milestones"].append({"milestone": "FIRST_PRESENT", "elapsed_ms": 800.0})
        rb["counters"]["presents"] = 21
        rb["elapsed_ms"] = 2800.0  # 20 intervals in 2000ms -> 10.0 fps
        rb["counters"]["skipped_draws"] = 3  # only in B -> omitted from gpu_delta

        gd = cbr.compute_gpu_delta(ra, rb)
        self.assertEqual(gd["real_pack_draws"], {"a": 0, "b": 15, "delta": 15, "pct": None})
        self.assertEqual(gd["fallback_draws"], {"a": 4, "b": 0, "delta": -4, "pct": -100.0})
        self.assertEqual(gd["non_black_canvas_frames"], {"a": 0, "b": 5, "delta": 5, "pct": None})
        self.assertEqual(gd["first_present_ms"], {"a": 1200.0, "b": 800.0, "delta": -400.0, "pct": -33.33})
        self.assertEqual(gd["present_fps"], {"a": 5.0, "b": 10.0, "delta": 5.0, "pct": 100.0})
        self.assertEqual(gd["frame_time_ms"], {"a": 200.0, "b": 100.0, "delta": -100.0, "pct": -50.0})
        self.assertNotIn("skipped_draws", gd)

    def test_gpu_delta_sanitization_rejects_booleans_strings_paths_and_non_finite(self):
        ra = make_report()
        rb = make_report()
        ra["counters"].update({
            "real_pack_draws": True,
            "fallback_draws": float("nan"),
            "skipped_draws": r"C:\Users\secret\game.exe",
            "gpu_validation_errors": float("inf"),
            "secret_host_path": "/home/alice/game.rpf",
        })
        rb["counters"].update({
            "real_pack_draws": 10,
            "fallback_draws": 5,
            "skipped_draws": 2,
            "gpu_validation_errors": 0,
            "secret_host_path": "/home/bob/game.rpf",
        })
        c = cbr.compare(ra, rb, self.tol)
        self.assertEqual(c.gpu_delta, {})
        result = cbr.to_result(c)
        self.assertNotIn("gpu_delta", result)
        serialized = json.dumps(result)
        self.assertNotIn("secret", serialized)
        self.assertNotIn("alice", serialized)

    def test_gpu_delta_is_additive_and_preserves_verdict_and_cli_exit_codes(self):
        # Identical reports pass with exit 0 while still showing gpu_delta summary.
        ra = make_browser_report(real_pack_draws=10, skipped_draws=0, gpu_validation_errors=0)
        rb = copy.deepcopy(ra)
        with tempfile.TemporaryDirectory() as d:
            pa, pb = Path(d, "a.json"), Path(d, "b.json")
            pa.write_text(json.dumps(ra), encoding="utf-8")
            pb.write_text(json.dumps(rb), encoding="utf-8")
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = cbr.main([str(pa), str(pb), "--json"])
            self.assertEqual(code, 0, err.getvalue())
            payload = json.loads(out.getvalue())
            self.assertEqual(payload["verdict"], "pass")
            self.assertEqual(payload["differences"], 0)
            self.assertIn("gpu_delta", payload)
            self.assertEqual(payload["gpu_delta"]["real_pack_draws"]["delta"], 0)

            # Changing only a gpu_delta-only counter (not in existing comparison checks) keeps verdict/exit 0!
            rb["summary"]["nonBlackCanvasFrames"] = 12
            pb.write_text(json.dumps(rb), encoding="utf-8")
            out2, err2 = io.StringIO(), io.StringIO()
            with redirect_stdout(out2), redirect_stderr(err2):
                code2 = cbr.main([str(pa), str(pb)])
            self.assertEqual(code2, 0, err2.getvalue())
            self.assertIn("[gpu_delta]", out2.getvalue())
            self.assertIn("non_black_canvas_frames: 4 -> 12 (+8, +200.0%)", out2.getvalue())
            self.assertIn("verdict: PASS", out2.getvalue())


class GpuAdapterAndVisibleFrameTests(unittest.TestCase):
    tol = cbr.Tolerances()

    def test_gpu_adapter_mismatch_emits_informational_warning_without_failing(self):
        ra = make_browser_report()
        rb = make_browser_report()
        ra["gpu_adapter"] = "nvidia / turing"
        rb["summary"]["gpu_adapter"] = "intel / gen-12lp"

        c = cbr.compare(ra, rb, self.tol)
        self.assertEqual(c.failures(), [])
        result = cbr.to_result(c)
        self.assertEqual(result["verdict"], "pass")
        self.assertEqual(
            result["gpu_adapter"],
            {"a": "nvidia / turing", "b": "intel / gen-12lp", "mismatch": True},
        )
        self.assertEqual(len(result["warnings"]), 1)
        self.assertIn("nvidia / turing", result["warnings"][0])
        self.assertIn("intel / gen-12lp", result["warnings"][0])

        text = cbr.format_text(c)
        self.assertIn("[gpu_adapter]", text)
        self.assertIn("adapter: nvidia / turing -> intel / gen-12lp", text)
        self.assertIn("[warnings]", text)
        self.assertIn("WARN gpu_adapter mismatch:", text)
        self.assertIn("verdict: PASS", text)

        with tempfile.TemporaryDirectory() as d:
            pa, pb = Path(d, "a.json"), Path(d, "b.json")
            pa.write_text(json.dumps(ra), encoding="utf-8")
            pb.write_text(json.dumps(rb), encoding="utf-8")
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = cbr.main([str(pa), str(pb), "--json"])
            self.assertEqual(code, 0, err.getvalue())
            payload = json.loads(out.getvalue())
            self.assertEqual(payload["verdict"], "pass")
            self.assertTrue(payload["gpu_adapter"]["mismatch"])
            self.assertEqual(len(payload["warnings"]), 1)

    def test_gpu_adapter_matching_and_canonical_normalization_emit_no_warning(self):
        ra = make_browser_report()
        rb = make_browser_report()
        ra["gpu_adapter"] = "NVIDIA / Turing"
        rb["summary"]["gpuAdapter"] = "nvidia/turing"

        c = cbr.compare(ra, rb, self.tol)
        result = cbr.to_result(c)
        self.assertEqual(
            result["gpu_adapter"],
            {"a": "NVIDIA / Turing", "b": "nvidia/turing", "mismatch": False},
        )
        self.assertNotIn("warnings", result)
        text = cbr.format_text(c)
        self.assertIn("[gpu_adapter]", text)
        self.assertNotIn("[warnings]", text)

    def test_gpu_adapter_single_side_and_dict_form(self):
        ra = make_browser_report()
        rb = make_browser_report()
        rb["gpu_adapter"] = {"vendor": "amd", "architecture": "rdna2", "secret": "/home/alice/x"}

        c = cbr.compare(ra, rb, self.tol)
        result = cbr.to_result(c)
        self.assertEqual(result["gpu_adapter"], {"a": None, "b": "amd / rdna2", "mismatch": False})
        self.assertNotIn("warnings", result)
        self.assertNotIn("alice", json.dumps(result))
        text = cbr.format_text(c)
        self.assertIn("adapter: <unknown> -> amd / rdna2", text)

    def test_gpu_adapter_sanitization_rejects_paths_control_chars_and_falls_through(self):
        ra = make_report()
        rb = make_report()
        ra["gpu_adapter"] = r"C:\Users\alice\nvidia.dll"
        ra["summary"] = {"gpu_adapter": "nvidia / turing"}
        rb["gpu_adapter"] = "/home/bob/driver.so"
        rb["summary"] = {"gpu_adapter": "intel\nFAKE_LOG_LINE"}

        c = cbr.compare(ra, rb, self.tol)
        result = cbr.to_result(c)
        self.assertEqual(result["gpu_adapter"], {"a": "nvidia / turing", "b": None, "mismatch": False})
        self.assertNotIn("warnings", result)
        serialized = json.dumps(result)
        self.assertNotIn("alice", serialized)
        self.assertNotIn("bob", serialized)
        self.assertNotIn("FAKE_LOG_LINE", serialized)

        for bad in (
            None,
            True,
            42,
            "",
            "   ",
            " / ",
            r"\\server\share\gpu",
            r"D:/Games/gpu.dll",
            "/Users/carol/libgpu.dylib",
            "/tmp/adapter",
            "nvidia\r\ninjected",
        ):
            self.assertIsNone(cbr.sanitize_gpu_adapter(bad), f"expected None for {bad!r}")

    def test_first_visible_frame_milestone_and_black_loading_ms_deltas(self):
        ra = make_browser_report(first_present_ms=45000.0)
        ra["milestones"].append({"milestone": "FIRST_VISIBLE_FRAME", "elapsed_ms": 195000.0})
        ra["summary"]["blackLoadingMs"] = 150000.0

        rb = make_browser_report(first_present_ms=45000.0)
        rb["milestones"].append({"milestone": "FIRST_VISIBLE_FRAME", "elapsed_ms": 65000.0})
        rb["summary"]["blackLoadingMs"] = 20000.0

        c = cbr.compare(ra, rb, self.tol)
        # Milestone timing regression/shift beyond tolerance is still checked in milestones section
        self.assertIn("milestones.FIRST_VISIBLE_FRAME.elapsed_ms", {e["field"] for e in c.failures()})

        result = cbr.to_result(c)
        gd = result["gpu_delta"]
        self.assertEqual(
            gd["first_visible_frame_ms"],
            {"a": 195000.0, "b": 65000.0, "delta": -130000.0, "pct": -66.67},
        )
        self.assertEqual(
            gd["black_loading_ms"],
            {"a": 150000.0, "b": 20000.0, "delta": -130000.0, "pct": -86.67},
        )

        text = cbr.format_text(c)
        self.assertIn("first_visible_frame_ms: 195000.0 -> 65000.0 (-130000.0 ms, -66.7%)", text)
        self.assertIn("black_loading_ms: 150000.0 -> 20000.0 (-130000.0 ms, -86.7%)", text)

        # Ignorable via --ignore while preserving gpu_delta output
        c_ignored = cbr.compare(ra, rb, self.tol, ignore=["milestones.FIRST_VISIBLE_FRAME.elapsed_ms"])
        self.assertEqual(c_ignored.failures(), [])
        self.assertIn("first_visible_frame_ms", c_ignored.gpu_delta)
        self.assertIn("black_loading_ms", c_ignored.gpu_delta)

    def test_black_loading_ms_derived_and_snake_case_and_zero_preserved(self):
        # Derived from FIRST_VISIBLE_FRAME - FIRST_PRESENT when explicit blackLoadingMs is absent
        ra = make_browser_report(first_present_ms=40000.0)
        ra["milestones"].append({"milestone": "FIRST_VISIBLE_FRAME", "elapsed_ms": 140000.0})
        rb = make_browser_report(first_present_ms=40000.0)
        rb["milestones"].append({"milestone": "FIRST_VISIBLE_FRAME", "elapsed_ms": 90000.0})
        gd = cbr.compute_gpu_delta(ra, rb)
        self.assertEqual(gd["black_loading_ms"], {"a": 100000.0, "b": 50000.0, "delta": -50000.0, "pct": -50.0})

        # Explicit falsy zero must NOT fall through to derived value, and snake_case works
        ra2 = make_report()
        rb2 = make_report()
        ra2["milestones"].extend([
            {"milestone": "FIRST_PRESENT", "elapsed_ms": 1000},
            {"milestone": "FIRST_VISIBLE_FRAME", "elapsed_ms": 5000},
        ])
        rb2["milestones"].extend([
            {"milestone": "FIRST_PRESENT", "elapsed_ms": 1000},
            {"milestone": "FIRST_VISIBLE_FRAME", "elapsed_ms": 5000},
        ])
        ra2["black_loading_ms"] = 0
        rb2["summary"] = {"black_loading_ms": 250}
        gd2 = cbr.compute_gpu_delta(ra2, rb2)
        self.assertEqual(gd2["first_visible_frame_ms"], {"a": 5000, "b": 5000, "delta": 0, "pct": 0.0})
        self.assertEqual(gd2["black_loading_ms"], {"a": 0, "b": 250, "delta": 250, "pct": None})

    def test_visible_frame_and_black_loading_reject_invalid_values(self):
        ra = make_report()
        rb = make_report()
        ra["summary"] = {
            "firstVisibleFrameMs": True,
            "blackLoadingMs": float("nan"),
        }
        rb["summary"] = {
            "firstVisibleFrameMs": r"C:\Users\secret\frame.png",
            "blackLoadingMs": float("inf"),
        }
        gd = cbr.compute_gpu_delta(ra, rb)
        self.assertNotIn("first_visible_frame_ms", gd)
        self.assertNotIn("black_loading_ms", gd)

    def test_preserves_1023_progress_counters_alongside_adapter_and_visible_frame(self):
        ra = make_report_v2()
        rb = make_report_v2()
        for r in (ra, rb):
            r["gpu_work"] = {"draws_emitted": 500, "draws_no_shader_module": 10}
            r["memory"] = {"cpu_window": {"alias_slots": 12, "alias_conflicts": 0}}
            r["first_present"]["elapsed_ms"] = 2000
            r["milestones"].extend([
                {"milestone": "FIRST_PRESENT", "elapsed_ms": 2000, "import_seq": 600},
                {"milestone": "FIRST_VISIBLE_FRAME", "elapsed_ms": 6000, "import_seq": 900},
            ])
        ra["gpu_adapter"] = "nvidia / turing"
        rb["gpu_adapter"] = "intel / gen-12lp"
        rb["gpu_work"]["draws_no_shader_module"] = 200  # regression in #1023 counter

        c = cbr.compare(ra, rb, self.tol)
        self.assertIn("gpu_work.draws_no_shader_module", {e["field"] for e in c.failures()})
        result = cbr.to_result(c)
        self.assertEqual(result["verdict"], "fail")
        self.assertTrue(result["gpu_adapter"]["mismatch"])
        self.assertEqual(result["gpu_delta"]["black_loading_ms"], {"a": 4000, "b": 4000, "delta": 0, "pct": 0.0})


class TelemetryDeltaTests(unittest.TestCase):
    tol = cbr.Tolerances()

    def test_full_chrome_boot_telemetry_diff_json_text_and_markdown(self):
        ra = make_browser_report()
        rb = make_browser_report()
        ra.update({
            "first_visible_frame_s": 50.4,
            "black_loading_s": 42.0,
            "js_heap_used_mb": 1220.0,
            "js_heap_total_mb": 1500.0,
            "audio_ring_attached": False,
            "audio_frames_drained": 0,
        })
        rb.update({
            "first_visible_frame_s": 12.2,
            "black_loading_s": 4.0,
            "js_heap_used_mb": 270.0,
            "js_heap_total_mb": 512.0,
            "audio_ring_attached": True,
            "audio_frames_drained": 48000,
        })

        c = cbr.compare(ra, rb, self.tol)
        self.assertEqual(c.failures(), [])
        result = cbr.to_result(c)
        self.assertEqual(result["verdict"], "pass")
        self.assertIn("telemetry_delta", result)
        td = result["telemetry_delta"]

        self.assertEqual(
            td["first_visible_frame_s"],
            {
                "a": 50.4,
                "b": 12.2,
                "delta": -38.2,
                "pct": -75.79,
                "formatted_delta": "-38.2s",
                "better": "lower",
                "status": "improved",
            },
        )
        self.assertEqual(
            td["black_loading_s"],
            {
                "a": 42.0,
                "b": 4.0,
                "delta": -38.0,
                "pct": -90.48,
                "formatted_delta": "-38.0s",
                "better": "lower",
                "status": "improved",
            },
        )
        self.assertEqual(
            td["js_heap_used_mb"],
            {
                "a": 1220.0,
                "b": 270.0,
                "delta": -950.0,
                "pct": -77.87,
                "formatted_delta": "-950.0 MB",
                "better": "lower",
                "status": "improved",
            },
        )
        self.assertEqual(
            td["js_heap_total_mb"],
            {
                "a": 1500.0,
                "b": 512.0,
                "delta": -988.0,
                "pct": -65.87,
                "formatted_delta": "-988.0 MB",
                "better": "lower",
                "status": "improved",
            },
        )
        self.assertEqual(
            td["audio_ring_attached"],
            {
                "a": False,
                "b": True,
                "delta": "false -> true",
                "changed": True,
                "transition": "false -> true",
                "better": "true",
                "status": "improved",
            },
        )
        self.assertEqual(
            td["audio_frames_drained"],
            {
                "a": 0,
                "b": 48000,
                "delta": 48000,
                "pct": None,
                "formatted_delta": "+48000",
                "better": "higher",
                "status": "improved",
            },
        )

        text = cbr.format_text(c)
        self.assertIn("[telemetry_delta]", text)
        self.assertIn("first_visible_frame_s: 50.4s -> 12.2s (-38.2s, -75.8%, improved)", text)
        self.assertIn("black_loading_s: 42.0s -> 4.0s (-38.0s, -90.5%, improved)", text)
        self.assertIn("js_heap_used_mb: 1220.0 MB -> 270.0 MB (-950.0 MB, -77.9%, improved)", text)
        self.assertIn("js_heap_total_mb: 1500.0 MB -> 512.0 MB (-988.0 MB, -65.9%, improved)", text)
        self.assertIn("audio_ring_attached: false -> true (improved)", text)
        self.assertIn("audio_frames_drained: 0 -> 48000 (+48000, improved)", text)

        md = cbr.format_markdown(c)
        self.assertIn("## Boot Report Comparison — PASS", md)
        self.assertIn("### Telemetry Delta", md)
        self.assertIn("| `first_visible_frame_s` | 50.4s | 12.2s | -38.2s | -75.8% | improved |", md)
        self.assertIn("| `js_heap_used_mb` | 1220.0 MB | 270.0 MB | -950.0 MB | -77.9% | improved |", md)
        self.assertIn("| `audio_ring_attached` | false | true | false -> true | — | improved |", md)
        self.assertIn("| `audio_frames_drained` | 0 | 48000 | +48000 | — | improved |", md)

    def test_regressions_and_unchanged_directionality(self):
        ra = make_browser_report()
        rb = make_browser_report()
        ra.update({
            "first_visible_frame_s": 12.2,
            "black_loading_s": 4.0,
            "js_heap_used_mb": 270.0,
            "js_heap_total_mb": 512.0,
            "audio_ring_attached": True,
            "audio_frames_drained": 48000,
        })
        rb.update({
            "first_visible_frame_s": 50.4,
            "black_loading_s": 4.0,
            "js_heap_used_mb": 1220.0,
            "js_heap_total_mb": 512.0,
            "audio_ring_attached": False,
            "audio_frames_drained": 12000,
        })

        td = cbr.compute_telemetry_delta(ra, rb)
        self.assertEqual(td["first_visible_frame_s"]["status"], "regressed")
        self.assertEqual(td["first_visible_frame_s"]["formatted_delta"], "+38.2s")
        self.assertEqual(td["black_loading_s"]["status"], "unchanged")
        self.assertEqual(td["black_loading_s"]["formatted_delta"], "0.0s")
        self.assertEqual(td["js_heap_used_mb"]["status"], "regressed")
        self.assertEqual(td["js_heap_used_mb"]["formatted_delta"], "+950.0 MB")
        self.assertEqual(td["js_heap_total_mb"]["status"], "unchanged")
        self.assertEqual(td["audio_ring_attached"]["status"], "regressed")
        self.assertEqual(td["audio_ring_attached"]["transition"], "true -> false")
        self.assertEqual(td["audio_frames_drained"]["status"], "regressed")
        self.assertEqual(td["audio_frames_drained"]["formatted_delta"], "-36000")
        self.assertEqual(td["audio_frames_drained"]["pct"], -75.0)

        # Unchanged boolean transitions (both true and both false)
        rc = copy.deepcopy(ra)
        td_same = cbr.compute_telemetry_delta(ra, rc)
        self.assertEqual(td_same["audio_ring_attached"]["status"], "unchanged")
        self.assertFalse(td_same["audio_ring_attached"]["changed"])
        self.assertEqual(td_same["audio_ring_attached"]["transition"], "true -> true")
        self.assertEqual(td_same["audio_frames_drained"]["status"], "unchanged")
        self.assertEqual(td_same["audio_frames_drained"]["formatted_delta"], "0")

    def test_nested_telemetry_memory_audio_and_unit_conversions(self):
        ra = make_browser_report(first_present_ms=10000.0)
        ra["milestones"].append({"milestone": "FIRST_VISIBLE_FRAME", "elapsed_ms": 50250.0})
        # black_loading_s derived as (50.25 - 10.0) = 40.25s
        ra["memory"] = {
            "cpu_window": {"alias_slots": 8, "alias_conflicts": 0},
            "js_heap_used_mb": 1024.0,
            "totalJSHeapSize": 1536 * 1024 * 1024,
        }
        ra["audio"] = {"ring_attached": False, "frames_drained": 0}

        rb = make_browser_report(first_present_ms=10000.0)
        rb["telemetry"] = {
            "first_visible_frame_s": 12.0,
            "black_loading_s": 2.0,
            "memory": {
                "usedJSHeapSize": 256 * 1024 * 1024,
                "js_heap_total_mb": 512.0,
            },
            "audio": {
                "ring": {"attached": True, "frames_drained": 96000.0},
            },
        }

        td = cbr.compute_telemetry_delta(ra, rb)
        self.assertEqual(td["first_visible_frame_s"]["a"], 50.25)
        self.assertEqual(td["first_visible_frame_s"]["b"], 12.0)
        self.assertEqual(td["first_visible_frame_s"]["formatted_delta"], "-38.25s")
        self.assertEqual(td["black_loading_s"]["a"], 40.25)
        self.assertEqual(td["black_loading_s"]["b"], 2.0)
        self.assertEqual(td["black_loading_s"]["formatted_delta"], "-38.25s")
        self.assertEqual(td["js_heap_used_mb"]["a"], 1024.0)
        self.assertEqual(td["js_heap_used_mb"]["b"], 256.0)
        self.assertEqual(td["js_heap_used_mb"]["formatted_delta"], "-768.0 MB")
        self.assertEqual(td["js_heap_total_mb"]["a"], 1536.0)
        self.assertEqual(td["js_heap_total_mb"]["b"], 512.0)
        self.assertEqual(td["js_heap_total_mb"]["formatted_delta"], "-1024.0 MB")
        self.assertEqual(td["audio_ring_attached"]["transition"], "false -> true")
        self.assertEqual(td["audio_frames_drained"]["a"], 0)
        self.assertEqual(td["audio_frames_drained"]["b"], 96000)
        self.assertEqual(td["audio_frames_drained"]["formatted_delta"], "+96000")

    def test_graceful_degradation_for_older_and_partial_reports(self):
        # Older native reports (even with XMem memory.cpu_window) omit telemetry_delta completely
        old_a = make_report_v2()
        old_b = make_report_v2()
        old_a["memory"] = {"cpu_window": {"alias_slots": 10, "alias_conflicts": 0}}
        old_b["memory"] = {"cpu_window": {"alias_slots": 10, "alias_conflicts": 0}}
        c_old = cbr.compare(old_a, old_b, self.tol)
        self.assertEqual(c_old.telemetry_delta, {})
        self.assertNotIn("telemetry_delta", cbr.to_result(c_old))
        self.assertNotIn("[telemetry_delta]", cbr.format_text(c_old))
        self.assertNotIn("### Telemetry Delta", cbr.format_markdown(c_old))

        # Older baseline (no T61 fields) vs newer candidate (has T61 fields)
        base = make_browser_report()
        cand = make_browser_report()
        cand["telemetry"] = {
            "js_heap_used_mb": 270.0,
            "audio_ring_attached": True,
            "audio_frames_drained": 48000,
        }
        c_added = cbr.compare(base, cand, self.tol)
        td = cbr.to_result(c_added)["telemetry_delta"]
        self.assertEqual(
            td["js_heap_used_mb"],
            {
                "a": None,
                "b": 270.0,
                "delta": None,
                "pct": None,
                "formatted_delta": None,
                "better": "lower",
                "status": "added",
            },
        )
        self.assertEqual(
            td["audio_ring_attached"],
            {
                "a": None,
                "b": True,
                "delta": None,
                "changed": None,
                "transition": None,
                "better": "true",
                "status": "added",
            },
        )
        self.assertEqual(td["audio_frames_drained"]["status"], "added")
        self.assertIsNone(td["audio_frames_drained"]["delta"])
        self.assertNotIn("first_visible_frame_s", td)
        self.assertNotIn("black_loading_s", td)
        self.assertNotIn("js_heap_total_mb", td)

        text_added = cbr.format_text(c_added)
        self.assertIn("js_heap_used_mb: <missing> -> 270.0 MB (n/a, added)", text_added)
        self.assertIn("audio_ring_attached: <missing> -> true (n/a, added)", text_added)
        self.assertIn("audio_frames_drained: <missing> -> 48000 (n/a, added)", text_added)

        md_added = cbr.format_markdown(c_added)
        self.assertIn("| `js_heap_used_mb` | — | 270.0 MB | — | — | added |", md_added)
        self.assertIn("| `audio_ring_attached` | — | true | — | — | added |", md_added)

        # Reverse direction: present in baseline, missing in candidate -> removed
        td_removed = cbr.compute_telemetry_delta(cand, base)
        self.assertEqual(td_removed["js_heap_used_mb"]["status"], "removed")
        self.assertEqual(td_removed["js_heap_used_mb"]["a"], 270.0)
        self.assertIsNone(td_removed["js_heap_used_mb"]["b"])
        self.assertEqual(td_removed["audio_ring_attached"]["status"], "removed")
        self.assertEqual(td_removed["audio_ring_attached"]["a"], True)
        self.assertIsNone(td_removed["audio_ring_attached"]["b"])

    def test_falsy_zero_and_false_are_preserved_without_fallthrough(self):
        ra = make_browser_report(first_present_ms=1000.0)
        ra["first_visible_frame_s"] = 0.0
        ra["first_visible_frame_ms"] = 99999.0
        ra["black_loading_s"] = 0.0
        ra["black_loading_ms"] = 88888.0
        ra["js_heap_used_mb"] = 0.0
        ra["memory"] = {"js_heap_used_mb": 777.0, "js_heap_total_mb": 666.0}
        ra["js_heap_total_mb"] = 0.0
        ra["audio_ring_attached"] = False
        ra["audio"] = {"attached": True, "frames_drained": 555}
        ra["audio_frames_drained"] = 0

        extracted = cbr.extract_telemetry_metrics(ra)
        self.assertEqual(extracted["first_visible_frame_s"], 0.0)
        self.assertEqual(extracted["black_loading_s"], 0.0)
        self.assertEqual(extracted["js_heap_used_mb"], 0.0)
        self.assertEqual(extracted["js_heap_total_mb"], 0.0)
        self.assertIs(extracted["audio_ring_attached"], False)
        self.assertEqual(extracted["audio_frames_drained"], 0)

    def test_sanitization_rejects_invalid_types_negatives_non_finite_and_paths(self):
        ra = make_report()
        rb = make_report()
        ra["telemetry"] = {
            "first_visible_frame_s": True,
            "black_loading_s": -5.0,
            "js_heap_used_mb": float("nan"),
            "js_heap_total_mb": float("inf"),
            "audio_ring_attached": "true",
            "audio_frames_drained": r"C:\Users\secret\audio.wav",
        }
        rb["telemetry"] = {
            "first_visible_frame_s": "/home/alice/frame.png",
            "black_loading_s": False,
            "js_heap_used_mb": -1.0,
            "js_heap_total_mb": "512",
            "audio_ring_attached": 1,
            "audio_frames_drained": -10,
        }
        c = cbr.compare(ra, rb, self.tol)
        self.assertEqual(c.telemetry_delta, {})
        result = cbr.to_result(c)
        self.assertNotIn("telemetry_delta", result)
        serialized = json.dumps(result)
        self.assertNotIn("secret", serialized)
        self.assertNotIn("alice", serialized)

    def test_cli_markdown_flag_and_mutually_exclusive_guard(self):
        ra = make_browser_report()
        rb = make_browser_report()
        ra["telemetry"] = {
            "first_visible_frame_s": 50.4,
            "js_heap_used_mb": 1220.0,
            "audio_ring_attached": False,
            "audio_frames_drained": 0,
        }
        rb["telemetry"] = {
            "first_visible_frame_s": 12.2,
            "js_heap_used_mb": 270.0,
            "audio_ring_attached": True,
            "audio_frames_drained": 48000,
        }
        with tempfile.TemporaryDirectory() as d:
            pa, pb = Path(d, "a.json"), Path(d, "b.json")
            pa.write_text(json.dumps(ra), encoding="utf-8")
            pb.write_text(json.dumps(rb), encoding="utf-8")

            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = cbr.main([str(pa), str(pb), "--markdown"])
            self.assertEqual(code, 0, err.getvalue())
            md = out.getvalue()
            self.assertIn("### Telemetry Delta", md)
            self.assertIn("-38.2s", md)
            self.assertIn("-950.0 MB", md)
            self.assertIn("+48000", md)

            # --json and --markdown together -> exit code 2
            out2, err2 = io.StringIO(), io.StringIO()
            with redirect_stdout(out2), redirect_stderr(err2):
                code2 = cbr.main([str(pa), str(pb), "--json", "--markdown"])
            self.assertEqual(code2, 2)
            self.assertIn("mutually exclusive", err2.getvalue())


class PresentFpsAvgAndMinRatioTests(unittest.TestCase):
    tol = cbr.Tolerances()

    def run_cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cbr.main([str(a) for a in args])
        return code, out.getvalue(), err.getvalue()

    def test_present_fps_avg_improvement_regression_and_unchanged_outputs(self):
        ra = make_browser_report()
        rb = make_browser_report()
        ra["present_fps_avg"] = 20.0
        rb["present_fps_avg"] = 30.0

        c_imp = cbr.compare(ra, rb, self.tol)
        self.assertEqual(c_imp.failures(), [])
        res_imp = cbr.to_result(c_imp)
        self.assertEqual(res_imp["verdict"], "pass")
        self.assertEqual(
            res_imp["telemetry_delta"]["present_fps_avg"],
            {
                "a": 20.0,
                "b": 30.0,
                "delta": 10.0,
                "pct": 50.0,
                "formatted_delta": "+10.0 fps",
                "better": "higher",
                "status": "improved",
            },
        )
        text_imp = cbr.format_text(c_imp)
        self.assertIn("[telemetry_delta]", text_imp)
        self.assertIn("present_fps_avg: 20.0 fps -> 30.0 fps (+10.0 fps, +50.0%, improved)", text_imp)
        md_imp = cbr.format_markdown(c_imp)
        self.assertIn("| `present_fps_avg` | 20.0 fps | 30.0 fps | +10.0 fps | +50.0% | improved |", md_imp)

        # Regression without --min-fps-ratio stays backwards-compatible (pass verdict)
        ra_reg = make_browser_report()
        rb_reg = make_browser_report()
        ra_reg["present_fps_avg"] = 30.0
        rb_reg["present_fps_avg"] = 24.0
        c_reg = cbr.compare(ra_reg, rb_reg, self.tol)
        self.assertEqual(c_reg.failures(), [])
        res_reg = cbr.to_result(c_reg)
        self.assertEqual(res_reg["verdict"], "pass")
        self.assertEqual(
            res_reg["telemetry_delta"]["present_fps_avg"],
            {
                "a": 30.0,
                "b": 24.0,
                "delta": -6.0,
                "pct": -20.0,
                "formatted_delta": "-6.0 fps",
                "better": "higher",
                "status": "regressed",
            },
        )
        self.assertIn(
            "present_fps_avg: 30.0 fps -> 24.0 fps (-6.0 fps, -20.0%, regressed)",
            cbr.format_text(c_reg),
        )

        # Unchanged and zero baseline
        ra_same = make_browser_report()
        rb_same = make_browser_report()
        ra_same["present_fps_avg"] = 30.0
        rb_same["present_fps_avg"] = 30.0
        td_same = cbr.compute_telemetry_delta(ra_same, rb_same)
        self.assertEqual(td_same["present_fps_avg"]["status"], "unchanged")
        self.assertEqual(td_same["present_fps_avg"]["formatted_delta"], "0.0 fps")
        self.assertEqual(td_same["present_fps_avg"]["pct"], 0.0)

        ra_zero = make_browser_report()
        rb_zero = make_browser_report()
        ra_zero["present_fps_avg"] = 0.0
        rb_zero["present_fps_avg"] = 30.0
        td_zero = cbr.compute_telemetry_delta(ra_zero, rb_zero)
        self.assertEqual(td_zero["present_fps_avg"]["status"], "improved")
        self.assertIsNone(td_zero["present_fps_avg"]["pct"])
        self.assertEqual(td_zero["present_fps_avg"]["formatted_delta"], "+30.0 fps")

    def test_present_fps_avg_extraction_sources_priority_and_falsy_zero(self):
        # Top-level wins over summary
        r1 = make_browser_report()
        r1["present_fps_avg"] = 29.876
        r1["summary"]["present_fps_avg"] = 12.0
        self.assertEqual(cbr.extract_present_fps_avg(r1), 29.88)

        # Top-level None falls through to summary.present_fps_avg
        r2 = make_browser_report()
        r2["present_fps_avg"] = None
        r2["summary"]["present_fps_avg"] = 24.5
        self.assertEqual(cbr.extract_present_fps_avg(r2), 24.5)

        # Top-level falsy 0.0 is preserved without falling through to summary
        r3 = make_browser_report()
        r3["present_fps_avg"] = 0.0
        r3["summary"]["present_fps_avg"] = 60.0
        self.assertEqual(cbr.extract_present_fps_avg(r3), 0.0)
        self.assertEqual(cbr.extract_telemetry_metrics(r3)["present_fps_avg"], 0.0)

        # Nested telemetry / counters / camelCase fallbacks
        r4 = make_report()
        r4["telemetry"] = {"presentFpsAvg": 27.25}
        self.assertEqual(cbr.extract_present_fps_avg(r4), 27.25)

        r5 = make_report()
        r5["summary"] = {"presentFpsAvg": 18.5}
        self.assertEqual(cbr.extract_present_fps_avg(r5), 18.5)

    def test_min_fps_ratio_pass_boundary_fail_and_ignore_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as d:
            pa, pb = Path(d, "a.json"), Path(d, "b.json")
            ra = make_browser_report()
            rb = make_browser_report()
            ra["present_fps_avg"] = 30.0
            rb["summary"]["present_fps_avg"] = 25.5  # exact 0.85 boundary (30 * 0.85 = 25.5)
            pa.write_text(json.dumps(ra), encoding="utf-8")
            pb.write_text(json.dumps(rb), encoding="utf-8")

            # Boundary pass -> exit 0
            code, out, err = self.run_cli(pa, pb, "--min-fps-ratio", "0.85", "--json")
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertEqual(payload["verdict"], "pass")
            self.assertEqual(payload["differences"], 0)
            self.assertEqual(payload["tolerances"]["min_fps_ratio"], 0.85)
            self.assertEqual(payload["telemetry_delta"]["present_fps_avg"]["a"], 30.0)
            self.assertEqual(payload["telemetry_delta"]["present_fps_avg"]["b"], 25.5)

            # Floating-point boundary: 3.0 * 0.1 = 0.30000000000000004 vs 0.3 -> pass
            ra_fp = make_browser_report()
            rb_fp = make_browser_report()
            ra_fp["present_fps_avg"] = 3.0
            rb_fp["present_fps_avg"] = 0.3
            pa.write_text(json.dumps(ra_fp), encoding="utf-8")
            pb.write_text(json.dumps(rb_fp), encoding="utf-8")
            code_fp, _, err_fp = self.run_cli(pa, pb, "--min-fps-ratio", "0.1")
            self.assertEqual(code_fp, 0, err_fp)

            # Regression below threshold (25.49 < 25.5) -> exit 1 in both text and JSON
            ra["present_fps_avg"] = 30.0
            rb = make_browser_report()
            rb["present_fps_avg"] = 25.49
            pa.write_text(json.dumps(ra), encoding="utf-8")
            pb.write_text(json.dumps(rb), encoding="utf-8")

            code_fail, out_json, err_json = self.run_cli(pa, pb, "--min-fps-ratio", "0.85", "--json")
            self.assertEqual(code_fail, 1, err_json)
            fail_payload = json.loads(out_json)
            self.assertEqual(fail_payload["verdict"], "fail")
            self.assertEqual(fail_payload["differences"], 1)
            self.assertEqual(fail_payload["tolerances"]["min_fps_ratio"], 0.85)
            self.assertEqual(
                fail_payload["entries"],
                [{"field": "telemetry.present_fps_avg", "a": 30.0, "b": 25.49, "mode": "min_fps_ratio", "ok": False}],
            )
            self.assertEqual(fail_payload["telemetry_delta"]["present_fps_avg"]["status"], "regressed")

            code_txt, out_txt, err_txt = self.run_cli(pa, pb, "--min-fps-ratio", "0.85")
            self.assertEqual(code_txt, 1, err_txt)
            self.assertIn("[telemetry]", out_txt)
            self.assertIn("DIFF telemetry.present_fps_avg: 30.0 != 25.49 (min_fps_ratio)", out_txt)
            self.assertIn("present_fps_avg: 30.0 fps -> 25.49 fps (-4.51 fps, -15.0%, regressed)", out_txt)
            self.assertIn("verdict: FAIL", out_txt)

            # Without --min-fps-ratio the same pair passes (backwards compatibility)
            code_default, out_default, err_default = self.run_cli(pa, pb, "--json")
            self.assertEqual(code_default, 0, err_default)
            default_payload = json.loads(out_default)
            self.assertEqual(default_payload["verdict"], "pass")
            self.assertNotIn("min_fps_ratio", default_payload["tolerances"])

            # Ignorable via --ignore telemetry.present_fps_avg
            code_ign, out_ign, err_ign = self.run_cli(
                pa, pb, "--min-fps-ratio", "0.85", "--ignore", "telemetry.present_fps_avg", "--json"
            )
            self.assertEqual(code_ign, 0, err_ign)
            ign_payload = json.loads(out_ign)
            self.assertEqual(ign_payload["verdict"], "pass")
            self.assertIn("present_fps_avg", ign_payload["telemetry_delta"])

            # Zero baseline with --min-fps-ratio -> pass (0.0 >= 0.0)
            ra_z, rb_z = make_browser_report(), make_browser_report()
            ra_z["present_fps_avg"] = 0.0
            rb_z["present_fps_avg"] = 0.0
            pa.write_text(json.dumps(ra_z), encoding="utf-8")
            pb.write_text(json.dumps(rb_z), encoding="utf-8")
            code_z, _, err_z = self.run_cli(pa, pb, "--min-fps-ratio", "0.85")
            self.assertEqual(code_z, 0, err_z)

    def test_min_fps_ratio_invalid_values_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            pa, pb = Path(d, "a.json"), Path(d, "b.json")
            pa.write_text(json.dumps(make_browser_report()), encoding="utf-8")
            pb.write_text(json.dumps(make_browser_report()), encoding="utf-8")
            for bad in ("0", "-0.5", "nan", "inf", "-inf"):
                code, _, err = self.run_cli(pa, pb, "--min-fps-ratio", bad)
                self.assertEqual(code, 2, f"expected exit 2 for --min-fps-ratio={bad}")
                self.assertIn("min-fps-ratio", err)

        for bad_val in (0, -0.1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                cbr.compare(make_browser_report(), make_browser_report(), self.tol, min_fps_ratio=bad_val)

    def test_missing_present_fps_avg_graceful_handling(self):
        # 1. Both older reports missing present_fps_avg, without --min-fps-ratio
        old_a, old_b = make_report(), make_report()
        c_old = cbr.compare(old_a, old_b, self.tol)
        res_old = cbr.to_result(c_old)
        self.assertEqual(
            set(res_old.keys()),
            {"verdict", "checked", "differences", "tolerances", "ignore", "entries"},
        )
        self.assertEqual(set(res_old["tolerances"].keys()), {"time_rel", "time_floor_ms", "count_rel"})

        # 2. Both older reports missing present_fps_avg, WITH --min-fps-ratio -> graceful pass + warning
        with tempfile.TemporaryDirectory() as d:
            pa, pb = Path(d, "a.json"), Path(d, "b.json")
            pa.write_text(json.dumps(old_a), encoding="utf-8")
            pb.write_text(json.dumps(old_b), encoding="utf-8")
            code, out, err = self.run_cli(pa, pb, "--min-fps-ratio", "0.85", "--json")
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertEqual(payload["verdict"], "pass")
            self.assertNotIn("telemetry_delta", payload)
            self.assertTrue(
                any("baseline and candidate" in w for w in payload.get("warnings", [])),
                payload.get("warnings"),
            )

            # 3. Baseline missing, candidate present -> added + warning, exit 0
            cand = copy.deepcopy(old_b)
            cand["present_fps_avg"] = 28.5
            pb.write_text(json.dumps(cand), encoding="utf-8")
            code_add, out_add, err_add = self.run_cli(pa, pb, "--min-fps-ratio", "0.85", "--json")
            self.assertEqual(code_add, 0, err_add)
            p_add = json.loads(out_add)
            self.assertEqual(p_add["telemetry_delta"]["present_fps_avg"]["status"], "added")
            self.assertIsNone(p_add["telemetry_delta"]["present_fps_avg"]["a"])
            self.assertEqual(p_add["telemetry_delta"]["present_fps_avg"]["b"], 28.5)
            self.assertTrue(any("missing from baseline" in w for w in p_add.get("warnings", [])))

            # 4. Baseline present, candidate missing -> removed + warning, exit 0
            code_rem, out_rem, err_rem = self.run_cli(pb, pa, "--min-fps-ratio", "0.85", "--json")
            self.assertEqual(code_rem, 0, err_rem)
            p_rem = json.loads(out_rem)
            self.assertEqual(p_rem["telemetry_delta"]["present_fps_avg"]["status"], "removed")
            self.assertEqual(p_rem["telemetry_delta"]["present_fps_avg"]["a"], 28.5)
            self.assertIsNone(p_rem["telemetry_delta"]["present_fps_avg"]["b"])
            self.assertTrue(any("missing from candidate" in w for w in p_rem.get("warnings", [])))

    def test_present_fps_avg_sanitization_and_json_schema_stability_with_t61(self):
        # Reject host paths, bools, strings, negatives, NaN, Inf
        ra_bad = make_report()
        rb_bad = make_report()
        ra_bad["present_fps_avg"] = r"C:\Users\secret\fps.json"
        ra_bad["summary"] = {"present_fps_avg": float("nan")}
        rb_bad["present_fps_avg"] = "/home/alice/boot-report.json"
        rb_bad["summary"] = {"present_fps_avg": -5.0}
        c_bad = cbr.compare(ra_bad, rb_bad, self.tol, min_fps_ratio=0.85)
        res_bad = cbr.to_result(c_bad)
        self.assertEqual(res_bad["verdict"], "pass")
        self.assertNotIn("telemetry_delta", res_bad)
        for rendered in (json.dumps(res_bad), cbr.format_text(c_bad), cbr.format_markdown(c_bad)):
            self.assertNotIn("secret", rendered)
            self.assertNotIn("alice", rendered)

        for invalid in (None, True, False, "30.0", -1.0, float("nan"), float("inf"), r"D:\game\a.rpf"):
            r = make_report()
            r["present_fps_avg"] = invalid
            self.assertIsNone(cbr.extract_present_fps_avg(r), f"expected None for {invalid!r}")

        # Invalid top-level falls through cleanly to valid summary.present_fps_avg
        r_fall = make_report()
        r_fall["present_fps_avg"] = r"C:\Users\secret\fps.json"
        r_fall["summary"] = {"present_fps_avg": 29.5}
        self.assertEqual(cbr.extract_present_fps_avg(r_fall), 29.5)

        # Full T61 + T73 report JSON schema stability
        ra = make_browser_report()
        rb = make_browser_report()
        ra.update({
            "first_visible_frame_s": 18.0,
            "black_loading_s": 6.0,
            "js_heap_used_mb": 320.0,
            "js_heap_total_mb": 512.0,
            "audio_ring_attached": True,
            "audio_frames_drained": 48000,
            "present_fps_avg": 28.0,
        })
        rb.update({
            "first_visible_frame_s": 14.0,
            "black_loading_s": 4.0,
            "js_heap_used_mb": 290.0,
            "js_heap_total_mb": 512.0,
            "audio_ring_attached": True,
            "audio_frames_drained": 64000,
            "present_fps_avg": 30.0,
        })
        c_full = cbr.compare(ra, rb, self.tol)
        res_full = cbr.to_result(c_full)
        self.assertEqual(
            set(res_full.keys()),
            {"verdict", "checked", "differences", "tolerances", "ignore", "entries", "gpu_delta", "telemetry_delta"},
        )
        self.assertEqual(
            tuple(res_full["telemetry_delta"].keys()),
            cbr.TELEMETRY_DELTA_METRICS,
        )
        self.assertEqual(
            set(res_full["telemetry_delta"]["present_fps_avg"].keys()),
            set(res_full["telemetry_delta"]["first_visible_frame_s"].keys()),
        )


class AudioCompareTests(unittest.TestCase):
    def audio(self, **over):
        a = {"render_frames": 4800, "non_silent_fraction": 0.5, "xapu_decoded_packets": 9,
             "non_silent_frames_per_window": [2400, 0]}
        a.update(over)
        r = make_report()
        r["audio"] = a
        return r

    def test_within_tolerance_passes(self):
        c = cbr.compare(self.audio(), self.audio(render_frames=4801), cbr.Tolerances())
        self.assertEqual(c.failures(), [])

    def test_window_and_total_differences_fail(self):
        b = self.audio(xapu_decoded_packets=0, non_silent_frames_per_window=[0, 0, 5])
        fields = {e["field"] for e in cbr.compare(self.audio(), b, cbr.Tolerances()).failures()}
        self.assertIn("audio.xapu_decoded_packets", fields)
        self.assertIn("audio.windows", fields)
        self.assertIn("audio.non_silent_frames_per_window[0]", fields)

    def test_one_sided_audio_is_skipped(self):
        c = cbr.compare(self.audio(), make_report(), cbr.Tolerances())
        self.assertFalse([e for e in c.failures() if e["field"].startswith("audio")])


if __name__ == "__main__":
    unittest.main()
