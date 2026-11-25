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


if __name__ == "__main__":
    unittest.main()
