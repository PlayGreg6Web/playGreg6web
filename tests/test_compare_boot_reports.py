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


if __name__ == "__main__":
    unittest.main()
