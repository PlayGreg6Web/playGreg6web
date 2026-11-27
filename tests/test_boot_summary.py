import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import boot_summary as bs  # noqa: E402


def base():
    return {
        "status": "limit_reached", "phase": "entry", "elapsed_ms": 90000,
        "message": "run bound reached",
        "counters": {"dispatches": 1234567},
        "milestones": [{"milestone": "PGIM_LOAD", "elapsed_ms": 200}, {"milestone": "FIRST_PRESENT", "elapsed_ms": 3400}],
        "present_samples_total": 5,
        "present_samples": [{"wall_ms": t} for t in (0, 33, 66, 100, 133, 400)],
        "screen_timeline": [{"start_time": 0, "end_time": 4000, "presents": 5, "draws_per_frame": 37}],
        "gpu_work": {"draws_no_shader_module": 0,
                     "drops_by_program": {"0x1": 5, "0x2": 50, "0x3": 7, "0x4": 1, "0x5": 9, "0x6": 3, "0x7": 0}},
        "native_executor": {"device_errors": 2, "shader_pack_hits": 10, "shader_pack_misses": 3, "frames_submitted": 5},
        "audio": {"xapu_decoded_packets": 152, "xapu_failed_packets": 0, "xapu_commands": 168, "non_silent_fraction": 0.5},
        "audio_files": {"audio_opens": 4, "audio_open_ok": 3, "audio_open_failed": 1},
        "audio_out": {"device_open": True, "frames_played": 100, "underruns": 7, "error": ""},
        "bink": {"binding": "shim", "total_calls": 9, "exports": [{"name": "BinkOpen", "calls": 9}]},
        "memory_budget": {"host_private_bytes": 1048576, "gpu_visible_bytes": 0,
                          "guest_arena": {"arena_peak_committed": 3 * 1024 ** 3}},
        "pipeline_prewarm": {"prewarmed": 8, "hit_before_use": 6, "wasted": 2, "failed": 0},
    }


def browser_report():
    """Shaped like a browser boot-report.json (sparse present samples carrying present_count)."""
    samples = []
    for i in range(41):  # 500 ms apart, 20 presents each = 40 fps; 10 draws/present from sample 20 on
        draws = 0 if i < 20 else (i - 20) * 200
        samples.append({"index": i, "wall_ms": 1000 + i * 500.0, "present_count": 1 + i * 20,
                        "counters": {"draws_real_pack": draws}})
    return {
        "schema": "playgta6-boot-report", "runner": "browser", "preset": "long", "status": "fault",
        "phase": "FIRST_VISIBLE_FRAME", "exit_code": 0, "elapsed_ms": 30000.0,
        "message": "[trap] INDIRECT_CALL_MISS at 0x0 (unknown_target, (no lifted function))",
        "fault": {"kind": "WASM_TRAP", "trap_kind": "WASM_TRAP", "va": "0x0", "exception_code": None,
                  "detail": "[trap] INDIRECT_CALL_MISS at 0x0 (unknown_target)"},
        "counters": {"presents": 801, "import_calls": 0},
        "milestones": [{"milestone": "FIRST_PRESENT", "elapsed_ms": 1000.0}],
        "presents": samples,
        "present_fps_avg": 39.11, "fps_steady_avg": 37.45, "fps_p50_window": 30.03, "fps_max_window": 67.96,
        "windows_ge_30fps_pct": 72.3, "first_visible_frame_s": 7.1, "black_loading_s": 5.0,
        "audio": {"ring_attached": True, "frames_drained": 16922400, "context_state": "running",
                  "output_state": "active", "sample_rate": 48000},
    }


class BootSummaryTest(unittest.TestCase):
    def test_device_error_kinds(self):
        b = base()
        b["native_executor"]["device_error_kinds"] = {"total": 2, "other": 0, "kinds": [
            {"count": 2, "opcode": 31, "text": "[Invalid BindGroupLayout (unlabeled)] is invalid"}]}
        out = bs.summarize(b)
        self.assertIn("gpu error kind: 2 x op 31: [Invalid BindGroupLayout (unlabeled)] is invalid", out)

    def test_limit_run(self):
        out = bs.summarize(base())
        self.assertIn("ended: limit", out)
        self.assertIn("device_errors=2", out)
        self.assertNotIn("gpu error kind:", out)
        self.assertIn("10 hits, 3 misses", out)
        self.assertIn("0x2=50, 0x5=9, 0x3=7, 0x1=5, 0x6=3", out)
        self.assertNotIn("0x7", out)
        self.assertIn("3 ok, 1 failed", out)
        self.assertIn("7 underruns", out)
        self.assertIn("BinkOpen=9", out)
        self.assertIn("arena peak 3.0 GiB", out)
        self.assertIn("6 hit before use", out)
        self.assertIn("fps:", out)

    def test_frame_times_printed_per_phase(self):
        r = base()
        r["frame_times"] = {"phases": [
            {"name": "all", "frames": 1000, "mean_ms": 17.2, "p50_ms": 16.7, "p99_ms": 33.3, "p999_ms": 100.0,
             "max_ms": 120.0, "low1_fps": 30.0, "low01_fps": 10.0, "missed_vblanks": 42, "late_frames": 30},
            {"name": "draws_100_999", "frames": 900, "mean_ms": 16.9, "p50_ms": 16.7, "p99_ms": 20.0, "p999_ms": 40.0,
             "max_ms": 60.0, "low1_fps": 50.0, "low01_fps": 25.0, "missed_vblanks": 2, "late_frames": 2}]}
        out = bs.summarize(r)
        self.assertIn("frame times (whole run, ms):", out)
        self.assertRegex(out, r"all\s+1,000 frames.*p99 33\.3.*low1% 30 fps.*missed vblanks 42 \(30 late frames\)")
        self.assertIn("draws_100_999", out)
        s = bs.frame_time_summary(r)
        self.assertEqual((s["low1_fps"], s["missed_vblanks"]), (30.0, 42.0))
        self.assertAlmostEqual(s["late_pct"], 3.0)
        self.assertNotIn("frame times", bs.summarize(base() | {"frame_times": {"phases": []}}))

    def test_gpu_phase_tables_printed(self):
        r = base()
        r["gpu_drop_table"] = {"entries": [{"hash": "0x1", "draws": 5, "dispatches": 0}], "other": {},
                               "phases": [{"name": "draws_0", "draws": 5, "dispatches": 0,
                                           "entries": [{"hash": "0x0000000000000001", "draws": 5, "dispatches": 0}]}]}
        r["binding_fallbacks"] = {"entries": [{"hash": "0x2", "draws": 9, "dispatches": 1}], "other": {},
                                  "phases": [{"name": "draws_100_999", "draws": 9, "dispatches": 1,
                                              "entries": [{"hash": "0x0000000000000002", "draws": 9, "dispatches": 1}]}]}
        out = bs.summarize(r)
        self.assertIn("gpu drops (whole run 5", out)
        self.assertIn("binding fallbacks (whole run 10", out)
        self.assertRegex(out, r"draws_100_999\s+9 draws\s+1 dispatches\s+0x0000000000000002 x10")
        self.assertNotIn("binding fallbacks", bs.summarize(base()))

    def test_missing_sections_skipped(self):
        out = bs.summarize({"status": "limit_reached", "elapsed_ms": 1000})
        self.assertIn("gpu errors: none", out)
        self.assertNotIn("bink", out)
        self.assertNotIn("prewarm", out)

    def test_fault_with_hard_stop(self):
        r = base()
        r["status"] = "fault"
        r["fault_record"] = {"kind": "unimplemented", "elapsed_ms": 5000, "rip": "0x140001000", "target": "0x0",
                             "function": {"entry": "0x140001000", "offset": "0x10"}}
        r["message"] = "com bridge: unimplemented xgameruntime.XStore {00000000-0000-0000-0000-000000000000} slot 66 (XStoreRegisterGameLicenseChanged)"
        md = "## 1. XStore (`xgameruntime.XStore`)\n\n| Slot | Function | Today | Set by |\n|---|---|---|---|\n| 66 / 67 | Reg | Trapping stub | #235 |\n"
        out = bs.summarize(r, None, md)
        self.assertIn("FAULT unimplemented", out)
        self.assertIn("slot 66", out)
        self.assertIn("hard stop: Reg -> Trapping stub", out)

    def test_structured_slot_and_identity(self):
        r = base()
        r["status"] = "fault"
        r["build"] = {"source_commit": "abcdef0123456789", "dirty": True, "profile": "Release", "executor": True,
                      "recomp_hash": "0011"}
        r["run"] = {"preset": "menu-flow", "name": "n1"}
        r["fault_record"] = {"kind": "unimplemented_slot", "elapsed_ms": 5000, "rip": "0x1", "target": "0x0",
                             "unimplemented_slot": {"interface": "XSystem", "guid": "g", "slot": 7,
                                                    "name": "XSystemGetAppSpecificDeviceId"}}
        md = "## 1. XSystem\n\n| Slot | Function | Today | Set by |\n|---|---|---|---|\n| 7 | XSystemGetAppSpecificDeviceId | Trapping stub | #1 |\n"
        out = bs.summarize(r, None, md)
        self.assertIn("build: abcdef0123+dirty, Release, executor, recomp 0011", out)
        self.assertIn("launch: preset menu-flow, name n1", out)
        self.assertIn("com slot: XSystem slot 7 XSystemGetAppSpecificDeviceId", out)
        self.assertIn("hard stop: XSystemGetAppSpecificDeviceId -> Trapping stub", out)

    def test_browser_report(self):
        r = browser_report()
        f = bs.fps_percentiles(r)
        self.assertAlmostEqual(f["p50"], 40.0)
        self.assertAlmostEqual(f["p1"], 40.0)
        out = bs.summarize(r)
        self.assertIn("ended: FAULT WASM_TRAP (recompiler gap)", out)
        self.assertNotIn("no fault record", out)
        self.assertIn("rip 0x0", out)
        self.assertIn("fps: p1 40.0", out)
        self.assertIn("(40 samples)", out)
        self.assertIn("2 timeline segments", out)
        self.assertIn("10.0 draws/frame", out)
        self.assertIn("801 presents", out)
        self.assertIn("browser fps: avg 39.1", out)
        self.assertIn("16,922,400 frames drained", out)
        self.assertNotIn("xapu", out)

    def test_native_fps_is_one_present_per_sample(self):
        f = bs.fps_percentiles(base())
        self.assertAlmostEqual(f["p50"], 1000 / 33, places=1)

    def test_browser_run_dir_merges_telemetry(self):
        r = browser_report()
        r.pop("first_visible_frame_s")
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "boot-report.json").write_text(json.dumps(r), encoding="utf-8")
            (Path(d) / "telemetry.json").write_text(
                json.dumps({"summary": {"first_visible_frame_s": 9.5}, "headless": False}), encoding="utf-8")
            loaded = bs.load(d)
        self.assertEqual(loaded["first_visible_frame_s"], 9.5)
        self.assertIn("first visible frame 9.5s", bs.summarize(loaded))

    def test_compare_and_cli(self):
        a, b = base(), base()
        b["native_executor"]["device_errors"] = 0
        b["elapsed_ms"] = 100000
        text = bs.compare(a, b)
        self.assertIn("native_executor.device_errors", text)
        self.assertIn("-2", text)
        self.assertIn("+10,000", text)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "boot-report.json").write_text(json.dumps(a), encoding="utf-8")
            self.assertEqual(bs.main([d]), 0)
            self.assertEqual(bs.main([str(Path(d) / "missing.json")]), 2)

    def test_compare_marks_regressions_and_improvements(self):
        a, b = base(), base()
        a["frame_times"] = {"hitches": [{"at_ms": 1, "ms": 100}]}
        b["frame_times"] = {"hitches": [{"at_ms": 1, "ms": 100}, {"at_ms": 5, "ms": 90}]}
        b["audio_out"]["underruns"] = 3  # 7 -> 3 improves
        a["binding_fallbacks"] = {"phases": [{"name": "menu", "draws": 4, "dispatches": 0}],
                                  "entries": [{"hash": "0xAA", "draws": 4}]}
        b["binding_fallbacks"] = {"phases": [{"name": "menu", "draws": 9, "dispatches": 0}],
                                  "entries": [{"hash": "0xBB", "draws": 9}]}
        b["fault_record"] = {"kind": "ACCESS_VIOLATION", "rip": "0x1000", "elapsed_ms": 5,
                             "function": {"entry": "0x1000", "offset": "0x4"}}
        lines = bs.compare(a, b).splitlines()

        def row(k):
            return next(ln for ln in lines if ln.startswith(k))
        self.assertTrue(row("frame_times.hitches").endswith("REGRESSION"))
        self.assertTrue(row("audio_out.underruns ").endswith("improvement"))
        self.assertTrue(row("binding_fallbacks.phase.menu").endswith("REGRESSION"))
        text = "\n".join(lines)
        self.assertIn("[REGRESSION]", text)
        self.assertIn("sub_1000+0x4", text)
        self.assertIn("0xAA x4", text)
        self.assertIn("0xBB x9", text)

    def test_compare_identical_and_last(self):
        a = base()
        self.assertIn("no differences", bs.compare(a, a))
        with tempfile.TemporaryDirectory() as d:
            for i, name in enumerate(("r1", "r2")):
                rd = Path(d) / "pg6x" / "runs" / name
                rd.mkdir(parents=True)
                rep = base()
                rep["elapsed_ms"] = 1000 * (i + 1)
                (rd / "boot-report.json").write_text(json.dumps(rep), encoding="utf-8")
                os.utime(rd / "boot-report.json", (1000 + i, 1000 + i))
            self.assertEqual(bs.main(["--compare-last", "--root", d]), 0)
            self.assertEqual(bs.main(["--compare-last", "5", "--root", d]), 2)

    def test_underruns_next_to_frame_hitches(self):
        rep = {"audio_out": {"device_open": True, "frames_played": 1, "underruns": 2,
                             "underruns_guest_stalled": 1, "underruns_host_starved": 1,
                             "underrun_events": [
                                 {"at_ms": 5300, "frames_missing": 480, "cause": "guest_stalled", "ms_since_push": 380,
                                  "ring_fill_ms": 40, "present_index": 311},
                                 {"at_ms": 9000, "frames_missing": 96, "cause": "host_starved", "ms_since_push": 3,
                                  "ring_fill_ms": 40, "present_index": 600}]},
               "frame_times": {"hitches": [{"at_ms": 5400, "ms": 410}, {"at_ms": 20000, "ms": 150}]}}
        text = chr(10).join(bs.underrun_lines(rep))
        self.assertIn("1 guest_stalled, 1 host_starved", text)
        self.assertIn("<- frame hitch 410 ms ending at 5400 ms", text)
        self.assertNotIn("9000 ms  96 frames missing  host_starved  3 ms since push, ring 40 ms, present #600  <-", text)
        self.assertIn("frame hitch 150 ms ending at 20000 ms: no underrun", text)

    def test_browser_memory_stats_validation_errors_reflection_and_pipeline_dump(self):
        r = browser_report()
        r["counters"]["gpu_validation_errors"] = 4
        r["reflection_join_failures"] = 2
        r["gpu_memory"] = {
            "vramBytes": 512 * 1024 * 1024,
            "totalLiveGpuBytes": 256 * 1024 * 1024,
            "peakBufferBytes": 128 * 1024 * 1024,
            "peakTextureBytes": 192 * 1024 * 1024,
            "stagingPoolBytes": 16 * 1024 * 1024,
        }
        r["renderer_memory"] = {"wasmLinearMemoryBytes": 1024 * 1024 * 1024}
        r["pipeline_dump"] = {"dumped": 42, "render": 35, "compute": 7, "failed": 1}
        r["scripted_inputs_sent"] = 6

        out = bs.summarize(r)
        self.assertIn("gpu_validation_errors=4", out)
        self.assertIn("reflection_join_failures=2", out)
        self.assertIn("gpu memory:", out)
        self.assertIn("vram 512.0 MiB", out)
        self.assertIn("peak tex 192.0 MiB", out)
        self.assertIn("wasm 1.0 GiB", out)
        self.assertIn("pipeline dump: dumped=42, render=35, compute=7, failed=1", out)
        self.assertIn("scripted inputs: 6 sent", out)

        # memory_stats alias & browser --compare delta tracking
        r2 = browser_report()
        r2.pop("present_fps_avg")
        r2["counters"]["gpu_validation_errors"] = 0
        r2["reflection_join_failures"] = 0
        r2["memory_stats"] = {"vram_peak_mb": 300.5, "peak_texture_bytes": 64 * 1024 * 1024}
        r2["pipeline_dump"] = {"dumped": 50, "failed": 0}
        r2["scripted_inputs_sent"] = 10
        out2 = bs.summarize(r2)
        self.assertIn("first visible frame 7.1s", out2)
        self.assertIn("gpu errors: none", out2)
        self.assertIn("vram peak 300.5 MiB", out2)

        diff = bs.compare(r, r2)
        self.assertIn("counters.gpu_validation_errors", diff)
        self.assertIn("reflection_join_failures", diff)
        self.assertIn("gpu_memory.peak_texture_bytes", diff)
        self.assertIn("pipeline_dump.dumped", diff)
        self.assertIn("scripted_inputs_sent", diff)

        # Verify lower-is-better classification on reflection_join_failures,
        # black_loading_s, and first_visible_frame_s in --compare verdicts.
        r_worse = browser_report()
        r_worse["reflection_join_failures"] = 5
        r_worse["black_loading_s"] = 12.5
        r_worse["first_visible_frame_s"] = 19.0
        r_base = browser_report()
        r_base["reflection_join_failures"] = 1
        r_base["black_loading_s"] = 3.0
        r_base["first_visible_frame_s"] = 7.1
        reg_diff = bs.compare(r_base, r_worse)
        self.assertRegex(reg_diff, r"reflection_join_failures.*REGRESSION")
        self.assertRegex(reg_diff, r"black_loading_s.*REGRESSION")
        self.assertRegex(reg_diff, r"first_visible_frame_s.*REGRESSION")


if __name__ == "__main__":
    unittest.main()


class StopRowTests(unittest.TestCase):
    HS = ("## 1. Stores\n\n| Slot | Function | Today |\n|---|---|---|\n"
          "| XStore 3 | QueryLicense | Stub: owner decision |\n")

    @staticmethod
    def slot_fault(iface, slot, name):
        r = base()
        r["status"] = "fault"
        r["fault_record"] = {"kind": "unimplemented_slot", "elapsed_ms": 5000, "rip": "0x1", "target": "0x2",
                             "unimplemented_slot": {"interface": iface, "guid": "g", "slot": slot, "name": name}}
        return r

    def test_no_fault_has_no_stop(self):
        self.assertIsNone(bs.stop_info(base()))
        self.assertIn("stop: none", bs.summarize(base()))

    def test_slot_stop_row(self):
        r = self.slot_fault("xgameruntime.XStore", 3, "QueryLicense")
        info = bs.stop_info(r, self.HS)
        self.assertEqual((info["key"], info["present_index"]), ("XStore[3]", 5))
        line = bs.stop_line(info)
        self.assertIn("QueryLicense", line)
        self.assertIn("hard stop:", line)
        self.assertIn("5.0s into boot, present #5", line)
        self.assertIn("stop: QueryLicense", bs.summarize(r, None, self.HS))

    def test_unlisted_slot_and_plain_fault(self):
        self.assertEqual(bs.stop_info(self.slot_fault("xgameruntime.XNew", 9, None), self.HS)["hard_stop"], "unlisted")
        r = base()
        r["fault_record"] = {"kind": "access_violation", "elapsed_ms": 10, "target": "0xdead"}
        self.assertEqual(bs.stop_info(r)["key"], "access_violation@0xdead")

    def test_feed_records_new_stop_once(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            run = root / "run1"
            run.mkdir()
            info = bs.stop_info(self.slot_fault("xgameruntime.XNew", 9, "Foo"), self.HS)
            cmd = bs.feed_stop(run, info, self.HS)
            self.assertIn("surface_survey.py", cmd)
            seen = json.loads((root / "stops_seen.json").read_text(encoding="utf-8"))
            self.assertEqual([e["key"] for e in seen], ["XNew[9]"])
            self.assertIsNone(bs.feed_stop(run, info, self.HS))
            self.assertEqual(len(json.loads((root / "stops_seen.json").read_text(encoding="utf-8"))), 1)

    def test_feed_skips_listed_hard_stop_and_no_fault(self):
        with tempfile.TemporaryDirectory() as t:
            run = Path(t) / "run1"
            run.mkdir()
            info = bs.stop_info(self.slot_fault("xgameruntime.XStore", 3, "QueryLicense"), self.HS)
            self.assertIsNone(bs.feed_stop(run, info, self.HS))
            self.assertIsNone(bs.feed_stop(run, None))
            self.assertFalse((Path(t) / "stops_seen.json").exists())


class TimingFieldsTests(unittest.TestCase):
    @staticmethod
    def rep(p99_bucket="lt16", nulls=2, flush_ns=50_000_000):
        bk = {k: 0 for k in ("lt0.5", "lt1", "lt2", "lt4", "lt8", "lt16", "lt32", "ge32")}
        bk["lt0.5"] = 98
        bk[p99_bucket] += 2
        blk = {"count": 100, "max_overshoot_ms": 40.0 if p99_bucket == "ge32" else 12.0, "overshoot_ms_buckets": bk}
        r = base()
        r["guest_sleep"] = dict(blk)
        r["guest_timed_wait"] = {"timed_out": dict(blk), "signalled": {"count": 5}}
        r["bink"] = {"binding": "shim", "total_calls": 9, "lookup_null": nulls, "lookup_null_names": ["BinkFoo"],
                     "open_log": [{"path": "a"}]}
        r["gpu_work"]["flush_buffer_ns"] = flush_ns
        r["gpu_work"]["flush_uniform_ns"] = 10_000_000
        return r

    def test_summary_lines(self):
        out = bs.summarize(self.rep())
        self.assertIn("guest_sleep: 100 calls, overshoot p50 <=0.5 ms, p99 <=12.0 ms", out)
        self.assertIn("guest_timed_wait: 100 calls", out)
        self.assertIn("bink lookups: 2 null (BinkFoo), 1 opens logged", out)
        self.assertIn("gpu flush: flush_buffer 10.000 ms/frame, flush_uniform 2.000 ms/frame", out)

    def test_absent_fields_skipped(self):
        out = bs.summarize(base())
        self.assertNotIn("guest_sleep", out)
        self.assertNotIn("gpu flush", out)

    def test_compare_directions(self):
        out = bs.compare(self.rep(), self.rep(p99_bucket="ge32", nulls=5, flush_ns=10_000_000))
        rows = {ln.split()[0]: ln for ln in out.splitlines() if ln.startswith(("guest_", "bink.", "gpu_work.flush"))}
        self.assertIn("REGRESSION", rows["bink.lookup_null"])
        self.assertIn("improvement", rows["gpu_work.flush_buffer_ms_per_frame"])
        self.assertIn("REGRESSION", rows["guest_sleep.overshoot_p99_ms"])
        self.assertIn("REGRESSION", rows["guest_timed_wait.overshoot_p99_ms"])
