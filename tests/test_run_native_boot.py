import io
import json
import os
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import run_native_boot as rnb  # noqa: E402

STUB_HARNESS = textwrap.dedent(
    """
    import json, sys
    a = sys.argv[1:]
    opt = lambda n: a[a.index(n) + 1] if n in a else None
    json.dump({
        "schema": "playgta6-boot-report", "schema_version": 2, "status": "fault", "phase": "entry",
        "exit_code": None, "message": "stub", "elapsed_ms": 5,
        "milestones": [{"milestone": "ENTRY_CALL", "elapsed_ms": 1, "import_seq": 0}],
        "fault": {"kind": "unimplemented", "va": "0x10", "function": {"entry": "0x08"}},
        "backtrace": {"frames": [{"function": "0x08", "offset": "0x8"}]},
        "counters": {"import_calls": 3},
        "presents": [{"index": 0, "wall_ms": 2.0, "dispatches": 4, "import_calls": 3,
                      "counters": {"draws_ok": 7, "heap_live_blocks": 9}}],
        "first_present": {"import_seq": 3, "elapsed_ms": 2, "imports": []},
        "args": a,
    }, open(opt("--report"), "w"))
    open(opt("--summary"), "w").write("stub summary\\n")
    print("stub harness ran")
    sys.exit(9)
    """
)


def touch(path: Path, when: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("x")
    os.utime(path, (when, when))


class Fixture:
    def __init__(self, root: Path) -> None:
        self.repo = root / "repo"
        self.build = root / "build"
        self.exe = root / "game.exe"
        self.pkg = root / "pkg"
        self.pkg.mkdir()
        self.harness = root / "stub_harness.py"
        self.harness.write_text(STUB_HARNESS)
        self.recomp = self.repo / "target" / "release" / ("playgta6-recomp" + rnb.EXE_SUFFIX)
        self.manifest = self.build / "program" / "program-manifest.json"
        base = 1_000_000.0
        touch(self.repo / "Cargo.toml", base)
        self.repo.joinpath("Cargo.toml").write_text('[workspace]\nmembers = ["recomp"]\n')
        os.utime(self.repo / "Cargo.toml", (base, base))
        touch(self.repo / "recomp" / "src" / "main.rs", base)
        touch(self.repo / "runtime" / "a.cpp", base)
        touch(self.repo / "tools" / "build_native_boot.py", base)
        touch(self.exe, base)
        touch(self.recomp, base + 10)
        touch(self.manifest, base + 20)
        self.base = base

    def run(self, *argv, runner=None, env=None):
        calls = []

        def default_runner(cmd, cwd, log, timeout):
            calls.append(cmd)
            if log is None:
                return 0
            return rnb.run_command(cmd, cwd, log, timeout)

        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = rnb.main(
                ["--exe", str(self.exe), "--game-dir", str(self.pkg), "--build-dir", str(self.build),
                 "--harness", str(self.harness), *argv],
                runner=runner or default_runner, env=env or {}, repo_root=self.repo)
        return code, out.getvalue(), err.getvalue(), calls


class StalenessTests(unittest.TestCase):
    def test_nothing_stale_runs_only_the_harness(self):
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(Path(d))
            code, out, err, calls = fx.run("--name", "r1")
            self.assertEqual(code, 0, err)
            self.assertEqual(len(calls), 1)  # just the harness
            self.assertNotIn("[plan]", err)

    def test_newer_recompiler_source_rebuilds_recomp_then_program(self):
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(Path(d))
            touch(fx.repo / "recomp" / "src" / "main.rs", fx.base + 100)
            code, _, err, calls = fx.run("--name", "r1")
            self.assertEqual(code, 0, err)
            self.assertEqual(calls[0][:5], ["cargo", "build", "--release", "-p", "playgta6-recomp"])
            self.assertEqual(calls[1][1], "build-program")
            self.assertIn("--target", calls[1])
            self.assertIn(str(fx.pkg), calls[1])
            self.assertIn("[plan] program: the recompiler is rebuilt", err)

    def test_newer_exe_rebuilds_only_the_program(self):
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(Path(d))
            touch(fx.exe, fx.base + 500)
            _, _, err, calls = fx.run("--name", "r1")
            self.assertEqual([c[1] if len(c) > 1 else c[0] for c in calls[:-1]], ["build-program"])
            self.assertNotIn("[plan] recomp", err)

    def test_newer_shim_header_or_export_table_regenerates_the_program(self):
        # The recompiler binds imports from the runtime's shim headers and export tables; a new
        # pg6_gthunk_* registration must regenerate the program, not only rebuild the harness.
        for rel in ("runtime/os/xbox_scarlett_guest.h", "runtime/os/xbox_scarlett_guest.cpp",
                    "runtime/guest/guest_callbacks.h"):
            with self.subTest(rel), tempfile.TemporaryDirectory() as d:
                fx = Fixture(Path(d))
                touch(fx.repo / rel, fx.base + 700)
                _, _, err, calls = fx.run("--name", "r1")
                self.assertEqual([c[1] if len(c) > 1 else c[0] for c in calls[:-1]], ["build-program"])
                self.assertIn("[plan] program: the runtime shim headers or export tables", err)
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(Path(d))
            touch(fx.repo / "runtime" / "guest" / "boot.cpp", fx.base + 700)  # not scanned
            _, _, err, calls = fx.run("--name", "r1")
            self.assertEqual(len(calls), 1)

    def test_plan_for_harness_and_flags(self):
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(Path(d))
            # Without the --harness override a missing harness is planned with the documented command.
            plan, program, harness = rnb.make_plan(fx.repo, fx.build, fx.exe, fx.pkg, 3, False, None)
            self.assertEqual([s["stage"] for s in plan.steps], ["harness"])
            cmd = plan.steps[0]["cmd"]
            self.assertTrue(cmd[1].endswith("build_native_boot.py"))
            for flag in ("--keep-going", "--delete-ir", "--debug-info"):
                self.assertIn(flag, cmd)
            # Objects come from the machine-wide cache shared by all checkouts unless one is given.
            self.assertNotIn("--cache-dir", cmd)
            plan, *_ = rnb.make_plan(fx.repo, fx.build, fx.exe, fx.pkg, 3, False, None, cache_dir=fx.build / "c")
            cmd = plan.steps[0]["cmd"]
            self.assertEqual(cmd[cmd.index("--cache-dir") + 1], str((fx.build / "c").resolve()))
            touch(harness, fx.base + 1000)
            plan, *_ = rnb.make_plan(fx.repo, fx.build, fx.exe, fx.pkg, 3, False, None)
            self.assertEqual(plan.steps, [])
            touch(fx.repo / "runtime" / "a.cpp", fx.base + 2000)
            plan, *_ = rnb.make_plan(fx.repo, fx.build, fx.exe, fx.pkg, 3, False, None)
            self.assertEqual([s["stage"] for s in plan.steps], ["harness"])
            plan, *_ = rnb.make_plan(fx.repo, fx.build, fx.exe, fx.pkg, 3, True, None)
            self.assertEqual([s["stage"] for s in plan.steps], ["recomp", "program", "harness"])

    def test_trust_build_skips_staleness(self):
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(Path(d))
            touch(fx.exe, fx.base + 500)
            touch(fx.repo / "runtime" / "a.cpp", fx.base + 600)
            code, _, err, calls = fx.run("--name", "r1", "--no-build", "--trust-build")
            self.assertEqual(code, 0, err)
            self.assertEqual(len(calls), 1)
            self.assertNotIn("[plan]", err)

    def test_no_build_and_dry_run(self):
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(Path(d))
            touch(fx.exe, fx.base + 500)
            code, _, err, calls = fx.run("--name", "r1", "--no-build")
            self.assertEqual((code, calls), (2, []))
            self.assertIn("stale", err)
            code, out, _, calls = fx.run("--name", "r2", "--dry-run")
            self.assertEqual((code, calls), (0, []))
            self.assertEqual(json.loads(out)["plan"][0]["stage"], "program")

    def test_failing_build_stops(self):
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(Path(d))
            touch(fx.exe, fx.base + 500)
            code, _, err, _ = fx.run("--name", "r1", runner=lambda *a: 1)
            self.assertEqual(code, 2)
            self.assertIn("program failed", err)


class RunTests(unittest.TestCase):
    def test_paths_required_and_from_env(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            self.assertEqual(rnb.main([], env={}), 2)
        self.assertIn("PLAYGTA6_EXE", err.getvalue())
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(Path(d))
            env = {"PLAYGTA6_EXE": str(fx.exe), "PLAYGTA6_GAME_DIR": str(fx.pkg)}
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = rnb.main(["--build-dir", str(fx.build), "--harness", str(fx.harness), "--name", "e"],
                                env=env, repo_root=fx.repo)
            self.assertEqual(code, 0)

    def test_missing_paths_name_the_flags(self):
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            self.assertEqual(rnb.main(["--exe", "x"], env={}), 2)
        self.assertIn("--game-dir", err.getvalue())
        self.assertNotIn("--exe (", err.getvalue())
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            self.assertEqual(rnb.main([], env={}), 2)
        self.assertIn("--exe", err.getvalue())
        self.assertIn("--game-dir", err.getvalue())

    def test_rebuild_program_forces_only_program_and_harness(self):
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(Path(d))
            code, _, err, calls = fx.run("--rebuild-program", "--name", "rp", "--dry-run")
            self.assertEqual(code, 0, err)
            self.assertIn("[plan] program: forced (--rebuild-program)", err)
            self.assertNotIn("[plan] recomp", err)

    def test_shader_pack_hint(self):
        work = {"draws_issued": 100, "draws_emitted": 10, "draws_no_shader_module": 90}
        hint = rnb.shader_pack_hint({"gpu_work": work}, None)
        self.assertIn("no shader pack: 90 of 100 draws dropped", hint)
        self.assertIn("pack-rdna2", hint)
        self.assertIsNone(rnb.shader_pack_hint({"gpu_work": work}, Path("p.pack")))
        self.assertIsNone(rnb.shader_pack_hint({"gpu_work": {"draws_issued": 100, "draws_no_shader_module": 10}}, None))
        self.assertIsNone(rnb.shader_pack_hint({}, None))
        self.assertIn("5 of 6", rnb.shader_pack_hint({"gpu_work": {"draws_emitted": 1, "draws_no_shader_module": 5}}, None))

    def test_run_directory_and_digest(self):
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(Path(d))
            code, out, _, _ = fx.run("--preset", "start-screen", "--name", "s1", "--gpu-capture-frames", "3",
                                     "--gpu-capture-payloads", "--gpu-capture-thin-uploads", "4", "--shader-pack", "p.wgsl.pack",
                                     "--dump-pipelines", "work/dump", "--", "--survey")
            self.assertEqual(code, 0)
            run = fx.build / "runs" / "s1"
            for name in ("boot-report.json", "summary.txt", "boot.log", "input-script.txt", "command.json", "digest.txt"):
                self.assertTrue((run / name).is_file(), name)
            args = json.loads((run / "boot-report.json").read_text())["args"]
            for expect in ("--gpu-capture", "--gpu-capture-payloads", "--shader-pack", "--dump-pipelines", "--survey",
                           "--input-script"):
                self.assertIn(expect, args)
            self.assertEqual(args[args.index("--gpu-capture-frames") + 1], "3")
            self.assertEqual(args[args.index("--gpu-capture-thin-uploads") + 1], "4")
            self.assertEqual(args[args.index("--max-ms") + 1], "120000")
            script = (run / "input-script.txt").read_text().splitlines()
            self.assertEqual(script[1], "5000 A 100")
            self.assertEqual(script[2], "6000 A 100")
            self.assertIn("ENTRY_CALL", out)
            self.assertIn("presents: 1", out)
            self.assertIn("stop: unimplemented at 0x10", out)
            self.assertIn("draws_ok: 7", out)
            self.assertNotIn("heap_live_blocks", out)
            self.assertEqual(json.loads((run / "command.json").read_text())["harness_exit"], 9)

    def test_digest_presents_without_capture(self):
        # No --gpu-capture: no per-Present samples, but the pacer's count and first_present.
        report = {"status": "limit_reached", "milestones": [], "counters": {"import_calls": 9},
                  "present_pacing": {"enabled": True, "presents": 5712},
                  "first_present": {"import_seq": 91460, "elapsed_ms": 2900, "imports": []}}
        text = rnb.digest(report, 7)
        self.assertIn("presents: 5712 (paced, no capture samples), first at import call 91460 (2900 ms)", text)
        report.pop("present_pacing")
        report["first_present"] = None
        self.assertIn("presents: 0\n", rnb.digest(report, 7))

    def test_timeline_presents_fallback(self):
        samples = [{"index": 0, "wall_ms": 1, "draws": 3, "resources": 1}]
        self.assertEqual(rnb.timeline_presents({"present_samples": samples})[0]["counters"]["draws_issued"], 3)
        cap = [{"wall_ms": 5, "counters": {}}]
        self.assertEqual(rnb.timeline_presents({"presents": cap, "present_samples": samples}), cap)
        self.assertEqual(rnb.timeline_presents({}), [])

    def test_menu_flow_preset(self):
        self.assertEqual(rnb.PRESETS["menu-flow"]["max_ms"], 240_000)
        script = rnb.input_script_text(240_000, "menu-flow").splitlines()
        events = [line.split() for line in script if not line.startswith("#")]
        self.assertEqual(events[0], ["20000", "A", "100"])
        # Down, Down, A for LEGAL NOTICES' "Next >", every 12 s from 64 s.
        i = events.index(["64000", "DDown", "150"])
        self.assertEqual(events[i + 1], ["66000", "DDown", "150"])
        self.assertEqual(events[i + 2], ["68500", "A", "100"])
        times = [int(e[0]) for e in events]
        self.assertEqual(times, sorted(times))
        self.assertLess(times[-1], 240_000)
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(Path(d))
            code, _, _, _ = fx.run("--preset", "menu-flow", "--name", "mf")
            self.assertEqual(code, 0)
            written = (fx.build / "runs" / "mf" / "input-script.txt").read_text().splitlines()
            self.assertIn("68500 A 100", written)
            args = json.loads((fx.build / "runs" / "mf" / "boot-report.json").read_text())["args"]
            self.assertEqual(args[args.index("--max-ms") + 1], "240000")

    def test_menu_flow_2_preset(self):
        self.assertEqual(rnb.PRESETS["menu-flow-2"]["max_ms"], 360_000)
        script = rnb.input_script_text(360_000, "menu-flow-2").splitlines()
        events = [line.split() for line in script if not line.startswith("#")]
        self.assertEqual(events[0], ["20000", "A", "100"])
        times = [int(e[0]) for e in events]
        self.assertEqual(times, sorted(times))
        self.assertLess(times[-1], 360_000)
        # Each cycle: Up x6 reset, then Down, Down, A (the list screen's third entry).
        i = events.index(["64000", "DUp", "100"])
        self.assertEqual([e[1] for e in events[i:i + 9]], ["DUp"] * 6 + ["DDown", "DDown", "A"])
        self.assertEqual(events[i + 8], ["70500", "A", "100"])
        self.assertIn(["78000", "DUp", "100"], events)
        self.assertEqual({e[1] for e in events}, {"A", "DUp", "DDown"})

    def test_presets(self):
        self.assertEqual(rnb.PRESETS["long"]["max_ms"], 600_000)
        self.assertFalse(rnb.PRESETS["smoke"]["input"])
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(Path(d))
            fx.run("--name", "m", "--max-ms", "1234", "--watchdog-ms", "99")
            args = json.loads((fx.build / "runs" / "m" / "boot-report.json").read_text())["args"]
            self.assertEqual(args[args.index("--max-ms") + 1], "1234")
            self.assertEqual(args[args.index("--watchdog-ms") + 1], "99")
            self.assertNotIn("--input-script", args)

    def test_missing_report_and_existing_run_dir(self):
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(Path(d))
            fx.harness.write_text("import sys; sys.exit(64)\n")
            code, _, err, _ = fx.run("--name", "bad")
            self.assertEqual(code, 3)
            self.assertIn("no report", err)
            code, _, err, _ = fx.run("--name", "bad")
            self.assertEqual(code, 2)
            self.assertIn("already exists", err)

    def test_baseline_comparison(self):
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(Path(d))
            self.assertEqual(fx.run("--name", "base")[0], 0)
            code, out, _, _ = fx.run("--name", "same", "--baseline", "base")
            self.assertEqual(code, 0)
            self.assertIn("PASS", out)
            # Change the baseline's stop: the next run now differs.
            rep = fx.build / "runs" / "base" / "boot-report.json"
            data = json.loads(rep.read_text())
            data["fault"]["va"] = "0x99"
            rep.write_text(json.dumps(data))
            code, out, _, _ = fx.run("--name", "diff", "--baseline", "base")
            self.assertEqual(code, 1)
            self.assertIn("stop.fault.va", out)
            code, _, err, _ = fx.run("--name", "x", "--baseline", "nonexistent")
            self.assertEqual(code, 2)



class ChooseBuildDirTest(unittest.TestCase):
    def test_explicit_dir_wins(self):
        d, note = rnb.choose_build_dir(Path("C:/" + "a" * 300), Path("H:/nb1"), {}, windows=True)
        self.assertEqual(d, Path("H:/nb1").resolve())
        self.assertIsNone(note)

    def test_short_repo_keeps_default(self):
        d, note = rnb.choose_build_dir(Path("H:/r"), None, {"LOCALAPPDATA": "C:/L"}, windows=True)
        self.assertEqual(d, (Path("H:/r") / "work" / "native-boot").resolve())
        self.assertIsNone(note)

    def test_non_windows_keeps_default(self):
        repo = Path("/" + "a" * 300)
        d, note = rnb.choose_build_dir(repo, None, {}, windows=False)
        self.assertEqual(d, (repo / "work" / "native-boot").resolve())
        self.assertIsNone(note)

    def test_deep_repo_falls_back_to_localappdata(self):
        repo = Path("H:/" + "/".join(["deep"] * 30))
        d, note = rnb.choose_build_dir(repo, None, {"LOCALAPPDATA": "C:/L"}, windows=True)
        self.assertEqual(d.parent, Path("C:/L") / "playgta6")
        self.assertTrue(d.name.startswith("nb-"))
        self.assertIn(str(d), note)
        self.assertLess(len(str(d)) + rnb.BUILD_SUBPATH_BUDGET, rnb.WINDOWS_PATH_LIMIT)

    def test_fallback_without_localappdata_uses_drive_and_is_stable(self):
        repo = Path("H:/" + "/".join(["deep"] * 30))
        d1, _ = rnb.choose_build_dir(repo, None, {}, windows=True)
        d2, _ = rnb.choose_build_dir(repo, None, {}, windows=True)
        other, _ = rnb.choose_build_dir(Path("H:/" + "/".join(["deeper"] * 30)), None, {}, windows=True)
        self.assertEqual(d1, d2)
        self.assertNotEqual(d1, other)
        self.assertEqual(d1.parent.name, "pg6b")

class ThumbnailTests(unittest.TestCase):
    SEGS = [
        {"start_time": 0.0, "end_time": 500.0, "first_present": 0, "presents": 5, "draws_per_frame": 3},
        {"start_time": 2000.0, "end_time": 5000.0, "first_present": 5, "presents": 10, "draws_per_frame": 120},
        {"start_time": 6000.0, "end_time": 6200.0, "first_present": 15, "presents": 2, "draws_per_frame": 9},
    ]

    def test_selection_middle_of_long_segments(self):
        self.assertEqual(rnb.thumbnail_frames(self.SEGS), [(1, 9, 120, 2.0)])

    def test_selection_skips_uncaptured_frames(self):
        self.assertEqual(rnb.thumbnail_frames(self.SEGS, total_presents=9), [])

    def test_name(self):
        self.assertEqual(rnb.thumbnail_name(1, 2.0, 120), "seg01_2s_120d.png")

    def test_replay_command(self):
        cmd = rnb.replay_command("chrome.exe", Path("run"), "p.pack", [9, 20], Path("run/thumbs/replay"))
        self.assertEqual(cmd[1].replace("\\", "/").split("/")[-1], "replay_capture.py")
        for flag, val in (("--chrome", "chrome.exe"), ("--pack", "p.pack"), ("--thumbat", "9,20"),
                          ("--trace", "capture.pg6trace")):
            self.assertEqual(cmd[cmd.index(flag) + 1], val)

    def test_make_thumbnails_renames_and_lists(self):
        import argparse
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / "capture.pg6trace").write_bytes(b"x")
            pack = run / "s.pack"
            pack.write_bytes(b"x")
            args = argparse.Namespace(gpu_capture_payloads=True, chrome="c", replay_pack=str(pack))
            calls = []

            def fake(cmd):
                calls.append(cmd)
                (run / "thumbs" / "replay-swap9.png").write_bytes(b"png")
                return 0
            lines = rnb.make_thumbnails(args, run, {"screen_timeline": self.SEGS}, {}, call=fake)
            self.assertEqual(len(calls), 1)
            self.assertTrue((run / "thumbs" / "seg01_2s_120d.png").is_file())
            self.assertIn("seg01_2s_120d.png", lines[1])

    def test_hint_without_capture(self):
        import argparse
        with tempfile.TemporaryDirectory() as d:
            args = argparse.Namespace(gpu_capture_payloads=False, chrome=None, replay_pack=None)
            lines = rnb.make_thumbnails(args, Path(d), {}, {})
            self.assertIn("--gpu-capture-payloads", lines[0])


if __name__ == "__main__":
    unittest.main()
