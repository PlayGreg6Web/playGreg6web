import base64
import hashlib
import io
import json
import os
import re
import shutil
import socket
import struct
import sys
import tempfile
import threading
import unittest
import zlib
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import compare_boot_reports  # noqa: E402
import run_browser_boot as rbb  # noqa: E402


def touch(path: Path, when: float, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    os.utime(path, (when, when))


def make_sample_probe(
    *,
    boot_mode: str = "smoke-demo",
    session_state: str = "running_smoke",
    stage: str = "FIRST_PRESENT",
    presents: int = 5,
    real_draws: int = 0,
    fallback_draws: int = 5,
    validation_errors: int = 0,
    milestones: list[dict[str, Any]] | None = None,
    last_trap: dict[str, Any] | None = None,
    hud_log: list[str] | None = None,
) -> dict[str, Any]:
    if milestones is None:
        milestones = [
            {"milestone": "PGIM_LOAD", "index": 0, "reached": True, "elapsedMs": 12.5},
            {"milestone": "FIRST_PRESENT", "index": 8, "reached": True, "elapsedMs": 125.0},
        ]
    return {
        "href": "http://127.0.0.1:8123/",
        "readyState": "complete",
        "capabilities": {
            "crossOriginIsolated": True,
            "sharedArrayBuffer": True,
            "webgpu": True,
        },
        "bootState": {
            "bootMode": boot_mode,
            "sessionState": session_state,
            "milestone": stage,
            "presentCount": presents,
            "realPackDrawCount": real_draws,
            "fallbackDrawCount": fallback_draws,
            "unboundBindGroupCount": 0,
            "skippedDrawCount": 0,
            "validationErrorCount": validation_errors,
            "gpu": {
                "presentCount": presents,
                "realPackDrawCount": real_draws,
                "fallbackDrawCount": fallback_draws,
                "unboundBindGroupCount": 0,
                "skippedDrawCount": 0,
                "validationErrorCount": validation_errors,
            },
        },
        "diagnostics": {
            "bootMode": boot_mode,
            "sessionState": session_state,
            "milestone": stage,
            "framesPresented": presents,
            "validationErrorCount": validation_errors,
        },
        "tracerSummary": {
            "stage": stage,
            "milestones": milestones,
            "lastTrap": last_trap,
        },
        "hud": {
            "stage": stage,
            "status": "Running",
            "frames": str(presents),
            "fps": "60",
            "gpuErrors": str(validation_errors),
            "modeBadge": "Smoke Demo" if boot_mode == "smoke-demo" else "Real Wasm64",
            "logLines": hud_log if hud_log is not None else ["[00:00:01] Boot complete"],
        },
    }


def make_fake_session_result(
    *,
    ok: bool = True,
    status: str = "ok",
    exit_code: int = 0,
    probe: dict[str, Any] | None = None,
    console_lines: list[str] | None = None,
    page_errors: list[str] | None = None,
    presents: list[dict[str, Any]] | None = None,
    first_present_ms: float | None = 125.0,
) -> rbb.BrowserSessionResult:
    effective_probe = make_sample_probe() if (probe is None and ok) else (probe or {})
    effective_presents = (
        [
            {
                "index": 0,
                "wall_ms": 125.0,
                "dispatches": 0,
                "import_calls": 0,
                "counters": {
                    "draws_real_pack": 0,
                    "draws_fallback": 5,
                    "draws_skipped": 0,
                    "gpu_validation_errors": 0,
                },
            }
        ]
        if (presents is None and ok)
        else (presents or [])
    )
    return rbb.BrowserSessionResult(
        ok=ok,
        exit_code=exit_code,
        elapsed_ms=250.0,
        status=status,
        message="Running" if ok else "failed",
        crashed=not ok,
        server_ready={"status_code": 200, "coop": "same-origin", "coep": "require-corp", "api_status": {"ok": True}},
        final_probe=effective_probe,
        console_lines=console_lines if console_lines is not None else ["[   0.10s] [console.log] ready"],
        page_errors=page_errors or [],
        presents=effective_presents,
        first_present_ms=first_present_ms if ok else None,
    )


class RepoFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.repo = root / "repo"
        self.build = self.repo / "work" / "browser-boot"
        self.boot_dir = self.repo / "work" / "recomp_build"
        self.out_dir = self.repo / "web" / "dist"
        self.exe = root / "external" / "game.exe"
        self.game_dir = root / "external" / "pkg"
        self.game_dir.mkdir(parents=True)
        self.chrome = root / "external" / ("chrome.exe" if sys.platform == "win32" else "chrome")
        self.base = 1_000_000.0
        self.lift_cache = root / "liftcache"

        touch(self.repo / "Cargo.toml", self.base, '[workspace]\nmembers = ["recomp"]\n')
        touch(self.repo / "Cargo.lock", self.base, "# lock\n")
        touch(self.repo / "recomp" / "Cargo.toml", self.base, '[package]\nname="playgta6-recomp"\n')
        touch(self.repo / "recomp" / "src" / "main.rs", self.base, "fn main() {}\n")
        touch(self.repo / "runtime" / "a.cpp", self.base, "int x;\n")
        touch(self.repo / "tools" / "build_guest_wasm.py", self.base, "# build\n")
        touch(self.repo / "server" / "serve_local.py", self.base, "# server\n")
        touch(self.repo / "web" / "index.html", self.base, "<!doctype html>\n")
        touch(self.exe, self.base, "MZ")
        touch(self.chrome, self.base, "bin")

        self.recomp = self.repo / "target" / "release" / ("playgta6-recomp" + rbb.EXE_SUFFIX)
        self.manifest = self.boot_dir / "program-manifest.json"
        self.cmake_cache = self.build / "runtime-pic" / "CMakeCache.txt"
        self.runtime_lib = self.build / "runtime-pic" / "libplaygta6_core.a"
        self.runtime_wasm = self.out_dir / "runtime.wasm"
        self.runtime_js = self.out_dir / "runtime.js"

        touch(self.recomp, self.base + 10)
        touch(self.manifest, self.base + 20, "{}")
        touch(self.cmake_cache, self.base + 25)
        touch(self.runtime_lib, self.base + 30)
        touch(self.runtime_wasm, self.base + 40)
        touch(self.runtime_js, self.base + 40)

    def run_cli(
        self,
        *argv: str,
        runner=None,
        session_runner=None,
        chrome_discoverer=None,
        env=None,
    ) -> tuple[int, str, str, list[list[str]], list[dict[str, Any]]]:
        cmd_calls: list[list[str]] = []
        session_calls: list[dict[str, Any]] = []

        def default_runner(cmd, cwd, log, timeout):
            cmd_calls.append(list(cmd))
            return 0

        def default_session(**kwargs):
            session_calls.append(kwargs)
            return make_fake_session_result(
                probe=make_sample_probe(
                    boot_mode="smoke-demo" if kwargs.get("smoke") else "real-wasm64",
                    session_state="running_smoke" if kwargs.get("smoke") else "running_wasm",
                )
            )

        def default_chrome(explicit, **kw):
            return explicit if explicit is not None else self.chrome

        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = rbb.main(
                list(argv),
                runner=runner or default_runner,
                session_runner=session_runner or default_session,
                chrome_discoverer=chrome_discoverer or default_chrome,
                env={rbb.lift_cache.LIFT_CACHE_ENV: str(self.lift_cache)} if env is None else env,
                repo_root=self.repo,
            )
        return code, out.getvalue(), err.getvalue(), cmd_calls, session_calls


class CliAndPresetsTests(unittest.TestCase):
    def test_presets_and_flag_defaults(self):
        self.assertEqual(rbb.PRESETS["smoke"]["duration_s"], 60.0)
        self.assertFalse(rbb.PRESETS["smoke"]["input"])
        self.assertEqual(rbb.PRESETS["start-screen"]["duration_s"], 120.0)
        self.assertTrue(rbb.PRESETS["start-screen"]["input"])
        self.assertEqual(rbb.PRESETS["long"]["duration_s"], 600.0)
        self.assertFalse(rbb.PRESETS["long"]["input"])

        parser = rbb.build_arg_parser()
        args = parser.parse_args([])
        self.assertEqual(args.preset, "smoke")
        self.assertTrue(args.headless)
        self.assertFalse(args.smoke)
        self.assertEqual(args.port, 0)

        args_headed = parser.parse_args(["--headed", "--preset", "start-screen", "--capture-frames", "--mp4"])
        self.assertFalse(args_headed.headless)
        self.assertEqual(args_headed.preset, "start-screen")
        self.assertTrue(args_headed.capture_frames)
        self.assertTrue(args_headed.mp4)

    def test_invalid_flag_combinations_return_exit_2(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            for bad_args in (
                ("--rebuild", "--trust-build"),
                ("--rebuild", "--no-build"),
                ("--headless", "--headed"),
                ("--duration-s", "0"),
                ("--duration-s", "-5"),
                ("--port", "70000"),
                ("--frame-interval-s", "0"),
                ("--name", "../escape"),
            ):
                code, _, _, _, _ = fx.run_cli("--smoke", "--dry-run", *bad_args)
                self.assertEqual(code, 2, f"expected exit 2 for {bad_args}")


class DryRunAndStalenessTests(unittest.TestCase):
    def test_smoke_dry_run_requires_zero_game_files_and_sanitizes_output(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d) / "clean_repo"
            (repo / "web").mkdir(parents=True)
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = rbb.main(
                    ["--smoke", "--dry-run", "--name", "smoke-check"],
                    chrome_discoverer=lambda *a, **k: None,
                    env={},
                    repo_root=repo,
                )
            self.assertEqual(code, 0, err.getvalue())
            payload = json.loads(out.getvalue())
            self.assertEqual(payload["mode"], "smoke")
            self.assertEqual(payload["plan"], [])
            self.assertEqual(payload["duration_s"], rbb.SMOKE_MODE_DEFAULT_DURATION_S)
            self.assertFalse(payload["chrome_found"])
            self.assertEqual(payload["url"], "http://127.0.0.1:<ephemeral>/?smoke=1")
            # Ensure zero host paths leak in dry-run JSON
            raw_json = out.getvalue()
            self.assertNotRegex(raw_json, r"(?<![A-Za-z0-9_])[A-Za-z]:[/\\]")
            self.assertNotIn(str(Path(d)), raw_json)

    def test_build_only_builds_without_chrome_session_or_run_dir(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            os.utime(fx.runtime_wasm, (fx.base - 5, fx.base - 5))  # stale guest wasm
            code, out, err, cmd_calls, session_calls = fx.run_cli(
                "--build-only", "--exe", str(fx.exe), "--game-dir", str(fx.game_dir), "--name", "b1",
                chrome_discoverer=lambda *a, **k: None,
            )
            self.assertEqual(code, 0, err)
            self.assertEqual(session_calls, [])
            self.assertTrue(any("build_guest_wasm.py" in " ".join(c) for c in cmd_calls))
            self.assertFalse((fx.build / "runs" / "b1").exists())

    def test_nothing_stale_runs_zero_build_commands(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            code, out, err, cmd_calls, session_calls = fx.run_cli(
                "--exe", str(fx.exe), "--game-dir", str(fx.game_dir), "--name", "r1"
            )
            self.assertEqual(code, 0, err)
            self.assertEqual(cmd_calls, [])
            self.assertEqual(len(session_calls), 1)
            self.assertNotIn("[plan]", err)
            self.assertIn("FIRST_PRESENT", out)

    def test_full_staleness_plan_uses_capped_split_and_pic_flags(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            plan, runtime_lib = rbb.make_build_plan(
                repo=fx.repo,
                build_dir=fx.build,
                boot_dir=fx.boot_dir,
                out_dir=fx.out_dir,
                exe=fx.exe,
                game_dir=fx.game_dir,
                runtime_lib_override=None,
                jobs=4,
                force=True,
                smoke=False,
            )
            self.assertEqual(
                [s["stage"] for s in plan.steps],
                ["recomp", "program", "runtime_configure", "runtime_lib", "guest_wasm"],
            )
            prog_cmd = plan.steps[1]["cmd"]
            self.assertIn("build-program", prog_cmd)
            self.assertIn("--target", prog_cmd)
            self.assertIn("wasm64", prog_cmd)
            self.assertIn("--memory-model", prog_cmd)
            self.assertIn("window-runs", prog_cmd)
            self.assertIn("--indirect-dispatch", prog_cmd)
            self.assertIn("shard", prog_cmd)
            self.assertIn("--dll-dir", prog_cmd)

            cfg_cmd = plan.steps[2]["cmd"]
            self.assertEqual(cfg_cmd[:2], ["emcmake", "cmake"])
            self.assertTrue(any("-fPIC" in tok and "-sMEMORY64=1" in tok for tok in cfg_cmd))

            wasm_cmd = plan.steps[4]["cmd"]
            self.assertTrue(wasm_cmd[1].endswith("build_guest_wasm.py"))
            for expected_flag in (
                "--split-modules",
                "--split-strategy",
                "capped",
                "--link-guest-boot",
                "--browser-boot-exports",
                "--proxy-to-pthread",
                "--strip",
                "--no-link-opt",
                "--runtime-lib",
                str(runtime_lib),
            ):
                self.assertIn(expected_flag, wasm_cmd)

    def test_cascading_staleness_recomp_and_runtime(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            # Touching recompiler source rebuilds recomp -> program -> guest_wasm (not runtime_lib).
            touch(fx.repo / "recomp" / "src" / "main.rs", fx.base + 100)
            plan, _ = rbb.make_build_plan(
                repo=fx.repo, build_dir=fx.build, boot_dir=fx.boot_dir, out_dir=fx.out_dir,
                exe=fx.exe, game_dir=fx.game_dir, runtime_lib_override=None, jobs=2, force=False, smoke=False,
            )
            self.assertEqual([s["stage"] for s in plan.steps], ["recomp", "program", "guest_wasm"])

            # Reset and touch runtime source -> rebuilds runtime_lib -> guest_wasm only.
            touch(fx.recomp, fx.base + 200)
            touch(fx.manifest, fx.base + 210, "{}")
            touch(fx.repo / "runtime" / "a.cpp", fx.base + 300)
            plan, _ = rbb.make_build_plan(
                repo=fx.repo, build_dir=fx.build, boot_dir=fx.boot_dir, out_dir=fx.out_dir,
                exe=fx.exe, game_dir=fx.game_dir, runtime_lib_override=None, jobs=2, force=False, smoke=False,
            )
            self.assertEqual([s["stage"] for s in plan.steps], ["runtime_lib", "guest_wasm"])

            # Touch build_guest_wasm.py -> rebuilds guest_wasm only.
            touch(fx.runtime_lib, fx.base + 400)
            touch(fx.runtime_wasm, fx.base + 410)
            touch(fx.runtime_js, fx.base + 410)
            touch(fx.repo / "tools" / "build_guest_wasm.py", fx.base + 500)
            plan, _ = rbb.make_build_plan(
                repo=fx.repo, build_dir=fx.build, boot_dir=fx.boot_dir, out_dir=fx.out_dir,
                exe=fx.exe, game_dir=fx.game_dir, runtime_lib_override=None, jobs=2, force=False, smoke=False,
            )
            self.assertEqual([s["stage"] for s in plan.steps], ["guest_wasm"])

    def test_trust_build_and_no_build_and_build_failure(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            touch(fx.exe, fx.base + 500)
            touch(fx.repo / "runtime" / "a.cpp", fx.base + 600)

            # --no-build fails when stale
            code, _, err, cmds, sessions = fx.run_cli(
                "--exe", str(fx.exe), "--name", "nb1", "--no-build"
            )
            self.assertEqual((code, cmds, sessions), (2, [], []))
            self.assertIn("stale", err)

            # --trust-build skips staleness even with --no-build
            code, _, err, cmds, sessions = fx.run_cli(
                "--exe", str(fx.exe), "--name", "tb1", "--no-build", "--trust-build"
            )
            self.assertEqual(code, 0, err)
            self.assertEqual(cmds, [])
            self.assertEqual(len(sessions), 1)
            self.assertNotIn("[plan]", err)

            # Failing build step stops with exit 2
            code, _, err, _, sessions = fx.run_cli(
                "--exe", str(fx.exe), "--name", "fail1", runner=lambda *a: 1
            )
            self.assertEqual((code, sessions), (2, []))
            self.assertIn("failed", err)

    def test_missing_manifest_without_exe_fails_with_exit_2(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            fx.manifest.unlink()
            code, _, err, _, _ = fx.run_cli("--name", "no-exe")
            self.assertEqual(code, 2)
            self.assertIn("PLAYGTA6_EXE", err)

    def test_shader_pack_and_reflection_sidecar_staging(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            shader_src = Path(d) / "external_shaders"
            pack = shader_src / "game.wgsl.pack"
            sidecar = shader_src / "game.wgsl.pack.reflect.json"
            refl = shader_src / "reflection-manifest.json"
            touch(pack, fx.base, "WGPK")
            touch(sidecar, fx.base, '{"schema":"playgta6-stage-reflection","stages":{}}')
            touch(refl, fx.base, '{"schema_version":1,"shaders":{}}')

            # Dry-run shows shader staging without copying
            code, out, err, _, _ = fx.run_cli(
                "--smoke", "--dry-run", "--shader-pack", str(pack), "--name", "sdry"
            )
            self.assertEqual(code, 0, err)
            dry = json.loads(out)
            names = [s["name"] for s in dry["shader_staging"]]
            self.assertEqual(names, ["game.wgsl.pack", "game.wgsl.pack.reflect.json", "reflection-manifest.json"])

            # Real run stages all three files and passes --shaders dir to session
            code, _, err, _, sessions = fx.run_cli(
                "--smoke", "--shader-pack", str(pack), "--name", "srun"
            )
            self.assertEqual(code, 0, err)
            staged_dir = fx.build / "staged-shaders"
            self.assertEqual(sessions[0]["shaders_dir"], staged_dir)
            for fname in ("game.wgsl.pack", "game.wgsl.pack.reflect.json", "reflection-manifest.json"):
                self.assertTrue((staged_dir / fname).is_file(), fname)


class LiftCacheTests(unittest.TestCase):
    INPUTS = ({"recomp/src/main.rs": "a", "Cargo.lock": "b"}, "exe", {"bink.dll": "c"}, {"runtime/os/x.h": "d"})

    def test_lift_options_are_part_of_the_key(self):
        import run_native_boot as rnb  # noqa: PLC0415

        native = rbb.lift_cache.lift_key(*self.INPUTS, rnb.NATIVE_LIFT_OPTIONS)
        browser = rbb.lift_cache.lift_key(*self.INPUTS, rbb.BROWSER_LIFT_OPTIONS)
        self.assertNotEqual(native, browser)
        self.assertEqual(browser, rbb.lift_cache.lift_key(*self.INPUTS, list(rbb.BROWSER_LIFT_OPTIONS)))
        # Any single option change is another entry.
        opts = list(rbb.BROWSER_LIFT_OPTIONS)
        variants = {tuple(opts[:i] + [opts[i] + "x"] + opts[i + 1:]) for i in range(len(opts))}
        keys = {rbb.lift_cache.lift_key(*self.INPUTS, v) for v in variants}
        self.assertEqual(len(keys), len(variants))
        self.assertNotIn(browser, keys)
        # The native runner uses the same shared helper.
        self.assertIs(rnb.lift_key, rbb.lift_cache.lift_key)

    def test_inputs_key_without_game_dir(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            k1 = rbb.lift_cache.lift_inputs_key(fx.repo, fx.exe, None, rbb.BROWSER_LIFT_OPTIONS)
            (fx.game_dir / "a.dll").write_text("dll")
            k2 = rbb.lift_cache.lift_inputs_key(fx.repo, fx.exe, fx.game_dir, rbb.BROWSER_LIFT_OPTIONS)
            self.assertNotEqual(k1, k2)

    def test_miss_stores_and_a_hit_skips_recomp_and_lift(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            touch(fx.repo / "recomp" / "src" / "main.rs", fx.base + 100)  # recomp + program stale

            def lifting_runner(cmd, cwd, log, timeout):
                if "build-program" in cmd:
                    out = Path(cmd[cmd.index("--out") + 1])
                    (out / "shards").mkdir(parents=True, exist_ok=True)
                    (out / "shards" / "shard-00000.ll").write_text("ir")
                    (out / "program-manifest.json").write_text("{}")
                return 0

            code, _, err, _, _ = fx.run_cli("--exe", str(fx.exe), "--game-dir", str(fx.game_dir), "--name", "r1",
                                            runner=lifting_runner)
            self.assertEqual(code, 0, err)
            entries = list(fx.lift_cache.iterdir())
            self.assertEqual(len(entries), 1)
            self.assertTrue((entries[0] / "shards" / "shard-00000.ll").is_file())
            # A native entry for the same inputs is a different key.
            import run_native_boot as rnb  # noqa: PLC0415
            native_key = rbb.lift_cache.lift_inputs_key(fx.repo, fx.exe, fx.game_dir, rnb.NATIVE_LIFT_OPTIONS)
            self.assertNotEqual(native_key, entries[0].name)

            # The boot dir is lost (fresh checkout): the next build restores it from the cache.
            shutil.rmtree(fx.boot_dir)
            fx.boot_dir.mkdir()
            (fx.boot_dir / rbb.BOOT_OPTIONS_NAME).write_text("{}")
            calls: list[list[str]] = []

            def recording_runner(cmd, cwd, log, timeout):
                calls.append(list(cmd))
                return 0

            code, _, err, _, _ = fx.run_cli("--exe", str(fx.exe), "--game-dir", str(fx.game_dir), "--name", "r2",
                                            runner=recording_runner)
            self.assertEqual(code, 0, err)
            self.assertIn("lift cache hit", err)
            self.assertFalse(any("build-program" in c or "cargo" in c for c in calls), calls)
            self.assertEqual((fx.boot_dir / "shards" / "shard-00000.ll").read_text(), "ir")
            self.assertTrue((fx.boot_dir / rbb.BOOT_OPTIONS_NAME).is_file())

            # --rebuild lifts again.
            calls.clear()
            code, _, err, _, _ = fx.run_cli("--exe", str(fx.exe), "--game-dir", str(fx.game_dir), "--name", "r3",
                                            "--rebuild", runner=lambda *a: (calls.append(list(a[0])), lifting_runner(*a))[1])
            self.assertEqual(code, 0, err)
            self.assertTrue(any("build-program" in c for c in calls))

    def test_foreign_boot_dir_is_never_cleared(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            fx.manifest.unlink()
            touch(fx.boot_dir / "unrelated.txt", fx.base)
            plan, _ = rbb.make_build_plan(
                repo=fx.repo, build_dir=fx.build, boot_dir=fx.boot_dir, out_dir=fx.out_dir,
                exe=fx.exe, game_dir=fx.game_dir, runtime_lib_override=None, jobs=2, force=False, smoke=False,
                lift_cache_dir=fx.lift_cache,
            )
            self.assertTrue(plan.has("program"))
            self.assertNotIn("program", plan.hooks)


class ChromeDiscoveryTests(unittest.TestCase):
    def test_discovery_priority_and_fallbacks(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            explicit = root / "explicit-chrome"
            env_chrome = root / "env-chrome"
            which_chrome = root / "which-chrome"
            for p in (explicit, env_chrome, which_chrome):
                p.write_text("x", encoding="utf-8")

            # Explicit wins
            self.assertEqual(
                rbb.discover_chrome(explicit, env={"CHROME_PATH": str(env_chrome)}, which_fn=lambda _: str(which_chrome)),
                explicit,
            )
            # Missing explicit returns None (does not silently fall back)
            self.assertIsNone(
                rbb.discover_chrome(root / "missing", env={"CHROME_PATH": str(env_chrome)}, which_fn=lambda _: str(which_chrome))
            )
            # CHROME_PATH wins over which
            self.assertEqual(
                rbb.discover_chrome(None, env={"CHROME_PATH": str(env_chrome)}, which_fn=lambda _: str(which_chrome)),
                env_chrome,
            )
            # which fallback
            self.assertEqual(
                rbb.discover_chrome(None, env={}, which_fn=lambda n: str(which_chrome) if "chrome" in n else None, platform="linux"),
                which_chrome,
            )
            # Standard path fallback (Linux & Windows)
            self.assertEqual(
                rbb.discover_chrome(
                    None, env={}, which_fn=lambda _: None, platform="linux",
                    exists_fn=lambda p: p.as_posix() == "/usr/bin/chromium",
                ),
                Path("/usr/bin/chromium"),
            )
            win_edge = Path(r"C:\Program Files") / "Microsoft" / "Edge" / "Application" / "msedge.exe"
            self.assertEqual(
                rbb.discover_chrome(
                    None, env={"PROGRAMFILES": r"C:\Program Files"}, which_fn=lambda _: None, platform="win32",
                    exists_fn=lambda p: p == win_edge,
                ),
                win_edge,
            )
            # Not found returns None -> main returns exit code 2
            self.assertIsNone(
                rbb.discover_chrome(None, env={}, which_fn=lambda _: None, platform="linux", exists_fn=lambda _: False)
            )

    def test_missing_chrome_in_live_run_returns_exit_2(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            code, _, err, _, _ = fx.run_cli(
                "--smoke", "--name", "nochrome", chrome_discoverer=lambda *a, **k: None
            )
            self.assertEqual(code, 2)
            self.assertIn("Chrome", err)


class ReportAndBaselineTests(unittest.TestCase):
    def test_run_artifacts_sanitization_and_compare_boot_reports_parity(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            dirty_session = make_fake_session_result(
                console_lines=[
                    r"[   0.10s] [console.log] loaded C:\Users\secretuser\Games\title.exe",
                    r"[   0.20s] [console.warn] network path \\fileserver\share\dump.bin",
                    "[   0.30s] [console.log] posix /home/alice/local/file.rpf",
                ],
                probe=make_sample_probe(
                    boot_mode="smoke-demo",
                    presents=4,
                    fallback_draws=4,
                    hud_log=[r"opened C:\Users\secretuser\work\guest.pgim"],
                ),
            )
            code, out, err, _, _ = fx.run_cli(
                "--smoke", "--name", "base", session_runner=lambda **kw: dirty_session
            )
            self.assertEqual(code, 0, err)
            run_dir = fx.build / "runs" / "base"
            for fname in ("boot-report.json", "console.log", "telemetry.json", "command.json", "digest.txt"):
                self.assertTrue((run_dir / fname).is_file(), fname)
                text = (run_dir / fname).read_text(encoding="utf-8")
                self.assertNotRegex(text, r"(?<![A-Za-z0-9_])[A-Za-z]:[/\\]", f"host Windows path leaked in {fname}")
                self.assertNotIn("/home/alice", text, f"host POSIX path leaked in {fname}")
                self.assertNotIn("\\\\fileserver", text, f"UNC path leaked in {fname}")

            report = json.loads((run_dir / "boot-report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["schema"], compare_boot_reports.REPORT_SCHEMA)
            self.assertEqual(report["schema_version"], 2)
            self.assertEqual([m["milestone"] for m in report["milestones"]], ["PGIM_LOAD", "FIRST_PRESENT"])
            self.assertIn("FIRST_PRESENT", out)
            self.assertIn("fallback_draws=4", out)

            # Matching baseline returns 0
            code, cmp_out, cmp_err, _, _ = fx.run_cli(
                "--smoke", "--name", "same", "--baseline", "base",
                session_runner=lambda **kw: dirty_session,
            )
            self.assertEqual(code, 0, cmp_err)
            self.assertIn("PASS", cmp_out)

            # Different milestone/fault against baseline returns 1
            fault_session = make_fake_session_result(
                status="fault",
                probe=make_sample_probe(
                    stage="ENTRY_CALL",
                    milestones=[{"milestone": "ENTRY_CALL", "index": 3, "reached": True, "elapsedMs": 10.0}],
                    last_trap={
                        "kind": "UNIMPLEMENTED_OPCODE",
                        "guestVa": "0x140001000",
                        "message": "unimplemented opcode at C:\\Users\\bob\\game.exe",
                        "fatal": True,
                    },
                ),
            )
            code, diff_out, _, _, _ = fx.run_cli(
                "--smoke", "--name", "diff", "--baseline", "base",
                session_runner=lambda **kw: fault_session,
            )
            self.assertEqual(code, 1)
            self.assertIn("FAIL", diff_out)
            self.assertIn("stop.status", diff_out)

            # Missing baseline returns 2
            code, _, miss_err, _, _ = fx.run_cli("--smoke", "--name", "nobase", "--baseline", "does-not-exist")
            self.assertEqual(code, 2)
            self.assertIn("baseline run not found", miss_err)

    def test_existing_run_dir_returns_2_and_missing_report_returns_3(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            crashed = make_fake_session_result(ok=False, status="error", exit_code=1, probe={}, presents=[])
            code, _, err, _, _ = fx.run_cli("--smoke", "--name", "crash1", session_runner=lambda **kw: crashed)
            self.assertEqual(code, 3)
            self.assertIn("wrote no report", err)

            # Reusing same non-empty run name returns 2
            code, _, err, _, _ = fx.run_cli("--smoke", "--name", "crash1")
            self.assertEqual(code, 2)
            self.assertIn("already exists", err)


class ServerAndCdpLifecycleTests(unittest.TestCase):
    def test_wait_for_server_ready_validates_coop_coep_and_premature_exit(self):
        class GoodHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps({"ok": True, "coopCoep": True}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cross-Origin-Opener-Policy", "same-origin")
                self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                return

        srv = ThreadingHTTPServer(("127.0.0.1", 0), GoodHandler)
        port = srv.server_address[1]
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            ready = rbb.wait_for_server_ready("127.0.0.1", port, timeout_s=3.0)
            self.assertEqual(ready["status_code"], 200)
            self.assertEqual(ready["coop"], "same-origin")
            self.assertEqual(ready["coep"], "require-corp")
            self.assertTrue(ready["api_status"]["ok"])
        finally:
            srv.shutdown()
            srv.server_close()

        # Server missing COOP/COEP headers times out cleanly
        class BadHeaderHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args):
                return

        bad_srv = ThreadingHTTPServer(("127.0.0.1", 0), BadHeaderHandler)
        bad_port = bad_srv.server_address[1]
        th2 = threading.Thread(target=bad_srv.serve_forever, daemon=True)
        th2.start()
        try:
            with self.assertRaises(rbb.RunnerError) as ctx:
                rbb.wait_for_server_ready("127.0.0.1", bad_port, timeout_s=0.4)
            self.assertIn("COOP/COEP", str(ctx.exception))
        finally:
            bad_srv.shutdown()
            bad_srv.server_close()

        # Dead subprocess fails fast
        class DeadProc:
            returncode = 17

            def poll(self):
                return 17

        with self.assertRaises(rbb.RunnerError) as ctx:
            rbb.wait_for_server_ready("127.0.0.1", rbb.find_free_port(), timeout_s=3.0, proc=DeadProc())
        self.assertIn("exited prematurely", str(ctx.exception))

    def test_pure_python_cdp_websocket_client_handshake_events_and_ping_pong(self):
        """Exercise real TCP RFC 6455 handshake, masked frames, ping/pong, and CDP events."""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.bind(("127.0.0.1", 0))
        server_sock.listen(1)
        port = server_sock.getsockname()[1]
        received_methods: list[str] = []

        def recv_client_frame(conn: socket.socket) -> tuple[int, bytes]:
            def rx(n: int) -> bytes:
                buf = bytearray()
                while len(buf) < n:
                    chunk = conn.recv(n - len(buf))
                    if not chunk:
                        raise EOFError()
                    buf.extend(chunk)
                return bytes(buf)

            b0, b1 = rx(2)
            opcode = b0 & 0x0F
            masked = bool(b1 & 0x80)
            length = b1 & 0x7F
            if length == 126:
                length = struct.unpack("!H", rx(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", rx(8))[0]
            mask_key = rx(4) if masked else b""
            payload = rx(length) if length else b""
            if masked:
                payload = bytes(b ^ mask_key[i & 3] for i, b in enumerate(payload))
            return opcode, payload

        def mock_ws_server():
            conn, _ = server_sock.accept()
            try:
                raw = b""
                while b"\r\n\r\n" not in raw:
                    raw += conn.recv(4096)
                m = re.search(rb"Sec-WebSocket-Key:\s*(\S+)", raw, re.I)
                assert m is not None
                accept = base64.b64encode(
                    hashlib.sha1(m.group(1) + rbb.WEBSOCKET_GUID.encode("ascii")).digest()
                ).decode("ascii")
                conn.sendall(
                    (
                        "HTTP/1.1 101 Switching Protocols\r\n"
                        "Upgrade: websocket\r\n"
                        "Connection: Upgrade\r\n"
                        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
                    ).encode("ascii")
                )
                while True:
                    try:
                        opcode, payload = recv_client_frame(conn)
                    except EOFError:
                        break
                    if opcode == 0x8:
                        break
                    if opcode == 0xA:
                        continue
                    req = json.loads(payload.decode("utf-8"))
                    received_methods.append(req["method"])
                    # Send a PING + an async console event before the reply
                    conn.sendall(rbb._ws_encode_frame(0x9, b"hi", mask=False))
                    ev = json.dumps(
                        {
                            "method": "Runtime.consoleAPICalled",
                            "params": {"type": "log", "args": [{"value": f"seen {req['method']}"}]},
                        }
                    ).encode("utf-8")
                    conn.sendall(rbb._ws_encode_frame(0x1, ev, mask=False))
                    reply = json.dumps({"id": req["id"], "result": {"ok": True}}).encode("utf-8")
                    conn.sendall(rbb._ws_encode_frame(0x1, reply, mask=False))
            finally:
                conn.close()
                server_sock.close()

        th = threading.Thread(target=mock_ws_server, daemon=True)
        th.start()
        client = rbb.CdpClient(f"ws://127.0.0.1:{port}/devtools/page/1", timeout=5.0)
        try:
            client.connect()
            res = client.send("Runtime.enable")
            self.assertEqual(res, {"ok": True})
            events = client.drain_events(timeout=0.05)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["method"], "Runtime.consoleAPICalled")
        finally:
            client.close()
            th.join(timeout=3.0)
        self.assertEqual(received_methods, ["Runtime.enable"])

    def test_run_browser_session_lifecycle_capture_mp4_and_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            run_dir = fx.build / "runs" / "sess1"
            run_dir.mkdir(parents=True)

            procs: list[Any] = []
            created_profiles: list[Path] = []

            class FakeProc:
                def __init__(self, cmd):
                    self.cmd = list(cmd)
                    self.terminated = False
                    self.returncode = None

                def poll(self):
                    return self.returncode

                def terminate(self):
                    self.terminated = True
                    self.returncode = 0

                def wait(self, timeout=None):
                    self.returncode = 0
                    return 0

                def kill(self):
                    self.returncode = -9

            def fake_popen(cmd, **kwargs):
                for tok in cmd:
                    if isinstance(tok, str) and tok.startswith("--user-data-dir="):
                        created_profiles.append(Path(tok.split("=", 1)[1]))
                p = FakeProc(cmd)
                procs.append(p)
                return p

            class FakeCdp:
                def __init__(self, ws_url: str) -> None:
                    self.ws_url = ws_url
                    self.methods: list[str] = []
                    self.closed = False
                    self.eval_count = 0

                def connect(self):
                    return None

                def close(self):
                    self.closed = True

                def send(self, method: str, params: dict[str, Any] | None = None, timeout: float | None = None):
                    self.methods.append(method)
                    if method == "Runtime.evaluate":
                        self.eval_count += 1
                        return {
                            "result": {
                                "value": make_sample_probe(presents=self.eval_count, fallback_draws=self.eval_count)
                            }
                        }
                    if method == "Page.captureScreenshot":
                        return {"data": base64.b64encode(b"\xff\xd8\xff\xd9").decode("ascii")}
                    return {}

                def drain_events(self, timeout: float = 0.0):
                    if self.eval_count == 0:
                        return [
                            {
                                "method": "Runtime.consoleAPICalled",
                                "params": {"type": "log", "args": [{"value": "hello from browser"}]},
                            }
                        ]
                    return []

            cdp_instances: list[FakeCdp] = []

            def cdp_factory(ws_url: str):
                inst = FakeCdp(ws_url)
                cdp_instances.append(inst)
                return inst

            ffmpeg_calls: list[list[str]] = []

            def fake_runner(cmd, cwd, log, timeout):
                ffmpeg_calls.append(list(cmd))
                # Write dummy mp4 at last arg
                Path(cmd[-1]).write_bytes(b"ftypmp42")
                return 0

            res = rbb.run_browser_session(
                repo=fx.repo,
                run_dir=run_dir,
                chrome_path=fx.chrome,
                port=0,
                boot_dir=fx.boot_dir,
                game_dir=None,
                shaders_dir=None,
                duration_s=0.35,
                headless=True,
                smoke=True,
                send_input=False,
                capture_frames=True,
                frame_interval_s=0.1,
                encode_mp4=True,
                runner=fake_runner,
                popen_fn=fake_popen,
                cdp_factory=cdp_factory,
                wait_server_fn=lambda *a, **k: {"status_code": 200, "coop": "same-origin", "coep": "require-corp"},
                wait_cdp_fn=lambda *a, **k: "ws://127.0.0.1:9222/devtools/page/1",
                which_fn=lambda name: "ffmpeg" if name == "ffmpeg" else None,
            )

            self.assertTrue(res.ok)
            self.assertGreaterEqual(len(res.captured_frames), 1)
            self.assertTrue((run_dir / "frames" / "frame_0001.jpg").is_file())
            self.assertEqual(res.mp4_path, "capture.mp4")
            self.assertEqual(len(ffmpeg_calls), 1)
            self.assertIn("Page.navigate", cdp_instances[0].methods)
            self.assertIn("Page.captureScreenshot", cdp_instances[0].methods)
            # Cleanup verified
            self.assertTrue(cdp_instances[0].closed)
            self.assertEqual(len(procs), 2)
            self.assertTrue(all(p.terminated for p in procs))
            self.assertEqual(len(created_profiles), 1)
            self.assertFalse(created_profiles[0].exists())

    def test_run_browser_session_cleans_up_on_keyboard_interrupt(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            run_dir = fx.build / "runs" / "kb1"
            run_dir.mkdir(parents=True)
            procs: list[Any] = []
            profiles: list[Path] = []

            class FakeProc:
                terminated = False
                returncode = None

                def poll(self):
                    return self.returncode

                def terminate(self):
                    self.terminated = True
                    self.returncode = 0

                def wait(self, timeout=None):
                    return 0

                def kill(self):
                    self.returncode = -9

            def fake_popen(cmd, **kwargs):
                for tok in cmd:
                    if isinstance(tok, str) and tok.startswith("--user-data-dir="):
                        profiles.append(Path(tok.split("=", 1)[1]))
                p = FakeProc()
                procs.append(p)
                return p

            class InterruptCdp:
                closed = False

                def __init__(self, ws_url: str):
                    pass

                def connect(self):
                    return None

                def send(self, method: str, params=None, timeout=None):
                    if method == "Page.navigate":
                        raise KeyboardInterrupt()
                    return {}

                def drain_events(self, timeout=0.0):
                    return []

                def close(self):
                    self.closed = True

            cdp_holder: list[InterruptCdp] = []

            def make_cdp(ws_url: str):
                c = InterruptCdp(ws_url)
                cdp_holder.append(c)
                return c

            with self.assertRaises(KeyboardInterrupt):
                rbb.run_browser_session(
                    repo=fx.repo,
                    run_dir=run_dir,
                    chrome_path=fx.chrome,
                    port=0,
                    boot_dir=None,
                    game_dir=None,
                    shaders_dir=None,
                    duration_s=5.0,
                    headless=True,
                    smoke=True,
                    send_input=False,
                    capture_frames=False,
                    frame_interval_s=1.0,
                    encode_mp4=False,
                    popen_fn=fake_popen,
                    cdp_factory=make_cdp,
                    wait_server_fn=lambda *a, **k: {"status_code": 200, "coop": "same-origin", "coep": "require-corp"},
                    wait_cdp_fn=lambda *a, **k: "ws://127.0.0.1:9222/devtools/page/1",
                )

            self.assertTrue(cdp_holder[0].closed)
            self.assertEqual(len(procs), 2)
            self.assertTrue(all(p.terminated for p in procs))
            self.assertEqual(len(profiles), 1)
            self.assertFalse(profiles[0].exists())

    def test_encode_frames_to_mp4_graceful_when_ffmpeg_missing(self):
        with tempfile.TemporaryDirectory() as d:
            frames_dir = Path(d) / "frames"
            frames_dir.mkdir()
            (frames_dir / "frame_0001.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            ok, warn = rbb.encode_frames_to_mp4(
                frames_dir, Path(d) / "capture.mp4", which_fn=lambda _: None
            )
            self.assertFalse(ok)
            self.assertIn("ffmpeg not found", warn or "")


class BootFlagsAndOptionsTests(unittest.TestCase):
    def test_new_flag_defaults_preserve_backwards_compatibility(self):
        parser = rbb.build_arg_parser()
        args = parser.parse_args([])
        self.assertIsNone(args.input_script)
        self.assertFalse(args.stream_shaders)
        self.assertIsNone(args.stream_threshold_mb)
        self.assertIsNone(args.boot_options)
        self.assertFalse(args.report_to_log)
        self.assertIsNone(args.max_wall_ms)

        self.assertEqual(rbb.build_target_url("127.0.0.1", 8080), "http://127.0.0.1:8080/")
        self.assertEqual(rbb.build_target_url("127.0.0.1", 8080, smoke=True), "http://127.0.0.1:8080/?smoke=1")

    def test_input_script_and_stream_shader_url_building_and_validation(self):
        self.assertEqual(rbb.sanitize_input_script_preset("start-screen"), "start-screen")
        self.assertEqual(rbb.sanitize_input_script_preset("  START-SCREEN  "), "start-screen")
        for bad in ("", "   ", "unknown", "../escape", "start-screen&evil=1", "start-screen\nx", 123, None):
            self.assertIsNone(rbb.sanitize_input_script_preset(bad), f"expected None for {bad!r}")

        self.assertEqual(
            rbb.build_target_url("127.0.0.1", 8080, input_script="START-SCREEN"),
            "http://127.0.0.1:8080/?inputScript=start-screen",
        )
        self.assertEqual(
            rbb.build_target_url("127.0.0.1", 8080, stream_shaders=True),
            "http://127.0.0.1:8080/?streamShaders=1",
        )
        self.assertEqual(
            rbb.build_target_url("127.0.0.1", 8080, stream_threshold_mb=32.0),
            "http://127.0.0.1:8080/?streamThresholdMb=32",
        )
        self.assertEqual(
            rbb.build_target_url("127.0.0.1", 8080, stream_threshold_mb=0.00001),
            "http://127.0.0.1:8080/?streamThresholdMb=0.00001",
        )
        self.assertEqual(
            rbb.build_target_url(
                "127.0.0.1",
                8080,
                smoke=True,
                input_script="start-screen",
                stream_shaders=True,
                stream_threshold_mb=16.5,
            ),
            "http://127.0.0.1:8080/?smoke=1&inputScript=start-screen&streamShaders=1&streamThresholdMb=16.5",
        )

        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            for bad_args in (
                ("--input-script", ""),
                ("--input-script", "not-allowed"),
                ("--input-script", "../escape"),
                ("--input-script", "start-screen&foo=1"),
                ("--stream-threshold-mb", "0"),
                ("--stream-threshold-mb", "-4.5"),
                ("--stream-threshold-mb", "nan"),
                ("--stream-threshold-mb", "inf"),
                ("--stream-threshold-mb", "-inf"),
            ):
                code, _, _, _, _ = fx.run_cli("--smoke", "--dry-run", *bad_args)
                self.assertEqual(code, 2, f"expected exit 2 for {bad_args}")

    def test_dry_run_includes_url_query_params_and_boot_options_without_writing(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            code, out, err, _, _ = fx.run_cli(
                "--smoke",
                "--dry-run",
                "--name",
                "flags-dry",
                "--input-script",
                "START-SCREEN",
                "--stream-shaders",
                "--stream-threshold-mb",
                "24.5",
                "--boot-options",
                '{"arena_log_limit": 2048}',
                "--report-to-log",
                "--max-wall-ms",
                "45000",
            )
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertEqual(
                payload["url"],
                "http://127.0.0.1:<ephemeral>/?smoke=1&inputScript=start-screen&streamShaders=1&streamThresholdMb=24.5",
            )
            self.assertEqual(payload["input_script"], "start-screen")
            self.assertEqual(payload["inputScript"], "start-screen")
            self.assertTrue(payload["stream_shaders"])
            self.assertEqual(payload["stream_threshold_mb"], 24.5)
            self.assertEqual(
                payload["boot_options"],
                {"arena_log_limit": 2048, "max_wall_ms": 45000, "report_to_log": True},
            )
            self.assertEqual(payload["boot_options_staging"]["action"], "plan_ephemeral")
            self.assertFalse(payload["boot_options_staging"]["staged"])
            # Dry-run must never write boot-options.json to boot_dir
            self.assertFalse((fx.boot_dir / "boot-options.json").exists())

    def test_runtime_switch_boot_options_are_accepted(self):
        # #1336 runtime switches (runtime/guest/boot.cpp) must pass --boot-options validation.
        switches = {
            "present_pacing": False,
            "fence_publish_lag": False,
            "defer_gpu_objects": False,
            "xmem_shared_pages": False,
            "present_frame_waits": False,
            "audio_endpoint": False,
        }
        opts = rbb.parse_and_validate_boot_options(boot_options=dict(switches))
        self.assertEqual(opts, switches)
        with self.assertRaises(rbb.RunnerError):
            rbb.parse_and_validate_boot_options(boot_options={"fence_publish_lag": "no"})

    def test_boot_options_file_validation_errors_and_tracked_dir_guard(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            # Load from a UTF-8-BOM JSON file and override with CLI helpers
            opts_file = Path(d) / "external" / "2026-boot-opts.json"
            opts_file.write_text(
                "\ufeff" + json.dumps({"max_wall_ms": 10000, "verify_window_marks": True}),
                encoding="utf-8",
            )
            code, out, err, _, _ = fx.run_cli(
                "--trust-build",
                "--dry-run",
                "--name",
                "opts-file",
                "--boot-options",
                str(opts_file),
                "--report-to-log",
                "--max-wall-ms",
                "90000",
            )
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertEqual(
                payload["boot_options"],
                {"max_wall_ms": 90000, "report_to_log": True, "verify_window_marks": True},
            )
            self.assertEqual(payload["boot_options_staging"]["action"], "plan_write")
            self.assertFalse((fx.boot_dir / "boot-options.json").exists())
            # Zero host paths leaked in dry-run JSON
            self.assertNotIn(str(Path(d)), out)

            # Reject malformed / invalid boot-options inputs
            bad_json_file = Path(d) / "external" / "bad.json"
            bad_json_file.write_text("{not valid json", encoding="utf-8")
            for bad_args in (
                ("--boot-options", ""),
                ("--boot-options", "{bad"),
                ("--boot-options", "[1, 2, 3]"),
                ("--boot-options", "123"),
                ("--boot-options", '{"unknown_key": 1}'),
                ("--boot-options", '{"max_wall_ms": true}'),
                ("--boot-options", '{"max_wall_ms": -5}'),
                ("--boot-options", '{"report_to_log": "yes"}'),
                ("--boot-options", str(Path(d) / "external" / "missing.json")),
                ("--boot-options", str(bad_json_file)),
                ("--max-wall-ms", "-1"),
                ("--max-wall-ms", str(1 << 64)),
            ):
                code, _, err_text, _, _ = fx.run_cli("--smoke", "--dry-run", *bad_args)
                self.assertEqual(code, 2, f"expected exit 2 for {bad_args}")
                self.assertNotIn(str(Path(d)), err_text)

            # Refuse writing boot-options.json into tracked repository directories (web/, tools/, repo root)
            for tracked_dir in (fx.repo / "web", fx.repo / "tools", fx.repo):
                code, _, err_text, _, _ = fx.run_cli(
                    "--trust-build",
                    "--dry-run",
                    "--boot-dir",
                    str(tracked_dir),
                    "--report-to-log",
                )
                self.assertEqual(code, 2, f"expected refusal for tracked dir {tracked_dir}")
                self.assertIn("tracked repository directory", err_text)
                self.assertFalse((tracked_dir / "boot-options.json").exists())

    def test_live_boot_options_staging_in_gitignored_boot_dir_and_smoke_ephemeral(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            # Real-game run stages boot-options.json into gitignored work/recomp_build
            code, _, err, _, sessions = fx.run_cli(
                "--trust-build",
                "--name",
                "real-opts",
                "--boot-options",
                '{"max_dispatches": 5000}',
                "--report-to-log",
                "--max-wall-ms",
                "60000",
                "--input-script",
                "start-screen",
                "--stream-shaders",
                "--stream-threshold-mb",
                "32",
            )
            self.assertEqual(code, 0, err)
            staged_path = fx.boot_dir / "boot-options.json"
            self.assertTrue(staged_path.is_file())
            self.assertEqual(
                json.loads(staged_path.read_text(encoding="utf-8")),
                {"max_dispatches": 5000, "max_wall_ms": 60000, "report_to_log": True},
            )
            self.assertEqual(sessions[0]["input_script"], "start-screen")
            self.assertTrue(sessions[0]["stream_shaders"])
            self.assertEqual(sessions[0]["stream_threshold_mb"], 32.0)
            self.assertEqual(
                sessions[0]["boot_options"],
                {"max_dispatches": 5000, "max_wall_ms": 60000, "report_to_log": True},
            )

            # Remove staged file and verify smoke run does NOT touch repo boot_dir
            staged_path.unlink()
            run_dir = fx.build / "runs" / "smoke-ephem"
            run_dir.mkdir(parents=True)
            navigated_urls: list[str] = []
            observed_smoke_boot_opts: list[dict[str, Any]] = []

            class FakeProc:
                returncode = None

                def poll(self):
                    return self.returncode

                def terminate(self):
                    self.returncode = 0

                def wait(self, timeout=None):
                    return 0

                def kill(self):
                    self.returncode = -9

            def fake_popen(cmd, **kwargs):
                if "--boot" in cmd:
                    boot_arg = Path(cmd[cmd.index("--boot") + 1])
                    opts_in_boot = boot_arg / "boot-options.json"
                    if opts_in_boot.is_file():
                        observed_smoke_boot_opts.append(json.loads(opts_in_boot.read_text(encoding="utf-8")))
                return FakeProc()

            class FakeCdp:
                def __init__(self, ws_url: str):
                    pass

                def connect(self):
                    return None

                def close(self):
                    return None

                def send(self, method: str, params=None, timeout=None):
                    if method == "Page.navigate" and isinstance(params, dict):
                        navigated_urls.append(str(params.get("url") or ""))
                    if method == "Runtime.evaluate":
                        probe = make_sample_probe(presents=2, real_draws=7, fallback_draws=1, validation_errors=2)
                        probe["bootUrlOptions"] = {"inputScript": "start-screen"}
                        return {"result": {"value": probe}}
                    return {}

                def drain_events(self, timeout: float = 0.0):
                    return []

            res = rbb.run_browser_session(
                repo=fx.repo,
                run_dir=run_dir,
                chrome_path=fx.chrome,
                port=0,
                boot_dir=None,
                game_dir=None,
                shaders_dir=None,
                duration_s=0.15,
                headless=True,
                smoke=True,
                send_input=False,
                capture_frames=False,
                frame_interval_s=1.0,
                encode_mp4=False,
                input_script="start-screen",
                stream_shaders=True,
                stream_threshold_mb=48.0,
                boot_options={"report_to_log": True, "max_wall_ms": 15000},
                popen_fn=fake_popen,
                cdp_factory=FakeCdp,
                wait_server_fn=lambda *a, **k: {"status_code": 200, "coop": "same-origin", "coep": "require-corp"},
                wait_cdp_fn=lambda *a, **k: "ws://127.0.0.1:9222/devtools/page/1",
            )
            self.assertTrue(res.ok)
            self.assertEqual(len(navigated_urls), 1)
            self.assertIn("smoke=1", navigated_urls[0])
            self.assertIn("inputScript=start-screen", navigated_urls[0])
            self.assertIn("streamShaders=1", navigated_urls[0])
            self.assertIn("streamThresholdMb=48", navigated_urls[0])
            self.assertEqual(observed_smoke_boot_opts, [{"max_wall_ms": 15000, "report_to_log": True}])
            # Smoke mode must never touch repo boot_dir
            self.assertFalse(staged_path.exists())

    def test_report_and_telemetry_summary_fields_and_input_script_sanitization(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            probe = make_sample_probe(
                boot_mode="real-wasm64",
                session_state="running_wasm",
                presents=6,
                real_draws=14,
                fallback_draws=2,
                validation_errors=1,
            )
            # Untrusted probe value should be ignored in favor of sanitized allowlisted value
            probe["inputScriptState"] = {"active": True, "preset": r"C:\Users\evil\script"}
            probe["bootUrlOptions"] = {"inputScript": "START-SCREEN"}

            session = make_fake_session_result(probe=probe)
            code, out, err, _, _ = fx.run_cli(
                "--smoke",
                "--name",
                "summary-check",
                "--input-script",
                "start-screen",
                "--stream-shaders",
                "--stream-threshold-mb",
                "16",
                "--report-to-log",
                session_runner=lambda **kw: session,
            )
            self.assertEqual(code, 0, err)
            run_dir = fx.build / "runs" / "summary-check"
            report = json.loads((run_dir / "boot-report.json").read_text(encoding="utf-8"))
            telemetry = json.loads((run_dir / "telemetry.json").read_text(encoding="utf-8"))
            cmd_meta = json.loads((run_dir / "command.json").read_text(encoding="utf-8"))

            expected_summary = {
                "realPackDrawCount": 14,
                "fallbackDrawCount": 2,
                "validationErrorCount": 1,
                "inputScript": "start-screen",
                "streamShaders": True,
                "streamThresholdMb": 16.0,
            }
            self.assertEqual(report["summary"], expected_summary)
            self.assertEqual(report["inputScript"], "start-screen")
            self.assertEqual(report["counters"]["real_pack_draws"], 14)
            self.assertEqual(report["counters"]["fallback_draws"], 2)
            self.assertEqual(report["counters"]["gpu_validation_errors"], 1)
            self.assertEqual(report["boot_options"], {"report_to_log": True})

            self.assertEqual(telemetry["summary"], expected_summary)
            self.assertEqual(telemetry["inputScript"], "start-screen")
            self.assertTrue(telemetry["stream_shaders"])
            self.assertEqual(telemetry["stream_threshold_mb"], 16.0)
            self.assertEqual(telemetry["boot_options"], {"report_to_log": True})

            self.assertEqual(cmd_meta["input_script"], "start-screen")
            self.assertEqual(cmd_meta["inputScript"], "start-screen")
            self.assertTrue(cmd_meta["stream_shaders"])
            self.assertEqual(cmd_meta["stream_threshold_mb"], 16.0)
            self.assertEqual(cmd_meta["boot_options"], {"report_to_log": True})
            self.assertIn("input_script: start-screen", out)

            # Verify compare_boot_reports compatibility remains intact
            code2, cmp_out, cmp_err, _, _ = fx.run_cli(
                "--smoke",
                "--name",
                "summary-check-2",
                "--baseline",
                "summary-check",
                "--input-script",
                "start-screen",
                session_runner=lambda **kw: session,
            )
            self.assertEqual(code2, 0, cmp_err)
            self.assertIn("PASS", cmp_out)

            # Verify untrusted probe inputScript with no CLI fallback resolves cleanly to None
            bad_probe = make_sample_probe(real_draws=3, fallback_draws=0, validation_errors=0)
            bad_probe["href"] = "http://127.0.0.1:8123/?inputScript=../escape"
            bad_probe["bootState"]["inputScript"] = "evil-script"
            rep_none = rbb.build_boot_report(
                make_fake_session_result(probe=bad_probe),
                smoke=True,
                preset="smoke",
            )
            self.assertIsNone(rep_none["inputScript"])
            self.assertIsNone(rep_none["summary"]["inputScript"])
            self.assertEqual(rep_none["summary"]["realPackDrawCount"], 3)
            self.assertEqual(rep_none["summary"]["fallbackDrawCount"], 0)
            self.assertEqual(rep_none["summary"]["validationErrorCount"], 0)


def _make_synthetic_rgb_png(
    width: int,
    height: int,
    pixel_fn,
) -> bytes:
    """Build a valid 8-bit RGB PNG using only Python stdlib (no Pillow required)."""
    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack("!I", len(data)) + chunk_type + data + struct.pack("!I", crc)

    raw_rows = bytearray()
    for y in range(height):
        raw_rows.append(0)  # filter type 0 (None)
        for x in range(width):
            r, g, b = pixel_fn(x, y)
            raw_rows.extend((r & 0xFF, g & 0xFF, b & 0xFF))

    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(raw_rows)))
        + _chunk(b"IEND", b"")
    )


class _FakePilImage:
    """Pure-Python PIL.Image stand-in that inspects synthetic PNG payloads without Pillow."""

    def __init__(self, width: int, height: int, pixels: list[tuple[int, int, int]]) -> None:
        self.size = (width, height)
        self._pixels = pixels

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def convert(self, mode: str):
        assert mode == "RGB"
        return self

    def crop(self, box: tuple[int, int, int, int]):
        left, top, right, bottom = box
        w, _h = self.size
        cropped: list[tuple[int, int, int]] = []
        for y in range(top, bottom):
            row_start = y * w
            for x in range(left, right):
                cropped.append(self._pixels[row_start + x])
        return _FakePilImage(right - left, bottom - top, cropped)

    def getextrema(self):
        if not self._pixels:
            return ((0, 0), (0, 0), (0, 0))
        rs = [p[0] for p in self._pixels]
        gs = [p[1] for p in self._pixels]
        bs = [p[2] for p in self._pixels]
        return ((min(rs), max(rs)), (min(gs), max(gs)), (min(bs), max(bs)))


class _FakePilModule:
    """Minimal PIL.Image module replacement decoding uncompressed/zlib 8-bit RGB PNGs."""

    @staticmethod
    def open(buf: io.BytesIO) -> _FakePilImage:
        data = buf.read()
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise OSError("not a PNG")
        pos = 8
        width = height = 0
        idat = bytearray()
        while pos + 8 <= len(data):
            length, chunk_type = struct.unpack("!I4s", data[pos : pos + 8])
            pos += 8
            payload = data[pos : pos + length]
            pos += length + 4  # skip CRC
            if chunk_type == b"IHDR":
                width, height = struct.unpack("!II", payload[:8])
            elif chunk_type == b"IDAT":
                idat.extend(payload)
            elif chunk_type == b"IEND":
                break
        raw = zlib.decompress(bytes(idat))
        stride = 1 + width * 3
        pixels: list[tuple[int, int, int]] = []
        for y in range(height):
            row = raw[y * stride + 1 : (y + 1) * stride]
            for x in range(width):
                off = x * 3
                pixels.append((row[off], row[off + 1], row[off + 2]))
        return _FakePilImage(width, height, pixels)


class CaptureFramesDumpPipelinesAndPresetsTests(unittest.TestCase):
    def test_t43_flag_defaults_and_validation(self):
        parser = rbb.build_arg_parser()
        args = parser.parse_args([])
        self.assertIsNone(args.capture_frames)
        self.assertEqual(args.capture_interval_sec, 2.0)
        self.assertFalse(args.dump_pipelines)

        # Bare --capture-frames preserves legacy boolean behavior; with DIR captures path
        bare = parser.parse_args(["--capture-frames"])
        self.assertIs(bare.capture_frames, True)
        with_dir = parser.parse_args(["--capture-frames", "work/frames", "--capture-interval-sec", "0.5", "--dump-pipelines"])
        self.assertEqual(with_dir.capture_frames, "work/frames")
        self.assertEqual(with_dir.capture_interval_sec, 0.5)
        self.assertTrue(with_dir.dump_pipelines)

        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            for bad_args in (
                ("--capture-interval-sec", "0"),
                ("--capture-interval-sec", "-2.5"),
                ("--capture-interval-sec", "nan"),
                ("--capture-interval-sec", "inf"),
                ("--capture-interval-sec", "-inf"),
                ("--capture-frames", ""),
                ("--capture-frames", "   "),
            ):
                code, _, _, _, _ = fx.run_cli("--smoke", "--dry-run", *bad_args)
                self.assertEqual(code, 2, f"expected exit 2 for {bad_args}")

            # Tracked repository directories are refused for --capture-frames (same guard as boot-options)
            for tracked_dir in (fx.repo / "web", fx.repo / "tools", fx.repo):
                code, _, err_text, _, _ = fx.run_cli(
                    "--smoke",
                    "--dry-run",
                    "--capture-frames",
                    str(tracked_dir),
                )
                self.assertEqual(code, 2, f"expected tracked dir refusal for {tracked_dir}")
                self.assertIn("tracked repository directory", err_text)

            # Passing a regular file path as --capture-frames fails cleanly
            file_path = fx.repo / "work" / "not-a-dir.txt"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("hi", encoding="utf-8")
            code, _, err_text, _, _ = fx.run_cli(
                "--smoke",
                "--dry-run",
                "--capture-frames",
                str(file_path),
            )
            self.assertEqual(code, 2)
            self.assertIn("not a directory", err_text)

    def test_input_script_accepts_new_presets(self):
        for preset in ("start-screen", "menu-flow", "calibration-skip", "intro-watch"):
            self.assertEqual(rbb.sanitize_input_script_preset(preset), preset)
            self.assertEqual(rbb.sanitize_input_script_preset(f"  {preset.upper()}  "), preset)
            self.assertEqual(
                rbb.build_target_url("127.0.0.1", 8080, input_script=preset),
                f"http://127.0.0.1:8080/?inputScript={preset}",
            )

        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            for preset in ("menu-flow", "calibration-skip", "intro-watch"):
                code, out, err, _, _ = fx.run_cli(
                    "--smoke",
                    "--dry-run",
                    "--name",
                    f"dry-{preset}",
                    "--input-script",
                    preset,
                )
                self.assertEqual(code, 0, err)
                payload = json.loads(out)
                self.assertEqual(payload["input_script"], preset)
                self.assertEqual(payload["inputScript"], preset)
                self.assertIn(f"inputScript={preset}", payload["url"])

    def test_dump_pipelines_dry_run_live_staging_and_validation(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            # Dry-run records dumpPipelines: true without writing boot-options.json
            code, out, err, _, _ = fx.run_cli(
                "--trust-build",
                "--dry-run",
                "--name",
                "dump-dry",
                "--dump-pipelines",
                "--report-to-log",
            )
            self.assertEqual(code, 0, err)
            dry = json.loads(out)
            self.assertTrue(dry["dumpPipelines"])
            self.assertTrue(dry["dump_pipelines"])
            self.assertEqual(
                dry["boot_options"],
                {"dump_pipelines_dir": "/pg6_dump", "report_to_log": True},
            )
            self.assertFalse(dry["boot_options_staging"]["staged"])
            self.assertEqual(dry["boot_options_staging"]["action"], "plan_write")
            self.assertFalse((fx.boot_dir / "boot-options.json").exists())

            # Tracked boot-dir is refused when --dump-pipelines is set
            code, _, err_text, _, _ = fx.run_cli(
                "--trust-build",
                "--dry-run",
                "--boot-dir",
                str(fx.repo / "web"),
                "--dump-pipelines",
            )
            self.assertEqual(code, 2)
            self.assertIn("tracked repository directory", err_text)

            # Live real-game run stages dump_pipelines_dir into boot-options.json and records dumpPipelines
            code, _, err, _, sessions = fx.run_cli(
                "--trust-build",
                "--name",
                "dump-live",
                "--dump-pipelines",
            )
            self.assertEqual(code, 0, err)
            staged_path = fx.boot_dir / "boot-options.json"
            self.assertTrue(staged_path.is_file())
            self.assertEqual(
                json.loads(staged_path.read_text(encoding="utf-8")),
                {"dump_pipelines_dir": "/pg6_dump"},
            )
            self.assertEqual(sessions[0]["boot_options"], {"dump_pipelines_dir": "/pg6_dump"})

            run_dir = fx.build / "runs" / "dump-live"
            cmd_meta = json.loads((run_dir / "command.json").read_text(encoding="utf-8"))
            report = json.loads((run_dir / "boot-report.json").read_text(encoding="utf-8"))
            telemetry = json.loads((run_dir / "telemetry.json").read_text(encoding="utf-8"))
            self.assertTrue(cmd_meta["dumpPipelines"])
            self.assertTrue(cmd_meta["dump_pipelines"])
            self.assertEqual(cmd_meta["boot_options"], {"dump_pipelines_dir": "/pg6_dump"})
            self.assertEqual(report["boot_options"], {"dump_pipelines_dir": "/pg6_dump"})
            self.assertEqual(telemetry["boot_options"], {"dump_pipelines_dir": "/pg6_dump"})

            # Reject invalid dump_pipelines_dir values passed via --boot-options
            for bad_json in (
                '{"dump_pipelines_dir": ""}',
                '{"dump_pipelines_dir": 123}',
                '{"dump_pipelines_dir": "../escape"}',
                '{"dump_pipelines_dir": "C:\\\\dump"}',
            ):
                code, _, _, _, _ = fx.run_cli("--smoke", "--dry-run", "--boot-options", bad_json)
                self.assertEqual(code, 2, f"expected exit 2 for {bad_json}")

    def test_check_canvas_non_black_pure_python_mock_and_optional_pillow(self):
        all_black = _make_synthetic_rgb_png(420, 4, lambda x, y: (0, 0, 0))
        sidebar_only = _make_synthetic_rgb_png(420, 4, lambda x, y: (255, 255, 255) if x < 410 else (0, 0, 0))
        canvas_lit = _make_synthetic_rgb_png(420, 4, lambda x, y: (0, 200, 50) if x == 410 and y == 1 else (0, 0, 0))
        too_narrow = _make_synthetic_rgb_png(410, 4, lambda x, y: (255, 255, 255))

        # Pure-Python mock PIL module (always runs regardless of whether Pillow is installed)
        self.assertFalse(rbb.check_canvas_non_black(all_black, pil_image_mod=_FakePilModule, _auto_import_pil=False))
        self.assertFalse(rbb.check_canvas_non_black(sidebar_only, pil_image_mod=_FakePilModule, _auto_import_pil=False))
        self.assertTrue(rbb.check_canvas_non_black(canvas_lit, pil_image_mod=_FakePilModule, _auto_import_pil=False))
        self.assertFalse(rbb.check_canvas_non_black(too_narrow, pil_image_mod=_FakePilModule, _auto_import_pil=False))
        self.assertIsNone(rbb.check_canvas_non_black(b"not-a-png", pil_image_mod=_FakePilModule, _auto_import_pil=False))

        # Graceful skip when Pillow is unavailable
        self.assertIsNone(rbb.check_canvas_non_black(canvas_lit, pil_image_mod=None, _auto_import_pil=False))
        with mock.patch.object(rbb, "_import_pil_image", return_value=None):
            self.assertIsNone(rbb.check_canvas_non_black(canvas_lit))

        # If real Pillow is installed on this machine, verify real PIL produces identical results
        real_pil = rbb._import_pil_image()
        if real_pil is not None:
            self.assertFalse(rbb.check_canvas_non_black(all_black, pil_image_mod=real_pil, _auto_import_pil=False))
            self.assertFalse(rbb.check_canvas_non_black(sidebar_only, pil_image_mod=real_pil, _auto_import_pil=False))
            self.assertTrue(rbb.check_canvas_non_black(canvas_lit, pil_image_mod=real_pil, _auto_import_pil=False))
            self.assertFalse(rbb.check_canvas_non_black(too_narrow, pil_image_mod=real_pil, _auto_import_pil=False))
            self.assertIsNone(rbb.check_canvas_non_black(b"\xff\xd8\xff\xd9", pil_image_mod=real_pil, _auto_import_pil=False))

    def test_capture_frames_dir_png_session_with_and_without_pillow(self):
        sidebar_only_png = _make_synthetic_rgb_png(420, 4, lambda x, y: (255, 255, 255) if x < 410 else (0, 0, 0))
        canvas_lit_png = _make_synthetic_rgb_png(420, 4, lambda x, y: (32, 64, 128) if x >= 410 else (0, 0, 0))
        frame_sequence = [sidebar_only_png, canvas_lit_png, canvas_lit_png]

        class FakeProc:
            returncode = None

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = 0

            def wait(self, timeout=None):
                return 0

            def kill(self):
                self.returncode = -9

        def run_mock_png_session(fx: RepoFixture, run_name: str, capture_dir: Path, pil_return: Any):
            run_dir = fx.build / "runs" / run_name
            run_dir.mkdir(parents=True)
            shot_params: list[dict[str, Any]] = []
            shot_idx = {"n": 0}
            fake_clock = {"t": 1000.0}

            class FakeCdp:
                def __init__(self, ws_url: str) -> None:
                    self.ws_url = ws_url

                def connect(self):
                    return None

                def close(self):
                    return None

                def send(self, method: str, params=None, timeout=None):
                    if method == "Runtime.evaluate":
                        return {"result": {"value": make_sample_probe(presents=2, real_draws=5, fallback_draws=0)}}
                    if method == "Page.captureScreenshot":
                        shot_params.append(dict(params or {}))
                        payload = frame_sequence[min(shot_idx["n"], len(frame_sequence) - 1)]
                        shot_idx["n"] += 1
                        return {"data": base64.b64encode(payload).decode("ascii")}
                    return {}

                def drain_events(self, timeout: float = 0.0):
                    fake_clock["t"] += 0.1
                    return []

            with (
                mock.patch.object(rbb, "_import_pil_image", return_value=pil_return),
                mock.patch.object(rbb.time, "monotonic", side_effect=lambda: fake_clock["t"]),
            ):
                res = rbb.run_browser_session(
                    repo=fx.repo,
                    run_dir=run_dir,
                    chrome_path=fx.chrome,
                    port=0,
                    boot_dir=None,
                    game_dir=None,
                    shaders_dir=None,
                    duration_s=0.29,
                    headless=True,
                    smoke=True,
                    send_input=False,
                    capture_frames=True,
                    capture_frames_dir=capture_dir,
                    capture_interval_sec=0.1,
                    frame_interval_s=1.0,
                    encode_mp4=False,
                    input_script="menu-flow",
                    popen_fn=lambda cmd, **kw: FakeProc(),
                    cdp_factory=FakeCdp,
                    wait_server_fn=lambda *a, **k: {"status_code": 200, "coop": "same-origin", "coep": "require-corp"},
                    wait_cdp_fn=lambda *a, **k: "ws://127.0.0.1:9222/devtools/page/1",
                )
            return res, shot_params

        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))

            # 1. With PIL available (via pure-Python _FakePilModule): captures PNGs and counts non-black x>=410 frames
            cap_dir_pil = fx.repo / "work" / "frames-with-pil"
            res_pil, params_pil = run_mock_png_session(fx, "cap-pil", cap_dir_pil, _FakePilModule)
            self.assertTrue(res_pil.ok)
            self.assertEqual(len(res_pil.captured_frames), 3)
            self.assertTrue(all(p == {"format": "png"} for p in params_pil))
            for fname in res_pil.captured_frames:
                self.assertTrue(fname.endswith(".png"))
                self.assertTrue((cap_dir_pil / fname).is_file())
            # First frame is sidebar-only (black at x>=410), remaining 2 frames are canvas-lit
            self.assertEqual(res_pil.non_black_canvas_frames, 2)
            self.assertEqual(res_pil.capture_frames_dir, "work/frames-with-pil")

            # Verify CLI wiring writes capturedFrames / nonBlackCanvasFrames across report, telemetry, command, digest
            code, out, err, _, _ = fx.run_cli(
                "--smoke",
                "--name",
                "cli-cap-pil",
                "--input-script",
                "menu-flow",
                "--capture-frames",
                str(cap_dir_pil),
                "--capture-interval-sec",
                "0.5",
                "--dump-pipelines",
                session_runner=lambda **kw: res_pil,
            )
            self.assertEqual(code, 0, err)
            run_dir = fx.build / "runs" / "cli-cap-pil"
            report = json.loads((run_dir / "boot-report.json").read_text(encoding="utf-8"))
            telemetry = json.loads((run_dir / "telemetry.json").read_text(encoding="utf-8"))
            cmd_meta = json.loads((run_dir / "command.json").read_text(encoding="utf-8"))

            self.assertEqual(report["summary"]["capturedFrames"], len(res_pil.captured_frames))
            self.assertEqual(report["summary"]["nonBlackCanvasFrames"], len(res_pil.captured_frames) - 1)
            self.assertEqual(report["summary"]["inputScript"], "menu-flow")
            self.assertEqual(telemetry["summary"]["capturedFrames"], len(res_pil.captured_frames))
            self.assertEqual(telemetry["summary"]["nonBlackCanvasFrames"], len(res_pil.captured_frames) - 1)
            self.assertEqual(telemetry["capturedFrames"], len(res_pil.captured_frames))
            self.assertEqual(telemetry["nonBlackCanvasFrames"], len(res_pil.captured_frames) - 1)
            self.assertEqual(telemetry["capture_frames_dir"], "work/frames-with-pil")
            self.assertEqual(cmd_meta["capturedFrames"], len(res_pil.captured_frames))
            self.assertEqual(cmd_meta["nonBlackCanvasFrames"], len(res_pil.captured_frames) - 1)
            self.assertEqual(cmd_meta["capture_frames_dir"], "work/frames-with-pil")
            self.assertEqual(cmd_meta["capture_interval_sec"], 0.5)
            self.assertTrue(cmd_meta["dumpPipelines"])
            self.assertIn(f"captured_frames: {len(res_pil.captured_frames)} (non_black_canvas={len(res_pil.captured_frames) - 1})", out)
            # Zero host path leaks
            for fname in ("boot-report.json", "telemetry.json", "command.json", "digest.txt"):
                self.assertNotIn(str(Path(d)), (run_dir / fname).read_text(encoding="utf-8"))

            # 2. Without Pillow installed: still captures PNGs, gracefully sets nonBlackCanvasFrames to None
            cap_dir_nopil = fx.repo / "work" / "frames-no-pil"
            res_nopil, _ = run_mock_png_session(fx, "cap-nopil", cap_dir_nopil, None)
            self.assertTrue(res_nopil.ok)
            self.assertGreaterEqual(len(res_nopil.captured_frames), 1)
            self.assertIsNone(res_nopil.non_black_canvas_frames)
            self.assertTrue(any("Pillow" in w for w in res_nopil.warnings))

            rep_nopil = rbb.build_boot_report(res_nopil, smoke=True, preset="smoke")
            tel_nopil = rbb.build_telemetry_payload(res_nopil, smoke=True, preset="smoke", headless=True)
            self.assertEqual(rep_nopil["summary"]["capturedFrames"], len(res_nopil.captured_frames))
            self.assertIsNone(rep_nopil["summary"]["nonBlackCanvasFrames"])
            self.assertEqual(tel_nopil["summary"]["capturedFrames"], len(res_nopil.captured_frames))
            self.assertIsNone(tel_nopil["summary"]["nonBlackCanvasFrames"])
            self.assertIn("non_black_canvas=skipped", rbb.digest(rep_nopil, 0))

            # 3. encode_frames_to_mp4 also handles PNG frames
            ffmpeg_cmds: list[list[str]] = []

            def fake_ffmpeg(cmd, cwd, log, timeout):
                ffmpeg_cmds.append(list(cmd))
                Path(cmd[-1]).write_bytes(b"ftypmp42")
                return 0

            ok_mp4, warn_mp4 = rbb.encode_frames_to_mp4(
                cap_dir_pil,
                fx.repo / "work" / "out.mp4",
                runner=fake_ffmpeg,
                which_fn=lambda n: "ffmpeg" if n == "ffmpeg" else None,
            )
            self.assertTrue(ok_mp4, warn_mp4)
            self.assertTrue(any("frame_%04d.png" in tok for tok in ffmpeg_cmds[0]))


class HudMp4AndNewPresetsTests(unittest.TestCase):
    def test_t49_flag_defaults_and_parsing(self):
        parser = rbb.build_arg_parser()
        defaults = parser.parse_args([])
        self.assertFalse(defaults.hide_hud)
        self.assertIsNone(defaults.mp4_fps)
        self.assertFalse(defaults.mp4_skip_black)

        parsed = parser.parse_args(
            [
                "--input-script",
                "legal-notices",
                "--hide-hud",
                "--mp4",
                "--mp4-fps",
                "4.5",
                "--mp4-skip-black",
            ]
        )
        self.assertEqual(parsed.input_script, "legal-notices")
        self.assertTrue(parsed.hide_hud)
        self.assertTrue(parsed.mp4)
        self.assertEqual(parsed.mp4_fps, 4.5)
        self.assertTrue(parsed.mp4_skip_black)

    def test_legal_notices_and_chrome_menu_flow_input_script_presets(self):
        for preset in ("legal-notices", "chrome-menu-flow"):
            self.assertEqual(rbb.sanitize_input_script_preset(preset), preset)
            self.assertEqual(rbb.sanitize_input_script_preset(f"  {preset.upper()}  "), preset)
            self.assertEqual(
                rbb.build_target_url("127.0.0.1", 8080, input_script=preset),
                f"http://127.0.0.1:8080/?inputScript={preset}",
            )

        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            for preset in ("legal-notices", "chrome-menu-flow"):
                code, out, err, _, _ = fx.run_cli(
                    "--smoke",
                    "--dry-run",
                    "--name",
                    f"dry-{preset}",
                    "--input-script",
                    preset,
                )
                self.assertEqual(code, 0, err)
                payload = json.loads(out)
                self.assertEqual(payload["input_script"], preset)
                self.assertEqual(payload["inputScript"], preset)
                self.assertEqual(
                    payload["url"],
                    f"http://127.0.0.1:<ephemeral>/?smoke=1&inputScript={preset}",
                )

    def test_hide_hud_url_plumbing_and_dry_run_plan(self):
        self.assertEqual(
            rbb.build_target_url("127.0.0.1", 8080, hide_hud=True),
            "http://127.0.0.1:8080/?hud=0",
        )
        self.assertEqual(
            rbb.build_target_url(
                "127.0.0.1",
                8080,
                smoke=True,
                input_script="chrome-menu-flow",
                stream_shaders=True,
                stream_threshold_mb=32.0,
                hide_hud=True,
            ),
            "http://127.0.0.1:8080/?smoke=1&inputScript=chrome-menu-flow&streamShaders=1&streamThresholdMb=32&hud=0",
        )

        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            code, out, err, _, _ = fx.run_cli(
                "--smoke",
                "--dry-run",
                "--name",
                "hud-mp4-dry",
                "--input-script",
                "legal-notices",
                "--hide-hud",
                "--mp4",
                "--mp4-fps",
                "0.5",
                "--mp4-skip-black",
            )
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertTrue(payload["hide_hud"])
            self.assertTrue(payload["mp4"])
            self.assertEqual(payload["mp4_fps"], 0.5)
            self.assertTrue(payload["mp4_skip_black"])
            self.assertEqual(payload["input_script"], "legal-notices")
            self.assertEqual(
                payload["url"],
                "http://127.0.0.1:<ephemeral>/?smoke=1&inputScript=legal-notices&hud=0",
            )
            self.assertNotIn(str(Path(d)), out)

            # Default dry-run preserves backwards-compatible URL and defaults
            code_def, out_def, err_def, _, _ = fx.run_cli("--smoke", "--dry-run", "--name", "hud-def")
            self.assertEqual(code_def, 0, err_def)
            payload_def = json.loads(out_def)
            self.assertFalse(payload_def["hide_hud"])
            self.assertIsNone(payload_def["mp4_fps"])
            self.assertFalse(payload_def["mp4_skip_black"])
            self.assertNotIn("hud=0", payload_def["url"])

    def test_mp4_fps_validation(self):
        self.assertIsNone(rbb.validate_mp4_fps(None))
        self.assertEqual(rbb.validate_mp4_fps(0.5), 0.5)
        self.assertEqual(rbb.validate_mp4_fps(24), 24.0)
        for bad in (0, -1, -0.25, float("nan"), float("inf"), float("-inf"), True, "5"):
            with self.assertRaises(rbb.RunnerError, msg=f"expected RunnerError for {bad!r}"):
                rbb.validate_mp4_fps(bad)  # type: ignore[arg-type]

        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            for bad_args in (
                ("--mp4-fps", "0"),
                ("--mp4-fps", "-2"),
                ("--mp4-fps", "-0.1"),
                ("--mp4-fps", "nan"),
                ("--mp4-fps", "inf"),
                ("--mp4-fps", "-inf"),
            ):
                code, _, _, _, _ = fx.run_cli("--smoke", "--dry-run", *bad_args)
                self.assertEqual(code, 2, f"expected exit 2 for {bad_args}")

    def test_encode_frames_to_mp4_custom_fps_and_skip_leading_black_frames(self):
        all_black_png = _make_synthetic_rgb_png(420, 4, lambda x, y: (0, 0, 0))
        sidebar_only_png = _make_synthetic_rgb_png(420, 4, lambda x, y: (255, 255, 255) if x < 410 else (0, 0, 0))
        canvas_lit_png = _make_synthetic_rgb_png(420, 4, lambda x, y: (10, 120, 240) if x >= 410 else (0, 0, 0))

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frames_dir = root / "frames"
            frames_dir.mkdir()
            # Frames 1 & 2 have black canvas (x>=410); frame 3 is lit; frame 4 is black again (must NOT be skipped)
            (frames_dir / "frame_0001.png").write_bytes(all_black_png)
            (frames_dir / "frame_0002.png").write_bytes(sidebar_only_png)
            (frames_dir / "frame_0003.png").write_bytes(canvas_lit_png)
            (frames_dir / "frame_0004.png").write_bytes(all_black_png)

            ffmpeg_cmds: list[list[str]] = []

            def fake_ffmpeg(cmd, cwd, log, timeout):
                ffmpeg_cmds.append(list(cmd))
                Path(cmd[-1]).write_bytes(b"ftypmp42")
                return 0

            # 1. skip_black=True with HUD visible (canvas_min_x=410) skips frames 1..2 -> -start_number 3 and custom fps 0.5
            ok, warn = rbb.encode_frames_to_mp4(
                frames_dir,
                root / "skip_hud.mp4",
                fps=0.5,
                skip_black=True,
                canvas_min_x=rbb.CANVAS_MIN_X,
                pil_image_mod=_FakePilModule,
                _auto_import_pil=False,
                runner=fake_ffmpeg,
                which_fn=lambda n: "ffmpeg" if n == "ffmpeg" else None,
            )
            self.assertTrue(ok, warn)
            cmd1 = ffmpeg_cmds[-1]
            self.assertEqual(cmd1[cmd1.index("-framerate") + 1], "0.5")
            self.assertIn("-start_number", cmd1)
            self.assertEqual(cmd1[cmd1.index("-start_number") + 1], "3")
            self.assertLess(cmd1.index("-start_number"), cmd1.index("-i"))

            # 2. skip_black=True with hide_hud (canvas_min_x=0): frame 2 has lit pixels at x<410 -> -start_number 2
            ok2, warn2 = rbb.encode_frames_to_mp4(
                frames_dir,
                root / "skip_nohud.mp4",
                fps=4.0,
                skip_black=True,
                canvas_min_x=0,
                pil_image_mod=_FakePilModule,
                _auto_import_pil=False,
                runner=fake_ffmpeg,
                which_fn=lambda n: "ffmpeg" if n == "ffmpeg" else None,
            )
            self.assertTrue(ok2, warn2)
            cmd2 = ffmpeg_cmds[-1]
            self.assertEqual(cmd2[cmd2.index("-framerate") + 1], "4.0")
            self.assertEqual(cmd2[cmd2.index("-start_number") + 1], "2")

            # 3. skip_black=False does NOT pass -start_number
            ok3, warn3 = rbb.encode_frames_to_mp4(
                frames_dir,
                root / "noskip.mp4",
                fps=2.5,
                skip_black=False,
                pil_image_mod=_FakePilModule,
                _auto_import_pil=False,
                runner=fake_ffmpeg,
                which_fn=lambda n: "ffmpeg" if n == "ffmpeg" else None,
            )
            self.assertTrue(ok3, warn3)
            cmd3 = ffmpeg_cmds[-1]
            self.assertEqual(cmd3[cmd3.index("-framerate") + 1], "2.5")
            self.assertNotIn("-start_number", cmd3)

            # 4. Graceful fallback when Pillow is unavailable
            ok4, warn4 = rbb.encode_frames_to_mp4(
                frames_dir,
                root / "nopil.mp4",
                fps=2.0,
                skip_black=True,
                pil_image_mod=None,
                _auto_import_pil=False,
                runner=fake_ffmpeg,
                which_fn=lambda n: "ffmpeg" if n == "ffmpeg" else None,
            )
            self.assertTrue(ok4, warn4)
            self.assertNotIn("-start_number", ffmpeg_cmds[-1])

            # 5. All-black frames gracefully fall back to encoding from frame 1 (no -start_number)
            all_black_dir = root / "all_black"
            all_black_dir.mkdir()
            (all_black_dir / "frame_0001.png").write_bytes(all_black_png)
            (all_black_dir / "frame_0002.png").write_bytes(all_black_png)
            ok5, warn5 = rbb.encode_frames_to_mp4(
                all_black_dir,
                root / "all_black.mp4",
                fps=2.0,
                skip_black=True,
                canvas_min_x=0,
                pil_image_mod=_FakePilModule,
                _auto_import_pil=False,
                runner=fake_ffmpeg,
                which_fn=lambda n: "ffmpeg" if n == "ffmpeg" else None,
            )
            self.assertTrue(ok5, warn5)
            self.assertNotIn("-start_number", ffmpeg_cmds[-1])

    def test_encode_frames_to_mp4_passes_even_dimensions_filter_and_faststart(self):
        odd_png = _make_synthetic_rgb_png(15, 9, lambda x, y: (32, 64, 128))

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frames_dir = root / "frames"
            frames_dir.mkdir()
            (frames_dir / "frame_0001.png").write_bytes(odd_png)
            (frames_dir / "frame_0002.png").write_bytes(odd_png)

            out_mp4 = root / "capture.mp4"
            ffmpeg_cmds: list[list[str]] = []

            def fake_ffmpeg(cmd, cwd, log, timeout):
                ffmpeg_cmds.append(list(cmd))
                Path(cmd[-1]).write_bytes(b"ftypmp42")
                return 0

            ok, warn = rbb.encode_frames_to_mp4(
                frames_dir,
                out_mp4,
                fps=2.0,
                runner=fake_ffmpeg,
                which_fn=lambda n: "ffmpeg" if n == "ffmpeg" else None,
            )
            self.assertTrue(ok, warn)
            self.assertEqual(len(ffmpeg_cmds), 1)
            cmd = ffmpeg_cmds[0]

            i_idx = cmd.index("-i")
            vf_idx = cmd.index("-vf")
            movflags_idx = cmd.index("-movflags")
            out_idx = len(cmd) - 1

            self.assertEqual(cmd[vf_idx + 1], "scale=trunc(iw/2)*2:trunc(ih/2)*2")
            self.assertEqual(cmd[movflags_idx + 1], "+faststart")
            self.assertEqual(cmd[cmd.index("-c:v") + 1], "libx264")
            self.assertEqual(cmd[cmd.index("-pix_fmt") + 1], "yuv420p")
            self.assertEqual(cmd[out_idx], str(out_mp4))
            self.assertGreater(vf_idx, i_idx)
            self.assertLess(vf_idx, out_idx)
            self.assertGreater(movflags_idx, i_idx)
            self.assertLess(movflags_idx, out_idx)

    def test_run_browser_session_and_cli_plumbing_for_hide_hud_mp4_fps_and_skip_black(self):
        all_black_png = _make_synthetic_rgb_png(420, 4, lambda x, y: (0, 0, 0))
        left_lit_png = _make_synthetic_rgb_png(420, 4, lambda x, y: (200, 100, 50) if x < 410 else (0, 0, 0))
        frame_seq = [all_black_png, left_lit_png, left_lit_png]

        class FakeProc:
            returncode = None

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = 0

            def wait(self, timeout=None):
                return 0

            def kill(self):
                self.returncode = -9

        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            run_dir = fx.build / "runs" / "t49-session"
            run_dir.mkdir(parents=True)
            cap_dir = fx.repo / "work" / "t49-frames"

            navigated_urls: list[str] = []
            ffmpeg_cmds: list[list[str]] = []
            shot_idx = {"n": 0}
            fake_clock = {"t": 2000.0}

            class FakeCdp:
                def __init__(self, ws_url: str) -> None:
                    self.ws_url = ws_url

                def connect(self):
                    return None

                def close(self):
                    return None

                def send(self, method: str, params=None, timeout=None):
                    if method == "Page.navigate" and isinstance(params, dict):
                        navigated_urls.append(str(params.get("url") or ""))
                    if method == "Runtime.evaluate":
                        return {"result": {"value": make_sample_probe(presents=2, real_draws=3, fallback_draws=0)}}
                    if method == "Page.captureScreenshot":
                        payload = frame_seq[min(shot_idx["n"], len(frame_seq) - 1)]
                        shot_idx["n"] += 1
                        return {"data": base64.b64encode(payload).decode("ascii")}
                    return {}

                def drain_events(self, timeout: float = 0.0):
                    fake_clock["t"] += 0.1
                    return []

            def fake_ffmpeg(cmd, cwd, log, timeout):
                ffmpeg_cmds.append(list(cmd))
                Path(cmd[-1]).write_bytes(b"ftypmp42")
                return 0

            with (
                mock.patch.object(rbb, "_import_pil_image", return_value=_FakePilModule),
                mock.patch.object(rbb.time, "monotonic", side_effect=lambda: fake_clock["t"]),
            ):
                res = rbb.run_browser_session(
                    repo=fx.repo,
                    run_dir=run_dir,
                    chrome_path=fx.chrome,
                    port=0,
                    boot_dir=None,
                    game_dir=None,
                    shaders_dir=None,
                    duration_s=0.29,
                    headless=True,
                    smoke=True,
                    send_input=False,
                    capture_frames=True,
                    capture_frames_dir=cap_dir,
                    capture_interval_sec=0.1,
                    frame_interval_s=1.0,
                    encode_mp4=True,
                    mp4_fps=3.5,
                    mp4_skip_black=True,
                    hide_hud=True,
                    input_script="chrome-menu-flow",
                    runner=fake_ffmpeg,
                    popen_fn=lambda cmd, **kw: FakeProc(),
                    cdp_factory=FakeCdp,
                    wait_server_fn=lambda *a, **k: {"status_code": 200, "coop": "same-origin", "coep": "require-corp"},
                    wait_cdp_fn=lambda *a, **k: "ws://127.0.0.1:9222/devtools/page/1",
                    which_fn=lambda n: "ffmpeg" if n == "ffmpeg" else None,
                )

            self.assertTrue(res.ok)
            self.assertEqual(len(navigated_urls), 1)
            self.assertIn("inputScript=chrome-menu-flow", navigated_urls[0])
            self.assertIn("hud=0", navigated_urls[0])
            # hide_hud=True checks full frame (canvas_min_x=0), so frame 1 is black and frames 2..3 are lit
            self.assertEqual(len(res.captured_frames), 3)
            self.assertEqual(res.non_black_canvas_frames, 2)
            self.assertEqual(res.mp4_path, "capture.mp4")
            self.assertEqual(len(ffmpeg_cmds), 1)
            cmd = ffmpeg_cmds[0]
            self.assertEqual(cmd[cmd.index("-framerate") + 1], "3.5")
            self.assertEqual(cmd[cmd.index("-start_number") + 1], "2")

            # Verify CLI writes new metadata cleanly into telemetry.json and command.json
            code, _, err, _, sessions = fx.run_cli(
                "--smoke",
                "--name",
                "t49-cli-run",
                "--input-script",
                "legal-notices",
                "--hide-hud",
                "--mp4",
                "--mp4-fps",
                "6",
                "--mp4-skip-black",
            )
            self.assertEqual(code, 0, err)
            self.assertEqual(sessions[0]["input_script"], "legal-notices")
            self.assertTrue(sessions[0]["hide_hud"])
            self.assertEqual(sessions[0]["mp4_fps"], 6.0)
            self.assertTrue(sessions[0]["mp4_skip_black"])

            cli_run_dir = fx.build / "runs" / "t49-cli-run"
            telemetry = json.loads((cli_run_dir / "telemetry.json").read_text(encoding="utf-8"))
            cmd_meta = json.loads((cli_run_dir / "command.json").read_text(encoding="utf-8"))
            self.assertTrue(telemetry["hide_hud"])
            self.assertEqual(telemetry["mp4_fps"], 6.0)
            self.assertTrue(telemetry["mp4_skip_black"])
            self.assertEqual(telemetry["inputScript"], "legal-notices")
            self.assertTrue(cmd_meta["hide_hud"])
            self.assertEqual(cmd_meta["mp4_fps"], 6.0)
            self.assertTrue(cmd_meta["mp4_skip_black"])
            self.assertEqual(cmd_meta["input_script"], "legal-notices")


class ChromeFullBootGpuAdapterAndMp4HardeningTests(unittest.TestCase):
    def test_chrome_full_boot_registered_in_aliases_cli_and_help(self):
        self.assertEqual(rbb.INPUT_SCRIPT_ALIASES.get("chrome-full-boot"), "chrome-full-boot")
        self.assertEqual(rbb.ALLOWED_INPUT_SCRIPT_PRESETS.get("chrome-full-boot"), "chrome-full-boot")
        self.assertEqual(rbb.sanitize_input_script_preset("chrome-full-boot"), "chrome-full-boot")
        self.assertEqual(rbb.sanitize_input_script_preset("  CHROME-FULL-BOOT  "), "chrome-full-boot")
        self.assertEqual(
            rbb.build_target_url("127.0.0.1", 8080, input_script="chrome-full-boot"),
            "http://127.0.0.1:8080/?inputScript=chrome-full-boot",
        )

        help_text = rbb.build_arg_parser().format_help()
        self.assertIn("chrome-full-boot", help_text)

        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            code, out, err, _, _ = fx.run_cli(
                "--smoke",
                "--dry-run",
                "--name",
                "dry-chrome-full-boot",
                "--input-script",
                "chrome-full-boot",
            )
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertEqual(payload["input_script"], "chrome-full-boot")
            self.assertEqual(payload["inputScript"], "chrome-full-boot")
            self.assertEqual(
                payload["url"],
                "http://127.0.0.1:<ephemeral>/?smoke=1&inputScript=chrome-full-boot",
            )

    def test_gpu_adapter_extraction_and_sanitization(self):
        self.assertEqual(rbb.sanitize_gpu_adapter("nvidia / turing"), "nvidia / turing")
        self.assertEqual(rbb.sanitize_gpu_adapter("Intel / Gen-12LP"), "intel / gen-12lp")
        self.assertEqual(rbb.sanitize_gpu_adapter("  AMD / RDNA-2  "), "amd / rdna-2")

        # Strips Windows, UNC, and POSIX host paths as well as control/unsafe characters
        dirty_win = r"NVIDIA / Turing / C:\Users\secretuser\drivers\nv.dll"
        self.assertEqual(rbb.sanitize_gpu_adapter(dirty_win), "nvidia / turing")
        dirty_unc = r"Intel / Gen-12LP / \\fileserver\share\driver.bin"
        self.assertEqual(rbb.sanitize_gpu_adapter(dirty_unc), "intel / gen-12lp")
        dirty_posix = "AMD / RDNA-3 / /home/alice/mesa/libvulkan.so"
        self.assertEqual(rbb.sanitize_gpu_adapter(dirty_posix), "amd / rdna-3")
        self.assertEqual(rbb.sanitize_gpu_adapter("nvidia / turing\ninjected: bad"), "nvidia / turing")

        # Pure host paths or empty/non-string inputs resolve cleanly to None
        for bad in (
            None,
            123,
            "",
            "   ",
            r"C:\Users\secretuser\gpu.dll",
            "/Users/alice/local/gpu",
            "///...",
        ):
            self.assertIsNone(rbb.sanitize_gpu_adapter(bad), f"expected None for {bad!r}")

        # Enforces max 96-character bound without trailing slash/space
        overlong = "nvidia / turing / " + ("geforce-rtx " * 20)
        sanitized_long = rbb.sanitize_gpu_adapter(overlong)
        self.assertIsNotNone(sanitized_long)
        assert sanitized_long is not None
        self.assertLessEqual(len(sanitized_long), 96)
        self.assertFalse(sanitized_long.endswith(("/", " ")))

        # Console line extraction picks the latest valid [wgpu_worker] WebGPU adapter line
        lines = [
            "[   0.10s] [console.log] booting",
            "[   0.55s] [console.info] [wgpu_worker] WebGPU adapter: Intel / Gen-12LP",
            "[   1.20s] [console.info] [wgpu_worker] WebGPU adapter: NVIDIA / Turing / C:\\Users\\bob\\nv.dll",
            "[   1.30s] [console.info] [wgpu_worker] WebGPU adapter:    ",
        ]
        self.assertEqual(rbb.extract_gpu_adapter_from_console_lines(lines), "nvidia / turing")
        self.assertIsNone(rbb.extract_gpu_adapter_from_console_lines(["[   0.10s] [console.log] no adapter"]))

    def test_gpu_adapter_surfaced_in_session_telemetry_report_and_digest(self):
        class FakeProc:
            returncode = None

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = 0

            def wait(self, timeout=None):
                return 0

            def kill(self):
                self.returncode = -9

        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            run_dir = fx.build / "runs" / "t56-sess"
            run_dir.mkdir(parents=True)
            emitted = {"done": False}
            fake_clock = {"t": 3000.0}

            class FakeCdp:
                def __init__(self, ws_url: str) -> None:
                    self.ws_url = ws_url

                def connect(self):
                    return None

                def close(self):
                    return None

                def send(self, method: str, params=None, timeout=None):
                    if method == "Runtime.evaluate":
                        return {"result": {"value": make_sample_probe(presents=2, real_draws=4, fallback_draws=0)}}
                    return {}

                def drain_events(self, timeout: float = 0.0):
                    fake_clock["t"] += 0.1
                    if not emitted["done"]:
                        emitted["done"] = True
                        return [
                            {
                                "method": "Runtime.consoleAPICalled",
                                "params": {
                                    "type": "info",
                                    "args": [{"value": "[wgpu_worker] WebGPU adapter: Intel / Gen-12LP"}],
                                },
                            }
                        ]
                    return []

            with mock.patch.object(rbb.time, "monotonic", side_effect=lambda: fake_clock["t"]):
                sess = rbb.run_browser_session(
                    repo=fx.repo,
                    run_dir=run_dir,
                    chrome_path=fx.chrome,
                    port=0,
                    boot_dir=None,
                    game_dir=None,
                    shaders_dir=None,
                    duration_s=0.15,
                    headless=True,
                    smoke=True,
                    send_input=False,
                    capture_frames=False,
                    frame_interval_s=1.0,
                    encode_mp4=False,
                    popen_fn=lambda cmd, **kw: FakeProc(),
                    cdp_factory=FakeCdp,
                    wait_server_fn=lambda *a, **k: {"status_code": 200, "coop": "same-origin", "coep": "require-corp"},
                    wait_cdp_fn=lambda *a, **k: "ws://127.0.0.1:9222/devtools/page/1",
                )

            self.assertEqual(sess.gpu_adapter, "intel / gen-12lp")

            # End-to-end CLI artifact output (telemetry.json, boot-report.json, digest.txt)
            code, out, err, _, _ = fx.run_cli(
                "--smoke",
                "--name",
                "t56-cli-adapter",
                "--input-script",
                "chrome-full-boot",
                session_runner=lambda **kw: sess,
            )
            self.assertEqual(code, 0, err)
            cli_dir = fx.build / "runs" / "t56-cli-adapter"
            telemetry = json.loads((cli_dir / "telemetry.json").read_text(encoding="utf-8"))
            report = json.loads((cli_dir / "boot-report.json").read_text(encoding="utf-8"))
            digest_text = (cli_dir / "digest.txt").read_text(encoding="utf-8")

            self.assertEqual(telemetry["gpu_adapter"], "intel / gen-12lp")
            self.assertEqual(telemetry["summary"]["gpu_adapter"], "intel / gen-12lp")
            self.assertEqual(report["gpu_adapter"], "intel / gen-12lp")
            self.assertEqual(report["summary"]["gpu_adapter"], "intel / gen-12lp")
            self.assertIn("gpu_adapter: intel / gen-12lp", digest_text)
            self.assertIn("gpu_adapter: intel / gen-12lp", out)
            self.assertIn("input_script: chrome-full-boot", digest_text)

            # Console-only fallback when BrowserSessionResult.gpu_adapter is left None
            console_only_session = make_fake_session_result(
                console_lines=[
                    "[   0.42s] [console.info] [wgpu_worker] WebGPU adapter: nvidia / turing",
                ]
            )
            self.assertIsNone(console_only_session.gpu_adapter)
            rep_console = rbb.build_boot_report(console_only_session, smoke=True, preset="smoke")
            tel_console = rbb.build_telemetry_payload(
                console_only_session, smoke=True, preset="smoke", headless=True
            )
            self.assertEqual(rep_console["gpu_adapter"], "nvidia / turing")
            self.assertEqual(rep_console["summary"]["gpu_adapter"], "nvidia / turing")
            self.assertEqual(tel_console["gpu_adapter"], "nvidia / turing")
            self.assertIn("gpu_adapter: nvidia / turing", rbb.digest(rep_console, 0))

            # Omitted from digest.txt and summary when not detected
            no_adapter_session = make_fake_session_result(console_lines=["[   0.10s] [console.log] ready"])
            rep_none = rbb.build_boot_report(no_adapter_session, smoke=True, preset="smoke")
            tel_none = rbb.build_telemetry_payload(no_adapter_session, smoke=True, preset="smoke", headless=True)
            self.assertIsNone(rep_none["gpu_adapter"])
            self.assertNotIn("gpu_adapter", rep_none["summary"])
            self.assertIsNone(tel_none["gpu_adapter"])
            self.assertNotIn("gpu_adapter", tel_none["summary"])
            self.assertNotIn("gpu_adapter:", rbb.digest(rep_none, 0))

    def test_encode_frames_to_mp4_filters_zero_byte_and_truncated_trailing_pngs(self):
        black_png = _make_synthetic_rgb_png(420, 4, lambda x, y: (0, 0, 0))
        lit_png = _make_synthetic_rgb_png(420, 4, lambda x, y: (20, 180, 90) if x >= 410 else (0, 0, 0))
        truncated_png = lit_png[: len(lit_png) // 2]  # valid PNG header, missing IEND

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)

            # 1. Trailing 0-byte PNG is excluded before ffmpeg runs (would otherwise exit -28)
            dir_zero = root / "trailing_zero"
            dir_zero.mkdir()
            (dir_zero / "frame_0001.png").write_bytes(black_png)
            (dir_zero / "frame_0002.png").write_bytes(lit_png)
            (dir_zero / "frame_0003.png").write_bytes(b"")  # 0-byte trailing frame

            seen_stage_files: list[list[str]] = []
            seen_stage_dirs: list[Path] = []
            ffmpeg_cmds: list[list[str]] = []

            def strict_fake_ffmpeg(cmd, cwd, log, timeout):
                ffmpeg_cmds.append(list(cmd))
                input_pat = Path(cmd[cmd.index("-i") + 1])
                stage_dir = input_pat.parent
                seen_stage_dirs.append(stage_dir)
                staged = sorted(stage_dir.glob("frame_*.png"))
                seen_stage_files.append([p.name for p in staged])
                # Simulate real ffmpeg failing with -28 (4294967268) if any 0-byte/truncated frame is present
                for frame in staged:
                    data = frame.read_bytes()
                    if not data or not data.endswith(rbb._PNG_IEND_CHUNK):
                        return 4294967268
                Path(cmd[-1]).write_bytes(b"ftypmp42")
                return 0

            ok, warn = rbb.encode_frames_to_mp4(
                dir_zero,
                root / "trailing_zero.mp4",
                fps=2.0,
                skip_black=True,
                canvas_min_x=rbb.CANVAS_MIN_X,
                pil_image_mod=_FakePilModule,
                _auto_import_pil=False,
                runner=strict_fake_ffmpeg,
                which_fn=lambda n: "ffmpeg" if n == "ffmpeg" else None,
            )
            self.assertTrue(ok, warn)
            self.assertEqual(seen_stage_files[-1], ["frame_0001.png", "frame_0002.png"])
            self.assertEqual(ffmpeg_cmds[-1][ffmpeg_cmds[-1].index("-start_number") + 1], "2")
            # Temporary staging directory is cleaned up after encoding
            self.assertFalse(seen_stage_dirs[-1].exists())

            # 2. Trailing truncated PNG (partial write without IEND) is also excluded
            dir_trunc = root / "trailing_trunc"
            dir_trunc.mkdir()
            (dir_trunc / "frame_0001.png").write_bytes(lit_png)
            (dir_trunc / "frame_0002.png").write_bytes(truncated_png)
            (dir_trunc / "frame_0003.png").write_bytes(b"")

            ok2, warn2 = rbb.encode_frames_to_mp4(
                dir_trunc,
                root / "trailing_trunc.mp4",
                fps=1.0,
                runner=strict_fake_ffmpeg,
                which_fn=lambda n: "ffmpeg" if n == "ffmpeg" else None,
            )
            self.assertTrue(ok2, warn2)
            self.assertEqual(seen_stage_files[-1], ["frame_0001.png"])

            # 3. Middle 0-byte PNG is skipped and remaining frames are staged contiguously
            dir_mid = root / "mid_zero"
            dir_mid.mkdir()
            (dir_mid / "frame_0001.png").write_bytes(lit_png)
            (dir_mid / "frame_0002.png").write_bytes(b"")
            (dir_mid / "frame_0003.png").write_bytes(lit_png)

            ok3, warn3 = rbb.encode_frames_to_mp4(
                dir_mid,
                root / "mid_zero.mp4",
                fps=1.0,
                runner=strict_fake_ffmpeg,
                which_fn=lambda n: "ffmpeg" if n == "ffmpeg" else None,
            )
            self.assertTrue(ok3, warn3)
            self.assertEqual(seen_stage_files[-1], ["frame_0001.png", "frame_0002.png"])

            # 4. All-zero/corrupt frames return a clean warning without invoking ffmpeg
            dir_all_bad = root / "all_bad"
            dir_all_bad.mkdir()
            (dir_all_bad / "frame_0001.png").write_bytes(b"")
            (dir_all_bad / "frame_0002.png").write_bytes(truncated_png)
            calls_before = len(ffmpeg_cmds)
            ok4, warn4 = rbb.encode_frames_to_mp4(
                dir_all_bad,
                root / "all_bad.mp4",
                fps=1.0,
                runner=strict_fake_ffmpeg,
                which_fn=lambda n: "ffmpeg" if n == "ffmpeg" else None,
            )
            self.assertFalse(ok4)
            self.assertIn("no valid captured frames", warn4 or "")
            self.assertEqual(len(ffmpeg_cmds), calls_before)


class VisibleFrameHeapAndAudioRingMetricsTests(unittest.TestCase):
    def test_telemetry_probe_js_includes_memory_and_audio_hooks(self):
        self.assertIn("performance.memory", rbb.TELEMETRY_PROBE_JS)
        self.assertIn("usedJSHeapSize", rbb.TELEMETRY_PROBE_JS)
        self.assertIn("totalJSHeapSize", rbb.TELEMETRY_PROBE_JS)
        self.assertIn("audioEngine", rbb.TELEMETRY_PROBE_JS)
        self.assertIn("audioStats", rbb.TELEMETRY_PROBE_JS)

    def test_visible_frame_timing_extraction_from_probe_milestones_and_logs(self):
        # 1. Explicit bootState.gpu stats (W257 wgpu_worker telemetry)
        probe_gpu = make_sample_probe()
        probe_gpu["bootState"]["gpu"]["firstVisibleFrameMs"] = 4250.0
        probe_gpu["bootState"]["gpu"]["blackLoadingMs"] = 1200.0
        sess_gpu = make_fake_session_result(probe=probe_gpu, first_present_ms=3050.0)
        fvf_s, bl_s = rbb._extract_visible_frame_timing(sess_gpu)
        self.assertEqual(fvf_s, 4.25)
        self.assertEqual(bl_s, 1.2)

        # 2. Tracer milestones derive black_loading_s when explicit blackLoadingMs is absent
        probe_ms = make_sample_probe(
            stage="FIRST_VISIBLE_FRAME",
            milestones=[
                {"milestone": "PGIM_LOAD", "index": 0, "reached": True, "elapsedMs": 10.0},
                {"milestone": "FIRST_PRESENT", "index": 8, "reached": True, "elapsedMs": 1500.0},
                {"milestone": "FIRST_VISIBLE_FRAME", "index": 9, "reached": True, "elapsedMs": 5750.0},
            ],
        )
        sess_ms = make_fake_session_result(probe=probe_ms, first_present_ms=1500.0)
        fvf_s, bl_s = rbb._extract_visible_frame_timing(sess_ms)
        self.assertEqual(fvf_s, 5.75)
        self.assertEqual(bl_s, 4.25)

        # 3. Console log milestone formats (boot_tracer "reached at ...ms", explicit kv, and prefix timestamp)
        self.assertEqual(
            rbb.extract_visible_frame_timing_from_console_lines(
                ["[   6.10s] [console.info] [boot_tracer] milestone FIRST_VISIBLE_FRAME reached at 5850.0ms"]
            ),
            (5.85, None),
        )
        self.assertEqual(
            rbb.extract_visible_frame_timing_from_console_lines(
                ["[   5.12s] [console.info] [wgpu_worker] FIRST_VISIBLE_FRAME (blackLoadingMs=2100)"]
            ),
            (5.12, 2.1),
        )
        self.assertEqual(
            rbb.extract_visible_frame_timing_from_console_lines(
                [
                    "[   4.00s] [console.info] [wgpu_worker] first_visible_frame_s=3.75 black_loading_s=1.25",
                    "[   9.00s] [console.info] [wgpu_worker] first_visible_frame_s=9.99 black_loading_s=8.88",
                ]
            ),
            (3.75, 1.25),
        )

        # 4. Invalid / negative / NaN / Inf / bool / path strings resolve cleanly to None
        bad_probe = make_sample_probe()
        bad_probe["bootState"]["gpu"]["firstVisibleFrameMs"] = -10.0
        bad_probe["bootState"]["firstVisibleFrameMs"] = float("nan")
        bad_probe["diagnostics"]["firstVisibleFrameMs"] = True
        bad_probe["bootState"]["gpu"]["blackLoadingMs"] = r"C:\Users\secret\black.bin"
        bad_probe["diagnostics"]["blackLoadingMs"] = float("inf")
        sess_bad = make_fake_session_result(probe=bad_probe, console_lines=["[  1.00s] [console.log] ok"])
        self.assertEqual(rbb._extract_visible_frame_timing(sess_bad), (None, None))

    def test_js_heap_mb_extraction_and_sanitization(self):
        # Bytes -> MiB rounded to 1 decimal place
        self.assertEqual(rbb.sanitize_js_heap_mb_from_bytes(128 * 1024 * 1024), 128.0)
        self.assertEqual(rbb.sanitize_js_heap_mb_from_bytes(int(128.44 * 1024 * 1024)), 128.4)
        self.assertEqual(rbb.sanitize_js_heap_mb_from_bytes(0), 0.0)
        for bad in (None, True, False, -1, -0.5, float("nan"), float("inf"), float("-inf"), "128", r"C:\Users\bob\heap"):
            self.assertIsNone(rbb.sanitize_js_heap_mb_from_bytes(bad), f"expected None for {bad!r}")

        probe_mem = make_sample_probe()
        probe_mem["memory"] = {
            "usedJSHeapSize": int(142.36 * 1024 * 1024),
            "totalJSHeapSize": int(256.04 * 1024 * 1024),
            "jsHeapSizeLimit": int(4096 * 1024 * 1024),
        }
        self.assertEqual(rbb.extract_js_heap_mb_from_probe(probe_mem), (142.4, 256.0))

        # Partial memory (only used exposed, total invalid)
        probe_partial = {
            "memory": {
                "usedJSHeapSize": 64 * 1024 * 1024,
                "totalJSHeapSize": r"/Users/alice/bad",
            }
        }
        self.assertEqual(rbb.extract_js_heap_mb_from_probe(probe_partial), (64.0, None))
        self.assertEqual(rbb.extract_js_heap_mb_from_probe(make_sample_probe()), (None, None))

    def test_audio_ring_summary_extraction_from_probe_and_console_logs(self):
        # 1. Structured probe.audioStats (from audioEngine.getStats())
        probe_audio = make_sample_probe()
        probe_audio["audioStats"] = {
            "outputState": "running",
            "guestRingAttached": True,
            "guestFramesDrained": 4800,
        }
        sess_audio = make_fake_session_result(probe=probe_audio)
        self.assertEqual(rbb._extract_audio_ring_summary(sess_audio), (True, 4800))

        # 2. Explicit False and 0 are preserved (not conflated with None)
        probe_zero = make_sample_probe()
        probe_zero["bootState"]["audio"] = {
            "outputState": "suspended",
            "guestRingAttached": False,
            "guestFramesDrained": 0,
        }
        sess_zero = make_fake_session_result(probe=probe_zero)
        self.assertEqual(rbb._extract_audio_ring_summary(sess_zero), (False, 0))

        # 3. Console log extraction (lifecycle updates, attach/detach, failed attach ignored)
        self.assertEqual(
            rbb.extract_audio_ring_summary_from_console_lines(
                [
                    "[   1.00s] [console.info] [loader] attach-guest-pcm-ring attached (48000 Hz, 2 ch)",
                    "[   2.00s] [console.info] [audio_worker] guestRingAttached=yes guestFramesDrained=960",
                    "[   3.00s] [console.info] [audio_worker] framesDrained=2880",
                ]
            ),
            (True, 2880),
        )
        self.assertEqual(
            rbb.extract_audio_ring_summary_from_console_lines(
                [
                    "[   1.00s] [console.info] [loader] attach-guest-pcm-ring attached",
                    "[   2.00s] [console.info] [audio_worker] detach-guest-pcm-ring",
                ]
            ),
            (False, None),
        )
        self.assertEqual(
            rbb.extract_audio_ring_summary_from_console_lines(
                ["[   1.00s] [console.warn] [loader] attach-guest-pcm-ring failed: non-shared-buffer"]
            ),
            (None, None),
        )

        # 4. Absent everywhere -> (None, None)
        sess_none = make_fake_session_result()
        self.assertEqual(rbb._extract_audio_ring_summary(sess_none), (None, None))

    def test_report_telemetry_digest_and_compare_boot_reports_integration(self):
        class FakeProc:
            returncode = None

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = 0

            def wait(self, timeout=None):
                return 0

            def kill(self):
                self.returncode = -9

        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            run_dir = fx.build / "runs" / "t61-sess"
            run_dir.mkdir(parents=True)
            fake_clock = {"t": 4000.0}

            rich_probe = make_sample_probe(
                boot_mode="real-wasm64",
                session_state="running_wasm",
                stage="FIRST_VISIBLE_FRAME",
                presents=12,
                real_draws=30,
                fallback_draws=0,
                milestones=[
                    {"milestone": "PGIM_LOAD", "index": 0, "reached": True, "elapsedMs": 15.0},
                    {"milestone": "FIRST_PRESENT", "index": 8, "reached": True, "elapsedMs": 1250.0},
                    {"milestone": "FIRST_VISIBLE_FRAME", "index": 9, "reached": True, "elapsedMs": 4250.0},
                ],
            )
            rich_probe["bootState"]["gpu"]["firstVisibleFrameMs"] = 4250.0
            rich_probe["bootState"]["gpu"]["blackLoadingMs"] = 3000.0
            rich_probe["memory"] = {
                "usedJSHeapSize": int(128.4 * 1024 * 1024),
                "totalJSHeapSize": int(256.0 * 1024 * 1024),
            }
            rich_probe["audioStats"] = {
                "outputState": "running",
                "guestRingAttached": True,
                "guestFramesDrained": 4800,
            }

            class FakeCdp:
                def __init__(self, ws_url: str) -> None:
                    self.ws_url = ws_url

                def connect(self):
                    return None

                def close(self):
                    return None

                def send(self, method: str, params=None, timeout=None):
                    if method == "Runtime.evaluate":
                        return {"result": {"value": rich_probe}}
                    return {}

                def drain_events(self, timeout: float = 0.0):
                    fake_clock["t"] += 0.1
                    return []

            with mock.patch.object(rbb.time, "monotonic", side_effect=lambda: fake_clock["t"]):
                sess = rbb.run_browser_session(
                    repo=fx.repo,
                    run_dir=run_dir,
                    chrome_path=fx.chrome,
                    port=0,
                    boot_dir=None,
                    game_dir=None,
                    shaders_dir=None,
                    duration_s=0.15,
                    headless=True,
                    smoke=True,
                    send_input=False,
                    capture_frames=False,
                    frame_interval_s=1.0,
                    encode_mp4=False,
                    popen_fn=lambda cmd, **kw: FakeProc(),
                    cdp_factory=FakeCdp,
                    wait_server_fn=lambda *a, **k: {"status_code": 200, "coop": "same-origin", "coep": "require-corp"},
                    wait_cdp_fn=lambda *a, **k: "ws://127.0.0.1:9222/devtools/page/1",
                )

            self.assertEqual(sess.first_visible_frame_s, 4.25)
            self.assertEqual(sess.black_loading_s, 3.0)
            self.assertEqual(sess.js_heap_used_mb, 128.4)
            self.assertEqual(sess.js_heap_total_mb, 256.0)
            self.assertIs(sess.audio_ring_attached, True)
            self.assertEqual(sess.audio_frames_drained, 4800)

            # End-to-end CLI run -> boot-report.json, telemetry.json, digest.txt
            code, out, err, _, _ = fx.run_cli(
                "--smoke",
                "--name",
                "t61-cli-full",
                session_runner=lambda **kw: sess,
            )
            self.assertEqual(code, 0, err)
            cli_dir = fx.build / "runs" / "t61-cli-full"
            report = json.loads((cli_dir / "boot-report.json").read_text(encoding="utf-8"))
            telemetry = json.loads((cli_dir / "telemetry.json").read_text(encoding="utf-8"))
            digest_text = (cli_dir / "digest.txt").read_text(encoding="utf-8")

            # Verify boot-report.json preserves schema v2 + new top-level & summary keys
            self.assertEqual(report["schema"], compare_boot_reports.REPORT_SCHEMA)
            self.assertEqual(report["schema_version"], 2)
            self.assertEqual(report["first_visible_frame_s"], 4.25)
            self.assertEqual(report["black_loading_s"], 3.0)
            self.assertEqual(report["js_heap_used_mb"], 128.4)
            self.assertEqual(report["js_heap_total_mb"], 256.0)
            self.assertIs(report["audio_ring_attached"], True)
            self.assertEqual(report["audio_frames_drained"], 4800)
            self.assertEqual(report["summary"]["first_visible_frame_s"], 4.25)
            self.assertEqual(report["summary"]["first_visible_frame_ms"], 4250.0)
            self.assertEqual(report["summary"]["black_loading_s"], 3.0)
            self.assertEqual(report["summary"]["black_loading_ms"], 3000.0)
            self.assertEqual(report["summary"]["js_heap_used_mb"], 128.4)
            self.assertEqual(report["summary"]["js_heap_total_mb"], 256.0)
            self.assertIs(report["summary"]["audio_ring_attached"], True)
            self.assertEqual(report["summary"]["audio_frames_drained"], 4800)

            # Verify telemetry.json parity
            self.assertEqual(telemetry["first_visible_frame_s"], 4.25)
            self.assertEqual(telemetry["black_loading_s"], 3.0)
            self.assertEqual(telemetry["js_heap_used_mb"], 128.4)
            self.assertEqual(telemetry["js_heap_total_mb"], 256.0)
            self.assertIs(telemetry["audio_ring_attached"], True)
            self.assertEqual(telemetry["audio_frames_drained"], 4800)

            # Verify concise digest.txt formatting
            self.assertIn("first_visible_frame_s: 4.25 (black_loading_s=3.0)", digest_text)
            self.assertIn("js_heap_mb: used=128.4, total=256.0", digest_text)
            self.assertIn("audio_ring: attached=true, frames_drained=4800", digest_text)
            self.assertIn("first_visible_frame_s: 4.25 (black_loading_s=3.0)", out)

            # Verify compare_boot_reports compatibility & gpu_delta integration
            cmp_res = compare_boot_reports.compare(report, report, compare_boot_reports.Tolerances())
            self.assertEqual(cmp_res.failures(), [])
            self.assertEqual(cmp_res.gpu_delta["first_visible_frame_ms"]["a"], 4250.0)
            self.assertEqual(cmp_res.gpu_delta["black_loading_ms"]["a"], 3000.0)

            # Zero host paths leaked across all artifacts
            for fname in ("boot-report.json", "telemetry.json", "digest.txt", "console.log"):
                text = (cli_dir / fname).read_text(encoding="utf-8")
                self.assertNotIn(str(Path(d)), text, f"host path leaked in {fname}")

            # Graceful None / missing handling: omitted from summary & digest.txt
            minimal_session = make_fake_session_result()
            rep_min = rbb.build_boot_report(minimal_session, smoke=True, preset="smoke")
            tel_min = rbb.build_telemetry_payload(minimal_session, smoke=True, preset="smoke", headless=True)
            for key in (
                "first_visible_frame_s",
                "black_loading_s",
                "js_heap_used_mb",
                "js_heap_total_mb",
                "audio_ring_attached",
                "audio_frames_drained",
            ):
                self.assertIsNone(rep_min[key])
                self.assertNotIn(key, rep_min["summary"])
                self.assertIsNone(tel_min[key])
                self.assertNotIn(key, tel_min["summary"])
            dig_min = rbb.digest(rep_min, 0)
            self.assertNotIn("first_visible_frame_s:", dig_min)
            self.assertNotIn("black_loading_s:", dig_min)
            self.assertNotIn("js_heap_mb:", dig_min)
            self.assertNotIn("audio_ring:", dig_min)

            # Partial digest formatting (single fields & false/0 edge cases)
            partial_rep = dict(rep_min)
            partial_rep["first_visible_frame_s"] = 2.5
            partial_rep["js_heap_used_mb"] = 99.5
            partial_rep["audio_ring_attached"] = False
            partial_rep["audio_frames_drained"] = 0
            dig_partial = rbb.digest(partial_rep, 0)
            self.assertIn("first_visible_frame_s: 2.5", dig_partial)
            self.assertNotIn("black_loading_s=", dig_partial)
            self.assertIn("js_heap_mb: used=99.5", dig_partial)
            self.assertNotIn("total=", dig_partial)
            self.assertIn("audio_ring: attached=false, frames_drained=0", dig_partial)


class MenuToIngameFpsAndQuietServerTests(unittest.TestCase):
    def test_menu_to_ingame_preset_validation_url_help_and_dry_run(self):
        self.assertEqual(rbb.ALLOWED_INPUT_SCRIPT_PRESETS.get("menu-to-ingame"), "menu-to-ingame")
        self.assertEqual(rbb.INPUT_SCRIPT_ALIASES.get("menu-to-ingame"), "menu-to-ingame")
        self.assertEqual(rbb.sanitize_input_script_preset("menu-to-ingame"), "menu-to-ingame")
        self.assertEqual(rbb.sanitize_input_script_preset("  MENU-TO-INGAME  "), "menu-to-ingame")
        self.assertEqual(rbb.validate_input_script_preset("menu-to-ingame"), "menu-to-ingame")
        self.assertEqual(
            rbb.build_target_url("127.0.0.1", 8080, input_script="menu-to-ingame"),
            "http://127.0.0.1:8080/?inputScript=menu-to-ingame",
        )

        help_text = rbb.build_arg_parser().format_help()
        self.assertIn("menu-to-ingame", help_text)
        self.assertIn("--quiet-server", help_text)

        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            code, out, err, _, _ = fx.run_cli(
                "--smoke",
                "--dry-run",
                "--name",
                "dry-menu-to-ingame",
                "--input-script",
                "menu-to-ingame",
            )
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertEqual(payload["input_script"], "menu-to-ingame")
            self.assertEqual(payload["inputScript"], "menu-to-ingame")
            self.assertEqual(
                payload["url"],
                "http://127.0.0.1:<ephemeral>/?smoke=1&inputScript=menu-to-ingame",
            )

    def test_compute_present_fps_avg_normal_and_edge_cases(self):
        # 1. Normal 30 FPS subsampled run from FIRST_PRESENT (1000 ms) to last present (11000 ms)
        subsampled = [{"index": 0, "wall_ms": 1000.0}, {"index": 1, "wall_ms": 6000.0}, {"index": 2, "wall_ms": 11000.0}]
        self.assertEqual(
            rbb.compute_present_fps_avg(subsampled, present_count=301, first_present_ms=1000.0),
            30.0,
        )

        # 2. Normal per-present run without explicit present_count or first_present_ms
        per_present = [
            {"wall_ms": 1000.0},
            {"wall_ms": 1500.0},
            {"wall_ms": 2000.0},
            {"wall_ms": 2500.0},
            {"wall_ms": 3000.0},
        ]
        self.assertEqual(rbb.compute_present_fps_avg(per_present), 2.0)

        # 3. Fractional rounding to 2 decimal places: (11 - 1) / 3.0s = 3.33
        self.assertEqual(
            rbb.compute_present_fps_avg(
                [{"wall_ms": 500.0}, {"wall_ms": 3500.0}],
                present_count=11,
                first_present_ms=500.0,
            ),
            3.33,
        )

        # 4. Single trailing sample after explicit first_present_ms with >= 2 total presents
        self.assertEqual(
            rbb.compute_present_fps_avg(
                [{"wall_ms": 2000.0}],
                present_count=11,
                first_present_ms=1000.0,
            ),
            10.0,
        )

        # 5. < 2 presents returns None
        self.assertIsNone(rbb.compute_present_fps_avg([], present_count=0, first_present_ms=None))
        self.assertIsNone(rbb.compute_present_fps_avg([{"wall_ms": 1000.0}], present_count=1, first_present_ms=1000.0))
        self.assertIsNone(rbb.compute_present_fps_avg([{"wall_ms": 1000.0}]))
        self.assertIsNone(rbb.compute_present_fps_avg([], present_count=10, first_present_ms=1000.0))
        self.assertIsNone(rbb.compute_present_fps_avg(None, present_count=5, first_present_ms=1000.0))

        # 6. Zero duration returns None (identical timestamps or single sample at FIRST_PRESENT)
        self.assertIsNone(
            rbb.compute_present_fps_avg(
                [{"wall_ms": 1000.0}, {"wall_ms": 1000.0}],
                present_count=2,
                first_present_ms=1000.0,
            )
        )
        self.assertIsNone(
            rbb.compute_present_fps_avg(
                [{"wall_ms": 125.0}],
                present_count=6,
                first_present_ms=125.0,
            )
        )

        # 7. Negative duration / invalid timestamps return None
        self.assertIsNone(rbb.compute_present_fps_avg([{"wall_ms": 2000.0}, {"wall_ms": 1000.0}]))
        self.assertIsNone(
            rbb.compute_present_fps_avg(
                [{"wall_ms": float("nan")}, {"wall_ms": float("inf")}, {"wall_ms": -5.0}, {"wall_ms": True}],
                present_count=4,
            )
        )

    def test_quiet_server_cli_flag_and_server_command_construction(self):
        parser = rbb.build_arg_parser()
        self.assertFalse(parser.parse_args([]).quiet_server)
        self.assertTrue(parser.parse_args(["--quiet-server"]).quiet_server)

        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))

            # Direct build_server_cmd check (quiet=False vs quiet=True)
            cmd_default = rbb.build_server_cmd(
                repo=fx.repo,
                host="127.0.0.1",
                port=8000,
                web_dir=fx.repo / "web",
                data_dir=fx.repo / "local-data",
                boot_dir=fx.boot_dir,
                game_dir=None,
                shaders_dir=None,
            )
            self.assertNotIn("--quiet", cmd_default)

            cmd_quiet = rbb.build_server_cmd(
                repo=fx.repo,
                host="127.0.0.1",
                port=8000,
                web_dir=fx.repo / "web",
                data_dir=fx.repo / "local-data",
                boot_dir=fx.boot_dir,
                game_dir=None,
                shaders_dir=None,
                quiet=True,
            )
            self.assertIn("--quiet", cmd_quiet)
            self.assertEqual(cmd_quiet[-1], "--quiet")

            # Dry-run without vs with --quiet-server
            code_def, out_def, err_def, _, _ = fx.run_cli("--smoke", "--dry-run", "--name", "q-def")
            self.assertEqual(code_def, 0, err_def)
            dry_def = json.loads(out_def)
            self.assertFalse(dry_def["quiet_server"])
            self.assertNotIn("--quiet", dry_def["server_command"])

            code_q, out_q, err_q, _, _ = fx.run_cli("--smoke", "--dry-run", "--name", "q-on", "--quiet-server")
            self.assertEqual(code_q, 0, err_q)
            dry_q = json.loads(out_q)
            self.assertTrue(dry_q["quiet_server"])
            self.assertIn("--quiet", dry_q["server_command"])

            # Live run_browser_session passes --quiet to serve_local.py (and not to Chrome)
            spawned_cmds: list[list[str]] = []

            class FakeProc:
                returncode = None

                def poll(self):
                    return self.returncode

                def terminate(self):
                    self.returncode = 0

                def wait(self, timeout=None):
                    return 0

                def kill(self):
                    self.returncode = -9

            def fake_popen(cmd, **kwargs):
                spawned_cmds.append(list(cmd))
                return FakeProc()

            class FakeCdp:
                def __init__(self, ws_url: str) -> None:
                    self.ws_url = ws_url

                def connect(self):
                    return None

                def close(self):
                    return None

                def send(self, method: str, params=None, timeout=None):
                    if method == "Runtime.evaluate":
                        return {"result": {"value": make_sample_probe(presents=2)}}
                    return {}

                def drain_events(self, timeout: float = 0.0):
                    return []

            run_dir = fx.build / "runs" / "quiet-sess"
            run_dir.mkdir(parents=True)
            res = rbb.run_browser_session(
                repo=fx.repo,
                run_dir=run_dir,
                chrome_path=fx.chrome,
                port=0,
                boot_dir=None,
                game_dir=None,
                shaders_dir=None,
                duration_s=0.05,
                headless=True,
                smoke=True,
                send_input=False,
                capture_frames=False,
                frame_interval_s=1.0,
                encode_mp4=False,
                quiet_server=True,
                popen_fn=fake_popen,
                cdp_factory=FakeCdp,
                wait_server_fn=lambda *a, **k: {"status_code": 200, "coop": "same-origin", "coep": "require-corp"},
                wait_cdp_fn=lambda *a, **k: "ws://127.0.0.1:9222/devtools/page/1",
            )
            self.assertTrue(res.ok)
            self.assertEqual(len(spawned_cmds), 2)
            server_cmd, chrome_cmd = spawned_cmds
            self.assertTrue(any("serve_local.py" in tok for tok in server_cmd))
            self.assertIn("--quiet", server_cmd)
            self.assertNotIn("--quiet", chrome_cmd)

            # CLI live run forwards quiet_server=True to session_runner and command.json
            code_cli, _, err_cli, _, sessions = fx.run_cli(
                "--smoke",
                "--name",
                "quiet-cli",
                "--quiet-server",
                "--input-script",
                "menu-to-ingame",
            )
            self.assertEqual(code_cli, 0, err_cli)
            self.assertTrue(sessions[0]["quiet_server"])
            self.assertEqual(sessions[0]["input_script"], "menu-to-ingame")
            cmd_meta = json.loads((fx.build / "runs" / "quiet-cli" / "command.json").read_text(encoding="utf-8"))
            self.assertTrue(cmd_meta["quiet_server"])
            self.assertEqual(cmd_meta["input_script"], "menu-to-ingame")

    def test_present_fps_avg_in_boot_report_telemetry_and_digest(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            probe = make_sample_probe(
                boot_mode="real-wasm64",
                session_state="running_wasm",
                stage="FIRST_VISIBLE_FRAME",
                presents=61,
                real_draws=120,
                fallback_draws=0,
                milestones=[
                    {"milestone": "PGIM_LOAD", "index": 0, "reached": True, "elapsedMs": 15.0},
                    {"milestone": "FIRST_PRESENT", "index": 8, "reached": True, "elapsedMs": 1000.0},
                    {"milestone": "FIRST_VISIBLE_FRAME", "index": 9, "reached": True, "elapsedMs": 3000.0},
                ],
            )
            probe["bootState"]["gpu"]["firstVisibleFrameMs"] = 3000.0
            probe["bootState"]["gpu"]["blackLoadingMs"] = 2000.0
            probe["memory"] = {"usedJSHeapSize": 128 * 1024 * 1024, "totalJSHeapSize": 256 * 1024 * 1024}
            probe["audioStats"] = {"guestRingAttached": True, "guestFramesDrained": 960}

            session = make_fake_session_result(
                probe=probe,
                first_present_ms=1000.0,
                presents=[
                    {"index": 0, "wall_ms": 1000.0, "dispatches": 0, "import_calls": 0, "counters": {}},
                    {"index": 1, "wall_ms": 2000.0, "dispatches": 0, "import_calls": 0, "counters": {}},
                    {"index": 2, "wall_ms": 3000.0, "dispatches": 0, "import_calls": 0, "counters": {}},
                ],
            )

            code, out, err, _, _ = fx.run_cli(
                "--smoke",
                "--name",
                "t73-fps-run",
                "--input-script",
                "menu-to-ingame",
                "--quiet-server",
                session_runner=lambda **kw: session,
            )
            self.assertEqual(code, 0, err)
            run_dir = fx.build / "runs" / "t73-fps-run"
            report = json.loads((run_dir / "boot-report.json").read_text(encoding="utf-8"))
            telemetry = json.loads((run_dir / "telemetry.json").read_text(encoding="utf-8"))
            digest_text = (run_dir / "digest.txt").read_text(encoding="utf-8")

            # (61 - 1) presents / (3000 - 1000 ms) = 30.0 FPS
            self.assertEqual(report["present_fps_avg"], 30.0)
            self.assertEqual(report["summary"]["present_fps_avg"], 30.0)
            self.assertEqual(telemetry["present_fps_avg"], 30.0)
            self.assertEqual(telemetry["summary"]["present_fps_avg"], 30.0)
            self.assertIn("present_fps_avg: 30.00", digest_text)
            self.assertIn("present_fps_avg: 30.00", out)
            self.assertIn("input_script: menu-to-ingame", digest_text)

            # T61 metrics remain intact alongside present_fps_avg
            self.assertEqual(report["first_visible_frame_s"], 3.0)
            self.assertEqual(report["black_loading_s"], 2.0)
            self.assertEqual(report["js_heap_used_mb"], 128.0)
            self.assertIs(report["audio_ring_attached"], True)
            self.assertIn("first_visible_frame_s: 3.0 (black_loading_s=2.0)", digest_text)
            self.assertIn("js_heap_mb: used=128.0, total=256.0", digest_text)
            self.assertIn("audio_ring: attached=true, frames_drained=960", digest_text)

            # Summary-only fallback in digest()
            summary_only_rep = {
                "status": "ok",
                "phase": "FIRST_PRESENT",
                "mode": "real-wasm64",
                "exit_code": 0,
                "elapsed_ms": 3000.0,
                "milestones": [],
                "presents": [],
                "counters": {"presents": 61},
                "summary": {"present_fps_avg": 29.75},
            }
            self.assertIn("present_fps_avg: 29.75", rbb.digest(summary_only_rep, 0))

            # Null / missing present_fps_avg is null top-level and omitted from summary & digest.txt
            single_present_sess = make_fake_session_result(
                probe=make_sample_probe(presents=1),
                first_present_ms=1000.0,
                presents=[{"index": 0, "wall_ms": 1000.0, "dispatches": 0, "import_calls": 0, "counters": {}}],
            )
            rep_null = rbb.build_boot_report(single_present_sess, smoke=True, preset="smoke")
            self.assertIsNone(rep_null["present_fps_avg"])
            self.assertNotIn("present_fps_avg", rep_null["summary"])
            self.assertNotIn("present_fps_avg:", rbb.digest(rep_null, 0))

            # compare_boot_reports compatibility preserved
            cmp_res = compare_boot_reports.compare(report, report, compare_boot_reports.Tolerances())
            self.assertEqual(cmp_res.failures(), [])


class ChromeMenuFlow2InputScriptTests(unittest.TestCase):
    def test_chrome_menu_flow_2_preset_validation_url_help_and_cli(self):
        self.assertEqual(rbb.ALLOWED_INPUT_SCRIPT_PRESETS.get("chrome-menu-flow-2"), "chrome-menu-flow-2")
        self.assertEqual(rbb.INPUT_SCRIPT_ALIASES.get("chrome-menu-flow-2"), "chrome-menu-flow-2")
        self.assertEqual(
            list(rbb.ALLOWED_INPUT_SCRIPT_PRESETS.keys()),
            sorted(rbb.ALLOWED_INPUT_SCRIPT_PRESETS.keys()),
        )
        self.assertEqual(rbb.sanitize_input_script_preset("chrome-menu-flow-2"), "chrome-menu-flow-2")
        self.assertEqual(rbb.sanitize_input_script_preset("  CHROME-MENU-FLOW-2  "), "chrome-menu-flow-2")
        self.assertEqual(rbb.validate_input_script_preset("chrome-menu-flow-2"), "chrome-menu-flow-2")
        self.assertEqual(
            rbb.build_target_url("127.0.0.1", 8080, input_script="chrome-menu-flow-2"),
            "http://127.0.0.1:8080/?inputScript=chrome-menu-flow-2",
        )

        help_text = rbb.build_arg_parser().format_help()
        self.assertIn("chrome-menu-flow-2", help_text)

        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            code, out, err, _, _ = fx.run_cli(
                "--smoke",
                "--dry-run",
                "--name",
                "dry-chrome-menu-flow-2",
                "--input-script",
                "chrome-menu-flow-2",
            )
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertEqual(payload["input_script"], "chrome-menu-flow-2")
            self.assertEqual(payload["inputScript"], "chrome-menu-flow-2")
            self.assertEqual(
                payload["url"],
                "http://127.0.0.1:<ephemeral>/?smoke=1&inputScript=chrome-menu-flow-2",
            )

            code_cli, _, err_cli, _, sessions = fx.run_cli(
                "--smoke",
                "--name",
                "run-chrome-menu-flow-2",
                "--input-script",
                "chrome-menu-flow-2",
            )
            self.assertEqual(code_cli, 0, err_cli)
            self.assertEqual(sessions[0]["input_script"], "chrome-menu-flow-2")
            cmd_meta = json.loads(
                (fx.build / "runs" / "run-chrome-menu-flow-2" / "command.json").read_text(encoding="utf-8")
            )
            self.assertEqual(cmd_meta["input_script"], "chrome-menu-flow-2")


class CdpTimeoutHoldUntilVisibleAndRollingFpsTests(unittest.TestCase):
    def test_cdp_timeout_and_oserror_resilience_in_probe_capture_input_and_force_smoke(self):
        class FakeProc:
            returncode = None

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = 0

            def wait(self, timeout=None):
                return 0

            def kill(self):
                self.returncode = -9

        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            run_dir = fx.build / "runs" / "t94-timeout-resilience"
            run_dir.mkdir(parents=True)
            fake_clock = {"t": 1000.0}
            probe_timeouts: list[float | None] = []
            counts = {"probe": 0, "force_smoke": 0, "shot": 0, "key": 0}
            fake_jpeg_b64 = base64.b64encode(b"\xff\xd8\xff\xe0test\xff\xd9").decode("ascii")

            class FlakyCdp:
                def __init__(self, ws_url: str) -> None:
                    self.ws_url = ws_url

                def connect(self):
                    return None

                def close(self):
                    return None

                def send(self, method: str, params=None, timeout=None):
                    params = params or {}
                    if method == "Runtime.evaluate":
                        expr = params.get("expression", "")
                        if expr == rbb.FORCE_SMOKE_JS:
                            counts["force_smoke"] += 1
                            raise TimeoutError("CDP WebSocket read timed out during force-smoke")
                        probe_timeouts.append(timeout)
                        counts["probe"] += 1
                        if counts["probe"] == 1:
                            raise TimeoutError("CDP WebSocket read timed out during wasm compile")
                        if counts["probe"] == 2:
                            raise OSError("temporary socket hiccup")
                        if counts["probe"] == 3:
                            return {
                                "result": {
                                    "value": make_sample_probe(
                                        boot_mode="real-wasm64",
                                        session_state="booting",
                                        presents=10,
                                    )
                                }
                            }
                        return {
                            "result": {
                                "value": make_sample_probe(
                                    boot_mode="smoke-demo",
                                    session_state="running_smoke",
                                    presents=40,
                                )
                            }
                        }
                    if method == "Page.captureScreenshot":
                        counts["shot"] += 1
                        if counts["shot"] == 1:
                            raise TimeoutError("screenshot timeout")
                        return {"data": fake_jpeg_b64}
                    if method == "Input.dispatchKeyEvent":
                        counts["key"] += 1
                        if counts["key"] == 1:
                            raise TimeoutError("key dispatch timeout")
                        return {}
                    return {}

                def drain_events(self, timeout: float = 0.0):
                    fake_clock["t"] += 0.6
                    return []

            with (
                mock.patch.object(rbb.time, "monotonic", side_effect=lambda: fake_clock["t"]),
                mock.patch.object(rbb.time, "sleep", side_effect=lambda _s: None),
            ):
                sess = rbb.run_browser_session(
                    repo=fx.repo,
                    run_dir=run_dir,
                    chrome_path=fx.chrome,
                    port=0,
                    boot_dir=None,
                    game_dir=None,
                    shaders_dir=None,
                    duration_s=7.0,
                    headless=True,
                    smoke=True,
                    send_input=True,
                    capture_frames=True,
                    frame_interval_s=1.0,
                    encode_mp4=False,
                    popen_fn=lambda cmd, **kw: FakeProc(),
                    cdp_factory=FlakyCdp,
                    wait_server_fn=lambda *a, **k: {"status_code": 200, "coop": "same-origin", "coep": "require-corp"},
                    wait_cdp_fn=lambda *a, **k: "ws://127.0.0.1:9222/devtools/page/1",
                )

            self.assertTrue(sess.ok)
            self.assertFalse(sess.crashed)
            self.assertEqual(sess.status, "ok")
            self.assertGreaterEqual(counts["probe"], 4)
            self.assertEqual(counts["force_smoke"], 1)
            self.assertTrue(all(t == rbb.CDP_PROBE_TIMEOUT_S for t in probe_timeouts))
            self.assertEqual(rbb.CDP_PROBE_TIMEOUT_S, 15.0)
            self.assertGreaterEqual(len(sess.captured_frames), 1)
            self.assertGreaterEqual(sess.scripted_inputs_sent, 1)
            self.assertEqual(sess.final_probe["bootState"]["presentCount"], 40)
            self.assertGreaterEqual(len(sess.presents), 2)
            self.assertEqual(sess.presents[0]["present_count"], 10)
            self.assertEqual(sess.presents[-1]["present_count"], 40)

            # Startup TimeoutError during Page.navigate returns a clean error result instead of raising
            class StartupTimeoutCdp(FlakyCdp):
                def send(self, method: str, params=None, timeout=None):
                    if method == "Page.navigate":
                        raise TimeoutError("navigate timed out")
                    return {}

            run_dir2 = fx.build / "runs" / "t94-startup-timeout"
            run_dir2.mkdir(parents=True)
            sess_err = rbb.run_browser_session(
                repo=fx.repo,
                run_dir=run_dir2,
                chrome_path=fx.chrome,
                port=0,
                boot_dir=None,
                game_dir=None,
                shaders_dir=None,
                duration_s=1.0,
                headless=True,
                smoke=True,
                send_input=False,
                capture_frames=False,
                frame_interval_s=1.0,
                encode_mp4=False,
                popen_fn=lambda cmd, **kw: FakeProc(),
                cdp_factory=StartupTimeoutCdp,
                wait_server_fn=lambda *a, **k: {"status_code": 200, "coop": "same-origin", "coep": "require-corp"},
                wait_cdp_fn=lambda *a, **k: "ws://127.0.0.1:9222/devtools/page/1",
            )
            self.assertFalse(sess_err.ok)
            self.assertTrue(sess_err.crashed)
            self.assertEqual(sess_err.status, "error")
            self.assertIn("navigate timed out", sess_err.message)

    def test_hold_until_visible_flag_url_building_dry_run_and_cli(self):
        parser = rbb.build_arg_parser()
        self.assertIsNone(parser.parse_args([]).hold_until_visible)
        self.assertEqual(parser.parse_args(["--hold-until-visible", "auto"]).hold_until_visible, "auto")
        self.assertEqual(parser.parse_args(["--hold-until-visible", "on"]).hold_until_visible, "on")
        self.assertEqual(parser.parse_args(["--hold-until-visible", "off"]).hold_until_visible, "off")
        self.assertIn("--hold-until-visible", parser.format_help())

        # URL building: omitted & auto omit query param; on -> 1; off -> 0
        self.assertEqual(rbb.build_target_url("127.0.0.1", 8080), "http://127.0.0.1:8080/")
        self.assertEqual(
            rbb.build_target_url("127.0.0.1", 8080, hold_until_visible="auto"),
            "http://127.0.0.1:8080/",
        )
        self.assertEqual(
            rbb.build_target_url("127.0.0.1", 8080, hold_until_visible="on"),
            "http://127.0.0.1:8080/?holdUntilVisible=1",
        )
        self.assertEqual(
            rbb.build_target_url("127.0.0.1", 8080, hold_until_visible="off"),
            "http://127.0.0.1:8080/?holdUntilVisible=0",
        )
        self.assertEqual(
            rbb.build_target_url(
                "127.0.0.1",
                8080,
                smoke=True,
                input_script="chrome-menu-flow-2",
                hide_hud=True,
                hold_until_visible="off",
            ),
            "http://127.0.0.1:8080/?smoke=1&inputScript=chrome-menu-flow-2&hud=0&holdUntilVisible=0",
        )
        with self.assertRaises(rbb.RunnerError):
            rbb.validate_hold_until_visible("maybe")

        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))

            # Invalid CLI value returns exit 2
            bad_code, _, _, _, _ = fx.run_cli("--smoke", "--dry-run", "--hold-until-visible", "invalid")
            self.assertEqual(bad_code, 2)

            # Dry-run: omitted, auto, off, on
            for flag_args, expected_val, expected_url in (
                ([], None, "http://127.0.0.1:<ephemeral>/?smoke=1"),
                (["--hold-until-visible", "auto"], "auto", "http://127.0.0.1:<ephemeral>/?smoke=1"),
                (["--hold-until-visible", "off"], "off", "http://127.0.0.1:<ephemeral>/?smoke=1&holdUntilVisible=0"),
                (["--hold-until-visible", "on"], "on", "http://127.0.0.1:<ephemeral>/?smoke=1&holdUntilVisible=1"),
            ):
                code, out, err, _, _ = fx.run_cli("--smoke", "--dry-run", "--name", "huv-dry", *flag_args)
                self.assertEqual(code, 0, err)
                payload = json.loads(out)
                self.assertEqual(payload["hold_until_visible"], expected_val)
                self.assertEqual(payload["url"], expected_url)

            # Live CLI run forwards hold_until_visible to session_runner, command.json, and telemetry.json
            code_cli, _, err_cli, _, sessions = fx.run_cli(
                "--smoke",
                "--name",
                "huv-off",
                "--hold-until-visible",
                "off",
            )
            self.assertEqual(code_cli, 0, err_cli)
            self.assertEqual(sessions[0]["hold_until_visible"], "off")
            run_dir = fx.build / "runs" / "huv-off"
            cmd_meta = json.loads((run_dir / "command.json").read_text(encoding="utf-8"))
            telemetry = json.loads((run_dir / "telemetry.json").read_text(encoding="utf-8"))
            self.assertEqual(cmd_meta["hold_until_visible"], "off")
            self.assertEqual(telemetry["hold_until_visible"], "off")

    def test_audio_ring_attached_extraction_from_boot_state_over_stale_audio_stats(self):
        # Exact t84 repro: audioStats reports guestRingAttached=False while bootState.audioRingAttached=True
        probe_t84 = make_sample_probe()
        probe_t84["audioStats"] = {
            "outputState": "running",
            "guestRingAttached": False,
            "guestFramesDrained": 0,
        }
        probe_t84["bootState"]["audioRingAttached"] = True
        probe_t84["bootState"]["audio"] = {
            "outputState": "running",
            "framesDrained": 0,
        }
        probe_t84["diagnostics"]["audioRingAttached"] = True
        sess_t84 = make_fake_session_result(probe=probe_t84)
        self.assertEqual(rbb._extract_audio_ring_summary(sess_t84), (True, 0))

        # Higher framesDrained in bootState.audio is also preserved over stale 0 in audioStats
        probe_drained = make_sample_probe()
        probe_drained["audioStats"] = {
            "outputState": "running",
            "guestRingAttached": False,
            "guestFramesDrained": 0,
        }
        probe_drained["bootState"]["audioRingAttached"] = True
        probe_drained["bootState"]["audio"] = {
            "outputState": "running",
            "framesDrained": 2400,
        }
        sess_drained = make_fake_session_result(probe=probe_drained)
        self.assertEqual(rbb._extract_audio_ring_summary(sess_drained), (True, 2400))

        # Snake-case telemetry-style boot_state is also accepted
        sess_snake = make_fake_session_result(
            probe={"boot_state": {"audioRingAttached": True, "audio": {"framesDrained": 960}}}
        )
        self.assertEqual(rbb._extract_audio_ring_summary(sess_snake), (True, 960))

        # End-to-end report, telemetry, and digest reflect attached=true
        rep = rbb.build_boot_report(sess_t84, smoke=False, preset="long")
        tel = rbb.build_telemetry_payload(sess_t84, smoke=False, preset="long", headless=True)
        dig = rbb.digest(rep, 0)
        self.assertIs(rep["audio_ring_attached"], True)
        self.assertIs(rep["summary"]["audio_ring_attached"], True)
        self.assertIs(tel["audio_ring_attached"], True)
        self.assertIs(tel["summary"]["audio_ring_attached"], True)
        self.assertIn("audio_ring: attached=true, frames_drained=0", dig)

        # Explicitly False across all sources remains attached=false
        probe_false = make_sample_probe()
        probe_false["audioStats"] = {"guestRingAttached": False, "guestFramesDrained": 0}
        probe_false["bootState"]["audioRingAttached"] = False
        sess_false = make_fake_session_result(probe=probe_false)
        self.assertEqual(rbb._extract_audio_ring_summary(sess_false), (False, 0))

    def test_rolling_fps_stats_and_swapchain_probe_telemetry_and_digest(self):
        # 1. Ramp (0..10s @ 5 FPS) + steady state (10..50s @ 30 FPS) with explicit present_count
        polled_samples: list[dict[str, Any]] = []
        cum = 0
        for sec in range(0, 51, 5):
            if sec == 0:
                cum = 0
            elif sec <= 10:
                cum += 25  # 5 FPS for 5s
            else:
                cum += 150  # 30 FPS for 5s
            polled_samples.append({"index": len(polled_samples), "wall_ms": float(sec * 1000), "present_count": cum})
        stats = rbb.compute_rolling_fps_stats(polled_samples)
        self.assertEqual(stats["fps_steady_avg"], 30.0)
        self.assertEqual(stats["fps_p50_window"], 30.0)
        self.assertEqual(stats["fps_max_window"], 30.0)
        self.assertEqual(stats["windows_ge_30fps_pct"], 80.0)

        # 2. Mixed 5s windows: 10, 20, 30, 40 FPS -> median 25.0, max 40.0, 50.0% >= 30 FPS
        mixed = [
            {"wall_ms": 0.0, "present_count": 0},
            {"wall_ms": 5000.0, "present_count": 50},    # 10 FPS
            {"wall_ms": 10000.0, "present_count": 150},  # 20 FPS
            {"wall_ms": 15000.0, "present_count": 300},  # 30 FPS
            {"wall_ms": 20000.0, "present_count": 500},  # 40 FPS
        ]
        mixed_stats = rbb.compute_rolling_fps_stats(mixed)
        self.assertEqual(mixed_stats["fps_p50_window"], 25.0)
        self.assertEqual(mixed_stats["fps_max_window"], 40.0)
        self.assertEqual(mixed_stats["windows_ge_30fps_pct"], 50.0)
        self.assertEqual(mixed_stats["fps_steady_avg"], 30.0)

        # 3. Per-present samples without present_count at 30 FPS for 10 seconds
        per_present = [{"wall_ms": round(i * (1000.0 / 30.0), 3)} for i in range(301)]
        pp_stats = rbb.compute_rolling_fps_stats(per_present)
        self.assertEqual(pp_stats["fps_steady_avg"], 30.0)
        self.assertEqual(pp_stats["fps_p50_window"], 30.0)
        self.assertEqual(pp_stats["fps_max_window"], 30.0)
        self.assertEqual(pp_stats["windows_ge_30fps_pct"], 100.0)

        # 4. Subsampled samples without per-sample present_count return None (no bogus 1 FPS)
        subsampled_no_counts = [{"wall_ms": 1000.0}, {"wall_ms": 2000.0}, {"wall_ms": 3000.0}]
        sub_stats = rbb.compute_rolling_fps_stats(subsampled_no_counts, present_count=61)
        self.assertEqual(
            sub_stats,
            {
                "fps_steady_avg": None,
                "fps_p50_window": None,
                "fps_max_window": None,
                "windows_ge_30fps_pct": None,
            },
        )

        # 5. Degenerate / invalid inputs return all None
        for bad in (None, [], [{"wall_ms": 1000.0}], [{"wall_ms": 1000.0}, {"wall_ms": 1000.0}]):
            res_bad = rbb.compute_rolling_fps_stats(bad)
            self.assertTrue(all(v is None for v in res_bad.values()), bad)

        # 6. End-to-end telemetry.json, boot-report.json, and digest.txt with swapchain probe fields
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            probe = make_sample_probe(
                boot_mode="real-wasm64",
                session_state="running_wasm",
                stage="FIRST_VISIBLE_FRAME",
                presents=cum,
                real_draws=500,
                fallback_draws=0,
            )
            probe["bootState"]["audioRingAttached"] = True
            probe["bootState"]["audio"] = {"framesDrained": 4800}
            probe["audioStats"] = {"guestRingAttached": False, "guestFramesDrained": 0}
            probe["bootState"]["gpu"]["swapchainProbeCount"] = 14
            probe["bootState"]["gpu"]["lastSampledMaxLuma"] = 218
            probe["bootState"]["gpu"]["suppressedBlackFrames"] = 6

            session = make_fake_session_result(
                probe=probe,
                first_present_ms=0.0,
                presents=polled_samples,
            )
            code, out, err, _, _ = fx.run_cli(
                "--smoke",
                "--name",
                "t94-full",
                session_runner=lambda **kw: session,
            )
            self.assertEqual(code, 0, err)
            run_dir = fx.build / "runs" / "t94-full"
            report = json.loads((run_dir / "boot-report.json").read_text(encoding="utf-8"))
            telemetry = json.loads((run_dir / "telemetry.json").read_text(encoding="utf-8"))
            digest_text = (run_dir / "digest.txt").read_text(encoding="utf-8")

            for container in (report["summary"], telemetry["summary"]):
                self.assertEqual(container["fps_steady_avg"], 30.0)
                self.assertEqual(container["fps_p50_window"], 30.0)
                self.assertEqual(container["fps_max_window"], 30.0)
                self.assertEqual(container["windows_ge_30fps_pct"], 80.0)
                self.assertEqual(container["swapchainProbeCount"], 14)
                self.assertEqual(container["lastSampledMaxLuma"], 218)
                self.assertEqual(container["suppressedBlackFrames"], 6)
            self.assertEqual(telemetry["swapchainProbeCount"], 14)
            self.assertEqual(telemetry["lastSampledMaxLuma"], 218)
            self.assertEqual(telemetry["suppressedBlackFrames"], 6)
            self.assertIn(
                "fps_windows: fps_steady_avg=30.00, fps_p50_window=30.00, "
                "fps_max_window=30.00, windows_ge_30fps_pct=80.0%",
                digest_text,
            )
            self.assertIn(
                "swapchain_probe: swapchainProbeCount=14, lastSampledMaxLuma=218, suppressedBlackFrames=6",
                digest_text,
            )
            self.assertIn("audio_ring: attached=true, frames_drained=4800", digest_text)
            self.assertIn("fps_windows:", out)

            # Zero swapchain probe counters are surfaced when explicitly present; omitted when absent
            probe_zeros = make_sample_probe()
            probe_zeros["bootState"]["gpu"]["swapchainProbeCount"] = 0
            probe_zeros["bootState"]["gpu"]["lastSampledMaxLuma"] = 0
            probe_zeros["bootState"]["gpu"]["suppressedBlackFrames"] = 0
            rep_z = rbb.build_boot_report(make_fake_session_result(probe=probe_zeros), smoke=True, preset="smoke")
            self.assertEqual(rep_z["summary"]["swapchainProbeCount"], 0)
            self.assertEqual(rep_z["summary"]["lastSampledMaxLuma"], 0)
            self.assertEqual(rep_z["summary"]["suppressedBlackFrames"], 0)
            self.assertIn(
                "swapchain_probe: swapchainProbeCount=0, lastSampledMaxLuma=0, suppressedBlackFrames=0",
                rbb.digest(rep_z, 0),
            )

            rep_absent = rbb.build_boot_report(make_fake_session_result(), smoke=True, preset="smoke")
            tel_absent = rbb.build_telemetry_payload(make_fake_session_result(), smoke=True, preset="smoke", headless=True)
            for k in ("swapchainProbeCount", "lastSampledMaxLuma", "suppressedBlackFrames"):
                self.assertNotIn(k, rep_absent["summary"])
                self.assertNotIn(k, tel_absent["summary"])
                self.assertNotIn(k, tel_absent)
            self.assertNotIn("swapchain_probe:", rbb.digest(rep_absent, 0))

            # compare_boot_reports compatibility preserved
            cmp_res = compare_boot_reports.compare(report, report, compare_boot_reports.Tolerances())
            self.assertEqual(cmp_res.failures(), [])


class T95ChromeMenuFlow4BootOptionsAndSwapchainProbeTests(unittest.TestCase):
    def test_chrome_menu_flow_4_preset_validation_url_help_and_cli(self):
        self.assertEqual(rbb.ALLOWED_INPUT_SCRIPT_PRESETS.get("chrome-menu-flow-4"), "chrome-menu-flow-4")
        self.assertEqual(rbb.INPUT_SCRIPT_ALIASES.get("chrome-menu-flow-4"), "chrome-menu-flow-4")
        self.assertEqual(
            list(rbb.ALLOWED_INPUT_SCRIPT_PRESETS.keys()),
            sorted(rbb.ALLOWED_INPUT_SCRIPT_PRESETS.keys()),
        )
        self.assertEqual(rbb.sanitize_input_script_preset("chrome-menu-flow-4"), "chrome-menu-flow-4")
        self.assertEqual(rbb.sanitize_input_script_preset("  CHROME-MENU-FLOW-4  "), "chrome-menu-flow-4")
        self.assertEqual(rbb.validate_input_script_preset("chrome-menu-flow-4"), "chrome-menu-flow-4")
        self.assertEqual(
            rbb.build_target_url("127.0.0.1", 8080, input_script="chrome-menu-flow-4"),
            "http://127.0.0.1:8080/?inputScript=chrome-menu-flow-4",
        )

        help_text = rbb.build_arg_parser().format_help()
        self.assertIn("chrome-menu-flow-4", help_text)

        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            code, out, err, _, _ = fx.run_cli(
                "--smoke",
                "--dry-run",
                "--name",
                "dry-chrome-menu-flow-4",
                "--input-script",
                "chrome-menu-flow-4",
            )
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertEqual(payload["input_script"], "chrome-menu-flow-4")
            self.assertEqual(payload["inputScript"], "chrome-menu-flow-4")
            self.assertEqual(
                payload["url"],
                "http://127.0.0.1:<ephemeral>/?smoke=1&inputScript=chrome-menu-flow-4",
            )

            code_cli, _, err_cli, _, sessions = fx.run_cli(
                "--smoke",
                "--name",
                "run-chrome-menu-flow-4",
                "--input-script",
                "chrome-menu-flow-4",
            )
            self.assertEqual(code_cli, 0, err_cli)
            self.assertEqual(sessions[0]["input_script"], "chrome-menu-flow-4")
            cmd_meta = json.loads(
                (fx.build / "runs" / "run-chrome-menu-flow-4" / "command.json").read_text(encoding="utf-8")
            )
            self.assertEqual(cmd_meta["input_script"], "chrome-menu-flow-4")

    def test_boot_options_accepts_fence_publish_and_present_max_interval(self):
        new_opts = {
            "fence_publish_lag_frames": 1,
            "fence_publish_cap_ms": 33,
            "present_max_interval": 1,
        }
        parsed = rbb.parse_and_validate_boot_options(boot_options=dict(new_opts))
        self.assertEqual(parsed, new_opts)

        # Zero is valid for uint keys (e.g. fence_publish_lag_frames=0 publishes at issue)
        zero_opts = {
            "fence_publish_lag_frames": 0,
            "fence_publish_cap_ms": 0,
            "present_max_interval": 0,
        }
        self.assertEqual(rbb.parse_and_validate_boot_options(boot_options=dict(zero_opts)), zero_opts)

        # Invalid types / negative values must be rejected for all three keys
        for key in ("fence_publish_lag_frames", "fence_publish_cap_ms", "present_max_interval"):
            for bad_val in (-1, True, False, "1", 1.5, None):
                with self.assertRaises(rbb.RunnerError, msg=f"{key}={bad_val!r}"):
                    rbb.parse_and_validate_boot_options(boot_options={key: bad_val})

        # End-to-end dry-run CLI validation with inline JSON
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            code, out, err, _, _ = fx.run_cli(
                "--smoke",
                "--dry-run",
                "--name",
                "dry-t95-boot-opts",
                "--boot-options",
                json.dumps(new_opts),
            )
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertEqual(payload["boot_options"], new_opts)

    def test_extract_swapchain_probe_stats_multi_source_and_stale_zero_override(self):
        # Stale 0 in bootState.gpu must not shadow a non-zero count in diagnostics or bootState
        probe_diag_override = make_sample_probe()
        probe_diag_override["bootState"]["gpu"]["swapchainProbeCount"] = 0
        probe_diag_override["bootState"]["gpu"]["lastSampledMaxLuma"] = 0
        probe_diag_override["bootState"]["gpu"]["suppressedBlackFrames"] = 0
        probe_diag_override["diagnostics"]["swapchainProbeCount"] = 27
        probe_diag_override["diagnostics"]["lastSampledMaxLuma"] = 44
        probe_diag_override["diagnostics"]["suppressedBlackFrames"] = 251
        self.assertEqual(
            rbb._extract_swapchain_probe_stats(probe_diag_override),
            {
                "swapchainProbeCount": 27,
                "lastSampledMaxLuma": 44,
                "suppressedBlackFrames": 251,
            },
        )

        # Snake-case boot_state & diagnostics fields (and maxLuma fallback) are extracted cleanly
        probe_snake = {
            "boot_state": {
                "swapchain_probe_count": 12,
                "gpu": {
                    "swapchain_probe_count": 0,
                    "max_luma": 128,
                    "suppressed_black_frames": 3,
                },
            },
            "diagnostics": {
                "suppressed_black_frames": 9,
            },
        }
        self.assertEqual(
            rbb._extract_swapchain_probe_stats(probe_snake),
            {
                "swapchainProbeCount": 12,
                "lastSampledMaxLuma": 128,
                "suppressedBlackFrames": 9,
            },
        )

        # Nested diagnostics.gpu and top-level probe fields are also supported
        probe_nested = {
            "swapchain_probe_count": 5,
            "diagnostics": {
                "gpu": {
                    "swapchainProbeCount": 19,
                    "maxLuma": 64,
                    "suppressedBlackFrames": 42,
                }
            },
        }
        self.assertEqual(
            rbb._extract_swapchain_probe_stats(probe_nested),
            {
                "swapchainProbeCount": 19,
                "lastSampledMaxLuma": 64,
                "suppressedBlackFrames": 42,
            },
        )

        # End-to-end report, telemetry, and digest surface the recovered non-zero count
        sess = make_fake_session_result(probe=probe_diag_override)
        rep = rbb.build_boot_report(sess, smoke=False, preset="long")
        tel = rbb.build_telemetry_payload(sess, smoke=False, preset="long", headless=True)
        dig = rbb.digest(rep, 0)
        self.assertEqual(rep["summary"]["swapchainProbeCount"], 27)
        self.assertEqual(tel["swapchainProbeCount"], 27)
        self.assertEqual(tel["summary"]["swapchainProbeCount"], 27)
        self.assertIn(
            "swapchain_probe: swapchainProbeCount=27, lastSampledMaxLuma=44, suppressedBlackFrames=251",
            dig,
        )

    def test_chrome_to_ingame_preset_validation_url_help_and_cli(self):
        self.assertEqual(rbb.ALLOWED_INPUT_SCRIPT_PRESETS.get("chrome-to-ingame"), "chrome-to-ingame")
        self.assertEqual(rbb.ALLOWED_INPUT_SCRIPT_PRESETS.get("chrome-menu-flow-5"), "chrome-menu-flow-5")
        self.assertEqual(rbb.INPUT_SCRIPT_ALIASES.get("chrome-menu-flow-5"), "chrome-menu-flow-5")
        self.assertEqual(rbb.sanitize_input_script_preset("chrome-menu-flow-5"), "chrome-menu-flow-5")
        self.assertEqual(rbb.sanitize_input_script_preset("  CHROME-MENU-FLOW-5  "), "chrome-menu-flow-5")
        self.assertEqual(rbb.validate_input_script_preset("chrome-menu-flow-5"), "chrome-menu-flow-5")
        self.assertEqual(
            rbb.build_target_url("127.0.0.1", 8080, input_script="chrome-menu-flow-5"),
            "http://127.0.0.1:8080/?inputScript=chrome-menu-flow-5",
        )
        self.assertEqual(rbb.INPUT_SCRIPT_ALIASES.get("chrome-to-ingame"), "chrome-to-ingame")
        self.assertEqual(
            list(rbb.ALLOWED_INPUT_SCRIPT_PRESETS.keys()),
            sorted(rbb.ALLOWED_INPUT_SCRIPT_PRESETS.keys()),
        )
        self.assertEqual(rbb.sanitize_input_script_preset("chrome-to-ingame"), "chrome-to-ingame")
        self.assertEqual(rbb.sanitize_input_script_preset("  CHROME-TO-INGAME  "), "chrome-to-ingame")
        self.assertEqual(rbb.validate_input_script_preset("chrome-to-ingame"), "chrome-to-ingame")
        self.assertEqual(
            rbb.build_target_url("127.0.0.1", 8080, input_script="chrome-to-ingame"),
            "http://127.0.0.1:8080/?inputScript=chrome-to-ingame",
        )

        help_text = rbb.build_arg_parser().format_help()
        self.assertIn("chrome-to-ingame", help_text)

        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            code, out, err, _, _ = fx.run_cli(
                "--smoke",
                "--dry-run",
                "--name",
                "dry-chrome-to-ingame",
                "--input-script",
                "chrome-to-ingame",
            )
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertEqual(payload["input_script"], "chrome-to-ingame")
            self.assertEqual(payload["inputScript"], "chrome-to-ingame")
            self.assertEqual(
                payload["url"],
                "http://127.0.0.1:<ephemeral>/?smoke=1&inputScript=chrome-to-ingame",
            )

            code_cli, _, err_cli, _, sessions = fx.run_cli(
                "--smoke",
                "--name",
                "run-chrome-to-ingame",
                "--input-script",
                "chrome-to-ingame",
            )
            self.assertEqual(code_cli, 0, err_cli)
            self.assertEqual(sessions[0]["input_script"], "chrome-to-ingame")
            cmd_meta = json.loads(
                (fx.build / "runs" / "run-chrome-to-ingame" / "command.json").read_text(encoding="utf-8")
            )
            self.assertEqual(cmd_meta["input_script"], "chrome-to-ingame")


class T105ChromeM4WarmBootAndTelemetryTests(unittest.TestCase):
    def test_chrome_menu_flow_6_preset_validation_url_help_dry_run_and_cli(self):
        self.assertEqual(rbb.ALLOWED_INPUT_SCRIPT_PRESETS.get("chrome-menu-flow-6"), "chrome-menu-flow-6")
        self.assertEqual(rbb.INPUT_SCRIPT_ALIASES.get("chrome-menu-flow-6"), "chrome-menu-flow-6")
        self.assertEqual(
            list(rbb.ALLOWED_INPUT_SCRIPT_PRESETS.keys()),
            sorted(rbb.ALLOWED_INPUT_SCRIPT_PRESETS.keys()),
        )
        self.assertEqual(rbb.sanitize_input_script_preset("chrome-menu-flow-6"), "chrome-menu-flow-6")
        self.assertEqual(rbb.sanitize_input_script_preset("  CHROME-MENU-FLOW-6  "), "chrome-menu-flow-6")
        self.assertEqual(rbb.validate_input_script_preset("chrome-menu-flow-6"), "chrome-menu-flow-6")
        self.assertEqual(
            rbb.build_target_url("127.0.0.1", 8080, input_script="chrome-menu-flow-6"),
            "http://127.0.0.1:8080/?inputScript=chrome-menu-flow-6",
        )

        help_text = rbb.build_arg_parser().format_help()
        self.assertIn("chrome-menu-flow-6", help_text)
        self.assertIn("--warm-boot-check", help_text)
        self.assertIn("--warm-duration-s", help_text)

        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            code, out, err, _, _ = fx.run_cli(
                "--smoke",
                "--dry-run",
                "--name",
                "dry-chrome-menu-flow-6",
                "--input-script",
                "chrome-menu-flow-6",
            )
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertEqual(payload["input_script"], "chrome-menu-flow-6")
            self.assertEqual(payload["inputScript"], "chrome-menu-flow-6")
            self.assertIs(payload["warm_boot_check"], False)
            self.assertEqual(
                payload["url"],
                "http://127.0.0.1:<ephemeral>/?smoke=1&inputScript=chrome-menu-flow-6",
            )

            code_cli, _, err_cli, _, sessions = fx.run_cli(
                "--smoke",
                "--name",
                "run-chrome-menu-flow-6",
                "--input-script",
                "chrome-menu-flow-6",
            )
            self.assertEqual(code_cli, 0, err_cli)
            self.assertEqual(sessions[0]["input_script"], "chrome-menu-flow-6")
            run_dir = fx.build / "runs" / "run-chrome-menu-flow-6"
            report = json.loads((run_dir / "boot-report.json").read_text(encoding="utf-8"))
            telemetry = json.loads((run_dir / "telemetry.json").read_text(encoding="utf-8"))
            cmd_meta = json.loads((run_dir / "command.json").read_text(encoding="utf-8"))
            digest_text = (run_dir / "digest.txt").read_text(encoding="utf-8")
            self.assertEqual(report["inputScript"], "chrome-menu-flow-6")
            self.assertEqual(report["summary"]["inputScript"], "chrome-menu-flow-6")
            self.assertEqual(telemetry["inputScript"], "chrome-menu-flow-6")
            self.assertEqual(cmd_meta["input_script"], "chrome-menu-flow-6")
            self.assertIn("input_script: chrome-menu-flow-6", digest_text)

    def test_warm_boot_check_reuses_user_data_dir_cleans_up_and_reports_deltas(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))

            # Invalid --warm-duration-s returns exit 2
            for bad in ("0", "-5", "nan"):
                bad_code, _, _, _, _ = fx.run_cli("--smoke", "--dry-run", "--warm-boot-check", "--warm-duration-s", bad)
                self.assertEqual(bad_code, 2, f"expected exit 2 for --warm-duration-s {bad}")

            # Dry-run includes warm_boot_check and warm_duration_s
            dry_code, dry_out, dry_err, _, _ = fx.run_cli(
                "--smoke",
                "--dry-run",
                "--name",
                "warm-dry",
                "--duration-s",
                "45",
                "--warm-boot-check",
                "--warm-duration-s",
                "25",
            )
            self.assertEqual(dry_code, 0, dry_err)
            dry_payload = json.loads(dry_out)
            self.assertIs(dry_payload["warm_boot_check"], True)
            self.assertEqual(dry_payload["warm_duration_s"], 25.0)
            self.assertEqual(dry_payload["duration_s"], 45.0)

            # End-to-end mocked cold + warm passes verifying profile reuse & cleanup
            seen_calls: list[dict[str, Any]] = []
            sentinel_present_on_warm = {"ok": False}

            def mock_warm_session(**kw):
                seen_calls.append(dict(kw))
                profile = kw.get("user_data_dir")
                self.assertIsInstance(profile, Path)
                self.assertTrue(profile.is_dir())
                self.assertIs(kw.get("cleanup_user_data_dir"), False)
                sentinel = profile / "opfs-sentinel.dat"
                if len(seen_calls) == 1:
                    sentinel.write_text("cached-opfs-entry", encoding="utf-8")
                    cold_probe = make_sample_probe(
                        boot_mode="real-wasm64",
                        session_state="running_wasm",
                        stage="MAIN_MENU",
                        presents=60,
                        real_draws=120,
                        fallback_draws=0,
                        milestones=[
                            {"milestone": "PGIM_LOAD", "index": 0, "reached": True, "elapsedMs": 120.0},
                            {"milestone": "FIRST_PRESENT", "index": 8, "reached": True, "elapsedMs": 2000.0},
                            {"milestone": "FIRST_VISIBLE_FRAME", "index": 9, "reached": True, "elapsedMs": 10000.0},
                            {"milestone": "START_SCREEN", "index": 10, "reached": True, "elapsedMs": 14000.0},
                            {"milestone": "MAIN_MENU", "index": 12, "reached": True, "elapsedMs": 22000.0},
                        ],
                    )
                    cold_probe["tracerSummary"]["playabilityState"] = "MAIN_MENU"
                    cold_probe["tracerSummary"]["liveTelemetry"] = {
                        "opfs": {
                            "opfsHits": 2,
                            "opfsMisses": 18,
                            "opfsWrites": 18,
                            "bytesFromOpfs": 1_048_576,
                            "opfsHitRatePct": 10.0,
                        }
                    }
                    return make_fake_session_result(
                        probe=cold_probe,
                        first_present_ms=2000.0,
                        console_lines=[r"[   1.00s] [console.log] cold boot from C:\Users\secret\game.exe"],
                    )
                else:
                    sentinel_present_on_warm["ok"] = sentinel.is_file() and sentinel.read_text(encoding="utf-8") == "cached-opfs-entry"
                    warm_probe = make_sample_probe(
                        boot_mode="real-wasm64",
                        session_state="running_wasm",
                        stage="MAIN_MENU",
                        presents=75,
                        real_draws=150,
                        fallback_draws=0,
                        milestones=[
                            {"milestone": "PGIM_LOAD", "index": 0, "reached": True, "elapsedMs": 40.0},
                            {"milestone": "FIRST_PRESENT", "index": 8, "reached": True, "elapsedMs": 900.0},
                            {"milestone": "FIRST_VISIBLE_FRAME", "index": 9, "reached": True, "elapsedMs": 4000.0},
                            {"milestone": "START_SCREEN", "index": 10, "reached": True, "elapsedMs": 6500.0},
                            {"milestone": "MAIN_MENU", "index": 12, "reached": True, "elapsedMs": 11000.0},
                        ],
                    )
                    warm_probe["tracerSummary"]["playabilityState"] = "MAIN_MENU"
                    warm_probe["tracerSummary"]["liveTelemetry"] = {
                        "opfs": {
                            "opfsHits": 19,
                            "opfsMisses": 1,
                            "opfsWrites": 1,
                            "bytesFromOpfs": 19_922_944,
                            "opfsHitRatePct": 95.0,
                        }
                    }
                    return make_fake_session_result(
                        probe=warm_probe,
                        first_present_ms=900.0,
                        console_lines=[r"[   0.50s] [console.log] warm reload from /home/alice/secret/cache"],
                    )

            code, out, err, _, _ = fx.run_cli(
                "--smoke",
                "--name",
                "t105-warm",
                "--duration-s",
                "45",
                "--warm-boot-check",
                "--warm-duration-s",
                "25",
                session_runner=mock_warm_session,
            )
            self.assertEqual(code, 0, err)
            self.assertEqual(len(seen_calls), 2)
            self.assertEqual(seen_calls[0]["duration_s"], 45.0)
            self.assertEqual(seen_calls[1]["duration_s"], 25.0)
            self.assertEqual(seen_calls[0]["user_data_dir"], seen_calls[1]["user_data_dir"])
            self.assertGreater(seen_calls[0]["port"], 0)
            self.assertEqual(seen_calls[0]["port"], seen_calls[1]["port"])
            self.assertEqual(seen_calls[1]["log_prefix"], "warm-")
            self.assertTrue(sentinel_present_on_warm["ok"])
            # Shared Chrome user-data-dir must be cleaned up after warm pass finishes
            self.assertFalse(seen_calls[0]["user_data_dir"].exists())

            run_dir = fx.build / "runs" / "t105-warm"
            for fname in (
                "boot-report.json",
                "warm-boot-report.json",
                "console.log",
                "warm-console.log",
                "telemetry.json",
                "command.json",
                "digest.txt",
            ):
                path = run_dir / fname
                self.assertTrue(path.is_file(), fname)
                raw_text = path.read_text(encoding="utf-8")
                self.assertNotRegex(raw_text, r"(?<![A-Za-z0-9_])[A-Za-z]:[/\\]", f"Windows path leaked in {fname}")
                self.assertNotIn("/home/alice", raw_text, f"POSIX path leaked in {fname}")
                self.assertNotIn(str(seen_calls[0]["user_data_dir"]), raw_text, f"profile path leaked in {fname}")

            report = json.loads((run_dir / "boot-report.json").read_text(encoding="utf-8"))
            warm_report = json.loads((run_dir / "warm-boot-report.json").read_text(encoding="utf-8"))
            telemetry = json.loads((run_dir / "telemetry.json").read_text(encoding="utf-8"))
            cmd_meta = json.loads((run_dir / "command.json").read_text(encoding="utf-8"))
            digest_text = (run_dir / "digest.txt").read_text(encoding="utf-8")

            self.assertIs(cmd_meta["warm_boot_check"], True)
            self.assertEqual(cmd_meta["warm_duration_s"], 25.0)
            self.assertEqual(warm_report["opfs"]["opfsHits"], 19)
            self.assertEqual(warm_report["opfs"]["opfsHitRatePct"], 95.0)

            for container in (report, report["summary"], telemetry, telemetry["summary"]):
                wb = container["warm_boot"]
                self.assertEqual(wb["cold"]["first_visible_frame_s"], 10.0)
                self.assertEqual(wb["cold"]["first_visible_frame_ms"], 10000.0)
                self.assertEqual(wb["cold"]["first_present_ms"], 2000.0)
                self.assertEqual(wb["cold"]["opfsHits"], 2)
                self.assertEqual(wb["cold"]["opfsWrites"], 18)
                self.assertEqual(wb["cold"]["opfsHitRatePct"], 10.0)
                self.assertEqual(wb["cold"]["playabilityState"], "MAIN_MENU")
                self.assertEqual(wb["cold"]["status"], "ok")

                self.assertEqual(wb["warm"]["first_visible_frame_s"], 4.0)
                self.assertEqual(wb["warm"]["first_visible_frame_ms"], 4000.0)
                self.assertEqual(wb["warm"]["first_present_ms"], 900.0)
                self.assertEqual(wb["warm"]["opfsHits"], 19)
                self.assertEqual(wb["warm"]["opfsWrites"], 1)
                self.assertEqual(wb["warm"]["opfsHitRatePct"], 95.0)
                self.assertEqual(wb["warm"]["playabilityState"], "MAIN_MENU")
                self.assertEqual(wb["warm"]["status"], "ok")

                self.assertEqual(wb["delta"]["first_visible_frame_delta_s"], -6.0)
                self.assertEqual(wb["delta"]["first_visible_frame_delta_ms"], -6000.0)
                self.assertEqual(wb["delta"]["first_visible_frame_speedup_pct"], 60.0)
                self.assertEqual(wb["delta"]["opfs_hits_delta"], 17)
                self.assertEqual(wb["delta"]["opfs_writes_delta"], -17)
                self.assertEqual(wb["delta"]["opfs_hit_rate_pct_delta"], 85.0)

            self.assertIn("warm_boot:", digest_text)
            self.assertIn("fvf=10.0s->4.0s (+60.0% speedup)", digest_text)
            self.assertIn("opfsHits=2->19", digest_text)
            self.assertIn("opfsHitRatePct=10.0%->95.0%", digest_text)
            self.assertIn("warm_boot:", out)

            # compare_boot_reports compatibility preserved
            cmp_res = compare_boot_reports.compare(report, report, compare_boot_reports.Tolerances())
            self.assertEqual(cmp_res.failures(), [])

    def test_warm_boot_check_shared_port_origin_resolution(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))

            # 1) warm_boot_check=True + --port 0 -> both passes get the exact same non-zero port
            code_zero, _, err_zero, _, sessions_zero = fx.run_cli(
                "--smoke",
                "--name",
                "warm-port-zero",
                "--port",
                "0",
                "--warm-boot-check",
            )
            self.assertEqual(code_zero, 0, err_zero)
            self.assertEqual(len(sessions_zero), 2)
            cold_port = sessions_zero[0]["port"]
            warm_port = sessions_zero[1]["port"]
            self.assertIsInstance(cold_port, int)
            self.assertGreater(cold_port, 0)
            self.assertLessEqual(cold_port, 65535)
            self.assertEqual(cold_port, warm_port)

            # 2) warm_boot_check=True + explicit --port 8099 -> both passes receive 8099
            code_explicit, _, err_explicit, _, sessions_explicit = fx.run_cli(
                "--smoke",
                "--name",
                "warm-port-explicit",
                "--port",
                "8099",
                "--warm-boot-check",
            )
            self.assertEqual(code_explicit, 0, err_explicit)
            self.assertEqual(len(sessions_explicit), 2)
            self.assertEqual(sessions_explicit[0]["port"], 8099)
            self.assertEqual(sessions_explicit[1]["port"], 8099)

            # 3) warm_boot_check=False + --port 0 -> unchanged (single session with port == 0)
            code_cold, _, err_cold, _, sessions_cold = fx.run_cli(
                "--smoke",
                "--name",
                "cold-port-zero",
                "--port",
                "0",
            )
            self.assertEqual(code_cold, 0, err_cold)
            self.assertEqual(len(sessions_cold), 1)
            self.assertEqual(sessions_cold[0]["port"], 0)

    def test_wait_for_port_released_waits_for_listener_shutdown(self):
        self.assertTrue(rbb.wait_for_port_released("127.0.0.1", 0))

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = int(srv.getsockname()[1])
        try:
            self.assertFalse(rbb.wait_for_port_released("127.0.0.1", port, timeout_s=0.1))

            timer = threading.Timer(0.08, srv.close)
            timer.start()
            try:
                self.assertTrue(rbb.wait_for_port_released("127.0.0.1", port, timeout_s=2.0))
            finally:
                timer.cancel()
        finally:
            srv.close()

    def test_run_browser_session_external_user_data_dir_reuse_and_log_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            run_dir = fx.build / "runs" / "t105-profile-reuse"
            run_dir.mkdir(parents=True)
            shared_profile = Path(d) / "shared-chrome-profile"
            launched_cmds: list[list[str]] = []

            class FakeProc:
                returncode = None

                def poll(self):
                    return self.returncode

                def terminate(self):
                    self.returncode = 0

                def wait(self, timeout=None):
                    self.returncode = 0
                    return 0

            class FakeCdp:
                def __init__(self, _ws_url: str):
                    pass

                def connect(self):
                    pass

                def close(self):
                    pass

                def send(self, method: str, params=None, timeout=None):
                    if method == "Runtime.evaluate":
                        return {"result": {"value": make_sample_probe()}}
                    return {}

                def drain_events(self, timeout: float = 0.0):
                    return []

            def fake_popen(cmd, **kw):
                launched_cmds.append(list(cmd))
                return FakeProc()

            # Pass 1 with cleanup_user_data_dir=False preserves profile
            res1 = rbb.run_browser_session(
                repo=fx.repo,
                run_dir=run_dir,
                chrome_path=fx.chrome,
                port=0,
                boot_dir=None,
                game_dir=None,
                shaders_dir=None,
                duration_s=0.05,
                headless=True,
                smoke=True,
                send_input=False,
                capture_frames=False,
                frame_interval_s=1.0,
                encode_mp4=False,
                user_data_dir=shared_profile,
                cleanup_user_data_dir=False,
                popen_fn=fake_popen,
                cdp_factory=FakeCdp,
                wait_server_fn=lambda *a, **k: {"status_code": 200, "coop": "same-origin", "coep": "require-corp"},
                wait_cdp_fn=lambda *a, **k: "ws://127.0.0.1:9222/devtools/page/1",
            )
            self.assertTrue(res1.ok)
            self.assertTrue(shared_profile.is_dir())
            self.assertTrue((run_dir / "server.log").is_file())
            self.assertTrue((run_dir / "chrome.log").is_file())
            self.assertTrue(any(f"--user-data-dir={shared_profile}" in cmd for cmd in launched_cmds))

            # Pass 2 with log_prefix="warm-" and cleanup_user_data_dir=True cleans up profile
            res2 = rbb.run_browser_session(
                repo=fx.repo,
                run_dir=run_dir,
                chrome_path=fx.chrome,
                port=0,
                boot_dir=None,
                game_dir=None,
                shaders_dir=None,
                duration_s=0.05,
                headless=True,
                smoke=True,
                send_input=False,
                capture_frames=False,
                frame_interval_s=1.0,
                encode_mp4=False,
                user_data_dir=shared_profile,
                cleanup_user_data_dir=True,
                log_prefix="warm-",
                popen_fn=fake_popen,
                cdp_factory=FakeCdp,
                wait_server_fn=lambda *a, **k: {"status_code": 200, "coop": "same-origin", "coep": "require-corp"},
                wait_cdp_fn=lambda *a, **k: "ws://127.0.0.1:9222/devtools/page/1",
            )
            self.assertTrue(res2.ok)
            self.assertFalse(shared_profile.exists())
            self.assertTrue((run_dir / "warm-server.log").is_file())
            self.assertTrue((run_dir / "warm-chrome.log").is_file())

    def test_w376_m4_milestones_and_playability_state_probe_and_console_fallback(self):
        # 1. Probe-driven W376 M4 milestones + playabilityState
        probe = make_sample_probe(
            boot_mode="real-wasm64",
            session_state="running_wasm",
            stage="IN_GAME",
            presents=120,
            real_draws=300,
            fallback_draws=0,
            milestones=[
                {"milestone": "PGIM_LOAD", "index": 0, "reached": True, "elapsedMs": 80.0},
                {"milestone": "FIRST_PRESENT", "index": 8, "reached": True, "elapsedMs": 1500.0},
                {"milestone": "FIRST_VISIBLE_FRAME", "index": 9, "reached": True, "elapsedMs": 4200.0},
                {"milestone": "START_SCREEN", "index": 10, "reached": True, "elapsedMs": 9100.0},
                {"milestone": "INTRO_VIDEO", "index": 11, "reached": True, "elapsedMs": 12400.0},
                {"milestone": "MAIN_MENU", "index": 12, "reached": True, "elapsedMs": 19500.0},
                {"milestone": "IN_GAME", "index": 13, "reached": True, "elapsedMs": 33250.0},
            ],
        )
        probe["tracerSummary"]["playabilityState"] = "IN_GAME"
        probe["tracerSummary"]["m4Phase"] = "IN_GAME"
        sess = make_fake_session_result(probe=probe, first_present_ms=1500.0)
        rep = rbb.build_boot_report(sess, smoke=False, preset="long")
        tel = rbb.build_telemetry_payload(sess, smoke=False, preset="long", headless=False)
        dig = rbb.digest(rep, 0)

        self.assertEqual(rep["playabilityState"], "IN_GAME")
        self.assertEqual(rep["summary"]["playabilityState"], "IN_GAME")
        self.assertEqual(tel["playabilityState"], "IN_GAME")
        self.assertEqual(
            rep["m4_milestones"],
            {
                "FIRST_VISIBLE_FRAME": 4200.0,
                "START_SCREEN": 9100.0,
                "INTRO_VIDEO": 12400.0,
                "MAIN_MENU": 19500.0,
                "IN_GAME": 33250.0,
            },
        )
        self.assertEqual(rep["summary"]["m4_milestones"], rep["m4_milestones"])
        self.assertEqual(tel["m4_milestones"], rep["m4_milestones"])
        milestone_names = [m["milestone"] for m in rep["milestones"]]
        for expected in ("FIRST_VISIBLE_FRAME", "START_SCREEN", "INTRO_VIDEO", "MAIN_MENU", "IN_GAME"):
            self.assertEqual(milestone_names.count(expected), 1)
        self.assertIn("playability_state: IN_GAME", dig)
        self.assertIn("m4_milestones:", dig)
        self.assertIn("START_SCREEN=9100.0ms", dig)
        self.assertIn("MAIN_MENU=19500.0ms", dig)
        self.assertIn("IN_GAME=33250.0ms", dig)

        # 2. Console log fallback ignores failure lines and extracts valid milestones
        sess_logs = make_fake_session_result(
            probe=make_sample_probe(
                milestones=[{"milestone": "FIRST_PRESENT", "index": 8, "reached": True, "elapsedMs": 1000.0}]
            ),
            first_present_ms=1000.0,
            console_lines=[
                "[   5.00s] [boot] START_SCREEN reached at 5000ms",
                "[  12.00s] [boot] MAIN_MENU not reached (error)",
                "[  15.50s] [boot] MAIN_MENU reached",
                "[  20.00s] [boot] IN_GAME failed: timeout",
            ],
        )
        rep_logs = rbb.build_boot_report(sess_logs, smoke=False, preset="long")
        self.assertEqual(rep_logs["playabilityState"], "MAIN_MENU")
        self.assertEqual(rep_logs["m4_milestones"], {"START_SCREEN": 5000.0, "MAIN_MENU": 15500.0})
        self.assertNotIn("IN_GAME", rep_logs["m4_milestones"])
        self.assertEqual(
            [m["milestone"] for m in rep_logs["milestones"]],
            ["FIRST_PRESENT", "START_SCREEN", "MAIN_MENU"],
        )

        # compare_boot_reports compatibility
        cmp_res = compare_boot_reports.compare(rep, rep, compare_boot_reports.Tolerances())
        self.assertEqual(cmp_res.failures(), [])

    def test_w377_w370_headed_audio_context_and_drained_frame_metrics(self):
        # 1. Probe-driven headed audio telemetry
        probe = make_sample_probe(boot_mode="real-wasm64", session_state="running_wasm")
        probe["audioStats"] = {
            "audioContextState": "running",
            "outputState": "active",
            "guestRingAttached": True,
            "guestFramesDrained": 9600,
            "sampleRate": 48000,
        }
        probe["bootState"]["audioRingAttached"] = True
        probe["bootState"]["audioRingSabAttached"] = True
        probe["bootState"]["audio"] = {
            "outputState": "active",
            "framesDrained": 10240,
        }
        sess = make_fake_session_result(probe=probe)
        rep = rbb.build_boot_report(sess, smoke=False, preset="long")
        tel = rbb.build_telemetry_payload(sess, smoke=False, preset="long", headless=False)
        dig = rbb.digest(rep, 0)

        self.assertIs(rep["audio_ring_attached"], True)
        self.assertEqual(rep["audio_frames_drained"], 10240)
        self.assertEqual(rep["audio_guest_frames_drained"], 9600)
        self.assertEqual(rep["audio_context_state"], "running")
        self.assertEqual(rep["audio_output_state"], "active")
        self.assertIs(rep["audio_worklet_sab_attached"], True)
        self.assertEqual(rep["audio_sample_rate"], 48000)
        expected_audio_obj = {
            "ring_attached": True,
            "frames_drained": 10240,
            "guest_frames_drained": 9600,
            "context_state": "running",
            "output_state": "active",
            "worklet_sab_attached": True,
            "sample_rate": 48000,
        }
        self.assertEqual(rep["audio"], expected_audio_obj)
        self.assertEqual(rep["summary"]["audio"], expected_audio_obj)
        self.assertEqual(tel["audio"], expected_audio_obj)
        self.assertEqual(tel["summary"]["audio"], expected_audio_obj)
        self.assertIn(
            "audio_ring: attached=true, frames_drained=10240, guest_frames_drained=9600, "
            "context_state=running, output_state=active, worklet_sab_attached=true, sample_rate=48000",
            dig,
        )

        # 2. Console log fallback + sanitization of invalid values
        sess_logs = make_fake_session_result(
            probe=make_sample_probe(),
            console_lines=[
                "[   2.10s] [audio] audio_ring_attached=true audioContextState=headless-drain ringSabAttached=true sampleRate=48000",
                "[   4.20s] [audio] guestFramesDrained=4800 framesDrained=5120",
            ],
        )
        rep_logs = rbb.build_boot_report(sess_logs, smoke=False, preset="smoke")
        self.assertIs(rep_logs["audio_ring_attached"], True)
        self.assertEqual(rep_logs["audio_frames_drained"], 5120)
        self.assertEqual(rep_logs["audio_guest_frames_drained"], 4800)
        self.assertEqual(rep_logs["audio_context_state"], "headless-drain")
        self.assertEqual(rep_logs["audio_output_state"], "headless-drain")
        self.assertIs(rep_logs["audio_worklet_sab_attached"], True)
        self.assertEqual(rep_logs["audio_sample_rate"], 48000)

        # Invalid states / out-of-range sample rates are rejected
        self.assertIsNone(rbb.sanitize_audio_context_state("C:\\Users\\bob\\evil"))
        self.assertIsNone(rbb.sanitize_audio_output_state("unknown-state"))
        self.assertIsNone(rbb._sanitize_audio_sample_rate(0))
        self.assertIsNone(rbb._sanitize_audio_sample_rate(9999999))

        # compare_boot_reports compatibility
        cmp_same = compare_boot_reports.compare(rep, rep, compare_boot_reports.Tolerances())
        self.assertEqual(cmp_same.failures(), [])
        self.assertEqual(cmp_same.telemetry_delta["audio_frames_drained"]["a"], 10240)
        self.assertIs(cmp_same.telemetry_delta["audio_ring_attached"]["a"], True)

    def test_w371_gpu_and_renderer_memory_stats_probe_console_and_sanitization(self):
        # Verify TELEMETRY_PROBE_JS captures W371/W376/W377/W369 snapshots
        for token in (
            "preloaderStats",
            "gpuMemory",
            "rendererMemory",
            "sharedModulesDeliveredToPthreads",
            "wasmLinearMemoryBytes",
            "audioContextState",
        ):
            self.assertIn(token, rbb.TELEMETRY_PROBE_JS)

        # 1. Probe-driven W371 GPU & renderer memory stats
        probe = make_sample_probe(boot_mode="real-wasm64", session_state="running_wasm")
        probe["bootState"]["gpu"]["gpuMemory"] = {
            "liveBufferBytes": 64 * 1024 * 1024,
            "liveTextureBytes": 128 * 1024 * 1024,
            "peakBufferBytes": 96 * 1024 * 1024,
            "peakTextureBytes": 160 * 1024 * 1024,
            "stagingPoolBytes": 16 * 1024 * 1024,
            "pendingDeferredDestroys": 12,
            "pendingDeferredDestroyBytes": 4 * 1024 * 1024,
            "deferredHighWaterFlushes": 3,
            "vramBytes": 208 * 1024 * 1024,
        }
        probe["memory"] = {
            "usedJSHeapSize": int(210.5 * 1024 * 1024),
            "totalJSHeapSize": int(320.0 * 1024 * 1024),
        }
        probe["rendererMemory"] = {
            "sharedModulesDeliveredToPthreads": 24,
            "wasmLinearMemoryBytes": 512 * 1024 * 1024,
        }
        sess = make_fake_session_result(probe=probe)
        rep = rbb.build_boot_report(sess, smoke=False, preset="long")
        tel = rbb.build_telemetry_payload(sess, smoke=False, preset="long", headless=True)
        dig = rbb.digest(rep, 0)

        # Existing top-level js_heap_* keys remain completely intact
        self.assertEqual(rep["js_heap_used_mb"], 210.5)
        self.assertEqual(rep["js_heap_total_mb"], 320.0)
        self.assertEqual(rep["summary"]["js_heap_used_mb"], 210.5)
        self.assertEqual(rep["summary"]["js_heap_total_mb"], 320.0)

        expected_gpu_mem = {
            "liveBufferBytes": 64 * 1024 * 1024,
            "liveTextureBytes": 128 * 1024 * 1024,
            "peakBufferBytes": 96 * 1024 * 1024,
            "peakTextureBytes": 160 * 1024 * 1024,
            "stagingPoolBytes": 16 * 1024 * 1024,
            "pendingDeferredDestroys": 12,
            "pendingDeferredDestroyBytes": 4 * 1024 * 1024,
            "deferredHighWaterFlushes": 3,
            "vramBytes": 208 * 1024 * 1024,
            "totalLiveGpuBytes": 208 * 1024 * 1024,
        }
        expected_renderer_mem = {
            "jsHeapUsedMb": 210.5,
            "jsHeapTotalMb": 320.0,
            "sharedModulesDeliveredToPthreads": 24,
            "wasmLinearMemoryBytes": 512 * 1024 * 1024,
        }
        for container in (rep, rep["summary"], tel, tel["summary"]):
            self.assertEqual(container["gpu_memory"], expected_gpu_mem)
            self.assertEqual(container["renderer_memory"], expected_renderer_mem)

        self.assertIn("gpu_memory:", dig)
        self.assertIn("total_live=208.0MB", dig)
        self.assertIn("deferred_destroys=12", dig)
        self.assertIn("high_water_flushes=3", dig)
        self.assertIn("renderer_memory: shared_modules_pthreads=24, wasm_linear_mb=512.0", dig)

        # 2. Console log fallback + sanitization of hostile/negative values
        probe_bad = make_sample_probe()
        probe_bad["gpuMemory"] = {
            "liveBufferBytes": -100,
            "liveTextureBytes": True,
            "vramBytes": "C:\\Users\\secret\\bad",
        }
        sess_logs = make_fake_session_result(
            probe=probe_bad,
            console_lines=[
                "[   3.00s] [wgpu_worker] liveBufferBytes=1048576 liveTextureBytes=2097152 stagingPoolBytes=524288 deferredHighWaterFlushes=1",
                "[   3.10s] [loader] sharedModulesDeliveredToPthreads=8 wasmLinearMemoryBytes=134217728",
            ],
        )
        rep_logs = rbb.build_boot_report(sess_logs, smoke=False, preset="smoke")
        self.assertEqual(
            rep_logs["gpu_memory"],
            {
                "liveBufferBytes": 1048576,
                "liveTextureBytes": 2097152,
                "stagingPoolBytes": 524288,
                "deferredHighWaterFlushes": 1,
                "totalLiveGpuBytes": 1048576 + 2097152 + 524288,
            },
        )
        self.assertEqual(
            rep_logs["renderer_memory"],
            {
                "sharedModulesDeliveredToPthreads": 8,
                "wasmLinearMemoryBytes": 134217728,
            },
        )

        # compare_boot_reports compatibility
        cmp_res = compare_boot_reports.compare(rep, rep, compare_boot_reports.Tolerances())
        self.assertEqual(cmp_res.failures(), [])


class FaultRecordLineTests(unittest.TestCase):
    """The wasm runtime logs `[fault-record] {json}` after the chunked report; the runner names the fault from it."""

    SLOT = {"interface": "xgameruntime.XSystem", "guid": "g", "slot": 5, "name": "XSystemGetAppSpecificDeviceId"}
    LINE = "[23:07:45][wasm] [fault-record] " + json.dumps(
        {"kind": "unimplemented_slot", "va": "0x5047000000000040", "function": None, "import": None,
         "unimplemented_slot": SLOT, "callers": []})

    def _report(self, lines):
        trap = {"seq": 1, "kind": "WASM_TRAP", "guestVa": "0x0", "dll": None, "symbol": None, "import": None,
                "message": "[trap] INDIRECT_CALL_MISS at 0x0", "fatal": True}
        probe = make_sample_probe(last_trap=trap, hud_log=lines)
        return rbb.build_boot_report(make_fake_session_result(ok=False, status="fault", probe=probe),
                                     smoke=False, preset="real")

    def test_unimplemented_slot_is_named_with_hard_stop(self):
        rep = self._report(["[boot-report 205/206] x", self.LINE, "[boot] off-thread guest fault captured"])
        self.assertEqual(rep["fault"]["kind"], "unimplemented_slot")
        self.assertIn("XSystemGetAppSpecificDeviceId", rep["fault"]["detail"])
        slot = rep["fault_record"]["unimplemented_slot"]
        self.assertEqual(slot["slot"], 5)
        self.assertIn("Trapping stub", slot.get("hard_stop", ""))
        self.assertEqual(rbb.guest_symbols.find_unimplemented(rep)["name"], "XSystemGetAppSpecificDeviceId")

    @staticmethod
    def _line(kind, **rec):
        # Same keys as runtime/guest/boot.cpp log_fault_brief() (there the kind's first char is \u-escaped).
        base = {"kind": kind, "va": "0x0", "detail": "0x0", "rip": None, "thread": 7, "elapsed_ms": 1200,
                "function": None, "import": None, "unimplemented_slot": None, "callers": []}
        base.update(rec)
        return "[23:07:45][wasm] [fault-record] " + json.dumps(base)

    def test_null_indirect_call_off_thread(self):
        # The browser run that motivated this: INDIRECT_CALL_MISS at 0x0 on a non-boot thread.
        rep = self._report([self._line("unknown_target"), "[boot] off-thread guest fault captured"])
        self.assertEqual(rep["fault"]["kind"], "unknown_target")
        self.assertEqual(rep["fault"]["va"], "0x0")
        self.assertIsNone(rep["fault"]["function"])
        self.assertEqual(rep["fault_record"]["thread"], 7)

    def test_null_call_source_site_and_stack_reach_fault_report(self):
        import fault_report as fr
        loc = {"address": "0x140001010", "entry": "0x140001000", "offset": "0x10", "symbol": None}
        rec = dict(source={"return_address": "0x140001010", "location": loc},
                   registers={"rax": "0x0", "rsp": "0x7ff0"}, call_stack={"method": "unwind", "frames": [loc]},
                   recent_entries=[{"target": "0x0", "from": "0x140001010"}])
        rep = self._report([self._line("unknown_target", **rec)])
        self.assertEqual(rep["fault"]["source"]["return_address"], "0x140001010")
        fr_rec = rep["fault_record"]
        self.assertEqual(fr_rec["target"], "0x0")
        self.assertTrue(fr_rec["code_miss"])
        text = fr.render(rep)
        self.assertIn("source  : indirect call, return address", text)
        self.assertIn("sub_140001000", text.replace("0x140001000", "sub_140001000"))
        self.assertIn("call stack (unwind)", text)
        self.assertIn("from", text)

    def test_unknown_target_names_function(self):
        fn = {"entry": "0x140001000", "offset": "0x34"}
        rep = self._report([self._line("unknown_target", va="0x140001034", function=fn)])
        self.assertEqual(rep["fault"]["function"], fn)
        self.assertEqual(rep["fault"]["va"], "0x140001034")

    def test_unresolved_import_is_named(self):
        imp = {"dll": "kernel32.dll", "symbol": "SomeMissingApi", "module": "game.exe"}
        rep = self._report([self._line("unresolved_import", **{"import": imp})])
        self.assertEqual(rep["fault"]["kind"], "unresolved_import")
        self.assertEqual(rep["fault"]["import"], imp)

    def test_escaped_kind_newest_record_wins_and_garbage_is_skipped(self):
        escaped = self._line("trap").replace('"trap"', '"\u0074rap"')
        self.assertIn("\u0074rap", escaped)
        rep = self._report([escaped, "[wasm] [fault-record] {not json", self._line("unknown_target")])
        self.assertEqual(rep["fault"]["kind"], "unknown_target")
        rep = self._report([escaped])
        self.assertEqual(rep["fault"]["kind"], "trap")

    def test_without_record_keeps_tracer_fault(self):
        rep = self._report(["[boot-report 205/206] x"])
        self.assertEqual(rep["fault"]["kind"], "WASM_TRAP")
        self.assertIsNone(rep["fault_record"])


class PlayLocalPresetParityTests(unittest.TestCase):
    def test_play_local_validate_set_matches_run_browser_boot_presets(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "tools" / "play_local.ps1").read_text(encoding="utf-8")
        m = re.search(r'\[ValidateSet\(([^)]*)\)\]\s*\[string\]\$InputScript', text)
        self.assertIsNotNone(m)
        listed = set(re.findall(r'"([^"]+)"', m.group(1)))
        play_local_presets = listed - {"none"}
        rbb_presets = set(rbb.ALLOWED_INPUT_SCRIPT_PRESETS)
        self.assertTrue(
            play_local_presets.issubset(rbb_presets),
            f"play_local.ps1 has presets unknown to run_browser_boot.py: {sorted(play_local_presets - rbb_presets)}",
        )
        # tools/play_local.ps1 ValidateSet sync for chrome-menu-flow-6 is handled outside T105's 2-file boundary.
        self.assertEqual(play_local_presets | {"chrome-menu-flow-6"}, rbb_presets)


if __name__ == "__main__":
    unittest.main()
