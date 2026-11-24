#!/usr/bin/env python3
"""Unit tests for tools/run_wasm_tests.py (Track 2 / O25).

Everything runs against fake executors: no Docker, Emscripten or Node is needed (the real run is
local-only by design).
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import run_wasm_tests as rwt  # noqa: E402

CTEST_JSON = json.dumps(
    {
        "kind": "ctestInfo",
        "version": {"major": 1, "minor": 0},
        "tests": [
            {"name": "alpha_tests", "command": ["/emsdk/node/bin/node", "/build/test_alpha.js"]},
            {"name": "playgta6_beta_tests", "command": ["/emsdk/node/bin/node", "/build/test_beta.js"]},
            {"name": "gamma", "command": ["/build/gamma.exe"]},
            {"name": "empty", "command": []},
        ],
    }
)


def _entries() -> list[rwt.CTestEntry]:
    return rwt.parse_ctest_json(CTEST_JSON)


class ArgParsingTests(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = rwt.config_from_args([])
        self.assertTrue(cfg.docker)
        self.assertFalse(cfg.keep_going)
        self.assertEqual(cfg.filters, [])
        self.assertEqual(cfg.emsdk_image, rwt.DEFAULT_EMSDK_IMAGE)
        self.assertEqual(cfg.node_image, rwt.DEFAULT_NODE_IMAGE)
        self.assertEqual(cfg.build_dir, REPO_ROOT / rwt.DEFAULT_BUILD_DIR_REL)
        self.assertIsNone(cfg.json_path)

    def test_all_options(self) -> None:
        cfg = rwt.config_from_args(
            ["--tests", "*scarlett*", "test_x", "--jobs", "3", "--build-dir", "out/b", "--keep-going",
             "--json", "r.json", "--emsdk-image", "e:1", "--node-image", "n:2", "--no-docker",
             "--timeout", "9.5", "--build-type", "Debug"]
        )
        self.assertEqual(cfg.filters, ["*scarlett*", "test_x"])
        self.assertEqual(cfg.jobs, 3)
        self.assertTrue(cfg.build_dir.is_absolute())
        self.assertEqual(cfg.build_dir.parts[-2:], ("out", "b"))
        self.assertTrue(cfg.keep_going)
        self.assertEqual(cfg.json_path, Path("r.json"))
        self.assertEqual((cfg.emsdk_image, cfg.node_image), ("e:1", "n:2"))
        self.assertFalse(cfg.docker)
        self.assertEqual(cfg.timeout_s, 9.5)
        self.assertEqual(cfg.build_type, "Debug")

    def test_rejects_bad_values(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                rwt.config_from_args(["--build-type", "Fast"])
            with self.assertRaises(SystemExit):
                rwt.config_from_args(["--jobs", "-1"])


class CTestDiscoveryTests(unittest.TestCase):
    def test_parse_extracts_targets(self) -> None:
        entries = _entries()
        self.assertEqual([e.name for e in entries], ["alpha_tests", "playgta6_beta_tests", "gamma"])
        self.assertEqual([e.target for e in entries], ["test_alpha", "test_beta", "gamma"])

    def test_parse_handles_windows_paths(self) -> None:
        text = json.dumps({"kind": "ctestInfo", "tests": [{"name": "w", "command": ["C:\\b\\test_w.js"]}]})
        self.assertEqual(rwt.parse_ctest_json(text)[0].target, "test_w")

    def test_parse_rejects_other_documents(self) -> None:
        with self.assertRaises(ValueError):
            rwt.parse_ctest_json(json.dumps({"kind": "other"}))
        with self.assertRaises(json.JSONDecodeError):
            rwt.parse_ctest_json("not json")

    def test_select_matches_name_or_target(self) -> None:
        entries = _entries()
        self.assertEqual(len(rwt.select_tests(entries, [])), 3)
        self.assertEqual([e.name for e in rwt.select_tests(entries, ["*beta*"])], ["playgta6_beta_tests"])
        self.assertEqual([e.name for e in rwt.select_tests(entries, ["test_alpha"])], ["alpha_tests"])
        self.assertEqual([e.name for e in rwt.select_tests(entries, ["gamma", "alpha_*"])], ["alpha_tests", "gamma"])
        self.assertEqual(rwt.select_tests(entries, ["nothing"]), [])


class ClassificationTests(unittest.TestCase):
    def test_classify(self) -> None:
        self.assertEqual(rwt.classify_result(0, False), rwt.STATUS_PASS)
        self.assertEqual(rwt.classify_result(1, False), rwt.STATUS_FAIL)
        self.assertEqual(rwt.classify_result(137, False), rwt.STATUS_FAIL)
        self.assertEqual(rwt.classify_result(None, False), rwt.STATUS_FAIL)
        self.assertEqual(rwt.classify_result(0, True), rwt.STATUS_TIMEOUT)

    def test_failure_lines_prefer_check_failures(self) -> None:
        out = "test_x\nsome noise\nFAIL [test_x.cpp:5]: a == b\nmore\nFAIL [test_x.cpp:9]: c\n1 check(s) failed\nlast\n"
        lines = rwt.extract_failure_lines(out)
        self.assertEqual(lines[0], "FAIL [test_x.cpp:5]: a == b")
        self.assertEqual(len(lines), 3)

    def test_failure_lines_detect_aborts_and_fall_back_to_tail(self) -> None:
        self.assertIn("Aborted", rwt.extract_failure_lines("start\nAborted(OOM)\n")[0])
        self.assertEqual(rwt.extract_failure_lines("a\nb\nc\nd\ne\n"), ["c", "d", "e"])
        self.assertEqual(rwt.extract_failure_lines(""), [])

    def test_failure_lines_are_truncated(self) -> None:
        self.assertEqual(len(rwt.extract_failure_lines("FAIL " + "x" * 500)[0]), 200)

    def test_probe_output_parsing(self) -> None:
        self.assertEqual(rwt.parse_probe_output("memory64:ok v24.22.3\n"), (True, "v24.22.3"))
        self.assertEqual(rwt.parse_probe_output("memory64:unsupported v22.16.0 bad"), (False, "v22.16.0"))
        self.assertEqual(rwt.parse_probe_output("garbage"), (False, "unknown"))


class ReportTests(unittest.TestCase):
    def _results(self) -> list[rwt.TestResult]:
        return [
            rwt.TestResult("a", "test_a", True, rwt.STATUS_PASS, 0, 1.25),
            rwt.TestResult("b", "test_b", True, rwt.STATUS_FAIL, 1, 2.0, failures=["FAIL [b.cpp:1]: x", "FAIL [b.cpp:2]: y"]),
            rwt.TestResult("c", "test_c", True, rwt.STATUS_TIMEOUT, None, 5.0),
            rwt.TestResult("d", "test_d", False, rwt.STATUS_NOT_BUILT, failures=["error: boom"]),
            rwt.TestResult("e", "test_e", True, rwt.STATUS_SKIPPED, skip_reason="needs a GPU"),
            rwt.TestResult("f", "test_f", True, rwt.STATUS_NOT_RUN),
        ]

    def test_summary_and_exit_code(self) -> None:
        results = self._results()
        self.assertEqual(
            rwt.summarize(results),
            {"total": 6, "passed": 1, "failed": 1, "timeout": 1, "skipped": 1, "not_built": 1, "not_run": 1},
        )
        self.assertEqual(rwt.exit_code_for(results), 1)
        self.assertEqual(rwt.exit_code_for(results[:1]), 0)
        skipped_only = [rwt.TestResult("s", "t", True, rwt.STATUS_SKIPPED, skip_reason="r")]
        self.assertEqual(rwt.exit_code_for(skipped_only), 0)
        self.assertEqual(rwt.exit_code_for([rwt.TestResult("n", "t", False, rwt.STATUS_NOT_BUILT)]), 1)

    def test_report_schema(self) -> None:
        cfg = rwt.Config(build_dir=Path("/b"), filters=["x*"], jobs=2)
        report = rwt.build_report(config=cfg, results=self._results(), node_version="v24.1.0",
                                  node_flags=[], build_ok=False, build_duration_s=3.14159,
                                  build_log="/b/build.log", started_at="2026-01-01T00:00:00+00:00",
                                  duration_s=9.87654)
        report = json.loads(json.dumps(report))  # must be JSON-serialisable
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["config"]["mode"], "docker")
        self.assertEqual(report["config"]["node_version"], "v24.1.0")
        self.assertEqual(report["config"]["filters"], ["x*"])
        self.assertEqual(report["build"], {"ok": False, "duration_s": 3.142, "log": "/b/build.log"})
        self.assertEqual(report["summary"]["total"], 6)
        first = report["tests"][0]
        self.assertEqual(set(first), {"name", "target", "built", "status", "returncode", "duration_s",
                                      "skip_reason", "failures"})
        self.assertEqual(report["tests"][4]["skip_reason"], "needs a GPU")
        local = rwt.build_report(config=rwt.Config(docker=False), results=[], node_version="v24", node_flags=[],
                                 build_ok=True, build_duration_s=0, build_log="", started_at="", duration_s=0)
        self.assertIsNone(local["config"]["emsdk_image"])

    def test_table_lists_every_test_and_details(self) -> None:
        text = rwt.format_table(self._results())
        for name in "abcdef":
            self.assertIn(f"\n{name} ", "\n" + text)
        self.assertIn("PASS", text)
        self.assertIn("TIMEOUT", text)
        self.assertIn("NOT-BUILT", text)
        self.assertIn("needs a GPU", text)
        self.assertIn("FAIL [b.cpp:2]: y", text)  # all failure lines of a failed test
        self.assertIn("1/6 passed, 1 failed, 1 timed out, 1 skipped, 1 not built, 1 not run", text)


class CommandBuilderTests(unittest.TestCase):
    def test_derived_image_tag(self) -> None:
        self.assertEqual(rwt.derived_image_tag("emscripten/emsdk:4.0.10"), "playgta6-wasm-tests:emsdk-4.0.10")

    def test_cmake_args(self) -> None:
        conf = rwt.cmake_configure_args("/src/runtime", "/build", "Release")
        self.assertEqual(conf[:2], ["emcmake", "cmake"])
        self.assertIn("-DPLAYGTA6_WASM_TESTS=ON", conf)
        build = rwt.cmake_build_args("/build", ["t1", "t2"], 8)
        self.assertEqual(build[-3:], ["--", "-k", "0"])
        self.assertEqual(build[build.index("--target") + 1:build.index("--target") + 3], ["t1", "t2"])
        self.assertNotIn("--target", rwt.cmake_build_args("/build", [], 0))
        self.assertNotIn("--parallel", rwt.cmake_build_args("/build", [], 0))

    def test_build_errors_for_target(self) -> None:
        log = ("[3/9] ok\nFAILED: CMakeFiles/test_a.dir/src/tests/test_a.cpp.o\nem++ ...\n"
               "/src/tests/test_a.cpp:4:1: error: expected ';'\nninja: build stopped\n")
        self.assertIn("expected ';'", rwt._build_errors_for(log, "test_a")[0])
        self.assertEqual(rwt._build_errors_for("fine", "test_a"), ["build failed (see build.log)"])


class FakeDocker:
    """Fake executor that answers every command run_suite issues in Docker mode."""

    def __init__(self, *, memory64: bool = True, unbuilt: set[str] = frozenset(),
                 test_outcomes: dict[str, tuple[int, str, bool]] | None = None) -> None:
        self.memory64 = memory64
        self.unbuilt = set(unbuilt)
        self.outcomes = test_outcomes or {}
        self.calls: list[list[str]] = []
        self.lock = threading.Lock()

    def __call__(self, argv, *, timeout=None, on_line=None, env=None, cwd=None, name=None, stdin_text=None):
        argv = list(argv)
        with self.lock:
            self.calls.append(argv)
        ok = rwt.CommandResult(0, "")
        if argv[:2] == ["docker", "version"]:
            return rwt.CommandResult(0, "27.0\n")
        if argv[:3] == ["docker", "image", "inspect"]:
            return ok
        if "-e" in argv and "node" in argv and argv[argv.index("-e") - 1] in ("node", rwt.NODE_MEMORY64_FLAG):
            flagged = rwt.NODE_MEMORY64_FLAG in argv
            if self.memory64 or flagged and self.memory64 is None:
                return rwt.CommandResult(0, "memory64:ok v24.1.0\n")
            return rwt.CommandResult(3, "memory64:unsupported v22.0.0 nope\n")
        if argv[:3] == ["docker", "run", "-d"] or argv[:3] == ["docker", "rm", "-f"]:
            return ok
        if argv[:2] == ["docker", "exec"]:
            inner = argv[3:]
            if inner[0] == "emcmake":
                return ok
            if inner[0] == "ctest":
                return rwt.CommandResult(0, CTEST_JSON)
            if inner[0] == "cmake":
                if on_line:
                    on_line("[1/2] Building CXX object x")
                if self.unbuilt:
                    return rwt.CommandResult(1, "[1/2] Building\nFAILED: test_beta.js\n/x.cpp:1: error: bad\n")
                return rwt.CommandResult(0, "[1/2] Building\n")
            if inner[0] == "ninja":
                target = inner[-1]
                return rwt.CommandResult(0, "ninja: no work to do.\n" if target not in self.unbuilt else "[1/1] Linking\n")
        for target, (rc, out, timed_out) in self.outcomes.items():
            if any(arg.endswith(f"/{target}.js") for arg in argv):
                return rwt.CommandResult(rc, out, timed_out, 0.5)
        if "node" in argv and any(a.endswith(".js") for a in argv):
            return rwt.CommandResult(0, "all good\n", False, 0.25)
        raise AssertionError(f"unexpected command: {argv}")


class RunSuiteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cfg = rwt.Config(build_dir=Path(self._tmp.name) / "build", jobs=2, timeout_s=10)

    def _run(self, fake, **overrides):
        for k, v in overrides.items():
            setattr(self.cfg, k, v)
        lines: list[str] = []
        code, report = rwt.run_suite(self.cfg, fake, out=lines.append)
        return code, report, "\n".join(lines)

    def test_all_pass(self) -> None:
        fake = FakeDocker()
        code, report, text = self._run(fake)
        self.assertEqual(code, 0)
        self.assertEqual(report["summary"]["passed"], 3)
        self.assertEqual(report["config"]["node_version"], "v24.1.0")
        self.assertTrue(report["build"]["ok"])
        self.assertIn("3/3 passed", text)
        self.assertTrue(any(c[:3] == ["docker", "rm", "-f"] for c in fake.calls))  # container cleaned up
        runs = [c for c in fake.calls if "timeout" in c]
        self.assertEqual(len(runs), 3)
        self.assertTrue(all(c[c.index("timeout") + 1:c.index("timeout") + 3] == ["-s", "KILL"] for c in runs))

    def test_failure_and_timeout_classification(self) -> None:
        fake = FakeDocker(test_outcomes={
            "test_alpha": (1, "FAIL [a.cpp:3]: x == y\n1 check(s) failed\n", False),
            "test_beta": (137, "", True),
        })
        code, report, _ = self._run(fake, keep_going=True)
        status = {t["name"]: t["status"] for t in report["tests"]}
        self.assertEqual(status, {"alpha_tests": "fail", "playgta6_beta_tests": "timeout", "gamma": "pass"})
        self.assertEqual(code, 1)
        failed = next(t for t in report["tests"] if t["name"] == "alpha_tests")
        self.assertEqual(failed["failures"][0], "FAIL [a.cpp:3]: x == y")
        self.assertEqual(failed["returncode"], 1)

    def test_fail_fast_without_keep_going(self) -> None:
        fake = FakeDocker(test_outcomes={"test_alpha": (1, "FAIL [a.cpp:3]: x\n", False)})
        code, report, _ = self._run(fake, jobs=1)
        status = [t["status"] for t in report["tests"]]
        self.assertEqual(status[0], "fail")
        self.assertEqual(status[1:], ["not-run", "not-run"])
        self.assertEqual(code, 1)

    def test_unbuilt_target_is_reported(self) -> None:
        fake = FakeDocker(unbuilt={"test_beta"})
        code, report, _ = self._run(fake, keep_going=True)
        beta = next(t for t in report["tests"] if t["target"] == "test_beta")
        self.assertFalse(beta["built"])
        self.assertEqual(beta["status"], "not-built")
        self.assertIn("error: bad", beta["failures"][0])
        self.assertEqual(report["summary"]["passed"], 2)
        self.assertFalse(report["build"]["ok"])
        self.assertEqual(code, 1)

    def test_filters_limit_build_and_runs(self) -> None:
        fake = FakeDocker()
        code, report, _ = self._run(fake, filters=["gamma"])
        self.assertEqual([t["name"] for t in report["tests"]], ["gamma"])
        build = next(c for c in fake.calls if c[:2] == ["docker", "exec"] and c[3] == "cmake")
        self.assertEqual(build[build.index("--target") + 1], "gamma")
        self.assertEqual(code, 0)

    def test_known_skip_is_reported_with_reason(self) -> None:
        saved = dict(rwt.KNOWN_SKIPS)
        self.addCleanup(lambda: (rwt.KNOWN_SKIPS.clear(), rwt.KNOWN_SKIPS.update(saved)))
        rwt.KNOWN_SKIPS["gamma"] = "toolchain limit"
        fake = FakeDocker()
        code, report, _ = self._run(fake)
        gamma = next(t for t in report["tests"] if t["name"] == "gamma")
        self.assertEqual((gamma["status"], gamma["skip_reason"]), ("skipped", "toolchain limit"))
        self.assertEqual(code, 0)
        self.assertFalse(any("gamma.js" in c for c in fake.calls))

    def test_node_without_memory64_aborts_instead_of_failing_every_test(self) -> None:
        fake = FakeDocker(memory64=False)
        with self.assertRaises(rwt.EnvironmentProblem) as ctx:
            self._run(fake)
        self.assertIn("memory64", str(ctx.exception))
        self.assertFalse(any("timeout" in c for c in fake.calls))  # no test was started

    def test_nothing_discovered_is_an_environment_problem(self) -> None:
        with self.assertRaises(rwt.EnvironmentProblem):
            self._run(FakeDocker(), filters=["does-not-exist"])

    def test_list_only(self) -> None:
        fake = FakeDocker()
        code, report, text = self._run(fake, list_only=True)
        self.assertEqual((code, report), (0, None))
        self.assertIn("alpha_tests\ttest_alpha", text)
        self.assertFalse(any(c[3:4] == ["cmake"] for c in fake.calls if c[:2] == ["docker", "exec"]))


class MainTests(unittest.TestCase):
    def test_main_reports_environment_problem_with_exit_2(self) -> None:
        original = rwt.run_suite
        rwt.run_suite = lambda cfg, *a, **k: (_ for _ in ()).throw(rwt.EnvironmentProblem("no docker"))
        try:
            err = io.StringIO()
            with redirect_stderr(err), redirect_stdout(io.StringIO()):
                self.assertEqual(rwt.main(["--no-docker"]), 2)
            self.assertIn("no docker", err.getvalue())
        finally:
            rwt.run_suite = original

    def test_main_writes_json_report(self) -> None:
        original = rwt.run_suite
        rwt.run_suite = lambda cfg, *a, **k: (0, {"schema_version": 1, "tests": []})
        try:
            with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
                path = Path(tmp) / "sub" / "report.json"
                self.assertEqual(rwt.main(["--json", str(path)]), 0)
                self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["schema_version"], 1)
        finally:
            rwt.run_suite = original


if __name__ == "__main__":
    unittest.main()
