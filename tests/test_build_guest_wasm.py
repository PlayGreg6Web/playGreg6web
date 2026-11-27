#!/usr/bin/env python3
"""Unit and CLI tests for tools/build_guest_wasm.py (Track 1 / R13, R21, R24, R25, R26, R27, R28, R30, R31, R33, R35, R37d, R37e, R37f; Track 4 / T44)."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import build_guest_wasm as bgw  # noqa: E402


def _write(path: Path, text: str = "unit\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _touch_lib(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"!<arch>\n")
    return path


def _fake_emsdk(repo: Path) -> None:
    em_dir = repo / "work" / "emsdk" / "upstream" / "emscripten"
    em_dir.mkdir(parents=True, exist_ok=True)
    _write(em_dir / "emcc", "#!/bin/sh\nexit 0\n")
    _write(em_dir / "em++", "#!/bin/sh\nexit 0\n")
    _write(em_dir / "emar", "#!/bin/sh\nexit 0\n")
    _write(em_dir / "emcc.bat", "@echo off\r\n")
    _write(em_dir / "em++.bat", "@echo off\r\n")
    _write(em_dir / "emar.bat", "@echo off\r\n")
    bin_dir = repo / "work" / "emsdk" / "upstream" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    _write(bin_dir / "llvm-nm", "#!/bin/sh\nexit 0\n")
    _write(bin_dir / "llvm-nm.exe", "MZ")


def _layout_sources(root: Path) -> Path:
    src = root / "recomp-out"
    _write(src / "shards" / "a.ll", "; ModuleID = 'a'\n")
    _write(src / "shards" / "b.ll", "; ModuleID = 'b'\n")
    _write(src / "pg6_import_thunks.cpp", "void pg6_thunk() {}\n")
    _write(src / "guest_table.cpp", "void guest_table() {}\n")
    _write(src / "import_thunks.cpp", "void import_thunks() {}\n")
    _write(src / "notes.txt", "not a source\n")
    return src


class RecordingRunner:
    """Fake emcc/em++ that writes requested -o outputs and records argv."""

    def __init__(
        self,
        *,
        fail_ids: frozenset[str] | None = None,
        fail_outputs: frozenset[str] | None = None,
        delay: float = 0.0,
        nm_outputs: Mapping[str, str] | None = None,
    ) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.compile_calls: list[tuple[str, ...]] = []
        self.link_calls: list[tuple[str, ...]] = []
        self.side_link_calls: list[tuple[str, ...]] = []
        self.ar_calls: list[tuple[str, ...]] = []
        self.nm_calls: list[tuple[str, ...]] = []
        self.fail_ids = fail_ids or frozenset()
        self.fail_outputs = fail_outputs or frozenset()
        self.delay = delay
        self.nm_outputs = nm_outputs
        self._lock = threading.Lock()
        self.current = 0
        self.peak = 0

    def __call__(self, argv: Sequence[str]) -> bgw.CommandResult:
        tokens = tuple(str(a) for a in argv)
        with self._lock:
            self.calls.append(tokens)
            self.current += 1
            self.peak = max(self.peak, self.current)
        try:
            if self.delay:
                time.sleep(self.delay)
            joined = " ".join(tokens)
            if "check_wasm_exports.py" in joined:
                return bgw.CommandResult(
                    returncode=0,
                    stdout=json.dumps({"ok": True, "errors": []}),
                )
            if tokens and Path(tokens[0]).name.lower().startswith("llvm-nm"):
                with self._lock:
                    self.nm_calls.append(tokens)
                if self.nm_outputs is not None:
                    out_lines: list[str] = []
                    for tok in tokens[1:]:
                        if tok.startswith("-"):
                            continue
                        for key, val in self.nm_outputs.items():
                            if key in tok:
                                out_lines.append(val)
                    return bgw.CommandResult(
                        returncode=0,
                        stdout="\n".join(out_lines) + ("\n" if out_lines else ""),
                    )
                return bgw.CommandResult(returncode=0, stdout="")
            if tokens and Path(tokens[0]).name.lower().startswith("emar"):
                with self._lock:
                    self.ar_calls.append(tokens)
                for tok in tokens[1:]:
                    if Path(tok).name in self.fail_outputs:
                        return bgw.CommandResult(returncode=1, stderr=f"failed {Path(tok).name}")
                return bgw.CommandResult(returncode=0)
            out_path = _dash_o(tokens)
            is_side_link = bgw.SIDE_MODULE_FLAG in tokens or any(
                t.startswith("-sSIDE_MODULE=") for t in tokens
            )
            is_link = "-lexports.js" in tokens or any(
                Path(t).name in {bgw.RUNTIME_JS, bgw.RUNTIME_WASM} for t in tokens
            )
            source_token = _source_token(tokens)
            if source_token and Path(source_token).name in self.fail_ids:
                return bgw.CommandResult(returncode=1, stderr=f"failed {Path(source_token).name}")
            if out_path is not None and Path(out_path).name in self.fail_outputs:
                return bgw.CommandResult(returncode=1, stderr=f"failed {Path(out_path).name}")
            if is_side_link and out_path is not None and Path(out_path).suffix.lower() == ".wasm":
                with self._lock:
                    self.side_link_calls.append(tokens)
                wasm = Path(out_path)
                wasm.parent.mkdir(parents=True, exist_ok=True)
                wasm.write_bytes(b"\x00asm\x01\x00\x00\x00")
                return bgw.CommandResult(returncode=0)
            if is_link and out_path is not None and Path(out_path).suffix.lower() == ".js":
                with self._lock:
                    self.link_calls.append(tokens)
                js = Path(out_path)
                js.parent.mkdir(parents=True, exist_ok=True)
                js.write_text("export default function createRuntimeModule() {}", encoding="utf-8")
                (js.with_suffix(".wasm")).write_bytes(b"\x00asm\x01\x00\x00\x00")
                return bgw.CommandResult(returncode=0)
            if out_path is not None:
                with self._lock:
                    self.compile_calls.append(tokens)
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                Path(out_path).write_bytes(b"obj")
                return bgw.CommandResult(returncode=0)
            return bgw.CommandResult(returncode=1, stderr="no -o")
        finally:
            with self._lock:
                self.current -= 1


def _dash_o(argv: Sequence[str]) -> str | None:
    for idx, token in enumerate(argv):
        if token == "-o" and idx + 1 < len(argv):
            return argv[idx + 1]
    return None


def _source_token(argv: Sequence[str]) -> str | None:
    for token in argv:
        suffix = Path(token).suffix.lower()
        if suffix in bgw.SOURCE_SUFFIXES:
            return token
    return None


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = bgw.main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


class TestDiscovery(unittest.TestCase):
    def test_glob_discovers_shards_and_thunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = _layout_sources(Path(tmp))
            units = bgw.discover_sources(src)
            ids = [u.unit_id for u in units]
            self.assertEqual(
                ids,
                [
                    "guest_table.cpp",
                    "import_thunks.cpp",
                    "pg6_import_thunks.cpp",
                    "shards/a.ll",
                    "shards/b.ll",
                ],
            )
            kinds = {u.unit_id: u.kind for u in units}
            self.assertEqual(kinds["shards/a.ll"], "ll")
            self.assertEqual(kinds["pg6_import_thunks.cpp"], "cpp")

    def test_manifest_and_glob_union(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "out"
            _write(src / "shards" / "from_glob.ll", "; glob\n")
            _write(src / "listed.cpp", "int x;\n")
            _write(src / "only_in_manifest.c", "int y;\n")
            (src / "build-manifest.json").write_text(
                json.dumps(
                    {
                        "sources": ["listed.cpp", {"path": "only_in_manifest.c"}],
                        "shards": [{"file": "shards/from_glob.ll"}],
                    }
                ),
                encoding="utf-8",
            )
            ids = [u.unit_id for u in bgw.discover_sources(src)]
            self.assertIn("listed.cpp", ids)
            self.assertIn("only_in_manifest.c", ids)
            self.assertIn("shards/from_glob.ll", ids)

    def test_r9_pipeline_manifest_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "main"
            _write(src / "lift" / "shard-00000.ll", "; shard\n")
            _write(src / "lift" / "functions.ll", "; registration\n")
            _write(src / "stubs" / "pg6_import_thunks.cpp", "void thunk() {}\n")
            _write(src / "stubs" / "pg6_guest_seam.h", "#pragma once\n")
            _write(src / "image" / "guest.pgim.json", "{}\n")
            (src / "build-manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "playgta6-recomp-build",
                        "schema_version": 1,
                        "lift": {
                            "shards": [
                                {
                                    "index": 0,
                                    "ll": "lift/shard-00000.ll",
                                    "obj": None,
                                }
                            ],
                            "registration": {"ll": "lift/functions.ll", "obj": None},
                        },
                        "files": [
                            {"path": "stubs/pg6_import_thunks.cpp", "size": 1},
                            {"path": "stubs/pg6_import_thunks.h", "size": 1},
                            {"path": "image/guest.pgim", "size": 1},
                            {"path": "lift/shard-00000.ll", "size": 1},
                            {"path": "lift/functions.ll", "size": 1},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            ids = [u.unit_id for u in bgw.discover_sources(src)]
            self.assertEqual(
                ids,
                [
                    "lift/functions.ll",
                    "lift/shard-00000.ll",
                    "stubs/pg6_import_thunks.cpp",
                ],
            )


class TestCacheKeys(unittest.TestCase):
    def test_content_and_flag_changes_invalidate(self) -> None:
        flags_o2 = bgw.compile_flags_for(opt="O2", kind="ll")
        flags_o3 = bgw.compile_flags_for(opt="O3", kind="ll")
        key_a = bgw.cache_key_for(b"ir-a", flags_o2, kind="ll")
        key_b = bgw.cache_key_for(b"ir-b", flags_o2, kind="ll")
        key_flag = bgw.cache_key_for(b"ir-a", flags_o3, kind="ll")
        self.assertEqual(len(key_a), 64)
        self.assertNotEqual(key_a, key_b)
        self.assertNotEqual(key_a, key_flag)
        self.assertEqual(key_a, bgw.cache_key_for(b"ir-a", flags_o2, kind="ll"))


class TestBuildOrchestrator(unittest.TestCase):
    def _repo(self, tmp: Path) -> Path:
        repo = tmp / "repo"
        _fake_emsdk(repo)
        _touch_lib(repo / "runtime" / "build-wasm64" / "libplaygta6_core.a")
        (repo / "runtime").mkdir(exist_ok=True)
        return repo

    def test_cache_hit_skips_recompile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _layout_sources(root)
            cache = root / "cache"
            out = root / "dist"
            runner = RecordingRunner()
            first = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=out,
                jobs=2,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner,
                which_fn=lambda _name: None,
            )
            self.assertTrue(first.ok, first.errors)
            self.assertEqual(first.cache_misses, 5)
            self.assertEqual(first.cache_hits, 0)
            compiles_first = len(runner.compile_calls)

            runner2 = RecordingRunner()
            second = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=out,
                jobs=2,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner2,
                which_fn=lambda _name: None,
            )
            self.assertTrue(second.ok, second.errors)
            self.assertEqual(second.cache_hits, 5)
            self.assertEqual(second.cache_misses, 0)
            self.assertEqual(runner2.compile_calls, [])
            self.assertEqual(len(runner2.link_calls), 1)
            self.assertGreater(compiles_first, 0)

            _write(src / "shards" / "a.ll", "; ModuleID = 'a-changed'\n")
            runner3 = RecordingRunner()
            third = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=out,
                jobs=2,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner3,
                which_fn=lambda _name: None,
            )
            self.assertTrue(third.ok, third.errors)
            self.assertEqual(third.cache_hits, 4)
            self.assertEqual(third.cache_misses, 1)
            self.assertEqual(len(runner3.compile_calls), 1)

            runner4 = RecordingRunner()
            fourth = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=out,
                jobs=2,
                opt="O3",
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner4,
                which_fn=lambda _name: None,
            )
            self.assertTrue(fourth.ok, fourth.errors)
            self.assertEqual(fourth.cache_hits, 0)
            self.assertEqual(fourth.cache_misses, 5)
            self.assertEqual(len(runner4.compile_calls), 5)

    def test_parallel_compile_fanout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _layout_sources(root)
            runner = RecordingRunner(delay=0.05)
            report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache",
                out_dir=root / "dist",
                jobs=4,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner,
                which_fn=lambda _name: None,
            )
            self.assertTrue(report.ok, report.errors)
            self.assertGreaterEqual(runner.peak, 2)
            self.assertEqual(len(runner.compile_calls), 5)

    def test_shard_failure_reports_unit_and_skips_success_masking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _layout_sources(root)
            runner = RecordingRunner(fail_ids=frozenset({"b.ll"}))
            report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache",
                out_dir=root / "dist",
                jobs=4,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner,
                which_fn=lambda _name: None,
            )
            self.assertFalse(report.ok)
            self.assertTrue(any("shards/b.ll" in e for e in report.errors), report.errors)
            failed = [u for u in report.sources if u.status == "error"]
            self.assertEqual(len(failed), 1)
            self.assertEqual(failed[0].unit.unit_id, "shards/b.ll")
            self.assertEqual(runner.link_calls, [])
            compiled_ok = [u for u in report.sources if u.status == "compiled"]
            self.assertGreaterEqual(len(compiled_ok), 1)

    def test_link_command_has_exact_wasm64_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _layout_sources(root)
            runner = RecordingRunner()
            report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache",
                out_dir=root / "dist",
                jobs=1,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner,
                which_fn=lambda _name: None,
            )
            self.assertTrue(report.ok, report.errors)
            self.assertEqual(len(runner.link_calls), 1)
            link = runner.link_calls[0]
            for flag in (
                "-sMEMORY64=1",
                "-pthread",
                "-lexports.js",
                "-sMODULARIZE=1",
                "-sEXPORT_NAME=createRuntimeModule",
                "-sPTHREAD_POOL_SIZE=4",
                "-sALLOW_MEMORY_GROWTH=1",
                "-sENVIRONMENT=worker",
                "-sERROR_ON_UNDEFINED_SYMBOLS=0",
            ):
                self.assertIn(flag, link)
            self.assertIn("libplaygta6_core.a", [Path(t).name for t in link])
            self.assertEqual(report.outputs, (bgw.RUNTIME_JS, bgw.RUNTIME_WASM))
            self.assertTrue((root / "dist" / "runtime.js").is_file())
            self.assertTrue((root / "dist" / "runtime.wasm").is_file())
            self.assertEqual((*bgw.LINK_FLAGS, "-sASSERTIONS=0", "-sSAFE_HEAP=0"), report.link_flags)

            compile_cmd = runner.compile_calls[0]
            self.assertIn("-sMEMORY64=1", compile_cmd)
            self.assertIn("-pthread", compile_cmd)
            self.assertIn("-O2", compile_cmd)
            self.assertIn("-c", compile_cmd)

    def test_compile_argv_builder_and_link_argv_builder(self) -> None:
        argv = bgw.build_compile_argv(
            Path("emcc"),
            Path("shard.ll"),
            Path("shard.o"),
            bgw.compile_flags_for(opt="O2", kind="ll"),
        )
        self.assertEqual(argv[0], "emcc")
        self.assertIn("-sMEMORY64=1", argv)
        self.assertIn("-pthread", argv)
        self.assertIn("-x", argv)
        self.assertIn("ir", argv)

        link = bgw.build_link_argv(
            "em++",
            [Path("a.o"), Path("b.o")],
            output_js=Path("web/dist/runtime.js"),
            runtime_libs=[Path("libplaygta6_core.a")],
        )
        self.assertEqual(link[0], "em++")
        self.assertIn("-sMEMORY64=1", link)
        self.assertIn("-pthread", link)
        self.assertIn("-lexports.js", link)
        self.assertIn("-sMODULARIZE=1", link)
        self.assertEqual(link[-2], "-o")
        self.assertTrue(str(link[-1]).replace("\\", "/").endswith("web/dist/runtime.js"))


class TestCli(unittest.TestCase):
    def test_check_only_discovers_work_emsdk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            _fake_emsdk(repo)
            old_path = os.environ.get("PATH", "")
            old_emsdk = os.environ.pop("EMSDK", None)
            try:
                os.environ["PATH"] = str(repo)
                code, out, err = _run_cli(
                    [
                        "--check-only",
                        "--json",
                        "--repo-root",
                        str(repo),
                    ]
                )
            finally:
                os.environ["PATH"] = old_path
                if old_emsdk is not None:
                    os.environ["EMSDK"] = old_emsdk
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["mode"], "check-only")
            self.assertTrue(payload["toolchain"]["emcc_found"])
            self.assertEqual(payload["toolchain"]["origin"], "work/emsdk")

    def test_check_only_missing_toolchain_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "empty"
            repo.mkdir()
            toolchain = bgw.discover_emscripten(
                repo_root=repo,
                env={},
                which_fn=lambda _name: None,
            )
            self.assertFalse(toolchain.ok)
            old_path = os.environ.get("PATH", "")
            old_emsdk = os.environ.pop("EMSDK", None)
            try:
                os.environ["PATH"] = str(repo)
                code, out, _err = _run_cli(
                    ["--check-only", "--json", "--repo-root", str(repo)]
                )
            finally:
                os.environ["PATH"] = old_path
                if old_emsdk is not None:
                    os.environ["EMSDK"] = old_emsdk
            self.assertEqual(code, 1)
            payload = json.loads(out)
            self.assertFalse(payload["ok"])
            self.assertTrue(payload["errors"])

    def test_dry_run_and_json_are_sanitized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _fake_emsdk(repo)
            src = _layout_sources(root)
            user = os.environ.get("USERNAME") or os.environ.get("USER") or "nobody"
            code, out, err = _run_cli(
                [
                    str(src),
                    "--dry-run",
                    "--json",
                    "--jobs",
                    "2",
                    "--repo-root",
                    str(repo),
                    "--cache-dir",
                    str(root / "cache"),
                    "--out-dir",
                    str(root / "dist"),
                ]
            )
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["mode"], "dry-run")
            self.assertEqual(payload["source_count"], 5)
            self.assertIn("-sMEMORY64=1", payload["link_flags"])
            self.assertIn("-pthread", payload["link_flags"])
            self.assertIn("-lexports.js", payload["link_flags"])
            self.assertIn("-sMODULARIZE=1", payload["link_flags"])
            dumped = json.dumps(payload)
            self.assertNotIn("\\\\", dumped)
            if len(user) >= 3:
                self.assertNotRegex(dumped, rf"(?<![A-Za-z0-9_]){user}(?![A-Za-z0-9_])")
            for unit in payload["sources"]:
                self.assertIn(unit["status"], {"planned", "hit"})
                self.assertTrue(unit["id"].replace("\\", "/") == unit["id"])

    def test_dry_run_text_lists_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _fake_emsdk(repo)
            src = _layout_sources(root)
            code, out, err = _run_cli(
                [
                    str(src),
                    "--dry-run",
                    "--sanitize",
                    "--repo-root",
                    str(repo),
                    "--cache-dir",
                    str(root / "cache"),
                    "--out-dir",
                    str(root / "dist"),
                ]
            )
            self.assertEqual(code, 0, err)
            self.assertIn("Guest Wasm64 build [PASS]", out)
            self.assertIn("shards/a.ll", out)
            self.assertIn("-sMEMORY64=1", out)
            self.assertIn("-lexports.js", out)

    def test_sanitize_argv_strips_host_paths(self) -> None:
        argv = [
            r"C:\Users\someone\emsdk\upstream\emscripten\emcc.bat",
            "-sMEMORY64=1",
            "-pthread",
            r"C:\Users\someone\work\shards\a.ll",
            "-o",
            r"C:\Users\someone\work\wasm_obj_cache\aa\a.o",
        ]
        cleaned = bgw.sanitize_argv(argv)
        self.assertEqual(cleaned[0], "emcc")
        self.assertIn("-sMEMORY64=1", cleaned)
        self.assertEqual(cleaned[-1], "a.o")
        joined = " ".join(cleaned)
        self.assertNotIn("someone", joined)
        self.assertNotIn("C:\\", joined)


def _module_layout(root: Path, name: str, *, shards: int = 2) -> Path:
    module = root / name
    for index in range(shards):
        _write(
            module / "lift" / f"shard-{index:05d}.ll",
            f"; ModuleID = '{name}-{index}'\n",
        )
    _write(module / "lift" / "functions.ll", f"; registration {name}\n")
    _write(module / "stubs" / "pg6_import_thunks.cpp", f"void thunk_{name}() {{}}\n")
    _write(module / "stubs" / "pg6_guest_seam.h", "#pragma once\n")
    (module / "build-manifest.json").write_text(
        json.dumps(
            {
                "schema": "playgta6-recomp-build",
                "schema_version": 1,
                "module": name,
                "lift": {
                    "shards": [
                        {
                            "index": index,
                            "ll": f"lift/shard-{index:05d}.ll",
                            "instructions": 10 * (index + 1),
                        }
                        for index in range(shards)
                    ],
                    "registration": {"ll": "lift/functions.ll"},
                },
                "files": [
                    {"path": "stubs/pg6_import_thunks.cpp"},
                    {"path": "lift/functions.ll"},
                ]
                + [{"path": f"lift/shard-{index:05d}.ll"} for index in range(shards)],
            }
        ),
        encoding="utf-8",
    )
    return module


def _program_layout(root: Path) -> Path:
    src = root / "program-out"
    _module_layout(src, "main", shards=3)
    _module_layout(src, "vcruntime140", shards=2)
    _write(src / "exports" / "pg6_export_thunks.cpp", "void pg6_export_thunks() {}\n")
    _write(src / "program_modules.cpp", "const int pg6_program_modules = 2;\n")
    _write(src / "pg6_program.cpp", "extern const int pg6_program_modules;\n")
    (src / "program-manifest.json").write_text(
        json.dumps(
            {
                "schema": "playgta6-recomp-program",
                "schema_version": 1,
                "modules": [
                    {"module": "main", "file": "game.exe"},
                    {"module": "vcruntime140", "file": "vcruntime140.dll"},
                ],
                "files": [
                    {"path": "exports/pg6_export_thunks.cpp"},
                    {"path": "program_modules.cpp"},
                    {"path": "pg6_program.cpp"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return src


class TestProgramManifestDiscovery(unittest.TestCase):
    def test_discovers_every_module_and_program_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = _program_layout(Path(tmp))
            units = bgw.discover_sources(src)
            ids = [u.unit_id for u in units]
            self.assertEqual(
                ids,
                [
                    "exports/pg6_export_thunks.cpp",
                    "main/lift/functions.ll",
                    "main/lift/shard-00000.ll",
                    "main/lift/shard-00001.ll",
                    "main/lift/shard-00002.ll",
                    "main/stubs/pg6_import_thunks.cpp",
                    "pg6_program.cpp",
                    "program_modules.cpp",
                    "vcruntime140/lift/functions.ll",
                    "vcruntime140/lift/shard-00000.ll",
                    "vcruntime140/lift/shard-00001.ll",
                    "vcruntime140/stubs/pg6_import_thunks.cpp",
                ],
            )
            by_id = {u.unit_id: u for u in units}
            self.assertEqual(by_id["main/lift/shard-00001.ll"].module, "main")
            self.assertEqual(by_id["vcruntime140/stubs/pg6_import_thunks.cpp"].module, "vcruntime140")
            self.assertEqual(by_id["program_modules.cpp"].module, "program")
            self.assertEqual(by_id["exports/pg6_export_thunks.cpp"].module, "program")
            self.assertEqual(by_id["main/lift/shard-00001.ll"].instructions, 20)

    def test_program_build_compiles_tagged_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _fake_emsdk(repo)
            src = _program_layout(root)
            runner = RecordingRunner()
            report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache",
                out_dir=root / "dist",
                jobs=2,
                run_command=runner,
                which_fn=lambda _name: None,
            )
            self.assertTrue(report.ok, report.errors)
            self.assertEqual(len(report.sources), 12)
            modules = {u.unit.module for u in report.sources}
            self.assertEqual(modules, {"main", "vcruntime140", "program"})
            self.assertEqual(len(runner.link_calls), 1)

    def test_program_manifest_file_argument(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = _program_layout(Path(tmp))
            units = bgw.discover_sources(src / "program-manifest.json")
            self.assertTrue(any(u.unit_id == "main/lift/shard-00000.ll" for u in units))
            self.assertTrue(any(u.unit_id == "program_modules.cpp" for u in units))

    def test_windows_backslash_paths_in_program_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "out"
            _write(src / "main" / "lift" / "shard-00000.ll", "; shard\n")
            _write(src / "main" / "lift" / "functions.ll", "; fn\n")
            _write(src / "main" / "stubs" / "pg6_import_thunks.cpp", "void t() {}\n")
            _write(src / "exports" / "pg6_export_thunks.cpp", "void e() {}\n")
            (src / "main" / "build-manifest.json").write_text(
                json.dumps(
                    {
                        "lift": {
                            "shards": [{"ll": r"lift\shard-00000.ll", "instructions": 4}],
                            "registration": {"ll": r"lift\functions.ll"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            (src / "program-manifest.json").write_text(
                json.dumps(
                    {
                        "modules": [{"module": "main"}],
                        "files": [{"path": r"exports\pg6_export_thunks.cpp"}],
                    }
                ),
                encoding="utf-8",
            )
            ids = [u.unit_id for u in bgw.discover_sources(src)]
            self.assertIn("main/lift/shard-00000.ll", ids)
            self.assertIn("exports/pg6_export_thunks.cpp", ids)
            for unit_id in ids:
                self.assertNotIn("\\", unit_id)


class TestResponseFileLinking(unittest.TestCase):
    def test_build_link_argv_uses_at_response_file(self) -> None:
        rsp = Path("work") / "wasm_obj_cache" / "link_objects.rsp"
        link = bgw.build_link_argv(
            "em++",
            [Path("a.o"), Path("b.o")],
            output_js=Path("web/dist/runtime.js"),
            runtime_libs=[Path("libplaygta6_core.a")],
            response_file=rsp,
        )
        self.assertTrue(any(t.startswith("@") and t.endswith("link_objects.rsp") for t in link))
        self.assertNotIn("a.o", [Path(t).name for t in link if not t.startswith("@")])
        self.assertIn("libplaygta6_core.a", [Path(t).name for t in link])

    def test_write_and_sanitize_windows_rsp_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rsp = Path(tmp) / "link_objects.rsp"
            win_objects = [
                r"C:\work\cache\aa\shard-00000.o",
                r"C:\Users\someone\work\wasm_obj_cache\bb\shard-00001.o",
                r"C:\Program Files\playgta6\cache\cc\shard 00002.o",
            ]
            bgw.write_link_response_file(rsp, win_objects)
            text = rsp.read_text(encoding="utf-8")
            self.assertEqual(
                text.splitlines(),
                [
                    "C:/work/cache/aa/shard-00000.o",
                    "C:/Users/someone/work/wasm_obj_cache/bb/shard-00001.o",
                    '"C:/Program Files/playgta6/cache/cc/shard 00002.o"',
                ],
            )
            self.assertNotIn("\\", text)
            cleaned = bgw.sanitize_text(text)
            self.assertNotIn("someone", cleaned)
            self.assertNotIn("C:\\", cleaned)
            self.assertNotIn("C:/", cleaned)
            argv = [
                r"C:\Users\someone\emsdk\upstream\emscripten\em++.bat",
                "-sMEMORY64=1",
                rf"@{rsp}".replace(str(rsp), r"C:\Users\someone\work\wasm_obj_cache\link_objects.rsp"),
                "-o",
                r"C:\Users\someone\web\dist\runtime.js",
            ]
            sanitized = bgw.sanitize_argv(argv)
            self.assertEqual(sanitized[0], "em++")
            self.assertEqual(sanitized[2], "@link_objects.rsp")
            self.assertEqual(sanitized[-1], "runtime.js")
            self.assertNotIn("someone", " ".join(sanitized))

            link_argv = bgw.build_link_argv(
                "em++",
                [Path("a.o")],
                output_js=Path("web/dist/runtime.js"),
                response_file=Path(r"C:\work\cache\link_objects.rsp"),
            )
            at_tokens = [t for t in link_argv if t.startswith("@")]
            self.assertEqual(len(at_tokens), 1)
            self.assertNotIn("\\", at_tokens[0])
            self.assertEqual(bgw.sanitize_argv(link_argv)[link_argv.index(at_tokens[0])], "@link_objects.rsp")

    def test_auto_response_file_when_over_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _fake_emsdk(repo)
            _touch_lib(repo / "runtime" / "build-wasm64" / "libplaygta6_core.a")
            src = root / "recomp-out"
            for index in range(bgw.RESPONSE_FILE_THRESHOLD + 1):
                _write(src / "lift" / f"shard-{index:05d}.ll", f"; s{index}\n")
            cache = root / "cache"
            runner = RecordingRunner()
            report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist",
                jobs=2,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner,
                which_fn=lambda _name: None,
            )
            self.assertTrue(report.ok, report.errors)
            self.assertEqual(report.response_file, "link_objects.rsp")
            rsp = cache / "link_objects.rsp"
            self.assertTrue(rsp.is_file())
            self.assertEqual(len(rsp.read_text(encoding="utf-8").splitlines()), bgw.RESPONSE_FILE_THRESHOLD + 1)
            self.assertEqual(len(runner.link_calls), 1)
            link = runner.link_calls[0]
            self.assertTrue(any(t.replace("\\", "/").endswith("/link_objects.rsp") and t.startswith("@") for t in link))
            self.assertIn("@link_objects.rsp", report.link_argv)
            dumped = json.dumps(report.to_dict())
            self.assertNotIn("\\\\", bgw.sanitize_text(dumped))

    def test_explicit_response_file_below_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _fake_emsdk(repo)
            src = _layout_sources(root)
            cache = root / "cache"
            runner = RecordingRunner()
            report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist",
                jobs=1,
                runtime_lib=None,
                run_command=runner,
                which_fn=lambda _name: None,
                response_file=True,
            )
            self.assertTrue(report.ok, report.errors)
            self.assertEqual(report.response_file, "link_objects.rsp")
            self.assertTrue((cache / "link_objects.rsp").is_file())
            self.assertTrue(any(t.startswith("@") for t in runner.link_calls[0]))

    def test_no_response_file_keeps_objects_on_argv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _fake_emsdk(repo)
            src = root / "recomp-out"
            for index in range(bgw.RESPONSE_FILE_THRESHOLD + 1):
                _write(src / "lift" / f"shard-{index:05d}.ll", f"; s{index}\n")
            runner = RecordingRunner()
            report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache",
                out_dir=root / "dist",
                jobs=2,
                run_command=runner,
                which_fn=lambda _name: None,
                response_file=False,
            )
            self.assertTrue(report.ok, report.errors)
            self.assertIsNone(report.response_file)
            self.assertFalse(any(t.startswith("@") for t in runner.link_calls[0]))


class TestSampleAndGuestBoot(unittest.TestCase):
    def test_sample_evenly_spaces_shards_and_skips_link(self) -> None:
        units = [
            bgw.SourceUnit(f"lift/shard-{i:05d}.ll", Path(f"shard-{i:05d}.ll"), "ll")
            for i in range(10)
        ]
        units.append(bgw.SourceUnit("lift/functions.ll", Path("functions.ll"), "ll"))
        picked = bgw.sample_units(units, 3)
        self.assertEqual(
            [u.unit_id for u in picked],
            ["lift/shard-00000.ll", "lift/shard-00003.ll", "lift/shard-00006.ll"],
        )

    def test_sample_build_extrapolates_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _fake_emsdk(repo)
            src = root / "recomp-out"
            for index in range(10):
                _write(src / "lift" / f"shard-{index:05d}.ll", f"; s{index}\n")
            _write(src / "lift" / "functions.ll", "; registration\n")
            runner = RecordingRunner(delay=0.01)
            report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache",
                out_dir=root / "dist",
                jobs=2,
                run_command=runner,
                which_fn=lambda _name: None,
                sample=3,
            )
            self.assertTrue(report.ok, report.errors)
            self.assertEqual(report.mode, "sample")
            self.assertEqual(report.sample, 3)
            self.assertEqual(len(report.sources), 3)
            self.assertEqual(len(runner.compile_calls), 3)
            self.assertEqual(runner.link_calls, [])
            self.assertIsNotNone(report.extrapolated)
            extra = report.extrapolated or {}
            self.assertEqual(extra["sample_units"], 3)
            self.assertEqual(extra["total_shards"], 10)
            self.assertGreater(extra["object_bytes"], 0)
            self.assertGreaterEqual(extra["compile_seconds"], 0)
            self.assertEqual(extra["object_bytes"], int(3 * 3 * (10 / 3)))

            code, out, err = _run_cli(
                [
                    str(src),
                    "--sample",
                    "3",
                    "--dry-run",
                    "--json",
                    "--jobs",
                    "2",
                    "--repo-root",
                    str(repo),
                    "--cache-dir",
                    str(root / "cache2"),
                    "--out-dir",
                    str(root / "dist2"),
                ]
            )
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertEqual(payload["mode"], "sample")
            self.assertIn("extrapolated", payload)
            self.assertIn("object_bytes", payload["extrapolated"])
            self.assertIn("compile_seconds", payload["extrapolated"])
            user = os.environ.get("USERNAME") or os.environ.get("USER") or "nobody"
            dumped = json.dumps(payload)
            self.assertNotIn("\\\\", dumped)
            if len(user) >= 3:
                self.assertNotRegex(dumped, rf"(?<![A-Za-z0-9_]){user}(?![A-Za-z0-9_])")

    def test_link_guest_boot_compiles_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _fake_emsdk(repo)
            _touch_lib(repo / "runtime" / "build-wasm64" / "libplaygta6_core.a")
            main_cpp = repo / "runtime" / "main.cpp"
            wasm_boot_cpp = repo / "runtime" / "wasm_boot.cpp"
            _write(main_cpp, "int main() { return 0; }\n")
            _write(wasm_boot_cpp, "void run_wasm_guest_boot() {}\n")
            src = _layout_sources(root)
            runner = RecordingRunner()
            report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache",
                out_dir=root / "dist",
                jobs=2,
                run_command=runner,
                which_fn=lambda _name: None,
                link_guest_boot=True,
            )
            self.assertTrue(report.ok, report.errors)
            self.assertTrue(report.linked_guest_boot)
            by_id = {u.unit.unit_id: u for u in report.sources}
            self.assertIn("runtime/main.cpp", by_id)
            self.assertIn("runtime/wasm_boot.cpp", by_id)
            self.assertEqual(by_id["runtime/main.cpp"].unit.module, "boot")
            self.assertEqual(by_id["runtime/wasm_boot.cpp"].unit.module, "boot")

            seen_boot: set[str] = set()
            for argv in runner.compile_calls:
                src_token = _source_token(argv)
                assert src_token is not None
                name = Path(src_token).name
                if name in {"main.cpp", "wasm_boot.cpp"}:
                    self.assertIn(bgw.BOOT_LINKED_GUEST_DEFINE, argv)
                    self.assertNotIn(bgw.BOOT_PROGRAM_DEFINE, argv)
                    seen_boot.add(name)
                else:
                    self.assertNotIn(bgw.BOOT_LINKED_GUEST_DEFINE, argv)
                    self.assertNotIn(bgw.BOOT_PROGRAM_DEFINE, argv)
            self.assertEqual(seen_boot, {"main.cpp", "wasm_boot.cpp"})

            # Boot object files must precede libplaygta6_core.a on the link line so
            # linked wasm_boot.o overrides the archive's non-linked stub.
            self.assertEqual(len(runner.link_calls), 1)
            link_names = [Path(t).name for t in runner.link_calls[0]]
            lib_idx = link_names.index("libplaygta6_core.a")
            wasm_boot_obj_name = by_id["runtime/wasm_boot.cpp"].object_path.name
            main_obj_name = by_id["runtime/main.cpp"].object_path.name
            self.assertLess(link_names.index(wasm_boot_obj_name), lib_idx)
            self.assertLess(link_names.index(main_obj_name), lib_idx)

            payload = json.loads(bgw.report_to_json(report))
            self.assertTrue(payload["linked_guest_boot"])

    def test_link_guest_boot_program_build_passes_boot_program_define(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _fake_emsdk(repo)
            _touch_lib(repo / "runtime" / "build-wasm64" / "libplaygta6_core.a")
            _write(repo / "runtime" / "main.cpp", "int main() { return 0; }\n")
            _write(repo / "runtime" / "wasm_boot.cpp", "void run_wasm_guest_boot() {}\n")
            prog_src = _program_layout(root)
            cache = root / "cache"
            runner = RecordingRunner()
            report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=prog_src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist",
                jobs=2,
                run_command=runner,
                which_fn=lambda _name: None,
                link_guest_boot=True,
            )
            self.assertTrue(report.ok, report.errors)
            by_id = {u.unit.unit_id: u for u in report.sources}
            self.assertIn("runtime/main.cpp", by_id)
            self.assertIn("runtime/wasm_boot.cpp", by_id)

            seen_boot: set[str] = set()
            for argv in runner.compile_calls:
                src_token = _source_token(argv)
                assert src_token is not None
                name = Path(src_token).name
                if name in {"main.cpp", "wasm_boot.cpp"}:
                    self.assertIn(bgw.BOOT_LINKED_GUEST_DEFINE, argv)
                    self.assertIn(bgw.BOOT_PROGRAM_DEFINE, argv)
                    seen_boot.add(name)
                else:
                    self.assertNotIn(bgw.BOOT_LINKED_GUEST_DEFINE, argv)
                    self.assertNotIn(bgw.BOOT_PROGRAM_DEFINE, argv)
            self.assertEqual(seen_boot, {"main.cpp", "wasm_boot.cpp"})

            # Single-module build sharing the same cache must not reuse program boot objects.
            single_src = _layout_sources(root)
            runner_single = RecordingRunner()
            single_report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=single_src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist-single",
                jobs=2,
                run_command=runner_single,
                which_fn=lambda _name: None,
                link_guest_boot=True,
            )
            self.assertTrue(single_report.ok, single_report.errors)
            single_by_id = {u.unit.unit_id: u for u in single_report.sources}
            self.assertNotEqual(
                by_id["runtime/wasm_boot.cpp"].cache_key,
                single_by_id["runtime/wasm_boot.cpp"].cache_key,
            )
            self.assertFalse(single_by_id["runtime/wasm_boot.cpp"].cache_hit)
            for argv in runner_single.compile_calls:
                src_token = _source_token(argv)
                assert src_token is not None
                if Path(src_token).name in {"main.cpp", "wasm_boot.cpp"}:
                    self.assertIn(bgw.BOOT_LINKED_GUEST_DEFINE, argv)
                    self.assertNotIn(bgw.BOOT_PROGRAM_DEFINE, argv)

            # Dry-run JSON remains sanitized.
            dry = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=prog_src,
                repo_root=repo,
                cache_dir=root / "cache-dry",
                out_dir=root / "dist-dry",
                jobs=2,
                dry_run=True,
                which_fn=lambda _name: None,
                link_guest_boot=True,
            )
            self.assertTrue(dry.ok, dry.errors)
            dumped = bgw.report_to_json(dry)
            self.assertNotIn("\\\\", dumped)
            self.assertIn(bgw.BOOT_PROGRAM_DEFINE, dumped)

    def test_link_guest_boot_pg6_program_cpp_without_manifest_sets_boot_program(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _fake_emsdk(repo)
            _write(repo / "runtime" / "main.cpp", "int main() { return 0; }\n")
            _write(repo / "runtime" / "wasm_boot.cpp", "void run_wasm_guest_boot() {}\n")
            src = _layout_sources(root)
            _write(src / "pg6_program.cpp", "int pg6_program_marker = 1;\n")
            runner = RecordingRunner()
            report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache",
                out_dir=root / "dist",
                jobs=1,
                run_command=runner,
                which_fn=lambda _name: None,
                link_guest_boot=True,
            )
            self.assertTrue(report.ok, report.errors)
            boot_calls = [
                argv
                for argv in runner.compile_calls
                if Path(_source_token(argv) or "").name in {"main.cpp", "wasm_boot.cpp"}
            ]
            self.assertEqual(len(boot_calls), 2)
            for argv in boot_calls:
                self.assertIn(bgw.BOOT_LINKED_GUEST_DEFINE, argv)
                self.assertIn(bgw.BOOT_PROGRAM_DEFINE, argv)

    def test_link_guest_boot_custom_entry_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _fake_emsdk(repo)
            entry = repo / "runtime" / "wasm_boot.cpp"
            _write(entry, "int boot() { return 0; }\n")
            src = _layout_sources(root)
            code, out, err = _run_cli(
                [
                    str(src),
                    "--dry-run",
                    "--json",
                    "--link-guest-boot",
                    "--guest-boot-entry",
                    str(entry),
                    "--repo-root",
                    str(repo),
                    "--cache-dir",
                    str(root / "cache"),
                    "--out-dir",
                    str(root / "dist"),
                ]
            )
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertTrue(payload["linked_guest_boot"])
            wasm_boot_units = [u for u in payload["sources"] if u["id"].endswith("wasm_boot.cpp")]
            self.assertEqual(len(wasm_boot_units), 1)


class TestWasm64IrRetargetAndFlags(unittest.TestCase):
    SAMPLE_SHARD_IR = (
        "; ModuleID = 'shard-00000'\n"
        'target datalayout = "e-m:e-p:64:64-p10:8:8-p20:8:8-i64:64-i128:128-n32:64-S128-ni:1:10:20"\n'
        'target triple = "wasm64-unknown-unknown"\n'
        "\n"
        "; target datalayout = \"comment-stays\"\n"
        "define void @sub_140001000(ptr %state) #0 {\n"
        "  ret void\n"
        "}\n"
        '\nattributes #0 = { "target-features"="+simd128" }\n'
    )

    def test_retarget_wasm64_ir_strips_datalayout_and_rewrites_triple(self) -> None:
        retargeted = bgw.retarget_wasm64_ir(self.SAMPLE_SHARD_IR)
        self.assertNotIn('target datalayout = "e-m:e', retargeted)
        self.assertIn('; target datalayout = "comment-stays"\n', retargeted)
        self.assertNotIn('target triple = "wasm64-unknown-unknown"', retargeted)
        self.assertIn('target triple = "wasm64-unknown-emscripten"\n', retargeted)
        self.assertIn("define void @sub_140001000(ptr %state) #0 {\n", retargeted)
        self.assertIn(
            'attributes #0 = { "target-features"="+simd128,+atomics,+bulk-memory,+mutable-globals" }\n',
            retargeted,
        )

        # CRLF line endings are preserved on surviving lines.
        crlf_in = self.SAMPLE_SHARD_IR.replace("\n", "\r\n")
        crlf_out = bgw.retarget_wasm64_ir(crlf_in)
        self.assertNotIn('target datalayout = "e-m:e', crlf_out)
        self.assertIn('target triple = "wasm64-unknown-emscripten"\r\n', crlf_out)

        # Already-emscripten and unrelated triples pass through unchanged.
        other = 'target triple = "x86_64-pc-windows-msvc"\n'
        self.assertEqual(bgw.retarget_wasm64_ir(other), other)
        em_ir = 'target triple = "wasm64-unknown-emscripten"\n'
        self.assertEqual(bgw.retarget_wasm64_ir(em_ir), em_ir)

    def test_ll_compile_flags_include_simd128_and_omit_include_dirs(self) -> None:
        includes = [Path("runtime"), Path("work/recomp_program")]
        ll_flags = bgw.compile_flags_for(opt="O2", kind="ll", include_dirs=includes)
        self.assertIn("-msimd128", ll_flags)
        self.assertIn("-x", ll_flags)
        self.assertIn("ir", ll_flags)
        self.assertIn("-c", ll_flags)
        self.assertNotIn("-I", ll_flags)
        self.assertEqual(ll_flags.count("-msimd128"), 1)

        cpp_flags = bgw.compile_flags_for(opt="O2", kind="cpp", include_dirs=includes)
        self.assertIn("-I", cpp_flags)
        self.assertIn("-std=c++20", cpp_flags)
        self.assertNotIn("-msimd128", cpp_flags)

        c_flags = bgw.compile_flags_for(opt="O2", kind="c", include_dirs=includes)
        self.assertIn("-I", c_flags)
        self.assertNotIn("-msimd128", c_flags)

        self.assertEqual(bgw.COMPILE_WASM_FLAGS, ("-sMEMORY64=1", "-pthread"))

    def test_shared_memory_features_passed_and_no_check_features_gone(self) -> None:
        for kind in ("ll", "cpp", "c"):
            flags = bgw.compile_flags_for(opt="O2", kind=kind)
            for flag in ("-matomics", "-mbulk-memory", "-mmutable-globals"):
                self.assertIn(flag, flags)
        # Features already present are not duplicated.
        ir = 'attributes #0 = { "target-features"="+atomics,+simd128" }\n'
        out = bgw.retarget_wasm64_ir(ir)
        self.assertEqual(out.count("+atomics"), 1)
        self.assertIn("+bulk-memory", out)
        for argv in (bgw.SIDE_MODULE_LINK_FLAGS, bgw.SIDE_MODULE_PRUNED_LINK_FLAGS):
            self.assertNotIn("-Wl,--no-check-features", argv)

    def test_cache_key_salts_ll_with_retarget_version(self) -> None:
        ll_flags = bgw.compile_flags_for(opt="O2", kind="ll")
        cpp_flags = bgw.compile_flags_for(opt="O2", kind="cpp")
        raw = self.SAMPLE_SHARD_IR.encode("utf-8")

        key_v1 = bgw.cache_key_for(raw, ll_flags, kind="ll")
        key_v2 = bgw.cache_key_for(raw, ll_flags, kind="ll", retarget_version="v2")
        self.assertNotEqual(key_v1, key_v2)

        old_version = bgw.WASM64_IR_RETARGET_VERSION
        try:
            bgw.WASM64_IR_RETARGET_VERSION = 999
            key_patched = bgw.cache_key_for(raw, ll_flags, kind="ll")
            self.assertNotEqual(key_v1, key_patched)
            cpp_before = bgw.cache_key_for(b"int x = 1;\n", cpp_flags, kind="cpp")
        finally:
            bgw.WASM64_IR_RETARGET_VERSION = old_version
        cpp_after = bgw.cache_key_for(b"int x = 1;\n", cpp_flags, kind="cpp")
        self.assertEqual(cpp_before, cpp_after)

    def test_run_build_retargets_ll_into_cache_dir_without_modifying_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _fake_emsdk(repo)
            _touch_lib(repo / "runtime" / "build-wasm64" / "libplaygta6_core.a")
            (repo / "runtime").mkdir(exist_ok=True)
            src = root / "recomp-out"
            shard_path = _write(src / "lift" / "shard-00000.ll", self.SAMPLE_SHARD_IR)
            _write(src / "stubs" / "pg6_import_thunks.cpp", "void pg6_thunk() {}\n")
            cache_dir = root / "cache"
            out_dir = root / "dist"

            # Dry-run must not materialize any retargeted IR file in cache_dir.
            dry_report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache_dir,
                out_dir=out_dir,
                jobs=1,
                dry_run=True,
                which_fn=lambda _name: None,
            )
            self.assertTrue(dry_report.ok, dry_report.errors)
            self.assertFalse(cache_dir.exists() and list(cache_dir.rglob("*.retargeted.ll")))

            seen_ll_sources: list[tuple[Path, str, tuple[str, ...]]] = []
            seen_cpp_argvs: list[tuple[str, ...]] = []
            base_runner = RecordingRunner()

            def inspecting_runner(argv: Sequence[str]) -> bgw.CommandResult:
                tokens = tuple(str(a) for a in argv)
                src_token = _source_token(tokens)
                if src_token is not None:
                    src_p = Path(src_token)
                    if src_p.suffix.lower() == ".ll":
                        seen_ll_sources.append((src_p, src_p.read_text(encoding="utf-8"), tokens))
                    elif src_p.suffix.lower() == ".cpp":
                        seen_cpp_argvs.append(tokens)
                return base_runner(tokens)

            report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache_dir,
                out_dir=out_dir,
                jobs=1,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=inspecting_runner,
                which_fn=lambda _name: None,
            )
            self.assertTrue(report.ok, report.errors)
            self.assertEqual(len(seen_ll_sources), 1)
            ll_path, ll_text, ll_argv = seen_ll_sources[0]
            self.assertTrue(ll_path.name.endswith(".retargeted.ll"))
            self.assertTrue(ll_path.resolve().is_relative_to(cache_dir.resolve()))
            self.assertNotIn('target datalayout = "e-m:e', ll_text)
            self.assertIn('target triple = "wasm64-unknown-emscripten"', ll_text)
            self.assertIn("define void @sub_140001000", ll_text)
            self.assertIn("-msimd128", ll_argv)
            self.assertNotIn("-I", ll_argv)

            self.assertEqual(len(seen_cpp_argvs), 1)
            self.assertIn("-I", seen_cpp_argvs[0])

            # Original shard on disk is untouched and temp .retargeted.ll was cleaned up.
            self.assertEqual(shard_path.read_text(encoding="utf-8"), self.SAMPLE_SHARD_IR)
            self.assertEqual(list(cache_dir.rglob("*.retargeted.ll")), [])

            # JSON output remains sanitized (no host paths or raw backslashes).
            dumped = bgw.report_to_json(report)
            self.assertNotIn("\\\\", dumped)
            user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
            if len(user) >= 3:
                self.assertNotRegex(dumped, rf"(?<![A-Za-z0-9_]){user}(?![A-Za-z0-9_])")


class TestStripPthreadOptAndV8SizeLimit(unittest.TestCase):
    def _repo(self, tmp: Path) -> Path:
        repo = tmp / "repo"
        _fake_emsdk(repo)
        _touch_lib(repo / "runtime" / "build-wasm64" / "libplaygta6_core.a")
        (repo / "runtime").mkdir(exist_ok=True)
        return repo

    def test_strip_appends_strip_all_and_assertions_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _layout_sources(root)
            runner = RecordingRunner()
            report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache",
                out_dir=root / "dist",
                jobs=2,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                strip=True,
                run_command=runner,
                which_fn=lambda _name: None,
            )
            self.assertTrue(report.ok, report.errors)
            self.assertEqual(len(runner.link_calls), 1)
            link = runner.link_calls[0]
            self.assertIn("-Wl,--strip-all", link)
            self.assertIn("-sASSERTIONS=0", link)
            self.assertIn("-Wl,--strip-all", report.link_flags)
            self.assertIn("-sASSERTIONS=0", report.link_flags)
            self.assertIn("-Wl,--strip-all", report.link_argv)
            self.assertIn("-sASSERTIONS=0", report.link_argv)
            # Default LINK_FLAGS constant remains untouched for CMake contract.
            self.assertNotIn("-Wl,--strip-all", bgw.LINK_FLAGS)
            self.assertNotIn("-sASSERTIONS=0", bgw.LINK_FLAGS)

            # CLI --strip works in dry-run JSON output.
            code, out, err = _run_cli(
                [
                    str(src),
                    "--dry-run",
                    "--json",
                    "--strip",
                    "--repo-root",
                    str(repo),
                    "--cache-dir",
                    str(root / "cache2"),
                    "--out-dir",
                    str(root / "dist2"),
                ]
            )
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertIn("-Wl,--strip-all", payload["link_flags"])
            self.assertIn("-sASSERTIONS=0", payload["link_flags"])
            self.assertIn("-Wl,--strip-all", payload["link_argv"])
            self.assertIn("-sASSERTIONS=0", payload["link_argv"])

    def test_extra_link_flag_cli_and_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _layout_sources(root)
            runner = RecordingRunner()
            report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache",
                out_dir=root / "dist",
                jobs=1,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                extra_link_flags=("-Wl,--gc-sections", "-sINITIAL_MEMORY=2GB"),
                run_command=runner,
                which_fn=lambda _name: None,
            )
            self.assertTrue(report.ok, report.errors)
            link = runner.link_calls[0]
            self.assertIn("-Wl,--gc-sections", link)
            self.assertIn("-sINITIAL_MEMORY=2GB", link)
            self.assertIn("-Wl,--gc-sections", report.link_flags)
            self.assertIn("-sINITIAL_MEMORY=2GB", report.link_flags)

            # Repeatable CLI --extra-link-flag accepts both space-separated and `=` forms.
            code, out, err = _run_cli(
                [
                    str(src),
                    "--dry-run",
                    "--json",
                    "--extra-link-flag",
                    "-Wl,--gc-sections",
                    "--extra-link-flag=-sINITIAL_MEMORY=2GB",
                    "--repo-root",
                    str(repo),
                    "--cache-dir",
                    str(root / "cache2"),
                    "--out-dir",
                    str(root / "dist2"),
                ]
            )
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertIn("-Wl,--gc-sections", payload["link_flags"])
            self.assertIn("-sINITIAL_MEMORY=2GB", payload["link_flags"])
            self.assertIn("-Wl,--gc-sections", payload["link_argv"])
            self.assertIn("-sINITIAL_MEMORY=2GB", payload["link_argv"])

    def test_pthread_pool_size_override_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _layout_sources(root)
            runner = RecordingRunner()
            report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache",
                out_dir=root / "dist",
                jobs=1,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                pthread_pool_size=8,
                run_command=runner,
                which_fn=lambda _name: None,
            )
            self.assertTrue(report.ok, report.errors)
            link = runner.link_calls[0]
            self.assertIn("-sPTHREAD_POOL_SIZE=8", link)
            self.assertNotIn("-sPTHREAD_POOL_SIZE=4", link)
            self.assertIn("-sPTHREAD_POOL_SIZE=8", report.link_flags)
            self.assertNotIn("-sPTHREAD_POOL_SIZE=4", report.link_flags)

            # CLI --pthread-pool-size overrides default pool size.
            code, out, err = _run_cli(
                [
                    str(src),
                    "--dry-run",
                    "--json",
                    "--pthread-pool-size",
                    "12",
                    "--repo-root",
                    str(repo),
                    "--cache-dir",
                    str(root / "cache2"),
                    "--out-dir",
                    str(root / "dist2"),
                ]
            )
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertIn("-sPTHREAD_POOL_SIZE=12", payload["link_flags"])
            self.assertNotIn("-sPTHREAD_POOL_SIZE=4", payload["link_flags"])
            self.assertIn("-sPTHREAD_POOL_SIZE=12", payload["link_argv"])

            # Invalid pool sizes (< 1) raise BuildGuestWasmError / CLI exit code 2.
            with self.assertRaises(bgw.BuildGuestWasmError):
                bgw.resolve_link_flags(pthread_pool_size=0)
            with self.assertRaises(bgw.BuildGuestWasmError):
                bgw.resolve_link_flags(pthread_pool_size=-2)
            bad_code, bad_out, _bad_err = _run_cli(
                [
                    str(src),
                    "--dry-run",
                    "--json",
                    "--pthread-pool-size",
                    "0",
                    "--repo-root",
                    str(repo),
                ]
            )
            self.assertEqual(bad_code, 2)
            bad_payload = json.loads(bad_out)
            self.assertFalse(bad_payload["ok"])
            self.assertTrue(any("pthread-pool-size" in e for e in bad_payload["errors"]))

    def test_opt_os_and_oz_compile_and_link_flags(self) -> None:
        self.assertEqual(bgw.parse_opt_level("Os"), "Os")
        self.assertEqual(bgw.parse_opt_level("-Os"), "Os")
        self.assertEqual(bgw.parse_opt_level("s"), "Os")
        self.assertEqual(bgw.parse_opt_level("Oz"), "Oz")
        self.assertEqual(bgw.parse_opt_level("-Oz"), "Oz")
        self.assertEqual(bgw.parse_opt_level("z"), "Oz")
        with self.assertRaises(bgw.BuildGuestWasmError):
            bgw.parse_opt_level("O4")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _layout_sources(root)
            for opt_level in ("Os", "Oz"):
                runner = RecordingRunner()
                report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                    source_dir=src,
                    repo_root=repo,
                    cache_dir=root / f"cache-{opt_level}",
                    out_dir=root / f"dist-{opt_level}",
                    jobs=2,
                    opt=opt_level,
                    runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                    run_command=runner,
                    which_fn=lambda _name: None,
                )
                self.assertTrue(report.ok, report.errors)
                self.assertEqual(report.opt, opt_level)
                expected_flag = f"-{opt_level}"
                self.assertIn(expected_flag, report.compile_flags)
                self.assertIn(expected_flag, report.link_flags)
                self.assertTrue(all(expected_flag in call for call in runner.compile_calls))
                self.assertIn(expected_flag, runner.link_calls[0])

            code, out, err = _run_cli(
                [
                    str(src),
                    "--dry-run",
                    "--json",
                    "--opt",
                    "Oz",
                    "--strip",
                    "--repo-root",
                    str(repo),
                    "--cache-dir",
                    str(root / "cache-cli"),
                    "--out-dir",
                    str(root / "dist-cli"),
                ]
            )
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertEqual(payload["opt"], "Oz")
            self.assertIn("-Oz", payload["compile_flags"])
            self.assertIn("-Oz", payload["link_flags"])
            self.assertIn("-Oz", payload["link_argv"])
            self.assertIn("-Wl,--strip-all", payload["link_argv"])

    def test_v8_1gib_size_check_and_warning(self) -> None:
        self.assertEqual(bgw.V8_MAX_WASM_MODULE_BYTES, 1024 * 1024 * 1024)
        self.assertEqual(bgw.V8_MAX_WASM_MODULE_BYTES, 1073741824)

        exceeds_exact, warn_exact = bgw.check_v8_wasm_size(bgw.V8_MAX_WASM_MODULE_BYTES)
        self.assertFalse(exceeds_exact)
        self.assertIsNone(warn_exact)

        exceeds_over1, warn_over1 = bgw.check_v8_wasm_size(bgw.V8_MAX_WASM_MODULE_BYTES + 1)
        self.assertTrue(exceeds_over1)
        self.assertIsNotNone(warn_over1)
        assert warn_over1 is not None
        self.assertIn(str(bgw.V8_MAX_WASM_MODULE_BYTES + 1), warn_over1)
        self.assertIn("1073741824", warn_over1)
        self.assertIn("--strip", warn_over1)
        self.assertIn("--opt Oz", warn_over1)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _layout_sources(root)

            # Under-limit real build records exact wasm_bytes and no V8 warning.
            small_runner = RecordingRunner()
            small_report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache-small",
                out_dir=root / "dist-small",
                jobs=1,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=small_runner,
                which_fn=lambda _name: None,
            )
            self.assertTrue(small_report.ok, small_report.errors)
            self.assertEqual(small_report.wasm_bytes, 8)
            self.assertFalse(small_report.exceeds_v8_1gib_limit)
            self.assertEqual(small_report.warnings, [])
            small_json = json.loads(bgw.report_to_json(small_report))
            self.assertEqual(small_json["wasm_bytes"], 8)
            self.assertFalse(small_json["exceeds_v8_1gib_limit"])
            small_text = bgw.format_report_text(small_report)
            self.assertIn("Wasm size: 8 bytes", small_text)
            self.assertNotIn("EXCEEDS V8 1 GiB LIMIT", small_text)

            # Over-limit build (live observed 1,092,331,701 bytes) emits clear warning.
            oversized_bytes = 1_092_331_701
            orig_object_bytes = bgw._object_bytes

            def fake_object_bytes(path: Path | None) -> int:
                if path is not None and path.name == bgw.RUNTIME_WASM:
                    return oversized_bytes
                return orig_object_bytes(path)

            big_runner = RecordingRunner()
            bgw._object_bytes = fake_object_bytes
            try:
                big_report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                    source_dir=src,
                    repo_root=repo,
                    cache_dir=root / "cache-big",
                    out_dir=root / "dist-big",
                    jobs=1,
                    runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                    run_command=big_runner,
                    which_fn=lambda _name: None,
                )
            finally:
                bgw._object_bytes = orig_object_bytes

            self.assertTrue(big_report.ok, big_report.errors)
            self.assertEqual(big_report.wasm_bytes, oversized_bytes)
            self.assertTrue(big_report.exceeds_v8_1gib_limit)
            self.assertEqual(len(big_report.warnings), 1)
            warning = big_report.warnings[0]
            self.assertIn(str(oversized_bytes), warning)
            self.assertIn("1073741824", warning)
            self.assertIn("--strip", warning)
            self.assertIn("--opt Oz", warning)

            big_json = json.loads(bgw.report_to_json(big_report))
            self.assertEqual(big_json["wasm_bytes"], oversized_bytes)
            self.assertTrue(big_json["exceeds_v8_1gib_limit"])
            self.assertIn(warning, big_json["warnings"])

            big_text = bgw.format_report_text(big_report)
            self.assertIn(f"Wasm size: {oversized_bytes} bytes", big_text)
            self.assertIn("[EXCEEDS V8 1 GiB LIMIT]", big_text)
            self.assertIn("--opt Oz", big_text)


class TestNoLinkOptAndShrinkLargestShards(unittest.TestCase):
    def _repo(self, tmp: Path) -> Path:
        repo = tmp / "repo"
        _fake_emsdk(repo)
        _touch_lib(repo / "runtime" / "build-wasm64" / "libplaygta6_core.a")
        (repo / "runtime").mkdir(exist_ok=True)
        return repo

    def test_no_link_opt_omits_link_opt_while_compiling_shards_at_opt(self) -> None:
        # Direct helper checks for Oz, Os, and O2.
        for opt_level in ("Oz", "Os", "O2"):
            flags = bgw.resolve_link_flags(opt=opt_level, strip=True, no_link_opt=True)
            self.assertFalse(
                any(f.startswith("-O") for f in flags),
                f"unexpected -O* flag in {flags}",
            )
            self.assertIn("-Wl,--strip-all", flags)
            self.assertIn("-sASSERTIONS=0", flags)

        # Backwards compatibility: no_link_opt=False (default) still appends -Oz/-Os.
        self.assertIn("-Oz", bgw.resolve_link_flags(opt="Oz"))
        self.assertIn("-Os", bgw.resolve_link_flags(opt="Os"))

        # build_link_argv forwards no_link_opt to resolve_link_flags.
        link_argv = bgw.build_link_argv(
            "em++",
            [Path("a.o")],
            output_js=Path("web/dist/runtime.js"),
            opt="Oz",
            strip=True,
            no_link_opt=True,
        )
        self.assertFalse(any(t.startswith("-O") for t in link_argv), link_argv)
        self.assertIn("-Wl,--strip-all", link_argv)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _layout_sources(root)
            runner = RecordingRunner()
            report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache",
                out_dir=root / "dist",
                jobs=2,
                opt="Oz",
                strip=True,
                no_link_opt=True,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner,
                which_fn=lambda _name: None,
            )
            self.assertTrue(report.ok, report.errors)
            self.assertTrue(report.no_link_opt)
            self.assertEqual(report.shrink_largest_shards, 0)
            self.assertIn("-Oz", report.compile_flags)
            self.assertTrue(all("-Oz" in call for call in runner.compile_calls))
            self.assertEqual(len(runner.link_calls), 1)
            link = runner.link_calls[0]
            self.assertFalse(any(t.startswith("-O") for t in link), link)
            self.assertFalse(any(t.startswith("-O") for t in report.link_flags), report.link_flags)
            self.assertFalse(any(t.startswith("-O") for t in report.link_argv), report.link_argv)
            self.assertIn("-Wl,--strip-all", link)
            self.assertIn("-sASSERTIONS=0", link)

            payload = json.loads(bgw.report_to_json(report))
            self.assertTrue(payload["no_link_opt"])
            self.assertEqual(payload["shrink_largest_shards"], 0)
            self.assertIn("-Oz", payload["compile_flags"])
            self.assertNotIn("-Oz", payload["link_flags"])
            self.assertNotIn("-Oz", payload["link_argv"])

            text = bgw.format_report_text(report)
            self.assertIn("No link opt: yes", text)

            # CLI --opt Oz --no-link-opt --strip in dry-run JSON mode.
            code, out, err = _run_cli(
                [
                    str(src),
                    "--dry-run",
                    "--json",
                    "--opt",
                    "Oz",
                    "--no-link-opt",
                    "--strip",
                    "--repo-root",
                    str(repo),
                    "--cache-dir",
                    str(root / "cache-cli"),
                    "--out-dir",
                    str(root / "dist-cli"),
                ]
            )
            self.assertEqual(code, 0, err)
            cli_payload = json.loads(out)
            self.assertTrue(cli_payload["ok"])
            self.assertTrue(cli_payload["no_link_opt"])
            self.assertEqual(cli_payload["shrink_largest_shards"], 0)
            self.assertIn("-Oz", cli_payload["compile_flags"])
            self.assertFalse(any(t.startswith("-O") for t in cli_payload["link_flags"]))
            self.assertFalse(any(t.startswith("-O") for t in cli_payload["link_argv"]))
            self.assertIn("-Wl,--strip-all", cli_payload["link_argv"])
            for unit in cli_payload["sources"]:
                self.assertIn("-Oz", unit["argv"])

    def test_shrink_largest_shards_by_file_size_tiebreak_and_warm_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = root / "recomp-out"
            # Distinct sizes with a tie on shard-00000 and shard-00002 (both 200 bytes).
            # functions.ll and .cpp are larger on disk but must NOT be chosen over shard-*.ll.
            _write(src / "lift" / "shard-00000.ll", ";" + "a" * 198 + "\n")
            _write(src / "lift" / "shard-00001.ll", ";" + "b" * 498 + "\n")
            _write(src / "lift" / "shard-00002.ll", ";" + "c" * 198 + "\n")
            _write(src / "lift" / "shard-00003.ll", ";" + "d" * 48 + "\n")
            _write(src / "lift" / "functions.ll", ";" + "f" * 998 + "\n")
            _write(src / "stubs" / "pg6_import_thunks.cpp", "//" + "t" * 997 + "\n")

            units = bgw.discover_sources(src)
            picked = bgw.select_shrink_shards(units, 2)
            # Top-2: shard-00001.ll (500 B), then shard-00000.ll (200 B, tie-broken before shard-00002.ll).
            self.assertEqual(picked, frozenset({"lift/shard-00001.ll", "lift/shard-00000.ll"}))

            cache = root / "cache"

            # Step 1: Warm the cache at -O1 (shrink_largest_shards=0).
            runner_o1 = RecordingRunner()
            first = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist-o1",
                jobs=2,
                opt="O1",
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner_o1,
                which_fn=lambda _name: None,
            )
            self.assertTrue(first.ok, first.errors)
            self.assertEqual(first.cache_misses, 6)
            self.assertEqual(first.cache_hits, 0)
            self.assertTrue(all("-O1" in call for call in runner_o1.compile_calls))
            o1_keys = {u.unit.unit_id: u.cache_key for u in first.sources}

            # Step 2: Re-run with --opt O1 --shrink-largest-shards 2 using the same cache.
            # Only the top-2 largest shards should recompile (at -Oz with new cache keys);
            # the other 4 units reuse their cached -O1 objects!
            runner_shrink = RecordingRunner()
            second = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist-shrink",
                jobs=2,
                opt="O1",
                shrink_largest_shards=2,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner_shrink,
                which_fn=lambda _name: None,
            )
            self.assertTrue(second.ok, second.errors)
            self.assertEqual(second.shrink_largest_shards, 2)
            self.assertEqual(second.cache_hits, 4)
            self.assertEqual(second.cache_misses, 2)
            self.assertEqual(len(runner_shrink.compile_calls), 2)

            by_id = {u.unit.unit_id: u for u in second.sources}
            for unit_id in ("lift/shard-00000.ll", "lift/shard-00001.ll"):
                item = by_id[unit_id]
                self.assertEqual(item.status, "compiled")
                self.assertIn("-Oz", item.argv)
                self.assertNotIn("-O1", item.argv)
                self.assertNotEqual(item.cache_key, o1_keys[unit_id])
            for unit_id in (
                "lift/shard-00002.ll",
                "lift/shard-00003.ll",
                "lift/functions.ll",
                "stubs/pg6_import_thunks.cpp",
            ):
                item = by_id[unit_id]
                self.assertEqual(item.status, "hit")
                self.assertIn("-O1", item.argv)
                self.assertNotIn("-Oz", item.argv)
                self.assertEqual(item.cache_key, o1_keys[unit_id])

            for call in runner_shrink.compile_calls:
                self.assertIn("-Oz", call)
                self.assertNotIn("-O1", call)
            self.assertFalse(any(t.startswith("-O") for t in runner_shrink.link_calls[0]))

            payload = json.loads(bgw.report_to_json(second))
            self.assertEqual(payload["shrink_largest_shards"], 2)
            self.assertFalse(payload["no_link_opt"])
            text = bgw.format_report_text(second)
            self.assertIn("Shrink largest shards: 2", text)

            # Step 3: Repeating --opt O1 --shrink-largest-shards 2 hits the cache for all 6 units.
            runner_warm = RecordingRunner()
            third = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist-warm",
                jobs=2,
                opt="O1",
                shrink_largest_shards=2,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner_warm,
                which_fn=lambda _name: None,
            )
            self.assertTrue(third.ok, third.errors)
            self.assertEqual(third.cache_hits, 6)
            self.assertEqual(third.cache_misses, 0)
            self.assertEqual(runner_warm.compile_calls, [])

    def test_shrink_largest_shards_manifest_instructions_oz_noop_and_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            prog_src = _program_layout(root)

            # In _program_layout:
            #   main/lift/shard-00002.ll: 30 instructions
            #   main/lift/shard-00001.ll: 20 instructions
            #   vcruntime140/lift/shard-00001.ll: 20 instructions
            #   main/lift/shard-00000.ll: 10 instructions
            #   vcruntime140/lift/shard-00000.ll: 10 instructions
            # Tie on 20 instructions breaks deterministically by unit_id ("main/..." < "vcruntime140/...").
            runner = RecordingRunner()
            report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=prog_src,
                repo_root=repo,
                cache_dir=root / "cache-prog",
                out_dir=root / "dist-prog",
                jobs=2,
                opt="O1",
                shrink_largest_shards=2,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner,
                which_fn=lambda _name: None,
            )
            self.assertTrue(report.ok, report.errors)
            expected_oz = {"main/lift/shard-00002.ll", "main/lift/shard-00001.ll"}
            for item in report.sources:
                if item.unit.unit_id in expected_oz:
                    self.assertIn("-Oz", item.argv, item.unit.unit_id)
                    self.assertNotIn("-O1", item.argv, item.unit.unit_id)
                else:
                    self.assertIn("-O1", item.argv, item.unit.unit_id)
                    self.assertNotIn("-Oz", item.argv, item.unit.unit_id)

            # When opt is already Oz, shrink_largest_shards is a clean no-op (single -Oz flag).
            runner_oz = RecordingRunner()
            report_oz = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=prog_src,
                repo_root=repo,
                cache_dir=root / "cache-oz",
                out_dir=root / "dist-oz",
                jobs=2,
                opt="Oz",
                no_link_opt=True,
                shrink_largest_shards=2,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner_oz,
                which_fn=lambda _name: None,
            )
            self.assertTrue(report_oz.ok, report_oz.errors)
            self.assertEqual(report_oz.shrink_largest_shards, 2)
            for item in report_oz.sources:
                self.assertEqual(item.argv.count("-Oz"), 1, item.argv)

            # Fallback for non-`shard-*` .ll layouts: picks largest .ll file and skips .cpp.
            plain_src = root / "plain-out"
            _write(plain_src / "shards" / "small.ll", "; s\n")
            _write(plain_src / "shards" / "big.ll", ";" + "x" * 300 + "\n")
            _write(plain_src / "thunks.cpp", "//" + "y" * 600 + "\n")
            plain_units = bgw.discover_sources(plain_src)
            self.assertEqual(bgw.select_shrink_shards(plain_units, 1), frozenset({"shards/big.ll"}))

    def test_shrink_largest_shards_cli_and_negative_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = root / "recomp-out"
            _write(src / "lift" / "shard-00000.ll", "; tiny\n")
            _write(src / "lift" / "shard-00001.ll", ";" + "z" * 400 + "\n")
            _write(src / "stubs" / "pg6_import_thunks.cpp", "void t() {}\n")

            with self.assertRaises(bgw.BuildGuestWasmError):
                bgw.run_build(
                link_cache=False,
                link_jobs=1,
                    source_dir=src,
                    repo_root=repo,
                    shrink_largest_shards=-1,
                    dry_run=True,
                    which_fn=lambda _name: None,
                )
            with self.assertRaises(bgw.BuildGuestWasmError):
                bgw.run_build(
                link_cache=False,
                link_jobs=1,
                    source_dir=src,
                    repo_root=repo,
                    shrink_largest_shards=True,  # type: ignore[arg-type]
                    dry_run=True,
                    which_fn=lambda _name: None,
                )

            bad_code, bad_out, _bad_err = _run_cli(
                [
                    str(src),
                    "--dry-run",
                    "--json",
                    "--shrink-largest-shards",
                    "-3",
                    "--repo-root",
                    str(repo),
                ]
            )
            self.assertEqual(bad_code, 2)
            bad_payload = json.loads(bad_out)
            self.assertFalse(bad_payload["ok"])
            self.assertTrue(any("shrink-largest-shards" in e for e in bad_payload["errors"]))

            code, out, err = _run_cli(
                [
                    str(src),
                    "--dry-run",
                    "--json",
                    "--opt",
                    "O1",
                    "--no-link-opt",
                    "--shrink-largest-shards",
                    "1",
                    "--repo-root",
                    str(repo),
                    "--cache-dir",
                    str(root / "cache-cli"),
                    "--out-dir",
                    str(root / "dist-cli"),
                ]
            )
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["no_link_opt"])
            self.assertEqual(payload["shrink_largest_shards"], 1)
            by_id = {u["id"]: u for u in payload["sources"]}
            self.assertIn("-Oz", by_id["lift/shard-00001.ll"]["argv"])
            self.assertNotIn("-O1", by_id["lift/shard-00001.ll"]["argv"])
            self.assertIn("-O1", by_id["lift/shard-00000.ll"]["argv"])
            self.assertNotIn("-Oz", by_id["lift/shard-00000.ll"]["argv"])
            self.assertIn("-O1", by_id["stubs/pg6_import_thunks.cpp"]["argv"])


class TestBrowserBootFlags(unittest.TestCase):
    EXPECTED_INCOMING_API = (
        "-sINCOMING_MODULE_JS_API=locateFile,instantiateWasm,mainScriptUrlOrBlob,"
        "print,printErr,preRun,onRuntimeInitialized,noInitialRun,setStatus,onExit"
    )
    EXPECTED_RUNTIME_METHODS = (
        "-sEXPORTED_RUNTIME_METHODS=wasmMemory,FS,HEAPU8,addRunDependency,removeRunDependency"
    )
    EXPECTED_EXPORTED_FUNCTIONS = (
        "-sEXPORTED_FUNCTIONS=_main,_playgta6_get_input_state_ptr,_playgta6_audio_ring"
    )

    def _repo(self, tmp: Path) -> Path:
        repo = tmp / "repo"
        _fake_emsdk(repo)
        _touch_lib(repo / "runtime" / "build-wasm64" / "libplaygta6_core.a")
        _write(repo / "runtime" / "main.cpp", "int main() { return 0; }\n")
        _write(repo / "runtime" / "wasm_boot.cpp", "void run_wasm_guest_boot() {}\n")
        return repo

    def test_wasm_exceptions_applies_to_cpp_and_link_not_ll_or_c_and_updates_cache_keys(self) -> None:
        # compile_flags_for only adds -fwasm-exceptions to C++ units.
        cpp_flags = bgw.compile_flags_for(opt="O2", kind="cpp", wasm_exceptions=True)
        ll_flags = bgw.compile_flags_for(opt="O2", kind="ll", wasm_exceptions=True)
        c_flags = bgw.compile_flags_for(opt="O2", kind="c", wasm_exceptions=True)
        self.assertIn("-fwasm-exceptions", cpp_flags)
        self.assertNotIn("-fwasm-exceptions", ll_flags)
        self.assertNotIn("-fwasm-exceptions", c_flags)
        self.assertNotIn("-fwasm-exceptions", bgw.compile_flags_for(opt="O2", kind="cpp"))

        # resolve_link_flags & build_link_argv include -fwasm-exceptions when enabled.
        self.assertIn("-fwasm-exceptions", bgw.resolve_link_flags(wasm_exceptions=True))
        self.assertNotIn("-fwasm-exceptions", bgw.resolve_link_flags())
        self.assertIn(
            "-fwasm-exceptions",
            bgw.build_link_argv(
                "em++",
                [Path("a.o")],
                output_js=Path("web/dist/runtime.js"),
                wasm_exceptions=True,
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _layout_sources(root)
            _write(src / "helper.c", "int helper(void) { return 1; }\n")
            cache = root / "cache"

            # Warm cache without wasm_exceptions.
            runner1 = RecordingRunner()
            first = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist1",
                jobs=2,
                link_guest_boot=True,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner1,
                which_fn=lambda _name: None,
            )
            self.assertTrue(first.ok, first.errors)
            self.assertFalse(first.wasm_exceptions)
            self.assertNotIn("-fwasm-exceptions", runner1.link_calls[0])
            first_keys = {u.unit.unit_id: u.cache_key for u in first.sources}

            # Rebuild with wasm_exceptions=True sharing the same cache:
            # 2 .ll shards + 1 .c unit must be cache hits; 5 .cpp units (3 layout + 2 boot) recompile.
            runner2 = RecordingRunner()
            second = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist2",
                jobs=2,
                link_guest_boot=True,
                wasm_exceptions=True,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner2,
                which_fn=lambda _name: None,
            )
            self.assertTrue(second.ok, second.errors)
            self.assertTrue(second.wasm_exceptions)
            self.assertEqual(second.cache_hits, 3)
            self.assertEqual(second.cache_misses, 5)
            self.assertEqual(len(runner2.compile_calls), 5)

            for item in second.sources:
                if item.unit.kind == "cpp":
                    self.assertEqual(item.status, "compiled")
                    self.assertIn("-fwasm-exceptions", item.argv)
                    self.assertNotEqual(item.cache_key, first_keys[item.unit.unit_id])
                else:
                    self.assertEqual(item.status, "hit")
                    self.assertNotIn("-fwasm-exceptions", item.argv)
                    self.assertEqual(item.cache_key, first_keys[item.unit.unit_id])

            for call in runner2.compile_calls:
                src_token = _source_token(call)
                assert src_token is not None
                self.assertEqual(Path(src_token).suffix.lower(), ".cpp")
                self.assertIn("-fwasm-exceptions", call)

            self.assertIn("-fwasm-exceptions", runner2.link_calls[0])
            self.assertIn("-fwasm-exceptions", second.link_flags)
            self.assertIn("-fwasm-exceptions", second.link_argv)
            self.assertNotIn("-fwasm-exceptions", bgw.LINK_FLAGS)

            payload = json.loads(bgw.report_to_json(second))
            self.assertTrue(payload["wasm_exceptions"])
            text = bgw.format_report_text(second)
            self.assertIn("Wasm exceptions: yes", text)

    def test_initial_memory_and_stack_size_flags(self) -> None:
        flags = bgw.resolve_link_flags(initial_memory=" 64MB ", stack_size="8MB")
        self.assertIn("-sINITIAL_MEMORY=64MB", flags)
        self.assertIn("-sSTACK_SIZE=8MB", flags)
        self.assertIn("-sDEFAULT_PTHREAD_STACK_SIZE=8MB", flags)

        # Numeric byte counts are also valid alphanumeric size tokens.
        byte_flags = bgw.resolve_link_flags(initial_memory="67108864", stack_size="8388608")
        self.assertIn("-sINITIAL_MEMORY=67108864", byte_flags)
        self.assertIn("-sSTACK_SIZE=8388608", byte_flags)
        self.assertIn("-sDEFAULT_PTHREAD_STACK_SIZE=8388608", byte_flags)

        link_argv = bgw.build_link_argv(
            "em++",
            [Path("a.o")],
            output_js=Path("web/dist/runtime.js"),
            initial_memory="64MB",
            stack_size="8MB",
        )
        self.assertIn("-sINITIAL_MEMORY=64MB", link_argv)
        self.assertIn("-sSTACK_SIZE=8MB", link_argv)
        self.assertIn("-sDEFAULT_PTHREAD_STACK_SIZE=8MB", link_argv)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _layout_sources(root)
            runner = RecordingRunner()
            report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache",
                out_dir=root / "dist",
                jobs=2,
                initial_memory="64MB",
                stack_size="8MB",
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner,
                which_fn=lambda _name: None,
            )
            self.assertTrue(report.ok, report.errors)
            self.assertEqual(report.initial_memory, "64MB")
            self.assertEqual(report.stack_size, "8MB")
            link = runner.link_calls[0]
            self.assertIn("-sINITIAL_MEMORY=64MB", link)
            self.assertIn("-sSTACK_SIZE=8MB", link)
            self.assertIn("-sDEFAULT_PTHREAD_STACK_SIZE=8MB", link)

            payload = json.loads(bgw.report_to_json(report))
            self.assertEqual(payload["initial_memory"], "64MB")
            self.assertEqual(payload["stack_size"], "8MB")
            text = bgw.format_report_text(report)
            self.assertIn("Initial memory: 64MB", text)
            self.assertIn("Stack size: 8MB", text)

    def test_assertions_flag_and_strip_precedence(self) -> None:
        # Standalone assertions=0 and assertions=1.
        flags0 = bgw.resolve_link_flags(assertions=0)
        self.assertIn("-sASSERTIONS=0", flags0)
        self.assertNotIn("-sASSERTIONS=1", flags0)

        flags1 = bgw.resolve_link_flags(assertions=1)
        self.assertIn("-sASSERTIONS=1", flags1)
        self.assertNotIn("-sASSERTIONS=0", flags1)

        # Explicit assertions=1 overrides --strip's -sASSERTIONS=0 while keeping --strip-all.
        strip_and_assert1 = bgw.resolve_link_flags(strip=True, assertions=1)
        self.assertIn("-Wl,--strip-all", strip_and_assert1)
        self.assertIn("-sASSERTIONS=1", strip_and_assert1)
        self.assertNotIn("-sASSERTIONS=0", strip_and_assert1)

        # Explicit assertions=0 with --strip does not duplicate -sASSERTIONS=0.
        strip_and_assert0 = bgw.resolve_link_flags(strip=True, assertions=0)
        self.assertIn("-Wl,--strip-all", strip_and_assert0)
        self.assertEqual(strip_and_assert0.count("-sASSERTIONS=0"), 1)

        # Default strip=True, assertions=None retains backwards-compatible -sASSERTIONS=0.
        strip_only = bgw.resolve_link_flags(strip=True)
        self.assertIn("-Wl,--strip-all", strip_only)
        self.assertIn("-sASSERTIONS=0", strip_only)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _layout_sources(root)
            runner = RecordingRunner()
            report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache",
                out_dir=root / "dist",
                jobs=1,
                strip=True,
                assertions=1,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner,
                which_fn=lambda _name: None,
            )
            self.assertTrue(report.ok, report.errors)
            self.assertEqual(report.assertions, 1)
            link = runner.link_calls[0]
            self.assertIn("-Wl,--strip-all", link)
            self.assertIn("-sASSERTIONS=1", link)
            self.assertNotIn("-sASSERTIONS=0", link)
            payload = json.loads(bgw.report_to_json(report))
            self.assertEqual(payload["assertions"], 1)
            self.assertIn("Assertions: 1", bgw.format_report_text(report))

    def test_browser_boot_exports_and_full_cli_combination(self) -> None:
        default_flags = bgw.resolve_link_flags()
        self.assertNotIn(self.EXPECTED_INCOMING_API, default_flags)
        self.assertNotIn(self.EXPECTED_RUNTIME_METHODS, default_flags)
        self.assertNotIn(self.EXPECTED_EXPORTED_FUNCTIONS, default_flags)

        boot_flags = bgw.resolve_link_flags(browser_boot_exports=True)
        self.assertIn(self.EXPECTED_INCOMING_API, boot_flags)
        self.assertIn(self.EXPECTED_RUNTIME_METHODS, boot_flags)
        self.assertIn(self.EXPECTED_EXPORTED_FUNCTIONS, boot_flags)

        split_boot_flags = bgw.resolve_link_flags(browser_boot_exports=True, main_module=True)
        self.assertIn(self.EXPECTED_RUNTIME_METHODS, split_boot_flags)
        self.assertIn(self.EXPECTED_EXPORTED_FUNCTIONS, split_boot_flags)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _layout_sources(root)
            code, out, err = _run_cli(
                [
                    str(src),
                    "--dry-run",
                    "--json",
                    "--link-guest-boot",
                    "--wasm-exceptions",
                    "--initial-memory",
                    "64MB",
                    "--stack-size",
                    "8MB",
                    "--strip",
                    "--assertions",
                    "1",
                    "--browser-boot-exports",
                    "--repo-root",
                    str(repo),
                    "--cache-dir",
                    str(root / "cache"),
                    "--out-dir",
                    str(root / "dist"),
                ]
            )
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["wasm_exceptions"])
            self.assertEqual(payload["initial_memory"], "64MB")
            self.assertEqual(payload["stack_size"], "8MB")
            self.assertEqual(payload["assertions"], 1)
            self.assertTrue(payload["browser_boot_exports"])

            for expected in (
                "-fwasm-exceptions",
                "-sINITIAL_MEMORY=64MB",
                "-sSTACK_SIZE=8MB",
                "-sDEFAULT_PTHREAD_STACK_SIZE=8MB",
                "-Wl,--strip-all",
                "-sASSERTIONS=1",
                self.EXPECTED_INCOMING_API,
                self.EXPECTED_RUNTIME_METHODS,
                self.EXPECTED_EXPORTED_FUNCTIONS,
            ):
                self.assertIn(expected, payload["link_flags"])
                self.assertIn(expected, payload["link_argv"])
            self.assertNotIn("-sASSERTIONS=0", payload["link_flags"])
            self.assertNotIn("-sASSERTIONS=0", payload["link_argv"])

            for unit in payload["sources"]:
                if unit["kind"] == "cpp":
                    self.assertIn("-fwasm-exceptions", unit["argv"])
                else:
                    self.assertNotIn("-fwasm-exceptions", unit["argv"])

            # Sanitized JSON output contains no raw backslashes or host username.
            dumped = json.dumps(payload)
            self.assertNotIn("\\\\", dumped)
            user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
            if len(user) >= 3:
                self.assertNotRegex(dumped, rf"(?<![A-Za-z0-9_]){user}(?![A-Za-z0-9_])")

            # Verify text summary includes browser boot exports line.
            code_txt, out_txt, err_txt = _run_cli(
                [
                    str(src),
                    "--dry-run",
                    "--browser-boot-exports",
                    "--repo-root",
                    str(repo),
                    "--cache-dir",
                    str(root / "cache2"),
                    "--out-dir",
                    str(root / "dist2"),
                ]
            )
            self.assertEqual(code_txt, 0, err_txt)
            self.assertIn("Browser boot exports: yes", out_txt)

    def test_browser_boot_exports_includes_audio_ring_symbol(self) -> None:
        exported = bgw.BROWSER_BOOT_EXPORTED_FUNCTIONS.split(",")
        self.assertEqual(
            exported,
            ["_main", "_playgta6_get_input_state_ptr", "_playgta6_audio_ring"],
        )

        # Disabled by default: neither -sEXPORTED_FUNCTIONS nor _playgta6_audio_ring is emitted.
        default_flags = bgw.resolve_link_flags()
        self.assertFalse(any("_playgta6_audio_ring" in f for f in default_flags))
        default_split_flags = bgw.resolve_link_flags(main_module=True)
        self.assertFalse(any("_playgta6_audio_ring" in f for f in default_split_flags))

        # Enabled: both monolithic and split-module builds export _playgta6_audio_ring alongside
        # _main and _playgta6_get_input_state_ptr.
        for main_module in (False, True):
            flags = bgw.resolve_link_flags(browser_boot_exports=True, main_module=main_module)
            exported_flags = [f for f in flags if f.startswith("-sEXPORTED_FUNCTIONS=")]
            self.assertEqual(exported_flags, [self.EXPECTED_EXPORTED_FUNCTIONS])
            symbols = exported_flags[0].split("=", 1)[1].split(",")
            self.assertIn("_main", symbols)
            self.assertIn("_playgta6_get_input_state_ptr", symbols)
            self.assertIn("_playgta6_audio_ring", symbols)

            link_argv = bgw.build_link_argv(
                "em++",
                [Path("a.o")],
                output_js=Path("web/dist/runtime.js"),
                browser_boot_exports=True,
                main_module=main_module,
            )
            self.assertIn(self.EXPECTED_EXPORTED_FUNCTIONS, link_argv)

    def test_invalid_browser_boot_flag_values_rejected(self) -> None:
        bad_sizes = ("", "   ", "0", "0MB", "-64MB", "64 MB", "abc", "64MB;rm", "8.5MB", "64MB/x")
        for bad in bad_sizes:
            with self.assertRaises(bgw.BuildGuestWasmError, msg=f"initial_memory={bad!r}"):
                bgw.resolve_link_flags(initial_memory=bad)
            with self.assertRaises(bgw.BuildGuestWasmError, msg=f"stack_size={bad!r}"):
                bgw.resolve_link_flags(stack_size=bad)

        with self.assertRaises(bgw.BuildGuestWasmError):
            bgw.resolve_link_flags(initial_memory=64)  # type: ignore[arg-type]
        with self.assertRaises(bgw.BuildGuestWasmError):
            bgw.resolve_link_flags(stack_size=8)  # type: ignore[arg-type]

        for bad_assert in (-1, 2, 99, True, False, "1"):
            with self.assertRaises(bgw.BuildGuestWasmError, msg=f"assertions={bad_assert!r}"):
                bgw.resolve_link_flags(assertions=bad_assert)  # type: ignore[arg-type]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _layout_sources(root)

            for cli_flag, bad_val, err_needle in (
                ("--initial-memory", "", "initial-memory"),
                ("--initial-memory", "64 MB", "initial-memory"),
                ("--initial-memory", "0MB", "initial-memory"),
                ("--stack-size", "abc", "stack-size"),
                ("--stack-size", "8MB;bad", "stack-size"),
                ("--assertions", "2", "assertions"),
                ("--assertions", "-1", "assertions"),
            ):
                code, out, _err = _run_cli(
                    [
                        str(src),
                        "--dry-run",
                        "--json",
                        cli_flag,
                        bad_val,
                        "--repo-root",
                        str(repo),
                    ]
                )
                self.assertEqual(code, 2, f"{cli_flag} {bad_val!r} -> {out}")
                payload = json.loads(out)
                self.assertFalse(payload["ok"])
                self.assertTrue(
                    any(err_needle in e for e in payload["errors"]),
                    f"expected {err_needle!r} in {payload['errors']}",
                )


class TestSplitModules(unittest.TestCase):
    def _repo(self, tmp: Path) -> Path:
        repo = tmp / "repo"
        _fake_emsdk(repo)
        _touch_lib(repo / "runtime" / "build-wasm64" / "libplaygta6_core.a")
        _write(repo / "runtime" / "main.cpp", "int main() { return 0; }\n")
        _write(repo / "runtime" / "wasm_boot.cpp", "void run_wasm_guest_boot() {}\n")
        return repo

    def test_split_modules_default_one_is_backwards_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _program_layout(root)
            runner = RecordingRunner()
            report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache",
                out_dir=root / "dist",
                jobs=2,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner,
                which_fn=lambda _name: None,
            )
            self.assertTrue(report.ok, report.errors)
            self.assertEqual(report.split_modules, 1)
            self.assertEqual(report.side_modules, [])
            self.assertEqual(report.outputs, (bgw.RUNTIME_JS, bgw.RUNTIME_WASM))
            self.assertNotIn("-fPIC", report.compile_flags)
            self.assertFalse(any(t.startswith("-sMAIN_MODULE=") for t in report.link_flags))
            self.assertIn("-sPTHREAD_POOL_SIZE=4", report.link_flags)
            self.assertFalse(any(t.startswith("-sINITIAL_MEMORY=") for t in report.link_flags))
            self.assertFalse(any(t.startswith("-sINCOMING_MODULE_JS_API=") for t in report.link_flags))
            self.assertEqual(runner.side_link_calls, [])
            self.assertEqual(runner.ar_calls, [])
            self.assertEqual(len(runner.link_calls), 1)
            main_call = runner.link_calls[0]
            self.assertFalse(any(t.startswith("-sMAIN_MODULE=") for t in main_call))
            self.assertNotIn("-Wl,--whole-archive", main_call)
            self.assertNotIn("-Wl,--no-whole-archive", main_call)
            self.assertIn("libplaygta6_core.a", [Path(t).name for t in main_call])
            by_id = {u.unit.unit_id: u for u in report.sources}
            for reg_id in ("main/lift/functions.ll", "vcruntime140/lift/functions.ll"):
                obj = by_id[reg_id].object_path
                assert obj is not None
                self.assertIn(obj.name, [Path(t).name for t in main_call])
            for call in runner.compile_calls:
                self.assertNotIn("-fPIC", call)
            self.assertNotIn("-fPIC", bgw.COMPILE_WASM_FLAGS)
            self.assertFalse(any(t.startswith("-sMAIN_MODULE=") for t in bgw.LINK_FLAGS))

            payload = json.loads(bgw.report_to_json(report))
            self.assertEqual(payload["split_modules"], 1)
            self.assertEqual(payload["side_modules"], [])

    def test_link_cache_skips_unchanged_part_and_runtime_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _program_layout(root)
            _write(src / "helper.c", "int helper(void) { return 7; }\n")
            common = dict(
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache",
                out_dir=root / "dist",
                jobs=2,
                link_guest_boot=True,
                split_modules=3,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                which_fn=lambda _name: None,
            )
            first_runner = RecordingRunner()
            first = bgw.run_build(run_command=first_runner, **common)
            self.assertTrue(first.ok, first.errors)
            self.assertGreater(len(first_runner.side_link_calls), 0)
            self.assertEqual(len(first_runner.link_calls), 1)
            parts = sorted((root / "dist").glob("guest_part_*.wasm"))
            before = {p.name: p.read_bytes() for p in parts}

            second_runner = RecordingRunner()
            second = bgw.run_build(run_command=second_runner, **common)
            self.assertTrue(second.ok, second.errors)
            self.assertEqual(second_runner.side_link_calls, [])
            self.assertEqual(second_runner.link_calls, [])
            self.assertTrue(all(m.link_cache_hit for m in second.side_modules))
            self.assertEqual(
                before, {p.name: p.read_bytes() for p in sorted((root / "dist").glob("guest_part_*.wasm"))}
            )

            third_runner = RecordingRunner()
            third = bgw.run_build(run_command=third_runner, link_cache=False, **common)
            self.assertTrue(third.ok, third.errors)
            self.assertEqual(len(third_runner.side_link_calls), len(first_runner.side_link_calls))
            self.assertEqual(len(third_runner.link_calls), 1)

    def test_split_modules_adds_fpic_and_separate_cache_keys(self) -> None:
        for kind in ("ll", "c", "cpp"):
            self.assertNotIn("-fPIC", bgw.compile_flags_for(opt="O2", kind=kind))
            pic_flags = bgw.compile_flags_for(opt="O2", kind=kind, pic=True)
            self.assertIn("-fPIC", pic_flags)
            self.assertEqual(pic_flags.count("-fPIC"), 1)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _program_layout(root)
            _write(src / "helper.c", "int helper(void) { return 7; }\n")
            cache = root / "cache"

            # Warm non-PIC cache with split_modules=1.
            runner1 = RecordingRunner()
            first = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist-single",
                jobs=2,
                link_guest_boot=True,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner1,
                which_fn=lambda _name: None,
            )
            self.assertTrue(first.ok, first.errors)
            first_keys = {u.unit.unit_id: u.cache_key for u in first.sources}

            # Rebuild with split_modules=3 sharing cache: every unit gets a new -fPIC cache key.
            runner2 = RecordingRunner()
            second = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist-split",
                jobs=2,
                link_guest_boot=True,
                split_modules=3,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner2,
                which_fn=lambda _name: None,
            )
            self.assertTrue(second.ok, second.errors)
            self.assertEqual(second.cache_hits, 0)
            self.assertEqual(second.cache_misses, len(second.sources))
            self.assertEqual(len(runner2.compile_calls), len(second.sources))
            for item in second.sources:
                self.assertIn("-fPIC", item.argv)
                self.assertNotEqual(item.cache_key, first_keys[item.unit.unit_id])
            for call in runner2.compile_calls:
                self.assertIn("-fPIC", call)

            # Repeating split_modules=3 hits cache for every unit.
            runner3 = RecordingRunner()
            third = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist-split2",
                jobs=2,
                link_guest_boot=True,
                split_modules=3,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner3,
                which_fn=lambda _name: None,
            )
            self.assertTrue(third.ok, third.errors)
            self.assertEqual(third.cache_hits, len(third.sources))
            self.assertEqual(third.cache_misses, 0)
            self.assertEqual(runner3.compile_calls, [])

    def test_split_modules_three_partitions_links_side_and_main_modules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _program_layout(root)
            cache = root / "cache"
            out_dir = root / "dist"
            orig_lib = repo / "runtime" / "build-wasm64" / "libplaygta6_core.a"
            orig_lib_bytes = orig_lib.read_bytes()
            runner = RecordingRunner()
            report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=out_dir,
                jobs=2,
                strip=True,
                wasm_exceptions=True,
                link_guest_boot=True,
                split_modules=3,
                runtime_lib=orig_lib,
                run_command=runner,
                which_fn=lambda _name: None,
            )
            self.assertTrue(report.ok, report.errors)
            self.assertEqual(report.split_modules, 3)
            self.assertIn("-fPIC", report.compile_flags)
            self.assertIn("-sMAIN_MODULE=1", report.link_flags)
            self.assertNotIn("-sMAIN_MODULE=2", report.link_flags)
            self.assertIn("-sPTHREAD_POOL_SIZE=48", report.link_flags)
            self.assertNotIn("-sPTHREAD_POOL_SIZE=4", report.link_flags)
            self.assertIn("-sALLOW_MEMORY_GROWTH=1", report.link_flags)
            self.assertIn("-sMAXIMUM_MEMORY=16GB", report.link_flags)
            self.assertIn("-sINITIAL_MEMORY=64MB", report.link_flags)
            self.assertEqual(report.initial_memory, "64MB")
            self.assertEqual(
                report.outputs,
                (
                    "runtime.js",
                    "runtime.wasm",
                    "guest_part_0.wasm",
                    "guest_part_1.wasm",
                    "guest_part_2.wasm",
                    "guest_part_3.wasm",
                ),
            )
            # 3 shard buckets (0..2) + 1 registration side module (3).
            self.assertEqual(len(report.side_modules), 4)
            self.assertEqual(len(runner.side_link_calls), 4)
            self.assertEqual(len(runner.ar_calls), 1)
            self.assertEqual(len(runner.link_calls), 1)

            # Verify every shard appears in exactly one shard side module (0..2),
            # and registration functions.ll units appear together in guest_part_3.wasm.
            expected_shards = {
                "main/lift/shard-00000.ll",
                "main/lift/shard-00001.ll",
                "main/lift/shard-00002.ll",
                "vcruntime140/lift/shard-00000.ll",
                "vcruntime140/lift/shard-00001.ll",
            }
            expected_regs = (
                "main/lift/functions.ll",
                "vcruntime140/lift/functions.ll",
            )
            seen_shards: list[str] = []
            by_id = {u.unit.unit_id: u for u in report.sources}
            for idx, (side, side_call) in enumerate(zip(report.side_modules, runner.side_link_calls)):
                self.assertEqual(side.index, idx)
                self.assertEqual(side.name, f"guest_part_{idx}.wasm")
                self.assertEqual(side.response_file, f"side_part_{idx}.rsp")
                self.assertEqual(side.wasm_bytes, 8)
                self.assertFalse(side.exceeds_v8_1gib_limit)
                self.assertTrue((out_dir / side.name).is_file())
                if idx < 3:
                    seen_shards.extend(side.shards)
                else:
                    self.assertEqual(side.shards, expected_regs)

                self.assertIn("-sMEMORY64=1", side_call)
                self.assertIn("-pthread", side_call)
                self.assertIn("-sSIDE_MODULE=1", side_call)
                self.assertIn("-fwasm-exceptions", side_call)
                self.assertIn("-Wl,--strip-all", side_call)
                self.assertNotIn("-sASSERTIONS=0", side_call)
                self.assertNotIn("-sMAIN_MODULE=1", side_call)
                self.assertNotIn("-sMAIN_MODULE=2", side_call)
                self.assertEqual(side_call[-2], "-o")
                self.assertEqual(Path(side_call[-1]).name, f"guest_part_{idx}.wasm")

                # Side response file uses POSIX forward slashes and lists exact bucket objects.
                at_tokens = [t for t in side_call if t.startswith("@")]
                self.assertEqual(len(at_tokens), 1)
                self.assertNotIn("\\", at_tokens[0])
                self.assertTrue(at_tokens[0].endswith(f"/side_part_{idx}.rsp"))
                rsp_file = cache / f"side_part_{idx}.rsp"
                self.assertTrue(rsp_file.is_file())
                rsp_text = rsp_file.read_text(encoding="utf-8")
                self.assertNotIn("\\", rsp_text)
                expected_objs = [
                    str(by_id[shard_id].object_path).replace("\\", "/") for shard_id in side.shards
                ]
                self.assertEqual(rsp_text.splitlines(), expected_objs)

                # Sanitized side link argv has @side_part_{idx}.rsp and guest_part_{idx}.wasm.
                self.assertIn(f"@side_part_{idx}.rsp", side.link_argv)
                self.assertIn(f"guest_part_{idx}.wasm", side.link_argv)
                self.assertIn("-sSIDE_MODULE=1", side.link_argv)

            self.assertEqual(len(seen_shards), len(expected_shards))
            self.assertEqual(set(seen_shards), expected_shards)
            # Balanced 5 shards across 3 buckets -> sizes (2, 2, 1).
            self.assertEqual(sorted(len(s.shards) for s in report.side_modules[:3]), [1, 2, 2])

            # Verify emar d ran on a cache copy of libplaygta6_core.a (never the original).
            filtered_lib = cache / "libplaygta6_core_split.a"
            self.assertTrue(filtered_lib.is_file())
            self.assertEqual(orig_lib.read_bytes(), orig_lib_bytes)
            ar_call = runner.ar_calls[0]
            self.assertTrue(Path(ar_call[0]).name.lower().startswith("emar"))
            self.assertEqual(ar_call[1], "d")
            self.assertEqual(Path(ar_call[2]).resolve(), filtered_lib.resolve())
            self.assertNotEqual(Path(ar_call[2]).resolve(), orig_lib.resolve())
            self.assertEqual(ar_call[3:], ("wasm_boot.cpp.o", "main.cpp.o"))

            # Main module link has -sMAIN_MODULE=1, --whole-archive with the filtered archive,
            # dynamicLibraries in INCOMING_MODULE_JS_API, growable 16GB memory, pool size 48,
            # NO guest_part_*.wasm arguments, and ZERO shard or registration .o files.
            main_call = runner.link_calls[0]
            self.assertIn("-sMAIN_MODULE=1", main_call)
            self.assertNotIn("-sMAIN_MODULE=2", main_call)
            self.assertIn("-sMEMORY64=1", main_call)
            self.assertIn("-pthread", main_call)
            self.assertIn("-sPTHREAD_POOL_SIZE=48", main_call)
            self.assertIn("-sALLOW_MEMORY_GROWTH=1", main_call)
            self.assertIn("-sMAXIMUM_MEMORY=16GB", main_call)
            self.assertIn("-sINITIAL_MEMORY=64MB", main_call)
            self.assertIn("-lexports.js", main_call)
            self.assertIn("-fwasm-exceptions", main_call)
            self.assertIn("-Wl,--strip-all", main_call)
            incoming_tokens = [t for t in main_call if t.startswith("-sINCOMING_MODULE_JS_API=")]
            self.assertEqual(len(incoming_tokens), 1)
            self.assertIn("dynamicLibraries", incoming_tokens[0].split("=", 1)[1].split(","))

            wa_idx = main_call.index("-Wl,--whole-archive")
            nwa_idx = main_call.index("-Wl,--no-whole-archive")
            self.assertEqual(nwa_idx, wa_idx + 2)
            self.assertEqual(Path(main_call[wa_idx + 1]).resolve(), filtered_lib.resolve())

            main_names = [Path(t).name for t in main_call]
            self.assertNotIn("libplaygta6_core.a", main_names)
            self.assertIn("libplaygta6_core_split.a", main_names)
            self.assertIn("libplaygta6_core_split.a", report.link_argv)
            self.assertIn("-Wl,--whole-archive", report.link_argv)
            self.assertIn("-Wl,--no-whole-archive", report.link_argv)
            for part_name in (
                "guest_part_0.wasm",
                "guest_part_1.wasm",
                "guest_part_2.wasm",
                "guest_part_3.wasm",
            ):
                self.assertNotIn(part_name, main_names)
                self.assertNotIn(part_name, report.link_argv)
            for non_shard_id in (
                "main/stubs/pg6_import_thunks.cpp",
                "exports/pg6_export_thunks.cpp",
                "pg6_program.cpp",
                "program_modules.cpp",
                "runtime/main.cpp",
                "runtime/wasm_boot.cpp",
            ):
                obj = by_id[non_shard_id].object_path
                assert obj is not None
                self.assertIn(obj.name, main_names)
            for excluded_id in (*expected_shards, *expected_regs):
                obj = by_id[excluded_id].object_path
                assert obj is not None
                self.assertNotIn(obj.name, main_names)

            # JSON and sanitized text never leak host paths, backslashes, or username.
            dumped = bgw.report_to_json(report)
            self.assertNotIn("\\\\", dumped)
            user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
            if len(user) >= 3:
                self.assertNotRegex(dumped, rf"(?<![A-Za-z0-9_]){user}(?![A-Za-z0-9_])")
            payload = json.loads(dumped)
            self.assertEqual(payload["split_modules"], 3)
            self.assertEqual(len(payload["side_modules"]), 4)
            self.assertEqual(
                [m["name"] for m in payload["side_modules"]],
                [
                    "guest_part_0.wasm",
                    "guest_part_1.wasm",
                    "guest_part_2.wasm",
                    "guest_part_3.wasm",
                ],
            )
            text = bgw.format_report_text(report, sanitize=True)
            self.assertIn("Split modules: 3", text)
            self.assertIn("-sMAIN_MODULE=1", text)
            self.assertIn("Side modules:", text)
            self.assertIn("guest_part_0.wasm", text)
            self.assertIn("guest_part_3.wasm", text)
            if len(user) >= 3:
                self.assertNotRegex(text, rf"(?<![A-Za-z0-9_]){user}(?![A-Za-z0-9_])")

    def test_split_modules_greedy_bin_packing_fewer_shards_and_side_link_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 6 shards with distinct sizes: 500, 450, 300, 250, 200, 100.
            # LPT greedy into 3 buckets:
            #   b0: 500 + 100 = 600
            #   b1: 450 + 200 = 650
            #   b2: 300 + 250 = 550
            sizes = {
                "lift/shard-00000.ll": 100,
                "lift/shard-00001.ll": 500,
                "lift/shard-00002.ll": 250,
                "lift/shard-00003.ll": 450,
                "lift/shard-00004.ll": 200,
                "lift/shard-00005.ll": 300,
            }
            items = [
                bgw.UnitBuildResult(
                    unit=bgw.SourceUnit(uid, root / uid, "ll"),
                    cache_key="k",
                    cache_hit=True,
                    object_path=root / f"{Path(uid).stem}.o",
                    argv=(),
                    status="hit",
                    object_bytes=sz,
                )
                for uid, sz in sizes.items()
            ]
            # Add a non-shard unit that must be ignored by partition_shard_units.
            items.append(
                bgw.UnitBuildResult(
                    unit=bgw.SourceUnit("lift/functions.ll", root / "lift/functions.ll", "ll"),
                    cache_key="k",
                    cache_hit=True,
                    object_path=root / "functions.o",
                    argv=(),
                    status="hit",
                    object_bytes=9999,
                )
            )
            buckets = bgw.partition_shard_units(items, 3)
            self.assertEqual(len(buckets), 3)
            self.assertEqual(
                [[it.unit.unit_id for it in b] for b in buckets],
                [
                    ["lift/shard-00000.ll", "lift/shard-00001.ll"],
                    ["lift/shard-00003.ll", "lift/shard-00004.ll"],
                    ["lift/shard-00002.ll", "lift/shard-00005.ll"],
                ],
            )
            # Identical output on reversed input order (100% deterministic).
            buckets_rev = bgw.partition_shard_units(list(reversed(items)), 3)
            self.assertEqual(
                [[it.unit.unit_id for it in b] for b in buckets_rev],
                [[it.unit.unit_id for it in b] for b in buckets],
            )

            # Fewer shards (2) than split_modules (5): emits 2 shard buckets + 1 registration side module.
            repo = self._repo(root)
            small_src = root / "two-shards"
            _write(small_src / "lift" / "shard-00000.ll", "; s0\n")
            _write(small_src / "lift" / "shard-00001.ll", "; s1\n")
            _write(small_src / "lift" / "functions.ll", "; reg\n")
            runner_two = RecordingRunner()
            report_two = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=small_src,
                repo_root=repo,
                cache_dir=root / "cache-two",
                out_dir=root / "dist-two",
                jobs=2,
                split_modules=5,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner_two,
                which_fn=lambda _name: None,
            )
            self.assertTrue(report_two.ok, report_two.errors)
            self.assertEqual(report_two.split_modules, 5)
            self.assertEqual(len(report_two.side_modules), 3)
            self.assertEqual(
                [m.name for m in report_two.side_modules],
                ["guest_part_0.wasm", "guest_part_1.wasm", "guest_part_2.wasm"],
            )
            self.assertEqual(report_two.side_modules[2].shards, ("lift/functions.ll",))
            self.assertEqual(len(runner_two.side_link_calls), 3)
            self.assertEqual(len(runner_two.link_calls), 1)

            # Failing side module link aborts before main module link.
            runner_fail = RecordingRunner(fail_outputs=frozenset({"guest_part_1.wasm"}))
            report_fail = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=small_src,
                repo_root=repo,
                cache_dir=root / "cache-two",
                out_dir=root / "dist-fail",
                jobs=2,
                split_modules=2,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner_fail,
                which_fn=lambda _name: None,
            )
            self.assertFalse(report_fail.ok)
            self.assertEqual(runner_fail.link_calls, [])
            self.assertTrue(
                any("guest_part_1.wasm" in e for e in report_fail.errors),
                report_fail.errors,
            )

    def test_split_modules_v8_1gib_limit_checked_per_wasm_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _program_layout(root)
            orig_object_bytes = bgw._object_bytes

            # Case A: total > 1 GiB (500 MiB + 3 * 600 MiB) but EVERY module <= 1 GiB -> no warning.
            sizes_ok = {
                "runtime.wasm": 500 * 1024 * 1024,
                "guest_part_0.wasm": 600 * 1024 * 1024,
                "guest_part_1.wasm": 600 * 1024 * 1024,
                "guest_part_2.wasm": 600 * 1024 * 1024,
            }

            def fake_bytes_ok(path: Path | None) -> int:
                if path is not None and path.name in sizes_ok:
                    return sizes_ok[path.name]
                return orig_object_bytes(path)

            bgw._object_bytes = fake_bytes_ok
            try:
                report_ok = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                    source_dir=src,
                    repo_root=repo,
                    cache_dir=root / "cache-ok",
                    out_dir=root / "dist-ok",
                    jobs=2,
                    split_modules=3,
                    runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                    run_command=RecordingRunner(),
                    which_fn=lambda _name: None,
                )
            finally:
                bgw._object_bytes = orig_object_bytes

            self.assertTrue(report_ok.ok, report_ok.errors)
            self.assertFalse(report_ok.exceeds_v8_1gib_limit)
            self.assertEqual(report_ok.warnings, [])
            self.assertEqual(report_ok.wasm_bytes, 500 * 1024 * 1024)
            self.assertTrue(all(not s.exceeds_v8_1gib_limit for s in report_ok.side_modules))

            # Case B: guest_part_1.wasm exceeds 1 GiB while runtime.wasm and other parts are under.
            over_one = bgw.V8_MAX_WASM_MODULE_BYTES + 1
            sizes_side_over = {
                "runtime.wasm": 10 * 1024 * 1024,
                "guest_part_0.wasm": 600 * 1024 * 1024,
                "guest_part_1.wasm": over_one,
                "guest_part_2.wasm": 400 * 1024 * 1024,
            }

            def fake_bytes_side_over(path: Path | None) -> int:
                if path is not None and path.name in sizes_side_over:
                    return sizes_side_over[path.name]
                return orig_object_bytes(path)

            bgw._object_bytes = fake_bytes_side_over
            try:
                report_side_over = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                    source_dir=src,
                    repo_root=repo,
                    cache_dir=root / "cache-ok",
                    out_dir=root / "dist-side-over",
                    jobs=2,
                    split_modules=3,
                    runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                    run_command=RecordingRunner(),
                    which_fn=lambda _name: None,
                )
            finally:
                bgw._object_bytes = orig_object_bytes

            self.assertTrue(report_side_over.ok, report_side_over.errors)
            self.assertTrue(report_side_over.exceeds_v8_1gib_limit)
            self.assertEqual(
                [s.exceeds_v8_1gib_limit for s in report_side_over.side_modules],
                [False, True, False, False],
            )
            self.assertEqual(len(report_side_over.warnings), 1)
            self.assertIn("guest_part_1.wasm", report_side_over.warnings[0])
            self.assertIn(str(over_one), report_side_over.warnings[0])
            self.assertIn("1073741824", report_side_over.warnings[0])
            text = bgw.format_report_text(report_side_over)
            self.assertIn("guest_part_1.wasm", text)
            self.assertIn("[EXCEEDS V8 1 GiB LIMIT]", text)

            # Case C: runtime.wasm exceeds 1 GiB while side modules are under.
            sizes_main_over = {
                "runtime.wasm": over_one,
                "guest_part_0.wasm": 300 * 1024 * 1024,
                "guest_part_1.wasm": 300 * 1024 * 1024,
                "guest_part_2.wasm": 300 * 1024 * 1024,
            }

            def fake_bytes_main_over(path: Path | None) -> int:
                if path is not None and path.name in sizes_main_over:
                    return sizes_main_over[path.name]
                return orig_object_bytes(path)

            bgw._object_bytes = fake_bytes_main_over
            try:
                report_main_over = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                    source_dir=src,
                    repo_root=repo,
                    cache_dir=root / "cache-ok",
                    out_dir=root / "dist-main-over",
                    jobs=2,
                    split_modules=3,
                    runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                    run_command=RecordingRunner(),
                    which_fn=lambda _name: None,
                )
            finally:
                bgw._object_bytes = orig_object_bytes

            self.assertTrue(report_main_over.ok, report_main_over.errors)
            self.assertTrue(report_main_over.exceeds_v8_1gib_limit)
            self.assertTrue(all(not s.exceeds_v8_1gib_limit for s in report_main_over.side_modules))
            self.assertEqual(len(report_main_over.warnings), 1)
            self.assertIn("runtime.wasm", report_main_over.warnings[0])

    def test_split_modules_cli_dry_run_sanitization_and_invalid_values(self) -> None:
        for bad in (0, -1, -7, True, False, "3", 1.5, None):
            with self.assertRaises(bgw.BuildGuestWasmError, msg=f"split_modules={bad!r}"):
                bgw.parse_split_modules(bad)  # type: ignore[arg-type]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _program_layout(root)
            cache = root / "cache-dry"

            for bad in (0, -1, True, False, "2"):
                with self.assertRaises(bgw.BuildGuestWasmError, msg=f"run_build split_modules={bad!r}"):
                    bgw.run_build(
                link_cache=False,
                link_jobs=1,
                        source_dir=src,
                        repo_root=repo,
                        split_modules=bad,  # type: ignore[arg-type]
                        dry_run=True,
                        which_fn=lambda _name: None,
                    )

            for bad_cli in ("0", "-1"):
                bad_code, bad_out, _bad_err = _run_cli(
                    [
                        str(src),
                        "--dry-run",
                        "--json",
                        "--split-modules",
                        bad_cli,
                        "--repo-root",
                        str(repo),
                    ]
                )
                self.assertEqual(bad_code, 2, bad_out)
                bad_payload = json.loads(bad_out)
                self.assertFalse(bad_payload["ok"])
                self.assertTrue(any("split-modules" in e for e in bad_payload["errors"]))

            with self.assertRaises(SystemExit) as exc_ctx:
                _run_cli(
                    [
                        str(src),
                        "--dry-run",
                        "--split-modules",
                        "abc",
                        "--repo-root",
                        str(repo),
                    ]
                )
            self.assertEqual(exc_ctx.exception.code, 2)

            # CLI --split-modules 3 --dry-run --json produces clean sanitized JSON without touching disk.
            code, out, err = _run_cli(
                [
                    str(src),
                    "--dry-run",
                    "--json",
                    "--split-modules",
                    "3",
                    "--strip",
                    "--wasm-exceptions",
                    "--repo-root",
                    str(repo),
                    "--cache-dir",
                    str(cache),
                    "--out-dir",
                    str(root / "dist-dry"),
                ]
            )
            self.assertEqual(code, 0, err)
            self.assertFalse(cache.exists())
            payload = json.loads(out)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["split_modules"], 3)
            self.assertIn("-fPIC", payload["compile_flags"])
            self.assertIn("-sMAIN_MODULE=1", payload["link_flags"])
            self.assertNotIn("-sMAIN_MODULE=2", payload["link_flags"])
            self.assertIn("-sMAIN_MODULE=1", payload["link_argv"])
            self.assertIn("-sPTHREAD_POOL_SIZE=48", payload["link_flags"])
            self.assertIn("-sALLOW_MEMORY_GROWTH=1", payload["link_flags"])
            self.assertIn("-sMAXIMUM_MEMORY=16GB", payload["link_flags"])
            self.assertIn("-sINITIAL_MEMORY=64MB", payload["link_flags"])
            self.assertIn("-Wl,--whole-archive", payload["link_argv"])
            self.assertIn("libplaygta6_core_split.a", payload["link_argv"])
            self.assertIn("-Wl,--no-whole-archive", payload["link_argv"])
            self.assertNotIn("libplaygta6_core.a", payload["link_argv"])
            incoming_flags = [
                f for f in payload["link_flags"] if f.startswith("-sINCOMING_MODULE_JS_API=")
            ]
            self.assertEqual(len(incoming_flags), 1)
            self.assertIn("dynamicLibraries", incoming_flags[0].split("=", 1)[1].split(","))
            self.assertEqual(len(payload["side_modules"]), 4)
            for idx, side in enumerate(payload["side_modules"]):
                self.assertEqual(side["name"], f"guest_part_{idx}.wasm")
                self.assertEqual(side["response_file"], f"side_part_{idx}.rsp")
                self.assertIn("-sSIDE_MODULE=1", side["link_argv"])
                self.assertIn(f"@side_part_{idx}.rsp", side["link_argv"])
                self.assertNotIn(f"guest_part_{idx}.wasm", payload["link_argv"])
            for unit in payload["sources"]:
                self.assertIn("-fPIC", unit["argv"])

            dumped = json.dumps(payload)
            self.assertNotIn("\\\\", dumped)
            user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
            if len(user) >= 3:
                self.assertNotRegex(dumped, rf"(?<![A-Za-z0-9_]){user}(?![A-Za-z0-9_])")

            # CLI text mode with --sanitize renders Split modules & Side modules cleanly.
            code_txt, out_txt, err_txt = _run_cli(
                [
                    str(src),
                    "--dry-run",
                    "--sanitize",
                    "--split-modules",
                    "3",
                    "--repo-root",
                    str(repo),
                    "--cache-dir",
                    str(root / "cache-txt"),
                    "--out-dir",
                    str(root / "dist-txt"),
                ]
            )
            self.assertEqual(code_txt, 0, err_txt)
            self.assertIn("Split modules: 3", out_txt)
            self.assertIn("Side modules:", out_txt)
            self.assertIn("guest_part_0.wasm", out_txt)
            self.assertIn("guest_part_3.wasm", out_txt)
            if len(user) >= 3:
                self.assertNotRegex(out_txt, rf"(?<![A-Za-z0-9_]){user}(?![A-Za-z0-9_])")

    def test_r31_split_registration_side_module_main_rsp_filtered_archive_and_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _program_layout(root)
            cache = root / "cache"
            out_dir = root / "dist"
            orig_lib = repo / "runtime" / "build-wasm64" / "libplaygta6_core.a"
            sentinel = b"!<arch>\noriginal-runtime-archive-bytes\n"
            orig_lib.write_bytes(sentinel)

            runner = RecordingRunner()
            report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=out_dir,
                jobs=2,
                split_modules=4,
                response_file=True,
                browser_boot_exports=True,
                link_guest_boot=True,
                runtime_lib=orig_lib,
                run_command=runner,
                which_fn=lambda _name: None,
            )
            self.assertTrue(report.ok, report.errors)
            # 5 shards across 4 shard buckets (guest_part_0..3.wasm) + 1 reg side module (guest_part_4.wasm).
            self.assertEqual(len(report.side_modules), 5)
            reg_side = report.side_modules[-1]
            self.assertEqual(reg_side.index, 4)
            self.assertEqual(reg_side.name, "guest_part_4.wasm")
            self.assertEqual(reg_side.response_file, "side_part_4.rsp")
            self.assertEqual(
                reg_side.shards,
                ("main/lift/functions.ll", "vcruntime140/lift/functions.ll"),
            )

            # Registration functions.ll objects are in side_part_4.rsp and NOT in link_objects.rsp.
            by_id = {u.unit.unit_id: u for u in report.sources}
            reg_rsp = (cache / "side_part_4.rsp").read_text(encoding="utf-8").splitlines()
            main_rsp = (cache / "link_objects.rsp").read_text(encoding="utf-8").splitlines()
            for reg_id in ("main/lift/functions.ll", "vcruntime140/lift/functions.ll"):
                obj = by_id[reg_id].object_path
                assert obj is not None
                obj_posix = str(obj).replace("\\", "/")
                self.assertIn(obj_posix, reg_rsp)
                self.assertNotIn(obj_posix, main_rsp)
            for main_id in (
                "exports/pg6_export_thunks.cpp",
                "main/stubs/pg6_import_thunks.cpp",
                "pg6_program.cpp",
                "program_modules.cpp",
                "runtime/main.cpp",
                "runtime/wasm_boot.cpp",
            ):
                obj = by_id[main_id].object_path
                assert obj is not None
                self.assertIn(str(obj).replace("\\", "/"), main_rsp)

            # emar d ran on cache copy, never modifying the original archive.
            self.assertEqual(len(runner.ar_calls), 1)
            ar_call = runner.ar_calls[0]
            filtered_copy = cache / "libplaygta6_core_split.a"
            self.assertTrue(filtered_copy.is_file())
            self.assertEqual(ar_call[1], "d")
            self.assertEqual(Path(ar_call[2]).resolve(), filtered_copy.resolve())
            self.assertTrue(Path(ar_call[2]).resolve().is_relative_to(cache.resolve()))
            self.assertNotEqual(Path(ar_call[2]).resolve(), orig_lib.resolve())
            self.assertEqual(ar_call[3:], ("wasm_boot.cpp.o", "main.cpp.o"))
            self.assertEqual(orig_lib.read_bytes(), sentinel)

            # Main link command checks: MAIN_MODULE=1, whole-archive filtered copy,
            # zero guest_part_*.wasm args, single INCOMING_MODULE_JS_API with dynamicLibraries,
            # EXPORTED_RUNTIME_METHODS present, growable 16GB memory + 64MB initial, pool size 48.
            self.assertEqual(len(runner.link_calls), 1)
            main_call = runner.link_calls[0]
            self.assertIn("-sMAIN_MODULE=1", main_call)
            self.assertNotIn("-sMAIN_MODULE=2", main_call)
            self.assertIn("-Wl,--whole-archive", main_call)
            self.assertIn("-Wl,--no-whole-archive", main_call)
            wa_idx = main_call.index("-Wl,--whole-archive")
            self.assertEqual(Path(main_call[wa_idx + 1]).resolve(), filtered_copy.resolve())
            self.assertEqual(main_call[wa_idx + 2], "-Wl,--no-whole-archive")
            self.assertFalse(any("guest_part_" in tok for tok in main_call))
            self.assertNotIn(str(orig_lib), main_call)
            incoming = [t for t in main_call if t.startswith("-sINCOMING_MODULE_JS_API=")]
            self.assertEqual(len(incoming), 1)
            api_items = incoming[0].split("=", 1)[1].split(",")
            self.assertIn("dynamicLibraries", api_items)
            self.assertIn("locateFile", api_items)
            self.assertIn(
                f"-sEXPORTED_RUNTIME_METHODS={bgw.BROWSER_BOOT_EXPORTED_RUNTIME_METHODS}",
                main_call,
            )
            self.assertIn("-sALLOW_MEMORY_GROWTH=1", main_call)
            self.assertIn("-sMAXIMUM_MEMORY=16GB", main_call)
            self.assertIn("-sINITIAL_MEMORY=64MB", main_call)
            self.assertIn("-sPTHREAD_POOL_SIZE=48", main_call)
            self.assertNotIn("-sPTHREAD_POOL_SIZE=4", main_call)

            # Explicit --initial-memory and --pthread-pool-size override split defaults.
            override_flags = bgw.resolve_link_flags(
                main_module=True,
                initial_memory="128MB",
                pthread_pool_size=12,
            )
            self.assertIn("-sINITIAL_MEMORY=128MB", override_flags)
            self.assertNotIn("-sINITIAL_MEMORY=64MB", override_flags)
            self.assertIn("-sPTHREAD_POOL_SIZE=12", override_flags)
            self.assertNotIn("-sPTHREAD_POOL_SIZE=48", override_flags)

            # emar d failure aborts before main link.
            runner_ar_fail = RecordingRunner(fail_outputs=frozenset({"libplaygta6_core_split.a"}))
            fail_report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist-ar-fail",
                jobs=2,
                split_modules=2,
                runtime_lib=orig_lib,
                run_command=runner_ar_fail,
                which_fn=lambda _name: None,
            )
            self.assertFalse(fail_report.ok)
            self.assertEqual(len(runner_ar_fail.ar_calls), 1)
            self.assertEqual(runner_ar_fail.link_calls, [])
            self.assertTrue(
                any("libplaygta6_core_split.a" in e for e in fail_report.errors),
                fail_report.errors,
            )

            # Split build with no functions.ll emits only the shard side modules.
            no_reg_src = root / "no-reg"
            _write(no_reg_src / "lift" / "shard-00000.ll", "; s0\n")
            _write(no_reg_src / "lift" / "shard-00001.ll", "; s1\n")
            _write(no_reg_src / "stubs" / "thunks.cpp", "void t() {}\n")
            runner_no_reg = RecordingRunner()
            no_reg_report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=no_reg_src,
                repo_root=repo,
                cache_dir=root / "cache-no-reg",
                out_dir=root / "dist-no-reg",
                jobs=2,
                split_modules=2,
                runtime_lib=orig_lib,
                run_command=runner_no_reg,
                which_fn=lambda _name: None,
            )
            self.assertTrue(no_reg_report.ok, no_reg_report.errors)
            self.assertEqual(
                [m.name for m in no_reg_report.side_modules],
                ["guest_part_0.wasm", "guest_part_1.wasm"],
            )

    def test_r31_split_modules_one_commands_unchanged_with_browser_boot_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _program_layout(root)
            orig_lib = repo / "runtime" / "build-wasm64" / "libplaygta6_core.a"
            runner = RecordingRunner()
            report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache-mono",
                out_dir=root / "dist-mono",
                jobs=2,
                split_modules=1,
                browser_boot_exports=True,
                link_guest_boot=True,
                runtime_lib=orig_lib,
                run_command=runner,
                which_fn=lambda _name: None,
            )
            self.assertTrue(report.ok, report.errors)
            self.assertEqual(runner.side_link_calls, [])
            self.assertEqual(runner.ar_calls, [])
            self.assertEqual(len(runner.link_calls), 1)
            link = runner.link_calls[0]
            self.assertEqual(
                report.link_flags,
                bgw.resolve_link_flags(browser_boot_exports=True, main_module=False),
            )
            self.assertIn("-sPTHREAD_POOL_SIZE=4", link)
            self.assertNotIn("-sPTHREAD_POOL_SIZE=48", link)
            self.assertNotIn("-sMAIN_MODULE=1", link)
            self.assertNotIn("-sMAIN_MODULE=2", link)
            self.assertNotIn("-Wl,--whole-archive", link)
            self.assertNotIn("-Wl,--no-whole-archive", link)
            self.assertIn(str(orig_lib), link)
            self.assertFalse(any(t.startswith("-sINITIAL_MEMORY=") for t in link))
            self.assertIn(
                f"-sINCOMING_MODULE_JS_API={bgw.BROWSER_BOOT_INCOMING_MODULE_JS_API}",
                link,
            )
            self.assertNotIn("dynamicLibraries", " ".join(link))


class TestNodeHeapAndHeaderCacheKeys(unittest.TestCase):
    """R30: NODE_OPTIONS heap scaling for split-module links & sibling C/C++ header hashing."""

    def _repo(self, tmp: Path) -> Path:
        repo = tmp / "repo"
        _fake_emsdk(repo)
        _touch_lib(repo / "runtime" / "build-wasm64" / "libplaygta6_core.a")
        _write(repo / "runtime" / "main.cpp", "int main() { return 0; }\n")
        _write(repo / "runtime" / "wasm_boot.cpp", "void run_wasm_guest_boot() {}\n")
        return repo

    def test_build_link_env_scaling_merging_and_validation(self) -> None:
        # split_modules=1 with no explicit override leaves NODE_OPTIONS unset or untouched.
        base_empty = {"PATH": "/usr/bin"}
        env_single = bgw.build_link_env(base_empty, split_modules=1)
        self.assertNotIn("NODE_OPTIONS", env_single)
        self.assertEqual(env_single, base_empty)
        self.assertIsNot(env_single, base_empty)

        base_flags = {"NODE_OPTIONS": "--trace-warnings --no-deprecation"}
        env_single_flags = bgw.build_link_env(base_flags, split_modules=1)
        self.assertEqual(env_single_flags["NODE_OPTIONS"], "--trace-warnings --no-deprecation")

        # split_modules > 1 auto-sets --max-old-space-size=16384 when absent.
        env_split = bgw.build_link_env(base_empty, split_modules=4)
        self.assertEqual(env_split["NODE_OPTIONS"], "--max-old-space-size=16384")
        self.assertNotIn("NODE_OPTIONS", base_empty)  # original dict not mutated

        # split_modules > 1 merges without clobbering existing NODE_OPTIONS flags.
        env_split_merged = bgw.build_link_env(base_flags, split_modules=4)
        self.assertEqual(
            env_split_merged["NODE_OPTIONS"],
            "--trace-warnings --no-deprecation --max-old-space-size=16384",
        )

        # Whitespace-only NODE_OPTIONS gets clean flag with no leading spaces.
        env_split_ws = bgw.build_link_env({"NODE_OPTIONS": "   "}, split_modules=2)
        self.assertEqual(env_split_ws["NODE_OPTIONS"], "--max-old-space-size=16384")

        # Existing --max-old-space-size (equals, space, or underscore form) is respected when unset.
        for existing_opt in (
            "--max-old-space-size=8192",
            "--trace-warnings --max-old-space-size=8192 --no-warnings",
            "--trace-warnings --max-old-space-size 8192",
            "--max_old_space_size=8192",
        ):
            env_respected = bgw.build_link_env(
                {"NODE_OPTIONS": existing_opt},
                split_modules=4,
            )
            self.assertEqual(env_respected["NODE_OPTIONS"], existing_opt)

        # Explicit node_max_old_space_mb overrides existing --max-old-space-size while preserving
        # all other flags (including Windows paths with backslashes and quoted spaces).
        existing_complex = (
            '--trace-warnings --max-old-space-size=4096 --require "C:\\Program Files\\hook.js"'
        )
        env_override = bgw.build_link_env(
            {"NODE_OPTIONS": existing_complex},
            split_modules=4,
            node_max_old_space_mb=24576,
        )
        self.assertEqual(
            env_override["NODE_OPTIONS"],
            '--trace-warnings --require "C:\\Program Files\\hook.js" --max-old-space-size=24576',
        )

        # Explicit node_max_old_space_mb also works for split_modules=1 and space-separated forms.
        env_single_explicit = bgw.build_link_env(
            {"NODE_OPTIONS": "--max-old-space-size 4096 --trace-warnings"},
            split_modules=1,
            node_max_old_space_mb=12288,
        )
        self.assertEqual(
            env_single_explicit["NODE_OPTIONS"],
            "--trace-warnings --max-old-space-size=12288",
        )
        env_single_fresh = bgw.build_link_env({}, split_modules=1, node_max_old_space_mb=12288)
        self.assertEqual(env_single_fresh["NODE_OPTIONS"], "--max-old-space-size=12288")

        # Invalid values raise BuildGuestWasmError.
        for bad in (0, -1, -16384, True, False, "16384", 1.5):
            with self.assertRaises(bgw.BuildGuestWasmError, msg=f"mb={bad!r}"):
                bgw.build_link_env({}, split_modules=4, node_max_old_space_mb=bad)  # type: ignore[arg-type]
        with self.assertRaises(bgw.BuildGuestWasmError):
            bgw.build_link_env({}, split_modules=0)

    def test_main_module_link_receives_scaled_node_options_env(self) -> None:
        class EnvCapturingRunner(RecordingRunner):
            def __init__(self) -> None:
                super().__init__()
                self.link_envs: list[dict[str, str] | None] = []
                self.side_link_envs: list[dict[str, str] | None] = []

            def __call__(
                self,
                argv: Sequence[str],
                *,
                env: dict[str, str] | None = None,
            ) -> bgw.CommandResult:
                tokens = tuple(str(a) for a in argv)
                is_side = bgw.SIDE_MODULE_FLAG in tokens
                is_main_link = "-lexports.js" in tokens
                if is_side:
                    self.side_link_envs.append(dict(env) if env is not None else None)
                elif is_main_link:
                    self.link_envs.append(dict(env) if env is not None else None)
                return super().__call__(argv)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _program_layout(root)
            cache = root / "cache"

            # 1) split_modules=3 with existing unrelated NODE_OPTIONS auto-merges 16384.
            runner_split = EnvCapturingRunner()
            report_split = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist-split",
                jobs=2,
                split_modules=3,
                env={"NODE_OPTIONS": "--trace-warnings", "KEEP_ME": "yes"},
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner_split,
                which_fn=lambda _name: None,
            )
            self.assertTrue(report_split.ok, report_split.errors)
            self.assertEqual(report_split.node_max_old_space_mb, 16384)
            self.assertEqual(len(runner_split.link_envs), 1)
            link_env = runner_split.link_envs[0]
            assert link_env is not None
            self.assertEqual(link_env["KEEP_ME"], "yes")
            self.assertEqual(
                link_env["NODE_OPTIONS"],
                "--trace-warnings --max-old-space-size=16384",
            )
            # Side module links (3 shard buckets + 1 registration side module) do not get main-module env override.
            self.assertEqual(len(runner_split.side_link_envs), 4)
            self.assertTrue(all(e is None for e in runner_split.side_link_envs))

            # 2) split_modules=3 respects existing --max-old-space-size in env.
            runner_respect = EnvCapturingRunner()
            report_respect = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist-respect",
                jobs=2,
                split_modules=3,
                env={"NODE_OPTIONS": "--max-old-space-size=8192"},
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner_respect,
                which_fn=lambda _name: None,
            )
            self.assertTrue(report_respect.ok, report_respect.errors)
            self.assertIsNone(report_respect.node_max_old_space_mb)
            assert runner_respect.link_envs[0] is not None
            self.assertEqual(
                runner_respect.link_envs[0]["NODE_OPTIONS"],
                "--max-old-space-size=8192",
            )

            # 3) Explicit node_max_old_space_mb overrides existing NODE_OPTIONS value.
            runner_override = EnvCapturingRunner()
            report_override = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist-override",
                jobs=2,
                split_modules=3,
                node_max_old_space_mb=20480,
                env={"NODE_OPTIONS": "--trace-warnings --max-old-space-size=8192"},
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner_override,
                which_fn=lambda _name: None,
            )
            self.assertTrue(report_override.ok, report_override.errors)
            self.assertEqual(report_override.node_max_old_space_mb, 20480)
            assert runner_override.link_envs[0] is not None
            self.assertEqual(
                runner_override.link_envs[0]["NODE_OPTIONS"],
                "--trace-warnings --max-old-space-size=20480",
            )

            # 4) split_modules=1 leaves NODE_OPTIONS unset unless explicitly requested.
            runner_single = EnvCapturingRunner()
            report_single = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache-single",
                out_dir=root / "dist-single",
                jobs=2,
                split_modules=1,
                env={"KEEP_ME": "1"},
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner_single,
                which_fn=lambda _name: None,
            )
            self.assertTrue(report_single.ok, report_single.errors)
            self.assertIsNone(report_single.node_max_old_space_mb)
            assert runner_single.link_envs[0] is not None
            self.assertNotIn("NODE_OPTIONS", runner_single.link_envs[0])

            runner_single_explicit = EnvCapturingRunner()
            report_single_explicit = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache-single",
                out_dir=root / "dist-single-explicit",
                jobs=2,
                split_modules=1,
                node_max_old_space_mb=10240,
                env={"KEEP_ME": "1"},
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner_single_explicit,
                which_fn=lambda _name: None,
            )
            self.assertTrue(report_single_explicit.ok, report_single_explicit.errors)
            self.assertEqual(report_single_explicit.node_max_old_space_mb, 10240)
            assert runner_single_explicit.link_envs[0] is not None
            self.assertEqual(
                runner_single_explicit.link_envs[0]["NODE_OPTIONS"],
                "--max-old-space-size=10240",
            )

            # 5) Legacy 1-arg runner remains 100% compatible.
            legacy_runner = RecordingRunner()
            report_legacy = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist-legacy",
                jobs=2,
                split_modules=3,
                node_max_old_space_mb=16384,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=legacy_runner,
                which_fn=lambda _name: None,
            )
            self.assertTrue(report_legacy.ok, report_legacy.errors)
            self.assertEqual(len(legacy_runner.link_calls), 1)

    def test_default_run_command_passes_env_to_subprocess(self) -> None:
        custom_env = dict(os.environ)
        custom_env["NODE_OPTIONS"] = "--max-old-space-size=16384"
        res = bgw.default_run_command(
            [sys.executable, "-c", "import os; print(os.environ.get('NODE_OPTIONS', ''))"],
            env=custom_env,
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(res.stdout.strip(), "--max-old-space-size=16384")

    def test_node_max_old_space_mb_cli_flags_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _program_layout(root)

            old_node_opts = os.environ.pop("NODE_OPTIONS", None)
            try:
                # Auto-scaled to 16384 for --split-modules 4.
                code, out, err = _run_cli(
                    [
                        str(src),
                        "--dry-run",
                        "--json",
                        "--split-modules",
                        "4",
                        "--repo-root",
                        str(repo),
                        "--cache-dir",
                        str(root / "cache1"),
                        "--out-dir",
                        str(root / "dist1"),
                    ]
                )
                self.assertEqual(code, 0, err)
                payload = json.loads(out)
                self.assertTrue(payload["ok"])
                self.assertEqual(payload["node_max_old_space_mb"], 16384)

                # Untouched (None) for default --split-modules 1.
                code_single, out_single, err_single = _run_cli(
                    [
                        str(src),
                        "--dry-run",
                        "--json",
                        "--repo-root",
                        str(repo),
                        "--cache-dir",
                        str(root / "cache2"),
                        "--out-dir",
                        str(root / "dist2"),
                    ]
                )
                self.assertEqual(code_single, 0, err_single)
                self.assertIsNone(json.loads(out_single)["node_max_old_space_mb"])

                # Explicit --node-max-old-space-mb sets JSON & text output.
                code_exp, out_exp, err_exp = _run_cli(
                    [
                        str(src),
                        "--dry-run",
                        "--json",
                        "--split-modules",
                        "4",
                        "--node-max-old-space-mb",
                        "24576",
                        "--repo-root",
                        str(repo),
                        "--cache-dir",
                        str(root / "cache3"),
                        "--out-dir",
                        str(root / "dist3"),
                    ]
                )
                self.assertEqual(code_exp, 0, err_exp)
                self.assertEqual(json.loads(out_exp)["node_max_old_space_mb"], 24576)

                code_txt, out_txt, err_txt = _run_cli(
                    [
                        str(src),
                        "--dry-run",
                        "--sanitize",
                        "--node-max-old-space-mb",
                        "12288",
                        "--repo-root",
                        str(repo),
                        "--cache-dir",
                        str(root / "cache4"),
                        "--out-dir",
                        str(root / "dist4"),
                    ]
                )
                self.assertEqual(code_txt, 0, err_txt)
                self.assertIn("Node max old space: 12288 MB", out_txt)
                self.assertIn("--max-old-space-size=12288", out_txt)

                # Invalid values exit with code 2.
                for bad_cli in ("0", "-512"):
                    bad_code, bad_out, _bad_err = _run_cli(
                        [
                            str(src),
                            "--dry-run",
                            "--json",
                            "--node-max-old-space-mb",
                            bad_cli,
                            "--repo-root",
                            str(repo),
                        ]
                    )
                    self.assertEqual(bad_code, 2, bad_out)
                    bad_payload = json.loads(bad_out)
                    self.assertFalse(bad_payload["ok"])
                    self.assertTrue(
                        any("node-max-old-space-mb" in e for e in bad_payload["errors"])
                    )
            finally:
                if old_node_opts is not None:
                    os.environ["NODE_OPTIONS"] = old_node_opts

    def test_sibling_header_changes_invalidate_cpp_and_c_while_ll_keys_stay_unchanged(self) -> None:
        # Direct cache_key_for backwards compatibility with extra_fingerprint.
        cpp_flags = bgw.compile_flags_for(opt="O2", kind="cpp")
        ll_flags = bgw.compile_flags_for(opt="O2", kind="ll")
        base_cpp_key = bgw.cache_key_for(b"int x = 1;\n", cpp_flags, kind="cpp")
        self.assertEqual(
            base_cpp_key,
            bgw.cache_key_for(b"int x = 1;\n", cpp_flags, kind="cpp", extra_fingerprint=b""),
        )
        fp_key_a = bgw.cache_key_for(b"int x = 1;\n", cpp_flags, kind="cpp", extra_fingerprint=b"hdr-a")
        fp_key_b = bgw.cache_key_for(b"int x = 1;\n", cpp_flags, kind="cpp", extra_fingerprint=b"hdr-b")
        self.assertNotEqual(base_cpp_key, fp_key_a)
        self.assertNotEqual(fp_key_a, fp_key_b)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _program_layout(root)
            # Add sibling headers matching real R9/R12 layout + a .hpp + a .c unit + stray .h next to .ll.
            seam_h = src / "main" / "stubs" / "pg6_guest_seam.h"
            _write(seam_h, "#pragma once\n// v1\n")
            _write(src / "main" / "stubs" / "pg6_import_thunks.h", "#pragma once\nvoid thunk_main();\n")
            export_hpp = src / "exports" / "pg6_export_thunks.hpp"
            _write(export_hpp, "#pragma once\n// export v1\n")
            helper_c = _write(src / "c_helpers" / "helper.c", "int helper(void) { return 1; }\n")
            helper_h = _write(src / "c_helpers" / "helper.h", "#pragma once\n#define HELPER_VAL 1\n")
            # Subdirectory ending in .h must be ignored.
            (src / "main" / "stubs" / "dir.h").mkdir(parents=True, exist_ok=True)
            # Stray header inside .ll shard directory must NEVER affect .ll cache keys.
            stray_ll_h = _write(src / "main" / "lift" / "stray.h", "#pragma once\n// v1\n")
            # Ensure helper.c is discovered via program-manifest.json too.
            manifest_path = src / "program-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"].append({"path": "c_helpers/helper.c"})
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            # Verify sibling_headers_fingerprint returns b"" when directory has no headers.
            no_hdr_cpp = _write(root / "isolated" / "plain.cpp", "int plain() { return 0; }\n")
            self.assertEqual(bgw.sibling_headers_fingerprint(no_hdr_cpp), b"")
            self.assertTrue(len(bgw.sibling_headers_fingerprint(helper_c)) > 0)

            cache = root / "cache"
            # Step 1: Warm the cache.
            runner1 = RecordingRunner()
            first = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist1",
                jobs=2,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner1,
                which_fn=lambda _name: None,
            )
            self.assertTrue(first.ok, first.errors)
            self.assertEqual(first.cache_hits, 0)
            self.assertEqual(first.cache_misses, len(first.sources))
            first_keys = {u.unit.unit_id: u.cache_key for u in first.sources}

            # Verify every .ll shard cache key matches bare cache_key_for(..., kind="ll") exactly!
            for item in first.sources:
                if item.unit.kind == "ll":
                    expected_ll_key = bgw.cache_key_for(
                        item.unit.path.read_bytes(),
                        ll_flags,
                        kind="ll",
                    )
                    self.assertEqual(item.cache_key, expected_ll_key, item.unit.unit_id)

            # Step 2: Unchanged rebuild -> 100% cache hits.
            runner2 = RecordingRunner()
            second = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist2",
                jobs=2,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner2,
                which_fn=lambda _name: None,
            )
            self.assertTrue(second.ok, second.errors)
            self.assertEqual(second.cache_hits, len(second.sources))
            self.assertEqual(second.cache_misses, 0)
            self.assertEqual(runner2.compile_calls, [])

            # Step 3: Update main/stubs/pg6_guest_seam.h -> ONLY main/stubs/pg6_import_thunks.cpp misses!
            _write(seam_h, "#pragma once\n// v2 updated seam signature\n")
            runner3 = RecordingRunner()
            third = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist3",
                jobs=2,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner3,
                which_fn=lambda _name: None,
            )
            self.assertTrue(third.ok, third.errors)
            self.assertEqual(third.cache_misses, 1)
            self.assertEqual(third.cache_hits, len(third.sources) - 1)
            self.assertEqual(len(runner3.compile_calls), 1)
            for item in third.sources:
                if item.unit.unit_id == "main/stubs/pg6_import_thunks.cpp":
                    self.assertEqual(item.status, "compiled")
                    self.assertNotEqual(item.cache_key, first_keys[item.unit.unit_id])
                else:
                    self.assertEqual(item.status, "hit", item.unit.unit_id)
                    self.assertEqual(item.cache_key, first_keys[item.unit.unit_id])

            # Step 4: Update exports/pg6_export_thunks.hpp -> ONLY exports/pg6_export_thunks.cpp misses!
            _write(export_hpp, "#pragma once\n// export v2\n")
            runner4 = RecordingRunner()
            fourth = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist4",
                jobs=2,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner4,
                which_fn=lambda _name: None,
            )
            self.assertTrue(fourth.ok, fourth.errors)
            self.assertEqual(fourth.cache_misses, 1)
            compiled_ids_4 = [u.unit.unit_id for u in fourth.sources if u.status == "compiled"]
            self.assertEqual(compiled_ids_4, ["exports/pg6_export_thunks.cpp"])

            # Step 5: Update c_helpers/helper.h -> ONLY c_helpers/helper.c misses!
            _write(helper_h, "#pragma once\n#define HELPER_VAL 2\n")
            runner5 = RecordingRunner()
            fifth = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist5",
                jobs=2,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner5,
                which_fn=lambda _name: None,
            )
            self.assertTrue(fifth.ok, fifth.errors)
            self.assertEqual(fifth.cache_misses, 1)
            compiled_ids_5 = [u.unit.unit_id for u in fifth.sources if u.status == "compiled"]
            self.assertEqual(compiled_ids_5, ["c_helpers/helper.c"])

            # Step 6: Modify stray header inside main/lift/ -> ZERO misses, .ll cache keys unchanged!
            _write(stray_ll_h, "#pragma once\n// v2 modified next to .ll\n")
            _write(src / "main" / "lift" / "added.hpp", "#pragma once\n")
            runner6 = RecordingRunner()
            sixth = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist6",
                jobs=2,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner6,
                which_fn=lambda _name: None,
            )
            self.assertTrue(sixth.ok, sixth.errors)
            self.assertEqual(sixth.cache_misses, 0)
            self.assertEqual(sixth.cache_hits, len(sixth.sources))
            self.assertEqual(runner6.compile_calls, [])
            for item in sixth.sources:
                if item.unit.kind == "ll":
                    self.assertEqual(item.cache_key, first_keys[item.unit.unit_id])

            # Step 7: Direct compile_unit() also respects sibling header invalidation.
            unit = bgw.SourceUnit(
                unit_id="main/stubs/pg6_import_thunks.cpp",
                path=src / "main" / "stubs" / "pg6_import_thunks.cpp",
                kind="cpp",
                module="main",
            )
            direct_flags = bgw.compile_flags_for(opt="O2", kind="cpp")
            direct_cache = root / "direct-cache"
            r_dir_1 = bgw.compile_unit(
                unit,
                compiler=Path("em++"),
                flags=direct_flags,
                cache_dir=direct_cache,
                dry_run=False,
                run_command=RecordingRunner(),
            )
            self.assertEqual(r_dir_1.status, "compiled")
            r_dir_2 = bgw.compile_unit(
                unit,
                compiler=Path("em++"),
                flags=direct_flags,
                cache_dir=direct_cache,
                dry_run=False,
                run_command=RecordingRunner(),
            )
            self.assertEqual(r_dir_2.status, "hit")
            _write(seam_h, "#pragma once\n// v3 direct compile_unit check\n")
            r_dir_3 = bgw.compile_unit(
                unit,
                compiler=Path("em++"),
                flags=direct_flags,
                cache_dir=direct_cache,
                dry_run=False,
                run_command=RecordingRunner(),
            )
            self.assertEqual(r_dir_3.status, "compiled")
            self.assertNotEqual(r_dir_1.cache_key, r_dir_3.cache_key)


class TestProxyToPthreadAndSplitJsOptimizer(unittest.TestCase):
    """R33: --proxy-to-pthread and split-module V8-cage GC + throttled pthread CMD_LOAD pass."""

    EMSCRIPTEN_701_FIXTURE = r"""
const CMD_LOAD = 'load';
const CMD_LOADED = 'loaded';
const CMD_RUN = 'run';

var LDSO = {
  loadedLibsByName: {
    'guest_part_0.wasm': { exports: { sub_1000: () => 10, keep_meta: 1 } },
    'guest_part_1.wasm': { exports: { sub_2000: (state) => state + 2 } },
    'other_plugin.wasm': { exports: { plugin_init: () => 'ok' } },
  },
};
var wasmImports = {
  sub_1000: () => 10,
  emscripten_notify: () => 'alive',
  malloc: () => 128,
};
var GOT = {
  sub_1000: { value: 1 },
  sub_2000: { value: 2 },
  __stack_pointer: { value: 4096 },
};
var functionsInTableMap = new Map([
  ['sub_1000', 1],
  ['sub_2000', 2],
]);

var __debugClearChecks = [];
var __debugResolveCalls = 0;

function instantiateModule(libName, localScope, exportsObj) {
  var moduleExports;
  function resolveSymbol(sym) {
    __debugResolveCalls++;
    var resolved = wasmImports[sym];
    if (!resolved && localScope) {
      resolved = localScope[sym];
    }
    if (!resolved && moduleExports) {
      resolved = moduleExports[sym];
    }
    return resolved;
  }

  var proxyHandler = {
    get(stubs, prop) {
      if (!(prop in stubs)) {
        var resolved;
        stubs[prop] = (...args) => {
          resolved ||= resolveSymbol(prop);
          return resolved(...args);
        };
      }
      return stubs[prop];
    }
  };
  var envProxy = new Proxy({}, proxyHandler);
  moduleExports = exportsObj;
  __debugClearChecks.push(() => ({
    moduleExportsIsNull: moduleExports === null,
    localScopeIsNull: localScope === null,
  }));
  return { envProxy, resolveSymbol };
}

var __createdStubs = {};

async function loadDylibs() {
  // Part 0 imports sub_2000 (not in wasmImports yet) and sub_missing (weak/undefined).
  var part0 = instantiateModule(
    'guest_part_0.wasm',
    { sub_local_0: () => 7 },
    { sub_1000: () => 10 }
  );
  __createdStubs.sub_2000 = part0.envProxy.sub_2000;
  __createdStubs.sub_local_0 = part0.envProxy.sub_local_0;
  __createdStubs.sub_missing = part0.envProxy.sub_missing;

  // Part 1 loads afterwards and publishes sub_2000 into wasmImports.
  wasmImports.sub_2000 = (state) => state + 2;
  instantiateModule(
    'guest_part_1.wasm',
    { sub_local_1: () => 9 },
    { sub_2000: wasmImports.sub_2000 }
  );
}

async function initRuntime() {
  await loadDylibs();
}

var PThread = {
  loadWasmModuleToWorker(worker) {
    return new Promise((onFinishedLoading) => {
      worker.onmessage = (e) => {
        var d = e.data;
        var cmd = d.cmd;
        switch (cmd) {
          case CMD_LOADED:
            worker.loaded = true;
            onFinishedLoading(worker);
            break;
          default:
            break;
        }
      };
      worker.postMessage({
        cmd: CMD_LOAD,
        urlOrBlob: 'runtime.js',
        wasmMemory: { shared: true },
        wasmModule: { id: 'main' },
        dynamicLibraries: ['guest_part_0.wasm', 'guest_part_1.wasm'],
      });
    });
  },
};
"""

    def _repo(self, tmp: Path) -> Path:
        repo = tmp / "repo"
        _fake_emsdk(repo)
        _touch_lib(repo / "runtime" / "build-wasm64" / "libplaygta6_core.a")
        _write(repo / "runtime" / "main.cpp", "int main() { return 0; }\n")
        _write(repo / "runtime" / "wasm_boot.cpp", "void run_wasm_guest_boot() {}\n")
        return repo

    def _require_node(self) -> str:
        node = shutil.which("node")
        if node is None:
            self.skipTest("node executable not available on PATH")
        return node

    def _run_node_check(self, js_text: str) -> None:
        node = self._require_node()
        with tempfile.TemporaryDirectory() as tmp:
            js_path = Path(tmp) / "runtime.js"
            js_path.write_text(js_text, encoding="utf-8")
            proc = subprocess.run(
                [node, "--check", str(js_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                proc.returncode,
                0,
                f"node --check failed:\nstdout={proc.stdout}\nstderr={proc.stderr}",
            )

    def _run_node_script(self, js_text: str) -> dict[str, object]:
        node = self._require_node()
        with tempfile.TemporaryDirectory() as tmp:
            js_path = Path(tmp) / "harness.js"
            js_path.write_text(js_text, encoding="utf-8")
            proc = subprocess.run(
                [node, str(js_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                proc.returncode,
                0,
                f"node harness failed:\nstdout={proc.stdout}\nstderr={proc.stderr}",
            )
            return json.loads(proc.stdout)

    def test_proxy_to_pthread_flag_api_and_cli_backwards_compatibility(self) -> None:
        # Default preserves existing flags identically (no -sPROXY_TO_PTHREAD=1).
        default_flags = bgw.resolve_link_flags()
        self.assertEqual(default_flags, (*bgw.LINK_FLAGS, "-sASSERTIONS=0", "-sSAFE_HEAP=0"))
        self.assertNotIn("-sPROXY_TO_PTHREAD=1", default_flags)
        self.assertNotIn("-sPROXY_TO_PTHREAD=1", bgw.LINK_FLAGS)

        # Enabled via resolve_link_flags and build_link_argv.
        proxied_flags = bgw.resolve_link_flags(proxy_to_pthread=True)
        self.assertIn("-sPROXY_TO_PTHREAD=1", proxied_flags)
        self.assertEqual(proxied_flags.count("-sPROXY_TO_PTHREAD=1"), 1)

        proxied_split_flags = bgw.resolve_link_flags(main_module=True, proxy_to_pthread=True)
        self.assertIn("-sMAIN_MODULE=1", proxied_split_flags)
        self.assertIn("-sPROXY_TO_PTHREAD=1", proxied_split_flags)

        argv_default = bgw.build_link_argv(
            "em++",
            [Path("a.o")],
            output_js=Path("web/dist/runtime.js"),
        )
        self.assertNotIn("-sPROXY_TO_PTHREAD=1", argv_default)

        argv_proxied = bgw.build_link_argv(
            "em++",
            [Path("a.o")],
            output_js=Path("web/dist/runtime.js"),
            proxy_to_pthread=True,
        )
        self.assertIn("-sPROXY_TO_PTHREAD=1", argv_proxied)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _layout_sources(root)

            # run_build default keeps proxy_to_pthread=False.
            runner_default = RecordingRunner()
            rep_default = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache1",
                out_dir=root / "dist1",
                jobs=1,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner_default,
                which_fn=lambda _name: None,
            )
            self.assertTrue(rep_default.ok, rep_default.errors)
            self.assertFalse(rep_default.proxy_to_pthread)
            self.assertNotIn("-sPROXY_TO_PTHREAD=1", rep_default.link_flags)
            self.assertNotIn("-sPROXY_TO_PTHREAD=1", runner_default.link_calls[0])

            # run_build with proxy_to_pthread=True appends -sPROXY_TO_PTHREAD=1.
            runner_proxied = RecordingRunner()
            rep_proxied = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache2",
                out_dir=root / "dist2",
                jobs=1,
                proxy_to_pthread=True,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner_proxied,
                which_fn=lambda _name: None,
            )
            self.assertTrue(rep_proxied.ok, rep_proxied.errors)
            self.assertTrue(rep_proxied.proxy_to_pthread)
            self.assertIn("-sPROXY_TO_PTHREAD=1", rep_proxied.link_flags)
            self.assertIn("-sPROXY_TO_PTHREAD=1", rep_proxied.link_argv)
            self.assertIn("-sPROXY_TO_PTHREAD=1", runner_proxied.link_calls[0])
            self.assertIn("Proxy to pthread: yes (-sPROXY_TO_PTHREAD=1)", bgw.format_report_text(rep_proxied))

            # CLI --proxy-to-pthread & --max-concurrent-pthread-loads dry-run JSON.
            code, out, err = _run_cli(
                [
                    str(src),
                    "--dry-run",
                    "--json",
                    "--proxy-to-pthread",
                    "--max-concurrent-pthread-loads",
                    "3",
                    "--repo-root",
                    str(repo),
                    "--cache-dir",
                    str(root / "cache-cli"),
                    "--out-dir",
                    str(root / "dist-cli"),
                ]
            )
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["proxy_to_pthread"])
            self.assertEqual(payload["max_concurrent_pthread_loads"], 3)
            self.assertIn("-sPROXY_TO_PTHREAD=1", payload["link_flags"])
            self.assertIn("-sPROXY_TO_PTHREAD=1", payload["link_argv"])

            # Invalid --max-concurrent-pthread-loads values are rejected.
            for bad in (-1, -5, True, False, "2"):
                with self.assertRaises(bgw.BuildGuestWasmError, msg=f"bad={bad!r}"):
                    bgw.parse_max_concurrent_pthread_loads(bad)  # type: ignore[arg-type]

            bad_code, bad_out, _bad_err = _run_cli(
                [
                    str(src),
                    "--dry-run",
                    "--json",
                    "--max-concurrent-pthread-loads",
                    "-1",
                    "--repo-root",
                    str(repo),
                ]
            )
            self.assertEqual(bad_code, 2, bad_out)
            bad_payload = json.loads(bad_out)
            self.assertFalse(bad_payload["ok"])
            self.assertTrue(any("max-concurrent-pthread-loads" in e for e in bad_payload["errors"]))

    def test_optimize_split_runtime_js_node_execution_v8_cage_cleanup_and_ordered_throttle(self) -> None:
        patched, status = bgw.optimize_split_runtime_js(
            self.EMSCRIPTEN_701_FIXTURE,
            max_concurrent_pthread_loads=2,
        )
        self.assertEqual(
            status,
            {
                "already_patched": False,
                "stub_binders": True,
                "export_clearers": True,
                "post_load_dylibs_cleanup": True,
                "pthread_cmd_load_throttle": True,
                "pthread_cmd_loaded_dispatch": True,
            },
        )
        self.assertIn(bgw.SPLIT_PTHREAD_OPT_MARKER, patched)
        self._run_node_check(patched)

        # Execute live Node.js harness verifying BOTH Patch 1 (V8 cage cleanup + bound stubs)
        # AND Patch 2 (max 2 concurrent CMD_LOADs + strict CMD_LOAD-before-CMD_RUN ordering).
        harness = patched + r"""
(async () => {
  // ---- Part 1: V8 4 GiB cage cleanup & eager cross-shard stub binding ----
  await initRuntime();

  const resolveCallsAfterInit = __debugResolveCalls;
  // Cross-shard lazy stub MUST still invoke resolved target AFTER cleanup without re-resolving!
  const crossShardResult = __createdStubs.sub_2000(40);
  const localScopeResult = __createdStubs.sub_local_0();
  const resolveCallsAfterStubInvocation = __debugResolveCalls;

  const wasmImportKeys = Object.keys(wasmImports).sort();
  const gotKeys = Object.keys(GOT).sort();
  const part0ExportKeys = Object.keys(LDSO.loadedLibsByName['guest_part_0.wasm'].exports);
  const part1ExportKeys = Object.keys(LDSO.loadedLibsByName['guest_part_1.wasm'].exports);
  const otherPluginKeys = Object.keys(LDSO.loadedLibsByName['other_plugin.wasm'].exports);
  const clearStates = __debugClearChecks.map((fn) => fn());

  // ---- Part 2: Ordered throttled pthread CMD_LOAD (max 2 concurrent) ----
  let inFlightLoads = 0;
  let peakInFlightLoads = 0;
  let runBeforeLoadViolations = 0;
  const workerLogs = [];

  class ProtoWorker {
    constructor(id) {
      this.id = id;
      this.onmessage = null;
      this.loaded = false;
      this.seenLoad = false;
      this.messages = [];
      workerLogs.push(this.messages);
    }
    postMessage(msg, transfer) {
      if (msg.cmd === CMD_LOAD) {
        this.seenLoad = true;
        inFlightLoads++;
        if (inFlightLoads > peakInFlightLoads) peakInFlightLoads = inFlightLoads;
      } else if (msg.cmd === CMD_RUN && !this.seenLoad) {
        runBeforeLoadViolations++;
      }
      this.messages.push({ cmd: msg.cmd, tag: msg.tag || null, hasTransfer: Array.isArray(transfer) });
    }
    completeLoad() {
      inFlightLoads--;
      this.onmessage({ data: { cmd: CMD_LOADED } });
    }
  }

  function makeOwnPostWorker(id) {
    const messages = [];
    workerLogs.push(messages);
    const w = {
      id,
      onmessage: null,
      loaded: false,
      seenLoad: false,
      messages,
      postMessage(msg, transfer) {
        if (msg.cmd === CMD_LOAD) {
          this.seenLoad = true;
          inFlightLoads++;
          if (inFlightLoads > peakInFlightLoads) peakInFlightLoads = inFlightLoads;
        } else if (msg.cmd === CMD_RUN && !this.seenLoad) {
          runBeforeLoadViolations++;
        }
        messages.push({ cmd: msg.cmd, tag: msg.tag || null, hasTransfer: Array.isArray(transfer) });
      },
      completeLoad() {
        inFlightLoads--;
        this.onmessage({ data: { cmd: CMD_LOADED } });
      },
    };
    return w;
  }

  // Mix prototype-based Workers (real browser Worker shape) and own-property mock Workers.
  const workers = [];
  const loadPromises = [];
  for (let i = 0; i < 8; i++) {
    const w = (i % 2 === 0) ? new ProtoWorker(i) : makeOwnPostWorker(i);
    workers.push(w);
    loadPromises.push(PThread.loadWasmModuleToWorker(w));
    // Immediately send CMD_RUN and an extra message with transferList BEFORE CMD_LOADED!
    w.postMessage({ cmd: CMD_RUN, tag: 'run-' + i });
    w.postMessage({ cmd: 'custom', tag: 'extra-' + i }, ['fake-port']);
  }

  const immediateInFlight = inFlightLoads;
  const queuedWorker2MsgsBeforeRelease = workers[2].messages.length;
  const queuedWorker7MsgsBeforeRelease = workers[7].messages.length;

  // Complete workers one by one in FIFO order; each completion should release exactly one queued load.
  const inFlightAfterEachStep = [];
  for (let i = 0; i < workers.length; i++) {
    workers[i].completeLoad();
    inFlightAfterEachStep.push(inFlightLoads);
  }
  await Promise.all(loadPromises);

  // Verify postMessage is cleanly unwrapped after load completes.
  workers[3].postMessage({ cmd: 'ping', tag: 'after-3' });
  const protoWorkerOwnPropRestored = !Object.prototype.hasOwnProperty.call(workers[2], 'postMessage');

  console.log(JSON.stringify({
    crossShardResult,
    localScopeResult,
    resolveCallsAfterInit,
    resolveCallsAfterStubInvocation,
    wasmImportKeys,
    gotKeys,
    part0ExportKeys,
    part1ExportKeys,
    otherPluginKeys,
    clearStates,
    functionsInTableMapSize: functionsInTableMap.size,
    stubBindersNull: globalThis.__pg6StubBinders === null,
    exportClearersNull: globalThis.__pg6ExportClearers === null,
    immediateInFlight,
    peakInFlightLoads,
    runBeforeLoadViolations,
    queuedWorker2MsgsBeforeRelease,
    queuedWorker7MsgsBeforeRelease,
    inFlightAfterEachStep,
    finalActiveLoads: PThread._activeLoads,
    finalQueueLength: PThread._loadQueue.length,
    protoWorkerOwnPropRestored,
    workerLogs,
  }));
})();
"""
        result = self._run_node_script(harness)
        # Patch 1 assertions.
        self.assertEqual(result["crossShardResult"], 42)
        self.assertEqual(result["localScopeResult"], 7)
        self.assertEqual(result["resolveCallsAfterStubInvocation"], result["resolveCallsAfterInit"])
        self.assertEqual(result["wasmImportKeys"], ["emscripten_notify", "malloc"])
        self.assertEqual(result["gotKeys"], ["__stack_pointer"])
        self.assertEqual(result["part0ExportKeys"], [])
        self.assertEqual(result["part1ExportKeys"], [])
        self.assertEqual(result["otherPluginKeys"], ["plugin_init"])
        self.assertEqual(
            result["clearStates"],
            [
                {"moduleExportsIsNull": True, "localScopeIsNull": True},
                {"moduleExportsIsNull": True, "localScopeIsNull": True},
            ],
        )
        self.assertEqual(result["functionsInTableMapSize"], 0)
        self.assertTrue(result["stubBindersNull"])
        self.assertTrue(result["exportClearersNull"])

        # Patch 2 assertions.
        self.assertEqual(result["immediateInFlight"], 2)
        self.assertEqual(result["peakInFlightLoads"], 2)
        self.assertEqual(result["runBeforeLoadViolations"], 0)
        self.assertEqual(result["queuedWorker2MsgsBeforeRelease"], 0)
        self.assertEqual(result["queuedWorker7MsgsBeforeRelease"], 0)
        self.assertEqual(result["inFlightAfterEachStep"], [2, 2, 2, 2, 2, 2, 1, 0])
        self.assertEqual(result["finalActiveLoads"], 0)
        self.assertEqual(result["finalQueueLength"], 0)
        self.assertTrue(result["protoWorkerOwnPropRestored"])

        worker_logs = result["workerLogs"]
        assert isinstance(worker_logs, list)
        self.assertEqual(len(worker_logs), 8)
        for idx, msgs in enumerate(worker_logs):
            expected_prefix = [
                {"cmd": "load", "tag": None, "hasTransfer": False},
                {"cmd": "run", "tag": f"run-{idx}", "hasTransfer": False},
                {"cmd": "custom", "tag": f"extra-{idx}", "hasTransfer": True},
            ]
            self.assertEqual(msgs[:3], expected_prefix, f"worker {idx}: {msgs}")
        self.assertEqual(
            worker_logs[3][-1],
            {"cmd": "ping", "tag": "after-3", "hasTransfer": False},
        )

    def test_optimize_split_runtime_js_idempotency_disable_throttle_and_safe_noop(self) -> None:
        # 1) Idempotent: calling twice yields byte-for-byte identical output.
        once, status1 = bgw.optimize_split_runtime_js(self.EMSCRIPTEN_701_FIXTURE)
        twice, status2 = bgw.optimize_split_runtime_js(once)
        self.assertEqual(twice, once)
        self.assertFalse(status1["already_patched"])
        self.assertTrue(status2["already_patched"])
        self.assertFalse(any(v for k, v in status2.items() if k != "already_patched"))

        # 2) max_concurrent_pthread_loads=0 disables Patch 2 while keeping Patch 1 active.
        no_throttle, status_nt = bgw.optimize_split_runtime_js(
            self.EMSCRIPTEN_701_FIXTURE,
            max_concurrent_pthread_loads=0,
        )
        self.assertTrue(status_nt["stub_binders"])
        self.assertTrue(status_nt["export_clearers"])
        self.assertTrue(status_nt["post_load_dylibs_cleanup"])
        self.assertFalse(status_nt["pthread_cmd_load_throttle"])
        self.assertFalse(status_nt["pthread_cmd_loaded_dispatch"])
        self.assertNotIn("PThread._loadQueue", no_throttle)
        self._run_node_check(no_throttle)

        # 3) Non-matching JS is returned completely unmodified without marker comment.
        plain_js = "export default function createRuntimeModule() { return {}; }\n"
        out_plain, status_plain = bgw.optimize_split_runtime_js(plain_js)
        self.assertEqual(out_plain, plain_js)
        self.assertNotIn(bgw.SPLIT_PTHREAD_OPT_MARKER, out_plain)
        self.assertFalse(any(status_plain.values()))

        # 4) Atomic guard: CMD_LOAD without CMD_LOADED does NOT apply throttle (prevents deadlock).
        load_only_js = (
            "var PThread = { loadWasmModuleToWorker(worker) {\n"
            "  worker.postMessage({ cmd: CMD_LOAD, wasmMemory });\n"
            "} };\n"
        )
        out_load_only, status_load_only = bgw.optimize_split_runtime_js(load_only_js)
        self.assertEqual(out_load_only, load_only_js)
        self.assertFalse(status_load_only["pthread_cmd_load_throttle"])
        self.assertFalse(status_load_only["pthread_cmd_loaded_dispatch"])

        # 5) Atomic guard: var moduleExports without await loadDylibs() leaves JS unmodified.
        exports_only_js = "function f() { var moduleExports; return moduleExports; }\n"
        out_exp_only, status_exp_only = bgw.optimize_split_runtime_js(exports_only_js)
        self.assertEqual(out_exp_only, exports_only_js)
        self.assertFalse(status_exp_only["export_clearers"])

    def test_optimize_split_runtime_js_whitespace_quote_and_weakmap_variations(self) -> None:
        # Realistic variation: quoted keys, string 'load'/'loaded', if-style handler,
        # WeakMap functionsInTableMap, non-first cmd property, strings with '})' inside postMessage,
        # and compact/minified lazy stub syntax.
        variant_js = r"""
var LDSO = { loadedLibsByName: { '/dist/guest_part_4.wasm': { exports: { sub_9: () => 99 } } } };
var wasmImports = { sub_9: () => 99, keep_fn: () => 1 };
var GOT = { sub_9: { value: 9 }, keep_got: { value: 1 } };
var functionsInTableMap = new WeakMap();
functionsInTableMap.set(wasmImports.sub_9, 1);

function loadWebAssemblyModule(localScope) {
  let moduleExports;
  function resolveSymbol(s) { return wasmImports[s] || (localScope && localScope[s]) || (moduleExports && moduleExports[s]); }
  var proxyHandler = {
    get(stubs, prop) {
      if(!(prop in stubs)){let resolved;stubs[prop]=(...args)=>{resolved||=resolveSymbol(prop);return resolved(...args)};}
      return stubs[prop];
    }
  };
  moduleExports = { sub_9: wasmImports.sub_9 };
  return new Proxy({}, proxyHandler);
}
var p;
async function loadDylibs() { p = loadWebAssemblyModule({}); p.sub_9; }
async function boot() { await loadDylibs(); }

var PThread = {
  loadWasmModuleToWorker(w) {
    w.onmessage = (e) => {
      var cmd = e.data.cmd;
      if (cmd === 'loaded') {
        w.done = true;
      }
    };
    w.postMessage({
      urlOrBlob: "https://example.invalid/runtime.js?a=({b})",
      'cmd': 'load',
      nested: { ok: true },
    });
  }
};
"""
        patched, status = bgw.optimize_split_runtime_js(variant_js, max_concurrent_pthread_loads=1)
        self.assertTrue(status["stub_binders"])
        self.assertTrue(status["export_clearers"])
        self.assertTrue(status["post_load_dylibs_cleanup"])
        self.assertTrue(status["pthread_cmd_load_throttle"])
        self.assertTrue(status["pthread_cmd_loaded_dispatch"])
        self._run_node_check(patched)

        res = self._run_node_script(
            patched
            + r"""
(async () => {
  await boot();
  const val = p.sub_9();
  const seen = [];
  const makeW = (id) => ({
    onmessage: null,
    postMessage(m) { seen.push([id, m.cmd]); },
  });
  const w0 = makeW(0), w1 = makeW(1);
  PThread.loadWasmModuleToWorker(w0);
  PThread.loadWasmModuleToWorker(w1);
  w1.postMessage({ cmd: 'run' });
  const beforeComplete = seen.slice();
  w0.onmessage({ data: { cmd: 'loaded' } });
  w1.onmessage({ data: { cmd: 'loaded' } });
  console.log(JSON.stringify({
    val,
    weakMapCleared: functionsInTableMap === undefined,
    part4Cleared: Object.keys(LDSO.loadedLibsByName['/dist/guest_part_4.wasm'].exports).length === 0,
    beforeComplete,
    seen,
    active: PThread._activeLoads,
  }));
})();
"""
        )
        self.assertEqual(res["val"], 99)
        self.assertTrue(res["weakMapCleared"])
        self.assertTrue(res["part4Cleared"])
        self.assertEqual(res["beforeComplete"], [[0, "load"]])
        self.assertEqual(res["seen"], [[0, "load"], [1, "load"], [1, "run"]])
        self.assertEqual(res["active"], 0)

    def test_run_build_auto_patches_split_runtime_js_only_for_non_dry_split_builds(self) -> None:
        fixture_js = self.EMSCRIPTEN_701_FIXTURE

        class EmscriptenJsRunner(RecordingRunner):
            def __call__(self, argv: Sequence[str]) -> bgw.CommandResult:
                res = super().__call__(argv)
                out_path = _dash_o(tuple(str(a) for a in argv))
                if out_path is not None and Path(out_path).name == bgw.RUNTIME_JS:
                    Path(out_path).write_text(fixture_js, encoding="utf-8")
                return res

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _program_layout(root)
            orig_lib = repo / "runtime" / "build-wasm64" / "libplaygta6_core.a"

            # 1) Split build (split_modules=2) patches runtime.js automatically.
            dist_split = root / "dist-split"
            rep_split = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache-split",
                out_dir=dist_split,
                jobs=2,
                split_modules=2,
                proxy_to_pthread=True,
                max_concurrent_pthread_loads=2,
                runtime_lib=orig_lib,
                run_command=EmscriptenJsRunner(),
                which_fn=lambda _name: None,
            )
            self.assertTrue(rep_split.ok, rep_split.errors)
            self.assertTrue(rep_split.proxy_to_pthread)
            self.assertIn("-sPROXY_TO_PTHREAD=1", rep_split.link_flags)
            self.assertEqual(
                rep_split.split_runtime_js_patches,
                {
                    "already_patched": False,
                    "stub_binders": True,
                    "export_clearers": True,
                    "post_load_dylibs_cleanup": True,
                    "pthread_cmd_load_throttle": True,
                    "pthread_cmd_loaded_dispatch": True,
                },
            )
            patched_text = (dist_split / bgw.RUNTIME_JS).read_text(encoding="utf-8")
            self.assertIn(bgw.SPLIT_PTHREAD_OPT_MARKER, patched_text)
            self._run_node_check(patched_text)
            self.assertIn("Split runtime JS opt:", bgw.format_report_text(rep_split))

            # Re-running file wrapper directly is idempotent.
            second_status = bgw.patch_split_runtime_js_file(dist_split / bgw.RUNTIME_JS)
            self.assertTrue(second_status["already_patched"])
            self.assertEqual((dist_split / bgw.RUNTIME_JS).read_text(encoding="utf-8"), patched_text)

            # Missing file degrades cleanly without raising.
            missing_status = bgw.patch_split_runtime_js_file(root / "does_not_exist.js")
            self.assertFalse(any(missing_status.values()))

            # 2) Monolithic build (split_modules=1) leaves runtime.js untouched.
            dist_mono = root / "dist-mono"
            rep_mono = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache-mono",
                out_dir=dist_mono,
                jobs=2,
                split_modules=1,
                runtime_lib=orig_lib,
                run_command=EmscriptenJsRunner(),
                which_fn=lambda _name: None,
            )
            self.assertTrue(rep_mono.ok, rep_mono.errors)
            self.assertIsNone(rep_mono.split_runtime_js_patches)
            self.assertEqual((dist_mono / bgw.RUNTIME_JS).read_text(encoding="utf-8"), fixture_js)

            # 3) Dry-run split build skips patching and leaves split_runtime_js_patches=None.
            rep_dry = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache-dry",
                out_dir=root / "dist-dry",
                jobs=2,
                split_modules=2,
                dry_run=True,
                runtime_lib=orig_lib,
                which_fn=lambda _name: None,
            )
            self.assertTrue(rep_dry.ok, rep_dry.errors)
            self.assertIsNone(rep_dry.split_runtime_js_patches)


class TestContiguousSplitBucketing(unittest.TestCase):
    """R35: Contiguous VA-order split-module bucketing (--split-strategy contiguous)."""

    def _repo(self, tmp: Path) -> Path:
        repo = tmp / "repo"
        _fake_emsdk(repo)
        _touch_lib(repo / "runtime" / "build-wasm64" / "libplaygta6_core.a")
        _write(repo / "runtime" / "main.cpp", "int main() { return 0; }\n")
        _write(repo / "runtime" / "wasm_boot.cpp", "void run_wasm_guest_boot() {}\n")
        return repo

    @staticmethod
    def _make_shard_items(root: Path, sizes: Sequence[tuple[str, int]]) -> list[bgw.UnitBuildResult]:
        return [
            bgw.UnitBuildResult(
                unit=bgw.SourceUnit(uid, root / uid, "ll"),
                cache_key="k",
                cache_hit=True,
                object_path=root / f"{Path(uid).stem}.o",
                argv=(),
                status="hit",
                object_bytes=sz,
            )
            for uid, sz in sizes
        ]

    @staticmethod
    def _brute_force_optimal_contiguous(weights: Sequence[int], k: int) -> tuple[int, int, int]:
        """Return exact (minimax_max, min_sum_sq_w, min_sum_sq_c) over all k-way contiguous splits."""
        from itertools import combinations

        n = len(weights)
        best: tuple[int, int, int] | None = None
        for cuts in combinations(range(1, n), k - 1):
            bounds = (0, *cuts, n)
            bucket_sums = [sum(weights[bounds[i] : bounds[i + 1]]) for i in range(k)]
            bucket_cnts = [bounds[i + 1] - bounds[i] for i in range(k)]
            score = (
                max(bucket_sums),
                sum(s * s for s in bucket_sums),
                sum(c * c for c in bucket_cnts),
            )
            if best is None or score < best:
                best = score
        assert best is not None
        return best

    def _assert_strictly_contiguous(
        self,
        buckets: Sequence[Sequence[bgw.UnitBuildResult]],
        expected_sorted_ids: Sequence[str],
    ) -> None:
        flattened: list[str] = []
        for bucket in buckets:
            self.assertTrue(bucket, "contiguous partition must not emit empty buckets")
            ids = [item.unit.unit_id for item in bucket]
            self.assertEqual(ids, sorted(ids))
            flattened.extend(ids)
        # Flattened bucket order equals the exact sorted unit_id sequence => zero interleaving.
        self.assertEqual(flattened, list(expected_sorted_ids))

    def test_contiguous_partition_strict_va_order_and_exact_optimality_across_2_3_4_buckets(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 1) Compare directly against LPT balanced on a mixed-size sequence:
            #    balanced interleaves shard-00000 with shard-00001 and shard-00002 with shard-00005,
            #    while contiguous preserves strict VA/unit_id slices.
            mixed = [
                ("lift/shard-00000.ll", 100),
                ("lift/shard-00001.ll", 500),
                ("lift/shard-00002.ll", 250),
                ("lift/shard-00003.ll", 450),
                ("lift/shard-00004.ll", 200),
                ("lift/shard-00005.ll", 300),
            ]
            items = self._make_shard_items(root, mixed)
            expected_ids = [uid for uid, _ in mixed]

            buckets_3 = bgw.partition_shards_contiguous(items, 3)
            self.assertEqual(len(buckets_3), 3)
            self._assert_strictly_contiguous(buckets_3, expected_ids)
            self.assertEqual(
                [[it.unit.unit_id for it in b] for b in buckets_3],
                [
                    ["lift/shard-00000.ll", "lift/shard-00001.ll"],
                    ["lift/shard-00002.ll", "lift/shard-00003.ll"],
                    ["lift/shard-00004.ll", "lift/shard-00005.ll"],
                ],
            )
            self.assertEqual([sum(it.object_bytes for it in b) for b in buckets_3], [600, 700, 500])

            # Input order independence (reversed input produces identical contiguous buckets).
            buckets_3_rev = bgw.partition_shards_contiguous(list(reversed(items)), 3)
            self.assertEqual(
                [[it.unit.unit_id for it in b] for b in buckets_3_rev],
                [[it.unit.unit_id for it in b] for b in buckets_3],
            )

            # 2) Verify exact minimax + min-variance optimality against brute force across
            #    2, 3, and 4 buckets on diverse distributions (balanced, heavily skewed front/mid/back,
            #    sawtooth, and zero-weight gaps).
            distributions: list[tuple[str, list[int]]] = [
                ("uniform_8", [100, 100, 100, 100, 100, 100, 100, 100]),
                ("mixed_6", [100, 500, 250, 450, 200, 300]),
                ("front_skewed", [1000, 100, 100, 100, 100, 100]),
                ("back_skewed", [50, 60, 70, 80, 90, 1500]),
                ("middle_spike", [80, 90, 1200, 110, 70, 95, 85]),
                ("sawtooth_10", [400, 50, 350, 40, 300, 60, 450, 30, 250, 370]),
                ("with_zeros", [0, 300, 0, 0, 250, 0, 350, 0]),
                ("all_zeros", [0, 0, 0, 0, 0, 0, 0]),
            ]
            for label, weights in distributions:
                dist_items = self._make_shard_items(
                    root,
                    [(f"lift/shard-{idx:05d}.ll", w) for idx, w in enumerate(weights)],
                )
                dist_ids = [it.unit.unit_id for it in dist_items]
                for k in (2, 3, 4):
                    buckets = bgw.partition_shards_contiguous(dist_items, k)
                    self.assertEqual(len(buckets), k, f"{label} k={k}")
                    self._assert_strictly_contiguous(buckets, dist_ids)
                    got_sums = [sum(it.object_bytes for it in b) for b in buckets]
                    got_cnts = [len(b) for b in buckets]
                    got_score = (
                        max(got_sums),
                        sum(s * s for s in got_sums),
                        sum(c * c for c in got_cnts),
                    )
                    expected_score = self._brute_force_optimal_contiguous(weights, k)
                    self.assertEqual(
                        got_score,
                        expected_score,
                        f"{label} k={k}: got sums={got_sums} cnts={got_cnts}",
                    )

            # 3) Skewed 2-bucket split: [700, 100, 100, 600] -> [700] | [100, 100, 600] (max 800)
            #    vs [700, 100] (800) | [100, 600] (700) -> minimax 800, smaller variance (800^2+700^2).
            skew2 = self._make_shard_items(
                root,
                [
                    ("lift/shard-00000.ll", 700),
                    ("lift/shard-00001.ll", 100),
                    ("lift/shard-00002.ll", 100),
                    ("lift/shard-00003.ll", 600),
                ],
            )
            b2 = bgw.partition_shards_contiguous(skew2, 2)
            self.assertEqual([sum(it.object_bytes for it in b) for b in b2], [800, 700])
            self.assertEqual(
                [[it.unit.unit_id for it in b] for b in b2],
                [
                    ["lift/shard-00000.ll", "lift/shard-00001.ll"],
                    ["lift/shard-00002.ll", "lift/shard-00003.ll"],
                ],
            )

    def test_contiguous_partition_edge_cases_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Empty shards -> [].
            self.assertEqual(bgw.partition_shards_contiguous([], 3), [])
            self.assertEqual(
                bgw.partition_shard_units([], 3, strategy="contiguous"),
                [],
            )

            items = self._make_shard_items(
                root,
                [
                    ("lift/shard-00002.ll", 300),
                    ("lift/shard-00000.ll", 100),
                    ("lift/shard-00001.ll", 200),
                ],
            )

            # Single bucket returns all shards in sorted unit_id order.
            single = bgw.partition_shards_contiguous(items, 1)
            self.assertEqual(len(single), 1)
            self.assertEqual(
                [it.unit.unit_id for it in single[0]],
                ["lift/shard-00000.ll", "lift/shard-00001.ll", "lift/shard-00002.ll"],
            )

            # Fewer shards (3) than buckets (6) emits 3 non-empty singleton buckets in VA order.
            fewer = bgw.partition_shards_contiguous(items, 6)
            self.assertEqual(len(fewer), 3)
            self.assertEqual(
                [[it.unit.unit_id for it in b] for b in fewer],
                [
                    ["lift/shard-00000.ll"],
                    ["lift/shard-00001.ll"],
                    ["lift/shard-00002.ll"],
                ],
            )

            # partition_shard_units ignores non-shard units (e.g. functions.ll) under contiguous too.
            with_reg = list(items) + [
                bgw.UnitBuildResult(
                    unit=bgw.SourceUnit("lift/functions.ll", root / "lift/functions.ll", "ll"),
                    cache_key="k",
                    cache_hit=True,
                    object_path=root / "functions.o",
                    argv=(),
                    status="hit",
                    object_bytes=99999,
                )
            ]
            via_wrapper = bgw.partition_shard_units(with_reg, 2, strategy="contiguous")
            self.assertEqual(
                [[it.unit.unit_id for it in b] for b in via_wrapper],
                [
                    ["lift/shard-00000.ll", "lift/shard-00001.ll"],
                    ["lift/shard-00002.ll"],
                ],
            )

            # Invalid strategy values rejected cleanly.
            self.assertEqual(bgw.parse_split_strategy("balanced"), "balanced")
            self.assertEqual(bgw.parse_split_strategy("contiguous"), "contiguous")
            self.assertEqual(bgw.parse_split_strategy(" CONTIGUOUS "), "contiguous")
            self.assertEqual(bgw.parse_split_strategy("asymmetric"), "asymmetric")
            self.assertEqual(bgw.parse_split_strategy(" ASYMMETRIC "), "asymmetric")
            for bad in ("", "roundrobin", "random", "lpt", None, 123, True):
                with self.assertRaises(bgw.BuildGuestWasmError, msg=f"strategy={bad!r}"):
                    bgw.parse_split_strategy(bad)  # type: ignore[arg-type]
                with self.assertRaises(bgw.BuildGuestWasmError, msg=f"partition strategy={bad!r}"):
                    bgw.partition_shard_units(items, 2, strategy=bad)  # type: ignore[arg-type]
            for bad_k in (0, -1, True):
                with self.assertRaises(bgw.BuildGuestWasmError, msg=f"bucket_count={bad_k!r}"):
                    bgw.partition_shards_contiguous(items, bad_k)  # type: ignore[arg-type]

    def test_run_build_contiguous_wires_side_rsps_registration_module_and_cli(self) -> None:
        class SizedCompileRunner(RecordingRunner):
            """Writes distinct object byte sizes per shard so balanced vs contiguous diverge."""

            def __init__(self, size_by_stem: dict[str, int]) -> None:
                super().__init__()
                self.size_by_stem = size_by_stem

            def __call__(self, argv: Sequence[str]) -> bgw.CommandResult:
                res = super().__call__(argv)
                tokens = tuple(str(a) for a in argv)
                out_path = _dash_o(tokens)
                src_token = _source_token(tokens)
                if (
                    res.returncode == 0
                    and out_path is not None
                    and Path(out_path).suffix.lower() == ".o"
                    and src_token is not None
                ):
                    stem = Path(src_token).stem
                    size = self.size_by_stem.get(stem)
                    if size is not None:
                        Path(out_path).write_bytes(b"\x00" * size)
                return res

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = root / "recomp-contiguous"
            # 6 shards whose sizes cause LPT balanced to interleave ([0, 2, 5] vs [1, 3, 4]),
            # whereas contiguous keeps strict VA slices ([0, 1, 2] | [3, 4, 5]).
            shard_sizes = {
                "shard-00000": 500,
                "shard-00001": 450,
                "shard-00002": 100,
                "shard-00003": 100,
                "shard-00004": 450,
                "shard-00005": 400,
            }
            for name in sorted(shard_sizes):
                _write(src / "lift" / f"{name}.ll", f"; {name}\n")
            _write(src / "lift" / "functions.ll", "; registration table\n")
            _write(src / "stubs" / "pg6_import_thunks.cpp", "void thunk() {}\n")

            orig_lib = repo / "runtime" / "build-wasm64" / "libplaygta6_core.a"
            cache = root / "cache-contig"
            out_dir = root / "dist-contig"
            runner = SizedCompileRunner(shard_sizes)
            report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=out_dir,
                jobs=2,
                split_modules=2,
                split_strategy="contiguous",
                runtime_lib=orig_lib,
                run_command=runner,
                which_fn=lambda _name: None,
            )
            self.assertTrue(report.ok, report.errors)
            self.assertEqual(report.split_modules, 2)
            self.assertEqual(report.split_strategy, "contiguous")
            # 2 contiguous shard side modules (guest_part_0, guest_part_1) + functions.ll (guest_part_2).
            self.assertEqual(len(report.side_modules), 3)
            self.assertEqual(
                [m.name for m in report.side_modules],
                ["guest_part_0.wasm", "guest_part_1.wasm", "guest_part_2.wasm"],
            )
            self.assertEqual(
                report.side_modules[0].shards,
                (
                    "lift/shard-00000.ll",
                    "lift/shard-00001.ll",
                    "lift/shard-00002.ll",
                ),
            )
            self.assertEqual(
                report.side_modules[1].shards,
                (
                    "lift/shard-00003.ll",
                    "lift/shard-00004.ll",
                    "lift/shard-00005.ll",
                ),
            )
            self.assertEqual(report.side_modules[2].shards, ("lift/functions.ll",))

            # Verify side_part_0.rsp, side_part_1.rsp, and side_part_2.rsp contain exact object paths.
            by_id = {u.unit.unit_id: u for u in report.sources}
            for idx, side in enumerate(report.side_modules):
                self.assertEqual(side.response_file, f"side_part_{idx}.rsp")
                rsp_path = cache / f"side_part_{idx}.rsp"
                self.assertTrue(rsp_path.is_file())
                expected_objs = [
                    str(by_id[uid].object_path).replace("\\", "/") for uid in side.shards
                ]
                self.assertEqual(rsp_path.read_text(encoding="utf-8").splitlines(), expected_objs)

            # Report JSON & text include split_strategy.
            payload = json.loads(bgw.report_to_json(report))
            self.assertEqual(payload["split_strategy"], "contiguous")
            self.assertEqual(payload["split_modules"], 2)
            text = bgw.format_report_text(report, sanitize=True)
            self.assertIn("Split modules: 2 (strategy=contiguous,", text)

            # Contrast with default balanced strategy on the exact same cached objects:
            # balanced interleaves shard-00000..shard-00005 by descending size.
            bal_report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist-bal",
                jobs=2,
                split_modules=2,
                runtime_lib=orig_lib,
                run_command=RecordingRunner(),
                which_fn=lambda _name: None,
            )
            self.assertTrue(bal_report.ok, bal_report.errors)
            self.assertEqual(bal_report.split_strategy, "balanced")
            self.assertNotEqual(bal_report.side_modules[0].shards, report.side_modules[0].shards)

            # CLI --split-strategy contiguous dry-run JSON & text.
            code, out, err = _run_cli(
                [
                    str(src),
                    "--dry-run",
                    "--json",
                    "--split-modules",
                    "2",
                    "--split-strategy",
                    "contiguous",
                    "--repo-root",
                    str(repo),
                    "--cache-dir",
                    str(cache),
                    "--out-dir",
                    str(root / "dist-cli"),
                ]
            )
            self.assertEqual(code, 0, err)
            cli_payload = json.loads(out)
            self.assertTrue(cli_payload["ok"])
            self.assertEqual(cli_payload["split_strategy"], "contiguous")
            self.assertEqual(
                cli_payload["side_modules"][0]["shards"],
                ["lift/shard-00000.ll", "lift/shard-00001.ll", "lift/shard-00002.ll"],
            )
            self.assertEqual(
                cli_payload["side_modules"][1]["shards"],
                ["lift/shard-00003.ll", "lift/shard-00004.ll", "lift/shard-00005.ll"],
            )
            self.assertEqual(cli_payload["side_modules"][2]["shards"], ["lift/functions.ll"])

            # Invalid --split-strategy rejected with non-zero CLI exit code (2) in JSON & text modes.
            bad_code, bad_out, _bad_err = _run_cli(
                [
                    str(src),
                    "--dry-run",
                    "--json",
                    "--split-modules",
                    "2",
                    "--split-strategy",
                    "bogus",
                    "--repo-root",
                    str(repo),
                ]
            )
            self.assertEqual(bad_code, 2, bad_out)
            bad_payload = json.loads(bad_out)
            self.assertFalse(bad_payload["ok"])
            self.assertTrue(any("split-strategy" in e for e in bad_payload["errors"]))

            bad_code_txt, _bad_out_txt, bad_err_txt = _run_cli(
                [
                    str(src),
                    "--dry-run",
                    "--split-modules",
                    "2",
                    "--split-strategy",
                    "bogus",
                    "--repo-root",
                    str(repo),
                ]
            )
            self.assertEqual(bad_code_txt, 2)
            self.assertIn("split-strategy", bad_err_txt)


class TestAsymmetricSideModuleSplit(unittest.TestCase):
    """R37e: asymmetric contiguous pack + -sSIDE_MODULE=2 under the V8 1 GiB cap."""

    def _fake_item(self, unit_id: str, size: int) -> bgw.UnitBuildResult:
        path = Path(unit_id)
        return bgw.UnitBuildResult(
            unit=bgw.SourceUnit(unit_id, path, "ll"),
            cache_key="k",
            cache_hit=True,
            object_path=path.with_suffix(".o"),
            argv=(),
            status="hit",
            object_bytes=size,
        )

    def test_constants_and_side_module_mode_defaults(self) -> None:
        self.assertEqual(bgw.SIDE_MODULE_FLAG, "-sSIDE_MODULE=1")
        self.assertEqual(bgw.SIDE_MODULE_PRUNED_FLAG, "-sSIDE_MODULE=2")
        self.assertEqual(bgw.DEFAULT_MAX_SIDE_MODULE_MIB, 994)
        self.assertLessEqual(bgw.DEFAULT_MAX_SIDE_MODULE_MIB, bgw.MAX_SIDE_MODULE_MIB_LIMIT)
        self.assertEqual(bgw.MAX_SIDE_MODULE_MIB_LIMIT, 1024)
        self.assertEqual(bgw.side_module_mode_for_strategy("balanced"), 1)
        self.assertEqual(bgw.side_module_mode_for_strategy("contiguous"), 1)
        self.assertEqual(bgw.side_module_mode_for_strategy("asymmetric"), 2)
        pruned = bgw.resolve_side_module_link_flags(pruned=True)
        legacy = bgw.resolve_side_module_link_flags(pruned=False)
        self.assertIn(bgw.SIDE_MODULE_PRUNED_FLAG, pruned)
        self.assertNotIn(bgw.SIDE_MODULE_FLAG, pruned)
        self.assertIn(bgw.SIDE_MODULE_FLAG, legacy)
        self.assertNotIn(bgw.SIDE_MODULE_PRUNED_FLAG, legacy)

    def test_capped_partition_auto_n_balanced_and_deterministic(self) -> None:
        sizes = [300, 250, 200, 250, 300, 200, 100]  # 1600 total, cap 900 -> 2 parts
        items = [self._fake_item(f"lift/shard-{i:05d}.ll", sz) for i, sz in enumerate(sizes)]
        a = bgw.partition_shards_capped(items, 8, max_bucket_bytes=900)
        b = bgw.partition_shards_capped(list(reversed(items)), 8, max_bucket_bytes=900)
        self.assertEqual(len(a), 2)
        self.assertEqual([[x.unit.unit_id for x in p] for p in a], [[x.unit.unit_id for x in p] for p in b])
        weights = [sum(bgw._partition_weight(x) for x in p) for p in a]
        self.assertTrue(all(w <= 900 for w in weights))
        self.assertLessEqual(max(weights) - min(weights), 300)
        flat = [x.unit.unit_id for p in a for x in p]
        self.assertEqual(flat, sorted(flat))

    def test_capped_partition_grows_parts_with_size_and_single_part_when_small(self) -> None:
        small = [self._fake_item(f"lift/shard-{i:05d}.ll", 100) for i in range(5)]
        self.assertEqual(len(bgw.partition_shards_capped(small, 4, max_bucket_bytes=900)), 1)
        big = [self._fake_item(f"lift/shard-{i:05d}.ll", 100) for i in range(30)]
        parts = bgw.partition_shards_capped(big, 8, max_bucket_bytes=900)
        self.assertEqual(len(parts), 4)
        for p in parts:
            self.assertLessEqual(sum(bgw._partition_weight(x) for x in p), 900)

    def test_capped_partition_errors_instead_of_over_cap(self) -> None:
        items = [self._fake_item(f"lift/shard-{i:05d}.ll", 100) for i in range(30)]
        with self.assertRaisesRegex(bgw.BuildGuestWasmError, "raise --split-modules"):
            bgw.partition_shards_capped(items, 2, max_bucket_bytes=900)
        huge = [self._fake_item("lift/shard-00000.ll", 1000)]
        with self.assertRaisesRegex(bgw.BuildGuestWasmError, "alone is"):
            bgw.partition_shards_capped(huge, 4, max_bucket_bytes=900)
        self.assertEqual(bgw.side_module_mode_for_strategy("capped"), 2)

    def test_asymmetric_partition_targets_994_171_two_part_shape(self) -> None:
        # Scale the proven ~994/171 MiB shape down to small byte weights for unit tests.
        # part_0 packs up to 994, part_1 gets the remaining 171.
        sizes = [400, 300, 294, 100, 71]  # sums to 1165; ceiling 994 -> [400+300+294]=994 | [100+71]=171
        items = [
            self._fake_item(f"lift/shard-{idx:05d}.ll", size) for idx, size in enumerate(sizes)
        ]
        buckets = bgw.partition_shards_asymmetric(items, 2, max_bucket_bytes=994)
        self.assertEqual(len(buckets), 2)
        self.assertEqual(
            [it.unit.unit_id for it in buckets[0]],
            ["lift/shard-00000.ll", "lift/shard-00001.ll", "lift/shard-00002.ll"],
        )
        self.assertEqual(
            [it.unit.unit_id for it in buckets[1]],
            ["lift/shard-00003.ll", "lift/shard-00004.ll"],
        )
        self.assertEqual(sum(bgw._partition_weight(it) for it in buckets[0]), 994)
        self.assertEqual(sum(bgw._partition_weight(it) for it in buckets[1]), 171)
        # Largest module stays under a 1024 MiB-equivalent ceiling.
        self.assertLessEqual(sum(bgw._partition_weight(it) for it in buckets[0]), 1024)

        via_wrapper = bgw.partition_shard_units(
            items,
            2,
            strategy="asymmetric",
            max_bucket_bytes=994,
        )
        self.assertEqual(
            [[it.unit.unit_id for it in b] for b in via_wrapper],
            [[it.unit.unit_id for it in b] for b in buckets],
        )

        # Default MiB ceiling converts to bytes via DEFAULT_MAX_SIDE_MODULE_MIB.
        default_cap = bgw.DEFAULT_MAX_SIDE_MODULE_MIB * 1024 * 1024
        big = [
            self._fake_item("lift/shard-00000.ll", default_cap - 10),
            self._fake_item("lift/shard-00001.ll", 5),
            self._fake_item("lift/shard-00002.ll", 20),
        ]
        packed = bgw.partition_shards_asymmetric(big, 2)
        self.assertEqual(len(packed), 2)
        self.assertEqual(
            [it.unit.unit_id for it in packed[0]],
            ["lift/shard-00000.ll", "lift/shard-00001.ll"],
        )
        self.assertEqual([it.unit.unit_id for it in packed[1]], ["lift/shard-00002.ll"])
        self.assertLessEqual(
            sum(bgw._partition_weight(it) for it in packed[0]),
            bgw.V8_MAX_WASM_MODULE_BYTES,
        )

    def test_asymmetric_partition_validation_and_edge_cases(self) -> None:
        items = [self._fake_item(f"lift/shard-{i:05d}.ll", 10 * (i + 1)) for i in range(4)]
        self.assertEqual(bgw.partition_shards_asymmetric([], 2), [])
        self.assertEqual(
            [[it.unit.unit_id for it in b] for b in bgw.partition_shards_asymmetric(items, 1)],
            [["lift/shard-00000.ll", "lift/shard-00001.ll", "lift/shard-00002.ll", "lift/shard-00003.ll"]],
        )
        # Fewer shards than buckets: one shard per bucket.
        singleton = bgw.partition_shards_asymmetric(items[:2], 5)
        self.assertEqual(len(singleton), 2)
        self.assertTrue(all(len(b) == 1 for b in singleton))

        self.assertEqual(bgw.parse_max_side_module_mib(994), 994)
        self.assertEqual(bgw.parse_max_side_module_mib(1024), 1024)
        for bad in (0, -1, 1025, True, None, "990"):
            with self.assertRaises(bgw.BuildGuestWasmError, msg=f"max_side_module_mib={bad!r}"):
                bgw.parse_max_side_module_mib(bad)  # type: ignore[arg-type]
        with self.assertRaises(bgw.BuildGuestWasmError):
            bgw.partition_shards_asymmetric(items, 2, max_bucket_bytes=0)
        with self.assertRaises(bgw.BuildGuestWasmError):
            bgw.partition_shards_asymmetric(items, 2, max_bucket_bytes=True)  # type: ignore[arg-type]

    def test_run_build_asymmetric_uses_side_module_2_and_records_sizes(self) -> None:
        class SizedCompileRunner(RecordingRunner):
            def __init__(self, size_by_stem: dict[str, int]) -> None:
                super().__init__()
                self.size_by_stem = size_by_stem

            def __call__(self, argv: Sequence[str]) -> bgw.CommandResult:
                res = super().__call__(argv)
                tokens = tuple(str(a) for a in argv)
                out_path = _dash_o(tokens)
                src_token = _source_token(tokens)
                if (
                    res.returncode == 0
                    and out_path is not None
                    and Path(out_path).suffix.lower() == ".o"
                    and src_token is not None
                ):
                    stem = Path(src_token).stem
                    size = self.size_by_stem.get(stem)
                    if size is not None:
                        Path(out_path).write_bytes(b"\x00" * size)
                return res

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _fake_emsdk(repo)
            _touch_lib(repo / "runtime" / "build-wasm64" / "libplaygta6_core.a")
            _write(repo / "runtime" / "main.cpp", "int main() { return 0; }\n")
            _write(repo / "runtime" / "wasm_boot.cpp", "void wasm_boot() {}\n")

            src = root / "recomp-asym"
            # Object weights: 500+300+194 = 994 for part_0; 100+71 = 171 for part_1.
            shard_sizes = {
                "shard-00000": 500,
                "shard-00001": 300,
                "shard-00002": 194,
                "shard-00003": 100,
                "shard-00004": 71,
            }
            for name in sorted(shard_sizes):
                _write(src / "lift" / f"{name}.ll", f"; {name}\n")
            _write(src / "lift" / "functions.ll", "; registration table\n")
            _write(src / "stubs" / "pg6_import_thunks.cpp", "void thunk() {}\n")

            orig_lib = repo / "runtime" / "build-wasm64" / "libplaygta6_core.a"
            cache = root / "cache-asym"
            out_dir = root / "dist-asym"
            # Object weights are tiny vs the 994 MiB ceiling, so part_0 takes every shard
            # except the one reserved for a non-empty part_1. The exact 994/171 cut is
            # covered by test_asymmetric_partition_targets_994_171_two_part_shape.
            runner = SizedCompileRunner(shard_sizes)
            report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=out_dir,
                jobs=2,
                split_modules=2,
                split_strategy="asymmetric",
                max_side_module_mib=994,
                runtime_lib=orig_lib,
                run_command=runner,
                which_fn=lambda _name: None,
            )
            self.assertTrue(report.ok, report.errors)
            self.assertEqual(report.split_strategy, "asymmetric")
            self.assertEqual(report.side_module_mode, 2)
            self.assertEqual(report.max_side_module_mib, 994)
            self.assertEqual(len(report.side_modules), 3)
            for side in report.side_modules:
                self.assertEqual(side.side_module_mode, 2)
                self.assertGreater(side.estimated_bytes, 0)
                self.assertIn(bgw.SIDE_MODULE_PRUNED_FLAG, side.link_argv)
                self.assertNotIn(bgw.SIDE_MODULE_FLAG, side.link_argv)
            for side_call in runner.side_link_calls:
                self.assertIn(bgw.SIDE_MODULE_PRUNED_FLAG, side_call)
                self.assertNotIn(bgw.SIDE_MODULE_FLAG, side_call)

            # With a 994 MiB ceiling and only ~1 KiB of objects, asymmetric still emits two
            # non-empty contiguous shard buckets (reserves at least one shard for part_1).
            self.assertEqual(len(report.side_modules[0].shards) + len(report.side_modules[1].shards), 5)
            self.assertGreaterEqual(len(report.side_modules[0].shards), 1)
            self.assertGreaterEqual(len(report.side_modules[1].shards), 1)
            self.assertEqual(report.side_modules[2].shards, ("lift/functions.ll",))
            self.assertEqual(
                report.side_modules[0].estimated_bytes + report.side_modules[1].estimated_bytes,
                sum(shard_sizes.values()),
            )

            payload = json.loads(bgw.report_to_json(report))
            self.assertEqual(payload["split_strategy"], "asymmetric")
            self.assertEqual(payload["side_module_mode"], 2)
            self.assertEqual(payload["max_side_module_mib"], 994)
            self.assertEqual(payload["side_modules"][0]["estimated_bytes"], report.side_modules[0].estimated_bytes)

            gm = json.loads((out_dir / bgw.GUEST_MANIFEST_FILE).read_text(encoding="utf-8"))
            self.assertEqual(gm["splitStrategy"], "asymmetric")
            self.assertEqual(gm["sideModuleMode"], 2)
            self.assertEqual(gm["maxSideModuleMiB"], 994)
            self.assertEqual(
                gm["sideModules"],
                ["guest_part_0.wasm", "guest_part_1.wasm", "guest_part_2.wasm"],
            )
            self.assertEqual(len(gm["sideModuleSizes"]), 3)
            self.assertEqual(gm["sideModuleSizes"][0]["estimatedBytes"], report.side_modules[0].estimated_bytes)
            self.assertEqual(gm["sideModuleSizes"][0]["sideModuleMode"], 2)
            self.assertEqual(gm["sideModuleSizes"][0]["wasmBytes"], report.side_modules[0].wasm_bytes)

            text = bgw.format_report_text(report, sanitize=True)
            self.assertIn("strategy=asymmetric, max_side=994MiB", text)
            self.assertIn(bgw.SIDE_MODULE_PRUNED_FLAG, text)

            # Default balanced callers still get SIDE_MODULE=1.
            bal = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist-bal",
                jobs=2,
                split_modules=2,
                split_strategy="balanced",
                runtime_lib=orig_lib,
                run_command=RecordingRunner(),
                which_fn=lambda _name: None,
            )
            self.assertTrue(bal.ok, bal.errors)
            self.assertEqual(bal.side_module_mode, 1)
            self.assertIn(bgw.SIDE_MODULE_FLAG, bal.side_modules[0].link_argv)
            self.assertNotIn(bgw.SIDE_MODULE_PRUNED_FLAG, bal.side_modules[0].link_argv)

            # CLI opt-in + invalid --max-side-module-mib.
            code, out, err = _run_cli(
                [
                    str(src),
                    "--dry-run",
                    "--json",
                    "--split-modules",
                    "2",
                    "--split-strategy",
                    "asymmetric",
                    "--max-side-module-mib",
                    "994",
                    "--repo-root",
                    str(repo),
                    "--cache-dir",
                    str(cache),
                    "--out-dir",
                    str(root / "dist-cli"),
                ]
            )
            self.assertEqual(code, 0, err)
            cli_payload = json.loads(out)
            self.assertTrue(cli_payload["ok"])
            self.assertEqual(cli_payload["split_strategy"], "asymmetric")
            self.assertEqual(cli_payload["side_module_mode"], 2)
            self.assertEqual(cli_payload["max_side_module_mib"], 994)
            self.assertIn(
                bgw.SIDE_MODULE_PRUNED_FLAG,
                cli_payload["side_modules"][0]["link_argv"],
            )

            bad_code, bad_out, _bad_err = _run_cli(
                [
                    str(src),
                    "--dry-run",
                    "--json",
                    "--split-modules",
                    "2",
                    "--split-strategy",
                    "asymmetric",
                    "--max-side-module-mib",
                    "2048",
                    "--repo-root",
                    str(repo),
                ]
            )
            self.assertEqual(bad_code, 2, bad_out)
            bad_payload = json.loads(bad_out)
            self.assertFalse(bad_payload["ok"])
            self.assertTrue(any("max-side-module-mib" in e for e in bad_payload["errors"]))


class TestLinkFlagContract(unittest.TestCase):
    def test_public_constants_match_cmake(self) -> None:
        cmake = (REPO_ROOT / "runtime" / "CMakeLists.txt").read_text(encoding="utf-8")
        for flag in bgw.LINK_FLAGS:
            self.assertIn(flag, cmake, flag)
        for flag in bgw.COMPILE_WASM_FLAGS:
            self.assertIn(flag, cmake, flag)


# ---------------------------------------------------------------------------
# Minimal Wasm binary encoder helpers for R37d pure-Python parser tests
# ---------------------------------------------------------------------------


def _uleb128(value: int) -> bytes:
    assert value >= 0
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _wasm_name(text: str) -> bytes:
    raw = text.encode("utf-8")
    return _uleb128(len(raw)) + raw


def _wasm_section(sec_id: int, payload: bytes) -> bytes:
    return bytes([sec_id]) + _uleb128(len(payload)) + payload


def _encode_dylink0_section(
    *,
    memory_size: int,
    memory_alignment: int = 4,
    table_size: int = 0,
    table_alignment: int = 0,
    needed: Sequence[str] = (),
) -> bytes:
    mem_info = (
        _uleb128(memory_size)
        + _uleb128(memory_alignment)
        + _uleb128(table_size)
        + _uleb128(table_alignment)
    )
    subs = bytearray()
    subs.append(bgw.WASM_DYLINK_MEM_INFO)
    subs.extend(_uleb128(len(mem_info)))
    subs.extend(mem_info)
    if needed:
        needed_body = _uleb128(len(needed)) + b"".join(_wasm_name(n) for n in needed)
        subs.append(2)  # WASM_DYLINK_NEEDED
        subs.extend(_uleb128(len(needed_body)))
        subs.extend(needed_body)
    return _wasm_section(bgw.WASM_SECTION_CUSTOM, _wasm_name("dylink.0") + bytes(subs))


def _encode_import_section(entries: Sequence[tuple[str, str, str]]) -> bytes:
    """Encode a Wasm import section from `(module, field, kind)` tuples.

    Supported kinds: `"func"`, `"got_global"`, `"memory64"`, `"table32"`, `"table64"`, `"tag"`.
    """
    body = bytearray(_uleb128(len(entries)))
    for mod, field, kind in entries:
        body.extend(_wasm_name(mod))
        body.extend(_wasm_name(field))
        if kind == "func":
            body.append(0x00)
            body.extend(_uleb128(0))
        elif kind == "table32":
            body.extend(bytes([0x01, 0x70, 0x00]))
            body.extend(_uleb128(0))
        elif kind == "table64":
            body.extend(bytes([0x01, 0x70, 0x05]))  # has_max | table64
            body.extend(_uleb128(16))
            body.extend(_uleb128(1024))
        elif kind == "memory64":
            body.extend(bytes([0x02, 0x07]))  # has_max | shared | memory64
            body.extend(_uleb128(16))
            body.extend(_uleb128(262144))
        elif kind == "got_global":
            body.extend(bytes([0x03, 0x7F, 0x01]))  # i32 mutable
        elif kind == "tag":
            body.extend(bytes([0x04, 0x00]))
            body.extend(_uleb128(0))
        else:
            raise AssertionError(f"unsupported test import kind {kind!r}")
    return _wasm_section(bgw.WASM_SECTION_IMPORT, bytes(body))


def _encode_table_section(
    tables: Sequence[tuple[int, int, int | None, bool]],
) -> bytes:
    """Encode a Wasm table section from `(reftype, min_sz, max_sz, extended_init)` tuples."""
    body = bytearray(_uleb128(len(tables)))
    for reftype, min_sz, max_sz, extended in tables:
        if extended:
            body.extend(bytes([0x40, 0x00]))
        body.append(reftype)
        flags = 0x01 if max_sz is not None else 0x00
        body.append(flags)
        body.extend(_uleb128(min_sz))
        if max_sz is not None:
            body.extend(_uleb128(max_sz))
        if extended:
            # ref.null func; end
            body.extend(bytes([0xD0, 0x70, 0x0B]))
    return _wasm_section(bgw.WASM_SECTION_TABLE, bytes(body))


def _encode_export_section(exports: Sequence[tuple[str, int, int]]) -> bytes:
    """Encode a Wasm export section from `(name, kind_code, index)` tuples."""
    body = bytearray(_uleb128(len(exports)))
    for name, kind_code, index in exports:
        body.extend(_wasm_name(name))
        body.append(kind_code)
        body.extend(_uleb128(index))
    return _wasm_section(bgw.WASM_SECTION_EXPORT, bytes(body))


def _build_synthetic_wasm(
    *,
    dylink: tuple[int, int] | None = None,
    imports: Sequence[tuple[str, str, str]] = (),
    tables: Sequence[tuple[int, int, int | None, bool]] = (),
    exports: Sequence[tuple[str, int, int]] = (),
    code_padding_bytes: int = 0,
) -> bytes:
    parts = [b"\x00asm\x01\x00\x00\x00"]
    if dylink is not None:
        parts.append(
            _encode_dylink0_section(
                memory_size=dylink[0],
                table_size=dylink[1],
                needed=("runtime.wasm",),
            )
        )
    if imports:
        parts.append(_encode_import_section(imports))
    if tables:
        parts.append(_encode_table_section(tables))
    if exports:
        parts.append(_encode_export_section(exports))
    if code_padding_bytes > 0:
        parts.append(_wasm_section(10, b"\x00" * code_padding_bytes))
    return b"".join(parts)


class TestShardDispatchDetection(unittest.TestCase):
    """R37d: Auto-detection of shard-dispatch (`Pg6ShardDispatchTable`) vs legacy `table` builds."""

    def test_detects_from_program_manifest_options_and_modules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 1) Top-level indirect_dispatch == "shard"
            src1 = _program_layout(root / "case1")
            manifest1 = src1 / "program-manifest.json"
            data1 = json.loads(manifest1.read_text(encoding="utf-8"))
            data1["indirect_dispatch"] = "shard"
            manifest1.write_text(json.dumps(data1), encoding="utf-8")
            det1 = bgw.detect_indirect_dispatch(src1, bgw.discover_sources(src1))
            self.assertEqual(det1, bgw.IndirectDispatchDetection("shard", "program-manifest.json"))

            # 2) Nested options.indirect_dispatch == "shard"
            src2 = _program_layout(root / "case2")
            manifest2 = src2 / "program-manifest.json"
            data2 = json.loads(manifest2.read_text(encoding="utf-8"))
            data2["options"] = {"indirect_dispatch": "shard"}
            manifest2.write_text(json.dumps(data2), encoding="utf-8")
            det2 = bgw.detect_indirect_dispatch(src2, bgw.discover_sources(src2))
            self.assertEqual(det2, bgw.IndirectDispatchDetection("shard", "program-manifest.json"))

            # 3) Module entry shard_dispatch_symbol
            src3 = _program_layout(root / "case3")
            manifest3 = src3 / "program-manifest.json"
            data3 = json.loads(manifest3.read_text(encoding="utf-8"))
            data3["modules"][0]["shard_dispatch_symbol"] = "__playgta6_shard_dispatch_main"
            manifest3.write_text(json.dumps(data3), encoding="utf-8")
            det3 = bgw.detect_indirect_dispatch(src3, bgw.discover_sources(src3))
            self.assertEqual(det3, bgw.IndirectDispatchDetection("shard", "program-manifest.json"))

            # 4) Explicit "table" in program-manifest.json
            src4 = _program_layout(root / "case4")
            manifest4 = src4 / "program-manifest.json"
            data4 = json.loads(manifest4.read_text(encoding="utf-8"))
            data4["indirect_dispatch"] = "table"
            manifest4.write_text(json.dumps(data4), encoding="utf-8")
            det4 = bgw.detect_indirect_dispatch(src4, bgw.discover_sources(src4))
            self.assertEqual(det4, bgw.IndirectDispatchDetection("table", "program-manifest.json"))

    def test_detects_from_build_manifest_pg6_program_functions_ll_and_shard_ir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            # 1) Nested module build-manifest.json with options.indirect_dispatch = "shard"
            src_bm = _program_layout(root / "bm")
            bm_path = src_bm / "main" / "build-manifest.json"
            bm_data = json.loads(bm_path.read_text(encoding="utf-8"))
            bm_data["options"] = {"indirect_dispatch": "shard"}
            bm_path.write_text(json.dumps(bm_data), encoding="utf-8")
            det_bm = bgw.detect_indirect_dispatch(src_bm, bgw.discover_sources(src_bm))
            self.assertEqual(det_bm, bgw.IndirectDispatchDetection("shard", "build-manifest.json"))

            # 2) pg6_program.cpp referencing __playgta6_shard_dispatch_main
            src_cpp = _program_layout(root / "cpp")
            _write(
                src_cpp / "pg6_program.cpp",
                'extern "C" const Pg6ShardDispatchTable __playgta6_shard_dispatch_main;\n',
            )
            det_cpp = bgw.detect_indirect_dispatch(src_cpp, bgw.discover_sources(src_cpp))
            self.assertEqual(det_cpp, bgw.IndirectDispatchDetection("shard", "pg6_program.cpp"))

            # 3) Large (>600 KiB) functions.ll with @__playgta6_shard_dispatch_main only at the tail
            src_fn = _program_layout(root / "fn")
            big_body = "; padding entry line\n" * 35_000
            _write(
                src_fn / "main" / "lift" / "functions.ll",
                big_body
                + "@__playgta6_shard_dispatch_main = constant { i32, i32, i64, ptr, ptr } zeroinitializer\n",
            )
            det_fn = bgw.detect_indirect_dispatch(src_fn, bgw.discover_sources(src_fn))
            self.assertEqual(det_fn, bgw.IndirectDispatchDetection("shard", "functions.ll"))

            # 4) Standalone shard IR without functions.ll or manifests
            src_ir = root / "shard_only"
            _write(
                src_ir / "lift" / "shard-00000.ll",
                "define internal void @sub_140001000(ptr %s) { ret void }\n"
                "define void @__playgta6_dispatch_shard_main_00000(i32 %idx, ptr %s) { ret void }\n",
            )
            det_ir = bgw.detect_indirect_dispatch(src_ir, bgw.discover_sources(src_ir))
            self.assertEqual(det_ir, bgw.IndirectDispatchDetection("shard", "shard-ir"))

            # 5) Default legacy table layout
            src_legacy = _program_layout(root / "legacy")
            det_legacy = bgw.detect_indirect_dispatch(src_legacy, bgw.discover_sources(src_legacy))
            self.assertEqual(det_legacy, bgw.IndirectDispatchDetection("table", "default"))


class TestWasmBinaryTelemetryAndShardBuilds(unittest.TestCase):
    """R37d: Pure-Python Wasm section parser, shard-dispatch split linking, telemetry & guest_manifest.json."""

    def _repo(self, tmp: Path) -> Path:
        repo = tmp / "repo"
        _fake_emsdk(repo)
        _touch_lib(repo / "runtime" / "build-wasm64" / "libplaygta6_core.a")
        _write(repo / "runtime" / "main.cpp", "int main() { return 0; }\n")
        _write(repo / "runtime" / "wasm_boot.cpp", "void run_wasm_guest_boot() {}\n")
        return repo

    def test_pure_python_wasm_parser_handles_imports_exports_tables_and_dylink0(self) -> None:
        # 1) Main Wasm64 module with memory64 import, extended 0x40 table + standard table, and exports.
        main_wasm = _build_synthetic_wasm(
            imports=[
                ("env", "memory", "memory64"),
                ("env", "host_log", "func"),
                ("env", "host_httpfs_open", "func"),
                ("GOT.mem", "__playgta6_shard_dispatch_main", "got_global"),
            ],
            tables=[
                (0x70, 48, 1024, False),
                (0x70, 16, None, True),  # extended 0x40 0x00 table with ref.null init expr
                (0x6F, 99, None, False),  # externref table must NOT count as funcref tableSize
            ],
            exports=[
                ("main", 0x00, 0),
                ("__playgta6_indirect_call", 0x00, 1),
                ("memory", 0x02, 0),
            ],
            code_padding_bytes=4096,
        )
        m_main = bgw.inspect_wasm_bytes(main_wasm, name="runtime.wasm", role="main")
        self.assertEqual(m_main.name, "runtime.wasm")
        self.assertEqual(m_main.role, "main")
        self.assertEqual(m_main.wasm_bytes, len(main_wasm))
        self.assertEqual(m_main.func_imports, 2)
        self.assertEqual(m_main.got_func_imports, 0)
        self.assertEqual(m_main.got_mem_imports, 1)
        self.assertEqual(m_main.func_exports, 2)
        self.assertEqual(m_main.table_size, 64)
        self.assertEqual(m_main.defined_table_size, 64)
        self.assertFalse(m_main.has_dylink)
        self.assertIsNone(m_main.dylink_table_size)

        # 2) Shard side module with dylink.0 (tableSize=2), 2 exported dispatchers, 1 cross-shard sub_* export.
        shard_side = _build_synthetic_wasm(
            dylink=(2048, 2),
            imports=[
                ("env", "memory", "memory64"),
                ("env", "__indirect_function_table", "table32"),
                ("env", "__playgta6_indirect_call", "func"),
                ("env", "sub_140009000", "func"),
            ],
            exports=[
                ("__playgta6_dispatch_shard_main_00000", 0x00, 0),
                ("__playgta6_dispatch_shard_main_00001", 0x00, 1),
                ("sub_140001000", 0x00, 2),
            ],
            code_padding_bytes=8192,
        )
        m_side = bgw.inspect_wasm_bytes(shard_side, name="guest_part_0.wasm", role="side")
        self.assertTrue(m_side.has_dylink)
        self.assertEqual(m_side.dylink_memory_size, 2048)
        self.assertEqual(m_side.dylink_table_size, 2)
        self.assertEqual(m_side.table_size, 2)
        self.assertEqual(m_side.func_imports, 2)
        self.assertEqual(m_side.sub_func_imports, 1)
        self.assertEqual(m_side.got_func_imports, 0)
        self.assertEqual(m_side.func_exports, 3)
        self.assertEqual(m_side.shard_dispatch_exports, 2)
        self.assertEqual(m_side.sub_func_exports, 1)

        # 3) Registration side module (functions.o in shard mode): 4 GOT.func dispatchers, 0 sub_* imports.
        reg_side = _build_synthetic_wasm(
            dylink=(65536, 0),
            imports=[
                ("GOT.func", "__playgta6_dispatch_shard_main_00000", "got_global"),
                ("GOT.func", "__playgta6_dispatch_shard_main_00001", "got_global"),
                ("GOT.func", "__playgta6_dispatch_shard_main_00002", "got_global"),
                ("GOT.func", "__playgta6_dispatch_shard_main_00003", "got_global"),
            ],
            exports=[
                ("__playgta6_shard_dispatch_main", 0x03, 0),
            ],
        )
        m_reg = bgw.inspect_wasm_bytes(reg_side, name="guest_part_2.wasm", role="side")
        self.assertEqual(m_reg.got_func_imports, 4)
        self.assertEqual(m_reg.got_func_shard_dispatch_imports, 4)
        self.assertEqual(m_reg.got_func_sub_imports, 0)
        self.assertEqual(m_reg.shard_table_exports, 1)
        self.assertEqual(m_reg.table_size, 0)

        # 4) Legacy "dylink" custom section (without .0) and malformed Wasm error handling.
        legacy_dylink_payload = _wasm_name("dylink") + _uleb128(512) + _uleb128(4) + _uleb128(7) + _uleb128(0)
        legacy_dylink_wasm = b"\x00asm\x01\x00\x00\x00" + _wasm_section(0, legacy_dylink_payload)
        m_leg = bgw.inspect_wasm_bytes(legacy_dylink_wasm, name="legacy.wasm", role="side")
        self.assertTrue(m_leg.has_dylink)
        self.assertEqual(m_leg.dylink_memory_size, 512)
        self.assertEqual(m_leg.dylink_table_size, 7)
        self.assertEqual(m_leg.table_size, 7)

        for bad in (b"", b"\x00asm", b"notwasm!", b"\x00asm\x02\x00\x00\x00", b"\x00asm\x01\x00\x00\x00\x02\x10"):
            with self.assertRaises(bgw.WasmInspectError):
                bgw.inspect_wasm_bytes(bad, name="bad.wasm")

    def test_shard_dispatch_split_build_telemetry_manifest_and_max_table_imports(self) -> None:
        wasm_by_name = {
            "runtime.wasm": _build_synthetic_wasm(
                imports=[
                    ("env", "memory", "memory64"),
                    ("env", "host_log", "func"),
                    ("GOT.mem", "__playgta6_shard_dispatch_main", "got_global"),
                ],
                tables=[(0x70, 64, None, False)],
                exports=[("main", 0x00, 0), ("__playgta6_indirect_call", 0x00, 1)],
                code_padding_bytes=2048,
            ),
            "guest_part_0.wasm": _build_synthetic_wasm(
                dylink=(4096, 2),
                imports=[("env", "__playgta6_indirect_call", "func")],
                exports=[
                    ("__playgta6_dispatch_shard_main_00000", 0x00, 0),
                    ("__playgta6_dispatch_shard_main_00001", 0x00, 1),
                ],
                code_padding_bytes=4096,
            ),
            "guest_part_1.wasm": _build_synthetic_wasm(
                dylink=(4096, 2),
                imports=[("env", "__playgta6_indirect_call", "func")],
                exports=[
                    ("__playgta6_dispatch_shard_main_00002", 0x00, 0),
                    ("__playgta6_dispatch_shard_main_00003", 0x00, 1),
                ],
                code_padding_bytes=4096,
            ),
            "guest_part_2.wasm": _build_synthetic_wasm(
                dylink=(16384, 0),
                imports=[
                    ("GOT.func", "__playgta6_dispatch_shard_main_00000", "got_global"),
                    ("GOT.func", "__playgta6_dispatch_shard_main_00001", "got_global"),
                    ("GOT.func", "__playgta6_dispatch_shard_main_00002", "got_global"),
                    ("GOT.func", "__playgta6_dispatch_shard_main_00003", "got_global"),
                ],
                exports=[("__playgta6_shard_dispatch_main", 0x03, 0)],
            ),
        }

        class SyntheticWasmRunner(RecordingRunner):
            def __call__(self, argv: Sequence[str]) -> bgw.CommandResult:
                res = super().__call__(argv)
                tokens = tuple(str(a) for a in argv)
                out_path = _dash_o(tokens)
                if res.returncode == 0 and out_path is not None:
                    p = Path(out_path)
                    if p.name in wasm_by_name:
                        p.write_bytes(wasm_by_name[p.name])
                    elif p.name == bgw.RUNTIME_JS:
                        p.with_suffix(".wasm").write_bytes(wasm_by_name["runtime.wasm"])
                return res

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = root / "shard-program"
            for idx in range(4):
                _write(
                    src / "main" / "lift" / f"shard-{idx:05d}.ll",
                    f"define internal void @sub_{0x140001000 + idx * 0x100:x}(ptr %s) {{ ret void }}\n"
                    f"define void @__playgta6_dispatch_shard_main_{idx:05d}(i32 %i, ptr %s) {{ ret void }}\n",
                )
            _write(
                src / "main" / "lift" / "functions.ll",
                "declare void @__playgta6_dispatch_shard_main_00000(i32, ptr)\n"
                "@__playgta6_shard_dispatch_main = constant { i32, i32, i64, ptr, ptr } zeroinitializer\n",
            )
            _write(src / "main" / "stubs" / "pg6_import_thunks.cpp", "void thunk_main() {}\n")
            _write(src / "exports" / "pg6_export_thunks.cpp", "void pg6_export_thunks() {}\n")
            _write(
                src / "pg6_program.cpp",
                'extern "C" const Pg6ShardDispatchTable __playgta6_shard_dispatch_main;\n',
            )
            (src / "program-manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "playgta6-recomp-program",
                        "schema_version": 1,
                        "indirect_dispatch": "shard",
                        "modules": [{"module": "main", "file": "game.exe"}],
                        "files": [
                            {"path": "exports/pg6_export_thunks.cpp"},
                            {"path": "pg6_program.cpp"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            out_dir = root / "dist"
            # Pre-seed an existing field in guest_manifest.json to verify backwards-compatible merge.
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / bgw.GUEST_MANIFEST_FILE).write_text(
                json.dumps({"customField": "preserved"}),
                encoding="utf-8",
            )

            orig_lib = repo / "runtime" / "build-wasm64" / "libplaygta6_core.a"
            runner = SyntheticWasmRunner()
            report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache",
                out_dir=out_dir,
                jobs=2,
                split_modules=2,
                split_strategy="contiguous",
                verify_max_table_imports=16,
                runtime_lib=orig_lib,
                run_command=runner,
                which_fn=lambda _name: None,
            )
            self.assertTrue(report.ok, report.errors)
            self.assertEqual(report.indirect_dispatch, "shard")
            self.assertEqual(report.indirect_dispatch_source, "program-manifest.json")
            self.assertEqual(report.guest_manifest, bgw.GUEST_MANIFEST_FILE)
            self.assertEqual(
                report.outputs,
                (
                    bgw.RUNTIME_JS,
                    bgw.RUNTIME_WASM,
                    "guest_part_0.wasm",
                    "guest_part_1.wasm",
                    "guest_part_2.wasm",
                ),
            )
            # Bucketing + side module roles: 2 contiguous shard modules + functions.ll last.
            self.assertEqual([m.role for m in report.side_modules], ["shards", "shards", "registration"])
            self.assertEqual(report.side_modules[2].shards, ("main/lift/functions.ll",))
            # Side-module link commands never force --export-all or unsupported --icf=safe.
            for side_cmd in runner.side_link_calls:
                self.assertIn("-sSIDE_MODULE=1", side_cmd)
                self.assertNotIn("--export-all", " ".join(side_cmd))
                self.assertNotIn("--icf=safe", " ".join(side_cmd))

            # Post-link Wasm metrics assertions.
            assert report.wasm_metrics is not None
            self.assertEqual(report.wasm_metrics.total_got_func_imports, 4)
            self.assertEqual(report.wasm_metrics.got_func_shard_dispatch_imports, 4)
            self.assertEqual(report.wasm_metrics.got_func_sub_imports, 0)
            self.assertEqual(report.wasm_metrics.side_table_size, 4)
            self.assertEqual(report.wasm_metrics.main_table_size, 64)
            self.assertEqual(report.wasm_metrics.total_table_size, 68)
            self.assertEqual(report.wasm_metrics.shard_dispatch_exports, 4)
            self.assertEqual(report.wasm_metrics.shard_table_exports, 1)
            self.assertEqual(report.warnings, [])

            # guest_manifest.json contains backwards-compatible sideModules + numeric wasmMetrics.
            gm_path = out_dir / bgw.GUEST_MANIFEST_FILE
            self.assertTrue(gm_path.is_file())
            gm = json.loads(gm_path.read_text(encoding="utf-8"))
            self.assertEqual(gm["customField"], "preserved")
            self.assertEqual(gm["indirectDispatch"], "shard")
            self.assertEqual(
                gm["sideModules"],
                ["guest_part_0.wasm", "guest_part_1.wasm", "guest_part_2.wasm"],
            )
            self.assertEqual(gm["wasmMetrics"]["totals"]["gotFuncImports"], 4)
            self.assertEqual(gm["wasmMetrics"]["totals"]["sideTableSize"], 4)
            self.assertEqual(gm["wasmMetrics"]["totals"]["mainTableSize"], 64)
            self.assertEqual(gm["wasmMetrics"]["totals"]["tableSize"], 68)
            self.assertEqual(gm["wasmMetrics"]["totals"]["shardDispatchExports"], 4)
            self.assertEqual(len(gm["wasmMetrics"]["modules"]), 4)
            # Zero host paths in guest_manifest.json or JSON report.
            self.assertNotIn("\\\\", gm_path.read_text(encoding="utf-8"))
            self.assertNotIn("\\\\", bgw.report_to_json(report))

            # Text report prints telemetry table + explicit [shard-dispatch] banner.
            text = bgw.format_report_text(report, sanitize=True)
            self.assertIn("Indirect dispatch: shard (detected from program-manifest.json)", text)
            self.assertIn("Wasm module telemetry:", text)
            self.assertIn("guest_part_2.wasm", text)
            self.assertIn("TOTAL", text)
            self.assertIn("[shard-dispatch] Active:", text)
            self.assertIn("437,813", text)
            self.assertIn("Guest manifest: guest_manifest.json", text)

            # Enforcing a tight --verify-max-table-imports limit (e.g. 2 < 4) fails cleanly.
            fail_rep = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache",
                out_dir=root / "dist-tight",
                jobs=2,
                split_modules=2,
                split_strategy="contiguous",
                verify_max_table_imports=2,
                runtime_lib=orig_lib,
                run_command=SyntheticWasmRunner(),
                which_fn=lambda _name: None,
            )
            self.assertFalse(fail_rep.ok)
            self.assertTrue(
                any("GOT.func imports (4) exceeds --verify-max-table-imports limit (2)" in e for e in fail_rep.errors),
                fail_rep.errors,
            )

    def test_legacy_table_build_telemetry_limits_and_shard_sub_warning(self) -> None:
        # Legacy table shape: functions.o (guest_part_1.wasm) imports 250 GOT.func.sub_* entries.
        legacy_wasms = {
            "runtime.wasm": _build_synthetic_wasm(
                imports=[("env", "host_log", "func")],
                tables=[(0x70, 32, None, False)],
                exports=[("main", 0x00, 0)],
            ),
            "guest_part_0.wasm": _build_synthetic_wasm(
                dylink=(4096, 250),
                exports=[(f"sub_{i:04x}", 0x00, i) for i in range(250)],
            ),
            "guest_part_1.wasm": _build_synthetic_wasm(
                dylink=(8192, 0),
                imports=[("GOT.func", f"sub_{i:04x}", "got_global") for i in range(250)],
                exports=[("__playgta6_lifted_functions", 0x03, 0)],
            ),
        }

        class LegacyWasmRunner(RecordingRunner):
            def __call__(self, argv: Sequence[str]) -> bgw.CommandResult:
                res = super().__call__(argv)
                out_path = _dash_o(tuple(str(a) for a in argv))
                if res.returncode == 0 and out_path is not None:
                    p = Path(out_path)
                    if p.name in legacy_wasms:
                        p.write_bytes(legacy_wasms[p.name])
                    elif p.name == bgw.RUNTIME_JS:
                        p.with_suffix(".wasm").write_bytes(legacy_wasms["runtime.wasm"])
                return res

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _module_layout(root, "legacy_mod", shards=2)
            orig_lib = repo / "runtime" / "build-wasm64" / "libplaygta6_core.a"

            # 1) Legacy table build succeeds normally without [shard-dispatch] banner.
            dist_ok = root / "dist-ok"
            rep_ok = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache",
                out_dir=dist_ok,
                jobs=2,
                split_modules=1,
                runtime_lib=orig_lib,
                run_command=LegacyWasmRunner(),
                which_fn=lambda _name: None,
            )
            self.assertTrue(rep_ok.ok, rep_ok.errors)
            self.assertEqual(rep_ok.indirect_dispatch, "table")
            self.assertNotIn("[shard-dispatch]", bgw.format_report_text(rep_ok))
            gm_mono = json.loads((dist_ok / bgw.GUEST_MANIFEST_FILE).read_text(encoding="utf-8"))
            self.assertEqual(gm_mono["indirectDispatch"], "table")
            self.assertEqual(gm_mono["sideModules"], [])
            self.assertEqual(gm_mono["wasmMetrics"]["totals"]["tableSize"], 32)

            # 2) Legacy split build fails --verify-max-table-imports 100.
            rep_fail = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache",
                out_dir=root / "dist-fail",
                jobs=2,
                split_modules=1,
                verify_max_table_imports=10,
                runtime_lib=orig_lib,
                run_command=LegacyWasmRunner(),
                which_fn=lambda _name: None,
            )
            self.assertFalse(rep_fail.ok)
            self.assertTrue(
                any("total wasmTable size (32) exceeds --verify-max-table-imports limit (10)" in e for e in rep_fail.errors),
                rep_fail.errors,
            )

            # 3) Shard manifest paired with stale GOT.func.sub_* emits warning.
            bm_path = src / "build-manifest.json"
            bm_data = json.loads(bm_path.read_text(encoding="utf-8"))
            bm_data["indirect_dispatch"] = "shard"
            bm_path.write_text(json.dumps(bm_data), encoding="utf-8")
            rep_warn = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache-warn",
                out_dir=root / "dist-warn",
                jobs=2,
                split_modules=2,
                runtime_lib=orig_lib,
                run_command=LegacyWasmRunner(),
                which_fn=lambda _name: None,
            )
            self.assertTrue(rep_warn.ok, rep_warn.errors)
            self.assertTrue(
                any("GOT.func sub_* import(s)" in w for w in rep_warn.warnings),
                rep_warn.warnings,
            )

            # 4) CLI rejects negative --verify-max-table-imports cleanly with exit code 2.
            bad_code, bad_out, _ = _run_cli(
                [
                    str(src),
                    "--dry-run",
                    "--json",
                    "--verify-max-table-imports",
                    "-5",
                    "--repo-root",
                    str(repo),
                ]
            )
            self.assertEqual(bad_code, 2, bad_out)
            bad_payload = json.loads(bad_out)
            self.assertFalse(bad_payload["ok"])
            self.assertTrue(any("verify-max-table-imports" in e for e in bad_payload["errors"]))

    def test_split_js_optimizer_preserves_shard_dispatch_symbols_in_got_and_wasm_imports(self) -> None:
        fixture = r"""
var LDSO = { loadedLibsByName: { 'guest_part_0.wasm': { exports: { __playgta6_dispatch_shard_main_00000: () => 1 } } } };
var wasmImports = {
  sub_140001000: () => 10,
  __playgta6_dispatch_shard_main_00000: () => 20,
  __playgta6_shard_dispatch_main: 4096,
};
var GOT = {
  sub_140001000: { value: 1 },
  __playgta6_dispatch_shard_main_00000: { value: 2 },
  __playgta6_shard_dispatch_main: { value: 4096 },
};
var functionsInTableMap = new Map();
function instantiate() {
  var moduleExports;
  function resolve(s) { return wasmImports[s]; }
  var proxy = new Proxy({}, {
    get(stubs, prop) {
      if (!(prop in stubs)) {
        var resolved;
        stubs[prop] = (...args) => {
          resolved ||= resolve(prop);
          return resolved(...args);
        };
      }
      return stubs[prop];
    }
  });
  moduleExports = {};
  return proxy;
}
async function loadDylibs() { instantiate().sub_140001000; }
async function boot() { await loadDylibs(); }
"""
        patched, status = bgw.optimize_split_runtime_js(fixture, max_concurrent_pthread_loads=0)
        self.assertTrue(status["post_load_dylibs_cleanup"])
        node = shutil.which("node")
        if node is None:
            self.skipTest("node not installed")
        proc = subprocess.run(
            [
                node,
                "-e",
                patched
                + "\n(async()=>{await boot();console.log(JSON.stringify({wi:Object.keys(wasmImports).sort(),got:Object.keys(GOT).sort()}));})();",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        res = json.loads(proc.stdout.strip())
        self.assertEqual(
            res["wi"],
            ["__playgta6_dispatch_shard_main_00000", "__playgta6_shard_dispatch_main"],
        )
        self.assertEqual(
            res["got"],
            ["__playgta6_dispatch_shard_main_00000", "__playgta6_shard_dispatch_main"],
        )


class TestSideModule2CrossExports(unittest.TestCase):
    """R37f: llvm-nm cross-module exports for -sSIDE_MODULE=2 and empty side-module guard."""

    def test_compute_side_module_cross_exports_prunes_private_and_keeps_cross_main_and_roots(
        self,
    ) -> None:
        # Three side buckets (part_0 shards, part_1 shards, part_2 registration) + main objects.
        nm_by_obj = {
            "shard-00000.o": "\n".join(
                [
                    "shard-00000.o:",
                    "00000000 T sub_140001000",  # called from part_1 -> MUST export
                    "00000020 T sub_140001010",  # intra-bucket helper called by shard-00001.o -> PRUNE
                    "00000040 T sub_140001020",  # private only in part_0 -> PRUNE
                    "00000060 t sub_local_only",  # local (lowercase t) -> NEVER export
                    "         T sub_main_entry",  # referenced by main.o -> MUST export
                    "         T __playgta6_dispatch_shard_main_00000",  # referenced by functions.o + required
                    "00000080 W pg6_entry_bootstrap",  # weak global matching pg6_entry_* -> MUST export
                    "000000a0 R guest_const_in_p0",  # R data referenced by part_1 -> MUST export
                    "000000c0 D private_data_in_p0",  # unreferenced data -> PRUNE
                    "         U sub_140009000",  # cross-call into part_1
                    "         U __playgta6_indirect_call",  # runtime import (not in any side bucket)
                ]
            ),
            "shard-00001.o": "\n".join(
                [
                    "shard-00001.o:",
                    "         T sub_140001050",  # unreferenced outside part_0 -> PRUNE
                    "         U sub_140001010",  # intra-bucket call within part_0 -> must NOT export!
                    "         U weak_cross_helper",  # weak def in part_1 -> MUST export from part_1
                ]
            ),
            "shard-00002.o": "\n".join(
                [
                    "shard-00002.o:",
                    "00000000 T sub_140009000",  # called from part_0 -> MUST export
                    "00000010 T sub_140009010",  # private in part_1 -> PRUNE
                    "00000020 W weak_cross_helper",  # weak def called from part_0 -> MUST export
                    "00000030 B bss_used_by_main",  # BSS symbol referenced by pg6_export_thunks -> MUST export
                    "         T pg6_dispatch_custom",  # required prefix -> MUST export
                    "         T __playgta6_dispatch_shard_main_00002",  # referenced by functions.o + required
                    "         U sub_140001000",  # cross-call into part_0
                    "         U guest_const_in_p0",  # cross-data ref into part_0
                ]
            ),
            "functions.o": "\n".join(
                [
                    "functions.o:",
                    "00000000 D __playgta6_shard_dispatch_main",  # referenced by pg6_program.o + required
                    "00000010 T __playgta6_register_lifted_functions_main",  # required registration root
                    "00000020 T pg6_register_custom_tables",  # required prefix
                    "00000030 D internal_scratch_table",  # unreferenced outside part_2 -> PRUNE
                    "         U __playgta6_dispatch_shard_main_00000",
                    "         U __playgta6_dispatch_shard_main_00002",
                    "         U sub_140009020",  # defined in part_1? (if not defined anywhere, ignored)
                ]
            ),
            "pg6_program.o": "\n".join(
                [
                    "pg6_program.o:",
                    "00000000 D pg6_program",
                    "         U __playgta6_shard_dispatch_main",
                ]
            ),
            "pg6_export_thunks.o": "\n".join(
                [
                    "pg6_export_thunks.o:",
                    "         U bss_used_by_main",
                ]
            ),
            "main.o": "\n".join(
                [
                    "main.o:",
                    "00000000 T main",
                    "         U sub_main_entry",
                ]
            ),
        }

        nm_invocations: list[tuple[str, ...]] = []

        def fake_nm(argv: Sequence[str]) -> bgw.CommandResult:
            tokens = tuple(str(a) for a in argv)
            nm_invocations.append(tokens)
            chunks = [nm_by_obj[Path(tok).name] for tok in tokens[1:] if not tok.startswith("-")]
            return bgw.CommandResult(returncode=0, stdout="\n".join(chunks) + "\n")

        side_buckets = [
            [Path("cache/shard-00000.o"), Path("cache/shard-00001.o")],
            [Path("cache/shard-00002.o")],
            [Path("cache/functions.o")],
        ]
        main_objects = [
            Path("cache/pg6_program.o"),
            Path("cache/pg6_export_thunks.o"),
            Path("cache/main.o"),
        ]

        exports = bgw.compute_side_module_cross_exports(
            side_buckets,
            main_objects,
            Path("/emsdk/upstream/bin/llvm-nm"),
            run_command=fake_nm,
        )

        self.assertEqual(len(exports), 3)
        self.assertEqual(len(nm_invocations), 4)  # 3 side buckets + 1 main batch
        for call in nm_invocations:
            self.assertIn("--extern-only", call)
            self.assertIn("--no-sort", call)

        # Bucket 0: cross-bucket + main-referenced + required roots kept; intra-bucket pruned.
        self.assertEqual(
            exports[0],
            [
                "__playgta6_dispatch_shard_main_00000",
                "guest_const_in_p0",
                "pg6_entry_bootstrap",
                "sub_140001000",
                "sub_main_entry",
            ],
        )
        self.assertNotIn("sub_140001010", exports[0])  # intra-bucket call between shard-0 and shard-1
        self.assertNotIn("sub_140001020", exports[0])
        self.assertNotIn("sub_140001050", exports[0])
        self.assertNotIn("sub_local_only", exports[0])
        self.assertNotIn("private_data_in_p0", exports[0])

        # Bucket 1: cross-bucket + main-referenced + required roots kept; private sub_* pruned.
        self.assertEqual(
            exports[1],
            [
                "__playgta6_dispatch_shard_main_00002",
                "bss_used_by_main",
                "pg6_dispatch_custom",
                "sub_140009000",
                "weak_cross_helper",
            ],
        )
        self.assertNotIn("sub_140009010", exports[1])

        # Bucket 2 (functions.o): registration symbols kept; unreferenced data pruned.
        self.assertEqual(
            exports[2],
            [
                "__playgta6_register_lifted_functions_main",
                "__playgta6_shard_dispatch_main",
                "pg6_register_custom_tables",
            ],
        )
        self.assertNotIn("internal_scratch_table", exports[2])

        # Response file emission uses -Wl,--export=<sym> (no leading-underscore mangling) + ctors.
        with tempfile.TemporaryDirectory() as tmp:
            rsp_path = Path(tmp) / "side_part_0_exports.rsp"
            bgw.write_side_exports_response_file(rsp_path, exports[0])
            lines = rsp_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], bgw.SIDE_MODULE_CTOR_EXPORT_FLAG)
            self.assertIn("-Wl,--export=sub_140001000", lines)
            self.assertIn("-Wl,--export=sub_main_entry", lines)
            self.assertIn("-Wl,--export=__playgta6_dispatch_shard_main_00000", lines)
            self.assertNotIn("-Wl,--export=sub_140001010", lines)
            self.assertNotIn("-sEXPORTED_FUNCTIONS", "\n".join(lines))

    def test_parse_llvm_nm_formats_chunking_and_error_handling(self) -> None:
        # Both BSD (hex addr, dashes, bare wasm func/undef) and POSIX formats parse cleanly.
        mixed = "\n".join(
            [
                "C:\\cache\\shard.o:",
                "0000000000000000 T bsd_hex_func",
                "---------------- D bsd_dash_data",
                "         T bsd_bare_wasm_func",
                "         U bsd_bare_undef",
                "         w bsd_weak_undef",
                "00000010 t local_ignored",
                "posix_func T 0 16",
                "posix_undef U",
            ]
        )
        defined, undefined = bgw.parse_llvm_nm_output(mixed)
        self.assertEqual(
            defined,
            {"bsd_hex_func", "bsd_dash_data", "bsd_bare_wasm_func", "posix_func"},
        )
        self.assertEqual(undefined, {"bsd_bare_undef", "bsd_weak_undef", "posix_undef"})

        # Batch chunking splits large object lists below max_cmd_chars.
        chunk_calls: list[tuple[str, ...]] = []

        def chunk_runner(argv: Sequence[str]) -> bgw.CommandResult:
            chunk_calls.append(tuple(str(a) for a in argv))
            return bgw.CommandResult(returncode=0, stdout="")

        many_objs = [Path(f"very_long_cache_directory/shard-{i:05d}.o") for i in range(6)]
        bgw.compute_side_module_cross_exports(
            [many_objs],
            [],
            Path("llvm-nm"),
            run_command=chunk_runner,
            max_cmd_chars=90,
        )
        self.assertGreaterEqual(len(chunk_calls), 3)

        # Non-zero llvm-nm exit raises BuildGuestWasmError.
        with self.assertRaises(bgw.BuildGuestWasmError):
            bgw.compute_side_module_cross_exports(
                [[Path("a.o")]],
                [],
                Path("llvm-nm"),
                run_command=lambda _argv: bgw.CommandResult(returncode=1, stderr="corrupt object"),
            )

    def test_discover_emscripten_locates_llvm_nm_alongside_emxx_and_on_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _fake_emsdk(repo)
            tc = bgw.discover_emscripten(repo_root=repo, env={}, which_fn=lambda _n: None)
            self.assertIsNotNone(tc.nm())
            assert tc.llvm_nm is not None
            self.assertEqual(tc.llvm_nm.parent.name, "bin")
            self.assertTrue(tc.llvm_nm.name.startswith("llvm-nm"))
            self.assertTrue(tc.to_dict()["llvm_nm_found"])

            # Fallback to PATH when upstream/bin/llvm-nm is absent.
            for f in (repo / "work" / "emsdk" / "upstream" / "bin").iterdir():
                f.unlink()
            fake_path_nm = root / "custom-llvm" / "llvm-nm"
            _write(fake_path_nm, "#!/bin/sh\n")
            tc_path = bgw.discover_emscripten(
                repo_root=repo,
                env={},
                which_fn=lambda n: str(fake_path_nm) if n == "llvm-nm" else None,
            )
            self.assertEqual(tc_path.nm(), fake_path_nm)

            # Sanitize strips host path from llvm-nm argv.
            cleaned = bgw.sanitize_argv(
                [r"C:\Users\someone\emsdk\upstream\bin\llvm-nm.exe", "--extern-only", r"C:\Users\someone\a.o"]
            )
            self.assertEqual(cleaned, ["llvm-nm", "--extern-only", "a.o"])

    def test_asymmetric_dry_run_and_mocked_build_pass_cross_export_rsp_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _fake_emsdk(repo)
            orig_lib = _touch_lib(repo / "runtime" / "build-wasm64" / "libplaygta6_core.a")
            _write(repo / "runtime" / "main.cpp", "int main() { return 0; }\n")
            _write(repo / "runtime" / "wasm_boot.cpp", "void wasm_boot() {}\n")

            src = root / "recomp-r37f"
            _write(src / "lift" / "shard-00000.ll", "; shard 0\n")
            _write(src / "lift" / "shard-00001.ll", "; shard 1\n")
            _write(src / "lift" / "shard-00002.ll", "; shard 2\n")
            _write(src / "lift" / "functions.ll", "; registration\n")
            _write(src / "stubs" / "pg6_import_thunks.cpp", "void thunk() {}\n")

            cache = root / "cache"
            out_dir = root / "dist"

            # 1) Dry-run command lines already include -sSIDE_MODULE=2, ctor flag, and @side_part_<i>_exports.rsp.
            dry = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=out_dir,
                jobs=2,
                split_modules=2,
                split_strategy="asymmetric",
                dry_run=True,
                runtime_lib=orig_lib,
                which_fn=lambda _n: None,
            )
            self.assertTrue(dry.ok, dry.errors)
            self.assertEqual(len(dry.side_modules), 3)
            for idx, side in enumerate(dry.side_modules):
                expected_rsp = f"side_part_{idx}_exports.rsp"
                self.assertEqual(side.exports_response_file, expected_rsp)
                self.assertIn(bgw.SIDE_MODULE_PRUNED_FLAG, side.link_argv)
                self.assertIn(bgw.SIDE_MODULE_CTOR_EXPORT_FLAG, side.link_argv)
                self.assertIn(f"@{expected_rsp}", side.link_argv)
            dry_json = bgw.report_to_json(dry)
            self.assertNotIn("\\\\", dry_json)
            self.assertIn("@side_part_0_exports.rsp", dry_json)

            # 2) Mocked real build computes cross-module exports via llvm-nm and writes response files.
            nm_outputs = {
                "shard-00000": "\n".join(
                    [
                        "00000000 T sub_cross_0_to_1",
                        "00000010 T sub_private_in_p0",
                        "         T __playgta6_dispatch_shard_main_00000",
                        "         U sub_cross_1_to_0",
                    ]
                ),
                "shard-00001": "\n".join(
                    [
                        "00000000 T sub_private_in_p0_b",
                        "         U sub_private_in_p0",  # same bucket as shard-00000 -> pruned!
                    ]
                ),
                "shard-00002": "\n".join(
                    [
                        "00000000 T sub_cross_1_to_0",
                        "00000010 T sub_private_in_p1",
                        "         T sub_from_main_thunk",
                        "         U sub_cross_0_to_1",
                    ]
                ),
                "functions": "\n".join(
                    [
                        "00000000 D __playgta6_shard_dispatch_main",
                        "         U __playgta6_dispatch_shard_main_00000",
                    ]
                ),
                "pg6_import_thunks": "\n".join(
                    [
                        "         U sub_from_main_thunk",
                        "         U __playgta6_shard_dispatch_main",
                    ]
                ),
            }
            runner = RecordingRunner(nm_outputs=nm_outputs)
            rep = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=out_dir,
                jobs=2,
                split_modules=2,
                split_strategy="asymmetric",
                runtime_lib=orig_lib,
                run_command=runner,
                which_fn=lambda _n: None,
            )
            self.assertTrue(rep.ok, rep.errors)
            self.assertEqual(len(runner.nm_calls), 4)  # 3 side buckets + main_objects
            self.assertEqual(len(runner.side_link_calls), 3)

            # Check actual side_part_0_exports.rsp written on disk and passed to em++.
            rsp0_path = cache / "side_part_0_exports.rsp"
            rsp1_path = cache / "side_part_1_exports.rsp"
            rsp2_path = cache / "side_part_2_exports.rsp"
            self.assertTrue(rsp0_path.is_file())
            self.assertTrue(rsp1_path.is_file())
            self.assertTrue(rsp2_path.is_file())

            rsp0_lines = rsp0_path.read_text(encoding="utf-8").splitlines()
            self.assertIn(bgw.SIDE_MODULE_CTOR_EXPORT_FLAG, rsp0_lines)
            self.assertIn("-Wl,--export=sub_cross_0_to_1", rsp0_lines)
            self.assertIn("-Wl,--export=__playgta6_dispatch_shard_main_00000", rsp0_lines)
            self.assertNotIn("-Wl,--export=sub_private_in_p0", rsp0_lines)
            self.assertNotIn("-Wl,--export=sub_private_in_p0_b", rsp0_lines)

            rsp1_lines = rsp1_path.read_text(encoding="utf-8").splitlines()
            self.assertIn("-Wl,--export=sub_cross_1_to_0", rsp1_lines)
            self.assertIn("-Wl,--export=sub_from_main_thunk", rsp1_lines)
            self.assertNotIn("-Wl,--export=sub_private_in_p1", rsp1_lines)

            rsp2_lines = rsp2_path.read_text(encoding="utf-8").splitlines()
            self.assertIn("-Wl,--export=__playgta6_shard_dispatch_main", rsp2_lines)

            for idx, side_call in enumerate(runner.side_link_calls):
                expected_arg = f"@{(cache / f'side_part_{idx}_exports.rsp').as_posix()}"
                self.assertIn(expected_arg, side_call)
                self.assertIn(bgw.SIDE_MODULE_PRUNED_FLAG, side_call)
                self.assertIn(bgw.SIDE_MODULE_CTOR_EXPORT_FLAG, side_call)

            self.assertEqual(rep.side_modules[0].exported_symbol_count, 2)
            self.assertEqual(rep.side_modules[1].exported_symbol_count, 2)
            self.assertEqual(rep.side_modules[2].exported_symbol_count, 1)

            gm = json.loads((out_dir / bgw.GUEST_MANIFEST_FILE).read_text(encoding="utf-8"))
            self.assertEqual(gm["sideModuleSizes"][0]["exportedSymbolCount"], 2)
            self.assertEqual(gm["sideModuleSizes"][1]["exportedSymbolCount"], 2)
            self.assertEqual(gm["sideModuleSizes"][2]["exportedSymbolCount"], 1)

            # 3) Missing llvm-nm in real asymmetric build fails loudly.
            for f in (repo / "work" / "emsdk" / "upstream" / "bin").iterdir():
                f.unlink()
            missing_rep = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist-no-nm",
                jobs=2,
                split_modules=2,
                split_strategy="asymmetric",
                runtime_lib=orig_lib,
                run_command=RecordingRunner(),
                which_fn=lambda _n: None,
            )
            self.assertFalse(missing_rep.ok)
            self.assertTrue(any("llvm-nm" in e for e in missing_rep.errors), missing_rep.errors)

    def test_empty_side_module_guard_catches_stripped_676_byte_output(self) -> None:
        # Unit check on helper thresholds.
        self.assertIsNone(
            bgw.check_side_module_not_empty(
                676,
                input_object_bytes=1024,  # tiny test inputs <= 64 KiB -> no false positive
                module_name="guest_part_0.wasm",
                side_module_mode=2,
            )
        )
        err = bgw.check_side_module_not_empty(
            676,
            input_object_bytes=128 * 1024,  # > 64 KiB
            module_name="guest_part_0.wasm",
            side_module_mode=2,
        )
        self.assertIsNotNone(err)
        assert err is not None
        self.assertIn("guest_part_0.wasm", err)
        self.assertIn("676 bytes", err)
        self.assertIn("-sSIDE_MODULE=2", err)
        self.assertIsNone(
            bgw.check_side_module_not_empty(
                8192,
                input_object_bytes=128 * 1024,
                module_name="guest_part_0.wasm",
                side_module_mode=2,
            )
        )

        # End-to-end build fails loudly when a >64 KiB shard bucket links to a 676-byte wasm.
        class StrippedSideRunner(RecordingRunner):
            def __init__(self, *, side_wasm_size: int) -> None:
                super().__init__()
                self.side_wasm_size = side_wasm_size

            def __call__(self, argv: Sequence[str]) -> bgw.CommandResult:
                res = super().__call__(argv)
                tokens = tuple(str(a) for a in argv)
                out_path = _dash_o(tokens)
                if res.returncode == 0 and out_path is not None:
                    p = Path(out_path)
                    if p.suffix.lower() == ".o":
                        # Write 96 KiB per compiled object so bucket input sum > 64 KiB.
                        p.write_bytes(b"\x00" * (96 * 1024))
                    elif p.name.startswith("guest_part_") and p.suffix.lower() == ".wasm":
                        payload = b"\x00asm\x01\x00\x00\x00"
                        pad = max(0, self.side_wasm_size - len(payload))
                        p.write_bytes(payload + b"\x00" * pad)
                return res

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _fake_emsdk(repo)
            orig_lib = _touch_lib(repo / "runtime" / "build-wasm64" / "libplaygta6_core.a")
            src = root / "recomp-guard"
            _write(src / "lift" / "shard-00000.ll", "; shard 0\n")
            _write(src / "lift" / "shard-00001.ll", "; shard 1\n")
            _write(src / "stubs" / "pg6_import_thunks.cpp", "void thunk() {}\n")

            bad_runner = StrippedSideRunner(side_wasm_size=676)
            bad_rep = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache-bad",
                out_dir=root / "dist-bad",
                jobs=2,
                split_modules=2,
                split_strategy="asymmetric",
                runtime_lib=orig_lib,
                run_command=bad_runner,
                which_fn=lambda _n: None,
            )
            self.assertFalse(bad_rep.ok)
            self.assertTrue(
                any("guest_part_0.wasm" in e and "676 bytes" in e for e in bad_rep.errors),
                bad_rep.errors,
            )
            # Aborts before linking runtime.js / runtime.wasm.
            self.assertEqual(bad_runner.link_calls, [])

            # Non-stripped (>= 4 KiB) side module succeeds normally.
            good_runner = StrippedSideRunner(side_wasm_size=8192)
            good_rep = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache-good",
                out_dir=root / "dist-good",
                jobs=2,
                split_modules=2,
                split_strategy="asymmetric",
                runtime_lib=orig_lib,
                run_command=good_runner,
                which_fn=lambda _n: None,
            )
            self.assertTrue(good_rep.ok, good_rep.errors)
            self.assertEqual(len(good_runner.link_calls), 1)


class TestRuntimeHeaderCacheKeys(unittest.TestCase):
    """W110 (#738): `runtime/` and generated header fingerprints in C/C++ cache keys only."""

    def _repo_with_runtime(self, root: Path) -> Path:
        repo = root / "repo"
        _fake_emsdk(repo)
        _touch_lib(repo / "runtime" / "build-wasm64" / "libplaygta6_core.a")
        _write(repo / "runtime" / "guest" / "boot.h", "#pragma once\nstruct BootCfg { int a; };\n")
        _write(repo / "runtime" / "os" / "win32_kernel.h", "#pragma once\nint pg6_open();\n")
        _write(repo / "runtime" / "os" / "inline_helpers.inl", "static inline int ok() { return 1; }\n")
        _write(repo / "runtime" / "gpu" / "state.hpp", "#pragma once\nstruct GpuState { int v; };\n")
        _write(repo / "runtime" / "main.cpp", '#include "guest/boot.h"\nint main() { return 0; }\n')
        _write(repo / "runtime" / "wasm_boot.cpp", '#include "guest/boot.h"\nvoid boot() {}\n')
        return repo

    def test_compute_runtime_headers_fingerprint_determinism_extensions_and_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo_with_runtime(root)

            fp1 = bgw.compute_runtime_headers_fingerprint(repo)
            fp2 = bgw.compute_runtime_headers_fingerprint(repo)
            self.assertEqual(len(fp1), 64)
            self.assertEqual(fp1, fp2)

            # Non-header files and excluded dirs (build*, target, hidden) must not affect digest.
            _write(repo / "runtime" / "os" / "win32_kernel.cpp", "int pg6_open() { return 0; }\n")
            _write(repo / "runtime" / "build" / "generated.h", "#pragma once\n#define X 1\n")
            _write(repo / "runtime" / "build-wasm64" / "config.hpp", "#pragma once\n#define Y 2\n")
            _write(repo / "runtime" / "target" / "debug.h", "#pragma once\n")
            _write(repo / "runtime" / ".hidden" / "secret.h", "#pragma once\n")
            (repo / "runtime" / "guest" / "dir_named.h").mkdir(parents=True, exist_ok=True)
            self.assertEqual(bgw.compute_runtime_headers_fingerprint(repo), fp1)

            # Modifying a .h file changes the fingerprint.
            _write(
                repo / "runtime" / "guest" / "boot.h",
                "#pragma once\nstruct BootCfg { int a; int b; };\n",
            )
            fp_h = bgw.compute_runtime_headers_fingerprint(repo)
            self.assertNotEqual(fp_h, fp1)

            # Modifying a .hpp file changes the fingerprint.
            _write(
                repo / "runtime" / "gpu" / "state.hpp",
                "#pragma once\nstruct GpuState { int v; int w; };\n",
            )
            fp_hpp = bgw.compute_runtime_headers_fingerprint(repo)
            self.assertNotEqual(fp_hpp, fp_h)

            # Modifying a .inl file changes the fingerprint.
            _write(
                repo / "runtime" / "os" / "inline_helpers.inl",
                "static inline int ok() { return 2; }\n",
            )
            fp_inl = bgw.compute_runtime_headers_fingerprint(repo)
            self.assertNotEqual(fp_inl, fp_hpp)

            # Explicit header_roots list is supported and deterministic.
            extra_dir = root / "extra_headers"
            _write(extra_dir / "custom.h", "#pragma once\n#define Z 1\n")
            fp_multi_1 = bgw.compute_runtime_headers_fingerprint(
                header_roots=[repo / "runtime", extra_dir]
            )
            fp_multi_2 = bgw.compute_runtime_headers_fingerprint(
                header_roots=[repo / "runtime", extra_dir]
            )
            self.assertEqual(fp_multi_1, fp_multi_2)
            self.assertNotEqual(fp_multi_1, fp_inl)
            _write(extra_dir / "custom.h", "#pragma once\n#define Z 2\n")
            fp_multi_3 = bgw.compute_runtime_headers_fingerprint(
                header_roots=[repo / "runtime", extra_dir]
            )
            self.assertNotEqual(fp_multi_3, fp_multi_1)

    def test_cache_key_for_runtime_headers_affects_c_cpp_never_ll(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo_with_runtime(root)
            cpp_flags = bgw.compile_flags_for(opt="O2", kind="cpp")
            c_flags = bgw.compile_flags_for(opt="O2", kind="c")
            ll_flags = bgw.compile_flags_for(opt="O2", kind="ll")

            cpp_bytes = b'#include "guest/boot.h"\nvoid thunk() {}\n'
            c_bytes = b'#include "os/win32_kernel.h"\nint helper(void) { return 0; }\n'
            ll_bytes = b"; ModuleID = 'shard-00000'\ndefine void @sub_1() { ret void }\n"

            fp_before = bgw.compute_runtime_headers_fingerprint(repo)
            cpp_before = bgw.cache_key_for(
                cpp_bytes, cpp_flags, kind="cpp", runtime_headers_fingerprint=fp_before
            )
            c_before = bgw.cache_key_for(
                c_bytes, c_flags, kind="c", runtime_headers_fingerprint=fp_before
            )
            ll_before = bgw.cache_key_for(
                ll_bytes, ll_flags, kind="ll", runtime_headers_fingerprint=fp_before
            )
            # Unchanged header -> identical deterministic keys.
            fp_same = bgw.compute_runtime_headers_fingerprint(repo)
            self.assertEqual(
                cpp_before,
                bgw.cache_key_for(cpp_bytes, cpp_flags, kind="cpp", runtime_headers_fingerprint=fp_same),
            )
            # .ll key must match bare cache_key_for without runtime_headers_fingerprint.
            self.assertEqual(
                ll_before,
                bgw.cache_key_for(ll_bytes, ll_flags, kind="ll"),
            )

            # Mutate a runtime/ header -> .cpp and .c keys change, .ll stays IDENTICAL.
            _write(
                repo / "runtime" / "guest" / "boot.h",
                "#pragma once\nstruct BootCfg { int a; long long expanded; };\n",
            )
            fp_after = bgw.compute_runtime_headers_fingerprint(repo)
            self.assertNotEqual(fp_before, fp_after)

            cpp_after = bgw.cache_key_for(
                cpp_bytes, cpp_flags, kind="cpp", runtime_headers_fingerprint=fp_after
            )
            c_after = bgw.cache_key_for(
                c_bytes, c_flags, kind="c", runtime_headers_fingerprint=fp_after
            )
            ll_after = bgw.cache_key_for(
                ll_bytes, ll_flags, kind="ll", runtime_headers_fingerprint=fp_after
            )
            self.assertNotEqual(cpp_before, cpp_after)
            self.assertNotEqual(c_before, c_after)
            self.assertEqual(ll_before, ll_after)

    def test_run_build_runtime_header_change_invalidates_cpp_c_and_boot_keeps_ll_hit(self) -> None:
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo_with_runtime(root)
            src = _program_layout(root)
            # Add .c, .cc, and .cxx units to verify all C/C++ extensions invalidate.
            _write(src / "extra" / "helper.c", "int h_c(void) { return 1; }\n")
            _write(src / "extra" / "helper.cc", "int h_cc() { return 2; }\n")
            _write(src / "extra" / "helper.cxx", "int h_cxx() { return 3; }\n")
            manifest_path = src / "program-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"].extend(
                [
                    {"path": "extra/helper.c"},
                    {"path": "extra/helper.cc"},
                    {"path": "extra/helper.cxx"},
                ]
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            cache = root / "cache"
            lib = repo / "runtime" / "build-wasm64" / "libplaygta6_core.a"

            # Step 1: Warm the cache and assert runtime header fingerprint is computed ONCE.
            runner1 = RecordingRunner()
            with mock.patch.object(
                bgw,
                "compute_runtime_headers_fingerprint",
                wraps=bgw.compute_runtime_headers_fingerprint,
            ) as spy_fp:
                first = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                    source_dir=src,
                    repo_root=repo,
                    cache_dir=cache,
                    out_dir=root / "dist1",
                    jobs=2,
                    runtime_lib=lib,
                    link_guest_boot=True,
                    run_command=runner1,
                    which_fn=lambda _n: None,
                )
            self.assertTrue(first.ok, first.errors)
            self.assertEqual(spy_fp.call_count, 1)
            self.assertEqual(first.cache_hits, 0)
            self.assertEqual(first.cache_misses, len(first.sources))
            first_keys = {u.unit.unit_id: u.cache_key for u in first.sources}
            ll_ids = {u.unit.unit_id for u in first.sources if u.unit.kind == "ll"}
            c_cpp_ids = {u.unit.unit_id for u in first.sources if u.unit.kind in {"c", "cpp"}}
            self.assertGreaterEqual(len(ll_ids), 7)
            self.assertIn("runtime/wasm_boot.cpp", c_cpp_ids)
            self.assertIn("runtime/main.cpp", c_cpp_ids)
            self.assertIn("extra/helper.c", c_cpp_ids)
            self.assertIn("extra/helper.cc", c_cpp_ids)
            self.assertIn("extra/helper.cxx", c_cpp_ids)

            # Step 2: Unchanged rebuild -> 100% cache hits.
            runner2 = RecordingRunner()
            second = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist2",
                jobs=2,
                runtime_lib=lib,
                link_guest_boot=True,
                run_command=runner2,
                which_fn=lambda _n: None,
            )
            self.assertTrue(second.ok, second.errors)
            self.assertEqual(second.cache_hits, len(second.sources))
            self.assertEqual(second.cache_misses, 0)
            self.assertEqual(runner2.compile_calls, [])

            # Step 3: Change runtime/guest/boot.h -> only C/C++ units that include it (transitively) recompile; every other unit stays a hit
            _write(
                repo / "runtime" / "guest" / "boot.h",
                "#pragma once\nstruct BootCfg { int a; uint64_t new_field; };\n",
            )
            runner3 = RecordingRunner()
            third = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist3",
                jobs=2,
                runtime_lib=lib,
                link_guest_boot=True,
                run_command=runner3,
                which_fn=lambda _n: None,
            )
            self.assertTrue(third.ok, third.errors)
            dependents = {
                u.unit.unit_id for u in third.sources if u.status == "compiled"
            }
            self.assertTrue(dependents)
            self.assertTrue(dependents <= c_cpp_ids)
            self.assertLess(len(dependents), len(c_cpp_ids))
            self.assertEqual(third.cache_misses, len(dependents))
            self.assertEqual(len(runner3.compile_calls), len(dependents))
            for item in third.sources:
                if item.unit.unit_id in dependents:
                    self.assertNotEqual(item.cache_key, first_keys[item.unit.unit_id])
                else:
                    self.assertEqual(item.status, "hit", item.unit.unit_id)
                    self.assertEqual(item.cache_key, first_keys[item.unit.unit_id])

            # Step 4: Modify excluded dirs under runtime/ (build*, target, .hidden) -> ZERO misses.
            _write(repo / "runtime" / "build-wasm64" / "temp.h", "#pragma once\n// v2\n")
            _write(repo / "runtime" / "target" / "temp.h", "#pragma once\n// v2\n")
            _write(repo / "runtime" / ".cache" / "temp.h", "#pragma once\n// v2\n")
            runner4 = RecordingRunner()
            fourth = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist4",
                jobs=2,
                runtime_lib=lib,
                link_guest_boot=True,
                run_command=runner4,
                which_fn=lambda _n: None,
            )
            self.assertTrue(fourth.ok, fourth.errors)
            self.assertEqual(fourth.cache_misses, 0)
            self.assertEqual(fourth.cache_hits, len(fourth.sources))
            self.assertEqual(runner4.compile_calls, [])

    def test_run_build_generated_program_headers_invalidate_cpp_not_ll(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo_with_runtime(root)
            src = _program_layout(root)
            prog_h = _write(src / "pg6_program.h", "#pragma once\n#define PG6_PROG_VER 1\n")
            exp_h = _write(
                src / "exports" / "pg6_export_thunks.h",
                "#pragma once\nvoid pg6_export_thunks();\n",
            )
            _write(
                src / "pg6_program.cpp",
                '#include "pg6_program.h"\n#include "exports/pg6_export_thunks.h"\n'
                "extern const int pg6_program_modules;\n",
            )
            cache = root / "cache"
            lib = repo / "runtime" / "build-wasm64" / "libplaygta6_core.a"

            first = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist1",
                jobs=2,
                runtime_lib=lib,
                run_command=RecordingRunner(),
                which_fn=lambda _n: None,
            )
            self.assertTrue(first.ok, first.errors)
            first_keys = {u.unit.unit_id: u.cache_key for u in first.sources}
            ll_ids = {u.unit.unit_id for u in first.sources if u.unit.kind == "ll"}
            cpp_ids = {u.unit.unit_id for u in first.sources if u.unit.kind in {"c", "cpp"}}

            # Mutate top-level generated pg6_program.h -> all C/C++ units recompile, .ll 100% hit.
            _write(prog_h, "#pragma once\n#define PG6_PROG_VER 2\n")
            second = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist2",
                jobs=2,
                runtime_lib=lib,
                run_command=RecordingRunner(),
                which_fn=lambda _n: None,
            )
            self.assertTrue(second.ok, second.errors)
            self.assertEqual(second.cache_misses, len(cpp_ids))
            self.assertEqual(second.cache_hits, len(ll_ids))
            for item in second.sources:
                if item.unit.kind == "ll":
                    self.assertEqual(item.cache_key, first_keys[item.unit.unit_id])
                else:
                    self.assertNotEqual(item.cache_key, first_keys[item.unit.unit_id])

            # Mutate exports/pg6_export_thunks.h -> both exports/pg6_export_thunks.cpp and
            # pg6_program.cpp (which #includes it) recompile; .ll shards stay 100% hit.
            _write(exp_h, "#pragma once\nvoid pg6_export_thunks(); // v2\n")
            third = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist3",
                jobs=2,
                runtime_lib=lib,
                run_command=RecordingRunner(),
                which_fn=lambda _n: None,
            )
            self.assertTrue(third.ok, third.errors)
            compiled_ids = {u.unit.unit_id for u in third.sources if u.status == "compiled"}
            self.assertEqual(
                compiled_ids,
                {"exports/pg6_export_thunks.cpp", "pg6_program.cpp"},
            )
            for item in third.sources:
                if item.unit.kind == "ll":
                    self.assertEqual(item.status, "hit", item.unit.unit_id)
                    self.assertEqual(item.cache_key, first_keys[item.unit.unit_id])


class TestSplitDefaultPthreadPoolSize(unittest.TestCase):
    """W158 (#852): Split-module builds default to -sPTHREAD_POOL_SIZE=48."""

    def test_split_default_constant_and_resolve_link_flags(self) -> None:
        self.assertEqual(bgw.SPLIT_DEFAULT_PTHREAD_POOL_SIZE, 48)

        split_flags = bgw.resolve_link_flags(main_module=True)
        self.assertIn("-sPTHREAD_POOL_SIZE=48", split_flags)
        self.assertNotIn("-sPTHREAD_POOL_SIZE=4", split_flags)
        self.assertNotIn("-sPTHREAD_POOL_SIZE=8", split_flags)

        mono_flags = bgw.resolve_link_flags(main_module=False)
        self.assertIn("-sPTHREAD_POOL_SIZE=4", mono_flags)
        self.assertNotIn("-sPTHREAD_POOL_SIZE=48", mono_flags)

        override_higher = bgw.resolve_link_flags(main_module=True, pthread_pool_size=64)
        self.assertIn("-sPTHREAD_POOL_SIZE=64", override_higher)
        self.assertNotIn("-sPTHREAD_POOL_SIZE=48", override_higher)

        override_lower = bgw.resolve_link_flags(main_module=True, pthread_pool_size=16)
        self.assertIn("-sPTHREAD_POOL_SIZE=16", override_lower)
        self.assertNotIn("-sPTHREAD_POOL_SIZE=48", override_lower)

    def test_cli_split_default_and_explicit_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _fake_emsdk(repo)
            _touch_lib(repo / "runtime" / "build-wasm64" / "libplaygta6_core.a")
            src = _program_layout(root)

            code, out, err = _run_cli(
                [
                    str(src),
                    "--dry-run",
                    "--json",
                    "--split-modules",
                    "2",
                    "--repo-root",
                    str(repo),
                    "--cache-dir",
                    str(root / "cache-default"),
                    "--out-dir",
                    str(root / "dist-default"),
                ]
            )
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertIn("-sPTHREAD_POOL_SIZE=48", payload["link_flags"])
            self.assertIn("-sPTHREAD_POOL_SIZE=48", payload["link_argv"])
            self.assertNotIn("-sPTHREAD_POOL_SIZE=8", payload["link_flags"])
            self.assertNotIn("-sPTHREAD_POOL_SIZE=4", payload["link_flags"])

            # Explicit CLI --pthread-pool-size overrides split default 48.
            code_ov, out_ov, err_ov = _run_cli(
                [
                    str(src),
                    "--dry-run",
                    "--json",
                    "--split-modules",
                    "2",
                    "--pthread-pool-size",
                    "16",
                    "--repo-root",
                    str(repo),
                    "--cache-dir",
                    str(root / "cache-override"),
                    "--out-dir",
                    str(root / "dist-override"),
                ]
            )
            self.assertEqual(code_ov, 0, err_ov)
            ov_payload = json.loads(out_ov)
            self.assertIn("-sPTHREAD_POOL_SIZE=16", ov_payload["link_flags"])
            self.assertIn("-sPTHREAD_POOL_SIZE=16", ov_payload["link_argv"])
            self.assertNotIn("-sPTHREAD_POOL_SIZE=48", ov_payload["link_flags"])
            self.assertNotIn("-sPTHREAD_POOL_SIZE=48", ov_payload["link_argv"])

    def test_cli_help_documents_split_default_48(self) -> None:
        parser = bgw.build_arg_parser()
        pool_action = next(
            a for a in parser._actions if "--pthread-pool-size" in a.option_strings
        )
        self.assertIn(
            "default: 4 monolithic, 48 when --split-modules > 1",
            pool_action.help or "",
        )


def _multimodule_bink2_program_layout(root: Path) -> Path:
    """Create a synthetic R12/R44 build-program tree with main + vcruntime140 + bink2scarlett."""
    src = root / "program-bink2"
    _module_layout(src, "vcruntime140", shards=2)
    _module_layout(src, "bink2scarlett", shards=2)
    _module_layout(src, "main", shards=3)
    # Include an extra non-`shard-*` guest DLL .ll unit to confirm it is bucketed as a side shard.
    _write(
        src / "bink2scarlett" / "bink2_extra.ll",
        "; ModuleID = 'bink2_extra'\n"
        "define void @sub_180002200(ptr %s) { ret void }\n",
    )
    _write(src / "exports" / "pg6_export_thunks.cpp", "void pg6_export_thunks() {}\n")
    _write(src / "program_modules.cpp", "const int pg6_program_modules = 3;\n")
    _write(
        src / "pg6_program.cpp",
        'extern "C" const int __playgta6_shard_dispatch_vcruntime140;\n'
        'extern "C" const int __playgta6_shard_dispatch_bink2scarlett;\n'
        'extern "C" const int __playgta6_shard_dispatch_main;\n',
    )
    manifest = {
        "schema": "playgta6-recomp-program",
        "schema_version": 1,
        "options": {
            "allow_allocator_split": False,
            "force_guest": ["bink2scarlett.dll"],
            "build": {
                "emit_obj": False,
                "shard_size": 128,
                "indirect_dispatch": "shard",
            },
        },
        "modules": [
            {
                "file": "vcruntime140.dll",
                "dll": "vcruntime140.dll",
                "module": "vcruntime140",
                "role": "dll",
                "entry_point": "0x180101000",
                "registration": {
                    "register_function": "__playgta6_register_lifted_functions_vcruntime140",
                    "table_symbol": "__playgta6_lifted_functions_vcruntime140",
                    "shard_dispatch_symbol": "__playgta6_shard_dispatch_vcruntime140",
                    "count": 40,
                },
                "bridge": {
                    "thunks_symbol": "pg6_import_thunks_vcruntime140",
                    "thunk_count_symbol": "pg6_import_thunk_count_vcruntime140",
                },
            },
            {
                "file": "Bink2Scarlett.dll",
                "dll": "bink2scarlett.dll",
                "module": "bink2scarlett",
                "role": "dll",
                "entry_point": "0x180001000",
                "tls": {"index": 1, "callbacks": ["0x180001080"]},
                "registration": {
                    "register_function": "__playgta6_register_lifted_functions_bink2scarlett",
                    "table_symbol": "__playgta6_lifted_functions_bink2scarlett",
                    "shard_dispatch_symbol": "__playgta6_shard_dispatch_bink2scarlett",
                    "count": 120,
                },
                "bridge": {
                    "thunks_symbol": "pg6_import_thunks_bink2scarlett",
                    "thunk_count_symbol": "pg6_import_thunk_count_bink2scarlett",
                },
                "exports": {
                    "total": 42,
                    "functions": 42,
                    "data": 0,
                },
            },
            {
                "file": "game.exe",
                "dll": "game.exe",
                "module": "main",
                "role": "exe",
                "entry_point": "0x140001000",
                "iat_prefill": [
                    {
                        "slot_rva": "0x2000",
                        "value": "0x180002000",
                        "import": "bink2scarlett.dll!BinkOpen",
                    }
                ],
                "registration": {
                    "register_function": "__playgta6_register_lifted_functions_main",
                    "table_symbol": "__playgta6_lifted_functions_main",
                    "shard_dispatch_symbol": "__playgta6_shard_dispatch_main",
                    "count": 500,
                },
                "bridge": {
                    "thunks_symbol": "pg6_import_thunks_main",
                    "thunk_count_symbol": "pg6_import_thunk_count_main",
                },
            },
        ],
        "imports": [
            {
                "importer": "game.exe",
                "dll": "Bink2Scarlett.dll",
                "name": "BinkOpen",
                "binding": "guest",
                "provider": "bink2scarlett.dll",
                "target": "0x180002000",
                "kind": "function",
            },
            {
                "importer": "game.exe",
                "dll": "Bink2Scarlett.dll",
                "name": "BinkDoFrame",
                "binding": "guest",
                "provider": "bink2scarlett.dll",
                "target": "0x180002100",
                "kind": "function",
            },
        ],
        "files": [
            {"path": "exports/pg6_export_thunks.cpp"},
            {"path": "program_modules.cpp"},
            {"path": "pg6_program.cpp"},
        ],
    }
    (src / "program-manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return src


class TestMultiModuleBink2AsymmetricSplit(unittest.TestCase):
    """T44 (#889): Multi-module build-program + Bink2Scarlett.dll under -sSIDE_MODULE=2."""

    def _repo(self, root: Path) -> Path:
        repo = root / "repo"
        _fake_emsdk(repo)
        _touch_lib(repo / "runtime" / "build-wasm64" / "libplaygta6_core.a")
        _write(repo / "runtime" / "main.cpp", "int main() { return 0; }\n")
        _write(repo / "runtime" / "wasm_boot.cpp", "void wasm_boot() {}\n")
        return repo

    def test_required_export_prefixes_and_manifest_extraction_preserve_guest_dll_symbols(
        self,
    ) -> None:
        for sym in (
            "pg6_import_thunks_bink2scarlett",
            "pg6_import_thunk_count_bink2scarlett",
            "pg6_import_thunks_vcruntime140",
            "pg6_import_thunk_count_vcruntime140",
            "pg6_import_thunks",
            "pg6_import_thunk_count",
            "pg6_export_thunks",
            "pg6_export_thunk_count",
            "__playgta6_register_lifted_functions_bink2scarlett",
            "__playgta6_lifted_functions_bink2scarlett",
            "__playgta6_lifted_function_count_bink2scarlett",
            "__playgta6_shard_dispatch_bink2scarlett",
            "__playgta6_dispatch_shard_bink2scarlett_00000",
        ):
            self.assertTrue(
                bgw.is_required_side_export_symbol(sym),
                f"expected {sym!r} to be preserved as a required side-module export",
            )
        self.assertFalse(bgw.is_required_side_export_symbol("sub_180009990"))
        self.assertFalse(bgw.is_required_side_export_symbol("bink2_internal_scratch"))

        with tempfile.TemporaryDirectory() as tmp:
            src = _multimodule_bink2_program_layout(Path(tmp))
            extracted = bgw.extract_program_required_exports(src)
            for expected_sym in (
                "__playgta6_register_lifted_functions_bink2scarlett",
                "__playgta6_lifted_functions_bink2scarlett",
                "__playgta6_lifted_function_count_bink2scarlett",
                "__playgta6_shard_dispatch_bink2scarlett",
                "pg6_import_thunks_bink2scarlett",
                "pg6_import_thunk_count_bink2scarlett",
                "__playgta6_register_lifted_functions_vcruntime140",
                "__playgta6_shard_dispatch_vcruntime140",
                "pg6_import_thunks_vcruntime140",
                "pg6_import_thunk_count_vcruntime140",
                "sub_180001000",  # bink2scarlett DllMain
                "sub_180001080",  # bink2scarlett TLS callback
                "sub_180002000",  # BinkOpen guest export target
                "sub_180002100",  # BinkDoFrame guest export target
                "sub_180101000",  # vcruntime140 DllMain
                "sub_140001000",  # main entry_point
            ):
                self.assertIn(expected_sym, extracted)

            # Direct compute_side_module_cross_exports preserves bridge thunk globals
            # even if placed in a side bucket, while still pruning private symbols.
            nm_by_obj = {
                "bink2_side.o": "\n".join(
                    [
                        "00000000 D pg6_import_thunks_bink2scarlett",
                        "00000008 D pg6_import_thunk_count_bink2scarlett",
                        "00000010 T __playgta6_dispatch_shard_bink2scarlett_00000",
                        "00000020 T sub_180002000",
                        "00000030 T sub_180009990",  # private -> PRUNE
                    ]
                )
            }
            exports = bgw.compute_side_module_cross_exports(
                [[Path("cache/bink2_side.o")]],
                [],
                Path("llvm-nm"),
                run_command=lambda argv: bgw.CommandResult(
                    returncode=0,
                    stdout="\n".join(
                        nm_by_obj[Path(tok).name]
                        for tok in argv[1:]
                        if not str(tok).startswith("-")
                    )
                    + "\n",
                ),
                required_names=bgw.SIDE_MODULE_REQUIRED_EXPORT_NAMES | extracted,
            )
            self.assertEqual(
                exports[0],
                [
                    "__playgta6_dispatch_shard_bink2scarlett_00000",
                    "pg6_import_thunk_count_bink2scarlett",
                    "pg6_import_thunks_bink2scarlett",
                    "sub_180002000",
                ],
            )
            self.assertNotIn("sub_180009990", exports[0])

    def test_discover_guest_dll_modules_sanitizes_and_detects_nested_shard_dispatch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 1) Full program-manifest.json (recomp build-program schema)
            src = _multimodule_bink2_program_layout(root)
            self.assertEqual(
                bgw.discover_guest_dll_modules(src),
                ("vcruntime140", "bink2scarlett"),
            )
            det = bgw.detect_indirect_dispatch(src, bgw.discover_sources(src))
            self.assertEqual(det, bgw.IndirectDispatchDetection("shard", "program-manifest.json"))

            # 2) Legacy manifest without `role` field (only `file`)
            legacy_src = _program_layout(root / "legacy")
            self.assertEqual(
                bgw.discover_guest_dll_modules(legacy_src),
                ("vcruntime140",),
            )

            # 3) Dirty host path in module entry is sanitized to bare module identifier
            dirty = root / "dirty"
            dirty.mkdir()
            (dirty / bgw.PROGRAM_MANIFEST_FILE).write_text(
                json.dumps(
                    {
                        "modules": [
                            {"module": r"C:\Users\alice\work\bink2scarlett", "role": "dll"},
                            {"module": "../secret/vcruntime140", "role": "dll"},
                            {"module": "bink2scarlett", "role": "dll"},  # duplicate ignored
                            {"module": "game_entry", "role": "exe"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            cleaned = bgw.discover_guest_dll_modules(dirty)
            self.assertEqual(cleaned, ("bink2scarlett", "vcruntime140"))
            for item in cleaned:
                self.assertNotIn("\\", item)
                self.assertNotIn("/", item)
                self.assertNotIn("alice", item)

            # 4) Single-module directory without program-manifest.json -> empty tuple
            single = root / "single"
            _write(single / "lift" / "shard-00000.ll", "; single\n")
            self.assertEqual(bgw.discover_guest_dll_modules(single), ())

    def test_multimodule_bink2scarlett_asymmetric_split_build_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            orig_lib = repo / "runtime" / "build-wasm64" / "libplaygta6_core.a"
            src = _multimodule_bink2_program_layout(root)
            cache = root / "cache"
            out_dir = root / "dist"

            # 1) CLI dry-run JSON reports guest_dll_modules and includes all bink2scarlett .ll shards.
            code, out, err = _run_cli(
                [
                    str(src),
                    "--dry-run",
                    "--json",
                    "--split-modules",
                    "2",
                    "--split-strategy",
                    "asymmetric",
                    "--repo-root",
                    str(repo),
                    "--cache-dir",
                    str(cache),
                    "--out-dir",
                    str(out_dir),
                ]
            )
            self.assertEqual(code, 0, err)
            dry_payload = json.loads(out)
            self.assertTrue(dry_payload["ok"])
            self.assertEqual(dry_payload["indirect_dispatch"], "shard")
            self.assertEqual(dry_payload["indirect_dispatch_source"], "program-manifest.json")
            self.assertEqual(
                dry_payload["guest_dll_modules"],
                ["vcruntime140", "bink2scarlett"],
            )
            self.assertEqual(dry_payload["side_module_mode"], 2)
            self.assertEqual(len(dry_payload["side_modules"]), 3)
            dry_shard_ids = [
                sid
                for side in dry_payload["side_modules"]
                if side["role"] == "shards"
                for sid in side["shards"]
            ]
            dry_reg_ids = [
                sid
                for side in dry_payload["side_modules"]
                if side["role"] == "registration"
                for sid in side["shards"]
            ]
            for expected_bink_shard in (
                "bink2scarlett/bink2_extra.ll",
                "bink2scarlett/lift/shard-00000.ll",
                "bink2scarlett/lift/shard-00001.ll",
                "vcruntime140/lift/shard-00000.ll",
                "vcruntime140/lift/shard-00001.ll",
                "main/lift/shard-00000.ll",
                "main/lift/shard-00001.ll",
                "main/lift/shard-00002.ll",
            ):
                self.assertIn(expected_bink_shard, dry_shard_ids)
            self.assertEqual(
                dry_reg_ids,
                [
                    "bink2scarlett/lift/functions.ll",
                    "main/lift/functions.ll",
                    "vcruntime140/lift/functions.ll",
                ],
            )
            self.assertNotIn("\\\\", out)

            # 2) Full mocked asymmetric (-sSIDE_MODULE=2) build with realistic llvm-nm symbols.
            nm_outputs = {
                "bink2scarlett__bink2_extra": "\n".join(
                    [
                        "00000000 T sub_180002200",  # called from main shard -> cross-bucket export
                        "00000010 T sub_180009901",  # private in bink2 -> PRUNE
                    ]
                ),
                "bink2scarlett__lift__shard-00000": "\n".join(
                    [
                        "00000000 T __playgta6_dispatch_shard_bink2scarlett_00000",
                        "00000010 T sub_180001000",  # DllMain (from manifest entry_point) -> KEEP
                        "00000020 T sub_180002000",  # BinkOpen (from manifest guest import/IAT) -> KEEP
                        "00000030 T sub_180009990",  # private internal Bink2 helper -> PRUNE
                        "         U sub_180009991",  # intra-bucket call to shard-00001 -> PRUNE
                    ]
                ),
                "bink2scarlett__lift__shard-00001": "\n".join(
                    [
                        "00000000 T __playgta6_dispatch_shard_bink2scarlett_00001",
                        "00000010 T sub_180002100",  # BinkDoFrame (from manifest guest import) -> KEEP
                        "00000020 T sub_180009991",  # intra-bucket helper -> PRUNE
                    ]
                ),
                "main__lift__shard-00000": "\n".join(
                    [
                        "00000000 T __playgta6_dispatch_shard_main_00000",
                        "00000010 T sub_140001000",  # main entry_point -> KEEP
                        "00000020 T sub_140009990",  # private in main -> PRUNE
                        "         U sub_180102000",  # cross-bucket call to vcruntime140 in part_1
                    ]
                ),
                "main__lift__shard-00001": "\n".join(
                    [
                        "00000000 T __playgta6_dispatch_shard_main_00001",
                        "00000010 T sub_140009991",  # private -> PRUNE
                    ]
                ),
                "main__lift__shard-00002": "\n".join(
                    [
                        "00000000 T __playgta6_dispatch_shard_main_00002",
                        "00000010 T sub_140009992",  # private -> PRUNE
                    ]
                ),
                "vcruntime140__lift__shard-00000": "\n".join(
                    [
                        "00000000 T __playgta6_dispatch_shard_vcruntime140_00000",
                        "00000010 T sub_180101000",  # vcruntime140 DllMain -> KEEP
                    ]
                ),
                "vcruntime140__lift__shard-00001": "\n".join(
                    [
                        "00000000 T __playgta6_dispatch_shard_vcruntime140_00001",
                        "00000010 T sub_180102000",  # called from part_0 -> cross-bucket KEEP
                        "00000020 T sub_180109999",  # private -> PRUNE
                        "         U sub_180002200",  # cross-bucket call into bink2_extra in part_0
                    ]
                ),
                "bink2scarlett__lift__functions": "\n".join(
                    [
                        "00000000 T __playgta6_register_lifted_functions_bink2scarlett",
                        "00000010 D __playgta6_lifted_functions_bink2scarlett",
                        "00000018 D __playgta6_lifted_function_count_bink2scarlett",
                        "00000020 D __playgta6_shard_dispatch_bink2scarlett",
                        "00000030 D bink2_reg_internal_pad",  # private -> PRUNE
                        "         U __playgta6_dispatch_shard_bink2scarlett_00000",
                        "         U __playgta6_dispatch_shard_bink2scarlett_00001",
                    ]
                ),
                "vcruntime140__lift__functions": "\n".join(
                    [
                        "00000000 T __playgta6_register_lifted_functions_vcruntime140",
                        "00000010 D __playgta6_lifted_functions_vcruntime140",
                        "00000018 D __playgta6_lifted_function_count_vcruntime140",
                        "00000020 D __playgta6_shard_dispatch_vcruntime140",
                        "         U __playgta6_dispatch_shard_vcruntime140_00000",
                        "         U __playgta6_dispatch_shard_vcruntime140_00001",
                    ]
                ),
                "main__lift__functions": "\n".join(
                    [
                        "00000000 T __playgta6_register_lifted_functions_main",
                        "00000010 D __playgta6_lifted_functions_main",
                        "00000018 D __playgta6_lifted_function_count_main",
                        "00000020 D __playgta6_shard_dispatch_main",
                        "         U __playgta6_dispatch_shard_main_00000",
                        "         U __playgta6_dispatch_shard_main_00001",
                        "         U __playgta6_dispatch_shard_main_00002",
                    ]
                ),
                "pg6_program": "\n".join(
                    [
                        "00000000 D pg6_program",
                        "         U __playgta6_shard_dispatch_bink2scarlett",
                        "         U __playgta6_shard_dispatch_vcruntime140",
                        "         U __playgta6_shard_dispatch_main",
                        "         U pg6_import_thunks_bink2scarlett",
                        "         U pg6_import_thunk_count_bink2scarlett",
                    ]
                ),
                "bink2scarlett__stubs__pg6_import_thunks": "\n".join(
                    [
                        "00000000 D pg6_import_thunks_bink2scarlett",
                        "00000008 D pg6_import_thunk_count_bink2scarlett",
                    ]
                ),
            }

            class SizedMultiModuleRunner(RecordingRunner):
                def __call__(self, argv: Sequence[str]) -> bgw.CommandResult:
                    res = super().__call__(argv)
                    tokens = tuple(str(a) for a in argv)
                    out_path = _dash_o(tokens)
                    if res.returncode == 0 and out_path is not None:
                        p = Path(out_path)
                        if p.suffix.lower() == ".o":
                            p.write_bytes(b"\x00" * 100)
                    return res

            runner = SizedMultiModuleRunner(nm_outputs=nm_outputs)
            report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=out_dir,
                jobs=2,
                split_modules=2,
                split_strategy="asymmetric",
                max_side_module_mib=994,
                runtime_lib=orig_lib,
                run_command=runner,
                which_fn=lambda _n: None,
            )
            self.assertTrue(report.ok, report.errors)
            self.assertEqual(report.side_module_mode, 2)
            self.assertEqual(report.guest_dll_modules, ("vcruntime140", "bink2scarlett"))
            self.assertEqual(len(report.side_modules), 3)
            self.assertEqual([m.role for m in report.side_modules], ["shards", "shards", "registration"])

            # Every shard of bink2scarlett, vcruntime140, and main is included in side modules,
            # and all 3 registration functions.ll live in guest_part_2.wasm.
            part0_shards = set(report.side_modules[0].shards)
            part1_shards = set(report.side_modules[1].shards)
            self.assertEqual(len(part0_shards & part1_shards), 0)
            all_bucketed = part0_shards | part1_shards
            self.assertEqual(
                all_bucketed,
                {
                    "bink2scarlett/bink2_extra.ll",
                    "bink2scarlett/lift/shard-00000.ll",
                    "bink2scarlett/lift/shard-00001.ll",
                    "main/lift/shard-00000.ll",
                    "main/lift/shard-00001.ll",
                    "main/lift/shard-00002.ll",
                    "vcruntime140/lift/shard-00000.ll",
                    "vcruntime140/lift/shard-00001.ll",
                },
            )
            self.assertEqual(
                report.side_modules[2].shards,
                (
                    "bink2scarlett/lift/functions.ll",
                    "main/lift/functions.ll",
                    "vcruntime140/lift/functions.ll",
                ),
            )

            # Verify bink2scarlett import thunks & program units are linked in main runtime.wasm,
            # while zero .ll shards leak into main link.
            main_call = runner.link_calls[0]
            main_names = {Path(t).name for t in main_call}
            by_id = {u.unit.unit_id: u for u in report.sources}
            for cpp_id in (
                "bink2scarlett/stubs/pg6_import_thunks.cpp",
                "vcruntime140/stubs/pg6_import_thunks.cpp",
                "main/stubs/pg6_import_thunks.cpp",
                "exports/pg6_export_thunks.cpp",
                "pg6_program.cpp",
                "program_modules.cpp",
            ):
                obj = by_id[cpp_id].object_path
                assert obj is not None
                self.assertIn(obj.name, main_names)
            for ll_id in all_bucketed | set(report.side_modules[2].shards):
                obj = by_id[ll_id].object_path
                assert obj is not None
                self.assertNotIn(obj.name, main_names)

            # Inspect generated -Wl,--export=<sym> response files for all 3 side modules.
            all_shard_export_lines = set(
                (cache / "side_part_0_exports.rsp").read_text(encoding="utf-8").splitlines()
            ) | set(
                (cache / "side_part_1_exports.rsp").read_text(encoding="utf-8").splitlines()
            )
            for expected_export in (
                "__playgta6_dispatch_shard_bink2scarlett_00000",
                "__playgta6_dispatch_shard_bink2scarlett_00001",
                "__playgta6_dispatch_shard_vcruntime140_00000",
                "__playgta6_dispatch_shard_vcruntime140_00001",
                "__playgta6_dispatch_shard_main_00000",
                "sub_180001000",  # bink2scarlett DllMain
                "sub_180002000",  # BinkOpen
                "sub_180002100",  # BinkDoFrame
                "sub_180002200",  # bink2_extra cross-bucket callee
                "sub_180101000",  # vcruntime140 DllMain
                "sub_180102000",  # vcruntime140 cross-bucket callee from part_0
                "sub_140001000",  # main entry
            ):
                self.assertIn(f"-Wl,--export={expected_export}", all_shard_export_lines)
            for pruned_internal in (
                "sub_180009901",
                "sub_180009990",
                "sub_180009991",
                "sub_140009990",
                "sub_140009991",
                "sub_140009992",
                "sub_180109999",
            ):
                self.assertNotIn(f"-Wl,--export={pruned_internal}", all_shard_export_lines)

            reg_export_lines = set(
                (cache / "side_part_2_exports.rsp").read_text(encoding="utf-8").splitlines()
            )
            self.assertIn(bgw.SIDE_MODULE_CTOR_EXPORT_FLAG, reg_export_lines)
            for expected_reg in (
                "__playgta6_register_lifted_functions_bink2scarlett",
                "__playgta6_lifted_functions_bink2scarlett",
                "__playgta6_lifted_function_count_bink2scarlett",
                "__playgta6_shard_dispatch_bink2scarlett",
                "__playgta6_register_lifted_functions_vcruntime140",
                "__playgta6_lifted_functions_vcruntime140",
                "__playgta6_lifted_function_count_vcruntime140",
                "__playgta6_shard_dispatch_vcruntime140",
                "__playgta6_register_lifted_functions_main",
                "__playgta6_lifted_functions_main",
                "__playgta6_lifted_function_count_main",
                "__playgta6_shard_dispatch_main",
            ):
                self.assertIn(f"-Wl,--export={expected_reg}", reg_export_lines)
            self.assertNotIn("-Wl,--export=bink2_reg_internal_pad", reg_export_lines)

            # Verify guest_manifest.json, JSON summary, and text report record guest_dll_modules.
            gm_text = (out_dir / bgw.GUEST_MANIFEST_FILE).read_text(encoding="utf-8")
            self.assertNotIn("\\\\", gm_text)
            gm = json.loads(gm_text)
            self.assertEqual(gm["guest_dll_modules"], ["vcruntime140", "bink2scarlett"])
            self.assertEqual(gm["splitStrategy"], "asymmetric")
            self.assertEqual(gm["sideModuleMode"], 2)
            self.assertEqual(gm["indirectDispatch"], "shard")

            json_summary = bgw.report_to_json(report)
            self.assertNotIn("\\\\", json_summary)
            parsed_summary = json.loads(json_summary)
            self.assertEqual(
                parsed_summary["guest_dll_modules"],
                ["vcruntime140", "bink2scarlett"],
            )

            text_summary = bgw.format_report_text(report, sanitize=True)
            self.assertIn("Guest DLL modules: vcruntime140, bink2scarlett", text_summary)


class TestT96LibopusAndOptModule(unittest.TestCase):
    """T96 (Refs #663): sibling libopus.a auto-linking and --opt-module overrides."""

    def _repo(self, root: Path) -> Path:
        repo = root / "repo"
        _fake_emsdk(repo)
        _touch_lib(repo / "runtime" / "build-wasm64" / "libplaygta6_core.a")
        _write(repo / "runtime" / "main.cpp", "int main() { return 0; }\n")
        _write(repo / "runtime" / "wasm_boot.cpp", "void pg6_boot() {}\n")
        return repo

    def _program_with_bink(self, root: Path) -> Path:
        src = root / "program-out"
        manifest = {
            "schema_version": 1,
            "modules": ["main", "bink2scarlett", "vcruntime140"],
            "indirect_dispatch": "shard",
        }
        _write(src / "program_manifest.json", json.dumps(manifest))
        _write(src / "pg6_program.cpp", "void pg6_program_init() {}\n")
        _write(src / "program_modules.cpp", "void pg6_modules() {}\n")
        _write(src / "exports" / "pg6_export_thunks.cpp", "void exp() {}\n")
        _write(src / "main" / "lift" / "shard-00000.ll", "; main shard 0\n" + ("; pad\n" * 40))
        _write(src / "main" / "lift" / "functions.ll", "; main reg\n")
        _write(src / "main" / "stubs" / "pg6_import_thunks.cpp", "void imp() {}\n")
        _write(src / "bink2scarlett" / "lift" / "shard-00000.ll", "; bink shard 0\n" + ("; pad\n" * 60))
        _write(src / "bink2scarlett" / "lift" / "functions.ll", "; bink reg\n")
        _write(src / "vcruntime140" / "lift" / "shard-00000.ll", "; vc shard 0\n")
        _write(src / "vcruntime140" / "lift" / "functions.ll", "; vc reg\n")
        return src

    def test_resolve_runtime_libs_auto_links_sibling_libopus_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            core = repo / "runtime" / "build-wasm64" / "libplaygta6_core.a"
            extra_obj = _write(root / "extra.o", "obj")

            # 1. Absent: unchanged when no sibling libopus.a exists.
            libs_absent = bgw.resolve_runtime_libs(
                repo_root=repo,
                runtime_lib=core,
                extra_objects=[extra_obj],
            )
            self.assertEqual(libs_absent, [core, extra_obj])

            # 2. Secondary fallback (<lib.parent>/libopus.a).
            fallback_opus = _touch_lib(core.parent / "libopus.a")
            libs_fallback = bgw.resolve_runtime_libs(
                repo_root=repo,
                runtime_lib=core,
                extra_objects=[extra_obj],
            )
            self.assertEqual(libs_fallback, [core, fallback_opus, extra_obj])

            # 3. Primary CMake FetchContent path (<lib.parent>/_deps/opus-build/libopus.a) preferred.
            fetch_opus = _touch_lib(core.parent / "_deps" / "opus-build" / "libopus.a")
            libs_fetch = bgw.resolve_runtime_libs(
                repo_root=repo,
                runtime_lib=core,
                extra_objects=[extra_obj],
            )
            self.assertEqual(libs_fetch, [core, fetch_opus, extra_obj])

            # 4. Auto-discovery (runtime_lib=None) also picks up sibling libopus.a.
            libs_auto = bgw.resolve_runtime_libs(
                repo_root=repo,
                runtime_lib=None,
                extra_objects=[],
            )
            self.assertEqual(
                [p.resolve() for p in libs_auto],
                [core.resolve(), fetch_opus.resolve()],
            )

            # 5. Deduplication when libopus.a is also passed via extra_objects (even non-normalized).
            non_norm_opus = core.parent / "_deps" / "opus-build" / ".." / "opus-build" / "libopus.a"
            libs_dedup = bgw.resolve_runtime_libs(
                repo_root=repo,
                runtime_lib=core,
                extra_objects=[non_norm_opus, extra_obj, fetch_opus],
            )
            self.assertEqual(libs_dedup, [core, fetch_opus, extra_obj])

    def test_split_and_monolithic_builds_preserve_libopus_without_stripping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            core = repo / "runtime" / "build-wasm64" / "libplaygta6_core.a"
            opus = _touch_lib(core.parent / "_deps" / "opus-build" / "libopus.a")
            opus_sentinel = b"!<arch>\nlibopus-real-bytes\n"
            opus.write_bytes(opus_sentinel)
            src = self._program_with_bink(root)

            # Monolithic link: libopus.a immediately follows libplaygta6_core.a.
            mono_runner = RecordingRunner()
            mono_report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache-mono",
                out_dir=root / "dist-mono",
                jobs=2,
                runtime_lib=core,
                run_command=mono_runner,
                which_fn=lambda _name: None,
            )
            self.assertTrue(mono_report.ok, mono_report.errors)
            self.assertEqual(len(mono_runner.link_calls), 1)
            mono_names = [Path(tok).name for tok in mono_runner.link_calls[0]]
            self.assertIn("libplaygta6_core.a", mono_names)
            self.assertIn("libopus.a", mono_names)
            self.assertEqual(
                mono_names.index("libopus.a"),
                mono_names.index("libplaygta6_core.a") + 1,
            )

            # Split-module link: libplaygta6_core.a is filtered with emar d + --whole-archive,
            # while libopus.a is preserved directly after it without emar d stripping or dropping.
            split_cache = root / "cache-split"
            split_runner = RecordingRunner()
            split_report = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=split_cache,
                out_dir=root / "dist-split",
                jobs=2,
                split_modules=2,
                link_guest_boot=True,
                runtime_lib=core,
                run_command=split_runner,
                which_fn=lambda _name: None,
            )
            self.assertTrue(split_report.ok, split_report.errors)
            # emar d ran ONLY on libplaygta6_core_split.a, never on libopus.a.
            self.assertEqual(len(split_runner.ar_calls), 1)
            self.assertEqual(
                Path(split_runner.ar_calls[0][2]).name,
                "libplaygta6_core_split.a",
            )
            self.assertFalse((split_cache / "libopus_split.a").exists())
            self.assertEqual(opus.read_bytes(), opus_sentinel)

            self.assertEqual(len(split_runner.link_calls), 1)
            main_call = split_runner.link_calls[0]
            wa_idx = main_call.index("-Wl,--whole-archive")
            self.assertEqual(Path(main_call[wa_idx + 1]).name, "libplaygta6_core_split.a")
            self.assertEqual(main_call[wa_idx + 2], "-Wl,--no-whole-archive")
            self.assertEqual(Path(main_call[wa_idx + 3]).resolve(), opus.resolve())
            self.assertIn("libopus.a", split_report.link_argv)
            self.assertNotIn("libopus_split.a", split_report.link_argv)

    def test_opt_module_validation_and_per_module_compilation_and_cache(self) -> None:
        # Validation checks.
        self.assertEqual(bgw.parse_opt_modules(None), {})
        self.assertEqual(
            bgw.parse_opt_modules(["bink2scarlett=O3", "main=-O2", "vcruntime140=z"]),
            {"bink2scarlett": "O3", "main": "O2", "vcruntime140": "Oz"},
        )
        self.assertEqual(
            bgw.parse_opt_modules(["bink2scarlett=O1", "bink2scarlett=O3"]),
            {"bink2scarlett": "O3"},
        )
        for bad in ("bink2scarlett", "=O3", "bink2scarlett=", "bink2scarlett=O4", "bink2scarlett=Ofast", "bad mod=O3"):
            with self.assertRaises(bgw.BuildGuestWasmError, msg=bad):
                bgw.parse_opt_modules([bad])

        # Prefix-before-__ fallback when SourceUnit.module is None.
        flat_unit = bgw.SourceUnit(
            unit_id="bink2scarlett__shard-00000.ll",
            path=Path("bink2scarlett__shard-00000.ll"),
            kind="ll",
            module=None,
        )
        other_flat = bgw.SourceUnit(
            unit_id="lift/shard-00000.ll",
            path=Path("lift/shard-00000.ll"),
            kind="ll",
            module=None,
        )
        self.assertEqual(
            bgw.opt_level_for_unit(
                flat_unit,
                default_opt="O2",
                opt_modules={"bink2scarlett": "O3"},
            ),
            "O3",
        )
        self.assertEqual(
            bgw.opt_level_for_unit(
                other_flat,
                default_opt="O2",
                opt_modules={"bink2scarlett": "O3"},
            ),
            "O2",
        )

        # End-to-end build with --opt O2 and --opt-module bink2scarlett=O3 (+ warm cache delta).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            core = repo / "runtime" / "build-wasm64" / "libplaygta6_core.a"
            src = self._program_with_bink(root)
            cache = root / "cache"

            # Warm build at uniform O2.
            runner1 = RecordingRunner()
            rep1 = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist1",
                jobs=2,
                opt="O2",
                runtime_lib=core,
                run_command=runner1,
                which_fn=lambda _name: None,
            )
            self.assertTrue(rep1.ok, rep1.errors)
            by_id_1 = {u.unit.unit_id: u for u in rep1.sources}
            self.assertIn("-O2", by_id_1["bink2scarlett/lift/shard-00000.ll"].argv)
            self.assertIn("-O2", by_id_1["main/lift/shard-00000.ll"].argv)

            # Rebuild with opt_modules=["bink2scarlett=O3"] and shrink_largest_shards=1:
            # bink2scarlett gets -O3 (cache miss; overrides shrink_largest_shards),
            # while main/vcruntime140 remain at -O2 (cache hits where unchanged).
            runner2 = RecordingRunner()
            rep2 = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=root / "dist2",
                jobs=2,
                opt="O2",
                opt_modules=["bink2scarlett=O3"],
                shrink_largest_shards=1,
                runtime_lib=core,
                run_command=runner2,
                which_fn=lambda _name: None,
            )
            self.assertTrue(rep2.ok, rep2.errors)
            self.assertEqual(rep2.opt_modules, {"bink2scarlett": "O3"})
            by_id_2 = {u.unit.unit_id: u for u in rep2.sources}
            for bink_id in ("bink2scarlett/lift/shard-00000.ll", "bink2scarlett/lift/functions.ll"):
                self.assertIn("-O3", by_id_2[bink_id].argv)
                self.assertNotIn("-O2", by_id_2[bink_id].argv)
                self.assertNotIn("-Oz", by_id_2[bink_id].argv)
                self.assertFalse(by_id_2[bink_id].cache_hit)
                self.assertNotEqual(by_id_2[bink_id].cache_key, by_id_1[bink_id].cache_key)
            for other_id in ("main/lift/shard-00000.ll", "vcruntime140/lift/shard-00000.ll"):
                self.assertIn("-O2", by_id_2[other_id].argv)
                self.assertNotIn("-O3", by_id_2[other_id].argv)
                self.assertTrue(by_id_2[other_id].cache_hit)
                self.assertEqual(by_id_2[other_id].cache_key, by_id_1[other_id].cache_key)

            # CLI integration: dry-run JSON and text output + invalid CLI flag exits 2.
            buf_out = io.StringIO()
            buf_err = io.StringIO()
            with redirect_stdout(buf_out), redirect_stderr(buf_err):
                code = bgw.main(
                    [
                        str(src),
                        "--repo-root",
                        str(repo),
                        "--cache-dir",
                        str(cache),
                        "--out-dir",
                        str(root / "dist-cli"),
                        "--opt",
                        "O2",
                        "--opt-module",
                        "bink2scarlett=O3",
                        "--dry-run",
                        "--json",
                    ]
                )
            self.assertEqual(code, 0, buf_err.getvalue())
            payload = json.loads(buf_out.getvalue())
            self.assertEqual(payload["opt_modules"], {"bink2scarlett": "O3"})
            by_cli_id = {u["id"]: u for u in payload["sources"]}
            self.assertIn("-O3", by_cli_id["bink2scarlett/lift/shard-00000.ll"]["argv"])
            self.assertIn("-O2", by_cli_id["main/lift/shard-00000.ll"]["argv"])

            text_report = bgw.format_report_text(rep2, sanitize=True)
            self.assertIn("Module opt overrides: bink2scarlett=-O3", text_report)

            for bad_arg in ("bink2scarlett=O4", "bink2scarlett"):
                err_out = io.StringIO()
                err_err = io.StringIO()
                with redirect_stdout(err_out), redirect_stderr(err_err):
                    bad_code = bgw.main(
                        [
                            str(src),
                            "--repo-root",
                            str(repo),
                            "--opt-module",
                            bad_arg,
                            "--dry-run",
                        ]
                    )
                self.assertEqual(bad_code, 2, bad_arg)
                self.assertIn("--opt-module", err_err.getvalue())


class TestT97FastSplitStubsAndSingleSideCollapse(unittest.TestCase):
    """T97 (Refs #663): zero-alloc fixed-arity split stubs and opt-in single code side module collapse."""

    _STUB_FIXTURE = r"""
var LDSO = { loadedLibsByName: { 'guest_part_0.wasm': { exports: {} } } };
var wasmImports = {};
var GOT = {};
var functionsInTableMap = new Map();
var __resolveCalls = {};

function instantiateModule(localScope) {
  var moduleExports;
  function resolveSymbol(sym) {
    __resolveCalls[sym] = (__resolveCalls[sym] || 0) + 1;
    return wasmImports[sym] || (localScope && localScope[sym]) || (moduleExports && moduleExports[sym]);
  }
  var proxyHandler = {
    get(stubs, prop) {
      if (!(prop in stubs)) {
        var resolved;
        stubs[prop] = (...args) => {
          resolved ||= resolveSymbol(prop);
          return resolved(...args);
        };
      }
      return stubs[prop];
    }
  };
  moduleExports = {};
  return new Proxy({}, proxyHandler);
}

var __envProxy;
async function loadDylibs() {
  __envProxy = instantiateModule({});
}
async function finishBoot() {
  await loadDylibs();
}
"""

    def _require_node(self) -> str:
        node = shutil.which("node")
        if node is None:
            self.skipTest("node executable not available on PATH")
        return node

    def _run_node(self, script: str) -> dict[str, object]:
        node = self._require_node()
        with tempfile.TemporaryDirectory() as tmp:
            js_path = Path(tmp) / "t97_harness.js"
            js_path.write_text(script, encoding="utf-8")
            chk = subprocess.run([node, "--check", str(js_path)], capture_output=True, text=True, check=False)
            self.assertEqual(chk.returncode, 0, f"node --check failed:\n{chk.stderr}")
            proc = subprocess.run([node, str(js_path)], capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 0, f"node run failed:\nstdout={proc.stdout}\nstderr={proc.stderr}")
            return json.loads(proc.stdout)

    def test_stub_binders_emits_fixed_arity_wrappers_and_executes_eager_and_lazy(self) -> None:
        patched, status = bgw.optimize_split_runtime_js(
            self._STUB_FIXTURE,
            max_concurrent_pthread_loads=0,
        )
        self.assertEqual(
            status,
            {
                "already_patched": False,
                "stub_binders": True,
                "export_clearers": True,
                "post_load_dylibs_cleanup": True,
                "pthread_cmd_load_throttle": False,
                "pthread_cmd_loaded_dispatch": False,
            },
        )
        self.assertIn(bgw.SPLIT_PTHREAD_OPT_MARKER, patched)
        self.assertIn("(a) => (resolved || (resolved = resolveSymbol(_p)))(a)", patched)
        self.assertIn("(a, b) => (resolved || (resolved = resolveSymbol(_p)))(a, b)", patched)
        self.assertIn("(...args) => (resolved || (resolved = resolveSymbol(_p)))(...args)", patched)

        harness = patched + r"""
(async () => {
  // 1) Lazy path BEFORE finishBoot() / __pg6StubBinders runs:
  const lazyProxy = instantiateModule({});
  const lazySub = lazyProxy.sub_140001000;
  const lazyDispatch = lazyProxy.__playgta6_dispatch_shard_main_00000;
  const lazyOther = lazyProxy.pg6_runtime_vararg;
  const lazySym = lazyProxy[Symbol('sym_key')];

  // Inspect wrapper arity & source before eager binding overwrites stubs[_p]:
  const shapes = {
    subLen: lazySub.length,
    subSrc: String(lazySub),
    dispatchLen: lazyDispatch.length,
    dispatchSrc: String(lazyDispatch),
    otherLen: lazyOther.length,
    otherSrc: String(lazyOther),
    symLen: lazySym.length,
  };

  // Populate targets AFTER stubs were created (simulating forward cross-part reference).
  wasmImports.sub_140001000 = (statePtr) => statePtr + 16n;
  wasmImports.__playgta6_dispatch_shard_main_00000 = (localIdx, statePtr) => `${localIdx}:${statePtr}`;
  wasmImports.pg6_runtime_vararg = (a, b, c, d) => a + b + c + d;

  // First lazy call resolves & caches; second call reuses cached `resolved`.
  const lazySubR1 = lazySub(0x1000n).toString();
  const lazySubR2 = lazySub(0x2000n).toString();
  const lazyDispatchR1 = lazyDispatch(7, 0x3000n);
  const lazyDispatchR2 = lazyDispatch(8, 0x4000n);
  const lazyOtherR1 = lazyOther(1, 2, 3, 4);
  const lazyOtherR2 = lazyOther(10, 20, 30, 40);
  const lazyResolveCounts = {
    sub: __resolveCalls.sub_140001000,
    dispatch: __resolveCalls.__playgta6_dispatch_shard_main_00000,
    other: __resolveCalls.pg6_runtime_vararg,
  };

  // 2) Eager path via finishBoot() -> await loadDylibs() -> __pg6StubBinders:
  // Create stubs inside loadDylibs(), then resolve them before cleanup runs.
  const origLoadDylibs = loadDylibs;
  let capturedEagerSub, capturedEagerDispatch, capturedEagerOther;
  let targetEagerSub, targetEagerDispatch, targetEagerOther;
  loadDylibs = async () => {
    await origLoadDylibs();
    capturedEagerSub = __envProxy.sub_180002000;
    capturedEagerDispatch = __envProxy.__playgta6_dispatch_shard_bink2_00001;
    capturedEagerOther = __envProxy.pg6_win32_multi;

    targetEagerSub = (statePtr) => statePtr * 2n;
    targetEagerDispatch = (idx, statePtr) => `${idx}@${statePtr}`;
    targetEagerOther = (a, b, c, d, e) => [a, b, c, d, e].join('-');
    wasmImports.sub_180002000 = targetEagerSub;
    wasmImports.__playgta6_dispatch_shard_bink2_00001 = targetEagerDispatch;
    wasmImports.pg6_win32_multi = targetEagerOther;
  };

  await finishBoot();
  const resolveAfterBoot = {
    sub: __resolveCalls.sub_180002000,
    dispatch: __resolveCalls.__playgta6_dispatch_shard_bink2_00001,
    other: __resolveCalls.pg6_win32_multi,
  };

  // Captured wrappers invoke pre-bound `resolved` with ZERO additional resolveSymbol calls,
  // even though post_load_dylibs_cleanup deleted wasmImports.sub_180002000!
  const eagerSubR = capturedEagerSub(50n).toString();
  const eagerDispatchR = capturedEagerDispatch(3, 99n);
  const eagerOtherR = capturedEagerOther('a', 'b', 'c', 'd', 'e');
  const resolveAfterCalls = {
    sub: __resolveCalls.sub_180002000,
    dispatch: __resolveCalls.__playgta6_dispatch_shard_bink2_00001,
    other: __resolveCalls.pg6_win32_multi,
  };

  // Subsequent property lookups on __envProxy get the raw resolved target directly!
  const directRebound = {
    subDirect: __envProxy.sub_180002000 === targetEagerSub,
    dispatchDirect: __envProxy.__playgta6_dispatch_shard_bink2_00001 === targetEagerDispatch,
    otherDirect: __envProxy.pg6_win32_multi === targetEagerOther,
  };

  console.log(JSON.stringify({
    shapes,
    lazySubR1,
    lazySubR2,
    lazyDispatchR1,
    lazyDispatchR2,
    lazyOtherR1,
    lazyOtherR2,
    lazyResolveCounts,
    resolveAfterBoot,
    resolveAfterCalls,
    eagerSubR,
    eagerDispatchR,
    eagerOtherR,
    directRebound,
    bindersCleared: globalThis.__pg6StubBinders === null,
    subDeletedFromWasmImports: !('sub_180002000' in wasmImports),
  }));
})();
"""
        res = self._run_node(harness)
        shapes = res["shapes"]
        assert isinstance(shapes, dict)
        self.assertEqual(shapes["subLen"], 1)
        self.assertNotIn("...", shapes["subSrc"])
        self.assertNotIn("startsWith", shapes["subSrc"])
        self.assertEqual(shapes["dispatchLen"], 2)
        self.assertNotIn("...", shapes["dispatchSrc"])
        self.assertNotIn("startsWith", shapes["dispatchSrc"])
        self.assertEqual(shapes["otherLen"], 0)
        self.assertIn("...", shapes["otherSrc"])
        self.assertEqual(shapes["symLen"], 0)

        self.assertEqual(res["lazySubR1"], str(0x1010))
        self.assertEqual(res["lazySubR2"], str(0x2010))
        self.assertEqual(res["lazyDispatchR1"], f"7:{0x3000}")
        self.assertEqual(res["lazyDispatchR2"], f"8:{0x4000}")
        self.assertEqual(res["lazyOtherR1"], 10)
        self.assertEqual(res["lazyOtherR2"], 100)
        self.assertEqual(res["lazyResolveCounts"], {"sub": 1, "dispatch": 1, "other": 1})

        self.assertEqual(res["resolveAfterBoot"], {"sub": 1, "dispatch": 1, "other": 1})
        self.assertEqual(res["resolveAfterCalls"], {"sub": 1, "dispatch": 1, "other": 1})
        self.assertEqual(res["eagerSubR"], "100")
        self.assertEqual(res["eagerDispatchR"], "3@99")
        self.assertEqual(res["eagerOtherR"], "a-b-c-d-e")
        self.assertEqual(
            res["directRebound"],
            {"subDirect": True, "dispatchDirect": True, "otherDirect": True},
        )
        self.assertTrue(res["bindersCleared"])
        self.assertTrue(res["subDeletedFromWasmImports"])

    def test_stub_binders_avoids_parameter_shadowing_with_minified_names(self) -> None:
        minified_fixture = r"""
var wasmImports = {};
function mk() {
  let moduleExports;
  function b(k) { return wasmImports[k]; }
  var h = {
    get(a, p) {
      if (!(p in a)) { let r; a[p] = (...c) => { r ||= b(p); return r(...c); }; }
      return a[p];
    }
  };
  moduleExports = {};
  return new Proxy({}, h);
}
async function loadDylibs() {}
async function boot() { await loadDylibs(); }
"""
        patched, status = bgw.optimize_split_runtime_js(minified_fixture, max_concurrent_pthread_loads=0)
        self.assertTrue(status["stub_binders"])
        res = self._run_node(
            patched
            + r"""
(async () => {
  const px = mk();
  const s = px.sub_42;
  const d = px.__playgta6_dispatch_shard_m_0;
  const o = px.other_fn;
  wasmImports.sub_42 = (x) => x + 1;
  wasmImports.__playgta6_dispatch_shard_m_0 = (x, y) => x * y;
  wasmImports.other_fn = (x, y, z) => x + y + z;
  await boot();
  console.log(JSON.stringify({ s: s(41), d: d(6, 7), o: o(10, 20, 30) }));
})();
"""
        )
        self.assertEqual(res, {"s": 42, "d": 42, "o": 60})

    def test_asymmetric_opt_in_collapse_single_side_module(self) -> None:
        def _fake(unit_id: str, size: int) -> bgw.UnitBuildResult:
            p = Path(unit_id)
            return bgw.UnitBuildResult(
                unit=bgw.SourceUnit(unit_id, p, "ll"),
                cache_key="k",
                cache_hit=True,
                object_path=p.with_suffix(".o"),
                argv=(),
                status="hit",
                object_bytes=size,
            )

        items = [_fake(f"lift/shard-{i:05d}.ll", 100) for i in range(4)]
        # Total = 400 <= 994: default keeps 2 buckets (backwards-compatible), collapse_if_fits collapses to 1.
        self.assertEqual(len(bgw.partition_shards_asymmetric(items, 2, max_bucket_bytes=994)), 2)
        collapsed = bgw.partition_shards_asymmetric(
            items, 2, max_bucket_bytes=994, collapse_if_fits=True
        )
        self.assertEqual(len(collapsed), 1)
        self.assertEqual([it.unit.unit_id for it in collapsed[0]], [it.unit.unit_id for it in items])

        # Exact-n case (2 shards, 2 buckets) also collapses when combined weight fits.
        collapsed_two = bgw.partition_shards_asymmetric(
            items[:2], 2, max_bucket_bytes=200, collapse_if_fits=True
        )
        self.assertEqual(len(collapsed_two), 1)

        # Total = 400 > 350: still splits into 2 buckets under the cap.
        still_split = bgw.partition_shards_asymmetric(
            items, 2, max_bucket_bytes=350, collapse_if_fits=True
        )
        self.assertEqual(len(still_split), 2)
        self.assertEqual(len(still_split[0]), 3)
        self.assertEqual(len(still_split[1]), 1)

        # End-to-end mocked build + CLI dry-run with --collapse-single-side-module:
        # 1 code side module (guest_part_0.wasm) + 1 registration module (guest_part_1.wasm),
        # with ZERO cross-part sub_* exports between shards!
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _fake_emsdk(repo)
            orig_lib = _touch_lib(repo / "runtime" / "build-wasm64" / "libplaygta6_core.a")
            _write(repo / "runtime" / "main.cpp", "int main() { return 0; }\n")
            _write(repo / "runtime" / "wasm_boot.cpp", "void wasm_boot() {}\n")

            src = root / "recomp-collapse"
            _write(src / "lift" / "shard-00000.ll", "; s0\n")
            _write(src / "lift" / "shard-00001.ll", "; s1\n")
            _write(src / "lift" / "functions.ll", "; reg\n")
            _write(src / "stubs" / "pg6_import_thunks.cpp", "void thunk() {}\n")

            nm_outputs = {
                "shard-00000": "\n".join(
                    [
                        "00000000 T sub_140001000",
                        "00000010 T __playgta6_dispatch_shard_main_00000",
                        "         U sub_140002000",
                    ]
                ),
                "shard-00001": "\n".join(
                    [
                        "00000000 T sub_140002000",
                        "00000010 T __playgta6_dispatch_shard_main_00001",
                        "         U sub_140001000",
                    ]
                ),
                "functions": "\n".join(
                    [
                        "00000000 D __playgta6_shard_dispatch_main",
                        "         U __playgta6_dispatch_shard_main_00000",
                        "         U __playgta6_dispatch_shard_main_00001",
                    ]
                ),
                "pg6_import_thunks": "         U __playgta6_shard_dispatch_main\n",
            }
            cache = root / "cache"
            out_dir = root / "dist"
            runner = RecordingRunner(nm_outputs=nm_outputs)
            rep = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=out_dir,
                jobs=2,
                split_modules=2,
                split_strategy="asymmetric",
                collapse_single_side_module=True,
                runtime_lib=orig_lib,
                run_command=runner,
                which_fn=lambda _n: None,
            )
            self.assertTrue(rep.ok, rep.errors)
            self.assertEqual([m.name for m in rep.side_modules], ["guest_part_0.wasm", "guest_part_1.wasm"])
            self.assertEqual([m.role for m in rep.side_modules], ["shards", "registration"])
            self.assertEqual(
                rep.side_modules[0].shards,
                ("lift/shard-00000.ll", "lift/shard-00001.ll"),
            )
            self.assertEqual(rep.side_modules[1].shards, ("lift/functions.ll",))

            # Both sub_140001000 and sub_140002000 are intra-module inside guest_part_0.wasm -> pruned!
            rsp0_lines = (cache / "side_part_0_exports.rsp").read_text(encoding="utf-8").splitlines()
            self.assertNotIn("-Wl,--export=sub_140001000", rsp0_lines)
            self.assertNotIn("-Wl,--export=sub_140002000", rsp0_lines)
            self.assertIn("-Wl,--export=__playgta6_dispatch_shard_main_00000", rsp0_lines)
            self.assertIn("-Wl,--export=__playgta6_dispatch_shard_main_00001", rsp0_lines)

            # CLI flag wiring check.
            code, out, err = _run_cli(
                [
                    str(src),
                    "--dry-run",
                    "--json",
                    "--split-modules",
                    "2",
                    "--split-strategy",
                    "asymmetric",
                    "--collapse-single-side-module",
                    "--repo-root",
                    str(repo),
                    "--cache-dir",
                    str(cache),
                    "--out-dir",
                    str(root / "dist-cli"),
                ]
            )
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertTrue(payload["ok"])
            self.assertEqual(
                [m["name"] for m in payload["side_modules"]],
                ["guest_part_0.wasm", "guest_part_1.wasm"],
            )


def _encode_code_section(
    bodies: Sequence[tuple[Sequence[tuple[int, int]], int]],
) -> bytes:
    """Build a valid Wasm Code section (id=10) from `(locals_list, target_body_bytes)` tuples.

    Each `locals_list` item is `(count, valtype_byte)`. If `valtype_byte` is `0x63` or `0x64`,
    a 1-byte heaptype `0x70` (funcref) is emitted after it. `target_body_bytes` is the desired
    encoded body length (padded with `0x01` nop bytes before trailing `0x0B` end).
    """
    payload = bytearray(_uleb128(len(bodies)))
    for locals_list, target_body_bytes in bodies:
        body = bytearray(_uleb128(len(locals_list)))
        for count, valtype in locals_list:
            body.extend(_uleb128(count))
            body.append(valtype)
            if valtype in (0x63, 0x64):
                body.append(0x70)
        pad_len = max(0, target_body_bytes - len(body) - 1)
        if pad_len > 0:
            body.extend(b"\x01" * pad_len)
        body.append(0x0B)
        payload.extend(_uleb128(len(body)))
        payload.extend(body)
    return _wasm_section(bgw.WASM_SECTION_CODE, bytes(payload))


class TestT100Wasm64V8HeadroomAndStaleCleanup(unittest.TestCase):
    """T100 (#1706, Refs #663): Code-section V8 headroom telemetry & stale side-module cleanup."""

    def test_code_section_streaming_parser_metrics_and_errors(self) -> None:
        # 1) Multi-function Code section with multiple local decls, typed ref locals,
        #    multi-byte ULEB128 local counts & body sizes, and exact/over 1 MiB threshold.
        over_1mib_a = bgw.LARGE_FUNC_BODY_THRESHOLD_BYTES + 1
        over_1mib_b = 1_700_000
        exact_1mib = bgw.LARGE_FUNC_BODY_THRESHOLD_BYTES
        code_sec = _encode_code_section(
            [
                ([], 4),  # 0 locals, 4-byte body
                ([(3, 0x7F), (1300, 0x7E), (3, 0x7B)], 512),  # 1,306 locals (multi-byte ULEB128)
                ([(1, 0x63), (2048, 0x7F)], exact_1mib),  # 2,049 locals, exact 1 MiB (NOT > 1 MiB)
                ([(250, 0x7E), (125, 0x7C)], over_1mib_a),  # 375 locals, > 1 MiB (#1)
                ([(40, 0x7F)], over_1mib_b),  # 40 locals, > 1 MiB (#2, largest)
            ]
        )
        wasm = (
            b"\x00asm\x01\x00\x00\x00"
            + _encode_dylink0_section(memory_size=4096, table_size=5103, needed=("runtime.wasm",))
            + _encode_export_section([("__playgta6_dispatch_shard_main_00000", 0x00, 0)])
            + code_sec
        )

        m_bytes = bgw.inspect_wasm_bytes(wasm, name="guest_part_0.wasm", role="side")
        self.assertEqual(m_bytes.code_func_count, 5)
        self.assertEqual(m_bytes.max_func_body_bytes, over_1mib_b)
        self.assertEqual(m_bytes.max_func_locals, 2049)
        self.assertEqual(m_bytes.large_func_bodies_over_1mib, 2)
        self.assertEqual(m_bytes.table_size, 5103)
        self.assertEqual(m_bytes.shard_dispatch_exports, 1)

        # File-based streaming path produces identical metrics.
        with tempfile.TemporaryDirectory() as tmp:
            wasm_path = Path(tmp) / "guest_part_0.wasm"
            wasm_path.write_bytes(wasm)
            m_file = bgw.inspect_wasm_file(wasm_path, role="side")
            self.assertEqual(m_file, m_bytes)

        # 2) Absent, empty (0-byte), and zero-function Code sections cleanly yield zeros.
        no_code = b"\x00asm\x01\x00\x00\x00"
        empty_code = b"\x00asm\x01\x00\x00\x00" + _wasm_section(bgw.WASM_SECTION_CODE, b"")
        zero_funcs = b"\x00asm\x01\x00\x00\x00" + _encode_code_section([])
        for label, blob in (("absent", no_code), ("empty", empty_code), ("zero", zero_funcs)):
            m = bgw.inspect_wasm_bytes(blob, name=f"{label}.wasm")
            self.assertEqual(m.code_func_count, 0, label)
            self.assertEqual(m.max_func_body_bytes, 0, label)
            self.assertEqual(m.max_func_locals, 0, label)
            self.assertEqual(m.large_func_bodies_over_1mib, 0, label)

        # 3) Truncated / malformed Code sections raise WasmInspectError.
        # 3a) func_count=2 but only 1 body present.
        trunc_missing_body = b"\x00asm\x01\x00\x00\x00" + _wasm_section(
            bgw.WASM_SECTION_CODE,
            bytes([0x02, 0x02, 0x00, 0x0B]),
        )
        # 3b) body_size claims 100 bytes but section ends after 3 bytes.
        trunc_body_past_sec = b"\x00asm\x01\x00\x00\x00" + _wasm_section(
            bgw.WASM_SECTION_CODE,
            bytes([0x01, 0x64, 0x00, 0x0B]),
        )
        # 3c) body_size=2 but local_decl_count=5 overflows body_end (even with a 2nd body after!).
        trunc_locals_past_body = b"\x00asm\x01\x00\x00\x00" + _wasm_section(
            bgw.WASM_SECTION_CODE,
            bytes([0x02, 0x02, 0x05, 0x01, 0x02, 0x00, 0x0B]),
        )
        # 3d) body_size=0 cannot even hold local_decl_count.
        trunc_zero_body = b"\x00asm\x01\x00\x00\x00" + _wasm_section(
            bgw.WASM_SECTION_CODE,
            bytes([0x01, 0x00]),
        )
        for bad_label, bad_wasm in (
            ("missing_body", trunc_missing_body),
            ("body_past_sec", trunc_body_past_sec),
            ("locals_past_body", trunc_locals_past_body),
            ("zero_body", trunc_zero_body),
        ):
            with self.assertRaises(bgw.WasmInspectError, msg=bad_label):
                bgw.inspect_wasm_bytes(bad_wasm, name=f"{bad_label}.wasm")

    def test_v8_headroom_calculations_warnings_and_manifest_emission(self) -> None:
        self.assertEqual(bgw.WASM_SECTION_CODE, 10)
        self.assertEqual(bgw.V8_MAX_WASM_FUNCTION_BYTES, 7_654_321)
        self.assertEqual(bgw.V8_MAX_WASM_FUNCTION_LOCALS, 50_000)
        self.assertEqual(bgw.LARGE_FUNC_BODY_THRESHOLD_BYTES, 1024 * 1024)

        # 1) Realistic comfortable #663 shape: 890 MiB side module, 1.70 MB func, 1,306 locals, 4 >1 MiB.
        m_main = bgw.WasmModuleMetrics(
            name="runtime.wasm",
            role="main",
            wasm_bytes=12 * 1024 * 1024,
            table_size=64,
            code_func_count=1200,
            max_func_body_bytes=64_000,
            max_func_locals=120,
            large_func_bodies_over_1mib=0,
        )
        m_side0 = bgw.WasmModuleMetrics(
            name="guest_part_0.wasm",
            role="side",
            wasm_bytes=890 * 1024 * 1024,
            table_size=5039,
            got_func_imports=0,
            code_func_count=412_000,
            max_func_body_bytes=1_700_000,
            max_func_locals=1306,
            large_func_bodies_over_1mib=4,
        )
        m_reg = bgw.WasmModuleMetrics(
            name="guest_part_1.wasm",
            role="side",
            wasm_bytes=18 * 1024 * 1024,
            table_size=0,
            got_func_imports=5039,
            code_func_count=10,
            max_func_body_bytes=1024,
            max_func_locals=4,
            large_func_bodies_over_1mib=0,
        )
        build_ok = bgw.WasmBuildMetrics(indirect_dispatch="shard", modules=(m_main, m_side0, m_reg))

        self.assertEqual(m_side0.v8_module_headroom_bytes, 134 * 1024 * 1024)
        self.assertEqual(m_side0.v8_module_headroom_mib, 134.0)
        self.assertAlmostEqual(m_side0.v8_module_used_pct, 86.91, places=2)
        self.assertEqual(m_side0.v8_func_body_headroom_bytes, 7_654_321 - 1_700_000)
        self.assertEqual(m_side0.v8_func_locals_headroom, 50_000 - 1306)
        self.assertFalse(m_side0.exceeds_v8_module_limit)
        self.assertFalse(m_side0.exceeds_v8_func_size_limit)
        self.assertFalse(m_side0.exceeds_v8_locals_limit)

        self.assertEqual(build_ok.max_module_name, "guest_part_0.wasm")
        self.assertEqual(build_ok.max_module_bytes, 890 * 1024 * 1024)
        self.assertEqual(build_ok.min_module_headroom_mib, 134.0)
        self.assertEqual(build_ok.max_func_body_bytes, 1_700_000)
        self.assertEqual(build_ok.max_func_locals, 1306)
        self.assertEqual(build_ok.large_func_bodies_over_1mib, 4)
        self.assertEqual(build_ok.total_code_func_count, 413_210)
        self.assertEqual(build_ok.total_table_size, 5103)
        self.assertTrue(build_ok.within_all_limits)
        self.assertEqual(bgw.check_v8_headroom_warnings(build_ok), [])

        # Verify both snake_case and camelCase dicts carry full v8 headroom fields.
        d = build_ok.to_dict()
        self.assertEqual(d["v8_headroom"]["max_module_name"], "guest_part_0.wasm")
        self.assertEqual(d["v8_headroom"]["min_module_headroom_mib"], 134.0)
        self.assertEqual(d["v8_headroom"]["v8_module_limit_mib"], 1024.0)
        self.assertEqual(d["v8_headroom"]["v8_func_body_limit_bytes"], 7_654_321)
        self.assertEqual(d["v8_headroom"]["v8_func_locals_limit"], 50_000)
        self.assertEqual(d["v8_headroom"]["large_func_bodies_over_1mib"], 4)
        self.assertEqual(d["v8_headroom"]["total_code_func_count"], 413_210)
        self.assertTrue(d["v8_headroom"]["within_all_limits"])

        md = build_ok.to_manifest_dict()
        self.assertEqual(md["v8Headroom"]["maxModuleName"], "guest_part_0.wasm")
        self.assertEqual(md["v8Headroom"]["minModuleHeadroomMiB"], 134.0)
        self.assertEqual(md["v8Headroom"]["v8ModuleLimitMiB"], 1024.0)
        self.assertEqual(md["v8Headroom"]["maxFuncBodyBytes"], 1_700_000)
        self.assertEqual(md["v8Headroom"]["maxFuncLocals"], 1306)
        self.assertEqual(md["v8Headroom"]["largeFuncBodiesOver1MiB"], 4)
        self.assertEqual(md["v8Headroom"]["totalCodeFuncCount"], 413_210)
        self.assertEqual(md["v8Headroom"]["totalGotFuncImports"], 5039)
        self.assertEqual(md["v8Headroom"]["totalTableSize"], 5103)
        self.assertTrue(md["v8Headroom"]["withinAllLimits"])
        self.assertEqual(md["modules"][1]["codeFuncCount"], 412_000)
        self.assertEqual(md["modules"][1]["v8ModuleHeadroomMiB"], 134.0)
        self.assertEqual(md["modules"][1]["v8FuncLocalsHeadroom"], 50_000 - 1306)

        formatted = "\n".join(bgw.format_wasm_metrics_lines(build_ok))
        self.assertIn("V8 headroom [OK]:", formatted)
        self.assertIn("guest_part_0.wasm=890.00 / 1024.00 MiB", formatted)
        self.assertIn("headroom 134.00 MiB", formatted)
        self.assertIn("1,700,000 B", formatted)
        self.assertIn("1,306 / 50,000", formatted)
        self.assertIn("bodies >1 MiB=4", formatted)

        # 2) Approaching limits (>=90% module, >=75% func body, >=80% locals) trigger proactive warnings.
        m_approach = bgw.WasmModuleMetrics(
            name="guest_part_0.wasm",
            role="side",
            wasm_bytes=950 * 1024 * 1024,  # 92.8% of 1024 MiB
            code_func_count=10,
            max_func_body_bytes=6_000_000,  # 78.4% of 7,654,321
            max_func_locals=42_000,  # 84.0% of 50,000
            large_func_bodies_over_1mib=1,
        )
        build_approach = bgw.WasmBuildMetrics(indirect_dispatch="shard", modules=(m_approach,))
        self.assertTrue(build_approach.within_all_limits)
        w_approach = bgw.check_v8_headroom_warnings(build_approach)
        self.assertEqual(len(w_approach), 3, w_approach)
        self.assertTrue(any("approaching Chrome/V8's 1024 MiB module cap" in w for w in w_approach))
        self.assertTrue(any("kV8MaxWasmFunctionSize" in w and "6,000,000" in w for w in w_approach))
        self.assertTrue(any("kV8MaxWasmFunctionLocals" in w and "42,000" in w for w in w_approach))

        # 3) Exceeding limits marks within_all_limits=False and emits exceeding warnings.
        m_exceed = bgw.WasmModuleMetrics(
            name="guest_part_0.wasm",
            role="side",
            wasm_bytes=bgw.V8_MAX_WASM_MODULE_BYTES + 2 * 1024 * 1024,
            code_func_count=1,
            max_func_body_bytes=bgw.V8_MAX_WASM_FUNCTION_BYTES + 100,
            max_func_locals=bgw.V8_MAX_WASM_FUNCTION_LOCALS + 5,
            large_func_bodies_over_1mib=1,
        )
        build_exceed = bgw.WasmBuildMetrics(indirect_dispatch="shard", modules=(m_exceed,))
        self.assertFalse(build_exceed.within_all_limits)
        w_exceed_all = bgw.check_v8_headroom_warnings(build_exceed, include_module_exceeded=True)
        self.assertEqual(len(w_exceed_all), 3, w_exceed_all)
        w_exceed_dedup = bgw.check_v8_headroom_warnings(build_exceed, include_module_exceeded=False)
        self.assertEqual(len(w_exceed_dedup), 2, w_exceed_dedup)
        self.assertIn("V8 headroom [OVER LIMIT]:", "\n".join(bgw.format_wasm_metrics_lines(build_exceed)))

        # 4) Empty modules edge case does not crash.
        empty_metrics = bgw.WasmBuildMetrics(indirect_dispatch="table", modules=())
        self.assertIsNone(empty_metrics.max_module_name)
        self.assertEqual(empty_metrics.min_module_headroom_bytes, bgw.V8_MAX_WASM_MODULE_BYTES)
        self.assertTrue(empty_metrics.within_all_limits)

        # 5) End-to-end run_build() writes wasmMetrics.v8Headroom into guest_manifest.json
        #    and appends proactive Code-section warnings without duplicating module warnings.
        big_func_wasm = (
            b"\x00asm\x01\x00\x00\x00"
            + _encode_dylink0_section(memory_size=4096, table_size=2, needed=("runtime.wasm",))
            + _encode_code_section(
                [
                    ([(41_000, 0x7F)], bgw.LARGE_FUNC_BODY_THRESHOLD_BYTES + 16),
                    ([(12, 0x7E)], 64),
                ]
            )
        )
        main_wasm = (
            b"\x00asm\x01\x00\x00\x00"
            + _encode_table_section([(0x70, 32, None, False)])
            + _encode_code_section([([(8, 0x7F)], 32)])
        )

        class CodeWasmRunner(RecordingRunner):
            def __call__(self, argv: Sequence[str]) -> bgw.CommandResult:
                res = super().__call__(argv)
                out_path = _dash_o(tuple(str(a) for a in argv))
                if res.returncode == 0 and out_path is not None:
                    p = Path(out_path)
                    if p.name == "guest_part_0.wasm":
                        p.write_bytes(big_func_wasm)
                    elif p.name == bgw.RUNTIME_JS:
                        p.with_suffix(".wasm").write_bytes(main_wasm)
                return res

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _fake_emsdk(repo)
            orig_lib = _touch_lib(repo / "runtime" / "build-wasm64" / "libplaygta6_core.a")
            src = root / "recomp-headroom"
            _write(src / "lift" / "shard-00000.ll", "; s0\n")
            _write(src / "lift" / "shard-00001.ll", "; s1\n")
            _write(src / "lift" / "functions.ll", "; reg\n")
            _write(src / "stubs" / "pg6_import_thunks.cpp", "void thunk() {}\n")
            out_dir = root / "dist"

            rep = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache",
                out_dir=out_dir,
                jobs=2,
                split_modules=2,
                split_strategy="asymmetric",
                collapse_single_side_module=True,
                runtime_lib=orig_lib,
                run_command=CodeWasmRunner(),
                which_fn=lambda _n: None,
            )
            self.assertTrue(rep.ok, rep.errors)
            self.assertTrue(
                any("41,000" in w and "kV8MaxWasmFunctionLocals" in w for w in rep.warnings),
                rep.warnings,
            )
            gm = json.loads((out_dir / bgw.GUEST_MANIFEST_FILE).read_text(encoding="utf-8"))
            v8h = gm["wasmMetrics"]["v8Headroom"]
            self.assertEqual(v8h["maxFuncBodyBytes"], bgw.LARGE_FUNC_BODY_THRESHOLD_BYTES + 16)
            self.assertEqual(v8h["maxFuncLocals"], 41_000)
            self.assertEqual(v8h["largeFuncBodiesOver1MiB"], 1)
            self.assertEqual(v8h["totalCodeFuncCount"], 3)
            self.assertTrue(v8h["withinAllLimits"])
            p0_manifest = next(m for m in gm["wasmMetrics"]["modules"] if m["name"] == "guest_part_0.wasm")
            self.assertEqual(p0_manifest["codeFuncCount"], 2)
            self.assertEqual(p0_manifest["maxFuncLocals"], 41_000)
            self.assertEqual(p0_manifest["largeFuncBodiesOver1MiB"], 1)
            self.assertEqual(p0_manifest["v8FuncLocalsHeadroom"], 9_000)
            self.assertNotIn("\\\\", bgw.report_to_json(rep))

    def test_stale_side_module_cleanup_collapse_dry_run_and_unrelated_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            # 1) Direct unit test of cleanup_stale_side_modules().
            d = root / "direct"
            d.mkdir()
            for fname in (
                "guest_part_0.wasm",
                "guest_part_1.wasm",
                "guest_part_2.wasm",
                "guest_part_10.wasm",
                "guest_part_3.wasm",
                "runtime.wasm",
                "custom.wasm",
                "guest_part_foo.wasm",
            ):
                (d / fname).write_bytes(b"\x00asm\x01\x00\x00\x00")
            dir_named = d / "guest_part_99.wasm"
            dir_named.mkdir()

            planned = bgw.cleanup_stale_side_modules(
                d, ["guest_part_0.wasm", "guest_part_1.wasm"], dry_run=True
            )
            self.assertEqual(planned, ["guest_part_2.wasm", "guest_part_3.wasm", "guest_part_10.wasm"])
            # Dry-run leaves every file intact.
            for fname in planned:
                self.assertTrue((d / fname).is_file())

            removed = bgw.cleanup_stale_side_modules(
                d, ["guest_part_0.wasm", "guest_part_1.wasm"], dry_run=False
            )
            self.assertEqual(removed, ["guest_part_2.wasm", "guest_part_3.wasm", "guest_part_10.wasm"])
            for fname in removed:
                self.assertFalse((d / fname).exists())
            for kept in ("guest_part_0.wasm", "guest_part_1.wasm", "runtime.wasm", "custom.wasm", "guest_part_foo.wasm"):
                self.assertTrue((d / kept).is_file(), kept)
            self.assertTrue(dir_named.is_dir())

            # 2) End-to-end --collapse-single-side-module build over prior 4-part output_dir.
            repo = root / "repo"
            _fake_emsdk(repo)
            orig_lib = _touch_lib(repo / "runtime" / "build-wasm64" / "libplaygta6_core.a")
            src = root / "recomp-stale"
            _write(src / "lift" / "shard-00000.ll", "; s0\n")
            _write(src / "lift" / "shard-00001.ll", "; s1\n")
            _write(src / "lift" / "functions.ll", "; reg\n")
            _write(src / "stubs" / "pg6_import_thunks.cpp", "void thunk() {}\n")

            out_dir = root / "dist"
            out_dir.mkdir()
            for idx in range(4):
                (out_dir / f"guest_part_{idx}.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00")
            (out_dir / "custom.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00")
            (out_dir / "guest_part_9.wasm").mkdir()

            # 2a) Dry-run reports stale files without deleting them.
            dry_rep = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache",
                out_dir=out_dir,
                jobs=2,
                split_modules=2,
                split_strategy="asymmetric",
                collapse_single_side_module=True,
                dry_run=True,
                runtime_lib=orig_lib,
                which_fn=lambda _n: None,
            )
            self.assertTrue(dry_rep.ok, dry_rep.errors)
            self.assertEqual(dry_rep.removed_stale_side_modules, ["guest_part_2.wasm", "guest_part_3.wasm"])
            self.assertTrue((out_dir / "guest_part_2.wasm").is_file())
            self.assertTrue((out_dir / "guest_part_3.wasm").is_file())
            self.assertIn(
                "Stale side modules (would remove): guest_part_2.wasm, guest_part_3.wasm",
                bgw.format_report_text(dry_rep),
            )
            dry_payload = json.loads(bgw.report_to_json(dry_rep))
            self.assertEqual(
                dry_payload["removed_stale_side_modules"],
                ["guest_part_2.wasm", "guest_part_3.wasm"],
            )

            # 2b) Failed side link preserves prior stale files on disk.
            fail_rep = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache",
                out_dir=out_dir,
                jobs=2,
                split_modules=2,
                split_strategy="asymmetric",
                collapse_single_side_module=True,
                runtime_lib=orig_lib,
                run_command=RecordingRunner(fail_outputs=frozenset({"guest_part_1.wasm"})),
                which_fn=lambda _n: None,
            )
            self.assertFalse(fail_rep.ok)
            self.assertEqual(fail_rep.removed_stale_side_modules, [])
            self.assertTrue((out_dir / "guest_part_2.wasm").is_file())
            self.assertTrue((out_dir / "guest_part_3.wasm").is_file())

            # 2c) Successful collapsed build removes stale guest_part_2/3, keeps active & custom.wasm.
            real_rep = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache",
                out_dir=out_dir,
                jobs=2,
                split_modules=2,
                split_strategy="asymmetric",
                collapse_single_side_module=True,
                runtime_lib=orig_lib,
                run_command=RecordingRunner(),
                which_fn=lambda _n: None,
            )
            self.assertTrue(real_rep.ok, real_rep.errors)
            self.assertEqual(real_rep.removed_stale_side_modules, ["guest_part_2.wasm", "guest_part_3.wasm"])
            self.assertTrue((out_dir / "guest_part_0.wasm").is_file())
            self.assertTrue((out_dir / "guest_part_1.wasm").is_file())
            self.assertFalse((out_dir / "guest_part_2.wasm").exists())
            self.assertFalse((out_dir / "guest_part_3.wasm").exists())
            self.assertTrue((out_dir / "custom.wasm").is_file())
            self.assertTrue((out_dir / "guest_part_9.wasm").is_dir())
            self.assertIn(
                "Removed stale side modules: guest_part_2.wasm, guest_part_3.wasm",
                bgw.format_report_text(real_rep),
            )

            # 2d) Subsequent monolithic build removes remaining guest_part_0/1, keeps custom.wasm.
            mono_rep = bgw.run_build(
                link_cache=False,
                link_jobs=1,
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache-mono",
                out_dir=out_dir,
                jobs=2,
                split_modules=1,
                runtime_lib=orig_lib,
                run_command=RecordingRunner(),
                which_fn=lambda _n: None,
            )
            self.assertTrue(mono_rep.ok, mono_rep.errors)
            self.assertEqual(mono_rep.removed_stale_side_modules, ["guest_part_0.wasm", "guest_part_1.wasm"])
            self.assertFalse((out_dir / "guest_part_0.wasm").exists())
            self.assertFalse((out_dir / "guest_part_1.wasm").exists())
            self.assertTrue((out_dir / "custom.wasm").is_file())
            self.assertTrue((out_dir / "runtime.wasm").is_file())


if __name__ == "__main__":
    unittest.main()
