#!/usr/bin/env python3
"""Unit and CLI tests for tools/build_native_boot.py (Track 1 / R14). Synthetic inputs only;
LLVM and CMake are replaced by a recording fake runner."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Sequence
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import build_native_boot as nb  # noqa: E402

WASM_SHARD = """; playgta6 recomp: x86-64 guest code lifted to LLVM IR (generated file)
target datalayout = "e-m:e-p:64:64-p10:8:8-p20:8:8-i64:64-i128:128-n32:64-S128-ni:1:10:20"
target triple = "wasm64-unknown-unknown"

define void @sub_140001000(ptr noalias %state) #0 {
  ret void
}

attributes #0 = { "target-features"="+simd128" }
"""

REGISTRATION = """; registration
target datalayout = "e-m:e-p:64:64-p10:8:8-p20:8:8-i64:64-i128:128-n32:64-S128-ni:1:10:20"
target triple = "wasm64-unknown-unknown"
"""


def make_module(root: Path, shards: int = 3, target: str | None = None, name: str = "main") -> Path:
    module = root / "out" / name
    (module / "lift").mkdir(parents=True)
    (module / "stubs").mkdir()
    (module / "image").mkdir()
    (module / "stubs" / "pg6_import_thunks.cpp").write_text("// thunks\n", encoding="utf-8")
    (module / "image" / "guest.pgim").write_bytes(b"PGIM")
    files = []
    shard_records = []
    for i in range(shards):
        rel = f"lift/shard-{i:05}.ll"
        (module / rel).write_text(WASM_SHARD.replace("140001000", f"14000{i}000"), encoding="utf-8")
        files.append({"path": rel, "size": 1, "fnv1a64": f"0x{i + 1:016x}"})
        shard_records.append({"index": i, "ll": rel, "obj": None, "instructions": 100 * (i + 1)})
    (module / "lift" / "functions.ll").write_text(REGISTRATION, encoding="utf-8")
    files.append({"path": "lift/functions.ll", "size": 1, "fnv1a64": "0x00000000000000ff"})
    options: dict = {"emit_obj": False}
    if target is not None:
        options["target"] = target
    manifest = {
        "schema": "playgta6-recomp-build",
        "schema_version": 1,
        "module": name,
        "options": options,
        "lift": {"shards": shard_records, "registration": {"ll": "lift/functions.ll"}},
        "files": files,
    }
    (module / nb.MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    return module


class FakeRunner:
    """Fake opt/llc/cmake: writes the `-o` outputs and records argv + stdin."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[tuple[list[str], bytes | None]] = []
        self.fail_on = fail_on
        self.lock = threading.Lock()

    def __call__(self, argv: Sequence[str], stdin: bytes | None) -> nb.CommandResult:
        argv = list(argv)
        with self.lock:
            self.calls.append((argv, stdin))
        if argv[-1] == "--version":
            return nb.CommandResult(0, "LLVM (http://llvm.org/):\n  LLVM version 99.0.0-test\n")
        if self.fail_on is not None and stdin is not None and self.fail_on.encode() in stdin:
            return nb.CommandResult(1, "", "error: bad things in the module\n")
        if "-o" in argv:
            out = Path(argv[argv.index("-o") + 1])
            out.write_bytes(b"OBJ" * 10)
        if "--build" in argv:
            build_dir = Path(argv[argv.index("--build") + 1])
            exe = build_dir.parent / "bin" / nb._exe(nb.HARNESS_OUTPUT)
            exe.parent.mkdir(parents=True, exist_ok=True)
            exe.write_bytes(b"EXE")
        return nb.CommandResult(0)


def fake_tools(root: Path) -> Path:
    bin_dir = root / "llvm"
    bin_dir.mkdir(exist_ok=True)
    for name in ("opt", "llc"):
        (bin_dir / nb._exe(name)).write_bytes(b"")
    return bin_dir


class RetargetTests(unittest.TestCase):
    def test_retarget_matches_the_lifter_rule(self) -> None:
        out = nb.retarget_ir(WASM_SHARD, "x86_64-pc-windows-msvc", "znver2")
        self.assertNotIn("target datalayout", out)
        self.assertIn('target triple = "x86_64-pc-windows-msvc"\n', out)
        self.assertIn('attributes #0 = { uwtable "target-cpu"="znver2" }\n', out)
        self.assertNotIn("simd128", out)
        # Everything else is untouched.
        self.assertIn("define void @sub_140001000(ptr noalias %state) #0 {\n  ret void\n}\n", out)
        self.assertEqual(nb.retarget_ir(out, "x86_64-pc-windows-msvc", "znver2"), out)

    def test_retarget_maps_the_wasm_swizzle(self) -> None:
        text = WASM_SHARD + (
            "declare <16 x i8> @llvm.wasm.swizzle(<16 x i8>, <16 x i8>)\n"
            "  %r = call <16 x i8> @llvm.wasm.swizzle(<16 x i8> %a, <16 x i8> %b)\n"
        )
        out = nb.retarget_ir(text, "x86_64-pc-windows-msvc", "x86-64-v3")
        self.assertNotIn("llvm.wasm.", out)
        self.assertIn("define internal <16 x i8> @__playgta6_swizzle(", out)
        self.assertIn("call <16 x i8> @__playgta6_swizzle(<16 x i8> %a, <16 x i8> %b)", out)
        self.assertIn("@llvm.x86.ssse3.pshuf.b.128", out)

    def test_retarget_gives_linkonce_globals_a_comdat(self) -> None:
        line = "@__playgta6_cmpxchg16b_locks = linkonce_odr hidden global [64 x i32] zeroinitializer, align 4\n"
        out = nb.retarget_ir(WASM_SHARD + line, "x86_64-pc-windows-msvc", "x86-64-v3")
        self.assertIn(
            "$__playgta6_cmpxchg16b_locks = comdat any\n"
            "@__playgta6_cmpxchg16b_locks = linkonce_odr hidden global [64 x i32] zeroinitializer, "
            "comdat, align 4\n",
            out,
        )
        self.assertEqual(nb.retarget_ir(out, "x86_64-pc-windows-msvc", "x86-64-v3"), out)

    def test_retarget_matches_the_rust_constant(self) -> None:
        rust = (REPO_ROOT / "recomp" / "src" / "lift" / "target.rs").read_text(encoding="utf-8")
        for line in nb.NATIVE_SWIZZLE_IR.splitlines():
            self.assertIn(line, rust)

    def test_retarget_keeps_crlf_bodies(self) -> None:
        text = WASM_SHARD.replace("\n", "\r\n")
        out = nb.retarget_ir(text, "x86_64-unknown-linux-gnu", "x86-64-v3")
        self.assertIn('target triple = "x86_64-unknown-linux-gnu"\n', out)
        self.assertIn("  ret void\r\n", out)


class LoadModuleTests(unittest.TestCase):
    def test_units_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module = make_module(Path(tmp), target="native:x86_64-unknown-linux-gnu")
            inputs = nb.load_module(module)
            self.assertEqual(inputs.module, "main")
            self.assertEqual(inputs.target, "native:x86_64-unknown-linux-gnu")
            self.assertEqual([u.name for u in inputs.units],
                             ["shard-00000", "shard-00001", "shard-00002", "functions"])
            self.assertEqual(inputs.units[1].manifest_hash, "0x0000000000000002")
            self.assertEqual(inputs.units[2].instructions, 300)
            self.assertEqual(inputs.pgim, module.resolve() / "image" / "guest.pgim")

    def test_rejects_bad_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(nb.NativeBootError):
                nb.load_module(Path(tmp))
            (Path(tmp) / nb.MANIFEST_FILE).write_text('{"schema": "other"}', encoding="utf-8")
            with self.assertRaises(nb.NativeBootError):
                nb.load_module(Path(tmp))

    def test_sample_spreads_over_shards(self) -> None:
        units = [nb.Unit(f"shard-{i:05}", Path(f"{i}.ll"), None) for i in range(10)]
        units.append(nb.Unit("functions", Path("f.ll"), None))
        self.assertEqual([u.name for u in nb.sample_units(units, 3)],
                         ["shard-00000", "shard-00003", "shard-00006"])
        self.assertEqual(len(nb.sample_units(units, 50)), 10)


class CompileTests(unittest.TestCase):
    def test_cache_key_changes_with_compiler_flags_and_toolchain(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runner = FakeRunner()
            unit = nb.Unit("shard-00000", Path("0.ll"), None)
            keys = {nb.cache_key(unit, self._cfg(root, runner, **kw), b"ir") for kw in
                    ({}, {"opt_level": "2"}, {"opt_level": "t"}, {"opt_level": "f"},
                     {"triple": "x86_64-unknown-linux-gnu"})}
            self.assertEqual(len(keys), 5)
    def _cfg(self, root: Path, runner: FakeRunner, **kw) -> nb.CompileConfig:
        tools = nb.find_llvm_tools(fake_tools(root), runner)
        return nb.CompileConfig(
            triple=kw.get("triple", "x86_64-pc-windows-msvc"),
            cpu="x86-64-v3",
            opt_level=kw.get("opt_level", "1"),
            cache_dir=root / "cache",
            log_dir=root / "logs",
            tools=tools,
            delete_ir=kw.get("delete_ir", False),
        )

    def test_same_content_units_compile_concurrently(self) -> None:
        # Two modules whose shards share a manifest hash share cache keys. Compiled on two workers
        # at once, one unit compiles and the other takes its cached object (before, both wrote
        # the same temporary and one failed to move it into the cache).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = nb.load_module(make_module(root, shards=1, name="a")).units[0]
            b = nb.load_module(make_module(root, shards=1, name="b")).units[0]
            self.assertEqual(a.manifest_hash, b.manifest_hash)
            inner = FakeRunner()

            def runner(argv: Sequence[str], stdin: bytes | None) -> nb.CommandResult:
                if Path(argv[0]).stem == "llc" and "--version" not in argv:
                    time.sleep(0.3)  # keep the first compile in flight while the second starts
                return inner(argv, stdin)

            cfg = self._cfg(root, inner)
            results = nb.compile_all([a, b], cfg, 2, runner)
            self.assertTrue(all(r.error is None and r.obj is not None and r.obj.is_file() for r in results))
            self.assertEqual(results[0].obj, results[1].obj)
            self.assertEqual(sorted(r.cached for r in results), [False, True])
            llc = [c for c in inner.calls if Path(c[0][0]).stem == "llc" and "--version" not in c[0]]
            self.assertEqual(len(llc), 1)
            self.assertEqual(sorted(p.name for p in cfg.cache_dir.iterdir()), [results[0].obj.name])

    def test_compiles_in_parallel_then_hits_the_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = nb.load_module(make_module(root, shards=6))
            runner = FakeRunner()
            cfg = self._cfg(root, runner)
            self.assertEqual(cfg.tools.version, "LLVM version 99.0.0-test")
            lines: list[str] = []
            results = nb.compile_all(inputs.units, cfg, 4, runner, progress=lines.append)
            self.assertEqual([r.unit.name for r in results], [u.name for u in inputs.units])
            self.assertTrue(all(r.obj is not None and r.obj.is_file() and not r.cached for r in results))
            self.assertTrue(all(r.obj.suffix == ".obj" for r in results))
            self.assertEqual(len(lines), 7)
            self.assertIn("eta", lines[-1])
            opt_calls = [c for c in runner.calls if Path(c[0][0]).stem == "opt"]
            llc_calls = [c for c in runner.calls if Path(c[0][0]).stem == "llc" and "--version" not in c[0]]
            self.assertEqual(len(opt_calls), 7)
            self.assertEqual(len(llc_calls), 7)
            # opt reads the retargeted IR from stdin.
            self.assertTrue(all(c[0][1:3] == ["-O1", "-"] for c in opt_calls))
            self.assertTrue(all("-O1" in c[0] for c in llc_calls))
            stdin = opt_calls[0][1].decode()
            self.assertIn('target triple = "x86_64-pc-windows-msvc"', stdin)
            self.assertNotIn("wasm64", stdin)
            self.assertIn("-mtriple=x86_64-pc-windows-msvc", llc_calls[0][0])
            self.assertIn("-mcpu=x86-64-v3", llc_calls[0][0])
            self.assertNotIn("-relocation-model=pic", llc_calls[0][0])
            self.assertFalse(any(p.name.endswith(".tmp.bc") for p in cfg.cache_dir.iterdir()))

            runner2 = FakeRunner()
            again = nb.compile_all(inputs.units, cfg, 4, runner2)
            self.assertTrue(all(r.cached for r in again))
            self.assertEqual(runner2.calls, [])

            # A different -O level is a different cache entry.
            cfg2 = self._cfg(root, runner2, opt_level="2")
            other = nb.compile_all(inputs.units[:1], cfg2, 1, runner2)
            self.assertFalse(other[0].cached)

    def test_linux_objects_are_pic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = nb.load_module(make_module(root, shards=1))
            runner = FakeRunner()
            cfg = self._cfg(root, runner, triple="x86_64-unknown-linux-gnu")
            results = nb.compile_all(inputs.units, cfg, 1, runner)
            self.assertTrue(all(r.obj.suffix == ".o" for r in results))
            llc = [c[0] for c in runner.calls if Path(c[0][0]).stem == "llc" and "--version" not in c[0]]
            self.assertTrue(all("-relocation-model=pic" in a for a in llc))

    def test_failures_are_reported_per_shard_with_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = nb.load_module(make_module(root, shards=4))
            runner = FakeRunner(fail_on="@sub_140002000")
            cfg = self._cfg(root, runner)
            results = nb.compile_all(inputs.units, cfg, 1, runner, keep_going=True)
            failed = [r for r in results if r.error]
            self.assertEqual([r.unit.name for r in failed], ["shard-00002"])
            self.assertIn("opt failed (exit 1): error: bad things", failed[0].error)
            log = cfg.log_dir / "shard-00002.opt.log"
            self.assertIn("bad things", log.read_text(encoding="utf-8"))

    def test_delete_ir_keeps_the_cache_usable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module = make_module(root, shards=2)
            inputs = nb.load_module(module)
            runner = FakeRunner()
            cfg = self._cfg(root, runner, delete_ir=True)
            nb.compile_all(inputs.units, cfg, 2, runner)
            self.assertFalse((module / "lift" / "shard-00000.ll").exists())
            again = nb.compile_all(inputs.units, cfg, 2, runner)
            self.assertTrue(all(r.cached for r in again))

    def test_delete_ir_also_on_a_cache_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module = make_module(root, shards=2)
            inputs = nb.load_module(module)
            runner = FakeRunner()
            nb.compile_all(inputs.units, self._cfg(root, runner), 2, runner)
            self.assertTrue((module / "lift" / "shard-00000.ll").exists())
            again = nb.compile_all(inputs.units, self._cfg(root, runner, delete_ir=True), 2, runner)
            self.assertTrue(all(r.cached for r in again))
            self.assertFalse((module / "lift" / "shard-00000.ll").exists())


class CacheTests(unittest.TestCase):
    KEY = "ab" * 32

    def test_default_dir_is_machine_wide_and_overridable(self) -> None:
        win = nb.default_cache_dir({"LOCALAPPDATA": r"C:\Users\u\AppData\Local"}, windows=True)
        self.assertEqual(win, Path(r"C:\Users\u\AppData\Local") / "playgta6" / "cache" / "objcache")
        self.assertEqual(nb.default_cache_dir({"XDG_CACHE_HOME": "/x/cache"}, windows=False),
                         Path("/x/cache") / "playgta6" / "cache" / "objcache")
        self.assertEqual(nb.default_cache_dir({}, windows=False), Path.home() / ".cache" / "playgta6" / "cache" / "objcache")
        env = {nb.CACHE_ENV: "/elsewhere", "LOCALAPPDATA": "C:/l"}
        self.assertEqual(nb.default_cache_dir(env, windows=True), Path("/elsewhere"))
        # Never inside the repository.
        self.assertNotIn(REPO_ROOT, nb.default_cache_dir().resolve().parents)
        self.assertEqual(nb.default_cache_max_gb({}), 40.0)
        self.assertEqual(nb.default_cache_max_gb({nb.CACHE_MAX_ENV: "2.5"}), 2.5)
        with self.assertRaises(nb.NativeBootError):
            nb.default_cache_max_gb({nb.CACHE_MAX_ENV: "lots"})

    def test_key_to_path(self) -> None:
        root = Path("cache")
        self.assertEqual(nb.cache_entry_path(root, self.KEY, ".obj"), root / f"{self.KEY}.obj")
        self.assertEqual(nb.cache_entry_path(root, self.KEY, ".o"), root / f"{self.KEY}.o")
        for key in ("ab" * 31, "AB" * 32, "../" + "a" * 61):
            with self.assertRaises(ValueError):
                nb.cache_entry_path(root, key, ".obj")
        with self.assertRaises(ValueError):
            nb.cache_entry_path(root, self.KEY, ".tmp.obj")
        # The key covers the retargeting code, not just its version number.
        self.assertRegex(nb.RETARGET_FINGERPRINT, r"^[0-9a-f]{16}$")

    def test_two_writers_publish_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root / f"{self.KEY}.obj"
            temps = [root / f"{self.KEY}.{i}.tmp.obj" for i in range(2)]
            for i, t in enumerate(temps):
                t.write_bytes(b"OBJ" * 1000)
            barrier = threading.Barrier(2)
            won: list[bool] = []

            def writer(t: Path) -> None:
                barrier.wait()
                won.append(nb.publish(t, dest))

            threads = [threading.Thread(target=writer, args=(t,)) for t in temps]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(dest.read_bytes(), b"OBJ" * 1000)
            self.assertGreaterEqual(won.count(True), 1)
            self.assertEqual([p.name for p in root.iterdir()], [dest.name])  # no temporaries left
            # A late writer never replaces a published entry (it may be being linked).
            late = root / "late.tmp.obj"
            late.write_bytes(b"NEW")
            self.assertFalse(nb.publish(late, dest))
            self.assertEqual(dest.read_bytes(), b"OBJ" * 1000)
            self.assertFalse(late.exists())

    def test_two_processes_compile_a_key_once(self) -> None:
        # Two build processes (two checkouts) compile the same unit into one shared cache: one
        # holds the entry's lock and compiles, the other waits and takes the published object.
        script = textwrap.dedent(
            """
            import sys, time
            from pathlib import Path
            sys.path.insert(0, sys.argv[1])
            import build_native_boot as nb
            root, cache, marks = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
            nb.LOCK_POLL_S = 0.05
            def runner(argv, stdin):
                if argv[-1] == "--version":
                    return nb.CommandResult(0, "LLVM version 99.0.0-test")
                if Path(argv[0]).stem == "llc":
                    (marks / f"llc-{sys.argv[5]}").write_text("x")
                    time.sleep(0.5)
                out = Path(argv[argv.index("-o") + 1])
                out.write_bytes(b"OBJ" * 10)
                return nb.CommandResult(0)
            unit = nb.load_module(root / "out" / "main").units[0]
            cfg = nb.CompileConfig("x86_64-pc-windows-msvc", "x86-64-v3", "1", cache, root / "logs",
                                   nb.find_llvm_tools(root / "llvm", runner))
            res = nb.compile_unit(unit, cfg, runner)
            assert res.error is None and res.obj.is_file(), res.error
            print("cached" if res.cached else "compiled")
            """
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_module(root, shards=1)
            fake_tools(root)
            cache, marks = root / "cache", root / "marks"
            cache.mkdir()
            marks.mkdir()
            procs = [
                subprocess.Popen(
                    [sys.executable, "-c", script, str(REPO_ROOT / "tools"), str(root), str(cache), str(marks), str(i)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                for i in range(2)
            ]
            outs = [p.communicate(timeout=60) for p in procs]
            for p, (_, err) in zip(procs, outs):
                self.assertEqual(p.returncode, 0, err)
            self.assertEqual(sorted(o.strip() for o, _ in outs), ["cached", "compiled"])
            self.assertEqual(len(list(marks.iterdir())), 1)
            self.assertEqual([p.suffix for p in cache.iterdir()], [".obj"])

    def test_stale_lock_is_broken(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / f"{self.KEY}.obj.lock"
            self.assertTrue(nb.try_lock(lock))
            self.assertFalse(nb.try_lock(lock))  # held
            nb._HEARTBEAT.discard(lock)  # its owner "dies": no more heartbeats
            old = time.time() - nb.LOCK_STALE_S - 10
            os.utime(lock, (old, old))
            self.assertFalse(nb.try_lock(lock))  # breaks the stale lock...
            self.assertTrue(nb.try_lock(lock))  # ...so the next attempt takes it
            nb.release_lock(lock)
            self.assertFalse(lock.exists())

    def test_trim_removes_least_recently_used_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = 1_000_000_000.0
            day = 86_400.0
            paths = []
            for i in range(5):  # entry i was last used i days ago, 100 bytes each
                p = root / f"{i:064x}.obj"
                p.write_bytes(b"x" * 100)
                os.utime(p, (now - i * day, now - 10 * day))
                paths.append(p)
            # A hit refreshes the access time only: entry 4 becomes the most recently used.
            os.utime(paths[4], (now - 0.5 * day, now - 20 * day))
            stale_tmp = root / f"{0:064x}.abc.tmp.obj"
            stale_tmp.write_bytes(b"t")
            os.utime(stale_tmp, (now - 2 * day, now - 2 * day))
            fresh_lock = root / f"{1:064x}.obj.lock"
            fresh_lock.write_text("1")
            os.utime(fresh_lock, (now, now))

            res = nb.trim_cache(root, 250, grace_s=3600, now=now)
            self.assertEqual((res.entries, res.bytes_before, res.removed, res.bytes_after), (5, 500, 3, 200))
            self.assertEqual(sorted(p.name for p in root.iterdir()),
                             sorted([paths[0].name, paths[4].name, fresh_lock.name]))
            # Under the cap: nothing to do.
            self.assertEqual(nb.trim_cache(root, 250, grace_s=3600, now=now).removed, 0)
            # Entries used within the grace period survive even over the cap.
            res = nb.trim_cache(root, 0, grace_s=2 * day, now=now)
            self.assertEqual((res.removed, res.recent_kept), (0, 2))

    def test_mark_used_keeps_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / f"{self.KEY}.obj"
            p.write_bytes(b"x")
            os.utime(p, (1000.0, 1000.0))
            nb.mark_used(p)
            st = p.stat()
            self.assertEqual(st.st_mtime, 1000.0)
            self.assertGreater(st.st_atime, 1000.0)

    def test_cli_cache_trim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            cache.mkdir()
            old = cache / f"{1:064x}.obj"
            old.write_bytes(b"x" * 2048)
            os.utime(old, (1000.0, 1000.0))
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = nb.main(["--cache-trim", "--cache-dir", str(cache), "--cache-max-gb", "0.000001", "--json"])
            self.assertEqual(code, 0, err.getvalue())
            self.assertFalse(old.exists())
            self.assertEqual(json.loads(out.getvalue())["cache"]["trim"]["removed"], 1)


class HarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        # CLI runs without --cache-dir use the machine-wide cache: point it at a temporary.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = mock.patch.dict(os.environ, {nb.CACHE_ENV: str(Path(tmp.name) / "objcache")})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_cmakelists_links_objects_at_a_high_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = nb.module_inputs(nb.load_module(make_module(root, shards=1)))
            objs = [root / "cache" / "a.obj", root / "cache" / "b.obj"]
            text = nb.harness_cmakelists(REPO_ROOT, inputs, objs, root / "bin", debug_info=False)
            self.assertIn("add_subdirectory(", text)
            self.assertIn("EXCLUDE_FROM_ALL", text)
            self.assertIn("boot_main.cpp", text)
            self.assertIn("pg6_import_thunks.cpp", text)
            self.assertIn("PG6_BOOT_LINKED_GUEST PG6_BOOT_MODULE=main", text)
            self.assertIn("EXTERNAL_OBJECT TRUE", text)
            self.assertIn(objs[1].resolve().as_posix(), text)
            self.assertIn("/BASE:0x7E0000000000", text)
            self.assertIn("/DYNAMICBASE:NO", text)
            self.assertIn("/INCREMENTAL:NO", text)
            # Parallel cl within each MSBuild project (runtime included: set before add_subdirectory).
            self.assertLess(text.index("add_compile_options(/MP)"), text.index("add_subdirectory("))
            self.assertIn("/DEBUG:NONE", text)
            self.assertIn("/DEBUG\n", nb.harness_cmakelists(REPO_ROOT, inputs, objs, root / "bin", True).replace(" /DYNAMICBASE", "\n"))

    def test_cli_end_to_end_with_fakes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module = make_module(root, shards=3, target="native:x86_64-pc-windows-msvc")
            llvm = fake_tools(root)
            runner = FakeRunner()
            out, err = io.StringIO(), io.StringIO()
            original = nb.find_cmake
            nb.find_cmake = lambda: "cmake"
            try:
                with redirect_stdout(out), redirect_stderr(err):
                    code = nb.main([str(module), "--llvm-bin", str(llvm), "--jobs", "2", "--json"], runner)
            finally:
                nb.find_cmake = original
            self.assertEqual(code, 0, err.getvalue())
            summary = json.loads(out.getvalue())
            self.assertEqual(summary["units"], 4)
            self.assertEqual(summary["compiled"], 4)
            self.assertEqual(summary["failed"], [])
            self.assertTrue(summary["harness"].endswith(nb._exe(nb.HARNESS_OUTPUT)))
            self.assertIn("run: ", err.getvalue())
            cmake_calls = [c[0] for c in runner.calls if c[0][0] == "cmake"]
            # configure + runtime prebuild (overlapped with the shard compile), then configure + harness build
            self.assertEqual(len(cmake_calls), 4)
            self.assertIn("playgta6_core", cmake_calls[1])
            self.assertIn(nb.HARNESS_TARGET, cmake_calls[3])
            self.assertTrue((module / "native" / "harness" / "CMakeLists.txt").is_file())

    def test_cli_sample_extrapolates_without_linking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module = make_module(root, shards=8)
            runner = FakeRunner()
            out = io.StringIO()
            with redirect_stdout(out), redirect_stderr(io.StringIO()):
                code = nb.main([str(module), "--llvm-bin", str(fake_tools(root)), "--sample", "2",
                                "--json"], runner)
            self.assertEqual(code, 0)
            summary = json.loads(out.getvalue())
            self.assertEqual(summary["units"], 2)
            self.assertIn("extrapolated", summary)
            self.assertNotIn("harness", summary)
            self.assertFalse(any(c[0][0] == "cmake" for c in runner.calls))

    def test_cli_builds_a_program(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = "native:x86_64-pc-windows-msvc"
            make_module(root, shards=2, target=target, name="msvcp140")
            make_module(root, shards=3, target=target, name="main")
            program = root / "out"
            (program / "pg6_program.cpp").write_text("// descriptor\n", encoding="utf-8")
            (program / "exports").mkdir()
            (program / "exports" / "pg6_export_thunks.cpp").write_text("// exports\n", encoding="utf-8")
            manifest = {
                "schema": "playgta6-recomp-program",
                "schema_version": 1,
                "modules": [{"module": "msvcp140"}, {"module": "main"}],
            }
            (program / nb.PROGRAM_MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
            inputs = nb.load_inputs(program)
            self.assertEqual(len(inputs.units), 3 + 4)
            self.assertEqual(inputs.units[0].name, "msvcp140/shard-00000")
            self.assertEqual(inputs.defines, ["PG6_BOOT_PROGRAM"])
            self.assertIn("--program-dir", inputs.run_args)

            runner = FakeRunner()
            original = nb.find_cmake
            nb.find_cmake = lambda: "cmake"
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    code = nb.main([str(program), "--llvm-bin", str(fake_tools(root)), "--jobs", "3"], runner)
            finally:
                nb.find_cmake = original
            self.assertEqual(code, 0)
            text = (program / "native" / "harness" / "CMakeLists.txt").read_text(encoding="utf-8")
            self.assertIn("pg6_program.cpp", text)
            self.assertIn("msvcp140/stubs/pg6_import_thunks.cpp", text)
            self.assertIn("main/stubs/pg6_import_thunks.cpp", text)
            self.assertIn("exports/pg6_export_thunks.cpp", text)  # R18: the program's export table
            self.assertIn("PRIVATE PG6_BOOT_PROGRAM", text)
            self.assertEqual(text.count("EXTERNAL_OBJECT"), 1)
            self.assertEqual(sum(1 for line in text.splitlines() if line.strip().endswith('.obj"')), 2 * 7)

            # Modules built for different targets are refused.
            make_module(root / "mixed", shards=1, target=None, name="a")
            make_module(root / "mixed", shards=1, target=target, name="main")
            (root / "mixed" / "out" / nb.PROGRAM_MANIFEST_FILE).write_text(
                json.dumps({"schema": "playgta6-recomp-program", "modules": [{"module": "a"}, {"module": "main"}]}),
                encoding="utf-8",
            )
            with self.assertRaises(nb.NativeBootError):
                nb.load_inputs(root / "mixed" / "out")

    def test_cli_reports_errors(self) -> None:
        err = io.StringIO()
        with redirect_stderr(err):
            code = nb.main([str(REPO_ROOT / "does-not-exist")])
        self.assertEqual(code, 2)
        self.assertIn("error:", err.getvalue())


class BuildInfoTests(unittest.TestCase):
    def test_header_and_json_written_beside_the_harness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = nb.module_inputs(nb.load_module(make_module(root, shards=1)))
            info = nb.build_info(REPO_ROOT, inputs, debug_info=True, native_executor=True)
            self.assertEqual(info["profile"], "RelWithDebInfo")
            self.assertTrue(info["executor"])
            self.assertTrue(info["recomp_hash"])
            project, bin_dir = root / "harness", root / "bin"
            project.mkdir()
            nb.write_build_info(project, bin_dir, info)
            header = (project / nb.BUILD_INFO_HEADER).read_text(encoding="utf-8")
            self.assertIn("PG6_BUILD_INFO_JSON", header)
            self.assertEqual(json.loads((bin_dir / nb.BUILD_INFO_JSON).read_text(encoding="utf-8")), info)
            self.assertIn("CMAKE_CURRENT_SOURCE_DIR", nb.harness_cmakelists(REPO_ROOT, inputs, [], bin_dir, False))


if __name__ == "__main__":
    unittest.main()
