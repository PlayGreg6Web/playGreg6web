#!/usr/bin/env python3
"""Translated CPU window end to end (O29, local only).

Builds tests/window_e2e/guest.c - a synthetic x64 program that reserves the title-shaped 16 GiB
CPU window through XMem, maps pool pages at window slots that are and are not host-adjacent,
accesses them from guest code (straddling loads and stores) and hands window buffers to host
shims (memcpy, strlen) - into a PE with MSVC, lifts it with
`playgta6-recomp build-program --memory-model window` and boots it under the runtime, where the
generated program descriptor selects the memory model:

* wasm64: the program is linked with the wasm test build's runtime library
  (`runtime/build-wasm64-tests/libplaygta6_core.a`, made by `tools/run_wasm_tests.py`) and
  tests/window_e2e/e2e_main.cpp inside the `emscripten/emsdk` image, and run under `node:24`;
* native: `tools/build_native_boot.py` links the native harness, which boots it.

Each passes when the guest returns 0x600D. Needs Windows with Visual Studio (cl, lib, link),
Docker, and a built `playgta6-recomp` (`--recomp`, default target/release). Synthetic data only.

    python tools/run_window_e2e.py [--out work/window-e2e] [--targets wasm64 native] [--jobs N]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCES = REPO / "tests" / "window_e2e"
EMSDK_IMAGE = "emscripten/emsdk:4.0.10"
NODE_IMAGE = "node:24"
EMCACHE_VOLUME = "playgta6-wasm-emcache"
PASS_CODE = 0x600D

LINK_SH = r"""#!/usr/bin/env bash
set -eu
B=/src/runtime/build-wasm64-tests
P=/e2e/prog-wasm
O=/e2e/wasm-out
mkdir -p "$O"
CXX=(-std=c++20 -O2 -pthread -sMEMORY64=1 -fexceptions -I/src/runtime -I"$P" -I"$P/main/stubs"
     -DPLAYGTA6_PROCESS_GUEST_SPACE_SIZE=4294967296ULL)
for ll in "$P"/main/lift/*.ll; do
  em++ -O2 -pthread -sMEMORY64=1 -c "$ll" -o "$O/$(basename "$ll" .ll).o" 2>&1 | grep -v "overriding the module target triple" || true
done
em++ "${CXX[@]}" -c "$P/main/stubs/pg6_import_thunks.cpp" -o "$O/imports.o"
em++ "${CXX[@]}" -I"$P/exports" -c "$P/exports/pg6_export_thunks.cpp" -o "$O/exports.o"
em++ "${CXX[@]}" -c "$P/pg6_program.cpp" -o "$O/program.o"
em++ "${CXX[@]}" /src/tests/window_e2e/e2e_main.cpp "$O"/*.o "$B/libplaygta6_core.a" \
  -sPTHREAD_POOL_SIZE=24 -sPROXY_TO_PTHREAD=1 -sEXIT_RUNTIME=1 -sSTACK_SIZE=1MB -sNODERAWFS=1 \
  -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=16GB -Wl,--wrap=sbrk -sENVIRONMENT=node \
  -sERROR_ON_UNDEFINED_SYMBOLS=1 --js-library=/src/tools/wasm_test_support/host_imports.js \
  -o "$O/e2e.js"
"""


def run(argv: list[str], **kw) -> subprocess.CompletedProcess:
    print("$", " ".join(str(a) for a in argv), flush=True)
    return subprocess.run([str(a) for a in argv], text=True, capture_output=True, **kw)


def vcvars() -> Path:
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
        "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.is_file():
        r = run([vswhere, "-latest", "-products", "*", "-property", "installationPath"])
        root = r.stdout.strip().splitlines()
        if root:
            bat = Path(root[0]) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
            if bat.is_file():
                return bat
    raise SystemExit("Visual Studio (vcvars64.bat) not found: the guest is built with MSVC")


def build_guest(out: Path) -> Path:
    work = out / "guest"
    work.mkdir(parents=True, exist_ok=True)
    for name in ("guest.c", "xmem.def", "vcruntime140.def", "crtstring.def"):
        shutil.copy2(SOURCES / name, work / name)
    bat = work / "build.bat"
    bat.write_text(
        "@echo off\r\n"
        f'call "{vcvars()}" >nul 2>&1\r\n'
        f'cd /d "{work}"\r\n'
        "lib /nologo /def:xmem.def /machine:x64 /out:xmem.lib || exit /b 1\r\n"
        "lib /nologo /def:vcruntime140.def /machine:x64 /out:vcruntime140.lib || exit /b 1\r\n"
        "lib /nologo /def:crtstring.def /machine:x64 /out:crtstring.lib || exit /b 1\r\n"
        "cl /nologo /c /O1 /GS- /Oi- /Zl guest.c || exit /b 1\r\n"
        "link /nologo /ENTRY:entry /SUBSYSTEM:CONSOLE /NODEFAULTLIB /DYNAMICBASE /OUT:guest.exe "
        "guest.obj xmem.lib vcruntime140.lib crtstring.lib || exit /b 1\r\n"
    )
    r = run(["cmd", "/c", bat])
    if r.returncode != 0:
        raise SystemExit(f"guest build failed:\n{r.stdout}{r.stderr}")
    return work / "guest.exe"


def build_program(recomp: Path, guest: Path, out: Path, jobs: int, native: bool) -> None:
    argv = [recomp, "build-program", guest, "--out", out, "--jobs", jobs, "--memory-model", "window"]
    if native:
        argv += ["--target", "native"]
    r = run(argv, cwd=REPO)
    if r.returncode != 0:
        raise SystemExit(f"build-program failed:\n{r.stdout}{r.stderr}")


def docker_path(p: Path) -> str:
    return str(p.resolve()).replace("\\", "/")


def run_wasm(recomp: Path, guest: Path, out: Path, jobs: int) -> bool:
    lib = REPO / "runtime" / "build-wasm64-tests" / "libplaygta6_core.a"
    if not lib.is_file():
        raise SystemExit("missing runtime/build-wasm64-tests/libplaygta6_core.a: run tools/run_wasm_tests.py first")
    prog = out / "prog-wasm"
    shutil.rmtree(prog, ignore_errors=True)
    build_program(recomp, guest, prog, jobs, native=False)
    (out / "link.sh").write_text(LINK_SH, newline="\n")
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    r = run(["docker", "run", "--rm", "-v", f"{EMCACHE_VOLUME}:/emsdk/upstream/emscripten/cache",
             "-v", f"{docker_path(REPO)}:/src:ro", "-v", f"{docker_path(out)}:/e2e", EMSDK_IMAGE,
             "bash", "/e2e/link.sh"], env=env)
    if r.returncode != 0:
        print(r.stdout[-4000:], r.stderr[-4000:])
        return False
    r = run(["docker", "run", "--rm", "-v", f"{docker_path(out)}:/e2e", "-w", "/e2e", NODE_IMAGE,
             "node", "wasm-out/e2e.js", "prog-wasm/main/image/guest.pgim"], env=env)
    print(r.stdout[-3000:], r.stderr[-2000:])
    return r.returncode == 0 and re.search(r"exit code 0x600d", r.stdout) is not None


def run_native(recomp: Path, guest: Path, out: Path, jobs: int) -> bool:
    prog = out / "prog-native"
    shutil.rmtree(prog, ignore_errors=True)
    build_program(recomp, guest, prog, jobs, native=True)
    r = run([sys.executable, REPO / "tools" / "build_native_boot.py", prog, "--out", out / "native", "--jobs", jobs])
    if r.returncode != 0:
        print(r.stdout[-4000:], r.stderr[-4000:])
        return False
    exe = out / "native" / "bin" / ("playgta6_boot.exe" if os.name == "nt" else "playgta6_boot")
    report = out / "native-report.json"
    r = run([exe, "--program-dir", prog, "--name", "guest.exe", "--report", report, "--watchdog-ms", "60000"])
    print(r.stdout[-2000:], r.stderr[-2000:])
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    print("memory:", json.dumps(data.get("memory")))
    return data.get("exit_code") == PASS_CODE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=REPO / "work" / "window-e2e")
    parser.add_argument("--recomp", type=Path,
                        default=REPO / "target" / "release" / ("playgta6-recomp.exe" if os.name == "nt" else "playgta6-recomp"))
    parser.add_argument("--targets", nargs="+", default=["wasm64", "native"], choices=["wasm64", "native"])
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    guest = build_guest(args.out)
    results = {}
    if "wasm64" in args.targets:
        results["wasm64"] = run_wasm(args.recomp, guest, args.out, args.jobs)
    if "native" in args.targets:
        results["native"] = run_native(args.recomp, guest, args.out, args.jobs)
    for target, ok in results.items():
        print(f"{target}: {'PASS' if ok else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
