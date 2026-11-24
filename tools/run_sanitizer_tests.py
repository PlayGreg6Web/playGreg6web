#!/usr/bin/env python3
"""Run the native runtime test suite under a sanitizer in Docker (Track 2 / O25). Local-only.

One command per sanitizer; the repo is mounted read-only into a GCC 13 container, the CMake build
tree lives in a named Docker volume (incremental across runs), and CTest runs the whole suite:

    python tools/run_sanitizer_tests.py thread     # ThreadSanitizer
    python tools/run_sanitizer_tests.py address    # AddressSanitizer + UBSan
    python tools/run_sanitizer_tests.py thread -R state_bridge   # extra args go to ctest

Exit code is CTest's (0 = every test passed with no sanitizer report). Shares the process helper
with tools/run_wasm_tests.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_wasm_tests as rwt  # noqa: E402

IMAGE = "playgta6-sanitizer-tests:gcc13"
DOCKERFILE = "FROM gcc:13\nRUN apt-get update && apt-get install -y --no-install-recommends cmake && rm -rf /var/lib/apt/lists/*\n"
SANITIZERS = {"thread": "thread", "address": "address,undefined"}


def container_script(sanitizer: str) -> str:
    flags = f"-fsanitize={SANITIZERS[sanitizer]} -fno-omit-frame-pointer -O1 -g"
    return (
        "set -e\n"
        f'cmake -S /src/runtime -B /build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS="{flags}" '
        f'-DCMAKE_EXE_LINKER_FLAGS="-fsanitize={SANITIZERS[sanitizer]}" > /dev/null\n'
        "cmake --build /build --parallel\n"
        'cd /build && exec ctest --output-on-failure --timeout 300 "$@"\n'
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sanitizer", choices=sorted(SANITIZERS))
    args, ctest_args = parser.parse_known_args(argv)
    repo = rwt.REPO_ROOT
    if rwt.run_command(["docker", "image", "inspect", IMAGE]).returncode != 0:
        print(f"building helper image {IMAGE} (one-time) ...")
        built = rwt.run_command(["docker", "build", "-t", IMAGE, "-"], stdin_text=DOCKERFILE, timeout=1200)
        if built.returncode != 0:
            print(built.output[-800:], file=sys.stderr)
            return 2
    res = rwt.run_command(
        ["docker", "run", "--rm", "-v", f"{repo}:/src:ro", "-v", f"playgta6-san-{args.sanitizer}:/build",
         IMAGE, "bash", "-c", container_script(args.sanitizer), "ctest", *ctest_args],
        on_line=print,
    )
    return res.returncode


if __name__ == "__main__":
    raise SystemExit(main())
