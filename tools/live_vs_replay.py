#!/usr/bin/env python3
"""Does the native LIVE executor render exactly what a capture replay renders?

One live boot (`playgta6_boot --live-executor`, headless, or `--window` with --show-window; via tools/run_native_boot.py) writes a GPU capture
(--gpu-capture, with payloads, not thinned) and, from the live executor itself, one replay digest per
presented frame (`--live-digest`: the stream image before the blit, hashed like replay.html's
`digest frame N:`). The capture is then replayed headless with pg6_native_exec at the live stream size
and its per-frame digests are compared. The live-only paths (guest-memory read-backs, lazy shader
compile, async pipelines + blob cache, blit) are exercised by the live run; the replay takes none of
them, so a mismatch names the first frame where a live-only path changed the pixels.

  python tools/live_vs_replay.py --exe <game.exe> --game-dir <package dir> --pack <std.wgsl.pack> \
      [--frames 600] [--max-ms 240000] [--build-dir work/native-boot] [--name livecmp] [--keep]
  python tools/live_vs_replay.py --run-dir <existing run dir> --pack <pack>   # compare a finished live run

Needs a Dawn-enabled harness (built here with --native-executor) and a GPU. The capture is large (it
holds the textures): it and the lifted program/ are deleted afterwards unless --keep. Game files stay
outside the repository; this script prints only counts and digests. Exit codes: 0 identical, 2 frames
differ, 1 failure.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LINE = re.compile(r"^digest frame (\d+): ([0-9a-f]{16})(?: (\d+)x(\d+) (\S+))?\s*$")


def parse_digests(text: str) -> dict[int, tuple[str, tuple | None]]:
    out = {}
    for line in text.splitlines():
        m = LINE.match(line)
        if m:
            size = (int(m[3]), int(m[4]), m[5]) if m[3] else None
            out[int(m[1])] = (m[2], size)
    return out


def find_exec(build_dir: Path) -> Path:
    """pg6_native_exec from the harness build; built on demand (the harness build does not include it)."""
    name = "pg6_native_exec.exe" if os.name == "nt" else "pg6_native_exec"
    harness_build = build_dir / "program" / "native" / "harness-build"
    for p in harness_build.rglob(name):
        return p
    cmake = shutil.which("cmake") or r"C:\Program Files\CMakein\cmake.exe"
    subprocess.run([cmake, "--build", ".", "--config", "Release", "--target", "pg6_native_exec"],
                   cwd=harness_build / "runtime", check=True)
    for p in harness_build.rglob(name):
        return p
    raise SystemExit(f"pg6_native_exec not found under {harness_build}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", default=os.environ.get("PLAYGTA6_EXE"))
    ap.add_argument("--game-dir", default=os.environ.get("GTA6_GAME_DIR"))
    ap.add_argument("--pack", required=True, help="WGSL pack (hash-only shader payloads)")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--max-ms", type=int, default=None, help="boot limit (default: the preset's)")
    ap.add_argument("--preset", default="menu-flow",
                    help="run_native_boot.py preset (default menu-flow: title screens, menus, legal notices)")
    ap.add_argument("--capture-start", default="script",
                    help="`script` (the preset's capture step), `boot` (frame 0), or N ms after boot")
    ap.add_argument("--build-dir", type=Path, default=REPO / "work" / "native-boot")
    ap.add_argument("--name", default="livecmp")
    ap.add_argument("--window-size", default="1280x720")
    ap.add_argument("--show-window", action="store_true", help="run the live side with a real window (default: headless --live-executor)")
    ap.add_argument("--run-dir", type=Path, help="skip the live run: compare this finished run directory")
    ap.add_argument("--native-exec", type=Path, help="pg6_native_exec binary (default: found under --build-dir)")
    ap.add_argument("--keep", action="store_true", help="keep the capture and the lifted program/")
    a = ap.parse_args()

    build_dir = a.build_dir.resolve()
    run_dir = a.run_dir
    if run_dir is None:
        if not a.exe or not a.game_dir:
            ap.error("--exe and --game-dir (or PLAYGTA6_EXE / GTA6_GAME_DIR) are required for the live run")
        run_dir = build_dir / "runs" / a.name
        shutil.rmtree(run_dir, ignore_errors=True)
        digest_file = build_dir / f"{a.name}-live-digest.txt"
        digest_file.unlink(missing_ok=True)
        cmd = [sys.executable, str(REPO / "tools" / "run_native_boot.py"), "--preset", a.preset, "--name", a.name,
               "--exe", a.exe, "--game-dir", a.game_dir, "--build-dir", str(build_dir), "--native-executor",
               "--shader-pack", a.pack, "--gpu-capture-frames", str(a.frames), "--gpu-capture-payloads",
               *(["--max-ms", str(a.max_ms), "--watchdog-ms", str(a.max_ms + 60000)] if a.max_ms else []),
               *({"boot": [], "script": ["--gpu-capture-start-at-script"]}.get(
                   a.capture_start, ["--gpu-capture-start-ms", a.capture_start])),
               *(["--show-window"] if a.show_window else []),
               "--", *(["--window"] if a.show_window else ["--live-executor"]), "--window-size", a.window_size, "--live-digest", str(digest_file)]
        print("live run:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=False)
        shutil.move(str(digest_file), run_dir / "live-digest.txt") if digest_file.is_file() else None
    run_dir = run_dir.resolve()
    live_file, trace = run_dir / "live-digest.txt", run_dir / "capture.pg6trace"
    if not live_file.is_file() or not trace.is_file():
        print(f"missing {live_file.name} or {trace.name} in {run_dir}: the live run produced no capture / digests")
        return 1
    live_all = parse_digests(live_file.read_text())
    # capture frame i is the live executor's present number first_present + i
    meta = json.loads((run_dir / "capture.pg6trace.json").read_text())
    first, complete = int(meta.get("first_present", 0)), int(meta.get("frames_complete", 0))
    print(f"capture: {complete} frames starting at live present {first}")
    live = {i: live_all[first + i] for i in range(complete) if first + i in live_all}
    frames = sorted(live)
    if not frames:
        print("the live executor logged no digest lines")
        return 1
    sizes = {live[f][1] for f in frames if live[f][1]}
    print(f"live: {len(frames)} frames digested, stream size(s) {sorted(sizes, key=str)}")
    if len(sizes) != 1:
        print("the stream size changed during the run; replay cannot cover that in one pass")
    w, h, _fmt = sorted(sizes, key=str)[0]

    exe = a.native_exec or find_exec(build_dir)
    rep_cmd = [str(exe), "--trace", str(trace), "--pack", a.pack, "--size", f"{w}x{h}",
               "--frames", str(frames[-1]), "--digest-at", ",".join(map(str, frames)),
               "--report", str(run_dir / "replay-stats.json")]
    print("replay:", " ".join(rep_cmd), flush=True)
    proc = subprocess.run(rep_cmd, capture_output=True, text=True)
    (run_dir / "replay.log").write_text(proc.stdout + proc.stderr)
    replay = parse_digests(proc.stdout)
    print(f"replay: exit {proc.returncode}, {len(replay)} digests")
    for line in proc.stdout.splitlines():
        if line.startswith(("VALIDATION", "last packet error")):
            print(" ", line)

    mismatches = [f for f in frames if f in replay and replay[f][0] != live[f][0]]
    missing = [f for f in frames if f not in replay]
    for f in mismatches[:10]:
        print(f"MISMATCH frame {f}: live {live[f][0]} replay {replay[f][0]}")
    if missing:
        print(f"{len(missing)} frames have no replay digest (first {missing[0]})")
    rc = 0 if not mismatches and not missing and proc.returncode == 0 else (2 if mismatches else 1)
    print(f"RESULT: {len(frames) - len(mismatches) - len(missing)}/{len(frames)} frames identical; "
          + (f"first mismatch at frame {mismatches[0]}" if mismatches else "no mismatch"))
    if not a.keep and rc == 0:
        trace.unlink(missing_ok=True)
        shutil.rmtree(build_dir / "program", ignore_errors=True)
        print("deleted the capture and program/")
    return rc


if __name__ == "__main__":
    sys.exit(main())
