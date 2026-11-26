#!/usr/bin/env python3
"""Replay ONE .pg6trace capture with two pg6_native_exec binaries and compare per-frame digests.

The native Dawn executor (docs/NATIVE_EXECUTOR_DESIGN.md) replays a capture headless and prints an exact
full-resolution digest per frame (`digest frame N: <16 hex>`). This tool runs two builds on the same
trace, reports the first differing frame and writes a diff PNG of that frame (red = differs), using
`pg6_native_exec --dump-frame N` (the screenshot PNG encoder) to get both images.

A side is a binary (--exe-a/--exe-b) or a git ref (--ref-a/--ref-b): the ref's `runtime/` is exported with
`git archive`, configured with -DPLAYGTA6_NATIVE_EXECUTOR=ON, built (target pg6_native_exec) and the
resulting binary is cached per commit next to the object cache (<objcache>/../native-exec/<sha>/).

--self-check replays twice with the SAME binary (side A) and reports any frame whose digest flips:
the noise floor / nondeterminism check to run before trusting an A-vs-B difference.

Example:
  python tools/native_replay_compare.py --ref-a origin/main~1 --ref-b origin/main \\
      --trace capture.pg6trace --pack shaders.wgsl.pack --frames 600 --every 10 --out work/ncmp
  python tools/native_replay_compare.py --exe-a build/pg6_native_exec --self-check --trace capture.pg6trace --frames 100

Exit code: 0 identical, 1 failure, 2 frames differ, 77 no WebGPU adapter.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_compare  # noqa: E402

REPO_ROOT = replay_compare.REPO_ROOT
parse_digests = replay_compare.parse_digests
frame_list = replay_compare.frame_list
diff_digests = replay_compare.diff_digests
diff_images = replay_compare.diff_images


def exe_command(exe):
    """argv prefix for a binary; a .py file (test doubles) runs under this interpreter."""
    return [sys.executable, str(exe)] if str(exe).endswith(".py") else [str(exe)]


def replay_args(a, frames, extra=()):
    args = ["--trace", a.trace, "--frames", str(max(frames)), "--digest-at", ",".join(map(str, frames))]
    if a.pack:
        args += ["--pack", a.pack]
    for r in a.reflection or ():
        args += ["--reflection", r]
    if a.size:
        args += ["--size", a.size]
    return args + list(extra)


def run_exec(exe, args, log_path=None):
    """(returncode, stdout+stderr text)."""
    proc = subprocess.run(exe_command(exe) + args, capture_output=True, text=True)
    text = proc.stdout + proc.stderr
    if log_path:
        Path(log_path).write_text(text, encoding="utf-8")
    return proc.returncode, text


def cache_root(env=None):
    env = os.environ if env is None else env
    if env.get("PLAYGTA6_NATIVE_EXEC_CACHE"):
        return Path(env["PLAYGTA6_NATIVE_EXEC_CACHE"])
    import build_native_boot  # the #1639 object-cache location
    return build_native_boot.default_cache_dir().parent / "native-exec"


def find_cmake():
    return shutil.which("cmake") or r"C:\Program Files\CMake\bin\cmake.exe"


def git(*args):
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], check=True, capture_output=True).stdout


def build_ref(ref, cache_dir, say=print):
    """Binary for git ref `ref`, built once per commit (cached). Returns its path."""
    sha = git("rev-parse", "--verify", ref + "^{commit}").decode().strip()
    bin_dir = Path(cache_dir) / sha[:16]
    name = "pg6_native_exec.exe" if os.name == "nt" else "pg6_native_exec"
    exe = bin_dir / name
    if exe.is_file():
        say(f"{ref} ({sha[:10]}): cached {exe}")
        return exe
    work = Path(tempfile.mkdtemp(prefix="pg6-nexec-"))
    try:
        say(f"{ref} ({sha[:10]}): building pg6_native_exec (first time; Dawn is fetched at configure time)")
        tar = work / "src.tar"
        tar.write_bytes(git("archive", "--format=tar", sha, "runtime", "tests"))
        src = work / "src"
        src.mkdir()
        subprocess.run(["tar", "-xf", str(tar), "-C", str(src)], check=True)
        build = work / "build"
        cm = find_cmake()
        subprocess.run([cm, "-S", str(src / "runtime"), "-B", str(build), "-DPLAYGTA6_NATIVE_EXECUTOR=ON",
                        "-DCMAKE_BUILD_TYPE=Release"], check=True)
        subprocess.run([cm, "--build", str(build), "--config", "Release", "--target", "pg6_native_exec"], check=True)
        found = next(build.rglob(name), None)
        if not found:
            raise SystemExit(f"{name} not found after building {ref}")
        bin_dir.mkdir(parents=True, exist_ok=True)
        for f in found.parent.iterdir():  # the exe plus dxcompiler.dll / dxil.dll
            if f.is_file() and (f.name == name or f.suffix.lower() == ".dll"):
                shutil.copy2(f, bin_dir / f.name)
        return exe
    finally:
        shutil.rmtree(work, ignore_errors=True)  # keeps the disk free; only the binary is cached


def flipping_frames(digests_list):
    """Frames whose digest is not identical across all runs (missing counts as different), ascending."""
    frames = set().union(*digests_list) if digests_list else set()
    return sorted(f for f in frames if len({d.get(f) for d in digests_list}) > 1)


def dump_frame_png(exe, a, frame, out_png):
    """Replays up to `frame` and writes that frame's swapchain image as out_png; False when it failed."""
    d = Path(tempfile.mkdtemp(prefix="pg6-dump-"))
    try:
        args = replay_args(a, [frame], ["--dump-frame", str(frame), "--dump-dir", str(d)])
        rc, _ = run_exec(exe, args)
        src = d / f"frame-{frame}.png"
        if rc != 0 or not src.is_file():
            return False
        shutil.copyfile(src, out_png)
        return True
    finally:
        shutil.rmtree(d, ignore_errors=True)


def compare(digests_a, digests_b, exe_a, exe_b, a, out):
    """Report dict; for the first differing frame dumps both images and writes <out>-frame<N>.diff.png."""
    bad = diff_digests(digests_a, digests_b)
    rep = {"frames_compared": len(set(digests_a) | set(digests_b)), "differing_frames": bad,
           "first_differing_frame": bad[0] if bad else None, "identical": not bad}
    if bad:
        f = bad[0]
        pa, pb = Path(f"{out}-A-frame{f}.png"), Path(f"{out}-B-frame{f}.png")
        if dump_frame_png(exe_a, a, f, pa) and dump_frame_png(exe_b, a, f, pb):
            n, total, mx, png = diff_images(pa.read_bytes(), pb.read_bytes())
            if png:
                dp = Path(f"{out}-frame{f}.diff.png")
                dp.write_bytes(png)
                rep.update(diff_pixels=n, total_pixels=total, max_delta=mx, diff_image=str(dp))
            else:
                rep["note"] = "frame sizes differ"
        else:
            rep["note"] = "could not dump the first differing frame"
    return rep


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for s in "ab":
        p.add_argument(f"--exe-{s}", help=f"pg6_native_exec binary, side {s.upper()}")
        p.add_argument(f"--ref-{s}", help=f"git ref built into side {s.upper()}'s binary")
    p.add_argument("--self-check", action="store_true",
                   help="replay twice with side A's binary and report digests that flip (nondeterminism)")
    p.add_argument("--trace", required=True)
    p.add_argument("--pack")
    p.add_argument("--reflection", action="append")
    p.add_argument("--size", help="WxH swapchain size (default: the executor's)")
    p.add_argument("--frames", type=int, required=True, help="last frame to replay")
    p.add_argument("--every", type=int, default=1, help="digest every Nth frame (default 1 = all)")
    p.add_argument("--out", default="native-cmp", help="output prefix")
    p.add_argument("--cache-dir", type=Path, help="binary cache for --ref-* (default: next to the object cache)")
    return p


def resolve_side(a, s, cache_dir):
    exe, ref = getattr(a, f"exe_{s}"), getattr(a, f"ref_{s}")
    if exe and ref:
        raise SystemExit(f"give only one of --exe-{s} / --ref-{s}")
    if ref:
        return build_ref(ref, cache_dir or cache_root())
    if exe:
        return Path(exe)
    raise SystemExit(f"side {s.upper()} needs --exe-{s} or --ref-{s}")


def main(argv=None):
    a = build_parser().parse_args(argv)
    frames = frame_list(a.every, a.frames)
    exe_a = resolve_side(a, "a", a.cache_dir)
    exe_b = exe_a if a.self_check else resolve_side(a, "b", a.cache_dir)
    runs = []
    for label, exe in (("A", exe_a), ("A2" if a.self_check else "B", exe_b)):
        rc, text = run_exec(exe, replay_args(a, frames), f"{a.out}-{label}.log.txt")
        dig = parse_digests(text)
        if rc == 77:
            print("no WebGPU adapter on this host", file=sys.stderr)
            return 77
        if rc != 0 or not dig:
            print(f"replay {label} failed (rc={rc}, digests={len(dig)}); see {a.out}-{label}.log.txt", file=sys.stderr)
            return 1
        runs.append(dig)
    if a.self_check:
        flips = flipping_frames(runs)
        if not flips:
            print(f"DETERMINISTIC: {len(runs[0])} frame digests identical across two runs")
            return 0
        print(f"NONDETERMINISTIC: {len(flips)} frame digests differ between two runs of the same binary; "
              f"first {flips[0]}")
        return 2
    rep = compare(runs[0], runs[1], exe_a, exe_b, a, a.out)
    if rep["identical"]:
        print(f"IDENTICAL: {rep['frames_compared']} frame digests match")
        return 0
    print(f"DIFFERENT: first differing frame {rep['first_differing_frame']}; "
          f"{len(rep['differing_frames'])}/{rep['frames_compared']} frames differ")
    if "diff_image" in rep:
        print(f"frame diff: {rep['diff_pixels']}/{rep['total_pixels']} px, max channel delta {rep['max_delta']}; "
              f"image {rep['diff_image']}")
    elif "note" in rep:
        print(rep["note"])
    return 2


if __name__ == "__main__":
    sys.exit(main())
