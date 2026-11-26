#!/usr/bin/env python3
"""Replay ONE recorded .pg6trace capture with two web/ runtime builds and compare per-frame output.

Rendering differences between two builds cannot be proven from two game boots (timing-dependent),
but a capture is a fixed command stream: replaying it through build A and build B must give
bit-identical frames unless the render path changed. Each build is replayed with
tools/replay_capture.py; the full-resolution frame digest (logged by tools/replay/replay.html) and the
960px thumbnail are compared per frame. Reports the first differing frame and writes a diff image.

A build is a web/ directory: --web-a/--web-b, or --ref-a/--ref-b git refs (web/ is exported with
`git archive` into a temp dir). Default for either side is the repo's own web/ (working tree).

Example:
  python tools/replay_compare.py --ref-a origin/main~1 --ref-b origin/main --replay-dir work/replay \\
      --trace capture.pg6trace --pack shaders.wgsl.pack --frames 600 --every 10 --out work/cmp

Exit code: 0 identical, 1 replay failure, 2 frames differ.
"""
import argparse
import io
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_capture  # noqa: E402

REPO_ROOT = replay_capture.REPO_ROOT
_DIGEST_RX = re.compile(r"digest frame (\d+): ([0-9a-f]{16})")


def parse_digests(text):
    """Log text -> {frame: 16-hex digest}."""
    return {int(m[1]): m[2] for m in _DIGEST_RX.finditer(text or "")}


def frame_list(every, last, extra=()):
    """Frames 0, every, 2*every, ... <= last, plus `last` and any `extra`, sorted and unique."""
    if every < 1:
        raise ValueError("--every must be >= 1")
    return sorted(set(range(0, last + 1, every)) | {last} | set(extra))


def diff_digests(a, b):
    """Frames present in either side whose digest differs or is missing, ascending."""
    return sorted(f for f in set(a) | set(b) if a.get(f) != b.get(f))


def encode_png_rgb(width, height, rgb):
    def chunk(typ, body):
        return struct.pack(">I", len(body)) + typ + body + struct.pack(">I", zlib.crc32(typ + body) & 0xFFFFFFFF)
    stride = width * 3
    raw = b"".join(b"\0" + bytes(rgb[y * stride:(y + 1) * stride]) for y in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def diff_images(png_a, png_b):
    """(differing_pixels, total_pixels, max_channel_delta, diff_png_bytes), or four Nones on size mismatch.

    Diff image: unchanged pixels dimmed to grey, differing pixels red (delta*8, at least 128)."""
    wa, ha, ra, _ = replay_capture.decode_png_rgb(png_a)
    wb, hb, rb, _ = replay_capture.decode_png_rgb(png_b)
    if (wa, ha) != (wb, hb):
        return None, None, None, None
    out = bytearray(len(ra))
    n_diff = max_d = 0
    for i in range(0, len(ra), 3):
        d = max(abs(ra[i] - rb[i]), abs(ra[i + 1] - rb[i + 1]), abs(ra[i + 2] - rb[i + 2]))
        if d:
            n_diff += 1
            max_d = max(max_d, d)
            out[i] = min(255, max(128, d * 8))
        else:
            out[i] = out[i + 1] = out[i + 2] = (ra[i] + ra[i + 1] + ra[i + 2]) // 12
    return n_diff, wa * ha, max_d, encode_png_rgb(wa, ha, out)


def export_ref_web(ref, dest):
    """git archive <ref> web -> dest/web; returns that path."""
    data = subprocess.run(["git", "-C", str(REPO_ROOT), "archive", "--format=tar", ref, "web"],
                          check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        tf.extractall(dest)
    return str(Path(dest) / "web")


def run_build(label, web_root, a, frames):
    out = f"{a.out}-{label}"
    argv = ["--replay-dir", a.replay_dir, "--trace", a.trace, "--pack", a.pack, "--out", out,
            "--thumbat", ",".join(map(str, frames)), "--frames", str(a.frames), "--web-root", web_root,
            "--timeout", str(a.timeout), "--max-trace-mb", str(a.max_trace_mb)]
    for opt, val in (("--chrome", a.chrome), ("--canvas", a.canvas)):
        if val:
            argv += [opt, val]
    if a.trim_every:
        argv += ["--trim-every", str(a.trim_every), "--trim-largest", str(a.trim_largest)]
    rc = replay_capture.main(argv)
    log = Path(out + ".log.txt")
    text = log.read_text(encoding="utf-8") if log.exists() else ""
    return rc, parse_digests(text), out


def compare(digests_a, digests_b, out_a, out_b, diff_prefix):
    """Report dict; writes <diff_prefix>-swap<N>.diff.png for the first differing frame."""
    bad = diff_digests(digests_a, digests_b)
    rep = {"frames_compared": len(set(digests_a) | set(digests_b)), "differing_frames": bad,
           "first_differing_frame": bad[0] if bad else None, "identical": not bad}
    if bad:
        f = bad[0]
        pa, pb = Path(f"{out_a}-swap{f}.png"), Path(f"{out_b}-swap{f}.png")
        if pa.exists() and pb.exists():
            n, total, mx, png = diff_images(pa.read_bytes(), pb.read_bytes())
            if png:
                dp = Path(f"{diff_prefix}-swap{f}.diff.png")
                dp.write_bytes(png)
                rep.update(thumb_diff_pixels=n, thumb_total_pixels=total, thumb_max_delta=mx, diff_image=str(dp))
            else:
                rep["note"] = "thumbnail size differs or missing"
    return rep


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--web-a")
    p.add_argument("--web-b")
    p.add_argument("--ref-a")
    p.add_argument("--ref-b")
    p.add_argument("--replay-dir", required=True)
    p.add_argument("--trace", required=True)
    p.add_argument("--pack", required=True)
    p.add_argument("--out", required=True, help="output prefix")
    p.add_argument("--frames", type=int, required=True, help="last frame to replay")
    p.add_argument("--every", type=int, default=1, help="digest every Nth frame (default 1 = all)")
    p.add_argument("--chrome", default=None)
    p.add_argument("--canvas", default="")
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--trim-every", type=int, default=0)
    p.add_argument("--trim-largest", type=int, default=3)
    p.add_argument("--max-trace-mb", type=int, default=1536)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    frames = frame_list(a.every, a.frames)
    tmp = tempfile.mkdtemp(prefix="replay-cmp-")
    try:
        roots = {}
        for label in "ab":
            web, ref = getattr(a, f"web_{label}"), getattr(a, f"ref_{label}")
            if web and ref:
                raise SystemExit(f"give only one of --web-{label} / --ref-{label}")
            if ref:
                d = Path(tmp) / label
                d.mkdir()
                web = export_ref_web(ref, d)
            roots[label] = web or str(REPO_ROOT / "web")
        res = {}
        for label in "ab":
            rc, dig, out = run_build(label.upper(), roots[label], a, frames)
            if rc != 0 or not dig:
                print(f"replay of build {label.upper()} failed (rc={rc}, digests={len(dig)})", file=sys.stderr)
                return 1
            res[label] = (dig, out)
        rep = compare(res["a"][0], res["b"][0], res["a"][1], res["b"][1], a.out)
        if rep["identical"]:
            print(f"IDENTICAL: {rep['frames_compared']} frame digests match")
            return 0
        print(f"DIFFERENT: first differing frame {rep['first_differing_frame']}; "
              f"{len(rep['differing_frames'])}/{rep['frames_compared']} frames differ")
        if "diff_image" in rep:
            print(f"thumbnail diff: {rep['thumb_diff_pixels']}/{rep['thumb_total_pixels']} px, "
                  f"max channel delta {rep['thumb_max_delta']}; image {rep['diff_image']}")
        return 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
