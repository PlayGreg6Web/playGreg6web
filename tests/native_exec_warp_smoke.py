"""End-to-end smoke of the native executor tools against a real pg6_native_exec on Dawn's software (WARP) adapter.

  python tests/native_exec_warp_smoke.py --exec <pg6_native_exec> --gen <gen_native_exec_trace> [--adapter warp]

Replays a generated synthetic trace (four swapchain clears), checks the digest lines, dumps a frame PNG and checks
it, then runs tools/native_replay_compare.py --self-check with the same binary. Exit 77 when there is no adapter.
"""
import argparse, re, struct, subprocess, sys, tempfile, zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd):
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=300)
    return p.returncode, p.stdout + p.stderr


def fail(msg, text=""):
    print("FAIL:", msg, "\n" + text, file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--adapter", default="warp")
    a = ap.parse_args()
    d = Path(tempfile.mkdtemp(prefix="pg6-warp-smoke-"))
    trace = d / "synthetic.pg6trace"
    rc, t = run([a.gen, trace])
    if rc or not trace.stat().st_size:
        fail("trace generation", t)
    size = ["--size", "64x32"]
    base = [a.exec, "--trace", trace, "--adapter", a.adapter] + size

    rc, t = run(base + ["--digest-at", "0,1,2,3", "--dump-frame", "1", "--dump-dir", d])
    if rc == 77:
        print("no adapter, skipped")
        return 77
    if rc:
        fail(f"replay rc={rc}", t)
    if "trace: 4 frames" not in t:
        fail("expected 4 frames", t)
    dg = dict(re.findall(r"digest frame (\d+): (\S+)", t))
    if len(dg) != 4 or dg["0"] != dg["3"] or len({dg["0"], dg["1"], dg["2"]}) != 3:
        fail(f"digests should be 4 lines, frame0 == frame3, red/green/blue distinct: {dg}", t)
    if "VALIDATION:" in t:
        fail("validation errors", t)

    png = (d / "frame-1.png").read_bytes()
    if png[:8] != b"\x89PNG\r\n\x1a\n":
        fail("frame-1.png is not a PNG")
    w, h = struct.unpack(">II", png[16:24])
    if (w, h) != (64, 32):
        fail(f"PNG size {w}x{h}")
    idat, pos = b"", 8
    while pos < len(png):
        n, typ = struct.unpack(">I4s", png[pos:pos + 8])
        if typ == b"IDAT":
            idat += png[pos + 8:pos + 8 + n]
        pos += 12 + n
    raw = zlib.decompress(idat)
    if raw[0] != 0 or tuple(raw[1:4]) != (0, 255, 0):  # frame 1 is green, filter byte 0, RGB8
        fail(f"frame 1 first pixel {tuple(raw[1:4])}, want green")

    rc, t = run([sys.executable, ROOT / "tools" / "native_replay_compare.py", "--exe-a", a.exec, "--self-check",
                 "--trace", trace, "--frames", "3", "--adapter", a.adapter, "--size", "64x32",
                 "--out", d / "cmp"])
    if rc or "DETERMINISTIC" not in t:
        fail(f"native_replay_compare --self-check rc={rc}", t)
    print("ok: replay, digests, PNG, self-check on", re.search(r"adapter: (.*)", run(base)[1]).group(1))
    return 0


sys.exit(main())
