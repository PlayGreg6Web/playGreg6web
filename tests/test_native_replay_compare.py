"""native_replay_compare against a fake pg6_native_exec (a .py double): no GPU, no game bytes.

The synthetic "trace" is a text file with one pixel value per frame; the double digests it like the executor
does, honours --digest-at / --dump-frame / --frames, can bias frames (env FAKE_BIAS from FAKE_BIAS_FROM) and
can flip one frame on every second run (env FAKE_COUNTER + FAKE_FLIP_FRAME).
"""
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import native_replay_compare as nrc  # noqa: E402
import replay_compare  # noqa: E402

FAKE = textwrap.dedent('''
    import hashlib, os, struct, sys, zlib
    argv = sys.argv[1:]
    def opt(n, d=None):
        return argv[argv.index(n) + 1] if n in argv else d
    vals = [int(x) for x in open(opt("--trace")).read().split()]
    bias = int(os.environ.get("FAKE_BIAS", "0"))
    bias_from = int(os.environ.get("FAKE_BIAS_FROM", "0"))
    counter = os.environ.get("FAKE_COUNTER")
    run = 0
    if counter:
        run = int(open(counter).read()) if os.path.exists(counter) else 0
        open(counter, "w").write(str(run + 1))
    flip_frame = int(os.environ.get("FAKE_FLIP_FRAME", "-1"))
    def px(i):
        v = vals[i] + (bias if i >= bias_from else 0)
        if i == flip_frame and run % 2:
            v += 7
        return v & 255
    def png(g):
        raw = (b"\\0" + bytes([g, 0, 0]) * 2)
        def c(t, b):
            return struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b))
        return (b"\\x89PNG\\r\\n\\x1a\\n" + c(b"IHDR", struct.pack(">IIBBBBB", 2, 1, 8, 2, 0, 0, 0))
                + c(b"IDAT", zlib.compress(raw)) + c(b"IEND", b""))
    last = int(opt("--frames", len(vals) - 1))
    dig = {int(x) for x in opt("--digest-at", "").split(",") if x}
    dump = {int(x) for x in opt("--dump-frame", "").split(",") if x}
    for i in range(min(len(vals) - 1, last) + 1):
        if i in dig:
            print("digest frame %d: %s" % (i, hashlib.sha256(bytes([px(i)])).hexdigest()[:16]))
        if i in dump:
            open(os.path.join(opt("--dump-dir", "."), "frame-%d.png" % i), "wb").write(png(px(i)))
''')


class NativeReplayCompareTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.d = Path(self.td.name)
        (self.d / "fake.py").write_text(FAKE)
        (self.d / "t.pg6trace").write_text("10 20 30 40 50 60")
        self.exe = str(self.d / "fake.py")
        self.out = str(self.d / "o")

    def tearDown(self):
        self.td.cleanup()

    def run_main(self, *extra, env=None):
        old = dict(os.environ)
        os.environ.update(env or {})
        try:
            return nrc.main(["--trace", str(self.d / "t.pg6trace"), "--frames", "5", "--out", self.out, *extra])
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_identical(self):
        self.assertEqual(self.run_main("--exe-a", self.exe, "--exe-b", self.exe), 0)

    def test_first_differing_frame_and_diff_png(self):
        b = self.d / "fakeb.py"  # side B: same double with frames >= 3 biased by 9
        b.write_text("import os\nos.environ['FAKE_BIAS'] = '9'\nos.environ['FAKE_BIAS_FROM'] = '3'\n" + FAKE)
        self.assertEqual(self.run_main("--exe-a", self.exe, "--exe-b", str(b)), 2)
        self.assertTrue(Path(self.out + "-frame3.diff.png").exists())
        n, total, mx, _ = replay_compare.diff_images(Path(self.out + "-A-frame3.png").read_bytes(),
                                                     Path(self.out + "-B-frame3.png").read_bytes())
        self.assertEqual((n, total, mx), (2, 2, 9))

    def test_self_check_deterministic(self):
        self.assertEqual(self.run_main("--exe-a", self.exe, "--self-check"), 0)

    def test_self_check_flags_nondeterminism(self):
        rc = self.run_main("--exe-a", self.exe, "--self-check",
                           env={"FAKE_COUNTER": str(self.d / "count"), "FAKE_FLIP_FRAME": "2"})
        self.assertEqual(rc, 2)

    def test_flipping_frames(self):
        self.assertEqual(nrc.flipping_frames([{0: "a", 1: "b"}, {0: "a", 1: "c", 2: "d"}]), [1, 2])
        self.assertEqual(nrc.flipping_frames([]), [])

    def test_failure_exit_codes(self):
        bad = self.d / "bad.py"
        bad.write_text("import sys; sys.exit(77)")
        self.assertEqual(self.run_main("--exe-a", str(bad), "--exe-b", str(bad)), 77)
        bad.write_text("import sys; sys.exit(1)")
        self.assertEqual(self.run_main("--exe-a", str(bad), "--exe-b", str(bad)), 1)

    def test_side_needs_exe_or_ref(self):
        with self.assertRaises(SystemExit):
            self.run_main("--exe-a", self.exe)

    def test_replay_args(self):
        a = nrc.build_parser().parse_args(["--trace", "t", "--frames", "9", "--pack", "p", "--size", "8x8"])
        self.assertEqual(nrc.replay_args(a, [0, 9]),
                         ["--trace", "t", "--frames", "9", "--digest-at", "0,9", "--pack", "p", "--size", "8x8"])


if __name__ == "__main__":
    unittest.main()
