"""End-to-end test of tools/replay_capture.py on a tiny synthetic .pg6trace (headless Chrome).

Skipped when no Chrome is found or PLAYGTA6_SKIP_BROWSER_TESTS is set.
"""
import contextlib
import io
import os
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import replay_capture  # noqa: E402


def _find_chrome():
    if os.environ.get("PLAYGTA6_SKIP_BROWSER_TESTS"):
        return None
    try:
        import run_browser_boot
        found = run_browser_boot.discover_chrome()
        return str(found) if found else None
    except Exception:
        return None


def pkt(op, payload=b""):
    return struct.pack("<HHI", op, 0, len(payload)) + payload


def synthetic_trace():
    """3 frames: create texture, thin-marker WriteTexture, clear to (0, 0.5, 1) and present."""
    create_tex = struct.pack("<8I", 1, 4, 4, 1, 1, 0, 0x16, 1)  # rgba8unorm, COPY_DST|TEXTURE_BINDING|RT
    thin_write = struct.pack("<12I", 1, 0, 0, 0, 0, 4, 4, 1, 16, 0, 0, replay_capture.THIN_MARKER)
    clear = struct.pack("<4f", 0.0, 0.5, 1.0, 1.0)
    out = pkt(12, create_tex) + pkt(13, thin_write)
    for _ in range(3):
        out += pkt(1) + pkt(2, clear) + pkt(7)
    return out


def png_first_pixel(data):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, w, ctype = 8, b"", 0, 2
    while pos < len(data):
        ln, typ = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + ln]
        if typ == b"IHDR":
            w, _h, _bd, ctype = struct.unpack(">IIBB", body[:10])
        elif typ == b"IDAT":
            idat += body
        pos += 12 + ln
    bpp = 4 if ctype == 6 else 3
    raw = zlib.decompress(idat)
    row = bytearray(raw[1:1 + w * bpp])
    ft = raw[0]
    for i in range(bpp, len(row)):
        a = row[i - bpp]
        if ft == 1:
            row[i] = (row[i] + a) & 255
        elif ft == 3:
            row[i] = (row[i] + a // 2) & 255
        elif ft == 4:
            row[i] = (row[i] + a) & 255  # Paeth with b=c=0 reduces to a
    return tuple(row[:3])


@unittest.skipUnless(_find_chrome(), "no Chrome found or PLAYGTA6_SKIP_BROWSER_TESTS set")
class ReplayCaptureE2E(unittest.TestCase):
    def test_replays_synthetic_capture(self):
        trace = synthetic_trace()
        self.assertTrue(replay_capture.is_thin_marker(trace, len(pkt(12, b"\0" * 32)), 13, 48))
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "t.pg6trace").write_bytes(trace)
            (d / "p.wgsl.pack").write_bytes(b"WGPK" + struct.pack("<3I", 1, 0, 0))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = replay_capture.main([
                    "--chrome", _find_chrome(), "--replay-dir", str(d), "--trace", "t.pg6trace",
                    "--pack", "p.wgsl.pack", "--thumbat", "2", "--out", str(d / "out"), "--timeout", "120"])
            out = buf.getvalue()
            if rc != 0 and ("navigator.gpu unavailable" in out
                            or ("non-black" not in out and "RESULT: exception" in out)):
                self.skipTest("Chrome has no usable WebGPU adapter here: " + out.strip()[:200])
            self.assertEqual(rc, 0, out)
            self.assertIn("non-black 100.00%", out)
            self.assertIn("distinct 1", out)
            self.assertIn("0 validation error", out)
            r, g, b = png_first_pixel((d / "out-swap2.png").read_bytes())
            self.assertLess(r, 8)
            self.assertAlmostEqual(g, 127, delta=3)
            self.assertGreater(b, 240)


if __name__ == "__main__":
    unittest.main()
