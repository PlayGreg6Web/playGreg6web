import struct
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import prebuilt_web  # noqa: E402
import run_browser_boot as rbb  # noqa: E402


def write_pack(path: Path, version: int, sidecar: bool = True) -> Path:
    path.write_bytes(struct.pack("<4sIII", b"WGPK", version, 0, 0) + b"\0" * 16)
    if sidecar:
        Path(str(path) + ".reflect.json").write_text("{}", encoding="utf-8")
    return path


class StreamPackTests(unittest.TestCase):
    def test_resolve_prefers_valid_explicit_then_env(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            v2 = write_pack(Path(d) / "std.wgsl.pack", 2)
            self.assertEqual(prebuilt_web.resolve_stream_pack(str(v2), None), v2)
            self.assertEqual(prebuilt_web.resolve_stream_pack(None, str(v2)), v2)
            self.assertIsNone(prebuilt_web.resolve_stream_pack(None, None))

    def test_env_pack_that_is_v1_or_lacks_sidecar_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            v1 = write_pack(Path(d) / "a.wgsl.pack", 1)
            nosc = write_pack(Path(d) / "b.wgsl.pack", 2, sidecar=False)
            self.assertIsNone(prebuilt_web.resolve_stream_pack(None, str(v1)))
            self.assertIsNone(prebuilt_web.resolve_stream_pack(None, str(nosc)))
            self.assertIsNone(prebuilt_web.resolve_stream_pack(None, str(Path(d) / "missing")))

    def test_explicit_invalid_pack_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            v1 = write_pack(Path(d) / "a.wgsl.pack", 1)
            with self.assertRaises(SystemExit):
                prebuilt_web.resolve_stream_pack(str(v1), None)

    def test_staging_serves_pack_and_sidecar_and_url_has_stream_flag(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            v2 = write_pack(Path(d) / "std.wgsl.pack", 2)
            stage = Path(d) / "staged-shaders-stream"
            rbb.stage_shader_pack(v2, stage)
            self.assertEqual((stage / "std.wgsl.pack").stat().st_size, v2.stat().st_size)
            self.assertTrue((stage / "std.wgsl.pack.reflect.json").is_file())
            self.assertEqual(sorted(p.name for p in stage.iterdir()), ["std.wgsl.pack", "std.wgsl.pack.reflect.json"])
        url = rbb.build_target_url("127.0.0.1", 8141, input_script="calibration-skip", stream_shaders=True)
        self.assertIn("streamShaders=1", url)


if __name__ == "__main__":
    unittest.main()
