import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")


@unittest.skipUnless(POWERSHELL and os.name == "nt", "needs Windows PowerShell")
class PlayScriptsDryRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        (self.tmp / "prebuilt.json").write_text(json.dumps({"commit": "0" * 40, "built_at": "x"}), encoding="utf-8")
        self.pack = self.tmp / "x.wgsl.pack"
        self.pack.write_bytes(b"x")
        staged = self.tmp / "build" / "staged-shaders"
        staged.mkdir(parents=True)
        (staged / "s.pack").write_bytes(b"x")

    def run_script(self, name: str, *args: str, **env: str) -> str:
        e = dict(os.environ, PLAYGTA6_EXE="C:\fake\game.exe", GTA6_GAME_DIR="C:\fake\pkg", **env)
        r = subprocess.run(
            [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
             str(REPO_ROOT / "tools" / name), "-DryRun", *args],
            capture_output=True, text=True, env=e, timeout=180, cwd=str(self.tmp),
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r.stdout + r.stderr

    def test_play_native_prebuilt(self) -> None:
        out = self.run_script("play_native.ps1", "-Prebuilt", "-Name", "dry1", "-ShaderPack", str(self.pack),
                              "-InputScript", "menu-flow", PLAYGTA6_PREBUILT_DIR=str(self.tmp))
        self.assertIn("prebuilt_stale.py --prebuilt", out)
        self.assertIn("python tools/run_native_boot.py --preset menu-flow --name dry1 --play", out)
        self.assertIn("--trust-build", out)
        self.assertIn(f"--build-dir {self.tmp}", out)
        self.assertIn(f"--shader-pack {self.pack}", out)

    def test_play_local_prebuilt(self) -> None:
        out = self.run_script("play_local.ps1", "-Prebuilt", "-Name", "dry2", "-PrebuiltDir", str(self.tmp))
        self.assertIn("prebuilt_stale.py --prebuilt", out)
        self.assertIn("--web", out)
        self.assertIn("python tools/run_browser_boot.py --preset long --name dry2 --headed", out)
        self.assertIn("--trust-build", out)
        self.assertIn("--input-script calibration-skip", out)
        self.assertIn("s.pack", out)


    def test_update_prebuilt_smoke_plan(self) -> None:
        out = self.run_script("update_prebuilt.ps1", "-Dir", str(self.tmp / "pb"))
        self.assertIn("prebuilt_smoke.py", out)
        self.assertIn("--live-executor", out)
        out = self.run_script("update_prebuilt.ps1", "-Dir", str(self.tmp / "pb"), "-SkipSmoke")
        self.assertIn("skipped (-SkipSmoke)", out)

if __name__ == "__main__":
    unittest.main()
