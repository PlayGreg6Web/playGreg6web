import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import triage_fault as tf  # noqa: E402

SCRIPT = str(REPO_ROOT / "tools" / "triage_fault.py")


def loc(entry, off):
    return {"address": hex(entry + off), "entry": hex(entry), "offset": hex(off), "symbol": None}


def report(source=None, function=True):
    rec = {"kind": "unknown_target", "code_miss": True, "thread": 1, "elapsed_ms": 1, "rip": "0x1", "target": "0x2"}
    if function:
        rec["function"] = loc(0x140001180, 0xB4)
    if source is not None:
        rec["source"] = source
    return {"status": "fault", "phase": "run", "fault_record": rec}


class TriageFaultTest(unittest.TestCase):
    def test_pick_site_prefers_call_return_address(self):
        rec = report({"return_address": "0x140001030", "location": loc(0x140001000, 0x30)})["fault_record"]
        self.assertEqual(tf.pick_site(rec)[:2], (0x140001030, True))

    def test_pick_site_falls_back_to_function_offset(self):
        self.assertEqual(tf.pick_site(report()["fault_record"])[:2], (0x140001234, False))
        self.assertIsNone(tf.pick_site(report(function=False)["fault_record"]))

    def test_command_flags(self):
        cmd = tf.command("recomp --", "g.exe", 0x140001030, True, True)
        self.assertEqual(cmd, ["recomp", "--", "explain-site", "g.exe", "0x140001030", "--return-address", "--no-refine"])

    def test_runs_stub_recompiler_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rep = d / "r.json"
            rep.write_text(json.dumps(report({"return_address": "0x140001030", "location": loc(0x140001000, 0x30)})))
            stub = d / "stub.py"
            stub.write_text("import sys\nprint('STUB ' + ' '.join(sys.argv[1:]))\n")
            out = subprocess.run(
                [sys.executable, SCRIPT, str(rep), "--pe", "synthetic.exe", "--recomp", f'"{sys.executable}" "{stub}"'],
                capture_output=True,
                text=True,
            )
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn("STUB explain-site synthetic.exe 0x140001030 --return-address", out.stdout)
            self.assertIn("FAULT unknown_target", out.stdout)

    def test_no_fault_record(self):
        with tempfile.TemporaryDirectory() as d:
            rep = Path(d) / "r.json"
            rep.write_text("{}")
            out = subprocess.run([sys.executable, SCRIPT, str(rep), "--pe", "x"], capture_output=True, text=True)
            self.assertEqual(out.returncode, 1)


if __name__ == "__main__":
    unittest.main()
