import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import fault_report as fr  # noqa: E402

SCRIPT = str(REPO_ROOT / "tools" / "fault_report.py")


def loc(entry, off, sym=None):
    return {"address": hex(entry + off), "entry": hex(entry), "offset": hex(off), "symbol": sym}


def report(kind="unknown_target"):
    rec = {
        "kind": kind,
        "code_miss": True,
        "thread": 3,
        "elapsed_ms": 4242,
        "rip": "0x140001234",
        "target": "0x140001234",
        "detail": "0x140001180",
        "present": {"presents": 2, "draws_per_frame": 17},
        "function": loc(0x140001180, 0xB4, {"module": "game.exe", "name": "Exported", "offset": "0x10"}),
        "source": {"return_address": "0x140001030", "location": loc(0x140001000, 0x30)},
        "registers": {"rax": "0x1", "rcx": "0x2", "rsp": "0x7ff00"},
        "call_stack": {"method": "rtl-virtual-unwind", "frames": [loc(0x140001000, 0x30)]},
        "recent_entries": [
            {"target": "0x140001180", "name": None, "from": "0x140001020"},
            {"target": "0x5047000000000010", "name": "KERNEL32.dll!Sleep", "from": "0x140001030"},
        ],
    }
    if kind == "unknown_resume_target":
        rec["source"] = {"resume_rip": "0x140002000", "location": loc(0x140001F00, 0x100)}
    if kind == "trap":
        rec["source"] = None
    return {
        "status": "Fault",
        "phase": "entry",
        "message": "boom",
        "fault_record": rec,
        "code_misses": {"unknown_target": {"count": 1, "targets": []}, "trap": {"count": 0, "targets": []}},
    }


class FaultReportTests(unittest.TestCase):
    def test_render_contains_core_facts(self):
        out = fr.render(report())
        for needle in (
            "FAULT unknown_target",
            "[recompiler gap]",
            "sub_140001180+0xb4",
            "game.exe!Exported+0x10",
            "draws/frame=17",
            "elapsed: 4242 ms",
            "KERNEL32.dll!Sleep",
            "return address 0x0000000140001030",
            "rtl-virtual-unwind",
            "rsp=",
            "code_misses in this run: unknown_target=1",
        ):
            self.assertIn(needle, out)

    def test_entries_limit(self):
        out = fr.render(report(), entries=1)
        self.assertIn("last 1 of 2", out)
        self.assertNotIn("0x0000000140001180  from", out)

    def test_suggestions_by_kind(self):
        s = fr.suggestions(report()["fault_record"])
        self.assertTrue(any("--entry 0x140001234" in x for x in s))
        self.assertTrue(any("--entry 0x140001000" in x for x in s))
        s = fr.suggestions(report("unknown_resume_target")["fault_record"])
        self.assertTrue(any("--entry 0x140002000" in x for x in s))
        s = fr.suggestions(report("trap")["fault_record"])
        self.assertTrue(any(" lift " in x and "--entry 0x140001180" in x for x in s))

    def test_no_record(self):
        self.assertEqual(fr.render({"status": "Exited"}), "")

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as d:
            ok = Path(d) / "ok.json"
            ok.write_text(json.dumps(report()), encoding="utf-8")
            none = Path(d) / "none.json"
            none.write_text("{}", encoding="utf-8")
            bad = Path(d) / "bad.json"
            bad.write_text("not json", encoding="utf-8")
            self.assertEqual(subprocess.run([sys.executable, SCRIPT, str(ok)], capture_output=True).returncode, 0)
            self.assertEqual(subprocess.run([sys.executable, SCRIPT, str(none)], capture_output=True).returncode, 1)
            self.assertEqual(subprocess.run([sys.executable, SCRIPT, str(bad)], capture_output=True).returncode, 2)
            j = subprocess.run([sys.executable, SCRIPT, str(ok), "--json"], capture_output=True, text=True)
            self.assertEqual(json.loads(j.stdout)["kind"], "unknown_target")


def synthetic_pe() -> bytes:
    """Minimal PE32+ (base 0x140000000): one export, one import, two .pdata entries. No real code."""
    import struct as st

    img = bytearray(0x800)
    img[0:2] = b"MZ"
    st.pack_into("<I", img, 0x3C, 0x40)
    img[0x40:0x44] = b"PE\0\0"
    st.pack_into("<HH", img, 0x44, 0x8664, 1)
    st.pack_into("<H", img, 0x40 + 20, 240)
    opt = 0x58
    st.pack_into("<H", img, opt, 0x20B)
    st.pack_into("<Q", img, opt + 24, 0x140000000)
    st.pack_into("<I", img, opt + 56, 0x3000)
    dd = opt + 112
    st.pack_into("<II", img, dd, 0x1100, 0x40)  # exports
    st.pack_into("<II", img, dd + 8, 0x1200, 0x28)  # imports
    st.pack_into("<II", img, dd + 24, 0x1400, 24)  # .pdata
    sec = opt + 240
    img[sec : sec + 5] = b".text"
    st.pack_into("<IIII", img, sec + 8, 0x2000, 0x1000, 0x400, 0x200)  # raw 0x200 -> rva 0x1000
    raw = lambda rva: rva - 0x1000 + 0x200
    e = raw(0x1100)
    st.pack_into("<I", img, e + 12, 0x1180)
    st.pack_into("<III", img, e + 16, 1, 1, 1)
    st.pack_into("<III", img, e + 28, 0x1140, 0x1150, 0x1160)
    st.pack_into("<I", img, raw(0x1140), 0x1010)  # export function rva
    st.pack_into("<I", img, raw(0x1150), 0x1170)  # name rva
    img[raw(0x1170) : raw(0x1170) + 8] = b"Exported"
    img[raw(0x1180) : raw(0x1180) + 9] = b"game.exe\0"
    i = raw(0x1200)
    st.pack_into("<IIIII", img, i, 0x1240, 0, 0, 0x1280, 0x1260)
    st.pack_into("<Q", img, raw(0x1240), 0x12A0)
    st.pack_into("<Q", img, raw(0x1260), 0x12A0)
    img[raw(0x1280) : raw(0x1280) + 13] = b"KERNEL32.dll\0"
    img[raw(0x12A0) + 2 : raw(0x12A0) + 8] = b"Sleep\0"
    st.pack_into("<III", img, raw(0x1400), 0x1010, 0x1080, 0)
    st.pack_into("<III", img, raw(0x1400) + 12, 0x1100, 0x1110, 0)
    return bytes(img)


HARD = """## 1. XStore (`xgameruntime.XStore`)
| Slot | Function | Today | Set by |
|---|---|---|---|
| 68 / 69 | XStoreRegisterPackageLicenseLost | Trapping stub | #235 |
## 4. XSystem, XGame
| Slot | Function | Today | Set by |
|---|---|---|---|
| XSystem 5 | XSystemGetAppSpecificDeviceId | **Trapping stub** (owner decision) | #235 |
"""


class SymbolTests(unittest.TestCase):
    def test_pe_names(self):
        import guest_symbols as gs

        syms = gs.PeSymbols(synthetic_pe(), "game.exe")
        d = syms.describe(0x140001020)
        self.assertIn("fn 0x140001010+0x10", d)
        self.assertIn("export Exported+0x10", d)
        self.assertEqual(syms.describe(0x140001260), "game.exe import thunk KERNEL32.dll!Sleep")
        self.assertEqual(syms.describe(0x150000000), "")

    def test_render_annotates_and_hints(self):
        import guest_symbols as gs

        out = fr.render(report(), 4, gs.PeSymbols(synthetic_pe(), "game.exe"), HARD)
        self.assertIn("export Exported", out)
        self.assertIn("recompiler gap -> tools/triage_fault.py", out)

    def test_unimplemented_slot(self):
        rep = report("trap")
        rep["fault_record"].update(kind="unimplemented_slot", code_miss=False, detail="0x5")
        guid = "{12345678-1234-1234-1234-123456789ABC}"
        rep["message"] = f"com bridge: unimplemented XSystem {guid} slot 5 (XSystemGetAppSpecificDeviceId) called"
        out = fr.render(rep, 4, None, HARD)
        self.assertIn(f"XSystem {guid} slot 5 XSystemGetAppSpecificDeviceId", out)
        self.assertIn("owner decision", out)
        self.assertIn("hard stop: XSystemGetAppSpecificDeviceId -> **Trapping stub**", out)
        rep["message"] = rep["message"].replace("slot 5", "slot 9").replace("XSystemGetAppSpecificDeviceId", "Other")
        self.assertIn("not listed", fr.render(rep, 4, None, HARD))


if __name__ == "__main__":
    unittest.main()
