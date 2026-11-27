import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import stop_message as sm  # noqa: E402

MD = """## 1. XStore (`xgameruntime.XStore`)

| Slot | Function | Today | Set by |
|---|---|---|---|
| 3 | XStoreDone | Implemented | #1 |
| 66 / 67 | Reg | Trapping stub | #1 |

## 4. XSystem, XGame

| Slot | Function | Today | Set by |
|---|---|---|---|
| XSystem 5 | XSystemGetAppSpecificDeviceId | **Trapping stub** | #1 |

## 7. Pending owner decisions

### 7.1 XSystemGetAppSpecificDeviceId (XSystem slot 5)

| Slot 14 | option table row | ignored |
"""


def rep(kind="unimplemented_slot", iface="xgameruntime.XSystem", slot=5, name="XSystemGetAppSpecificDeviceId"):
    rec = {"kind": kind, "va": "0x140001000"}
    if kind == "unimplemented_slot":
        rec["unimplemented_slot"] = {"interface": iface, "guid": "g", "slot": slot, "name": name}
    return {"status": "fault", "fault_record": rec}


class StopMessageTests(unittest.TestCase):
    def test_listed_slot_is_a_stop_not_a_crash(self):
        text = sm.select(rep(), MD, "R")
        self.assertEqual(text, "Stopped: XSystemGetAppSpecificDeviceId \u2014 this needs an owner decision "
                               "(see docs/HARD_STOPS.md \u00a77). Not a crash.")

    def test_stop_without_detail_section_and_without_name(self):
        text = sm.select(rep(iface="xgameruntime.XStore", slot=66, name=None), MD, "R")
        self.assertEqual(text, "Stopped: xstore slot 66 \u2014 this needs an owner decision (see docs/HARD_STOPS.md). Not a crash.")

    def test_unlisted_slot_and_implemented_row_are_crashes(self):
        for r in (rep(slot=9, name="Other"), rep(iface="xgameruntime.XStore", slot=3, name="XStoreDone")):
            text = sm.select(r, MD, "H:/runs/a")
            self.assertEqual(text, "Crashed: unimplemented_slot at 0x140001000 \u2014 report: H:/runs/a "
                                   "(see docs/WHEN_IT_CRASHES.md)")

    def test_other_fault_kinds_crash(self):
        text = sm.select(rep("unknown_target"), MD, "H:/runs/b")
        self.assertEqual(text, "Crashed: unknown_target at 0x140001000 \u2014 report: H:/runs/b (see docs/WHEN_IT_CRASHES.md)")

    def test_no_fault_no_message(self):
        self.assertIsNone(sm.select({"status": "ok"}, MD, "R"))
        self.assertIsNone(sm.select(None, MD))

    def test_without_hard_stops_file_everything_is_a_crash(self):
        self.assertTrue(sm.select(rep(), None, "R").startswith("Crashed:"))

    def test_pending_arg_for_the_harness(self):
        self.assertEqual(sm.pending_arg(MD), "xstore/66/-;xstore/67/-;xsystem/5/7")

    def test_real_hard_stops_doc_lists_the_device_id_with_section_7(self):
        md = (REPO_ROOT / "docs" / "HARD_STOPS.md").read_text(encoding="utf-8")
        self.assertIn("xsystem/5/7", sm.pending_arg(md).split(";"))

    def test_browser_report_fault_block_without_fault_record(self):
        r = {"status": "fault", "fault": {"kind": "INDIRECT_CALL_MISS", "trap_kind": "INDIRECT_CALL_MISS", "va": "0x1400abcd0"},
             "fault_record": None}
        self.assertEqual(sm.select(r, MD, "work/browser-boot/runs/x"),
                         "Crashed: INDIRECT_CALL_MISS at 0x1400abcd0 — report: work/browser-boot/runs/x "
                         "(see docs/WHEN_IT_CRASHES.md)")

    def test_browser_report_with_unimplemented_slot_record_is_a_stop(self):
        r = rep()
        r["fault"] = {"kind": "unimplemented_slot", "va": "0x1"}
        self.assertTrue(sm.select(r, MD, "R").startswith("Stopped:"))

    def test_browser_report_without_fault_is_silent(self):
        self.assertIsNone(sm.select({"status": "ok", "fault": None, "fault_record": None}, MD, "R"))

    def test_cli_reads_browser_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "boot-report.json").write_text(json.dumps({"fault": {"kind": "INDIRECT_CALL_MISS", "va": "0x1"}}))
            self.assertEqual(sm.main([tmp]), 0)

    def test_cli_prints_line_and_exits_1_without_fault(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "boot-report.json").write_text(json.dumps({"status": "ok"}))
            self.assertEqual(sm.main([tmp]), 1)


if __name__ == "__main__":
    unittest.main()
