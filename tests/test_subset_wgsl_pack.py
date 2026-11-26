import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import inspect_wgsl_pack as wgpk  # noqa: E402
import subset_wgsl_pack as sp  # noqa: E402


class SubsetTest(unittest.TestCase):
    def test_v2_to_v1_cap_budget_priority(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            ents = [(h, 1, b"x" * n) for h, n in ((1, 10), (2, 20), (3, 30), (4, 5000))]
            (d / "a.wgsl.pack").write_bytes(wgpk.build_wgpk(ents, version=2))
            (d / "a.wgsl.pack.reflect.json").write_text(json.dumps({"schema": "s", "version": 3, "stages": {f"0x{h:016x}": {} for h in (1, 2, 3, 4)}}))
            self.assertEqual(sp.main([str(d / "a.wgsl.pack"), "--out", str(d / "o.wgsl.pack"), "--max-wgsl-bytes", "100", "--max-total-bytes", "40", "--priority-hashes", str(d / "p.txt")] if (d / "p.txt").write_text("0x0000000000000003") else []), 0)
            toc = sp.read_toc(d / "o.wgsl.pack")
            self.assertEqual({e[0] for e in toc}, {1, 3})  # 3 is priority (30), then 1 (10); 2 would exceed 40
            self.assertEqual(wgpk.build_wgpk([(1, 1, b"x" * 10), (3, 1, b"x" * 30)], version=1), (d / "o.wgsl.pack").read_bytes())
            self.assertEqual(set(json.loads((d / "o.wgsl.pack.reflect.json").read_text())["stages"]), {f"0x{h:016x}" for h in (1, 3)})


if __name__ == "__main__":
    unittest.main()
