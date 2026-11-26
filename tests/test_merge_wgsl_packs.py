import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import inspect_wgsl_pack as wgpk  # noqa: E402
import merge_wgsl_packs as mp  # noqa: E402


def write_pack(d: Path, name: str, entries, stages=None, version=None) -> Path:
    p = d / name
    p.write_bytes(wgpk.build_wgpk(entries, version=version))
    if stages is not None:
        (d / (name + ".reflect.json")).write_text(
            json.dumps({"schema": "playgta6-stage-reflection", "version": 3, "stages": stages})
        )
    return p


def stage(n_resolved: int, n_unresolved: int = 0):
    b = [{"group": 0, "binding": i, "kind": "uniform", "source": {"user_sgpr": 8}} for i in range(n_resolved)]
    b += [{"group": 0, "binding": 10 + i, "kind": "uniform"} for i in range(n_unresolved)]
    return {"stage": "vertex", "bindings": b}


class MergeTests(unittest.TestCase):
    def test_union_prefers_sidecar_and_resolved(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            a = write_pack(d, "a.wgsl.pack", [(3, "vertex", "// a3"), (1, "vertex", "// a1")],
                           {"0x0000000000000003": stage(0, 1), "0x0000000000000001": stage(1)})
            b = write_pack(d, "b.wgsl.pack", [(3, "vertex", "// b3"), (2, "fragment", "// b2"), (1, "vertex", "// b1")],
                           {"0x0000000000000003": stage(2)})
            out = d / "out" / "m.wgsl.pack"
            s = mp.merge([a, b], out)
            self.assertEqual(s["merged_entries"], 3)
            self.assertEqual(s["duplicates_dropped"], 2)
            bundle = wgpk.parse_wgpk(out.read_bytes())
            self.assertEqual([e.hash for e in bundle.entries], [1, 2, 3])
            texts = {e.hash: e.wgsl for e in bundle.entries}
            self.assertEqual(texts, {1: "// a1", 2: "// b2", 3: "// b3"})
            side = json.loads((out.parent / "m.wgsl.pack.reflect.json").read_text())
            self.assertEqual(sorted(side["stages"]), ["0x0000000000000001", "0x0000000000000003"])
            self.assertEqual(len(side["stages"]["0x0000000000000003"]["bindings"]), 2)

    def test_deterministic_and_tie_goes_to_first(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            a = write_pack(d, "a.wgsl.pack", [(5, "compute", "// A")])
            b = write_pack(d, "b.wgsl.pack", [(5, "compute", "// B")])
            o1, o2 = d / "o1.wgsl.pack", d / "o2.wgsl.pack"
            mp.merge([a, b], o1)
            mp.merge([a, b], o2)
            self.assertEqual(o1.read_bytes(), o2.read_bytes())
            self.assertEqual(wgpk.parse_wgpk(o1.read_bytes()).entries[0].wgsl, "// A")
            self.assertFalse((d / "o1.wgsl.pack.reflect.json").exists())

    def test_reads_v2_input_and_cli(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            a = write_pack(d, "a.wgsl.pack", [(9, "vertex", "// v2")], {"0x0000000000000009": stage(1)}, version=2)
            out = d / "m.wgsl.pack"
            self.assertEqual(mp.main([str(a), "--out", str(out)]), 0)
            bundle = wgpk.parse_wgpk(out.read_bytes())
            self.assertEqual(bundle.version, 1)
            self.assertEqual(bundle.entries[0].wgsl, "// v2")
            self.assertEqual(mp.main([str(a), "--out", str(a)]), 2)

    def test_v2_output_when_over_u32(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            a = write_pack(d, "a.wgsl.pack", [(1, "vertex", "// x" * 10)])
            old = mp.U32_MAX
            mp.U32_MAX = 40
            try:
                out = d / "m.wgsl.pack"
                s = mp.merge([a], out)
            finally:
                mp.U32_MAX = old
            self.assertEqual(s["pack_version"], 2)
            self.assertEqual(wgpk.parse_wgpk(out.read_bytes()).entries[0].wgsl, "// x" * 10)


if __name__ == "__main__":
    unittest.main()
