"""Tests for tools/stage_reflection_sidecar.py (synthetic translator JSON only)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import stage_reflection_sidecar as sidecar  # noqa: E402


def vertex_entry(with_keys: bool) -> dict:
    entry = {
        "hash_fnv1a64": "0x00000000000000a1",
        "stage": "vertex",
        "bindings": [{"group": 0, "binding": 0, "kind": "uniform", "name": "cb0", "min_binding_size_bytes": 16}],
        "binding_summary": {"vertex_input_mode": "attributes"},
    }
    if with_keys:
        entry["bindings"][0]["source"] = {"user_sgpr": 8, "table_byte_offset": 0}
        entry["vertex_inputs"] = [{"location": 0, "vb_slot": 0, "byte_offset": 8, "components": 2,
                                   "component_type": "f32"}]
    return entry


class SidecarTests(unittest.TestCase):
    def test_stage_with_join_keys_is_written(self):
        doc, skipped = sidecar.build([vertex_entry(True)])
        self.assertEqual(doc["schema"], "playgta6-stage-reflection")
        self.assertEqual(doc["version"], 1)
        stage = doc["stages"]["0x00000000000000a1"]
        self.assertEqual(stage["stage"], "vertex")
        self.assertEqual(stage["bindings"][0]["source"], {"user_sgpr": 8, "table_byte_offset": 0})
        self.assertEqual(stage["bindings"][0]["min_binding_size_bytes"], 16)
        self.assertEqual(stage["vertex_inputs"][0]["byte_offset"], 8)
        self.assertEqual(skipped, {})

    def test_today_output_without_keys_is_skipped(self):
        doc, skipped = sidecar.build([{"entries": [vertex_entry(False)]}])
        self.assertEqual(doc["stages"], {})
        self.assertEqual(skipped, {"binding without source": 1})
        no_inputs = vertex_entry(True)
        del no_inputs["vertex_inputs"]
        doc, skipped = sidecar.build([no_inputs])
        self.assertEqual(skipped, {"vertex stage without vertex_inputs": 1})

    def test_kind_spellings_and_cli(self):
        entry = {"hash_fnv1a64": "0x00000000000000b2", "stage": "fragment", "bindings": [
            {"group": 0, "binding": 1, "kind": "texture_2d", "source": {"user_sgpr": 8, "table_byte_offset": 32}},
            {"group": 0, "binding": 2, "kind": "sampler", "source": {"static_sampler": 0}}]}
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.json"
            out = Path(tmp) / "x.wgsl.pack.reflect.json"
            src.write_text(json.dumps(entry), encoding="utf-8")
            self.assertEqual(sidecar.main([str(src), "--out", str(out), "--json"]), 0)
            written = json.loads(out.read_text(encoding="utf-8"))
        kinds = [b["kind"] for b in written["stages"]["0x00000000000000b2"]["bindings"]]
        self.assertEqual(kinds, ["texture", "sampler"])


if __name__ == "__main__":
    unittest.main()
