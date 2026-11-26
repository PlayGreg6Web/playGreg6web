"""Tests for tools/program_census.py (synthetic packs and reports only)."""

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import program_census as pc  # noqa: E402


def h(n: int) -> str:
    return "0x%016x" % n


def entry(n, stage="fragment", unsupported=0, bindings=(), mesh=None, validated=True):
    e = {"hash_fnv1a64": h(n), "stage": stage, "naga_validated": validated, "unsupported_op_count": unsupported,
         "bindings": list(bindings)}
    if mesh:
        e["mesh_output"] = mesh
    return e


def b(kind, source=True):
    d = {"group": 0, "binding": 0, "kind": kind, "name": "x"}
    if source:
        d["source"] = {"user_sgpr": 8, "table_byte_offset": 0}
    return d


def wgpk(entries, version=1):
    codes = {"vertex": 1, "fragment": 2, "compute": 3}
    out = struct.pack("<4sIII", b"WGPK", version, len(entries), 0)
    for n, stage in entries:
        out += struct.pack("<QB3sII", n, codes[stage], b"\0\0\0", 0, 0) if version == 1 else struct.pack("<QB3sIQ", n, codes[stage], b"\0\0\0", 0, 0)
    return out


REPORT = {
    "total_blobs": 10, "translated_ok": 6, "skipped_or_failed": 4, "failed": 0, "skipped_oversize": 4,
    "entries": [
        entry(1),
        entry(2, unsupported=3),
        entry(3, bindings=[b("texture_2d", False), b("sampler")]),
        entry(4, "compute", bindings=[b("storage-texture", False)]),
        entry(5, "compute", mesh={"max_vertices": 256, "max_primitives": 256, "vertex_words": 40}),
        entry(6, "compute", mesh={"max_vertices": 64, "max_primitives": 64, "vertex_words": 12}),
    ],
}


class CensusTests(unittest.TestCase):
    def test_reasons_and_primary_attribution(self):
        c = pc.build_census(pack_report=REPORT)
        reasons = {r["reason"]: r for r in c["packed_reasons"]}
        self.assertEqual(reasons["unsupported_ops"]["primary_count"], 1)
        self.assertEqual(reasons["binding_without_source"]["primary_count"], 1)
        self.assertEqual(reasons["writable_binding_without_source"]["primary_count"], 1)
        self.assertEqual(reasons["mesh_oversize"]["primary_count"], 1)
        self.assertEqual(c["mesh"], {"programs": 2, "over_limit": 1, "largest_record_bytes": [4 * (4 + 256 * 40 + 768), 4 * (4 + 64 * 12 + 192)]})
        self.assertEqual(c["not_in_pack"][0]["count"], 4)
        kinds = {k["kind"]: k["bindings"] for k in c["unresolved_bindings_by_kind"]}
        self.assertEqual(kinds, {"texture_2d": 1, "storage_texture": 1})

    def test_missing_reflection_entry(self):
        reflect = {"stages": {h(n): {"stage": "fragment", "bindings": []} for n in (1, 2, 3, 4, 5)}}
        toc = {h(n): "fragment" for n in range(1, 8)}
        c = pc.build_census(pack_report=REPORT, reflect=reflect, pack_toc=toc)
        reasons = {r["reason"]: r for r in c["packed_reasons"]}
        self.assertEqual(reasons["no_reflection_entry"]["tagged_count"], 2)
        self.assertEqual(c["packed_programs"]["total"], 7)

    def test_pack_toc_v1_and_v2(self):
        with tempfile.TemporaryDirectory() as d:
            for v in (1, 2):
                p = Path(d) / f"p{v}.pack"
                p.write_bytes(wgpk([(7, "vertex"), (9, "compute")], v))
                self.assertEqual(pc.read_pack_toc(p), {h(7): "vertex", h(9): "compute"})
            bad = Path(d) / "bad.pack"
            bad.write_bytes(b"NOPE" + b"\0" * 20)
            with self.assertRaises(ValueError):
                pc.read_pack_toc(bad)

    def test_boot_log_and_report(self):
        log = [
            "d3d12_xs: no shader module for DXIL identity 0x00000000000000aa (compute shader, DXIL kind 5; dispatches with it are skipped)\n",
            "d3d12_xs: reflection join failed (vertex 0x0000000000000001, pixel 0x0000000000000003):\n",
            "g0 b4 (texture): no root parameter covers s12\n",
            "g0 b5 (texture): no root parameter covers s14\n",
            "d3d12_xs: mesh pipeline layout join failed (mesh 0x0000000000000005):\n",
            "g0 b1 (uniform): no root parameter covers s9\n",
        ]
        rep = {"gpu_work": {"draws_no_shader_module": 5, "draws_dropped_by_bridge": 2, "dispatches_dropped_by_bridge": 1, "draws_issued": 100}}
        c = pc.build_census(pack_report=REPORT, boot_report=rep, boot_log=log)
        rt = c["runtime"]
        self.assertEqual(rt["draws_lost_total"], 7)
        self.assertEqual(rt["dispatches_lost_total"], 1)
        self.assertEqual(rt["boot_log"]["join_failure_reasons"], [{"reason": "no root parameter covers sN", "bindings": 3, "programs": 3}])
        self.assertEqual(rt["boot_log"]["join_failed_mesh_programs"], 1)
        self.assertEqual(rt["boot_log"]["no_shader_module_programs"], {"compute": 1})
        self.assertIn("per_program_drop_counts", rt)

    def test_drop_counts_weighting(self):
        c = pc.build_census(pack_report=REPORT, drop_counts={h(2): 10, h(3): 5, h(5): 7, h(99): 3, h(1): 1})
        w = {r["reason"]: r["draws_lost"] for r in c["runtime"]["drops_weighted_by_reason"]}
        self.assertEqual(w, {"unsupported_ops": 10, "mesh_oversize": 7, "binding_without_source": 5, "not_in_pack": 3, "packed_no_static_reason": 1})
        self.assertEqual(sum(w.values()), 26)
        self.assertEqual(c["runtime"]["top_dropped_programs"][0]["hash"], h(2))

    def test_boot_report_drops_by_program(self):
        report = {"gpu_work": {"draws_no_shader_module": 12, "drops_by_program": {h(2): 10, h(5): 2}}, "gpu_drop_table": {"entries": []}}
        c = pc.build_census(pack_report=REPORT, boot_report=report)
        top = c["runtime"]["top_dropped_programs"]
        self.assertEqual([(t["hash"], t["draws_lost"]) for t in top], [(h(2), 10), (h(5), 2)])
        self.assertNotIn("per_program_drop_counts", c["runtime"])

    def test_cli_writes_json_and_markdown_without_payloads(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "r.json").write_text(json.dumps(REPORT))
            self.assertEqual(pc.main(["--pack-report", str(d / "r.json"), "--json-out", str(d / "o.json"), "--md-out", str(d / "o.md")]), 0)
            self.assertEqual(json.loads((d / "o.json").read_text())["schema"], pc.SCHEMA)
            self.assertIn("# Program census", (d / "o.md").read_text())

    def test_sidecar_unresolved_heap_and_missing_stages(self):
        def sb(kind, **src):
            return {"group": 0, "binding": 0, "kind": kind, "source": src}
        reflect = {"stages": {
            h(1): {"stage": "fragment", "bindings": [sb("texture", unresolved=True), sb("sampler", unresolved=True),
                                                      sb("texture", cbv_indexed_resource_heap={"cb": 1})]},
            h(2): {"stage": "compute", "bindings": [sb("texture", unresolved=True), sb("storage", lane_indexed_resource_heap={})]},
        }}
        toc = {h(1): "fragment", h(2): "compute", h(3): "compute"}
        c = pc.build_census(reflect=reflect, pack_toc=toc, drop_counts={h(1): 7, h(2): 3})
        un = {u["kind"]: (u["bindings"], u["programs"]) for u in c["sidecar_unresolved_by_kind"]}
        self.assertEqual(un, {"texture": (2, 2), "sampler": (1, 1)})
        heap = {u["kind"]: u["bindings"] for u in c["sidecar_heap_sourced_by_kind"]}
        self.assertEqual(heap, {"texture:cbv_indexed_resource_heap": 1, "storage:lane_indexed_resource_heap": 1})
        self.assertEqual(c["sidecar_missing_stages"]["total"], 1)
        self.assertEqual(c["sidecar_missing_stages"]["by_stage"], {"compute": 1})
        w = {r["kind"]: r["draws_lost"] for r in c["runtime"]["drops_weighted_by_sidecar_unresolved_kind"]}
        self.assertEqual(w, {"texture": 10, "sampler": 7})
        self.assertIn("Sidecar unresolved bindings by kind", pc.render_markdown(c))


if __name__ == "__main__":
    unittest.main()
