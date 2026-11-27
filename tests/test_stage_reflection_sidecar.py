"""Tests for tools/stage_reflection_sidecar.py (synthetic translator JSON only)."""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import stage_reflection_sidecar as sidecar  # noqa: E402

_STAGE_TO_CODE = {"vertex": 1, "fragment": 2, "compute": 3}


def build_synthetic_wgpk(entries: list[tuple[int, str, bytes]], *, flags: int = 0) -> bytes:
    """Build a valid WGPK v1 binary archive from (hash_int, stage_name, payload_bytes) tuples."""
    data_start = sidecar.WGPK_HEADER_BYTES + len(entries) * sidecar.WGPK_ENTRY_BYTES
    header = sidecar.WGPK_HEADER_STRUCT.pack(sidecar.WGPK_MAGIC, sidecar.WGPK_VERSION, len(entries), flags)
    toc = bytearray()
    payloads = bytearray()
    cursor = data_start
    for hash_int, stage_name, payload in entries:
        toc.extend(
            sidecar.WGPK_ENTRY_STRUCT.pack(
                hash_int,
                _STAGE_TO_CODE[stage_name],
                b"\x00\x00\x00",
                cursor,
                len(payload),
            )
        )
        payloads.extend(payload)
        cursor += len(payload)
    return bytes(header) + bytes(toc) + bytes(payloads)


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

    def test_heap_sources_are_kept(self):
        sampler_heap = {"sampler_heap": {"base_user_sgpr": 11, "stride": 16, "index": {"user_sgpr": 9, "shift": 4}}}
        resource_heap = {"resource_heap": {"base_user_sgpr": 4, "stride": 32, "index": {"user_sgpr": 5, "shift": 5}}}
        entry = {"hash_fnv1a64": "0x00000000000000c3", "stage": "fragment", "bindings": [
            {"group": 0, "binding": 0, "kind": "texture", "view_dimension": "2d-array", "source": resource_heap},
            {"group": 0, "binding": 1, "kind": "sampler", "source": sampler_heap}]}
        doc, skipped = sidecar.build([entry])
        self.assertEqual(skipped, {})
        bindings = doc["stages"]["0x00000000000000c3"]["bindings"]
        self.assertEqual(bindings[0]["source"], resource_heap)
        self.assertEqual(bindings[1]["source"], sampler_heap)

    def test_malformed_heap_source_is_skipped(self):
        bad = [
            {"sampler_heap": {"base_user_sgpr": 11, "stride": 16}},
            {"sampler_heap": {"base_user_sgpr": 11, "stride": 16, "index": {"user_sgpr": 9}}},
            {"resource_heap": {"base_user_sgpr": 4, "stride": 32, "index": {"user_sgpr": 5, "shift": 40}}},
            {"resource_heap": {"base_user_sgpr": "4", "stride": 32, "index": {"user_sgpr": 5, "shift": 5}}},
            {"sampler_heap": 3},
        ]
        for i, source in enumerate(bad):
            entry = {"hash_fnv1a64": f"0x{i:016x}", "stage": "fragment",
                     "bindings": [{"group": 0, "binding": 0, "kind": "sampler", "source": source}]}
            doc, skipped = sidecar.build([entry])
            self.assertEqual(doc["stages"], {}, source)
            self.assertEqual(skipped, {"malformed heap source": 1}, source)

    def test_cbv_indexed_heap_sources_are_kept(self):
        # G62 (#482): the heap index is a dword of a root CBV, a table CBV or root constants.
        sources = [
            {"cbv_indexed_resource_heap": {"cbv": {"user_sgpr": 10, "root_descriptor": True},
                                           "byte_offset": 16, "shift": 5, "stride": 32}},
            {"cbv_indexed_resource_heap": {"cbv": {"user_sgpr": 8, "table_byte_offset": 64},
                                           "byte_offset": 4, "shift": 5, "bit_offset": 16, "mask": 0xFFFF}},
            {"cbv_indexed_sampler_heap": {"cbv": {"user_sgpr": 12, "root_constants": True},
                                          "byte_offset": 0, "shift": 4}},
        ]
        entry = {"hash_fnv1a64": "0x00000000000000c4", "stage": "fragment", "bindings": [
            {"group": 0, "binding": 0, "kind": "texture", "source": sources[0]},
            {"group": 0, "binding": 1, "kind": "texture", "source": sources[1]},
            {"group": 0, "binding": 2, "kind": "sampler", "source": sources[2]}]}
        doc, skipped = sidecar.build([entry])
        self.assertEqual(skipped, {})
        bindings = doc["stages"]["0x00000000000000c4"]["bindings"]
        self.assertEqual([b["source"] for b in bindings], sources)
        summary = sidecar.summarize_stages(doc["stages"])
        self.assertEqual(summary["cbv_indexed_heap_stages"], 1)
        self.assertEqual(summary["cbv_indexed_heap_bindings"], 3)
        self.assertEqual(summary["resource_heap_bindings"], 0)

    def test_malformed_cbv_indexed_heap_source_is_skipped(self):
        root_cbv = {"user_sgpr": 10, "root_descriptor": True}
        bad = [
            {"cbv_indexed_resource_heap": {"byte_offset": 0, "shift": 5}},
            {"cbv_indexed_resource_heap": {"cbv": {"user_sgpr": 10}, "byte_offset": 0, "shift": 5}},
            {"cbv_indexed_resource_heap": {"cbv": {"user_sgpr": 10, "root_descriptor": True, "table_byte_offset": 0},
                                           "byte_offset": 0, "shift": 5}},
            {"cbv_indexed_resource_heap": {"cbv": {"user_sgpr": 8, "table_byte_offset": 40}, "byte_offset": 0, "shift": 5}},
            {"cbv_indexed_resource_heap": {"cbv": root_cbv, "byte_offset": 6, "shift": 5}},
            {"cbv_indexed_resource_heap": {"cbv": root_cbv, "byte_offset": 0, "shift": 32}},
            {"cbv_indexed_resource_heap": {"cbv": root_cbv, "byte_offset": 0, "shift": 5, "bit_offset": 32}},
            {"cbv_indexed_resource_heap": {"cbv": root_cbv, "byte_offset": 0, "shift": 5, "mask": 0}},
            {"cbv_indexed_resource_heap": {"cbv": root_cbv, "byte_offset": "0", "shift": 5}},
            {"cbv_indexed_sampler_heap": 3},
        ]
        for i, source in enumerate(bad):
            entry = {"hash_fnv1a64": f"0x{i + 32:016x}", "stage": "fragment",
                     "bindings": [{"group": 0, "binding": 0, "kind": "texture", "source": source}]}
            doc, skipped = sidecar.build([entry])
            self.assertEqual(doc["stages"], {}, source)
            self.assertEqual(skipped, {"malformed CB-indexed heap source": 1}, source)

    def test_lane_indexed_heap_sources_are_kept(self):
        # G62 (#482): per-lane bindless, a window of max_count descriptors, base 0 or a CB dword.
        sources = [
            {"lane_indexed_resource_heap": {"max_count": 64, "shift": 5, "stride": 32}},
            {"lane_indexed_sampler_heap": {"max_count": 16, "shift": 4, "window_base": {
                "cbv": {"user_sgpr": 10, "root_descriptor": True}, "byte_offset": 8, "bit_offset": 8, "mask": 0xFF}}},
        ]
        entry = {"hash_fnv1a64": "0x00000000000000c5", "stage": "fragment", "bindings": [
            {"group": 0, "binding": 0, "kind": "texture", "source": sources[0]},
            {"group": 0, "binding": 1, "kind": "sampler", "source": sources[1]}]}
        doc, skipped = sidecar.build([entry])
        self.assertEqual(skipped, {})
        bindings = doc["stages"]["0x00000000000000c5"]["bindings"]
        self.assertEqual([b["source"] for b in bindings], sources)
        summary = sidecar.summarize_stages(doc["stages"])
        self.assertEqual(summary["lane_indexed_heap_stages"], 1)
        self.assertEqual(summary["lane_indexed_heap_bindings"], 2)
        self.assertEqual(summary["cbv_indexed_heap_bindings"], 0)

    def test_malformed_lane_indexed_heap_source_is_skipped(self):
        root_cbv = {"user_sgpr": 10, "root_descriptor": True}
        bad = [
            {"lane_indexed_resource_heap": {"shift": 5}},
            {"lane_indexed_resource_heap": {"max_count": 0, "shift": 5}},
            {"lane_indexed_resource_heap": {"max_count": 257, "shift": 5}},
            {"lane_indexed_resource_heap": {"max_count": 8, "shift": 32}},
            {"lane_indexed_resource_heap": {"max_count": 8, "shift": 5, "stride": -1}},
            {"lane_indexed_resource_heap": {"max_count": 8, "shift": 5, "window_base": 0}},
            {"lane_indexed_resource_heap": {"max_count": 8, "shift": 5, "window_base": {"byte_offset": 0}}},
            {"lane_indexed_resource_heap": {"max_count": 8, "shift": 5,
                                            "window_base": {"cbv": root_cbv, "byte_offset": 6}}},
            {"lane_indexed_resource_heap": {"max_count": 8, "shift": 5,
                                            "window_base": {"cbv": root_cbv, "byte_offset": 0, "mask": 0}}},
            {"lane_indexed_sampler_heap": 3},
        ]
        for i, source in enumerate(bad):
            entry = {"hash_fnv1a64": f"0x{i + 64:016x}", "stage": "fragment",
                     "bindings": [{"group": 0, "binding": 0, "kind": "texture", "source": source}]}
            doc, skipped = sidecar.build([entry])
            self.assertEqual(doc["stages"], {}, source)
            self.assertEqual(skipped, {"malformed lane-indexed heap source": 1}, source)

    def test_inline_sampler_is_kept(self):
        # Hand-made GFX10 S#: clamp-to-edge bilinear, linear mips, max LOD 15.
        source = {"inline_sampler": {"sdwords": [0x92, 0x00F00000, 0x08500000, 0]}}
        entry = {"hash_fnv1a64": "0x00000000000000d4", "stage": "fragment",
                 "bindings": [{"group": 0, "binding": 34, "kind": "sampler", "source": source}]}
        doc, skipped = sidecar.build([entry])
        self.assertEqual(skipped, {})
        self.assertEqual(doc["stages"]["0x00000000000000d4"]["bindings"][0]["source"], source)

    def test_malformed_inline_sampler_is_skipped(self):
        bad = [
            {"inline_sampler": {"sdwords": [1, 2, 3]}},
            {"inline_sampler": {"sdwords": [1, 2, 3, -1]}},
            {"inline_sampler": {"sdwords": [1, 2, 3, 0x100000000]}},
            {"inline_sampler": {"sdwords": [1, 2, 3, "4"]}},
            {"inline_sampler": {"sdwords": [1, 2, 3, True]}},
            {"inline_sampler": [1, 2, 3, 4]},
        ]
        for i, source in enumerate(bad):
            entry = {"hash_fnv1a64": f"0x{i + 16:016x}", "stage": "fragment",
                     "bindings": [{"group": 0, "binding": 0, "kind": "sampler", "source": source}]}
            doc, skipped = sidecar.build([entry])
            self.assertEqual(doc["stages"], {}, source)
            self.assertEqual(skipped, {"malformed inline sampler": 1}, source)

    def test_merge_existing_preserves_and_overrides(self):
        existing_doc = {
            "schema": "playgta6-stage-reflection",
            "version": 1,
            "stages": {
                "0x0000000000000001": {
                    "stage": "fragment",
                    "bindings": [
                        {"group": 0, "binding": 0, "kind": "uniform",
                         "source": {"user_sgpr": 2, "table_byte_offset": 0}}
                    ],
                },
                "0x00000000000000a1": {
                    "stage": "vertex",
                    "bindings": [
                        {"group": 0, "binding": 0, "kind": "uniform",
                         "source": {"user_sgpr": 4, "table_byte_offset": 16}}
                    ],
                },
            },
        }
        new_entries = {
            "entries": [
                # Overrides 0x...00a1 with user_sgpr=8, table_byte_offset=0
                vertex_entry(True),
                # Adds brand-new compute stage 0x...0002
                {
                    "hash_fnv1a64": "0x0000000000000002",
                    "stage": "compute",
                    "bindings": [
                        {"group": 0, "binding": 0, "kind": "storage",
                         "source": {"user_sgpr": 6, "table_byte_offset": 0}}
                    ],
                },
                # Skipped entry for existing hash 0x...0001 must NOT clobber preserved stage
                {
                    "hash_fnv1a64": "0x0000000000000001",
                    "stage": "fragment",
                    "bindings": [{"group": 0, "binding": 0, "kind": "uniform", "name": "cb0"}],
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            existing_path = Path(tmp) / "existing.reflect.json"
            src_path = Path(tmp) / "new.json"
            out_path = Path(tmp) / "merged.reflect.json"
            existing_path.write_text(json.dumps(existing_doc), encoding="utf-8")
            src_path.write_text(json.dumps(new_entries), encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = sidecar.main([
                    str(src_path),
                    "--out", str(out_path),
                    "--merge-existing", str(existing_path),
                    "--json",
                ])
            self.assertEqual(rc, 0)
            summary = json.loads(stdout.getvalue())
            written = json.loads(out_path.read_text(encoding="utf-8"))

        self.assertEqual(summary, {
            "sidecar_version": 1,
            "stages_written": 3,
            "skipped": {"binding without source": 1},
            "merged_existing": 2,
            "by_stage": {"vertex": 1, "fragment": 1, "compute": 1},
            "inline_sampler_stages": 0,
            "inline_sampler_bindings": 0,
            "resource_heap_stages": 0,
            "resource_heap_bindings": 0,
            "cbv_indexed_heap_stages": 0,
            "cbv_indexed_heap_bindings": 0,
            "lane_indexed_heap_stages": 0,
            "lane_indexed_heap_bindings": 0,
            "root_descriptor_table_offset_stages": 0,
            "root_descriptor_table_offset_bindings": 0,
            "rt_arena_stages": 0,
            "rt_arena_bindings": 0,
            "vertex_index_stages": 0,
            "mesh_output_stages": 0,
            "unresolved_stages": 0,
            "unresolved_bindings": 0,
            "unresolved_by_kind": {},
            "unresolved_by_reason": {},
            "gap_breakdown": {
                "skipped_total": 1,
                "skipped_by_cause": [
                    {"cause": "binding without source", "count": 1, "pct_of_skipped": 100.0}
                ],
                "unresolved_stages": 0,
                "unresolved_bindings": 0,
                "unresolved_by_kind": {},
                "unresolved_by_reason": {},
            },
        })
        self.assertEqual(set(written["stages"].keys()), {
            "0x0000000000000001",
            "0x00000000000000a1",
            "0x0000000000000002",
        })
        # Preserved stage kept intact
        self.assertEqual(
            written["stages"]["0x0000000000000001"]["bindings"][0]["source"],
            {"user_sgpr": 2, "table_byte_offset": 0},
        )
        # Matching hash overridden by new entry
        self.assertEqual(
            written["stages"]["0x00000000000000a1"]["bindings"][0]["source"],
            {"user_sgpr": 8, "table_byte_offset": 0},
        )
        # New stage added
        self.assertEqual(written["stages"]["0x0000000000000002"]["stage"], "compute")

    def test_merge_existing_missing_file_and_in_place_update(self):
        entry1 = vertex_entry(True)
        entry2 = {
            "hash_fnv1a64": "0x00000000000000b2",
            "stage": "fragment",
            "bindings": [
                {"group": 0, "binding": 0, "kind": "sampler", "source": {"static_sampler": 1}}
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            in1 = Path(tmp) / "in1.json"
            in2 = Path(tmp) / "in2.json"
            out = Path(tmp) / "pack.reflect.json"
            in1.write_text(json.dumps(entry1), encoding="utf-8")
            in2.write_text(json.dumps(entry2), encoding="utf-8")

            # First run: --merge-existing does not exist yet
            stdout1 = io.StringIO()
            with redirect_stdout(stdout1):
                rc1 = sidecar.main([str(in1), "--out", str(out), "--merge-existing", str(out), "--json"])
            self.assertEqual(rc1, 0)
            self.assertEqual(json.loads(stdout1.getvalue()), {
                "sidecar_version": 1,
                "stages_written": 1,
                "skipped": {},
                "merged_existing": 0,
                "by_stage": {"vertex": 1, "fragment": 0, "compute": 0},
                "inline_sampler_stages": 0,
                "inline_sampler_bindings": 0,
                "resource_heap_stages": 0,
                "resource_heap_bindings": 0,
                "cbv_indexed_heap_stages": 0,
                "cbv_indexed_heap_bindings": 0,
                "lane_indexed_heap_stages": 0,
                "lane_indexed_heap_bindings": 0,
                "root_descriptor_table_offset_stages": 0,
                "root_descriptor_table_offset_bindings": 0,
                "rt_arena_stages": 0,
                "rt_arena_bindings": 0,
                "vertex_index_stages": 0,
                "mesh_output_stages": 0,
                "unresolved_stages": 0,
                "unresolved_bindings": 0,
                "unresolved_by_kind": {},
                "unresolved_by_reason": {},
                "gap_breakdown": {
                    "skipped_total": 0,
                    "skipped_by_cause": [],
                    "unresolved_stages": 0,
                    "unresolved_bindings": 0,
                    "unresolved_by_kind": {},
                    "unresolved_by_reason": {},
                },
            })

            # Second run: in-place merge onto the file written in run 1
            stdout2 = io.StringIO()
            with redirect_stdout(stdout2):
                rc2 = sidecar.main([str(in2), "--out", str(out), "--merge-existing", str(out), "--json"])
            self.assertEqual(rc2, 0)
            self.assertEqual(json.loads(stdout2.getvalue()), {
                "sidecar_version": 1,
                "stages_written": 2,
                "skipped": {},
                "merged_existing": 1,
                "by_stage": {"vertex": 1, "fragment": 1, "compute": 0},
                "inline_sampler_stages": 0,
                "inline_sampler_bindings": 0,
                "resource_heap_stages": 0,
                "resource_heap_bindings": 0,
                "cbv_indexed_heap_stages": 0,
                "cbv_indexed_heap_bindings": 0,
                "lane_indexed_heap_stages": 0,
                "lane_indexed_heap_bindings": 0,
                "root_descriptor_table_offset_stages": 0,
                "root_descriptor_table_offset_bindings": 0,
                "rt_arena_stages": 0,
                "rt_arena_bindings": 0,
                "vertex_index_stages": 0,
                "mesh_output_stages": 0,
                "unresolved_stages": 0,
                "unresolved_bindings": 0,
                "unresolved_by_kind": {},
                "unresolved_by_reason": {},
                "gap_breakdown": {
                    "skipped_total": 0,
                    "skipped_by_cause": [],
                    "unresolved_stages": 0,
                    "unresolved_bindings": 0,
                    "unresolved_by_kind": {},
                    "unresolved_by_reason": {},
                },
            })
            written = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(set(written["stages"].keys()), {"0x00000000000000a1", "0x00000000000000b2"})

    def test_merge_existing_invalid_schema_fails_cleanly(self):
        bad_payloads = [
            "{not valid json",
            json.dumps([1, 2, 3]),
            json.dumps({"schema": "wrong-schema", "version": 1, "stages": {}}),
            json.dumps({"schema": "playgta6-stage-reflection", "version": 0, "stages": {}}),
            json.dumps({"schema": "playgta6-stage-reflection", "version": 4, "stages": {}}),
            json.dumps({"schema": "playgta6-stage-reflection", "version": True, "stages": {}}),
            json.dumps({"schema": "playgta6-stage-reflection", "version": 1, "stages": []}),
            json.dumps({
                "schema": "playgta6-stage-reflection",
                "version": 1,
                "stages": {"0x1": {"stage": "geometry", "bindings": []}},
            }),
            json.dumps({
                "schema": "playgta6-stage-reflection",
                "version": 1,
                "stages": {"0x1": {"stage": "vertex", "bindings": "not-a-list"}},
            }),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.json"
            src.write_text(json.dumps(vertex_entry(True)), encoding="utf-8")
            for idx, payload in enumerate(bad_payloads):
                bad_path = Path(tmp) / f"bad_{idx}.reflect.json"
                out_path = Path(tmp) / f"out_{idx}.reflect.json"
                bad_path.write_text(payload, encoding="utf-8")
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    rc = sidecar.main([
                        str(src),
                        "--out", str(out_path),
                        "--merge-existing", str(bad_path),
                        "--json",
                    ])
                self.assertEqual(rc, 1, payload)
                self.assertIn("error:", stderr.getvalue(), payload)
                self.assertFalse(out_path.exists(), payload)

    def test_json_output_without_merge_existing_omits_merged_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.json"
            out = Path(tmp) / "out.reflect.json"
            src.write_text(json.dumps(vertex_entry(True)), encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = sidecar.main([str(src), "--out", str(out), "--json"])
            self.assertEqual(rc, 0)
            summary = json.loads(stdout.getvalue())
        self.assertEqual(summary, {
            "sidecar_version": 1,
            "stages_written": 1,
            "skipped": {},
            "by_stage": {"vertex": 1, "fragment": 0, "compute": 0},
            "inline_sampler_stages": 0,
            "inline_sampler_bindings": 0,
            "resource_heap_stages": 0,
            "resource_heap_bindings": 0,
            "cbv_indexed_heap_stages": 0,
            "cbv_indexed_heap_bindings": 0,
            "lane_indexed_heap_stages": 0,
            "lane_indexed_heap_bindings": 0,
            "root_descriptor_table_offset_stages": 0,
            "root_descriptor_table_offset_bindings": 0,
            "rt_arena_stages": 0,
            "rt_arena_bindings": 0,
            "vertex_index_stages": 0,
            "mesh_output_stages": 0,
            "unresolved_stages": 0,
            "unresolved_bindings": 0,
            "unresolved_by_kind": {},
            "unresolved_by_reason": {},
            "gap_breakdown": {
                "skipped_total": 0,
                "skipped_by_cause": [],
                "unresolved_stages": 0,
                "unresolved_bindings": 0,
                "unresolved_by_kind": {},
                "unresolved_by_reason": {},
            },
        })
        self.assertNotIn("merged_existing", summary)
        self.assertNotIn("diff", summary)

    def test_report_missing_sources_counts_sorting_and_sample_cap(self):
        entries = []
        # 7 fragment/pixel stages missing texture (using both "texture" and "texture_2d" spellings)
        # to verify kind normalization and the 5-sample hash cap.
        for i in range(7):
            entries.append({
                "hash_fnv1a64": f"0x000000000000100{i}",
                "stage": "pixel" if i % 2 else "fragment",
                "bindings": [
                    {"group": 0, "binding": 0, "kind": "uniform", "name": "cb0",
                     "source": {"user_sgpr": 2, "table_byte_offset": 0}},
                    {"group": 0, "binding": 1, "kind": "texture_2d" if i % 2 else "texture", "name": "t_albedo"},
                ],
            })
        # 2 vertex stages missing cb_frame AND 1 of them also missing s_linear
        entries.append({
            "hash_fnv1a64": "0x0000000000002001",
            "stage": "vertex",
            "bindings": [
                {"group": 0, "binding": 0, "kind": "uniform", "name": "cb_frame"},
                {"group": 0, "binding": 1, "kind": "sampler", "name": "s_linear"},
            ],
        })
        entries.append({
            "hash_fnv1a64": "0x0000000000002002",
            "stage": "vertex",
            "bindings": [
                {"group": 0, "binding": 0, "kind": "uniform", "name": "cb_frame"},
            ],
        })
        # 1 compute stage with 2 missing bindings of the exact same pattern (sample_hashes deduplicated)
        entries.append({
            "hash_fnv1a64": "0x0000000000003001",
            "stage": "compute",
            "bindings": [
                {"group": 0, "binding": 0, "kind": "read-only-storage", "name": "buf_in"},
                {"group": 0, "binding": 1, "kind": "read_only_storage", "name": "buf_in"},
            ],
        })
        # Non-missing-source skips must NOT appear in the missing-sources report
        entries.append({
            "hash_fnv1a64": "0x0000000000004001",
            "stage": "fragment",
            "bindings": [{"group": 0, "binding": 0, "kind": "sampler", "source": {"sampler_heap": 3}}],
        })
        # Valid stage must NOT appear in the missing-sources report
        entries.append(vertex_entry(True))

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "pack.json"
            out = Path(tmp) / "pack.reflect.json"
            report_path = Path(tmp) / "missing_sources.json"
            src.write_text(json.dumps({"entries": entries}), encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = sidecar.main([
                    str(src),
                    "--out", str(out),
                    "--report-missing-sources", str(report_path),
                    "--json",
                ])
            self.assertEqual(rc, 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["schema"], "playgta6-stage-reflection-missing-sources")
        self.assertEqual(report["version"], 1)
        self.assertEqual(report["total_skipped_missing_source"], 10)
        self.assertEqual(report["by_stage"], {"vertex": 2, "fragment": 7, "compute": 1})

        top = report["top_missing_bindings"]
        # Sorted descending by count, then deterministic tie-break on (stage, kind, name)
        self.assertEqual(
            [(item["stage"], item["kind"], item["name"], item["count"]) for item in top],
            [
                ("fragment", "texture", "t_albedo", 7),
                ("compute", "read_only_storage", "buf_in", 2),
                ("vertex", "uniform", "cb_frame", 2),
                ("vertex", "sampler", "s_linear", 1),
            ],
        )
        # Sample hashes capped at 5
        self.assertEqual(len(top[0]["sample_hashes"]), 5)
        self.assertEqual(top[0]["sample_hashes"], [f"0x000000000000100{i}" for i in range(5)])
        # Duplicate bindings on the same stage deduplicate sample_hashes
        self.assertEqual(top[1]["sample_hashes"], ["0x0000000000003001"])
        self.assertEqual(top[2]["sample_hashes"], ["0x0000000000002001", "0x0000000000002002"])
        self.assertEqual(top[3]["sample_hashes"], ["0x0000000000002001"])

    def test_report_missing_sources_sanitizes_wgsl_and_host_paths(self):
        hostile_entries = {
            "input_path": r"C:\Users\secret_user\games\title\shaders.bin",
            "entries": [
                {
                    "hash_fnv1a64": "0x0000000000009001",
                    "stage": "fragment",
                    "wgsl": "@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }",
                    "source_path": "/home/secret_user/dumps/shader_9001.wgsl",
                    "file": r"D:\private\build\shader.dxil",
                    "bindings": [
                        {
                            "group": 0,
                            "binding": 0,
                            "kind": "uniform",
                            "name": r"C:\Users\secret_user\cb0",
                        },
                        {
                            "group": 0,
                            "binding": 1,
                            "kind": "texture",
                            "name": "@group(0) @binding(1) var tex0: texture_2d<f32>;",
                        },
                    ],
                },
                {
                    "hash_fnv1a64": "/home/secret_user/leak_hash",
                    "stage": "compute",
                    "bindings": [
                        {
                            "group": 0,
                            "binding": 0,
                            "kind": "/etc/passwd",
                            "name": "safe_name_0",
                        },
                    ],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "hostile.json"
            out = Path(tmp) / "out.reflect.json"
            report_path = Path(tmp) / "report.json"
            src.write_text(json.dumps(hostile_entries), encoding="utf-8")

            rc = sidecar.main([
                str(src),
                "--out", str(out),
                "--report-missing-sources", str(report_path),
            ])
            self.assertEqual(rc, 0)
            raw_report = report_path.read_text(encoding="utf-8")
            report = json.loads(raw_report)

        # Ensure zero WGSL source or host paths leak anywhere in the written report
        for forbidden in (
            "secret_user",
            "fs_main",
            "@fragment",
            "@group",
            "vec4<f32>",
            "/home/",
            "C:\\Users",
            "D:\\private",
            "/etc/passwd",
            "shader_9001",
            str(tmp),
        ):
            self.assertNotIn(forbidden, raw_report)

        self.assertEqual(report["total_skipped_missing_source"], 2)
        self.assertEqual(report["by_stage"], {"vertex": 0, "fragment": 1, "compute": 1})
        by_stage_kind = {(item["stage"], item["kind"], item["name"]): item for item in report["top_missing_bindings"]}
        self.assertIn(("fragment", "uniform", "<redacted>"), by_stage_kind)
        self.assertIn(("fragment", "texture", "<redacted>"), by_stage_kind)
        self.assertIn(("compute", "<redacted>", "safe_name_0"), by_stage_kind)
        self.assertEqual(by_stage_kind[("compute", "<redacted>", "safe_name_0")]["sample_hashes"], ["<redacted>"])

    def test_json_breakdown_telemetry_counts_all_categories(self):
        inline_src_a = {"inline_sampler": {"sdwords": [0x92, 0x00F00000, 0x08500000, 0]}}
        inline_src_b = {"inline_sampler": {"sdwords": [0x00, 0x00F00000, 0x00000000, 0]}}
        res_heap_a = {"resource_heap": {"base_user_sgpr": 4, "stride": 32, "index": {"user_sgpr": 5, "shift": 5}}}
        res_heap_b = {"resource_heap": {"base_user_sgpr": 6, "stride": 32, "index": {"user_sgpr": 7, "shift": 5}}}
        samp_heap = {"sampler_heap": {"base_user_sgpr": 11, "stride": 16, "index": {"user_sgpr": 9, "shift": 4}}}

        entries = [
            # Vertex stage with attributes (NOT vertex_index)
            vertex_entry(True),
            # Vertex stage using vertex_index + 1 resource_heap binding
            {
                "hash_fnv1a64": "0x00000000000000a2",
                "stage": "vertex",
                "binding_summary": {"vertex_input_mode": "vertex_index"},
                "bindings": [
                    {"group": 0, "binding": 0, "kind": "storage", "source": res_heap_a},
                ],
            },
            # Vertex stage with omitted binding_summary & no vertex_inputs (buffer-pulling -> vertex_index)
            {
                "hash_fnv1a64": "0x00000000000000a3",
                "stage": "vertex",
                "bindings": [
                    {"group": 0, "binding": 0, "kind": "uniform",
                     "source": {"user_sgpr": 2, "table_byte_offset": 0}},
                ],
            },
            # Fragment stage with 2 inline_samplers, 2 resource_heap bindings, and 1 sampler_heap
            # (sampler_heap must NOT be counted as resource_heap)
            {
                "hash_fnv1a64": "0x00000000000000b1",
                "stage": "pixel",
                "bindings": [
                    {"group": 0, "binding": 0, "kind": "texture", "source": res_heap_a},
                    {"group": 0, "binding": 1, "kind": "texture", "source": res_heap_b},
                    {"group": 0, "binding": 2, "kind": "sampler", "source": samp_heap},
                    {"group": 0, "binding": 3, "kind": "sampler", "source": inline_src_a},
                    {"group": 0, "binding": 4, "kind": "sampler", "source": inline_src_b},
                ],
            },
            # Compute stage with 1 inline_sampler
            {
                "hash_fnv1a64": "0x00000000000000c1",
                "stage": "compute",
                "bindings": [
                    {"group": 0, "binding": 0, "kind": "sampler", "source": inline_src_a},
                ],
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "pack.json"
            out = Path(tmp) / "pack.reflect.json"
            src.write_text(json.dumps({"entries": entries}), encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = sidecar.main([str(src), "--out", str(out), "--json"])
            self.assertEqual(rc, 0)
            summary = json.loads(stdout.getvalue())
            written = json.loads(out.read_text(encoding="utf-8"))

        # Sidecar JSON v1 schema must remain strictly unchanged
        self.assertEqual(set(written.keys()), {"schema", "version", "stages"})
        self.assertEqual(written["schema"], "playgta6-stage-reflection")
        self.assertEqual(written["version"], 1)

        self.assertEqual(summary, {
            "sidecar_version": 1,
            "stages_written": 5,
            "skipped": {},
            "by_stage": {"vertex": 3, "fragment": 1, "compute": 1},
            "inline_sampler_stages": 2,
            "inline_sampler_bindings": 3,
            "resource_heap_stages": 2,
            "resource_heap_bindings": 3,
            "cbv_indexed_heap_stages": 0,
            "cbv_indexed_heap_bindings": 0,
            "lane_indexed_heap_stages": 0,
            "lane_indexed_heap_bindings": 0,
            "root_descriptor_table_offset_stages": 0,
            "root_descriptor_table_offset_bindings": 0,
            "rt_arena_stages": 0,
            "rt_arena_bindings": 0,
            "vertex_index_stages": 2,
            "mesh_output_stages": 0,
            "unresolved_stages": 0,
            "unresolved_bindings": 0,
            "unresolved_by_kind": {},
            "unresolved_by_reason": {},
            "gap_breakdown": {
                "skipped_total": 0,
                "skipped_by_cause": [],
                "unresolved_stages": 0,
                "unresolved_bindings": 0,
                "unresolved_by_kind": {},
                "unresolved_by_reason": {},
            },
        })

    def test_summarize_stages_empty_and_explicit_vertex_input_mode(self):
        self.assertEqual(sidecar.summarize_stages({}), {
            "by_stage": {"vertex": 0, "fragment": 0, "compute": 0},
            "inline_sampler_stages": 0,
            "inline_sampler_bindings": 0,
            "resource_heap_stages": 0,
            "resource_heap_bindings": 0,
            "cbv_indexed_heap_stages": 0,
            "cbv_indexed_heap_bindings": 0,
            "lane_indexed_heap_stages": 0,
            "lane_indexed_heap_bindings": 0,
            "root_descriptor_table_offset_stages": 0,
            "root_descriptor_table_offset_bindings": 0,
            "rt_arena_stages": 0,
            "rt_arena_bindings": 0,
            "vertex_index_stages": 0,
            "mesh_output_stages": 0,
            "unresolved_stages": 0,
            "unresolved_bindings": 0,
            "unresolved_by_kind": {},
            "unresolved_by_reason": {},
        })
        stages = {
            "0x1": {"stage": "vertex", "bindings": [], "vertex_inputs": []},
            "0x2": {"stage": "vertex", "bindings": [], "vertex_input_mode": "vertex_index"},
            "0x3": {"stage": "vertex", "bindings": [], "vertex_input_mode": "attributes"},
        }
        summary = sidecar.summarize_stages(stages)
        self.assertEqual(summary["by_stage"], {"vertex": 3, "fragment": 0, "compute": 0})
        self.assertEqual(summary["vertex_index_stages"], 2)

    def test_diff_sidecars_unit_counts_key_order_and_sample_caps(self):
        # Identical stage with different dict key ordering must be unchanged
        baseline = {
            "0x0000000000000001": {
                "bindings": [{"source": {"static_sampler": 0}, "kind": "sampler", "binding": 0, "group": 0}],
                "stage": "fragment",
            },
            "0x0000000000000002": {
                "stage": "vertex",
                "bindings": [{"group": 0, "binding": 0, "kind": "uniform",
                              "source": {"user_sgpr": 2, "table_byte_offset": 0}}],
            },
            # 6 removed stages (4 fragment, 2 compute) to test cap at 5
            **{
                f"0x000000000000010{i}": {
                    "stage": "fragment" if i < 4 else "compute",
                    "bindings": [],
                }
                for i in range(6)
            },
        }
        current = {
            "0x0000000000000001": {
                "stage": "fragment",
                "bindings": [{"group": 0, "binding": 0, "kind": "sampler", "source": {"static_sampler": 0}}],
            },
            # Changed binding source
            "0x0000000000000002": {
                "stage": "vertex",
                "bindings": [{"group": 0, "binding": 0, "kind": "uniform",
                              "source": {"user_sgpr": 4, "table_byte_offset": 16}}],
            },
            # 2 added stages (1 vertex, 1 compute)
            "0x0000000000000201": {"stage": "vertex", "bindings": []},
            "0x0000000000000202": {"stage": "compute", "bindings": []},
        }

        diff = sidecar.diff_sidecars(baseline, current)
        self.assertEqual(diff, {
            "baseline_stages": 8,
            "current_stages": 4,
            "net_delta": -4,
            "added_count": 2,
            "removed_count": 6,
            "changed_count": 1,
            "unchanged_count": 1,
            "added_by_stage": {"vertex": 1, "fragment": 0, "compute": 1},
            "removed_by_stage": {"vertex": 0, "fragment": 4, "compute": 2},
            "added_sample_hashes": ["0x0000000000000201", "0x0000000000000202"],
            "removed_sample_hashes": [f"0x000000000000010{i}" for i in range(5)],
            "changed_sample_hashes": ["0x0000000000000002"],
        })

    def test_diff_baseline_cli_and_fail_on_regressions(self):
        baseline_doc = {
            "schema": "playgta6-stage-reflection",
            "version": 1,
            "stages": {
                # Will remain unchanged
                "0x00000000000000a1": {
                    "stage": "vertex",
                    "bindings": [{"group": 0, "binding": 0, "kind": "uniform",
                                  "source": {"user_sgpr": 8, "table_byte_offset": 0},
                                  "min_binding_size_bytes": 16}],
                    "vertex_inputs": [{"location": 0, "vb_slot": 0, "byte_offset": 8,
                                       "components": 2, "component_type": "f32"}],
                },
                # Will be changed
                "0x00000000000000b1": {
                    "stage": "fragment",
                    "bindings": [{"group": 0, "binding": 0, "kind": "sampler", "source": {"static_sampler": 0}}],
                },
                # Will be removed (regression)
                "0x00000000000000c1": {
                    "stage": "compute",
                    "bindings": [{"group": 0, "binding": 0, "kind": "storage",
                                  "source": {"user_sgpr": 2, "table_byte_offset": 0}}],
                },
            },
        }
        current_entries = {
            "entries": [
                # Unchanged
                vertex_entry(True),
                # Changed (static_sampler 0 -> 1)
                {
                    "hash_fnv1a64": "0x00000000000000b1",
                    "stage": "fragment",
                    "bindings": [{"group": 0, "binding": 0, "kind": "sampler", "source": {"static_sampler": 1}}],
                },
                # Added
                {
                    "hash_fnv1a64": "0x00000000000000d1",
                    "stage": "fragment",
                    "bindings": [{"group": 0, "binding": 0, "kind": "sampler",
                                  "source": {"inline_sampler": {"sdwords": [1, 2, 3, 4]}}}],
                },
                # Skipped (for missing-sources report check)
                {
                    "hash_fnv1a64": "0x00000000000000e1",
                    "stage": "compute",
                    "bindings": [{"group": 0, "binding": 0, "kind": "storage", "name": "buf0"}],
                },
            ]
        }

        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = Path(tmp) / "prev.reflect.json"
            src_path = Path(tmp) / "new.json"
            out_path = Path(tmp) / "curr.reflect.json"
            report_path = Path(tmp) / "missing.json"
            baseline_path.write_text(json.dumps(baseline_doc), encoding="utf-8")
            src_path.write_text(json.dumps(current_entries), encoding="utf-8")

            # Without --fail-on-regressions: exits 0 even though 1 stage was removed
            stdout_ok = io.StringIO()
            with redirect_stdout(stdout_ok):
                rc_ok = sidecar.main([
                    str(src_path),
                    "--out", str(out_path),
                    "--diff-baseline", str(baseline_path),
                    "--json",
                ])
            self.assertEqual(rc_ok, 0)
            summary_ok = json.loads(stdout_ok.getvalue())
            self.assertEqual(summary_ok["diff"], {
                "baseline_stages": 3,
                "current_stages": 3,
                "net_delta": 0,
                "added_count": 1,
                "removed_count": 1,
                "changed_count": 1,
                "unchanged_count": 1,
                "added_by_stage": {"vertex": 0, "fragment": 1, "compute": 0},
                "removed_by_stage": {"vertex": 0, "fragment": 0, "compute": 1},
                "added_sample_hashes": ["0x00000000000000d1"],
                "removed_sample_hashes": ["0x00000000000000c1"],
                "changed_sample_hashes": ["0x00000000000000b1"],
            })

            # With --fail-on-regressions: exits 2, still writes sidecar, report, and full JSON summary
            out_fail = Path(tmp) / "curr_fail.reflect.json"
            stdout_fail = io.StringIO()
            stderr_fail = io.StringIO()
            with redirect_stdout(stdout_fail), redirect_stderr(stderr_fail):
                rc_fail = sidecar.main([
                    str(src_path),
                    "--out", str(out_fail),
                    "--diff-baseline", str(baseline_path),
                    "--fail-on-regressions",
                    "--report-missing-sources", str(report_path),
                    "--json",
                ])
            self.assertEqual(rc_fail, 2)
            self.assertTrue(out_fail.exists())
            self.assertTrue(report_path.exists())
            self.assertIn("regression detected", stderr_fail.getvalue())
            self.assertEqual(json.loads(stdout_fail.getvalue()), summary_ok)
            written = json.loads(out_fail.read_text(encoding="utf-8"))
            self.assertEqual(set(written["stages"].keys()), {
                "0x00000000000000a1",
                "0x00000000000000b1",
                "0x00000000000000d1",
            })

            # Human-readable output includes concise diff summary
            stdout_text = io.StringIO()
            with redirect_stdout(stdout_text):
                rc_text = sidecar.main([
                    str(src_path),
                    "--out", str(out_path),
                    "--diff-baseline", str(baseline_path),
                ])
            self.assertEqual(rc_text, 0)
            self.assertIn("diff: +1/-1/~1/=1 (net +0)", stdout_text.getvalue())

    def test_fail_on_regressions_exits_zero_when_no_stages_removed_and_inplace_baseline(self):
        entry1 = vertex_entry(True)
        entry2 = {
            "hash_fnv1a64": "0x00000000000000b2",
            "stage": "fragment",
            "bindings": [{"group": 0, "binding": 0, "kind": "sampler", "source": {"static_sampler": 2}}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            in1 = Path(tmp) / "in1.json"
            in2 = Path(tmp) / "in2.json"
            out = Path(tmp) / "pack.reflect.json"
            in1.write_text(json.dumps(entry1), encoding="utf-8")
            in2.write_text(json.dumps({"entries": [entry1, entry2]}), encoding="utf-8")

            # Seed initial sidecar
            self.assertEqual(sidecar.main([str(in1), "--out", str(out)]), 0)

            # In-place overwrite with --diff-baseline pointing to --out itself + --fail-on-regressions
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = sidecar.main([
                    str(in2),
                    "--out", str(out),
                    "--diff-baseline", str(out),
                    "--fail-on-regressions",
                    "--json",
                ])
            self.assertEqual(rc, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["diff"]["baseline_stages"], 1)
            self.assertEqual(summary["diff"]["current_stages"], 2)
            self.assertEqual(summary["diff"]["net_delta"], 1)
            self.assertEqual(summary["diff"]["added_count"], 1)
            self.assertEqual(summary["diff"]["removed_count"], 0)
            self.assertEqual(summary["diff"]["unchanged_count"], 1)

    def test_diff_baseline_errors_and_fail_on_regressions_without_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.json"
            src.write_text(json.dumps(vertex_entry(True)), encoding="utf-8")

            # --fail-on-regressions without --diff-baseline fails cleanly with code 1
            out_no_base = Path(tmp) / "no_base.reflect.json"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                rc = sidecar.main([str(src), "--out", str(out_no_base), "--fail-on-regressions", "--json"])
            self.assertEqual(rc, 1)
            self.assertIn("--fail-on-regressions requires --diff-baseline", stderr.getvalue())
            self.assertFalse(out_no_base.exists())

            # Missing --diff-baseline file fails cleanly with code 1
            missing = Path(tmp) / "does_not_exist.reflect.json"
            out_missing = Path(tmp) / "out_missing.reflect.json"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                rc = sidecar.main([str(src), "--out", str(out_missing), "--diff-baseline", str(missing), "--json"])
            self.assertEqual(rc, 1)
            self.assertIn("error: failed to load --diff-baseline", stderr.getvalue())
            self.assertFalse(out_missing.exists())

            # Invalid schema in --diff-baseline fails cleanly with code 1
            bad_base = Path(tmp) / "bad.reflect.json"
            out_bad = Path(tmp) / "out_bad.reflect.json"
            bad_base.write_text(json.dumps({"schema": "wrong", "version": 1, "stages": {}}), encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                rc = sidecar.main([str(src), "--out", str(out_bad), "--diff-baseline", str(bad_base), "--json"])
            self.assertEqual(rc, 1)
            self.assertIn("error: failed to load --diff-baseline", stderr.getvalue())
            self.assertFalse(out_bad.exists())

    def test_json_summary_and_diff_sanitize_hostile_keys_and_paths(self):
        baseline_doc = {
            "schema": "playgta6-stage-reflection",
            "version": 1,
            "stages": {
                r"C:\Users\secret_user\games\removed.wgsl": {
                    "stage": "fragment",
                    "bindings": [],
                },
                "/home/secret_user/changed.wgsl": {
                    "stage": "compute",
                    "bindings": [{"group": 0, "binding": 0, "kind": "sampler", "source": {"static_sampler": 0}}],
                },
            },
        }
        current_entries = {
            "input_path": r"C:\Users\secret_user\games\shaders.bin",
            "entries": [
                {
                    "hash_fnv1a64": "/home/secret_user/changed.wgsl",
                    "stage": "compute",
                    "wgsl": "@compute fn cs_main() {}",
                    "bindings": [{"group": 0, "binding": 0, "kind": "sampler", "source": {"static_sampler": 1}}],
                },
                {
                    "hash_fnv1a64": "/home/secret_user/added.wgsl",
                    "stage": "vertex",
                    "wgsl": "@vertex fn vs_main() {}",
                    "bindings": [],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = Path(tmp) / "base.reflect.json"
            src_path = Path(tmp) / "in.json"
            out_path = Path(tmp) / "out.reflect.json"
            baseline_path.write_text(json.dumps(baseline_doc), encoding="utf-8")
            src_path.write_text(json.dumps(current_entries), encoding="utf-8")

            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = sidecar.main([
                    str(src_path),
                    "--out", str(out_path),
                    "--diff-baseline", str(baseline_path),
                    "--fail-on-regressions",
                    "--json",
                ])
            self.assertEqual(rc, 2)
            raw_out = stdout.getvalue()
            raw_err = stderr.getvalue()

        for forbidden in ("secret_user", "cs_main", "vs_main", "@compute", "@vertex", "/home/", "C:\\Users", str(tmp)):
            self.assertNotIn(forbidden, raw_out)
            self.assertNotIn(forbidden, raw_err)

        summary = json.loads(raw_out)
        self.assertEqual(summary["diff"]["added_sample_hashes"], ["<redacted>"])
        self.assertEqual(summary["diff"]["removed_sample_hashes"], ["<redacted>"])
        self.assertEqual(summary["diff"]["changed_sample_hashes"], ["<redacted>"])

    def test_verify_pack_100_percent_match_and_cli_output(self):
        pack_bytes = build_synthetic_wgpk([
            (0x00000000000000A1, "vertex", b"@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }"),
            (0x00000000000000B2, "fragment", b"@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }"),
            (0x00000000000000C3, "compute", b"@compute @workgroup_size(1) fn cs() {}"),
        ])
        entries = {
            "entries": [
                # Uppercase hex hash verifies canonical hash matching against WGPK lowercase hex
                {**vertex_entry(True), "hash_fnv1a64": "0x00000000000000A1"},
                {
                    "hash_fnv1a64": "0x00000000000000b2",
                    "stage": "fragment",
                    "bindings": [{"group": 0, "binding": 0, "kind": "sampler", "source": {"static_sampler": 0}}],
                },
                {
                    "hash_fnv1a64": "0x00000000000000c3",
                    "stage": "compute",
                    "bindings": [{"group": 0, "binding": 0, "kind": "storage",
                                  "source": {"user_sgpr": 4, "table_byte_offset": 0}}],
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.json"
            out = Path(tmp) / "shaders.wgsl.pack.reflect.json"
            pack_path = Path(tmp) / "shaders.wgsl.pack"
            src.write_text(json.dumps(entries), encoding="utf-8")
            pack_path.write_bytes(pack_bytes)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = sidecar.main([
                    str(src),
                    "--out", str(out),
                    "--verify-pack", str(pack_path),
                    "--json",
                ])
            self.assertEqual(rc, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["pack_verification"], {
                "pack_label": "shaders.wgsl.pack",
                "pack_version": 1,
                "pack_bytes": len(pack_bytes),
                "toc_entry_bytes": sidecar.WGPK_ENTRY_BYTES,
                "is_wgpk_v2": False,
                "pack_entry_count": 3,
                "reflection_stage_count": 3,
                "matched_hashes": 3,
                "missing_reflection_hashes": {"count": 0, "samples": []},
                "extra_reflection_hashes": {"count": 0, "samples": []},
                "coverage_pct": 100.0,
                "pack_by_stage": {"vertex": 1, "fragment": 1, "compute": 1},
                "missing_by_stage": {"vertex": 0, "fragment": 0, "compute": 0},
                "stage_mismatches": 0,
                "stage_mismatch_samples": [],
                "gap_counts": {
                    "missing_reflection": 0,
                    "stage_mismatches": 0,
                    "extra_reflection": 0,
                    "unresolved_matched_stages": 0,
                },
            })

            # Human-readable output includes concise pack verification summary
            stdout_text = io.StringIO()
            with redirect_stdout(stdout_text):
                rc_text = sidecar.main([
                    str(src),
                    "--out", str(out),
                    "--verify-pack", str(pack_path),
                ])
            self.assertEqual(rc_text, 0)
            self.assertIn(
                "pack verify (shaders.wgsl.pack): 3/3 matched (100.00%), 0 missing, 0 extra",
                stdout_text.getvalue(),
            )

    def test_verify_pack_partial_match_bounded_samples_extra_and_stage_mismatch(self):
        # 8 pack entries: 2 matched (1 with stage mismatch), 6 missing (tests 5-sample cap)
        pack_entries = [
            (0x0000000000000001, "vertex", b"// vs1"),
            (0x0000000000000002, "vertex", b"// vs2 - mismatched in sidecar"),
        ] + [
            (0x0000000000001000 + i, "fragment" if i < 4 else "compute", f"// miss {i}".encode("utf-8"))
            for i in range(6)
        ]
        pack_bytes = build_synthetic_wgpk(pack_entries)

        # 9 reflection stages: 2 matched + 7 extra (tests 5-sample cap on extra_reflection_hashes too)
        stages = {
            "0x0000000000000001": {"stage": "vertex", "bindings": []},
            "0x0000000000000002": {"stage": "fragment", "bindings": []},
            **{
                f"0x000000000000200{i}": {"stage": "compute", "bindings": []}
                for i in range(7)
            },
        }
        toc = sidecar.parse_wgpk_toc_bytes(pack_bytes)
        result = sidecar.verify_sidecar_against_pack(toc, stages, pack_label="partial.wgsl.pack")
        self.assertEqual(result, {
            "pack_label": "partial.wgsl.pack",
            "pack_version": 1,
            "pack_bytes": len(pack_bytes),
            "toc_entry_bytes": sidecar.WGPK_ENTRY_BYTES,
            "is_wgpk_v2": False,
            "pack_entry_count": 8,
            "reflection_stage_count": 9,
            "matched_hashes": 2,
            "missing_reflection_hashes": {
                "count": 6,
                "samples": [f"0x000000000000100{i}" for i in range(5)],
            },
            "extra_reflection_hashes": {
                "count": 7,
                "samples": [f"0x000000000000200{i}" for i in range(5)],
            },
            "coverage_pct": 25.0,
            "pack_by_stage": {"vertex": 2, "fragment": 4, "compute": 2},
            "missing_by_stage": {"vertex": 0, "fragment": 4, "compute": 2},
            "stage_mismatches": 1,
            "stage_mismatch_samples": ["0x0000000000000002"],
            "gap_counts": {
                "missing_reflection": 6,
                "stage_mismatches": 1,
                "extra_reflection": 7,
                "unresolved_matched_stages": 0,
            },
        })

    def test_verify_pack_empty_pack_zero_division_guard(self):
        empty_pack = build_synthetic_wgpk([])
        toc = sidecar.parse_wgpk_toc_bytes(empty_pack)
        self.assertEqual(toc, [])

        # Empty pack + empty sidecar -> 0.0% coverage (no ZeroDivisionError)
        res_empty = sidecar.verify_sidecar_against_pack(toc, {}, pack_label="empty.wgsl.pack")
        self.assertEqual(res_empty["pack_entry_count"], 0)
        self.assertEqual(res_empty["reflection_stage_count"], 0)
        self.assertEqual(res_empty["matched_hashes"], 0)
        self.assertEqual(res_empty["missing_reflection_hashes"], {"count": 0, "samples": []})
        self.assertEqual(res_empty["extra_reflection_hashes"], {"count": 0, "samples": []})
        self.assertEqual(res_empty["coverage_pct"], 0.0)

        # Empty pack + non-empty sidecar via CLI -> 0.0% coverage and extra hash reported
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.json"
            out = Path(tmp) / "out.reflect.json"
            pack_path = Path(tmp) / "empty.wgsl.pack"
            src.write_text(json.dumps(vertex_entry(True)), encoding="utf-8")
            pack_path.write_bytes(empty_pack)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = sidecar.main([str(src), "--out", str(out), "--verify-pack", str(pack_path), "--json"])
            self.assertEqual(rc, 0)
            verify = json.loads(stdout.getvalue())["pack_verification"]
        self.assertEqual(verify["pack_entry_count"], 0)
        self.assertEqual(verify["reflection_stage_count"], 1)
        self.assertEqual(verify["matched_hashes"], 0)
        self.assertEqual(verify["extra_reflection_hashes"], {"count": 1, "samples": ["0x00000000000000a1"]})
        self.assertEqual(verify["coverage_pct"], 0.0)

    def test_verify_pack_malformed_bounds_and_header_rejection(self):
        bad_packs: list[tuple[str, bytes]] = [
            ("truncated_header", b"WGPK\x01\x00"),
            ("bad_magic", sidecar.WGPK_HEADER_STRUCT.pack(b"NOPE", 1, 0, 0)),
            ("unsupported_version", sidecar.WGPK_HEADER_STRUCT.pack(b"WGPK", 3, 0, 0)),
            ("truncated_toc", sidecar.WGPK_HEADER_STRUCT.pack(b"WGPK", 1, 2, 0)),
            ("toc_u32_overflow", sidecar.WGPK_HEADER_STRUCT.pack(b"WGPK", 1, 0x10000000, 0)),
            (
                "invalid_stage",
                sidecar.WGPK_HEADER_STRUCT.pack(b"WGPK", 1, 1, 0)
                + sidecar.WGPK_ENTRY_STRUCT.pack(0x1, 4, b"\x00\x00\x00", 36, 4)
                + b"abcd",
            ),
            (
                "offset_overlaps_toc",
                sidecar.WGPK_HEADER_STRUCT.pack(b"WGPK", 1, 1, 0)
                + sidecar.WGPK_ENTRY_STRUCT.pack(0x1, 1, b"\x00\x00\x00", 12, 4)
                + b"abcd",
            ),
            (
                "payload_exceeds_eof",
                sidecar.WGPK_HEADER_STRUCT.pack(b"WGPK", 1, 1, 0)
                + sidecar.WGPK_ENTRY_STRUCT.pack(0x1, 1, b"\x00\x00\x00", 36, 16)
                + b"abcd",
            ),
            (
                "payload_range_u32_overflow",
                sidecar.WGPK_HEADER_STRUCT.pack(b"WGPK", 1, 1, 0)
                + sidecar.WGPK_ENTRY_STRUCT.pack(0x1, 1, b"\x00\x00\x00", 0xFFFFFFF0, 32)
                + b"abcd",
            ),
            (
                "overlapping_payloads",
                sidecar.WGPK_HEADER_STRUCT.pack(b"WGPK", 1, 2, 0)
                + sidecar.WGPK_ENTRY_STRUCT.pack(0x1, 1, b"\x00\x00\x00", 56, 6)
                + sidecar.WGPK_ENTRY_STRUCT.pack(0x2, 2, b"\x00\x00\x00", 60, 4)
                + b"0123456789",
            ),
            (
                "duplicate_hash",
                sidecar.WGPK_HEADER_STRUCT.pack(b"WGPK", 1, 2, 0)
                + sidecar.WGPK_ENTRY_STRUCT.pack(0x99, 1, b"\x00\x00\x00", 56, 2)
                + sidecar.WGPK_ENTRY_STRUCT.pack(0x99, 2, b"\x00\x00\x00", 58, 2)
                + b"abcd",
            ),
        ]

        for name, raw in bad_packs:
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    sidecar.parse_wgpk_toc_bytes(raw)

        # CLI must fail cleanly with exit code 1 and not write --out on malformed/missing/directory pack
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.json"
            src.write_text(json.dumps(vertex_entry(True)), encoding="utf-8")

            for name, raw in bad_packs:
                pack_path = Path(tmp) / f"{name}.wgsl.pack"
                out_path = Path(tmp) / f"out_{name}.reflect.json"
                pack_path.write_bytes(raw)
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    rc = sidecar.main([str(src), "--out", str(out_path), "--verify-pack", str(pack_path), "--json"])
                self.assertEqual(rc, 1, name)
                self.assertIn("error: failed to load --verify-pack:", stderr.getvalue(), name)
                self.assertFalse(out_path.exists(), name)

            # Missing pack file and directory path
            for label, bad_target in (
                ("missing", Path(tmp) / "secret_user_dir" / "absent.wgsl.pack"),
                ("directory", Path(tmp)),
            ):
                out_path = Path(tmp) / f"out_{label}.reflect.json"
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    rc = sidecar.main([str(src), "--out", str(out_path), "--verify-pack", str(bad_target), "--json"])
                self.assertEqual(rc, 1, label)
                self.assertIn("error: failed to load --verify-pack:", stderr.getvalue(), label)
                self.assertNotIn("secret_user_dir", stderr.getvalue(), label)
                self.assertNotIn(str(tmp), stderr.getvalue(), label)
                self.assertFalse(out_path.exists(), label)

    def test_verify_pack_json_sanitizes_host_paths_and_usernames(self):
        pack_bytes = build_synthetic_wgpk([
            (0x00000000000000A1, "vertex", b"// vs"),
            (0x00000000000000F9, "fragment", b"// fs"),
        ])
        hostile_entries = {
            "input_path": r"C:\Users\secret_user\games\shaders.bin",
            "entries": [
                vertex_entry(True),
                {
                    "hash_fnv1a64": "/home/secret_user/extra_shader.wgsl",
                    "stage": "compute",
                    "wgsl": "@compute fn cs_secret() {}",
                    "bindings": [{"group": 0, "binding": 0, "kind": "sampler", "source": {"static_sampler": 0}}],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            secret_dir = Path(tmp) / "Users" / "secret_user" / "private_build"
            secret_dir.mkdir(parents=True)
            src = secret_dir / "in.json"
            out = secret_dir / "out.reflect.json"
            pack_path = secret_dir / "title_shaders.wgsl.pack"
            src.write_text(json.dumps(hostile_entries), encoding="utf-8")
            pack_path.write_bytes(pack_bytes)

            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = sidecar.main([
                    str(src),
                    "--out", str(out),
                    "--verify-pack", str(pack_path),
                    "--json",
                ])
            self.assertEqual(rc, 0)
            raw_out = stdout.getvalue()
            raw_err = stderr.getvalue()

        for forbidden in (
            "secret_user",
            "private_build",
            "cs_secret",
            "@compute",
            "/home/",
            "C:\\Users",
            str(tmp),
            str(secret_dir),
        ):
            self.assertNotIn(forbidden, raw_out)
            self.assertNotIn(forbidden, raw_err)

        summary = json.loads(raw_out)
        verify = summary["pack_verification"]
        self.assertEqual(verify["pack_label"], "title_shaders.wgsl.pack")
        self.assertEqual(verify["matched_hashes"], 1)
        self.assertEqual(verify["missing_reflection_hashes"], {"count": 1, "samples": ["0x00000000000000f9"]})
        self.assertEqual(verify["extra_reflection_hashes"], {"count": 1, "samples": ["<redacted>"]})
        self.assertEqual(verify["coverage_pct"], 50.0)

        # Path label sanitizer unit edge cases
        self.assertEqual(sidecar._sanitize_path_label(r"C:\Users\alice\work\game.wgsl.pack"), "game.wgsl.pack")
        self.assertEqual(sidecar._sanitize_path_label("/home/alice/work/game.wgsl.pack"), "game.wgsl.pack")
        self.assertEqual(sidecar._sanitize_path_label(r"C:\Users\alice\my secret pack.pack"), "<redacted>")
        self.assertEqual(sidecar._sanitize_path_label("C:\\"), "<unknown>")
        self.assertEqual(sidecar._sanitize_path_label(".."), "<unknown>")

    def test_verify_pack_standalone_existing_sidecar_and_omitted_without_flag(self):
        pack_bytes = build_synthetic_wgpk([
            (0x00000000000000A1, "vertex", b"// vs"),
            (0x00000000000000B2, "fragment", b"// fs"),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.json"
            out = Path(tmp) / "existing.reflect.json"
            pack_path = Path(tmp) / "pack.wgsl.pack"
            src.write_text(json.dumps(vertex_entry(True)), encoding="utf-8")
            pack_path.write_bytes(pack_bytes)

            # Without --verify-pack: pack_verification key must be absent
            stdout_plain = io.StringIO()
            with redirect_stdout(stdout_plain):
                rc_plain = sidecar.main([str(src), "--out", str(out), "--json"])
            self.assertEqual(rc_plain, 0)
            self.assertNotIn("pack_verification", json.loads(stdout_plain.getvalue()))
            original_content = out.read_text(encoding="utf-8")

            # Standalone verify mode (no positional inputs): reads --out without modifying it
            stdout_verify = io.StringIO()
            with redirect_stdout(stdout_verify):
                rc_verify = sidecar.main(["--out", str(out), "--verify-pack", str(pack_path), "--json"])
            self.assertEqual(rc_verify, 0)
            self.assertEqual(out.read_text(encoding="utf-8"), original_content)
            summary = json.loads(stdout_verify.getvalue())
            self.assertEqual(summary["stages_written"], 0)
            self.assertEqual(summary["pack_verification"]["pack_entry_count"], 2)
            self.assertEqual(summary["pack_verification"]["reflection_stage_count"], 1)
            self.assertEqual(summary["pack_verification"]["matched_hashes"], 1)
            self.assertEqual(summary["pack_verification"]["coverage_pct"], 50.0)

    def test_v2_root_descriptor_table_byte_offset_accepted_and_upgrades_version(self):
        # G62 (#1471 / #482): v2 root_descriptor sources may include a 32-byte-aligned table_byte_offset.
        v2_entry = {
            "hash_fnv1a64": "0x00000000000000e1",
            "stage": "fragment",
            "bindings": [
                {
                    "group": 0,
                    "binding": 32,
                    "kind": "uniform",
                    "source": {"user_sgpr": 8, "root_descriptor": True, "table_byte_offset": 64},
                },
                {
                    "group": 0,
                    "binding": 33,
                    "kind": "uniform",
                    "source": {"user_sgpr": 10, "root_descriptor": True},
                },
                {
                    "group": 0,
                    "binding": 34,
                    "kind": "uniform",
                    "source": {"user_sgpr": 12, "root_descriptor": True, "table_byte_offset": 0},
                },
            ],
        }
        doc, skipped = sidecar.build([v2_entry])
        self.assertEqual(skipped, {})
        self.assertEqual(doc["schema"], "playgta6-stage-reflection")
        self.assertEqual(doc["version"], 2)
        bindings = doc["stages"]["0x00000000000000e1"]["bindings"]
        self.assertEqual(bindings[0]["source"], {"user_sgpr": 8, "root_descriptor": True, "table_byte_offset": 64})
        self.assertEqual(bindings[1]["source"], {"user_sgpr": 10, "root_descriptor": True})
        self.assertEqual(bindings[2]["source"], {"user_sgpr": 12, "root_descriptor": True, "table_byte_offset": 0})

        # Pure v1 root_descriptor (without table_byte_offset) stays version 1 for backward compatibility.
        v1_only = {
            "hash_fnv1a64": "0x00000000000000e2",
            "stage": "compute",
            "bindings": [
                {
                    "group": 0,
                    "binding": 0,
                    "kind": "uniform",
                    "source": {"user_sgpr": 4, "root_descriptor": True},
                }
            ],
        }
        doc_v1, skipped_v1 = sidecar.build([v1_only])
        self.assertEqual(skipped_v1, {})
        self.assertEqual(doc_v1["version"], 1)

    def test_v2_root_descriptor_non_multiple_of_32_offset_is_rejected(self):
        bad_sources = [
            {"user_sgpr": 8, "root_descriptor": True, "table_byte_offset": 4},
            {"user_sgpr": 8, "root_descriptor": True, "table_byte_offset": 16},
            {"user_sgpr": 8, "root_descriptor": True, "table_byte_offset": 36},
            {"user_sgpr": 8, "root_descriptor": True, "table_byte_offset": -32},
            {"user_sgpr": 8, "root_descriptor": True, "table_byte_offset": "32"},
            {"user_sgpr": 8, "root_descriptor": True, "table_byte_offset": True},
            {"root_descriptor": True, "table_byte_offset": 32},
        ]
        for idx, source in enumerate(bad_sources):
            entry = {
                "hash_fnv1a64": f"0x{idx + 128:016x}",
                "stage": "fragment",
                "bindings": [{"group": 0, "binding": 0, "kind": "uniform", "source": source}],
            }
            doc, skipped = sidecar.build([entry])
            self.assertEqual(doc["stages"], {}, source)
            self.assertEqual(skipped, {"malformed root_descriptor source": 1}, source)

    def test_v2_sidecar_merge_diff_and_standalone_verify_roundtrip(self):
        v2_doc = {
            "schema": "playgta6-stage-reflection",
            "version": 2,
            "stages": {
                "0x00000000000000e1": {
                    "stage": "fragment",
                    "bindings": [
                        {
                            "group": 0,
                            "binding": 32,
                            "kind": "uniform",
                            "source": {"user_sgpr": 8, "root_descriptor": True, "table_byte_offset": 32},
                        }
                    ],
                }
            },
        }
        pack_bytes = build_synthetic_wgpk([
            (0x00000000000000A1, "vertex", b"// vs"),
            (0x00000000000000E1, "fragment", b"// fs"),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            existing_v2 = Path(tmp) / "existing_v2.reflect.json"
            src_v1 = Path(tmp) / "in_v1.json"
            out_merged = Path(tmp) / "merged.reflect.json"
            pack_path = Path(tmp) / "test.wgsl.pack"
            existing_v2.write_text(json.dumps(v2_doc), encoding="utf-8")
            src_v1.write_text(json.dumps(vertex_entry(True)), encoding="utf-8")
            pack_path.write_bytes(pack_bytes)

            # Merging a v1 entry onto an existing v2 sidecar preserves v2 and bumps output version to 2
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = sidecar.main([
                    str(src_v1),
                    "--out", str(out_merged),
                    "--merge-existing", str(existing_v2),
                    "--diff-baseline", str(existing_v2),
                    "--verify-pack", str(pack_path),
                    "--json",
                ])
            self.assertEqual(rc, 0)
            written = json.loads(out_merged.read_text(encoding="utf-8"))
            self.assertEqual(written["version"], 2)
            self.assertEqual(
                written["stages"]["0x00000000000000e1"]["bindings"][0]["source"],
                {"user_sgpr": 8, "root_descriptor": True, "table_byte_offset": 32},
            )
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["pack_verification"]["matched_hashes"], 2)
            self.assertEqual(summary["diff"]["added_count"], 1)

            # Standalone --verify-pack accepts v2 sidecar directly
            stdout2 = io.StringIO()
            with redirect_stdout(stdout2):
                rc2 = sidecar.main(["--out", str(out_merged), "--verify-pack", str(pack_path), "--json"])
            self.assertEqual(rc2, 0)
            self.assertEqual(json.loads(stdout2.getvalue())["pack_verification"]["coverage_pct"], 100.0)

            # Loading a v2 sidecar with a misaligned root_descriptor table_byte_offset fails cleanly
            bad_v2 = Path(tmp) / "bad_v2.reflect.json"
            bad_v2.write_text(json.dumps({
                "schema": "playgta6-stage-reflection",
                "version": 2,
                "stages": {
                    "0x00000000000000e9": {
                        "stage": "fragment",
                        "bindings": [{
                            "group": 0,
                            "binding": 0,
                            "kind": "uniform",
                            "source": {"user_sgpr": 8, "root_descriptor": True, "table_byte_offset": 12},
                        }],
                    }
                },
            }), encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                rc_bad = sidecar.main(["--out", str(bad_v2), "--verify-pack", str(pack_path), "--json"])
            self.assertEqual(rc_bad, 1)
            self.assertIn("malformed v2 root_descriptor source", stderr.getvalue())

    def test_mesh_output_valid_compute_stage_preserved_and_counted(self):
        # G66 / S62 (#482, #1537): compute stage with valid mesh_output bounds (1..256, 4 + 4*n, n <= 15).
        cases = [
            ("0x4d53435300000001", {"max_vertices": 1, "max_primitives": 1, "vertex_words": 4}),
            ("0x4d53435300000002", {"max_vertices": 64, "max_primitives": 128, "vertex_words": 8}),
            ("0x4d53435300000003", {"max_vertices": 256, "max_primitives": 256, "vertex_words": 64}),
        ]
        entries = [
            {
                "hash_fnv1a64": key,
                "stage": "compute",
                "bindings": [
                    {
                        "group": 0,
                        "binding": 0,
                        "kind": "uniform",
                        "source": {"user_sgpr": 8, "table_byte_offset": 0},
                    }
                ],
                "mesh_output": mesh_out,
            }
            for key, mesh_out in cases
        ]
        # Regular compute stage without mesh_output must not be counted as mesh_output_stages.
        entries.append({
            "hash_fnv1a64": "0x00000000000000c9",
            "stage": "compute",
            "bindings": [],
        })

        doc, skipped = sidecar.build([{"entries": entries}])
        self.assertEqual(skipped, {})
        self.assertEqual(doc["version"], 1)
        for key, mesh_out in cases:
            self.assertEqual(doc["stages"][key]["stage"], "compute")
            self.assertEqual(doc["stages"][key]["mesh_output"], mesh_out)
        self.assertNotIn("mesh_output", doc["stages"]["0x00000000000000c9"])

        summary = sidecar.summarize_stages(doc["stages"])
        self.assertEqual(summary["by_stage"], {"vertex": 0, "fragment": 0, "compute": 4})
        self.assertEqual(summary["mesh_output_stages"], 3)

    def test_mesh_output_invalid_bounds_and_non_compute_rejected(self):
        bad_mesh_outputs = [
            # max_vertices out of 1..256
            {"max_vertices": 0, "max_primitives": 1, "vertex_words": 4},
            {"max_vertices": 257, "max_primitives": 1, "vertex_words": 4},
            {"max_vertices": -1, "max_primitives": 1, "vertex_words": 4},
            # max_primitives out of 1..256
            {"max_vertices": 1, "max_primitives": 0, "vertex_words": 4},
            {"max_vertices": 1, "max_primitives": 257, "vertex_words": 4},
            # vertex_words not 4 + 4*n with 0 <= n <= 15
            {"max_vertices": 3, "max_primitives": 1, "vertex_words": 0},
            {"max_vertices": 3, "max_primitives": 1, "vertex_words": 2},
            {"max_vertices": 3, "max_primitives": 1, "vertex_words": 6},
            {"max_vertices": 3, "max_primitives": 1, "vertex_words": 68},
            # Wrong types / missing keys
            {"max_vertices": True, "max_primitives": 1, "vertex_words": 4},
            {"max_vertices": "3", "max_primitives": 1, "vertex_words": 4},
            {"max_vertices": 3, "max_primitives": 1},
            "not-an-object",
            [3, 1, 4],
        ]
        for idx, bad_mo in enumerate(bad_mesh_outputs):
            entry = {
                "hash_fnv1a64": f"0x4d534353000001{idx:02x}",
                "stage": "compute",
                "bindings": [],
                "mesh_output": bad_mo,
            }
            doc, skipped = sidecar.build([entry])
            self.assertEqual(doc["stages"], {}, bad_mo)
            self.assertEqual(skipped, {"malformed mesh_output": 1}, bad_mo)

        # Valid mesh_output on non-compute stages (vertex / fragment) must be rejected.
        for bad_stage in ("vertex", "fragment", "pixel"):
            entry = {
                "hash_fnv1a64": "0x00000000000000f1",
                "stage": bad_stage,
                "bindings": [],
                "mesh_output": {"max_vertices": 3, "max_primitives": 1, "vertex_words": 8},
            }
            doc, skipped = sidecar.build([entry])
            self.assertEqual(doc["stages"], {}, bad_stage)
            self.assertEqual(skipped, {"malformed mesh_output": 1}, bad_stage)

    def test_mesh_output_roundtrip_with_verify_pack_and_salted_key(self):
        ms_raw_key = 0x0123456789ABCDEF
        ms_pack_key = ms_raw_key ^ 0x4D53435300000000
        ms_pack_hex = f"0x{ms_pack_key:016x}"
        ps_pack_key = 0x00000000000000B2
        ps_pack_hex = f"0x{ps_pack_key:016x}"

        pack_bytes = build_synthetic_wgpk([
            (ps_pack_key, "fragment", b"// fs"),
            (ms_pack_key, "compute", b"// mesh compute"),
        ])
        translator_json = {
            "entries": [
                {
                    "hash_fnv1a64": ms_pack_hex,
                    "stage": "compute",
                    "bindings": [
                        {
                            "group": 0,
                            "binding": 0,
                            "kind": "uniform",
                            "source": {"user_sgpr": 8, "table_byte_offset": 0},
                        }
                    ],
                    "mesh_output": {"max_vertices": 3, "max_primitives": 1, "vertex_words": 8},
                },
                {
                    "hash_fnv1a64": ps_pack_hex,
                    "stage": "fragment",
                    "bindings": [
                        {"group": 0, "binding": 32, "kind": "sampler", "source": {"static_sampler": 0}}
                    ],
                },
            ]
        }

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "pack_summary.json"
            out = Path(tmp) / "bundle.wgsl.pack.reflect.json"
            pack_path = Path(tmp) / "bundle.wgsl.pack"
            src.write_text(json.dumps(translator_json), encoding="utf-8")
            pack_path.write_bytes(pack_bytes)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = sidecar.main([str(src), "--out", str(out), "--verify-pack", str(pack_path), "--json"])
            self.assertEqual(rc, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["stages_written"], 2)
            self.assertEqual(summary["mesh_output_stages"], 1)
            self.assertEqual(summary["pack_verification"]["matched_hashes"], 2)
            self.assertEqual(summary["pack_verification"]["stage_mismatches"], 0)
            self.assertEqual(summary["pack_verification"]["coverage_pct"], 100.0)

            written = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(
                written["stages"][ms_pack_hex]["mesh_output"],
                {"max_vertices": 3, "max_primitives": 1, "vertex_words": 8},
            )

            # Standalone --verify-pack reloads the sidecar and validates mesh_output cleanly
            stdout2 = io.StringIO()
            with redirect_stdout(stdout2):
                rc2 = sidecar.main(["--out", str(out), "--verify-pack", str(pack_path), "--json"])
            self.assertEqual(rc2, 0)
            summary2 = json.loads(stdout2.getvalue())
            self.assertEqual(summary2["mesh_output_stages"], 1)
            self.assertEqual(summary2["pack_verification"]["coverage_pct"], 100.0)

            # Corrupt mesh_output in an existing sidecar fails cleanly on load
            bad_sidecar = Path(tmp) / "bad_mesh.reflect.json"
            bad_sidecar.write_text(json.dumps({
                "schema": "playgta6-stage-reflection",
                "version": 1,
                "stages": {
                    ms_pack_hex: {
                        "stage": "compute",
                        "bindings": [],
                        "mesh_output": {"max_vertices": 300, "max_primitives": 1, "vertex_words": 8},
                    }
                },
            }), encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                rc_bad = sidecar.main(["--out", str(bad_sidecar), "--verify-pack", str(pack_path), "--json"])
            self.assertEqual(rc_bad, 1)
            self.assertIn("malformed mesh_output", stderr.getvalue())

    @staticmethod
    def _sourceless_fragment(key: str, kinds: list[str]) -> dict:
        bindings = [{"group": 0, "binding": 0, "kind": "uniform", "name": "cb0",
                     "source": {"user_sgpr": 8, "table_byte_offset": 288}}]
        for i, kind in enumerate(kinds, start=1):
            bindings.append({"group": 0, "binding": i, "kind": kind, "name": f"r{i}", "view_dimension": "2d"})
        return {"hash_fnv1a64": key, "stage": "fragment", "bindings": bindings,
                "binding_summary": {"vertex_input_mode": "none"}}

    def test_unresolved_sources_keep_read_only_sourceless_stages(self):
        # G62 / #482: with unresolved sources on, read-only bindings without a source become
        # {"unresolved": true} (version 3); writable ones still skip the stage.
        tex = self._sourceless_fragment("0x00000000000000f1", ["texture_2d", "sampler"])
        uav = self._sourceless_fragment("0x00000000000000f2", ["storage-texture"])
        rw = self._sourceless_fragment("0x00000000000000f3", ["storage"])
        doc, skipped = sidecar.build([{"entries": [tex, uav, rw]}])
        self.assertEqual(doc["stages"], {})
        self.assertEqual(doc["version"], 1)
        self.assertEqual(skipped, {"binding without source": 3})

        doc, skipped = sidecar.build([{"entries": [tex, uav, rw]}], unresolved=True)
        self.assertEqual(doc["version"], 3)
        self.assertEqual(skipped, {"writable binding without source": 2})
        bindings = doc["stages"]["0x00000000000000f1"]["bindings"]
        self.assertEqual(bindings[0]["source"], {"user_sgpr": 8, "table_byte_offset": 288})
        self.assertEqual(bindings[1]["kind"], "texture")
        self.assertEqual(bindings[1]["source"], {"unresolved": True})
        self.assertEqual(bindings[2]["source"], {"unresolved": True})
        summary = sidecar.summarize_stages(doc["stages"])
        self.assertEqual((summary["unresolved_stages"], summary["unresolved_bindings"]), (1, 2))

        # A traced entry stays exactly as before (no unresolved marker, version unchanged).
        doc, skipped = sidecar.build([vertex_entry(True)], unresolved=True)
        self.assertEqual((doc["version"], skipped), (1, {}))

        # The missing-sources report keeps describing what the translator did not trace.
        report = sidecar.build_missing_sources_report([{"entries": [tex]}])
        self.assertEqual(report["total_skipped_missing_source"], 1)

    def test_merge_keeps_traced_existing_over_unresolved_new(self):
        traced = {"stage": "fragment", "bindings": [{"group": 0, "binding": 0, "kind": "texture",
                                                      "source": {"user_sgpr": 2, "table_byte_offset": 0}}]}
        entry = {"hash_fnv1a64": "0xaa", "stage": "pixel",
                 "bindings": [{"group": 0, "binding": 0, "kind": "texture_2d"}]}
        doc, _ = sidecar.build([{"entries": [entry]}], existing_stages={"0xaa": traced}, unresolved=True)
        self.assertEqual(doc["stages"]["0xaa"], traced)
        doc, _ = sidecar.build([{"entries": [entry]}], existing_stages={}, unresolved=True)
        self.assertEqual(doc["stages"]["0xaa"]["bindings"][0]["source"], {"unresolved": True})

    def test_unresolved_source_from_translator_is_validated(self):
        ok = self._sourceless_fragment("0x00000000000000f4", [])
        ok["bindings"].append({"group": 0, "binding": 1, "kind": "texture_2d", "source": {"unresolved": True}})
        ok["bindings"].append({"group": 0, "binding": 2, "kind": "sampler",
                               "source": {"unresolved": "cb_dynamic_offset"}})
        doc, skipped = sidecar.build([ok])
        self.assertEqual((doc["version"], skipped), (3, {}))
        self.assertEqual(doc["stages"]["0x00000000000000f4"]["bindings"][2]["source"],
                         {"unresolved": "cb_dynamic_offset"})
        for bad_source, kind in (({"unresolved": False}, "texture_2d"), ({"unresolved": True, "x": 1}, "texture_2d"),
                                 ({"unresolved": True}, "storage"), ({"unresolved": ""}, "texture_2d"),
                                 ({"unresolved": "C:\\Users\\x"}, "texture_2d")):
            bad = self._sourceless_fragment("0x00000000000000f5", [])
            bad["bindings"].append({"group": 0, "binding": 1, "kind": kind, "source": bad_source})
            doc, skipped = sidecar.build([bad])
            self.assertEqual(skipped, {"malformed unresolved source": 1})

    def test_unresolved_sources_cli_merge_and_version_checks(self):
        tex = self._sourceless_fragment("0x00000000000000f6", ["texture_2d"])
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "pack.json"
            out = Path(tmp) / "pk.wgsl.pack.reflect.json"
            src.write_text(json.dumps({"entries": [tex]}), encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = sidecar.main([str(src), "--out", str(out), "--unresolved-sources", "--json"])
            self.assertEqual(rc, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual((summary["stages_written"], summary["unresolved_stages"]), (1, 1))
            written = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(written["version"], 3)

            # A v3 sidecar merges back in and keeps version 3.
            empty = Path(tmp) / "empty.json"
            empty.write_text(json.dumps({"entries": []}), encoding="utf-8")
            merged = Path(tmp) / "merged.reflect.json"
            with redirect_stdout(io.StringIO()):
                rc = sidecar.main([str(empty), "--out", str(merged), "--merge-existing", str(out)])
            self.assertEqual(rc, 0)
            self.assertEqual(json.loads(merged.read_text(encoding="utf-8"))["version"], 3)

            # An unresolved source in a version 2 document is rejected.
            written["version"] = 2
            bad = Path(tmp) / "bad.reflect.json"
            bad.write_text(json.dumps(written), encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                rc = sidecar.main([str(empty), "--out", str(merged), "--merge-existing", str(bad)])
            self.assertEqual(rc, 1)
            self.assertIn("malformed unresolved source", stderr.getvalue())


class _VirtualWgpkStream:
    """Seekable virtual BinaryIO stream with a real header+TOC prefix and an arbitrary 64-bit
    virtual total_len (e.g. >4 GiB) so tests can verify streaming WGPK v2 validation with zero RAM
    or disk blowup while asserting that no payload bytes are ever read."""

    def __init__(
        self,
        prefix: bytes,
        total_len: int,
        *,
        max_return_bytes: int | None = None,
        allow_eof: bool = False,
    ) -> None:
        self._prefix = prefix
        self._total_len = total_len
        self._pos = 0
        self._max_return_bytes = max_return_bytes
        self._allow_eof = allow_eof
        self.read_sizes: list[int] = []
        self.max_pos_read = 0

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self._total_len + offset
        else:
            raise ValueError(f"unsupported whence {whence}")
        if self._pos < 0:
            raise ValueError("negative seek position")
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            raise AssertionError("unbounded read() must not be called on streaming WGPK reader")
        self.read_sizes.append(size)
        if self._pos >= len(self._prefix):
            if self._allow_eof:
                return b""
            raise AssertionError(
                f"attempted to read payload bytes at pos={self._pos} >= prefix_len={len(self._prefix)}"
            )
        end = min(self._pos + size, len(self._prefix))
        if self._max_return_bytes is not None:
            end = min(end, self._pos + self._max_return_bytes)
        data = self._prefix[self._pos:end]
        self._pos = end
        self.max_pos_read = max(self.max_pos_read, self._pos)
        return data

    def __enter__(self) -> "_VirtualWgpkStream":
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


def build_synthetic_wgpk_v2_prefix(
    entries: list[tuple[int, str, int, int]],
    *,
    flags: int = 0,
) -> bytes:
    """Build a WGPK v2 header + TOC prefix from (hash_int, stage_name, offset_u64, length_u32)."""
    header = sidecar.WGPK_HEADER_STRUCT.pack(
        sidecar.WGPK_MAGIC,
        sidecar.WGPK_VERSION_V2,
        len(entries),
        flags,
    )
    toc = bytearray()
    for hash_int, stage_name, offset, length in entries:
        toc.extend(
            sidecar.WGPK_ENTRY_STRUCT_V2.pack(
                hash_int,
                _STAGE_TO_CODE[stage_name],
                b"\x00\x00\x00",
                length,
                offset,
            )
        )
    return bytes(header) + bytes(toc)


class WgpkV2Tests(unittest.TestCase):
    def _v2_pack(self, entries):
        import inspect_wgsl_pack as w
        return w.build_wgpk(list(entries), version=2)

    def test_v2_toc_parses_and_verifies(self):
        raw = self._v2_pack([(1, "vertex", b"// a"), (2, "compute", b"// bb")])
        toc, meta = sidecar.parse_wgpk_toc_bytes_with_meta(raw)
        self.assertEqual([(e["hash"], e["stage"], e["length"]) for e in toc],
                         [("0x0000000000000001", "vertex", 4), ("0x0000000000000002", "compute", 5)])
        self.assertEqual(meta, {
            "pack_version": 2,
            "pack_bytes": len(raw),
            "toc_entry_bytes": 24,
            "is_wgpk_v2": True,
        })
        res = sidecar.verify_sidecar_against_pack(
            toc,
            {"0x0000000000000001": {"stage": "vertex"}},
            pack_meta=meta,
        )
        self.assertEqual(res["matched_hashes"], 1)
        self.assertEqual(res["pack_version"], 2)
        self.assertEqual(res["pack_bytes"], len(raw))
        self.assertEqual(res["toc_entry_bytes"], 24)
        self.assertTrue(res["is_wgpk_v2"])

    def test_v2_load_from_disk(self):
        with tempfile.TemporaryDirectory() as t:
            pack = Path(t) / "p.wgsl.pack"
            pack.write_bytes(self._v2_pack([(1, "vertex", b"// a")]))
            self.assertEqual(sidecar.load_wgpk_toc(pack)[0]["hash"], "0x0000000000000001")

    def test_v2_bounds_checked(self):
        raw = self._v2_pack([(1, "vertex", b"// a")])
        with self.assertRaises(ValueError):
            sidecar.parse_wgpk_toc_bytes(raw[:-2])

    def test_streaming_wgpk_v2_over_4gib_bounded_chunks_and_short_reads(self):
        # Synthetic 5 GiB WGPK v2 archive (> 0xFFFFFFFF) validated without reading any payload bytes.
        five_gib = 5 * 1024 * 1024 * 1024  # 0x1_4000_0000
        entries_spec = [
            (0x00000000000000A1, "vertex", 16 + 5 * 24, 128),
            (0x00000000000000B2, "fragment", 0x00000000FFFFF000, 4096),
            (0x00000000000000C3, "compute", 0x0000000100000000, 8192),  # exactly 4 GiB offset
            (0x00000000000000D4, "fragment", 0x0000000120000000, 65536),
            (0x00000000000000E5, "compute", five_gib - 64, 64),
        ]
        prefix = build_synthetic_wgpk_v2_prefix(entries_spec)
        # chunk_entries=2 forces 3 chunk reads (2 + 2 + 1 entries), and max_return_bytes=11 exercises short reads.
        stream = _VirtualWgpkStream(prefix, five_gib, max_return_bytes=11)
        toc, meta = sidecar.load_wgpk_toc_stream(stream, label="huge.wgsl.pack", chunk_entries=2)

        self.assertEqual(len(toc), 5)
        self.assertEqual(toc[2]["offset"], 0x100000000)
        self.assertEqual(toc[4]["offset"], five_gib - 64)
        self.assertEqual(meta, {
            "pack_version": 2,
            "pack_bytes": five_gib,
            "toc_entry_bytes": 24,
            "is_wgpk_v2": True,
        })
        # Never read past the TOC prefix and never requested > chunk_entries * 24 bytes in a single read.
        self.assertEqual(stream.max_pos_read, len(prefix))
        self.assertTrue(all(sz <= 2 * 24 for sz in stream.read_sizes))

        # Inferred metadata without pack_meta still detects WGPK v2 from >4 GiB offsets.
        inferred = sidecar.verify_sidecar_against_pack(
            toc,
            {"0x00000000000000c3": {"stage": "compute"}},
            pack_label="huge.wgsl.pack",
        )
        self.assertEqual(inferred["pack_version"], 2)
        self.assertTrue(inferred["is_wgpk_v2"])
        self.assertEqual(inferred["toc_entry_bytes"], 24)
        self.assertEqual(inferred["pack_bytes"], five_gib)
        self.assertEqual(inferred["matched_hashes"], 1)
        self.assertEqual(inferred["coverage_pct"], 20.0)

    def test_streaming_wgpk_v2_over_4gib_errors(self):
        five_gib = 5 * 1024 * 1024 * 1024
        data_start_2 = 16 + 2 * 24
        bad_streams: list[tuple[str, bytes, int]] = [
            (
                "v1_exceeds_4gib",
                sidecar.WGPK_HEADER_STRUCT.pack(sidecar.WGPK_MAGIC, sidecar.WGPK_VERSION, 0, 0),
                five_gib,
            ),
            (
                "v2_truncated_toc_mid_chunk",
                build_synthetic_wgpk_v2_prefix([
                    (1, "vertex", data_start_2, 4),
                    (2, "fragment", 0x100000000, 4),
                ])[:-10],
                five_gib,
            ),
            (
                "v2_offset_overlaps_toc",
                build_synthetic_wgpk_v2_prefix([(1, "vertex", 16, 4)]),
                five_gib,
            ),
            (
                "v2_payload_exceeds_5gib_eof",
                build_synthetic_wgpk_v2_prefix([(1, "compute", five_gib - 4, 16)]),
                five_gib,
            ),
            (
                "v2_payload_u64_overflow",
                build_synthetic_wgpk_v2_prefix([(1, "compute", sidecar.WGPK_U64_MAX - 2, 16)]),
                sidecar.WGPK_U64_MAX,
            ),
            (
                "v2_overlapping_64bit_payloads",
                build_synthetic_wgpk_v2_prefix([
                    (1, "vertex", 0x100000000, 32),
                    (2, "fragment", 0x100000010, 32),
                ]),
                five_gib,
            ),
            (
                "v2_duplicate_hash",
                build_synthetic_wgpk_v2_prefix([
                    (0xDEAD, "vertex", 0x100000000, 8),
                    (0xDEAD, "compute", 0x100000010, 8),
                ]),
                five_gib,
            ),
            (
                "v2_invalid_stage_code",
                sidecar.WGPK_HEADER_STRUCT.pack(sidecar.WGPK_MAGIC, sidecar.WGPK_VERSION_V2, 1, 0)
                + sidecar.WGPK_ENTRY_STRUCT_V2.pack(1, 9, b"\x00\x00\x00", 4, 0x100000000),
                five_gib,
            ),
        ]
        for case, prefix, total_len in bad_streams:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    sidecar.load_wgpk_toc_stream(
                        _VirtualWgpkStream(prefix, total_len, allow_eof=(case == "v2_truncated_toc_mid_chunk")),
                        chunk_entries=1,
                    )

    def test_cli_verify_pack_v2_over_4gib_and_verify_only_preserves_sidecar_version(self):
        six_gib = 6 * 1024 * 1024 * 1024
        prefix = build_synthetic_wgpk_v2_prefix([
            (0x00000000000000A1, "vertex", 0x100000000, 64),
            (0x00000000000000F6, "fragment", 0x140000000, 128),
            (0x00000000000000C9, "compute", 0x17FFFFFC0, 64),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            pack_path = Path(tmp) / "mega_v2.wgsl.pack"
            pack_path.write_bytes(prefix)  # stub file on disk; open() is mocked to serve 6 GiB virtual stream

            real_open = Path.open

            def fake_open(path_self: Path, mode: str = "r", *args, **kwargs):
                if path_self == pack_path and "b" in mode:
                    return _VirtualWgpkStream(prefix, six_gib)
                return real_open(path_self, mode, *args, **kwargs)

            # Verify standalone (verify-only) mode preserves v1, v2, and v3 sidecar versions accurately.
            sidecar_docs = [
                (
                    1,
                    {
                        "schema": sidecar.SCHEMA,
                        "version": 1,
                        "stages": {
                            "0x00000000000000a1": {
                                "stage": "vertex",
                                "bindings": [
                                    {"group": 0, "binding": 0, "kind": "uniform",
                                     "source": {"user_sgpr": 8, "table_byte_offset": 0}}
                                ],
                            }
                        },
                    },
                ),
                (
                    2,
                    {
                        "schema": sidecar.SCHEMA,
                        "version": 2,
                        "stages": {
                            "0x00000000000000a1": {
                                "stage": "vertex",
                                "bindings": [
                                    {"group": 0, "binding": 0, "kind": "uniform",
                                     "source": {"user_sgpr": 8, "root_descriptor": True, "table_byte_offset": 32}}
                                ],
                            }
                        },
                    },
                ),
                (
                    3,
                    {
                        "schema": sidecar.SCHEMA,
                        "version": 3,
                        "stages": {
                            "0x00000000000000a1": {
                                "stage": "vertex",
                                "bindings": [
                                    {"group": 0, "binding": 0, "kind": "uniform",
                                     "source": {"user_sgpr": 8, "root_descriptor": True, "table_byte_offset": 32}}
                                ],
                            },
                            "0x00000000000000f6": {
                                "stage": "fragment",
                                "bindings": [
                                    {"group": 0, "binding": 0, "kind": "texture",
                                     "source": {"unresolved": "cb_dynamic_offset"}}
                                ],
                            },
                        },
                    },
                ),
            ]
            with mock.patch.object(Path, "open", autospec=True, side_effect=fake_open):
                for expected_ver, doc in sidecar_docs:
                    sidecar_path = Path(tmp) / f"v{expected_ver}.reflect.json"
                    sidecar_path.write_text(json.dumps(doc), encoding="utf-8")

                    stdout = io.StringIO()
                    with redirect_stdout(stdout):
                        rc = sidecar.main(["--out", str(sidecar_path), "--verify-pack", str(pack_path), "--json"])
                    self.assertEqual(rc, 0)
                    summary = json.loads(stdout.getvalue())
                    self.assertEqual(summary["sidecar_version"], expected_ver)
                    pv = summary["pack_verification"]
                    self.assertEqual(pv["pack_version"], 2)
                    self.assertEqual(pv["pack_bytes"], six_gib)
                    self.assertEqual(pv["toc_entry_bytes"], 24)
                    self.assertTrue(pv["is_wgpk_v2"])
                    if expected_ver == 3:
                        self.assertEqual(pv["matched_hashes"], 2)
                        self.assertEqual(pv["gap_counts"]["unresolved_matched_stages"], 1)
                        self.assertEqual(summary["root_descriptor_table_offset_stages"], 1)
                        self.assertEqual(summary["root_descriptor_table_offset_bindings"], 1)

                # Text output surfaces WGPK v2 metadata, sidecar version, and actionable gaps.
                stdout_txt = io.StringIO()
                with redirect_stdout(stdout_txt):
                    rc_txt = sidecar.main([
                        "--out", str(Path(tmp) / "v3.reflect.json"),
                        "--verify-pack", str(pack_path),
                    ])
                self.assertEqual(rc_txt, 0)
                text = stdout_txt.getvalue()
                self.assertIn("sidecar v3", text)
                self.assertIn(f"[WGPK v2, {six_gib} B, 24 B/entry]", text)
                self.assertIn("unresolved=1 (1 stages; kinds=[texture:1]; reasons=[cb_dynamic_offset:1])", text)
                self.assertIn("pack_missing_by_stage=v:0/f:0/c:1", text)


class V3SchemaAndGapBreakdownTests(unittest.TestCase):
    def test_rt_arena_stage_flag_and_group3_bindings_with_and_without_kind(self):
        entries = [
            # Stage-level rt_arena: true with empty bindings
            {
                "hash_fnv1a64": "0x000000000000007b",
                "stage": "compute",
                "rt_arena": True,
                "bindings": [],
            },
            # Stage-level rt_arena: false must be omitted in output
            {
                "hash_fnv1a64": "0x000000000000007c",
                "stage": "compute",
                "rt_arena": False,
                "bindings": [],
            },
            # Per-binding group 3, bindings 16..21: both WITH kind and WITHOUT kind
            {
                "hash_fnv1a64": "0x000000000000007a",
                "stage": "compute",
                "bindings": [
                    {"group": 3, "binding": 16, "kind": "storage", "source": {"rt_arena": True}},
                    {"group": 3, "binding": 20, "kind": "read-only-storage", "source": {"rt_arena": True}},
                    {"group": 3, "binding": 21, "source": {"rt_arena": True}},
                    {"group": 0, "binding": 0, "kind": "uniform",
                     "source": {"user_sgpr": 4, "root_constants": True}},
                ],
            },
            # Both stage flag AND per-binding rt_arena must count as 1 rt_arena_stage (not 2)
            {
                "hash_fnv1a64": "0x000000000000007d",
                "stage": "fragment",
                "rt_arena": True,
                "bindings": [
                    {"group": 3, "binding": 17, "source": {"rt_arena": True}},
                    {"group": 3, "binding": 18, "kind": "storage", "source": {"rt_arena": True}},
                ],
            },
        ]
        doc, skipped = sidecar.build([{"entries": entries}])
        self.assertEqual(skipped, {})
        self.assertEqual(doc["version"], 1)
        self.assertTrue(doc["stages"]["0x000000000000007b"]["rt_arena"])
        self.assertNotIn("rt_arena", doc["stages"]["0x000000000000007c"])
        b_7a = doc["stages"]["0x000000000000007a"]["bindings"]
        self.assertEqual(b_7a[0], {"group": 3, "binding": 16, "kind": "storage", "source": {"rt_arena": True}})
        self.assertEqual(b_7a[1], {"group": 3, "binding": 20, "kind": "read_only_storage", "source": {"rt_arena": True}})
        self.assertEqual(b_7a[2], {"group": 3, "binding": 21, "source": {"rt_arena": True}})
        self.assertNotIn("kind", b_7a[2])

        summary = sidecar.summarize_stages(doc["stages"])
        self.assertEqual(summary["rt_arena_stages"], 3)
        self.assertEqual(summary["rt_arena_bindings"], 5)

        # Round-trip through _load_sidecar succeeds for both with-kind and without-kind bindings.
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "rt.reflect.json"
            p.write_text(json.dumps(doc), encoding="utf-8")
            loaded = sidecar._load_sidecar(p, label="sidecar", allow_missing=False)
            self.assertEqual(loaded, doc["stages"])

    def test_rt_arena_invalid_group_binding_and_types_rejected_in_convert_and_load(self):
        bad_bindings = [
            {"group": 2, "binding": 16, "kind": "storage", "source": {"rt_arena": True}},
            {"group": 4, "binding": 16, "kind": "storage", "source": {"rt_arena": True}},
            {"group": 3, "binding": 15, "kind": "storage", "source": {"rt_arena": True}},
            {"group": 3, "binding": 22, "kind": "storage", "source": {"rt_arena": True}},
            {"group": 3, "binding": 16, "kind": "storage", "source": {"rt_arena": False}},
            {"group": 3, "binding": 16, "kind": "storage", "source": {"rt_arena": 1}},
            {"group": 3, "binding": 16, "kind": "storage", "source": {"rt_arena": "true"}},
            {"group": 3, "binding": 16, "kind": "storage", "source": {"rt_arena": True, "extra": 1}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for idx, bad_b in enumerate(bad_bindings):
                entry = {"hash_fnv1a64": f"0x{idx + 1:016x}", "stage": "compute", "bindings": [bad_b]}
                doc, skipped = sidecar.build([entry])
                self.assertEqual(doc["stages"], {}, bad_b)
                self.assertEqual(skipped, {"malformed rt_arena source": 1}, bad_b)

                p = Path(tmp) / f"bad_rt_{idx}.reflect.json"
                p.write_text(json.dumps({
                    "schema": sidecar.SCHEMA,
                    "version": 3,
                    "stages": {"0x1": {"stage": "compute", "bindings": [bad_b]}},
                }), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "malformed rt_arena source"):
                    sidecar._load_sidecar(p, label="sidecar", allow_missing=False)

            # Invalid stage-level rt_arena flag (non-bool) rejected in both convert and load
            for idx, bad_flag in enumerate((1, "true", None, [])):
                entry = {"hash_fnv1a64": "0x0000000000000099", "stage": "compute", "rt_arena": bad_flag, "bindings": []}
                doc, skipped = sidecar.build([entry])
                self.assertEqual(doc["stages"], {}, bad_flag)
                self.assertEqual(skipped, {"malformed rt_arena flag": 1}, bad_flag)

                p = Path(tmp) / f"bad_flag_{idx}.reflect.json"
                p.write_text(json.dumps({
                    "schema": sidecar.SCHEMA,
                    "version": 3,
                    "stages": {"0x1": {"stage": "compute", "rt_arena": bad_flag, "bindings": []}},
                }), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "malformed rt_arena flag"):
                    sidecar._load_sidecar(p, label="sidecar", allow_missing=False)

    def test_load_sidecar_consistently_validates_all_v1_v2_v3_sources(self):
        # Every source schema validated by convert() must also be enforced by _load_sidecar().
        bad_cases: list[tuple[str, int, dict, str]] = [
            # Writable kinds with unresolved rejected even in v3
            (
                "unresolved_storage_v3",
                3,
                {"stage": "compute", "bindings": [
                    {"group": 0, "binding": 0, "kind": "storage", "source": {"unresolved": True}}
                ]},
                "malformed unresolved source",
            ),
            (
                "unresolved_storage_texture_v3",
                3,
                {"stage": "compute", "bindings": [
                    {"group": 0, "binding": 0, "kind": "storage_texture", "source": {"unresolved": "rw_uav"}}
                ]},
                "malformed unresolved source",
            ),
            (
                "unresolved_in_v1",
                1,
                {"stage": "fragment", "bindings": [
                    {"group": 0, "binding": 0, "kind": "texture", "source": {"unresolved": True}}
                ]},
                "malformed unresolved source",
            ),
            # root_descriptor.table_byte_offset malformed rejected in v1, v2, and v3
            (
                "root_descriptor_bad_offset_v1",
                1,
                {"stage": "fragment", "bindings": [
                    {"group": 0, "binding": 0, "kind": "uniform",
                     "source": {"user_sgpr": 8, "root_descriptor": True, "table_byte_offset": 12}}
                ]},
                "malformed v1 root_descriptor source",
            ),
            (
                "root_descriptor_bad_offset_v3",
                3,
                {"stage": "fragment", "bindings": [
                    {"group": 0, "binding": 0, "kind": "uniform",
                     "source": {"user_sgpr": 8, "root_descriptor": True, "table_byte_offset": 16}}
                ]},
                "malformed v3 root_descriptor source",
            ),
            # lane_indexed_*_heap malformed on load
            (
                "lane_heap_bad_count",
                3,
                {"stage": "fragment", "bindings": [
                    {"group": 0, "binding": 0, "kind": "texture",
                     "source": {"lane_indexed_resource_heap": {"max_count": 300, "shift": 5}}}
                ]},
                "malformed lane-indexed heap source",
            ),
            (
                "lane_sampler_bad_window_base",
                3,
                {"stage": "fragment", "bindings": [
                    {"group": 0, "binding": 0, "kind": "sampler",
                     "source": {"lane_indexed_sampler_heap": {
                         "max_count": 16, "shift": 4,
                         "window_base": {"cbv": {"user_sgpr": 8, "root_descriptor": True}, "byte_offset": 6},
                     }}}
                ]},
                "malformed lane-indexed heap source",
            ),
            # cbv_indexed_*_heap malformed on load
            (
                "cbv_heap_bad_shift",
                3,
                {"stage": "fragment", "bindings": [
                    {"group": 0, "binding": 0, "kind": "texture",
                     "source": {"cbv_indexed_resource_heap": {
                         "cbv": {"user_sgpr": 8, "root_descriptor": True}, "byte_offset": 0, "shift": 32,
                     }}}
                ]},
                "malformed CB-indexed heap source",
            ),
            # inline_sampler malformed on load
            (
                "inline_sampler_bad_sdwords",
                3,
                {"stage": "fragment", "bindings": [
                    {"group": 0, "binding": 0, "kind": "sampler",
                     "source": {"inline_sampler": {"sdwords": [1, 2, 3]}}}
                ]},
                "malformed inline sampler",
            ),
            # direct heap malformed on load
            (
                "heap_bad_shift",
                1,
                {"stage": "fragment", "bindings": [
                    {"group": 0, "binding": 0, "kind": "texture",
                     "source": {"resource_heap": {"base_user_sgpr": 4, "stride": 32,
                                                  "index": {"user_sgpr": 5, "shift": 40}}}}
                ]},
                "malformed heap source",
            ),
            # non-compute mesh_output rejected on load
            (
                "mesh_output_on_fragment",
                3,
                {"stage": "fragment", "bindings": [],
                 "mesh_output": {"max_vertices": 3, "max_primitives": 1, "vertex_words": 8}},
                "malformed mesh_output",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for name, ver, stage_obj, expected_err in bad_cases:
                with self.subTest(case=name):
                    p = Path(tmp) / f"{name}.reflect.json"
                    p.write_text(json.dumps({
                        "schema": sidecar.SCHEMA,
                        "version": ver,
                        "stages": {"0x00000000000000aa": stage_obj},
                    }), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, expected_err):
                        sidecar._load_sidecar(p, label="sidecar", allow_missing=False)

    def test_merge_existing_preserves_fewer_unresolved_and_upgrades_more_traced(self):
        # Regression guard for PR #1735: fewer unresolved bindings always wins, and a fully traced
        # stage upgrades an existing unresolved stage.
        one_unresolved = {
            "stage": "fragment",
            "bindings": [
                {"group": 0, "binding": 0, "kind": "texture", "source": {"unresolved": True}},
                {"group": 0, "binding": 1, "kind": "sampler", "source": {"static_sampler": 0}},
            ],
        }
        two_unresolved_entry = {
            "hash_fnv1a64": "0x00000000000000bb",
            "stage": "fragment",
            "bindings": [
                {"group": 0, "binding": 0, "kind": "texture"},
                {"group": 0, "binding": 1, "kind": "sampler"},
            ],
        }
        fully_traced_entry = {
            "hash_fnv1a64": "0x00000000000000bb",
            "stage": "fragment",
            "bindings": [
                {"group": 0, "binding": 0, "kind": "texture", "source": {"user_sgpr": 8, "table_byte_offset": 0}},
                {"group": 0, "binding": 1, "kind": "sampler", "source": {"static_sampler": 0}},
            ],
        }
        # Existing has 1 unresolved, new has 2 -> existing preserved
        doc1, _ = sidecar.build([two_unresolved_entry], existing_stages={"0x00000000000000bb": one_unresolved}, unresolved=True)
        self.assertEqual(doc1["stages"]["0x00000000000000bb"], one_unresolved)
        # Existing has 1 unresolved, new has 0 (fully traced) -> new wins and version drops to 1
        doc2, _ = sidecar.build([fully_traced_entry], existing_stages={"0x00000000000000bb": one_unresolved}, unresolved=True)
        self.assertEqual(doc2["version"], 1)
        self.assertEqual(doc2["stages"]["0x00000000000000bb"]["bindings"][0]["source"], {"user_sgpr": 8, "table_byte_offset": 0})

    def test_actionable_per_cause_gap_breakdown_and_strict_sanitization(self):
        # Pack with 7 entries: 1 matched valid, 6 stage mismatches (tests 5-sample cap on mismatch samples)
        pack_entries = [(0x00000000000000A1, "vertex", b"// vs")] + [
            (0x0000000000003000 + i, "compute", f"// cs {i}".encode("utf-8")) for i in range(6)
        ] + [(0x0000000000004001, "fragment", b"// missing fs")]
        pack_bytes = build_synthetic_wgpk(pack_entries)

        entries = [
            # Matched valid vertex stage
            vertex_entry(True),
            # 6 fragment stages matching compute hashes in pack (6 stage mismatches, 2 unresolved kinds/reasons)
            *[
                {
                    "hash_fnv1a64": f"0x000000000000300{i}",
                    "stage": "fragment",
                    "bindings": [
                        {
                            "group": 0,
                            "binding": 0,
                            "kind": "texture_2d" if i < 4 else "sampler",
                            "source": {"unresolved": "cb_dynamic_offset" if i < 3 else True},
                        }
                    ],
                }
                for i in range(6)
            ],
            # Skipped stages across multiple causes (including hostile path/WGSL in kind!)
            {
                "hash_fnv1a64": "0x0000000000005001",
                "stage": "compute",
                "bindings": [{"group": 0, "binding": 0, "kind": "storage", "name": "rw0"}],
            },
            {
                "hash_fnv1a64": "0x0000000000005002",
                "stage": "compute",
                "bindings": [{"group": 0, "binding": 0, "kind": "storage_texture", "name": "uav0"}],
            },
            {
                "hash_fnv1a64": "0x0000000000005003",
                "stage": "fragment",
                "bindings": [{"group": 0, "binding": 0, "kind": r"C:\Users\secret_user\bad_kind",
                              "source": {"user_sgpr": 8, "table_byte_offset": 0}}],
            },
            {
                "hash_fnv1a64": "0x0000000000005004",
                "stage": "fragment",
                "wgsl": "@fragment fn leak_wgsl() -> @location(0) vec4<f32> { return vec4<f32>(0.0); }",
                "bindings": [{"group": 3, "binding": 99, "kind": "storage", "source": {"rt_arena": True}}],
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            secret_dir = Path(tmp) / "Users" / "secret_user" / "private_ws"
            secret_dir.mkdir(parents=True)
            src = secret_dir / "in.json"
            out = secret_dir / "out.reflect.json"
            report_path = secret_dir / "gaps.json"
            pack_path = secret_dir / "game.wgsl.pack"
            src.write_text(json.dumps({"entries": entries}), encoding="utf-8")
            pack_path.write_bytes(pack_bytes)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = sidecar.main([
                    str(src),
                    "--out", str(out),
                    "--unresolved-sources",
                    "--report-missing-sources", str(report_path),
                    "--verify-pack", str(pack_path),
                    "--json",
                ])
            self.assertEqual(rc, 0)
            raw_json = stdout.getvalue()
            raw_report = report_path.read_text(encoding="utf-8")

            stdout_text = io.StringIO()
            with redirect_stdout(stdout_text):
                rc_text = sidecar.main([
                    str(src),
                    "--out", str(out),
                    "--unresolved-sources",
                    "--verify-pack", str(pack_path),
                ])
            self.assertEqual(rc_text, 0)
            raw_text = stdout_text.getvalue()

            # Missing input file OSError must not leak host directory or username
            stderr_err = io.StringIO()
            with redirect_stderr(stderr_err):
                rc_err = sidecar.main([str(secret_dir / "missing_input.json"), "--out", str(out)])
            self.assertEqual(rc_err, 1)
            raw_err = stderr_err.getvalue()

        for blob in (raw_json, raw_report, raw_text, raw_err):
            for forbidden in ("secret_user", "private_ws", "leak_wgsl", "@fragment", "C:\\Users", str(tmp)):
                self.assertNotIn(forbidden, blob)

        summary = json.loads(raw_json)
        self.assertEqual(summary["sidecar_version"], 3)
        self.assertEqual(summary["unresolved_by_kind"], {"texture": 4, "sampler": 2})
        self.assertEqual(summary["unresolved_by_reason"], {"cb_dynamic_offset": 3, "unspecified": 3})

        gb = summary["gap_breakdown"]
        self.assertEqual(gb["skipped_total"], 4)
        self.assertEqual(gb["skipped_by_cause"], [
            {"cause": "writable binding without source", "count": 2, "pct_of_skipped": 50.0},
            {"cause": "malformed rt_arena source", "count": 1, "pct_of_skipped": 25.0},
            {"cause": "unknown binding kind '<redacted>'", "count": 1, "pct_of_skipped": 25.0},
        ])
        self.assertEqual(gb["pack_missing_stages"], 1)
        self.assertEqual(gb["pack_missing_by_stage"], {"vertex": 0, "fragment": 1, "compute": 0})
        self.assertEqual(gb["pack_stage_mismatches"], 6)

        pv = summary["pack_verification"]
        self.assertEqual(pv["stage_mismatches"], 6)
        self.assertEqual(len(pv["stage_mismatch_samples"]), 5)
        self.assertEqual(pv["stage_mismatch_samples"], [f"0x000000000000300{i}" for i in range(5)])
        self.assertEqual(pv["gap_counts"], {
            "missing_reflection": 1,
            "stage_mismatches": 6,
            "extra_reflection": 0,
            "unresolved_matched_stages": 6,
        })

        report = json.loads(raw_report)
        self.assertEqual(report["total_skipped_missing_source"], 2)
        self.assertEqual(report["missing_by_kind"], {"storage": 1, "storage_texture": 1})
        self.assertEqual(report["skipped_total"], 4)
        self.assertEqual(report["skipped_by_cause"], gb["skipped_by_cause"])
        self.assertEqual(report["unresolved_by_kind"], {"texture": 4, "sampler": 2})
        self.assertEqual(report["unresolved_by_reason"], {"cb_dynamic_offset": 3, "unspecified": 3})

        # Text summary includes actionable gap causes
        self.assertIn("writable binding without source:2 (50.0%)", raw_text)
        self.assertIn("stage_mismatches=6", raw_text)

    def test_cli_help_reflects_wgpk_v1_v2_and_sidecar_v1_v2_v3(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as ctx:
                sidecar.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)
        help_text = " ".join(stdout.getvalue().split())
        self.assertIn("WGPK v1/v2", help_text)
        self.assertIn("v1/v2/v3 sidecar", help_text)
        self.assertNotIn("WGPK v1 .wgsl.pack", help_text)
        self.assertNotIn("existing v1 sidecar", help_text)

    def test_vs_outputs_and_ps_inputs_propagated_and_validated(self):
        # Issue #2085: Brightness Calibration VS_Blit (COLOR0@0, TEXCOORD0@1) +
        # PS_BlitCalibration (unused COLOR0 mask=0, TEXCOORD0@0 mask=3).
        vs_entry = {
            "hash_fnv1a64": "0x5fe7a88d9cb680b3",
            "stage": "vertex",
            "bindings": [
                {"group": 0, "binding": 0, "kind": "uniform",
                 "source": {"user_sgpr": 8, "table_byte_offset": 0}},
            ],
            "vs_outputs": [
                {"semantic_reg": 1, "location": 0},
                {"semantic_reg": 2, "location": 1},
            ],
        }
        ps_entry = {
            "hash_fnv1a64": "0x56214b8d3b10a108",
            "stage": "fragment",
            "bindings": [
                {"group": 0, "binding": 0, "kind": "texture",
                 "source": {"user_sgpr": 8, "table_byte_offset": 0}},
            ],
            "ps_inputs": [
                {"semantic_reg": 1, "location": 0, "mask": 0},
                {"semantic_reg": 2, "location": 0, "mask": 3},
            ],
        }
        # Empty arrays are omitted for compact backward-compatible sidecars
        vs_empty = {
            "hash_fnv1a64": "0x000000000000aa01",
            "stage": "vertex",
            "bindings": [],
            "vs_outputs": [],
        }

        doc, skipped = sidecar.build([{"entries": [vs_entry, ps_entry, vs_empty]}])
        self.assertEqual(skipped, {})
        self.assertEqual(doc["version"], 1)
        self.assertEqual(
            doc["stages"]["0x5fe7a88d9cb680b3"]["vs_outputs"],
            [{"semantic_reg": 1, "location": 0}, {"semantic_reg": 2, "location": 1}],
        )
        self.assertNotIn("ps_inputs", doc["stages"]["0x5fe7a88d9cb680b3"])
        self.assertEqual(
            doc["stages"]["0x56214b8d3b10a108"]["ps_inputs"],
            [{"semantic_reg": 1, "location": 0, "mask": 0}, {"semantic_reg": 2, "location": 0, "mask": 3}],
        )
        self.assertNotIn("vs_outputs", doc["stages"]["0x56214b8d3b10a108"])
        self.assertNotIn("vs_outputs", doc["stages"]["0x000000000000aa01"])

        # Round-trip through CLI + --merge-existing preserves vs_outputs & ps_inputs
        with tempfile.TemporaryDirectory() as tmp:
            src1 = Path(tmp) / "in1.json"
            src2 = Path(tmp) / "in2.json"
            out = Path(tmp) / "pack.reflect.json"
            src1.write_text(json.dumps({"entries": [vs_entry]}), encoding="utf-8")
            src2.write_text(json.dumps({"entries": [ps_entry]}), encoding="utf-8")
            self.assertEqual(sidecar.main([str(src1), "--out", str(out)]), 0)
            self.assertEqual(sidecar.main([str(src2), "--out", str(out), "--merge-existing", str(out)]), 0)
            merged = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(merged["stages"]["0x5fe7a88d9cb680b3"]["vs_outputs"], vs_entry["vs_outputs"])
        self.assertEqual(merged["stages"]["0x56214b8d3b10a108"]["ps_inputs"], ps_entry["ps_inputs"])

        # Malformed vs_outputs / ps_inputs are rejected cleanly
        bad_cases = [
            ({"hash_fnv1a64": "0x1", "stage": "vertex", "bindings": [], "vs_outputs": "nope"}, "malformed vs_outputs"),
            ({"hash_fnv1a64": "0x2", "stage": "vertex", "bindings": [], "vs_outputs": [{"semantic_reg": 256, "location": 0}]}, "malformed vs_outputs"),
            ({"hash_fnv1a64": "0x3", "stage": "vertex", "bindings": [], "vs_outputs": [{"semantic_reg": True, "location": 0}]}, "malformed vs_outputs"),
            ({"hash_fnv1a64": "0x4", "stage": "fragment", "bindings": [], "vs_outputs": [{"semantic_reg": 1, "location": 0}]}, "malformed vs_outputs"),
            ({"hash_fnv1a64": "0x5", "stage": "fragment", "bindings": [], "ps_inputs": [{"semantic_reg": 2, "location": -1}]}, "malformed ps_inputs"),
            ({"hash_fnv1a64": "0x6", "stage": "fragment", "bindings": [], "ps_inputs": [{"semantic_reg": 2, "location": 0, "mask": 16}]}, "malformed ps_inputs"),
            ({"hash_fnv1a64": "0x7", "stage": "vertex", "bindings": [], "ps_inputs": [{"semantic_reg": 2, "location": 0, "mask": 3}]}, "malformed ps_inputs"),
        ]
        for bad_entry, expected_reason in bad_cases:
            bad_doc, bad_skipped = sidecar.build([bad_entry])
            self.assertEqual(bad_doc["stages"], {}, bad_entry)
            self.assertEqual(bad_skipped, {expected_reason: 1}, bad_entry)


if __name__ == "__main__":
    unittest.main()
