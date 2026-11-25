"""Tests for tools/stage_reflection_sidecar.py (synthetic translator JSON only)."""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
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
            "stages_written": 3,
            "skipped": {"binding without source": 1},
            "merged_existing": 2,
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
                "stages_written": 1,
                "skipped": {},
                "merged_existing": 0,
            })

            # Second run: in-place merge onto the file written in run 1
            stdout2 = io.StringIO()
            with redirect_stdout(stdout2):
                rc2 = sidecar.main([str(in2), "--out", str(out), "--merge-existing", str(out), "--json"])
            self.assertEqual(rc2, 0)
            self.assertEqual(json.loads(stdout2.getvalue()), {
                "stages_written": 2,
                "skipped": {},
                "merged_existing": 1,
            })
            written = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(set(written["stages"].keys()), {"0x00000000000000a1", "0x00000000000000b2"})

    def test_merge_existing_invalid_schema_fails_cleanly(self):
        bad_payloads = [
            "{not valid json",
            json.dumps([1, 2, 3]),
            json.dumps({"schema": "wrong-schema", "version": 1, "stages": {}}),
            json.dumps({"schema": "playgta6-stage-reflection", "version": 2, "stages": {}}),
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
        self.assertEqual(summary, {"stages_written": 1, "skipped": {}})
        self.assertNotIn("merged_existing", summary)

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


if __name__ == "__main__":
    unittest.main()
