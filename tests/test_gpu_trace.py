import io
import json
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import gpu_trace  # noqa: E402


SAMPLE_VS_WGSL = """@vertex
fn vs_main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(pos, 1.0);
}
"""

SAMPLE_FS_WGSL = """@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.5, 0.25, 1.0);
}
"""


class TestGpuTraceDisassembler(unittest.TestCase):
    def test_abi_struct_sizes_match_header(self) -> None:
        self.assertEqual(gpu_trace.ABI_VERSION, 1)
        self.assertEqual(gpu_trace.HEADER_SIZE, 8)
        self.assertEqual(gpu_trace.CLEAR_COLOR_STRUCT.size, 16)
        self.assertEqual(gpu_trace.CREATE_BUFFER_STRUCT.size, 16)
        self.assertEqual(gpu_trace.WRITE_BUFFER_STRUCT.size, 16)
        self.assertEqual(gpu_trace.CREATE_SHADER_MODULE_STRUCT.size, 16)
        self.assertEqual(gpu_trace.DRAW_STRUCT.size, 16)
        self.assertEqual(gpu_trace.CREATE_RENDER_PIPELINE_STRUCT.size, 24)
        self.assertEqual(gpu_trace.SET_PIPELINE_STRUCT.size, 4)
        self.assertEqual(gpu_trace.SET_VERTEX_BUFFER_STRUCT.size, 16)
        self.assertEqual(gpu_trace.DESTROY_RESOURCE_STRUCT.size, 4)
        self.assertEqual(gpu_trace.CREATE_TEXTURE_STRUCT.size, 32)
        self.assertEqual(gpu_trace.WRITE_TEXTURE_STRUCT.size, 48)
        self.assertEqual(gpu_trace.CREATE_SAMPLER_STRUCT.size, 32)
        self.assertEqual(gpu_trace.BIND_GROUP_ENTRY_STRUCT.size, 32)
        self.assertEqual(gpu_trace.CREATE_BIND_GROUP_STRUCT.size, 16)
        self.assertEqual(gpu_trace.SET_BIND_GROUP_STRUCT.size, 8)
        self.assertEqual(gpu_trace.SET_INDEX_BUFFER_STRUCT.size, 16)
        self.assertEqual(gpu_trace.DRAW_INDEXED_STRUCT.size, 20)
        self.assertEqual(gpu_trace.CREATE_COMPUTE_PIPELINE_STRUCT.size, 16)
        self.assertEqual(gpu_trace.SET_COMPUTE_PIPELINE_STRUCT.size, 4)
        self.assertEqual(gpu_trace.DISPATCH_COMPUTE_STRUCT.size, 12)

    def test_decode_all_frozen_opcodes_0_through_11_roundtrip(self) -> None:
        vertex_blob = bytes(range(64))
        stream = b"".join(
            [
                gpu_trace.encode_nop(flags=0x0003),
                gpu_trace.encode_begin_frame(),
                gpu_trace.encode_clear_color(0.25, 0.5, 0.75, 1.0),
                gpu_trace.encode_create_buffer(
                    resource_id=10,
                    usage_flags=0x0020 | 0x0008,  # VERTEX | COPY_DST
                    byte_size=1024,
                ),
                gpu_trace.encode_write_buffer(
                    resource_id=10,
                    dst_offset=16,
                    data=vertex_blob,
                    reserved=7,
                ),
                gpu_trace.encode_create_shader_module(
                    resource_id=20,
                    stage=gpu_trace.GpuShaderStage.Vertex,
                    wgsl_code=SAMPLE_VS_WGSL,
                    reserved=0,
                ),
                gpu_trace.encode_create_shader_module(
                    resource_id=21,
                    stage=gpu_trace.GpuShaderStage.Fragment,
                    wgsl_code=SAMPLE_FS_WGSL,
                    reserved=0,
                ),
                gpu_trace.encode_create_render_pipeline(
                    resource_id=30,
                    vs_shader_id=20,
                    fs_shader_id=21,
                    vertex_stride=12,
                    topology=gpu_trace.GpuPrimitiveTopology.TriangleStrip,
                    pipeline_flags=0x1234,
                ),
                gpu_trace.encode_set_pipeline(pipeline_id=30),
                gpu_trace.encode_set_vertex_buffer(slot=1, buffer_id=10, offset=256),
                gpu_trace.encode_draw(
                    vertex_count=36,
                    instance_count=4,
                    first_vertex=3,
                    first_instance=2,
                ),
                gpu_trace.encode_present(),
                gpu_trace.encode_destroy_resource(resource_id=10),
            ]
        )

        trace = gpu_trace.decode_stream(stream, strict=True)
        self.assertTrue(trace.ok)
        self.assertEqual(trace.diagnostics, ())
        self.assertEqual(trace.consumed_bytes, len(stream))
        self.assertEqual(trace.total_bytes, len(stream))
        self.assertEqual(trace.packet_count, 13)
        self.assertEqual(trace.draw_calls, 1)
        self.assertEqual(trace.unknown_opcode_count, 0)

        # 0: Nop
        p0 = trace.packets[0]
        self.assertEqual(p0.opcode, 0)
        self.assertEqual(p0.name, "Nop")
        self.assertEqual(p0.flags, 0x0003)
        self.assertEqual(p0.payload_bytes, 0)
        self.assertEqual(p0.fields, {})

        # 1: BeginFrame
        p1 = trace.packets[1]
        self.assertEqual(p1.opcode, 1)
        self.assertEqual(p1.name, "BeginFrame")
        self.assertEqual(p1.payload_bytes, 0)

        # 2: ClearColor
        p2 = trace.packets[2]
        self.assertEqual(p2.opcode, 2)
        self.assertEqual(p2.name, "ClearColor")
        self.assertAlmostEqual(p2.fields["r"], 0.25, places=6)
        self.assertAlmostEqual(p2.fields["g"], 0.5, places=6)
        self.assertAlmostEqual(p2.fields["b"], 0.75, places=6)
        self.assertAlmostEqual(p2.fields["a"], 1.0, places=6)

        # 3: CreateBuffer
        p3 = trace.packets[3]
        self.assertEqual(p3.opcode, 3)
        self.assertEqual(p3.name, "CreateBuffer")
        self.assertEqual(p3.fields["id"], 10)
        self.assertEqual(p3.fields["usage_flags"], 0x0028)
        self.assertEqual(p3.fields["usage_names"], ["COPY_DST", "VERTEX"])
        self.assertEqual(p3.fields["byte_size"], 1024)

        # 4: WriteBuffer (metadata + preview/CRC32, without dumping full 64B buffer)
        p4 = trace.packets[4]
        self.assertEqual(p4.opcode, 4)
        self.assertEqual(p4.name, "WriteBuffer")
        self.assertEqual(p4.fields["id"], 10)
        self.assertEqual(p4.fields["dst_offset"], 16)
        self.assertEqual(p4.fields["data_len"], 64)
        self.assertEqual(p4.fields["reserved"], 7)
        expected_crc = f"0x{zlib.crc32(vertex_blob) & 0xFFFFFFFF:08x}"
        self.assertEqual(p4.fields["crc32"], expected_crc)
        self.assertEqual(p4.fields["preview_bytes"], gpu_trace.DEFAULT_PREVIEW_BYTES)
        self.assertEqual(
            p4.fields["preview_hex"],
            vertex_blob[: gpu_trace.DEFAULT_PREVIEW_BYTES].hex(),
        )
        self.assertTrue(p4.fields["truncated_preview"])

        # 5: CreateShaderModule (Vertex + Fragment)
        p5 = trace.packets[5]
        self.assertEqual(p5.opcode, 5)
        self.assertEqual(p5.name, "CreateShaderModule")
        self.assertEqual(p5.fields["id"], 20)
        self.assertEqual(p5.fields["stage"], 0)
        self.assertEqual(p5.fields["stage_name"], "Vertex")
        self.assertEqual(p5.fields["code_len"], len(SAMPLE_VS_WGSL.encode("utf-8")))
        self.assertEqual(p5.fields["line_count"], 4)
        self.assertEqual(p5.fields["char_count"], len(SAMPLE_VS_WGSL))
        self.assertEqual(
            p5.fields["wgsl_hash_fnv1a64"],
            f"0x{gpu_trace.fnv1a_64(SAMPLE_VS_WGSL.encode('utf-8')):016x}",
        )
        self.assertTrue(p5.fields["utf8_valid"])

        p6 = trace.packets[6]
        self.assertEqual(p6.fields["id"], 21)
        self.assertEqual(p6.fields["stage_name"], "Fragment")

        # 8: CreateRenderPipeline
        p7 = trace.packets[7]
        self.assertEqual(p7.opcode, 8)
        self.assertEqual(p7.name, "CreateRenderPipeline")
        self.assertEqual(p7.fields["id"], 30)
        self.assertEqual(p7.fields["vs_shader_id"], 20)
        self.assertEqual(p7.fields["fs_shader_id"], 21)
        self.assertEqual(p7.fields["vertex_stride"], 12)
        self.assertEqual(p7.fields["topology"], 1)
        self.assertEqual(p7.fields["topology_name"], "TriangleStrip")
        self.assertEqual(p7.fields["topology_webgpu"], "triangle-strip")
        self.assertEqual(p7.fields["flags"], 0x1234)

        # 9: SetPipeline
        p8 = trace.packets[8]
        self.assertEqual(p8.opcode, 9)
        self.assertEqual(p8.name, "SetPipeline")
        self.assertEqual(p8.fields["pipeline_id"], 30)

        # 10: SetVertexBuffer
        p9 = trace.packets[9]
        self.assertEqual(p9.opcode, 10)
        self.assertEqual(p9.name, "SetVertexBuffer")
        self.assertEqual(p9.fields["slot"], 1)
        self.assertEqual(p9.fields["buffer_id"], 10)
        self.assertEqual(p9.fields["offset"], 256)

        # 6: Draw
        p10 = trace.packets[10]
        self.assertEqual(p10.opcode, 6)
        self.assertEqual(p10.name, "Draw")
        self.assertEqual(p10.fields["vertex_count"], 36)
        self.assertEqual(p10.fields["instance_count"], 4)
        self.assertEqual(p10.fields["first_vertex"], 3)
        self.assertEqual(p10.fields["first_instance"], 2)

        # 7: Present
        p11 = trace.packets[11]
        self.assertEqual(p11.opcode, 7)
        self.assertEqual(p11.name, "Present")
        self.assertEqual(p11.payload_bytes, 0)

        # 11: DestroyResource
        p12 = trace.packets[12]
        self.assertEqual(p12.opcode, 11)
        self.assertEqual(p12.name, "DestroyResource")
        self.assertEqual(p12.fields["id"], 10)

    def test_fnv1a_64_known_vectors(self) -> None:
        # Standard FNV-1a 64-bit reference test vectors.
        self.assertEqual(gpu_trace.fnv1a_64(b""), 0xCBF29CE484222325)
        self.assertEqual(gpu_trace.fnv1a_64(b"a"), 0xAF63DC4C8601EC8C)
        self.assertEqual(gpu_trace.fnv1a_64(b"foobar"), 0x85944171F73967E8)

    def test_all_buffer_usage_flags_and_unknown_bits(self) -> None:
        all_known = 0x03FF
        names, unknown = gpu_trace.decode_buffer_usage(all_known)
        self.assertEqual(unknown, 0)
        self.assertEqual(
            names,
            [
                "MAP_READ",
                "MAP_WRITE",
                "COPY_SRC",
                "COPY_DST",
                "INDEX",
                "VERTEX",
                "UNIFORM",
                "STORAGE",
                "INDIRECT",
                "QUERY_RESOLVE",
            ],
        )

        # Unknown high bit triggers warning in non-strict mode, error in strict mode.
        pkt = gpu_trace.encode_create_buffer(1, 0x8020, 64)
        t_lenient = gpu_trace.decode_stream(pkt, strict=False)
        self.assertTrue(t_lenient.ok)
        self.assertEqual(t_lenient.packets[0].fields["usage_names"], ["VERTEX", "UNKNOWN(0x8000)"])
        self.assertEqual(len(t_lenient.diagnostics), 1)
        self.assertEqual(t_lenient.diagnostics[0].severity, "warning")
        self.assertEqual(t_lenient.diagnostics[0].code, "unknown_buffer_usage_bits")

        t_strict = gpu_trace.decode_stream(pkt, strict=True)
        self.assertFalse(t_strict.ok)
        self.assertEqual(t_strict.diagnostics[0].severity, "error")

    def test_frame_splitting_and_histograms(self) -> None:
        f0 = b"".join(
            [
                gpu_trace.encode_begin_frame(),
                gpu_trace.encode_clear_color(0.0, 0.0, 0.0, 1.0),
                gpu_trace.encode_draw(3, 1, 0, 0),
                gpu_trace.encode_draw(6, 2, 0, 0),
                gpu_trace.encode_present(),
            ]
        )
        f1 = b"".join(
            [
                gpu_trace.encode_begin_frame(),
                gpu_trace.encode_set_pipeline(1),
                gpu_trace.encode_draw(12, 1, 0, 0),
                gpu_trace.encode_present(),
            ]
        )
        trailing = gpu_trace.encode_begin_frame() + gpu_trace.encode_nop()

        trace = gpu_trace.decode_stream(f0 + f1 + trailing)
        self.assertEqual(trace.frame_count, 3)

        self.assertEqual(trace.frames[0].frame_index, 0)
        self.assertEqual(trace.frames[0].packet_count, 5)
        self.assertEqual(trace.frames[0].draw_calls, 2)
        self.assertEqual(trace.frames[0].bytes, len(f0))
        self.assertTrue(trace.frames[0].complete)
        self.assertTrue(trace.frames[0].has_begin_frame)
        self.assertEqual(
            trace.frames[0].opcode_histogram,
            {"BeginFrame": 1, "ClearColor": 1, "Draw": 2, "Present": 1},
        )

        self.assertEqual(trace.frames[1].frame_index, 1)
        self.assertEqual(trace.frames[1].packet_count, 4)
        self.assertEqual(trace.frames[1].draw_calls, 1)
        self.assertEqual(trace.frames[1].bytes, len(f1))
        self.assertTrue(trace.frames[1].complete)
        self.assertEqual(
            trace.frames[1].opcode_histogram,
            {"BeginFrame": 1, "SetPipeline": 1, "Draw": 1, "Present": 1},
        )

        self.assertEqual(trace.frames[2].frame_index, 2)
        self.assertEqual(trace.frames[2].packet_count, 2)
        self.assertEqual(trace.frames[2].draw_calls, 0)
        self.assertEqual(trace.frames[2].bytes, len(trailing))
        self.assertFalse(trace.frames[2].complete)
        self.assertTrue(trace.frames[2].has_begin_frame)

        self.assertEqual(
            trace.opcode_histogram,
            {
                "BeginFrame": 3,
                "ClearColor": 1,
                "SetPipeline": 1,
                "Draw": 3,
                "Present": 2,
                "Nop": 1,
            },
        )

    def test_unknown_future_opcodes_forward_compatibility(self) -> None:
        future_payload = b"\xde\xad\xbe\xef" * 10
        stream = b"".join(
            [
                gpu_trace.encode_begin_frame(),
                gpu_trace.encode_packet(200, future_payload, flags=0x0042),
                gpu_trace.encode_packet(99, b""),
                gpu_trace.encode_present(),
            ]
        )

        # Non-strict mode handles unknown opcodes (>= 35) cleanly without errors.
        lenient = gpu_trace.decode_stream(stream, strict=False)
        self.assertTrue(lenient.ok)
        self.assertEqual(lenient.diagnostics, ())
        self.assertEqual(lenient.packet_count, 4)
        self.assertEqual(lenient.unknown_opcode_count, 2)

        u200 = lenient.packets[1]
        self.assertFalse(u200.known)
        self.assertTrue(u200.valid)
        self.assertEqual(u200.opcode, 200)
        self.assertEqual(u200.name, "Unknown(200)")
        self.assertEqual(u200.flags, 0x0042)
        self.assertEqual(u200.payload_bytes, len(future_payload))
        self.assertEqual(u200.fields["payload_bytes"], len(future_payload))
        self.assertEqual(u200.fields["crc32"], gpu_trace.crc32_hex(future_payload))
        self.assertTrue(u200.fields["truncated_preview"])

        u99 = lenient.packets[2]
        self.assertEqual(u99.name, "Unknown(99)")
        self.assertEqual(u99.payload_bytes, 0)

        # Strict mode flags unknown opcodes as structured errors.
        strict = gpu_trace.decode_stream(stream, strict=True)
        self.assertFalse(strict.ok)
        self.assertTrue(strict.has_errors)
        self.assertEqual([d.code for d in strict.diagnostics], ["unknown_opcode", "unknown_opcode"])
        self.assertFalse(strict.packets[1].valid)

    def test_malformed_and_truncated_stream_diagnostics(self) -> None:
        # 1. Truncated header (5 trailing bytes)
        valid_prefix = gpu_trace.encode_nop()
        trunc_hdr = gpu_trace.decode_stream(valid_prefix + b"\x01\x02\x03\x04\x05")
        self.assertFalse(trunc_hdr.ok)
        self.assertEqual(trunc_hdr.packet_count, 1)
        self.assertEqual(trunc_hdr.consumed_bytes, 8)
        self.assertEqual(len(trunc_hdr.diagnostics), 1)
        self.assertEqual(trunc_hdr.diagnostics[0].code, "truncated_header")
        self.assertEqual(trunc_hdr.diagnostics[0].severity, "error")
        self.assertEqual(trunc_hdr.diagnostics[0].offset, 8)

        # 2. Truncated payload (header declares 100 bytes, only 4 follow)
        bad_header = struct.pack("<HHI", gpu_trace.GpuOpcode.ClearColor, 0, 100) + b"\x00" * 4
        trunc_pay = gpu_trace.decode_stream(bad_header)
        self.assertFalse(trunc_pay.ok)
        self.assertEqual(trunc_pay.packet_count, 0)
        self.assertEqual(trunc_pay.diagnostics[0].code, "truncated_payload")
        self.assertEqual(trunc_pay.diagnostics[0].severity, "error")

        # 3. Fixed payload too short (ClearColor with only 8 bytes) followed by valid Present
        short_clear = gpu_trace.encode_packet(gpu_trace.GpuOpcode.ClearColor, b"\x00" * 8)
        recovered = gpu_trace.decode_stream(short_clear + gpu_trace.encode_present())
        self.assertFalse(recovered.ok)
        self.assertEqual(recovered.packet_count, 2)
        self.assertFalse(recovered.packets[0].valid)
        self.assertTrue(recovered.packets[1].valid)
        self.assertEqual(recovered.packets[1].name, "Present")
        self.assertEqual(recovered.diagnostics[0].code, "payload_too_short")

        # 4. Fixed payload too large (SetPipeline with 8 bytes instead of 4)
        oversize_set_pipe = gpu_trace.encode_packet(
            gpu_trace.GpuOpcode.SetPipeline,
            struct.pack("<II", 42, 0xDEADBEEF),
        )
        over = gpu_trace.decode_stream(oversize_set_pipe, strict=False)
        self.assertTrue(over.ok)
        self.assertTrue(over.has_warnings)
        self.assertEqual(over.packets[0].fields["pipeline_id"], 42)
        self.assertEqual(over.packets[0].fields["trailing_bytes"], 4)
        self.assertEqual(over.diagnostics[0].code, "payload_size_mismatch")
        self.assertEqual(over.diagnostics[0].severity, "warning")

        over_strict = gpu_trace.decode_stream(oversize_set_pipe, strict=True)
        self.assertFalse(over_strict.ok)
        self.assertEqual(over_strict.diagnostics[0].severity, "error")

        # 5. Variable payload length mismatch (WriteBuffer claims 32 bytes, only 4 provided)
        bad_wb = gpu_trace.encode_packet(
            gpu_trace.GpuOpcode.WriteBuffer,
            struct.pack("<IIII", 5, 0, 32, 0) + b"abcd",
        )
        wb_trace = gpu_trace.decode_stream(bad_wb)
        self.assertFalse(wb_trace.ok)
        self.assertFalse(wb_trace.packets[0].valid)
        self.assertEqual(wb_trace.packets[0].fields["available_bytes"], 4)
        self.assertEqual(wb_trace.diagnostics[0].code, "variable_payload_truncated")

    def test_text_and_json_serialization(self) -> None:
        stream = b"".join(
            [
                gpu_trace.encode_begin_frame(),
                gpu_trace.encode_clear_color(0.1, 0.2, 0.3, 1.0),
                gpu_trace.encode_draw(3, 1, 0, 0),
                gpu_trace.encode_present(),
            ]
        )
        trace = gpu_trace.decode_stream(stream)

        full_text = gpu_trace.format_trace_text(trace)
        self.assertIn("playgta6 GPU Trace (ABI v1)", full_text)
        self.assertIn("BeginFrame", full_text)
        self.assertIn("ClearColor", full_text)
        self.assertIn("Draw", full_text)
        self.assertIn("Present", full_text)
        self.assertIn("Frame 000 [complete]", full_text)

        stats_text = gpu_trace.format_trace_text(trace, stats_only=True)
        self.assertNotIn("Packets:\n", stats_text)
        self.assertIn("Opcode Histogram:", stats_text)

        parsed_full = json.loads(gpu_trace.trace_to_json(trace))
        self.assertEqual(parsed_full["abi_version"], 1)
        self.assertEqual(parsed_full["packet_count"], 4)
        self.assertEqual(len(parsed_full["packets"]), 4)
        self.assertEqual(len(parsed_full["frames"]), 1)
        self.assertEqual(parsed_full["frames"][0]["draw_calls"], 1)

        parsed_stats = json.loads(gpu_trace.trace_to_json(trace, stats_only=True))
        self.assertNotIn("packets", parsed_stats)
        self.assertEqual(parsed_stats["packet_count"], 4)

    def test_cli_execution_modes(self) -> None:
        stream = b"".join(
            [
                gpu_trace.encode_begin_frame(),
                gpu_trace.encode_create_shader_module(
                    1,
                    gpu_trace.GpuShaderStage.Compute,
                    "@compute @workgroup_size(1) fn cs_main() {}",
                ),
                gpu_trace.encode_packet(42, b"\x01\x02\x03"),
                gpu_trace.encode_present(),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "capture.bin"
            trace_path.write_bytes(stream)

            # Default text mode via subprocess
            proc = subprocess.run(
                [sys.executable, str(REPO_ROOT / "tools" / "gpu_trace.py"), str(trace_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("CreateShaderModule", proc.stdout)
            self.assertIn("stage=Compute", proc.stdout)
            self.assertIn("Unknown(42)", proc.stdout)

            # --json --stats mode via in-process main()
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = gpu_trace.main([str(trace_path), "--json", "--stats"])
            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["unknown_opcode_count"], 1)
            self.assertNotIn("packets", payload)

            # --strict fails on Unknown(42)
            buf_strict = io.StringIO()
            with redirect_stdout(buf_strict):
                rc_strict = gpu_trace.main([str(trace_path), "--json", "--strict"])
            self.assertEqual(rc_strict, 1)
            strict_payload = json.loads(buf_strict.getvalue())
            self.assertFalse(strict_payload["ok"])
            self.assertEqual(strict_payload["diagnostics"][0]["code"], "unknown_opcode")

            # Missing file returns non-zero exit code
            err_buf = io.StringIO()
            with redirect_stderr(err_buf):
                rc_missing = gpu_trace.main([str(Path(tmp) / "does_not_exist.bin")])
            self.assertEqual(rc_missing, 2)
            self.assertIn("trace file not found", err_buf.getvalue())

    def test_edge_cases_empty_preamble_and_invalid_enums(self) -> None:
        # Empty stream
        empty = gpu_trace.decode_stream(b"", strict=True)
        self.assertTrue(empty.ok)
        self.assertEqual(empty.packet_count, 0)
        self.assertEqual(empty.frame_count, 0)
        self.assertIn("(none)", gpu_trace.format_trace_text(empty))

        # Setup preamble before BeginFrame + invalid UTF-8 WGSL + unknown stage/topology
        stream = b"".join(
            [
                gpu_trace.encode_create_shader_module(1, 99, b"\xff\xfe@vertex"),
                gpu_trace.encode_create_render_pipeline(2, 1, 0, 12, 77, 0),
                gpu_trace.encode_begin_frame(),
                gpu_trace.encode_present(),
            ]
        )
        trace = gpu_trace.decode_stream(stream, strict=False)
        self.assertTrue(trace.ok)
        self.assertEqual(trace.frame_count, 2)
        self.assertFalse(trace.frames[0].has_begin_frame)
        self.assertFalse(trace.frames[0].complete)
        self.assertTrue(trace.frames[1].has_begin_frame)
        self.assertTrue(trace.frames[1].complete)
        self.assertEqual(trace.packets[0].fields["stage_name"], "Unknown(99)")
        self.assertFalse(trace.packets[0].fields["utf8_valid"])
        self.assertEqual(trace.packets[1].fields["topology_name"], "Unknown(77)")
        self.assertIsNone(trace.packets[1].fields["topology_webgpu"])
        diag_codes = {d.code for d in trace.diagnostics}
        self.assertEqual(
            diag_codes,
            {"unknown_shader_stage", "invalid_utf8_wgsl", "unknown_topology"},
        )

        # Strict mode promotes those warnings to errors
        strict_trace = gpu_trace.decode_stream(stream, strict=True)
        self.assertFalse(strict_trace.ok)
        self.assertTrue(all(d.severity == "error" for d in strict_trace.diagnostics))

        # encode_packet range validation
        with self.assertRaises(ValueError):
            gpu_trace.encode_packet(0x10000)
        with self.assertRaises(ValueError):
            gpu_trace.encode_packet(0, b"", flags=0x10000)

    def test_decode_all_g5_opcodes_12_through_21_roundtrip(self) -> None:
        texel_blob = bytes((i * 7) & 0xFF for i in range(64))
        stream = b"".join(
            [
                gpu_trace.encode_begin_frame(),
                gpu_trace.encode_create_texture(
                    resource_id=100,
                    width=512,
                    height=256,
                    depth_or_layers=1,
                    mip_levels=4,
                    texture_format=gpu_trace.GpuTextureFormat.Bc7RgbaUnorm,
                    usage_flags=0x0002 | 0x0004,  # COPY_DST | TEXTURE_BINDING
                    dimension=gpu_trace.GpuTextureDimension.Dim2D,
                ),
                gpu_trace.encode_write_texture(
                    texture_id=100,
                    width=16,
                    height=16,
                    data=texel_blob,
                    mip_level=1,
                    origin_x=4,
                    origin_y=8,
                    origin_z=0,
                    depth_or_layers=1,
                    bytes_per_row=256,
                    rows_per_image=16,
                    reserved=3,
                ),
                gpu_trace.encode_create_sampler(
                    resource_id=101,
                    address_u=gpu_trace.GpuAddressMode.Repeat,
                    address_v=gpu_trace.GpuAddressMode.MirrorRepeat,
                    address_w=gpu_trace.GpuAddressMode.ClampToEdge,
                    mag_filter=gpu_trace.GpuFilterMode.Linear,
                    min_filter=gpu_trace.GpuFilterMode.Linear,
                    mipmap_filter=gpu_trace.GpuFilterMode.Nearest,
                    max_anisotropy=8,
                ),
                gpu_trace.encode_create_bind_group(
                    resource_id=102,
                    pipeline_id=30,
                    group_index=1,
                    entries=[
                        gpu_trace.BindGroupEntry(
                            binding=0,
                            resource_type=gpu_trace.GpuBindResourceType.Buffer,
                            resource_id=10,
                            offset=64,
                            size=256,
                            reserved=0,
                        ),
                        {
                            "binding": 1,
                            "resource_type": gpu_trace.GpuBindResourceType.TextureView,
                            "resource_id": 100,
                        },
                        gpu_trace.encode_bind_group_entry(
                            binding=2,
                            resource_type=gpu_trace.GpuBindResourceType.Sampler,
                            resource_id=101,
                        ),
                    ],
                ),
                gpu_trace.encode_set_bind_group(group_index=1, bind_group_id=102),
                gpu_trace.encode_set_index_buffer(
                    buffer_id=11,
                    index_format=gpu_trace.GpuIndexFormat.Uint32,
                    offset=128,
                ),
                gpu_trace.encode_draw_indexed(
                    index_count=72,
                    instance_count=3,
                    first_index=6,
                    base_vertex=-16,
                    first_instance=2,
                ),
                gpu_trace.encode_create_compute_pipeline(
                    resource_id=200,
                    cs_shader_id=22,
                    pipeline_flags=0x0005,
                    reserved=9,
                ),
                gpu_trace.encode_set_compute_pipeline(pipeline_id=200),
                gpu_trace.encode_dispatch_compute(workgroup_x=16, workgroup_y=8, workgroup_z=2),
                gpu_trace.encode_present(),
            ]
        )

        trace = gpu_trace.decode_stream(stream, strict=True)
        self.assertTrue(trace.ok)
        self.assertEqual(trace.diagnostics, ())
        self.assertEqual(trace.consumed_bytes, len(stream))
        self.assertEqual(trace.packet_count, 12)
        self.assertEqual(trace.unknown_opcode_count, 0)

        # 12: CreateTexture
        p1 = trace.packets[1]
        self.assertEqual(p1.opcode, 12)
        self.assertEqual(p1.name, "CreateTexture")
        self.assertEqual(p1.payload_bytes, 32)
        self.assertEqual(p1.fields["id"], 100)
        self.assertEqual(p1.fields["width"], 512)
        self.assertEqual(p1.fields["height"], 256)
        self.assertEqual(p1.fields["depth_or_layers"], 1)
        self.assertEqual(p1.fields["mip_levels"], 4)
        self.assertEqual(p1.fields["format"], 5)
        self.assertEqual(p1.fields["format_name"], "Bc7RgbaUnorm")
        self.assertEqual(p1.fields["format_webgpu"], "bc7-rgba-unorm")
        self.assertEqual(p1.fields["usage_flags"], 0x0006)
        self.assertEqual(p1.fields["usage_names"], ["COPY_DST", "TEXTURE_BINDING"])
        self.assertEqual(p1.fields["dimension"], 1)
        self.assertEqual(p1.fields["dimension_name"], "Dim2D")
        self.assertEqual(p1.fields["dimension_webgpu"], "2d")

        # 13: WriteTexture
        p2 = trace.packets[2]
        self.assertEqual(p2.opcode, 13)
        self.assertEqual(p2.name, "WriteTexture")
        self.assertEqual(p2.payload_bytes, 48 + len(texel_blob))
        self.assertEqual(p2.fields["texture_id"], 100)
        self.assertEqual(p2.fields["mip_level"], 1)
        self.assertEqual(p2.fields["origin_x"], 4)
        self.assertEqual(p2.fields["origin_y"], 8)
        self.assertEqual(p2.fields["origin_z"], 0)
        self.assertEqual(p2.fields["width"], 16)
        self.assertEqual(p2.fields["height"], 16)
        self.assertEqual(p2.fields["depth_or_layers"], 1)
        self.assertEqual(p2.fields["bytes_per_row"], 256)
        self.assertEqual(p2.fields["rows_per_image"], 16)
        self.assertEqual(p2.fields["data_len"], 64)
        self.assertEqual(p2.fields["reserved"], 3)
        self.assertEqual(p2.fields["crc32"], gpu_trace.crc32_hex(texel_blob))
        self.assertEqual(p2.fields["preview_bytes"], gpu_trace.DEFAULT_PREVIEW_BYTES)
        self.assertEqual(
            p2.fields["preview_hex"],
            texel_blob[: gpu_trace.DEFAULT_PREVIEW_BYTES].hex(),
        )
        self.assertTrue(p2.fields["truncated_preview"])

        # 14: CreateSampler
        p3 = trace.packets[3]
        self.assertEqual(p3.opcode, 14)
        self.assertEqual(p3.name, "CreateSampler")
        self.assertEqual(p3.payload_bytes, 32)
        self.assertEqual(p3.fields["id"], 101)
        self.assertEqual(p3.fields["address_u_name"], "Repeat")
        self.assertEqual(p3.fields["address_u_webgpu"], "repeat")
        self.assertEqual(p3.fields["address_v_name"], "MirrorRepeat")
        self.assertEqual(p3.fields["address_v_webgpu"], "mirror-repeat")
        self.assertEqual(p3.fields["address_w_name"], "ClampToEdge")
        self.assertEqual(p3.fields["address_w_webgpu"], "clamp-to-edge")
        self.assertEqual(p3.fields["mag_filter_name"], "Linear")
        self.assertEqual(p3.fields["min_filter_name"], "Linear")
        self.assertEqual(p3.fields["mipmap_filter_name"], "Nearest")
        self.assertEqual(p3.fields["max_anisotropy"], 8)

        # 15: CreateBindGroup (multi-entry)
        p4 = trace.packets[4]
        self.assertEqual(p4.opcode, 15)
        self.assertEqual(p4.name, "CreateBindGroup")
        self.assertEqual(p4.payload_bytes, 16 + 3 * 32)
        self.assertEqual(p4.fields["id"], 102)
        self.assertEqual(p4.fields["pipeline_id"], 30)
        self.assertEqual(p4.fields["group_index"], 1)
        self.assertEqual(p4.fields["entry_count"], 3)
        self.assertEqual(len(p4.fields["entries"]), 3)
        self.assertEqual(p4.fields["entries"][0]["binding"], 0)
        self.assertEqual(p4.fields["entries"][0]["resource_type_name"], "Buffer")
        self.assertEqual(p4.fields["entries"][0]["resource_id"], 10)
        self.assertEqual(p4.fields["entries"][0]["offset"], 64)
        self.assertEqual(p4.fields["entries"][0]["size"], 256)
        self.assertEqual(p4.fields["entries"][1]["binding"], 1)
        self.assertEqual(p4.fields["entries"][1]["resource_type_name"], "TextureView")
        self.assertEqual(p4.fields["entries"][1]["resource_id"], 100)
        self.assertEqual(p4.fields["entries"][2]["binding"], 2)
        self.assertEqual(p4.fields["entries"][2]["resource_type_name"], "Sampler")
        self.assertEqual(p4.fields["entries"][2]["resource_id"], 101)

        # 16: SetBindGroup
        p5 = trace.packets[5]
        self.assertEqual(p5.opcode, 16)
        self.assertEqual(p5.name, "SetBindGroup")
        self.assertEqual(p5.payload_bytes, 8)
        self.assertEqual(p5.fields["group_index"], 1)
        self.assertEqual(p5.fields["bind_group_id"], 102)

        # 17: SetIndexBuffer
        p6 = trace.packets[6]
        self.assertEqual(p6.opcode, 17)
        self.assertEqual(p6.name, "SetIndexBuffer")
        self.assertEqual(p6.payload_bytes, 16)
        self.assertEqual(p6.fields["buffer_id"], 11)
        self.assertEqual(p6.fields["index_format"], 1)
        self.assertEqual(p6.fields["index_format_name"], "Uint32")
        self.assertEqual(p6.fields["index_format_webgpu"], "uint32")
        self.assertEqual(p6.fields["offset"], 128)

        # 18: DrawIndexed (including negative signed int32 base_vertex)
        p7 = trace.packets[7]
        self.assertEqual(p7.opcode, 18)
        self.assertEqual(p7.name, "DrawIndexed")
        self.assertEqual(p7.payload_bytes, 20)
        self.assertEqual(p7.fields["index_count"], 72)
        self.assertEqual(p7.fields["instance_count"], 3)
        self.assertEqual(p7.fields["first_index"], 6)
        self.assertEqual(p7.fields["base_vertex"], -16)
        self.assertEqual(p7.fields["first_instance"], 2)

        # 19: CreateComputePipeline
        p8 = trace.packets[8]
        self.assertEqual(p8.opcode, 19)
        self.assertEqual(p8.name, "CreateComputePipeline")
        self.assertEqual(p8.payload_bytes, 16)
        self.assertEqual(p8.fields["id"], 200)
        self.assertEqual(p8.fields["cs_shader_id"], 22)
        self.assertEqual(p8.fields["flags"], 0x0005)
        self.assertEqual(p8.fields["reserved"], 9)

        # 20: SetComputePipeline
        p9 = trace.packets[9]
        self.assertEqual(p9.opcode, 20)
        self.assertEqual(p9.name, "SetComputePipeline")
        self.assertEqual(p9.payload_bytes, 4)
        self.assertEqual(p9.fields["pipeline_id"], 200)

        # 21: DispatchCompute
        p10 = trace.packets[10]
        self.assertEqual(p10.opcode, 21)
        self.assertEqual(p10.name, "DispatchCompute")
        self.assertEqual(p10.payload_bytes, 12)
        self.assertEqual(p10.fields["workgroup_x"], 16)
        self.assertEqual(p10.fields["workgroup_y"], 8)
        self.assertEqual(p10.fields["workgroup_z"], 2)

        # Text formatting includes readable G5 summaries
        text = gpu_trace.format_trace_text(trace)
        self.assertIn("CreateTexture", text)
        self.assertIn("fmt=bc7-rgba-unorm", text)
        self.assertIn("COPY_DST|TEXTURE_BINDING", text)
        self.assertIn("CreateBindGroup", text)
        self.assertIn("@0:Buffer(#10)[64:256]", text)
        self.assertIn("@1:TextureView(#100)", text)
        self.assertIn("@2:Sampler(#101)", text)
        self.assertIn("base_vertex=-16", text)
        self.assertIn("workgroups=(16, 8, 2)", text)

    def test_draw_indexed_negative_base_vertex_extremes(self) -> None:
        stream = b"".join(
            [
                gpu_trace.encode_draw_indexed(3, base_vertex=-1),
                gpu_trace.encode_draw_indexed(6, base_vertex=-2147483648),
                gpu_trace.encode_draw_indexed(9, base_vertex=2147483647),
            ]
        )
        trace = gpu_trace.decode_stream(stream, strict=True)
        self.assertTrue(trace.ok)
        self.assertEqual(trace.packets[0].fields["base_vertex"], -1)
        self.assertEqual(trace.packets[1].fields["base_vertex"], -2147483648)
        self.assertEqual(trace.packets[2].fields["base_vertex"], 2147483647)

    def test_g5_enums_and_invalid_enum_diagnostics(self) -> None:
        # All known texture usage flags
        names, unknown = gpu_trace.decode_texture_usage(0x001F)
        self.assertEqual(unknown, 0)
        self.assertEqual(
            names,
            [
                "COPY_SRC",
                "COPY_DST",
                "TEXTURE_BINDING",
                "STORAGE_BINDING",
                "RENDER_ATTACHMENT",
            ],
        )

        # Unknown texture format, usage bit, dimension, sampler modes, bind resource type, index fmt
        stream = b"".join(
            [
                gpu_trace.encode_create_texture(
                    resource_id=1,
                    width=32,
                    height=32,
                    texture_format=99,
                    usage_flags=0x0044,  # TEXTURE_BINDING | unknown 0x40
                    dimension=7,
                ),
                gpu_trace.encode_create_sampler(
                    resource_id=2,
                    address_u=9,
                    mag_filter=8,
                ),
                gpu_trace.encode_create_bind_group(
                    resource_id=3,
                    pipeline_id=1,
                    group_index=0,
                    entries=[gpu_trace.BindGroupEntry(binding=0, resource_type=42, resource_id=1)],
                ),
                gpu_trace.encode_set_index_buffer(buffer_id=4, index_format=5, offset=0),
            ]
        )

        lenient = gpu_trace.decode_stream(stream, strict=False)
        self.assertTrue(lenient.ok)
        self.assertEqual(lenient.packets[0].fields["format_name"], "Unknown(99)")
        self.assertIsNone(lenient.packets[0].fields["format_webgpu"])
        self.assertEqual(
            lenient.packets[0].fields["usage_names"],
            ["TEXTURE_BINDING", "UNKNOWN(0x40)"],
        )
        self.assertEqual(lenient.packets[0].fields["dimension_name"], "Unknown(7)")
        self.assertEqual(lenient.packets[1].fields["address_u_name"], "Unknown(9)")
        self.assertEqual(lenient.packets[1].fields["mag_filter_name"], "Unknown(8)")
        self.assertEqual(
            lenient.packets[2].fields["entries"][0]["resource_type_name"],
            "Unknown(42)",
        )
        self.assertEqual(lenient.packets[3].fields["index_format_name"], "Unknown(5)")
        self.assertIsNone(lenient.packets[3].fields["index_format_webgpu"])
        self.assertEqual(
            {d.code for d in lenient.diagnostics},
            {
                "unknown_texture_format",
                "unknown_texture_usage_bits",
                "unknown_texture_dimension",
                "unknown_address_mode",
                "unknown_filter_mode",
                "unknown_bind_resource_type",
                "unknown_index_format",
            },
        )

        strict = gpu_trace.decode_stream(stream, strict=True)
        self.assertFalse(strict.ok)
        self.assertTrue(all(d.severity == "error" for d in strict.diagnostics))

    def test_write_texture_and_bind_group_length_mismatch_diagnostics(self) -> None:
        # 1. WriteTexture header too short (< 48 bytes)
        wt_short_hdr = gpu_trace.encode_packet(gpu_trace.GpuOpcode.WriteTexture, b"\x00" * 24)
        t_wt_hdr = gpu_trace.decode_stream(wt_short_hdr)
        self.assertFalse(t_wt_hdr.ok)
        self.assertFalse(t_wt_hdr.packets[0].valid)
        self.assertEqual(t_wt_hdr.diagnostics[0].code, "payload_too_short")

        # 2. WriteTexture truncated texel bytes (claims 64 bytes, only 10 follow)
        wt_head = gpu_trace.WRITE_TEXTURE_STRUCT.pack(7, 0, 0, 0, 0, 4, 4, 1, 16, 4, 64, 0)
        wt_trunc = gpu_trace.encode_packet(gpu_trace.GpuOpcode.WriteTexture, wt_head + b"\xaa" * 10)
        t_wt_trunc = gpu_trace.decode_stream(wt_trunc)
        self.assertFalse(t_wt_trunc.ok)
        self.assertFalse(t_wt_trunc.packets[0].valid)
        self.assertEqual(t_wt_trunc.packets[0].fields["data_len"], 64)
        self.assertEqual(t_wt_trunc.packets[0].fields["available_bytes"], 10)
        self.assertEqual(t_wt_trunc.diagnostics[0].code, "variable_payload_truncated")

        # 3. WriteTexture trailing extra bytes (claims 4 bytes, 8 provided)
        wt_head_small = gpu_trace.WRITE_TEXTURE_STRUCT.pack(7, 0, 0, 0, 0, 1, 1, 1, 4, 1, 4, 0)
        wt_extra = gpu_trace.encode_packet(
            gpu_trace.GpuOpcode.WriteTexture, wt_head_small + b"\x01\x02\x03\x04\xff\xff\xff\xff"
        )
        t_wt_extra = gpu_trace.decode_stream(wt_extra, strict=False)
        self.assertTrue(t_wt_extra.ok)
        self.assertTrue(t_wt_extra.has_warnings)
        self.assertEqual(t_wt_extra.packets[0].fields["trailing_bytes"], 4)
        self.assertEqual(t_wt_extra.packets[0].fields["crc32"], gpu_trace.crc32_hex(b"\x01\x02\x03\x04"))
        self.assertEqual(t_wt_extra.diagnostics[0].code, "payload_size_mismatch")
        self.assertFalse(gpu_trace.decode_stream(wt_extra, strict=True).ok)

        # 4. CreateBindGroup truncated entries (claims 2 entries = 64B, only 40B follow -> 1 complete)
        entry0 = gpu_trace.encode_bind_group_entry(
            0, gpu_trace.GpuBindResourceType.Buffer, 55, offset=0, size=128
        )
        bg_trunc = gpu_trace.encode_packet(
            gpu_trace.GpuOpcode.CreateBindGroup,
            gpu_trace.CREATE_BIND_GROUP_STRUCT.pack(9, 1, 0, 2) + entry0 + b"\x00" * 8,
        )
        t_bg_trunc = gpu_trace.decode_stream(bg_trunc)
        self.assertFalse(t_bg_trunc.ok)
        self.assertFalse(t_bg_trunc.packets[0].valid)
        self.assertEqual(t_bg_trunc.packets[0].fields["entry_count"], 2)
        self.assertEqual(len(t_bg_trunc.packets[0].fields["entries"]), 1)
        self.assertEqual(t_bg_trunc.packets[0].fields["available_bytes"], 40)
        self.assertEqual(t_bg_trunc.diagnostics[0].code, "variable_payload_truncated")

        # 5. CreateBindGroup trailing extra bytes
        bg_extra = gpu_trace.encode_packet(
            gpu_trace.GpuOpcode.CreateBindGroup,
            gpu_trace.CREATE_BIND_GROUP_STRUCT.pack(9, 1, 0, 1) + entry0 + b"\xbe\xef",
        )
        t_bg_extra = gpu_trace.decode_stream(bg_extra, strict=False)
        self.assertTrue(t_bg_extra.ok)
        self.assertEqual(t_bg_extra.packets[0].fields["trailing_bytes"], 2)
        self.assertEqual(t_bg_extra.diagnostics[0].code, "payload_size_mismatch")
        self.assertFalse(gpu_trace.decode_stream(bg_extra, strict=True).ok)

        # 6. Raw bind group entry length validation on encode
        with self.assertRaises(ValueError):
            gpu_trace.encode_create_bind_group(1, 1, 0, [b"\x00" * 16])

    def test_compute_and_textured_indexed_draw_frame_summaries(self) -> None:
        frame0 = b"".join(
            [
                gpu_trace.encode_begin_frame(),
                gpu_trace.encode_create_texture(10, 64, 64, usage_flags=0x0006),
                gpu_trace.encode_create_texture(11, 64, 64, usage_flags=0x0014),
                gpu_trace.encode_create_sampler(12),
                gpu_trace.encode_create_compute_pipeline(20, cs_shader_id=1),
                gpu_trace.encode_create_bind_group(
                    30,
                    pipeline_id=20,
                    group_index=0,
                    entries=[
                        gpu_trace.BindGroupEntry(0, gpu_trace.GpuBindResourceType.Buffer, 1),
                    ],
                ),
                gpu_trace.encode_create_bind_group(
                    31,
                    pipeline_id=21,
                    group_index=0,
                    entries=[
                        gpu_trace.BindGroupEntry(0, gpu_trace.GpuBindResourceType.TextureView, 10),
                        gpu_trace.BindGroupEntry(1, gpu_trace.GpuBindResourceType.Sampler, 12),
                    ],
                ),
                # Compute pass #1 (two dispatches back-to-back)
                gpu_trace.encode_set_compute_pipeline(20),
                gpu_trace.encode_set_bind_group(0, 30),
                gpu_trace.encode_dispatch_compute(8, 8, 1),
                gpu_trace.encode_dispatch_compute(4, 4, 1),
                # Render pass (1 DrawIndexed + 1 Draw)
                gpu_trace.encode_set_pipeline(21),
                gpu_trace.encode_set_bind_group(0, 31),
                gpu_trace.encode_set_index_buffer(2, gpu_trace.GpuIndexFormat.Uint16, 0),
                gpu_trace.encode_draw_indexed(36, 1, 0, -4, 0),
                gpu_trace.encode_draw(3, 1, 0, 0),
                # Compute pass #2 (after render pass -> increments compute_passes)
                gpu_trace.encode_set_compute_pipeline(20),
                gpu_trace.encode_dispatch_compute(2, 2, 1),
                gpu_trace.encode_present(),
            ]
        )

        trace = gpu_trace.decode_stream(frame0, strict=True)
        self.assertTrue(trace.ok)
        self.assertEqual(trace.frame_count, 1)

        f0 = trace.frames[0]
        self.assertTrue(f0.complete)
        self.assertTrue(f0.has_begin_frame)
        self.assertEqual(f0.draw_calls, 2)
        self.assertEqual(f0.draw_indexed_calls, 1)
        self.assertEqual(f0.compute_dispatches, 3)
        self.assertEqual(f0.compute_passes, 2)
        self.assertEqual(f0.textures_created, 2)
        self.assertEqual(f0.bind_groups_created, 2)

        # Trace-level aggregates match frame totals
        self.assertEqual(trace.draw_calls, 2)
        self.assertEqual(trace.draw_indexed_calls, 1)
        self.assertEqual(trace.compute_dispatches, 3)
        self.assertEqual(trace.compute_passes, 2)
        self.assertEqual(trace.textures_created, 2)
        self.assertEqual(trace.bind_groups_created, 2)

        # JSON output includes all new counters alongside existing ones
        parsed = json.loads(gpu_trace.trace_to_json(trace, stats_only=True))
        self.assertEqual(parsed["draw_calls"], 2)
        self.assertEqual(parsed["draw_indexed_calls"], 1)
        self.assertEqual(parsed["compute_dispatches"], 3)
        self.assertEqual(parsed["compute_passes"], 2)
        self.assertEqual(parsed["textures_created"], 2)
        self.assertEqual(parsed["bind_groups_created"], 2)
        self.assertEqual(parsed["frames"][0]["draw_indexed_calls"], 1)
        self.assertEqual(parsed["frames"][0]["compute_dispatches"], 3)
        self.assertEqual(parsed["frames"][0]["compute_passes"], 2)
        self.assertEqual(parsed["frames"][0]["textures_created"], 2)
        self.assertEqual(parsed["frames"][0]["bind_groups_created"], 2)

        # Text summary prints the new frame metrics
        text = gpu_trace.format_trace_text(trace, stats_only=True)
        self.assertIn("Dispatches: 3", text)
        self.assertIn("indexed=1", text)
        self.assertIn("dispatches=3", text)
        self.assertIn("compute_passes=2", text)
        self.assertIn("textures=2", text)
        self.assertIn("bind_groups=2", text)


def build_render_pipeline_key() -> bytes:
    """376-byte GpuRenderPipelineKey: rgba8unorm + blend, depth24plus-stencil8, 2 vertex buffers."""
    key = bytearray(376)
    struct.pack_into("<QQII", key, 0, 0xAAAA, 0xBBBB, 7, 2)
    # color target 0: rgba8unorm (22), blend add src-alpha / one-minus-src-alpha, mask 0xF
    struct.pack_into("<BBBxBBBxBBBx", key, 24, 22, 1, 0xF, 0, 4, 5, 0, 1, 0)
    # color target 1: null hole (all zero)
    # depth-stencil: depth24plus-stencil8 (47), write, less-equal (4), stencil always/keep/replace
    struct.pack_into(
        "<BBBxBBBBBBBBIIiII", key, 120, 47, 1, 4, 8, 0, 0, 2, 8, 0, 0, 2, 0xFF, 0x0F, -3,
        struct.unpack("<I", struct.pack("<f", 1.5))[0], 0,
    )
    struct.pack_into("<BBBBB3x", key, 152, 1, 2, 1, 2, 0)  # strip, uint32, cw, back
    struct.pack_into("<IIB3x", key, 160, 4, 0xF, 1)
    struct.pack_into("<II", key, 172, 2, 3)
    struct.pack_into("<IBB2x", key, 180, 20, 1, 2)  # slot 0: vertex, 2 attributes
    struct.pack_into("<IBB2x", key, 188, 16, 2, 1)  # slot 1: instance, 1 attribute
    struct.pack_into("<IBBBx", key, 244, 0, 30, 0, 0)  # float32x3 @0
    struct.pack_into("<IBBBx", key, 252, 12, 29, 1, 0)  # float32x2 @1
    struct.pack_into("<IBBBx", key, 260, 0, 31, 2, 1)  # float32x4 @2 (buffer 1)
    return bytes(key)


class TestGpuTraceV2Packets(unittest.TestCase):
    def test_v2_struct_sizes_match_header(self) -> None:
        self.assertEqual(gpu_trace.TEXTURE_VIEW_DESC_STRUCT.size, 16)
        self.assertEqual(gpu_trace.BIND_GROUP_LAYOUT_ENTRY_STRUCT.size, 24)
        self.assertEqual(gpu_trace.CREATE_BIND_GROUP_LAYOUT_STRUCT.size, 24)
        self.assertEqual(gpu_trace.CREATE_PIPELINE_LAYOUT_STRUCT.size, 24)
        self.assertEqual(gpu_trace.CREATE_RENDER_PIPELINE2_SIZE, 392)
        self.assertEqual(gpu_trace.KEY_COLOR_TARGET_STRUCT.size, 12)
        self.assertEqual(gpu_trace.KEY_DEPTH_STENCIL_STRUCT.size, 32)
        self.assertEqual(gpu_trace.KEY_PRIMITIVE_STRUCT.size, 8)
        self.assertEqual(gpu_trace.KEY_MULTISAMPLE_STRUCT.size, 12)
        self.assertEqual(gpu_trace.KEY_VERTEX_BUFFER_STRUCT.size, 8)
        self.assertEqual(gpu_trace.KEY_VERTEX_ATTRIBUTE_STRUCT.size, 8)
        self.assertEqual(gpu_trace.CREATE_COMPUTE_PIPELINE2_STRUCT.size, 32)
        self.assertEqual(gpu_trace.RENDER_PASS_COLOR_ATTACHMENT_STRUCT.size, 40)
        self.assertEqual(gpu_trace.RENDER_PASS_DEPTH_STENCIL_STRUCT.size, 40)
        self.assertEqual(gpu_trace.SET_VIEWPORT_STRUCT.size, 24)
        self.assertEqual(gpu_trace.BIND_GROUP_ENTRY2_STRUCT.size, 48)
        self.assertEqual(gpu_trace.SET_BIND_GROUP2_STRUCT.size, 16)
        self.assertEqual(gpu_trace.CREATE_SAMPLER2_STRUCT.size, 24)
        self.assertEqual(len(gpu_trace.encode_begin_render_pass([])), 8 + 368)
        self.assertEqual(len(gpu_trace.WGPU_TEXTURE_FORMAT_NAMES), 64)
        self.assertEqual(len(gpu_trace.WGPU_VERTEX_FORMAT_NAMES), 42)
        self.assertEqual(
            [op.value for op in gpu_trace.GpuOpcode if op.value >= 22], list(range(22, 36))
        )

    def test_decode_full_v2_frame(self) -> None:
        view_2d_array = gpu_trace.encode_texture_view_desc(40, 3, 0, 1, 2, 4, 1)
        stream = b"".join(
            [
                gpu_trace.encode_create_bind_group_layout(
                    100,
                    [
                        (0, 0x3, 0, 0, 0, 0x1, 0, 256),  # uniform, dynamic, minBindingSize 256
                        (1, 0x2, 2, 0, 2),  # texture float 2d
                        (2, 0x2, 1, 2),  # comparison sampler
                        (3, 0x4, 3, 0, 2, 0, 16),  # storage texture write-only r32float
                    ],
                    content_id=0x472F25790659490F,
                ),
                gpu_trace.encode_create_bind_group_layout(
                    101, [], content_id=0x1234, flags=gpu_trace.MANIFEST_REFERENCE_FLAG
                ),
                gpu_trace.encode_create_pipeline_layout(102, [100, 0, 101], content_id=0x99),
                gpu_trace.encode_create_render_pipeline2(103, 1, 2, 102, build_render_pipeline_key()),
                gpu_trace.encode_create_compute_pipeline2(104, 3, 102, cs_shader=0xC5, root_signature_id=9),
                gpu_trace.encode_create_sampler2(105, (1, 1, 0), (1, 1, 0), compare=4, max_anisotropy=8),
                gpu_trace.encode_create_bind_group2(
                    106,
                    100,
                    [
                        (0, gpu_trace.GpuBindResourceType.Buffer, 50, 256, 64),
                        (1, gpu_trace.GpuBindResourceType.TextureView, 51, 0, 0, view_2d_array),
                        (2, gpu_trace.GpuBindResourceType.Sampler, 105),
                        (3, gpu_trace.GpuBindResourceType.TextureView, 0),
                    ],
                ),
                gpu_trace.encode_begin_frame(),
                gpu_trace.encode_begin_render_pass(
                    [
                        {
                            "texture_id": gpu_trace.SWAPCHAIN_TEXTURE_ID,
                            "load_op": 2,
                            "store_op": 1,
                            "clear_value": (0.25, 0.5, 0.75, 1.0),
                        },
                        None,
                        {"texture_id": 60, "load_op": 1, "store_op": 2},
                    ],
                    {
                        "texture_id": 61,
                        "depth_load_op": 2,
                        "depth_store_op": 1,
                        "depth_clear_value": 0.0,
                        "stencil_read_only": True,
                    },
                ),
                gpu_trace.encode_set_pipeline(103),
                gpu_trace.encode_set_viewport(0.0, 0.0, 1280.0, 720.0, 0.0, 1.0),
                gpu_trace.encode_set_scissor_rect(0, 0, 1280, 720),
                gpu_trace.encode_set_blend_constant(1.0, 0.5, 0.25, 0.0),
                gpu_trace.encode_set_stencil_reference(0x80),
                gpu_trace.encode_set_bind_group2(0, 106, [512]),
                gpu_trace.encode_draw(3),
                gpu_trace.encode_end_render_pass(),
                gpu_trace.encode_set_compute_pipeline(104),
                gpu_trace.encode_dispatch_compute(1),
                gpu_trace.encode_present(),
            ]
        )

        trace = gpu_trace.decode_stream(stream, strict=True)
        self.assertTrue(trace.ok, [d.message for d in trace.diagnostics])
        self.assertEqual(trace.unknown_opcode_count, 0)
        self.assertEqual(trace.bind_groups_created, 1)
        p = trace.packets

        bgl = p[0].fields
        self.assertEqual(bgl["id"], 100)
        self.assertEqual(bgl["content_id"], "0x472f25790659490f")
        self.assertFalse(bgl["manifest_reference"])
        e0, e1, e2, e3 = bgl["entries"]
        self.assertEqual(e0["type"], "buffer")
        self.assertEqual(e0["binding_type"], "uniform")
        self.assertTrue(e0["has_dynamic_offset"])
        self.assertEqual(e0["min_binding_size"], 256)
        self.assertEqual(e0["visibility_names"], ["VERTEX", "FRAGMENT"])
        self.assertEqual((e1["type"], e1["binding_type"], e1["view_dimension"]), ("texture", "float", "2d"))
        self.assertFalse(e1["multisampled"])
        self.assertEqual((e2["type"], e2["binding_type"]), ("sampler", "comparison"))
        self.assertEqual(
            (e3["type"], e3["binding_type"], e3["storage_format"]),
            ("storageTexture", "write-only", "r32float"),
        )
        self.assertTrue(p[1].fields["manifest_reference"])
        self.assertEqual(p[2].fields["bind_group_layout_ids"], [100, 0, 101])
        self.assertEqual(p[2].fields["content_id"], "0x0000000000000099")

        rp = p[3].fields
        self.assertEqual(
            (rp["id"], rp["vs_module_id"], rp["fs_module_id"], rp["pipeline_layout_id"]),
            (103, 1, 2, 102),
        )
        key = rp["key"]
        self.assertEqual(key["vs_shader"], "0x000000000000aaaa")
        self.assertEqual(key["root_signature_id"], 7)
        self.assertEqual(key["color_target_count"], 2)
        self.assertEqual(key["color_targets"][0]["format"], "rgba8unorm")
        self.assertEqual(
            key["color_targets"][0]["blend"]["color"],
            {"operation": "add", "src_factor": "src-alpha", "dst_factor": "one-minus-src-alpha"},
        )
        self.assertIsNone(key["color_targets"][1])
        ds = key["depth_stencil"]
        self.assertEqual(ds["format"], "depth24plus-stencil8")
        self.assertEqual(ds["depth_compare"], "less-equal")
        self.assertEqual(ds["stencil_front"]["pass_op"], "replace")
        self.assertEqual(ds["depth_bias"], -3)
        self.assertEqual(ds["depth_bias_slope_scale"], 1.5)
        self.assertEqual(
            key["primitive"],
            {
                "topology": "triangle-strip",
                "strip_index_format": "uint32",
                "front_face": "cw",
                "cull_mode": "back",
                "unclipped_depth": False,
            },
        )
        self.assertEqual(key["multisample"]["count"], 4)
        vb0, vb1 = key["vertex_buffers"]
        self.assertEqual((vb0["array_stride"], vb0["step_mode"]), (20, "vertex"))
        self.assertEqual([a["format"] for a in vb0["attributes"]], ["float32x3", "float32x2"])
        self.assertEqual((vb1["step_mode"], vb1["attributes"][0]["shader_location"]), ("instance", 2))
        self.assertEqual(key["key_fnv1a64"], f"0x{gpu_trace.fnv1a_64(build_render_pipeline_key()):016x}")

        cp = p[4].fields
        self.assertEqual((cp["cs_module_id"], cp["pipeline_layout_id"], cp["root_signature_id"]), (3, 102, 9))
        sampler = p[5].fields
        self.assertEqual((sampler["address_u"], sampler["address_w"]), ("repeat", "clamp-to-edge"))
        self.assertEqual((sampler["compare"], sampler["max_anisotropy"]), ("less-equal", 8))
        bg = p[6].fields
        self.assertEqual(bg["layout_id"], 100)
        self.assertEqual((bg["entries"][0]["offset"], bg["entries"][0]["size"]), (256, 64))
        self.assertEqual(
            bg["entries"][1]["view"],
            {
                "format": "rgba16float",
                "dimension": "2d-array",
                "aspect": "all",
                "base_mip_level": 1,
                "mip_level_count": 2,
                "base_array_layer": 4,
                "array_layer_count": 1,
            },
        )
        self.assertTrue(bg["entries"][3]["null_binding"])

        rpass = p[8].fields
        self.assertEqual(rpass["color_count"], 3)
        c0, c1, c2 = rpass["color_attachments"]
        self.assertTrue(c0["swapchain"])
        self.assertEqual((c0["load_op"], c0["store_op"], c0["clear_value"]), ("clear", "store", [0.25, 0.5, 0.75, 1.0]))
        self.assertIsNone(c1)
        self.assertEqual((c2["texture_id"], c2["load_op"], c2["store_op"]), (60, "load", "discard"))
        dsa = rpass["depth_stencil"]
        self.assertEqual((dsa["texture_id"], dsa["depth_load_op"], dsa["depth_clear_value"]), (61, "clear", 0.0))
        self.assertEqual((dsa["stencil_load_op"], dsa["stencil_read_only"]), ("undefined", True))

        self.assertEqual(p[10].fields["width"], 1280.0)
        self.assertEqual(p[11].fields, {"x": 0, "y": 0, "width": 1280, "height": 720})
        self.assertEqual(p[12].fields["g"], 0.5)
        self.assertEqual(p[13].fields["reference"], 0x80)
        self.assertEqual(p[14].fields["dynamic_offsets"], [512])
        self.assertEqual(p[16].name, "EndRenderPass")

        frame = trace.frames[1]
        self.assertTrue(frame.complete)
        self.assertEqual(frame.draw_calls, 1)
        self.assertEqual(frame.compute_passes, 1)
        self.assertEqual(frame.begin_render_passes, 1)
        self.assertEqual(frame.end_render_passes, 1)
        self.assertEqual(frame.viewport_changes, 1)
        self.assertEqual(frame.scissor_changes, 1)
        self.assertEqual(frame.blend_constant_changes, 1)
        self.assertEqual(frame.stencil_ref_changes, 1)
        self.assertEqual(frame.color_load_ops, {"clear": 1, "load": 1})
        self.assertEqual(frame.color_store_ops, {"store": 1, "discard": 1})
        self.assertEqual(frame.depth_load_ops, {"clear": 1})
        self.assertEqual(frame.depth_store_ops, {"store": 1})

        # Trace-level ABI v2 aggregates (setup frame + present frame).
        self.assertEqual(trace.begin_render_passes, 1)
        self.assertEqual(trace.end_render_passes, 1)
        self.assertEqual(trace.bind_group_layouts_created, 2)
        self.assertEqual(trace.pipeline_layouts_created, 1)
        self.assertEqual(trace.render_pipelines2_created, 1)
        self.assertEqual(trace.compute_pipelines2_created, 1)
        self.assertEqual(trace.viewport_changes, 1)
        self.assertEqual(trace.scissor_changes, 1)
        self.assertEqual(trace.blend_constant_changes, 1)
        self.assertEqual(trace.stencil_ref_changes, 1)
        self.assertEqual(trace.malformed_v2_packets, 0)
        self.assertEqual(trace.truncated_v2_diagnostics, 0)
        self.assertEqual(
            set(p.opcode for p in trace.packets if 22 <= p.opcode <= 34),
            set(range(22, 35)),
        )

        text = gpu_trace.format_trace_text(trace)
        self.assertIn("BeginRenderPass", text)
        self.assertIn("c0=swapchain clear(0.25,0.5,0.75,1)/store", text)
        self.assertIn("c1=null", text)
        self.assertIn("ro=S", text)
        self.assertIn("targets=[rgba8unorm,null] depth=depth24plus-stencil8 samples=4", text)
        self.assertIn("@1:TextureView(#51)<rgba16float,2d-array,mip=1+2,layer=4+1>", text)
        self.assertIn("@3:TextureView(null)", text)
        self.assertIn("manifest-ref", text)
        self.assertIn("dynamic_offsets=[512]", text)
        self.assertIn("compare=less-equal", text)
        self.assertIn("RenderPasses: begin=1 end=1", text)
        self.assertIn("Layouts: bgl=2 pll=1", text)
        self.assertIn("Dynamic: vp=1 scissor=1 blend=1 stencil=1", text)
        self.assertIn("ColorLoadOps: clear=1, load=1", text)
        parsed = json.loads(gpu_trace.trace_to_json(trace, stats_only=True))
        self.assertEqual(parsed["begin_render_passes"], 1)
        self.assertEqual(parsed["render_pipelines2_created"], 1)
        self.assertEqual(parsed["viewport_changes"], 1)
        self.assertEqual(parsed["malformed_v2_packets"], 0)

    def test_v2_malformed_packets(self) -> None:
        # Truncated layout tail, short BeginRenderPass, unknown enum codes, oversize color_count.
        bad_layout = gpu_trace.encode_packet(
            gpu_trace.GpuOpcode.CreateBindGroupLayout,
            gpu_trace.CREATE_BIND_GROUP_LAYOUT_STRUCT.pack(1, 2, 0, 0, 0) + bytes(24),
        )
        short_pass = gpu_trace.encode_packet(gpu_trace.GpuOpcode.BeginRenderPass, bytes(100))
        weird_pass = bytearray(gpu_trace.encode_begin_render_pass([{"texture_id": 5, "load_op": 9, "store_op": 1}]))
        oversize = bytearray(gpu_trace.encode_begin_render_pass([]))
        struct.pack_into("<I", oversize, 8, 9)
        trace = gpu_trace.decode_stream(bad_layout + short_pass + bytes(weird_pass) + bytes(oversize))
        codes = [d.code for d in trace.diagnostics]
        self.assertIn("variable_payload_truncated", codes)
        self.assertIn("payload_too_short", codes)
        self.assertIn("unknown_enum_code", codes)
        self.assertIn("invalid_count", codes)
        self.assertFalse(trace.packets[0].valid)
        self.assertEqual(len(trace.packets[0].fields["entries"]), 1)
        self.assertFalse(trace.packets[1].valid)
        self.assertEqual(trace.packets[2].fields["color_attachments"][0]["load_op"], "Unknown(9)")
        self.assertFalse(trace.packets[3].valid)

        # Extra trailing bytes on a fixed-size v2 packet: warning (error under --strict).
        padded = gpu_trace.encode_packet(gpu_trace.GpuOpcode.SetStencilReference, struct.pack("<II", 1, 0))
        lenient = gpu_trace.decode_stream(padded)
        self.assertTrue(lenient.ok)
        self.assertEqual(lenient.packets[0].fields["trailing_bytes"], 4)
        self.assertFalse(gpu_trace.decode_stream(padded, strict=True).ok)

        self.assertGreaterEqual(trace.malformed_v2_packets, 2)
        self.assertGreaterEqual(trace.truncated_v2_diagnostics, 2)
        self.assertEqual(trace.frames[0].malformed_v2_packets, trace.malformed_v2_packets)

        # Truncated mid-stream v2 header/payload (BeginRenderPass claims 368B, only 8 follow).
        trunc_hdr = struct.pack("<HHI", gpu_trace.GpuOpcode.BeginRenderPass, 0, 368) + b"\x00" * 8
        trunc_trace = gpu_trace.decode_stream(trunc_hdr)
        self.assertFalse(trunc_trace.ok)
        self.assertEqual(trunc_trace.packet_count, 0)
        self.assertEqual(trunc_trace.diagnostics[0].code, "truncated_payload")
        self.assertEqual(trunc_trace.truncated_v2_diagnostics, 1)

    def test_state_bridge_fixture_disassembles_strictly(self) -> None:
        # Packet stream recorded by runtime/gpu/state_bridge (tests/test_state_bridge.cpp).
        fixture = REPO_ROOT / "tests" / "fixtures" / "state_bridge_frame.hex"
        if not fixture.is_file():
            self.skipTest("state_bridge_frame.hex fixture not present")
        hex_text = fixture.read_text(encoding="utf-8")
        trace = gpu_trace.decode_stream(bytes.fromhex("".join(hex_text.split())), strict=True)
        self.assertTrue(trace.ok, [d.message for d in trace.diagnostics])
        self.assertEqual(trace.packet_count, 27)
        self.assertEqual(trace.draw_calls, 2)
        self.assertEqual(trace.opcode_histogram["BeginRenderPass"], 2)
        self.assertEqual(trace.opcode_histogram["EndRenderPass"], 2)
        self.assertEqual(trace.begin_render_passes, 2)
        self.assertEqual(trace.end_render_passes, 2)
        self.assertEqual(trace.render_pipelines2_created, 2)
        self.assertGreaterEqual(trace.viewport_changes, 1)
        self.assertGreaterEqual(trace.scissor_changes, 1)
        self.assertEqual(trace.malformed_v2_packets, 0)
        self.assertEqual(trace.truncated_v2_diagnostics, 0)
        first_pass = trace.packets[0].fields
        self.assertTrue(first_pass["color_attachments"][0]["swapchain"])
        self.assertEqual(first_pass["depth_stencil"]["view"]["format"], "depth32float")
        pipelines = [p for p in trace.packets if p.name == "CreateRenderPipeline2"]
        self.assertEqual([p.fields["key"]["depth_stencil"] is None for p in pipelines], [False, True])
        self.assertEqual(pipelines[0].fields["key"]["color_targets"][0]["format"], "bgra8unorm")
        set_groups = [p.fields["dynamic_offsets"] for p in trace.packets if p.name == "SetBindGroup2"]
        self.assertEqual(set_groups, [[0], [256]])
        stats = json.loads(gpu_trace.trace_to_json(trace, stats_only=True))
        self.assertEqual(stats["begin_render_passes"], 2)
        self.assertIn("clear", stats["color_load_ops"] or stats["depth_load_ops"])

    def test_every_v2_opcode_22_through_34_encodes_and_decodes(self) -> None:
        """Synthetic coverage for each ABI v2 opcode payload size and typed fields."""
        key = build_render_pipeline_key()
        packets = {
            22: gpu_trace.encode_create_bind_group_layout(1, [(0, 0x1, 0, 0)], content_id=7),
            23: gpu_trace.encode_create_pipeline_layout(2, [1, 0], content_id=8),
            24: gpu_trace.encode_create_render_pipeline2(3, 10, 11, 2, key),
            25: gpu_trace.encode_create_compute_pipeline2(4, 12, 2, cs_shader=9),
            26: gpu_trace.encode_begin_render_pass(
                [{"texture_id": 1, "load_op": 1, "store_op": 1}],
                {"texture_id": 2, "depth_load_op": 2, "depth_store_op": 1},
            ),
            27: gpu_trace.encode_end_render_pass(),
            28: gpu_trace.encode_set_viewport(0, 0, 64, 64),
            29: gpu_trace.encode_set_scissor_rect(1, 2, 3, 4),
            30: gpu_trace.encode_set_blend_constant(0.1, 0.2, 0.3, 0.4),
            31: gpu_trace.encode_set_stencil_reference(7),
            32: gpu_trace.encode_create_bind_group2(
                5, 1, [(0, gpu_trace.GpuBindResourceType.Buffer, 9, 0, 16)]
            ),
            33: gpu_trace.encode_set_bind_group2(0, 5, [32]),
            34: gpu_trace.encode_create_sampler2(6),
        }
        self.assertEqual(sorted(packets), list(range(22, 35)))
        stream = b"".join(packets[op] for op in range(22, 35))
        trace = gpu_trace.decode_stream(stream, strict=True)
        self.assertTrue(trace.ok, [d.message for d in trace.diagnostics])
        self.assertEqual([p.opcode for p in trace.packets], list(range(22, 35)))
        self.assertEqual(trace.begin_render_passes, 1)
        self.assertEqual(trace.end_render_passes, 1)
        self.assertEqual(trace.bind_group_layouts_created, 1)
        self.assertEqual(trace.pipeline_layouts_created, 1)
        self.assertEqual(trace.render_pipelines2_created, 1)
        self.assertEqual(trace.compute_pipelines2_created, 1)
        self.assertEqual(trace.viewport_changes, 1)
        self.assertEqual(trace.scissor_changes, 1)
        self.assertEqual(trace.blend_constant_changes, 1)
        self.assertEqual(trace.stencil_ref_changes, 1)
        self.assertEqual(trace.packets[11].fields["dynamic_offsets"], [32])
        self.assertEqual(trace.color_load_ops, {"load": 1})
        self.assertEqual(trace.depth_load_ops, {"clear": 1})

    def test_encoder_rejects_wrong_key_size(self) -> None:
        with self.assertRaises(ValueError):
            gpu_trace.encode_create_render_pipeline2(1, 2, 3, 0, bytes(375))


if __name__ == "__main__":
    unittest.main()
