#!/usr/bin/env python3
"""Binary GPU command stream disassembler and frame trace inspector for playgta6web.

Matches the packet layout in `runtime/abi/playgta6_abi.h` and
`docs/gpu_commands.md` (v1 opcodes 0..21, v2 opcodes 22..34, plus forward-compatible
handling of unknown/future opcodes >= 35).

Pure Python 3.12 stdlib only (zero external dependencies). Usable both as an
importable library and as a CLI:

    python tools/gpu_trace.py <trace.bin> [--json] [--stats] [--strict]
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import zlib
from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Sequence

ABI_VERSION = 1

HEADER_STRUCT = struct.Struct("<HHI")
HEADER_SIZE = HEADER_STRUCT.size  # 8 bytes

CLEAR_COLOR_STRUCT = struct.Struct("<ffff")  # 16 bytes
CREATE_BUFFER_STRUCT = struct.Struct("<IIQ")  # 16 bytes
WRITE_BUFFER_STRUCT = struct.Struct("<IIII")  # 16 bytes + data_len
CREATE_SHADER_MODULE_STRUCT = struct.Struct("<IIII")  # 16 bytes + code_len
DRAW_STRUCT = struct.Struct("<IIII")  # 16 bytes
CREATE_RENDER_PIPELINE_STRUCT = struct.Struct("<IIIIII")  # 24 bytes
SET_PIPELINE_STRUCT = struct.Struct("<I")  # 4 bytes
SET_VERTEX_BUFFER_STRUCT = struct.Struct("<IIQ")  # 16 bytes
DESTROY_RESOURCE_STRUCT = struct.Struct("<I")  # 4 bytes
CREATE_TEXTURE_STRUCT = struct.Struct("<IIIIIIII")  # 32 bytes
WRITE_TEXTURE_STRUCT = struct.Struct("<IIIIIIIIIIII")  # 48 bytes + data_len
CREATE_SAMPLER_STRUCT = struct.Struct("<IIIIIIII")  # 32 bytes
BIND_GROUP_ENTRY_STRUCT = struct.Struct("<IIIIQQ")  # 32 bytes
CREATE_BIND_GROUP_STRUCT = struct.Struct("<IIII")  # 16 bytes + entry_count * 32
SET_BIND_GROUP_STRUCT = struct.Struct("<II")  # 8 bytes
SET_INDEX_BUFFER_STRUCT = struct.Struct("<IIQ")  # 16 bytes
DRAW_INDEXED_STRUCT = struct.Struct("<IIIiI")  # 20 bytes
CREATE_COMPUTE_PIPELINE_STRUCT = struct.Struct("<IIII")  # 16 bytes
SET_COMPUTE_PIPELINE_STRUCT = struct.Struct("<I")  # 4 bytes
DISPATCH_COMPUTE_STRUCT = struct.Struct("<III")  # 12 bytes

# Packet ABI v2 (opcodes 22..34).
TEXTURE_VIEW_DESC_STRUCT = struct.Struct("<BBBBHHII")  # 16 bytes
BIND_GROUP_LAYOUT_ENTRY_STRUCT = struct.Struct("<IIBBBBB3xQ")  # 24 bytes
CREATE_BIND_GROUP_LAYOUT_STRUCT = struct.Struct("<IIIIQ")  # 24 bytes + entry_count * 24
CREATE_PIPELINE_LAYOUT_STRUCT = struct.Struct("<IIIIQ")  # 24 bytes + count * 4
RENDER_PIPELINE_KEY_SIZE = 376
CREATE_RENDER_PIPELINE2_HEAD_STRUCT = struct.Struct("<IIII")  # 16 bytes + 376-byte key
CREATE_RENDER_PIPELINE2_SIZE = CREATE_RENDER_PIPELINE2_HEAD_STRUCT.size + RENDER_PIPELINE_KEY_SIZE
KEY_HEAD_STRUCT = struct.Struct("<QQII")  # vs_shader, ps_shader, root_signature_id, color_target_count
KEY_COLOR_TARGET_STRUCT = struct.Struct("<BBBxBBBxBBBx")  # 12 bytes
KEY_DEPTH_STENCIL_STRUCT = struct.Struct("<BBBxBBBBBBBBIIiII")  # 32 bytes
KEY_PRIMITIVE_STRUCT = struct.Struct("<BBBBB3x")  # 8 bytes
KEY_MULTISAMPLE_STRUCT = struct.Struct("<IIB3x")  # 12 bytes
KEY_COUNTS_STRUCT = struct.Struct("<II")
KEY_VERTEX_BUFFER_STRUCT = struct.Struct("<IBB2x")  # 8 bytes
KEY_VERTEX_ATTRIBUTE_STRUCT = struct.Struct("<IBBBx")  # 8 bytes
CREATE_COMPUTE_PIPELINE2_STRUCT = struct.Struct("<IIIIQII")  # 32 bytes
RENDER_PASS_COLOR_ATTACHMENT_STRUCT = struct.Struct("<IBBH16sffff")  # 40 bytes
RENDER_PASS_DEPTH_STENCIL_STRUCT = struct.Struct("<IBBBB16sfIBBHI")  # 40 bytes
BEGIN_RENDER_PASS_HEAD_STRUCT = struct.Struct("<II")
BEGIN_RENDER_PASS_SIZE = 368
SET_VIEWPORT_STRUCT = struct.Struct("<ffffff")  # 24 bytes
SET_SCISSOR_RECT_STRUCT = struct.Struct("<IIII")  # 16 bytes
SET_BLEND_CONSTANT_STRUCT = struct.Struct("<ffff")  # 16 bytes
SET_STENCIL_REFERENCE_STRUCT = struct.Struct("<I")  # 4 bytes
BIND_GROUP_ENTRY2_STRUCT = struct.Struct("<IIIIQQ16s")  # 48 bytes
CREATE_BIND_GROUP2_STRUCT = struct.Struct("<IIII")  # 16 bytes + entry_count * 48
SET_BIND_GROUP2_STRUCT = struct.Struct("<IIII")  # 16 bytes + count * 4
CREATE_SAMPLER2_STRUCT = struct.Struct("<IBBBBBBBxHxxff")  # 24 bytes

SWAPCHAIN_TEXTURE_ID = 0xFFFFFFFF
MANIFEST_REFERENCE_FLAG = 0x1

DEFAULT_PREVIEW_BYTES = 16

FNV1A_64_OFFSET_BASIS = 0xCBF29CE484222325
FNV1A_64_PRIME = 0x100000001B3
FNV1A_64_MASK = 0xFFFFFFFFFFFFFFFF


class GpuOpcode(IntEnum):
    """Frozen GPU command stream opcodes from runtime/abi/playgta6_abi.h."""

    Nop = 0
    BeginFrame = 1
    ClearColor = 2
    CreateBuffer = 3
    WriteBuffer = 4
    CreateShaderModule = 5
    Draw = 6
    Present = 7
    CreateRenderPipeline = 8
    SetPipeline = 9
    SetVertexBuffer = 10
    DestroyResource = 11
    CreateTexture = 12
    WriteTexture = 13
    CreateSampler = 14
    CreateBindGroup = 15
    SetBindGroup = 16
    SetIndexBuffer = 17
    DrawIndexed = 18
    CreateComputePipeline = 19
    SetComputePipeline = 20
    DispatchCompute = 21
    CreateBindGroupLayout = 22
    CreatePipelineLayout = 23
    CreateRenderPipeline2 = 24
    CreateComputePipeline2 = 25
    BeginRenderPass = 26
    EndRenderPass = 27
    SetViewport = 28
    SetScissorRect = 29
    SetBlendConstant = 30
    SetStencilReference = 31
    CreateBindGroup2 = 32
    SetBindGroup2 = 33
    CreateSampler2 = 34


class GpuTextureFormat(IntEnum):
    """Texture format enum from runtime/abi/playgta6_abi.h."""

    Rgba8Unorm = 0
    Bgra8Unorm = 1
    Rgba16Float = 2
    Depth24Plus = 3
    Depth32Float = 4
    Bc7RgbaUnorm = 5


class GpuTextureDimension(IntEnum):
    """Texture dimension enum from runtime/abi/playgta6_abi.h."""

    Dim1D = 0
    Dim2D = 1
    Dim3D = 2


class GpuAddressMode(IntEnum):
    """Sampler address mode enum from runtime/abi/playgta6_abi.h."""

    ClampToEdge = 0
    Repeat = 1
    MirrorRepeat = 2


class GpuFilterMode(IntEnum):
    """Sampler filter mode enum from runtime/abi/playgta6_abi.h."""

    Nearest = 0
    Linear = 1


class GpuBindResourceType(IntEnum):
    """Bind group entry resource type enum from runtime/abi/playgta6_abi.h."""

    Buffer = 0
    TextureView = 1
    Sampler = 2


class GpuIndexFormat(IntEnum):
    """Index buffer format enum from runtime/abi/playgta6_abi.h."""

    Uint16 = 0
    Uint32 = 1


class GpuShaderStage(IntEnum):
    """Shader stage enum from runtime/abi/playgta6_abi.h."""

    Vertex = 0
    Fragment = 1
    Compute = 2


class GpuPrimitiveTopology(IntEnum):
    """Primitive topology enum from runtime/abi/playgta6_abi.h."""

    TriangleList = 0
    TriangleStrip = 1
    LineList = 2
    LineStrip = 3
    PointList = 4


TEXTURE_FORMAT_WEBGPU_NAMES: dict[int, str] = {
    GpuTextureFormat.Rgba8Unorm: "rgba8unorm",
    GpuTextureFormat.Bgra8Unorm: "bgra8unorm",
    GpuTextureFormat.Rgba16Float: "rgba16float",
    GpuTextureFormat.Depth24Plus: "depth24plus",
    GpuTextureFormat.Depth32Float: "depth32float",
    GpuTextureFormat.Bc7RgbaUnorm: "bc7-rgba-unorm",
}

TEXTURE_DIMENSION_WEBGPU_NAMES: dict[int, str] = {
    GpuTextureDimension.Dim1D: "1d",
    GpuTextureDimension.Dim2D: "2d",
    GpuTextureDimension.Dim3D: "3d",
}

ADDRESS_MODE_WEBGPU_NAMES: dict[int, str] = {
    GpuAddressMode.ClampToEdge: "clamp-to-edge",
    GpuAddressMode.Repeat: "repeat",
    GpuAddressMode.MirrorRepeat: "mirror-repeat",
}

FILTER_MODE_WEBGPU_NAMES: dict[int, str] = {
    GpuFilterMode.Nearest: "nearest",
    GpuFilterMode.Linear: "linear",
}

INDEX_FORMAT_WEBGPU_NAMES: dict[int, str] = {
    GpuIndexFormat.Uint16: "uint16",
    GpuIndexFormat.Uint32: "uint32",
}

TOPOLOGY_WEBGPU_NAMES: dict[int, str] = {
    GpuPrimitiveTopology.TriangleList: "triangle-list",
    GpuPrimitiveTopology.TriangleStrip: "triangle-strip",
    GpuPrimitiveTopology.LineList: "line-list",
    GpuPrimitiveTopology.LineStrip: "line-strip",
    GpuPrimitiveTopology.PointList: "point-list",
}

SHADER_STAGE_ENTRY_POINTS: dict[int, str] = {
    GpuShaderStage.Vertex: "vs_main",
    GpuShaderStage.Fragment: "fs_main",
    GpuShaderStage.Compute: "cs_main",
}

BUFFER_USAGE_FLAGS: tuple[tuple[int, str], ...] = (
    (0x0001, "MAP_READ"),
    (0x0002, "MAP_WRITE"),
    (0x0004, "COPY_SRC"),
    (0x0008, "COPY_DST"),
    (0x0010, "INDEX"),
    (0x0020, "VERTEX"),
    (0x0040, "UNIFORM"),
    (0x0080, "STORAGE"),
    (0x0100, "INDIRECT"),
    (0x0200, "QUERY_RESOLVE"),
)

BUFFER_USAGE_KNOWN_MASK = 0
for _bit, _name in BUFFER_USAGE_FLAGS:
    BUFFER_USAGE_KNOWN_MASK |= _bit

TEXTURE_USAGE_FLAGS: tuple[tuple[int, str], ...] = (
    (0x0001, "COPY_SRC"),
    (0x0002, "COPY_DST"),
    (0x0004, "TEXTURE_BINDING"),
    (0x0008, "STORAGE_BINDING"),
    (0x0010, "RENDER_ATTACHMENT"),
)

TEXTURE_USAGE_KNOWN_MASK = 0
for _bit, _name in TEXTURE_USAGE_FLAGS:
    TEXTURE_USAGE_KNOWN_MASK |= _bit

OPCODE_NAMES: dict[int, str] = {op.value: op.name for op in GpuOpcode}

# v2 "WebGPU enum codes" (docs/gpu_commands.md; numbering of pso::Wgpu* in runtime/gpu/pso_cache.h).
WGPU_TEXTURE_FORMAT_NAMES: tuple[str, ...] = (
    "undefined", "r8unorm", "r8snorm", "r8uint", "r8sint", "r16unorm", "r16snorm", "r16uint",
    "r16sint", "r16float", "rg8unorm", "rg8snorm", "rg8uint", "rg8sint", "r32uint", "r32sint",
    "r32float", "rg16unorm", "rg16snorm", "rg16uint", "rg16sint", "rg16float", "rgba8unorm",
    "rgba8unorm-srgb", "rgba8snorm", "rgba8uint", "rgba8sint", "bgra8unorm", "bgra8unorm-srgb",
    "rgb9e5ufloat", "rgb10a2uint", "rgb10a2unorm", "rg11b10ufloat", "rg32uint", "rg32sint",
    "rg32float", "rgba16unorm", "rgba16snorm", "rgba16uint", "rgba16sint", "rgba16float",
    "rgba32uint", "rgba32sint", "rgba32float", "stencil8", "depth16unorm", "depth24plus",
    "depth24plus-stencil8", "depth32float", "depth32float-stencil8", "bc1-rgba-unorm",
    "bc1-rgba-unorm-srgb", "bc2-rgba-unorm", "bc2-rgba-unorm-srgb", "bc3-rgba-unorm",
    "bc3-rgba-unorm-srgb", "bc4-r-unorm", "bc4-r-snorm", "bc5-rg-unorm", "bc5-rg-snorm",
    "bc6h-rgb-ufloat", "bc6h-rgb-float", "bc7-rgba-unorm", "bc7-rgba-unorm-srgb",
)
WGPU_VERTEX_FORMAT_NAMES: tuple[str, ...] = (
    "undefined", "uint8", "uint8x2", "uint8x4", "sint8", "sint8x2", "sint8x4", "unorm8",
    "unorm8x2", "unorm8x4", "snorm8", "snorm8x2", "snorm8x4", "uint16", "uint16x2", "uint16x4",
    "sint16", "sint16x2", "sint16x4", "unorm16", "unorm16x2", "unorm16x4", "snorm16",
    "snorm16x2", "snorm16x4", "float16", "float16x2", "float16x4", "float32", "float32x2",
    "float32x3", "float32x4", "uint32", "uint32x2", "uint32x3", "uint32x4", "sint32",
    "sint32x2", "sint32x3", "sint32x4", "unorm10-10-10-2", "unorm8x4-bgra",
)
WGPU_BLEND_FACTOR_NAMES: tuple[str, ...] = (
    "zero", "one", "src", "one-minus-src", "src-alpha", "one-minus-src-alpha", "dst",
    "one-minus-dst", "dst-alpha", "one-minus-dst-alpha", "src-alpha-saturated", "constant",
    "one-minus-constant", "src1", "one-minus-src1", "src1-alpha", "one-minus-src1-alpha",
)
WGPU_BLEND_OPERATION_NAMES: tuple[str, ...] = ("add", "subtract", "reverse-subtract", "min", "max")
WGPU_COMPARE_FUNCTION_NAMES: tuple[str, ...] = (
    "undefined", "never", "less", "equal", "less-equal", "greater", "not-equal",
    "greater-equal", "always",
)
WGPU_STENCIL_OPERATION_NAMES: tuple[str, ...] = (
    "keep", "zero", "replace", "invert", "increment-clamp", "decrement-clamp",
    "increment-wrap", "decrement-wrap",
)
WGPU_CULL_MODE_NAMES: tuple[str, ...] = ("none", "front", "back")
WGPU_FRONT_FACE_NAMES: tuple[str, ...] = ("ccw", "cw")
WGPU_INDEX_FORMAT_NAMES: tuple[str, ...] = ("undefined", "uint16", "uint32")
WGPU_VERTEX_STEP_MODE_NAMES: tuple[str, ...] = ("unused", "vertex", "instance")
WGPU_LOAD_OP_NAMES: tuple[str, ...] = ("undefined", "load", "clear")
WGPU_STORE_OP_NAMES: tuple[str, ...] = ("undefined", "store", "discard")
WGPU_TEXTURE_ASPECT_NAMES: tuple[str, ...] = ("all", "depth-only", "stencil-only")
WGPU_VIEW_DIMENSION_NAMES: tuple[str, ...] = (
    "undefined", "1d", "2d", "2d-array", "cube", "cube-array", "3d",
)
BINDING_LAYOUT_TYPE_NAMES: tuple[str, ...] = ("buffer", "sampler", "texture", "storageTexture")
BINDING_SUBTYPE_NAMES: dict[int, tuple[str, ...]] = {
    0: ("uniform", "storage", "read-only-storage"),
    1: ("filtering", "non-filtering", "comparison"),
    2: ("float", "unfilterable-float", "depth", "sint", "uint"),
    3: ("write-only", "read-only", "read-write"),
}
SHADER_STAGE_BITS: tuple[tuple[int, str], ...] = ((0x1, "VERTEX"), (0x2, "FRAGMENT"), (0x4, "COMPUTE"))


def fnv1a_64(data: bytes | bytearray | memoryview) -> int:
    """Compute unsigned 64-bit FNV-1a hash over raw bytes."""
    h = FNV1A_64_OFFSET_BASIS
    for byte in data:
        h ^= byte
        h = (h * FNV1A_64_PRIME) & FNV1A_64_MASK
    return h


def crc32_hex(data: bytes | bytearray | memoryview) -> str:
    """Compute unsigned 32-bit CRC32 formatted as lowercase 0x-prefixed hex."""
    return f"0x{zlib.crc32(data) & 0xFFFFFFFF:08x}"


def decode_buffer_usage(usage_flags: int) -> tuple[list[str], int]:
    """Decode GPUBufferUsage bitmask into ordered flag names and unknown bits."""
    names = [name for bit, name in BUFFER_USAGE_FLAGS if usage_flags & bit]
    unknown_bits = usage_flags & ~BUFFER_USAGE_KNOWN_MASK
    if unknown_bits:
        names.append(f"UNKNOWN(0x{unknown_bits:x})")
    return names, unknown_bits


def decode_shader_stage(stage: int) -> tuple[str, bool]:
    """Return human-readable shader stage name and whether it is known."""
    try:
        return GpuShaderStage(stage).name, True
    except ValueError:
        return f"Unknown({stage})", False


def decode_primitive_topology(topology: int) -> tuple[str, str | None, bool]:
    """Return (enum_name, webgpu_name_or_none, is_known) for topology value."""
    try:
        enum_val = GpuPrimitiveTopology(topology)
        return enum_val.name, TOPOLOGY_WEBGPU_NAMES[enum_val], True
    except ValueError:
        return f"Unknown({topology})", None, False


def decode_texture_usage(usage_flags: int) -> tuple[list[str], int]:
    """Decode GPUTextureUsage bitmask into ordered flag names and unknown bits."""
    names = [name for bit, name in TEXTURE_USAGE_FLAGS if usage_flags & bit]
    unknown_bits = usage_flags & ~TEXTURE_USAGE_KNOWN_MASK
    if unknown_bits:
        names.append(f"UNKNOWN(0x{unknown_bits:x})")
    return names, unknown_bits


def decode_texture_format(texture_format: int) -> tuple[str, str | None, bool]:
    """Return (enum_name, webgpu_name_or_none, is_known) for GpuTextureFormat."""
    try:
        enum_val = GpuTextureFormat(texture_format)
        return enum_val.name, TEXTURE_FORMAT_WEBGPU_NAMES[enum_val], True
    except ValueError:
        return f"Unknown({texture_format})", None, False


def decode_texture_dimension(dimension: int) -> tuple[str, str | None, bool]:
    """Return (enum_name, webgpu_name_or_none, is_known) for GpuTextureDimension."""
    try:
        enum_val = GpuTextureDimension(dimension)
        return enum_val.name, TEXTURE_DIMENSION_WEBGPU_NAMES[enum_val], True
    except ValueError:
        return f"Unknown({dimension})", None, False


def decode_address_mode(mode: int) -> tuple[str, str | None, bool]:
    """Return (enum_name, webgpu_name_or_none, is_known) for GpuAddressMode."""
    try:
        enum_val = GpuAddressMode(mode)
        return enum_val.name, ADDRESS_MODE_WEBGPU_NAMES[enum_val], True
    except ValueError:
        return f"Unknown({mode})", None, False


def decode_filter_mode(mode: int) -> tuple[str, str | None, bool]:
    """Return (enum_name, webgpu_name_or_none, is_known) for GpuFilterMode."""
    try:
        enum_val = GpuFilterMode(mode)
        return enum_val.name, FILTER_MODE_WEBGPU_NAMES[enum_val], True
    except ValueError:
        return f"Unknown({mode})", None, False


def decode_bind_resource_type(resource_type: int) -> tuple[str, bool]:
    """Return (enum_name, is_known) for GpuBindResourceType."""
    try:
        return GpuBindResourceType(resource_type).name, True
    except ValueError:
        return f"Unknown({resource_type})", False


def decode_index_format(index_format: int) -> tuple[str, str | None, bool]:
    """Return (enum_name, webgpu_name_or_none, is_known) for GpuIndexFormat."""
    try:
        enum_val = GpuIndexFormat(index_format)
        return enum_val.name, INDEX_FORMAT_WEBGPU_NAMES[enum_val], True
    except ValueError:
        return f"Unknown({index_format})", None, False


def _format_preview(data: bytes | memoryview, preview_bytes: int) -> dict[str, Any]:
    raw = bytes(data)
    limit = max(0, preview_bytes)
    head = raw[:limit]
    return {
        "crc32": crc32_hex(raw),
        "preview_hex": head.hex(),
        "preview_bytes": len(head),
        "truncated_preview": len(raw) > len(head),
    }


@dataclass(frozen=True)
class TraceDiagnostic:
    """Structured warning or error emitted while decoding a GPU command stream."""

    severity: str  # "warning" or "error"
    code: str
    offset: int
    packet_index: int | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "offset": self.offset,
            "packet_index": self.packet_index,
            "message": self.message,
        }


@dataclass(frozen=True)
class DecodedPacket:
    """Single decoded GPU command packet from a binary stream."""

    index: int
    offset: int
    opcode: int
    name: str
    flags: int
    payload_bytes: int
    total_bytes: int
    known: bool
    fields: dict[str, Any]
    valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "offset": self.offset,
            "opcode": self.opcode,
            "name": self.name,
            "flags": self.flags,
            "payload_bytes": self.payload_bytes,
            "total_bytes": self.total_bytes,
            "known": self.known,
            "valid": self.valid,
            "fields": self.fields,
        }


@dataclass(frozen=True)
class BindGroupEntry:
    """Single resource binding entry for CreateBindGroup (32 bytes)."""

    binding: int
    resource_type: int | GpuBindResourceType
    resource_id: int
    offset: int = 0
    size: int = 0
    reserved: int = 0


@dataclass(frozen=True)
class FrameSummary:
    """Summary of packets grouped into a frame (BeginFrame..Present)."""

    frame_index: int
    start_packet_index: int
    end_packet_index: int
    packet_count: int
    draw_calls: int
    bytes: int
    opcode_histogram: dict[str, int]
    complete: bool
    has_begin_frame: bool
    draw_indexed_calls: int = 0
    compute_dispatches: int = 0
    compute_passes: int = 0
    textures_created: int = 0
    bind_groups_created: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "start_packet_index": self.start_packet_index,
            "end_packet_index": self.end_packet_index,
            "packet_count": self.packet_count,
            "draw_calls": self.draw_calls,
            "draw_indexed_calls": self.draw_indexed_calls,
            "compute_dispatches": self.compute_dispatches,
            "compute_passes": self.compute_passes,
            "textures_created": self.textures_created,
            "bind_groups_created": self.bind_groups_created,
            "bytes": self.bytes,
            "opcode_histogram": self.opcode_histogram,
            "complete": self.complete,
            "has_begin_frame": self.has_begin_frame,
        }


@dataclass(frozen=True)
class DecodedTrace:
    """Complete decoded representation of a binary GPU command stream."""

    abi_version: int
    total_bytes: int
    consumed_bytes: int
    packets: tuple[DecodedPacket, ...]
    frames: tuple[FrameSummary, ...]
    opcode_histogram: dict[str, int]
    diagnostics: tuple[TraceDiagnostic, ...]
    strict: bool = False

    @property
    def packet_count(self) -> int:
        return len(self.packets)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def draw_calls(self) -> int:
        return sum(
            1
            for p in self.packets
            if p.opcode in (GpuOpcode.Draw, GpuOpcode.DrawIndexed) and p.valid
        )

    @property
    def draw_indexed_calls(self) -> int:
        return sum(1 for p in self.packets if p.opcode == GpuOpcode.DrawIndexed and p.valid)

    @property
    def compute_dispatches(self) -> int:
        return sum(1 for p in self.packets if p.opcode == GpuOpcode.DispatchCompute and p.valid)

    @property
    def compute_passes(self) -> int:
        return sum(f.compute_passes for f in self.frames)

    @property
    def textures_created(self) -> int:
        return sum(1 for p in self.packets if p.opcode == GpuOpcode.CreateTexture and p.valid)

    @property
    def bind_groups_created(self) -> int:
        return sum(
            1
            for p in self.packets
            if p.opcode in (GpuOpcode.CreateBindGroup, GpuOpcode.CreateBindGroup2) and p.valid
        )

    @property
    def unknown_opcode_count(self) -> int:
        return sum(1 for p in self.packets if not p.known)

    @property
    def has_errors(self) -> bool:
        return any(d.severity == "error" for d in self.diagnostics)

    @property
    def has_warnings(self) -> bool:
        return any(d.severity == "warning" for d in self.diagnostics)

    @property
    def ok(self) -> bool:
        if self.strict:
            return len(self.diagnostics) == 0
        return not self.has_errors

    def to_dict(self, *, stats_only: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "abi_version": self.abi_version,
            "strict": self.strict,
            "ok": self.ok,
            "total_bytes": self.total_bytes,
            "consumed_bytes": self.consumed_bytes,
            "packet_count": self.packet_count,
            "frame_count": self.frame_count,
            "draw_calls": self.draw_calls,
            "draw_indexed_calls": self.draw_indexed_calls,
            "compute_dispatches": self.compute_dispatches,
            "compute_passes": self.compute_passes,
            "textures_created": self.textures_created,
            "bind_groups_created": self.bind_groups_created,
            "unknown_opcode_count": self.unknown_opcode_count,
            "opcode_histogram": dict(self.opcode_histogram),
            "frames": [frame.to_dict() for frame in self.frames],
            "diagnostics": [diag.to_dict() for diag in self.diagnostics],
        }
        if not stats_only:
            result["packets"] = [pkt.to_dict() for pkt in self.packets]
        return result


# ---------------------------------------------------------------------------
# Packet encoding helpers
# ---------------------------------------------------------------------------


def encode_packet(
    opcode: int | GpuOpcode,
    payload: bytes | bytearray | memoryview = b"",
    flags: int = 0,
) -> bytes:
    """Encode a single GpuPacketHeader + payload binary packet."""
    op_int = int(opcode)
    flags_int = int(flags)
    payload_bytes = bytes(payload)
    if not (0 <= op_int <= 0xFFFF):
        raise ValueError(f"opcode out of u16 range: {op_int}")
    if not (0 <= flags_int <= 0xFFFF):
        raise ValueError(f"flags out of u16 range: {flags_int}")
    if len(payload_bytes) > 0xFFFFFFFF:
        raise ValueError(f"payload exceeds u32 length: {len(payload_bytes)}")
    return HEADER_STRUCT.pack(op_int, flags_int, len(payload_bytes)) + payload_bytes


def encode_nop(flags: int = 0) -> bytes:
    return encode_packet(GpuOpcode.Nop, b"", flags=flags)


def encode_begin_frame(flags: int = 0) -> bytes:
    return encode_packet(GpuOpcode.BeginFrame, b"", flags=flags)


def encode_clear_color(r: float, g: float, b: float, a: float, flags: int = 0) -> bytes:
    return encode_packet(
        GpuOpcode.ClearColor,
        CLEAR_COLOR_STRUCT.pack(r, g, b, a),
        flags=flags,
    )


def encode_create_buffer(
    resource_id: int,
    usage_flags: int,
    byte_size: int,
    flags: int = 0,
) -> bytes:
    return encode_packet(
        GpuOpcode.CreateBuffer,
        CREATE_BUFFER_STRUCT.pack(resource_id, usage_flags, byte_size),
        flags=flags,
    )


def encode_write_buffer(
    resource_id: int,
    dst_offset: int,
    data: bytes | bytearray | memoryview,
    *,
    reserved: int = 0,
    flags: int = 0,
) -> bytes:
    raw = bytes(data)
    head = WRITE_BUFFER_STRUCT.pack(resource_id, dst_offset, len(raw), reserved)
    return encode_packet(GpuOpcode.WriteBuffer, head + raw, flags=flags)


def encode_create_shader_module(
    resource_id: int,
    stage: int | GpuShaderStage,
    wgsl_code: str | bytes | bytearray,
    *,
    reserved: int = 0,
    flags: int = 0,
) -> bytes:
    raw = wgsl_code.encode("utf-8") if isinstance(wgsl_code, str) else bytes(wgsl_code)
    head = CREATE_SHADER_MODULE_STRUCT.pack(resource_id, int(stage), len(raw), reserved)
    return encode_packet(GpuOpcode.CreateShaderModule, head + raw, flags=flags)


def encode_draw(
    vertex_count: int,
    instance_count: int = 1,
    first_vertex: int = 0,
    first_instance: int = 0,
    flags: int = 0,
) -> bytes:
    return encode_packet(
        GpuOpcode.Draw,
        DRAW_STRUCT.pack(vertex_count, instance_count, first_vertex, first_instance),
        flags=flags,
    )


def encode_present(flags: int = 0) -> bytes:
    return encode_packet(GpuOpcode.Present, b"", flags=flags)


def encode_create_render_pipeline(
    resource_id: int,
    vs_shader_id: int,
    fs_shader_id: int,
    vertex_stride: int,
    topology: int | GpuPrimitiveTopology,
    pipeline_flags: int = 0,
    flags: int = 0,
) -> bytes:
    return encode_packet(
        GpuOpcode.CreateRenderPipeline,
        CREATE_RENDER_PIPELINE_STRUCT.pack(
            resource_id,
            vs_shader_id,
            fs_shader_id,
            vertex_stride,
            int(topology),
            pipeline_flags,
        ),
        flags=flags,
    )


def encode_set_pipeline(pipeline_id: int, flags: int = 0) -> bytes:
    return encode_packet(
        GpuOpcode.SetPipeline,
        SET_PIPELINE_STRUCT.pack(pipeline_id),
        flags=flags,
    )


def encode_set_vertex_buffer(
    slot: int,
    buffer_id: int,
    offset: int = 0,
    flags: int = 0,
) -> bytes:
    return encode_packet(
        GpuOpcode.SetVertexBuffer,
        SET_VERTEX_BUFFER_STRUCT.pack(slot, buffer_id, offset),
        flags=flags,
    )


def encode_destroy_resource(resource_id: int, flags: int = 0) -> bytes:
    return encode_packet(
        GpuOpcode.DestroyResource,
        DESTROY_RESOURCE_STRUCT.pack(resource_id),
        flags=flags,
    )


def encode_create_texture(
    resource_id: int,
    width: int,
    height: int,
    depth_or_layers: int = 1,
    mip_levels: int = 1,
    texture_format: int | GpuTextureFormat = GpuTextureFormat.Rgba8Unorm,
    usage_flags: int = 0,
    dimension: int | GpuTextureDimension = GpuTextureDimension.Dim2D,
    flags: int = 0,
) -> bytes:
    return encode_packet(
        GpuOpcode.CreateTexture,
        CREATE_TEXTURE_STRUCT.pack(
            resource_id,
            width,
            height,
            depth_or_layers,
            mip_levels,
            int(texture_format),
            usage_flags,
            int(dimension),
        ),
        flags=flags,
    )


def encode_write_texture(
    texture_id: int,
    width: int,
    height: int,
    data: bytes | bytearray | memoryview,
    *,
    mip_level: int = 0,
    origin_x: int = 0,
    origin_y: int = 0,
    origin_z: int = 0,
    depth_or_layers: int = 1,
    bytes_per_row: int = 0,
    rows_per_image: int = 0,
    reserved: int = 0,
    flags: int = 0,
) -> bytes:
    raw = bytes(data)
    head = WRITE_TEXTURE_STRUCT.pack(
        texture_id,
        mip_level,
        origin_x,
        origin_y,
        origin_z,
        width,
        height,
        depth_or_layers,
        bytes_per_row,
        rows_per_image,
        len(raw),
        reserved,
    )
    return encode_packet(GpuOpcode.WriteTexture, head + raw, flags=flags)


def encode_create_sampler(
    resource_id: int,
    address_u: int | GpuAddressMode = GpuAddressMode.ClampToEdge,
    address_v: int | GpuAddressMode = GpuAddressMode.ClampToEdge,
    address_w: int | GpuAddressMode = GpuAddressMode.ClampToEdge,
    mag_filter: int | GpuFilterMode = GpuFilterMode.Linear,
    min_filter: int | GpuFilterMode = GpuFilterMode.Linear,
    mipmap_filter: int | GpuFilterMode = GpuFilterMode.Nearest,
    max_anisotropy: int = 1,
    flags: int = 0,
) -> bytes:
    return encode_packet(
        GpuOpcode.CreateSampler,
        CREATE_SAMPLER_STRUCT.pack(
            resource_id,
            int(address_u),
            int(address_v),
            int(address_w),
            int(mag_filter),
            int(min_filter),
            int(mipmap_filter),
            max_anisotropy,
        ),
        flags=flags,
    )


def encode_bind_group_entry(
    binding: int,
    resource_type: int | GpuBindResourceType,
    resource_id: int,
    offset: int = 0,
    size: int = 0,
    reserved: int = 0,
) -> bytes:
    return BIND_GROUP_ENTRY_STRUCT.pack(
        binding,
        int(resource_type),
        resource_id,
        reserved,
        offset,
        size,
    )


def encode_create_bind_group(
    resource_id: int,
    pipeline_id: int,
    group_index: int,
    entries: Sequence[BindGroupEntry | dict[str, Any] | bytes | bytearray | memoryview],
    *,
    flags: int = 0,
) -> bytes:
    packed_entries: list[bytes] = []
    for entry in entries:
        if isinstance(entry, BindGroupEntry):
            packed_entries.append(
                encode_bind_group_entry(
                    binding=entry.binding,
                    resource_type=entry.resource_type,
                    resource_id=entry.resource_id,
                    offset=entry.offset,
                    size=entry.size,
                    reserved=entry.reserved,
                )
            )
        elif isinstance(entry, dict):
            packed_entries.append(
                encode_bind_group_entry(
                    binding=entry["binding"],
                    resource_type=entry["resource_type"],
                    resource_id=entry["resource_id"],
                    offset=entry.get("offset", 0),
                    size=entry.get("size", 0),
                    reserved=entry.get("reserved", 0),
                )
            )
        else:
            raw_entry = bytes(entry)
            if len(raw_entry) != BIND_GROUP_ENTRY_STRUCT.size:
                raise ValueError(
                    f"Raw bind group entry must be {BIND_GROUP_ENTRY_STRUCT.size} bytes, "
                    f"got {len(raw_entry)}"
                )
            packed_entries.append(raw_entry)
    head = CREATE_BIND_GROUP_STRUCT.pack(resource_id, pipeline_id, group_index, len(packed_entries))
    return encode_packet(GpuOpcode.CreateBindGroup, head + b"".join(packed_entries), flags=flags)


def encode_set_bind_group(
    group_index: int,
    bind_group_id: int,
    flags: int = 0,
) -> bytes:
    return encode_packet(
        GpuOpcode.SetBindGroup,
        SET_BIND_GROUP_STRUCT.pack(group_index, bind_group_id),
        flags=flags,
    )


def encode_set_index_buffer(
    buffer_id: int,
    index_format: int | GpuIndexFormat = GpuIndexFormat.Uint16,
    offset: int = 0,
    flags: int = 0,
) -> bytes:
    return encode_packet(
        GpuOpcode.SetIndexBuffer,
        SET_INDEX_BUFFER_STRUCT.pack(buffer_id, int(index_format), offset),
        flags=flags,
    )


def encode_draw_indexed(
    index_count: int,
    instance_count: int = 1,
    first_index: int = 0,
    base_vertex: int = 0,
    first_instance: int = 0,
    flags: int = 0,
) -> bytes:
    return encode_packet(
        GpuOpcode.DrawIndexed,
        DRAW_INDEXED_STRUCT.pack(
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        ),
        flags=flags,
    )


def encode_create_compute_pipeline(
    resource_id: int,
    cs_shader_id: int,
    pipeline_flags: int = 0,
    *,
    reserved: int = 0,
    flags: int = 0,
) -> bytes:
    return encode_packet(
        GpuOpcode.CreateComputePipeline,
        CREATE_COMPUTE_PIPELINE_STRUCT.pack(resource_id, cs_shader_id, pipeline_flags, reserved),
        flags=flags,
    )


def encode_set_compute_pipeline(pipeline_id: int, flags: int = 0) -> bytes:
    return encode_packet(
        GpuOpcode.SetComputePipeline,
        SET_COMPUTE_PIPELINE_STRUCT.pack(pipeline_id),
        flags=flags,
    )


def encode_dispatch_compute(
    workgroup_x: int,
    workgroup_y: int = 1,
    workgroup_z: int = 1,
    flags: int = 0,
) -> bytes:
    return encode_packet(
        GpuOpcode.DispatchCompute,
        DISPATCH_COMPUTE_STRUCT.pack(workgroup_x, workgroup_y, workgroup_z),
        flags=flags,
    )


def encode_texture_view_desc(
    texture_format: int = 0,
    dimension: int = 0,
    aspect: int = 0,
    base_mip_level: int = 0,
    mip_level_count: int = 0,
    base_array_layer: int = 0,
    array_layer_count: int = 0,
) -> bytes:
    """Encode a 16-byte GpuTextureViewDesc (all zero = default view)."""
    return TEXTURE_VIEW_DESC_STRUCT.pack(
        texture_format, dimension, aspect, 0,
        base_mip_level, mip_level_count, base_array_layer, array_layer_count,
    )


def encode_create_bind_group_layout(
    layout_id: int,
    entries: Sequence[Sequence[int]],
    *,
    content_id: int = 0,
    flags: int = 0,
    packet_flags: int = 0,
) -> bytes:
    """Encode CreateBindGroupLayout. Each entry is (binding, visibility, type, binding_type
    [, view_dimension, entry_flags, storage_format, min_binding_size])."""
    body = bytearray(
        CREATE_BIND_GROUP_LAYOUT_STRUCT.pack(layout_id, len(entries), flags, 0, content_id)
    )
    for entry in entries:
        padded = tuple(entry) + (0,) * (8 - len(entry))
        body += BIND_GROUP_LAYOUT_ENTRY_STRUCT.pack(*padded)
    return encode_packet(GpuOpcode.CreateBindGroupLayout, body, packet_flags)


def encode_create_pipeline_layout(
    layout_id: int,
    bind_group_layout_ids: Sequence[int],
    *,
    content_id: int = 0,
    flags: int = 0,
    packet_flags: int = 0,
) -> bytes:
    """Encode CreatePipelineLayout (group i uses bind_group_layout_ids[i]; 0 = empty layout)."""
    body = CREATE_PIPELINE_LAYOUT_STRUCT.pack(
        layout_id, len(bind_group_layout_ids), flags, 0, content_id
    ) + struct.pack(f"<{len(bind_group_layout_ids)}I", *bind_group_layout_ids)
    return encode_packet(GpuOpcode.CreatePipelineLayout, body, packet_flags)


def encode_create_render_pipeline2(
    pipeline_id: int,
    vs_module_id: int,
    fs_module_id: int,
    pipeline_layout_id: int,
    key: bytes | bytearray,
    flags: int = 0,
) -> bytes:
    """Encode CreateRenderPipeline2 with a raw 376-byte GpuRenderPipelineKey."""
    if len(key) != RENDER_PIPELINE_KEY_SIZE:
        raise ValueError(
            f"render pipeline key must be {RENDER_PIPELINE_KEY_SIZE} bytes, got {len(key)}"
        )
    head = CREATE_RENDER_PIPELINE2_HEAD_STRUCT.pack(
        pipeline_id, vs_module_id, fs_module_id, pipeline_layout_id
    )
    return encode_packet(GpuOpcode.CreateRenderPipeline2, head + bytes(key), flags)


def encode_create_compute_pipeline2(
    pipeline_id: int,
    cs_module_id: int,
    pipeline_layout_id: int,
    cs_shader: int = 0,
    root_signature_id: int = 0,
    flags: int = 0,
) -> bytes:
    return encode_packet(
        GpuOpcode.CreateComputePipeline2,
        CREATE_COMPUTE_PIPELINE2_STRUCT.pack(
            pipeline_id, cs_module_id, pipeline_layout_id, 0, cs_shader, root_signature_id, 0
        ),
        flags,
    )


def encode_begin_render_pass(
    colors: Sequence[dict[str, Any] | None],
    depth_stencil: dict[str, Any] | None = None,
    flags: int = 0,
) -> bytes:
    """Encode BeginRenderPass.

    Color dicts: texture_id, load_op, store_op[, clear_value, view]. The depth-stencil dict:
    texture_id[, depth_load_op, depth_store_op, stencil_load_op, stencil_store_op,
    depth_clear_value, stencil_clear_value, depth_read_only, stencil_read_only, view].
    """
    if len(colors) > 8:
        raise ValueError("at most 8 color attachments")
    body = bytearray(BEGIN_RENDER_PASS_HEAD_STRUCT.pack(len(colors), 0))
    zero_view = encode_texture_view_desc()
    for i in range(8):
        c = colors[i] if i < len(colors) else None
        if c is None:
            body += bytes(RENDER_PASS_COLOR_ATTACHMENT_STRUCT.size)
            continue
        r, g, b, a = c.get("clear_value", (0.0, 0.0, 0.0, 0.0))
        body += RENDER_PASS_COLOR_ATTACHMENT_STRUCT.pack(
            c["texture_id"], c["load_op"], c["store_op"], 0, c.get("view", zero_view), r, g, b, a
        )
    if depth_stencil is None:
        body += bytes(RENDER_PASS_DEPTH_STENCIL_STRUCT.size)
    else:
        ds = depth_stencil
        body += RENDER_PASS_DEPTH_STENCIL_STRUCT.pack(
            ds["texture_id"],
            ds.get("depth_load_op", 0),
            ds.get("depth_store_op", 0),
            ds.get("stencil_load_op", 0),
            ds.get("stencil_store_op", 0),
            ds.get("view", zero_view),
            ds.get("depth_clear_value", 1.0),
            ds.get("stencil_clear_value", 0),
            int(bool(ds.get("depth_read_only", False))),
            int(bool(ds.get("stencil_read_only", False))),
            0,
            0,
        )
    return encode_packet(GpuOpcode.BeginRenderPass, body, flags)


def encode_end_render_pass(flags: int = 0) -> bytes:
    return encode_packet(GpuOpcode.EndRenderPass, b"", flags)


def encode_set_viewport(
    x: float,
    y: float,
    width: float,
    height: float,
    min_depth: float = 0.0,
    max_depth: float = 1.0,
    flags: int = 0,
) -> bytes:
    return encode_packet(
        GpuOpcode.SetViewport,
        SET_VIEWPORT_STRUCT.pack(x, y, width, height, min_depth, max_depth),
        flags,
    )


def encode_set_scissor_rect(x: int, y: int, width: int, height: int, flags: int = 0) -> bytes:
    return encode_packet(
        GpuOpcode.SetScissorRect, SET_SCISSOR_RECT_STRUCT.pack(x, y, width, height), flags
    )


def encode_set_blend_constant(r: float, g: float, b: float, a: float, flags: int = 0) -> bytes:
    return encode_packet(
        GpuOpcode.SetBlendConstant, SET_BLEND_CONSTANT_STRUCT.pack(r, g, b, a), flags
    )


def encode_set_stencil_reference(reference: int, flags: int = 0) -> bytes:
    return encode_packet(
        GpuOpcode.SetStencilReference, SET_STENCIL_REFERENCE_STRUCT.pack(reference), flags
    )


def encode_create_bind_group2(
    bind_group_id: int,
    layout_id: int,
    entries: Sequence[Sequence[Any]],
    flags: int = 0,
) -> bytes:
    """Encode CreateBindGroup2. Each entry is (binding, resource_type, resource_id
    [, offset, size, view_bytes])."""
    body = bytearray(CREATE_BIND_GROUP2_STRUCT.pack(bind_group_id, layout_id, len(entries), 0))
    for entry in entries:
        binding, rtype, rid = entry[0], int(entry[1]), entry[2]
        e_offset = entry[3] if len(entry) > 3 else 0
        e_size = entry[4] if len(entry) > 4 else 0
        view = entry[5] if len(entry) > 5 else encode_texture_view_desc()
        body += BIND_GROUP_ENTRY2_STRUCT.pack(binding, rtype, rid, 0, e_offset, e_size, view)
    return encode_packet(GpuOpcode.CreateBindGroup2, body, flags)


def encode_set_bind_group2(
    group_index: int,
    bind_group_id: int,
    dynamic_offsets: Sequence[int] = (),
    flags: int = 0,
) -> bytes:
    body = SET_BIND_GROUP2_STRUCT.pack(
        group_index, bind_group_id, len(dynamic_offsets), 0
    ) + struct.pack(f"<{len(dynamic_offsets)}I", *dynamic_offsets)
    return encode_packet(GpuOpcode.SetBindGroup2, body, flags)


def encode_create_sampler2(
    sampler_id: int,
    address: tuple[int, int, int] = (0, 0, 0),
    filters: tuple[int, int, int] = (1, 1, 1),
    compare: int = 0,
    max_anisotropy: int = 1,
    lod_min_clamp: float = 0.0,
    lod_max_clamp: float = 32.0,
    flags: int = 0,
) -> bytes:
    """Encode CreateSampler2 (address = (u, v, w), filters = (mag, min, mipmap))."""
    return encode_packet(
        GpuOpcode.CreateSampler2,
        CREATE_SAMPLER2_STRUCT.pack(
            sampler_id, *address, *filters, compare, max_anisotropy, lod_min_clamp, lod_max_clamp
        ),
        flags,
    )


# ---------------------------------------------------------------------------
# Decoding & frame grouping
# ---------------------------------------------------------------------------


def _warn_severity(strict: bool) -> str:
    return "error" if strict else "warning"


def _check_fixed_size(
    *,
    opcode_name: str,
    expected_size: int,
    payload_bytes: int,
    offset: int,
    packet_index: int,
    strict: bool,
    diagnostics: list[TraceDiagnostic],
) -> tuple[bool, bool]:
    """Validate fixed payload size. Returns (can_decode, is_exact_size)."""
    if payload_bytes < expected_size:
        diagnostics.append(
            TraceDiagnostic(
                severity="error",
                code="payload_too_short",
                offset=offset,
                packet_index=packet_index,
                message=(
                    f"{opcode_name} expects {expected_size} payload byte(s), "
                    f"got {payload_bytes}"
                ),
            )
        )
        return False, False
    if payload_bytes > expected_size:
        diagnostics.append(
            TraceDiagnostic(
                severity=_warn_severity(strict),
                code="payload_size_mismatch",
                offset=offset,
                packet_index=packet_index,
                message=(
                    f"{opcode_name} expects {expected_size} payload byte(s), "
                    f"got {payload_bytes} ({payload_bytes - expected_size} extra byte(s))"
                ),
            )
        )
        return True, False
    return True, True


def _decode_known_payload(
    *,
    opcode: GpuOpcode,
    payload: memoryview,
    offset: int,
    packet_index: int,
    strict: bool,
    preview_bytes: int,
    diagnostics: list[TraceDiagnostic],
) -> tuple[dict[str, Any], bool]:
    """Decode payload for a known opcode (0..34). Returns (fields, valid)."""
    payload_len = len(payload)
    name = opcode.name

    if opcode in (GpuOpcode.Nop, GpuOpcode.BeginFrame, GpuOpcode.Present):
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=0,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        fields: dict[str, Any] = {}
        if not exact and payload_len > 0:
            fields["trailing_bytes"] = payload_len
        return fields, can_decode

    if opcode == GpuOpcode.ClearColor:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=CLEAR_COLOR_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {"expected_bytes": CLEAR_COLOR_STRUCT.size, "actual_bytes": payload_len}, False
        r, g, b, a = CLEAR_COLOR_STRUCT.unpack_from(payload, 0)
        for comp_name, val in (("r", r), ("g", g), ("b", b), ("a", a)):
            if not math.isfinite(val):
                diagnostics.append(
                    TraceDiagnostic(
                        severity=_warn_severity(strict),
                        code="non_finite_float",
                        offset=offset,
                        packet_index=packet_index,
                        message=f"ClearColor.{comp_name} is non-finite ({val})",
                    )
                )
        fields = {"r": r, "g": g, "b": b, "a": a}
        if not exact:
            fields["trailing_bytes"] = payload_len - CLEAR_COLOR_STRUCT.size
        return fields, True

    if opcode == GpuOpcode.CreateBuffer:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=CREATE_BUFFER_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {"expected_bytes": CREATE_BUFFER_STRUCT.size, "actual_bytes": payload_len}, False
        res_id, usage_flags, byte_size = CREATE_BUFFER_STRUCT.unpack_from(payload, 0)
        usage_names, unknown_bits = decode_buffer_usage(usage_flags)
        if unknown_bits:
            diagnostics.append(
                TraceDiagnostic(
                    severity=_warn_severity(strict),
                    code="unknown_buffer_usage_bits",
                    offset=offset,
                    packet_index=packet_index,
                    message=(
                        f"CreateBuffer(id={res_id}) has unrecognized usage_flags "
                        f"bits: 0x{unknown_bits:x}"
                    ),
                )
            )
        fields = {
            "id": res_id,
            "usage_flags": usage_flags,
            "usage_names": usage_names,
            "byte_size": byte_size,
        }
        if not exact:
            fields["trailing_bytes"] = payload_len - CREATE_BUFFER_STRUCT.size
        return fields, True

    if opcode == GpuOpcode.WriteBuffer:
        if payload_len < WRITE_BUFFER_STRUCT.size:
            diagnostics.append(
                TraceDiagnostic(
                    severity="error",
                    code="payload_too_short",
                    offset=offset,
                    packet_index=packet_index,
                    message=(
                        f"WriteBuffer header expects at least {WRITE_BUFFER_STRUCT.size} byte(s), "
                        f"got {payload_len}"
                    ),
                )
            )
            return {
                "expected_min_bytes": WRITE_BUFFER_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        res_id, dst_offset, data_len, reserved = WRITE_BUFFER_STRUCT.unpack_from(payload, 0)
        expected_total = WRITE_BUFFER_STRUCT.size + data_len
        available = payload_len - WRITE_BUFFER_STRUCT.size
        valid = True
        if expected_total > payload_len:
            valid = False
            diagnostics.append(
                TraceDiagnostic(
                    severity="error",
                    code="variable_payload_truncated",
                    offset=offset,
                    packet_index=packet_index,
                    message=(
                        f"WriteBuffer(id={res_id}) declares data_len={data_len}, "
                        f"but only {available} byte(s) are present in packet payload"
                    ),
                )
            )
            data_slice = payload[WRITE_BUFFER_STRUCT.size :]
        else:
            if expected_total < payload_len:
                diagnostics.append(
                    TraceDiagnostic(
                        severity=_warn_severity(strict),
                        code="payload_size_mismatch",
                        offset=offset,
                        packet_index=packet_index,
                        message=(
                            f"WriteBuffer(id={res_id}) expects {expected_total} payload byte(s) "
                            f"(16 + {data_len}), got {payload_len}"
                        ),
                    )
                )
            data_slice = payload[WRITE_BUFFER_STRUCT.size : expected_total]

        preview = _format_preview(data_slice, preview_bytes)
        fields = {
            "id": res_id,
            "dst_offset": dst_offset,
            "data_len": data_len,
            "reserved": reserved,
            **preview,
        }
        if not valid:
            fields["available_bytes"] = available
        elif expected_total < payload_len:
            fields["trailing_bytes"] = payload_len - expected_total
        return fields, valid

    if opcode == GpuOpcode.CreateShaderModule:
        if payload_len < CREATE_SHADER_MODULE_STRUCT.size:
            diagnostics.append(
                TraceDiagnostic(
                    severity="error",
                    code="payload_too_short",
                    offset=offset,
                    packet_index=packet_index,
                    message=(
                        "CreateShaderModule header expects at least "
                        f"{CREATE_SHADER_MODULE_STRUCT.size} byte(s), got {payload_len}"
                    ),
                )
            )
            return {
                "expected_min_bytes": CREATE_SHADER_MODULE_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        res_id, stage, code_len, reserved = CREATE_SHADER_MODULE_STRUCT.unpack_from(payload, 0)
        stage_name, stage_known = decode_shader_stage(stage)
        if not stage_known:
            diagnostics.append(
                TraceDiagnostic(
                    severity=_warn_severity(strict),
                    code="unknown_shader_stage",
                    offset=offset,
                    packet_index=packet_index,
                    message=f"CreateShaderModule(id={res_id}) has unknown stage={stage}",
                )
            )
        expected_total = CREATE_SHADER_MODULE_STRUCT.size + code_len
        available = payload_len - CREATE_SHADER_MODULE_STRUCT.size
        valid = True
        if expected_total > payload_len:
            valid = False
            diagnostics.append(
                TraceDiagnostic(
                    severity="error",
                    code="variable_payload_truncated",
                    offset=offset,
                    packet_index=packet_index,
                    message=(
                        f"CreateShaderModule(id={res_id}) declares code_len={code_len}, "
                        f"but only {available} byte(s) are present in packet payload"
                    ),
                )
            )
            code_raw = bytes(payload[CREATE_SHADER_MODULE_STRUCT.size :])
        else:
            if expected_total < payload_len:
                diagnostics.append(
                    TraceDiagnostic(
                        severity=_warn_severity(strict),
                        code="payload_size_mismatch",
                        offset=offset,
                        packet_index=packet_index,
                        message=(
                            f"CreateShaderModule(id={res_id}) expects {expected_total} payload "
                            f"byte(s) (16 + {code_len}), got {payload_len}"
                        ),
                    )
                )
            code_raw = bytes(payload[CREATE_SHADER_MODULE_STRUCT.size : expected_total])

        utf8_valid = True
        try:
            code_text = code_raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            utf8_valid = False
            code_text = code_raw.decode("utf-8", errors="replace")
            diagnostics.append(
                TraceDiagnostic(
                    severity=_warn_severity(strict),
                    code="invalid_utf8_wgsl",
                    offset=offset,
                    packet_index=packet_index,
                    message=f"CreateShaderModule(id={res_id}) contains invalid UTF-8: {exc}",
                )
            )

        line_count = len(code_text.splitlines()) if code_text else 0
        char_count = len(code_text)
        hash_hex = f"0x{fnv1a_64(code_raw):016x}"
        fields = {
            "id": res_id,
            "stage": stage,
            "stage_name": stage_name,
            "code_len": code_len,
            "reserved": reserved,
            "line_count": line_count,
            "char_count": char_count,
            "wgsl_hash_fnv1a64": hash_hex,
            "utf8_valid": utf8_valid,
        }
        if not valid:
            fields["available_bytes"] = available
        elif expected_total < payload_len:
            fields["trailing_bytes"] = payload_len - expected_total
        return fields, valid

    if opcode == GpuOpcode.Draw:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=DRAW_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {"expected_bytes": DRAW_STRUCT.size, "actual_bytes": payload_len}, False
        vertex_count, instance_count, first_vertex, first_instance = DRAW_STRUCT.unpack_from(
            payload, 0
        )
        fields = {
            "vertex_count": vertex_count,
            "instance_count": instance_count,
            "first_vertex": first_vertex,
            "first_instance": first_instance,
        }
        if not exact:
            fields["trailing_bytes"] = payload_len - DRAW_STRUCT.size
        return fields, True

    if opcode == GpuOpcode.CreateRenderPipeline:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=CREATE_RENDER_PIPELINE_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {
                "expected_bytes": CREATE_RENDER_PIPELINE_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        (
            res_id,
            vs_shader_id,
            fs_shader_id,
            vertex_stride,
            topology,
            pipeline_flags,
        ) = CREATE_RENDER_PIPELINE_STRUCT.unpack_from(payload, 0)
        topology_name, topology_webgpu, topo_known = decode_primitive_topology(topology)
        if not topo_known:
            diagnostics.append(
                TraceDiagnostic(
                    severity=_warn_severity(strict),
                    code="unknown_topology",
                    offset=offset,
                    packet_index=packet_index,
                    message=(
                        f"CreateRenderPipeline(id={res_id}) has unknown topology={topology}"
                    ),
                )
            )
        fields = {
            "id": res_id,
            "vs_shader_id": vs_shader_id,
            "fs_shader_id": fs_shader_id,
            "vertex_stride": vertex_stride,
            "topology": topology,
            "topology_name": topology_name,
            "topology_webgpu": topology_webgpu,
            "flags": pipeline_flags,
        }
        if not exact:
            fields["trailing_bytes"] = payload_len - CREATE_RENDER_PIPELINE_STRUCT.size
        return fields, True

    if opcode == GpuOpcode.SetPipeline:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=SET_PIPELINE_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {"expected_bytes": SET_PIPELINE_STRUCT.size, "actual_bytes": payload_len}, False
        (pipeline_id,) = SET_PIPELINE_STRUCT.unpack_from(payload, 0)
        fields = {"pipeline_id": pipeline_id}
        if not exact:
            fields["trailing_bytes"] = payload_len - SET_PIPELINE_STRUCT.size
        return fields, True

    if opcode == GpuOpcode.SetVertexBuffer:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=SET_VERTEX_BUFFER_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {
                "expected_bytes": SET_VERTEX_BUFFER_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        slot, buffer_id, vb_offset = SET_VERTEX_BUFFER_STRUCT.unpack_from(payload, 0)
        fields = {
            "slot": slot,
            "buffer_id": buffer_id,
            "offset": vb_offset,
        }
        if not exact:
            fields["trailing_bytes"] = payload_len - SET_VERTEX_BUFFER_STRUCT.size
        return fields, True

    if opcode == GpuOpcode.DestroyResource:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=DESTROY_RESOURCE_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {
                "expected_bytes": DESTROY_RESOURCE_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        (res_id,) = DESTROY_RESOURCE_STRUCT.unpack_from(payload, 0)
        fields = {"id": res_id}
        if not exact:
            fields["trailing_bytes"] = payload_len - DESTROY_RESOURCE_STRUCT.size
        return fields, True

    if opcode == GpuOpcode.CreateTexture:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=CREATE_TEXTURE_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {
                "expected_bytes": CREATE_TEXTURE_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        (
            res_id,
            width,
            height,
            depth_or_layers,
            mip_levels,
            tex_format,
            usage_flags,
            dimension,
        ) = CREATE_TEXTURE_STRUCT.unpack_from(payload, 0)
        format_name, format_webgpu, fmt_known = decode_texture_format(tex_format)
        if not fmt_known:
            diagnostics.append(
                TraceDiagnostic(
                    severity=_warn_severity(strict),
                    code="unknown_texture_format",
                    offset=offset,
                    packet_index=packet_index,
                    message=f"CreateTexture(id={res_id}) has unknown format={tex_format}",
                )
            )
        usage_names, unknown_bits = decode_texture_usage(usage_flags)
        if unknown_bits:
            diagnostics.append(
                TraceDiagnostic(
                    severity=_warn_severity(strict),
                    code="unknown_texture_usage_bits",
                    offset=offset,
                    packet_index=packet_index,
                    message=(
                        f"CreateTexture(id={res_id}) has unrecognized usage_flags "
                        f"bits: 0x{unknown_bits:x}"
                    ),
                )
            )
        dim_name, dim_webgpu, dim_known = decode_texture_dimension(dimension)
        if not dim_known:
            diagnostics.append(
                TraceDiagnostic(
                    severity=_warn_severity(strict),
                    code="unknown_texture_dimension",
                    offset=offset,
                    packet_index=packet_index,
                    message=f"CreateTexture(id={res_id}) has unknown dimension={dimension}",
                )
            )
        fields = {
            "id": res_id,
            "width": width,
            "height": height,
            "depth_or_layers": depth_or_layers,
            "mip_levels": mip_levels,
            "format": tex_format,
            "format_name": format_name,
            "format_webgpu": format_webgpu,
            "usage_flags": usage_flags,
            "usage_names": usage_names,
            "dimension": dimension,
            "dimension_name": dim_name,
            "dimension_webgpu": dim_webgpu,
        }
        if not exact:
            fields["trailing_bytes"] = payload_len - CREATE_TEXTURE_STRUCT.size
        return fields, True

    if opcode == GpuOpcode.WriteTexture:
        if payload_len < WRITE_TEXTURE_STRUCT.size:
            diagnostics.append(
                TraceDiagnostic(
                    severity="error",
                    code="payload_too_short",
                    offset=offset,
                    packet_index=packet_index,
                    message=(
                        f"WriteTexture header expects at least {WRITE_TEXTURE_STRUCT.size} byte(s), "
                        f"got {payload_len}"
                    ),
                )
            )
            return {
                "expected_min_bytes": WRITE_TEXTURE_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        (
            texture_id,
            mip_level,
            origin_x,
            origin_y,
            origin_z,
            width,
            height,
            depth_or_layers,
            bytes_per_row,
            rows_per_image,
            data_len,
            reserved,
        ) = WRITE_TEXTURE_STRUCT.unpack_from(payload, 0)
        expected_total = WRITE_TEXTURE_STRUCT.size + data_len
        available = payload_len - WRITE_TEXTURE_STRUCT.size
        valid = True
        if expected_total > payload_len:
            valid = False
            diagnostics.append(
                TraceDiagnostic(
                    severity="error",
                    code="variable_payload_truncated",
                    offset=offset,
                    packet_index=packet_index,
                    message=(
                        f"WriteTexture(texture_id={texture_id}) declares data_len={data_len}, "
                        f"but only {available} byte(s) are present in packet payload"
                    ),
                )
            )
            data_slice = payload[WRITE_TEXTURE_STRUCT.size :]
        else:
            if expected_total < payload_len:
                diagnostics.append(
                    TraceDiagnostic(
                        severity=_warn_severity(strict),
                        code="payload_size_mismatch",
                        offset=offset,
                        packet_index=packet_index,
                        message=(
                            f"WriteTexture(texture_id={texture_id}) expects {expected_total} "
                            f"payload byte(s) ({WRITE_TEXTURE_STRUCT.size} + {data_len}), "
                            f"got {payload_len}"
                        ),
                    )
                )
            data_slice = payload[WRITE_TEXTURE_STRUCT.size : expected_total]

        preview = _format_preview(data_slice, preview_bytes)
        fields = {
            "texture_id": texture_id,
            "mip_level": mip_level,
            "origin_x": origin_x,
            "origin_y": origin_y,
            "origin_z": origin_z,
            "width": width,
            "height": height,
            "depth_or_layers": depth_or_layers,
            "bytes_per_row": bytes_per_row,
            "rows_per_image": rows_per_image,
            "data_len": data_len,
            "reserved": reserved,
            **preview,
        }
        if not valid:
            fields["available_bytes"] = available
        elif expected_total < payload_len:
            fields["trailing_bytes"] = payload_len - expected_total
        return fields, valid

    if opcode == GpuOpcode.CreateSampler:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=CREATE_SAMPLER_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {
                "expected_bytes": CREATE_SAMPLER_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        (
            res_id,
            address_u,
            address_v,
            address_w,
            mag_filter,
            min_filter,
            mipmap_filter,
            max_anisotropy,
        ) = CREATE_SAMPLER_STRUCT.unpack_from(payload, 0)

        addr_u_name, addr_u_webgpu, addr_u_known = decode_address_mode(address_u)
        addr_v_name, addr_v_webgpu, addr_v_known = decode_address_mode(address_v)
        addr_w_name, addr_w_webgpu, addr_w_known = decode_address_mode(address_w)
        for axis, raw_val, is_known in (
            ("address_u", address_u, addr_u_known),
            ("address_v", address_v, addr_v_known),
            ("address_w", address_w, addr_w_known),
        ):
            if not is_known:
                diagnostics.append(
                    TraceDiagnostic(
                        severity=_warn_severity(strict),
                        code="unknown_address_mode",
                        offset=offset,
                        packet_index=packet_index,
                        message=f"CreateSampler(id={res_id}) has unknown {axis}={raw_val}",
                    )
                )

        mag_name, mag_webgpu, mag_known = decode_filter_mode(mag_filter)
        min_name, min_webgpu, min_known = decode_filter_mode(min_filter)
        mip_name, mip_webgpu, mip_known = decode_filter_mode(mipmap_filter)
        for filt_field, raw_val, is_known in (
            ("mag_filter", mag_filter, mag_known),
            ("min_filter", min_filter, min_known),
            ("mipmap_filter", mipmap_filter, mip_known),
        ):
            if not is_known:
                diagnostics.append(
                    TraceDiagnostic(
                        severity=_warn_severity(strict),
                        code="unknown_filter_mode",
                        offset=offset,
                        packet_index=packet_index,
                        message=f"CreateSampler(id={res_id}) has unknown {filt_field}={raw_val}",
                    )
                )

        fields = {
            "id": res_id,
            "address_u": address_u,
            "address_u_name": addr_u_name,
            "address_u_webgpu": addr_u_webgpu,
            "address_v": address_v,
            "address_v_name": addr_v_name,
            "address_v_webgpu": addr_v_webgpu,
            "address_w": address_w,
            "address_w_name": addr_w_name,
            "address_w_webgpu": addr_w_webgpu,
            "mag_filter": mag_filter,
            "mag_filter_name": mag_name,
            "mag_filter_webgpu": mag_webgpu,
            "min_filter": min_filter,
            "min_filter_name": min_name,
            "min_filter_webgpu": min_webgpu,
            "mipmap_filter": mipmap_filter,
            "mipmap_filter_name": mip_name,
            "mipmap_filter_webgpu": mip_webgpu,
            "max_anisotropy": max_anisotropy,
        }
        if not exact:
            fields["trailing_bytes"] = payload_len - CREATE_SAMPLER_STRUCT.size
        return fields, True

    if opcode == GpuOpcode.CreateBindGroup:
        if payload_len < CREATE_BIND_GROUP_STRUCT.size:
            diagnostics.append(
                TraceDiagnostic(
                    severity="error",
                    code="payload_too_short",
                    offset=offset,
                    packet_index=packet_index,
                    message=(
                        f"CreateBindGroup header expects at least "
                        f"{CREATE_BIND_GROUP_STRUCT.size} byte(s), got {payload_len}"
                    ),
                )
            )
            return {
                "expected_min_bytes": CREATE_BIND_GROUP_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        res_id, pipeline_id, group_index, entry_count = CREATE_BIND_GROUP_STRUCT.unpack_from(
            payload, 0
        )
        entry_stride = BIND_GROUP_ENTRY_STRUCT.size
        expected_total = CREATE_BIND_GROUP_STRUCT.size + entry_count * entry_stride
        available = payload_len - CREATE_BIND_GROUP_STRUCT.size
        valid = True
        if expected_total > payload_len:
            valid = False
            diagnostics.append(
                TraceDiagnostic(
                    severity="error",
                    code="variable_payload_truncated",
                    offset=offset,
                    packet_index=packet_index,
                    message=(
                        f"CreateBindGroup(id={res_id}) declares entry_count={entry_count} "
                        f"({entry_count * entry_stride} byte(s)), but only {available} byte(s) "
                        f"are present in packet payload"
                    ),
                )
            )
            decodable_entries = available // entry_stride
        else:
            if expected_total < payload_len:
                diagnostics.append(
                    TraceDiagnostic(
                        severity=_warn_severity(strict),
                        code="payload_size_mismatch",
                        offset=offset,
                        packet_index=packet_index,
                        message=(
                            f"CreateBindGroup(id={res_id}) expects {expected_total} payload "
                            f"byte(s) ({CREATE_BIND_GROUP_STRUCT.size} + {entry_count} * "
                            f"{entry_stride}), got {payload_len}"
                        ),
                    )
                )
            decodable_entries = entry_count

        decoded_entries: list[dict[str, Any]] = []
        for idx in range(decodable_entries):
            entry_off = CREATE_BIND_GROUP_STRUCT.size + idx * entry_stride
            (
                binding,
                resource_type,
                resource_id,
                reserved,
                entry_offset,
                entry_size,
            ) = BIND_GROUP_ENTRY_STRUCT.unpack_from(payload, entry_off)
            rtype_name, rtype_known = decode_bind_resource_type(resource_type)
            if not rtype_known:
                diagnostics.append(
                    TraceDiagnostic(
                        severity=_warn_severity(strict),
                        code="unknown_bind_resource_type",
                        offset=offset,
                        packet_index=packet_index,
                        message=(
                            f"CreateBindGroup(id={res_id}) entry[{idx}] (binding={binding}) "
                            f"has unknown resource_type={resource_type}"
                        ),
                    )
                )
            decoded_entries.append(
                {
                    "binding": binding,
                    "resource_type": resource_type,
                    "resource_type_name": rtype_name,
                    "resource_id": resource_id,
                    "reserved": reserved,
                    "offset": entry_offset,
                    "size": entry_size,
                }
            )

        fields = {
            "id": res_id,
            "pipeline_id": pipeline_id,
            "group_index": group_index,
            "entry_count": entry_count,
            "entries": decoded_entries,
        }
        if not valid:
            fields["available_bytes"] = available
        elif expected_total < payload_len:
            fields["trailing_bytes"] = payload_len - expected_total
        return fields, valid

    if opcode == GpuOpcode.SetBindGroup:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=SET_BIND_GROUP_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {
                "expected_bytes": SET_BIND_GROUP_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        group_index, bind_group_id = SET_BIND_GROUP_STRUCT.unpack_from(payload, 0)
        fields = {
            "group_index": group_index,
            "bind_group_id": bind_group_id,
        }
        if not exact:
            fields["trailing_bytes"] = payload_len - SET_BIND_GROUP_STRUCT.size
        return fields, True

    if opcode == GpuOpcode.SetIndexBuffer:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=SET_INDEX_BUFFER_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {
                "expected_bytes": SET_INDEX_BUFFER_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        buffer_id, index_format, ib_offset = SET_INDEX_BUFFER_STRUCT.unpack_from(payload, 0)
        fmt_name, fmt_webgpu, fmt_known = decode_index_format(index_format)
        if not fmt_known:
            diagnostics.append(
                TraceDiagnostic(
                    severity=_warn_severity(strict),
                    code="unknown_index_format",
                    offset=offset,
                    packet_index=packet_index,
                    message=(
                        f"SetIndexBuffer(buffer_id={buffer_id}) has unknown "
                        f"index_format={index_format}"
                    ),
                )
            )
        fields = {
            "buffer_id": buffer_id,
            "index_format": index_format,
            "index_format_name": fmt_name,
            "index_format_webgpu": fmt_webgpu,
            "offset": ib_offset,
        }
        if not exact:
            fields["trailing_bytes"] = payload_len - SET_INDEX_BUFFER_STRUCT.size
        return fields, True

    if opcode == GpuOpcode.DrawIndexed:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=DRAW_INDEXED_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {
                "expected_bytes": DRAW_INDEXED_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        (
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        ) = DRAW_INDEXED_STRUCT.unpack_from(payload, 0)
        fields = {
            "index_count": index_count,
            "instance_count": instance_count,
            "first_index": first_index,
            "base_vertex": base_vertex,
            "first_instance": first_instance,
        }
        if not exact:
            fields["trailing_bytes"] = payload_len - DRAW_INDEXED_STRUCT.size
        return fields, True

    if opcode == GpuOpcode.CreateComputePipeline:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=CREATE_COMPUTE_PIPELINE_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {
                "expected_bytes": CREATE_COMPUTE_PIPELINE_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        res_id, cs_shader_id, pipeline_flags, reserved = (
            CREATE_COMPUTE_PIPELINE_STRUCT.unpack_from(payload, 0)
        )
        fields = {
            "id": res_id,
            "cs_shader_id": cs_shader_id,
            "flags": pipeline_flags,
            "reserved": reserved,
        }
        if not exact:
            fields["trailing_bytes"] = payload_len - CREATE_COMPUTE_PIPELINE_STRUCT.size
        return fields, True

    if opcode == GpuOpcode.SetComputePipeline:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=SET_COMPUTE_PIPELINE_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {
                "expected_bytes": SET_COMPUTE_PIPELINE_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        (pipeline_id,) = SET_COMPUTE_PIPELINE_STRUCT.unpack_from(payload, 0)
        fields = {"pipeline_id": pipeline_id}
        if not exact:
            fields["trailing_bytes"] = payload_len - SET_COMPUTE_PIPELINE_STRUCT.size
        return fields, True

    if opcode == GpuOpcode.DispatchCompute:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=DISPATCH_COMPUTE_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {
                "expected_bytes": DISPATCH_COMPUTE_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        workgroup_x, workgroup_y, workgroup_z = DISPATCH_COMPUTE_STRUCT.unpack_from(payload, 0)
        fields = {
            "workgroup_x": workgroup_x,
            "workgroup_y": workgroup_y,
            "workgroup_z": workgroup_z,
        }
        if not exact:
            fields["trailing_bytes"] = payload_len - DISPATCH_COMPUTE_STRUCT.size
        return fields, True

    if opcode >= GpuOpcode.CreateBindGroupLayout:
        return _decode_v2_payload(
            opcode=opcode,
            payload=payload,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )

    raise AssertionError(f"Unhandled known opcode: {opcode}")


# ---------------------------------------------------------------------------
# Packet ABI v2 decoding (opcodes 22..34)
# ---------------------------------------------------------------------------


class _V2Context:
    """Shared diagnostics plumbing for one v2 packet."""

    def __init__(
        self,
        name: str,
        offset: int,
        packet_index: int,
        strict: bool,
        diagnostics: list[TraceDiagnostic],
    ) -> None:
        self.name = name
        self.offset = offset
        self.packet_index = packet_index
        self.strict = strict
        self.diagnostics = diagnostics

    def warn(self, code: str, message: str) -> None:
        self.diagnostics.append(
            TraceDiagnostic(
                severity=_warn_severity(self.strict),
                code=code,
                offset=self.offset,
                packet_index=self.packet_index,
                message=f"{self.name}: {message}",
            )
        )

    def error(self, code: str, message: str) -> None:
        self.diagnostics.append(
            TraceDiagnostic(
                severity="error",
                code=code,
                offset=self.offset,
                packet_index=self.packet_index,
                message=f"{self.name}: {message}",
            )
        )

    def code(self, table: Sequence[str], value: int, what: str) -> str:
        """Name for an enum code; unknown codes produce an `unknown_enum_code` diagnostic."""
        if 0 <= value < len(table):
            return table[value]
        self.warn("unknown_enum_code", f"unknown {what} code {value}")
        return f"Unknown({value})"

    def fixed(self, payload_len: int, size: int) -> tuple[bool, bool]:
        return _check_fixed_size(
            opcode_name=self.name,
            expected_size=size,
            payload_bytes=payload_len,
            offset=self.offset,
            packet_index=self.packet_index,
            strict=self.strict,
            diagnostics=self.diagnostics,
        )

    def tail(self, payload_len: int, head_size: int, count: int, stride: int) -> tuple[int, bool, int]:
        """Validate `head + count * stride`; returns (decodable_count, valid, trailing_bytes)."""
        expected = head_size + count * stride
        available = payload_len - head_size
        if expected > payload_len:
            self.error(
                "variable_payload_truncated",
                f"declares {count} element(s) ({count * stride} byte(s)) but only "
                f"{available} byte(s) are present",
            )
            return available // stride, False, 0
        if expected < payload_len:
            self.warn(
                "payload_size_mismatch",
                f"expects {expected} payload byte(s), got {payload_len}",
            )
        return count, True, payload_len - expected


def _f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _decode_view_desc(ctx: _V2Context, raw: bytes | memoryview) -> dict[str, Any]:
    fmt, dim, aspect, _reserved, base_mip, mip_count, base_layer, layer_count = (
        TEXTURE_VIEW_DESC_STRUCT.unpack(bytes(raw))
    )
    return {
        "format": ctx.code(WGPU_TEXTURE_FORMAT_NAMES, fmt, "texture format"),
        "dimension": ctx.code(WGPU_VIEW_DIMENSION_NAMES, dim, "view dimension"),
        "aspect": ctx.code(WGPU_TEXTURE_ASPECT_NAMES, aspect, "texture aspect"),
        "base_mip_level": base_mip,
        "mip_level_count": mip_count,
        "base_array_layer": base_layer,
        "array_layer_count": layer_count,
    }


def _decode_render_pipeline_key(ctx: _V2Context, key: bytes) -> dict[str, Any]:
    vs_shader, ps_shader, root_signature_id, color_target_count = KEY_HEAD_STRUCT.unpack_from(key, 0)
    if color_target_count > 8:
        ctx.error("invalid_count", f"color_target_count {color_target_count} > 8")
    color_targets: list[dict[str, Any] | None] = []
    for i in range(min(color_target_count, 8)):
        (fmt, blend_enabled, write_mask, c_op, c_src, c_dst, a_op, a_src, a_dst) = (
            KEY_COLOR_TARGET_STRUCT.unpack_from(key, 24 + 12 * i)
        )
        if fmt == 0:
            color_targets.append(None)
            continue
        target: dict[str, Any] = {
            "format": ctx.code(WGPU_TEXTURE_FORMAT_NAMES, fmt, "texture format"),
            "write_mask": write_mask,
        }
        if blend_enabled:
            target["blend"] = {
                "color": {
                    "operation": ctx.code(WGPU_BLEND_OPERATION_NAMES, c_op, "blend operation"),
                    "src_factor": ctx.code(WGPU_BLEND_FACTOR_NAMES, c_src, "blend factor"),
                    "dst_factor": ctx.code(WGPU_BLEND_FACTOR_NAMES, c_dst, "blend factor"),
                },
                "alpha": {
                    "operation": ctx.code(WGPU_BLEND_OPERATION_NAMES, a_op, "blend operation"),
                    "src_factor": ctx.code(WGPU_BLEND_FACTOR_NAMES, a_src, "blend factor"),
                    "dst_factor": ctx.code(WGPU_BLEND_FACTOR_NAMES, a_dst, "blend factor"),
                },
            }
        color_targets.append(target)

    (
        ds_format, depth_write, depth_compare,
        f_cmp, f_fail, f_dfail, f_pass, b_cmp, b_fail, b_dfail, b_pass,
        read_mask, write_mask_ds, depth_bias, slope_bits, clamp_bits,
    ) = KEY_DEPTH_STENCIL_STRUCT.unpack_from(key, 120)
    depth_stencil: dict[str, Any] | None = None
    if ds_format != 0:

        def face(cmp: int, fail: int, dfail: int, pss: int) -> dict[str, str]:
            return {
                "compare": ctx.code(WGPU_COMPARE_FUNCTION_NAMES, cmp, "compare function"),
                "fail_op": ctx.code(WGPU_STENCIL_OPERATION_NAMES, fail, "stencil operation"),
                "depth_fail_op": ctx.code(WGPU_STENCIL_OPERATION_NAMES, dfail, "stencil operation"),
                "pass_op": ctx.code(WGPU_STENCIL_OPERATION_NAMES, pss, "stencil operation"),
            }

        depth_stencil = {
            "format": ctx.code(WGPU_TEXTURE_FORMAT_NAMES, ds_format, "texture format"),
            "depth_write_enabled": bool(depth_write),
            "depth_compare": ctx.code(WGPU_COMPARE_FUNCTION_NAMES, depth_compare, "compare function"),
            "stencil_front": face(f_cmp, f_fail, f_dfail, f_pass),
            "stencil_back": face(b_cmp, b_fail, b_dfail, b_pass),
            "stencil_read_mask": read_mask,
            "stencil_write_mask": write_mask_ds,
            "depth_bias": depth_bias,
            "depth_bias_slope_scale": _f32_from_bits(slope_bits),
            "depth_bias_clamp": _f32_from_bits(clamp_bits),
        }

    topology, strip_index_format, front_face, cull_mode, unclipped_depth = (
        KEY_PRIMITIVE_STRUCT.unpack_from(key, 152)
    )
    count, mask, alpha_to_coverage = KEY_MULTISAMPLE_STRUCT.unpack_from(key, 160)
    vertex_buffer_count, vertex_attribute_count = KEY_COUNTS_STRUCT.unpack_from(key, 172)
    if vertex_buffer_count > 8 or vertex_attribute_count > 16:
        ctx.error(
            "invalid_count",
            f"vertex_buffer_count {vertex_buffer_count} / vertex_attribute_count "
            f"{vertex_attribute_count} exceed 8 / 16",
        )
    attributes = []
    for i in range(min(vertex_attribute_count, 16)):
        attr_offset, attr_format, location, buffer_index = KEY_VERTEX_ATTRIBUTE_STRUCT.unpack_from(
            key, 244 + 8 * i
        )
        attributes.append(
            {
                "offset": attr_offset,
                "format": ctx.code(WGPU_VERTEX_FORMAT_NAMES, attr_format, "vertex format"),
                "shader_location": location,
                "buffer_index": buffer_index,
            }
        )
    vertex_buffers: list[dict[str, Any] | None] = []
    for i in range(min(vertex_buffer_count, 8)):
        stride, step_mode, attribute_count = KEY_VERTEX_BUFFER_STRUCT.unpack_from(key, 180 + 8 * i)
        if step_mode == 0:
            vertex_buffers.append(None)
            continue
        vertex_buffers.append(
            {
                "array_stride": stride,
                "step_mode": ctx.code(WGPU_VERTEX_STEP_MODE_NAMES, step_mode, "vertex step mode"),
                "attributes": [a for a in attributes if a["buffer_index"] == i],
            }
        )

    return {
        "vs_shader": f"0x{vs_shader:016x}",
        "ps_shader": f"0x{ps_shader:016x}",
        "root_signature_id": root_signature_id,
        "color_target_count": color_target_count,
        "color_targets": color_targets,
        "depth_stencil": depth_stencil,
        "primitive": {
            "topology": ctx.code(tuple(TOPOLOGY_WEBGPU_NAMES.values()), topology, "topology"),
            "strip_index_format": ctx.code(WGPU_INDEX_FORMAT_NAMES, strip_index_format, "index format"),
            "front_face": ctx.code(WGPU_FRONT_FACE_NAMES, front_face, "front face"),
            "cull_mode": ctx.code(WGPU_CULL_MODE_NAMES, cull_mode, "cull mode"),
            "unclipped_depth": bool(unclipped_depth),
        },
        "multisample": {
            "count": count,
            "mask": mask,
            "alpha_to_coverage_enabled": bool(alpha_to_coverage),
        },
        "vertex_buffer_count": vertex_buffer_count,
        "vertex_attribute_count": vertex_attribute_count,
        "vertex_buffers": vertex_buffers,
        "key_fnv1a64": f"0x{fnv1a_64(key):016x}",
    }


def _decode_v2_payload(
    *,
    opcode: GpuOpcode,
    payload: memoryview,
    offset: int,
    packet_index: int,
    strict: bool,
    diagnostics: list[TraceDiagnostic],
) -> tuple[dict[str, Any], bool]:
    """Decode payload for a v2 opcode (22..34). Returns (fields, valid)."""
    ctx = _V2Context(opcode.name, offset, packet_index, strict, diagnostics)
    payload_len = len(payload)

    def too_short(size: int) -> tuple[dict[str, Any], bool]:
        ctx.error("payload_too_short", f"expects at least {size} payload byte(s), got {payload_len}")
        return {"expected_min_bytes": size, "actual_bytes": payload_len}, False

    def fixed_fields(size: int) -> tuple[bool, dict[str, Any]]:
        can_decode, exact = ctx.fixed(payload_len, size)
        extra: dict[str, Any] = {}
        if not can_decode:
            extra = {"expected_bytes": size, "actual_bytes": payload_len}
        elif not exact:
            extra = {"trailing_bytes": payload_len - size}
        return can_decode, extra

    if opcode in (GpuOpcode.CreateBindGroupLayout, GpuOpcode.CreatePipelineLayout):
        head = CREATE_BIND_GROUP_LAYOUT_STRUCT
        if payload_len < head.size:
            return too_short(head.size)
        res_id, count, flags, _reserved, content_id = head.unpack_from(payload, 0)
        manifest_ref = bool(flags & MANIFEST_REFERENCE_FLAG)
        if manifest_ref and count:
            ctx.warn("manifest_reference_with_entries", f"ManifestReference with {count} element(s)")
        fields: dict[str, Any] = {
            "id": res_id,
            "flags": flags,
            "manifest_reference": manifest_ref,
            "content_id": f"0x{content_id:016x}",
        }
        if opcode == GpuOpcode.CreatePipelineLayout:
            n, valid, trailing = ctx.tail(payload_len, head.size, count, 4)
            fields["bind_group_layout_count"] = count
            fields["bind_group_layout_ids"] = list(
                struct.unpack_from(f"<{n}I", payload, head.size)
            )
        else:
            stride = BIND_GROUP_LAYOUT_ENTRY_STRUCT.size
            n, valid, trailing = ctx.tail(payload_len, head.size, count, stride)
            entries = []
            for i in range(n):
                (binding, visibility, ltype, subtype, view_dim, eflags, storage_format, min_size) = (
                    BIND_GROUP_LAYOUT_ENTRY_STRUCT.unpack_from(payload, head.size + i * stride)
                )
                entry: dict[str, Any] = {
                    "binding": binding,
                    "visibility": visibility,
                    "visibility_names": [n_ for bit, n_ in SHADER_STAGE_BITS if visibility & bit],
                    "type": ctx.code(BINDING_LAYOUT_TYPE_NAMES, ltype, "binding layout type"),
                    "binding_type": ctx.code(
                        BINDING_SUBTYPE_NAMES.get(ltype, ()), subtype, "binding type"
                    ),
                }
                if ltype in (2, 3):
                    entry["view_dimension"] = ctx.code(
                        WGPU_VIEW_DIMENSION_NAMES, view_dim, "view dimension"
                    )
                if ltype == 0:
                    entry["has_dynamic_offset"] = bool(eflags & 0x1)
                    entry["min_binding_size"] = min_size
                if ltype == 2:
                    entry["multisampled"] = bool(eflags & 0x2)
                if ltype == 3:
                    entry["storage_format"] = ctx.code(
                        WGPU_TEXTURE_FORMAT_NAMES, storage_format, "texture format"
                    )
                entries.append(entry)
            fields["entry_count"] = count
            fields["entries"] = entries
        if trailing:
            fields["trailing_bytes"] = trailing
        return fields, valid

    if opcode == GpuOpcode.CreateRenderPipeline2:
        can_decode, extra = fixed_fields(CREATE_RENDER_PIPELINE2_SIZE)
        if not can_decode:
            return extra, False
        res_id, vs_id, fs_id, layout_id = CREATE_RENDER_PIPELINE2_HEAD_STRUCT.unpack_from(payload, 0)
        key = bytes(payload[16:CREATE_RENDER_PIPELINE2_SIZE])
        return {
            "id": res_id,
            "vs_module_id": vs_id,
            "fs_module_id": fs_id,
            "pipeline_layout_id": layout_id,
            "key": _decode_render_pipeline_key(ctx, key),
            **extra,
        }, True

    if opcode == GpuOpcode.CreateComputePipeline2:
        can_decode, extra = fixed_fields(CREATE_COMPUTE_PIPELINE2_STRUCT.size)
        if not can_decode:
            return extra, False
        res_id, cs_id, layout_id, _r, cs_shader, root_sig, _r2 = (
            CREATE_COMPUTE_PIPELINE2_STRUCT.unpack_from(payload, 0)
        )
        return {
            "id": res_id,
            "cs_module_id": cs_id,
            "pipeline_layout_id": layout_id,
            "cs_shader": f"0x{cs_shader:016x}",
            "root_signature_id": root_sig,
            **extra,
        }, True

    if opcode == GpuOpcode.BeginRenderPass:
        can_decode, extra = fixed_fields(BEGIN_RENDER_PASS_SIZE)
        if not can_decode:
            return extra, False
        color_count, _reserved = BEGIN_RENDER_PASS_HEAD_STRUCT.unpack_from(payload, 0)
        valid = True
        if color_count > 8:
            ctx.error("invalid_count", f"color_count {color_count} > 8")
            valid = False
        colors: list[dict[str, Any] | None] = []
        for i in range(min(color_count, 8)):
            (tex, load_op, store_op, _r, view, r, g, b, a) = (
                RENDER_PASS_COLOR_ATTACHMENT_STRUCT.unpack_from(payload, 8 + 40 * i)
            )
            if tex == 0:
                colors.append(None)
                continue
            colors.append(
                {
                    "texture_id": tex,
                    "swapchain": tex == SWAPCHAIN_TEXTURE_ID,
                    "load_op": ctx.code(WGPU_LOAD_OP_NAMES, load_op, "load op"),
                    "store_op": ctx.code(WGPU_STORE_OP_NAMES, store_op, "store op"),
                    "clear_value": [r, g, b, a],
                    "view": _decode_view_desc(ctx, view),
                }
            )
        (
            ds_tex, d_load, d_store, s_load, s_store, ds_view, depth_clear, stencil_clear,
            depth_ro, stencil_ro, _r0, _r1,
        ) = RENDER_PASS_DEPTH_STENCIL_STRUCT.unpack_from(payload, 328)
        depth_stencil: dict[str, Any] | None = None
        if ds_tex != 0:
            depth_stencil = {
                "texture_id": ds_tex,
                "depth_load_op": ctx.code(WGPU_LOAD_OP_NAMES, d_load, "load op"),
                "depth_store_op": ctx.code(WGPU_STORE_OP_NAMES, d_store, "store op"),
                "stencil_load_op": ctx.code(WGPU_LOAD_OP_NAMES, s_load, "load op"),
                "stencil_store_op": ctx.code(WGPU_STORE_OP_NAMES, s_store, "store op"),
                "depth_clear_value": depth_clear,
                "stencil_clear_value": stencil_clear,
                "depth_read_only": bool(depth_ro),
                "stencil_read_only": bool(stencil_ro),
                "view": _decode_view_desc(ctx, ds_view),
            }
        return {
            "color_count": color_count,
            "color_attachments": colors,
            "depth_stencil": depth_stencil,
            **extra,
        }, valid

    if opcode == GpuOpcode.EndRenderPass:
        can_decode, extra = fixed_fields(0)
        return extra, can_decode

    simple: dict[GpuOpcode, tuple[struct.Struct, tuple[str, ...]]] = {
        GpuOpcode.SetViewport: (
            SET_VIEWPORT_STRUCT,
            ("x", "y", "width", "height", "min_depth", "max_depth"),
        ),
        GpuOpcode.SetScissorRect: (SET_SCISSOR_RECT_STRUCT, ("x", "y", "width", "height")),
        GpuOpcode.SetBlendConstant: (SET_BLEND_CONSTANT_STRUCT, ("r", "g", "b", "a")),
        GpuOpcode.SetStencilReference: (SET_STENCIL_REFERENCE_STRUCT, ("reference",)),
    }
    if opcode in simple:
        layout, names = simple[opcode]
        can_decode, extra = fixed_fields(layout.size)
        if not can_decode:
            return extra, False
        values = layout.unpack_from(payload, 0)
        for field_name, value in zip(names, values):
            if isinstance(value, float) and not math.isfinite(value):
                ctx.warn("non_finite_float", f"{field_name} is non-finite ({value})")
        return {**dict(zip(names, values)), **extra}, True

    if opcode == GpuOpcode.CreateBindGroup2:
        head = CREATE_BIND_GROUP2_STRUCT
        if payload_len < head.size:
            return too_short(head.size)
        res_id, layout_id, entry_count, _reserved = head.unpack_from(payload, 0)
        stride = BIND_GROUP_ENTRY2_STRUCT.size
        n, valid, trailing = ctx.tail(payload_len, head.size, entry_count, stride)
        entries = []
        for i in range(n):
            binding, rtype, rid, _r, e_offset, e_size, view = BIND_GROUP_ENTRY2_STRUCT.unpack_from(
                payload, head.size + i * stride
            )
            rtype_name, rtype_known = decode_bind_resource_type(rtype)
            if not rtype_known:
                ctx.warn("unknown_bind_resource_type", f"entry[{i}] has resource_type={rtype}")
            entry = {
                "binding": binding,
                "resource_type": rtype,
                "resource_type_name": rtype_name,
                "resource_id": rid,
                "null_binding": rid == 0,
                "offset": e_offset,
                "size": e_size,
            }
            if rtype == GpuBindResourceType.TextureView:
                entry["view"] = _decode_view_desc(ctx, view)
            entries.append(entry)
        fields = {
            "id": res_id,
            "layout_id": layout_id,
            "entry_count": entry_count,
            "entries": entries,
        }
        if trailing:
            fields["trailing_bytes"] = trailing
        return fields, valid

    if opcode == GpuOpcode.SetBindGroup2:
        head = SET_BIND_GROUP2_STRUCT
        if payload_len < head.size:
            return too_short(head.size)
        group_index, bind_group_id, count, _reserved = head.unpack_from(payload, 0)
        n, valid, trailing = ctx.tail(payload_len, head.size, count, 4)
        fields = {
            "group_index": group_index,
            "bind_group_id": bind_group_id,
            "dynamic_offsets": list(struct.unpack_from(f"<{n}I", payload, head.size)),
        }
        if trailing:
            fields["trailing_bytes"] = trailing
        return fields, valid

    if opcode == GpuOpcode.CreateSampler2:
        can_decode, extra = fixed_fields(CREATE_SAMPLER2_STRUCT.size)
        if not can_decode:
            return extra, False
        (res_id, au, av, aw, mag, min_f, mip, compare, aniso, lod_min, lod_max) = (
            CREATE_SAMPLER2_STRUCT.unpack_from(payload, 0)
        )
        addr_names = tuple(ADDRESS_MODE_WEBGPU_NAMES.values())
        filter_names = tuple(FILTER_MODE_WEBGPU_NAMES.values())
        return {
            "id": res_id,
            "address_u": ctx.code(addr_names, au, "address mode"),
            "address_v": ctx.code(addr_names, av, "address mode"),
            "address_w": ctx.code(addr_names, aw, "address mode"),
            "mag_filter": ctx.code(filter_names, mag, "filter mode"),
            "min_filter": ctx.code(filter_names, min_f, "filter mode"),
            "mipmap_filter": ctx.code(filter_names, mip, "filter mode"),
            "compare": ctx.code(WGPU_COMPARE_FUNCTION_NAMES, compare, "compare function"),
            "max_anisotropy": aniso,
            "lod_min_clamp": lod_min,
            "lod_max_clamp": lod_max,
            **extra,
        }, True

    raise AssertionError(f"Unhandled v2 opcode: {opcode}")


def split_frames(packets: Sequence[DecodedPacket]) -> list[FrameSummary]:
    """Group packets by BeginFrame..Present into structured FrameSummary objects.

    Every packet in the sequence is accounted for:
    - Standard frames span from `BeginFrame` through `Present` (`complete=True`).
    - Setup packets appearing before the first `BeginFrame` (or between `Present`
      and the next `BeginFrame`) are preserved as a frame group with
      `has_begin_frame=False`.
    - An unclosed frame at EOF (or interrupted by a new `BeginFrame`) is emitted
      with `complete=False`.
    """
    frames: list[FrameSummary] = []
    current: list[DecodedPacket] = []

    render_pass_opcodes = {
        GpuOpcode.SetPipeline,
        GpuOpcode.SetVertexBuffer,
        GpuOpcode.SetIndexBuffer,
        GpuOpcode.Draw,
        GpuOpcode.DrawIndexed,
        GpuOpcode.BeginRenderPass,
    }
    compute_pass_opcodes = {
        GpuOpcode.SetComputePipeline,
        GpuOpcode.DispatchCompute,
    }

    def _flush(*, complete: bool) -> None:
        if not current:
            return
        histogram: Counter[str] = Counter()
        draw_calls = 0
        draw_indexed_calls = 0
        compute_dispatches = 0
        compute_passes = 0
        textures_created = 0
        bind_groups_created = 0
        total_bytes = 0
        active_pass: str | None = None
        for pkt in current:
            histogram[pkt.name] += 1
            total_bytes += pkt.total_bytes
            if not pkt.valid:
                continue
            if pkt.opcode == GpuOpcode.Draw:
                draw_calls += 1
            elif pkt.opcode == GpuOpcode.DrawIndexed:
                draw_calls += 1
                draw_indexed_calls += 1
            elif pkt.opcode == GpuOpcode.DispatchCompute:
                compute_dispatches += 1
            elif pkt.opcode == GpuOpcode.CreateTexture:
                textures_created += 1
            elif pkt.opcode in (GpuOpcode.CreateBindGroup, GpuOpcode.CreateBindGroup2):
                bind_groups_created += 1

            if pkt.opcode in compute_pass_opcodes:
                if active_pass != "compute":
                    compute_passes += 1
                    active_pass = "compute"
            elif pkt.opcode in render_pass_opcodes:
                active_pass = "render"
            elif pkt.opcode == GpuOpcode.EndRenderPass:
                active_pass = None
        frames.append(
            FrameSummary(
                frame_index=len(frames),
                start_packet_index=current[0].index,
                end_packet_index=current[-1].index,
                packet_count=len(current),
                draw_calls=draw_calls,
                bytes=total_bytes,
                opcode_histogram=dict(histogram),
                complete=complete,
                has_begin_frame=(current[0].opcode == GpuOpcode.BeginFrame),
                draw_indexed_calls=draw_indexed_calls,
                compute_dispatches=compute_dispatches,
                compute_passes=compute_passes,
                textures_created=textures_created,
                bind_groups_created=bind_groups_created,
            )
        )
        current.clear()

    for pkt in packets:
        if pkt.opcode == GpuOpcode.BeginFrame and current:
            _flush(complete=False)
        current.append(pkt)
        if pkt.opcode == GpuOpcode.Present:
            _flush(complete=True)

    if current:
        _flush(complete=False)

    return frames


def decode_stream(
    data: bytes | bytearray | memoryview,
    *,
    strict: bool = False,
    preview_bytes: int = DEFAULT_PREVIEW_BYTES,
) -> DecodedTrace:
    """Decode a raw binary GPU command stream into a DecodedTrace."""
    view = memoryview(data)
    total_bytes = len(view)
    offset = 0
    packet_index = 0
    packets: list[DecodedPacket] = []
    diagnostics: list[TraceDiagnostic] = []
    histogram: Counter[str] = Counter()

    while offset < total_bytes:
        remaining = total_bytes - offset
        if remaining < HEADER_SIZE:
            diagnostics.append(
                TraceDiagnostic(
                    severity="error",
                    code="truncated_header",
                    offset=offset,
                    packet_index=packet_index,
                    message=(
                        f"Truncated packet header at byte {offset}: "
                        f"need {HEADER_SIZE} bytes, only {remaining} byte(s) remaining"
                    ),
                )
            )
            break

        raw_opcode, flags, payload_bytes = HEADER_STRUCT.unpack_from(view, offset)
        payload_offset = offset + HEADER_SIZE
        packet_end = payload_offset + payload_bytes

        try:
            known_opcode: GpuOpcode | None = GpuOpcode(raw_opcode)
            name = known_opcode.name
        except ValueError:
            known_opcode = None
            name = f"Unknown({raw_opcode})"

        if packet_end > total_bytes:
            available = total_bytes - payload_offset
            diagnostics.append(
                TraceDiagnostic(
                    severity="error",
                    code="truncated_payload",
                    offset=offset,
                    packet_index=packet_index,
                    message=(
                        f"Truncated payload for {name} (opcode {raw_opcode}) at byte {offset}: "
                        f"header declares {payload_bytes} byte(s), only {available} byte(s) remaining"
                    ),
                )
            )
            break

        payload = view[payload_offset:packet_end]
        if known_opcode is not None:
            fields, valid = _decode_known_payload(
                opcode=known_opcode,
                payload=payload,
                offset=offset,
                packet_index=packet_index,
                strict=strict,
                preview_bytes=preview_bytes,
                diagnostics=diagnostics,
            )
            pkt = DecodedPacket(
                index=packet_index,
                offset=offset,
                opcode=raw_opcode,
                name=name,
                flags=flags,
                payload_bytes=payload_bytes,
                total_bytes=HEADER_SIZE + payload_bytes,
                known=True,
                fields=fields,
                valid=valid,
            )
        else:
            if strict:
                diagnostics.append(
                    TraceDiagnostic(
                        severity="error",
                        code="unknown_opcode",
                        offset=offset,
                        packet_index=packet_index,
                        message=(
                            f"Unknown opcode {raw_opcode} at byte {offset} "
                            f"({payload_bytes} payload byte(s))"
                        ),
                    )
                )
            preview = _format_preview(payload, preview_bytes)
            pkt = DecodedPacket(
                index=packet_index,
                offset=offset,
                opcode=raw_opcode,
                name=name,
                flags=flags,
                payload_bytes=payload_bytes,
                total_bytes=HEADER_SIZE + payload_bytes,
                known=False,
                fields={
                    "payload_bytes": payload_bytes,
                    **preview,
                },
                valid=not strict,
            )

        packets.append(pkt)
        histogram[pkt.name] += 1
        offset = packet_end
        packet_index += 1

    frames = split_frames(packets)
    return DecodedTrace(
        abi_version=ABI_VERSION,
        total_bytes=total_bytes,
        consumed_bytes=offset,
        packets=tuple(packets),
        frames=tuple(frames),
        opcode_histogram=dict(histogram),
        diagnostics=tuple(diagnostics),
        strict=strict,
    )


# ---------------------------------------------------------------------------
# Formatting & JSON serialization
# ---------------------------------------------------------------------------


def _format_packet_fields(pkt: DecodedPacket) -> str:
    f = pkt.fields
    if not pkt.known:
        preview = f.get("preview_hex", "")
        suffix = "..." if f.get("truncated_preview") else ""
        return (
            f"payload_bytes={pkt.payload_bytes} crc32={f.get('crc32', '')}"
            + (f" preview=0x{preview}{suffix}" if preview else "")
        )

    if not pkt.valid:
        return " ".join(f"{k}={v}" for k, v in f.items())

    op = pkt.opcode
    if op in (GpuOpcode.Nop, GpuOpcode.BeginFrame, GpuOpcode.Present):
        return f"trailing_bytes={f['trailing_bytes']}" if "trailing_bytes" in f else ""
    if op == GpuOpcode.ClearColor:
        return f"rgba=({f['r']:.4f}, {f['g']:.4f}, {f['b']:.4f}, {f['a']:.4f})"
    if op == GpuOpcode.CreateBuffer:
        usage = "|".join(f["usage_names"]) if f["usage_names"] else "NONE"
        return f"id={f['id']} size={f['byte_size']} usage=0x{f['usage_flags']:04x}({usage})"
    if op == GpuOpcode.WriteBuffer:
        preview = f["preview_hex"]
        suffix = "..." if f["truncated_preview"] else ""
        preview_str = f" preview=0x{preview}{suffix}" if preview else ""
        return (
            f"id={f['id']} dst_offset={f['dst_offset']} data_len={f['data_len']} "
            f"crc32={f['crc32']}{preview_str}"
        )
    if op == GpuOpcode.CreateShaderModule:
        return (
            f"id={f['id']} stage={f['stage_name']} code_len={f['code_len']} "
            f"lines={f['line_count']} chars={f['char_count']} fnv1a64={f['wgsl_hash_fnv1a64']}"
        )
    if op == GpuOpcode.Draw:
        return (
            f"vertices={f['vertex_count']} instances={f['instance_count']} "
            f"first_vertex={f['first_vertex']} first_instance={f['first_instance']}"
        )
    if op == GpuOpcode.CreateRenderPipeline:
        topo = f["topology_webgpu"] or f["topology_name"]
        return (
            f"id={f['id']} vs={f['vs_shader_id']} fs={f['fs_shader_id']} "
            f"stride={f['vertex_stride']} topology={topo} flags=0x{f['flags']:x}"
        )
    if op == GpuOpcode.SetPipeline:
        return f"pipeline_id={f['pipeline_id']}"
    if op == GpuOpcode.SetVertexBuffer:
        return f"slot={f['slot']} buffer_id={f['buffer_id']} offset={f['offset']}"
    if op == GpuOpcode.DestroyResource:
        return f"id={f['id']}"
    if op == GpuOpcode.CreateTexture:
        usage = "|".join(f["usage_names"]) if f["usage_names"] else "NONE"
        fmt = f["format_webgpu"] or f["format_name"]
        dim = f["dimension_webgpu"] or f["dimension_name"]
        return (
            f"id={f['id']} {f['width']}x{f['height']}x{f['depth_or_layers']} "
            f"mips={f['mip_levels']} dim={dim} fmt={fmt} "
            f"usage=0x{f['usage_flags']:04x}({usage})"
        )
    if op == GpuOpcode.WriteTexture:
        preview = f["preview_hex"]
        suffix = "..." if f["truncated_preview"] else ""
        preview_str = f" preview=0x{preview}{suffix}" if preview else ""
        return (
            f"texture_id={f['texture_id']} mip={f['mip_level']} "
            f"origin=({f['origin_x']},{f['origin_y']},{f['origin_z']}) "
            f"extent=({f['width']}x{f['height']}x{f['depth_or_layers']}) "
            f"bpr={f['bytes_per_row']} rpi={f['rows_per_image']} "
            f"data_len={f['data_len']} crc32={f['crc32']}{preview_str}"
        )
    if op == GpuOpcode.CreateSampler:
        addr_u = f["address_u_webgpu"] or f["address_u_name"]
        addr_v = f["address_v_webgpu"] or f["address_v_name"]
        addr_w = f["address_w_webgpu"] or f["address_w_name"]
        mag = f["mag_filter_webgpu"] or f["mag_filter_name"]
        min_f = f["min_filter_webgpu"] or f["min_filter_name"]
        mip = f["mipmap_filter_webgpu"] or f["mipmap_filter_name"]
        return (
            f"id={f['id']} addr=({addr_u},{addr_v},{addr_w}) "
            f"filter=(mag={mag},min={min_f},mip={mip}) aniso={f['max_anisotropy']}"
        )
    if op == GpuOpcode.CreateBindGroup:
        entry_parts: list[str] = []
        for entry in f["entries"]:
            base = f"@{entry['binding']}:{entry['resource_type_name']}(#{entry['resource_id']})"
            if entry["resource_type"] == GpuBindResourceType.Buffer and (
                entry["offset"] or entry["size"]
            ):
                size_str = str(entry["size"]) if entry["size"] else "end"
                base += f"[{entry['offset']}:{size_str}]"
            entry_parts.append(base)
        entries_str = ", ".join(entry_parts)
        return (
            f"id={f['id']} pipeline={f['pipeline_id']} group={f['group_index']} "
            f"entries={f['entry_count']} [{entries_str}]"
        )
    if op == GpuOpcode.SetBindGroup:
        return f"group={f['group_index']} bind_group_id={f['bind_group_id']}"
    if op == GpuOpcode.SetIndexBuffer:
        fmt = f["index_format_webgpu"] or f["index_format_name"]
        return f"buffer_id={f['buffer_id']} format={fmt} offset={f['offset']}"
    if op == GpuOpcode.DrawIndexed:
        return (
            f"indices={f['index_count']} instances={f['instance_count']} "
            f"first_index={f['first_index']} base_vertex={f['base_vertex']} "
            f"first_instance={f['first_instance']}"
        )
    if op == GpuOpcode.CreateComputePipeline:
        return f"id={f['id']} cs={f['cs_shader_id']} flags=0x{f['flags']:x}"
    if op == GpuOpcode.SetComputePipeline:
        return f"pipeline_id={f['pipeline_id']}"
    if op == GpuOpcode.DispatchCompute:
        return f"workgroups=({f['workgroup_x']}, {f['workgroup_y']}, {f['workgroup_z']})"
    if op >= GpuOpcode.CreateBindGroupLayout:
        return _format_v2_fields(op, f)
    return " ".join(f"{k}={v}" for k, v in f.items())


def _format_view(view: dict[str, Any]) -> str:
    parts = []
    if view["format"] != "undefined":
        parts.append(view["format"])
    if view["dimension"] != "undefined":
        parts.append(view["dimension"])
    if view["aspect"] != "all":
        parts.append(view["aspect"])
    if view["base_mip_level"] or view["mip_level_count"]:
        parts.append(f"mip={view['base_mip_level']}+{view['mip_level_count'] or 'all'}")
    if view["base_array_layer"] or view["array_layer_count"]:
        parts.append(f"layer={view['base_array_layer']}+{view['array_layer_count'] or 'all'}")
    return f"<{','.join(parts)}>" if parts else ""


def _format_v2_fields(op: int, f: dict[str, Any]) -> str:
    if op in (GpuOpcode.CreateBindGroupLayout, GpuOpcode.CreatePipelineLayout):
        base = f"id={f['id']} content={f['content_id']}"
        if f["manifest_reference"]:
            return base + " manifest-ref"
        if op == GpuOpcode.CreatePipelineLayout:
            return base + f" groups={f['bind_group_layout_ids']}"
        entries = []
        for e in f["entries"]:
            stages = "|".join(e["visibility_names"]) or "NONE"
            extra = ""
            if e.get("has_dynamic_offset"):
                extra += ",dynamic"
            if "view_dimension" in e and e["view_dimension"] != "undefined":
                extra += f",{e['view_dimension']}"
            if "storage_format" in e:
                extra += f",{e['storage_format']}"
            entries.append(f"@{e['binding']}:{e['type']}({e['binding_type']}{extra})[{stages}]")
        return base + f" entries={f['entry_count']} [{', '.join(entries)}]"
    if op == GpuOpcode.CreateRenderPipeline2:
        k = f["key"]
        targets = ",".join(t["format"] if t else "null" for t in k["color_targets"])
        ds = k["depth_stencil"]["format"] if k["depth_stencil"] else "none"
        return (
            f"id={f['id']} vs={f['vs_module_id']} fs={f['fs_module_id']} "
            f"layout={f['pipeline_layout_id']} topology={k['primitive']['topology']} "
            f"targets=[{targets}] depth={ds} samples={k['multisample']['count']} "
            f"vbufs={k['vertex_buffer_count']} attrs={k['vertex_attribute_count']} "
            f"key={k['key_fnv1a64']}"
        )
    if op == GpuOpcode.CreateComputePipeline2:
        return f"id={f['id']} cs={f['cs_module_id']} layout={f['pipeline_layout_id']}"
    if op == GpuOpcode.BeginRenderPass:
        parts = []
        for i, c in enumerate(f["color_attachments"]):
            if c is None:
                parts.append(f"c{i}=null")
                continue
            tex = "swapchain" if c["swapchain"] else f"#{c['texture_id']}"
            load = c["load_op"]
            if load == "clear":
                load += "(" + ",".join(f"{v:.3g}" for v in c["clear_value"]) + ")"
            parts.append(f"c{i}={tex}{_format_view(c['view'])} {load}/{c['store_op']}")
        ds = f["depth_stencil"]
        if ds is not None:
            depth = f"{ds['depth_load_op']}/{ds['depth_store_op']}"
            if ds["depth_load_op"] == "clear":
                depth += f"({ds['depth_clear_value']:.3g})"
            stencil = f"{ds['stencil_load_op']}/{ds['stencil_store_op']}"
            if ds["stencil_load_op"] == "clear":
                stencil += f"({ds['stencil_clear_value']})"
            ro = ""
            if ds["depth_read_only"] or ds["stencil_read_only"]:
                ro = " ro=" + "".join(
                    flag
                    for flag, on in (("D", ds["depth_read_only"]), ("S", ds["stencil_read_only"]))
                    if on
                )
            parts.append(
                f"ds=#{ds['texture_id']}{_format_view(ds['view'])} depth={depth} stencil={stencil}{ro}"
            )
        return f"colors={f['color_count']} " + " ".join(parts)
    if op == GpuOpcode.EndRenderPass:
        return f"trailing_bytes={f['trailing_bytes']}" if "trailing_bytes" in f else ""
    if op == GpuOpcode.SetViewport:
        return (
            f"rect=({f['x']:g},{f['y']:g} {f['width']:g}x{f['height']:g}) "
            f"depth=[{f['min_depth']:g},{f['max_depth']:g}]"
        )
    if op == GpuOpcode.SetScissorRect:
        return f"rect=({f['x']},{f['y']} {f['width']}x{f['height']})"
    if op == GpuOpcode.SetBlendConstant:
        return f"rgba=({f['r']:.4f}, {f['g']:.4f}, {f['b']:.4f}, {f['a']:.4f})"
    if op == GpuOpcode.SetStencilReference:
        return f"reference={f['reference']}"
    if op == GpuOpcode.CreateBindGroup2:
        entries = []
        for e in f["entries"]:
            res = "null" if e["null_binding"] else f"#{e['resource_id']}"
            base = f"@{e['binding']}:{e['resource_type_name']}({res})"
            if e["resource_type"] == GpuBindResourceType.Buffer and (e["offset"] or e["size"]):
                base += f"[{e['offset']}:{e['size'] or 'end'}]"
            if "view" in e:
                base += _format_view(e["view"])
            entries.append(base)
        return (
            f"id={f['id']} layout={f['layout_id']} entries={f['entry_count']} "
            f"[{', '.join(entries)}]"
        )
    if op == GpuOpcode.SetBindGroup2:
        return (
            f"group={f['group_index']} bind_group_id={f['bind_group_id']} "
            f"dynamic_offsets={f['dynamic_offsets']}"
        )
    if op == GpuOpcode.CreateSampler2:
        cmp = f" compare={f['compare']}" if f["compare"] != "undefined" else ""
        return (
            f"id={f['id']} addr=({f['address_u']},{f['address_v']},{f['address_w']}) "
            f"filter=(mag={f['mag_filter']},min={f['min_filter']},mip={f['mipmap_filter']}) "
            f"aniso={f['max_anisotropy']} lod=[{f['lod_min_clamp']:g},{f['lod_max_clamp']:g}]{cmp}"
        )
    return " ".join(f"{k}={v}" for k, v in f.items())


def format_trace_text(trace: DecodedTrace, *, stats_only: bool = False) -> str:
    """Format a DecodedTrace as human-readable disassembly and/or statistics."""
    lines: list[str] = [
        f"=== playgta6 GPU Trace (ABI v{trace.abi_version}) ===",
        (
            f"Bytes: {trace.consumed_bytes}/{trace.total_bytes} | "
            f"Packets: {trace.packet_count} | Frames: {trace.frame_count} | "
            f"Draws: {trace.draw_calls} | Dispatches: {trace.compute_dispatches} | "
            f"Unknown: {trace.unknown_opcode_count} | "
            f"Diagnostics: {len(trace.diagnostics)}"
        ),
    ]

    if not stats_only:
        lines.append("")
        lines.append("Packets:")
        if not trace.packets:
            lines.append("  (none)")
        for pkt in trace.packets:
            flag_part = f" flags=0x{pkt.flags:04x}" if pkt.flags else ""
            status_part = "" if pkt.valid else " [INVALID]"
            detail = _format_packet_fields(pkt)
            detail_part = f"  {detail}" if detail else ""
            lines.append(
                f"  #{pkt.index:04d} @0x{pkt.offset:06x}  "
                f"{pkt.name:<22s} ({pkt.payload_bytes:5d} B){flag_part}{status_part}{detail_part}"
            )

    lines.append("")
    lines.append("Frames:")
    if not trace.frames:
        lines.append("  (none)")
    for frame in trace.frames:
        status = "complete" if frame.complete else "incomplete"
        if not frame.has_begin_frame:
            status += ",no-begin"
        hist_str = ", ".join(f"{k}={v}" for k, v in sorted(frame.opcode_histogram.items()))
        extras: list[str] = []
        if frame.draw_indexed_calls:
            extras.append(f"indexed={frame.draw_indexed_calls}")
        if frame.compute_dispatches:
            extras.append(f"dispatches={frame.compute_dispatches}")
        if frame.compute_passes:
            extras.append(f"compute_passes={frame.compute_passes}")
        if frame.textures_created:
            extras.append(f"textures={frame.textures_created}")
        if frame.bind_groups_created:
            extras.append(f"bind_groups={frame.bind_groups_created}")
        extra_str = (" " + " ".join(extras)) if extras else ""
        lines.append(
            f"  Frame {frame.frame_index:03d} [{status}]: "
            f"packets={frame.packet_count} (#{frame.start_packet_index}..#{frame.end_packet_index}) "
            f"draws={frame.draw_calls}{extra_str} bytes={frame.bytes} | {hist_str}"
        )

    lines.append("")
    lines.append("Opcode Histogram:")
    if not trace.opcode_histogram:
        lines.append("  (none)")
    for op_name, count in sorted(trace.opcode_histogram.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {op_name:<22s}: {count}")

    if trace.diagnostics:
        lines.append("")
        lines.append("Diagnostics:")
        for diag in trace.diagnostics:
            pkt_str = f" packet=#{diag.packet_index}" if diag.packet_index is not None else ""
            lines.append(
                f"  [{diag.severity.upper()}] {diag.code} @0x{diag.offset:06x}{pkt_str}: "
                f"{diag.message}"
            )

    return "\n".join(lines)


def trace_to_json(
    trace: DecodedTrace,
    *,
    stats_only: bool = False,
    indent: int = 2,
) -> str:
    """Serialize a DecodedTrace to formatted JSON."""
    return json.dumps(trace.to_dict(stats_only=stats_only), indent=indent)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Disassemble and inspect a playgta6 binary GPU command stream."
    )
    parser.add_argument(
        "trace_file",
        help="Path to raw binary GPU command stream (or '-' to read from stdin).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit structured JSON instead of text output.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Emit summary statistics, frame breakdown, and opcode histogram only.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail with non-zero exit code on unknown opcodes (>= 35), "
            "size mismatches, or malformed/truncated packets."
        ),
    )
    parser.add_argument(
        "--preview-bytes",
        type=int,
        default=DEFAULT_PREVIEW_BYTES,
        metavar="N",
        help=(
            "Maximum raw payload preview bytes for WriteBuffer/WriteTexture/Unknown "
            f"(default: {DEFAULT_PREVIEW_BYTES})."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.trace_file == "-":
        raw_bytes = sys.stdin.buffer.read()
    else:
        path = Path(args.trace_file)
        if not path.is_file():
            print(f"Error: trace file not found: {path}", file=sys.stderr)
            return 2
        raw_bytes = path.read_bytes()

    trace = decode_stream(
        raw_bytes,
        strict=args.strict,
        preview_bytes=args.preview_bytes,
    )

    if args.as_json:
        print(trace_to_json(trace, stats_only=args.stats))
    else:
        print(format_trace_text(trace, stats_only=args.stats))

    if args.strict and not trace.ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
