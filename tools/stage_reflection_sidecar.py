#!/usr/bin/env python3
"""Write the stage-reflection sidecar (`<pack>.reflect.json`, G62 / #651) from the translator's
JSON output.

Input: one or more JSON files, each either a `playgta6-shaders translate-rdna2 --json` result (one
stage) or a `playgta6-shaders pack-rdna2 --json` summary (its `entries` list). A stage enters the
sidecar only when its reflection carries the join keys the runtime needs:
  - every binding has a `source` ({user_sgpr, table_byte_offset} | {user_sgpr, root_constants} |
    {user_sgpr, root_descriptor} | {static_sampler} | {sampler_heap | resource_heap: {base_user_sgpr,
    stride, index: {user_sgpr, shift}}} (S34: a directly indexed heap, index from a root constant) |
    {inline_sampler: {sdwords: [d0, d1, d2, d3]}} (#482: the four GFX10 S# dwords of a sampler the
    stage builds from constants; the runtime decodes them) |
    {cbv_indexed_resource_heap | cbv_indexed_sampler_heap: {cbv: {user_sgpr, root_descriptor: true} |
    {user_sgpr, table_byte_offset} | {user_sgpr, root_constants: true}, byte_offset, shift,
    bit_offset?, mask?, stride?}} (G62 / #482: SM 6.6 bindless, the heap index is the dword at
    byte_offset of a constant buffer, read by the runtime at draw time) |
    {lane_indexed_resource_heap | lane_indexed_sampler_heap: {max_count (1..256), shift, stride?,
    window_base?: {cbv: <as above>, byte_offset, bit_offset?, mask?}}} (G62 / #482: per-lane
    (VGPR / waterfall) bindless, a window of max_count heap descriptors read as a binding_array;
    kept and counted, the runtime refuses the join with a counted reason until it binds arrays) |
    {unresolved: true | "<reason>"} (G62 / #482, sidecar version 3; added with --unresolved-sources: the translator
    reflected the binding but could not trace its descriptor, e.g. a bindless heap index read from a
    constant buffer at an offset computed from a root constant; the runtime binds the worker's
    default for it and joins the rest of the stage; writable kinds stay skipped));
  - a vertex stage that fetches attributes lists `vertex_inputs` ({location, vb_slot, byte_offset,
    components, component_type, fetch_format?}).
Stages without them are skipped and counted, so the runtime keeps its conventions for them.

    python tools/stage_reflection_sidecar.py <json>... --out <pack>.reflect.json [--unresolved-sources] [--merge-existing <reflect.json>] [--diff-baseline <reflect.json>] [--fail-on-regressions] [--report-missing-sources <report.json>] [--verify-pack <wgsl.pack>] [--json]

The sidecar is derived from the title's shaders: write it under work/, never into the repository.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any

SCHEMA = "playgta6-stage-reflection"
VERSION = 1
VERSION_ROOT_TABLE_OFFSET = 2
VERSION_UNRESOLVED = 3
MAX_VERSION = 3
SUPPORTED_VERSIONS = (1, 2, 3)
REPORT_SCHEMA = "playgta6-stage-reflection-missing-sources"
REPORT_VERSION = 1
DEFAULT_MAX_SAMPLE_HASHES = 5
STAGES = ("vertex", "fragment", "compute")

# WGPK v1 binary container constants (matching shaders/src/pack.rs byte-for-byte)
WGPK_MAGIC = b"WGPK"
WGPK_VERSION = 1
WGPK_HEADER_STRUCT = struct.Struct("<4sIII")
WGPK_HEADER_BYTES = WGPK_HEADER_STRUCT.size  # 16 bytes
WGPK_ENTRY_STRUCT = struct.Struct("<QB3sII")
WGPK_ENTRY_BYTES = WGPK_ENTRY_STRUCT.size  # 20 bytes
WGPK_VERSION_V2 = 2
WGPK_ENTRY_STRUCT_V2 = struct.Struct("<QB3sIQ")  # hash, stage, reserved, length, offset (u64)
WGPK_ENTRY_BYTES_V2 = WGPK_ENTRY_STRUCT_V2.size  # 24 bytes
WGPK_U32_MAX = 0xFFFFFFFF
WGPK_U64_MAX = 0xFFFFFFFFFFFFFFFF
WGPK_STAGE_NAMES: dict[int, str] = {
    1: "vertex",
    2: "fragment",
    3: "compute",
}

_SAFE_LABEL_RE = re.compile(r"^[A-Za-z0-9_.$<>\[\]-]{1,64}$")
_SAFE_HASH_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")

KIND_MAP = {
    "uniform": "uniform",
    "storage": "storage",
    "read_only_storage": "read_only_storage",
    "read-only-storage": "read_only_storage",
    "texture": "texture",
    "texture_2d": "texture",
    "storage_texture": "storage_texture",
    "storage-texture": "storage_texture",
    "sampler": "sampler",
    "comparison_sampler": "comparison_sampler",
    "sampler_comparison": "comparison_sampler",
}


# Kinds that may join with an unresolved source (bound as the worker's default): read-only ones.
# A writable binding without a source stays skipped, its writes would go nowhere.
UNRESOLVABLE_KINDS = ("uniform", "read_only_storage", "texture", "sampler", "comparison_sampler")
UNRESOLVED_SOURCE = {"unresolved": True}


def stage_name(stage: str) -> str | None:
    return {"vertex": "vertex", "fragment": "fragment", "pixel": "fragment", "compute": "compute"}.get(stage)


def heap_source_ok(source: dict[str, Any]) -> bool | None:
    """None: not a heap source; else whether it has base_user_sgpr, stride, index {user_sgpr, shift}."""
    heap = source.get("sampler_heap", source.get("resource_heap"))
    if heap is None:
        return None
    if not isinstance(heap, dict) or not isinstance(heap.get("index"), dict):
        return False
    ints = (heap.get("base_user_sgpr"), heap.get("stride"), heap["index"].get("user_sgpr"), heap["index"].get("shift"))
    return all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in ints) and ints[3] < 32


def _u32(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 0xFFFFFFFF


MAX_LANE_INDEXED_WINDOW = 256
MAX_MESH_VERTICES = 256
MAX_MESH_PRIMITIVES = 256
MAX_MESH_ATTRIBUTES = 15
MAX_MESH_VERTEX_WORDS = 4 + 4 * MAX_MESH_ATTRIBUTES


def mesh_output_ok(mesh_output: Any) -> bool:
    """Whether `mesh_output` follows the G66 / S62 contract (#482, #1537):
    max_vertices and max_primitives in 1..256, vertex_words = 4 + 4 * n (0 <= n <= 15)."""
    if not isinstance(mesh_output, dict):
        return False
    mv = mesh_output.get("max_vertices")
    mp = mesh_output.get("max_primitives")
    vw = mesh_output.get("vertex_words")
    if not (_u32(mv) and _u32(mp) and _u32(vw)):
        return False
    return (
        1 <= mv <= MAX_MESH_VERTICES
        and 1 <= mp <= MAX_MESH_PRIMITIVES
        and 4 <= vw <= MAX_MESH_VERTEX_WORDS
        and vw % 4 == 0
    )


def _cb_dword_ok(heap: dict[str, Any]) -> bool:
    """Whether `heap` names a CB dword: cbv {user_sgpr, one of root_descriptor / root_constants /
    table_byte_offset}, a dword-aligned byte_offset, optional bit_offset < 32 and non-zero mask."""
    if not isinstance(heap.get("cbv"), dict):
        return False
    cbv = heap["cbv"]
    where = [cbv.get("root_descriptor") is True, cbv.get("root_constants") is True, "table_byte_offset" in cbv]
    if not _u32(cbv.get("user_sgpr")) or sum(where) != 1:
        return False
    if where[2] and (not _u32(cbv["table_byte_offset"]) or cbv["table_byte_offset"] % 32 != 0):
        return False
    byte_offset = heap.get("byte_offset")
    if not _u32(byte_offset) or byte_offset % 4 != 0:
        return False
    bit_offset, mask = heap.get("bit_offset", 0), heap.get("mask", 0xFFFFFFFF)
    return _u32(bit_offset) and bit_offset < 32 and _u32(mask) and mask != 0


def lane_heap_source_ok(source: dict[str, Any]) -> bool | None:
    """None: not a lane-indexed heap source; else whether it follows the G62 contract (#482)."""
    heap = source.get("lane_indexed_resource_heap", source.get("lane_indexed_sampler_heap"))
    if heap is None:
        return None
    if not isinstance(heap, dict):
        return False
    count, shift, stride = heap.get("max_count"), heap.get("shift"), heap.get("stride", 0)
    if not _u32(count) or not 1 <= count <= MAX_LANE_INDEXED_WINDOW:
        return False
    if not _u32(shift) or shift >= 32 or not _u32(stride):
        return False
    base = heap.get("window_base")
    return base is None or (isinstance(base, dict) and _cb_dword_ok(base))


def cbv_heap_source_ok(source: dict[str, Any]) -> bool | None:
    """None: not a CB-indexed heap source; else whether it follows the G62 contract (#482)."""
    heap = source.get("cbv_indexed_resource_heap", source.get("cbv_indexed_sampler_heap"))
    if heap is None:
        return None
    if not isinstance(heap, dict) or not isinstance(heap.get("cbv"), dict):
        return False
    cbv = heap["cbv"]
    where = [cbv.get("root_descriptor") is True, cbv.get("root_constants") is True, "table_byte_offset" in cbv]
    if not _u32(cbv.get("user_sgpr")) or sum(where) != 1:
        return False
    if where[2] and (not _u32(cbv["table_byte_offset"]) or cbv["table_byte_offset"] % 32 != 0):
        return False
    byte_offset, shift = heap.get("byte_offset"), heap.get("shift")
    if not _u32(byte_offset) or byte_offset % 4 != 0 or not _u32(shift) or shift >= 32:
        return False
    bit_offset, mask, stride = heap.get("bit_offset", 0), heap.get("mask", 0xFFFFFFFF), heap.get("stride", 0)
    return _u32(bit_offset) and bit_offset < 32 and _u32(mask) and mask != 0 and _u32(stride)


def root_descriptor_source_ok(source: dict[str, Any]) -> bool | None:
    """None: not a root_descriptor source; else whether user_sgpr and optional v2 table_byte_offset
    (multiple of 32, G62 / #1471 / #482) are valid."""
    if source.get("root_descriptor") is not True:
        return None
    if not _u32(source.get("user_sgpr")):
        return False
    if "table_byte_offset" in source:
        tbo = source["table_byte_offset"]
        return _u32(tbo) and tbo % 32 == 0
    return True


def _stages_require_v2(stages: dict[str, Any]) -> bool:
    """Return True when any stage binding carries a v2-only root_descriptor table_byte_offset."""
    for stage_obj in stages.values():
        if not isinstance(stage_obj, dict):
            continue
        bindings = stage_obj.get("bindings")
        if not isinstance(bindings, list):
            continue
        for binding in bindings:
            if not isinstance(binding, dict):
                continue
            source = binding.get("source")
            if isinstance(source, dict) and source.get("root_descriptor") is True and "table_byte_offset" in source:
                return True
    return False


def is_unresolved_source(source: Any) -> bool:
    return isinstance(source, dict) and "unresolved" in source


def unresolved_source_ok(source: dict[str, Any], kind: Any) -> bool | None:
    """None: not an unresolved source; else whether it is exactly {unresolved: true} on a read-only kind."""
    if "unresolved" not in source:
        return None
    if len(source) != 1 or KIND_MAP.get(str(kind)) not in UNRESOLVABLE_KINDS:
        return False
    mark = source["unresolved"]
    return mark is True or (isinstance(mark, str) and _SAFE_LABEL_RE.match(mark) is not None)


def _stages_require_v3(stages: dict[str, Any]) -> bool:
    """Return True when any stage binding carries a v3-only unresolved source."""
    for stage_obj in stages.values():
        if not isinstance(stage_obj, dict) or not isinstance(stage_obj.get("bindings"), list):
            continue
        for binding in stage_obj["bindings"]:
            if isinstance(binding, dict) and is_unresolved_source(binding.get("source")):
                return True
    return False


def inline_sampler_ok(source: dict[str, Any]) -> bool | None:
    """None: not an inline sampler; else whether sdwords is four 32-bit unsigned integers."""
    if "inline_sampler" not in source:
        return None
    sampler = source["inline_sampler"]
    words = sampler.get("sdwords") if isinstance(sampler, dict) else None
    return (isinstance(words, list) and len(words) == 4 and
            all(isinstance(w, int) and not isinstance(w, bool) and 0 <= w <= 0xFFFFFFFF for w in words))


def convert(entry: dict[str, Any], *, unresolved: bool = False) -> tuple[str | None, dict[str, Any] | None, str]:
    """(key, sidecar stage, reason it was skipped). With `unresolved`, a read-only binding without a
    source is kept with {"unresolved": true} instead of skipping the stage."""
    key = entry.get("hash_fnv1a64")
    stage = stage_name(str(entry.get("stage", "")))
    if not key or stage is None:
        return None, None, "no hash or stage"
    out: dict[str, Any] = {"stage": stage, "bindings": []}
    for b in entry.get("bindings", []):
        source = b.get("source")
        if not isinstance(source, dict):
            if not unresolved:
                return key, None, "binding without source"
            if KIND_MAP.get(str(b.get("kind", ""))) not in UNRESOLVABLE_KINDS:
                return key, None, "writable binding without source"
            source = dict(UNRESOLVED_SOURCE)
        if unresolved_source_ok(source, b.get("kind")) is False:
            return key, None, "malformed unresolved source"
        if root_descriptor_source_ok(source) is False:
            return key, None, "malformed root_descriptor source"
        if heap_source_ok(source) is False:
            return key, None, "malformed heap source"
        if inline_sampler_ok(source) is False:
            return key, None, "malformed inline sampler"
        if cbv_heap_source_ok(source) is False:
            return key, None, "malformed CB-indexed heap source"
        if lane_heap_source_ok(source) is False:
            return key, None, "malformed lane-indexed heap source"
        kind = KIND_MAP.get(str(b.get("kind", "")))
        if kind is None:
            return key, None, f"unknown binding kind {b.get('kind')!r}"
        item: dict[str, Any] = {"group": b.get("group", 0), "binding": b["binding"], "kind": kind, "source": source}
        for field in ("view_dimension", "sample_type", "multisampled", "storage_format", "min_binding_size_bytes"):
            if field in b:
                item[field] = b[field]
        out["bindings"].append(item)
    summary = entry.get("binding_summary") or {}
    fetches_attributes = summary.get("vertex_input_mode") == "attributes"
    if stage == "vertex" and fetches_attributes:
        inputs = entry.get("vertex_inputs")
        if not isinstance(inputs, list):
            return key, None, "vertex stage without vertex_inputs"
        out["vertex_inputs"] = inputs
    if "mesh_output" in entry and entry["mesh_output"] is not None:
        mesh_output = entry["mesh_output"]
        if stage != "compute" or not mesh_output_ok(mesh_output):
            return key, None, "malformed mesh_output"
        out["mesh_output"] = {
            "max_vertices": mesh_output["max_vertices"],
            "max_primitives": mesh_output["max_primitives"],
            "vertex_words": mesh_output["vertex_words"],
        }
    return key, out, ""


def entries_of(doc: Any) -> list[dict[str, Any]]:
    if isinstance(doc, dict) and isinstance(doc.get("entries"), list):
        return [e for e in doc["entries"] if isinstance(e, dict)]
    if isinstance(doc, dict) and "hash_fnv1a64" in doc:
        return [doc]
    return []


def _sanitize_label(value: Any, *, fallback: str = "") -> str:
    if not isinstance(value, str):
        return fallback
    text = value.strip()
    if not text:
        return fallback
    if _SAFE_LABEL_RE.match(text):
        return text
    return "<redacted>"


def _sanitize_hash(value: Any) -> str:
    if value is None:
        return "<redacted>"
    text = str(value).strip()
    if text and _SAFE_HASH_RE.match(text):
        return text
    return "<redacted>"


def _sanitize_path_label(path: Path | str | None) -> str:
    """Return a strictly sanitized basename-only label (no host paths or usernames)."""
    if path is None:
        return "<unknown>"
    raw = str(path).strip()
    if not raw:
        return "<unknown>"
    stripped = raw.rstrip("/\\")
    if not stripped:
        return "<unknown>"
    parts = re.split(r"[/\\]+", stripped)
    base = parts[-1]
    if not base or base in (".", "..") or (len(base) == 2 and base[1] == ":"):
        return "<unknown>"
    return _sanitize_label(base, fallback="<redacted>")


def _normalize_hash_key(key: str) -> str:
    """Canonicalize valid 64-bit hex shader hashes to 0x + 16 lowercase hex digits."""
    trimmed = key.strip()
    raw_hex = trimmed[2:] if trimmed.lower().startswith("0x") else trimmed
    if raw_hex and len(raw_hex) <= 16 and _HEX_RE.fullmatch(raw_hex):
        return f"0x{int(raw_hex, 16):016x}"
    return key


def _wgpk_entry_bytes(version: int) -> int:
    return WGPK_ENTRY_BYTES_V2 if version == WGPK_VERSION_V2 else WGPK_ENTRY_BYTES


def _validate_wgpk_header(
    header: bytes | bytearray | memoryview, total_len: int
) -> tuple[int, int, int]:
    """Validate a 16-byte WGPK (v1 or v2) header against total_len; returns (version, entry_count, data_start)."""
    if total_len < WGPK_HEADER_BYTES or len(header) < WGPK_HEADER_BYTES:
        raise ValueError(
            f"WGPK pack truncated: {total_len} bytes (minimum header size: {WGPK_HEADER_BYTES})"
        )
    magic, version, entry_count, _flags = WGPK_HEADER_STRUCT.unpack_from(header, 0)
    if magic != WGPK_MAGIC:
        magic_hex = " ".join(f"0x{b:02x}" for b in magic)
        raise ValueError(f"invalid WGPK pack magic [{magic_hex}] (expected 'WGPK')")
    if version not in (WGPK_VERSION, WGPK_VERSION_V2):
        raise ValueError(f"unsupported WGPK pack version {version} (expected {WGPK_VERSION} or {WGPK_VERSION_V2})")
    if version == WGPK_VERSION and total_len > WGPK_U32_MAX:
        raise ValueError(f"WGPK pack size {total_len} exceeds u32 max ({WGPK_U32_MAX})")
    table_bytes = entry_count * _wgpk_entry_bytes(version)
    data_start = WGPK_HEADER_BYTES + table_bytes
    if version == WGPK_VERSION and data_start > WGPK_U32_MAX:
        raise ValueError(f"WGPK entry table size overflows u32 for {entry_count} entries")
    if total_len < data_start:
        raise ValueError(
            f"WGPK pack truncated: {total_len} bytes "
            f"(header + {entry_count} entries require {data_start} bytes)"
        )
    return version, entry_count, data_start


def _parse_wgpk_toc_entries(
    toc_bytes: bytes | bytearray | memoryview,
    *,
    version: int = WGPK_VERSION,
    entry_count: int,
    data_start: int,
    total_len: int,
) -> list[dict[str, Any]]:
    """Decode and bounds-check all WGPK TOC records (v1: 20-byte, v2: 24-byte with a u64 offset)."""
    entries: list[dict[str, Any]] = []
    seen_hashes: dict[int, int] = {}
    spans: list[tuple[int, int, int]] = []
    entry_bytes = _wgpk_entry_bytes(version)
    limit = WGPK_U64_MAX if version == WGPK_VERSION_V2 else WGPK_U32_MAX

    for idx in range(entry_count):
        base = idx * entry_bytes
        if version == WGPK_VERSION_V2:
            hash_int, stage_code, _reserved, length, offset = WGPK_ENTRY_STRUCT_V2.unpack_from(toc_bytes, base)
        else:
            hash_int, stage_code, _reserved, offset, length = WGPK_ENTRY_STRUCT.unpack_from(toc_bytes, base)
        stage = WGPK_STAGE_NAMES.get(stage_code)
        if stage is None:
            raise ValueError(f"invalid WGPK shader stage code {stage_code} at entry {idx} (expected 1..3)")
        end = offset + length
        if end > limit:
            raise ValueError(f"WGPK entry {idx} payload [{offset}..+{length}] overflows u{64 if limit == WGPK_U64_MAX else 32}")
        if offset < data_start:
            raise ValueError(
                f"WGPK entry {idx} payload offset {offset} overlaps header/TOC (data_start={data_start})"
            )
        if end > total_len:
            raise ValueError(
                f"WGPK entry {idx} payload [{offset}..+{length}] exceeds pack bounds ({total_len} bytes)"
            )
        if hash_int in seen_hashes:
            raise ValueError(
                f"WGPK pack contains duplicate shader hash 0x{hash_int:016x} "
                f"at entries {seen_hashes[hash_int]} and {idx}"
            )
        seen_hashes[hash_int] = idx
        if length > 0:
            spans.append((offset, end, idx))
        entries.append({
            "hash": f"0x{hash_int:016x}",
            "stage": stage,
            "offset": offset,
            "length": length,
        })

    if len(spans) > 1:
        spans.sort(key=lambda item: (item[0], item[1]))
        for prev, curr in zip(spans, spans[1:]):
            if curr[0] < prev[1]:
                raise ValueError(
                    f"WGPK entry {prev[2]} payload [{prev[0]}..{prev[1]}) "
                    f"overlaps entry {curr[2]} payload [{curr[0]}..{curr[1]})"
                )

    return entries


def parse_wgpk_toc_bytes(data: bytes | bytearray | memoryview) -> list[dict[str, Any]]:
    """Parse and bounds-check a WGPK v1 header and TOC from an in-memory buffer."""
    view = memoryview(data)
    total_len = len(view)
    version, entry_count, data_start = _validate_wgpk_header(view, total_len)
    return _parse_wgpk_toc_entries(
        view[WGPK_HEADER_BYTES:data_start],
        version=version,
        entry_count=entry_count,
        data_start=data_start,
        total_len=total_len,
    )


def load_wgpk_toc(path: Path) -> list[dict[str, Any]]:
    """Read and validate a WGPK v1 header and TOC from disk without loading WGSL payloads."""
    label = _sanitize_path_label(path)
    if not path.exists():
        raise ValueError(f"WGPK pack {label} does not exist")
    if path.is_dir():
        raise ValueError(f"WGPK pack {label} is a directory")
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            total_len = f.tell()
            if total_len < WGPK_HEADER_BYTES:
                raise ValueError(
                    f"WGPK pack truncated: {total_len} bytes (minimum header size: {WGPK_HEADER_BYTES})"
                )
            f.seek(0)
            header = f.read(WGPK_HEADER_BYTES)
            version, entry_count, data_start = _validate_wgpk_header(header, total_len)
            table_bytes = data_start - WGPK_HEADER_BYTES
            toc_bytes = f.read(table_bytes)
            if len(toc_bytes) < table_bytes:
                raise ValueError(
                    f"WGPK pack truncated while reading TOC: got {len(toc_bytes)} of {table_bytes} bytes"
                )
    except OSError as exc:
        reason = exc.strerror or type(exc).__name__
        raise ValueError(f"failed to read WGPK pack {label}: {reason}") from None
    return _parse_wgpk_toc_entries(
        toc_bytes,
        version=version,
        entry_count=entry_count,
        data_start=data_start,
        total_len=total_len,
    )


def verify_sidecar_against_pack(
    pack_entries: list[dict[str, Any]],
    stages: dict[str, Any],
    *,
    pack_label: str | Path = "<pack>",
    max_samples: int = DEFAULT_MAX_SAMPLE_HASHES,
) -> dict[str, Any]:
    """Cross-check reflection sidecar stage hashes against a parsed WGPK v1 TOC."""
    safe_pack_label = _sanitize_path_label(pack_label)
    pack_by_hash: dict[str, str] = {}
    pack_by_stage: dict[str, int] = {stage: 0 for stage in STAGES}
    for entry in pack_entries:
        h = entry["hash"]
        stage = entry["stage"]
        pack_by_hash[h] = stage
        if stage in pack_by_stage:
            pack_by_stage[stage] += 1

    reflection_by_hash: dict[str, str | None] = {}
    for raw_key, stage_obj in stages.items():
        norm_key = _normalize_hash_key(raw_key)
        stage_val = stage_obj.get("stage") if isinstance(stage_obj, dict) else None
        reflection_by_hash[norm_key] = stage_val if isinstance(stage_val, str) else None

    pack_keys = set(pack_by_hash.keys())
    reflection_keys = set(reflection_by_hash.keys())

    matched_keys = sorted(pack_keys & reflection_keys)
    missing_keys = sorted(pack_keys - reflection_keys)
    extra_keys = sorted(reflection_keys - pack_keys)

    missing_by_stage: dict[str, int] = {stage: 0 for stage in STAGES}
    for key in missing_keys:
        stage = pack_by_hash[key]
        if stage in missing_by_stage:
            missing_by_stage[stage] += 1

    stage_mismatches = sum(
        1
        for key in matched_keys
        if reflection_by_hash.get(key) is not None and reflection_by_hash[key] != pack_by_hash[key]
    )

    pack_entry_count = len(pack_entries)
    reflection_stage_count = len(stages)
    matched_count = len(matched_keys)
    coverage_pct = round((matched_count * 100.0) / pack_entry_count, 2) if pack_entry_count > 0 else 0.0

    return {
        "pack_label": safe_pack_label,
        "pack_entry_count": pack_entry_count,
        "reflection_stage_count": reflection_stage_count,
        "matched_hashes": matched_count,
        "missing_reflection_hashes": {
            "count": len(missing_keys),
            "samples": _sample_hashes(missing_keys, max_samples=max_samples),
        },
        "extra_reflection_hashes": {
            "count": len(extra_keys),
            "samples": _sample_hashes(extra_keys, max_samples=max_samples),
        },
        "coverage_pct": coverage_pct,
        "pack_by_stage": pack_by_stage,
        "missing_by_stage": missing_by_stage,
        "stage_mismatches": stage_mismatches,
    }


def _load_sidecar(path: Path, *, label: str, allow_missing: bool) -> dict[str, Any]:
    """Load and validate a v1/v2/v3 sidecar JSON file."""
    if not path.exists():
        if allow_missing:
            return {}
        raise ValueError(f"{label} file does not exist")
    if path.is_dir():
        raise ValueError(f"{label} path is a directory")
    try:
        with path.open(encoding="utf-8-sig") as f:
            doc = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {label}: {exc.msg}") from exc
    if not isinstance(doc, dict):
        raise ValueError(f"{label} must be a JSON object")
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{label} schema must be {SCHEMA!r}, got {_sanitize_label(doc.get('schema'), fallback='<invalid>')!r}")
    version = doc.get("version")
    if type(version) is not int or version not in SUPPORTED_VERSIONS:
        raise ValueError(f"{label} version must be in {SUPPORTED_VERSIONS}, got {version!r}")
    stages = doc.get("stages")
    if not isinstance(stages, dict):
        raise ValueError(f"{label} 'stages' must be an object")
    validated: dict[str, Any] = {}
    for k, v in stages.items():
        if not isinstance(k, str) or not k:
            raise ValueError(f"{label} stage keys must be non-empty strings")
        safe_k = _sanitize_hash(k)
        if not isinstance(v, dict):
            raise ValueError(f"{label} stage {safe_k!r} must be an object")
        if v.get("stage") not in STAGES:
            raise ValueError(
                f"{label} stage {safe_k!r} has invalid stage "
                f"{_sanitize_label(v.get('stage'), fallback='<invalid>')!r}"
            )
        bindings = v.get("bindings")
        if not isinstance(bindings, list):
            raise ValueError(f"{label} stage {safe_k!r} must have a 'bindings' list")
        if version >= 2:
            for b in bindings:
                if isinstance(b, dict) and isinstance(b.get("source"), dict):
                    if root_descriptor_source_ok(b["source"]) is False:
                        raise ValueError(
                            f"{label} stage {safe_k!r} has malformed v2 root_descriptor source"
                        )
        for b in bindings:
            if isinstance(b, dict) and is_unresolved_source(b.get("source")):
                if version < VERSION_UNRESOLVED or not unresolved_source_ok(b["source"], b.get("kind")):
                    raise ValueError(f"{label} stage {safe_k!r} has malformed unresolved source")
        if "mesh_output" in v:
            if v.get("stage") != "compute" or not mesh_output_ok(v["mesh_output"]):
                raise ValueError(f"{label} stage {safe_k!r} has malformed mesh_output")
        validated[k] = v
    return validated


def load_existing_sidecar(path: Path) -> dict[str, Any]:
    """Load and validate an existing v1 sidecar JSON; returns {} if the file does not exist."""
    return _load_sidecar(path, label="existing sidecar", allow_missing=True)


def load_baseline_sidecar(path: Path) -> dict[str, Any]:
    """Load and validate a baseline v1 sidecar JSON; raises ValueError if missing or invalid."""
    return _load_sidecar(path, label="baseline sidecar", allow_missing=False)


def _is_vertex_index_stage(stage_obj: dict[str, Any]) -> bool:
    """Return True when a vertex stage uses vertex_index (no vertex attribute inputs)."""
    if stage_obj.get("stage") != "vertex":
        return False
    raw_mode = stage_obj.get("vertex_input_mode")
    if isinstance(raw_mode, str) and raw_mode.strip():
        return raw_mode.strip().lower() == "vertex_index"
    inputs = stage_obj.get("vertex_inputs")
    return not isinstance(inputs, list) or len(inputs) == 0


def summarize_stages(stages: dict[str, Any]) -> dict[str, Any]:
    """Compute sanitized breakdown telemetry for a sidecar's stages map."""
    by_stage: dict[str, int] = {stage: 0 for stage in STAGES}
    inline_sampler_stages = 0
    inline_sampler_bindings = 0
    resource_heap_stages = 0
    resource_heap_bindings = 0
    cbv_indexed_heap_stages = 0
    cbv_indexed_heap_bindings = 0
    lane_indexed_heap_stages = 0
    lane_indexed_heap_bindings = 0
    vertex_index_stages = 0
    mesh_output_stages = 0
    unresolved_stages = 0
    unresolved_bindings = 0

    for stage_obj in stages.values():
        if not isinstance(stage_obj, dict):
            continue
        stage = stage_obj.get("stage")
        if stage in by_stage:
            by_stage[stage] += 1

        has_inline_sampler = False
        has_resource_heap = False
        has_cbv_indexed_heap = False
        has_lane_indexed_heap = False
        has_unresolved = False
        bindings = stage_obj.get("bindings")
        if isinstance(bindings, list):
            for binding in bindings:
                if not isinstance(binding, dict):
                    continue
                source = binding.get("source")
                if not isinstance(source, dict):
                    continue
                if "inline_sampler" in source:
                    inline_sampler_bindings += 1
                    has_inline_sampler = True
                if "resource_heap" in source:
                    resource_heap_bindings += 1
                    has_resource_heap = True
                if "cbv_indexed_resource_heap" in source or "cbv_indexed_sampler_heap" in source:
                    cbv_indexed_heap_bindings += 1
                    has_cbv_indexed_heap = True
                if "lane_indexed_resource_heap" in source or "lane_indexed_sampler_heap" in source:
                    lane_indexed_heap_bindings += 1
                    has_lane_indexed_heap = True
                if "unresolved" in source:
                    unresolved_bindings += 1
                    has_unresolved = True

        if has_inline_sampler:
            inline_sampler_stages += 1
        if has_resource_heap:
            resource_heap_stages += 1
        if has_cbv_indexed_heap:
            cbv_indexed_heap_stages += 1
        if has_lane_indexed_heap:
            lane_indexed_heap_stages += 1
        if has_unresolved:
            unresolved_stages += 1
        if _is_vertex_index_stage(stage_obj):
            vertex_index_stages += 1
        if stage == "compute" and isinstance(stage_obj.get("mesh_output"), dict):
            mesh_output_stages += 1

    return {
        "by_stage": by_stage,
        "inline_sampler_stages": inline_sampler_stages,
        "inline_sampler_bindings": inline_sampler_bindings,
        "resource_heap_stages": resource_heap_stages,
        "resource_heap_bindings": resource_heap_bindings,
        "cbv_indexed_heap_stages": cbv_indexed_heap_stages,
        "cbv_indexed_heap_bindings": cbv_indexed_heap_bindings,
        "lane_indexed_heap_stages": lane_indexed_heap_stages,
        "lane_indexed_heap_bindings": lane_indexed_heap_bindings,
        "vertex_index_stages": vertex_index_stages,
        "mesh_output_stages": mesh_output_stages,
        "unresolved_stages": unresolved_stages,
        "unresolved_bindings": unresolved_bindings,
    }


def _stage_canonical_json(stage_obj: dict[str, Any]) -> str:
    return json.dumps(stage_obj, sort_keys=True, separators=(",", ":"))


def _sample_hashes(keys: list[str], *, max_samples: int = DEFAULT_MAX_SAMPLE_HASHES) -> list[str]:
    seen: set[str] = set()
    samples: list[str] = []
    for key in keys:
        safe_hash = _sanitize_hash(key)
        if safe_hash not in seen and len(samples) < max_samples:
            seen.add(safe_hash)
            samples.append(safe_hash)
    return samples


def diff_sidecars(
    baseline_stages: dict[str, Any],
    current_stages: dict[str, Any],
    *,
    max_samples: int = DEFAULT_MAX_SAMPLE_HASHES,
) -> dict[str, Any]:
    """Compare baseline and current sidecar stage maps and return a sanitized diff summary."""
    baseline_keys = set(baseline_stages.keys())
    current_keys = set(current_stages.keys())
    added_keys = sorted(current_keys - baseline_keys)
    removed_keys = sorted(baseline_keys - current_keys)
    common_keys = sorted(baseline_keys & current_keys)

    changed_keys: list[str] = []
    unchanged_count = 0
    for key in common_keys:
        if _stage_canonical_json(baseline_stages[key]) == _stage_canonical_json(current_stages[key]):
            unchanged_count += 1
        else:
            changed_keys.append(key)

    added_by_stage: dict[str, int] = {stage: 0 for stage in STAGES}
    for key in added_keys:
        stage = current_stages[key].get("stage")
        if stage in added_by_stage:
            added_by_stage[stage] += 1

    removed_by_stage: dict[str, int] = {stage: 0 for stage in STAGES}
    for key in removed_keys:
        stage = baseline_stages[key].get("stage")
        if stage in removed_by_stage:
            removed_by_stage[stage] += 1

    baseline_count = len(baseline_stages)
    current_count = len(current_stages)
    return {
        "baseline_stages": baseline_count,
        "current_stages": current_count,
        "net_delta": current_count - baseline_count,
        "added_count": len(added_keys),
        "removed_count": len(removed_keys),
        "changed_count": len(changed_keys),
        "unchanged_count": unchanged_count,
        "added_by_stage": added_by_stage,
        "removed_by_stage": removed_by_stage,
        "added_sample_hashes": _sample_hashes(added_keys, max_samples=max_samples),
        "removed_sample_hashes": _sample_hashes(removed_keys, max_samples=max_samples),
        "changed_sample_hashes": _sample_hashes(changed_keys, max_samples=max_samples),
    }


def build(
    docs: list[Any],
    existing_stages: dict[str, Any] | None = None,
    *,
    unresolved: bool = False,
) -> tuple[dict[str, Any], dict[str, int]]:
    stages: dict[str, Any] = dict(existing_stages) if existing_stages else {}
    skipped: dict[str, int] = {}
    for doc in docs:
        for entry in entries_of(doc):
            key, stage, why = convert(entry, unresolved=unresolved)
            if stage is None:
                skipped[why] = skipped.get(why, 0) + 1
                continue
            stages[key] = stage
    if _stages_require_v3(stages):
        out_version = VERSION_UNRESOLVED
    elif _stages_require_v2(stages):
        out_version = VERSION_ROOT_TABLE_OFFSET
    else:
        out_version = VERSION
    return {"schema": SCHEMA, "version": out_version, "stages": stages}, skipped


def build_missing_sources_report(
    docs: list[Any],
    *,
    max_samples: int = DEFAULT_MAX_SAMPLE_HASHES,
) -> dict[str, Any]:
    """Build a sanitized summary of stages skipped for 'binding without source'."""
    total_skipped = 0
    by_stage: dict[str, int] = {stage: 0 for stage in STAGES}
    counts_by_pattern: dict[tuple[str, str, str], int] = {}
    samples_by_pattern: dict[tuple[str, str, str], list[str]] = {}
    seen_hashes_by_pattern: dict[tuple[str, str, str], set[str]] = {}

    for doc in docs:
        for entry in entries_of(doc):
            key, stage_obj, why = convert(entry)
            if stage_obj is not None or why != "binding without source":
                continue
            stage = stage_name(str(entry.get("stage", "")))
            if stage is None:
                continue
            total_skipped += 1
            by_stage[stage] += 1
            safe_hash = _sanitize_hash(key)
            for b in entry.get("bindings", []):
                if not isinstance(b, dict) or isinstance(b.get("source"), dict):
                    continue
                raw_kind = b.get("kind")
                mapped_kind = KIND_MAP.get(raw_kind) if isinstance(raw_kind, str) else None
                kind = mapped_kind if mapped_kind is not None else _sanitize_label(raw_kind, fallback="unknown")
                name = _sanitize_label(b.get("name"), fallback="")
                pat = (stage, kind, name)
                counts_by_pattern[pat] = counts_by_pattern.get(pat, 0) + 1
                seen = seen_hashes_by_pattern.setdefault(pat, set())
                samples = samples_by_pattern.setdefault(pat, [])
                if safe_hash not in seen and len(samples) < max_samples:
                    seen.add(safe_hash)
                    samples.append(safe_hash)

    top_missing_bindings = [
        {
            "stage": stage,
            "kind": kind,
            "name": name,
            "count": count,
            "sample_hashes": samples_by_pattern[(stage, kind, name)],
        }
        for (stage, kind, name), count in sorted(
            counts_by_pattern.items(),
            key=lambda kv: (-kv[1], kv[0][0], kv[0][1], kv[0][2]),
        )
    ]

    return {
        "schema": REPORT_SCHEMA,
        "version": REPORT_VERSION,
        "total_skipped_missing_source": total_skipped,
        "by_stage": by_stage,
        "top_missing_bindings": top_missing_bindings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--merge-existing",
        type=Path,
        default=None,
        help="load an existing v1 sidecar JSON (if present) and merge newly converted stages on top",
    )
    parser.add_argument(
        "--diff-baseline",
        type=Path,
        default=None,
        help="compare the newly generated sidecar against a previous v1 sidecar JSON",
    )
    parser.add_argument(
        "--fail-on-regressions",
        action="store_true",
        help="with --diff-baseline, exit with code 2 if any previously reflected stage is removed",
    )
    parser.add_argument(
        "--report-missing-sources",
        type=Path,
        default=None,
        help="write a sanitized JSON report summarizing skipped 'binding without source' stages",
    )
    parser.add_argument(
        "--verify-pack",
        type=Path,
        default=None,
        help="cross-check reflection sidecar hashes against a WGPK v1 .wgsl.pack header/TOC",
    )
    parser.add_argument(
        "--unresolved-sources",
        action="store_true",
        help="keep stages whose read-only bindings have no source, marking those {\"unresolved\": true} "
        "(sidecar version 3; the runtime binds the worker's default for them)",
    )
    parser.add_argument("--json", action="store_true", help="print the counts as JSON")
    args = parser.parse_args(argv)

    if not args.inputs and args.verify_pack is None:
        parser.error("the following arguments are required: inputs")

    if not args.inputs and (args.merge_existing is not None or args.report_missing_sources is not None):
        print("error: --merge-existing and --report-missing-sources require input JSON files", file=sys.stderr)
        return 1

    if args.fail_on_regressions and args.diff_baseline is None:
        print("error: --fail-on-regressions requires --diff-baseline", file=sys.stderr)
        return 1

    existing_stages: dict[str, Any] = {}
    if args.merge_existing is not None:
        try:
            existing_stages = load_existing_sidecar(args.merge_existing)
        except (OSError, ValueError) as exc:
            print(f"error: failed to load --merge-existing: {exc}", file=sys.stderr)
            return 1

    baseline_stages: dict[str, Any] | None = None
    if args.diff_baseline is not None:
        try:
            baseline_stages = load_baseline_sidecar(args.diff_baseline)
        except (OSError, ValueError) as exc:
            print(f"error: failed to load --diff-baseline: {exc}", file=sys.stderr)
            return 1

    pack_entries: list[dict[str, Any]] | None = None
    if args.verify_pack is not None:
        try:
            pack_entries = load_wgpk_toc(args.verify_pack)
        except ValueError as exc:
            print(f"error: failed to load --verify-pack: {exc}", file=sys.stderr)
            return 1

    if args.inputs:
        docs = []
        for path in args.inputs:
            with path.open(encoding="utf-8-sig") as f:
                docs.append(json.load(f))

        sidecar, skipped = build(
            docs,
            existing_stages=existing_stages if args.merge_existing is not None else None,
            unresolved=args.unresolved_sources,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(sidecar, indent=1) + "\n", encoding="utf-8")

        if args.report_missing_sources is not None:
            report = build_missing_sources_report(docs)
            args.report_missing_sources.parent.mkdir(parents=True, exist_ok=True)
            args.report_missing_sources.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

        stages_written = len(sidecar["stages"])
    else:
        try:
            loaded_stages = _load_sidecar(args.out, label="sidecar", allow_missing=False)
        except OSError as exc:
            reason = exc.strerror or type(exc).__name__
            print(
                f"error: failed to load --out sidecar {_sanitize_path_label(args.out)}: {reason}",
                file=sys.stderr,
            )
            return 1
        except ValueError as exc:
            print(f"error: failed to load --out sidecar: {exc}", file=sys.stderr)
            return 1
        sidecar = {"schema": SCHEMA, "version": VERSION, "stages": loaded_stages}
        skipped = {}
        stages_written = 0

    counts: dict[str, Any] = {"stages_written": stages_written, "skipped": skipped}
    if args.merge_existing is not None:
        counts["merged_existing"] = len(existing_stages)
    counts.update(summarize_stages(sidecar["stages"]))

    diff: dict[str, Any] | None = None
    if baseline_stages is not None:
        diff = diff_sidecars(baseline_stages, sidecar["stages"])
        counts["diff"] = diff

    pack_verification: dict[str, Any] | None = None
    if pack_entries is not None and args.verify_pack is not None:
        pack_verification = verify_sidecar_against_pack(
            pack_entries,
            sidecar["stages"],
            pack_label=args.verify_pack,
        )
        counts["pack_verification"] = pack_verification

    if args.json:
        print(json.dumps(counts))
    else:
        if args.merge_existing is not None:
            line = (
                f"{counts['stages_written']} stage(s) written "
                f"(merged {counts['merged_existing']} existing); skipped: {skipped or 'none'}"
            )
        else:
            line = f"{counts['stages_written']} stage(s) written; skipped: {skipped or 'none'}"
        if diff is not None:
            line += (
                f"; diff: +{diff['added_count']}/-{diff['removed_count']}"
                f"/~{diff['changed_count']}/={diff['unchanged_count']}"
                f" (net {diff['net_delta']:+d})"
            )
        if pack_verification is not None:
            line += (
                f"; pack verify ({pack_verification['pack_label']}): "
                f"{pack_verification['matched_hashes']}/{pack_verification['pack_entry_count']} matched "
                f"({pack_verification['coverage_pct']:.2f}%), "
                f"{pack_verification['missing_reflection_hashes']['count']} missing, "
                f"{pack_verification['extra_reflection_hashes']['count']} extra"
            )
        print(line)

    if args.fail_on_regressions and diff is not None and diff["removed_count"] > 0:
        print(
            f"error: regression detected: {diff['removed_count']} stage(s) removed compared to baseline",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
