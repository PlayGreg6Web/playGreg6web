#!/usr/bin/env python3
"""Write the stage-reflection sidecar (`<pack>.reflect.json`, G62 / #651) from the translator's
JSON output.

Input: one or more JSON files, each either a `playgta6-shaders translate-rdna2 --json` result (one
stage) or a `playgta6-shaders pack-rdna2 --json` summary (its `entries` list). A stage enters the
sidecar only when its reflection carries the join keys the runtime needs:
  - every binding has a `source` ({user_sgpr, table_byte_offset} | {user_sgpr, root_constants} |
    {user_sgpr, root_descriptor} | {static_sampler} | {sampler_heap | resource_heap: {base_user_sgpr,
    stride, index: {user_sgpr, shift}}} (S34: a directly indexed heap, index from a root constant));
  - a vertex stage that fetches attributes lists `vertex_inputs` ({location, vb_slot, byte_offset,
    components, component_type, fetch_format?}).
Stages without them are skipped and counted, so the runtime keeps its conventions for them.

    python tools/stage_reflection_sidecar.py <json>... --out <pack>.reflect.json [--json]

The sidecar is derived from the title's shaders: write it under work/, never into the repository.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "playgta6-stage-reflection"
VERSION = 1

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


def convert(entry: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None, str]:
    """(key, sidecar stage, reason it was skipped)."""
    key = entry.get("hash_fnv1a64")
    stage = stage_name(str(entry.get("stage", "")))
    if not key or stage is None:
        return None, None, "no hash or stage"
    out: dict[str, Any] = {"stage": stage, "bindings": []}
    for b in entry.get("bindings", []):
        source = b.get("source")
        if not isinstance(source, dict):
            return key, None, "binding without source"
        if heap_source_ok(source) is False:
            return key, None, "malformed heap source"
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
    return key, out, ""


def entries_of(doc: Any) -> list[dict[str, Any]]:
    if isinstance(doc, dict) and isinstance(doc.get("entries"), list):
        return [e for e in doc["entries"] if isinstance(e, dict)]
    if isinstance(doc, dict) and "hash_fnv1a64" in doc:
        return [doc]
    return []


def build(docs: list[Any]) -> tuple[dict[str, Any], dict[str, int]]:
    stages: dict[str, Any] = {}
    skipped: dict[str, int] = {}
    for doc in docs:
        for entry in entries_of(doc):
            key, stage, why = convert(entry)
            if stage is None:
                skipped[why] = skipped.get(why, 0) + 1
                continue
            stages[key] = stage
    return {"schema": SCHEMA, "version": VERSION, "stages": stages}, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json", action="store_true", help="print the counts as JSON")
    args = parser.parse_args(argv)
    docs = []
    for path in args.inputs:
        with path.open(encoding="utf-8-sig") as f:
            docs.append(json.load(f))
    sidecar, skipped = build(docs)
    args.out.write_text(json.dumps(sidecar, indent=1) + "\n", encoding="utf-8")
    counts = {"stages_written": len(sidecar["stages"]), "skipped": skipped}
    print(json.dumps(counts) if args.json else
          f"{counts['stages_written']} stage(s) written; skipped: {skipped or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
