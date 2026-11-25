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
    stage builds from constants; the runtime decodes them));
  - a vertex stage that fetches attributes lists `vertex_inputs` ({location, vb_slot, byte_offset,
    components, component_type, fetch_format?}).
Stages without them are skipped and counted, so the runtime keeps its conventions for them.

    python tools/stage_reflection_sidecar.py <json>... --out <pack>.reflect.json [--merge-existing <reflect.json>] [--report-missing-sources <report.json>] [--json]

The sidecar is derived from the title's shaders: write it under work/, never into the repository.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA = "playgta6-stage-reflection"
VERSION = 1
REPORT_SCHEMA = "playgta6-stage-reflection-missing-sources"
REPORT_VERSION = 1
DEFAULT_MAX_SAMPLE_HASHES = 5
STAGES = ("vertex", "fragment", "compute")

_SAFE_LABEL_RE = re.compile(r"^[A-Za-z0-9_.$<>\[\]-]{1,64}$")
_SAFE_HASH_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

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


def inline_sampler_ok(source: dict[str, Any]) -> bool | None:
    """None: not an inline sampler; else whether sdwords is four 32-bit unsigned integers."""
    if "inline_sampler" not in source:
        return None
    sampler = source["inline_sampler"]
    words = sampler.get("sdwords") if isinstance(sampler, dict) else None
    return (isinstance(words, list) and len(words) == 4 and
            all(isinstance(w, int) and not isinstance(w, bool) and 0 <= w <= 0xFFFFFFFF for w in words))


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
        if inline_sampler_ok(source) is False:
            return key, None, "malformed inline sampler"
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


def load_existing_sidecar(path: Path) -> dict[str, Any]:
    """Load and validate an existing v1 sidecar JSON; returns {} if the file does not exist."""
    if not path.exists():
        return {}
    if path.is_dir():
        raise ValueError("existing sidecar path is a directory")
    try:
        with path.open(encoding="utf-8-sig") as f:
            doc = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in existing sidecar: {exc.msg}") from exc
    if not isinstance(doc, dict):
        raise ValueError("existing sidecar must be a JSON object")
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"existing sidecar schema must be {SCHEMA!r}, got {doc.get('schema')!r}")
    version = doc.get("version")
    if type(version) is not int or version != VERSION:
        raise ValueError(f"existing sidecar version must be {VERSION}, got {version!r}")
    stages = doc.get("stages")
    if not isinstance(stages, dict):
        raise ValueError("existing sidecar 'stages' must be an object")
    validated: dict[str, Any] = {}
    for k, v in stages.items():
        if not isinstance(k, str) or not k:
            raise ValueError("existing sidecar stage keys must be non-empty strings")
        if not isinstance(v, dict):
            raise ValueError(f"existing sidecar stage {k!r} must be an object")
        if v.get("stage") not in STAGES:
            raise ValueError(f"existing sidecar stage {k!r} has invalid stage {v.get('stage')!r}")
        if not isinstance(v.get("bindings"), list):
            raise ValueError(f"existing sidecar stage {k!r} must have a 'bindings' list")
        validated[k] = v
    return validated


def build(docs: list[Any], existing_stages: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, int]]:
    stages: dict[str, Any] = dict(existing_stages) if existing_stages else {}
    skipped: dict[str, int] = {}
    for doc in docs:
        for entry in entries_of(doc):
            key, stage, why = convert(entry)
            if stage is None:
                skipped[why] = skipped.get(why, 0) + 1
                continue
            stages[key] = stage
    return {"schema": SCHEMA, "version": VERSION, "stages": stages}, skipped


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
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--merge-existing",
        type=Path,
        default=None,
        help="load an existing v1 sidecar JSON (if present) and merge newly converted stages on top",
    )
    parser.add_argument(
        "--report-missing-sources",
        type=Path,
        default=None,
        help="write a sanitized JSON report summarizing skipped 'binding without source' stages",
    )
    parser.add_argument("--json", action="store_true", help="print the counts as JSON")
    args = parser.parse_args(argv)

    existing_stages: dict[str, Any] = {}
    if args.merge_existing is not None:
        try:
            existing_stages = load_existing_sidecar(args.merge_existing)
        except (OSError, ValueError) as exc:
            print(f"error: failed to load --merge-existing: {exc}", file=sys.stderr)
            return 1

    docs = []
    for path in args.inputs:
        with path.open(encoding="utf-8-sig") as f:
            docs.append(json.load(f))

    sidecar, skipped = build(docs, existing_stages=existing_stages if args.merge_existing is not None else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(sidecar, indent=1) + "\n", encoding="utf-8")

    if args.report_missing_sources is not None:
        report = build_missing_sources_report(docs)
        args.report_missing_sources.parent.mkdir(parents=True, exist_ok=True)
        args.report_missing_sources.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    counts: dict[str, Any] = {"stages_written": len(sidecar["stages"]), "skipped": skipped}
    if args.merge_existing is not None:
        counts["merged_existing"] = len(existing_stages)

    if args.json:
        print(json.dumps(counts))
    elif args.merge_existing is not None:
        print(
            f"{counts['stages_written']} stage(s) written "
            f"(merged {counts['merged_existing']} existing); skipped: {skipped or 'none'}"
        )
    else:
        print(f"{counts['stages_written']} stage(s) written; skipped: {skipped or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
