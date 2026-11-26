#!/usr/bin/env python3
"""Union of several WGPK shader packs and their `.reflect.json` stage sidecars.

    python tools/merge_wgsl_packs.py a.wgsl.pack b.wgsl.pack --out merged.wgsl.pack [--json]

Each input `<pack>` may have a `<pack>.reflect.json` next to it (missing is fine). Stages are
deduplicated by stage hash. When several inputs carry the same hash, the one whose sidecar has an
entry wins, then the one with more bindings that have a resolved `source`, then the earlier input.
The payload and the sidecar entry both come from that winner. Output order is ascending hash, so
the result is deterministic. The output is WGPK v1 when it fits in u32 offsets, else v2. Packs are
streamed (payloads are copied in chunks), so packs larger than memory are fine. Pure stdlib.

The merged pack and sidecar are derived from the title's shaders: write them outside the
repository (e.g. under work/), never commit them.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any

HEADER = struct.Struct("<4sIII")
ENTRY_V1 = struct.Struct("<QB3sII")  # hash, stage, reserved, offset, length
ENTRY_V2 = struct.Struct("<QB3sIQ")  # hash, stage, reserved, length, offset
U32_MAX = 0xFFFFFFFF
CHUNK = 8 * 1024 * 1024


def read_toc(path: Path) -> list[tuple[int, int, int, int]]:
    """Return [(hash, stage_code, offset, length)] after bounds checks (header + TOC only)."""
    with path.open("rb") as f:
        total = f.seek(0, 2)
        f.seek(0)
        raw = f.read(HEADER.size)
        if len(raw) < HEADER.size:
            raise ValueError(f"{path}: truncated WGPK header")
        magic, version, count, _flags = HEADER.unpack(raw)
        if magic != b"WGPK" or version not in (1, 2):
            raise ValueError(f"{path}: not a WGPK v1/v2 pack")
        ent = ENTRY_V2 if version == 2 else ENTRY_V1
        table = f.read(count * ent.size)
    if len(table) != count * ent.size:
        raise ValueError(f"{path}: truncated entry table")
    out = []
    seen: set[int] = set()
    for i in range(count):
        if version == 2:
            h, stage, _r, length, offset = ent.unpack_from(table, i * ent.size)
        else:
            h, stage, _r, offset, length = ent.unpack_from(table, i * ent.size)
        if stage not in (1, 2, 3):
            raise ValueError(f"{path}: invalid stage code {stage} at entry {i}")
        if offset + length > total:
            raise ValueError(f"{path}: entry {i} payload exceeds pack bounds")
        if h in seen:
            raise ValueError(f"{path}: duplicate hash 0x{h:016x}")
        seen.add(h)
        out.append((h, stage, offset, length))
    return out


def load_sidecar(pack: Path) -> dict[str, Any] | None:
    side = pack.with_name(pack.name + ".reflect.json")
    if not side.exists():
        return None
    with side.open(encoding="utf-8-sig") as f:
        doc = json.load(f)
    if not isinstance(doc, dict) or not isinstance(doc.get("stages"), dict):
        raise ValueError(f"{side}: not a stage reflection sidecar")
    return doc


def resolved_bindings(stage_obj: Any) -> int:
    if not isinstance(stage_obj, dict):
        return 0
    return sum(1 for b in stage_obj.get("bindings", []) if isinstance(b, dict) and "source" in b)


def merge(pack_paths: list[Path], out: Path) -> dict[str, Any]:
    tocs = [read_toc(p) for p in pack_paths]
    sidecars = [load_sidecar(p) for p in pack_paths]
    side_by_hash: list[dict[int, Any]] = [
        {int(k, 16): v for k, v in (sc["stages"].items() if sc else [])} for sc in sidecars
    ]

    best: dict[int, tuple[tuple[int, int, int], int, int, int, int]] = {}
    seen_total = 0
    for idx, toc in enumerate(tocs):
        for h, stage, offset, length in toc:
            seen_total += 1
            entry = side_by_hash[idx].get(h)
            rank = (1 if entry is not None else 0, resolved_bindings(entry), -idx)
            cur = best.get(h)
            if cur is None or rank > cur[0]:
                best[h] = (rank, idx, stage, offset, length)

    order = sorted(best)
    total_payload = sum(best[h][4] for h in order)
    v1_size = HEADER.size + len(order) * ENTRY_V1.size + total_payload
    version = 1 if v1_size <= U32_MAX else 2
    ent = ENTRY_V2 if version == 2 else ENTRY_V1
    data_start = HEADER.size + len(order) * ent.size

    toc_out = bytearray()
    cursor = data_start
    for h in order:
        _rank, _idx, stage, _off, length = best[h]
        if version == 2:
            toc_out += ent.pack(h, stage, b"\0\0\0", length, cursor)
        else:
            toc_out += ent.pack(h, stage, b"\0\0\0", cursor, length)
        cursor += length

    out.parent.mkdir(parents=True, exist_ok=True)
    handles = [p.open("rb") for p in pack_paths]
    try:
        with out.open("wb") as w:
            w.write(HEADER.pack(b"WGPK", version, len(order), 0))
            w.write(toc_out)
            for h in order:
                _rank, idx, _stage, offset, length = best[h]
                src = handles[idx]
                src.seek(offset)
                left = length
                while left:
                    chunk = src.read(min(CHUNK, left))
                    if not chunk:
                        raise ValueError(f"{pack_paths[idx]}: short read while copying 0x{h:016x}")
                    w.write(chunk)
                    left -= len(chunk)
    finally:
        for fh in handles:
            fh.close()

    stages_out: dict[str, Any] = {}
    for h in order:
        entry = side_by_hash[best[h][1]].get(h)
        if entry is not None:
            stages_out["0x%016x" % h] = entry
    docs = [sc for sc in sidecars if sc]
    if docs:
        merged_doc = {k: v for k, v in docs[0].items() if k != "stages"}
        merged_doc["version"] = max(int(d.get("version", 1)) for d in docs)
        merged_doc["stages"] = stages_out
        sidecar_path = out.with_name(out.name + ".reflect.json")
        with sidecar_path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(merged_doc, f, indent=1)
            f.write("\n")

    names = {1: "vertex", 2: "fragment", 3: "compute"}
    stage_counts = {"vertex": 0, "fragment": 0, "compute": 0}
    for h in order:
        stage_counts[names[best[h][2]]] += 1
    return {
        "inputs": [p.name for p in pack_paths],
        "input_entries": [len(t) for t in tocs],
        "input_entries_total": seen_total,
        "merged_entries": len(order),
        "duplicates_dropped": seen_total - len(order),
        "stage_counts": stage_counts,
        "sidecar_entries": len(stages_out) if docs else None,
        "pack_version": version,
        "pack_bytes": data_start + total_payload,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("packs", nargs="+", type=Path, help="input .wgsl.pack files (earlier wins ties)")
    ap.add_argument("--out", required=True, type=Path, help="merged .wgsl.pack to write")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = ap.parse_args(argv)
    outs = args.out.resolve()
    if any(p.resolve() == outs for p in args.packs):
        print("error: --out must differ from every input", file=sys.stderr)
        return 2
    try:
        summary = merge(args.packs, args.out)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, indent=1))
    else:
        print(", ".join(f"{k}={v}" for k, v in summary.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
