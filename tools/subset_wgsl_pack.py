#!/usr/bin/env python3
"""Cut a browser-loadable WGPK v1 pack (+ `.reflect.json` sidecar) out of a larger (v1 or v2) pack.

    python tools/subset_wgsl_pack.py std.wgsl.pack --out browser.wgsl.pack \
        [--max-wgsl-bytes 262144] [--max-total-bytes 1600000000] \
        [--priority-pack splash.wgsl.pack ...] [--priority-hashes hashes.txt ...] [--json]

The web preloader (web/runtime/pgim_preloader.js) only parses WGPK v1 and holds the whole pack in
memory, so the output is always v1 (u32 offsets) and bounded by --max-total-bytes. Stages over
--max-wgsl-bytes are dropped (same cap as `pack-rdna2 --max-wgsl-bytes`). Selection order under the
total budget: stages named by --priority-pack / --priority-hashes first (still subject to the per
shader cap), then all other stages by ascending size. The sidecar keeps only the selected stages.
Output is ascending hash order. Streamed, so the source may be many GiB. Pure stdlib.
Never commit the result: it is derived from the title's shaders.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

HEADER = struct.Struct("<4sIII")
ENTRY_V1 = struct.Struct("<QB3sII")
ENTRY_V2 = struct.Struct("<QB3sIQ")
U32_MAX = 0xFFFFFFFF
CHUNK = 8 * 1024 * 1024
DEFAULT_MAX_WGSL_BYTES = 262144
DEFAULT_MAX_TOTAL_BYTES = 1_600_000_000


def read_toc(path: Path):
    """Return [(hash, stage, length, offset)] in file order."""
    with open(path, "rb") as f:
        magic, version, count, _flags = HEADER.unpack(f.read(HEADER.size))
        if magic != b"WGPK" or version not in (1, 2):
            raise SystemExit(f"{path.name}: not a WGPK v1/v2 pack")
        ent = ENTRY_V1 if version == 1 else ENTRY_V2
        raw = f.read(ent.size * count)
    out = []
    for i in range(count):
        if version == 1:
            h, s, _r, off, ln = ENTRY_V1.unpack_from(raw, i * ent.size)
        else:
            h, s, _r, ln, off = ENTRY_V2.unpack_from(raw, i * ent.size)
        out.append((h, s, ln, off))
    return out


def load_hashes(path: Path) -> set[int]:
    return {int(m, 16) for m in re.findall(r"0x[0-9a-fA-F]{1,16}|\b[0-9a-fA-F]{16}\b", path.read_text(encoding="utf-8"))}


def select(toc, max_wgsl, max_total, priority: set[int]):
    cand = [e for e in toc if 0 < e[2] <= max_wgsl]
    pri = sorted((e for e in cand if e[0] in priority), key=lambda e: (e[2], e[0]))
    rest = sorted((e for e in cand if e[0] not in priority), key=lambda e: (e[2], e[0]))
    chosen, total = [], 0
    for e in pri + rest:
        if total + e[2] > max_total:
            continue
        chosen.append(e)
        total += e[2]
    chosen.sort(key=lambda e: e[0])
    return chosen, total, len(pri)


def write_pack(src: Path, chosen, out: Path) -> int:
    table = HEADER.size + ENTRY_V1.size * len(chosen)
    off, entries = table, []
    for h, s, ln, _o in chosen:
        entries.append(ENTRY_V1.pack(h, s, b"\0\0\0", off, ln))
        off += ln
    if off > U32_MAX:
        raise SystemExit("selection exceeds 4 GiB; lower --max-total-bytes")
    tmp = out.with_name(out.name + ".tmp")
    with open(src, "rb") as fin, open(tmp, "wb") as fo:
        fo.write(HEADER.pack(b"WGPK", 1, len(chosen), 0))
        fo.write(b"".join(entries))
        for _h, _s, ln, o in sorted(chosen, key=lambda e: e[0]):
            fin.seek(o)
            left = ln
            while left:
                b = fin.read(min(CHUNK, left))
                if not b:
                    raise SystemExit("source pack truncated")
                fo.write(b)
                left -= len(b)
    tmp.replace(out)
    return off


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pack", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-wgsl-bytes", type=int, default=DEFAULT_MAX_WGSL_BYTES)
    ap.add_argument("--max-total-bytes", type=int, default=DEFAULT_MAX_TOTAL_BYTES)
    ap.add_argument("--priority-pack", type=Path, action="append", default=[])
    ap.add_argument("--priority-hashes", type=Path, action="append", default=[])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    toc = read_toc(a.pack)
    priority: set[int] = set()
    for p in a.priority_pack:
        priority |= {e[0] for e in read_toc(p)}
    for p in a.priority_hashes:
        priority |= load_hashes(p)
    chosen, total, npri = select(toc, a.max_wgsl_bytes, a.max_total_bytes, priority)
    size = write_pack(a.pack, chosen, a.out)
    keep = {f"0x{e[0]:016x}" for e in chosen}
    side_in = Path(str(a.pack) + ".reflect.json")
    kept_reflect = None
    if side_in.is_file():
        d = json.loads(side_in.read_text(encoding="utf-8"))
        d["stages"] = {k: v for k, v in d.get("stages", {}).items() if k.lower() in keep}
        kept_reflect = len(d["stages"])
        side_out = Path(str(a.out) + ".reflect.json")
        side_out.write_text(json.dumps(d, indent=1), encoding="utf-8")
    rep = {"source_stages": len(toc), "stages": len(chosen), "priority_stages_kept": npri,
           "payload_bytes": total, "pack_bytes": size, "max_wgsl_bytes": a.max_wgsl_bytes,
           "reflect_stages": kept_reflect, "dropped_over_cap": sum(1 for e in toc if e[2] > a.max_wgsl_bytes)}
    print(json.dumps(rep) if a.json else "\n".join(f"{k}: {v}" for k, v in rep.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
