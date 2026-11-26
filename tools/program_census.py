#!/usr/bin/env python3
"""Census of shader programs that cannot run, so dropped draws can be explained.

Reads (all optional, at least one of --pack-report/--pack/--reflect is needed):
  --pack-report  translator summary JSON written next to the pack (`entries`, skip counters)
  --pack         the .wgsl.pack itself (only the 16-byte header and entry table are read)
  --reflect      the `<pack>.reflect.json` stage sidecar
  --boot-report  a runtime boot-report.json (aggregate draw/dispatch drop counters)
  --boot-log     the matching boot.log (reflection / mesh join failures, no-shader-module lines)
  --drop-counts  JSON {"0x<hash>": draws_lost} when per-program drop counts are available

Writes a sanitized summary (hashes, counts and categories only, never shader bytes) as JSON and/or
markdown. Pure stdlib.

    python tools/program_census.py --pack-report mesh.json --reflect mesh.wgsl.pack.reflect.json \
        --pack mesh.wgsl.pack --boot-report runs/x/boot-report.json --boot-log runs/x/boot.log \
        --json-out census.json --md-out census.md
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "playgta6-program-census"
VERSION = 1
MESH_RECORD_LIMIT_BYTES = 32 * 1024  # the host's per-mesh-group output limit named in the backlog
MESH_MAX_DIM = 256
STAGES = ("vertex", "fragment", "compute")
STAGE_CODES = {1: "vertex", 2: "fragment", 3: "compute"}

# A packed stage is attributed to the first of these that applies (every reason is also counted as a tag).
PRIMARY_ORDER = (
    "mesh_oversize",
    "not_naga_validated",
    "unsupported_ops",
    "writable_binding_without_source",
    "binding_without_source",
    "no_reflection_entry",
)
EXPLAIN = {
    "not_in_pack_oversize": "over the pack's WGSL size cap, so not packed (raw-address/BVH builders and large geometry shaders land here)",
    "not_in_pack_other": "failed translation, over budget or otherwise not packed",
    "mesh_oversize": "mesh program whose per-group output record exceeds the output limit",
    "not_naga_validated": "packed but the WGSL failed validation",
    "unsupported_ops": "packed with at least one op the translator emits as unsupported (the program may still run, but computes wrong values)",
    "writable_binding_without_source": "UAV (storage / storage texture) with no resolvable source: writes go nowhere",
    "binding_without_source": "read-only binding with no resolvable source (runtime binds a default or null)",
    "no_reflection_entry": "in the pack but absent from the reflection sidecar (runtime keeps its conventions)",
}


def _norm_kind(kind: str) -> str:
    return kind.replace("-", "_")


def _is_hash(h: Any) -> bool:
    return isinstance(h, str) and re.fullmatch(r"0x[0-9a-fA-F]{1,16}", h) is not None


def _canon(h: str) -> str:
    return "0x%016x" % int(h, 16)


def read_pack_toc(path: Path) -> dict[str, str]:
    """hash -> stage name, from the WGPK header and entry table only (v1 20 B, v2 24 B entries)."""
    with path.open("rb") as f:
        magic, version, count, _flags = struct.unpack("<4sIII", f.read(16))
        if magic != b"WGPK" or version not in (1, 2):
            raise ValueError(f"{path}: not a WGPK v1/v2 pack")
        size = 24 if version == 2 else 20
        table = f.read(count * size)
    if len(table) != count * size:
        raise ValueError(f"{path}: truncated entry table")
    out: dict[str, str] = {}
    for i in range(count):
        h = struct.unpack_from("<Q", table, i * size)[0]
        out["0x%016x" % h] = STAGE_CODES.get(table[i * size + 8], "unknown")
    return out


def mesh_record_bytes(m: dict[str, Any]) -> int:
    return 4 * (4 + int(m["max_vertices"]) * int(m["vertex_words"]) + int(m["max_primitives"]) * 3)


def mesh_problem(m: dict[str, Any], limit: int) -> str | None:
    try:
        mv, mp = int(m["max_vertices"]), int(m["max_primitives"])
        int(m["vertex_words"])
    except (KeyError, TypeError, ValueError):
        return "mesh_output malformed"
    if not (1 <= mv <= MESH_MAX_DIM and 1 <= mp <= MESH_MAX_DIM):
        return "mesh dimensions outside 1..256"
    if mesh_record_bytes(m) > limit:
        return "mesh record over output limit"
    return None


def collect_stages(
    entries: Iterable[dict[str, Any]],
    reflect_stages: dict[str, Any] | None,
    pack_toc: dict[str, str] | None,
    limit: int,
) -> dict[str, dict[str, Any]]:
    """Per-hash facts: stage, reasons (set), unresolved binding kinds, mesh flag."""
    stages: dict[str, dict[str, Any]] = {}
    for e in entries:
        h = e.get("hash_fnv1a64")
        if not _is_hash(h):
            continue
        reasons: set[str] = set()
        unresolved: Counter[str] = Counter()
        if e.get("naga_validated") is False:
            reasons.add("not_naga_validated")
        if int(e.get("unsupported_op_count") or 0) > 0:
            reasons.add("unsupported_ops")
        for b in e.get("bindings", []):
            if "source" in b:
                continue
            kind = _norm_kind(str(b.get("kind", "unknown")))
            unresolved[kind] += 1
            reasons.add("writable_binding_without_source" if kind in ("storage", "storage_texture") else "binding_without_source")
        mesh = e.get("mesh_output")
        if isinstance(mesh, dict) and mesh_problem(mesh, limit):
            reasons.add("mesh_oversize")
        stages[_canon(h)] = {"stage": e.get("stage", "unknown"), "reasons": reasons, "unresolved": unresolved,
                             "mesh": isinstance(mesh, dict), "mesh_output": mesh if isinstance(mesh, dict) else None}
    reflect_keys: set[str] = set()
    if reflect_stages is not None:
        for key, s in reflect_stages.items():
            if not _is_hash(key):
                continue
            h = _canon(key)
            reflect_keys.add(h)
            st = stages.setdefault(h, {"stage": s.get("stage", "unknown"), "reasons": set(), "unresolved": Counter(),
                                       "mesh": False, "mesh_output": None})
            mesh = s.get("mesh_output")
            if isinstance(mesh, dict):
                st["mesh"] = True
                st["mesh_output"] = mesh
                if mesh_problem(mesh, limit):
                    st["reasons"].add("mesh_oversize")
    if pack_toc is not None:
        for h, name in pack_toc.items():
            stages.setdefault(h, {"stage": name, "reasons": set(), "unresolved": Counter(), "mesh": False, "mesh_output": None})
    if reflect_stages is not None:
        for h, st in stages.items():
            if h not in reflect_keys:
                st["reasons"].add("no_reflection_entry")
    return stages


JOIN_HEAD = re.compile(r"reflection join failed \((vertex|compute) (0x[0-9a-fA-F]+)(?:, pixel (0x[0-9a-fA-F]+))?\)")
MESH_HEAD = re.compile(r"mesh pipeline layout join failed \(mesh (0x[0-9a-fA-F]+)\)")
NO_MODULE = re.compile(r"no shader module for DXIL identity (0x[0-9a-fA-F]+) \((vertex|pixel|compute) shader")


def normalize_why(why: str) -> str:
    why = re.sub(r"0x[0-9a-fA-F]+", "0xH", why)
    why = re.sub(r"\bs\d+\b", "sN", why)
    why = re.sub(r"\b\d+\b", "N", why)
    return why.strip()


def parse_boot_log(lines: Iterable[str]) -> dict[str, Any]:
    reasons: Counter[str] = Counter()
    per_reason: dict[str, set[str]] = defaultdict(set)
    heads = {"reflection_join": 0, "mesh_layout_join": 0}
    no_module: dict[str, set[str]] = defaultdict(set)
    current: list[str] | None = None
    for raw in lines:
        line = raw.rstrip("\n")
        m = NO_MODULE.search(line)
        if m:
            no_module[m.group(2)].add(_canon(m.group(1)))
            current = None
            continue
        j, mm = JOIN_HEAD.search(line), MESH_HEAD.search(line)
        if j:
            heads["reflection_join"] += 1
            current = [_canon(j.group(2))] + ([_canon(j.group(3))] if j.group(3) else [])
            continue
        if mm:
            heads["mesh_layout_join"] += 1
            current = [_canon(mm.group(1))]
            continue
        if current is not None:
            if not line.strip() or "d3d12_xs:" in line:
                current = None
                continue
            if ": " in line:
                why = normalize_why(line.split(": ", 1)[1])
                reasons[why] += 1
                per_reason[why].update(current)
    return {
        "join_failure_events": heads,
        "join_failure_reasons": [{"reason": r, "bindings": n, "programs": len(per_reason[r])} for r, n in reasons.most_common()],
        "join_failed_hashes": sorted({h for s in per_reason.values() for h in s}),
        "no_shader_module_programs": {k: len(v) for k, v in sorted(no_module.items())},
    }


DROP_COUNTER_KEYS = (
    "draws_issued", "draws_emitted", "draws_no_shader_module", "draws_mesh_pipeline", "draws_rect_list",
    "draws_dropped_by_bridge", "dispatches_issued", "dispatches_emitted", "dispatches_no_shader_module",
    "dispatches_dropped_by_bridge", "indirect_executions_skipped", "copies_skipped",
    "reflection_unresolved_bindings", "reflection_lane_indexed_unbound", "reflection_table_slots_outside_rs",
    "reflection_vertex_inputs_defaulted",
)


def read_boot_report(doc: dict[str, Any]) -> dict[str, Any]:
    gw = doc.get("gpu_work") or {}
    counters = {k: gw[k] for k in DROP_COUNTER_KEYS if isinstance(gw.get(k), int)}
    out: dict[str, Any] = {
        "gpu_work": counters,
        "draws_lost_total": sum(counters.get(k, 0) for k in ("draws_no_shader_module", "draws_mesh_pipeline", "draws_dropped_by_bridge")),
        "dispatches_lost_total": sum(counters.get(k, 0) for k in ("dispatches_no_shader_module", "dispatches_dropped_by_bridge")),
    }
    # Per-program tables, if a report carries them: any {hash: count} dict under gpu_work.
    per: dict[str, int] = {}
    for val in gw.values():
        if isinstance(val, dict) and val and all(_is_hash(k) and isinstance(v, int) for k, v in val.items()):
            for k, v in val.items():
                per[_canon(k)] = per.get(_canon(k), 0) + v
    if per:
        out["per_program_drops"] = per
    return out


def build_census(
    *,
    pack_report: dict[str, Any] | None = None,
    reflect: dict[str, Any] | None = None,
    pack_toc: dict[str, str] | None = None,
    boot_report: dict[str, Any] | None = None,
    boot_log: Iterable[str] | None = None,
    drop_counts: dict[str, int] | None = None,
    mesh_limit: int = MESH_RECORD_LIMIT_BYTES,
    top: int = 20,
) -> dict[str, Any]:
    entries = (pack_report or {}).get("entries", [])
    stages = collect_stages(entries, (reflect or {}).get("stages"), pack_toc, mesh_limit)
    census: dict[str, Any] = {"schema": SCHEMA, "version": VERSION, "mesh_record_limit_bytes": mesh_limit}

    if pack_report:
        keys = ("total_blobs", "translated_ok", "skipped_or_failed", "failed", "skipped_oversize", "skipped_budget",
                "omitted_by_budget", "manifest_duplicates_skipped")
        census["translator_summary"] = {k: pack_report[k] for k in keys if isinstance(pack_report.get(k), int)}
        nip = []
        if pack_report.get("skipped_oversize"):
            nip.append({"reason": "not_in_pack_oversize", "count": pack_report["skipped_oversize"], "explain": EXPLAIN["not_in_pack_oversize"]})
        other = sum(int(pack_report.get(k) or 0) for k in ("failed", "skipped_budget", "omitted_by_budget"))
        if other:
            nip.append({"reason": "not_in_pack_other", "count": other, "explain": EXPLAIN["not_in_pack_other"]})
        census["not_in_pack"] = nip
        fails: Counter[str] = Counter()
        for f in pack_report.get("failures", []) or []:
            text = (f if isinstance(f, str) else json.dumps(f)).lower()
            fails["raw_address_or_bvh" if re.search(r"bvh|address", text) else "other"] += 1
        census["translator_failures_by_class"] = dict(fails)

    by_stage: dict[str, dict[str, int]] = {s: {"programs": 0, "with_reason": 0} for s in STAGES}
    primary: Counter[str] = Counter()
    tagged: Counter[str] = Counter()
    primary_by_stage: dict[str, Counter[str]] = defaultdict(Counter)
    unresolved_kinds: Counter[str] = Counter()
    unresolved_programs: Counter[str] = Counter()
    for st in stages.values():
        s = st["stage"] if st["stage"] in by_stage else None
        if s:
            by_stage[s]["programs"] += 1
        for k, n in st["unresolved"].items():
            unresolved_kinds[k] += n
            unresolved_programs[k] += 1
        if not st["reasons"]:
            continue
        if s:
            by_stage[s]["with_reason"] += 1
        tagged.update(st["reasons"])
        p = next(r for r in PRIMARY_ORDER if r in st["reasons"])
        st["primary"] = p
        primary[p] += 1
        primary_by_stage[p][st["stage"]] += 1
    census["packed_programs"] = {"total": len(stages), "by_stage": by_stage}
    census["packed_reasons"] = [
        {"reason": r, "primary_count": primary[r], "tagged_count": tagged[r], "by_stage": dict(primary_by_stage[r]), "explain": EXPLAIN[r]}
        for r in PRIMARY_ORDER if tagged[r]
    ]
    census["unresolved_bindings_by_kind"] = [
        {"kind": k, "bindings": n, "programs": unresolved_programs[k]} for k, n in unresolved_kinds.most_common()
    ]
    meshes = [s["mesh_output"] for s in stages.values() if s["mesh_output"] and mesh_problem(s["mesh_output"], 1 << 62) is None]
    census["mesh"] = {
        "programs": sum(1 for s in stages.values() if s["mesh"]),
        "over_limit": tagged["mesh_oversize"],
        "largest_record_bytes": sorted((mesh_record_bytes(m) for m in meshes), reverse=True)[:5],
    }

    runtime: dict[str, Any] = {}
    if boot_report:
        runtime.update(read_boot_report(boot_report))
    if boot_log is not None:
        log = parse_boot_log(boot_log)
        failed = log.pop("join_failed_hashes")
        known = [h for h in failed if h in stages]
        log["join_failed_programs_in_pack"] = len(known)
        log["join_failed_programs_not_in_pack"] = len(failed) - len(known)
        log["join_failed_mesh_programs"] = sum(1 for h in known if stages[h]["mesh"])
        runtime["boot_log"] = log
    drops = dict(runtime.pop("per_program_drops", {}) or {})
    for k, v in (drop_counts or {}).items():
        if _is_hash(k):
            drops[_canon(k)] = drops.get(_canon(k), 0) + int(v)
    if drops:
        def reason_of(h: str) -> str:
            st = stages.get(h)
            if st is None:
                return "not_in_pack"
            return st.get("primary", "packed_no_static_reason")
        weighted: Counter[str] = Counter()
        for h, n in drops.items():
            weighted[reason_of(h)] += n
        runtime["drops_weighted_by_reason"] = [{"reason": r, "draws_lost": n} for r, n in weighted.most_common()]
        runtime["top_dropped_programs"] = [
            {"hash": h, "draws_lost": n, "stage": (stages.get(h) or {}).get("stage"), "reason": reason_of(h)}
            for h, n in sorted(drops.items(), key=lambda kv: -kv[1])[:top]
        ]
    else:
        runtime["per_program_drop_counts"] = "unavailable: the boot report only has aggregate drop counters; pass --drop-counts to weight programs"
    census["runtime"] = runtime
    return census


def render_markdown(c: dict[str, Any]) -> str:
    L: list[str] = ["# Program census", ""]
    ts = c.get("translator_summary")
    if ts:
        L += ["## Translator output", "", "| counter | value |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in ts.items()] + [""]
    if c.get("not_in_pack"):
        L += ["## Never reached the pack", "", "| reason | programs | meaning |", "|---|---:|---|"]
        L += [f"| {r['reason']} | {r['count']} | {r['explain']} |" for r in c["not_in_pack"]] + [""]
    pp = c["packed_programs"]
    L += ["## Packed programs that cannot fully run", "", f"{pp['total']} programs are packed.", "",
          "| stage | packed | with at least one reason |", "|---|---:|---:|"]
    L += [f"| {s} | {v['programs']} | {v['with_reason']} |" for s, v in pp["by_stage"].items()] + [""]
    L += ["| reason | programs (primary) | programs (any) | by stage | meaning |", "|---|---:|---:|---|---|"]
    for r in c["packed_reasons"]:
        bs = ", ".join(f"{k} {v}" for k, v in sorted(r["by_stage"].items()))
        L.append(f"| {r['reason']} | {r['primary_count']} | {r['tagged_count']} | {bs} | {r['explain']} |")
    L += ["", "## Unresolved bindings by kind", "", "| kind | bindings | programs |", "|---|---:|---:|"]
    L += [f"| {u['kind']} | {u['bindings']} | {u['programs']} |" for u in c["unresolved_bindings_by_kind"]]
    m = c["mesh"]
    L += ["", "## Mesh programs", "", f"{m['programs']} mesh programs, {m['over_limit']} over the {c['mesh_record_limit_bytes']} B output limit."]
    rt = c.get("runtime", {})
    if rt:
        L += ["", "## Runtime drops", ""]
        if "gpu_work" in rt:
            L += [f"Draws lost: {rt['draws_lost_total']}, dispatches lost: {rt['dispatches_lost_total']}.", "",
                  "| counter | value |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in rt["gpu_work"].items()] + [""]
        bl = rt.get("boot_log")
        if bl:
            L += [f"Join failures: {bl['join_failure_events']}; programs with no shader module: {bl['no_shader_module_programs']}.",
                  f"Join-failed programs found in the pack: {bl['join_failed_programs_in_pack']} "
                  f"({bl['join_failed_mesh_programs']} mesh), not in the pack: {bl['join_failed_programs_not_in_pack']}.", "",
                  "| join failure reason | bindings | programs |", "|---|---:|---:|"]
            L += [f"| {r['reason']} | {r['bindings']} | {r['programs']} |" for r in bl["join_failure_reasons"]] + [""]
        if "drops_weighted_by_reason" in rt:
            L += ["| reason | draws lost |", "|---|---:|"] + [f"| {r['reason']} | {r['draws_lost']} |" for r in rt["drops_weighted_by_reason"]] + [""]
            L += ["| program | stage | reason | draws lost |", "|---|---|---|---:|"]
            L += [f"| {p['hash']} | {p['stage']} | {p['reason']} | {p['draws_lost']} |" for p in rt["top_dropped_programs"]] + [""]
        elif "per_program_drop_counts" in rt:
            L += [rt["per_program_drop_counts"], ""]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack-report", type=Path)
    ap.add_argument("--pack", type=Path)
    ap.add_argument("--reflect", type=Path)
    ap.add_argument("--boot-report", type=Path)
    ap.add_argument("--boot-log", type=Path)
    ap.add_argument("--drop-counts", type=Path)
    ap.add_argument("--mesh-limit-bytes", type=int, default=MESH_RECORD_LIMIT_BYTES)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--md-out", type=Path)
    a = ap.parse_args(argv)
    if not (a.pack_report or a.pack or a.reflect):
        ap.error("need at least one of --pack-report, --pack, --reflect")

    def load(p: Path) -> Any:
        return json.loads(p.read_text(encoding="utf-8"))

    log_lines = None
    if a.boot_log:
        with a.boot_log.open("r", encoding="utf-8", errors="replace") as f:
            log_lines = list(f)
    census = build_census(
        pack_report=load(a.pack_report) if a.pack_report else None,
        reflect=load(a.reflect) if a.reflect else None,
        pack_toc=read_pack_toc(a.pack) if a.pack else None,
        boot_report=load(a.boot_report) if a.boot_report else None,
        boot_log=log_lines,
        drop_counts=load(a.drop_counts) if a.drop_counts else None,
        mesh_limit=a.mesh_limit_bytes,
        top=a.top,
    )
    if a.json_out:
        a.json_out.write_text(json.dumps(census, indent=1) + "\n", encoding="utf-8")
    md = render_markdown(census)
    if a.md_out:
        a.md_out.write_text(md, encoding="utf-8")
    if not a.json_out and not a.md_out:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
