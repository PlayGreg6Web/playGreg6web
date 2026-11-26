#!/usr/bin/env python3
"""Static survey of the platform surface a title can call, ranked by how likely it stops a boot.

Joins static inputs (names, GUIDs, slot numbers and counts only; no bytes are printed):

  provider tables   runtime/os/xgameruntime_provider.cpp: every COM interface the runtime
                    registers and, per slot, implemented (`method<>`), trapping stub (`stub()` or a
                    gap), or fixed-answer stub (a `StubApi` with `SlotPolicy::ReturnFixed`).
  interface survey  `playgta6-recomp survey-interfaces --source name=qapi,proc=QueryApiImpl,
                    out=2,key=0+1 --json`: which slots of which {provider, iid} the title calls and
                    from which functions (call-site addresses).
  call graph        `playgta6-recomp discover --out`: direct and tail call edges per function, plus
                    optionally the recorded indirect targets of #1654 (`<site> <target>` pairs, the
                    site being a return address) as extra edges and as "observed executed" marks.
  import bridge     `program-manifest.json` of `build-program`: imports whose binding is
                    `unresolved`; their IAT call sites are located by scanning the executable's code
                    for `call/jmp [rip+disp]` through that slot (addresses only).

Each trapping slot is given a reachability tier from boot:
  A  the call site's function is reachable from the entry point over direct, tail and recorded
     indirect edges
  B  not reachable that way, but a function observed executing in a run is at most --anc-depth
     reverse direct-call hops above it (dispatch through code pointers, past where the recorded run
     got to)
  C  neither

Hard stops (docs/HARD_STOPS.md: store, licence, entitlement, attestation, device identity,
sign-in) are listed separately and never ranked. Pure stdlib.

    python tools/surface_survey.py --pe <exe> --recomp target/release/playgta6-recomp \
        --targets <indirect-targets.txt> --program-manifest <program-manifest.json> \
        --json-out survey.json --md-out survey.md
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import struct
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "playgta6-surface-survey"
VERSION = 1
DEFAULT_PROVIDER_SRC = "runtime/os/xgameruntime_provider.cpp"
QAPI_SOURCE = "name=qapi,proc=QueryApiImpl,out=2,key=0+1"
DEFAULT_ANC_DEPTH = 6
UNKNOWN_FN_GAP = 0x20000  # a site farther than this past the nearest function entry belongs to none

# Interface labels whose unimplemented slots are hard stops, and slot-specific hard stops of other
# interfaces. Source: docs/HARD_STOPS.md.
HARD_INTERFACE_LABELS = ("xgameruntime.XStore", "xgameruntime.XStoreExt", "xgameruntime.XUser",
                         "xgameruntime.XUserExt")
HARD_SLOTS = {("xgameruntime.XSystem", 4), ("xgameruntime.XSystem", 5)}
HARD_REASON = {
    "xgameruntime.XStore": "store/licence/entitlement",
    "xgameruntime.XStoreExt": "store/licence/entitlement",
    "xgameruntime.XUser": "user account / sign-in",
    "xgameruntime.XUserExt": "user account / sign-in",
    "xgameruntime.XSystem": "device identity / attestation-adjacent",
}


# ---------------------------------------------------------------------------
# Provider tables
# ---------------------------------------------------------------------------

GUID_RE = re.compile(r'constexpr\s+Guid\s+(\w+)\s*=\s*com::make_guid\(\s*"([0-9a-fA-F-]{36})"\s*\)')
ENTRY_RE = re.compile(r'(\w+)::(method|stub)\s*(?:<[^>]*>)?\s*\(\s*"([^"]*)"\s*\)')
SLOT_ENTRY_RE = re.compile(r'\{\s*(\d+)\s*,\s*(\w+)::(method|stub)\s*(?:<[^>]*>)?\s*\(\s*"([^"]*)"\s*\)')
SLOT_TABLE_RE = re.compile(r'constexpr\s+auto\s+(\w+)\s*=\s*slot_table<\s*\d+\s*>\(\{(.*?)\}\);', re.S)
METHOD_ARRAY_RE = re.compile(r'constexpr\s+com::MethodDesc\s+(\w+)\s*\[\]\s*=\s*\{(.*?)\};', re.S)
DECLARE_RE = re.compile(
    r'constexpr\s+auto\s+(\w+)\s*=\s*com::declare_interface<\s*\w+\s*>\(\s*"([^"]+)"\s*,\s*(\w+)\s*,\s*'
    r'([\w.]+|\{\s*\})\s*,\s*\{([^}]*)\}\s*\)', re.S)
API_RE = re.compile(r'\{\s*(k\w+)\s*,\s*(k\w+)\s*,\s*&(\w+)\.info')


def parse_provider(text: str) -> list[dict[str, Any]]:
    """Registered interfaces: label, provider GUID, iid, slot_count, policy and per-slot status."""
    guids = {n: g.lower() for n, g in GUID_RE.findall(text)}
    tables: dict[str, dict[int, tuple[str, str]]] = {}
    for name, body in SLOT_TABLE_RE.findall(text):
        tables[name] = {int(s): (nm, kind) for s, _, kind, nm in SLOT_ENTRY_RE.findall(body)}
    for name, body in METHOD_ARRAY_RE.findall(text):
        tables[name] = {3 + i: (nm, kind) for i, (_, kind, nm) in enumerate(ENTRY_RE.findall(body))}
    declared: dict[str, dict[str, Any]] = {}
    for var, label, _guid_name, table_ref, opts in DECLARE_RE.findall(text):
        slots = tables.get(table_ref.split(".")[0], {})
        count = re.search(r'\.slot_count\s*=\s*(\d+)', opts)
        policy = "ReturnFixed" if "ReturnFixed" in opts else "Trap"
        declared[var] = {"label": label, "slots": slots, "slot_count": int(count.group(1)) if count else 0,
                         "policy": policy}
    out = []
    for prov, iid, var in API_RE.findall(text):
        d = declared.get(var)
        if d is None or prov not in guids or iid not in guids:
            continue
        out.append({"label": d["label"], "provider": guids[prov], "iid": guids[iid], "slot_count": d["slot_count"],
                    "policy": d["policy"], "slots": d["slots"]})
    return out


def slot_status(iface: dict[str, Any], slot: int) -> tuple[str, str]:
    """(status, name) of one slot: implemented, trap, or fixed (a non-trapping fixed answer)."""
    entry = iface["slots"].get(slot)
    if entry is not None and entry[1] == "method":
        return "implemented", entry[0]
    name = entry[0] if entry else f"slot {slot}"
    if iface["policy"] == "ReturnFixed" and slot < iface["slot_count"]:
        return "fixed", name
    return "trap", name


def is_hard(label: str, slot: int) -> str | None:
    if label in HARD_INTERFACE_LABELS or (label, slot) in HARD_SLOTS:
        return HARD_REASON[label]
    return None


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def hx(text: str) -> int:
    return int(text, 16)


class Graph:
    def __init__(self, functions: Iterable[dict[str, Any]]):
        self.entries: list[int] = []
        self.out: dict[int, set[int]] = defaultdict(set)
        for f in functions:
            e = hx(f["entry"])
            self.entries.append(e)
            for c in list(f.get("callees", ())) + list(f.get("tail_calls", ())):
                self.out[e].add(hx(c))
        self.entries.sort()
        self.entry_set = set(self.entries)
        self.executed: set[int] = set()
        self._rev: dict[int, set[int]] | None = None

    def function_of(self, va: int) -> int | None:
        i = bisect.bisect_right(self.entries, va) - 1
        if i < 0 or va - self.entries[i] > UNKNOWN_FN_GAP:
            return None
        return self.entries[i]

    def add_trace(self, pairs: Iterable[tuple[int, int]]) -> int:
        """Adds dispatched (return-address site, target) pairs as edges; returns new edges."""
        n = 0
        for site, target in pairs:
            src = self.function_of(site - 1)
            dst = self.function_of(target)
            if dst is not None:
                self.executed.add(dst)
            if src is not None:
                self.executed.add(src)
                if dst is not None and dst not in self.out[src]:
                    self.out[src].add(dst)
                    n += 1
        self._rev = None
        return n

    def reverse(self) -> dict[int, set[int]]:
        if self._rev is None:
            rev: dict[int, set[int]] = defaultdict(set)
            for s, ts in self.out.items():
                for t in ts:
                    rev[t].add(s)
            self._rev = rev
        return self._rev

    def forward_depths(self, root: int) -> dict[int, int]:
        start = root if root in self.entry_set else self.function_of(root)
        if start is None:
            return {}
        depth = {start: 0}
        q = deque([start])
        while q:
            f = q.popleft()
            for t in self.out.get(f, ()):
                if t not in depth:
                    depth[t] = depth[f] + 1
                    q.append(t)
        return depth

    def ancestor_hops(self, fn: int, limit: int) -> int | None:
        """Fewest reverse hops from fn up to a function observed executing, within limit."""
        if fn in self.executed:
            return 0
        rev = self.reverse()
        seen = {fn}
        frontier = [fn]
        for hop in range(1, limit + 1):
            nxt = []
            for f in frontier:
                for p in rev.get(f, ()):
                    if p in seen:
                        continue
                    if p in self.executed:
                        return hop
                    seen.add(p)
                    nxt.append(p)
            frontier = nxt
            if not frontier:
                break
        return None


def parse_targets(text: str) -> list[tuple[int, int]]:
    pairs = []
    for raw in text.splitlines():
        line = raw.split("#")[0].split()
        if len(line) == 2:
            try:
                pairs.append((hx(line[0]), hx(line[1])))
            except ValueError:
                pass
    return pairs


# ---------------------------------------------------------------------------
# PE header (integers only) and import call sites
# ---------------------------------------------------------------------------

def read_pe_layout(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        head = f.read(0x1000)
    lfanew = struct.unpack_from("<I", head, 0x3C)[0]
    nsec, = struct.unpack_from("<H", head, lfanew + 6)
    opt_size, = struct.unpack_from("<H", head, lfanew + 20)
    opt = lfanew + 24
    entry_rva, = struct.unpack_from("<I", head, opt + 16)
    image_base, = struct.unpack_from("<Q", head, opt + 24)
    sec = opt + opt_size
    sections = []
    for i in range(nsec):
        name, vsize, va, rsize, raw, _r, _l, _n, _m, chars = struct.unpack_from("<8sIIIIIIHHI", head, sec + 40 * i)
        sections.append({"name": name.rstrip(b"\0").decode("latin1"), "va": va, "vsize": vsize,
                         "raw": raw, "rsize": rsize, "exec": bool(chars & 0x20000000)})
    return {"image_base": image_base, "entry": image_base + entry_rva, "sections": sections}


def find_iat_sites(path: Path, layout: dict[str, Any], slot_rvas: dict[int, str]) -> dict[str, list[int]]:
    """VAs of `call [rip+d]` / `jmp [rip+d]` through one of the given IAT slots (addresses only)."""
    found: dict[str, list[int]] = defaultdict(list)
    pat = re.compile(rb"\xff[\x15\x25]")
    with path.open("rb") as f:
        for s in layout["sections"]:
            if not s["exec"]:
                continue
            f.seek(s["raw"])
            data = f.read(min(s["rsize"], s["vsize"]))
            for m in pat.finditer(data):
                i = m.start()
                if i + 6 > len(data):
                    continue
                disp, = struct.unpack_from("<i", data, i + 2)
                name = slot_rvas.get(s["va"] + i + 6 + disp)
                if name is not None:
                    found[name].append(layout["image_base"] + s["va"] + i)
    return found


# ---------------------------------------------------------------------------
# Survey
# ---------------------------------------------------------------------------

TIER_RANK = {"A": 0, "B": 1, "C": 2}


def normalise_guid(g: str) -> str:
    return g.strip("{}").lower()


def classify_reach(graph: Graph, fwd: dict[int, int], site: int, anc_depth: int) -> dict[str, Any]:
    fn = graph.function_of(site)
    if fn is None:
        return {"tier": "C", "depth": None, "anc": None}
    if fn in fwd:
        return {"tier": "A", "depth": fwd[fn], "anc": None}
    anc = graph.ancestor_hops(fn, anc_depth)
    return {"tier": "B" if anc is not None else "C", "depth": None, "anc": anc}


def best_reach(reaches: list[dict[str, Any]]) -> dict[str, Any]:
    big = 10**6
    return min(reaches, key=lambda r: (TIER_RANK[r["tier"]], r["depth"] if r["depth"] is not None else big,
                                       r["anc"] if r["anc"] is not None else big))


def build_survey(provider: list[dict[str, Any]], survey: dict[str, Any], graph: Graph | None, entry: int | None,
                 stop: tuple[str, int], anc_depth: int, imports: dict[str, Any] | None) -> dict[str, Any]:
    by_key = {(i["provider"], i["iid"]): i for i in provider}
    fwd = graph.forward_depths(entry) if graph is not None and entry is not None else {}
    rows: list[dict[str, Any]] = []
    unregistered: list[dict[str, Any]] = []
    for src in survey.get("sources", []):
        for k in src.get("keys", []):
            if len(k["key"]) != 2:
                continue
            prov, iid = normalise_guid(k["key"][0]), normalise_guid(k["key"][1])
            iface = by_key.get((prov, iid))
            for sl in k["slots"]:
                slot = sl["slot"]
                if slot < 3:
                    continue  # IUnknown
                sites = [hx(a) for a in sl.get("examples", [])]
                if iface is None:
                    unregistered.append({"provider": prov, "iid": iid, "slot": slot, "sites": sl["sites"]})
                    continue
                status, name = slot_status(iface, slot)
                reach = best_reach([classify_reach(graph, fwd, s, anc_depth) for s in sites]) \
                    if graph is not None and sites else None
                rows.append({
                    "interface": iface["label"], "provider": prov, "iid": iid, "slot": slot, "name": name,
                    "status": status, "hard_stop": is_hard(iface["label"], slot),
                    "call_sites": sl["sites"], "wrapper_functions": sl["functions"],
                    "caller_sites": sl["caller_sites"], "min_args": sl["min_args"],
                    "tier": reach["tier"] if reach else "?", "depth": reach["depth"] if reach else None,
                    "anc_hops": reach["anc"] if reach else None,
                    "is_current_stop": (iface["label"], slot) == stop,
                    "_sites": sites,
                })
    # Only a trapping slot stops a boot; a fixed-answer stub answers. Order: reachability tier,
    # blast radius (direct callers of the wrappers), then closeness to the entry.
    ranked = [r for r in rows if r["status"] == "trap" and not r["hard_stop"] and not r["is_current_stop"]]
    ranked.sort(key=lambda r: (TIER_RANK.get(r["tier"], 3), -r["caller_sites"],
                               r["depth"] if r["depth"] is not None else 10**6, r["interface"], r["slot"]))
    hard = [r for r in rows if r["status"] == "trap" and r["hard_stop"]]
    hard.sort(key=lambda r: (TIER_RANK.get(r["tier"], 3), r["interface"], r["slot"]))
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["status"]] += 1
    stop_rows = [r for r in rows if r["is_current_stop"]]
    return {
        "schema": SCHEMA, "version": VERSION,
        "stop": {"interface": stop[0], "slot": stop[1]},
        "interfaces_registered": len(provider),
        "called_slots": len(rows), "by_status": dict(counts),
        "graph": None if graph is None else {
            "functions": len(graph.entries), "reachable_from_entry": len(fwd),
            "observed_executed": len(graph.executed)},
        "unregistered_interface_slots": unregistered,
        "current_stop": stop_rows[0] if stop_rows else None,
        "ranked": ranked, "hard_stops": hard, "all_called_slots": rows, "imports": imports,
    }


def summarise_imports(manifest: dict[str, Any], layout: dict[str, Any] | None, pe: Path | None,
                      graph: Graph | None, fwd: dict[int, int], anc_depth: int) -> dict[str, Any]:
    imps = manifest.get("imports", [])
    main_name = next((m.get("file") for m in manifest.get("modules", []) if m.get("role") == "exe"), None)
    counts: dict[str, int] = defaultdict(int)
    unresolved = []
    for i in imps:
        counts[i["binding"]] += 1
        if i["binding"] == "unresolved" and (main_name is None or i["importer"] == main_name):
            unresolved.append(i)

    def label(i: dict[str, Any]) -> str:
        return f'{i["dll"]}!{i["name"] or "#" + str(i["ordinal"])}'

    slot_rvas = {hx(i["slot_rva"]): label(i) for i in unresolved}
    sites = find_iat_sites(pe, layout, slot_rvas) if layout and pe and slot_rvas else {}
    rows = []
    for i in unresolved:
        lab = label(i)
        reach = best_reach([classify_reach(graph, fwd, s, anc_depth) for s in sites[lab]]) \
            if graph is not None and sites.get(lab) else None
        rows.append({"import": lab, "reason": i.get("reason"), "call_sites": len(sites.get(lab, [])),
                     "tier": reach["tier"] if reach else "?", "depth": reach["depth"] if reach else None})
    rows.sort(key=lambda r: (TIER_RANK.get(r["tier"], 3), -r["call_sites"], r["import"]))
    return {"total": len(imps), "by_binding": dict(counts), "unresolved_in_main_module": rows}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def public_row(r: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in r.items() if not k.startswith("_")}


def to_json(s: dict[str, Any], with_addresses: bool) -> str:
    out = dict(s)
    for key in ("ranked", "hard_stops", "all_called_slots"):
        rows = []
        for r in s[key]:
            p = public_row(r)
            if with_addresses:
                p["call_site_addresses"] = [f"0x{a:x}" for a in r.get("_sites", [])]
            rows.append(p)
        out[key] = rows
    if out["current_stop"]:
        out["current_stop"] = public_row(out["current_stop"])
    return json.dumps(out, indent=2)


def reach_text(r: dict[str, Any]) -> str:
    if r["tier"] == "A":
        return f"A (depth {r['depth']})"
    if r["tier"] == "B":
        return f"B ({r['anc_hops']} hops)"
    return r["tier"]


def to_markdown(s: dict[str, Any], top: int) -> str:
    L = ["# Platform surface survey (static)", ""]
    st = s["by_status"]
    L.append(f"Called slots (title, via QueryApiImpl): {s['called_slots']} = {st.get('implemented', 0)} implemented, "
             f"{st.get('trap', 0)} trapping, {st.get('fixed', 0)} fixed-answer stub. "
             f"Registered interfaces: {s['interfaces_registered']}.")
    g = s["graph"]
    if g:
        L.append(f"Call graph: {g['functions']} functions, {g['reachable_from_entry']} reachable from the entry "
                 f"over direct + recorded indirect edges, {g['observed_executed']} observed executing.")
    cs = s["current_stop"]
    if cs:
        L.append(f"Current stop: {cs['interface']} slot {cs['slot']} {cs['name']} (excluded from the ranking).")
    L += ["", "Reach: A = call site reachable from entry by the graph; B = within a few reverse hops of code "
          "observed running (code-pointer dispatch); C = no evidence. Callers = direct callers of the wrapper "
          "functions (blast radius).", "",
          "## Ranked: trapping slots, not hard stops, excluding the current stop", "",
          "| # | Interface | Slot | Name | Call sites | Callers | Reach |", "|---|---|---|---|---|---|---|"]
    for n, r in enumerate(s["ranked"][:top], 1):
        L.append(f"| {n} | {r['interface']} | {r['slot']} | {r['name']} | {r['call_sites']} | "
                 f"{r['caller_sites']} | {reach_text(r)} |")
    if len(s["ranked"]) > top:
        L.append(f"| | ... {len(s['ranked']) - top} more | | | | | |")
    L += ["", "## Hard stops (owner decision; never implemented by agents)", "",
          "| Interface | Slot | Name | Why | Call sites | Reach |", "|---|---|---|---|---|---|"]
    for r in s["hard_stops"]:
        L.append(f"| {r['interface']} | {r['slot']} | {r['name']} | {r['hard_stop']} | {r['call_sites']} | "
                 f"{reach_text(r)} |")
    if s["unregistered_interface_slots"]:
        L += ["", "## Called slots of interfaces the runtime does not register"]
        for u in s["unregistered_interface_slots"]:
            L.append(f"- {{{u['provider']}}}/{{{u['iid']}}} slot {u['slot']} ({u['sites']} sites)")
    im = s["imports"]
    if im:
        L += ["", "## Imports (program manifest)", "",
              f"{im['total']} imports across modules; bindings: " +
              ", ".join(f"{k} {v}" for k, v in sorted(im["by_binding"].items())) + "."]
        if im["unresolved_in_main_module"]:
            L += ["", "| Unresolved import (main module) | Reason | Call sites | Reach |", "|---|---|---|---|"]
            for r in im["unresolved_in_main_module"][:top]:
                L.append(f"| {r['import']} | {r['reason']} | {r['call_sites']} | {r['tier']} |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_recomp(recomp: Path, args: list[str]) -> None:
    proc = subprocess.run([str(recomp), *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"{recomp.name} {args[0]} failed: {proc.stderr.strip()[:300]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Static platform-surface survey ranked by boot-stop likelihood.")
    ap.add_argument("--pe", type=Path, help="the title executable (read only: header and import-slot scan)")
    ap.add_argument("--recomp", type=Path, help="playgta6-recomp binary, to produce missing --survey-json/--discover-json")
    ap.add_argument("--provider-src", type=Path,
                    default=Path(__file__).resolve().parent.parent / DEFAULT_PROVIDER_SRC)
    ap.add_argument("--survey-json", type=Path, help="survey-interfaces --json output (source: QueryApiImpl)")
    ap.add_argument("--discover-json", type=Path, help="discover --out output (callees per function)")
    ap.add_argument("--targets", type=Path, help="recorded indirect targets (#1654 file)")
    ap.add_argument("--program-manifest", type=Path, help="build-program program-manifest.json (imports)")
    ap.add_argument("--stop", default="xgameruntime.XSystem:5", help="current stop, <interface label>:<slot>")
    ap.add_argument("--anc-depth", type=int, default=DEFAULT_ANC_DEPTH)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--md-out", type=Path)
    ap.add_argument("--with-addresses", action="store_true", help="keep call-site addresses in the JSON (local use)")
    a = ap.parse_args(argv)

    survey_path, discover_path = a.survey_json, a.discover_json
    if survey_path is None or discover_path is None:
        if not (a.recomp and a.pe):
            ap.error("--survey-json and --discover-json need --recomp and --pe to be produced")
        work = (a.json_out or a.md_out or Path("surface_survey")).resolve().parent
        if survey_path is None:
            survey_path = work / "surface_survey.qapi.json"
            run_recomp(a.recomp, ["survey-interfaces", str(a.pe), "--source", QAPI_SOURCE, "--examples", "1000",
                                  "--json", "--out", str(survey_path)])
        if discover_path is None:
            discover_path = work / "surface_survey.discover.json"
            run_recomp(a.recomp, ["discover", str(a.pe), "--out", str(discover_path)])

    provider = parse_provider(a.provider_src.read_text(encoding="utf-8", errors="replace"))
    survey = json.loads(survey_path.read_text(encoding="utf-8"))
    graph = Graph(json.loads(discover_path.read_text(encoding="utf-8"))["functions"])
    if a.targets and a.targets.is_file():
        graph.add_trace(parse_targets(a.targets.read_text()))
    layout = read_pe_layout(a.pe) if a.pe else None
    entry = layout["entry"] if layout else None
    label, _, slot = a.stop.rpartition(":")
    imports = None
    if a.program_manifest:
        fwd = graph.forward_depths(entry) if entry is not None else {}
        imports = summarise_imports(json.loads(a.program_manifest.read_text(encoding="utf-8")), layout, a.pe,
                                    graph, fwd, a.anc_depth)
    result = build_survey(provider, survey, graph, entry, (label, int(slot)), a.anc_depth, imports)
    md = to_markdown(result, a.top)
    if a.json_out:
        a.json_out.write_text(to_json(result, a.with_addresses), encoding="utf-8")
    if a.md_out:
        a.md_out.write_text(md, encoding="utf-8")
    if not a.json_out and not a.md_out:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
