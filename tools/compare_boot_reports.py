#!/usr/bin/env python3
"""Compare two native boot runs and print a structured diff with a pass/fail verdict.

Inputs are the artifacts of `playgta6_boot` (runtime/guest/boot_main.cpp):

  * the boot report JSON (`--report`, schema "playgta6-boot-report"), always required;
  * optionally the GPU captures (`--gpu-capture`, a `.pg6trace` plus its `<trace>.json`
    sidecar, schema "playgta6-frame-capture"), passed with `--capture A.pg6trace B.pg6trace`;
  * optionally the import traces (`--import-trace`), passed with
    `--import-trace A.txt B.txt`.

Report schema 2 (R42) carries what schema 1 needed the extra files for: `milestones[].import_seq`,
`presents` (the per-Present samples of a `--gpu-capture` run) and `first_present` (import calls
per guest thread and name up to the first Present). The tool prefers these report fields and
falls back, per side, to the capture sidecar `presents` and to the import trace (bounded by the
first Present's import-call index in the sidecar) for schema 1 reports. Both versions are read.

What is compared (section names are the prefixes of the field paths used by --ignore):

  milestones   named boot milestones reached (set and order exact), elapsed_ms (time tolerance),
               import_seq (count tolerance; schema 2 reports)
  first_calls  first call of the timeline APIs (device creation, ...): name set exact,
               import-call index (count tolerance), elapsed_ms (time tolerance)
  stop         status, exit_code, phase, fault kind / trap kind / address / function / import,
               backtrace method and function addresses: all exact
  imports      total import calls and per-name call counts from the report (count tolerance)
  address_space  reservation log sequence (event, ok, va, size, protect) exact, plus failure
               count and reserved peak
  import_trace import-call counts by name up to the first Present, overall and on the busiest
               thread (count tolerance); from the report's first_present or the import traces
  presents     per-Present sample distributions from the reports or the sidecars (min / median / max of
               dispatches, import calls, reserved / committed bytes, every named counter,
               and wall_ms with the time tolerance)
  capture      frame by frame: packet count, byte count and per-opcode packet counts (exact);
               omitted-payload hashes and frames_complete

Tolerances: a value pair matches when |a - b| <= max(floor, rel * max(|a|, |b|)).
  --time-tolerance REL / --time-floor-ms MS   for elapsed_ms and wall_ms (default 0.5, 100)
  --count-tolerance REL                        for wall-clock dependent counts (default 0.05)
Everything else must match exactly. `--ignore PATTERN` (repeatable, fnmatch on the field path,
e.g. `milestones.*.elapsed_ms` or `address_space.*`) drops known-different fields.

Output is text by default and `--json` for automation. Exit code: 0 pass, 1 differences,
2 usage error (missing or malformed input).

    python tools/compare_boot_reports.py a.json b.json
    python tools/compare_boot_reports.py a.json b.json --capture a.pg6trace b.pg6trace --json
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_trace  # noqa: E402

REPORT_SCHEMA = "playgta6-boot-report"
CAPTURE_SCHEMA = "playgta6-frame-capture"


class InputError(Exception):
    """Unusable input: reported as a usage error (exit code 2)."""


@dataclass
class Tolerances:
    time_rel: float = 0.5
    time_floor_ms: float = 100.0
    count_rel: float = 0.05

    def within(self, a: float, b: float, rel: float, floor: float = 0.0) -> bool:
        return abs(a - b) <= max(floor, rel * max(abs(a), abs(b)))


@dataclass
class Comparison:
    tol: Tolerances
    ignore: list[str]
    entries: list[dict[str, Any]] = field(default_factory=list)

    def _ignored(self, path: str) -> bool:
        return any(fnmatch.fnmatchcase(path, p) for p in self.ignore)

    def record(self, path: str, a: Any, b: Any, ok: bool, mode: str) -> None:
        if self._ignored(path):
            return
        self.entries.append({"field": path, "a": a, "b": b, "mode": mode, "ok": ok})

    def exact(self, path: str, a: Any, b: Any) -> None:
        self.record(path, a, b, a == b, "exact")

    def time(self, path: str, a: Any, b: Any) -> None:
        self._num(path, a, b, self.tol.time_rel, self.tol.time_floor_ms, "time")

    def count(self, path: str, a: Any, b: Any) -> None:
        self._num(path, a, b, self.tol.count_rel, 0.0, "count")

    def _num(self, path: str, a: Any, b: Any, rel: float, floor: float, mode: str) -> None:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            ok = self.tol.within(float(a), float(b), rel, floor)
        else:
            ok = a == b
        self.record(path, a, b, ok, mode)

    def mapping(self, prefix: str, a: dict[str, Any], b: dict[str, Any], cmp: str) -> None:
        """Compare two name -> number maps key by key; a missing key counts as absent."""
        fn = getattr(self, cmp)
        for key in sorted(set(a) | set(b)):
            fn(f"{prefix}.{key}", a.get(key), b.get(key))

    def failures(self) -> list[dict[str, Any]]:
        return [e for e in self.entries if not e["ok"]]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_json(path: Path, schema: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema") != schema:
        raise InputError(f"{path} is not a {schema} file")
    return data


def load_import_trace(path: Path) -> list[tuple[int, int, str]]:
    """Parses `--import-trace` lines `<seq> <thread> <label> <args...>` to (seq, thread, label)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc
    rows = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 3)
        if len(parts) < 3 or not parts[0].isdigit() or not parts[1].isdigit():
            raise InputError(f"{path}: malformed import trace line: {line[:60]!r}")
        rows.append((int(parts[0]), int(parts[1]), parts[2]))
    return rows


@dataclass
class Capture:
    sidecar: dict[str, Any]
    trace: gpu_trace.DecodedTrace


def load_capture(trace_path: Path) -> Capture:
    sidecar = load_json(Path(str(trace_path) + ".json"), CAPTURE_SCHEMA)
    try:
        raw = trace_path.read_bytes()
    except OSError as exc:
        raise InputError(f"cannot read {trace_path}: {exc}") from exc
    return Capture(sidecar, gpu_trace.decode_stream(raw))


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def keyed_milestones(report: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Milestones with unique keys: a repeated name (one PGIM_LOAD per module) becomes NAME#2, NAME#3."""
    seen: Counter = Counter()
    out = []
    for m in report.get("milestones") or []:
        seen[m["milestone"]] += 1
        n = seen[m["milestone"]]
        out.append((m["milestone"] if n == 1 else f"{m['milestone']}#{n}", m))
    return out


def compare_milestones(c: Comparison, ra: dict[str, Any], rb: dict[str, Any]) -> None:
    ka, kb = keyed_milestones(ra), keyed_milestones(rb)
    ma = [(k, m.get("elapsed_ms")) for k, m in ka]
    mb = [(k, m.get("elapsed_ms")) for k, m in kb]
    c.exact("milestones.sequence", [n for n, _ in ma], [n for n, _ in mb])
    ta, tb = dict(ma), dict(mb)
    for name in sorted(set(ta) | set(tb)):
        c.time(f"milestones.{name}.elapsed_ms", ta.get(name), tb.get(name))
    # import_seq exists from report schema 2 on; compare it only when both sides have it.
    sa = {k: m["import_seq"] for k, m in ka if "import_seq" in m}
    sb = {k: m["import_seq"] for k, m in kb if "import_seq" in m}
    for name in sorted(set(sa) & set(sb)):
        c.count(f"milestones.{name}.import_seq", sa[name], sb[name])
    fa = {f["name"]: f for f in ra.get("first_calls") or []}
    fb = {f["name"]: f for f in rb.get("first_calls") or []}
    c.exact("first_calls.names", sorted(fa), sorted(fb))
    for name in sorted(set(fa) & set(fb)):
        c.count(f"first_calls.{name}.import_seq", fa[name].get("import_seq"), fb[name].get("import_seq"))
        c.time(f"first_calls.{name}.elapsed_ms", fa[name].get("elapsed_ms"), fb[name].get("elapsed_ms"))


def compare_stop(c: Comparison, ra: dict[str, Any], rb: dict[str, Any]) -> None:
    for key in ("status", "exit_code", "phase"):
        c.exact(f"stop.{key}", ra.get(key), rb.get(key))
    fa, fb = ra.get("fault"), rb.get("fault")
    c.exact("stop.fault.present", fa is not None, fb is not None)
    if fa is not None and fb is not None:
        for key in ("kind", "trap_kind", "va", "detail", "exception_code"):
            c.exact(f"stop.fault.{key}", fa.get(key), fb.get(key))
        c.exact("stop.fault.function", fa.get("function"), fb.get("function"))
        c.exact("stop.fault.import", fa.get("import"), fb.get("import"))
    ba, bb = ra.get("backtrace") or {}, rb.get("backtrace") or {}
    c.exact("stop.backtrace.method", ba.get("method"), bb.get("method"))
    c.exact(
        "stop.backtrace.functions",
        [f.get("function") for f in ba.get("frames") or []],
        [f.get("function") for f in bb.get("frames") or []],
    )


def compare_imports(c: Comparison, ra: dict[str, Any], rb: dict[str, Any]) -> None:
    ca, cb = ra.get("counters") or {}, rb.get("counters") or {}
    c.count("imports.total_calls", ca.get("import_calls"), cb.get("import_calls"))

    def by_name(counters: dict[str, Any]) -> dict[str, int]:
        return {f"{i.get('dll')}!{i.get('symbol')}": i.get("count", 0) for i in counters.get("imports") or []}

    c.mapping("imports.calls", by_name(ca), by_name(cb), "count")


def compare_address_space(c: Comparison, ra: dict[str, Any], rb: dict[str, Any]) -> None:
    sa, sb = ra.get("address_space") or {}, rb.get("address_space") or {}
    c.exact("address_space.failures", sa.get("failures"), sb.get("failures"))
    c.exact("address_space.reserved_peak", sa.get("reserved_peak"), sb.get("reserved_peak"))
    c.exact("address_space.events", sa.get("events"), sb.get("events"))

    def seq(s: dict[str, Any]) -> list[tuple]:
        return [(e.get("event"), e.get("ok"), e.get("va"), e.get("size"), e.get("protect")) for e in s.get("log") or []]

    la, lb = seq(sa), seq(sb)
    first = next((i for i, (x, y) in enumerate(zip(la, lb)) if x != y), None)
    if first is None and len(la) != len(lb):
        first = min(len(la), len(lb))
    c.record(
        "address_space.log",
        {"entries": len(la), "first_difference": first, "entry": la[first] if first is not None and first < len(la) else None},
        {"entries": len(lb), "first_difference": first, "entry": lb[first] if first is not None and first < len(lb) else None},
        first is None,
        "exact",
    )


@dataclass
class FirstPresent:
    import_seq: int | None
    overall: dict[str, int]
    busiest: dict[str, int]


def trace_counts(rows: list[tuple[int, int, str]], limit: int | None) -> tuple[Counter, Counter]:
    rows = [r for r in rows if limit is None or r[0] <= limit]
    overall = Counter(label for _, _, label in rows)
    threads = Counter(t for _, t, _ in rows)
    busiest = threads.most_common(1)[0][0] if threads else None
    return overall, Counter(label for _, t, label in rows if t == busiest)


def first_present_from_report(report: dict[str, Any]) -> FirstPresent | None:
    """Schema 2: the report's own `first_present` (null or absent: not available)."""
    fp = report.get("first_present")
    if not isinstance(fp, dict):
        return None
    overall: Counter = Counter()
    per_thread: dict[Any, Counter] = {}
    for row in fp.get("imports") or []:
        name = f"{row.get('dll')}!{row.get('symbol')}"
        overall[name] += row.get("count", 0)
        per_thread.setdefault(row.get("thread"), Counter())[name] += row.get("count", 0)
    busiest = max(per_thread.values(), key=lambda t: sum(t.values()), default=Counter())
    return FirstPresent(fp.get("import_seq"), dict(overall), dict(busiest))


def first_present_from_trace(rows: list | None, cap: Capture | None) -> FirstPresent | None:
    """Schema 1 fallback: the import trace bounded by the sidecar's first Present."""
    presents = (cap.sidecar.get("presents") or []) if cap else []
    if rows is None or not presents:
        return None
    seq = presents[0].get("import_calls")
    overall, busiest = trace_counts(rows, seq)
    return FirstPresent(seq, dict(overall), dict(busiest))


def compare_first_present(c: Comparison, a: FirstPresent | None, b: FirstPresent | None) -> None:
    c.exact("import_trace.first_present_seen", a is not None, b is not None)
    if a is None or b is None:
        return
    c.count("import_trace.first_present_import_call", a.import_seq, b.import_seq)
    c.mapping("import_trace.overall", a.overall, b.overall, "count")
    c.mapping("import_trace.busiest_thread", a.busiest, b.busiest, "count")


def distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def present_series(presents: list[dict[str, Any]]) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {}
    for p in presents:
        flat = {k: p.get(k) for k in ("wall_ms", "dispatches", "import_calls", "reserved_bytes", "committed_bytes")}
        flat.update({f"counters.{k}": v for k, v in (p.get("counters") or {}).items()})
        for key, value in flat.items():
            if isinstance(value, (int, float)):
                series.setdefault(key, []).append(value)
    return series


def presents_of(report: dict[str, Any], cap: Capture | None) -> list[dict[str, Any]] | None:
    """The report's `presents` (schema 2) if it has samples, else the capture sidecar's."""
    own = report.get("presents")
    if isinstance(own, list) and own:
        return own
    if cap is not None:
        return cap.sidecar.get("presents") or []
    return None


def compare_presents(c: Comparison, pa: list[dict[str, Any]], pb: list[dict[str, Any]]) -> None:
    c.count("presents.count", len(pa), len(pb))
    sa, sb = present_series(pa), present_series(pb)
    for key in sorted(set(sa) | set(sb)):
        da, db = distribution(sa.get(key, [])), distribution(sb.get(key, []))
        for stat in ("min", "median", "max"):
            path = f"presents.{key}.{stat}"
            (c.time if key == "wall_ms" else c.count)(path, da[stat], db[stat])


def compare_captures(c: Comparison, ca: Capture, cb: Capture) -> None:
    for key in ("frames_complete", "payloads"):
        c.exact(f"capture.{key}", ca.sidecar.get(key), cb.sidecar.get(key))
    c.exact("capture.diagnostics", len(ca.trace.diagnostics), len(cb.trace.diagnostics))
    c.exact(
        "capture.omitted_hashes",
        [o.get("fnv1a64") for o in ca.sidecar.get("omitted") or []],
        [o.get("fnv1a64") for o in cb.sidecar.get("omitted") or []],
    )
    fa, fb = ca.trace.frames, cb.trace.frames
    c.exact("capture.frame_count", len(fa), len(fb))
    for i in range(max(len(fa), len(fb))):
        a = fa[i] if i < len(fa) else None
        b = fb[i] if i < len(fb) else None
        if a is None or b is None:
            c.record(f"capture.frame[{i}]", "present" if a else "missing", "present" if b else "missing", False, "exact")
            continue
        c.exact(f"capture.frame[{i}].packets", a.packet_count, b.packet_count)
        c.exact(f"capture.frame[{i}].bytes", a.bytes, b.bytes)
        ha, hb = a.opcode_histogram, b.opcode_histogram
        for op in sorted(set(ha) | set(hb)):
            c.exact(f"capture.frame[{i}].opcode.{op}", ha.get(op), hb.get(op))


def compare(
    ra: dict[str, Any],
    rb: dict[str, Any],
    tol: Tolerances,
    ignore: Sequence[str] = (),
    captures: tuple[Capture, Capture] | None = None,
    import_traces: tuple[list, list] | None = None,
) -> Comparison:
    c = Comparison(tol, list(ignore))
    compare_milestones(c, ra, rb)
    compare_stop(c, ra, rb)
    compare_imports(c, ra, rb)
    compare_address_space(c, ra, rb)
    caps = captures or (None, None)
    traces = import_traces or (None, None)
    fa = first_present_from_report(ra) or first_present_from_trace(traces[0], caps[0])
    fb = first_present_from_report(rb) or first_present_from_trace(traces[1], caps[1])
    if fa is not None or fb is not None or import_traces:
        compare_first_present(c, fa, fb)
    pa, pb = presents_of(ra, caps[0]), presents_of(rb, caps[1])
    if pa is not None and pb is not None:
        compare_presents(c, pa, pb)
    if captures:
        compare_captures(c, *captures)
    return c


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def to_result(c: Comparison) -> dict[str, Any]:
    bad = c.failures()
    return {
        "verdict": "fail" if bad else "pass",
        "checked": len(c.entries),
        "differences": len(bad),
        "tolerances": {
            "time_rel": c.tol.time_rel,
            "time_floor_ms": c.tol.time_floor_ms,
            "count_rel": c.tol.count_rel,
        },
        "ignore": c.ignore,
        "entries": bad,
    }


def format_text(c: Comparison, verbose: bool = False) -> str:
    result = to_result(c)
    lines = []
    section = None
    for e in c.entries if verbose else c.failures():
        head = e["field"].split(".", 1)[0].split("[", 1)[0]
        if head != section:
            section = head
            lines.append(f"[{section}]")
        if e["ok"]:
            lines.append(f"  ok   {e['field']}: {e['a']!r} ({e['mode']})")
        else:
            lines.append(f"  DIFF {e['field']}: {e['a']!r} != {e['b']!r} ({e['mode']})")
    lines.append(
        f"verdict: {result['verdict'].upper()} ({result['differences']} difference(s) in {result['checked']} checks)"
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare two native boot runs (playgta6_boot reports, optional GPU captures and "
        "import traces) and print a structured diff with a pass/fail verdict.",
        epilog="Exit code: 0 pass, 1 differences, 2 usage error. See the module docstring for the compared fields.",
    )
    p.add_argument("report_a", type=Path, help="boot report JSON of run A (playgta6_boot --report)")
    p.add_argument("report_b", type=Path, help="boot report JSON of run B")
    p.add_argument("--capture", nargs=2, type=Path, metavar=("A", "B"),
                   help="GPU captures (.pg6trace; the .json sidecar is read next to each)")
    p.add_argument("--import-trace", nargs=2, type=Path, metavar=("A", "B"),
                   help="import traces (playgta6_boot --import-trace); needs --capture")
    p.add_argument("--ignore", action="append", default=[], metavar="PATTERN",
                   help="fnmatch pattern on a field path to skip, repeatable (e.g. 'milestones.*.elapsed_ms')")
    p.add_argument("--time-tolerance", type=float, default=0.5, metavar="REL",
                   help="relative tolerance for elapsed_ms / wall_ms (default 0.5)")
    p.add_argument("--time-floor-ms", type=float, default=100.0, metavar="MS",
                   help="absolute tolerance floor for timings in ms (default 100)")
    p.add_argument("--count-tolerance", type=float, default=0.05, metavar="REL",
                   help="relative tolerance for wall-clock dependent counts (default 0.05)")
    p.add_argument("--json", dest="as_json", action="store_true", help="print the result as JSON")
    p.add_argument("-v", "--verbose", action="store_true", help="also list the matching fields (text mode)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code in (0, None) else 2
    if args.import_trace and not args.capture:
        print("error: --import-trace needs --capture", file=sys.stderr)
        return 2
    if min(args.time_tolerance, args.time_floor_ms, args.count_tolerance) < 0:
        print("error: tolerances must not be negative", file=sys.stderr)
        return 2
    try:
        ra = load_json(args.report_a, REPORT_SCHEMA)
        rb = load_json(args.report_b, REPORT_SCHEMA)
        captures = tuple(load_capture(p) for p in args.capture) if args.capture else None
        traces = tuple(load_import_trace(p) for p in args.import_trace) if args.import_trace else None
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    tol = Tolerances(args.time_tolerance, args.time_floor_ms, args.count_tolerance)
    c = compare(ra, rb, tol, args.ignore, captures, traces)  # type: ignore[arg-type]
    if args.as_json:
        print(json.dumps(to_result(c), indent=2))
    else:
        print(format_text(c, args.verbose))
    return 1 if c.failures() else 0


if __name__ == "__main__":
    raise SystemExit(main())
