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
  gpu_work     draws / dispatches issued, emitted and skipped by reason (count tolerance;
               both reports must carry it)
  audio        totals, non_silent_fraction and per-window non-silent frames (count tolerance;
               both reports must carry it)
  memory.alias XMem alias_slots / alias_conflicts (count tolerance; both reports must carry it)
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
The report's `screen_timeline` (R40, tools/screen_timeline.py) is informational and never compared.

Output is text by default, `--markdown` for Markdown tables, and `--json` for automation. When
GPU/draw/frame-pacing metrics (including `FIRST_VISIBLE_FRAME` and `black_loading_ms` /
`blackLoadingMs`) are present in both reports, an additive, strictly-sanitized `gpu_delta` summary
is included without altering pass/fail verdicts or exit codes. When Chrome boot telemetry metrics
(`first_visible_frame_s`, `black_loading_s`, `js_heap_used_mb`, `js_heap_total_mb`,
`audio_ring_attached`, `audio_frames_drained`, `present_fps_avg`) are present in either report, an
additive, strictly-sanitized `telemetry_delta` summary is included without altering pass/fail
verdicts or exit codes (unless `--min-fps-ratio` is passed to flag `present_fps_avg` regressions).
When sanitized `gpu_adapter` labels differ between baseline and candidate, an informational warning
note is emitted without altering pass/fail verdicts or exit codes. Exit code: 0 pass, 1
differences, 2 usage error (missing or malformed input).

    python tools/compare_boot_reports.py a.json b.json
    python tools/compare_boot_reports.py a.json b.json --capture a.pg6trace b.pg6trace --json
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import re
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
    min_fps_ratio: float | None = None
    entries: list[dict[str, Any]] = field(default_factory=list)
    gpu_delta: dict[str, dict[str, int | float | None]] = field(default_factory=dict)
    telemetry_delta: dict[str, dict[str, Any]] = field(default_factory=dict)
    gpu_adapter: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)

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


def compare_progress(c: Comparison, ra: dict[str, Any], rb: dict[str, Any]) -> None:
    """R40 progress counters: `gpu_work` (draws / dispatches emitted vs skipped by reason) and the
    XMem alias stats (`memory.cpu_window.alias_*`). Wall-clock dependent, so count tolerance; each
    section is compared only when both reports carry it (older reports lack them)."""
    wa, wb = ra.get("gpu_work"), rb.get("gpu_work")
    if isinstance(wa, dict) and isinstance(wb, dict):
        c.mapping("gpu_work", wa, wb, "count")

    aua, aub = ra.get("audio"), rb.get("audio")
    if isinstance(aua, dict) and isinstance(aub, dict):
        for key in sorted(set(aua) | set(aub)):
            va, vb = aua.get(key), aub.get(key)
            if key == "non_silent_frames_per_window":
                va, vb = va or [], vb or []
                c.count("audio.windows", len(va), len(vb))
                for i in range(min(len(va), len(vb))):
                    c.count(f"audio.{key}[{i}]", va[i], vb[i])
            else:
                c.count(f"audio.{key}", va, vb)

    def alias(r: dict[str, Any]) -> dict[str, Any]:
        mem = r.get("memory") if isinstance(r.get("memory"), dict) else {}
        win = mem.get("cpu_window") if isinstance(mem.get("cpu_window"), dict) else {}
        return {k: win[k] for k in ("alias_slots", "alias_conflicts") if k in win}

    aa, ab = alias(ra), alias(rb)
    if aa and ab:
        c.mapping("memory.alias", aa, ab, "count")


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


GPU_DELTA_METRICS: tuple[str, ...] = (
    "real_pack_draws",
    "fallback_draws",
    "skipped_draws",
    "gpu_validation_errors",
    "non_black_canvas_frames",
    "first_present_ms",
    "first_visible_frame_ms",
    "black_loading_ms",
    "presents",
    "present_fps",
    "frame_time_ms",
    "frame_time_median_ms",
)

_GPU_DELTA_UNITS: dict[str, str] = {
    "first_present_ms": " ms",
    "first_visible_frame_ms": " ms",
    "black_loading_ms": " ms",
    "present_fps": " fps",
    "frame_time_ms": " ms",
    "frame_time_median_ms": " ms",
}

_GPU_ADAPTER_PATH_RE = re.compile(
    r"(?:\\\\|[A-Za-z]:[\\/]|(?:^|[\s(])/(?:Users|home|tmp|var|opt|root|mnt|private|etc|usr|Library|Volumes)(?:/|$))"
)
_GPU_ADAPTER_SAFE_CHAR_RE = re.compile(r"[^A-Za-z0-9 ._/+:(),\-]")
_GPU_ADAPTER_MAX_LEN = 96


def sanitize_gpu_adapter(raw: Any) -> str | None:
    """Return a strictly-sanitized WebGPU adapter label (e.g. 'nvidia / turing'), or None."""
    if isinstance(raw, dict):
        parts: list[str] = []
        for key in ("vendor", "architecture"):
            val = sanitize_gpu_adapter(raw.get(key))
            if val is not None:
                parts.append(val)
        return " / ".join(parts) if parts else None
    if not isinstance(raw, str):
        return None
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in raw):
        return None
    if "\\" in raw or _GPU_ADAPTER_PATH_RE.search(raw):
        return None
    cleaned = _GPU_ADAPTER_SAFE_CHAR_RE.sub("", raw)
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > _GPU_ADAPTER_MAX_LEN:
        cleaned = cleaned[:_GPU_ADAPTER_MAX_LEN].rstrip()
    if not cleaned or not any(ch.isalnum() for ch in cleaned):
        return None
    return cleaned


def _canonical_gpu_adapter(label: str) -> str:
    """Normalize an already-sanitized adapter label for case/spacing-insensitive comparison."""
    parts = [part.strip().lower() for part in label.split("/") if part.strip()]
    return " / ".join(parts) if parts else label.strip().lower()


def extract_gpu_adapter(report: dict[str, Any]) -> str | None:
    """Extract sanitized `gpu_adapter` from top-level or `summary` of a boot report."""
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    for candidate in (
        report.get("gpu_adapter"),
        report.get("gpuAdapter"),
        summary.get("gpu_adapter"),
        summary.get("gpuAdapter"),
    ):
        sanitized = sanitize_gpu_adapter(candidate)
        if sanitized is not None:
            return sanitized
    return None


def compare_gpu_adapter(
    ra: dict[str, Any],
    rb: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Compare sanitized GPU adapter labels; returns (adapter_info, informational_warning)."""
    adapter_a = extract_gpu_adapter(ra)
    adapter_b = extract_gpu_adapter(rb)
    if adapter_a is None and adapter_b is None:
        return None, None
    mismatch = (
        adapter_a is not None
        and adapter_b is not None
        and _canonical_gpu_adapter(adapter_a) != _canonical_gpu_adapter(adapter_b)
    )
    info: dict[str, Any] = {
        "a": adapter_a,
        "b": adapter_b,
        "mismatch": mismatch,
    }
    warning: str | None = None
    if mismatch:
        warning = (
            f"gpu_adapter mismatch: baseline '{adapter_a}' vs candidate '{adapter_b}' "
            "(timing deltas may reflect different GPU adapters, not a regression)"
        )
    return info, warning


def _finite_number(value: Any) -> int | float | None:
    """Return finite numeric values only (rejects bool, non-numeric, NaN, Inf)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _first_number(*candidates: Any) -> int | float | None:
    for candidate in candidates:
        num = _finite_number(candidate)
        if num is not None:
            return num
    return None


def _non_negative_number(value: Any) -> int | float | None:
    num = _finite_number(value)
    if num is None or float(num) < 0.0:
        return None
    return num


def _first_non_negative_number(*candidates: Any) -> int | float | None:
    for candidate in candidates:
        num = _non_negative_number(candidate)
        if num is not None:
            return num
    return None


def _strict_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _first_bool(*candidates: Any) -> bool | None:
    for candidate in candidates:
        b = _strict_bool(candidate)
        if b is not None:
            return b
    return None


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def extract_gpu_metrics(
    report: dict[str, Any],
    presents: list[dict[str, Any]] | None = None,
) -> dict[str, int | float]:
    """Extract strictly-sanitized numeric GPU/draw/frame-pacing metrics from one report."""
    counters = report.get("counters") if isinstance(report.get("counters"), dict) else {}
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    sample_list = presents if isinstance(presents, list) else (
        report.get("presents") if isinstance(report.get("presents"), list) else []
    )
    last_present_counters: dict[str, Any] = {}
    for sample in reversed(sample_list):
        if isinstance(sample, dict) and isinstance(sample.get("counters"), dict):
            last_present_counters = sample["counters"]
            break

    metrics: dict[str, int | float] = {}

    def set_if_present(name: str, value: int | float | None) -> None:
        if value is not None:
            metrics[name] = value

    set_if_present(
        "real_pack_draws",
        _first_number(
            counters.get("real_pack_draws"),
            summary.get("realPackDrawCount"),
            summary.get("real_pack_draws"),
            last_present_counters.get("draws_real_pack"),
        ),
    )
    set_if_present(
        "fallback_draws",
        _first_number(
            counters.get("fallback_draws"),
            summary.get("fallbackDrawCount"),
            summary.get("fallback_draws"),
            last_present_counters.get("draws_fallback"),
        ),
    )
    set_if_present(
        "skipped_draws",
        _first_number(
            counters.get("skipped_draws"),
            summary.get("skippedDrawCount"),
            summary.get("skipped_draws"),
            last_present_counters.get("draws_skipped"),
        ),
    )
    set_if_present(
        "gpu_validation_errors",
        _first_number(
            counters.get("gpu_validation_errors"),
            summary.get("validationErrorCount"),
            summary.get("gpu_validation_errors"),
            last_present_counters.get("gpu_validation_errors"),
        ),
    )
    set_if_present(
        "non_black_canvas_frames",
        _first_number(
            summary.get("nonBlackCanvasFrames"),
            summary.get("non_black_canvas_frames"),
            counters.get("non_black_canvas_frames"),
            report.get("nonBlackCanvasFrames"),
            report.get("non_black_canvas_frames"),
        ),
    )

    first_present_elapsed: Any = None
    fp = report.get("first_present")
    if isinstance(fp, dict):
        first_present_elapsed = fp.get("elapsed_ms")
    milestone_first_present: Any = None
    for m in report.get("milestones") or []:
        if isinstance(m, dict) and m.get("milestone") == "FIRST_PRESENT":
            milestone_first_present = m.get("elapsed_ms")
            break
    set_if_present(
        "first_present_ms",
        _first_number(first_present_elapsed, milestone_first_present, report.get("first_present_ms")),
    )

    first_visible_elapsed: Any = None
    fvf = report.get("first_visible_frame")
    if isinstance(fvf, dict):
        first_visible_elapsed = fvf.get("elapsed_ms")
    milestone_first_visible: Any = None
    for m in report.get("milestones") or []:
        if isinstance(m, dict) and m.get("milestone") == "FIRST_VISIBLE_FRAME":
            milestone_first_visible = m.get("elapsed_ms")
            break
    set_if_present(
        "first_visible_frame_ms",
        _first_number(
            first_visible_elapsed,
            milestone_first_visible,
            report.get("first_visible_frame_ms"),
            report.get("firstVisibleFrameMs"),
            summary.get("first_visible_frame_ms"),
            summary.get("firstVisibleFrameMs"),
            counters.get("first_visible_frame_ms"),
            counters.get("firstVisibleFrameMs"),
        ),
    )

    derived_black_loading_ms: int | float | None = None
    if "first_visible_frame_ms" in metrics and "first_present_ms" in metrics:
        fvf_ms = metrics["first_visible_frame_ms"]
        fp_ms = metrics["first_present_ms"]
        if float(fvf_ms) >= float(fp_ms):
            if isinstance(fvf_ms, int) and isinstance(fp_ms, int):
                derived_black_loading_ms = fvf_ms - fp_ms
            else:
                derived_black_loading_ms = round(float(fvf_ms) - float(fp_ms), 2)
    set_if_present(
        "black_loading_ms",
        _first_number(
            summary.get("blackLoadingMs"),
            summary.get("black_loading_ms"),
            counters.get("black_loading_ms"),
            counters.get("blackLoadingMs"),
            report.get("black_loading_ms"),
            report.get("blackLoadingMs"),
            derived_black_loading_ms,
        ),
    )

    total_presents = _first_number(counters.get("presents"))
    if total_presents is None and sample_list:
        total_presents = len(sample_list)
    set_if_present("presents", total_presents)

    walls: list[float] = []
    for sample in sample_list:
        if isinstance(sample, dict):
            w = _finite_number(sample.get("wall_ms"))
            if w is not None:
                walls.append(float(w))

    present_fps: float | None = None
    frame_time_ms: float | None = None
    frame_time_median_ms: float | None = None

    if len(walls) >= 2 and walls[-1] > walls[0]:
        span_ms = walls[-1] - walls[0]
        sample_count = len(walls)
        is_subsampled = total_presents is not None and float(total_presents) > float(sample_count)
        intervals = (float(total_presents) - 1.0) if is_subsampled else float(sample_count - 1)
        if intervals > 0.0:
            present_fps = round(intervals * 1000.0 / span_ms, 2)
            frame_time_ms = round(span_ms / intervals, 2)
        if not is_subsampled:
            dts = [walls[i] - walls[i - 1] for i in range(1, sample_count) if walls[i] >= walls[i - 1]]
            if dts:
                frame_time_median_ms = round(statistics.median(dts), 2)
    elif (
        total_presents is not None
        and float(total_presents) >= 2.0
        and "first_present_ms" in metrics
    ):
        elapsed_ms = _finite_number(report.get("elapsed_ms"))
        if elapsed_ms is not None and float(elapsed_ms) > float(metrics["first_present_ms"]):
            span_ms = float(elapsed_ms) - float(metrics["first_present_ms"])
            intervals = float(total_presents) - 1.0
            present_fps = round(intervals * 1000.0 / span_ms, 2)
            frame_time_ms = round(span_ms / intervals, 2)

    set_if_present(
        "present_fps",
        _first_number(
            present_fps,
            counters.get("present_fps"),
            summary.get("presentFps"),
            summary.get("present_fps"),
        ),
    )
    set_if_present(
        "frame_time_ms",
        _first_number(
            frame_time_ms,
            counters.get("frame_time_ms"),
            summary.get("frameTimeMs"),
            summary.get("frame_time_ms"),
        ),
    )
    set_if_present(
        "frame_time_median_ms",
        _first_number(
            frame_time_median_ms,
            counters.get("frame_time_median_ms"),
            summary.get("frameTimeMedianMs"),
            summary.get("frame_time_median_ms"),
        ),
    )
    return metrics


def compute_gpu_delta(
    ra: dict[str, Any],
    rb: dict[str, Any],
    pa: list[dict[str, Any]] | None = None,
    pb: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, int | float | None]]:
    """Build a compact, strictly-sanitized delta map for GPU/draw/frame-pacing fields in both reports."""
    ma = extract_gpu_metrics(ra, pa)
    mb = extract_gpu_metrics(rb, pb)
    out: dict[str, dict[str, int | float | None]] = {}
    for key in GPU_DELTA_METRICS:
        if key not in ma or key not in mb:
            continue
        va, vb = ma[key], mb[key]
        if isinstance(va, int) and isinstance(vb, int):
            delta: int | float = vb - va
        else:
            va = round(float(va), 2)
            vb = round(float(vb), 2)
            delta = round(vb - va, 2)
        if va != 0:
            pct: float | None = round(((vb - va) / va) * 100.0, 2)
        elif vb == 0:
            pct = 0.0
        else:
            pct = None
        out[key] = {"a": va, "b": vb, "delta": delta, "pct": pct}
    return out


TELEMETRY_DELTA_METRICS: tuple[str, ...] = (
    "first_visible_frame_s",
    "black_loading_s",
    "js_heap_used_mb",
    "js_heap_total_mb",
    "audio_ring_attached",
    "audio_frames_drained",
    "present_fps_avg",
)

_TELEMETRY_METRIC_SPECS: dict[str, dict[str, str]] = {
    "first_visible_frame_s": {"kind": "float", "unit": "s", "better": "lower"},
    "black_loading_s": {"kind": "float", "unit": "s", "better": "lower"},
    "js_heap_used_mb": {"kind": "float", "unit": " MB", "better": "lower"},
    "js_heap_total_mb": {"kind": "float", "unit": " MB", "better": "lower"},
    "audio_ring_attached": {"kind": "bool", "unit": "", "better": "true"},
    "audio_frames_drained": {"kind": "int", "unit": "", "better": "higher"},
    "present_fps_avg": {"kind": "float", "unit": " fps", "better": "higher"},
}

_BYTES_PER_MIB = 1024.0 * 1024.0


def _format_decimal(value: float, *, signed: bool = False) -> str:
    """Format a float with 1 decimal place when exact to 0.1 (e.g. -38.2, -950.0), else 2 decimals."""
    rounded = round(float(value), 2)
    if rounded == 0.0:
        rounded = 0.0
    if abs(rounded * 10.0 - round(rounded * 10.0)) < 1e-6:
        fmt = "+.1f" if signed and rounded != 0.0 else ".1f"
    else:
        fmt = "+.2f" if signed and rounded != 0.0 else ".2f"
    return format(rounded, fmt)


def _format_telemetry_value(metric: str, value: Any) -> str:
    if value is None:
        return "<missing>"
    spec = _TELEMETRY_METRIC_SPECS[metric]
    kind = spec["kind"]
    unit = spec["unit"]
    if kind == "bool":
        return "true" if value else "false"
    if kind == "int":
        if isinstance(value, int) and not isinstance(value, bool):
            return f"{value}{unit}"
        return f"{_format_decimal(float(value))}{unit}"
    return f"{_format_decimal(float(value))}{unit}"


def _format_telemetry_signed_delta(metric: str, delta: Any) -> str | None:
    if delta is None:
        return None
    spec = _TELEMETRY_METRIC_SPECS[metric]
    kind = spec["kind"]
    unit = spec["unit"]
    if kind == "bool":
        return str(delta)
    if kind == "int":
        if isinstance(delta, int) and not isinstance(delta, bool):
            if delta == 0:
                return f"0{unit}"
            return f"{delta:+d}{unit}"
        return f"{_format_decimal(float(delta), signed=True)}{unit}"
    return f"{_format_decimal(float(delta), signed=True)}{unit}"


def extract_present_fps_avg(report: dict[str, Any]) -> float | None:
    """Extract sanitized non-negative T73 `present_fps_avg` from a boot report, or None."""
    telemetry = _dict_or_empty(report.get("telemetry"))
    summary = _dict_or_empty(report.get("summary"))
    counters = _dict_or_empty(report.get("counters"))
    val = _first_non_negative_number(
        report.get("present_fps_avg"),
        report.get("presentFpsAvg"),
        telemetry.get("present_fps_avg"),
        telemetry.get("presentFpsAvg"),
        summary.get("present_fps_avg"),
        summary.get("presentFpsAvg"),
        counters.get("present_fps_avg"),
        counters.get("presentFpsAvg"),
    )
    if val is None:
        return None
    return round(float(val), 2)


def extract_telemetry_metrics(report: dict[str, Any]) -> dict[str, int | float | bool]:
    """Extract sanitized Chrome boot telemetry (visible frame, black loading, JS heap, audio ring, FPS)."""
    telemetry = _dict_or_empty(report.get("telemetry"))
    summary = _dict_or_empty(report.get("summary"))
    counters = _dict_or_empty(report.get("counters"))
    memory = _dict_or_empty(report.get("memory"))
    telemetry_memory = _dict_or_empty(telemetry.get("memory"))
    summary_memory = _dict_or_empty(summary.get("memory"))
    audio = _dict_or_empty(report.get("audio"))
    audio_ring = _dict_or_empty(audio.get("ring")) or _dict_or_empty(audio.get("pcm_ring"))
    top_audio_ring = _dict_or_empty(report.get("audio_ring"))
    telemetry_audio = _dict_or_empty(telemetry.get("audio"))
    telemetry_audio_ring = (
        _dict_or_empty(telemetry_audio.get("ring"))
        or _dict_or_empty(telemetry_audio.get("pcm_ring"))
        or _dict_or_empty(telemetry.get("audio_ring"))
    )
    summary_audio = _dict_or_empty(summary.get("audio"))
    summary_audio_ring = _dict_or_empty(summary.get("audio_ring"))

    metrics: dict[str, int | float | bool] = {}

    # 1. first_visible_frame_s (direct seconds, else converted from ms)
    fvf_obj = _dict_or_empty(report.get("first_visible_frame"))
    fvf_s = _first_non_negative_number(
        report.get("first_visible_frame_s"),
        report.get("firstVisibleFrameS"),
        telemetry.get("first_visible_frame_s"),
        telemetry.get("firstVisibleFrameS"),
        summary.get("first_visible_frame_s"),
        summary.get("firstVisibleFrameS"),
        counters.get("first_visible_frame_s"),
        counters.get("firstVisibleFrameS"),
        fvf_obj.get("elapsed_s"),
    )
    if fvf_s is None:
        milestone_fvf_ms: Any = None
        for m in report.get("milestones") or []:
            if isinstance(m, dict) and m.get("milestone") == "FIRST_VISIBLE_FRAME":
                milestone_fvf_ms = m.get("elapsed_ms")
                break
        fvf_ms = _first_non_negative_number(
            fvf_obj.get("elapsed_ms"),
            milestone_fvf_ms,
            report.get("first_visible_frame_ms"),
            report.get("firstVisibleFrameMs"),
            telemetry.get("first_visible_frame_ms"),
            telemetry.get("firstVisibleFrameMs"),
            summary.get("first_visible_frame_ms"),
            summary.get("firstVisibleFrameMs"),
            counters.get("first_visible_frame_ms"),
            counters.get("firstVisibleFrameMs"),
        )
        if fvf_ms is not None:
            fvf_s = round(float(fvf_ms) / 1000.0, 2)
    if fvf_s is not None:
        metrics["first_visible_frame_s"] = round(float(fvf_s), 2)

    # 2. black_loading_s (direct seconds, else converted from ms, else derived from visible - present)
    black_s = _first_non_negative_number(
        report.get("black_loading_s"),
        report.get("blackLoadingS"),
        telemetry.get("black_loading_s"),
        telemetry.get("blackLoadingS"),
        summary.get("black_loading_s"),
        summary.get("blackLoadingS"),
        counters.get("black_loading_s"),
        counters.get("blackLoadingS"),
    )
    if black_s is None:
        black_ms = _first_non_negative_number(
            report.get("black_loading_ms"),
            report.get("blackLoadingMs"),
            telemetry.get("black_loading_ms"),
            telemetry.get("blackLoadingMs"),
            summary.get("black_loading_ms"),
            summary.get("blackLoadingMs"),
            counters.get("black_loading_ms"),
            counters.get("blackLoadingMs"),
        )
        if black_ms is not None:
            black_s = round(float(black_ms) / 1000.0, 2)
    if black_s is None and "first_visible_frame_s" in metrics:
        fp_obj = _dict_or_empty(report.get("first_present"))
        fp_s = _first_non_negative_number(
            report.get("first_present_s"),
            telemetry.get("first_present_s"),
            summary.get("first_present_s"),
            fp_obj.get("elapsed_s"),
        )
        if fp_s is None:
            milestone_fp_ms: Any = None
            for m in report.get("milestones") or []:
                if isinstance(m, dict) and m.get("milestone") == "FIRST_PRESENT":
                    milestone_fp_ms = m.get("elapsed_ms")
                    break
            fp_ms = _first_non_negative_number(
                fp_obj.get("elapsed_ms"),
                milestone_fp_ms,
                report.get("first_present_ms"),
                telemetry.get("first_present_ms"),
                summary.get("first_present_ms"),
            )
            if fp_ms is not None:
                fp_s = float(fp_ms) / 1000.0
        if fp_s is not None and float(metrics["first_visible_frame_s"]) >= float(fp_s):
            black_s = round(float(metrics["first_visible_frame_s"]) - float(fp_s), 2)
    if black_s is not None:
        metrics["black_loading_s"] = round(float(black_s), 2)

    # 3. js_heap_used_mb (direct MB, else converted from bytes)
    heap_used_mb = _first_non_negative_number(
        report.get("js_heap_used_mb"),
        report.get("jsHeapUsedMb"),
        report.get("jsHeapUsedMB"),
        telemetry.get("js_heap_used_mb"),
        telemetry.get("jsHeapUsedMb"),
        telemetry.get("jsHeapUsedMB"),
        memory.get("js_heap_used_mb"),
        memory.get("jsHeapUsedMb"),
        memory.get("jsHeapUsedMB"),
        telemetry_memory.get("js_heap_used_mb"),
        telemetry_memory.get("jsHeapUsedMb"),
        telemetry_memory.get("jsHeapUsedMB"),
        summary.get("js_heap_used_mb"),
        summary.get("jsHeapUsedMb"),
        summary.get("jsHeapUsedMB"),
        summary_memory.get("js_heap_used_mb"),
        summary_memory.get("jsHeapUsedMb"),
        summary_memory.get("jsHeapUsedMB"),
        counters.get("js_heap_used_mb"),
        counters.get("jsHeapUsedMb"),
    )
    if heap_used_mb is None:
        heap_used_bytes = _first_non_negative_number(
            report.get("js_heap_used_bytes"),
            report.get("usedJSHeapSize"),
            telemetry.get("js_heap_used_bytes"),
            telemetry.get("usedJSHeapSize"),
            memory.get("js_heap_used_bytes"),
            memory.get("usedJSHeapSize"),
            telemetry_memory.get("js_heap_used_bytes"),
            telemetry_memory.get("usedJSHeapSize"),
            summary.get("js_heap_used_bytes"),
            summary.get("usedJSHeapSize"),
            summary_memory.get("js_heap_used_bytes"),
            summary_memory.get("usedJSHeapSize"),
        )
        if heap_used_bytes is not None:
            heap_used_mb = round(float(heap_used_bytes) / _BYTES_PER_MIB, 2)
    if heap_used_mb is not None:
        metrics["js_heap_used_mb"] = round(float(heap_used_mb), 2)

    # 4. js_heap_total_mb (direct MB, else converted from bytes)
    heap_total_mb = _first_non_negative_number(
        report.get("js_heap_total_mb"),
        report.get("jsHeapTotalMb"),
        report.get("jsHeapTotalMB"),
        telemetry.get("js_heap_total_mb"),
        telemetry.get("jsHeapTotalMb"),
        telemetry.get("jsHeapTotalMB"),
        memory.get("js_heap_total_mb"),
        memory.get("jsHeapTotalMb"),
        memory.get("jsHeapTotalMB"),
        telemetry_memory.get("js_heap_total_mb"),
        telemetry_memory.get("jsHeapTotalMb"),
        telemetry_memory.get("jsHeapTotalMB"),
        summary.get("js_heap_total_mb"),
        summary.get("jsHeapTotalMb"),
        summary.get("jsHeapTotalMB"),
        summary_memory.get("js_heap_total_mb"),
        summary_memory.get("jsHeapTotalMb"),
        summary_memory.get("jsHeapTotalMB"),
        counters.get("js_heap_total_mb"),
        counters.get("jsHeapTotalMb"),
    )
    if heap_total_mb is None:
        heap_total_bytes = _first_non_negative_number(
            report.get("js_heap_total_bytes"),
            report.get("totalJSHeapSize"),
            telemetry.get("js_heap_total_bytes"),
            telemetry.get("totalJSHeapSize"),
            memory.get("js_heap_total_bytes"),
            memory.get("totalJSHeapSize"),
            telemetry_memory.get("js_heap_total_bytes"),
            telemetry_memory.get("totalJSHeapSize"),
            summary.get("js_heap_total_bytes"),
            summary.get("totalJSHeapSize"),
            summary_memory.get("js_heap_total_bytes"),
            summary_memory.get("totalJSHeapSize"),
        )
        if heap_total_bytes is not None:
            heap_total_mb = round(float(heap_total_bytes) / _BYTES_PER_MIB, 2)
    if heap_total_mb is not None:
        metrics["js_heap_total_mb"] = round(float(heap_total_mb), 2)

    # 5. audio_ring_attached (strict bool)
    audio_attached = _first_bool(
        report.get("audio_ring_attached"),
        report.get("audioRingAttached"),
        telemetry.get("audio_ring_attached"),
        telemetry.get("audioRingAttached"),
        audio.get("audio_ring_attached"),
        audio.get("audioRingAttached"),
        audio.get("ring_attached"),
        audio.get("ringAttached"),
        audio.get("attached"),
        audio_ring.get("attached"),
        top_audio_ring.get("attached"),
        telemetry_audio.get("audio_ring_attached"),
        telemetry_audio.get("audioRingAttached"),
        telemetry_audio.get("ring_attached"),
        telemetry_audio.get("ringAttached"),
        telemetry_audio.get("attached"),
        telemetry_audio_ring.get("attached"),
        summary.get("audio_ring_attached"),
        summary.get("audioRingAttached"),
        summary_audio.get("audio_ring_attached"),
        summary_audio.get("audioRingAttached"),
        summary_audio.get("ring_attached"),
        summary_audio.get("ringAttached"),
        summary_audio.get("attached"),
        summary_audio_ring.get("attached"),
        counters.get("audio_ring_attached"),
    )
    if audio_attached is not None:
        metrics["audio_ring_attached"] = audio_attached

    # 6. audio_frames_drained (non-negative frame count)
    frames_drained = _first_non_negative_number(
        report.get("audio_frames_drained"),
        report.get("audioFramesDrained"),
        telemetry.get("audio_frames_drained"),
        telemetry.get("audioFramesDrained"),
        audio.get("audio_frames_drained"),
        audio.get("audioFramesDrained"),
        audio.get("frames_drained"),
        audio.get("framesDrained"),
        audio_ring.get("frames_drained"),
        audio_ring.get("framesDrained"),
        top_audio_ring.get("frames_drained"),
        top_audio_ring.get("framesDrained"),
        telemetry_audio.get("audio_frames_drained"),
        telemetry_audio.get("audioFramesDrained"),
        telemetry_audio.get("frames_drained"),
        telemetry_audio.get("framesDrained"),
        telemetry_audio_ring.get("frames_drained"),
        telemetry_audio_ring.get("framesDrained"),
        summary.get("audio_frames_drained"),
        summary.get("audioFramesDrained"),
        summary_audio.get("audio_frames_drained"),
        summary_audio.get("audioFramesDrained"),
        summary_audio.get("frames_drained"),
        summary_audio.get("framesDrained"),
        summary_audio_ring.get("frames_drained"),
        summary_audio_ring.get("framesDrained"),
        counters.get("audio_frames_drained"),
    )
    if frames_drained is not None:
        if isinstance(frames_drained, int):
            metrics["audio_frames_drained"] = frames_drained
        elif float(frames_drained).is_integer():
            metrics["audio_frames_drained"] = int(frames_drained)
        else:
            metrics["audio_frames_drained"] = round(float(frames_drained), 2)

    # 7. present_fps_avg (T73 average presents/sec; omitted when absent on older reports)
    present_fps_avg = extract_present_fps_avg(report)
    if present_fps_avg is not None:
        metrics["present_fps_avg"] = present_fps_avg

    return metrics


def compute_telemetry_delta(
    ra: dict[str, Any],
    rb: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build a strictly-sanitized delta map for Chrome boot telemetry present in either report."""
    ma = extract_telemetry_metrics(ra)
    mb = extract_telemetry_metrics(rb)
    out: dict[str, dict[str, Any]] = {}
    for key in TELEMETRY_DELTA_METRICS:
        in_a = key in ma
        in_b = key in mb
        if not in_a and not in_b:
            continue
        spec = _TELEMETRY_METRIC_SPECS[key]
        kind = spec["kind"]
        better = spec["better"]
        va: Any = ma.get(key)
        vb: Any = mb.get(key)

        if kind == "bool":
            if in_a and in_b:
                changed = bool(va) != bool(vb)
                transition = f"{'true' if va else 'false'} -> {'true' if vb else 'false'}"
                if not changed:
                    status = "unchanged"
                elif vb:
                    status = "improved"
                else:
                    status = "regressed"
                out[key] = {
                    "a": bool(va),
                    "b": bool(vb),
                    "delta": transition,
                    "changed": changed,
                    "transition": transition,
                    "better": better,
                    "status": status,
                }
            else:
                out[key] = {
                    "a": bool(va) if in_a else None,
                    "b": bool(vb) if in_b else None,
                    "delta": None,
                    "changed": None,
                    "transition": None,
                    "better": better,
                    "status": "added" if not in_a else "removed",
                }
            continue

        if in_a and in_b:
            if isinstance(va, int) and isinstance(vb, int):
                delta: int | float = vb - va
            else:
                va = round(float(va), 2)
                vb = round(float(vb), 2)
                delta = round(vb - va, 2)
                if delta == 0.0:
                    delta = 0.0
            if va != 0:
                pct: float | None = round(((vb - va) / va) * 100.0, 2)
            elif vb == 0:
                pct = 0.0
            else:
                pct = None
            if delta == 0:
                status = "unchanged"
            elif better == "lower":
                status = "improved" if delta < 0 else "regressed"
            else:
                status = "improved" if delta > 0 else "regressed"
            out[key] = {
                "a": va,
                "b": vb,
                "delta": delta,
                "pct": pct,
                "formatted_delta": _format_telemetry_signed_delta(key, delta),
                "better": better,
                "status": status,
            }
        else:
            out[key] = {
                "a": va if in_a else None,
                "b": vb if in_b else None,
                "delta": None,
                "pct": None,
                "formatted_delta": None,
                "better": better,
                "status": "added" if not in_a else "removed",
            }
    return out


def compare_present_fps_ratio(c: Comparison, ra: dict[str, Any], rb: dict[str, Any]) -> None:
    """Optional T74 regression check: fail if candidate present_fps_avg < baseline * min_fps_ratio."""
    if c.min_fps_ratio is None:
        return
    fps_a = extract_present_fps_avg(ra)
    fps_b = extract_present_fps_avg(rb)
    if fps_a is None or fps_b is None:
        if fps_a is None and fps_b is None:
            missing = "baseline and candidate"
        elif fps_a is None:
            missing = "baseline"
        else:
            missing = "candidate"
        c.warnings.append(
            f"min_fps_ratio check skipped: present_fps_avg missing from {missing}"
        )
        return
    threshold = fps_a * c.min_fps_ratio
    ok = (fps_b >= threshold) or math.isclose(fps_b, threshold, rel_tol=1e-9, abs_tol=1e-12)
    c.record("telemetry.present_fps_avg", fps_a, fps_b, ok, "min_fps_ratio")


def compare(
    ra: dict[str, Any],
    rb: dict[str, Any],
    tol: Tolerances,
    ignore: Sequence[str] = (),
    captures: tuple[Capture, Capture] | None = None,
    import_traces: tuple[list, list] | None = None,
    min_fps_ratio: float | None = None,
) -> Comparison:
    if min_fps_ratio is not None:
        if (
            isinstance(min_fps_ratio, bool)
            or not isinstance(min_fps_ratio, (int, float))
            or not math.isfinite(float(min_fps_ratio))
            or float(min_fps_ratio) <= 0.0
        ):
            raise ValueError("min_fps_ratio must be a positive finite number")
        min_fps_ratio = float(min_fps_ratio)
    c = Comparison(tol, list(ignore), min_fps_ratio=min_fps_ratio)
    compare_milestones(c, ra, rb)
    compare_stop(c, ra, rb)
    compare_imports(c, ra, rb)
    compare_address_space(c, ra, rb)
    compare_progress(c, ra, rb)
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
    compare_present_fps_ratio(c, ra, rb)
    c.gpu_adapter, adapter_warning = compare_gpu_adapter(ra, rb)
    if adapter_warning is not None:
        c.warnings.append(adapter_warning)
    c.gpu_delta = compute_gpu_delta(ra, rb, pa, pb)
    c.telemetry_delta = compute_telemetry_delta(ra, rb)
    return c


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def to_result(c: Comparison) -> dict[str, Any]:
    bad = c.failures()
    tolerances: dict[str, float] = {
        "time_rel": c.tol.time_rel,
        "time_floor_ms": c.tol.time_floor_ms,
        "count_rel": c.tol.count_rel,
    }
    if c.min_fps_ratio is not None:
        tolerances["min_fps_ratio"] = c.min_fps_ratio
    result: dict[str, Any] = {
        "verdict": "fail" if bad else "pass",
        "checked": len(c.entries),
        "differences": len(bad),
        "tolerances": tolerances,
        "ignore": c.ignore,
        "entries": bad,
    }
    if c.gpu_adapter is not None:
        result["gpu_adapter"] = c.gpu_adapter
    if c.warnings:
        result["warnings"] = list(c.warnings)
    if c.gpu_delta:
        result["gpu_delta"] = c.gpu_delta
    if c.telemetry_delta:
        result["telemetry_delta"] = c.telemetry_delta
    return result


def _format_gpu_delta_line(metric: str, item: dict[str, int | float | None]) -> str:
    unit = _GPU_DELTA_UNITS.get(metric, "")
    a, b, delta, pct = item["a"], item["b"], item["delta"], item["pct"]
    if delta == 0:
        change = f"0{unit}"
    else:
        sign = "+" if (delta is not None and delta > 0) else ""
        change = f"{sign}{delta}{unit}"
        if pct is not None:
            change += f", {pct:+.1f}%"
    return f"  {metric}: {a} -> {b} ({change})"


def _format_telemetry_delta_line(metric: str, item: dict[str, Any]) -> str:
    a_str = _format_telemetry_value(metric, item.get("a"))
    b_str = _format_telemetry_value(metric, item.get("b"))
    status = item.get("status")
    spec = _TELEMETRY_METRIC_SPECS[metric]
    if spec["kind"] == "bool":
        if item.get("delta") is None:
            return f"  {metric}: {a_str} -> {b_str} (n/a, {status})"
        return f"  {metric}: {a_str} -> {b_str} ({status})"
    formatted_delta = item.get("formatted_delta")
    pct = item.get("pct")
    if formatted_delta is None:
        return f"  {metric}: {a_str} -> {b_str} (n/a, {status})"
    parts = [str(formatted_delta)]
    if item.get("delta") != 0 and pct is not None:
        parts.append(f"{pct:+.1f}%")
    if status:
        parts.append(str(status))
    return f"  {metric}: {a_str} -> {b_str} ({', '.join(parts)})"


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
    if c.gpu_adapter is not None:
        lines.append("[gpu_adapter]")
        a_label = c.gpu_adapter["a"] if c.gpu_adapter["a"] is not None else "<unknown>"
        b_label = c.gpu_adapter["b"] if c.gpu_adapter["b"] is not None else "<unknown>"
        lines.append(f"  adapter: {a_label} -> {b_label}")
    if c.warnings:
        lines.append("[warnings]")
        for warning in c.warnings:
            lines.append(f"  WARN {warning}")
    if c.gpu_delta:
        lines.append("[gpu_delta]")
        for metric, item in c.gpu_delta.items():
            lines.append(_format_gpu_delta_line(metric, item))
    if c.telemetry_delta:
        lines.append("[telemetry_delta]")
        for metric, item in c.telemetry_delta.items():
            lines.append(_format_telemetry_delta_line(metric, item))
    lines.append(
        f"verdict: {result['verdict'].upper()} ({result['differences']} difference(s) in {result['checked']} checks)"
    )
    return "\n".join(lines)


def format_markdown(c: Comparison, verbose: bool = False) -> str:
    """Format the comparison result as clean GitHub-flavored Markdown."""
    result = to_result(c)
    lines = [
        f"## Boot Report Comparison — {result['verdict'].upper()} "
        f"({result['differences']} difference(s) in {result['checked']} checks)"
    ]
    shown = c.entries if verbose else c.failures()
    if shown:
        lines.append("")
        lines.append("### Checks" if verbose else "### Differences")
        lines.append("| Field | Status | Baseline | Candidate | Mode |")
        lines.append("|---|---|---|---|---|")
        for e in shown:
            st = "ok" if e["ok"] else "DIFF"
            lines.append(f"| `{e['field']}` | {st} | `{e['a']!r}` | `{e['b']!r}` | {e['mode']} |")
    if c.gpu_adapter is not None:
        lines.append("")
        lines.append("### GPU Adapter")
        a_label = c.gpu_adapter["a"] if c.gpu_adapter["a"] is not None else "<unknown>"
        b_label = c.gpu_adapter["b"] if c.gpu_adapter["b"] is not None else "<unknown>"
        suffix = " (mismatch)" if c.gpu_adapter.get("mismatch") else ""
        lines.append(f"- `{a_label}` -> `{b_label}`{suffix}")
    if c.warnings:
        lines.append("")
        lines.append("### Warnings")
        for warning in c.warnings:
            lines.append(f"- {warning}")
    if c.telemetry_delta:
        lines.append("")
        lines.append("### Telemetry Delta")
        lines.append("| Metric | Baseline | Candidate | Delta | % | Status |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for metric, item in c.telemetry_delta.items():
            a_str = _format_telemetry_value(metric, item.get("a")) if item.get("a") is not None else "—"
            b_str = _format_telemetry_value(metric, item.get("b")) if item.get("b") is not None else "—"
            spec = _TELEMETRY_METRIC_SPECS[metric]
            if spec["kind"] == "bool":
                delta_str = item.get("transition") or "—"
                pct_str = "—"
            else:
                delta_str = item.get("formatted_delta") or "—"
                pct = item.get("pct")
                pct_str = f"{pct:+.1f}%" if (item.get("delta") not in (None, 0) and pct is not None) else "—"
            status_str = str(item.get("status") or "—")
            lines.append(f"| `{metric}` | {a_str} | {b_str} | {delta_str} | {pct_str} | {status_str} |")
    if c.gpu_delta:
        lines.append("")
        lines.append("### GPU Delta")
        lines.append("| Metric | Baseline | Candidate | Delta | % |")
        lines.append("|---|---:|---:|---:|---:|")
        for metric, item in c.gpu_delta.items():
            unit = _GPU_DELTA_UNITS.get(metric, "")
            a, b, delta, pct = item["a"], item["b"], item["delta"], item["pct"]
            if delta == 0:
                delta_str = f"0{unit}"
                pct_str = "0.0%"
            else:
                sign = "+" if (delta is not None and delta > 0) else ""
                delta_str = f"{sign}{delta}{unit}"
                pct_str = f"{pct:+.1f}%" if pct is not None else "—"
            lines.append(f"| `{metric}` | {a} | {b} | {delta_str} | {pct_str} |")
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
    p.add_argument("--min-fps-ratio", type=float, default=None, metavar="RATIO",
                   help="fail if candidate present_fps_avg < baseline * RATIO (e.g. 0.85; disabled by default)")
    p.add_argument("--json", dest="as_json", action="store_true", help="print the result as JSON")
    p.add_argument("--markdown", dest="as_markdown", action="store_true", help="print the result as Markdown")
    p.add_argument("-v", "--verbose", action="store_true", help="also list the matching fields (text/Markdown mode)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code in (0, None) else 2
    if args.as_json and args.as_markdown:
        print("error: --json and --markdown are mutually exclusive", file=sys.stderr)
        return 2
    if args.import_trace and not args.capture:
        print("error: --import-trace needs --capture", file=sys.stderr)
        return 2
    if min(args.time_tolerance, args.time_floor_ms, args.count_tolerance) < 0:
        print("error: tolerances must not be negative", file=sys.stderr)
        return 2
    if args.min_fps_ratio is not None and (
        not math.isfinite(args.min_fps_ratio) or args.min_fps_ratio <= 0.0
    ):
        print("error: --min-fps-ratio must be a positive finite number", file=sys.stderr)
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
    c = compare(
        ra,
        rb,
        tol,
        args.ignore,
        captures,  # type: ignore[arg-type]
        traces,  # type: ignore[arg-type]
        min_fps_ratio=args.min_fps_ratio,
    )
    if args.as_json:
        print(json.dumps(to_result(c), indent=2))
    elif args.as_markdown:
        print(format_markdown(c, args.verbose))
    else:
        print(format_text(c, args.verbose))
    return 1 if c.failures() else 0


if __name__ == "__main__":
    raise SystemExit(main())
