#!/usr/bin/env python3
"""Short human summary of a playgta6_boot report.

Usage:
  tools/boot_summary.py <boot-report.json | run dir> [--pe <game.exe>]
  tools/boot_summary.py --compare A B        (side-by-side deltas, regressions marked)
  tools/boot_summary.py --compare-last [N] [--root H:/]   (newest run vs the one N runs before)

Covers: run length, how it ended (fault_record with symbols via guest_symbols, the docs/HARD_STOPS.md
status of an unimplemented COM slot, or a run limit), a `stop:` row (slot/API, hard-stop status, time,
present index; a not-yet-seen stop is appended to <run root>/stops_seen.json with a survey refresh command), milestones and the draws/frame timeline, fps
percentiles, GPU errors, shader pack hits/misses, top drop programs, audio, bink, memory budget peak,
pipeline prewarm and native executor stats. Sections whose data the report lacks are skipped.
Output holds counts, addresses and symbol names only (no game bytes).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guest_symbols as gs  # noqa: E402

HARD_STOPS = Path(__file__).resolve().parent.parent / "docs" / "HARD_STOPS.md"


def load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if p.is_dir():
        p = p / "boot-report.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("not a boot report")
    tele = p.with_name("telemetry.json")
    if tele.is_file():  # browser run dir: fill what the report lacks from the telemetry
        try:
            t = json.loads(tele.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            t = None
        if isinstance(t, dict):
            for k, v in t.items():
                data.setdefault(k, v)
            for k, v in _d(t.get("summary")).items():
                data.setdefault(k, v)
    return data


def _d(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _n(v: Any) -> float:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0


def _i(v: Any) -> str:
    return f"{int(_n(v)):,}"


def _bytes(v: Any) -> str:
    x = v
    if isinstance(v, str):
        try:
            x = int(v, 0)
        except ValueError:
            return str(v)
    x = _n(x)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if x < 1024 or unit == "GiB":
            return f"{x:.1f} {unit}" if unit != "B" else f"{int(x)} B"
        x /= 1024
    return str(v)


def _dur(ms: Any) -> str:
    s = _n(ms) / 1000
    return f"{int(s // 60)}m{s % 60:04.1f}s" if s >= 60 else f"{s:.1f}s"


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    k = (len(v) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def present_series(report: dict[str, Any]) -> list[tuple[float, float, float]]:
    """(wall_ms, cumulative presents, cumulative real draws or -1) per sample.

    Native samples are one present each (no present_count); browser samples are sparse (about one per 0.5 s)
    and carry the running present_count, so fps must come from the count delta, not from the sample spacing."""
    samples = report.get("presents") or report.get("present_samples") or []
    out: list[tuple[float, float, float]] = []
    for i, s in enumerate(x for x in samples if isinstance(x, dict) and "wall_ms" in x):
        cnt = s.get("present_count")
        cum = _n(cnt) if isinstance(cnt, (int, float)) and not isinstance(cnt, bool) else float(i + 1)
        d = _d(s.get("counters")).get("draws_real_pack")
        out.append((_n(s.get("wall_ms")), cum, _n(d) if isinstance(d, (int, float)) else -1.0))
    return out


def fps_percentiles(report: dict[str, Any]) -> dict[str, float] | None:
    """fps between consecutive samples (presents delta / wall time delta); p1/p5 are the slow tail."""
    ser = present_series(report)
    fps = [1000.0 * (c2 - c1) / (t2 - t1) for (t1, c1, _), (t2, c2, _) in zip(ser, ser[1:])
           if t2 > t1 and c2 > c1]
    if len(fps) < 2:
        return None
    return {"p1": percentile(fps, 0.01), "p5": percentile(fps, 0.05), "p50": percentile(fps, 0.5),
            "p95": percentile(fps, 0.95), "n": float(len(fps))}


def frame_time_lines(report: dict[str, Any]) -> list[str]:
    """Whole-run frame-time stats per phase (boot report `frame_times`): mean/p50/p99/p99.9/max ms, the 1% and 0.1%
    lows as fps, and vblank intervals missed relative to the title's requested interval."""
    ft = _d(report.get("frame_times"))
    phases = [p for p in (ft.get("phases") or []) if isinstance(p, dict) and _n(p.get("frames")) > 0]
    if not phases:
        return []
    out = ["frame times (whole run, ms):"]
    for p in phases:
        out.append(f"  {str(p.get('name')):<16} {_i(p.get('frames')):>8} frames  mean {_n(p.get('mean_ms')):.1f}  "
                   f"p50 {_n(p.get('p50_ms')):.1f}  p99 {_n(p.get('p99_ms')):.1f}  p99.9 {_n(p.get('p999_ms')):.1f}  "
                   f"max {_n(p.get('max_ms')):.1f}  low1% {_n(p.get('low1_fps')):.0f} fps  "
                   f"low0.1% {_n(p.get('low01_fps')):.0f} fps  missed vblanks {_i(p.get('missed_vblanks'))} "
                   f"({_i(p.get('late_frames'))} late frames)")
    return out


def underrun_lines(report: dict[str, Any]) -> list[str]:
    """audio_out.underrun_events printed next to the frame_times hitches: an underrun within ~500 ms of a
    guest frame hitch (or already classed guest_stalled) is a guest stall, not host starvation."""
    ao = _d(report.get("audio_out"))
    events = [e for e in ao.get("underrun_events") or [] if isinstance(e, dict)]
    if not events:
        return []
    hitches = [h for h in _d(report.get("frame_times")).get("hitches") or [] if isinstance(h, dict)]
    out = [f"audio underruns ({_i(ao.get('underruns_guest_stalled'))} guest_stalled, "
           f"{_i(ao.get('underruns_host_starved'))} host_starved):"]
    for e in events:
        at = _n(e.get("at_ms"))
        near = [h for h in hitches if _n(h.get("at_ms")) - _n(h.get("ms")) - 500 <= at <= _n(h.get("at_ms")) + 500]
        tag = ""
        if near:
            h = max(near, key=lambda x: _n(x.get("ms")))
            tag = f"  <- frame hitch {_n(h.get('ms')):.0f} ms ending at {_n(h.get('at_ms')):.0f} ms"
        out.append(f"  at {at:.0f} ms  {_i(e.get('frames_missing'))} frames missing  {e.get('cause')}  "
                   f"{_n(e.get('ms_since_push')):.0f} ms since push, ring {_n(e.get('ring_fill_ms')):.0f} ms, "
                   f"present #{_i(e.get('present_index'))}{tag}")
    for h in hitches:
        if not any(_n(e.get("at_ms")) >= _n(h.get("at_ms")) - _n(h.get("ms")) - 500 and
                   _n(e.get("at_ms")) <= _n(h.get("at_ms")) + 500 for e in events):
            out.append(f"  frame hitch {_n(h.get('ms')):.0f} ms ending at {_n(h.get('at_ms')):.0f} ms: no underrun")
    return out


def _gpu_table_phase_lines(title: str, table: dict[str, Any]) -> list[str]:
    phases = [p for p in (table.get("phases") or []) if isinstance(p, dict)]
    whole = sum(int(_n(e.get("draws")) + _n(e.get("dispatches"))) for e in (table.get("entries") or []) if isinstance(e, dict))
    whole += int(_n(_d(table.get("other")).get("draws")) + _n(_d(table.get("other")).get("dispatches")))
    if not phases or whole == 0:
        return []
    out = [f"{title} (whole run {whole}; per draws/frame phase, attributed to the previous frame's regime):"]
    for p in phases:
        ents = [e for e in (p.get("entries") or []) if isinstance(e, dict)][:3]
        top = ", ".join(f"{e.get('hash')} x{int(_n(e.get('draws')) + _n(e.get('dispatches')))}" for e in ents)
        out.append(f"  {str(p.get('name')):<16} {_i(p.get('draws')):>8} draws {_i(p.get('dispatches')):>6} dispatches"
                   f"  {top}")
    return out


def gpu_phase_lines(report: dict[str, Any]) -> list[str]:
    """Per-phase gpu_drop_table (draws lost) and binding_fallbacks (draws that bound a null/dummy resource:
    unresolved sidecar entry or empty slot; not dropped, the picture may be wrong)."""
    return (_gpu_table_phase_lines("gpu drops", _d(report.get("gpu_drop_table"))) +
            _gpu_table_phase_lines("binding fallbacks", _d(report.get("binding_fallbacks"))))


def frame_time_summary(report: dict[str, Any]) -> dict[str, float] | None:
    """The `all` phase of `frame_times`: {low1_fps, low01_fps, missed_vblanks, late_pct}, None if absent."""
    for p in _d(report.get("frame_times")).get("phases") or []:
        if isinstance(p, dict) and p.get("name") == "all" and _n(p.get("frames")) > 0:
            fr = _n(p.get("frames"))
            return {"low1_fps": _n(p.get("low1_fps")), "low01_fps": _n(p.get("low01_fps")),
                    "missed_vblanks": _n(p.get("missed_vblanks")), "late_pct": 100.0 * _n(p.get("late_frames")) / fr}
    return None


def screen_segments(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Native screen_timeline, or for browser runs segments derived from the presents' draw counters
    (a new segment when draws/frame leaves +-35% of the running segment mean)."""
    segs = [s for s in (report.get("screen_timeline") or []) if isinstance(s, dict)]
    if segs:
        return segs
    ser = present_series(report)
    out: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    for (t1, c1, d1), (t2, c2, d2) in zip(ser, ser[1:]):
        if d1 < 0 or d2 < 0 or c2 <= c1:
            continue
        dpf = (d2 - d1) / (c2 - c1)
        if cur is not None:
            mean = cur["_draws"] / max(cur["presents"], 1)
            if abs(dpf - mean) <= 0.35 * max(mean, 1.0):
                cur["end_time"] = t2
                cur["presents"] += c2 - c1
                cur["_draws"] += d2 - d1
                continue
        cur = {"start_time": t1, "end_time": t2, "presents": c2 - c1, "_draws": d2 - d1}
        out.append(cur)
    for s in out:
        s["draws_per_frame"] = round(s.pop("_draws") / max(s["presents"], 1), 1)
    return out


def fault_info(report: dict[str, Any]) -> dict[str, Any] | None:
    """The fault: a native fault_record, or a browser `fault` block (WASM_TRAP ...); None if the run did not fault."""
    rec = report.get("fault_record")
    if isinstance(rec, dict):
        return rec
    f = _d(report.get("fault"))
    if f and (report.get("status") == "fault" or f.get("trap_kind") or f.get("exception_code")):
        return {"kind": f.get("trap_kind") or f.get("kind"), "elapsed_ms": report.get("elapsed_ms"),
                "rip": f.get("va"), "target": f.get("va"), "function": f.get("function"),
                "code_miss": "INDIRECT_CALL_MISS" in str(f.get("detail") or report.get("message") or "")}
    return None


def _total_presents(report: dict[str, Any]) -> float:
    return (_n(report.get("present_samples_total")) or _n(_d(report.get("present_pacing")).get("presents"))
            or _n(_d(report.get("counters")).get("presents")) or len(report.get("presents") or []))


def drops_top(report: dict[str, Any], n: int = 5) -> list[tuple[str, int]]:
    d = _d(_d(report.get("gpu_work")).get("drops_by_program"))
    items = [(k, int(_n(v))) for k, v in d.items() if _n(v) > 0]
    return sorted(items, key=lambda kv: -kv[1])[:n]


def _ending(report: dict[str, Any], syms: "gs.PeSymbols | None", hard_stops: str | None) -> list[str]:
    out: list[str] = []
    status = report.get("status")
    rec = fault_info(report)
    if rec is not None:
        def named(v: Any) -> str:
            try:
                a = int(v, 16)
            except (TypeError, ValueError):
                return "-"
            d = syms.describe(a) if syms else ""
            return f"0x{a:x}" + (f" ({d})" if d else "")
        fn = rec.get("function")
        loc = ""
        if isinstance(fn, dict) and fn.get("entry"):
            loc = f" in sub_{str(fn['entry'])[2:].upper()}+{fn.get('offset', '0x0')}"
            sym = fn.get("symbol")
            if isinstance(sym, dict):
                loc += f" [{sym.get('module')}!{sym.get('name')}]"
        flag = " (recompiler gap)" if rec.get("code_miss") else ""
        out.append(f"ended: FAULT {rec.get('kind')}{flag}, status {status}, at {_dur(rec.get('elapsed_ms'))}")
        out.append(f"  rip {named(rec.get('rip'))} target {named(rec.get('target'))}{loc}")
        slot = gs.find_unimplemented(report)
        if slot:
            out.append(f"  com slot: {slot['iface']} slot {slot['slot']}"
                       f"{' ' + slot['name'] if slot.get('name') else ''}")
            st = gs.slot_hard_stop(hard_stops, slot)
            out.append(f"  hard stop: {st}" if st else "  hard stop: not listed in docs/HARD_STOPS.md")
    elif "limit" in str(status) or status is None:
        out.append(f"ended: limit ({_d(report.get('fault')).get('kind') or status}), no fault")
    else:
        out.append(f"ended: status {status}, exit_code {report.get('exit_code')}, no fault record")
    if report.get("message"):
        out.append(f"  message: {report['message']}")
    return out


def stop_info(report: dict[str, Any], hard_stops: str | None = None) -> dict[str, Any] | None:
    """Where the run ended, None when it did not fault: `key` (stable id: `iface[slot]` of an unimplemented
    COM slot, else `kind@target`), `name`, `hard_stop` (HARD_STOPS.md status, '' when not a slot, 'unlisted'
    when the slot is not in the file), `elapsed_ms` and `present_index` (presents completed before the stop)."""
    rec = fault_info(report)
    if rec is None:
        return None
    slot = gs.find_unimplemented(report)
    if slot:
        short = str(slot["iface"]).split(".")[-1]
        key = f"{short}[{slot['slot']}]"
        name = slot.get("name") or key
        st = gs.slot_hard_stop(hard_stops, slot) or "unlisted"
    else:
        target = rec.get("target") or rec.get("rip") or "?"
        key = f"{rec.get('kind')}@{target}"
        name, st = key, ""
    return {"key": key, "name": str(name), "hard_stop": st, "elapsed_ms": _n(rec.get("elapsed_ms")
            if rec.get("elapsed_ms") is not None else report.get("elapsed_ms")),
            "present_index": int(_total_presents(report)), "kind": str(rec.get("kind"))}


def stop_line(info: dict[str, Any] | None) -> str:
    if info is None:
        return "stop: none (no fault)"
    return (f"stop: {info['name']}" + (f" [{info['key']}]" if info["name"] != info["key"] else "")
            + (f", hard stop: {info['hard_stop']}" if info["hard_stop"] else "")
            + f", {_dur(info['elapsed_ms'])} into boot, present #{info['present_index']}")


STOPS_SEEN = "stops_seen.json"


def feed_stop(run_dir: Path, info: dict[str, Any] | None, hard_stops: str | None = None) -> str | None:
    """Re-survey feed: when the run ended on a stop not yet in `<run root>/stops_seen.json` (the run dir's
    parent), append it and return the one command that refreshes the stop survey; None otherwise."""
    if info is None or info["hard_stop"] not in ("", "unlisted"):
        return None  # no stop, or a listed hard stop (never ranked by the survey)
    f = run_dir.parent / STOPS_SEEN
    try:
        seen = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else []
    except (OSError, ValueError):
        seen = []
    if not isinstance(seen, list):
        seen = []
    if any(isinstance(e, dict) and e.get("key") == info["key"] for e in seen):
        return None
    seen.append({"key": info["key"], "run": run_dir.name, "elapsed_ms": info["elapsed_ms"],
                 "present_index": info["present_index"], "hard_stop": info["hard_stop"]})
    try:
        f.write_text(json.dumps(seen, indent=1) + "\n", encoding="utf-8")
    except OSError:
        return None
    return (f"new stop {info['key']}: refresh the survey with `python tools/surface_survey.py --pe <exe> "
            f"--recomp target/release/playgta6-recomp --targets <indirect-targets.txt> --stop {info['key']} "
            f"--json-out survey.json` (see {f})")


def _identity(report: dict[str, Any]) -> list[str]:
    """`build:` and `launch:` lines from the report's build / run blocks (absent in older reports)."""
    out: list[str] = []
    b, r = _d(report.get("build")), _d(report.get("run"))
    if b:
        commit = str(b.get("source_commit") or "?")[:10] + ("+dirty" if b.get("dirty") else "")
        bits = [commit, str(b.get("profile") or "?"), "executor" if b.get("executor") else "no executor"]
        if b.get("recomp_hash"):
            bits.append(f"recomp {b['recomp_hash']}")
        out.append("build: " + ", ".join(bits))
    if r:
        out.append(f"launch: preset {r.get('preset')}, name {r.get('name')}")
    return out


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _first_num(*vals: Any) -> float | None:
    for v in vals:
        if _is_num(v):
            return float(v)
    return None


def _gpu_memory_dict(report: dict[str, Any]) -> dict[str, Any]:
    s = _d(report.get("summary"))
    for cand in (report.get("gpu_memory"), s.get("gpu_memory"), report.get("memory_stats"), s.get("memory_stats")):
        if isinstance(cand, dict) and cand:
            return cand
    return {}


def _renderer_memory_dict(report: dict[str, Any]) -> dict[str, Any]:
    s = _d(report.get("summary"))
    for cand in (report.get("renderer_memory"), s.get("renderer_memory")):
        if isinstance(cand, dict) and cand:
            return cand
    return {}


def _pipeline_dump_dict(report: dict[str, Any]) -> dict[str, Any]:
    s = _d(report.get("summary"))
    for cand in (report.get("pipeline_dump"), s.get("pipeline_dump")):
        if isinstance(cand, dict) and cand:
            return cand
    return {}


def _reflection_join_failures(report: dict[str, Any]) -> int | None:
    c = _d(report.get("counters"))
    s = _d(report.get("summary"))
    ne = _d(report.get("native_executor"))
    gw = _d(report.get("gpu_work"))
    v = _first_num(
        report.get("reflection_join_failures"),
        c.get("reflection_join_failures"),
        s.get("reflection_join_failures"),
        s.get("reflectionJoinFailures"),
        ne.get("reflection_join_failures"),
        gw.get("reflection_join_failures"),
    )
    return int(v) if v is not None else None


def _browser_gpu_validation_errors(report: dict[str, Any]) -> int | None:
    c = _d(report.get("counters"))
    s = _d(report.get("summary"))
    v = _first_num(
        c.get("gpu_validation_errors"),
        s.get("validationErrorCount"),
        s.get("gpu_validation_errors"),
        report.get("gpu_validation_errors"),
    )
    return int(v) if v is not None else None


_BUCKET_UPPER_MS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
_BUCKET_KEYS = ("lt0.5", "lt1", "lt2", "lt4", "lt8", "lt16", "lt32", "ge32")


def overshoot_quantile(block: dict[str, Any], q: float) -> float | None:
    """Upper bound (ms) of the overshoot histogram bucket holding quantile q, capped by max_overshoot_ms;
    None when the block has no histogram samples."""
    bk = _d(block.get("overshoot_ms_buckets"))
    counts = [_n(bk.get(k)) for k in _BUCKET_KEYS]
    total = sum(counts)
    if total <= 0:
        return None
    need, acc = q * total, 0.0
    mx = _n(block.get("max_overshoot_ms"))
    for i, c in enumerate(counts):
        acc += c
        if acc >= need:
            return min(_BUCKET_UPPER_MS[i], mx) if i < len(_BUCKET_UPPER_MS) and mx > 0 else (
                _BUCKET_UPPER_MS[i] if i < len(_BUCKET_UPPER_MS) else mx)
    return mx


def timing_blocks(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The overshoot-bearing blocks: guest_sleep and guest_timed_wait.timed_out (the signalled waits have none)."""
    out = {}
    if _d(report.get("guest_sleep")):
        out["guest_sleep"] = _d(report.get("guest_sleep"))
    tw = _d(_d(report.get("guest_timed_wait")).get("timed_out"))
    if tw:
        out["guest_timed_wait"] = tw
    return out


def flush_ms_per_frame(report: dict[str, Any]) -> dict[str, float]:
    """gpu_work.flush_buffer_ns / flush_uniform_ns as ms per present (empty when absent or no presents)."""
    gw = _d(report.get("gpu_work"))
    frames = _total_presents(report)
    if frames <= 0:
        return {}
    return {k: _n(gw.get(f"{k}_ns")) / 1e6 / frames for k in ("flush_buffer", "flush_uniform")
            if isinstance(gw.get(f"{k}_ns"), (int, float))}


def timing_lines(report: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for name, blk in timing_blocks(report).items():
        p50, p99 = overshoot_quantile(blk, 0.5), overshoot_quantile(blk, 0.99)
        out.append(f"{name}: {_i(blk.get('count'))} calls, overshoot p50 "
                   f"{'<=%.1f' % p50 if p50 is not None else '-'} ms, p99 "
                   f"{'<=%.1f' % p99 if p99 is not None else '-'} ms, max {_n(blk.get('max_overshoot_ms')):.1f} ms")
    fl = flush_ms_per_frame(report)
    if fl:
        out.append("gpu flush: " + ", ".join(f"{k} {v:.3f} ms/frame" for k, v in fl.items()))
    return out


def summarize(report: dict[str, Any], syms: "gs.PeSymbols | None" = None, hard_stops: str | None = None) -> str:
    L: list[str] = []
    L += _identity(report)
    c = _d(report.get("counters"))
    L.append(f"run: {_dur(report.get('elapsed_ms'))}, phase {report.get('phase')}, "
             + (f"browser {report.get('preset')}, {_i(c.get('presents'))} presents"
                if report.get("runner") == "browser" else f"{_i(c.get('dispatches'))} dispatches"))
    L += _ending(report, syms, hard_stops)
    L.append(stop_line(stop_info(report, hard_stops)))
    ms = report.get("milestones") or []
    if ms:
        seen: dict[str, Any] = {}
        for m in ms:
            if isinstance(m, dict):
                seen.setdefault(str(m.get("milestone")), m.get("elapsed_ms"))
        L.append("milestones: " + ", ".join(f"{k} {_dur(v)}" for k, v in seen.items()))
    pres = _total_presents(report)
    segs = screen_segments(report)
    L.append(f"screens: {int(pres)} presents, {len(segs)} timeline segments")
    for s in segs[-6:]:
        L.append(f"  {_dur(s.get('start_time'))}-{_dur(s.get('end_time'))}: {_i(s.get('presents'))} presents, "
                 f"{s.get('draws_per_frame')} draws/frame")
    f = fps_percentiles(report)
    if f:
        L.append(f"fps: p1 {f['p1']:.1f}  p5 {f['p5']:.1f}  p50 {f['p50']:.1f}  p95 {f['p95']:.1f} "
                 f"({int(f['n'])} samples)")
    L += frame_time_lines(report)
    L += gpu_phase_lines(report)
    summary_sec = _d(report.get("summary"))
    if report.get("present_fps_avg") is not None:
        fps_p50 = _first_num(report.get("fps_p50_window"), report.get("fps_p50"), summary_sec.get("fps_p50_window"))
        L.append(f"browser fps: avg {_n(report.get('present_fps_avg')):.1f}, steady {_n(report.get('fps_steady_avg')):.1f}, "
                 f"p50 window {_n(fps_p50):.1f}, max window {_n(report.get('fps_max_window')):.1f}, "
                 f">=30 fps in {_n(report.get('windows_ge_30fps_pct')):.0f}% of windows")
        L.append(f"browser: first visible frame {_n(report.get('first_visible_frame_s')):.1f}s, black loading "
                 f"{_n(report.get('black_loading_s')):.1f}s, heap {_n(report.get('js_heap_used_mb')):.0f}/"
                 f"{_n(report.get('js_heap_total_mb')):.0f} MB, adapter {report.get('gpu_adapter')}")
    elif any(_is_num(report.get(k)) or _is_num(summary_sec.get(k)) for k in ("first_visible_frame_s", "black_loading_s")):
        fvf_s = _first_num(report.get("first_visible_frame_s"), summary_sec.get("first_visible_frame_s"))
        bl_s = _first_num(report.get("black_loading_s"), summary_sec.get("black_loading_s"))
        hu = _first_num(report.get("js_heap_used_mb"), summary_sec.get("js_heap_used_mb"))
        ht = _first_num(report.get("js_heap_total_mb"), summary_sec.get("js_heap_total_mb"))
        L.append(f"browser: first visible frame {_n(fvf_s):.1f}s, black loading {_n(bl_s):.1f}s, "
                 f"heap {_n(hu):.0f}/{_n(ht):.0f} MB, adapter {report.get('gpu_adapter') or summary_sec.get('gpu_adapter')}")
    ne = _d(report.get("native_executor"))
    gw = _d(report.get("gpu_work"))
    errs: dict[str, Any] = {
        k: ne.get(k) for k in ("device_errors", "device_lost", "op_errors", "packet_errors", "present_failures")
        if _n(ne.get(k))
    }
    gve = _browser_gpu_validation_errors(report)
    if gve is not None and gve > 0:
        errs["gpu_validation_errors"] = gve
    rjf = _reflection_join_failures(report)
    if rjf is not None and rjf > 0:
        errs["reflection_join_failures"] = rjf
    L.append("gpu errors: " + (", ".join(f"{k}={_i(v)}" for k, v in errs.items()) if errs else "none")
             + (f"; draws no_shader_module={_i(gw.get('draws_no_shader_module'))}"
                if _n(gw.get("draws_no_shader_module")) else ""))
    kinds = [k for k in _d(ne.get("device_error_kinds")).get("kinds") or [] if isinstance(k, dict)]
    for k in kinds[:3]:
        L.append(f"  gpu error kind: {_i(k.get('count'))} x op {k.get('opcode')}: {str(k.get('text') or '')[:140]}")
    if ne:
        L.append(f"shader pack: {_i(ne.get('shader_pack_hits'))} hits, {_i(ne.get('shader_pack_misses'))} misses")
        L.append(f"native executor: {_i(ne.get('frames_submitted'))} frames, {_i(ne.get('draws'))} draws "
                 f"({_i(ne.get('draws_skipped'))} skipped), {_i(ne.get('dispatches'))} dispatches, "
                 f"{_i(ne.get('render_pipelines'))}+{_i(ne.get('compute_pipelines'))} pipelines")
    top = drops_top(report)
    if top:
        L.append("drops by program (top 5): " + ", ".join(f"{k}={_i(v)}" for k, v in top))
    a = _d(report.get("audio"))
    if a and "xapu_decoded_packets" not in a and "frames_drained" in a:
        L.append(f"audio: ring {'attached' if a.get('ring_attached') else 'detached'}, {_i(a.get('frames_drained'))} "
                 f"frames drained, context {a.get('context_state')}, output {a.get('output_state')}, "
                 f"{_i(a.get('sample_rate'))} Hz")
    elif a:
        L.append(f"audio: xapu {_i(a.get('xapu_decoded_packets'))} packets ({_i(a.get('xapu_failed_packets'))} failed), "
                 f"{_i(a.get('xapu_commands'))} cmds, non-silent {_n(a.get('non_silent_fraction')) * 100:.2f}%")
    ap = _d(report.get("audio_probes"))
    if ap:
        called = sum(1 for v in ap.values() if isinstance(v, dict) and _n(v.get("calls")))
        L.append(f"audio probes: {called}/{len(ap)} called")
    af = _d(report.get("audio_files"))
    if af:
        L.append(f"audio files: {_i(af.get('audio_open_ok'))} ok, {_i(af.get('audio_open_failed'))} failed "
                 f"of {_i(af.get('audio_opens'))} opens")
    ao = _d(report.get("audio_out"))
    if ao:
        L.append(f"audio out: device {'open' if ao.get('device_open') else 'closed'}, "
                 f"{_i(ao.get('frames_played'))} frames played, {_i(ao.get('underruns'))} underruns"
                 + (f", error {ao.get('error')}" if ao.get("error") else ""))
        L.extend(underrun_lines(report))
    b = _d(report.get("bink"))
    if b:
        callers = [e for e in b.get("exports") or [] if isinstance(e, dict) and _n(e.get("calls"))]
        callers.sort(key=lambda e: -_n(e.get("calls")))
        L.append(f"bink: {b.get('binding')}, {_i(b.get('total_calls'))} calls"
                 + (" (" + ", ".join(f"{e.get('name')}={_i(e.get('calls'))}" for e in callers[:3]) + ")"
                    if callers else ""))
    if b.get("lookup_null") is not None:
        names = [str(x) for x in (b.get("lookup_null_names") or [])][:3]
        L.append(f"bink lookups: {_i(b.get('lookup_null'))} null" + (f" ({', '.join(names)})" if names else "")
                 + f", {len(b.get('open_log') or [])} opens logged")
    L += timing_lines(report)
    mb = _d(report.get("memory_budget"))
    if mb:
        ga =_d(mb.get("guest_arena"))
        L.append(f"memory: arena peak {_bytes(ga.get('arena_peak_committed'))}, host private "
                 f"{_bytes(mb.get('host_private_bytes'))}, gpu visible {_bytes(mb.get('gpu_visible_bytes'))}")
    gm = _gpu_memory_dict(report)
    rm = _renderer_memory_dict(report)
    if gm or rm:
        parts: list[str] = []
        vram_b = _first_num(gm.get("vramBytes"), gm.get("vram_bytes"))
        if vram_b is not None:
            parts.append(f"vram {_bytes(vram_b)}")
        elif _is_num(gm.get("vram_peak_mb")):
            parts.append(f"vram peak {_n(gm.get('vram_peak_mb')):.1f} MiB")
        live_b = _first_num(gm.get("totalLiveGpuBytes"), gm.get("total_live_gpu_bytes"))
        if live_b is not None:
            parts.append(f"live {_bytes(live_b)}")
        pk_buf = _first_num(gm.get("peakBufferBytes"), gm.get("peak_buffer_bytes"))
        if pk_buf is not None:
            parts.append(f"peak buf {_bytes(pk_buf)}")
        pk_tex = _first_num(gm.get("peakTextureBytes"), gm.get("peak_texture_bytes"))
        if pk_tex is not None:
            parts.append(f"peak tex {_bytes(pk_tex)}")
        stg = _first_num(gm.get("stagingPoolBytes"), gm.get("staging_pool_bytes"))
        if stg is not None:
            parts.append(f"staging {_bytes(stg)}")
        wasm_mem = _first_num(rm.get("wasmLinearMemoryBytes"), rm.get("wasm_linear_memory_bytes"), gm.get("wasm_linear_memory_bytes"))
        if wasm_mem is not None:
            parts.append(f"wasm {_bytes(wasm_mem)}")
        if parts:
            L.append("gpu memory: " + ", ".join(parts))
    pd = _pipeline_dump_dict(report)
    if pd:
        pd_bits: list[str] = []
        for label_k, keys in (
            ("dumped", ("dumped", "count", "total", "pipelines")),
            ("render", ("render_pipelines", "render")),
            ("compute", ("compute_pipelines", "compute")),
            ("failed", ("failed", "errors")),
        ):
            val = _first_num(*(pd.get(k) for k in keys))
            if val is not None:
                pd_bits.append(f"{label_k}={_i(val)}")
        if pd_bits:
            L.append("pipeline dump: " + ", ".join(pd_bits))
    elif _d(report.get("boot_options")).get("dump_pipelines_dir"):
        L.append("pipeline dump: enabled")
    si = _first_num(report.get("scripted_inputs_sent"), summary_sec.get("scripted_inputs_sent"), c.get("scripted_inputs_sent"))
    if si is not None:
        L.append(f"scripted inputs: {_i(si)} sent")
    pw = _d(report.get("pipeline_prewarm"))
    if pw:
        L.append(f"prewarm: {_i(pw.get('prewarmed'))} built, {_i(pw.get('hit_before_use'))} hit before use, "
                 f"{_i(pw.get('wasted'))} wasted, {_i(pw.get('failed'))} failed")
    return "\n".join(L)


def numeric_facts(report: dict[str, Any]) -> dict[str, float]:
    facts: dict[str, float] = {"elapsed_ms": _n(report.get("elapsed_ms")),
                               "dispatches": _n(_d(report.get("counters")).get("dispatches")),
                               "presents": _total_presents(report)}
    f = fps_percentiles(report)
    if f:
        facts.update({f"fps_{k}": v for k, v in f.items() if k != "n"})
    for sec, keys in (("native_executor", ("device_errors", "op_errors", "packet_errors", "shader_pack_hits",
                                            "shader_pack_misses", "draws", "draws_skipped", "frames_submitted")),
                      ("audio", ("xapu_decoded_packets", "xapu_failed_packets", "non_silent_fraction")),
                      ("audio_files", ("audio_open_ok", "audio_open_failed")),
                      ("audio_out", ("frames_played", "underruns")),
                      ("bink", ("total_calls",)),
                      ("pipeline_prewarm", ("prewarmed", "hit_before_use", "wasted", "failed")),
                      ("gpu_work", ("draws_emitted", "draws_no_shader_module", "draws_dropped_by_bridge"))):
        for k in keys:
            v = _d(report.get(sec)).get(k)
            if _is_num(v):
                facts[f"{sec}.{k}"] = float(v)
    summary_sec = _d(report.get("summary"))
    for k in ("present_fps_avg", "fps_steady_avg", "fps_p50_window", "fps_max_window",
              "windows_ge_30fps_pct", "first_visible_frame_s", "black_loading_s",
              "js_heap_used_mb", "js_heap_total_mb", "scripted_inputs_sent"):
        v = _first_num(report.get(k), summary_sec.get(k))
        if v is not None:
            facts[k] = v
    if "fps_p50_window" not in facts and _is_num(report.get("fps_p50")):
        facts["fps_p50_window"] = float(report["fps_p50"])
    gve = _browser_gpu_validation_errors(report)
    if gve is not None:
        facts["counters.gpu_validation_errors"] = float(gve)
    rjf = _reflection_join_failures(report)
    if rjf is not None:
        facts["reflection_join_failures"] = float(rjf)
    gm = _gpu_memory_dict(report)
    for dst, keys in (
        ("gpu_memory.vram_bytes", ("vramBytes", "vram_bytes")),
        ("gpu_memory.total_live_gpu_bytes", ("totalLiveGpuBytes", "total_live_gpu_bytes")),
        ("gpu_memory.peak_buffer_bytes", ("peakBufferBytes", "peak_buffer_bytes")),
        ("gpu_memory.peak_texture_bytes", ("peakTextureBytes", "peak_texture_bytes")),
        ("gpu_memory.staging_pool_bytes", ("stagingPoolBytes", "staging_pool_bytes")),
        ("gpu_memory.vram_peak_mb", ("vram_peak_mb",)),
    ):
        val = _first_num(*(gm.get(k) for k in keys))
        if val is not None:
            facts[dst] = val
    rm = _renderer_memory_dict(report)
    wasm_mem = _first_num(rm.get("wasmLinearMemoryBytes"), rm.get("wasm_linear_memory_bytes"))
    if wasm_mem is not None:
        facts["renderer_memory.wasm_linear_memory_bytes"] = wasm_mem
    pd = _pipeline_dump_dict(report)
    for dst, keys in (
        ("pipeline_dump.dumped", ("dumped", "count", "total", "pipelines")),
        ("pipeline_dump.failed", ("failed", "errors")),
    ):
        val = _first_num(*(pd.get(k) for k in keys))
        if val is not None:
            facts[dst] = val
    peak = _d(_d(report.get("memory_budget")).get("guest_arena")).get("arena_peak_committed")
    if _is_num(peak):
        facts["memory.arena_peak_committed"] = float(peak)
    lookup_null = _d(report.get("bink")).get("lookup_null")
    if isinstance(lookup_null, (int, float)) and not isinstance(lookup_null, bool):
        facts["bink.lookup_null"] = float(lookup_null)
    for name, blk in timing_blocks(report).items():
        for tag, q in (("p50", 0.5), ("p99", 0.99)):
            v = overshoot_quantile(blk, q)
            if v is not None:
                facts[f"{name}.overshoot_{tag}_ms"] = v
        if "max_overshoot_ms" in blk:
            facts[f"{name}.overshoot_max_ms"] = _n(blk.get("max_overshoot_ms"))
    for k, v in flush_ms_per_frame(report).items():
        facts[f"gpu_work.{k}_ms_per_frame"] = v
    drops = _d(_d(report.get("gpu_work")).get("drops_by_program"))
    facts["drops_total"] = float(sum(int(_n(v)) for v in drops.values()))
    return facts


def _fmt(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:,.2f}" if v != int(v) else f"{int(v):,}"


HIGHER_BETTER = ("fps_", "presents", "frames_played", "shader_pack_hits", "xapu_decoded_packets", "hit_before_use",
                 "audio_open_ok", "frames_submitted", "non_silent_fraction", "low1_fps", "low01_fps", "prewarmed")
LOWER_BETTER = ("errors", "failures", "misses", "underruns", "hitches", "drops", "failed", "wasted", "elapsed_ms",
                "late_pct", "missed_vblanks", "binding_fallbacks", "draws_skipped", "no_shader_module", "peak",
                "draws_dropped_by_bridge", "black_loading", "first_visible_frame", "overshoot", "lookup_null",
                "_ms_per_frame")


def _direction(key: str) -> int:
    """+1 when a higher value is better, -1 when lower is better, 0 when it is only informational."""
    if any(t in key for t in LOWER_BETTER):
        return -1
    if any(t in key for t in HIGHER_BETTER):
        return 1
    return 0


def _table_phase_draws(table: dict[str, Any], prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for p in table.get("phases") or []:
        if isinstance(p, dict) and p.get("name") is not None:
            out[f"{prefix}.phase.{p['name']}"] = _n(p.get("draws")) + _n(p.get("dispatches"))
    return out


def compare_facts(report: dict[str, Any]) -> dict[str, float]:
    """numeric_facts plus the extra comparison metrics: hitches, underrun classes, frame-time lows,
    gpu drops / binding fallbacks per phase."""
    f = numeric_facts(report)
    ft = frame_time_summary(report)
    if ft:
        f.update({f"frame_times.{k}": v for k, v in ft.items()})
    if "frame_times" in report:
        hitches = [h for h in _d(report.get("frame_times")).get("hitches") or [] if isinstance(h, dict)]
        f["frame_times.hitches"] = float(len(hitches))
    ao = _d(report.get("audio_out"))
    for k in ("underruns_guest_stalled", "underruns_host_starved"):
        if k in ao:
            f[f"audio_out.{k}"] = _n(ao.get(k))
    f.update(_table_phase_draws(_d(report.get("gpu_drop_table")), "gpu_drops"))
    f.update(_table_phase_draws(_d(report.get("binding_fallbacks")), "binding_fallbacks"))
    return f


def _fallback_top(report: dict[str, Any], n: int = 3) -> dict[str, int]:
    ents = [e for e in _d(report.get("binding_fallbacks")).get("entries") or [] if isinstance(e, dict)]
    items = {str(e.get("hash")): int(_n(e.get("draws")) + _n(e.get("dispatches"))) for e in ents}
    return dict(sorted(items.items(), key=lambda kv: -kv[1])[:n])


def _fault_site(report: dict[str, Any]) -> str:
    rec = fault_info(report)
    if rec is None:
        return "no fault"
    fn = rec.get("function")
    site = f"rip {rec.get('rip')}"
    if isinstance(fn, dict) and fn.get("entry"):
        site = f"sub_{str(fn['entry'])[2:].upper()}+{fn.get('offset', '0x0')}"
    return f"FAULT {rec.get('kind')} at {site}"


def compare(a: dict[str, Any], b: dict[str, Any]) -> str:
    """Side-by-side A -> B diff of the key metrics; the last column marks regression / improvement."""
    fa, fb = compare_facts(a), compare_facts(b)
    L = [f"A {_ending(a, None, None)[0]}", f"B {_ending(b, None, None)[0]}"]
    sa, sb = _fault_site(a), _fault_site(b)
    if sa != sb:
        faulted_a, faulted_b = fault_info(a) is not None, fault_info(b) is not None
        mark = "REGRESSION" if faulted_b and not faulted_a else ("improvement" if faulted_a and not faulted_b
                                                                else "changed")
        L.append(f"end state: {sa}  ->  {sb}  [{mark}]")
    else:
        L.append(f"end state: {sa} (same)")
    L.append(f"{'metric':<40}{'A':>16}{'B':>16}{'delta':>16}  verdict")
    rows = regress = improve = 0
    for k in sorted(set(fa) | set(fb)):
        x, y = fa.get(k), fb.get(k)
        if x == y:
            continue
        d = (y or 0) - (x or 0)
        dr = _direction(k)
        verdict = ""
        if dr and d:
            good = (d > 0) == (dr > 0)
            verdict = "improvement" if good else "REGRESSION"
            improve += good
            regress += not good
        L.append(f"{k:<40}{_fmt(x):>16}{_fmt(y):>16}{('+' if d > 0 else '') + _fmt(d):>16}  {verdict}".rstrip())
        rows += 1
    if not rows:
        L.append("(no differences in the tracked metrics)")
    ta, tb = _fallback_top(a), _fallback_top(b)
    if ta or tb:
        L.append("binding fallbacks top: A " + (", ".join(f"{k} x{v}" for k, v in ta.items()) or "none")
                 + "  |  B " + (", ".join(f"{k} x{v}" for k, v in tb.items()) or "none"))
    L.append(f"{regress} regression(s), {improve} improvement(s)")
    return "\n".join(L)


def last_runs(root: str | Path, n: int) -> list[Path]:
    """The newest run's boot-report.json and the one `n` runs before it, found with run_history discovery
    (<root>/<build dir>/runs/*/boot-report.json, oldest first by mtime)."""
    import run_history as rh
    pairs = rh.discover([Path(root)], "*", None)
    reports = sorted((rd / "boot-report.json" for rd, _ in pairs), key=lambda p: p.stat().st_mtime)
    if len(reports) < n + 1:
        raise ValueError(f"need {n + 1} runs under {root}, found {len(reports)}")
    return [reports[-(n + 1)], reports[-1]]


def render_path(path: str | Path, pe: str | None = None, feed: bool = True) -> str:
    syms = gs.PeSymbols.from_path(pe) if pe else None
    md = HARD_STOPS.read_text(encoding="utf-8") if HARD_STOPS.is_file() else None
    out = summarize(load(path), syms, md)
    p = Path(path)
    if feed and p.is_dir():
        cmd = feed_stop(p, stop_info(load(p), md), md)
        if cmd:
            out += "\n" + cmd
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", nargs="?", help="boot-report.json or a run directory")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="show deltas between two reports/run dirs")
    ap.add_argument("--compare-last", type=int, metavar="N", nargs="?", const=1,
                    help="compare the newest run with the one N runs before it (default 1), found via run_history "
                         "discovery under --root (default H:/)")
    ap.add_argument("--root", default="H:/", help="build-dir root for --compare-last (see tools/run_history.py)")
    ap.add_argument("--no-feed", action="store_true", help="do not record the stop in <run root>/stops_seen.json")
    ap.add_argument("--pe", help="game PE (read-only) to name fault addresses")
    args = ap.parse_args(argv)
    try:
        if args.compare_last is not None:
            runs = last_runs(args.root, args.compare_last)
            print(f"A {runs[0].parent}\nB {runs[-1].parent}")
            print(compare(load(runs[0]), load(runs[-1])))
        elif args.compare:
            print(compare(load(args.compare[0]), load(args.compare[1])))
        elif args.report:
            print(render_path(args.report, args.pe, not args.no_feed))
        else:
            ap.error("give a report or --compare A B")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
