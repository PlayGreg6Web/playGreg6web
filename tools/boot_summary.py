#!/usr/bin/env python3
"""Short human summary of a playgta6_boot report.

Usage:
  tools/boot_summary.py <boot-report.json | run dir> [--pe <game.exe>]
  tools/boot_summary.py --compare A B        (deltas of the numeric facts)

Covers: run length, how it ended (fault_record with symbols via guest_symbols, the docs/HARD_STOPS.md
status of an unimplemented COM slot, or a run limit), milestones and the draws/frame timeline, fps
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


def summarize(report: dict[str, Any], syms: "gs.PeSymbols | None" = None, hard_stops: str | None = None) -> str:
    L: list[str] = []
    L += _identity(report)
    c = _d(report.get("counters"))
    L.append(f"run: {_dur(report.get('elapsed_ms'))}, phase {report.get('phase')}, "
             + (f"browser {report.get('preset')}, {_i(c.get('presents'))} presents"
                if report.get("runner") == "browser" else f"{_i(c.get('dispatches'))} dispatches"))
    L += _ending(report, syms, hard_stops)
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
    if report.get("present_fps_avg") is not None:
        L.append(f"browser fps: avg {_n(report.get('present_fps_avg')):.1f}, steady {_n(report.get('fps_steady_avg')):.1f}, "
                 f"p50 window {_n(report.get('fps_p50_window')):.1f}, max window {_n(report.get('fps_max_window')):.1f}, "
                 f">=30 fps in {_n(report.get('windows_ge_30fps_pct')):.0f}% of windows")
        L.append(f"browser: first visible frame {_n(report.get('first_visible_frame_s')):.1f}s, black loading "
                 f"{_n(report.get('black_loading_s')):.1f}s, heap {_n(report.get('js_heap_used_mb')):.0f}/"
                 f"{_n(report.get('js_heap_total_mb')):.0f} MB, adapter {report.get('gpu_adapter')}")
    ne = _d(report.get("native_executor"))
    gw = _d(report.get("gpu_work"))
    errs = {k: ne.get(k) for k in ("device_errors", "device_lost", "op_errors", "packet_errors", "present_failures")
            if _n(ne.get(k))}
    L.append("gpu errors: " + (", ".join(f"{k}={_i(v)}" for k, v in errs.items()) if errs else "none")
             + (f"; draws no_shader_module={_i(gw.get('draws_no_shader_module'))}"
                if _n(gw.get("draws_no_shader_module")) else ""))
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
    b = _d(report.get("bink"))
    if b:
        callers = [e for e in b.get("exports") or [] if isinstance(e, dict) and _n(e.get("calls"))]
        callers.sort(key=lambda e: -_n(e.get("calls")))
        L.append(f"bink: {b.get('binding')}, {_i(b.get('total_calls'))} calls"
                 + (" (" + ", ".join(f"{e.get('name')}={_i(e.get('calls'))}" for e in callers[:3]) + ")"
                    if callers else ""))
    mb = _d(report.get("memory_budget"))
    if mb:
        ga = _d(mb.get("guest_arena"))
        L.append(f"memory: arena peak {_bytes(ga.get('arena_peak_committed'))}, host private "
                 f"{_bytes(mb.get('host_private_bytes'))}, gpu visible {_bytes(mb.get('gpu_visible_bytes'))}")
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
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                facts[f"{sec}.{k}"] = float(v)
    peak = _d(_d(report.get("memory_budget")).get("guest_arena")).get("arena_peak_committed")
    if isinstance(peak, (int, float)):
        facts["memory.arena_peak_committed"] = float(peak)
    drops = _d(_d(report.get("gpu_work")).get("drops_by_program"))
    facts["drops_total"] = float(sum(int(_n(v)) for v in drops.values()))
    return facts


def _fmt(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:,.2f}" if v != int(v) else f"{int(v):,}"


def compare(a: dict[str, Any], b: dict[str, Any]) -> str:
    fa, fb = numeric_facts(a), numeric_facts(b)
    L = [f"A {_ending(a, None, None)[0]}", f"B {_ending(b, None, None)[0]}",
         f"{'metric':<40}{'A':>16}{'B':>16}{'delta':>16}"]
    rows = 0
    for k in sorted(set(fa) | set(fb)):
        x, y = fa.get(k), fb.get(k)
        if x == y:
            continue
        d = (y or 0) - (x or 0)
        L.append(f"{k:<40}{_fmt(x):>16}{_fmt(y):>16}{('+' if d > 0 else '') + _fmt(d):>16}")
        rows += 1
    if not rows:
        L.append("(no differences in the tracked metrics)")
    return "\n".join(L)


def render_path(path: str | Path, pe: str | None = None) -> str:
    syms = gs.PeSymbols.from_path(pe) if pe else None
    md = HARD_STOPS.read_text(encoding="utf-8") if HARD_STOPS.is_file() else None
    return summarize(load(path), syms, md)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", nargs="?", help="boot-report.json or a run directory")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="show deltas between two reports/run dirs")
    ap.add_argument("--pe", help="game PE (read-only) to name fault addresses")
    args = ap.parse_args(argv)
    try:
        if args.compare:
            print(compare(load(args.compare[0]), load(args.compare[1])))
        elif args.report:
            print(render_path(args.report, args.pe))
        else:
            ap.error("give a report or --compare A B")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
