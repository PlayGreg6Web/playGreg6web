#!/usr/bin/env python3
"""Table of past boot runs, sorted by time, to spot regressions between source commits.

Usage:
  tools/run_history.py --root H:/ [--pattern 'pg6*'] [--prebuilt-dir DIR] [--since 2026-11-01|3d]
                       [--format md|csv] [--out FILE] [--tail N] [--group-by commit] [--fail-on-regression]

Scans <root>/<pattern>/runs/*/boot-report.json plus the prebuilt runs under %LOCALAPPDATA%/playgta6/*/runs.
Each report is parsed with tools/boot_summary.py. The source commit comes from the report (commit / git_commit /
source_commit) or the build's prebuilt.json. Read-only: nothing is ever written into a run directory (--out
writes elsewhere). Output holds counts, names and addresses only (no game bytes).

--group-by commit: one row per (preset, commit) with medians, ordered by first run; consecutive commits of the
same preset are compared. --fail-on-regression exits 1 if the newest commit of any preset regressed against the
one before it (fps p50 or max draws/frame -10%, more GPU errors / pack misses / underruns, or a new fault).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fnmatch
import io
import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import boot_summary as bs  # noqa: E402
import guest_symbols as gs  # noqa: E402

COLUMNS = ["time", "run", "commit", "preset", "duration", "ended", "hard_stop", "max_screen", "fps_p50", "fps_p5",
           "gpu_errors", "pack_misses", "underruns", "private_peak_mib"]
GROUP_COLUMNS = ["time", "preset", "commit", "runs", "faults", "max_draws", "fps_p50", "fps_p5", "gpu_errors",
                 "pack_misses", "underruns", "private_peak_mib", "change"]
GPU_ERR_KEYS = ("device_errors", "device_lost", "op_errors", "packet_errors", "present_failures")


def _read_json(p: Path) -> dict[str, Any]:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def find_commit(report: dict[str, Any], build_dir: Path | None) -> str:
    for src in (report, bs._d(report.get("build")), bs._d(report.get("boot_options"))):
        for k in ("source_commit", "git_commit", "commit"):
            v = src.get(k)
            if isinstance(v, str) and v:
                return v[:10] + ("*" if src.get("dirty") is True else "")
    if build_dir is not None:
        v = _read_json(build_dir / "prebuilt.json").get("commit")
        if isinstance(v, str) and v:
            return v[:10]
    return "-"


def _max_draws(report: dict[str, Any]) -> float:
    best = 0.0
    for s in report.get("screen_timeline") or []:
        if isinstance(s, dict):
            best = max(best, bs._n(s.get("draws_per_frame")))
    return best


def _last_milestone(report: dict[str, Any]) -> str:
    ms = [m for m in report.get("milestones") or [] if isinstance(m, dict) and m.get("milestone")]
    return str(ms[-1]["milestone"]) if ms else str(report.get("phase") or "-")


def ending(report: dict[str, Any], hard_stops: str | None = None) -> tuple[str, str]:
    """(short ending text, hard-stop status or '')."""
    rec = report.get("fault_record")
    if isinstance(rec, dict):
        what = str(rec.get("kind"))
        fn = rec.get("function")
        sym = fn.get("symbol") if isinstance(fn, dict) else None
        if isinstance(sym, dict) and sym.get("name"):
            what += f" {sym.get('module')}!{sym.get('name')}"
        slot = gs.find_unimplemented(report)
        st = ""
        if slot:
            what += f" {slot['iface']}[{slot['slot']}]" + (f" {slot['name']}" if slot.get("name") else "")
            st = gs.slot_hard_stop(hard_stops, slot) or "unlisted"
        return "FAULT " + what, st
    status = report.get("status")
    fk = bs._d(report.get("fault")).get("kind")
    return (f"limit ({fk})" if fk else f"{status}" if status else "limit"), ""


def row_from_report(report: dict[str, Any], name: str, when: float, commit: str,
                    hard_stops: str | None = None) -> dict[str, Any]:
    f = bs.fps_percentiles(report)
    p50 = f["p50"] if f else (bs._n(report.get("present_fps_avg")) or None)
    ne = bs._d(report.get("native_executor"))
    end, hs = ending(report, hard_stops)
    mb = bs._d(report.get("memory_budget"))
    peak = mb.get("host_private_bytes")
    if peak is None:
        peak = bs._d(mb.get("guest_arena")).get("arena_peak_committed")
    if isinstance(peak, str):
        try:
            peak = int(peak, 0)
        except ValueError:
            peak = None
    ao = bs._d(report.get("audio_out"))
    draws = _max_draws(report)
    ms = _last_milestone(report)
    return {
        "time": dt.datetime.fromtimestamp(when).strftime("%Y-%m-%d %H:%M"), "_t": when,
        "run": name, "commit": commit, "preset": str(bs._d(report.get("run")).get("preset") or report.get("preset") or "-"),
        "duration": bs._dur(report.get("elapsed_ms")) if report.get("elapsed_ms") is not None else "-",
        "ended": end, "hard_stop": hs,
        "max_screen": f"{ms} / {draws:g} draws/frame" if draws else ms,
        "fps_p50": round(p50, 1) if p50 else None, "fps_p5": round(f["p5"], 1) if f else None,
        "gpu_errors": int(sum(bs._n(ne.get(k)) for k in GPU_ERR_KEYS)) if ne else None,
        "pack_misses": int(bs._n(ne.get("shader_pack_misses"))) if "shader_pack_misses" in ne else None,
        "underruns": int(bs._n(ao.get("underruns"))) if "underruns" in ao else None,
        "private_peak_mib": round(bs._n(peak) / 1048576, 1) if peak else None,
        "_draws": draws, "_fault": end.startswith("FAULT"),
    }


def discover(roots: list[Path], pattern: str, prebuilt_dir: Path | None) -> list[tuple[Path, Path]]:
    """(run dir, build dir) pairs that hold a boot-report.json."""
    builds: list[Path] = []
    for r in roots:
        try:
            for c in sorted(r.iterdir()):
                if c.is_dir() and fnmatch.fnmatch(c.name, pattern) and (c / "runs").is_dir():
                    builds.append(c)
        except OSError:
            pass
    if prebuilt_dir and prebuilt_dir.is_dir():
        builds += [c for c in sorted(prebuilt_dir.iterdir()) if (c / "runs").is_dir()]
    out: list[tuple[Path, Path]] = []
    for b in builds:
        try:
            for rd in sorted((b / "runs").iterdir()):
                if (rd / "boot-report.json").is_file():
                    out.append((rd, b))
        except OSError:
            pass
    return out


def parse_since(s: str | None, now: float | None = None) -> float | None:
    if not s:
        return None
    m = re.fullmatch(r"(\d+)([dh])", s)
    if m:
        return (now if now is not None else dt.datetime.now().timestamp()) - int(m[1]) * (86400 if m[2] == "d" else 3600)
    return dt.datetime.fromisoformat(s).timestamp()


def collect(pairs: list[tuple[Path, Path]], since: float | None = None,
            hard_stops: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for rd, build in pairs:
        p = rd / "boot-report.json"
        try:
            report = bs.load(p)
            when = p.stat().st_mtime
        except (OSError, ValueError):
            continue
        if since is not None and when < since:
            continue
        if not report.get("preset"):  # native reports carry no preset; the runner's command.json does
            report["preset"] = _read_json(rd / "command.json").get("preset")
        rows.append(row_from_report(report, rd.name, when, find_commit(report, build), hard_stops))
    return sorted(rows, key=lambda r: r["_t"])


def group_by_commit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["preset"], r["commit"]), []).append(r)

    def med(rs: list[dict[str, Any]], k: str) -> float | None:
        v = [x[k] for x in rs if x[k] is not None]
        return round(statistics.median(v), 1) if v else None

    out = []
    for (preset, commit), rs in groups.items():
        out.append({"time": rs[0]["time"], "_t": rs[0]["_t"], "preset": preset, "commit": commit, "runs": len(rs),
                    "faults": sum(1 for x in rs if x["_fault"]), "max_draws": max(x["_draws"] for x in rs),
                    "fps_p50": med(rs, "fps_p50"), "fps_p5": med(rs, "fps_p5"),
                    "gpu_errors": med(rs, "gpu_errors"), "pack_misses": med(rs, "pack_misses"),
                    "underruns": med(rs, "underruns"), "private_peak_mib": med(rs, "private_peak_mib"),
                    "change": ""})
    out.sort(key=lambda g: (g["preset"], g["_t"]))
    prev = None
    for g in out:
        if prev is not None and prev["preset"] == g["preset"]:
            g["change"] = compare_groups(prev, g)
        prev = g
    return sorted(out, key=lambda g: g["_t"])


def compare_groups(a: dict[str, Any], b: dict[str, Any]) -> str:
    """'' when unchanged, else 'REGRESSION: ...' and/or 'improved: ...' (a older, b newer)."""
    bad: list[str] = []
    good: list[str] = []

    def rel(k: str, label: str, tol: float = 0.10) -> None:
        x, y = a.get(k), b.get(k)
        if not x or y is None:
            return
        if y < x * (1 - tol):
            bad.append(f"{label} {x:g}->{y:g}")
        elif y > x * (1 + tol):
            good.append(f"{label} {x:g}->{y:g}")

    def cnt(k: str, label: str) -> None:
        x, y = a.get(k), b.get(k)
        if x is None or y is None or x == y:
            return
        (bad if y > x else good).append(f"{label} {x:g}->{y:g}")

    rel("fps_p50", "fps p50")
    rel("max_draws", "max draws/frame")
    for k in ("gpu_errors", "pack_misses", "underruns"):
        cnt(k, k.replace("_", " "))
    if b["faults"] and not a["faults"]:
        bad.append("new fault")
    elif a["faults"] and not b["faults"]:
        good.append("fault gone")
    parts = []
    if bad:
        parts.append("REGRESSION: " + ", ".join(bad))
    if good:
        parts.append("improved: " + ", ".join(good))
    return "; ".join(parts)


def regressions(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Groups that are the newest commit of their preset and regressed."""
    newest: dict[str, dict[str, Any]] = {}
    for g in groups:
        newest[g["preset"]] = g
    return [g for g in newest.values() if g["change"].startswith("REGRESSION")]


def render(rows: list[dict[str, Any]], cols: list[str], fmt: str) -> str:
    def cell(v: Any) -> str:
        return "" if v is None else str(v)
    if fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([cell(r.get(c)) for c in cols])
        return buf.getvalue().rstrip("\n")
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(cell(r.get(c)).replace("|", "/") or "-" for c in cols) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", default=None, help="directory holding build dirs (repeatable)")
    ap.add_argument("--pattern", default="pg6*", help="build dir name glob under each --root (default pg6*)")
    la = os.environ.get("LOCALAPPDATA")
    ap.add_argument("--prebuilt-dir", default=str(Path(la) / "playgta6") if la else None,
                    help="prebuilt runs root (default %%LOCALAPPDATA%%/playgta6)")
    ap.add_argument("--since", help="ISO date/time or Nd / Nh")
    ap.add_argument("--format", choices=("md", "csv"), default="md")
    ap.add_argument("--group-by", choices=("commit",))
    ap.add_argument("--fail-on-regression", action="store_true")
    ap.add_argument("--out", help="write the table here instead of stdout (never inside a run dir)")
    ap.add_argument("--tail", type=int, help="only the last N rows")
    args = ap.parse_args(argv)
    roots = [Path(r) for r in (args.root or [])]
    pairs = discover(roots, args.pattern, Path(args.prebuilt_dir) if args.prebuilt_dir else None)
    hs = bs.HARD_STOPS.read_text(encoding="utf-8") if bs.HARD_STOPS.is_file() else None
    rows = collect(pairs, parse_since(args.since), hs)
    groups = group_by_commit(rows) if (args.group_by or args.fail_on_regression) else []
    shown, cols = (groups, GROUP_COLUMNS) if args.group_by else (rows, COLUMNS)
    if args.tail:
        shown = shown[-args.tail:]
    if not (args.fail_on_regression and not args.group_by):
        text = render(shown, cols, args.format)
        if args.out:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
    if args.fail_on_regression:
        bad = regressions(groups)
        for g in bad:
            print(f"regression: {g['preset']} @ {g['commit']}: {g['change']}", file=sys.stderr)
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
