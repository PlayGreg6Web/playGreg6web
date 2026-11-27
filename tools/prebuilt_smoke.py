#!/usr/bin/env python3
"""Pass/fail decision for the smoke boot in tools/update_prebuilt.ps1 (swap the new prebuilt in, or keep the old one).

    prebuilt_smoke.py <run dir | boot-report.json>   # prints one line; exit 0 = swap, 1 = keep the old prebuilt

Pass: the report exists (a run that ended on wall_time_limit, i.e. ran the full duration, counts as no fault), at least one present happened, and the run either did not fault or ended on a known
hard stop ("Stopped: ..." from tools/stop_message.py, docs/HARD_STOPS.md). A crash ("Crashed: ...") or a run
that never presented fails; the line printed is then the stop line (or the reason) and the run dir.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import boot_summary  # noqa: E402
import stop_message  # noqa: E402

HARD_STOPS = Path(__file__).resolve().parent.parent / "docs" / "HARD_STOPS.md"


def _ran_full_duration(report: dict[str, Any]) -> bool:
    fr = report.get("fault_record")
    kinds = [report.get("status"), report.get("stop_reason"), fr.get("kind") if isinstance(fr, dict) else None]
    return any(k == "wall_time_limit" for k in kinds)


def decide(report: dict[str, Any] | None, hard_stops_md: str | None, run_dir: str | Path = "") -> tuple[bool, str]:
    """(swap?, one-line reason)."""
    if not isinstance(report, dict):
        return False, f"smoke boot left no boot-report.json - report: {run_dir}"
    presents = int(boot_summary._total_presents(report))
    if _ran_full_duration(report):  # the boot ran until its --max-ms: that is a pass when it presented
        if presents <= 0:
            return False, f"smoke boot reached 0 presents - report: {run_dir}"
        return True, f"smoke boot ok: {presents} presents, ran the full duration (wall_time_limit)"
    stop = stop_message.select(report, hard_stops_md, run_dir)
    if stop is not None and not stop.startswith("Stopped:"):
        return False, stop
    if presents <= 0:
        return False, (stop or f"smoke boot reached 0 presents") + f" - report: {run_dir}"
    return True, f"smoke boot ok: {presents} presents" + (f", {stop}" if stop else ", no fault")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    target = Path(args[0])
    path = target / "boot-report.json" if target.is_dir() else target
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        report = None
    md = HARD_STOPS.read_text(encoding="utf-8") if HARD_STOPS.is_file() else None
    ok, line = decide(report, md, path.parent)
    stop_message.emit(line)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
