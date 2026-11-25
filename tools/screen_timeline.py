"""Screen timeline (R40): a compact per-screen view of a boot report's per-Present samples.

Agents scripting menus tell screens apart by draws per frame and new resources. Each Present sample
carries the cumulative `draws_issued` and `resources_created` counters (D3D12 layer); this module
turns them into per-present deltas and collapses consecutive presents with the same draw count
(+-1) and no new resources into one segment:
  {start_time, end_time, first_present, presents, draws_per_frame, new_resources_at_start,
   input_events_in_segment}
Capped at MAX_SEGMENTS (later presents are folded into the last segment). Times are wall_ms.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

MAX_SEGMENTS = 500


def _num(v: Any) -> int:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0


def parse_input_times(text: str) -> list[float]:
    """Event times (ms) of an input script (`<time_ms> <buttons> [hold_ms]`, `#` comments)."""
    out: list[float] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.match(r"(\d+(?:\.\d+)?)\s+\S", line)
        if m:
            out.append(float(m.group(1)))
    return out


def build_timeline(presents: Sequence[dict[str, Any]], input_times: Sequence[float] = (),
                   max_segments: int = MAX_SEGMENTS) -> list[dict[str, Any]]:
    segs: list[dict[str, Any]] = []
    prev_draws = prev_res = 0
    for i, p in enumerate(presents):
        c = p.get("counters") or {}
        draws_total, res_total = _num(c.get("draws_issued")), _num(c.get("resources_created"))
        draws, new_res = max(0, draws_total - prev_draws), max(0, res_total - prev_res)
        prev_draws, prev_res = draws_total, res_total
        t = float(p.get("wall_ms") or 0)
        cur = segs[-1] if segs else None
        if cur is not None and (len(segs) >= max_segments or
                                (new_res == 0 and abs(draws - cur["draws_per_frame"]) <= 1)):
            cur["end_time"] = t
            cur["presents"] += 1
            continue
        segs.append({"start_time": t, "end_time": t, "first_present": i, "presents": 1,
                     "draws_per_frame": draws, "new_resources_at_start": new_res,
                     "input_events_in_segment": 0})
    for k, s in enumerate(segs):
        # A segment spans [its start, the next one's start); the last one is open-ended.
        end = segs[k + 1]["start_time"] if k + 1 < len(segs) else float("inf")
        s["input_events_in_segment"] = sum(1 for x in input_times if s["start_time"] <= x < end)
    return segs


def format_table(segs: Sequence[dict[str, Any]], limit: int = 20) -> list[str]:
    lines = [f"screen timeline: {len(segs)} segment(s)",
             "  start_ms  end_ms  present  n  draws/frame  new_res  inputs"]
    for s in segs[:limit]:
        lines.append(f"  {s['start_time']:8.0f}  {s['end_time']:6.0f}  {s['first_present']:7d}  {s['presents']:d}  "
                     f"{s['draws_per_frame']:11d}  {s['new_resources_at_start']:7d}  {s['input_events_in_segment']:6d}")
    if len(segs) > limit:
        lines.append(f"  ... {len(segs) - limit} more")
    return lines
