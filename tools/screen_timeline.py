"""Screen timeline (R40 + T64 + T81): per-screen timelines for boot reports and frame captures.

R40 (boot report Present samples):
Agents scripting menus tell screens apart by draws per frame and new resources. Each Present sample
carries the cumulative `draws_issued` and `resources_created` counters (D3D12 layer); this module
turns them into per-present deltas and collapses consecutive presents with the same draw count
(+-1) and no new resources into one segment:
  {start_time, end_time, first_present, presents, draws_per_frame, new_resources_at_start,
   input_events_in_segment}
Capped at MAX_SEGMENTS: later presents are folded into the last segment, which then gets
`folded: true`, `draws_min` / `draws_max` over the folded presents, and `draws_per_frame` of the latest
present (not one screen). Times are wall_ms.

T64 + T81 (Chrome capture frame directory CLI):
Analyzes a directory of captured frames (e.g. `frame_0001.png` / `frame_0001.jpg`), groups
consecutive frames into distinct visual/black screen segments, computes optional dwell durations,
leading black-loading duration (`black_loading_s`), first non-black timing (`first_non_black_s`),
frame counts (`black_frame_count`, `non_black_frame_count`, `all_black`), and per-transition
milestone deltas (`delta_s`) via `--interval-sec <f>`, and outputs either a sanitized plain-text
table or structured `--json` summary without leaking host paths or usernames.
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import re
import statistics
import struct
import sys
from typing import Any, Mapping, Sequence
import zlib

MAX_SEGMENTS = 500
DEFAULT_BLACK_LUMA = 2.0
DEFAULT_DIFF_THRESHOLD = 4.0
THUMB_SIZE = (32, 18)
DEFAULT_PHASE_WINDOW_MS = 5000.0
DEFAULT_BINK2_DRAWS_RANGE: tuple[float, float] = (3.0, 5.0)
DEFAULT_MIN_BINK2_PRESENTS = 3
DEFAULT_MIN_LOADING_SPINNER_S = 2.0

_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})
_FRAME_NUM_RE = re.compile(r"(\d+)(?!.*\d)")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_UNC_PATH_RE = re.compile(r"\\\\[^\\\s\"'<>|]+\\[^\\\s\"'<>|]+(?:\\[^\s\"'<>|]+)*")
_WIN_ABS_PATH_RE = re.compile(r"(?i)(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s\"'<>|]*")
_UNIX_HOST_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])/(?:Users|home|tmp|var|private|mnt|root|opt|srv|Volumes)(?:/[^\s\"'<>|]*)?"
)


def _basename_from_path_text(raw: str) -> str:
    parts = [p for p in re.split(r"[\\/]+", raw) if p]
    if not parts:
        return ""
    if len(parts[0]) == 2 and parts[0][1] == ":":
        parts = parts[1:]
    if not parts:
        return ""
    if len(parts) <= 2 and parts[0].lower() in {"users", "home"}:
        return "<redacted>"
    return parts[-1]


def sanitize_path_label(value: str | Path) -> str:
    """Reduce any host/absolute/traversal path to a safe basename or relative label."""
    text = str(value).strip()
    if not text:
        return ""
    if (
        re.match(r"(?i)^[A-Za-z]:[\\/]", text)
        or text.startswith(("/", "\\"))
        or ".." in re.split(r"[\\/]+", text)
    ):
        return _basename_from_path_text(text)
    normalized = text.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p and p != "."]
    if parts and parts[0].lower() in {"users", "home"}:
        return _basename_from_path_text(normalized)
    return "/".join(parts)


def sanitize_text(text: str) -> str:
    """Scrub host filesystem paths and user home prefixes from arbitrary text."""
    out = _UNC_PATH_RE.sub(lambda m: _basename_from_path_text(m.group(0)), text)
    out = _WIN_ABS_PATH_RE.sub(lambda m: _basename_from_path_text(m.group(0)), out)
    out = _UNIX_HOST_PATH_RE.sub(lambda m: _basename_from_path_text(m.group(0)), out)
    return out


def sanitize_json_payload(value: Any) -> Any:
    """Recursively sanitize a JSON-serializable payload so no host paths or usernames leak."""
    if isinstance(value, Path):
        return sanitize_path_label(value)
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, dict):
        return {
            (sanitize_text(k) if isinstance(k, str) else k): sanitize_json_payload(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_json_payload(item) for item in value]
    return value


def _num(v: Any) -> int:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0


def presents_from_samples(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Capture-style presents (cumulative counters) from the runtime's `present_samples`
    ({index, wall_ms, draws, resources}: per-present deltas), for runs without a GPU capture."""
    out: list[dict[str, Any]] = []
    draws = res = 0
    for s in samples:
        draws += _num(s.get("draws"))
        res += _num(s.get("resources"))
        out.append({"wall_ms": s.get("wall_ms") or 0,
                    "counters": {"draws_issued": draws, "resources_created": res}})
    return out


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
        if cur is not None and not cur.get("folded") and new_res == 0 and abs(draws - cur["draws_per_frame"]) <= 1:
            cur["end_time"] = t
            cur["presents"] += 1
            continue
        if cur is not None and len(segs) >= max_segments:
            # Past the cap the last segment holds every later present. Mark it so readers do not take its
            # first draw count for a screen that lasted to the end: it carries the folded draws range and the
            # most recent draw count.
            if not cur.get("folded"):
                cur["folded"] = True
                cur["draws_min"] = cur["draws_max"] = cur["draws_per_frame"]
            cur["end_time"] = t
            cur["presents"] += 1
            cur["draws_min"] = min(cur["draws_min"], draws)
            cur["draws_max"] = max(cur["draws_max"], draws)
            cur["draws_per_frame"] = draws
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
                     f"{s['draws_per_frame']:11d}  {s['new_resources_at_start']:7d}  {s['input_events_in_segment']:6d}"
                     + (f"  folded past the cap: {s['draws_min']}-{s['draws_max']} draws/frame" if s.get("folded") else ""))
    if len(segs) > limit:
        lines.append(f"  ... {len(segs) - limit} more")
    return lines


_PHASE_TAG_ALIASES: dict[str, str] = {
    "bink2_video": "bink2_video",
    "bink2": "bink2_video",
    "4k_bink2": "bink2_video",
    "bink2_4k": "bink2_video",
    "video": "bink2_video",
    "menu": "menu",
    "ui": "menu",
    "main_menu": "menu",
    "setup_menu": "menu",
    "legal_notices": "menu",
    "loading_spinner": "loading_spinner",
    "spinner": "loading_spinner",
    "loading": "loading_spinner",
}


def _finite_non_negative(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    val = float(value)
    if not math.isfinite(val) or val < 0.0:
        return None
    return val


def _finite_non_negative_int(value: Any) -> int | None:
    val = _finite_non_negative(value)
    if val is None:
        return None
    return int(val)


def _first_non_negative(*candidates: Any) -> float | None:
    for candidate in candidates:
        val = _finite_non_negative(candidate)
        if val is not None:
            return val
    return None


def _normalize_phase_tag(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    cleaned = re.sub(r"[\s\-]+", "_", raw.strip().lower())
    if not cleaned:
        return None
    return _PHASE_TAG_ALIASES.get(cleaned, cleaned)


def _validate_window_ms(window_ms: float) -> float:
    if isinstance(window_ms, bool) or not isinstance(window_ms, (int, float)):
        raise ValueError("window_ms must be a positive finite number")
    val = float(window_ms)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError("window_ms must be a positive finite number")
    return val


def _validate_draws_range(draws_range: Sequence[float]) -> tuple[float, float]:
    if not isinstance(draws_range, (tuple, list)) or len(draws_range) != 2:
        raise ValueError("bink2_draws_range must be a 2-tuple (min_draws, max_draws)")
    lo = _finite_non_negative(draws_range[0])
    hi = _finite_non_negative(draws_range[1])
    if lo is None or hi is None or lo > hi:
        raise ValueError("bink2_draws_range must contain non-negative finite bounds with min <= max")
    return lo, hi


def _normalize_phase_presents(presents: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """Normalize heterogeneous `presents` / `present_samples` entries into a uniform list."""
    if not presents:
        return []
    raw_items = [p for p in presents if isinstance(p, Mapping)]
    if not raw_items:
        return []

    explicit_counts: list[int | None] = []
    for p in raw_items:
        cnt = _finite_non_negative_int(
            p.get("present_count") if "present_count" in p else p.get("presents")
        )
        explicit_counts.append(cnt)
    use_explicit_counts = all(c is not None for c in explicit_counts)

    normalized: list[dict[str, Any]] = []
    prev_draws_issued: float | None = None
    prev_res: int = 0
    prev_cum_presents: int | None = None
    seq_presents = 0

    for idx, p in enumerate(raw_items):
        wall_ms = _finite_non_negative(p.get("wall_ms"))
        if wall_ms is None:
            continue

        counters = p.get("counters") if isinstance(p.get("counters"), Mapping) else {}
        explicit_phase = _normalize_phase_tag(p.get("phase"))

        if use_explicit_counts:
            cum_p = int(explicit_counts[idx])  # type: ignore[arg-type]
            if prev_cum_presents is None:
                delta_p = 1 if cum_p > 0 else 0
            else:
                if cum_p < prev_cum_presents:
                    cum_p = prev_cum_presents
                delta_p = cum_p - prev_cum_presents
            prev_cum_presents = cum_p
        else:
            seq_presents += 1
            cum_p = seq_presents
            delta_p = 1

        # Resources delta (cumulative counters.resources_created or per-present resources)
        res_total = _finite_non_negative_int(counters.get("resources_created"))
        if res_total is not None:
            new_res = max(0, res_total - prev_res)
            prev_res = res_total
        else:
            res_delta = _finite_non_negative_int(p.get("resources"))
            new_res = res_delta if res_delta is not None else 0

        # Draws per frame: prefer explicit draws_per_frame, then per-present draws, then cumulative draws_issued.
        dpf = _finite_non_negative(p.get("draws_per_frame"))
        if dpf is None:
            dpf = _finite_non_negative(p.get("draws"))
        if dpf is None:
            draws_issued = _first_non_negative(counters.get("draws_issued"), p.get("draws_issued"))
            if draws_issued is not None:
                if prev_draws_issued is None:
                    delta_draws = draws_issued
                    denom = float(cum_p) if (use_explicit_counts and cum_p > 0) else 1.0
                else:
                    delta_draws = max(0.0, draws_issued - prev_draws_issued)
                    denom = float(delta_p) if delta_p > 0 else 0.0
                prev_draws_issued = draws_issued
                if denom > 0.0:
                    dpf = delta_draws / denom

        normalized.append({
            "wall_ms": wall_ms,
            "cum_presents": cum_p,
            "delta_presents": delta_p,
            "draws_per_frame": dpf,
            "explicit_phase": explicit_phase,
            "new_res": new_res,
        })

    return normalized


def _draw_bucket(dpf: float | None, lo: float, hi: float) -> str:
    if dpf is None:
        return "none"
    if lo <= dpf <= hi:
        return "bink2"
    if dpf > hi:
        return "high"
    if dpf == 0.0:
        return "zero"
    return "low"


def _compute_span_fps_windows(
    points: Sequence[tuple[float, int]],
    window_ms: float,
) -> tuple[list[float], list[float], float, int]:
    """Return `(rolling_windows, inter_present_fps, total_dt_ms, total_dc)` for one contiguous span."""
    deduped: list[tuple[float, int]] = []
    for w, c in points:
        if deduped and w < deduped[-1][0]:
            continue
        if deduped and w == deduped[-1][0]:
            deduped[-1] = (w, max(deduped[-1][1], c))
        else:
            if deduped and c < deduped[-1][1]:
                c = deduped[-1][1]
            deduped.append((w, c))

    if len(deduped) < 2:
        return [], [], 0.0, 0

    total_dt = deduped[-1][0] - deduped[0][0]
    total_dc = deduped[-1][1] - deduped[0][1]
    if total_dt <= 0.0 or total_dc < 0:
        return [], [], 0.0, 0

    rolling: list[float] = []
    has_full_window = False
    i = 0
    n = len(deduped)
    while i < n - 1:
        t_start, c_start = deduped[i]
        j = i + 1
        while j < n - 1 and (deduped[j][0] - t_start) < window_ms:
            j += 1
        dt = deduped[j][0] - t_start
        dc = deduped[j][1] - c_start
        if dt > 0.0 and dc >= 0:
            if dt >= window_ms:
                has_full_window = True
                rolling.append(float(dc) * 1000.0 / dt)
            elif has_full_window and dt >= (window_ms * 0.5):
                rolling.append(float(dc) * 1000.0 / dt)
        i = j

    inter: list[float] = []
    for k in range(1, n):
        dt = deduped[k][0] - deduped[k - 1][0]
        dc = deduped[k][1] - deduped[k - 1][1]
        if dt > 0.0 and dc >= 0:
            inter.append(float(dc) * 1000.0 / dt)

    return rolling, inter, total_dt, total_dc


def _aggregate_phase_stats(
    raw_segments: Sequence[dict[str, Any]],
    seg_indices: Sequence[int],
    window_ms: float,
) -> dict[str, Any] | None:
    if not seg_indices:
        return None

    # Group contiguous segment indices into contiguous spans so FPS windows never bridge gaps.
    spans: list[list[dict[str, Any]]] = []
    prev_idx: int | None = None
    for idx in seg_indices:
        if prev_idx is None or idx != prev_idx + 1:
            spans.append([raw_segments[idx]])
        else:
            spans[-1].append(raw_segments[idx])
        prev_idx = idx

    rolling_all: list[float] = []
    inter_all: list[float] = []
    seg_avgs: list[float] = []
    total_dt_ms = 0.0
    total_dc = 0
    total_presents = 0
    start_ms = raw_segments[seg_indices[0]]["samples"][0]["wall_ms"]
    end_ms = raw_segments[seg_indices[-1]]["samples"][-1]["wall_ms"]

    for span in spans:
        span_samples = [s for seg in span for s in seg["samples"]]
        total_presents += sum(s["delta_presents"] for s in span_samples)
        points = [(s["wall_ms"], s["cum_presents"]) for s in span_samples]
        rolling, inter, dt_ms, dc = _compute_span_fps_windows(points, window_ms)
        rolling_all.extend(rolling)
        inter_all.extend(inter)
        if dt_ms > 0.0:
            total_dt_ms += dt_ms
            total_dc += dc
            seg_avgs.append(float(dc) * 1000.0 / dt_ms)

    fps_samples = rolling_all if rolling_all else (inter_all if inter_all else seg_avgs)
    if fps_samples:
        fps_min: float | None = round(min(fps_samples), 2)
        fps_median: float | None = round(float(statistics.median(fps_samples)), 2)
        fps_max: float | None = round(max(fps_samples), 2)
    else:
        fps_min = None
        fps_median = None
        fps_max = None

    fps_avg: float | None = (
        round(float(total_dc) * 1000.0 / total_dt_ms, 2) if total_dt_ms > 0.0 else None
    )

    return {
        "fps_min": fps_min,
        "fps_median": fps_median,
        "fps_max": fps_max,
        "fps_avg": fps_avg,
        "duration_s": round(total_dt_ms / 1000.0, 2),
        "presents": int(total_presents),
        "segments": len(seg_indices),
        "start_s": round(start_ms / 1000.0, 2),
        "end_s": round(end_ms / 1000.0, 2),
    }


def analyze_boot_phases(
    presents: Sequence[Mapping[str, Any]] | None,
    *,
    window_ms: float = DEFAULT_PHASE_WINDOW_MS,
    bink2_draws_range: tuple[float, float] = DEFAULT_BINK2_DRAWS_RANGE,
    black_loading_s: float | None = None,
    loading_spinner_s: float | None = None,
) -> dict[str, Any]:
    """Classify boot phases (`bink2_video`, `menu`, `loading_spinner`) and compute FPS stats."""
    clean_window_ms = _validate_window_ms(window_ms)
    lo, hi = _validate_draws_range(bink2_draws_range)
    clean_black_s = _finite_non_negative(black_loading_s)
    clean_spinner_s = _finite_non_negative(loading_spinner_s)

    samples = _normalize_phase_presents(presents)

    # Collapse consecutive samples into raw draw/phase segments.
    raw_segs: list[dict[str, Any]] = []
    for sample in samples:
        tag = sample["explicit_phase"]
        dpf = sample["draws_per_frame"]
        bucket = _draw_bucket(dpf, lo, hi)
        cur = raw_segs[-1] if raw_segs else None
        same = False
        if cur is not None and cur["explicit_phase"] == tag:
            if tag is not None:
                same = True
            elif cur["bucket"] == bucket:
                if dpf is None and cur["anchor_dpf"] is None:
                    same = True
                elif (
                    dpf is not None
                    and cur["anchor_dpf"] is not None
                    and sample["new_res"] == 0
                    and abs(dpf - cur["anchor_dpf"]) <= 1.0
                ):
                    same = True
        if same and cur is not None:
            cur["samples"].append(sample)
            cur["presents"] += sample["delta_presents"]
            if dpf is not None:
                cur["dpf_sum"] += dpf
                cur["dpf_count"] += 1
        else:
            raw_segs.append({
                "explicit_phase": tag,
                "bucket": bucket,
                "anchor_dpf": dpf,
                "dpf_sum": dpf if dpf is not None else 0.0,
                "dpf_count": 1 if dpf is not None else 0,
                "presents": sample["delta_presents"],
                "samples": [sample],
                "phase": None,
            })

    # Pass 1: Explicit phase tags and Bink2 video (`draws_per_frame` in [lo, hi]).
    for seg in raw_segs:
        tag = seg["explicit_phase"]
        if tag in ("bink2_video", "menu", "loading_spinner"):
            seg["phase"] = tag
        elif tag is not None:
            seg["phase"] = "other"

    i = 0
    n_segs = len(raw_segs)
    while i < n_segs:
        seg = raw_segs[i]
        if seg["phase"] is None and seg["bucket"] == "bink2":
            j = i
            run_presents = 0
            while j < n_segs and raw_segs[j]["phase"] is None and raw_segs[j]["bucket"] == "bink2":
                run_presents += raw_segs[j]["presents"]
                j += 1
            if run_presents >= DEFAULT_MIN_BINK2_PRESENTS:
                for k in range(i, j):
                    raw_segs[k]["phase"] = "bink2_video"
            i = j
        else:
            i += 1

    # Pass 2: Interactive menu (`draws_per_frame > hi`; prefer post-Bink2 when both pre & post exist).
    first_bink2_idx = next((idx for idx, s in enumerate(raw_segs) if s["phase"] == "bink2_video"), None)
    has_post_bink2_high = (
        first_bink2_idx is not None
        and any(
            s["phase"] is None and s["bucket"] == "high"
            for s in raw_segs[first_bink2_idx + 1:]
        )
    )
    for idx, seg in enumerate(raw_segs):
        if seg["phase"] is None and seg["bucket"] == "high":
            if has_post_bink2_high and first_bink2_idx is not None and idx < first_bink2_idx:
                seg["phase"] = "start_screen"
            else:
                seg["phase"] = "menu"

    # Pass 3: Loading spinner (sustained >= 2s low-draw steady segments).
    first_menu_idx = next((idx for idx, s in enumerate(raw_segs) if s["phase"] == "menu"), None)
    low_max = min(2.0, lo - 1e-9) if lo > 0.0 else -1.0
    i = 0
    while i < n_segs:
        seg = raw_segs[i]
        dpf_avg = (seg["dpf_sum"] / seg["dpf_count"]) if seg["dpf_count"] > 0 else None
        is_post_menu = first_menu_idx is not None and i > first_menu_idx
        eligible = (
            seg["phase"] is None
            and dpf_avg is not None
            and dpf_avg <= low_max
            and (is_post_menu or dpf_avg >= 1.0)
        )
        if eligible:
            j = i
            while j < n_segs:
                sj = raw_segs[j]
                sj_dpf = (sj["dpf_sum"] / sj["dpf_count"]) if sj["dpf_count"] > 0 else None
                sj_post_menu = first_menu_idx is not None and j > first_menu_idx
                if (
                    sj["phase"] is None
                    and sj_dpf is not None
                    and sj_dpf <= low_max
                    and (sj_post_menu or sj_dpf >= 1.0)
                ):
                    j += 1
                else:
                    break
            run_start_ms = raw_segs[i]["samples"][0]["wall_ms"]
            run_end_ms = raw_segs[j - 1]["samples"][-1]["wall_ms"]
            run_dur_s = (run_end_ms - run_start_ms) / 1000.0
            run_presents = sum(raw_segs[k]["presents"] for k in range(i, j))
            if run_dur_s >= DEFAULT_MIN_LOADING_SPINNER_S and run_presents >= 2:
                for k in range(i, j):
                    raw_segs[k]["phase"] = "loading_spinner"
            i = j
        else:
            i += 1

    bink2_indices = [idx for idx, s in enumerate(raw_segs) if s["phase"] == "bink2_video"]
    menu_indices = [idx for idx, s in enumerate(raw_segs) if s["phase"] == "menu"]
    spinner_indices = [idx for idx, s in enumerate(raw_segs) if s["phase"] == "loading_spinner"]

    bink2_stats = _aggregate_phase_stats(raw_segs, bink2_indices, clean_window_ms)
    menu_stats = _aggregate_phase_stats(raw_segs, menu_indices, clean_window_ms)
    spinner_stats = _aggregate_phase_stats(raw_segs, spinner_indices, clean_window_ms)

    effective_spinner_s: float | None = None
    if spinner_stats is not None:
        effective_spinner_s = spinner_stats["duration_s"]
    elif clean_spinner_s is not None:
        effective_spinner_s = round(clean_spinner_s, 2)
        spinner_stats = {
            "fps_min": None,
            "fps_median": None,
            "fps_max": None,
            "fps_avg": None,
            "duration_s": effective_spinner_s,
            "presents": 0,
            "segments": 0,
            "start_s": None,
            "end_s": None,
        }

    return {
        "window_ms": round(clean_window_ms, 2),
        "bink2_draws_range": [
            int(lo) if lo.is_integer() else round(lo, 2),
            int(hi) if hi.is_integer() else round(hi, 2),
        ],
        "bink2_video": bink2_stats,
        "menu": menu_stats,
        "loading_spinner": spinner_stats,
        "black_loading_s": round(clean_black_s, 2) if clean_black_s is not None else None,
        "loading_spinner_s": effective_spinner_s,
    }


def _analyze_timeline_segments_fallback(
    timeline: Sequence[Mapping[str, Any]],
    *,
    window_ms: float = DEFAULT_PHASE_WINDOW_MS,
    bink2_draws_range: tuple[float, float] = DEFAULT_BINK2_DRAWS_RANGE,
) -> dict[str, Any]:
    """Synthesize phase stats from pre-collapsed `screen_timeline` segment dicts."""
    lo, hi = _validate_draws_range(bink2_draws_range)
    segs: list[dict[str, Any]] = []
    for raw in timeline:
        if not isinstance(raw, Mapping):
            continue
        start_ms = _finite_non_negative(raw.get("start_time"))
        end_ms = _finite_non_negative(raw.get("end_time"))
        dpf = _finite_non_negative(raw.get("draws_per_frame"))
        presents = _finite_non_negative_int(raw.get("presents"))
        if start_ms is None or end_ms is None or dpf is None or presents is None or presents <= 0:
            continue
        if end_ms < start_ms:
            end_ms = start_ms
        segs.append({
            "start_ms": start_ms,
            "end_ms": end_ms,
            "dpf": dpf,
            "presents": presents,
            "explicit_phase": _normalize_phase_tag(raw.get("phase")),
            "phase": None,
        })

    for s in segs:
        if s["explicit_phase"] in ("bink2_video", "menu", "loading_spinner"):
            s["phase"] = s["explicit_phase"]
        elif s["explicit_phase"] is not None:
            s["phase"] = "other"
        elif lo <= s["dpf"] <= hi and s["presents"] >= DEFAULT_MIN_BINK2_PRESENTS:
            s["phase"] = "bink2_video"

    first_bink2 = next((i for i, s in enumerate(segs) if s["phase"] == "bink2_video"), None)
    has_post_bink2_high = (
        first_bink2 is not None
        and any(s["phase"] is None and s["dpf"] > hi for s in segs[first_bink2 + 1:])
    )
    for i, s in enumerate(segs):
        if s["phase"] is None and s["dpf"] > hi:
            s["phase"] = "start_screen" if (has_post_bink2_high and first_bink2 is not None and i < first_bink2) else "menu"

    first_menu = next((i for i, s in enumerate(segs) if s["phase"] == "menu"), None)
    low_max = min(2.0, lo - 1e-9) if lo > 0.0 else -1.0
    for i, s in enumerate(segs):
        if s["phase"] is None and s["dpf"] <= low_max:
            dur_s = (s["end_ms"] - s["start_ms"]) / 1000.0
            is_post_menu = first_menu is not None and i > first_menu
            if dur_s >= DEFAULT_MIN_LOADING_SPINNER_S and s["presents"] >= 2 and (is_post_menu or s["dpf"] >= 1.0):
                s["phase"] = "loading_spinner"

    def build_phase(name: str) -> dict[str, Any] | None:
        matched = [s for s in segs if s["phase"] == name]
        if not matched:
            return None
        total_dt_ms = sum(max(0.0, s["end_ms"] - s["start_ms"]) for s in matched)
        total_presents = sum(s["presents"] for s in matched)
        total_intervals = sum(max(0, s["presents"] - 1) for s in matched if s["end_ms"] > s["start_ms"])
        seg_fps = [
            float(max(0, s["presents"] - 1)) * 1000.0 / (s["end_ms"] - s["start_ms"])
            for s in matched
            if s["end_ms"] > s["start_ms"] and s["presents"] >= 2
        ]
        return {
            "fps_min": round(min(seg_fps), 2) if seg_fps else None,
            "fps_median": round(float(statistics.median(seg_fps)), 2) if seg_fps else None,
            "fps_max": round(max(seg_fps), 2) if seg_fps else None,
            "fps_avg": round(float(total_intervals) * 1000.0 / total_dt_ms, 2) if total_dt_ms > 0.0 else None,
            "duration_s": round(total_dt_ms / 1000.0, 2),
            "presents": int(total_presents),
            "segments": len(matched),
            "start_s": round(matched[0]["start_ms"] / 1000.0, 2),
            "end_s": round(matched[-1]["end_ms"] / 1000.0, 2),
        }

    spinner = build_phase("loading_spinner")
    return {
        "window_ms": round(_validate_window_ms(window_ms), 2),
        "bink2_draws_range": [
            int(lo) if lo.is_integer() else round(lo, 2),
            int(hi) if hi.is_integer() else round(hi, 2),
        ],
        "bink2_video": build_phase("bink2_video"),
        "menu": build_phase("menu"),
        "loading_spinner": spinner,
        "black_loading_s": None,
        "loading_spinner_s": spinner["duration_s"] if spinner is not None else None,
    }


def _merge_phase_overrides(
    existing: dict[str, Any] | None,
    overrides: Mapping[str, float | int | None],
) -> dict[str, Any] | None:
    clean = {k: v for k, v in overrides.items() if v is not None}
    if not clean:
        return existing
    base = dict(existing) if isinstance(existing, dict) else {
        "fps_min": None,
        "fps_median": None,
        "fps_max": None,
        "fps_avg": None,
        "duration_s": None,
        "presents": None,
        "segments": None,
        "start_s": None,
        "end_s": None,
    }
    for key, val in clean.items():
        if base.get(key) is None:
            base[key] = val
    return base


def extract_boot_phases_from_report(
    report: Mapping[str, Any],
    *,
    window_ms: float = DEFAULT_PHASE_WINDOW_MS,
) -> dict[str, Any]:
    """Extract sanitized boot-phase FPS and duration stats from a boot report dict."""
    clean_window_ms = _validate_window_ms(window_ms)
    if not isinstance(report, Mapping):
        return analyze_boot_phases([], window_ms=clean_window_ms)

    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    telemetry = report.get("telemetry") if isinstance(report.get("telemetry"), Mapping) else {}
    counters = report.get("counters") if isinstance(report.get("counters"), Mapping) else {}
    phases_obj = report.get("phases") if isinstance(report.get("phases"), Mapping) else (
        report.get("boot_phases") if isinstance(report.get("boot_phases"), Mapping) else {}
    )
    summary_phases = summary.get("phases") if isinstance(summary.get("phases"), Mapping) else {}
    telemetry_phases = telemetry.get("phases") if isinstance(telemetry.get("phases"), Mapping) else {}

    sources = (report, phases_obj, summary, summary_phases, telemetry, telemetry_phases, counters)

    def pick_float(*keys: str) -> float | None:
        for src in sources:
            for key in keys:
                val = _finite_non_negative(src.get(key))
                if val is not None:
                    return round(val, 2)
        return None

    def pick_nested(section: str, *keys: str) -> float | None:
        for container in (phases_obj, summary_phases, telemetry_phases, report, summary, telemetry):
            sub = container.get(section)
            if isinstance(sub, Mapping):
                for key in keys:
                    val = _finite_non_negative(sub.get(key))
                    if val is not None:
                        return round(val, 2)
        return None

    def pick_nested_int(section: str, *keys: str) -> int | None:
        for container in (phases_obj, summary_phases, telemetry_phases, report, summary, telemetry):
            sub = container.get(section)
            if isinstance(sub, Mapping):
                for key in keys:
                    val = _finite_non_negative_int(sub.get(key))
                    if val is not None:
                        return val
        return None

    black_loading_s = pick_float("black_loading_s", "blackLoadingS")
    if black_loading_s is None:
        black_loading_ms = pick_float("black_loading_ms", "blackLoadingMs")
        if black_loading_ms is not None:
            black_loading_s = round(black_loading_ms / 1000.0, 2)

    explicit_spinner_s = _first_non_negative(
        pick_float("loading_spinner_s", "loadingSpinnerS"),
        pick_nested("loading_spinner", "duration_s"),
    )
    if explicit_spinner_s is None:
        spinner_ms = pick_float("loading_spinner_ms", "loadingSpinnerMs")
        if spinner_ms is not None:
            explicit_spinner_s = round(spinner_ms / 1000.0, 2)

    # 1. Analyze presents or present_samples.
    presents_src = report.get("presents")
    if not isinstance(presents_src, list) or not presents_src:
        presents_src = report.get("present_samples")
    result = analyze_boot_phases(
        presents_src if isinstance(presents_src, list) else [],
        window_ms=clean_window_ms,
        black_loading_s=black_loading_s,
        loading_spinner_s=explicit_spinner_s,
    )

    # 2. Fallback to pre-collapsed `screen_timeline` segments if no phases were detected from presents.
    if (
        result["bink2_video"] is None
        and result["menu"] is None
        and result["loading_spinner"] is None
        and isinstance(report.get("screen_timeline"), list)
        and report["screen_timeline"]
    ):
        tl_result = _analyze_timeline_segments_fallback(
            report["screen_timeline"],
            window_ms=clean_window_ms,
        )
        result["bink2_video"] = tl_result["bink2_video"]
        result["menu"] = tl_result["menu"]
        result["loading_spinner"] = tl_result["loading_spinner"]
        if result["loading_spinner_s"] is None:
            result["loading_spinner_s"] = tl_result["loading_spinner_s"]

    # 3. Merge optional pre-computed summary/telemetry phase fields when absent from presents analysis.
    bink2_overrides = {
        "fps_min": _first_non_negative(pick_float("bink2_fps_min", "bink2FpsMin"), pick_nested("bink2_video", "fps_min")),
        "fps_median": _first_non_negative(pick_float("bink2_fps_median", "bink2FpsMedian"), pick_nested("bink2_video", "fps_median")),
        "fps_max": _first_non_negative(pick_float("bink2_fps_max", "bink2FpsMax"), pick_nested("bink2_video", "fps_max")),
        "fps_avg": _first_non_negative(pick_float("bink2_fps_avg", "bink2FpsAvg"), pick_nested("bink2_video", "fps_avg")),
        "duration_s": _first_non_negative(pick_float("bink2_duration_s", "bink2DurationS"), pick_nested("bink2_video", "duration_s")),
        "presents": pick_nested_int("bink2_video", "presents"),
        "segments": pick_nested_int("bink2_video", "segments"),
        "start_s": pick_nested("bink2_video", "start_s"),
        "end_s": pick_nested("bink2_video", "end_s"),
    }
    result["bink2_video"] = _merge_phase_overrides(result["bink2_video"], bink2_overrides)

    menu_overrides = {
        "fps_min": _first_non_negative(pick_float("menu_fps_min", "menuFpsMin"), pick_nested("menu", "fps_min")),
        "fps_median": _first_non_negative(pick_float("menu_fps_median", "menuFpsMedian"), pick_nested("menu", "fps_median")),
        "fps_max": _first_non_negative(pick_float("menu_fps_max", "menuFpsMax"), pick_nested("menu", "fps_max")),
        "fps_avg": _first_non_negative(pick_float("menu_fps_avg", "menuFpsAvg"), pick_nested("menu", "fps_avg")),
        "duration_s": _first_non_negative(pick_float("menu_duration_s", "menuDurationS"), pick_nested("menu", "duration_s")),
        "presents": pick_nested_int("menu", "presents"),
        "segments": pick_nested_int("menu", "segments"),
        "start_s": pick_nested("menu", "start_s"),
        "end_s": pick_nested("menu", "end_s"),
    }
    result["menu"] = _merge_phase_overrides(result["menu"], menu_overrides)

    spinner_overrides = {
        "fps_min": _first_non_negative(pick_float("loading_spinner_fps_min"), pick_nested("loading_spinner", "fps_min")),
        "fps_median": _first_non_negative(pick_float("loading_spinner_fps_median"), pick_nested("loading_spinner", "fps_median")),
        "fps_max": _first_non_negative(pick_float("loading_spinner_fps_max"), pick_nested("loading_spinner", "fps_max")),
        "fps_avg": _first_non_negative(pick_float("loading_spinner_fps_avg"), pick_nested("loading_spinner", "fps_avg")),
        "duration_s": round(explicit_spinner_s, 2) if explicit_spinner_s is not None else None,
        "presents": pick_nested_int("loading_spinner", "presents"),
        "segments": pick_nested_int("loading_spinner", "segments"),
        "start_s": pick_nested("loading_spinner", "start_s"),
        "end_s": pick_nested("loading_spinner", "end_s"),
    }
    result["loading_spinner"] = _merge_phase_overrides(result["loading_spinner"], spinner_overrides)
    if result["loading_spinner_s"] is None and isinstance(result["loading_spinner"], dict):
        result["loading_spinner_s"] = result["loading_spinner"].get("duration_s")

    return sanitize_json_payload(result)


def format_phase_summary(phases: Mapping[str, Any]) -> list[str]:
    """Render sanitized plain-text summary lines for analyzed boot phases."""
    window_ms = _finite_non_negative(phases.get("window_ms"))
    header = (
        f"boot phases (window={window_ms:.0f}ms):"
        if window_ms is not None
        else "boot phases:"
    )
    lines: list[str] = [header]
    any_phase = False

    def _fmt_opt(val: Any, suffix: str = "") -> str:
        num = _finite_non_negative(val)
        return f"{num:.2f}{suffix}" if num is not None else "n/a"

    for key in ("bink2_video", "menu", "loading_spinner"):
        section = phases.get(key)
        if not isinstance(section, Mapping):
            continue
        any_phase = True
        parts: list[str] = []
        dur = _finite_non_negative(section.get("duration_s"))
        if dur is not None:
            parts.append(f"duration={dur:.2f}s")
        pres = _finite_non_negative_int(section.get("presents"))
        if pres is not None:
            parts.append(f"presents={pres}")
        segs = _finite_non_negative_int(section.get("segments"))
        if segs is not None:
            parts.append(f"segments={segs}")
        fps_vals = [section.get("fps_min"), section.get("fps_median"), section.get("fps_max"), section.get("fps_avg")]
        if any(_finite_non_negative(v) is not None for v in fps_vals):
            parts.append(
                "fps(min/med/max/avg)="
                f"{_fmt_opt(section.get('fps_min'))}/"
                f"{_fmt_opt(section.get('fps_median'))}/"
                f"{_fmt_opt(section.get('fps_max'))}/"
                f"{_fmt_opt(section.get('fps_avg'))}"
            )
        start_s = _finite_non_negative(section.get("start_s"))
        end_s = _finite_non_negative(section.get("end_s"))
        if start_s is not None and end_s is not None:
            parts.append(f"span={start_s:.2f}s..{end_s:.2f}s")
        lines.append(f"  {key}: {', '.join(parts) if parts else 'detected'}")

    black_s = _finite_non_negative(phases.get("black_loading_s"))
    if black_s is not None:
        lines.append(f"  black_loading_s: {black_s:.2f}s")
    spinner_s = _finite_non_negative(phases.get("loading_spinner_s"))
    if spinner_s is not None and not isinstance(phases.get("loading_spinner"), Mapping):
        any_phase = True
        lines.append(f"  loading_spinner_s: {spinner_s:.2f}s")

    if not any_phase and black_s is None:
        return ["boot phases: none detected"]
    return [sanitize_text(line) for line in lines]


def _import_pil_image() -> Any | None:
    try:
        from PIL import Image  # type: ignore[import-not-found]

        return Image
    except ImportError:
        return None


def _paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _decode_png_rgb_fallback(data: bytes) -> tuple[int, int, bytes] | None:
    """Minimal stdlib 8-bit PNG decoder returning `(width, height, rgb_bytes)`."""
    if len(data) < 33 or not data.startswith(_PNG_SIGNATURE):
        return None
    pos = len(_PNG_SIGNATURE)
    width = height = bit_depth = color_type = interlace = 0
    palette = b""
    idat_parts: list[bytes] = []
    while pos + 12 <= len(data):
        length, ctype = struct.unpack(">I4s", data[pos:pos + 8])
        pos += 8
        if pos + length + 4 > len(data):
            return None
        chunk = data[pos:pos + length]
        pos += length + 4
        if ctype == b"IHDR":
            if length != 13:
                return None
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
                ">IIBBBBB", chunk
            )
            if compression != 0 or filter_method != 0 or interlace != 0 or bit_depth != 8:
                return None
        elif ctype == b"PLTE":
            palette = chunk
        elif ctype == b"IDAT":
            idat_parts.append(chunk)
        elif ctype == b"IEND":
            break
    if width <= 0 or height <= 0 or not idat_parts:
        return None
    bpp_map = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    bpp = bpp_map.get(color_type)
    if bpp is None:
        return None
    if color_type == 3 and len(palette) < 3:
        return None
    try:
        raw = zlib.decompress(b"".join(idat_parts))
    except zlib.error:
        return None
    stride = width * bpp
    if len(raw) < height * (1 + stride):
        return None
    prev = bytearray(stride)
    rgb = bytearray(width * height * 3)
    out_pos = 0
    in_pos = 0
    for _ in range(height):
        filt = raw[in_pos]
        in_pos += 1
        row = bytearray(raw[in_pos:in_pos + stride])
        in_pos += stride
        if filt == 1:
            for i in range(bpp, stride):
                row[i] = (row[i] + row[i - bpp]) & 0xFF
        elif filt == 2:
            for i in range(stride):
                row[i] = (row[i] + prev[i]) & 0xFF
        elif filt == 3:
            for i in range(stride):
                left = row[i - bpp] if i >= bpp else 0
                row[i] = (row[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif filt == 4:
            for i in range(stride):
                left = row[i - bpp] if i >= bpp else 0
                up = prev[i]
                up_left = prev[i - bpp] if i >= bpp else 0
                row[i] = (row[i] + _paeth_predictor(left, up, up_left)) & 0xFF
        elif filt != 0:
            return None
        prev = row
        if color_type == 2:
            rgb[out_pos:out_pos + stride] = row
            out_pos += stride
        elif color_type == 6:
            for x in range(0, stride, 4):
                rgb[out_pos] = row[x]
                rgb[out_pos + 1] = row[x + 1]
                rgb[out_pos + 2] = row[x + 2]
                out_pos += 3
        elif color_type == 0:
            for v in row:
                rgb[out_pos] = v
                rgb[out_pos + 1] = v
                rgb[out_pos + 2] = v
                out_pos += 3
        elif color_type == 4:
            for x in range(0, stride, 2):
                v = row[x]
                rgb[out_pos] = v
                rgb[out_pos + 1] = v
                rgb[out_pos + 2] = v
                out_pos += 3
        elif color_type == 3:
            max_idx = (len(palette) // 3) - 1
            for idx in row:
                p_off = min(idx, max_idx) * 3
                rgb[out_pos] = palette[p_off]
                rgb[out_pos + 1] = palette[p_off + 1]
                rgb[out_pos + 2] = palette[p_off + 2]
                out_pos += 3
    return width, height, bytes(rgb)


def _sample_rgb_metrics(
    width: int,
    height: int,
    rgb_bytes: bytes,
    thumb_size: tuple[int, int] = THUMB_SIZE,
) -> tuple[float, bytes]:
    pixel_count = width * height
    if pixel_count <= 0:
        return 0.0, b""
    luma_total = 0.0
    for i in range(0, pixel_count * 3, 3):
        luma_total += 0.299 * rgb_bytes[i] + 0.587 * rgb_bytes[i + 1] + 0.114 * rgb_bytes[i + 2]
    mean_luma = luma_total / pixel_count

    tw, th = thumb_size
    thumb = bytearray(tw * th * 3)
    out_pos = 0
    for ty in range(th):
        sy = min(height - 1, (ty * height) // th)
        row_off = sy * width * 3
        for tx in range(tw):
            sx = min(width - 1, (tx * width) // tw)
            src_off = row_off + sx * 3
            thumb[out_pos] = rgb_bytes[src_off]
            thumb[out_pos + 1] = rgb_bytes[src_off + 1]
            thumb[out_pos + 2] = rgb_bytes[src_off + 2]
            out_pos += 3
    return mean_luma, bytes(thumb)


def analyze_frame_image(
    frame_path: Path,
    *,
    thumb_size: tuple[int, int] = THUMB_SIZE,
    pil_image_mod: Any | None = None,
    _auto_import_pil: bool = True,
) -> tuple[float, bytes] | None:
    """Return `(mean_luma, rgb_thumbnail_bytes)` for one image, or `None` if unreadable."""
    try:
        raw = frame_path.read_bytes()
    except OSError:
        return None
    if not raw:
        return None

    img_mod = pil_image_mod
    if img_mod is None and _auto_import_pil:
        img_mod = _import_pil_image()

    if img_mod is not None:
        try:
            with img_mod.open(io.BytesIO(raw)) as img:
                rgb = img.convert("RGB")
                width, height = rgb.size
                if width <= 0 or height <= 0:
                    return None
                gray = rgb.convert("L")
                hist = gray.histogram()
                total_px = width * height
                mean_luma = sum(val * count for val, count in enumerate(hist[:256])) / total_px
                resampling = getattr(img_mod, "Resampling", img_mod)
                resample = getattr(resampling, "BILINEAR", 2)
                thumb = rgb.resize(thumb_size, resample=resample).tobytes()
                return float(mean_luma), thumb
        except Exception:
            return None

    decoded = _decode_png_rgb_fallback(raw)
    if decoded is None:
        return None
    width, height, rgb_bytes = decoded
    return _sample_rgb_metrics(width, height, rgb_bytes, thumb_size=thumb_size)


def list_capture_frames(frames_dir: Path) -> list[tuple[int, Path]]:
    """Return capture image files in natural frame order as `(frame_number, path)`."""
    if not frames_dir.exists():
        raise FileNotFoundError("frames directory does not exist")
    if not frames_dir.is_dir():
        raise NotADirectoryError("frames path is not a directory")

    numbered: list[tuple[int, int, str, Path]] = []
    unnumbered: list[tuple[str, Path]] = []
    for entry in frames_dir.iterdir():
        if not entry.is_file() or entry.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        match = _FRAME_NUM_RE.search(entry.stem)
        if match is not None:
            frame_num = int(match.group(1))
            numbered.append((0, frame_num, entry.name.lower(), entry))
        else:
            unnumbered.append((entry.name.lower(), entry))

    numbered.sort(key=lambda item: (item[0], item[1], item[2]))
    unnumbered.sort(key=lambda item: item[0])

    result: list[tuple[int, Path]] = [(num, path) for _, num, _, path in numbered]
    next_frame = (result[-1][0] + 1) if result else 1
    for _, path in unnumbered:
        result.append((next_frame, path))
        next_frame += 1
    return result


def _validate_interval_sec(interval_sec: float | None) -> float | None:
    if interval_sec is None:
        return None
    if isinstance(interval_sec, bool) or not isinstance(interval_sec, (int, float)):
        raise ValueError("--interval-sec must be a positive finite number")
    val = float(interval_sec)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError("--interval-sec must be a positive finite number")
    return val


def _validate_non_negative(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a non-negative finite number")
    val = float(value)
    if not math.isfinite(val) or val < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return val


def _thumb_mad(a: Sequence[int] | bytes, b: Sequence[int] | bytes) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(abs(x - y) for x, y in zip(a, b)) / n


def summarize_frame_samples(
    samples: Sequence[dict[str, Any]],
    *,
    interval_sec: float | None = None,
    black_luma: float = DEFAULT_BLACK_LUMA,
    diff_threshold: float = DEFAULT_DIFF_THRESHOLD,
) -> dict[str, Any]:
    """Group ordered frame metrics into sanitized screen-transition segments with optional dwell times."""
    clean_interval = _validate_interval_sec(interval_sec)
    clean_black_luma = _validate_non_negative(black_luma, "--black-luma")
    clean_diff = _validate_non_negative(diff_threshold, "--diff-threshold")

    total_frames = len(samples)
    black_frames = 0
    non_black_frames = 0
    first_non_black_frame: int | None = None
    first_non_black_ord: int | None = None

    raw_segments: list[dict[str, Any]] = []
    for ord_idx, sample in enumerate(samples):
        frame_num = int(sample["frame"])
        luma = float(sample["mean_luma"])
        thumb = sample.get("thumb")
        is_black = bool(luma <= clean_black_luma)

        if is_black:
            black_frames += 1
        else:
            non_black_frames += 1
            if first_non_black_frame is None:
                first_non_black_frame = frame_num
                first_non_black_ord = ord_idx

        cur = raw_segments[-1] if raw_segments else None
        same_segment = False
        if cur is not None and cur["is_black"] == is_black:
            if is_black:
                same_segment = True
            else:
                anchor_thumb = cur["anchor_thumb"]
                if thumb is not None and anchor_thumb is not None:
                    diff = _thumb_mad(thumb, anchor_thumb)
                else:
                    diff = abs(luma - cur["anchor_luma"])
                same_segment = diff <= clean_diff

        if same_segment and cur is not None:
            cur["end_frame"] = frame_num
            cur["frame_count"] += 1
            cur["luma_sum"] += luma
        else:
            raw_segments.append({
                "start_ord": ord_idx,
                "start_frame": frame_num,
                "end_frame": frame_num,
                "frame_count": 1,
                "luma_sum": luma,
                "is_black": is_black,
                "anchor_luma": luma,
                "anchor_thumb": thumb,
            })

    segments: list[dict[str, Any]] = []
    for seg_idx, seg in enumerate(raw_segments):
        count = seg["frame_count"]
        start_ord = seg["start_ord"]
        if clean_interval is not None:
            start_s: float | None = round(start_ord * clean_interval, 3)
            end_s: float | None = round((start_ord + count) * clean_interval, 3)
            dwell_s: float | None = round(count * clean_interval, 3)
            prev_start_ord = raw_segments[seg_idx - 1]["start_ord"] if seg_idx > 0 else start_ord
            delta_s: float | None = round((start_ord - prev_start_ord) * clean_interval, 3)
        else:
            start_s = None
            end_s = None
            dwell_s = None
            delta_s = None
        segments.append({
            "index": seg_idx,
            "start_frame": seg["start_frame"],
            "end_frame": seg["end_frame"],
            "frame_count": count,
            "start_s": start_s,
            "end_s": end_s,
            "dwell_s": dwell_s,
            "delta_s": delta_s,
            "mean_luma": round(seg["luma_sum"] / count, 2),
            "is_black": seg["is_black"],
        })

    all_black = total_frames > 0 and non_black_frames == 0
    first_non_black_s: float | None = (
        round(first_non_black_ord * clean_interval, 3)
        if clean_interval is not None and first_non_black_ord is not None
        else None
    )
    leading_black_frames = first_non_black_ord if first_non_black_ord is not None else total_frames
    black_loading_s: float | None = (
        round(leading_black_frames * clean_interval, 3)
        if clean_interval is not None
        else None
    )

    return {
        "total_frames": total_frames,
        "unique_segments": len(segments),
        "black_frames": black_frames,
        "non_black_frames": non_black_frames,
        "black_frame_count": black_frames,
        "non_black_frame_count": non_black_frames,
        "all_black": all_black,
        "first_non_black_frame": first_non_black_frame,
        "first_non_black_s": first_non_black_s,
        "black_loading_s": black_loading_s,
        "segments": segments,
    }


def analyze_frames_dir(
    frames_dir: Path,
    *,
    interval_sec: float | None = None,
    black_luma: float = DEFAULT_BLACK_LUMA,
    diff_threshold: float = DEFAULT_DIFF_THRESHOLD,
    thumb_size: tuple[int, int] = THUMB_SIZE,
    pil_image_mod: Any | None = None,
    _auto_import_pil: bool = True,
) -> dict[str, Any]:
    """Scan a capture frame directory and build its sanitized dwell-time timeline summary."""
    _validate_interval_sec(interval_sec)
    _validate_non_negative(black_luma, "--black-luma")
    _validate_non_negative(diff_threshold, "--diff-threshold")

    frames = list_capture_frames(frames_dir)
    img_mod = pil_image_mod
    if img_mod is None and _auto_import_pil:
        img_mod = _import_pil_image()

    samples: list[dict[str, Any]] = []
    for frame_num, frame_path in frames:
        metrics = analyze_frame_image(
            frame_path,
            thumb_size=thumb_size,
            pil_image_mod=img_mod,
            _auto_import_pil=False,
        )
        if metrics is None:
            continue
        mean_luma, thumb = metrics
        samples.append({
            "frame": frame_num,
            "mean_luma": mean_luma,
            "thumb": thumb,
        })

    return summarize_frame_samples(
        samples,
        interval_sec=interval_sec,
        black_luma=black_luma,
        diff_threshold=diff_threshold,
    )


def format_capture_summary(summary: dict[str, Any]) -> list[str]:
    """Render a sanitized plain-text table for a capture-directory timeline summary."""
    total = summary["total_frames"]
    unique = summary["unique_segments"]
    black = summary.get("black_frame_count", summary["black_frames"])
    non_black = summary.get("non_black_frame_count", summary["non_black_frames"])
    first_frame = summary["first_non_black_frame"]
    first_s = summary["first_non_black_s"]
    black_loading_s = summary.get("black_loading_s")
    all_black = bool(summary.get("all_black", False))
    segments = summary["segments"]

    seg_word = "segment" if unique == 1 else "segments"
    frame_word = "frame" if total == 1 else "frames"
    if total == 0:
        return ["screen timeline: 0 frames, 0 segments (black: 0, non-black: 0)"]

    if first_frame is None:
        first_desc = "none"
    elif first_s is not None:
        first_desc = f"frame {first_frame} @ {first_s:.2f}s"
    else:
        first_desc = f"frame {first_frame}"

    header_parts = [
        f"black: {black}",
        f"non-black: {non_black}",
        f"first_non_black: {first_desc}",
    ]
    if black_loading_s is not None:
        header_parts.append(f"black_loading_s: {black_loading_s:.2f}s")
    if all_black:
        header_parts.append("all_black: true")

    lines = [
        f"screen timeline: {total} {frame_word}, {unique} {seg_word} "
        f"({', '.join(header_parts)})"
    ]
    has_times = any(s.get("dwell_s") is not None for s in segments)
    if has_times:
        lines.append("  idx  frames      count  start_s    end_s  dwell_s  delta_s    luma  state")
    else:
        lines.append("  idx  frames      count    luma  state")

    for s in segments:
        span = f"{s['start_frame']}..{s['end_frame']}"
        state = "black" if s["is_black"] else "visible"
        if has_times:
            delta_val = s.get("delta_s")
            delta_col = f"{delta_val:7.2f}" if delta_val is not None else "    n/a"
            lines.append(
                f"  {s['index']:3d}  {span:10s}  {s['frame_count']:5d}  "
                f"{s['start_s']:7.2f}  {s['end_s']:7.2f}  {s['dwell_s']:7.2f}  {delta_col}  "
                f"{s['mean_luma']:6.2f}  {state}"
            )
        else:
            lines.append(
                f"  {s['index']:3d}  {span:10s}  {s['frame_count']:5d}  "
                f"{s['mean_luma']:6.2f}  {state}"
            )
    return lines


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="screen_timeline.py",
        description="Summarize screen transitions and dwell durations from captured frames or boot reports.",
    )
    parser.add_argument(
        "frames_dir",
        nargs="?",
        default=None,
        type=Path,
        help="Directory containing captured frame images (e.g. frame_0001.png / frame_0001.jpg).",
    )
    parser.add_argument(
        "--boot-report",
        type=Path,
        default=None,
        help="Path to a playgta6-boot-report JSON file for phase-by-phase FPS analysis.",
    )
    parser.add_argument(
        "--window-ms",
        type=float,
        default=DEFAULT_PHASE_WINDOW_MS,
        help=f"Rolling FPS window in milliseconds for --boot-report analysis (default: {DEFAULT_PHASE_WINDOW_MS}).",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=None,
        help=(
            "Capture interval in seconds per frame; populates start_s, end_s, dwell_s, "
            "delta_s, first_non_black_s, and black_loading_s."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a sanitized structured JSON summary instead of a plain-text table.",
    )
    parser.add_argument(
        "--black-luma",
        type=float,
        default=DEFAULT_BLACK_LUMA,
        help=f"Mean luma threshold (0-255) for black frames (default: {DEFAULT_BLACK_LUMA}).",
    )
    parser.add_argument(
        "--diff-threshold",
        type=float,
        default=DEFAULT_DIFF_THRESHOLD,
        help=f"Thumbnail RGB MAD threshold (0-255) for screen transitions (default: {DEFAULT_DIFF_THRESHOLD}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.boot_report is not None and args.frames_dir is not None:
        print("error: specify either frames_dir or --boot-report, not both", file=sys.stderr)
        return 2
    if args.boot_report is None and args.frames_dir is None:
        print("error: frames_dir or --boot-report is required", file=sys.stderr)
        return 2

    if args.boot_report is not None:
        try:
            _validate_window_ms(args.window_ms)
            raw_text = args.boot_report.read_text(encoding="utf-8")
            report = json.loads(raw_text)
        except FileNotFoundError:
            print("error: boot report file does not exist", file=sys.stderr)
            return 2
        except (OSError, UnicodeDecodeError):
            print("error: unable to read boot report file", file=sys.stderr)
            return 2
        except json.JSONDecodeError:
            print("error: boot report file is not valid JSON", file=sys.stderr)
            return 2
        except ValueError as exc:
            print(f"error: {sanitize_text(str(exc))}", file=sys.stderr)
            return 2
        if not isinstance(report, dict):
            print("error: boot report JSON must be an object", file=sys.stderr)
            return 2
        phases = extract_boot_phases_from_report(report, window_ms=args.window_ms)
        if args.json:
            print(json.dumps(sanitize_json_payload(phases), indent=2))
        else:
            print("\n".join(format_phase_summary(phases)))
        return 0

    try:
        summary = analyze_frames_dir(
            args.frames_dir,
            interval_sec=args.interval_sec,
            black_luma=args.black_luma,
            diff_threshold=args.diff_threshold,
        )
    except FileNotFoundError:
        print("error: frames directory does not exist", file=sys.stderr)
        return 2
    except NotADirectoryError:
        print("error: frames path is not a directory", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {sanitize_text(str(exc))}", file=sys.stderr)
        return 2
    except OSError:
        print("error: unable to read frames directory", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(sanitize_json_payload(summary), indent=2))
    else:
        print("\n".join(format_capture_summary(summary)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
