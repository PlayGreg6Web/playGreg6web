"""Screen timeline (R40 + T64): per-screen timelines for boot reports and frame captures.

R40 (boot report Present samples):
Agents scripting menus tell screens apart by draws per frame and new resources. Each Present sample
carries the cumulative `draws_issued` and `resources_created` counters (D3D12 layer); this module
turns them into per-present deltas and collapses consecutive presents with the same draw count
(+-1) and no new resources into one segment:
  {start_time, end_time, first_present, presents, draws_per_frame, new_resources_at_start,
   input_events_in_segment}
Capped at MAX_SEGMENTS (later presents are folded into the last segment). Times are wall_ms.

T64 (Chrome capture frame directory CLI):
Analyzes a directory of captured frames (e.g. `frame_0001.png` / `frame_0001.jpg`), groups
consecutive frames into distinct visual/black screen segments, computes optional dwell durations
via `--interval-sec <f>`, and outputs either a sanitized plain-text table or structured `--json`
summary without leaking host paths.
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import re
import struct
import sys
from typing import Any, Sequence
import zlib

MAX_SEGMENTS = 500
DEFAULT_BLACK_LUMA = 2.0
DEFAULT_DIFF_THRESHOLD = 4.0
THUMB_SIZE = (32, 18)

_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})
_FRAME_NUM_RE = re.compile(r"(\d+)(?!.*\d)")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


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
        else:
            start_s = None
            end_s = None
            dwell_s = None
        segments.append({
            "index": seg_idx,
            "start_frame": seg["start_frame"],
            "end_frame": seg["end_frame"],
            "frame_count": count,
            "start_s": start_s,
            "end_s": end_s,
            "dwell_s": dwell_s,
            "mean_luma": round(seg["luma_sum"] / count, 2),
            "is_black": seg["is_black"],
        })

    first_non_black_s: float | None = (
        round(first_non_black_ord * clean_interval, 3)
        if clean_interval is not None and first_non_black_ord is not None
        else None
    )

    return {
        "total_frames": total_frames,
        "unique_segments": len(segments),
        "black_frames": black_frames,
        "non_black_frames": non_black_frames,
        "first_non_black_frame": first_non_black_frame,
        "first_non_black_s": first_non_black_s,
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
    black = summary["black_frames"]
    non_black = summary["non_black_frames"]
    first_frame = summary["first_non_black_frame"]
    first_s = summary["first_non_black_s"]
    segments = summary["segments"]

    seg_word = "segment" if unique == 1 else "segments"
    frame_word = "frame" if total == 1 else "frames"
    if total == 0:
        return [f"screen timeline: 0 frames, 0 segments (black: 0, non-black: 0)"]

    if first_frame is None:
        first_desc = "none"
    elif first_s is not None:
        first_desc = f"frame {first_frame} @ {first_s:.2f}s"
    else:
        first_desc = f"frame {first_frame}"

    lines = [
        f"screen timeline: {total} {frame_word}, {unique} {seg_word} "
        f"(black: {black}, non-black: {non_black}, first_non_black: {first_desc})"
    ]
    has_times = any(s.get("dwell_s") is not None for s in segments)
    if has_times:
        lines.append("  idx  frames      count  start_s    end_s  dwell_s    luma  state")
    else:
        lines.append("  idx  frames      count    luma  state")

    for s in segments:
        span = f"{s['start_frame']}..{s['end_frame']}"
        state = "black" if s["is_black"] else "visible"
        if has_times:
            lines.append(
                f"  {s['index']:3d}  {span:10s}  {s['frame_count']:5d}  "
                f"{s['start_s']:7.2f}  {s['end_s']:7.2f}  {s['dwell_s']:7.2f}  "
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
        description="Summarize screen transitions and dwell durations from captured frames.",
    )
    parser.add_argument(
        "frames_dir",
        type=Path,
        help="Directory containing captured frame images (e.g. frame_0001.png / frame_0001.jpg).",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=None,
        help="Capture interval in seconds per frame; populates start_s, end_s, dwell_s, and first_non_black_s.",
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
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("\n".join(format_capture_summary(summary)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
