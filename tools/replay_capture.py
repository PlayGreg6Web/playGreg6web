#!/usr/bin/env python3
"""Replay a native .pg6trace capture in headless Chrome and read back pixels.

Steps: (optional) trim large WriteTexture uploads, serve web/ plus the replay dir, drive headless
Chrome over CDP, save thumbnails of the swapchain at requested frames and print per-frame
non-black percentage / distinct colour counts. Chrome is always closed on exit.

Example:
  python tools/replay_capture.py --chrome "C:/Program Files/Google/Chrome/Application/chrome.exe" \\
      --replay-dir work/replay --trace capture.pg6trace --pack shaders.wgsl.pack \\
      --thumbat 0,100,500 --out work/replay/out --trim-every 50 --trim-largest 3

--replay-dir holds the trace and the wgsl pack (served at /data/). With --trim-every N the
--trim-largest textures (by total uploaded bytes) keep only the uploads in frames where
frame % N == 0, and the trimmed copy <trace>.trimmed is replayed instead. An upload is dropped only
when a later upload of the same texture and mip rewrites its whole region within a few frames, so
incremental sub-rect writes (a glyph atlas filled band by band, #691) survive the trim.
Writes <out>-swap<frame>.png thumbnails and <out>.log.txt.
"""
import argparse
import base64
import http.server
import io
import json
import math
import mmap
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import zlib
from collections import defaultdict
from pathlib import Path

OP_WRITE_TEXTURE = 13
OP_WRITE_BUFFER = 4
OP_PRESENT = 7
THIN_MARKER = 0x6E696874  # 'thin': reserved field of a WriteBuffer/WriteTexture whose payload was thinned out
# opcode -> (fixed payload size, offset of data_len); the reserved u32 is the last field.
_UPLOAD_LAYOUT = {OP_WRITE_BUFFER: (16, 8), OP_WRITE_TEXTURE: (48, 40)}
sys.path.insert(0, str(Path(__file__).resolve().parent))
import find_chrome  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
REPLAY_PAGE = REPO_ROOT / "tools" / "replay" / "replay.html"
FPS_TARGET = 30.0
FRAME_BUDGET_MS_30FPS = 1000.0 / FPS_TARGET
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_BPP = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
_REDACTED_PATH = "<redacted-path>"
_HOST_PATH_PATTERNS = (
    re.compile(r"file://[^\s\"'<>()]+", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:\\(?:[^\\:\"'<>(),;\r\n]+\\)*[^\s\"'<>(),;\r\n]+"),
    re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:/(?:[^/:\"'<>(),;\r\n]+/)*[^\s\"'<>(),;\r\n]+"),
    re.compile(r"\\\\[^\s\"'<>()\\]+\\[^\s\"'<>()]+"),
    re.compile(r"(?<![A-Za-z0-9_.:/])/(?:[A-Za-z0-9._@-]+/)+[A-Za-z0-9._@-]*"),
    re.compile(r"(?<![A-Za-z0-9_.:/])/(?:home|Users|tmp|var|etc|opt|usr|mnt|private|root|Volumes)(?:/[^\s\"'<>()]+)?"),
)


def parse_frames(text):
    """'0,100,5' -> [0, 100, 5]; empty -> []."""
    out = []
    for part in text.split(","):
        part = part.strip()
        if part:
            n = int(part)
            if n < 0:
                raise ValueError(f"negative frame {n}")
            out.append(n)
    return out


def iter_packets(data):
    """Yield (offset, opcode, payload_len) for each packet (<HHI header, then payload)."""
    off, n = 0, len(data)
    while off + 8 <= n:
        op, _flags, ln = struct.unpack_from("<HHI", data, off)
        yield off, op, ln
        off += 8 + ln


def is_thin_marker(data, off, op, ln):
    """True for a thinned upload marker (data_len 0, reserved == 'thin'): the replay must skip the
    write and keep the destination's previous contents."""
    layout = _UPLOAD_LAYOUT.get(op)
    if layout is None or ln != layout[0]:
        return False
    fixed, len_off = layout
    return (struct.unpack_from("<I", data, off + 8 + len_off)[0] == 0
            and struct.unpack_from("<I", data, off + 8 + fixed - 4)[0] == THIN_MARKER)


def strip_thin_markers(data):
    """Return `data` without thinned upload markers (skipping them == keeping the previous contents)."""
    out = bytearray()
    for off, op, ln in iter_packets(data):
        if not is_thin_marker(data, off, op, ln):
            out += data[off:off + 8 + ln]
    return bytes(out)


def largest_textures(data, count):
    """Texture ids with the most WriteTexture bytes, largest first."""
    sizes = defaultdict(int)
    for off, op, ln in iter_packets(data):
        if op == OP_WRITE_TEXTURE and ln >= 4 and not is_thin_marker(data, off, op, ln):
            sizes[struct.unpack_from("<I", data, off + 8)[0]] += ln
    return [t for t, _ in sorted(sizes.items(), key=lambda kv: -kv[1])[:count]]


# A write is dropped only when its rewrite follows within this many presents: otherwise the replay
# shows the stale texture until then (#691: an atlas band written at frame 819 and rewritten at
# 1084 was dropped, so the UI TEXT SIZE screen drew glyphs that were never uploaded).
SUPERSEDE_HORIZON_FRAMES = 4


def superseded_writes(data, ids, horizon=SUPERSEDE_HORIZON_FRAMES):
    """Offsets of the WriteTexture packets for `ids` whose region (origin + extent) a later
    WriteTexture of the same texture and mip lies around completely, at most `horizon` presents
    later. Only those are safe to drop: any other write (a sub-rect update nothing rewrites soon,
    e.g. new glyphs in an atlas, #691) would leave the texture stale in the replay."""
    ids = set(ids)
    writes = []
    frame = 0
    for off, op, ln in iter_packets(data):
        if op == OP_WRITE_TEXTURE and ln >= 48 and not is_thin_marker(data, off, op, ln):
            tex, mip, ox, oy, oz, w, h, d = struct.unpack_from("<8I", data, off + 8)
            if tex in ids:
                writes.append((off, frame, (tex, mip), (ox, oy, oz, ox + w, oy + h, oz + d)))
        elif op == OP_PRESENT:
            frame += 1
    later = defaultdict(dict)  # (texture, mip) -> region -> earliest frame it is written after the current packet
    out = set()
    for off, frame, key, r in reversed(writes):
        seen = later[key]
        if any(o[0] <= r[0] and o[1] <= r[1] and o[2] <= r[2] and o[3] >= r[3] and o[4] >= r[4] and o[5] >= r[5]
               and f - frame <= horizon for o, f in seen.items()):
            out.add(off)
        seen[r] = frame
    return out


def trim_trace(data, every, ids):
    """Drop WriteTexture packets for `ids` except in frames with frame % every == 0, and only those
    a later write of the same region supersedes (superseded_writes).
    Returns (trimmed_bytes, frames, dropped_bytes)."""
    droppable = superseded_writes(data, ids)
    ids = set(ids)
    out = bytearray()
    frame = dropped = 0
    for off, op, ln in iter_packets(data):
        end = off + 8 + ln
        if is_thin_marker(data, off, op, ln):
            pass  # marker: nothing to drop, the replay skips it
        elif (op == OP_WRITE_TEXTURE and ln >= 4 and frame % every != 0 and (ln < 48 or off in droppable)
                and struct.unpack_from("<I", data, off + 8)[0] in ids):
            dropped += ln
            end = None
        if end is not None:
            out += data[off:end]
        if op == OP_PRESENT:
            frame += 1
    return bytes(out), frame, dropped


_LOG_RX = re.compile(r"swapchain frame (\d+) \((\d+)x(\d+) (\S+)\): non-black (\d+)/(\d+) \(([\d.]+)%\), distinct (\d+)")
_RESULT_RX = re.compile(r"RESULT:\s*replayed\s+(\d+)\s+frame\(s\)(?:,\s*(\d+)\s+validation error\(s\))?")
_FRAME_MS_RX = re.compile(r"(?:present_ms|frame_ms|dt_ms)\s*[=:]\s*([\d.]+)", re.IGNORECASE)


AUTO_TRIM_MAX_BYTES = 1536 * 1048576  # Chrome fails to fetch() traces much beyond ~2 GB
AUTO_TRIM_EVERY = 100
AUTO_TRIM_LARGEST = 12


def resolve_trim(trace_size, trim_every, trim_largest, max_trace_mb=1536):
    """Decide the trim parameters. An explicit --trim-every wins; otherwise traces larger than
    max_trace_mb (0 = never) are auto-trimmed with AUTO_TRIM_EVERY / max(trim_largest, AUTO_TRIM_LARGEST).
    Returns (every, largest, auto)."""
    if trim_every > 0:
        return trim_every, trim_largest, False
    if max_trace_mb > 0 and trace_size > max_trace_mb * 1048576:
        return AUTO_TRIM_EVERY, max(trim_largest, AUTO_TRIM_LARGEST), True
    return 0, trim_largest, False


def parse_replay_log(text):
    """Extract per-frame stats from the page log lines."""
    return [{"frame": int(m[1]), "width": int(m[2]), "height": int(m[3]), "format": m[4],
             "non_black_pct": float(m[7]), "distinct": int(m[8])} for m in _LOG_RX.finditer(text)]


def parse_replay_result(text):
    """Extract replay status, replayed frame count, and validation error count from log text."""
    text = text or ""
    m = _RESULT_RX.search(text)
    if m:
        return {
            "ok": True,
            "replayed_frames": int(m.group(1)),
            "validation_errors": int(m.group(2)) if m.group(2) is not None else 0,
            "result_line": m.group(0),
        }
    result_line = next((ln for ln in text.splitlines() if ln.startswith("RESULT")), "")
    return {
        "ok": "RESULT: replayed" in text,
        "replayed_frames": 0,
        "validation_errors": 0,
        "result_line": result_line,
    }


def parse_frame_times_from_log(text):
    """Extract optional per-frame durations in ms (e.g. `present_ms=16.5`) from log text."""
    return [float(m.group(1)) for m in _FRAME_MS_RX.finditer(text or "")]


def sanitize_filename(name):
    """Return only the final basename component, stripping any POSIX or Windows directory prefix."""
    text = str(name or "").strip()
    if not text:
        return ""
    parts = [p for p in re.split(r"[/\\]+", text) if p]
    return parts[-1] if parts else ""


def sanitize_text(text):
    """Scrub host filesystem paths (Windows, UNC, POSIX, file://) from arbitrary text."""
    out = str(text)
    for rx in _HOST_PATH_PATTERNS:
        out = rx.sub(_REDACTED_PATH, out)
    return out


def sanitize_json_value(value):
    """Recursively sanitize all string values in a JSON-serializable structure."""
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, dict):
        return {str(k): sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json_value(v) for v in value]
    return value


def decode_data_url_png(data_url):
    """Decode a `data:image/png;base64,...` data URL into raw PNG bytes."""
    if not isinstance(data_url, str) or "," not in data_url:
        raise ValueError("invalid PNG data URL")
    return base64.b64decode(data_url.split(",", 1)[1])


def _paeth_predictor(a, b, c):
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def decode_png_rgb_pure_python(png_bytes):
    """Pure-Python 8-bit non-interlaced PNG decoder returning (width, height, rgb_bytes)."""
    data = bytes(png_bytes)
    if len(data) < 8 or data[:8] != _PNG_SIGNATURE:
        raise ValueError("not a PNG image")
    pos = 8
    ihdr = None
    palette = None
    idat_parts = []
    while pos + 8 <= len(data):
        length, chunk_type = struct.unpack_from(">I4s", data, pos)
        body_start = pos + 8
        body_end = body_start + length
        if body_end + 4 > len(data):
            raise ValueError("truncated PNG chunk")
        body = data[body_start:body_end]
        if chunk_type == b"IHDR":
            if len(body) < 13:
                raise ValueError("invalid IHDR chunk")
            ihdr = struct.unpack(">IIBBBBB", body[:13])
        elif chunk_type == b"PLTE":
            palette = body
        elif chunk_type == b"IDAT":
            idat_parts.append(body)
        elif chunk_type == b"IEND":
            break
        pos = body_end + 4
    if ihdr is None:
        raise ValueError("missing IHDR chunk")
    width, height, bit_depth, color_type, compression, filter_method, interlace = ihdr
    if width <= 0 or height <= 0:
        raise ValueError("invalid PNG dimensions")
    if bit_depth != 8 or compression != 0 or filter_method != 0 or interlace != 0:
        raise ValueError(
            f"unsupported PNG format (bit_depth={bit_depth}, interlace={interlace}, "
            f"compression={compression}, filter={filter_method})"
        )
    bpp = _PNG_BPP.get(color_type)
    if bpp is None:
        raise ValueError(f"unsupported PNG color type {color_type}")
    if color_type == 3 and palette is None:
        raise ValueError("indexed PNG missing PLTE chunk")
    raw = zlib.decompress(b"".join(idat_parts))
    stride = width * bpp
    expected = height * (1 + stride)
    if len(raw) < expected:
        raise ValueError("truncated PNG scanline data")
    prev = bytearray(stride)
    rgb_out = bytearray()
    for y in range(height):
        row_off = y * (1 + stride)
        ftype = raw[row_off]
        src = raw[row_off + 1:row_off + 1 + stride]
        cur = bytearray(stride)
        if ftype == 0:
            cur[:] = src
        elif ftype == 1:
            for i in range(stride):
                a = cur[i - bpp] if i >= bpp else 0
                cur[i] = (src[i] + a) & 0xFF
        elif ftype == 2:
            for i in range(stride):
                cur[i] = (src[i] + prev[i]) & 0xFF
        elif ftype == 3:
            for i in range(stride):
                a = cur[i - bpp] if i >= bpp else 0
                cur[i] = (src[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:
            for i in range(stride):
                a = cur[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                cur[i] = (src[i] + _paeth_predictor(a, b, c)) & 0xFF
        else:
            raise ValueError(f"unsupported PNG filter type {ftype}")
        if color_type == 2:
            rgb_out += cur
        elif color_type == 6:
            for i in range(0, stride, 4):
                rgb_out.append(cur[i])
                rgb_out.append(cur[i + 1])
                rgb_out.append(cur[i + 2])
        elif color_type == 0:
            for v in cur:
                rgb_out.extend((v, v, v))
        elif color_type == 4:
            for i in range(0, stride, 2):
                v = cur[i]
                rgb_out.extend((v, v, v))
        else:  # color_type == 3 (palette)
            for idx in cur:
                poff = idx * 3
                if poff + 3 > len(palette):
                    raise ValueError("PNG palette index out of range")
                rgb_out += palette[poff:poff + 3]
        prev = cur
    return width, height, bytes(rgb_out)


def decode_png_rgb(png_bytes, *, use_pillow=True):
    """Decode PNG bytes into (width, height, rgb_bytes, decoder_name).

    Uses Pillow when installed and `use_pillow=True`; falls back to pure-Python 8-bit PNG decoding.
    """
    if use_pillow:
        try:
            from PIL import Image  # type: ignore[import-not-found]
        except ImportError:
            Image = None
        if Image is not None:
            with Image.open(io.BytesIO(png_bytes)) as img:
                rgb_img = img.convert("RGB")
                return rgb_img.width, rgb_img.height, rgb_img.tobytes(), "pillow"
    width, height, rgb = decode_png_rgb_pure_python(png_bytes)
    return width, height, rgb, "pure_python"


def _compute_rgb_luminance(rgb_bytes, width, height, *, luma_threshold=0.0):
    if luma_threshold < 0.0:
        raise ValueError(f"luma_threshold must be >= 0, got {luma_threshold}")
    total_pixels = width * height
    if total_pixels == 0 or not any(rgb_bytes):
        return {
            "width": width,
            "height": height,
            "total_pixels": total_pixels,
            "non_black_pixels": 0,
            "non_black_pct": 0.0,
            "max_luma": 0.0,
            "mean_luma": 0.0,
            "is_non_black": False,
        }
    non_black = 0
    max_luma = 0.0
    sum_luma = 0.0
    for r, g, b in zip(rgb_bytes[0::3], rgb_bytes[1::3], rgb_bytes[2::3]):
        luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
        sum_luma += luma
        if luma > max_luma:
            max_luma = luma
        if luma > luma_threshold:
            non_black += 1
    return {
        "width": width,
        "height": height,
        "total_pixels": total_pixels,
        "non_black_pixels": non_black,
        "non_black_pct": round(100.0 * non_black / total_pixels, 2),
        "max_luma": round(max_luma, 3),
        "mean_luma": round(sum_luma / total_pixels, 3),
        "is_non_black": non_black > 0,
    }


def _coerce_png_bytes(item):
    if isinstance(item, (bytes, bytearray, memoryview)):
        return bytes(item)
    if isinstance(item, Path):
        return item.read_bytes()
    if isinstance(item, str):
        if item.startswith("data:"):
            return decode_data_url_png(item)
        return Path(item).read_bytes()
    raise TypeError(f"unsupported PNG source type: {type(item).__name__}")


def analyze_png_luminance(png_source, *, luma_threshold=0.0, use_pillow=True):
    """Compute Rec.709 luminance statistics for a single PNG image (bytes, data URL, or Path)."""
    png_bytes = _coerce_png_bytes(png_source)
    width, height, rgb_bytes, decoder = decode_png_rgb(png_bytes, use_pillow=use_pillow)
    stats = _compute_rgb_luminance(rgb_bytes, width, height, luma_threshold=luma_threshold)
    stats["decoder"] = decoder
    return stats


def verify_non_black_captures(frames, *, luma_threshold=0.0, use_pillow=True):
    """Verify pixel luminance across captured PNG frames and detect all-black regressions.

    `frames` may be a mapping `{label: png_source}` or a sequence of PNG sources (bytes, data URL,
    or Path). Returns a sanitized report containing `non_black_frames`, `black_frames`, `max_luma`,
    and `all_black_regression`.
    """
    if luma_threshold < 0.0:
        raise ValueError(f"luma_threshold must be >= 0, got {luma_threshold}")
    if isinstance(frames, dict):
        items = [(sanitize_filename(k) or str(k), v) for k, v in frames.items()]
    else:
        items = []
        for idx, item in enumerate(frames or ()):
            if isinstance(item, Path):
                label = sanitize_filename(item.name) or f"frame_{idx}"
            elif isinstance(item, str) and not item.startswith("data:"):
                label = sanitize_filename(item) or f"frame_{idx}"
            else:
                label = f"frame_{idx}"
            items.append((label, item))
    per_frame = []
    non_black_frames = 0
    black_frames = 0
    max_luma = 0.0
    for label, source in items:
        stats = analyze_png_luminance(source, luma_threshold=luma_threshold, use_pillow=use_pillow)
        stats["label"] = sanitize_text(label)
        per_frame.append(stats)
        if stats["is_non_black"]:
            non_black_frames += 1
        else:
            black_frames += 1
        if stats["max_luma"] > max_luma:
            max_luma = stats["max_luma"]
    total_frames = len(per_frame)
    all_black_regression = (total_frames == 0) or (non_black_frames == 0)
    return {
        "total_frames": total_frames,
        "non_black_frames": non_black_frames,
        "black_frames": black_frames,
        "max_luma": round(max_luma, 3),
        "luma_threshold": round(float(luma_threshold), 3),
        "all_black_regression": all_black_regression,
        "frames": per_frame,
    }


def summarize_pacing(frame_times_ms=None, *, replayed_frames=0, elapsed_seconds=0.0, target_fps=FPS_TARGET):
    """Compute 30 FPS replay pacing summary (`present_fps_avg`, `frames_within_30fps_budget`, etc.)."""
    if target_fps <= 0.0:
        raise ValueError(f"target_fps must be > 0, got {target_fps}")
    budget_ms = 1000.0 / float(target_fps)
    times = [float(t) for t in (frame_times_ms or ())]
    if any(t < 0.0 for t in times):
        raise ValueError("frame_times_ms values must be >= 0")
    if times:
        count = len(times)
        total_ms = sum(times)
        avg_ms = total_ms / count
        present_fps_avg = (1000.0 / avg_ms) if avg_ms > 0.0 else 0.0
        within = sum(1 for dt in times if dt <= budget_ms + 1e-9)
        over = count - within
        ordered = sorted(times)
        p95_idx = min(count - 1, max(0, int(math.ceil(0.95 * count)) - 1))
        min_ms, max_ms, p95_ms = ordered[0], ordered[-1], ordered[p95_idx]
        timing_source = "per_frame"
    elif replayed_frames > 0 and elapsed_seconds > 0.0:
        count = int(replayed_frames)
        total_ms = float(elapsed_seconds) * 1000.0
        avg_ms = total_ms / count
        present_fps_avg = count / float(elapsed_seconds)
        within = count if avg_ms <= budget_ms + 1e-9 else 0
        over = count - within
        min_ms = max_ms = p95_ms = avg_ms
        timing_source = "wall_clock"
    else:
        count = max(0, int(replayed_frames))
        total_ms = max(0.0, float(elapsed_seconds) * 1000.0)
        avg_ms = min_ms = max_ms = p95_ms = 0.0
        present_fps_avg = 0.0
        within = 0
        over = count
        timing_source = "none"
    return {
        "target_fps": round(float(target_fps), 3),
        "budget_ms": round(budget_ms, 3),
        "frame_count": count,
        "elapsed_ms": round(total_ms, 3),
        "avg_frame_ms": round(avg_ms, 3),
        "min_frame_ms": round(min_ms, 3),
        "max_frame_ms": round(max_ms, 3),
        "p95_frame_ms": round(p95_ms, 3),
        "present_fps_avg": round(present_fps_avg, 3),
        "frames_within_30fps_budget": within,
        "frames_over_30fps_budget": over,
        "within_30fps_budget_pct": round(100.0 * within / count, 2) if count > 0 else 0.0,
        "meets_30fps_target": (present_fps_avg >= float(target_fps) - 1e-9) if count > 0 else False,
        "timing_source": timing_source,
    }


def build_replay_summary(
    *,
    trace_name,
    pack_name,
    log_text,
    thumbs,
    elapsed_seconds,
    frame_times_ms=None,
    verify_non_black=False,
    luma_threshold=0.0,
    trim_info=None,
    error=None,
    use_pillow=True,
):
    """Build a strictly sanitized JSON-serializable summary (zero host paths)."""
    result = parse_replay_result(log_text)
    log_frames = parse_replay_log(log_text)
    effective_times = list(frame_times_ms) if frame_times_ms else parse_frame_times_from_log(log_text)
    replayed_frames = result["replayed_frames"] or len(log_frames)
    pacing = summarize_pacing(
        effective_times,
        replayed_frames=replayed_frames,
        elapsed_seconds=elapsed_seconds,
    )
    luminance = verify_non_black_captures(
        thumbs or {},
        luma_threshold=luma_threshold,
        use_pillow=use_pillow,
    )
    replay_ok = result["ok"] and error is None
    regression_failed = bool(verify_non_black and luminance["all_black_regression"])
    ok = replay_ok and not regression_failed
    summary = {
        "schema_version": 1,
        "ok": ok,
        "trace": sanitize_filename(trace_name),
        "pack": sanitize_filename(pack_name),
        "replayed_frames": replayed_frames,
        "validation_errors": result["validation_errors"],
        "frames": log_frames,
        "present_fps_avg": pacing["present_fps_avg"],
        "frames_within_30fps_budget": pacing["frames_within_30fps_budget"],
        "pacing": pacing,
        "verify_non_black": bool(verify_non_black),
        "non_black_frames": luminance["non_black_frames"],
        "black_frames": luminance["black_frames"],
        "max_luma": luminance["max_luma"],
        "all_black_regression": luminance["all_black_regression"],
        "luminance": luminance,
        "trim": trim_info,
        "error": sanitize_text(error) if error else None,
    }
    return sanitize_json_value(summary)


def make_handler(web_root, data_root):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def translate_path(self, path):
            path = path.split("?", 1)[0].split("#", 1)[0]
            if path.startswith("/data/"):
                return os.path.join(data_root, path[len("/data/"):])
            if path == "/replay.html":
                return str(REPLAY_PAGE)
            return os.path.join(web_root, path.lstrip("/"))

        def log_message(self, *args):
            pass
    return Handler


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Cdp:
    """Minimal stdlib websocket client for the Chrome DevTools protocol."""

    def __init__(self, ws_url):
        host, rest = ws_url[len("ws://"):].split("/", 1)
        h, p = host.split(":")
        self.sock = socket.create_connection((h, int(p)), timeout=120)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET /{rest} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\n"
                           f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                           "Sec-WebSocket-Version: 13\r\n\r\n").encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += self.sock.recv(1)
        self.next_id = 0

    def _recv(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("CDP socket closed")
            buf += chunk
        return buf

    def _read_message(self):
        payload = b""
        while True:
            b0, b1 = self._recv(2)
            ln = b1 & 0x7F
            if ln == 126:
                ln = struct.unpack(">H", self._recv(2))[0]
            elif ln == 127:
                ln = struct.unpack(">Q", self._recv(8))[0]
            payload += self._recv(ln)
            if b0 & 0x80:
                return payload

    def call(self, method, **params):
        self.next_id += 1
        data = json.dumps({"id": self.next_id, "method": method, "params": params}).encode()
        mask = os.urandom(4)
        n = len(data)
        if n < 126:
            head = bytes([0x81, 0x80 | n])
        elif n < 65536:
            head = bytes([0x81, 0x80 | 126]) + struct.pack(">H", n)
        else:
            head = bytes([0x81, 0x80 | 127]) + struct.pack(">Q", n)
        self.sock.sendall(head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))
        while True:
            msg = json.loads(self._read_message())
            if msg.get("id") == self.next_id:
                return msg.get("result", {})

    def evaluate(self, expr):
        return self.call("Runtime.evaluate", expression=expr, returnByValue=True).get("result", {}).get("value")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chrome", default=None, help="path to chrome executable (default $PLAYGTA6_CHROME, then auto-detected)")
    p.add_argument("--replay-dir", required=True, help="dir with the trace and wgsl pack (served at /data/)")
    p.add_argument("--trace", required=True, help="trace file name inside --replay-dir")
    p.add_argument("--pack", required=True, help="wgsl pack file name inside --replay-dir")
    p.add_argument("--out", required=True, help="output prefix for PNG thumbnails and log")
    p.add_argument("--thumbat", default=[0], type=parse_frames, help="comma list of frames to read back (default 0)")
    p.add_argument("--frames", type=int, default=None, help="last frame to replay (default: max of --thumbat)")
    p.add_argument("--timeout", type=int, default=900, help="seconds to wait for the replay (default 900)")
    p.add_argument("--trim-every", type=int, default=0, help="keep every Nth upload of the largest textures (0 = no trim)")
    p.add_argument("--trim-largest", type=int, default=3, help="how many largest textures to trim (default 3)")
    p.add_argument("--max-trace-mb", type=int, default=1536,
                   help="auto-trim (every 100th upload, 12 largest textures) when the trace is bigger than this, "
                        "since Chrome cannot fetch ~2 GB traces; 0 disables (default 1536)")
    p.add_argument("--web-root", default=str(REPO_ROOT / "web"), help="web/ dir to serve (default repo web/)")
    p.add_argument("--canvas", default="", help="canvas format override, e.g. rgba8unorm")
    p.add_argument("--verify-non-black", action="store_true",
                   help="check thumbnail pixel luminance and exit non-zero if all captured frames are black")
    p.add_argument("--luma-threshold", type=float, default=0.0,
                   help="Rec.709 luminance threshold (0..255) above which a pixel counts as non-black (default 0.0)")
    p.add_argument("--json", action="store_true",
                   help="emit strictly sanitized JSON summary (zero host paths) to stdout")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.chrome = find_chrome.resolve_chrome(args.chrome, log=lambda m: print(m, file=sys.stderr))
    if not args.chrome:
        raise SystemExit("no Chrome found: pass --chrome or set $PLAYGTA6_CHROME")
    if args.luma_threshold < 0.0:
        raise ValueError(f"--luma-threshold must be >= 0, got {args.luma_threshold}")
    replay_dir = Path(args.replay_dir).resolve()
    trace = args.trace
    trim_info = None
    trim_every, trim_largest, auto = resolve_trim((replay_dir / trace).stat().st_size, args.trim_every,
                                                  args.trim_largest, args.max_trace_mb)
    if auto:
        print(f"trace is {(replay_dir / trace).stat().st_size // 1048576} MiB (> {args.max_trace_mb} MiB): "
              f"auto-trimming with --trim-every {trim_every} --trim-largest {trim_largest}", file=sys.stderr)
    if trim_every > 0:
        with open(replay_dir / trace, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as d:
            ids = largest_textures(d, trim_largest)
            data, frames, dropped = trim_trace(d, trim_every, ids)
        trace += ".trimmed"
        (replay_dir / trace).write_bytes(data)
        trim_info = {
            "auto": auto,
            "trimmed_texture_ids": ids,
            "frames": frames,
            "dropped_bytes": dropped,
            "kept_bytes": len(data),
        }
        if not args.json:
            print(f"trimmed textures {ids}: frames {frames}, dropped {dropped // 1048576} MiB, kept {len(data) // 1048576} MiB")
    last = args.frames if args.frames is not None else (max(args.thumbat) if args.thumbat else 0)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), make_handler(args.web_root, str(replay_dir)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port, profile = free_port(), tempfile.mkdtemp(prefix="replay-chrome-")
    chrome = subprocess.Popen([
        args.chrome, "--headless=new", "--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPU",
        "--ignore-gpu-blocklist", f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
        "--no-first-run", "--window-size=1000,1300", "--js-flags=--max-old-space-size=8192", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cdp = None
    try:
        target = None
        for _ in range(100):
            time.sleep(0.3)
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=2))
                target = next((t for t in tabs if t["type"] == "page"), None)
            except OSError:
                continue
            if target:
                break
        if not target:
            if args.json:
                summary = build_replay_summary(
                    trace_name=trace,
                    pack_name=args.pack,
                    log_text="",
                    thumbs={},
                    elapsed_seconds=0.0,
                    verify_non_black=args.verify_non_black,
                    luma_threshold=args.luma_threshold,
                    trim_info=trim_info,
                    error="no CDP target",
                )
                print(json.dumps(summary, indent=2, sort_keys=True))
            else:
                print("no CDP target", file=sys.stderr)
            return 1
        cdp = Cdp(target["webSocketDebuggerUrl"])
        cdp.call("Runtime.enable")
        cdp.call("Page.enable")
        url = (f"http://127.0.0.1:{server.server_address[1]}/replay.html?trace={trace}&pack={args.pack}"
               f"&frames={last}&thumbat={','.join(map(str, args.thumbat))}&canvas={args.canvas}")
        t_start = time.monotonic()
        cdp.call("Page.navigate", url=url)
        text = ""
        deadline = time.time() + args.timeout
        while time.time() < deadline:
            time.sleep(1)
            text = cdp.evaluate("document.getElementById('log') ? document.getElementById('log').textContent : ''") or ""
            if "RESULT:" in text:
                break
        elapsed_seconds = max(0.0, time.monotonic() - t_start)
        thumbs = json.loads(cdp.evaluate("JSON.stringify(window.__thumbs || {})") or "{}")
        raw_frame_times = cdp.evaluate("JSON.stringify(window.__frameTimesMs || [])")
        frame_times_ms = json.loads(raw_frame_times) if isinstance(raw_frame_times, str) and raw_frame_times else []
        decoded_thumbs = {}
        for name, data_url in thumbs.items():
            png_bytes = decode_data_url_png(data_url)
            decoded_thumbs[name] = png_bytes
            Path(f"{args.out}-{name}.png").write_bytes(png_bytes)
        Path(args.out + ".log.txt").write_text(text, encoding="utf-8")
        summary = build_replay_summary(
            trace_name=trace,
            pack_name=args.pack,
            log_text=text,
            thumbs=decoded_thumbs,
            elapsed_seconds=elapsed_seconds,
            frame_times_ms=frame_times_ms,
            verify_non_black=args.verify_non_black,
            luma_threshold=args.luma_threshold,
            trim_info=trim_info,
        )
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            for s in summary["frames"]:
                print(f"frame {s['frame']}: {s['width']}x{s['height']} {s['format']} "
                      f"non-black {s['non_black_pct']:.2f}% distinct {s['distinct']}")
            print("\n".join(l for l in text.splitlines() if l.startswith(("RESULT", "VALIDATION"))))
            print(f"PACING: present_fps_avg={summary['present_fps_avg']:.3f} "
                  f"frames_within_30fps_budget={summary['frames_within_30fps_budget']}/{summary['pacing']['frame_count']}")
            if args.verify_non_black:
                print(f"VERIFY_NON_BLACK: non_black_frames={summary['non_black_frames']} "
                      f"black_frames={summary['black_frames']} max_luma={summary['max_luma']:.3f} "
                      f"all_black_regression={summary['all_black_regression']}")
        return 0 if summary["ok"] else 1
    finally:
        if cdp is not None and getattr(cdp, "sock", None) is not None:
            cdp.sock.close()
        chrome.kill()
        chrome.wait()
        server.shutdown()
        server.server_close()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
