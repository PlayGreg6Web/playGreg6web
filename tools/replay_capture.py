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
frame % N == 0, and the trimmed copy <trace>.trimmed is replayed instead.
Writes <out>-swap<frame>.png thumbnails and <out>.log.txt.
"""
import argparse
import base64
import http.server
import json
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
from collections import defaultdict
from pathlib import Path

OP_WRITE_TEXTURE = 13
OP_PRESENT = 7
REPO_ROOT = Path(__file__).resolve().parent.parent
REPLAY_PAGE = REPO_ROOT / "tools" / "replay" / "replay.html"


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


def largest_textures(data, count):
    """Texture ids with the most WriteTexture bytes, largest first."""
    sizes = defaultdict(int)
    for off, op, ln in iter_packets(data):
        if op == OP_WRITE_TEXTURE and ln >= 4:
            sizes[struct.unpack_from("<I", data, off + 8)[0]] += ln
    return [t for t, _ in sorted(sizes.items(), key=lambda kv: -kv[1])[:count]]


def trim_trace(data, every, ids):
    """Drop WriteTexture packets for `ids` except in frames with frame % every == 0.
    Returns (trimmed_bytes, frames, dropped_bytes)."""
    ids = set(ids)
    out = bytearray()
    frame = dropped = 0
    for off, op, ln in iter_packets(data):
        end = off + 8 + ln
        if (op == OP_WRITE_TEXTURE and ln >= 4 and frame % every != 0
                and struct.unpack_from("<I", data, off + 8)[0] in ids):
            dropped += ln
        else:
            out += data[off:end]
        if op == OP_PRESENT:
            frame += 1
    return bytes(out), frame, dropped


_LOG_RX = re.compile(r"swapchain frame (\d+) \((\d+)x(\d+) (\S+)\): non-black (\d+)/(\d+) \(([\d.]+)%\), distinct (\d+)")


def parse_replay_log(text):
    """Extract per-frame stats from the page log lines."""
    return [{"frame": int(m[1]), "width": int(m[2]), "height": int(m[3]), "format": m[4],
             "non_black_pct": float(m[7]), "distinct": int(m[8])} for m in _LOG_RX.finditer(text)]


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
    p.add_argument("--chrome", required=True, help="path to chrome executable")
    p.add_argument("--replay-dir", required=True, help="dir with the trace and wgsl pack (served at /data/)")
    p.add_argument("--trace", required=True, help="trace file name inside --replay-dir")
    p.add_argument("--pack", required=True, help="wgsl pack file name inside --replay-dir")
    p.add_argument("--out", required=True, help="output prefix for PNG thumbnails and log")
    p.add_argument("--thumbat", default=[0], type=parse_frames, help="comma list of frames to read back (default 0)")
    p.add_argument("--frames", type=int, default=None, help="last frame to replay (default: max of --thumbat)")
    p.add_argument("--timeout", type=int, default=900, help="seconds to wait for the replay (default 900)")
    p.add_argument("--trim-every", type=int, default=0, help="keep every Nth upload of the largest textures (0 = no trim)")
    p.add_argument("--trim-largest", type=int, default=3, help="how many largest textures to trim (default 3)")
    p.add_argument("--web-root", default=str(REPO_ROOT / "web"), help="web/ dir to serve (default repo web/)")
    p.add_argument("--canvas", default="", help="canvas format override, e.g. rgba8unorm")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    replay_dir = Path(args.replay_dir).resolve()
    trace = args.trace
    if args.trim_every > 0:
        with open(replay_dir / trace, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as d:
            ids = largest_textures(d, args.trim_largest)
            data, frames, dropped = trim_trace(d, args.trim_every, ids)
        trace += ".trimmed"
        (replay_dir / trace).write_bytes(data)
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
            print("no CDP target", file=sys.stderr)
            return 1
        cdp = Cdp(target["webSocketDebuggerUrl"])
        cdp.call("Runtime.enable")
        cdp.call("Page.enable")
        url = (f"http://127.0.0.1:{server.server_address[1]}/replay.html?trace={trace}&pack={args.pack}"
               f"&frames={last}&thumbat={','.join(map(str, args.thumbat))}&canvas={args.canvas}")
        cdp.call("Page.navigate", url=url)
        text = ""
        deadline = time.time() + args.timeout
        while time.time() < deadline:
            time.sleep(1)
            text = cdp.evaluate("document.getElementById('log') ? document.getElementById('log').textContent : ''") or ""
            if "RESULT:" in text:
                break
        thumbs = json.loads(cdp.evaluate("JSON.stringify(window.__thumbs || {})") or "{}")
        for name, data_url in thumbs.items():
            Path(f"{args.out}-{name}.png").write_bytes(base64.b64decode(data_url.split(",", 1)[1]))
        Path(args.out + ".log.txt").write_text(text, encoding="utf-8")
        for s in parse_replay_log(text):
            print(f"frame {s['frame']}: {s['width']}x{s['height']} {s['format']} "
                  f"non-black {s['non_black_pct']:.2f}% distinct {s['distinct']}")
        print("\n".join(l for l in text.splitlines() if l.startswith(("RESULT", "VALIDATION"))))
        return 0 if "RESULT: replayed" in text else 1
    finally:
        chrome.kill()
        chrome.wait()
        server.shutdown()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
