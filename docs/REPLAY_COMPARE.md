# Comparing two runtime builds on one recorded capture

Two game boots are timing-dependent, so their frame digests differ even for identical code. A
`.pg6trace` capture is a fixed command stream, so replaying it through two `web/` builds proves (or
disproves) identical rendering.

```
python tools/replay_compare.py --ref-a origin/main~1 --ref-b origin/main \
    --replay-dir work/replay --trace capture.pg6trace --pack shaders.wgsl.pack \
    --frames 600 --every 10 --out work/cmp
```

- A build is a `web/` directory (`--web-a/--web-b`) or a git ref (`--ref-a/--ref-b`, exported with
  `git archive`). An omitted side is the working-tree `web/`.
- Each build is replayed with `tools/replay_capture.py` (headless Chrome, WebGPU). `tools/replay/replay.html`
  logs an exact full-resolution digest (`digest frame N: <16 hex>`, two FNV-1a lanes over every pixel) for
  each requested frame; the tool compares digests and, for the first differing frame, the 960 px
  thumbnails, and writes `<out>-swap<N>.diff.png` (grey = same, red = differs).
- Output: `IDENTICAL` (exit 0), `DIFFERENT: first differing frame N` (exit 2), or a replay failure (exit 1).
  `<out>-A-swapN.png` / `<out>-B-swapN.png` and the logs are kept.
- Use the same `--trim-every/--max-trace-mb` for both sides (they are shared by construction).

## Determinism of the replay

- Frames are executed strictly in trace order and each frame awaits `queue.onSubmittedWorkDone()` before the
  next; there is no wall-clock, random, or rAF dependence in the page.
- Not controlled (same machine, adapter and Chrome for A and B is required): GPU driver/float behaviour,
  Chrome version, canvas format (`--canvas`), and any asynchronous work inside the builds under test
  (for example async pipeline compilation or lazily created objects finishing at different times). A digest
  that flips between two runs of the *same* build points at such a path; run the same build twice first
  (`--ref-a X --ref-b X`) to establish the noise floor.
- Trimmed traces (`--trim-every`, automatic for traces over 1.5 GiB) change contents deterministically.
- The native replayer is not covered; this tool compares the web runtime only.

Unit tests: `python -m unittest tests.test_replay_compare` (no browser needed).
