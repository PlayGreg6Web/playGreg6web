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
- The native replayer is covered by `tools/native_replay_compare.py` (below).

Unit tests: `python -m unittest tests.test_replay_compare` (no browser needed).

## Native executor: `tools/native_replay_compare.py`

Same idea for the Dawn executor (`pg6_native_exec`): replay one capture with two binaries and compare the
per-frame digests (`digest frame N: <16 hex>`, headless `--digest-at`).

```
python tools/native_replay_compare.py --ref-a origin/main~1 --ref-b origin/main     --trace capture.pg6trace --pack shaders.wgsl.pack --frames 600 --every 10 --out work/ncmp
python tools/native_replay_compare.py --exe-a build/pg6_native_exec --exe-b other/pg6_native_exec     --trace capture.pg6trace --frames 100 --size 1280x720
python tools/native_replay_compare.py --exe-a build/pg6_native_exec --self-check     --trace capture.pg6trace --frames 100
```

- A side is a binary (`--exe-a/--exe-b`) or a git ref (`--ref-a/--ref-b`). A ref's `runtime/` is exported with
  `git archive`, configured with `-DPLAYGTA6_NATIVE_EXECUTOR=ON` (Release) and built for the target
  `pg6_native_exec`; the binary (plus `dxcompiler.dll`/`dxil.dll` on Windows) is cached per commit in
  `<objcache>/../native-exec/<sha16>/` (`--cache-dir` or `PLAYGTA6_NATIVE_EXEC_CACHE` override), the temporary
  source and build trees are deleted. Later runs for the same commit reuse it.
- `--pack`, `--reflection` (repeatable), `--size`, `--frames`, `--every` are passed to both sides.
- On a digest difference the tool re-runs both binaries with `--dump-frame N --dump-dir D` (the screenshot PNG
  encoder, `gpu/native_exec/screenshot_model.h`; writes `D/frame-N.png`) for the first differing frame and
  writes `<out>-A-frameN.png`, `<out>-B-frameN.png` and `<out>-frameN.diff.png` (grey = same, red = differs).
- `--self-check` replays twice with side A's binary: `DETERMINISTIC` (exit 0) or `NONDETERMINISTIC: ... first N`
  (exit 2). Run it before trusting an A-vs-B difference.
- Exit codes: 0 identical, 1 replay failure, 2 differs, 77 no WebGPU adapter. Logs: `<out>-A.log.txt`, `<out>-B.log.txt`.

Unit tests: `python -m unittest tests.test_native_replay_compare` (a fake executor script and a synthetic text
trace; no GPU, no game bytes).
