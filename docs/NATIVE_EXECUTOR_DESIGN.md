# Native WebGPU Executor

The native harness (`playgta6_boot`) produces the same GPU packet stream as the browser runtime
([gpu_commands.md](./gpu_commands.md)), but nothing native consumes it: native runs have no window,
GPU results are only visible by replaying a `.pg6trace` in Chrome (`tools/replay_capture.py`), and
things like per-pass GPU timestamps (#1608) or presenting an arbitrary target cannot be measured
natively. The native executor consumes that packet stream with a native WebGPU implementation,
presents to an offscreen swapchain (headless) or a native window (`--window`), and keeps the
semantics of the browser worker (`web/workers/wgpu_worker.js`), so a native frame and a Chrome
frame of the same stream can be compared bit for bit.

The executor mirrors `web/`; it never changes it. When the worker's behaviour changes, the executor
follows.

## Library: Dawn

| | Dawn | wgpu-native |
| :--- | :--- | :--- |
| Licence | BSD-3-Clause | MIT / Apache-2.0 |
| Windows build | MSVC, D3D12 (+ Vulkan) | MSVC/GNU, D3D12/Vulkan |
| Prebuilt binaries | official GitHub releases (`google/dawn`, Windows/Ubuntu/macOS tarballs with a CMake package) | official GitHub releases (zip per target) |
| Shader compiler | Tint (WGSL -> HLSL -> DXC) | naga |
| Same implementation as the worker's? | **yes**: Chrome's WebGPU is Dawn | no |

Both licences fit the repository and both ship prebuilt Windows binaries. Dawn wins because it is
the WebGPU implementation inside Chrome, which is where the worker runs: the same WGSL front end
(Tint), the same validation rules (an invalid packet fails the same way natively and in the
browser), and the same D3D12 backend with DXC. Bit-identical results are a realistic goal with
Dawn; with wgpu-native the WGSL translation (naga), robustness transforms and validation differ, so
every mismatch would first have to be blamed on the library.

Costs: the Windows release archive is 119 MB and its static `webgpu_dawn.lib` 620 MB (linked
executables are much smaller); it is a Release build with `/MD`, so the executor builds in Release
or RelWithDebInfo only. Dawn's D3D12 backend loads `dxcompiler.dll` / `dxil.dll` from the
executable's directory, so the build also fetches the official DXC release (NCSA/MIT; `dxil.dll`
under Microsoft's redistributable licence, the same files Chrome ships) and copies them next to the
executables. Nothing is vendored: both archives are fetched by CMake `FetchContent` at configure
time and pinned by SHA-256.

Pinned versions: Dawn `v20261109.214834` (commit `7435fd87`), DXC `v1.9.2609`. Moving the Dawn pin
should track the Dawn revision of the Chrome stable used by `tools/replay_capture.py` (Chrome's
`DEPS` names it), so the two sides run the same code.

## Build

```
cmake -S runtime -B build-native-exec -DPLAYGTA6_NATIVE_EXECUTOR=ON
cmake --build build-native-exec --config Release --target pg6_native_exec test_native_executor
ctest --test-dir build-native-exec -C Release -R native_exec
```

`PLAYGTA6_NATIVE_EXECUTOR` is `OFF` by default; CI and every other build are unchanged.
`test_native_exec_model` (the GPU-independent part) is built on every host build and runs in CI.

`pg6_native_exec --trace <file.pg6trace> [--window] [--size WxH] [--frames N] [--digest-at 0,10]
[--report stats.json]` replays a capture. It cuts the trace into frames exactly like
`tools/replay/replay.html` (`splitFrames`, thinned uploads dropped), uses the replay page's
3840x2160 canvas size by default, and with `--digest-at` prints the page's own `digest frame N:` and
`swapchain frame N (...): non-black ..., distinct ...` lines, so its output diffs line by line
against `replay_capture.py`'s log.

## Layout

| File | Role |
| :--- | :--- |
| `runtime/gpu/native_exec/stream_model.h` | Packet walk + payload decoding with the worker's rules. No WebGPU types; unit-tested in CI. |
| `runtime/gpu/native_exec/replay_frames.h` | `splitFrames`, frame digest and sampled stats of `tools/replay/replay.html`. No WebGPU types. |
| `runtime/gpu/native_exec/native_executor.{h,cpp}` | Dawn device, swapchain (offscreen texture or surface), packet handlers, Present. |
| `runtime/gpu/native_exec/native_window.{h,cpp}` | Optional Win32 window (other hosts: headless only for now). |
| `runtime/gpu/native_exec/pg6_native_exec_main.cpp` | Trace replay tool. |
| `tests/test_native_exec_model.cpp` | Model tests (CI). |
| `tests/test_native_executor.cpp` | GPU tests, headless; exit 77 (CTest skip) without an adapter. |

Every handler names the worker function it mirrors.

## How it stays identical to the worker

1. **Same order of effects.** The worker applies uploads (`WriteBuffer`, later `WriteTexture`) at
   packet time and records render/compute commands into `_passCommands`, which `_flushPresent`
   encodes at `Present`. So every upload of a frame lands before any pass of that frame, wherever it
   sits in the stream. The executor does the same: uploads go to the queue at packet time, pass
   commands are recorded and encoded at `Present`; lookups by id (buffers, textures, pipelines)
   happen at encode time, as in the worker.
2. **Same per-packet rules**, ported literally and unit-tested on the model: truncated packets stop
   the batch; short payloads are op errors; `BeginFrame` resets the clear colour to
   `DEFAULT_CLEAR_COLOR` (0.05, 0.07, 0.11, 1) and the pass list; the v1 clearing swapchain pass runs
   only when no render pass ran; `BeginRenderPass` normalizes load/store ops (`undefined` -> load /
   store), sanitizes clear values (`finiteOr(v, 0) || 0`, alpha falls back to 1), drops the whole
   packet when an attachment texture is unknown and drops colour attachments whose size differs from
   the first sized one; `WriteBuffer` takes the staging path (zero-padded to 4 bytes) when the offset
   is aligned and the padded copy fits, otherwise `queue.writeBuffer` in 4 MiB chunks, and leaves the
   rejection of unaligned writes to WebGPU's own validation, which is Dawn's in both places.
3. **Same canvas.** The swapchain is `bgra8unorm` (Chrome's preferred canvas format on Windows),
   opaque, 3840x2160 in replays. A canvas texture starts each frame as transparent black, so a
   `load` of the swapchain before anything was drawn in the frame reads zeros (implemented as a clear
   to 0). A stream that declares a canvas format on a swapchain attachment (`rgba8unorm`,
   `bgra8unorm`, `rgba16float`) switches the swapchain to it (#709).
4. **Same implementation.** Dawn at Chrome's revision, D3D12 + DXC, default adapter; device toggles
   are to follow Chrome's (milestone 1 pins them explicitly).
5. **No invented results.** A packet the executor does not implement is skipped by `payload_bytes`
   and counted per opcode (`stats.unhandled`, `--report`); it is never approximated. Every number
   the tests check is read back from the GPU.

Known divergences, to be closed or documented as they come up:

- Device creation is bare (the worker negotiates optional features and raised limits in
  `requestNegotiatedDevice`): milestone 1.
- Non-canvas stream swapchain formats (sRGB, others) use the worker's intermediate texture + blit
  (`WgpuSwapchain`); the executor counts them (`swapchain_format_unsupported`) and keeps the canvas
  format: milestone 2.
- `CreateBuffer` with `byte_size` above 2^53 throws a `TypeError` in the worker (WebIDL
  `[EnforceRange]`); natively it is a WebGPU validation error.
- Worker-only machinery with no GPU-visible effect on a healthy run is not mirrored: device-loss
  recovery and resource replay (W100/W104), early read-back flushes, first-visible-frame probes,
  telemetry. A lost device stops the executor (later batches are counted as dropped).

## Packet coverage plan

| Opcodes | Worker source | Milestone |
| :--- | :--- | :--- |
| `0` Nop, `1` BeginFrame, `2` ClearColor, `7` Present | `wgpu_worker.js` `_dispatchPacket`, `_flushPresent` | **M0 (done)** |
| `3` CreateBuffer, `4` WriteBuffer, `11` DestroyResource (buffers) | `_handleCreateBuffer`, `_handleWriteBuffer`, `_handleDestroyResource` | **M0 (done)** |
| `26` BeginRenderPass, `27` EndRenderPass (swapchain colour attachments) | `render_state_packets.js` `_beginRenderPass`, `buildRenderPassDescriptor` | **M0 (done)** |
| `12` CreateTexture, `35` CreateTexture2, `13` WriteTexture, texture attachments + depth-stencil in `26` | `_handleCreateTexture`, `render_state_packets.js` `_createTexture2`, `texture_packets.js` | M1 |
| `14`/`34` samplers, `5` CreateShaderModule (inline WGSL and WGSL pack lookup) | `_handleCreateShaderModule`, `wgsl_pack_loader.js` | M1 |
| `22`/`23` layouts (packet and `layout-manifest.json`), `24`/`25`, `8`/`19` pipelines, `15`/`16`/`32`/`33` bind groups | `render_state_packets.js`, `reflection_loader.js` | M1 |
| `6`/`18` draws, `9`/`10`/`17` state, `20`/`21` compute, `28`-`31` dynamic state | `_flushPresent` pass commands | M1 |
| `36`-`38` indirect, `39`/`40` read-back into producer memory + completion counter | `indirect_draw.js`, `_startRecordedReadbacks` | M2 |
| Swapchain blit fallback, feature/limit negotiation, GPU pass timestamps (#1608) | `wgpu_swapchain.js`, `requestNegotiatedDevice`, `gpu_timer.js` | M2 |
| Live boot: `playgta6_boot --native-gpu [--window]`, present any target | `os::wgpu_submit` observer | M3 |

The order follows what a real capture needs to draw anything: M1 is the first milestone at which
title frames render natively.

## Threading

- **M0:** single-threaded. The executor, its device and its window belong to the thread that
  created them; `pg6_native_exec` feeds it one frame (a `splitFrames` frame) per `execute()` call
  and pumps window messages between frames. Window presentation is FIFO (vsync-paced).
- **M3 (live boot):** `os::wgpu_submit` hands complete packet batches to one observer, on any guest
  thread, with the d3d12 device's submit mutex held. Today the frame capture owns that single
  observer slot, so the boot hook first turns it into a small fan-out (capture and executor both
  attached). The observer copies each batch into a bounded queue (single producer under the submit
  mutex, single consumer) and returns; an executor thread owns the Dawn device and the window,
  drains the queue in order and pumps messages. When the queue is full the observer blocks: the
  GPU side paces the title the way the browser's GPU ring does (`gpu/gpu_ring.h`), and present
  pacing (`gpu/present_pacing.h`) keeps its meaning. Read-backs (`39`/`40`) are completed by the
  executor thread writing the bytes to `dst_addr` and incrementing the `u32` at `done_addr`
  (`std::atomic_ref`), the native equivalent of the worker's `Atomics.add`; the producer already
  waits on that counter (`gpu/d3d12_readback.cpp`), and how `dst_addr` maps to a host pointer in the
  native harness is settled in that milestone.

## Test strategy

1. **Model tests (CI, every host build):** `tests/test_native_exec_model.cpp` pins the packet walk,
   the payload rules above, `splitFrames` and the replay digest (its reference value is computed by
   the replay page's own code).
2. **GPU unit test (local, `-DPLAYGTA6_NATIVE_EXECUTOR=ON`):** `tests/test_native_executor.cpp`
   creates a headless device and checks, by reading the swapchain and buffers back from the GPU:
   the default and explicit clear colours, `BeginFrame` resetting the colour, explicit clear / load
   passes, the fresh-canvas rule, an unknown attachment dropping the pass, the padded `WriteBuffer`
   path byte for byte, the rejection of an unaligned write (a device error, contents unchanged),
   buffer re-creation and destruction, and the per-opcode accounting of truncated and unhandled
   packets. Without an adapter it exits 77 and CTest reports it as skipped. CI has no GPU and keeps
   the option off.
3. **Differential replay against Chrome:** replay the same `.pg6trace` with
   `tools/replay_capture.py --thumbat F...` and `pg6_native_exec --digest-at F...`; both cut the trace
   the same way and print the same `digest frame` lines (a full-resolution hash of the swapchain), so
   equal lines mean identical frames. Milestone 0 check (NVIDIA GPU, D3D12, Chrome stable): a
   synthetic 4-frame trace (default clear, `ClearColor`, explicit clear pass, load of a fresh canvas)
   gives identical digests on all four frames. From M1 on the same comparison runs on sanitized
   real captures (local only; captures never enter the repository); a mismatch is bisected by frame,
   then by pass, with `--report` naming the opcodes the executor skipped.
