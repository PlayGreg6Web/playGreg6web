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

`pg6_native_exec --trace <file.pg6trace> [--pack <file.wgsl.pack>] [--window] [--size WxH] [--frames N]
[--digest-at 0,10] [--report stats.json]` replays a capture (`--pack` is the replay page's WGSL pack for
hash-only shader payloads; only its entry table is read up front, WGSL is read on demand, so a 1 GiB pack
costs a few MiB). It cuts the trace into frames exactly like
`tools/replay/replay.html` (`splitFrames`, thinned uploads dropped), uses the replay page's
3840x2160 canvas size by default, and with `--digest-at` prints the page's own `digest frame N:` and
`swapchain frame N (...): non-black ..., distinct ...` lines, so its output diffs line by line
against `replay_capture.py`'s log.

## Layout

| File | Role |
| :--- | :--- |
| `runtime/gpu/native_exec/stream_model.h` | Packet walk + payload decoding with the worker's rules. No WebGPU types; unit-tested in CI. |
| `runtime/gpu/native_exec/m1_model.h` | M1 rules that decide what reaches the GPU: WGSL pack index, hash-only payloads, entry point resolution, texture format tables, WriteTexture layout planning, viewport / scissor clamps. No WebGPU types; unit-tested in CI. |
| `runtime/gpu/native_exec/replay_frames.h` | `splitFrames`, frame digest and sampled stats of `tools/replay/replay.html`. No WebGPU types. |
| `runtime/gpu/native_exec/native_executor.{h,cpp}` | Dawn device, swapchain (offscreen texture or surface), packet handlers, Present. |
| `runtime/gpu/native_exec/native_window.{h,cpp}` | Optional Win32 window (other hosts: headless only for now). |
| `runtime/gpu/native_exec/live_pump.h` | Bounded queue + executor thread for live play (`--window`); executor-agnostic, unit-tested. |
| `runtime/gpu/native_exec/pg6_native_exec_main.cpp` | Trace replay tool. |
| `tests/test_native_exec_model.cpp` | Model tests (CI). |
| `tests/test_live_pump.cpp` | Live pump tests with a fake executor (CI). |
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

- Device creation mirrors the worker's `requestNegotiatedDevice`: the six optional features it requests
  (when the adapter has them) and the nine optional limits raised to `min(adapter, cap)` when that is
  above the WebGPU default (`kOptionalLimitSpecs`, `limits_raised` in the stats), with the same fallback
  order (features + limits, features only, bare), plus `timestamp-query` for the GPU pass timer (with
  Dawn's timestamp quantization toggle off: Chrome rounds timestamps to 100 us, the native table gets
  exact values).
- A pipeline with `pipeline_layout_id == 0` ("auto") or whose explicit layout does not match the WGSL
  `@group(0)` bindings gets Dawn's `layout: auto` without the worker's bind-group remap
  (`_getOrCreateRemappedBindGroup`); `auto_layout_pipelines` counts the first case. Manifest layouts
  (`layout-manifest.json`, flag `ManifestReference`) are unavailable, as in the replay page
  (`manifest_layouts_unavailable`). The remap, the manifest layouts and the reflection loader stay open.
- v1 pipelines (`8`/`19`) and bind groups (`15`/`16`) take the worker's path for "no reflection registry
  loaded" (the replay page's and the producer's situation): layout `auto`, entry points `vs_main` /
  `fs_main` / `cs_main`, one `float32xN` attribute at location 0 for a non-zero stride, the canvas format
  as the only target, bind group layouts from `pipeline.getBindGroupLayout(group)`, a missing
  `resource_id == 0` sampler replaced by the default linear sampler. Counted as `reflection_unavailable`.
  A stream that depends on a loaded reflection registry (explicit S6 layouts, reflected vertex
  attributes) or on the G34 fallback shader profiles (the worker's shader router) is not reproduced.
- Read-backs (`39`/`40`) write to the memory given by `attach_guest_memory`: a base pointer + size (the
  browser's wasm memory offsets) or absolute host addresses (what the native `ReadbackGate` hands
  out). Nothing attached: nothing is written and the counter does not move (the worker's `!memory`).
  The live boot (`--window`) attaches absolute host addresses (the producer's `CommandList::readback_target`
  already hands out host pointers), so read-backs land in the running guest. A frame that cannot be presented completes its queued read-backs as failed (the worker
  would leave the counter standing, and the producer's fence hold with it).
- GPU pass timestamps time a pass as a whole (WebGPU has no timestamp writes inside a pass). A pass is
  keyed by the pipeline its first `SetPipeline` / `SetComputePipeline` selects: the fragment shader's
  hash for a render pipeline (else the vertex shader's), the compute shader's hash for a compute
  pipeline, `0` for a pass without a pipeline (the v1 clearing pass) or an inline-WGSL shader. A pass
  that switches pipelines is attributed entirely to the first one; nothing is split or estimated.
- WGSL that the pack does not hold and the fallback router would replace: the replay page has no router,
  so the hash text is compiled and rejected on both sides (`shader_pack_misses`); the router itself is not
  mirrored.
- Uploads: `WriteTexture` takes the worker's effective layout (row pitch inference, block clamping, mip
  bounds clamp) and a direct `queue.WriteTexture`; the worker's staging pool and 8 MiB strip chunking only
  change how the same texels reach the texture and are not mirrored.
- Non-canvas stream swapchain formats (sRGB, others) use the worker's intermediate texture + blit
  (`WgpuSwapchain`); the executor counts them (`swapchain_format_unsupported`) and keeps the canvas
  format: not mirrored yet.
- `CreateBuffer` with `byte_size` above 2^53 throws a `TypeError` in the worker (WebIDL
  `[EnforceRange]`); natively it is a WebGPU validation error.
- Worker-only machinery with no GPU-visible effect on a healthy run is not mirrored: device-loss
  recovery and resource replay (W100/W104), first-visible-frame probes, telemetry. (The early read-back
  flush is mirrored: it is what lets a title that waits on a read-back fence make progress.) A lost device stops the executor (later batches are counted as dropped).

## Packet coverage plan

| Opcodes | Worker source | Milestone |
| :--- | :--- | :--- |
| `0` Nop, `1` BeginFrame, `2` ClearColor, `7` Present | `wgpu_worker.js` `_dispatchPacket`, `_flushPresent` | **M0 (done)** |
| `3` CreateBuffer, `4` WriteBuffer, `11` DestroyResource (buffers) | `_handleCreateBuffer`, `_handleWriteBuffer`, `_handleDestroyResource` | **M0 (done)** |
| `26` BeginRenderPass, `27` EndRenderPass (swapchain colour attachments) | `render_state_packets.js` `_beginRenderPass`, `buildRenderPassDescriptor` | **M0 (done)** |
| `12` CreateTexture, `35` CreateTexture2, `13` WriteTexture (BC included), texture attachments + depth-stencil in `26` | `texture_packets.js` `_create` / `handleWriteTexture` | **M1 (done)** |
| `14`/`34` samplers, `5` CreateShaderModule (inline WGSL and WGSL pack lookup) | `_handleCreateShaderModule`, `wgsl_pack_loader.js` | **M1 (done)** |
| `22`/`23` layouts, `24`/`25` pipelines, `32`/`33` bind groups | `render_state_packets.js` | **M1 (done)** |
| `6`/`18` draws, `9`/`10`/`17` state, `20`/`21` compute, `28`-`31` dynamic state, `36`-`38` indirect | `_flushPresent` pass commands | **M1 (done)** |
| `8`/`19` v1 pipelines, `15`/`16` v1 bind groups (the no-reflection path, see the divergences) | `_handleCreateRenderPipeline`, `_handleCreateComputePipeline`, `_handleCreateBindGroup` | **M2 (done)** |
| `39`/`40` read-back into producer memory + completion counter, early read-back flush | `_recordReadbackCopy`, `_startRecordedReadbacks`, `_flushReadbacksNow` | **M2 (done)** |
| GPU pass timestamps feeding `runtime/gpu/gpu_time` (#1608), `gpu_time` in the boot report | `gpu_timer.js` | **M2 (done)** |
| `layout-manifest.json` layouts, the reflection loader, the layout:auto bind-group remap | `reflection_loader.js`, `_getOrCreateRemappedBindGroup` | M2b (open) |
| `requestNegotiatedDevice` limit negotiation | `requestNegotiatedDevice` | **M2b (done)** |
| Swapchain blit fallback | `wgpu_swapchain.js` | M2b (open) |
| Live boot: `playgta6_boot --native-gpu [--window]`, present any target | `os::wgpu_submit` observer | M3 |

The order follows what a real capture needs to draw anything: M1 is the first milestone at which
title frames render natively. The two real captures below contain only M1 opcodes (plus `3`/`4`/`11`/`26`/`27`):
every packet is handled, none skipped.

## Live play: `playgta6_boot --window`

Built only with `-DPLAYGTA6_NATIVE_EXECUTOR=ON` (`playgta6_boot` links `playgta6_native_exec` then;
`tools/build_native_boot.py --native-executor` does the same for the generated full harness; without
the option `--window` exits with a usage error).

```
playgta6_boot --pgim ... --window [--window-size 1920x1080] [--host-input] [--gpu-capture f.pg6trace]
```

- **Hand-off** (`runtime/gpu/native_exec/live_pump.h`, `LivePump<Executor>`): the harness installs one
  `os::wgpu_submit` observer that feeds the capture (when `--gpu-capture` is given, which then does
  not attach itself) and copies the batch into the pump. The pump thread creates the executor (so
  the Dawn device and the window belong to it), executes queued batches in order and polls (window
  messages, WebGPU callbacks) after every batch and every `idle_poll_ms` (8) while idle.
- **Backpressure:** the queue is bounded (64 batches / 256 MiB; a single larger batch is admitted
  into an empty queue). When the executor falls behind, `submit()` blocks the submitting guest
  thread until room is made, like the browser worker's ring; nothing is dropped. Blocked submits
  and their total wait are counted (`blocked_submits`, `blocked_ms`) and appear with `submitted`,
  `executed`, `presents`, `max_queue_depth` in the boot report under `native_executor`.
- **Close = clean shutdown:** closing the window makes the pump discard the queue, release any
  blocked submitter and call `guest::request_boot_stop()`; the boot stops at its next dispatch like
  a bound (`stop_requested`), the pump is drained and joined, and the normal report and summary are
  written. After close `submit()` drops and returns at once (`dropped_after_close`), so the guest
  can never deadlock behind a dead consumer.
- **Focus and `--host-input`:** the pump forwards window focus changes. The XInput pad is always
  read; the keyboard (read globally with `GetAsyncKeyState`) only counts while the window has focus
  (`HostInputSource::set_keyboard_enabled`).
- **Tests:** `tests/test_live_pump.cpp` (CI, every host build) drives the pump with a fake executor:
  order, executor created/used/destroyed on the pump thread, backpressure (blocks at the bound,
  counted, nothing lost), byte bound and oversized batch, close unblocking a blocked submitter,
  focus forwarding, creation failure, stop draining.

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
  (`std::atomic_ref`, M2: `attach_guest_memory`), the native equivalent of the worker's `Atomics.add`;
  the producer already waits on that counter (`gpu/d3d12_readback.cpp`), and how `dst_addr` maps to a
  host pointer in the native harness is settled in that milestone (M2 supports a base pointer + size
  or absolute host addresses). The completion needs the executor's `poll()` (WebGPU map callbacks): the
  live pump already polls after every batch.

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
   packets. Milestone 2 adds, all read back from the GPU into a host buffer standing in for the guest
   memory: a compute result and a texture region (rows repacked to a 16-byte guest pitch, the gap
   untouched), the early flush (nothing presented, the data and counter arrive), failed copies (counter
   advances, nothing written), no memory attached, v1 pipelines / bind groups (a textured draw, a compute
   dispatch), and the GPU timer (a pack-keyed compute pass lands in a private `GpuTimeTable` with its
   hash, a non-zero duration and the right pass counts). Without an adapter it exits 77 and CTest
   reports it as skipped. CI has no GPU and keeps
   the option off.
3. **Differential replay against Chrome:** replay the same `.pg6trace` with
   `tools/replay_capture.py --thumbat F...` and `pg6_native_exec --digest-at F...`; both cut the trace
   the same way and print the same `digest frame` lines (a full-resolution hash of the swapchain), so
   equal lines mean identical frames. Milestone 0 check (NVIDIA GPU, D3D12, Chrome stable): a
   synthetic 4-frame trace (default clear, `ClearColor`, explicit clear pass, load of a fresh canvas)
   gives identical digests on all four frames. From M1 on the same comparison runs on sanitized
   real captures (local only; captures never enter the repository); a mismatch is bisected by frame,
   then by pass, with `--report` naming the opcodes the executor skipped.

   Milestone 1 check (NVIDIA GPU, D3D12, Chrome stable, the replay page's 3840x2160 `rgba8unorm` canvas):
   two sanitized boot captures of 2500 presented frames each (a 671 MB and a 607 MB trace, 1.04 M and
   0.56 M packets, 3085 shader modules from a 1.2 GB pack, 649 textures incl. BC, 98 k / 81 k draws,
   5000 dispatches). Native digests equal Chrome's on every compared frame (frames 0, 63, 500, 1000, 1500,
   2000, 2499 of the first; 0, 700, 1600, 2499 of the second), with no device error and no packet error;
   the executor handled 100% of the packets (`unhandled` empty). Replaying the first trace natively takes
   about 20 s. The GPU unit test covers a textured draw, a compute dispatch (direct and indirect), a
   pack lookup and a BC7 upload with read-back.
